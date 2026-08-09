use anyhow::{bail, Result};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ri_core::{OpenedSession, SessionRepository, SessionSummary};

use crate::picker::{draw_picker, read_action, PickerAction, PickerState};
use crate::terminal::TerminalGuard;

pub fn pick(repository: &SessionRepository) -> Result<Option<OpenedSession>> {
    let mut terminal = TerminalGuard::new()?;
    let result = pick_in_terminal(&mut terminal, repository);
    drop(terminal);
    result
}

pub fn pick_in_terminal(
    terminal: &mut TerminalGuard,
    repository: &SessionRepository,
) -> Result<Option<OpenedSession>> {
    let Some(path) = pick_path_in_terminal(terminal, repository)? else {
        return Ok(None);
    };
    let opened = repository
        .open_path(path)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    Ok(Some(opened))
}

pub fn pick_path_in_terminal(
    terminal: &mut TerminalGuard,
    repository: &SessionRepository,
) -> Result<Option<std::path::PathBuf>> {
    let summaries = repository
        .list()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    if summaries.is_empty() {
        bail!(
            "no saved sessions found for {}",
            repository.workspace_root().display()
        );
    }

    let mut picker = PickerState::new(summaries.len(), 0);
    loop {
        let rows = rows(&summaries, picker.selected());
        draw_picker(terminal, "Resume session", &rows, &mut picker)?;
        match read_action()? {
            PickerAction::Up => picker.move_up(false),
            PickerAction::Down => picker.move_down(false),
            PickerAction::Confirm => return Ok(Some(summaries[picker.selected()].path.clone())),
            PickerAction::Cancel => return Ok(None),
        }
    }
}

fn rows(summaries: &[SessionSummary], selected: usize) -> Vec<Line<'static>> {
    summaries
        .iter()
        .enumerate()
        .map(|(index, summary)| {
            let marker = if index == selected { ">" } else { " " };
            let marker_style = if index == selected {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            let title = summary
                .name
                .as_deref()
                .or(summary.first_user_preview.as_deref())
                .unwrap_or("unnamed session");
            let title = truncate(title, 30);
            Line::from(vec![
                Span::styled(format!("{marker} "), marker_style),
                Span::raw(format!(
                    "{:<20} {:<30} {} messages",
                    summary.updated_at, title, summary.message_count
                )),
            ])
        })
        .collect()
}

fn truncate(text: &str, limit: usize) -> String {
    let mut result = text.chars().take(limit.max(1)).collect::<String>();
    if text.chars().count() > limit.max(1) {
        result.push('…');
    }
    result
}
