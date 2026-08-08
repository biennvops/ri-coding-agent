use anyhow::{bail, Result};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ri_core::{OpenedSession, SessionRepository, SessionSummary};

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
    let mut selected = 0usize;
    loop {
        draw(terminal, &summaries, selected)?;
        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                selected = selected.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                selected = (selected + 1).min(summaries.len().saturating_sub(1));
            }
            KeyCode::Enter => return Ok(Some(summaries[selected].path.clone())),
            KeyCode::Esc => return Ok(None),
            _ => {}
        }
    }
}

fn draw(terminal: &mut TerminalGuard, summaries: &[SessionSummary], selected: usize) -> Result<()> {
    terminal.terminal_mut().draw(|frame| {
        let area = frame.area();
        let block = Block::default()
            .title(" Resume session ")
            .borders(Borders::ALL);
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let rows = Layout::vertical(
            summaries
                .iter()
                .map(|_| Constraint::Length(1))
                .collect::<Vec<_>>(),
        )
        .split(inner);
        for (index, summary) in summaries.iter().enumerate() {
            let marker = if index == selected { ">" } else { " " };
            let title = summary
                .name
                .as_deref()
                .or(summary.first_user_preview.as_deref())
                .unwrap_or("unnamed session");
            let title = truncate(title, inner.width.saturating_sub(32) as usize);
            let time = &summary.updated_at;
            let line = Line::from(vec![
                Span::styled(
                    format!("{marker} "),
                    if index == selected {
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default()
                    },
                ),
                Span::raw(format!("{time:<20} {title:<30} ")),
                Span::styled(
                    format!("{} messages", summary.message_count),
                    Style::default().fg(Color::DarkGray),
                ),
            ]);
            if let Some(row) = rows.get(index) {
                frame.render_widget(Paragraph::new(line), *row);
            }
        }
    })?;
    Ok(())
}

fn truncate(text: &str, limit: usize) -> String {
    let mut result = text.chars().take(limit.max(1)).collect::<String>();
    if text.chars().count() > limit.max(1) {
        result.push('…');
    }
    result
}
