use anyhow::Result;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ri_core::{ResolvedModel, ThinkingLevel};

use crate::picker::{draw_picker, read_action, PickerAction, PickerState};
use crate::terminal::TerminalGuard;

pub(crate) fn pick_thinking_level_in_terminal(
    terminal: &mut TerminalGuard,
    model: &ResolvedModel,
    current: Option<ThinkingLevel>,
) -> Result<Option<ThinkingLevel>> {
    let levels = model.supported_thinking_levels();
    let selected = current
        .and_then(|current| levels.iter().position(|level| *level == current))
        .unwrap_or(0);
    let mut picker = PickerState::new(levels.len(), selected);

    loop {
        let rows = rows(&levels, picker.selected());
        draw_picker(terminal, "Thinking level", &rows, &mut picker)?;
        match read_action()? {
            PickerAction::Up => picker.move_up(true),
            PickerAction::Down => picker.move_down(true),
            PickerAction::Confirm => return Ok(Some(levels[picker.selected()])),
            PickerAction::Cancel => return Ok(None),
        }
    }
}

fn rows(levels: &[ThinkingLevel], selected: usize) -> Vec<Line<'static>> {
    levels
        .iter()
        .enumerate()
        .map(|(index, level)| {
            let marker = if index == selected { ">" } else { " " };
            let marker_style = if index == selected {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            Line::from(vec![
                Span::styled(format!("{marker} "), marker_style),
                Span::raw(level.to_string()),
            ])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_mark_only_the_selected_level() {
        let rows = rows(
            &[ThinkingLevel::Off, ThinkingLevel::Low, ThinkingLevel::High],
            1,
        );
        let rendered = rows.iter().map(|line| line.to_string()).collect::<Vec<_>>();

        assert_eq!(rendered, ["  off", "> low", "  high"]);
    }
}
