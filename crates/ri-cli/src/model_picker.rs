use anyhow::Result;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ri_core::{ModelCatalog, ModelRef, ResolvedModel};

use crate::picker::{draw_picker, read_action, PickerAction, PickerState};
use crate::terminal::TerminalGuard;

pub(crate) fn pick_model_in_terminal(
    terminal: &mut TerminalGuard,
    catalog: &ModelCatalog,
    current: Option<&ModelRef>,
) -> Result<Option<ResolvedModel>> {
    if catalog.models().is_empty() {
        return Ok(None);
    }
    let selected = current
        .and_then(|current| {
            catalog
                .models()
                .iter()
                .position(|model| model.model_ref == *current)
        })
        .unwrap_or(0);
    let mut picker = PickerState::new(catalog.models().len(), selected);

    loop {
        let rows = rows(catalog, picker.selected());
        draw_picker(terminal, "Select model", &rows, &mut picker)?;
        match read_action()? {
            PickerAction::Up => picker.move_up(true),
            PickerAction::Down => picker.move_down(true),
            PickerAction::Confirm => return Ok(Some(catalog.models()[picker.selected()].clone())),
            PickerAction::Cancel => return Ok(None),
        }
    }
}

fn rows(catalog: &ModelCatalog, selected: usize) -> Vec<Line<'static>> {
    catalog
        .models()
        .iter()
        .enumerate()
        .map(|(index, model)| {
            let marker = if index == selected { ">" } else { " " };
            let marker_style = if index == selected {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            let context = model
                .context_window
                .map(format_context_window)
                .unwrap_or_else(|| "?".to_owned());
            let reasoning = if model.reasoning { " · reasoning" } else { "" };
            Line::from(vec![
                Span::styled(format!("{marker} "), marker_style),
                Span::raw(format!(
                    "{:<32} {:<24} {}{}",
                    model.model_ref.display_name(),
                    model.name,
                    context,
                    reasoning
                )),
            ])
        })
        .collect()
}

fn format_context_window(tokens: u64) -> String {
    if tokens >= 1_000 {
        format!("{}k", tokens.div_ceil(1_000))
    } else {
        tokens.to_string()
    }
}
