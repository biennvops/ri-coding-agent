use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ri_core::{ResolvedModel, ThinkingLevel};
use unicode_width::UnicodeWidthStr;

use crate::input::Action;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ThinkingPickerOutcome {
    Pending,
    Selected(ThinkingLevel),
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ThinkingPickerState {
    levels: Vec<ThinkingLevel>,
    selected: usize,
}

impl ThinkingPickerState {
    pub(crate) fn new(model: &ResolvedModel, current: Option<ThinkingLevel>) -> Self {
        Self::from_levels(model.supported_thinking_levels(), current)
    }

    fn from_levels(levels: Vec<ThinkingLevel>, current: Option<ThinkingLevel>) -> Self {
        let selected = current
            .and_then(|current| levels.iter().position(|level| *level == current))
            .unwrap_or(0);
        Self { levels, selected }
    }

    pub(crate) fn len(&self) -> usize {
        self.levels.len()
    }

    pub(crate) fn longest_level_width(&self) -> usize {
        self.levels
            .iter()
            .map(|level| UnicodeWidthStr::width(level.as_str()))
            .max()
            .unwrap_or_default()
    }

    pub(crate) fn handle_action(&mut self, action: Action) -> ThinkingPickerOutcome {
        match action {
            Action::Up => {
                self.move_up();
                ThinkingPickerOutcome::Pending
            }
            Action::Down => {
                self.move_down();
                ThinkingPickerOutcome::Pending
            }
            Action::Submit => ThinkingPickerOutcome::Selected(self.selected_level()),
            Action::Escape => ThinkingPickerOutcome::Cancelled,
            _ => ThinkingPickerOutcome::Pending,
        }
    }

    pub(crate) fn rows(&self, visible_rows: usize) -> Vec<Line<'static>> {
        let visible_rows = visible_rows.min(self.levels.len());
        let start = self
            .selected
            .saturating_sub(visible_rows.saturating_sub(1))
            .min(self.levels.len().saturating_sub(visible_rows));
        self.levels
            .iter()
            .enumerate()
            .skip(start)
            .take(visible_rows)
            .map(|(index, level)| {
                let marker = if index == self.selected { ">" } else { " " };
                let marker_style = if index == self.selected {
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

    fn selected_level(&self) -> ThinkingLevel {
        self.levels
            .get(self.selected)
            .copied()
            .unwrap_or(ThinkingLevel::Off)
    }

    fn move_up(&mut self) {
        if self.levels.is_empty() {
            return;
        }
        if self.selected == 0 {
            self.selected = self.levels.len() - 1;
        } else {
            self.selected -= 1;
        }
    }

    fn move_down(&mut self) {
        if self.levels.is_empty() {
            return;
        }
        if self.selected + 1 == self.levels.len() {
            self.selected = 0;
        } else {
            self.selected += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ri_core::{ApiKind, Compatibility, CostMetadata, ModelRef};

    use super::*;

    fn model(levels: &[ThinkingLevel]) -> ResolvedModel {
        ResolvedModel {
            model_ref: ModelRef::new("test", "model"),
            name: "Test model".to_owned(),
            base_url: "https://example.invalid".to_owned(),
            api: ApiKind::OpenAiResponses,
            api_key: None,
            headers: BTreeMap::new(),
            auth_header: true,
            compatibility: Compatibility::default(),
            reasoning: true,
            thinking_level_map: ThinkingLevel::ALL
                .into_iter()
                .filter(|level| *level != ThinkingLevel::Off)
                .map(|level| {
                    let effort = levels.contains(&level).then(|| level.to_string());
                    (level, effort)
                })
                .collect(),
            input: vec!["text".to_owned()],
            context_window: Some(100_000),
            max_tokens: Some(4_096),
            cost: CostMetadata::default(),
            sampling_params: BTreeMap::new(),
        }
    }

    #[test]
    fn current_supported_level_is_selected_initially() {
        let picker = ThinkingPickerState::new(
            &model(&[
                ThinkingLevel::Low,
                ThinkingLevel::High,
                ThinkingLevel::XHigh,
            ]),
            Some(ThinkingLevel::High),
        );

        let rendered = picker
            .rows(picker.len())
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>();
        assert_eq!(rendered, ["  off", "  low", "> high", "  xhigh"]);
    }

    #[test]
    fn navigation_wraps_and_enter_returns_the_selected_level() {
        let mut picker = ThinkingPickerState::from_levels(
            vec![ThinkingLevel::Off, ThinkingLevel::Low, ThinkingLevel::High],
            Some(ThinkingLevel::Off),
        );

        assert_eq!(
            picker.handle_action(Action::Up),
            ThinkingPickerOutcome::Pending
        );
        assert_eq!(
            picker.handle_action(Action::Submit),
            ThinkingPickerOutcome::Selected(ThinkingLevel::High)
        );
        assert_eq!(
            picker.handle_action(Action::Down),
            ThinkingPickerOutcome::Pending
        );
        assert_eq!(
            picker.handle_action(Action::Submit),
            ThinkingPickerOutcome::Selected(ThinkingLevel::Off)
        );
    }

    #[test]
    fn escape_cancels_and_unsupported_levels_are_not_listed() {
        let mut picker = ThinkingPickerState::new(
            &model(&[ThinkingLevel::Low, ThinkingLevel::High]),
            Some(ThinkingLevel::Max),
        );

        assert_eq!(
            picker
                .rows(picker.len())
                .iter()
                .map(Line::to_string)
                .collect::<Vec<_>>(),
            ["> off", "  low", "  high"]
        );
        assert_eq!(
            picker.handle_action(Action::Escape),
            ThinkingPickerOutcome::Cancelled
        );
    }
}
