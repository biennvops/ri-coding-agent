use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use ratatui::layout::Rect;
use ratatui::text::Line;
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::terminal::TerminalGuard;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PickerAction {
    Up,
    Down,
    Confirm,
    Cancel,
    Redraw,
}

pub(crate) fn action_for(key: KeyEvent) -> Option<PickerAction> {
    if key.kind != KeyEventKind::Press {
        return None;
    }
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => Some(PickerAction::Up),
        KeyCode::Down | KeyCode::Char('j') => Some(PickerAction::Down),
        KeyCode::Enter => Some(PickerAction::Confirm),
        KeyCode::Esc => Some(PickerAction::Cancel),
        _ => None,
    }
}

fn action_for_event(event: Event) -> Option<PickerAction> {
    match event {
        Event::Key(key) => action_for(key),
        Event::Resize(..) => Some(PickerAction::Redraw),
        _ => None,
    }
}

pub(crate) fn read_action() -> Result<PickerAction> {
    loop {
        if let Some(action) = action_for_event(event::read()?) {
            return Ok(action);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PickerState {
    selected: usize,
    offset: usize,
    count: usize,
    visible_rows: usize,
}

impl PickerState {
    pub(crate) fn new(count: usize, selected: usize) -> Self {
        Self {
            selected: selected.min(count.saturating_sub(1)),
            offset: 0,
            count,
            visible_rows: usize::MAX,
        }
    }

    pub(crate) fn selected(&self) -> usize {
        self.selected
    }

    pub(crate) fn offset(&self) -> usize {
        self.offset
    }

    pub(crate) fn move_up(&mut self, wrap: bool) {
        if self.count == 0 {
            return;
        }
        if self.selected == 0 {
            if wrap {
                self.selected = self.count - 1;
            }
        } else {
            self.selected -= 1;
        }
        self.ensure_visible(self.visible_rows);
    }

    pub(crate) fn move_down(&mut self, wrap: bool) {
        if self.count == 0 {
            return;
        }
        if self.selected + 1 >= self.count {
            if wrap {
                self.selected = 0;
            }
        } else {
            self.selected += 1;
        }
        self.ensure_visible(self.visible_rows);
    }

    pub(crate) fn set_visible_rows(&mut self, visible_rows: usize) {
        self.visible_rows = visible_rows;
        self.ensure_visible(visible_rows);
    }

    fn ensure_visible(&mut self, visible_rows: usize) {
        if self.count == 0 || visible_rows == usize::MAX {
            if self.count == 0 {
                self.offset = 0;
            }
            return;
        }
        if self.selected < self.offset {
            self.offset = self.selected;
        } else if self.selected >= self.offset.saturating_add(visible_rows) {
            self.offset = self.selected + 1 - visible_rows.max(1);
        }
        self.offset = self
            .offset
            .min(self.count.saturating_sub(visible_rows.max(1)));
    }
}

pub(crate) fn draw_picker(
    terminal: &mut TerminalGuard,
    title: &str,
    rows: &[Line<'static>],
    state: &mut PickerState,
) -> Result<()> {
    terminal.terminal_mut().draw(|frame| {
        let area = frame.area();
        let block = Block::default()
            .title(format!(" {title} "))
            .borders(Borders::ALL);
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let visible_rows = inner.height as usize;
        state.set_visible_rows(visible_rows);
        for (index, row) in rows
            .iter()
            .enumerate()
            .skip(state.offset())
            .take(visible_rows)
        {
            let row_area = Rect::new(
                inner.x,
                inner.y + (index - state.offset()) as u16,
                inner.width,
                1,
            );
            frame.render_widget(Paragraph::new(row.clone()), row_area);
        }
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_events_request_a_redraw() {
        assert_eq!(
            action_for_event(Event::Resize(120, 40)),
            Some(PickerAction::Redraw)
        );
    }

    #[test]
    fn picker_selection_clamps_or_wraps() {
        let mut picker = PickerState::new(3, 0);
        picker.move_up(false);
        assert_eq!(picker.selected(), 0);
        picker.move_down(false);
        picker.move_down(false);
        picker.move_down(false);
        assert_eq!(picker.selected(), 2);

        picker.move_down(true);
        assert_eq!(picker.selected(), 0);
        picker.move_up(true);
        assert_eq!(picker.selected(), 2);
    }

    #[test]
    fn picker_initial_selection_is_clamped_and_empty_is_safe() {
        assert_eq!(PickerState::new(3, 99).selected(), 2);
        let mut empty = PickerState::new(0, 0);
        empty.move_up(true);
        empty.move_down(true);
        empty.set_visible_rows(2);
        assert_eq!(empty.selected(), 0);
        assert_eq!(empty.offset(), 0);
    }

    #[test]
    fn picker_viewport_follows_selection() {
        let mut picker = PickerState::new(10, 0);
        picker.set_visible_rows(3);
        for _ in 0..5 {
            picker.move_down(false);
        }
        assert_eq!(picker.selected(), 5);
        assert_eq!(picker.offset(), 3);
        picker.move_up(false);
        assert_eq!(picker.offset(), 3);
        picker.move_up(false);
        assert_eq!(picker.offset(), 3);
    }
}
