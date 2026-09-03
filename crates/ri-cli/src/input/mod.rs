mod layout;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};

pub(crate) use layout::VisualLayout;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Submit,
    Newline,
    Complete,
    Escape,
    CtrlC,
    ToggleToolOutput,
    Insert(char),
    Backspace,
    Delete,
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    PageUp,
    PageDown,
    MouseScrollUp,
    MouseScrollDown,
}

pub fn action_for(key: KeyEvent) -> Option<Action> {
    let modifiers = key.modifiers;

    if modifiers.contains(KeyModifiers::CONTROL) {
        return match key.code {
            KeyCode::Char('c') => Some(Action::CtrlC),
            KeyCode::Char('o') => Some(Action::ToggleToolOutput),
            KeyCode::Char('u') => Some(Action::PageUp),
            KeyCode::Char('d') => Some(Action::PageDown),
            _ => None,
        };
    }

    match key.code {
        KeyCode::Enter if modifiers.contains(KeyModifiers::SHIFT) => Some(Action::Newline),
        KeyCode::Enter => Some(Action::Submit),
        KeyCode::Tab => Some(Action::Complete),
        KeyCode::Esc => Some(Action::Escape),
        KeyCode::Char(character) => Some(Action::Insert(character)),
        KeyCode::Backspace => Some(Action::Backspace),
        KeyCode::Delete => Some(Action::Delete),
        KeyCode::Left => Some(Action::Left),
        KeyCode::Right => Some(Action::Right),
        KeyCode::Up => Some(Action::Up),
        KeyCode::Down => Some(Action::Down),
        KeyCode::Home => Some(Action::Home),
        KeyCode::End => Some(Action::End),
        KeyCode::PageUp => Some(Action::PageUp),
        KeyCode::PageDown => Some(Action::PageDown),
        _ => None,
    }
}

pub fn action_for_mouse(mouse: MouseEvent) -> Option<Action> {
    match mouse.kind {
        MouseEventKind::ScrollUp => Some(Action::MouseScrollUp),
        MouseEventKind::ScrollDown => Some(Action::MouseScrollDown),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinguishes_newline_escape_and_exit_keys() {
        assert_eq!(
            action_for(KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT)),
            Some(Action::Newline)
        );
        assert_eq!(
            action_for(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            Some(Action::Submit)
        );
        assert_eq!(
            action_for(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
            Some(Action::Escape)
        );
        assert_eq!(
            action_for(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)),
            Some(Action::Complete)
        );
        assert_eq!(
            action_for(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            Some(Action::CtrlC)
        );
    }

    #[test]
    fn maps_control_and_mouse_scrollback_actions() {
        assert_eq!(
            action_for(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            Some(Action::CtrlC)
        );
        assert_eq!(
            action_for(KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL)),
            Some(Action::ToggleToolOutput)
        );
        assert_eq!(
            action_for(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL)),
            Some(Action::PageUp)
        );
        assert_eq!(
            action_for(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL)),
            Some(Action::PageDown)
        );
        assert_eq!(
            action_for_mouse(MouseEvent {
                kind: MouseEventKind::ScrollUp,
                column: 0,
                row: 0,
                modifiers: KeyModifiers::NONE,
            }),
            Some(Action::MouseScrollUp)
        );
        assert_eq!(
            action_for_mouse(MouseEvent {
                kind: MouseEventKind::ScrollDown,
                column: 0,
                row: 0,
                modifiers: KeyModifiers::NONE,
            }),
            Some(Action::MouseScrollDown)
        );
    }
}
