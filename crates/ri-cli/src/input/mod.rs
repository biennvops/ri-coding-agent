mod layout;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

pub(crate) use layout::VisualLayout;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Submit,
    Newline,
    Escape,
    CtrlC,
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
}

pub fn action_for(key: KeyEvent) -> Option<Action> {
    let modifiers = key.modifiers;

    if modifiers.contains(KeyModifiers::CONTROL) {
        return match key.code {
            KeyCode::Char('c') => Some(Action::CtrlC),
            _ => None,
        };
    }

    match key.code {
        KeyCode::Enter if modifiers.contains(KeyModifiers::SHIFT) => Some(Action::Newline),
        KeyCode::Enter => Some(Action::Submit),
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
            action_for(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            Some(Action::CtrlC)
        );
    }
}
