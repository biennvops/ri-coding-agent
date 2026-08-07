use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;
use ri_core::{AppState, MessageRole};
use unicode_width::UnicodeWidthChar;

use crate::terminal::TerminalGuard;

const TAB_WIDTH: usize = 4;

pub fn draw(
    terminal: &mut TerminalGuard,
    state: &AppState,
    scroll_from_bottom: usize,
) -> std::io::Result<()> {
    terminal.terminal_mut().draw(|frame| {
        render_frame(frame, state, scroll_from_bottom);
    })?;
    Ok(())
}

fn render_frame(frame: &mut Frame<'_>, state: &AppState, scroll_from_bottom: usize) {
    let area = frame.area();
    let editor_width = area.width.saturating_sub(2).max(1) as usize;
    let (editor_cursor_row, _) =
        visual_cursor_position(state.input(), state.cursor(), editor_width);
    let editor_rows = visual_row_count(state.input(), editor_width).max(editor_cursor_row + 1);
    let editor_height = (editor_rows.saturating_add(2).min(u16::MAX as usize) as u16)
        .min(area.height.saturating_sub(2))
        .max(3);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(editor_height),
            Constraint::Length(1),
        ])
        .split(area);

    let transcript_width = chunks[0].width.saturating_sub(2).max(1) as usize;
    let transcript = transcript_lines(state, transcript_width);
    let visible_lines = chunks[0].height.saturating_sub(2) as usize;
    let maximum_scroll = transcript.len().saturating_sub(visible_lines);
    let scroll = maximum_scroll.saturating_sub(scroll_from_bottom.min(maximum_scroll));
    let transcript = Paragraph::new(transcript)
        .block(Block::default().borders(Borders::ALL).title(" transcript "))
        .scroll((scroll.min(u16::MAX as usize) as u16, 0));
    frame.render_widget(transcript, chunks[0]);

    let editor_width = chunks[1].width.saturating_sub(2).max(1) as usize;
    let (editor_lines, cursor_row, cursor_column) =
        editor_lines_and_cursor(state.input(), state.cursor(), editor_width);
    let editor_visible_lines = chunks[1].height.saturating_sub(2) as usize;
    let editor_scroll = cursor_row.saturating_sub(editor_visible_lines.saturating_sub(1));
    let editor_title = if state.is_turn_active() {
        " input · Esc cancels "
    } else {
        " input · Enter submits · Shift+Enter newline "
    };
    let editor = Paragraph::new(editor_lines)
        .block(Block::default().borders(Borders::ALL).title(editor_title))
        .scroll((editor_scroll.min(u16::MAX as usize) as u16, 0));
    frame.render_widget(editor, chunks[1]);

    let footer = footer_text(state, chunks[2].width);
    frame.render_widget(Paragraph::new(footer), chunks[2]);

    if !state.is_turn_active() && chunks[1].height > 2 {
        let x = chunks[1]
            .x
            .saturating_add(1)
            .saturating_add(cursor_column as u16);
        let y = chunks[1]
            .y
            .saturating_add(1)
            .saturating_add(cursor_row.saturating_sub(editor_scroll) as u16);
        let max_x = chunks[1]
            .x
            .saturating_add(chunks[1].width.saturating_sub(2));
        let max_y = chunks[1]
            .y
            .saturating_add(chunks[1].height.saturating_sub(2));
        frame.set_cursor_position((x.min(max_x), y.min(max_y)));
    }
}

fn transcript_lines(state: &AppState, width: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();

    for message in state.messages() {
        let (label, style) = match message.role {
            MessageRole::User => ("you", Style::default().fg(Color::Yellow)),
            MessageRole::Assistant => ("assistant", Style::default().fg(Color::Green)),
        };
        lines.extend(wrapped_styled_lines(&format!("▶ {label}"), style, width));
        append_wrapped_content(&mut lines, &message.content, Style::default(), width);
        if let Some(thinking) = &message.thinking {
            lines.extend(wrapped_styled_lines(
                "  thinking:",
                Style::default().fg(Color::Cyan),
                width,
            ));
            append_wrapped_content(
                &mut lines,
                thinking,
                Style::default().fg(Color::DarkGray),
                width,
            );
        }
        lines.push(Line::default());
    }

    if let Some((content, thinking)) = state.streaming_assistant() {
        lines.extend(wrapped_styled_lines(
            "▶ assistant · streaming",
            Style::default().fg(Color::Green),
            width,
        ));
        append_wrapped_content(&mut lines, content, Style::default(), width);
        if !thinking.is_empty() {
            lines.extend(wrapped_styled_lines(
                "  thinking:",
                Style::default().fg(Color::Cyan),
                width,
            ));
            append_wrapped_content(
                &mut lines,
                thinking,
                Style::default().fg(Color::DarkGray),
                width,
            );
        }
    }

    if lines.is_empty() {
        lines.extend(wrapped_styled_lines(
            "Start by describing a task.",
            Style::default().fg(Color::DarkGray),
            width,
        ));
    }

    lines
}

fn append_wrapped_content(
    lines: &mut Vec<Line<'static>>,
    content: &str,
    style: Style,
    width: usize,
) {
    for line in content.split('\n') {
        lines.extend(wrapped_styled_lines(&format!("  {line}"), style, width));
    }
}

fn wrapped_styled_lines(text: &str, style: Style, width: usize) -> Vec<Line<'static>> {
    wrapped_segments(text, width)
        .into_iter()
        .map(|segment| Line::from(Span::styled(segment, style)))
        .collect()
}

fn editor_lines_and_cursor(
    text: &str,
    cursor: usize,
    width: usize,
) -> (Vec<Line<'static>>, usize, usize) {
    let (cursor_row, cursor_column) = visual_cursor_position(text, cursor, width);
    let mut lines: Vec<Line<'static>> = wrapped_segments_for_text(text, width)
        .into_iter()
        .map(Line::from)
        .collect();

    while lines.len() <= cursor_row {
        lines.push(Line::default());
    }

    (lines, cursor_row, cursor_column)
}

fn visual_row_count(text: &str, width: usize) -> usize {
    wrapped_segments_for_text(text, width).len()
}

fn wrapped_segments_for_text(text: &str, width: usize) -> Vec<String> {
    text.split('\n')
        .flat_map(|line| wrapped_segments(line, width))
        .collect()
}

fn wrapped_segments(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    if text.is_empty() {
        return vec![String::new()];
    }

    let mut lines = Vec::new();
    let mut current = String::new();
    let mut current_width = 0;

    for character in text.chars() {
        let character_width = display_width(character);
        if !current.is_empty() && current_width + character_width > width {
            lines.push(std::mem::take(&mut current));
            current_width = 0;
        }
        current.push(character);
        current_width += character_width;
    }

    lines.push(current);
    lines
}

fn visual_cursor_position(text: &str, cursor: usize, width: usize) -> (usize, usize) {
    let width = width.max(1);
    let cursor = cursor.min(text.len());
    let prefix = text.get(..cursor).unwrap_or(text);
    let mut row = 0;
    let mut column = 0;

    for character in prefix.chars() {
        if character == '\n' {
            row += 1;
            column = 0;
            continue;
        }

        let character_width = display_width(character);
        if column > 0 && column + character_width > width {
            row += 1;
            column = 0;
        }
        column += character_width;
    }

    if column >= width {
        if !matches!(
            text.get(cursor..).and_then(|rest| rest.chars().next()),
            Some('\n')
        ) {
            row += 1;
            column = 0;
        } else {
            column = width.saturating_sub(1);
        }
    }

    (row, column.min(width.saturating_sub(1)))
}

fn display_width(character: char) -> usize {
    if character == '\t' {
        TAB_WIDTH
    } else {
        UnicodeWidthChar::width(character).unwrap_or(0)
    }
}

fn footer_text(state: &AppState, width: u16) -> Line<'static> {
    let text = if let Some(error) = state.last_error() {
        format!("error: {error}")
    } else if state.is_turn_active() {
        "mock · streaming · Esc cancel · Ctrl+C cancel".to_owned()
    } else {
        "mock · ready · Enter submit · Ctrl+C exit".to_owned()
    };
    let truncated: String = text
        .chars()
        .take(width.saturating_sub(1) as usize)
        .collect();
    Line::from(Span::styled(truncated, Style::default().fg(Color::Gray)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapping_returns_visual_rows_for_long_lines() {
        assert_eq!(
            wrapped_segments_for_text("abcdefghij", 4),
            ["abcd", "efgh", "ij"]
        );
        assert_eq!(wrapped_segments_for_text("one\ntwo", 10), ["one", "two"]);
    }

    #[test]
    fn cursor_moves_to_visual_rows_when_a_line_wraps() {
        assert_eq!(visual_cursor_position("abcdefghij", 4, 4), (1, 0));
        assert_eq!(visual_cursor_position("abcdefghij", 10, 4), (2, 2));
        assert_eq!(visual_cursor_position("abcd\nef", 4, 4), (0, 3));
    }

    #[test]
    fn editor_adds_a_cursor_row_after_a_full_wrapped_line() {
        let (lines, row, column) = editor_lines_and_cursor("abcd", 4, 4);
        assert_eq!(lines.len(), 2);
        assert_eq!((row, column), (1, 0));
    }
}
