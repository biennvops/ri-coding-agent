use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;
use ri_core::{AppState, MessageRole};
use unicode_width::UnicodeWidthStr;

use crate::terminal::TerminalGuard;

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
    let editor_height = (state
        .input()
        .lines()
        .count()
        .max(1)
        .saturating_add(2)
        .min(u16::MAX as usize) as u16)
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

    let transcript = transcript_lines(state);
    let visible_lines = chunks[0].height.saturating_sub(2) as usize;
    let maximum_scroll = transcript.len().saturating_sub(visible_lines);
    let scroll = maximum_scroll.saturating_sub(scroll_from_bottom.min(maximum_scroll));
    let transcript = Paragraph::new(transcript)
        .block(Block::default().borders(Borders::ALL).title(" transcript "))
        .wrap(Wrap { trim: false })
        .scroll((scroll as u16, 0));
    frame.render_widget(transcript, chunks[0]);

    let editor_title = if state.is_turn_active() {
        " input · Esc cancels "
    } else {
        " input · Enter submits · Shift+Enter newline "
    };
    let editor = Paragraph::new(state.input().to_owned())
        .block(Block::default().borders(Borders::ALL).title(editor_title))
        .wrap(Wrap { trim: false });
    frame.render_widget(editor, chunks[1]);

    let footer = footer_text(state, chunks[2].width);
    frame.render_widget(Paragraph::new(footer), chunks[2]);

    if !state.is_turn_active() && chunks[1].height > 2 {
        let (line, column) = cursor_position(state);
        let x = chunks[1].x.saturating_add(1).saturating_add(column as u16);
        let y = chunks[1].y.saturating_add(1).saturating_add(line as u16);
        let max_x = chunks[1]
            .x
            .saturating_add(chunks[1].width.saturating_sub(2));
        let max_y = chunks[1]
            .y
            .saturating_add(chunks[1].height.saturating_sub(2));
        frame.set_cursor_position((x.min(max_x), y.min(max_y)));
    }
}

fn transcript_lines(state: &AppState) -> Vec<Line<'static>> {
    let mut lines = Vec::new();

    for message in state.messages() {
        let (label, style) = match message.role {
            MessageRole::User => ("you", Style::default().fg(Color::Yellow)),
            MessageRole::Assistant => ("assistant", Style::default().fg(Color::Green)),
        };
        lines.push(Line::from(Span::styled(format!("▶ {label}"), style)));
        append_content(&mut lines, &message.content, Style::default());
        if let Some(thinking) = &message.thinking {
            lines.push(Line::from(Span::styled(
                "  thinking:",
                Style::default().fg(Color::Cyan),
            )));
            append_content(&mut lines, thinking, Style::default().fg(Color::DarkGray));
        }
        lines.push(Line::default());
    }

    if let Some((content, thinking)) = state.streaming_assistant() {
        lines.push(Line::from(Span::styled(
            "▶ assistant · streaming",
            Style::default().fg(Color::Green),
        )));
        append_content(&mut lines, content, Style::default());
        if !thinking.is_empty() {
            lines.push(Line::from(Span::styled(
                "  thinking:",
                Style::default().fg(Color::Cyan),
            )));
            append_content(&mut lines, thinking, Style::default().fg(Color::DarkGray));
        }
    }

    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "Start by describing a task.",
            Style::default().fg(Color::DarkGray),
        )));
    }

    lines
}

fn append_content(lines: &mut Vec<Line<'static>>, content: &str, style: Style) {
    for line in content.split('\n') {
        lines.push(Line::from(Span::styled(format!("  {line}"), style)));
    }
}

fn cursor_position(state: &AppState) -> (usize, usize) {
    let prefix = &state.input()[..state.cursor()];
    let line = prefix.bytes().filter(|byte| *byte == b'\n').count();
    let column_text = prefix.rsplit('\n').next().unwrap_or_default();
    (line, column_text.width())
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
