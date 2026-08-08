use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;
use ri_core::{AppState, MessageRole, ModelRef, ToolStatus, TranscriptEntry};

use crate::input::VisualLayout;
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
    let editor_width = area.width.saturating_sub(2).max(1) as usize;
    let editor_layout = VisualLayout::new(state.input(), editor_width);
    let editor_cursor_row = editor_layout.cursor_position(state.cursor()).row;
    let editor_rows = editor_layout.row_count().max(editor_cursor_row + 1);
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
    let editor_layout = VisualLayout::new(state.input(), editor_width);
    let cursor = editor_layout.cursor_position(state.cursor());
    let mut editor_lines: Vec<Line<'static>> = editor_layout
        .rows()
        .iter()
        .cloned()
        .map(Line::from)
        .collect();
    while editor_lines.len() <= cursor.row {
        editor_lines.push(Line::default());
    }
    let editor_visible_lines = chunks[1].height.saturating_sub(2) as usize;
    let editor_scroll = cursor
        .row
        .saturating_sub(editor_visible_lines.saturating_sub(1));
    let editor_title = if state.is_busy() {
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

    if !state.is_busy() && chunks[1].height > 2 {
        let x = chunks[1]
            .x
            .saturating_add(1)
            .saturating_add(cursor.column as u16);
        let y = chunks[1]
            .y
            .saturating_add(1)
            .saturating_add(cursor.row.saturating_sub(editor_scroll) as u16);
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

    for entry in state.transcript_entries() {
        match entry {
            TranscriptEntry::Message(message) => {
                let (label, style) = match message.role {
                    MessageRole::System => ("system", Style::default().fg(Color::Cyan)),
                    MessageRole::User => ("you", Style::default().fg(Color::Yellow)),
                    MessageRole::Assistant => ("assistant", Style::default().fg(Color::Green)),
                };
                lines.extend(layout_styled_lines(&format!("▶ {label}"), style, width));
                append_layout_content(&mut lines, &message.content, Style::default(), width);
                if let Some(thinking) = &message.thinking {
                    lines.extend(layout_styled_lines(
                        "  thinking:",
                        Style::default().fg(Color::Cyan),
                        width,
                    ));
                    append_layout_content(
                        &mut lines,
                        thinking,
                        Style::default().fg(Color::DarkGray),
                        width,
                    );
                }
            }
            TranscriptEntry::Tool(tool) => append_tool_entry(&mut lines, tool, width),
        }
        lines.push(Line::default());
    }

    if let Some((content, thinking)) = state.streaming_assistant() {
        lines.extend(layout_styled_lines(
            "▶ assistant · streaming",
            Style::default().fg(Color::Green),
            width,
        ));
        append_layout_content(&mut lines, content, Style::default(), width);
        if !thinking.is_empty() {
            lines.extend(layout_styled_lines(
                "  thinking:",
                Style::default().fg(Color::Cyan),
                width,
            ));
            append_layout_content(
                &mut lines,
                thinking,
                Style::default().fg(Color::DarkGray),
                width,
            );
        }
    }

    if lines.is_empty() {
        lines.extend(layout_styled_lines(
            "Start by describing a task.",
            Style::default().fg(Color::DarkGray),
            width,
        ));
    }

    lines
}

fn append_tool_entry(
    lines: &mut Vec<Line<'static>>,
    tool: &ri_core::ToolTranscriptEntry,
    width: usize,
) {
    let (marker, style) = match &tool.status {
        ToolStatus::Running => ("▶", Style::default().fg(Color::Yellow)),
        ToolStatus::Finished(metadata) if metadata.success => {
            ("✓", Style::default().fg(Color::Green))
        }
        ToolStatus::Finished(_) => ("✗", Style::default().fg(Color::Red)),
    };
    let title = if tool.arguments.is_empty() {
        format!("{marker} {}", tool.name)
    } else {
        format!("{marker} {} {}", tool.name, tool.arguments)
    };
    lines.extend(layout_styled_lines(&title, style, width));
    if tool.output.is_empty() && matches!(tool.status, ToolStatus::Running) {
        lines.extend(layout_styled_lines(
            "  running...",
            Style::default().fg(Color::DarkGray),
            width,
        ));
    } else {
        append_layout_content(lines, &tool.output, Style::default(), width);
    }
    if let ToolStatus::Finished(metadata) = &tool.status {
        let status = if metadata.cancelled {
            "cancelled".to_owned()
        } else if metadata.timed_out {
            "timed out".to_owned()
        } else if let Some(exit_code) = metadata.exit_code {
            format!("exited {exit_code}")
        } else if metadata.success {
            "completed".to_owned()
        } else {
            "failed".to_owned()
        };
        lines.extend(layout_styled_lines(
            &format!("  {status} · {:.1}s", metadata.duration.as_secs_f64()),
            Style::default().fg(Color::DarkGray),
            width,
        ));
        if metadata.truncated || tool.output_truncated {
            let spill = metadata
                .full_output_path
                .as_ref()
                .map(|path| format!(" · full output: {}", path.display()))
                .unwrap_or_default();
            lines.extend(layout_styled_lines(
                &format!("  output truncated{spill}"),
                Style::default().fg(Color::DarkGray),
                width,
            ));
        }
    }
}

fn append_layout_content(
    lines: &mut Vec<Line<'static>>,
    content: &str,
    style: Style,
    width: usize,
) {
    for line in content.split('\n') {
        lines.extend(layout_styled_lines(&format!("  {line}"), style, width));
    }
}

fn layout_styled_lines(text: &str, style: Style, width: usize) -> Vec<Line<'static>> {
    VisualLayout::new(text, width)
        .rows()
        .iter()
        .cloned()
        .map(|row| Line::from(Span::styled(row, style)))
        .collect()
}

fn footer_text(state: &AppState, width: u16) -> Line<'static> {
    let model = state
        .active_model()
        .map(ModelRef::display_name)
        .unwrap_or_else(|| "mock/mock".to_owned());
    let session = state
        .session_info()
        .map(|info| format!("session: {}", info.display_name()))
        .unwrap_or_else(|| "session: ephemeral".to_owned());
    let usage = state.context_usage();
    let current = format_token_count(usage.current_tokens());
    let context = match usage.context_window {
        Some(window)
            if matches!(usage.source, ri_core::UsageSource::Provider)
                && usage.input_tokens.is_some() =>
        {
            format!("ctx {current}/{}", format_token_count(window))
        }
        Some(window) => format!("ctx ~{current}/{}", format_token_count(window)),
        None => format!("ctx ~{current}"),
    };
    let text = if let Some(error) = state.last_error() {
        format!("{model} · {context} · {session} · error: {error}")
    } else if state.is_busy() {
        format!("{model} · {context} · {session} · busy · Esc cancel")
    } else {
        format!("{model} · {context} · {session} · ready · Enter submit · Ctrl+C exit")
    };
    let truncated: String = text
        .chars()
        .take(width.saturating_sub(1) as usize)
        .collect();
    Line::from(Span::styled(truncated, Style::default().fg(Color::Gray)))
}

fn format_token_count(value: u64) -> String {
    if value < 1_000 {
        value.to_string()
    } else if value < 1_000_000 {
        trim_decimal(format!("{:.1}k", value as f64 / 1_000.0))
    } else {
        trim_decimal(format!("{:.1}m", value as f64 / 1_000_000.0))
    }
}

fn trim_decimal(value: String) -> String {
    value.trim_end_matches('0').trim_end_matches('.').to_owned()
}
