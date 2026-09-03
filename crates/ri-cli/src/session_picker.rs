use anyhow::{bail, Result};
use chrono::{DateTime, FixedOffset, Local};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ri_core::{OpenedSession, SessionRepository, SessionSummary};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use crate::picker::{draw_picker, read_action, PickerAction, PickerState};
use crate::terminal::TerminalGuard;

const TIMESTAMP_FORMAT: &str = "%Y-%m-%d %H:%M";
const PREFERRED_TITLE_WIDTH: usize = 12;

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

    let mut picker = PickerState::new(summaries.len(), 0);
    loop {
        let inner_width = terminal.terminal_mut().size()?.width.saturating_sub(2) as usize;
        let rows = rows(&summaries, picker.selected(), inner_width);
        draw_picker(terminal, "Resume session", &rows, &mut picker)?;
        match read_action()? {
            PickerAction::Up => picker.move_up(false),
            PickerAction::Down => picker.move_down(false),
            PickerAction::Confirm => return Ok(Some(summaries[picker.selected()].path.clone())),
            PickerAction::Cancel => return Ok(None),
            PickerAction::Redraw => {}
        }
    }
}

fn rows(summaries: &[SessionSummary], selected: usize, inner_width: usize) -> Vec<Line<'static>> {
    summaries
        .iter()
        .enumerate()
        .map(|(index, summary)| {
            let is_selected = index == selected;
            let marker_style = if is_selected {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            let marker_width = inner_width.min(2);
            let marker = match marker_width {
                0 => String::new(),
                1 if is_selected => ">".to_owned(),
                1 => " ".to_owned(),
                _ if is_selected => "> ".to_owned(),
                _ => "  ".to_owned(),
            };
            let title = summary
                .name
                .as_deref()
                .or(summary.first_user_preview.as_deref())
                .unwrap_or("unnamed session");
            let timestamp = format_local_timestamp(&summary.updated_at);
            let content = session_row_content(
                &timestamp,
                title,
                summary.message_count,
                inner_width.saturating_sub(marker_width),
            );
            Line::from(vec![Span::styled(marker, marker_style), Span::raw(content)])
        })
        .collect()
}

fn session_row_content(timestamp: &str, title: &str, message_count: usize, width: usize) -> String {
    let full_suffix = format!("{message_count} messages");
    let compact_suffix = format!("{message_count} msgs");
    let candidates = [
        (Some(full_suffix.as_str()), 2, PREFERRED_TITLE_WIDTH),
        (Some(compact_suffix.as_str()), 2, PREFERRED_TITLE_WIDTH),
        (Some(compact_suffix.as_str()), 1, PREFERRED_TITLE_WIDTH),
        (Some(compact_suffix.as_str()), 2, 1),
        (Some(compact_suffix.as_str()), 1, 1),
        (None, 2, 1),
        (None, 1, 1),
        (None, 0, 1),
    ];

    for (suffix, gap, minimum_title_width) in candidates {
        if let Some(row) =
            session_row_candidate(timestamp, title, suffix, gap, minimum_title_width, width)
        {
            return row;
        }
    }

    truncate_display_width(timestamp, width)
}

fn session_row_candidate(
    timestamp: &str,
    title: &str,
    suffix: Option<&str>,
    gap: usize,
    minimum_title_width: usize,
    width: usize,
) -> Option<String> {
    let timestamp_width = UnicodeWidthStr::width(timestamp);
    let suffix_width = suffix.map(UnicodeWidthStr::width).unwrap_or_default();
    let fixed_width = timestamp_width
        .saturating_add(gap)
        .saturating_add(suffix.map(|_| gap).unwrap_or_default())
        .saturating_add(suffix_width);
    let title_width = width.checked_sub(fixed_width)?;
    if title_width < minimum_title_width {
        return None;
    }

    let gap = " ".repeat(gap);
    let title = pad_to_display_width(truncate_display_width(title, title_width), title_width);
    Some(match suffix {
        Some(suffix) => format!("{timestamp}{gap}{title}{gap}{suffix}"),
        None => format!("{timestamp}{gap}{}", title.trim_end()),
    })
}

fn truncate_display_width(text: &str, max_width: usize) -> String {
    if UnicodeWidthStr::width(text) <= max_width {
        return text.to_owned();
    }
    if max_width == 0 {
        return String::new();
    }

    let content_width = max_width.saturating_sub(UnicodeWidthStr::width("…"));
    let mut result = String::new();
    let mut width = 0usize;
    for grapheme in text.graphemes(true) {
        let grapheme_width = UnicodeWidthStr::width(grapheme);
        if width.saturating_add(grapheme_width) > content_width {
            break;
        }
        result.push_str(grapheme);
        width = width.saturating_add(grapheme_width);
    }
    result.push('…');
    result
}

fn pad_to_display_width(mut text: String, width: usize) -> String {
    let padding = width.saturating_sub(UnicodeWidthStr::width(text.as_str()));
    text.push_str(&" ".repeat(padding));
    text
}

fn parse_timestamp(timestamp: &str) -> Option<DateTime<FixedOffset>> {
    DateTime::parse_from_rfc3339(timestamp).ok()
}

#[cfg(test)]
fn format_timestamp_with_offset(timestamp: &str, offset: FixedOffset) -> Option<String> {
    parse_timestamp(timestamp).map(|timestamp| {
        timestamp
            .with_timezone(&offset)
            .format(TIMESTAMP_FORMAT)
            .to_string()
    })
}

fn format_local_timestamp(timestamp: &str) -> String {
    parse_timestamp(timestamp)
        .map(|timestamp| {
            timestamp
                .with_timezone(&Local)
                .format(TIMESTAMP_FORMAT)
                .to_string()
        })
        .unwrap_or_else(|| timestamp.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_rows_use_title_space_beyond_the_old_fixed_limit() {
        let title = "Implement compact thinking selector and tool rendering";
        let row = session_row_content("2026-09-02 10:34", title, 53, 100);

        assert!(row.contains(title));
        assert!(row.contains("selector and tool rendering"));
        assert!(row.ends_with("53 messages"));
        assert_eq!(UnicodeWidthStr::width(row.as_str()), 100);
    }

    #[test]
    fn narrow_rows_ellipsize_the_title_and_compact_the_suffix() {
        let row = session_row_content(
            "2026-09-02 10:34",
            "Implement compact thinking selector and tool rendering",
            53,
            42,
        );

        assert_eq!(row, "2026-09-02 10:34  Implement comp…  53 msgs");
        assert_eq!(UnicodeWidthStr::width(row.as_str()), 42);
    }

    #[test]
    fn display_width_truncation_preserves_wide_graphemes() {
        let truncated = truncate_display_width("ab界cd", 5);

        assert_eq!(truncated, "ab界…");
        assert_eq!(UnicodeWidthStr::width(truncated.as_str()), 5);
        assert_eq!(truncate_display_width("界", 1), "…");
        assert_eq!(truncate_display_width("anything", 0), "");
    }

    #[test]
    fn fixed_offset_timestamp_formatting_is_deterministic() {
        let offset = FixedOffset::east_opt(2 * 60 * 60).expect("valid offset");

        assert_eq!(
            format_timestamp_with_offset("2026-09-02T23:34:56Z", offset).as_deref(),
            Some("2026-09-03 01:34")
        );
    }

    #[test]
    fn malformed_timestamps_fall_back_to_the_original_value() {
        assert!(parse_timestamp("not a timestamp").is_none());
        assert_eq!(format_local_timestamp("not a timestamp"), "not a timestamp");
    }

    #[test]
    fn extremely_narrow_rows_do_not_underflow() {
        for width in 0..24 {
            let row = session_row_content("2026-09-02 10:34", "session title", 5, width);
            assert!(UnicodeWidthStr::width(row.as_str()) <= width);
        }
    }
}
