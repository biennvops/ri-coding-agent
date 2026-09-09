//! CommonMark stays in the presentation layer; cached rows never replace source text.
use std::ops::Range;

use pulldown_cmark::{CodeBlockKind, Event, LinkType, Options, Parser, Tag, TagEnd};
use ratatui::style::{Color, Modifier, Style};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use super::{CachedRow, StyledRange};

struct MarkdownStyles {
    heading: Style,
    link: Style,
    dim: Style,
    code: Style,
    quote: Style,
    bullet: Style,
}

impl Default for MarkdownStyles {
    fn default() -> Self {
        Self {
            heading: Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
            link: Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::UNDERLINED),
            dim: Style::default().fg(Color::DarkGray),
            code: Style::default().fg(Color::Yellow),
            quote: Style::default().fg(Color::Gray),
            bullet: Style::default().fg(Color::Cyan),
        }
    }
}

#[derive(Default)]
struct LogicalLine {
    text: String,
    spans: Vec<(Range<usize>, Style)>,
}

impl LogicalLine {
    fn push(&mut self, text: &str, style: Style) {
        let start = self.text.len();
        for ch in text.chars() {
            match ch {
                '\t' => self.text.push_str("    "),
                ch if ch.is_control() => self.text.push('�'),
                ch => self.text.push(ch),
            }
        }
        self.spans.push((start..self.text.len(), style));
    }
}

struct Prefix {
    first: String,
    rest: String,
    style: Style,
    pending: bool,
}

struct Renderer {
    rows: Vec<CachedRow>,
    line: LogicalLine,
    prefixes: Vec<Prefix>,
    styles: Vec<Style>,
    palette: MarkdownStyles,
    lists: Vec<Option<u64>>,
    links: Vec<(String, bool, bool)>,
    width: usize,
}

impl Renderer {
    fn style(&self) -> Style {
        self.styles.last().copied().unwrap_or_default()
    }

    fn push_style(&mut self, style: Style) {
        self.styles.push(self.style().patch(style));
    }

    fn text(&mut self, text: &str) {
        self.line.push(text, self.style());
    }

    fn prefix(&mut self) -> CachedRow {
        let mut row = CachedRow {
            text: String::new(),
            style: Style::default(),
            spans: Vec::new(),
        };
        // Reserve room for a wide grapheme even in deeply nested/narrow layouts.
        let budget = self.width.saturating_sub(2);
        for prefix in &mut self.prefixes {
            let text = if prefix.pending {
                &prefix.first
            } else {
                &prefix.rest
            };
            let start = row.text.width();
            for g in text.graphemes(true) {
                if row.text.width() + g.width() > budget {
                    break;
                }
                row.text.push_str(g);
            }
            row.spans.push(StyledRange {
                start,
                width: row.text.width() - start,
                style: prefix.style,
            });
            prefix.pending = false;
        }
        row
    }

    fn flush(&mut self, empty: bool) {
        if self.line.text.is_empty() && !empty {
            return;
        }
        let line = std::mem::take(&mut self.line);
        let mut row = self.prefix();
        let mut content = false;
        let mut span_index = 0;
        for (offset, grapheme) in line.text.grapheme_indices(true) {
            let mut displayed = grapheme;
            if grapheme.width() > self.width {
                displayed = "�";
            }
            if content && row.text.width() + displayed.width() > self.width {
                self.rows.push(row);
                row = self.prefix();
            }
            while span_index + 1 < line.spans.len() && line.spans[span_index].0.end <= offset {
                span_index += 1;
            }
            let style = line.spans.get(span_index).map(|s| s.1).unwrap_or_default();
            let start = row.text.width();
            row.text.push_str(displayed);
            let width = displayed.width();
            if let Some(last) = row
                .spans
                .last_mut()
                .filter(|s| s.style == style && s.start + s.width == start)
            {
                last.width += width;
            } else {
                row.spans.push(StyledRange {
                    start,
                    width,
                    style,
                });
            }
            content = true;
        }
        self.rows.push(row);
    }

    fn separate(&mut self) {
        self.flush(false);
        if self.rows.last().is_some_and(|r| !r.text.trim().is_empty()) {
            self.rows.push(CachedRow {
                text: String::new(),
                style: Style::default(),
                spans: Vec::new(),
            });
        }
    }

    fn event(&mut self, event: Event<'_>) {
        match event {
            Event::Start(tag) => match tag {
                Tag::Paragraph => {
                    self.flush(false);
                }
                Tag::Heading { .. } => {
                    self.flush(false);
                    self.push_style(self.palette.heading);
                }
                Tag::Emphasis => self.push_style(Style::default().add_modifier(Modifier::ITALIC)),
                Tag::Strong => self.push_style(Style::default().add_modifier(Modifier::BOLD)),
                Tag::Strikethrough => {
                    self.push_style(Style::default().add_modifier(Modifier::CROSSED_OUT))
                }
                Tag::BlockQuote(_) => {
                    self.flush(false);
                    self.prefixes.push(Prefix {
                        first: "│ ".into(),
                        rest: "│ ".into(),
                        style: self.palette.dim,
                        pending: true,
                    });
                    self.push_style(self.palette.quote);
                }
                Tag::List(start) => {
                    self.flush(false);
                    self.lists.push(start);
                }
                Tag::Item => {
                    self.flush(false);
                    let marker = match self.lists.last_mut() {
                        Some(Some(n)) => {
                            let marker = format!("{n}. ");
                            *n = n.saturating_add(1);
                            marker
                        }
                        _ => "• ".into(),
                    };
                    self.prefixes.push(Prefix {
                        rest: " ".repeat(marker.width()),
                        first: marker,
                        style: self.palette.bullet,
                        pending: true,
                    });
                }
                Tag::CodeBlock(kind) => {
                    self.flush(false);
                    if let CodeBlockKind::Fenced(info) = kind {
                        if !info.is_empty() {
                            self.line.push(&info, self.palette.dim);
                            self.flush(false);
                        }
                    }
                    self.prefixes.push(Prefix {
                        first: "│ ".into(),
                        rest: "│ ".into(),
                        style: self.palette.dim,
                        pending: true,
                    });
                }
                Tag::Link {
                    dest_url,
                    link_type,
                    ..
                } => {
                    self.links.push((
                        dest_url.into_string(),
                        matches!(link_type, LinkType::Autolink | LinkType::Email),
                        false,
                    ));
                    self.push_style(self.palette.link);
                }
                Tag::Image { dest_url, .. } => {
                    self.text("[image: ");
                    self.links.push((dest_url.into_string(), false, true));
                    self.push_style(self.palette.link);
                }
                _ => {}
            },
            Event::End(tag) => match tag {
                TagEnd::Paragraph => {
                    self.flush(false);
                    if self.lists.is_empty() {
                        self.separate();
                    }
                }
                TagEnd::Heading(_) => {
                    self.flush(false);
                    self.styles.pop();
                    self.separate();
                }
                TagEnd::Emphasis | TagEnd::Strong | TagEnd::Strikethrough => {
                    self.styles.pop();
                }
                TagEnd::BlockQuote(_) => {
                    self.flush(false);
                    self.prefixes.pop();
                    self.styles.pop();
                    self.separate();
                }
                TagEnd::Item => {
                    self.flush(self.prefixes.last().is_some_and(|p| p.pending));
                    self.prefixes.pop();
                }
                TagEnd::List(_) => {
                    self.flush(false);
                    self.lists.pop();
                    if self.lists.is_empty() {
                        self.separate();
                    }
                }
                TagEnd::CodeBlock => {
                    self.flush(false);
                    self.prefixes.pop();
                    self.separate();
                }
                TagEnd::Link | TagEnd::Image => {
                    self.styles.pop();
                    if let Some((url, auto, image)) = self.links.pop() {
                        if image {
                            self.text("]");
                        }
                        if !auto {
                            self.line
                                .push(&format!(" ({url})"), self.style().patch(self.palette.dim));
                        }
                    }
                }
                _ => {}
            },
            Event::Text(text) | Event::Html(text) | Event::InlineHtml(text) => {
                let mut parts = text.split('\n').peekable();
                while let Some(part) = parts.next() {
                    self.text(part);
                    if parts.peek().is_some() {
                        self.flush(true);
                    }
                }
            }
            Event::Code(text) => self.line.push(&text, self.style().patch(self.palette.code)),
            Event::SoftBreak | Event::HardBreak => self.flush(true),
            Event::Rule => {
                self.flush(false);
                self.line.push("───", self.palette.dim);
                self.flush(false);
                self.separate();
            }
            Event::TaskListMarker(checked) => self.text(if checked { "[x] " } else { "[ ] " }),
            _ => {}
        }
    }
}

pub(super) fn layout_markdown(source: &str, width: usize) -> Vec<CachedRow> {
    let mut renderer = Renderer {
        rows: Vec::new(),
        line: LogicalLine::default(),
        prefixes: vec![Prefix {
            first: "  ".into(),
            rest: "  ".into(),
            style: Style::default(),
            pending: true,
        }],
        styles: Vec::new(),
        palette: MarkdownStyles::default(),
        lists: Vec::new(),
        links: Vec::new(),
        width: width.max(1),
    };
    for event in Parser::new_ext(
        source,
        Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS,
    ) {
        renderer.event(event);
    }
    renderer.flush(false);
    while renderer.rows.last().is_some_and(|r| r.text.is_empty()) {
        renderer.rows.pop();
    }
    renderer.rows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn visible(source: &str, width: usize) -> String {
        layout_markdown(source, width)
            .iter()
            .map(|r| r.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn style_at(rows: &[CachedRow], text: &str) -> Style {
        let row = rows.iter().find(|r| r.text.contains(text)).unwrap();
        let column = row.text[..row.text.find(text).unwrap()].width();
        row.spans
            .iter()
            .filter(|s| s.start <= column && column < s.start + s.width)
            .fold(row.style, |style, s| style.patch(s.style))
    }

    #[test]
    fn inline_styles_compose_and_syntax_disappears() {
        let rows = layout_markdown(
            "plain **bold and *italic* and `code`** ~~strike~~ [link](https://example.com)",
            200,
        );
        assert_eq!(
            rows[0].text,
            "  plain bold and italic and code strike link (https://example.com)"
        );
        assert!(style_at(&rows, "bold")
            .add_modifier
            .contains(Modifier::BOLD));
        assert!(style_at(&rows, "italic")
            .add_modifier
            .contains(Modifier::BOLD | Modifier::ITALIC));
        assert!(style_at(&rows, "code")
            .add_modifier
            .contains(Modifier::BOLD));
        assert_eq!(style_at(&rows, "code").fg, Some(Color::Yellow));
        assert!(style_at(&rows, "strike")
            .add_modifier
            .contains(Modifier::CROSSED_OUT));
        assert!(style_at(&rows, "link")
            .add_modifier
            .contains(Modifier::UNDERLINED));
        assert_eq!(
            visible(r"\*literal\* &amp; <https://example.com>", 100),
            "  *literal* & https://example.com"
        );
        assert_eq!(visible("a  \nb\nc", 20), "  a\n  b\n  c");
    }

    #[test]
    fn blocks_and_hanging_prefixes() {
        assert_eq!(visible("# ATX\n\nSetext\n---", 80), "  ATX\n\n  Setext");
        assert_eq!(
            visible("9. abcdefghijkl\n10. next\n    - child", 12),
            "  9. abcdefg\n     hijkl\n  10. next\n      • chil\n        d"
        );
        assert_eq!(visible("> abcdefghijkl", 10), "  │ abcdef\n  │ ghijkl");
        assert!(visible("- [x] done\n- [ ] later", 80).contains("• [x] done"));
        assert!(visible("---", 80).contains("───"));
    }

    #[test]
    fn code_is_literal_and_keeps_blank_lines_and_prefixes() {
        assert_eq!(
            visible("```rust\n**abc**\n\n\txyz\n```", 80),
            "  rust\n  │ **abc**\n  │ \n  │     xyz"
        );
        assert_eq!(visible("    **literal**", 80), "  │ **literal**");
        assert_eq!(
            visible("```\nabcdefghijk\n```", 10),
            "  │ abcdef\n  │ ghijk"
        );
    }

    #[test]
    fn fallback_is_safe_and_readable() {
        assert_eq!(
            visible("![architecture](diagram.png)", 100),
            "  [image: architecture] (diagram.png)"
        );
        assert!(visible("<details>hi</details>", 100).contains("<details>hi</details>"));
        assert!(visible("a | b\n-- | --\n1 | 2", 100).contains("-- | --"));
        assert!(!visible("```\n\u{1b}[31m\u{7}\n```", 100).contains('\u{1b}'));
    }

    #[test]
    fn unicode_styles_wrap_with_valid_columns_even_at_tiny_widths() {
        for width in 1..30 {
            let rows = layout_markdown("**ASCII 世界 🦀 e\u{301} more text**", width);
            for row in &rows {
                assert!(row.text.width() <= width, "{width}: {}", row.text);
                for span in &row.spans {
                    assert!(span.start + span.width <= row.text.width());
                }
                for span in row.spans.iter().filter(|s| s.width > 0 && s.start >= 2) {
                    assert!(span.style.add_modifier.contains(Modifier::BOLD));
                }
            }
        }
    }

    #[test]
    fn every_streaming_prefix_of_fixture_is_safe() {
        let source = super::super::MARKDOWN_REPORT;
        for end in (0..=source.len()).filter(|&i| source.is_char_boundary(i)) {
            for width in [1, 8, 40, 100] {
                for row in layout_markdown(&source[..end], width) {
                    assert!(row.text.width() <= width);
                    assert!(row
                        .spans
                        .iter()
                        .all(|s| s.start + s.width <= row.text.width()));
                }
            }
        }
        for source in [
            "*",
            "*bo",
            "**bold",
            "**bold**",
            "[unfinished link](",
            "```rust\nfn main(",
        ] {
            assert!(!visible(source, 80).is_empty());
        }
    }
}
