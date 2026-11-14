//! CommonMark stays in the presentation layer; cached rows never replace source text.
use std::ops::Range;

use pulldown_cmark::{Alignment, CodeBlockKind, Event, LinkType, Options, Parser, Tag, TagEnd};
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

impl CachedRow {
    fn push(&mut self, text: &str, style: Style) {
        if text.is_empty() {
            return;
        }
        let start = self.text.width();
        self.text.push_str(text);
        let width = self.text.width().saturating_sub(start);
        if let Some(last) = self
            .spans
            .last_mut()
            .filter(|s| s.style == style && s.start + s.width == start)
        {
            last.width += width;
        } else {
            self.spans.push(StyledRange {
                start,
                width,
                style,
            });
        }
    }
}

struct Prefix {
    first: String,
    rest: String,
    style: Style,
    pending: bool,
}

struct CodeBlockState {
    info: String,
    source: String,
}

struct TableState {
    alignments: Vec<Alignment>,
    header: Vec<LogicalLine>,
    rows: Vec<Vec<LogicalLine>>,
    current_row: Vec<LogicalLine>,
}

#[derive(Default)]
struct WrappedCellLine {
    width: usize,
    spans: Vec<(String, Style)>,
}

fn wrap_cell(cell: &LogicalLine, col_width: usize) -> Vec<WrappedCellLine> {
    if cell.text.is_empty() {
        return vec![WrappedCellLine::default()];
    }
    let mut lines = Vec::new();
    let mut current = WrappedCellLine::default();
    let mut span_index = 0;
    for (offset, grapheme) in cell.text.grapheme_indices(true) {
        let displayed = if grapheme.width() > col_width {
            ""
        } else {
            grapheme
        };
        let width = displayed.width();
        if current.width > 0 && current.width + width > col_width {
            lines.push(current);
            current = WrappedCellLine::default();
        }
        while span_index + 1 < cell.spans.len() && cell.spans[span_index].0.end <= offset {
            span_index += 1;
        }
        let style = cell.spans.get(span_index).map(|s| s.1).unwrap_or_default();
        if let Some(last) = current.spans.last_mut().filter(|s| s.1 == style) {
            last.0.push_str(displayed);
        } else {
            current.spans.push((displayed.to_owned(), style));
        }
        current.width += width;
    }
    if !current.spans.is_empty() || lines.is_empty() {
        lines.push(current);
    }
    lines
}

struct Renderer {
    max_streaming_code_bytes: Option<usize>,
    code_block: Option<CodeBlockState>,
    table: Option<TableState>,
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

    fn prefix_width(&self) -> usize {
        let budget = self.width.saturating_sub(2);
        let mut width = 0;
        for prefix in &self.prefixes {
            for g in prefix.rest.graphemes(true) {
                if width + g.width() > budget {
                    break;
                }
                width += g.width();
            }
        }
        width
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
            row.push(displayed, style);
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

    fn finish_code_block(&mut self) {
        let Some(block) = self.code_block.take() else {
            return;
        };
        let highlight = self
            .max_streaming_code_bytes
            .is_none_or(|limit| block.source.len() <= limit);
        if highlight {
            if let Some(lines) = super::syntax::highlight_code(&block.info, &block.source) {
                for segments in lines {
                    for (text, style) in segments {
                        self.line.push(text, self.style().patch(style));
                    }
                    self.flush(true);
                }
            } else {
                self.render_plain_code(&block.source);
            }
        } else {
            self.render_plain_code(&block.source);
        }
        self.prefixes.pop();
        self.separate();
    }

    fn render_plain_code(&mut self, source: &str) {
        for line in source.split_terminator('\n') {
            self.text(line);
            self.flush(true);
        }
    }

    fn finish_table(&mut self) {
        let Some(table) = self.table.take() else {
            return;
        };
        let num_cols = table
            .alignments
            .len()
            .max(table.header.len())
            .max(table.rows.iter().map(Vec::len).max().unwrap_or(0));
        if num_cols == 0 {
            return;
        }
        let mut natural_widths = vec![1usize; num_cols];
        for (c, cell) in table.header.iter().enumerate() {
            natural_widths[c] = natural_widths[c].max(cell.text.width());
        }
        for row in &table.rows {
            for (c, cell) in row.iter().enumerate() {
                natural_widths[c] = natural_widths[c].max(cell.text.width());
            }
        }
        let avail = self.width.saturating_sub(self.prefix_width());
        let border_overhead = 3 * num_cols + 1;
        if avail < border_overhead + num_cols {
            for row in std::iter::once(table.header).chain(table.rows) {
                if row.is_empty() {
                    continue;
                }
                for (c, cell) in row.into_iter().enumerate() {
                    if c > 0 {
                        self.line.push(" │ ", self.palette.dim);
                    }
                    let offset = self.line.text.len();
                    self.line.text.push_str(&cell.text);
                    for (range, style) in cell.spans {
                        self.line
                            .spans
                            .push((offset + range.start..offset + range.end, style));
                    }
                }
                self.flush(true);
            }
            self.separate();
            return;
        }
        let content_budget = avail - border_overhead;
        let mut col_widths: Vec<usize> = natural_widths
            .iter()
            .map(|&w| w.clamp(1, content_budget))
            .collect();
        let mut total_width: usize = col_widths.iter().sum();
        while total_width > content_budget {
            if let Some(max_col) = col_widths.iter_mut().max() {
                *max_col -= 1;
                total_width -= 1;
            }
        }
        self.push_table_border(&col_widths, "┌", "┬", "┐");
        if !table.header.is_empty() {
            self.push_table_row(&table.header, &col_widths, &table.alignments);
            if !table.rows.is_empty() {
                self.push_table_border(&col_widths, "├", "┼", "┤");
            }
        }
        for row in &table.rows {
            self.push_table_row(row, &col_widths, &table.alignments);
        }
        self.push_table_border(&col_widths, "└", "┴", "┘");
        self.separate();
    }

    fn push_table_border(&mut self, col_widths: &[usize], left: &str, mid: &str, right: &str) {
        let mut row = self.prefix();
        let mut border = String::from(left);
        for (i, &w) in col_widths.iter().enumerate() {
            if i > 0 {
                border.push_str(mid);
            }
            border.push_str(&"─".repeat(w + 2));
        }
        border.push_str(right);
        row.push(&border, self.palette.dim);
        self.rows.push(row);
    }

    fn push_table_row(
        &mut self,
        cells: &[LogicalLine],
        col_widths: &[usize],
        alignments: &[Alignment],
    ) {
        let num_cols = col_widths.len();
        let empty = LogicalLine::default();
        let wrapped_cols: Vec<Vec<WrappedCellLine>> = (0..num_cols)
            .map(|c| wrap_cell(cells.get(c).unwrap_or(&empty), col_widths[c]))
            .collect();
        let row_height = wrapped_cols.iter().map(Vec::len).max().unwrap_or(1).max(1);
        for line_idx in 0..row_height {
            let mut row = self.prefix();
            row.push("│", self.palette.dim);
            for (c, &col_width) in col_widths.iter().enumerate() {
                let alignment = alignments.get(c).copied().unwrap_or(Alignment::None);
                if let Some(cell_line) = wrapped_cols[c].get(line_idx) {
                    let pad = col_width.saturating_sub(cell_line.width);
                    let left_pad = match alignment {
                        Alignment::Left | Alignment::None => 0,
                        Alignment::Right => pad,
                        Alignment::Center => pad / 2,
                    };
                    row.push(&" ".repeat(1 + left_pad), Style::default());
                    let cell_start = row.text.width();
                    for (text, style) in &cell_line.spans {
                        row.push(text, *style);
                    }
                    let actual_width = row.text.width().saturating_sub(cell_start);
                    let right_pad = col_width.saturating_sub(left_pad.saturating_add(actual_width));
                    row.push(&" ".repeat(right_pad + 1), Style::default());
                } else {
                    row.push(&" ".repeat(col_width + 2), Style::default());
                }
                row.push("│", self.palette.dim);
            }
            self.rows.push(row);
        }
    }

    fn event(&mut self, event: Event<'_>) {
        if let (Some(block), Event::Text(text)) = (&mut self.code_block, &event) {
            block.source.push_str(text);
            return;
        }
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
                    let info = match kind {
                        CodeBlockKind::Fenced(info) => info.into_string(),
                        CodeBlockKind::Indented => String::new(),
                    };
                    if !info.is_empty() {
                        self.line.push(&info, self.palette.dim);
                        self.flush(false);
                    }
                    self.code_block = Some(CodeBlockState {
                        info,
                        source: String::new(),
                    });
                    self.prefixes.push(Prefix {
                        first: "│ ".into(),
                        rest: "│ ".into(),
                        style: self.palette.dim,
                        pending: true,
                    });
                }
                Tag::Table(alignments) => {
                    self.flush(false);
                    self.table = Some(TableState {
                        alignments,
                        header: Vec::new(),
                        rows: Vec::new(),
                        current_row: Vec::new(),
                    });
                }
                Tag::TableHead => {
                    self.push_style(self.palette.heading);
                    if let Some(table) = &mut self.table {
                        table.current_row.clear();
                    }
                }
                Tag::TableRow => {
                    if let Some(table) = &mut self.table {
                        table.current_row.clear();
                    }
                }
                Tag::TableCell => {
                    self.line = LogicalLine::default();
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
                TagEnd::CodeBlock => self.finish_code_block(),
                TagEnd::TableHead => {
                    self.styles.pop();
                    if let Some(table) = &mut self.table {
                        table.header = std::mem::take(&mut table.current_row);
                    }
                }
                TagEnd::TableRow => {
                    if let Some(table) = &mut self.table {
                        table.rows.push(std::mem::take(&mut table.current_row));
                    }
                }
                TagEnd::TableCell => {
                    let cell = std::mem::take(&mut self.line);
                    if let Some(table) = &mut self.table {
                        table.current_row.push(cell);
                    }
                }
                TagEnd::Table => self.finish_table(),
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
                        if self.table.is_some() {
                            self.text(" ");
                        } else {
                            self.flush(true);
                        }
                    }
                }
            }
            Event::Code(text) => self.line.push(&text, self.style().patch(self.palette.code)),
            Event::SoftBreak => self.text(" "),
            Event::HardBreak => {
                if self.table.is_some() {
                    self.text(" ");
                } else {
                    self.flush(true);
                }
            }
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
    layout_markdown_with_code_limit(source, width, None)
}

pub(super) fn layout_streaming_markdown(source: &str, width: usize) -> Vec<CachedRow> {
    layout_markdown_with_code_limit(source, width, Some(MAX_STREAMING_CODE_BYTES))
}

// Keep active large fences responsive; finalized messages use the uncapped highlighting path.
const MAX_STREAMING_CODE_BYTES: usize = 16 * 1024;

fn layout_markdown_with_code_limit(
    source: &str,
    width: usize,
    max_streaming_code_bytes: Option<usize>,
) -> Vec<CachedRow> {
    let mut renderer = Renderer {
        max_streaming_code_bytes,
        code_block: None,
        table: None,
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
        Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS | Options::ENABLE_TABLES,
    ) {
        renderer.event(event);
    }
    renderer.finish_code_block();
    renderer.finish_table();
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
        assert_eq!(visible("a  \nb\nc", 20), "  a\n  b c");
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
        assert!(visible("a | b\n1 | 2", 100).contains("a | b 1 | 2"));
        assert!(!visible("```\n\u{1b}[31m\u{7}\n```", 100).contains('\u{1b}'));
    }

    #[test]
    fn tables_render_borders_alignments_inline_styles_and_wrap_cells() {
        let table = "\
| Left | Center | Right |\n\
| :--- | :----: | ----: |\n\
| **bold** | `code` | 42 |\n\
| 世界 | 🦀 | [x](https://e.co) |";
        let rows = layout_markdown(table, 100);
        let rendered = rows
            .iter()
            .map(|r| r.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            rendered,
            [
                "  ┌──────┬────────┬──────────────────┐",
                "  │ Left │ Center │            Right │",
                "  ├──────┼────────┼──────────────────┤",
                "  │ bold │  code  │               42 │",
                "  │ 世界 │   🦀   │ x (https://e.co) │",
                "  └──────┴────────┴──────────────────┘",
            ]
            .join("\n")
        );
        assert_eq!(
            style_at(&rows, "Left"),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        );
        assert!(style_at(&rows, "bold")
            .add_modifier
            .contains(Modifier::BOLD));
        assert_eq!(style_at(&rows, "code").fg, Some(Color::Yellow));
        assert!(style_at(&rows, "x (https://e.co)")
            .add_modifier
            .contains(Modifier::UNDERLINED));
        assert_eq!(style_at(&rows, "┌"), Style::default().fg(Color::DarkGray));

        let wide = "\
| Flag | Description |\n\
| --- | --- |\n\
| `--release` | Build artifacts in release mode with optimizations |";
        assert_eq!(
            visible(wide, 40),
            [
                "  ┌───────────┬────────────────────────┐",
                "  │ Flag      │ Description            │",
                "  ├───────────┼────────────────────────┤",
                "  │ --release │ Build artifacts in rel │",
                "  │           │ ease mode with optimiz │",
                "  │           │ ations                 │",
                "  └───────────┴────────────────────────┘",
            ]
            .join("\n")
        );

        let quoted = "> | A | B |\n> | - | - |\n> | 1 | 2 |";
        assert_eq!(
            visible(quoted, 80),
            [
                "  │ ┌───┬───┐",
                "  │ │ A │ B │",
                "  │ ├───┼───┤",
                "  │ │ 1 │ 2 │",
                "  │ └───┴───┘",
            ]
            .join("\n")
        );

        for width in 1..60 {
            for row in layout_markdown(table, width) {
                assert!(row.text.width() <= width, "{width}: {}", row.text);
                assert!(row
                    .spans
                    .iter()
                    .all(|s| s.start + s.width <= row.text.width()));
            }
        }
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
    fn highlighted_wrapping_preserves_text_styles_and_safety() {
        let source = "let long_name = \"世界 🦀 e\u{301} a very long string\";";
        let markdown = format!("```rust\n{source}\n```");
        let reference = layout_markdown(&markdown, 200);
        let string_style = style_at(&reference, "世界");
        let keyword_style = style_at(&reference, "let");
        assert_ne!(string_style, keyword_style);
        for width in 1..50 {
            let rows = layout_markdown(&markdown, width);
            for row in &rows {
                assert!(row.text.width() <= width);
                assert!(row
                    .spans
                    .iter()
                    .all(|s| s.start + s.width <= row.text.width()));
            }
            if width >= 6 {
                let code = &rows[1..];
                assert!(code.iter().all(|r| r.text.starts_with("  │ ")));
                assert_eq!(
                    code.iter()
                        .map(|r| &r.text["  │ ".len()..])
                        .collect::<String>(),
                    source
                );
                assert_eq!(style_at(code, "世"), string_style);
                assert!(code
                    .iter()
                    .flat_map(|r| &r.spans)
                    .filter(|s| s.style == string_style || s.style == keyword_style)
                    .all(|s| s.start >= 4));
            }
        }
        let rows = layout_markdown("```rust\n\tlet x = \"\u{1b}[31m\u{7}\";\n\n```", 100);
        assert!(rows.iter().all(|r| !r.text.chars().any(char::is_control)));
        assert!(rows.iter().any(|r| r.text == "  │ "));
        assert!(rows.iter().any(|r| r.text.starts_with("  │     let")));
    }

    #[test]
    fn plain_fallback_and_incomplete_fences() {
        for language in ["ri-weird", "", "text", "plaintext"] {
            let rows = layout_markdown(&format!("```{language}\nfoo **bar**\n```"), 80);
            assert_eq!(rows.last().unwrap().text, "  │ foo **bar**");
            assert_eq!(style_at(&rows, "foo"), Style::default());
        }
        for source in [
            "```r",
            "```rust",
            "```rust\nfn",
            "```rust\nfn main(",
            "```rust\nfn main() {",
        ] {
            let open = layout_markdown(source, 80);
            let closed = layout_markdown(&format!("{source}\n```"), 80);
            assert_eq!(format!("{open:?}"), format!("{closed:?}"));
            if let Some((_, body)) = source.split_once('\n') {
                assert_eq!(open.last().unwrap().text, format!("  │ {body}"));
                assert!(style_at(&open, "fn").fg.is_some());
            }
        }
    }

    #[test]
    fn large_streaming_fences_use_plain_code_until_final_layout() {
        let source = format!(
            "```rust\n{}",
            "fn main() { let value = 1; }\n"
                .repeat(MAX_STREAMING_CODE_BYTES / "fn main() { let value = 1; }\n".len() + 1)
        );
        let streaming = layout_streaming_markdown(&source, 80);
        assert_eq!(style_at(&streaming, "fn"), Style::default());

        let completed = layout_markdown(&source, 80);
        assert_ne!(style_at(&completed, "fn"), Style::default());
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
