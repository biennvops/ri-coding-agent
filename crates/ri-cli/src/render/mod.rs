mod fixtures;

pub use fixtures::{append_streaming_delta, synthetic_transcript};

use std::collections::{HashMap, HashSet};

use ratatui::backend::Backend;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::{Frame, Terminal};
use ri_core::{
    AppState, MessageRole, ModelRef, StreamingAssistantState, ToolStatus, ToolTranscriptEntry,
    TranscriptEntry, TranscriptEntryId, TranscriptEntryState,
};

use crate::commands::{matching_commands, CommandSuggestions};
use crate::input::VisualLayout;

const MAX_VISIBLE_COMMAND_SUGGESTIONS: usize = 6;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RenderStats {
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub entries_reflowed: usize,
    pub bytes_reflowed: usize,
    pub rows_visited: usize,
    pub visible_rows: usize,
    pub index_lookups: usize,
    pub entries_reindexed: usize,
    pub editor_cache_hits: usize,
    pub editor_cache_misses: usize,
}

pub struct TuiRenderer {
    transcript: TranscriptLayoutCache,
    editor: EditorLayoutCache,
    last_stats: RenderStats,
    transcript_viewport_rows: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptScroll {
    top_row: usize,
    maximum_scroll: usize,
    following_bottom: bool,
}

impl Default for TranscriptScroll {
    fn default() -> Self {
        Self {
            top_row: 0,
            maximum_scroll: 0,
            following_bottom: true,
        }
    }
}

impl TranscriptScroll {
    pub fn scroll_up(&mut self, rows: usize) {
        self.top_row = self.top_row.saturating_sub(rows);
        self.following_bottom = self.top_row == self.maximum_scroll;
    }

    pub fn scroll_down(&mut self, rows: usize) {
        self.top_row = self.top_row.saturating_add(rows).min(self.maximum_scroll);
        self.following_bottom = self.top_row == self.maximum_scroll;
    }

    pub fn follow_bottom(&mut self) {
        self.top_row = self.maximum_scroll;
        self.following_bottom = true;
    }

    pub fn from_bottom(&self) -> usize {
        self.maximum_scroll.saturating_sub(self.top_row)
    }

    pub(crate) fn update_maximum(&mut self, maximum_scroll: usize) {
        self.maximum_scroll = maximum_scroll;
        if self.following_bottom {
            self.top_row = maximum_scroll;
        } else {
            self.top_row = self.top_row.min(maximum_scroll);
            self.following_bottom = self.top_row == maximum_scroll;
        }
    }
}

enum Viewport<'a> {
    FromBottom(usize),
    Interactive(&'a mut TranscriptScroll),
}

impl Default for TuiRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl TuiRenderer {
    pub fn new() -> Self {
        Self {
            transcript: TranscriptLayoutCache::default(),
            editor: EditorLayoutCache::default(),
            last_stats: RenderStats::default(),
            transcript_viewport_rows: 0,
        }
    }

    pub fn stats(&self) -> &RenderStats {
        &self.last_stats
    }

    pub fn cached_transcript_entries(&self) -> usize {
        self.transcript.cached_entries()
    }

    pub fn cached_transcript_rows(&self) -> usize {
        self.transcript.cached_rows()
    }

    pub fn transcript_total_rows(&self) -> usize {
        self.transcript.total_rows()
    }

    pub fn transcript_page_rows(&self) -> usize {
        self.transcript_viewport_rows.max(1)
    }

    pub fn move_editor_vertical(
        &mut self,
        state: &AppState,
        width: usize,
        cursor: usize,
        direction: isize,
        preferred_column: Option<usize>,
    ) -> Option<(usize, usize)> {
        let mut stats = RenderStats::default();
        let layout = self.editor.ensure(state, width, &mut stats);
        layout.move_vertical(cursor, direction, preferred_column)
    }

    pub fn draw<B: Backend>(
        &mut self,
        terminal: &mut Terminal<B>,
        state: &AppState,
        scroll_from_bottom: usize,
    ) -> Result<(), B::Error> {
        self.last_stats = RenderStats::default();
        terminal.draw(|frame| {
            self.render_frame(frame, state, Viewport::FromBottom(scroll_from_bottom), None);
        })?;
        Ok(())
    }

    pub(crate) fn draw_interactive<B: Backend>(
        &mut self,
        terminal: &mut Terminal<B>,
        state: &AppState,
        scroll: &mut TranscriptScroll,
        suggestions: &CommandSuggestions,
    ) -> Result<(), B::Error> {
        self.last_stats = RenderStats::default();
        terminal.draw(|frame| {
            self.render_frame(
                frame,
                state,
                Viewport::Interactive(scroll),
                Some(suggestions),
            );
        })?;
        Ok(())
    }

    fn render_frame(
        &mut self,
        frame: &mut Frame<'_>,
        state: &AppState,
        viewport: Viewport<'_>,
        suggestions: Option<&CommandSuggestions>,
    ) {
        let area = frame.area();
        let editor_width = area.width.saturating_sub(2).max(1) as usize;
        let (editor_rows, editor_cursor_row) = {
            let editor_layout = self
                .editor
                .ensure(state, editor_width, &mut self.last_stats);
            (
                editor_layout.row_count(),
                editor_layout.cursor_position(state.cursor()).row,
            )
        };
        let editor_rows = editor_rows.max(editor_cursor_row + 1);
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
        self.transcript
            .prepare(state, transcript_width, &mut self.last_stats);
        let visible_lines = chunks[0].height.saturating_sub(2) as usize;
        self.transcript_viewport_rows = visible_lines;
        let maximum_scroll = self.transcript.total_rows().saturating_sub(visible_lines);
        let (scroll, scroll_from_bottom) = match viewport {
            Viewport::FromBottom(from_bottom) => (
                maximum_scroll.saturating_sub(from_bottom.min(maximum_scroll)),
                from_bottom.min(maximum_scroll),
            ),
            Viewport::Interactive(state) => {
                state.update_maximum(maximum_scroll);
                (state.top_row, state.from_bottom())
            }
        };
        let transcript_block = Block::default().borders(Borders::ALL).title(" transcript ");
        let transcript_inner = transcript_block.inner(chunks[0]);
        frame.render_widget(transcript_block, chunks[0]);
        self.transcript.render_visible(
            transcript_inner,
            scroll,
            &mut self.last_stats,
            frame.buffer_mut(),
        );

        let editor_width = chunks[1].width.saturating_sub(2).max(1) as usize;
        let (cursor, editor_lines) = {
            let editor_layout = self
                .editor
                .ensure(state, editor_width, &mut self.last_stats);
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
            (cursor, editor_lines)
        };
        let editor_visible_lines = chunks[1].height.saturating_sub(2) as usize;
        let editor_scroll = cursor
            .row
            .saturating_sub(editor_visible_lines.saturating_sub(1));
        let editor_title = if state.is_busy() {
            " input · Esc cancels · PgUp scroll "
        } else {
            " input · Enter submits · Shift+Enter newline · PgUp scroll "
        };
        let editor = Paragraph::new(editor_lines)
            .block(Block::default().borders(Borders::ALL).title(editor_title))
            .scroll((editor_scroll.min(u16::MAX as usize) as u16, 0));
        frame.render_widget(editor, chunks[1]);
        if let Some(suggestions) = suggestions {
            render_command_suggestions(frame, state, suggestions, chunks[1]);
        }

        let footer = footer_text(state, chunks[2].width, scroll_from_bottom);
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
}

fn render_command_suggestions(
    frame: &mut Frame<'_>,
    state: &AppState,
    suggestions: &CommandSuggestions,
    editor_area: Rect,
) {
    if !suggestions.is_visible(state) {
        return;
    }
    let total = matching_commands(state.input()).count();
    let available_height = editor_area.y.saturating_sub(frame.area().y) as usize;
    let width = editor_area.width.saturating_sub(2).min(64);
    if total == 0 || available_height < 3 || width < 4 {
        return;
    }

    let visible = total
        .min(MAX_VISIBLE_COMMAND_SUGGESTIONS)
        .min(available_height.saturating_sub(2));
    if visible == 0 {
        return;
    }
    let selected = suggestions.selected(state);
    let start = selected
        .saturating_sub(visible.saturating_sub(1))
        .min(total.saturating_sub(visible));
    let lines = matching_commands(state.input())
        .skip(start)
        .take(visible)
        .enumerate()
        .map(|(offset, spec)| {
            let text = format!("/{:<10} {}", spec.name, spec.description);
            if start + offset == selected {
                Line::styled(text, Style::default().fg(Color::Black).bg(Color::Cyan))
            } else {
                Line::from(text)
            }
        })
        .collect::<Vec<_>>();
    let height = visible.saturating_add(2) as u16;
    let area = Rect::new(
        editor_area.x.saturating_add(1),
        editor_area.y.saturating_sub(height),
        width,
        height,
    );
    frame.render_widget(Clear, area);
    frame.render_widget(
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" commands ")),
        area,
    );
}

#[derive(Clone, Debug)]
struct CachedRow {
    text: String,
    style: Style,
}

#[derive(Clone, Debug)]
struct CachedLayout {
    revision: u64,
    rows: Vec<CachedRow>,
}

#[derive(Clone, Debug)]
struct StreamingCachedLayout {
    revision: u64,
    header: CachedRow,
    thinking_header: CachedRow,
    content_len: usize,
    content_rows: Vec<CachedRow>,
    content_starts: Vec<usize>,
    thinking_len: usize,
    thinking_rows: Vec<CachedRow>,
    thinking_starts: Vec<usize>,
}

impl StreamingCachedLayout {
    fn row_count(&self) -> usize {
        1usize
            .saturating_add(self.content_rows.len())
            .saturating_add((!self.thinking_rows.is_empty()) as usize)
            .saturating_add(self.thinking_rows.len())
    }

    fn row_at(&self, row: usize) -> Option<&CachedRow> {
        if row == 0 {
            return Some(&self.header);
        }
        let row = row - 1;
        if row < self.content_rows.len() {
            return self.content_rows.get(row);
        }
        let row = row - self.content_rows.len();
        if self.thinking_rows.is_empty() {
            return None;
        }
        if row == 0 {
            return Some(&self.thinking_header);
        }
        self.thinking_rows.get(row - 1)
    }
}

#[derive(Clone, Copy, Debug)]
struct EntryLayoutIndex {
    id: TranscriptEntryId,
    start: usize,
    row_count: usize,
}

#[derive(Default)]
struct TranscriptLayoutCache {
    epoch: Option<u64>,
    width: usize,
    layouts: HashMap<TranscriptEntryId, CachedLayout>,
    indices: Vec<EntryLayoutIndex>,
    positions: HashMap<TranscriptEntryId, usize>,
    static_total_rows: usize,
    streaming_id: Option<TranscriptEntryId>,
    streaming_row_count: usize,
    streaming_layout: Option<StreamingCachedLayout>,
    fallback: Option<CachedLayout>,
}

impl TranscriptLayoutCache {
    fn cached_entries(&self) -> usize {
        self.layouts.len() + usize::from(self.streaming_layout.is_some())
    }

    fn cached_rows(&self) -> usize {
        self.layouts
            .values()
            .map(|layout| layout.rows.len())
            .sum::<usize>()
            .saturating_add(
                self.streaming_layout
                    .as_ref()
                    .map(StreamingCachedLayout::row_count)
                    .unwrap_or_default(),
            )
    }

    fn total_rows(&self) -> usize {
        let rows = self
            .static_total_rows
            .saturating_add(self.streaming_row_count);
        if rows == 0 {
            self.fallback
                .as_ref()
                .map(|layout| layout.rows.len())
                .unwrap_or_default()
        } else {
            rows
        }
    }

    fn prepare(&mut self, state: &AppState, width: usize, stats: &mut RenderStats) {
        let width = width.max(1);
        let cold = self.epoch != Some(state.transcript_epoch()) || self.width != width;
        if cold {
            self.reset(state.transcript_epoch(), width);
            for entry in state.transcript_entries() {
                self.append_static_entry(entry, width, stats);
            }
            self.update_streaming(state.streaming_assistant_state(), width, stats);
            self.update_fallback(width, stats);
            return;
        }

        if state.transcript_entries().len() < self.indices.len() {
            self.reset(state.transcript_epoch(), width);
            for entry in state.transcript_entries() {
                self.append_static_entry(entry, width, stats);
            }
        } else {
            let first_new = self.indices.len();
            for entry in state.transcript_entries().iter().skip(first_new) {
                self.append_static_entry(entry, width, stats);
            }
        }

        let mut changed = HashSet::new();
        for id in state.pending_transcript_changes() {
            changed.insert(*id);
        }
        for id in changed {
            if let Some(&index) = self.positions.get(&id) {
                self.refresh_static_entry(index, state, width, stats);
            }
        }
        self.update_streaming(state.streaming_assistant_state(), width, stats);
        self.update_fallback(width, stats);
    }

    fn reset(&mut self, epoch: u64, width: usize) {
        self.epoch = Some(epoch);
        self.width = width;
        self.layouts = HashMap::new();
        self.indices = Vec::new();
        self.positions = HashMap::new();
        self.static_total_rows = 0;
        self.streaming_id = None;
        self.streaming_row_count = 0;
        self.streaming_layout = None;
        self.fallback = None;
    }

    fn append_static_entry(
        &mut self,
        entry: &TranscriptEntryState,
        width: usize,
        stats: &mut RenderStats,
    ) {
        let start = self.static_total_rows;
        self.ensure_layout(entry.id, entry.revision, &entry.entry, width, stats);
        let row_count = self
            .layouts
            .get(&entry.id)
            .map(|layout| layout.rows.len())
            .unwrap_or_default();
        let index = self.indices.len();
        self.indices.push(EntryLayoutIndex {
            id: entry.id,
            start,
            row_count,
        });
        self.positions.insert(entry.id, index);
        self.static_total_rows = self.static_total_rows.saturating_add(row_count);
    }

    fn refresh_static_entry(
        &mut self,
        index: usize,
        state: &AppState,
        width: usize,
        stats: &mut RenderStats,
    ) {
        let Some(entry) = state.transcript_entries().get(index) else {
            return;
        };
        let old_row_count = self.indices[index].row_count;
        self.ensure_layout(entry.id, entry.revision, &entry.entry, width, stats);
        let new_row_count = self
            .layouts
            .get(&entry.id)
            .map(|layout| layout.rows.len())
            .unwrap_or_default();
        if old_row_count == new_row_count {
            return;
        }
        self.indices[index].row_count = new_row_count;
        if index + 1 == self.indices.len() {
            self.static_total_rows = self
                .static_total_rows
                .saturating_sub(old_row_count)
                .saturating_add(new_row_count);
        } else {
            self.reindex_from(index + 1, stats);
        }
    }

    fn reindex_from(&mut self, start: usize, stats: &mut RenderStats) {
        let mut next = if start == 0 {
            0
        } else {
            let previous = &self.indices[start - 1];
            previous.start.saturating_add(previous.row_count)
        };
        for index in start..self.indices.len() {
            self.indices[index].start = next;
            next = next.saturating_add(self.indices[index].row_count);
        }
        self.static_total_rows = next;
        stats.entries_reindexed = stats
            .entries_reindexed
            .saturating_add(self.indices.len().saturating_sub(start));
    }

    fn update_streaming(
        &mut self,
        streaming: Option<&StreamingAssistantState>,
        width: usize,
        stats: &mut RenderStats,
    ) {
        let current_id = streaming.map(|assistant| assistant.id);
        if self.streaming_id != current_id {
            self.streaming_id = current_id;
            self.streaming_row_count = 0;
            self.streaming_layout = None;
        }

        if let Some(streaming) = streaming {
            self.ensure_streaming_layout(streaming, width, stats);
            self.streaming_row_count = self
                .streaming_layout
                .as_ref()
                .map(StreamingCachedLayout::row_count)
                .unwrap_or_default();
        } else {
            self.streaming_row_count = 0;
            self.streaming_layout = None;
        }
    }

    fn update_fallback(&mut self, width: usize, stats: &mut RenderStats) {
        if self.static_total_rows == 0 && self.streaming_row_count == 0 {
            if self.fallback.is_none() {
                let rows = layout_styled_lines(
                    "Start by describing a task.",
                    Style::default().fg(Color::DarkGray),
                    width,
                );
                stats.cache_misses = stats.cache_misses.saturating_add(1);
                self.fallback = Some(CachedLayout { revision: 0, rows });
            } else {
                stats.cache_hits = stats.cache_hits.saturating_add(1);
            }
        } else {
            self.fallback = None;
        }
    }

    fn ensure_layout(
        &mut self,
        id: TranscriptEntryId,
        revision: u64,
        entry: &TranscriptEntry,
        width: usize,
        stats: &mut RenderStats,
    ) {
        if self
            .layouts
            .get(&id)
            .is_some_and(|layout| layout.revision == revision)
        {
            stats.cache_hits = stats.cache_hits.saturating_add(1);
            return;
        }
        let rows = layout_entry(entry, width);
        stats.cache_misses = stats.cache_misses.saturating_add(1);
        stats.entries_reflowed = stats.entries_reflowed.saturating_add(1);
        stats.bytes_reflowed = stats
            .bytes_reflowed
            .saturating_add(entry_render_bytes(entry));
        self.layouts.insert(id, CachedLayout { revision, rows });
    }

    fn ensure_streaming_layout(
        &mut self,
        streaming: &StreamingAssistantState,
        width: usize,
        stats: &mut RenderStats,
    ) {
        let previous = self.streaming_layout.take();
        let Some(mut layout) = previous else {
            let (content_rows, content_starts) =
                layout_content_section(&streaming.content, Style::default(), width);
            let (thinking_rows, thinking_starts) = if streaming.thinking.is_empty() {
                (Vec::new(), Vec::new())
            } else {
                layout_content_section(
                    &streaming.thinking,
                    Style::default().fg(Color::DarkGray),
                    width,
                )
            };
            stats.cache_misses = stats.cache_misses.saturating_add(1);
            stats.entries_reflowed = stats.entries_reflowed.saturating_add(1);
            stats.bytes_reflowed = stats.bytes_reflowed.saturating_add(
                streaming
                    .content
                    .len()
                    .saturating_add(streaming.thinking.len()),
            );
            self.streaming_layout = Some(StreamingCachedLayout {
                revision: streaming.revision,
                header: CachedRow {
                    text: "▶ assistant · streaming".to_owned(),
                    style: Style::default().fg(Color::Green),
                },
                thinking_header: CachedRow {
                    text: "  thinking:".to_owned(),
                    style: Style::default().fg(Color::Cyan),
                },
                content_len: streaming.content.len(),
                content_rows,
                content_starts,
                thinking_len: streaming.thinking.len(),
                thinking_rows,
                thinking_starts,
            });
            return;
        };

        if layout.revision == streaming.revision {
            stats.cache_hits = stats.cache_hits.saturating_add(1);
            self.streaming_layout = Some(layout);
            return;
        }

        let mut bytes_reflowed: usize = 0;
        let append_only = streaming.content.len() >= layout.content_len
            && streaming.thinking.len() >= layout.thinking_len;
        if append_only {
            if streaming.content.len() > layout.content_len {
                let (rows, starts, bytes) = append_content_section(
                    layout.content_rows,
                    layout.content_starts,
                    &streaming.content,
                    Style::default(),
                    width,
                );
                layout.content_rows = rows;
                layout.content_starts = starts;
                bytes_reflowed = bytes_reflowed.saturating_add(bytes);
            }
            if streaming.thinking.len() > layout.thinking_len {
                let (rows, starts, bytes) = append_content_section(
                    layout.thinking_rows,
                    layout.thinking_starts,
                    &streaming.thinking,
                    Style::default().fg(Color::DarkGray),
                    width,
                );
                layout.thinking_rows = rows;
                layout.thinking_starts = starts;
                bytes_reflowed = bytes_reflowed.saturating_add(bytes);
            }
        } else {
            let (content_rows, content_starts) =
                layout_content_section(&streaming.content, Style::default(), width);
            let (thinking_rows, thinking_starts) = if streaming.thinking.is_empty() {
                (Vec::new(), Vec::new())
            } else {
                layout_content_section(
                    &streaming.thinking,
                    Style::default().fg(Color::DarkGray),
                    width,
                )
            };
            layout.content_rows = content_rows;
            layout.content_starts = content_starts;
            layout.thinking_rows = thinking_rows;
            layout.thinking_starts = thinking_starts;
            bytes_reflowed = streaming
                .content
                .len()
                .saturating_add(streaming.thinking.len());
        }
        layout.content_len = streaming.content.len();
        layout.thinking_len = streaming.thinking.len();
        layout.revision = streaming.revision;
        stats.cache_misses = stats.cache_misses.saturating_add(1);
        stats.entries_reflowed = stats.entries_reflowed.saturating_add(1);
        stats.bytes_reflowed = stats.bytes_reflowed.saturating_add(bytes_reflowed);
        self.streaming_layout = Some(layout);
    }

    fn render_visible(
        &self,
        area: Rect,
        start: usize,
        stats: &mut RenderStats,
        buffer: &mut Buffer,
    ) {
        let total_rows = self.total_rows();
        let start = start.min(total_rows.saturating_sub(1));
        let visible_rows = (area.height as usize).min(total_rows.saturating_sub(start));
        stats.visible_rows = visible_rows;
        for offset in 0..visible_rows {
            let y = area.y.saturating_add(offset as u16);
            for x in area.x..area.right() {
                buffer[(x, y)].reset();
            }
            stats.rows_visited = stats.rows_visited.saturating_add(1);
            let Some(row) = self.row_at(start.saturating_add(offset), stats) else {
                continue;
            };
            buffer.set_stringn(area.x, y, row.text.as_str(), area.width as usize, row.style);
        }
    }

    fn row_at(&self, row: usize, stats: &mut RenderStats) -> Option<&CachedRow> {
        stats.index_lookups = stats.index_lookups.saturating_add(1);
        if self.static_total_rows > 0 && row < self.static_total_rows {
            let mut low = 0;
            let mut high = self.indices.len();
            while low < high {
                let middle = low + (high - low) / 2;
                if self.indices[middle].start <= row {
                    low = middle + 1;
                } else {
                    high = middle;
                }
            }
            let index = low.checked_sub(1)?;
            let entry = self.indices.get(index)?;
            let local = row.saturating_sub(entry.start);
            stats.cache_hits = stats.cache_hits.saturating_add(1);
            return self.layouts.get(&entry.id)?.rows.get(local);
        }

        if self.streaming_id.is_some() {
            let local = row.saturating_sub(self.static_total_rows);
            stats.cache_hits = stats.cache_hits.saturating_add(1);
            return self.streaming_layout.as_ref()?.row_at(local);
        }

        stats.cache_hits = stats.cache_hits.saturating_add(1);
        self.fallback.as_ref()?.rows.get(row)
    }
}

#[derive(Default)]
struct EditorLayoutCache {
    revision: Option<u64>,
    width: usize,
    layout: Option<VisualLayout>,
}

impl EditorLayoutCache {
    fn ensure<'a>(
        &'a mut self,
        state: &AppState,
        width: usize,
        stats: &mut RenderStats,
    ) -> &'a VisualLayout {
        let width = width.max(1);
        let revision = state.input_revision();
        if self.revision == Some(revision) && self.width == width {
            stats.editor_cache_hits = stats.editor_cache_hits.saturating_add(1);
            return self
                .layout
                .as_ref()
                .expect("editor layout cache is populated");
        }
        stats.editor_cache_misses = stats.editor_cache_misses.saturating_add(1);
        self.revision = Some(revision);
        self.width = width;
        self.layout = Some(VisualLayout::new(state.input(), width));
        self.layout
            .as_ref()
            .expect("editor layout cache is populated")
    }
}

fn layout_entry(entry: &TranscriptEntry, width: usize) -> Vec<CachedRow> {
    let mut rows = match entry {
        TranscriptEntry::Message(message) => layout_message(
            message.role,
            &message.content,
            message.thinking.as_deref(),
            width,
            match message.role {
                MessageRole::System => "system",
                MessageRole::User => "you",
                MessageRole::Assistant => "assistant",
            },
        ),
        TranscriptEntry::Tool(tool) => {
            let mut rows = Vec::new();
            append_tool_entry(&mut rows, tool, width);
            rows
        }
    };
    rows.push(CachedRow {
        text: String::new(),
        style: Style::default(),
    });
    rows
}

fn layout_message(
    role: MessageRole,
    content: &str,
    thinking: Option<&str>,
    width: usize,
    label: &str,
) -> Vec<CachedRow> {
    let style = match role {
        MessageRole::System => Style::default().fg(Color::Cyan),
        MessageRole::User => Style::default().fg(Color::Yellow),
        MessageRole::Assistant => Style::default().fg(Color::Green),
    };
    let mut rows = layout_styled_lines(&format!("▶ {label}"), style, width);
    append_layout_content(&mut rows, content, Style::default(), width);
    if let Some(thinking) = thinking {
        rows.extend(layout_styled_lines(
            "  thinking:",
            Style::default().fg(Color::Cyan),
            width,
        ));
        append_layout_content(
            &mut rows,
            thinking,
            Style::default().fg(Color::DarkGray),
            width,
        );
    }
    rows
}

fn append_tool_entry(lines: &mut Vec<CachedRow>, tool: &ToolTranscriptEntry, width: usize) {
    let (marker, style) = match &tool.status {
        ToolStatus::Running => ("▶", Style::default().fg(Color::Yellow)),
        ToolStatus::Finished(metadata) if metadata.success => {
            ("✓", Style::default().fg(Color::Green))
        }
        ToolStatus::Finished(_) => ("✗", Style::default().fg(Color::Red)),
    };
    lines.extend(layout_styled_lines(
        &format!("{marker} {}", tool.summary),
        style,
        width,
    ));
    if let Some(preview) = &tool.preview {
        append_layout_content(lines, preview, Style::default(), width);
    }
    if tool.output.is_empty() && matches!(tool.status, ToolStatus::Running) {
        lines.extend(layout_styled_lines(
            "  running...",
            Style::default().fg(Color::DarkGray),
            width,
        ));
    } else if !tool.output.is_empty() {
        if tool.preview.is_some() {
            lines.push(CachedRow {
                text: String::new(),
                style: Style::default(),
            });
        }
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

fn append_layout_content(lines: &mut Vec<CachedRow>, content: &str, style: Style, width: usize) {
    for line in content.split('\n') {
        lines.extend(layout_styled_lines(&format!("  {line}"), style, width));
    }
}

fn layout_content_section(
    content: &str,
    style: Style,
    width: usize,
) -> (Vec<CachedRow>, Vec<usize>) {
    layout_content_section_from(content, style, width, true)
}

fn layout_content_section_from(
    content: &str,
    style: Style,
    width: usize,
    first_line_prefix: bool,
) -> (Vec<CachedRow>, Vec<usize>) {
    let mut rows = Vec::new();
    let mut starts = Vec::new();
    let mut source_offset: usize = 0;
    for (line_index, line) in content.split('\n').enumerate() {
        let prefix = if line_index == 0 && !first_line_prefix {
            ""
        } else {
            "  "
        };
        let formatted = format!("{prefix}{line}");
        let prefix_len = prefix.len();
        let layout = VisualLayout::new(&formatted, width);
        for (row, start) in layout.rows().iter().zip(layout.row_start_offsets()) {
            rows.push(CachedRow {
                text: row.clone(),
                style,
            });
            starts.push(
                source_offset.saturating_add(start.saturating_sub(prefix_len).min(line.len())),
            );
        }
        source_offset = source_offset.saturating_add(line.len()).saturating_add(1);
    }
    (rows, starts)
}

fn append_content_section(
    mut old_rows: Vec<CachedRow>,
    mut old_starts: Vec<usize>,
    content: &str,
    style: Style,
    width: usize,
) -> (Vec<CachedRow>, Vec<usize>, usize) {
    let start = old_starts
        .last()
        .copied()
        .unwrap_or_default()
        .min(content.len());
    let keep = old_starts
        .iter()
        .position(|&offset| offset >= start)
        .unwrap_or(old_rows.len());
    old_rows.truncate(keep);
    old_starts.truncate(keep);
    let first_line_prefix = start == 0 || content.as_bytes().get(start - 1) == Some(&b'\n');
    let (suffix_rows, suffix_starts) =
        layout_content_section_from(&content[start..], style, width, first_line_prefix);
    old_rows.extend(suffix_rows);
    old_starts.extend(
        suffix_starts
            .into_iter()
            .map(|offset| start.saturating_add(offset)),
    );
    let bytes = content.len().saturating_sub(start);
    (old_rows, old_starts, bytes)
}

fn layout_styled_lines(text: &str, style: Style, width: usize) -> Vec<CachedRow> {
    VisualLayout::new(text, width)
        .rows()
        .iter()
        .cloned()
        .map(|row| CachedRow { text: row, style })
        .collect()
}

fn entry_render_bytes(entry: &TranscriptEntry) -> usize {
    match entry {
        TranscriptEntry::Message(message) => message.content.len().saturating_add(
            message
                .thinking
                .as_deref()
                .map(str::len)
                .unwrap_or_default(),
        ),
        TranscriptEntry::Tool(tool) => tool
            .summary
            .len()
            .saturating_add(tool.preview.as_deref().map(str::len).unwrap_or_default())
            .saturating_add(tool.output.len()),
    }
}

fn footer_text(state: &AppState, width: u16, scroll_from_bottom: usize) -> Line<'static> {
    let model = state
        .active_model()
        .map(ModelRef::display_name)
        .unwrap_or_else(|| "no model".to_owned());
    let thinking = state.thinking_level().map(|level| format!("think {level}"));
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
    let status = if state.last_error().is_some() {
        "error — see transcript"
    } else if state.is_busy() {
        "busy · Esc cancel"
    } else {
        "ready · Enter submit · Ctrl+C exit"
    };
    let thinking = state
        .thinking_level()
        .map(|level| format!(" · think {level}"))
        .unwrap_or_default();
    let text = if scroll_from_bottom > 0 {
        format!(
            "{status} · ↑ {scroll_from_bottom} lines · PgDn latest · {model}{thinking} · {context} · {session}"
        )
    } else {
        format!("{model}{thinking} · {context} · {session} · {status}")
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

#[cfg(test)]
mod tests {
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use ri_core::{AgentEvent, AppState, ModelMessage};

    use super::*;

    fn terminal() -> Terminal<TestBackend> {
        Terminal::new(TestBackend::new(20, 10)).expect("test terminal")
    }

    #[test]
    fn transcript_scroll_follows_growth_only_at_the_bottom() {
        let mut scroll = TranscriptScroll::default();
        scroll.update_maximum(100);
        assert_eq!(scroll.top_row, 100);
        assert_eq!(scroll.from_bottom(), 0);

        scroll.scroll_up(20);
        assert_eq!(scroll.top_row, 80);
        assert_eq!(scroll.from_bottom(), 20);
        scroll.update_maximum(130);
        assert_eq!(scroll.top_row, 80);
        assert_eq!(scroll.from_bottom(), 50);

        scroll.follow_bottom();
        assert_eq!(scroll.from_bottom(), 0);
        scroll.update_maximum(150);
        assert_eq!(scroll.top_row, 150);
        assert_eq!(scroll.from_bottom(), 0);
    }

    #[test]
    fn resize_clamping_to_the_bottom_resumes_follow_mode() {
        let mut scroll = TranscriptScroll::default();
        scroll.update_maximum(100);
        scroll.scroll_up(10);
        assert_eq!(scroll.top_row, 90);
        assert!(!scroll.following_bottom);

        scroll.update_maximum(80);
        assert_eq!(scroll.top_row, 80);
        assert_eq!(scroll.from_bottom(), 0);
        assert!(scroll.following_bottom);

        scroll.update_maximum(81);
        assert_eq!(scroll.top_row, 81);
        assert_eq!(scroll.from_bottom(), 0);
    }

    #[test]
    fn transcript_scroll_is_bounded_at_both_ends() {
        let mut scroll = TranscriptScroll::default();
        scroll.update_maximum(40);
        scroll.scroll_up(usize::MAX);
        assert_eq!(scroll.top_row, 0);
        assert_eq!(scroll.from_bottom(), 40);
        scroll.scroll_down(usize::MAX);
        assert_eq!(scroll.top_row, 40);
        assert_eq!(scroll.from_bottom(), 0);
    }

    #[test]
    fn unchanged_frames_reuse_entry_layouts_and_visit_only_the_viewport() {
        let state = synthetic_transcript(1_000, 100);
        let mut renderer = TuiRenderer::new();
        let mut terminal = terminal();

        renderer
            .draw(&mut terminal, &state, 0)
            .expect("first draw should succeed");
        assert!(renderer.stats().entries_reflowed > 0);
        let total_rows = renderer.transcript_total_rows();

        renderer
            .draw(&mut terminal, &state, 0)
            .expect("cached draw should succeed");
        assert_eq!(renderer.stats().entries_reflowed, 0);
        assert_eq!(renderer.stats().cache_misses, 0);
        assert!(renderer.stats().rows_visited < total_rows);
        assert_eq!(renderer.stats().rows_visited, renderer.stats().visible_rows);

        renderer
            .draw(&mut terminal, &state, total_rows)
            .expect("scroll draw should succeed");
        assert_eq!(renderer.stats().entries_reflowed, 0);
        assert_eq!(renderer.stats().cache_misses, 0);
        assert_eq!(renderer.stats().rows_visited, renderer.stats().visible_rows);
    }

    #[test]
    fn appending_one_entry_does_not_reflow_static_history() {
        let mut state = synthetic_transcript(1_000, 100);
        let mut renderer = TuiRenderer::new();
        let mut terminal = terminal();
        renderer
            .draw(&mut terminal, &state, 0)
            .expect("first draw should succeed");
        state.acknowledge_transcript_changes();

        state.add_system_message("new entry");
        renderer
            .draw(&mut terminal, &state, 0)
            .expect("append draw should succeed");
        assert_eq!(renderer.stats().entries_reflowed, 1);
        assert_eq!(renderer.stats().bytes_reflowed, "new entry".len());
    }

    #[test]
    fn streaming_incremental_layout_matches_cold_layout_for_unicode_chunks() {
        let mut state = AppState::new();
        state.reduce(AgentEvent::AssistantMessageStarted);
        let mut renderer = TuiRenderer::new();
        let mut terminal = terminal();
        let width = 18;
        let chunks = ["abcdef", "😀", "e\u{301}", "\n", "世界", " trailing text"];

        for chunk in chunks {
            append_streaming_delta(&mut state, chunk);
            renderer
                .draw(&mut terminal, &state, 0)
                .expect("stream draw should succeed");
            let streaming = state
                .streaming_assistant_state()
                .expect("stream should remain active");
            let cached = renderer
                .transcript
                .streaming_layout
                .as_ref()
                .expect("stream layout should be cached");
            let (expected_rows, expected_starts) =
                layout_content_section(&streaming.content, Style::default(), width);
            assert_eq!(
                cached
                    .content_rows
                    .iter()
                    .map(|row| row.text.as_str())
                    .collect::<Vec<_>>(),
                expected_rows
                    .iter()
                    .map(|row| row.text.as_str())
                    .collect::<Vec<_>>()
            );
            assert_eq!(cached.content_starts, expected_starts);
        }

        state.reduce(AgentEvent::AssistantThinkingDelta {
            item_id: None,
            text: "thinking 世界".to_owned(),
        });
        renderer
            .draw(&mut terminal, &state, 0)
            .expect("thinking draw should succeed");
        let streaming = state
            .streaming_assistant_state()
            .expect("stream should remain active");
        let cached = renderer
            .transcript
            .streaming_layout
            .as_ref()
            .expect("stream layout should be cached");
        let (expected_rows, _) = layout_content_section(
            &streaming.thinking,
            Style::default().fg(Color::DarkGray),
            width,
        );
        assert_eq!(
            cached
                .thinking_rows
                .iter()
                .map(|row| row.text.as_str())
                .collect::<Vec<_>>(),
            expected_rows
                .iter()
                .map(|row| row.text.as_str())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn transcripts_beyond_u16_rows_scroll_to_distinct_viewports() {
        let mut state = AppState::new();
        for index in 0..34_000 {
            state.add_system_message(format!("marker-{index:05}"));
        }
        let mut renderer = TuiRenderer::new();
        let mut terminal = terminal();
        renderer
            .draw(&mut terminal, &state, 0)
            .expect("initial draw should succeed");
        let maximum_scroll = renderer
            .transcript_total_rows()
            .saturating_sub(renderer.stats().visible_rows);

        renderer
            .draw(&mut terminal, &state, maximum_scroll)
            .expect("top draw should succeed");
        assert!(buffer_contains(terminal.backend().buffer(), "marker-00000"));

        renderer
            .draw(&mut terminal, &state, 0)
            .expect("bottom draw should succeed");
        assert!(buffer_contains(terminal.backend().buffer(), "marker-33999"));

        let middle_scroll = maximum_scroll / 2;
        renderer
            .draw(&mut terminal, &state, middle_scroll)
            .expect("middle draw should succeed");
        let expected_index = (maximum_scroll - middle_scroll) / 3;
        assert!(buffer_contains(
            terminal.backend().buffer(),
            &format!("marker-{expected_index:05}")
        ));
    }

    #[test]
    fn resize_reflows_once_and_releases_old_width_layouts() {
        let state = synthetic_transcript(10_000, 1_000);
        let mut renderer = TuiRenderer::new();
        let mut terminal = Terminal::new(TestBackend::new(80, 10)).expect("test terminal");

        renderer
            .draw(&mut terminal, &state, 0)
            .expect("width A draw should succeed");
        let entry_count = renderer.cached_transcript_entries();
        terminal.backend_mut().resize(120, 10);
        renderer
            .draw(&mut terminal, &state, 0)
            .expect("width B draw should succeed");
        assert_eq!(renderer.stats().entries_reflowed, entry_count);
        assert_eq!(renderer.cached_transcript_entries(), entry_count);
        renderer
            .draw(&mut terminal, &state, 0)
            .expect("warm width B draw should succeed");
        assert_eq!(renderer.stats().entries_reflowed, 0);

        terminal.backend_mut().resize(60, 10);
        renderer
            .draw(&mut terminal, &state, 0)
            .expect("width C draw should succeed");
        assert_eq!(renderer.cached_transcript_entries(), entry_count);
    }

    #[test]
    fn resize_during_streaming_cold_reflows_once_then_returns_to_incremental() {
        let mut state = synthetic_transcript(1_000, 100);
        state.reduce(AgentEvent::AssistantMessageStarted);
        append_streaming_delta(&mut state, &"x".repeat(500));
        let mut renderer = TuiRenderer::new();
        let mut terminal = Terminal::new(TestBackend::new(80, 10)).expect("test terminal");
        renderer
            .draw(&mut terminal, &state, 0)
            .expect("initial draw should succeed");
        let static_entries = renderer.cached_transcript_entries().saturating_sub(1);
        state.acknowledge_transcript_changes();

        terminal.backend_mut().resize(120, 10);
        renderer
            .draw(&mut terminal, &state, 0)
            .expect("resize draw should succeed");
        assert_eq!(renderer.stats().entries_reflowed, static_entries + 1);
        state.acknowledge_transcript_changes();

        append_streaming_delta(&mut state, " + delta");
        renderer
            .draw(&mut terminal, &state, 0)
            .expect("post-resize stream draw should succeed");
        assert_eq!(renderer.stats().entries_reflowed, 1);
        assert!(renderer.stats().bytes_reflowed < 500);
    }

    #[test]
    fn session_replacement_discards_old_cached_rows() {
        let mut state = synthetic_transcript(10_000, 1_000);
        let mut renderer = TuiRenderer::new();
        let mut terminal = terminal();
        renderer
            .draw(&mut terminal, &state, 0)
            .expect("large draw should succeed");
        assert!(renderer.cached_transcript_entries() > 100);
        let old_capacity = renderer.transcript.layouts.capacity();

        state.replace_history(&[ModelMessage::user("small replacement")]);
        renderer
            .draw(&mut terminal, &state, 0)
            .expect("replacement draw should succeed");
        assert_eq!(renderer.cached_transcript_entries(), 1);
        assert!(renderer.cached_transcript_rows() < 10);
        assert!(renderer.transcript.layouts.capacity() < old_capacity);
    }

    #[test]
    fn command_suggestions_render_safely_in_a_narrow_terminal() {
        let mut state = AppState::new();
        state.insert_text("/");
        let suggestions = CommandSuggestions::default();
        let mut scroll = TranscriptScroll::default();
        let mut renderer = TuiRenderer::new();
        let mut terminal = Terminal::new(TestBackend::new(8, 5)).expect("test terminal");

        renderer
            .draw_interactive(&mut terminal, &state, &mut scroll, &suggestions)
            .expect("narrow suggestion draw should succeed");
    }

    #[test]
    fn editor_layout_reuses_wrapping_for_cursor_motion() {
        let mut state = AppState::new();
        state.insert_text(&"line\n".repeat(200));
        let mut renderer = TuiRenderer::new();
        let mut terminal = terminal();
        renderer
            .draw(&mut terminal, &state, 0)
            .expect("initial editor draw should succeed");
        state.move_left();
        renderer
            .draw(&mut terminal, &state, 0)
            .expect("cursor draw should succeed");
        assert_eq!(renderer.stats().editor_cache_misses, 0);
        assert!(renderer.stats().editor_cache_hits > 0);
        state.insert_text("x");
        renderer
            .draw(&mut terminal, &state, 0)
            .expect("edited editor draw should succeed");
        assert_eq!(renderer.stats().editor_cache_misses, 1);
    }

    #[test]
    fn tool_layout_renders_cached_semantic_presentation_and_result() {
        let cases = [
            (
                "read",
                r#"{"path":"src/foo.rs","offset":10,"limit":20}"#,
                "✓ read src/foo.rs · lines 10–29",
                "",
            ),
            (
                "write",
                r#"{"path":"src/foo.rs","content":"one\ntwo"}"#,
                "✓ write src/foo.rs",
                "one\ntwo",
            ),
            (
                "edit",
                r#"{"path":"src/foo.rs","old_text":"old","new_text":"new"}"#,
                "✓ edit src/foo.rs",
                "-old\n+new",
            ),
            (
                "bash",
                r#"{"command":"cargo test -p ri-core"}"#,
                "✓ bash",
                "cargo test -p ri-core",
            ),
        ];

        for (name, arguments, expected_title, expected_preview) in cases {
            let mut state = AppState::new();
            state.reduce(AgentEvent::ToolExecutionStarted {
                call_id: name.to_owned(),
                name: name.to_owned(),
                arguments: arguments.to_owned(),
            });
            state.reduce(AgentEvent::ToolExecutionFinished {
                call_id: name.to_owned(),
                name: name.to_owned(),
                result: ri_core::ToolExecutionResult::success("tool result"),
            });
            let rendered = layout_entry(&state.transcript_entries()[0].entry, 200)
                .into_iter()
                .map(|row| row.text)
                .collect::<Vec<_>>()
                .join("\n");

            assert!(rendered.contains(expected_title), "{rendered}");
            assert!(
                expected_preview
                    .lines()
                    .all(|line| rendered.contains(&format!("  {line}"))),
                "{rendered}"
            );
            assert!(rendered.contains("tool result"), "{rendered}");
            assert!(!rendered.contains("{\""), "{rendered}");
        }
    }

    #[test]
    fn live_tool_output_reflows_only_the_bounded_tool_entry() {
        let mut state = synthetic_transcript(1_000, 100);
        let mut renderer = TuiRenderer::new();
        let mut terminal = terminal();
        renderer
            .draw(&mut terminal, &state, 0)
            .expect("history draw should succeed");
        state.acknowledge_transcript_changes();

        state.reduce(AgentEvent::ToolExecutionStarted {
            call_id: "live-tool".to_owned(),
            name: "bash".to_owned(),
            arguments: "{}".to_owned(),
        });
        renderer
            .draw(&mut terminal, &state, 0)
            .expect("tool start draw should succeed");
        state.acknowledge_transcript_changes();

        state.reduce(AgentEvent::ToolExecutionOutput {
            call_id: "live-tool".to_owned(),
            stream: ri_core::ToolOutputStream::Stdout,
            chunk: "new tool output\n".to_owned(),
        });
        renderer
            .draw(&mut terminal, &state, 0)
            .expect("tool output draw should succeed");
        assert_eq!(renderer.stats().entries_reflowed, 1);
        assert!(renderer.stats().bytes_reflowed >= "new tool output\n".len());
    }

    #[test]
    fn incremental_streaming_rows_match_cold_layout_across_chunkings_and_widths() {
        let text = "ab😀e\u{301}\n世界xyz";
        for width in [1, 2, 3, 4, 8, 18] {
            let mut state = AppState::new();
            state.reduce(AgentEvent::AssistantMessageStarted);
            let mut renderer = TuiRenderer::new();
            let mut terminal =
                Terminal::new(TestBackend::new(width + 2, 10)).expect("test terminal");
            for character in text.chars() {
                append_streaming_delta(&mut state, &character.to_string());
                renderer
                    .draw(&mut terminal, &state, 0)
                    .expect("stream draw should succeed");
                let streaming = state
                    .streaming_assistant_state()
                    .expect("stream should remain active");
                let cached = renderer
                    .transcript
                    .streaming_layout
                    .as_ref()
                    .expect("stream layout should be cached");
                let (expected_rows, expected_starts) =
                    layout_content_section(&streaming.content, Style::default(), width as usize);
                assert_eq!(
                    cached
                        .content_rows
                        .iter()
                        .map(|row| row.text.as_str())
                        .collect::<Vec<_>>(),
                    expected_rows
                        .iter()
                        .map(|row| row.text.as_str())
                        .collect::<Vec<_>>()
                );
                assert_eq!(cached.content_starts, expected_starts);
            }
        }
    }

    #[test]
    fn streaming_append_reflows_only_the_active_entry_suffix() {
        let mut state = synthetic_transcript(1_000, 100);
        state.reduce(AgentEvent::AssistantMessageStarted);
        append_streaming_delta(&mut state, &"x".repeat(500));
        let mut renderer = TuiRenderer::new();
        let mut terminal = terminal();
        renderer
            .draw(&mut terminal, &state, 0)
            .expect("initial stream draw should succeed");
        state.acknowledge_transcript_changes();

        append_streaming_delta(&mut state, " + delta");
        renderer
            .draw(&mut terminal, &state, 0)
            .expect("append stream draw should succeed");
        assert_eq!(renderer.stats().entries_reflowed, 1);
        assert!(renderer.stats().bytes_reflowed < 500);
    }

    #[test]
    fn scroll_indicator_only_appears_away_from_the_bottom() {
        let state = AppState::new();
        let at_bottom = footer_text(&state, 200, 0)
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();
        let scrolled = footer_text(&state, 200, 42)
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();

        assert!(!at_bottom.contains("PgDn latest"));
        assert!(scrolled.contains("↑ 42 lines · PgDn latest"));
    }

    #[test]
    fn footer_reports_errors_without_embedding_the_provider_message() {
        let mut state = AppState::new();
        let provider_message = "provider returned HTTP 400: a very detailed response body";
        state.reduce(AgentEvent::Error(ri_core::AgentError::new(
            provider_message,
        )));

        let footer = footer_text(&state, 200, 0);
        let text = footer
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();

        assert!(text.contains("error — see transcript"));
        assert!(!text.contains(provider_message));
        assert!(text.len() < 120);
    }

    fn buffer_contains(buffer: &Buffer, needle: &str) -> bool {
        (0..buffer.area.height).any(|y| {
            (0..buffer.area.width)
                .map(|x| buffer[(x, y)].symbol())
                .collect::<String>()
                .contains(needle)
        })
    }
}
