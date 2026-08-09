use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

const TAB_STOP: usize = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CursorPosition {
    pub row: usize,
    pub column: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CursorBoundary {
    source: usize,
    column: usize,
}

pub(crate) struct VisualLayout {
    rows: Vec<String>,
    row_boundaries: Vec<Vec<CursorBoundary>>,
    positions: Vec<CursorPosition>,
    width: usize,
}

impl VisualLayout {
    pub(crate) fn new(text: &str, width: usize) -> Self {
        let width = width.max(1);
        let graphemes: Vec<(usize, &str)> = text.grapheme_indices(true).collect();
        let mut rows = Vec::new();
        let mut row_boundaries = vec![Vec::new()];
        let mut positions = vec![CursorPosition::default(); text.len() + 1];
        let mut current = String::new();
        let mut current_width = 0;
        let mut needs_final_empty_row = false;
        let mut row = 0;

        record_position(
            &mut positions,
            &mut row_boundaries,
            0,
            CursorPosition { row, column: 0 },
            width,
        );

        for (index, (source_start, grapheme)) in graphemes.iter().enumerate() {
            let source_start = *source_start;
            let source_end = graphemes
                .get(index + 1)
                .map(|(offset, _)| *offset)
                .unwrap_or(text.len());

            if *grapheme == "\n" {
                needs_final_empty_row = true;
                record_position(
                    &mut positions,
                    &mut row_boundaries,
                    source_start,
                    CursorPosition {
                        row,
                        column: current_width,
                    },
                    width,
                );
                rows.push(std::mem::take(&mut current));
                row += 1;
                row_boundaries.push(Vec::new());
                current_width = 0;
                record_position(
                    &mut positions,
                    &mut row_boundaries,
                    source_end,
                    CursorPosition { row, column: 0 },
                    width,
                );
                continue;
            }

            needs_final_empty_row = false;
            let (mut rendered, mut grapheme_width) = expand_grapheme(grapheme, current_width);
            if !current.is_empty() && current_width + grapheme_width > width {
                rows.push(std::mem::take(&mut current));
                row += 1;
                row_boundaries.push(Vec::new());
                current_width = 0;
                (rendered, grapheme_width) = expand_grapheme(grapheme, current_width);
            }

            record_position(
                &mut positions,
                &mut row_boundaries,
                source_start,
                CursorPosition {
                    row,
                    column: current_width,
                },
                width,
            );
            current.push_str(&rendered);
            current_width += grapheme_width;

            let next_is_newline =
                matches!(graphemes.get(index + 1).map(|(_, next)| *next), Some("\n"));
            if current_width >= width && !next_is_newline {
                rows.push(std::mem::take(&mut current));
                row += 1;
                row_boundaries.push(Vec::new());
                current_width = 0;
                record_position(
                    &mut positions,
                    &mut row_boundaries,
                    source_end,
                    CursorPosition { row, column: 0 },
                    width,
                );
            } else {
                record_position(
                    &mut positions,
                    &mut row_boundaries,
                    source_end,
                    CursorPosition {
                        row,
                        column: current_width,
                    },
                    width,
                );
            }
        }

        if !current.is_empty() || rows.is_empty() || needs_final_empty_row {
            rows.push(current);
        }

        Self {
            rows,
            row_boundaries,
            positions,
            width,
        }
    }

    pub(crate) fn rows(&self) -> &[String] {
        &self.rows
    }

    pub(crate) fn row_count(&self) -> usize {
        self.rows.len()
    }

    pub(crate) fn row_start_offsets(&self) -> Vec<usize> {
        self.row_boundaries
            .iter()
            .map(|boundaries| {
                boundaries
                    .first()
                    .map(|boundary| boundary.source)
                    .unwrap_or_default()
            })
            .collect()
    }

    pub(crate) fn cursor_position(&self, cursor: usize) -> CursorPosition {
        self.positions
            .get(cursor.min(self.positions.len().saturating_sub(1)))
            .copied()
            .unwrap_or_default()
    }

    pub(crate) fn move_vertical(
        &self,
        cursor: usize,
        direction: isize,
        preferred_column: Option<usize>,
    ) -> Option<(usize, usize)> {
        let current = self.cursor_position(cursor);
        let target_row = current.row as isize + direction;
        if target_row < 0 || target_row as usize >= self.row_boundaries.len() {
            return None;
        }

        let desired_column = preferred_column.unwrap_or(current.column);
        let boundary = self.row_boundaries[target_row as usize]
            .iter()
            .min_by_key(|boundary| (boundary.column.abs_diff(desired_column), boundary.source))?;
        Some((
            boundary.source,
            desired_column.min(self.width.saturating_sub(1)),
        ))
    }
}

fn record_position(
    positions: &mut [CursorPosition],
    row_boundaries: &mut [Vec<CursorBoundary>],
    source: usize,
    position: CursorPosition,
    width: usize,
) {
    let position = CursorPosition {
        row: position.row,
        column: position.column.min(width.saturating_sub(1)),
    };
    if let Some(slot) = positions.get_mut(source) {
        *slot = position;
    }
    if let Some(boundaries) = row_boundaries.get_mut(position.row) {
        if let Some(existing) = boundaries
            .iter_mut()
            .find(|boundary| boundary.source == source)
        {
            existing.column = position.column;
        } else {
            boundaries.push(CursorBoundary {
                source,
                column: position.column,
            });
        }
    }
}

fn expand_grapheme(grapheme: &str, column: usize) -> (String, usize) {
    if grapheme == "\t" {
        let width = TAB_STOP - (column % TAB_STOP);
        (" ".repeat(width), width)
    } else {
        (grapheme.to_owned(), UnicodeWidthStr::width(grapheme))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertical_movement_follows_soft_wrapped_rows() {
        let text = "abcdefghij";
        let layout = VisualLayout::new(text, 4);

        assert_eq!(
            layout.cursor_position(text.len()),
            CursorPosition { row: 2, column: 2 }
        );
        let (cursor, preferred) = layout
            .move_vertical(text.len(), -1, None)
            .expect("there should be a row above");
        assert_eq!(&text[..cursor], "abcdef");
        assert_eq!(preferred, 2);

        let (cursor, _) = layout
            .move_vertical(cursor, 1, Some(preferred))
            .expect("there should be a row below");
        assert_eq!(cursor, text.len());
    }

    #[test]
    fn grapheme_clusters_are_measured_and_wrapped_as_one_unit() {
        let family = "👨‍👩‍👧‍👦";
        let family_layout = VisualLayout::new(&format!("{family}x"), 2);
        assert_eq!(family_layout.rows(), &[family.to_owned(), "x".to_owned()]);

        let modifier = "👍🏽";
        let modifier_layout = VisualLayout::new(&format!("{modifier}x"), 2);
        assert_eq!(
            modifier_layout.rows(),
            &[modifier.to_owned(), "x".to_owned()]
        );
    }

    #[test]
    fn full_width_lines_do_not_add_transcript_rows() {
        let layout = VisualLayout::new("abcd", 4);
        assert_eq!(layout.rows(), &["abcd".to_owned()]);
    }

    #[test]
    fn tabs_expand_to_the_next_tab_stop() {
        let layout = VisualLayout::new("ab\tc", 4);
        assert_eq!(layout.rows(), &["ab  ".to_owned(), "c".to_owned()]);
        assert_eq!(
            layout.cursor_position(3),
            CursorPosition { row: 1, column: 0 }
        );
    }
}
