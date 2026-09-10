//! Width-independent fenced-code styling; Markdown still owns layout and sanitization.
use std::sync::LazyLock;

use ratatui::style::{Color, Modifier, Style};
use two_face::re_exports::syntect::{
    easy::HighlightLines,
    highlighting::{self, FontStyle, StyleModifier, Theme, ThemeItem},
    parsing::{SyntaxReference, SyntaxSet},
    util::LinesWithEndings,
};

static SYNTAXES: LazyLock<SyntaxSet> = LazyLock::new(two_face::syntax::extra_newlines);
static THEME: LazyLock<Theme> = LazyLock::new(|| {
    let mut theme = Theme::default();
    theme.settings.foreground = Some(rgb(220, 220, 220));
    for (scope, color, font_style) in [
        ("comment", rgb(130, 140, 150), FontStyle::ITALIC),
        ("string", rgb(150, 200, 140), FontStyle::empty()),
        ("keyword, storage", rgb(190, 150, 220), FontStyle::empty()),
        ("constant", rgb(225, 180, 120), FontStyle::empty()),
        (
            "entity.name.function",
            rgb(120, 190, 220),
            FontStyle::empty(),
        ),
        (
            "entity.name.type, entity.other.inherited-class",
            rgb(110, 200, 190),
            FontStyle::empty(),
        ),
        ("support", rgb(130, 180, 220), FontStyle::empty()),
        ("variable", rgb(215, 200, 170), FontStyle::empty()),
        (
            "punctuation, keyword.operator",
            rgb(175, 185, 195),
            FontStyle::empty(),
        ),
        ("markup.inserted", rgb(150, 200, 140), FontStyle::empty()),
        ("markup.deleted", rgb(225, 140, 140), FontStyle::empty()),
    ] {
        theme.scopes.push(ThemeItem {
            scope: scope.parse().expect("ri syntax scope"),
            style: StyleModifier {
                foreground: Some(color),
                font_style: Some(font_style),
                ..StyleModifier::default()
            },
        });
    }
    theme
});

fn rgb(r: u8, g: u8, b: u8) -> highlighting::Color {
    highlighting::Color { r, g, b, a: 255 }
}

fn resolve(info: &str) -> Option<&'static SyntaxReference> {
    let token = info
        .split_whitespace()
        .next()?
        .split(',')
        .next()?
        .to_ascii_lowercase();
    if matches!(
        token.as_str(),
        "" | "text" | "txt" | "plain" | "plaintext" | "none"
    ) {
        return None;
    }
    SYNTAXES.find_syntax_by_token(&token).or_else(|| {
        let alias = match token.as_str() {
            "shell" | "zsh" | "sh" => "bash",
            "c++" => "cpp",
            "cs" => "C#",
            _ => return None,
        };
        SYNTAXES.find_syntax_by_token(alias)
    })
}

fn terminal_style(style: highlighting::Style) -> Style {
    let mut result = Style::default().fg(Color::Rgb(
        style.foreground.r,
        style.foreground.g,
        style.foreground.b,
    ));
    for (syntax, terminal) in [
        (FontStyle::BOLD, Modifier::BOLD),
        (FontStyle::ITALIC, Modifier::ITALIC),
        (FontStyle::UNDERLINE, Modifier::UNDERLINED),
    ] {
        if style.font_style.contains(syntax) {
            result = result.add_modifier(terminal);
        }
    }
    result
}

// None asks the caller to preserve its existing plain code style for the whole block.
pub(super) fn highlight_code<'a>(
    info: &str,
    source: &'a str,
) -> Option<Vec<Vec<(&'a str, Style)>>> {
    let mut highlighter = HighlightLines::new(resolve(info)?, &THEME);
    LinesWithEndings::from(source)
        .map(|line| {
            Some(
                highlighter
                    .highlight_line(line, &SYNTAXES)
                    .ok()?
                    .into_iter()
                    .map(|(style, text)| (text.trim_end_matches('\n'), terminal_style(style)))
                    .collect(),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fence_resolution_and_bundle_coverage() {
        for token in [
            "rust",
            "RUST",
            "rs",
            " rust,ignore extra",
            "rust,no_run",
            "python",
            "py",
            "javascript",
            "js",
            "typescript",
            "ts",
            "tsx",
            "bash",
            "sh",
            "shell",
            "zsh",
            "json",
            "yaml",
            "yml",
            "toml",
            "html",
            "css",
            "sql",
            "c",
            "cpp",
            "c++",
            "cs",
            "go",
            "java",
            "dockerfile",
            "diff",
            "terraform",
            "zig",
            "kotlin",
            "nix",
            "md",
        ] {
            assert!(resolve(token).is_some(), "{token}");
        }
        for token in [
            "",
            "  ",
            "text",
            "txt",
            "plain",
            "plaintext",
            "none",
            "TEXT",
            "unknown-made-up-language",
        ] {
            assert!(resolve(token).is_none(), "{token}");
            assert!(highlight_code(token, "literal **code**").is_none());
        }
    }

    fn style_at(lines: &[Vec<(&str, Style)>], needle: &str) -> Style {
        lines
            .iter()
            .flatten()
            .find(|(text, _)| text.contains(needle))
            .unwrap()
            .1
    }

    #[test]
    fn semantic_styles_preserve_source_and_multiline_state() {
        for (language, source) in [
            (
                "rust",
                "// comment\nfn main() {\n    let value = \"hello\";\n}\n",
            ),
            ("ts", "const value: string = \"hello\";\n"),
            ("toml", "[package]\nname = \"ri\"\nversion = \"0.1.0\"\n"),
            ("diff", "-old\n+new\n"),
        ] {
            let lines = highlight_code(language, source).unwrap();
            let text = lines
                .iter()
                .map(|line| line.iter().map(|(s, _)| *s).collect::<String>())
                .collect::<Vec<_>>()
                .join("\n");
            assert_eq!(text, source.trim_end_matches('\n'));
            let styles: std::collections::HashSet<_> =
                lines.iter().flatten().map(|(_, style)| *style).collect();
            assert!(styles.len() >= 2, "{language}");
            assert!(styles.iter().all(|style| style.bg.is_none()));
        }
        let lines = highlight_code(
            "rust",
            "/* first\nsecond\n*/\nfn main() { let x = \"hello\"; }",
        )
        .unwrap();
        assert_eq!(style_at(&lines, "first"), style_at(&lines, "second"));
        assert_ne!(style_at(&lines, "second"), style_at(&lines, "fn"));
        assert_ne!(style_at(&lines, "fn"), style_at(&lines, "hello"));
        assert_ne!(style_at(&lines, "second"), style_at(&lines, "hello"));
        let lines = highlight_code("python", "x = \"\"\"first\nsecond\n\"\"\"").unwrap();
        assert_eq!(style_at(&lines, "first"), style_at(&lines, "second"));
    }

    #[test]
    fn conversion_ignores_background_and_maps_modifiers() {
        let style = terminal_style(highlighting::Style {
            foreground: rgb(1, 2, 3),
            background: rgb(4, 5, 6),
            font_style: FontStyle::BOLD | FontStyle::ITALIC | FontStyle::UNDERLINE,
        });
        assert_eq!(style.fg, Some(Color::Rgb(1, 2, 3)));
        assert_eq!(style.bg, None);
        assert_eq!(
            style.add_modifier,
            Modifier::BOLD | Modifier::ITALIC | Modifier::UNDERLINED
        );
    }
}
