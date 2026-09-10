use std::hint::black_box;
use std::time::{Duration, Instant};

use ratatui::backend::TestBackend;
use ratatui::Terminal;
use ri::render::{self, TuiRenderer};
use ri_core::{AgentEvent, ToolOutputStream};

const WIDTH: u16 = 100;
const HEIGHT: u16 = 28;

fn main() {
    syntax_workloads();
    markdown_workloads();
    let fresh_state = render::synthetic_transcript(20, 4);
    measure("fresh first frame", 10, || {
        let mut terminal = test_terminal(WIDTH, HEIGHT);
        let mut renderer = TuiRenderer::new();
        renderer
            .draw(&mut terminal, &fresh_state, 0)
            .expect("test backend draw");
    });

    for (rows, entries) in [
        (1_000, 100),
        (10_000, 1_000),
        (100_000, 10_000),
        (100_000, 100),
    ] {
        let state = render::synthetic_transcript(rows, entries);
        let label = if entries == 100 {
            format!("cold layout · {rows} rows · 100 entries")
        } else {
            format!("cold layout · {rows} rows")
        };
        measure(&label, 1, || {
            let mut terminal = test_terminal(WIDTH, HEIGHT);
            let mut renderer = TuiRenderer::new();
            renderer
                .draw(&mut terminal, &state, 0)
                .expect("test backend draw");
        });
    }

    let cached_state = render::synthetic_transcript(100_000, 10_000);
    let mut cached_terminal = test_terminal(WIDTH, HEIGHT);
    let mut cached_renderer = TuiRenderer::new();
    cached_renderer
        .draw(&mut cached_terminal, &cached_state, 0)
        .expect("warmup draw");
    measure("cached redraw · 100k rows · collapsed", 10, || {
        cached_renderer
            .draw(&mut cached_terminal, &cached_state, 0)
            .expect("test backend draw");
    });

    let mut expanded_terminal = test_terminal(WIDTH, HEIGHT);
    let mut expanded_renderer = TuiRenderer::new();
    expanded_renderer.toggle_tool_output();
    expanded_renderer
        .draw(&mut expanded_terminal, &cached_state, 0)
        .expect("expanded warmup draw");
    measure("cached redraw · 100k rows · expanded", 10, || {
        expanded_renderer
            .draw(&mut expanded_terminal, &cached_state, 0)
            .expect("test backend draw");
    });

    measure("scroll · 100k rows", 10, || {
        cached_renderer
            .draw(&mut cached_terminal, &cached_state, 20_000)
            .expect("test backend draw");
    });

    let mut streaming_state = render::synthetic_transcript(100_000, 10_000);
    streaming_state.reduce(AgentEvent::AssistantMessageStarted);
    render::append_streaming_delta(&mut streaming_state, "initial streaming response");
    let mut streaming_terminal = test_terminal(WIDTH, HEIGHT);
    let mut streaming_renderer = TuiRenderer::new();
    streaming_renderer
        .draw(&mut streaming_terminal, &streaming_state, 0)
        .expect("warmup draw");
    measure("single streaming append · 100k rows", 10, || {
        render::append_streaming_delta(&mut streaming_state, " + delta");
        streaming_renderer
            .draw(&mut streaming_terminal, &streaming_state, 0)
            .expect("test backend draw");
    });

    let mut burst_state = render::synthetic_transcript(100_000, 10_000);
    burst_state.reduce(AgentEvent::AssistantMessageStarted);
    let mut burst_terminal = test_terminal(WIDTH, HEIGHT);
    let mut burst_renderer = TuiRenderer::new();
    burst_renderer
        .draw(&mut burst_terminal, &burst_state, 0)
        .expect("warmup draw");
    measure("1,000 streaming deltas · 100k rows", 1, || {
        for _ in 0..1_000 {
            render::append_streaming_delta(&mut burst_state, " token");
            burst_renderer
                .draw(&mut burst_terminal, &burst_state, 0)
                .expect("test backend draw");
        }
    });

    let resize_state = render::synthetic_transcript(100_000, 10_000);
    let mut resize_terminal = test_terminal(WIDTH, HEIGHT);
    let mut resize_renderer = TuiRenderer::new();
    resize_renderer
        .draw(&mut resize_terminal, &resize_state, 0)
        .expect("warmup draw");
    measure("resize · 100k rows", 3, || {
        resize_terminal.backend_mut().resize(80, HEIGHT);
        resize_renderer
            .draw(&mut resize_terminal, &resize_state, 0)
            .expect("width 80 draw");
        resize_terminal.backend_mut().resize(WIDTH, HEIGHT);
        resize_renderer
            .draw(&mut resize_terminal, &resize_state, 0)
            .expect("width 100 draw");
    });

    let mut tool_state = render::synthetic_transcript(10_000, 1_000);
    tool_state.reduce(AgentEvent::ToolExecutionStarted {
        call_id: "live-tool".to_owned(),
        name: "bash".to_owned(),
        arguments: "{}".to_owned(),
    });
    let mut tool_terminal = test_terminal(WIDTH, HEIGHT);
    let mut tool_renderer = TuiRenderer::new();
    tool_renderer
        .draw(&mut tool_terminal, &tool_state, 0)
        .expect("warmup draw");
    measure("large live tool-output burst · collapsed", 3, || {
        for index in 0..100 {
            tool_state.reduce(AgentEvent::ToolExecutionOutput {
                call_id: "live-tool".to_owned(),
                stream: ToolOutputStream::Stdout,
                chunk: format!("output {index}\n"),
            });
            tool_renderer
                .draw(&mut tool_terminal, &tool_state, 0)
                .expect("test backend draw");
        }
    });

    tool_renderer.toggle_tool_output();
    tool_renderer
        .draw(&mut tool_terminal, &tool_state, 0)
        .expect("expanded tool warmup draw");
    measure("large live tool-output burst · expanded", 3, || {
        for index in 100..200 {
            tool_state.reduce(AgentEvent::ToolExecutionOutput {
                call_id: "live-tool".to_owned(),
                stream: ToolOutputStream::Stdout,
                chunk: format!("output {index}\n"),
            });
            tool_renderer
                .draw(&mut tool_terminal, &tool_state, 0)
                .expect("test backend draw");
        }
    });
}

fn test_terminal(width: u16, height: u16) -> Terminal<TestBackend> {
    Terminal::new(TestBackend::new(width, height)).expect("test terminal")
}

fn measure<F>(label: &str, iterations: usize, mut operation: F)
where
    F: FnMut(),
{
    let iterations = iterations.max(1);
    let started = Instant::now();
    for _ in 0..iterations {
        operation();
        black_box(());
    }
    let elapsed = started.elapsed();
    println!(
        "{label:40} total={:>10} avg={:>10}",
        format_duration(elapsed),
        format_duration(elapsed / iterations as u32)
    );
}

fn format_duration(duration: Duration) -> String {
    if duration.as_secs() > 0 {
        format!("{:.3}s", duration.as_secs_f64())
    } else {
        format!("{}µs", duration.as_micros())
    }
}

fn markdown_workloads() {
    let history = render::markdown_transcript(100);
    measure("Markdown mixed history · cold · 100 entries", 3, || {
        let mut terminal = test_terminal(WIDTH, HEIGHT);
        let mut renderer = TuiRenderer::new();
        renderer.draw(&mut terminal, &history, 0).unwrap();
        assert_eq!(renderer.stats().entries_reflowed, 100);
    });
    let mut terminal = test_terminal(WIDTH, HEIGHT);
    let mut renderer = TuiRenderer::new();
    renderer.draw(&mut terminal, &history, 0).unwrap();
    for scroll in [0, 300] {
        measure(
            &format!("Markdown history · cached scroll={scroll}"),
            10,
            || {
                renderer.draw(&mut terminal, &history, scroll).unwrap();
                assert_eq!(renderer.stats().bytes_reflowed, 0);
                assert_eq!(renderer.stats().cache_misses, 0);
            },
        );
    }
    measure("Markdown history · resize", 3, || {
        for width in [80, WIDTH] {
            terminal.backend_mut().resize(width, HEIGHT);
            renderer.draw(&mut terminal, &history, 0).unwrap();
            assert_eq!(renderer.stats().entries_reflowed, 100);
            renderer.draw(&mut terminal, &history, 0).unwrap();
            assert_eq!(renderer.stats().bytes_reflowed, 0);
        }
    });
    let mut state = history;
    state.acknowledge_transcript_changes();
    state.reduce(AgentEvent::AssistantMessageStarted);
    let mut size = 0;
    for target in [1_024usize, 8_192, 32_768, 65_536] {
        let source = render::MARKDOWN_REPORT.repeat(target.div_ceil(render::MARKDOWN_REPORT.len()));
        let mut end = target - size;
        while !source.is_char_boundary(end) {
            end += 1;
        }
        render::append_streaming_delta(&mut state, &source[..end]);
        size += end;
        measure(&format!("Markdown stream · {target} bytes"), 1, || {
            renderer.draw(&mut terminal, &state, 0).unwrap();
            assert_eq!(renderer.stats().entries_reflowed, 1);
            assert_eq!(renderer.stats().bytes_reflowed, size);
        });
        renderer.draw(&mut terminal, &state, 0).unwrap();
        assert_eq!(renderer.stats().bytes_reflowed, 0);
    }
}

fn syntax_workloads() {
    let snippet = "// A representative Rust function\nfn greeting(name: &str) -> String {\n    let prefix = \"hello\";\n    format!(\"{prefix}, {name}!\")\n}\n";
    let source = format!(
        "```rust\n{}",
        snippet.repeat(65_536usize.div_ceil(snippet.len()))
    );
    let mut state = ri_core::AppState::new();
    state.reduce(AgentEvent::AssistantMessageStarted);
    render::append_streaming_delta(&mut state, &source[..1_024]);
    measure("Rust first highlight · 1 KiB · lazy setup", 1, || {
        let mut terminal = test_terminal(WIDTH, HEIGHT);
        TuiRenderer::new().draw(&mut terminal, &state, 0).unwrap();
    });

    for size in [8_192, 32_768, 65_536] {
        let mut history = ri_core::AppState::new();
        history.reduce(AgentEvent::AssistantMessageStarted);
        render::append_streaming_delta(&mut history, &format!("{}\n```", &source[..size]));
        history.reduce(AgentEvent::AssistantMessageFinished { items: Vec::new() });
        measure(&format!("Rust completed · {size} bytes"), 3, || {
            let mut terminal = test_terminal(WIDTH, HEIGHT);
            let mut renderer = TuiRenderer::new();
            renderer.draw(&mut terminal, &history, 0).unwrap();
            assert_eq!(renderer.stats().entries_reflowed, 1);
        });
        let mut terminal = test_terminal(WIDTH, HEIGHT);
        let mut renderer = TuiRenderer::new();
        renderer.draw(&mut terminal, &history, 0).unwrap();
        history.acknowledge_transcript_changes();
        for scroll in [0, 100] {
            measure(
                &format!("Rust cached · {size} · scroll={scroll}"),
                10,
                || {
                    renderer.draw(&mut terminal, &history, scroll).unwrap();
                    assert_eq!(renderer.stats().bytes_reflowed, 0);
                    assert_eq!(renderer.stats().cache_misses, 0);
                },
            );
        }
        history.reduce(AgentEvent::AssistantMessageStarted);
        let mut previous = 0;
        for target in [1_024, 8_192, 32_768, 65_536] {
            render::append_streaming_delta(&mut history, &source[previous..target]);
            previous = target;
            measure(
                &format!("Rust stream · {target} · history={size}"),
                1,
                || {
                    renderer.draw(&mut terminal, &history, 0).unwrap();
                    assert_eq!(renderer.stats().entries_reflowed, 1);
                    assert_eq!(renderer.stats().bytes_reflowed, target);
                },
            );
            renderer.draw(&mut terminal, &history, 0).unwrap();
            assert_eq!(renderer.stats().bytes_reflowed, 0);
        }
    }

    let mut prose = ri_core::AppState::new();
    prose.reduce(AgentEvent::AssistantMessageStarted);
    render::append_streaming_delta(&mut prose, &"## Report\n\nSome **bold** prose and `inline code`.\n\n- First change\n- Second change\n\n".repeat(800));
    measure("Markdown prose/lists · ~64 KiB", 3, || {
        let mut terminal = test_terminal(WIDTH, HEIGHT);
        TuiRenderer::new().draw(&mut terminal, &prose, 0).unwrap();
    });
}
