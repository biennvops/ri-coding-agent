use std::hint::black_box;
use std::time::{Duration, Instant};

use ratatui::backend::TestBackend;
use ratatui::Terminal;
use ri::render::{self, TuiRenderer};
use ri_core::{AgentEvent, ToolOutputStream};

const WIDTH: u16 = 100;
const HEIGHT: u16 = 28;

fn main() {
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
    measure("cached redraw · 100k rows", 10, || {
        cached_renderer
            .draw(&mut cached_terminal, &cached_state, 0)
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
    measure("large live tool-output burst", 3, || {
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
