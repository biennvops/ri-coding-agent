use std::hint::black_box;
use std::time::{Duration, Instant};

use ratatui::backend::TestBackend;
use ratatui::Terminal;
use ri::render;
use ri_core::{AgentEvent, ToolOutputStream};

const WIDTH: u16 = 100;
const HEIGHT: u16 = 28;

fn main() {
    measure("fresh first frame", 10, || {
        let state = render::synthetic_transcript(20, 4);
        let mut terminal = test_terminal(WIDTH, HEIGHT);
        render::draw_terminal(&mut terminal, &state, 0).expect("test backend draw");
    });

    for rows in [1_000, 10_000, 100_000] {
        measure(&format!("cold layout · {rows} rows"), 1, || {
            let state = render::synthetic_transcript(rows, (rows / 10).max(1));
            let mut terminal = test_terminal(WIDTH, HEIGHT);
            render::draw_terminal(&mut terminal, &state, 0).expect("test backend draw");
        });
    }

    measure("cached redraw · 100k rows", 10, || {
        let state = render::synthetic_transcript(100_000, 10_000);
        let mut terminal = test_terminal(WIDTH, HEIGHT);
        render::draw_terminal(&mut terminal, &state, 0).expect("test backend draw");
        render::draw_terminal(&mut terminal, &state, 0).expect("test backend draw");
    });

    measure("scroll · 100k rows", 10, || {
        let state = render::synthetic_transcript(100_000, 10_000);
        let mut terminal = test_terminal(WIDTH, HEIGHT);
        render::draw_terminal(&mut terminal, &state, 20_000).expect("test backend draw");
    });

    measure("single streaming append · 100k rows", 10, || {
        let mut state = render::synthetic_transcript(100_000, 10_000);
        state.reduce(AgentEvent::AssistantMessageStarted);
        state.reduce(AgentEvent::AssistantTextDelta {
            index: None,
            text: "initial streaming response".to_owned(),
        });
        let mut terminal = test_terminal(WIDTH, HEIGHT);
        render::draw_terminal(&mut terminal, &state, 0).expect("test backend draw");
        render::append_streaming_delta(&mut state, " + delta");
        render::draw_terminal(&mut terminal, &state, 0).expect("test backend draw");
    });

    measure("1,000 streaming deltas · 100k rows", 1, || {
        let mut state = render::synthetic_transcript(100_000, 10_000);
        state.reduce(AgentEvent::AssistantMessageStarted);
        let mut terminal = test_terminal(WIDTH, HEIGHT);
        render::draw_terminal(&mut terminal, &state, 0).expect("test backend draw");
        for _ in 0..1_000 {
            render::append_streaming_delta(&mut state, " token");
            render::draw_terminal(&mut terminal, &state, 0).expect("test backend draw");
        }
    });

    measure("resize · 100k rows", 3, || {
        let state = render::synthetic_transcript(100_000, 10_000);
        let mut terminal = test_terminal(WIDTH, HEIGHT);
        render::draw_terminal(&mut terminal, &state, 0).expect("test backend draw");
        terminal.backend_mut().resize(80, HEIGHT);
        render::draw_terminal(&mut terminal, &state, 0).expect("test backend draw");
    });

    measure("large live tool-output burst", 3, || {
        let mut state = render::synthetic_transcript(10_000, 1_000);
        state.reduce(AgentEvent::ToolExecutionStarted {
            call_id: "live-tool".to_owned(),
            name: "bash".to_owned(),
            arguments: "{}".to_owned(),
        });
        let mut terminal = test_terminal(WIDTH, HEIGHT);
        render::draw_terminal(&mut terminal, &state, 0).expect("test backend draw");
        for index in 0..100 {
            state.reduce(AgentEvent::ToolExecutionOutput {
                call_id: "live-tool".to_owned(),
                stream: ToolOutputStream::Stdout,
                chunk: format!("output {index}\n"),
            });
            render::draw_terminal(&mut terminal, &state, 0).expect("test backend draw");
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
        black_box(operation());
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
