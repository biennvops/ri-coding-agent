use ri_core::{AgentEvent, AppState, ToolExecutionMetadata, ToolExecutionResult, ToolOutputStream};

/// Build a deterministic transcript with a mix of entry kinds and Unicode.
///
/// `approx_rows` controls the amount of text and `entry_count` controls how
/// many independent entries receive that text. The exact row count depends on
/// the renderer width, so callers should treat it as a workload size.
pub fn synthetic_transcript(approx_rows: usize, entry_count: usize) -> AppState {
    let mut state = AppState::new();
    let entry_count = entry_count.max(1);
    let rows_per_entry = approx_rows.div_ceil(entry_count).max(1);

    for index in 0..entry_count {
        let body = synthetic_body(index, rows_per_entry);
        match index % 4 {
            0 => state.add_system_message(body),
            1 => {
                state.insert_text(&body);
                let _ = state.submit_input();
                state.reduce(AgentEvent::TurnFinished {
                    reason: ri_core::StopReason::Stop,
                });
            }
            2 => {
                state.reduce(AgentEvent::AssistantMessageStarted);
                state.reduce(AgentEvent::AssistantThinkingDelta {
                    item_id: None,
                    text: format!("reasoning for entry {index} · 🧭"),
                });
                state.reduce(AgentEvent::AssistantTextDelta {
                    index: None,
                    text: body,
                });
                state.reduce(AgentEvent::AssistantMessageFinished { items: Vec::new() });
            }
            _ => {
                let call_id = format!("synthetic-call-{index}");
                state.reduce(AgentEvent::ToolExecutionStarted {
                    call_id: call_id.clone(),
                    name: "bash".to_owned(),
                    arguments: format!(r#"{{"index":{index}}}"#),
                });
                state.reduce(AgentEvent::ToolExecutionOutput {
                    call_id: call_id.clone(),
                    stream: ToolOutputStream::Stdout,
                    chunk: body.clone(),
                });
                state.reduce(AgentEvent::ToolExecutionFinished {
                    call_id,
                    name: "bash".to_owned(),
                    result: ToolExecutionResult {
                        model_content: body,
                        metadata: ToolExecutionMetadata::success(),
                    },
                });
            }
        }
    }

    state
}

pub fn append_streaming_delta(state: &mut AppState, text: &str) {
    state.reduce(AgentEvent::AssistantTextDelta {
        index: None,
        text: text.to_owned(),
    });
}

fn synthetic_body(index: usize, rows: usize) -> String {
    (0..rows)
        .map(|row| {
            format!(
                "entry {index:05} row {row:04} · deterministic Unicode: 世界 🦀 e\u{301} · {}",
                "wrapped text ".repeat(3)
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}
