use std::fs::{File, OpenOptions};
use std::io::Write;
use std::process::Stdio;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::model::ToolDefinition;

use super::write::temporary_path;
use super::{
    BoundedText, Tool, ToolContext, ToolError, ToolEvent, ToolEventSender, ToolExecutionMetadata,
    ToolExecutionResult, ToolOutputStream, MAX_TOOL_OUTPUT_BYTES,
};

pub const DEFAULT_BASH_TIMEOUT_MS: u64 = 120_000;
const POLL_INTERVAL: Duration = Duration::from_millis(10);
const TERMINATION_GRACE: Duration = Duration::from_millis(75);
const STREAM_OUTPUT_LIMIT: usize = MAX_TOOL_OUTPUT_BYTES / 2;
const BASH_CHUNK_BYTES: usize = 8 * 1024;

pub(crate) struct BashTool;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BashArguments {
    command: String,
    timeout_ms: Option<u64>,
}

#[derive(Debug)]
enum BashStreamEvent {
    Chunk {
        stream: ToolOutputStream,
        bytes: Vec<u8>,
    },
    Closed,
    Error {
        stream: ToolOutputStream,
        message: String,
    },
}

#[async_trait]
impl Tool for BashTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "bash".to_owned(),
            description: Some("Run a shell command from the workspace directory.".to_owned()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute from the workspace"
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional command timeout in milliseconds"
                    }
                },
                "required": ["command"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(
        &self,
        arguments: Value,
        context: &ToolContext,
        events: ToolEventSender,
        cancel: CancellationToken,
    ) -> Result<ToolExecutionResult, ToolError> {
        if cancel.is_cancelled() {
            return Err(ToolError::Cancelled);
        }
        let arguments: BashArguments = serde_json::from_value(arguments)
            .map_err(|error| ToolError::InvalidArguments(error.to_string()))?;
        if arguments.command.trim().is_empty() {
            return Err(ToolError::InvalidArguments(
                "command must not be empty".to_owned(),
            ));
        }
        let timeout_ms = arguments.timeout_ms.unwrap_or(DEFAULT_BASH_TIMEOUT_MS);
        if timeout_ms == 0 {
            return Err(ToolError::InvalidArguments(
                "timeout_ms must be at least 1".to_owned(),
            ));
        }
        let timeout = Duration::from_millis(timeout_ms);
        let mut command = shell_command(&arguments.command);
        command
            .current_dir(&context.workspace_root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        configure_process_group(&mut command);

        let mut child = command.spawn().map_err(|error| {
            ToolError::Failed(format!("could not start shell command: {error}"))
        })?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| ToolError::Failed("shell stdout pipe was unavailable".to_owned()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| ToolError::Failed("shell stderr pipe was unavailable".to_owned()))?;

        let (stream_tx, mut stream_rx) = mpsc::channel(64);
        tokio::spawn(read_stream(
            stdout,
            ToolOutputStream::Stdout,
            stream_tx.clone(),
        ));
        tokio::spawn(read_stream(stderr, ToolOutputStream::Stderr, stream_tx));

        let started = Instant::now();
        let mut output = BashOutput::new();
        let mut status = None;
        let mut active_readers = 2usize;
        let mut timed_out = false;
        let mut cancelled = false;
        let mut event_stream_closed = false;
        let mut process_error = None;
        let mut timeout_sleep = Box::pin(tokio::time::sleep(timeout));

        while status.is_none() || active_readers > 0 {
            if status.is_some() && active_readers == 0 {
                break;
            }
            tokio::select! {
                stream_event = stream_rx.recv(), if active_readers > 0 => {
                    match stream_event {
                        Some(BashStreamEvent::Chunk { stream, bytes }) => {
                            let text = String::from_utf8_lossy(&bytes).into_owned();
                            if let Err(error) = output.push(stream.clone(), &text) {
                                process_error.get_or_insert(error);
                            }
                            if !event_stream_closed {
                                match events.try_send(ToolEvent::Output {
                                    stream,
                                    chunk: text,
                                }) {
                                    Ok(()) | Err(mpsc::error::TrySendError::Full(_)) => {}
                                    Err(mpsc::error::TrySendError::Closed(_)) => {
                                        event_stream_closed = true;
                                    }
                                }
                            }
                        }
                        Some(BashStreamEvent::Closed) => {
                            active_readers = active_readers.saturating_sub(1);
                        }
                        Some(BashStreamEvent::Error { stream, message }) => {
                            active_readers = active_readers.saturating_sub(1);
                            let name = match stream {
                                ToolOutputStream::Stdout => "stdout",
                                ToolOutputStream::Stderr => "stderr",
                            };
                            process_error.get_or_insert_with(|| {
                                ToolError::Failed(format!("could not read command {name}: {message}"))
                            });
                        }
                        None => active_readers = 0,
                    }
                }
                _ = cancel.cancelled(), if status.is_none() && !cancelled => {
                    cancelled = true;
                    terminate_child(&mut child).await?;
                    status = child.try_wait().map_err(|error| {
                        ToolError::Failed(format!("could not collect cancelled command: {error}"))
                    })?;
                }
                _ = &mut timeout_sleep, if status.is_none() && !timed_out && !cancelled => {
                    timed_out = true;
                    terminate_child(&mut child).await?;
                    status = child.try_wait().map_err(|error| {
                        ToolError::Failed(format!("could not collect timed-out command: {error}"))
                    })?;
                }
                _ = tokio::time::sleep(POLL_INTERVAL), if status.is_none() => {
                    match child.try_wait() {
                        Ok(next_status) => status = next_status,
                        Err(error) => {
                            process_error = Some(ToolError::Failed(format!("could not inspect command: {error}")));
                            terminate_child(&mut child).await?;
                            status = child.try_wait().ok().flatten();
                        }
                    }
                }
                else => break,
            }
        }

        if let Some(error) = process_error {
            return Err(error);
        }

        let duration = started.elapsed();
        let exit_code = status.as_ref().and_then(std::process::ExitStatus::code);
        let command_succeeded = status
            .as_ref()
            .is_some_and(std::process::ExitStatus::success);
        let truncated = output.truncated();
        let full_output_path = output.finish()?;
        let mut model_content = if timed_out {
            format!("Command timed out after {timeout_ms}ms.")
        } else if cancelled {
            "Command cancelled by user.".to_owned()
        } else if let Some(exit_code) = exit_code {
            format!(
                "Command exited with code {exit_code} after {:.2}s.",
                duration.as_secs_f64()
            )
        } else {
            format!(
                "Command finished without an exit code after {:.2}s.",
                duration.as_secs_f64()
            )
        };

        let stdout_text = output.stdout();
        let stderr_text = output.stderr();
        if !stdout_text.is_empty() {
            model_content.push_str("\n\nstdout:\n");
            model_content.push_str(&stdout_text);
        }
        if !stderr_text.is_empty() {
            model_content.push_str("\n\nstderr:\n");
            model_content.push_str(&stderr_text);
        }
        if truncated {
            if let Some(path) = &full_output_path {
                model_content.push_str(&format!(
                    "\n\n[Output truncated. Full output saved to {}]",
                    path.display()
                ));
            } else {
                model_content
                    .push_str("\n\n[Output truncated; the full output could not be saved.]");
            }
        }

        Ok(ToolExecutionResult {
            model_content,
            metadata: ToolExecutionMetadata {
                success: !cancelled && !timed_out && command_succeeded,
                exit_code,
                timed_out,
                cancelled,
                truncated,
                duration,
                full_output_path,
            },
        })
    }
}

fn shell_command(command: &str) -> Command {
    #[cfg(unix)]
    {
        let mut shell = Command::new("/bin/sh");
        shell.args(["-lc", command]);
        shell
    }
    #[cfg(windows)]
    {
        let mut shell = Command::new("cmd");
        shell.args(["/C", command]);
        shell
    }
}

async fn read_stream<R>(
    mut reader: R,
    stream: ToolOutputStream,
    events: mpsc::Sender<BashStreamEvent>,
) where
    R: AsyncRead + Unpin,
{
    let mut buffer = vec![0; BASH_CHUNK_BYTES];
    loop {
        match reader.read(&mut buffer).await {
            Ok(0) => break,
            Ok(length) => {
                if events
                    .send(BashStreamEvent::Chunk {
                        stream: stream.clone(),
                        bytes: buffer[..length].to_vec(),
                    })
                    .await
                    .is_err()
                {
                    return;
                }
            }
            Err(error) => {
                let _ = events
                    .send(BashStreamEvent::Error {
                        stream: stream.clone(),
                        message: error.to_string(),
                    })
                    .await;
                return;
            }
        }
    }
    let _ = events.send(BashStreamEvent::Closed).await;
}

async fn terminate_child(child: &mut Child) -> Result<(), ToolError> {
    #[cfg(unix)]
    signal_process_group(child, SIGTERM);
    #[cfg(not(unix))]
    child.start_kill().map_err(|error| {
        ToolError::Failed(format!("could not terminate shell command: {error}"))
    })?;

    tokio::time::sleep(TERMINATION_GRACE).await;

    #[cfg(unix)]
    signal_process_group(child, SIGKILL);
    child
        .start_kill()
        .map_err(|error| ToolError::Failed(format!("could not kill shell command: {error}")))?;
    let _ = child
        .wait()
        .await
        .map_err(|error| ToolError::Failed(format!("could not wait for shell command: {error}")))?;
    Ok(())
}

#[cfg(unix)]
const SIGTERM: i32 = 15;
#[cfg(unix)]
const SIGKILL: i32 = 9;

#[cfg(unix)]
unsafe extern "C" {
    fn kill(pid: i32, signal: i32) -> i32;
    fn setpgid(pid: i32, process_group: i32) -> i32;
}

#[cfg(unix)]
fn configure_process_group(command: &mut Command) {
    unsafe {
        command.pre_exec(|| {
            if setpgid(0, 0) == 0 {
                Ok(())
            } else {
                Err(std::io::Error::last_os_error())
            }
        });
    }
}

#[cfg(not(unix))]
fn configure_process_group(_command: &mut Command) {}

#[cfg(unix)]
fn signal_process_group(child: &Child, signal: i32) {
    if let Some(pid) = child.id() {
        unsafe {
            let _ = kill(-(pid as i32), signal);
        }
    }
}

struct BashOutput {
    stdout: BoundedText,
    stderr: BoundedText,
    combined: BoundedText,
    spill: Option<File>,
    spill_path: Option<std::path::PathBuf>,
}

impl BashOutput {
    fn new() -> Self {
        Self {
            stdout: BoundedText::new(STREAM_OUTPUT_LIMIT),
            stderr: BoundedText::new(STREAM_OUTPUT_LIMIT),
            combined: BoundedText::new(MAX_TOOL_OUTPUT_BYTES),
            spill: None,
            spill_path: None,
        }
    }

    fn push(&mut self, stream: ToolOutputStream, text: &str) -> Result<(), ToolError> {
        let stream_would_truncate = match stream {
            ToolOutputStream::Stdout => {
                self.stdout.total_bytes().saturating_add(text.len()) > STREAM_OUTPUT_LIMIT
            }
            ToolOutputStream::Stderr => {
                self.stderr.total_bytes().saturating_add(text.len()) > STREAM_OUTPUT_LIMIT
            }
        };
        let label = match stream {
            ToolOutputStream::Stdout => "[stdout]\n",
            ToolOutputStream::Stderr => "[stderr]\n",
        };
        let mut combined = String::with_capacity(label.len().saturating_add(text.len()));
        combined.push_str(label);
        combined.push_str(text);

        let combined_would_truncate =
            self.combined.total_bytes().saturating_add(combined.len()) > MAX_TOOL_OUTPUT_BYTES;
        if self.spill.is_none() && (stream_would_truncate || combined_would_truncate) {
            self.create_spill()?;
        }
        if let Some(spill) = &mut self.spill {
            spill.write_all(combined.as_bytes()).map_err(|error| {
                ToolError::Failed(format!("could not append output spill file: {error}"))
            })?;
        }
        match stream {
            ToolOutputStream::Stdout => self.stdout.push(text),
            ToolOutputStream::Stderr => self.stderr.push(text),
        }
        self.combined.push(&combined);
        Ok(())
    }

    fn create_spill(&mut self) -> Result<(), ToolError> {
        let path = temporary_path(&std::env::temp_dir());
        let mut spill = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| {
                ToolError::Failed(format!("could not create output spill file: {error}"))
            })?;
        spill
            .write_all(self.combined.retained().as_bytes())
            .map_err(|error| {
                ToolError::Failed(format!("could not initialize output spill file: {error}"))
            })?;
        self.spill_path = Some(path);
        self.spill = Some(spill);
        Ok(())
    }

    fn stdout(&self) -> String {
        self.stdout.render()
    }

    fn stderr(&self) -> String {
        self.stderr.render()
    }

    fn truncated(&self) -> bool {
        self.combined.is_truncated() || self.stdout.is_truncated() || self.stderr.is_truncated()
    }

    fn finish(&mut self) -> Result<Option<std::path::PathBuf>, ToolError> {
        if let Some(spill) = &mut self.spill {
            spill.sync_all().map_err(|error| {
                ToolError::Failed(format!("could not flush output spill file: {error}"))
            })?;
        }
        Ok(self.spill_path.clone())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[tokio::test]
    async fn captures_stdout_stderr_and_nonzero_exit() {
        let root = unique_test_dir("bash");
        fs::create_dir_all(&root).unwrap();
        let context = ToolContext::new(&root).unwrap();
        let (events, mut event_rx) = mpsc::channel(32);
        let result = BashTool
            .execute(
                json!({"command":"printf out; printf err >&2; exit 7"}),
                &context,
                events,
                CancellationToken::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.metadata.exit_code, Some(7));
        assert!(!result.metadata.success);
        assert!(result.model_content.contains("stdout:"));
        assert!(result.model_content.contains("out"));
        assert!(result.model_content.contains("stderr:"));
        assert!(result.model_content.contains("err"));
        assert!(event_rx.recv().await.is_some());
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn timeout_returns_a_recoverable_result() {
        let root = unique_test_dir("bash-timeout");
        fs::create_dir_all(&root).unwrap();
        let context = ToolContext::new(&root).unwrap();
        let result = BashTool
            .execute(
                json!({"command":"sleep 5", "timeout_ms":50}),
                &context,
                mpsc::channel(8).0,
                CancellationToken::new(),
            )
            .await
            .unwrap();

        assert!(result.metadata.timed_out);
        assert!(!result.metadata.success);
        assert!(result.model_content.contains("timed out"));
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn cancellation_terminates_command_and_reports_it() {
        let root = unique_test_dir("bash-cancel");
        fs::create_dir_all(&root).unwrap();
        let context = ToolContext::new(&root).unwrap();
        let cancel = CancellationToken::new();
        let task_cancel = cancel.clone();
        let task = tokio::spawn(async move {
            BashTool
                .execute(
                    json!({"command":"sleep 5"}),
                    &context,
                    mpsc::channel(8).0,
                    task_cancel,
                )
                .await
        });
        tokio::time::sleep(Duration::from_millis(30)).await;
        cancel.cancel();
        let result = task.await.unwrap().unwrap();
        assert!(result.metadata.cancelled);
        assert!(result.model_content.contains("cancelled"));
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn stdout_truncation_spills_before_combined_limit() {
        let root = unique_test_dir("bash-stdout-boundary");
        fs::create_dir_all(&root).unwrap();
        let context = ToolContext::new(&root).unwrap();
        let result = BashTool
            .execute(
                json!({"command": format!("yes x | head -c {}", STREAM_OUTPUT_LIMIT + 1024)}),
                &context,
                mpsc::channel(128).0,
                CancellationToken::new(),
            )
            .await
            .unwrap();

        assert!(result.metadata.truncated);
        let path = result
            .metadata
            .full_output_path
            .expect("stdout truncation should spill");
        assert!(fs::metadata(path).unwrap().len() > STREAM_OUTPUT_LIMIT as u64);
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn stderr_truncation_spills_before_combined_limit() {
        let root = unique_test_dir("bash-stderr-boundary");
        fs::create_dir_all(&root).unwrap();
        let context = ToolContext::new(&root).unwrap();
        let result = BashTool
            .execute(
                json!({"command": format!("yes x | head -c {} 1>&2", STREAM_OUTPUT_LIMIT + 1024)}),
                &context,
                mpsc::channel(128).0,
                CancellationToken::new(),
            )
            .await
            .unwrap();

        assert!(result.metadata.truncated);
        let path = result
            .metadata
            .full_output_path
            .expect("stderr truncation should spill");
        assert!(fs::metadata(path).unwrap().len() > STREAM_OUTPUT_LIMIT as u64);
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn large_output_is_bounded_and_spilled() {
        let root = unique_test_dir("bash-large");
        fs::create_dir_all(&root).unwrap();
        let context = ToolContext::new(&root).unwrap();
        let result = BashTool
            .execute(
                json!({"command": format!("yes x | head -c {}", MAX_TOOL_OUTPUT_BYTES + 1024)}),
                &context,
                mpsc::channel(128).0,
                CancellationToken::new(),
            )
            .await
            .unwrap();

        assert!(result.metadata.truncated);
        let path = result
            .metadata
            .full_output_path
            .expect("output should spill");
        assert!(fs::metadata(path).unwrap().len() > MAX_TOOL_OUTPUT_BYTES as u64);
        fs::remove_dir_all(root).unwrap();
    }

    fn unique_test_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "ri-{name}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }
}
