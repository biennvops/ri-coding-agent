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
        text: String,
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
        let process_tree = match ProcessTree::attach(&child) {
            Ok(process_tree) => process_tree,
            Err(error) => {
                let _ = child.kill().await;
                return Err(error);
            }
        };
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
                        Some(BashStreamEvent::Chunk { stream, text }) => {
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
                    terminate_child(&mut child, &process_tree).await?;
                    status = child.try_wait().map_err(|error| {
                        ToolError::Failed(format!("could not collect cancelled command: {error}"))
                    })?;
                }
                _ = &mut timeout_sleep, if status.is_none() && !timed_out && !cancelled => {
                    timed_out = true;
                    terminate_child(&mut child, &process_tree).await?;
                    status = child.try_wait().map_err(|error| {
                        ToolError::Failed(format!("could not collect timed-out command: {error}"))
                    })?;
                }
                _ = tokio::time::sleep(POLL_INTERVAL), if status.is_none() => {
                    match child.try_wait() {
                        Ok(next_status) => status = next_status,
                        Err(error) => {
                            process_error = Some(ToolError::Failed(format!("could not inspect command: {error}")));
                            terminate_child(&mut child, &process_tree).await?;
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
    let mut decoder = Utf8Decoder::default();
    loop {
        match reader.read(&mut buffer).await {
            Ok(0) => break,
            Ok(length) => {
                let text = decoder.decode(&buffer[..length]);
                if text.is_empty() {
                    continue;
                }
                if events
                    .send(BashStreamEvent::Chunk {
                        stream: stream.clone(),
                        text,
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
    let text = decoder.finish();
    if !text.is_empty()
        && events
            .send(BashStreamEvent::Chunk { stream, text })
            .await
            .is_err()
    {
        return;
    }
    let _ = events.send(BashStreamEvent::Closed).await;
}

#[derive(Default)]
struct Utf8Decoder {
    pending: Vec<u8>,
}

impl Utf8Decoder {
    fn decode(&mut self, bytes: &[u8]) -> String {
        let mut input = std::mem::take(&mut self.pending);
        input.extend_from_slice(bytes);
        self.decode_input(&input)
    }

    fn finish(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let pending = std::mem::take(&mut self.pending);
        String::from_utf8_lossy(&pending).into_owned()
    }

    fn decode_input(&mut self, input: &[u8]) -> String {
        let mut output = String::new();
        let mut cursor = 0;
        while cursor < input.len() {
            match std::str::from_utf8(&input[cursor..]) {
                Ok(text) => {
                    output.push_str(text);
                    cursor = input.len();
                }
                Err(error) => {
                    let valid = error.valid_up_to();
                    if valid > 0 {
                        output.push_str(
                            std::str::from_utf8(&input[cursor..cursor + valid])
                                .expect("valid UTF-8 prefix should decode"),
                        );
                        cursor += valid;
                    }
                    if let Some(error_length) = error.error_len() {
                        output.push('\u{FFFD}');
                        cursor += error_length;
                    } else {
                        self.pending.extend_from_slice(&input[cursor..]);
                        break;
                    }
                }
            }
        }
        output
    }
}

struct ProcessTree {
    #[cfg(windows)]
    job: WindowsJob,
}

impl ProcessTree {
    fn attach(child: &Child) -> Result<Self, ToolError> {
        #[cfg(windows)]
        {
            let job = WindowsJob::attach(child)?;
            resume_suspended_process(child)?;
            return Ok(Self { job });
        }
        #[cfg(not(windows))]
        {
            let _ = child;
            Ok(Self {})
        }
    }

    #[cfg(windows)]
    fn terminate(&self) -> Result<(), ToolError> {
        self.job.terminate()
    }
}

async fn terminate_child(child: &mut Child, process_tree: &ProcessTree) -> Result<(), ToolError> {
    #[cfg(unix)]
    let _ = process_tree;
    #[cfg(unix)]
    signal_process_group(child, SIGTERM);
    #[cfg(windows)]
    process_tree.terminate()?;
    #[cfg(not(any(unix, windows)))]
    child.start_kill().map_err(|error| {
        ToolError::Failed(format!("could not terminate shell command: {error}"))
    })?;

    tokio::time::sleep(TERMINATION_GRACE).await;

    #[cfg(unix)]
    signal_process_group(child, SIGKILL);
    #[cfg(windows)]
    if child
        .try_wait()
        .map_err(|error| {
            ToolError::Failed(format!("could not inspect terminated command: {error}"))
        })?
        .is_none()
    {
        process_tree.terminate()?;
    }
    #[cfg(not(any(unix, windows)))]
    child
        .start_kill()
        .map_err(|error| ToolError::Failed(format!("could not kill shell command: {error}")))?;
    let _ = child
        .wait()
        .await
        .map_err(|error| ToolError::Failed(format!("could not wait for shell command: {error}")))?;
    Ok(())
}

#[cfg(windows)]
struct WindowsJob {
    handle: usize,
}

#[cfg(windows)]
impl WindowsJob {
    fn attach(child: &Child) -> Result<Self, ToolError> {
        let handle = unsafe { CreateJobObjectW(std::ptr::null_mut(), std::ptr::null()) };
        if handle.is_null() {
            return Err(ToolError::Failed(format!(
                "could not create Windows job object: {}",
                std::io::Error::last_os_error()
            )));
        }
        let mut limits = JobObjectExtendedLimitInformation::default();
        limits.basic_limit_information.limit_flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let configured = unsafe {
            SetInformationJobObject(
                handle,
                JOBOBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
                &mut limits as *mut _ as *mut std::ffi::c_void,
                std::mem::size_of::<JobObjectExtendedLimitInformation>() as u32,
            )
        } != 0;
        if !configured {
            unsafe {
                CloseHandle(handle);
            }
            return Err(ToolError::Failed(format!(
                "could not configure Windows job object: {}",
                std::io::Error::last_os_error()
            )));
        }
        let process_handle = child.raw_handle().ok_or_else(|| {
            unsafe {
                CloseHandle(handle);
            }
            ToolError::Failed("shell process exited before job assignment".to_owned())
        })?;
        if unsafe { AssignProcessToJobObject(handle, process_handle) } == 0 {
            let error = std::io::Error::last_os_error();
            unsafe {
                CloseHandle(handle);
            }
            return Err(ToolError::Failed(format!(
                "could not assign shell process to Windows job object: {error}"
            )));
        }
        Ok(Self {
            handle: handle as usize,
        })
    }

    fn terminate(&self) -> Result<(), ToolError> {
        if unsafe { TerminateJobObject(self.handle as *mut std::ffi::c_void, 1) } == 0 {
            return Err(ToolError::Failed(format!(
                "could not terminate Windows process tree: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok(())
    }
}

#[cfg(windows)]
impl Drop for WindowsJob {
    fn drop(&mut self) {
        unsafe {
            let _ = CloseHandle(self.handle as *mut std::ffi::c_void);
        }
    }
}

#[cfg(windows)]
const JOBOBJECT_EXTENDED_LIMIT_INFORMATION_CLASS: u32 = 9;
#[cfg(windows)]
const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: u32 = 0x0000_2000;

#[cfg(windows)]
#[repr(C)]
#[derive(Default)]
struct JobObjectBasicLimitInformation {
    per_process_user_time_limit: i64,
    per_job_user_time_limit: i64,
    limit_flags: u32,
    minimum_working_set_size: usize,
    maximum_working_set_size: usize,
    active_process_limit: u32,
    affinity: usize,
    priority_class: u32,
    scheduling_class: u32,
}

#[cfg(windows)]
#[repr(C)]
#[derive(Default)]
struct IoCounters {
    read_operation_count: u64,
    write_operation_count: u64,
    other_operation_count: u64,
    read_transfer_count: u64,
    write_transfer_count: u64,
    other_transfer_count: u64,
}

#[cfg(windows)]
#[repr(C)]
#[derive(Default)]
struct JobObjectExtendedLimitInformation {
    basic_limit_information: JobObjectBasicLimitInformation,
    io_information: IoCounters,
    process_memory_limit: usize,
    job_memory_limit: usize,
    peak_process_memory_used: usize,
    peak_job_memory_used: usize,
}

#[cfg(windows)]
#[repr(C)]
#[derive(Default)]
struct ThreadEntry32 {
    dw_size: u32,
    cnt_usage: u32,
    th32_thread_id: u32,
    th32_owner_process_id: u32,
    tp_base_pri: i32,
    tp_delta_pri: i32,
    dw_flags: u32,
}

#[cfg(windows)]
const CREATE_SUSPENDED: u32 = 0x0000_0004;
#[cfg(windows)]
const TH32CS_SNAPTHREAD: u32 = 0x0000_0004;
#[cfg(windows)]
const THREAD_SUSPEND_RESUME: u32 = 0x0002;
#[cfg(windows)]
const INVALID_HANDLE_VALUE: *mut std::ffi::c_void = (-1isize) as *mut std::ffi::c_void;

#[cfg(windows)]
unsafe extern "system" {
    fn CreateJobObjectW(
        attributes: *mut std::ffi::c_void,
        name: *const u16,
    ) -> *mut std::ffi::c_void;
    fn SetInformationJobObject(
        job: *mut std::ffi::c_void,
        information_class: u32,
        job_object_information: *mut std::ffi::c_void,
        job_object_information_length: u32,
    ) -> i32;
    fn AssignProcessToJobObject(job: *mut std::ffi::c_void, process: *mut std::ffi::c_void) -> i32;
    fn TerminateJobObject(job: *mut std::ffi::c_void, exit_code: u32) -> i32;
    fn CloseHandle(handle: *mut std::ffi::c_void) -> i32;
    fn CreateToolhelp32Snapshot(flags: u32, process_id: u32) -> *mut std::ffi::c_void;
    fn Thread32First(snapshot: *mut std::ffi::c_void, entry: *mut ThreadEntry32) -> i32;
    fn Thread32Next(snapshot: *mut std::ffi::c_void, entry: *mut ThreadEntry32) -> i32;
    fn OpenThread(
        desired_access: u32,
        inherit_handle: i32,
        thread_id: u32,
    ) -> *mut std::ffi::c_void;
    fn ResumeThread(thread: *mut std::ffi::c_void) -> u32;
}

#[cfg(windows)]
fn resume_suspended_process(child: &Child) -> Result<(), ToolError> {
    let process_id = child
        .id()
        .ok_or_else(|| ToolError::Failed("shell process exited before resume".to_owned()))?;
    let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) };
    if snapshot.is_null() || snapshot == INVALID_HANDLE_VALUE {
        return Err(ToolError::Failed(format!(
            "could not inspect suspended shell threads: {}",
            std::io::Error::last_os_error()
        )));
    }

    let mut entry = ThreadEntry32 {
        dw_size: std::mem::size_of::<ThreadEntry32>() as u32,
        ..ThreadEntry32::default()
    };
    let mut thread_id = None;
    let mut found = unsafe { Thread32First(snapshot, &mut entry) } != 0;
    while found {
        if entry.th32_owner_process_id == process_id {
            thread_id = Some(entry.th32_thread_id);
            break;
        }
        found = unsafe { Thread32Next(snapshot, &mut entry) } != 0;
    }
    unsafe {
        CloseHandle(snapshot);
    }

    let thread_id = thread_id.ok_or_else(|| {
        ToolError::Failed("could not find suspended shell primary thread".to_owned())
    })?;
    let thread = unsafe { OpenThread(THREAD_SUSPEND_RESUME, 0, thread_id) };
    if thread.is_null() {
        return Err(ToolError::Failed(format!(
            "could not open suspended shell primary thread: {}",
            std::io::Error::last_os_error()
        )));
    }
    let resumed = unsafe { ResumeThread(thread) };
    unsafe {
        CloseHandle(thread);
    }
    if resumed == u32::MAX {
        return Err(ToolError::Failed(format!(
            "could not resume shell primary thread: {}",
            std::io::Error::last_os_error()
        )));
    }
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

#[cfg(windows)]
fn configure_process_group(command: &mut Command) {
    command.creation_flags(CREATE_SUSPENDED);
}

#[cfg(not(any(unix, windows)))]
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

    #[test]
    fn utf8_decoder_preserves_code_points_split_between_reads() {
        let mut decoder = Utf8Decoder::default();
        assert_eq!(decoder.decode(&[0xf0, 0x9f]), "");
        assert_eq!(decoder.decode(&[0xa6, 0x80]), "🦀");
        assert_eq!(decoder.finish(), "");
    }

    #[tokio::test]
    async fn captures_stdout_stderr_and_nonzero_exit() {
        let root = unique_test_dir("bash");
        fs::create_dir_all(&root).unwrap();
        let context = ToolContext::new(&root).unwrap();
        let (events, mut event_rx) = mpsc::channel(32);
        let command = if cfg!(windows) {
            "echo out & echo err 1>&2 & exit /B 7"
        } else {
            "printf out; printf err >&2; exit 7"
        };
        let result = BashTool
            .execute(
                json!({"command": command}),
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
        let command = if cfg!(windows) {
            "ping -n 6 127.0.0.1 >NUL"
        } else {
            "sleep 5"
        };
        let result = BashTool
            .execute(
                json!({"command":command, "timeout_ms":50}),
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
        let command = if cfg!(windows) {
            "ping -n 6 127.0.0.1 >NUL"
        } else {
            "sleep 5"
        };
        let task = tokio::spawn(async move {
            BashTool
                .execute(
                    json!({"command":command}),
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

    #[cfg(windows)]
    #[tokio::test]
    async fn cancellation_kills_windows_descendants() {
        let root = unique_test_dir("bash-windows-tree");
        fs::create_dir_all(&root).unwrap();
        let context = ToolContext::new(&root).unwrap();
        let cancel = CancellationToken::new();
        let task_cancel = cancel.clone();
        let task = tokio::spawn(async move {
            BashTool
                .execute(
                    json!({"command":"start \"\" /B cmd /C \"ping -n 6 127.0.0.1 >NUL & echo leaked > descendant.txt\" & ping -n 6 127.0.0.1 >NUL"}),
                    &context,
                    mpsc::channel(8).0,
                    task_cancel,
                )
                .await
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel.cancel();
        let result = task.await.unwrap().unwrap();
        assert!(result.metadata.cancelled);
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!root.join("descendant.txt").exists());
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
