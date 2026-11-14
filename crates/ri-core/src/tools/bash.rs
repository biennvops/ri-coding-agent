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
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::model::ToolDefinition;

use super::write::temporary_path;
use super::{
    bounded_preview, preview_lines, suffix_at_byte_boundary, Tool, ToolCallPresentation,
    ToolContext, ToolError, ToolEvent, ToolEventSender, ToolExecutionMetadata, ToolExecutionResult,
    ToolOutputStream, ToolPreviewKind,
};

pub const DEFAULT_BASH_TIMEOUT_MS: u64 = 120_000;
const POLL_INTERVAL: Duration = Duration::from_millis(10);
const TERMINATION_GRACE: Duration = Duration::from_millis(75);
const READER_DRAIN_GRACE: Duration = Duration::from_millis(100);
const MAX_BASH_OUTPUT_LINES: usize = 2000;
const MAX_BASH_OUTPUT_BYTES: usize = 50 * 1024;
// Only the end of the output is kept once it spills, so memory stays bounded.
const TAIL_RETAINED_BYTES: usize = MAX_BASH_OUTPUT_BYTES * 2;
const BASH_CHUNK_BYTES: usize = 8 * 1024;

pub(crate) struct BashTool;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BashArguments {
    command: String,
    timeout_ms: Option<u64>,
}

impl BashArguments {
    fn parse(arguments: &Value) -> Result<Self, ToolError> {
        serde_json::from_value(arguments.clone())
            .map_err(|error| ToolError::InvalidArguments(error.to_string()))
    }
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

    fn presentation(&self, arguments: &Value) -> ToolCallPresentation {
        let Ok(arguments) = BashArguments::parse(arguments) else {
            return ToolCallPresentation::fallback("bash", arguments);
        };
        ToolCallPresentation {
            summary: "bash".to_owned(),
            summary_kind: super::ToolSummaryKind::Normal,
            output_kind: super::ToolOutputKind::Normal,
            preview: preview_lines(
                bounded_preview(
                    arguments.command.lines().map(|line| (None, line)),
                    arguments.command.lines().count(),
                ),
                ToolPreviewKind::Command,
            ),
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
        let arguments = BashArguments::parse(&arguments)?;
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
            .stderr(Stdio::piped())
            // This only covers the shell itself. ProcessTreeGuard below is
            // still required for best-effort cleanup of descendants that stay
            // in the owned group or Job; Unix process groups are not complete
            // ownership of descendants that detach with setsid.
            .kill_on_drop(true);
        configure_process_group(&mut command);

        let mut child = command.spawn().map_err(|error| {
            ToolError::Failed(format!("could not start shell command: {error}"))
        })?;
        let mut process_tree = match ProcessTreeGuard::attach(&child) {
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
        let reader_tasks = ReaderTasks::new(
            tokio::spawn(read_stream(
                stdout,
                ToolOutputStream::Stdout,
                stream_tx.clone(),
            )),
            tokio::spawn(read_stream(stderr, ToolOutputStream::Stderr, stream_tx)),
        );

        let started = Instant::now();
        let mut output = BashOutput::new();
        let mut status = None;
        let mut active_readers = 2usize;
        let mut timed_out = false;
        let mut cancelled = false;
        let mut event_stream_closed = false;
        let mut process_error = None;
        let mut tree_cleanup_started = false;
        let mut timeout_sleep = Box::pin(tokio::time::sleep(timeout));
        let mut reader_drain_sleep = Box::pin(tokio::time::sleep(READER_DRAIN_GRACE));

        while status.is_none() || active_readers > 0 {
            let was_running = status.is_none();
            if status.is_some() && !tree_cleanup_started {
                process_tree.kill_best_effort();
                tree_cleanup_started = true;
            }
            if status.is_some() && active_readers == 0 {
                break;
            }
            tokio::select! {
                stream_event = stream_rx.recv(), if active_readers > 0 => {
                    match stream_event {
                        Some(BashStreamEvent::Chunk { stream, text }) => {
                            if let Err(error) = output.push(&text) {
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
                _ = &mut reader_drain_sleep, if status.is_some() && active_readers > 0 => {
                    reader_tasks.abort();
                    active_readers = 0;
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
            if was_running && status.is_some() {
                reader_drain_sleep
                    .as_mut()
                    .reset(tokio::time::Instant::now() + READER_DRAIN_GRACE);
            }
        }

        if status.is_some() && !tree_cleanup_started {
            process_tree.kill_best_effort();
        }
        reader_tasks.join().await;

        if let Some(error) = process_error {
            return Err(error);
        }
        process_tree.disarm();

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

        let tail = output.tail_truncation();
        if !tail.content.is_empty() {
            model_content.push_str("\n\n");
            model_content.push_str(&tail.content);
        }
        if truncated {
            model_content.push_str("\n\n");
            model_content.push_str(&output.truncation_notice(&tail, full_output_path.as_deref()));
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
                    .send(BashStreamEvent::Chunk { stream, text })
                    .await
                    .is_err()
                {
                    return;
                }
            }
            Err(error) => {
                let _ = events
                    .send(BashStreamEvent::Error {
                        stream,
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

struct ReaderTasks {
    handles: Vec<JoinHandle<()>>,
}

impl ReaderTasks {
    fn new(stdout: JoinHandle<()>, stderr: JoinHandle<()>) -> Self {
        Self {
            handles: vec![stdout, stderr],
        }
    }

    fn abort(&self) {
        for handle in &self.handles {
            handle.abort();
        }
    }

    async fn join(mut self) {
        for handle in self.handles.drain(..) {
            let _ = handle.await;
        }
    }
}

impl Drop for ReaderTasks {
    fn drop(&mut self) {
        self.abort();
    }
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

// On Unix this is a process-group ownership boundary, not a complete
// representation of every descendant process.
struct ProcessTree {
    #[cfg(unix)]
    process_group: Option<i32>,
    #[cfg(windows)]
    job: WindowsJob,
}

impl ProcessTree {
    fn attach(child: &Child) -> Result<Self, ToolError> {
        #[cfg(unix)]
        {
            let process_group = child.id().and_then(|pid| {
                let pid = pid as i32;
                // SAFETY: getpgrp has no pointers or borrowed state; it only
                // reads the calling process's group for the ownership check.
                let own_group = unsafe { getpgrp() };
                (pid > 1 && pid != own_group).then_some(pid)
            });
            Ok(Self { process_group })
        }
        #[cfg(windows)]
        {
            let job = WindowsJob::attach(child)?;
            resume_suspended_process(child)?;
            Ok(Self { job })
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = child;
            Ok(Self {})
        }
    }

    fn terminate(&self, child: &mut Child) -> Result<(), ToolError> {
        #[cfg(unix)]
        self.signal(Some(child), SIGTERM);
        #[cfg(windows)]
        {
            let _ = child;
            self.job.terminate()?;
        }
        #[cfg(not(any(unix, windows)))]
        child.start_kill().map_err(|error| {
            ToolError::Failed(format!("could not terminate shell command: {error}"))
        })?;
        Ok(())
    }

    fn kill_best_effort(&self, child: Option<&Child>) {
        #[cfg(unix)]
        self.signal(child, SIGKILL);
        #[cfg(windows)]
        {
            let _ = child;
            let _ = self.job.terminate();
        }
        #[cfg(not(any(unix, windows)))]
        let _ = child;
    }

    #[cfg(unix)]
    fn signal(&self, child: Option<&Child>, signal: i32) {
        if let Some(process_group) = self.process_group {
            // SAFETY: process_group is the child PID after the child-side
            // setpgid(0, 0) setup, and attach rejects ri's own process group.
            unsafe {
                let _ = kill(-process_group, signal);
            }
        } else if let Some(pid) = child.and_then(Child::id) {
            // SAFETY: this fallback targets only the spawned child PID when
            // process-group discovery was unavailable.
            unsafe {
                let _ = kill(pid as i32, signal);
            }
        }
    }
}

struct ProcessTreeGuard {
    tree: ProcessTree,
    armed: bool,
}

impl ProcessTreeGuard {
    fn attach(child: &Child) -> Result<Self, ToolError> {
        Ok(Self {
            tree: ProcessTree::attach(child)?,
            armed: true,
        })
    }

    fn terminate(&self, child: &mut Child) -> Result<(), ToolError> {
        self.tree.terminate(child)
    }

    fn kill_best_effort(&self) {
        self.tree.kill_best_effort(None);
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ProcessTreeGuard {
    fn drop(&mut self) {
        if self.armed {
            // Drop cannot await. The synchronous group/Job kill is the
            // ownership backstop for an aborted bash future.
            self.tree.kill_best_effort(None);
        }
    }
}

async fn terminate_child(
    child: &mut Child,
    process_tree: &ProcessTreeGuard,
) -> Result<(), ToolError> {
    // Cleanup races with natural exit. In that case the later status poll will
    // still trigger best-effort descendant cleanup, but termination itself is
    // not an infrastructure failure.
    if child
        .try_wait()
        .map_err(|error| {
            ToolError::Failed(format!("could not inspect command during cleanup: {error}"))
        })?
        .is_some()
    {
        return Ok(());
    }
    process_tree.terminate(child)?;
    tokio::time::sleep(TERMINATION_GRACE).await;
    process_tree.kill_best_effort();
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
        // SAFETY: null attributes and name request a private unnamed job;
        // the returned handle is checked before it is stored or used.
        let handle = unsafe { CreateJobObjectW(std::ptr::null_mut(), std::ptr::null()) };
        if handle.is_null() {
            return Err(ToolError::Failed(format!(
                "could not create Windows job object: {}",
                std::io::Error::last_os_error()
            )));
        }
        let mut limits = JobObjectExtendedLimitInformation::default();
        limits.basic_limit_information.limit_flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        // SAFETY: handle is a successfully created job and limits points to a
        // live, correctly sized C-compatible structure for this call.
        let configured = unsafe {
            SetInformationJobObject(
                handle,
                JOBOBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
                &mut limits as *mut _ as *mut std::ffi::c_void,
                std::mem::size_of::<JobObjectExtendedLimitInformation>() as u32,
            )
        } != 0;
        if !configured {
            // SAFETY: handle was returned by CreateJobObjectW and has not been
            // closed on this failure path.
            unsafe {
                CloseHandle(handle);
            }
            return Err(ToolError::Failed(format!(
                "could not configure Windows job object: {}",
                std::io::Error::last_os_error()
            )));
        }
        let process_handle = child.raw_handle().ok_or_else(|| {
            // SAFETY: handle is still owned exclusively by this setup path.
            unsafe {
                CloseHandle(handle);
            }
            ToolError::Failed("shell process exited before job assignment".to_owned())
        })?;
        // SAFETY: both handles are valid for the duration of this call;
        // process_handle is borrowed from Tokio's live child.
        if unsafe { AssignProcessToJobObject(handle, process_handle) } == 0 {
            let error = std::io::Error::last_os_error();
            // SAFETY: handle remains exclusively owned on this failure path.
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
        // SAFETY: self.handle is closed only by Drop and remains valid while
        // this borrowed WindowsJob is used.
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
        // SAFETY: this handle is created and owned by WindowsJob; Drop is the
        // single close path after all explicit setup failures have closed it.
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
    // SAFETY: the snapshot is created for thread enumeration and is checked
    // before any enumeration or close operation.
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
    // SAFETY: snapshot is valid and entry points to the initialized structure
    // owned by this function.
    let mut found = unsafe { Thread32First(snapshot, &mut entry) } != 0;
    while found {
        if entry.th32_owner_process_id == process_id {
            thread_id = Some(entry.th32_thread_id);
            break;
        }
        // SAFETY: snapshot and entry remain valid for the enumeration.
        found = unsafe { Thread32Next(snapshot, &mut entry) } != 0;
    }
    // SAFETY: snapshot was successfully created and is closed exactly once.
    unsafe {
        CloseHandle(snapshot);
    }

    let thread_id = thread_id.ok_or_else(|| {
        ToolError::Failed("could not find suspended shell primary thread".to_owned())
    })?;
    // SAFETY: the thread ID came from the snapshot and the returned handle is
    // checked before use.
    let thread = unsafe { OpenThread(THREAD_SUSPEND_RESUME, 0, thread_id) };
    if thread.is_null() {
        return Err(ToolError::Failed(format!(
            "could not open suspended shell primary thread: {}",
            std::io::Error::last_os_error()
        )));
    }
    // SAFETY: thread is a valid handle returned by OpenThread and remains
    // open for this resume call.
    let resumed = unsafe { ResumeThread(thread) };
    // SAFETY: thread is owned by this function and is closed exactly once.
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
    fn getpgrp() -> i32;
    fn kill(pid: i32, signal: i32) -> i32;
    fn setpgid(pid: i32, process_group: i32) -> i32;
}

#[cfg(unix)]
fn configure_process_group(command: &mut Command) {
    // SAFETY: pre_exec runs after fork and before exec in the child. The call
    // uses pid 0 and cannot change ri's parent process group.
    unsafe {
        command.pre_exec(|| {
            // SAFETY: this is the child process calling setpgid on itself.
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

struct TailTruncation {
    content: String,
    output_lines: usize,
    by_bytes: bool,
    last_line_partial: bool,
}

/// Keeps the last lines of `text` that fit both limits, never splitting a
/// line unless a single final line alone exceeds the byte limit.
fn truncate_tail(text: &str, max_lines: usize, max_bytes: usize) -> TailTruncation {
    let mut lines: Vec<&str> = text.split('\n').collect();
    if text.is_empty() || text.ends_with('\n') {
        lines.pop();
    }
    let mut selected: Vec<&str> = Vec::new();
    let mut bytes = 0usize;
    let mut by_bytes = false;
    let mut last_line_partial = false;
    for line in lines.iter().rev() {
        if selected.len() >= max_lines {
            break;
        }
        let cost = line.len() + usize::from(!selected.is_empty());
        if bytes + cost > max_bytes {
            by_bytes = true;
            if selected.is_empty() {
                selected.push(suffix_at_byte_boundary(line, max_bytes));
                last_line_partial = true;
            }
            break;
        }
        selected.push(line);
        bytes += cost;
    }
    selected.reverse();
    TailTruncation {
        output_lines: selected.len(),
        content: selected.join("\n"),
        by_bytes,
        last_line_partial,
    }
}

fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// Merged stdout/stderr in arrival order. Everything is buffered until a limit
/// is crossed; from then on the full output goes to a spill file and only a
/// rolling tail stays in memory.
struct BashOutput {
    tail: String,
    total_bytes: usize,
    newlines: usize,
    open_line: bool,
    current_line_bytes: usize,
    spill: Option<File>,
    spill_path: Option<std::path::PathBuf>,
}

impl BashOutput {
    fn new() -> Self {
        Self {
            tail: String::new(),
            total_bytes: 0,
            newlines: 0,
            open_line: false,
            current_line_bytes: 0,
            spill: None,
            spill_path: None,
        }
    }

    fn total_lines(&self) -> usize {
        self.newlines + usize::from(self.open_line)
    }

    fn push(&mut self, text: &str) -> Result<(), ToolError> {
        if text.is_empty() {
            return Ok(());
        }
        self.total_bytes = self.total_bytes.saturating_add(text.len());
        self.newlines += text.matches('\n').count();
        self.open_line = !text.ends_with('\n');
        self.current_line_bytes = match text.rfind('\n') {
            Some(index) => text.len() - index - 1,
            None => self.current_line_bytes.saturating_add(text.len()),
        };
        self.tail.push_str(text);

        if let Some(spill) = &mut self.spill {
            spill.write_all(text.as_bytes()).map_err(|error| {
                ToolError::Failed(format!("could not append output spill file: {error}"))
            })?;
            if self.tail.len() > TAIL_RETAINED_BYTES * 2 {
                self.tail = suffix_at_byte_boundary(&self.tail, TAIL_RETAINED_BYTES).to_owned();
            }
        } else if self.truncated() {
            self.create_spill()?;
        }
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
        spill.write_all(self.tail.as_bytes()).map_err(|error| {
            ToolError::Failed(format!("could not initialize output spill file: {error}"))
        })?;
        self.spill_path = Some(path);
        self.spill = Some(spill);
        Ok(())
    }

    fn truncated(&self) -> bool {
        self.total_bytes > MAX_BASH_OUTPUT_BYTES || self.total_lines() > MAX_BASH_OUTPUT_LINES
    }

    fn tail_truncation(&self) -> TailTruncation {
        truncate_tail(&self.tail, MAX_BASH_OUTPUT_LINES, MAX_BASH_OUTPUT_BYTES)
    }

    fn truncation_notice(&self, tail: &TailTruncation, path: Option<&std::path::Path>) -> String {
        let total = self.total_lines();
        let start = total.saturating_sub(tail.output_lines) + 1;
        let full = path
            .map(|path| format!("Full output: {}", path.display()))
            .unwrap_or_default();
        if tail.last_line_partial {
            format!(
                "[Showing last {} of line {total} (line is {}). {full}]",
                format_size(tail.content.len()),
                format_size(self.current_line_bytes)
            )
        } else if tail.by_bytes {
            format!(
                "[Showing lines {start}-{total} of {total} ({} limit). {full}]",
                format_size(MAX_BASH_OUTPUT_BYTES)
            )
        } else {
            format!("[Showing lines {start}-{total} of {total}. {full}]")
        }
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
    fn presents_command_without_json_wrapper() {
        let presentation = BashTool.presentation(&json!({
            "command": "cargo test -p ri-core"
        }));

        assert_eq!(presentation.summary, "bash");
        assert_eq!(presentation.preview.len(), 1);
        assert_eq!(presentation.preview[0].text, "cargo test -p ri-core");
        assert_eq!(presentation.preview[0].kind, ToolPreviewKind::Command);
        assert!(!presentation.preview[0].text.contains("\"command\":"));
    }

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
        assert!(result.model_content.contains("out"));
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
    async fn escaped_descendant_cannot_block_reader_drain_forever() {
        let root = unique_test_dir("bash-escaped-descendant");
        fs::create_dir_all(&root).unwrap();
        let pid_path = root.join("escaped.pid");
        let context = ToolContext::new(&root).unwrap();
        let command = format!(
            "setsid sh -c 'sleep 30' & echo $! > \"{}\"; exit 0",
            pid_path.display()
        );

        let execution = tokio::time::timeout(
            Duration::from_secs(1),
            BashTool.execute(
                json!({"command": command}),
                &context,
                mpsc::channel(8).0,
                CancellationToken::new(),
            ),
        )
        .await;

        if let Some(pid) = fs::read_to_string(&pid_path)
            .ok()
            .and_then(|pid| pid.trim().parse::<i32>().ok())
        {
            // The escaped command creates its own process group, so clean up
            // the deliberately long-lived fixture after the assertion.
            unsafe {
                let _ = kill(-pid, SIGKILL);
            }
        }
        let result = execution
            .expect("escaped descendants must not block command completion")
            .unwrap();
        assert!(result.metadata.success);
        remove_test_dir(root);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn normal_completion_kills_a_background_descendant_that_keeps_pipes_open() {
        let root = unique_test_dir("bash-normal-tree");
        fs::create_dir_all(&root).unwrap();
        let marker = root.join("marker");
        let context = ToolContext::new(&root).unwrap();
        let command = format!(
            "printf x >> \"{}\"; (while :; do printf x >> \"{}\"; sleep 0.02; done) & exit 0",
            marker.display(),
            marker.display()
        );

        let result = BashTool
            .execute(
                json!({"command": command}),
                &context,
                mpsc::channel(8).0,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(result.metadata.success);
        wait_for_marker(&marker).await;
        let size = fs::metadata(&marker).unwrap().len();
        tokio::time::sleep(Duration::from_millis(120)).await;
        assert_eq!(fs::metadata(&marker).unwrap().len(), size);
        remove_test_dir(root);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn timeout_kills_a_term_ignoring_process_tree() {
        let root = unique_test_dir("bash-term-ignore");
        fs::create_dir_all(&root).unwrap();
        let marker = root.join("marker");
        let context = ToolContext::new(&root).unwrap();
        let command = format!(
            "trap '' TERM; while :; do printf x >> \"{}\"; sleep 0.02; done",
            marker.display()
        );

        let result = BashTool
            .execute(
                json!({"command": command, "timeout_ms": 50}),
                &context,
                mpsc::channel(8).0,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(result.metadata.timed_out);
        wait_for_marker(&marker).await;
        let size = fs::metadata(&marker).unwrap().len();
        tokio::time::sleep(Duration::from_millis(120)).await;
        assert_eq!(fs::metadata(&marker).unwrap().len(), size);
        remove_test_dir(root);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn dropping_execution_future_kills_the_owned_process_tree() {
        let root = unique_test_dir("bash-drop-tree");
        fs::create_dir_all(&root).unwrap();
        let marker = root.join("marker");
        let context = ToolContext::new(&root).unwrap();
        let command = format!(
            "sh -c 'while :; do printf x >> \"{}\"; sleep 0.02; done'",
            marker.display()
        );
        let task = tokio::spawn(async move {
            BashTool
                .execute(
                    json!({"command": command}),
                    &context,
                    mpsc::channel(8).0,
                    CancellationToken::new(),
                )
                .await
        });
        wait_for_marker(&marker).await;
        task.abort();
        let _ = task.await;

        let size = fs::metadata(&marker).unwrap().len();
        tokio::time::sleep(Duration::from_millis(120)).await;
        assert_eq!(fs::metadata(&marker).unwrap().len(), size);
        remove_test_dir(root);
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

    #[cfg(windows)]
    #[tokio::test]
    async fn dropping_execution_future_kills_windows_descendants() {
        let root = unique_test_dir("bash-windows-drop");
        fs::create_dir_all(&root).unwrap();
        let context = ToolContext::new(&root).unwrap();
        let task = tokio::spawn(async move {
            BashTool
                .execute(
                    json!({"command":"start \"\" /B cmd /C \"ping -n 6 127.0.0.1 >NUL & echo leaked > descendant.txt\" & ping -n 6 127.0.0.1 >NUL"}),
                    &context,
                    mpsc::channel(8).0,
                    CancellationToken::new(),
                )
                .await
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        task.abort();
        let _ = task.await;
        assert!(!root.join("descendant.txt").exists());
        remove_test_dir(root).await;
    }

    #[cfg(unix)]
    async fn run_bash(name: &str, command: String) -> ToolExecutionResult {
        let root = unique_test_dir(name);
        fs::create_dir_all(&root).unwrap();
        let context = ToolContext::new(&root).unwrap();
        let result = BashTool
            .execute(
                json!({ "command": command }),
                &context,
                mpsc::channel(128).0,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        fs::remove_dir_all(root).unwrap();
        result
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn small_output_is_returned_whole_without_a_spill_file() {
        let result = run_bash("bash-small", "seq 1 100".to_owned()).await;

        assert!(!result.metadata.truncated);
        assert!(result.metadata.full_output_path.is_none());
        assert!(result.model_content.contains("\n1\n2\n"));
        assert!(result.model_content.ends_with("\n100"));
        assert!(!result.model_content.contains("Full output"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn line_limit_keeps_tail_and_spills_full_output() {
        let total = MAX_BASH_OUTPUT_LINES + 500;
        let result = run_bash("bash-lines", format!("seq 1 {total}")).await;

        assert!(result.metadata.truncated);
        let path = result.metadata.full_output_path.clone().unwrap();
        let spilled = fs::read_to_string(&path).unwrap();
        assert_eq!(spilled.lines().count(), total);
        assert!(result.model_content.contains(&format!("\n{total}\n")));
        assert!(!result.model_content.contains("\n1\n"));
        let start = total - MAX_BASH_OUTPUT_LINES + 1;
        assert!(result.model_content.contains(&format!(
            "[Showing lines {start}-{total} of {total}. Full output: {}]",
            path.display()
        )));
        fs::remove_file(path).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn byte_limit_bounds_what_the_model_receives() {
        let size = 10 * 1024 * 1024;
        let line = "a".repeat(99);
        let result = run_bash("bash-bytes", format!("yes {line} | head -c {size}")).await;

        assert!(result.metadata.truncated);
        assert!(result.model_content.len() < MAX_BASH_OUTPUT_BYTES + 1024);
        assert!(result.model_content.contains("50.0KB limit"));
        let path = result.metadata.full_output_path.unwrap();
        assert_eq!(fs::metadata(&path).unwrap().len(), size as u64);
        fs::remove_file(path).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn stderr_is_merged_and_counts_toward_the_limit() {
        let size = MAX_BASH_OUTPUT_BYTES * 2;
        let result = run_bash("bash-stderr", format!("yes x | head -c {size} 1>&2")).await;

        assert!(result.metadata.truncated);
        assert!(result.model_content.len() < MAX_BASH_OUTPUT_BYTES + 1024);
        let path = result.metadata.full_output_path.unwrap();
        assert_eq!(fs::metadata(&path).unwrap().len(), size as u64);
        fs::remove_file(path).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn oversized_final_line_is_clipped_to_its_end() {
        let size = MAX_BASH_OUTPUT_BYTES * 2;
        let result = run_bash(
            "bash-long-line",
            format!("head -c {size} /dev/zero | tr '\\0' 'a'; printf END"),
        )
        .await;

        assert!(result.metadata.truncated);
        assert!(result
            .model_content
            .contains("END\n\n[Showing last 50.0KB of line 1"));
        assert!(result.model_content.len() < MAX_BASH_OUTPUT_BYTES + 1024);
        fs::remove_file(result.metadata.full_output_path.unwrap()).unwrap();
    }

    #[test]
    fn truncate_tail_respects_both_limits_and_line_boundaries() {
        let text = "one\ntwo\nthree\nfour\n";
        let by_lines = truncate_tail(text, 2, 1024);
        assert_eq!(by_lines.content, "three\nfour");
        assert!(!by_lines.by_bytes);

        let by_bytes = truncate_tail(text, 10, 10);
        assert_eq!(by_bytes.content, "three\nfour");
        assert!(by_bytes.by_bytes && !by_bytes.last_line_partial);

        let partial = truncate_tail("abcdefghij", 10, 4);
        assert_eq!(partial.content, "ghij");
        assert!(partial.last_line_partial);
    }

    #[cfg(unix)]
    async fn wait_for_marker(path: &std::path::Path) {
        for _ in 0..200 {
            if path.exists() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("marker was not created: {}", path.display());
    }

    #[cfg(windows)]
    async fn remove_test_dir(path: PathBuf) {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            match fs::remove_dir_all(&path) {
                Ok(()) => return,
                Err(error)
                    if matches!(error.raw_os_error(), Some(5 | 32))
                        && Instant::now() < deadline =>
                {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                Err(error) => panic!(
                    "could not remove test directory {}: {error}",
                    path.display()
                ),
            }
        }
    }

    #[cfg(unix)]
    fn remove_test_dir(path: PathBuf) {
        let _ = fs::remove_dir_all(path);
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
