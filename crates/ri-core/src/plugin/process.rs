use std::collections::HashMap;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::StreamExt;
use serde_json::Value;
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::{mpsc, oneshot, Mutex as AsyncMutex, Semaphore};
use tokio::task::JoinHandle;
use tokio_util::codec::{FramedRead, LinesCodec};
use tokio_util::sync::CancellationToken;

use super::manifest::{LoadedPluginManifest, PluginManifest, PluginManifestError};
use super::protocol::*;
use tokio::process::{Child, Command};
use tokio::time::timeout;

const PLUGIN_NOTIFICATION_CHANNEL_CAPACITY: usize = 64;
const MAX_PENDING_REQUESTS: usize = 64;

#[derive(Debug, Error)]
pub enum PluginProcessError {
    #[error("invalid plugin manifest: {0}")]
    Manifest(#[from] PluginManifestError),
    #[error("could not spawn plugin: {0}")]
    Spawn(std::io::Error),
    #[error("plugin shutdown timed out")]
    ShutdownTimeout,
    #[error("plugin exited: {0}")]
    Exited(std::process::ExitStatus),
    #[error("plugin initialization timed out")]
    StartupTimeout,
    #[error("plugin protocol mismatch: {0}")]
    ProtocolMismatch(String),
    #[error("plugin identity mismatch for {field}: expected {expected:?}, received {actual:?}")]
    IdentityMismatch {
        field: &'static str,
        expected: String,
        actual: String,
    },
    #[error("{source}; stderr: {stderr:?}")]
    Diagnostics {
        source: Box<PluginProcessError>,
        stderr: PluginDiagnostics,
    },
    #[error("could not reap plugin: {0}")]
    Reap(std::io::Error),
    #[error("plugin transport: {0}")]
    Transport(String),
    #[error("plugin remote error: {0:?}")]
    RemoteError(RpcErrorObject),
}

pub const PLUGIN_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);
pub const PLUGIN_STARTUP_TIMEOUT: Duration = Duration::from_secs(5);
pub const MAX_PLUGIN_STDERR_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, Default)]
pub struct PluginDiagnostics {
    pub text: String,
    pub truncated: bool,
}

#[derive(Default)]
struct StderrCapture {
    bytes: Vec<u8>,
    truncated: bool,
}

impl StderrCapture {
    fn append(&mut self, bytes: &[u8]) {
        let retained = bytes.len().min(MAX_PLUGIN_STDERR_BYTES - self.bytes.len());
        self.bytes.extend_from_slice(&bytes[..retained]);
        self.truncated |= retained < bytes.len();
    }

    fn snapshot(&self) -> PluginDiagnostics {
        let mut text = String::from_utf8_lossy(&self.bytes).into_owned();
        let mut end = text.len().min(MAX_PLUGIN_STDERR_BYTES);
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
        PluginDiagnostics {
            text,
            truncated: self.truncated || end < String::from_utf8_lossy(&self.bytes).len(),
        }
    }
}

pub struct PluginProcess {
    manifest: PluginManifest,
    capabilities: PluginCapabilities,
    child: Child,
    transport: Transport,
    stderr: Arc<Mutex<StderrCapture>>,
    stderr_task: JoinHandle<()>,
}

impl PluginProcess {
    /// Explicitly executes the manifest command; this is not a sandbox or discovery API.
    pub async fn start(loaded: LoadedPluginManifest) -> Result<Self, PluginProcessError> {
        loaded.manifest.validate(&loaded.path)?;
        let entrypoint = &loaded.manifest.entrypoint;
        let command_path = std::path::Path::new(&entrypoint.command);
        // Resolve relative executable paths before changing the child's working directory.
        let command = if command_path.is_relative() && command_path.components().count() > 1 {
            loaded.directory.join(command_path)
        } else {
            command_path.to_owned()
        };
        let mut child = Command::new(command)
            .args(&entrypoint.args)
            .current_dir(&loaded.directory)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(PluginProcessError::Spawn)?;
        let mut stderr_pipe = child.stderr.take().expect("piped stderr");
        let stderr = Arc::new(Mutex::new(StderrCapture::default()));
        let capture = stderr.clone();
        let stderr_task = tokio::spawn(async move {
            let mut buffer = [0; 8192];
            loop {
                match stderr_pipe.read(&mut buffer).await {
                    Ok(0) | Err(_) => break,
                    Ok(count) => capture.lock().unwrap().append(&buffer[..count]),
                }
            }
        });
        let transport = Transport::new(
            child.stdout.take().expect("piped stdout"),
            child.stdin.take().expect("piped stdin"),
        );
        let mut process = Self {
            manifest: loaded.manifest,
            capabilities: PluginCapabilities::default(),
            child,
            transport,
            stderr,
            stderr_task,
        };
        let initialized = timeout(PLUGIN_STARTUP_TIMEOUT, process.initialize()).await;
        let error = match initialized {
            Ok(Ok(())) => return Ok(process),
            Ok(Err(error)) => error,
            Err(_) => PluginProcessError::StartupTimeout,
        };
        Err(process.cleanup_error(error).await)
    }

    async fn initialize(&mut self) -> Result<(), PluginProcessError> {
        let params = InitializeParams {
            protocol_version: PLUGIN_PROTOCOL_VERSION.into(),
            host: HostIdentity {
                name: "ri".into(),
                version: env!("CARGO_PKG_VERSION").into(),
            },
        };
        let result = self
            .transport
            .request(
                "initialize",
                serde_json::to_value(params).expect("serializable initialize params"),
            )
            .await?;
        let result: InitializeResult = serde_json::from_value(result)
            .map_err(|error| PluginProcessError::Transport(error.to_string()))?;
        if result.protocol_version != PLUGIN_PROTOCOL_VERSION {
            return Err(PluginProcessError::ProtocolMismatch(
                result.protocol_version,
            ));
        }
        for (field, expected, actual) in [
            ("id", &self.manifest.id, result.plugin.id),
            ("name", &self.manifest.name, result.plugin.name),
            ("version", &self.manifest.version, result.plugin.version),
        ] {
            if *expected != actual {
                return Err(PluginProcessError::IdentityMismatch {
                    field,
                    expected: expected.clone(),
                    actual,
                });
            }
        }
        self.capabilities = result.capabilities;
        Ok(())
    }

    pub fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }
    pub fn capabilities(&self) -> &PluginCapabilities {
        &self.capabilities
    }
    pub fn diagnostics(&self) -> PluginDiagnostics {
        self.stderr.lock().unwrap().snapshot()
    }

    pub(crate) fn client(&self) -> PluginClient {
        self.transport.client.clone()
    }

    /// Sends a generic request. Callers choose their own post-startup request deadline.
    pub async fn request(&self, method: &str, params: Value) -> Result<Value, PluginProcessError> {
        self.transport.request(method, params).await
    }

    pub async fn recv_notification(&self) -> Option<RpcNotification> {
        self.transport.notifications.lock().await.recv().await
    }

    pub async fn shutdown(mut self) -> Result<(), PluginProcessError> {
        let graceful = timeout(PLUGIN_SHUTDOWN_TIMEOUT, async {
            if let Some(status) = self.child.try_wait().map_err(PluginProcessError::Reap)? {
                return Err(PluginProcessError::Exited(status));
            }
            self.transport
                .request("shutdown", serde_json::json!({}))
                .await?;
            self.transport.close().await;
            let status = self.child.wait().await.map_err(PluginProcessError::Reap)?;
            if !status.success() {
                return Err(PluginProcessError::Exited(status));
            }
            Ok(())
        })
        .await;
        match graceful {
            Ok(Ok(())) => {
                self.finish_stderr().await;
                Ok(())
            }
            Ok(Err(error)) => Err(self.cleanup_error(error).await),
            Err(_) => Err(self
                .cleanup_error(PluginProcessError::ShutdownTimeout)
                .await),
        }
    }

    async fn finish_stderr(&mut self) {
        // A descendant may inherit stderr; never wait indefinitely for pipe EOF.
        if timeout(Duration::from_millis(100), &mut self.stderr_task)
            .await
            .is_err()
        {
            self.stderr_task.abort();
            let _ = (&mut self.stderr_task).await;
        }
    }

    async fn cleanup_error(&mut self, error: PluginProcessError) -> PluginProcessError {
        self.transport
            .client
            .state
            .dispatch
            .lock()
            .unwrap()
            .fail("plugin stdin closed".into());
        self.transport.reader.abort();
        self.transport.writer.abort();
        let _ = self.child.start_kill();
        let error = match self.child.wait().await {
            Ok(_) => error,
            Err(error) => PluginProcessError::Reap(error),
        };
        self.finish_stderr().await;
        PluginProcessError::Diagnostics {
            source: Box::new(error),
            stderr: self.diagnostics(),
        }
    }
}

impl Drop for PluginProcess {
    fn drop(&mut self) {
        self.stderr_task.abort();
    }
}

type Reply = oneshot::Sender<Result<Value, PluginProcessError>>;

#[derive(Default)]
struct Dispatch {
    pending: HashMap<u64, Reply>,
    failure: Option<String>,
    next_id: u64,
}

impl Dispatch {
    fn fail(&mut self, reason: String) {
        if self.failure.is_none() {
            self.failure = Some(reason.clone());
        }
        for (_, sender) in self.pending.drain() {
            let _ = sender.send(Err(PluginProcessError::Transport(reason.clone())));
        }
    }
}

struct PendingRequest {
    id: u64,
    dispatch: Arc<Mutex<Dispatch>>,
}

impl Drop for PendingRequest {
    fn drop(&mut self) {
        self.dispatch.lock().unwrap().pending.remove(&self.id);
    }
}

enum WriterCommand {
    Frame(String),
    Close,
}

struct ClientState {
    outgoing: mpsc::Sender<WriterCommand>,
    dispatch: Arc<Mutex<Dispatch>>,
    permits: Semaphore,
}

#[derive(Clone)]
pub(crate) struct PluginClient {
    state: Arc<ClientState>,
}

struct Transport {
    client: PluginClient,
    notifications: AsyncMutex<mpsc::Receiver<RpcNotification>>,
    reader: JoinHandle<()>,
    writer: JoinHandle<()>,
}

impl Transport {
    fn new<R, W>(read: R, mut write: W) -> Self
    where
        R: AsyncRead + Unpin + Send + 'static,
        W: AsyncWrite + Unpin + Send + 'static,
    {
        let dispatch = Arc::new(Mutex::new(Dispatch {
            next_id: 1,
            ..Default::default()
        }));
        let (notifications_tx, notifications) = mpsc::channel(PLUGIN_NOTIFICATION_CHANNEL_CAPACITY);
        let (outgoing, mut outgoing_rx) = mpsc::channel::<WriterCommand>(MAX_PENDING_REQUESTS);
        let writer_failed = CancellationToken::new();
        let failure_signal = writer_failed.clone();
        let state = dispatch.clone();
        let writer = tokio::spawn(async move {
            while let Some(command) = outgoing_rx.recv().await {
                let WriterCommand::Frame(line) = command else {
                    state.lock().unwrap().fail("plugin stdin closed".into());
                    break;
                };
                if let Err(error) = write.write_all(line.as_bytes()).await {
                    state.lock().unwrap().fail(error.to_string());
                    failure_signal.cancel();
                    return;
                }
            }
            let _ = write.shutdown().await;
        });
        let writer_abort = writer.abort_handle();
        let state = dispatch.clone();
        let reader = tokio::spawn(async move {
            let mut frames = FramedRead::new(
                read,
                LinesCodec::new_with_max_length(MAX_PLUGIN_FRAME_BYTES),
            );
            let reason = loop {
                let frame = tokio::select! {
                    frame = frames.next() => frame,
                    _ = writer_failed.cancelled() => break "plugin stdin writer failed".into(),
                };
                let message = match frame {
                    Some(Ok(line)) => match decode_plugin_message(&line) {
                        Ok(message) => message,
                        Err(error) => break error.to_string(),
                    },
                    Some(Err(error)) => break error.to_string(),
                    None => break "plugin stdout closed".into(),
                };
                match message {
                    PluginMessage::Response(response) => {
                        let sender = state.lock().unwrap().pending.remove(&response.id);
                        // Late responses to cancelled requests are harmless.
                        if let Some(sender) = sender {
                            let result = match response.error {
                                Some(error) => Err(PluginProcessError::RemoteError(error)),
                                None => Ok(response.result.expect("validated response")),
                            };
                            let _ = sender.send(result);
                        }
                    }
                    PluginMessage::Notification(notification) => {
                        // Fail closed rather than block response routing or silently lose events.
                        if notifications_tx.try_send(notification).is_err() {
                            break "plugin notification queue full or closed".into();
                        }
                    }
                }
            };
            state.lock().unwrap().fail(reason);
            writer_abort.abort();
        });
        Self {
            client: PluginClient {
                state: Arc::new(ClientState {
                    outgoing,
                    dispatch,
                    permits: Semaphore::new(MAX_PENDING_REQUESTS),
                }),
            },
            notifications: AsyncMutex::new(notifications),
            reader,
            writer,
        }
    }

    async fn close(&self) {
        self.client
            .state
            .dispatch
            .lock()
            .unwrap()
            .fail("plugin stdin closed".into());
        let _ = self.client.state.outgoing.send(WriterCommand::Close).await;
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value, PluginProcessError> {
        self.client.request(method, params).await
    }
}

impl PluginClient {
    pub(crate) async fn request(
        &self,
        method: &str,
        params: Value,
    ) -> Result<Value, PluginProcessError> {
        if let Some(reason) = &self.state.dispatch.lock().unwrap().failure {
            return Err(PluginProcessError::Transport(reason.clone()));
        }
        let _permit = self
            .state
            .permits
            .acquire()
            .await
            .map_err(|error| PluginProcessError::Transport(error.to_string()))?;
        let (sender, receiver) = oneshot::channel();
        let id = {
            let mut state = self.state.dispatch.lock().unwrap();
            if let Some(reason) = &state.failure {
                return Err(PluginProcessError::Transport(reason.clone()));
            }
            let id = state.next_id;
            state.next_id = id
                .checked_add(1)
                .ok_or_else(|| PluginProcessError::Transport("request IDs exhausted".into()))?;
            state.pending.insert(id, sender);
            id
        };
        let _pending = PendingRequest {
            id,
            dispatch: self.state.dispatch.clone(),
        };
        let mut line = encode_request(id, method, params)
            .map_err(|error| PluginProcessError::Transport(error.to_string()))?;
        line.push('\n');
        self.state
            .outgoing
            .send(WriterCommand::Frame(line))
            .await
            .map_err(|_| PluginProcessError::Transport("stdin writer closed".into()))?;
        receiver
            .await
            .map_err(|_| PluginProcessError::Transport("response reader closed".into()))?
    }
}

impl Drop for Transport {
    fn drop(&mut self) {
        self.client
            .state
            .dispatch
            .lock()
            .unwrap()
            .fail("plugin stdin closed".into());
        self.reader.abort();
        self.writer.abort();
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use tokio::io::{AsyncBufReadExt, BufReader};

    pub(crate) struct Fixture {
        directory: std::path::PathBuf,
    }

    impl Fixture {
        pub(crate) fn scripted(tools: bool, exchanges: Vec<(&str, Value, Value)>) -> Self {
            let mut initialize = initialize_result();
            initialize["capabilities"]["tools"] = Value::Bool(tools);
            let fixture = Self::new(initialize.clone(), "", false);
            fixture.change_script(|_| {
                let mut exchanges = exchanges;
                exchanges.insert(0, ("initialize", serde_json::to_value(InitializeParams {
                    protocol_version: PLUGIN_PROTOCOL_VERSION.into(),
                    host: HostIdentity { name: "ri".into(), version: env!("CARGO_PKG_VERSION").into() },
                }).unwrap(), serde_json::json!({"result": initialize})));
                exchanges.push(("shutdown", serde_json::json!({}), serde_json::json!({"result":null})));
                #[cfg(unix)]
                let mut script = String::new();
                #[cfg(windows)]
                let mut script = String::from("@echo off\r\n");
                for (index, (method, params, mut response)) in exchanges.into_iter().enumerate() {
                    let id = index as u64 + 1;
                    let request = encode_request(id, method, params).unwrap();
                    response["jsonrpc"] = Value::String("2.0".into());
                    response["id"] = Value::from(id);
                    #[cfg(unix)]
                    script.push_str(&format!("IFS= read -r request\n[ \"$request\" = '{request}' ] || exit 7\nprintf '%s\\n' '{response}'\n"));
                    #[cfg(windows)]
                    script.push_str(&format!("set /p REQUEST=\r\nif not \"%REQUEST%\"==\"{request}\" exit /b 7\r\necho {response}\r\n"));
                }
                #[cfg(windows)]
                script.push_str("exit /b 0\r\n");
                script
            });
            fixture
        }

        fn new(result: Value, diagnostic: &str, stall: bool) -> Self {
            static NEXT_FIXTURE: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            let directory = std::env::temp_dir().join(format!(
                "ri-plugin-{}-{}-{}",
                std::process::id(),
                NEXT_FIXTURE.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir(&directory).unwrap();
            let response = serde_json::json!({"jsonrpc":"2.0","id":1,"result":result}).to_string();
            #[cfg(unix)]
            let (command, args, name, script) = ("/bin/sh", vec!["fixture.sh"], "fixture.sh", format!("IFS= read -r initialize\nprintf '%s\\n' '{diagnostic}' >&2\n{}\nprintf '%s\\n' '{response}'\nIFS= read -r shutdown\nprintf '%s\\n' '{{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":null}}'\n", if stall { "while :; do :; done" } else { "" }));
            #[cfg(windows)]
            let (command, args, name, script) = ("cmd.exe", vec!["/C", "fixture.cmd"], "fixture.cmd", format!("@echo off\r\nset /p INITIALIZE=\r\necho {diagnostic} 1>&2\r\n{}\r\necho {response}\r\nset /p SHUTDOWN=\r\necho {{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":null}}\r\nexit /b 0\r\n", if stall { ":stall\r\ngoto stall" } else { "" }));
            std::fs::write(directory.join(name), script).unwrap();
            std::fs::write(directory.join("plugin.json"), serde_json::json!({"manifestVersion":1,"id":"test.echo","name":"Echo","version":"0.1.0","protocolVersion":"ri.plugin.v1","entrypoint":{"command":command,"args":args}}).to_string()).unwrap();
            Self { directory }
        }

        pub(crate) fn load(&self) -> LoadedPluginManifest {
            super::super::manifest::load_plugin_manifest(self.directory.join("plugin.json"))
                .unwrap()
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.directory).unwrap();
        }
    }

    fn initialize_result() -> Value {
        serde_json::json!({"protocolVersion":"ri.plugin.v1","plugin":{"id":"test.echo","name":"Echo","version":"0.1.0"},"capabilities":{"tools":false,"future":true}})
    }

    impl Fixture {
        pub(crate) fn change_script(&self, change: impl FnOnce(String) -> String) {
            #[cfg(unix)]
            let name = "fixture.sh";
            #[cfg(windows)]
            let name = "fixture.cmd";
            let path = self.directory.join(name);
            std::fs::write(&path, change(std::fs::read_to_string(&path).unwrap())).unwrap();
        }
    }

    #[tokio::test]
    async fn closing_full_client_rejects_new_requests_without_waiting_for_permits() {
        let (host, _peer) = tokio::io::duplex(65536);
        let (read, write) = tokio::io::split(host);
        let transport = Transport::new(read, write);
        let client = transport.client.clone();
        let mut pending = Vec::new();
        for _ in 0..MAX_PENDING_REQUESTS {
            let mut request = Box::pin(client.request("pending", Value::Null));
            assert!(futures_util::poll!(&mut request).is_pending());
            pending.push(request);
        }
        assert_eq!(
            client.state.dispatch.lock().unwrap().pending.len(),
            MAX_PENDING_REQUESTS
        );
        let mut extra = Box::pin(client.request("extra", Value::Null));
        assert!(futures_util::poll!(&mut extra).is_pending());
        assert_eq!(client.state.dispatch.lock().unwrap().next_id, 65);
        transport.close().await;
        assert!(matches!(
            timeout(
                Duration::from_secs(1),
                client.request("closed", Value::Null)
            )
            .await
            .unwrap(),
            Err(PluginProcessError::Transport(_))
        ));
        drop(pending);
        assert!(matches!(extra.await, Err(PluginProcessError::Transport(_))));
        assert!(client.state.dispatch.lock().unwrap().pending.is_empty());
    }

    #[tokio::test]
    async fn cloned_client_and_transport_share_request_ids() {
        let (host, peer) = tokio::io::duplex(4096);
        let (read, write) = tokio::io::split(host);
        let transport = Transport::new(read, write);
        let client = transport.client.clone();
        let peer_task = tokio::spawn(async move {
            let (read, mut write) = tokio::io::split(peer);
            let mut lines = BufReader::new(read).lines();
            for id in 1..=3 {
                let request: Value =
                    serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
                assert_eq!(request["id"], id);
                write
                    .write_all(
                        format!("{{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{id}}}\n").as_bytes(),
                    )
                    .await
                    .unwrap();
            }
            assert!(lines.next_line().await.unwrap().is_none());
        });
        assert_eq!(transport.request("a", Value::Null).await.unwrap(), 1);
        assert_eq!(client.request("b", Value::Null).await.unwrap(), 2);
        assert_eq!(client.clone().request("c", Value::Null).await.unwrap(), 3);
        transport.close().await;
        assert!(matches!(
            client.request("closed", Value::Null).await,
            Err(PluginProcessError::Transport(_))
        ));
        timeout(Duration::from_secs(1), peer_task)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn shutdown_closes_stdin_even_while_client_clones_exist() {
        let fixture = Fixture::new(initialize_result(), "", false);
        fixture.change_script(|script| {
            #[cfg(unix)]
            let script = format!("{script}while IFS= read -r line; do :; done\n");
            #[cfg(windows)]
            let script = script.replace("exit /b 0", "set /p EOF=\r\nexit /b 0");
            script
        });
        let process = PluginProcess::start(fixture.load()).await.unwrap();
        let client = process.client();
        process.shutdown().await.unwrap();
        assert!(matches!(
            client.request("closed", Value::Null).await,
            Err(PluginProcessError::Transport(_))
        ));
    }

    #[tokio::test]
    async fn stdout_failure_resolves_requests_even_when_stdin_is_blocked() {
        let (host, peer) = tokio::io::duplex(1);
        let (read, write) = tokio::io::split(host);
        let transport = Transport::new(read, write);
        let (_peer_read, mut peer_write) = tokio::io::split(peer);
        transport
            .client
            .state
            .outgoing
            .send(WriterCommand::Frame("blocked\n".into()))
            .await
            .unwrap();
        tokio::task::yield_now().await;
        for _ in 0..MAX_PENDING_REQUESTS {
            transport
                .client
                .state
                .outgoing
                .try_send(WriterCommand::Frame("queued\n".into()))
                .unwrap();
        }
        let result = timeout(Duration::from_secs(1), async {
            let (result, _) = tokio::join!(transport.request("test", Value::Null), async {
                tokio::time::sleep(Duration::from_millis(10)).await;
                peer_write.write_all(b"invalid\n").await.unwrap();
            });
            result
        })
        .await
        .expect("stdout failure must unblock requests");
        assert!(matches!(result, Err(PluginProcessError::Transport(_))));
    }

    #[tokio::test]
    async fn startup_remote_error_malformed_output_and_early_exit() {
        for mode in ["remote", "malformed", "exit"] {
            let fixture = Fixture::new(initialize_result(), "failure diagnostic", false);
            fixture.change_script(|script| {
                let response = serde_json::json!({"jsonrpc":"2.0","id":1,"result":initialize_result()}).to_string();
                match mode {
                    "remote" => script.replace(&response, r#"{"jsonrpc":"2.0","id":1,"error":{"code":-1,"message":"initialize failed"}}"#),
                    "malformed" => script.replace(&response, "not JSON"),
                    _ => {
                        #[cfg(unix)]
                        let script = script.replace(&format!("printf '%s\\n' '{response}'"), "exit 0");
                        #[cfg(windows)]
                        let script = script.replace(&format!("echo {response}"), "exit /b 0");
                        script
                    }
                }
            });
            let error = PluginProcess::start(fixture.load()).await.err().unwrap();
            assert!(error.to_string().contains("failure diagnostic"), "{error}");
            if mode == "remote" {
                assert!(
                    matches!(error, PluginProcessError::Diagnostics { source, .. } if matches!(*source, PluginProcessError::RemoteError(_)))
                );
            }
        }
    }

    #[tokio::test]
    async fn stderr_is_bounded_but_continues_draining() {
        let fixture = Fixture::new(initialize_result(), "diagnostic", false);
        fixture.change_script(|script| {
            let text = "x".repeat(1024);
            #[cfg(unix)]
            let noise = format!("printf '%s\\n' '{text}' >&2\n").repeat(128);
            #[cfg(windows)]
            let noise = format!(
                "@echo off\r\n{}",
                format!("echo {text} 1>&2\r\n").repeat(128)
            );
            format!("{noise}{script}")
        });
        let process = PluginProcess::start(fixture.load()).await.unwrap();
        let diagnostics = process.diagnostics();
        assert_eq!(diagnostics.text.len(), MAX_PLUGIN_STDERR_BYTES);
        assert!(diagnostics.truncated);
        process.shutdown().await.unwrap();
    }

    #[test]
    fn non_utf8_diagnostics_remain_bounded() {
        let mut capture = StderrCapture::default();
        capture.append(&vec![0xff; MAX_PLUGIN_STDERR_BYTES + 1]);
        assert_eq!(capture.bytes.len(), MAX_PLUGIN_STDERR_BYTES);
        let snapshot = capture.snapshot();
        assert!(snapshot.text.len() <= MAX_PLUGIN_STDERR_BYTES);
        assert!(snapshot.truncated);
    }

    #[tokio::test]
    async fn notification_overflow_fails_without_blocking_responses() {
        let (host, peer) = tokio::io::duplex(4096);
        let (read, write) = tokio::io::split(host);
        let transport = Transport::new(read, write);
        let peer_task = tokio::spawn(async move {
            let (read, mut write) = tokio::io::split(peer);
            let mut lines = BufReader::new(read).lines();
            lines.next_line().await.unwrap();
            for _ in 0..=PLUGIN_NOTIFICATION_CHANNEL_CAPACITY {
                let _ = write
                    .write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"event\",\"params\":{}}\n")
                    .await;
            }
        });
        let result = timeout(
            Duration::from_secs(1),
            transport.request("test", Value::Null),
        )
        .await
        .unwrap();
        assert!(
            matches!(result, Err(PluginProcessError::Transport(reason)) if reason.contains("queue full"))
        );
        peer_task.await.unwrap();
    }

    #[tokio::test]
    async fn notifications_can_be_drained_while_request_is_pending() {
        let notification_count = PLUGIN_NOTIFICATION_CHANNEL_CAPACITY + 1;
        let fixture = Fixture::new(initialize_result(), "diagnostic", false);
        fixture.change_script(|_| {
            let initialize = serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "result": initialize_result(),
            })
            .to_string();
            let response = serde_json::json!({
                "jsonrpc": "2.0",
                "id": 2,
                "result": 42,
            })
            .to_string();
            let acknowledgement_lines = (0..notification_count)
                .map(|index| {
                    serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": index + 3,
                        "result": null,
                    })
                    .to_string()
                })
                .collect::<Vec<_>>();
            let shutdown = serde_json::json!({
                "jsonrpc": "2.0",
                "id": notification_count + 3,
                "result": null,
            })
            .to_string();
            let notification_lines = (0..notification_count)
                .map(|index| {
                    serde_json::json!({
                        "jsonrpc": "2.0",
                        "method": "event/progress",
                        "params": {"index": index},
                    })
                    .to_string()
                })
                .collect::<Vec<_>>();
            #[cfg(unix)]
            {
                let mut interaction =
                    format!("printf '%s\\n' '{}'\n", notification_lines.first().unwrap());
                for (index, acknowledgement) in acknowledgement_lines.iter().enumerate() {
                    interaction.push_str("IFS= read -r acknowledgement\n");
                    interaction.push_str(&format!("printf '%s\\n' '{acknowledgement}'\n"));
                    if let Some(notification) = notification_lines.get(index + 1) {
                        interaction.push_str(&format!("printf '%s\\n' '{notification}'\n"));
                    }
                }
                format!(
                    "IFS= read -r initialize\nprintf '%s\\n' '{initialize}'\nIFS= read -r request\n{interaction}printf '%s\\n' '{response}'\nIFS= read -r shutdown\nprintf '%s\\n' '{shutdown}'\n"
                )
            }
            #[cfg(windows)]
            {
                let mut interaction = format!("echo {}\r\n", notification_lines.first().unwrap());
                for (index, acknowledgement) in acknowledgement_lines.iter().enumerate() {
                    interaction.push_str("set /p ACK=\r\n");
                    interaction.push_str(&format!("echo {acknowledgement}\r\n"));
                    if let Some(notification) = notification_lines.get(index + 1) {
                        interaction.push_str(&format!("echo {notification}\r\n"));
                    }
                }
                format!(
                    "@echo off\r\nset /p INITIALIZE=\r\necho {initialize}\r\nset /p REQUEST=\r\n{interaction}echo {response}\r\nset /p SHUTDOWN=\r\necho {shutdown}\r\nexit /b 0\r\n"
                )
            }
        });
        let process = PluginProcess::start(fixture.load()).await.unwrap();
        let drain = async {
            for index in 0..notification_count {
                let notification = process.recv_notification().await.unwrap();
                assert_eq!(notification.method, "event/progress");
                assert_eq!(notification.params["index"], index);
                process
                    .request("notification/ack", Value::Null)
                    .await
                    .unwrap();
            }
            notification_count
        };
        let (result, received) = timeout(Duration::from_secs(2), async {
            tokio::join!(process.request("pending", Value::Null), drain)
        })
        .await
        .unwrap();
        assert_eq!(result.unwrap(), 42);
        assert_eq!(received, notification_count);
        process.shutdown().await.unwrap();
    }

    #[cfg(unix)]
    async fn assert_reaped(pid: u32) {
        for _ in 0..100 {
            let status = Command::new("/bin/kill")
                .args(["-0", &pid.to_string()])
                .stderr(Stdio::null())
                .status()
                .await
                .unwrap();
            if !status.success() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("child {pid} still exists");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn dropping_process_kills_child_and_failure_reaps_child() {
        let fixture = Fixture::new(initialize_result(), "diagnostic", false);
        let process = PluginProcess::start(fixture.load()).await.unwrap();
        let pid = process.child.id().unwrap();
        drop(process);
        assert_reaped(pid).await;

        let fixture = Fixture::new(Value::Null, "invalid initialize", false);
        fixture.change_script(|script| format!("echo $$ > plugin.pid\n{script}"));
        assert!(PluginProcess::start(fixture.load()).await.is_err());
        let pid = std::fs::read_to_string(fixture.directory.join("plugin.pid"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert_reaped(pid).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn relative_executable_and_literal_arguments() {
        use std::os::unix::fs::PermissionsExt;
        let fixture = Fixture::new(initialize_result(), "diagnostic", false);
        fixture.change_script(|script| {
            format!("#!/bin/sh\n[ \"$1\" = 'literal;not a shell command' ] || exit 3\n{script}")
        });
        let script = fixture.directory.join("fixture.sh");
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();
        let mut loaded = fixture.load();
        loaded.manifest.entrypoint.command = "./fixture.sh".into();
        loaded.manifest.entrypoint.args = vec!["literal;not a shell command".into()];
        PluginProcess::start(loaded)
            .await
            .unwrap()
            .shutdown()
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn real_process_lifecycle() {
        let fixture = Fixture::new(initialize_result(), "diagnostic", false);
        let process = PluginProcess::start(fixture.load()).await.unwrap();
        assert_eq!(process.manifest().id, "test.echo");
        assert!(!process.capabilities().tools);
        assert_eq!(process.capabilities().extra["future"], true);
        process.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn shutdown_timeout_and_remote_error_clean_up() {
        for remote_error in [false, true] {
            let fixture = Fixture::new(initialize_result(), "shutdown diagnostic", false);
            fixture.change_script(|script| {
                if remote_error {
                    script.replace(
                        "\"id\":2,\"result\":null",
                        "\"id\":2,\"error\":{\"code\":-1,\"message\":\"shutdown failed\"}",
                    )
                } else {
                    #[cfg(unix)]
                    let script = format!("{script}while :; do :; done\n");
                    #[cfg(windows)]
                    let script = script.replace("exit /b 0", ":stall\r\ngoto stall");
                    script
                }
            });
            let process = PluginProcess::start(fixture.load()).await.unwrap();
            let error = process.shutdown().await.unwrap_err();
            match error {
                PluginProcessError::Diagnostics { source, stderr } => {
                    assert!(stderr.text.contains("shutdown diagnostic"));
                    if remote_error {
                        assert!(matches!(*source, PluginProcessError::RemoteError(_)));
                    } else {
                        assert!(matches!(*source, PluginProcessError::ShutdownTimeout));
                    }
                }
                _ => panic!("unexpected {error}"),
            }
        }
    }

    #[tokio::test]
    async fn shutdown_after_exit_is_reported() {
        let fixture = Fixture::new(initialize_result(), "exit diagnostic", false);
        fixture.change_script(|script| {
            #[cfg(unix)]
            let script = script.replace("IFS= read -r shutdown", "exit 0");
            #[cfg(windows)]
            let script = script.replace("set /p SHUTDOWN=", "exit /b 0");
            script
        });
        let mut process = PluginProcess::start(fixture.load()).await.unwrap();
        process.child.wait().await.unwrap();
        assert!(process.shutdown().await.is_err());
    }

    #[tokio::test]
    async fn initialization_verifies_identity_and_protocol() {
        for field in ["id", "name", "version", "protocolVersion"] {
            let mut result = initialize_result();
            if field == "protocolVersion" {
                result[field] = Value::String("wrong".into());
            } else {
                result["plugin"][field] = Value::String("wrong".into());
            }
            let fixture = Fixture::new(result, "startup diagnostic", false);
            let error = PluginProcess::start(fixture.load())
                .await
                .err()
                .expect("must reject mismatch");
            assert!(error.to_string().contains("startup diagnostic"), "{error}");
            match error {
                PluginProcessError::Diagnostics { source, .. } if field == "protocolVersion" => {
                    assert!(matches!(*source, PluginProcessError::ProtocolMismatch(_)))
                }
                PluginProcessError::Diagnostics { source, .. } => assert!(matches!(
                    *source,
                    PluginProcessError::IdentityMismatch { .. }
                )),
                _ => panic!("unexpected error: {error}"),
            }
        }
    }

    #[tokio::test]
    async fn initialization_timeout_cleans_up() {
        let fixture = Fixture::new(initialize_result(), "timeout diagnostic", true);
        let error = PluginProcess::start(fixture.load()).await.err().unwrap();
        assert!(
            matches!(error, PluginProcessError::Diagnostics { source, .. } if matches!(*source, PluginProcessError::StartupTimeout))
        );
    }

    #[tokio::test]
    async fn routes_out_of_order_responses_and_notifications() {
        let (host, peer) = tokio::io::duplex(4096);
        let (read, write) = tokio::io::split(host);
        let transport = Transport::new(read, write);
        let peer_task = tokio::spawn(async move {
            let (read, mut write) = tokio::io::split(peer);
            let mut lines = BufReader::new(read).lines();
            for id in [1, 2] {
                let line = lines.next_line().await.unwrap().unwrap();
                assert_eq!(serde_json::from_str::<Value>(&line).unwrap()["id"], id);
            }
            write.write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"event/test\",\"params\":{}}\n{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":22}\n{\"jsonrpc\":\"2.0\",\"id\":1,\"error\":{\"code\":-1,\"message\":\"remote\"}}\n").await.unwrap();
        });
        let (a, b) = tokio::join!(
            transport.request("a", Value::Null),
            transport.request("b", Value::Null)
        );
        assert!(matches!(a, Err(PluginProcessError::RemoteError(_))));
        assert_eq!(b.unwrap(), 22);
        assert_eq!(
            transport
                .notifications
                .lock()
                .await
                .recv()
                .await
                .unwrap()
                .method,
            "event/test"
        );
        peer_task.await.unwrap();
    }

    #[tokio::test]
    async fn invalid_frames_and_eof_fail_pending_requests() {
        for frame in [
            b"not json\n".to_vec(),
            vec![b'x'; MAX_PLUGIN_FRAME_BYTES + 2],
            Vec::new(),
        ] {
            let (host, peer) = tokio::io::duplex(4096);
            let (read, write) = tokio::io::split(host);
            let transport = Transport::new(read, write);
            let peer_task = tokio::spawn(async move {
                let (read, mut write) = tokio::io::split(peer);
                let mut lines = BufReader::new(read).lines();
                lines.next_line().await.unwrap();
                let _ = write.write_all(&frame).await;
            });
            assert!(matches!(
                transport.request("test", Value::Null).await,
                Err(PluginProcessError::Transport(_))
            ));
            peer_task.await.unwrap();
        }
    }
}
