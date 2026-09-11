use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures_util::StreamExt;
use serde_json::Value;
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
use tokio::sync::{mpsc, oneshot, Semaphore};
use tokio::task::JoinHandle;
use tokio_util::codec::{FramedRead, LinesCodec};

use super::protocol::*;

const PLUGIN_NOTIFICATION_CHANNEL_CAPACITY: usize = 64;
const MAX_PENDING_REQUESTS: usize = 64;

#[derive(Debug, Error)]
pub enum PluginProcessError {
    #[error("plugin transport: {0}")]
    Transport(String),
    #[error("plugin remote error: {0:?}")]
    RemoteError(RpcErrorObject),
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

struct Transport {
    outgoing: Option<mpsc::Sender<String>>,
    notifications: mpsc::Receiver<RpcNotification>,
    dispatch: Arc<Mutex<Dispatch>>,
    permits: Semaphore,
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
        let (outgoing, mut outgoing_rx) = mpsc::channel::<String>(MAX_PENDING_REQUESTS);
        let state = dispatch.clone();
        let writer = tokio::spawn(async move {
            while let Some(line) = outgoing_rx.recv().await {
                if let Err(error) = write.write_all(line.as_bytes()).await {
                    state.lock().unwrap().fail(error.to_string());
                    return;
                }
            }
            let _ = write.shutdown().await;
        });
        let state = dispatch.clone();
        let reader = tokio::spawn(async move {
            let mut frames = FramedRead::new(
                read,
                LinesCodec::new_with_max_length(MAX_PLUGIN_FRAME_BYTES),
            );
            let reason = loop {
                let message = match frames.next().await {
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
        });
        Self {
            outgoing: Some(outgoing),
            notifications,
            dispatch,
            permits: Semaphore::new(MAX_PENDING_REQUESTS),
            reader,
            writer,
        }
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value, PluginProcessError> {
        let _permit = self
            .permits
            .acquire()
            .await
            .map_err(|error| PluginProcessError::Transport(error.to_string()))?;
        let (sender, receiver) = oneshot::channel();
        let id = {
            let mut state = self.dispatch.lock().unwrap();
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
            dispatch: self.dispatch.clone(),
        };
        let mut line = encode_request(id, method, params)
            .map_err(|error| PluginProcessError::Transport(error.to_string()))?;
        line.push('\n');
        self.outgoing
            .as_ref()
            .ok_or_else(|| PluginProcessError::Transport("stdin closed".into()))?
            .send(line)
            .await
            .map_err(|_| PluginProcessError::Transport("stdin writer closed".into()))?;
        receiver
            .await
            .map_err(|_| PluginProcessError::Transport("response reader closed".into()))?
    }
}

impl Drop for Transport {
    fn drop(&mut self) {
        self.reader.abort();
        self.writer.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncBufReadExt, BufReader};

    #[tokio::test]
    async fn routes_out_of_order_responses_and_notifications() {
        let (host, peer) = tokio::io::duplex(4096);
        let (read, write) = tokio::io::split(host);
        let mut transport = Transport::new(read, write);
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
            transport.notifications.recv().await.unwrap().method,
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
