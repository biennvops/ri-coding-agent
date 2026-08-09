use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::conversation::{CompactionSummary, ConversationHistory};
use crate::model::{ModelAssistantItem, ModelMessage, ModelThinking, ModelToolCall};

pub const SESSION_VERSION: u32 = 1;
pub const MAX_SESSION_RECORD_BYTES: usize = 8 * 1024 * 1024;
const SESSION_FILE_EXTENSION: &str = "jsonl";
const SYNTHETIC_TOOL_RESULT: &str = "Tool execution did not complete because the previous ri process ended before a result was recorded.";

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(String);

impl SessionId {
    pub fn new() -> Self {
        Self(new_uuid())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<String> for SessionId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for SessionId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WorkspaceId(String);

impl WorkspaceId {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for WorkspaceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MessageId(String);

impl MessageId {
    pub fn new() -> Self {
        Self(new_uuid())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for MessageId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<String> for MessageId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for MessageId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl fmt::Display for MessageId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionHeader {
    #[serde(rename = "type")]
    pub record_type: String,
    pub version: u32,
    pub id: SessionId,
    #[serde(rename = "createdAt")]
    pub created_at: String,
    #[serde(rename = "workspaceRoot")]
    pub workspace_root: PathBuf,
    #[serde(rename = "projectRoot")]
    pub project_root: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum SessionAssistantItem {
    Text {
        content: String,
    },
    Reasoning {
        #[serde(rename = "itemId")]
        item_id: Option<String>,
        summary: String,
        content: String,
        #[serde(rename = "encryptedContent")]
        encrypted_content: Option<String>,
    },
    Refusal {
        content: String,
    },
    ToolCall {
        index: usize,
        #[serde(rename = "callId")]
        call_id: Option<String>,
        #[serde(rename = "itemId")]
        item_id: Option<String>,
        name: Option<String>,
        arguments: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum SessionMessage {
    User {
        content: String,
    },
    Assistant {
        items: Vec<SessionAssistantItem>,
    },
    ToolResult {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
        content: String,
    },
}

impl TryFrom<&ModelMessage> for SessionMessage {
    type Error = SessionError;

    fn try_from(message: &ModelMessage) -> Result<Self, Self::Error> {
        Self::from_model(message)
    }
}

impl TryFrom<ModelMessage> for SessionMessage {
    type Error = SessionError;

    fn try_from(message: ModelMessage) -> Result<Self, Self::Error> {
        Self::from_model(&message)
    }
}

impl From<SessionMessage> for ModelMessage {
    fn from(message: SessionMessage) -> Self {
        message.into_model()
    }
}

impl SessionMessage {
    pub fn from_model(message: &ModelMessage) -> Result<Self, SessionError> {
        match message {
            ModelMessage::User { content } => Ok(Self::User {
                content: content.clone(),
            }),
            ModelMessage::Assistant { items } => Ok(Self::Assistant {
                items: items.iter().map(SessionAssistantItem::from_model).collect(),
            }),
            ModelMessage::ToolResult {
                tool_call_id,
                tool_name,
                content,
            } => Ok(Self::ToolResult {
                tool_call_id: tool_call_id.clone(),
                tool_name: tool_name.clone(),
                content: content.clone(),
            }),
            ModelMessage::System { .. } | ModelMessage::Developer { .. } => {
                Err(SessionError::NonSemanticMessage)
            }
        }
    }

    pub fn into_model(self) -> ModelMessage {
        match self {
            Self::User { content } => ModelMessage::User { content },
            Self::Assistant { items } => ModelMessage::Assistant {
                items: items.into_iter().map(Into::into).collect(),
            },
            Self::ToolResult {
                tool_call_id,
                tool_name,
                content,
            } => ModelMessage::ToolResult {
                tool_call_id,
                tool_name,
                content,
            },
        }
    }
}

impl From<&ModelAssistantItem> for SessionAssistantItem {
    fn from(item: &ModelAssistantItem) -> Self {
        Self::from_model(item)
    }
}

impl From<ModelAssistantItem> for SessionAssistantItem {
    fn from(item: ModelAssistantItem) -> Self {
        Self::from_model(&item)
    }
}

impl SessionAssistantItem {
    pub fn from_model(item: &ModelAssistantItem) -> Self {
        match item {
            ModelAssistantItem::Text { content } => Self::Text {
                content: content.clone(),
            },
            ModelAssistantItem::Reasoning(thinking) => Self::Reasoning {
                item_id: thinking.item_id.clone(),
                summary: thinking.summary.clone(),
                content: thinking.content.clone(),
                encrypted_content: thinking.encrypted_content.clone(),
            },
            ModelAssistantItem::Refusal { content } => Self::Refusal {
                content: content.clone(),
            },
            ModelAssistantItem::ToolCall(call) => Self::ToolCall {
                index: call.index,
                call_id: call.call_id.clone(),
                item_id: call.item_id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            },
        }
    }
}

impl From<SessionAssistantItem> for ModelAssistantItem {
    fn from(item: SessionAssistantItem) -> Self {
        match item {
            SessionAssistantItem::Text { content } => Self::Text { content },
            SessionAssistantItem::Reasoning {
                item_id,
                summary,
                content,
                encrypted_content,
            } => Self::Reasoning(ModelThinking {
                item_id,
                summary,
                content,
                encrypted_content,
            }),
            SessionAssistantItem::Refusal { content } => Self::Refusal { content },
            SessionAssistantItem::ToolCall {
                index,
                call_id,
                item_id,
                name,
                arguments,
            } => Self::ToolCall(ModelToolCall {
                index,
                call_id,
                item_id,
                name,
                arguments,
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum SessionRecord {
    #[serde(rename = "session")]
    Session {
        version: u32,
        id: SessionId,
        #[serde(rename = "createdAt")]
        created_at: String,
        #[serde(rename = "workspaceRoot")]
        workspace_root: PathBuf,
        #[serde(rename = "projectRoot")]
        project_root: PathBuf,
    },
    #[serde(rename = "message")]
    Message {
        id: MessageId,
        #[serde(rename = "parentId")]
        parent_id: Option<MessageId>,
        timestamp: String,
        message: SessionMessage,
    },
    #[serde(rename = "metadata")]
    Metadata { timestamp: String, name: String },
    #[serde(rename = "compaction")]
    Compaction {
        id: MessageId,
        #[serde(rename = "parentId")]
        parent_id: Option<MessageId>,
        timestamp: String,
        summary: String,
        #[serde(rename = "retainedMessageIds")]
        retained_message_ids: Vec<MessageId>,
    },
}

impl SessionRecord {
    fn header(&self) -> Option<SessionHeader> {
        match self {
            Self::Session {
                version,
                id,
                created_at,
                workspace_root,
                project_root,
            } => Some(SessionHeader {
                record_type: "session".to_owned(),
                version: *version,
                id: id.clone(),
                created_at: created_at.clone(),
                workspace_root: workspace_root.clone(),
                project_root: project_root.clone(),
            }),
            Self::Message { .. } | Self::Metadata { .. } | Self::Compaction { .. } => None,
        }
    }

    fn timestamp(&self) -> Option<&str> {
        match self {
            Self::Session { created_at, .. } => Some(created_at),
            Self::Message { timestamp, .. }
            | Self::Metadata { timestamp, .. }
            | Self::Compaction { timestamp, .. } => Some(timestamp),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionInfo {
    pub id: SessionId,
    pub path: PathBuf,
    pub name: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub workspace_root: PathBuf,
    pub project_root: PathBuf,
    pub message_count: usize,
    pub first_user_preview: Option<String>,
    pub materialized: bool,
}

impl SessionInfo {
    pub fn display_name(&self) -> String {
        self.name
            .clone()
            .or_else(|| self.first_user_preview.clone())
            .unwrap_or_else(|| self.id.to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionSummary {
    pub id: SessionId,
    pub path: PathBuf,
    pub name: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub workspace_root: PathBuf,
    pub message_count: usize,
    pub first_user_preview: Option<String>,
}

impl From<&SessionInfo> for SessionSummary {
    fn from(info: &SessionInfo) -> Self {
        Self {
            id: info.id.clone(),
            path: info.path.clone(),
            name: info.name.clone(),
            created_at: info.created_at.clone(),
            updated_at: info.updated_at.clone(),
            workspace_root: info.workspace_root.clone(),
            message_count: info.message_count,
            first_user_preview: info.first_user_preview.clone(),
        }
    }
}

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("could not access session {path}: {source}")]
    Io { path: PathBuf, source: io::Error },

    #[error("session {path} is corrupted at line {line}: {message}")]
    Corrupted {
        path: PathBuf,
        line: usize,
        message: String,
    },

    #[error("session {path} is missing its first header record")]
    MissingHeader { path: PathBuf },

    #[error("unsupported ri session version {version}")]
    UnsupportedVersion { version: u32 },

    #[error("session record is not a semantic conversation message")]
    NonSemanticMessage,

    #[error("session message parent is invalid at line {line}: {message}")]
    InvalidParent { line: usize, message: String },

    #[error("session {path} has an invalid record: {message}")]
    InvalidRecord { path: PathBuf, message: String },

    #[error(
        "session belongs to {session_workspace} but ri was launched from {current_workspace}; launch ri from {session_workspace} to resume this session"
    )]
    WorkspaceMismatch {
        session_workspace: PathBuf,
        current_workspace: PathBuf,
    },

    #[error("session record at line {line} exceeds the {limit} byte limit")]
    RecordTooLarge { line: usize, limit: usize },

    #[error("session is already open by another ri process")]
    AlreadyOpen,

    #[error("session id {id:?} matches multiple sessions; provide more characters")]
    AmbiguousId { id: String },

    #[error("no session matches id {id:?}")]
    MissingId { id: String },

    #[error("invalid session name: {0}")]
    InvalidName(String),

    #[error("session writer state was poisoned")]
    WriterPoisoned,
}

fn io_error(path: impl Into<PathBuf>, source: io::Error) -> SessionError {
    SessionError::Io {
        path: path.into(),
        source,
    }
}

#[derive(Clone, Debug)]
pub struct SessionSnapshot {
    pub info: SessionInfo,
    /// The active provider projection, including a generated summary when present.
    pub history: Vec<ModelMessage>,
    /// Every persisted semantic message, including messages summarized out of `history`.
    pub transcript: Vec<ModelMessage>,
    pub active_summary: Option<CompactionSummary>,
    pub warnings: Vec<String>,
    active_message_ids: Vec<MessageId>,
    truncate_at: Option<u64>,
    needs_newline: bool,
    last_message_id: Option<MessageId>,
}

#[derive(Clone)]
pub struct SessionHandle {
    inner: Arc<Mutex<SessionWriter>>,
}

impl fmt::Debug for SessionHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SessionHandle")
            .finish_non_exhaustive()
    }
}

impl PartialEq for SessionHandle {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for SessionHandle {}

impl SessionHandle {
    pub fn info(&self) -> Result<SessionInfo, SessionError> {
        self.inner
            .lock()
            .map(|writer| writer.info.clone())
            .map_err(|_| SessionError::WriterPoisoned)
    }

    pub fn append_message(&self, message: &ModelMessage) -> Result<SessionInfo, SessionError> {
        self.inner
            .lock()
            .map_err(|_| SessionError::WriterPoisoned)?
            .append_message(message)
    }

    pub fn append_compaction(
        &self,
        summary: &str,
        retained_messages: &[ModelMessage],
    ) -> Result<SessionInfo, SessionError> {
        self.inner
            .lock()
            .map_err(|_| SessionError::WriterPoisoned)?
            .append_compaction(summary, retained_messages)
    }

    pub fn transcript_history(&self) -> Result<Vec<ModelMessage>, SessionError> {
        self.inner
            .lock()
            .map(|writer| writer.transcript.clone())
            .map_err(|_| SessionError::WriterPoisoned)
    }

    pub fn rename(&self, name: &str) -> Result<SessionInfo, SessionError> {
        let name = validate_name(name)?;
        self.inner
            .lock()
            .map_err(|_| SessionError::WriterPoisoned)?
            .rename(&name)
    }
}

#[derive(Clone, Default)]
pub enum SessionMode {
    #[default]
    Disabled,
    Enabled(SessionHandle),
}

impl fmt::Debug for SessionMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disabled => formatter.write_str("Disabled"),
            Self::Enabled(handle) => formatter.debug_tuple("Enabled").field(handle).finish(),
        }
    }
}

impl SessionMode {
    pub fn info(&self) -> Result<Option<SessionInfo>, SessionError> {
        match self {
            Self::Disabled => Ok(None),
            Self::Enabled(handle) => handle.info().map(Some),
        }
    }

    pub(crate) fn append_message(
        &self,
        message: &ModelMessage,
    ) -> Result<Option<SessionInfo>, SessionError> {
        match self {
            Self::Disabled => Ok(None),
            Self::Enabled(handle) => handle.append_message(message).map(Some),
        }
    }

    pub(crate) fn append_compaction(
        &self,
        summary: &str,
        retained_messages: &[ModelMessage],
    ) -> Result<Option<SessionInfo>, SessionError> {
        match self {
            Self::Disabled => Ok(None),
            Self::Enabled(handle) => handle
                .append_compaction(summary, retained_messages)
                .map(Some),
        }
    }

    pub(crate) fn transcript_history(&self) -> Result<Option<Vec<ModelMessage>>, SessionError> {
        match self {
            Self::Disabled => Ok(None),
            Self::Enabled(handle) => handle.transcript_history().map(Some),
        }
    }
}

pub struct OpenedSession {
    pub handle: SessionHandle,
    pub info: SessionInfo,
    /// The active provider projection, including a generated summary when present.
    pub history: Vec<ModelMessage>,
    /// The full semantic transcript, independent of active provider compaction.
    pub transcript: Vec<ModelMessage>,
    pub active_summary: Option<CompactionSummary>,
    pub warnings: Vec<String>,
}

struct SessionWriter {
    info: SessionInfo,
    header: SessionHeader,
    file: Option<File>,
    _lock: Option<File>,
    head: Option<MessageId>,
    active_messages: Vec<(MessageId, ModelMessage)>,
    transcript: Vec<ModelMessage>,
}

impl SessionWriter {
    fn lazy(
        root: PathBuf,
        workspace_root: PathBuf,
        project_root: PathBuf,
    ) -> Result<SessionHandle, SessionError> {
        let id = SessionId::new();
        let created_at = now_timestamp();
        let path = root.join(format!(
            "{}_{}.{}",
            filename_timestamp(&created_at),
            id,
            SESSION_FILE_EXTENSION
        ));
        let header = SessionHeader {
            record_type: "session".to_owned(),
            version: SESSION_VERSION,
            id: id.clone(),
            created_at: created_at.clone(),
            workspace_root: workspace_root.clone(),
            project_root: project_root.clone(),
        };
        let info = SessionInfo {
            id,
            path,
            name: None,
            created_at: created_at.clone(),
            updated_at: created_at,
            workspace_root,
            project_root,
            message_count: 0,
            first_user_preview: None,
            materialized: false,
        };
        Ok(SessionHandle {
            inner: Arc::new(Mutex::new(Self {
                info,
                header,
                file: None,
                _lock: None,
                head: None,
                active_messages: Vec::new(),
                transcript: Vec::new(),
            })),
        })
    }

    fn open(
        path: PathBuf,
        current_workspace: &Path,
    ) -> Result<(SessionHandle, SessionSnapshot), SessionError> {
        let lock = open_session_lock(&path)?;
        lock_exclusive(&lock)?;
        let mut options = OpenOptions::new();
        options.read(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options
            .open(&path)
            .map_err(|source| io_error(&path, source))?;
        let snapshot = read_session_from_file(&path, &mut file, true)?;
        if snapshot.info.workspace_root != current_workspace {
            return Err(SessionError::WorkspaceMismatch {
                session_workspace: snapshot.info.workspace_root,
                current_workspace: current_workspace.to_path_buf(),
            });
        }
        if let Some(offset) = snapshot.truncate_at {
            file.set_len(offset)
                .map_err(|source| io_error(&path, source))?;
            file.seek(SeekFrom::End(0))
                .map_err(|source| io_error(&path, source))?;
            file.sync_data().map_err(|source| io_error(&path, source))?;
        } else if snapshot.needs_newline {
            file.seek(SeekFrom::End(0))
                .map_err(|source| io_error(&path, source))?;
            file.write_all(b"\n")
                .and_then(|_| file.flush())
                .and_then(|_| file.sync_data())
                .map_err(|source| io_error(&path, source))?;
        }
        file.seek(SeekFrom::End(0))
            .map_err(|source| io_error(&path, source))?;
        let mut info = snapshot.info.clone();
        info.materialized = true;
        let active_messages = snapshot
            .active_message_ids
            .iter()
            .cloned()
            .zip(
                snapshot
                    .history
                    .iter()
                    .filter(|message| !matches!(message, ModelMessage::Developer { .. }))
                    .cloned(),
            )
            .collect();
        let writer = Self {
            info,
            header: SessionHeader {
                record_type: "session".to_owned(),
                version: SESSION_VERSION,
                id: snapshot.info.id.clone(),
                created_at: snapshot.info.created_at.clone(),
                workspace_root: snapshot.info.workspace_root.clone(),
                project_root: snapshot.info.project_root.clone(),
            },
            file: Some(file),
            _lock: Some(lock),
            head: snapshot.last_message_id.clone(),
            active_messages,
            transcript: snapshot.transcript.clone(),
        };
        let handle = SessionHandle {
            inner: Arc::new(Mutex::new(writer)),
        };
        let info = handle.info()?;
        Ok((handle, SessionSnapshot { info, ..snapshot }))
    }

    fn append_message(&mut self, message: &ModelMessage) -> Result<SessionInfo, SessionError> {
        let message = SessionMessage::from_model(message)?;
        self.materialize()?;
        let id = MessageId::new();
        let timestamp = now_timestamp();
        let record = SessionRecord::Message {
            id: id.clone(),
            parent_id: self.head.clone(),
            timestamp: timestamp.clone(),
            message: message.clone(),
        };
        append_record(
            self.file.as_mut().expect("materialized session"),
            &record,
            &self.info.path,
        )?;
        self.head = Some(id.clone());
        self.active_messages
            .push((id, message.clone().into_model()));
        self.transcript.push(message.clone().into_model());
        self.info.message_count += 1;
        self.info.updated_at = timestamp;
        if self.info.first_user_preview.is_none() {
            if let SessionMessage::User { content } = message {
                self.info.first_user_preview = Some(preview(&content));
            }
        }
        Ok(self.info.clone())
    }

    fn append_compaction(
        &mut self,
        summary: &str,
        retained_messages: &[ModelMessage],
    ) -> Result<SessionInfo, SessionError> {
        let mut retained_ids = Vec::with_capacity(retained_messages.len());
        let mut search_end = self.active_messages.len();
        for retained in retained_messages.iter().rev() {
            let Some(index) = self.active_messages[..search_end]
                .iter()
                .rposition(|(_, message)| message == retained)
            else {
                return Err(SessionError::InvalidRecord {
                    path: self.info.path.clone(),
                    message: "compaction retained message is not in the active session history"
                        .to_owned(),
                });
            };
            retained_ids.push(self.active_messages[index].0.clone());
            search_end = index;
        }
        retained_ids.reverse();

        self.materialize()?;
        let id = MessageId::new();
        let timestamp = now_timestamp();
        let record = SessionRecord::Compaction {
            id: id.clone(),
            parent_id: self.head.clone(),
            timestamp: timestamp.clone(),
            summary: summary.to_owned(),
            retained_message_ids: retained_ids,
        };
        append_record(
            self.file.as_mut().expect("materialized session"),
            &record,
            &self.info.path,
        )?;
        self.head = Some(id);
        self.active_messages = retained_messages
            .iter()
            .enumerate()
            .map(|(index, message)| {
                let retained_id = match &record {
                    SessionRecord::Compaction {
                        retained_message_ids,
                        ..
                    } => retained_message_ids[index].clone(),
                    _ => unreachable!(),
                };
                (retained_id, message.clone())
            })
            .collect();
        self.info.updated_at = timestamp;
        Ok(self.info.clone())
    }

    fn rename(&mut self, name: &str) -> Result<SessionInfo, SessionError> {
        self.materialize()?;
        let timestamp = now_timestamp();
        let record = SessionRecord::Metadata {
            timestamp: timestamp.clone(),
            name: name.to_owned(),
        };
        append_record(
            self.file.as_mut().expect("materialized session"),
            &record,
            &self.info.path,
        )?;
        self.info.name = Some(name.to_owned());
        self.info.updated_at = timestamp;
        Ok(self.info.clone())
    }

    fn materialize(&mut self) -> Result<(), SessionError> {
        if self.file.is_some() {
            debug_assert!(self._lock.is_some());
            return Ok(());
        }
        ensure_private_directory(self.info.path.parent().unwrap_or_else(|| Path::new(".")))?;
        let lock = open_session_lock(&self.info.path)?;
        lock_exclusive(&lock)?;
        let mut options = OpenOptions::new();
        options.create_new(true).read(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options
            .open(&self.info.path)
            .map_err(|source| io_error(&self.info.path, source))?;
        append_record(
            &mut file,
            &SessionRecord::Session {
                version: self.header.version,
                id: self.header.id.clone(),
                created_at: self.header.created_at.clone(),
                workspace_root: self.header.workspace_root.clone(),
                project_root: self.header.project_root.clone(),
            },
            &self.info.path,
        )?;
        self.file = Some(file);
        self._lock = Some(lock);
        self.info.path = fs::canonicalize(&self.info.path)
            .map_err(|source| io_error(&self.info.path, source))?;
        self.info.materialized = true;
        Ok(())
    }
}

pub struct SessionRepository {
    root: PathBuf,
    workspace_root: PathBuf,
    project_root: PathBuf,
    workspace_id: WorkspaceId,
}

impl fmt::Debug for SessionRepository {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SessionRepository")
            .field("root", &self.root)
            .field("workspace_root", &self.workspace_root)
            .field("project_root", &self.project_root)
            .field("workspace_id", &self.workspace_id)
            .finish()
    }
}

impl SessionRepository {
    pub fn new(
        sessions_root: impl Into<PathBuf>,
        workspace_root: impl AsRef<Path>,
        project_root: impl AsRef<Path>,
    ) -> Result<Self, SessionError> {
        let workspace_root = canonical_directory(workspace_root.as_ref())?;
        let project_root = canonical_directory(project_root.as_ref())?;
        let workspace_id = workspace_id(&workspace_root)?;
        Ok(Self {
            root: sessions_root.into().join(workspace_id.as_str()),
            workspace_root,
            project_root,
            workspace_id,
        })
    }

    pub fn for_workspace(
        workspace_root: impl AsRef<Path>,
        project_root: impl AsRef<Path>,
    ) -> Result<Self, SessionError> {
        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(PathBuf::from)
            .ok_or_else(|| SessionError::InvalidRecord {
                path: PathBuf::from("~/.ri/agent/sessions"),
                message: "could not determine the user home directory".to_owned(),
            })?;
        Self::new(
            home.join(".ri/agent/sessions"),
            workspace_root,
            project_root,
        )
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    pub fn workspace_id(&self) -> &WorkspaceId {
        &self.workspace_id
    }

    pub fn create(&self) -> Result<SessionHandle, SessionError> {
        SessionWriter::lazy(
            self.root.clone(),
            self.workspace_root.clone(),
            self.project_root.clone(),
        )
    }

    pub fn list(&self) -> Result<Vec<SessionSummary>, SessionError> {
        let entries = match fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(source) => return Err(io_error(&self.root, source)),
        };
        let mut summaries = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|source| io_error(&self.root, source))?;
            let raw_path = entry.path();
            if raw_path
                .extension()
                .and_then(|extension| extension.to_str())
                != Some(SESSION_FILE_EXTENSION)
            {
                continue;
            }
            let path = fs::canonicalize(&raw_path).map_err(|source| io_error(&raw_path, source))?;
            let snapshot = read_session_summary(&path)?;
            if snapshot.info.workspace_root != self.workspace_root {
                continue;
            }
            let modified = entry
                .metadata()
                .and_then(|metadata| metadata.modified())
                .unwrap_or(UNIX_EPOCH);
            summaries.push((SessionSummary::from(&snapshot.info), modified));
        }
        summaries.sort_by(|(left, left_modified), (right, right_modified)| {
            right
                .updated_at
                .cmp(&left.updated_at)
                .then_with(|| right_modified.cmp(left_modified))
        });
        Ok(summaries.into_iter().map(|(summary, _)| summary).collect())
    }

    pub fn latest(&self) -> Result<Option<SessionSummary>, SessionError> {
        Ok(self.list()?.into_iter().next())
    }

    pub fn resolve(&self, selector: &str) -> Result<PathBuf, SessionError> {
        if selector.ends_with(".jsonl") || selector.contains('/') || selector.contains('\\') {
            let path = PathBuf::from(selector);
            return fs::canonicalize(&path).map_err(|source| io_error(path, source));
        }
        let matches: Vec<_> = self
            .list()?
            .into_iter()
            .filter(|summary| summary.id.as_str().starts_with(selector))
            .collect();
        match matches.as_slice() {
            [] => Err(SessionError::MissingId {
                id: selector.to_owned(),
            }),
            [summary] => Ok(summary.path.clone()),
            _ => Err(SessionError::AmbiguousId {
                id: selector.to_owned(),
            }),
        }
    }

    pub fn open_selector(&self, selector: &str) -> Result<OpenedSession, SessionError> {
        let path = self.resolve(selector)?;
        self.open_path(path)
    }

    pub fn open_path(&self, path: impl AsRef<Path>) -> Result<OpenedSession, SessionError> {
        let path =
            fs::canonicalize(path.as_ref()).map_err(|source| io_error(path.as_ref(), source))?;
        let (handle, snapshot) = SessionWriter::open(path, &self.workspace_root)?;
        let mut history = snapshot.history.clone();
        let mut transcript = snapshot.transcript.clone();
        let unresolved = unresolved_tool_calls(&history);
        for call in unresolved {
            let Some(call_id) = call.call_id.clone() else {
                continue;
            };
            let Some(tool_name) = call.name.clone() else {
                continue;
            };
            let message = ModelMessage::ToolResult {
                tool_call_id: call_id,
                tool_name,
                content: SYNTHETIC_TOOL_RESULT.to_owned(),
            };
            handle.append_message(&message)?;
            history.push(message.clone());
            transcript.push(message);
        }
        let info = handle.info()?;
        Ok(OpenedSession {
            handle,
            info,
            history,
            transcript,
            active_summary: snapshot.active_summary,
            warnings: snapshot.warnings,
        })
    }
}

pub fn workspace_id(path: &Path) -> Result<WorkspaceId, SessionError> {
    let canonical = canonical_directory(path)?;
    let digest = sha256(canonical.to_string_lossy().as_bytes());
    Ok(WorkspaceId(
        digest[..16]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
    ))
}

pub fn validate_name(name: &str) -> Result<String, SessionError> {
    let name = name.trim();
    if name.is_empty() {
        return Err(SessionError::InvalidName("name cannot be empty".to_owned()));
    }
    if name.chars().count() > 120 {
        return Err(SessionError::InvalidName(
            "name cannot exceed 120 characters".to_owned(),
        ));
    }
    if name.chars().any(char::is_control) {
        return Err(SessionError::InvalidName(
            "name cannot contain control characters or newlines".to_owned(),
        ));
    }
    Ok(name.to_owned())
}

pub fn read_session(path: impl AsRef<Path>) -> Result<SessionSnapshot, SessionError> {
    read_session_inner(path.as_ref(), true)
}

fn read_session_summary(path: &Path) -> Result<SessionSnapshot, SessionError> {
    read_session_inner(path, false)
}

fn read_session_inner(path: &Path, include_history: bool) -> Result<SessionSnapshot, SessionError> {
    let mut file = File::open(path).map_err(|source| io_error(path, source))?;
    read_session_from_file(path, &mut file, include_history)
}

fn read_session_from_file(
    path: &Path,
    file: &mut File,
    include_history: bool,
) -> Result<SessionSnapshot, SessionError> {
    file.seek(SeekFrom::Start(0))
        .map_err(|source| io_error(path, source))?;
    let mut reader = BufReader::new(file);
    let mut line_bytes = Vec::new();
    let mut offset = 0u64;
    let mut line_number = 0usize;
    let mut needs_newline = false;
    let mut truncate_at = None;
    let mut warnings = Vec::new();
    let mut pending_invalid: Option<(usize, u64, String)> = None;
    let mut header: Option<SessionHeader> = None;
    let mut messages = Vec::new();
    let mut compactions = Vec::new();
    let mut message_count = 0usize;
    let mut first_user_preview_record = None;
    let mut last_message_id = None;
    let mut message_ids = HashSet::new();
    let mut entry_ids = HashSet::new();
    let mut name = None;
    let mut updated_at = None;

    loop {
        line_bytes.clear();
        let has_newline =
            match read_bounded_line(&mut reader, &mut line_bytes, MAX_SESSION_RECORD_BYTES) {
                Ok(None) => break,
                Ok(Some(has_newline)) => has_newline,
                Err(source) if source.kind() == io::ErrorKind::InvalidData => {
                    return Err(SessionError::RecordTooLarge {
                        line: line_number + 1,
                        limit: MAX_SESSION_RECORD_BYTES,
                    });
                }
                Err(source) => return Err(io_error(path, source)),
            };
        line_number += 1;
        let line_start = offset;
        offset = offset.saturating_add(line_bytes.len() as u64);
        if line_bytes.len() > MAX_SESSION_RECORD_BYTES {
            return Err(SessionError::RecordTooLarge {
                line: line_number,
                limit: MAX_SESSION_RECORD_BYTES,
            });
        }
        let mut content = line_bytes.as_slice();
        if has_newline {
            content = &content[..content.len() - 1];
            if content.last() == Some(&b'\r') {
                content = &content[..content.len() - 1];
            }
        }
        if content.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        if let Some((invalid_line, _, message)) = pending_invalid.take() {
            return Err(SessionError::Corrupted {
                path: path.to_path_buf(),
                line: invalid_line,
                message,
            });
        }
        let record = match serde_json::from_slice::<SessionRecord>(content) {
            Ok(record) => record,
            Err(error) if has_newline => {
                return Err(SessionError::Corrupted {
                    path: path.to_path_buf(),
                    line: line_number,
                    message: error.to_string(),
                });
            }
            Err(error) => {
                pending_invalid = Some((line_number, line_start, error.to_string()));
                truncate_at = Some(line_start);
                continue;
            }
        };
        if header.is_none() {
            let Some(parsed_header) = record.header() else {
                return Err(SessionError::MissingHeader {
                    path: path.to_path_buf(),
                });
            };
            if parsed_header.version != SESSION_VERSION {
                return Err(SessionError::UnsupportedVersion {
                    version: parsed_header.version,
                });
            }
            if parsed_header.record_type != "session" {
                return Err(SessionError::MissingHeader {
                    path: path.to_path_buf(),
                });
            }
            header = Some(parsed_header);
        } else if matches!(record, SessionRecord::Session { .. }) {
            return Err(SessionError::Corrupted {
                path: path.to_path_buf(),
                line: line_number,
                message: "session header must be the first record".to_owned(),
            });
        }

        match &record {
            SessionRecord::Message {
                id,
                parent_id,
                timestamp,
                message,
            } => {
                if entry_ids.contains(id) {
                    return Err(SessionError::Corrupted {
                        path: path.to_path_buf(),
                        line: line_number,
                        message: format!("duplicate session record id {id}"),
                    });
                }
                validate_parent(
                    parent_id.as_ref(),
                    entry_ids.is_empty(),
                    &entry_ids,
                    line_number,
                )?;
                entry_ids.insert(id.clone());
                message_ids.insert(id.clone());
                message_count += 1;
                last_message_id = Some(id.clone());
                if first_user_preview_record.is_none() {
                    if let SessionMessage::User { content } = message {
                        first_user_preview_record = Some(preview(content));
                    }
                }
                if include_history {
                    messages.push(StoredMessage {
                        id: id.clone(),
                        parent_id: parent_id.clone(),
                        message: message.clone(),
                        sequence: line_number,
                    });
                }
                updated_at = Some(timestamp.clone());
            }
            SessionRecord::Compaction {
                id,
                parent_id,
                timestamp,
                summary,
                retained_message_ids,
            } => {
                if entry_ids.is_empty() || parent_id.is_none() {
                    return Err(SessionError::InvalidParent {
                        line: line_number,
                        message: "a compaction checkpoint must follow a session record".to_owned(),
                    });
                }
                validate_parent(parent_id.as_ref(), false, &entry_ids, line_number)?;
                if entry_ids.contains(id) {
                    return Err(SessionError::Corrupted {
                        path: path.to_path_buf(),
                        line: line_number,
                        message: format!("duplicate session record id {id}"),
                    });
                }
                let mut retained = HashSet::new();
                for retained_id in retained_message_ids {
                    if !message_ids.contains(retained_id) || !retained.insert(retained_id) {
                        return Err(SessionError::InvalidRecord {
                            path: path.to_path_buf(),
                            message: format!(
                                "compaction retained unknown or duplicate message {retained_id}"
                            ),
                        });
                    }
                }
                entry_ids.insert(id.clone());
                last_message_id = Some(id.clone());
                if include_history {
                    compactions.push(StoredCompaction {
                        id: id.clone(),
                        parent_id: parent_id.clone(),
                        summary: summary.clone(),
                        retained_message_ids: retained_message_ids.clone(),
                        sequence: line_number,
                    });
                }
                updated_at = Some(timestamp.clone());
            }
            SessionRecord::Metadata {
                timestamp,
                name: value,
            } => {
                let normalized_name =
                    validate_name(value).map_err(|error| SessionError::Corrupted {
                        path: path.to_path_buf(),
                        line: line_number,
                        message: error.to_string(),
                    })?;
                name = Some(normalized_name);
                updated_at = Some(timestamp.clone());
            }
            SessionRecord::Session { .. } => {}
        }
        if let Some(timestamp) = record.timestamp() {
            if updated_at.is_none() {
                updated_at = Some(timestamp.to_owned());
            }
        }
        needs_newline = !has_newline;
    }

    if let Some((invalid_line, _, message)) = pending_invalid {
        let Some(header) = header.as_ref() else {
            return Err(SessionError::MissingHeader {
                path: path.to_path_buf(),
            });
        };
        let _ = header;
        warnings.push(format!(
            "ri: warning: ignored incomplete trailing record in session {}",
            path.display()
        ));
        let _ = invalid_line;
        let _ = message;
    }

    let Some(header) = header else {
        return Err(SessionError::MissingHeader {
            path: path.to_path_buf(),
        });
    };
    let (history, transcript, active_summary, active_message_ids) = if include_history {
        active_projection(&messages, &compactions, last_message_id.as_ref())
    } else {
        (Vec::new(), Vec::new(), None, Vec::new())
    };
    let info = SessionInfo {
        id: header.id.clone(),
        path: path.to_path_buf(),
        name,
        created_at: header.created_at.clone(),
        updated_at: updated_at.unwrap_or_else(|| header.created_at.clone()),
        workspace_root: header.workspace_root.clone(),
        project_root: header.project_root.clone(),
        message_count,
        first_user_preview: first_user_preview_record,
        materialized: true,
    };
    Ok(SessionSnapshot {
        info,
        history,
        transcript,
        active_summary,
        warnings,
        active_message_ids,
        truncate_at,
        needs_newline,
        last_message_id,
    })
}

#[derive(Clone)]
struct StoredMessage {
    id: MessageId,
    parent_id: Option<MessageId>,
    message: SessionMessage,
    sequence: usize,
}

#[derive(Clone)]
struct StoredCompaction {
    id: MessageId,
    parent_id: Option<MessageId>,
    summary: String,
    retained_message_ids: Vec<MessageId>,
    sequence: usize,
}

fn validate_parent(
    parent_id: Option<&MessageId>,
    first: bool,
    known_ids: &HashSet<MessageId>,
    line: usize,
) -> Result<(), SessionError> {
    if first {
        if parent_id.is_some() {
            return Err(SessionError::InvalidParent {
                line,
                message: "the first history record must have parentId null".to_owned(),
            });
        }
    } else if let Some(parent_id) = parent_id {
        if !known_ids.contains(parent_id) {
            return Err(SessionError::InvalidParent {
                line,
                message: format!("parent record {parent_id} does not precede this record"),
            });
        }
    } else {
        return Err(SessionError::InvalidParent {
            line,
            message: "a later history record must have a parentId".to_owned(),
        });
    }
    Ok(())
}

fn active_projection(
    messages: &[StoredMessage],
    compactions: &[StoredCompaction],
    latest_id: Option<&MessageId>,
) -> (
    Vec<ModelMessage>,
    Vec<ModelMessage>,
    Option<CompactionSummary>,
    Vec<MessageId>,
) {
    let transcript: Vec<ModelMessage> = messages
        .iter()
        .map(|message| message.message.clone().into_model())
        .collect();
    let mut parents = HashMap::with_capacity(messages.len() + compactions.len());
    for message in messages {
        parents.insert(message.id.clone(), message.parent_id.clone());
    }
    for compaction in compactions {
        parents.insert(compaction.id.clone(), compaction.parent_id.clone());
    }

    let mut active_chain = HashSet::new();
    let mut next = latest_id.cloned();
    while let Some(id) = next {
        if !active_chain.insert(id.clone()) {
            break;
        }
        next = parents.get(&id).cloned().flatten();
    }

    let active_compaction = compactions
        .iter()
        .rfind(|compaction| active_chain.contains(&compaction.id));
    let Some(compaction) = active_compaction else {
        let active_messages: Vec<(MessageId, ModelMessage)> = messages
            .iter()
            .filter(|message| active_chain.contains(&message.id))
            .map(|message| (message.id.clone(), message.message.clone().into_model()))
            .collect();
        let active_ids = active_messages.iter().map(|(id, _)| id.clone()).collect();
        let history = active_messages
            .into_iter()
            .map(|(_, message)| message)
            .collect();
        return (history, transcript, None, active_ids);
    };

    let retained: HashSet<&MessageId> = compaction.retained_message_ids.iter().collect();
    let mut active_messages = Vec::new();
    let mut active_ids = Vec::new();
    for message in messages {
        if !active_chain.contains(&message.id) {
            continue;
        }
        if (message.sequence < compaction.sequence && retained.contains(&message.id))
            || message.sequence > compaction.sequence
        {
            active_ids.push(message.id.clone());
            active_messages.push(message.message.clone().into_model());
        }
    }
    let summary = CompactionSummary::new(compaction.summary.clone());
    let history =
        ConversationHistory::new(Some(summary.clone()), active_messages).provider_messages();
    (history, transcript, Some(summary), active_ids)
}

fn unresolved_tool_calls(history: &[ModelMessage]) -> Vec<ModelToolCall> {
    let mut pending = Vec::<ModelToolCall>::new();
    for message in history {
        match message {
            ModelMessage::User { .. } => pending.clear(),
            ModelMessage::Assistant { items } => {
                pending.clear();
                for item in items {
                    if let ModelAssistantItem::ToolCall(call) = item {
                        if call.call_id.is_some() {
                            pending.push(call.clone());
                        }
                    }
                }
            }
            ModelMessage::ToolResult { tool_call_id, .. } => {
                pending.retain(|call| call.call_id.as_deref() != Some(tool_call_id));
            }
            ModelMessage::System { .. } | ModelMessage::Developer { .. } => {}
        }
    }
    pending
}

fn read_bounded_line<R: Read>(
    reader: &mut BufReader<R>,
    buffer: &mut Vec<u8>,
    limit: usize,
) -> io::Result<Option<bool>> {
    loop {
        let chunk = reader.fill_buf()?;
        if chunk.is_empty() {
            return Ok((!buffer.is_empty()).then_some(false));
        }
        if let Some(newline) = chunk.iter().position(|byte| *byte == b'\n') {
            let length = newline + 1;
            if buffer.len().saturating_add(length) > limit {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "session record is too large",
                ));
            }
            buffer.extend_from_slice(&chunk[..length]);
            reader.consume(length);
            return Ok(Some(true));
        }
        if buffer.len().saturating_add(chunk.len()) > limit {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "session record is too large",
            ));
        }
        let length = chunk.len();
        buffer.extend_from_slice(chunk);
        reader.consume(length);
    }
}

fn append_record(file: &mut File, record: &SessionRecord, path: &Path) -> Result<(), SessionError> {
    let bytes = serde_json::to_vec(record).map_err(|error| SessionError::InvalidRecord {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    if bytes.len().saturating_add(1) > MAX_SESSION_RECORD_BYTES {
        return Err(SessionError::RecordTooLarge {
            line: 0,
            limit: MAX_SESSION_RECORD_BYTES,
        });
    }
    file.write_all(&bytes)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.flush())
        .and_then(|_| file.sync_data())
        .map_err(|source| io_error(path, source))
}

fn canonical_directory(path: &Path) -> Result<PathBuf, SessionError> {
    let canonical = fs::canonicalize(path).map_err(|source| io_error(path, source))?;
    if !canonical.is_dir() {
        return Err(SessionError::InvalidRecord {
            path: path.to_path_buf(),
            message: "workspace root is not a directory".to_owned(),
        });
    }
    Ok(canonical)
}

fn ensure_private_directory(path: &Path) -> Result<(), SessionError> {
    fs::create_dir_all(path).map_err(|source| io_error(path, source))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .map_err(|source| io_error(path, source))?;
    }
    Ok(())
}

fn open_session_lock(path: &Path) -> Result<File, SessionError> {
    let lock_path = session_lock_path(path);
    let mut options = OpenOptions::new();
    options.create(true).read(true).write(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
        .open(&lock_path)
        .map_err(|source| io_error(&lock_path, source))
}

fn session_lock_path(path: &Path) -> PathBuf {
    let mut lock = path.as_os_str().to_os_string();
    lock.push(".lock");
    PathBuf::from(lock)
}

fn lock_exclusive(file: &File) -> Result<(), SessionError> {
    use fs2::FileExt;

    file.try_lock_exclusive().map_err(|error| {
        if error.kind() == fs2::lock_contended_error().kind() {
            SessionError::AlreadyOpen
        } else {
            io_error("session lock", error)
        }
    })
}

fn preview(content: &str) -> String {
    let collapsed = content.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut result = collapsed.chars().take(80).collect::<String>();
    if collapsed.chars().count() > 80 {
        result.push('…');
    }
    result
}

fn filename_timestamp(timestamp: &str) -> String {
    timestamp
        .chars()
        .filter(|character| character.is_ascii_digit() || *character == 'T' || *character == 'Z')
        .collect()
}

fn now_timestamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    utc_timestamp(seconds)
}

fn utc_timestamp(seconds: u64) -> String {
    let days = (seconds / 86_400) as i64;
    let day_seconds = seconds % 86_400;
    let hour = day_seconds / 3_600;
    let minute = (day_seconds % 3_600) / 60;
    let second = day_seconds % 60;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let day_of_year = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month + 2) / 5 + 1;
    let month = month + if month < 10 { 3 } else { -9 };
    let year = year + if month <= 2 { 1 } else { 0 };
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

fn new_uuid() -> String {
    let mut bytes = [0u8; 16];
    let random_ok = File::open("/dev/urandom")
        .and_then(|mut file| file.read_exact(&mut bytes))
        .is_ok();
    if !random_ok {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let counter = ID_COUNTER.fetch_add(1, Ordering::Relaxed) as u128;
        let value = now ^ ((std::process::id() as u128) << 64) ^ counter;
        bytes.copy_from_slice(&value.to_le_bytes());
    }
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15]
    )
}

fn sha256(input: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut state = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut data = input.to_vec();
    let bit_length = (data.len() as u64).saturating_mul(8);
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&bit_length.to_be_bytes());
    for chunk in data.chunks_exact(64) {
        let mut words = [0u32; 64];
        for (index, word) in words[..16].iter_mut().enumerate() {
            let start = index * 4;
            *word = u32::from_be_bytes([
                chunk[start],
                chunk[start + 1],
                chunk[start + 2],
                chunk[start + 3],
            ]);
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }
        let mut a = state[0];
        let mut b = state[1];
        let mut c = state[2];
        let mut d = state[3];
        let mut e = state[4];
        let mut f = state[5];
        let mut g = state[6];
        let mut h = state[7];
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(choice)
                .wrapping_add(K[index])
                .wrapping_add(words[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }
    let mut digest = [0u8; 32];
    for (index, word) in state.iter().enumerate() {
        digest[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_message_round_trip_preserves_reasoning_and_distinct_ids() {
        let message = ModelMessage::Assistant {
            items: vec![
                ModelAssistantItem::Text {
                    content: "hello\n世界".to_owned(),
                },
                ModelAssistantItem::Reasoning(ModelThinking {
                    item_id: Some("rs_1".to_owned()),
                    summary: "summary".to_owned(),
                    content: "private reasoning".to_owned(),
                    encrypted_content: Some("encrypted".to_owned()),
                }),
                ModelAssistantItem::ToolCall(ModelToolCall {
                    index: 2,
                    call_id: Some("call_123".to_owned()),
                    item_id: Some("fc_456".to_owned()),
                    name: Some("read".to_owned()),
                    arguments: r#"{"path":"a.txt"}"#.to_owned(),
                }),
            ],
        };
        let durable = SessionMessage::from_model(&message).unwrap();
        let encoded = serde_json::to_string(&durable).unwrap();
        let decoded: SessionMessage = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.into_model(), message);
        assert!(encoded.contains("call_123"));
        assert!(encoded.contains("fc_456"));
    }

    #[test]
    fn workspace_hash_is_sha256_prefix_and_deterministic() {
        assert_eq!(
            sha256(b"abc"),
            [
                0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
                0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
                0xf2, 0x00, 0x15, 0xad,
            ]
        );
        let root = test_dir("hash");
        fs::create_dir_all(&root).unwrap();
        let first = workspace_id(&root).unwrap();
        let second = workspace_id(&root).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 32);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn session_writer_is_lazy_and_parent_links_are_linear() {
        let workspace = test_dir("writer");
        let sessions = workspace.join("sessions");
        fs::create_dir_all(&workspace).unwrap();
        let repository = SessionRepository::new(&sessions, &workspace, &workspace).unwrap();
        let handle = repository.create().unwrap();
        assert!(!handle.info().unwrap().materialized);
        handle.append_message(&ModelMessage::user("one")).unwrap();
        handle
            .append_message(&ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: "two".to_owned(),
                }],
            })
            .unwrap();
        let content = fs::read_to_string(handle.info().unwrap().path).unwrap();
        let lines: Vec<serde_json::Value> = content
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[1]["parentId"], serde_json::Value::Null);
        assert_eq!(lines[2]["parentId"], lines[1]["id"]);
        fs::remove_dir_all(workspace).unwrap();
    }

    #[test]
    fn active_projection_follows_latest_parent_chain_but_keeps_full_transcript() {
        let root = test_dir("branch");
        fs::create_dir_all(&root).unwrap();
        let repository = SessionRepository::new(root.join("sessions"), &root, &root).unwrap();
        let handle = repository.create().unwrap();
        handle
            .append_message(&ModelMessage::user("root request"))
            .unwrap();
        handle
            .append_message(&ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: "abandoned answer".to_owned(),
                }],
            })
            .unwrap();
        let path = handle.info().unwrap().path;
        drop(handle);

        let records: Vec<SessionRecord> = fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let root_id = match &records[1] {
            SessionRecord::Message { id, .. } => id.clone(),
            record => panic!("expected root message, got {record:?}"),
        };
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        append_record(
            &mut file,
            &SessionRecord::Message {
                id: MessageId::from("active-branch"),
                parent_id: Some(root_id),
                timestamp: now_timestamp(),
                message: SessionMessage::User {
                    content: "active branch request".to_owned(),
                },
            },
            &path,
        )
        .unwrap();

        let snapshot = read_session(&path).unwrap();
        assert_eq!(snapshot.transcript.len(), 3);
        assert_eq!(
            snapshot.history,
            vec![
                ModelMessage::user("root request"),
                ModelMessage::user("active branch request"),
            ]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unterminated_trailing_record_is_recovered() {
        let root = test_dir("recovery");
        fs::create_dir_all(&root).unwrap();
        let repository = SessionRepository::new(root.join("sessions"), &root, &root).unwrap();
        let handle = repository.create().unwrap();
        handle.append_message(&ModelMessage::user("hello")).unwrap();
        let path = handle.info().unwrap().path;
        drop(handle);
        let mut content = fs::read_to_string(&path).unwrap();
        content.push_str("{\"type\":\"mess");
        fs::write(&path, content).unwrap();

        let snapshot = read_session(&path).unwrap();
        assert_eq!(snapshot.history, vec![ModelMessage::user("hello")]);
        assert_eq!(snapshot.warnings.len(), 1);
        repository.open_path(&path).unwrap();
        assert!(fs::read_to_string(&path).unwrap().ends_with('\n'));
        assert_eq!(fs::read_to_string(&path).unwrap().lines().count(), 2);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn newline_terminated_corrupted_trailing_record_fails() {
        let root = test_dir("newline-corruption");
        fs::create_dir_all(&root).unwrap();
        let repository = SessionRepository::new(root.join("sessions"), &root, &root).unwrap();
        let handle = repository.create().unwrap();
        handle.append_message(&ModelMessage::user("hello")).unwrap();
        let path = handle.info().unwrap().path;
        drop(handle);
        let mut content = fs::read_to_string(&path).unwrap();
        content.push_str("{}\n");
        fs::write(&path, &content).unwrap();

        let error = repository.open_path(&path);
        assert!(matches!(
            error,
            Err(SessionError::Corrupted { line: 3, .. })
        ));
        assert_eq!(fs::read_to_string(&path).unwrap(), content);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn interrupted_tool_batches_are_repaired_in_call_order_without_execution() {
        let root = test_dir("repair");
        fs::create_dir_all(&root).unwrap();
        let repository = SessionRepository::new(root.join("sessions"), &root, &root).unwrap();
        let handle = repository.create().unwrap();
        let call_a = ModelToolCall {
            index: 0,
            call_id: Some("call-a".to_owned()),
            item_id: Some("item-a".to_owned()),
            name: Some("bash".to_owned()),
            arguments: r#"{"command":"printf a"}"#.to_owned(),
        };
        let call_b = ModelToolCall {
            index: 1,
            call_id: Some("call-b".to_owned()),
            item_id: Some("item-b".to_owned()),
            name: Some("write".to_owned()),
            arguments: r#"{"path":"out.txt","content":"b"}"#.to_owned(),
        };
        handle
            .append_message(&ModelMessage::user("run tools"))
            .unwrap();
        handle
            .append_message(&ModelMessage::Assistant {
                items: vec![
                    ModelAssistantItem::ToolCall(call_a.clone()),
                    ModelAssistantItem::ToolCall(call_b.clone()),
                ],
            })
            .unwrap();
        handle
            .append_message(&ModelMessage::ToolResult {
                tool_call_id: "call-a".to_owned(),
                tool_name: "bash".to_owned(),
                content: "a".to_owned(),
            })
            .unwrap();
        let path = handle.info().unwrap().path;
        drop(handle);

        let opened = repository.open_path(&path).unwrap();
        assert_eq!(opened.history.len(), 4);
        assert!(matches!(
            &opened.history[3],
            ModelMessage::ToolResult {
                tool_call_id,
                tool_name,
                content,
            } if tool_call_id == "call-b"
                && tool_name == "write"
                && content == SYNTHETIC_TOOL_RESULT
        ));
        assert!(session_lock_path(&path).exists());
        let lines = fs::read_to_string(path).unwrap().lines().count();
        assert_eq!(lines, 5);
        drop(opened);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn active_session_data_can_be_read_and_listed_while_writer_is_open() {
        let root = test_dir("live-read");
        fs::create_dir_all(&root).unwrap();
        let repository = SessionRepository::new(root.join("sessions"), &root, &root).unwrap();
        let handle = repository.create().unwrap();
        handle
            .append_message(&ModelMessage::user("live session"))
            .unwrap();
        let path = handle.info().unwrap().path;

        let snapshot = read_session(&path).unwrap();
        assert_eq!(snapshot.history, vec![ModelMessage::user("live session")]);
        let summaries = repository.list().unwrap();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].path, path);
        drop(handle);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn exclusive_session_open_is_released_when_the_first_handle_drops() {
        let root = test_dir("lock");
        fs::create_dir_all(&root).unwrap();
        let repository = SessionRepository::new(root.join("sessions"), &root, &root).unwrap();
        let handle = repository.create().unwrap();
        handle.append_message(&ModelMessage::user("hello")).unwrap();
        let path = handle.info().unwrap().path;
        drop(handle);

        let first = repository.open_path(&path).unwrap();
        assert!(matches!(
            repository.open_path(&path),
            Err(SessionError::AlreadyOpen)
        ));
        drop(first);
        assert!(repository.open_path(&path).is_ok());
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn opening_and_resuming_session_reads_through_the_locked_handle() {
        let root = test_dir("windows-resume-lock");
        fs::create_dir_all(&root).unwrap();
        let repository = SessionRepository::new(root.join("sessions"), &root, &root).unwrap();
        let handle = repository.create().unwrap();
        handle
            .append_message(&ModelMessage::user("before resume"))
            .unwrap();
        let path = handle.info().unwrap().path;
        drop(handle);

        let opened = repository.open_path(&path).unwrap();
        opened
            .handle
            .append_message(&ModelMessage::user("after resume"))
            .unwrap();

        let transcript = opened.handle.transcript_history().unwrap();
        assert_eq!(transcript.len(), 2);
        assert_eq!(transcript[1], ModelMessage::user("after resume"));
        drop(opened);
        let snapshot = read_session(&path).unwrap();
        assert_eq!(snapshot.transcript.len(), 2);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn foreign_workspace_open_does_not_repair_the_session() {
        let root = test_dir("workspace-mismatch");
        let current_workspace = root.join("current");
        let foreign_workspace = root.join("foreign");
        let sessions = root.join("sessions");
        fs::create_dir_all(&current_workspace).unwrap();
        fs::create_dir_all(&foreign_workspace).unwrap();
        let foreign_repository =
            SessionRepository::new(&sessions, &foreign_workspace, &foreign_workspace).unwrap();
        let handle = foreign_repository.create().unwrap();
        handle
            .append_message(&ModelMessage::user("from another workspace"))
            .unwrap();
        let path = handle.info().unwrap().path;
        drop(handle);

        let mut content = fs::read_to_string(&path).unwrap();
        content.push_str("{\"type\":\"mess");
        fs::write(&path, &content).unwrap();
        let before = fs::read(&path).unwrap();
        let current_repository =
            SessionRepository::new(&sessions, &current_workspace, &current_workspace).unwrap();

        assert!(matches!(
            current_repository.open_path(&path),
            Err(SessionError::WorkspaceMismatch { .. })
        ));
        assert_eq!(fs::read(&path).unwrap(), before);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unsupported_header_versions_are_rejected_explicitly() {
        let root = test_dir("version");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("session.jsonl");
        fs::write(
            &path,
            r#"{"type":"session","version":999,"id":"session","createdAt":"2026-01-01T00:00:00Z","workspaceRoot":"/tmp","projectRoot":"/tmp"}
"#,
        )
        .unwrap();
        assert!(matches!(
            read_session(&path),
            Err(SessionError::UnsupportedVersion { version: 999 })
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn compaction_checkpoint_preserves_full_transcript_and_active_projection() {
        let root = test_dir("compaction");
        fs::create_dir_all(&root).unwrap();
        let repository = SessionRepository::new(root.join("sessions"), &root, &root).unwrap();
        let handle = repository.create().unwrap();
        let old_messages = vec![
            ModelMessage::user("old request"),
            ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: "old answer".to_owned(),
                }],
            },
            ModelMessage::user("recent request"),
            ModelMessage::Assistant {
                items: vec![ModelAssistantItem::Text {
                    content: "recent answer".to_owned(),
                }],
            },
        ];
        for message in &old_messages {
            handle.append_message(message).unwrap();
        }
        let retained = old_messages[2..].to_vec();
        handle
            .append_compaction("remember the old work", &retained)
            .unwrap();
        let new_request = ModelMessage::user("new request");
        handle.append_message(&new_request).unwrap();
        handle
            .append_compaction(
                "remember the newest work",
                &[
                    old_messages[2].clone(),
                    old_messages[3].clone(),
                    new_request,
                ],
            )
            .unwrap();
        let path = handle.info().unwrap().path;
        drop(handle);

        let snapshot = read_session(&path).unwrap();
        assert_eq!(
            snapshot.transcript,
            [
                old_messages.clone(),
                vec![ModelMessage::user("new request")]
            ]
            .concat()
        );
        assert_eq!(
            snapshot.active_summary,
            Some(CompactionSummary::new("remember the newest work"))
        );
        assert!(matches!(
            snapshot.history.first(),
            Some(ModelMessage::Developer { content }) if content.contains("remember the newest work")
        ));
        assert!(snapshot
            .history
            .contains(&ModelMessage::user("recent request")));
        assert!(snapshot
            .history
            .contains(&ModelMessage::user("new request")));
        assert!(!snapshot
            .history
            .contains(&ModelMessage::user("old request")));

        let opened = repository.open_path(&path).unwrap();
        assert_eq!(opened.transcript.len(), 5);
        assert_eq!(opened.history.len(), 4);
        assert!(opened.history.iter().any(|message| {
            matches!(message, ModelMessage::Developer { content } if content.contains("remember the newest work"))
        }));
        let raw = fs::read_to_string(&path).unwrap();
        let records: Vec<serde_json::Value> = raw
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(
            records
                .iter()
                .filter(|record| record["type"] == "compaction")
                .count(),
            2
        );
        assert_eq!(
            records
                .iter()
                .filter(|record| record["type"] == "message")
                .count(),
            5
        );
        drop(opened);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn names_are_validated_and_metadata_is_append_only() {
        let root = test_dir("name");
        fs::create_dir_all(&root).unwrap();
        let repository = SessionRepository::new(root.join("sessions"), &root, &root).unwrap();
        let handle = repository.create().unwrap();
        handle.rename(" First name ").unwrap();
        handle.rename("Second name").unwrap();
        assert_eq!(handle.info().unwrap().name.as_deref(), Some("Second name"));
        let path = handle.info().unwrap().path;
        let lines = fs::read_to_string(path).unwrap().lines().count();
        assert_eq!(lines, 3);
        assert!(validate_name("\n").is_err());
        fs::remove_dir_all(root).unwrap();
    }

    fn test_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "ri-session-{name}-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            ID_COUNTER.fetch_add(1, Ordering::Relaxed)
        ))
    }
}
