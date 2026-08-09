use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Result};
use tracing::field::{Field, Visit};
use tracing::level_filters::LevelFilter;
use tracing::span;
use tracing::{Event, Level, Metadata, Subscriber};

static LOG_PATH: OnceLock<PathBuf> = OnceLock::new();

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum LogLevel {
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4,
    Trace = 5,
}

impl LogLevel {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "error" => Some(Self::Error),
            "warn" | "warning" => Some(Self::Warn),
            "info" => Some(Self::Info),
            "debug" => Some(Self::Debug),
            "trace" => Some(Self::Trace),
            _ => None,
        }
    }

    fn allows(self, level: &Level) -> bool {
        let level = match *level {
            Level::ERROR => Self::Error,
            Level::WARN => Self::Warn,
            Level::INFO => Self::Info,
            Level::DEBUG => Self::Debug,
            Level::TRACE => Self::Trace,
        };
        level <= self
    }

    fn max_level_filter(self) -> LevelFilter {
        match self {
            Self::Error => LevelFilter::ERROR,
            Self::Warn => LevelFilter::WARN,
            Self::Info => LevelFilter::INFO,
            Self::Debug => LevelFilter::DEBUG,
            Self::Trace => LevelFilter::TRACE,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LogFilter {
    default: LogLevel,
    targets: Vec<(String, LogLevel)>,
}

impl LogFilter {
    fn parse(value: &str) -> Result<Self> {
        if value.trim().is_empty() {
            bail!("invalid RI_LOG filter: value must not be empty");
        }
        let mut default = None;
        let mut targets = Vec::new();
        for directive in value.split(',') {
            let directive = directive.trim();
            if directive.is_empty() {
                bail!("invalid RI_LOG filter {value:?}: empty directive");
            }
            if let Some((target, level)) = directive.split_once('=') {
                if target.trim().is_empty() {
                    bail!("invalid RI_LOG filter {value:?}: target must not be empty");
                }
                let level = LogLevel::parse(level).ok_or_else(|| {
                    anyhow!(
                        "invalid RI_LOG filter {value:?}: unknown level {level:?}; expected error, warn, info, debug, or trace"
                    )
                })?;
                targets.push((target.trim().to_owned(), level));
            } else {
                if default.is_some() {
                    bail!("invalid RI_LOG filter {value:?}: only one default level is allowed");
                }
                default = Some(LogLevel::parse(directive).ok_or_else(|| {
                    anyhow!(
                        "invalid RI_LOG filter {value:?}: expected error, warn, info, debug, or trace"
                    )
                })?);
            }
        }
        Ok(Self {
            default: default.unwrap_or(LogLevel::Info),
            targets,
        })
    }

    fn level_for(&self, target: &str) -> LogLevel {
        self.targets
            .iter()
            .filter(|(prefix, _)| target == prefix || target.starts_with(&format!("{prefix}::")))
            .max_by_key(|(prefix, _)| prefix.len())
            .map(|(_, level)| *level)
            .unwrap_or(self.default)
    }
}

struct FileSubscriber {
    filter: LogFilter,
    file: Mutex<File>,
    next_span: AtomicU64,
}

impl FileSubscriber {
    fn new(filter: LogFilter, file: File) -> Self {
        Self {
            filter,
            file: Mutex::new(file),
            next_span: AtomicU64::new(1),
        }
    }

    fn write_event(&self, event: &Event<'_>) {
        let mut visitor = EventVisitor::default();
        event.record(&mut visitor);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis())
            .unwrap_or_default();
        let fields = if visitor.fields.is_empty() {
            String::new()
        } else {
            format!(" {}", visitor.fields.join(" "))
        };
        let line = format!(
            "{timestamp} {} {} {}{fields}\n",
            event.metadata().level(),
            event.metadata().target(),
            event.metadata().name(),
        );
        if let Ok(mut file) = self.file.lock() {
            let _ = file.write_all(line.as_bytes());
            let _ = file.flush();
        }
    }
}

impl Subscriber for FileSubscriber {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        self.filter
            .level_for(metadata.target())
            .allows(metadata.level())
    }

    fn max_level_hint(&self) -> Option<LevelFilter> {
        Some(
            self.filter
                .targets
                .iter()
                .map(|(_, level)| *level)
                .max()
                .unwrap_or(self.filter.default)
                .max_level_filter(),
        )
    }

    fn new_span(&self, _span: &span::Attributes<'_>) -> span::Id {
        span::Id::from_u64(self.next_span.fetch_add(1, Ordering::Relaxed))
    }

    fn record(&self, _span: &span::Id, _values: &span::Record<'_>) {}

    fn record_follows_from(&self, _span: &span::Id, _follows: &span::Id) {}

    fn event(&self, event: &Event<'_>) {
        self.write_event(event);
    }

    fn enter(&self, _span: &span::Id) {}

    fn exit(&self, _span: &span::Id) {}
}

#[derive(Default)]
struct EventVisitor {
    fields: Vec<String>,
}

impl Visit for EventVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        let name = field.name();
        let value = if is_sensitive_field(name) {
            "<redacted>".to_owned()
        } else {
            format!("{value:?}")
        };
        self.fields.push(format!("{name}={value}"));
    }
}

fn is_sensitive_field(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.contains("api_key")
        || name.contains("apikey")
        || name.contains("authorization")
        || name.contains("auth_header")
        || name.contains("custom_header")
        || name.contains("secret")
        || name.contains("password")
        || name.contains("credential")
        || name.contains("access_token")
        || name.contains("refresh_token")
}

pub fn init() -> Result<Option<PathBuf>> {
    let Some(value) = env::var_os("RI_LOG") else {
        return Ok(None);
    };
    let value = value.to_string_lossy();
    let filter = LogFilter::parse(&value)?;
    let directory = default_log_directory()?;
    fs::create_dir_all(&directory).map_err(|source| {
        anyhow!(
            "could not create diagnostic log directory {}: {source}",
            directory.display()
        )
    })?;
    set_private_directory_permissions(&directory)?;

    let path = directory.join(format!("ri-{}-{}.log", utc_timestamp(), std::process::id()));
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&path)
        .map_err(|source| {
            anyhow!(
                "could not create diagnostic log {}: {source}",
                path.display()
            )
        })?;
    set_private_file_permissions(&file, &path)?;

    tracing::subscriber::set_global_default(FileSubscriber::new(filter, file)).map_err(|_| {
        anyhow!("could not initialize RI_LOG; a global logger is already installed")
    })?;
    let _ = LOG_PATH.set(path.clone());
    Ok(Some(path))
}

pub(crate) fn path() -> Option<&'static Path> {
    LOG_PATH.get().map(PathBuf::as_path)
}

fn default_log_directory() -> Result<PathBuf> {
    let home = env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .ok_or_else(|| anyhow!("could not determine the user home directory for RI_LOG"))?;
    Ok(PathBuf::from(home).join(".ri/agent/logs"))
}

fn utc_timestamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default();
    let days = seconds / 86_400;
    let seconds_of_day = seconds % 86_400;
    let hour = seconds_of_day / 3_600;
    let minute = seconds_of_day % 3_600 / 60;
    let second = seconds_of_day % 60;
    let (year, month, day) = civil_from_days(days as i64);
    format!("{year:04}{month:02}{day:02}T{hour:02}{minute:02}{second:02}Z")
}

fn civil_from_days(days_since_epoch: i64) -> (i64, i64, i64) {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let day_of_year = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let month_part = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_part + 2) / 5 + 1;
    let month = month_part + if month_part < 10 { 3 } else { -9 };
    let year = year + i64::from(month <= 2);
    (year, month, day)
}

fn set_private_directory_permissions(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn set_private_file_permissions(file: &File, _path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tracing::dispatcher::{self, Dispatch};

    #[test]
    fn parses_levels_and_target_overrides() {
        let filter = LogFilter::parse("ri_core=trace,ri=debug").unwrap();
        assert_eq!(filter.level_for("ri_core::agent"), LogLevel::Trace);
        assert_eq!(filter.level_for("ri"), LogLevel::Debug);
        assert_eq!(filter.level_for("other"), LogLevel::Info);
        assert!(LogFilter::parse("verbose").is_err());
        assert!(LogFilter::parse("info,debug").is_err());
    }

    #[test]
    fn sensitive_fields_are_redacted_in_file_output() {
        let root = unique_test_dir("log-redaction");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("ri.log");
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .unwrap();
        let subscriber = FileSubscriber::new(LogFilter::parse("trace").unwrap(), file);
        let dispatch = Dispatch::new(subscriber);
        dispatcher::with_default(&dispatch, || {
            tracing::debug!(
                target: "ri",
                api_key = "super-secret-api-key-123",
                authorization = "Bearer secret-authorization",
                custom_header = "secret-header",
                prompt_bytes = 12,
                "startup metadata"
            );
        });
        drop(dispatch);

        let output = fs::read_to_string(&path).unwrap();
        assert!(!output.contains("super-secret-api-key-123"));
        assert!(!output.contains("Bearer secret-authorization"));
        assert!(!output.contains("secret-header"));
        assert!(output.contains("<redacted>"));
        assert!(output.contains("prompt_bytes=12"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn utc_timestamp_has_stable_machine_friendly_shape() {
        let timestamp = utc_timestamp();
        assert_eq!(timestamp.len(), 16);
        assert_eq!(&timestamp[8..9], "T");
        assert_eq!(&timestamp[15..16], "Z");
    }

    fn unique_test_dir(name: &str) -> PathBuf {
        env::temp_dir().join(format!(
            "ri-{name}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }
}
