use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::thread::{self, JoinHandle};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

#[test]
fn print_mode_keeps_stdout_plain_text() {
    let fixture = Fixture::new(success_body());
    let output = fixture.run(&["-p", "hello", "--no-session", "--no-context"]);

    assert!(output.status.success(), "stderr: {}", text(&output.stderr));
    assert_eq!(text(&output.stdout), "hello\n");
    assert!(!text(&output.stdout).contains("\"version\""));
    assert!(text(&output.stderr).contains("context: disabled"));
    fixture.finish();
}

#[test]
fn json_mode_emits_only_versioned_ndjson_events() {
    let fixture = Fixture::new(success_body());
    let output = fixture.run(&["--json", "-p", "hello", "--no-session", "--no-context"]);

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        text(&output.stdout),
        text(&output.stderr)
    );
    let records = parse_records(&output.stdout);
    let types: Vec<_> = records
        .iter()
        .map(|record| record["type"].as_str().unwrap())
        .collect();
    assert_eq!(types.first(), Some(&"run_started"));
    assert_eq!(types.last(), Some(&"run_finished"));
    assert!(types.contains(&"turn_started"));
    assert!(types.contains(&"assistant_message_started"));
    assert!(types.contains(&"assistant_text_delta"));
    assert!(types.contains(&"assistant_message_finished"));
    assert!(types.contains(&"turn_finished"));
    assert_eq!(records.last().unwrap()["data"]["success"], true);
    for (sequence, record) in records.iter().enumerate() {
        assert_eq!(record["version"], 1);
        assert_eq!(record["seq"], sequence as u64);
    }
    assert!(!text(&output.stderr).contains("hello"));
    fixture.finish();
}

#[test]
fn json_provider_failure_emits_terminal_error_events_and_status_one() {
    let fixture = Fixture::with_response("500 Internal Server Error", "provider exploded");
    let output = fixture.run(&["--json", "-p", "hello", "--no-session", "--no-context"]);

    assert_eq!(
        output.status.code(),
        Some(1),
        "stdout: {}\nstderr: {}",
        text(&output.stdout),
        text(&output.stderr)
    );
    let records = parse_records(&output.stdout);
    let types: Vec<_> = records
        .iter()
        .map(|record| record["type"].as_str().unwrap())
        .collect();
    assert_eq!(types.first(), Some(&"run_started"));
    assert!(types.contains(&"turn_started"));
    assert!(types.contains(&"error"));
    assert!(types.contains(&"turn_finished"));
    assert_eq!(types.last(), Some(&"run_finished"));
    assert_eq!(records.last().unwrap()["data"]["success"], false);
    fixture.finish();
}

#[test]
fn json_logging_keeps_stdout_parseable_and_redacts_configured_secrets() {
    let secret = "super-secret-api-key-123";
    let fixture = Fixture::new_with_api_key(success_body(), secret);
    let output = fixture.run_with_log(
        &["--json", "-p", "hello", "--no-session", "--no-context"],
        "debug",
    );

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        text(&output.stdout),
        text(&output.stderr)
    );
    let _ = parse_records(&output.stdout);
    assert!(text(&output.stderr).contains("ri: logging to"));
    let log = read_single_log(&fixture.home);
    assert!(!log.contains(secret));
    assert!(!log.contains("Authorization"));
    assert!(!log.contains("hello"));
    fixture.finish();
}

#[test]
fn setup_and_cli_errors_use_status_two() {
    let home = unique_dir("cli-status");
    fs::create_dir_all(&home).unwrap();
    let missing_models = home.join("missing-models.json");
    let output = Command::new(env!("CARGO_BIN_EXE_ri"))
        .args(["--no-session", "--no-context"])
        .env("HOME", &home)
        .env("RI_MODELS_FILE", &missing_models)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(
        text(&output.stderr).contains("RI_MODELS_FILE"),
        "stderr: {}",
        text(&output.stderr)
    );

    let output = Command::new(env!("CARGO_BIN_EXE_ri"))
        .args(["--json"])
        .env("HOME", &home)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(text(&output.stderr).contains("--json requires --print"));
    assert!(output.stdout.is_empty());
    let _ = fs::remove_dir_all(home);
}

fn parse_records(stdout: &[u8]) -> Vec<Value> {
    let output = text(stdout);
    assert!(!output.is_empty(), "JSON stdout was empty");
    output
        .lines()
        .map(|line| {
            serde_json::from_str(line)
                .unwrap_or_else(|error| panic!("invalid JSON line {line:?}: {error}"))
        })
        .collect()
}

fn read_single_log(home: &Path) -> String {
    let directory = home.join(".ri/agent/logs");
    let entries = fs::read_dir(directory)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(entries.len(), 1);
    fs::read_to_string(entries[0].path()).unwrap()
}

fn text(value: &[u8]) -> String {
    String::from_utf8_lossy(value).into_owned()
}

fn success_body() -> &'static str {
    concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n",
        "data: [DONE]\n\n"
    )
}

struct Fixture {
    home: PathBuf,
    models: PathBuf,
    server: TestServer,
}

impl Fixture {
    fn new(body: &'static str) -> Self {
        Self::with_response_and_api_key("200 OK", body, None)
    }

    fn new_with_api_key(body: &'static str, api_key: &str) -> Self {
        Self::with_response_and_api_key("200 OK", body, Some(api_key))
    }

    fn with_response(status: &'static str, body: &'static str) -> Self {
        Self::with_response_and_api_key(status, body, None)
    }

    fn with_response_and_api_key(
        status: &'static str,
        body: &'static str,
        api_key: Option<&str>,
    ) -> Self {
        let home = unique_dir("cli-fixture");
        fs::create_dir_all(&home).unwrap();
        let models = home.join("models.json");
        let server = spawn_server(status, body);
        let api_key = api_key
            .map(|value| format!(",\"apiKey\":{value:?}"))
            .unwrap_or_default();
        let source = format!(
            r#"{{"providers":{{"test":{{"baseUrl":"{}","api":"openai-completions"{} ,"models":[{{"id":"model","contextWindow":200000}}]}}}}}}"#,
            server.url.as_str(),
            api_key
        );
        fs::write(&models, source).unwrap();
        Self {
            home,
            models,
            server,
        }
    }

    fn run(&self, args: &[&str]) -> Output {
        self.run_with_env(args, None)
    }

    fn run_with_log(&self, args: &[&str], level: &str) -> Output {
        self.run_with_env(args, Some(level))
    }

    fn run_with_env(&self, args: &[&str], log_level: Option<&str>) -> Output {
        let mut command = Command::new(env!("CARGO_BIN_EXE_ri"));
        command
            .args(args)
            .env("HOME", &self.home)
            .env("RI_MODELS_FILE", &self.models);
        if let Some(level) = log_level {
            command.env("RI_LOG", level);
        } else {
            command.env_remove("RI_LOG");
        }
        command.output().unwrap()
    }

    fn finish(self) {
        self.server.join();
        let _ = fs::remove_dir_all(self.home);
    }
}

struct TestServer {
    url: String,
    join: JoinHandle<()>,
}

impl TestServer {
    fn join(self) {
        self.join.join().unwrap();
    }
}

fn spawn_server(status: &'static str, body: &'static str) -> TestServer {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let join = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let _ = read_request(&mut stream);
        stream.write_all(response.as_bytes()).unwrap();
    });
    TestServer {
        url: format!("http://{address}"),
        join,
    }
}

fn read_request(stream: &mut TcpStream) -> Vec<u8> {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let read = stream.read(&mut buffer).unwrap_or(0);
        if read == 0 {
            break;
        }
        request.extend_from_slice(&buffer[..read]);
        if request.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
    }
    request
}

fn unique_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "ri-{name}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}
