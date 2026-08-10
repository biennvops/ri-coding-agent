use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::process::Stdio;
use std::process::{Command, Output};
use std::sync::{Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

static CLI_TEST_LOCK: Mutex<()> = Mutex::new(());

fn cli_test_lock() -> MutexGuard<'static, ()> {
    CLI_TEST_LOCK.lock().unwrap()
}

#[cfg(unix)]
#[test]
fn tui_restores_the_pty_after_early_termination_signals() {
    let _lock = cli_test_lock();
    for signal in ["TERM", "HUP"] {
        run_tui_signal_torture(signal);
    }
}

#[cfg(unix)]
fn run_tui_signal_torture(signal: &str) {
    if Command::new("expect").arg("-v").output().is_err() {
        return;
    }

    let root = unique_dir("tui-signal");
    fs::create_dir_all(&root).unwrap();
    let models = root.join("models.json");
    let pid_path = root.join("ri.pid");
    let tty_state_path = root.join("tty-state");
    let expect_script = root.join("tui.exp");
    fs::write(
        &models,
        r#"{"providers":{"test":{"baseUrl":"http://127.0.0.1:1","api":"openai-completions","models":[{"id":"model"}]}}}"#,
    )
    .unwrap();
    fs::write(
        &expect_script,
        r#"
            set timeout 5
            spawn $env(RI_TEST_BIN) --no-session --no-context
            set pid [exp_pid]
            set slave $spawn_out(slave,name)
            set pid_file [open $env(RI_TEST_PID) w]
            puts $pid_file $pid
            close $pid_file
            expect {
                -re {\x1b\[\?1049h} {
                    send "\033\[1;1R"
                    exec kill [format "-%s" $env(RI_TEST_SIGNAL)] $pid
                }
                timeout { exit 10 }
                eof { exit 11 }
            }
            after 100
            set state [exec stty -a < $slave]
            expect {
                eof {}
                timeout { exit 12 }
            }
            set state_file [open $env(RI_TEST_TTY) w]
            puts $state_file $state
            close $state_file
            exit 0
        "#,
    )
    .unwrap();

    let output = Command::new("expect")
        .arg(&expect_script)
        .env("HOME", &root)
        .env("RI_MODELS_FILE", &models)
        .env("RI_TEST_BIN", env!("CARGO_BIN_EXE_ri"))
        .env("RI_TEST_PID", &pid_path)
        .env("RI_TEST_TTY", &tty_state_path)
        .env("RI_TEST_SIGNAL", signal)
        .output()
        .expect("expect should launch a PTY");
    assert!(
        output.status.success(),
        "expect failed for SIG{signal}: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let tty_state = fs::read_to_string(&tty_state_path).unwrap_or_default();
    assert!(
        !tty_state.contains("-icanon"),
        "SIG{signal} left the PTY in raw mode: {tty_state}"
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn print_mode_keeps_stdout_plain_text() {
    let _lock = cli_test_lock();
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
    let _lock = cli_test_lock();
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
fn json_responses_emits_final_item_content_without_text_delta() {
    let _lock = cli_test_lock();
    let fixture = Fixture::responses(responses_final_item_body());
    let output = fixture.run(&["--json", "-p", "hello", "--no-session", "--no-context"]);

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        text(&output.stdout),
        text(&output.stderr)
    );
    let records = parse_records(&output.stdout);
    assert!(!records
        .iter()
        .any(|record| record["type"] == "assistant_text_delta"));
    let finished = records
        .iter()
        .find(|record| record["type"] == "assistant_message_finished")
        .expect("assistant_message_finished event should be present");
    assert_eq!(finished["data"]["itemCount"], 1);
    assert_eq!(finished["data"]["items"][0]["type"], "text");
    assert_eq!(finished["data"]["items"][0]["content"], "authoritative");
    fixture.finish();
}

#[test]
fn malformed_recent_state_is_recovered_and_restored_on_the_next_launch() {
    let _lock = cli_test_lock();
    let fixture = Fixture::with_models(&["model", "second"], success_body());
    let state_path = fixture.home.join(".ri/agent/state.json");
    fs::create_dir_all(state_path.parent().unwrap()).unwrap();
    fs::write(&state_path, "not json").unwrap();

    let first_output = fixture.run(&[
        "--json",
        "-p",
        "hello",
        "--provider",
        "test",
        "--model",
        "second",
        "--no-session",
        "--no-context",
    ]);
    assert!(
        first_output.status.success(),
        "stdout: {}\nstderr: {}",
        text(&first_output.stdout),
        text(&first_output.stderr)
    );
    let home = fixture.keep_home();

    let state: Value =
        serde_json::from_str(&fs::read_to_string(home.join(".ri/agent/state.json")).unwrap())
            .unwrap();
    assert_eq!(state["version"], 1);
    assert_eq!(state["lastModel"]["provider"], "test");
    assert_eq!(state["lastModel"]["model"], "second");
    assert_eq!(
        fs::read_to_string(home.join(".ri/agent/state.json.corrupt")).unwrap(),
        "not json"
    );

    let second = Fixture::in_home_with_models(home, &["model", "second"], success_body());
    let second_output = second.run(&["--json", "-p", "again", "--no-session", "--no-context"]);
    assert!(
        second_output.status.success(),
        "stdout: {}\nstderr: {}",
        text(&second_output.stdout),
        text(&second_output.stderr)
    );
    let second_records = parse_records(&second_output.stdout);
    assert_eq!(second_records[0]["data"]["model"]["provider"], "test");
    assert_eq!(second_records[0]["data"]["model"]["model"], "second");
    second.finish();
}

#[test]
fn json_continue_reuses_the_persistent_session() {
    let _lock = cli_test_lock();
    let first = Fixture::new(success_body());
    let first_output = first.run(&["--json", "-p", "hello", "--no-context"]);
    assert!(
        first_output.status.success(),
        "stdout: {}\nstderr: {}",
        text(&first_output.stdout),
        text(&first_output.stderr)
    );
    let first_records = parse_records(&first_output.stdout);
    let session_id = first_records[0]["data"]["session"]["id"]
        .as_str()
        .expect("persistent run should have a session id")
        .to_owned();
    let home = first.keep_home();

    let second = Fixture::in_home(home.clone(), "200 OK", success_body(), None);
    let second_output = second.run(&["--json", "-p", "continue", "-c", "--no-context"]);
    assert!(
        second_output.status.success(),
        "stdout: {}\nstderr: {}",
        text(&second_output.stdout),
        text(&second_output.stderr)
    );
    let second_records = parse_records(&second_output.stdout);
    assert_eq!(second_records[0]["data"]["session"]["id"], session_id);
    second.finish();
}

#[test]
fn json_mode_emits_tool_lifecycle_events_from_the_shared_runtime() {
    let fixture = Fixture::in_home_with_responses(
        unique_dir("cli-tool-fixture"),
        vec![("200 OK", tool_call_body()), ("200 OK", success_body())],
        None,
    );
    let output = fixture.run(&[
        "--json",
        "-p",
        "run the tool",
        "--no-session",
        "--no-context",
    ]);

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        text(&output.stdout),
        text(&output.stderr)
    );
    let records = parse_records(&output.stdout);
    let tool_started = records
        .iter()
        .find(|record| record["type"] == "tool_started")
        .expect("tool_started event should be present");
    assert_eq!(tool_started["data"]["name"], "bash");
    assert_eq!(tool_started["data"]["callId"], "call-1");
    assert!(records.iter().any(|record| {
        record["type"] == "tool_output" && record["data"]["chunk"] == "tool-output"
    }));
    let tool_finished = records
        .iter()
        .find(|record| record["type"] == "tool_finished")
        .expect("tool_finished event should be present");
    assert_eq!(tool_finished["data"]["success"], true);
    fixture.finish();
}

#[test]
fn json_provider_failure_emits_terminal_error_events_and_status_one() {
    let _lock = cli_test_lock();
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
    let _lock = cli_test_lock();
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
fn provider_failure_logs_sanitized_http_diagnostics() {
    let _lock = cli_test_lock();
    let secret = "super-secret-api-key-123";
    let body = r#"{"error":{"message":"invalid request for prompt hello and super-secret-api-key-123","api_key":"super-secret-api-key-123"},"prompt":"private prompt"}"#;
    let fixture = Fixture::with_response_and_api_key("400 Bad Request", body, Some(secret));
    let output = fixture.run_with_log(&["-p", "hello", "--no-session", "--no-context"], "debug");

    assert_eq!(output.status.code(), Some(1));
    let log = read_single_log(&fixture.home);
    assert!(log.contains("provider HTTP request failed"));
    assert!(log.contains("provider=test"));
    assert!(log.contains("model=model"));
    assert!(log.contains("api=openai-completions"));
    assert!(log.contains("status=400"));
    assert!(log.contains("error_body_truncated=false"));
    assert!(log.contains("invalid request"));
    assert!(!log.contains("hello"));
    assert!(!log.contains(secret));
    assert!(!log.contains("private prompt"));
    fixture.finish();
}

#[test]
fn malformed_models_and_settings_fail_before_runtime_setup() {
    let _lock = cli_test_lock();
    let home = unique_dir("cli-config-errors");
    fs::create_dir_all(home.join(".ri/agent")).unwrap();
    let models = home.join("models.json");

    fs::write(&models, "{ not json").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_ri"))
        .args(["-p", "hello", "--no-session", "--no-context"])
        .env("HOME", &home)
        .env_remove("USERPROFILE")
        .env_remove("RI_MODELS_FILE")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(text(&output.stderr).contains("models.json"));
    assert!(output.stdout.is_empty());

    fs::write(&models, valid_models()).unwrap();
    let settings = home.join(".ri/agent/settings.json");
    fs::write(&settings, "{ not json").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_ri"))
        .args(["-p", "hello", "--no-session", "--no-context"])
        .env("HOME", &home)
        .env_remove("USERPROFILE")
        .env_remove("RI_MODELS_FILE")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(text(&output.stderr).contains(&settings.display().to_string()));
    assert!(output.stdout.is_empty());
    let _ = fs::remove_dir_all(home);
}

#[test]
fn malformed_model_environment_references_are_actionable() {
    let _lock = cli_test_lock();
    let home = unique_dir("cli-model-env-error");
    fs::create_dir_all(&home).unwrap();
    let models = home.join("models.json");
    fs::write(
        &models,
        r#"{"providers":{"p":{"baseUrl":"https://example.test","api":"openai-responses","apiKey":"$RI_TEST_MISSING","models":[{"id":"m"}]}}}"#,
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_ri"))
        .args(["-p", "hello", "--no-session", "--no-context"])
        .env("HOME", &home)
        .env_remove("USERPROFILE")
        .env("RI_MODELS_FILE", &models)
        .env_remove("RI_TEST_MISSING")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(text(&output.stderr).contains("RI_TEST_MISSING"));
    assert!(output.stdout.is_empty());
    let _ = fs::remove_dir_all(home);
}

#[test]
fn invalid_context_fails_before_any_provider_request() {
    let _lock = cli_test_lock();
    for (name, contents, expected) in [
        ("invalid-utf8", vec![0xff, 0xfe, 0xfd], "not valid UTF-8"),
        ("too-large", vec![b'x'; 128 * 1024 + 1], "maximum is"),
    ] {
        let home = unique_dir(&format!("cli-context-home-{name}"));
        let project = unique_dir(name);
        fs::create_dir_all(&home).unwrap();
        fs::create_dir_all(&project).unwrap();
        let models = home.join("models.json");
        fs::write(&models, valid_models()).unwrap();
        fs::write(project.join("AGENTS.md"), contents).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_ri"))
            .args(["-p", "hello", "--no-session"])
            .current_dir(&project)
            .env("HOME", &home)
            .env_remove("USERPROFILE")
            .env("RI_MODELS_FILE", &models)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(text(&output.stderr).contains(expected));
        assert!(output.stdout.is_empty());
        let _ = fs::remove_dir_all(home);
        let _ = fs::remove_dir_all(project);
    }
}

#[test]
fn structurally_corrupt_session_fails_with_a_path_and_status_two() {
    let _lock = cli_test_lock();
    let home = unique_dir("cli-session-error");
    fs::create_dir_all(&home).unwrap();
    let models = home.join("models.json");
    fs::write(&models, valid_models()).unwrap();
    let workspace = std::env::current_dir().unwrap();
    let session = home.join("corrupt.jsonl");
    let header = serde_json::json!({
        "type": "session",
        "version": 1,
        "id": "bad-session",
        "createdAt": "2026-01-01T00:00:00Z",
        "workspaceRoot": workspace,
        "projectRoot": std::env::current_dir().unwrap(),
    });
    fs::write(&session, format!("{}\nnot json\n", header)).unwrap();
    let canonical_session = fs::canonicalize(&session).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_ri"))
        .args([
            "--session",
            session.to_str().unwrap(),
            "-p",
            "hello",
            "--no-context",
        ])
        .env("HOME", &home)
        .env_remove("USERPROFILE")
        .env("RI_MODELS_FILE", &models)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(text(&output.stderr).contains(&canonical_session.display().to_string()));
    assert!(output.stdout.is_empty());
    let _ = fs::remove_dir_all(home);
}

#[cfg(unix)]
#[test]
fn json_interrupt_preserves_ndjson_and_reports_cancellation() {
    let _lock = cli_test_lock();
    let home = unique_dir("cli-json-interrupt");
    fs::create_dir_all(&home).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (accepted_tx, accepted_rx) = std::sync::mpsc::channel();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        accepted_tx.send(()).unwrap();
        let _ = read_request(&mut stream);
        let _ = stream.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
        );
        let mut buffer = [0_u8; 1024];
        while stream.read(&mut buffer).unwrap_or(0) > 0 {}
    });
    let models = home.join("models.json");
    fs::write(
        &models,
        format!(
            r#"{{"providers":{{"test":{{"baseUrl":"http://{}","api":"openai-completions","models":[{{"id":"model"}}]}}}}}}"#,
            address
        ),
    )
    .unwrap();

    let child = Command::new(env!("CARGO_BIN_EXE_ri"))
        .args(["--json", "-p", "hello", "--no-session", "--no-context"])
        .env("HOME", &home)
        .env_remove("USERPROFILE")
        .env("RI_MODELS_FILE", &models)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    accepted_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("provider request should reach the test server");
    std::thread::sleep(std::time::Duration::from_millis(100));
    Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .unwrap();
    let output = child.wait_with_output().unwrap();
    server.join().unwrap();

    assert_eq!(
        output.status.code(),
        Some(1),
        "stderr: {}",
        text(&output.stderr)
    );
    let records = parse_records(&output.stdout);
    assert!(records.iter().any(|record| {
        record["type"] == "turn_finished" && record["data"]["reason"] == "cancelled"
    }));
    assert_eq!(records.last().unwrap()["type"], "run_finished");
    assert_eq!(records.last().unwrap()["data"]["success"], false);
    let _ = fs::remove_dir_all(home);
}

#[test]
fn help_and_version_do_not_require_configuration() {
    let _lock = cli_test_lock();
    for args in [["--help"].as_slice(), &["--version"]] {
        let output = Command::new(env!("CARGO_BIN_EXE_ri"))
            .args(args)
            .env_remove("HOME")
            .env_remove("USERPROFILE")
            .env_remove("RI_MODELS_FILE")
            .output()
            .unwrap();
        assert!(output.status.success(), "stderr: {}", text(&output.stderr));
        assert!(!output.stdout.is_empty());
    }
    let version = Command::new(env!("CARGO_BIN_EXE_ri"))
        .arg("-V")
        .env_remove("HOME")
        .env_remove("USERPROFILE")
        .env_remove("RI_MODELS_FILE")
        .output()
        .unwrap();
    assert_eq!(text(&version.stdout), "ri 0.1.0\n");
}

#[test]
fn setup_and_cli_errors_use_status_two() {
    let _lock = cli_test_lock();
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
        .args(["--no-session", "--no-context"])
        .env("HOME", &home)
        .env_remove("RI_MODELS_FILE")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(text(&output.stderr).contains("no models.json found"));

    let empty_models = home.join("empty-models.json");
    fs::write(&empty_models, r#"{"providers":{}}"#).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_ri"))
        .args(["--no-session", "--no-context"])
        .env("HOME", &home)
        .env("RI_MODELS_FILE", &empty_models)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(text(&output.stderr).contains("no selectable model"));

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

fn tool_call_body() -> &'static str {
    concat!(
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"function\":{\"name\":\"bash\",\"arguments\":\"{\\\"command\\\":\\\"printf tool-output\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: [DONE]\n\n"
    )
}

fn success_body() -> &'static str {
    concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n",
        "data: [DONE]\n\n"
    )
}

fn responses_final_item_body() -> &'static str {
    concat!(
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"authoritative\"}]}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n",
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
        Self::in_home(unique_dir("cli-fixture"), status, body, api_key)
    }

    fn in_home(
        home: PathBuf,
        status: &'static str,
        body: &'static str,
        api_key: Option<&str>,
    ) -> Self {
        Self::in_home_with_responses(home, vec![(status, body)], api_key)
    }

    fn responses(body: &'static str) -> Self {
        Self::in_home_with_responses_and_models(
            unique_dir("cli-responses-fixture"),
            vec![("200 OK", body)],
            None,
            "openai-responses",
            &["model"],
        )
    }

    fn with_models(model_ids: &[&str], body: &'static str) -> Self {
        Self::in_home_with_models(unique_dir("cli-state-fixture"), model_ids, body)
    }

    fn in_home_with_models(home: PathBuf, model_ids: &[&str], body: &'static str) -> Self {
        Self::in_home_with_responses_and_models(
            home,
            vec![("200 OK", body)],
            None,
            "openai-completions",
            model_ids,
        )
    }

    fn in_home_with_responses(
        home: PathBuf,
        responses: Vec<(&'static str, &'static str)>,
        api_key: Option<&str>,
    ) -> Self {
        Self::in_home_with_responses_and_models(
            home,
            responses,
            api_key,
            "openai-completions",
            &["model"],
        )
    }

    fn in_home_with_responses_and_models(
        home: PathBuf,
        responses: Vec<(&'static str, &'static str)>,
        api_key: Option<&str>,
        api: &'static str,
        model_ids: &[&str],
    ) -> Self {
        fs::create_dir_all(&home).unwrap();
        let models_path = home.join("models.json");
        let server = spawn_servers(responses);
        let api_key = api_key
            .map(|value| format!(",\"apiKey\":{value:?}"))
            .unwrap_or_default();
        let model_entries = model_ids
            .iter()
            .map(|id| format!(r#"{{"id":"{id}","contextWindow":200000}}"#))
            .collect::<Vec<_>>()
            .join(",");
        let source = format!(
            r#"{{"providers":{{"test":{{"baseUrl":"{}","api":"{}"{} ,"models":[{}]}}}}}}"#,
            server.url.as_str(),
            api,
            api_key,
            model_entries
        );
        fs::write(&models_path, source).unwrap();
        Self {
            home,
            models: models_path,
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
        let home = self.keep_home();
        let _ = fs::remove_dir_all(home);
    }

    fn keep_home(self) -> PathBuf {
        self.server.join();
        self.home
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

fn spawn_servers(responses: Vec<(&'static str, &'static str)>) -> TestServer {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let join = thread::spawn(move || {
        for (status, body) in responses {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = read_request(&mut stream);
            let response = format!(
                "HTTP/1.1 {status}\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
        }
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

fn valid_models() -> &'static str {
    r#"{"providers":{"test":{"baseUrl":"https://example.test","api":"openai-responses","models":[{"id":"model"}]}}}"#
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
