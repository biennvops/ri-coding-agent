mod app;
mod input;
mod json_output;
mod logging;
mod model_picker;
mod model_selection;
mod picker;
mod render;
mod session_picker;
mod terminal;

#[tokio::main]
async fn main() {
    std::process::exit(run_main().await);
}

async fn run_main() -> i32 {
    let options = match app::Options::parse(std::env::args().skip(1)) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("ri: error: {error}");
            return 2;
        }
    };
    if options.show_help {
        app::Options::print_help();
        return 0;
    }

    let log_path = match logging::init() {
        Ok(path) => path,
        Err(error) => {
            eprintln!("ri: error: {error}");
            return 2;
        }
    };
    if let Some(path) = log_path {
        eprintln!("ri: logging to {}", path.display());
        tracing::info!(target: "ri", log_path = %path.display(), "diagnostic logging enabled");
    }

    match app::run(options).await {
        Ok(()) => 0,
        Err(app::RunError::Setup(error)) => {
            eprintln!("ri: error: {error}");
            2
        }
        Err(app::RunError::Runtime(error)) => {
            eprintln!("ri: error: {error}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn exit_codes_are_stable() {
        assert_eq!(super::run_main_exit_code_for_test(None), 0);
        assert_eq!(
            super::run_main_exit_code_for_test(Some(crate::app::RunError::Setup(anyhow::anyhow!(
                "setup"
            ),))),
            2
        );
        assert_eq!(
            super::run_main_exit_code_for_test(Some(crate::app::RunError::Runtime(
                anyhow::anyhow!("runtime"),
            ))),
            1
        );
    }
}

#[cfg(test)]
fn run_main_exit_code_for_test(error: Option<app::RunError>) -> i32 {
    match error {
        None => 0,
        Some(app::RunError::Setup(_)) => 2,
        Some(app::RunError::Runtime(_)) => 1,
    }
}
