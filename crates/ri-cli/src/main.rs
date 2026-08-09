use ri::{app, logging};

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
    if options.show_version {
        app::Options::print_version();
        return 0;
    }
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
        Err(error) => {
            eprintln!("ri: error: {error}");
            run_error_exit_code(&error)
        }
    }
}

fn run_error_exit_code(error: &app::RunError) -> i32 {
    match error {
        app::RunError::Setup(_) => 2,
        app::RunError::Runtime(_) => 1,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn exit_codes_are_stable() {
        assert_eq!(
            super::run_error_exit_code(&ri::app::RunError::Setup(anyhow::anyhow!("setup"))),
            2
        );
        assert_eq!(
            super::run_error_exit_code(&ri::app::RunError::Runtime(anyhow::anyhow!("runtime"))),
            1
        );
    }
}
