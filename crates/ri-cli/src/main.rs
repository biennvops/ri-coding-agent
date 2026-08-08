mod app;
mod input;
mod render;
mod session_picker;
mod terminal;

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    let options = app::Options::parse(std::env::args().skip(1))?;
    if options.show_help {
        app::Options::print_help();
        return Ok(());
    }

    app::run(options).await
}
