use clap::Parser;
use std::error::Error;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod api;
mod app;
mod app_error;
mod calendar_agent;
mod config;
mod context;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Clone, Copy, Debug, Parser)]
#[command(about, author, long_about = None, version)]
pub struct Args {
    /// Port
    #[arg(short, long)]
    pub port: Option<u16>,
}

#[allow(clippy::missing_errors_doc)]
pub async fn run() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                "travel_assistant=error,runtime=error,tokio=error,tower_http=error".into()
            }),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();

    let app = app::get(args);

    let listener =
        tokio::net::TcpListener::bind(format!("0.0.0.0:{}", app.context.config.port)).await?;

    info!("listening on {}", listener.local_addr()?);

    axum::serve(listener, app.router).await?;

    Ok(())
}
