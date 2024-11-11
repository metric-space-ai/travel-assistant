#![forbid(unsafe_code)]

use std::error::Error;
use tracing::error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let result = travel_assistant::run();

    if let Err(e) = result.await {
        error!("Error: {:?}", e);
    }

    Ok(())
}
