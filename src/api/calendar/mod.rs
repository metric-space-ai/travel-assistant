use crate::{app_error::AppError, context::Context};
use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use std::sync::Arc;

#[axum_macros::debug_handler]
pub async fn action(State(_context): State<Arc<Context>>) -> Result<impl IntoResponse, AppError> {
    Ok((StatusCode::OK, Json("{}")).into_response())
}
