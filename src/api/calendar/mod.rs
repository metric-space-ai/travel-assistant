use crate::{app_error::AppError, context::Context};
use axum::{
    body::Body,
    extract::{Request, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use http_body_util::BodyExt;
use std::sync::Arc;

#[axum_macros::debug_handler]
pub async fn action(
    State(_context): State<Arc<Context>>,
    request: Request<Body>,
) -> Result<impl IntoResponse, AppError> {
    let body = BodyExt::collect(request.into_body())
        .await?
        .to_bytes()
        .to_vec();

    let payload = String::from_utf8(body)?;

    tracing::info!("{:?}", payload);
    Ok((StatusCode::OK, Json("{}")).into_response())
}
