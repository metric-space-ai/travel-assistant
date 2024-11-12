use crate::{app_error::AppError, calendar_agent, context::Context};
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
    State(context): State<Arc<Context>>,
    request: Request<Body>,
) -> Result<impl IntoResponse, AppError> {
    let body = BodyExt::collect(request.into_body())
        .await?
        .to_bytes()
        .to_vec();

    let payload = String::from_utf8(body)?;

    let response = calendar_agent::agent(context, payload).await?;

    tracing::info!("{:?}", response);
    Ok((StatusCode::OK, Json("{}")).into_response())
}
