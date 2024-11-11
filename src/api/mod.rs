use crate::context::Context;
use axum::{
    error_handling::HandleErrorLayer,
    http::{header, Method, StatusCode},
    routing::post,
    Router,
};
use std::{sync::Arc, time::Duration};
use tower::{BoxError, ServiceBuilder};
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};

mod calendar;

#[allow(clippy::too_many_lines)]
pub fn router(context: Arc<Context>) -> Router {
    Router::new()
        .route("/api/v1/calendar", post(calendar::action))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(vec![
                    Method::DELETE,
                    Method::GET,
                    Method::OPTIONS,
                    Method::POST,
                    Method::PUT,
                ])
                .allow_headers(vec![header::CONTENT_TYPE]),
        )
        .layer(
            ServiceBuilder::new()
                .layer(HandleErrorLayer::new(|error: BoxError| async move {
                    if error.is::<tower::timeout::error::Elapsed>() {
                        Ok(StatusCode::REQUEST_TIMEOUT)
                    } else {
                        Err((
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Unhandled internal error: {error}"),
                        ))
                    }
                }))
                .timeout(Duration::from_secs(60))
                .layer(TraceLayer::new_for_http())
                .into_inner(),
        )
        .with_state(context)
}
