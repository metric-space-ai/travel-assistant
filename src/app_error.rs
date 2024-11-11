use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use serde_json::json;
use std::error::Error;
use strum_macros::Display;
use tracing::error;

#[derive(Debug, Display)]
pub enum AppError {
    Generic(Box<dyn Error + Send + Sync>),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            AppError::Generic(error) => {
                error!("Error: {:?}", error);

                (StatusCode::INTERNAL_SERVER_ERROR, "Generic error.")
            }
        };

        let body = Json(json!(ResponseError {
            error: error_message.to_string(),
        }));

        (status, body).into_response()
    }
}

impl From<Box<dyn Error + Send + Sync>> for AppError {
    fn from(inner: Box<dyn Error + Send + Sync>) -> Self {
        AppError::Generic(inner)
    }
}

impl serde::ser::StdError for AppError {}

#[derive(Debug, Serialize)]
pub struct ResponseError {
    pub error: String,
}
