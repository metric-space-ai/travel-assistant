use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use serde_json::json;
use std::{error::Error, string::FromUtf8Error};
use strum_macros::Display;
use tracing::error;

#[derive(Debug, Display)]
pub enum AppError {
    Axum(axum::Error),
    Generic(Box<dyn Error + Send + Sync>),
    Utf8(FromUtf8Error),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            AppError::Axum(error) => {
                error!("Error: {:?}", error);

                (StatusCode::INTERNAL_SERVER_ERROR, "Axum error.")
            }
            AppError::Generic(error) => {
                error!("Error: {:?}", error);

                (StatusCode::INTERNAL_SERVER_ERROR, "Generic error.")
            }
            AppError::Utf8(error) => {
                error!("Error: {:?}", error);

                (StatusCode::INTERNAL_SERVER_ERROR, "Utf8 error.")
            }
        };

        let body = Json(json!(ResponseError {
            error: error_message.to_string(),
        }));

        (status, body).into_response()
    }
}

impl From<axum::Error> for AppError {
    fn from(inner: axum::Error) -> Self {
        AppError::Axum(inner)
    }
}

impl From<Box<dyn Error + Send + Sync>> for AppError {
    fn from(inner: Box<dyn Error + Send + Sync>) -> Self {
        AppError::Generic(inner)
    }
}

impl From<FromUtf8Error> for AppError {
    fn from(inner: FromUtf8Error) -> Self {
        AppError::Utf8(inner)
    }
}

impl serde::ser::StdError for AppError {}

#[derive(Debug, Serialize)]
pub struct ResponseError {
    pub error: String,
}
