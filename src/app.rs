use crate::{api, config, context::Context, Args};
use axum::Router;
use std::sync::Arc;

pub struct App {
    pub context: Arc<Context>,
    pub router: Router,
}

pub fn get(args: Args) -> App {
    let context = Arc::new(get_context(args));

    App {
        context: context.clone(),
        router: api::router(context),
    }
}

pub fn get_context(args: Args) -> Context {
    let config = config::load(args);

    Context::new(config)
}
