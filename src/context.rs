use crate::config::Config;

pub struct Context {
    pub config: Config,
}

impl Context {
    pub fn new(config: Config) -> Self {
        Self { config }
    }
}
