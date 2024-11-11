use crate::Args;

#[derive(Clone, Debug)]
pub struct Config {
    pub port: u16,
}

impl Config {
    pub fn new(port: u16) -> Self {
        Self { port }
    }
}

pub fn load(args: Args) -> Config {
    let mut port = 8080;

    if let Some(val) = args.port {
        port = val;
    }

    Config::new(port)
}
