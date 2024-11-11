use crate::Args;

#[derive(Clone, Debug)]
pub struct Config {
    pub openai_api_key: String,
    pub port: u16,
}

impl Config {
    pub fn new(openai_api_key: String, port: u16) -> Self {
        Self {
            openai_api_key,
            port,
        }
    }
}

pub fn load(args: Args) -> Config {
    let mut openai_api_key: Option<String> = None;
    let mut port = 8080;

    if let Ok(val) = std::env::var("OPENAI_API_KEY") {
        openai_api_key = Some(val);
    }

    if let Some(val) = args.port {
        port = val;
    }

    Config::new(openai_api_key.expect("Unknown OpenAI API key"), port)
}
