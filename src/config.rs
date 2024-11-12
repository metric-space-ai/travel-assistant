use crate::Args;

#[derive(Clone, Debug)]
pub struct Config {
    pub caldav_password: String,
    pub caldav_url: String,
    pub caldav_username: String,
    pub openai_api_key: String,
    pub port: u16,
}

impl Config {
    pub fn new(
        caldav_password: String,
        caldav_url: String,
        caldav_username: String,
        openai_api_key: String,
        port: u16,
    ) -> Self {
        Self {
            caldav_password,
            caldav_url,
            caldav_username,
            openai_api_key,
            port,
        }
    }
}

pub fn load(args: Args) -> Config {
    let mut caldav_password: Option<String> = None;
    let mut caldav_url: Option<String> = None;
    let mut caldav_username: Option<String> = None;
    let mut openai_api_key: Option<String> = None;
    let mut port = 8080;

    if let Ok(val) = std::env::var("CALDAV_PASSWORD") {
        caldav_password = Some(val);
    }

    if let Ok(val) = std::env::var("CALDAV_URL") {
        caldav_url = Some(val);
    }

    if let Ok(val) = std::env::var("CALDAV_USERNAME") {
        caldav_username = Some(val);
    }

    if let Ok(val) = std::env::var("OPENAI_API_KEY") {
        openai_api_key = Some(val);
    }

    if let Some(val) = args.port {
        port = val;
    }

    Config::new(
        caldav_password.expect("Unknown CalDAV password"),
        caldav_url.expect("Unknown CalDAV url"),
        caldav_username.expect("Unknown CalDAV username"),
        openai_api_key.expect("Unknown OpenAI API key"),
        port,
    )
}
