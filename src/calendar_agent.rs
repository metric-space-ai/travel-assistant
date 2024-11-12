use crate::{context::Context, Result};
use minicaldav::{Credentials, Event};
use std::sync::Arc;
use ureq::Agent;
use url::Url;

pub async fn agent(context: Arc<Context>, payload: String) -> Result<String> {
    let agent = Agent::new();
    let url = Url::parse(&context.config.caldav_url)?;
    let credentials = Credentials::Basic(
        context.config.caldav_username.clone(),
        context.config.caldav_password.clone(),
    );
    let calendars = minicaldav::get_calendars(agent.clone(), &credentials, &url)?;
    tracing::info!("calendars {:?}", calendars);
    for calendar in calendars {
        tracing::info!("calendar {:?}", calendar);
        let new_event = Event::builder(calendar.url().clone())
            .summary("Sample event summary".to_string())
            .description(Some("Sample event description".to_string()))
            .end("20241112T210000Z".to_string(), vec![])
            .etag(Some("686897696a7c876b7e".to_string()))
            .location(Some("Sample event location".to_string()))
            .priority("1".to_string())
            .start("20241112T190000Z".to_string(), vec![])
            .status("CONFIRMED".to_string())
            .build();
        tracing::info!("new_event = {:?}", new_event);
        let new_event = minicaldav::save_event(agent.clone(), &credentials, new_event)?;
        tracing::info!("new_event = {:?}", new_event);
        let (events, errors) = minicaldav::get_events(agent.clone(), &credentials, &calendar)?;
        for event in events {
            tracing::info!("event {:?}", event);
        }
        for error in errors {
            tracing::info!("Error: {:?}", error);
        }
    }

    Ok(payload)
}
