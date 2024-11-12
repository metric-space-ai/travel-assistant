use crate::{context::Context, Result};
use minicaldav::{Credentials, Event};
use std::sync::Arc;
use ureq::Agent;
use url::Url;

pub struct EventPost {
    pub description: Option<String>,
    pub end: Option<String>,
    pub location: Option<String>,
    pub priority: Option<String>,
    pub start: String,
    pub status: Option<String>,
    pub summary: String,
}

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
        let event_post = EventPost {
            description: Some("Sample event description 2".to_string()),
            end: Some("20241112T120000Z".to_string()),
            location: Some("Sample event location 2".to_string()),
            priority: Some("1".to_string()),
            start: "20241112T110000Z".to_string(),
            status: Some("CONFIRMED".to_string()),
            summary: "Sample event summary 2".to_string(),
        };

        let new_event = save_event(
            agent.clone(),
            &credentials,
            event_post,
            calendar.url().clone(),
        )
        .await?;
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

pub async fn save_event(
    agent: Agent,
    credentials: &Credentials,
    event_post: EventPost,
    url: Url,
) -> Result<Event> {
    let event_builder = Event::builder(url)
        .description(event_post.description)
        .location(event_post.location)
        .start(event_post.start, vec![])
        .summary(event_post.summary);

    let event_builder = if let Some(end) = event_post.end {
        event_builder.end(end, vec![])
    } else {
        event_builder
    };

    let event_builder = if let Some(priority) = event_post.priority {
        event_builder.priority(priority)
    } else {
        event_builder
    };

    let event_builder = if let Some(status) = event_post.status {
        event_builder.status(status)
    } else {
        event_builder
    };

    let event = event_builder.build();

    tracing::info!("event = {:?}", event);

    let new_event = minicaldav::save_event(agent.clone(), credentials, event)?;

    Ok(new_event)
}
