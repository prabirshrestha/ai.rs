use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use crate::event_stream::AssistantMessageEventStream;
use crate::types::{Context, Model, SimpleStreamOptions, StreamOptions};
use crate::{Error, Result};

pub type ApiStreamFunction =
    Arc<dyn Fn(Model, Context, StreamOptions) -> Result<AssistantMessageEventStream> + Send + Sync>;
pub type ApiStreamSimpleFunction = Arc<
    dyn Fn(Model, Context, SimpleStreamOptions) -> Result<AssistantMessageEventStream>
        + Send
        + Sync,
>;

#[derive(Clone)]
pub struct ApiProvider {
    pub api: String,
    pub stream: ApiStreamFunction,
    pub stream_simple: ApiStreamSimpleFunction,
}

#[derive(Clone)]
struct RegisteredApiProvider {
    provider: ApiProvider,
    source_id: Option<String>,
}

fn registry() -> &'static RwLock<HashMap<String, RegisteredApiProvider>> {
    static REGISTRY: OnceLock<RwLock<HashMap<String, RegisteredApiProvider>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

pub fn register_api_provider(provider: ApiProvider, source_id: Option<String>) {
    registry().write().expect("api registry poisoned").insert(
        provider.api.clone(),
        RegisteredApiProvider {
            provider,
            source_id,
        },
    );
}

pub fn get_api_provider(api: &str) -> Option<ApiProvider> {
    registry()
        .read()
        .expect("api registry poisoned")
        .get(api)
        .map(|entry| entry.provider.clone())
}

pub fn get_api_providers() -> Vec<ApiProvider> {
    registry()
        .read()
        .expect("api registry poisoned")
        .values()
        .map(|entry| entry.provider.clone())
        .collect()
}

pub fn unregister_api_providers(source_id: &str) {
    registry()
        .write()
        .expect("api registry poisoned")
        .retain(|_, entry| entry.source_id.as_deref() != Some(source_id));
}

pub fn clear_api_providers() {
    registry().write().expect("api registry poisoned").clear();
}

pub fn wrap_stream<F>(api: &'static str, stream: F) -> ApiStreamFunction
where
    F: Fn(Model, Context, StreamOptions) -> Result<AssistantMessageEventStream>
        + Send
        + Sync
        + 'static,
{
    Arc::new(move |model, context, options| {
        if model.api != api {
            return Err(Error::UnsupportedApi(format!(
                "Mismatched api: {} expected {}",
                model.api, api
            )));
        }
        stream(model, context, options)
    })
}

pub fn wrap_stream_simple<F>(api: &'static str, stream: F) -> ApiStreamSimpleFunction
where
    F: Fn(Model, Context, SimpleStreamOptions) -> Result<AssistantMessageEventStream>
        + Send
        + Sync
        + 'static,
{
    Arc::new(move |model, context, options| {
        if model.api != api {
            return Err(Error::UnsupportedApi(format!(
                "Mismatched api: {} expected {}",
                model.api, api
            )));
        }
        stream(model, context, options)
    })
}
