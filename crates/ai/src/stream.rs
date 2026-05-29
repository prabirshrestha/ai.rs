use crate::AssistantMessageEventStream;
use crate::api_registry::get_api_provider;
use crate::providers::register_builtins::ensure_builtins_registered;
use crate::types::{AssistantMessage, Context, Model, SimpleStreamOptions, StreamOptions};
use crate::{Error, Result};

pub fn stream(
    model: Model,
    context: Context,
    options: Option<StreamOptions>,
) -> Result<AssistantMessageEventStream> {
    ensure_builtins_registered();
    let provider =
        get_api_provider(&model.api).ok_or_else(|| Error::UnsupportedApi(model.api.clone()))?;
    (provider.stream)(model, context, options.unwrap_or_default())
}

pub async fn complete(
    model: Model,
    context: Context,
    options: Option<StreamOptions>,
) -> Result<AssistantMessage> {
    let mut stream = stream(model, context, options)?;
    while futures::StreamExt::next(&mut stream).await.is_some() {}
    stream.result().await
}

pub fn stream_simple(
    model: Model,
    context: Context,
    options: Option<SimpleStreamOptions>,
) -> Result<AssistantMessageEventStream> {
    let options = options.unwrap_or_default();
    ensure_builtins_registered();
    let provider =
        get_api_provider(&model.api).ok_or_else(|| Error::UnsupportedApi(model.api.clone()))?;
    (provider.stream_simple)(model, context, options)
}

pub async fn complete_simple(
    model: Model,
    context: Context,
    options: Option<SimpleStreamOptions>,
) -> Result<AssistantMessage> {
    let mut stream = stream_simple(model, context, options)?;
    while futures::StreamExt::next(&mut stream).await.is_some() {}
    stream.result().await
}
