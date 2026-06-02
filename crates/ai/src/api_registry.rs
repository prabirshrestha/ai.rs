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

fn registry() -> &'static RwLock<Vec<RegisteredApiProvider>> {
    static REGISTRY: OnceLock<RwLock<Vec<RegisteredApiProvider>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(Vec::new()))
}

fn mismatched_api_error(actual: &str, expected: &str) -> Error {
    Error::UnsupportedApi(format!("Mismatched api: {actual} expected {expected}"))
}

fn wrap_registered_provider(provider: ApiProvider) -> ApiProvider {
    let api = provider.api;
    let expected_api = api.clone();
    let stream = provider.stream;
    let stream_simple = provider.stream_simple;

    ApiProvider {
        api,
        stream: Arc::new({
            let expected_api = expected_api.clone();
            move |model, context, options| {
                if model.api.as_str() != expected_api.as_str() {
                    return Err(mismatched_api_error(&model.api, &expected_api));
                }
                stream(model, context, options)
            }
        }),
        stream_simple: Arc::new(move |model, context, options| {
            if model.api.as_str() != expected_api.as_str() {
                return Err(mismatched_api_error(&model.api, &expected_api));
            }
            stream_simple(model, context, options)
        }),
    }
}

pub fn register_api_provider(provider: ApiProvider, source_id: Option<String>) {
    let provider = wrap_registered_provider(provider);
    let mut registry = registry().write().expect("api registry poisoned");
    if let Some(existing) = registry
        .iter_mut()
        .find(|entry| entry.provider.api == provider.api)
    {
        *existing = RegisteredApiProvider {
            provider,
            source_id,
        };
    } else {
        registry.push(RegisteredApiProvider {
            provider,
            source_id,
        });
    }
}

pub fn get_api_provider(api: &str) -> Option<ApiProvider> {
    registry()
        .read()
        .expect("api registry poisoned")
        .iter()
        .find(|entry| entry.provider.api == api)
        .map(|entry| entry.provider.clone())
}

pub fn unregister_api_providers(source_id: &str) {
    registry()
        .write()
        .expect("api registry poisoned")
        .retain(|entry| entry.source_id.as_deref() != Some(source_id));
}

pub fn clear_api_providers() {
    registry().write().expect("api registry poisoned").clear();
}

pub(crate) fn wrap_stream<F>(api: &'static str, stream: F) -> ApiStreamFunction
where
    F: Fn(Model, Context, StreamOptions) -> Result<AssistantMessageEventStream>
        + Send
        + Sync
        + 'static,
{
    Arc::new(move |model, context, options| {
        if model.api != api {
            return Err(mismatched_api_error(&model.api, api));
        }
        stream(model, context, options)
    })
}

pub(crate) fn wrap_stream_simple<F>(api: &'static str, stream: F) -> ApiStreamSimpleFunction
where
    F: Fn(Model, Context, SimpleStreamOptions) -> Result<AssistantMessageEventStream>
        + Send
        + Sync
        + 'static,
{
    Arc::new(move |model, context, options| {
        if model.api != api {
            return Err(mismatched_api_error(&model.api, api));
        }
        stream(model, context, options)
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    use crate::types::{
        AssistantContent, AssistantMessage, AssistantMessageEvent, ModelCost, StopReason,
        TextContent, Usage,
    };

    use super::*;

    fn test_model(api: &str) -> Model {
        Model {
            id: "mock".to_string(),
            name: "mock".to_string(),
            api: api.to_string(),
            provider: "test-provider".to_string(),
            base_url: "https://example.invalid".to_string(),
            reasoning: false,
            input: vec![crate::ModelInput::Text],
            cost: ModelCost::default(),
            context_window: 8192,
            max_tokens: 2048,
            ..Model::default()
        }
    }

    fn assistant_text(text: &str) -> AssistantMessage {
        AssistantMessage {
            content: vec![AssistantContent::Text(TextContent {
                text: text.to_string(),
                text_signature: None,
            })],
            api: "test-custom-api".to_string(),
            provider: "test-provider".to_string(),
            model: "mock".to_string(),
            response_model: None,
            response_id: None,
            diagnostics: Vec::new(),
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: crate::utils::time::now_millis(),
        }
    }

    fn done_stream(message: AssistantMessage) -> AssistantMessageEventStream {
        let reason = message.stop_reason;
        let (mut sender, stream) = AssistantMessageEventStream::channel();
        sender.push(AssistantMessageEvent::Done { reason, message });
        stream
    }

    #[tokio::test]
    async fn registered_provider_dispatches_stream_simple() {
        let source_id = "api-registry-test";
        let called = Arc::new(AtomicBool::new(false));
        register_api_provider(
            ApiProvider {
                api: "test-custom-api".to_string(),
                stream: wrap_stream("test-custom-api", |_model, _context, _options| {
                    Ok(done_stream(assistant_text("stream")))
                }),
                stream_simple: wrap_stream_simple("test-custom-api", {
                    let called = called.clone();
                    move |_model, _context, _options| {
                        called.store(true, Ordering::SeqCst);
                        Ok(done_stream(assistant_text("simple")))
                    }
                }),
            },
            Some(source_id.to_string()),
        );

        let mut stream =
            crate::stream_simple(test_model("test-custom-api"), Context::default(), None)
                .expect("custom provider stream");
        while futures::StreamExt::next(&mut stream).await.is_some() {}
        let message = stream.result().await.expect("stream result");

        assert!(called.load(Ordering::SeqCst));
        assert!(matches!(
            message.content.first(),
            Some(AssistantContent::Text(text)) if text.text == "simple"
        ));

        unregister_api_providers(source_id);
        assert!(get_api_provider("test-custom-api").is_none());
    }

    #[test]
    fn registered_provider_rejects_mismatched_api_without_manual_wrap() {
        let source_id = "api-registry-auto-wrap-test";
        let called = Arc::new(AtomicBool::new(false));
        register_api_provider(
            ApiProvider {
                api: "test-auto-wrap-api".to_string(),
                stream: Arc::new(|_model, _context, _options| {
                    Ok(done_stream(assistant_text("unreachable")))
                }),
                stream_simple: {
                    let called = called.clone();
                    Arc::new(move |_model, _context, _options| {
                        called.store(true, Ordering::SeqCst);
                        Ok(done_stream(assistant_text("unreachable")))
                    })
                },
            },
            Some(source_id.to_string()),
        );

        let provider = get_api_provider("test-auto-wrap-api").expect("registered provider");
        let error = match (provider.stream_simple)(
            test_model("actual-api"),
            Context::default(),
            SimpleStreamOptions::default(),
        ) {
            Ok(_) => panic!("expected mismatched api error"),
            Err(error) => error,
        };

        assert!(!called.load(Ordering::SeqCst));
        assert!(
            matches!(error, Error::UnsupportedApi(message) if message == "Mismatched api: actual-api expected test-auto-wrap-api")
        );
        unregister_api_providers(source_id);
    }

    #[test]
    fn wrapped_provider_rejects_mismatched_api() {
        let provider = wrap_stream_simple("expected-api", |_model, _context, _options| {
            Ok(done_stream(assistant_text("unreachable")))
        });
        let error = match provider(
            test_model("actual-api"),
            Context::default(),
            SimpleStreamOptions::default(),
        ) {
            Ok(_) => panic!("expected mismatched api error"),
            Err(error) => error,
        };

        assert!(
            matches!(error, Error::UnsupportedApi(message) if message == "Mismatched api: actual-api expected expected-api")
        );
    }
}
