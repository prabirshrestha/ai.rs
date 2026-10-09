//! Port of `providers/cloudflare-stream.ts`: fills the Cloudflare endpoint
//! placeholders (`{CLOUDFLARE_ACCOUNT_ID}`, `{CLOUDFLARE_GATEWAY_ID}`) of a
//! model's base URL from the resolved provider env before dispatch.
//!
//! Divergence: Pi's generic `resolveCloudflareModel<TModel>` is generic over
//! the [`CloudflareModel`] trait, implemented for chat and classifier models.

use std::sync::Arc;

use async_trait::async_trait;

use crate::types::{
    ClassifierContext, ClassifierModel, ClassifierOptions, ClassifierResult, Model,
    ProviderClassifier, ProviderEnv, ProviderStreams, SimpleStreamOptions, StreamOptions,
    TranscriptContext,
};
use crate::utils::event_stream::AssistantMessageEventStream;

const CLOUDFLARE_ACCOUNT_ID: &str = "CLOUDFLARE_ACCOUNT_ID";
const CLOUDFLARE_GATEWAY_ID: &str = "CLOUDFLARE_GATEWAY_ID";

/// A model with a base URL that may hold Cloudflare placeholders.
pub trait CloudflareModel {
    fn base_url_mut(&mut self) -> &mut String;
}

impl CloudflareModel for Model {
    fn base_url_mut(&mut self) -> &mut String {
        &mut self.base_url
    }
}

impl CloudflareModel for ClassifierModel {
    fn base_url_mut(&mut self) -> &mut String {
        &mut self.base_url
    }
}

/// `resolveCloudflareModel(model, env)`: a placeholder without an env value is
/// left in place.
pub fn resolve_cloudflare_model<M: CloudflareModel>(mut model: M, env: Option<&ProviderEnv>) -> M {
    let Some(env) = env else {
        return model;
    };
    let base_url = model.base_url_mut();
    let mut resolved = base_url.clone();
    for name in [CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_GATEWAY_ID] {
        let placeholder = format!("{{{name}}}");
        let value = env
            .get(name)
            .cloned()
            .unwrap_or_else(|| placeholder.clone());
        resolved = resolved.replace(&placeholder, &value);
    }
    *base_url = resolved;
    model
}

/// `cloudflareStreams(streams)`: wraps an API implementation so Cloudflare
/// account/gateway endpoint placeholders materialize from the resolved
/// provider env before dispatch.
pub fn cloudflare_streams(streams: Arc<dyn ProviderStreams>) -> Arc<dyn ProviderStreams> {
    Arc::new(CloudflareStreams { streams })
}

struct CloudflareStreams {
    streams: Arc<dyn ProviderStreams>,
}

impl ProviderStreams for CloudflareStreams {
    fn stream(
        &self,
        model: Model,
        context: TranscriptContext,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        let model = resolve_cloudflare_model(model, options.env.as_ref());
        self.streams.stream(model, context, options)
    }

    fn stream_simple(
        &self,
        model: Model,
        context: TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        let model = resolve_cloudflare_model(model, options.env.as_ref());
        self.streams.stream_simple(model, context, options)
    }
}

/// `cloudflareClassifier(classifier)`: classifier counterpart of
/// [`cloudflare_streams`].
pub fn cloudflare_classifier(
    classifier: Arc<dyn ProviderClassifier>,
) -> Arc<dyn ProviderClassifier> {
    Arc::new(CloudflareClassifier { classifier })
}

struct CloudflareClassifier {
    classifier: Arc<dyn ProviderClassifier>,
}

#[async_trait]
impl ProviderClassifier for CloudflareClassifier {
    async fn classify(
        &self,
        model: ClassifierModel,
        context: ClassifierContext,
        options: ClassifierOptions,
    ) -> ClassifierResult {
        let model = resolve_cloudflare_model(model, options.env.as_ref());
        self.classifier.classify(model, context, options).await
    }
}

#[cfg(test)]
mod tests {
    //! Port of `test/cloudflare-stream.test.ts`.

    use parking_lot::Mutex;
    use serde_json::json;

    use super::*;
    use crate::api::openai_client::test_support::{context, model};

    fn gateway_model() -> Model {
        model(json!({
            "id": "model",
            "name": "model",
            "api": "openai-completions",
            "provider": "cloudflare-ai-gateway",
            "baseUrl": "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/openai",
            "reasoning": false,
            "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 1000,
            "maxTokens": 100,
        }))
    }

    struct Capturing(Arc<Mutex<Vec<String>>>);

    impl ProviderStreams for Capturing {
        fn stream(
            &self,
            model: Model,
            _: TranscriptContext,
            _: StreamOptions,
        ) -> AssistantMessageEventStream {
            self.0.lock().push(model.base_url);
            AssistantMessageEventStream::new()
        }

        fn stream_simple(
            &self,
            model: Model,
            _: TranscriptContext,
            _: SimpleStreamOptions,
        ) -> AssistantMessageEventStream {
            self.0.lock().push(model.base_url);
            AssistantMessageEventStream::new()
        }
    }

    #[test]
    fn materializes_the_model_endpoint_before_dispatch() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let streams = cloudflare_streams(Arc::new(Capturing(captured.clone())));
        let env = ProviderEnv::from([
            ("CLOUDFLARE_ACCOUNT_ID".to_string(), "account".to_string()),
            ("CLOUDFLARE_GATEWAY_ID".to_string(), "gateway".to_string()),
        ]);

        streams.stream(
            gateway_model(),
            context(json!({ "messages": [] })),
            StreamOptions {
                env: Some(env.clone()),
                ..Default::default()
            },
        );
        streams.stream_simple(
            gateway_model(),
            context(json!({ "messages": [] })),
            SimpleStreamOptions {
                stream: StreamOptions {
                    env: Some(env),
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        assert_eq!(
            *captured.lock(),
            [
                "https://gateway.ai.cloudflare.com/v1/account/gateway/openai",
                "https://gateway.ai.cloudflare.com/v1/account/gateway/openai",
            ]
        );
    }

    #[test]
    fn keeps_placeholders_when_the_provider_env_does_not_resolve_them() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let streams = cloudflare_streams(Arc::new(Capturing(captured.clone())));

        streams.stream_simple(
            gateway_model(),
            context(json!({ "messages": [] })),
            SimpleStreamOptions::default(),
        );

        assert_eq!(*captured.lock(), [gateway_model().base_url]);
    }
}
