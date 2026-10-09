//! Port of `providers/cloudflare-workers-ai.ts`: Workers AI chat models over
//! the OpenAI-compatible Chat Completions endpoint and the System One
//! classifiers over the Workers AI REST API. The account ID comes from the
//! resolved provider env (`CLOUDFLARE_ACCOUNT_ID`).

use std::sync::Arc;

use super::catalog::{cloudflare_workers_ai_classifier_models, cloudflare_workers_ai_models};
use super::cloudflare_auth::cloudflare_workers_ai_auth;
use super::cloudflare_stream::{cloudflare_classifier, cloudflare_streams};
use crate::api::cloudflare_workers_ai_system_one::cloudflare_workers_ai_system_one_api;
use crate::api::openai_completions::openai_completions_api;
use crate::auth::ProviderAuth;
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::types::{AnyModel, KnownClassifierApi};

/// `cloudflareWorkersAIProvider()`.
pub fn cloudflare_workers_ai_provider() -> Arc<dyn Provider> {
    create_provider(CreateProviderOptions {
        id: "cloudflare-workers-ai".to_string(),
        name: Some("Cloudflare Workers AI".to_string()),
        auth: ProviderAuth {
            api_key: Some(cloudflare_workers_ai_auth()),
            oauth: None,
        },
        models: cloudflare_workers_ai_models()
            .values()
            .cloned()
            .map(AnyModel::Chat)
            .chain(
                cloudflare_workers_ai_classifier_models()
                    .values()
                    .cloned()
                    .map(AnyModel::Classifier),
            )
            .collect(),
        api: Some(ProviderApi::Single(cloudflare_streams(
            openai_completions_api(),
        ))),
        classifiers: Some(
            [(
                KnownClassifierApi::CloudflareWorkersAiSystemOne
                    .as_str()
                    .to_string(),
                cloudflare_classifier(cloudflare_workers_ai_system_one_api()),
            )]
            .into_iter()
            .collect(),
        ),
        ..Default::default()
    })
    .expect("the Cloudflare Workers AI provider has an api implementation")
}
