//! Port of `providers/typesafe.ts`: TypeSafe's System One classifiers.

use std::sync::Arc;

use super::catalog::typesafe_classifier_models;
use crate::api::typesafe_system_one::typesafe_system_one_api;
use crate::auth::{ProviderAuth, env_api_key_auth};
use crate::models::{CreateProviderOptions, Provider, create_provider};
use crate::types::{AnyModel, KnownClassifierApi};

/// `typesafeProvider()`.
pub fn typesafe_provider() -> Arc<dyn Provider> {
    create_provider(CreateProviderOptions {
        id: "typesafe".to_string(),
        name: Some("TypeSafe".to_string()),
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth("TypeSafe API key", &["TYPESAFE_API_KEY"])),
            oauth: None,
        },
        models: typesafe_classifier_models()
            .values()
            .cloned()
            .map(AnyModel::Classifier)
            .collect(),
        classifiers: Some(
            [(
                KnownClassifierApi::TypesafeSystemOne.as_str().to_string(),
                typesafe_system_one_api(),
            )]
            .into_iter()
            .collect(),
        ),
        ..Default::default()
    })
    .expect("the TypeSafe provider has a classifier implementation")
}
