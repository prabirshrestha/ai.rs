//! Port of `auth/credential-store.ts`.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use indexmap::IndexMap;
use parking_lot::Mutex;

use super::types::{
    AuthOperationOptions, Credential, CredentialInfo, CredentialModifier, CredentialStore,
    throw_if_aborted,
};
use crate::Result;
use crate::utils::abort::{operation_signal, race_with_abort_signal};

/// Default in-memory credential store. Apps inject persistent stores.
/// Keyed by `Provider.id`, one credential per provider; see `CredentialStore`.
/// Writes are serialized per provider (Pi: a promise chain; here a per-provider
/// async mutex).
#[derive(Default)]
pub struct InMemoryCredentialStore {
    credentials: Mutex<IndexMap<String, Credential>>,
    chains: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

impl InMemoryCredentialStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn chain(&self, provider_id: &str) -> Arc<tokio::sync::Mutex<()>> {
        self.chains
            .lock()
            .entry(provider_id.to_string())
            .or_default()
            .clone()
    }
}

#[async_trait]
impl CredentialStore for InMemoryCredentialStore {
    async fn read(
        &self,
        provider_id: &str,
        options: AuthOperationOptions,
    ) -> Result<Option<Credential>> {
        if let Some(signal) = &options.signal {
            throw_if_aborted(signal)?;
        }
        Ok(self.credentials.lock().get(provider_id).cloned())
    }

    async fn list(&self, options: AuthOperationOptions) -> Result<Vec<CredentialInfo>> {
        if let Some(signal) = &options.signal {
            throw_if_aborted(signal)?;
        }
        Ok(self
            .credentials
            .lock()
            .iter()
            .map(|(provider_id, credential)| CredentialInfo {
                provider_id: provider_id.clone(),
                credential_type: credential.credential_type(),
            })
            .collect())
    }

    async fn modify(
        &self,
        provider_id: &str,
        modifier: CredentialModifier,
        options: AuthOperationOptions,
    ) -> Result<Option<Credential>> {
        let signal = operation_signal(options.signal.as_ref());
        let chain = self.chain(provider_id);
        let queued = async {
            let _guard = chain.lock().await;
            throw_if_aborted(&signal)?;
            let current = self.credentials.lock().get(provider_id).cloned();
            let next = modifier(current.clone()).await?;
            throw_if_aborted(&signal)?;
            match next {
                Some(next) => {
                    self.credentials
                        .lock()
                        .insert(provider_id.to_string(), next.clone());
                    Ok(Some(next))
                }
                None => Ok(current),
            }
        };
        race_with_abort_signal(queued, &signal).await
    }

    async fn delete(&self, provider_id: &str, options: AuthOperationOptions) -> Result<()> {
        let signal = operation_signal(options.signal.as_ref());
        let chain = self.chain(provider_id);
        let queued = async {
            let _guard = chain.lock().await;
            throw_if_aborted(&signal)?;
            self.credentials.lock().shift_remove(provider_id);
            Ok(())
        };
        race_with_abort_signal(queued, &signal).await
    }
}
