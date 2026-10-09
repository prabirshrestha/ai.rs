//! Port of `models-store.ts`.

use std::collections::HashMap;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::Result;
use crate::auth::throw_if_aborted;
use crate::types::AnyModel;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsStoreEntry {
    /// Persisted models of every type. Deserializing drops models whose type
    /// this version does not know (Pi's `withKnownModelTypes()` on read).
    #[serde(deserialize_with = "crate::types::deserialize_known_models")]
    pub models: Vec<AnyModel>,
    /// Unix timestamp from the remote catalog's Last-Modified header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_modified: Option<u64>,
    /// Unix timestamp of the last completed remote check.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checked_at: Option<u64>,
    /// Opaque validator from the remote catalog's ETag header, stored verbatim
    /// (quotes included) and echoed back as If-None-Match.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub etag: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ModelsStoreOperationOptions {
    pub signal: Option<CancellationToken>,
}

impl ModelsStoreOperationOptions {
    fn throw_if_aborted(&self) -> Result<()> {
        match &self.signal {
            Some(signal) => throw_if_aborted(signal),
            None => Ok(()),
        }
    }
}

/// Persistent model catalogs keyed by provider ID.
#[async_trait]
pub trait ModelsStore: Send + Sync {
    async fn read(
        &self,
        provider_id: &str,
        options: ModelsStoreOperationOptions,
    ) -> Result<Option<ModelsStoreEntry>>;
    async fn write(
        &self,
        provider_id: &str,
        entry: ModelsStoreEntry,
        options: ModelsStoreOperationOptions,
    ) -> Result<()>;
    async fn delete(&self, provider_id: &str, options: ModelsStoreOperationOptions) -> Result<()>;
}

#[derive(Default)]
pub struct InMemoryModelsStore {
    entries: Mutex<HashMap<String, ModelsStoreEntry>>,
}

impl InMemoryModelsStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ModelsStore for InMemoryModelsStore {
    async fn read(
        &self,
        provider_id: &str,
        options: ModelsStoreOperationOptions,
    ) -> Result<Option<ModelsStoreEntry>> {
        options.throw_if_aborted()?;
        Ok(self.entries.lock().get(provider_id).cloned())
    }

    async fn write(
        &self,
        provider_id: &str,
        entry: ModelsStoreEntry,
        options: ModelsStoreOperationOptions,
    ) -> Result<()> {
        options.throw_if_aborted()?;
        self.entries.lock().insert(provider_id.to_string(), entry);
        Ok(())
    }

    async fn delete(&self, provider_id: &str, options: ModelsStoreOperationOptions) -> Result<()> {
        options.throw_if_aborted()?;
        self.entries.lock().remove(provider_id);
        Ok(())
    }
}
