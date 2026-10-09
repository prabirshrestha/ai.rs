//! Port of durable `src/harness/provider.ts`.

use std::sync::{Arc, LazyLock};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::chord::Context;
use crate::durable::documents::{DocToken, define_doc};
use crate::durable::errors::{Error, Result};
use crate::durable::types::{DocDefinition, LatestConversation, LatestFork};
use crate::utils::uuid::uuidv7;

use super::scheduler::RuntimeCore;

/// Stable provider-facing identity of one conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderState {
    pub session_id: String,
}

/// Built-in provider state; every fork starts with a fresh identity instead of copying its parent.
pub static PROVIDER_DOC: LazyLock<DocToken<ProviderState, LatestConversation>> =
    LazyLock::new(|| {
        define_doc(
            DocDefinition::new(
                "pi.provider",
                1,
                LatestConversation {
                    fork: LatestFork::Initial,
                },
                || ProviderState {
                    session_id: uuidv7(None).expect("current time is in range"),
                },
            )
            .checkpoint_when(|_, _, _| Ok(true)),
        )
        .expect("valid pi.provider definition")
    });

/// Return the persisted identity without writing in the normal path. A legacy conversation without `pi.provider` gets
/// one migration commit whose `tx.doc()` runs `initial()` before the provider request starts.
pub async fn ensure_provider_session_id(
    runtime: &Arc<RuntimeCore>,
    context: &Context,
) -> Result<String> {
    let conversation_id = runtime.conversation_id();
    if let Some(existing) = runtime
        .reader()
        .snapshot(&*PROVIDER_DOC, conversation_id, context)
        .await?
    {
        return Ok(existing.session_id);
    }
    let created: Arc<Mutex<Option<String>>> = Arc::default();
    let slot = created.clone();
    runtime
        .commit(
            move |tx, _| async move {
                let state = tx.doc(&*PROVIDER_DOC, conversation_id).await?;
                *slot.lock() = Some(state.get()?.session_id);
                Ok(None)
            },
            context,
        )
        .await?;
    let created = created.lock().take();
    created.ok_or_else(|| {
        Error::message(format!(
            "Conversation {conversation_id} has no provider session ID"
        ))
    })
}
