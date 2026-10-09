//! Port of durable `src/harness/usage.ts`.

use std::sync::LazyLock;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::durable::documents::{DocToken, define_doc};
use crate::durable::errors::Result;
use crate::durable::ids::ConversationId;
use crate::durable::session::Transaction;
use crate::durable::types::{DocDefinition, LatestConversation, LatestFork};
use crate::types::Usage;

/// Ledger of one conversation's own spend: its entries, and compaction summarization attempts, which have none.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct UsageState {
    /// Assistant entries and summarization attempts, keyed `provider/modelId`.
    pub models: IndexMap<String, Usage>,
    /// Tool results, keyed by tool name; their usage has no model identity.
    pub tools: IndexMap<String, Usage>,
}

/// `keyof UsageState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsageBucket {
    Models,
    Tools,
}

pub static USAGE_DOC: LazyLock<DocToken<UsageState, LatestConversation>> = LazyLock::new(|| {
    define_doc(
        DocDefinition::new(
            "pi.usage",
            1,
            LatestConversation {
                fork: LatestFork::Initial,
            },
            UsageState::default,
        )
        .checkpoint_when(|_, _, _| Ok(true)),
    )
    .expect("valid pi.usage definition")
});

/// Add `usage` to one bucket of the conversation's `pi.usage`, in the commit that records the response.
pub async fn record_usage(
    tx: &Transaction,
    conversation_id: ConversationId,
    bucket: UsageBucket,
    key: &str,
    usage: &Usage,
) -> Result<()> {
    let state = tx.doc(&*USAGE_DOC, conversation_id).await?;
    state.edit(|state| {
        let totals = match bucket {
            UsageBucket::Models => &mut state.models,
            UsageBucket::Tools => &mut state.tools,
        };
        match totals.get_mut(key) {
            None => {
                totals.insert(key.to_string(), usage.clone());
            }
            Some(total) => add_usage(total, usage),
        }
    })
}

/// Add every counter of `usage` to `total`; optional counters are added once either side reports them.
pub fn add_usage(total: &mut Usage, usage: &Usage) {
    total.input += usage.input;
    total.output += usage.output;
    total.cache_read += usage.cache_read;
    total.cache_write += usage.cache_write;
    total.total_tokens += usage.total_tokens;
    if let Some(value) = usage.cache_write_1h {
        total.cache_write_1h = Some(total.cache_write_1h.unwrap_or(0) + value);
    }
    if let Some(value) = usage.reasoning {
        total.reasoning = Some(total.reasoning.unwrap_or(0) + value);
    }
    total.cost.input += usage.cost.input;
    total.cost.output += usage.cost.output;
    total.cost.cache_read += usage.cost.cache_read;
    total.cost.cache_write += usage.cost.cache_write;
    total.cost.total += usage.cost.total;
}

/// Add every bucket of `state` into `sum`.
pub fn add_usage_state(sum: &mut UsageState, state: &UsageState) {
    for (sum, state) in [
        (&mut sum.models, &state.models),
        (&mut sum.tools, &state.tools),
    ] {
        for (key, usage) in state {
            match sum.get_mut(key) {
                Some(total) => add_usage(total, usage),
                None => {
                    sum.insert(key.clone(), usage.clone());
                }
            }
        }
    }
}
