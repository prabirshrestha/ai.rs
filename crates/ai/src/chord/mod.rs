//! Port of the subset of `@earendil-works/chord` 1.0.2 (`packages/chord/src`)
//! that durable uses: `context/`, `json.ts`, `delta/` and the replicated-state
//! source/attachment parts of `services/state.ts`.
//!
//! Not ported: facets, services, the RPC wire, bundles and the node loader
//! (durable does not use them; `chord-guide.test.ts` is the only consumer).
//! Each module documents its divergences from Pi.

pub mod context;
pub mod delta;
pub mod json;
pub mod state;

pub use context::{
    AbortController, AbortReason, AbortSignal, BACKGROUND_CONTEXT, CancelContext, Context,
    ContextKey, TODO_CONTEXT, await_with_context, background_context, create_context_key,
    todo_context, with_abort_signal, with_cancel, with_context_value, without_abort_signal,
};
pub use json::{CopyJsonOptions, JsonError, JsonValue, copy_json, is_json_value};
pub use state::{
    AttachedReplicatedState, DeliveryKind, ListenerOutcome, MutableReplicatedState,
    ReplicatedState, ReplicatedStateDelivery, ReplicatedStateSnapshot, ReplicatedStateSource,
    ReplicatedStateSourceAttachment, ReplicatedStateSourceFrame, ReplicatedStateSourceOptions,
    StateError, Unsubscribe, mutable_replicated_state, replicated_state,
};
