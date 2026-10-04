//! Port of chord `src/delta/tracker.ts` (`track`, `Tracker`, `Change`, `Prepared`).
//!
//! Divergence from Pi (unavoidable): Rust has no `Proxy`, so a [`Change`] does
//! not record mutations through traps. Its draft is a private working copy of
//! the base revision ([`Change::state_mut`], or typed through [`Change::edit`]),
//! and [`Change::prepare`] emits ops with [`diff_revisions`] (chord's own
//! revision diff) instead of from dirty overlay nodes. The lifecycle is ported
//! as is: competing changes, staleness after another adoption, revocation on
//! settle, `adopt` as a pointer swap with the same checks and messages, no-op
//! normalization to the previous revision, and the 4,096-op fold to `r`.
//! Op *shapes* can differ from Pi's tracker for complex array edits (Pi:
//! "Operation shape is not canonical"); the resulting values are identical.
//! Values placed into a draft are owned copies by construction, so the
//! placement validation and handle-detachment rules of the proxy do not apply.

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use super::diff::{diff_revisions, equal_json};
use super::{DeltaError, Op, apply_immutable};

const OPEN: u8 = 0;
const PREPARED: u8 = 1;
const CONSUMED: u8 = 2;
const ABORTED: u8 = 3;
const STALE: u8 = 4;

/// The context `status` cell shared by a change and its preparation.
#[derive(Debug)]
struct StatusCell(AtomicU8);

impl StatusCell {
    fn get(&self) -> u8 {
        self.0.load(Ordering::SeqCst)
    }

    fn set(&self, status: u8) {
        self.0.store(status, Ordering::SeqCst);
    }

    fn is_settled(&self) -> bool {
        matches!(self.get(), CONSUMED | ABORTED | STALE)
    }
}

static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

/// One immutable revision sequence (`Tracker<T>`).
#[derive(Debug)]
pub struct Tracker {
    owner: u64,
    value: Arc<Value>,
    revision: u64,
    contexts: Vec<Weak<StatusCell>>,
    prune_budget: usize,
}

/// Take immutable ownership of an alias-free strict-JSON root in O(1).
pub fn track(initial: impl Into<Arc<Value>>) -> Tracker {
    Tracker {
        owner: NEXT_OWNER.fetch_add(1, Ordering::Relaxed),
        value: initial.into(),
        revision: 0,
        contexts: Vec::new(),
        prune_budget: 256,
    }
}

impl Tracker {
    /// `tracker.value`: the latest adopted revision.
    pub fn value(&self) -> &Arc<Value> {
        &self.value
    }

    /// `tracker.revision`.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    fn register(&mut self) -> Arc<StatusCell> {
        let status = Arc::new(StatusCell(AtomicU8::new(OPEN)));
        self.contexts.push(Arc::downgrade(&status));
        self.prune_budget -= 1;
        if self.prune_budget == 0 {
            self.contexts.retain(|context| context.strong_count() > 0);
            self.prune_budget = self.contexts.len().max(256);
        }
        status
    }

    /// Open a draft over the current revision (`beginChange()`).
    pub fn begin_change(&mut self) -> Change {
        let status = self.register();
        Change {
            owner: self.owner,
            base_revision: self.revision,
            base: self.value.clone(),
            draft: Some((*self.value).clone()),
            status: Some(status),
            prepared_status: None,
            settled: false,
        }
    }

    /// A whole-root replacement, not a diff (`prepareReplace(value)`). A deeply
    /// equal value keeps the current root with empty ops.
    pub fn prepare_replace(&mut self, value: impl Into<Arc<Value>>) -> Prepared {
        let value = value.into();
        let status = self.register();
        status.set(PREPARED);
        let base = self.value.clone();
        let noop = equal_json(&base, &value);
        let (value, ops): (Arc<Value>, Arc<[Op]>) = if noop {
            (base.clone(), Arc::from(Vec::new()))
        } else {
            let ops = Arc::from(vec![Op::R((*value).clone())]);
            (value, ops)
        };
        Prepared {
            owner: self.owner,
            base_revision: self.revision,
            base,
            value,
            ops,
            status,
        }
    }

    /// Check the preparation and swap the root pointer (`adopt(prepared)`).
    pub fn adopt(&mut self, prepared: &Prepared) -> Result<(), DeltaError> {
        if prepared.owner != self.owner {
            return Err(DeltaError::error(
                "Prepared change belongs to a different tracker",
            ));
        }
        match prepared.status.get() {
            CONSUMED => return Err(DeltaError::error("Prepared change has already been used")),
            ABORTED => return Err(DeltaError::error("Prepared change has been aborted")),
            STALE => return Err(DeltaError::error("Prepared change is stale")),
            PREPARED => {}
            _ => return Err(DeltaError::error("Prepared change is not ready")),
        }
        if prepared.base_revision != self.revision || !Arc::ptr_eq(&self.value, &prepared.base) {
            prepared.status.set(STALE);
            return Err(DeltaError::error("Prepared change is stale"));
        }
        // Materialization happened during prepare. Adoption is an infallible
        // pointer swap so storage failure can discard the candidate.
        self.value = prepared.value.clone();
        prepared.status.set(CONSUMED);
        self.revision += 1;
        self.invalidate(&prepared.status);
        Ok(())
    }

    fn invalidate(&mut self, winner: &Arc<StatusCell>) {
        for context in self.contexts.drain(..) {
            let Some(context) = context.upgrade() else {
                continue;
            };
            if Arc::ptr_eq(&context, winner) {
                continue;
            }
            if matches!(context.get(), OPEN | PREPARED) {
                context.set(STALE);
            }
        }
        self.prune_budget = 256;
    }
}

/// An open draft over one revision (`Change<T>`).
#[derive(Debug)]
pub struct Change {
    owner: u64,
    base_revision: u64,
    base: Arc<Value>,
    draft: Option<Value>,
    status: Option<Arc<StatusCell>>,
    prepared_status: Option<Arc<StatusCell>>,
    settled: bool,
}

fn settled_overlay() -> DeltaError {
    DeltaError::type_error("Cannot use a settled overlay")
}

impl Change {
    fn readable(&self) -> Result<(), DeltaError> {
        match &self.status {
            Some(status) if !status.is_settled() && self.draft.is_some() => Ok(()),
            _ => Err(settled_overlay()),
        }
    }

    /// `change.state` for reading. Fails once the change is settled or stale.
    pub fn state(&self) -> Result<&Value, DeltaError> {
        self.readable()?;
        Ok(self.draft.as_ref().expect("readable draft"))
    }

    /// `change.state` for writing. Fails once the change is settled or stale.
    pub fn state_mut(&mut self) -> Result<&mut Value, DeltaError> {
        self.readable()?;
        Ok(self.draft.as_mut().expect("readable draft"))
    }

    /// Typed draft access: decode the draft as `T`, mutate it, and store it back.
    pub fn edit<T, R>(&mut self, mutate: impl FnOnce(&mut T) -> R) -> Result<R, DeltaError>
    where
        T: Serialize + DeserializeOwned,
    {
        let draft = self.state_mut()?;
        let mut typed: T = serde_json::from_value(draft.clone()).map_err(|error| {
            DeltaError::type_error(format!("Draft does not match its type: {error}"))
        })?;
        let result = mutate(&mut typed);
        *draft = serde_json::to_value(&typed).map_err(|error| {
            DeltaError::type_error(format!("Draft is not strict JSON: {error}"))
        })?;
        Ok(result)
    }

    /// Materialize the candidate revision and its ops (`prepare()`). The draft
    /// is unusable afterwards.
    pub fn prepare(&mut self) -> Result<Prepared, DeltaError> {
        if self.settled {
            return Err(DeltaError::error("Change has already been settled"));
        }
        let status = self.status.clone().expect("unsettled change has a status");
        if status.is_settled() {
            return Err(settled_overlay());
        }
        if status.get() != OPEN {
            return Err(DeltaError::type_error("Prepared overlays are read-only"));
        }
        status.set(PREPARED);
        let draft = self.draft.take().expect("open change has a draft");
        self.status = None;
        self.settled = true;
        let ops = diff_revisions(&self.base, &draft);
        let value = match ops.as_slice() {
            [] => Ok(self.base.clone()),
            [Op::R(_)] => Ok(Arc::new(draft)),
            // Replicas compute the next revision from the ops, so materialize it the same way.
            _ => apply_immutable(&self.base, &ops).map(Arc::new),
        };
        let value = match value {
            Ok(value) => value,
            Err(error) => {
                status.set(ABORTED);
                return Err(error);
            }
        };
        self.prepared_status = Some(status.clone());
        Ok(Prepared {
            owner: self.owner,
            base_revision: self.base_revision,
            base: self.base.clone(),
            value,
            ops: Arc::from(ops),
            status,
        })
    }

    /// Discard the change (`abort()`). After `prepare()`, it aborts the prepared
    /// result instead. Idempotent.
    pub fn abort(&mut self) {
        if self.settled {
            if let Some(status) = self.prepared_status.take()
                && status.get() == PREPARED
            {
                status.set(ABORTED);
            }
            return;
        }
        self.settled = true;
        if let Some(status) = self.status.take()
            && !status.is_settled()
        {
            status.set(ABORTED);
        }
        self.draft = None;
    }
}

/// A materialized candidate revision (`Prepared<T>`). `base`, `value` and `ops`
/// are immutable and stay readable after abort or staleness.
#[derive(Debug, Clone)]
pub struct Prepared {
    owner: u64,
    base_revision: u64,
    base: Arc<Value>,
    value: Arc<Value>,
    ops: Arc<[Op]>,
    status: Arc<StatusCell>,
}

impl Prepared {
    pub fn base(&self) -> &Arc<Value> {
        &self.base
    }

    pub fn value(&self) -> &Arc<Value> {
        &self.value
    }

    pub fn ops(&self) -> &Arc<[Op]> {
        &self.ops
    }

    pub fn base_revision(&self) -> u64 {
        self.base_revision
    }

    /// Prevent adoption (`abort()`). Idempotent.
    pub fn abort(&self) {
        if !self.status.is_settled() {
            self.status.set(ABORTED);
        }
    }
}
