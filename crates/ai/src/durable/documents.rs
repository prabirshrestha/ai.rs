//! Port of durable `src/documents.ts`: document definitions, typed tokens,
//! address resolution, and stored-value checks.
//!
//! Divergences from Pi:
//! - TS `defineDoc` overloads on the literal `scope`/`history`/`fork` fields and
//!   `tx.doc(token, ...args)` is variadic. Here the token carries a scope marker
//!   ([`SessionScope`], [`LatestConversation`], [`RewindableConversation`],
//!   [`TaskScope`]) and every access takes one typed argument value selected by
//!   [`DocAccess`]: `()` / an owner ID / `(owner, key)` / `(owner, key, seed)`.
//!   The argument checks TS performs at compile time stay compile-time checks.
//! - Values are `serde` types instead of `JsonObject` subtypes. The erased
//!   [`AnyDocDefinition`] serializes initial and migrated values with
//!   `copy_json` (the strict-JSON check), and a value that is not a JSON object
//!   is rejected with a `TypeError`.
//! - `version` is a `u32`, so the non-integer case of the positive-integer check
//!   is unrepresentable; zero is still rejected.

use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::chord::delta::Op;
use crate::chord::{JsonValue, copy_json};

use super::errors::{Error, Result};
use super::ids::{ConversationId, DocumentId, TaskId};
use super::types::{
    CheckpointInfo, CheckpointWhen, DocDefinition, DocFamilyDefinition, DocScope, DocumentAddress,
    DocumentCreate, DocumentRecord, DocumentScope, DocumentSemantics, JsonObject,
    LatestConversation, RewindableConversation, ScopeKind, SessionScope, StoredDocument, TaskScope,
};

type ErasedInitial = Arc<dyn Fn(Option<&JsonValue>) -> Result<JsonObject> + Send + Sync>;
type ErasedMigrate = Arc<dyn Fn(JsonObject, u32) -> Result<JsonObject> + Send + Sync>;

/// Erased definition shape used by the Session after overload resolution.
pub struct AnyDocDefinition {
    pub kind: String,
    pub version: u32,
    pub semantics: DocumentSemantics,
    pub family: bool,
    initial: ErasedInitial,
    migrate: Option<ErasedMigrate>,
    checkpoint_when: Option<CheckpointWhen>,
}

impl fmt::Debug for AnyDocDefinition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnyDocDefinition")
            .field("kind", &self.kind)
            .field("version", &self.version)
            .field("semantics", &self.semantics)
            .field("family", &self.family)
            .finish_non_exhaustive()
    }
}

impl AnyDocDefinition {
    pub fn scope(&self) -> ScopeKind {
        self.semantics.scope()
    }

    /// `initial()` for a singleton, `initial(seed)` for a family member, detached as strict JSON.
    pub fn initial(&self, seed: Option<&JsonValue>) -> Result<JsonObject> {
        (self.initial)(seed)
    }

    pub fn has_migrate(&self) -> bool {
        self.migrate.is_some()
    }

    /// `copyJson(definition.migrate(value, fromVersion))`.
    pub fn migrate(&self, value: JsonObject, from_version: u32) -> Result<JsonObject> {
        match &self.migrate {
            Some(migrate) => migrate(value, from_version),
            None => Err(Error::type_error(format!(
                "Document {} has no migration",
                self.kind
            ))),
        }
    }

    pub fn has_checkpoint_when(&self) -> bool {
        self.checkpoint_when.is_some()
    }

    /// `checkpointWhen?.(value, ops, info)`; `false` when absent.
    pub fn checkpoint_when(
        &self,
        value: &JsonValue,
        ops: &[Op],
        info: CheckpointInfo,
    ) -> Result<bool> {
        match &self.checkpoint_when {
            Some(predicate) => predicate(value, ops, info),
            None => Ok(false),
        }
    }
}

/// Serialize a document value as a strict-JSON object.
pub(crate) fn to_json_object<T: Serialize + ?Sized>(kind: &str, value: &T) -> Result<JsonObject> {
    match copy_json(value, None)? {
        JsonValue::Object(object) => Ok(object),
        _ => Err(Error::type_error(format!(
            "Document {kind} value must be a JSON object"
        ))),
    }
}

/// Decode a stored JSON value as a document's typed value.
pub(crate) fn from_json<T: DeserializeOwned>(kind: &str, value: &JsonValue) -> Result<T> {
    T::deserialize(value).map_err(|error| {
        Error::type_error(format!(
            "Document {kind} value does not match its type: {error}"
        ))
    })
}

/// Erased singleton or family token (`AnyDocToken`).
pub type AnyDocToken = Arc<AnyDocDefinition>;

/// Typed singleton document token passed explicitly to typed access (`DocToken<T, D>`).
pub struct DocToken<T, S: DocScope> {
    definition: Arc<AnyDocDefinition>,
    scope: S,
    _value: PhantomData<fn() -> T>,
}

/// Typed document family token passed explicitly to typed access (`DocFamilyToken<T, I, D>`).
pub struct DocFamilyToken<T, I, S: DocScope> {
    definition: Arc<AnyDocDefinition>,
    scope: S,
    _value: PhantomData<fn(I) -> T>,
}

impl<T, S: DocScope> Clone for DocToken<T, S> {
    fn clone(&self) -> Self {
        Self {
            definition: self.definition.clone(),
            scope: self.scope,
            _value: PhantomData,
        }
    }
}

impl<T, I, S: DocScope> Clone for DocFamilyToken<T, I, S> {
    fn clone(&self) -> Self {
        Self {
            definition: self.definition.clone(),
            scope: self.scope,
            _value: PhantomData,
        }
    }
}

impl<T, S: DocScope> fmt::Debug for DocToken<T, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DocToken").field(&self.definition).finish()
    }
}

impl<T, I, S: DocScope> fmt::Debug for DocFamilyToken<T, I, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DocFamilyToken")
            .field(&self.definition)
            .finish()
    }
}

impl<T, S: DocScope> DocToken<T, S> {
    /// `token.definition`.
    pub fn definition(&self) -> &Arc<AnyDocDefinition> {
        &self.definition
    }

    pub fn scope(&self) -> S {
        self.scope
    }
}

impl<T, I, S: DocScope> DocFamilyToken<T, I, S> {
    /// `token.definition`.
    pub fn definition(&self) -> &Arc<AnyDocDefinition> {
        &self.definition
    }

    pub fn scope(&self) -> S {
        self.scope
    }
}

fn validate_definition(kind: &str, version: u32) -> Result<()> {
    if version < 1 {
        return Err(Error::type_error(format!(
            "Document {kind} version must be a positive integer"
        )));
    }
    Ok(())
}

fn erase_migrate<T: Serialize + 'static>(
    kind: &str,
    migrate: Option<super::types::Migrate<T>>,
) -> Option<ErasedMigrate> {
    migrate.map(|migrate| {
        let kind = kind.to_string();
        Arc::new(move |value: JsonObject, from_version: u32| {
            let migrated = migrate(value, from_version)?;
            to_json_object(&kind, &migrated)
        }) as ErasedMigrate
    })
}

/// Define a singleton document: Session-scoped, latest or rewindable conversation, or task-scoped.
pub fn define_doc<T, S>(definition: DocDefinition<T, S>) -> Result<DocToken<T, S>>
where
    T: Serialize + DeserializeOwned + 'static,
    S: DocScope,
{
    validate_definition(&definition.kind, definition.version)?;
    let semantics = definition.scope.semantics();
    let kind = definition.kind.clone();
    let initial = definition.initial.clone();
    let initial_kind = kind.clone();
    let erased = AnyDocDefinition {
        kind: kind.clone(),
        version: definition.version,
        semantics,
        family: false,
        initial: Arc::new(move |_seed| to_json_object(&initial_kind, &initial())),
        migrate: erase_migrate(&kind, definition.migrate),
        checkpoint_when: definition.checkpoint_when,
    };
    Ok(DocToken {
        definition: Arc::new(erased),
        scope: definition.scope,
        _value: PhantomData,
    })
}

/// Define a document family: Session-scoped, latest or rewindable conversation, or task-scoped.
pub fn define_doc_family<T, I, S>(
    definition: DocFamilyDefinition<T, I, S>,
) -> Result<DocFamilyToken<T, I, S>>
where
    T: Serialize + DeserializeOwned + 'static,
    I: Serialize + DeserializeOwned + 'static,
    S: DocScope,
{
    validate_definition(&definition.kind, definition.version)?;
    let semantics = definition.scope.semantics();
    let kind = definition.kind.clone();
    let initial = definition.initial.clone();
    let initial_kind = kind.clone();
    let erased = AnyDocDefinition {
        kind: kind.clone(),
        version: definition.version,
        semantics,
        family: true,
        initial: Arc::new(move |seed| {
            let seed: I = from_json(&initial_kind, seed.unwrap_or(&JsonValue::Null))?;
            to_json_object(&initial_kind, &initial(seed))
        }),
        migrate: erase_migrate(&kind, definition.migrate),
        checkpoint_when: definition.checkpoint_when,
    };
    Ok(DocFamilyToken {
        definition: Arc::new(erased),
        scope: definition.scope,
        _value: PhantomData,
    })
}

/// The erased owner and family key of one access (the TS argument list).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DocArgs {
    pub owner: Option<u64>,
    pub key: Option<String>,
}

/// Typed access to a document token: which arguments select an address (`Address`,
/// used by snapshots, states, watches and retirement) and which acquire it
/// (`Args`, used by `Tx.doc()`, which adds the family creation seed).
pub trait DocAccess: Send + Sync {
    type Value: Serialize + DeserializeOwned + Send + 'static;
    type Address: Send + 'static;
    type Args: Send + 'static;
    fn definition(&self) -> &Arc<AnyDocDefinition>;
    fn address_args(&self, address: Self::Address) -> DocArgs;
    /// The address arguments plus the detached creation seed of a family member.
    fn acquire_args(&self, args: Self::Args) -> Result<(DocArgs, Option<JsonValue>)>;
}

/// Tokens whose documents keep history for `Session.snapshotAsOf()`.
pub trait RewindableDocAccess: DocAccess {}

impl<T, S> DocAccess for DocToken<T, S>
where
    T: Serialize + DeserializeOwned + Send + 'static,
    S: DocScope,
{
    type Value = T;
    type Address = S::Owner;
    type Args = S::Owner;

    fn definition(&self) -> &Arc<AnyDocDefinition> {
        &self.definition
    }

    fn address_args(&self, owner: S::Owner) -> DocArgs {
        DocArgs {
            owner: S::owner_id(owner),
            key: None,
        }
    }

    fn acquire_args(&self, owner: S::Owner) -> Result<(DocArgs, Option<JsonValue>)> {
        Ok((self.address_args(owner), None))
    }
}

impl<T> RewindableDocAccess for DocToken<T, RewindableConversation> where
    T: Serialize + DeserializeOwned + Send + 'static
{
}

fn seed_json<I: Serialize>(seed: &I) -> Result<Option<JsonValue>> {
    Ok(Some(copy_json(seed, None)?))
}

impl<T, I> DocAccess for DocFamilyToken<T, I, SessionScope>
where
    T: Serialize + DeserializeOwned + Send + 'static,
    I: Serialize + Send + 'static,
{
    type Value = T;
    type Address = String;
    type Args = (String, I);

    fn definition(&self) -> &Arc<AnyDocDefinition> {
        &self.definition
    }

    fn address_args(&self, key: String) -> DocArgs {
        DocArgs {
            owner: None,
            key: Some(key),
        }
    }

    fn acquire_args(&self, (key, seed): (String, I)) -> Result<(DocArgs, Option<JsonValue>)> {
        Ok((self.address_args(key), seed_json(&seed)?))
    }
}

macro_rules! owned_family_access {
    ($scope:ty, $owner:ty) => {
        impl<T, I> DocAccess for DocFamilyToken<T, I, $scope>
        where
            T: Serialize + DeserializeOwned + Send + 'static,
            I: Serialize + Send + 'static,
        {
            type Value = T;
            type Address = ($owner, String);
            type Args = ($owner, String, I);

            fn definition(&self) -> &Arc<AnyDocDefinition> {
                &self.definition
            }

            fn address_args(&self, (owner, key): ($owner, String)) -> DocArgs {
                DocArgs {
                    owner: <$scope as DocScope>::owner_id(owner),
                    key: Some(key),
                }
            }

            fn acquire_args(
                &self,
                (owner, key, seed): ($owner, String, I),
            ) -> Result<(DocArgs, Option<JsonValue>)> {
                Ok((self.address_args((owner, key)), seed_json(&seed)?))
            }
        }
    };
}

owned_family_access!(TaskScope, TaskId);
owned_family_access!(LatestConversation, ConversationId);
owned_family_access!(RewindableConversation, ConversationId);

impl<T, I> RewindableDocAccess for DocFamilyToken<T, I, RewindableConversation>
where
    T: Serialize + DeserializeOwned + Send + 'static,
    I: Serialize + Send + 'static,
{
}

/// Logical address plus its string identity for maps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedAddress {
    pub address: DocumentAddress,
    pub id: String,
}

/// Resolve an argument list into the logical address and its identity.
pub fn resolve_address(definition: &AnyDocDefinition, args: &DocArgs) -> Result<ResolvedAddress> {
    let owner_id = || {
        args.owner.ok_or_else(|| {
            Error::type_error(format!(
                "Document {} requires a {} ID",
                definition.kind,
                definition.scope()
            ))
        })
    };
    let scope = match definition.scope() {
        ScopeKind::Session => DocumentScope::Session,
        ScopeKind::Conversation => DocumentScope::Conversation {
            conversation_id: ConversationId(owner_id()?),
        },
        ScopeKind::Task => DocumentScope::Task {
            task_id: TaskId::new(owner_id()?),
        },
    };
    let key = if definition.family {
        Some(args.key.clone().ok_or_else(|| {
            Error::type_error(format!(
                "Document {} requires a family key",
                definition.kind
            ))
        })?)
    } else {
        None
    };
    let address = DocumentAddress {
        kind: definition.kind.clone(),
        scope,
        key,
    };
    let id = address_id(&address);
    Ok(ResolvedAddress { address, id })
}

/// Stable string identity of one logical address.
pub fn address_id(address: &DocumentAddress) -> String {
    let owner = match address.scope {
        DocumentScope::Session => JsonValue::Null,
        DocumentScope::Conversation { conversation_id } => JsonValue::from(conversation_id.0),
        DocumentScope::Task { task_id } => JsonValue::from(task_id.0),
    };
    let key = address
        .key
        .as_ref()
        .map_or(JsonValue::Null, |key| JsonValue::String(key.clone()));
    serde_json::to_string(&serde_json::json!([
        address.kind,
        address.scope.kind().as_str(),
        owner,
        key
    ]))
    .expect("address identities serialize")
}

/// Build the storage create record for a new incarnation at an address.
pub fn document_create(
    definition: &AnyDocDefinition,
    address: &DocumentAddress,
    id: DocumentId,
) -> DocumentCreate {
    DocumentCreate {
        id,
        kind: address.kind.clone(),
        key: address.key.clone(),
        scope: address.scope,
        history: match address.scope {
            DocumentScope::Conversation { .. } => definition.semantics.history(),
            _ => None,
        },
        fork: match address.scope {
            DocumentScope::Conversation { .. } => definition.semantics.fork(),
            _ => None,
        },
    }
}

/// The parts of a `DocumentCreate | DocumentRecord` the checks read.
pub trait RecordIdentity {
    fn record_id(&self) -> DocumentId;
    fn record_kind(&self) -> &str;
    fn record_scope(&self) -> &DocumentScope;
    fn record_history(&self) -> Option<super::types::History>;
    fn record_fork(&self) -> Option<super::types::ForkPolicy>;
}

macro_rules! record_identity {
    ($type:ty) => {
        impl RecordIdentity for $type {
            fn record_id(&self) -> DocumentId {
                self.id
            }
            fn record_kind(&self) -> &str {
                &self.kind
            }
            fn record_scope(&self) -> &DocumentScope {
                &self.scope
            }
            fn record_history(&self) -> Option<super::types::History> {
                self.history
            }
            fn record_fork(&self) -> Option<super::types::ForkPolicy> {
                self.fork
            }
        }
    };
}

record_identity!(DocumentCreate);
record_identity!(DocumentRecord);

/// Reject typed access whose token disagrees with the persisted scope, history, or fork semantics.
pub fn check_record_scope(
    definition: &AnyDocDefinition,
    record: &impl RecordIdentity,
) -> Result<()> {
    let scope = record.record_scope().kind();
    if scope != definition.scope()
        || (scope == ScopeKind::Conversation
            && (record.record_history() != definition.semantics.history()
                || record.record_fork() != definition.semantics.fork()))
    {
        return Err(Error::type_error(format!(
            "Document {} ({}) does not match the supplied definition semantics",
            record.record_id(),
            record.record_kind()
        )));
    }
    Ok(())
}

/// Reject typed access to a stored version the supplied definition cannot use.
pub fn check_record_version(
    definition: &AnyDocDefinition,
    record: &impl RecordIdentity,
    version: u32,
) -> Result<()> {
    if version > definition.version {
        return Err(Error::message(format!(
            "Document {} ({}) has newer version {} than {}",
            record.record_id(),
            record.record_kind(),
            version,
            definition.version
        )));
    }
    if version < definition.version && !definition.has_migrate() {
        return Err(Error::message(format!(
            "Document {} ({}) requires migration from version {}",
            record.record_id(),
            record.record_kind(),
            version
        )));
    }
    Ok(())
}

/// Validate and materialize a detached stored value for typed access.
pub fn materialize_document(
    definition: &AnyDocDefinition,
    stored: StoredDocument,
) -> Result<JsonObject> {
    materialize_document_value(definition, &stored.record, stored.version, stored.value)
}

/// Validate and materialize one detached value before its first persisted incarnation.
pub fn materialize_document_value(
    definition: &AnyDocDefinition,
    record: &impl RecordIdentity,
    version: u32,
    value: JsonObject,
) -> Result<JsonObject> {
    check_record_scope(definition, record)?;
    check_record_version(definition, record, version)?;
    if version == definition.version {
        return Ok(value);
    }
    definition.migrate(value, version)
}

#[cfg(test)]
mod tests {
    //! Port of `test/session-definitions.test.ts`. "types every owner, key,
    //! and seed overload" and "types historical reads, states, watches, and
    //! typed entries" are `expectTypeOf`/`@ts-expect-error` checks: each Rust
    //! token takes one typed address (`()` for Session documents, an ID, a
    //! key or a seed tuple), so the rejected overloads do not compile, and
    //! the accepted ones run in the session documents, states, watches and
    //! entries suites.

    use super::*;
    use crate::durable::types::{LatestFork, RewindableFork};
    use serde::Deserialize;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct State {
        value: i64,
    }

    #[test]
    fn validates_persisted_version_semantics() {
        let error = define_doc(DocDefinition::new("k", 0, SessionScope, || State {
            value: 0,
        }))
        .unwrap_err();
        assert!(error.to_string().contains("positive integer"));
        let empty = define_doc(DocDefinition::new("", 1, SessionScope, || State {
            value: 0,
        }))
        .unwrap();
        assert_eq!(empty.definition().kind, "");
    }

    #[test]
    fn resolves_addresses_with_stable_identities() {
        let latest = define_doc(DocDefinition::new(
            "t.latest",
            1,
            LatestConversation {
                fork: LatestFork::Current,
            },
            || State { value: 0 },
        ))
        .unwrap();
        let resolved =
            resolve_address(latest.definition(), &latest.address_args(ConversationId(7))).unwrap();
        assert_eq!(resolved.id, r#"["t.latest","conversation",7,null]"#);
        let family = define_doc_family(DocFamilyDefinition::new(
            "t.family",
            1,
            RewindableConversation {
                fork: RewindableFork::AsOf,
            },
            |seed: i64| State { value: seed },
        ))
        .unwrap();
        let (args, seed) = family
            .acquire_args((ConversationId(3), "k".to_string(), 5))
            .unwrap();
        assert_eq!(seed, Some(JsonValue::from(5)));
        let resolved = resolve_address(family.definition(), &args).unwrap();
        assert_eq!(resolved.id, r#"["t.family","conversation",3,"k"]"#);
        let create = document_create(family.definition(), &resolved.address, DocumentId(9));
        assert_eq!(
            serde_json::to_value(&create).unwrap(),
            serde_json::json!({
                "id": 9,
                "kind": "t.family",
                "key": "k",
                "scope": { "kind": "conversation", "conversationId": 3 },
                "history": "rewindable",
                "fork": "asOf"
            })
        );
        assert_eq!(
            family.definition().initial(seed.as_ref()).unwrap(),
            serde_json::json!({ "value": 5 })
                .as_object()
                .unwrap()
                .clone()
        );
    }
}
