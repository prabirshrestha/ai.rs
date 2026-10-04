//! Port of durable `src/testing/` (`@earendil-works/pi-durable/testing`): the
//! runner-independent storage conformance suite that every [`Storage`]
//! backend must pass.
//!
//! Available in-crate under `cfg(test)` and to other crates with the
//! `durable-testing` feature.
//!
//! Divergences from Pi: the assertion facade (`StorageConformanceAssertions`,
//! `createExpectAssertions`) and `registerStorageConformance` collapse into
//! Rust's panicking assertions plus [`storage_conformance_tests!`], which
//! registers one `#[tokio::test]` per case. [`create_storage_conformance`]
//! keeps the runner-independent case list. The benchmark helpers
//! (`storage-benchmark.ts`) are not ported.
//!
//! [`Storage`]: crate::durable::types::Storage
//! [`storage_conformance_tests!`]: crate::storage_conformance_tests

pub mod storage_conformance;

use std::sync::Arc;

use futures::future::BoxFuture;

use crate::durable::types::Storage;

pub use storage_conformance::create_storage_conformance;

/// One conformance body; the provider must call and await it exactly once.
pub type ConformanceTest = Box<dyn FnOnce(Arc<dyn Storage>) -> BoxFuture<'static, ()> + Send>;

/// `StorageConformanceProvider`: supplies a fresh storage to one test body and cleans up after it.
pub type StorageConformanceProvider =
    Arc<dyn Fn(ConformanceTest) -> BoxFuture<'static, ()> + Send + Sync>;

/// `StorageConformanceCase`.
pub struct StorageConformanceCase {
    pub name: &'static str,
    run: Box<dyn FnOnce() -> BoxFuture<'static, ()> + Send>,
}

impl StorageConformanceCase {
    pub(crate) fn new(
        name: &'static str,
        run: impl FnOnce() -> BoxFuture<'static, ()> + Send + 'static,
    ) -> Self {
        Self {
            name,
            run: Box::new(run),
        }
    }

    /// Run the case; assertion failures panic.
    pub fn run(self) -> BoxFuture<'static, ()> {
        (self.run)()
    }
}

/// Register one `#[tokio::test]` per storage conformance case
/// (`registerStorageConformance`). The argument is an expression of type
/// [`StorageConformanceProvider`]; it is evaluated once per test.
///
/// ```ignore
/// ai::storage_conformance_tests!(std::sync::Arc::new(|test| {
///     Box::pin(test(std::sync::Arc::new(ai::durable::MemoryStorage::new())))
/// }));
/// ```
#[macro_export]
macro_rules! storage_conformance_tests {
    ($provider:expr) => {
        $crate::__storage_conformance_tests!(
            $provider;
            reserves_id_1_for_the_immutable_root_conversation,
            commits_mixed_table_writes_atomically_and_rolls_all_of_them_back_on_failure,
            detaches_retained_writes_and_every_returned_record,
            detaches_prototype_like_json_keys_without_changing_object_prototypes,
            indexes_entries_committed_out_of_id_order,
            continues_an_entry_cursor_below_its_last_item_after_a_newer_commit,
            paginates_conversations_by_opaque_cursor_in_ascending_id_order,
            filters_and_pages_conversations_by_durable_owner_edges,
            scans_deep_fork_history_newest_first_through_every_ancestor_cap,
            replaces_complete_task_records_and_pages_filtered_task_scans,
            stores_owners_and_scans_waiting_and_completing_tasks_by_status,
            indexes_request_ids_per_conversation_and_replaces_complete_submission_records,
            stores_passive_write_submissions_without_input_only_lifecycle_states,
            reconstructs_rewindable_documents_and_preserves_half_open_incarnations,
            streams_long_document_tails_across_root_replacement_deltas,
            copies_stored_document_bases_independently_and_rejects_ambiguous_sources,
            uses_bases_for_version_transitions_and_rejects_historical_reads_of_current_only_documents,
            indexes_logical_addresses_and_exact_scope_scans_independently,
            keeps_document_lifecycle_failures_atomic_and_gives_create_plus_retire_an_empty_lifetime,
            rolls_back_record_tables_and_secondary_indexes_when_a_document_command_fails,
            keeps_indexed_string_identities_lossless,
            keeps_one_global_record_id_namespace_and_rejects_exhausted_id_minting,
            rejects_every_operation_after_close,
        );
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __storage_conformance_tests {
    ($provider:expr; $($case:ident),* $(,)?) => {
        $(
            #[tokio::test]
            async fn $case() {
                let provider: $crate::durable::testing::StorageConformanceProvider = $provider;
                provider(Box::new(|storage| {
                    Box::pin($crate::durable::testing::storage_conformance::$case(storage))
                }))
                .await;
            }
        )*
    };
}
