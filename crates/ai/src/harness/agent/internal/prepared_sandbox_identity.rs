//! Port of `agent/internal/prepared-sandbox-identity.ts`.

use indexmap::IndexMap;

const PREPARED_SANDBOX_IDENTITY_VERSION: u32 = 1;

/// `resolvePreparedSandboxIdentity`.
///
/// [rust] Entries sort by code point; TS uses `localeCompare`, which agrees
/// for the ASCII harness ids in use.
pub fn resolve_prepared_sandbox_identity(
    recipe_identities: &IndexMap<String, String>,
    bootstrap_hash: Option<&str>,
    work_dir: Option<&str>,
) -> Option<String> {
    let mut entries: Vec<(&String, &String)> = recipe_identities.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    if entries.is_empty() && bootstrap_hash.is_none() {
        return None;
    }
    let version = PREPARED_SANDBOX_IDENTITY_VERSION.to_string();
    let mut values: Vec<&str> = vec![
        &version,
        work_dir.unwrap_or(""),
        bootstrap_hash.unwrap_or(""),
    ];
    for (harness_id, identity) in entries {
        values.push(harness_id);
        values.push(identity);
    }
    Some(super::hash_strings(&values, 8))
}
