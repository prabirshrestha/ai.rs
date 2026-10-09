//! Port of durable `src/tools/path-utils.ts`.

use std::sync::LazyLock;

use indexmap::IndexSet;
use regex::Regex;
use unicode_normalization::UnicodeNormalization;

use crate::chord::Context;
use crate::durable::env::{ExecutionEnv, get_or_throw};
use crate::durable::errors::Result;

const NARROW_NO_BREAK_SPACE: &str = "\u{202F}";

static AM_PM: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i) (AM|PM)\.").unwrap());

fn is_unicode_space(c: char) -> bool {
    matches!(
        c,
        '\u{00A0}' | '\u{2000}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}'
    )
}

fn normalize_tool_path(path: &str) -> String {
    let normalized: String = path
        .chars()
        .map(|c| if is_unicode_space(c) { ' ' } else { c })
        .collect();
    match normalized.strip_prefix('@') {
        Some(rest) => rest.to_string(),
        None => normalized,
    }
}

pub async fn resolve_tool_path(
    env: &dyn ExecutionEnv,
    path: &str,
    context: &Context,
) -> Result<String> {
    get_or_throw(env.absolute_path(&normalize_tool_path(path), context).await)
}

pub async fn resolve_read_tool_path(
    env: &dyn ExecutionEnv,
    path: &str,
    context: &Context,
) -> Result<String> {
    let resolved = resolve_tool_path(env, path, context).await?;
    let nfd: String = resolved.nfd().collect();
    let variants: IndexSet<String> = [
        resolved.clone(),
        AM_PM
            .replace_all(&resolved, format!("{NARROW_NO_BREAK_SPACE}$1."))
            .into_owned(),
        nfd.clone(),
        resolved.replace('\'', "\u{2019}"),
        nfd.replace('\'', "\u{2019}"),
    ]
    .into_iter()
    .collect();

    for variant in variants {
        if get_or_throw(env.exists(&variant, context).await)? {
            return Ok(variant);
        }
    }
    Ok(resolved)
}
