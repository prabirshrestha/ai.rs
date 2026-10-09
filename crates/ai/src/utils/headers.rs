//! Port of `utils/headers.ts`, plus the reqwest plumbing that applies
//! [`ProviderHeaders`] to a request.

use indexmap::IndexMap;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

use crate::types::ProviderHeaders;
use crate::{Error, Result};

/// `headersToRecord()`: response headers as an ordered record. Values that
/// are not valid UTF-8 are skipped; repeated headers are joined with `", "`
/// like the Fetch `Headers` iterator.
pub fn headers_to_record(headers: &HeaderMap) -> IndexMap<String, String> {
    let mut result: IndexMap<String, String> = IndexMap::new();
    for (name, value) in headers {
        let Ok(value) = value.to_str() else {
            continue;
        };
        result
            .entry(name.as_str().to_string())
            .and_modify(|existing| {
                existing.push_str(", ");
                existing.push_str(value);
            })
            .or_insert_with(|| value.to_string());
    }
    result
}

/// `providerHeadersToRecord()`: merge header sources in order, matching names
/// case-insensitively. A later source replaces an earlier value (keeping the
/// later spelling and moving it to the end); `None` removes it. Returns
/// `None` when nothing remains.
pub fn provider_headers_to_record(
    header_sources: &[Option<&ProviderHeaders>],
) -> Option<IndexMap<String, String>> {
    let mut merged: IndexMap<String, (String, String)> = IndexMap::new();
    for source in header_sources.iter().flatten() {
        for (name, value) in source.iter() {
            let normalized_name = name.to_lowercase();
            merged.shift_remove(&normalized_name);
            if let Some(value) = value {
                merged.insert(normalized_name, (name.clone(), value.clone()));
            }
        }
    }
    (!merged.is_empty()).then(|| merged.into_values().collect())
}

pub(crate) fn has_non_empty_header(headers: &ProviderHeaders, expected: &str) -> bool {
    headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case(expected)
            && value
                .as_deref()
                .is_some_and(|value| !crate::utils::text::js_trim(value).is_empty())
    })
}

/// Whether a header carries a credential and must not appear in `Debug`
/// output (an ai.rs safety measure; Pi has no `Debug`).
pub(crate) fn is_sensitive_header(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    matches!(
        name.as_str(),
        "authorization" | "proxy-authorization" | "cookie" | "set-cookie"
    ) || ["key", "token", "secret", "session"]
        .iter()
        .any(|part| name.contains(part))
}

/// A header for `Debug` output: credential values become `<redacted>`.
pub(crate) fn redact_header<'a>(name: &'a str, value: &'a str) -> (&'a str, &'a str) {
    if is_sensitive_header(name) {
        (name, "<redacted>")
    } else {
        (name, value)
    }
}

/// [`ProviderHeaders`] for `Debug` output, with credential values redacted.
pub(crate) fn redacted_provider_headers(headers: &ProviderHeaders) -> Vec<(&str, Option<&str>)> {
    headers
        .iter()
        .map(|(name, value)| match value {
            Some(value) => {
                let (name, value) = redact_header(name, value);
                (name, Some(value))
            }
            None => (name.as_str(), None),
        })
        .collect()
}

/// Applies per-request provider header overrides after provider defaults.
/// A `None` value removes an existing header. `HeaderMap` names are
/// case-insensitive, so replacement and suppression are too.
pub fn apply_provider_headers(headers: &mut HeaderMap, overrides: &ProviderHeaders) -> Result<()> {
    for (name, value) in overrides {
        let Ok(name) = HeaderName::from_bytes(name.as_bytes()) else {
            continue;
        };
        headers.remove(&name);
        if let Some(value) = value {
            let value = HeaderValue::from_str(value)
                .map_err(|error| Error::InvalidHeaderValue(name.to_string(), error))?;
            headers.insert(name, value);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(entries: &[(&str, Option<&str>)]) -> ProviderHeaders {
        entries
            .iter()
            .map(|(name, value)| (name.to_string(), value.map(str::to_string)))
            .collect()
    }

    #[test]
    fn merges_provider_headers_case_insensitively_with_null_suppression() {
        let defaults = headers(&[("User-Agent", Some("pi")), ("X-Remove", Some("1"))]);
        let overrides = headers(&[("user-agent", Some("custom")), ("x-remove", None)]);
        let merged =
            provider_headers_to_record(&[Some(&defaults), None, Some(&overrides)]).unwrap();
        assert_eq!(
            merged.into_iter().collect::<Vec<_>>(),
            vec![("user-agent".to_string(), "custom".to_string())]
        );
        assert_eq!(provider_headers_to_record(&[None]), None);
        assert_eq!(
            provider_headers_to_record(&[Some(&headers(&[("a", None)]))]),
            None
        );
    }

    #[test]
    fn applies_overrides_to_reqwest_header_maps() {
        let mut map = HeaderMap::new();
        map.insert("x-default", HeaderValue::from_static("1"));
        map.insert("x-keep", HeaderValue::from_static("2"));
        apply_provider_headers(
            &mut map,
            &headers(&[("X-Default", None), ("x-new", Some("3"))]),
        )
        .unwrap();
        assert!(map.get("x-default").is_none());
        assert_eq!(map["x-keep"], "2");
        assert_eq!(map["x-new"], "3");
        assert_eq!(headers_to_record(&map).len(), 2);
    }
}
