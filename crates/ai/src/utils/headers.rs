use std::collections::HashMap;

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

use crate::types::ProviderHeaders;
use crate::{Error, Result};

pub fn headers_to_record(headers: &reqwest::header::HeaderMap) -> HashMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.as_str().to_string(), value.to_string()))
        })
        .collect()
}

pub(crate) fn has_non_empty_header(headers: &ProviderHeaders, expected: &str) -> bool {
    headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case(expected)
            && value
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty())
    })
}

/// Applies per-request provider header overrides after provider defaults.
///
/// A `None` value removes an existing header. `HeaderMap` names are
/// case-insensitive, so both replacement and suppression are case-insensitive.
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
