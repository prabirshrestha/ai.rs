//! [rust] TS errors are `unknown`: an `Error` object or any thrown value.
//! [`ErrorValue`] carries either, and the helpers here reproduce the JS
//! conversions the AI SDK applies to them (`String(err)`, `getErrorMessage`,
//! `error.name`).

use std::error::Error as StdError;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::harness::errors::{AbortError, BoxError, HarnessError};
use crate::harness::provider::{JsonValue, TypeValidationError};

/// A shareable error object (a JS `Error` that several promises reject
/// with).
pub type SharedError = Arc<dyn StdError + Send + Sync + 'static>;

/// Box a shared error back into a [`BoxError`] (keeps `downcast_ref` working
/// through [`SharedErrorBox::inner`]).
#[derive(Clone)]
pub struct SharedErrorBox(pub SharedError);

impl fmt::Debug for SharedErrorBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl fmt::Display for SharedErrorBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl StdError for SharedErrorBox {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.0.source()
    }
}

impl SharedErrorBox {
    /// The wrapped error.
    pub fn inner(&self) -> &(dyn StdError + 'static) {
        &*self.0
    }
}

/// A thrown value that is not an `Error` (`throw 'boom'`, `throw { code }`).
#[derive(Debug, Clone, PartialEq)]
pub struct ThrownValue(pub JsonValue);

impl fmt::Display for ThrownValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&js_string_of_value(&self.0))
    }
}

impl StdError for ThrownValue {}

/// `unknown` in an error position.
#[derive(Clone)]
pub enum ErrorValue {
    /// An `Error` instance.
    Error(SharedError),
    /// Any other value.
    Value(JsonValue),
}

impl fmt::Debug for ErrorValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Error(e) => write!(f, "Error({}: {})", error_name(&**e), e),
            Self::Value(v) => write!(f, "Value({v})"),
        }
    }
}

impl PartialEq for ErrorValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Error(a), Self::Error(b)) => {
                Arc::ptr_eq(a, b)
                    || (error_name(&**a) == error_name(&**b) && a.to_string() == b.to_string())
            }
            (Self::Value(a), Self::Value(b)) => a == b,
            _ => false,
        }
    }
}

impl ErrorValue {
    /// `new Error(message)`.
    pub fn message(message: impl Into<String>) -> Self {
        let err: BoxError = message.into().into();
        Self::Error(Arc::from(err))
    }

    /// Wrap a boxed error. A [`ThrownValue`] becomes [`ErrorValue::Value`]
    /// and a [`SharedErrorBox`] is unwrapped.
    pub fn from_box(err: BoxError) -> Self {
        if let Some(thrown) = err.downcast_ref::<ThrownValue>() {
            return Self::Value(thrown.0.clone());
        }
        if let Some(shared) = err.downcast_ref::<SharedErrorBox>() {
            return Self::Error(shared.0.clone());
        }
        Self::Error(Arc::from(err))
    }

    /// Wrap a shared error.
    pub fn from_shared(err: SharedError) -> Self {
        if let Some(thrown) = err.downcast_ref::<ThrownValue>() {
            return Self::Value(thrown.0.clone());
        }
        Self::Error(err)
    }

    /// The error as a [`BoxError`] (what a promise rejects with).
    pub fn to_box(&self) -> BoxError {
        match self {
            Self::Error(e) => Box::new(SharedErrorBox(e.clone())),
            Self::Value(v) => Box::new(ThrownValue(v.clone())),
        }
    }

    /// The error as a [`SharedError`].
    pub fn to_shared(&self) -> SharedError {
        match self {
            Self::Error(e) => e.clone(),
            Self::Value(v) => Arc::new(ThrownValue(v.clone())),
        }
    }

    /// The `Error` instance, if this is one.
    pub fn as_error(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Error(e) => Some(unwrap_shared(&**e)),
            Self::Value(_) => None,
        }
    }

    /// `err instanceof Error`.
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }

    /// `String(err)`.
    pub fn js_string(&self) -> String {
        match self {
            Self::Error(e) => js_error_to_string(&**e),
            Self::Value(v) => js_string_of_value(v),
        }
    }

    /// `JSON.parse(JSON.stringify(err))`: an `Error` has no enumerable
    /// properties, so it becomes `{}`.
    pub fn to_js_json(&self) -> JsonValue {
        match self {
            Self::Error(_) => JsonValue::Object(Default::default()),
            Self::Value(v) => v.clone(),
        }
    }

    /// `getErrorMessage(err)` (`@ai-sdk/provider`).
    pub fn get_error_message(&self) -> String {
        match self {
            Self::Error(e) => js_error_to_string(&**e),
            Self::Value(JsonValue::Null) => "unknown error".into(),
            Self::Value(JsonValue::String(s)) => s.clone(),
            Self::Value(v) => v.to_string(),
        }
    }
}

impl From<BoxError> for ErrorValue {
    fn from(value: BoxError) -> Self {
        Self::from_box(value)
    }
}

impl From<JsonValue> for ErrorValue {
    fn from(value: JsonValue) -> Self {
        Self::Value(value)
    }
}

impl Serialize for ErrorValue {
    /// [rust] `JSON.stringify(new Error())` is `{}`; the port writes the
    /// `name` and `message` so the error stays readable on the wire.
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Error(e) => {
                use serde::ser::SerializeMap as _;
                let mut map = s.serialize_map(Some(2))?;
                map.serialize_entry("name", error_name(&**e))?;
                map.serialize_entry("message", &e.to_string())?;
                map.end()
            }
            Self::Value(v) => v.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for ErrorValue {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        JsonValue::deserialize(d).map(Self::Value)
    }
}

pub(crate) fn unwrap_shared<'a>(err: &'a (dyn StdError + 'static)) -> &'a (dyn StdError + 'static) {
    match err.downcast_ref::<SharedErrorBox>() {
        Some(shared) => unwrap_shared(shared.inner()),
        None => err,
    }
}

/// `error.name` of the error classes the port knows. Other errors are plain
/// `Error`s.
pub fn error_name(err: &(dyn StdError + 'static)) -> &'static str {
    let err = unwrap_shared(err);
    if let Some(e) = err.downcast_ref::<HarnessError>() {
        return e.name();
    }
    if err.is::<AbortError>() {
        return "AbortError";
    }
    if err.is::<TypeValidationError>() {
        return "AI_TypeValidationError";
    }
    if let Some(named) = crate::harness::sdk::named_error_name(err) {
        return named;
    }
    "Error"
}

/// `Error.prototype.toString`: `name: message`, or `name` alone when the
/// message is empty.
pub fn js_error_to_string(err: &(dyn StdError + 'static)) -> String {
    if let Some(thrown) = unwrap_shared(err).downcast_ref::<ThrownValue>() {
        return js_string_of_value(&thrown.0);
    }
    let name = error_name(err);
    let message = err.to_string();
    if message.is_empty() {
        name.to_string()
    } else {
        format!("{name}: {message}")
    }
}

/// `String(value)` for a JSON value.
pub fn js_string_of_value(value: &JsonValue) -> String {
    match value {
        JsonValue::Null => "null".into(),
        JsonValue::Bool(b) => b.to_string(),
        JsonValue::Number(n) => n.to_string(),
        JsonValue::String(s) => s.clone(),
        JsonValue::Array(items) => items
            .iter()
            .map(|v| match v {
                JsonValue::Null => String::new(),
                other => js_string_of_value(other),
            })
            .collect::<Vec<_>>()
            .join(","),
        JsonValue::Object(_) => "[object Object]".into(),
    }
}

/// `getErrorMessage(error)` for a boxed error (`error.toString()`).
pub fn get_error_message(err: &(dyn StdError + 'static)) -> String {
    js_error_to_string(err)
}

/// `generateId()` (`@ai-sdk/provider-utils`): 16 random characters from
/// `0-9A-Za-z`.
pub fn generate_id() -> String {
    use ring::rand::SecureRandom as _;
    const ALPHABET: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    let mut bytes = [0u8; 16];
    ring::rand::SystemRandom::new()
        .fill(&mut bytes)
        .expect("the system random number generator failed");
    bytes
        .iter()
        .map(|b| ALPHABET[(*b as usize) % ALPHABET.len()] as char)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn js_string_matches_string_of_error_and_values() {
        assert_eq!(ErrorValue::message("boom").js_string(), "Error: boom");
        assert_eq!(ErrorValue::message("").js_string(), "Error");
        let err: BoxError = HarnessError::new("x").into();
        assert_eq!(ErrorValue::from(err).js_string(), "AI_HarnessError: x");
        assert_eq!(ErrorValue::Value(json!("boom")).js_string(), "boom");
        assert_eq!(
            ErrorValue::Value(json!({"a": 1})).js_string(),
            "[object Object]"
        );
        assert_eq!(ErrorValue::Value(json!([1, null, "a"])).js_string(), "1,,a");
        assert_eq!(ErrorValue::Value(json!(42)).js_string(), "42");
    }

    #[test]
    fn thrown_values_round_trip_through_box() {
        let boxed: BoxError = Box::new(ThrownValue(json!({"code": 1})));
        let value = ErrorValue::from(boxed);
        assert_eq!(value, ErrorValue::Value(json!({"code": 1})));
        let again = ErrorValue::from(value.to_box());
        assert_eq!(again, value);
    }

    #[test]
    fn shared_errors_keep_their_identity() {
        let err: BoxError = HarnessError::new("kept").into();
        let value = ErrorValue::from(err);
        let boxed = value.to_box();
        assert_eq!(
            crate::harness::errors::get_harness_error_message(
                ErrorValue::from(boxed).as_error().unwrap()
            ),
            "kept"
        );
    }

    #[test]
    fn get_error_message_mirrors_the_provider_helper() {
        assert_eq!(
            ErrorValue::Value(JsonValue::Null).get_error_message(),
            "unknown error"
        );
        assert_eq!(ErrorValue::Value(json!("s")).get_error_message(), "s");
        assert_eq!(
            ErrorValue::Value(json!({"a": 1})).get_error_message(),
            r#"{"a":1}"#
        );
        assert_eq!(ErrorValue::message("m").get_error_message(), "Error: m");
    }

    #[test]
    fn serializes_errors_with_name_and_message() {
        assert_eq!(
            serde_json::to_value(ErrorValue::message("m")).unwrap(),
            json!({"name": "Error", "message": "m"})
        );
        assert_eq!(
            serde_json::to_value(ErrorValue::Value(json!(1))).unwrap(),
            json!(1)
        );
    }

    #[test]
    fn generate_id_is_16_alphanumeric_characters() {
        let id = generate_id();
        assert_eq!(id.len(), 16);
        assert!(id.chars().all(|c| c.is_ascii_alphanumeric()));
    }
}
