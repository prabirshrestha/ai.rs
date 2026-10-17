//! Port of chord `src/json.ts` (`copyJson`, `isJsonValue`) and the `JsonValue` type.
//!
//! `JsonValue` is [`serde_json::Value`] (with `preserve_order`, so object keys
//! keep insertion order). Most strict-JSON violations JS has to check for at
//! runtime (cycles, accessors, symbol keys, class instances, sparse arrays,
//! functions, bigint) cannot be represented by a serde data model, so the only
//! runtime check left is the finite-number one.
//!
//! Divergences from Pi:
//! - Rust has no `undefined`. `Option::None` serializes as `null` unless the type
//!   uses `#[serde(skip_serializing_if = "Option::is_none")]`, which is the Rust
//!   spelling of an omitted property, so [`CopyJsonOptions::omit_undefined_properties`]
//!   has no effect and is kept for call-site parity.
//! - JS integer-like object keys sort first; serde keeps insertion order.

use serde::Serialize;
use serde::ser;

pub use serde_json::Value as JsonValue;

/// `CopyJsonOptions`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CopyJsonOptions {
    /// Omit undefined object properties while preserving strict array semantics.
    pub omit_undefined_properties: bool,
}

/// A value that is not strict JSON (`TypeError` in Pi).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct JsonError(pub String);

impl ser::Error for JsonError {
    fn custom<T: std::fmt::Display>(message: T) -> Self {
        JsonError(message.to_string())
    }
}

const NON_FINITE: &str = "Value contains a non-finite number and is not strict JSON";

/// Copy a value into an alias-free strict-JSON tree owned by the caller.
pub fn copy_json<T: Serialize + ?Sized>(
    value: &T,
    options: Option<CopyJsonOptions>,
) -> Result<JsonValue, JsonError> {
    let _ = options.unwrap_or_default().omit_undefined_properties;
    value.serialize(StrictCheck)?;
    serde_json::to_value(value)
        .map_err(|error| JsonError(format!("Value is not strict JSON: {error}")))
}

/// Return whether a value is finite strict JSON with plain objects and no cycles.
pub fn is_json_value<T: Serialize + ?Sized>(value: &T) -> bool {
    value.serialize(StrictCheck).is_ok() && serde_json::to_value(value).is_ok()
}

/// Walks a serde value and rejects what `JSON.stringify` would silently turn into `null`.
struct StrictCheck;

macro_rules! ok_scalar {
    ($($name:ident: $ty:ty),* $(,)?) => {
        $(fn $name(self, _value: $ty) -> Result<(), JsonError> { Ok(()) })*
    };
}

impl ser::Serializer for StrictCheck {
    type Ok = ();
    type Error = JsonError;
    type SerializeSeq = Self;
    type SerializeTuple = Self;
    type SerializeTupleStruct = Self;
    type SerializeTupleVariant = Self;
    type SerializeMap = Self;
    type SerializeStruct = Self;
    type SerializeStructVariant = Self;

    ok_scalar!(
        serialize_bool: bool,
        serialize_i8: i8,
        serialize_i16: i16,
        serialize_i32: i32,
        serialize_i64: i64,
        serialize_i128: i128,
        serialize_u8: u8,
        serialize_u16: u16,
        serialize_u32: u32,
        serialize_u64: u64,
        serialize_u128: u128,
        serialize_char: char,
        serialize_str: &str,
        serialize_bytes: &[u8],
    );

    fn serialize_f32(self, value: f32) -> Result<(), JsonError> {
        if value.is_finite() {
            Ok(())
        } else {
            Err(JsonError(NON_FINITE.into()))
        }
    }

    fn serialize_f64(self, value: f64) -> Result<(), JsonError> {
        if value.is_finite() {
            Ok(())
        } else {
            Err(JsonError(NON_FINITE.into()))
        }
    }

    fn serialize_none(self) -> Result<(), JsonError> {
        Ok(())
    }

    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<(), JsonError> {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<(), JsonError> {
        Ok(())
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<(), JsonError> {
        Ok(())
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _index: u32,
        _variant: &'static str,
    ) -> Result<(), JsonError> {
        Ok(())
    }

    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<(), JsonError> {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        _index: u32,
        _variant: &'static str,
        value: &T,
    ) -> Result<(), JsonError> {
        value.serialize(self)
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Self, JsonError> {
        Ok(self)
    }

    fn serialize_tuple(self, _len: usize) -> Result<Self, JsonError> {
        Ok(self)
    }

    fn serialize_tuple_struct(self, _name: &'static str, _len: usize) -> Result<Self, JsonError> {
        Ok(self)
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self, JsonError> {
        Ok(self)
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Self, JsonError> {
        Ok(self)
    }

    fn serialize_struct(self, _name: &'static str, _len: usize) -> Result<Self, JsonError> {
        Ok(self)
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self, JsonError> {
        Ok(self)
    }
}

macro_rules! compound {
    ($trait:ident, $method:ident) => {
        impl ser::$trait for StrictCheck {
            type Ok = ();
            type Error = JsonError;
            fn $method<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), JsonError> {
                value.serialize(StrictCheck)
            }
            fn end(self) -> Result<(), JsonError> {
                Ok(())
            }
        }
    };
}

compound!(SerializeSeq, serialize_element);
compound!(SerializeTuple, serialize_element);
compound!(SerializeTupleStruct, serialize_field);
compound!(SerializeTupleVariant, serialize_field);

impl ser::SerializeMap for StrictCheck {
    type Ok = ();
    type Error = JsonError;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), JsonError> {
        key.serialize(StrictCheck)
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), JsonError> {
        value.serialize(StrictCheck)
    }
    fn end(self) -> Result<(), JsonError> {
        Ok(())
    }
}

impl ser::SerializeStruct for StrictCheck {
    type Ok = ();
    type Error = JsonError;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        _key: &'static str,
        value: &T,
    ) -> Result<(), JsonError> {
        value.serialize(StrictCheck)
    }
    fn end(self) -> Result<(), JsonError> {
        Ok(())
    }
}

impl ser::SerializeStructVariant for StrictCheck {
    type Ok = ();
    type Error = JsonError;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        _key: &'static str,
        value: &T,
    ) -> Result<(), JsonError> {
        value.serialize(StrictCheck)
    }
    fn end(self) -> Result<(), JsonError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::BTreeMap;

    #[test]
    fn checks_strict_json_without_normalizing_it() {
        assert!(is_json_value(&json!({ "nested": [1, true, null] })));
        assert!(!is_json_value(&f64::INFINITY));
        assert!(!is_json_value(&vec![1.0, f64::NAN]));
        let mut map = BTreeMap::new();
        map.insert("value", f32::NEG_INFINITY);
        assert!(!is_json_value(&map));
    }

    #[test]
    fn copies_strict_json_without_retaining_aliases() {
        #[derive(Serialize)]
        struct Shared {
            value: i32,
        }
        #[derive(Serialize)]
        struct Input<'a> {
            left: &'a Shared,
            right: &'a Shared,
        }
        let shared = Shared { value: 1 };
        let copied = copy_json(
            &Input {
                left: &shared,
                right: &shared,
            },
            None,
        )
        .unwrap();
        assert_eq!(
            copied,
            json!({ "left": { "value": 1 }, "right": { "value": 1 } })
        );
    }

    #[test]
    fn preserves_own_proto_data_properties() {
        let input: JsonValue = serde_json::from_str(r#"{"__proto__":{"safe":true}}"#).unwrap();
        let copied = copy_json(&input, None).unwrap();
        assert_eq!(copied["__proto__"], json!({ "safe": true }));
    }

    #[test]
    fn rejects_non_finite_numbers() {
        let error = copy_json(&json_with_nan(), None).unwrap_err();
        assert!(error.to_string().contains("strict JSON"));
        let options = CopyJsonOptions {
            omit_undefined_properties: true,
        };
        assert!(copy_json(&vec![f64::NAN], Some(options)).is_err());
    }

    fn json_with_nan() -> BTreeMap<&'static str, f64> {
        BTreeMap::from([("kept", 1.0), ("broken", f64::NAN)])
    }
}
