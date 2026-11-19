//! [rust] Zero-sized types for TS string-literal fields (`type: 'start'`,
//! `specificationVersion: 'harness-v1'`). They serialize as the literal and
//! refuse any other value, like a zod `z.literal`.

macro_rules! string_literal {
    ($(#[$meta:meta])* $name:ident = $lit:literal) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
        pub struct $name;

        impl $name {
            /// The literal value.
            pub const VALUE: &'static str = $lit;
        }

        impl serde::Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_str($lit)
            }
        }

        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                let value = <std::borrow::Cow<'de, str> as serde::Deserialize>::deserialize(d)?;
                if value == $lit {
                    Ok($name)
                } else {
                    Err(serde::de::Error::invalid_value(
                        serde::de::Unexpected::Str(&value),
                        &concat!("'", $lit, "'"),
                    ))
                }
            }
        }
    };
}

pub(crate) use string_literal;

string_literal!(
    /// `specificationVersion: 'harness-v1'`.
    HarnessV1SpecificationVersion = "harness-v1"
);
