//! Harness adapters, each behind its own feature.

#[cfg(any(feature = "harness-native", feature = "harness-pi"))]
pub mod loop_session;
#[cfg(feature = "harness-native")]
pub mod native;
#[cfg(any(feature = "harness-native", feature = "harness-pi"))]
pub mod pi;

#[cfg(test)]
pub(crate) mod test_temp_dir;
