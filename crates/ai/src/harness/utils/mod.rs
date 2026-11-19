//! Port of `@ai-sdk/harness/utils`: helpers harness adapters share.

pub mod extract_harness_prompt_content;
pub mod is_sandbox_write_protected_path;
pub mod os;
pub mod sandbox_home_dir;
pub mod shell_quote;
pub mod sleep;
pub mod tool_model_output;
pub mod write_instructions;
pub mod write_skills;

pub use crate::harness::spec::paths::encode_harness_path_segment;
pub use extract_harness_prompt_content::{
    HarnessPromptContent, HarnessPromptContentPart, extract_harness_prompt_content,
};
pub use is_sandbox_write_protected_path::*;
pub use os::*;
pub use sandbox_home_dir::*;
pub use shell_quote::*;
pub use sleep::*;
pub use tool_model_output::*;
pub use write_instructions::*;
pub use write_skills::*;

use crate::harness::errors::BoxError;

/// `new Error(message)`: a plain error that `get_harness_error_message`
/// masks.
pub(crate) fn plain_error(message: impl Into<String>) -> BoxError {
    message.into().into()
}

/// `String.prototype.localeCompare` for the identifiers and paths the
/// harness sorts.
///
/// [rust] Approximates the ICU root collation: case-insensitive first, with
/// punctuation before digits before letters, then lowercase before
/// uppercase. This keeps hashes and orderings equal to TS for skill names
/// and file paths.
pub(crate) fn locale_compare(a: &str, b: &str) -> std::cmp::Ordering {
    fn primary(c: char) -> (u8, char) {
        let lower = c.to_lowercase().next().unwrap_or(c);
        let class = if c.is_alphabetic() {
            2
        } else if c.is_numeric() {
            1
        } else {
            0
        };
        (class, lower)
    }
    let pa = a.chars().map(primary);
    let pb = b.chars().map(primary);
    pa.cmp(pb).then_with(|| {
        // Tertiary: lowercase sorts before uppercase.
        let ta = a.chars().map(|c| c.is_uppercase());
        let tb = b.chars().map(|c| c.is_uppercase());
        ta.cmp(tb)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    #[test]
    fn locale_compare_matches_icu_for_common_cases() {
        assert_eq!(locale_compare("reference.md", "SKILL.md"), Ordering::Less);
        assert_eq!(locale_compare("alpha", "zeta"), Ordering::Less);
        assert_eq!(locale_compare("a", "A"), Ordering::Less);
        assert_eq!(locale_compare("_x", "1"), Ordering::Less);
        assert_eq!(locale_compare("1", "a"), Ordering::Less);
        assert_eq!(locale_compare("same", "same"), Ordering::Equal);
    }
}
