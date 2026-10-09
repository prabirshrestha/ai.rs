//! Port of `utils/sanitize-unicode.ts`.

/// Removes unpaired Unicode surrogate characters from a string.
///
/// A Rust `&str` is valid UTF-8 and cannot contain unpaired surrogates, so
/// this is the identity. It is kept so API modules can call it where Pi does.
pub fn sanitize_surrogates(text: &str) -> String {
    text.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_valid_emoji() {
        assert_eq!(sanitize_surrogates("Hello 🙈 World"), "Hello 🙈 World");
    }
}
