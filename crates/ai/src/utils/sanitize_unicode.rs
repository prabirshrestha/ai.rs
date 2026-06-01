/// Rust `str` values are already valid Unicode scalar values, so they cannot
/// contain the unpaired UTF-16 surrogate code units that upstream Pi removes
/// from JavaScript strings. Valid emoji and other non-BMP characters are
/// preserved.
pub fn sanitize_surrogates(text: &str) -> String {
    text.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_handle_emoji_in_tool_results() {
        let text = "Mario Zechner wann? Wo? Bin grad aeussersr eventuninformiert 🙈";

        assert_eq!(sanitize_surrogates(text), text);
    }
}
