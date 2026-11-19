//! Port of `shell-quote.ts`.

/// `shellQuote`: single-quote a value for a POSIX shell.
pub fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', r"'\''"))
}

#[cfg(test)]
mod tests {
    use super::*;

    // shell-quote.test.ts
    #[test]
    fn wraps_values_in_single_quotes() {
        assert_eq!(shell_quote("hello"), "'hello'");
    }

    #[test]
    fn escapes_embedded_single_quotes() {
        assert_eq!(shell_quote("it's"), r"'it'\''s'");
    }

    #[test]
    fn handles_empty_strings() {
        assert_eq!(shell_quote(""), "''");
    }
}
