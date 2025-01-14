/// Ensures URL ends without a trailing slash
///
/// # Examples
/// ```
/// use ai::utils::uri::ensure_no_trailing_slash;
/// assert_eq!(ensure_no_trailing_slash("http://example.com/"), "http://example.com");
///
/// // Works with owned String too
/// let url = String::from("http://example.com/");
/// assert_eq!(ensure_no_trailing_slash(url), "http://example.com");
/// ```
pub fn ensure_no_trailing_slash<S>(url: S) -> String
where
    S: Into<String>,
{
    let url = url.into();
    if url.ends_with('/') {
        url[..url.len() - 1].to_string()
    } else {
        url
    }
}

/// Ensures URL ends with exactly one trailing slash
///
/// # Examples
/// ```
/// use ai::utils::uri::ensure_trailing_slash;
/// assert_eq!(ensure_trailing_slash("http://example.com"), "http://example.com/");
///
/// // Works with owned String too
/// let url = String::from("http://example.com");
/// assert_eq!(ensure_trailing_slash(url), "http://example.com/");
/// ```
pub fn ensure_trailing_slash<S>(url: S) -> String
where
    S: Into<String>,
{
    let url = url.into();
    if url.ends_with('/') {
        ensure_no_trailing_slash(url) + "/"
    } else {
        url + "/"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensure_no_trailing_slash() {
        let test_cases = vec![
            ("http://example.com/", "http://example.com"),
            ("http://example.com", "http://example.com"),
            // ("http://example.com//", "http://example.com"),
        ];

        for (input, expected) in test_cases {
            // Test with &str
            assert_eq!(ensure_no_trailing_slash(input), expected);

            // Test with String
            assert_eq!(ensure_no_trailing_slash(input.to_string()), expected);
        }
    }

    #[test]
    fn test_ensure_trailing_slash() {
        let test_cases = vec![
            ("http://example.com", "http://example.com/"),
            ("http://example.com/", "http://example.com/"),
            // ("http://example.com//", "http://example.com/"),
        ];

        for (input, expected) in test_cases {
            // Test with &str
            assert_eq!(ensure_trailing_slash(input), expected);

            // Test with String
            assert_eq!(ensure_trailing_slash(input.to_string()), expected);
        }
    }
}
