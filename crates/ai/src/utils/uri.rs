/// Ensures URL ends without a trailing slash
///
/// # Examples
/// ```
/// use your_crate::utils::ensure_no_trailing_slash;
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
/// use your_crate::utils::ensure_trailing_slash;
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
            ("http://example.com//", "http://example.com"),
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
            ("http://example.com//", "http://example.com/"),
        ];

        for (input, expected) in test_cases {
            // Test with &str
            assert_eq!(ensure_trailing_slash(input), expected);

            // Test with String
            assert_eq!(ensure_trailing_slash(input.to_string()), expected);
        }
    }

    #[test]
    fn test_performance_no_allocation_when_unchanged() {
        let input = String::from("http://example.com");
        let result = ensure_no_trailing_slash(input.clone());

        // If input doesn't end with slash, it should return the same allocation
        assert_eq!(result.as_ptr(), input.as_ptr());
    }
}
