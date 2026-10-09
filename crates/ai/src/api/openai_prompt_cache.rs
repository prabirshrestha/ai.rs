//! Port of `api/openai-prompt-cache.ts`.

pub const OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH: usize = 64;

/// Clamp a prompt cache key to OpenAI's limit, counted in code points
/// (`Array.from(key)`).
pub fn clamp_openai_prompt_cache_key(key: Option<&str>) -> Option<String> {
    let key = key?;
    if key.chars().count() <= OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH {
        return Some(key.to_string());
    }
    Some(
        key.chars()
            .take(OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH)
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamps_by_code_points() {
        assert_eq!(clamp_openai_prompt_cache_key(None), None);
        assert_eq!(
            clamp_openai_prompt_cache_key(Some("abc")).as_deref(),
            Some("abc")
        );
        let long = "é".repeat(70);
        assert_eq!(
            clamp_openai_prompt_cache_key(Some(&long))
                .unwrap()
                .chars()
                .count(),
            64
        );
    }
}
