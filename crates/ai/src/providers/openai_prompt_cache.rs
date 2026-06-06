pub const OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH: usize = 64;

pub fn clamp_openai_prompt_cache_key(key: Option<&str>) -> Option<String> {
    let key = key?;
    let chars = key.chars().collect::<Vec<_>>();
    if chars.len() <= OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH {
        return Some(key.to_string());
    }
    Some(
        chars
            .into_iter()
            .take(OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH)
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamps_by_characters_not_bytes() {
        let key = format!("{}{}", "a".repeat(63), "🤖🤖");
        let clamped = clamp_openai_prompt_cache_key(Some(&key)).expect("clamped");

        assert_eq!(clamped.chars().count(), 64);
        assert!(clamped.ends_with('🤖'));
    }

    #[test]
    fn returns_none_for_missing_key() {
        assert_eq!(clamp_openai_prompt_cache_key(None), None);
    }
}
