//! `path.matchesGlob(path, pattern)` on POSIX, for the adapters that filter
//! sandbox file lists with it (Pi, fx).


fn expand_braces(pattern: &str) -> Vec<String> {
    let bytes: Vec<char> = pattern.chars().collect();
    let mut depth = 0usize;
    let mut start = None;
    let mut escaped = false;
    for (index, character) in bytes.iter().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        match character {
            '\\' => escaped = true,
            '{' => {
                if depth == 0 {
                    start = Some(index);
                }
                depth += 1;
            }
            '}' if depth > 0 => {
                depth -= 1;
                if depth == 0
                    && let Some(open) = start
                {
                    let inner: String = bytes[open + 1..index].iter().collect();
                    let alternatives = split_top_level_commas(&inner);
                    if alternatives.len() < 2 {
                        start = None;
                        continue;
                    }
                    let prefix: String = bytes[..open].iter().collect();
                    let suffix: String = bytes[index + 1..].iter().collect();
                    return alternatives
                        .iter()
                        .flat_map(|alternative| {
                            expand_braces(&format!("{prefix}{alternative}{suffix}"))
                        })
                        .collect();
                }
            }
            _ => {}
        }
    }
    vec![pattern.to_string()]
}

fn split_top_level_commas(input: &str) -> Vec<String> {
    let mut parts = vec![String::new()];
    let mut depth = 0usize;
    let mut escaped = false;
    for character in input.chars() {
        if escaped {
            escaped = false;
            parts.last_mut().unwrap().push(character);
            continue;
        }
        match character {
            '\\' => {
                escaped = true;
                parts.last_mut().unwrap().push(character);
            }
            '{' => {
                depth += 1;
                parts.last_mut().unwrap().push(character);
            }
            '}' => {
                depth = depth.saturating_sub(1);
                parts.last_mut().unwrap().push(character);
            }
            ',' if depth == 0 => parts.push(String::new()),
            _ => parts.last_mut().unwrap().push(character),
        }
    }
    parts
}

fn segment_regex(segment: &str) -> Option<regex::Regex> {
    let characters: Vec<char> = segment.chars().collect();
    let mut source = String::from("^");
    let mut index = 0;
    while index < characters.len() {
        let character = characters[index];
        match character {
            '\\' if index + 1 < characters.len() => {
                index += 1;
                source.push_str(&regex::escape(&characters[index].to_string()));
            }
            '*' => source.push_str("[^/]*"),
            '?' => source.push_str("[^/]"),
            '[' => {
                let close = characters[index + 1..]
                    .iter()
                    .enumerate()
                    .skip(1)
                    .find(|(_, character)| **character == ']')
                    .map(|(offset, _)| index + 1 + offset);
                match close {
                    Some(close) => {
                        let mut class: String = characters[index + 1..close].iter().collect();
                        if let Some(rest) = class.strip_prefix(['!', '^']) {
                            class = format!("^{rest}");
                        }
                        source.push('[');
                        source.push_str(&class.replace('\\', r"\\").replace('[', r"\["));
                        source.push(']');
                        index = close;
                    }
                    None => source.push_str(r"\["),
                }
            }
            other => source.push_str(&regex::escape(&other.to_string())),
        }
        index += 1;
    }
    source.push('$');
    regex::Regex::new(&source).ok()
}

fn match_segments(pattern: &[&str], path: &[&str]) -> bool {
    let Some((first, rest)) = pattern.split_first() else {
        return path.is_empty();
    };
    if *first == "**" {
        // `**` matches zero or more segments, never dot segments.
        if match_segments(rest, path) {
            return true;
        }
        return match path.split_first() {
            Some((segment, remaining)) if !segment.starts_with('.') => {
                match_segments(pattern, remaining)
            }
            _ => false,
        };
    }
    let Some((segment, remaining)) = path.split_first() else {
        return false;
    };
    if segment.starts_with('.') && !first.starts_with('.') {
        return false;
    }
    if matches!(*segment, "." | "..") && *first != *segment {
        return false;
    }
    segment_regex(first).is_some_and(|regex| regex.is_match(segment))
        && match_segments(rest, remaining)
}

/// `path.matchesGlob(path, pattern)` on POSIX.
///
/// [rust] Covers the minimatch subset Pi's `find` patterns use: `*`, `?`,
/// `[...]`, `{a,b}` and `**`, with leading dots matched only explicitly.
pub(crate) fn matches_glob(path: &str, pattern: &str) -> bool {
    let path_segments: Vec<&str> = path.split('/').filter(|part| !part.is_empty()).collect();
    expand_braces(pattern).iter().any(|pattern| {
        let pattern_segments: Vec<&str> =
            pattern.split('/').filter(|part| !part.is_empty()).collect();
        match_segments(&pattern_segments, &path_segments)
    })
}

