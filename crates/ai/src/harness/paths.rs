//! Port of `harness-session-data-directory-path.ts` and
//! `harnessStateDirectoryPath` (`harness-v1-network-sandbox-session.ts`).

/// Fixed directory, relative to the sandbox's own HOME, that holds every
/// piece of harness-generated state.
const HARNESS_V1_STATE_DIRECTORY_NAME: &str = ".ai-sdk-harness";

/// `encodeURIComponent`: escape everything except `A-Z a-z 0-9 - _ . ! ~ * ' ( )`.
fn encode_uri_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for b in value.bytes() {
        let keep = b.is_ascii_alphanumeric()
            || matches!(
                b,
                b'-' | b'_' | b'.' | b'!' | b'~' | b'*' | b'\'' | b'(' | b')'
            );
        if keep {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// `encodeHarnessPathSegment`: encode a string as exactly one path segment.
/// The mapping is injective, and the result never contains a separator or
/// resolves to `.` or `..`.
///
/// [rust] A Rust `&str` cannot hold lone surrogates, so the TS `URIError`
/// case cannot occur.
pub fn encode_harness_path_segment(value: &str) -> String {
    let encoded = encode_uri_component(value);
    match encoded.as_str() {
        "" => "%".into(),
        "." => "%2E".into(),
        ".." => "%2E%2E".into(),
        _ => encoded,
    }
}

/// `posix.join` for the simple cases used here: joins with `/` and
/// collapses duplicate separators.
pub(crate) fn posix_join(parts: &[&str]) -> String {
    let joined = parts
        .iter()
        .filter(|p| !p.is_empty())
        .copied()
        .collect::<Vec<_>>()
        .join("/");
    normalize_posix(&joined)
}

/// `posix.normalize`.
pub(crate) fn normalize_posix(path: &str) -> String {
    if path.is_empty() {
        return ".".into();
    }
    let absolute = path.starts_with('/');
    let trailing = path.ends_with('/');
    let mut stack: Vec<&str> = Vec::new();
    for seg in path.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                if stack.last().is_some_and(|s| *s != "..") {
                    stack.pop();
                } else if !absolute {
                    stack.push("..");
                }
            }
            s => stack.push(s),
        }
    }
    let mut out = stack.join("/");
    if absolute {
        out.insert(0, '/');
    }
    if out.is_empty() {
        return if absolute { "/".into() } else { ".".into() };
    }
    if trailing && !out.ends_with('/') {
        out.push('/');
    }
    out
}

/// `posix.isAbsolute`.
pub(crate) fn is_absolute_posix(path: &str) -> bool {
    path.starts_with('/')
}

/// `posix.resolve(base, path)` for an absolute `base`.
pub(crate) fn resolve_posix(base: &str, path: &str) -> String {
    let joined = if is_absolute_posix(path) {
        path.to_string()
    } else {
        format!("{base}/{path}")
    };
    let normalized = normalize_posix(&joined);
    if normalized.len() > 1 {
        normalized.trim_end_matches('/').to_string()
    } else {
        normalized
    }
}

/// `posix.dirname`.
#[allow(dead_code)] // [rust] Used only by some adapter features.
pub(crate) fn dirname_posix(path: &str) -> String {
    let trimmed = if path.len() > 1 {
        path.trim_end_matches('/')
    } else {
        path
    };
    match trimmed.rfind('/') {
        None => ".".into(),
        Some(0) => "/".into(),
        Some(i) => trimmed[..i].to_string(),
    }
}

/// `harnessSessionDataDirectoryPath({ stateDirectory, sessionId })`:
/// `<stateDirectory>/.agent-runs/<encoded sessionId>`.
pub fn harness_session_data_directory_path(state_directory: &str, session_id: &str) -> String {
    posix_join(&[
        state_directory,
        ".agent-runs",
        &encode_harness_path_segment(session_id),
    ])
}

/// `harnessStateDirectoryPath({ sandboxHomeDir })`: `<home>/.ai-sdk-harness`.
pub fn harness_state_directory_path(sandbox_home_dir: &str) -> String {
    posix_join(&[sandbox_home_dir, HARNESS_V1_STATE_DIRECTORY_NAME])
}

#[cfg(test)]
mod tests {
    use super::*;

    const STATE_DIRECTORY: &str = "/home/agent/.ai-sdk-harness";
    const AGENT_RUNS_DIRECTORY: &str = "/home/agent/.ai-sdk-harness/.agent-runs";
    const ENCODED_SEGMENTS: [(&str, &str); 10] = [
        (
            "../../../../work/project",
            "..%2F..%2F..%2F..%2Fwork%2Fproject",
        ),
        ("/work/project", "%2Fwork%2Fproject"),
        ("..", "%2E%2E"),
        (".", "%2E"),
        ("", "%"),
        ("%", "%25"),
        ("a/b", "a%2Fb"),
        ("a%2Fb", "a%252Fb"),
        ("a\\..", "a%5C.."),
        ("a\0b", "a%00b"),
    ];

    // harness-session-data-directory-path.test.ts
    #[test]
    fn encodes_each_value_as_its_segment() {
        for (value, segment) in ENCODED_SEGMENTS {
            assert_eq!(encode_harness_path_segment(value), segment, "{value:?}");
        }
    }

    #[test]
    fn maps_distinct_values_to_distinct_segments() {
        assert_ne!(
            encode_harness_path_segment("a/b"),
            encode_harness_path_segment("a-b")
        );
        assert_ne!(
            encode_harness_path_segment("."),
            encode_harness_path_segment("%2E")
        );
    }

    #[test]
    fn keeps_ordinary_session_ids_readable() {
        assert_eq!(
            harness_session_data_directory_path(STATE_DIRECTORY, "s1-abc"),
            format!("{AGENT_RUNS_DIRECTORY}/s1-abc")
        );
    }

    #[test]
    fn encodes_session_ids_as_one_path_segment() {
        for (session_id, segment) in ENCODED_SEGMENTS {
            let directory = harness_session_data_directory_path(STATE_DIRECTORY, session_id);
            assert_eq!(directory, format!("{AGENT_RUNS_DIRECTORY}/{segment}"));
            let parent = &directory[..directory.rfind('/').unwrap()];
            assert_eq!(parent, AGENT_RUNS_DIRECTORY);
        }
    }

    #[test]
    fn encodes_non_ascii_as_utf8_percent_escapes() {
        assert_eq!(encode_harness_path_segment("é"), "%C3%A9");
    }

    // harness-v1-network-sandbox-session.test.ts
    #[test]
    fn uses_the_sandbox_home_for_state() {
        assert_eq!(
            harness_state_directory_path("/home/agent"),
            "/home/agent/.ai-sdk-harness"
        );
    }

    #[test]
    fn can_form_the_symbolic_path_used_by_the_bootstrap_hash() {
        assert_eq!(
            harness_state_directory_path("$HOME"),
            "$HOME/.ai-sdk-harness"
        );
    }

    #[test]
    fn posix_join_matches_node() {
        assert_eq!(posix_join(&["/a/", "b"]), "/a/b");
        assert_eq!(posix_join(&["/a", "../b"]), "/b");
        assert_eq!(posix_join(&["a", "."]), "a");
    }
}
