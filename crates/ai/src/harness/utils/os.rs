//! Port of `os.ts`. `platform` is a Node `process.platform` value.

/// `isMacOS`.
pub fn is_macos(platform: &str) -> bool {
    platform == "darwin"
}

/// `isLinux`.
pub fn is_linux(platform: &str) -> bool {
    platform == "linux"
}

/// `isWindows`.
pub fn is_windows(platform: &str) -> bool {
    platform == "win32"
}

/// [rust] `process.platform` of the host.
pub fn current_platform() -> &'static str {
    match std::env::consts::OS {
        "macos" => "darwin",
        "windows" => "win32",
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // os.test.ts
    #[test]
    fn identifies_macos() {
        assert!(is_macos("darwin"));
        assert!(!is_macos("linux"));
        assert!(!is_macos("win32"));
    }

    #[test]
    fn identifies_linux() {
        assert!(!is_linux("darwin"));
        assert!(is_linux("linux"));
        assert!(!is_linux("win32"));
    }

    #[test]
    fn identifies_windows() {
        assert!(!is_windows("darwin"));
        assert!(!is_windows("linux"));
        assert!(is_windows("win32"));
    }
}
