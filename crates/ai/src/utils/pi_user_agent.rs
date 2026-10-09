//! Port of `utils/pi-user-agent.ts`.

use std::sync::OnceLock;

/// `pi (<platform> <release>; <arch>)` with Node.js `os.platform()`,
/// `os.release()` and `os.arch()` names.
pub fn get_pi_user_agent() -> String {
    static USER_AGENT: OnceLock<String> = OnceLock::new();
    USER_AGENT
        .get_or_init(|| format!("pi ({} {}; {})", node_platform(), os_release(), node_arch()))
        .clone()
}

fn node_platform() -> &'static str {
    match std::env::consts::OS {
        "macos" => "darwin",
        "windows" => "win32",
        other => other,
    }
}

fn node_arch() -> &'static str {
    match std::env::consts::ARCH {
        "x86_64" => "x64",
        "x86" => "ia32",
        "aarch64" => "arm64",
        "powerpc64" => "ppc64",
        "loongarch64" => "loong64",
        other => other,
    }
}

fn os_release() -> String {
    if let Ok(release) = std::fs::read_to_string("/proc/sys/kernel/osrelease") {
        return release.trim().to_string();
    }
    std::process::Command::new("uname")
        .arg("-r")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|release| release.trim().to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_like_node() {
        let user_agent = get_pi_user_agent();
        assert!(user_agent.starts_with(&format!("pi ({} ", node_platform())));
        assert!(user_agent.ends_with(&format!("; {})", node_arch())));
    }
}
