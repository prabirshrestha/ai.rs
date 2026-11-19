//! Port of `utils/native-subscription/macos-keychain.ts`.

use super::{ExecFile, SystemExecFile};

/// `readMacOSKeychainPassword({ service, account })`: a generic password
/// from the macOS keychain, trimmed, or `None` when it is missing or empty.
pub async fn read_macos_keychain_password(service: &str, account: &str) -> Option<String> {
    read_with(&SystemExecFile, service, account).await
}

async fn read_with(exec: &dyn ExecFile, service: &str, account: &str) -> Option<String> {
    let args = ["find-generic-password", "-s", service, "-a", account, "-w"].map(String::from);
    let stdout = exec
        .exec_file("/usr/bin/security", &args, None)
        .await
        .ok()?;
    let password = stdout.trim();
    (!password.is_empty()).then(|| password.to_string())
}

#[cfg(test)]
mod tests {
    use super::super::test_exec::MockExecFile;
    use super::*;

    #[tokio::test]
    async fn reads_and_trims_a_generic_password() {
        let exec = MockExecFile::resolves("stored-password\n");
        assert_eq!(
            read_with(&exec, "service-name", "account-name").await,
            Some("stored-password".into())
        );
        assert_eq!(
            *exec.calls.lock(),
            vec![(
                "/usr/bin/security".to_string(),
                [
                    "find-generic-password",
                    "-s",
                    "service-name",
                    "-a",
                    "account-name",
                    "-w"
                ]
                .map(String::from)
                .to_vec(),
                None,
            )]
        );
    }

    #[tokio::test]
    async fn returns_none_for_empty_output() {
        let exec = MockExecFile::resolves("\n");
        assert_eq!(read_with(&exec, "service", "user").await, None);
    }

    #[tokio::test]
    async fn returns_none_when_the_command_fails() {
        let exec = MockExecFile::rejects("not found");
        assert_eq!(read_with(&exec, "service", "user").await, None);
    }
}
