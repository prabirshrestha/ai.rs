//! Port of `utils/native-subscription/linux-secret-service.ts`.

use indexmap::IndexMap;

use super::{ExecFile, SystemExecFile};

/// `readLinuxSecretServicePassword({ attributes })`: a password from the
/// Secret Service via `secret-tool lookup`, untrimmed, or `None` when it is
/// missing or empty.
pub async fn read_linux_secret_service_password(
    attributes: &IndexMap<String, String>,
) -> Option<String> {
    read_with(&SystemExecFile, attributes).await
}

async fn read_with(exec: &dyn ExecFile, attributes: &IndexMap<String, String>) -> Option<String> {
    let args: Vec<String> = std::iter::once("lookup".to_string())
        .chain(
            attributes
                .iter()
                .flat_map(|(attribute, value)| [attribute.clone(), value.clone()]),
        )
        .collect();
    let stdout = exec.exec_file("secret-tool", &args, None).await.ok()?;
    (!stdout.is_empty()).then_some(stdout)
}

#[cfg(test)]
mod tests {
    use super::super::test_exec::MockExecFile;
    use super::*;
    use crate::harness::utils::ai_gateway_auth::env as attributes;

    #[tokio::test]
    async fn reads_a_password_using_the_provided_attributes() {
        let exec = MockExecFile::resolves(" stored-password ");
        assert_eq!(
            read_with(
                &exec,
                &attributes(&[
                    ("service", "service-name"),
                    ("username", "account-name"),
                    ("target", "default"),
                ])
            )
            .await,
            Some(" stored-password ".into())
        );
        assert_eq!(
            *exec.calls.lock(),
            vec![(
                "secret-tool".to_string(),
                [
                    "lookup",
                    "service",
                    "service-name",
                    "username",
                    "account-name",
                    "target",
                    "default",
                ]
                .map(String::from)
                .to_vec(),
                None,
            )]
        );
    }

    #[tokio::test]
    async fn returns_none_for_empty_output() {
        let exec = MockExecFile::resolves("");
        let attrs = attributes(&[("service", "service"), ("username", "user")]);
        assert_eq!(read_with(&exec, &attrs).await, None);
    }

    #[tokio::test]
    async fn returns_none_when_the_command_fails() {
        let exec = MockExecFile::rejects("not found");
        let attrs = attributes(&[("service", "service"), ("username", "user")]);
        assert_eq!(read_with(&exec, &attrs).await, None);
    }
}
