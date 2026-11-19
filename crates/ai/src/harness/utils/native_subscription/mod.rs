//! Port of `utils/native-subscription/*`: reading native CLI subscription
//! credentials from the host.

pub mod jwt;
pub mod linux_secret_service;
pub mod macos_keychain;
pub mod should_resolve_native;
pub mod windows_credential_manager;

pub use jwt::*;
pub use linux_secret_service::*;
pub use macos_keychain::*;
pub use should_resolve_native::*;
pub use windows_credential_manager::*;

use async_trait::async_trait;
use indexmap::IndexMap;

use crate::harness::error::BoxError;
use crate::harness::utils::plain_error;

/// The options the readers pass to `execFile`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ExecFileOptions {
    /// Variables added on top of the inherited process environment.
    pub env: IndexMap<String, String>,
    pub windows_hide: bool,
}

/// `promisify(execFile)`: runs a program and resolves with its stdout, or
/// rejects when it cannot start or exits with a non-zero status.
///
/// [rust] A seam so the tests can mock the process, as the TS tests mock
/// `node:child_process`.
#[async_trait]
pub(crate) trait ExecFile: Send + Sync {
    async fn exec_file(
        &self,
        file: &str,
        args: &[String],
        options: Option<ExecFileOptions>,
    ) -> Result<String, BoxError>;
}

/// The real `execFile`.
pub(crate) struct SystemExecFile;

#[async_trait]
impl ExecFile for SystemExecFile {
    async fn exec_file(
        &self,
        file: &str,
        args: &[String],
        options: Option<ExecFileOptions>,
    ) -> Result<String, BoxError> {
        let mut command = tokio::process::Command::new(file);
        command
            .args(args)
            .stdin(std::process::Stdio::null())
            .kill_on_drop(true);
        if let Some(options) = options {
            command.envs(options.env);
            #[cfg(windows)]
            if options.windows_hide {
                const CREATE_NO_WINDOW: u32 = 0x0800_0000;
                command.creation_flags(CREATE_NO_WINDOW);
            }
        }
        let output = command.output().await?;
        if !output.status.success() {
            return Err(plain_error(format!("Command failed: {file}")));
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }
}

#[cfg(test)]
pub(crate) mod test_exec {
    use super::*;
    use parking_lot::Mutex;

    pub(crate) type Call = (String, Vec<String>, Option<ExecFileOptions>);

    /// `vi.fn()` standing in for `execFileAsync`.
    pub(crate) struct MockExecFile {
        pub result: Result<String, String>,
        pub calls: Mutex<Vec<Call>>,
    }

    impl MockExecFile {
        pub fn resolves(stdout: &str) -> Self {
            Self {
                result: Ok(stdout.to_string()),
                calls: Mutex::default(),
            }
        }

        pub fn rejects(message: &str) -> Self {
            Self {
                result: Err(message.to_string()),
                calls: Mutex::default(),
            }
        }
    }

    #[async_trait]
    impl ExecFile for MockExecFile {
        async fn exec_file(
            &self,
            file: &str,
            args: &[String],
            options: Option<ExecFileOptions>,
        ) -> Result<String, BoxError> {
            self.calls
                .lock()
                .push((file.to_string(), args.to_vec(), options));
            self.result.clone().map_err(plain_error)
        }
    }
}
