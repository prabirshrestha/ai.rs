//! Port of `utils/native-subscription/windows-credential-manager.ts`.

use indexmap::IndexMap;

use super::{ExecFile, ExecFileOptions, SystemExecFile};

const CREDENTIAL_MANAGER_SOURCE: &str = r#"
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class AISDKCredentialManager {
  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
  public struct Credential {
    public UInt32 Flags; public UInt32 Type; public string TargetName;
    public string Comment; public System.Runtime.InteropServices.ComTypes.FILETIME LastWritten;
    public UInt32 CredentialBlobSize; public IntPtr CredentialBlob;
    public UInt32 Persist; public UInt32 AttributeCount; public IntPtr Attributes;
    public string TargetAlias; public string UserName;
  }
  [DllImport("advapi32.dll", EntryPoint = "CredReadW", CharSet = CharSet.Unicode, SetLastError = true)]
  static extern bool CredRead(string target, UInt32 type, UInt32 flags, out IntPtr credential);
  [DllImport("advapi32.dll", SetLastError = true)] static extern void CredFree(IntPtr credential);
  public static string Read(string target) {
    IntPtr pointer;
    if (!CredRead(target, 1, 0, out pointer)) return null;
    try {
      Credential value = Marshal.PtrToStructure<Credential>(pointer);
      byte[] bytes = new byte[value.CredentialBlobSize];
      Marshal.Copy(value.CredentialBlob, bytes, 0, bytes.Length);
      return Encoding.Unicode.GetString(bytes);
    } finally { CredFree(pointer); }
  }
}"#;

const COMMAND: &str = r#"Add-Type -TypeDefinition $env:AI_SDK_WINDOWS_CREDENTIAL_MANAGER_SOURCE; $value = [AISDKCredentialManager]::Read($env:AI_SDK_WINDOWS_CREDENTIAL_MANAGER_TARGET); if ($null -ne $value) { [Console]::Out.Write($value) }"#;

/// `readWindowsCredentialManagerPassword({ targetName })`: a generic
/// credential from the Windows Credential Manager via PowerShell, untrimmed,
/// or `None` when it is missing or empty.
pub async fn read_windows_credential_manager_password(target_name: &str) -> Option<String> {
    read_with(&SystemExecFile, target_name).await
}

async fn read_with(exec: &dyn ExecFile, target_name: &str) -> Option<String> {
    let args = ["-NoProfile", "-NonInteractive", "-Command", COMMAND].map(String::from);
    let options = ExecFileOptions {
        env: IndexMap::from([
            (
                "AI_SDK_WINDOWS_CREDENTIAL_MANAGER_SOURCE".to_string(),
                CREDENTIAL_MANAGER_SOURCE.to_string(),
            ),
            (
                "AI_SDK_WINDOWS_CREDENTIAL_MANAGER_TARGET".to_string(),
                target_name.to_string(),
            ),
        ]),
        windows_hide: true,
    };
    let stdout = exec
        .exec_file("powershell.exe", &args, Some(options))
        .await
        .ok()?;
    (!stdout.is_empty()).then_some(stdout)
}

#[cfg(test)]
mod tests {
    use super::super::test_exec::MockExecFile;
    use super::*;

    #[tokio::test]
    async fn reads_a_password_using_the_provided_target_name() {
        let exec = MockExecFile::resolves(" stored-password ");
        assert_eq!(
            read_with(&exec, "account.service").await,
            Some(" stored-password ".into())
        );
        let calls = exec.calls.lock();
        assert_eq!(calls.len(), 1);
        let (file, args, options) = &calls[0];
        assert_eq!(file, "powershell.exe");
        assert_eq!(&args[..3], &["-NoProfile", "-NonInteractive", "-Command"]);
        assert_eq!(args.len(), 4);
        assert!(args[3].contains(
            "[AISDKCredentialManager]::Read($env:AI_SDK_WINDOWS_CREDENTIAL_MANAGER_TARGET)"
        ));
        let options = options.as_ref().unwrap();
        assert!(options.windows_hide);
        assert_eq!(options.env.len(), 2);
        assert!(options.env["AI_SDK_WINDOWS_CREDENTIAL_MANAGER_SOURCE"].contains("CredReadW"));
        assert_eq!(
            options.env["AI_SDK_WINDOWS_CREDENTIAL_MANAGER_TARGET"],
            "account.service"
        );
    }

    #[tokio::test]
    async fn returns_none_for_empty_output() {
        let exec = MockExecFile::resolves("");
        assert_eq!(read_with(&exec, "account.service").await, None);
    }

    #[tokio::test]
    async fn returns_none_when_the_command_fails() {
        let exec = MockExecFile::rejects("not found");
        assert_eq!(read_with(&exec, "account.service").await, None);
    }
}
