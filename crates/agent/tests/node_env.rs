use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent::{
    ExecOptions, ExecutionError, ExecutionErrorCode, FileErrorCode, FileKind, NodeExecutionEnv,
    ShellCaptureOptions, execute_shell_with_capture, sanitize_binary_output,
};

fn temp_root(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "ai-rs-pi-env-{label}-{}",
        agent::create_session_id()
    ))
}

#[tokio::test]
async fn env_reads_writes_lists_and_removes_files() {
    let root = temp_root("files");
    let env = NodeExecutionEnv::new(root.clone());

    assert_eq!(
        env.absolute_path("nested/child").unwrap(),
        root.join("nested/child")
    );
    assert_eq!(
        env.join_path(&[
            root.as_path(),
            std::path::Path::new("nested"),
            std::path::Path::new("child")
        ])
        .unwrap(),
        root.join("nested/child")
    );
    env.create_dir("nested/child", true).await.unwrap();
    env.write_file("nested/child/file.txt", "hel")
        .await
        .unwrap();
    env.append_file("nested/child/file.txt", "lo")
        .await
        .unwrap();
    assert_eq!(
        env.read_text_file("nested/child/file.txt").await.unwrap(),
        "hello"
    );
    assert_eq!(
        env.read_text_lines("nested/child/file.txt", Some(1))
            .await
            .unwrap(),
        ["hello"]
    );
    assert_eq!(
        String::from_utf8(env.read_binary_file("nested/child/file.txt").await.unwrap()).unwrap(),
        "hello"
    );

    let entries = env.list_dir("nested/child").await.unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].name, "file.txt");
    assert_eq!(entries[0].kind, FileKind::File);
    assert_eq!(entries[0].size, 5);
    assert!(entries[0].mtime_ms > 0.0);

    assert!(env.exists("nested/child/file.txt").await.unwrap());
    env.remove("nested/child/file.txt", false, false)
        .await
        .unwrap();
    assert!(!env.exists("nested/child/file.txt").await.unwrap());
    let _ = tokio::fs::remove_dir_all(root).await;
}

#[tokio::test]
async fn env_reports_symlink_metadata_without_following() {
    let root = temp_root("symlink");
    let env = NodeExecutionEnv::new(root.clone());
    env.write_file("dir/file.txt", "hello").await.unwrap();
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(root.join("dir/file.txt"), root.join("file-link")).unwrap();
        std::os::unix::fs::symlink(root.join("dir"), root.join("dir-link")).unwrap();
    }
    #[cfg(windows)]
    {
        std::os::windows::fs::symlink_file(root.join("dir/file.txt"), root.join("file-link"))
            .unwrap();
        std::os::windows::fs::symlink_dir(root.join("dir"), root.join("dir-link")).unwrap();
    }

    assert_eq!(
        env.file_info("dir").await.unwrap().kind,
        FileKind::Directory
    );
    assert_eq!(
        env.file_info("dir/file.txt").await.unwrap().kind,
        FileKind::File
    );
    assert_eq!(
        env.file_info("file-link").await.unwrap().kind,
        FileKind::Symlink
    );
    assert_eq!(
        env.file_info("dir-link").await.unwrap().kind,
        FileKind::Symlink
    );
    assert_eq!(
        env.canonical_path("file-link").await.unwrap(),
        tokio::fs::canonicalize(root.join("dir/file.txt"))
            .await
            .unwrap()
    );
    let _ = tokio::fs::remove_dir_all(root).await;
}

#[tokio::test]
async fn env_returns_typed_file_errors() {
    let root = temp_root("errors");
    let env = NodeExecutionEnv::new(root.clone());
    let err = env.file_info("missing.txt").await.unwrap_err();
    assert_eq!(err.code, FileErrorCode::NotFound);
    assert_eq!(
        err.path.as_deref(),
        Some(root.join("missing.txt").to_string_lossy().as_ref())
    );
    assert!(!env.exists("missing.txt").await.unwrap());

    env.write_file("file.txt", "hello").await.unwrap();
    let err = env.list_dir("file.txt").await.unwrap_err();
    assert!(matches!(
        err.code,
        FileErrorCode::NotDirectory | FileErrorCode::Unknown
    ));
    let _ = tokio::fs::remove_dir_all(root).await;
}

#[tokio::test]
async fn env_executes_commands_with_env_and_callbacks() {
    let root = temp_root("exec");
    tokio::fs::create_dir_all(&root).await.unwrap();
    let env = NodeExecutionEnv::new(root.clone());
    let mut options = ExecOptions::default();
    options.env = HashMap::from([("NODE_ENV_TEST".to_string(), "ok".to_string())]);
    let output = env
        .exec("printf '%s:%s' \"$PWD\" \"$NODE_ENV_TEST\"", options)
        .await
        .unwrap();
    let canonical_root = tokio::fs::canonicalize(&root).await.unwrap();
    assert_eq!(
        output.stdout,
        format!("{}:ok", canonical_root.to_string_lossy())
    );
    assert_eq!(output.stderr, "");
    assert_eq!(output.exit_code, 0);

    let stdout = Arc::new(Mutex::new(String::new()));
    let stderr = Arc::new(Mutex::new(String::new()));
    let mut options = ExecOptions::default();
    options.on_stdout = Some(Arc::new({
        let stdout = stdout.clone();
        move |chunk| {
            stdout.lock().unwrap().push_str(chunk);
            Ok(())
        }
    }));
    options.on_stderr = Some(Arc::new({
        let stderr = stderr.clone();
        move |chunk| {
            stderr.lock().unwrap().push_str(chunk);
            Ok(())
        }
    }));
    let output = env
        .exec("printf out; printf err >&2", options)
        .await
        .unwrap();
    assert_eq!(output.stdout, "out");
    assert_eq!(output.stderr, "err");
    assert_eq!(stdout.lock().unwrap().as_str(), "out");
    assert_eq!(stderr.lock().unwrap().as_str(), "err");
    let _ = tokio::fs::remove_dir_all(root).await;
}

#[tokio::test]
async fn env_exec_reports_timeout_callback_and_shell_errors() {
    let root = temp_root("exec-errors");
    tokio::fs::create_dir_all(&root).await.unwrap();
    let env = NodeExecutionEnv::new(root.clone());
    let mut options = ExecOptions::default();
    options.timeout = Some(Duration::from_millis(10));
    let err = env.exec("sleep 5", options).await.unwrap_err();
    assert_eq!(err.code, ExecutionErrorCode::Timeout);

    let mut options = ExecOptions::default();
    options.on_stdout = Some(Arc::new(|_| {
        Err(ExecutionError::new(
            ExecutionErrorCode::CallbackError,
            "callback failed",
        ))
    }));
    let err = env.exec("printf out", options).await.unwrap_err();
    assert_eq!(err.code, ExecutionErrorCode::CallbackError);
    assert_eq!(err.message(), "callback failed");

    let missing_shell =
        NodeExecutionEnv::new(root.clone()).with_shell_path(root.join("missing-shell"));
    let err = missing_shell
        .exec("printf ok", ExecOptions::default())
        .await
        .unwrap_err();
    assert_eq!(err.code, ExecutionErrorCode::ShellUnavailable);
    let _ = tokio::fs::remove_dir_all(root).await;
}

#[tokio::test]
async fn shell_capture_sanitizes_and_writes_large_output_file() {
    assert_eq!(sanitize_binary_output("a\u{0}b\tc\n"), "ab\tc\n");

    let root = temp_root("capture");
    tokio::fs::create_dir_all(&root).await.unwrap();
    let env = NodeExecutionEnv::new(root.clone());
    let result = execute_shell_with_capture(
        &env,
        "yes line | head -n 15000",
        ShellCaptureOptions::default(),
    )
    .await
    .unwrap();

    assert!(result.truncated);
    let full_output_path = result.full_output_path.expect("full output path");
    let full_output = tokio::fs::read_to_string(full_output_path).await.unwrap();
    assert!(full_output.split('\n').count() > 10_000);
    assert!(result.output.len() < full_output.len());
    let _ = tokio::fs::remove_dir_all(root).await;
}
