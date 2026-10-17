//! Port of `test/env-node.test.ts` and `test/env-node-spill.test.ts`.
//!
//! Not ported: the `FileHandle.prototype` spies (fsync call/close checks, the
//! gated read cancelled mid-flight, spill backpressure), the tests overriding
//! `process.platform` (WSL stdin transport, `taskkill`), and the Windows-only
//! inherited-stdio test. Commands that ran `node -e` use `printf`/`head`.

use std::sync::Arc;
use std::time::Duration;

use indexmap::IndexMap;
use parking_lot::Mutex;

use super::*;
use crate::chord::context::{AbortController, with_abort_signal};
use crate::durable::Error;
use crate::durable::env::ShellSpillOptions;
use crate::durable::storage::test_support::{TempDir, context};

fn temp_root() -> (TempDir, String) {
    let dir = TempDir::new("pi-durable-env-");
    let root = dir
        .join("")
        .to_string_lossy()
        .trim_end_matches('/')
        .to_owned();
    (dir, root)
}

fn aborted_context() -> Context {
    let controller = AbortController::new();
    controller.abort(None);
    with_abort_signal(controller.signal(), context())
}

fn realpath(path: &str) -> String {
    std::fs::canonicalize(path)
        .expect("realpath")
        .to_string_lossy()
        .into_owned()
}

fn remove_parent(path: &str) {
    let _ = std::fs::remove_dir_all(parent(path));
}

async fn collect_shell_output(
    env: &LocalExecutionEnv,
    command: &str,
    options: ShellExecOptions,
    context: &Context,
) -> (Result<ShellExecResult, ExecutionError>, String) {
    let output = Arc::new(Mutex::new(String::new()));
    let sink = output.clone();
    let result = env
        .exec(
            command,
            ShellExecOptions {
                on_output: Some(Arc::new(move |text: &str, _: &Context| {
                    sink.lock().push_str(text);
                    Ok(())
                })),
                ..options
            },
            context,
        )
        .await;
    let output = output.lock().clone();
    (result, output)
}

fn env_map(pairs: &[(&str, &str)]) -> IndexMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
        .collect()
}

fn spill(after_bytes: u64, after_lines: u64) -> Option<ShellSpillOptions> {
    Some(ShellSpillOptions {
        after_bytes,
        after_lines,
    })
}

// ─── filesystem ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn reads_writes_lists_and_removes_files_and_directories() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    let ctx = context();
    assert_eq!(
        env.absolute_path("nested/child", ctx).await.unwrap(),
        format!("{root}/nested/child")
    );
    assert_eq!(
        env.join_path(&[&root, "nested", "child"], ctx)
            .await
            .unwrap(),
        format!("{root}/nested/child")
    );
    env.create_dir("nested/child", CreateDirOptions::default(), ctx)
        .await
        .unwrap();
    env.write_file("nested/child/file.txt", b"hel", ctx)
        .await
        .unwrap();
    env.append_file("nested/child/file.txt", b"lo", ctx)
        .await
        .unwrap();
    assert_eq!(
        env.read_text_file("nested/child/file.txt", ctx)
            .await
            .unwrap(),
        "hello"
    );
    assert_eq!(
        env.read_text_lines(
            "nested/child/file.txt",
            ReadTextLinesOptions { max_lines: Some(1) },
            ctx
        )
        .await
        .unwrap(),
        vec!["hello"]
    );
    assert_eq!(
        env.read_binary_file("nested/child/file.txt", ctx)
            .await
            .unwrap(),
        b"hello"
    );

    let entries = env.list_dir("nested/child", ctx).await.unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].name, "file.txt");
    assert_eq!(entries[0].path, format!("{root}/nested/child/file.txt"));
    assert_eq!(entries[0].kind, FileKind::File);
    assert_eq!(entries[0].size, 5);
    assert!(entries[0].mtime_ms > 0.0);

    assert!(env.exists("nested/child/file.txt", ctx).await.unwrap());
    env.remove("nested/child/file.txt", RemoveOptions::default(), ctx)
        .await
        .unwrap();
    assert!(!env.exists("nested/child/file.txt", ctx).await.unwrap());
}

#[tokio::test]
async fn expands_home_relative_paths_and_file_urls() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    assert_eq!(
        env.absolute_path("~/pi-node-env-test", context())
            .await
            .unwrap(),
        join(&[&homedir(), "pi-node-env-test"])
    );
    let file_path = format!("{root}/file with spaces.txt");
    let url = format!("file://{}", file_path.replace(' ', "%20"));
    assert_eq!(env.absolute_path(&url, context()).await.unwrap(), file_path);
}

#[cfg(unix)]
#[tokio::test]
async fn returns_file_info_for_files_directories_and_symlinks_without_following_symlinks() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    let ctx = context();
    env.create_dir(
        "dir",
        CreateDirOptions {
            recursive: Some(true),
        },
        ctx,
    )
    .await
    .unwrap();
    env.write_file("dir/file.txt", b"hello", ctx).await.unwrap();
    std::os::unix::fs::symlink(format!("{root}/dir/file.txt"), format!("{root}/file-link"))
        .unwrap();
    std::os::unix::fs::symlink(format!("{root}/dir"), format!("{root}/dir-link")).unwrap();

    let dir = env.file_info("dir", ctx).await.unwrap();
    assert_eq!(
        (dir.name.as_str(), dir.path, dir.kind),
        ("dir", format!("{root}/dir"), FileKind::Directory)
    );
    let file = env.file_info("dir/file.txt", ctx).await.unwrap();
    assert_eq!(
        (file.name.as_str(), file.path, file.kind, file.size),
        (
            "file.txt",
            format!("{root}/dir/file.txt"),
            FileKind::File,
            5
        )
    );
    let file_link = env.file_info("file-link", ctx).await.unwrap();
    assert_eq!(
        (file_link.name.as_str(), file_link.path, file_link.kind),
        ("file-link", format!("{root}/file-link"), FileKind::Symlink)
    );
    let dir_link = env.file_info("dir-link", ctx).await.unwrap();
    assert_eq!(
        (dir_link.name.as_str(), dir_link.kind),
        ("dir-link", FileKind::Symlink)
    );
    assert_eq!(
        env.canonical_path("file-link", ctx).await.unwrap(),
        realpath(&format!("{root}/dir/file.txt"))
    );
}

#[cfg(unix)]
#[tokio::test]
async fn lists_symlinks_as_symlinks() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    env.write_file("target.txt", b"hello", context())
        .await
        .unwrap();
    std::os::unix::fs::symlink(format!("{root}/target.txt"), format!("{root}/link.txt")).unwrap();

    let mut entries: Vec<(String, FileKind)> = env
        .list_dir(".", context())
        .await
        .unwrap()
        .into_iter()
        .map(|entry| (entry.name, entry.kind))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        entries,
        vec![
            ("link.txt".into(), FileKind::Symlink),
            ("target.txt".into(), FileKind::File)
        ]
    );
}

#[tokio::test]
async fn stops_reading_text_lines_at_the_requested_limit() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    env.write_file("file.txt", b"one\ntwo\nthree", context())
        .await
        .unwrap();
    assert_eq!(
        env.read_text_lines(
            "file.txt",
            ReadTextLinesOptions { max_lines: Some(1) },
            context()
        )
        .await
        .unwrap(),
        vec!["one"]
    );
}

#[tokio::test]
async fn returns_file_error_for_missing_paths_and_keeps_exists_false() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    let error = env.file_info("missing.txt", context()).await.unwrap_err();
    assert_eq!(error.name(), "FileError");
    assert_eq!(error.code, FileErrorCode::NotFound);
    assert_eq!(error.path, Some(format!("{root}/missing.txt")));
    assert!(!env.exists("missing.txt", context()).await.unwrap());
}

#[tokio::test]
async fn returns_file_error_for_listing_non_directories() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    env.write_file("file.txt", b"hello", context())
        .await
        .unwrap();
    let error = env.list_dir("file.txt", context()).await.unwrap_err();
    assert_eq!(error.code, FileErrorCode::NotDirectory);
}

#[tokio::test]
async fn appends_to_new_files_and_creates_parent_directories() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    env.append_file("new/nested/file.txt", b"a", context())
        .await
        .unwrap();
    env.append_file("new/nested/file.txt", b"b", context())
        .await
        .unwrap();
    assert_eq!(
        env.read_text_file("new/nested/file.txt", context())
            .await
            .unwrap(),
        "ab"
    );
}

#[tokio::test]
async fn atomically_renames_a_file_and_replaces_the_destination() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    let ctx = context();
    env.write_file("source.txt", b"new", ctx).await.unwrap();
    env.write_file("destination.txt", b"old", ctx)
        .await
        .unwrap();
    env.rename_file("source.txt", "destination.txt", ctx)
        .await
        .unwrap();
    assert!(!env.exists("source.txt", ctx).await.unwrap());
    assert_eq!(
        env.read_text_file("destination.txt", ctx).await.unwrap(),
        "new"
    );
}

#[tokio::test]
async fn reports_the_source_path_when_rename_fails_because_the_source_is_missing() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    let ctx = context();
    env.write_file("destination.txt", b"unchanged", ctx)
        .await
        .unwrap();
    let error = env
        .rename_file("missing-source.txt", "destination.txt", ctx)
        .await
        .unwrap_err();
    assert_eq!(error.code, FileErrorCode::NotFound);
    assert_eq!(error.path, Some(format!("{root}/missing-source.txt")));
    assert_eq!(
        env.read_text_file("destination.txt", ctx).await.unwrap(),
        "unchanged"
    );
}

#[tokio::test]
async fn creates_temporary_directories_and_files() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    let ctx = context();
    let temp_dir = env
        .create_temp_dir(Some("node-env-test-"), ctx)
        .await
        .unwrap();
    assert!(std::path::Path::new(&temp_dir).is_dir());
    let _ = std::fs::remove_dir_all(&temp_dir);
    let temp_file = env
        .create_temp_file(
            TempFileOptions {
                prefix: Some("prefix-".into()),
                suffix: Some(".txt".into()),
            },
            ctx,
        )
        .await
        .unwrap();
    assert!(std::path::Path::new(&temp_file).is_file());
    assert!(temp_file.ends_with(".txt"));
    assert_eq!(env.read_text_file(&temp_file, ctx).await.unwrap(), "");
    remove_parent(&temp_file);
}

#[tokio::test]
async fn honors_create_dir_recursive_false_and_remove_recursive_force_options() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    let ctx = context();
    let error = env
        .create_dir(
            "missing/child",
            CreateDirOptions {
                recursive: Some(false),
            },
            ctx,
        )
        .await
        .unwrap_err();
    assert_eq!(error.code, FileErrorCode::NotFound);

    env.write_file("dir/child/file.txt", b"hello", ctx)
        .await
        .unwrap();
    assert!(
        env.remove(
            "dir",
            RemoveOptions {
                recursive: Some(false),
                force: None
            },
            ctx
        )
        .await
        .is_err()
    );
    env.remove(
        "dir",
        RemoveOptions {
            recursive: Some(true),
            force: None,
        },
        ctx,
    )
    .await
    .unwrap();
    assert!(!env.exists("dir", ctx).await.unwrap());

    assert!(
        env.remove(
            "missing",
            RemoveOptions {
                recursive: None,
                force: Some(false)
            },
            ctx
        )
        .await
        .is_err()
    );
    env.remove(
        "missing",
        RemoveOptions {
            recursive: None,
            force: Some(true),
        },
        ctx,
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn returns_aborted_results_without_side_effects_for_pre_aborted_file_operations() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    env.write_file("file.txt", b"hello", context())
        .await
        .unwrap();
    let ctx = &aborted_context();

    let codes = vec![
        env.read_text_file("file.txt", ctx).await.err(),
        env.read_text_lines("file.txt", ReadTextLinesOptions::default(), ctx)
            .await
            .err(),
        env.read_binary_file("file.txt", ctx).await.err(),
        env.open_text_line_reader("file.txt", ctx).await.err(),
        env.write_file("other.txt", b"hello", ctx).await.err(),
        env.append_file("file.txt", b" world", ctx).await.err(),
        env.truncate_file("file.txt", 1, ctx).await.err(),
        env.flush_file("file.txt", ctx).await.err(),
        env.rename_file("file.txt", "renamed.txt", ctx).await.err(),
        env.file_info("file.txt", ctx).await.err(),
        env.list_dir(".", ctx).await.err(),
        env.canonical_path("file.txt", ctx).await.err(),
        env.exists("file.txt", ctx).await.err(),
        env.create_dir("dir", CreateDirOptions::default(), ctx)
            .await
            .err(),
        env.remove("file.txt", RemoveOptions::default(), ctx)
            .await
            .err(),
        env.create_temp_dir(None, ctx).await.err(),
        env.create_temp_file(TempFileOptions::default(), ctx)
            .await
            .err(),
    ];
    for error in codes {
        assert_eq!(error.map(|error| error.code), Some(FileErrorCode::Aborted));
    }
    assert_eq!(
        env.read_text_file("file.txt", context()).await.unwrap(),
        "hello"
    );
    let names: Vec<String> = env
        .list_dir(".", context())
        .await
        .unwrap()
        .into_iter()
        .map(|entry| entry.name)
        .collect();
    assert_eq!(names, vec!["file.txt"]);
}

#[tokio::test]
async fn truncates_and_extends_files_to_exact_byte_sizes() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    let ctx = context();
    env.write_file("file.bin", &[0x61, 0xc3, 0xa9, 0x0a, 0x62, 0x0a], ctx)
        .await
        .unwrap();

    // Truncation is byte-exact even when the boundary splits a UTF-8 sequence.
    env.truncate_file("file.bin", 2, ctx).await.unwrap();
    assert_eq!(
        env.read_binary_file("file.bin", ctx).await.unwrap(),
        [0x61, 0xc3]
    );
    env.truncate_file("file.bin", 4, ctx).await.unwrap();
    assert_eq!(
        env.read_binary_file("file.bin", ctx).await.unwrap(),
        [0x61, 0xc3, 0, 0]
    );
    env.truncate_file("file.bin", 0, ctx).await.unwrap();
    assert_eq!(env.file_info("file.bin", ctx).await.unwrap().size, 0);
}

#[tokio::test]
async fn rejects_invalid_truncation_sizes_and_never_creates_missing_files() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    let ctx = context();
    env.write_file("file.txt", b"hello", ctx).await.unwrap();
    // Negative, fractional and non-finite sizes are unrepresentable as `u64`.
    let error = env
        .truncate_file("file.txt", MAX_SAFE_INTEGER + 1, ctx)
        .await
        .unwrap_err();
    assert_eq!(error.code, FileErrorCode::Invalid);
    assert_eq!(error.path, Some(format!("{root}/file.txt")));
    assert_eq!(env.read_text_file("file.txt", ctx).await.unwrap(), "hello");

    let error = env.truncate_file("missing.txt", 0, ctx).await.unwrap_err();
    assert_eq!(error.code, FileErrorCode::NotFound);
    assert_eq!(error.path, Some(format!("{root}/missing.txt")));
    assert!(!env.exists("missing.txt", ctx).await.unwrap());
}

#[tokio::test]
async fn flushes_existing_files_without_changing_content_and_reports_missing_paths() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    let ctx = context();
    env.write_file("file.txt", b"durable", ctx).await.unwrap();
    env.flush_file("file.txt", ctx).await.unwrap();
    assert_eq!(
        env.read_text_file("file.txt", ctx).await.unwrap(),
        "durable"
    );

    let error = env.flush_file("missing.txt", ctx).await.unwrap_err();
    assert_eq!(error.code, FileErrorCode::NotFound);
    assert_eq!(error.path, Some(format!("{root}/missing.txt")));
    assert!(!env.exists("missing.txt", ctx).await.unwrap());

    env.create_dir("dir", CreateDirOptions::default(), ctx)
        .await
        .unwrap();
    assert_eq!(
        env.flush_file("dir", ctx).await.unwrap_err().code,
        FileErrorCode::IsDirectory
    );
}

#[tokio::test]
async fn cleanup_is_best_effort() {
    let (_dir, root) = temp_root();
    LocalExecutionEnv::at(&root).cleanup(context()).await;
}

// ─── text line reader ────────────────────────────────────────────────────────

async fn read_all(reader: &mut Box<dyn TextLineReader>) -> Vec<TextLine> {
    let mut lines = Vec::new();
    while let Some(line) = reader.read_line(context()).await.unwrap() {
        lines.push(line);
    }
    lines
}

fn line(text: &str, terminated: bool) -> TextLine {
    TextLine {
        text: text.into(),
        terminated,
    }
}

#[tokio::test]
async fn reports_whether_each_line_was_newline_terminated_and_preserves_carriage_returns() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    env.write_file("lines.txt", b"one\r\n\ntwo\npartial", context())
        .await
        .unwrap();
    let mut reader = env
        .open_text_line_reader("lines.txt", context())
        .await
        .unwrap();
    assert_eq!(
        read_all(&mut reader).await,
        vec![
            line("one\r", true),
            line("", true),
            line("two", true),
            line("partial", false)
        ]
    );
    assert_eq!(reader.read_line(context()).await.unwrap(), None);
    reader.close(context()).await;
}

#[tokio::test]
async fn returns_no_lines_for_an_empty_file_and_one_terminated_empty_line_for_a_lone_newline() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    env.write_file("empty.txt", b"", context()).await.unwrap();
    env.write_file("newline.txt", b"\n", context())
        .await
        .unwrap();

    let mut empty = env
        .open_text_line_reader("empty.txt", context())
        .await
        .unwrap();
    assert_eq!(empty.read_line(context()).await.unwrap(), None);
    empty.close(context()).await;

    let mut newline = env
        .open_text_line_reader("newline.txt", context())
        .await
        .unwrap();
    assert_eq!(
        newline.read_line(context()).await.unwrap(),
        Some(line("", true))
    );
    assert_eq!(newline.read_line(context()).await.unwrap(), None);
    newline.close(context()).await;
}

#[tokio::test]
async fn decodes_multi_byte_characters_split_across_read_chunks_and_long_lines() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    // The reader uses 64 KiB chunks; place a four-byte character across the first boundary.
    let first = format!("{}😀tail", "a".repeat(64 * 1024 - 2));
    let second = "é".repeat(100_000);
    env.write_file(
        "large.txt",
        format!("{first}\n{second}").as_bytes(),
        context(),
    )
    .await
    .unwrap();
    let mut reader = env
        .open_text_line_reader("large.txt", context())
        .await
        .unwrap();
    assert_eq!(
        reader.read_line(context()).await.unwrap(),
        Some(line(&first, true))
    );
    assert_eq!(
        reader.read_line(context()).await.unwrap(),
        Some(line(&second, false))
    );
    assert_eq!(reader.read_line(context()).await.unwrap(), None);
    reader.close(context()).await;
}

#[tokio::test]
async fn rejects_an_aborted_read_without_consuming_its_bytes() {
    // Stands in for the gated-read test: the abort check after a read keeps the offset.
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    env.write_file("lines.txt", b"one\ntwo\n", context())
        .await
        .unwrap();
    let mut reader = env
        .open_text_line_reader("lines.txt", context())
        .await
        .unwrap();
    let error = reader.read_line(&aborted_context()).await.unwrap_err();
    assert_eq!(error.code, FileErrorCode::Aborted);
    assert_eq!(
        read_all(&mut reader).await,
        vec![line("one", true), line("two", true)]
    );
    reader.close(context()).await;
}

#[tokio::test]
async fn rejects_reads_after_close_and_closes_idempotently() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    env.write_file("lines.txt", b"one\n", context())
        .await
        .unwrap();
    let mut reader = env
        .open_text_line_reader("lines.txt", context())
        .await
        .unwrap();
    reader.close(context()).await;
    reader.close(context()).await;
    let error = reader.read_line(context()).await.unwrap_err();
    assert_eq!(error.code, FileErrorCode::Invalid);
    assert_eq!(error.path, Some(format!("{root}/lines.txt")));
}

#[tokio::test]
async fn reports_missing_files_when_opening_a_reader() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    let error = env
        .open_text_line_reader("missing.txt", context())
        .await
        .err()
        .unwrap();
    assert_eq!(error.code, FileErrorCode::NotFound);
    assert_eq!(error.path, Some(format!("{root}/missing.txt")));
}

// ─── shell ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn executes_commands_in_cwd_with_env_overrides() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    let (result, output) = collect_shell_output(
        &env,
        r#"printf '%s:%s' "$PWD" "$NODE_ENV_TEST""#,
        ShellExecOptions {
            env: Some(env_map(&[("NODE_ENV_TEST", "ok")])),
            ..Default::default()
        },
        context(),
    )
    .await;
    assert_eq!(output, format!("{}:ok", realpath(&root)));
    assert_eq!(result.unwrap().exit_code, 0);
}

#[tokio::test]
async fn applies_string_shell_environment_overrides() {
    for (overrides, expected) in [
        (None, "x:/stale/parent.jsonl"),
        (Some(env_map(&[("PI_SESSION_FILE", "")])), "x:"),
        (
            Some(env_map(&[("PI_SESSION_FILE", "/sessions/current.jsonl")])),
            "x:/sessions/current.jsonl",
        ),
    ] {
        let (_dir, root) = temp_root();
        let env = LocalExecutionEnv::new(LocalExecutionEnvOptions {
            cwd: root,
            shell_path: None,
            shell_env: Some(env_map(&[
                ("PI_SESSION_FILE", "/stale/parent.jsonl"),
                ("PI_CODING_AGENT", "true"),
                ("PI_NODE_ENV_PRESERVED_TEST", "preserved"),
            ])),
        });
        let (result, output) = collect_shell_output(
            &env,
            r#"printf '%s:%s|%s|%s' "${PI_SESSION_FILE+x}" "${PI_SESSION_FILE-}" "$PI_CODING_AGENT" "$PI_NODE_ENV_PRESERVED_TEST""#,
            ShellExecOptions {
                env: overrides,
                ..Default::default()
            },
            context(),
        )
        .await;
        result.unwrap();
        assert_eq!(output, format!("{expected}|true|preserved"));
    }
}

#[tokio::test]
async fn can_replace_rather_than_inherit_the_default_shell_environment() {
    // Inherited from the test process: every process has PATH.
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::new(LocalExecutionEnvOptions {
        cwd: root,
        shell_path: None,
        shell_env: Some(env_map(&[("PI_NODE_ENV_CONFIGURED_TEST", "configured")])),
    });
    let (result, output) = collect_shell_output(
        &env,
        r#"printf '%s:%s:%s' "${HOME-}" "${PI_NODE_ENV_CONFIGURED_TEST-}" "${PI_NODE_ENV_EXPLICIT_TEST-}""#,
        ShellExecOptions {
            inherit_env: Some(false),
            env: Some(env_map(&[("PI_NODE_ENV_EXPLICIT_TEST", "explicit")])),
            ..Default::default()
        },
        context(),
    )
    .await;
    result.unwrap();
    assert_eq!(output, "::explicit");
}

#[test]
fn recognizes_legacy_wsl_bash_paths() {
    // Stands in for the stdin transport test, which overrides `process.platform`.
    assert!(get_bash_shell_config("C:\\Windows\\System32\\bash.exe").command_from_stdin);
    assert!(get_bash_shell_config("c:/windows/sysnative/bash.exe").command_from_stdin);
    assert!(!get_bash_shell_config("/bin/bash").command_from_stdin);
}

#[tokio::test]
async fn cleanup_terminates_active_shell_processes() {
    let (_dir, root) = temp_root();
    let env = Arc::new(LocalExecutionEnv::at(&root));
    let execution = tokio::spawn({
        let env = env.clone();
        async move {
            env.exec(
                "touch started; sleep 60",
                ShellExecOptions::default(),
                context(),
            )
            .await
        }
    });
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while !env.exists("started", context()).await.unwrap() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "command did not start"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    env.cleanup(context()).await;
    let result = tokio::time::timeout(Duration::from_secs(3), execution)
        .await
        .expect("settles after cleanup")
        .unwrap();
    assert!(result.is_ok());
}

#[tokio::test]
async fn streams_combined_stdout_and_stderr() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    let (result, output) = collect_shell_output(
        &env,
        "printf out; printf err >&2",
        ShellExecOptions::default(),
        context(),
    )
    .await;
    assert_eq!(
        result.unwrap(),
        ShellExecResult {
            exit_code: 0,
            spill_path: None
        }
    );
    assert!(output.contains("out"));
    assert!(output.contains("err"));
}

#[tokio::test]
async fn decodes_utf8_split_across_raw_process_chunks() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    let (_, output) = collect_shell_output(
        &env,
        r"printf '\360\237'; sleep 0.05; printf '\230\200'",
        ShellExecOptions::default(),
        context(),
    )
    .await;
    assert_eq!(output, "😀");
}

#[tokio::test]
async fn reports_a_missing_working_directory_before_spawning() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(format!("{root}/missing"));
    let error = env
        .exec("printf ok", ShellExecOptions::default(), context())
        .await
        .unwrap_err();
    assert_eq!(error.code, ExecutionErrorCode::SpawnError);
    assert!(error.message.contains("Working directory does not exist"));
}

#[tokio::test]
async fn returns_non_zero_command_exit_codes_as_successful_execution_results() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    let result = env
        .exec("exit 7", ShellExecOptions::default(), context())
        .await
        .unwrap();
    assert_eq!(
        result,
        ShellExecResult {
            exit_code: 7,
            spill_path: None
        }
    );
}

#[cfg(unix)]
#[tokio::test]
async fn maps_signal_killed_processes_to_a_non_zero_exit_code() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    let result = env
        .exec("kill -9 $$", ShellExecOptions::default(), context())
        .await
        .unwrap();
    assert_eq!(result.exit_code, 128 + 9);
}

#[tokio::test]
async fn returns_timeout_errors_for_commands_exceeding_the_timeout() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    let error = env
        .exec(
            "sleep 5",
            ShellExecOptions {
                timeout: Some(0.01),
                ..Default::default()
            },
            context(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.code, ExecutionErrorCode::Timeout);
    assert_eq!(error.message, "timeout:0.01");
}

#[tokio::test]
async fn rejects_invalid_timeouts_before_spawning() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    for timeout in [0.0, -1.0, f64::NAN, f64::INFINITY, 2_147_484.0] {
        let error = env
            .exec(
                "touch spawned",
                ShellExecOptions {
                    timeout: Some(timeout),
                    ..Default::default()
                },
                context(),
            )
            .await
            .unwrap_err();
        assert_eq!(error.code, ExecutionErrorCode::Timeout);
        assert!(error.message.contains("Invalid timeout"));
    }
    assert!(!env.exists("spawned", context()).await.unwrap());
}

#[tokio::test]
async fn returns_callback_errors_from_exec_stream_handlers() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    let error = env
        .exec(
            "printf out",
            ShellExecOptions {
                on_output: Some(Arc::new(|_: &str, _: &Context| {
                    Err(Error::message("callback failed"))
                })),
                ..Default::default()
            },
            context(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.code, ExecutionErrorCode::CallbackError);
    assert_eq!(error.message, "callback failed");
}

#[cfg(unix)]
#[tokio::test]
async fn returns_shell_unavailable_and_spawn_errors() {
    let (_dir, root) = temp_root();
    let missing_shell_env = LocalExecutionEnv::new(LocalExecutionEnvOptions {
        cwd: root.clone(),
        shell_path: Some(format!("{root}/missing-shell")),
        shell_env: None,
    });
    let error = missing_shell_env
        .exec("printf ok", ShellExecOptions::default(), context())
        .await
        .unwrap_err();
    assert_eq!(error.code, ExecutionErrorCode::ShellUnavailable);

    let shell_path = format!("{root}/not-executable-shell");
    LocalExecutionEnv::at(&root)
        .write_file(&shell_path, b"not executable", context())
        .await
        .unwrap();
    let spawn_error_env = LocalExecutionEnv::new(LocalExecutionEnvOptions {
        cwd: root,
        shell_path: Some(shell_path),
        shell_env: None,
    });
    let error = spawn_error_env
        .exec("printf ok", ShellExecOptions::default(), context())
        .await
        .unwrap_err();
    assert_eq!(error.code, ExecutionErrorCode::SpawnError);
}

#[tokio::test]
async fn returns_an_aborted_result_for_pre_aborted_and_aborted_commands() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    let error = env
        .exec(
            "touch spawned",
            ShellExecOptions::default(),
            &aborted_context(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.code, ExecutionErrorCode::Aborted);
    assert!(!env.exists("spawned", context()).await.unwrap());

    let controller = AbortController::new();
    let ctx = with_abort_signal(controller.signal(), context());
    let execution = env.exec("sleep 5", ShellExecOptions::default(), &ctx);
    let abort = async {
        tokio::time::sleep(Duration::from_millis(20)).await;
        controller.abort(None);
    };
    let (result, ()) = tokio::join!(execution, abort);
    assert_eq!(result.unwrap_err().code, ExecutionErrorCode::Aborted);
}

#[tokio::test]
async fn does_not_create_a_spill_before_output_crosses_its_thresholds() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    let result = env
        .exec(
            "printf short",
            ShellExecOptions {
                spill: spill(100, 10),
                ..Default::default()
            },
            context(),
        )
        .await
        .unwrap();
    assert_eq!(result.spill_path, None);
}

#[tokio::test]
async fn preserves_exact_raw_bytes_in_the_spill_while_streaming_decoded_text() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    let (result, output) = collect_shell_output(
        &env,
        r"printf 'f\200\000o'",
        ShellExecOptions {
            spill: spill(1, 10),
            ..Default::default()
        },
        context(),
    )
    .await;
    let spill_path = result.unwrap().spill_path.expect("spilled");
    assert_eq!(output, "f\u{fffd}\0o");
    assert_eq!(
        env.read_binary_file(&spill_path, context()).await.unwrap(),
        [0x66, 0x80, 0x00, 0x6f]
    );
    remove_parent(&spill_path);
}

#[tokio::test]
async fn reports_the_spill_of_a_command_that_times_out() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    let error = env
        .exec(
            "printf 12345678901234567890; sleep 5",
            ShellExecOptions {
                timeout: Some(0.3),
                spill: spill(10, 10),
                ..Default::default()
            },
            context(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.code, ExecutionErrorCode::Timeout);
    let spill_path = error.spill_path.expect("spill path");
    assert_eq!(
        env.read_text_file(&spill_path, context()).await.unwrap(),
        "12345678901234567890"
    );
    remove_parent(&spill_path);
}

#[tokio::test]
async fn fails_rather_than_silently_losing_a_requested_spill() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    *env.spill_path_override.lock() = Some(format!("{root}/missing/spill.log"));
    let error = env
        .exec(
            "printf 12345678901234567890",
            ShellExecOptions {
                spill: spill(10, 10),
                ..Default::default()
            },
            context(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.code, ExecutionErrorCode::Unknown);
    assert!(
        error
            .message
            .contains("Failed to preserve complete shell output")
    );
}

#[tokio::test]
async fn preserves_complete_large_output_in_the_spill() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    let size = 500_000;
    let result = env
        .exec(
            &format!("head -c {size} /dev/zero | tr '\\0' x"),
            ShellExecOptions {
                spill: spill(10, 10),
                ..Default::default()
            },
            context(),
        )
        .await
        .unwrap();
    let spill_path = result.spill_path.expect("spilled");
    assert_eq!(
        env.read_text_file(&spill_path, context())
            .await
            .unwrap()
            .len(),
        size
    );
    remove_parent(&spill_path);
}

#[tokio::test]
async fn streams_every_line_and_spills_them_all_once_output_crosses_its_line_threshold() {
    let (_dir, root) = temp_root();
    let env = LocalExecutionEnv::at(&root);
    let (result, output) = collect_shell_output(
        &env,
        "i=1; while [ $i -le 15000 ]; do echo line-$i; i=$((i+1)); done",
        ShellExecOptions {
            spill: spill(1024 * 1024, 100),
            ..Default::default()
        },
        context(),
    )
    .await;
    let expected: String = (1..=15000).map(|index| format!("line-{index}\n")).collect();
    assert_eq!(output, expected);
    let spill_path = result.unwrap().spill_path.expect("spilled");
    let spilled = env
        .read_text_lines(&spill_path, ReadTextLinesOptions::default(), context())
        .await
        .unwrap();
    assert_eq!(spilled.len(), 15000);
    assert_eq!(spilled.last().map(String::as_str), Some("line-15000"));
    remove_parent(&spill_path);
}
