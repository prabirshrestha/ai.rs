//! Port of durable `test/tools.test.ts`: the tools run against a [`LocalExecutionEnv`] through a detached
//! [`ToolExecutionApi`] (TS `fakeApi`), so output and diagnostics are collected raw.
//!
//! Divergences from Pi:
//! - The TS environment subclasses (`SlowReadExecutionEnv`, `BlockingWriteExecutionEnv`, ...) are one [`TestEnv`]
//!   wrapper with optional behaviors.
//! - jsdiff's `applyPatch` is not available: the edit test compares the patch with jsdiff's output instead, and
//!   `jsdiff_goldens` checks patches and display diffs against jsdiff 8.0.4 outputs captured in
//!   `fixtures/jsdiff-goldens.json`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use indexmap::IndexMap;
use parking_lot::Mutex;
use serde_json::json;
use tokio::sync::watch;

use super::edit_diff::{generate_diff_string, generate_unified_patch};
use super::file_mutation_queue::with_file_mutation_queue;
use super::image::detect_supported_image_mime_type;
use super::*;
use crate::chord::context::{AbortController, with_abort_signal};
use crate::chord::{BACKGROUND_CONTEXT, Context};
use crate::durable::env::local::{LocalExecutionEnv, LocalExecutionEnvOptions};
use crate::durable::env::{
    CreateDirOptions, ExecutionEnv, ExecutionError, ExecutionErrorCode, FileError, FileInfo,
    FileSystem, ReadTextLinesOptions, RemoveOptions, Shell, ShellExecOptions, ShellExecResult,
    TempFileOptions, TextLineReader, get_or_throw,
};
use crate::durable::harness::tool::{DetachedCalls, ToolExecutionApi};
use crate::durable::harness::types::{ToolDiagnostic, ToolExecutionResult, ToolRegistration};
use crate::durable::storage::test_support::TempDir;
use crate::durable::truncate::DEFAULT_MAX_LINES;
use crate::types::UserContent;

fn ctx() -> &'static Context {
    &BACKGROUND_CONTEXT
}

#[derive(Clone)]
struct Deferred(Arc<watch::Sender<bool>>);

impl Deferred {
    fn new() -> Self {
        Self(Arc::new(watch::channel(false).0))
    }

    fn resolve(&self) {
        self.0.send_replace(true);
    }

    async fn wait(&self) {
        let mut receiver = self.0.subscribe();
        let _ = receiver.wait_for(|resolved| *resolved).await;
    }
}

/// `BlockingWriteExecutionEnv`'s state.
#[derive(Clone)]
struct BlockingWrite {
    first_write_started: Deferred,
    finish_first_write: Deferred,
    second_write_started: Arc<AtomicBool>,
}

impl BlockingWrite {
    fn new() -> Self {
        Self {
            first_write_started: Deferred::new(),
            finish_first_write: Deferred::new(),
            second_write_started: Arc::new(AtomicBool::new(false)),
        }
    }
}

/// `BlockingEditExecutionEnv`'s state.
#[derive(Clone)]
struct BlockingEdit {
    first_edit_write_started: Deferred,
    finish_first_edit_write: Deferred,
    first_edit_write_settled: Arc<AtomicBool>,
    second_edit_write_started: Arc<AtomicBool>,
}

const TRUNCATED_OUTPUT_LINES: usize = DEFAULT_MAX_LINES + 1;

/// A [`LocalExecutionEnv`] with the TS test subclasses' overrides.
struct TestEnv {
    inner: LocalExecutionEnv,
    id: Option<&'static str>,
    slow_read: bool,
    blocking_write: Option<BlockingWrite>,
    blocking_edit: Option<BlockingEdit>,
    timeout_output: bool,
}

impl TestEnv {
    fn new(cwd: &str) -> Self {
        Self {
            inner: LocalExecutionEnv::at(cwd),
            id: None,
            slow_read: false,
            blocking_write: None,
            blocking_edit: None,
            timeout_output: false,
        }
    }
}

#[async_trait]
impl FileSystem for TestEnv {
    fn id(&self) -> &str {
        self.id.unwrap_or(self.inner.id())
    }
    fn cwd(&self) -> String {
        self.inner.cwd()
    }
    fn set_cwd(&self, cwd: String) {
        self.inner.set_cwd(cwd)
    }
    async fn absolute_path(&self, path: &str, context: &Context) -> Result<String, FileError> {
        self.inner.absolute_path(path, context).await
    }
    async fn join_path(&self, parts: &[&str], context: &Context) -> Result<String, FileError> {
        self.inner.join_path(parts, context).await
    }
    async fn read_text_file(&self, path: &str, context: &Context) -> Result<String, FileError> {
        if self.slow_read {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        self.inner.read_text_file(path, context).await
    }
    async fn open_text_line_reader(
        &self,
        path: &str,
        context: &Context,
    ) -> Result<Box<dyn TextLineReader>, FileError> {
        self.inner.open_text_line_reader(path, context).await
    }
    async fn read_text_lines(
        &self,
        path: &str,
        options: ReadTextLinesOptions,
        context: &Context,
    ) -> Result<Vec<String>, FileError> {
        self.inner.read_text_lines(path, options, context).await
    }
    async fn read_binary_file(&self, path: &str, context: &Context) -> Result<Vec<u8>, FileError> {
        self.inner.read_binary_file(path, context).await
    }
    async fn write_file(
        &self,
        path: &str,
        content: &[u8],
        context: &Context,
    ) -> Result<(), FileError> {
        if let Some(blocking) = &self.blocking_write {
            if content == b"first\n" {
                blocking.first_write_started.resolve();
                blocking.finish_first_write.wait().await;
            } else if content == b"second\n" {
                blocking.second_write_started.store(true, Ordering::SeqCst);
            }
        }
        if let Some(blocking) = &self.blocking_edit {
            if content == b"ALPHA\nbeta\n" {
                blocking.first_edit_write_started.resolve();
                blocking.finish_first_edit_write.wait().await;
                let result = self.inner.write_file(path, content, ctx()).await;
                blocking
                    .first_edit_write_settled
                    .store(true, Ordering::SeqCst);
                return result;
            }
            if content == b"ALPHA\nBETA\n" || content == b"alpha\nBETA\n" {
                blocking
                    .second_edit_write_started
                    .store(true, Ordering::SeqCst);
            }
        }
        self.inner.write_file(path, content, context).await
    }
    async fn append_file(
        &self,
        path: &str,
        content: &[u8],
        context: &Context,
    ) -> Result<(), FileError> {
        self.inner.append_file(path, content, context).await
    }
    async fn truncate_file(
        &self,
        path: &str,
        size: u64,
        context: &Context,
    ) -> Result<(), FileError> {
        self.inner.truncate_file(path, size, context).await
    }
    async fn flush_file(&self, path: &str, context: &Context) -> Result<(), FileError> {
        self.inner.flush_file(path, context).await
    }
    async fn rename_file(
        &self,
        source_path: &str,
        destination_path: &str,
        context: &Context,
    ) -> Result<(), FileError> {
        self.inner
            .rename_file(source_path, destination_path, context)
            .await
    }
    async fn file_info(&self, path: &str, context: &Context) -> Result<FileInfo, FileError> {
        self.inner.file_info(path, context).await
    }
    async fn list_dir(&self, path: &str, context: &Context) -> Result<Vec<FileInfo>, FileError> {
        self.inner.list_dir(path, context).await
    }
    async fn canonical_path(&self, path: &str, context: &Context) -> Result<String, FileError> {
        self.inner.canonical_path(path, context).await
    }
    async fn exists(&self, path: &str, context: &Context) -> Result<bool, FileError> {
        self.inner.exists(path, context).await
    }
    async fn create_dir(
        &self,
        path: &str,
        options: CreateDirOptions,
        context: &Context,
    ) -> Result<(), FileError> {
        self.inner.create_dir(path, options, context).await
    }
    async fn remove(
        &self,
        path: &str,
        options: RemoveOptions,
        context: &Context,
    ) -> Result<(), FileError> {
        self.inner.remove(path, options, context).await
    }
    async fn create_temp_dir(
        &self,
        prefix: Option<&str>,
        context: &Context,
    ) -> Result<String, FileError> {
        self.inner.create_temp_dir(prefix, context).await
    }
    async fn create_temp_file(
        &self,
        options: TempFileOptions,
        context: &Context,
    ) -> Result<String, FileError> {
        self.inner.create_temp_file(options, context).await
    }
    async fn cleanup(&self, context: &Context) {
        FileSystem::cleanup(&self.inner, context).await
    }
}

#[async_trait]
impl Shell for TestEnv {
    async fn exec(
        &self,
        command: &str,
        options: ShellExecOptions,
        context: &Context,
    ) -> Result<ShellExecResult, ExecutionError> {
        if !self.timeout_output {
            return self.inner.exec(command, options, context).await;
        }
        // `TimeoutOutputExecutionEnv`: spill and stream more than the limits, then time out.
        let output = format!(
            "{}\n",
            (1..=TRUNCATED_OUTPUT_LINES)
                .map(|index| format!("line-{index}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
        let spill_path = self
            .create_temp_file(
                TempFileOptions {
                    prefix: Some("timeout-".into()),
                    suffix: Some(".log".into()),
                },
                context,
            )
            .await
            .unwrap();
        self.write_file(&spill_path, output.as_bytes(), context)
            .await
            .unwrap();
        if let Some(on_output) = &options.on_output {
            on_output(&output, context).unwrap();
        }
        let mut error = ExecutionError::new(
            ExecutionErrorCode::Timeout,
            format!("timeout:{:?}", options.timeout),
        );
        error.spill_path = Some(spill_path);
        Err(error)
    }
    async fn cleanup(&self, context: &Context) {
        Shell::cleanup(&self.inner, context).await
    }
}

fn temp_dir() -> (TempDir, String) {
    let dir = TempDir::new("pi-durable-tools-");
    let path = dir
        .join("")
        .to_string_lossy()
        .trim_end_matches('/')
        .to_owned();
    (dir, path)
}

fn create_env() -> (TempDir, Arc<LocalExecutionEnv>) {
    let (dir, path) = temp_dir();
    (dir, Arc::new(LocalExecutionEnv::at(path)))
}

type Ran = (crate::durable::Result<ToolExecutionResult>, DetachedCalls);

async fn run_in(
    tool: &ToolRegistration,
    args: serde_json::Value,
    env: Option<Arc<dyn ExecutionEnv>>,
    context: &Context,
) -> Ran {
    let (api, recorded) = ToolExecutionApi::detached("call", env);
    let result = (tool.execute)(args, api, context.clone()).await;
    let recorded = recorded.lock().clone();
    (result, recorded)
}

async fn run(tool: &ToolRegistration, args: serde_json::Value, env: Arc<dyn ExecutionEnv>) -> Ran {
    run_in(tool, args, Some(env), ctx()).await
}

/// Run on a spawned task, like an unawaited TS promise.
fn spawn_run(
    tool: &ToolRegistration,
    args: serde_json::Value,
    env: Arc<dyn ExecutionEnv>,
    context: Context,
) -> tokio::task::JoinHandle<Ran> {
    let tool = tool.clone();
    tokio::spawn(async move { run_in(&tool, args, Some(env), &context).await })
}

fn text_output(result: &ToolExecutionResult) -> String {
    result
        .content
        .iter()
        .flatten()
        .filter_map(|part| match part {
            UserContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn diagnostic_text(result: &ToolExecutionResult) -> String {
    result
        .diagnostics
        .iter()
        .flatten()
        .map(|diagnostic| diagnostic.message.clone())
        .collect::<Vec<_>>()
        .join("\n")
}

fn error_message(ran: &Ran) -> String {
    ran.0
        .as_ref()
        .expect_err("expected the tool to throw")
        .to_string()
}

async fn write(env: &dyn ExecutionEnv, path: &str, content: &str) {
    get_or_throw(env.write_file(path, content.as_bytes(), ctx()).await).unwrap();
}

async fn read(env: &dyn ExecutionEnv, path: &str) -> String {
    get_or_throw(env.read_text_file(path, ctx()).await).unwrap()
}

fn delay(ms: u64) -> tokio::time::Sleep {
    tokio::time::sleep(Duration::from_millis(ms))
}

fn full_output_path(diagnostics: &[ToolDiagnostic]) -> Option<String> {
    diagnostics
        .first()
        .and_then(|diagnostic| diagnostic.message.strip_prefix("Full output: "))
        .map(str::to_string)
}

#[tokio::test]
async fn fails_with_an_ordinary_error_when_no_environment_is_configured() {
    let ran = run_in(&create_read_tool(), json!({ "path": "x" }), None, ctx()).await;
    assert!(error_message(&ran).contains("No execution environment"));
}

// ---- read ----

#[test]
fn detects_the_complete_gif_signatures() {
    for signature in ["GIF87a", "GIF89a"] {
        assert_eq!(
            detect_supported_image_mime_type(signature.as_bytes()),
            Some("image/gif")
        );
    }
}

#[tokio::test]
async fn read_with_offsets_and_limits_reports_continuation_as_a_diagnostic() {
    let (_dir, env) = create_env();
    let text = (1..=100)
        .map(|index| format!("Line {index}"))
        .collect::<Vec<_>>()
        .join("\n");
    write(env.as_ref(), "test.txt", &text).await;
    let (result, _) = run(
        &create_read_tool(),
        json!({ "path": "test.txt", "offset": 41, "limit": 20 }),
        env,
    )
    .await;
    let result = result.unwrap();
    let output = text_output(&result);
    assert!(!output.contains("Line 40"));
    assert!(output.contains("Line 41"));
    assert!(output.contains("Line 60"));
    assert!(!output.contains("Line 61"));
    assert!(!output.contains("more lines"));
    assert_eq!(
        diagnostic_text(&result),
        "40 more lines in file. Use offset=61 to continue."
    );
}

#[tokio::test]
async fn read_truncates_large_text_by_line_count() {
    let (_dir, env) = create_env();
    let text = (1..=2500)
        .map(|index| format!("Line {index}"))
        .collect::<Vec<_>>()
        .join("\n");
    write(env.as_ref(), "large.txt", &text).await;
    let (result, _) = run(&create_read_tool(), json!({ "path": "large.txt" }), env).await;
    let result = result.unwrap();
    assert_eq!(
        diagnostic_text(&result),
        "Showing lines 1-2000 of 2500. Use offset=2001 to continue."
    );
    assert_eq!(
        result.diagnostics.as_ref().unwrap()[0].code.as_deref(),
        Some("truncated")
    );
    let truncation = &result.details.as_ref().unwrap()["truncation"];
    assert_eq!(truncation["truncated"], json!(true));
    assert_eq!(truncation["truncatedBy"], json!("lines"));
    assert_eq!(truncation["totalLines"], json!(2500));
    assert_eq!(truncation["outputLines"], json!(2000));
}

#[tokio::test]
async fn read_does_not_count_a_trailing_newline_as_an_extra_line_at_the_limit() {
    let (_dir, env) = create_env();
    let text = format!("{}\n", vec!["x"; 2000].join("\n"));
    write(env.as_ref(), "exact.txt", &text).await;
    let (result, _) = run(&create_read_tool(), json!({ "path": "exact.txt" }), env).await;
    let result = result.unwrap();
    assert_eq!(result.details, None);
    assert_eq!(result.diagnostics, Some(Vec::new()));
}

#[tokio::test]
async fn read_shows_the_start_of_a_line_longer_than_the_byte_limit() {
    let (_dir, env) = create_env();
    write(
        env.as_ref(),
        "long.txt",
        &format!("{}\nnext\n", "é".repeat(40_000)),
    )
    .await;
    let (result, _) = run(&create_read_tool(), json!({ "path": "long.txt" }), env).await;
    let result = result.unwrap();
    // Two-byte characters: the cut lands on a character boundary at or below the limit.
    assert_eq!(text_output(&result), "é".repeat(25_600));
    assert_eq!(
        diagnostic_text(&result),
        "Line 1 is 78.1KB, exceeds the 50.0KB limit; showing its first 50.0KB. Use bash: sed -n '1p' long.txt | tail -c +51201"
    );
    let truncation = &result.details.as_ref().unwrap()["truncation"];
    assert_eq!(truncation["truncated"], json!(true));
    assert_eq!(truncation["firstLineExceedsLimit"], json!(true));
    assert_eq!(truncation["outputBytes"], json!(51_200));
    assert_eq!(truncation["outputLines"], json!(1));
    assert!(truncation.get("content").is_none());
}

#[tokio::test]
async fn read_rejects_offsets_beyond_the_file() {
    let (_dir, env) = create_env();
    write(env.as_ref(), "short.txt", "one\ntwo\nthree").await;
    let ran = run(
        &create_read_tool(),
        json!({ "path": "short.txt", "offset": 100 }),
        env,
    )
    .await;
    assert!(error_message(&ran).contains("Offset 100 is beyond end of file (3 lines total)"));
}

#[tokio::test]
async fn read_reports_images_by_content_as_unsupported() {
    use base64::Engine;
    let (_dir, env) = create_env();
    let png = base64::engine::general_purpose::STANDARD
        .decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAX+XDSwAAAABJRU5ErkJggg==")
        .unwrap();
    get_or_throw(env.write_file("image.txt", &png, ctx()).await).unwrap();
    let (result, _) = run(&create_read_tool(), json!({ "path": "image.txt" }), env).await;
    let result = result.unwrap();
    assert_eq!(result.content, Some(Vec::new()));
    assert_eq!(result.is_error, Some(true));
    assert_eq!(
        diagnostic_text(&result),
        "image.txt is an image (image/png); reading images is not supported"
    );
}

// ---- write ----

#[tokio::test]
async fn write_writes_files_and_creates_parent_directories() {
    let (_dir, env) = create_env();
    let (result, _) = run(
        &create_write_tool(),
        json!({ "path": "nested/dir/file.txt", "content": "hello" }),
        env.clone(),
    )
    .await;
    assert_eq!(
        text_output(&result.unwrap()),
        "Successfully wrote to nested/dir/file.txt"
    );
    assert_eq!(read(env.as_ref(), "nested/dir/file.txt").await, "hello");
}

#[tokio::test]
async fn write_keeps_the_mutation_queue_locked_until_an_aborted_write_settles() {
    let (_dir, path) = temp_dir();
    let blocking = BlockingWrite::new();
    let env: Arc<dyn ExecutionEnv> = Arc::new(TestEnv {
        blocking_write: Some(blocking.clone()),
        ..TestEnv::new(&path)
    });
    let tool = create_write_tool();
    let controller = AbortController::new();
    let first_write = spawn_run(
        &tool,
        json!({ "path": "file.txt", "content": "first\n" }),
        env.clone(),
        with_abort_signal(controller.signal(), ctx()),
    );
    blocking.first_write_started.wait().await;
    controller.abort(None);
    let second_write = spawn_run(
        &tool,
        json!({ "path": "file.txt", "content": "second\n" }),
        env.clone(),
        ctx().clone(),
    );
    delay(20).await;
    assert!(!blocking.second_write_started.load(Ordering::SeqCst));
    blocking.finish_first_write.resolve();
    assert!(first_write.await.unwrap().0.is_err());
    second_write.await.unwrap().0.unwrap();
    assert_eq!(read(env.as_ref(), "file.txt").await, "second\n");
}

// ---- edit ----

#[tokio::test]
async fn edit_applies_disjoint_edits_and_returns_both_diff_formats() {
    let (_dir, env) = create_env();
    let original = "alpha\nbeta\ngamma\ndelta\n";
    write(env.as_ref(), "edit.txt", original).await;
    let (result, _) = run(
        &create_edit_tool(),
        json!({
            "path": "edit.txt",
            "edits": [
                { "oldText": "alpha\n", "newText": "ALPHA\n" },
                { "oldText": "gamma\n", "newText": "GAMMA\n" },
            ],
        }),
        env.clone(),
    )
    .await;
    let result = result.unwrap();
    let details = result.details.clone().unwrap();
    assert_eq!(
        text_output(&result),
        "Successfully replaced 2 block(s) in edit.txt."
    );
    assert!(details["diff"].as_str().unwrap().contains("ALPHA"));
    assert!(details["diff"].as_str().unwrap().contains("GAMMA"));
    // jsdiff's patch for this edit; TS checks `applyPatch(original, patch)`.
    assert_eq!(
        details["patch"],
        json!(
            "--- edit.txt\n+++ edit.txt\n@@ -1,4 +1,4 @@\n-alpha\n+ALPHA\n beta\n-gamma\n+GAMMA\n delta\n"
        )
    );
    assert_eq!(
        read(env.as_ref(), "edit.txt").await,
        "ALPHA\nbeta\nGAMMA\ndelta\n"
    );
}

#[test]
fn edit_repairs_edits_sent_as_a_json_string_a_single_object_or_top_level_old_text_new_text() {
    let prepare = create_edit_tool().prepare_arguments.unwrap();
    let edit = json!({ "oldText": "a", "newText": "b" });
    let as_string = json!({ "path": "f", "edits": serde_json::to_string(&json!([edit])).unwrap() });
    assert_eq!(
        prepare(as_string.clone()).unwrap(),
        json!({ "path": "f", "edits": [edit] })
    );
    assert_eq!(
        as_string["edits"],
        json!(serde_json::to_string(&json!([edit])).unwrap())
    );
    assert_eq!(
        prepare(json!({ "path": "f", "edits": serde_json::to_string(&edit).unwrap() })).unwrap(),
        json!({ "path": "f", "edits": [edit] })
    );
    assert_eq!(
        prepare(json!({ "path": "f", "edits": edit })).unwrap(),
        json!({ "path": "f", "edits": [edit] })
    );
    assert_eq!(
        prepare(json!({ "path": "f", "edits": [edit], "oldText": "c", "newText": "d" })).unwrap(),
        json!({ "path": "f", "edits": [edit, { "oldText": "c", "newText": "d" }] })
    );
    assert_eq!(
        prepare(json!({ "path": "f", "edits": "not json" })).unwrap(),
        json!({ "path": "f", "edits": "not json" })
    );
}

#[tokio::test]
async fn edit_matches_all_edits_against_the_original_and_rejects_overlaps() {
    let (_dir, env) = create_env();
    write(env.as_ref(), "edit.txt", "one\ntwo\nthree\n").await;
    let ran = run(
        &create_edit_tool(),
        json!({
            "path": "edit.txt",
            "edits": [
                { "oldText": "one\ntwo\n", "newText": "ONE\nTWO\n" },
                { "oldText": "two\nthree\n", "newText": "TWO\nTHREE\n" },
            ],
        }),
        env.clone(),
    )
    .await;
    assert!(error_message(&ran).contains("overlap"));
    assert_eq!(read(env.as_ref(), "edit.txt").await, "one\ntwo\nthree\n");
}

#[tokio::test]
async fn edit_rejects_missing_and_duplicate_target_text() {
    let (_dir, env) = create_env();
    write(env.as_ref(), "edit.txt", "foo foo foo").await;
    let tool = create_edit_tool();
    let missing = run(
        &tool,
        json!({ "path": "edit.txt", "edits": [{ "oldText": "bar", "newText": "baz" }] }),
        env.clone(),
    )
    .await;
    assert!(error_message(&missing).contains("Could not find the exact text"));
    let duplicate = run(
        &tool,
        json!({ "path": "edit.txt", "edits": [{ "oldText": "foo", "newText": "bar" }] }),
        env,
    )
    .await;
    assert!(error_message(&duplicate).contains("Found 3 occurrences"));
}

#[tokio::test]
async fn edit_keeps_the_mutation_queue_locked_until_an_aborted_edit_write_settles() {
    let (_dir, path) = temp_dir();
    let blocking = BlockingEdit {
        first_edit_write_started: Deferred::new(),
        finish_first_edit_write: Deferred::new(),
        first_edit_write_settled: Arc::new(AtomicBool::new(false)),
        second_edit_write_started: Arc::new(AtomicBool::new(false)),
    };
    let env: Arc<dyn ExecutionEnv> = Arc::new(TestEnv {
        blocking_edit: Some(blocking.clone()),
        ..TestEnv::new(&path)
    });
    write(env.as_ref(), "file.txt", "alpha\nbeta\n").await;
    let tool = create_edit_tool();
    let controller = AbortController::new();
    let first_edit = spawn_run(
        &tool,
        json!({ "path": "file.txt", "edits": [{ "oldText": "alpha", "newText": "ALPHA" }] }),
        env.clone(),
        with_abort_signal(controller.signal(), ctx()),
    );
    blocking.first_edit_write_started.wait().await;
    controller.abort(None);
    let second_edit = spawn_run(
        &tool,
        json!({ "path": "file.txt", "edits": [{ "oldText": "beta", "newText": "BETA" }] }),
        env.clone(),
        ctx().clone(),
    );
    delay(20).await;
    assert!(!blocking.second_edit_write_started.load(Ordering::SeqCst));
    blocking.finish_first_edit_write.resolve();
    assert_eq!(
        error_message(&first_edit.await.unwrap()),
        "Operation aborted"
    );
    second_edit.await.unwrap().0.unwrap();
    assert!(blocking.first_edit_write_settled.load(Ordering::SeqCst));
    assert_eq!(read(env.as_ref(), "file.txt").await, "ALPHA\nBETA\n");
}

#[tokio::test]
async fn edit_serializes_concurrent_edits_through_canonical_and_symlink_paths() {
    let (_dir, path) = temp_dir();
    let env: Arc<dyn ExecutionEnv> = Arc::new(TestEnv {
        slow_read: true,
        ..TestEnv::new(&path)
    });
    write(env.as_ref(), "target.txt", "alpha\nbeta\ngamma\n").await;
    std::os::unix::fs::symlink("target.txt", format!("{path}/link.txt")).unwrap();
    let tool = create_edit_tool();
    let (first, second) = tokio::join!(
        run(
            &tool,
            json!({ "path": "target.txt", "edits": [{ "oldText": "alpha", "newText": "ALPHA" }] }),
            env.clone(),
        ),
        run(
            &tool,
            json!({ "path": "link.txt", "edits": [{ "oldText": "beta", "newText": "BETA" }] }),
            env.clone(),
        ),
    );
    first.0.unwrap();
    second.0.unwrap();
    assert_eq!(
        read(env.as_ref(), "target.txt").await,
        "ALPHA\nBETA\ngamma\n"
    );
}

#[tokio::test]
async fn edit_serializes_edits_of_one_file_across_environment_objects_of_one_file_system() {
    let (_dir, path) = temp_dir();
    let first: Arc<dyn ExecutionEnv> = Arc::new(TestEnv {
        slow_read: true,
        ..TestEnv::new(&path)
    });
    let second: Arc<dyn ExecutionEnv> = Arc::new(TestEnv {
        slow_read: true,
        ..TestEnv::new(&path)
    });
    write(first.as_ref(), "file.txt", "alpha\nbeta\n").await;
    let tool = create_edit_tool();
    let (a, b) = tokio::join!(
        run(
            &tool,
            json!({ "path": "file.txt", "edits": [{ "oldText": "alpha", "newText": "ALPHA" }] }),
            first.clone(),
        ),
        run(
            &tool,
            json!({ "path": "file.txt", "edits": [{ "oldText": "beta", "newText": "BETA" }] }),
            second,
        ),
    );
    a.0.unwrap();
    b.0.unwrap();
    assert_eq!(read(first.as_ref(), "file.txt").await, "ALPHA\nBETA\n");
}

#[tokio::test]
async fn edit_serializes_a_new_file_created_through_a_symlinked_directory_with_its_canonical_path()
{
    let (_dir, path) = temp_dir();
    let blocking = BlockingWrite::new();
    let env: Arc<dyn ExecutionEnv> = Arc::new(TestEnv {
        blocking_write: Some(blocking.clone()),
        ..TestEnv::new(&path)
    });
    std::fs::create_dir(format!("{path}/real")).unwrap();
    std::os::unix::fs::symlink(format!("{path}/real"), format!("{path}/link")).unwrap();
    let tool = create_write_tool();
    let first = spawn_run(
        &tool,
        json!({ "path": "link/new.txt", "content": "first\n" }),
        env.clone(),
        ctx().clone(),
    );
    blocking.first_write_started.wait().await;
    let second = spawn_run(
        &tool,
        json!({ "path": "real/new.txt", "content": "second\n" }),
        env.clone(),
        ctx().clone(),
    );
    delay(20).await;
    assert!(!blocking.second_write_started.load(Ordering::SeqCst));
    blocking.finish_first_write.resolve();
    first.await.unwrap().0.unwrap();
    second.await.unwrap().0.unwrap();
    assert_eq!(read(env.as_ref(), "real/new.txt").await, "second\n");
}

#[tokio::test]
async fn edit_keys_a_missing_file_whose_name_contains_a_backslash_like_the_created_file() {
    let (_dir, env) = create_env();
    let created = Deferred::new();
    let release = Deferred::new();
    let first = {
        let (env, created, release) = (env.clone(), created.clone(), release.clone());
        tokio::spawn(async move {
            with_file_mutation_queue(
                env.as_ref(),
                "a\\b.txt",
                || async {
                    write(env.as_ref(), "a\\b.txt", "first\n").await;
                    created.resolve();
                    release.wait().await;
                    Ok(())
                },
                ctx(),
            )
            .await
        })
    };
    created.wait().await;
    let entered = Arc::new(AtomicBool::new(false));
    let second = {
        let (env, entered) = (env.clone(), entered.clone());
        tokio::spawn(async move {
            with_file_mutation_queue(
                env.as_ref(),
                "a\\b.txt",
                || async {
                    entered.store(true, Ordering::SeqCst);
                    Ok(())
                },
                ctx(),
            )
            .await
        })
    };
    delay(20).await;
    assert!(!entered.load(Ordering::SeqCst));
    release.resolve();
    first.await.unwrap().unwrap();
    second.await.unwrap().unwrap();
    assert!(entered.load(Ordering::SeqCst));
}

#[tokio::test]
async fn edit_does_not_serialize_the_same_path_on_different_file_systems() {
    let (_dir, path) = temp_dir();
    let local_blocking = BlockingWrite::new();
    let other_blocking = BlockingWrite::new();
    let local: Arc<dyn ExecutionEnv> = Arc::new(TestEnv {
        blocking_write: Some(local_blocking.clone()),
        ..TestEnv::new(&path)
    });
    let other: Arc<dyn ExecutionEnv> = Arc::new(TestEnv {
        id: Some("other"),
        blocking_write: Some(other_blocking.clone()),
        ..TestEnv::new(&path)
    });
    let tool = create_write_tool();
    let blocked = spawn_run(
        &tool,
        json!({ "path": "file.txt", "content": "first\n" }),
        local,
        ctx().clone(),
    );
    local_blocking.first_write_started.wait().await;
    run(
        &tool,
        json!({ "path": "file.txt", "content": "second\n" }),
        other,
    )
    .await
    .0
    .unwrap();
    assert!(other_blocking.second_write_started.load(Ordering::SeqCst));
    local_blocking.finish_first_write.resolve();
    blocked.await.unwrap().0.unwrap();
}

#[tokio::test]
async fn edit_edits_regular_files_through_symlinks() {
    let (_dir, env) = create_env();
    write(env.as_ref(), "target.txt", "before\n").await;
    std::os::unix::fs::symlink("target.txt", format!("{}/link.txt", env.cwd())).unwrap();
    run(
        &create_edit_tool(),
        json!({ "path": "link.txt", "edits": [{ "oldText": "before", "newText": "after" }] }),
        env.clone(),
    )
    .await
    .0
    .unwrap();
    assert_eq!(read(env.as_ref(), "target.txt").await, "after\n");
}

#[tokio::test]
async fn edit_preserves_bom_and_crlf_line_endings() {
    let (_dir, env) = create_env();
    write(env.as_ref(), "edit.txt", "\u{FEFF}one\r\ntwo\r\n").await;
    run(
        &create_edit_tool(),
        json!({ "path": "edit.txt", "edits": [{ "oldText": "two", "newText": "TWO" }] }),
        env.clone(),
    )
    .await
    .0
    .unwrap();
    assert_eq!(
        read(env.as_ref(), "edit.txt").await,
        "\u{FEFF}one\r\nTWO\r\n"
    );
}

#[test]
fn jsdiff_goldens() {
    let goldens: Vec<serde_json::Value> =
        serde_json::from_str(include_str!("fixtures/jsdiff-goldens.json")).unwrap();
    for golden in goldens {
        let (a, b) = (golden["a"].as_str().unwrap(), golden["b"].as_str().unwrap());
        assert_eq!(
            generate_unified_patch("f.txt", a, b, None),
            golden["patch"].as_str().unwrap(),
            "patch of {a:?} -> {b:?}"
        );
        let diff = generate_diff_string(a, b, None);
        assert_eq!(
            diff.diff,
            golden["diff"]["diff"].as_str().unwrap(),
            "diff of {a:?} -> {b:?}"
        );
        assert_eq!(
            diff.first_changed_line.map(|line| json!(line)),
            golden["diff"].get("firstChangedLine").cloned()
        );
    }
}

// ---- bash ----

#[tokio::test]
async fn bash_streams_combined_stdout_and_stderr_and_returns_no_content_of_its_own() {
    let (_dir, env) = create_env();
    let (result, recorded) = run(
        &create_bash_tool(BashToolOptions::default()),
        json!({ "command": "printf out; printf err >&2" }),
        env,
    )
    .await;
    let output = recorded.output.concat();
    assert!(output.contains("out"));
    assert!(output.contains("err"));
    assert_eq!(result.unwrap().content, None);
}

#[tokio::test]
async fn bash_throws_on_nonzero_exits_and_timeouts_after_streaming_the_output() {
    let (_dir, env) = create_env();
    let tool = create_bash_tool(BashToolOptions::default());
    let failed = run(
        &tool,
        json!({ "command": "printf failed; exit 7" }),
        env.clone(),
    )
    .await;
    assert_eq!(error_message(&failed), "Command exited with code 7");
    assert_eq!(failed.1.output.concat(), "failed");
    let slow = run(&tool, json!({ "command": "sleep 2", "timeout": 0.01 }), env).await;
    assert_eq!(error_message(&slow), "Command timed out after 0.01 seconds");
}

#[tokio::test]
async fn bash_reports_the_spill_of_a_command_that_times_out() {
    let (_dir, path) = temp_dir();
    let env: Arc<dyn ExecutionEnv> = Arc::new(TestEnv {
        timeout_output: true,
        ..TestEnv::new(&path)
    });
    let failed = run(
        &create_bash_tool(BashToolOptions::default()),
        json!({ "command": "emit-output-then-time-out", "timeout": 0.05 }),
        env.clone(),
    )
    .await;
    assert_eq!(
        error_message(&failed),
        "Command timed out after 0.05 seconds"
    );
    let full_output_path = full_output_path(&failed.1.diagnostics).unwrap();
    let full_output = read(env.as_ref(), &full_output_path).await;
    assert!(full_output.contains("line-1\nline-2"));
    assert!(full_output.contains(&format!(
        "line-{DEFAULT_MAX_LINES}\nline-{TRUNCATED_OUTPUT_LINES}"
    )));
    let _ = std::fs::remove_file(full_output_path);
}

#[tokio::test]
async fn bash_prepares_command_cwd_and_an_explicit_environment_with_the_calls_api() {
    let (_dir, path) = temp_dir();
    let env = Arc::new(LocalExecutionEnv::new(LocalExecutionEnvOptions {
        cwd: path.clone(),
        shell_env: Some(IndexMap::from([(
            "PI_BASH_PREPARE_INHERITED".to_string(),
            "inherited".to_string(),
        )])),
        ..Default::default()
    }));
    get_or_throw(
        env.create_dir("workspace", CreateDirOptions::default(), ctx())
            .await,
    )
    .unwrap();
    let workspace = format!("{path}/workspace");
    let controller = AbortController::new();
    let received: Arc<Mutex<Option<(bool, bool)>>> = Arc::new(Mutex::new(None));
    let expected_env: Arc<dyn ExecutionEnv> = env.clone();
    let tool = create_bash_tool(BashToolOptions {
        command_prefix: Some("prefix=ready".into()),
        prepare: Some(Arc::new({
            let received = received.clone();
            let workspace = workspace.clone();
            let expected_env = expected_env.clone();
            let signal = controller.signal().clone();
            move |mut execution, api, call_context| {
                let same_env = api
                    .env()
                    .is_some_and(|env| Arc::ptr_eq(&env, &expected_env));
                let same_signal = call_context
                    .abort_signal()
                    .is_some_and(|received| received.ptr_eq(&signal));
                *received.lock() = Some((same_env, same_signal));
                execution.cwd = workspace.clone();
                execution.env = IndexMap::from([(
                    "PI_BASH_PREPARE_EXPLICIT".to_string(),
                    "explicit".to_string(),
                )]);
                execution.inherit_env = false;
                execution.command.push_str(
                    "\nprintf '%s:%s:%s:%s' \"$prefix\" \"${PI_BASH_PREPARE_INHERITED-}\" \"$PI_BASH_PREPARE_EXPLICIT\" \"$PWD\"",
                );
                Box::pin(async move { Ok(execution) })
            }
        })),
    });
    let (result, recorded) = run_in(
        &tool,
        json!({ "command": ":" }),
        Some(expected_env.clone()),
        &with_abort_signal(controller.signal(), ctx()),
    )
    .await;
    result.unwrap();
    assert_eq!(*received.lock(), Some((true, true)));
    assert_eq!(
        recorded.output.concat(),
        format!(
            "ready::explicit:{}",
            get_or_throw(env.canonical_path(&workspace, ctx()).await).unwrap()
        )
    );
}

#[tokio::test]
async fn bash_supports_command_prefixes() {
    let (_dir, env) = create_env();
    let (result, recorded) = run(
        &create_bash_tool(BashToolOptions {
            command_prefix: Some("value=hello".into()),
            ..BashToolOptions::default()
        }),
        json!({ "command": "printf $value" }),
        env,
    )
    .await;
    result.unwrap();
    assert_eq!(recorded.output.concat(), "hello");
}

#[tokio::test]
async fn bash_streams_every_byte_and_spills_complete_output_beyond_the_default_limits() {
    let (_dir, env) = create_env();
    let (result, recorded) = run(
        &create_bash_tool(BashToolOptions::default()),
        json!({ "command": "i=1; while [ $i -le 3000 ]; do echo line-$i; i=$((i + 1)); done" }),
        env.clone(),
    )
    .await;
    result.unwrap();
    let expected: String = (1..=3000).map(|index| format!("line-{index}\n")).collect();
    assert_eq!(recorded.output.concat(), expected);
    let full_output_path = full_output_path(&recorded.diagnostics).unwrap();
    assert_eq!(read(env.as_ref(), &full_output_path).await, expected);
    let _ = std::fs::remove_file(full_output_path);
}

#[tokio::test]
async fn bash_does_not_spill_output_within_the_limits() {
    let (_dir, env) = create_env();
    let (result, recorded) = run(
        &create_bash_tool(BashToolOptions::default()),
        json!({ "command": "printf small" }),
        env,
    )
    .await;
    result.unwrap();
    assert_eq!(recorded.diagnostics, Vec::new());
}
