//! Port of `harness-pi/src/pi-remote-ops.test.ts`.
//!
//! [rust] The `just-bash` and native-shell cases run against a local `bash`
//! sandbox rooted in a temporary directory.

use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine as _;
use bytes::Bytes;
use parking_lot::Mutex;

use super::*;
use crate::harness::adapters::pi::pi_paths::{
    CreatePiPathMapperOptions, PiFileToolPathPolicy, PiReadablePathRoot, create_pi_path_mapper,
};
use crate::harness::adapters::pi::pi_test_support::{LocalSandbox, canonical, is_root};
use crate::harness::adapters::test_temp_dir::mkdtemp;
use crate::harness::sandbox::*;

const SANDBOX_WORK_DIR: &str = "/sandbox/workspace";

type RunBehavior = Arc<dyn Fn(&str) -> Option<(String, String, i32)> + Send + Sync>;
type RealpathBehavior = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;
type RealpathBytesBehavior = Arc<dyn Fn(&str) -> Option<Vec<u8>> + Send + Sync>;
type ReadBehavior = Arc<dyn Fn(&str) -> Option<Vec<u8>> + Send + Sync>;

#[derive(Default, Clone)]
struct Behaviors {
    run: Option<RunBehavior>,
    realpath: Option<RealpathBehavior>,
    realpath_bytes: Option<RealpathBytesBehavior>,
    read_binary: Option<ReadBehavior>,
    /// Replaces every `run` result (`vi.mocked(sandbox.run).mockResolvedValue`).
    fixed_run: Option<(String, String, i32)>,
}

#[derive(Default)]
struct Calls {
    runs: Vec<(String, Option<String>)>,
    reads: Vec<String>,
    writes: Vec<(String, String)>,
}

#[derive(Clone)]
struct MockSandbox {
    behaviors: Behaviors,
    calls: Arc<Mutex<Calls>>,
}

fn mock_realpath_command(
    command: &str,
    resolve_path: Option<&RealpathBehavior>,
    resolve_bytes: Option<&RealpathBytesBehavior>,
) -> Option<(String, String, i32)> {
    if !command.contains("realpath") {
        return None;
    }
    let targets: Vec<String> = regex::Regex::new(r"target='([^']+)'")
        .unwrap()
        .captures_iter(command)
        .map(|capture| capture[1].to_string())
        .collect();
    if targets.is_empty() {
        return None;
    }
    let frame = regex::Regex::new(r"realpath_marker='([^']+)'")
        .unwrap()
        .captures(command)
        .map(|capture| capture[1].to_string());
    let mut stdout = String::new();
    for target in targets {
        let resolved_path = match resolve_path {
            Some(resolve) => resolve(&target),
            None => Some(target.clone()),
        };
        let Some(resolved_path) = resolved_path else {
            return Some((stdout + "__PI_REALPATH_NOT_FOUND__\n", String::new(), 2));
        };
        let resolved_bytes = resolve_bytes
            .and_then(|resolve| resolve(&target))
            .unwrap_or_else(|| resolved_path.clone().into_bytes());
        let output_path = if command.contains("base64") {
            base64::engine::general_purpose::STANDARD.encode(&resolved_bytes)
        } else {
            String::from_utf8_lossy(&resolved_bytes).into_owned()
        };
        match &frame {
            None => stdout += &format!("{resolved_path}\n"),
            Some(frame) => stdout += &format!("{output_path}{frame}"),
        }
    }
    Some((stdout, String::new(), 0))
}

#[async_trait]
impl SandboxSession for MockSandbox {
    fn description(&self) -> String {
        "mock".into()
    }

    async fn read_file(&self, _o: SandboxReadFileOptions) -> HarnessResult<Option<ByteStream>> {
        unimplemented!()
    }

    async fn read_binary_file(
        &self,
        o: SandboxReadBinaryFileOptions,
    ) -> HarnessResult<Option<Bytes>> {
        self.calls.lock().reads.push(o.path.clone());
        Ok(self
            .behaviors
            .read_binary
            .as_ref()
            .and_then(|read| read(&o.path))
            .map(Bytes::from))
    }

    async fn read_text_file(&self, _o: SandboxReadTextFileOptions) -> HarnessResult<Option<String>> {
        unimplemented!()
    }

    async fn write_file(&self, _o: SandboxWriteFileOptions) -> HarnessResult<()> {
        unimplemented!()
    }

    async fn write_binary_file(&self, _o: SandboxWriteBinaryFileOptions) -> HarnessResult<()> {
        unimplemented!()
    }

    async fn write_text_file(&self, o: SandboxWriteTextFileOptions) -> HarnessResult<()> {
        self.calls.lock().writes.push((o.path, o.content));
        Ok(())
    }

    async fn stat(&self, _o: SandboxStatOptions) -> HarnessResult<Option<SandboxFileStat>> {
        unimplemented!()
    }

    async fn list_directory(
        &self,
        _o: SandboxListDirectoryOptions,
    ) -> HarnessResult<Option<Vec<SandboxDirectoryEntry>>> {
        unimplemented!()
    }

    async fn delete_file(&self, _o: SandboxDeleteFileOptions) -> HarnessResult<bool> {
        unimplemented!()
    }

    async fn spawn(&self, _o: SandboxSpawnOptions) -> HarnessResult<SandboxProcess> {
        unimplemented!()
    }

    async fn run(&self, o: SandboxRunOptions) -> HarnessResult<SandboxRunResult> {
        let command = o.process.command.clone();
        self.calls
            .lock()
            .runs
            .push((command.clone(), o.process.working_directory.clone()));
        let (stdout, stderr, exit_code) = self.behaviors.fixed_run.clone().unwrap_or_else(|| {
            mock_realpath_command(
                &command,
                self.behaviors.realpath.as_ref(),
                self.behaviors.realpath_bytes.as_ref(),
            )
            .or_else(|| self.behaviors.run.as_ref().and_then(|run| run(&command)))
            .unwrap_or_default()
        });
        Ok(SandboxRunResult {
            exit_code,
            stdout,
            stderr,
            ..Default::default()
        })
    }
}

fn make_mock_sandbox(behaviors: Behaviors) -> MockSandbox {
    MockSandbox {
        behaviors,
        calls: Arc::default(),
    }
}

fn ops_for(
    sandbox: Arc<dyn SandboxSession>,
    paths: PiPathMapper,
    on_file_change: Option<PiRemoteFileChangeHandler>,
) -> PiRemoteOps {
    create_pi_remote_ops(PiRemoteOpsOptions {
        sandbox,
        paths,
        env: None,
        on_file_change,
    })
}

fn mapper(sandbox_work_dir: &str) -> PiPathMapper {
    create_pi_path_mapper(CreatePiPathMapperOptions {
        sandbox_work_dir: sandbox_work_dir.into(),
        ..Default::default()
    })
    .unwrap()
}

fn make_ops(behaviors: Behaviors) -> (MockSandbox, PiRemoteOps) {
    let sandbox = make_mock_sandbox(behaviors);
    let paths = create_pi_path_mapper(CreatePiPathMapperOptions {
        sandbox_work_dir: SANDBOX_WORK_DIR.into(),
        readable_roots: Some(vec![PiReadablePathRoot {
            sandbox_dir: "/home/vercel-sandbox/.agents/skills".into(),
        }]),
        ..Default::default()
    })
    .unwrap();
    let ops = ops_for(Arc::new(sandbox.clone()), paths, None);
    (sandbox, ops)
}

fn run(stdout: &str, exit_code: i32) -> Option<RunBehavior> {
    let stdout = stdout.to_string();
    Some(Arc::new(move |_| Some((stdout.clone(), String::new(), exit_code))))
}

fn read_map(entries: Vec<(String, Vec<u8>)>) -> Option<ReadBehavior> {
    Some(Arc::new(move |path| {
        entries
            .iter()
            .find(|(candidate, _)| candidate == path)
            .map(|(_, bytes)| bytes.clone())
    }))
}

fn alias_realpath(alias: String, target: String) -> Option<RealpathBehavior> {
    Some(Arc::new(move |path| {
        Some(if path == alias { target.clone() } else { path.into() })
    }))
}

fn err_text<T: std::fmt::Debug>(result: HarnessResult<T>) -> String {
    result.unwrap_err().to_string()
}

fn bash(sandbox: &LocalSandbox, command: &str) {
    let status = std::process::Command::new("bash")
        .arg("-c")
        .arg(command)
        .current_dir(sandbox.cwd.as_deref().unwrap())
        .status()
        .unwrap();
    assert!(status.success(), "setup failed: {command}");
}

/// A local workspace standing in for the TS `just-bash` sandbox.
struct LocalWorkspace {
    _root: crate::harness::adapters::test_temp_dir::TempDir,
    root: String,
    workspace: String,
    sandbox: LocalSandbox,
}

fn local_workspace() -> LocalWorkspace {
    let temp = mkdtemp("pi-remote-ops-");
    let root = canonical(temp.path());
    let workspace = format!("{root}/workspace");
    std::fs::create_dir_all(&workspace).unwrap();
    LocalWorkspace {
        _root: temp,
        root,
        sandbox: LocalSandbox::new(workspace.clone()),
        workspace,
    }
}

fn local_ops(local: &LocalWorkspace) -> PiRemoteOps {
    ops_for(Arc::new(local.sandbox.clone()), mapper(&local.workspace), None)
}

fn read(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

fn write(path: &str, content: &str) {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, content).unwrap();
}

mod with_local_bash {
    use super::*;

    #[tokio::test]
    async fn supports_file_operations_with_the_seeded_realpath_command() {
        let local = local_workspace();
        write(&format!("{}/notes.md", local.workspace), "alpha\n");
        let ops = local_ops(&local);
        assert_eq!(ops.read_buffer("notes.md").await.unwrap(), b"alpha\n");
        ops.write_file("created.md", "created\n").await.unwrap();
        assert_eq!(
            ops.edit_file("notes.md", "alpha", "beta").await.unwrap(),
            "beta\n"
        );
        assert_eq!(
            ops.find_files("*.md", Some("."), None).await.unwrap(),
            vec!["created.md", "notes.md"]
        );
    }

    #[tokio::test]
    async fn resolves_chained_workspace_symlinks_for_file_operations() {
        let local = local_workspace();
        let w = &local.workspace;
        bash(
            &local.sandbox,
            &[
                format!("mkdir -p {w}/target/docs"),
                format!("printf 'read content\\n' > {w}/target/docs/read.txt"),
                format!("printf 'old content\\n' > {w}/target/docs/edit.txt"),
                format!("printf 'search needle\\n' > {w}/target/docs/search.txt"),
                format!("ln -s target {w}/intermediate"),
                format!("ln -s intermediate/docs {w}/linked-docs"),
            ]
            .join(" && "),
        );
        let ops = local_ops(&local);

        assert_eq!(
            ops.read_buffer("linked-docs/read.txt").await.unwrap(),
            b"read content\n"
        );
        ops.write_file("linked-docs/written.txt", "written content\n")
            .await
            .unwrap();
        assert_eq!(
            read(&format!("{w}/target/docs/written.txt")).as_deref(),
            Some("written content\n")
        );
        assert_eq!(
            ops.edit_file("linked-docs/edit.txt", "old", "updated")
                .await
                .unwrap(),
            "updated content\n"
        );
        assert_eq!(
            ops.list_directory(Some("linked-docs"), None).await.unwrap(),
            vec!["edit.txt", "read.txt", "search.txt", "written.txt"]
        );
        assert_eq!(
            ops.find_files("*.txt", Some("linked-docs"), None)
                .await
                .unwrap(),
            vec!["edit.txt", "read.txt", "search.txt", "written.txt"]
        );
        let output = ops
            .grep_files(
                "needle",
                PiGrepInput {
                    path: Some("linked-docs".into()),
                    literal: true,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(output.contains("search.txt:1:search needle"), "{output}");
        let runs = local.sandbox.calls.lock().runs.clone();
        let grep_commands: Vec<_> = runs
            .iter()
            .filter(|(command, _)| command.contains("grep "))
            .collect();
        assert_eq!(grep_commands.len(), 1);
        assert!(grep_commands[0].0.contains("target/docs"));
        assert!(!grep_commands[0].0.contains("linked-docs"));
    }

    async fn assert_canonical_names(read_name: &str, write_name: &str) {
        let local = local_workspace();
        let w = &local.workspace;
        let outside = format!("{}/outside", local.root);
        let outside_read = format!("{outside}/read-secret.txt");
        let outside_write = format!("{outside}/write-target.txt");
        let inside_read = format!("{w}/{read_name}");
        let inside_write = format!("{w}/{write_name}");
        write(&outside_read, "outside read\n");
        write(&outside_write, "outside write\n");
        write(&inside_read, "inside read\n");
        write(&inside_write, "inside write\n");
        let trimmed_read = read_name.trim_end();
        let trimmed_write = write_name.trim_end();
        let mut setup = Vec::new();
        if trimmed_read != read_name {
            setup.push(format!(
                "ln -s {} {}",
                shell_quote(&outside_read),
                shell_quote(&format!("{w}/{trimmed_read}"))
            ));
        }
        setup.push(format!(
            "ln -s {} {}",
            shell_quote(read_name),
            shell_quote(&format!("{w}/read-alias"))
        ));
        if trimmed_write != write_name {
            setup.push(format!(
                "ln -s {} {}",
                shell_quote(&outside_write),
                shell_quote(&format!("{w}/{trimmed_write}"))
            ));
        }
        setup.push(format!(
            "ln -s {} {}",
            shell_quote(write_name),
            shell_quote(&format!("{w}/write-alias"))
        ));
        bash(&local.sandbox, &setup.join(" && "));

        let ops = local_ops(&local);
        assert_eq!(ops.read_buffer("read-alias").await.unwrap(), b"inside read\n");
        ops.write_file("write-alias", "updated inside\n")
            .await
            .unwrap();
        assert_eq!(read(&inside_write).as_deref(), Some("updated inside\n"));
        assert_eq!(read(&outside_write).as_deref(), Some("outside write\n"));
    }

    #[tokio::test]
    async fn preserves_trailing_whitespace_in_canonical_paths() {
        assert_canonical_names("read-target ", "write-target ").await;
    }

    #[tokio::test]
    async fn preserves_newlines_in_canonical_paths() {
        assert_canonical_names("read\nname", "write\nname").await;
    }

    #[tokio::test]
    async fn preserves_trailing_newlines_in_canonical_paths() {
        assert_canonical_names("trailing-read\n", "trailing-write\n").await;
    }

    #[tokio::test]
    async fn rejects_intermediate_workspace_symlinks_outside_readable_roots() {
        let local = local_workspace();
        let w = &local.workspace;
        let outside = format!("{}/outside", local.root);
        bash(
            &local.sandbox,
            &[
                format!("mkdir -p {outside}"),
                format!("printf 'outside secret\\n' > {outside}/secret.txt"),
                format!("printf 'old outside\\n' > {outside}/edit.txt"),
                format!("ln -s ../outside {w}/outside-link"),
            ]
            .join(" && "),
        );
        let ops = local_ops(&local);
        assert!(
            err_text(ops.read_buffer("outside-link/secret.txt").await)
                .contains("escapes the readable roots")
        );
        assert!(
            err_text(
                ops.write_file("outside-link/written.txt", "should not be written\n")
                    .await
            )
            .contains("escapes the workspace")
        );
        assert!(
            err_text(ops.edit_file("outside-link/edit.txt", "old", "updated").await)
                .contains("escapes the readable roots")
        );
        assert!(
            err_text(ops.find_files("*.txt", Some("outside-link"), None).await)
                .contains("escapes the readable roots")
        );
        assert!(
            err_text(
                ops.grep_files(
                    "secret",
                    PiGrepInput {
                        path: Some("outside-link".into()),
                        literal: true,
                        ..Default::default()
                    }
                )
                .await
            )
            .contains("escapes the readable roots")
        );
        assert_eq!(
            read(&format!("{outside}/edit.txt")).as_deref(),
            Some("old outside\n")
        );
        assert_eq!(read(&format!("{outside}/written.txt")), None);
    }
}

mod read_buffer {
    use super::*;

    #[tokio::test]
    async fn reads_via_read_binary_file_and_returns_a_buffer() {
        let (_, ops) = make_ops(Behaviors {
            read_binary: read_map(vec![(format!("{SANDBOX_WORK_DIR}/hello.txt"), b"hi".to_vec())]),
            ..Default::default()
        });
        assert_eq!(ops.read_buffer("hello.txt").await.unwrap(), b"hi");
    }

    #[tokio::test]
    async fn throws_when_file_does_not_exist() {
        let (_, ops) = make_ops(Behaviors::default());
        assert!(err_text(ops.read_buffer("nope.txt").await).contains("Path not found"));
    }

    #[tokio::test]
    async fn reads_configured_sandbox_skill_roots() {
        let skill_path = "/home/vercel-sandbox/.agents/skills/weather-codes/SKILL.md";
        let (_, ops) = make_ops(Behaviors {
            read_binary: read_map(vec![(skill_path.into(), b"skill".to_vec())]),
            ..Default::default()
        });
        assert_eq!(ops.read_buffer(skill_path).await.unwrap(), b"skill");
    }

    #[tokio::test]
    async fn rejects_workspace_symlinks_that_resolve_outside_readable_roots() {
        let outside = "/home/vercel-sandbox/CODEX_API_KEY";
        let (sandbox, ops) = make_ops(Behaviors {
            realpath: alias_realpath(
                format!("{SANDBOX_WORK_DIR}/repo-controlled-secret-link"),
                outside.into(),
            ),
            read_binary: read_map(vec![(outside.into(), b"CODEX_API_KEY=secret".to_vec())]),
            ..Default::default()
        });
        assert!(
            err_text(ops.read_buffer("repo-controlled-secret-link").await)
                .contains("escapes the readable roots")
        );
        assert!(sandbox.calls.lock().reads.is_empty());
    }

    #[tokio::test]
    async fn rejects_canonical_paths_with_invalid_utf8_before_file_access() {
        let alias = format!("{SANDBOX_WORK_DIR}/raw-alias");
        let mut invalid = format!("{SANDBOX_WORK_DIR}/raw-").into_bytes();
        invalid.push(0xff);
        let replacement = format!("{SANDBOX_WORK_DIR}/raw-\u{fffd}");
        let (sandbox, ops) = make_ops(Behaviors {
            realpath_bytes: Some(Arc::new(move |path| (path == alias).then(|| invalid.clone()))),
            read_binary: read_map(vec![(replacement, b"outside secret".to_vec())]),
            ..Default::default()
        });
        assert!(err_text(ops.read_buffer("raw-alias").await).contains("Unable to resolve path"));
        assert!(
            err_text(ops.write_file("raw-alias", "should not be written").await)
                .contains("Unable to resolve path")
        );
        assert!(sandbox.calls.lock().reads.is_empty());
        assert!(sandbox.calls.lock().writes.is_empty());
    }

    #[tokio::test]
    async fn rejects_a_later_configured_root_record() {
        let b64 = |value: &[u8]| base64::engine::general_purpose::STANDARD.encode(value);
        for record in [
            "not-base64!".to_string(),
            b64(&[0xff]),
            b64(b"relative"),
        ] {
            let root = "/tmp/pi-artifacts";
            let stdout = [
                b64(format!("{SANDBOX_WORK_DIR}/notes.txt").as_bytes()),
                b64(SANDBOX_WORK_DIR.as_bytes()),
                record,
                String::new(),
            ]
            .join("__PI_REALPATH_FRAME__");
            let sandbox = make_mock_sandbox(Behaviors {
                fixed_run: Some((stdout, String::new(), 0)),
                ..Default::default()
            });
            let ops = ops_for(
                Arc::new(sandbox.clone()),
                create_pi_path_mapper(CreatePiPathMapperOptions {
                    sandbox_work_dir: SANDBOX_WORK_DIR.into(),
                    writable_roots: Some(vec![root.into()]),
                    ..Default::default()
                })
                .unwrap(),
                None,
            );
            let expected = format!("Unable to resolve path: {root}");
            assert_eq!(err_text(ops.read_buffer("notes.txt").await), expected);
            assert_eq!(err_text(ops.write_file("notes.txt", "blocked").await), expected);
            assert!(sandbox.calls.lock().reads.is_empty());
            assert!(sandbox.calls.lock().writes.is_empty());
        }
    }

    #[tokio::test]
    async fn rejects_invalid_utf8_paths_from_native_shell_output() {
        let local = local_workspace();
        let workspace = &local.workspace;
        let outside = format!("{}/outside", local.root);
        let alias = format!("{workspace}/raw-alias");
        let replacement = format!("{workspace}/raw-\u{fffd}");
        let secret = format!("{outside}/secret.txt");
        write(&secret, "outside\n");
        bash(
            &local.sandbox,
            &[
                format!(
                    "raw_path={}$(printf '\\377')",
                    shell_quote(&format!("{workspace}/raw-"))
                ),
                "printf 'inside\\n' > \"$raw_path\"".into(),
                format!("ln -s {} {}", shell_quote(&secret), shell_quote(&replacement)),
                format!("ln -s \"$raw_path\" {}", shell_quote(&alias)),
            ]
            .join("; "),
        );
        let ops = local_ops(&local);
        assert!(err_text(ops.read_buffer("raw-alias").await).contains("Unable to resolve path"));
        assert!(
            err_text(ops.write_file("raw-alias", "should not be written").await)
                .contains("Unable to resolve path")
        );
        assert!(local.sandbox.calls.lock().binary_reads.is_empty());
        assert!(local.sandbox.calls.lock().text_writes.is_empty());
        assert_eq!(read(&secret).as_deref(), Some("outside\n"));
    }
}

mod write_file {
    use super::*;

    #[tokio::test]
    async fn mkdir_p_the_parent_and_writes_via_write_text_file() {
        let (sandbox, ops) = make_ops(Behaviors::default());
        ops.write_file("src/new.ts", "export {};").await.unwrap();
        let calls = sandbox.calls.lock();
        assert!(calls.runs[1].0.contains("mkdir -p"));
        assert!(calls.runs[1].0.contains(&format!("'{SANDBOX_WORK_DIR}/src'")));
        assert_eq!(
            calls.writes,
            vec![(format!("{SANDBOX_WORK_DIR}/src/new.ts"), "export {};".into())]
        );
    }

    fn recorder() -> (
        Arc<Mutex<Vec<(PiRemoteFileChangeKind, String)>>>,
        PiRemoteFileChangeHandler,
    ) {
        let events: Arc<Mutex<Vec<(PiRemoteFileChangeKind, String)>>> = Arc::default();
        let sink = events.clone();
        (
            events,
            Arc::new(move |kind, path: &str, _content: &[u8]| {
                sink.lock().push((kind, path.to_string()))
            }),
        )
    }

    #[tokio::test]
    async fn fires_on_file_change_with_create_when_previous_is_empty() {
        let (events, handler) = recorder();
        let sandbox = make_mock_sandbox(Behaviors::default());
        let ops = ops_for(Arc::new(sandbox), mapper(SANDBOX_WORK_DIR), Some(handler));
        ops.write_file("a.txt", "x").await.unwrap();
        assert_eq!(
            *events.lock(),
            vec![(PiRemoteFileChangeKind::Create, "a.txt".to_string())]
        );
    }

    #[tokio::test]
    async fn fires_on_file_change_with_modify_when_previous_exists() {
        let (events, handler) = recorder();
        let sandbox = make_mock_sandbox(Behaviors {
            read_binary: Some(Arc::new(|_| Some(vec![1, 2, 3]))),
            ..Default::default()
        });
        let ops = ops_for(Arc::new(sandbox), mapper(SANDBOX_WORK_DIR), Some(handler));
        ops.write_file("a.txt", "x").await.unwrap();
        assert_eq!(
            *events.lock(),
            vec![(PiRemoteFileChangeKind::Modify, "a.txt".to_string())]
        );
    }

    #[tokio::test]
    async fn rejects_workspace_symlinks_that_resolve_outside_the_workspace() {
        let outside = "/home/vercel-sandbox/victim-config.json";
        let (sandbox, ops) = make_ops(Behaviors {
            realpath: alias_realpath(
                format!("{SANDBOX_WORK_DIR}/repo-controlled-write-link"),
                outside.into(),
            ),
            read_binary: read_map(vec![(outside.into(), b"{}".to_vec())]),
            ..Default::default()
        });
        assert!(
            err_text(
                ops.write_file("repo-controlled-write-link", "{\"owned\":true}\n")
                    .await
            )
            .contains("escapes the workspace")
        );
        let calls = sandbox.calls.lock();
        assert!(calls.writes.is_empty());
        assert!(!calls.runs.iter().any(|(command, _)| command.contains("mkdir -p")));
    }
}

mod edit_file {
    use super::*;

    #[tokio::test]
    async fn replaces_first_occurrence_and_writes_back() {
        let sandbox = make_mock_sandbox(Behaviors {
            read_binary: Some(Arc::new(|_| {
                Some(b"old text here, and old text again".to_vec())
            })),
            ..Default::default()
        });
        let ops = ops_for(Arc::new(sandbox), mapper(SANDBOX_WORK_DIR), None);
        assert_eq!(
            ops.edit_file("a.txt", "old text", "new text").await.unwrap(),
            "new text here, and old text again"
        );
    }

    #[tokio::test]
    async fn throws_if_old_text_not_found() {
        let sandbox = make_mock_sandbox(Behaviors {
            read_binary: Some(Arc::new(|_| Some(b"hello".to_vec()))),
            ..Default::default()
        });
        let ops = ops_for(Arc::new(sandbox), mapper(SANDBOX_WORK_DIR), None);
        assert!(err_text(ops.edit_file("a.txt", "missing", "x").await).contains("not found"));
    }
}

mod list_directory {
    use super::*;

    #[tokio::test]
    async fn uses_compatible_ls_flags_and_preserves_trailing_directory_markers() {
        let (sandbox, ops) = make_ops(Behaviors {
            run: run("src/\nREADME.md\nnode_modules/\n", 0),
            ..Default::default()
        });
        assert_eq!(
            ops.list_directory(Some("."), None).await.unwrap(),
            vec!["node_modules/", "README.md", "src/"]
        );
        let calls = sandbox.calls.lock();
        let command = calls
            .runs
            .iter()
            .find(|(command, _)| command.contains("ls -1AF"))
            .map(|(command, _)| command.clone())
            .unwrap_or_default();
        assert!(command.contains("ls -1AF"));
        assert!(!command.contains("ls -1Ap"));
    }

    #[tokio::test]
    async fn throws_on_ls_not_found_sentinel() {
        let (_, ops) = make_ops(Behaviors {
            run: run("__PI_LS_NOT_FOUND__\n", 2),
            ..Default::default()
        });
        assert!(err_text(ops.list_directory(Some("nope"), None).await).contains("Path not found"));
    }
}

mod grep_files {
    use super::*;

    fn grep_command(sandbox: &MockSandbox, needle: &str) -> String {
        sandbox
            .calls
            .lock()
            .runs
            .iter()
            .find(|(command, _)| command.contains(needle))
            .map(|(command, _)| command.clone())
            .unwrap_or_default()
    }

    #[tokio::test]
    async fn builds_the_grep_command_with_the_requested_flags() {
        let (sandbox, ops) = make_ops(Behaviors {
            run: run("foo.ts:1:hit\n", 0),
            ..Default::default()
        });
        let out = ops
            .grep_files(
                "TODO",
                PiGrepInput {
                    ignore_case: true,
                    literal: true,
                    context: Some(2),
                    glob: Some("*.ts".into()),
                    limit: Some(50),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(out.contains("foo.ts:1:hit"));
        let cmd = grep_command(&sandbox, "grep ");
        for expected in [
            "grep",
            "-r",
            "-i",
            "-F",
            "-C",
            "'-m' '50'",
            "'--include=*.ts'",
            "-e 'TODO'",
            "binary_option_error=$(grep --binary-files=without-match",
            "grep $binary_option",
            "head -n 50",
            "head -c 8192 \"$grep_stderr\" >&2",
            "find ",
            "-exec bash -c",
            "grep_remaining",
        ] {
            assert!(cmd.contains(expected), "missing {expected}");
        }
        assert!(!cmd.contains("-R"));
        assert!(!cmd.contains("2>/dev/null"));
        assert!(!cmd.contains("grep_output=$(grep"));
    }

    #[tokio::test]
    async fn returns_no_matches_found_on_empty_output() {
        let (_, ops) = make_ops(Behaviors {
            run: run("", 1),
            ..Default::default()
        });
        assert_eq!(
            ops.grep_files("x", PiGrepInput::default()).await.unwrap(),
            "No matches found"
        );
    }

    #[tokio::test]
    async fn accepts_grep_sigpipe_when_head_stops_after_the_requested_limit() {
        let (_, ops) = make_ops(Behaviors {
            run: run("foo.ts:1:hit\n", 141),
            ..Default::default()
        });
        assert_eq!(
            ops.grep_files(
                "hit",
                PiGrepInput {
                    limit: Some(1),
                    ..Default::default()
                }
            )
            .await
            .unwrap(),
            "foo.ts:1:hit"
        );
    }

    #[tokio::test]
    async fn prefixes_option_like_relative_targets_with_dot_slash() {
        let (sandbox, ops) = make_ops(Behaviors {
            run: run("./-notes.txt:1:hit\n", 0),
            ..Default::default()
        });
        ops.grep_files(
            "hit",
            PiGrepInput {
                path: Some("-notes.txt".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert!(grep_command(&sandbox, "grep $binary_option").contains("'./-notes.txt'"));
    }

    #[tokio::test]
    async fn surfaces_grep_errors_instead_of_reporting_no_matches() {
        let (_, ops) = make_ops(Behaviors {
            run: Some(Arc::new(|_| {
                Some((String::new(), "grep: invalid option\n".into(), 0))
            })),
            ..Default::default()
        });
        assert_eq!(
            err_text(ops.grep_files("x", PiGrepInput::default()).await),
            "grep: invalid option"
        );
    }

    #[tokio::test]
    async fn preserves_gnu_grep_matches_when_recursive_traversal_also_reports_diagnostics() {
        if is_root() {
            // [rust] File modes do not deny root.
            return;
        }
        let local = local_workspace();
        let w = &local.workspace;
        write(&format!("{w}/match.txt"), "needle\n");
        write(&format!("{w}/blocked/hidden.txt"), "needle\n");
        bash(&local.sandbox, &format!("chmod 0 {w}/blocked"));
        let output = local_ops(&local)
            .grep_files(
                "needle",
                PiGrepInput {
                    literal: true,
                    ..Default::default()
                },
            )
            .await;
        bash(&local.sandbox, &format!("chmod 700 {w}/blocked"));
        let output = output.unwrap();
        assert!(output.contains("match.txt:1:needle"));
        assert!(output.contains("Permission denied"));
    }

    #[tokio::test]
    async fn bounds_gnu_grep_diagnostics_independently_of_the_match_limit() {
        if is_root() {
            return;
        }
        let local = local_workspace();
        let w = &local.workspace;
        write(&format!("{w}/match.txt"), "needle\n");
        bash(
            &local.sandbox,
            "for i in $(seq -w 0 249); do mkdir blocked-$i; chmod 0 blocked-$i; done",
        );
        let output = local_ops(&local)
            .grep_files(
                "needle",
                PiGrepInput {
                    literal: true,
                    limit: Some(1),
                    ..Default::default()
                },
            )
            .await;
        bash(&local.sandbox, "chmod 700 blocked-*");
        let output = output.unwrap();
        assert!(output.contains("match.txt:1:needle"));
        assert!(output.contains("Permission denied"));
        assert!(output.len() < 8_300);
    }

    #[tokio::test]
    async fn rejects_workspace_symlinks_before_running_grep_outside_readable_roots() {
        let (sandbox, ops) = make_ops(Behaviors {
            realpath: alias_realpath(
                format!("{SANDBOX_WORK_DIR}/repo-controlled-secret-link"),
                "/home/vercel-sandbox/CODEX_API_KEY".into(),
            ),
            run: run("CODEX_API_KEY=secret\n", 0),
            ..Default::default()
        });
        assert!(
            err_text(
                ops.grep_files(
                    "secret",
                    PiGrepInput {
                        path: Some("repo-controlled-secret-link".into()),
                        literal: true,
                        ..Default::default()
                    }
                )
                .await
            )
            .contains("escapes the readable roots")
        );
        assert!(
            !sandbox
                .calls
                .lock()
                .runs
                .iter()
                .any(|(command, _)| command.contains("grep "))
        );
    }

    #[tokio::test]
    async fn returns_text_matches_when_the_workspace_also_contains_matching_binary_data() {
        let local = local_workspace();
        write(&format!("{}/text.txt", local.workspace), "hello from text\n");
        std::fs::write(
            format!("{}/data.bin", local.workspace),
            [0, 104, 101, 108, 108, 111, 10],
        )
        .unwrap();
        let output = local_ops(&local)
            .grep_files(
                "hello",
                PiGrepInput {
                    literal: true,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(output.contains("text.txt:1:hello from text"));
    }

    #[tokio::test]
    async fn searches_an_option_like_filename() {
        let local = local_workspace();
        write(&format!("{}/-notes.txt", local.workspace), "hello\n");
        let output = local_ops(&local)
            .grep_files(
                "hello",
                PiGrepInput {
                    path: Some("-notes.txt".into()),
                    literal: true,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(output.contains("hello"));
    }

    #[tokio::test]
    async fn surfaces_grep_diagnostics() {
        let local = local_workspace();
        write(&format!("{}/text.txt", local.workspace), "hello\n");
        // [rust] GNU grep words the diagnostic differently from just-bash.
        let error = err_text(
            local_ops(&local)
                .grep_files("[", PiGrepInput::default())
                .await,
        );
        assert!(error.starts_with("grep:"), "{error}");
    }

    #[tokio::test]
    async fn limits_high_cardinality_grep_output() {
        let local = local_workspace();
        let content: String = (0..5_000).map(|index| format!("match {index}\n")).collect();
        write(&format!("{}/many.txt", local.workspace), &content);
        let output = local_ops(&local)
            .grep_files(
                "match",
                PiGrepInput {
                    literal: true,
                    limit: Some(3),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(output.split('\n').count(), 3);
        assert!(output.contains("many.txt:1:match 0"));
    }

    #[tokio::test]
    async fn limits_matches_spread_across_many_files() {
        let local = local_workspace();
        for index in 0..250 {
            write(
                &format!("{}/match-{index:03}.ts", local.workspace),
                &format!("match {index}\n"),
            );
        }
        let output = local_ops(&local)
            .grep_files(
                "match",
                PiGrepInput {
                    glob: Some("*.ts".into()),
                    literal: true,
                    limit: Some(3),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(output.split('\n').count(), 3);
    }
}

mod exec {
    use super::*;

    #[tokio::test]
    async fn runs_through_bash_and_returns_exit_code() {
        let (_, ops) = make_ops(Behaviors {
            run: run("hello\n", 0),
            ..Default::default()
        });
        let mut chunks: Vec<u8> = Vec::new();
        let mut on_data = |data: &[u8]| chunks.extend_from_slice(data);
        let result = ops
            .exec(
                "echo hello",
                ".",
                PiExecInput {
                    on_data: Some(&mut on_data),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(result, Some(0));
        assert_eq!(chunks, b"hello\n");
    }

    struct AbortWatcher;

    #[async_trait]
    impl SandboxSession for AbortWatcher {
        fn description(&self) -> String {
            "abort watcher".into()
        }
        async fn read_file(&self, _o: SandboxReadFileOptions) -> HarnessResult<Option<ByteStream>> {
            unimplemented!()
        }
        async fn read_binary_file(
            &self,
            _o: SandboxReadBinaryFileOptions,
        ) -> HarnessResult<Option<Bytes>> {
            unimplemented!()
        }
        async fn read_text_file(
            &self,
            _o: SandboxReadTextFileOptions,
        ) -> HarnessResult<Option<String>> {
            unimplemented!()
        }
        async fn write_file(&self, _o: SandboxWriteFileOptions) -> HarnessResult<()> {
            unimplemented!()
        }
        async fn write_binary_file(&self, _o: SandboxWriteBinaryFileOptions) -> HarnessResult<()> {
            unimplemented!()
        }
        async fn write_text_file(&self, _o: SandboxWriteTextFileOptions) -> HarnessResult<()> {
            unimplemented!()
        }
        async fn stat(&self, _o: SandboxStatOptions) -> HarnessResult<Option<SandboxFileStat>> {
            unimplemented!()
        }
        async fn list_directory(
            &self,
            _o: SandboxListDirectoryOptions,
        ) -> HarnessResult<Option<Vec<SandboxDirectoryEntry>>> {
            unimplemented!()
        }
        async fn delete_file(&self, _o: SandboxDeleteFileOptions) -> HarnessResult<bool> {
            unimplemented!()
        }
        async fn spawn(&self, _o: SandboxSpawnOptions) -> HarnessResult<SandboxProcess> {
            unimplemented!()
        }
        async fn run(&self, o: SandboxRunOptions) -> HarnessResult<SandboxRunResult> {
            let started = tokio::time::Instant::now();
            o.process.abort_signal.unwrap().cancelled().await;
            Ok(SandboxRunResult {
                exit_code: 130,
                stdout: started.elapsed().as_millis().to_string(),
                ..Default::default()
            })
        }
    }

    #[tokio::test(start_paused = true)]
    async fn schedules_the_abort_timeout_in_seconds_not_milliseconds() {
        let ops = ops_for(Arc::new(AbortWatcher), mapper(SANDBOX_WORK_DIR), None);
        let mut output = Vec::new();
        let mut on_data = |data: &[u8]| output.extend_from_slice(data);
        ops.exec(
            "echo hi",
            ".",
            PiExecInput {
                on_data: Some(&mut on_data),
                // A 30-second timeout must abort 30_000 ms out, not 30 ms.
                timeout: Some(30.0),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(String::from_utf8(output).unwrap(), "30000");
    }
}

struct PathPolicyFixture {
    local: LocalWorkspace,
    home: String,
    denied: String,
    dot_denied: String,
    ops: PiRemoteOps,
}

fn make_path_policy_fixture() -> PathPolicyFixture {
    let local = local_workspace();
    let workspace = local.workspace.clone();
    let home = format!("{}/home", local.root);
    let denied = format!("{home}/credentials");
    let dot_denied = format!("{home}/..private");
    std::fs::create_dir_all(format!("{workspace}/private/..backup")).unwrap();
    std::fs::create_dir_all(format!("{home}/docs")).unwrap();
    std::fs::create_dir_all(&denied).unwrap();
    std::fs::create_dir_all(&dot_denied).unwrap();
    write(&format!("{workspace}/main.txt"), "needle workspace\n");
    write(
        &format!("{workspace}/private/..backup/token.txt"),
        "needle backup\n",
    );
    write(&format!("{home}/docs/notes.txt"), "needle notes\n");
    write(&format!("{denied}/token.txt"), "needle token\n");
    write(&format!("{dot_denied}/token.txt"), "needle dotted secret\n");
    let ops = ops_for(
        Arc::new(local.sandbox.clone()),
        create_pi_path_mapper(CreatePiPathMapperOptions {
            sandbox_work_dir: workspace.clone(),
            readable_roots: Some(vec![PiReadablePathRoot {
                sandbox_dir: home.clone(),
            }]),
            denied_roots: Some(vec![
                denied.clone(),
                dot_denied.clone(),
                format!("{workspace}/private"),
            ]),
            ..Default::default()
        })
        .unwrap(),
        None,
    );
    PathPolicyFixture {
        local,
        home,
        denied,
        dot_denied,
        ops,
    }
}

fn grep(path: Option<&str>, literal: bool) -> PiGrepInput {
    PiGrepInput {
        path: path.map(String::from),
        literal,
        ..Default::default()
    }
}

mod with_denied_roots {
    use super::*;

    const DENIED: &str = "inside a denied root";

    #[tokio::test]
    async fn refuses_a_symlinked_denied_root_through_its_canonical_workspace_path() {
        let local = local_workspace();
        let workspace = local.workspace.clone();
        let home = format!("{}/home", local.root);
        let target = format!("{workspace}/private");
        let alias = format!("{home}/blocked");
        write(&format!("{target}/notes.txt"), "needle private\n");
        std::fs::create_dir_all(&home).unwrap();
        write(&format!("{workspace}/public.txt"), "needle public\n");
        write(&format!("{workspace}/private-copy.txt"), "needle copy\n");
        std::os::unix::fs::symlink(&target, &alias).unwrap();

        let canonical_root =
            resolve_pi_sandbox_path_or_parent(&local.sandbox, &alias, &alias, None, None)
                .await
                .unwrap();
        assert_eq!(canonical_root, target);

        let ops = ops_for(
            Arc::new(local.sandbox.clone()),
            create_pi_path_mapper(CreatePiPathMapperOptions {
                sandbox_work_dir: workspace.clone(),
                readable_roots: Some(vec![PiReadablePathRoot {
                    sandbox_dir: home.clone(),
                }]),
                denied_roots: Some(vec![alias.clone(), canonical_root]),
                ..Default::default()
            })
            .unwrap(),
            None,
        );

        assert!(err_text(ops.read_buffer("private/notes.txt").await).contains(DENIED));
        assert!(err_text(ops.read_buffer(&format!("{alias}/notes.txt")).await).contains(DENIED));
        assert!(err_text(ops.access("private/notes.txt").await).contains(DENIED));
        assert!(err_text(ops.list_directory(Some("private"), None).await).contains(DENIED));
        assert!(err_text(ops.find_files("*.txt", Some("private"), None).await).contains(DENIED));
        assert!(
            err_text(ops.grep_files("needle", grep(Some("private"), true)).await).contains(DENIED)
        );
        assert!(
            err_text(ops.write_file("private/new.txt", "new private\n").await).contains(DENIED)
        );
        assert!(
            err_text(ops.edit_file("private/notes.txt", "private", "changed").await)
                .contains(DENIED)
        );
        assert!(local.sandbox.calls.lock().binary_reads.is_empty());
        assert!(local.sandbox.calls.lock().text_writes.is_empty());

        assert_eq!(
            ops.find_files("*.txt", Some("."), None).await.unwrap(),
            vec!["private-copy.txt", "public.txt"]
        );
        let output = ops.grep_files("needle", grep(None, true)).await.unwrap();
        assert!(output.contains("public.txt:1:needle public"));
        assert!(output.contains("private-copy.txt:1:needle copy"));
        assert!(!output.contains("needle private"));
        assert_eq!(
            ops.read_buffer("private-copy.txt").await.unwrap(),
            b"needle copy\n"
        );
    }

    #[tokio::test]
    async fn refuses_to_read_a_path_inside_a_denied_root() {
        let fixture = make_path_policy_fixture();
        let ops = &fixture.ops;
        assert!(
            err_text(ops.read_buffer(&format!("{}/token.txt", fixture.denied)).await)
                .contains(DENIED)
        );
        assert!(err_text(ops.list_directory(Some(&fixture.denied), None).await).contains(DENIED));
        assert!(err_text(ops.read_buffer("private/..backup/token.txt").await).contains(DENIED));
        assert!(
            err_text(
                ops.read_buffer(&format!("{}/token.txt", fixture.dot_denied))
                    .await
            )
            .contains(DENIED)
        );
        assert!(fixture.local.sandbox.calls.lock().binary_reads.is_empty());
        assert_eq!(
            ops.read_buffer(&format!("{}/docs/notes.txt", fixture.home))
                .await
                .unwrap(),
            b"needle notes\n"
        );
    }

    #[tokio::test]
    async fn refuses_to_read_through_a_symlink_into_a_denied_root() {
        let fixture = make_path_policy_fixture();
        let ops = &fixture.ops;
        std::os::unix::fs::symlink(
            format!("{}/token.txt", fixture.denied),
            format!("{}/token-link.txt", fixture.local.workspace),
        )
        .unwrap();
        let alias = format!("{}/alias", fixture.home);
        std::os::unix::fs::symlink(&fixture.denied, &alias).unwrap();

        assert!(err_text(ops.read_buffer("token-link.txt").await).contains(DENIED));
        assert!(err_text(ops.read_buffer(&format!("{alias}/token.txt")).await).contains(DENIED));
        assert!(err_text(ops.find_files("*.txt", Some(&alias), None).await).contains(DENIED));
        assert!(err_text(ops.grep_files("needle", grep(Some(&alias), false)).await).contains(DENIED));
        assert!(fixture.local.sandbox.calls.lock().binary_reads.is_empty());
    }

    #[tokio::test]
    async fn refuses_to_write_inside_a_denied_root() {
        let fixture = make_path_policy_fixture();
        assert!(
            err_text(fixture.ops.write_file("private/journal.txt", "overwritten\n").await)
                .contains(DENIED)
        );
        assert!(
            err_text(
                fixture
                    .ops
                    .write_file("private/..backup/new.txt", "overwritten\n")
                    .await
            )
            .contains(DENIED)
        );
        assert!(fixture.local.sandbox.calls.lock().text_writes.is_empty());
    }

    #[tokio::test]
    async fn skips_denied_roots_when_finding_and_searching_files() {
        let fixture = make_path_policy_fixture();
        let ops = &fixture.ops;
        let home = &fixture.home;
        std::os::unix::fs::symlink(&fixture.denied, format!("{home}/alias")).unwrap();
        write(
            &format!("{home}/docs/credentials/public.txt"),
            "needle nested public\n",
        );
        write(
            &format!("{}/credentials/public.txt", fixture.local.workspace),
            "needle public\n",
        );

        assert_eq!(
            ops.find_files("**/*.txt", Some(home), None).await.unwrap(),
            vec!["docs/credentials/public.txt", "docs/notes.txt"]
        );
        assert!(
            ops.find_files("..private/*.txt", Some(home), None)
                .await
                .unwrap()
                .is_empty()
        );
        let output = ops.grep_files("needle", grep(Some(home), true)).await.unwrap();
        assert!(output.contains("notes.txt:1:needle notes"));
        assert!(output.contains("docs/credentials/public.txt:1:needle nested public"));
        assert!(!output.contains("needle token"));
        assert!(!output.contains("needle dotted secret"));
        assert!(
            ops.grep_files("needle", grep(None, true))
                .await
                .unwrap()
                .contains("credentials/public.txt:1:needle public")
        );
    }

    #[tokio::test]
    async fn skips_denied_roots_in_the_per_file_grep_fallback() {
        let local = local_workspace();
        let w = local.workspace.clone();
        write(&format!("{w}/docs/notes.txt"), "needle notes\n");
        write(&format!("{w}/private/token.txt"), "needle token\n");
        write(&format!("{w}/..private/token.txt"), "needle dotted secret\n");
        let ops = ops_for(
            Arc::new(local.sandbox.clone()),
            create_pi_path_mapper(CreatePiPathMapperOptions {
                sandbox_work_dir: w.clone(),
                denied_roots: Some(vec![format!("{w}/private"), format!("{w}/..private")]),
                ..Default::default()
            })
            .unwrap(),
            None,
        );
        assert_eq!(
            ops.find_files("**/*.txt", Some("."), None).await.unwrap(),
            vec!["docs/notes.txt"]
        );
        assert!(
            ops.find_files("..private/*.txt", Some("."), None)
                .await
                .unwrap()
                .is_empty()
        );
        let output = ops.grep_files("needle", grep(None, true)).await.unwrap();
        assert!(output.contains("docs/notes.txt:1:needle notes"));
        assert!(!output.contains("needle token"));
        assert!(!output.contains("needle dotted secret"));
    }

    #[tokio::test]
    async fn resolves_denied_symlinks_and_excludes_their_targets() {
        let local = local_workspace();
        let w = local.workspace.clone();
        let target = format!("{w}/private");
        let alias = format!("{w}/blocked");
        write(&format!("{target}/notes.txt"), "needle private\n");
        write(&format!("{w}/public.txt"), "needle public\n");
        std::os::unix::fs::symlink("private", &alias).unwrap();

        let canonical_root =
            resolve_pi_sandbox_path_or_parent(&local.sandbox, &alias, &alias, None, None)
                .await
                .unwrap();
        assert_eq!(canonical_root, target);
        let ops = ops_for(
            Arc::new(local.sandbox.clone()),
            create_pi_path_mapper(CreatePiPathMapperOptions {
                sandbox_work_dir: w.clone(),
                denied_roots: Some(vec![alias, canonical_root]),
                ..Default::default()
            })
            .unwrap(),
            None,
        );
        assert!(err_text(ops.read_buffer("private/notes.txt").await).contains(DENIED));
        assert_eq!(
            ops.find_files("*.txt", Some("."), None).await.unwrap(),
            vec!["public.txt"]
        );
        let output = ops.grep_files("needle", grep(None, true)).await.unwrap();
        assert!(output.contains("public.txt:1:needle public"));
        assert!(!output.contains("needle private"));
    }

    #[tokio::test]
    async fn resolves_a_missing_denied_directory_beneath_a_symlinked_parent() {
        let local = local_workspace();
        let w = &local.workspace;
        std::fs::create_dir_all(format!("{w}/actual")).unwrap();
        std::os::unix::fs::symlink("actual", format!("{w}/link")).unwrap();
        let alias = format!("{w}/link/future/private");
        assert_eq!(
            resolve_pi_sandbox_path_or_parent(&local.sandbox, &alias, &alias, None, None)
                .await
                .unwrap(),
            format!("{w}/actual/future/private")
        );
    }

    #[tokio::test]
    async fn refuses_malformed_canonical_denied_paths_from_the_sandbox() {
        let alias = format!("{SANDBOX_WORK_DIR}/blocked");
        let sandbox = make_mock_sandbox(Behaviors {
            realpath_bytes: Some(Arc::new(|_| Some(vec![0xff]))),
            ..Default::default()
        });
        assert_eq!(
            err_text(resolve_pi_sandbox_path_or_parent(&sandbox, &alias, &alias, None, None).await),
            format!("Unable to resolve path: {alias}")
        );
    }

    #[tokio::test]
    async fn keeps_the_stock_find_and_grep_commands_without_a_policy() {
        let (sandbox, ops) = make_ops(Behaviors {
            run: run("", 0),
            ..Default::default()
        });
        ops.find_files("*.ts", Some("."), None).await.unwrap();
        let _ = ops.grep_files("needle", PiGrepInput::default()).await;
        let commands: String = sandbox
            .calls
            .lock()
            .runs
            .iter()
            .map(|(command, _)| command.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(commands.contains(&format!("find '{SANDBOX_WORK_DIR}' -type f -print")));
        assert!(commands.contains("'-r' '-n' '-m' '100' -e 'needle' '.'"));
        assert!(commands.contains("find '.' -type f -exec"));
        assert!(!commands.contains("-prune"));
    }
}

mod configured_writable_roots {
    use super::*;

    fn make_policy_ops(
        behaviors: Behaviors,
        policy: PiFileToolPathPolicy,
        on_file_change: Option<PiRemoteFileChangeHandler>,
    ) -> (MockSandbox, PiRemoteOps) {
        let sandbox = make_mock_sandbox(behaviors);
        let paths = create_pi_path_mapper(CreatePiPathMapperOptions {
            sandbox_work_dir: SANDBOX_WORK_DIR.into(),
            readable_roots: policy.readable_roots.map(|roots| {
                roots
                    .into_iter()
                    .map(|sandbox_dir| PiReadablePathRoot { sandbox_dir })
                    .collect()
            }),
            writable_roots: policy.writable_roots,
            denied_roots: policy.denied_roots,
            home_dir: None,
        })
        .unwrap();
        let ops = ops_for(Arc::new(sandbox.clone()), paths, on_file_change);
        (sandbox, ops)
    }

    fn strings(values: &[&str]) -> Option<Vec<String>> {
        Some(values.iter().map(|value| value.to_string()).collect())
    }

    #[tokio::test]
    async fn batches_configured_roots_and_the_target_in_one_sandbox_call() {
        let (sandbox, ops) = make_policy_ops(
            Behaviors {
                read_binary: Some(Arc::new(|_| Some(b"reference".to_vec()))),
                ..Default::default()
            },
            PiFileToolPathPolicy {
                readable_roots: strings(&["/mnt/reference", "/mnt/other-reference"]),
                writable_roots: strings(&["/tmp"]),
                denied_roots: strings(&["/tmp/private"]),
            },
            None,
        );
        ops.read_buffer("/mnt/reference/info.txt").await.unwrap();
        let calls = sandbox.calls.lock();
        assert_eq!(calls.runs.len(), 1);
        assert!(calls.runs[0].0.contains("target='/mnt/reference'"));
        assert!(calls.runs[0].0.contains("target='/tmp/private'"));
    }

    #[tokio::test]
    async fn allows_external_writes_and_edits_without_workspace_file_change_events() {
        let events: Arc<Mutex<usize>> = Arc::default();
        let sink = events.clone();
        let (sandbox, ops) = make_policy_ops(
            Behaviors {
                read_binary: read_map(vec![("/tmp/existing.txt".into(), b"old value".to_vec())]),
                ..Default::default()
            },
            PiFileToolPathPolicy {
                readable_roots: strings(&["/mnt/reference"]),
                writable_roots: strings(&["/tmp"]),
                denied_roots: strings(&["/tmp/private"]),
            },
            Some(Arc::new(move |_, _, _| *sink.lock() += 1)),
        );
        ops.write_file("/tmp/new.txt", "new value").await.unwrap();
        assert_eq!(
            ops.edit_file("/tmp/existing.txt", "old", "updated")
                .await
                .unwrap(),
            "updated value"
        );
        assert_eq!(
            sandbox.calls.lock().writes,
            vec![
                ("/tmp/new.txt".to_string(), "new value".to_string()),
                ("/tmp/existing.txt".to_string(), "updated value".to_string()),
            ]
        );
        assert_eq!(*events.lock(), 0);
        assert!(
            err_text(ops.write_file("/mnt/reference/no-write.txt", "no").await)
                .contains("escapes the workspace")
        );
        assert!(
            err_text(ops.read_buffer("/tmp/private/token").await).contains("inside a denied root")
        );
    }

    #[tokio::test]
    async fn rechecks_missing_write_paths_through_their_canonical_parents() {
        let (sandbox, ops) = make_policy_ops(
            Behaviors {
                realpath: alias_realpath(
                    "/tmp/link/missing/output.json".into(),
                    "/etc/missing/output.json".into(),
                ),
                ..Default::default()
            },
            PiFileToolPathPolicy {
                writable_roots: strings(&["/tmp"]),
                ..Default::default()
            },
            None,
        );
        assert!(
            err_text(ops.write_file("/tmp/link/missing/output.json", "{}").await)
                .contains("escapes the workspace")
        );
        let calls = sandbox.calls.lock();
        assert!(calls.writes.is_empty());
        assert!(
            calls.runs[0]
                .0
                .contains("while [ ! -e \"$dir\" ] && [ ! -L \"$dir\" ]")
        );
    }
}

mod glob {
    use super::*;

    #[test]
    fn matches_node_matches_glob_semantics() {
        assert!(matches_glob("notes.md", "*.md"));
        assert!(!matches_glob("docs/notes.md", "*.md"));
        assert!(matches_glob("docs/notes.md", "**/*.md"));
        assert!(matches_glob("notes.md", "**/*.md"));
        assert!(matches_glob("a/b/c.ts", "a/**/c.ts"));
        assert!(matches_glob("a.ts", "*.{ts,js}"));
        assert!(matches_glob("a.js", "*.{ts,js}"));
        assert!(!matches_glob("a.md", "*.{ts,js}"));
        assert!(matches_glob("a1.ts", "a?.ts"));
        assert!(matches_glob("ab.ts", "a[a-c].ts"));
        assert!(!matches_glob("ad.ts", "a[a-c].ts"));
        assert!(matches_glob("ad.ts", "a[!a-c].ts"));
        assert!(!matches_glob(".hidden", "*"));
        assert!(matches_glob(".hidden", ".*"));
        assert!(!matches_glob(".git/config", "**/config"));
        assert!(matches_glob("..private/token.txt", "..private/*.txt"));
    }
}
