//! Port of `is-sandbox-write-protected-path.ts` and
//! `write-protected-path-matching.ts`.

use std::collections::BTreeSet;

use base64::Engine as _;
use tokio_util::sync::CancellationToken;

use super::shell_quote;
use crate::harness::error::{BoxError, HarnessResult};
use crate::harness::paths::{normalize_posix, posix_join};
use crate::harness::sandbox::{SandboxProcessOptions, SandboxRunOptions, SandboxSession};

/// Upper bound on entries scanned inside write-protected directories per
/// check. Exceeding it reports the write as protected.
pub const WRITE_PROTECTION_SCAN_LIMIT: usize = 2000;

/// Shell functions shared by every write protection script (bash 3.2,
/// GNU / BSD / BusyBox tools). Copied verbatim from the TS source.
const WRITE_PROTECTION_SHELL_FUNCTIONS: &str = include_str!("write_protection.sh");

/// The `TypeError` raised for invalid protected path entries.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct WriteProtectedPathEntryError(pub String);

/// `assertWriteProtectedPathEntries`.
pub fn assert_write_protected_path_entries(
    entries: &[String],
) -> Result<(), WriteProtectedPathEntryError> {
    for entry in entries {
        if entry.is_empty()
            || entry.starts_with('/')
            || entry
                .split('/')
                .any(|s| s.is_empty() || s == "." || s == "..")
        {
            return Err(WriteProtectedPathEntryError(format!(
                "Write-protected path entries must be relative paths without empty, \".\" or \"..\" segments; received {}.",
                serde_json::to_string(entry).unwrap_or_default()
            )));
        }
    }
    Ok(())
}

/// `isInsideWriteProtectionPath({ parent, candidate })`.
pub fn is_inside_write_protection_path(parent: &str, candidate: &str) -> bool {
    let parent = normalize_posix(parent);
    let candidate = normalize_posix(candidate);
    let parent = if parent.len() > 1 {
        parent.trim_end_matches('/')
    } else {
        parent.as_str()
    };
    let candidate = if candidate.len() > 1 {
        candidate.trim_end_matches('/')
    } else {
        candidate.as_str()
    };
    if parent == candidate || parent == "/" {
        return true;
    }
    candidate
        .strip_prefix(parent)
        .is_some_and(|rest| rest.starts_with('/'))
}

/// A target of [`matches_write_protected_path`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteProtectionTarget {
    pub path: String,
    pub is_multi_link_file: bool,
}

/// `matchesWriteProtectedPath`: a target matches when it is a protected path,
/// lies inside one, or is a parent of one; while any protected path exists,
/// files with more than one link match too.
pub fn matches_write_protected_path(
    targets: &[WriteProtectionTarget],
    protected_paths: &[String],
    any_protected_path_exists: bool,
) -> bool {
    targets.iter().any(|t| {
        protected_paths.iter().any(|p| {
            is_inside_write_protection_path(p, &t.path)
                || is_inside_write_protection_path(&t.path, p)
        }) || (any_protected_path_exists && t.is_multi_link_file)
    })
}

#[derive(Debug, Clone)]
struct Record {
    kind: String,
    flags: Vec<String>,
    path: Option<String>,
}

fn without_trailing_slashes(p: &str) -> String {
    let trimmed = p.trim_end_matches('/');
    if trimmed.is_empty() {
        "/".into()
    } else {
        trimmed.to_string()
    }
}

/// `isSandboxWriteProtectedPath`: whether writing any of `paths` touches a
/// write-protected path (entries relative to `work_dir`). Any failure
/// reports the write as protected.
pub async fn is_sandbox_write_protected_path(
    sandbox: &dyn SandboxSession,
    work_dir: &str,
    paths: &[String],
    write_protected_paths: &[String],
    abort_signal: Option<CancellationToken>,
) -> Result<bool, WriteProtectedPathEntryError> {
    assert_write_protected_path_entries(write_protected_paths)?;
    if paths.is_empty() {
        return Ok(true);
    }
    if write_protected_paths.is_empty() {
        return Ok(false);
    }
    Ok(check(
        sandbox,
        work_dir,
        paths,
        write_protected_paths,
        abort_signal,
    )
    .await
    .unwrap_or(true))
}

async fn check(
    sandbox: &dyn SandboxSession,
    work_dir: &str,
    paths: &[String],
    write_protected_paths: &[String],
    abort_signal: Option<CancellationToken>,
) -> HarnessResult<bool> {
    let mut target_paths = Vec::new();
    for file_path in paths {
        if file_path.trim().is_empty() || !file_path.starts_with('/') {
            return Ok(true);
        }
        let normalized = normalize_posix(file_path);
        let mut candidates = vec![file_path.clone()];
        if normalized != *file_path {
            candidates.push(normalized);
        }
        for c in candidates {
            target_paths.push(without_trailing_slashes(&c));
        }
    }

    let quoted = |v: &[String]| {
        v.iter()
            .map(|s| shell_quote(s))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let script = [
        WRITE_PROTECTION_SHELL_FUNCTIONS.to_string(),
        format!("workdir={}", shell_quote(work_dir)),
        "canonical \"$workdir\" || exit 3".into(),
        "canonical_workdir=$canonical_path".into(),
        "emit W \"$(b64 \"$canonical_workdir\")\"".into(),
        "scan_roots=()".into(),
        format!("for entry in {}; do", quoted(write_protected_paths)),
        "  resolve \"$workdir/$entry\" || exit 3".into(),
        "  root_exists=0".into(),
        "  root_is_dir=0".into(),
        "  if [ -e \"$resolved\" ] || [ -L \"$resolved\" ]; then root_exists=1; fi".into(),
        "  if [ \"$root_exists\" = 1 ] && [ -d \"$resolved\" ] && [ ! -L \"$resolved\" ]; then".into(),
        "    root_is_dir=1".into(),
        "    root_seen=0".into(),
        "    for scan_root in \"${scan_roots[@]}\"; do [ \"$scan_root\" = \"$resolved\" ] && root_seen=1; done".into(),
        "    [ \"$root_seen\" = 1 ] || scan_roots+=(\"$resolved\")".into(),
        "  fi".into(),
        "  emit P \"$root_exists\" \"$root_is_dir\" \"$(b64 \"$resolved\")\"".into(),
        "done".into(),
        format!("for target in {}; do", quoted(&target_paths)),
        "  resolve \"$target\" || exit 3".into(),
        "  target_multi_link=0".into(),
        "  if [ -e \"$resolved\" ] && [ -n \"$(find \"$resolved\" -prune -type f -links +1 -print)\" ]; then target_multi_link=1; fi".into(),
        "  emit T \"$target_multi_link\" \"$(b64 \"$resolved\")\"".into(),
        "done".into(),
        format!("scan_budget={WRITE_PROTECTION_SCAN_LIMIT}"),
        "for scan_root in \"${scan_roots[@]}\"; do scan \"$scan_root\" || exit 3; done".into(),
        "emit E".into(),
    ]
    .join("\n");
    let Some(initial) = run_script(sandbox, &script, abort_signal.clone()).await? else {
        return Ok(true);
    };

    let canonical_work_dir = initial
        .iter()
        .find(|r| r.kind == "W")
        .and_then(|r| r.path.clone());
    let roots: Vec<&Record> = initial.iter().filter(|r| r.kind == "P").collect();
    let targets: Vec<&Record> = initial.iter().filter(|r| r.kind == "T").collect();
    let Some(canonical_work_dir) = canonical_work_dir else {
        return Ok(true);
    };
    if roots.len() != write_protected_paths.len() || targets.len() != target_paths.len() {
        return Ok(true);
    }

    let mut protected_paths: Vec<String> = write_protected_paths
        .iter()
        .map(|e| posix_join(&[&canonical_work_dir, e]))
        .chain(roots.iter().filter_map(|r| r.path.clone()))
        .collect();
    let mut scanned: BTreeSet<String> = roots
        .iter()
        .filter(|r| r.flags.get(1).map(String::as_str) == Some("1"))
        .filter_map(|r| r.path.clone())
        .collect();
    let any_protected_path_exists = roots.iter().any(|r| r.flags[0] == "1");
    let targets: Vec<WriteProtectionTarget> = targets
        .iter()
        .map(|r| WriteProtectionTarget {
            path: r.path.clone().unwrap_or_default(),
            is_multi_link_file: r.flags[0] == "1",
        })
        .collect();

    let mut records = initial;
    let mut entry_count = 0usize;
    loop {
        if records.iter().any(|r| r.kind == "X") {
            return Ok(true);
        }
        for r in &records {
            if r.kind == "C" {
                entry_count += r.flags[0].parse::<usize>().unwrap_or(usize::MAX / 2);
            }
        }
        if entry_count > WRITE_PROTECTION_SCAN_LIMIT {
            return Ok(true);
        }
        let mut queue = Vec::new();
        for r in &records {
            if r.kind != "S" {
                continue;
            }
            let p = r.path.clone().unwrap_or_default();
            protected_paths.push(p.clone());
            if r.flags[0] == "1"
                && !scanned.contains(&p)
                && is_inside_write_protection_path(&canonical_work_dir, &p)
            {
                scanned.insert(p.clone());
                queue.push(p);
            }
        }
        if queue.is_empty() {
            break;
        }
        let script = [
            WRITE_PROTECTION_SHELL_FUNCTIONS.to_string(),
            format!("scan_budget={}", WRITE_PROTECTION_SCAN_LIMIT - entry_count),
            format!(
                "for scan_root in {}; do scan \"$scan_root\" || exit 3; done",
                quoted(&queue)
            ),
            "emit E".into(),
        ]
        .join("\n");
        match run_script(sandbox, &script, abort_signal.clone()).await? {
            Some(next) => records = next,
            None => return Ok(true),
        }
    }

    Ok(matches_write_protected_path(
        &targets,
        &protected_paths,
        any_protected_path_exists,
    ))
}

fn path_field_count(kind: &str) -> Option<usize> {
    match kind {
        "W" => Some(0),
        "P" => Some(2),
        "T" => Some(1),
        "S" => Some(1),
        _ => None,
    }
}

async fn run_script(
    sandbox: &dyn SandboxSession,
    script: &str,
    abort_signal: Option<CancellationToken>,
) -> Result<Option<Vec<Record>>, BoxError> {
    let result = sandbox
        .run(SandboxRunOptions {
            process: SandboxProcessOptions {
                command: format!("bash -c {}", shell_quote(script)),
                abort_signal,
                ..Default::default()
            },
            max_output_length: None,
        })
        .await?;
    if result.exit_code != 0 {
        return Ok(None);
    }
    let mut lines: Vec<&str> = result.stdout.split('\n').collect();
    if lines.pop() != Some("") || lines.last() != Some(&"E") {
        return Ok(None);
    }
    lines.pop();
    let mut records = Vec::new();
    for line in lines {
        let mut parts = line.split(' ');
        let kind = parts.next().unwrap_or_default().to_string();
        let fields: Vec<String> = parts.map(str::to_string).collect();
        let Some(flag_count) = path_field_count(&kind) else {
            if kind == "C"
                && fields.len() == 1
                && !fields[0].is_empty()
                && fields[0].bytes().all(|b| b.is_ascii_digit())
            {
                records.push(Record {
                    kind,
                    flags: fields,
                    path: None,
                });
                continue;
            }
            if kind == "X" && fields.is_empty() {
                records.push(Record {
                    kind,
                    flags: vec![],
                    path: None,
                });
                continue;
            }
            return Ok(None);
        };
        if fields.len() != flag_count + 1 {
            return Ok(None);
        }
        let flags = fields[..flag_count].to_vec();
        if !flags.iter().all(|f| f == "0" || f == "1") {
            return Ok(None);
        }
        let Some(decoded) = decode_record_path(&fields[flag_count]) else {
            return Ok(None);
        };
        if !decoded.starts_with('/') {
            return Ok(None);
        }
        records.push(Record {
            kind,
            flags,
            path: Some(decoded),
        });
    }
    Ok(Some(records))
}

fn decode_record_path(encoded: &str) -> Option<String> {
    if encoded.is_empty() {
        return None;
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .ok()?;
    String::from_utf8(bytes).ok()
}

// The cases run real bash against a temp directory, as the TS tests do.
#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::harness::sandbox::SandboxRunResult;
    use crate::harness::test_support::MockSandbox;
    use std::os::unix::fs::symlink;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn entries() -> Vec<String> {
        [".agents/skills", ".claude", "config.json"]
            .map(String::from)
            .to_vec()
    }

    struct TempDir(PathBuf);
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn temp_dir() -> TempDir {
        static N: AtomicUsize = AtomicUsize::new(0);
        let p = std::env::temp_dir().join(format!(
            "write-protection-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }

    fn local_sandbox(counter: Arc<AtomicUsize>) -> MockSandbox {
        MockSandbox::new(move |o, _| {
            counter.fetch_add(1, Ordering::SeqCst);
            let out = std::process::Command::new("bash")
                .arg("-c")
                .arg(&o.process.command)
                .output()
                .unwrap();
            Ok(SandboxRunResult {
                exit_code: out.status.code().unwrap_or(1),
                stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
                ..Default::default()
            })
        })
    }

    fn with_skill(w: &Path) {
        std::fs::create_dir_all(w.join(".agents/skills/test")).unwrap();
        std::fs::write(
            w.join(".agents/skills/test/SKILL.md"),
            "---\nname: test\ndescription: Test skill.\n---\n",
        )
        .unwrap();
    }

    type Setup = fn(&Path);

    // is-sandbox-write-protected-path.test.ts
    #[tokio::test]
    async fn reports_each_case_correctly() {
        let cases: Vec<(&str, Option<Setup>, Vec<&str>, bool)> = vec![
            (
                "an unprotected file",
                Some(with_skill),
                vec!["TEST.md"],
                false,
            ),
            (
                "a file inside a protected directory",
                Some(with_skill),
                vec![".agents/skills/test/SKILL.md"],
                true,
            ),
            (
                "a new file inside a missing protected directory",
                None,
                vec![".claude/settings.json"],
                true,
            ),
            ("a protected file", None, vec!["config.json"], true),
            (
                "a parent of a protected directory",
                Some(with_skill),
                vec![".agents"],
                true,
            ),
            (
                "a sibling sharing a protected name prefix",
                Some(with_skill),
                vec![".agents/skills-other/file.md"],
                false,
            ),
            (
                "any of several paths",
                Some(with_skill),
                vec!["TEST.md", ".agents/skills/test/SKILL.md"],
                true,
            ),
            (
                "an alias that points into a protected directory",
                Some(|w| {
                    with_skill(w);
                    symlink(".agents/skills", w.join("skills")).unwrap();
                }),
                vec!["skills/test/SKILL.md"],
                true,
            ),
            (
                "the target of a symlink inside a protected directory",
                Some(|w| {
                    with_skill(w);
                    std::fs::create_dir(w.join("shared")).unwrap();
                    symlink("../../shared", w.join(".agents/skills/shared")).unwrap();
                }),
                vec!["shared/new.md"],
                true,
            ),
            (
                "a target reached through nested symlinks",
                Some(|w| {
                    with_skill(w);
                    std::fs::create_dir(w.join("first")).unwrap();
                    std::fs::create_dir(w.join("second")).unwrap();
                    symlink("../../first", w.join(".agents/skills/first")).unwrap();
                    symlink("../second", w.join("first/second")).unwrap();
                }),
                vec!["second/new.md"],
                true,
            ),
            (
                "a file targeted by a symlink inside a protected directory",
                Some(|w| {
                    with_skill(w);
                    std::fs::write(w.join("notes.md"), "notes\n").unwrap();
                    symlink("../../notes.md", w.join(".agents/skills/notes.md")).unwrap();
                }),
                vec!["notes.md"],
                true,
            ),
            (
                "a hard-linked file while a protected path exists",
                Some(|w| {
                    with_skill(w);
                    std::fs::hard_link(w.join(".agents/skills/test/SKILL.md"), w.join("hard.md"))
                        .unwrap();
                }),
                vec!["hard.md"],
                true,
            ),
            (
                "a hard-linked file while no protected path exists",
                Some(|w| {
                    std::fs::write(w.join("one.md"), "one\n").unwrap();
                    std::fs::hard_link(w.join("one.md"), w.join("two.md")).unwrap();
                }),
                vec!["two.md"],
                false,
            ),
            (
                "any write while a protected directory holds a dangling symlink",
                Some(|w| {
                    with_skill(w);
                    symlink("missing", w.join(".agents/skills/dangling")).unwrap();
                }),
                vec!["TEST.md"],
                true,
            ),
            (
                "a looping symlink",
                Some(|w| {
                    symlink("loop-b", w.join("loop-a")).unwrap();
                    symlink("loop-a", w.join("loop-b")).unwrap();
                }),
                vec!["loop-a"],
                true,
            ),
            (
                "a normalized parent component after a symlink",
                Some(|w| {
                    with_skill(w);
                    std::fs::create_dir_all(w.join("deep/nested")).unwrap();
                    symlink("deep/nested", w.join("link")).unwrap();
                }),
                vec!["link/../.agents/skills/new.md"],
                true,
            ),
            (
                "any write when protected directories exceed the scan limit",
                Some(|w| {
                    std::fs::create_dir_all(w.join(".agents/skills/many")).unwrap();
                    for i in 0..2001 {
                        std::fs::write(w.join(format!(".agents/skills/many/{i}.md")), "").unwrap();
                    }
                }),
                vec!["TEST.md"],
                true,
            ),
        ];
        for (name, setup, paths, expected) in cases {
            let dir = temp_dir();
            if let Some(setup) = setup {
                setup(&dir.0);
            }
            let work_dir = dir.0.to_string_lossy().into_owned();
            let sandbox = local_sandbox(Arc::default());
            let paths: Vec<String> = paths.iter().map(|p| format!("{work_dir}/{p}")).collect();
            let got =
                is_sandbox_write_protected_path(&sandbox, &work_dir, &paths, &entries(), None)
                    .await
                    .unwrap();
            assert_eq!(got, expected, "{name}");
        }
    }

    #[tokio::test]
    async fn uses_a_single_sandbox_command_without_symlinked_directories() {
        let dir = temp_dir();
        let work_dir = std::fs::canonicalize(&dir.0)
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let counter = Arc::new(AtomicUsize::new(0));
        let sandbox = local_sandbox(counter.clone());
        assert!(
            !is_sandbox_write_protected_path(
                &sandbox,
                &work_dir,
                &[format!("{work_dir}/TEST.md")],
                &entries(),
                None
            )
            .await
            .unwrap()
        );
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn reports_failed_or_malformed_sandbox_output_as_protected() {
        let dir = temp_dir();
        let work_dir = dir.0.to_string_lossy().into_owned();
        for (code, stdout) in [(1, ""), (0, "W not-base64\nE\n"), (0, "")] {
            let sandbox = MockSandbox::with_result(SandboxRunResult {
                exit_code: code,
                stdout: stdout.into(),
                stderr: if code == 1 {
                    "failed".into()
                } else {
                    String::new()
                },
                ..Default::default()
            });
            assert!(
                is_sandbox_write_protected_path(
                    &sandbox,
                    &work_dir,
                    &[format!("{work_dir}/TEST.md")],
                    &entries(),
                    None
                )
                .await
                .unwrap()
            );
        }
    }

    #[tokio::test]
    async fn reports_empty_and_relative_paths_as_protected() {
        let dir = temp_dir();
        let work_dir = dir.0.to_string_lossy().into_owned();
        let counter = Arc::new(AtomicUsize::new(0));
        let sandbox = local_sandbox(counter.clone());
        for paths in [vec![], vec![String::new()], vec!["TEST.md".to_string()]] {
            assert!(
                is_sandbox_write_protected_path(&sandbox, &work_dir, &paths, &entries(), None)
                    .await
                    .unwrap()
            );
        }
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn rejects_invalid_protected_path_entries() {
        let dir = temp_dir();
        let work_dir = dir.0.to_string_lossy().into_owned();
        let sandbox = local_sandbox(Arc::default());
        assert!(
            is_sandbox_write_protected_path(
                &sandbox,
                &work_dir,
                &[format!("{work_dir}/TEST.md")],
                &["../outside".to_string()],
                None
            )
            .await
            .is_err()
        );
    }
}
