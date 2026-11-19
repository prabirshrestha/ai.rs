//! Port of `write-skills.ts`: materialize skills into a sandbox directory,
//! tracked by an ownership manifest so later writes only touch what the
//! harness owns.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;
use std::sync::Arc;

use regex::Regex;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use super::{locale_compare, plain_error, shell_quote};
use crate::harness::errors::HarnessResult;
use crate::harness::sandbox::{
    SandboxProcessOptions, SandboxReadTextFileOptions, SandboxRunOptions, SandboxSession,
    SandboxWriteTextFileOptions,
};
use crate::harness::spec::paths::{normalize_posix, posix_join};
use crate::harness::spec::tools::HarnessV1Skill;

const SKILLS_MANIFEST_FILENAME: &str = ".ai-sdk-harness-skills.json";
const SKILLS_MANIFEST_VERSION: u32 = 1;

/// `SkillFilePathMode`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SkillFilePathMode {
    #[default]
    Relative,
    StripLeadingSlashes,
}

/// `invalidSkillNameMessage`.
pub type InvalidSkillNameMessage = Arc<dyn Fn(&str) -> String + Send + Sync>;
/// `invalidSkillFilePathMessage({ skillName, filePath })`.
pub type InvalidSkillFilePathMessage = Arc<dyn Fn(&str, &str) -> String + Send + Sync>;

/// `WriteSkillsOptions`.
pub struct WriteSkillsOptions<'a> {
    pub sandbox: &'a dyn SandboxSession,
    pub home_path: String,
    pub skills_dir: String,
    pub skills: Vec<HarnessV1Skill>,
    pub abort_signal: Option<CancellationToken>,
    /// Defaults to `^[A-Za-z0-9._-]+$`.
    pub skill_name_pattern: Option<Regex>,
    pub invalid_skill_name_message: Option<InvalidSkillNameMessage>,
    pub file_path_mode: SkillFilePathMode,
    pub invalid_skill_file_path_message: Option<InvalidSkillFilePathMessage>,
    pub trailing_newline: bool,
}

impl<'a> WriteSkillsOptions<'a> {
    pub fn new(
        sandbox: &'a dyn SandboxSession,
        home_path: impl Into<String>,
        skills_dir: impl Into<String>,
        skills: Vec<HarnessV1Skill>,
    ) -> Self {
        Self {
            sandbox,
            home_path: home_path.into(),
            skills_dir: skills_dir.into(),
            skills,
            abort_signal: None,
            skill_name_pattern: None,
            invalid_skill_name_message: None,
            file_path_mode: SkillFilePathMode::Relative,
            invalid_skill_file_path_message: None,
            trailing_newline: false,
        }
    }
}

/// `WriteSkillsResult`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WriteSkillsResult {
    pub changed: bool,
    pub written: Vec<String>,
    pub removed: Vec<String>,
    pub unchanged: Vec<String>,
}

struct ProjectedSkill {
    name: String,
    hash: String,
    files: Vec<(String, String)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SkillsManifestEntry {
    name: String,
    hash: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ManifestState {
    Complete,
    Pending,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SkillsManifest {
    version: u32,
    state: ManifestState,
    skills: Vec<SkillsManifestEntry>,
}

fn sort_names(names: &mut [String]) {
    names.sort_by(|a, b| locale_compare(a, b));
}

/// `writeSkills`.
pub async fn write_skills(options: WriteSkillsOptions<'_>) -> HarnessResult<WriteSkillsResult> {
    let WriteSkillsOptions {
        sandbox,
        home_path,
        skills_dir,
        skills,
        abort_signal,
        skill_name_pattern,
        invalid_skill_name_message,
        file_path_mode,
        invalid_skill_file_path_message,
        trailing_newline,
    } = options;
    let pattern = skill_name_pattern
        .unwrap_or_else(|| Regex::new(r"^[A-Za-z0-9._-]+$").expect("valid regex"));
    let name_message = invalid_skill_name_message
        .unwrap_or_else(|| Arc::new(|name: &str| format!("Invalid skill name: {name}")));
    let path_message = invalid_skill_file_path_message.unwrap_or_else(|| {
        Arc::new(|skill: &str, path: &str| format!("Invalid skill file path for {skill}: {path}"))
    });

    let root_dir = resolve_skills_root_dir(&home_path, &skills_dir)?;
    let mut projected = skills
        .iter()
        .map(|skill| {
            project_skill(
                skill,
                &pattern,
                &*name_message,
                file_path_mode,
                &*path_message,
                trailing_newline,
            )
        })
        .collect::<HarnessResult<Vec<_>>>()?;
    projected.sort_by(|a, b| locale_compare(&a.name, &b.name));
    for pair in projected.windows(2) {
        if pair[0].name == pair[1].name {
            return Err(plain_error(format!(
                "Duplicate skill name: {}",
                pair[1].name
            )));
        }
    }

    let manifest_path = posix_join(&[&root_dir, SKILLS_MANIFEST_FILENAME]);
    let existing = read_skills_manifest(sandbox, &manifest_path, abort_signal.clone()).await?;
    let next_entries: Vec<SkillsManifestEntry> = projected
        .iter()
        .map(|s| SkillsManifestEntry {
            name: s.name.clone(),
            hash: s.hash.clone(),
        })
        .collect();
    let next_by_name: HashMap<&str, &SkillsManifestEntry> =
        next_entries.iter().map(|e| (e.name.as_str(), e)).collect();

    // Delete pending-manifest directories before rewriting the requested
    // skills; any listed directory may contain only part of its files.
    if let Some(manifest) = existing
        .as_ref()
        .filter(|m| m.state == ManifestState::Pending)
    {
        ensure_skills_directory(sandbox, &root_dir, abort_signal.clone()).await?;
        let recovery: Vec<String> = manifest.skills.iter().map(|s| s.name.clone()).collect();
        remove_skill_directories(sandbox, &root_dir, &recovery, abort_signal.clone()).await?;
        let mut removed: Vec<String> = recovery
            .iter()
            .filter(|n| !next_by_name.contains_key(n.as_str()))
            .cloned()
            .collect();
        sort_names(&mut removed);
        write_projected_skills(sandbox, &root_dir, projected.iter(), abort_signal.clone()).await?;
        write_skills_manifest(
            sandbox,
            &manifest_path,
            &SkillsManifest {
                version: SKILLS_MANIFEST_VERSION,
                state: ManifestState::Complete,
                skills: next_entries.clone(),
            },
            abort_signal,
        )
        .await?;
        return Ok(WriteSkillsResult {
            changed: !recovery.is_empty() || !projected.is_empty(),
            written: projected.iter().map(|s| s.name.clone()).collect(),
            removed,
            unchanged: vec![],
        });
    }

    let previous: Vec<SkillsManifestEntry> = existing
        .as_ref()
        .map(|m| m.skills.clone())
        .unwrap_or_default();
    let previous_by_name: HashMap<&str, &SkillsManifestEntry> =
        previous.iter().map(|e| (e.name.as_str(), e)).collect();
    let mut removed: Vec<String> = previous
        .iter()
        .filter(|e| !next_by_name.contains_key(e.name.as_str()))
        .map(|e| e.name.clone())
        .collect();
    sort_names(&mut removed);
    let same_hash = |s: &ProjectedSkill| {
        previous_by_name
            .get(s.name.as_str())
            .is_some_and(|p| p.hash == s.hash)
    };
    let mut written: Vec<String> = projected
        .iter()
        .filter(|s| !same_hash(s))
        .map(|s| s.name.clone())
        .collect();
    sort_names(&mut written);
    let mut unchanged: Vec<String> = projected
        .iter()
        .filter(|s| same_hash(s))
        .map(|s| s.name.clone())
        .collect();
    sort_names(&mut unchanged);
    let changed = !removed.is_empty() || !written.is_empty();

    if !changed && existing.is_some() {
        return Ok(WriteSkillsResult {
            changed: false,
            written,
            removed,
            unchanged,
        });
    }

    let previously_owned: BTreeSet<&str> = previous.iter().map(|e| e.name.as_str()).collect();
    for name in &written {
        if previously_owned.contains(name.as_str()) {
            continue;
        }
        assert_skill_directory_available(sandbox, &root_dir, name, abort_signal.clone()).await?;
    }

    ensure_skills_directory(sandbox, &root_dir, abort_signal.clone()).await?;

    let mut pending_names: Vec<String> = previous
        .iter()
        .map(|e| e.name.clone())
        .chain(next_entries.iter().map(|e| e.name.clone()))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    sort_names(&mut pending_names);
    write_skills_manifest(
        sandbox,
        &manifest_path,
        &SkillsManifest {
            version: SKILLS_MANIFEST_VERSION,
            state: ManifestState::Pending,
            skills: pending_names
                .iter()
                .map(|name| SkillsManifestEntry {
                    name: name.clone(),
                    hash: next_by_name
                        .get(name.as_str())
                        .or_else(|| previous_by_name.get(name.as_str()))
                        .map(|e| e.hash.clone())
                        .unwrap_or_default(),
                })
                .collect(),
        },
        abort_signal.clone(),
    )
    .await?;

    let mut to_remove = removed.clone();
    to_remove.extend(
        written
            .iter()
            .filter(|n| previously_owned.contains(n.as_str()))
            .cloned(),
    );
    remove_skill_directories(sandbox, &root_dir, &to_remove, abort_signal.clone()).await?;
    let written_set: BTreeSet<&str> = written.iter().map(String::as_str).collect();
    write_projected_skills(
        sandbox,
        &root_dir,
        projected
            .iter()
            .filter(|s| written_set.contains(s.name.as_str())),
        abort_signal.clone(),
    )
    .await?;
    write_skills_manifest(
        sandbox,
        &manifest_path,
        &SkillsManifest {
            version: SKILLS_MANIFEST_VERSION,
            state: ManifestState::Complete,
            skills: next_entries.clone(),
        },
        abort_signal,
    )
    .await?;

    Ok(WriteSkillsResult {
        changed,
        written,
        removed,
        unchanged,
    })
}

async fn run_sandbox_command(
    sandbox: &dyn SandboxSession,
    command: String,
    abort_signal: Option<CancellationToken>,
    error_message: String,
) -> HarnessResult<()> {
    let result = sandbox
        .run(SandboxRunOptions {
            process: SandboxProcessOptions {
                command,
                abort_signal,
                ..Default::default()
            },
            max_output_length: None,
        })
        .await?;
    if result.exit_code != 0 {
        let stderr = if result.stderr.is_empty() {
            String::new()
        } else {
            format!(": {}", result.stderr)
        };
        return Err(plain_error(format!(
            "{error_message} (exit {}){stderr}",
            result.exit_code
        )));
    }
    Ok(())
}

async fn ensure_skills_directory(
    sandbox: &dyn SandboxSession,
    root_dir: &str,
    abort_signal: Option<CancellationToken>,
) -> HarnessResult<()> {
    run_sandbox_command(
        sandbox,
        format!("mkdir -p {}", shell_quote(root_dir)),
        abort_signal,
        format!("Failed to create skills directory: {root_dir}"),
    )
    .await
}

fn project_skill(
    skill: &HarnessV1Skill,
    pattern: &Regex,
    name_message: &(dyn Fn(&str) -> String + Send + Sync),
    mode: SkillFilePathMode,
    path_message: &(dyn Fn(&str, &str) -> String + Send + Sync),
    trailing_newline: bool,
) -> HarnessResult<ProjectedSkill> {
    let name = &skill.name;
    if !pattern.is_match(name) || name == "." || name == ".." {
        return Err(plain_error(name_message(name)));
    }
    let mut files: BTreeMap<String, String> = BTreeMap::new();
    files.insert(
        "SKILL.md".into(),
        render_skill_file(skill, trailing_newline),
    );
    for file in skill.files.iter().flatten() {
        files.insert(
            normalize_skill_file_path(name, &file.path, mode, path_message)?,
            file.content.clone(),
        );
    }
    let mut files: Vec<(String, String)> = files.into_iter().collect();
    files.sort_by(|a, b| locale_compare(&a.0, &b.0));
    let mut ctx = ring::digest::Context::new(&ring::digest::SHA256);
    for (path, content) in &files {
        ctx.update(path.len().to_string().as_bytes());
        ctx.update(b":");
        ctx.update(path.as_bytes());
        ctx.update(content.len().to_string().as_bytes());
        ctx.update(b":");
        ctx.update(content.as_bytes());
    }
    let mut hash = String::with_capacity(64);
    for b in ctx.finish().as_ref() {
        let _ = write!(hash, "{b:02x}");
    }
    Ok(ProjectedSkill {
        name: name.clone(),
        hash,
        files,
    })
}

fn is_safe_manifest_skill_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
        && name != "."
        && name != ".."
}

fn is_skills_manifest(value: &serde_json::Value) -> bool {
    let Some(m) = value.as_object() else {
        return false;
    };
    if m.get("version") != Some(&serde_json::json!(SKILLS_MANIFEST_VERSION))
        || !matches!(
            m.get("state").and_then(|s| s.as_str()),
            Some("complete" | "pending")
        )
    {
        return false;
    }
    let Some(skills) = m.get("skills").and_then(|s| s.as_array()) else {
        return false;
    };
    let mut names = BTreeSet::new();
    for entry in skills {
        let Some(entry) = entry.as_object() else {
            return false;
        };
        let (Some(name), Some(hash)) = (
            entry.get("name").and_then(|v| v.as_str()),
            entry.get("hash").and_then(|v| v.as_str()),
        ) else {
            return false;
        };
        let hash_ok =
            hash.len() == 64 && hash.bytes().all(|b| matches!(b, b'a'..=b'f' | b'0'..=b'9'));
        if !is_safe_manifest_skill_name(name) || !hash_ok || !names.insert(name.to_string()) {
            return false;
        }
    }
    true
}

async fn read_skills_manifest(
    sandbox: &dyn SandboxSession,
    manifest_path: &str,
    abort_signal: Option<CancellationToken>,
) -> HarnessResult<Option<SkillsManifest>> {
    let content = sandbox
        .read_text_file(SandboxReadTextFileOptions {
            path: manifest_path.into(),
            abort_signal,
            ..Default::default()
        })
        .await?;
    let Some(content) = content else {
        return Ok(None);
    };
    let invalid = || {
        plain_error(format!(
            "Invalid AI SDK harness skills manifest: {manifest_path}"
        ))
    };
    let value: serde_json::Value = serde_json::from_str(&content).map_err(|_| invalid())?;
    if !is_skills_manifest(&value) {
        return Err(invalid());
    }
    serde_json::from_value(value)
        .map(Some)
        .map_err(|_| invalid())
}

async fn write_skills_manifest(
    sandbox: &dyn SandboxSession,
    manifest_path: &str,
    manifest: &SkillsManifest,
    abort_signal: Option<CancellationToken>,
) -> HarnessResult<()> {
    let temporary = format!("{manifest_path}.tmp");
    sandbox
        .write_text_file(SandboxWriteTextFileOptions {
            path: temporary.clone(),
            content: format!(
                "{}\n",
                serde_json::to_string_pretty(manifest).expect("serializable")
            ),
            abort_signal: abort_signal.clone(),
            encoding: None,
        })
        .await?;
    run_sandbox_command(
        sandbox,
        format!(
            "mv -f {} {}",
            shell_quote(&temporary),
            shell_quote(manifest_path)
        ),
        abort_signal,
        format!("Failed to update skills manifest: {manifest_path}"),
    )
    .await
}

async fn assert_skill_directory_available(
    sandbox: &dyn SandboxSession,
    root_dir: &str,
    skill_name: &str,
    abort_signal: Option<CancellationToken>,
) -> HarnessResult<()> {
    let skill_dir = posix_join(&[root_dir, skill_name]);
    let result = sandbox
        .run(SandboxRunOptions {
            process: SandboxProcessOptions {
                command: format!("test ! -e {}", shell_quote(&skill_dir)),
                abort_signal,
                ..Default::default()
            },
            max_output_length: None,
        })
        .await?;
    if result.exit_code != 0 {
        return Err(plain_error(format!(
            "Cannot write harness skill '{skill_name}': {skill_dir} already exists and is not owned by the AI SDK harness."
        )));
    }
    Ok(())
}

async fn remove_skill_directories(
    sandbox: &dyn SandboxSession,
    root_dir: &str,
    names: &[String],
    abort_signal: Option<CancellationToken>,
) -> HarnessResult<()> {
    if names.is_empty() {
        return Ok(());
    }
    let mut unique: Vec<String> = names
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    sort_names(&mut unique);
    let directories = unique
        .iter()
        .map(|n| shell_quote(&posix_join(&[root_dir, n])))
        .collect::<Vec<_>>()
        .join(" ");
    run_sandbox_command(
        sandbox,
        format!("rm -rf -- {directories}"),
        abort_signal,
        format!("Failed to replace harness skills in: {root_dir}"),
    )
    .await
}

async fn write_projected_skills<'s>(
    sandbox: &dyn SandboxSession,
    root_dir: &str,
    skills: impl Iterator<Item = &'s ProjectedSkill>,
    abort_signal: Option<CancellationToken>,
) -> HarnessResult<()> {
    for skill in skills {
        let skill_dir = posix_join(&[root_dir, &skill.name]);
        for (path, content) in &skill.files {
            sandbox
                .write_text_file(SandboxWriteTextFileOptions {
                    path: posix_join(&[&skill_dir, path]),
                    content: content.clone(),
                    abort_signal: abort_signal.clone(),
                    encoding: None,
                })
                .await?;
        }
    }
    Ok(())
}

fn normalize_skill_file_path(
    skill_name: &str,
    file_path: &str,
    mode: SkillFilePathMode,
    message: &(dyn Fn(&str, &str) -> String + Send + Sync),
) -> HarnessResult<String> {
    let normalized = match mode {
        SkillFilePathMode::StripLeadingSlashes => file_path.trim_start_matches('/').to_string(),
        SkillFilePathMode::Relative => normalize_posix(file_path),
    };
    let relative = mode == SkillFilePathMode::Relative;
    let invalid = normalized.is_empty()
        || (relative && normalized == ".")
        || normalized.starts_with("../")
        || normalized.contains("/../")
        || normalized.ends_with("/..")
        || normalized == ".."
        || (relative && normalized.starts_with('/'));
    if invalid {
        return Err(plain_error(message(skill_name, file_path)));
    }
    Ok(normalized)
}

fn render_skill_file(skill: &HarnessV1Skill, trailing_newline: bool) -> String {
    let content = format!(
        "---\nname: {}\ndescription: {}\n---\n\n{}",
        skill.name, skill.description, skill.content
    );
    if trailing_newline {
        format!("{content}\n")
    } else {
        content
    }
}

fn is_win32_absolute(path: &str) -> bool {
    let b = path.as_bytes();
    path.starts_with('\\')
        || path.starts_with('/')
        || (b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':')
}

fn resolve_skills_root_dir(home_path: &str, skills_dir: &str) -> HarnessResult<String> {
    if home_path.trim().is_empty() {
        return Err(plain_error(
            "Invalid homePath: expected a non-empty string.",
        ));
    }
    let json = |s: &str| serde_json::to_string(s).unwrap_or_default();
    if !home_path.starts_with('/') {
        return Err(plain_error(format!(
            "Invalid homePath {}: expected an absolute POSIX path.",
            json(home_path)
        )));
    }
    let invalid = || {
        plain_error(format!(
            "Invalid skillsDir {}: expected a relative POSIX path without traversal.",
            json(skills_dir)
        ))
    };
    if skills_dir.trim().is_empty() {
        return Err(invalid());
    }
    let contains_traversal = skills_dir.split(['\\', '/']).any(|s| s == "..");
    let normalized = normalize_posix(skills_dir.trim());
    let no_trailing = normalized.trim_end_matches('/').to_string();
    if skills_dir.contains('\\')
        || skills_dir.starts_with('/')
        || is_win32_absolute(skills_dir)
        || contains_traversal
        || no_trailing.is_empty()
        || no_trailing == "."
        || normalized.starts_with("../")
        || normalized.contains("/../")
        || normalized.ends_with("/..")
    {
        return Err(invalid());
    }
    let root_dir = posix_join(&[home_path, &no_trailing]);
    let home = normalize_posix(home_path);
    let home = home.trim_end_matches('/');
    let inside = root_dir
        .strip_prefix(home)
        .is_some_and(|rest| rest.starts_with('/') && rest.len() > 1);
    if !inside {
        return Err(plain_error(format!(
            "Invalid skillsDir {}: must be a subpath within homePath {}.",
            json(skills_dir),
            json(home_path)
        )));
    }
    Ok(root_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::sandbox::SandboxRunResult;
    use crate::harness::spec::tools::HarnessV1SkillFile;
    use crate::harness::test_support::{MockSandbox, MockState, quoted_values};

    fn handle(o: &SandboxRunOptions, s: &mut MockState) -> HarnessResult<SandboxRunResult> {
        let command = &o.process.command;
        let ok = SandboxRunResult::default();
        if command.starts_with("test ! -e ") {
            let target = &quoted_values(command)[0];
            let exists = s.directories.contains(target)
                || s.files.keys().any(|p| p.starts_with(&format!("{target}/")));
            return Ok(SandboxRunResult {
                exit_code: if exists { 1 } else { 0 },
                ..ok
            });
        }
        if command.starts_with("mv -f ") {
            let q = quoted_values(command);
            let content = s.files.remove(&q[0]).unwrap();
            s.files.insert(q[1].clone(), content);
        }
        if command.starts_with("rm -rf -- ") {
            for target in quoted_values(command) {
                s.directories.remove(&target);
                s.files
                    .retain(|p, _| !(p == &target || p.starts_with(&format!("{target}/"))));
            }
        }
        Ok(ok)
    }

    fn make_sandbox() -> MockSandbox {
        MockSandbox::new(handle)
    }

    fn demo_skill() -> HarnessV1Skill {
        HarnessV1Skill {
            name: "demo".into(),
            description: "Demo skill.".into(),
            content: "Use reference.md.".into(),
            files: Some(vec![HarnessV1SkillFile {
                path: "reference.md".into(),
                content: "# Reference".into(),
            }]),
        }
    }

    fn skill(name: &str, d: &str, c: &str) -> HarnessV1Skill {
        HarnessV1Skill {
            name: name.into(),
            description: d.into(),
            content: c.into(),
            files: None,
        }
    }

    async fn write(
        sandbox: &MockSandbox,
        skills: Vec<HarnessV1Skill>,
    ) -> HarnessResult<WriteSkillsResult> {
        write_skills(WriteSkillsOptions::new(
            sandbox,
            "/home/user",
            ".agents/skills",
            skills,
        ))
        .await
    }

    fn result(
        changed: bool,
        written: &[&str],
        removed: &[&str],
        unchanged: &[&str],
    ) -> WriteSkillsResult {
        let v = |x: &[&str]| x.iter().map(|s| s.to_string()).collect();
        WriteSkillsResult {
            changed,
            written: v(written),
            removed: v(removed),
            unchanged: v(unchanged),
        }
    }

    // write-skills.test.ts
    #[tokio::test]
    async fn writes_skills_and_a_completed_ownership_manifest() {
        let sb = make_sandbox();
        let r = write(&sb, vec![demo_skill()]).await.unwrap();
        assert_eq!(r, result(true, &["demo"], &[], &[]));
        let s = sb.state.lock();
        assert_eq!(
            s.files["/home/user/.agents/skills/demo/SKILL.md"],
            "---\nname: demo\ndescription: Demo skill.\n---\n\nUse reference.md."
        );
        assert_eq!(
            s.files["/home/user/.agents/skills/demo/reference.md"],
            "# Reference"
        );
        let manifest: serde_json::Value =
            serde_json::from_str(&s.files["/home/user/.agents/skills/.ai-sdk-harness-skills.json"])
                .unwrap();
        assert_eq!(
            manifest,
            serde_json::json!({
                "skills": [{ "hash": "c14ffcc71e9d9c39fb21765d4ac61acd2fc428e973a224e6f20d76f23aa3954b", "name": "demo" }],
                "state": "complete",
                "version": 1
            })
        );
    }

    #[tokio::test]
    async fn does_not_write_or_delete_anything_when_all_skill_hashes_match() {
        let sb = make_sandbox();
        write(&sb, vec![demo_skill()]).await.unwrap();
        let (writes, runs) = {
            let s = sb.state.lock();
            (s.writes.len(), s.runs.len())
        };
        let r = write(&sb, vec![demo_skill()]).await.unwrap();
        assert_eq!(r, result(false, &[], &[], &["demo"]));
        let s = sb.state.lock();
        assert_eq!(s.writes.len(), writes);
        assert_eq!(s.runs.len(), runs);
    }

    #[tokio::test]
    async fn replaces_changed_skills_and_removes_only_manifest_owned_skills() {
        let sb = make_sandbox();
        {
            let mut s = sb.state.lock();
            s.directories
                .insert("/home/user/.agents/skills/external".into());
            s.files.insert(
                "/home/user/.agents/skills/external/SKILL.md".into(),
                "external".into(),
            );
        }
        write(&sb, vec![demo_skill(), skill("old", "Old.", "Old.")])
            .await
            .unwrap();
        let mut changed = demo_skill();
        changed.content = "Changed.".into();
        let r = write(&sb, vec![changed]).await.unwrap();
        assert_eq!(r, result(true, &["demo"], &["old"], &[]));
        let s = sb.state.lock();
        assert!(s.files["/home/user/.agents/skills/demo/SKILL.md"].contains("Changed."));
        assert!(
            !s.files
                .contains_key("/home/user/.agents/skills/old/SKILL.md")
        );
        assert_eq!(
            s.files["/home/user/.agents/skills/external/SKILL.md"],
            "external"
        );
    }

    #[tokio::test]
    async fn sorts_result_and_manifest_entries_by_skill_name() {
        let sb = make_sandbox();
        let r = write(
            &sb,
            vec![skill("zeta", "Z.", "Z."), skill("alpha", "A.", "A.")],
        )
        .await
        .unwrap();
        assert_eq!(r.written, vec!["alpha", "zeta"]);
        let s = sb.state.lock();
        let manifest: serde_json::Value =
            serde_json::from_str(&s.files["/home/user/.agents/skills/.ai-sdk-harness-skills.json"])
                .unwrap();
        let names: Vec<_> = manifest["skills"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["name"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(names, vec!["alpha", "zeta"]);
    }

    #[tokio::test]
    async fn rejects_collisions_with_skill_directories_not_owned_by_the_manifest() {
        let sb = make_sandbox();
        sb.state
            .lock()
            .directories
            .insert("/home/user/.agents/skills/demo".into());
        let err = write(&sb, vec![demo_skill()]).await.unwrap_err();
        assert!(err.to_string().contains("already exists and is not owned"));
        assert!(
            !sb.state
                .lock()
                .files
                .contains_key("/home/user/.agents/skills/.ai-sdk-harness-skills.json")
        );
    }

    #[tokio::test]
    async fn recovers_a_pending_update_by_clearing_every_recorded_directory() {
        let sb = make_sandbox();
        {
            let mut s = sb.state.lock();
            s.files.insert(
                "/home/user/.agents/skills/.ai-sdk-harness-skills.json".into(),
                serde_json::json!({
                    "version": 1, "state": "pending",
                    "skills": [{ "name": "demo", "hash": "a".repeat(64) }, { "name": "old", "hash": "b".repeat(64) }]
                })
                .to_string(),
            );
            s.files.insert(
                "/home/user/.agents/skills/demo/partial.txt".into(),
                "partial".into(),
            );
            s.files.insert(
                "/home/user/.agents/skills/old/SKILL.md".into(),
                "old".into(),
            );
        }
        let r = write(&sb, vec![demo_skill()]).await.unwrap();
        assert_eq!(r, result(true, &["demo"], &["old"], &[]));
        let s = sb.state.lock();
        assert!(
            !s.files
                .contains_key("/home/user/.agents/skills/demo/partial.txt")
        );
        assert!(
            !s.files
                .contains_key("/home/user/.agents/skills/old/SKILL.md")
        );
        assert!(
            s.files
                .contains_key("/home/user/.agents/skills/demo/SKILL.md")
        );
    }

    #[tokio::test]
    async fn validates_all_skill_paths_before_writing() {
        let sb = make_sandbox();
        let mut bad = demo_skill();
        bad.files = Some(vec![HarnessV1SkillFile {
            path: "../reference.md".into(),
            content: "# Reference".into(),
        }]);
        let err = write(&sb, vec![bad]).await.unwrap_err();
        assert!(err.to_string().contains("Invalid skill file path"));
        let s = sb.state.lock();
        assert!(s.runs.is_empty());
        assert!(s.writes.is_empty());
    }

    #[tokio::test]
    async fn supports_stricter_skill_name_patterns() {
        let sb = make_sandbox();
        let mut o = WriteSkillsOptions::new(
            &sb,
            "/home/user",
            ".agents/skills",
            vec![skill("Demo", "Demo.", "Demo.")],
        );
        o.skill_name_pattern = Some(Regex::new(r"^[a-z0-9]([a-z0-9-]{0,62}[a-z0-9])?$").unwrap());
        o.invalid_skill_name_message = Some(Arc::new(|name: &str| {
            format!(
                "Invalid deepagents skill name '{name}': must be lowercase alphanumeric with hyphens, 1-64 chars."
            )
        }));
        let err = write_skills(o).await.unwrap_err();
        assert!(
            err.to_string()
                .contains("Invalid deepagents skill name 'Demo'")
        );
    }

    #[tokio::test]
    async fn can_strip_leading_slashes_from_attached_file_paths() {
        let sb = make_sandbox();
        let mut s = skill("demo", "Demo.", "Demo.");
        s.files = Some(vec![HarnessV1SkillFile {
            path: "/reference.md".into(),
            content: "# Reference".into(),
        }]);
        let mut o = WriteSkillsOptions::new(&sb, "/home/user", ".agents/skills", vec![s]);
        o.file_path_mode = SkillFilePathMode::StripLeadingSlashes;
        write_skills(o).await.unwrap();
        assert_eq!(
            sb.state.lock().files["/home/user/.agents/skills/demo/reference.md"],
            "# Reference"
        );
    }

    #[tokio::test]
    async fn rejects_non_absolute_or_empty_home_path() {
        let sb = make_sandbox();
        let err = write_skills(WriteSkillsOptions::new(
            &sb,
            "relative/home",
            ".agents/skills",
            vec![demo_skill()],
        ))
        .await
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("Invalid homePath \"relative/home\": expected an absolute POSIX path.")
        );
        let err = write_skills(WriteSkillsOptions::new(
            &sb,
            "   ",
            ".agents/skills",
            vec![demo_skill()],
        ))
        .await
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("Invalid homePath: expected a non-empty string.")
        );
    }

    #[tokio::test]
    async fn rejects_invalid_skills_dirs() {
        for skills_dir in [
            "/skills",
            "C:\\skills",
            "skills\\sub",
            "../skills",
            "foo/../../bar",
            "",
            "   ",
            ".",
            "./",
        ] {
            let sb = make_sandbox();
            let err = write_skills(WriteSkillsOptions::new(
                &sb,
                "/home/user",
                skills_dir,
                vec![demo_skill()],
            ))
            .await
            .unwrap_err();
            assert!(
                err.to_string()
                    .contains("expected a relative POSIX path without traversal"),
                "{skills_dir:?}: {err}"
            );
        }
    }
}
