use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::types::Skill;

const MAX_NAME_LENGTH: usize = 64;
const MAX_DESCRIPTION_LENGTH: usize = 1024;
const IGNORE_FILE_NAMES: [&str; 3] = [".gitignore", ".ignore", ".fdignore"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillDiagnosticCode {
    FileInfoFailed,
    ListFailed,
    ReadFailed,
    ParseFailed,
    InvalidMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillDiagnostic {
    #[serde(rename = "type")]
    pub kind: String,
    pub code: SkillDiagnosticCode,
    pub message: String,
    pub path: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillLoadResult {
    pub skills: Vec<Skill>,
    pub diagnostics: Vec<SkillDiagnostic>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourcedSkillInput<TSource> {
    pub path: PathBuf,
    pub source: TSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourcedSkill<TSource> {
    pub skill: Skill,
    pub source: TSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourcedSkillDiagnostic<TSource> {
    pub diagnostic: SkillDiagnostic,
    pub source: TSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourcedSkillLoadResult<TSource> {
    pub skills: Vec<SourcedSkill<TSource>>,
    pub diagnostics: Vec<SourcedSkillDiagnostic<TSource>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileKind {
    File,
    Directory,
    Symlink,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FileInfo {
    name: String,
    path: PathBuf,
    kind: FileKind,
}

#[derive(Debug, Clone, Default)]
struct IgnoreMatcher {
    rules: Vec<IgnoreRule>,
}

#[derive(Debug, Clone)]
struct IgnoreRule {
    pattern: String,
    negated: bool,
}

impl IgnoreMatcher {
    fn add(&mut self, pattern: String) {
        let (negated, pattern) = pattern
            .strip_prefix('!')
            .map(|pattern| (true, pattern.to_string()))
            .unwrap_or((false, pattern));
        self.rules.push(IgnoreRule { pattern, negated });
    }

    fn ignores(&self, path: &str) -> bool {
        let mut ignored = false;
        for rule in &self.rules {
            if rule.matches(path) {
                ignored = !rule.negated;
            }
        }
        ignored
    }
}

impl IgnoreRule {
    fn matches(&self, path: &str) -> bool {
        let pattern = self.pattern.trim();
        if pattern.is_empty() {
            return false;
        }
        if let Some(dir) = pattern.strip_suffix('/') {
            return path == dir || path.starts_with(&format!("{dir}/"));
        }
        if pattern.contains('/') {
            return path == pattern || path.starts_with(&format!("{pattern}/"));
        }
        path.rsplit('/').any(|segment| segment == pattern)
    }
}

pub fn format_skill_invocation(skill: &Skill, additional_instructions: Option<&str>) -> String {
    let skill_block = format!(
        "<skill name=\"{}\" location=\"{}\">\nReferences are relative to {}.\n\n{}\n</skill>",
        skill.name,
        skill.file_path,
        dirname_env_path(&skill.file_path),
        skill.content
    );
    additional_instructions
        .map(|instructions| format!("{skill_block}\n\n{instructions}"))
        .unwrap_or(skill_block)
}

pub async fn load_skills(cwd: impl AsRef<Path>, dirs: &[impl AsRef<Path>]) -> SkillLoadResult {
    let cwd = cwd.as_ref();
    let mut skills = Vec::new();
    let mut diagnostics = Vec::new();

    for dir in dirs {
        let root_path = absolute_path(cwd, dir.as_ref());
        let root_info = match file_info(&root_path).await {
            Ok(info) => info,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(err) => {
                diagnostics.push(diagnostic(
                    SkillDiagnosticCode::FileInfoFailed,
                    err.to_string(),
                    root_path,
                ));
                continue;
            }
        };
        if resolve_kind(&root_info, &mut diagnostics).await != Some(FileKind::Directory) {
            continue;
        }
        let result = load_skills_from_dir_internal(
            &root_info.path,
            true,
            &mut IgnoreMatcher::default(),
            &root_info.path,
        )
        .await;
        skills.extend(result.skills);
        diagnostics.extend(result.diagnostics);
    }

    SkillLoadResult {
        skills,
        diagnostics,
    }
}

pub async fn load_sourced_skills<TSource: Clone>(
    cwd: impl AsRef<Path>,
    inputs: &[SourcedSkillInput<TSource>],
) -> SourcedSkillLoadResult<TSource> {
    let mut skills = Vec::new();
    let mut diagnostics = Vec::new();
    for input in inputs {
        let result = load_skills(cwd.as_ref(), &[input.path.as_path()]).await;
        skills.extend(result.skills.into_iter().map(|skill| SourcedSkill {
            skill,
            source: input.source.clone(),
        }));
        diagnostics.extend(result.diagnostics.into_iter().map(|diagnostic| {
            SourcedSkillDiagnostic {
                diagnostic,
                source: input.source.clone(),
            }
        }));
    }
    SourcedSkillLoadResult {
        skills,
        diagnostics,
    }
}

async fn load_skills_from_dir_internal(
    dir: &Path,
    include_root_files: bool,
    ignore_matcher: &mut IgnoreMatcher,
    root_dir: &Path,
) -> SkillLoadResult {
    let mut skills = Vec::new();
    let mut diagnostics = Vec::new();

    let dir_info = match file_info(dir).await {
        Ok(info) => info,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return SkillLoadResult {
                skills,
                diagnostics,
            };
        }
        Err(err) => {
            diagnostics.push(diagnostic(
                SkillDiagnosticCode::FileInfoFailed,
                err.to_string(),
                dir,
            ));
            return SkillLoadResult {
                skills,
                diagnostics,
            };
        }
    };
    if resolve_kind(&dir_info, &mut diagnostics).await != Some(FileKind::Directory) {
        return SkillLoadResult {
            skills,
            diagnostics,
        };
    }

    add_ignore_rules(ignore_matcher, dir, root_dir, &mut diagnostics).await;

    let mut entries = match list_dir_infos(dir, &mut diagnostics).await {
        Some(entries) => entries,
        None => {
            return SkillLoadResult {
                skills,
                diagnostics,
            };
        }
    };

    for entry in &entries {
        if entry.name != "SKILL.md" {
            continue;
        }
        if resolve_kind(entry, &mut diagnostics).await != Some(FileKind::File) {
            continue;
        }
        let rel_path = relative_env_path(root_dir, &entry.path);
        if ignore_matcher.ignores(&rel_path) {
            continue;
        }
        let result = load_skill_from_file(&entry.path).await;
        if let Some(skill) = result.skill {
            skills.push(skill);
        }
        diagnostics.extend(result.diagnostics);
        return SkillLoadResult {
            skills,
            diagnostics,
        };
    }

    entries.sort_by(|a, b| a.name.cmp(&b.name));
    for entry in entries {
        if entry.name.starts_with('.') || entry.name == "node_modules" {
            continue;
        }
        let Some(kind) = resolve_kind(&entry, &mut diagnostics).await else {
            continue;
        };
        let rel_path = relative_env_path(root_dir, &entry.path);
        let ignore_path = if kind == FileKind::Directory {
            format!("{rel_path}/")
        } else {
            rel_path
        };
        if ignore_matcher.ignores(&ignore_path) {
            continue;
        }

        if kind == FileKind::Directory {
            let result = Box::pin(load_skills_from_dir_internal(
                &entry.path,
                false,
                ignore_matcher,
                root_dir,
            ))
            .await;
            skills.extend(result.skills);
            diagnostics.extend(result.diagnostics);
            continue;
        }

        if kind != FileKind::File || !include_root_files || !entry.name.ends_with(".md") {
            continue;
        }
        let result = load_skill_from_file(&entry.path).await;
        if let Some(skill) = result.skill {
            skills.push(skill);
        }
        diagnostics.extend(result.diagnostics);
    }

    SkillLoadResult {
        skills,
        diagnostics,
    }
}

async fn add_ignore_rules(
    ignore_matcher: &mut IgnoreMatcher,
    dir: &Path,
    root_dir: &Path,
    diagnostics: &mut Vec<SkillDiagnostic>,
) {
    let relative_dir = relative_env_path(root_dir, dir);
    let prefix = if relative_dir.is_empty() {
        String::new()
    } else {
        format!("{relative_dir}/")
    };

    for filename in IGNORE_FILE_NAMES {
        let ignore_path = dir.join(filename);
        let info = match file_info(&ignore_path).await {
            Ok(info) => info,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(err) => {
                diagnostics.push(diagnostic(
                    SkillDiagnosticCode::FileInfoFailed,
                    err.to_string(),
                    ignore_path,
                ));
                continue;
            }
        };
        if info.kind != FileKind::File {
            continue;
        }
        let content = match tokio::fs::read_to_string(&ignore_path).await {
            Ok(content) => content,
            Err(err) => {
                diagnostics.push(diagnostic(
                    SkillDiagnosticCode::ReadFailed,
                    err.to_string(),
                    ignore_path,
                ));
                continue;
            }
        };
        for line in content
            .lines()
            .filter_map(|line| prefix_ignore_pattern(line, &prefix))
        {
            ignore_matcher.add(line);
        }
    }
}

fn prefix_ignore_pattern(line: &str, prefix: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with('#') && !trimmed.starts_with("\\#") {
        return None;
    }

    let mut pattern = line.to_string();
    let mut negated = false;
    if let Some(rest) = pattern.strip_prefix('!') {
        negated = true;
        pattern = rest.to_string();
    } else if let Some(rest) = pattern.strip_prefix("\\!") {
        pattern = rest.to_string();
    }
    if let Some(rest) = pattern.strip_prefix('/') {
        pattern = rest.to_string();
    }
    let prefixed = if prefix.is_empty() {
        pattern
    } else {
        format!("{prefix}{pattern}")
    };
    Some(if negated {
        format!("!{prefixed}")
    } else {
        prefixed
    })
}

struct SkillFileLoadResult {
    skill: Option<Skill>,
    diagnostics: Vec<SkillDiagnostic>,
}

async fn load_skill_from_file(file_path: &Path) -> SkillFileLoadResult {
    let mut diagnostics = Vec::new();
    let raw_content = match tokio::fs::read_to_string(file_path).await {
        Ok(content) => content,
        Err(err) => {
            diagnostics.push(diagnostic(
                SkillDiagnosticCode::ReadFailed,
                err.to_string(),
                file_path,
            ));
            return SkillFileLoadResult {
                skill: None,
                diagnostics,
            };
        }
    };
    let parsed = match parse_frontmatter(&raw_content) {
        Ok(parsed) => parsed,
        Err(err) => {
            diagnostics.push(diagnostic(SkillDiagnosticCode::ParseFailed, err, file_path));
            return SkillFileLoadResult {
                skill: None,
                diagnostics,
            };
        }
    };

    let skill_dir = dirname_env_path(&file_path.to_string_lossy());
    let parent_dir_name = basename_env_path(&skill_dir);
    for error in validate_description(parsed.description.as_deref()) {
        diagnostics.push(diagnostic(
            SkillDiagnosticCode::InvalidMetadata,
            error,
            file_path,
        ));
    }

    let name = parsed
        .name
        .filter(|name| !name.is_empty())
        .unwrap_or(parent_dir_name.clone());
    for error in validate_name(&name, &parent_dir_name) {
        diagnostics.push(diagnostic(
            SkillDiagnosticCode::InvalidMetadata,
            error,
            file_path,
        ));
    }

    let Some(description) = parsed
        .description
        .filter(|description| !description.trim().is_empty())
    else {
        return SkillFileLoadResult {
            skill: None,
            diagnostics,
        };
    };

    SkillFileLoadResult {
        skill: Some(Skill {
            name,
            description,
            content: parsed.body,
            file_path: file_path.to_string_lossy().into_owned(),
            disable_model_invocation: parsed.disable_model_invocation,
        }),
        diagnostics,
    }
}

struct ParsedSkillFrontmatter {
    name: Option<String>,
    description: Option<String>,
    disable_model_invocation: bool,
    body: String,
}

fn parse_frontmatter(content: &str) -> Result<ParsedSkillFrontmatter, String> {
    let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
    if !normalized.starts_with("---") {
        return Ok(ParsedSkillFrontmatter {
            name: None,
            description: None,
            disable_model_invocation: false,
            body: normalized,
        });
    }
    let Some(end_index) = normalized[3..].find("\n---").map(|index| index + 3) else {
        return Ok(ParsedSkillFrontmatter {
            name: None,
            description: None,
            disable_model_invocation: false,
            body: normalized,
        });
    };
    let frontmatter_start = if normalized.as_bytes().get(3) == Some(&b'\n') {
        4
    } else {
        3
    };
    let yaml = if end_index >= frontmatter_start {
        &normalized[frontmatter_start..end_index]
    } else {
        ""
    };
    let body = normalized[end_index + 4..].trim().to_string();
    let frontmatter = serde_yaml::from_str::<serde_yaml::Value>(yaml)
        .map_err(|err| err.to_string())?
        .as_mapping()
        .cloned()
        .unwrap_or_default();
    let name = frontmatter
        .get(serde_yaml::Value::String("name".to_string()))
        .and_then(|value| value.as_str())
        .map(ToString::to_string);
    let description = frontmatter
        .get(serde_yaml::Value::String("description".to_string()))
        .and_then(|value| value.as_str())
        .map(ToString::to_string);
    let disable_model_invocation = frontmatter
        .get(serde_yaml::Value::String(
            "disable-model-invocation".to_string(),
        ))
        .and_then(|value| value.as_bool())
        == Some(true);
    Ok(ParsedSkillFrontmatter {
        name,
        description,
        disable_model_invocation,
        body,
    })
}

fn validate_name(name: &str, parent_dir_name: &str) -> Vec<String> {
    let mut errors = Vec::new();
    if name != parent_dir_name {
        errors.push(format!(
            "name \"{name}\" does not match parent directory \"{parent_dir_name}\""
        ));
    }
    if name.len() > MAX_NAME_LENGTH {
        errors.push(format!(
            "name exceeds {MAX_NAME_LENGTH} characters ({})",
            name.len()
        ));
    }
    if !name
        .chars()
        .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-')
    {
        errors.push(
            "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)"
                .to_string(),
        );
    }
    if name.starts_with('-') || name.ends_with('-') {
        errors.push("name must not start or end with a hyphen".to_string());
    }
    if name.contains("--") {
        errors.push("name must not contain consecutive hyphens".to_string());
    }
    errors
}

fn validate_description(description: Option<&str>) -> Vec<String> {
    match description {
        None => vec!["description is required".to_string()],
        Some(description) if description.trim().is_empty() => {
            vec!["description is required".to_string()]
        }
        Some(description) if description.len() > MAX_DESCRIPTION_LENGTH => {
            vec![format!(
                "description exceeds {MAX_DESCRIPTION_LENGTH} characters ({})",
                description.len()
            )]
        }
        Some(_) => Vec::new(),
    }
}

async fn resolve_kind(info: &FileInfo, diagnostics: &mut Vec<SkillDiagnostic>) -> Option<FileKind> {
    match info.kind {
        FileKind::File | FileKind::Directory => Some(info.kind),
        FileKind::Symlink => {
            let canonical = match tokio::fs::canonicalize(&info.path).await {
                Ok(path) => path,
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => return None,
                Err(err) => {
                    diagnostics.push(diagnostic(
                        SkillDiagnosticCode::FileInfoFailed,
                        err.to_string(),
                        &info.path,
                    ));
                    return None;
                }
            };
            match file_info(&canonical).await {
                Ok(target) if matches!(target.kind, FileKind::File | FileKind::Directory) => {
                    Some(target.kind)
                }
                Ok(_) => None,
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
                Err(err) => {
                    diagnostics.push(diagnostic(
                        SkillDiagnosticCode::FileInfoFailed,
                        err.to_string(),
                        &info.path,
                    ));
                    None
                }
            }
        }
    }
}

async fn list_dir_infos(
    dir: &Path,
    diagnostics: &mut Vec<SkillDiagnostic>,
) -> Option<Vec<FileInfo>> {
    let mut entries = match tokio::fs::read_dir(dir).await {
        Ok(entries) => entries,
        Err(err) => {
            diagnostics.push(diagnostic(
                SkillDiagnosticCode::ListFailed,
                err.to_string(),
                dir,
            ));
            return None;
        }
    };
    let mut infos = Vec::new();
    loop {
        match entries.next_entry().await {
            Ok(Some(entry)) => match file_info(&entry.path()).await {
                Ok(info) => infos.push(info),
                Err(err) => diagnostics.push(diagnostic(
                    SkillDiagnosticCode::FileInfoFailed,
                    err.to_string(),
                    entry.path(),
                )),
            },
            Ok(None) => return Some(infos),
            Err(err) => {
                diagnostics.push(diagnostic(
                    SkillDiagnosticCode::ListFailed,
                    err.to_string(),
                    dir,
                ));
                return None;
            }
        }
    }
}

async fn file_info(path: &Path) -> std::io::Result<FileInfo> {
    let metadata = tokio::fs::symlink_metadata(path).await?;
    let file_type = metadata.file_type();
    let kind = if file_type.is_file() {
        FileKind::File
    } else if file_type.is_dir() {
        FileKind::Directory
    } else if file_type.is_symlink() {
        FileKind::Symlink
    } else {
        FileKind::File
    };
    Ok(FileInfo {
        name: path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_string(),
        path: path.to_path_buf(),
        kind,
    })
}

fn absolute_path(cwd: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
}

fn join_env_path(base: &str, child: &str) -> String {
    format!(
        "{}/{}",
        base.trim_end_matches('/'),
        child.trim_start_matches('/')
    )
}

fn dirname_env_path(path: &str) -> String {
    let normalized = path.trim_end_matches('/');
    let slash_index = normalized.rfind('/');
    match slash_index {
        Some(0) | None => "/".to_string(),
        Some(index) => normalized[..index].to_string(),
    }
}

fn basename_env_path(path: &str) -> String {
    let normalized = path.trim_end_matches('/');
    normalized
        .rsplit('/')
        .next()
        .unwrap_or(normalized)
        .to_string()
}

fn relative_env_path(root: &Path, path: &Path) -> String {
    let root = root.to_string_lossy();
    let path = path.to_string_lossy();
    let normalized_root = root.trim_end_matches('/');
    let normalized_path = path.trim_end_matches('/');
    if normalized_path == normalized_root {
        return String::new();
    }
    normalized_path
        .strip_prefix(&format!("{normalized_root}/"))
        .map(ToString::to_string)
        .unwrap_or_else(|| normalized_path.trim_start_matches('/').to_string())
}

fn diagnostic(
    code: SkillDiagnosticCode,
    message: impl Into<String>,
    path: impl AsRef<Path>,
) -> SkillDiagnostic {
    SkillDiagnostic {
        kind: "warning".to_string(),
        code,
        message: message.into(),
        path: path.as_ref().to_string_lossy().into_owned(),
    }
}

#[allow(dead_code)]
fn _join_env_path_for_parity(base: &str, child: &str) -> String {
    join_env_path(base, child)
}
