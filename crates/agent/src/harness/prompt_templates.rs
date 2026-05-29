use std::path::{Path, PathBuf};

use regex::Regex;
use serde::{Deserialize, Serialize};

use super::types::PromptTemplate;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptTemplateDiagnosticCode {
    FileInfoFailed,
    ListFailed,
    ReadFailed,
    ParseFailed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptTemplateDiagnostic {
    #[serde(rename = "type")]
    pub kind: String,
    pub code: PromptTemplateDiagnosticCode,
    pub message: String,
    pub path: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptTemplateLoadResult {
    pub prompt_templates: Vec<PromptTemplate>,
    pub diagnostics: Vec<PromptTemplateDiagnostic>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourcedPromptTemplateInput<TSource> {
    pub path: PathBuf,
    pub source: TSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourcedPromptTemplate<TSource> {
    pub prompt_template: PromptTemplate,
    pub source: TSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourcedPromptTemplateDiagnostic<TSource> {
    pub diagnostic: PromptTemplateDiagnostic,
    pub source: TSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourcedPromptTemplateLoadResult<TSource> {
    pub prompt_templates: Vec<SourcedPromptTemplate<TSource>>,
    pub diagnostics: Vec<SourcedPromptTemplateDiagnostic<TSource>>,
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

pub async fn load_prompt_templates(
    cwd: impl AsRef<Path>,
    paths: &[impl AsRef<Path>],
) -> PromptTemplateLoadResult {
    let cwd = cwd.as_ref();
    let mut prompt_templates = Vec::new();
    let mut diagnostics = Vec::new();

    for path in paths {
        let input_path = path.as_ref();
        let absolute = absolute_path(cwd, input_path);
        let info = match file_info(&absolute).await {
            Ok(info) => info,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(err) => {
                diagnostics.push(diagnostic(
                    PromptTemplateDiagnosticCode::FileInfoFailed,
                    err.to_string(),
                    absolute,
                ));
                continue;
            }
        };

        let Some(kind) = resolve_kind(&info, &mut diagnostics).await else {
            continue;
        };
        if kind == FileKind::Directory {
            let result = load_templates_from_dir(&info.path).await;
            prompt_templates.extend(result.prompt_templates);
            diagnostics.extend(result.diagnostics);
        } else if kind == FileKind::File && info.name.ends_with(".md") {
            let result = load_template_from_file(&info.path).await;
            if let Some(prompt_template) = result.prompt_template {
                prompt_templates.push(prompt_template);
            }
            diagnostics.extend(result.diagnostics);
        }
    }

    PromptTemplateLoadResult {
        prompt_templates,
        diagnostics,
    }
}

pub async fn load_sourced_prompt_templates<TSource: Clone>(
    cwd: impl AsRef<Path>,
    inputs: &[SourcedPromptTemplateInput<TSource>],
) -> SourcedPromptTemplateLoadResult<TSource> {
    let mut prompt_templates = Vec::new();
    let mut diagnostics = Vec::new();

    for input in inputs {
        let result = load_prompt_templates(cwd.as_ref(), &[input.path.as_path()]).await;
        prompt_templates.extend(result.prompt_templates.into_iter().map(|prompt_template| {
            SourcedPromptTemplate {
                prompt_template,
                source: input.source.clone(),
            }
        }));
        diagnostics.extend(result.diagnostics.into_iter().map(|diagnostic| {
            SourcedPromptTemplateDiagnostic {
                diagnostic,
                source: input.source.clone(),
            }
        }));
    }

    SourcedPromptTemplateLoadResult {
        prompt_templates,
        diagnostics,
    }
}

async fn load_templates_from_dir(dir: &Path) -> PromptTemplateLoadResult {
    let mut prompt_templates = Vec::new();
    let mut diagnostics = Vec::new();
    let mut entries = match tokio::fs::read_dir(dir).await {
        Ok(entries) => entries,
        Err(err) => {
            diagnostics.push(diagnostic(
                PromptTemplateDiagnosticCode::ListFailed,
                err.to_string(),
                dir,
            ));
            return PromptTemplateLoadResult {
                prompt_templates,
                diagnostics,
            };
        }
    };

    let mut infos = Vec::new();
    while let Ok(Some(entry)) = entries.next_entry().await {
        match file_info(&entry.path()).await {
            Ok(info) => infos.push(info),
            Err(err) => diagnostics.push(diagnostic(
                PromptTemplateDiagnosticCode::FileInfoFailed,
                err.to_string(),
                entry.path(),
            )),
        }
    }
    infos.sort_by(|a, b| a.name.cmp(&b.name));

    for entry in infos {
        let Some(kind) = resolve_kind(&entry, &mut diagnostics).await else {
            continue;
        };
        if kind != FileKind::File || !entry.name.ends_with(".md") {
            continue;
        }
        let result = load_template_from_file(&entry.path).await;
        if let Some(prompt_template) = result.prompt_template {
            prompt_templates.push(prompt_template);
        }
        diagnostics.extend(result.diagnostics);
    }

    PromptTemplateLoadResult {
        prompt_templates,
        diagnostics,
    }
}

struct TemplateFileLoadResult {
    prompt_template: Option<PromptTemplate>,
    diagnostics: Vec<PromptTemplateDiagnostic>,
}

async fn load_template_from_file(file_path: &Path) -> TemplateFileLoadResult {
    let mut diagnostics = Vec::new();
    let raw_content = match tokio::fs::read_to_string(file_path).await {
        Ok(content) => content,
        Err(err) => {
            diagnostics.push(diagnostic(
                PromptTemplateDiagnosticCode::ReadFailed,
                err.to_string(),
                file_path,
            ));
            return TemplateFileLoadResult {
                prompt_template: None,
                diagnostics,
            };
        }
    };

    let parsed = match parse_frontmatter(&raw_content) {
        Ok(parsed) => parsed,
        Err(err) => {
            diagnostics.push(diagnostic(
                PromptTemplateDiagnosticCode::ParseFailed,
                err,
                file_path,
            ));
            return TemplateFileLoadResult {
                prompt_template: None,
                diagnostics,
            };
        }
    };

    let first_line = parsed.body.lines().find(|line| !line.trim().is_empty());
    let mut description = parsed.description.unwrap_or_default();
    if description.is_empty()
        && let Some(first_line) = first_line
    {
        description = first_line.chars().take(60).collect();
        if first_line.chars().count() > 60 {
            description.push_str("...");
        }
    }

    TemplateFileLoadResult {
        prompt_template: Some(PromptTemplate {
            name: basename_env_path(file_path)
                .trim_end_matches(".md")
                .to_string(),
            description: if description.is_empty() {
                None
            } else {
                Some(description)
            },
            content: parsed.body,
        }),
        diagnostics,
    }
}

struct ParsedFrontmatter {
    description: Option<String>,
    body: String,
}

fn parse_frontmatter(content: &str) -> Result<ParsedFrontmatter, String> {
    let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
    if !normalized.starts_with("---") {
        return Ok(ParsedFrontmatter {
            description: None,
            body: normalized,
        });
    }
    let Some(end_index) = normalized[3..].find("\n---").map(|index| index + 3) else {
        return Ok(ParsedFrontmatter {
            description: None,
            body: normalized,
        });
    };
    let frontmatter_start = if normalized.as_bytes().get(3) == Some(&b'\n') {
        4
    } else {
        3
    };
    let yaml = if end_index >= frontmatter_start {
        normalized[frontmatter_start..end_index].to_string()
    } else {
        String::new()
    };
    let body = normalized[end_index + 4..].trim().to_string();
    let frontmatter = serde_yaml::from_str::<serde_yaml::Value>(&yaml)
        .map_err(|err| err.to_string())?
        .as_mapping()
        .cloned()
        .unwrap_or_default();
    let description = frontmatter
        .get(serde_yaml::Value::String("description".to_string()))
        .and_then(|value| value.as_str())
        .map(ToString::to_string);
    Ok(ParsedFrontmatter { description, body })
}

async fn resolve_kind(
    info: &FileInfo,
    diagnostics: &mut Vec<PromptTemplateDiagnostic>,
) -> Option<FileKind> {
    match info.kind {
        FileKind::File | FileKind::Directory => Some(info.kind),
        FileKind::Symlink => {
            let canonical = match tokio::fs::canonicalize(&info.path).await {
                Ok(path) => path,
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => return None,
                Err(err) => {
                    diagnostics.push(diagnostic(
                        PromptTemplateDiagnosticCode::FileInfoFailed,
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
                        PromptTemplateDiagnosticCode::FileInfoFailed,
                        err.to_string(),
                        &info.path,
                    ));
                    None
                }
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

fn basename_env_path(path: &Path) -> String {
    let normalized = path.to_string_lossy();
    let normalized = normalized.trim_end_matches('/');
    normalized
        .rsplit('/')
        .next()
        .unwrap_or(normalized)
        .to_string()
}

fn diagnostic(
    code: PromptTemplateDiagnosticCode,
    message: impl Into<String>,
    path: impl AsRef<Path>,
) -> PromptTemplateDiagnostic {
    PromptTemplateDiagnostic {
        kind: "warning".to_string(),
        code,
        message: message.into(),
        path: path.as_ref().to_string_lossy().into_owned(),
    }
}

pub fn parse_command_args(args_string: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut in_quote = None;

    for character in args_string.chars() {
        if let Some(quote) = in_quote {
            if character == quote {
                in_quote = None;
            } else {
                current.push(character);
            }
        } else if character == '"' || character == '\'' {
            in_quote = Some(character);
        } else if character == ' ' || character == '\t' {
            if !current.is_empty() {
                args.push(std::mem::take(&mut current));
            }
        } else {
            current.push(character);
        }
    }
    if !current.is_empty() {
        args.push(current);
    }
    args
}

pub fn substitute_args(content: &str, args: &[String]) -> String {
    let positional = Regex::new(r"\$(\d+)").expect("valid positional regex");
    let slices = Regex::new(r"\$\{@:(\d+)(?::(\d+))?\}").expect("valid slice regex");

    let result = positional
        .replace_all(content, |captures: &regex::Captures<'_>| {
            let index = captures[1].parse::<usize>().unwrap_or_default();
            args.get(index.saturating_sub(1))
                .cloned()
                .unwrap_or_default()
        })
        .to_string();

    let result = slices
        .replace_all(&result, |captures: &regex::Captures<'_>| {
            let start = captures[1].parse::<usize>().unwrap_or(1).saturating_sub(1);
            let Some(length) = captures
                .get(2)
                .and_then(|value| value.as_str().parse::<usize>().ok())
            else {
                return args.get(start..).unwrap_or_default().join(" ");
            };
            args.get(start..start.saturating_add(length))
                .unwrap_or_default()
                .join(" ")
        })
        .to_string();

    let all_args = args.join(" ");
    result
        .replace("$ARGUMENTS", &all_args)
        .replace("$@", &all_args)
}

pub fn format_prompt_template_invocation(template: &PromptTemplate, args: &[String]) -> String {
    substitute_args(&template.content, args)
}
