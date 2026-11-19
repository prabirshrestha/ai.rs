//! Port of `harness-pi/src/pi-paths.ts`.

use serde::{Deserialize, Serialize};

use crate::harness::error::HarnessError;
use crate::harness::paths::{is_absolute_posix, normalize_posix, posix_join};

/// `path.posix.resolve(path)` for the absolute or root-relative paths used
/// here: normalized, without a trailing separator.
fn resolve(path: &str) -> String {
    let absolute = if is_absolute_posix(path) {
        path.to_string()
    } else {
        format!("/{path}")
    };
    let normalized = normalize_posix(&absolute);
    if normalized.len() > 1 {
        normalized.trim_end_matches('/').to_string()
    } else {
        normalized
    }
}

/// `path.posix.relative(from, to)`.
///
/// [rust] Relative inputs resolve against `/` instead of `process.cwd()`.
pub(crate) fn relative_posix(from: &str, to: &str) -> String {
    let from = resolve(from);
    let to = resolve(to);
    if from == to {
        return String::new();
    }
    let from_parts: Vec<&str> = from.split('/').filter(|part| !part.is_empty()).collect();
    let to_parts: Vec<&str> = to.split('/').filter(|part| !part.is_empty()).collect();
    let common = from_parts
        .iter()
        .zip(&to_parts)
        .take_while(|(left, right)| left == right)
        .count();
    let mut parts: Vec<&str> = vec![".."; from_parts.len() - common];
    parts.extend(&to_parts[common..]);
    parts.join("/")
}

/// `path.posix.basename(path)`.
pub(crate) fn basename_posix(path: &str) -> String {
    let trimmed = path.trim_end_matches('/');
    trimmed
        .rsplit_once('/')
        .map_or(trimmed, |(_, base)| base)
        .to_string()
}

/// `path.posix.join(...)`, keeping Node's `.` for an empty result.
pub(crate) fn join_posix(parts: &[&str]) -> String {
    posix_join(parts)
}

fn is_inside_posix_path(parent: &str, candidate: &str) -> bool {
    let relative = relative_posix(parent, candidate);
    relative.is_empty()
        || (relative != ".." && !relative.starts_with("../") && !is_absolute_posix(&relative))
}

/// `PiFileToolPathPolicy`: additional sandbox paths exposed to Pi's native
/// file tools.
///
/// The session workspace remains read-write and harness-provided skills
/// remain read-only. Writable roots are also readable, and denied roots
/// take precedence over every allowed root. This policy does not restrict
/// Pi's `bash` tool.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PiFileToolPathPolicy {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub readable_roots: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub writable_roots: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub denied_roots: Option<Vec<String>>,
}

/// `PiReadablePathRoot`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PiReadablePathRoot {
    pub sandbox_dir: String,
}

/// `CreatePiPathMapperOptions`.
#[derive(Debug, Clone, Default)]
pub struct CreatePiPathMapperOptions {
    pub sandbox_work_dir: String,
    pub readable_roots: Option<Vec<PiReadablePathRoot>>,
    pub writable_roots: Option<Vec<String>>,
    pub denied_roots: Option<Vec<String>>,
    pub home_dir: Option<String>,
}

/// `PiPathMapper`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PiPathMapper {
    /// The sandbox-side working directory where tools actually operate.
    pub sandbox_work_dir: String,
    /// Additional sandbox roots accepted by read-only tools.
    pub readable_sandbox_roots: Vec<String>,
    /// Additional sandbox roots accepted by mutating tools.
    pub writable_sandbox_roots: Vec<String>,
    /// Sandbox roots rejected by every native file tool.
    pub denied_sandbox_roots: Vec<String>,
    home_dir: Option<String>,
}

/// `createPiPathMapper(options)`.
pub fn create_pi_path_mapper(
    options: CreatePiPathMapperOptions,
) -> Result<PiPathMapper, HarnessError> {
    let normalize_roots = |roots: Option<Vec<String>>, kind: &str| {
        roots
            .unwrap_or_default()
            .into_iter()
            .map(|root| {
                if !is_absolute_posix(&root) {
                    return Err(HarnessError::new(format!(
                        "Pi {kind} root must be an absolute sandbox path: {root}"
                    )));
                }
                Ok(normalize_posix(&root))
            })
            .collect::<Result<Vec<_>, _>>()
    };
    Ok(PiPathMapper {
        sandbox_work_dir: normalize_posix(&options.sandbox_work_dir),
        readable_sandbox_roots: normalize_roots(
            options
                .readable_roots
                .map(|roots| roots.into_iter().map(|root| root.sandbox_dir).collect()),
            "readable",
        )?,
        writable_sandbox_roots: normalize_roots(options.writable_roots, "writable")?,
        denied_sandbox_roots: normalize_roots(options.denied_roots, "denied")?,
        home_dir: options.home_dir,
    })
}

impl PiPathMapper {
    fn expand_home(&self, input_path: &str) -> String {
        match &self.home_dir {
            Some(home_dir) if input_path == "~" || input_path.starts_with("~/") => {
                join_posix(&[home_dir, &input_path[1..]])
            }
            _ => input_path.to_string(),
        }
    }

    fn absolute_sandbox_path(&self, input_path: &str) -> String {
        if is_absolute_posix(input_path) {
            input_path.to_string()
        } else {
            join_posix(&[&self.sandbox_work_dir, input_path])
        }
    }

    fn writable_areas(&self) -> Vec<&String> {
        std::iter::once(&self.sandbox_work_dir)
            .chain(&self.writable_sandbox_roots)
            .collect()
    }

    fn readable_areas(&self) -> Vec<&String> {
        let mut areas = self.writable_areas();
        areas.extend(&self.readable_sandbox_roots);
        areas
    }

    fn assert_allowed(
        &self,
        sandbox_path: &str,
        input_path: &str,
        allowed_roots: &[&String],
        escape_message: &str,
    ) -> Result<String, HarnessError> {
        let normalized = normalize_posix(sandbox_path);
        if self
            .denied_sandbox_roots
            .iter()
            .any(|root| is_inside_posix_path(root, &normalized))
        {
            return Err(HarnessError::new(format!(
                "Pi path is inside a denied root: {input_path}"
            )));
        }
        if !allowed_roots
            .iter()
            .any(|root| is_inside_posix_path(root, &normalized))
        {
            return Err(HarnessError::new(format!(
                "Pi path {escape_message}: {input_path}"
            )));
        }
        Ok(normalized)
    }

    /// `toSandboxPath(inputPath)`: translate a tool path (relative,
    /// absolute, or `~`-prefixed) to a normalized writable sandbox path.
    pub fn to_sandbox_path(&self, input_path: &str) -> Result<String, HarnessError> {
        self.assert_allowed(
            &self.absolute_sandbox_path(&self.expand_home(input_path)),
            input_path,
            &self.writable_areas(),
            "escapes the workspace",
        )
    }

    /// `toReadableSandboxPath(inputPath)`.
    pub fn to_readable_sandbox_path(&self, input_path: &str) -> Result<String, HarnessError> {
        self.assert_allowed(
            &self.absolute_sandbox_path(&self.expand_home(input_path)),
            input_path,
            &self.readable_areas(),
            "escapes the workspace",
        )
    }

    /// `assertSandboxPath(inputPath)`.
    pub fn assert_sandbox_path(&self, input_path: &str) -> Result<String, HarnessError> {
        self.assert_allowed(
            input_path,
            input_path,
            &self.writable_areas(),
            "escapes the workspace",
        )
    }

    /// `assertReadableSandboxPath(inputPath)`.
    pub fn assert_readable_sandbox_path(&self, input_path: &str) -> Result<String, HarnessError> {
        self.assert_allowed(
            input_path,
            input_path,
            &self.readable_areas(),
            "escapes the readable roots",
        )
    }

    /// `isWorkspacePath(inputPath)`.
    pub fn is_workspace_path(&self, input_path: &str) -> bool {
        is_inside_posix_path(&self.sandbox_work_dir, &normalize_posix(input_path))
    }

    /// `relativeDeniedRootsUnder(sandboxDir)`.
    pub fn relative_denied_roots_under(&self, sandbox_dir: &str) -> Vec<String> {
        self.denied_sandbox_roots
            .iter()
            .filter(|root| is_inside_posix_path(sandbox_dir, root))
            .map(|root| relative_posix(sandbox_dir, root))
            .collect()
    }

    /// `toRelativePath(inputPath)`.
    pub fn to_relative_path(&self, input_path: &str) -> String {
        let sandbox_path = if is_absolute_posix(input_path) {
            normalize_posix(input_path)
        } else {
            join_posix(&[&self.sandbox_work_dir, input_path])
        };
        let relative = relative_posix(&self.sandbox_work_dir, &sandbox_path);
        if relative.is_empty() {
            ".".into()
        } else {
            relative
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SANDBOX_WORK_DIR: &str = "/sandbox/work/session";

    fn mapper(options: CreatePiPathMapperOptions) -> PiPathMapper {
        create_pi_path_mapper(CreatePiPathMapperOptions {
            sandbox_work_dir: SANDBOX_WORK_DIR.into(),
            ..options
        })
        .unwrap()
    }

    fn base() -> PiPathMapper {
        mapper(Default::default())
    }

    fn roots(values: &[&str]) -> Option<Vec<String>> {
        Some(values.iter().map(|value| value.to_string()).collect())
    }

    fn readable(values: &[&str]) -> Option<Vec<PiReadablePathRoot>> {
        Some(
            values
                .iter()
                .map(|value| PiReadablePathRoot {
                    sandbox_dir: value.to_string(),
                })
                .collect(),
        )
    }

    fn policy_mapper(policy: PiFileToolPathPolicy) -> Result<PiPathMapper, HarnessError> {
        create_pi_path_mapper(CreatePiPathMapperOptions {
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
    }

    fn err(result: Result<String, HarnessError>) -> String {
        result.unwrap_err().message().to_string()
    }

    #[test]
    fn translates_relative_paths_to_sandbox_posix_paths() {
        assert_eq!(
            base().to_sandbox_path("src/foo.ts").unwrap(),
            format!("{SANDBOX_WORK_DIR}/src/foo.ts")
        );
    }

    #[test]
    fn handles_the_workspace_root_itself() {
        assert_eq!(base().to_sandbox_path(".").unwrap(), SANDBOX_WORK_DIR);
    }

    #[test]
    fn returns_already_sandbox_absolute_paths_inside_the_sandbox_root_unchanged() {
        let path = format!("{SANDBOX_WORK_DIR}/already/here.ts");
        assert_eq!(base().to_sandbox_path(&path).unwrap(), path);
    }

    #[test]
    fn throws_when_a_path_escapes_the_workspace() {
        assert!(err(base().to_sandbox_path("../escape.ts")).contains("escapes the workspace"));
    }

    #[test]
    fn accepts_two_dot_prefixed_names_inside_the_workspace() {
        let mapper = base();
        let path = format!("{SANDBOX_WORK_DIR}/..notes/file.txt");
        assert_eq!(mapper.to_sandbox_path("..notes/file.txt").unwrap(), path);
        assert_eq!(mapper.assert_sandbox_path(&path).unwrap(), path);
        assert!(
            err(mapper.assert_sandbox_path("/sandbox/work/escape.txt"))
                .contains("escapes the workspace")
        );
    }

    #[test]
    fn allows_configured_read_only_sandbox_roots_for_readable_paths() {
        let mapper = mapper(CreatePiPathMapperOptions {
            readable_roots: readable(&["/home/vercel-sandbox/.agents/skills"]),
            ..Default::default()
        });
        let path = "/home/vercel-sandbox/.agents/skills/weather-codes/SKILL.md";
        assert_eq!(mapper.to_readable_sandbox_path(path).unwrap(), path);
        assert!(err(mapper.to_sandbox_path(path)).contains("escapes the workspace"));
    }

    #[test]
    fn refuses_readable_paths_inside_a_denied_root() {
        let mapper = mapper(CreatePiPathMapperOptions {
            readable_roots: readable(&["/home/vercel-sandbox"]),
            denied_roots: roots(&["/home/vercel-sandbox/.credentials"]),
            ..Default::default()
        });
        assert!(
            err(mapper.to_readable_sandbox_path("/home/vercel-sandbox/.credentials/token"))
                .contains("inside a denied root")
        );
        assert!(
            err(mapper.assert_readable_sandbox_path("/home/vercel-sandbox/.credentials"))
                .contains("inside a denied root")
        );
        assert_eq!(
            mapper
                .to_readable_sandbox_path("/home/vercel-sandbox/.credentials-old/a")
                .unwrap(),
            "/home/vercel-sandbox/.credentials-old/a"
        );
    }

    #[test]
    fn refuses_workspace_paths_inside_a_denied_root() {
        let mapper = mapper(CreatePiPathMapperOptions {
            denied_roots: roots(&[&format!("{SANDBOX_WORK_DIR}/.private")]),
            ..Default::default()
        });
        assert!(
            err(mapper.to_readable_sandbox_path(".private/journal"))
                .contains("inside a denied root")
        );
        assert!(err(mapper.to_sandbox_path(".private/journal")).contains("inside a denied root"));
        assert!(
            err(mapper.assert_sandbox_path(&format!("{SANDBOX_WORK_DIR}/.private/journal")))
                .contains("inside a denied root")
        );
        assert_eq!(
            mapper.to_sandbox_path("src/foo.ts").unwrap(),
            format!("{SANDBOX_WORK_DIR}/src/foo.ts")
        );
    }

    #[test]
    fn denies_two_dot_prefixed_descendants_and_includes_two_dot_prefixed_denied_roots() {
        let mapper = mapper(CreatePiPathMapperOptions {
            denied_roots: roots(&[
                &format!("{SANDBOX_WORK_DIR}/private"),
                &format!("{SANDBOX_WORK_DIR}/..private"),
            ]),
            ..Default::default()
        });
        assert!(
            err(mapper.to_readable_sandbox_path("private/..backup/token.txt"))
                .contains("inside a denied root")
        );
        assert!(
            err(mapper.to_readable_sandbox_path(&format!("{SANDBOX_WORK_DIR}/..private/token.txt")))
                .contains("inside a denied root")
        );
        assert_eq!(
            mapper.relative_denied_roots_under(SANDBOX_WORK_DIR),
            vec!["private", "..private"]
        );
        assert_eq!(
            mapper
                .to_readable_sandbox_path("private-copy/..backup/token.txt")
                .unwrap(),
            format!("{SANDBOX_WORK_DIR}/private-copy/..backup/token.txt")
        );
    }

    #[test]
    fn expands_tilde_against_the_configured_home_directory() {
        let mapper = mapper(CreatePiPathMapperOptions {
            home_dir: Some("/home/vercel-sandbox".into()),
            readable_roots: readable(&["/home/vercel-sandbox"]),
            ..Default::default()
        });
        assert_eq!(
            mapper.to_readable_sandbox_path("~/notes.txt").unwrap(),
            "/home/vercel-sandbox/notes.txt"
        );
        assert_eq!(
            mapper.to_readable_sandbox_path("~").unwrap(),
            "/home/vercel-sandbox"
        );
        assert!(err(mapper.to_sandbox_path("~/notes.txt")).contains("escapes the workspace"));
    }

    #[test]
    fn treats_tilde_as_a_workspace_relative_name_without_a_home_directory() {
        assert_eq!(
            base().to_readable_sandbox_path("~/notes.txt").unwrap(),
            format!("{SANDBOX_WORK_DIR}/~/notes.txt")
        );
    }

    #[test]
    fn allows_additional_read_only_roots_without_making_them_writable() {
        let mapper = policy_mapper(PiFileToolPathPolicy {
            readable_roots: roots(&["/mnt/reference/../reference"]),
            ..Default::default()
        })
        .unwrap();
        let path = "/mnt/reference/catalog.json";
        assert_eq!(mapper.to_readable_sandbox_path(path).unwrap(), path);
        assert_eq!(mapper.assert_readable_sandbox_path(path).unwrap(), path);
        assert!(err(mapper.to_sandbox_path(path)).contains("escapes the workspace"));
        assert!(err(mapper.assert_sandbox_path(path)).contains("escapes the workspace"));
    }

    #[test]
    fn makes_writable_roots_readable_and_keeps_path_prefix_boundaries() {
        let mapper = policy_mapper(PiFileToolPathPolicy {
            writable_roots: roots(&["/tmp/"]),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            mapper.to_sandbox_path("/tmp/output.json").unwrap(),
            "/tmp/output.json"
        );
        assert_eq!(
            mapper.to_readable_sandbox_path("/tmp/output.json").unwrap(),
            "/tmp/output.json"
        );
        assert_eq!(
            mapper.assert_sandbox_path("/tmp/output.json").unwrap(),
            "/tmp/output.json"
        );
        assert!(
            err(mapper.to_sandbox_path("/tmp-evil/output.json")).contains("escapes the workspace")
        );
        assert!(
            err(mapper.to_readable_sandbox_path("/tmp-evil/output.json"))
                .contains("escapes the workspace")
        );
    }

    #[test]
    fn requires_every_public_policy_root_to_be_an_absolute_sandbox_path() {
        let message = |policy| policy_mapper(policy).unwrap_err().message().to_string();
        assert_eq!(
            message(PiFileToolPathPolicy {
                readable_roots: roots(&["reference"]),
                ..Default::default()
            }),
            "Pi readable root must be an absolute sandbox path: reference"
        );
        assert_eq!(
            message(PiFileToolPathPolicy {
                writable_roots: roots(&["scratch"]),
                ..Default::default()
            }),
            "Pi writable root must be an absolute sandbox path: scratch"
        );
        assert_eq!(
            message(PiFileToolPathPolicy {
                denied_roots: roots(&["private"]),
                ..Default::default()
            }),
            "Pi denied root must be an absolute sandbox path: private"
        );
    }

    #[test]
    fn gives_denied_roots_precedence_over_every_readable_and_writable_root() {
        let skill_root = "/home/vercel-sandbox/.agents/skills";
        let mapper = mapper(CreatePiPathMapperOptions {
            readable_roots: readable(&[skill_root, "/mnt/reference"]),
            writable_roots: roots(&["/tmp"]),
            denied_roots: roots(&[
                &format!("{SANDBOX_WORK_DIR}/private"),
                &format!("{skill_root}/private"),
                "/mnt/reference/private",
                "/tmp/private",
            ]),
            ..Default::default()
        });
        assert!(err(mapper.to_sandbox_path("private/token")).contains("inside a denied root"));
        assert!(
            err(mapper.to_readable_sandbox_path(&format!("{skill_root}/private/SKILL.md")))
                .contains("inside a denied root")
        );
        assert!(
            err(mapper.to_readable_sandbox_path("/mnt/reference/private/data.json"))
                .contains("inside a denied root")
        );
        assert!(
            err(mapper.to_sandbox_path("/tmp/private/result.json"))
                .contains("inside a denied root")
        );
        assert_eq!(
            mapper
                .to_sandbox_path(&format!("{SANDBOX_WORK_DIR}/public.txt"))
                .unwrap(),
            format!("{SANDBOX_WORK_DIR}/public.txt")
        );
        assert_eq!(
            mapper
                .to_readable_sandbox_path(&format!("{skill_root}/public/SKILL.md"))
                .unwrap(),
            format!("{skill_root}/public/SKILL.md")
        );
        assert_eq!(
            mapper
                .to_readable_sandbox_path("/mnt/reference/public.json")
                .unwrap(),
            "/mnt/reference/public.json"
        );
        assert_eq!(
            mapper.to_sandbox_path("/tmp/public.json").unwrap(),
            "/tmp/public.json"
        );
    }

    #[test]
    fn reports_workspace_membership_with_normalized_boundary_safe_checks() {
        let mapper = policy_mapper(PiFileToolPathPolicy {
            writable_roots: roots(&["/tmp"]),
            ..Default::default()
        })
        .unwrap();
        assert!(mapper.is_workspace_path(SANDBOX_WORK_DIR));
        assert!(mapper.is_workspace_path(&format!("{SANDBOX_WORK_DIR}/src/../package.json")));
        assert!(!mapper.is_workspace_path(&format!("{SANDBOX_WORK_DIR}-other/file.txt")));
        assert!(!mapper.is_workspace_path("/tmp/output.json"));
    }

    #[test]
    fn returns_normalized_denied_descendants_within_a_recursive_root() {
        let mapper = policy_mapper(PiFileToolPathPolicy {
            denied_roots: roots(&[
                &format!("{SANDBOX_WORK_DIR}/private"),
                &format!("{SANDBOX_WORK_DIR}/nested/../credentials"),
                "/tmp/private",
                "/tmp-other/not-a-descendant",
            ]),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            mapper.relative_denied_roots_under(SANDBOX_WORK_DIR),
            vec!["private", "credentials"]
        );
        assert_eq!(mapper.relative_denied_roots_under("/tmp"), vec!["private"]);
        assert_eq!(mapper.relative_denied_roots_under("/tmp/private"), vec![""]);
    }

    #[test]
    fn to_relative_path_returns_dot_for_the_sandbox_root() {
        assert_eq!(base().to_relative_path(SANDBOX_WORK_DIR), ".");
    }

    #[test]
    fn to_relative_path_returns_posix_relative_form_for_nested_paths() {
        assert_eq!(
            base().to_relative_path(&format!("{SANDBOX_WORK_DIR}/a/b/c.ts")),
            "a/b/c.ts"
        );
    }
}
