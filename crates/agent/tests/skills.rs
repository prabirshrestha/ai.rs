use agent::{
    Skill, SkillDiagnosticCode, SourcedSkillInput, format_skill_invocation,
    format_skills_for_system_prompt, load_skills, load_sourced_skills,
};

fn temp_root(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "ai-rs-pi-skills-{label}-{}",
        agent::create_session_id()
    ))
}

#[tokio::test]
async fn loads_skill_md_files_from_directories() {
    let root = temp_root("basic");
    tokio::fs::create_dir_all(root.join(".agents/skills/example"))
        .await
        .unwrap();
    tokio::fs::write(
        root.join(".agents/skills/example/SKILL.md"),
        "---\nname: example\ndescription: Example skill\ndisable-model-invocation: true\n---\nUse this skill.\n",
    )
    .await
    .unwrap();

    let result = load_skills(&root, &[std::path::Path::new(".agents/skills")]).await;

    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.skills,
        [Skill {
            name: "example".to_string(),
            description: "Example skill".to_string(),
            content: "Use this skill.".to_string(),
            file_path: root
                .join(".agents/skills/example/SKILL.md")
                .to_string_lossy()
                .into_owned(),
            disable_model_invocation: true,
        }]
    );
    let _ = tokio::fs::remove_dir_all(root).await;
}

#[tokio::test]
async fn loads_skills_through_symlinked_directories() {
    let root = temp_root("symlink");
    tokio::fs::create_dir_all(root.join("actual/example"))
        .await
        .unwrap();
    tokio::fs::write(
        root.join("actual/example/SKILL.md"),
        "---\nname: example\ndescription: Example skill\n---\nUse this skill.",
    )
    .await
    .unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(root.join("actual"), root.join("skills-link")).unwrap();
    #[cfg(windows)]
    std::os::windows::fs::symlink_dir(root.join("actual"), root.join("skills-link")).unwrap();

    let result = load_skills(&root, &[std::path::Path::new("skills-link")]).await;

    assert_eq!(
        result
            .skills
            .iter()
            .map(|skill| skill.name.as_str())
            .collect::<Vec<_>>(),
        ["example"]
    );
    assert_eq!(
        result.skills[0].file_path,
        root.join("skills-link/example/SKILL.md")
            .to_string_lossy()
            .into_owned()
    );
    let _ = tokio::fs::remove_dir_all(root).await;
}

#[tokio::test]
async fn sourced_skills_attach_source_to_results_and_diagnostics() {
    let root = temp_root("sourced");
    tokio::fs::create_dir_all(root.join("user/example"))
        .await
        .unwrap();
    tokio::fs::create_dir_all(root.join("bad/broken"))
        .await
        .unwrap();
    tokio::fs::write(
        root.join("user/example/SKILL.md"),
        "---\nname: example\ndescription: Example skill\n---\nUse this skill.",
    )
    .await
    .unwrap();
    tokio::fs::write(
        root.join("bad/broken/SKILL.md"),
        "---\nname: broken\n---\nMissing description.",
    )
    .await
    .unwrap();

    let result = load_sourced_skills(
        &root,
        &[
            SourcedSkillInput {
                path: "user".into(),
                source: "user",
            },
            SourcedSkillInput {
                path: "bad".into(),
                source: "bad",
            },
        ],
    )
    .await;

    assert_eq!(result.skills.len(), 1);
    assert_eq!(result.skills[0].source, "user");
    assert_eq!(result.skills[0].skill.name, "example");
    assert_eq!(result.diagnostics.len(), 1);
    assert_eq!(result.diagnostics[0].source, "bad");
    assert_eq!(
        result.diagnostics[0].diagnostic.code,
        SkillDiagnosticCode::InvalidMetadata
    );
    assert_eq!(
        result.diagnostics[0].diagnostic.message,
        "description is required"
    );
    let _ = tokio::fs::remove_dir_all(root).await;
}

#[tokio::test]
async fn loads_direct_markdown_children_only_from_root_directory() {
    let root = temp_root("root-md");
    tokio::fs::create_dir_all(root.join("skills/nested"))
        .await
        .unwrap();
    tokio::fs::write(
        root.join("skills/root.md"),
        "---\ndescription: Root skill\n---\nRoot content",
    )
    .await
    .unwrap();
    tokio::fs::write(
        root.join("skills/nested/ignored.md"),
        "---\ndescription: Ignored\n---\nIgnored content",
    )
    .await
    .unwrap();

    let result = load_skills(&root, &[std::path::Path::new("skills")]).await;

    assert_eq!(
        result
            .skills
            .iter()
            .map(|skill| skill.name.as_str())
            .collect::<Vec<_>>(),
        ["skills"]
    );
    assert_eq!(result.skills[0].content, "Root content");
    let _ = tokio::fs::remove_dir_all(root).await;
}

#[tokio::test]
async fn skill_loading_honors_basic_ignore_rules() {
    let root = temp_root("ignore");
    tokio::fs::create_dir_all(root.join("skills/keep"))
        .await
        .unwrap();
    tokio::fs::create_dir_all(root.join("skills/skip"))
        .await
        .unwrap();
    tokio::fs::write(root.join("skills/.gitignore"), "skip/\n")
        .await
        .unwrap();
    tokio::fs::write(
        root.join("skills/keep/SKILL.md"),
        "---\nname: keep\ndescription: Keep skill\n---\nKeep.",
    )
    .await
    .unwrap();
    tokio::fs::write(
        root.join("skills/skip/SKILL.md"),
        "---\nname: skip\ndescription: Skip skill\n---\nSkip.",
    )
    .await
    .unwrap();

    let result = load_skills(&root, &[std::path::Path::new("skills")]).await;

    assert_eq!(
        result
            .skills
            .iter()
            .map(|skill| skill.name.as_str())
            .collect::<Vec<_>>(),
        ["keep"]
    );
    let _ = tokio::fs::remove_dir_all(root).await;
}

#[test]
fn formats_skill_invocations_with_additional_instructions() {
    let skill = Skill {
        name: "inspect".to_string(),
        description: "Inspect things".to_string(),
        content: "Use inspection tools.".to_string(),
        file_path: "/project/.pi/skills/inspect/SKILL.md".to_string(),
        disable_model_invocation: false,
    };

    assert_eq!(
        format_skill_invocation(&skill, Some("Check errors.")),
        "<skill name=\"inspect\" location=\"/project/.pi/skills/inspect/SKILL.md\">\nReferences are relative to /project/.pi/skills/inspect.\n\nUse inspection tools.\n</skill>\n\nCheck errors."
    );
}

#[test]
fn formats_visible_skills_for_system_prompt() {
    let visible = Skill {
        name: "visible".to_string(),
        description: "Use <this> & that".to_string(),
        content: "visible content".to_string(),
        file_path: "/skills/visible/SKILL.md".to_string(),
        disable_model_invocation: false,
    };
    let disabled = Skill {
        name: "hidden".to_string(),
        description: "Hidden".to_string(),
        content: "hidden content".to_string(),
        file_path: "/skills/hidden/SKILL.md".to_string(),
        disable_model_invocation: true,
    };

    let formatted = format_skills_for_system_prompt(&[visible, disabled]);
    assert!(formatted.contains("<name>visible</name>"));
    assert!(formatted.contains("<description>Use &lt;this&gt; &amp; that</description>"));
    assert!(!formatted.contains("hidden"));
    assert_eq!(format_skills_for_system_prompt(&[]), "");
}
