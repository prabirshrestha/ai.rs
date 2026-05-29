use agent::{
    PromptTemplate, PromptTemplateDiagnosticCode, SourcedPromptTemplateInput,
    format_prompt_template_invocation, load_prompt_templates, load_sourced_prompt_templates,
    parse_command_args,
};

fn temp_root(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "ai-rs-pi-prompts-{label}-{}",
        agent::create_session_id()
    ))
}

#[tokio::test]
async fn loads_markdown_templates_non_recursively_from_dirs() {
    let root = temp_root("dirs");
    tokio::fs::create_dir_all(root.join("a/nested"))
        .await
        .unwrap();
    tokio::fs::create_dir_all(root.join("b")).await.unwrap();
    tokio::fs::write(
        root.join("a/one.md"),
        "---\ndescription: One template\n---\nHello $1",
    )
    .await
    .unwrap();
    tokio::fs::write(root.join("a/nested/ignored.md"), "Ignored")
        .await
        .unwrap();
    tokio::fs::write(root.join("b/two.md"), "First line description\nBody")
        .await
        .unwrap();

    let result = load_prompt_templates(
        &root,
        &[std::path::Path::new("a"), std::path::Path::new("b")],
    )
    .await;

    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.prompt_templates,
        [
            PromptTemplate {
                name: "one".to_string(),
                description: Some("One template".to_string()),
                content: "Hello $1".to_string(),
            },
            PromptTemplate {
                name: "two".to_string(),
                description: Some("First line description".to_string()),
                content: "First line description\nBody".to_string(),
            },
        ]
    );
    let _ = tokio::fs::remove_dir_all(root).await;
}

#[tokio::test]
async fn sourced_prompt_templates_attach_source_to_results_and_diagnostics() {
    let root = temp_root("sourced");
    tokio::fs::create_dir_all(root.join("prompts"))
        .await
        .unwrap();
    tokio::fs::write(
        root.join("prompts/example.md"),
        "---\ndescription: Example\n---\nExample body",
    )
    .await
    .unwrap();
    tokio::fs::write(
        root.join("broken.md"),
        "---\ndescription: [unterminated\n---\nBody",
    )
    .await
    .unwrap();

    let result = load_sourced_prompt_templates(
        &root,
        &[
            SourcedPromptTemplateInput {
                path: "prompts".into(),
                source: "project",
            },
            SourcedPromptTemplateInput {
                path: "broken.md".into(),
                source: "user",
            },
        ],
    )
    .await;

    assert_eq!(result.prompt_templates.len(), 1);
    assert_eq!(result.prompt_templates[0].source, "project");
    assert_eq!(result.prompt_templates[0].prompt_template.name, "example");
    assert_eq!(result.diagnostics.len(), 1);
    assert_eq!(result.diagnostics[0].source, "user");
    assert_eq!(
        result.diagnostics[0].diagnostic.code,
        PromptTemplateDiagnosticCode::ParseFailed
    );
    let _ = tokio::fs::remove_dir_all(root).await;
}

#[tokio::test]
async fn loads_explicit_markdown_files_and_symlinks() {
    let root = temp_root("symlink");
    tokio::fs::create_dir_all(&root).await.unwrap();
    tokio::fs::write(
        root.join("target.md"),
        "---\ndescription: Target\n---\nTarget body",
    )
    .await
    .unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(root.join("target.md"), root.join("link.md")).unwrap();
    #[cfg(windows)]
    std::os::windows::fs::symlink_file(root.join("target.md"), root.join("link.md")).unwrap();

    let result = load_prompt_templates(
        &root,
        &[
            std::path::Path::new("target.md"),
            std::path::Path::new("link.md"),
        ],
    )
    .await;

    assert_eq!(
        result
            .prompt_templates
            .iter()
            .map(|template| template.name.as_str())
            .collect::<Vec<_>>(),
        ["target", "link"]
    );
    assert!(result.diagnostics.is_empty());
    let _ = tokio::fs::remove_dir_all(root).await;
}

#[test]
fn parses_and_substitutes_prompt_arguments() {
    assert_eq!(
        parse_command_args("one 'two words' \"three words\""),
        ["one", "two words", "three words"]
    );

    let template = PromptTemplate {
        name: "review".to_string(),
        description: None,
        content: "Review $1 ${@:2} ${@:1:2} with $ARGUMENTS and $@".to_string(),
    };
    let args = vec!["a.ts".to_string(), "care".to_string(), "now".to_string()];
    assert_eq!(
        format_prompt_template_invocation(&template, &args),
        "Review a.ts care now a.ts care with a.ts care now and a.ts care now"
    );
}
