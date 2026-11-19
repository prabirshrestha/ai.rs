//! Port of `acp-harness.test.ts`: MCP and host tool resolution, skills and
//! instructions.

use std::sync::Arc;

use indexmap::IndexMap;
use parking_lot::Mutex;
use serde_json::{Value, json};

use crate::harness::spec::harness::{HarnessV1Prompt, HarnessV1Session};
use crate::harness::spec::tools::HarnessV1Skill;

use super::acp_harness::create_acp;
use super::acp_harness_test_support::*;
use super::acp_tool_call::ACPResolvedHostToolCall;
use super::v1::settings::{ACPInstallCommandSource, ACPInstructionMapping, ACPSource};

fn skill(value: Value) -> HarnessV1Skill {
    serde_json::from_value(value).unwrap()
}

/// Runs a turn to its `finish`.
async fn finish_turn(
    session: &dyn HarnessV1Session,
    socket: &FakeSocket,
    options: crate::harness::spec::harness::HarnessV1PromptTurnOptions,
) -> Value {
    let before = socket.sent().len();
    let control = session.do_prompt_turn(options).await.unwrap();
    let sent = wait_for_sent(socket, before + 1).await;
    socket.emit(finish_frame());
    done(&*control).await.unwrap();
    sent[before].clone()
}

#[tokio::test]
async fn forwards_mcp_servers_and_lets_the_agent_classify_unknown_acp_tools() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let classifications = Arc::new(Mutex::new(0));
    let mcp_servers = IndexMap::from([(
        "external".to_string(),
        json!({ "command": "/usr/bin/external-mcp", "args": [], "env": [] }),
    )]);
    let mut settings = agent_settings("codex-acp");
    settings.mcp_servers = Some(mcp_servers.clone());
    settings.is_mcp_tool_call = {
        let classifications = classifications.clone();
        Arc::new(move |tool_call| {
            *classifications.lock() += 1;
            Ok(tool_call
                .meta
                .clone()
                .flatten()
                .and_then(|meta| meta.get("is_mcp_tool_call").cloned())
                == Some(json!(true)))
        })
    };
    let harness = create_acp(harness_settings(settings, &bridge)).unwrap();
    let session = start(&harness, start_options("session-1", sandbox.network())).await;
    let (options, events) = prompt_options("Use the external tool.");
    let control = session.do_prompt_turn(options).await.unwrap();
    let channel = bridge.socket(0);

    let sent = wait_for_sent(&channel, 1).await;
    assert_match_object(
        &sent[0],
        &json!({ "type": "start", "mcpServers": mcp_servers }),
    );
    channel.emit(json!({
        "type": "acp-tool-call-candidate",
        "requestId": "candidate-1",
        "toolCall": {
            "toolCallId": "call-1",
            "title": "External tool",
            "_meta": { "is_mcp_tool_call": true },
        },
    }));
    channel.emit(json!({
        "type": "tool-call",
        "toolCallId": "call-1",
        "toolName": "external_tool",
        "input": "{}",
        "providerExecuted": true,
    }));
    channel.emit(json!({
        "type": "tool-result",
        "toolCallId": "call-1",
        "toolName": "external_tool",
        "result": { "ok": true },
    }));
    channel.emit(finish_frame());

    done(&*control).await.unwrap();
    assert_eq!(*classifications.lock(), 1);
    assert_eq!(
        events_json(&events),
        vec![
            json!({
                "dynamic": true,
                "input": "{}",
                "providerExecuted": true,
                "toolCallId": "call-1",
                "toolName": "external_tool",
                "type": "tool-call",
            }),
            json!({
                "dynamic": true,
                "result": { "ok": true },
                "toolCallId": "call-1",
                "toolName": "external_tool",
                "type": "tool-result",
            }),
            json!({
                "finishReason": { "raw": "end_turn", "unified": "stop" },
                "totalUsage": unknown_usage(),
                "type": "finish",
            }),
        ]
    );
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn resolves_host_tool_calls_through_the_adapter_resolver() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let calls: Arc<Mutex<Vec<Value>>> = Arc::default();
    let mut settings = agent_settings("codex-acp");
    settings.resolve_host_tool_call = {
        let calls = calls.clone();
        Arc::new(move |options| {
            calls.lock().push(json!({
                "toolCall": options.tool_call,
                "serverName": options.server_name,
                "toolNames": options.tool_names,
            }));
            let raw_input = options.tool_call.raw_input.clone().unwrap_or(Value::Null);
            let tool = raw_input["tool"].as_str().unwrap_or("").to_string();
            Ok(
                (raw_input["server"] == options.server_name && options.tool_names.contains(&tool))
                    .then(|| ACPResolvedHostToolCall {
                        tool_name: tool,
                        input: serde_json::from_value(raw_input["arguments"].clone()).unwrap(),
                    }),
            )
        })
    };
    let harness = create_acp(harness_settings(settings, &bridge)).unwrap();
    let session = start(&harness, start_options("session-1", sandbox.network())).await;
    let (options, _) = prompt_options("Use the weather tool.");
    let control = session.do_prompt_turn(options).await.unwrap();
    let channel = bridge.socket(0);
    let weather_call = json!({
        "toolCallId": "call-1",
        "title": "mcp.ai-sdk-harness-tools.weather",
        "rawInput": {
            "server": "ai-sdk-harness-tools",
            "tool": "weather",
            "arguments": { "city": "Lima" },
        },
    });

    channel.emit(json!({
        "type": "acp-host-tool-call-resolution-request",
        "requestId": "resolution-1",
        "toolCall": weather_call,
        "serverName": "ai-sdk-harness-tools",
        "toolNames": ["weather"],
    }));
    channel.emit(json!({
        "type": "acp-host-tool-call-resolution-request",
        "requestId": "resolution-2",
        "toolCall": {
            "toolCallId": "call-2",
            "title": "Read README.md",
            "rawInput": { "path": "README.md" },
        },
        "serverName": "ai-sdk-harness-tools",
        "toolNames": ["weather"],
    }));
    channel.emit(finish_frame());
    done(&*control).await.unwrap();

    assert!(calls.lock().contains(&json!({
        "toolCall": weather_call,
        "serverName": "ai-sdk-harness-tools",
        "toolNames": ["weather"],
    })));
    let tool_results: Vec<Value> = channel
        .sent()
        .into_iter()
        .filter(|message| message["type"] == "tool-result")
        .collect();
    assert_eq!(
        tool_results,
        vec![
            json!({
                "output": { "hostToolCall": { "input": { "city": "Lima" }, "toolName": "weather" } },
                "toolCallId": "resolution-1",
                "type": "tool-result",
            }),
            json!({
                "output": { "hostToolCall": null },
                "toolCallId": "resolution-2",
                "type": "tool-result",
            }),
        ]
    );
    session.do_destroy().await.unwrap();
}

/// [rust] The `non-object input` case cannot be expressed: the resolved
/// input is a typed object.
#[tokio::test]
async fn fails_the_turn_when_the_host_tool_resolver_misbehaves() {
    let resolvers: Vec<super::acp_tool_call::ACPResolveHostToolCall> = vec![
        Arc::new(|_| {
            Ok(Some(ACPResolvedHostToolCall {
                tool_name: "unknown".into(),
                input: IndexMap::new(),
            }))
        }),
        Arc::new(|_| Err(crate::harness::utils::plain_error("resolver failed"))),
    ];
    for resolver in resolvers {
        let bridge = FakeBridge::default();
        let sandbox = FakeSandbox::default();
        let mut settings = agent_settings("codex-acp");
        settings.resolve_host_tool_call = resolver;
        let harness = create_acp(harness_settings(settings, &bridge)).unwrap();
        let session = start(&harness, start_options("session-1", sandbox.network())).await;
        let (options, events) = prompt_options("Use the weather tool.");
        let control = session.do_prompt_turn(options).await.unwrap();
        let channel = bridge.socket(0);

        channel.emit(json!({
            "type": "acp-host-tool-call-resolution-request",
            "requestId": "resolution-1",
            "toolCall": { "toolCallId": "call-1", "title": "Weather", "rawInput": {} },
            "serverName": "ai-sdk-harness-tools",
            "toolNames": ["weather"],
        }));

        assert!(done(&*control).await.is_err());
        assert!(
            events_json(&events)
                .iter()
                .any(|event| event["type"] == "error")
        );
        let sent: Vec<Value> = wait_for_sent(&channel, 3)
            .await
            .into_iter()
            .filter(|message| message["type"] == "tool-result" || message["type"] == "abort")
            .collect();
        assert_eq!(
            sent,
            vec![
                json!({
                    "type": "tool-result",
                    "toolCallId": "resolution-1",
                    "output": { "hostToolCall": null },
                }),
                json!({ "type": "abort" }),
            ]
        );
        session.do_destroy().await.unwrap();
    }
}

#[tokio::test]
async fn materializes_skills_in_the_native_directory_without_adding_them_to_prompt_guidance() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let harness = create_acp(harness_settings(agent_settings("codex-acp"), &bridge)).unwrap();
    let skills = vec![skill(json!({
        "name": "release-notes",
        "description": "Prepare concise release notes.",
        "content": "Full private skill instructions.",
        "files": [{ "path": "references/style.md", "content": "Use active voice." }],
    }))];
    let session = start(&harness, start_options("session-1", sandbox.network())).await;
    let channel = bridge.socket(0);

    let (mut options, _) = prompt_options("");
    options.settings.skills = skills.clone();
    options.settings.instructions = Some("Use the supplied project context.".into());
    options.prompt = HarnessV1Prompt::Message(
        serde_json::from_value(json!({
            "role": "user",
            "content": [
                { "type": "text", "text": "Draft" },
                { "type": "text", "text": "the notes." },
            ],
        }))
        .unwrap(),
    );
    let first_start = finish_turn(&*session, &channel, options).await;
    let writes = sandbox.writes();
    let skill_definition = writes
        .iter()
        .find(|(path, _)| path.ends_with("/release-notes/SKILL.md"))
        .cloned()
        .unwrap();
    assert_eq!(
        skill_definition.0,
        "/home/agent/.agents/skills/release-notes/SKILL.md"
    );
    assert!(
        skill_definition
            .1
            .contains("Full private skill instructions.")
    );
    assert!(
        writes.contains(&(
            skill_definition
                .0
                .replace("/SKILL.md", "/references/style.md"),
            "Use active voice.".to_string()
        ))
    );
    assert!(
        sandbox
            .runs()
            .contains(&"printf \"%s\" \"$HOME\"".to_string())
    );
    let prompt = first_start["prompt"].as_array().unwrap();
    assert_eq!(
        prompt[1..].to_vec(),
        vec![
            json!({ "type": "text", "text": "Draft" }),
            json!({ "type": "text", "text": "the notes." }),
        ]
    );
    let guidance = prompt[0]["text"].as_str().unwrap();
    assert!(guidance.contains("<session-guidance>"));
    assert!(guidance.contains("Use the supplied project context."));
    let prompt_json = first_start["prompt"].to_string();
    for hidden in [
        "release-notes",
        skill_definition.0.as_str(),
        "Full private skill instructions.",
        "Use active voice.",
    ] {
        assert!(!prompt_json.contains(hidden), "{hidden}");
    }

    let (mut options, _) = prompt_options("Revise them.");
    options.settings.skills = skills;
    options.settings.instructions = Some("Use the revised project context.".into());
    let second_start = finish_turn(&*session, &channel, options).await;
    assert_eq!(second_start["type"], "start");
    let prompt = second_start["prompt"].as_array().unwrap();
    assert_eq!(
        prompt[1..].to_vec(),
        vec![json!({ "type": "text", "text": "Revise them." })]
    );
    let guidance = prompt[0]["text"].as_str().unwrap();
    assert!(guidance.contains("<session-guidance>"));
    assert!(guidance.contains("Use the revised project context."));

    let lifecycle_state = session.do_stop().await.unwrap();
    assert!(lifecycle_state.data["agentIdentity"].is_string());
    assert!(
        regex::Regex::new("^[a-f0-9]{64}$").unwrap().is_match(
            lifecycle_state.data["authenticationProfile"]["digest"]
                .as_str()
                .unwrap()
        )
    );
    assert_match_object(
        &lifecycle_state.data,
        &json!({
            "authenticationProfile": { "providerKind": "direct" },
            "initialGuidanceApplied": true,
            "skillsDirectory": "/home/agent/.agents/skills",
        }),
    );
    assert!(lifecycle_state.data.get("skillsFingerprint").is_none());
    assert!(lifecycle_state.data.get("skillsMaterialized").is_none());
}

#[tokio::test]
async fn materializes_install_command_skills_relative_to_the_private_agent_home() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let mut settings = agent_settings("cursor-skills-acp");
    settings.source = ACPSource::InstallCommand(ACPInstallCommandSource {
        command: "curl https://cursor.com/install -fsS | bash".into(),
    });
    settings.executable = "agent".into();
    settings.args = Some(vec!["--disable-auto-update".into(), "acp".into()]);
    settings.forward_env = None;
    let harness = create_acp(harness_settings(settings, &bridge)).unwrap();
    let session = start(&harness, start_options("session-1", sandbox.network())).await;
    let (mut options, _) = prompt_options("Draft release notes.");
    options.settings.skills = vec![skill(json!({
        "name": "release-notes",
        "description": "Prepare release notes.",
        "content": "Use active voice.",
    }))];
    finish_turn(&*session, &bridge.socket(0), options).await;

    assert!(sandbox.writes().contains(&(
        "/home/agent/.ai-sdk-harness/.harness-bootstrap/cursor-skills-acp/home/.agents/skills/release-notes/SKILL.md"
            .to_string(),
        "---\nname: release-notes\ndescription: Prepare release notes.\n---\n\nUse active voice."
            .to_string()
    )));
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn uses_native_instructions_initially_and_prompt_guidance_when_they_change() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let mut settings = agent_settings("claude-acp");
    settings.skills_directory = Some(".claude/skills".into());
    settings.instruction_mapping = Some(ACPInstructionMapping::SessionMeta {
        path: vec!["systemPrompt".into(), "append".into()],
    });
    let harness = create_acp(harness_settings(settings, &bridge)).unwrap();
    let skills = vec![skill(json!({
        "name": "release-notes",
        "description": "Prepare concise release notes.",
        "content": "Use active voice.",
    }))];
    let session = start(&harness, start_options("session-1", sandbox.network())).await;
    let channel = bridge.socket(0);

    let (mut options, _) = prompt_options("Draft release notes.");
    options.settings.skills = skills.clone();
    options.settings.instructions = Some("Answer every question in German.".into());
    let first_start = finish_turn(&*session, &channel, options).await;
    assert_match_object(
        &first_start,
        &json!({
            "type": "start",
            "instructions": "Answer every question in German.",
            "instructionMapping": { "type": "session-meta", "path": ["systemPrompt", "append"] },
            "prompt": [{ "type": "text", "text": "Draft release notes." }],
        }),
    );
    assert!(sandbox.writes().contains(&(
        "/home/agent/.claude/skills/release-notes/SKILL.md".to_string(),
        "---\nname: release-notes\ndescription: Prepare concise release notes.\n---\n\nUse active voice."
            .to_string()
    )));
    let prompt_json = first_start["prompt"].to_string();
    assert!(!prompt_json.contains("Answer every question in German."));
    assert!(!prompt_json.contains("release-notes"));

    let (mut options, _) = prompt_options("Revise them.");
    options.settings.skills = skills;
    options.settings.instructions = Some("Answer every question in French.".into());
    let second_start = finish_turn(&*session, &channel, options).await;
    assert_match_object(
        &second_start,
        &json!({
            "type": "start",
            "instructions": "Answer every question in French.",
            "instructionMapping": { "type": "session-meta", "path": ["systemPrompt", "append"] },
        }),
    );
    let prompt = second_start["prompt"].as_array().unwrap();
    assert_eq!(
        prompt[1..].to_vec(),
        vec![json!({ "type": "text", "text": "Revise them." })]
    );
    assert!(
        prompt[0]["text"]
            .as_str()
            .unwrap()
            .contains("Answer every question in French.")
    );
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn materializes_instructions_into_the_filesystem_and_never_prepends_prompt_guidance() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let mut settings = agent_settings("cursor-acp");
    settings.instruction_mapping = Some(ACPInstructionMapping::Filesystem {
        path: ".cursor/rules/AGENTS.md".into(),
    });
    let harness = create_acp(harness_settings(settings, &bridge)).unwrap();
    let session = start(&harness, start_options("session-1", sandbox.network())).await;
    let channel = bridge.socket(0);

    for (prompt, instructions) in [
        ("Draft release notes.", "Always run tests."),
        ("Revise them.", "Answer every question in French."),
    ] {
        let (mut options, _) = prompt_options(prompt);
        options.settings.instructions = Some(instructions.into());
        let start = finish_turn(&*session, &channel, options).await;
        assert_match_object(
            &start,
            &json!({
                "type": "start",
                "instructions": instructions,
                "instructionMapping": { "type": "filesystem", "path": ".cursor/rules/AGENTS.md" },
                "prompt": [{ "type": "text", "text": prompt }],
            }),
        );
        assert!(sandbox.writes().contains(&(
            "/home/agent/.cursor/rules/AGENTS.md".to_string(),
            format!("{instructions}\n")
        )));
        assert!(!start["prompt"].to_string().contains(instructions));
    }
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn replaces_changed_skills_before_starting_the_next_turn() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let harness = create_acp(harness_settings(agent_settings("codex-acp"), &bridge)).unwrap();
    let session = start(&harness, start_options("session-1", sandbox.network())).await;
    let channel = bridge.socket(0);
    let (mut options, _) = prompt_options("Review this change.");
    options.settings.skills = vec![skill(json!({
        "name": "review-workflow",
        "description": "Review changes.",
        "content": "Review the implementation.",
    }))];
    finish_turn(&*session, &channel, options).await;

    let (mut options, _) = prompt_options("Prepare the release.");
    options.settings.skills = vec![skill(json!({
        "name": "release-workflow",
        "description": "Prepare releases.",
        "content": "Draft the release notes.",
    }))];
    let second_start = finish_turn(&*session, &channel, options).await;
    assert!(
        sandbox
            .runs()
            .contains(&"rm -rf -- '/home/agent/.agents/skills/review-workflow'".to_string())
    );
    assert!(
        sandbox
            .writes()
            .iter()
            .any(|(path, _)| path.ends_with("/release-workflow/SKILL.md"))
    );
    assert_match_object(
        &second_start,
        &json!({
            "type": "start",
            "prompt": [{ "type": "text", "text": "Prepare the release." }],
        }),
    );
    session.do_destroy().await.unwrap();
}
