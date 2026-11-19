//! Port of `harness-github-copilot/src/github-copilot-gateway-model.ts`.

use std::sync::Arc;

use async_trait::async_trait;
use indexmap::IndexMap;
use tokio_util::sync::CancellationToken;

use crate::harness::errors::HarnessResult;
use crate::harness::sdk::tool::Schema;
use crate::harness::spec::HarnessV1StreamPart;
use crate::harness::spec::bootstrap::HarnessV1Bootstrap;
use crate::harness::spec::harness::{
    HarnessV1, HarnessV1ContinueTurnOptions, HarnessV1Emit, HarnessV1PromptControl,
    HarnessV1PromptTurnOptions, HarnessV1ReadHistoryResult, HarnessV1Session,
    HarnessV1StartOptions,
};
use crate::harness::spec::lifecycle_state::{
    HarnessV1ContinueTurnState, HarnessV1ResumeSessionState,
};
use crate::harness::spec::tools::HarnessV1BuiltinTool;

/// `withGitHubCopilotGatewayModel({ harness, gatewayModel })`: fills in the
/// AI Gateway model on `stream-start` parts that do not report a model.
///
/// An upstream GitHub Copilot CLI limitation prevents reading the model
/// through the general ACP layer when the CLI runs with a BYOK provider,
/// which is how AI Gateway authentication is configured. In that mode, the
/// CLI's ACP `session/new` response carries neither a `model` config option
/// nor `models.currentModelId`, and no session update reports the model
/// either. The CLI also refuses to start without the `COPILOT_MODEL`
/// provider environment variable, so the configured AI Gateway model is the
/// model the session runs with.
///
/// With GitHub authentication the CLI reports its model over ACP, and an
/// explicitly requested model is reported by the ACP bridge, so a
/// `stream-start` part without a `modelId` only occurs for AI Gateway
/// sessions without an explicit model.
///
/// [rust] The TS `Proxy` becomes a wrapper that forwards every member.
pub fn with_github_copilot_gateway_model<H: HarnessV1>(
    harness: H,
    gateway_model: impl Into<String>,
) -> GitHubCopilotGatewayModelHarness<H> {
    GitHubCopilotGatewayModelHarness {
        harness,
        gateway_model: gateway_model.into(),
    }
}

/// The harness [`with_github_copilot_gateway_model`] returns.
pub struct GitHubCopilotGatewayModelHarness<H> {
    harness: H,
    gateway_model: String,
}

impl<H> GitHubCopilotGatewayModelHarness<H> {
    /// The wrapped harness.
    pub fn inner(&self) -> &H {
        &self.harness
    }
}

fn wrap_emit(emit: HarnessV1Emit, gateway_model: String) -> HarnessV1Emit {
    Arc::new(move |part| {
        emit(match part {
            HarnessV1StreamPart::StreamStart { warnings, model_id }
                if model_id.as_deref().is_none_or(str::is_empty) =>
            {
                HarnessV1StreamPart::StreamStart {
                    warnings,
                    model_id: Some(gateway_model.clone()),
                }
            }
            part => part,
        })
    })
}

#[async_trait]
impl<H: HarnessV1> HarnessV1 for GitHubCopilotGatewayModelHarness<H> {
    fn specification_version(&self) -> &'static str {
        self.harness.specification_version()
    }

    fn harness_id(&self) -> &str {
        self.harness.harness_id()
    }

    fn builtin_tools(&self) -> &IndexMap<String, HarnessV1BuiltinTool> {
        self.harness.builtin_tools()
    }

    fn supports_builtin_tool_approvals(&self) -> bool {
        self.harness.supports_builtin_tool_approvals()
    }

    fn supports_builtin_tool_filtering(&self) -> bool {
        self.harness.supports_builtin_tool_filtering()
    }

    fn lifecycle_state_schema(&self) -> Option<&Schema> {
        self.harness.lifecycle_state_schema()
    }

    fn supports_get_bootstrap(&self) -> bool {
        self.harness.supports_get_bootstrap()
    }

    async fn get_bootstrap(
        &self,
        abort_signal: Option<CancellationToken>,
    ) -> HarnessResult<HarnessV1Bootstrap> {
        self.harness.get_bootstrap(abort_signal).await
    }

    async fn do_start(
        &self,
        options: HarnessV1StartOptions,
    ) -> HarnessResult<Box<dyn HarnessV1Session>> {
        let session = self.harness.do_start(options).await?;
        Ok(Box::new(GatewayModelSession {
            session,
            gateway_model: self.gateway_model.clone(),
        }))
    }
}

struct GatewayModelSession {
    session: Box<dyn HarnessV1Session>,
    gateway_model: String,
}

#[async_trait]
impl HarnessV1Session for GatewayModelSession {
    fn session_id(&self) -> String {
        self.session.session_id()
    }

    fn is_resume(&self) -> bool {
        self.session.is_resume()
    }

    async fn do_prompt_turn(
        &self,
        mut options: HarnessV1PromptTurnOptions,
    ) -> HarnessResult<Box<dyn HarnessV1PromptControl>> {
        options.emit = wrap_emit(options.emit, self.gateway_model.clone());
        self.session.do_prompt_turn(options).await
    }

    async fn do_compact(&self, custom_instructions: Option<String>) -> HarnessResult<()> {
        self.session.do_compact(custom_instructions).await
    }

    fn supports_read_history(&self) -> bool {
        self.session.supports_read_history()
    }

    async fn do_read_history(
        &self,
        since: Option<String>,
    ) -> HarnessResult<HarnessV1ReadHistoryResult> {
        self.session.do_read_history(since).await
    }

    async fn do_continue_turn(
        &self,
        mut options: HarnessV1ContinueTurnOptions,
    ) -> HarnessResult<Box<dyn HarnessV1PromptControl>> {
        options.emit = wrap_emit(options.emit, self.gateway_model.clone());
        self.session.do_continue_turn(options).await
    }

    async fn do_suspend_turn(&self) -> HarnessResult<HarnessV1ContinueTurnState> {
        self.session.do_suspend_turn().await
    }

    async fn do_detach(&self) -> HarnessResult<HarnessV1ResumeSessionState> {
        self.session.do_detach().await
    }

    async fn do_stop(&self) -> HarnessResult<HarnessV1ResumeSessionState> {
        self.session.do_stop().await
    }

    async fn do_destroy(&self) -> HarnessResult<()> {
        self.session.do_destroy().await
    }
}

#[cfg(test)]
mod tests {
    use futures::FutureExt as _;
    use futures::future::BoxFuture;
    use serde_json::{Value as JsonValue, json};

    use super::*;
    use crate::harness::adapters::acp::acp_harness_test_support::{
        FakeSandbox, events_json, prompt_options, start_options,
    };
    use crate::harness::spec::harness::HarnessV1SubmitToolResult;

    struct Control;

    #[async_trait]
    impl HarnessV1PromptControl for Control {
        async fn submit_tool_result(&self, _input: HarnessV1SubmitToolResult) -> HarnessResult<()> {
            Ok(())
        }

        fn done(&self) -> BoxFuture<'static, HarnessResult<()>> {
            async { Ok(()) }.boxed()
        }
    }

    struct FakeSession {
        parts: Vec<HarnessV1StreamPart>,
    }

    impl FakeSession {
        fn run_turn(&self, emit: &HarnessV1Emit) -> Box<dyn HarnessV1PromptControl> {
            for part in &self.parts {
                emit(part.clone());
            }
            Box::new(Control)
        }
    }

    #[async_trait]
    impl HarnessV1Session for FakeSession {
        fn session_id(&self) -> String {
            "session-1".into()
        }

        fn is_resume(&self) -> bool {
            false
        }

        async fn do_prompt_turn(
            &self,
            options: HarnessV1PromptTurnOptions,
        ) -> HarnessResult<Box<dyn HarnessV1PromptControl>> {
            Ok(self.run_turn(&options.emit))
        }

        async fn do_compact(&self, _custom_instructions: Option<String>) -> HarnessResult<()> {
            unimplemented!()
        }

        async fn do_continue_turn(
            &self,
            options: HarnessV1ContinueTurnOptions,
        ) -> HarnessResult<Box<dyn HarnessV1PromptControl>> {
            Ok(self.run_turn(&options.emit))
        }

        async fn do_suspend_turn(&self) -> HarnessResult<HarnessV1ContinueTurnState> {
            unimplemented!()
        }

        async fn do_detach(&self) -> HarnessResult<HarnessV1ResumeSessionState> {
            unimplemented!()
        }

        async fn do_stop(&self) -> HarnessResult<HarnessV1ResumeSessionState> {
            unimplemented!()
        }

        async fn do_destroy(&self) -> HarnessResult<()> {
            unimplemented!()
        }
    }

    struct FakeHarness {
        parts: Vec<HarnessV1StreamPart>,
        builtin_tools: IndexMap<String, HarnessV1BuiltinTool>,
    }

    #[async_trait]
    impl HarnessV1 for FakeHarness {
        fn harness_id(&self) -> &str {
            "github-copilot"
        }

        fn builtin_tools(&self) -> &IndexMap<String, HarnessV1BuiltinTool> {
            &self.builtin_tools
        }

        async fn do_start(
            &self,
            _options: HarnessV1StartOptions,
        ) -> HarnessResult<Box<dyn HarnessV1Session>> {
            Ok(Box::new(FakeSession {
                parts: self.parts.clone(),
            }))
        }
    }

    fn create_fake_harness(parts: Vec<JsonValue>) -> FakeHarness {
        FakeHarness {
            parts: parts
                .into_iter()
                .map(|part| serde_json::from_value(part).unwrap())
                .collect(),
            builtin_tools: IndexMap::new(),
        }
    }

    async fn collect_prompt_turn_parts(parts: Vec<JsonValue>, continued: bool) -> Vec<JsonValue> {
        let wrapped =
            with_github_copilot_gateway_model(create_fake_harness(parts), "openai/gpt-5.5");
        let session = wrapped
            .do_start(start_options("session-1", FakeSandbox::default().basic()))
            .await
            .unwrap();
        let (options, events) = prompt_options("Hello");
        if continued {
            session
                .do_continue_turn(HarnessV1ContinueTurnOptions {
                    settings: options.settings,
                    response_format: None,
                    abort_signal: None,
                    emit: options.emit,
                })
                .await
                .unwrap();
        } else {
            session.do_prompt_turn(options).await.unwrap();
        }
        events_json(&events)
    }

    #[tokio::test]
    async fn fills_in_the_gateway_model_when_stream_start_reports_no_model() {
        assert_eq!(
            collect_prompt_turn_parts(vec![json!({ "type": "stream-start" })], false).await,
            vec![json!({ "type": "stream-start", "modelId": "openai/gpt-5.5" })]
        );
    }

    #[tokio::test]
    async fn keeps_a_model_reported_by_the_acp_bridge() {
        assert_eq!(
            collect_prompt_turn_parts(
                vec![json!({ "type": "stream-start", "modelId": "claude-sonnet-5" })],
                false
            )
            .await,
            vec![json!({ "type": "stream-start", "modelId": "claude-sonnet-5" })]
        );
    }

    #[tokio::test]
    async fn fills_in_the_gateway_model_for_continued_turns() {
        assert_eq!(
            collect_prompt_turn_parts(vec![json!({ "type": "stream-start" })], true).await,
            vec![json!({ "type": "stream-start", "modelId": "openai/gpt-5.5" })]
        );
    }

    #[tokio::test]
    async fn forwards_other_parts_unchanged() {
        let text_start = json!({ "type": "text-start", "id": "text-1" });
        assert_eq!(
            collect_prompt_turn_parts(vec![text_start.clone()], false).await,
            vec![text_start]
        );
    }

    #[tokio::test]
    async fn preserves_the_wrapped_harness_and_session_properties() {
        let wrapped =
            with_github_copilot_gateway_model(create_fake_harness(vec![]), "openai/gpt-5.5");
        let session = wrapped
            .do_start(start_options("session-1", FakeSandbox::default().basic()))
            .await
            .unwrap();
        assert_eq!(wrapped.harness_id(), "github-copilot");
        assert_eq!(session.session_id(), "session-1");
        assert!(!session.is_resume());
    }
}
