//! Port of `ai` `stop-condition.ts`.

use std::sync::Arc;

use futures::future::BoxFuture;

use crate::harness::sdk::content::StepResult;

/// `StopCondition`: `({ steps }) => PromiseLike<boolean> | boolean`.
pub type StopCondition = Arc<dyn Fn(&[StepResult]) -> BoxFuture<'static, bool> + Send + Sync>;

/// A synchronous stop condition.
pub fn stop_condition(f: impl Fn(&[StepResult]) -> bool + Send + Sync + 'static) -> StopCondition {
    Arc::new(move |steps| {
        let met = f(steps);
        Box::pin(async move { met })
    })
}

/// `isStepCount(stepCount)` (exported as `stepCountIs`).
pub fn is_step_count(step_count: usize) -> StopCondition {
    stop_condition(move |steps| steps.len() == step_count)
}

/// `stepCountIs`.
pub fn step_count_is(step_count: usize) -> StopCondition {
    is_step_count(step_count)
}

/// `isLoopFinished()`.
pub fn is_loop_finished() -> StopCondition {
    stop_condition(|_| false)
}

/// `hasToolCall(...toolName)`.
pub fn has_tool_call(tool_names: Vec<String>) -> StopCondition {
    stop_condition(move |steps| {
        steps.last().is_some_and(|step| {
            step.tool_calls()
                .iter()
                .any(|c| tool_names.iter().any(|n| n == &c.tool_name))
        })
    })
}

/// `isStopConditionMet({ stopConditions, steps })`.
pub async fn is_stop_condition_met(
    stop_conditions: &[StopCondition],
    steps: &[StepResult],
) -> bool {
    let results = futures::future::join_all(stop_conditions.iter().map(|c| c(steps))).await;
    results.into_iter().any(|r| r)
}
