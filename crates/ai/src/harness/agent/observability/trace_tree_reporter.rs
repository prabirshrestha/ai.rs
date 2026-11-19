//! Port of `agent/observability/trace-tree-reporter.ts`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use futures::future::BoxFuture;
use parking_lot::Mutex;

use crate::harness::sdk::telemetry::{Telemetry, TelemetryEvent};

/// Where the rendered tree goes.
pub type TraceTreeWrite = Arc<dyn Fn(&str) + Send + Sync>;

/// `TraceTreeReporterOptions`.
#[derive(Clone, Default)]
pub struct TraceTreeReporterOptions {
    /// Default: write to stderr.
    pub write: Option<TraceTreeWrite>,
}

struct Node {
    label: String,
    start_ms: f64,
    end_ms: Option<f64>,
    children: Vec<usize>,
}

#[derive(Default)]
struct Turn {
    nodes: Vec<Node>,
    step: Option<usize>,
    tools: HashMap<String, usize>,
}

/// `createTraceTreeReporter`: renders an ASCII trace tree of a turn's span
/// lifecycle (turn → steps → tools) at turn end.
pub struct TraceTreeReporter {
    write: TraceTreeWrite,
    turns: Mutex<HashMap<String, Turn>>,
}

/// `createTraceTreeReporter(options)`.
pub fn create_trace_tree_reporter(options: TraceTreeReporterOptions) -> TraceTreeReporter {
    TraceTreeReporter {
        write: options.write.unwrap_or_else(|| {
            Arc::new(|chunk: &str| {
                use std::io::Write as _;
                let _ = std::io::stderr().write_all(chunk.as_bytes());
            })
        }),
        turns: Mutex::default(),
    }
}

fn now_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0)
}

fn call_id(e: &TelemetryEvent) -> String {
    e.get("callId")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string()
}

fn render(turn: &Turn, node: usize, depth: usize, out: &mut String) {
    let n = &turn.nodes[node];
    let dur = match n.end_ms {
        Some(end) => format!("{}ms", (end - n.start_ms).round().max(0.0) as u64),
        None => "(open)".into(),
    };
    out.push_str(&format!("{}- {} {dur}\n", "  ".repeat(depth), n.label));
    for &child in &n.children {
        render(turn, child, depth + 1, out);
    }
}

fn done() -> BoxFuture<'static, ()> {
    Box::pin(async {})
}

impl Telemetry for TraceTreeReporter {
    fn on_start(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
        let operation = e
            .get("operationId")
            .and_then(|v| v.as_str())
            .unwrap_or("turn");
        let model = e
            .get("modelId")
            .and_then(|v| v.as_str())
            .filter(|m| !m.is_empty())
            .map(|m| format!(" {m}"))
            .unwrap_or_default();
        let turn = Turn {
            nodes: vec![Node {
                label: format!("{operation}{model}"),
                start_ms: now_ms(),
                end_ms: None,
                children: vec![],
            }],
            ..Default::default()
        };
        self.turns.lock().insert(call_id(e), turn);
        done()
    }

    fn on_step_start(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
        let mut turns = self.turns.lock();
        if let Some(turn) = turns.get_mut(&call_id(e)) {
            let number = e
                .get("stepNumber")
                .and_then(|v| v.as_u64())
                .unwrap_or(turn.nodes[0].children.len() as u64);
            turn.nodes.push(Node {
                label: format!("step {}", number + 1),
                start_ms: now_ms(),
                end_ms: None,
                children: vec![],
            });
            let index = turn.nodes.len() - 1;
            turn.step = Some(index);
            turn.nodes[0].children.push(index);
        }
        done()
    }

    fn on_tool_execution_start(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
        let mut turns = self.turns.lock();
        if let Some(turn) = turns.get_mut(&call_id(e)) {
            let call = e.get("toolCall").cloned().unwrap_or_default();
            let parent = turn.step.unwrap_or(0);
            turn.nodes.push(Node {
                label: format!("tool {}", call["toolName"].as_str().unwrap_or_default()),
                start_ms: now_ms(),
                end_ms: None,
                children: vec![],
            });
            let index = turn.nodes.len() - 1;
            turn.tools.insert(
                call["toolCallId"].as_str().unwrap_or_default().to_string(),
                index,
            );
            turn.nodes[parent].children.push(index);
        }
        done()
    }

    fn on_tool_execution_end(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
        let mut turns = self.turns.lock();
        if let Some(turn) = turns.get_mut(&call_id(e)) {
            let id = e
                .get("toolCall")
                .and_then(|c| c["toolCallId"].as_str())
                .unwrap_or_default();
            if let Some(&index) = turn.tools.get(id) {
                let node = &mut turn.nodes[index];
                node.end_ms = Some(now_ms());
                // [rust] Mirrors TS, which compares `toolOutput.type` with
                // `'error'`.
                if e.get("toolOutput").map(|o| &o["type"]) == Some(&"error".into()) {
                    node.label.push_str(" [error]");
                }
            }
        }
        done()
    }

    fn on_step_finish(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
        let mut turns = self.turns.lock();
        if let Some(turn) = turns.get_mut(&call_id(e))
            && let Some(step) = turn.step.take()
        {
            turn.nodes[step].end_ms = Some(now_ms());
        }
        done()
    }

    fn on_end(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
        let turn = self.turns.lock().remove(&call_id(e));
        if let Some(mut turn) = turn {
            turn.nodes[0].end_ms = Some(now_ms());
            let mut out = String::from("\n");
            render(&turn, 0, 0, &mut out);
            (self.write)(&out);
        }
        done()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn event(value: serde_json::Value) -> TelemetryEvent {
        value.as_object().unwrap().clone()
    }

    #[tokio::test]
    async fn renders_turn_steps_and_tools() {
        let out: Arc<Mutex<String>> = Arc::default();
        let sink = out.clone();
        let reporter = create_trace_tree_reporter(TraceTreeReporterOptions {
            write: Some(Arc::new(move |chunk: &str| sink.lock().push_str(chunk))),
        });
        reporter
            .on_start(&event(
                json!({"callId": "c", "operationId": "ai.harness", "modelId": "m"}),
            ))
            .await;
        reporter
            .on_step_start(&event(json!({"callId": "c", "stepNumber": 0})))
            .await;
        reporter
            .on_tool_execution_start(&event(
                json!({"callId": "c", "toolCall": {"toolName": "bash", "toolCallId": "t1"}}),
            ))
            .await;
        reporter
            .on_tool_execution_end(&event(json!({
                "callId": "c", "toolCall": {"toolCallId": "t1"}, "toolOutput": {"type": "error"}
            })))
            .await;
        reporter
            .on_step_finish(&event(json!({"callId": "c"})))
            .await;
        reporter.on_end(&event(json!({"callId": "c"}))).await;
        let text = out.lock().clone();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "");
        assert!(lines[1].starts_with("- ai.harness m ") && lines[1].ends_with("ms"));
        assert!(lines[2].starts_with("  - step 1 "));
        assert!(lines[3].starts_with("    - tool bash [error] "));
    }
}
