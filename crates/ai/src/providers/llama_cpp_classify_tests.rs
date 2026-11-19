//! llama.cpp classifier tests against a fake `llama-server`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::llama_cpp_classify::{
    answer_from_probabilities, label_probabilities, llama_server_root, peak_confidence,
    render_question,
};
use crate::classifier::{
    ClassifierAnswer, ClassifierContext, ClassifierOptions, ClassifierQuestion,
    ClassifierStopReason, classify,
};
use crate::providers::openai;
use crate::types::{ImageContent, Model};

type Next = Arc<dyn Fn(&str, u64) -> Vec<(&'static str, f64)> + Send + Sync>;
type Tokenize = Arc<dyn Fn(&str) -> Vec<i64> + Send + Sync>;
type Template = Arc<dyn Fn(&[Value]) -> String + Send + Sync>;

#[derive(Clone, Default)]
struct Fake {
    next: Option<Next>,
    tokenize: Option<Tokenize>,
    template: Option<Template>,
    /// Answer `/completion` with this status and body instead.
    fail: Option<(&'static str, &'static str)>,
}

/// One recorded request: its path, lowercased head, and JSON body.
type Recorded = Arc<Mutex<Vec<(String, String, Value)>>>;

struct Server {
    url: String,
    requests: Recorded,
}

impl Server {
    async fn start(fake: Fake) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        let requests: Recorded = Arc::default();
        let recorded = Arc::clone(&requests);
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let fake = fake.clone();
                let recorded = Arc::clone(&recorded);
                tokio::spawn(async move {
                    let raw = read_request(&mut socket).await;
                    let (head, body) = raw.split_once("\r\n\r\n").unwrap_or((&raw, ""));
                    let path = head
                        .split_whitespace()
                        .nth(1)
                        .unwrap_or_default()
                        .to_string();
                    let body: Value = serde_json::from_str(body).unwrap_or(Value::Null);
                    recorded.lock().expect("requests").push((
                        path.clone(),
                        head.to_ascii_lowercase(),
                        body.clone(),
                    ));
                    let (status, reply) = answer(&fake, &path, &body);
                    let response = format!(
                        "HTTP/1.1 {status}\r\ncontent-type: application/json\r\nconnection: close\r\ncontent-length: {}\r\n\r\n{reply}",
                        reply.len()
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                });
            }
        });
        Self {
            url: format!("http://{addr}/v1"),
            requests,
        }
    }

    fn requests(&self, path: &str) -> Vec<(String, Value)> {
        self.requests
            .lock()
            .expect("requests")
            .iter()
            .filter(|(request, _, _)| request == path)
            .map(|(_, head, body)| (head.clone(), body.clone()))
            .collect()
    }

    fn count(&self) -> usize {
        self.requests.lock().expect("requests").len()
    }

    fn model(&self) -> Model {
        openai::builder()
            .provider_id("llama.cpp")
            .api_key(Some("local"))
            .base_url(&self.url)
            .build()
            .expect("provider")
            .llama_cpp_classifier_model("qwen")
            .build_classifier()
            .expect("model")
    }
}

fn answer(fake: &Fake, path: &str, body: &Value) -> (&'static str, String) {
    match path {
        "/tokenize" => {
            let content = body["content"].as_str().unwrap_or_default();
            let tokens = match &fake.tokenize {
                Some(tokenize) => tokenize(content),
                None => char_tokens(content),
            };
            ("200 OK", json!({ "tokens": tokens }).to_string())
        }
        "/apply-template" => {
            let messages = body["messages"].as_array().cloned().unwrap_or_default();
            let prompt = match &fake.template {
                Some(template) => template(&messages),
                None => {
                    let mut prompt = messages
                        .iter()
                        .map(|message| {
                            format!(
                                "<|{}|>\n{}\n",
                                message["role"].as_str().unwrap_or_default(),
                                message["content"].as_str().unwrap_or_default()
                            )
                        })
                        .collect::<String>();
                    prompt.push_str("<|assistant|>\n");
                    prompt
                }
            };
            ("200 OK", json!({ "prompt": prompt }).to_string())
        }
        "/completion" => {
            if let Some((status, body)) = fake.fail {
                return (status, body.to_string());
            }
            let prompt = body["prompt"].as_str().unwrap_or_default();
            let depth = body["n_probs"].as_u64().unwrap_or_default();
            let next = match &fake.next {
                Some(next) => next(prompt, depth),
                None => vec![("A", -0.1), ("B", -2.5)],
            };
            let top = next
                .iter()
                .map(|(token, logprob)| {
                    json!({
                        "id": token.chars().next().map_or(0, |char| char as i64),
                        "token": token,
                        "bytes": [],
                        "logprob": logprob,
                    })
                })
                .collect::<Vec<_>>();
            (
                "200 OK",
                json!({
                    "content": "A",
                    "completion_probabilities": [{"id": 65, "token": "A", "top_logprobs": top}]
                })
                .to_string(),
            )
        }
        _ => ("404 Not Found", "{}".to_string()),
    }
}

async fn read_request(socket: &mut tokio::net::TcpStream) -> String {
    let mut buffer = Vec::new();
    let mut temp = [0; 8192];
    while let Ok(read) = socket.read(&mut temp).await {
        buffer.extend_from_slice(&temp[..read]);
        if let Some(end) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buffer[..end]).to_ascii_lowercase();
            let length = head
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            if buffer.len() >= end + 4 + length {
                break;
            }
        }
        if read == 0 {
            break;
        }
    }
    String::from_utf8_lossy(&buffer).into_owned()
}

/// Token IDs: one per character, the character code.
fn char_tokens(content: &str) -> Vec<i64> {
    content.chars().map(|char| char as i64).collect()
}

/// Maps `Yes` and `No` to single tokens, as a real vocabulary would.
fn word_tokens(content: &str) -> Vec<i64> {
    let words = HashMap::from([("Yes", 89), ("No", 78)]);
    let mut tokens = Vec::new();
    let mut rest = content;
    while !rest.is_empty() {
        if let Some(after) = rest.strip_prefix('\n') {
            tokens.push(10);
            rest = after;
            continue;
        }
        let end = rest.find('\n').unwrap_or(rest.len());
        let part = &rest[..end];
        match words.get(part) {
            Some(id) => tokens.push(*id),
            None => tokens.extend(char_tokens(part)),
        }
        rest = &rest[end..];
    }
    tokens
}

/// Log-probabilities for bool (Yes/No), score and letter labels; a token's
/// ID is its first character.
fn answer_by_prompt(prompt: &str, _depth: u64) -> Vec<(&'static str, f64)> {
    if prompt.contains("Answer Yes or No.") {
        vec![("Y", -0.05), ("N", -3.0)]
    } else if prompt.contains("Answer with one level number.") {
        vec![("2", -0.2), ("1", -1.8), ("0", -4.0)]
    } else {
        vec![("B", -0.3), ("A", -1.5), ("C", -3.0)]
    }
}

fn ticket() -> ClassifierContext {
    ClassifierContext::from_state(
        json!({"message": "Help! My payouts have been failing for 3 days."}),
    )
    .expect("state")
    .question(
        "team",
        ClassifierQuestion::choice(
            "Which team should handle this?",
            [
                ("billing", "Payments and refunds"),
                ("technical", "Bugs and outages"),
                ("sales", ""),
            ],
        ),
    )
    .question(
        "urgent",
        ClassifierQuestion::bool(
            "Does this convey urgency?",
            "The user needs help soon",
            "No time pressure",
        ),
    )
    .question(
        "severity",
        ClassifierQuestion::score("How severe is this?", ["low", "medium", "high"]),
    )
}

fn pick() -> ClassifierContext {
    ClassifierContext::default().question(
        "pick",
        ClassifierQuestion::choice("Pick one", [("a", ""), ("b", "")]),
    )
}

#[tokio::test]
async fn answers_choice_bool_and_score_questions_from_label_logprobs() {
    let server = Server::start(Fake {
        next: Some(Arc::new(answer_by_prompt)),
        tokenize: Some(Arc::new(word_tokens)),
        ..Fake::default()
    })
    .await;

    let result = classify(server.model(), ticket(), None).await;

    assert_eq!(result.error_message, None);
    assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
    assert_eq!(result.api, "llama-cpp-classify");
    let team = label_probabilities(&[-1.5, -0.3, -3.0], 1.0);
    assert_eq!(
        result.answer("team"),
        Some(&ClassifierAnswer::Choice {
            choice: "technical".to_string(),
            probabilities: [
                ("billing", team[0]),
                ("technical", team[1]),
                ("sales", team[2])
            ]
            .into_iter()
            .map(|(key, probability)| (key.to_string(), probability))
            .collect(),
            confidence: peak_confidence(&team),
        })
    );
    assert_eq!(
        result.probability("urgent"),
        Some(label_probabilities(&[-0.05, -3.0], 1.0)[0])
    );
    let severity = label_probabilities(&[-4.0, -1.8, -0.2], 1.0);
    assert_eq!(
        result.answer("severity"),
        Some(&ClassifierAnswer::Score {
            score: severity[1] + 2.0 * severity[2],
            confidence: peak_confidence(&severity),
        })
    );
    assert_eq!(result.usage, None);

    for path in ["/tokenize", "/apply-template", "/completion"] {
        for (head, body) in server.requests(path) {
            assert_eq!(body["model"], "qwen");
            assert!(head.contains("\r\nauthorization: bearer local"), "{head}");
        }
    }
    let (_, completion) = server.requests("/completion")[0].clone();
    assert_eq!(completion["n_predict"], 1);
    assert_eq!(completion["n_probs"], 256);
    assert_eq!(completion["post_sampling_probs"], false);
    assert_eq!(completion["cache_prompt"], true);
    assert_eq!(completion["temperature"], 0);
    let (_, template) = server.requests("/apply-template")[0].clone();
    assert_eq!(
        template["chat_template_kwargs"],
        json!({"enable_thinking": false})
    );
}

#[test]
fn repeats_the_state_around_all_questions_and_ends_with_the_labels() {
    let rendered = render_question(&ticket(), "team").expect("rendered");
    let state = "State:\n{\n \"message\": \"Help! My payouts have been failing for 3 days.\"\n}";
    assert_eq!(rendered.labels, ["A", "B", "C"]);
    assert_eq!(rendered.keys, ["billing", "technical", "sales"]);
    assert_eq!(
        rendered.content,
        [
            state,
            "",
            "Task: answer each of the following questions about the state.",
            "",
            "Question: Which team should handle this?",
            "",
            "Options:",
            "- billing: Payments and refunds",
            "- technical: Bugs and outages",
            "- sales",
            "",
            "Question: Does this convey urgency?",
            "",
            "Yes means: The user needs help soon",
            "No means: No time pressure",
            "",
            "Question: How severe is this?",
            "",
            "Levels:",
            "0. low",
            "1. medium",
            "2. high",
            "",
            state,
            "",
            "Question: Which team should handle this?",
            "",
            "Options:",
            "A. billing: Payments and refunds",
            "B. technical: Bugs and outages",
            "C. sales",
            "",
            "Answer with one letter.",
        ]
        .join("\n")
    );
}

#[test]
fn shares_everything_before_the_final_question() {
    let context = ticket();
    let prefix = |id: &str| {
        let content = render_question(&context, id).expect("rendered").content;
        content[..content.rfind("Question:").expect("question")].to_string()
    };
    assert_eq!(prefix("urgent"), prefix("team"));
    assert_eq!(prefix("severity"), prefix("team"));
    assert!(
        render_question(&context, "urgent")
            .expect("rendered")
            .content
            .ends_with("No means: No time pressure\n\nAnswer Yes or No.")
    );
    assert!(
        render_question(&context, "severity")
            .expect("rendered")
            .content
            .ends_with("2. high\n\nAnswer with one level number.")
    );
}

#[tokio::test]
async fn divides_label_logprobs_by_the_temperature() {
    let server = Server::start(Fake::default()).await;
    let options = ClassifierOptions::builder().temperature(2.0).build();

    let result = classify(server.model(), pick(), Some(options)).await;

    let expected = label_probabilities(&[-0.1, -2.5], 2.0);
    let Some(ClassifierAnswer::Choice { probabilities, .. }) = result.answer("pick") else {
        panic!("{result:?}");
    };
    assert_eq!(probabilities["a"], expected[0]);
    assert_eq!(probabilities["b"], expected[1]);
    let weight = (2.4f64 / 2.0).exp();
    assert!((expected[0] - weight / (weight + 1.0)).abs() < 1e-12);
}

#[tokio::test]
async fn rejects_non_positive_temperatures_before_any_request() {
    let server = Server::start(Fake::default()).await;
    let options = ClassifierOptions::builder().temperature(0.0).build();

    let result = classify(server.model(), pick(), Some(options)).await;

    assert_eq!(result.stop_reason, ClassifierStopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("Temperature must be a positive number, got 0")
    );
    assert_eq!(server.count(), 0);
}

fn depths(server: &Server) -> Vec<u64> {
    server
        .requests("/completion")
        .iter()
        .map(|(_, body)| body["n_probs"].as_u64().expect("depth"))
        .collect()
}

#[tokio::test]
async fn reads_deeper_when_a_label_is_missing_and_never_invents_zeros() {
    let deep = Server::start(Fake {
        next: Some(Arc::new(|_, depth| {
            if depth >= 4096 {
                vec![("A", -0.1), ("B", -9.0)]
            } else {
                vec![("A", -0.1)]
            }
        })),
        ..Fake::default()
    })
    .await;
    let recovered = classify(deep.model(), pick(), None).await;
    assert_eq!(
        recovered.stop_reason,
        ClassifierStopReason::Stop,
        "{recovered:?}"
    );
    assert_eq!(depths(&deep), [256, 4096]);

    let never = Server::start(Fake {
        next: Some(Arc::new(|_, _| vec![("A", -0.1)])),
        ..Fake::default()
    })
    .await;
    let failed = classify(never.model(), pick(), None).await;
    assert_eq!(failed.stop_reason, ClassifierStopReason::Error);
    assert!(failed.answers.is_empty());
    assert_eq!(
        failed.error_message.as_deref(),
        Some("llama.cpp did not rank labels B for pick within the top 32768 tokens")
    );
    assert_eq!(depths(&never), [256, 4096, 32768]);
}

#[tokio::test]
async fn fails_when_every_label_underflows() {
    let server = Server::start(Fake {
        next: Some(Arc::new(|_, _| vec![("A", -1e30), ("B", -1e30)])),
        ..Fake::default()
    })
    .await;
    let result = classify(server.model(), pick(), None).await;
    assert_eq!(
        result.error_message.as_deref(),
        Some("qwen gave no probability to any answer label for pick")
    );
}

#[tokio::test]
async fn closes_a_reasoning_block_the_template_leaves_open() {
    let server = Server::start(Fake {
        template: Some(Arc::new(|_| "<|assistant|>\n<think>".to_string())),
        ..Fake::default()
    })
    .await;

    let result = classify(server.model(), pick(), None).await;

    assert_eq!(result.stop_reason, ClassifierStopReason::Stop, "{result:?}");
    assert_eq!(
        server.requests("/completion")[0].1["prompt"],
        "<|assistant|>\n<think></think>"
    );
}

#[tokio::test]
async fn reads_labels_in_reply_position_and_rejects_labels_of_several_tokens() {
    // "\nYes" merges into one token that differs from "Yes" alone.
    let merged = Server::start(Fake {
        tokenize: Some(Arc::new(|content| match content {
            "\n" => vec![10],
            "\nYes" => vec![10, 89],
            "\nNo" => vec![10, 78],
            other => char_tokens(other),
        })),
        next: Some(Arc::new(|_, _| vec![("Y", -0.2), ("N", -1.7)])),
        ..Fake::default()
    })
    .await;
    let context =
        ClassifierContext::default().question("ok", ClassifierQuestion::bool("Fine?", "", ""));
    let result = classify(merged.model(), context.clone(), None).await;
    assert_eq!(result.stop_reason, ClassifierStopReason::Stop, "{result:?}");
    assert_eq!(
        result.probability("ok"),
        Some(label_probabilities(&[-0.2, -1.7], 1.0)[0])
    );

    let split = Server::start(Fake::default()).await;
    let result = classify(split.model(), context, None).await;
    assert_eq!(result.stop_reason, ClassifierStopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("Label \"Yes\" is not a single token for qwen")
    );
}

#[tokio::test]
async fn caches_label_tokens_per_server_and_model() {
    let server = Server::start(Fake::default()).await;
    let model = server.model();

    classify(model.clone(), pick(), None).await;
    let first = server.requests("/tokenize").len();
    classify(model, pick(), None).await;

    assert!(first > 0);
    assert_eq!(server.requests("/tokenize").len(), first);
}

#[tokio::test]
async fn checks_option_counts_before_any_request() {
    let server = Server::start(Fake::default()).await;
    let too_many = ClassifierContext::default().question(
        "pick",
        ClassifierQuestion::choice(
            "Pick",
            (0..63).map(|index| (format!("o{index}"), String::new())),
        ),
    );
    let too_few =
        ClassifierContext::default().question("level", ClassifierQuestion::score("Rate", ["only"]));

    let many = classify(server.model(), too_many, None).await;
    let few = classify(server.model(), too_few, None).await;

    assert_eq!(
        many.error_message.as_deref(),
        Some("A choice question needs 2 to 62 options, got 63")
    );
    assert_eq!(
        few.error_message.as_deref(),
        Some("A score question needs 2 to 10 levels, got 1")
    );
    assert_eq!(server.count(), 0);
}

#[tokio::test]
async fn passes_completion_payloads_and_responses_through_the_hooks() {
    let server = Server::start(Fake::default()).await;
    let payloads = Arc::new(Mutex::new(Vec::new()));
    let statuses = Arc::new(Mutex::new(Vec::new()));
    let mut options = ClassifierOptions::default();
    let seen = Arc::clone(&payloads);
    options.base.on_payload = Some(Arc::new(move |payload: Value, _model: &Model| {
        seen.lock().expect("payloads").push(payload.clone());
        let mut payload = payload;
        payload["id_slot"] = json!(1);
        Box::pin(async move { Ok(Some(payload)) })
    }));
    let seen = Arc::clone(&statuses);
    options.base.on_response = Some(Arc::new(move |response, _model: &Model| {
        seen.lock().expect("statuses").push(response.status);
        Box::pin(async { Ok(()) })
    }));

    let result = classify(server.model(), pick(), Some(options)).await;

    assert_eq!(result.stop_reason, ClassifierStopReason::Stop, "{result:?}");
    let payloads = payloads.lock().expect("payloads");
    assert_eq!(payloads.len(), 1);
    assert_eq!(payloads[0]["n_predict"], 1);
    assert_eq!(*statuses.lock().expect("statuses"), [200]);
    assert_eq!(server.requests("/completion")[0].1["id_slot"], 1);
}

#[tokio::test]
async fn reports_server_errors_and_cancellation() {
    let failing = Server::start(Fake {
        fail: Some(("400 Bad Request", r#"{"error":"context overflow"}"#)),
        ..Fake::default()
    })
    .await;
    let result = classify(failing.model(), pick(), None).await;
    assert_eq!(result.stop_reason, ClassifierStopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some(r#"llama.cpp error (400): {"error":"context overflow"}"#)
    );

    let server = Server::start(Fake::default()).await;
    let token = tokio_util::sync::CancellationToken::new();
    token.cancel();
    let options = ClassifierOptions::builder()
        .cancellation_token(token)
        .build();
    let aborted = classify(server.model(), pick(), Some(options)).await;
    assert_eq!(aborted.stop_reason, ClassifierStopReason::Aborted);
}

#[tokio::test]
async fn takes_no_images_and_no_other_classifier_apis() {
    let server = Server::start(Fake::default()).await;
    let image = pick().image(ImageContent {
        data: "AA==".to_string(),
        mime_type: "image/png".to_string(),
    });
    let result = classify(server.model(), image, None).await;
    assert_eq!(
        result.error_message.as_deref(),
        Some("Model llama.cpp/qwen does not accept image input")
    );

    let mut model = server.model();
    model.api = "typesafe-system-one".to_string();
    let result = classify(model, pick(), None).await;
    assert_eq!(
        result.error_message.as_deref(),
        Some("Unsupported classifier API: typesafe-system-one")
    );
}

#[test]
fn the_server_root_drops_a_trailing_v1() {
    assert_eq!(
        llama_server_root("http://127.0.0.1:9931/v1/"),
        "http://127.0.0.1:9931"
    );
    assert_eq!(
        llama_server_root("http://127.0.0.1:9931"),
        "http://127.0.0.1:9931"
    );
}

#[test]
fn bool_answers_take_the_probability_of_true() {
    let question = ClassifierQuestion::bool("Ok?", "", "");
    assert_eq!(
        answer_from_probabilities(
            &question,
            &["true".to_string(), "false".to_string()],
            &[0.7, 0.3]
        ),
        ClassifierAnswer::Bool { probability: 0.7 }
    );
}
