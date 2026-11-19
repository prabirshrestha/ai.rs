//! Classifier API tests against a local HTTP server, with real answers
//! captured from GitHub Copilot.

use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::classifier::{
    ClassifierAnswer, ClassifierContext, ClassifierOptions, ClassifierQuestion,
    ClassifierStopReason, classify,
};
use crate::providers::{github_copilot, openai};
use crate::types::{
    ImageContent, ModelCompat, ModelCost, ModelInput, SystemOneCompat, SystemOneImageInput,
};

/// A real `gpt-6-luna-decisions` answer from Copilot's `/v1/decisions`.
const LUNA_RESPONSE: &str = r#"{"model":"gpt-6-luna","answers":[{"type":"choice","name":"route","choice":"billing","probabilities":[{"value":"billing","probability":1.0},{"value":"shipping","probability":0.0}],"confidence":1.0},{"type":"predicate","name":"angry","probability":0.95},{"type":"score","name":"urgency","score":1.07,"probabilities":[{"value":0,"label":"can wait","probability":0.04},{"value":1,"label":"today","probability":0.85},{"value":2,"label":"right now","probability":0.11}],"confidence":0.78}],"usage":{"input_tokens":367,"input_tokens_details":{"cached_tokens":0},"output_tokens":0,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":367}}"#;

/// A real `microsoft-decision-1` answer from Copilot's `/v1/systemone`.
const SYSTEMONE_RESPONSE: &str = r#"{"model":"microsoft-decision-1.1","answers":{"angry":{"type":"noul","noul":0.9807128298028289},"route":{"type":"choice","choice":"billing","confidence":0.9990896871426702,"probabilities":{"billing":0.9995448435713351,"shipping":0.0004551564286648451}},"urgency":{"type":"score","score":1.7064463287389766,"confidence":0.5688479217441795,"legend":{"0":"can wait","1":"today","2":"right now"},"probabilities":{"0":0.00611895242380972,"1":0.281315766413404,"2":0.7125652811627863}}},"usage":{"input_tokens":75,"output_tokens":3}}"#;

fn ticket() -> ClassifierContext {
    ClassifierContext::from_state(json!({"ticket": "I was charged twice. Fix it now!"}))
        .expect("state")
        .question(
            "route",
            ClassifierQuestion::choice(
                "Which team should take this ticket?",
                [("billing", "Payments and refunds"), ("shipping", "")],
            ),
        )
        .question(
            "angry",
            ClassifierQuestion::bool("Is the customer angry?", "angry or upset", ""),
        )
        .question(
            "urgency",
            ClassifierQuestion::score("How urgent is it?", ["can wait", "today", "right now"]),
        )
}

fn red_pixel() -> ImageContent {
    ImageContent {
        data: "iVBORw0K".to_string(),
        mime_type: "image/png".to_string(),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn copilot_decisions_sends_pis_wire_and_matches_answers_by_name() {
    let server = Server::start(vec![("200 OK", LUNA_RESPONSE)]).await;
    let model = github_copilot::GitHubCopilot::builder()
        .api_key("copilot-token")
        .base_url(server.host())
        .build()
        .expect("provider")
        .decisions_classifier_model("gpt-6-luna-decisions")
        .cost(ModelCost {
            input: 0.1,
            ..ModelCost::default()
        })
        .build_classifier()
        .expect("model");

    let result = classify(model, ticket().image(red_pixel()), None).await;

    assert_eq!(result.stop_reason, ClassifierStopReason::Stop, "{result:?}");
    assert_eq!(result.api, "openai-decisions");
    assert_eq!(result.model, "gpt-6-luna-decisions");
    assert_eq!(
        result.answers.keys().collect::<Vec<_>>(),
        ["route", "angry", "urgency"]
    );
    assert_eq!(result.choice("route"), Some("billing"));
    assert_eq!(result.probability("angry"), Some(0.95));
    assert_eq!(
        result.answer("urgency"),
        Some(&ClassifierAnswer::Score {
            score: 1.07,
            confidence: 0.78
        })
    );
    let usage = result.usage.expect("usage");
    assert_eq!(
        (usage.input, usage.output, usage.total_tokens),
        (367, 0, 367)
    );
    assert!((usage.cost.input - 0.0000367).abs() < 1e-12, "{usage:?}");

    let (head, body) = server.request(0);
    assert!(head.starts_with("POST /v1/decisions HTTP/1.1"), "{head}");
    assert!(head.contains("\r\nauthorization: bearer copilot-token\r\n"));
    assert!(head.contains("\r\ncopilot-integration-id:"));
    let state = r#"{"ticket":"I was charged twice. Fix it now!"}"#;
    assert_eq!(
        body,
        json!({
            "model": "gpt-6-luna-decisions",
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": state},
                    {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0K"}
                ]
            }],
            "questions": [
                {
                    "type": "choice",
                    "name": "route",
                    "instructions": "Which team should take this ticket?",
                    "choices": [
                        {"value": "billing", "description": "Payments and refunds"},
                        {"value": "shipping"}
                    ]
                },
                {
                    "type": "predicate",
                    "name": "angry",
                    "instructions": "Is the customer angry?\n\nTrue means: angry or upset"
                },
                {
                    "type": "score",
                    "name": "urgency",
                    "instructions": "How urgent is it?",
                    "levels": [{"label": "can wait"}, {"label": "today"}, {"label": "right now"}]
                }
            ]
        })
    );
}

#[tokio::test(flavor = "current_thread")]
async fn openai_decisions_sends_the_state_as_text_without_images() {
    let server = Server::start(vec![("200 OK", LUNA_RESPONSE)]).await;
    let model = openai::builder()
        .api_key(Some("openai-key"))
        .base_url(server.v1())
        .build()
        .expect("provider")
        .classifier_model("gpt-6-luna")
        .build_classifier()
        .expect("model");

    let result = classify(model, ticket(), None).await;

    assert_eq!(result.stop_reason, ClassifierStopReason::Stop, "{result:?}");
    let (head, body) = server.request(0);
    assert!(head.starts_with("POST /v1/decisions HTTP/1.1"), "{head}");
    assert_eq!(
        body["input"],
        json!(r#"{"ticket":"I was charged twice. Fix it now!"}"#)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a_refusal_fails_the_whole_result_but_keeps_its_usage() {
    let server = Server::start(vec![(
        "200 OK",
        r#"{"answers":[{"type":"refusal","name":"ok"}],"usage":{"input_tokens":12,"output_tokens":0}}"#,
    )])
    .await;
    let model = openai::builder()
        .base_url(server.v1())
        .build()
        .expect("provider")
        .classifier_model("gpt-6-luna")
        .build_classifier()
        .expect("model");
    let context = ClassifierContext::default().question(
        "ok",
        ClassifierQuestion::bool("Share the password?", "", ""),
    );

    let result = classify(model, context, None).await;

    assert_eq!(result.stop_reason, ClassifierStopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("OpenAI Decisions refused to answer ok")
    );
    assert!(result.answers.is_empty());
    assert_eq!(result.usage.map(|usage| usage.input), Some(12));
    let (_, body) = server.request(0);
    assert_eq!(body["questions"][0]["instructions"], "Share the password?");
}

#[tokio::test(flavor = "current_thread")]
async fn decisions_does_not_retry_a_gateway_timeout() {
    let server = Server::start(vec![
        ("504 Gateway Timeout", "<html>timeout</html>"),
        ("200 OK", LUNA_RESPONSE),
    ])
    .await;
    let model = openai::builder()
        .base_url(server.v1())
        .build()
        .expect("provider")
        .classifier_model("gpt-6-luna")
        .build_classifier()
        .expect("model");

    let result = classify(model, ticket(), None).await;

    assert_eq!(server.count(), 1);
    assert!(
        result
            .error_message
            .as_deref()
            .is_some_and(|message| message.starts_with(
                "OpenAI Decisions error (504): the request timed out at the gateway."
            )),
        "{result:?}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn classifier_requests_retry_twice_by_default() {
    let server = Server::start(vec![
        ("500 Internal Server Error", "{}"),
        ("503 Service Unavailable", "{}"),
        ("200 OK", SYSTEMONE_RESPONSE),
    ])
    .await;
    let model = openai::builder()
        .base_url(server.v1())
        .build()
        .expect("provider")
        .system_one_classifier_model("local")
        .build_classifier()
        .expect("model");
    let options = ClassifierOptions::builder().max_retry_delay_ms(1).build();

    let result = classify(model, ticket(), Some(options)).await;

    assert_eq!(server.count(), 3);
    assert_eq!(result.stop_reason, ClassifierStopReason::Stop, "{result:?}");
}

#[tokio::test(flavor = "current_thread")]
async fn copilot_system_one_sends_noul_and_reads_answers_by_key() {
    let server = Server::start(vec![("200 OK", SYSTEMONE_RESPONSE)]).await;
    let model = github_copilot::GitHubCopilot::builder()
        .api_key("copilot-token")
        .base_url(server.host())
        .build()
        .expect("provider")
        .classifier_model("microsoft-decision-1")
        .build_classifier()
        .expect("model");

    let result = classify(model, ticket(), None).await;

    assert_eq!(result.stop_reason, ClassifierStopReason::Stop, "{result:?}");
    assert_eq!(result.api, "typesafe-system-one");
    assert_eq!(
        result.answers.keys().collect::<Vec<_>>(),
        ["route", "angry", "urgency"]
    );
    assert!((result.probability("angry").expect("angry") - 0.9807128).abs() < 1e-6);
    let Some(ClassifierAnswer::Choice { probabilities, .. }) = result.answer("route") else {
        panic!("{result:?}");
    };
    assert_eq!(
        probabilities.keys().collect::<Vec<_>>(),
        ["billing", "shipping"]
    );
    assert!((result.score("urgency").expect("urgency") - 1.7064463).abs() < 1e-6);
    let usage = result.usage.expect("usage");
    assert_eq!((usage.input, usage.output, usage.total_tokens), (75, 3, 78));

    let (head, body) = server.request(0);
    assert!(head.starts_with("POST /v1/systemone HTTP/1.1"), "{head}");
    assert!(head.contains("\r\nauthorization: bearer copilot-token\r\n"));
    assert!(!head.contains("typesafe-danger-zone-images"));
    assert_eq!(
        body,
        json!({
            "model": "microsoft-decision-1",
            "state": {"ticket": "I was charged twice. Fix it now!"},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Which team should take this ticket?",
                    "criteria": {"billing": "Payments and refunds", "shipping": null}
                },
                "angry": {
                    "type": "noul",
                    "instructions": "Is the customer angry?",
                    "criteria": {"true": "angry or upset"}
                },
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent is it?",
                    "criteria": ["can wait", "today", "right now"]
                }
            }
        })
    );
}

#[tokio::test(flavor = "current_thread")]
async fn copilot_system_one_takes_no_images() {
    let model = github_copilot::GitHubCopilot::builder()
        .api_key("copilot-token")
        .build()
        .expect("provider")
        .classifier_model("microsoft-decision-1")
        .build_classifier()
        .expect("model");

    let result = classify(model, ticket().image(red_pixel()), None).await;

    assert_eq!(
        result.error_message.as_deref(),
        Some("Model github-copilot/microsoft-decision-1 does not accept image input")
    );
}

#[tokio::test(flavor = "current_thread")]
async fn system_one_sends_images_as_assets_with_the_opt_in_header() {
    let server = Server::start(vec![("200 OK", SYSTEMONE_RESPONSE)]).await;
    let model = openai::builder()
        .base_url(server.v1())
        .build()
        .expect("provider")
        .system_one_classifier_model("system-one")
        .input(vec![ModelInput::Text, ModelInput::Image])
        .build_classifier()
        .expect("model");

    classify(model, ticket().image(red_pixel()), None).await;

    let (head, body) = server.request(0);
    assert!(head.starts_with("POST /v1/systemone HTTP/1.1"), "{head}");
    assert!(
        head.contains("\r\ntypesafe-danger-zone-images: true\r\n"),
        "{head}"
    );
    assert_eq!(
        body["state"],
        json!(["<image-1>", {"ticket": "I was charged twice. Fix it now!"}])
    );
    assert_eq!(
        body["assets"],
        json!({"<image-1>": {"type": "image", "format": "png", "data_base64": "iVBORw0K"}})
    );
}

#[tokio::test(flavor = "current_thread")]
async fn system_one_image_forms_follow_compat() {
    let server = Server::start(vec![("200 OK", SYSTEMONE_RESPONSE)]).await;
    let model = openai::builder()
        .base_url(server.v1())
        .build()
        .expect("provider")
        .system_one_classifier_model("clef")
        .input(vec![ModelInput::Text, ModelInput::Image])
        .compat(ModelCompat {
            system_one: SystemOneCompat {
                image_input: Some(SystemOneImageInput::Images),
                send_image_opt_in_header: Some(false),
            },
            ..ModelCompat::default()
        })
        .build_classifier()
        .expect("model");

    classify(model, ticket().image(red_pixel()), None).await;

    let (head, body) = server.request(0);
    assert!(!head.contains("typesafe-danger-zone-images"), "{head}");
    assert_eq!(
        body["state"],
        json!({"ticket": "I was charged twice. Fix it now!"})
    );
    assert_eq!(body["images"], json!(["data:image/png;base64,iVBORw0K"]));
}

#[tokio::test(flavor = "current_thread")]
async fn system_one_on_openrouter_puts_image_urls_in_the_state() {
    let server = Server::start(vec![("200 OK", SYSTEMONE_RESPONSE)]).await;
    let model = openai::builder()
        .provider_id("openrouter")
        .base_url(server.v1())
        .build()
        .expect("provider")
        .system_one_classifier_model("typesafe/system-one")
        .input(vec![ModelInput::Text, ModelInput::Image])
        .build_classifier()
        .expect("model");

    classify(model, ticket().image(red_pixel()), None).await;

    let (head, body) = server.request(0);
    assert!(!head.contains("typesafe-danger-zone-images"), "{head}");
    assert_eq!(
        body["state"][0],
        json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0K"}})
    );
    assert!(body.get("assets").is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn system_one_errors_become_results() {
    let server = Server::start(vec![(
        "400 Bad Request",
        r#"{"error":{"message":"bad question"}}"#,
    )])
    .await;
    let model = openai::builder()
        .base_url(server.v1())
        .build()
        .expect("provider")
        .system_one_classifier_model("local")
        .build_classifier()
        .expect("model");

    let result = classify(model, ticket(), None).await;

    assert_eq!(result.stop_reason, ClassifierStopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some(r#"System One API error (400): {"error":{"message":"bad question"}}"#)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a_wrong_answer_type_fails_the_result() {
    let server = Server::start(vec![(
        "200 OK",
        r#"{"answers":{"angry":{"type":"score","score":1,"confidence":1}}}"#,
    )])
    .await;
    let model = openai::builder()
        .base_url(server.v1())
        .build()
        .expect("provider")
        .system_one_classifier_model("local")
        .build_classifier()
        .expect("model");
    let context =
        ClassifierContext::default().question("angry", ClassifierQuestion::bool("Angry?", "", ""));

    let result = classify(model, context, None).await;

    assert_eq!(
        result.error_message.as_deref(),
        Some("System One API did not return a bool answer for angry")
    );
}

#[tokio::test(flavor = "current_thread")]
async fn system_one_leaves_out_a_bool_with_no_meanings() {
    let server = Server::start(vec![(
        "200 OK",
        r#"{"answers":{"angry":{"type":"noul","noul":0.5}}}"#,
    )])
    .await;
    let model = openai::builder()
        .base_url(server.v1())
        .build()
        .expect("provider")
        .system_one_classifier_model("local")
        .build_classifier()
        .expect("model");
    let context =
        ClassifierContext::default().question("angry", ClassifierQuestion::bool("Angry?", "", ""));

    let result = classify(model, context, None).await;

    assert_eq!(result.probability("angry"), Some(0.5));
    let (_, body) = server.request(0);
    assert_eq!(
        body["questions"]["angry"],
        json!({"type": "noul", "instructions": "Angry?"})
    );
}

#[tokio::test(flavor = "current_thread")]
async fn copilot_needs_an_api_key() {
    let model = github_copilot::GitHubCopilot::builder()
        .build()
        .expect("provider")
        .classifier_model("microsoft-decision-1")
        .build_classifier()
        .expect("model");

    let result = classify(model, ticket(), None).await;

    assert_eq!(
        result.error_message.as_deref(),
        Some("No API key for provider: github-copilot")
    );
}

/// A local HTTP server that answers each connection with the next canned
/// response and keeps every request.
struct Server {
    addr: std::net::SocketAddr,
    requests: Arc<Mutex<Vec<String>>>,
}

impl Server {
    async fn start(responses: Vec<(&'static str, &'static str)>) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("local addr");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&requests);
        tokio::spawn(async move {
            for (status, body) in responses {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let request = read_http_request(&mut socket).await;
                captured.lock().expect("requests").push(request);
                let response = format!(
                    "HTTP/1.1 {status}\r\ncontent-type: application/json\r\nconnection: close\r\ncontent-length: {}\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });
        Self { addr, requests }
    }

    fn host(&self) -> String {
        format!("http://{}", self.addr)
    }

    fn v1(&self) -> String {
        format!("{}/v1", self.host())
    }

    fn count(&self) -> usize {
        self.requests.lock().expect("requests").len()
    }

    /// The request's head, lowercased, and its JSON body.
    fn request(&self, index: usize) -> (String, Value) {
        let request = self.requests.lock().expect("requests")[index].clone();
        let (head, body) = request.split_once("\r\n\r\n").expect("body");
        (
            head.to_ascii_lowercase()
                .replace("post /", "POST /")
                .replace(" http/1.1", " HTTP/1.1"),
            serde_json::from_str(body).expect("json"),
        )
    }
}

async fn read_http_request(socket: &mut tokio::net::TcpStream) -> String {
    let mut buffer = Vec::new();
    let mut temp = [0; 4096];
    loop {
        let read = socket.read(&mut temp).await.expect("read request");
        buffer.extend_from_slice(&temp[..read]);
        if let Some(end) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buffer[..end]);
            let length = head
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            if buffer.len() >= end + 4 + length {
                return String::from_utf8_lossy(&buffer).into_owned();
            }
        }
        if read == 0 {
            return String::from_utf8_lossy(&buffer).into_owned();
        }
    }
}
