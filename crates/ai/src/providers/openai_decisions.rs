//! OpenAI's Decisions API: `POST {base_url}/decisions` with `{ model, input,
//! questions }`. Ports Pi's `openai-decisions.ts`.
//!
//! The state goes as JSON text. With images, the input is one user message
//! with the state as `input_text` and then `input_image` data URLs. `choice`
//! goes as `choice`, `score` as `score`, and `bool` as `predicate`. A
//! predicate has no criteria, so the meanings of true and false are added to
//! its instructions. A refusal fails the whole result.

use async_trait::async_trait;
use indexmap::IndexMap;
use serde_json::{Map, Value, json};

use crate::classifier::{
    ClassifierAnswer, ClassifierContext, ClassifierOptions, ClassifierQuestion, ClassifierResult,
};
use crate::provider::ClassifierModelApi;
use crate::providers::classifier_shared::{
    ClassifierAuth, ClassifierError, ClassifierOutcome, as_record, finish, parse_classifier_usage,
    post_classifier_request, required_number,
};
use crate::types::{KnownApi, Model};

const LABEL: &str = "OpenAI Decisions";

/// The endpoint takes at most this many image parts per request.
const MAX_IMAGES: usize = 128;

/// Cloudflare in front of api.openai.com answers 504 when a request runs
/// longer than about five seconds. Very large inputs hit this limit again on
/// a retry, so 504 is not retried.
const NO_RETRY_STATUSES: &[u16] = &[504];

#[derive(Clone)]
pub(crate) struct OpenAiDecisionsClassifierApi {
    auth: ClassifierAuth,
}

impl OpenAiDecisionsClassifierApi {
    pub(crate) fn new(
        api_key: Option<String>,
        allow_missing_api_key: bool,
        http_client: Option<reqwest::Client>,
    ) -> Self {
        Self {
            auth: ClassifierAuth {
                api_key,
                allow_missing_api_key,
                http_client,
            },
        }
    }
}

#[async_trait]
impl ClassifierModelApi for OpenAiDecisionsClassifierApi {
    fn id(&self) -> &str {
        KnownApi::OpenaiDecisions.as_str()
    }

    async fn classify(
        &self,
        model: Model,
        context: ClassifierContext,
        mut options: ClassifierOptions,
    ) -> ClassifierResult {
        self.auth.apply(&mut options);
        let mut result = ClassifierResult::empty_for(&model);
        let outcome = run(&self.auth, &model, &context, &options, &mut result).await;
        finish(result, outcome, LABEL, &options, error_message)
    }
}

async fn run(
    auth: &ClassifierAuth,
    model: &Model,
    context: &ClassifierContext,
    options: &ClassifierOptions,
    result: &mut ClassifierResult,
) -> ClassifierOutcome<()> {
    if model.api != KnownApi::OpenaiDecisions.as_str() {
        return Err(ClassifierError::message(format!(
            "Unsupported classifier API: {}",
            model.api
        )));
    }
    let payload = json!({
        "model": model.id,
        "input": wire_input(context)?,
        "questions": context
            .questions
            .iter()
            .map(|(id, question)| wire_question(id, question))
            .collect::<Vec<_>>(),
    });
    let url = format!("{}/decisions", model.base_url.trim_end_matches('/'));
    let body = post_classifier_request(
        LABEL,
        &url,
        model,
        payload,
        auth,
        options,
        NO_RETRY_STATUSES,
    )
    .await?;
    let body = as_record(&body).ok_or_else(unexpected)?;
    // Set before the answers: a request with malformed or refused answers
    // was still billed.
    result.usage = parse_classifier_usage(body.get("usage"), model);
    result.answers = parse_answers(body.get("answers"), context)?;
    Ok(())
}

fn error_message(error: &ClassifierError, label: &str) -> String {
    match error {
        ClassifierError::Status { status: 504, .. } => format!(
            "{label} error (504): the request timed out at the gateway. Very large inputs \
             (above roughly 600K tokens) currently exceed its time limit."
        ),
        error => error.format(&format!("{label} error")),
    }
}

fn predicate_instructions(instructions: &str, when_true: &str, when_false: &str) -> String {
    let meanings = [
        (!when_true.is_empty()).then(|| format!("True means: {when_true}")),
        (!when_false.is_empty()).then(|| format!("False means: {when_false}")),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    if meanings.is_empty() {
        instructions.to_string()
    } else {
        format!("{instructions}\n\n{}", meanings.join("\n"))
    }
}

fn wire_question(name: &str, question: &ClassifierQuestion) -> Value {
    match question {
        ClassifierQuestion::Choice {
            instructions,
            criteria,
        } => json!({
            "type": "choice",
            "name": name,
            "instructions": instructions,
            "choices": criteria
                .iter()
                .map(|(value, description)| {
                    if description.is_empty() {
                        json!({ "value": value })
                    } else {
                        json!({ "value": value, "description": description })
                    }
                })
                .collect::<Vec<_>>(),
        }),
        ClassifierQuestion::Score {
            instructions,
            criteria,
        } => json!({
            "type": "score",
            "name": name,
            "instructions": instructions,
            "levels": criteria.iter().map(|label| json!({ "label": label })).collect::<Vec<_>>(),
        }),
        ClassifierQuestion::Bool {
            instructions,
            criteria,
        } => json!({
            "type": "predicate",
            "name": name,
            "instructions": predicate_instructions(
                instructions,
                &criteria.when_true,
                &criteria.when_false,
            ),
        }),
    }
}

fn wire_input(context: &ClassifierContext) -> ClassifierOutcome<Value> {
    let state = Value::Object(context.state.clone()).to_string();
    if context.images.is_empty() {
        return Ok(Value::String(state));
    }
    if context.images.len() > MAX_IMAGES {
        return Err(ClassifierError::message(format!(
            "{LABEL} accepts at most {MAX_IMAGES} images, got {}",
            context.images.len()
        )));
    }
    let mut content = vec![json!({ "type": "input_text", "text": state })];
    content.extend(context.images.iter().map(|image| {
        json!({
            "type": "input_image",
            "image_url": format!("data:{};base64,{}", image.mime_type, image.data),
        })
    }));
    Ok(json!([{ "role": "user", "content": content }]))
}

fn choice_probabilities(
    value: Option<&Value>,
    id: &str,
) -> ClassifierOutcome<IndexMap<String, f64>> {
    let invalid =
        || ClassifierError::message(format!("{LABEL} returned invalid probabilities for {id}"));
    let entries = value.and_then(Value::as_array).ok_or_else(invalid)?;
    entries
        .iter()
        .map(|entry| {
            let entry = as_record(entry).ok_or_else(invalid)?;
            let value = entry
                .get("value")
                .and_then(Value::as_str)
                .ok_or_else(invalid)?;
            let probability = required_number(
                LABEL,
                entry.get("probability"),
                &format!("probability for {id}.{value}"),
            )?;
            Ok((value.to_string(), probability))
        })
        .collect()
}

fn parse_answer(
    id: &str,
    question: &ClassifierQuestion,
    answer: &Map<String, Value>,
) -> ClassifierOutcome<ClassifierAnswer> {
    let kind = answer.get("type").and_then(Value::as_str);
    if kind == Some("refusal") {
        return Err(ClassifierError::message(format!(
            "{LABEL} refused to answer {id}"
        )));
    }
    let wrong = |expected: &str| {
        ClassifierError::message(format!(
            "{LABEL} did not return a {expected} answer for {id}"
        ))
    };
    match question {
        ClassifierQuestion::Choice { .. } => {
            let choice = answer.get("choice").and_then(Value::as_str);
            let (Some("choice"), Some(choice)) = (kind, choice) else {
                return Err(wrong("choice"));
            };
            Ok(ClassifierAnswer::Choice {
                choice: choice.to_string(),
                probabilities: choice_probabilities(answer.get("probabilities"), id)?,
                confidence: required_number(
                    LABEL,
                    answer.get("confidence"),
                    &format!("confidence for {id}"),
                )?,
            })
        }
        ClassifierQuestion::Score { .. } => {
            if kind != Some("score") {
                return Err(wrong("score"));
            }
            Ok(ClassifierAnswer::Score {
                score: required_number(LABEL, answer.get("score"), &format!("score for {id}"))?,
                confidence: required_number(
                    LABEL,
                    answer.get("confidence"),
                    &format!("confidence for {id}"),
                )?,
            })
        }
        ClassifierQuestion::Bool { .. } => {
            if kind != Some("predicate") {
                return Err(wrong("predicate"));
            }
            Ok(ClassifierAnswer::Bool {
                probability: required_number(
                    LABEL,
                    answer.get("probability"),
                    &format!("probability for {id}"),
                )?,
            })
        }
    }
}

fn parse_answers(
    value: Option<&Value>,
    context: &ClassifierContext,
) -> ClassifierOutcome<IndexMap<String, ClassifierAnswer>> {
    let answers = value.and_then(Value::as_array).ok_or_else(unexpected)?;
    let mut by_name = std::collections::HashMap::new();
    for answer in answers {
        if let Some(answer) = as_record(answer)
            && let Some(name) = answer.get("name").and_then(Value::as_str)
        {
            by_name.insert(name, answer);
        }
    }
    context
        .questions
        .iter()
        .map(|(id, question)| {
            let answer = by_name.get(id.as_str()).ok_or_else(|| {
                ClassifierError::message(format!("{LABEL} did not return an answer for {id}"))
            })?;
            Ok((id.clone(), parse_answer(id, question, answer)?))
        })
        .collect()
}

fn unexpected() -> ClassifierError {
    ClassifierError::message(format!("{LABEL} returned an unexpected response"))
}
