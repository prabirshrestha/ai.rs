//! TypeSafe's System One protocol: `POST {base_url}/systemone` with `{ model,
//! state, questions }`. Ports Pi's `typesafe-system-one.ts` and
//! `system-one-shared.ts`. OpenRouter, GitHub Copilot, and llama.cpp's
//! `llama-server` serve the same protocol.
//!
//! `bool` questions go as System One's `noul`. Images go in one of three
//! forms (see [`SystemOneImageInput`]), set by `compat.imageInput` or
//! detected from the provider and base URL.

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
use crate::types::{ImageContent, KnownApi, Model, ModelInput, SystemOneImageInput};

const LABEL: &str = "System One API";

/// TypeSafe's opt-in header for its experimental image input.
const IMAGE_OPT_IN_HEADER: (&str, &str) = ("TypeSafe-Danger-Zone-Images", "true");

/// Clef takes at most this many images per request.
const MAX_DATA_URL_IMAGES: usize = 4;

#[derive(Clone)]
pub(crate) struct TypesafeSystemOneClassifierApi {
    auth: ClassifierAuth,
    /// Joined to the model's base URL: `/systemone` after a `.../v1` base,
    /// `/v1/systemone` after a bare host such as Copilot's.
    path: &'static str,
}

impl TypesafeSystemOneClassifierApi {
    pub(crate) fn new(
        api_key: Option<String>,
        allow_missing_api_key: bool,
        http_client: Option<reqwest::Client>,
        path: &'static str,
    ) -> Self {
        Self {
            auth: ClassifierAuth {
                api_key,
                allow_missing_api_key,
                http_client,
            },
            path,
        }
    }
}

#[async_trait]
impl ClassifierModelApi for TypesafeSystemOneClassifierApi {
    fn id(&self) -> &str {
        KnownApi::TypesafeSystemOne.as_str()
    }

    async fn classify(
        &self,
        model: Model,
        context: ClassifierContext,
        mut options: ClassifierOptions,
    ) -> ClassifierResult {
        self.auth.apply(&mut options);
        let mut result = ClassifierResult::empty_for(&model);
        let outcome = self.run(&model, &context, &options, &mut result).await;
        finish(result, outcome, LABEL, &options, |error, label| {
            error.format(&format!("{label} error"))
        })
    }
}

impl TypesafeSystemOneClassifierApi {
    async fn run(
        &self,
        model: &Model,
        context: &ClassifierContext,
        options: &ClassifierOptions,
        result: &mut ClassifierResult,
    ) -> ClassifierOutcome<()> {
        if model.api != KnownApi::TypesafeSystemOne.as_str() {
            return Err(ClassifierError::message(format!(
                "Unsupported classifier API: {}",
                model.api
            )));
        }
        let mut request = wire_request(context);
        let mut options = options.clone();
        if !context.images.is_empty() {
            if !model.input.contains(&ModelInput::Image) {
                return Err(ClassifierError::message(format!(
                    "{LABEL} does not support image input for {}",
                    model.id
                )));
            }
            let (detected_input, detected_header) = detect_compat(model);
            let compat = &model.compat.system_one;
            let image_input = compat.image_input.unwrap_or(detected_input);
            add_images(model, image_input, &context.images, &mut request)?;
            if compat.send_image_opt_in_header.unwrap_or(detected_header) {
                let (name, value) = IMAGE_OPT_IN_HEADER;
                if !options
                    .base
                    .headers
                    .iter()
                    .any(|(key, _)| key.eq_ignore_ascii_case(name))
                {
                    options
                        .base
                        .headers
                        .insert(name.to_string(), Some(value.to_string()));
                }
            }
        }
        let mut payload = Map::new();
        payload.insert("model".to_string(), json!(model.id));
        payload.extend(request);
        let url = format!("{}{}", model.base_url.trim_end_matches('/'), self.path);
        let body = post_classifier_request(
            LABEL,
            &url,
            model,
            Value::Object(payload),
            &self.auth,
            &options,
            &[],
        )
        .await?;
        let body = as_record(&body).ok_or_else(unexpected)?;
        // Set before the answers: a request with malformed answers was
        // still billed.
        result.usage = parse_classifier_usage(body.get("usage"), model);
        result.answers = parse_answers(body.get("answers"), context)?;
        Ok(())
    }
}

/// OpenRouter, by provider ID (which survives a base URL override) or by
/// host, takes images as `image_url` parts and needs no opt-in header.
/// Every other service gets TypeSafe's `assets` and the header.
fn detect_compat(model: &Model) -> (SystemOneImageInput, bool) {
    if model.provider == "openrouter" || model.base_url.contains("openrouter.ai") {
        (SystemOneImageInput::StateImageUrls, false)
    } else {
        (SystemOneImageInput::Assets, true)
    }
}

/// The System One request without the model: the state and the questions,
/// with `bool` sent as `noul`. An empty meaning is left out (a choice's
/// becomes `null`), as OpenAI Decisions leaves out an empty description:
/// Copilot's System One rejects empty strings (checked live on 2026-11-18).
/// Pi sends them as they are.
fn wire_request(context: &ClassifierContext) -> Map<String, Value> {
    let questions = context
        .questions
        .iter()
        .map(|(id, question)| {
            let mut wire = serde_json::to_value(question).expect("a question serializes");
            match question {
                ClassifierQuestion::Choice { criteria, .. } => {
                    wire["criteria"] = criteria
                        .iter()
                        .map(|(value, meaning)| {
                            let meaning = (!meaning.is_empty()).then(|| json!(meaning));
                            (value.clone(), meaning.unwrap_or(Value::Null))
                        })
                        .collect::<Map<_, _>>()
                        .into();
                }
                ClassifierQuestion::Bool { criteria, .. } => {
                    wire["type"] = json!("noul");
                    let meanings = [
                        ("true", &criteria.when_true),
                        ("false", &criteria.when_false),
                    ]
                    .into_iter()
                    .filter(|(_, meaning)| !meaning.is_empty())
                    .map(|(key, meaning)| (key.to_string(), json!(meaning)))
                    .collect::<Map<_, _>>();
                    let wire = wire.as_object_mut().expect("a question is an object");
                    if meanings.is_empty() {
                        wire.remove("criteria");
                    } else {
                        wire.insert("criteria".to_string(), Value::Object(meanings));
                    }
                }
                ClassifierQuestion::Score { .. } => {}
            }
            (id.clone(), wire)
        })
        .collect::<Map<_, _>>();
    let mut request = Map::new();
    request.insert("state".to_string(), Value::Object(context.state.clone()));
    request.insert("questions".to_string(), Value::Object(questions));
    request
}

/// Adds the context's images to the request in the service's form.
fn add_images(
    model: &Model,
    input: SystemOneImageInput,
    images: &[ImageContent],
    request: &mut Map<String, Value>,
) -> ClassifierOutcome<()> {
    let state = request.remove("state").unwrap_or(Value::Null);
    match input {
        SystemOneImageInput::Images => {
            if images.len() > MAX_DATA_URL_IMAGES {
                return Err(ClassifierError::message(format!(
                    "{LABEL} accepts at most {MAX_DATA_URL_IMAGES} images for {}, got {}",
                    model.id,
                    images.len()
                )));
            }
            let urls = images
                .iter()
                .map(|image| {
                    if !matches!(
                        image.mime_type.as_str(),
                        "image/png" | "image/jpeg" | "image/webp"
                    ) {
                        return Err(ClassifierError::message(format!(
                            "{LABEL} accepts PNG, JPEG, and WebP images for {}, got {}",
                            model.id, image.mime_type
                        )));
                    }
                    Ok(json!(data_url(image)))
                })
                .collect::<ClassifierOutcome<Vec<_>>>()?;
            request.insert("state".to_string(), state);
            request.insert("images".to_string(), Value::Array(urls));
        }
        SystemOneImageInput::StateImageUrls => {
            let mut parts = images
                .iter()
                .map(
                    |image| json!({ "type": "image_url", "image_url": { "url": data_url(image) } }),
                )
                .collect::<Vec<_>>();
            parts.push(state);
            request.insert("state".to_string(), Value::Array(parts));
        }
        SystemOneImageInput::Assets => {
            let mut assets = Map::new();
            for (index, image) in images.iter().enumerate() {
                let format = match image.mime_type.as_str() {
                    "image/png" => "png",
                    "image/jpeg" => "jpeg",
                    other => {
                        return Err(ClassifierError::message(format!(
                            "{LABEL} accepts PNG and JPEG images for {}, got {other}",
                            model.id
                        )));
                    }
                };
                assets.insert(
                    format!("<image-{}>", index + 1),
                    json!({ "type": "image", "format": format, "data_base64": image.data }),
                );
            }
            // Every asset must be named in the request. The names go before
            // the state, where Clef also places its images.
            let mut names = (1..=images.len())
                .map(|index| Value::String(format!("<image-{index}>")))
                .collect::<Vec<_>>();
            names.push(state);
            request.insert("state".to_string(), Value::Array(names));
            request.insert("assets".to_string(), Value::Object(assets));
        }
    }
    Ok(())
}

fn data_url(image: &ImageContent) -> String {
    format!("data:{};base64,{}", image.mime_type, image.data)
}

fn probabilities(value: Option<&Value>, id: &str) -> ClassifierOutcome<IndexMap<String, f64>> {
    let map = value.and_then(as_record).ok_or_else(|| {
        ClassifierError::message(format!("{LABEL} returned invalid probabilities for {id}"))
    })?;
    map.iter()
        .map(|(key, probability)| {
            Ok((
                key.clone(),
                required_number(
                    LABEL,
                    Some(probability),
                    &format!("probability for {id}.{key}"),
                )?,
            ))
        })
        .collect()
}

fn parse_answers(
    value: Option<&Value>,
    context: &ClassifierContext,
) -> ClassifierOutcome<IndexMap<String, ClassifierAnswer>> {
    let answers = value.and_then(as_record).ok_or_else(unexpected)?;
    context
        .questions
        .iter()
        .map(|(id, question)| {
            let answer = answers.get(id).and_then(as_record).ok_or_else(|| {
                ClassifierError::message(format!("{LABEL} did not return an answer for {id}"))
            })?;
            Ok((id.clone(), parse_answer(id, question, answer)?))
        })
        .collect()
}

fn parse_answer(
    id: &str,
    question: &ClassifierQuestion,
    answer: &Map<String, Value>,
) -> ClassifierOutcome<ClassifierAnswer> {
    let kind = answer.get("type").and_then(Value::as_str);
    let wrong = |expected: &str| {
        ClassifierError::message(format!(
            "{LABEL} did not return a {expected} answer for {id}"
        ))
    };
    let confidence = || {
        required_number(
            LABEL,
            answer.get("confidence"),
            &format!("confidence for {id}"),
        )
    };
    match question {
        ClassifierQuestion::Choice { .. } => {
            let choice = answer.get("choice").and_then(Value::as_str);
            let (Some("choice"), Some(choice)) = (kind, choice) else {
                return Err(wrong("choice"));
            };
            Ok(ClassifierAnswer::Choice {
                choice: choice.to_string(),
                probabilities: probabilities(answer.get("probabilities"), id)?,
                confidence: confidence()?,
            })
        }
        ClassifierQuestion::Score { .. } => {
            if kind != Some("score") {
                return Err(wrong("score"));
            }
            Ok(ClassifierAnswer::Score {
                score: required_number(LABEL, answer.get("score"), &format!("score for {id}"))?,
                confidence: confidence()?,
            })
        }
        ClassifierQuestion::Bool { .. } => {
            if kind != Some("noul") {
                return Err(wrong("bool"));
            }
            Ok(ClassifierAnswer::Bool {
                probability: required_number(
                    LABEL,
                    answer.get("noul"),
                    &format!("probability for {id}"),
                )?,
            })
        }
    }
}

fn unexpected() -> ClassifierError {
    ClassifierError::message(format!("{LABEL} returned an unexpected response"))
}
