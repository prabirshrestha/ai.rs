//! Classification: typed answers to questions about a JSON state and
//! optional images, from a classifier model. This ports Pi's `classify()`
//! (`@earendil-works/pi-ai`): the same context, questions, answers, and
//! result, with the same error rules.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio_util::sync::CancellationToken;

use crate::types::{
    Api, ImageContent, Model, ModelInput, ProviderHeaders, ProviderId, RequestOptions, Usage,
};

/// A JSON object: the state a classifier judges.
pub type JsonObject = Map<String, Value>;

/// What a classifier judges, and the questions it answers. Questions keep
/// their order, and their keys name the answers.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ClassifierContext {
    pub state: JsonObject,
    /// Images judged together with `state`. Only models whose `input`
    /// includes [`ModelInput::Image`] accept them; other models return an
    /// error result.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<ImageContent>,
    pub questions: IndexMap<String, ClassifierQuestion>,
}

impl ClassifierContext {
    pub fn new(state: JsonObject) -> Self {
        Self {
            state,
            ..Self::default()
        }
    }

    /// Builds a context from any value that serializes to a JSON object.
    pub fn from_state(state: impl Serialize) -> crate::Result<Self> {
        match serde_json::to_value(state)? {
            Value::Object(state) => Ok(Self::new(state)),
            _ => Err(crate::Error::Validation(
                "classifier state must be a JSON object".to_string(),
            )),
        }
    }

    pub fn image(mut self, image: ImageContent) -> Self {
        self.images.push(image);
        self
    }

    pub fn question(mut self, id: impl Into<String>, question: ClassifierQuestion) -> Self {
        self.questions.insert(id.into(), question);
        self
    }
}

/// One question. `choice` picks one key of `criteria`, `score` picks a level
/// of `criteria` (from 0), and `bool` gives the probability of true.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ClassifierQuestion {
    Choice {
        instructions: String,
        /// Each choice and what it means. An empty meaning is allowed.
        criteria: IndexMap<String, String>,
    },
    Score {
        instructions: String,
        /// The levels, lowest first.
        criteria: Vec<String>,
    },
    Bool {
        instructions: String,
        criteria: ClassifierBoolCriteria,
    },
}

impl ClassifierQuestion {
    pub fn choice<K, V>(
        instructions: impl Into<String>,
        criteria: impl IntoIterator<Item = (K, V)>,
    ) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        Self::Choice {
            instructions: instructions.into(),
            criteria: criteria
                .into_iter()
                .map(|(value, meaning)| (value.into(), meaning.into()))
                .collect(),
        }
    }

    pub fn score<L: Into<String>>(
        instructions: impl Into<String>,
        levels: impl IntoIterator<Item = L>,
    ) -> Self {
        Self::Score {
            instructions: instructions.into(),
            criteria: levels.into_iter().map(Into::into).collect(),
        }
    }

    pub fn bool(
        instructions: impl Into<String>,
        when_true: impl Into<String>,
        when_false: impl Into<String>,
    ) -> Self {
        Self::Bool {
            instructions: instructions.into(),
            criteria: ClassifierBoolCriteria {
                when_true: when_true.into(),
                when_false: when_false.into(),
            },
        }
    }

    pub fn instructions(&self) -> &str {
        match self {
            Self::Choice { instructions, .. }
            | Self::Score { instructions, .. }
            | Self::Bool { instructions, .. } => instructions,
        }
    }
}

/// What true and false mean for a `bool` question.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassifierBoolCriteria {
    #[serde(rename = "true")]
    pub when_true: String,
    #[serde(rename = "false")]
    pub when_false: String,
}

/// One answer, of the same type as its question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ClassifierAnswer {
    Choice {
        choice: String,
        probabilities: IndexMap<String, f64>,
        confidence: f64,
    },
    Score {
        score: f64,
        confidence: f64,
    },
    Bool {
        probability: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ClassifierStopReason {
    Stop,
    Error,
    Aborted,
}

/// The result of [`classify`]. A failure is a result with `stop_reason`
/// `error` (or `aborted`) and an `error_message`, never an `Err`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassifierResult {
    pub api: Api,
    pub provider: ProviderId,
    pub model: String,
    /// Answers keyed by question ID. Empty when the request failed.
    pub answers: IndexMap<String, ClassifierAnswer>,
    /// Token usage and its cost at the model's price, when the service
    /// reports token counts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    pub stop_reason: ClassifierStopReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Unix time in milliseconds.
    pub timestamp: u64,
}

impl ClassifierResult {
    /// An empty, successful result for `model`.
    pub fn empty_for(model: &Model) -> Self {
        Self {
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            answers: IndexMap::new(),
            usage: None,
            stop_reason: ClassifierStopReason::Stop,
            error_message: None,
            timestamp: crate::utils::time::now_millis(),
        }
    }

    /// A failed result for `model`.
    pub fn error_for(model: &Model, message: impl Into<String>, aborted: bool) -> Self {
        Self {
            stop_reason: if aborted {
                ClassifierStopReason::Aborted
            } else {
                ClassifierStopReason::Error
            },
            error_message: Some(message.into()),
            ..Self::empty_for(model)
        }
    }

    pub fn is_error(&self) -> bool {
        self.stop_reason != ClassifierStopReason::Stop
    }

    pub fn answer(&self, id: &str) -> Option<&ClassifierAnswer> {
        self.answers.get(id)
    }

    /// The chosen value of a `choice` answer.
    pub fn choice(&self, id: &str) -> Option<&str> {
        match self.answers.get(id)? {
            ClassifierAnswer::Choice { choice, .. } => Some(choice),
            _ => None,
        }
    }

    /// The score of a `score` answer.
    pub fn score(&self, id: &str) -> Option<f64> {
        match self.answers.get(id)? {
            ClassifierAnswer::Score { score, .. } => Some(*score),
            _ => None,
        }
    }

    /// The probability of true of a `bool` answer.
    pub fn probability(&self, id: &str) -> Option<f64> {
        match self.answers.get(id)? {
            ClassifierAnswer::Bool { probability } => Some(*probability),
            _ => None,
        }
    }
}

#[derive(Clone, Default)]
pub struct ClassifierOptions {
    pub base: RequestOptions,
    /// Divides the label log-probabilities of `llama-cpp-classify` before the
    /// softmax (default 1; above 1 softens them). System One and OpenAI
    /// Decisions ignore it.
    pub temperature: Option<f64>,
}

impl ClassifierOptions {
    pub fn builder() -> ClassifierOptionsBuilder {
        ClassifierOptionsBuilder::default()
    }

    pub(crate) fn is_aborted(&self) -> bool {
        self.base
            .cancellation_token
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
    }
}

/// Builds [`ClassifierOptions`].
#[derive(Clone, Default)]
pub struct ClassifierOptionsBuilder {
    options: ClassifierOptions,
}

impl ClassifierOptionsBuilder {
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.options.base.api_key = Some(api_key.into());
        self
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.options.temperature = Some(temperature);
        self
    }

    pub fn timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.options.base.timeout_ms = Some(timeout_ms);
        self
    }

    /// Retries after a retryable failure. Classifier requests default to 2.
    pub fn max_retries(mut self, max_retries: u32) -> Self {
        self.options.base.max_retries = Some(max_retries);
        self
    }

    pub fn max_retry_delay_ms(mut self, max_retry_delay_ms: u64) -> Self {
        self.options.base.max_retry_delay_ms = Some(max_retry_delay_ms);
        self
    }

    pub fn headers(mut self, headers: ProviderHeaders) -> Self {
        self.options.base.headers = headers;
        self
    }

    pub fn cancellation_token(mut self, token: CancellationToken) -> Self {
        self.options.base.cancellation_token = Some(token);
        self
    }

    pub fn http_client(mut self, http_client: reqwest::Client) -> Self {
        self.options.base.http_client = Some(http_client);
        self
    }

    pub fn build(self) -> ClassifierOptions {
        self.options
    }
}

/// Answers the context's questions with a classifier model. Every failure
/// (a model that is not a classifier, images for a text-only model, an HTTP
/// error, a refusal, a malformed answer) resolves to a result with
/// `stop_reason` `error`, or `aborted` after a cancellation.
pub async fn classify(
    model: Model,
    context: ClassifierContext,
    options: Option<ClassifierOptions>,
) -> ClassifierResult {
    let options = options.unwrap_or_default();
    let Some(api) = model.classifier_api() else {
        let message = format!(
            "Model {}/{} is not a classifier model",
            model.provider, model.id
        );
        return ClassifierResult::error_for(&model, message, options.is_aborted());
    };
    if !context.images.is_empty() && !model.input.contains(&ModelInput::Image) {
        let message = format!(
            "Model {}/{} does not accept image input",
            model.provider, model.id
        );
        return ClassifierResult::error_for(&model, message, options.is_aborted());
    }
    api.classify(model, context, options).await
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::providers::openai;

    #[test]
    fn questions_and_answers_use_pis_json() {
        let context = ClassifierContext::from_state(json!({"text": "hi"}))
            .expect("state")
            .question(
                "color",
                ClassifierQuestion::choice("Pick", [("red", "warm"), ("blue", "")]),
            )
            .question("level", ClassifierQuestion::score("Rate", ["low", "high"]))
            .question("ok", ClassifierQuestion::bool("Is it fine?", "yes", "no"));
        assert_eq!(
            serde_json::to_value(&context).expect("json"),
            json!({
                "state": {"text": "hi"},
                "questions": {
                    "color": {"type": "choice", "instructions": "Pick", "criteria": {"red": "warm", "blue": ""}},
                    "level": {"type": "score", "instructions": "Rate", "criteria": ["low", "high"]},
                    "ok": {"type": "bool", "instructions": "Is it fine?", "criteria": {"true": "yes", "false": "no"}}
                }
            })
        );
        let answer: ClassifierAnswer =
            serde_json::from_value(json!({"type": "bool", "probability": 0.25})).expect("answer");
        assert_eq!(answer, ClassifierAnswer::Bool { probability: 0.25 });
    }

    #[tokio::test]
    async fn a_model_that_is_not_a_classifier_gives_an_error_result() {
        let model = openai::builder()
            .api_key(Some("key"))
            .build()
            .expect("provider")
            .model("gpt-5")
            .build()
            .expect("model");
        let result = classify(model, ClassifierContext::default(), None).await;
        assert_eq!(result.stop_reason, ClassifierStopReason::Error);
        assert_eq!(
            result.error_message.as_deref(),
            Some("Model openai/gpt-5 is not a classifier model")
        );
        assert!(result.answers.is_empty());
    }

    #[tokio::test]
    async fn images_for_a_text_only_classifier_give_an_error_result() {
        let model = openai::builder()
            .api_key(Some("key"))
            .build()
            .expect("provider")
            .classifier_model("text-only")
            .input(vec![ModelInput::Text])
            .build_classifier()
            .expect("model");
        let context = ClassifierContext::default().image(ImageContent {
            data: "AA==".to_string(),
            mime_type: "image/png".to_string(),
        });
        let result = classify(model, context, None).await;
        assert_eq!(
            result.error_message.as_deref(),
            Some("Model openai/text-only does not accept image input")
        );
        assert_eq!(result.api, "openai-decisions");
    }
}
