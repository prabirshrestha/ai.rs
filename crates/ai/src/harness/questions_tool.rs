//! The ask-user-questions builtin tool.

use serde::{Deserialize, Serialize};
use serde_json::json;

use serde_json::Value as JsonValue;

use crate::Tool;
use crate::harness::error::HarnessError;

fn parse_json<T: serde::de::DeserializeOwned>(value: JsonValue) -> Result<T, HarnessError> {
    serde_json::from_value(value).map_err(|error| HarnessError::new(error.to_string()))
}

/// One option of a question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuestionOption {
    pub id: String,
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preview: Option<String>,
}

/// `allowFreeForm: boolean | { secret: boolean }`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AllowFreeForm {
    Enabled(bool),
    Secret { secret: bool },
}

/// One question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Question {
    pub id: String,
    pub question: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub header: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<Vec<QuestionOption>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_multiple: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_free_form: Option<AllowFreeForm>,
}

/// `QuestionsToolInput`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuestionsToolInput {
    pub allow_partial_answers: bool,
    pub questions: Vec<Question>,
}

/// One answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuestionAnswer {
    pub option_ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub freeform: Option<String>,
}

/// `QuestionsToolOutput`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "kebab-case")]
pub enum QuestionsToolOutput {
    Answered {
        answers: indexmap::IndexMap<String, QuestionAnswer>,
    },
    PartiallyAnswered {
        answers: indexmap::IndexMap<String, QuestionAnswer>,
    },
    Declined,
    Cancelled,
}

fn non_empty(_value: &JsonValue, field: &str, s: &str) -> Result<(), HarnessError> {
    if s.is_empty() {
        Err(HarnessError::new(format!(
            "{field}: Too small: expected string to have >=1 characters"
        )))
    } else {
        Ok(())
    }
}

impl QuestionsToolInput {
    /// `harnessV1QuestionsToolInputSchema.parse(value)`.
    pub fn parse(value: JsonValue) -> Result<Self, HarnessError> {
        let input: Self = parse_json(value.clone())?;
        if input.questions.is_empty() {
            return Err(HarnessError::new(
                "questions: Too small: expected array to have >=1 items",
            ));
        }
        for q in &input.questions {
            non_empty(&value, "id", &q.id)?;
            non_empty(&value, "question", &q.question)?;
            for o in q.options.iter().flatten() {
                non_empty(&value, "options.id", &o.id)?;
                non_empty(&value, "options.label", &o.label)?;
            }
        }
        Ok(input)
    }
}

impl QuestionsToolOutput {
    /// `harnessV1QuestionsToolOutputSchema.parse(value)`.
    pub fn parse(value: JsonValue) -> Result<Self, HarnessError> {
        parse_json(value)
    }
}

/// JSON Schema of `harnessV1QuestionsToolInputSchema`.
pub fn questions_tool_input_schema() -> JsonValue {
    let non_empty = json!({ "type": "string", "minLength": 1 });
    json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "allowPartialAnswers": { "type": "boolean" },
            "questions": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": non_empty,
                        "question": non_empty,
                        "header": { "type": "string" },
                        "options": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": non_empty,
                                    "label": non_empty,
                                    "description": { "type": "string" },
                                    "preview": { "type": "string" }
                                },
                                "required": ["id", "label"],
                                "additionalProperties": false
                            }
                        },
                        "allowMultiple": { "type": "boolean" },
                        "allowFreeForm": {
                            "anyOf": [
                                { "type": "boolean" },
                                {
                                    "type": "object",
                                    "properties": { "secret": { "type": "boolean" } },
                                    "required": ["secret"],
                                    "additionalProperties": false
                                }
                            ]
                        }
                    },
                    "required": ["id", "question"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["allowPartialAnswers", "questions"],
        "additionalProperties": false
    })
}

/// JSON Schema of `harnessV1QuestionsToolOutputSchema`.
pub fn questions_tool_output_schema() -> JsonValue {
    let answers = json!({
        "type": "object",
        "additionalProperties": {
            "type": "object",
            "properties": {
                "optionIds": { "type": "array", "items": { "type": "string" } },
                "freeform": { "type": "string" }
            },
            "required": ["optionIds"],
            "additionalProperties": false
        }
    });
    let action = |name: &str, with_answers: bool| {
        let mut properties = serde_json::Map::new();
        properties.insert("action".into(), json!({ "type": "string", "const": name }));
        let mut required = vec![json!("action")];
        if with_answers {
            properties.insert("answers".into(), answers.clone());
            required.push(json!("answers"));
        }
        json!({
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": false
        })
    };
    json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "oneOf": [
            action("answered", true),
            action("partially-answered", true),
            action("declined", false),
            action("cancelled", false)
        ]
    })
}

/// `harnessV1QuestionsTool`: the ask-user-questions builtin, as an ai.rs [`Tool`].
pub fn questions_tool() -> Tool {
    Tool {
        name: "askUserQuestions".into(),
        description: "Ask the user one or more questions".into(),
        parameters: questions_tool_input_schema(),
        constrained_sampling: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // harness-v1-questions-tool.test.ts
    #[test]
    fn accepts_the_canonical_question_input() {
        let input = json!({
            "allowPartialAnswers": true,
            "questions": [{
                "id": "framework",
                "question": "Which framework?",
                "header": "Framework",
                "options": [{
                    "id": "react",
                    "label": "React",
                    "description": "Use React.",
                    "preview": "<App />"
                }],
                "allowMultiple": false,
                "allowFreeForm": { "secret": true }
            }]
        });
        let parsed = QuestionsToolInput::parse(input.clone()).unwrap();
        assert_eq!(serde_json::to_value(parsed).unwrap(), input);
    }

    #[test]
    fn requires_the_partial_answer_capability() {
        assert!(
            QuestionsToolInput::parse(json!({
                "questions": [{ "id": "framework", "question": "Which framework?" }]
            }))
            .is_err()
        );
    }

    #[test]
    fn rejects_empty_questions_and_ids() {
        assert!(
            QuestionsToolInput::parse(json!({
                "allowPartialAnswers": false, "questions": []
            }))
            .is_err()
        );
        assert!(
            QuestionsToolInput::parse(json!({
                "allowPartialAnswers": false, "questions": [{ "id": "", "question": "q" }]
            }))
            .is_err()
        );
    }

    #[test]
    fn accepts_each_output_action() {
        for output in [
            json!({ "action": "answered", "answers": { "framework": { "optionIds": ["react"], "freeform": "notes" } } }),
            json!({ "action": "partially-answered", "answers": { "framework": { "optionIds": [] } } }),
            json!({ "action": "declined" }),
            json!({ "action": "cancelled" }),
        ] {
            let parsed = QuestionsToolOutput::parse(output.clone()).unwrap();
            assert_eq!(serde_json::to_value(parsed).unwrap(), output);
        }
    }

    #[test]
    fn tool_carries_the_input_schema() {
        let tool = questions_tool();
        assert_eq!(tool.name, "askUserQuestions");
        assert_eq!(tool.description, "Ask the user one or more questions");
        assert_eq!(tool.parameters, questions_tool_input_schema());
    }
}
