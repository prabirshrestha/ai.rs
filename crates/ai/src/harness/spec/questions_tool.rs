//! Port of `harness-v1-questions-tool.ts`.

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::harness::provider::{JsonValue, TypeValidationError, parse_json};
use crate::harness::sdk::tool::{Tool, json_schema};

/// One option of a question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1QuestionOption {
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
pub enum HarnessV1AllowFreeForm {
    Enabled(bool),
    Secret { secret: bool },
}

/// One question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1Question {
    pub id: String,
    pub question: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub header: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<Vec<HarnessV1QuestionOption>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_multiple: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_free_form: Option<HarnessV1AllowFreeForm>,
}

/// `HarnessV1QuestionsToolInput`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1QuestionsToolInput {
    pub allow_partial_answers: bool,
    pub questions: Vec<HarnessV1Question>,
}

/// One answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1QuestionAnswer {
    pub option_ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub freeform: Option<String>,
}

/// `HarnessV1QuestionsToolOutput`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "kebab-case")]
pub enum HarnessV1QuestionsToolOutput {
    Answered {
        answers: indexmap::IndexMap<String, HarnessV1QuestionAnswer>,
    },
    PartiallyAnswered {
        answers: indexmap::IndexMap<String, HarnessV1QuestionAnswer>,
    },
    Declined,
    Cancelled,
}

fn non_empty(value: &JsonValue, field: &str, s: &str) -> Result<(), TypeValidationError> {
    if s.is_empty() {
        Err(TypeValidationError::new(
            value.clone(),
            format!("{field}: Too small: expected string to have >=1 characters"),
        ))
    } else {
        Ok(())
    }
}

impl HarnessV1QuestionsToolInput {
    /// `harnessV1QuestionsToolInputSchema.parse(value)`.
    pub fn parse(value: JsonValue) -> Result<Self, TypeValidationError> {
        let input: Self = parse_json(value.clone())?;
        if input.questions.is_empty() {
            return Err(TypeValidationError::new(
                value,
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

impl HarnessV1QuestionsToolOutput {
    /// `harnessV1QuestionsToolOutputSchema.parse(value)`.
    pub fn parse(value: JsonValue) -> Result<Self, TypeValidationError> {
        parse_json(value)
    }
}

/// JSON Schema of `harnessV1QuestionsToolInputSchema`.
pub fn harness_v1_questions_tool_input_schema() -> JsonValue {
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
pub fn harness_v1_questions_tool_output_schema() -> JsonValue {
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

/// `harnessV1QuestionsTool`.
pub fn harness_v1_questions_tool() -> Tool {
    Tool::new(
        json_schema(harness_v1_questions_tool_input_schema()).with_validate(|v| {
            HarnessV1QuestionsToolInput::parse(v.clone())
                .map(|i| serde_json::to_value(i).expect("serializable"))
                .map_err(Into::into)
        }),
    )
    .with_description("Ask the user one or more questions")
    .with_output_schema(
        json_schema(harness_v1_questions_tool_output_schema()).with_validate(|v| {
            HarnessV1QuestionsToolOutput::parse(v.clone())
                .map(|o| serde_json::to_value(o).expect("serializable"))
                .map_err(Into::into)
        }),
    )
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
        let parsed = HarnessV1QuestionsToolInput::parse(input.clone()).unwrap();
        assert_eq!(serde_json::to_value(parsed).unwrap(), input);
    }

    #[test]
    fn requires_the_partial_answer_capability() {
        assert!(
            HarnessV1QuestionsToolInput::parse(json!({
                "questions": [{ "id": "framework", "question": "Which framework?" }]
            }))
            .is_err()
        );
    }

    #[test]
    fn rejects_empty_questions_and_ids() {
        assert!(
            HarnessV1QuestionsToolInput::parse(json!({
                "allowPartialAnswers": false, "questions": []
            }))
            .is_err()
        );
        assert!(
            HarnessV1QuestionsToolInput::parse(json!({
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
            let parsed = HarnessV1QuestionsToolOutput::parse(output.clone()).unwrap();
            assert_eq!(serde_json::to_value(parsed).unwrap(), output);
        }
    }

    #[test]
    fn tool_validates_through_its_schemas() {
        let tool = harness_v1_questions_tool();
        assert_eq!(
            tool.description.as_deref(),
            Some("Ask the user one or more questions")
        );
        assert!(
            tool.input_schema
                .validate_value(&json!({ "questions": [] }))
                .is_err()
        );
        assert!(
            tool.output_schema
                .unwrap()
                .validate_value(&json!({ "action": "declined" }))
                .is_ok()
        );
    }
}
