//! Claude Code's `AskUserQuestion` as the harness questions tool
//! (`question-tool.ts`).

use indexmap::IndexMap;
use serde_json::{Map, Value, json};

use crate::harness::questions_tool::QuestionsToolOutput;

fn questions(native_input: &Value) -> &[Value] {
    native_input
        .get("questions")
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}

/// `toHarnessQuestionsInput`.
pub fn to_harness_questions_input(native_input: &Value) -> Value {
    let questions: Vec<Value> = questions(native_input)
        .iter()
        .enumerate()
        .map(|(question_index, question)| {
            let options: Vec<Value> = question
                .get("options")
                .and_then(Value::as_array)
                .map_or(&[][..], Vec::as_slice)
                .iter()
                .enumerate()
                .map(|(option_index, option)| {
                    let mut entry = Map::new();
                    entry.insert("id".into(), json!(format!("option-{}", option_index + 1)));
                    entry.insert(
                        "label".into(),
                        option.get("label").cloned().unwrap_or(json!("")),
                    );
                    if let Some(description) = option.get("description") {
                        entry.insert("description".into(), description.clone());
                    }
                    if let Some(preview) = option.get("preview") {
                        entry.insert("preview".into(), preview.clone());
                    }
                    Value::Object(entry)
                })
                .collect();
            let mut entry = Map::new();
            entry.insert(
                "id".into(),
                json!(format!("question-{}", question_index + 1)),
            );
            entry.insert(
                "question".into(),
                question.get("question").cloned().unwrap_or(json!("")),
            );
            if let Some(header) = question.get("header") {
                entry.insert("header".into(), header.clone());
            }
            entry.insert("options".into(), Value::Array(options));
            if let Some(multi) = question.get("multiSelect") {
                entry.insert("allowMultiple".into(), multi.clone());
            }
            entry.insert("allowFreeForm".into(), json!(true));
            Value::Object(entry)
        })
        .collect();
    json!({ "allowPartialAnswers": true, "questions": questions })
}

/// The decision `toClaudeCodeQuestionResult` returns.
#[derive(Debug, Clone, PartialEq)]
pub enum ClaudeCodeQuestionResult {
    Allow { updated_input: Value },
    Deny { message: String },
}

fn positional_index(id: &str, prefix: &str) -> Option<usize> {
    let index: usize = id.strip_prefix(prefix)?.parse().ok()?;
    (index > 0).then(|| index - 1)
}

/// `toClaudeCodeQuestionResult`.
pub fn to_claude_code_question_result(
    native_input: &Value,
    output: &QuestionsToolOutput,
) -> ClaudeCodeQuestionResult {
    let answers = match output {
        QuestionsToolOutput::Declined => {
            return ClaudeCodeQuestionResult::Deny {
                message: "The user declined the questions.".into(),
            };
        }
        QuestionsToolOutput::Cancelled => {
            return ClaudeCodeQuestionResult::Deny {
                message: "The user cancelled the questions.".into(),
            };
        }
        QuestionsToolOutput::Answered { answers }
        | QuestionsToolOutput::PartiallyAnswered { answers } => answers,
    };
    let mut native_answers: IndexMap<String, Value> = IndexMap::new();
    for (question_index, question) in questions(native_input).iter().enumerate() {
        let Some(answer) = answers.get(&format!("question-{}", question_index + 1)) else {
            continue;
        };
        let text = match &answer.freeform {
            Some(freeform) => freeform.clone(),
            None => {
                let options = question
                    .get("options")
                    .and_then(Value::as_array)
                    .map_or(&[][..], Vec::as_slice);
                let labels: Vec<String> = answer
                    .option_ids
                    .iter()
                    .filter_map(|id| positional_index(id, "option-"))
                    .filter_map(|index| options.get(index))
                    .filter_map(|option| option.get("label").and_then(Value::as_str))
                    .map(str::to_string)
                    .collect();
                if question.get("multiSelect").and_then(Value::as_bool) == Some(true) {
                    labels.join(", ")
                } else {
                    labels.into_iter().next().unwrap_or_default()
                }
            }
        };
        let key = question
            .get("question")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        native_answers.insert(key, Value::String(text));
    }
    let mut updated = native_input.as_object().cloned().unwrap_or_default();
    updated.insert(
        "answers".into(),
        Value::Object(native_answers.into_iter().collect()),
    );
    ClaudeCodeQuestionResult::Allow {
        updated_input: Value::Object(updated),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::questions_tool::QuestionAnswer;

    fn native() -> Value {
        json!({
            "questions": [
                {
                    "question": "Pick a color",
                    "header": "Color",
                    "multiSelect": false,
                    "options": [
                        { "label": "Red", "description": "Warm" },
                        { "label": "Blue", "description": "Cool", "preview": "#00f" }
                    ]
                },
                {
                    "question": "Pick sizes",
                    "header": "Size",
                    "multiSelect": true,
                    "options": [{ "label": "S", "description": "" }, { "label": "M", "description": "" }]
                }
            ]
        })
    }

    #[test]
    fn converts_questions_to_the_harness_shape() {
        let input = to_harness_questions_input(&native());
        assert_eq!(input["allowPartialAnswers"], true);
        assert_eq!(input["questions"][0]["id"], "question-1");
        assert_eq!(input["questions"][0]["options"][1]["id"], "option-2");
        assert_eq!(input["questions"][0]["options"][1]["preview"], "#00f");
        assert_eq!(input["questions"][1]["allowMultiple"], true);
        assert_eq!(input["questions"][1]["allowFreeForm"], true);
        crate::harness::questions_tool::QuestionsToolInput::parse(input).unwrap();
    }

    #[test]
    fn converts_answers_back() {
        let output = QuestionsToolOutput::Answered {
            answers: IndexMap::from([
                (
                    "question-1".to_string(),
                    QuestionAnswer {
                        option_ids: vec!["option-2".into()],
                        freeform: None,
                    },
                ),
                (
                    "question-2".to_string(),
                    QuestionAnswer {
                        option_ids: vec!["option-1".into(), "option-2".into()],
                        freeform: None,
                    },
                ),
            ]),
        };
        let ClaudeCodeQuestionResult::Allow { updated_input } =
            to_claude_code_question_result(&native(), &output)
        else {
            panic!("expected allow");
        };
        assert_eq!(updated_input["answers"]["Pick a color"], "Blue");
        assert_eq!(updated_input["answers"]["Pick sizes"], "S, M");

        let freeform = QuestionsToolOutput::PartiallyAnswered {
            answers: IndexMap::from([(
                "question-1".to_string(),
                QuestionAnswer {
                    option_ids: vec![],
                    freeform: Some("Green".into()),
                },
            )]),
        };
        let ClaudeCodeQuestionResult::Allow { updated_input } =
            to_claude_code_question_result(&native(), &freeform)
        else {
            panic!("expected allow");
        };
        assert_eq!(updated_input["answers"], json!({ "Pick a color": "Green" }));
        assert_eq!(
            to_claude_code_question_result(&native(), &QuestionsToolOutput::Declined),
            ClaudeCodeQuestionResult::Deny {
                message: "The user declined the questions.".into()
            }
        );
    }
}
