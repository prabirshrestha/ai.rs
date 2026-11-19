//! Port of `ai` `ui/process-ui-message-stream.test.ts`.
//!
//! [rust] Only the chunk application is ported (see the module docs), so
//! these tests check the state after each chunk is applied:
//! - the 'should call the update function with the correct arguments'
//!   cases are not ported (there is no `write()`/`runUpdateMessageJob`);
//! - 'replay start' (`resetStateOnFirstMessageStart`), 'data part
//!   schemas', 'onToolCall' / 'onData' assertions, and the `reset-step`,
//!   `steering-message`, `file`, `source-*`, `data-*` and `custom` chunk
//!   tests cover chat-client options or chunk types that are not ported;
//! - the `Object.prototype` cases have no counterpart (ids index an
//!   `IndexMap`), and `AI_SDK_LOG_WARNINGS` warnings are not checked.
//!
//! The chunks, initial state and expected messages of the final-state
//! tests are taken verbatim from the TS file (`undefined` fields dropped,
//! as `toEqual` ignores them).

use serde_json::json;

use super::*;
use crate::harness::provider::JsonValue;

/// Apply `chunks` to a fresh state; a processing failure fails the test
/// (`onError: error => { throw error; }`).
fn run_chunks(init: &JsonValue, chunks: &JsonValue) -> (StreamingUIMessageState, Vec<ErrorValue>) {
    let last_message: Option<UIMessage> = init
        .get("lastMessage")
        .filter(|m| !m.is_null())
        .map(|m| serde_json::from_value(m.clone()).unwrap());
    let message_id = init["messageId"].as_str().unwrap_or_default();
    let mut state = create_streaming_ui_message_state(last_message, message_id);
    let errors = std::cell::RefCell::new(Vec::new());
    for chunk in chunks.as_array().unwrap() {
        let chunk: UIMessageChunk =
            serde_json::from_value(chunk.clone()).unwrap_or_else(|e| panic!("chunk {chunk}: {e}"));
        state
            .apply_chunk(&chunk, &|error| errors.borrow_mut().push(error.clone()))
            .unwrap();
    }
    (state, errors.into_inner())
}

fn try_run(chunks: JsonValue) -> Result<StreamingUIMessageState, UIMessageStreamError> {
    let mut state = create_streaming_ui_message_state(None, "msg-123");
    for chunk in chunks.as_array().unwrap() {
        let chunk: UIMessageChunk = serde_json::from_value(chunk.clone()).unwrap();
        state.apply_chunk(&chunk, &|_| {})?;
    }
    Ok(state)
}

fn run(chunks: JsonValue) -> StreamingUIMessageState {
    try_run(chunks).unwrap()
}

fn parts(state: &StreamingUIMessageState) -> JsonValue {
    serde_json::to_value(&state.message.parts).unwrap()
}

/// A TS `describe` whose `beforeEach` runs `chunks` from `init`.
struct Fixture {
    init: &'static str,
    chunks: &'static str,
    expected: &'static str,
}

impl Fixture {
    fn run(&self) -> (StreamingUIMessageState, Vec<ErrorValue>) {
        run_chunks(
            &serde_json::from_str(self.init).unwrap(),
            &serde_json::from_str(self.chunks).unwrap(),
        )
    }
}

/// `expect(state.message)` (or `state.message.parts`) after the TS
/// `beforeEach` stream.
fn check_final_state(fixture: &Fixture) {
    let (state, _) = fixture.run();
    let expected: JsonValue = serde_json::from_str(fixture.expected).unwrap();
    // some TS cases snapshot `state.message.parts` rather than the message
    let actual = if expected.is_array() {
        parts(&state)
    } else {
        serde_json::to_value(&state.message).unwrap()
    };
    let mut diffs = Vec::new();
    diff("", &actual, &expected, &mut diffs);
    assert!(diffs.is_empty(), "message differs:\n{}", diffs.join("\n"));
}

fn diff(path: &str, actual: &JsonValue, expected: &JsonValue, out: &mut Vec<String>) {
    match (actual, expected) {
        (JsonValue::Object(a), JsonValue::Object(e)) => {
            for key in a.keys().chain(e.keys().filter(|k| !a.contains_key(*k))) {
                let (a, e) = (a.get(key), e.get(key));
                match (a, e) {
                    (Some(a), Some(e)) => diff(&format!("{path}.{key}"), a, e, out),
                    _ => out.push(format!("{path}.{key}: actual {a:?}, expected {e:?}")),
                }
            }
        }
        (JsonValue::Array(a), JsonValue::Array(e)) if a.len() == e.len() => {
            for (i, (a, e)) in a.iter().zip(e).enumerate() {
                diff(&format!("{path}[{i}]"), a, e, out);
            }
        }
        _ if actual != expected => {
            out.push(format!("{path}: actual {actual}, expected {expected}"))
        }
        _ => {}
    }
}

// ─── final message state ───────────────────────────────────────────────

/// `text`
const TEXT: Fixture = Fixture {
    init: r#"{
 "messageId": "msg-123"
}"#,
    chunks: r#"[
 {
  "type": "start",
  "messageId": "msg-123"
 },
 {
  "type": "start-step"
 },
 {
  "type": "text-start",
  "id": "text-1"
 },
 {
  "type": "text-delta",
  "id": "text-1",
  "delta": "Hello, "
 },
 {
  "type": "text-delta",
  "id": "text-1",
  "delta": "world!"
 },
 {
  "type": "text-end",
  "id": "text-1"
 },
 {
  "type": "finish-step"
 },
 {
  "type": "finish"
 }
]"#,
    expected: r#"{
 "id": "msg-123",
 "parts": [
  {
   "type": "step-start"
  },
  {
   "state": "done",
   "text": "Hello, world!",
   "type": "text"
  }
 ],
 "role": "assistant"
}"#,
};

#[test]
fn text_has_the_correct_final_message_state() {
    check_final_state(&TEXT);
}

/// `errors`
const ERRORS: Fixture = Fixture {
    init: r#"{
 "messageId": "msg-123"
}"#,
    chunks: r#"[
 {
  "type": "error",
  "errorText": "test error"
 }
]"#,
    expected: r#"{
 "id": "msg-123",
 "parts": [],
 "role": "assistant"
}"#,
};

#[test]
fn errors_has_the_correct_final_message_state() {
    check_final_state(&ERRORS);
}

/// `server-side tool roundtrip`
const SERVER_SIDE_TOOL_ROUNDTRIP: Fixture = Fixture {
    init: r#"{
 "messageId": "msg-123"
}"#,
    chunks: r#"[
 {
  "type": "start",
  "messageId": "msg-123"
 },
 {
  "type": "start-step"
 },
 {
  "type": "tool-input-available",
  "toolCallId": "tool-call-id",
  "toolName": "tool-name",
  "input": {
   "city": "London"
  }
 },
 {
  "type": "tool-output-available",
  "toolCallId": "tool-call-id",
  "output": {
   "weather": "sunny"
  }
 },
 {
  "type": "finish-step"
 },
 {
  "type": "start-step"
 },
 {
  "type": "text-start",
  "id": "text-1"
 },
 {
  "type": "text-delta",
  "id": "text-1",
  "delta": "The weather in London is sunny."
 },
 {
  "type": "text-end",
  "id": "text-1"
 },
 {
  "type": "finish-step"
 },
 {
  "type": "finish"
 }
]"#,
    expected: r#"{
 "id": "msg-123",
 "parts": [
  {
   "type": "step-start"
  },
  {
   "input": {
    "city": "London"
   },
   "output": {
    "weather": "sunny"
   },
   "state": "output-available",
   "toolCallId": "tool-call-id",
   "type": "tool-tool-name"
  },
  {
   "type": "step-start"
  },
  {
   "state": "done",
   "text": "The weather in London is sunny.",
   "type": "text"
  }
 ],
 "role": "assistant"
}"#,
};

#[test]
fn server_side_tool_roundtrip_has_the_correct_final_message_state() {
    check_final_state(&SERVER_SIDE_TOOL_ROUNDTRIP);
}

/// `server-side tool roundtrip with existing assistant message`
const SERVER_SIDE_TOOL_ROUNDTRIP_WITH_EXISTING_ASSISTANT_MESSAGE: Fixture = Fixture {
    init: r#"{
 "messageId": "msg-123",
 "lastMessage": {
  "role": "assistant",
  "id": "original-id",
  "parts": [
   {
    "type": "tool-tool-name-original",
    "toolCallId": "tool-call-id-original",
    "state": "output-available",
    "input": {},
    "output": {
     "location": "Berlin"
    }
   }
  ]
 }
}"#,
    chunks: r#"[
 {
  "type": "start",
  "messageId": "msg-123"
 },
 {
  "type": "start-step"
 },
 {
  "type": "tool-input-available",
  "toolCallId": "tool-call-id",
  "toolName": "tool-name",
  "input": {
   "city": "London"
  }
 },
 {
  "type": "tool-output-available",
  "toolCallId": "tool-call-id",
  "output": {
   "weather": "sunny"
  }
 },
 {
  "type": "finish-step"
 },
 {
  "type": "start-step"
 },
 {
  "type": "text-start",
  "id": "text-1"
 },
 {
  "type": "text-delta",
  "id": "text-1",
  "delta": "The weather in London is sunny."
 },
 {
  "type": "text-end",
  "id": "text-1"
 },
 {
  "type": "finish-step"
 },
 {
  "type": "finish"
 }
]"#,
    expected: r#"{
 "id": "msg-123",
 "parts": [
  {
   "input": {},
   "output": {
    "location": "Berlin"
   },
   "state": "output-available",
   "toolCallId": "tool-call-id-original",
   "type": "tool-tool-name-original"
  },
  {
   "type": "step-start"
  },
  {
   "input": {
    "city": "London"
   },
   "output": {
    "weather": "sunny"
   },
   "state": "output-available",
   "toolCallId": "tool-call-id",
   "type": "tool-tool-name"
  },
  {
   "type": "step-start"
  },
  {
   "state": "done",
   "text": "The weather in London is sunny.",
   "type": "text"
  }
 ],
 "role": "assistant"
}"#,
};

#[test]
fn server_side_tool_roundtrip_with_existing_assistant_message_has_the_correct_final_message_state()
{
    check_final_state(&SERVER_SIDE_TOOL_ROUNDTRIP_WITH_EXISTING_ASSISTANT_MESSAGE);
}

/// `server-side tool roundtrip with multiple assistant texts`
const SERVER_SIDE_TOOL_ROUNDTRIP_WITH_MULTIPLE_ASSISTANT_TEXTS: Fixture = Fixture {
    init: r#"{
 "messageId": "msg-123"
}"#,
    chunks: r#"[
 {
  "type": "start",
  "messageId": "msg-123"
 },
 {
  "type": "start-step"
 },
 {
  "type": "text-start",
  "id": "text-1"
 },
 {
  "type": "text-delta",
  "id": "text-1",
  "delta": "I will "
 },
 {
  "type": "text-delta",
  "id": "text-1",
  "delta": "use a tool to get the weather in London."
 },
 {
  "type": "text-end",
  "id": "text-1"
 },
 {
  "type": "tool-input-available",
  "toolCallId": "tool-call-id",
  "toolName": "tool-name",
  "input": {
   "city": "London"
  }
 },
 {
  "type": "tool-output-available",
  "toolCallId": "tool-call-id",
  "output": {
   "weather": "sunny"
  }
 },
 {
  "type": "finish-step"
 },
 {
  "type": "start-step"
 },
 {
  "type": "text-start",
  "id": "text-2"
 },
 {
  "type": "text-delta",
  "id": "text-2",
  "delta": "The weather in London "
 },
 {
  "type": "text-delta",
  "id": "text-2",
  "delta": "is sunny."
 },
 {
  "type": "text-end",
  "id": "text-2"
 },
 {
  "type": "finish-step"
 },
 {
  "type": "finish"
 }
]"#,
    expected: r#"{
 "id": "msg-123",
 "parts": [
  {
   "type": "step-start"
  },
  {
   "state": "done",
   "text": "I will use a tool to get the weather in London.",
   "type": "text"
  },
  {
   "input": {
    "city": "London"
   },
   "output": {
    "weather": "sunny"
   },
   "state": "output-available",
   "toolCallId": "tool-call-id",
   "type": "tool-tool-name"
  },
  {
   "type": "step-start"
  },
  {
   "state": "done",
   "text": "The weather in London is sunny.",
   "type": "text"
  }
 ],
 "role": "assistant"
}"#,
};

#[test]
fn server_side_tool_roundtrip_with_multiple_assistant_texts_has_the_correct_final_message_state() {
    check_final_state(&SERVER_SIDE_TOOL_ROUNDTRIP_WITH_MULTIPLE_ASSISTANT_TEXTS);
}

/// `server-side tool roundtrip with multiple assistant reasoning`
const SERVER_SIDE_TOOL_ROUNDTRIP_WITH_MULTIPLE_ASSISTANT_REASONING: Fixture = Fixture {
    init: r#"{
 "messageId": "msg-123"
}"#,
    chunks: r#"[
 {
  "type": "start",
  "messageId": "msg-123"
 },
 {
  "type": "start-step"
 },
 {
  "type": "reasoning-start",
  "id": "reasoning-1"
 },
 {
  "type": "reasoning-delta",
  "id": "reasoning-1",
  "delta": "I will ",
  "providerMetadata": {
   "testProvider": {
    "signature": "1234567890"
   }
  }
 },
 {
  "type": "reasoning-delta",
  "id": "reasoning-1",
  "delta": "use a tool to get the weather in London."
 },
 {
  "type": "reasoning-end",
  "id": "reasoning-1"
 },
 {
  "type": "tool-input-available",
  "toolCallId": "tool-call-id",
  "toolName": "tool-name",
  "input": {
   "city": "London"
  }
 },
 {
  "type": "tool-output-available",
  "toolCallId": "tool-call-id",
  "output": {
   "weather": "sunny"
  }
 },
 {
  "type": "finish-step"
 },
 {
  "type": "start-step"
 },
 {
  "type": "reasoning-start",
  "id": "reasoning-2"
 },
 {
  "type": "reasoning-delta",
  "id": "reasoning-2",
  "delta": "I now know the weather in London.",
  "providerMetadata": {
   "testProvider": {
    "signature": "abc123"
   }
  }
 },
 {
  "type": "reasoning-end",
  "id": "reasoning-2"
 },
 {
  "type": "text-start",
  "id": "text-1"
 },
 {
  "type": "text-delta",
  "id": "text-1",
  "delta": "The weather in London is sunny."
 },
 {
  "type": "text-end",
  "id": "text-1"
 },
 {
  "type": "finish-step"
 },
 {
  "type": "finish"
 }
]"#,
    expected: r#"{
 "id": "msg-123",
 "parts": [
  {
   "type": "step-start"
  },
  {
   "id": "reasoning-1",
   "providerMetadata": {
    "testProvider": {
     "signature": "1234567890"
    }
   },
   "state": "done",
   "text": "I will use a tool to get the weather in London.",
   "type": "reasoning"
  },
  {
   "input": {
    "city": "London"
   },
   "output": {
    "weather": "sunny"
   },
   "state": "output-available",
   "toolCallId": "tool-call-id",
   "type": "tool-tool-name"
  },
  {
   "type": "step-start"
  },
  {
   "id": "reasoning-2",
   "providerMetadata": {
    "testProvider": {
     "signature": "abc123"
    }
   },
   "state": "done",
   "text": "I now know the weather in London.",
   "type": "reasoning"
  },
  {
   "state": "done",
   "text": "The weather in London is sunny.",
   "type": "text"
  }
 ],
 "role": "assistant"
}"#,
};

#[test]
fn server_side_tool_roundtrip_with_multiple_assistant_reasoning_has_the_correct_final_message_state()
 {
    check_final_state(&SERVER_SIDE_TOOL_ROUNDTRIP_WITH_MULTIPLE_ASSISTANT_REASONING);
}

/// `server-side tool roundtrip with output-error`
const SERVER_SIDE_TOOL_ROUNDTRIP_WITH_OUTPUT_ERROR: Fixture = Fixture {
    init: r#"{
 "messageId": "msg-123"
}"#,
    chunks: r#"[
 {
  "type": "start",
  "messageId": "msg-123"
 },
 {
  "type": "start-step"
 },
 {
  "type": "tool-input-available",
  "toolCallId": "tool-call-id",
  "toolName": "tool-name",
  "input": {
   "city": "London"
  }
 },
 {
  "type": "tool-output-error",
  "toolCallId": "tool-call-id",
  "errorText": "error-text"
 },
 {
  "type": "finish-step"
 },
 {
  "type": "start-step"
 },
 {
  "type": "text-start",
  "id": "text-1"
 },
 {
  "type": "text-delta",
  "id": "text-1",
  "delta": "The weather in London is sunny."
 },
 {
  "type": "text-end",
  "id": "text-1"
 },
 {
  "type": "finish-step"
 },
 {
  "type": "finish"
 }
]"#,
    expected: r#"{
 "id": "msg-123",
 "parts": [
  {
   "type": "step-start"
  },
  {
   "errorText": "error-text",
   "input": {
    "city": "London"
   },
   "state": "output-error",
   "toolCallId": "tool-call-id",
   "type": "tool-tool-name"
  },
  {
   "type": "step-start"
  },
  {
   "state": "done",
   "text": "The weather in London is sunny.",
   "type": "text"
  }
 ],
 "role": "assistant"
}"#,
};

#[test]
fn server_side_tool_roundtrip_with_output_error_has_the_correct_final_message_state() {
    check_final_state(&SERVER_SIDE_TOOL_ROUNDTRIP_WITH_OUTPUT_ERROR);
}

/// `message metadata`
const MESSAGE_METADATA: Fixture = Fixture {
    init: r#"{
 "messageId": "msg-123"
}"#,
    chunks: r#"[
 {
  "type": "start",
  "messageId": "msg-123",
  "messageMetadata": {
   "start": "start-1",
   "shared": {
    "key1": "value-1a",
    "key2": "value-2a"
   }
  }
 },
 {
  "type": "start-step"
 },
 {
  "type": "text-start",
  "id": "text-1"
 },
 {
  "type": "text-delta",
  "id": "text-1",
  "delta": "t1"
 },
 {
  "type": "message-metadata",
  "messageMetadata": {
   "metadata": "metadata-1"
  }
 },
 {
  "type": "text-delta",
  "id": "text-1",
  "delta": "t2"
 },
 {
  "type": "text-end",
  "id": "text-1"
 },
 {
  "type": "finish-step"
 },
 {
  "type": "finish",
  "messageMetadata": {
   "finish": "finish-1",
   "shared": {
    "key1": "value-1e",
    "key6": "value-6e"
   }
  }
 }
]"#,
    expected: r#"{
 "id": "msg-123",
 "metadata": {
  "finish": "finish-1",
  "metadata": "metadata-1",
  "shared": {
   "key1": "value-1e",
   "key2": "value-2a",
   "key6": "value-6e"
  },
  "start": "start-1"
 },
 "parts": [
  {
   "type": "step-start"
  },
  {
   "state": "done",
   "text": "t1t2",
   "type": "text"
  }
 ],
 "role": "assistant"
}"#,
};

#[test]
fn message_metadata_has_the_correct_final_message_state() {
    check_final_state(&MESSAGE_METADATA);
}

/// `message metadata delayed after finish`
const MESSAGE_METADATA_DELAYED_AFTER_FINISH: Fixture = Fixture {
    init: r#"{
 "messageId": "msg-123"
}"#,
    chunks: r#"[
 {
  "type": "start",
  "messageId": "msg-123"
 },
 {
  "type": "start-step"
 },
 {
  "type": "text-start",
  "id": "text-1"
 },
 {
  "type": "text-delta",
  "id": "text-1",
  "delta": "t1"
 },
 {
  "type": "text-end",
  "id": "text-1"
 },
 {
  "type": "finish-step"
 },
 {
  "type": "finish"
 },
 {
  "type": "message-metadata",
  "messageMetadata": {
   "key1": "value-1"
  }
 }
]"#,
    expected: r#"{
 "id": "msg-123",
 "metadata": {
  "key1": "value-1"
 },
 "parts": [
  {
   "type": "step-start"
  },
  {
   "state": "done",
   "text": "t1",
   "type": "text"
  }
 ],
 "role": "assistant"
}"#,
};

#[test]
fn message_metadata_delayed_after_finish_has_the_correct_final_message_state() {
    check_final_state(&MESSAGE_METADATA_DELAYED_AFTER_FINISH);
}

/// `message metadata with existing assistant lastMessage`
const MESSAGE_METADATA_WITH_EXISTING_ASSISTANT_LASTMESSAGE: Fixture = Fixture {
    init: r#"{
 "messageId": "msg-123",
 "lastMessage": {
  "role": "assistant",
  "id": "original-id",
  "metadata": {
   "key1": "value-1a",
   "key3": "value-3a"
  },
  "parts": []
 }
}"#,
    chunks: r#"[
 {
  "type": "start",
  "messageId": "msg-123",
  "messageMetadata": {
   "key1": "value-1b",
   "key2": "value-2b"
  }
 },
 {
  "type": "start-step"
 },
 {
  "type": "text-start",
  "id": "text-1"
 },
 {
  "type": "text-delta",
  "id": "text-1",
  "delta": "t1"
 },
 {
  "type": "text-end",
  "id": "text-1"
 },
 {
  "type": "finish-step"
 },
 {
  "type": "finish"
 }
]"#,
    expected: r#"{
 "id": "msg-123",
 "metadata": {
  "key1": "value-1b",
  "key2": "value-2b",
  "key3": "value-3a"
 },
 "parts": [
  {
   "type": "step-start"
  },
  {
   "state": "done",
   "text": "t1",
   "type": "text"
  }
 ],
 "role": "assistant"
}"#,
};

#[test]
fn message_metadata_with_existing_assistant_lastmessage_has_the_correct_final_message_state() {
    check_final_state(&MESSAGE_METADATA_WITH_EXISTING_ASSISTANT_LASTMESSAGE);
}

/// `tool call streaming`
const TOOL_CALL_STREAMING: Fixture = Fixture {
    init: r#"{
 "messageId": "msg-123"
}"#,
    chunks: r#"[
 {
  "type": "start",
  "messageId": "msg-123"
 },
 {
  "type": "start-step"
 },
 {
  "type": "tool-input-start",
  "toolCallId": "tool-call-0",
  "toolName": "test-tool"
 },
 {
  "type": "tool-input-delta",
  "toolCallId": "tool-call-0",
  "inputTextDelta": "{\"testArg\":\"t"
 },
 {
  "type": "tool-input-delta",
  "toolCallId": "tool-call-0",
  "inputTextDelta": "est-value\"}}"
 },
 {
  "type": "tool-input-available",
  "toolCallId": "tool-call-0",
  "toolName": "test-tool",
  "input": {
   "testArg": "test-value"
  }
 },
 {
  "type": "tool-output-available",
  "toolCallId": "tool-call-0",
  "output": "test-result"
 },
 {
  "type": "finish-step"
 },
 {
  "type": "finish"
 }
]"#,
    expected: r#"{
 "id": "msg-123",
 "parts": [
  {
   "type": "step-start"
  },
  {
   "input": {
    "testArg": "test-value"
   },
   "output": "test-result",
   "state": "output-available",
   "toolCallId": "tool-call-0",
   "type": "tool-test-tool"
  }
 ],
 "role": "assistant"
}"#,
};

#[test]
fn tool_call_streaming_has_the_correct_final_message_state() {
    check_final_state(&TOOL_CALL_STREAMING);
}

/// `start with message id`
const START_WITH_MESSAGE_ID: Fixture = Fixture {
    init: r#"{
 "messageId": "msg-123"
}"#,
    chunks: r#"[
 {
  "type": "start",
  "messageId": "msg-123"
 },
 {
  "type": "start-step"
 },
 {
  "type": "text-start",
  "id": "text-1"
 },
 {
  "type": "text-delta",
  "id": "text-1",
  "delta": "Hello, "
 },
 {
  "type": "text-delta",
  "id": "text-1",
  "delta": "world!"
 },
 {
  "type": "text-end",
  "id": "text-1"
 },
 {
  "type": "finish-step"
 },
 {
  "type": "finish"
 }
]"#,
    expected: r#"{
 "id": "msg-123",
 "parts": [
  {
   "type": "step-start"
  },
  {
   "state": "done",
   "text": "Hello, world!",
   "type": "text"
  }
 ],
 "role": "assistant"
}"#,
};

#[test]
fn start_with_message_id_has_the_correct_final_message_state() {
    check_final_state(&START_WITH_MESSAGE_ID);
}

/// `reasoning`
const REASONING: Fixture = Fixture {
    init: r#"{
 "messageId": "msg-123"
}"#,
    chunks: r#"[
 {
  "type": "start",
  "messageId": "msg-123"
 },
 {
  "type": "start-step"
 },
 {
  "type": "reasoning-start",
  "id": "reasoning-1"
 },
 {
  "type": "reasoning-delta",
  "id": "reasoning-1",
  "delta": "I will open the conversation"
 },
 {
  "type": "reasoning-delta",
  "id": "reasoning-1",
  "delta": " with witty banter. ",
  "providerMetadata": {
   "testProvider": {
    "signature": "1234567890"
   }
  }
 },
 {
  "type": "reasoning-end",
  "id": "reasoning-1"
 },
 {
  "type": "reasoning-start",
  "id": "reasoning-2"
 },
 {
  "type": "reasoning-delta",
  "id": "reasoning-2",
  "delta": "redacted-data",
  "providerMetadata": {
   "testProvider": {
    "isRedacted": true
   }
  }
 },
 {
  "type": "reasoning-end",
  "id": "reasoning-2"
 },
 {
  "type": "reasoning-start",
  "id": "reasoning-3"
 },
 {
  "type": "reasoning-delta",
  "id": "reasoning-3",
  "delta": "Once the user has relaxed,"
 },
 {
  "type": "reasoning-delta",
  "id": "reasoning-3",
  "delta": " I will pry for valuable information.",
  "providerMetadata": {
   "testProvider": {
    "signature": "abc123"
   }
  }
 },
 {
  "type": "reasoning-end",
  "id": "reasoning-3"
 },
 {
  "type": "text-start",
  "id": "text-1"
 },
 {
  "type": "text-delta",
  "id": "text-1",
  "delta": "Hi there!"
 },
 {
  "type": "text-end",
  "id": "text-1"
 },
 {
  "type": "finish-step"
 },
 {
  "type": "finish"
 }
]"#,
    expected: r#"{
 "id": "msg-123",
 "parts": [
  {
   "type": "step-start"
  },
  {
   "id": "reasoning-1",
   "providerMetadata": {
    "testProvider": {
     "signature": "1234567890"
    }
   },
   "state": "done",
   "text": "I will open the conversation with witty banter. ",
   "type": "reasoning"
  },
  {
   "id": "reasoning-2",
   "providerMetadata": {
    "testProvider": {
     "isRedacted": true
    }
   },
   "state": "done",
   "text": "redacted-data",
   "type": "reasoning"
  },
  {
   "id": "reasoning-3",
   "providerMetadata": {
    "testProvider": {
     "signature": "abc123"
    }
   },
   "state": "done",
   "text": "Once the user has relaxed, I will pry for valuable information.",
   "type": "reasoning"
  },
  {
   "state": "done",
   "text": "Hi there!",
   "type": "text"
  }
 ],
 "role": "assistant"
}"#,
};

#[test]
fn reasoning_has_the_correct_final_message_state() {
    check_final_state(&REASONING);
}

/// `onToolCall is executed`
const ONTOOLCALL_IS_EXECUTED: Fixture = Fixture {
    init: r#"{
 "messageId": "msg-123"
}"#,
    chunks: r#"[
 {
  "type": "start",
  "messageId": "msg-123"
 },
 {
  "type": "start-step"
 },
 {
  "type": "tool-input-available",
  "toolCallId": "tool-call-id",
  "toolName": "tool-name",
  "input": {
   "city": "London"
  }
 },
 {
  "type": "finish-step"
 },
 {
  "type": "finish"
 }
]"#,
    expected: r#"{
 "id": "msg-123",
 "parts": [
  {
   "type": "step-start"
  },
  {
   "input": {
    "city": "London"
   },
   "state": "input-available",
   "toolCallId": "tool-call-id",
   "type": "tool-tool-name"
  }
 ],
 "role": "assistant"
}"#,
};

#[test]
fn ontoolcall_is_executed_has_the_correct_final_message_state() {
    check_final_state(&ONTOOLCALL_IS_EXECUTED);
}

/// `provider-executed static tools`
const PROVIDER_EXECUTED_STATIC_TOOLS: Fixture = Fixture {
    init: r#"{
 "messageId": "msg-123"
}"#,
    chunks: r#"[
 {
  "type": "start",
  "messageId": "msg-123"
 },
 {
  "type": "start-step"
 },
 {
  "type": "tool-input-start",
  "toolCallId": "tool-call-1",
  "toolName": "tool-name",
  "providerExecuted": true
 },
 {
  "type": "tool-input-delta",
  "toolCallId": "tool-call-1",
  "inputTextDelta": "{ \"query\": \"test\" }"
 },
 {
  "type": "tool-input-available",
  "toolCallId": "tool-call-1",
  "toolName": "tool-name",
  "input": {
   "query": "test"
  },
  "providerExecuted": true
 },
 {
  "type": "tool-output-available",
  "toolCallId": "tool-call-1",
  "output": {
   "result": "provider-result"
  },
  "providerExecuted": true
 },
 {
  "type": "tool-input-available",
  "toolCallId": "tool-call-2",
  "toolName": "tool-name",
  "input": {
   "query": "test"
  },
  "providerExecuted": true
 },
 {
  "type": "tool-output-error",
  "toolCallId": "tool-call-2",
  "errorText": "error-text",
  "providerExecuted": true
 },
 {
  "type": "finish-step"
 },
 {
  "type": "finish"
 }
]"#,
    expected: r#"[
 {
  "type": "step-start"
 },
 {
  "input": {
   "query": "test"
  },
  "output": {
   "result": "provider-result"
  },
  "providerExecuted": true,
  "state": "output-available",
  "toolCallId": "tool-call-1",
  "type": "tool-tool-name"
 },
 {
  "errorText": "error-text",
  "input": {
   "query": "test"
  },
  "providerExecuted": true,
  "state": "output-error",
  "toolCallId": "tool-call-2",
  "type": "tool-tool-name"
 }
]"#,
};

#[test]
fn provider_executed_static_tools_has_the_correct_final_message_state() {
    check_final_state(&PROVIDER_EXECUTED_STATIC_TOOLS);
}

/// `provider-executed dynamic tools`
const PROVIDER_EXECUTED_DYNAMIC_TOOLS: Fixture = Fixture {
    init: r#"{
 "messageId": "msg-123"
}"#,
    chunks: r#"[
 {
  "type": "start",
  "messageId": "msg-123"
 },
 {
  "type": "start-step"
 },
 {
  "type": "tool-input-start",
  "toolCallId": "tool-call-1",
  "toolName": "tool-name",
  "providerExecuted": true,
  "dynamic": true
 },
 {
  "type": "tool-input-delta",
  "toolCallId": "tool-call-1",
  "inputTextDelta": "{ \"query\": \"test\" }"
 },
 {
  "type": "tool-input-available",
  "toolCallId": "tool-call-1",
  "toolName": "tool-name",
  "input": {
   "query": "test"
  },
  "providerExecuted": true,
  "dynamic": true
 },
 {
  "type": "tool-output-available",
  "toolCallId": "tool-call-1",
  "output": {
   "result": "provider-result"
  },
  "providerExecuted": true,
  "dynamic": true
 },
 {
  "type": "tool-input-available",
  "toolCallId": "tool-call-2",
  "toolName": "tool-name",
  "input": {
   "query": "test"
  },
  "providerExecuted": true,
  "dynamic": true
 },
 {
  "type": "tool-output-error",
  "toolCallId": "tool-call-2",
  "errorText": "error-text",
  "providerExecuted": true,
  "dynamic": true
 },
 {
  "type": "finish-step"
 },
 {
  "type": "finish"
 }
]"#,
    expected: r#"[
 {
  "type": "step-start"
 },
 {
  "input": {
   "query": "test"
  },
  "output": {
   "result": "provider-result"
  },
  "providerExecuted": true,
  "state": "output-available",
  "toolCallId": "tool-call-1",
  "toolName": "tool-name",
  "type": "dynamic-tool"
 },
 {
  "errorText": "error-text",
  "input": {
   "query": "test"
  },
  "providerExecuted": true,
  "state": "output-error",
  "toolCallId": "tool-call-2",
  "toolName": "tool-name",
  "type": "dynamic-tool"
 }
]"#,
};

#[test]
fn provider_executed_dynamic_tools_has_the_correct_final_message_state() {
    check_final_state(&PROVIDER_EXECUTED_DYNAMIC_TOOLS);
}

/// `dynamic tools`
const DYNAMIC_TOOLS: Fixture = Fixture {
    init: r#"{
 "messageId": "msg-123"
}"#,
    chunks: r#"[
 {
  "type": "start",
  "messageId": "msg-123"
 },
 {
  "type": "start-step"
 },
 {
  "type": "tool-input-start",
  "toolCallId": "tool-call-1",
  "toolName": "t1",
  "dynamic": true
 },
 {
  "type": "tool-input-delta",
  "toolCallId": "tool-call-1",
  "inputTextDelta": "{ \"query\": \"test\" }"
 },
 {
  "type": "tool-input-available",
  "toolCallId": "tool-call-1",
  "toolName": "t1",
  "input": {
   "query": "test"
  },
  "dynamic": true
 },
 {
  "type": "tool-input-available",
  "toolCallId": "tool-call-2",
  "toolName": "t1",
  "input": {
   "query": "test"
  },
  "dynamic": true
 },
 {
  "type": "tool-output-available",
  "toolCallId": "tool-call-1",
  "output": {
   "result": "provider-result"
  },
  "dynamic": true
 },
 {
  "type": "tool-output-error",
  "toolCallId": "tool-call-2",
  "errorText": "error-text",
  "dynamic": true
 },
 {
  "type": "finish-step"
 },
 {
  "type": "finish"
 }
]"#,
    expected: r#"[
 {
  "type": "step-start"
 },
 {
  "input": {
   "query": "test"
  },
  "output": {
   "result": "provider-result"
  },
  "state": "output-available",
  "toolCallId": "tool-call-1",
  "toolName": "t1",
  "type": "dynamic-tool"
 },
 {
  "errorText": "error-text",
  "input": {
   "query": "test"
  },
  "state": "output-error",
  "toolCallId": "tool-call-2",
  "toolName": "t1",
  "type": "dynamic-tool"
 }
]"#,
};

#[test]
fn dynamic_tools_has_the_correct_final_message_state() {
    check_final_state(&DYNAMIC_TOOLS);
}

/// `provider metadata`
const PROVIDER_METADATA: Fixture = Fixture {
    init: r#"{
 "messageId": "msg-123"
}"#,
    chunks: r#"[
 {
  "type": "start",
  "messageId": "msg-123"
 },
 {
  "type": "start-step"
 },
 {
  "type": "text-start",
  "id": "1",
  "providerMetadata": {
   "testProvider": {
    "signature": "1"
   }
  }
 },
 {
  "type": "text-delta",
  "id": "1",
  "delta": "Hello"
 },
 {
  "type": "text-delta",
  "id": "1",
  "delta": ", "
 },
 {
  "type": "text-delta",
  "id": "1",
  "delta": "world!"
 },
 {
  "type": "text-end",
  "id": "1"
 },
 {
  "type": "tool-input-available",
  "toolCallId": "tool-call-id",
  "toolName": "tool-name",
  "input": {
   "query": "test"
  },
  "providerMetadata": {
   "testProvider": {
    "signature": "2"
   }
  }
 },
 {
  "type": "finish-step"
 },
 {
  "type": "finish"
 }
]"#,
    expected: r#"[
 {
  "type": "step-start"
 },
 {
  "providerMetadata": {
   "testProvider": {
    "signature": "1"
   }
  },
  "state": "done",
  "text": "Hello, world!",
  "type": "text"
 },
 {
  "callProviderMetadata": {
   "testProvider": {
    "signature": "2"
   }
  },
  "input": {
   "query": "test"
  },
  "state": "input-available",
  "toolCallId": "tool-call-id",
  "type": "tool-tool-name"
 }
]"#,
};

#[test]
fn provider_metadata_has_the_correct_final_message_state() {
    check_final_state(&PROVIDER_METADATA);
}

/// `preliminary tool results`
const PRELIMINARY_TOOL_RESULTS: Fixture = Fixture {
    init: r#"{
 "messageId": "msg-123"
}"#,
    chunks: r#"[
 {
  "type": "start"
 },
 {
  "type": "start-step"
 },
 {
  "input": {
   "city": "San Francisco"
  },
  "toolCallId": "call-1",
  "toolName": "cityAttractions",
  "type": "tool-input-available"
 },
 {
  "output": {
   "status": "loading",
   "text": "Getting weather for San Francisco"
  },
  "preliminary": true,
  "toolCallId": "call-1",
  "type": "tool-output-available"
 },
 {
  "output": {
   "status": "success",
   "temperature": 72,
   "text": "The weather in San Francisco is 72\u00b0F"
  },
  "preliminary": true,
  "toolCallId": "call-1",
  "type": "tool-output-available"
 },
 {
  "output": {
   "status": "success",
   "temperature": 72,
   "text": "The weather in San Francisco is 72\u00b0F"
  },
  "toolCallId": "call-1",
  "type": "tool-output-available"
 },
 {
  "type": "finish-step"
 },
 {
  "type": "finish"
 }
]"#,
    expected: r#"[
 {
  "type": "step-start"
 },
 {
  "input": {
   "city": "San Francisco"
  },
  "output": {
   "status": "success",
   "temperature": 72,
   "text": "The weather in San Francisco is 72\u00b0F"
  },
  "state": "output-available",
  "toolCallId": "call-1",
  "type": "tool-cityAttractions"
 }
]"#,
};

#[test]
fn preliminary_tool_results_has_the_correct_final_message_state() {
    check_final_state(&PRELIMINARY_TOOL_RESULTS);
}

/// `tool approval requests (static tool)`
const TOOL_APPROVAL_REQUESTS_STATIC_TOOL: Fixture = Fixture {
    init: r#"{
 "messageId": "msg-123"
}"#,
    chunks: r#"[
 {
  "type": "start"
 },
 {
  "type": "start-step"
 },
 {
  "input": {
   "value": "value"
  },
  "toolCallId": "call-1",
  "toolName": "tool1",
  "type": "tool-input-available"
 },
 {
  "approvalId": "id-1",
  "toolCallId": "call-1",
  "type": "tool-approval-request"
 },
 {
  "type": "finish-step"
 },
 {
  "type": "finish"
 }
]"#,
    expected: r#"[
 {
  "type": "step-start"
 },
 {
  "approval": {
   "id": "id-1"
  },
  "input": {
   "value": "value"
  },
  "state": "approval-requested",
  "toolCallId": "call-1",
  "type": "tool-tool1"
 }
]"#,
};

#[test]
fn tool_approval_requests_static_tool_has_the_correct_final_message_state() {
    check_final_state(&TOOL_APPROVAL_REQUESTS_STATIC_TOOL);
}

/// `tool approval requests (dynamic tool)`
const TOOL_APPROVAL_REQUESTS_DYNAMIC_TOOL: Fixture = Fixture {
    init: r#"{
 "messageId": "msg-123"
}"#,
    chunks: r#"[
 {
  "type": "start"
 },
 {
  "type": "start-step"
 },
 {
  "input": {
   "value": "value"
  },
  "toolCallId": "call-1",
  "toolName": "tool1",
  "type": "tool-input-available",
  "dynamic": true
 },
 {
  "approvalId": "id-1",
  "toolCallId": "call-1",
  "type": "tool-approval-request"
 },
 {
  "type": "finish-step"
 },
 {
  "type": "finish"
 }
]"#,
    expected: r#"[
 {
  "type": "step-start"
 },
 {
  "approval": {
   "id": "id-1"
  },
  "input": {
   "value": "value"
  },
  "state": "approval-requested",
  "toolCallId": "call-1",
  "toolName": "tool1",
  "type": "dynamic-tool"
 }
]"#,
};

#[test]
fn tool_approval_requests_dynamic_tool_has_the_correct_final_message_state() {
    check_final_state(&TOOL_APPROVAL_REQUESTS_DYNAMIC_TOOL);
}

/// `initial tool execution after approval (static tool)`
const INITIAL_TOOL_EXECUTION_AFTER_APPROVAL_STATIC_TOOL: Fixture = Fixture {
    init: r#"{
 "messageId": "msg-123",
 "lastMessage": {
  "role": "assistant",
  "id": "original-id",
  "parts": [
   {
    "type": "step-start"
   },
   {
    "approval": {
     "id": "id-1",
     "approved": true
    },
    "input": {
     "value": "value"
    },
    "state": "approval-responded",
    "toolCallId": "call-1",
    "type": "tool-tool1"
   }
  ]
 }
}"#,
    chunks: r#"[
 {
  "type": "start"
 },
 {
  "output": "result1",
  "toolCallId": "call-1",
  "type": "tool-output-available"
 },
 {
  "type": "start-step"
 },
 {
  "id": "1",
  "type": "text-start"
 },
 {
  "delta": "Hello, world!",
  "id": "1",
  "type": "text-delta"
 },
 {
  "id": "1",
  "type": "text-end"
 },
 {
  "type": "finish-step"
 },
 {
  "type": "finish"
 }
]"#,
    expected: r#"[
 {
  "type": "step-start"
 },
 {
  "approval": {
   "approved": true,
   "id": "id-1"
  },
  "input": {
   "value": "value"
  },
  "output": "result1",
  "state": "output-available",
  "toolCallId": "call-1",
  "type": "tool-tool1"
 },
 {
  "type": "step-start"
 },
 {
  "state": "done",
  "text": "Hello, world!",
  "type": "text"
 }
]"#,
};

#[test]
fn initial_tool_execution_after_approval_static_tool_has_the_correct_final_message_state() {
    check_final_state(&INITIAL_TOOL_EXECUTION_AFTER_APPROVAL_STATIC_TOOL);
}

/// `initial tool execution after approval (dynamic tool)`
const INITIAL_TOOL_EXECUTION_AFTER_APPROVAL_DYNAMIC_TOOL: Fixture = Fixture {
    init: r#"{
 "messageId": "msg-123",
 "lastMessage": {
  "role": "assistant",
  "id": "original-id",
  "parts": [
   {
    "type": "step-start"
   },
   {
    "approval": {
     "id": "id-1",
     "approved": true
    },
    "input": {
     "value": "value"
    },
    "state": "approval-responded",
    "toolCallId": "call-1",
    "type": "dynamic-tool",
    "toolName": "tool1"
   }
  ]
 }
}"#,
    chunks: r#"[
 {
  "type": "start"
 },
 {
  "output": "result1",
  "toolCallId": "call-1",
  "type": "tool-output-available",
  "dynamic": true
 },
 {
  "type": "start-step"
 },
 {
  "id": "1",
  "type": "text-start"
 },
 {
  "delta": "Hello, world!",
  "id": "1",
  "type": "text-delta"
 },
 {
  "id": "1",
  "type": "text-end"
 },
 {
  "type": "finish-step"
 },
 {
  "type": "finish"
 }
]"#,
    expected: r#"[
 {
  "type": "step-start"
 },
 {
  "approval": {
   "approved": true,
   "id": "id-1"
  },
  "input": {
   "value": "value"
  },
  "output": "result1",
  "state": "output-available",
  "toolCallId": "call-1",
  "toolName": "tool1",
  "type": "dynamic-tool"
 },
 {
  "type": "step-start"
 },
 {
  "state": "done",
  "text": "Hello, world!",
  "type": "text"
 }
]"#,
};

#[test]
fn initial_tool_execution_after_approval_dynamic_tool_has_the_correct_final_message_state() {
    check_final_state(&INITIAL_TOOL_EXECUTION_AFTER_APPROVAL_DYNAMIC_TOOL);
}

/// `initial tool execution with preliminary results after approval (static tool)`
const INITIAL_TOOL_EXECUTION_WITH_PRELIMINARY_RESULTS_AFTER_APPROVAL_STATIC_TOOL: Fixture =
    Fixture {
        init: r#"{
 "messageId": "msg-123",
 "lastMessage": {
  "role": "assistant",
  "id": "original-id",
  "parts": [
   {
    "type": "step-start"
   },
   {
    "approval": {
     "id": "id-1",
     "approved": true
    },
    "input": {
     "value": "value"
    },
    "state": "approval-responded",
    "toolCallId": "call-1",
    "type": "tool-tool1"
   }
  ]
 }
}"#,
        chunks: r#"[
 {
  "type": "start"
 },
 {
  "output": "preliminary-result",
  "preliminary": true,
  "toolCallId": "call-1",
  "type": "tool-output-available"
 },
 {
  "output": "final-result",
  "preliminary": true,
  "toolCallId": "call-1",
  "type": "tool-output-available"
 },
 {
  "output": "final-result",
  "toolCallId": "call-1",
  "type": "tool-output-available"
 },
 {
  "type": "start-step"
 },
 {
  "id": "1",
  "type": "text-start"
 },
 {
  "delta": "Hello, world!",
  "id": "1",
  "type": "text-delta"
 },
 {
  "id": "1",
  "type": "text-end"
 },
 {
  "type": "finish-step"
 },
 {
  "type": "finish"
 }
]"#,
        expected: r#"[
 {
  "type": "step-start"
 },
 {
  "approval": {
   "approved": true,
   "id": "id-1"
  },
  "input": {
   "value": "value"
  },
  "output": "final-result",
  "state": "output-available",
  "toolCallId": "call-1",
  "type": "tool-tool1"
 },
 {
  "type": "step-start"
 },
 {
  "state": "done",
  "text": "Hello, world!",
  "type": "text"
 }
]"#,
    };

#[test]
fn initial_tool_execution_with_preliminary_results_after_approval_static_tool_has_the_correct_final_message_state()
 {
    check_final_state(&INITIAL_TOOL_EXECUTION_WITH_PRELIMINARY_RESULTS_AFTER_APPROVAL_STATIC_TOOL);
}

/// `tool execution denial (static tool)`
const TOOL_EXECUTION_DENIAL_STATIC_TOOL: Fixture = Fixture {
    init: r#"{
 "messageId": "msg-123",
 "lastMessage": {
  "role": "assistant",
  "id": "original-id",
  "parts": [
   {
    "type": "step-start"
   },
   {
    "approval": {
     "id": "id-1",
     "approved": false
    },
    "input": {
     "value": "value"
    },
    "state": "approval-responded",
    "toolCallId": "call-1",
    "type": "tool-tool1"
   }
  ]
 }
}"#,
    chunks: r#"[
 {
  "type": "start"
 },
 {
  "toolCallId": "call-1",
  "type": "tool-output-denied"
 },
 {
  "type": "start-step"
 },
 {
  "id": "1",
  "type": "text-start"
 },
 {
  "id": "1",
  "type": "text-delta",
  "delta": "I did not execute the tool."
 },
 {
  "id": "1",
  "type": "text-end"
 },
 {
  "type": "finish-step"
 },
 {
  "type": "finish"
 }
]"#,
    expected: r#"[
 {
  "type": "step-start"
 },
 {
  "approval": {
   "approved": false,
   "id": "id-1"
  },
  "input": {
   "value": "value"
  },
  "state": "output-denied",
  "toolCallId": "call-1",
  "type": "tool-tool1"
 },
 {
  "type": "step-start"
 },
 {
  "state": "done",
  "text": "I did not execute the tool.",
  "type": "text"
 }
]"#,
};

#[test]
fn tool_execution_denial_static_tool_has_the_correct_final_message_state() {
    check_final_state(&TOOL_EXECUTION_DENIAL_STATIC_TOOL);
}

/// `tool execution denial (dynamic tool)`
const TOOL_EXECUTION_DENIAL_DYNAMIC_TOOL: Fixture = Fixture {
    init: r#"{
 "messageId": "msg-123",
 "lastMessage": {
  "role": "assistant",
  "id": "original-id",
  "parts": [
   {
    "type": "step-start"
   },
   {
    "approval": {
     "id": "id-1",
     "approved": false
    },
    "input": {
     "value": "value"
    },
    "state": "approval-responded",
    "toolCallId": "call-1",
    "type": "dynamic-tool",
    "toolName": "tool1"
   }
  ]
 }
}"#,
    chunks: r#"[
 {
  "type": "start"
 },
 {
  "toolCallId": "call-1",
  "type": "tool-output-denied"
 },
 {
  "type": "start-step"
 },
 {
  "id": "1",
  "type": "text-start"
 },
 {
  "id": "1",
  "type": "text-delta",
  "delta": "I did not execute the tool."
 },
 {
  "id": "1",
  "type": "text-end"
 },
 {
  "type": "finish-step"
 },
 {
  "type": "finish"
 }
]"#,
    expected: r#"[
 {
  "type": "step-start"
 },
 {
  "approval": {
   "approved": false,
   "id": "id-1"
  },
  "input": {
   "value": "value"
  },
  "state": "output-denied",
  "toolCallId": "call-1",
  "toolName": "tool1",
  "type": "dynamic-tool"
 },
 {
  "type": "step-start"
 },
 {
  "state": "done",
  "text": "I did not execute the tool.",
  "type": "text"
 }
]"#,
};

#[test]
fn tool_execution_denial_dynamic_tool_has_the_correct_final_message_state() {
    check_final_state(&TOOL_EXECUTION_DENIAL_DYNAMIC_TOOL);
}

// ─── individual cases ──────────────────────────────────────────────────

fn tool_part<'a>(state: &'a StreamingUIMessageState, tool_call_id: &str) -> &'a UIToolPart {
    state
        .message
        .parts
        .iter()
        .find_map(|part| match part {
            UIMessagePart::Tool(tool) if tool.tool_call_id == tool_call_id => Some(tool),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no tool part {tool_call_id}"))
}

fn tool_json(state: &StreamingUIMessageState, tool_call_id: &str) -> JsonValue {
    serde_json::to_value(UIMessagePart::Tool(tool_part(state, tool_call_id).clone())).unwrap()
}

/// `parts[index]` after each chunk that changes the message
/// (`writeCalls.map(call => call.message.parts[index])`), skipping chunks
/// before the part exists.
fn part_after_each_chunk(chunks: JsonValue, index: usize) -> Vec<JsonValue> {
    let mut state = create_streaming_ui_message_state(None, "msg-123");
    let mut snapshots: Vec<JsonValue> = Vec::new();
    for chunk in chunks.as_array().unwrap() {
        let chunk: UIMessageChunk = serde_json::from_value(chunk.clone()).unwrap();
        state.apply_chunk(&chunk, &|_| {}).unwrap();
        if let Some(part) = state.message.parts.get(index) {
            let part = serde_json::to_value(part).unwrap();
            if snapshots.last() != Some(&part) {
                snapshots.push(part);
            }
        }
    }
    snapshots
}

#[test]
fn finish_step_preserves_active_text_and_reasoning_parts_across_interleaved_step_boundaries() {
    let state = run(json!([
        { "type": "text-start", "id": "text-1" },
        { "type": "text-delta", "id": "text-1", "delta": "first " },
        { "type": "reasoning-start", "id": "reasoning-1" },
        { "type": "reasoning-delta", "id": "reasoning-1", "delta": "thinking " },
        { "type": "start-step" },
        { "type": "finish-step" },
        { "type": "text-delta", "id": "text-1", "delta": "second" },
        { "type": "reasoning-delta", "id": "reasoning-1", "delta": "continued" },
        { "type": "text-end", "id": "text-1" },
        { "type": "reasoning-end", "id": "reasoning-1" },
    ]));
    assert_eq!(
        parts(&state),
        json!([
            { "type": "text", "text": "first second", "state": "done" },
            { "type": "reasoning", "id": "reasoning-1", "text": "thinking continued", "state": "done" },
            { "type": "step-start" },
        ])
    );
    assert!(state.active_text_parts.is_empty());
    assert!(state.active_reasoning_parts.is_empty());
}

#[test]
fn errors_calls_on_error_with_the_correct_arguments() {
    let (_, errors) = ERRORS.run();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].as_error().unwrap().to_string(), "test error");
}

// malformed stream errors

fn assert_rejected(chunk: JsonValue, chunk_type: &str, chunk_id: &str, message: &str) {
    let error = try_run(json!([
        { "type": "start", "messageId": "msg-123" },
        { "type": "start-step" },
        chunk,
    ]))
    .expect_err("the chunk is rejected");
    assert_eq!(error.to_string(), message);
    assert_eq!(error.chunk_type, chunk_type);
    assert_eq!(error.chunk_id, chunk_id);
}

#[test]
fn should_throw_descriptive_error_when_text_delta_is_received_without_text_start() {
    assert_rejected(
        json!({ "type": "text-delta", "id": "text-1", "delta": "Hello" }),
        "text-delta",
        "text-1",
        "Received text-delta for missing text part with ID \"text-1\". Ensure a \"text-start\" chunk is sent before any \"text-delta\" chunks.",
    );
}

#[test]
fn should_throw_descriptive_error_when_reasoning_delta_is_received_without_reasoning_start() {
    assert_rejected(
        json!({ "type": "reasoning-delta", "id": "reasoning-1", "delta": "Thinking..." }),
        "reasoning-delta",
        "reasoning-1",
        "Received reasoning-delta for missing reasoning part with ID \"reasoning-1\". Ensure a \"reasoning-start\" chunk is sent before any \"reasoning-delta\" chunks.",
    );
}

#[test]
fn should_throw_descriptive_error_when_tool_input_delta_is_received_without_tool_input_start() {
    assert_rejected(
        json!({ "type": "tool-input-delta", "toolCallId": "tool-1", "inputTextDelta": "{\"key\":" }),
        "tool-input-delta",
        "tool-1",
        "Received tool-input-delta for missing tool call with ID \"tool-1\". Ensure a \"tool-input-start\" chunk is sent before any \"tool-input-delta\" chunks.",
    );
}

#[test]
fn should_throw_descriptive_error_when_text_end_is_received_without_text_start() {
    assert_rejected(
        json!({ "type": "text-end", "id": "text-1" }),
        "text-end",
        "text-1",
        "Received text-end for missing text part with ID \"text-1\". Ensure a \"text-start\" chunk is sent before any \"text-end\" chunks.",
    );
}

#[test]
fn should_throw_descriptive_error_when_reasoning_end_is_received_without_reasoning_start() {
    assert_rejected(
        json!({ "type": "reasoning-end", "id": "reasoning-1" }),
        "reasoning-end",
        "reasoning-1",
        "Received reasoning-end for missing reasoning part with ID \"reasoning-1\". Ensure a \"reasoning-start\" chunk is sent before any \"reasoning-end\" chunks.",
    );
}

#[test]
fn should_throw_ui_message_stream_error_with_correct_properties_for_missing_ids() {
    let error = try_run(json!([
        { "type": "start", "messageId": "msg-123" },
        { "type": "start-step" },
        { "type": "text-delta", "id": "missing-id", "delta": "Hello" },
    ]))
    .unwrap_err();
    assert_eq!(
        (error.chunk_type.as_str(), error.chunk_id.as_str()),
        ("text-delta", "missing-id")
    );
    let error = try_run(json!([
        { "type": "start", "messageId": "msg-123" },
        { "type": "start-step" },
        { "type": "tool-input-delta", "toolCallId": "missing-tool-id", "inputTextDelta": "{\"key\":" },
    ]))
    .unwrap_err();
    assert_eq!(
        (error.chunk_type.as_str(), error.chunk_id.as_str()),
        ("tool-input-delta", "missing-tool-id")
    );
}

#[test]
fn static_tool_name_changes_use_the_completed_input_tool_name_for_an_existing_streamed_part() {
    let state = run(json!([
        { "type": "start-step" },
        { "type": "tool-input-start", "toolCallId": "tool-call-0", "toolName": "originalTool" },
        { "type": "tool-input-delta", "toolCallId": "tool-call-0", "inputTextDelta": "{\"value\":\"test\"}" },
        { "type": "tool-input-available", "toolCallId": "tool-call-0", "toolName": "renamedTool", "input": { "value": "test" } },
        { "type": "tool-approval-request", "toolCallId": "tool-call-0", "approvalId": "approval-0" },
    ]));
    assert_eq!(
        parts(&state)[1],
        json!({
            "type": "tool-renamedTool",
            "toolCallId": "tool-call-0",
            "state": "approval-requested",
            "input": { "value": "test" },
            "approval": { "id": "approval-0" },
        })
    );
}

#[test]
fn reasoning_should_preserve_reasoning_part_ids() {
    let (state, _) = REASONING.run();
    let ids: Vec<_> = state
        .message
        .parts
        .iter()
        .filter_map(|part| match part {
            UIMessagePart::Reasoning(reasoning) => reasoning.id.clone(),
            _ => None,
        })
        .collect();
    assert_eq!(ids, ["reasoning-1", "reasoning-2", "reasoning-3"]);
}

#[test]
fn should_preserve_separate_call_and_result_provider_metadata_for_static_and_dynamic_tools() {
    for (dynamic, output) in [
        (
            None,
            json!({ "type": "tool-output-available", "output": { "result": "provider-result" } }),
        ),
        (
            Some(true),
            json!({ "type": "tool-output-error", "errorText": "error-text" }),
        ),
    ] {
        let mut input = json!({
            "type": "tool-input-available",
            "toolCallId": "tool-call-1",
            "toolName": "tool-name",
            "input": { "query": "test" },
            "providerExecuted": true,
            "providerMetadata": { "testProvider": { "itemId": "call-item" } },
        });
        let mut output = output;
        output["toolCallId"] = json!("tool-call-1");
        output["providerExecuted"] = json!(true);
        output["providerMetadata"] = json!({ "testProvider": { "itemId": "result-item" } });
        if let Some(dynamic) = dynamic {
            input["dynamic"] = json!(dynamic);
            output["dynamic"] = json!(dynamic);
        }
        let state = run(json!([
            { "type": "start", "messageId": "msg-123" },
            { "type": "start-step" },
            input,
            output,
            { "type": "finish-step" },
            { "type": "finish" },
        ]));
        let part = tool_json(&state, "tool-call-1");
        assert_eq!(
            part["callProviderMetadata"],
            json!({ "testProvider": { "itemId": "call-item" } })
        );
        assert_eq!(
            part["resultProviderMetadata"],
            json!({ "testProvider": { "itemId": "result-item" } })
        );
    }
}

#[test]
fn should_preserve_tool_metadata_on_dynamic_tool_parts() {
    let state = run(json!([
        { "type": "start", "messageId": "msg-123" },
        { "type": "start-step" },
        {
            "type": "tool-input-available",
            "toolCallId": "tool-call-1",
            "toolName": "tool-name",
            "input": { "query": "test" },
            "dynamic": true,
            "toolMetadata": { "clientName": "MyMCPClient" },
        },
        { "type": "tool-output-available", "toolCallId": "tool-call-1", "output": { "result": "provider-result" }, "dynamic": true },
        { "type": "finish-step" },
        { "type": "finish" },
    ]));
    assert_eq!(
        tool_json(&state, "tool-call-1"),
        json!({
            "input": { "query": "test" },
            "output": { "result": "provider-result" },
            "state": "output-available",
            "toolCallId": "tool-call-1",
            "toolMetadata": { "clientName": "MyMCPClient" },
            "toolName": "tool-name",
            "type": "dynamic-tool",
        })
    );
}

#[test]
fn should_apply_output_chunk_tool_metadata_to_static_and_dynamic_tool_parts() {
    let input = |id: &str, dynamic: bool, phase: Option<&str>| {
        let mut chunk = json!({ "type": "tool-input-available", "toolCallId": id, "toolName": "tool-name", "input": { "query": "test" } });
        if dynamic {
            chunk["dynamic"] = json!(true);
        }
        if let Some(phase) = phase {
            chunk["toolMetadata"] = json!({ "phase": phase });
        }
        chunk
    };
    let output = |id: &str, dynamic: bool, error: bool, phase: &str| {
        let mut chunk = if error {
            json!({ "type": "tool-output-error", "toolCallId": id, "errorText": "error-text" })
        } else {
            json!({ "type": "tool-output-available", "toolCallId": id, "output": { "result": "provider-result" } })
        };
        if dynamic {
            chunk["dynamic"] = json!(true);
        }
        chunk["toolMetadata"] = json!({ "phase": phase });
        chunk
    };
    let state = run(json!([
        { "type": "start", "messageId": "msg-123" },
        { "type": "start-step" },
        input("dynamic-success", true, None),
        output("dynamic-success", true, false, "dynamic-output-available"),
        input("dynamic-error", true, Some("dynamic-input")),
        output("dynamic-error", true, true, "dynamic-output-error"),
        input("static-success", false, None),
        output("static-success", false, false, "static-output-available"),
        input("static-error", false, Some("static-input")),
        output("static-error", false, true, "static-output-error"),
        { "type": "finish-step" },
        { "type": "finish" },
    ]));
    let summary: Vec<_> = state
        .message
        .parts
        .iter()
        .filter_map(|part| match part {
            UIMessagePart::Tool(_) => {
                let part = serde_json::to_value(part).unwrap();
                Some(json!({
                    "toolCallId": part["toolCallId"],
                    "state": part["state"],
                    "toolMetadata": part["toolMetadata"],
                }))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        summary,
        vec![
            json!({ "toolCallId": "dynamic-success", "state": "output-available", "toolMetadata": { "phase": "dynamic-output-available" } }),
            json!({ "toolCallId": "dynamic-error", "state": "output-error", "toolMetadata": { "phase": "dynamic-output-error" } }),
            json!({ "toolCallId": "static-success", "state": "output-available", "toolMetadata": { "phase": "static-output-available" } }),
            json!({ "toolCallId": "static-error", "state": "output-error", "toolMetadata": { "phase": "static-output-error" } }),
        ]
    );
}

#[test]
fn client_executed_tools_reach_input_available() {
    // [rust] the `onToolCall` call is not ported; the part is.
    let state = run(json!([
        { "type": "start", "messageId": "msg-123" },
        { "type": "start-step" },
        { "type": "tool-input-available", "toolCallId": "tool-call-id", "toolName": "tool-name", "input": { "query": "test" } },
        { "type": "finish-step" },
        { "type": "finish" },
    ]));
    assert_eq!(
        parts(&state),
        json!([
            { "type": "step-start" },
            { "input": { "query": "test" }, "state": "input-available", "toolCallId": "tool-call-id", "type": "tool-tool-name" },
        ])
    );
}

#[test]
fn provider_metadata_during_input_streaming() {
    let chunks = json!([
        { "type": "start", "messageId": "msg-123" },
        { "type": "start-step" },
        {
            "type": "tool-input-start",
            "toolCallId": "tool-call-id",
            "toolName": "tool-name",
            "providerMetadata": { "testProvider": { "someKey": "someValue" } },
        },
        { "type": "tool-input-delta", "toolCallId": "tool-call-id", "inputTextDelta": "{\"query\":" },
        { "type": "tool-input-delta", "toolCallId": "tool-call-id", "inputTextDelta": "\"test\"}" },
        { "type": "tool-input-available", "toolCallId": "tool-call-id", "toolName": "tool-name", "input": { "query": "test" } },
        { "type": "finish-step" },
        { "type": "finish" },
    ]);
    // should propagate providerMetadata during input-streaming state
    let streaming = part_after_each_chunk(chunks.clone(), 1)
        .into_iter()
        .find(|part| part["state"] == "input-streaming")
        .unwrap();
    assert_eq!(
        streaming["callProviderMetadata"],
        json!({ "testProvider": { "someKey": "someValue" } })
    );
    // should retain providerMetadata in final input-available state
    let part = tool_json(&run(chunks), "tool-call-id");
    assert_eq!(part["state"], "input-available");
    assert_eq!(
        part["callProviderMetadata"],
        json!({ "testProvider": { "someKey": "someValue" } })
    );
}

#[test]
fn tool_input_error_has_the_correct_final_message_state() {
    let state = run(json!([
        { "type": "start" },
        { "type": "start-step" },
        { "toolCallId": "call-1", "toolName": "cityAttractions", "type": "tool-input-start" },
        { "inputTextDelta": "{ \"cities\": \"San Francisco\" }", "toolCallId": "call-1", "type": "tool-input-delta" },
        {
            "errorText": "Invalid input for tool cityAttractions",
            "input": "{ \"cities\": \"San Francisco\" }",
            "toolCallId": "call-1",
            "toolName": "cityAttractions",
            "type": "tool-input-error",
        },
        { "errorText": "Invalid input for tool cityAttractions", "toolCallId": "call-1", "type": "tool-output-error" },
        { "type": "finish-step" },
        { "type": "finish" },
    ]));
    assert_eq!(
        parts(&state),
        json!([
            { "type": "step-start" },
            {
                "errorText": "Invalid input for tool cityAttractions",
                "input": "{ \"cities\": \"San Francisco\" }",
                "state": "output-error",
                "toolCallId": "call-1",
                "type": "tool-cityAttractions",
            },
        ])
    );
}

#[test]
fn dynamic_tool_errors_after_input_streaming_clear_raw_input() {
    for terminal in [
        json!({
            "type": "tool-input-error",
            "toolCallId": "call-1",
            "toolName": "cityAttractions",
            "input": { "cities": ["San Francisco"] },
            "errorText": "Invalid input for tool cityAttractions",
            "dynamic": true,
        }),
        json!({ "type": "tool-output-error", "toolCallId": "call-1", "errorText": "Tool execution failed", "dynamic": true }),
    ] {
        let state = run(json!([
            { "type": "start" },
            { "type": "start-step" },
            { "type": "tool-input-start", "toolCallId": "call-1", "toolName": "cityAttractions", "dynamic": true },
            { "type": "tool-input-delta", "toolCallId": "call-1", "inputTextDelta": "{ \"cities\": [\"San Francisco\"] }" },
            terminal,
            { "type": "finish-step" },
            { "type": "finish" },
        ]));
        let part = tool_json(&state, "call-1");
        assert_eq!(part["type"], "dynamic-tool");
        assert_eq!(part["state"], "output-error");
        assert!(part.get("rawInput").is_none());
    }
}

#[test]
fn tool_input_error_with_dynamic_flag_mismatch() {
    let state = run(json!([
        { "type": "start" },
        { "type": "start-step" },
        // dynamic is not set: the tool is not in the tools object
        { "toolCallId": "call-1", "toolName": "nonExistentTool", "type": "tool-input-start" },
        { "inputTextDelta": "{ \"foo\": \"bar\" }", "toolCallId": "call-1", "type": "tool-input-delta" },
        {
            "errorText": "Model tried to call unavailable tool 'nonExistentTool'.",
            "input": "{ \"foo\": \"bar\" }",
            "toolCallId": "call-1",
            "toolName": "nonExistentTool",
            "type": "tool-input-error",
            // parseToolCall's NoSuchToolError catch sets dynamic
            "dynamic": true,
        },
        { "errorText": "Model tried to call unavailable tool 'nonExistentTool'.", "toolCallId": "call-1", "type": "tool-output-error" },
        { "type": "finish-step" },
        { "type": "finish" },
    ]));
    // exactly one tool part (no duplicate), keeping the static type
    assert_eq!(
        parts(&state),
        json!([
            { "type": "step-start" },
            {
                "errorText": "Model tried to call unavailable tool 'nonExistentTool'.",
                "input": "{ \"foo\": \"bar\" }",
                "state": "output-error",
                "toolCallId": "call-1",
                "type": "tool-nonExistentTool",
            },
        ])
    );
}

// tool title support

#[test]
fn static_tool_with_title() {
    let chunks = json!([
        { "type": "start", "messageId": "msg-123" },
        { "type": "start-step" },
        { "type": "tool-input-start", "toolCallId": "tool-call-0", "toolName": "weatherTool", "title": "Weather Information" },
        { "type": "tool-input-delta", "toolCallId": "tool-call-0", "inputTextDelta": "{\"location\":\"Paris\"}" },
        { "type": "tool-input-available", "toolCallId": "tool-call-0", "toolName": "weatherTool", "input": { "location": "Paris" }, "title": "Weather Information" },
        { "type": "tool-output-available", "toolCallId": "tool-call-0", "output": "Sunny, 22°C" },
        { "type": "finish-step" },
        { "type": "finish" },
    ]);
    // should include title in tool invocation parts
    assert_eq!(
        tool_part(&run(chunks.clone()), "tool-call-0")
            .title
            .as_deref(),
        Some("Weather Information")
    );
    // should preserve title through all states
    let snapshots = part_after_each_chunk(chunks, 1);
    for state in ["input-streaming", "input-available", "output-available"] {
        let part = snapshots
            .iter()
            .find(|part| part["state"] == state)
            .unwrap();
        assert_eq!(part["title"], "Weather Information", "{state}");
    }
}

#[test]
fn dynamic_tool_with_title() {
    let mut state = create_streaming_ui_message_state(None, "msg-456");
    for chunk in [
        json!({ "type": "start", "messageId": "msg-456" }),
        json!({ "type": "start-step" }),
        json!({ "type": "tool-input-start", "toolCallId": "tool-call-1", "toolName": "calculate", "dynamic": true, "title": "Calculator" }),
        json!({ "type": "tool-input-delta", "toolCallId": "tool-call-1", "inputTextDelta": "{\"a\":5,\"b\":3}" }),
        json!({ "type": "tool-input-available", "toolCallId": "tool-call-1", "toolName": "calculate", "input": { "a": 5, "b": 3 }, "dynamic": true, "title": "Calculator" }),
        json!({ "type": "tool-output-available", "toolCallId": "tool-call-1", "output": { "result": 8 }, "dynamic": true }),
        json!({ "type": "finish-step" }),
        json!({ "type": "finish" }),
    ] {
        state
            .apply_chunk(&serde_json::from_value(chunk).unwrap(), &|_| {})
            .unwrap();
    }
    let part = tool_json(&state, "tool-call-1");
    assert_eq!(part["type"], "dynamic-tool");
    assert_eq!(part["title"], "Calculator");
    assert_eq!(part["toolName"], "calculate");
    assert_eq!(part["state"], "output-available");
    assert_eq!(part["input"], json!({ "a": 5, "b": 3 }));
    assert_eq!(part["output"], json!({ "result": 8 }));
}

#[test]
fn tool_with_title_in_error_state() {
    let state = run(json!([
        { "type": "start", "messageId": "msg-error" },
        { "type": "start-step" },
        { "type": "tool-input-start", "toolCallId": "tool-call-error", "toolName": "errorTool", "title": "Error Tool" },
        { "type": "tool-input-available", "toolCallId": "tool-call-error", "toolName": "errorTool", "input": { "invalid": "data" }, "title": "Error Tool" },
        { "type": "tool-output-error", "toolCallId": "tool-call-error", "errorText": "Tool execution failed" },
        { "type": "finish-step" },
        { "type": "finish" },
    ]));
    let part = tool_json(&state, "tool-call-error");
    assert_eq!(part["type"], "tool-errorTool");
    assert_eq!(part["title"], "Error Tool");
    assert_eq!(part["state"], "output-error");
    assert_eq!(part["errorText"], "Tool execution failed");
}

// tool approvals

fn approval_chunks(request: JsonValue, response: Option<JsonValue>) -> JsonValue {
    let mut chunks = vec![
        json!({ "type": "start" }),
        json!({ "type": "start-step" }),
        json!({ "input": { "value": "value" }, "toolCallId": "call-1", "toolName": "tool1", "type": "tool-input-available" }),
        request,
    ];
    chunks.extend(response);
    chunks.extend([
        json!({ "type": "finish-step" }),
        json!({ "type": "finish" }),
    ]);
    JsonValue::Array(chunks)
}

#[test]
fn tool_approval_request_with_signature_propagates_request_details_into_the_approval_object() {
    let state = run(approval_chunks(
        json!({ "approvalId": "id-1", "toolCallId": "call-1", "type": "tool-approval-request", "reason": "requires operator review", "signature": "test-sig" }),
        None,
    ));
    let part = tool_json(&state, "call-1");
    assert_eq!(part["state"], "approval-requested");
    assert_eq!(
        part["approval"],
        json!({ "id": "id-1", "requestReason": "requires operator review", "signature": "test-sig" })
    );
}

#[test]
fn tool_approval_response_with_signature_preserves_request_details_separately_from_the_response_reason()
 {
    let state = run(approval_chunks(
        json!({ "approvalId": "id-1", "toolCallId": "call-1", "type": "tool-approval-request", "reason": "requires operator review", "signature": "test-sig" }),
        Some(
            json!({ "approvalId": "id-1", "approved": true, "reason": "approved by operator", "type": "tool-approval-response" }),
        ),
    ));
    let part = tool_json(&state, "call-1");
    assert_eq!(part["state"], "approval-responded");
    assert_eq!(
        part["approval"],
        json!({
            "id": "id-1",
            "approved": true,
            "requestReason": "requires operator review",
            "reason": "approved by operator",
            "signature": "test-sig",
        })
    );
}

fn descriptor() -> JsonValue {
    json!({ "action": "deleteAccount", "permissions": ["account:delete"], "risk": "high" })
}

#[test]
fn preserves_approval_descriptors_through_request_and_response_states() {
    let approvals: Vec<_> = part_after_each_chunk(
        json!([
            { "input": { "userId": "user-123" }, "toolCallId": "call-1", "toolName": "deleteAccount", "type": "tool-input-available" },
            { "approvalDescriptor": descriptor(), "approvalId": "approval-1", "toolCallId": "call-1", "type": "tool-approval-request" },
            { "approvalId": "approval-1", "approved": true, "type": "tool-approval-response" },
        ]),
        0,
    )
    .into_iter()
    .filter_map(|part| part.get("approval").cloned())
    .collect();
    assert_eq!(
        approvals,
        vec![
            json!({ "id": "approval-1", "descriptor": descriptor() }),
            json!({ "id": "approval-1", "approved": true, "descriptor": descriptor() }),
        ]
    );
}

#[test]
fn preserves_an_approval_descriptor_restored_from_a_persisted_message() {
    // the approval is requested on one connection and answered on another
    let (state, _) = run_chunks(
        &json!({
            "messageId": "msg-123",
            "lastMessage": {
                "role": "assistant",
                "id": "msg-123",
                "parts": [{
                    "type": "tool-deleteAccount",
                    "toolCallId": "call-1",
                    "state": "approval-requested",
                    "input": { "userId": "user-123" },
                    "approval": { "id": "approval-1", "descriptor": descriptor() },
                }],
            },
        }),
        &json!([{ "approvalId": "approval-1", "approved": true, "type": "tool-approval-response" }]),
    );
    assert_eq!(
        tool_json(&state, "call-1")["approval"],
        json!({ "id": "approval-1", "approved": true, "descriptor": descriptor() })
    );
}

#[test]
fn tool_approval_request_without_signature_does_not_include_signature_in_the_approval_object() {
    let state = run(approval_chunks(
        json!({ "approvalId": "id-1", "toolCallId": "call-1", "type": "tool-approval-request" }),
        None,
    ));
    let part = tool_json(&state, "call-1");
    assert_eq!(part["state"], "approval-requested");
    assert_eq!(part["approval"], json!({ "id": "id-1" }));
}

#[test]
fn automatic_tool_approval_denial_static_tool_keeps_automatic_approval_metadata_through_denial() {
    let mut chunks = approval_chunks(
        json!({ "approvalId": "id-1", "isAutomatic": true, "toolCallId": "call-1", "type": "tool-approval-request" }),
        Some(
            json!({ "approvalId": "id-1", "approved": false, "reason": "Policy denied execution", "type": "tool-approval-response" }),
        ),
    );
    chunks.as_array_mut().unwrap().insert(
        5,
        json!({ "toolCallId": "call-1", "type": "tool-output-denied" }),
    );
    let base =
        json!({ "input": { "value": "value" }, "toolCallId": "call-1", "type": "tool-tool1" });
    let with = |state: &str, approval: Option<JsonValue>| {
        let mut part = base.clone();
        part["state"] = json!(state);
        if let Some(approval) = approval {
            part["approval"] = approval;
        }
        part
    };
    let denied = json!({ "approved": false, "id": "id-1", "isAutomatic": true, "reason": "Policy denied execution" });
    let snapshots = part_after_each_chunk(chunks.clone(), 1);
    assert_eq!(
        snapshots,
        vec![
            with("input-available", None),
            with(
                "approval-requested",
                Some(json!({ "id": "id-1", "isAutomatic": true }))
            ),
            with("approval-responded", Some(denied.clone())),
            with("output-denied", Some(denied.clone())),
        ]
    );
    assert_eq!(parts(&run(chunks))[1], with("output-denied", Some(denied)));
}

#[test]
fn automatic_tool_approval_with_execution_dynamic_tool_keeps_automatic_approval_metadata_through_execution()
 {
    let chunks = json!([
        { "type": "start" },
        { "type": "start-step" },
        { "dynamic": true, "input": { "value": "value" }, "toolCallId": "call-1", "toolName": "tool1", "type": "tool-input-available" },
        { "approvalId": "id-1", "isAutomatic": true, "toolCallId": "call-1", "type": "tool-approval-request" },
        { "approvalId": "id-1", "approved": true, "reason": "trusted internal tool", "type": "tool-approval-response" },
        { "output": "result1", "toolCallId": "call-1", "type": "tool-output-available" },
        { "type": "finish-step" },
        { "type": "finish" },
    ]);
    let base = json!({ "input": { "value": "value" }, "toolCallId": "call-1", "toolName": "tool1", "type": "dynamic-tool" });
    let with = |state: &str, approval: Option<JsonValue>, output: Option<&str>| {
        let mut part = base.clone();
        part["state"] = json!(state);
        if let Some(approval) = approval {
            part["approval"] = approval;
        }
        if let Some(output) = output {
            part["output"] = json!(output);
        }
        part
    };
    let approved = json!({ "approved": true, "id": "id-1", "isAutomatic": true, "reason": "trusted internal tool" });
    assert_eq!(
        part_after_each_chunk(chunks.clone(), 1),
        vec![
            with("input-available", None, None),
            with(
                "approval-requested",
                Some(json!({ "id": "id-1", "isAutomatic": true })),
                None
            ),
            with("approval-responded", Some(approved.clone()), None),
            with("output-available", Some(approved.clone()), Some("result1")),
        ]
    );
    assert_eq!(
        parts(&run(chunks))[1],
        with("output-available", Some(approved), Some("result1"))
    );
}
