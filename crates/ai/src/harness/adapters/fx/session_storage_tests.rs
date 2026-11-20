//! Port of `fx-session-storage.test.ts`.

use std::sync::{Arc, Mutex as StdMutex};

use serde_json::{Value, json};

use super::*;

#[derive(Default)]
struct MemoryStorage {
    entries: StdMutex<Vec<FxSessionStorageEntry>>,
    raw: Option<Vec<Value>>,
    received_token: StdMutex<bool>,
}

#[async_trait]
impl FxSessionStorage for MemoryStorage {
    async fn append(&self, _session_id: &str, entry: FxSessionStorageEntry) -> HarnessResult<()> {
        self.entries.lock().unwrap().push(entry);
        Ok(())
    }

    async fn read(
        &self,
        _session_id: &str,
        _leaf_id: &str,
        cancellation_token: Option<CancellationToken>,
    ) -> HarnessResult<Vec<Value>> {
        *self.received_token.lock().unwrap() = cancellation_token.is_some();
        if let Some(raw) = &self.raw {
            return Ok(raw.clone());
        }
        Ok(self
            .entries
            .lock()
            .unwrap()
            .iter()
            .map(|entry| serde_json::to_value(entry).unwrap())
            .collect())
    }
}

fn turn() -> Value {
    json!({ "id": "turn-1", "prompt": "go", "signature": "signature", "models": [], "tools": [], "emittedKeys": [], "steers": [] })
}

fn state(checkpoint: Option<&str>, turn: Option<Value>) -> FxSessionState {
    FxSessionState {
        provider: FxProvider::Gateway,
        checkpoint: checkpoint.map(str::to_string),
        turn,
    }
}

fn entries(storage: &MemoryStorage) -> Vec<Value> {
    storage
        .entries
        .lock()
        .unwrap()
        .iter()
        .map(|entry| serde_json::to_value(entry).unwrap())
        .collect()
}

#[tokio::test]
async fn appends_the_whole_state_as_the_first_entry() {
    let storage = Arc::new(MemoryStorage::default());
    let writer = FxSessionStorageWriter::new(storage.clone(), "session-1", None, None);
    let leaf_id = writer.write(&state(Some("AQID"), Some(turn()))).await.unwrap();
    assert_eq!(
        entries(&storage),
        vec![json!({ "entryId": leaf_id, "data": { "provider": "gateway", "checkpoint": "AQID", "turn": turn() } })]
    );
}

#[tokio::test]
async fn appends_only_the_changed_fields_linked_to_the_previous_entry() {
    let storage = Arc::new(MemoryStorage::default());
    let writer = FxSessionStorageWriter::new(storage.clone(), "session-1", None, None);
    let first = writer.write(&state(Some("AQID"), Some(turn()))).await.unwrap();
    let mut changed = turn();
    changed["emittedKeys"] = json!(["start"]);
    let second = writer.write(&state(Some("AQID"), Some(changed.clone()))).await.unwrap();
    assert_eq!(
        entries(&storage)[1],
        json!({ "entryId": second, "parentEntryId": first, "data": { "turn": changed } })
    );
}

#[tokio::test]
async fn records_an_ended_turn_and_a_new_checkpoint() {
    let storage = Arc::new(MemoryStorage::default());
    let writer = FxSessionStorageWriter::new(storage.clone(), "session-1", None, None);
    writer.write(&state(Some("AQID"), Some(turn()))).await.unwrap();
    let leaf_id = writer.write(&state(Some("BAUG"), None)).await.unwrap();
    assert_eq!(entries(&storage)[1]["data"], json!({ "checkpoint": "BAUG", "turn": null }));
    assert_eq!(
        read_fx_session_state(storage.as_ref(), "session-1", &leaf_id, None)
            .await
            .unwrap(),
        state(Some("BAUG"), None)
    );
}

#[tokio::test]
async fn appends_nothing_when_the_state_did_not_change() {
    let storage = Arc::new(MemoryStorage::default());
    let writer = FxSessionStorageWriter::new(storage.clone(), "session-1", None, None);
    let first = writer.write(&state(Some("AQID"), None)).await.unwrap();
    assert_eq!(writer.write(&state(Some("AQID"), None)).await.unwrap(), first);
    assert_eq!(entries(&storage).len(), 1);
}

#[tokio::test]
async fn continues_from_a_restored_leaf() {
    let storage = Arc::new(MemoryStorage::default());
    let saved = state(Some("AQID"), None);
    let writer = FxSessionStorageWriter::new(storage.clone(), "session-1", Some("restored".into()), Some(&saved));
    assert_eq!(writer.write(&saved).await.unwrap(), "restored");
    writer.write(&state(Some("AQID"), Some(turn()))).await.unwrap();
    let written = entries(&storage);
    assert_eq!(written.len(), 1);
    assert_eq!(written[0]["parentEntryId"], "restored");
    assert_eq!(written[0]["data"], json!({ "turn": turn() }));
}

#[tokio::test]
async fn folds_only_the_entries_leading_to_the_leaf_in_any_order() {
    let storage = MemoryStorage {
        raw: Some(vec![
            json!({ "entryId": "b", "parentEntryId": "a", "data": { "checkpoint": "Ag==" } }),
            json!({ "entryId": "other", "parentEntryId": "a", "data": { "checkpoint": "Aw==" } }),
            json!({ "entryId": "a", "data": { "provider": "codex", "checkpoint": "AQ==" } }),
        ]),
        ..Default::default()
    };
    assert_eq!(
        read_fx_session_state(&storage, "session-1", "b", Some(CancellationToken::new()))
            .await
            .unwrap(),
        FxSessionState {
            provider: FxProvider::Codex,
            checkpoint: Some("Ag==".into()),
            turn: None,
        }
    );
    assert!(*storage.received_token.lock().unwrap());
}

#[tokio::test]
async fn fails_on_invalid_chains() {
    let cases = [
        (
            vec![json!({ "entryId": "a", "data": { "provider": "unknown" } })],
            "a",
            "Entry 0 of the fx session \"session-1\" is not an fx session entry.",
        ),
        (
            vec![json!({ "entryId": "a", "data": { "provider": "gateway" } })],
            "missing",
            "The session storage has no entry \"missing\" for the fx session \"session-1\".",
        ),
        (
            vec![json!({ "entryId": "a", "parentEntryId": "gone", "data": {} })],
            "a",
            "The session storage has no entry \"gone\" for the fx session \"session-1\".",
        ),
        (
            vec![
                json!({ "entryId": "a", "parentEntryId": "b", "data": {} }),
                json!({ "entryId": "b", "parentEntryId": "a", "data": {} }),
            ],
            "a",
            "The entries of the fx session \"session-1\" form a loop at entry \"a\".",
        ),
        (
            vec![json!({ "entryId": "a", "data": { "checkpoint": "AQ==" } })],
            "a",
            "The entries leading to \"a\" do not form an fx session for \"session-1\".",
        ),
    ];
    for (raw, leaf_id, error) in cases {
        let storage = MemoryStorage {
            raw: Some(raw),
            ..Default::default()
        };
        let result = read_fx_session_state(&storage, "session-1", leaf_id, None).await;
        assert_eq!(result.unwrap_err().to_string(), error);
    }
}
