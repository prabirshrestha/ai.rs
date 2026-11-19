//! Port of `utils/classify-disk-log.ts`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// `DiskLogRecoveryMode`: recovery rung selected from an on-disk bridge
/// event log when attach is not possible (the bridge process is gone).
/// `Replay` when the log holds a complete turn the host can resume from its
/// cursor, `Rerun` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DiskLogRecoveryMode {
    Replay,
    Rerun,
}

/// `classifyDiskLog(eventLog)`: whether a respawned bridge can replay a turn
/// from its persisted `event-log.ndjson`, or must rerun it from scratch.
///
/// A turn is replayable only when its log ends in a terminal `finish` event.
/// A log that is missing, empty, or ends mid-turn means the bridge died
/// before completing the turn.
pub fn classify_disk_log(event_log: Option<&str>) -> DiskLogRecoveryMode {
    let Some(last_line) = event_log
        .unwrap_or_default()
        .split('\n')
        .map(str::trim)
        .rfind(|line| !line.is_empty())
    else {
        return DiskLogRecoveryMode::Rerun;
    };
    match serde_json::from_str::<Value>(last_line) {
        Ok(Value::Object(event)) if event.get("type") == Some(&Value::from("finish")) => {
            DiskLogRecoveryMode::Replay
        }
        _ => DiskLogRecoveryMode::Rerun,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn line(value: Value) -> String {
        value.to_string()
    }

    #[test]
    fn returns_rerun_for_absent_logs() {
        for log in [None, Some(""), Some("   \n  \n")] {
            assert_eq!(classify_disk_log(log), DiskLogRecoveryMode::Rerun);
        }
    }

    #[test]
    fn returns_replay_when_the_last_event_is_a_finish() {
        let log = [
            line(json!({ "type": "text-delta", "id": "m", "delta": "hi", "seq": 1 })),
            line(json!({ "type": "finish", "seq": 2 })),
        ]
        .join("\n");
        assert_eq!(classify_disk_log(Some(&log)), DiskLogRecoveryMode::Replay);
    }

    #[test]
    fn tolerates_a_trailing_newline() {
        let log = format!("{}\n", line(json!({ "type": "finish", "seq": 1 })));
        assert_eq!(classify_disk_log(Some(&log)), DiskLogRecoveryMode::Replay);
    }

    #[test]
    fn returns_rerun_when_the_turn_ended_mid_stream() {
        let log = [
            line(json!({ "type": "text-delta", "id": "m", "delta": "one", "seq": 1 })),
            line(json!({ "type": "text-delta", "id": "m", "delta": "two", "seq": 2 })),
        ]
        .join("\n");
        assert_eq!(classify_disk_log(Some(&log)), DiskLogRecoveryMode::Rerun);
    }

    #[test]
    fn returns_rerun_when_the_last_line_is_a_non_finish_event() {
        let log = [
            line(json!({ "type": "finish", "seq": 1 })),
            line(json!({ "type": "text-delta", "id": "m", "delta": "late", "seq": 2 })),
        ]
        .join("\n");
        assert_eq!(classify_disk_log(Some(&log)), DiskLogRecoveryMode::Rerun);
    }

    #[test]
    fn returns_rerun_when_the_last_line_is_corrupt() {
        let log = format!(
            "{}\n{{not json",
            line(json!({ "type": "finish", "seq": 1 }))
        );
        assert_eq!(classify_disk_log(Some(&log)), DiskLogRecoveryMode::Rerun);
    }
}
