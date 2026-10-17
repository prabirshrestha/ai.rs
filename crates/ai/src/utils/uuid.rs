//! Port of `utils/uuid.ts`.

use parking_lot::Mutex;
use ring::rand::{SecureRandom, SystemRandom};

const MAX_UUID_V7_TIMESTAMP: u64 = 0xffff_ffff_ffff;
const MAX_SEQUENCE: u64 = (1 << 41) - 1;

struct GeneratorState {
    last_ordinary_timestamp: Option<u64>,
    sequence: Option<u64>,
}

static STATE: Mutex<GeneratorState> = Mutex::new(GeneratorState {
    last_ordinary_timestamp: None,
    sequence: None,
});

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct UuidRangeError(String);

/// Generate a time-ordered UUIDv7. A supplied timestamp is preserved for
/// follower ids.
pub fn uuidv7(timestamp_ms: Option<u64>) -> Result<String, UuidRangeError> {
    uuidv7_with(
        &STATE,
        timestamp_ms,
        crate::utils::time::now_millis(),
        &mut |bytes| {
            let _ = SystemRandom::new().fill(bytes);
        },
    )
}

fn uuidv7_with(
    state: &Mutex<GeneratorState>,
    timestamp_ms: Option<u64>,
    now: u64,
    random: &mut dyn FnMut(&mut [u8]),
) -> Result<String, UuidRangeError> {
    let requested_timestamp = timestamp_ms.unwrap_or(now);
    if requested_timestamp > MAX_UUID_V7_TIMESTAMP {
        return Err(UuidRangeError(format!(
            "UUIDv7 timestamp must be an integer between 0 and {MAX_UUID_V7_TIMESTAMP}"
        )));
    }

    let mut state = state.lock();
    let effective_timestamp = match timestamp_ms {
        Some(timestamp) => timestamp,
        None => {
            let effective = state
                .last_ordinary_timestamp
                .map_or(requested_timestamp, |last| requested_timestamp.max(last));
            state.last_ordinary_timestamp = Some(effective);
            effective
        }
    };

    let mut bytes = [0u8; 16];
    random(&mut bytes);
    let sequence = match state.sequence {
        None => {
            (u64::from(bytes[1]) << 32)
                | (u64::from(bytes[2]) << 24)
                | (u64::from(bytes[3]) << 16)
                | (u64::from(bytes[4]) << 8)
                | u64::from(bytes[5])
        }
        Some(sequence) => {
            if sequence == MAX_SEQUENCE {
                return Err(UuidRangeError(
                    "UUIDv7 generator sequence exhausted".to_string(),
                ));
            }
            sequence + 1
        }
    };
    state.sequence = Some(sequence);
    drop(state);

    for index in (0..=5).rev() {
        bytes[index] = ((effective_timestamp >> ((5 - index) * 8)) & 0xff) as u8;
    }
    bytes[6] = 0x70 | ((sequence >> 37) & 0x0f) as u8;
    bytes[7] = ((sequence >> 29) & 0xff) as u8;
    bytes[8] = 0x80 | ((sequence >> 23) & 0x3f) as u8;
    bytes[9] = ((sequence >> 15) & 0xff) as u8;
    bytes[10] = ((sequence >> 7) & 0xff) as u8;
    bytes[11] = (((sequence & 0x7f) << 1) as u8) | (bytes[11] & 0x01);

    let hex: Vec<String> = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
    Ok(format!(
        "{}-{}-{}-{}-{}",
        hex[0..4].concat(),
        hex[4..6].concat(),
        hex[6..8].concat(),
        hex[8..10].concat(),
        hex[10..].concat()
    ))
}

#[cfg(test)]
mod tests {
    use regex::Regex;

    use super::*;

    const TIMESTAMP: u64 = 0x0123_4567_89ab;

    fn uuid_v7_re() -> Regex {
        Regex::new(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
            .unwrap()
    }

    fn parse_timestamp(uuid: &str) -> u64 {
        u64::from_str_radix(&uuid.replace('-', "")[..12], 16).unwrap()
    }

    fn system_random(bytes: &mut [u8]) {
        SystemRandom::new().fill(bytes).unwrap();
    }

    #[test]
    fn generates_ordered_uuidv7s_while_preserving_follower_timestamps() {
        let state = Mutex::new(GeneratorState {
            last_ordinary_timestamp: None,
            sequence: None,
        });
        let state = &state;
        let mut random = system_random;
        let first = uuidv7_with(state, None, TIMESTAMP, &mut random).unwrap();
        let second = uuidv7_with(state, None, TIMESTAMP, &mut random).unwrap();
        let after_rollback = uuidv7_with(state, None, TIMESTAMP - 1, &mut random).unwrap();
        let after_advance = uuidv7_with(state, None, TIMESTAMP + 1, &mut random).unwrap();
        let ordinary = vec![first, second, after_rollback, after_advance];
        let follower_timestamp = TIMESTAMP - 1_000;
        let followers = vec![
            uuidv7_with(state, Some(follower_timestamp), TIMESTAMP + 1, &mut random).unwrap(),
            uuidv7_with(state, Some(follower_timestamp), TIMESTAMP + 1, &mut random).unwrap(),
        ];

        for id in ordinary.iter().chain(&followers) {
            assert!(uuid_v7_re().is_match(id), "{id}");
        }
        let mut sorted = ordinary.clone();
        sorted.sort();
        assert_eq!(ordinary, sorted);
        assert_eq!(
            ordinary
                .iter()
                .map(|id| parse_timestamp(id))
                .collect::<Vec<_>>(),
            vec![TIMESTAMP, TIMESTAMP, TIMESTAMP, TIMESTAMP + 1]
        );
        assert_eq!(
            followers
                .iter()
                .map(|id| parse_timestamp(id))
                .collect::<Vec<_>>(),
            vec![follower_timestamp, follower_timestamp]
        );
        assert_ne!(followers[0], followers[1]);

        // Uses fresh randomness for every UUID tail.
        let mut random_byte = 0u8;
        let mut fill = |bytes: &mut [u8]| {
            random_byte += 1;
            bytes.fill(random_byte);
        };
        let tails = [
            uuidv7_with(state, Some(TIMESTAMP), 0, &mut fill).unwrap()[28..].to_string(),
            uuidv7_with(state, Some(TIMESTAMP), 0, &mut fill).unwrap()[28..].to_string(),
        ];
        assert_eq!(tails, ["01010101", "02020202"]);

        // Timestamp boundaries.
        for timestamp in [0, (1u64 << 48) - 1] {
            assert_eq!(
                parse_timestamp(&uuidv7(Some(timestamp)).unwrap()),
                timestamp
            );
        }
        assert!(uuidv7(Some(1 << 48)).is_err());
    }
}
