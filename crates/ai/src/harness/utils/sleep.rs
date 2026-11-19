//! Port of `sleep.ts`.

use std::time::Duration;

use tokio_util::sync::CancellationToken;

/// `sleep({ ms, abortSignal })`: resolves after `ms`, or as soon as the
/// signal fires (never rejects).
pub async fn sleep(ms: u64, abort_signal: Option<&CancellationToken>) {
    match abort_signal {
        Some(signal) => {
            if signal.is_cancelled() {
                return;
            }
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_millis(ms)) => {}
                _ = signal.cancelled() => {}
            }
        }
        None => tokio::time::sleep(Duration::from_millis(ms)).await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // sleep.test.ts
    #[tokio::test]
    async fn resolves_when_the_abort_signal_fires() {
        let signal = CancellationToken::new();
        let s = signal.clone();
        let pending = tokio::spawn(async move { sleep(1_000_000, Some(&s)).await });
        signal.cancel();
        tokio::time::timeout(Duration::from_secs(5), pending)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn resolves_immediately_for_an_already_aborted_signal() {
        let signal = CancellationToken::new();
        signal.cancel();
        tokio::time::timeout(Duration::from_secs(5), sleep(1_000_000, Some(&signal)))
            .await
            .unwrap();
    }
}
