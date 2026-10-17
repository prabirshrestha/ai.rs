//! Port of `utils/sleep.ts`.

use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::{Error, Result};

/// Sleep for `ms`, rejecting with an abort error when `signal` aborts.
pub async fn sleep(ms: u64, signal: &CancellationToken) -> Result<()> {
    if signal.is_cancelled() {
        return Err(Error::aborted());
    }
    tokio::select! {
        _ = signal.cancelled() => Err(Error::aborted()),
        _ = tokio::time::sleep(Duration::from_millis(ms)) => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn resolves_after_the_delay_and_rejects_when_aborted() {
        let signal = CancellationToken::new();
        sleep(10, &signal).await.unwrap();
        signal.cancel();
        assert!(sleep(10, &signal).await.unwrap_err().is_abort());
    }
}
