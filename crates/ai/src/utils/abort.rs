//! Port of `utils/abort.ts`. `AbortSignal` is a [`CancellationToken`].

use std::future::Future;

use tokio_util::sync::CancellationToken;

use crate::{Error, Result};

/// Create an operation-local signal for public APIs whose signal is optional.
pub fn operation_signal(signal: Option<&CancellationToken>) -> CancellationToken {
    signal.cloned().unwrap_or_default()
}

/// Stop waiting for an operation when its signal aborts.
///
/// Pi keeps observing the abandoned promise; in Rust the operation future is
/// dropped (cancelled) instead, which is how Rust futures stop.
pub async fn race_with_abort_signal<T>(
    operation: impl Future<Output = Result<T>>,
    signal: &CancellationToken,
) -> Result<T> {
    if signal.is_cancelled() {
        return Err(Error::aborted());
    }
    tokio::select! {
        _ = signal.cancelled() => Err(Error::aborted()),
        result = operation => result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn rejects_when_already_aborted_and_resolves_otherwise() {
        let signal = CancellationToken::new();
        assert_eq!(
            race_with_abort_signal(async { Ok(1) }, &signal)
                .await
                .unwrap(),
            1
        );
        signal.cancel();
        let error = race_with_abort_signal(async { Ok(1) }, &signal)
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "The operation was aborted");
        assert!(!operation_signal(None).is_cancelled());
        assert!(operation_signal(Some(&signal)).is_cancelled());
    }

    #[tokio::test]
    async fn rejects_a_pending_operation_when_the_signal_aborts() {
        let signal = CancellationToken::new();
        let abort = signal.clone();
        tokio::spawn(async move { abort.cancel() });
        let error = race_with_abort_signal(std::future::pending::<Result<()>>(), &signal)
            .await
            .unwrap_err();
        assert!(error.is_abort());
    }
}
