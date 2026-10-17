//! Port of `utils/abort-signals.ts`.

use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// A signal that aborts when any source signal aborts. Dropping it (or
/// calling `cleanup`) stops forwarding.
pub struct CombinedAbortSignal {
    pub signal: Option<CancellationToken>,
    forwarder: Option<JoinHandle<()>>,
}

impl CombinedAbortSignal {
    pub fn cleanup(&mut self) {
        if let Some(forwarder) = self.forwarder.take() {
            forwarder.abort();
        }
    }
}

impl Drop for CombinedAbortSignal {
    fn drop(&mut self) {
        self.cleanup();
    }
}

/// Combine optional signals. With more than one active signal a forwarding
/// task is spawned on the current Tokio runtime (Pi adds abort listeners).
pub fn combine_abort_signals(signals: &[Option<&CancellationToken>]) -> CombinedAbortSignal {
    let active: Vec<CancellationToken> = signals.iter().flatten().map(|s| (*s).clone()).collect();
    match active.len() {
        0 => CombinedAbortSignal {
            signal: None,
            forwarder: None,
        },
        1 => CombinedAbortSignal {
            signal: active.into_iter().next(),
            forwarder: None,
        },
        _ => {
            let combined = CancellationToken::new();
            if active.iter().any(CancellationToken::is_cancelled) {
                combined.cancel();
                return CombinedAbortSignal {
                    signal: Some(combined),
                    forwarder: None,
                };
            }
            let target = combined.clone();
            let forwarder = tokio::spawn(async move {
                let waits = active
                    .iter()
                    .map(|signal| Box::pin(signal.cancelled()))
                    .collect::<Vec<_>>();
                futures::future::select_all(waits).await;
                target.cancel();
            });
            CombinedAbortSignal {
                signal: Some(combined),
                forwarder: Some(forwarder),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn combines_zero_one_and_many_signals() {
        assert!(combine_abort_signals(&[None]).signal.is_none());
        let first = CancellationToken::new();
        let only = combine_abort_signals(&[None, Some(&first)]);
        first.cancel();
        assert!(only.signal.as_ref().unwrap().is_cancelled());

        let a = CancellationToken::new();
        let b = CancellationToken::new();
        let combined = combine_abort_signals(&[Some(&a), Some(&b)]);
        let signal = combined.signal.clone().unwrap();
        assert!(!signal.is_cancelled());
        b.cancel();
        signal.cancelled().await;
        assert!(!a.is_cancelled());
    }
}
