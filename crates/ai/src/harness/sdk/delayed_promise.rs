//! Port of `@ai-sdk/provider-utils` `DelayedPromise`.
//!
//! [rust] The promise settles once: the first `resolve` or `reject` wins,
//! which is what a JS promise does once someone has read `.promise`.

use tokio::sync::watch;

use crate::harness::sdk::error_value::SharedError;

type Status<T> = Option<Result<T, SharedError>>;

/// `DelayedPromise<T>`.
pub struct DelayedPromise<T: Clone + Send + Sync> {
    tx: watch::Sender<Status<T>>,
}

impl<T: Clone + Send + Sync> Default for DelayedPromise<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + Send + Sync> DelayedPromise<T> {
    pub fn new() -> Self {
        Self {
            tx: watch::channel(None).0,
        }
    }

    /// `resolve(value)`.
    pub fn resolve(&self, value: T) {
        self.tx.send_if_modified(|s| {
            if s.is_none() {
                *s = Some(Ok(value));
                true
            } else {
                false
            }
        });
    }

    /// `reject(error)`.
    pub fn reject(&self, error: SharedError) {
        self.tx.send_if_modified(|s| {
            if s.is_none() {
                *s = Some(Err(error));
                true
            } else {
                false
            }
        });
    }

    /// `promise`: wait for the settled value.
    pub fn promise(
        &self,
    ) -> impl std::future::Future<Output = Result<T, SharedError>> + Send + 'static
    where
        T: 'static,
    {
        let mut rx = self.tx.subscribe();
        async move {
            loop {
                if let Some(status) = rx.borrow_and_update().clone() {
                    return status;
                }
                if rx.changed().await.is_err() {
                    // The promise was dropped unsettled; it never settles.
                    std::future::pending::<()>().await;
                }
            }
        }
    }

    pub fn is_resolved(&self) -> bool {
        matches!(*self.tx.borrow(), Some(Ok(_)))
    }

    pub fn is_rejected(&self) -> bool {
        matches!(*self.tx.borrow(), Some(Err(_)))
    }

    pub fn is_pending(&self) -> bool {
        self.tx.borrow().is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // delayed-promise.test.ts
    #[tokio::test]
    async fn resolves_when_accessed_after_resolution() {
        let p = DelayedPromise::new();
        p.resolve(1);
        assert_eq!(p.promise().await.unwrap(), 1);
        assert!(p.is_resolved());
    }

    #[tokio::test]
    async fn rejects_when_accessed_after_rejection() {
        let p: DelayedPromise<u32> = DelayedPromise::new();
        p.reject(Arc::new(std::io::Error::other("boom")));
        assert_eq!(p.promise().await.unwrap_err().to_string(), "boom");
        assert!(p.is_rejected());
    }

    #[tokio::test]
    async fn resolves_when_accessed_before_resolution() {
        let p = Arc::new(DelayedPromise::new());
        let fut = p.promise();
        assert!(p.is_pending());
        p.resolve("x");
        assert_eq!(fut.await.unwrap(), "x");
    }

    #[tokio::test]
    async fn keeps_the_first_settlement() {
        let p = DelayedPromise::new();
        p.resolve(1);
        p.reject(Arc::new(std::io::Error::other("late")));
        p.resolve(2);
        assert_eq!(p.promise().await.unwrap(), 1);
    }
}
