use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use crate::{Error, Result};

pub type SessionResourceCleanup = Arc<dyn Fn(Option<&str>) -> Result<()> + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionResourceCleanupRegistration {
    id: u64,
}

impl SessionResourceCleanupRegistration {
    pub fn unregister(self) {
        unregister_session_resource_cleanup(self.id);
    }
}

struct RegisteredSessionResourceCleanup {
    id: u64,
    cleanup: SessionResourceCleanup,
}

fn registry() -> &'static RwLock<Vec<RegisteredSessionResourceCleanup>> {
    static REGISTRY: OnceLock<RwLock<Vec<RegisteredSessionResourceCleanup>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(Vec::new()))
}

pub fn register_session_resource_cleanup<F>(cleanup: F) -> SessionResourceCleanupRegistration
where
    F: Fn(Option<&str>) -> Result<()> + Send + Sync + 'static,
{
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    registry()
        .write()
        .expect("session resource cleanup registry poisoned")
        .push(RegisteredSessionResourceCleanup {
            id,
            cleanup: Arc::new(cleanup),
        });
    SessionResourceCleanupRegistration { id }
}

pub fn cleanup_session_resources(session_id: Option<&str>) -> Result<()> {
    let cleanups = registry()
        .read()
        .expect("session resource cleanup registry poisoned")
        .iter()
        .map(|entry| entry.cleanup.clone())
        .collect::<Vec<_>>();
    let errors = cleanups
        .into_iter()
        .filter_map(|cleanup| cleanup(session_id).err())
        .map(|error| error.to_string())
        .collect::<Vec<_>>();

    if errors.is_empty() {
        Ok(())
    } else {
        Err(Error::Provider(format!(
            "Failed to cleanup session resources: {}",
            errors.join("; ")
        )))
    }
}

fn unregister_session_resource_cleanup(id: u64) {
    registry()
        .write()
        .expect("session resource cleanup registry poisoned")
        .retain(|entry| entry.id != id);
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn clear_registry_for_test() {
        registry()
            .write()
            .expect("session resource cleanup registry poisoned")
            .clear();
    }

    #[test]
    fn invokes_registered_cleanups_with_session_id() {
        let _guard = TEST_LOCK.lock().expect("test lock poisoned");
        clear_registry_for_test();
        let seen = Arc::new(RwLock::new(Vec::<Option<String>>::new()));
        let seen_clone = seen.clone();
        let registration = register_session_resource_cleanup(move |session_id| {
            seen_clone
                .write()
                .expect("seen lock poisoned")
                .push(session_id.map(ToString::to_string));
            Ok(())
        });

        cleanup_session_resources(Some("session-1")).expect("cleanup");
        registration.unregister();

        assert_eq!(
            *seen.read().expect("seen lock poisoned"),
            vec![Some("session-1".to_string())]
        );
    }

    #[test]
    fn unregister_removes_cleanup() {
        let _guard = TEST_LOCK.lock().expect("test lock poisoned");
        clear_registry_for_test();
        let call_count = Arc::new(AtomicU64::new(0));
        let call_count_clone = call_count.clone();
        let registration = register_session_resource_cleanup(move |_session_id| {
            call_count_clone.fetch_add(1, Ordering::Relaxed);
            Ok(())
        });
        registration.unregister();

        cleanup_session_resources(None).expect("cleanup");

        assert_eq!(call_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn aggregates_cleanup_errors_after_running_all_callbacks() {
        let _guard = TEST_LOCK.lock().expect("test lock poisoned");
        clear_registry_for_test();
        let call_count = Arc::new(AtomicU64::new(0));
        let call_count_clone = call_count.clone();
        let first = register_session_resource_cleanup(|_session_id| {
            Err(Error::Provider("first".to_string()))
        });
        let second = register_session_resource_cleanup(move |_session_id| {
            call_count_clone.fetch_add(1, Ordering::Relaxed);
            Err(Error::Provider("second".to_string()))
        });

        let error = cleanup_session_resources(None).expect_err("cleanup should aggregate errors");

        assert_eq!(call_count.load(Ordering::Relaxed), 1);
        assert!(
            matches!(error, Error::Provider(message) if message.contains("first") && message.contains("second"))
        );
        first.unregister();
        second.unregister();
    }
}
