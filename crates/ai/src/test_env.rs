use std::sync::{Mutex, MutexGuard};

static ENV_LOCK: Mutex<()> = Mutex::new(());

pub(crate) struct EnvVarGuard {
    key: &'static str,
    value: Option<String>,
    _guard: MutexGuard<'static, ()>,
}

impl EnvVarGuard {
    pub(crate) fn set(key: &'static str, value: &str) -> Self {
        let guard = ENV_LOCK.lock().expect("env lock poisoned");
        let original = std::env::var(key).ok();
        unsafe {
            std::env::set_var(key, value);
        }

        Self {
            key,
            value: original,
            _guard: guard,
        }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(value) = &self.value {
                std::env::set_var(self.key, value);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }
}
