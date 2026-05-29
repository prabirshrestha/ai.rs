use std::sync::OnceLock;

pub fn ensure_builtins_registered() {
    static REGISTERED: OnceLock<()> = OnceLock::new();
    REGISTERED.get_or_init(register_builtins);
}

pub fn register_builtins() {}
