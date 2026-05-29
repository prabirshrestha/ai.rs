use std::sync::OnceLock;

use crate::images_api_registry::{self, ImagesApiProvider};
use crate::providers::images::openrouter;

pub fn ensure_builtins_registered() {
    static REGISTERED: OnceLock<()> = OnceLock::new();
    REGISTERED.get_or_init(register_builtins);
}

pub fn register_builtins() {
    images_api_registry::register_images_api_provider(
        ImagesApiProvider {
            api: "openrouter-images".to_string(),
            generate_images: images_api_registry::wrap_generate_images(
                "openrouter-images",
                openrouter::generate_images_openrouter,
            ),
        },
        Some("builtin".to_string()),
    );
}
