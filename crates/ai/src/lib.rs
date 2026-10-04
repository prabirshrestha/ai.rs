//! Rust port of Pi's `@earendil-works/pi-ai` 1.0 (`packages/ai`).
//!
//! The crate root mirrors Pi's `index.ts`: core types plus the side-effect
//! free utilities. Divergences from Pi are documented on the items involved.

pub mod error;
pub mod types;
pub mod utils;

pub use error::{Error, Result};
pub use types::*;
pub use utils::assistant_message_frame::*;
pub use utils::diagnostics::*;
pub use utils::event_stream::*;
pub use utils::json_parse::*;
pub use utils::overflow::*;
pub use utils::retry::*;
pub use utils::text::{content_text, get_system_message_text, render_system_message_update};
pub use utils::transcript::*;
pub use utils::uuid::uuidv7;
pub use utils::validation::*;
