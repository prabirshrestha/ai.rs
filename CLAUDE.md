# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Building and Testing
- `cargo build` - Build all workspace members (library and examples)
- `cargo test` - Run all tests across workspace
- `cargo test --package ai` - Run tests for just the ai crate
- `cargo check` - Quick check compilation without building
- `cargo clippy` - Run linter
- `cargo fmt` - Format code

### Running Examples
Examples are in the `examples/` directory. Run them with:
- `cargo run --bin openai_chat_completions`
- `cargo run --bin chat_completions_streaming`
- `cargo run --bin chat_console`

Each example has its own Cargo.toml and can be run from its directory.

## Architecture

This is a Rust workspace with the main AI library in `crates/ai/` and examples in `examples/`.

### Core Library Structure (`crates/ai/src/`)
- **lib.rs** - Main entry point exposing public modules
- **chat_completions.rs** - Chat completion types, traits, and request/response models
- **embeddings.rs** - Text embedding functionality
- **clients/** - Client implementations for different AI providers
  - **openai.rs** - OpenAI API client (default feature)
  - **azure_openai.rs** - Azure OpenAI client (default feature)  
  - **ollama.rs** - Ollama client (optional feature)
- **error.rs** - Error handling and Result type
- **utils/** - Utility modules for time and URI handling

### Client Architecture
All clients implement the `Client` trait which combines `ChatCompletion` and `DynClone` traits. This allows for dynamic client selection at runtime. The trait is object-safe and cloneable.

### Features
The library uses Cargo features to enable/disable clients:
- `openai_client` (default) - OpenAI API support
- `azure_openai_client` (default) - Azure OpenAI support  
- `ollama_client` - Ollama support
- TLS options: `rustls_tls` (default) or `native_tls`

### Message Types
Chat completions use an enum-based message system with role-specific types:
- `ChatCompletionMessage` enum with variants for System, User, Assistant, Developer
- Each role has its own struct type with specific fields
- Supports conversion from `(role, content)` tuples for convenience