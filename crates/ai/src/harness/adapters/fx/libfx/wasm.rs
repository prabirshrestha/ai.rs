//! Locates `fx-core.wasm`, the fx core that libfx ships.
//!
//! [rust] libfx's npm package carries the module beside its JavaScript. The
//! Rust port takes it from bytes, a path, the `AI_FX_CORE_WASM` environment
//! variable, or downloads the libfx package it was built against once and
//! keeps the verified module in a cache directory.

use std::io::Read as _;
use std::path::{Path, PathBuf};

use crate::harness::error::{HarnessError, HarnessResult};

/// The libfx release the port follows.
pub const LIBFX_VERSION: &str = "0.0.16";
/// Where that release's package is.
pub const LIBFX_TARBALL_URL: &str = "https://registry.npmjs.org/libfx/-/libfx-0.0.16.tgz";
/// The package's npm integrity (base64 SHA-512).
pub const LIBFX_TARBALL_SHA512: &str =
    "qA/b37rsJ8DNJXkg1+iyQCmzBbiXO+2lCXtdZzP6GjVPcStJEb3QkmhXEpOURb+VCfNc7L9gB6TiTXeS3/CPXQ==";
/// SHA-256 of the package's `fx-core.wasm`, hex.
pub const FX_CORE_SHA256: &str = "c2b17c90d1c4ce76f72d322bb7bd210e3d066eb1db05aaae053a4e4371cabb69";
/// The environment variable that names a local `fx-core.wasm`.
pub const FX_CORE_WASM_ENV: &str = "AI_FX_CORE_WASM";

/// Where fx's core comes from.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum FxCoreSource {
    /// `AI_FX_CORE_WASM`, else the cached or downloaded libfx module.
    #[default]
    Default,
    Path(PathBuf),
    Bytes(Vec<u8>),
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex(ring::digest::digest(&ring::digest::SHA256, bytes).as_ref())
}

fn base64_sha512(bytes: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD
        .encode(ring::digest::digest(&ring::digest::SHA512, bytes).as_ref())
}

fn cache_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("XDG_CACHE_HOME").filter(|dir| !dir.is_empty()) {
        return PathBuf::from(dir).join("ai-rs");
    }
    if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        let home = PathBuf::from(home);
        return if cfg!(target_os = "macos") {
            home.join("Library/Caches/ai-rs")
        } else {
            home.join(".cache/ai-rs")
        };
    }
    std::env::temp_dir().join("ai-rs")
}

/// The cached module's path.
pub fn cached_core_path() -> PathBuf {
    cache_dir()
        .join("libfx")
        .join(LIBFX_VERSION)
        .join("fx-core.wasm")
}

async fn read(path: &Path) -> HarnessResult<Vec<u8>> {
    tokio::fs::read(path).await.map_err(|error| {
        HarnessError::new(format!(
            "libfx could not read the fx core at {}: {error}",
            path.display()
        ))
        .into()
    })
}

/// Extracts `package/dist/fx-core.wasm` from libfx's package.
pub(crate) fn extract_core(tarball: &[u8]) -> HarnessResult<Vec<u8>> {
    let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(tarball));
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        if path.file_name().is_some_and(|name| name == "fx-core.wasm") {
            let mut wasm = Vec::new();
            entry.read_to_end(&mut wasm)?;
            return Ok(wasm);
        }
    }
    Err(HarnessError::new("The libfx package holds no fx-core.wasm.").into())
}

async fn download() -> HarnessResult<Vec<u8>> {
    let response = reqwest::get(LIBFX_TARBALL_URL).await.map_err(|error| {
        HarnessError::new(format!(
            "libfx could not download {LIBFX_TARBALL_URL}: {error}"
        ))
    })?;
    if !response.status().is_success() {
        return Err(HarnessError::new(format!(
            "libfx could not download {LIBFX_TARBALL_URL}: HTTP {}",
            response.status().as_u16()
        ))
        .into());
    }
    let tarball = response.bytes().await?;
    if base64_sha512(&tarball) != LIBFX_TARBALL_SHA512 {
        return Err(HarnessError::new(format!(
            "The libfx package at {LIBFX_TARBALL_URL} does not match its integrity."
        ))
        .into());
    }
    extract_core(&tarball)
}

/// Loads fx's core.
pub async fn load_core(source: &FxCoreSource) -> HarnessResult<Vec<u8>> {
    match source {
        FxCoreSource::Bytes(bytes) => Ok(bytes.clone()),
        FxCoreSource::Path(path) => read(path).await,
        FxCoreSource::Default => {
            if let Some(path) = std::env::var_os(FX_CORE_WASM_ENV).filter(|path| !path.is_empty()) {
                return read(Path::new(&path)).await;
            }
            let cached = cached_core_path();
            if let Ok(bytes) = tokio::fs::read(&cached).await {
                if sha256_hex(&bytes) == FX_CORE_SHA256 {
                    return Ok(bytes);
                }
            }
            let wasm = download().await?;
            if sha256_hex(&wasm) != FX_CORE_SHA256 {
                return Err(HarnessError::new(
                    "The libfx package's fx-core.wasm does not match its checksum.",
                )
                .into());
            }
            if let Some(parent) = cached.parent() {
                let _ = tokio::fs::create_dir_all(parent).await;
            }
            let temporary = cached.with_extension(format!("wasm.{}", std::process::id()));
            if tokio::fs::write(&temporary, &wasm).await.is_ok() {
                let _ = tokio::fs::rename(&temporary, &cached).await;
            }
            Ok(wasm)
        }
    }
}
