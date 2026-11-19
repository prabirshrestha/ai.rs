//! Port of `utils/bridge-asset.ts`.

use std::path::PathBuf;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use indexmap::IndexMap;

/// The reader [`create_read_bridge_asset`] returns.
pub type ReadBridgeAsset =
    Arc<dyn Fn(&str) -> BoxFuture<'static, std::io::Result<String>> + Send + Sync>;

/// `createReadBridgeAsset(assetUrls)`: reads a named bridge asset as UTF-8.
///
/// [rust] TS restricts `name` to the map's keys at compile time; here an
/// unknown name fails with `NotFound`.
pub fn create_read_bridge_asset(asset_paths: IndexMap<String, PathBuf>) -> ReadBridgeAsset {
    let asset_paths = Arc::new(asset_paths);
    Arc::new(move |name: &str| {
        let path = asset_paths.get(name).cloned();
        let name = name.to_string();
        async move {
            let path = path.ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Unknown bridge asset: {name}"),
                )
            })?;
            tokio::fs::read_to_string(path).await
        }
        .boxed()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary_directory() -> PathBuf {
        let directory = std::env::temp_dir().join(format!(
            "ai-sdk-harness-bridge-asset-{}",
            crate::harness::utils::generate_sandbox_credential_placeholder()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    #[tokio::test]
    async fn reads_multiple_utf8_assets_using_the_configured_map() {
        let directory = temporary_directory();
        std::fs::write(directory.join("first.txt"), "first: ä").unwrap();
        std::fs::write(directory.join("second.txt"), "second: 文").unwrap();
        let read_bridge_asset = create_read_bridge_asset(IndexMap::from([
            ("first.txt".to_string(), directory.join("first.txt")),
            ("second.txt".to_string(), directory.join("second.txt")),
        ]));

        assert_eq!(read_bridge_asset("first.txt").await.unwrap(), "first: ä");
        assert_eq!(read_bridge_asset("second.txt").await.unwrap(), "second: 文");
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn propagates_filesystem_errors_unchanged() {
        let directory = temporary_directory();
        let read_bridge_asset = create_read_bridge_asset(IndexMap::from([(
            "missing.txt".to_string(),
            directory.join("missing.txt"),
        )]));

        let error = read_bridge_asset("missing.txt").await.unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
