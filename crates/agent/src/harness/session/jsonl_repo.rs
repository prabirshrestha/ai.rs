use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::Session;
use super::jsonl_storage::{JsonlSessionStorage, load_jsonl_session_metadata};
use super::repo_utils::{create_session_id, create_timestamp, get_entries_to_fork, to_session};
use crate::harness::types::{
    JsonlSessionCreateOptions, JsonlSessionForkOptions, JsonlSessionListOptions,
    JsonlSessionMetadata, SessionError, SessionErrorCode, SessionRepo, SessionResult,
    SessionStorage, SessionTreeEntry,
};

#[derive(Debug, Clone)]
pub struct JsonlSessionRepo {
    sessions_root_input: PathBuf,
    sessions_root: Arc<Mutex<Option<PathBuf>>>,
}

impl JsonlSessionRepo {
    pub fn new(sessions_root: impl Into<PathBuf>) -> Self {
        Self {
            sessions_root_input: sessions_root.into(),
            sessions_root: Arc::new(Mutex::new(None)),
        }
    }

    fn get_sessions_root(&self) -> SessionResult<PathBuf> {
        let mut cached = self.sessions_root.lock().map_err(|_| {
            SessionError::new(
                SessionErrorCode::Storage,
                "jsonl session repo lock poisoned",
            )
        })?;
        if let Some(path) = cached.as_ref() {
            return Ok(path.clone());
        }
        let path = absolute_path(&self.sessions_root_input)?;
        *cached = Some(path.clone());
        Ok(path)
    }

    fn get_session_dir(&self, cwd: &str) -> SessionResult<PathBuf> {
        Ok(self.get_sessions_root()?.join(encode_cwd(cwd)))
    }

    fn create_session_file_path(
        &self,
        cwd: &str,
        session_id: &str,
        timestamp: &str,
    ) -> SessionResult<PathBuf> {
        Ok(self.get_session_dir(cwd)?.join(format!(
            "{}_{session_id}.jsonl",
            timestamp.replace([':', '.'], "-")
        )))
    }

    async fn list_session_dirs(&self) -> SessionResult<Vec<PathBuf>> {
        let sessions_root = self.get_sessions_root()?;
        if !exists(&sessions_root).await? {
            return Ok(Vec::new());
        }
        let mut entries = tokio::fs::read_dir(&sessions_root).await.map_err(|err| {
            io_error(
                err,
                format!("Failed to list sessions root {}", sessions_root.display()),
            )
        })?;
        let mut dirs = Vec::new();
        while let Some(entry) = entries.next_entry().await.map_err(|err| {
            io_error(
                err,
                format!("Failed to list sessions root {}", sessions_root.display()),
            )
        })? {
            let file_type = entry.file_type().await.map_err(|err| {
                io_error(
                    err,
                    format!("Failed to inspect session path {}", entry.path().display()),
                )
            })?;
            if file_type.is_dir() {
                dirs.push(entry.path());
            }
        }
        Ok(dirs)
    }
}

#[async_trait]
impl
    SessionRepo<
        JsonlSessionMetadata,
        JsonlSessionCreateOptions,
        JsonlSessionListOptions,
        JsonlSessionForkOptions,
    > for JsonlSessionRepo
{
    async fn create(
        &self,
        options: JsonlSessionCreateOptions,
    ) -> SessionResult<Session<JsonlSessionMetadata>> {
        let id = options.id.unwrap_or_else(create_session_id);
        let created_at = create_timestamp();
        let session_dir = self.get_session_dir(&options.cwd)?;
        tokio::fs::create_dir_all(&session_dir)
            .await
            .map_err(|err| {
                io_error(
                    err,
                    format!(
                        "Failed to create session directory {}",
                        session_dir.display()
                    ),
                )
            })?;
        let file_path = self.create_session_file_path(&options.cwd, &id, &created_at)?;
        let storage =
            JsonlSessionStorage::create(&file_path, options.cwd, id, options.parent_session_path)
                .await?;
        Ok(to_session(Arc::new(storage)))
    }

    async fn open(
        &self,
        metadata: JsonlSessionMetadata,
    ) -> SessionResult<Session<JsonlSessionMetadata>> {
        let path = PathBuf::from(&metadata.path);
        if !exists(&path).await? {
            return Err(SessionError::new(
                SessionErrorCode::NotFound,
                format!("Session not found: {}", metadata.path),
            ));
        }
        let storage = JsonlSessionStorage::open(path).await?;
        Ok(to_session(Arc::new(storage)))
    }

    async fn list(
        &self,
        options: Option<JsonlSessionListOptions>,
    ) -> SessionResult<Vec<JsonlSessionMetadata>> {
        let dirs = if let Some(cwd) = options.and_then(|options| options.cwd) {
            vec![self.get_session_dir(&cwd)?]
        } else {
            self.list_session_dirs().await?
        };

        let mut sessions = Vec::new();
        for dir in dirs {
            if !exists(&dir).await? {
                continue;
            }
            let mut entries = tokio::fs::read_dir(&dir).await.map_err(|err| {
                io_error(err, format!("Failed to list sessions in {}", dir.display()))
            })?;
            while let Some(entry) = entries.next_entry().await.map_err(|err| {
                io_error(err, format!("Failed to list sessions in {}", dir.display()))
            })? {
                let path = entry.path();
                let file_type = entry.file_type().await.map_err(|err| {
                    io_error(
                        err,
                        format!("Failed to inspect session path {}", path.display()),
                    )
                })?;
                if file_type.is_dir()
                    || path.extension().and_then(|value| value.to_str()) != Some("jsonl")
                {
                    continue;
                }
                match load_jsonl_session_metadata(&path).await {
                    Ok(metadata) => sessions.push(metadata),
                    Err(err) if err.code == SessionErrorCode::InvalidSession => {}
                    Err(err) => return Err(err),
                }
            }
        }
        sessions
            .sort_by(|a, b| timestamp_millis(&b.created_at).cmp(&timestamp_millis(&a.created_at)));
        Ok(sessions)
    }

    async fn delete(&self, metadata: JsonlSessionMetadata) -> SessionResult<()> {
        match tokio::fs::remove_file(&metadata.path).await {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(io_error(
                err,
                format!("Failed to delete session {}", metadata.path),
            )),
        }
    }

    async fn fork(
        &self,
        source: JsonlSessionMetadata,
        options: JsonlSessionForkOptions,
    ) -> SessionResult<Session<JsonlSessionMetadata>> {
        let source_session = self.open(source.clone()).await?;
        let fork_options = (&options).into();
        let forked_entries: Vec<SessionTreeEntry> =
            get_entries_to_fork(source_session.get_storage(), &fork_options).await?;
        let id = options.id.unwrap_or_else(create_session_id);
        let created_at = create_timestamp();
        let session_dir = self.get_session_dir(&options.cwd)?;
        tokio::fs::create_dir_all(&session_dir)
            .await
            .map_err(|err| {
                io_error(
                    err,
                    format!(
                        "Failed to create session directory {}",
                        session_dir.display()
                    ),
                )
            })?;
        let file_path = self.create_session_file_path(&options.cwd, &id, &created_at)?;
        let storage = JsonlSessionStorage::create(
            &file_path,
            options.cwd,
            id,
            options.parent_session_path.or(Some(source.path)),
        )
        .await?;
        for entry in forked_entries {
            storage.append_entry(entry).await?;
        }
        Ok(to_session(Arc::new(storage)))
    }
}

pub fn encode_cwd(cwd: &str) -> String {
    let trimmed = cwd
        .strip_prefix('/')
        .or_else(|| cwd.strip_prefix('\\'))
        .unwrap_or(cwd);
    let encoded = trimmed
        .chars()
        .map(|ch| {
            if matches!(ch, '/' | '\\' | ':') {
                '-'
            } else {
                ch
            }
        })
        .collect::<String>();
    format!("--{encoded}--")
}

fn absolute_path(path: &Path) -> SessionResult<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .map_err(|err| {
                io_error(
                    err,
                    format!("Failed to resolve sessions root {}", path.display()),
                )
            })
    }
}

async fn exists(path: &Path) -> SessionResult<bool> {
    match tokio::fs::metadata(path).await {
        Ok(_) => Ok(true),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(io_error(
            err,
            format!("Failed to check session path {}", path.display()),
        )),
    }
}

fn timestamp_millis(timestamp: &str) -> i128 {
    time::OffsetDateTime::parse(timestamp, &time::format_description::well_known::Rfc3339)
        .map(|value| value.unix_timestamp_nanos() / 1_000_000)
        .unwrap_or_default()
}

fn io_error(error: std::io::Error, message: String) -> SessionError {
    let code = if error.kind() == std::io::ErrorKind::NotFound {
        SessionErrorCode::NotFound
    } else {
        SessionErrorCode::Storage
    };
    SessionError::new(code, format!("{message}: {error}"))
}
