use std::sync::{Arc, Mutex};

use crate::harness::env::{ExecChunkCallback, ExecOptions, NodeExecutionEnv};
use crate::harness::types::{ExecutionError, ExecutionErrorCode, ExecutionResult};
use crate::harness::utils::{DEFAULT_MAX_BYTES, truncate_tail};

#[derive(Clone, Default)]
pub struct ShellCaptureOptions {
    pub exec: ExecOptions,
    pub on_chunk: Option<ExecChunkCallback>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellCaptureResult {
    pub output: String,
    pub exit_code: Option<i32>,
    pub cancelled: bool,
    pub truncated: bool,
    pub full_output_path: Option<String>,
}

pub fn sanitize_binary_output(value: &str) -> String {
    value
        .chars()
        .filter(|ch| {
            let code = *ch as u32;
            if matches!(code, 0x09 | 0x0a | 0x0d) {
                return true;
            }
            if code <= 0x1f {
                return false;
            }
            if (0xfff9..=0xfffb).contains(&code) {
                return false;
            }
            true
        })
        .collect()
}

pub async fn execute_shell_with_capture(
    env: &NodeExecutionEnv,
    command: &str,
    options: ShellCaptureOptions,
) -> ExecutionResult<ShellCaptureResult> {
    let output_chunks = Arc::new(Mutex::new(Vec::<String>::new()));
    let output_bytes = Arc::new(Mutex::new(0usize));
    let total_bytes = Arc::new(Mutex::new(0usize));
    let max_output_bytes = DEFAULT_MAX_BYTES * 2;

    let mut exec_options = options.exec;
    let previous_stdout = exec_options.on_stdout.clone();
    let previous_stderr = exec_options.on_stderr.clone();
    let on_chunk = options.on_chunk.clone();
    let stdout_chunks = output_chunks.clone();
    let stdout_output_bytes = output_bytes.clone();
    let stdout_total_bytes = total_bytes.clone();
    let callback = Arc::new(move |chunk: &str| {
        if let Some(previous) = previous_stdout.as_ref() {
            previous(chunk)?;
        }
        capture_chunk(
            chunk,
            &stdout_chunks,
            &stdout_output_bytes,
            &stdout_total_bytes,
            max_output_bytes,
            on_chunk.as_ref(),
        )
    });
    let stderr_on_chunk = options.on_chunk.clone();
    let stderr_chunks = output_chunks.clone();
    let stderr_output_bytes = output_bytes.clone();
    let stderr_total_bytes = total_bytes.clone();
    let stderr_callback = Arc::new(move |chunk: &str| {
        if let Some(previous) = previous_stderr.as_ref() {
            previous(chunk)?;
        }
        capture_chunk(
            chunk,
            &stderr_chunks,
            &stderr_output_bytes,
            &stderr_total_bytes,
            max_output_bytes,
            stderr_on_chunk.as_ref(),
        )
    });
    exec_options.on_stdout = Some(callback);
    exec_options.on_stderr = Some(stderr_callback);

    let result = env.exec(command, exec_options).await;
    let tail_output = output_chunks
        .lock()
        .map_err(|_| ExecutionError::new(ExecutionErrorCode::Unknown, "output lock poisoned"))?
        .join("");
    let truncation_result = truncate_tail(&tail_output, None);

    match result {
        Ok(output) => {
            let full_output_path = if truncation_result.truncated {
                Some(write_full_output(env, &tail_output).await?)
            } else {
                None
            };
            Ok(ShellCaptureResult {
                output: if truncation_result.truncated {
                    truncation_result.content
                } else {
                    tail_output
                },
                exit_code: Some(output.exit_code),
                cancelled: false,
                truncated: truncation_result.truncated,
                full_output_path,
            })
        }
        Err(err) if err.code == ExecutionErrorCode::Aborted => {
            let full_output_path = if truncation_result.truncated {
                Some(write_full_output(env, &tail_output).await?)
            } else {
                None
            };
            Ok(ShellCaptureResult {
                output: if truncation_result.truncated {
                    truncation_result.content
                } else {
                    tail_output
                },
                exit_code: None,
                cancelled: true,
                truncated: truncation_result.truncated,
                full_output_path,
            })
        }
        Err(err) => Err(err),
    }
}

fn capture_chunk(
    chunk: &str,
    output_chunks: &Arc<Mutex<Vec<String>>>,
    output_bytes: &Arc<Mutex<usize>>,
    total_bytes: &Arc<Mutex<usize>>,
    max_output_bytes: usize,
    on_chunk: Option<&ExecChunkCallback>,
) -> ExecutionResult<()> {
    let text = sanitize_binary_output(chunk).replace('\r', "");
    *total_bytes
        .lock()
        .map_err(|_| ExecutionError::new(ExecutionErrorCode::Unknown, "output lock poisoned"))? +=
        chunk.len();
    let mut chunks = output_chunks
        .lock()
        .map_err(|_| ExecutionError::new(ExecutionErrorCode::Unknown, "output lock poisoned"))?;
    let mut bytes = output_bytes
        .lock()
        .map_err(|_| ExecutionError::new(ExecutionErrorCode::Unknown, "output lock poisoned"))?;
    chunks.push(text.clone());
    *bytes += text.len();
    while *bytes > max_output_bytes && chunks.len() > 1 {
        let removed = chunks.remove(0);
        *bytes = bytes.saturating_sub(removed.len());
    }
    if let Some(callback) = on_chunk {
        callback(&text)?;
    }
    Ok(())
}

async fn write_full_output(env: &NodeExecutionEnv, output: &str) -> ExecutionResult<String> {
    let path = env
        .create_temp_file(Some("bash-"), Some(".log"))
        .await
        .map_err(|err| {
            ExecutionError::new(ExecutionErrorCode::Unknown, err.message().to_string())
        })?;
    env.append_file(&path, output.as_bytes())
        .await
        .map_err(|err| {
            ExecutionError::new(ExecutionErrorCode::Unknown, err.message().to_string())
        })?;
    Ok(path.to_string_lossy().into_owned())
}
