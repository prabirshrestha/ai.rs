//! Port of libfx's `wasm-runtime.js`: the WebAssembly host for fx's core,
//! on wasmtime. It implements WASI preview 1 as libfx's JavaScript host does
//! (stdin, stdout and stderr only; no filesystem) and fx's own host imports:
//! HTTP, host tools, steering, attachments and the stores libfx does not use.
//!
//! [rust] JSPI's `WebAssembly.Suspending` imports are wasmtime async host
//! functions. Everything the core asks of its host that has an order (its
//! output lines, host tool calls and HTTP requests) reaches the engine as one
//! ordered stream of [`CoreItem`]s, so the engine sees a model request start
//! after the output that came before it.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use bytes::Bytes;
use futures::StreamExt as _;
use futures::future::BoxFuture;
use serde_json::Value;
use tokio::sync::{Notify, mpsc, oneshot, watch};
use tokio_util::sync::CancellationToken;
use wasmtime::{Caller, Engine, Linker, Memory, Module, Store};

use super::fetch::{FxBody, FxFetchError, FxHttpRequest, FxHttpResponse};
use crate::harness::error::{HarnessError, HarnessResult};

/// `maxCoreMessageBytes`.
pub(crate) const MAX_CORE_MESSAGE_BYTES: usize = 64 * 1024 * 1024;
/// `maxPendingAttachments`.
pub(crate) const MAX_PENDING_ATTACHMENTS: usize = 8;
/// `maxSteeringMessageBytes`.
pub(crate) const MAX_STEERING_MESSAGE_BYTES: usize = 64 * 1024;
/// `maxSteeringMessages`.
pub(crate) const MAX_STEERING_MESSAGES: usize = 64;
/// `maxSteeringQueueBytes`.
pub(crate) const MAX_STEERING_QUEUE_BYTES: usize = 1024 * 1024;
const MAX_OUTBOUND_ATTACHMENTS: usize = 4;
const STREAM_READS_PER_TASK_YIELD: u32 = 32;
const STDERR_TAIL_LINES: usize = 40;

/// What the core asks of the engine, in the order it asks.
pub(crate) enum CoreItem {
    /// One line of ACP output and its size in bytes.
    Line(Value, #[allow(dead_code)] usize),
    /// `fx_host_tool_call`.
    HostToolCall {
        name: String,
        input: Value,
        call_id: String,
        reply: oneshot::Sender<HostToolReply>,
    },
    /// An HTTP request (`options.fetch`).
    Fetch {
        request: FxHttpRequest,
        signal: CancellationToken,
        reply: oneshot::Sender<Result<FxHttpResponse, FxFetchError>>,
    },
}

/// `hostToolExecutor`'s result.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct HostToolReply {
    pub content: String,
    pub is_error: bool,
    pub rich: bool,
    pub cancelled: bool,
}

#[derive(Default)]
struct Stdin {
    chunks: VecDeque<Vec<u8>>,
    closed: bool,
}

struct SteeringEntry {
    id: String,
    value: Vec<u8>,
}

#[derive(Default)]
struct Steering {
    entries: VecDeque<SteeringEntry>,
    bytes: usize,
    open: bool,
    /// A steer that cannot join the running turn waits to run next.
    handoff: bool,
}

impl Steering {
    fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
        self.handoff = false;
    }
}

#[derive(Default)]
struct Outbound {
    attachments: HashMap<i32, Vec<u8>>,
    next: i32,
}

/// State the engine and the core share.
pub(crate) struct Shared {
    stdin: Mutex<Stdin>,
    stdin_notify: Notify,
    steering: Mutex<Steering>,
    inbound: Mutex<HashMap<u32, Vec<u8>>>,
    outbound: Mutex<Outbound>,
    /// Cancels the host effects in flight; replaced once cancelled.
    effects: Mutex<CancellationToken>,
    pending_tool_result: Mutex<Option<Vec<u8>>>,
    items: mpsc::UnboundedSender<CoreItem>,
    stderr: Mutex<VecDeque<String>>,
    exit: watch::Sender<Option<i32>>,
    error: Mutex<Option<String>>,
    aborted: AtomicBool,
}

impl Shared {
    fn effects_token(&self) -> CancellationToken {
        self.effects.lock().unwrap().child_token()
    }

    fn wake_stdin(&self) {
        self.stdin_notify.notify_waiters();
    }

    /// Waits until stdin has data or closes, or `timeout` passes. Returns
    /// whether it woke for data or a close.
    async fn wait_stdin(&self, timeout: Option<Duration>) -> bool {
        let notified = self.stdin_notify.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        {
            let stdin = self.stdin.lock().unwrap();
            if stdin.closed || !stdin.chunks.is_empty() {
                return true;
            }
        }
        match timeout {
            Some(timeout) => tokio::time::timeout(timeout, notified).await.is_ok(),
            None => {
                notified.await;
                true
            }
        }
    }

    fn mark_exited(&self, code: i32) {
        self.exit.send_if_modified(|exit| {
            if exit.is_some() {
                return false;
            }
            *exit = Some(code);
            true
        });
    }
}

/// The engine's handle on a running core.
#[derive(Clone)]
pub(crate) struct FxCore {
    shared: Arc<Shared>,
    exit: watch::Receiver<Option<i32>>,
}

impl FxCore {
    /// `runtime.write(data)`: one ACP frame, newline included.
    pub fn write(&self, data: &[u8]) -> HarnessResult<()> {
        let mut stdin = self.shared.stdin.lock().unwrap();
        if stdin.closed {
            return Err(HarnessError::new("fx runtime stdin is closed").into());
        }
        if !data.is_empty() {
            stdin.chunks.push_back(data.to_vec());
        }
        drop(stdin);
        self.shared.wake_stdin();
        Ok(())
    }

    /// `writeAttachment(id, data)`.
    pub fn write_attachment(&self, id: u32, data: &[u8]) -> HarnessResult<()> {
        let mut inbound = self.shared.inbound.lock().unwrap();
        if inbound.len() >= MAX_PENDING_ATTACHMENTS {
            return Err(HarnessError::new("attachment table is full").into());
        }
        inbound.insert(id, data.to_vec());
        Ok(())
    }

    /// `takeAttachment(id)`.
    pub fn take_attachment(&self, id: i64) -> Option<Vec<u8>> {
        let id = i32::try_from(id).ok()?;
        self.shared.outbound.lock().unwrap().attachments.remove(&id)
    }

    /// `discardAttachments()`.
    pub fn discard_attachments(&self) {
        self.shared.inbound.lock().unwrap().clear();
    }

    /// `closeStdin()`.
    pub fn close_stdin(&self) {
        {
            let mut steering = self.shared.steering.lock().unwrap();
            steering.open = false;
            steering.clear();
        }
        self.shared.stdin.lock().unwrap().closed = true;
        self.shared.wake_stdin();
    }

    /// `openSteering()`.
    pub fn open_steering(&self) {
        let mut steering = self.shared.steering.lock().unwrap();
        steering.clear();
        steering.open = true;
    }

    /// `closeSteering()`.
    pub fn close_steering(&self) {
        let mut steering = self.shared.steering.lock().unwrap();
        steering.open = false;
        steering.clear();
    }

    /// `steer: queueSteering(text, id)`.
    pub fn steer(&self, text: &str, id: &str) -> HarnessResult<()> {
        let mut steering = self.shared.steering.lock().unwrap();
        if !steering.open {
            return Err(HarnessError::new("no prompt is running").into());
        }
        let value = text.as_bytes().to_vec();
        if value.is_empty() {
            return Err(HarnessError::new("steering text cannot be empty").into());
        }
        if value.len() > MAX_STEERING_MESSAGE_BYTES {
            return Err(HarnessError::new(format!(
                "steering text exceeds the {MAX_STEERING_MESSAGE_BYTES} byte libfx limit"
            ))
            .into());
        }
        if steering.entries.len() >= MAX_STEERING_MESSAGES
            || value.len() > MAX_STEERING_QUEUE_BYTES - steering.bytes
        {
            return Err(HarnessError::new("steering queue is full").into());
        }
        steering.bytes += value.len();
        steering.entries.push_back(SteeringEntry {
            id: id.to_string(),
            value,
        });
        Ok(())
    }

    /// `withdrawSteering(id)`: false once the core took it.
    #[allow(dead_code)]
    pub fn withdraw_steering(&self, id: &str) -> bool {
        let mut steering = self.shared.steering.lock().unwrap();
        let Some(index) = steering.entries.iter().position(|entry| entry.id == id) else {
            return false;
        };
        let entry = steering.entries.remove(index).unwrap();
        steering.bytes -= entry.value.len();
        true
    }

    /// `handOffSteering()`.
    #[allow(dead_code)]
    pub fn hand_off_steering(&self) -> bool {
        let mut steering = self.shared.steering.lock().unwrap();
        if !steering.open {
            return false;
        }
        steering.handoff = true;
        true
    }

    /// `abortHostEffects()`.
    pub fn abort_host_effects(&self) {
        *self.shared.pending_tool_result.lock().unwrap() = None;
        let mut effects = self.shared.effects.lock().unwrap();
        effects.cancel();
        *effects = CancellationToken::new();
    }

    /// `abort(error)`.
    pub fn abort(&self, error: impl Into<String>) {
        self.shared.aborted.store(true, Ordering::SeqCst);
        self.shared
            .error
            .lock()
            .unwrap()
            .get_or_insert(error.into());
        self.abort_host_effects();
        self.shared.stdin.lock().unwrap().closed = true;
        self.shared.wake_stdin();
        self.shared.mark_exited(130);
    }

    /// Resolves with the exit code once the core exits.
    pub async fn exited(&self) -> i32 {
        let mut exit = self.exit.clone();
        loop {
            if let Some(code) = *exit.borrow_and_update() {
                return code;
            }
            if exit.changed().await.is_err() {
                return exit.borrow().unwrap_or(1);
            }
        }
    }

    #[allow(dead_code)]
    pub fn exit_code(&self) -> Option<i32> {
        *self.exit.borrow()
    }

    /// `runtime.error`: why the core stopped, when it failed.
    pub fn error(&self) -> Option<String> {
        self.shared.error.lock().unwrap().clone()
    }

    /// The core's last stderr lines.
    #[allow(dead_code)]
    pub fn stderr(&self) -> String {
        self.shared
            .stderr
            .lock()
            .unwrap()
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Splits the core's stdout into lines (`CoreOutput`).
#[derive(Default)]
struct CoreOutput {
    buffer: Vec<u8>,
}

impl CoreOutput {
    fn write(
        &mut self,
        chunk: &[u8],
        items: &mpsc::UnboundedSender<CoreItem>,
    ) -> Result<(), String> {
        let mut offset = 0;
        while offset < chunk.len() {
            let newline = chunk[offset..].iter().position(|byte| *byte == b'\n');
            let end = newline.map_or(chunk.len(), |position| offset + position);
            let length = self.buffer.len() + (end - offset);
            if length + usize::from(newline.is_some()) > MAX_CORE_MESSAGE_BYTES {
                return Err("core output message exceeds 64 MiB".into());
            }
            self.buffer.extend_from_slice(&chunk[offset..end]);
            let Some(_) = newline else {
                return Ok(());
            };
            offset = end + 1;
            let line = std::mem::take(&mut self.buffer);
            let size = line.len() + 1;
            if line.is_empty() {
                continue;
            }
            let text = std::str::from_utf8(&line).map_err(|error| error.to_string())?;
            let value: Value = serde_json::from_str(text).map_err(|error| error.to_string())?;
            let _ = items.send(CoreItem::Line(value, size));
        }
        Ok(())
    }

    fn finish(&self) -> Result<(), String> {
        if self.buffer.is_empty() {
            Ok(())
        } else {
            Err("core output ended within a message".into())
        }
    }
}

type PendingRead = BoxFuture<'static, (FxBody, Option<Result<Bytes, FxFetchError>>)>;

#[derive(Default)]
struct StreamState {
    signal: CancellationToken,
    response: Option<oneshot::Receiver<Result<FxHttpResponse, FxFetchError>>>,
    /// The settled response: its status, or how it failed.
    settled: Option<Result<u16, FxFetchError>>,
    body: Option<FxBody>,
    pending_read: Option<PendingRead>,
    leftover: Bytes,
    read_error: Option<FxFetchError>,
    done: bool,
    reads_since_task_yield: u32,
}

impl StreamState {
    fn settle(
        &mut self,
        response: Result<Result<FxHttpResponse, FxFetchError>, oneshot::error::RecvError>,
    ) {
        self.response = None;
        self.settled = Some(match response {
            Ok(Ok(response)) => {
                self.body = Some(response.body);
                Ok(response.status)
            }
            Ok(Err(error)) => Err(error),
            Err(_) => Err(FxFetchError::Failed("fetch failed".into())),
        });
    }
}

/// The store's data.
pub(crate) struct HostData {
    shared: Arc<Shared>,
    memory: Option<Memory>,
    args: Vec<String>,
    env: Vec<String>,
    output: CoreOutput,
    streams: HashMap<i32, StreamState>,
    next_handle: i32,
}

fn memory_of(caller: &Caller<'_, HostData>) -> Memory {
    caller.data().memory.expect("fx core memory")
}

/// `checkedBytes(ptr, len)`: a copy, or None out of bounds.
fn read_bytes(caller: &Caller<'_, HostData>, ptr: i32, len: i32) -> Option<Vec<u8>> {
    let (ptr, len) = (usize::try_from(ptr).ok()?, usize::try_from(len).ok()?);
    let data = memory_of(caller).data(caller);
    data.get(ptr..ptr.checked_add(len)?).map(<[u8]>::to_vec)
}

fn read_text(caller: &Caller<'_, HostData>, ptr: i32, len: i32) -> String {
    read_bytes(caller, ptr, len)
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        .unwrap_or_default()
}

fn in_bounds(caller: &Caller<'_, HostData>, ptr: i32, len: i32) -> bool {
    let (Ok(ptr), Ok(len)) = (usize::try_from(ptr), usize::try_from(len)) else {
        return false;
    };
    ptr.checked_add(len)
        .is_some_and(|end| end <= memory_of(caller).data_size(caller))
}

fn write_bytes(caller: &mut Caller<'_, HostData>, ptr: i32, bytes: &[u8]) -> bool {
    let Ok(ptr) = usize::try_from(ptr) else {
        return false;
    };
    let memory = memory_of(caller);
    match memory
        .data_mut(caller)
        .get_mut(ptr..ptr.saturating_add(bytes.len()))
    {
        Some(target) => {
            target.copy_from_slice(bytes);
            true
        }
        None => false,
    }
}

fn write_u16(caller: &mut Caller<'_, HostData>, ptr: i32, value: u16) {
    write_bytes(caller, ptr, &value.to_le_bytes());
}

fn write_u32(caller: &mut Caller<'_, HostData>, ptr: i32, value: u32) {
    write_bytes(caller, ptr, &value.to_le_bytes());
}

fn write_u64(caller: &mut Caller<'_, HostData>, ptr: i32, value: u64) {
    write_bytes(caller, ptr, &value.to_le_bytes());
}

fn read_u32(caller: &Caller<'_, HostData>, ptr: i32) -> u32 {
    read_bytes(caller, ptr, 4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .unwrap_or(0)
}

/// `writeVector(values, ptrs, data)`.
fn write_vector(caller: &mut Caller<'_, HostData>, values: &[String], ptrs: i32, data: i32) {
    let mut cursor = data;
    for (index, value) in values.iter().enumerate() {
        let mut encoded = value.as_bytes().to_vec();
        encoded.push(0);
        write_u32(caller, ptrs + (index as i32) * 4, cursor as u32);
        write_bytes(caller, cursor, &encoded);
        cursor += encoded.len() as i32;
    }
}

fn vector_size(values: &[String]) -> u32 {
    values.iter().map(|value| value.len() as u32 + 1).sum()
}

/// The iovecs at `iovs`: (pointer, length) pairs.
fn iovecs(caller: &Caller<'_, HostData>, iovs: i32, count: i32) -> Vec<(i32, i32)> {
    (0..count)
        .map(|index| {
            let base = iovs + index * 8;
            (
                read_u32(caller, base) as i32,
                read_u32(caller, base + 4) as i32,
            )
        })
        .collect()
}

fn now_nanos() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| (duration.as_millis() as u64) * 1_000_000)
        .unwrap_or(0)
}

/// `fdRead`'s attempt: copies what stdin holds, or None when it is empty.
fn read_stdin(caller: &mut Caller<'_, HostData>, iovs: &[(i32, i32)]) -> Option<u32> {
    let shared = caller.data().shared.clone();
    let mut stdin = shared.stdin.lock().unwrap();
    let mut total = 0u32;
    for &(ptr, len) in iovs {
        let Some(chunk) = stdin.chunks.front_mut() else {
            break;
        };
        let take = chunk.len().min(len.max(0) as usize);
        let value: Vec<u8> = chunk.drain(..take).collect();
        if chunk.is_empty() {
            stdin.chunks.pop_front();
        }
        write_bytes(caller, ptr, &value);
        total += value.len() as u32;
        if value.len() < len as usize {
            break;
        }
    }
    (total > 0).then_some(total)
}

fn trap(message: impl Into<String>) -> wasmtime::Error {
    wasmtime::Error::msg(message.into())
}

const UNAVAILABLE: i32 = 52;

static ENGINE: LazyLock<Engine> = LazyLock::new(|| {
    let config = wasmtime::Config::new();
    Engine::new(&config).expect("wasmtime engine")
});

static MODULES: LazyLock<Mutex<HashMap<Vec<u8>, Module>>> = LazyLock::new(Mutex::default);

/// Compiles fx's core once per process for each module.
pub(crate) fn compile(wasm: &[u8]) -> HarnessResult<Module> {
    let key = ring::digest::digest(&ring::digest::SHA256, wasm)
        .as_ref()
        .to_vec();
    if let Some(module) = MODULES.lock().unwrap().get(&key) {
        return Ok(module.clone());
    }
    let module = Module::new(&ENGINE, wasm).map_err(|error| {
        HarnessError::new(format!(
            "libfx could not load the fx WebAssembly core: {error}"
        ))
    })?;
    MODULES.lock().unwrap().insert(key, module.clone());
    Ok(module)
}

fn linker() -> HarnessResult<Linker<HostData>> {
    let mut linker = Linker::new(&ENGINE);
    define_wasi(&mut linker).map_err(|error| HarnessError::new(error.to_string()))?;
    define_fx(&mut linker).map_err(|error| HarnessError::new(error.to_string()))?;
    Ok(linker)
}

fn define_wasi(linker: &mut Linker<HostData>) -> wasmtime::Result<()> {
    const W: &str = "wasi_snapshot_preview1";
    linker.func_wrap(
        W,
        "args_sizes_get",
        |mut caller: Caller<'_, HostData>, count: i32, size: i32| {
            let args = caller.data().args.clone();
            write_u32(&mut caller, count, args.len() as u32);
            write_u32(&mut caller, size, vector_size(&args));
            0i32
        },
    )?;
    linker.func_wrap(
        W,
        "args_get",
        |mut caller: Caller<'_, HostData>, ptrs: i32, data: i32| {
            let args = caller.data().args.clone();
            write_vector(&mut caller, &args, ptrs, data);
            0i32
        },
    )?;
    linker.func_wrap(
        W,
        "environ_sizes_get",
        |mut caller: Caller<'_, HostData>, count: i32, size: i32| {
            let env = caller.data().env.clone();
            write_u32(&mut caller, count, env.len() as u32);
            write_u32(&mut caller, size, vector_size(&env));
            0i32
        },
    )?;
    linker.func_wrap(
        W,
        "environ_get",
        |mut caller: Caller<'_, HostData>, ptrs: i32, data: i32| {
            let env = caller.data().env.clone();
            write_vector(&mut caller, &env, ptrs, data);
            0i32
        },
    )?;
    linker.func_wrap(
        W,
        "fd_write",
        |mut caller: Caller<'_, HostData>,
         fd: i32,
         iovs: i32,
         count: i32,
         nwritten: i32|
         -> wasmtime::Result<i32> {
            let vectors = iovecs(&caller, iovs, count);
            let total: usize = vectors.iter().map(|(_, len)| *len as usize).sum();
            if total > MAX_CORE_MESSAGE_BYTES {
                return Err(trap("core output message exceeds 64 MiB"));
            }
            if fd == 1 || fd == 2 {
                let mut chunk = Vec::with_capacity(total);
                for (ptr, len) in vectors {
                    chunk.extend(read_bytes(&caller, ptr, len).unwrap_or_default());
                }
                let shared = caller.data().shared.clone();
                if fd == 1 {
                    let items = shared.items.clone();
                    caller
                        .data_mut()
                        .output
                        .write(&chunk, &items)
                        .map_err(trap)?;
                } else {
                    let mut stderr = shared.stderr.lock().unwrap();
                    for line in String::from_utf8_lossy(&chunk).lines() {
                        stderr.push_back(line.to_string());
                        while stderr.len() > STDERR_TAIL_LINES {
                            stderr.pop_front();
                        }
                    }
                }
            }
            write_u32(&mut caller, nwritten, total as u32);
            Ok(0)
        },
    )?;
    linker.func_wrap_async(
        W,
        "fd_read",
        |mut caller: Caller<'_, HostData>, (fd, iovs, count, nread): (i32, i32, i32, i32)| {
            Box::new(async move {
                if fd != 0 {
                    return Ok(8i32);
                }
                let vectors = iovecs(&caller, iovs, count);
                if let Some(total) = read_stdin(&mut caller, &vectors) {
                    write_u32(&mut caller, nread, total);
                    return Ok(0);
                }
                let shared = caller.data().shared.clone();
                if shared.stdin.lock().unwrap().closed {
                    write_u32(&mut caller, nread, 0);
                    return Ok(0);
                }
                shared.wait_stdin(None).await;
                let total = read_stdin(&mut caller, &vectors).unwrap_or(0);
                write_u32(&mut caller, nread, total);
                Ok(0)
            })
        },
    )?;
    linker.func_wrap(W, "fd_close", |_: Caller<'_, HostData>, _fd: i32| 0i32)?;
    linker.func_wrap(
        W,
        "fd_fdstat_get",
        |mut caller: Caller<'_, HostData>, fd: i32, out: i32| {
            let mut stat = [0u8; 24];
            stat[0] = if fd <= 2 { 2 } else { 0 };
            stat[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
            stat[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
            write_bytes(&mut caller, out, &stat);
            0i32
        },
    )?;
    linker.func_wrap(
        W,
        "fd_filestat_get",
        |_: Caller<'_, HostData>, _: i32, _: i32| UNAVAILABLE,
    )?;
    linker.func_wrap(
        W,
        "fd_filestat_set_size",
        |_: Caller<'_, HostData>, _: i32, _: i64| UNAVAILABLE,
    )?;
    linker.func_wrap(
        W,
        "fd_filestat_set_times",
        |_: Caller<'_, HostData>, _: i32, _: i64, _: i64, _: i32| UNAVAILABLE,
    )?;
    linker.func_wrap(
        W,
        "fd_pread",
        |_: Caller<'_, HostData>, _: i32, _: i32, _: i32, _: i64, _: i32| UNAVAILABLE,
    )?;
    linker.func_wrap(
        W,
        "fd_prestat_get",
        |_: Caller<'_, HostData>, _: i32, _: i32| 8i32,
    )?;
    linker.func_wrap(
        W,
        "fd_prestat_dir_name",
        |_: Caller<'_, HostData>, _: i32, _: i32, _: i32| UNAVAILABLE,
    )?;
    linker.func_wrap(
        W,
        "fd_pwrite",
        |_: Caller<'_, HostData>, _: i32, _: i32, _: i32, _: i64, _: i32| UNAVAILABLE,
    )?;
    linker.func_wrap(
        W,
        "fd_readdir",
        |_: Caller<'_, HostData>, _: i32, _: i32, _: i32, _: i64, _: i32| UNAVAILABLE,
    )?;
    linker.func_wrap(
        W,
        "fd_seek",
        |_: Caller<'_, HostData>, _: i32, _: i64, _: i32, _: i32| 29i32,
    )?;
    linker.func_wrap(W, "fd_sync", |_: Caller<'_, HostData>, _: i32| 0i32)?;
    linker.func_wrap(
        W,
        "clock_res_get",
        |mut caller: Caller<'_, HostData>, _id: i32, out: i32| {
            write_u64(&mut caller, out, 1_000_000);
            0i32
        },
    )?;
    linker.func_wrap(
        W,
        "clock_time_get",
        |mut caller: Caller<'_, HostData>, _id: i32, _precision: i64, out: i32| {
            write_u64(&mut caller, out, now_nanos());
            0i32
        },
    )?;
    linker.func_wrap(
        W,
        "path_create_directory",
        |_: Caller<'_, HostData>, _: i32, _: i32, _: i32| UNAVAILABLE,
    )?;
    linker.func_wrap(
        W,
        "path_filestat_get",
        |_: Caller<'_, HostData>, _: i32, _: i32, _: i32, _: i32, _: i32| UNAVAILABLE,
    )?;
    linker.func_wrap(
        W,
        "path_filestat_set_times",
        |_: Caller<'_, HostData>, _: i32, _: i32, _: i32, _: i32, _: i64, _: i64, _: i32| {
            UNAVAILABLE
        },
    )?;
    linker.func_wrap(
        W,
        "path_link",
        |_: Caller<'_, HostData>, _: i32, _: i32, _: i32, _: i32, _: i32, _: i32, _: i32| {
            UNAVAILABLE
        },
    )?;
    linker.func_wrap(
        W,
        "path_open",
        |_: Caller<'_, HostData>,
         _: i32,
         _: i32,
         _: i32,
         _: i32,
         _: i32,
         _: i64,
         _: i64,
         _: i32,
         _: i32| { UNAVAILABLE },
    )?;
    linker.func_wrap(
        W,
        "path_readlink",
        |_: Caller<'_, HostData>, _: i32, _: i32, _: i32, _: i32, _: i32, _: i32| UNAVAILABLE,
    )?;
    linker.func_wrap(
        W,
        "path_remove_directory",
        |_: Caller<'_, HostData>, _: i32, _: i32, _: i32| UNAVAILABLE,
    )?;
    linker.func_wrap(
        W,
        "path_rename",
        |_: Caller<'_, HostData>, _: i32, _: i32, _: i32, _: i32, _: i32, _: i32| UNAVAILABLE,
    )?;
    linker.func_wrap(
        W,
        "path_symlink",
        |_: Caller<'_, HostData>, _: i32, _: i32, _: i32, _: i32, _: i32| UNAVAILABLE,
    )?;
    linker.func_wrap(
        W,
        "path_unlink_file",
        |_: Caller<'_, HostData>, _: i32, _: i32, _: i32| UNAVAILABLE,
    )?;
    linker.func_wrap(
        W,
        "random_get",
        |mut caller: Caller<'_, HostData>, ptr: i32, len: i32| {
            let mut bytes = vec![0u8; len.max(0) as usize];
            let _ = ring::rand::SecureRandom::fill(&ring::rand::SystemRandom::new(), &mut bytes);
            write_bytes(&mut caller, ptr, &bytes);
            0i32
        },
    )?;
    linker.func_wrap_async(
        W,
        "poll_oneoff",
        |mut caller: Caller<'_, HostData>,
         (subscriptions, events, count, nevents): (i32, i32, i32, i32)| {
            Box::new(async move {
                let shared = caller.data().shared.clone();
                let mut timeout: Option<Duration> = None;
                for index in 0..count {
                    let base = subscriptions + index * 48;
                    let subscription = read_bytes(&caller, base, 48).unwrap_or_else(|| vec![0; 48]);
                    let kind = subscription[8];
                    if kind == 1 && !shared.stdin.lock().unwrap().chunks.is_empty() {
                        let mut event = [0u8; 32];
                        event[..8].copy_from_slice(&subscription[..8]);
                        event[10] = 1;
                        write_bytes(&mut caller, events, &event);
                        write_u32(&mut caller, nevents, 1);
                        return Ok(0i32);
                    }
                    if kind == 0 {
                        let nanos = u64::from_le_bytes(subscription[24..32].try_into().unwrap());
                        timeout = Some(Duration::from_millis(nanos / 1_000_000));
                    }
                }
                shared.wait_stdin(timeout).await;
                let first = read_bytes(&caller, subscriptions, 8).unwrap_or_else(|| vec![0; 8]);
                let mut event = [0u8; 32];
                event[..8].copy_from_slice(&first);
                write_bytes(&mut caller, events, &event);
                write_u32(&mut caller, nevents, 1);
                Ok(0)
            })
        },
    )?;
    linker.func_wrap(
        W,
        "proc_exit",
        |caller: Caller<'_, HostData>, code: i32| -> wasmtime::Result<()> {
            caller.data().shared.mark_exited(code);
            Err(trap(format!("proc_exit({code})")))
        },
    )?;
    Ok(())
}

/// The headers JSON the core sends: `[{ name, value }]`.
fn headers_from_json(text: &str) -> Vec<(String, String)> {
    let Ok(Value::Array(entries)) =
        serde_json::from_str::<Value>(if text.is_empty() { "[]" } else { text })
    else {
        return Vec::new();
    };
    entries
        .iter()
        .filter_map(|entry| {
            Some((
                entry.get("name")?.as_str()?.to_string(),
                entry.get("value")?.as_str()?.to_string(),
            ))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn request_from(
    caller: &Caller<'_, HostData>,
    method: (i32, i32),
    url: (i32, i32),
    headers: (i32, i32),
    body: (i32, i32),
) -> FxHttpRequest {
    FxHttpRequest {
        method: read_text(caller, method.0, method.1),
        url: read_text(caller, url.0, url.1),
        headers: headers_from_json(&read_text(caller, headers.0, headers.1)),
        body: (body.1 > 0)
            .then(|| Bytes::from(read_bytes(caller, body.0, body.1).unwrap_or_default())),
    }
}

/// Sends a request to the engine, which fetches it in order.
fn fetch(
    shared: &Shared,
    request: FxHttpRequest,
    signal: CancellationToken,
) -> oneshot::Receiver<Result<FxHttpResponse, FxFetchError>> {
    let (reply, receiver) = oneshot::channel();
    let _ = shared.items.send(CoreItem::Fetch {
        request,
        signal,
        reply,
    });
    receiver
}

fn define_fx(linker: &mut Linker<HostData>) -> wasmtime::Result<()> {
    const F: &str = "fx";
    linker.func_wrap(
        F,
        "fx_http_stream_open",
        |mut caller: Caller<'_, HostData>,
         method_ptr: i32,
         method_len: i32,
         url_ptr: i32,
         url_len: i32,
         headers_ptr: i32,
         headers_len: i32,
         body_ptr: i32,
         body_len: i32| {
            let request = request_from(
                &caller,
                (method_ptr, method_len),
                (url_ptr, url_len),
                (headers_ptr, headers_len),
                (body_ptr, body_len),
            );
            let shared = caller.data().shared.clone();
            let signal = shared.effects_token();
            let response = fetch(&shared, request, signal.clone());
            let data = caller.data_mut();
            data.next_handle += 1;
            let handle = data.next_handle;
            data.streams.insert(
                handle,
                StreamState {
                    signal,
                    response: Some(response),
                    ..Default::default()
                },
            );
            handle
        },
    )?;
    linker.func_wrap_async(
        F,
        "fx_http_stream_status",
        |mut caller: Caller<'_, HostData>, (handle, status_out): (i32, i32)| {
            Box::new(async move {
                let Some(state) = caller.data_mut().streams.get_mut(&handle) else {
                    return Ok(-1i32);
                };
                if state.settled.is_none()
                    && let Some(response) = state.response.as_mut()
                    && let Ok(response) =
                        tokio::time::timeout(Duration::from_millis(50), response).await
                {
                    state.settle(response);
                }
                let state = caller.data_mut().streams.get_mut(&handle).unwrap();
                Ok(match state.settled.clone() {
                    None => 0,
                    Some(Err(FxFetchError::Aborted)) => -2,
                    Some(Err(_)) => -1,
                    Some(Ok(status)) => {
                        write_u16(&mut caller, status_out, status);
                        1
                    }
                })
            })
        },
    )?;
    linker.func_wrap_async(
        F,
        "fx_http_stream_next",
        |mut caller: Caller<'_, HostData>, (handle, out_ptr, out_cap): (i32, i32, i32)| {
            Box::new(async move {
                let Some(state) = caller.data_mut().streams.get_mut(&handle) else {
                    return Ok(-1i32);
                };
                // A body read before the response settled waits for it.
                if state.settled.is_none()
                    && let Some(response) = state.response.as_mut()
                {
                    match tokio::time::timeout(Duration::from_millis(50), response).await {
                        Ok(response) => state.settle(response),
                        Err(_) => return Ok(-3),
                    }
                }
                let cap = out_cap.max(0) as usize;
                let result: i32 = 'read: {
                    if !state.leftover.is_empty() {
                        break 'read -10;
                    }
                    if let Some(error) = &state.read_error {
                        break 'read if *error == FxFetchError::Aborted {
                            -2
                        } else {
                            -1
                        };
                    }
                    if state.done {
                        break 'read 0;
                    }
                    let mut pending = match state.pending_read.take() {
                        Some(pending) => pending,
                        None => {
                            let Some(mut body) = state.body.take() else {
                                break 'read 0;
                            };
                            let signal = state.signal.clone();
                            Box::pin(async move {
                                let item = tokio::select! {
                                    item = body.next() => item,
                                    _ = signal.cancelled() => Some(Err(FxFetchError::Aborted)),
                                };
                                (body, item)
                            })
                        }
                    };
                    match tokio::time::timeout(Duration::from_millis(50), &mut pending).await {
                        Err(_) => {
                            state.pending_read = Some(pending);
                            -3
                        }
                        Ok((body, item)) => {
                            state.body = Some(body);
                            match item {
                                None => {
                                    state.done = true;
                                    0
                                }
                                Some(Err(error)) => {
                                    let aborted = error == FxFetchError::Aborted;
                                    state.read_error = Some(error);
                                    if aborted { -2 } else { -1 }
                                }
                                Some(Ok(chunk)) if chunk.is_empty() => -3,
                                Some(Ok(chunk)) => {
                                    state.leftover = chunk;
                                    -10
                                }
                            }
                        }
                    }
                };
                if result != -10 {
                    return Ok(result);
                }
                let state = caller.data_mut().streams.get_mut(&handle).unwrap();
                let take = state.leftover.len().min(cap);
                let written = state.leftover.split_to(take);
                state.reads_since_task_yield += 1;
                let yield_now = state.reads_since_task_yield >= STREAM_READS_PER_TASK_YIELD;
                if yield_now {
                    state.reads_since_task_yield = 0;
                }
                write_bytes(&mut caller, out_ptr, &written);
                if yield_now {
                    tokio::task::yield_now().await;
                }
                Ok(written.len() as i32)
            })
        },
    )?;
    linker.func_wrap(
        F,
        "fx_http_stream_close",
        |mut caller: Caller<'_, HostData>, handle: i32| {
            if let Some(state) = caller.data_mut().streams.remove(&handle) {
                state.signal.cancel();
            }
        },
    )?;
    linker.func_wrap_async(
        F,
        "fx_http_request",
        |mut caller: Caller<'_, HostData>,
         (
            method_ptr,
            method_len,
            url_ptr,
            url_len,
            headers_ptr,
            headers_len,
            body_ptr,
            body_len,
            status_out,
            response_ptr,
            response_cap,
        ): (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)| {
            Box::new(async move {
                let request = request_from(
                    &caller,
                    (method_ptr, method_len),
                    (url_ptr, url_len),
                    (headers_ptr, headers_len),
                    (body_ptr, body_len),
                );
                let shared = caller.data().shared.clone();
                let signal = shared.effects_token();
                let response = fetch(&shared, request, signal.clone());
                let outcome = async {
                    let response = response.await.ok()?.ok()?;
                    let status = response.status;
                    let body = response.bytes().await.ok()?;
                    Some((status, body))
                };
                let outcome = tokio::select! {
                    outcome = outcome => outcome,
                    _ = signal.cancelled() => None,
                };
                let Some((status, body)) = outcome else {
                    return Ok(-1i32);
                };
                if signal.is_cancelled() {
                    return Ok(-1);
                }
                write_u16(&mut caller, status_out, status);
                if body.len() > response_cap.max(0) as usize {
                    return Ok(-2);
                }
                write_bytes(&mut caller, response_ptr, &body);
                Ok(body.len() as i32)
            })
        },
    )?;
    linker.func_wrap_async(
        F,
        "fx_host_tool_call",
        |mut caller: Caller<'_, HostData>,
         (
            name_ptr,
            name_len,
            arguments_ptr,
            arguments_len,
            output_ptr,
            output_cap,
            status_ptr,
            call_id_ptr,
            call_id_len,
        ): (i32, i32, i32, i32, i32, i32, i32, i32, i32)| {
            Box::new(async move {
                let shared = caller.data().shared.clone();
                *shared.pending_tool_result.lock().unwrap() = None;
                let Ok(input) = serde_json::from_str::<Value>(&read_text(
                    &caller,
                    arguments_ptr,
                    arguments_len,
                )) else {
                    return Ok(-1i32);
                };
                let (reply, receiver) = oneshot::channel();
                let _ = shared.items.send(CoreItem::HostToolCall {
                    name: read_text(&caller, name_ptr, name_len),
                    input,
                    call_id: read_text(&caller, call_id_ptr, call_id_len),
                    reply,
                });
                let Ok(result) = receiver.await else {
                    return Ok(-1);
                };
                if result.cancelled {
                    return Ok(-2);
                }
                let output = result.content.into_bytes();
                write_bytes(
                    &mut caller,
                    status_ptr,
                    &[u8::from(result.is_error) + if result.rich { 2 } else { 0 }],
                );
                if output.len() > output_cap.max(0) as usize {
                    if !result.rich || output.len() > 8 * 1024 * 1024 {
                        return Ok(-3);
                    }
                    let length = output.len() as i32;
                    *shared.pending_tool_result.lock().unwrap() = Some(output);
                    return Ok(length);
                }
                write_bytes(&mut caller, output_ptr, &output);
                Ok(output.len() as i32)
            })
        },
    )?;
    linker.func_wrap(
        F,
        "fx_host_tool_result_read",
        |mut caller: Caller<'_, HostData>, offset: i32, ptr: i32, cap: i32| {
            let shared = caller.data().shared.clone();
            let pending = shared.pending_tool_result.lock().unwrap();
            let Some(result) = pending.as_ref() else {
                return -1i32;
            };
            if offset < 0 || offset as usize > result.len() {
                return -1;
            }
            let start = offset as usize;
            let end = (start + cap.max(0) as usize).min(result.len());
            let chunk = result[start..end].to_vec();
            drop(pending);
            write_bytes(&mut caller, ptr, &chunk);
            chunk.len() as i32
        },
    )?;
    linker.func_wrap(
        F,
        "fx_host_tool_result_release",
        |caller: Caller<'_, HostData>| {
            *caller.data().shared.pending_tool_result.lock().unwrap() = None;
        },
    )?;
    // Resolves once the journal holds every event the core sent; libfx's
    // adapter keeps no journal, so that is at once.
    linker.func_wrap_async(F, "fx_journal_flush", |_: Caller<'_, HostData>, (): ()| {
        Box::new(async move { Ok(0i32) })
    })?;
    linker.func_wrap(
        F,
        "fx_steering_take",
        |mut caller: Caller<'_, HostData>, output_ptr: i32, output_cap: i32| {
            let shared = caller.data().shared.clone();
            let mut steering = shared.steering.lock().unwrap();
            let Some(entry) = steering.entries.front() else {
                return 0i32;
            };
            let length = entry.id.len() + 1 + entry.value.len();
            if !in_bounds(&caller, output_ptr, output_cap) || length > output_cap as usize {
                return -1;
            }
            let mut message = entry.id.as_bytes().to_vec();
            message.push(b'\n');
            message.extend_from_slice(&entry.value);
            let entry = steering.entries.pop_front().unwrap();
            steering.bytes -= entry.value.len();
            drop(steering);
            write_bytes(&mut caller, output_ptr, &message);
            length as i32
        },
    )?;
    linker.func_wrap(F, "fx_steering_close", |caller: Caller<'_, HostData>| {
        let mut steering = caller.data().shared.steering.lock().unwrap();
        steering.open = false;
        steering.clear();
    })?;
    // -1 once the turn takes no more steering, else twice how many steers
    // wait, plus one when a handoff does.
    linker.func_wrap(F, "fx_steering_state", |caller: Caller<'_, HostData>| {
        let steering = caller.data().shared.steering.lock().unwrap();
        if steering.open {
            steering.entries.len() as i32 * 2 + i32::from(steering.handoff)
        } else {
            -1i32
        }
    })?;
    linker.func_wrap(
        F,
        "fx_attachment_size",
        |caller: Caller<'_, HostData>, id: i32| {
            caller
                .data()
                .shared
                .inbound
                .lock()
                .unwrap()
                .get(&(id as u32))
                .map_or(-1i32, |value| value.len() as i32)
        },
    )?;
    linker.func_wrap(
        F,
        "fx_attachment_take",
        |mut caller: Caller<'_, HostData>, id: i32, output_ptr: i32, output_cap: i32| {
            let shared = caller.data().shared.clone();
            let key = id as u32;
            let Some(value) = shared.inbound.lock().unwrap().get(&key).cloned() else {
                return -1i32;
            };
            // An empty payload needs no output buffer, whose pointer may be arbitrary.
            if !value.is_empty() {
                if !in_bounds(&caller, output_ptr, output_cap) || value.len() > output_cap as usize
                {
                    return -1;
                }
                write_bytes(&mut caller, output_ptr, &value);
            }
            shared.inbound.lock().unwrap().remove(&key);
            value.len() as i32
        },
    )?;
    linker.func_wrap(
        F,
        "fx_attachment_put",
        |caller: Caller<'_, HostData>, input_ptr: i32, input_len: i32| {
            let Some(input) = read_bytes(&caller, input_ptr, input_len) else {
                return -1i32;
            };
            let mut outbound = caller.data().shared.outbound.lock().unwrap();
            if outbound.attachments.len() >= MAX_OUTBOUND_ATTACHMENTS {
                return -1;
            }
            let id = outbound.next.max(1);
            outbound.next = if id == 0x7fff_ffff { 1 } else { id + 1 };
            outbound.attachments.insert(id, input);
            id
        },
    )?;
    // libfx gives the core no session or OAuth store.
    linker.func_wrap_async(
        F,
        "fx_session_load",
        |_: Caller<'_, HostData>, _: (i32, i32, i32, i32, i32, i32, i32)| {
            Box::new(async { Ok(-1i32) })
        },
    )?;
    linker.func_wrap_async(
        F,
        "fx_session_commit",
        |_: Caller<'_, HostData>, _: (i32, i32, i32, i32, i32, i32, i32, i32, i32)| {
            Box::new(async { Ok(-1i32) })
        },
    )?;
    linker.func_wrap_async(
        F,
        "fx_session_list",
        |_: Caller<'_, HostData>, _: (i32, i32)| Box::new(async { Ok(-1i32) }),
    )?;
    linker.func_wrap_async(
        F,
        "fx_session_remove",
        |_: Caller<'_, HostData>, _: (i32, i32)| Box::new(async { Ok(-1i32) }),
    )?;
    linker.func_wrap_async(
        F,
        "fx_oauth_session_load",
        |_: Caller<'_, HostData>, _: (i32, i32, i32, i32, i32)| Box::new(async { Ok(-1i32) }),
    )?;
    linker.func_wrap_async(
        F,
        "fx_oauth_session_commit",
        |_: Caller<'_, HostData>, _: (i32, i32, i32, i32, i32, i32, i32)| {
            Box::new(async { Ok(-1i32) })
        },
    )?;
    linker.func_wrap_async(
        F,
        "fx_oauth_session_remove",
        |_: Caller<'_, HostData>, _: (i32, i32)| Box::new(async { Ok(-1i32) }),
    )?;
    Ok(())
}

/// `instantiate(options)`: starts fx's core as `fx <args>` with `env`.
/// Returns its handle and the ordered items it sends the engine.
pub(crate) async fn instantiate(
    module: &Module,
    args: &[&str],
    env: &[(String, String)],
) -> HarnessResult<(FxCore, mpsc::UnboundedReceiver<CoreItem>)> {
    let (items, receiver) = mpsc::unbounded_channel();
    let (exit, exit_receiver) = watch::channel(None);
    let shared = Arc::new(Shared {
        stdin: Mutex::default(),
        stdin_notify: Notify::new(),
        steering: Mutex::default(),
        inbound: Mutex::default(),
        outbound: Mutex::new(Outbound {
            attachments: HashMap::new(),
            next: 1,
        }),
        effects: Mutex::new(CancellationToken::new()),
        pending_tool_result: Mutex::default(),
        items,
        stderr: Mutex::default(),
        exit,
        error: Mutex::default(),
        aborted: AtomicBool::new(false),
    });
    let mut all_args = vec!["fx".to_string()];
    all_args.extend(args.iter().map(|arg| arg.to_string()));
    let data = HostData {
        shared: shared.clone(),
        memory: None,
        args: all_args,
        env: env
            .iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect(),
        output: CoreOutput::default(),
        streams: HashMap::new(),
        next_handle: 0,
    };
    let mut store = Store::new(&ENGINE, data);
    let instance = linker()?
        .instantiate_async(&mut store, module)
        .await
        .map_err(|error| {
            HarnessError::new(format!("libfx could not start the fx core: {error}"))
        })?;
    let memory = instance
        .get_memory(&mut store, "memory")
        .ok_or_else(|| HarnessError::new("The fx core exports no memory."))?;
    store.data_mut().memory = Some(memory);
    let start = instance
        .get_typed_func::<(), ()>(&mut store, "_start")
        .map_err(|error| HarnessError::new(error.to_string()))?;
    let core = FxCore {
        shared: shared.clone(),
        exit: exit_receiver,
    };
    let runner = core.clone();
    tokio::spawn(async move {
        let result = start.call_async(&mut store, ()).await;
        let finished = store.data().output.finish();
        drop(store);
        match result {
            Ok(()) => match finished {
                Ok(()) => shared.mark_exited(0),
                Err(error) => runner.abort(error),
            },
            Err(error) => {
                let message = format!("{error:#}");
                if !message.contains("proc_exit") {
                    runner.abort(message);
                } else if let Err(error) = finished {
                    runner.abort(error);
                }
                shared.mark_exited(if shared.aborted.load(Ordering::SeqCst) {
                    130
                } else {
                    1
                });
            }
        }
        shared.wake_stdin();
    });
    Ok((core, receiver))
}
