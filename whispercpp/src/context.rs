//! `Context` — the loaded whisper model.
//!
//! Owns the `whisper_context*` returned by
//! `whisper_init_from_file_with_params`. Drop calls
//! `whisper_free`. Cloning is intentionally NOT supported — the
//! underlying whisper.cpp object is a unique owned resource. To
//! run multiple inference threads against the same model, share
//! `Arc<Context>` and call [`Context::create_state`] per thread
//! (each `State` carries its own KV cache).

#![allow(unsafe_code)]

use core::{
  ptr::NonNull,
  sync::atomic::{AtomicBool, Ordering},
};
use std::{
  ffi::CString,
  path::Path,
  sync::{Arc, Mutex, MutexGuard},
};

use crate::{
  error::{WhisperError, WhisperResult},
  state::State,
  sys,
};

/// Acquire the process-wide mutex guarding every FFI call
/// that mutates ggml's global logger state.
///
/// `whisper_init_state` calls
/// `whisper_backend_init_gpu`, which unconditionally invokes
/// `ggml_log_set(g_state.log_callback, …)` — writing to
/// ggml's file-static logger globals without any
/// synchronisation. `whisper_init_from_file_with_params_no_state`
/// is in the same family (touches `g_state` indirectly through
/// backend probing). With `unsafe impl Sync for Context`, two
/// safe-Rust threads holding `Arc<Context>` could call
/// `create_state` (or `Context::new`) concurrently and race on
/// those globals — a C/C++ data race reachable from safe Rust.
///
/// The mutex serialises both init paths. Cost: one mutex
/// acquire per `Context::new` and per `create_state`. Both are
/// init-time, not hot-path; whispery's worker pool
/// pre-creates one `State` per worker at startup, so this is
/// microseconds-per-startup-once.
pub(crate) fn init_lock() -> MutexGuard<'static, ()> {
  static LOCK: Mutex<()> = Mutex::new(());
  // Recover a poisoned lock — we don't hold any state on
  // the inner ``, so re-acquiring after an unrelated panic
  // in a sibling thread is fine.
  LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Knobs forwarded to `whisper_context_default_params` before
/// loading. Mirrors the subset of `whisper_context_params` whispery
/// uses today.
///
/// All fields are private; access goes through `const fn`
/// accessors and `with_*` builder methods so the type's invariants
/// stay encapsulated and the public surface evolves
/// independently of the underlying C struct.
#[derive(Debug, Clone, Copy)]
pub struct ContextParams {
  use_gpu: bool,
  gpu_device: i32,
  flash_attn: bool,
}

impl ContextParams {
  /// Defaults: GPU on (Metal/CUDA where compiled in), device 0,
  /// flash-attn off.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      use_gpu: true,
      gpu_device: 0,
      flash_attn: false,
    }
  }

  /// Whether the encoder dispatches to a GPU backend (Metal /
  /// CUDA). On Apple Silicon: `true` is required to avoid the
  /// BLAS-only encode path that hits whisper.cpp's `failed to
  /// encode` error on `large-v3-turbo`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn use_gpu(&self) -> bool {
    self.use_gpu
  }

  /// Chained setter for [`Self::use_gpu`]. `const fn` so callers
  /// can build a `ContextParams` in `const` context (e.g. in
  /// per-runner config statics).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_use_gpu(mut self, on: bool) -> Self {
    self.use_gpu = on;
    self
  }

  /// GPU device index (default `0` = primary).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn gpu_device(&self) -> i32 {
    self.gpu_device
  }

  /// Chained setter for [`Self::gpu_device`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_gpu_device(mut self, idx: i32) -> Self {
    self.gpu_device = idx;
    self
  }

  /// Whether flash-attention is enabled. Default `false`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn flash_attn(&self) -> bool {
    self.flash_attn
  }

  /// Chained setter for [`Self::flash_attn`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_flash_attn(mut self, on: bool) -> Self {
    self.flash_attn = on;
    self
  }
}

impl Default for ContextParams {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

/// Loaded whisper.cpp model. Cheap to share via `Arc`.
pub struct Context {
  // `NonNull` (vs. `*mut`) makes the Drop impl total — there is
  // no "uninitialised" representation to guard against.
  ptr: NonNull<sys::whisper_context>,
  // bound the per-Context leak budget under
  // `WhisperError::StateLost`. A `State::full` exception
  // poisons the State (we MUST NOT free a possibly-corrupt
  // `whisper_state`) and leaks that state's native
  // allocations (~360 MB on `large-v3-turbo`). Without this
  // flag, callers retrying `create_state` on the same Context
  // accumulate one leak per attempt until the host runs out
  // of memory. With it, `create_state` short-circuits to
  // `ContextPoisoned` after the FIRST `StateLost`, capping
  // the total leak at one State per Context. Recovery
  // requires dropping this Context and constructing a fresh
  // one (model reload — slow but bounded).
  lost: AtomicBool,
  // Serialise `State::full` calls through this Context.
  // Without this lock, multiple
  // workers each holding their own `State` (the documented
  // pattern) can ALL be inside `whispercpp_full_with_state`
  // simultaneously when an OOM / system_error fires. Each
  // would poison its own state and leak ~360 MB before any
  // of them got to mark the Context lost — the per-Context
  // cap claim becomes a per-concurrent-worker cap, defeating
  // the point. Holding this mutex across the FFI call makes
  // the cap structural: at most one in-flight call per
  // Context, so at most one leaked state per Context.
  //
  // Throughput cost: serialised inference per Context. On
  // GPU backends (Metal, CUDA, Vulkan) the underlying
  // command queue is already serialised, so the cost is
  // small. On CPU-only inference, throughput drops to one
  // inference at a time per Context — callers who need
  // parallel CPU inference should run multiple Contexts
  // (each loads its own copy of the model).
  full_lock: Mutex<()>,
}

// SAFETY: whisper.cpp's context is read-only after init —
// `whisper_init_from_file_with_params` is the only mutator and
// runs entirely before we hand out the pointer. Per-thread state
// (KV cache, scratch buffers) lives in `State`, not in `Context`.
// Verified against whisper.cpp v1.8.4 (the submodule pin).
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
  /// Load a `.bin` (GGML / GGUF) model from disk.
  ///
  /// Returns [`WhisperError::ContextLoad`] when whisper.cpp could
  /// not parse the file or initialise the requested backend, or
  /// [`WhisperError::InvalidCString`] if `path` contains an
  /// interior NUL. **Panic-free.**
  pub fn new(path: impl AsRef<Path>, params: ContextParams) -> WhisperResult<Self> {
    let path_ref = path.as_ref();
    let path_str = path_ref.to_string_lossy();
    let cpath = CString::new(path_str.as_ref())
      .map_err(|_| WhisperError::InvalidCString(smol_str::SmolStr::new(path_str.as_ref())))?;

    // SAFETY: pure C call returning a value-typed defaults struct.
    let mut cparams = unsafe { sys::whisper_context_default_params() };
    cparams.use_gpu = params.use_gpu();
    cparams.gpu_device = params.gpu_device();
    cparams.flash_attn = params.flash_attn();

    // Serialise init: backend probing inside whisper.cpp
    // touches ggml's global logger state.
    let _lock = init_lock();

    // SAFETY: cpath outlives the call (held on the stack);
    // cparams is value-typed.
    //
    // We use the C++ exception-catching shim
    // `whispercpp_init_from_file_no_state`:
    // upstream allocates `std::vector` / `std::ifstream`
    // buffers that can throw `std::bad_alloc` on OOM, and
    // unwinding C++ exceptions across `extern "C"` into Rust
    // is undefined behaviour. The shim catches everything
    // and collapses to a NULL return.
    //
    // The shim itself wraps the `_no_state` form — that's
    // intentional: the default
    // `whisper_init_from_file_with_params` allocates an
    // extra ~360 MB `whisper_state` into `ctx->state` that
    // we never use (every inference path creates its own via
    // [`Context::create_state`]).
    // (`src/whisper.cpp:3735`).
    //
    // # Leak-on-OOM discrimination
    //
    // Upstream's
    // `whisper_init_from_file_with_params_no_state` does
    // `whisper_context * ctx = new whisper_context;` and
    // then performs throwing model-load work (vector
    // allocations for tensors, GPU buffer allocations on
    // Apple Silicon / CUDA, file-stream reads). If a
    // `std::bad_alloc` or `std::system_error` fires AFTER
    // the raw `new` succeeded but BEFORE the function's own
    // explicit-cleanup branches run, the partial
    // `whisper_context` and any tensor/backend buffers
    // already allocated leak — the shim catches the
    // exception but has no pointer to clean up.
    //
    // The shim keeps a thread-local sentinel that
    // distinguishes the two flavours of NULL return:
    //
    // * `take_last_constructor_exception == 0` →
    //   upstream returned NULL CLEANLY (file-not-found,
    //   wrong magic, backend refused — no `new` happened
    //   yet, or upstream's own bool-failure paths cleaned
    //   up). Surface as `ContextLoad`, retryable.
    // * `take_last_constructor_exception != 0` → the
    //   shim caught a C++ throw with the `new
    //   whisper_context` already allocated. Surface as
    //   `ConstructorLost`, NOT retryable — see that
    //   variant's docs for the recovery contract.
    let raw = unsafe { sys::whispercpp_init_from_file_no_state(cpath.as_ptr(), cparams) };

    if let Some(ptr) = NonNull::new(raw) {
      return Ok(Self {
        ptr,
        lost: AtomicBool::new(false),
        full_lock: Mutex::new(()),
      });
    }
    // SAFETY: pure C call; thread-local read on the same
    // thread that made the constructor call, with no other
    // shim entry between them.
    let exc = unsafe { sys::whispercpp_take_last_constructor_exception() };
    if exc != 0 {
      return Err(WhisperError::ConstructorLost {
        origin: "context",
        code: exc,
      });
    }
    Err(WhisperError::ContextLoad {
      path: smol_str::SmolStr::new(path_str.as_ref()),
      reason: smol_str::SmolStr::new(
        "whispercpp_init_from_file_no_state returned NULL (upstream load failure, no native exception caught)",
      ),
    })
  }

  /// Create a fresh inference [`State`] tied to this model.
  ///
  /// Takes `Arc<Self>` because the returned `State` owns a clone
  /// of the Arc — that's what keeps the Context alive across the
  /// state's lifetime without forcing callers to thread a `'ctx`
  /// borrow through every storage location. Construct
  /// `Arc::new(Context::new(...)?)` once per model, then call
  /// `create_state` per worker.
  pub fn create_state(self: &Arc<Self>) -> WhisperResult<State> {
    // refuse if a prior `State::full` on this
    // Context returned `WhisperError::StateLost`. Each
    // `StateLost` leaks the State's native allocations
    // (~360 MB on `large-v3-turbo`); allowing `create_state`
    // to allocate a fresh one would compound the leak per
    // retry attempt. Callers must drop this Context and
    // construct a fresh one (re-loading the model) to
    // recover.
    if self.lost.load(Ordering::Acquire) {
      return Err(WhisperError::ContextPoisoned);
    }
    // Serialise init: `whisper_backend_init_gpu` calls
    // `ggml_log_set(...)` on ggml's file-static logger
    // globals without any synchronisation. Two threads
    // creating states concurrently from a shared
    // `Arc<Context>` would race on those globals — a C/C++
    // data race reachable from safe Rust through
    // `unsafe impl Sync for Context`.
    let _lock = init_lock();

    // SAFETY: self.ptr is non-null (NonNull invariant) and
    // the Arc clone we hand to State keeps the Context (and
    // therefore the underlying whisper_context*) alive for
    // the State's lifetime.
    //
    // We route through the exception-catching shim
    // `whispercpp_init_state`: upstream allocates KV-cache
    // and scratch buffers via `std::vector` (each potentially
    // throws `std::bad_alloc`), and on Apple Silicon also
    // initialises the Metal backend (which can throw on
    // device-init failure).
    //
    // # NULL-discrimination contract
    //
    // Same flavour split as `Context::new`: upstream's
    // `whisper_init_state` either returns NULL via its
    // bool-failure paths (every `if (!whisper_kv_cache_init…)`
    // branch runs `whisper_free_state(state); return nullptr;`
    // before returning — leak-free) OR throws a C++
    // exception that our shim catches AFTER `new
    // whisper_state` already happened (partial leak).
    //
    // Read the thread-local sentinel to distinguish:
    // * `0`     → `StateInit` (retryable, no leak)
    // * `≠ 0`   → `ConstructorLost { origin: "state", … }`
    //   (fatal, partial allocation leaked, do not auto-retry)
    let raw = unsafe { sys::whispercpp_init_state(self.ptr.as_ptr()) };
    // TOCTOU close. Between the entry-time
    // `lost.load` above and `whispercpp_init_state` returning,
    // another thread may have transitioned an existing State
    // through `StateLost` and called `mark_lost`. If we
    // published this fresh State to the caller, they'd add
    // another leak-prone State to a Context whose poison flag
    // is now true. Re-check after the alloc; if the flag
    // flipped, free the just-created state (it's intact —
    // came straight out of `whisper_init_state`) and return
    // `ContextPoisoned`. This bounds the leak window to the
    // duration of the FFI call rather than zero, but the
    // freshly-allocated state is always freed cleanly so no
    // permanent leak accumulates.
    if self.lost.load(Ordering::Acquire) {
      if let Some(state_ptr) = NonNull::new(raw) {
        // SAFETY: `raw` is the just-returned, never-published
        // result of `whispercpp_init_state`; nothing else
        // holds it. `whisper_free_state` is the matching
        // deallocator.
        unsafe { sys::whisper_free_state(state_ptr.as_ptr()) };
      }
      // Even if the alloc threw (raw is null), drain the
      // thread-local sentinel so it doesn't leak across into
      // the next constructor call's catch-block.
      let _ = unsafe { sys::whispercpp_take_last_constructor_exception() };
      return Err(WhisperError::ContextPoisoned);
    }
    if let Some(state_ptr) = NonNull::new(raw) {
      return Ok(State::from_raw(state_ptr, Arc::clone(self)));
    }
    // SAFETY: pure C call; thread-local read on the same
    // thread, no other shim call between.
    let exc = unsafe { sys::whispercpp_take_last_constructor_exception() };
    if exc != 0 {
      // A caught constructor exception means upstream
      // `whisper_init_state` left partial native allocations
      // that we cannot reliably free (the throw could have
      // happened mid-init at any sub-call). Poison the
      // Context so subsequent
      // `create_state` calls fail with `ContextPoisoned`
      // instead of repeating the same OOM / system_error
      // path and compounding leaks.
      self.lost.store(true, Ordering::Release);
      return Err(WhisperError::ConstructorLost {
        origin: "state",
        code: exc,
      });
    }
    Err(WhisperError::StateInit)
  }

  /// Internal: hand the raw pointer to siblings in this crate
  /// that need to call FFI functions taking `whisper_context*`.
  pub(crate) fn as_raw(&self) -> *mut sys::whisper_context {
    self.ptr.as_ptr()
  }

  /// Internal: mark this Context as poisoned because a
  /// `State::full` on one of its States returned a
  /// `WhisperError::StateLost`. Subsequent
  /// [`Context::create_state`] calls return
  /// [`WhisperError::ContextPoisoned`].
  ///
  /// Idempotent: subsequent calls are cheap atomic stores.
  /// `Ordering::Release` pairs with the
  /// `Ordering::Acquire` load in `create_state` so threads
  /// observing the flag also observe everything that led up
  /// to the poisoning (per the C++ memory model: writes
  /// before a Release become visible after the matching
  /// Acquire).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn mark_lost(&self) {
    self.lost.store(true, Ordering::Release);
  }

  /// Whether [`Context::create_state`] will refuse to
  /// allocate a new [`State`]. `true` after any `State::full`
  /// on this Context has returned
  /// [`WhisperError::StateLost`]. Recovery requires dropping
  /// this Context and constructing a fresh one.
  pub fn is_poisoned(&self) -> bool {
    self.lost.load(Ordering::Acquire)
  }

  /// Acquire the per-Context inference lock for the duration
  /// of one [`State::full`] FFI call. held
  /// across the leak-prone shim entry so concurrent workers
  /// can't each leak under the same OOM event before
  /// poisoning fires. Recovers from a poisoned mutex (a
  /// previous holder panicked) by adopting the inner unit —
  /// the inner state is ``, so there's no value to be
  /// inconsistent.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn full_lock(&self) -> MutexGuard<'_, ()> {
    self
      .full_lock
      .lock()
      .unwrap_or_else(|poison| poison.into_inner())
  }

  // ── Model introspection ────────────────────────────────────

  /// `true` if the loaded checkpoint carries the multilingual
  /// decoder (e.g. `large-v3-turbo`). `false` for English-only
  /// checkpoints (`tiny.en`, `base.en`, …).
  pub fn is_multilingual(&self) -> bool {
    // SAFETY: ctx pointer invariant.
    unsafe { sys::whisper_is_multilingual(self.ptr.as_ptr()) != 0 }
  }

  /// Vocabulary size (number of tokens the decoder can emit).
  pub fn n_vocab(&self) -> i32 {
    // SAFETY: ctx pointer invariant.
    unsafe { sys::whisper_n_vocab(self.ptr.as_ptr()) }
  }

  /// Audio context window (encoder mel-frame budget). 1500 for
  /// the vanilla 30 s checkpoints.
  pub fn n_audio_ctx(&self) -> i32 {
    // SAFETY: ctx pointer invariant.
    unsafe { sys::whisper_n_audio_ctx(self.ptr.as_ptr()) }
  }

  /// Text context window (decoder past-token budget). 448 for
  /// the standard checkpoints.
  pub fn n_text_ctx(&self) -> i32 {
    // SAFETY: ctx pointer invariant.
    unsafe { sys::whisper_n_text_ctx(self.ptr.as_ptr()) }
  }

  /// Human-readable model size string baked into the checkpoint
  /// (`"tiny"`, `"base"`, `"large-v3-turbo"`, …). Returns
  /// `None` if whisper.cpp returned a NULL pointer or non-UTF-8
  /// (model corruption).
  pub fn model_type(&self) -> Option<&'static str> {
    // SAFETY: pure C accessor; pointer into a static
    // const-table baked into libwhisper.
    let raw = unsafe { sys::whisper_model_type_readable(self.ptr.as_ptr()) };
    if raw.is_null() {
      return None;
    }
    // SAFETY: NUL-terminated; static lifetime per whisper.cpp.
    let bytes = unsafe { core::ffi::CStr::from_ptr(raw).to_bytes() };
    core::str::from_utf8(bytes).ok()
  }

  // ── Special token ids ──────────────────────────────────────

  /// `<|endoftext|>` — emitted at the end of every successful
  /// decode. Useful for sentinel checks against `Token::id`.
  pub fn token_eot(&self) -> i32 {
    // SAFETY: ctx pointer invariant.
    unsafe { sys::whisper_token_eot(self.ptr.as_ptr()) }
  }

  /// `<|startoftranscript|>`.
  pub fn token_sot(&self) -> i32 {
    // SAFETY: ctx pointer invariant.
    unsafe { sys::whisper_token_sot(self.ptr.as_ptr()) }
  }

  /// First timestamp token (`<|0.00|>`). Token ids `>= token_beg`
  /// encode timestamps; `< token_beg` encode text.
  pub fn token_beg(&self) -> i32 {
    // SAFETY: ctx pointer invariant.
    unsafe { sys::whisper_token_beg(self.ptr.as_ptr()) }
  }

  /// Decode a single token id back to its surface form. Useful
  /// for token-level diagnostics. Returns `None` when:
  ///
  /// * `token` is outside `[0, n_vocab)` — would otherwise
  ///   throw `std::out_of_range` from
  ///   `id_to_token.at(token)` across the C ABI (UB) per
  ///   `whisper.cpp:4201`. Pre-checking the bound here keeps
  ///   the unwound exception from crossing `extern "C"`.
  /// * the underlying `c_str` is NULL or non-UTF-8 (model
  ///   corruption).
  ///
  /// The returned slice borrows from a `std::string` owned by
  /// the context's vocab table; it stays valid for as long as
  /// `self` is alive. (Unlike [`system_info`], this does NOT
  /// alias mutable C++ state — `id_to_token` is built once at
  /// load time and never modified.)
  pub fn token_to_str(&self, token: i32) -> Option<&str> {
    // Validate before the FFI call — the upstream `at` throw
    // would cross `extern "C"` and is UB.
    let n = self.n_vocab();
    if token < 0 || token >= n {
      return None;
    }
    // SAFETY: token bound checked above; ctx pointer invariant.
    let raw = unsafe { sys::whisper_token_to_str(self.ptr.as_ptr(), token) };
    if raw.is_null() {
      return None;
    }
    // SAFETY: NUL-terminated; lives as long as Context.
    let bytes = unsafe { core::ffi::CStr::from_ptr(raw).to_bytes() };
    core::str::from_utf8(bytes).ok()
  }
}

/// System-info string assembled by libwhisper — backend caps
/// (BLAS / Metal / CUDA / OpenMP), CPU SIMD flags whisper.cpp
/// detected, and the build id. Useful at startup-time logging
/// to confirm which backend the runtime linked against.
///
/// Returns `None` if the C++ accessor handed back a NULL pointer
/// or non-UTF-8 bytes (corrupt build).
///
/// # Soundness notes
///
/// `whisper_print_system_info` re-builds a file-scope
/// `static std::string s` on every invocation
/// (`s = ""; s += "..."; return s.c_str;`). Two unsoundness
/// problems follow that we paper over here:
///
/// 1. The `c_str` returned to a previous caller becomes
///    dangling on the next call — so we can't return
///    `&'static str`. We copy into an owned [`SmolStr`](smol_str::SmolStr).
/// 2. Two concurrent callers race on the static buffer (no
///    upstream lock). We serialise behind a Rust-side
///    [`OnceLock`](std::sync::OnceLock) AND a mutex so the
///    underlying C call runs AT MOST ONCE per process,
///    eliminating both the race AND the redundant work (the
///    system info doesn't change after libwhisper loads).
///
/// Both hazards are documented against whisper.cpp v1.8.4 at
/// `src/whisper.cpp:4315`.
pub fn system_info() -> Option<smol_str::SmolStr> {
  use std::sync::{Mutex, OnceLock};
  // OnceLock holds the cached result; the inner Mutex
  // serialises the FIRST call so two threads can't race the
  // upstream static buffer. After init, OnceLock returns
  // without locking on every call.
  static CACHE: OnceLock<Option<smol_str::SmolStr>> = OnceLock::new();
  static INIT_LOCK: Mutex<()> = Mutex::new(());
  if let Some(v) = CACHE.get() {
    return v.clone();
  }
  // Recover a poisoned mutex (matches `init_lock` and
  // `full_lock`). The inner `()` carries no state, so a panic
  // in a sibling caller can't have left anything inconsistent.
  let _guard = INIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
  // Re-check inside the lock.
  if let Some(v) = CACHE.get() {
    return v.clone();
  }
  // SAFETY: pure C accessor; the surrounding mutex prevents
  // concurrent invocations on the static `std::string s`.
  // Routed through the C++ exception-catching shim — the
  // upstream `whisper_print_system_info` rebuilds the static
  // string via `s = ""; s += "..."; s += std::to_string(…);`,
  // any of which can throw `std::bad_alloc` across the C ABI.
  //
  let raw = unsafe { sys::whispercpp_print_system_info() };
  let result = if raw.is_null() {
    None
  } else {
    // SAFETY: NUL-terminated; copy IMMEDIATELY into an owned
    // `SmolStr` so the borrow does not outlive the C call.
    let bytes = unsafe { core::ffi::CStr::from_ptr(raw).to_bytes() };
    core::str::from_utf8(bytes).ok().map(smol_str::SmolStr::new)
  };
  // Best-effort set; if a racing thread won the OnceLock between
  // our `get` checks (impossible under the mutex but defensive),
  // we just use whichever value got cached first.
  let _ = CACHE.set(result.clone());
  result
}

impl Drop for Context {
  fn drop(&mut self) {
    // SAFETY: ptr is non-null and produced by
    // whisper_init_from_file_with_params; whisper_free is the
    // matching deallocator. Called exactly once per Context.
    unsafe {
      sys::whisper_free(self.ptr.as_ptr());
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// invariant: a fresh Context starts
  /// non-poisoned. Pin the initial state so a future refactor
  /// of the `lost: AtomicBool` initialiser cannot quietly
  /// flip the contract.
  #[test]
  fn fresh_context_marker_starts_unpoisoned() {
    // We can't construct a real `Context` without a model
    // file, so build the struct directly. SAFETY (test-only):
    // the dangling pointer never crosses the FFI; we only
    // exercise the lost-flag accessors.
    let dangling = NonNull::<sys::whisper_context>::dangling();
    let ctx = Context {
      ptr: dangling,
      lost: AtomicBool::new(false),
      full_lock: Mutex::new(()),
    };
    assert!(!ctx.is_poisoned());
    ctx.mark_lost();
    assert!(ctx.is_poisoned());
    // Skip the real Drop — `whisper_free` would dereference
    // the dangling pointer.
    core::mem::forget(ctx);
  }

  /// `mark_lost` is idempotent — extra calls are cheap
  /// atomic stores, never reset the flag.
  #[test]
  fn mark_lost_is_idempotent_and_monotonic() {
    let dangling = NonNull::<sys::whisper_context>::dangling();
    let ctx = Context {
      ptr: dangling,
      lost: AtomicBool::new(false),
      full_lock: Mutex::new(()),
    };
    ctx.mark_lost();
    ctx.mark_lost();
    ctx.mark_lost();
    assert!(ctx.is_poisoned(), "stays true across repeated marks");
    core::mem::forget(ctx);
  }

  /// `mark_lost` is observable from any
  /// thread that holds an `Arc<Context>` (it's the path
  /// `State::full` uses to consult sibling poisoning before
  /// entering FFI). Stress the Acquire/Release pairing: a
  /// background thread flips the flag, the main thread
  /// observes it.
  #[test]
  fn mark_lost_visible_across_threads() {
    let dangling = NonNull::<sys::whisper_context>::dangling();
    let ctx = Arc::new(Context {
      ptr: dangling,
      lost: AtomicBool::new(false),
      full_lock: Mutex::new(()),
    });
    let ctx_b = Arc::clone(&ctx);
    let handle = std::thread::spawn(move || {
      ctx_b.mark_lost();
    });
    handle.join().unwrap();
    assert!(
      ctx.is_poisoned(),
      "the post-join Acquire load must see the spawn-side Release store"
    );
    // Skip Drop on the Arc — the dangling pointer must not
    // reach `whisper_free`. Two `forget`s, one per Arc clone
    // we manually upgraded.
    core::mem::forget(Arc::try_unwrap(ctx).ok().unwrap());
  }

  /// `full_lock` survives the documented
  /// concurrent-worker pattern. Two threads contend on the
  /// same lock, both eventually finish, neither panics. The
  /// guard's lifetime constrains the lock window so the
  /// per-Context leak cap is structural.
  #[test]
  fn full_lock_serialises_concurrent_holders() {
    let dangling = NonNull::<sys::whisper_context>::dangling();
    let ctx = Arc::new(Context {
      ptr: dangling,
      lost: AtomicBool::new(false),
      full_lock: Mutex::new(()),
    });
    let counter = Arc::new(std::sync::atomic::AtomicU32::new(0));

    let mut handles = Vec::new();
    for _ in 0..4 {
      let ctx_t = Arc::clone(&ctx);
      let counter_t = Arc::clone(&counter);
      handles.push(std::thread::spawn(move || {
        let _g = ctx_t.full_lock();
        // Inside the critical section: increment, sleep
        // briefly to provoke contention, decrement-confirm.
        let pre = counter_t.fetch_add(1, Ordering::SeqCst);
        assert_eq!(pre, 0, "another holder slipped past the mutex");
        std::thread::sleep(std::time::Duration::from_millis(2));
        let post = counter_t.fetch_sub(1, Ordering::SeqCst);
        assert_eq!(post, 1, "another holder is concurrent with us");
      }));
    }
    for h in handles {
      h.join().unwrap();
    }
    core::mem::forget(Arc::try_unwrap(ctx).ok().unwrap());
  }
}
