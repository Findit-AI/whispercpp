//! Crate-level error type.

use smol_str::SmolStr;
use thiserror::Error;

/// Result alias used throughout the crate's safe API.
pub type WhisperResult<T> = Result<T, WhisperError>;

/// Failure modes from the whisper.cpp FFI surface.
///
/// The variants are deliberately coarse — whisper.cpp itself
/// reports outcomes via integer return codes that don't carry
/// detailed semantics. We attach context strings where the C API
/// gives us nothing structured to propagate.
///
/// Diagnostic strings (paths, language hints, the truncated
/// interior-NUL slice) ride in `SmolStr` rather than `String`:
/// they are typically ≤ 23 bytes and inline, so the unhappy-path
/// allocator hit goes from "1 heap allocation per error" to "0".
#[derive(Debug, Error)]
pub enum WhisperError {
  /// `whisper_init_from_file_with_params` returned `NULL`
  /// **without** unwinding a C++ exception. The model path was
  /// wrong, the file is corrupt, or the requested backend
  /// (Metal / CoreML / CUDA) refused to initialise.
  ///
  /// **Retryable.** No partial native allocation leaked —
  /// upstream's bool-failure paths in `whisper_init_*` all run
  /// `whisper_free_state(state); return nullptr;` before
  /// returning. Callers may try a different path / backend.
  ///
  /// The exception-caught counterpart is
  /// [`ConstructorLost`](Self::ConstructorLost).
  #[error("failed to load model from {path}: {reason}")]
  ContextLoad {
    /// Path the caller passed in. Stored so logs can pinpoint
    /// which model file failed.
    path: SmolStr,
    /// Any extra context whisper.cpp surfaced (often empty —
    /// the C API just returns NULL).
    reason: SmolStr,
  },

  /// `whisper_init_state` returned `NULL` **without** unwinding
  /// a C++ exception. Usually an OOM on the compute buffers
  /// reported via the bool-returning failure path (encode
  /// allocates the largest one).
  ///
  /// **Retryable.** Upstream cleans up partials on this path
  /// (every `if (!whisper_kv_cache_init(...))` branch calls
  /// `whisper_free_state(state); return nullptr;`).
  ///
  /// The exception-caught counterpart is
  /// [`ConstructorLost`](Self::ConstructorLost).
  #[error("failed to allocate whisper state")]
  StateInit,

  /// `Context::create_state` was called on a `Context` whose
  /// previous [`State::full`](crate::State::full) returned
  /// [`StateLost`](Self::StateLost). The Context is poisoned;
  /// further state allocation would compound the per-Context
  /// leak budget.
  ///
  /// **Recovery contract.** Drop this `Context` and
  /// construct a fresh one (model reload — slow but bounded).
  /// Re-using the same `Arc<Context>` against this error
  /// without dropping it leaves the per-State leak (~360 MB
  /// on `large-v3-turbo`) in place forever; only `Drop` of
  /// the Context releases what's still freeable.
  ///
  /// **Why this exists.** `StateLost` cannot reliably free
  /// the State's native allocations (we cannot distinguish
  /// "intact" from "mid-rebuild" without upstream RAII). A
  /// retry loop creating fresh States on the same Context
  /// would leak per attempt; this variant caps the budget at
  /// one. The fix is structural — the cap survives careless
  /// callers — rather than purely documentary.
  #[error("Context was poisoned by a prior StateLost; drop and reconstruct to recover")]
  ContextPoisoned,

  /// **The native init path threw a C++ exception that our
  /// shim caught.** Either
  /// [`whispercpp_init_from_file_no_state`](crate::sys::whispercpp_init_from_file_no_state)
  /// or
  /// [`whispercpp_init_state`](crate::sys::whispercpp_init_state)
  /// returned `nullptr` AFTER catching `std::bad_alloc` /
  /// `std::system_error` / `std::exception` / unknown.
  ///
  /// **Not retryable.** Upstream `whisper_init_state` /
  /// `whisper_init_from_file_with_params_no_state` allocate
  /// raw `whisper_state` / `whisper_context` objects and
  /// per-backend buffers BEFORE doing the throwing
  /// model/backend/KV-cache work. Those locals are not
  /// RAII-owned: a caught throw partway through leaks the
  /// partial allocation (state struct + every backend / cache
  /// already initialised — typically tens to hundreds of MB
  /// per attempt).
  ///
  /// **Recovery contract.** Surface this as a fatal worker /
  /// process error. Do not auto-recreate the [`Context`](crate::Context) /
  /// [`State`](crate::State) inside a retry loop on the same process —
  /// each attempt under the same memory / system pressure
  /// leaks again. The recommended response is to escalate to
  /// the supervisor and let the worker process recycle. A
  /// future round of upstream patches that wraps the init
  /// paths in RAII would let us downgrade some of these to
  /// `ContextLoad` / `StateInit`.
  #[error(
    "whisper.cpp init threw {origin} (sentinel {code}); native partial allocation leaked, \
     not retryable"
  )]
  ConstructorLost {
    /// Which constructor caught the exception. `"context"` for
    /// the model-load path, `"state"` for the per-call state
    /// allocation.
    origin: &'static str,
    /// The shim sentinel set inside the catch block (one of
    /// `WHISPERCPP_ERR_BAD_ALLOC`, `_SYSTEM_ERROR`,
    /// `_STD_EXCEPTION`, `_UNKNOWN_EXCEPTION`).
    code: i32,
  },

  /// `whisper_full_with_state` returned a non-zero, **non-fatal**
  /// code. The state remains intact and may be reused for a
  /// fresh call.
  ///
  /// Examples: `-1` (whisper_pcm_to_mel), `-2`
  /// (whisper_set_mel), `-3..-6` (intermediate encode/decode
  /// failures whisper.cpp marks as recoverable). `-7` is **not**
  /// here — it surfaces as [`StateLost`](Self::StateLost).
  #[error("whisper_full failed with code {code}")]
  Full {
    /// The whisper.cpp return code. See `whisper.h` for the
    /// (sparse) documented values.
    code: i32,
  },

  /// **The native `whisper_state` is gone.** Either whisper.cpp
  /// freed it from underneath us (`-7`, multi-decoder KV-cache
  /// allocation failure: upstream calls `whisper_free_state`
  /// before returning), or our exception shim caught a C++
  /// throw partway through `whisper_full_with_state` (sentinels
  /// ≤ `-100`).
  ///
  /// **Not retryable on this `State`.** The Rust `State` has
  /// been poisoned: every accessor short-circuits to a safe
  /// zero/None. The native allocation IS released — either
  /// by upstream's own `whisper_free_state` on the `-7` path,
  /// or explicitly from `State::full`'s sentinel handler on
  /// caught-exception paths. The fork's idempotent
  /// `whisper_kv_cache_free` patch closed the double-free
  /// hazard that previously forced us to leak the state.
  ///
  /// **Recovery contract.** Receivers should still treat
  /// this as fatal at the worker level:
  ///
  /// 1. Do **not** call [`State::full`](crate::State::full)
  ///    again on this `State`. Drop it.
  /// 2. The parent [`Context`](crate::Context) is poisoned
  ///    too — [`Context::create_state`](crate::Context::create_state)
  ///    will return [`ContextPoisoned`](Self::ContextPoisoned)
  ///    until the Context is dropped and reconstructed. This
  ///    is defensive: the underlying pressure that caused the
  ///    throw (OOM, thread-table exhaustion, fatal backend
  ///    error) is likely still present, so retries on the
  ///    same Context would just re-fail.
  /// 3. The recommended response is to surface the error to
  ///    your supervisor, drop the Context, and reload the
  ///    model in a fresh Context once pressure has resolved.
  #[error("whisper_full lost the native state (code {code}); state freed, Context poisoned")]
  StateLost {
    /// The whisper.cpp return code or shim sentinel that
    /// triggered poisoning. `-7` = upstream KV-cache failure;
    /// `≤ -100` = exception sentinel from
    /// `whispercpp_full_with_state` (see
    /// `whispercpp_shim.h::WHISPERCPP_ERR_*`).
    code: i32,
  },

  /// A path passed to the safe API contained an interior NUL
  /// byte. The whisper.cpp C API requires NUL-terminated strings.
  #[error("argument contained an interior NUL byte: {0}")]
  InvalidCString(SmolStr),

  /// UTF-8 decode failure on a string returned from whisper.cpp
  /// (segment text or token text). The model vocabulary should
  /// always emit valid UTF-8; this would indicate a corrupt model
  /// file.
  #[error("whisper.cpp returned non-UTF-8 text: {0}")]
  Utf8(#[from] core::str::Utf8Error),

  /// Audio buffer length exceeded `i32::MAX` samples. whisper.cpp's
  /// C API takes the count as `int`. At 16 kHz this caps at
  /// ~37 hours per call — well above any realistic chunk — so this
  /// surfaces only when callers misuse the API (bytes-vs-samples
  /// confusion, accidental double-pad, etc.).
  #[error("audio buffer too large: {samples} samples > i32::MAX")]
  SamplesOverflow {
    /// The provided buffer length, for diagnostics.
    samples: usize,
  },

  /// Audio buffer was too short for whisper.cpp's mel
  /// spectrogram preprocessor.
  ///
  /// `log_mel_spectrogram` performs a reflective pad at the
  /// start of the buffer:
  /// `std::reverse_copy(samples + 1, samples + 1 + 200, …)`,
  /// so it reads `samples[1..201]`. Inputs shorter than 201
  /// samples (≈ 12.5 ms at 16 kHz) trigger an out-of-bounds
  /// read in the C++ kernel before whisper.cpp's later
  /// short-input check fires. The safe wrapper rejects them
  /// up-front instead of forwarding the UB across the FFI.
  ///
  /// Callers feeding sub-201-sample buffers should pad with
  /// silence (zeros) up to at least 201 samples, or batch the
  /// audio into longer windows upstream.
  #[error(
    "audio buffer too short: {samples} samples < {min_required} (reflective-pad lower bound)"
  )]
  SamplesTooShort {
    /// The provided buffer length, for diagnostics.
    samples: usize,
    /// Minimum samples whisper.cpp's mel preprocessor requires.
    min_required: usize,
  },

  /// A token id passed to
  /// [`Context::token_to_str`](crate::Context::token_to_str)
  /// fell outside the model's vocabulary. The C API
  /// (`whisper_token_to_str`) uses `id_to_token.at(token)`
  /// which throws `std::out_of_range`; a C++ exception across
  /// `extern "C"` is undefined behaviour. The safe wrapper
  /// validates the bound first and surfaces this error
  /// instead.
  #[error("token id {token} out of range [0, {vocab_size})")]
  TokenOutOfRange {
    /// The id the caller passed in.
    token: i32,
    /// The model's vocab size at validation time.
    vocab_size: i32,
  },
}
