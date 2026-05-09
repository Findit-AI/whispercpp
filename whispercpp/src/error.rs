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
    /// Any extra context whisper.cpp surfaced. Set to one of
    /// a small set of `&'static str` values via
    /// `SmolStr::new_static`, never `format!` — this error
    /// is reached on the `bad_alloc` / `system_error` path,
    /// and the `format!` macro plus
    /// `SmolStr::new(formatted_string)` would each
    /// heap-allocate while the global allocator is already
    /// under pressure.
    reason: SmolStr,
    /// Constructor exception sentinel from the C++ shim.
    /// `Some(WHISPERCPP_ERR_*)` when the shim caught a
    /// throw; `None` when upstream returned NULL via its
    /// own bool-failure path. Kept as a separate `Option<i32>`
    /// (instead of formatted into `reason`) so error
    /// construction allocates nothing on the
    /// caught-exception branch.
    code: Option<i32>,
  },

  /// `whisper_init_state` failed to produce a usable state.
  ///
  /// Two underlying paths land here:
  ///
  /// * Clean NULL — an internal `bool`-returning sub-init
  ///   (e.g. `whisper_kv_cache_init`, `whisper_sched_graph_init`)
  ///   reported failure. Upstream's fail branch already
  ///   ran `whisper_free_state(state); return nullptr;`.
  ///   `code = None`.
  ///
  /// * Caught C++ exception — `whisper_init_state` threw
  ///   (`std::bad_alloc`, `std::system_error`, etc.); the
  ///   `init_state RAII exit` patch in the submodule caught
  ///   it, ran `whisper_free_state(state)`, then rethrew;
  ///   the shim caught the rethrow and stored a sentinel.
  ///   `code = Some(...)`.
  ///
  /// Both paths leave **no native allocation leaked**.
  ///
  /// **Retryable.** Callers may try again immediately
  /// (recoverable system pressure) or fall back to a
  /// different backend / smaller model. The Context is NOT
  /// poisoned by this error; sibling States and subsequent
  /// `create_state` calls are unaffected.
  // The `{code:?}` formatter uses `Option<i32>`'s Debug impl,
  // which `write!`s "Some(N)" / "None" without heap allocation.
  // Earlier versions used `code.map(|c| format!(...))` inside
  // the format string; that allocates at Display time, on
  // exactly the OOM path the variant reports. Keep this
  // allocation-free at both construction and Display.
  #[error("failed to allocate whisper state (sentinel: {code:?})")]
  StateInit {
    /// Optional shim sentinel from the catch block when the
    /// init path threw (one of `WHISPERCPP_ERR_BAD_ALLOC`,
    /// `_SYSTEM_ERROR`, `_STD_EXCEPTION`,
    /// `_UNKNOWN_EXCEPTION` from `crate::sys` — the wildcard
    /// in the constant name prevents an intra-doc link, hence
    /// the verbatim spelling). `None` means the bool-failure
    /// path (no exception thrown).
    code: Option<i32>,
  },

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

  /// **The native init path threw a C++ exception WITHOUT a
  /// matching RAII cleanup.** Reserved for the case where
  /// the shim caught an exception and partial native
  /// allocations are known to be leaked.
  ///
  /// **Unreachable under the stock build.** Both
  /// constructor paths are now leak-clean:
  ///
  /// * [`Context::new`](crate::Context::new) — the
  ///   submodule's `init_context RAII exit` patch reclaims
  ///   `ctx`-tracked allocations (`model.ctxs`,
  ///   `model.buffers`, `state`), and the per-pointer
  ///   guards (`model_load RAII for raw ggml allocations`,
  ///   `model_load tensor-prep RAII`,
  ///   `model_load buffer-registration RAII`) wrap each
  ///   raw `ggml_context*` / backend buffer until
  ///   ownership is committed to a structure
  ///   `whisper_free` walks. Every caught exception
  ///   surfaces as [`ContextLoad`](Self::ContextLoad)
  ///   with the sentinel embedded in `reason`.
  ///
  /// * [`Context::create_state`](crate::Context::create_state)
  ///   — `whisper_init_state`'s `init_state RAII exit`
  ///   patch is genuinely complete: every allocation lives
  ///   inside the captured `state`, and
  ///   `whisper_free_state(state)` walks every field
  ///   (the `kv_cache_free idempotent fix` plus
  ///   default-zero scalars in `whisper_state` make
  ///   partial-state cleanup safe). Caught exceptions
  ///   surface as
  ///   [`StateInit { code: Some(...) }`](Self::StateInit).
  ///
  /// The variant is retained for two reasons:
  ///
  /// 1. **Defensive type for future regressions.** A
  ///    future upstream rebase that drops one of the
  ///    `*_RAII_exit` patches or the per-pointer guards
  ///    can re-enable this classification without an API
  ///    change.
  /// 2. **Out-of-tree builds.** Consumers patching their
  ///    own submodule may disable some guards; the variant
  ///    keeps the leak-tainted recovery contract
  ///    available.
  ///
  /// **Recovery contract (when produced).** Surface as a
  /// fatal worker / process error. Do not auto-recreate
  /// the [`Context`](crate::Context) /
  /// [`State`](crate::State) in a retry loop on the same
  /// process — each attempt under the same pressure leaks
  /// again. Escalate and recycle the worker.
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

  /// A Rust-side allocation on the FFI input copy path failed.
  /// Surfaced by public setters that copy unbounded
  /// caller-controlled data (`set_initial_prompt`,
  /// `set_tokens`) — they pre-scan for invariants and use
  /// `Vec::try_reserve_exact` so allocation pressure becomes
  /// a typed error instead of a process abort.
  ///
  /// **Retryable** in the same sense as
  /// [`StateInit`](Self::StateInit): the failure is system
  /// state, not data, and a smaller payload may succeed.
  #[error("rust-side allocation failed for {context}")]
  AllocationFailed {
    /// Which input the wrapper was trying to copy when the
    /// allocator returned an error. Static string so the
    /// error variant carries no further heap allocation.
    context: &'static str,
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

  /// A safe-API setter rejected an input that exceeded its
  /// length cap. Distinct from [`InvalidCString`](Self::InvalidCString)
  /// (which is specifically about interior NUL bytes), so callers
  /// can distinguish "your string is too long" from "your string
  /// has a NUL embedded in it" without parsing error messages.
  ///
  /// The payload carries:
  /// * a UTF-8 prefix of the offending input (truncated to a
  ///   bounded head — typically ≤ 64 chars, ≤ 256 bytes — so
  ///   `Display` can't be turned into log amplification by
  ///   attacker-controlled values);
  /// * `len` — the raw input length in bytes;
  /// * `cap` — the cap the setter enforced.
  ///
  /// Currently emitted by:
  /// * [`Params::set_language`](crate::Params::set_language) —
  ///   cap is 32 bytes (matches [`crate::lang_id_for`]'s cap);
  /// * [`Params::set_initial_prompt`](crate::Params::set_initial_prompt)
  ///   — cap is 1 MiB.
  #[error("input exceeds length cap ({len} > {cap}): {head:?}")]
  InputTooLong {
    /// Bounded UTF-8 head of the offending input (for diagnostics).
    head: SmolStr,
    /// Raw byte length of the input.
    len: usize,
    /// The cap the setter enforced.
    cap: usize,
  },

  /// A language hint passed to
  /// [`Params::set_language`](crate::Params::set_language)
  /// was not recognised by whisper.cpp's `whisper_lang_id`
  /// lookup. `whisper_full_with_state` would otherwise
  /// silently fall back to id `-1` and push
  /// `whisper_token_lang(ctx, -1)` into the decoder prompt
  /// — producing wrong transcripts with no error signal.
  /// The wrapper validates at the setter so a config typo
  /// surfaces as `Err` immediately.
  ///
  /// The empty string `""` and `"auto"` are accepted as
  /// "no language hint / auto-detect" sentinels and do
  /// NOT produce this error.
  #[error(
    "unknown language hint {0:?}; expected ISO-639-1 short code, English name, \"\", or \"auto\""
  )]
  UnknownLanguage(SmolStr),

  /// A language hint was recognised by whisper.cpp's
  /// process-global `g_lang` table but is NOT supported by
  /// the loaded model. `whisper_token_lang(ctx, lang_id)`
  /// computes `token_sot + 1 + lang_id` without bounds-
  /// checking against the model's actual language-token
  /// range — for a checkpoint with fewer language tokens
  /// than `g_lang` has entries (e.g. `g_lang` includes
  /// `yue` at id 99 but the model has only 99 language
  /// tokens at ids 0..=98), the resulting token lands on
  /// `token_translate` / `token_transcribe` / further
  /// special-token slots instead of a real language token.
  ///
  /// The decode then runs with a corrupted SOT-style
  /// prefix (the token says "translate to English" or
  /// "transcribe" instead of "language X"), producing
  /// wrong-language or task-biased transcripts with no
  /// other signal. The wrapper validates that the
  /// resolved token sits in `(token_sot, token_translate)`
  /// and reports this error instead.
  #[error(
    "language {0:?} is not supported by the loaded model — \
     resolved token falls outside the language-token range"
  )]
  LanguageNotSupportedByModel(SmolStr),

  /// `Params::set_offset_ms` / `set_duration_ms` requested an
  /// audio range (`offset_ms .. offset_ms + duration_ms`)
  /// extending past the actual audio length. Upstream
  /// `whisper_full_with_state` doesn't bound `seek_end` to
  /// the available mel length — it computes
  /// `seek_end = seek_start + duration_ms / 10` and runs
  /// the encode/decode loop over that range, with
  /// `whisper_encode_internal` zero-filling reads past the
  /// real input. A `duration_ms = i32::MAX` slip on a
  /// short sample buffer therefore drives ~71 000
  /// 30-second windows of zero-padded decode work — looks
  /// like a hung inference rather than a clean parameter
  /// error.
  ///
  /// The wrapper validates at `State::full` entry where
  /// both `samples.len()` and `Params::offset_ms` /
  /// `duration_ms` are visible.
  #[error(
    "duration_ms range out of bounds: offset_ms={offset_ms}, duration_ms={duration_ms}, \
     audio_duration_ms={audio_duration_ms}"
  )]
  InvalidDuration {
    /// The configured start offset (ms).
    offset_ms: i32,
    /// The configured duration (ms). 0 means "to end of input"
    /// and is always accepted.
    duration_ms: i32,
    /// The actual audio length in ms (`samples.len() * 1000
    /// / 16000`, integer-truncated).
    audio_duration_ms: i64,
  },

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
