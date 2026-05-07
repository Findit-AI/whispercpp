//! `Params` — the configuration passed to a single
//! [`State::full`](crate::State::full) call.
//!
//! # Ownership model
//!
//! `Params` owns every `CString` it hands to whisper.cpp. The
//! crate's whole point — fix the leak class in whisper-rs's
//! `set_initial_prompt` / `set_language` — depends on this. Each
//! setter that takes a string stores the `CString` in the
//! `Params` struct and replaces the pointer in the FFI struct
//! with `as_ptr`. When `Params` drops, the strings drop with
//! it.
//!
//! # Abort callback
//!
//! `Params` owns the abort closure as
//! `Box<UnsafeCell<Box<dyn FnMut() -> bool>>>` — the [`AbortCallback`]
//! type alias. The outer `Box` gives a heap-stable address so
//! the FFI `user_data` pointer survives `Params` moves; the
//! `UnsafeCell` legitimises the `&mut`-from-`&` borrow the
//! trampoline takes through the C ABI; the inner
//! `Box<dyn FnMut() -> bool>` is the type-erased closure shape
//! whisper.cpp's callback expects. The whisper-rs UB we
//! diagnosed in earlier work (`*mut F` cast to read a
//! `*mut Box<dyn FnMut>` fat-pointer-on-heap) is structurally
//! absent: the trampoline only ever reads through the
//! `UnsafeCell`'s known layout.
//!
//! # Panic-free
//!
//! Every setter returns `Result` if it can fail (interior NUL in
//! a string), and field-only setters are infallible chained
//! returns of `&mut Self`. There is no `expect`/`unwrap`/`panic!`
//! anywhere in this module's safe surface.

#![allow(unsafe_code)]

use core::{cell::UnsafeCell, ffi::c_void};
use std::{
  ffi::CString,
  panic::{AssertUnwindSafe, catch_unwind},
};

/// Boxed-in-`UnsafeCell`-boxed-`FnMut` storage for the abort
/// callback. The outer `Box` gives a stable heap address that
/// survives `Params` moves; `UnsafeCell` legitimises the
/// `&mut`-from-`&` access the trampoline performs; the inner
/// `Box<dyn FnMut() -> bool>` is the type-erased closure shape
/// `set_abort_callback` accepts. Aliased so the
/// `_abort_callback` field stays readable.
type AbortCallback = Box<UnsafeCell<Box<dyn FnMut() -> bool>>>;

/// Upper bound applied by [`Params::set_n_threads`] and
/// [`Params::new`].
///
/// **Capped at `1`** — the only value provably safe under
/// whisper.cpp's current threading patterns. Two distinct
/// `vector<std::thread>` use sites combine to require this:
///
/// * **Mel-spectrogram parallelism** (`whisper.cpp:3212-3217`)
///   spawns workers in a loop with no caller-thread work
///   between iterations. For this site alone, `n = 2` was
///   safe (single iteration, atomic success/fail).
/// * **Multi-decoder process loop**
///   (`whisper.cpp:7233-7242` and `:7483-7493`) spawns
///   `threads[0..n-2]` then runs `process` on the CALLER
///   thread before joining. With `n = 2`: `threads[0]` is
///   joinable, then `process` runs and can throw
///   (vector pushes inside `whisper_sample_token_topk`,
///   `whisper_process_logits`, etc. → `std::bad_alloc`).
///   On throw, stack unwinds → `vector<std::thread>`
///   destructor destroys `threads[0]` while joinable →
///   `std::terminate` BEFORE our shim's `catch (...)` can
///   run. This path is reachable from safe Rust via
///   `Greedy { best_of ≥ 2 }` or `BeamSearch`.
///
/// `n = 1` short-circuits the multi-decoder branch
/// (`if (n_threads == 1) { process; }` upstream), spawns
/// no mel workers, and is the only value that doesn't reach
/// any abort path.
///
/// History of this constant across review rounds:
/// `1024 → 64 → 16 → 2 → 1`. Each step closed a previously-
/// unanalysed thread-spawn or caller-thread-throw shape. A
/// real fix to allow `n ≥ 2` needs upstream RAII join guards
/// or per-region exception catches (unfixed bug in
/// whisper.cpp itself). When that lands, this constant can
/// be raised.
///
/// Callers who can prove host headroom themselves can opt
/// into higher counts via
/// [`Params::set_n_threads_unchecked`] — `unsafe`, with the
/// caller's safety contract that no `std::thread` constructor
/// AND no `process` invocation will throw under the
/// workload's pressure.
pub const MAX_N_THREADS: i32 = 1;

/// Upper bound applied by [`SamplingStrategy::BeamSearch`]'s
/// `beam_size` and [`SamplingStrategy::Greedy`]'s `best_of`
/// before they reach `whisper_full_params`.
///
/// `whisper.cpp`'s `whisper_sample_token_topk` indexes
/// `beam_candidates[0]` and forms an iterator from
/// `vector::begin`; `k <= 0` collapses both into invalid
/// memory access. Upstream's only internal
/// guard clamps the *decoder count* to ≥ 1 — the original
/// `beam_size` flows through untouched. We clamp here to
/// `[1, MAX_BEAM_SIZE]` so safe Rust cannot reach the C++ UB.
///
/// `64` is generous: empirical work on Whisper rarely
/// exceeds `beam_size = 5` (OpenAI's default) and quality
/// saturates by 8–16; the cap is a sanity ceiling, not a
/// tuning knob.
///
/// Multi-decoder safety relies on the kv_cache_free
/// idempotent patch the `whispercpp-sys` build script
/// applies — that's why the system /
/// pkg-config link path was removed:
/// linking against a stock unpatched libwhisper would
/// silently restore the double-free this cap depends on.
pub const MAX_BEAM_SIZE: i32 = 64;

/// Clamp a candidate-count knob (`beam_size` / `best_of`) to
/// the safe `[1, MAX_BEAM_SIZE]` range. `const fn` so it can
/// run inside `Params::new`'s match.
#[cfg_attr(not(tarpaulin), inline(always))]
const fn clamp_topk(k: i32) -> i32 {
  if k < 1 {
    1
  } else if k > MAX_BEAM_SIZE {
    MAX_BEAM_SIZE
  } else {
    k
  }
}

/// Clamp `n_threads` to `[1, MAX_N_THREADS]`.
///
/// Used both by [`Params::set_n_threads`] (caller-supplied
/// values) AND by [`Params::new`] (the value
/// `whisper_full_default_params` inherits from
/// `std::min(4, hardware_concurrency)` — `hardware_concurrency`
/// is allowed by the C++ spec to return `0` on hosts where it
/// can't determine the count, which would propagate `0 - 1`
/// underflow into the upstream
/// `vector<std::thread>(n_threads - 1)` constructor).
#[cfg_attr(not(tarpaulin), inline(always))]
const fn clamp_n_threads(n: i32) -> i32 {
  if n < 1 {
    1
  } else if n > MAX_N_THREADS {
    MAX_N_THREADS
  } else {
    n
  }
}

/// Hard ceiling for [`Params::set_max_initial_ts`], **in seconds**.
///
/// `max_initial_ts` is whisper.cpp's biased timestamp ceiling
/// for the FIRST segment of a chunk — it suppresses logits at
/// timestamp tokens beyond `tid0 = round(max_initial_ts /
/// precision)` where `precision = WHISPER_CHUNK_SIZE /
/// n_audio_ctx ≈ 0.02 s/frame` (`whisper.cpp:6604-6610`).
/// OpenAI's reference uses `1.0` second (`decoding.py:L426`).
///
/// `30.0` matches the model-native chunk width (30 s); any
/// value at or below the chunk width is well-defined. Above
/// that, `tid0` walks past the timestamp-token range and the
/// bias loop becomes a no-op — not unsound, but the knob loses
/// meaning. We cap at the chunk width so the value the safe
/// API forwards always has a legitimate effect.
///
/// NaN, ±∞, and negatives collapse to `0.0` (the upstream
/// "ignore" sentinel — see `if (max_initial_ts > 0.0)` at
/// `whisper.cpp:6604`) because the
/// `round(t / precision) → int` conversion is UB on
/// non-finite or extreme floats.
pub const MAX_INITIAL_TS_S: f32 = 30.0;

/// Hard ceiling applied by [`Params::set_temperature`].
///
/// Whisper's softmax-temperature contract is `t ∈ [0.0, 1.0]`
/// (1.0 = uniform-over-vocab). Values above 1.0 still type-
/// check inside whisper.cpp but produce no useful sampling
/// behaviour, so we cap there. The cap also bounds upstream's
/// `for (float t = temperature; t < 1.0 + 1e-6; t += inc)`
/// ladder so a `temperature = f32::MAX` start can't blow past
/// the comparison.
pub const MAX_TEMPERATURE: f32 = 1.0;

/// Smallest positive `temperature_inc` that
/// [`Params::set_temperature_inc`] will forward to the ladder.
///
/// `1e-3` is well above `ULP(1.0) ≈ 1.19e-7` (the precision
/// floor where `t += inc` stops advancing in `float`), and
/// also bounds the ladder length: from `temperature = 0.0` to
/// `1.0`, the longest legal ladder has `1.0 / 1e-3 ≈ 1000`
/// entries — comfortably below any allocation worry. The
/// upstream OpenAI default is `0.2`; whispery's runner pins
/// `inc = 0.0` for deterministic behaviour. Anything in
/// between is fine; anything below this floor (or NaN /
/// negative) clamps DOWN to `0.0` ("no ladder").
pub const MIN_TEMPERATURE_INC: f32 = 1e-3;

/// Clamp an `f32` timestamp index to a finite, in-range
/// value the upstream `round` → `int` conversion can
/// safely consume. `const fn` because `f32::is_nan` /
/// `is_finite` are stable in const since Rust 1.83.
#[cfg_attr(not(tarpaulin), inline(always))]
const fn clamp_max_initial_ts(t: f32) -> f32 {
  // NaN, ±∞, and negatives all collapse to 0.0 (the
  // upstream "ignore" sentinel — see `if (max_initial_ts >
  // 0.0)` guard at `whisper.cpp` line ~7829). The ceiling
  // covers the legitimate-but-extreme f32::MAX case.
  if !t.is_finite() || t < 0.0 {
    0.0
  } else if t > MAX_INITIAL_TS_S {
    MAX_INITIAL_TS_S
  } else {
    t
  }
}

/// Clamp the per-attempt decoding temperature to a finite,
/// in-range value the upstream ladder can safely loop over.
#[cfg_attr(not(tarpaulin), inline(always))]
const fn clamp_temperature(t: f32) -> f32 {
  // NaN / -∞ / negatives → 0.0 (single-attempt at
  // greedy / argmax). +∞ / huge → MAX_TEMPERATURE.
  if !t.is_finite() || t < 0.0 {
    0.0
  } else if t > MAX_TEMPERATURE {
    MAX_TEMPERATURE
  } else {
    t
  }
}

/// Clamp the temperature-ladder step to either `0.0` ("no
/// ladder, single attempt") or a value large enough that
/// `t += inc` actually advances `t` once it nears the
/// `1.0 + 1e-6` upstream sentinel.
#[cfg_attr(not(tarpaulin), inline(always))]
const fn clamp_temperature_inc(inc: f32) -> f32 {
  // NaN / negatives / subnormal-positive → 0.0. Upstream
  // treats `inc <= 0.0` as "single attempt" (`temperature_inc
  // > 0.0f` guard at whisper.cpp:6845), so we don't need a
  // separate sentinel.
  if !inc.is_finite() || inc < MIN_TEMPERATURE_INC {
    0.0
  } else if inc > 1.0 {
    1.0
  } else {
    inc
  }
}

use crate::{
  error::{WhisperError, WhisperResult},
  sys,
};

/// Sampling strategy. Mirrors `whisper_sampling_strategy`.
#[derive(Debug, Clone, Copy)]
pub enum SamplingStrategy {
  /// Greedy / argmax decoding with optional best-of resampling.
  ///
  /// `best_of > 1` activates whisper.cpp's multi-decoder
  /// path, which recreates the KV cache when the decoder
  /// count grows. Under allocation pressure that path can
  /// throw between freeing the old cache and rebuilding the
  /// new one, leaving the C-side `whisper_state`'s `kv_self`
  /// freed while `state` itself is still live. The Rust
  /// `State` poisons itself on every
  /// shim exception sentinel — accessors safely return
  /// zero/None — but you cannot recover the in-flight
  /// `full` call. Stick to `best_of: 1` if you need
  /// guaranteed forward progress under OOM.
  Greedy {
    /// Number of independent decoding attempts at each
    /// temperature; the highest-scoring is kept. 1 = pure
    /// greedy. See the type-level note about multi-decoder
    /// OOM behaviour.
    best_of: i32,
  },
  /// Beam-search decoding.
  ///
  /// Always activates the multi-decoder path (see the
  /// `Greedy` doc-note). Same OOM caveat applies: if
  /// allocation fails inside the KV-cache rebuild, the
  /// `State` poisons itself and you must construct a fresh
  /// one to retry. For workloads where OOM is a credible
  /// failure mode, prefer `Greedy { best_of: 1 }`.
  BeamSearch {
    /// Number of beams kept per step.
    beam_size: i32,
    /// Beam patience hyperparameter; -1 disables.
    patience: f32,
  },
}

/// Builder + storage for `whisper_full_params`. Construct via
/// [`Params::new`], chain setters, then pass an immutable
/// reference to [`State::full`](crate::State::full).
pub struct Params {
  raw: sys::whisper_full_params,
  // Stored CStrings keep the pointers in `raw` valid for the
  // entire `Params` lifetime. Drop order: `raw` is plain data,
  // these are dropped after the struct is unlinked from any
  // FFI call (caller is required to ensure no in-flight `full`
  // observes us mid-drop — enforced by `&Params` borrow on
  // `State::full`).
  _initial_prompt: Option<CString>,
  _language: Option<CString>,
  // Owned prompt-token buffer kept alongside `raw.prompt_tokens`
  // (which carries `&[whisper_token]` as a raw pointer). Lifetime
  // ties to `Params` like the CStrings above.
  _prompt_tokens: Option<Vec<sys::whisper_token>>,
  // Boxed abort closure, wrapped in `UnsafeCell` so the
  // trampoline can `&mut`-call it through a shared `&Params`
  // borrow without violating Rust's aliasing rules. The Box
  // gives us a stable address that survives `Params` moves;
  // `UnsafeCell` legitimises the interior mutability the
  // trampoline performs.
  //
  // `Params` itself stays `!Sync`-by-default because
  // `UnsafeCell` removes the auto-Sync impl — that matches
  // `whisper.cpp`'s contract: a single `Params` may not be
  // shared between two concurrent `State::full` calls.
  _abort_callback: Option<AbortCallback>,
}

impl Params {
  /// Build a fresh `Params` for the given strategy. Defaults are
  /// whisper.cpp's `whisper_full_default_params(strategy)`.
  pub fn new(strategy: SamplingStrategy) -> Self {
    let cstrategy = match strategy {
      SamplingStrategy::Greedy { .. } => sys::whisper_sampling_strategy_WHISPER_SAMPLING_GREEDY,
      SamplingStrategy::BeamSearch { .. } => {
        sys::whisper_sampling_strategy_WHISPER_SAMPLING_BEAM_SEARCH
      }
    };
    // SAFETY: pure C call returning a value-typed defaults
    // struct.
    let mut raw = unsafe { sys::whisper_full_default_params(cstrategy as _) };
    // Clamp `best_of` / `beam_size` to `[1, MAX_BEAM_SIZE]`.
    // Both feed into upstream `whisper_sample_token_topk` and
    // related candidate-vector indexing, where `k <= 0` is
    // OOB / iterator-before-begin. The clamp
    // is silent because the legitimate use case for any of
    // these knobs is `1..=8`; values outside that range are
    // programming errors that we'd rather convert to "still
    // works" than to a C++ abort across `extern "C"`.
    match strategy {
      SamplingStrategy::Greedy { best_of } => {
        raw.greedy.best_of = clamp_topk(best_of);
      }
      SamplingStrategy::BeamSearch {
        beam_size,
        patience,
      } => {
        raw.beam_search.beam_size = clamp_topk(beam_size);
        raw.beam_search.patience = patience;
      }
    }
    // Sanitize the C-supplied default `n_threads` before
    // anyone can call `State::full` against fresh `Params`.
    // `whisper_full_default_params` derives this from
    // `std::min(4, hardware_concurrency)`; the C++ spec
    // allows `hardware_concurrency` to return `0`, which
    // would propagate `0 - 1` underflow into the mel path's
    // `vector<thread>(n - 1)` constructor before any caller
    // had a chance to invoke `set_n_threads`.
    raw.n_threads = clamp_n_threads(raw.n_threads);
    Self {
      raw,
      _initial_prompt: None,
      _language: None,
      _prompt_tokens: None,
      _abort_callback: None,
    }
  }

  // ── String setters (fallible: interior NUL → InvalidCString). ──

  /// Provide a language hint (e.g. `"en"`, `"zh"`, `"auto"`).
  /// Stores the `CString` for the lifetime of `self` — fixing the
  /// `whisper-rs` leak.
  ///
  /// Returns [`WhisperError::InvalidCString`] if `lang` contains
  /// an interior NUL byte. **Panic-free.**
  pub fn set_language(&mut self, lang: &str) -> WhisperResult<&mut Self> {
    let cstr =
      CString::new(lang).map_err(|_| WhisperError::InvalidCString(smol_str::SmolStr::new(lang)))?;
    self.raw.language = cstr.as_ptr();
    self._language = Some(cstr);
    Ok(self)
  }

  /// Set the initial prompt (`<|prompt|>` text, decoded by the
  /// model before generation). Owns the `CString`.
  ///
  /// Returns [`WhisperError::InvalidCString`] on interior NUL.
  /// **Panic-free.**
  pub fn set_initial_prompt(&mut self, prompt: &str) -> WhisperResult<&mut Self> {
    let cstr = CString::new(prompt).map_err(|_| {
      // The prompt may be very long; trim the diagnostic so the
      // error doesn't drag a kilobyte of audio context into log
      // tails. SmolStr inlines short captures (≤23 bytes); a 64-
      // char head usually allocates but stays bounded.
      let head: String = prompt.chars().take(64).collect();
      WhisperError::InvalidCString(smol_str::SmolStr::new(head))
    })?;
    self.raw.initial_prompt = cstr.as_ptr();
    self._initial_prompt = Some(cstr);
    Ok(self)
  }

  // ── Primitive setters (infallible chained `&mut Self`). ──

  /// Whether to detect language from the audio (overrides
  /// `set_language`'s hint).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_detect_language(&mut self, on: bool) -> &mut Self {
    self.raw.detect_language = on;
    self
  }

  /// Number of CPU threads for the encode/decode loop.
  ///
  /// Clamped to `[1, MAX_N_THREADS]` (see [`MAX_N_THREADS`]
  /// for the per-`n` safety analysis). Callers who know the
  /// host has sufficient thread-table headroom can opt into
  /// higher counts via [`Self::set_n_threads_unchecked`] —
  /// that's an `unsafe fn`, with the safety contract that the
  /// caller asserts no `std::thread` constructor will throw
  /// under the workload's pressure.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_n_threads(&mut self, n: i32) -> &mut Self {
    self.raw.n_threads = clamp_n_threads(n);
    self
  }

  /// Set `n_threads` without applying [`MAX_N_THREADS`]'s
  /// upper bound.
  ///
  /// Negative / zero `n` are still clamped up to `1` because
  /// `n ≤ 0` triggers the upstream `vector<thread>(n - 1)`
  /// underflow path (a different bug class from the
  /// thread-spawn-throw abort that `MAX_N_THREADS` guards
  /// against). Inputs `≥ 1` pass through verbatim.
  ///
  /// # Safety
  ///
  /// The caller must guarantee that, in the runtime
  /// environment where [`State::full`](crate::State::full)
  /// will execute, no `std::thread` constructor inside
  /// whisper.cpp's mel-spectrogram worker loop can fail.
  /// Practically that means:
  ///
  /// * the host's per-process thread limit (`ulimit -u` on
  ///   POSIX, container PIDs cgroup, Windows job-object
  ///   limits) admits at least `n_threads - 1` more threads
  ///   than the process has already spawned, AND
  /// * sufficient address space and TLS reserve are
  ///   available for each new thread.
  ///
  /// If a `std::thread` constructor throws partway through
  /// the loop, whisper.cpp's `vector<std::thread>` destructor
  /// destroys joinable threads during stack unwinding, which
  /// invokes `std::terminate` BEFORE our exception shim
  /// can convert the throw into a [`WhisperError`]. The
  /// process aborts. **No Rust-level recovery is possible.**
  ///
  /// Using this function with values outside the
  /// `[1, MAX_N_THREADS]` range therefore trades the safe
  /// API's "guaranteed-no-process-termination" property for
  /// runtime parallelism the safe ceiling forbids. ggml's
  /// internal planner usually caps useful parallelism in
  /// the single-digit range, so values much beyond `~8` rarely
  /// improve performance.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const unsafe fn set_n_threads_unchecked(&mut self, n: i32) -> &mut Self {
    // Lower bound stays — this guards a different bug
    // (size_t::MAX underflow in `vector<thread>(n - 1)`)
    // that's not in the unsafe contract.
    self.raw.n_threads = if n < 1 { 1 } else { n };
    self
  }

  /// Set `BeamSearch::beam_size` without applying
  /// [`MAX_BEAM_SIZE`]'s upper bound.
  ///
  /// `MAX_BEAM_SIZE = 64` is already generous (Whisper
  /// quality saturates by `beam_size = 8–16` per OpenAI's
  /// own work), so the safe API covers the realistic range.
  /// This unchecked setter exists for diagnostic /
  /// stress-test scenarios that legitimately need a value
  /// past the cap.
  ///
  /// Mirrors [`Self::set_n_threads_unchecked`]: the lower
  /// bound of `1` is preserved (guards `topk` underflow into
  /// `vector::begin` iteration in
  /// `whisper_sample_token_topk`); only the upper cap is
  /// bypassed.
  ///
  /// # Safety
  ///
  /// The cap exists as a sanity ceiling, not a memory-safety
  /// requirement (the multi-decoder OOM double-free that
  /// motivated previous caps is patched at build
  /// time — see `whispercpp-sys/build.rs::PATCHES`). Going
  /// past the cap therefore trades the safe API's "no silly
  /// allocations" property for the right to over-allocate
  /// candidate tables on hosts with the memory budget for it.
  ///
  /// `n ≤ 0` still clamps to `1` (separate underflow bug
  /// outside the unsafe contract).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const unsafe fn set_beam_size_unchecked(&mut self, n: i32) -> &mut Self {
    self.raw.beam_search.beam_size = if n < 1 { 1 } else { n };
    self
  }

  /// Set `Greedy::best_of` without applying [`MAX_BEAM_SIZE`]'s
  /// upper bound. Same safety contract as
  /// [`Self::set_beam_size_unchecked`] — see that function's
  /// docs.
  ///
  /// `n ≤ 0` still clamps to `1`.
  ///
  /// # Safety
  ///
  /// See [`Self::set_beam_size_unchecked`]. Bypassing the cap
  /// trades the safe API's "no silly allocations" property
  /// for the right to over-allocate candidate tables; the
  /// memory-safety properties of multi-decoder paths still
  /// hold ('s idempotent `whisper_kv_cache_free` is
  /// always applied).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const unsafe fn set_best_of_unchecked(&mut self, n: i32) -> &mut Self {
    self.raw.greedy.best_of = if n < 1 { 1 } else { n };
    self
  }

  /// Disable transcript prompting from the previous segment's
  /// tokens (matches `whisper-rs`'s `set_no_context`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_no_context(&mut self, on: bool) -> &mut Self {
    self.raw.no_context = on;
    self
  }

  /// `no_speech_prob` threshold. Segments above this are flagged
  /// as silence and may be retried at higher temperature.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_no_speech_thold(&mut self, t: f32) -> &mut Self {
    self.raw.no_speech_thold = t;
    self
  }

  /// Decoding temperature for this single attempt. See
  /// [`Self::set_temperature_inc`] for the internal ladder.
  ///
  /// Clamped to `[0.0, MAX_TEMPERATURE]`. NaN, ±∞, and
  /// negatives collapse to `0.0`. Upstream's fallback ladder
  /// loop is `for (float t = temperature; t < 1.0 + 1e-6;
  /// t += inc)`; passing `temperature = -∞` would trip the
  /// comparison forever and `push_back` into a `vector` until
  /// OOM.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_temperature(&mut self, t: f32) -> &mut Self {
    self.raw.temperature = clamp_temperature(t);
    self
  }

  /// Internal temperature-ladder step. `0.0` pins the decoder
  /// to exactly one attempt at `temperature`.
  ///
  /// Clamped to `{0.0} ∪ [MIN_TEMPERATURE_INC, 1.0]`:
  /// * NaN / negative / `(0.0, MIN_TEMPERATURE_INC)` → `0.0`
  ///   (treated as "ladder disabled, single attempt").
  /// * `> 1.0` → `1.0`.
  ///
  /// Upstream loops
  /// `for (float t = temperature; t < 1.0 + 1e-6; t += inc)`.
  /// With a positive `inc` smaller than `ULP(1.0) ≈ 1.19e-7`,
  /// `t += inc` does not advance once `t` reaches `1.0` — the
  /// loop spins forever, pushing floats into a vector until
  /// OOM. Clamping subnormal positive `inc` up to `0.0`
  /// (= "no ladder") closes that path while preserving the
  /// common shapes (`inc = 0.0` for single-attempt, `inc =
  /// 0.2` for the OpenAI-default 5-step ladder).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_temperature_inc(&mut self, inc: f32) -> &mut Self {
    self.raw.temperature_inc = clamp_temperature_inc(inc);
    self
  }

  /// Suppress empty output bias.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_suppress_blank(&mut self, on: bool) -> &mut Self {
    self.raw.suppress_blank = on;
    self
  }

  /// Suppress non-speech tokens.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_suppress_nst(&mut self, on: bool) -> &mut Self {
    self.raw.suppress_nst = on;
    self
  }

  /// Toggles every `print_*` field off in one call. Whisper.cpp
  /// otherwise scribbles to stdout/stderr during decode, which
  /// is rarely what production callers want.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn silence_print_toggles(&mut self) -> &mut Self {
    self.raw.print_special = false;
    self.raw.print_progress = false;
    self.raw.print_realtime = false;
    self.raw.print_timestamps = false;
    self
  }

  /// Install an abort callback. Whisper.cpp invokes it during
  /// the encode loop; returning `true` causes `whisper_full` to
  /// bail out early.
  ///
  /// The closure is stored as `Box<Box<dyn FnMut -> bool>>` —
  /// a stable address whose layout matches the C trampoline
  /// installed under the hood. This is the structural fix for
  /// the whisper-rs `set_abort_callback_safe` UB.
  pub fn set_abort_callback<F>(&mut self, f: F) -> &mut Self
  where
    F: FnMut() -> bool + 'static,
  {
    // ordering: the previous implementation
    // first published a fresh `user_data` pointer into
    // `self.raw`, then assigned `self._abort_callback =
    // Some(outer)`. Replacement-assignment drops the OLD
    // `Some(outer)`; if its captured closure's `Drop` panics
    // and a caller has wrapped this setter in `catch_unwind`,
    // the unwind tears down the NEW `outer` while
    // `self.raw.abort_callback_user_data` still points at it
    // — leaving `Params` with a dangling FFI pointer that the
    // next `State::full` call would dereference. Reorder so
    // the raw fields are NEVER pointing at a freed payload.
    //
    // 1. Null the raw hooks first. If anything else fails
    //    below, the state is "no callback installed", which
    //    is always safe.
    self.raw.abort_callback = None;
    self.raw.abort_callback_user_data = core::ptr::null_mut();
    // 2. Drop the old owner explicitly. If the old closure's
    //    `Drop` panics, we unwind out of this function with
    //    `raw` already cleared (step 1) — the caller's
    //    `catch_unwind` cleanup leaves `Params` in a
    //    no-callback state, no dangling pointer.
    let _old = self._abort_callback.take();
    drop(_old);
    // 3. Build the new owner. `Box::new` could panic on OOM;
    //    if it does, `self._abort_callback` stays `None` and
    //    `raw` stays cleared.
    let outer: AbortCallback = Box::new(UnsafeCell::new(Box::new(f)));
    // 4. Derive the FFI pointer from `outer` BEFORE moving it
    //    into `self._abort_callback`. `Box<UnsafeCell<…>>`
    //    derefs to a heap address that is stable across the
    //    move into `Option<Box<…>>` (the `Box` value itself
    //    moves; the heap allocation it owns does not). This
    //    keeps the setter panic-free — no `expect` /
    //    `unwrap` after the assignment.
    let user_data = (&*outer) as *const UnsafeCell<Box<dyn FnMut() -> bool>> as *mut c_void;
    self._abort_callback = Some(outer);
    // 5. Publish the trampoline + user_data (from the FFI's
    //    perspective — whisper.cpp doesn't run concurrently
    //    here because we hold `&mut self`).
    self.raw.abort_callback_user_data = user_data;
    self.raw.abort_callback = Some(abort_trampoline);
    self
  }

  // ── Audio windowing ─────────────────────────────────────────

  /// Start offset into the audio, in milliseconds. Whisper.cpp
  /// internally seeks past the first `offset_ms` worth of mel
  /// frames before decoding. Defaults to `0`.
  ///
  /// Negative values are silently clamped to `0`. Upstream
  /// turns `offset_ms` into a negative `mel_offset` that
  /// reaches `mel_inp.data[j*n_len + i]` with `i < 0` — an
  /// out-of-bounds native read reachable from safe Rust per
  /// `whisper.cpp:2393–2398`. Clamping at the safe setter
  /// keeps the UB from crossing the FFI.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_offset_ms(&mut self, ms: i32) -> &mut Self {
    self.raw.offset_ms = if ms < 0 { 0 } else { ms };
    self
  }

  /// Hard cap on audio duration to decode, in milliseconds.
  /// `0` means "to end of input". Defaults to `0`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_duration_ms(&mut self, ms: i32) -> &mut Self {
    self.raw.duration_ms = ms;
    self
  }

  /// Override the encoder's audio context window. `0` keeps the
  /// model's native value (e.g. 1500 frames = 30s for the
  /// vanilla checkpoints); smaller values trade quality for
  /// speed on chunks << 30s.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_audio_ctx(&mut self, n: i32) -> &mut Self {
    self.raw.audio_ctx = n;
    self
  }

  // ── Decoding limits ────────────────────────────────────────

  /// Cap on tokens decoded per attempt. `0` lets whisper.cpp
  /// run to its natural EOT.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_max_tokens(&mut self, n: i32) -> &mut Self {
    self.raw.max_tokens = n;
    self
  }

  /// Maximum characters per segment. Combined with
  /// [`Self::set_split_on_word`], this is whisper.cpp's
  /// segment-shaping mechanism.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_max_len(&mut self, n: i32) -> &mut Self {
    self.raw.max_len = n;
    self
  }

  /// Maximum index a `<|t_x|>` token may take on the FIRST
  /// segment of a chunk. Whisper.cpp uses this to bias against
  /// implausibly large initial timestamps. Defaults to `1.0`
  /// (i.e. ≤ 1s lead-in is the typical valid range).
  ///
  /// Non-finite values (`NaN` / `±∞`) and negatives are
  /// silently clamped to `0.0`; values above
  /// [`MAX_INITIAL_TS_S`] are clamped to that ceiling.
  /// Upstream converts `std::round(max_initial_ts /
  /// precision)` to `int`, which is undefined behaviour in
  /// C++ for non-finite or out-of-int-range floats.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_max_initial_ts(&mut self, t: f32) -> &mut Self {
    self.raw.max_initial_ts = clamp_max_initial_ts(t);
    self
  }

  /// Cap on text context (`tokens_prev`) carried over between
  /// segments. Whisper.cpp internally truncates from the head
  /// when the prompt would exceed this.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_n_max_text_ctx(&mut self, n: i32) -> &mut Self {
    self.raw.n_max_text_ctx = n;
    self
  }

  /// Force whisper.cpp to emit at most ONE segment per `full`
  /// call. Useful when callers do their own segmentation
  /// upstream.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_single_segment(&mut self, on: bool) -> &mut Self {
    self.raw.single_segment = on;
    self
  }

  // ── Quality gates ──────────────────────────────────────────

  /// Whisper.cpp's internal logprob threshold for the temperature
  /// fallback ladder. Lower (more negative) = accept more
  /// uncertain decodes. Whispery's runner usually pins
  /// `temperature_inc=0` and gates externally on its own
  /// `log_prob_threshold` knob, so this is mostly a passthrough
  /// for callers that want whisper.cpp's built-in ladder.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_logprob_thold(&mut self, t: f32) -> &mut Self {
    self.raw.logprob_thold = t;
    self
  }

  /// Whisper.cpp's entropy gate (token-prob distribution must
  /// not collapse). Higher = stricter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_entropy_thold(&mut self, t: f32) -> &mut Self {
    self.raw.entropy_thold = t;
    self
  }

  /// Per-token timestamp probability threshold. Defaults to
  /// `0.01` upstream.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_thold_pt(&mut self, t: f32) -> &mut Self {
    self.raw.thold_pt = t;
    self
  }

  /// Sum-over-timestamps probability threshold. Defaults to
  /// `0.01` upstream.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_thold_ptsum(&mut self, t: f32) -> &mut Self {
    self.raw.thold_ptsum = t;
    self
  }

  /// Beam-search length penalty. Negative = penalise longer
  /// outputs; `-1.0` disables.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_length_penalty(&mut self, p: f32) -> &mut Self {
    self.raw.length_penalty = p;
    self
  }

  // ── Output shape ───────────────────────────────────────────

  /// Disable timestamp tokens in the output. Faster and
  /// suppresses the `<|t_x|>` markers; segment `t0`/`t1` still
  /// land via whisper.cpp's segment splitter, but per-token
  /// timestamps are not emitted.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_no_timestamps(&mut self, on: bool) -> &mut Self {
    self.raw.no_timestamps = on;
    self
  }

  /// Split segments only at word boundaries. Combines with
  /// [`Self::set_max_len`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_split_on_word(&mut self, on: bool) -> &mut Self {
    self.raw.split_on_word = on;
    self
  }

  /// Compute per-token timestamps via DTW. Whispery handles
  /// word-level alignment via wav2vec2 instead, so this is
  /// rarely useful — exposed for parity.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_token_timestamps(&mut self, on: bool) -> &mut Self {
    self.raw.token_timestamps = on;
    self
  }

  // ── Decoding seed ──────────────────────────────────────────

  /// Switch whisper.cpp to translation (transcribe → English).
  /// Whispery is transcription-only in production; exposed for
  /// completeness.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_translate(&mut self, on: bool) -> &mut Self {
    self.raw.translate = on;
    self
  }

  /// Seed the decoder with previously-emitted tokens (acts as
  /// `tokens_prev`). The slice is COPIED into a `Vec` owned by
  /// `self`; the caller's slice may be dropped after this call
  /// returns. Pass `&[]` to clear a previously-set prompt.
  pub fn set_tokens(&mut self, tokens: &[i32]) -> &mut Self {
    if tokens.is_empty() {
      self.raw.prompt_tokens = core::ptr::null();
      self.raw.prompt_n_tokens = 0;
      self._prompt_tokens = None;
    } else {
      // Bound the slice copy at `i32::MAX` elements so the
      // assignment to `prompt_n_tokens` (a C `int`) cannot
      // wrap to a negative or truncated count. `prompt_n_tokens`
      // crosses the FFI as `int`, and `tokens.len() as i32`
      // would silently truncate / wrap on slices wider than
      // 2 GiB of `whisper_token` (impossible in practice on
      // any current platform, but the cast is unsound under
      // the crate's panic-free contract).
      let max_len = i32::MAX as usize;
      let take = tokens.len().min(max_len);
      let owned: Vec<sys::whisper_token> = tokens[..take].to_vec();
      self.raw.prompt_tokens = owned.as_ptr();
      self.raw.prompt_n_tokens = owned.len() as i32;
      self._prompt_tokens = Some(owned);
    }
    self
  }

  /// Internal: hand the raw C struct to `state::full`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn as_raw(&self) -> sys::whisper_full_params {
    self.raw
  }

  /// Internal: borrow the prompt-token slice (if any). Used by
  /// `State::full` to range-check against the model's vocab
  /// before forwarding the FFI call. Returns `None` when no
  /// prompt has been set (or [`Self::set_tokens`] was last
  /// called with an empty slice).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn prompt_tokens(&self) -> Option<&[i32]> {
    self._prompt_tokens.as_deref()
  }
}

// Manual `Debug` because the boxed abort callback is `dyn FnMut`
// (no `Debug` impl). We elide it; the rest of the params surface
// renders fine via the bindgen-derived `Debug` on
// `whisper_full_params`.
impl core::fmt::Debug for Params {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Params")
      .field("raw", &self.raw)
      .field("language", &self._language)
      .field("initial_prompt", &self._initial_prompt)
      .field(
        "abort_callback",
        &self
          ._abort_callback
          .as_ref()
          .map(|_| "<installed>")
          .unwrap_or("<none>"),
      )
      .field(
        "prompt_tokens",
        &self._prompt_tokens.as_ref().map(|v| v.len()).unwrap_or(0),
      )
      .finish()
  }
}

unsafe extern "C" fn abort_trampoline(user_data: *mut c_void) -> bool {
  // SAFETY: `user_data` is the pointer we stored in
  // `set_abort_callback`. It points to a live
  // `UnsafeCell<Box<dyn FnMut -> bool>>` whose lifetime is
  // tied to the owning `Params` (which the caller of
  // `State::full` borrows for the duration of the call). The
  // `UnsafeCell` is what makes the `&mut`-borrow we form below
  // legal — `Params` is reachable to safe code only through
  // a shared `&Params` reference, and ordinary fields of a
  // shared reference cannot be mutated; routing through
  // `UnsafeCell` is the canonical opt-in for this pattern.
  // (Whisper.cpp guarantees no concurrent invocation of the
  // callback for a single state, so the FnMut borrow is
  // exclusive at all times.)
  let cell: &UnsafeCell<Box<dyn FnMut() -> bool>> =
    unsafe { &*(user_data as *const UnsafeCell<Box<dyn FnMut() -> bool>>) };
  // SAFETY: see above; whisper.cpp invokes the callback
  // serially, so this is the only outstanding access.
  let boxed: &mut Box<dyn FnMut() -> bool> = unsafe { &mut *cell.get() };

  // Catch unwinds — panicking across `extern "C"` is
  // undefined behaviour. On panic, return `true` so
  // whisper.cpp aborts the in-flight encode rather than
  // continuing into an inconsistent state, and let the panic
  // surface on the Rust side via the per-thread panic info
  // (`std::thread::Result`-like — callers wrapping
  // `State::full` in `catch_unwind` will see it).
  catch_unwind(AssertUnwindSafe(boxed)).unwrap_or(true)
}

// FFI-touching tests are skipped under Miri (every test
// here that constructs `Params` reaches `whisper_full_default_params`,
// which Miri can't execute). The pure-helper tests
// (`clamp_*`, constants) stay enabled.
// reinstated Miri coverage for the safe wrapper.
#[cfg(test)]
mod tests {
  use super::*;

  /// Default-constructed `Params` must never carry an
  /// `n_threads < 1`. `whisper_full_default_params` initialises
  /// it from `std::min(4, hardware_concurrency)`, and
  /// `hardware_concurrency` may legally return `0`; without
  /// our `clamp_n_threads` in `Params::new`, that `0` would
  /// cross the FFI boundary into a `vector<thread>(0 - 1)`
  /// underflow.
  #[test]
  #[cfg_attr(miri, ignore = "FFI: whisper_full_default_params")]
  fn default_params_n_threads_normalises_to_at_least_one() {
    let p = Params::new(SamplingStrategy::Greedy { best_of: 1 });
    assert!(
      p.raw.n_threads >= 1,
      "default n_threads = {}; must be ≥ 1 to dodge the upstream vector<thread>(n - 1) underflow",
      p.raw.n_threads,
    );
    assert!(
      p.raw.n_threads <= MAX_N_THREADS,
      "default n_threads = {} above MAX_N_THREADS = {}",
      p.raw.n_threads,
      MAX_N_THREADS,
    );
  }

  /// Caller-supplied invalid `n_threads` clamps in both
  /// directions, with no panic and no FFI involvement.
  #[test]
  #[cfg_attr(miri, ignore = "FFI: whisper_full_default_params")]
  fn set_n_threads_clamps_zero_negative_and_oversized() {
    let mut p = Params::new(SamplingStrategy::Greedy { best_of: 1 });
    p.set_n_threads(0);
    assert_eq!(p.raw.n_threads, 1, "0 → 1");
    p.set_n_threads(-42);
    assert_eq!(p.raw.n_threads, 1, "negative → 1");
    p.set_n_threads(i32::MIN);
    assert_eq!(p.raw.n_threads, 1, "i32::MIN → 1");
    p.set_n_threads(MAX_N_THREADS + 1);
    assert_eq!(p.raw.n_threads, MAX_N_THREADS, "above-cap → MAX_N_THREADS");
    p.set_n_threads(i32::MAX);
    assert_eq!(p.raw.n_threads, MAX_N_THREADS, "i32::MAX → MAX_N_THREADS");
  }

  /// `clamp_n_threads` is the canonical helper used by both
  /// `Params::new` and `set_n_threads`. Pin its surface so a
  /// future refactor can't quietly change behaviour.
  #[test]
  fn clamp_n_threads_pins_invariants() {
    assert_eq!(clamp_n_threads(0), 1);
    assert_eq!(clamp_n_threads(-1), 1);
    assert_eq!(clamp_n_threads(1), 1);
    assert_eq!(clamp_n_threads(MAX_N_THREADS), MAX_N_THREADS);
    assert_eq!(clamp_n_threads(MAX_N_THREADS + 1), MAX_N_THREADS);
    // Anything ≥ MAX_N_THREADS clamps down.
    // narrowed this to 1 (the multi-decoder process loop
    // races caller-thread throws against joinable workers
    // even at n=2). Pin a value that used to pass through
    // under the cap of 2 to lock in the new
    // ceiling.
    assert_eq!(clamp_n_threads(2), MAX_N_THREADS);
    assert_eq!(clamp_n_threads(8), MAX_N_THREADS);
  }

  /// `set_n_threads_unchecked` bypasses [`MAX_N_THREADS`] but
  /// still applies the lower-bound clamp (the underflow
  /// guard for `vector<thread>(n - 1)`).
  #[test]
  #[cfg_attr(miri, ignore = "FFI: whisper_full_default_params")]
  fn set_n_threads_unchecked_bypasses_upper_cap_only() {
    let mut p = Params::new(SamplingStrategy::Greedy { best_of: 1 });
    // Above the safe cap → passes through verbatim.
    // SAFETY (test): we never call State::full here, so the
    // unsafe contract about `std::thread` headroom is
    // vacuously satisfied.
    unsafe { p.set_n_threads_unchecked(8) };
    assert_eq!(p.raw.n_threads, 8);
    unsafe { p.set_n_threads_unchecked(MAX_N_THREADS + 1) };
    assert_eq!(p.raw.n_threads, MAX_N_THREADS + 1);
    unsafe { p.set_n_threads_unchecked(64) };
    assert_eq!(p.raw.n_threads, 64);
    // Lower bound still clamps — `n ≤ 0` is a separate bug
    // shape (vector underflow) that the unsafe contract
    // does NOT cover.
    unsafe { p.set_n_threads_unchecked(0) };
    assert_eq!(p.raw.n_threads, 1, "0 → 1 even on unchecked path");
    unsafe { p.set_n_threads_unchecked(-7) };
    assert_eq!(p.raw.n_threads, 1, "negative → 1 even on unchecked path");
  }

  /// Pin `MAX_BEAM_SIZE`. The bundled build patches
  /// `whisper_kv_cache_free` to be idempotent so
  /// multi-decoder is safe; `64` is the sanity ceiling.
  #[test]
  fn max_beam_size_pins_to_64() {
    assert_eq!(MAX_BEAM_SIZE, 64);
  }

  /// `set_beam_size_unchecked` / `set_best_of_unchecked`
  /// bypass `MAX_BEAM_SIZE` but still clamp `n ≤ 0` to 1
  /// (separate underflow bug shape outside the unsafe
  /// contract). Same shape as `set_n_threads_unchecked`.
  #[test]
  #[cfg_attr(miri, ignore = "FFI: whisper_full_default_params")]
  fn unchecked_topk_setters_bypass_upper_cap_only() {
    let mut p = Params::new(SamplingStrategy::BeamSearch {
      beam_size: 1,
      patience: -1.0,
    });
    // Above MAX_BEAM_SIZE — passes through verbatim.
    // SAFETY (test): no State::full call, contract vacuously
    // satisfied.
    unsafe { p.set_beam_size_unchecked(MAX_BEAM_SIZE + 1) };
    assert_eq!(p.raw.beam_search.beam_size, MAX_BEAM_SIZE + 1);
    unsafe { p.set_beam_size_unchecked(128) };
    assert_eq!(p.raw.beam_search.beam_size, 128);
    // Lower bound still clamps.
    unsafe { p.set_beam_size_unchecked(0) };
    assert_eq!(p.raw.beam_search.beam_size, 1);
    unsafe { p.set_beam_size_unchecked(-9) };
    assert_eq!(p.raw.beam_search.beam_size, 1);

    let mut g = Params::new(SamplingStrategy::Greedy { best_of: 1 });
    unsafe { g.set_best_of_unchecked(MAX_BEAM_SIZE + 1) };
    assert_eq!(g.raw.greedy.best_of, MAX_BEAM_SIZE + 1);
    unsafe { g.set_best_of_unchecked(0) };
    assert_eq!(g.raw.greedy.best_of, 1);
  }
}
