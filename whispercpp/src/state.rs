//! Inference state, segments, and tokens.
//!
//! `State` owns an [`Arc`] of its parent [`Context`], which keeps
//! the model alive for the state's lifetime. We picked Arc-
//! ownership over a `'ctx` borrow because the realistic usage
//! pattern (worker pools storing per-thread state across jobs)
//! is hard to express with a lifetime — the borrow checker can't
//! see that the parent Arc lives in the same stack frame as the
//! State without explicit annotation. Arc-owned lets `State` be
//! `'static` and storable in `Option<State>` / channels.
//!
//! The state is single-threaded by design (whisper.cpp scratch
//! buffers + KV cache are not thread-safe); we mark it `!Sync`
//! implicitly by holding a raw pointer.

#![allow(unsafe_code)]

use core::{ptr::NonNull, str};
use std::sync::Arc;

use crate::{
  context::Context,
  error::{WhisperError, WhisperResult},
  lang::Lang,
  params::Params,
  sys,
};

/// Per-call inference state. Owns an [`Arc<Context>`] so the
/// model outlives every per-call buffer.
///
/// # Poisoning
///
/// `whisper_full_with_state` calls `whisper_free_state` on
/// itself before returning `-7` from a multi-decoder KV-cache
/// allocation failure (see `whisper.cpp:7126` in the patched
/// fork). After that the underlying C++ pointer is gone — we
/// MUST NOT free it again on `Drop` and accessor methods MUST
/// NOT touch it. We model this by storing the pointer as
/// `Option<NonNull<…>>`: on `-7` we `take()` it, flipping the
/// `State` into a "poisoned" mode where every method
/// short-circuits to a safe zero/None and `Drop` becomes a
/// no-op.
///
/// On caught-exception sentinels (`rc <= -100`), `State::full`
/// instead calls `whisper_free_state` EXPLICITLY to release the
/// native allocation, then nulls `self.ptr`. This is viable
/// because the fork's idempotent `whisper_kv_cache_free` patch
/// closed the double-free hazard that previously forced us to
/// leak the state.
///
/// # Recovery contract on `StateLost`
///
/// 1. Drop this `State`. The native allocation is already
///    released (no leak).
/// 2. The parent `Context` is poisoned —
///    [`Context::create_state`] will return
///    `WhisperError::ContextPoisoned` until you drop and
///    reconstruct the Context. This is defensive: the
///    pressure that caused the throw is likely still
///    present.
/// 3. Surface the error to your supervisor; once pressure
///    has resolved, reload the model in a fresh `Context`.
///
/// `whispery`'s worker pool (`whisper_pool.rs`) treats
/// `WhisperError::StateLost` as a `WorkFailure::AsrFailed`
/// without auto-recreating the `State`, matching this
/// contract. Other consumers must implement equivalent
/// supervision.
pub struct State {
  ptr: Option<NonNull<sys::whisper_state>>,
  // Keeps the parent Context alive. No `'ctx` lifetime: makes
  // State `'static` for storage in `Option<State>` / channels /
  // the long-lived worker structs whispery uses.
  ctx: Arc<Context>,
}

// SAFETY: the `whisper_state` pointer is owned exclusively by us
// (no aliases). whisper.cpp permits passing a state across
// threads as long as no two threads call `whisper_full` on it
// concurrently — that's the same guarantee `Send` requires. The
// Arc<Context> is itself Send.
unsafe impl Send for State {}

impl State {
  /// Internal constructor used by [`Context::create_state`].
  pub(crate) fn from_raw(ptr: NonNull<sys::whisper_state>, ctx: Arc<Context>) -> Self {
    Self {
      ptr: Some(ptr),
      ctx,
    }
  }

  /// Borrow the parent context. Useful when calling sites need
  /// the same Arc to construct sibling state objects.
  pub fn context(&self) -> &Arc<Context> {
    &self.ctx
  }

  /// `true` if the underlying whisper.cpp state was freed
  /// behind the Rust owner (multi-decoder KV-cache allocation
  /// failure → `whisper_full_with_state` returned `-7` after
  /// internally calling `whisper_free_state`). After
  /// poisoning, every accessor returns a zero/None default
  /// and `Drop` skips the C-side free; the only sound move
  /// for callers is to drop the [`State`] and (if desired)
  /// allocate a fresh one via
  /// [`Context::create_state`](crate::Context::create_state).
  pub fn is_poisoned(&self) -> bool {
    self.ptr.is_none()
  }

  /// Internal: shorthand for accessor methods that need the
  /// raw pointer. Returns `None` for a poisoned state.
  #[inline]
  fn raw(&self) -> Option<*mut sys::whisper_state> {
    self.ptr.map(NonNull::as_ptr)
  }

  /// Run the encoder + decoder over `samples` (16 kHz mono f32).
  ///
  /// Returns `Ok()` on the success contract; the segment list
  /// is then accessible via [`State::n_segments`] and
  /// [`State::segment`]. **Panic-free.** Returns
  /// * [`WhisperError::SamplesOverflow`] when `samples.len`
  ///   does not fit in the C `int` whisper.cpp expects.
  /// * [`WhisperError::SamplesTooShort`] when the buffer is
  ///   smaller than the 201-sample lower bound whisper.cpp's
  ///   `log_mel_spectrogram` reflective pad requires.
  /// * [`WhisperError::TokenOutOfRange`] when a `prompt_tokens`
  ///   id passed to [`Params::set_tokens`] does not lie in
  ///   `[0, n_vocab)`. Upstream feeds those ids straight
  ///   into `ggml_get_rows(model.d_te, embd)`; out-of-range
  ///   indices either trip a CPU-side assert or cause invalid
  ///   memory access in GPU kernels, both of which abort
  ///   across the FFI.
  pub fn full(&mut self, params: &Params, samples: &[f32]) -> WhisperResult<()> {
    // `log_mel_spectrogram` runs
    // `reverse_copy(samples + 1, samples + 1 + 200, …)` for its
    // start-of-buffer reflective pad, reading
    // `samples[1..201]`. Sub-201-sample inputs trigger an
    // out-of-bounds read in the C++ kernel before whisper.cpp's
    // later short-input check fires. Reject them here so the
    // UB never crosses the FFI boundary.
    // surfaced this as a critical finding against whisper.cpp
    // v1.8.4 (`src/whisper.cpp:3201`).
    const MIN_SAMPLES_FOR_REFLECTIVE_PAD: usize = 201;
    if samples.len() < MIN_SAMPLES_FOR_REFLECTIVE_PAD {
      return Err(WhisperError::SamplesTooShort {
        samples: samples.len(),
        min_required: MIN_SAMPLES_FOR_REFLECTIVE_PAD,
      });
    }
    let len = i32::try_from(samples.len()).map_err(|_| WhisperError::SamplesOverflow {
      samples: samples.len(),
    })?;
    // Prompt-token range check. `Params::set_tokens` accepts an
    // arbitrary `&[i32]`; we couldn't validate at setter time
    // without forcing callers to thread a `Context` through
    // `Params::new`. Doing the check here (we have the Context
    // via `self.ctx`) catches the unsoundness while keeping
    // `Params` free of model state. surfaced
    // this against whisper.cpp v1.8.4 (`src/whisper.cpp:6915`):
    // upstream feeds these ids into `ggml_get_rows(d_te)`, where
    // CPU asserts and GPU kernels both abort on OOB.
    if let Some(prompt) = params.prompt_tokens() {
      let vocab = self.ctx.n_vocab();
      for &tok in prompt {
        if tok < 0 || tok >= vocab {
          return Err(WhisperError::TokenOutOfRange {
            token: tok,
            vocab_size: vocab,
          });
        }
      }
    }
    // Poisoned state can't run inference — the C-side pointer
    // is gone. Surface this as `StateLost` so callers can
    // distinguish "drop this State, do not auto-retry" from
    // "transient error, state still usable".
    let state_ptr = match self.ptr {
      Some(p) => p.as_ptr(),
      None => return Err(WhisperError::StateLost { code: -7 }),
    };
    // finding 2: serialise FFI entry through
    // the Context's per-Context mutex. Without this, multiple
    // workers each holding their own State (the documented
    // shared-`Arc<Context>` pattern) could ALL be inside
    // `whispercpp_full_with_state` simultaneously when an OOM
    // / system_error hits, each leaking ~360 MB on its own
    // sentinel return before any of them got to mark the
    // Context lost — turning the per-Context leak cap into a
    // per-concurrent-worker cap. Holding the mutex across
    // the FFI call makes the cap structural: at most one
    // in-flight call per Context, so at most one leaked
    // state per Context.
    //
    // The lock acquisition is the FIRST thing that happens
    // on the inference path — the prompt-token / sample-len
    // checks above don't touch native state and are
    // contention-free, so we keep them outside the critical
    // section.
    let _full_guard = self.ctx.full_lock();
    // (re-applied under the lock): refuse to
    // enter FFI if a sibling State on the same
    // `Arc<Context>` has already poisoned the Context. The
    // mutex above ensures we observe the latest poison
    // state (the previous holder set `lost` via Release
    // store BEFORE releasing the mutex; our Acquire load
    // here sees that store). The sibling's `Drop` still
    // frees its native state cleanly (its `self.ptr` is
    // still `Some`); we only block the FFI entry that could
    // turn that intact native state into another leaked one.
    if self.ctx.is_poisoned() {
      return Err(WhisperError::ContextPoisoned);
    }
    // SAFETY:
    // - `self.ctx.as_raw` is a non-null whisper_context
    //   (NonNull invariant on Context); kept alive by the Arc
    //   we own.
    // - `state_ptr` is non-null (just unwrapped from NonNull).
    // - `params.as_raw` is a fully-initialised
    //   `whisper_full_params` whose owned CStrings live as long
    //   as `params`.
    // - `samples.as_ptr` is valid for `len` f32 reads
    //   (slice invariant).
    //
    // Routed through the exception-catching shim
    // `whispercpp_full_with_state`: upstream constructs
    // `std::thread` workers and allocates `std::vector`
    // buffers, both of which can throw (`std::system_error`,
    // `std::bad_alloc`) under realistic resource pressure.
    // C++ exceptions across `extern "C"` are UB; sentinel
    // codes documented on
    // `whispercpp_shim.h::WHISPERCPP_ERR_*`.
    let rc = unsafe {
      sys::whispercpp_full_with_state(
        self.ctx.as_raw(),
        state_ptr,
        params.as_raw(),
        samples.as_ptr(),
        len,
      )
    };
    if rc == 0 {
      return Ok(());
    }
    // Failure regimes (— the leak that
    // earlier rounds documented as "unavoidable" is now
    // freeable thanks to the idempotent
    // `whisper_kv_cache_free` patch):
    //
    // 1. `rc == -7` — multi-decoder KV-cache rebuild
    //    failure. Upstream calls `whisper_free_state(state)`
    //    before returning the code, so our pointer is
    //    dangling. We MUST NOT free again. Suppress Drop
    //    and surface as `StateLost`.
    //
    // 2. `rc <= WHISPERCPP_ERR_BAD_ALLOC` — the shim caught
    //    a C++ exception (`std::bad_alloc`,
    //    `std::system_error`, other `std::exception`, or
    //    unknown). Upstream did NOT call
    //    `whisper_free_state`. Earlier rounds intentionally
    //    leaked the state (~360 MB on `large-v3-turbo`)
    //    because the multi-decoder rebuild path could leave
    //    `kv_self.buffer` freed-but-not-nulled — a second
    //    free would crash. The patch closed that
    //    door by making `whisper_kv_cache_free` idempotent
    //    (it now nulls `cache.buffer` after free; re-call
    //    short-circuits via `ggml_backend_buffer_free`'s own
    //    null guard at `ggml-backend.cpp:107-109`). Every
    //    other state member (`std::vector<...>` fields,
    //    backend lists, sched slots, `whisper_batch`) either
    //    self-destructs or is freed exactly once by
    //    `whisper_free_state`. So calling
    //    `whisper_free_state` on a state that the shim
    //    rescued from a throw is safe — and avoids the
    //    leak.
    //
    // 3. `rc < 0` otherwise — recoverable upstream failure
    //    (encode/decode failure, etc.). State is intact;
    //    return `Full { code }` and keep the pointer alive
    //    so future calls can reuse the state.
    if rc == -7 {
      // Upstream already freed; suppress our Drop.
      self.ptr = None;
      // poison the parent Context so
      // subsequent `create_state` calls fail with
      // `ContextPoisoned`. The native state is gone, but
      // the failure mode (multi-decoder OOM) is a strong
      // signal of resource pressure on this Context.
      self.ctx.mark_lost();
      return Err(WhisperError::StateLost { code: rc });
    }
    if rc <= sys::WHISPERCPP_ERR_BAD_ALLOC {
      // finding 2: poison the Context BEFORE
      // touching the native state. `Context::create_state`
      // does not take `full_lock`, so without this ordering
      // another thread could observe `lost == false` during
      // the `whisper_free_state` window, allocate a fresh
      // State, and publish it before our `mark_lost` ran.
      // Releasing the poison flag first closes the race:
      // any concurrent `create_state` either sees `lost ==
      // true` (returns `ContextPoisoned`) or completes
      // before we reach this branch.
      self.ctx.mark_lost();

      // Shim caught a C++ throw. Free the native state
      // explicitly. `take` clears
      // `self.ptr` so Drop becomes a no-op.
      if let Some(p) = self.ptr.take() {
        // SAFETY: `p` was returned by `whisper_init_state`
        // and held exclusively by `self`; the shim's catch
        // means `whisper_full_with_state` did NOT call
        // `whisper_free_state`, so `p` is still owned by us.
        // The idempotent kv_cache_free patch makes
        // this call safe even when the throw left
        // `kv_self.buffer` released.
        unsafe { sys::whisper_free_state(p.as_ptr()) };
      }
      return Err(WhisperError::StateLost { code: rc });
    }
    Err(WhisperError::Full { code: rc })
  }

  /// Number of segments produced by the most recent
  /// [`State::full`] call. Returns `0` for a poisoned state
  /// (see [`Self::is_poisoned`]).
  pub fn n_segments(&self) -> i32 {
    let Some(state) = self.raw() else { return 0 };
    // SAFETY: state non-null; pure read.
    unsafe { sys::whisper_full_n_segments_from_state(state) }
  }

  /// Borrow segment `idx` (0-indexed). Returns `None` for a
  /// poisoned state, or when `idx` is out of range.
  pub fn segment(&self, idx: i32) -> Option<Segment<'_>> {
    let state_ptr = self.ptr?;
    if idx < 0 || idx >= self.n_segments() {
      return None;
    }
    Some(Segment {
      state: state_ptr,
      idx,
      _marker: core::marker::PhantomData,
    })
  }

  /// Detected (or forced) language for the most recent
  /// [`State::full`] call.
  ///
  /// Returns `None` when:
  /// * the state is poisoned (see [`Self::is_poisoned`]);
  /// * whisper.cpp set the internal lang id to `-1` (no
  ///   detection ran, no hint set);
  /// * the id is out of whisper.cpp's published table; or
  /// * the table entry is not valid UTF-8 (corrupt build).
  ///
  /// Returns the strongly-typed [`Lang`] (canonicalised
  /// through `Lang::from_iso639_1`) so callers don't pattern-
  /// match on raw ISO strings. Known whisper.cpp codes round-
  /// trip to their named variant; unknown codes land in
  /// `Lang::Other` with the lowercase ISO string preserved.
  pub fn detected_lang(&self) -> Option<Lang> {
    let state = self.raw()?;
    // SAFETY: state non-null; pure read.
    let id = unsafe { sys::whisper_full_lang_id_from_state(state) };
    if id < 0 {
      return None;
    }
    // SAFETY: whisper_lang_str is a pure C accessor returning
    // a pointer into a static const-table baked into
    // libwhisper. The returned slice lives forever.
    let raw = unsafe { sys::whisper_lang_str(id) };
    if raw.is_null() {
      return None;
    }
    // SAFETY: NUL-terminated; static lifetime per whisper.cpp.
    let bytes = unsafe { core::ffi::CStr::from_ptr(raw).to_bytes() };
    let code = str::from_utf8(bytes).ok()?;
    Some(Lang::from_iso639_1(code))
  }
}

impl Drop for State {
  fn drop(&mut self) {
    // Poisoned state — whisper.cpp already freed itself. We
    // MUST NOT call `whisper_free_state` again.
    if let Some(p) = self.ptr.take() {
      // SAFETY: ptr was non-null and produced by
      // whisper_init_state; never freed by anyone else
      // because `is_poisoned` was false.
      unsafe { sys::whisper_free_state(p.as_ptr()) }
    }
  }
}

/// Borrowed view of one segment.
///
/// Reaches into the `State` lazily — calling [`Segment::text`]
/// performs an FFI call each time. That matches whisper.cpp's
/// own model: segments are addressed by index, not pre-extracted.
#[derive(Clone, Copy)]
pub struct Segment<'a> {
  state: NonNull<sys::whisper_state>,
  idx: i32,
  _marker: core::marker::PhantomData<&'a ()>,
}

impl<'a> Segment<'a> {
  /// Start time, in centiseconds (whisper.cpp's native unit).
  /// Multiply by 0.01 for seconds.
  pub fn t0(&self) -> i64 {
    // SAFETY: state pointer invariant; idx is in-range (we
    // checked at construction in `State::segment`).
    unsafe { sys::whisper_full_get_segment_t0_from_state(self.state.as_ptr(), self.idx) }
  }

  /// End time, in centiseconds.
  pub fn t1(&self) -> i64 {
    // SAFETY: see `t0`.
    unsafe { sys::whisper_full_get_segment_t1_from_state(self.state.as_ptr(), self.idx) }
  }

  /// Decoded text for this segment. Returned slice is valid
  /// while `self` is held — whisper.cpp owns the buffer.
  pub fn text(&self) -> WhisperResult<&'a str> {
    // SAFETY: idx in-range; whisper_full_get_segment_text returns
    // a pointer into the state's owned buffer; we do not store
    // it past the returned &str's lifetime.
    let raw =
      unsafe { sys::whisper_full_get_segment_text_from_state(self.state.as_ptr(), self.idx) };
    if raw.is_null() {
      return Ok("");
    }
    // SAFETY: whisper.cpp guarantees NUL-terminated UTF-8 text
    // for any valid model vocabulary.
    let bytes = unsafe { core::ffi::CStr::from_ptr(raw).to_bytes() };
    str::from_utf8(bytes).map_err(WhisperError::from)
  }

  /// `no_speech_prob` for this segment — whisper.cpp's gate for
  /// the silent-segment shortcut. Higher = more confident the
  /// segment is silence.
  pub fn no_speech_prob(&self) -> f32 {
    // SAFETY: idx in-range; pure read.
    unsafe {
      sys::whisper_full_get_segment_no_speech_prob_from_state(self.state.as_ptr(), self.idx)
    }
  }

  /// Number of tokens decoded inside this segment.
  pub fn n_tokens(&self) -> i32 {
    // SAFETY: idx in-range; pure read.
    unsafe { sys::whisper_full_n_tokens_from_state(self.state.as_ptr(), self.idx) }
  }

  /// Borrow token `tok_idx` of this segment. Returns `None` if
  /// `tok_idx` is out of range.
  pub fn token(&self, tok_idx: i32) -> Option<Token> {
    if tok_idx < 0 || tok_idx >= self.n_tokens() {
      return None;
    }
    // SAFETY: indices in-range; whisper.cpp returns a value-
    // typed `whisper_token_data`. We project into our private
    // `Token` view via `Token::from_raw`.
    let raw = unsafe {
      sys::whisper_full_get_token_data_from_state(self.state.as_ptr(), self.idx, tok_idx)
    };
    Some(Token::from_raw(raw))
  }

  /// `true` if the next segment marks a speaker change
  /// (whisper.cpp's tinydiarize / `--tdrz` mode). Always
  /// `false` outside TDRZ-enabled checkpoints; exposed for
  /// completeness so callers running TDRZ models don't have to
  /// reach into raw FFI.
  pub fn speaker_turn_next(&self) -> bool {
    // SAFETY: idx in-range; pure read.
    unsafe {
      sys::whisper_full_get_segment_speaker_turn_next_from_state(self.state.as_ptr(), self.idx)
    }
  }
}

/// Per-token data exposed by whisper.cpp.
///
/// Read-only snapshot. All fields are private; access goes
/// through `const fn` accessors to keep the public surface
/// stable as `whisper_token_data` evolves upstream.
#[derive(Debug, Clone, Copy)]
pub struct Token {
  id: i32,
  p: f32,
  plog: f32,
  pt: f32,
  ptsum: f32,
  t0: i64,
  t1: i64,
  vlen: f32,
}

impl Token {
  /// Token id in the model vocabulary.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn id(&self) -> i32 {
    self.id
  }

  /// Probability of this token at decode time.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn p(&self) -> f32 {
    self.p
  }

  /// Log-probability (matches whisper.cpp's internal score).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn plog(&self) -> f32 {
    self.plog
  }

  /// Timestamp probability if this token is a `<|t|>` marker.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn pt(&self) -> f32 {
    self.pt
  }

  /// Sum of all timestamp-token probabilities.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn ptsum(&self) -> f32 {
    self.ptsum
  }

  /// DTW-derived start time (centiseconds), if available.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn t0(&self) -> i64 {
    self.t0
  }

  /// DTW-derived end time (centiseconds), if available.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn t1(&self) -> i64 {
    self.t1
  }

  /// Voice activity score, if available.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn vlen(&self) -> f32 {
    self.vlen
  }

  /// Internal constructor used by [`State`] when projecting
  /// `whisper_token_data` into the safe view.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn from_raw(raw: crate::sys::whisper_token_data) -> Self {
    Self {
      id: raw.id,
      p: raw.p,
      plog: raw.plog,
      pt: raw.pt,
      ptsum: raw.ptsum,
      t0: raw.t0,
      t1: raw.t1,
      vlen: raw.vlen,
    }
  }
}
