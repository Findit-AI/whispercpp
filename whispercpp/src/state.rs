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

  /// Internal test-only constructor. Builds a `State` already
  /// in the "poisoned" mode (`ptr: None`, see
  /// [`State::is_poisoned`]) so accessors short-circuit to
  /// safe defaults (`n_segments() == 0`, `segment(_)` →
  /// `None`). Used by unit tests to exercise Rust-side
  /// iterator logic without spinning up a real
  /// `whisper_state`. The associated `Arc<Context>` should be
  /// a `Context::dangling_for_test`; callers must
  /// `core::mem::forget` the unwrapped Arc so the dangling
  /// `whisper_context*` never reaches `whisper_free`.
  #[cfg(test)]
  pub(crate) fn poisoned_for_test(ctx: Arc<Context>) -> Self {
    Self { ptr: None, ctx }
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
  /// * [`WhisperError::InvalidDuration`] when the configured
  ///   `offset_ms` / `duration_ms` describe an audio range
  ///   extending past the actual sample buffer. Upstream
  ///   does not bound `seek_end` to the mel length, so an
  ///   over-large duration drives a long zero-padded decode
  ///   loop instead of erroring; the wrapper rejects up-front.
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
    // Model-bound language preflight. `Params::set_language`
    // validated against whisper.cpp's process-global
    // `g_lang` table — that catches typos but accepts
    // language ids whisper.cpp's table knows about even
    // when the LOADED MODEL has fewer language-token
    // slots. `whisper_token_lang(ctx, lang_id)` is a bare
    // `token_sot + 1 + lang_id` calculation with no bounds
    // check; if `lang_id` exceeds the model's actual
    // language-token count the resulting token lands on
    // `token_translate` / `token_transcribe` / further
    // special slots and the decoder runs with a corrupted
    // SOT-style prefix (silently producing wrong-language
    // or task-biased transcripts).
    //
    // Skip the check when:
    //   * Hint is empty / "auto" (auto-detect sentinels).
    //   * `params.detect_language` is true — upstream
    //     ignores `params.language` in this mode and runs
    //     language detection on the audio. A stale or
    //     model-incompatible hint is harmless.
    //
    // Otherwise, the validation depends on whether the
    // loaded model is multilingual:
    //
    // * Multilingual: the resolved
    //   `whisper_token_lang(ctx, lang_id)` must lie in
    //   `(token_sot, token_translate)` — the model's
    //   actual language-token range.
    //
    // * English-only checkpoint (`.en` suffix; no language
    //   tokens at all): upstream's
    //   `whisper_full_with_state` only pushes a `<|lang|>`
    //   token inside `if (is_multilingual)`. So
    //   `params.language = "en"` (id 0) is a no-op accept
    //   on `.en` models. Non-English hints, however,
    //   indicate caller confusion (the model can't
    //   actually transcribe Chinese / French / etc.) — we
    //   reject with `LanguageNotSupportedByModel` so the
    //   bad config surfaces early instead of silently
    //   producing English output.
    let lang_opt = params.language();
    let auto_detect = params.detect_language()
      || match lang_opt {
        None => true,
        Some(s) => s.is_empty() || s == "auto",
      };
    if auto_detect {
      // Auto-detect path. Upstream's
      // `whisper_lang_auto_detect_with_state` loop iterates
      // every `g_lang` entry and reads
      // `state->logits[whisper_token_lang(ctx, id)]` with no
      // bound against `vocab.num_languages()`. The
      // submodule's `auto-detect bounded to model lang
      // range` patch filters the candidate set so an
      // out-of-range id can no longer be selected; this
      // wrapper-level guard rejects the case the patch
      // can't recover (no language tokens at all) with a
      // typed error instead of upstream's bare `-3`.
      //
      // English-only checkpoints (`is_multilingual()`
      // false) have no language tokens; auto-detect would
      // score regular vocabulary tokens. Reject up-front.
      if !self.ctx.is_multilingual() {
        return Err(WhisperError::LanguageNotSupportedByModel(
          smol_str::SmolStr::new_static("auto"),
        ));
      }
    } else if let Some(lang) = lang_opt {
      let model_supports = if let Some(lang_id) = crate::lang::lang_id_for(lang) {
        if self.ctx.is_multilingual() {
          // SAFETY: pure C addition; no allocation.
          let token = unsafe { sys::whisper_token_lang(self.ctx.as_raw(), lang_id) };
          let sot = self.ctx.token_sot();
          // SAFETY: pure C read of vocab field.
          let translate = unsafe { sys::whisper_token_translate(self.ctx.as_raw()) };
          token > sot && token < translate
        } else {
          // English-only model — only English (id 0,
          // matching `"en"` and `"english"`) is a valid
          // no-op accept. Other languages would be
          // silently ignored upstream, but their presence
          // signals likely caller confusion.
          lang_id == 0
        }
      } else {
        // Unknown language. `set_language` should have
        // rejected this already; defend against
        // `Default + struct update` bypassing the setter.
        false
      };
      if !model_supports {
        return Err(WhisperError::LanguageNotSupportedByModel(
          smol_str::SmolStr::new(lang),
        ));
      }
    }
    // duration_ms / offset_ms range preflight. Upstream's
    // `whisper_full_with_state` computes
    //   seek_end = params.duration_ms == 0
    //              ? whisper_n_len_from_state(state)
    //              : seek_start + params.duration_ms / 10;
    // (`src/whisper.cpp:7442`) and runs the encoder/decoder
    // loop over `[seek_start, seek_end)` without bounding
    // `seek_end` to the actual mel length. Reads past the
    // real input are silently zero-filled by
    // `whisper_encode_internal`, so an `i32::MAX` duration
    // on a short buffer drives ~71 000 thirty-second
    // windows of zero-padded decode — looks like a hung
    // worker rather than a clean parameter error.
    //
    // The setter clamps to `>= 0`, but `Params::default()`
    // followed by a struct-update could bypass it; defend
    // here. `duration_ms == 0` is upstream's "to end of
    // input" sentinel and is always accepted. Negative
    // values (only reachable via the bypass) are rejected.
    //
    // Audio length in ms: integer-truncated
    // `samples.len() * 1000 / 16000`. Computed in i64 to
    // avoid overflow at the i32::MAX-sample upper bound the
    // earlier `SamplesOverflow` check already enforces.
    let offset_ms = i64::from(params.offset_ms());
    let duration_ms = i64::from(params.duration_ms());
    let audio_duration_ms = (samples.len() as i64) * 1000 / 16_000;
    if offset_ms < 0
      || duration_ms < 0
      || (duration_ms > 0 && offset_ms.saturating_add(duration_ms) > audio_duration_ms)
      || (duration_ms == 0 && offset_ms > audio_duration_ms)
    {
      return Err(WhisperError::InvalidDuration {
        offset_ms: params.offset_ms(),
        duration_ms: params.duration_ms(),
        audio_duration_ms,
      });
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

  /// Number of mel-spectrogram frames in the audio that the
  /// most recent [`State::full`] call processed. One frame =
  /// 10 ms of input audio (whisper's mel hop).
  ///
  /// Returns `0` for a poisoned state or when no inference
  /// has run yet on this state. Wraps
  /// `whisper_n_len_from_state`.
  pub fn n_mel_frames(&self) -> i32 {
    let Some(state) = self.raw() else { return 0 };
    // SAFETY: state non-null; pure read of an int field on
    // the state's mel struct, no allocation, no throw. The
    // `whisper_mel POD field default-init` patch in the
    // submodule guarantees this read is well-defined even
    // before any `State::full` has run.
    unsafe { sys::whisper_n_len_from_state(state) }
  }

  /// Print this `State`'s timing breakdown to stderr
  /// (load / fallbacks / mel / sample / encode / decode /
  /// batchd / prompt). Wraps the patched
  /// `whispercpp_print_timings_with_state(ctx, state)`
  /// (the upstream `whisper_print_timings(ctx)` reads
  /// `ctx->state` only, which is always null in this wrapper
  /// because `Context::new` uses the `_no_state`
  /// initializer).
  ///
  /// Output diverges from upstream in one place: the
  /// "total time" line (upstream's `ggml_time_us() -
  /// ctx->t_start_us`) is omitted. `t_start_us` is
  /// Context-shared in this wrapper's
  /// `Arc<Context>` + multiple-`State` pattern; pairing
  /// it with state-bound counters that
  /// [`Self::reset_timings`] can zero would produce
  /// internally inconsistent output (per-stage values at
  /// 0 alongside a non-zero "total" that started at
  /// Context init). The state-aware printer therefore
  /// emits only state-bound counters plus the load-time
  /// read from `ctx->t_load_us` (immutable after
  /// `whisper_init_*`).
  ///
  /// Safe under shared `&self` because `State` is
  /// `Send`-but-not-`Sync`; no two threads can hold `&self`
  /// to the same `State` simultaneously, so the
  /// `state->n_*` / `state->t_*` reads can't race with a
  /// concurrent writer (`State::full` requires `&mut self`).
  /// No-op for a poisoned state.
  pub fn print_timings(&self) {
    let Some(state) = self.raw() else { return };
    let ctx = self.ctx.as_raw();
    // SAFETY: ctx and state are both non-null. The patched
    // shim re-implements upstream's print logic against the
    // passed `state` (instead of `ctx->state`); only writes
    // to stderr, no allocation, no throw.
    unsafe { sys::whispercpp_print_timings_with_state(ctx, state) };
  }

  /// Reset this `State`'s timing accumulators (encode /
  /// decode / sample / mel / batchd / prompt counters and
  /// elapsed-time totals). Wraps the patched
  /// `whispercpp_reset_timings_with_state(state)`.
  ///
  /// Scope: state-bound only. Does NOT touch
  /// `ctx->t_start_us` (the Context-shared timestamp
  /// upstream's `whisper_reset_timings` rebases). In the
  /// wrapper's `Arc<Context>` + multiple-`State` pattern
  /// rebasing that field from one State would silently
  /// invalidate any sibling State's wall-clock readings on
  /// the same Context, and would race against concurrent
  /// `print_timings` calls on those siblings. The matching
  /// [`Self::print_timings`] omits the total-time line for
  /// the same reason, so the post-reset output stays
  /// internally consistent (every line either reset or
  /// load-time).
  ///
  /// Useful when the same `State` is reused across multiple
  /// distinct runs and you want per-run numbers. Safe under
  /// `&mut self` exclusive access — `State` is
  /// `Send`-but-not-`Sync`, so the borrow checker prevents
  /// any concurrent reader (`Self::print_timings`) or writer
  /// (`Self::full`) from observing a mid-write state.
  /// No-op for a poisoned state.
  pub fn reset_timings(&mut self) {
    let Some(state) = self.raw() else { return };
    // SAFETY: state non-null. The shim writes integer
    // fields on the state's accumulator struct; no
    // allocation, no throw. `&mut self` ensures no other
    // method on this `State` runs concurrently.
    unsafe { sys::whispercpp_reset_timings_with_state(state) };
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

  /// Iterate every segment produced by the most recent
  /// [`State::full`] call, in order. Equivalent to
  /// `(0..state.n_segments()).map(|i| state.segment(i).unwrap())`
  /// but composes with adapters (`.filter` / `.map` /
  /// `.collect`) without an index variable.
  ///
  /// The yielded [`Segment`] values borrow from `&self`, so
  /// the iterator (and every yielded `Segment`) cannot
  /// outlive the `State`. Multiple `Segment`s can be alive
  /// at once: each call to a `Segment` accessor is a pure
  /// read of immutable per-call output buffers, so no
  /// aliasing arises. A poisoned state yields zero items
  /// (its `n_segments()` is `0`).
  pub fn segments_iter(&self) -> Segments<'_> {
    Segments {
      state: self,
      next: 0,
      end: self.n_segments().max(0),
    }
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

  /// Iterate every token in this segment, in decode order.
  /// Equivalent to
  /// `(0..segment.n_tokens()).map(|j| segment.token(j).unwrap())`
  /// but composes with iterator adapters.
  ///
  /// The returned iterator owns a `Copy` of the `Segment`
  /// (which is itself a thin index + pointer projection
  /// from the parent [`State`]), so adapter chains like
  /// `state.segments_iter().flat_map(|s| s.tokens_iter())`
  /// compile — the iterator does not borrow from a
  /// closure-local `Segment` value. The `'state` lifetime
  /// on the returned [`Tokens`] still ties the iterator
  /// (and every yielded item's pointer projections) to the
  /// owning `State`.
  ///
  /// Yielded [`Token`] values are owned snapshots (the
  /// underlying `whisper_token_data` is value-typed), so
  /// they have no further lifetime constraint and can be
  /// collected into a `Vec<Token>` for use after the
  /// `State` is dropped.
  pub fn tokens_iter(&self) -> Tokens<'a> {
    Tokens {
      segment: *self,
      next: 0,
      end: self.n_tokens().max(0),
    }
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

/// Iterator over the segments of a [`State`].
///
/// Returned by [`State::segments_iter`]. Borrows from the
/// `State`; the borrow chain ties every yielded
/// [`Segment`]'s lifetime to that of the iterator's
/// underlying `&'a State`, so a yielded segment cannot
/// outlive the state.
///
/// The segment count is captured at construction by reading
/// `whisper_full_n_segments_from_state` once. Reads of
/// per-call output buffers from concurrent
/// `Segment` views are safe: the underlying state is
/// immutable for the duration of the borrow (`State::full`
/// requires `&mut self`, which Rust's borrow checker rules
/// out while `&self` is held by this iterator).
///
/// Implements [`Iterator`], [`ExactSizeIterator`],
/// [`DoubleEndedIterator`], and
/// [`core::iter::FusedIterator`]; the length is known
/// up-front and never changes, so reverse iteration via
/// `.rev()` and `next_back` are O(1) per call.
pub struct Segments<'a> {
  state: &'a State,
  next: i32,
  end: i32,
}

impl<'a> Iterator for Segments<'a> {
  type Item = Segment<'a>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.next >= self.end {
      return None;
    }
    // `State::segment(i)` would re-call `n_segments()` (FFI)
    // for its bounds check. We already captured `end` from
    // `n_segments()` at construction, and the borrow chain
    // pins the underlying `whisper_state` (the count cannot
    // change while `&self.state` is held — `State::full`
    // requires `&mut self`). Construct the `Segment`
    // directly to skip the redundant FFI call.
    //
    // `self.state.ptr` is a `pub(super)` field readable from
    // this module. `Some(_)` is implied by `end > 0` (a
    // poisoned state's `n_segments()` returns 0 via the
    // `self.raw()?` early-return), but checking it
    // defensively keeps this branch verifiable in
    // isolation.
    let state_ptr = self.state.ptr?;
    let idx = self.next;
    self.next += 1;
    Some(Segment {
      state: state_ptr,
      idx,
      _marker: core::marker::PhantomData,
    })
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    let remaining = (self.end - self.next).max(0) as usize;
    (remaining, Some(remaining))
  }
}

impl DoubleEndedIterator for Segments<'_> {
  fn next_back(&mut self) -> Option<Self::Item> {
    if self.next >= self.end {
      return None;
    }
    let state_ptr = self.state.ptr?;
    self.end -= 1;
    Some(Segment {
      state: state_ptr,
      idx: self.end,
      _marker: core::marker::PhantomData,
    })
  }
}

impl ExactSizeIterator for Segments<'_> {
  fn len(&self) -> usize {
    (self.end - self.next).max(0) as usize
  }
}

impl core::iter::FusedIterator for Segments<'_> {}

/// Iterator over the tokens of a [`Segment`].
///
/// Returned by [`Segment::tokens_iter`]. Owns a copy of
/// the [`Segment`] (which is `Copy`: thin index + pointer
/// projection), so adapter chains such as
/// `state.segments_iter().flat_map(|s| s.tokens_iter())`
/// compile — the iterator does not borrow a
/// closure-local `Segment` value. The `'state` lifetime
/// transitively prevents the iterator from outliving the
/// owning [`State`].
///
/// Yielded [`Token`] values are owned, value-typed
/// snapshots projected from `whisper_token_data`. They
/// carry no borrow and can be collected into a
/// `Vec<Token>` for use after the iterator (and the
/// `State`) is dropped.
///
/// The token count is captured at construction by reading
/// `whisper_full_n_tokens_from_state(state, segment_idx)`
/// once. Implements [`Iterator`], [`ExactSizeIterator`],
/// [`DoubleEndedIterator`], and
/// [`core::iter::FusedIterator`].
pub struct Tokens<'state> {
  segment: Segment<'state>,
  next: i32,
  end: i32,
}

impl Iterator for Tokens<'_> {
  type Item = Token;

  fn next(&mut self) -> Option<Self::Item> {
    if self.next >= self.end {
      return None;
    }
    let tok_idx = self.next;
    self.next += 1;
    // SAFETY: `tok_idx ∈ [0, end)` by the bounds check
    // above; `end` was captured from `self.segment.n_tokens()`
    // at construction. The borrow chain (`Tokens<'state>` →
    // owned `Segment<'state>` → `&'state State`) pins the
    // underlying `whisper_state` for the iterator's
    // lifetime — `State::full` requires `&mut self` on the
    // owning State, so the token count cannot change
    // mid-iteration. Skipping `Segment::token`'s
    // `n_tokens()` FFI bounds-check shaves one FFI call
    // per yielded token, which is the dominant cost on
    // long segments (hundreds to low thousands of tokens
    // per state).
    let raw = unsafe {
      sys::whisper_full_get_token_data_from_state(
        self.segment.state.as_ptr(),
        self.segment.idx,
        tok_idx,
      )
    };
    Some(Token::from_raw(raw))
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    let remaining = (self.end - self.next).max(0) as usize;
    (remaining, Some(remaining))
  }
}

impl DoubleEndedIterator for Tokens<'_> {
  fn next_back(&mut self) -> Option<Self::Item> {
    if self.next >= self.end {
      return None;
    }
    self.end -= 1;
    // SAFETY: same as `next` — `self.end - 1 ∈ [next,
    // captured-end)` because `next < end` was just
    // checked.
    let raw = unsafe {
      sys::whisper_full_get_token_data_from_state(
        self.segment.state.as_ptr(),
        self.segment.idx,
        self.end,
      )
    };
    Some(Token::from_raw(raw))
  }
}

impl ExactSizeIterator for Tokens<'_> {
  fn len(&self) -> usize {
    (self.end - self.next).max(0) as usize
  }
}

impl core::iter::FusedIterator for Tokens<'_> {}

/// `for seg in &state { ... }` yields the same items as
/// [`State::segments_iter`]. Provided so the segment
/// iteration reads as one of Rust's standard collection
/// idioms.
impl<'a> IntoIterator for &'a State {
  type Item = Segment<'a>;
  type IntoIter = Segments<'a>;

  fn into_iter(self) -> Segments<'a> {
    self.segments_iter()
  }
}

/// `for tok in seg { ... }` yields the same items as
/// [`Segment::tokens_iter`]. By-value form is cheap
/// because [`Segment`] is `Copy`.
impl<'a> IntoIterator for Segment<'a> {
  type Item = Token;
  type IntoIter = Tokens<'a>;

  fn into_iter(self) -> Tokens<'a> {
    self.tokens_iter()
  }
}

/// `for tok in &seg { ... }` form, mirroring the
/// `for x in &collection` idiom. Equivalent to the
/// by-value impl above (since `Segment` is `Copy`).
impl<'a> IntoIterator for &Segment<'a> {
  type Item = Token;
  type IntoIter = Tokens<'a>;

  fn into_iter(self) -> Tokens<'a> {
    self.tokens_iter()
  }
}

/// Per-token data exposed by whisper.cpp.
///
/// Read-only snapshot. All fields are private; access goes
/// through `const fn` accessors to keep the public surface
/// stable as `whisper_token_data` evolves upstream.
///
/// # Two timestamp sources
///
/// Whisper.cpp can produce per-token timing two different ways,
/// and they live in different fields:
///
/// * [`Self::t0`] / [`Self::t1`] come from the **timestamp-token
///   path** — whisper.cpp's standard heuristic that pairs each
///   token with the surrounding `<|t_x|>` markers. These are
///   populated when [`crate::Params::set_token_timestamps`] is
///   `true`, regardless of DTW.
/// * [`Self::t_dtw`] comes from the **DTW backtrace** —
///   independently derived from cross-attention weights of the
///   alignment heads. Only populated when DTW is enabled at
///   [`Context`] load time via
///   [`crate::ContextParams::with_dtw_token_timestamps`] (and
///   a non-`None`
///   [`crate::ContextParams::with_dtw_aheads_preset`]).
///
/// DTW is generally more robust to long silences and repeated
/// tokens than the timestamp-token path; the timestamp-token
/// path is cheaper but more sensitive to attention misallocation.
#[derive(Debug, Clone, Copy)]
pub struct Token {
  id: i32,
  p: f32,
  plog: f32,
  pt: f32,
  ptsum: f32,
  t0: i64,
  t1: i64,
  t_dtw: i64,
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

  /// Token start time, in centiseconds, derived from the
  /// timestamp-token path. `0` when
  /// [`crate::Params::set_token_timestamps`] is `false`.
  /// **Not** the DTW timestamp — see [`Self::t_dtw`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn t0(&self) -> i64 {
    self.t0
  }

  /// Token end time, in centiseconds, derived from the
  /// timestamp-token path. `0` when
  /// [`crate::Params::set_token_timestamps`] is `false`.
  /// **Not** the DTW timestamp — see [`Self::t_dtw`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn t1(&self) -> i64 {
    self.t1
  }

  /// DTW-derived token timestamp, in centiseconds. Roughly the
  /// moment in the audio at which whisper.cpp emitted this
  /// token, computed by running DTW over the configured
  /// alignment heads' cross-attention weights.
  ///
  /// Returns `Some(t)` when DTW computed a real timestamp for
  /// this token. Returns `None` when DTW timing is unavailable
  /// for any of these reasons:
  ///
  /// * DTW was not enabled at [`Context`] construction
  ///   (`with_dtw_token_timestamps(false)`, or preset left at
  ///   [`crate::AlignmentHeadsPreset::None`]).
  /// * The token is a non-text (special / timestamp) token —
  ///   DTW only writes timing for text tokens.
  /// * DTW skipped this segment because the chunk's
  ///   `audio_ctx` (overridden by
  ///   [`crate::Params::set_audio_ctx`]) was too small for
  ///   the chunk duration.
  /// * DTW skipped this segment because the audio window was
  ///   too short for the median-filter pass
  ///   (`n_audio_tokens <= 1`, ≤20 ms).
  ///
  /// The `whispercpp-sys: dtw t_dtw sentinel init` patch in
  /// `whisper.cpp` initialises every text token's `t_dtw` to
  /// `-1` at the start of the DTW pass; successful
  /// computation overwrites with a non-negative timestamp,
  /// while skip paths leave the sentinel in place. Negative
  /// timestamps are unreachable for valid DTW output, so `-1`
  /// uniquely identifies "unavailable."
  ///
  /// Whisper.cpp ships validated alignment-head presets for
  /// every standard checkpoint (see
  /// [`crate::AlignmentHeadsPreset`]); using a preset that
  /// doesn't match the loaded model produces unreliable DTW
  /// timings without erroring — but this method still returns
  /// `Some(...)` because the values were "computed", just
  /// not meaningfully. Match the preset to the model.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn t_dtw(&self) -> Option<i64> {
    if self.t_dtw < 0 {
      None
    } else {
      Some(self.t_dtw)
    }
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
      t_dtw: raw.t_dtw,
      vlen: raw.vlen,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// `Token::from_raw` projects every field — including
  /// [`Token::t_dtw`], which earlier
  /// versions of this wrapper missed entirely (the C struct
  /// carried it but the safe view didn't surface it). Pin
  /// the projection so a future refactor can't quietly drop
  /// a field again.
  #[test]
  fn token_from_raw_projects_every_field_including_t_dtw() {
    let raw = sys::whisper_token_data {
      id: 1234,
      tid: 5678,
      p: 0.8,
      plog: -0.22,
      pt: 0.05,
      ptsum: 0.12,
      t0: 100,
      t1: 250,
      t_dtw: 175,
      vlen: 0.42,
    };
    let tok = Token::from_raw(raw);
    assert_eq!(tok.id(), 1234);
    assert!((tok.p() - 0.8).abs() < 1e-6);
    assert!((tok.plog() - -0.22).abs() < 1e-6);
    assert!((tok.pt() - 0.05).abs() < 1e-6);
    assert!((tok.ptsum() - 0.12).abs() < 1e-6);
    assert_eq!(tok.t0(), 100);
    assert_eq!(tok.t1(), 250);
    assert_eq!(
      tok.t_dtw(),
      Some(175),
      "Token::from_raw must project the DTW timestamp",
    );
    assert!((tok.vlen() - 0.42).abs() < 1e-6);
  }

  /// The DTW timestamp is independent from `t0`/`t1` — it
  /// comes from a different mechanism (cross-attention DTW vs.
  /// timestamp-token decoding). Confirm the safe view exposes
  /// distinct values rather than aliasing them.
  #[test]
  fn t_dtw_is_independent_of_t0_t1() {
    let raw = sys::whisper_token_data {
      id: 0,
      tid: 0,
      p: 0.0,
      plog: 0.0,
      pt: 0.0,
      ptsum: 0.0,
      t0: 100,
      t1: 200,
      t_dtw: 150,
      vlen: 0.0,
    };
    let tok = Token::from_raw(raw);
    assert_eq!(tok.t0(), 100);
    assert_eq!(tok.t1(), 200);
    assert_eq!(tok.t_dtw(), Some(150));
    // Sanity: distinct values flow through distinct accessors,
    // not collapsed into one.
    assert_ne!(tok.t_dtw(), Some(tok.t0()));
    assert_ne!(tok.t_dtw(), Some(tok.t1()));
  }

  /// `t_dtw == -1` is the sentinel set by the
  /// `whispercpp-sys: dtw t_dtw sentinel init` patch when DTW
  /// is enabled but skipped for a segment (audio_ctx mismatch
  /// or short-window medfilt). The wrapper must surface that
  /// as `None` so callers can distinguish "DTW skipped" from
  /// "DTW computed at audio offset 0."
  #[test]
  fn t_dtw_sentinel_minus_one_maps_to_none() {
    let raw = sys::whisper_token_data {
      id: 0,
      tid: 0,
      p: 0.0,
      plog: 0.0,
      pt: 0.0,
      ptsum: 0.0,
      t0: 0,
      t1: 0,
      t_dtw: -1,
      vlen: 0.0,
    };
    let tok = Token::from_raw(raw);
    assert_eq!(
      tok.t_dtw(),
      None,
      "t_dtw == -1 must surface as None (DTW unavailable for token)",
    );
  }

  /// `t_dtw == 0` is a *valid* DTW result for a token that
  /// starts at audio offset 0. It must NOT be confused with
  /// the unavailable sentinel — pin so a future "treat 0 as
  /// missing" refactor can't silently break this.
  #[test]
  fn t_dtw_zero_maps_to_some_zero() {
    let raw = sys::whisper_token_data {
      id: 0,
      tid: 0,
      p: 0.0,
      plog: 0.0,
      pt: 0.0,
      ptsum: 0.0,
      t0: 0,
      t1: 0,
      t_dtw: 0,
      vlen: 0.0,
    };
    let tok = Token::from_raw(raw);
    assert_eq!(
      tok.t_dtw(),
      Some(0),
      "t_dtw == 0 is a valid timestamp (token at audio start), not the sentinel",
    );
  }

  // ── Iterator tests (issue #3) ────────────────────────────
  //
  // All iterator unit tests run on a **poisoned** `State`
  // (its `ptr` is `None`, so `n_segments()` returns `0` and
  // `segment(_)` returns `None`). Real-state coverage —
  // segment counts > 0 and per-segment token streams —
  // requires a model file and lives in the integration test
  // suite. The tests here pin the Rust-side iterator
  // contract: empty-state correctness, `ExactSizeIterator`
  // length, fused behaviour, and that the borrow chain
  // composes (multiple iterators alive concurrently, nested
  // iteration). The compile-only test below proves the
  // lifetime threading is sound for the non-empty case too.

  /// Build a poisoned `State` and forget the dangling Arc on
  /// drop so the test never hits `whisper_free` on a fake
  /// pointer. Returns a `ManuallyDrop`-shaped pair: the
  /// `State` to drive iterators on, plus a manual cleanup
  /// closure that mem::forgets the underlying Arc payload.
  fn poisoned_state_for_test() -> State {
    // SAFETY: `Context::dangling_for_test` is unsafe
    // because its `Drop` would call `whisper_free` on a
    // dangling pointer. We satisfy the precondition by
    // `mem::forget`'ing the only `Arc<Context>` we hold
    // here AND by `forget_poisoned_state` at every call
    // site of this helper. The `State` returned shares
    // its own Arc handle with the leaked one above; when
    // the State drops, refcount goes 2 → 1, but the
    // forgotten Arc keeps the count at 1 forever — so
    // `Arc::drop` never reaches refcount 0 and
    // `Context::drop` never runs.
    //
    // Miri note: every call to this helper leaks one
    // `ArcInner` (40 bytes) by design — Miri's default
    // leak checker correctly flags it. Tests that drive
    // this fixture are tagged `#[cfg_attr(miri, ignore =
    // "...")]` so Miri skips them; pure compile-time
    // tests (PhantomData / trait-bound assertions) stay
    // covered.
    let ctx = Arc::new(unsafe { Context::dangling_for_test() });
    let state = State::poisoned_for_test(Arc::clone(&ctx));
    core::mem::forget(ctx);
    state
  }

  /// Drop helper: leak the State's internal Arc<Context> so
  /// the dangling pointer in the test Context never reaches
  /// `whisper_free`. Mirrors the `mem::forget` pattern in
  /// `context.rs::tests::fresh_context_marker_starts_unpoisoned`.
  fn forget_poisoned_state(state: State) {
    core::mem::forget(state);
  }

  /// Empty (poisoned) state yields zero segments. Exercises
  /// the base case the iterator MUST handle: a state whose
  /// `n_segments()` returned `0`. Anything that yields a
  /// nonzero count or panics on this input is broken.
  #[cfg_attr(
    miri,
    ignore = "intentional Arc leak in poisoned_state_for_test fixture"
  )]
  #[test]
  fn segments_iter_empty_state_yields_zero_items() {
    let state = poisoned_state_for_test();
    let count = state.segments_iter().count();
    assert_eq!(count, 0, "poisoned state must yield zero segments");
    forget_poisoned_state(state);
  }

  /// Iterator length agrees with `n_segments()`. Pins the
  /// contract that a future refactor of either side (e.g.
  /// caching the count differently) can't desynchronise.
  #[cfg_attr(
    miri,
    ignore = "intentional Arc leak in poisoned_state_for_test fixture"
  )]
  #[test]
  fn segments_iter_count_matches_n_segments() {
    let state = poisoned_state_for_test();
    let expected = state.n_segments();
    assert_eq!(expected, 0, "test fixture is poisoned");
    let actual = state.segments_iter().count() as i32;
    assert_eq!(actual, expected);
    forget_poisoned_state(state);
  }

  /// `ExactSizeIterator::len()` returns the same value as
  /// `count()` (and matches `n_segments()`). The
  /// `len()` impl is what `Vec::from_iter` uses to pre-
  /// allocate, so a wrong `len()` would either over-allocate
  /// or panic.
  #[cfg_attr(
    miri,
    ignore = "intentional Arc leak in poisoned_state_for_test fixture"
  )]
  #[test]
  fn segments_iter_exact_size_len_matches_count() {
    let state = poisoned_state_for_test();
    let iter = state.segments_iter();
    let len_before = iter.len();
    let counted = iter.count();
    assert_eq!(len_before, counted);
    assert_eq!(len_before, 0);
    forget_poisoned_state(state);
  }

  /// `size_hint` returns `(len, Some(len))` because the
  /// segment count is known up-front and never changes
  /// mid-iteration. Pins the lower bound + upper bound so
  /// adapters like `Vec::extend` can pre-allocate optimally.
  #[cfg_attr(
    miri,
    ignore = "intentional Arc leak in poisoned_state_for_test fixture"
  )]
  #[test]
  fn segments_iter_size_hint_is_exact() {
    let state = poisoned_state_for_test();
    let iter = state.segments_iter();
    let (lower, upper) = iter.size_hint();
    assert_eq!(lower, 0);
    assert_eq!(upper, Some(0));
    forget_poisoned_state(state);
  }

  /// `FusedIterator`: once exhausted, repeat `next()` calls
  /// keep returning `None` rather than producing items
  /// again. The `fuse()`-free explicit impl on `Segments`
  /// promises this — pin it.
  #[cfg_attr(
    miri,
    ignore = "intentional Arc leak in poisoned_state_for_test fixture"
  )]
  #[test]
  fn segments_iter_fused_after_exhaustion() {
    let state = poisoned_state_for_test();
    let mut iter = state.segments_iter();
    assert!(iter.next().is_none());
    assert!(iter.next().is_none());
    assert!(iter.next().is_none());
    forget_poisoned_state(state);
  }

  /// Two `Segments` iterators alive simultaneously must not
  /// fight: both borrow `&self`, and `&self` is `Copy`-able
  /// at the borrow level (multiple shared references are
  /// fine). The iterator only mutates its own index counter,
  /// not any shared state, so this exercises the
  /// "concurrent reads safe" claim from the doc comment.
  #[cfg_attr(
    miri,
    ignore = "intentional Arc leak in poisoned_state_for_test fixture"
  )]
  #[test]
  fn multiple_segments_iter_alive_concurrently() {
    let state = poisoned_state_for_test();
    let it1 = state.segments_iter();
    let it2 = state.segments_iter();
    assert_eq!(it1.len(), it2.len());
    assert_eq!(it1.count(), 0);
    assert_eq!(it2.count(), 0);
    forget_poisoned_state(state);
  }

  /// Iterator composes with adapters (`map`, `collect`).
  /// Pins that the trait impl is shaped right for the
  /// standard library's adapter chain — a common breakage
  /// when iterator types accidentally pick up unintended
  /// trait bounds.
  #[cfg_attr(
    miri,
    ignore = "intentional Arc leak in poisoned_state_for_test fixture"
  )]
  #[test]
  fn segments_iter_composes_with_adapters() {
    let state = poisoned_state_for_test();
    let collected: Vec<_> = state.segments_iter().map(|seg| seg.t0()).collect();
    assert!(collected.is_empty());
    forget_poisoned_state(state);
  }

  /// Compile-only: nested iteration `for seg in
  /// state.segments_iter() { for tok in seg.tokens_iter()
  /// { ... } }` typechecks. `Tokens<'state>` owns a copy
  /// of the `Segment`, so the inner loop's iterator does
  /// not borrow the closure-local `seg` value — adapter
  /// composition (see
  /// `tokens_iter_composes_with_flat_map`) works for the
  /// same reason. This test never executes the inner body
  /// on the poisoned fixture, but the body is still
  /// parsed and type-checked.
  #[cfg_attr(
    miri,
    ignore = "intentional Arc leak in poisoned_state_for_test fixture"
  )]
  #[test]
  fn nested_segments_and_tokens_iter_compiles() {
    let state = poisoned_state_for_test();
    let mut total: i32 = 0;
    for seg in state.segments_iter() {
      for tok in seg.tokens_iter() {
        // Use the token so the type-check is real (a no-op
        // closure body could be elided). On the poisoned
        // fixture this branch is unreachable.
        total = total.wrapping_add(tok.id());
      }
    }
    assert_eq!(total, 0);
    forget_poisoned_state(state);
  }

  /// Type-shape pin: `Segments` is `Iterator<Item =
  /// Segment<'_>>` and `Tokens` is `Iterator<Item =
  /// Token>`. Pin via a function-pointer cast that
  /// requires the trait bound at the type-system level —
  /// a future change that broke either bound would fail
  /// to compile here.
  #[cfg_attr(
    miri,
    ignore = "intentional Arc leak in poisoned_state_for_test fixture"
  )]
  #[test]
  fn iterator_type_bounds_are_correct() {
    fn assert_iter<I: Iterator>(_: I) {}
    fn assert_exact_size<I: ExactSizeIterator>(_: I) {}
    fn assert_fused<I: core::iter::FusedIterator>(_: I) {}
    let state = poisoned_state_for_test();
    assert_iter(state.segments_iter());
    assert_exact_size(state.segments_iter());
    assert_fused(state.segments_iter());
    forget_poisoned_state(state);
  }

  /// Adapter composition: `flat_map` over `segments_iter`
  /// returning each segment's `tokens_iter` must compile
  /// and produce a single flat token stream. The previous
  /// design (`Tokens<'seg, 'state>` borrowing
  /// `&'seg Segment<'state>`) failed to compile because
  /// the closure-local `Segment` value goes out of scope
  /// when the closure returns; the iterator would have
  /// been a dangling borrow. With `Tokens<'state>` owning
  /// a `Copy` of the `Segment`, the inner iterator
  /// outlives the closure and the chain works.
  ///
  /// Pin runtime + type behaviour: the poisoned fixture
  /// yields zero segments, so the flattened stream is
  /// empty. The type-check is the load-bearing assertion;
  /// the runtime count is a defense-in-depth check that
  /// the iterator wires up correctly.
  #[cfg_attr(
    miri,
    ignore = "intentional Arc leak in poisoned_state_for_test fixture"
  )]
  #[test]
  fn tokens_iter_composes_with_flat_map() {
    let state = poisoned_state_for_test();
    let total: usize = state
      .segments_iter()
      .flat_map(|seg| seg.tokens_iter())
      .count();
    assert_eq!(total, 0);
    // Also pin Iterator-trait-bound on the flattened
    // chain. Compile-time check that `flat_map` returns
    // `Iterator<Item = Token>`.
    fn assert_token_iter<I: Iterator<Item = Token>>(_: I) {}
    let state2 = poisoned_state_for_test();
    assert_token_iter(state2.segments_iter().flat_map(|seg| seg.tokens_iter()));
    forget_poisoned_state(state);
    forget_poisoned_state(state2);
  }

  /// `for seg in &state` works via [`IntoIterator for
  /// &State`]. Pin the trait shape (rather than the
  /// runtime count, which is zero on the poisoned
  /// fixture) so a future change can't quietly drop the
  /// impl. The compile-time check is the load-bearing
  /// assertion.
  #[cfg_attr(
    miri,
    ignore = "intentional Arc leak in poisoned_state_for_test fixture"
  )]
  #[test]
  fn into_iter_for_state_ref_yields_segments() {
    let state = poisoned_state_for_test();
    fn assert_segment_iter<'a, I: IntoIterator<Item = Segment<'a>>>(_: I) {}
    assert_segment_iter(&state);
    let count = (&state).into_iter().count();
    assert_eq!(count, 0);
    forget_poisoned_state(state);
  }

  /// `Segment: IntoIterator<Item = Token>` (by-value;
  /// `Segment` is `Copy` so the consumption is cheap)
  /// and `&Segment: IntoIterator<Item = Token>`. Both are
  /// type-pinned here. Runtime exercise is impossible
  /// without a real model (the poisoned fixture yields
  /// zero segments, so we can't construct a `Segment`).
  /// The compile-time `assert_token_iter` calls verify
  /// the trait shape. Regression coverage uses the
  /// `flat_map` test which already drives the closure
  /// path.
  #[test]
  fn into_iter_for_segment_compiles() {
    fn assert_token_iter<I: IntoIterator<Item = Token>>(_: PhantomData<I>) {}
    use core::marker::PhantomData;
    assert_token_iter::<Segment<'_>>(PhantomData);
    assert_token_iter::<&Segment<'_>>(PhantomData);
  }

  /// `DoubleEndedIterator` for `Segments`: `next_back`
  /// returns segments in reverse index order, and
  /// `.rev()` chains correctly. Empty fixture means the
  /// runtime check is "never yields"; the load-bearing
  /// assertion is the trait-bound on `.rev()`.
  #[cfg_attr(
    miri,
    ignore = "intentional Arc leak in poisoned_state_for_test fixture"
  )]
  #[test]
  fn segments_iter_double_ended_compiles_and_empty_yields_none() {
    let state = poisoned_state_for_test();
    fn assert_dei<I: DoubleEndedIterator>(_: I) {}
    assert_dei(state.segments_iter());
    let mut iter = state.segments_iter();
    assert!(iter.next_back().is_none());
    let rev_count = state.segments_iter().rev().count();
    assert_eq!(rev_count, 0);
    forget_poisoned_state(state);
  }

  /// `DoubleEndedIterator` for `Tokens`: same as above —
  /// trait shape pinned via `assert_dei` on a Tokens
  /// instance. Constructing a real `Tokens` without a
  /// model is impossible (we'd need a `Segment`, which
  /// requires a non-poisoned `State`), so this is a
  /// pure type-check.
  #[test]
  fn tokens_iter_double_ended_compiles() {
    fn assert_dei<I: DoubleEndedIterator>(_: PhantomData<I>) {}
    use core::marker::PhantomData;
    assert_dei::<Tokens<'_>>(PhantomData);
  }
}
