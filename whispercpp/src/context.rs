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

use smol_str::SmolStr;

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

/// Default DTW working-memory budget (128 MiB).
///
/// Forwarded into [`ContextParams::dtw_mem_size`] when callers
/// don't override it. Adequate for the small-head presets
/// (`Tiny*`, `Base*`, `Small`, `Medium`, `LargeV1`, `LargeV3`,
/// `LargeV3Turbo` — all ≤ 10 alignment heads). For higher-head
/// presets (`SmallEn` 19 heads, `MediumEn` 18 heads, `LargeV2`
/// 23 heads), [`Context::new`] silently raises the value to
/// the per-preset requirement returned by
/// [`required_dtw_mem_size_for`]; callers don't need to do
/// the math themselves.
///
/// Whisper.cpp's struct comment marks `dtw_mem_size` as
/// "TODO: remove" — the buffer is expected to migrate behind
/// the encoder's standard arena. Until then we forward it
/// faithfully; the safe API stays compatible when upstream
/// drops the field.
pub const DEFAULT_DTW_MEM_SIZE: usize = 128 * 1024 * 1024;

/// Absolute lower bound applied by
/// [`ContextParams::with_dtw_mem_size`] (and by
/// [`ContextParams::new`]).
///
/// Whisper.cpp's
/// `whisper_exp_compute_token_level_timestamps_dtw` allocates
/// a scratch `ggml_context` sized by `dtw_mem_size`. The DTW
/// pipeline then materialises three live `n_tokens ×
/// n_audio_tokens × n_heads × f32` tensors (the working
/// cross-attention tensor, the `ggml_norm` output, and the
/// `ggml_map_custom1` median-filter output) plus a small
/// backtrace lattice. The ggml context header alone needs a
/// few MiB before any tensor lands — anything below that
/// floor makes `ggml_init` return NULL and the next access
/// fault.
///
/// `ggml_init` returns NULL when the requested arena is too
/// small, and `ggml_new_tensor_3d` aborts (via `GGML_ASSERT`)
/// when the arena cannot fit a tensor. Both shapes terminate
/// the process from inside whisper.cpp without giving the
/// `whispercpp_full_with_state` exception shim a chance to
/// catch — `GGML_ASSERT` calls `abort()`, and the NULL deref
/// is a fatal signal. **Both are reachable from safe Rust**
/// if the budget is unconstrained.
///
/// Floor at 128 MiB covers the smallest preset's realistic
/// peak with comfortable headroom. Higher-head presets
/// require more; [`Context::new`] enforces the per-preset
/// minimum on top of this absolute floor — see
/// [`required_dtw_mem_size_for`] for the formula.
pub const MIN_DTW_MEM_SIZE: usize = DEFAULT_DTW_MEM_SIZE;

/// Upper bound applied by [`ContextParams::with_dtw_mem_size`]
/// (and by [`ContextParams::new`]).
///
/// `ggml_init` mallocs `dtw_mem_size + WHISPER_GGML_OBJECT_SIZE`
/// internally; passing `usize::MAX` overflows that addition and
/// drives `ggml_init` to NULL on the malloc step, with the
/// same null-deref / GGML_ASSERT consequences as the lower-
/// bound failure shape (see [`MIN_DTW_MEM_SIZE`]).
///
/// The cap is **target-pointer-width-dependent**:
///
/// * 64-bit (`target_pointer_width = "64"`): 4 GiB — three
///   orders of magnitude above the realistic worst case
///   ([`required_dtw_mem_size_for(LargeV2)`][required_dtw_mem_size_for]
///   = 278 MiB), so a `usize::MAX` slip collapses to a
///   large-but-allocatable value rather than an
///   overflow-induced abort.
/// * 32-bit (`target_pointer_width = "32"`): 1 GiB.
///   `4 * 1024 * 1024 * 1024 = 2^32` exceeds `usize::MAX =
///   2^32 - 1` on 32-bit targets, which would make the crate
///   fail to compile there. 1 GiB is still ~3.7× the
///   `LargeV2` per-preset minimum and well below
///   `usize::MAX`, so the safety property (saturate above
///   the realistic worst case to dodge `ggml_init` overflow)
///   is preserved.
/// * 16-bit (`target_pointer_width = "16"`): same value as
///   32-bit; falls back to the smaller cap. Whisper.cpp
///   does not realistically run on 16-bit targets.
#[cfg(target_pointer_width = "64")]
pub const MAX_DTW_MEM_SIZE: usize = 4 * 1024 * 1024 * 1024;

/// 32-bit / 16-bit ceiling — see [`MAX_DTW_MEM_SIZE`]'s
/// docstring on the 64-bit variant for the full explanation.
#[cfg(not(target_pointer_width = "64"))]
pub const MAX_DTW_MEM_SIZE: usize = 1024 * 1024 * 1024;

/// Clamp a DTW memory budget to `[MIN_DTW_MEM_SIZE,
/// MAX_DTW_MEM_SIZE]`. `const fn` so it composes inside
/// [`ContextParams::new`]'s defaults.
///
/// `0` and other below-floor values rise to
/// [`MIN_DTW_MEM_SIZE`]; `usize::MAX` and other ceiling-busting
/// values fall to [`MAX_DTW_MEM_SIZE`]. Both ends close
/// crash-from-safe-Rust paths inside whisper.cpp's DTW
/// allocator (see those constants' docs).
#[cfg_attr(not(tarpaulin), inline(always))]
const fn clamp_dtw_mem_size(n: usize) -> usize {
  if n < MIN_DTW_MEM_SIZE {
    MIN_DTW_MEM_SIZE
  } else if n > MAX_DTW_MEM_SIZE {
    MAX_DTW_MEM_SIZE
  } else {
    n
  }
}

/// Alignment-head count for a given preset.
///
/// Verified against `whisper.cpp/src/whisper.cpp:399-410`
/// (the `g_aheads_*` static arrays) and `:412-424` (the
/// `g_aheads` map that pairs each preset with its head count).
/// `None` and any never-mapped variant return `0`.
#[cfg_attr(not(tarpaulin), inline(always))]
const fn alignment_head_count(preset: AlignmentHeadsPreset) -> usize {
  match preset {
    AlignmentHeadsPreset::None => 0,
    AlignmentHeadsPreset::TinyEn => 8,
    AlignmentHeadsPreset::Tiny => 6,
    AlignmentHeadsPreset::BaseEn => 5,
    AlignmentHeadsPreset::Base => 8,
    AlignmentHeadsPreset::SmallEn => 19,
    AlignmentHeadsPreset::Small => 10,
    AlignmentHeadsPreset::MediumEn => 18,
    AlignmentHeadsPreset::Medium => 6,
    AlignmentHeadsPreset::LargeV1 => 9,
    AlignmentHeadsPreset::LargeV2 => 23,
    AlignmentHeadsPreset::LargeV3 => 10,
    AlignmentHeadsPreset::LargeV3Turbo => 6,
  }
}

/// Largest `n_text_ctx` (decoder text-context window) the
/// safe DTW wrapper budgets for.
///
/// Every standard whisper checkpoint — `tiny.en` through
/// `large-v3-turbo` — has `n_text_ctx = 448`, so this matches
/// the universe of officially-released models. Some
/// fine-tuned / extended-context checkpoints carry larger
/// values (the bundled GGUF loader accepts `n_text_ctx` up
/// to several thousand), and the DTW helper sizes its
/// working tensor from the actual decoder output rather than
/// from this constant. To prevent a non-standard model from
/// silently overflowing the [`required_dtw_mem_size_for`]
/// budget and tripping `GGML_ASSERT` inside
/// `ggml_new_tensor_3d` during decode, [`Context::new`]
/// reads the loaded model's `n_text_ctx` after init and
/// refuses to publish a `Context` that has DTW enabled
/// together with `n_text_ctx > SUPPORTED_DTW_N_TEXT_CTX`.
/// Affected callers can either:
///
/// 1. Disable DTW for that model — call
///    [`ContextParams::with_dtw_token_timestamps`] with
///    `false`. The rest of the API stays available.
/// 2. Use a standard checkpoint.
///
/// Pre-allocating for a higher upper bound (e.g. 2048) was
/// considered and rejected: it would force a ~3-4× larger
/// DTW arena (≥ 1.27 GiB on `LargeV2`) on every DTW-enabled
/// context, including the standard-checkpoint case that
/// dominates real usage.
pub const SUPPORTED_DTW_N_TEXT_CTX: i32 = 448;

/// Worst-case DTW scratch requirement for a given preset.
///
/// Whisper.cpp's
/// `whisper_exp_compute_token_level_timestamps_dtw` materialises
/// up to three live `n_tokens × n_audio_tokens × n_heads × f32`
/// tensors during the DTW pipeline (the working cross-attention
/// tensor, the `ggml_norm` output, and the `ggml_map_custom1`
/// median-filter output). Worst-case bounds:
///
/// * `n_tokens` ≤ [`SUPPORTED_DTW_N_TEXT_CTX`] — whisper's
///   `n_text_ctx` for every standard checkpoint. Non-standard
///   checkpoints with larger values are rejected by
///   [`Context::new`] when DTW is on; see that constant's
///   docs for the contract.
/// * `n_audio_tokens` = `n_frames / 2`, with `n_frames` capped
///   at `WHISPER_CHUNK_SIZE * 100 = 3000` (centiseconds for a
///   30 s chunk), giving a max of 1500.
/// * `n_heads` from `alignment_head_count(preset)` —
///   23 for `LargeV2`, 19 for `SmallEn`, 18 for `MediumEn`,
///   ≤ 10 for the rest.
///
/// Per-tensor: `SUPPORTED_DTW_N_TEXT_CTX × 1500 × n_heads ×
/// 4 bytes`. With three live tensors plus a 50% safety
/// margin for the backtrace lattice / tensor metadata /
/// backend compute scratch, the minimum scales linearly in
/// `n_heads`. For presets whose computed minimum falls below
/// [`MIN_DTW_MEM_SIZE`], the floor wins (the small-preset
/// case where ggml context overhead dominates).
///
/// Returns `0` for [`AlignmentHeadsPreset::None`] — DTW is
/// disabled and no scratch is needed.
///
/// Earlier versions of this wrapper derived the floor from
/// the wrong worst-case shapes (`n_audio_tokens ≤ 750` instead
/// of 1500, and an underestimate of the per-preset head
/// counts). The 128 MiB floor those numbers produced was too
/// small for `LargeV2` / `SmallEn` / `MediumEn` — `ggml_init`
/// could return NULL or `ggml_new_tensor_3d` could `GGML_ASSERT`
/// during decode. This function fixes the analysis by reading
/// head counts straight from
/// `whisper.cpp:399-424` and using whisper.cpp's actual
/// dimension caps.
#[cfg_attr(not(tarpaulin), inline(always))]
pub const fn required_dtw_mem_size_for(preset: AlignmentHeadsPreset) -> usize {
  let n_heads = alignment_head_count(preset);
  if n_heads == 0 {
    return 0;
  }
  // Worst-case dimensions baked from whisper.cpp:
  // - n_tokens (text context cap): SUPPORTED_DTW_N_TEXT_CTX
  // - n_audio_tokens: WHISPER_CHUNK_SIZE * 100 / 2 = 1500
  // - bytes per element (f32): 4
  let per_tensor = (SUPPORTED_DTW_N_TEXT_CTX as usize) * 1500 * n_heads * 4;
  // Three live large tensors during the DTW pipeline + 50%
  // safety margin for backtrace state, tensor metadata, and
  // ggml backend scratch.
  let with_safety = (per_tensor * 3) * 3 / 2;
  // Floor at MIN_DTW_MEM_SIZE — even small presets need at
  // least this much for ggml context overhead.
  if with_safety < MIN_DTW_MEM_SIZE {
    MIN_DTW_MEM_SIZE
  } else if with_safety > MAX_DTW_MEM_SIZE {
    MAX_DTW_MEM_SIZE
  } else {
    with_safety
  }
}

/// Which set of cross-attention heads whisper.cpp samples for
/// the DTW backtrace. Mirrors `whisper_alignment_heads_preset`.
///
/// Each shipping whisper checkpoint has its own set of "alignment
/// heads" — the decoder heads whose attention patterns correlate
/// best with the underlying acoustic timing. The presets below
/// pick those known-good heads; using the wrong preset for a
/// given checkpoint produces noisy timestamps. Match the preset
/// to the model file.
///
/// # Why `NTopMost` and `Custom` are not exposed
///
/// `WHISPER_AHEADS_N_TOP_MOST` is intentionally omitted because
/// the resulting alignment-head count is `n_top × n_text_head`,
/// which on a large model (32 layers × 20 heads) reaches 640
/// heads — pushing the DTW working tensor (`n_tokens ×
/// n_audio_tokens × n_heads × f32`) to ~860 MiB and overflowing
/// even [`MAX_DTW_MEM_SIZE`] under realistic decoder context.
/// `ggml_new_tensor_3d` aborts via `GGML_ASSERT` when the arena
/// cannot fit, terminating the process from inside whisper.cpp
/// before the exception shim can catch. Exposing the preset
/// requires the wrapper to compute scratch size from the
/// loaded model's head count first; that's not in this iteration.
///
/// `WHISPER_AHEADS_CUSTOM` is also omitted: the C variant
/// requires a pointer to a caller-owned `whisper_ahead` array,
/// which would force `ContextParams` to own a `Vec` and lose
/// the `Copy` derive — and shares the same scratch-size
/// validation gap as `N_TOP_MOST`.
///
/// Result: only the validated per-checkpoint presets ship, each
/// with a known-bounded alignment-head count whose scratch
/// requirement comfortably fits [`DEFAULT_DTW_MEM_SIZE`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentHeadsPreset {
  /// No DTW alignment heads — disables DTW even when
  /// [`ContextParams::dtw_token_timestamps`] is `true`.
  None,
  /// `tiny.en` — English-only.
  TinyEn,
  /// `tiny` — multilingual.
  Tiny,
  /// `base.en` — English-only.
  BaseEn,
  /// `base` — multilingual.
  Base,
  /// `small.en` — English-only.
  SmallEn,
  /// `small` — multilingual.
  Small,
  /// `medium.en` — English-only.
  MediumEn,
  /// `medium` — multilingual.
  Medium,
  /// `large-v1`.
  LargeV1,
  /// `large-v2`.
  LargeV2,
  /// `large-v3`.
  LargeV3,
  /// `large-v3-turbo` — the distilled-decoder variant used by
  /// whispery in production.
  LargeV3Turbo,
}

impl AlignmentHeadsPreset {
  /// Map to the C enum value bindgen produced. `const fn` so
  /// the conversion participates in `ContextParams`'s
  /// `with_*` chain.
  #[cfg_attr(not(tarpaulin), inline(always))]
  const fn to_raw(self) -> sys::whisper_alignment_heads_preset {
    match self {
      Self::None => sys::whisper_alignment_heads_preset_WHISPER_AHEADS_NONE,
      Self::TinyEn => sys::whisper_alignment_heads_preset_WHISPER_AHEADS_TINY_EN,
      Self::Tiny => sys::whisper_alignment_heads_preset_WHISPER_AHEADS_TINY,
      Self::BaseEn => sys::whisper_alignment_heads_preset_WHISPER_AHEADS_BASE_EN,
      Self::Base => sys::whisper_alignment_heads_preset_WHISPER_AHEADS_BASE,
      Self::SmallEn => sys::whisper_alignment_heads_preset_WHISPER_AHEADS_SMALL_EN,
      Self::Small => sys::whisper_alignment_heads_preset_WHISPER_AHEADS_SMALL,
      Self::MediumEn => sys::whisper_alignment_heads_preset_WHISPER_AHEADS_MEDIUM_EN,
      Self::Medium => sys::whisper_alignment_heads_preset_WHISPER_AHEADS_MEDIUM,
      Self::LargeV1 => sys::whisper_alignment_heads_preset_WHISPER_AHEADS_LARGE_V1,
      Self::LargeV2 => sys::whisper_alignment_heads_preset_WHISPER_AHEADS_LARGE_V2,
      Self::LargeV3 => sys::whisper_alignment_heads_preset_WHISPER_AHEADS_LARGE_V3,
      Self::LargeV3Turbo => sys::whisper_alignment_heads_preset_WHISPER_AHEADS_LARGE_V3_TURBO,
    }
  }
}

/// Knobs forwarded to `whisper_context_default_params` before
/// loading. Mirrors the subset of `whisper_context_params` whispery
/// uses today.
///
/// All fields are private; access goes through `const fn`
/// accessors and `with_*` builder methods so the type's invariants
/// stay encapsulated and the public surface evolves
/// independently of the underlying C struct.
///
/// # DTW (token-level alignment via cross-attention)
///
/// DTW is enabled at MODEL LOAD time, not per-decode. Whisper.cpp
/// builds a slightly different decoder graph when DTW is on (the
/// alignment heads' attention weights need to be exposed to the
/// post-decode DTW pass), so the choice has to be made before
/// [`Context::new`] runs.
///
/// Once enabled, every [`crate::State::full`] call against the
/// resulting context populates [`crate::Token::t_dtw`] alongside
/// the standard `t0`/`t1` timestamp-token timings. The DTW
/// timestamp is independently derived from the cross-attention
/// pattern and is generally more robust to long silences and
/// repeated tokens than the timestamp-token path.
#[derive(Debug, Clone, Copy)]
pub struct ContextParams {
  use_gpu: bool,
  gpu_device: i32,
  flash_attn: bool,
  dtw_token_timestamps: bool,
  dtw_aheads_preset: AlignmentHeadsPreset,
  dtw_mem_size: usize,
}

impl ContextParams {
  /// Defaults: GPU on (Metal/CUDA where compiled in), device 0,
  /// flash-attn off, DTW off.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      use_gpu: true,
      gpu_device: 0,
      flash_attn: false,
      dtw_token_timestamps: false,
      dtw_aheads_preset: AlignmentHeadsPreset::None,
      dtw_mem_size: DEFAULT_DTW_MEM_SIZE,
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

  // ── DTW (token-level alignment via cross-attention) ──────────

  /// Whether the loaded context will compute DTW per-token
  /// timestamps during decode.
  ///
  /// When `true`, the decoder graph is built to expose
  /// cross-attention weights from the heads selected by
  /// [`Self::dtw_aheads_preset`], and each
  /// [`crate::Token::t_dtw`] is populated after decode. Costs
  /// ~5–15% extra decode time and a one-time
  /// [`Self::dtw_mem_size`] allocation; eliminates a separate
  /// forced-alignment pass for callers that only need
  /// approximate per-token timing.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn dtw_token_timestamps(&self) -> bool {
    self.dtw_token_timestamps
  }

  /// Chained setter for [`Self::dtw_token_timestamps`].
  ///
  /// When enabling DTW, also pick a matching preset via
  /// [`Self::with_dtw_aheads_preset`] — leaving the preset on
  /// [`AlignmentHeadsPreset::None`] disables DTW even when this
  /// flag is `true`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_dtw_token_timestamps(mut self, on: bool) -> Self {
    self.dtw_token_timestamps = on;
    self
  }

  /// Which alignment-heads preset DTW samples. Each shipping
  /// whisper checkpoint has its own validated preset; mismatched
  /// presets produce noisy timestamps without erroring.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn dtw_aheads_preset(&self) -> AlignmentHeadsPreset {
    self.dtw_aheads_preset
  }

  /// Chained setter for [`Self::dtw_aheads_preset`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_dtw_aheads_preset(mut self, preset: AlignmentHeadsPreset) -> Self {
    self.dtw_aheads_preset = preset;
    self
  }

  /// Working-memory budget (in bytes) for the DTW backtrace.
  /// Default [`DEFAULT_DTW_MEM_SIZE`] (128 MiB).
  ///
  /// Whisper.cpp's struct comment flags this field as
  /// "TODO: remove" — the buffer is expected to migrate behind
  /// the encoder's standard arena. The Rust API will keep the
  /// setter when that lands so callers don't break; the value
  /// will simply become a no-op.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn dtw_mem_size(&self) -> usize {
    self.dtw_mem_size
  }

  /// Chained setter for [`Self::dtw_mem_size`].
  ///
  /// Clamped to `[MIN_DTW_MEM_SIZE, MAX_DTW_MEM_SIZE]`. Both
  /// ends close native-code abort paths reachable from safe
  /// Rust through whisper.cpp's DTW arena allocator — see the
  /// constants' docs for the full failure analysis.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_dtw_mem_size(mut self, n: usize) -> Self {
    self.dtw_mem_size = clamp_dtw_mem_size(n);
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
    // DTW (token-level alignment via cross-attention). The
    // `dtw_aheads` pointer-and-length pair stays at whatever
    // `whisper_context_default_params` sets it to (currently
    // `{ n_heads: 0, heads: NULL }`, which whisper.cpp reads
    // as "no custom heads — fall back to the preset").
    // `AlignmentHeadsPreset::Custom` is not exposed at the
    // safe-API level today; if a downstream caller needs
    // hand-tuned heads, that's the field to thread through.
    //
    // Two safety conversions before forwarding:
    //
    // 1. `dtw_token_timestamps && preset == None` is a
    //    misconfiguration whisper.cpp aborts on
    //    (`WHISPER_ASSERT(ctx->params.dtw_aheads_preset !=
    //    WHISPER_AHEADS_NONE)` in
    //    `whisper_exp_compute_token_level_timestamps_dtw`).
    //    Reachable from safe Rust because `with_dtw_*`
    //    setters compose independently. Coerce to "DTW off"
    //    instead of letting the abort cross the FFI: no
    //    preset means there's no useful DTW work to do
    //    anyway.
    //
    // 2. `dtw_mem_size` clamps to
    //    `[MIN_DTW_MEM_SIZE, MAX_DTW_MEM_SIZE]`. The setter
    //    already clamps, but we re-clamp here to defend
    //    against `ContextParams` constructed via field-init
    //    syntax in some future internal path (or callers
    //    poking through `Default + struct update`). Cheap.
    let dtw_on =
      params.dtw_token_timestamps() && params.dtw_aheads_preset() != AlignmentHeadsPreset::None;

    // Reject `dtw_on + flash_attn` BEFORE the FFI init.
    // whisper.cpp's loader logs a warning and silently
    // disables DTW under flash-attention
    // (`whisper.cpp:3956`). Without this check the safe Rust
    // API would return `Ok(Context)` for a configuration
    // whose docs promise `Token::t_dtw` will be populated —
    // every t_dtw stays at the default 0 with no signal to
    // the caller. Refuse the combination explicitly so the
    // caller has to disable one knob and document which.
    // The check happens before `init_lock` so we avoid
    // taking the global init mutex for a configuration we
    // know is going to fail.
    if dtw_on && params.flash_attn() {
      return Err(WhisperError::ContextLoad {
        path: smol_str::SmolStr::new(path_str.as_ref()),
        reason: SmolStr::new_static(
          "DTW token timestamps cannot be combined with flash_attn — \
           whisper.cpp silently disables DTW under flash_attn. \
           Set with_flash_attn(false) or with_dtw_token_timestamps(false).",
        ),
        code: None,
      });
    }

    cparams.dtw_token_timestamps = dtw_on;
    cparams.dtw_aheads_preset = if dtw_on {
      params.dtw_aheads_preset().to_raw()
    } else {
      AlignmentHeadsPreset::None.to_raw()
    };
    // `dtw_n_top` only matters when preset is N_TOP_MOST,
    // which is not exposed by the safe API
    // (see `AlignmentHeadsPreset`'s doc-comment for the
    // scratch-size analysis that motivated its omission).
    // Leave the C field at whatever
    // `whisper_context_default_params()` set it to.
    //
    // DTW memory budget: clamp the user value first, then —
    // when DTW is actually on — raise to the per-preset
    // minimum from `required_dtw_mem_size_for`. The 128 MiB
    // floor is adequate for small-head presets but
    // dangerously low for `SmallEn` / `MediumEn` / `LargeV2`,
    // whose 18–23 alignment heads drive the DTW working
    // tensor past the budget; without this raise the
    // `ggml_new_tensor_3d` call inside the DTW path
    // `GGML_ASSERT`s and aborts the process. Silent raise
    // matches the existing "clamp invalid inputs to safe"
    // pattern in `Params::new`.
    let clamped_user = clamp_dtw_mem_size(params.dtw_mem_size());
    cparams.dtw_mem_size = if dtw_on {
      let required = required_dtw_mem_size_for(params.dtw_aheads_preset());
      if clamped_user >= required {
        clamped_user
      } else {
        required
      }
    } else {
      clamped_user
    };

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
    // `std::bad_alloc` or `std::system_error` fires after
    // the raw `new whisper_context` succeeded, the
    // submodule's `init_context RAII exit` patch
    // (`try { ... } catch { whisper_free(ctx); throw; }`)
    // reclaims everything reachable through `ctx` —
    // `model.ctxs`, `model.buffers`, `state`.
    //
    // The model-load path closes its previously-leaky
    // windows around raw ggml_context / backend buffer
    // pointers via the patches `model_load RAII for raw
    // ggml allocations`, `model_load tensor-prep RAII`,
    // and `model_load buffer-registration RAII`. Each raw
    // pointer that survives a throwing registration sits
    // under a `ggml_context_ptr` / `ggml_backend_buffer_ptr`
    // guard; on throw the guard frees, on success the
    // ownership is committed to `model.ctxs` /
    // `model.buffers` BEFORE the guard releases. Both
    // structures are walked by `whisper_free`.
    //
    // The shim's thread-local sentinel distinguishes the
    // two flavours of NULL return — used here for
    // diagnostic value, not recovery classification:
    //
    // * `take_last_constructor_exception == 0` → upstream
    //   returned NULL via its own bool-failure path
    //   (file-not-found, wrong magic, backend refused).
    // * `take_last_constructor_exception != 0` → the shim
    //   caught a C++ throw; the RAII catch + per-pointer
    //   guards reclaimed every native allocation made on
    //   the failing path.
    //
    // Both surface as `ContextLoad` (retryable, no leak).
    // The sentinel value, when present, is embedded in
    // the error's `reason` field for log triage.
    let raw = unsafe { sys::whispercpp_init_from_file_no_state(cpath.as_ptr(), cparams) };

    if let Some(ptr) = NonNull::new(raw) {
      // DTW-enabled contexts validate that the loaded model's
      // text-context window fits the budget assumed by
      // [`required_dtw_mem_size_for`]. Standard whisper
      // checkpoints all carry `n_text_ctx = 448`, but the GGUF
      // loader accepts larger values from custom / extended-
      // context fine-tunes. If a non-standard model with
      // `n_text_ctx > SUPPORTED_DTW_N_TEXT_CTX` is loaded
      // alongside DTW, the DTW helper sizes its working tensor
      // from `state->aheads_cross_QKs->ne[0]` (= actual decoded
      // tokens, bounded by `n_text_ctx`) and overflows the
      // pre-allocated arena — `ggml_new_tensor_3d`
      // `GGML_ASSERT`s and the process aborts. Pre-allocating
      // for a higher `n_text_ctx` upper bound (e.g. 2048)
      // would force ~3-4× more DTW arena on every context;
      // refusing here keeps the common-case budget tight and
      // gives the caller an explicit recovery path.
      if dtw_on {
        // SAFETY: ptr is non-null (just unwrapped from
        // NonNull); pure C accessor reading a const field.
        let n_text_ctx = unsafe { sys::whisper_n_text_ctx(ptr.as_ptr()) };
        if n_text_ctx > SUPPORTED_DTW_N_TEXT_CTX {
          // SAFETY: ptr was returned by
          // `whispercpp_init_from_file_no_state` and held only
          // by us; nothing else has observed it.
          // `whisper_free` is the matching deallocator.
          unsafe { sys::whisper_free(ptr.as_ptr()) };
          return Err(WhisperError::ContextLoad {
            path: smol_str::SmolStr::new(path_str.as_ref()),
            reason: SmolStr::new_static(
              "DTW enabled with a model whose n_text_ctx exceeds SUPPORTED_DTW_N_TEXT_CTX (448) — \
               disable DTW (with_dtw_token_timestamps(false)) or use a standard checkpoint",
            ),
            code: None,
          });
        }
      }
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
    // Allocation-free reason on the caught-exception path.
    // The previous design `format!(...)` + `SmolStr::new(...)`
    // heap-allocated TWICE on the very path most likely
    // running under memory pressure (a recoverable
    // `bad_alloc` from upstream); both calls route through
    // the abort-on-OOM global allocator and could kill the
    // process while reporting the recoverable failure.
    // Static reason strings + a separate `code` field
    // construct the error without any further allocation
    // beyond the `path` SmolStr (whose allocation is
    // unavoidable — paths are caller-controlled and the
    // diagnostic needs to identify the file). Even that
    // inlines for paths ≤ 23 bytes.
    let (reason, code) = if exc != 0 {
      (
        SmolStr::new_static(
          "whispercpp_init_from_file_no_state caught C++ exception; \
           native cleanup completed via init_context RAII exit + model_load RAII guards",
        ),
        Some(exc),
      )
    } else {
      (
        SmolStr::new_static(
          "whispercpp_init_from_file_no_state returned NULL (upstream load failure, no native exception caught)",
        ),
        None,
      )
    };
    Err(WhisperError::ContextLoad {
      path: smol_str::SmolStr::new(path_str.as_ref()),
      reason,
      code,
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
    // Same flavour split as `Context::new`. Both NULL-return
    // paths in `whisper_init_state` are leak-free with the
    // submodule's RAII patches in place:
    //
    //   * Bool-failure paths each call `whisper_free_state(state);
    //     return nullptr;` before returning.
    //   * The outer `init_state RAII exit` patch wraps the
    //     throwing region in `try { ... } catch {
    //     whisper_free_state(state); throw; }`, so a caught
    //     exception leaves no partial state behind.
    //

    // Read the thread-local sentinel for diagnostic value
    // only. With the submodule's `init_state RAII exit` patch
    // both paths leave native memory clean:
    //   * `0`     → upstream bool-failure path (already freed)
    //   * `≠ 0`   → caught C++ exception that the RAII
    //               `try { ... } catch { whisper_free_state(state);
    //               throw; }` block freed before rethrowing
    //
    // Both surface as `StateInit { code: ... }` — retryable,
    // no Context poison.
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
        // deallocator. Drain the thread-local sentinel
        // unconditionally so a stale value from this call
        // doesn't leak into the next constructor's
        // catch-block. On the success-then-poisoned path we
        // expect sentinel == 0; the drain is defensive.
        unsafe { sys::whisper_free_state(state_ptr.as_ptr()) };
        let _ = unsafe { sys::whispercpp_take_last_constructor_exception() };
        return Err(WhisperError::ContextPoisoned);
      }
      // Constructor returned NULL while the Context was
      // poisoned by a sibling. The sentinel is drained
      // (defensive) but ContextPoisoned wins: the caller's
      // recovery contract is "drop the Context", and that
      // already covers any state-init detail we'd otherwise
      // report.
      // SAFETY: pure thread-local read+write; same thread as
      // the constructor call, no intervening shim entry.
      let _ = unsafe { sys::whispercpp_take_last_constructor_exception() };
      return Err(WhisperError::ContextPoisoned);
    }
    if let Some(state_ptr) = NonNull::new(raw) {
      // Drain a stray sentinel from a prior, possibly-aborted
      // call on this thread before publishing the state, so
      // a future `create_state` doesn't observe stale data.
      // SAFETY: pure thread-local read+write; same thread.
      let _ = unsafe { sys::whispercpp_take_last_constructor_exception() };
      return Ok(State::from_raw(state_ptr, Arc::clone(self)));
    }
    // NULL return on a healthy Context.
    //
    // Read the sentinel for diagnostic value: with the
    // submodule's `init_state RAII exit` patch in place, a
    // non-zero sentinel means whisper.cpp's outer
    // `try { ... } catch { whisper_free_state(state); throw; }`
    // already reclaimed the partial state before the rethrow
    // — there is nothing left to leak, and the failure is
    // recoverable. Both paths therefore report `StateInit`;
    // the sentinel only differentiates them for telemetry.
    //
    // The Context is NOT poisoned here. Sibling States and
    // future `create_state` calls remain valid; an immediate
    // retry under reduced load may succeed.
    //
    // SAFETY: pure C call; thread-local read on the same
    // thread, no other shim call between.
    let exc = unsafe { sys::whispercpp_take_last_constructor_exception() };
    Err(WhisperError::StateInit {
      code: if exc == 0 { None } else { Some(exc) },
    })
  }

  /// Internal: hand the raw pointer to siblings in this crate
  /// that need to call FFI functions taking `whisper_context*`.
  pub(crate) fn as_raw(&self) -> *mut sys::whisper_context {
    self.ptr.as_ptr()
  }

  /// Internal test-only constructor. Builds a `Context`
  /// whose `ptr` is `NonNull::dangling` — useful for unit
  /// tests that exercise Rust-side logic (e.g. iterator
  /// drivers on a poisoned `State`) without needing a real
  /// model file.
  ///
  /// # Safety
  ///
  /// The returned `Context`'s [`Drop`] impl
  /// unconditionally invokes `whisper_free(self.ptr.as_ptr())`,
  /// which would dereference `NonNull::dangling()` — UB. The
  /// caller MUST guarantee that the returned value (or any
  /// `Arc<Context>` derived from it) is `core::mem::forget`'d
  /// before its drop runs. The unsafety is on this
  /// constructor (not the resulting `Context`) so the
  /// precondition is enforced at every call site by the
  /// borrow checker via the `unsafe` block.
  ///
  /// `unsafe fn` is preferred over returning
  /// `ManuallyDrop<Self>` because production code (and the
  /// `State::poisoned_for_test` helper that consumes this)
  /// expects an `Arc<Context>`, not an
  /// `Arc<ManuallyDrop<Context>>`. The two are different
  /// types — `ManuallyDrop` IS sufficient to suppress the
  /// inner `Context`'s drop (its destructor is a no-op),
  /// so the UB itself would be prevented; the issue is
  /// purely API-shape compatibility with the production
  /// `Arc<Context>` field on `State`. The
  /// `PoisonedStateFixture` test guard sidesteps this by
  /// holding a separate `ManuallyDrop<Arc<Context>>` clone
  /// alongside the State — the `unsafe fn` here is the
  /// raw-pointer producer, the guard handles the leak
  /// invariant via composition.
  #[cfg(test)]
  pub(crate) unsafe fn dangling_for_test() -> Self {
    Self {
      ptr: NonNull::<sys::whisper_context>::dangling(),
      lost: AtomicBool::new(false),
      full_lock: Mutex::new(()),
    }
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
  /// * `token` is outside `[0, n_vocab)` (cheap pre-check;
  ///   avoids an FFI round-trip on caller-supplied invalid
  ///   ids).
  /// * `token` is in `[0, n_vocab)` but absent from the
  ///   loaded vocab table — sparse-vocab models can have
  ///   `hparams.n_vocab` larger than the number of entries
  ///   actually populated by the loader. The `whispercpp-sys:
  ///   token_to_str sparse-vocab no-throw` patch in
  ///   `whisper_token_to_str` returns NULL in this case
  ///   (was: `id_to_token.at(token)` threw `std::out_of_range`
  ///   across `extern "C"`, undefined behaviour).
  /// * the underlying `c_str` is NULL or non-UTF-8 (model
  ///   corruption).
  ///
  /// The returned slice borrows from a `std::string` owned by
  /// the context's vocab table; it stays valid for as long as
  /// `self` is alive. (Unlike [`system_info`], this does NOT
  /// alias mutable C++ state — `id_to_token` is built once at
  /// load time and never modified.)
  pub fn token_to_str(&self, token: i32) -> Option<&str> {
    // Cheap pre-check — saves an FFI round-trip when the
    // caller supplies an obviously-invalid id. The C-side
    // patch in `whisper_token_to_str` is what actually
    // makes the call no-throw on sparse-vocab misses.
    let n = self.n_vocab();
    if token < 0 || token >= n {
      return None;
    }
    // SAFETY: ctx pointer invariant. The C-side
    // `whisper_token_to_str` returns NULL on any miss
    // (out-of-range OR sparse-vocab gap), no throw across
    // the boundary.
    let raw = unsafe { sys::whisper_token_to_str(self.ptr.as_ptr(), token) };
    if raw.is_null() {
      return None;
    }
    // SAFETY: NUL-terminated; lives as long as Context.
    let bytes = unsafe { core::ffi::CStr::from_ptr(raw).to_bytes() };
    core::str::from_utf8(bytes).ok()
  }

  /// Raw byte view of a token's vocab entry. The same data
  /// [`Self::token_to_str`] returns, but as `&[u8]` so the
  /// caller can decide whether non-UTF-8 BPE-merge bytes are
  /// expected or not.
  ///
  /// Returns `None` for the same conditions as
  /// [`Self::token_to_str`] (out-of-range or sparse-vocab miss).
  ///
  /// **NUL-byte caveat.** The underlying C accessor returns a
  /// NUL-terminated `c_str`; if a token's vocab entry happens
  /// to contain an interior NUL byte the returned slice is
  /// truncated at that NUL. The standard whisper checkpoints
  /// don't produce vocab entries with interior NULs, but
  /// custom checkpoints theoretically could. If you need
  /// guaranteed full-byte access, call this and verify the
  /// expected length yourself, or upstream a length-aware
  /// C accessor.
  pub fn token_to_bytes(&self, token: i32) -> Option<&[u8]> {
    let n = self.n_vocab();
    if token < 0 || token >= n {
      return None;
    }
    // SAFETY: token bound checked above; the patched
    // `whisper_token_to_str` returns NULL on sparse-vocab
    // miss, no throw.
    let raw = unsafe { sys::whisper_token_to_str(self.ptr.as_ptr(), token) };
    if raw.is_null() {
      return None;
    }
    // SAFETY: NUL-terminated; lives as long as Context.
    let bytes = unsafe { core::ffi::CStr::from_ptr(raw).to_bytes() };
    Some(bytes)
  }

  // ── Special token ids (force-prefix decoding seeds) ────────

  /// `<|translate|>` — task token that selects the translate
  /// flow when prepended to the decoder prompt.
  pub fn token_translate(&self) -> i32 {
    // SAFETY: pure read of vocab table; no throw.
    unsafe { sys::whisper_token_translate(self.ptr.as_ptr()) }
  }

  /// `<|transcribe|>` — task token that selects transcription
  /// (the default).
  pub fn token_transcribe(&self) -> i32 {
    // SAFETY: pure read.
    unsafe { sys::whisper_token_transcribe(self.ptr.as_ptr()) }
  }

  /// `<|prev|>` — start-of-prev marker, prepends a
  /// previous-context prompt segment.
  pub fn token_prev(&self) -> i32 {
    // SAFETY: pure read.
    unsafe { sys::whisper_token_prev(self.ptr.as_ptr()) }
  }

  /// `<|nospeech|>` — emitted when the model classifies the
  /// audio as silence / non-speech.
  pub fn token_nosp(&self) -> i32 {
    // SAFETY: pure read.
    unsafe { sys::whisper_token_nosp(self.ptr.as_ptr()) }
  }

  /// `<|notimestamps|>` — disables timestamp-token emission.
  pub fn token_not(&self) -> i32 {
    // SAFETY: pure read.
    unsafe { sys::whisper_token_not(self.ptr.as_ptr()) }
  }

  /// `<|startoflm|>` — start-of-language-model marker (rare,
  /// used by some prompt-engineering setups).
  pub fn token_solm(&self) -> i32 {
    // SAFETY: pure read.
    unsafe { sys::whisper_token_solm(self.ptr.as_ptr()) }
  }

  /// Token id for a specific [`Lang`](crate::Lang) (the `<|en|>` / `<|zh|>`
  /// language tokens whisper.cpp emits at the start of every
  /// transcript).
  ///
  /// Returns `None` when:
  /// * the model is not multilingual (English-only
  ///   checkpoints have no language tokens at all);
  /// * `lang` is `Lang::Other(...)` with a code whisper.cpp's
  ///   global table doesn't recognise (`whisper_lang_id`
  ///   returns -1);
  /// * the language code contains an interior NUL byte;
  /// * the resolved token falls outside the model's
  ///   language-token range `(token_sot, token_translate)`.
  ///   Whisper.cpp's `whisper_token_lang(ctx, lang_id)` is a
  ///   bare `token_sot + 1 + lang_id` calculation with no
  ///   bounds check; for a checkpoint with fewer language
  ///   tokens than the global `g_lang` table has entries,
  ///   the result can collide with `token_translate` /
  ///   `token_transcribe` and silently corrupt the decoder
  ///   prompt. We validate the range here and return
  ///   `None` for out-of-range computations.
  ///
  /// Useful as input to [`Params::set_tokens`](crate::Params::set_tokens)
  /// for callers who want to seed the decoder with an explicit
  /// `<|lang|>` prefix instead of relying on auto-detect.
  pub fn token_for_lang(&self, lang: &crate::Lang) -> Option<i32> {
    if !self.is_multilingual() {
      return None;
    }
    let lang_id = crate::lang_id_for(lang.as_str())?;
    // SAFETY: pure read; ctx pointer invariant. Upstream's
    // `whisper_token_lang` returns
    // `vocab.token_sot + 1 + lang_id` — pure addition, no
    // allocation, no `at()` lookup, no throw.
    let token = unsafe { sys::whisper_token_lang(self.ptr.as_ptr(), lang_id) };
    let sot = self.token_sot();
    // SAFETY: ctx pointer invariant; pure vocab read.
    let translate = unsafe { sys::whisper_token_translate(self.ptr.as_ptr()) };
    if token > sot && token < translate {
      Some(token)
    } else {
      // Resolved token landed on a task-token slot or
      // further — model doesn't actually support this
      // language even though the global table did.
      None
    }
  }

  // ── Tokenisation ───────────────────────────────────────────

  /// Tokenise `text` into the model's vocabulary ids. Wraps
  /// `whisper_tokenize` through the
  /// `whispercpp_tokenize` exception-catching shim — upstream
  /// can throw `std::bad_alloc` from the internal
  /// `std::vector` / `std::string` builds, and a throw across
  /// `extern "C"` is undefined behaviour without the shim.
  ///
  /// Returns `None` on:
  /// * interior NUL byte in `text` (rejected at the safe-Rust
  ///   boundary before any FFI / allocation);
  /// * the fallible NUL-terminated copy fails (`try_reserve_exact`
  ///   reports an allocator failure on attacker-sized inputs
  ///   under memory pressure);
  /// * shim caught a C++ exception during tokenisation
  ///   (`std::bad_alloc` from the internal `std::vector` /
  ///   `std::string` builds). The shim signals this via
  ///   `INT_MIN` return; we surface it as `None`.
  ///
  /// **Sentinel discipline.** Upstream encodes
  /// "buffer too small, need `-return` more tokens" in the
  /// negative return domain — `whisper.cpp:4297`. The shim
  /// reserves `INT_MIN` (and ONLY `INT_MIN`) for caught
  /// exceptions so it cannot collide with any realistic
  /// `-needed_count`. An earlier design used the
  /// `WHISPERCPP_ERR_*` ladder at `-100..=-103`, which
  /// silently failed on any input requiring exactly
  /// 100..=103 tokens.
  ///
  /// Allocation cost: a single tokenise call against an
  /// initial capacity (256 ids — comfortable for typical
  /// segment-length inputs); on negative-return ("need more
  /// tokens") we grow the buffer once and retry. So the
  /// happy path is one upstream tokenisation; only inputs
  /// exceeding the initial capacity pay for a second.
  pub fn tokenize(&self, text: &str) -> Option<Vec<i32>> {
    // Build a NUL-terminated byte buffer with FALLIBLE
    // allocation. `CString::new(text)` would call
    // `Vec::from(slice)` internally, which uses the
    // infallible global allocator path and aborts the
    // process on OOM. A safe-Rust caller passing a long
    // attacker-controlled string under memory pressure
    // shouldn't kill the process; surface OOM as `None`.
    let bytes = text.as_bytes();
    if bytes.contains(&0) {
      return None;
    }
    let mut nul_terminated: Vec<u8> = Vec::new();
    nul_terminated.try_reserve_exact(bytes.len() + 1).ok()?;
    nul_terminated.extend_from_slice(bytes);
    nul_terminated.push(0);
    let cstr_ptr: *const core::ffi::c_char = nul_terminated.as_ptr().cast();

    let ctx_ptr = self.ptr.as_ptr();

    // Single-pass with retry. The previous round's design
    // probed via `whispercpp_token_count` and then wrote
    // via `whispercpp_tokenize` — TWO upstream
    // `tokenize(vocab, text)` invocations per Rust call.
    // The current `whispercpp-sys: no-log tokenize shim`
    // patch makes `whispercpp_tokenize` itself no-log on
    // too-small returns, so we can call it once with a
    // generous initial capacity and retry only on
    // negative-count: ONE upstream tokenization on the
    // happy path, two only on retry.
    //
    // 256 covers most realistic inputs (whisper segments
    // typically tokenize to 50–150 tokens; longer texts
    // up to the 448-token text-context). If the input
    // exceeds it, we pay the retry cost — same as the
    // previous probe-and-write design.
    const INITIAL_CAPACITY: usize = 256;
    let mut buf: Vec<i32> = Vec::new();
    buf.try_reserve_exact(INITIAL_CAPACITY).ok()?;

    // SAFETY: buf has capacity ≥ INITIAL_CAPACITY; nul_terminated
    // outlives the call; ctx_ptr is non-null.
    let written = unsafe {
      sys::whispercpp_tokenize(ctx_ptr, cstr_ptr, buf.as_mut_ptr(), INITIAL_CAPACITY as i32)
    };
    if written == i32::MIN {
      return None;
    }
    if written >= 0 {
      // Happy path — input fit in INITIAL_CAPACITY.
      let written_usize = written as usize;
      if written_usize > buf.capacity() {
        return None;
      }
      // SAFETY: upstream wrote `written_usize` `i32`s.
      unsafe { buf.set_len(written_usize) };
      return Some(buf);
    }
    // Negative non-`INT_MIN` → upstream's
    // `-(needed_count)` for "buffer too small". Allocate
    // the right size and retry.
    let needed = (-written) as usize;
    let mut buf: Vec<i32> = Vec::new();
    buf.try_reserve_exact(needed).ok()?;
    // SAFETY: buf.as_mut_ptr() valid for `needed` writes;
    // nul_terminated still alive.
    let written =
      unsafe { sys::whispercpp_tokenize(ctx_ptr, cstr_ptr, buf.as_mut_ptr(), needed as i32) };
    if written == i32::MIN {
      return None;
    }
    if written < 0 {
      // Defensive: shouldn't happen at exact capacity.
      return None;
    }
    let written_usize = written as usize;
    if written_usize > buf.capacity() {
      return None;
    }
    // SAFETY: upstream wrote `written_usize` `i32`s.
    unsafe { buf.set_len(written_usize) };
    Some(buf)
  }

  /// Tokenise `text` and return the single resulting token id,
  /// or `None` if the text doesn't tokenise to exactly one token.
  ///
  /// Useful for converting short literal markers (`"<|en|>"`,
  /// `" Hello"`) into ids without round-tripping through a
  /// `Vec<i32>`.
  ///
  /// Direct single-pass call with a stack output buffer:
  /// avoids the heap allocation `tokenize` would do for the
  /// `Vec<i32>` we'd then immediately discard. The input
  /// NUL-terminated buffer is still heap-allocated
  /// (text length is unbounded), but the output buffer is
  /// 4 bytes on the stack.
  pub fn tokenize_one(&self, text: &str) -> Option<i32> {
    let bytes = text.as_bytes();
    if bytes.contains(&0) {
      return None;
    }
    let mut nul_terminated: Vec<u8> = Vec::new();
    nul_terminated.try_reserve_exact(bytes.len() + 1).ok()?;
    nul_terminated.extend_from_slice(bytes);
    nul_terminated.push(0);
    let cstr_ptr: *const core::ffi::c_char = nul_terminated.as_ptr().cast();
    let ctx_ptr = self.ptr.as_ptr();

    // Stack-only output buffer for the single token id.
    // Asking for capacity 1 means upstream returns:
    //   * `1` if `text` tokenised to exactly one token
    //     (success — `out[0]` holds the id);
    //   * `0` if `text` was empty / yielded zero tokens
    //     (fail — return `None`);
    //   * `-(needed)` if `text` would tokenise to >1 tokens
    //     (fail — `tokenize_one`'s contract);
    //   * `INT_MIN` on caught C++ exception.
    let mut out = [0i32; 1];
    // SAFETY: nul_terminated outlives the call; out has 1
    // element of capacity; ctx_ptr is non-null.
    let written = unsafe { sys::whispercpp_tokenize(ctx_ptr, cstr_ptr, out.as_mut_ptr(), 1) };
    if written == 1 { Some(out[0]) } else { None }
  }

  // ── Model introspection ────────────────────────────────────

  /// Snapshot of the loaded model's hyper-parameters. Wraps
  /// the per-field `whisper_model_n_*` accessors.
  pub fn model_dims(&self) -> ModelDims {
    let p = self.ptr.as_ptr();
    // SAFETY: ctx pointer invariant. Each accessor reads a
    // const field of the loaded model's hparams struct; no
    // allocations, no throw.
    unsafe {
      ModelDims {
        n_audio_state: sys::whisper_model_n_audio_state(p),
        n_audio_head: sys::whisper_model_n_audio_head(p),
        n_audio_layer: sys::whisper_model_n_audio_layer(p),
        n_text_state: sys::whisper_model_n_text_state(p),
        n_text_head: sys::whisper_model_n_text_head(p),
        n_text_layer: sys::whisper_model_n_text_layer(p),
        n_mels: sys::whisper_model_n_mels(p),
        model_ftype: sys::whisper_model_ftype(p),
      }
    }
  }

  // Timing helpers live on `State`, not `Context`. Upstream's
  // `whisper_print_timings(ctx)` / `whisper_reset_timings(ctx)`
  // only operate on `ctx->state`, which is always nullptr in
  // this wrapper because `Context::new` uses the `_no_state`
  // initializer. The state-aware
  // `whispercpp_*_timings_with_state` shims (in the patched
  // `whisper.cpp`) accept the actual State the wrapper hands
  // out; see `State::print_timings` /
  // `State::reset_timings`.
}

/// Loaded-model hyper-parameter snapshot returned by
/// [`Context::model_dims`].
///
/// All fields are passthroughs from the C-side
/// `whisper_model_n_*` accessors. None of them are validated
/// here beyond the bounds whisper.cpp's own loader applies
/// (see the `whispercpp-sys: hparams validation` patch in the
/// linked submodule for what's enforced at load time).
///
/// Mostly useful for diagnostics / format-detection code that
/// wants to know which checkpoint variant got loaded
/// (e.g. distinguishing `large-v3` from `large-v3-turbo` by
/// `n_text_layer`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelDims {
  /// Encoder hidden size.
  pub n_audio_state: i32,
  /// Encoder attention heads per layer.
  pub n_audio_head: i32,
  /// Encoder layers.
  pub n_audio_layer: i32,
  /// Decoder hidden size.
  pub n_text_state: i32,
  /// Decoder attention heads per layer.
  pub n_text_head: i32,
  /// Decoder layers.
  pub n_text_layer: i32,
  /// Mel-spectrogram bin count (80 for vanilla checkpoints,
  /// 128 for `large-v3+`).
  pub n_mels: i32,
  /// File-type tag baked into the GGUF header (quantisation
  /// flavour, etc.). Raw integer; consult whisper.cpp's
  /// `e_ftype` enum for the meaning.
  pub model_ftype: i32,
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

  /// Fresh `ContextParams` defaults to DTW disabled with a
  /// 128 MiB working budget. Pin the contract so a future
  /// refactor can't quietly enable DTW for callers that didn't
  /// ask for it (they'd silently pay the ~5–15% decode-time
  /// overhead).
  #[test]
  fn default_context_params_have_dtw_off_and_default_mem_budget() {
    let p = ContextParams::new();
    assert!(!p.dtw_token_timestamps());
    assert_eq!(p.dtw_aheads_preset(), AlignmentHeadsPreset::None);
    assert_eq!(p.dtw_mem_size(), DEFAULT_DTW_MEM_SIZE);
    assert_eq!(DEFAULT_DTW_MEM_SIZE, 128 * 1024 * 1024);
  }

  /// `with_dtw_*` chained setters compose end-to-end without
  /// consuming intermediate state. Mirrors the existing
  /// `with_use_gpu` / `with_gpu_device` builder shape.
  ///
  /// Uses a `dtw_mem_size` above [`MIN_DTW_MEM_SIZE`] so the
  /// clamp passes the value through unchanged. The clamp's
  /// own boundary behaviour is pinned by
  /// [`with_dtw_mem_size_clamps_zero_and_usize_max`].
  #[test]
  fn context_params_chained_dtw_setters_compose() {
    let custom_mem = MIN_DTW_MEM_SIZE * 2;
    let p = ContextParams::new()
      .with_use_gpu(false)
      .with_dtw_token_timestamps(true)
      .with_dtw_aheads_preset(AlignmentHeadsPreset::LargeV3Turbo)
      .with_dtw_mem_size(custom_mem);
    assert!(!p.use_gpu());
    assert!(p.dtw_token_timestamps());
    assert_eq!(p.dtw_aheads_preset(), AlignmentHeadsPreset::LargeV3Turbo);
    assert_eq!(p.dtw_mem_size(), custom_mem);
  }

  /// `clamp_dtw_mem_size` raises below-floor inputs to
  /// `MIN_DTW_MEM_SIZE` and lowers above-ceiling inputs to
  /// `MAX_DTW_MEM_SIZE`. Both ends close
  /// native-code abort / null-deref paths inside whisper.cpp's
  /// DTW arena allocator (see constants' docs for the
  /// failure analysis). `usize::MIN` (= 0) and `usize::MAX`
  /// are the boundary cases that motivated the clamp; pin
  /// them so a future refactor can't quietly drop a guard.
  #[test]
  fn clamp_dtw_mem_size_pins_invariants() {
    // Below floor → MIN.
    assert_eq!(clamp_dtw_mem_size(0), MIN_DTW_MEM_SIZE);
    assert_eq!(clamp_dtw_mem_size(1), MIN_DTW_MEM_SIZE);
    assert_eq!(clamp_dtw_mem_size(1024), MIN_DTW_MEM_SIZE);
    assert_eq!(clamp_dtw_mem_size(MIN_DTW_MEM_SIZE - 1), MIN_DTW_MEM_SIZE);
    // At and just above floor → passthrough.
    assert_eq!(clamp_dtw_mem_size(MIN_DTW_MEM_SIZE), MIN_DTW_MEM_SIZE);
    assert_eq!(
      clamp_dtw_mem_size(MIN_DTW_MEM_SIZE + 1),
      MIN_DTW_MEM_SIZE + 1
    );
    // Inside range → passthrough.
    assert_eq!(
      clamp_dtw_mem_size(256 * 1024 * 1024),
      256 * 1024 * 1024,
      "256 MiB sits between MIN ({MIN_DTW_MEM_SIZE}) and MAX ({MAX_DTW_MEM_SIZE})",
    );
    // At and just below ceiling → passthrough.
    assert_eq!(clamp_dtw_mem_size(MAX_DTW_MEM_SIZE), MAX_DTW_MEM_SIZE);
    assert_eq!(
      clamp_dtw_mem_size(MAX_DTW_MEM_SIZE - 1),
      MAX_DTW_MEM_SIZE - 1
    );
    // Above ceiling → MAX.
    assert_eq!(clamp_dtw_mem_size(MAX_DTW_MEM_SIZE + 1), MAX_DTW_MEM_SIZE);
    assert_eq!(clamp_dtw_mem_size(usize::MAX), MAX_DTW_MEM_SIZE);
    // Floor & ceiling order pin. Comparison is between two
    // `const`s, so clippy's `assertions_on_constants` lint
    // wants a `const { ... }` block to make the compile-time
    // evaluation explicit.
    const { assert!(MIN_DTW_MEM_SIZE <= MAX_DTW_MEM_SIZE) };
    assert_eq!(MIN_DTW_MEM_SIZE, DEFAULT_DTW_MEM_SIZE);
  }

  /// `with_dtw_mem_size` clamps caller-supplied values into
  /// the safe range. The clamp is the safe API's defense
  /// against a `dtw_mem_size = 0` / `usize::MAX` slip
  /// triggering whisper.cpp's `ggml_init` NULL-return /
  /// arena-overflow abort path. Pin both directions.
  #[test]
  fn with_dtw_mem_size_clamps_zero_and_usize_max() {
    let p = ContextParams::new().with_dtw_mem_size(0);
    assert_eq!(
      p.dtw_mem_size(),
      MIN_DTW_MEM_SIZE,
      "0 → MIN (defends against ggml_init NULL on zero arena)",
    );
    let p = ContextParams::new().with_dtw_mem_size(usize::MAX);
    assert_eq!(
      p.dtw_mem_size(),
      MAX_DTW_MEM_SIZE,
      "usize::MAX → MAX (defends against ggml_init internal arena math overflow)",
    );
    // In-range value passes through.
    let p = ContextParams::new().with_dtw_mem_size(MIN_DTW_MEM_SIZE * 2);
    assert_eq!(p.dtw_mem_size(), MIN_DTW_MEM_SIZE * 2);
  }

  /// `Context::new` refuses to publish a context configured
  /// with both `flash_attn` and DTW token-timestamps.
  /// Whisper.cpp silently disables DTW under flash-attention
  /// (`whisper.cpp:3956`); without an explicit Rust-side
  /// rejection, callers would observe `Ok(Context)` for a
  /// configuration that promises `Token::t_dtw` to be
  /// populated and then receive only zeros.
  ///
  /// The check fires before any FFI file-load attempt, so
  /// the test path doesn't need to exist on disk — the
  /// validation is decided from `ContextParams` alone.
  #[test]
  #[cfg_attr(miri, ignore = "FFI: whisper_context_default_params")]
  fn context_new_rejects_dtw_plus_flash_attn() {
    let params = ContextParams::new()
      .with_flash_attn(true)
      .with_dtw_token_timestamps(true)
      .with_dtw_aheads_preset(AlignmentHeadsPreset::LargeV3Turbo);
    let result = Context::new("/nonexistent/dtw+flash-attn-test.bin", params);
    match result {
      Err(WhisperError::ContextLoad { reason, .. }) => {
        assert!(
          reason.contains("DTW") && reason.contains("flash_attn"),
          "ContextLoad reason must explain DTW + flash_attn incompatibility — got: {}",
          reason,
        );
      }
      Err(e) => panic!(
        "expected ContextLoad with DTW + flash_attn rejection, got: {:?}",
        e,
      ),
      Ok(_) => panic!("expected error for DTW + flash_attn config, got Ok(Context)"),
    }
  }

  /// Mirror of the rejection test for the inverse setter
  /// order — `with_dtw_token_timestamps` before
  /// `with_flash_attn`. The chained-builder shape means
  /// either order can land in `ContextParams`; both must
  /// reach the rejection.
  #[test]
  #[cfg_attr(miri, ignore = "FFI: whisper_context_default_params")]
  fn context_new_rejects_dtw_plus_flash_attn_setter_order_invariant() {
    let params = ContextParams::new()
      .with_dtw_token_timestamps(true)
      .with_dtw_aheads_preset(AlignmentHeadsPreset::LargeV3Turbo)
      .with_flash_attn(true);
    let result = Context::new("/nonexistent/dtw+flash-attn-test2.bin", params);
    let err = result
      .err()
      .expect("expected ContextLoad error, got Ok(Context)");
    assert!(
      matches!(err, WhisperError::ContextLoad { .. }),
      "expected ContextLoad regardless of setter order, got: {:?}",
      err,
    );
  }

  /// `flash_attn` + `dtw_token_timestamps(true)` but
  /// `preset = None` is NOT rejected — the effective DTW
  /// state is "off" (per the preset coercion in
  /// `Context::new`), so flash_attn is fine. Pin this so a
  /// future tightening of the rejection doesn't accidentally
  /// reject a valid configuration.
  #[test]
  #[cfg_attr(miri, ignore = "FFI: whisper_context_default_params")]
  fn context_new_accepts_flash_attn_with_dtw_timestamps_but_no_preset() {
    let params = ContextParams::new()
      .with_flash_attn(true)
      .with_dtw_token_timestamps(true);
    // Preset stays at the default `None` — DTW is effectively off.
    // Context::new should NOT trip the flash_attn + DTW rejection,
    // but the file-load FFI will still fail (path doesn't exist).
    // Either ContextLoad with a generic load message OR
    // InvalidCString is acceptable — what we care about here is
    // that the DTW + flash_attn rejection did NOT fire.
    let result = Context::new("/nonexistent/no-dtw-fine.bin", params);
    if let Err(WhisperError::ContextLoad { reason, .. }) = &result {
      assert!(
        !(reason.contains("DTW") && reason.contains("flash_attn")),
        "DTW + flash_attn rejection fired for a config where DTW is off: {}",
        reason,
      );
    }
  }

  /// Pin the [`SUPPORTED_DTW_N_TEXT_CTX`] constant.
  ///
  /// The value `448` is the `n_text_ctx` for every standard
  /// whisper checkpoint (`tiny.en` through
  /// `large-v3-turbo`). [`required_dtw_mem_size_for`] uses it
  /// as the worst-case `n_tokens` axis when sizing the DTW
  /// scratch arena, and [`Context::new`] uses it as the
  /// model-load gate that refuses non-standard checkpoints
  /// when DTW is enabled. Drift here invalidates both the
  /// budget calc and the load-time validation — pin so a
  /// future refactor has to be deliberate.
  #[test]
  fn supported_dtw_n_text_ctx_pins_to_standard_whisper_value() {
    assert_eq!(
      SUPPORTED_DTW_N_TEXT_CTX, 448,
      "Standard whisper checkpoints all use n_text_ctx = 448. \
       If you changed this, also re-derive the DTW scratch budget \
       and update the byte-count pins in \
       `required_dtw_mem_size_pins_per_preset_minimums`.",
    );
  }

  /// Per-preset alignment-head counts must match the
  /// `g_aheads_*` tables in
  /// `whisper.cpp/src/whisper.cpp:399-410`. A drift here
  /// (e.g. an upstream rebuild renumbers a preset's heads or
  /// our match arm typo'd a count) makes
  /// [`required_dtw_mem_size_for`] return an under-sized
  /// budget, which lets `ggml_new_tensor_3d` inside the DTW
  /// path abort the process from safe Rust. Pin every
  /// shipping preset's head count.
  #[test]
  fn alignment_head_count_matches_whisper_cpp_tables() {
    use AlignmentHeadsPreset::*;
    // Counts taken from g_aheads at whisper.cpp:412-424 of
    // the patched submodule.
    assert_eq!(alignment_head_count(None), 0);
    assert_eq!(alignment_head_count(TinyEn), 8);
    assert_eq!(alignment_head_count(Tiny), 6);
    assert_eq!(alignment_head_count(BaseEn), 5);
    assert_eq!(alignment_head_count(Base), 8);
    assert_eq!(alignment_head_count(SmallEn), 19);
    assert_eq!(alignment_head_count(Small), 10);
    assert_eq!(alignment_head_count(MediumEn), 18);
    assert_eq!(alignment_head_count(Medium), 6);
    assert_eq!(alignment_head_count(LargeV1), 9);
    assert_eq!(alignment_head_count(LargeV2), 23);
    assert_eq!(alignment_head_count(LargeV3), 10);
    assert_eq!(alignment_head_count(LargeV3Turbo), 6);
  }

  /// `required_dtw_mem_size_for` must keep every shipping
  /// preset above its realistic worst-case scratch peak.
  /// The original 128 MiB floor was too small for the
  /// 18–23-head presets (`SmallEn`, `MediumEn`, `LargeV2`),
  /// whose DTW working tensor + `ggml_norm` output +
  /// median-filter output add up to 145–186 MiB just for the
  /// three live tensors. Pin the per-preset minimums so a
  /// future refactor can't quietly shrink them back below
  /// the abort threshold.
  #[test]
  fn required_dtw_mem_size_pins_per_preset_minimums() {
    use AlignmentHeadsPreset::*;
    // None → 0 (DTW disabled, no scratch needed).
    assert_eq!(required_dtw_mem_size_for(None), 0);
    // Small-head presets (≤10 heads) collapse to MIN floor.
    for preset in [
      TinyEn,
      Tiny,
      BaseEn,
      Base,
      Small,
      Medium,
      LargeV1,
      LargeV3,
      LargeV3Turbo,
    ] {
      let req = required_dtw_mem_size_for(preset);
      assert!(
        req >= MIN_DTW_MEM_SIZE,
        "{:?} requires {} bytes; must be ≥ MIN_DTW_MEM_SIZE ({})",
        preset,
        req,
        MIN_DTW_MEM_SIZE,
      );
      assert!(
        req <= MAX_DTW_MEM_SIZE,
        "{:?} requires {} bytes; must be ≤ MAX_DTW_MEM_SIZE ({})",
        preset,
        req,
        MAX_DTW_MEM_SIZE,
      );
    }
    // High-head presets (the 128 MiB regression class) MUST
    // exceed the floor — this is the regression sentinel for
    // the original analysis bug.
    for preset in [SmallEn, MediumEn, LargeV2] {
      let req = required_dtw_mem_size_for(preset);
      assert!(
        req > MIN_DTW_MEM_SIZE,
        "{:?} requires only {} bytes — must exceed MIN_DTW_MEM_SIZE ({}) \
         to fit its high-head DTW pipeline; without this the wrapper's \
         floor would let whisper.cpp abort during decode",
        preset,
        req,
        MIN_DTW_MEM_SIZE,
      );
    }
    // Spot-check the explicit byte counts so a future change
    // to the formula has to update this pin too. Math:
    //   per_tensor = 448 * 1500 * n_heads * 4
    //   required   = (per_tensor * 3) * 3 / 2
    // For LargeV2 (n_heads=23):
    //   per_tensor = 448 * 1500 * 23 * 4 = 61_824_000 bytes
    //   required   = 61_824_000 * 3 * 3 / 2 = 278_208_000 bytes
    assert_eq!(required_dtw_mem_size_for(LargeV2), 278_208_000);
    // SmallEn (n_heads=19): 448 * 1500 * 19 * 4 = 51_072_000 bytes
    //   required = 51_072_000 * 3 * 3 / 2 = 229_824_000 bytes
    assert_eq!(required_dtw_mem_size_for(SmallEn), 229_824_000);
    // MediumEn (n_heads=18): 448 * 1500 * 18 * 4 = 48_384_000 bytes
    //   required = 48_384_000 * 3 * 3 / 2 = 217_728_000 bytes
    assert_eq!(required_dtw_mem_size_for(MediumEn), 217_728_000);
  }

  /// Every `AlignmentHeadsPreset` variant maps to a distinct
  /// `whisper_alignment_heads_preset` raw value. If a future
  /// upstream renumbering collapsed two presets to the same
  /// value (or our match arm typo'd one), the timestamps
  /// produced for the affected models would silently become
  /// noise. Pin the bijection here as a regression sentinel.
  #[test]
  fn alignment_heads_preset_maps_to_distinct_raw_values() {
    use AlignmentHeadsPreset::*;
    let presets = [
      None,
      TinyEn,
      Tiny,
      BaseEn,
      Base,
      SmallEn,
      Small,
      MediumEn,
      Medium,
      LargeV1,
      LargeV2,
      LargeV3,
      LargeV3Turbo,
    ];
    let raws: Vec<sys::whisper_alignment_heads_preset> =
      presets.iter().map(|p| p.to_raw()).collect();
    // No duplicates: every preset maps somewhere different.
    let mut sorted = raws.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(
      sorted.len(),
      presets.len(),
      "AlignmentHeadsPreset → raw mapping must be injective: got {:?}",
      raws,
    );
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
