# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-05-09

Additive release. Cargo-compatible with `whispercpp = "0.2"` consumers; no
changes to `whispercpp-sys` (the `-sys` crate stays at `0.2.0`).

### Added

- `Segments<'a>` and `Tokens<'state>` iterator types in `whispercpp::state`,
  plus `State::segments_iter()` and `Segment::tokens_iter()` constructors.
  Both implement `Iterator + ExactSizeIterator + DoubleEndedIterator +
  FusedIterator`.
- `IntoIterator` impls so the standard collection idioms work:
  - `for seg in &state { ... }` — `IntoIterator for &State`.
  - `for tok in seg { ... }` — `IntoIterator for Segment<'a>` (by-value;
    `Segment` is `Copy`, so the consumption is cheap).
  - `for tok in &seg { ... }` — `IntoIterator for &Segment<'a>`.
- `Tokens<'state>` owns a `Copy` of the `Segment` so adapter chains like
  `state.segments_iter().flat_map(|s| s.tokens_iter())` compile (the inner
  iterator does not borrow the closure-local `Segment` value).

### Performance

- `Segments::next` and `Tokens::next` inline the construction / pointer
  projection rather than calling back through `State::segment(i)` /
  `Segment::token(j)`, saving one `n_segments()` / `n_tokens()` FFI call
  per yielded item. The iterator's captured `end` plus the `&self` borrow
  chain (`State::full` requires `&mut self`) make the per-call bounds-check
  redundant. Dominant on `Tokens` — typical states have hundreds to low
  thousands of tokens per `State::full` invocation.

### Internal

- New `#[cfg(test)] pub(crate)` test fixtures in `context.rs` /
  `state.rs` (`Context::dangling_for_test` `unsafe fn`,
  `State::poisoned_for_test`) so iterator behaviour can be exercised
  without a real model file. The `unsafe fn` makes the
  "caller must `mem::forget`" precondition explicit at every call site.
- `safety_audit.rs` matrix gains a `segments_iter` / `tokens_iter` row
  walking all ten safety axes; inlined-FFI projection rationale documented
  under axis #1 (throw).

### Fixed

No bug fixes — this is a feature-only release.

[#9]: https://github.com/Findit-AI/whispercpp/pull/9

## [0.2.0] - 2026-05-09

Two feature streams (DTW timestamps in [#7], the issue-#2 accessors in [#8])
plus a comprehensive safety pass that hardened the FFI seam against panic,
leak, double-free, OOM-abort, dangling-alias, and integer-boundary classes
reachable from safe Rust through `Context::new` / `Context::create_state` /
`State::full`.

### Added

#### DTW token timestamps (#7)

- `ContextParams::with_dtw_token_timestamps(bool)` — enable the DTW pass at
  context construction.
- `ContextParams::with_dtw_aheads_preset(AlignmentHeadsPreset)` — pick the
  alignment-head set; one variant per shipping whisper checkpoint
  (`TinyEn` through `LargeV3Turbo`). `None` disables DTW even when the flag
  is on.
- `ContextParams::with_dtw_mem_size(usize)` — override the DTW scratch
  arena. Clamped to `[MIN_DTW_MEM_SIZE, MAX_DTW_MEM_SIZE]` and raised to
  the per-preset minimum from `required_dtw_mem_size_for(preset)`.
- `Token::t_dtw() -> Option<i64>` — DTW timestamp; `None` covers both
  "DTW not enabled at Context construction" and "DTW skipped for this
  segment".
- `required_dtw_mem_size_for`, `MIN_DTW_MEM_SIZE`, `MAX_DTW_MEM_SIZE`,
  `SUPPORTED_DTW_N_TEXT_CTX` constants.
- `Context::new` validates DTW × flash-attention combinations up-front
  (whisper.cpp silently disables DTW under flash-attn) and rejects
  models whose `n_text_ctx` exceeds the DTW budget.

#### Issue-#2 accessor surface (#8)

- `Context::token_to_bytes(token)` — borrow the raw token text without
  UTF-8 conversion.
- `Context::tokenize(text) -> Option<Vec<i32>>` /
  `Context::tokenize_one(text) -> Option<i32>` — single-pass tokenisation
  via a no-log shim.
- `Context::model_dims() -> ModelDims` — vocab size, n_audio_*, n_text_*
  in one call.
- `Context::token_for_lang(&Lang) -> Option<i32>` — model-bound language
  token id (validated against the loaded model's actual language-token
  range, not just `g_lang`).
- `Context::token_translate / token_transcribe / token_prev / token_nosp
  / token_not / token_solm` — special-token id accessors.
- `Lang::full_name() -> Option<SmolStr>` — the English language name
  (owned `SmolStr`, not `&'static str`).
- `State::print_timings(&self)` / `State::reset_timings(&mut self)` /
  `State::n_mel_frames(&self)` — state-bound timing API. Replaces the
  upstream `whisper_print_timings(ctx)` which doesn't see this wrapper's
  separately-allocated states.
- Top-level `version() -> Option<&'static str>`, `lang_max_id() -> i32`,
  `lang_id_for(name) -> Option<i32>`.

#### Error variants

- `WhisperError::UnknownLanguage(SmolStr)` — `set_language` rejects codes
  not in whisper.cpp's `g_lang` table.
- `WhisperError::LanguageNotSupportedByModel(SmolStr)` — the loaded model
  doesn't carry the language token (e.g. non-English on `.en`
  checkpoints, or auto-detect on a non-multilingual model).
- `WhisperError::InvalidDuration { offset_ms, duration_ms,
  audio_duration_ms }` — `set_duration_ms` / `set_offset_ms` describe a
  range past the actual audio length. Avoids upstream's unbounded
  `seek_end` driving long zero-padded decode loops.
- `WhisperError::AllocationFailed { context: &'static str }` — Rust-side
  copy buffer reservation failed (`try_reserve_exact`).

#### Safety audit

- `whispercpp/src/safety_audit.rs` — comments-only module documenting
  the per-method × per-axis safety matrix, the 14+ axes the wrapper
  checks against (throw, sync, allocation, lifetime, linkage, sentinel
  collisions, log pollution, error-payload bounds, race / classification,
  model-bound semantics, metadata consistency, DoS amplification,
  default-state mirror, state vs ctx scope, integer boundary,
  abort-bypass of shim, invariant preservation), and the native-side
  patches the audit relies on.

### Changed

- `Params::set_initial_prompt` — fallible on allocation pressure (uses
  `Vec::try_reserve_exact`); caps input at 1 MiB up-front; pre-scans for
  interior NUL.
- `Params::set_language` — validates against the language table before
  any allocation; 32-byte length cap; trims overlong rejection diagnostics
  to a 64-char head so the error variant payload stays bounded.
- `Lang::full_name` returns `Option<SmolStr>` (was `Option<&'static
  str>`) — the underlying `whisper_lang_str_full` returns a pointer
  into a `std::map<std::string, ...>` member, not a string literal.
- `Params::language()` (internal) falls back to `raw.language` when the
  Rust mirror is unset, so fresh `Params::new(...)` reports `Some("en")`
  matching upstream's default.
- `State::print_timings` no longer prints a "total time" line (the
  context-shared `t_start_us` doesn't pair coherently with state-bound
  per-stage counters that `reset_timings` can zero).

### Fixed

- C++ exceptions on every constructor and inference shim are caught and
  translated to typed errors instead of crossing `extern "C"` (UB).
- Recoverable OOM throughout the create-state path no longer aborts:
  `ggml_backend_sched_new`, `ggml_gallocr_new_n`,
  `ggml_gallocr_reserve_n_impl`, the dynamic tensor allocator, and the
  hash-set construction all return `NULL` / `false` cleanly under
  allocation pressure. Failure paths preserve the gallocr's invariants
  (atomic-commit growth for `node_allocs` / `leaf_allocs` / `vbuffer` /
  `hash_set` + `hash_values`) so a follow-up call can still observe a
  coherent allocator.
- Native leak windows in `whisper_model_load` /
  `whisper_init_state` / `whisper_backend_init` are closed by RAII
  guards (`ggml_context_ptr`, `ggml_backend_buffer_ptr`,
  `ggml_backend_ptr`); a caught throw frees every allocation made on
  the failing path before the rethrow.
- `whisper_tokenize`, `whisper_token_count`, and the shim variants
  guard `res.size() > INT_MAX` so the existing
  `prompt_tokens.resize(-n_needed)` and `-whisper_tokenize(...)`
  expressions can't reach `-INT_MIN` (signed-overflow UB).
- Vocab-count consistency check at load time
  (`n_vocab != hparams.n_vocab` plus the post-synthesis
  `id_to_token.size() == vocab.n_vocab` assertion) prevents
  `WHISPER_ASSERT` aborts during decode on malformed model files.
- Auto-detect candidate set is filtered to ids the model actually
  carries, so out-of-range language ids can't alias onto
  `token_translate` / `token_transcribe`.
- `Context::create_state` no longer poisons the Context on a recoverable
  state-init exception (caught throws now flow through `StateInit`,
  not `ConstructorLost`).
- `set_duration_ms` / `set_offset_ms` ranges are validated at
  `State::full` entry against the actual sample buffer (avoids the 71k
  zero-padded decode windows an `i32::MAX` slip would otherwise drive).

### Breaking

- `Params::set_tokens` returns `WhisperResult<&mut Self>` (was
  `&mut Self`) — propagates the `try_reserve_exact` failure path.
- `WhisperError::StateInit` is now a struct variant carrying
  `code: Option<i32>` (was a unit variant). Patterns like
  `WhisperError::StateInit` need to become `WhisperError::StateInit { .. }`.
- `WhisperError::ContextLoad` gains a `code: Option<i32>` field.
  Patterns destructuring with explicit fields need to add `code` or use
  `..`.
- `Lang::full_name` returns `Option<SmolStr>` instead of
  `Option<&'static str>`.

### Internal

- 37 patches against the bundled `whisper.cpp` (`Findit-AI/whisper.cpp`,
  `rust` branch) are pinned via a `REQUIRED_MARKERS` table in
  `whispercpp-sys/build.rs`. A submodule rebase that drops any sentinel
  fails patch-verification at build time.
- Process-wide `init_lock` mutex serialises `Context::new` and
  `Context::create_state` so concurrent callers don't race on
  `ggml_log_set` and the GPU-init globals.
- Per-`Context` `full_lock` mutex caps in-flight
  `whisper_full_with_state` calls to one per Context, bounding the
  per-call leak budget on the `StateLost` path.

[#7]: https://github.com/Findit-AI/whispercpp/pull/7
[#8]: https://github.com/Findit-AI/whispercpp/pull/8

## [0.1.0] - Initial release

Initial public release of `whispercpp` (safe Rust wrapper) and
`whispercpp-sys` (raw FFI). Bundles a vendored, patch-verified
`whisper.cpp` and exposes the model load / state allocation / inference
surface needed for production transcription.
