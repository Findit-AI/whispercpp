//! Safety audit of the public API surface.
//!
//! This module is **comments-only** — no runtime code. Its
//! purpose is to capture the per-method, per-axis safety
//! audit for the wrapper's public API so future review
//! passes start from "what's been checked" instead of
//! re-deriving the surface from scratch.
//!
//! Adversarial review surfaced a pattern: each new pass
//! catches genuine bugs along an axis the previous pass
//! didn't probe. Walking the full axis matrix once, before
//! shipping, converges the review cycle.
//!
//! When adding a new public method:
//! 1. Walk every axis in [Audit axes](#audit-axes).
//! 2. Update the [Per-method coverage](#per-method-coverage)
//!    matrix below.
//! 3. If an axis fails, fix it before merging — don't
//!    rely on the next reviewer to catch it.
//!
//! # Audit axes
//!
//! For every public method that crosses the FFI boundary,
//! check each of these 10 axes:
//!
//! 1. **Throw safety.** Does the FFI call have any path
//!    that throws a C++ exception? If yes, it must go
//!    through a `whispercpp_*` shim (try/catch wrapper).
//!    If "no", the answer must be derived from upstream
//!    source, not assumed. `std::map<std::string, ...>`
//!    lookups via `const char *` allocate a temporary
//!    `std::string` — that's a throw point.
//!    `std::vector::push_back` is a throw point.
//!    `new T[]` is a throw point.
//!
//! 2. **Sync safety.** For methods on a `Sync` type
//!    (`Context`), does this call mutate native state? If
//!    yes, multiple threads holding `Arc<T>` could race
//!    via concurrent `&self` calls. Either lock or remove
//!    the `&self` access. Pure reads of fields written
//!    only at construction are safe; pure reads of fields
//!    that another method (or upstream) writes during
//!    inference are NOT safe without serialisation.
//!
//! 3. **Allocation safety.** Every Rust-side allocation
//!    on a path that takes attacker-controlled input must
//!    be fallible (`try_reserve_exact`, `Vec::new()` +
//!    fallible reserve, etc.). `Vec::with_capacity`,
//!    `Vec::from(slice)`, and `CString::new` (which
//!    internally allocates a Vec) all use the infallible
//!    global allocator that aborts on OOM.
//!
//! 4. **Lifetime safety.** Returned references must point
//!    at storage that genuinely outlives the lifetime
//!    advertised. `&'static str` requires a literal or
//!    truly-immortal data — `std::map<std::string, ...>`
//!    members are NOT `'static` because the `std::string`
//!    contents are heap-owned and destroyed during static-
//!    object cleanup. Copy into an owned type
//!    (`SmolStr`/`String`) on the way out.
//!
//! 5. **Linkage.** Functions defined in `whisper.cpp`
//!    (the patched submodule) and called from Rust through
//!    the bindgen header (which puts them in
//!    `extern "C"`) MUST be wrapped in `extern "C" { ... }`
//!    in their definition site. Otherwise the C++ compiler
//!    emits Itanium-mangled symbols that the Rust extern
//!    `"C"` references cannot resolve at link time.
//!    Test binaries that don't actually call the function
//!    can still link via dead-symbol elimination — the
//!    failure surfaces only in downstream binaries that
//!    do, so verify with `nm` or a forced
//!    function-pointer-load test.
//!
//! 6. **Sentinel collisions.** Functions returning `int`
//!    where the wrapper layers shim sentinels (e.g.
//!    `WHISPERCPP_ERR_BAD_ALLOC = -100`) onto the same
//!    domain must keep sentinels disjoint from valid
//!    return values. `whisper_tokenize` returns
//!    `-(needed_count)` on too-small-buffer, which
//!    overlaps `-100..=-103` for inputs of 100..=103
//!    tokens — collision. Use distinct domains
//!    (`INT_MIN` for "exception caught" + thread-local
//!    sentinel for the class) or out-params.
//!
//! 7. **Log pollution.** Probe-style FFI calls
//!    (`fn(.., NULL, 0)` to get a required size) often
//!    trip an upstream "buffer too small" log on every
//!    call — making real failures invisible in production
//!    logs. Add a no-log shim that calls the internal
//!    helper directly.
//!
//! 8. **Error-payload bounds.** Error variants that store
//!    caller-controlled strings (`SmolStr(input)`) must
//!    bound the payload before construction. `SmolStr`
//!    inlines ≤ 23 bytes and heap-allocates beyond that —
//!    a 100 KiB attacker input becomes a 100 KiB error
//!    payload, propagated via `Display` to log tails.
//!    Trim to a fixed prefix (typical: 64 chars =
//!    ≤ 256 bytes UTF-8).
//!
//! 9. **Race conditions / TOCTOU.** Init/destroy windows
//!    (e.g. `create_state` + `mark_lost` interleaving)
//!    must preserve the leak-classification taxonomy.
//!    Constructors that catch an exception leave native
//!    state allocated; collapsing that case to a "clean"
//!    poison error misleads callers about whether
//!    retries compound the leak.
//!
//! 10. **Model-bound semantics.** FFI inputs validated
//!     against process-global tables (e.g.
//!     `whisper_lang_id`'s `g_lang`) may still be
//!     unsupported by the loaded model (different vocab
//!     size, fewer language tokens, English-only checkpoint).
//!     For language ids specifically, the resolved token
//!     must lie in `(token_sot, token_translate)` — the
//!     model's actual language-token range. Validate at
//!     `State::full` entry where both Context and Params
//!     are present.
//!
//! 11. **Metadata-consistency.** A loaded model carries
//!     two views of the same hyperparameter: the file
//!     header value (`hparams.n_vocab`) and a
//!     post-load count derived from what was actually
//!     read in (`vocab.n_vocab` after the
//!     `n_vocab = read_safe<int32_t>(loader)` line). When
//!     a downstream FFI call uses one and the runtime
//!     uses the other, mismatch becomes a native abort or
//!     OOB index. Re-enable upstream's
//!     `n_vocab != hparams.n_vocab` consistency check at
//!     load time so corrupt or hand-edited files fail
//!     fast. Same shape applies to other dual-view fields:
//!     anything where Rust validates against `n_vocab()`
//!     but the C++ runtime indexes via `id_to_token.size()`
//!     is a candidate.
//!
//! 12. **DoS amplification.** Wrapper-accepted parameter
//!     values that upstream does not bound to physical
//!     resources can amplify a small misuse into a long
//!     hang. `whisper_full_with_state` computes
//!     `seek_end = seek_start + duration_ms / 10` without
//!     clamping to `whisper_n_len_from_state` and runs the
//!     decoder over the whole range with reads past EOF
//!     zero-filled by `whisper_encode_internal`. An
//!     `i32::MAX` slip drives ~71 000 decode windows
//!     instead of erroring. Validate physical bounds
//!     (`offset_ms + duration_ms <= audio_duration_ms`) at
//!     the wrapper boundary where the sample buffer is
//!     visible.
//!
//! # Per-method coverage
//!
//! Audit pass on `feat/accessors` (issue #2 wrapper additions
//! plus the modifications they triggered). Each row records
//! one public method × every axis. ✓ = checked & passes;
//! N/A = axis doesn't apply for this method's shape
//! (e.g. axis 2 doesn't apply to top-level functions).
//!
//! ## Top-level functions (`crate::`)
//!
//! ### `version()`
//! 1. throw: `whisper_version` returns `WHISPER_VERSION`
//!    string literal — no throw. ✓
//! 2. sync: top-level fn. N/A
//! 3. alloc: no Rust alloc. ✓
//! 4. lifetime: returns `&'static str` from C string
//!    literal in the static lib. Genuinely `'static`. ✓
//! 5. linkage: upstream public C API. ✓
//! 6. sentinels: returns Option, NULL = build issue. ✓
//! 7. log pollution: no log. ✓
//! 8. error bounds: returns Option, no error variant. ✓
//! 9. race: no shared state. ✓
//! 10. model-bound: not applicable (no model). N/A
//!
//! ### `lang_max_id()`
//! 1. throw: iterates const `g_lang` — no throw. ✓
//! 2. sync: top-level fn. N/A
//! 3. alloc: none. ✓
//! 4. lifetime: returns i32. ✓
//! 5. linkage: upstream public C API. ✓
//! 6. sentinels: count, never negative. ✓
//! 7. log pollution: no log. ✓
//! 8. error bounds: no error. ✓
//! 9. race: const table. ✓
//! 10. model-bound: returns global table size. ✓
//!
//! ### `lang_id_for(name)`
//! 1. throw: routes through `whispercpp_lang_id` shim
//!    (try/catch around `g_lang.count`/`.at` which
//!    construct a temporary `std::string` from
//!    `const char *` — a throw point on OOM). ✓
//! 2. sync: top-level fn. N/A
//! 3. alloc: `CString::new(name)` is infallible, but
//!    bounded by the 32-byte length cap → max ~33-byte
//!    alloc per call. Practically safe. ✓
//! 4. lifetime: returns `Option<i32>`. ✓
//! 5. linkage: shim is `extern "C"`. ✓
//! 6. sentinels: shim returns -1 (not found) or
//!    `WHISPERCPP_ERR_*` at -100..=-103 (caught). Disjoint
//!    from valid non-negative ids. ✓
//! 7. log pollution: 32-byte cap stops adversarial
//!    inputs from reaching the upstream
//!    `WHISPER_LOG_ERROR("unknown language '%s'")` path.
//!    The `whispercpp-sys: log_internal va_copy` patch
//!    closes the va-list-reuse + heap-throw UB inside
//!    the logger structurally — defense in depth. ✓
//! 8. error bounds: no error variant. ✓
//! 9. race: shim is reentrant on a const table. ✓
//! 10. model-bound: documented as global lookup; callers
//!     who need model-bound checks use
//!     `Context::token_for_lang`. ✓
//!
//! ## `Lang` methods
//!
//! ### `Lang::full_name()`
//! 1. throw: `whisper_lang_str_full` iterates const map,
//!    returns `c_str()` — no throw. ✓
//! 2. sync: instance method on owned `Lang`. N/A
//! 3. alloc: `SmolStr::new(bytes)` — names ≤ ~13 bytes,
//!    all inline. Bounded. ✓
//! 4. lifetime: returns `Option<SmolStr>` (owned, copied
//!    on read). NOT `&'static str` — the underlying
//!    `c_str()` is from a `std::map<std::string, ...>`
//!    member, not a literal. ✓
//! 5. linkage: upstream public C API. ✓
//! 6. sentinels: Option. ✓
//! 7. log pollution: bounded by `lang_id_for`'s 32-byte
//!    cap. ✓
//! 8. error bounds: no error. ✓
//! 9. race: const map. ✓
//! 10. model-bound: returns the global English name; not
//!     model-specific. ✓
//!
//! ## `Context` methods (`&self` on `Sync`)
//!
//! ### `token_to_bytes(token)`
//! 1. throw: patched `whisper_token_to_str` uses `.find()`
//!    returning NULL on miss. No throw. ✓
//! 2. sync: pure read of `vocab.id_to_token` (built at
//!    load, immutable thereafter). ✓
//! 3. alloc: none. ✓
//! 4. lifetime: returns `Option<&[u8]>` tied to `&self`.
//!    The bytes live in a `std::string` member of
//!    `Context::vocab.id_to_token` — destroyed only by
//!    `whisper_free` in `Drop`. Lifetime correctly
//!    bounded. ✓
//! 5. linkage: upstream public C API. ✓
//! 6. sentinels: Option. ✓
//! 7. log pollution: no log. ✓
//! 8. error bounds: no error. ✓
//! 9. race: const map after load. ✓
//! 10. model-bound: token range-checked vs `n_vocab()`. ✓
//!
//! ### `token_translate / token_transcribe / token_prev /
//! ###  token_nosp / token_not / token_solm()`
//! 1. throw: pure C reads of vocab fields. ✓
//! 2. sync: const reads of fields set at load. ✓
//! 3. alloc: none. ✓
//! 4. lifetime: i32. ✓
//! 5. linkage: upstream public C API. ✓
//! 6. sentinels: token id, no special values. ✓
//! 7. log pollution: no log. ✓
//! 8. error bounds: no error. ✓
//! 9. race: const reads. ✓
//! 10. model-bound: returns the model's actual token id. ✓
//!
//! ### `token_for_lang(&Lang)`
//! 1. throw: `lang_id_for` (shim) +
//!    `whisper_token_lang` (pure addition) +
//!    `whisper_token_translate` (pure read). No throw. ✓
//! 2. sync: const reads. ✓
//! 3. alloc: bounded via `lang_id_for`'s cap. ✓
//! 4. lifetime: `Option<i32>`. ✓
//! 5. linkage: all upstream public C API. ✓
//! 6. sentinels: Option. ✓
//! 7. log pollution: bounded via `lang_id_for`'s cap. ✓
//! 8. error bounds: no error. ✓
//! 9. race: const reads. ✓
//! 10. model-bound: validates `is_multilingual()` AND
//!     resolved token in `(token_sot, token_translate)`.
//!     ✓
//!
//! ### `tokenize(text)`
//! 1. throw: probe via `whispercpp_token_count` shim
//!    (no-log shim around internal `tokenize(vocab, text)`,
//!    try/catch); write via `whispercpp_tokenize` shim
//!    (try/catch). Both catch `bad_alloc` /
//!    `system_error`. ✓
//! 2. sync: internal `tokenize(vocab, text)` reads
//!    const vocab; concurrent reads safe. ✓
//! 3. alloc: NUL-terminated input via
//!    `Vec::try_reserve_exact(len + 1)`; output via
//!    `Vec::try_reserve_exact(needed)`. Both fallible. ✓
//! 4. lifetime: returns owned `Vec<i32>`. ✓
//! 5. linkage: shims are `extern "C"`. ✓
//! 6. sentinels: probe shim returns count or `INT_MIN`
//!    (exception); write shim returns count, negative
//!    "still too small", or `INT_MIN`. `INT_MIN` is
//!    unreachable from any realistic count. ✓
//! 7. log pollution: probe uses no-log
//!    `whispercpp_token_count`; write call sized exactly
//!    so the too-small log branch is unreachable. ✓
//! 8. error bounds: returns Option. ✓
//! 9. race: const vocab. ✓
//! 10. model-bound: token ids are in the model's vocab
//!     range. Caller can pass to `Params::set_tokens`,
//!     where State::full validates against `n_vocab`. ✓
//!
//! ### `tokenize_one(text)`
//! Delegates to `tokenize`; same audit. ✓
//!
//! ### `model_dims()`
//! 1. throw: 8 pure C reads. ✓
//! 2. sync: const reads of model.hparams. ✓
//! 3. alloc: none. ✓
//! 4. lifetime: returns owned `ModelDims`. ✓
//! 5. linkage: all upstream public C API. ✓
//! 6. sentinels: struct of i32s. ✓
//! 7. log pollution: no log. ✓
//! 8. error bounds: no error. ✓
//! 9. race: const reads. ✓
//! 10. model-bound: returns the loaded model's actual
//!     dims. ✓
//!
//! ### `create_state()`
//! 1. throw: `whispercpp_init_state` shim (try/catch). ✓
//! 2. sync: holds `init_lock` (process-wide
//!    serialisation of GGML logger init). Atomic
//!    `lost.load`. ✓
//! 3. alloc: ggml-side allocation; goes through OOM-safe
//!    `ggml_init` patch. ✓
//! 4. lifetime: returns owned `State`. ✓
//! 5. linkage: shim is `extern "C"`. ✓
//! 6. sentinels: NULL + thread-local exception sentinel
//!    via `whispercpp_take_last_constructor_exception`. ✓
//! 7. log pollution: not applicable. ✓
//! 8. error bounds: error variants
//!    (`ContextPoisoned` / `ConstructorLost` /
//!    `StateInit`) carry no caller strings. ✓
//! 9. race: TOCTOU between entry-time `lost.load` and
//!    post-FFI re-check preserves `ConstructorLost` over
//!    `ContextPoisoned` when the sentinel is non-zero. ✓
//! 10. model-bound: state per-Context. ✓
//!
//! ## `State` methods (`!Sync`)
//!
//! ### `n_mel_frames()`
//! 1. throw: pure C read. ✓
//! 2. sync: State is `!Sync`, no concurrent access. ✓
//! 3. alloc: none. ✓
//! 4. lifetime: i32. ✓
//! 5. linkage: upstream public C API. ✓
//! 6. sentinels: 0 for poisoned/uninit. ✓
//! 7. log pollution: no log. ✓
//! 8. error bounds: no error. ✓
//! 9. race: !Sync. ✓
//! 10. model-bound: reads state's mel field. Default-
//!     init'd to 0 by the `whisper_mel POD field
//!     default-init` patch. ✓
//!
//! ### `print_timings(&self)` / `reset_timings(&mut self)`
//! 1. throw: shims write integers / call vsnprintf to
//!    stack buffer. No throw (the long-message heap
//!    allocation was eliminated by the
//!    `log_internal exception-safe truncation` patch). ✓
//! 2. sync: State is `!Sync`. `&self` / `&mut self`
//!    enforce exclusive Rust-side access. ✓
//! 3. alloc: none. ✓
//! 4. lifetime: void return. ✓
//! 5. linkage: shims defined in `whisper.cpp` inside
//!    `extern "C" { ... }` block. ✓
//! 6. sentinels: void. ✓
//! 7. log pollution: prints intentionally — that's the
//!    purpose. Truncation patch caps message size. ✓
//! 8. error bounds: no error. ✓
//! 9. race: handled by `!Sync` + `&mut`. State::full
//!    holds `&mut self` so concurrent timing access
//!    is impossible. The shim only touches
//!    state-bound fields, NOT `ctx->t_*_us` writers
//!    (`ctx->t_load_us` and `ctx->t_start_us` are
//!    written only at `whisper_init_*` time, never
//!    during inference). ✓
//! 10. model-bound: state-bound (correct). ✓
//!
//! ### `segments_iter(&self)` / `Segment::tokens_iter(&self)`
//! 1. throw: `Segments::next` and `Tokens::next` inline
//!    the pointer-projection (private-field access on
//!    `State.ptr` / `Segment.state`) instead of calling
//!    back through `State::segment(i)` / `Segment::token(j)`,
//!    which would re-call `n_segments()` / `n_tokens()`
//!    (FFI) for their bounds check. The captured `end`
//!    field plus the `&self` borrow chain make the
//!    bounds-check redundant: `State::full` requires
//!    `&mut self`, so the count cannot change while any
//!    iterator borrow is alive. The inlined unsafe FFI
//!    is `whisper_full_get_token_data_from_state` —
//!    pure C accessor, no allocation, no throw. ✓
//! 2. sync: `State` is `!Sync`. `&self` on the iterator
//!    permits multiple iterators alive simultaneously,
//!    which is sound because the underlying buffers are
//!    immutable for the borrow's duration (`State::full`
//!    requires `&mut self`, ruled out by the borrow
//!    checker while any `&self` iterator exists). ✓
//! 3. alloc: iterator state is two i32s + a borrow; no
//!    Rust-side alloc. ✓
//! 4. lifetime: `Segments<'a>` ties yielded `Segment<'a>`
//!    to the source `&'a State`. `Tokens<'state>` owns a
//!    copied `Segment<'state>` (which is `Copy`), so
//!    adapter composition like
//!    `state.segments_iter().flat_map(|s|
//!    s.tokens_iter())` typechecks (the iterator does
//!    not borrow a closure-local `Segment`). The
//!    `'state` lifetime still ties yielded item pointer
//!    projections to the parent `State`. Yielded `Token`
//!    is value-typed (owned snapshot) so has no further
//!    lifetime constraint. ✓
//! 5. linkage: no new FFI symbols. ✓
//! 6. sentinels: `next` and `end` index counters;
//!    bounded at construction by `n_segments()` /
//!    `n_tokens()`. `next_back` (DoubleEndedIterator)
//!    decrements `end`; the `next < end` guard at the
//!    top of every direction's call rejects the
//!    converged-cursor case. ✓
//! 7. log pollution: no log path. ✓
//! 8. error bounds: iterator yields `Option<Item>`; no
//!    error variant. ✓
//! 9. race: `!Sync` rules out concurrent `&self` from
//!    two threads. ✓
//! 10. model-bound: yields whatever `State::full`
//!     produced; bound to the loaded model implicitly via
//!     the parent state. ✓
//!
//! `IntoIterator` impls (`for &State`, `for Segment`,
//! `for &Segment`) delegate to the existing
//! `segments_iter` / `tokens_iter` constructors, so they
//! inherit the same axis coverage — no separate row.
//!
//! ### `full(&mut self, ...)`
//! Pre-existing FFI surface; the audit here is on the
//! preflight additions:
//! 9. race: holds `Context::full_lock` mutex. ✓
//! 10. model-bound: language preflight checks
//!     `is_multilingual` AND resolved
//!     `whisper_token_lang(ctx, lang_id)` is in
//!     `(token_sot, token_translate)`. Non-multilingual
//!     checkpoints (`.en`) accept `lang_id == 0`
//!     (English) and skip the check when
//!     `params.detect_language` is true. ✓
//! 11. metadata-consistency: enforced at LOAD time by
//!     the `vocab count consistency check` patch
//!     (`n_vocab != hparams.n_vocab` rejects the file).
//!     Once the model is in memory, `prompt_tokens`
//!     range-check uses `n_vocab()` and the runtime
//!     uses the same count, so they cannot disagree. ✓
//! 12. DoS amplification: duration-range preflight.
//!     `offset_ms`, `duration_ms`, `samples.len()` are
//!     read from `Params` and the argument; rejected
//!     with `InvalidDuration` when
//!     the requested range exceeds the audio. The
//!     `duration_ms == 0` "to end of input" sentinel is
//!     accepted. ✓
//!
//!
//! ## `Params` methods
//!
//! ### `set_duration_ms(&mut self, ms)` (pre-existing)
//! 1. throw: const setter, no FFI. ✓
//! 2. sync: `&mut self`. ✓
//! 3. alloc: none. ✓
//! 4. lifetime: `&mut Self`. ✓
//! 5. linkage: no FFI. ✓
//! 6. sentinels: `0` is upstream's "to end of input"
//!    marker — pass-through. ✓
//! 7. log pollution: no log. ✓
//! 8. error bounds: no error path; out-of-range values
//!    pass through and are caught at `State::full`. ✓
//! 9. race: `&mut self`. ✓
//! 10. model-bound: not applicable (audio param). N/A
//! 11. metadata-consistency: not applicable. N/A
//! 12. DoS amplification: physical-bounds check happens
//!     at `State::full` entry where the sample buffer
//!     is visible. ✓
//!
//! ### `set_offset_ms(&mut self, ms)` (pre-existing)
//! 1–10: as `set_duration_ms`. ✓
//! 11. metadata-consistency: N/A
//! 12. DoS amplification: clamps negatives to `0` at the
//!     setter (negative offset is its own UB axis —
//!     OOB mel read in upstream). The
//!     `offset_ms + duration_ms <= audio_len_ms` bound
//!     is enforced at `State::full`. ✓
//!
//! ### `set_language(&mut self, lang)` (modified)
//! 1. throw: routes lang lookup through `lang_id_for`
//!    (shim'd). ✓
//! 2. sync: `&mut self`. ✓
//! 3. alloc: `CString::new(lang)` after the 32-byte cap
//!    → bounded. ✓
//! 4. lifetime: stored as `CString` owned by `Params`. ✓
//! 5. linkage: routes via shim. ✓
//! 6. sentinels: returns
//!    `Result<&mut Self, WhisperError>`. ✓
//! 7. log pollution: 32-byte cap stops upstream log
//!    paths reaching with attacker-controlled strings. ✓
//! 8. error bounds: 64-char-head trim on the
//!    length-rejection path; `UnknownLanguage` payload
//!    is bounded by the (already-passed) 32-byte cap. ✓
//! 9. race: `&mut self`. ✓
//! 10. model-bound: `set_language` validates against
//!     global table (catches typos); the model-bound
//!     check fires later in `State::full` because
//!     Context isn't available at setter time. ✓
//!
//! # Native-side patches the audit relies on
//!
//! The wrapper assumes these submodule patches are present.
//! Build hard-fails (`whispercpp-sys/build.rs::REQUIRED_MARKERS`)
//! if any sentinel marker is missing:
//!
//! - `kv_cache_free idempotent fix`
//! - `read_safe zero-init`
//! - `init_state RAII entry`
//! - `init_context RAII entry`
//! - `tensor header validation (model_load)`
//! - `ggml_log_set once-per-process`
//! - `hparams validation`
//! - `lang_str null guard`
//! - `special-token bounds check`
//! - `path_model assignment guard`
//! - `sched abort callback wiring`
//! - `vad_init RAII guard`
//! - `dtw scratch RAII guard`
//! - `dtw scratch alloc-fail throws`
//! - `dtw token assignment bounded`
//! - `dtw short-window medfilt clamp`
//! - `dtw audio_ctx override guard`
//! - `ggml_init throw-on-null wrapper`
//! - `dtw decode failure throws`
//! - `kv buffer null throws`
//! - `dtw backtrace impossible-case throws`
//! - `dtw aheads_cross_QKs invariants throw`
//! - `token_to_str sparse-vocab no-throw`
//! - `hparams head divisibility check`
//! - `dtw backend compute throws`
//! - `dtw t_dtw sentinel init`
//! - `whisper_mel POD field default-init`
//! - `state-aware timing entry points`
//! - `log_internal va_copy` (now: `log_internal
//!   exception-safe truncation`)
//! - `no-log token count shim`
//! - `no-log tokenize shim`
//! - `vocab count consistency check`
//! - `vocab post-synthesis size check`
//! - `model_load RAII for raw ggml allocations`
//! - `model_load tensor-prep RAII`
//! - `model_load buffer-registration RAII`
//! - `vad_load RAII for raw ggml allocations`
//! - `vad_load tensor-prep RAII`
//! - `vad_load buffer-registration RAII`
//! - `tokenize size_t→int overflow guard`
//! - `whisper_tokenize size_t→int overflow guard`
//! - `whisper_tokenize INT_MIN propagation`
//! - `whisper_token_count INT_MIN propagation`
//! - `gallocr_new_n OOM-safe alloc` (in `ggml/src/ggml-alloc.c`)
//! - `gallocr_reserve_n_impl OOM-safe paths` (in `ggml/src/ggml-alloc.c`)
//! - `dyn_tallocr_new OOM-safe alloc` (in `ggml/src/ggml-alloc.c`)
//! - `dyn_tallocr_alloc OOM-safe sentinel` (in `ggml/src/ggml-alloc.c`)
//! - `gallocr alloc-failure flag` (in `ggml/src/ggml-alloc.c`)
//! - `gallocr_free_node invalid-chunk guard` (in `ggml/src/ggml-alloc.c`)
//! - `hash_set / hash_values atomic commit` (in `ggml/src/ggml-alloc.c`)
//! - `node_allocs growth transactional` (in `ggml/src/ggml-alloc.c`)
//! - `leaf_allocs growth transactional` (in `ggml/src/ggml-alloc.c`)
//! - `vbuffer realloc transactional` (in `ggml/src/ggml-alloc.c`)
//! - `sched_alloc_splits reserve_n return check` (in `ggml/src/ggml-backend.cpp`)
//! - `backend_init RAII`
//! - `sched_graph_init NULL guard`
//! - `backend_sched_new OOM-safe alloc` (in `ggml/src/ggml-backend.cpp`)
//! - `hash_set_new OOM-safe alloc` (in `ggml/src/ggml-backend.cpp`)
//! - `state-aware print drops total time`
//! - `auto-detect bounded to model lang range`
//! - `ggml_init OOM-safe context alloc` (in `ggml.c`)
//!
