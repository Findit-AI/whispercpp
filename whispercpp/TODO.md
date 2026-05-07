# whispercpp — unsupported surface

This crate intentionally exposes a narrow slice of whisper.cpp.
Everything below is reachable from the auto-generated FFI in the
sibling `whispercpp-sys` crate (`whispercpp-sys/src/generated.rs`)
but is NOT wrapped in safe Rust today.

Three categories:

1. **Deliberately omitted** — whispery doesn't need it; wrapping would
   add maintenance + surface area without a caller.
2. **Could add on demand** — small wrapper, not justified yet.
3. **Larger work** — would need design choices about safety / lifetimes
   before exposing.

When adding something below, also extend the `allowlist_function`
/ `allowlist_type` directives in
`whispercpp-sys/build.rs::generate_bindings()` if the symbol
isn't already in `whispercpp-sys/src/generated.rs`.

---

## 1. Deliberately omitted

### Built-in VAD

whisper.cpp ships its own VAD (Silero ONNX). Whispery uses the
`silero` crate for VAD upstream of `whispercpp`, so the in-tree path
is the canonical one. Re-wrapping whisper.cpp's wrapper duplicates
state and complicates the call chain.

Symbols: `whisper_vad_*`, `whisper_full_params::vad`,
`vad_model_path`, `vad_params`, `set_min_speech_duration_ms`,
`set_max_speech_duration_s`, `set_min_silence_duration_ms`,
`set_speech_pad_ms`, `set_threshold`.

### Grammar

Whispery doesn't constrain decoding via grammar. The grammar
machinery in whisper.cpp pulls a sizeable struct hierarchy
(`whisper_grammar_element`, rules, stacks) and a non-trivial
ownership model. No caller has asked for it.

Symbols: `whisper_full_params::grammar_rules`, `grammar_n_rules`,
`grammar_i_start_rule`, `grammar_penalty`, `set_grammar`,
`set_grammar_penalty`, `set_start_rule`, `whisper_grammar_*`.

### Translate task

Whispery is transcribe-only. `set_translate(true)` (translate audio
→ English) is wrapped (one-line passthrough), but the full
translate-task flow (token id remapping, prompt seeding) is not
exercised by any caller and we don't ship test coverage for it.

### Tinydiarize controls

`Segment::speaker_turn_next()` IS wrapped (it's a 1-byte read).
Configuring `--tdrz` on the input side (`set_tdrz_enable`) is not
— it requires a TDRZ-enabled checkpoint which whispery doesn't ship,
and whispery's diarization runs upstream via pyannote-style
clustering on word ranges.

Symbols: `whisper_full_params::tdrz_enable`, `set_tdrz_enable`.

### Lower-level entry points

We expose `state.full()` only. The lower-level encode/decode flow
(running the encoder, then `decode` token-by-token with custom
sampling) is meaningful for research / custom samplers but doesn't
fit whispery's pump architecture.

Symbols: `whisper_encode`, `whisper_encode_with_state`,
`whisper_decode`, `whisper_decode_with_state`, `whisper_get_logits`,
`whisper_get_logits_from_state`, `whisper_set_mel`,
`whisper_set_mel_with_state`, `whisper_pcm_to_mel`,
`whisper_pcm_to_mel_with_state`.

### Mid-decode callbacks

Whisper.cpp can fire callbacks on every new segment, every logits
emission, and at encoder start. Each requires the same trampoline
discipline as the abort callback and adds another `Box<dyn FnMut>`
field to `Params`. None is wired into whispery's pump (which works
chunk-at-a-time, not token-at-a-time).

Symbols: `set_progress_callback`, `set_progress_callback_safe`,
`set_progress_callback_user_data`, `set_new_segment_callback`,
`set_segment_callback_safe`, `set_segment_callback_safe_lossy`,
`set_new_segment_callback_user_data`, `set_filter_logits_callback`,
`set_filter_logits_callback_user_data`, `set_start_encoder_callback`,
`set_start_encoder_callback_user_data`.

### Global logging hooks

Whispery routes diagnostics through its own `eprintln!` / `tracing`
layer. whisper.cpp's `set_log_callback` is a global hook that fires
across all instances; mixing it with Rust logging frameworks
requires more design than a 1:1 port.

Symbols: `whisper_set_log_callback`, `set_debug_mode`,
`whisper_log_callback`.

### DTW token timestamps

Whispery uses wav2vec2 forced alignment for word-level timing.
whisper.cpp's DTW path is a parallel mechanism with its own
configuration (`dtw_aheads`, `dtw_n_top`, `dtw_mem_size`). Wrapping
it would invite confusion about which timestamping path is
authoritative.

Symbols: `whisper_full_params::dtw_token_timestamps` (true at
construction, but `Params::set_dtw_*` and `dtw_aheads` array are
not exposed), `whisper_aheads`, `whisper_full_get_token_dtw_t0_*`.

### Buffer-load constructors

We support `Context::new(path, params)` only. Loading from an
in-memory buffer (`whisper_init_from_buffer_with_params`) or via a
custom `whisper_model_loader` is rare and adds lifetime/ownership
complexity.

Symbols: `whisper_init_from_buffer_with_params`,
`whisper_init_with_params` (custom loader), the `whisper_model_loader`
struct.

### Beam-search + greedy sampler details (advanced)

Symbols: `set_beam_size`, `set_patience` are reachable through
`SamplingStrategy::BeamSearch { beam_size, patience }` already.
Direct `whisper_full_params::beam_search.beam_size` / `patience`
accessors aren't exposed (use `Params::new(strategy)` with the
right variant).

---

## 2. Could add on demand

These are 5–15-line wrappers around an existing FFI symbol. None is
required for whispery's current flow; each is justifiable when a
concrete caller appears.

| Whisper.cpp symbol | Suggested Rust API | Why might we want it |
|---|---|---|
| `whisper_token_text(ctx, token)` (alias of `token_to_str`) | already covered | — |
| `whisper_token_to_bytes` | `Context::token_to_bytes(token) -> Option<&[u8]>` | non-UTF-8 byte sequences from BPE merges |
| `whisper_lang_id(name)` | `Context::lang_id_for(name: &str) -> Option<i32>` | reverse of `detected_lang` |
| `whisper_lang_max_id()` | `pub const LANG_MAX_ID: i32 = …` | iterate languages |
| `whisper_lang_str_full(id)` | `Lang::full_name() -> &'static str` | "english" vs "en" |
| `whisper_token_translate / transcribe / prev / nosp / not / solm` | `Context::token_translate() -> i32`, etc. | force-prefix decoding seeds |
| `whisper_token_lang(ctx, lang_id)` | `Context::token_for_lang(Lang) -> i32` | language-specific seeds |
| `whisper_token_id(ctx, token: &str)` | `Context::tokenize_one(text) -> Option<i32>` | turn a string back into a token id |
| `whisper_tokenize(ctx, text, tokens, max)` | `Context::tokenize(text) -> Vec<i32>` | batch tokenization for `set_tokens` |
| `whisper_n_len_from_state(state)` | `State::n_mel_frames() -> i32` | mel buffer length |
| `whisper_print_timings(ctx)` | `Context::print_timings()` | end-of-run cost breakdown |
| `whisper_reset_timings(ctx)` | `Context::reset_timings()` | per-chunk timing |
| `whisper_get_whisper_version()` | `pub fn version() -> &'static str` | diagnostic |
| Model layer counts (`whisper_model_n_audio_state` / `n_audio_head` / `n_audio_layer` / `n_text_state` / `n_text_head` / `n_text_layer` / `n_mels` / `model_ftype`) | `Context::model_dims() -> ModelDims` (struct of ints) | architecture-aware diagnostics |
| `whisper_full_get_token_p_from_state` | `Token::posterior() -> f32` | already covered indirectly via `Token::p()` reading `whisper_token_data.p` — verify the two agree under wildcard / temperature |
| `whisper_full_n_tokens_from_state` | already covered (`Segment::n_tokens()`) | — |

Adding any of these means: extend the safe wrapper module, run the
existing test suite (`cargo test -p whispercpp --features serde`),
and confirm no rebuild loop on `src/generated.rs` (build.rs short-
circuits when the bindgen output is byte-identical).

---

## 3. Larger work

### Token-stream `Iterator`

`State::segments_iter()` and `Segment::tokens_iter()` would be nice
ergonomics. The lifetime story is non-trivial — each `Segment` /
`Token` borrows from `State` via raw pointer. A correct iterator
needs to project through that lifetime without aliasing.

### Async-friendly `full`

`State::full` blocks for the duration of the decode (seconds to
minutes). A `tokio`-friendly variant that runs the FFI on a
blocking task pool and yields completion would help server use
cases. Currently callers spawn their own threads.

### Streaming / partial-result API

whisper.cpp's `whisper_full` is a one-shot call. Streaming
transcription requires either (a) the new-segment callback path
(see "Mid-decode callbacks" above), or (b) external chunking + one
`Context::create_state()` per chunk. Whispery does (b) at the
runner layer.

### CoreML companion model build

Whispery ships `coreml` as an opt-in feature, but generating the
`.mlmodelc` companion file (whisper.cpp's `models/generate-coreml-
model.sh`) is out-of-band. A `whispercpp-tools` crate or a build.rs
helper that converts a checkpoint at install time would close the
loop, but it requires `coremltools` (a Python dep) at build time —
not great.

---

## Audit policy

Before adding new public functions to `whispercpp`:

1. Confirm the FFI symbol is in
   `whispercpp-sys/src/generated.rs`. If not, extend the
   allowlists in `whispercpp-sys/build.rs::generate_bindings()`.
2. Replicate the safety rules used by the closest existing wrapper:
   pointer is non-null, lifetime tied to the parent struct, no
   aliasing across threads. Document the SAFETY block.
3. Keep the public surface minimal — accessors private until a
   caller materialises. The crate's value is "small, audited, no
   leaks"; that holds only if every `unsafe` block has an obvious
   justification.

For deliberately-omitted items, prefer documenting the omission here
rather than wrapping speculatively.
