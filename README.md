<div align="center">
<h1>whispercpp</h1>
</div>
<div align="center">

Safe Rust bindings for [whisper.cpp][whisper-cpp] speech-to-text inference.

[<img alt="github" src="https://img.shields.io/badge/github-findit--ai/whispercpp-8da0cb?style=for-the-badge&logo=Github" height="22">][Github-url]
<img alt="LoC" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fal8n%2F327b2a8aef9003246e45c6e47fe63937%2Fraw%2Fwhispercpp" height="22">
[<img alt="Build" src="https://img.shields.io/github/actions/workflow/status/findit-ai/whispercpp/ci.yml?logo=Github-Actions&style=for-the-badge" height="22">][CI-url]
[<img alt="codecov" src="https://img.shields.io/codecov/c/gh/findit-ai/whispercpp?style=for-the-badge&token=6R3QFWRWHL&logo=codecov" height="22">][codecov-url]

[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-whispercpp-66c2a5?style=for-the-badge&labelColor=555555&logo=data:image/svg+xml;base64,PHN2ZyByb2xlPSJpbWciIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDUxMiA1MTIiPjxwYXRoIGZpbGw9IiNmNWY1ZjUiIGQ9Ik00ODguNiAyNTAuMkwzOTIgMjE0VjEwNS41YzAtMTUtOS4zLTI4LjQtMjMuNC0zMy43bC0xMDAtMzcuNWMtOC4xLTMuMS0xNy4xLTMuMS0yNS4zIDBsLTEwMCAzNy41Yy0xNC4xIDUuMy0yMy40IDE4LjctMjMuNCAzMy43VjIxNGwtOTYuNiAzNi4yQzkuMyAyNTUuNSAwIDI2OC45IDAgMjgzLjlWMzk0YzAgMTMuNiA3LjcgMjYuMSAxOS45IDMyLjJsMTAwIDUwYzEwLjEgNS4xIDIyLjEgNS4xIDMyLjIgMGwxMDMuOS01MiAxMDMuOSA1MmMxMC4xIDUuMSAyMi4xIDUuMSAzMi4yIDBsMTAwLTUwYzEyLjItNi4xIDE5LjktMTguNiAxOS45LTMyLjJWMjgzLjljMC0xNS05LjMtMjguNC0yMy40LTMzLjd6TTM1OCAyMTQuOGwtODUgMzEuOXYtNjguMmw4NS0zN3Y3My4zek0xNTQgMTA0LjFsMTAyLTM4LjIgMTAyIDM4LjJ2LjZsLTEwMiA0MS40LTEwMi00MS40di0uNnptODQgMjkxLjFsLTg1IDQyLjV2LTc5LjFsODUtMzguOHY3NS40em0wLTExMmwtMTAyIDQxLjQtMTAyLTQxLjR2LS42bDEwMi0zOC4yIDEwMiAzOC4ydi42em0yNDAgMTEybC04NSA0Mi41di03OS4xbDg1LTM4Ljh2NzUuNHptMC0xMTJsLTEwMiA0MS40LTEwMi00MS40di0uNmwxMDItMzguMiAxMDIgMzguMnYuNnoiPjwvcGF0aD48L3N2Zz4K" height="20">][doc-url]
[<img alt="crates.io" src="https://img.shields.io/crates/v/whispercpp?style=for-the-badge&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iaXNvLTg4NTktMSI/Pg0KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDE5LjAuMCwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPg0KPHN2ZyB2ZXJzaW9uPSIxLjEiIGlkPSJMYXllcl8xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB4PSIwcHgiIHk9IjBweCINCgkgdmlld0JveD0iMCAwIDUxMiA1MTIiIHhtbDpzcGFjZT0icHJlc2VydmUiPg0KPGc+DQoJPGc+DQoJCTxwYXRoIGQ9Ik0yNTYsMEwzMS41MjgsMTEyLjIzNnYyODcuNTI4TDI1Niw1MTJsMjI0LjQ3Mi0xMTIuMjM2VjExMi4yMzZMMjU2LDB6IE0yMzQuMjc3LDQ1Mi41NjRMNzQuOTc0LDM3Mi45MTNWMTYwLjgxDQoJCQlsMTU5LjMwMyw3OS42NTFWNDUyLjU2NHogTTEwMS44MjYsMTI1LjY2MkwyNTYsNDguNTc2bDE1NC4xNzQsNzcuMDg3TDI1NiwyMDIuNzQ5TDEwMS44MjYsMTI1LjY2MnogTTQzNy4wMjYsMzcyLjkxMw0KCQkJbC0xNTkuMzAzLDc5LjY1MVYyNDAuNDYxbDE1OS4zMDMtNzkuNjUxVjM3Mi45MTN6IiBmaWxsPSIjRkZGIi8+DQoJPC9nPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPC9zdmc+DQo=" height="22">][crates-url]
[<img alt="crates.io" src="https://img.shields.io/crates/d/whispercpp?color=critical&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI/PjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+PHN2ZyB0PSIxNjQ1MTE3MzMyOTU5IiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjM0MjEiIGRhdGEtc3BtLWFuY2hvci1pZD0iYTMxM3guNzc4MTA2OS4wLmkzIiB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIj48ZGVmcz48c3R5bGUgdHlwZT0idGV4dC9jc3MiPjwvc3R5bGU+PC9kZWZzPjxwYXRoIGQ9Ik00NjkuMzEyIDU3MC4yNHYtMjU2aDg1LjM3NnYyNTZoMTI4TDUxMiA3NTYuMjg4IDM0MS4zMTIgNTcwLjI0aDEyOHpNMTAyNCA2NDAuMTI4QzEwMjQgNzgyLjkxMiA5MTkuODcyIDg5NiA3ODcuNjQ4IDg5NmgtNTEyQzEyMy45MDQgODk2IDAgNzYxLjYgMCA1OTcuNTA0IDAgNDUxLjk2OCA5NC42NTYgMzMxLjUyIDIyNi40MzIgMzAyLjk3NiAyODQuMTYgMTk1LjQ1NiAzOTEuODA4IDEyOCA1MTIgMTI4YzE1Mi4zMiAwIDI4Mi4xMTIgMTA4LjQxNiAzMjMuMzkyIDI2MS4xMkM5NDEuODg4IDQxMy40NCAxMDI0IDUxOS4wNCAxMDI0IDY0MC4xOTJ6IG0tMjU5LjItMjA1LjMxMmMtMjQuNDQ4LTEyOS4wMjQtMTI4Ljg5Ni0yMjIuNzItMjUyLjgtMjIyLjcyLTk3LjI4IDAtMTgzLjA0IDU3LjM0NC0yMjQuNjQgMTQ3LjQ1NmwtOS4yOCAyMC4yMjQtMjAuOTI4IDIuOTQ0Yy0xMDMuMzYgMTQuNC0xNzguMzY4IDEwNC4zMi0xNzguMzY4IDIxNC43MiAwIDExNy45NTIgODguODMyIDIxNC40IDE5Ni45MjggMjE0LjRoNTEyYzg4LjMyIDAgMTU3LjUwNC03NS4xMzYgMTU3LjUwNC0xNzEuNzEyIDAtODguMDY0LTY1LjkyLTE2NC45MjgtMTQ0Ljk2LTE3MS43NzZsLTI5LjUwNC0yLjU2LTUuODg4LTMwLjk3NnoiIGZpbGw9IiNmZmZmZmYiIHAtaWQ9IjM0MjIiIGRhdGEtc3BtLWFuY2hvci1pZD0iYTMxM3guNzc4MTA2OS4wLmkwIiBjbGFzcz0iIj48L3BhdGg+PC9zdmc+&style=for-the-badge" height="22">][crates-url]
<img alt="license" src="https://img.shields.io/badge/License-Apache%202.0/MIT-blue.svg?style=for-the-badge&fontColor=white&logoColor=f5c076&logo=data:image/svg+xml;base64,PCFET0NUWVBFIHN2ZyBQVUJMSUMgIi0vL1czQy8vRFREIFNWRyAxLjEvL0VOIiAiaHR0cDovL3d3dy53My5vcmcvR3JhcGhpY3MvU1ZHLzEuMS9EVEQvc3ZnMTEuZHRkIj4KDTwhLS0gVXBsb2FkZWQgdG86IFNWRyBSZXBvLCB3d3cuc3ZncmVwby5jb20sIFRyYW5zZm9ybWVkIGJ5OiBTVkcgUmVwbyBNaXhlciBUb29scyAtLT4KPHN2ZyBmaWxsPSIjZmZmZmZmIiBoZWlnaHQ9IjgwMHB4IiB3aWR0aD0iODAwcHgiIHZlcnNpb249IjEuMSIgaWQ9IkNhcGFfMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgdmlld0JveD0iMCAwIDI3Ni43MTUgMjc2LjcxNSIgeG1sOnNwYWNlPSJwcmVzZXJ2ZSIgc3Ryb2tlPSIjZmZmZmZmIj4KDTxnIGlkPSJTVkdSZXBvX2JnQ2FycmllciIgc3Ryb2tlLXdpZHRoPSIwIi8+Cg08ZyBpZD0iU1ZHUmVwb190cmFjZXJDYXJyaWVyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KDTxnIGlkPSJTVkdSZXBvX2ljb25DYXJyaWVyIj4gPGc+IDxwYXRoIGQ9Ik0xMzguMzU3LDBDNjIuMDY2LDAsMCw2Mi4wNjYsMCwxMzguMzU3czYyLjA2NiwxMzguMzU3LDEzOC4zNTcsMTM4LjM1N3MxMzguMzU3LTYyLjA2NiwxMzguMzU3LTEzOC4zNTcgUzIxNC42NDgsMCwxMzguMzU3LDB6IE0xMzguMzU3LDI1OC43MTVDNzEuOTkyLDI1OC43MTUsMTgsMjA0LjcyMywxOCwxMzguMzU3UzcxLjk5MiwxOCwxMzguMzU3LDE4IHMxMjAuMzU3LDUzLjk5MiwxMjAuMzU3LDEyMC4zNTdTMjA0LjcyMywyNTguNzE1LDEzOC4zNTcsMjU4LjcxNXoiLz4gPHBhdGggZD0iTTE5NC43OTgsMTYwLjkwM2MtNC4xODgtMi42NzctOS43NTMtMS40NTQtMTIuNDMyLDIuNzMyYy04LjY5NCwxMy41OTMtMjMuNTAzLDIxLjcwOC0zOS42MTQsMjEuNzA4IGMtMjUuOTA4LDAtNDYuOTg1LTIxLjA3OC00Ni45ODUtNDYuOTg2czIxLjA3Ny00Ni45ODYsNDYuOTg1LTQ2Ljk4NmMxNS42MzMsMCwzMC4yLDcuNzQ3LDM4Ljk2OCwyMC43MjMgYzIuNzgyLDQuMTE3LDguMzc1LDUuMjAxLDEyLjQ5NiwyLjQxOGM0LjExOC0yLjc4Miw1LjIwMS04LjM3NywyLjQxOC0xMi40OTZjLTEyLjExOC0xNy45MzctMzIuMjYyLTI4LjY0NS01My44ODItMjguNjQ1IGMtMzUuODMzLDAtNjQuOTg1LDI5LjE1Mi02NC45ODUsNjQuOTg2czI5LjE1Miw2NC45ODYsNjQuOTg1LDY0Ljk4NmMyMi4yODEsMCw0Mi43NTktMTEuMjE4LDU0Ljc3OC0zMC4wMDkgQzIwMC4yMDgsMTY5LjE0NywxOTguOTg1LDE2My41ODIsMTk0Ljc5OCwxNjAuOTAzeiIvPiA8L2c+IDwvZz4KDTwvc3ZnPg==" height="22">

</div>

## Introduction

Safe Rust bindings for [whisper.cpp][whisper-cpp] speech-to-text inference.

- **Always-bundled build.** `whispercpp-sys` cmake-builds a vendored,
  patched whisper.cpp; there is no pkg-config / system-install path.
  The patched source lives on a fork branch with each fix as a
  reviewable commit (see [Memory safety](#memory-safety) below).
- **Panic-free safe surface.** Every FFI call is wrapped in a C++
  exception-catching shim, every fallible setter returns
  `WhisperError`, every accessor short-circuits on poisoned state.
- **`Send + Sync`** `Context`; per-`Context` `State` is `Send`.
  Concurrent inference is serialized through a per-`Context` mutex
  so per-call leak budgets are structural, not documentary.
- **Backend matrix.** Metal, CoreML, Vulkan, OpenCL, CUDA, ROCm
  (HIP), oneAPI (SYCL), Moore Threads (MUSA), OpenVINO, OpenBLAS —
  all opt-in via Cargo features.
- **DTW token timestamps.** Built-in token-level timing via DTW
  over the configured alignment heads (`AlignmentHeadsPreset`),
  with safe per-token availability through
  `Token::t_dtw() -> Option<i64>`. See
  [DTW timestamps](#dtw-timestamps).

## Installation

```toml
[dependencies]
whispercpp = "0.2"
```

The default build is plain CPU. Opt into accelerators per-target:

```toml
# macOS Apple Silicon
[target.'cfg(all(target_os = "macos", target_arch = "aarch64"))'.dependencies]
whispercpp = { version = "0.2", features = ["metal", "coreml"] }

# Linux + NVIDIA
[target.'cfg(all(target_os = "linux", target_arch = "x86_64"))'.dependencies]
whispercpp = { version = "0.2", features = ["cuda"] }
```

## Examples

A working end-to-end example lives at
[`whispercpp/examples/smoke.rs`](whispercpp/examples/smoke.rs).

## Backends

All backend features chain to the matching `whispercpp-sys` feature
which toggles the corresponding ggml / whisper CMake flag.

| Feature    | Backend                              | Platforms                |
|------------|--------------------------------------|--------------------------|
| `metal`    | Metal GPU                            | Apple                    |
| `coreml`   | CoreML / ANE encoder                 | Apple (with `.mlmodelc`) |
| `vulkan`   | Vulkan compute                       | Linux / Windows / Android / MoltenVK on macOS |
| `opencl`   | OpenCL (mobile / Adreno)             | Linux / Android          |
| `cuda`     | NVIDIA CUDA                          | Linux / Windows          |
| `hipblas`  | AMD ROCm / HIP                       | Linux                    |
| `sycl`     | Intel oneAPI / Arc                   | Linux / Windows          |
| `musa`     | Moore Threads MUSA                   | Linux                    |
| `openvino` | Intel OpenVINO encoder               | Linux / Windows          |
| `openblas` | OpenBLAS CPU                         | Any                      |
| `serde`    | `Serialize` / `Deserialize` for `Lang` (lowercase ISO-639-1) | — |

GPU backends require the corresponding vendor SDK (CUDA Toolkit,
ROCm, oneAPI, etc.) installed at link time. CI exercises the
bundled CPU path on Linux/macOS/Windows and Metal+CoreML on macOS.

## DTW timestamps

Token-level timestamps via DTW over the decoder's
cross-attention weights. Enable at `Context` construction:

```rust
use whispercpp::{Context, ContextParams, AlignmentHeadsPreset};

let ctx = Context::new(
    "ggml-large-v3-turbo.bin",
    ContextParams::new()
        .with_use_gpu(true)
        .with_dtw_token_timestamps(true)
        .with_dtw_aheads_preset(AlignmentHeadsPreset::LargeV3Turbo),
)?;
```

Match `AlignmentHeadsPreset` to your model — the safe API
ships every standard checkpoint preset (`TinyEn` through
`LargeV3Turbo`). Mismatched presets produce noisy timings
without erroring; bound-checked by `required_dtw_mem_size_for`
and rejected at load if the model's `n_text_ctx` exceeds
`SUPPORTED_DTW_N_TEXT_CTX`.

After `state.full(&params, &samples)`, read per-token DTW
timing as `Option<i64>` (centiseconds):

```rust
for i in 0..state.n_segments() {
    let seg = state.segment(i).unwrap();
    for j in 0..seg.n_tokens() {
        let token = seg.token(j).unwrap();
        match token.t_dtw() {
            Some(t) => println!("token={} t_dtw={:.2}s",
                token.id(), t as f64 / 100.0),
            None    => /* DTW unavailable for this token */ (),
        }
    }
}
```

`None` covers four cases: DTW not enabled at construction,
non-text token (special / timestamp), per-segment DTW skip
because `Params::set_audio_ctx` was overridden too small, or
audio window too short for the median-filter pass. The
underlying C-side patch (`whispercpp-sys: dtw t_dtw sentinel
init`) initialises `t_dtw = -1` before every DTW pass so the
sentinel uniquely identifies "unavailable" — `Some(0)` is a
valid timestamp (token at audio offset 0), not the sentinel.

Constraints (enforced at `Context::new`):

| Constraint | What it does |
|---|---|
| `dtw + flash_attn` | Rejected. Whisper.cpp silently disables DTW under flash-attn; the wrapper refuses the combination explicitly. |
| `dtw + custom n_text_ctx > 448` | Rejected. The DTW scratch arena is sized for standard whisper checkpoints; non-standard models with larger text context would overflow it. |
| `dtw_mem_size` | Clamped to `[MIN_DTW_MEM_SIZE, MAX_DTW_MEM_SIZE]`, then raised to the per-preset minimum from `required_dtw_mem_size_for`. |

Native abort paths inside the DTW helper
(allocation failures, invalid windows, decoder errors) are
all converted to `WhisperError::StateLost` via the existing
exception shim — no `abort()` is reachable from safe Rust
through this surface.

## Memory safety

`whisper.cpp` is a binary parser of attacker-controllable model files
plus a substantial C++ inference path. The vendored submodule is
pinned to our fork branch
([`Findit-AI/whisper.cpp@rust`][fork-rust-branch]), which carries
fixes for upstream issues reachable from safe Rust:

- `whisper_kv_cache_free` made idempotent (closes a multi-decoder
  OOM double-free of a ggml backend buffer).
- `whisper_init_state` / `whisper_init_with_params_no_state` /
  `whisper_vad_init_with_params` wrapped in RAII so a throw mid-init
  releases the partial allocation rather than leaking the
  whisper_context / whisper_state.
- Tensor headers fully validated: `n_dims ∈ [0, 4]`, name length
  bounded, `ttype < GGML_TYPE_COUNT`, per-dim positivity, 64-bit
  overflow check on `nelements`.
- Hparams validated against generous-but-bounded ranges; min
  `n_text_ctx` enforced so the decode batch can hold the
  worst-case prompt.
- Special-token ids verified to fit `n_vocab` after the
  multilingual shift (closes a corrupt-vocab OOB into `logits[]`).
- File / buffer loaders throw on partial reads (peek-based EOF
  detection so clean end-of-tensor-list still terminates).
- Tensor-name set tracking rejects models that satisfy the
  loaded-count check by repeating one name.
- `ggml_log_set` installed once per process via `std::atomic`
  so concurrent `create_state` + `State::full` don't race on
  ggml's static logger globals.
- `vocab.num_languages()` synthesis null-checks
  `whisper_lang_str` (closes `std::string(nullptr)` UB).
- The abort callback is wired through every sched-based graph
  compute so cancellation interrupts the long-running encoder /
  decoder paths, not just the gaps between them.

A C++ exception-catching shim layer (`whispercpp_shim.cpp`) sits
between the safe Rust API and every throwing entry point. The
bindgen allowlist is enumerated symbol-by-symbol — only no-throw
raw `whisper_*` functions are exposed; every throwing function
goes through a `whispercpp_*` shim that catches and surfaces the
exception class as a sentinel (`WhisperError::ConstructorLost`,
`StateLost`, etc.).

`build.rs` includes a canary that scans the linked source for the
required patch markers and hard-fails the build if any are missing.

For the design details, the per-finding analysis lives on the fork
branch's commit history.

## Crate structure

| Crate            | Purpose                                                                                         |
|------------------|-------------------------------------------------------------------------------------------------|
| `whispercpp`     | Safe Rust API (`Context`, `State`, `Params`, `Lang`, `WhisperError`). End-user dependency.      |
| `whispercpp-sys` | Bindgen output + `build.rs` (cmake build, link directives) + the C++ exception-catching shim.   |

End users should depend on `whispercpp`. `whispercpp-sys` is
re-exported as `whispercpp::sys` for callers who need a raw
escape hatch (review every use carefully — only no-throw symbols
are exposed but it's `unsafe` regardless).

## Supported platforms

CI runs on `ubuntu-latest`, `macos-latest`, and `windows-latest`.
Sanitizer (ASan + UBSan) and Miri jobs gate the `unsafe` boundary
on every PR. MSRV is pinned in `Cargo.toml` and enforced via
`rust-version`.

## License

`whispercpp` is under the terms of both the MIT license and the
Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE), [LICENSE-MIT](LICENSE-MIT) for details.

Copyright (c) 2026 FinDIT Studio authors.

[whisper-cpp]: https://github.com/ggerganov/whisper.cpp
[fork-rust-branch]: https://github.com/Findit-AI/whisper.cpp/tree/rust
[Github-url]: https://github.com/findit-ai/whispercpp/
[CI-url]: https://github.com/findit-ai/whispercpp/actions/workflows/ci.yml
[doc-url]: https://docs.rs/whispercpp
[crates-url]: https://crates.io/crates/whispercpp
[codecov-url]: https://app.codecov.io/gh/findit-ai/whispercpp/
