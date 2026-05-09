//! Build script for the whisper.cpp FFI bindings.
//!
//! Compiles the vendored `whisper.cpp/` git submodule via
//! cmake-rs, links static. Feature flags translate to
//! `-DGGML_METAL=ON` etc. Output is a static `libwhisper.a`
//! plus the ggml satellite libraries that whisper.cpp's
//! CMakeLists produces.
//!
//! There is no pkg-config / system-install path: the bundled
//! source is patched in `OUT_DIR/whisper-src/` to close
//! several upstream memory-safety bugs,
//! and routing safe-Rust code through a stock libwhisper
//! would silently drop those guarantees.
//!
//! Bindgen runs against the resolved header set so the Rust
//! FFI matches the linked library's ABI. Output goes to
//! `OUT_DIR/generated.rs` (— must NOT mutate
//! the source tree).
//!
//! Bootstrap behaviour: when the submodule is missing this
//! script emits clear `cargo:warning=`s rather than panicking,
//! so `cargo check` still resolves the API. The actual link
//! step fails downstream, by design.

use std::{
  env,
  path::{Path, PathBuf},
};

fn main() {
  println!("cargo:rerun-if-changed=build.rs");
  println!("cargo:rerun-if-changed=wrapper.h");

  bundled_build();
}

// ─── Bundled path ────────────────────────────────────────────

fn bundled_build() {
  let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
  // The vendored submodule pinned via `.gitmodules` to the
  // `Findit-AI/whisper.cpp` fork's `rust` branch — which
  // carries our memory-safety patches as committed history
  // — is the SOLE source of truth for the whisper.cpp build.
  //
  // No `WHISPER_CPP_DIR` override: the Rust safety surface
  // (e.g. `State::full`'s free-on-sentinel path) relies on
  // the fork's idempotent `whisper_kv_cache_free` and other
  // patches being present in the linked binary. A pristine
  // upstream checkout shares the same ABI but lacks those
  // patches, so an env-var override would silently
  // reintroduce the double-free / use-after-free class the
  // wrapper closes. Users who need a different source must
  // edit `.gitmodules` (reviewable) rather than flip an env
  // var.
  let whisper_src = crate_dir.join("whisper.cpp");

  if !whisper_src.join("CMakeLists.txt").is_file() {
    println!(
      "cargo:warning=whisper.cpp source not found at {:?}.",
      whisper_src
    );
    println!("cargo:warning=Run `git submodule update --init --recursive` from the repo root.");
    println!(
      "cargo:warning=Skipping cmake + bindgen for now; link step will fail until the source is available."
    );
    return;
  }

  // Verify the linked source carries our patches. Cheap
  // canary: scan for any sentinel comment from the patch
  // set. If absent, the build hard-fails — Rust safety
  // assumptions in the wrapper depend on this.
  verify_patched_source(&whisper_src);

  // Tell cargo to rerun build.rs when files in the submodule
  // change so `git submodule update` picks up automatically.
  for top in ["CMakeLists.txt", "cmake", "include", "src", "ggml"] {
    let p = whisper_src.join(top);
    if p.exists() {
      println!("cargo:rerun-if-changed={}", p.display());
    }
  }

  let dst = build_whisper_cpp(&whisper_src);
  let bundled_includes = vec![
    whisper_src.join("include"),
    whisper_src.join("ggml").join("include"),
  ];
  // Build the shim BEFORE emitting whisper.cpp's link
  // directives. GNU ld resolves left-to-right; the shim
  // depends on `whisper_*` symbols so it must appear first
  // in the link list. cc::Build emits its `link-lib` line
  // immediately on `compile`.
  build_shim(&bundled_includes);
  emit_bundled_link_directives(&dst);
  let bundled_args: Vec<String> = bundled_includes
    .iter()
    .map(|p| format!("-I{}", p.display()))
    .collect();
  generate_bindings_with_args(&bundled_args);
}

/// Hard-fail the build if the linked whisper.cpp source is
/// missing the rust-branch patch set. The Rust wrapper's
/// memory-safety guarantees (e.g. `State::full`'s
/// free-on-sentinel path in relying on 's
/// idempotent `whisper_kv_cache_free`) are unsound against a
/// pristine upstream tree even though the ABI is identical.
///
/// Strategy: scan `src/whisper.cpp` for one or more sentinel
/// comments inserted by the rust-branch patches. If any
/// expected marker is missing the build refuses to proceed.
///
/// This catches both `git submodule update` against unpatched
/// upstream AND someone manually replacing the submodule with
/// a different tree.
fn verify_patched_source(whisper_src: &Path) {
  // Sentinels chosen from the highest-leverage patches —
  // the ones whose absence would re-introduce the
  // double-free / null-deref / leak / native-abort hazards
  // the Rust wrapper assumes are closed. Each entry is
  // `(file_relative_to_whisper_src, expected_marker)`; the
  // build hard-fails if any are absent.
  //
  // We split across both `src/whisper.cpp` and
  // `ggml/src/ggml.c` because some safety patches sit in
  // each. The ggml patch (OOM-safe `ggml_init`) is what
  // turns the DTW scratch-allocation OOM path from
  // `abort()`-uncatchable into a `WhisperError::StateLost`
  // recovery — without it the wrapper's `dtw scratch
  // alloc-fail throws` patch is dead code.
  const REQUIRED_MARKERS: &[(&str, &str)] = &[
    (
      "src/whisper.cpp",
      "whispercpp-sys: kv_cache_free idempotent fix",
    ),
    ("src/whisper.cpp", "whispercpp-sys: read_safe zero-init"),
    ("src/whisper.cpp", "whispercpp-sys: init_state RAII entry"),
    ("src/whisper.cpp", "whispercpp-sys: init_context RAII entry"),
    (
      "src/whisper.cpp",
      "whispercpp-sys: tensor header validation (model_load)",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: ggml_log_set once-per-process",
    ),
    ("src/whisper.cpp", "whispercpp-sys: hparams validation"),
    ("src/whisper.cpp", "whispercpp-sys: lang_str null guard"),
    (
      "src/whisper.cpp",
      "whispercpp-sys: special-token bounds check",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: path_model assignment guard",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: sched abort callback wiring",
    ),
    ("src/whisper.cpp", "whispercpp-sys: vad_init RAII guard"),
    ("src/whisper.cpp", "whispercpp-sys: dtw scratch RAII guard"),
    (
      "src/whisper.cpp",
      "whispercpp-sys: dtw scratch alloc-fail throws",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: dtw token assignment bounded",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: dtw short-window medfilt clamp",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: dtw audio_ctx override guard",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: ggml_init throw-on-null wrapper",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: dtw decode failure throws",
    ),
    ("src/whisper.cpp", "whispercpp-sys: kv buffer null throws"),
    (
      "src/whisper.cpp",
      "whispercpp-sys: dtw backtrace impossible-case throws",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: dtw aheads_cross_QKs invariants throw",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: token_to_str sparse-vocab no-throw",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: hparams head divisibility check",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: dtw backend compute throws",
    ),
    ("src/whisper.cpp", "whispercpp-sys: dtw t_dtw sentinel init"),
    (
      "src/whisper.cpp",
      "whispercpp-sys: whisper_mel POD field default-init",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: state-aware timing entry points",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: state-aware print drops total time",
    ),
    ("src/whisper.cpp", "whispercpp-sys: log_internal va_copy"),
    ("src/whisper.cpp", "whispercpp-sys: no-log token count shim"),
    ("src/whisper.cpp", "whispercpp-sys: no-log tokenize shim"),
    (
      "src/whisper.cpp",
      "whispercpp-sys: tokenize size_t→int overflow guard",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: whisper_tokenize size_t→int overflow guard",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: whisper_tokenize INT_MIN propagation",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: whisper_token_count INT_MIN propagation",
    ),
    (
      "ggml/src/ggml-alloc.c",
      "whispercpp-sys: gallocr_new_n OOM-safe alloc",
    ),
    (
      "ggml/src/ggml-alloc.c",
      "whispercpp-sys: gallocr_reserve_n_impl OOM-safe paths",
    ),
    (
      "ggml/src/ggml-alloc.c",
      "whispercpp-sys: dyn_tallocr_new OOM-safe alloc",
    ),
    (
      "ggml/src/ggml-alloc.c",
      "whispercpp-sys: dyn_tallocr_alloc OOM-safe sentinel",
    ),
    (
      "ggml/src/ggml-alloc.c",
      "whispercpp-sys: gallocr alloc-failure flag",
    ),
    (
      "ggml/src/ggml-alloc.c",
      "whispercpp-sys: gallocr_free_node invalid-chunk guard",
    ),
    (
      "ggml/src/ggml-alloc.c",
      "whispercpp-sys: hash_set / hash_values atomic commit",
    ),
    (
      "ggml/src/ggml-alloc.c",
      "whispercpp-sys: node_allocs growth transactional",
    ),
    (
      "ggml/src/ggml-alloc.c",
      "whispercpp-sys: leaf_allocs growth transactional",
    ),
    (
      "ggml/src/ggml-alloc.c",
      "whispercpp-sys: vbuffer realloc transactional",
    ),
    (
      "ggml/src/ggml-backend.cpp",
      "whispercpp-sys: sched_alloc_splits reserve_n return check",
    ),
    ("src/whisper.cpp", "whispercpp-sys: backend_init RAII"),
    (
      "src/whisper.cpp",
      "whispercpp-sys: sched_graph_init NULL guard",
    ),
    (
      "ggml/src/ggml-backend.cpp",
      "whispercpp-sys: backend_sched_new OOM-safe alloc",
    ),
    (
      "ggml/src/ggml-backend.cpp",
      "whispercpp-sys: hash_set_new OOM-safe alloc",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: vocab count consistency check",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: vocab post-synthesis size check",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: model_load RAII for raw ggml allocations",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: model_load tensor-prep RAII",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: model_load buffer-registration RAII",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: vad_load RAII for raw ggml allocations",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: vad_load tensor-prep RAII",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: vad_load buffer-registration RAII",
    ),
    (
      "src/whisper.cpp",
      "whispercpp-sys: auto-detect bounded to model lang range",
    ),
    (
      "ggml/src/ggml.c",
      "whispercpp-sys: ggml_init OOM-safe context alloc",
    ),
  ];

  // Read each referenced file once, then check every
  // marker that points at it. Group markers by file so we
  // don't re-read the same source on every iteration.
  use std::collections::HashMap;
  let mut by_file: HashMap<&str, Vec<&str>> = HashMap::new();
  for (file, marker) in REQUIRED_MARKERS {
    by_file.entry(*file).or_default().push(*marker);
  }

  let mut missing: Vec<(&str, &str)> = Vec::new();
  for (rel, markers) in &by_file {
    let target = whisper_src.join(rel);
    let body = match std::fs::read_to_string(&target) {
      Ok(b) => b,
      Err(e) => panic!(
        "whispercpp-sys: failed to read {} for patch verification: {e}",
        target.display()
      ),
    };
    for m in markers {
      if !body.contains(*m) {
        missing.push((*rel, *m));
      }
    }
  }

  if !missing.is_empty() {
    panic!(
      "whispercpp-sys: the linked whisper.cpp source under {} is missing rust-branch patches \
       (required marker{} absent: {:?}).\n\n\
       The Rust safety surface depends on these patches; building against unpatched upstream \
       reintroduces multi-decoder double-free / use-after-free / null-deref / native-abort \
       classes.\n\n\
       Fix: ensure the submodule tracks `Findit-AI/whisper.cpp` branch `rust`. Run\n  \
       git submodule update --init --recursive\n\
       from the repo root. If you intentionally pointed at a different source, add equivalent \
       patches and the matching marker comments before retrying.",
      whisper_src.display(),
      if missing.len() == 1 { "" } else { "s" },
      missing,
    );
  }
}

/// Compile `whispercpp_shim.cpp` into a `libwhispercpp_shim.a`
/// staticlib in `OUT_DIR`, and emit the link directive for it.
///
/// The shim catches C++ exceptions inside whisper.cpp so they
/// can't unwind across `extern "C"` into Rust. It must be
/// linked BEFORE the whisper static libs in the GNU ld
/// dependency chain so the shim's references to `whisper_*`
/// resolve.
fn build_shim(include_paths: &[PathBuf]) {
  let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
  let mut build = cc::Build::new();
  build
    .cpp(true)
    .file(crate_dir.join("whispercpp_shim.cpp"))
    .flag_if_supported("-std=c++17")
    .flag_if_supported("/std:c++17");
  for inc in include_paths {
    build.include(inc);
  }
  // `cc::Build::compile` emits `cargo:rustc-link-lib=static=...`
  // and `cargo:rustc-link-search=native=...` automatically.
  build.compile("whispercpp_shim");
  // Tell cargo to rerun the shim build when the source files
  // change. (cc doesn't do this for us.)
  println!("cargo:rerun-if-changed=whispercpp_shim.cpp");
  println!("cargo:rerun-if-changed=whispercpp_shim.h");
}

/// Drive the cmake build. Returns the install root cmake-rs
/// produced (typically `OUT_DIR/`).
fn build_whisper_cpp(whisper_src: &PathBuf) -> PathBuf {
  let mut cfg = cmake::Config::new(whisper_src);
  cfg
    .define("BUILD_SHARED_LIBS", "OFF")
    .define("WHISPER_BUILD_EXAMPLES", "OFF")
    .define("WHISPER_BUILD_TESTS", "OFF")
    .define("WHISPER_BUILD_SERVER", "OFF")
    // Force OpenMP off. ggml's CMake auto-
    // detects OpenMP; if the host has it (Linux + libgomp,
    // macOS + brew libomp, etc.) it links against the
    // OpenMP runtime which our `cargo:rustc-link-lib=` set
    // doesn't emit, producing platform-specific link
    // surprises. The wrapper also caps `n_threads = 1`, so
    // OpenMP can't help anyway. Explicit OFF makes the
    // bundled build deterministic across runners.
    .define("GGML_OPENMP", "OFF")
    // ggml fast-math + Apple Accelerate / OpenBLAS are decided
    // per-feature below.
    .profile("Release");

  if cfg!(feature = "metal") {
    cfg.define("GGML_METAL", "ON");
    cfg.define("GGML_METAL_NDEBUG", "ON");
    // Embed the metal shader library bytes into libggml-metal.a
    // so the runtime doesn't need a sibling `default.metallib`.
    cfg.define("GGML_METAL_EMBED_LIBRARY", "ON");
  } else {
    cfg.define("GGML_METAL", "OFF");
  }

  if cfg!(feature = "coreml") {
    cfg.define("WHISPER_COREML", "ON");
    // Enable the post-init fallback: if the `.mlmodelc`
    // companion is missing at runtime, fall back to the GGML
    // encoder rather than aborting. This is what whisper-cli
    // does by default.
    cfg.define("WHISPER_COREML_ALLOW_FALLBACK", "ON");
  }

  if cfg!(feature = "openblas") {
    cfg.define("GGML_BLAS", "ON");
    cfg.define("GGML_BLAS_VENDOR", "OpenBLAS");
  } else if cfg!(target_vendor = "apple") && !cfg!(feature = "metal") {
    // Apple CPU build: prefer the system Accelerate framework.
    cfg.define("GGML_BLAS", "ON");
    cfg.define("GGML_BLAS_VENDOR", "Apple");
  }

  // ── Vendor-specific GPU backends ────────────────────────
  // Each `-DGGML_*=ON` triggers cmake's matching find_package
  // / FetchContent for the SDK (CUDA Toolkit, ROCm, oneAPI,
  // etc.). The user is expected to have the SDK installed; we
  // don't auto-fetch.
  if cfg!(feature = "cuda") {
    cfg.define("GGML_CUDA", "ON");
  }
  if cfg!(feature = "hipblas") {
    // Renamed `GGML_HIPBLAS` → `GGML_HIP` upstream around
    // ggml 0.10. We keep the Rust feature name `hipblas` to
    // match the convention whisper-rs / llama-cpp-rs adopted
    // before the upstream rename.
    cfg.define("GGML_HIP", "ON");
  }
  if cfg!(feature = "sycl") {
    cfg.define("GGML_SYCL", "ON");
  }
  if cfg!(feature = "musa") {
    cfg.define("GGML_MUSA", "ON");
  }

  // ── Cross-platform GPU ─────────────────────────────────
  if cfg!(feature = "vulkan") {
    cfg.define("GGML_VULKAN", "ON");
  }
  if cfg!(feature = "opencl") {
    cfg.define("GGML_OPENCL", "ON");
  }

  // ── Encoder accelerators ───────────────────────────────
  if cfg!(feature = "openvino") {
    cfg.define("WHISPER_OPENVINO", "ON");
  }

  cfg.build()
}

/// Tell cargo which static libraries to link, in the right
/// order for the GNU/macos/MSVC linkers. cmake-rs's `build`
/// returns `<OUT_DIR>/`, with libs under `lib/`.
fn emit_bundled_link_directives(install_root: &Path) {
  let lib_dir = install_root.join("lib");
  println!("cargo:rustc-link-search=native={}", lib_dir.display());

  // Order matters for GNU ld: depending libs first, low-level
  // last. whisper depends on ggml; ggml's metal/blas/coreml
  // sub-libs are leaves.
  println!("cargo:rustc-link-lib=static=whisper");
  println!("cargo:rustc-link-lib=static=ggml");
  println!("cargo:rustc-link-lib=static=ggml-base");
  println!("cargo:rustc-link-lib=static=ggml-cpu");

  // On Apple Silicon, whisper.cpp's CMake also builds the
  // ggml-blas backend automatically (the BLAS-via-Accelerate
  // path), even when Metal is the primary backend. We link it
  // unconditionally on Apple targets so the resulting binary
  // resolves `ggml_backend_blas_reg`.
  if cfg!(target_vendor = "apple") {
    println!("cargo:rustc-link-lib=static=ggml-blas");
    println!("cargo:rustc-link-lib=framework=Accelerate");
  }
  if cfg!(feature = "metal") {
    println!("cargo:rustc-link-lib=static=ggml-metal");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalKit");
    println!("cargo:rustc-link-lib=framework=Foundation");
  }
  if cfg!(feature = "coreml") {
    println!("cargo:rustc-link-lib=static=whisper.coreml");
    println!("cargo:rustc-link-lib=framework=CoreML");
  }
  if cfg!(feature = "openblas") {
    println!("cargo:rustc-link-lib=dylib=openblas");
  }

  // ── CUDA ───────────────────────────────────────────────
  // cmake produces `libggml-cuda.a`; the runtime resolves
  // CUDA Toolkit symbols via `cudart`/`cublas` dylibs in
  // `$CUDA_PATH/lib64` (Linux) or `\lib\x64` (Windows). The
  // user must have the CUDA Toolkit installed; we don't ship
  // it. `cargo:rustc-link-search` is left to the system
  // default — `LD_LIBRARY_PATH` / Windows `PATH` covers it.
  if cfg!(feature = "cuda") {
    println!("cargo:rustc-link-lib=static=ggml-cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
  }

  // ── ROCm / HIP (AMD) ───────────────────────────────────
  if cfg!(feature = "hipblas") {
    println!("cargo:rustc-link-lib=static=ggml-hip");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=hipblas");
    println!("cargo:rustc-link-lib=dylib=rocblas");
  }

  // ── Intel SYCL / oneAPI ────────────────────────────────
  if cfg!(feature = "sycl") {
    println!("cargo:rustc-link-lib=static=ggml-sycl");
    println!("cargo:rustc-link-lib=dylib=sycl");
    println!("cargo:rustc-link-lib=dylib=OpenCL");
    println!("cargo:rustc-link-lib=dylib=mkl_sycl");
    println!("cargo:rustc-link-lib=dylib=mkl_intel_ilp64");
    println!("cargo:rustc-link-lib=dylib=mkl_tbb_thread");
    println!("cargo:rustc-link-lib=dylib=mkl_core");
  }

  // ── Moore Threads MUSA ─────────────────────────────────
  if cfg!(feature = "musa") {
    println!("cargo:rustc-link-lib=static=ggml-musa");
    println!("cargo:rustc-link-lib=dylib=musa");
    println!("cargo:rustc-link-lib=dylib=musart");
    println!("cargo:rustc-link-lib=dylib=mublas");
  }

  // ── Vulkan (cross-platform GPU) ────────────────────────
  if cfg!(feature = "vulkan") {
    println!("cargo:rustc-link-lib=static=ggml-vulkan");
    if cfg!(target_os = "macos") {
      // MoltenVK ships a `vulkan` dylib that translates to Metal.
      println!("cargo:rustc-link-lib=dylib=vulkan");
    } else if cfg!(target_os = "windows") {
      println!("cargo:rustc-link-lib=dylib=vulkan-1");
    } else {
      println!("cargo:rustc-link-lib=dylib=vulkan");
    }
  }

  // ── OpenCL (mobile GPUs / Adreno) ──────────────────────
  if cfg!(feature = "opencl") {
    println!("cargo:rustc-link-lib=static=ggml-opencl");
    if cfg!(target_os = "macos") {
      println!("cargo:rustc-link-lib=framework=OpenCL");
    } else {
      println!("cargo:rustc-link-lib=dylib=OpenCL");
    }
  }

  // ── OpenVINO (Intel encoder accelerator) ───────────────
  if cfg!(feature = "openvino") {
    println!("cargo:rustc-link-lib=static=whisper.openvino");
    println!("cargo:rustc-link-lib=dylib=openvino");
    println!("cargo:rustc-link-lib=dylib=openvino_c");
  }

  // C++ stdlib — whisper.cpp / ggml are C++.
  if cfg!(target_os = "macos") {
    println!("cargo:rustc-link-lib=dylib=c++");
  } else if cfg!(target_os = "linux") {
    println!("cargo:rustc-link-lib=dylib=stdc++");
  }
}

// ─── Bindgen ─────────────────────────────────────────────────

/// Run bindgen against a curated `wrapper.h` and write the
/// result to `$OUT_DIR/generated.rs`.
///
/// **Why OUT_DIR, not in-tree.** flagged the
/// previous in-tree path (`src/generated.rs`) as breaking
/// read-only builds — cargo's standard `vendor` workflow,
/// Nix-style fixed-output derivations, Bazel sandboxes, and
/// verified-source registry checkouts all forbid build.rs
/// from mutating the source tree. Per cargo's contract, every
/// build.rs side-effect goes under `OUT_DIR`. The
/// `include!` glue lives in `src/lib.rs`.
///
/// Trade-off: the FFI surface is no longer grep-able from a
/// fresh checkout. Inspect via `cargo expand
/// -p whispercpp-sys` or look at
/// `target/<host>/<profile>/build/whispercpp-sys-<hash>/out/generated.rs`
/// after a build.
fn generate_bindings_with_args(clang_args: &[String]) {
  let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
  let header = crate_dir.join("wrapper.h");

  let mut builder = bindgen::Builder::default().header(header.to_string_lossy().to_string());
  for arg in clang_args {
    builder = builder.clang_arg(arg);
  }
  let bindings = builder
    // Only the symbols the safe wrapper actually consumes.
    // narrowed this from `whisper_.*` because
    // the broad allowlist exposed unshimmed throwing C++
    // entry points (e.g. `whisper_vad_init_*` whose file
    // loaders throw `std::runtime_error` on truncated
    // models, and `whisper_full_with_state` whose
    // exceptions cross `extern "C"` into Rust as UB). New
    // raw symbols need an explicit allowlist add and a
    // matching audit: confirm the upstream function cannot
    // throw, OR add a `whispercpp_*` shim wrapping it in
    // try/catch.
    //
    // No-throw raw entry points (verified):
    //   - `*_default_params`           — value-returning
    //   - `*_free`, `*_free_state`     — destructors
    //   - `*_n_*`, `*_token_*`,
    //     `*_is_multilingual`,
    //     `*_lang_str`,
    //     `*_model_type_readable`,
    //     `*_full_get_*_from_state`    — pure read accessors
    //   - `*_token_to_str`             — would throw via
    //     `id_to_token.at` but the safe wrapper validates
    //     the bound first.
    //
    // Throwing entry points routed through `whispercpp_*`
    // shims:
    //   - `whisper_init_from_file_with_params_no_state` →
    //     `whispercpp_init_from_file_no_state`
    //   - `whisper_init_state` →
    //     `whispercpp_init_state`
    //   - `whisper_full_with_state` →
    //     `whispercpp_full_with_state`
    //   - `whisper_print_system_info` →
    //     `whispercpp_print_system_info`
    //
    // VAD entry points (`whisper_vad_*`) are NOT exposed —
    // the safe wrapper doesn't surface VAD, and their file
    // loaders throw on truncated models.
    .allowlist_function("whisper_context_default_params")
    .allowlist_function("whisper_full_default_params")
    .allowlist_function("whisper_free")
    .allowlist_function("whisper_free_state")
    .allowlist_function("whisper_is_multilingual")
    .allowlist_function("whisper_n_vocab")
    .allowlist_function("whisper_n_audio_ctx")
    .allowlist_function("whisper_n_text_ctx")
    .allowlist_function("whisper_token_eot")
    .allowlist_function("whisper_token_sot")
    .allowlist_function("whisper_token_beg")
    .allowlist_function("whisper_token_to_str")
    .allowlist_function("whisper_lang_str")
    .allowlist_function("whisper_model_type_readable")
    .allowlist_function("whisper_full_n_segments_from_state")
    .allowlist_function("whisper_full_lang_id_from_state")
    .allowlist_function("whisper_full_get_segment_t0_from_state")
    .allowlist_function("whisper_full_get_segment_t1_from_state")
    .allowlist_function("whisper_full_get_segment_text_from_state")
    .allowlist_function("whisper_full_get_segment_no_speech_prob_from_state")
    .allowlist_function("whisper_full_get_segment_speaker_turn_next_from_state")
    .allowlist_function("whisper_full_n_tokens_from_state")
    .allowlist_function("whisper_full_get_token_data_from_state")
    // Issue #2 accessors — no-throw plain reads of vocab /
    // hparams / static const tables. Each was audited by
    // reading whisper.cpp's source for the symbol; functions
    // that touched `std::vector` / `std::string` allocations
    // (e.g. `whisper_tokenize`) go through a `whispercpp_*`
    // shim instead.
    .allowlist_function("whisper_version")
    .allowlist_function("whisper_lang_max_id")
    // `whisper_lang_id` is intentionally NOT exposed: it
    // does `g_lang.count(const char *)` / `.at(const char *)`
    // which implicitly constructs a `std::string` from the
    // C string, throwing `std::bad_alloc` under memory
    // pressure. Safe Rust must go through the
    // `whispercpp_lang_id` shim instead.
    .allowlist_function("whisper_lang_str_full")
    .allowlist_function("whisper_n_len_from_state")
    .allowlist_function("whisper_model_n_audio_state")
    .allowlist_function("whisper_model_n_audio_head")
    .allowlist_function("whisper_model_n_audio_layer")
    .allowlist_function("whisper_model_n_text_state")
    .allowlist_function("whisper_model_n_text_head")
    .allowlist_function("whisper_model_n_text_layer")
    .allowlist_function("whisper_model_n_mels")
    .allowlist_function("whisper_model_ftype")
    .allowlist_function("whisper_token_translate")
    .allowlist_function("whisper_token_transcribe")
    .allowlist_function("whisper_token_prev")
    .allowlist_function("whisper_token_nosp")
    .allowlist_function("whisper_token_not")
    .allowlist_function("whisper_token_solm")
    .allowlist_function("whisper_token_lang")
    // `whisper_print_timings(ctx)` / `whisper_reset_timings(ctx)`
    // are intentionally NOT exposed: both upstream
    // implementations only operate on `ctx->state`, but the
    // wrapper loads contexts via `_no_state` so `ctx->state`
    // is always nullptr. Safe Rust uses the state-aware
    // `whispercpp_print_timings_with_state` /
    // `whispercpp_reset_timings_with_state` shims (in
    // `whisper.cpp` under the `state-aware timing entry
    // points` patch) instead — exposed via `State::print_timings`
    // and `State::reset_timings`.
    // ggml's logger setter is referenced from our context
    // init lock comment but not directly called. We expose
    // the whole ggml_log_* family for diagnostic use.
    .allowlist_function("ggml_log_.*")
    // Shim entry points — no-throw at the boundary.
    .allowlist_function("whispercpp_.*")
    // Type allowlist: every struct / enum the function
    // signatures above transitively require.
    .allowlist_type("whisper_context")
    .allowlist_type("whisper_state")
    .allowlist_type("whisper_context_params")
    .allowlist_type("whisper_full_params")
    .allowlist_type("whisper_token")
    .allowlist_type("whisper_token_data")
    .allowlist_type("whisper_pos")
    .allowlist_type("whisper_seq_id")
    .allowlist_type("whisper_sampling_strategy")
    .allowlist_type("whisper_grammar_element")
    .allowlist_type("whisper_segment")
    .allowlist_type("whisper_progress_callback")
    .allowlist_type("whisper_new_segment_callback")
    .allowlist_type("whisper_encoder_begin_callback")
    .allowlist_type("whisper_logits_filter_callback")
    .allowlist_type("ggml_log_.*")
    .allowlist_var("WHISPER_.*")
    // Shim exception sentinels (WHISPERCPP_ERR_*). state.rs
    // needs them to discriminate "shim caught a C++ exception
    // → state may be corrupt → poison" from "whisper.cpp
    // returned a documented error code".
    .allowlist_var("WHISPERCPP_.*")
    // CargoCallbacks calls
    // `println!("cargo:rerun-if-changed=...")` for every
    // header bindgen pulled. Those land under whisper.cpp/...
    // (or the system include path) so we DO want them — a
    // header change should re-bindgen.
    .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
    .layout_tests(false)
    .derive_default(true)
    .derive_debug(true)
    .generate()
    .expect("bindgen failed");

  let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
  let dest = out_dir.join("generated.rs");
  let body = bindings.to_string();
  let header_comment = format!(
    "// @generated\n\
     //\n\
     // whisper.cpp FFI surface — produced by bindgen against\n\
     // the bundled submodule (`whispercpp-sys/whisper.cpp/`),\n\
     // patched in OUT_DIR. Do not edit by hand.\n\
     //\n\
     // Source crate: {pkg} {ver}\n\
     // Source header: wrapper.h -> whisper.h + whispercpp_shim.h\n\
     //\n\n",
    pkg = env!("CARGO_PKG_NAME"),
    ver = env!("CARGO_PKG_VERSION"),
  );

  let new_contents = format!("{header_comment}{body}");
  std::fs::write(&dest, new_contents).expect("failed to write OUT_DIR/generated.rs");
}
