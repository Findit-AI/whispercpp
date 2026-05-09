#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]

mod context;
mod error;
mod lang;
mod params;
// `safety_audit` is comments-only — captures the per-method,
// per-axis safety audit so future review passes start from
// "what's been checked" instead of re-deriving the matrix
// every round. Private; no public re-export.
mod safety_audit;
mod state;
mod sys;

pub use context::{
  AlignmentHeadsPreset, Context, ContextParams, DEFAULT_DTW_MEM_SIZE, MAX_DTW_MEM_SIZE,
  MIN_DTW_MEM_SIZE, ModelDims, SUPPORTED_DTW_N_TEXT_CTX, required_dtw_mem_size_for, system_info,
};
pub use error::{WhisperError, WhisperResult};
pub use lang::{Lang, lang_id_for, lang_max_id};
pub use params::{
  MAX_BEAM_SIZE, MAX_INITIAL_TS_S, MAX_N_THREADS, MAX_TEMPERATURE, MIN_TEMPERATURE_INC, Params,
  SamplingStrategy,
};
pub use state::{Segment, State, Token};

/// Linked libwhisper version string (e.g. `"1.8.4"`).
///
/// Reads `WHISPER_VERSION` baked into the static library at
/// build time via `whisper_version()`. The returned slice
/// points into a static const literal owned by libwhisper;
/// lifetime is `'static`.
///
/// Returns `None` only if libwhisper hands back a NULL or
/// non-UTF-8 pointer (build corruption — should never happen
/// in a healthy build).
pub fn version() -> Option<&'static str> {
  // SAFETY: pure C accessor returning a pointer into a static
  // const literal. No per-context state, no lock, no throw.
  let raw = unsafe { sys::whisper_version() };
  if raw.is_null() {
    return None;
  }
  // SAFETY: NUL-terminated; static lifetime per whisper.cpp.
  let bytes = unsafe { core::ffi::CStr::from_ptr(raw).to_bytes() };
  core::str::from_utf8(bytes).ok()
}

#[cfg(test)]
mod tests {
  use super::*;

  /// `version()` should always return `Some("X.Y.Z")` for a
  /// healthy build — the bundled whisper.cpp's `WHISPER_VERSION`
  /// macro is a string literal baked into the static library at
  /// `cmake` time. NULL or non-UTF-8 would indicate a build-system
  /// corruption.
  #[test]
  #[cfg_attr(miri, ignore = "FFI: calls whisper_version")]
  fn version_returns_some_string_for_healthy_build() {
    let v = version().expect("whisper_version returned NULL or non-UTF-8");
    assert!(!v.is_empty(), "whisper_version returned empty string");
    // Must look like a typical semver-ish version (digit anywhere).
    assert!(
      v.bytes().any(|b| b.is_ascii_digit()),
      "version {v:?} contains no digits — looks corrupt"
    );
  }

  /// State-aware timing shims must be exported with C
  /// linkage (`extern "C"` block in the patched
  /// `whisper.cpp`). The shim header declares them inside
  /// `extern "C"` and bindgen generates Rust calls against
  /// the unmangled C symbols; if the C++ definitions ship
  /// without matching linkage, the symbols come out
  /// Itanium-mangled and won't resolve at final link.
  ///
  /// Test binaries that don't actually invoke
  /// `State::print_timings` / `reset_timings` link
  /// successfully even with mangled-name backing because
  /// of dead-symbol elimination — so the test must FORCE a
  /// symbol reference. Taking the function pointer is
  /// enough: the resulting cast emits an address load that
  /// the linker has to resolve, and a mangled-vs-unmangled
  /// mismatch surfaces as an unresolved-symbol link error.
  #[test]
  fn state_timing_shims_link_with_c_abi() {
    // Compile-time type signatures cross-check the bindgen-
    // generated Rust types against the C declarations in
    // `whispercpp_shim.h`. The runtime fn-pointer load
    // forces the linker to resolve the symbols (would fail
    // if the C++ definitions were name-mangled).
    let print_timings: unsafe extern "C" fn(*mut sys::whisper_context, *mut sys::whisper_state) =
      sys::whispercpp_print_timings_with_state;
    let reset_timings: unsafe extern "C" fn(*mut sys::whisper_state) =
      sys::whispercpp_reset_timings_with_state;
    // Use the addresses so a smart compiler can't elide the
    // load. Pointers compare unequal because they point at
    // distinct functions.
    assert_ne!(
      print_timings as usize, 0,
      "whispercpp_print_timings_with_state failed to link"
    );
    assert_ne!(
      reset_timings as usize, 0,
      "whispercpp_reset_timings_with_state failed to link"
    );
    assert_ne!(
      print_timings as usize, reset_timings as usize,
      "the two timing shims must be distinct symbols"
    );
  }
}
