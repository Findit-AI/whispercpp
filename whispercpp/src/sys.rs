//! Raw FFI bindings to whisper.cpp, sourced from the sibling
//! `whispercpp-sys` crate.
//!
//! This module is a thin re-export: `whispercpp-sys` owns the
//! build (cmake against the vendored `whisper.cpp/` submodule,
//! pinned to a patched fork branch) and the bindgen output. We
//! re-export here so `crate::sys::whisper_*` resolves through
//! the safe-wrapper crate without a path prefix change.
//!
//! All `unsafe` lives below this re-export boundary. Safe
//! wrappers in `context.rs`, `state.rs`, `params.rs` are
//! responsible for upholding lifetime + aliasing invariants.
pub use whispercpp_sys::*;
