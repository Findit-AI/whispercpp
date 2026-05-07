//! `whispercpp-sys` — raw FFI bindings to whisper.cpp.
//!
//! Everything below is `unsafe`-callable C ABI surface. Higher
//! layers (the `whispercpp` crate) wrap these in safe types;
//! end users should depend on `whispercpp` rather than this
//! crate directly.
//!
//! `build.rs` cmake-builds the vendored `whisper.cpp/` submodule
//! (pinned to a patched fork branch) and statically links the
//! resulting libraries. There is no pkg-config / system-install
//! path: the safe surface in the upper crate depends on patches
//! that only the bundled build supplies, and a stock libwhisper
//! would silently lose those guarantees. Bindgen writes the FFI
//! surface to `OUT_DIR/generated.rs`.

#![allow(unsafe_code)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]
#![allow(missing_docs)]

// Bindgen output is written to `OUT_DIR` by build.rs and
// `include!`'d here. An in-tree path (`src/generated.rs`)
// would break read-only builds (cargo vendor, Nix, Bazel,
// verified-source registry checkouts) and could race across
// builds with different feature sets.
//
// Trade-off: the FFI surface is no longer grep-able from a
// fresh checkout. Inspect via `cargo expand -p whispercpp-sys`
// or look at `target/.../build/whispercpp-sys-*/out/generated.rs`
// after a build.
include!(concat!(env!("OUT_DIR"), "/generated.rs"));
