// bindgen entry point. Pulls only what we use. Adding new
// whisper.cpp surface to the safe wrapper means adding the
// matching `#include` here AND extending the `allowlist_*`
// directives in `build.rs` — there is no implicit re-export.
//
// `whispercpp_shim.h` exposes the exception-catching C ABI
// shim layer. Every safe-Rust entry point
// that can run user-controlled allocations / thread spawns
// goes through these shims rather than calling whisper.cpp
// directly.
#include "whisper.h"
#include "whispercpp_shim.h"
