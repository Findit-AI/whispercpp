/// C-ABI shims around the whisper.cpp public API.
///
/// Every function declared here wraps its whisper.cpp
/// counterpart in a `try { ... } catch (...) { ... }` block.
/// flagged that whisper.cpp's `extern "C"`
/// entry points internally allocate `std::vector` and
/// construct `std::thread`, both of which can throw
/// (`std::bad_alloc`, `std::system_error`) under realistic
/// resource pressure. C++ exceptions propagating across an
/// `extern "C"` boundary into Rust code that hasn't compiled
/// with `panic=unwind` ABI compatibility is undefined
/// behaviour.
///
/// Convention:
///
/// * Constructors that return `T*` on success return
///   `nullptr` on caught exception (matches the C API's
///   existing failure mode).
/// * `int`-returning `whisper_full_with_state` returns a
///   negative sentinel for caught exceptions:
///     * `-100` for `std::bad_alloc` (OOM)
///     * `-101` for `std::system_error` (thread/system call)
///     * `-102` for any other `std::exception`
///     * `-103` for unknown / non-`std::exception` throws
///   These overlap whisper.cpp's own negative return codes
///   (which top out at `-7` in v1.8.4) without colliding;
///   the safe-Rust wrapper translates them into typed
///   `WhisperError` variants.

#ifndef WHISPERCPP_SHIM_H
#define WHISPERCPP_SHIM_H

#include "whisper.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Exception sentinels returned by `whispercpp_full_with_state`.
/// Defined as macros (not enums) so bindgen treats them as
/// plain integer constants the safe wrapper can match on.
#define WHISPERCPP_ERR_BAD_ALLOC      -100
#define WHISPERCPP_ERR_SYSTEM_ERROR   -101
#define WHISPERCPP_ERR_STD_EXCEPTION  -102
#define WHISPERCPP_ERR_UNKNOWN_EXCEPTION -103

/// `whisper_init_from_file_with_params_no_state` wrapped in
/// try/catch.
///
/// Returns `nullptr` on either:
/// * the upstream C API's documented failure (file not found,
///   model corrupt, backend init refused, etc. — these return
///   nullptr without throwing), OR
/// * a caught C++ exception inside the upstream init path
///   (`std::bad_alloc`, `std::system_error`,
///   `std::exception`, or anything else).
///
/// Use [`whispercpp_take_last_constructor_exception`] AFTER
/// observing `nullptr` to discriminate the two cases — the
/// caller MUST treat the exception case as fatal (the
/// upstream code has no RAII around `new whisper_context;`,
/// so any throw mid-init leaks the partial allocation).
/// 
struct whisper_context * whispercpp_init_from_file_no_state(
    const char * path_model,
    struct whisper_context_params params);

/// `whisper_init_state` wrapped in try/catch.
///
/// Same `nullptr` discrimination contract as
/// [`whispercpp_init_from_file_no_state`]: pair every
/// `nullptr` observation with
/// [`whispercpp_take_last_constructor_exception`] to
/// distinguish "upstream returned nullptr cleanly" (retryable)
/// from "exception caught, partial native allocation leaked"
/// (fatal). 
struct whisper_state * whispercpp_init_state(struct whisper_context * ctx);

/// Read-and-clear the most recent **constructor** exception
/// sentinel.
///
/// Set by [`whispercpp_init_from_file_no_state`] and
/// [`whispercpp_init_state`] inside their `catch` blocks; reset
/// to `0` on entry to those functions and again by this
/// accessor.
///
/// Returns one of:
/// * `0` — no exception was caught on the most recent
///   constructor call on this thread (a `nullptr` return means
///   the upstream C API returned `nullptr` cleanly, no leak).
/// * `WHISPERCPP_ERR_BAD_ALLOC` — `std::bad_alloc` during init.
/// * `WHISPERCPP_ERR_SYSTEM_ERROR` — `std::system_error`.
/// * `WHISPERCPP_ERR_STD_EXCEPTION` — other `std::exception`.
/// * `WHISPERCPP_ERR_UNKNOWN_EXCEPTION` — non-`std::exception`
///   throw.
///
/// Thread-local: each thread observes its own most-recent
/// sentinel. Callers must invoke this on the SAME thread that
/// made the constructor call, immediately after observing the
/// `nullptr` return. Inserting other shim calls between the
/// constructor and this read clobbers the sentinel.
int whispercpp_take_last_constructor_exception(void);

/// `whisper_full_with_state` wrapped in try/catch.
int whispercpp_full_with_state(
    struct whisper_context * ctx,
    struct whisper_state * state,
    struct whisper_full_params params,
    const float * samples,
    int n_samples);

/// `whisper_print_system_info` wrapped in try/catch. Upstream
/// rebuilds a static `std::string` via `s = ""; s += "..."; s
/// += std::to_string(...);` which can throw `std::bad_alloc`
/// across the C ABI. Returns NULL on any caught exception.
const char * whispercpp_print_system_info(void);

/// No-log tokenize, count-or-write.
///
/// Calls whisper.cpp's internal `tokenize(vocab, text)`
/// directly — bypasses the public `whisper_tokenize`'s
/// "too many tokens" `WHISPER_LOG_ERROR` so callers can
/// retry-on-too-small without polluting production logs.
/// The intended caller pattern in safe Rust is "try with
/// generous initial capacity; if the return is negative,
/// allocate `-return` and retry." Single upstream
/// tokenization on the happy path; two only on retry.
///
/// Return contract:
///
/// * `>= 0` — number of tokens written to `tokens` on
///   success.
/// * `< 0` and `> INT_MIN` —
///   `-(needed_count)` (buffer too small).
/// * `INT_MIN` — caught C++ exception
///   (`std::bad_alloc` under memory pressure, etc.).
///
/// The exception sentinel is `INT_MIN` (not the shared
/// `WHISPERCPP_ERR_*` ladder) so it cannot collide with any
/// realistic `-needed_count` value — no realistic decode
/// produces 2³¹ tokens.
int whispercpp_tokenize(
    struct whisper_context * ctx,
    const char * text,
    whisper_token * tokens,
    int n_max_tokens);

/// No-log token-count probe.
///
/// Calls whisper.cpp's internal `tokenize(vocab, text)`
/// directly — bypasses the public `whisper_tokenize`'s
/// "too many tokens" `WHISPER_LOG_ERROR` that would
/// otherwise fire on every non-empty probe-with-NULL-buffer
/// call. Returns the token count on success, or `INT_MIN`
/// on caught C++ exception.
int whispercpp_token_count(struct whisper_context * ctx, const char * text);

/// State-aware print/reset timing entry points.
///
/// Upstream `whisper_print_timings(ctx)` and
/// `whisper_reset_timings(ctx)` only read/reset
/// `ctx->state` — but the whispercpp Rust wrapper loads
/// contexts via `whispercpp_init_from_file_no_state` (so
/// `ctx->state == nullptr`) and runs inference through
/// separately-allocated `whisper_state` instances. The
/// legacy `whisper_*_timings` variants therefore observe
/// nothing useful in this configuration. These two
/// `_with_state` variants take an explicit
/// `whisper_state *` so the safe Rust `State::print_timings`
/// / `reset_timings` accessors route to the actually-active
/// state.
///
/// Both are no-throw: `_print_timings_with_state` only
/// calls `WHISPER_LOG_INFO` (fprintf-shaped, no allocation
/// in the format path); `_reset_timings_with_state` only
/// writes integer fields. No try/catch needed but exposed
/// here as `whispercpp_*` for the safe wrapper to address.
void whispercpp_print_timings_with_state(
    struct whisper_context * ctx,
    struct whisper_state * state);
void whispercpp_reset_timings_with_state(struct whisper_state * state);

/// `whisper_lang_id` wrapped in try/catch.
///
/// Upstream `whisper_lang_id` does
/// `g_lang.count(lang)` / `g_lang.at(lang)` with a
/// `const char *` key — both implicitly construct a
/// temporary `std::string` from the C string, which can
/// throw `std::bad_alloc`. The unknown-language log path
/// can also allocate. A throw across the `extern "C"`
/// boundary into Rust is undefined behaviour.
///
/// Return contract:
///
/// * `>= 0` — language id (success).
/// * `-1` — language not found (passthrough from upstream).
/// * `WHISPERCPP_ERR_BAD_ALLOC` / `..._SYSTEM_ERROR` /
///   `..._STD_EXCEPTION` / `..._UNKNOWN_EXCEPTION` —
///   caught C++ exception. These sit at `-100..=-103`,
///   distinct from the `-1` "not found" sentinel.
int whispercpp_lang_id(const char * lang);

#ifdef __cplusplus
}
#endif

#endif // WHISPERCPP_SHIM_H
