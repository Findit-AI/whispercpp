// C++ exception-catching shim around whisper.cpp's public API.
//
// See whispercpp_shim.h for the rationale. Every wrapper in
// this file isolates its whisper_* call inside `try/catch (…)`
// so a `std::bad_alloc` / `std::system_error` / any other
// throw inside whisper.cpp becomes a sentinel return value
// instead of unwinding through the `extern "C"` boundary into
// Rust — which is undefined behaviour.

#include "whispercpp_shim.h"

#include <new>
#include <stdexcept>
#include <system_error>

// Per-thread "most recent caught constructor exception" slot.
//
// the constructor shims previously collapsed
// every failure (including caught exceptions) onto `nullptr`,
// indistinguishable from an upstream "init failed cleanly"
// nullptr return. Callers therefore couldn't tell a retryable
// failure (bad path, missing file) from a partial-init exception
// that leaked the `new whisper_context` / `new whisper_state`
// allocations.
//
// We expose a thread-local sentinel. Each constructor entry
// resets it to 0 and writes a `WHISPERCPP_ERR_*` value on catch.
// Callers pair every `nullptr` observation with
// `whispercpp_take_last_constructor_exception` to discriminate
// — and surface the exception case as a non-retryable fatal
// error so workers don't compound the leak.
//
// Why thread-local: concurrent context/state inits on different
// threads must not interleave their sentinels. Cross-thread
// reads are forbidden by the API contract (read on the same
// thread that made the call).
//
// Why a single slot for both `init_from_file` and `init_state`:
// the safe Rust API reads the sentinel synchronously after each
// constructor call, before any other shim entry on the same
// thread. There's no observation window where one constructor's
// exception could be misread as another's.
static thread_local int g_last_constructor_exception = 0;

extern "C" {

struct whisper_context * whispercpp_init_from_file_no_state(
    const char * path_model,
    struct whisper_context_params params)
{
    g_last_constructor_exception = 0;
    try {
        return whisper_init_from_file_with_params_no_state(path_model, params);
    } catch (const std::bad_alloc &) {
        g_last_constructor_exception = WHISPERCPP_ERR_BAD_ALLOC;
        return nullptr;
    } catch (const std::system_error &) {
        g_last_constructor_exception = WHISPERCPP_ERR_SYSTEM_ERROR;
        return nullptr;
    } catch (const std::exception &) {
        g_last_constructor_exception = WHISPERCPP_ERR_STD_EXCEPTION;
        return nullptr;
    } catch (...) {
        g_last_constructor_exception = WHISPERCPP_ERR_UNKNOWN_EXCEPTION;
        return nullptr;
    }
}

struct whisper_state * whispercpp_init_state(struct whisper_context * ctx)
{
    g_last_constructor_exception = 0;
    try {
        return whisper_init_state(ctx);
    } catch (const std::bad_alloc &) {
        g_last_constructor_exception = WHISPERCPP_ERR_BAD_ALLOC;
        return nullptr;
    } catch (const std::system_error &) {
        g_last_constructor_exception = WHISPERCPP_ERR_SYSTEM_ERROR;
        return nullptr;
    } catch (const std::exception &) {
        g_last_constructor_exception = WHISPERCPP_ERR_STD_EXCEPTION;
        return nullptr;
    } catch (...) {
        g_last_constructor_exception = WHISPERCPP_ERR_UNKNOWN_EXCEPTION;
        return nullptr;
    }
}

int whispercpp_take_last_constructor_exception(void)
{
    int v = g_last_constructor_exception;
    g_last_constructor_exception = 0;
    return v;
}

int whispercpp_full_with_state(
    struct whisper_context * ctx,
    struct whisper_state * state,
    struct whisper_full_params params,
    const float * samples,
    int n_samples)
{
    try {
        return whisper_full_with_state(ctx, state, params, samples, n_samples);
    } catch (const std::bad_alloc &) {
        return WHISPERCPP_ERR_BAD_ALLOC;
    } catch (const std::system_error &) {
        return WHISPERCPP_ERR_SYSTEM_ERROR;
    } catch (const std::exception &) {
        return WHISPERCPP_ERR_STD_EXCEPTION;
    } catch (...) {
        return WHISPERCPP_ERR_UNKNOWN_EXCEPTION;
    }
}

const char * whispercpp_print_system_info(void)
{
    try {
        return whisper_print_system_info();
    } catch (...) {
        return nullptr;
    }
}

} // extern "C"
