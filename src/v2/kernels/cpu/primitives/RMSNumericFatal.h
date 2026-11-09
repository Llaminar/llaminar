/**
 * @file RMSNumericFatal.h
 * @brief Release-safe fatal sink for CPU RMS arithmetic corruption and retained-policy rejection.
 *
 * RMS arithmetic corruption is unrecoverable: a nonfinite summary or an
 * unrepresentable inverse must never be published as repaired bytes or be
 * routed into a different arithmetic program. Exceptions cannot cross an OpenMP
 * collective, so production leaves terminate through one unconditional abort
 * sink instead of a throw that may decay to std::terminate without a precise
 * diagnostic. The failure authority is typed (`NumericFailure`) so no cause is
 * null or ambiguous, and every report carries the exact phase-boundary detail.
 * Rows already published by an earlier pass are not rolled back.
 * The sink is active in every build type; ASSERT is disabled in Release, so it
 * is not a substitute. Cold policy admission follows the same rule: a
 * std::invalid_argument that is recoverable for a standalone caller must not
 * escape an external retained team, so `rejectPolicy` throws standalone and
 * terminates explicitly inside the retained collective.
 */
#pragma once
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <stdexcept>

namespace llaminar2::primitives::rms_numeric
{
/** @brief Typed cause of a fatal RMS numeric failure; never null or ambiguous. */
enum class NumericFailure
{
    NonfiniteSummary,           ///< Observed squared-summary is NaN/Inf.
    NonfiniteInverse,           ///< Computed inverse is NaN/Inf, including an undefined zero denominator.
    UnrepresentablePublication, ///< Normalization, scale or code is nonfinite/unrepresentable.
};
/** @brief Private cause spelling for one atomic fatal line. */
inline const char *causeName(NumericFailure cause) noexcept
{
    switch (cause)
    {
    case NumericFailure::NonfiniteSummary: return "NonfiniteSummary";
    case NumericFailure::NonfiniteInverse: return "NonfiniteInverse";
    case NumericFailure::UnrepresentablePublication: return "UnrepresentablePublication";
    }
    return "Unknown";
}
/** @brief Terminate for a typed RMS arithmetic failure with one precise diagnostic.
 * @param cause Typed failure authority; @param detail Exact phase-boundary defect.
 * @details Unconditional single-line fprintf + abort: one stdio call keeps the
 * prefix, typed cause and detail atomic under concurrent failing workers (no
 * interleaved partial lines from separate fputs calls). An exception crossing
 * a retained team would rotate to std::terminate without this guarantee. No
 * throw can return, so an OpenMP worker never continues into publication.
 */
[[noreturn]] inline void rejectNumeric(NumericFailure cause, const char *detail) noexcept
{
    std::fprintf(stderr, "ERROR: CPU RMS numeric contract violation [%s]: %s\n", causeName(cause), detail);
    std::abort();
}
/** @brief Terminate for an unsupported policy observed inside an external retained team.
 * @param detail Exact unsupported-policy defect.
 * @details The same cold admission that throws std::invalid_argument for a
 * standalone caller must not escape an external retained OpenMP collective;
 * this explicit abort keeps the diagnostic and prevents collective corruption
 * without a fallback path, retry or alternate program.
 */
[[noreturn]] inline void rejectFatalPolicy(const char *detail) noexcept
{
    std::fprintf(stderr, "ERROR: CPU RMS unsupported policy in retained collective: %s\n", detail);
    std::abort();
}
/** @brief Cold policy admission: recoverable standalone invalid_argument or fatal retained state.
 * @param detail Exact unsupported-policy or invalid-metadata defect.
 * @details Detection is lexical (`omp_get_level() > 0`, not the active-claim
 * `omp_in_parallel()`): a serialized `num_threads(1)` or inactive
 * `if(false)` retained structured region is still a team boundary, so the
 * call must terminate explicitly instead of throwing across it. Standalone
 * callers keep a recoverable typed std::invalid_argument. No alternate
 * program, retry or deferred flag is selected.
 */
[[noreturn]] inline void rejectPolicy(const char *detail)
{
    if (omp_get_level() > 0) rejectFatalPolicy(detail);
    throw std::invalid_argument(detail);
}
/** @brief Cold metadata admission preserving the original std::out_of_range class standalone.
 * @param detail Exact out-of-range metadata defect.
 * @details Row-index accessors historically throw std::out_of_range; that
 * standalone class is preserved while a lexical retained collective
 * (omp_get_level() > 0) gets the explicit fatal instead of an exception
 * escaping the structured region.
 */
[[noreturn]] inline void rejectRangePolicy(const char *detail)
{
    if (omp_get_level() > 0) rejectFatalPolicy(detail);
    throw std::out_of_range(detail);
}
/** @brief Cold admission preserving the original std::runtime_error class standalone.
 * @param detail Exact unsupported-implementation defect.
 * @details Standalone callers keep the recoverable std::runtime_error; a
 * lexical retained collective terminates explicitly.
 */
[[noreturn]] inline void rejectRuntimePolicy(const char *detail)
{
    if (omp_get_level() > 0) rejectFatalPolicy(detail);
    throw std::runtime_error(detail);
}
} // namespace llaminar2::primitives::rms_numeric
