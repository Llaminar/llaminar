#pragma once

/**
 * @file Assertions.h
 * @brief Fail-fast assertion macros for Llaminar V2
 *
 * PHILOSOPHY: Fail loudly and immediately when invariants are violated.
 * Silent failures lead to subtle bugs that are hard to track down.
 *
 * BUILD TYPE BEHAVIOR:
 *   - Debug:      Assertions ACTIVE (no NDEBUG defined)
 *   - Integration: Assertions ACTIVE (LLAMINAR_ENABLE_ASSERTIONS defined)
 *   - Release:    Assertions COMPILED OUT (NDEBUG, no LLAMINAR_ENABLE_ASSERTIONS)
 *   - E2ERelease: Assertions COMPILED OUT (NDEBUG, no LLAMINAR_ENABLE_ASSERTIONS)
 *
 * ASSERTION MACROS:
 *   - LLAMINAR_ASSERT(cond, msg)         - Basic condition check
 *   - LLAMINAR_ASSERT_NOT_NULL(ptr, name) - Null pointer check
 *   - LLAMINAR_ASSERTF(cond, msg_stream) - Formatted message
 *   - LLAMINAR_ASSERT_CAST(result, type, desc) - dynamic_cast validation
 *   - LLAMINAR_UNREACHABLE(msg_stream)   - Unreachable code marker
 *   - LLAMINAR_SNAPSHOT_ASSERT*          - Snapshot-only assertions
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#include "Logger.h"
#include <stdexcept>
#include <sstream>
#include <cstdlib>

// =============================================================================
// ASSERTION ENABLE LOGIC
// =============================================================================
// Assertions are active when:
//   1. NDEBUG is NOT defined (Debug builds), OR
//   2. LLAMINAR_ENABLE_ASSERTIONS is defined (Integration builds)
//
// This allows Integration builds to have NDEBUG (for stdlib optimizations)
// while still running our assertions for testing.
// =============================================================================

#if !defined(NDEBUG) || defined(LLAMINAR_ENABLE_ASSERTIONS)
#define LLAMINAR_ASSERTIONS_ACTIVE 1
#else
#define LLAMINAR_ASSERTIONS_ACTIVE 0
#endif

namespace llaminar2
{

    /**
     * @brief Exception thrown when a runtime assertion fails
     */
    class AssertionError : public std::runtime_error
    {
    public:
        AssertionError(const std::string &msg, const char *file, int line, const char *func)
            : std::runtime_error(formatMessage(msg, file, line, func)),
              file_(file), line_(line), func_(func) {}

        const char *file() const { return file_; }
        int line() const { return line_; }
        const char *func() const { return func_; }

    private:
        static std::string formatMessage(const std::string &msg, const char *file, int line, const char *func)
        {
            std::ostringstream oss;
            oss << "\n"
                << "╔══════════════════════════════════════════════════════════════════╗\n"
                << "║                     ASSERTION FAILED                              ║\n"
                << "╠══════════════════════════════════════════════════════════════════╣\n"
                << "║ " << msg << "\n"
                << "║\n"
                << "║ Location: " << file << ":" << line << "\n"
                << "║ Function: " << func << "\n"
                << "╚══════════════════════════════════════════════════════════════════╝\n";
            return oss.str();
        }

        const char *file_;
        int line_;
        const char *func_;
    };

    /**
     * @brief Helper to throw assertion errors with context
     */
    [[noreturn]] inline void throwAssertionError(const std::string &msg,
                                                 const char *file, int line, const char *func)
    {
        LOG_ERROR("ASSERTION FAILED: " << msg << " at " << file << ":" << line);
        throw AssertionError(msg, file, line, func);
    }

} // namespace llaminar2

// =============================================================================
// ASSERTION MACROS
// =============================================================================

#if LLAMINAR_ASSERTIONS_ACTIVE

/**
 * @brief Assert that a condition is true, throw if false
 * @param cond The condition to check
 * @param msg Error message if assertion fails
 *
 * Usage:
 *   LLAMINAR_ASSERT(ptr != nullptr, "Expected non-null pointer for context_snapshot");
 *   LLAMINAR_ASSERT(size > 0, "Buffer size must be positive");
 */
#define LLAMINAR_ASSERT(cond, msg)                                               \
    do                                                                           \
    {                                                                            \
        if (!(cond))                                                             \
        {                                                                        \
            ::llaminar2::throwAssertionError(msg, __FILE__, __LINE__, __func__); \
        }                                                                        \
    } while (0)

/**
 * @brief Assert that a pointer is not null
 * @param ptr The pointer to check
 * @param name Human-readable name for the pointer
 *
 * Usage:
 *   LLAMINAR_ASSERT_NOT_NULL(context_snapshot, "context_snapshot buffer");
 */
#define LLAMINAR_ASSERT_NOT_NULL(ptr, name)                                  \
    do                                                                       \
    {                                                                        \
        if ((ptr) == nullptr)                                                \
        {                                                                    \
            std::ostringstream _oss;                                         \
            _oss << "Expected non-null: " << name;                           \
            ::llaminar2::throwAssertionError(_oss.str(), __FILE__, __LINE__, \
                                             __func__);                      \
        }                                                                    \
    } while (0)

/**
 * @brief Assert with formatted message
 * @param cond The condition to check
 *
 * Usage:
 *   LLAMINAR_ASSERTF(idx < size, "Index " << idx << " out of bounds (size=" << size << ")");
 */
#define LLAMINAR_ASSERTF(cond, msg_stream)                                   \
    do                                                                       \
    {                                                                        \
        if (!(cond))                                                         \
        {                                                                    \
            std::ostringstream _oss;                                         \
            _oss << msg_stream;                                              \
            ::llaminar2::throwAssertionError(_oss.str(), __FILE__, __LINE__, \
                                             __func__);                      \
        }                                                                    \
    } while (0)

/**
 * @brief Unconditionally fail with a message (for unreachable code paths)
 *
 * NOTE: LLAMINAR_UNREACHABLE is ALWAYS active (not compiled out in Release)
 * because hitting unreachable code is a critical error that should never be silent.
 *
 * Usage:
 *   default:
 *       LLAMINAR_UNREACHABLE("Unknown tensor type: " << type);
 */
#define LLAMINAR_UNREACHABLE(msg_stream)                                 \
    do                                                                   \
    {                                                                    \
        std::ostringstream _oss;                                         \
        _oss << "Unreachable code reached: " << msg_stream;              \
        ::llaminar2::throwAssertionError(_oss.str(), __FILE__, __LINE__, \
                                         __func__);                      \
    } while (0)

/**
 * @brief Assert that a dynamic_cast succeeded
 * @param result The result of dynamic_cast
 * @param expected_type Human-readable name of expected type
 * @param actual_desc Description of what was being cast
 *
 * Usage:
 *   auto* fp32 = dynamic_cast<FP32Tensor*>(tensor);
 *   LLAMINAR_ASSERT_CAST(fp32, "FP32Tensor", "context_snapshot tensor");
 */
#define LLAMINAR_ASSERT_CAST(result, expected_type, actual_desc)             \
    do                                                                       \
    {                                                                        \
        if ((result) == nullptr)                                             \
        {                                                                    \
            std::ostringstream _oss;                                         \
            _oss << "dynamic_cast failed: expected " << expected_type        \
                 << " but " << actual_desc << " has incompatible type";      \
            ::llaminar2::throwAssertionError(_oss.str(), __FILE__, __LINE__, \
                                             __func__);                      \
        }                                                                    \
    } while (0)

#else // LLAMINAR_ASSERTIONS_ACTIVE == 0 (Release builds)

// Assertions compile to no-ops in Release builds for zero overhead
#define LLAMINAR_ASSERT(cond, msg) ((void)0)
#define LLAMINAR_ASSERT_NOT_NULL(ptr, name) ((void)0)
#define LLAMINAR_ASSERTF(cond, msg_stream) ((void)0)
#define LLAMINAR_ASSERT_CAST(result, expected_type, actual_desc) ((void)0)

// LLAMINAR_UNREACHABLE is ALWAYS active - hitting unreachable code is critical
#define LLAMINAR_UNREACHABLE(msg_stream)                                 \
    do                                                                   \
    {                                                                    \
        std::ostringstream _oss;                                         \
        _oss << "Unreachable code reached: " << msg_stream;              \
        ::llaminar2::throwAssertionError(_oss.str(), __FILE__, __LINE__, \
                                         __func__);                      \
    } while (0)

#endif // LLAMINAR_ASSERTIONS_ACTIVE

// =============================================================================
// SNAPSHOT-ONLY ASSERTIONS
// =============================================================================
// These assertions are ONLY active when ENABLE_PIPELINE_SNAPSHOTS is defined.
// Use for snapshot-specific invariants that don't need to be checked in
// non-snapshot builds (e.g., "context_snapshot buffer must not be null").
// =============================================================================

/**
 * @brief Assert during snapshot capture - only checked when ENABLE_PIPELINE_SNAPSHOTS is defined
 *
 * Use this for snapshot-specific assertions that should only fire when
 * snapshot capture is explicitly enabled. These delegate to LLAMINAR_ASSERT
 * when active, so they inherit the Debug/Integration vs Release behavior.
 *
 * Usage:
 *   LLAMINAR_SNAPSHOT_ASSERT(context_snapshot != nullptr, "context_snapshot required for E2E testing");
 */
#ifdef ENABLE_PIPELINE_SNAPSHOTS
#define LLAMINAR_SNAPSHOT_ASSERT(cond, msg) LLAMINAR_ASSERT(cond, msg)
#define LLAMINAR_SNAPSHOT_ASSERT_NOT_NULL(ptr, name) LLAMINAR_ASSERT_NOT_NULL(ptr, name)
#else
#define LLAMINAR_SNAPSHOT_ASSERT(cond, msg) ((void)0)
#define LLAMINAR_SNAPSHOT_ASSERT_NOT_NULL(ptr, name) ((void)0)
#endif
