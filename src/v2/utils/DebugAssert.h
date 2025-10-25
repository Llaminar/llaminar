/**
 * @file DebugAssert.h
 * @brief Debug-only assertions that compile out in release builds
 *
 * Provides assertion macros for validation that:
 * - Enable comprehensive checks in debug builds
 * - Compile to no-ops in release builds (zero overhead)
 * - Use LOG_ERROR for failed assertions before aborting
 *
 * @author David Sanftenberg
 */

#pragma once

#include "Logger.h"
#include <cstdlib>

namespace llaminar2
{

#ifndef NDEBUG
// Debug build: Full assertion checks with logging

/**
 * @brief Assert condition with message
 * @param cond Condition to check
 * @param msg Error message (can use stream operators)
 *
 * Example: DEBUG_ASSERT(ptr != nullptr, "Null pointer: " << name);
 */
#define DEBUG_ASSERT(cond, msg)                                                                \
    do                                                                                         \
    {                                                                                          \
        if (!(cond))                                                                           \
        {                                                                                      \
            LOG_ERROR("[ASSERTION FAILED] " << msg                                             \
                                            << " (at " << __FILE__ << ":" << __LINE__ << ")"); \
            std::abort();                                                                      \
        }                                                                                      \
    } while (0)

/**
 * @brief Assert equality with detailed error
 * @param actual Actual value
 * @param expected Expected value
 * @param msg Context message
 *
 * Example: DEBUG_ASSERT_EQ(tensor->shape()[0], seq_len, "Sequence length mismatch");
 */
#define DEBUG_ASSERT_EQ(actual, expected, msg)                                                 \
    do                                                                                         \
    {                                                                                          \
        if ((actual) != (expected))                                                            \
        {                                                                                      \
            LOG_ERROR("[ASSERTION FAILED] " << msg                                             \
                                            << ": expected " << (expected)                     \
                                            << ", got " << (actual)                            \
                                            << " (at " << __FILE__ << ":" << __LINE__ << ")"); \
            std::abort();                                                                      \
        }                                                                                      \
    } while (0)

/**
 * @brief Assert non-null pointer
 * @param ptr Pointer to check
 * @param msg Context message
 *
 * Example: DEBUG_ASSERT_NOT_NULL(tensor.get(), "Failed to create tensor");
 */
#define DEBUG_ASSERT_NOT_NULL(ptr, msg)                                                                      \
    do                                                                                                       \
    {                                                                                                        \
        if ((ptr) == nullptr)                                                                                \
        {                                                                                                    \
            LOG_ERROR("[ASSERTION FAILED] Null pointer: " << msg                                             \
                                                          << " (at " << __FILE__ << ":" << __LINE__ << ")"); \
            std::abort();                                                                                    \
        }                                                                                                    \
    } while (0)

/**
 * @brief Assert range check
 * @param value Value to check
 * @param min Minimum (inclusive)
 * @param max Maximum (exclusive)
 * @param msg Context message
 *
 * Example: DEBUG_ASSERT_RANGE(layer_idx, 0, n_layers_, "Invalid layer index");
 */
#define DEBUG_ASSERT_RANGE(value, min, max, msg)                                                  \
    do                                                                                            \
    {                                                                                             \
        if ((value) < (min) || (value) >= (max))                                                  \
        {                                                                                         \
            LOG_ERROR("[ASSERTION FAILED] " << msg                                                \
                                            << ": value " << (value)                              \
                                            << " out of range [" << (min) << ", " << (max) << ")" \
                                            << " (at " << __FILE__ << ":" << __LINE__ << ")");    \
            std::abort();                                                                         \
        }                                                                                         \
    } while (0)

/**
 * @brief Unconditional assertion failure
 * @param msg Error message
 *
 * Example: DEBUG_FAIL("Unreachable code path");
 */
#define DEBUG_FAIL(msg)                                                                    \
    do                                                                                     \
    {                                                                                      \
        LOG_ERROR("[ASSERTION FAILED] " << msg                                             \
                                        << " (at " << __FILE__ << ":" << __LINE__ << ")"); \
        std::abort();                                                                      \
    } while (0)

#else
    // Release build: All assertions compile to no-ops

#define DEBUG_ASSERT(cond, msg) ((void)0)
#define DEBUG_ASSERT_EQ(actual, expected, msg) ((void)0)
#define DEBUG_ASSERT_NOT_NULL(ptr, msg) ((void)0)
#define DEBUG_ASSERT_RANGE(value, min, max, msg) ((void)0)
#define DEBUG_FAIL(msg) ((void)0)

#endif // NDEBUG

} // namespace llaminar2
