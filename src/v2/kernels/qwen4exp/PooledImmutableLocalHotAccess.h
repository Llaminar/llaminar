/**
 * @file PooledImmutableLocalHotAccess.h
 * @brief Immutable invocation intent, distinct from actual demand consumption.
 *
 * Neither intent grants policy credit. The original model-consumer endpoint
 * must consume the same frame once before replacement policy or fills run.
 * Future read-ahead may instead discard its one bounded response/pin window.
 */
#pragma once
#include "AdaptiveImmutableRowCacheState.h"
namespace llaminar2::qwen4exp {
/** @brief Why the original graph requests rows; neither value means consumed. */
enum class PooledImmutableLocalHotAccessIntent : std::uint32_t {
    CurrentInvocation, ///< Early or consumer-scheduled demand for this real invocation.
    FutureReadAhead ///< Not-yet-consumed lookahead, held only in the admitted frame.
};
/** @return Whether the immutable graph named one supported access intent. */
QFN_ADAPTIVE_HD constexpr bool validLocalHotAccessIntent(PooledImmutableLocalHotAccessIntent intent)
{
    return intent == PooledImmutableLocalHotAccessIntent::CurrentInvocation ||
        intent == PooledImmutableLocalHotAccessIntent::FutureReadAhead;
}
} // namespace llaminar2::qwen4exp
