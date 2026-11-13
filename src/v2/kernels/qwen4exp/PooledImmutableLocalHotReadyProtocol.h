/**
 * @file PooledImmutableLocalHotReadyProtocol.h
 * @brief Single-frame ready frontiers with an absorbing native failure value.
 *
 * A probe may reset only a physically retired previous frame. The caller proves
 * that frame lifecycle through Operations::begin; this stateless helper then
 * prevents a concurrent cancellation from being overwritten by normal reset.
 */
#pragma once
#include "AdaptiveImmutableRowCacheState.h"
#include <cstdint>
namespace llaminar2::qwen4exp::local_hot {
/**
 * @brief Authenticate the original forward controller before or after response admission.
 * Ok0/error0 is the canonical prefill state; Ok1/error0 is native generation.
 * A sticky error or malformed Ok value never authorizes local-cache work.
 */
QFN_ADAPTIVE_HD constexpr bool validForwardController(int ok, int error)
{
    return error == 0 && (ok == 0 || ok == 1);
}
inline constexpr std::uint64_t ReadyEmpty = 0, ReadyData = 1, ReadyReadComplete = 2;
inline constexpr std::uint64_t ReadyFailure = UINT64_MAX;
/**
 * @brief Reset one valid retired frontier with one acquire and one compare/exchange.
 * @param acquire Original system-acquire operation, invoked exactly once.
 * @param compare Original system compare/exchange, invoked once only for a valid frontier.
 * @return False for failure, malformed frontier or concurrent cancellation; no retry occurs.
 */
template<class Acquire, class Compare>
QFN_ADAPTIVE_HD inline bool resetReadyForProbe(Acquire acquire, Compare compare)
{
    const std::uint64_t previous = acquire();
    return previous <= ReadyReadComplete && compare(previous, ReadyEmpty);
}
} // namespace llaminar2::qwen4exp::local_hot
