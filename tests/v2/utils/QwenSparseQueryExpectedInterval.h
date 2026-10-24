/**
 * @file QwenSparseQueryExpectedInterval.h
 * @brief Independent canonical-history oracle for original QSA query coordinates.
 *
 * This test consumer uses original submitted history, never a native work decision.
 * The pool/ranking equation is intentionally independent of the produced threshold.
 * It contains no execution pointers, payload access, or numerical acceptance.
 */
#pragma once
#include <cstddef>
#include <limits>
#include <optional>

namespace llaminar2::test::parity
{
/** @brief Original canonical request history before the observed Decode invocation. */
struct QwenSparseQueryHistory
{
    std::size_t promptRows=0; ///< Independently authenticated original prompt token count.
    std::size_t decodeRowsBefore=0; ///< Original Decode rows preceding this invocation, not accepted candidates.
    int liveRows=1; ///< Original submitted live width before any query-work selection.
    int sourceCapacity=0; ///< Context bound authenticated from the frozen admitted configuration.
};
/** @brief Exact expected request-local suffix in original row coordinates. */
struct QwenSparseQueryExpectedInterval
{
    int prefixRows=0,firstRow=0,endRow=0;
    bool operator==(const QwenSparseQueryExpectedInterval &) const=default;
};
/**
 * @brief Derive expected Q work independently from causal complete-pool demand.
 * @param history Original prompt/submitted Decode history and frozen context.
 * @return Original request-local interval, or absence for malformed history.
 * @details A row needs ranking exactly when its visible complete-pool count
 * exceeds512. The independent pool equation avoids using native first/end
 * or its threshold as the oracle. Grouped callers must supply their real prior
 * history; multiplying a serial step by the grouped width is not authentication.
 */
inline std::optional<QwenSparseQueryExpectedInterval> expectedQwenSparseQueryInterval(
    QwenSparseQueryHistory history) noexcept
{
    constexpr auto maximum=static_cast<std::size_t>(std::numeric_limits<int>::max());
    if(!history.promptRows || history.liveRows<0 || history.sourceCapacity<=0 ||
       history.promptRows>maximum || history.decodeRowsBefore>maximum-history.promptRows)
        return std::nullopt;
    const auto prefix=history.promptRows+history.decodeRowsBefore;
    if(history.liveRows>history.sourceCapacity ||
       prefix>static_cast<std::size_t>(history.sourceCapacity-history.liveRows))
        return std::nullopt;
    // The first visible length with more than512 complete four-token pools is
    // 4*(512+1). Subtracting one converts that length to its original row index.
    constexpr int firstRequiredLogicalRow=4*(512+1)-1;
    const auto missing=static_cast<long long>(firstRequiredLogicalRow)-static_cast<long long>(prefix);
    const int first=missing<=0?0:missing>=history.liveRows?history.liveRows:static_cast<int>(missing);
    return QwenSparseQueryExpectedInterval{static_cast<int>(prefix),first,history.liveRows};
}
} // namespace llaminar2::test::parity
