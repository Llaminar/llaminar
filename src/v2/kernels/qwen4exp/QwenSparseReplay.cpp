/**
 * @file QwenSparseReplay.cpp
 * @brief Static and CPU execution-owner publication of sparse transaction metadata.
 *
 * Physical origin and absolute position are independent values. Rejected
 * candidates never advance this immutable transaction snapshot; the canonical
 * generation controller owns accepted-cache publication and admission bounds.
 */
#include "QwenSparseReplay.h"
#include <limits>
#include <stdexcept>

namespace llaminar2::qwen4exp
{
void validateQwenSparsePrepareReplay(const QwenSparsePrepareReplay &p)
{
    if(!p.sequence.cursor || !p.sequence.cached_tokens ||
        !p.replay || p.sequence.capacity<=0 || p.row_capacity<=0 ||
        (p.sequence.encoding!=QwenSparseRingCursor::OldestRow && p.sequence.encoding!=QwenSparseRingCursor::NextWriteRow))
        throw std::invalid_argument("QSA replay publication requires exact preappend cache/request pointers and admitted geometry");
}
void cpuQwenSparsePrepareReplay(const QwenSparsePrepareReplay &p)
{
    validateQwenSparsePrepareReplay(p);
    const int cursor=*p.sequence.cursor,prefix=*p.sequence.cached_tokens,live=p.live_rows ? *p.live_rows : p.row_capacity,capacity=p.sequence.capacity;
    const int firstPosition=p.position_ids ? p.position_ids[0] : p.first_position;
    if(cursor<0 || cursor>=capacity || prefix<0 || prefix>capacity || live<0 || live>p.row_capacity ||
        std::int64_t(prefix)+live>capacity || firstPosition<prefix ||
        std::int64_t(firstPosition)+live>std::numeric_limits<int>::max())
        throw std::invalid_argument("QSA replay publication exceeds admitted context or has invalid canonical position/cursor state");
    const int origin=p.sequence.encoding==QwenSparseRingCursor::OldestRow ? cursor :
        cursor>=prefix ? cursor-prefix : cursor-prefix+capacity;
    for(int row=0;row<live;++row)
        if(p.position_ids && p.position_ids[row]!=firstPosition+row)
            throw std::invalid_argument("QSA replay publication requires exact contiguous canonical absolute positions");
    *p.replay={prefix,live,firstPosition-prefix};
    if(p.row_parameters)for(int row=0;row<p.row_capacity;++row)
        p.row_parameters[row]=row<live ? attention::AttentionDeviceParams{prefix+row+1,capacity,firstPosition+row,0,origin,capacity} :
            attention::AttentionDeviceParams{};
}
} // namespace llaminar2::qwen4exp
