/**
 * @file ReadAheadParticipantContract.h
 * @brief Authenticate a complete one-participant assignment without null-context assumptions.
 *
 * Degree-one rank-local contexts can carry the exact unsharded model. Their
 * original device and complete head/FFN/vocabulary assignment must agree;
 * merely seeing a non-null TP pointer is neither rejection nor authorization.
 */
#pragma once
#include "backends/GlobalDeviceAddress.h"
#include "config/TensorParallelConfig.h"
#include <span>
#include <stdexcept>

namespace llaminar2 {
/** @brief Full original model dimensions, independent of any participant shard. */
struct ReadAheadFullGeometry {int query_heads,kv_heads,ffn_width,vocabulary;};
/**
 * @brief Validate source-owned local addresses and complete original assignment.
 * @param device Exact participant from the admitted graph.
 * @param addresses Empty without a context, otherwise the original rank-local domain's devices.
 * @param assignments Empty without a sharding declaration, otherwise its original assignments.
 * @param full Original complete model dimensions.
 * @throws std::invalid_argument For multiple/foreign participants or partial/shifted ownership.
 * @note The caller authenticates the original context is rank-local with degree one;
 * this function never manufactures a context, assignment or collective capability.
 */
inline void requireReadAheadSingleParticipantShape(DeviceId device,
    std::span<const GlobalDeviceAddress> addresses,std::span<const DeviceShardingAssignment> assignments,
    ReadAheadFullGeometry full)
{
    if(!device.is_gpu() || full.query_heads<=0 || full.kv_heads<=0 || full.ffn_width<=0 || full.vocabulary<=0 ||
       addresses.size()>1 || assignments.size()>1)
        throw std::invalid_argument("Read-ahead requires one complete GPU participant");
    if(!addresses.empty() && (addresses.front().device_type!=device.type || addresses.front().device_ordinal!=device.ordinal))
        throw std::invalid_argument("Read-ahead rank-local context names a different original device");
    if(!assignments.empty())
    {
        const auto& a=assignments.front();
        if(a.device!=device || a.local_rank!=0 || a.work_fraction!=1.0f ||
           a.head_start!=0 || a.head_count!=full.query_heads || a.kv_head_start!=0 || a.kv_head_count!=full.kv_heads ||
           a.d_ff_start!=0 || a.d_ff_count!=full.ffn_width || a.vocab_start!=0 || a.vocab_count!=full.vocabulary)
            throw std::invalid_argument("Read-ahead degree-one assignment is not the complete original model");
    }
}
}
