/** @file LocalHotRecordedResponse.h
 * @brief Immutable replay inputs for actual local-cache kernels, never an L2/Rank completion proof.
 */
#pragma once
#include "kernels/qwen4exp/PooledImmutableLocalHotOffsets.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotState.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotTransportOps.h"
namespace llaminar2::test::local_hot_recorded {
/** @brief Two explicit diagnostic input producers, separate from production cache operations. */
enum class Action { ForwardOriginalAdmission,RecordedResponse };
/** @brief Original prepared aliases and frozen native rows; no mutable state is mirrored on the host. */
struct Args {
    std::byte* frame=nullptr;
    qwen4exp::PooledImmutableLocalHotFrameOffsets offsets;
    qwen4exp::local_hot_transport::ConsumerGeometry consumer{};
    qwen4exp::local_hot_transport::SourceAdmissionHeader* admission=nullptr;
    qwen4exp::local_hot_transport::RequestHeader* request=nullptr;
    qwen4exp::local_hot_transport::RequestLaneIdentity* lane=nullptr;
    std::uint64_t* key_bytes=nullptr;
    const std::uint64_t* original_admission=nullptr;
    const std::uint64_t* original_invocation=nullptr;
    std::uint64_t table=0,source_generation=0;
    const std::uint64_t* recorded_keys=nullptr;
    const std::byte* recorded_rows=nullptr;
    std::uint32_t recorded_count=0;
    std::uint64_t row_bytes=0;
};
/** @brief Submit one diagnostic producer on the same original exact stream. */
void launchCUDA(Action,const Args&,void*);
/** @brief Symmetric HIP diagnostic input producer; this never grants source completion authority. */
void launchROCm(Action,const Args&,void*);
}
