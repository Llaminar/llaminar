/**
 * @file PooledImmutableDirectedOperation.h
 * @brief Cold immutable descriptions used to seal the actual directed owner recipe.
 *
 * Descriptions grant no transfer, capture or admission authority. The original
 * typed bindings still own all pointers, streams, generations and live counts.
 * The directed builder compares these actual operations with its pure declared
 * recipe before native recording, preventing estimator/launcher drift.
 */
#pragma once
#include "transfer/CapturedTransferChannel.h"
#include "transfer/CapturedTransferKernelPlan.h"
#include <optional>
#include <stdexcept>
namespace llaminar2 {class IComputeStage;}
namespace llaminar2::qwen4exp {
/** @brief Closed semantic wire families in their original participant-local program. */
enum class PooledImmutableDirectedWireFamily:std::uint32_t {Pin,PinReturn,Free,Victim,Source,Publication,Read,Retirement};
/** @brief Cold-selected transport; native library recipes cannot use a flat kernel count. */
enum class PooledImmutableDirectedTransportKind : std::uint8_t {CapturedChannel,NativeCollective};
/** @brief One actual capture-frozen endpoint, excluding physical pointers and mutable counts. */
struct PooledImmutableDirectedTransferOperation final {
    PooledImmutableDirectedWireFamily family;
    std::uint32_t role;
    CapturedTransferEndpoint endpoint;
    std::uint64_t maximumBytes;
    CapturedTransferExtentSource extent;
    bool sourceSlice;
    PooledImmutableDirectedTransportKind transport=PooledImmutableDirectedTransportKind::CapturedChannel;
    /** @return Native invocations from the very selector used by CUDA and HIP. */
    unsigned kernelCount()const{if(transport!=PooledImmutableDirectedTransportKind::CapturedChannel)throw std::logic_error("Native collective has no captured-channel flat kernel recipe");return capturedTransferKernelPlan(maximumBytes)==CapturedTransferKernelPlan::SingleBlock?1u:2u;}
    bool operator==(const PooledImmutableDirectedTransferOperation&)const=default;
};
}
namespace llaminar2 {
/** @return Metadata only for this translation unit's exact original typed Pin endpoint stages. */
std::optional<qwen4exp::PooledImmutableDirectedTransferOperation> describePooledPinTransfer(const IComputeStage&);
/** @return Metadata only for this translation unit's exact original typed Free endpoint stages. */
std::optional<qwen4exp::PooledImmutableDirectedTransferOperation> describePooledFreeTransfer(const IComputeStage&);
/** @return Metadata only for this translation unit's exact original typed Victim endpoint stages. */
std::optional<qwen4exp::PooledImmutableDirectedTransferOperation> describePooledVictimTransfer(const IComputeStage&);
/** @return Metadata only for this translation unit's exact original typed Source endpoint stages. */
std::optional<qwen4exp::PooledImmutableDirectedTransferOperation> describePooledSourceTransfer(const IComputeStage&);
/** @return Metadata only for this translation unit's exact original typed Publication endpoint stages. */
std::optional<qwen4exp::PooledImmutableDirectedTransferOperation> describePooledPublicationTransfer(const IComputeStage&);
/** @return Metadata only for this translation unit's exact original typed Read endpoint stages. */
std::optional<qwen4exp::PooledImmutableDirectedTransferOperation> describePooledReadTransfer(const IComputeStage&);
/** @return Metadata only for this translation unit's exact original typed Retirement endpoint stages. */
std::optional<qwen4exp::PooledImmutableDirectedTransferOperation> describePooledRetirementTransfer(const IComputeStage&);
}
