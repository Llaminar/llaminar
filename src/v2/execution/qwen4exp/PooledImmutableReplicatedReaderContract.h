/**
 * @file PooledImmutableReplicatedReaderContract.h
 * @brief Exact source-frame dependency for replicated result metadata stages.
 *
 * Header creation reads the original coordinator frame even without a payload
 * operand. Declaring that same arena owner lets retained child capture use its
 * real producer edge instead of importing an unrelated prior completion event.
 */
#pragma once
#include "kernels/qwen4exp/PooledImmutableReplicatedReaderKernels.h"
#include "memory/BufferArena.h"
#include "memory/StageBufferContract.h"
#include <optional>
#include <stdexcept>
namespace llaminar2::qwen4exp::detail {
/** @param arena Original participant arena. @param frame Original coordinator frame.
 * @return Exact canonical frame. @throws std::logic_error For absent or substituted ownership. */
inline TensorBase* checkedReplicatedReaderSourceFrame(BufferArena& arena,TensorBase* frame){
    if(!frame || arena.getTensor(BufferId::QFN_POOLED_COORDINATOR_FRAME)!=frame)
        throw std::logic_error("Shared PLE source metadata lost its original canonical coordinator frame");
    return frame;
}
/** @param action Original source or reader metadata action. @param output Original result buffer.
 * @param count Optional original live-row count buffer. @return Complete metadata stage arena contract. */
inline StageBufferContract replicatedReaderPhaseContract(replicated_reader::Action action,
    BufferId output,std::optional<BufferId> count){
    auto contract=StageBufferContract::build();
    contract.addPreallocatedInOut(BufferId::MTP_GENERATION_CONTROL,"INT32");
    if(action==replicated_reader::Action::PrepareResultHeader || action==replicated_reader::Action::PrepareResult ||
       action==replicated_reader::Action::ConfirmAck)
        contract.addInput(BufferId::QFN_POOLED_COORDINATOR_FRAME,"INT8");
    if(action==replicated_reader::Action::AcknowledgeResult)contract.addOutput(output,"FP32");
    if(count && *count!=BufferId::MTP_GENERATION_CONTROL)contract.addInput(*count,"INT32");
    return contract;
}
}
