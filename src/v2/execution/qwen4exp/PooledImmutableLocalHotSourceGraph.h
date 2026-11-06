/**
 * @file PooledImmutableLocalHotSourceGraph.h
 * @brief One original combined source-body lowering, without model-parent authority.
 *
 * The body borrows actual prepared source/local roots and returns its existing
 * native retirement inventory. It does not install a SourcePlan, issue a source
 * seal, choose a serving envelope or create/submit an executable. The original
 * retained model owner performs those distinct operations after composition.
 */
#pragma once
#include "PooledImmutableLocalHotGraph.h"
#include "PooledImmutableDirectedLookupGraph.h"
#include <span>
namespace llaminar2 {
/** @brief Actual graph observations; neither member grants source/retirement capability. */
struct PooledImmutableLocalHotSourceBody final {
    PooledImmutableLocalHotGraphEndpoints local;
    PooledImmutableSourceRetirementCompletion pool_retirement;
};
/**
 * @brief Append the same original compact source phases used by the retained model builder.
 * @param graph Actual participant-local graph receiving the nodes.
 * @param arena Original shared arena and controller/input/output roles.
 * @param transport Original directed source owner; never a fabricated source-to-self channel.
 * @param locals Complete ordered actual local-root roster, bound to this same original source.
 * @param poolCode Original native pooled code. @param local Actual coordinator local-cache root.
 * @param localCode Original native local-cache code. @param tokens Full original token extent.
 * @param ids Actual hash producer. @param count Optional original token-count producer.
 * @param control Same original request controller. @param output Sole model-output span.
 * @param identity Immutable node prefix. @param outputIdentity Actual LocalRead node name.
 * @return Complete main/auxiliary observations and original pool retirement descriptor.
 * @throws std::exception For foreign, incomplete or mismatched original roots before use.
 * @note The caller must retain every original root through native retirement and
 * independently provide original model SourcePlan/range attachment when serving.
 */
PooledImmutableLocalHotSourceBody appendPooledImmutableLocalHotSourceBody(
    ComputeGraph&,BufferArena&,
    std::shared_ptr<const qwen4exp::PooledImmutableDirectedTransport>,
    std::span<const std::shared_ptr<qwen4exp::PooledImmutableLocalHotPreparedStorage>>,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels>,
    std::shared_ptr<qwen4exp::PooledImmutableLocalHotPreparedStorage>,
    std::shared_ptr<const qwen4exp::IPooledImmutableLocalHotKernels>,
    DeviceRowRange,qwen4exp::PooledNativeTensorSpan,std::optional<qwen4exp::PooledNativeTensorSpan>,
    DeviceGenerationControlBinding,qwen4exp::PooledNativeTensorSpan,const std::string&,const std::string&);
}
