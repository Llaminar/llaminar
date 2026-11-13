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
/** @brief Real future producer; local pins/response survive its completed source/ACK terminal. */
struct PooledImmutableLocalHotFutureSourceBody final {
    std::string probe,acquired,transport_retired;
    PooledImmutableSourceRetirementCompletion pool_retirement;
    RetainedGraphParticipantAuxiliaryRange auxiliary;
};
/**
 * @brief Append genuine lookup-only source delivery without a later model-read wait.
 * @param graph Actual participant-local graph. @param arena Same source arena.
 * @param transport Original directed source and complete local-root roster.
 * @param locals Original ordered consumer roots. @param poolCode Same source native code.
 * @param local Same coordinator local root. @param localCode Same local native code.
 * @param tokens Actual full token-count producer. @param ids Already-authored original ordered hashes.
 * @param count Exact optional original count. @param control Same original controller row.
 * @param identity Unique original stage namespace.
 * @return Probe/acquisition and actual pool/source retirement; no local read/consume/retire.
 * @note This low-level component does not mint model token/history publication.
 * The model owner must independently order its genuine history commit and
 * previous local Retire before the actual future Hash/Probe.
 */
PooledImmutableLocalHotFutureSourceBody appendPooledImmutableLocalHotFutureSourceBody(
    ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<const qwen4exp::PooledImmutableDirectedTransport> transport,
    std::span<const std::shared_ptr<qwen4exp::PooledImmutableLocalHotPreparedStorage>> locals,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> poolCode,
    std::shared_ptr<qwen4exp::PooledImmutableLocalHotPreparedStorage> local,
    std::shared_ptr<const qwen4exp::IPooledImmutableLocalHotKernels> localCode,
    DeviceRowRange tokens,qwen4exp::PooledNativeTensorSpan ids,
    std::optional<qwen4exp::PooledNativeTensorSpan> count,DeviceGenerationControlBinding control,
    const std::string& identity);

}
