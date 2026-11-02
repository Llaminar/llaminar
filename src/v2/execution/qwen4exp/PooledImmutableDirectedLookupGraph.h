/**
 * @file PooledImmutableDirectedLookupGraph.h
 * @brief Declarative source and owner programs over original directed immutable lookup channels.
 *
 * These builders append participant-local nodes only. The original model engine
 * retains its complete source capture; explicit retained participant ownership
 * submits the separate owner programs. No builder launches or synchronizes work.
 */
#pragma once
#include "PooledImmutableDirectedTransport.h"
#include "PooledImmutableDirectedOwnerProgram.h"
#include "PooledImmutableLocalLookupGraph.h"
namespace llaminar2 {
/** @brief Actual source terminal descriptors, retained for native reset and reader lifetime authentication. */
struct PooledImmutableDirectedLookupEndpoints final {
    PooledImmutableLookupGraphEndpoints source;
    PooledImmutableRetirementCompletion retirement;
};
/**
 * @brief Append one original complete mixed source transaction without embedding a remote graph.
 * @param graph Actual participant-local model fragment receiving the source program.
 * @param arena Original source arena; prepared native roots remain independently owned.
 * @param boundary Original model-owned immutable source and admitted worker cohort.
 * @param transport Actual directed two-channel ownership; no transport-to-self exists.
 * @param code Original source-native kernel bundle.
 * @param original Actual counted original sixteen-head query rows.
 * @param begin Exact current Hash-ID/count/admission operands; no host mirror is formed.
 * @param control Original device generation controller.
 * @param output Original model embedding bank and exact capacity.
 * @param identity Unique source fragment namespace.
 * @param output_identity Original semantic snapshot/output identity.
 * @return Actual source and all-owner retirement frontiers.
 */
PooledImmutableDirectedLookupEndpoints appendPooledImmutableDirectedLookupGraph(
    ComputeGraph& graph,BufferArena& arena,std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary> boundary,
    std::shared_ptr<const qwen4exp::PooledImmutableDirectedTransport> transport,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,
    qwen4exp::pooled_native::OriginalRows original,qwen4exp::PooledNativeOperandSpans begin,
    DeviceGenerationControlBinding control,qwen4exp::PooledNativeTensorSpan output,
    const std::string& identity,const std::string& output_identity);
/**
 * @brief Append the addressed original owner program, independent of source model variants.
 * @param graph Exclusive participant-local remote graph.
 * @param arena Original remote arena; every operand retains its actual prepared root.
 * @param transport Immutable original addressed channel chain.
 * @param code Original addressed owner's native kernel bundle.
 * @param program Original pre-admission source-authored remote dispatch/geometry declaration.
 * @param identity Unique local program namespace.
 * @return Actual original terminal retirement send; it is not native stream completion.
 */
PooledPinParticipantNodes appendPooledImmutableDirectedOwnerGraph(ComputeGraph& graph,BufferArena& arena,
    const qwen4exp::PooledImmutableRemoteTransport& transport,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,const PooledRemoteGraphProgram& program,const std::string& identity);
}
