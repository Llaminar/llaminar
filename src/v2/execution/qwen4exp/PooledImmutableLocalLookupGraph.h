/**
 * @file PooledImmutableLocalLookupGraph.h
 * @brief Declarative complete local pooled lookup wiring through the existing real-provider boundary.
 *
 * Native owner and coordinator phases retain their original arbitrary-owner
 * interfaces. This builder admits only the explicitly materialized local edge;
 * absent directed transport is a hard error, never a topology capability.
 * Source-success and failure cleanup branches are fixed captured nodes.
 */
#pragma once
#include "execution/compute_stages/stages/qwen4exp/PooledImmutableSourcePhaseStage.h"
#include "execution/compute_stages/stages/qwen4exp/PooledImmutableNativePhaseStage.h"
#include "execution/local_execution/graph/ComputeGraph.h"
namespace llaminar2 {
/** @brief Exact endpoints for actual hash/control producer and next model consumer dependencies. */
struct PooledImmutableLookupGraphEndpoints {std::string first,last;};
/**
 * @brief Append a complete captured local transaction with actual provider/ACK/Ready/read/release ownership.
 * @param graph Existing participant-local model graph, retaining its actual source boundary.
 * @param arena Original already-bound owner/coordinator/source/output/controller roots.
 * @param boundary Original real-provider pooled arm, retained through terminal and final readers.
 * @param code Stateless owner/coordinator bundle for the exact native backend.
 * @param original Original complete counted IDs and actual width, preserving stable coalescing/permutation.
 * @param begin Actual ID/count/admission tensor roles; no host values are copied.
 * @param control Same canonical request controller borrowed by every source branch.
 * @param output Exact prepared original-order FP32 output span.
 * @param identity Unique stable node prefix; no execution policy is selected by strings.
 * @return Actual first/last graph nodes for explicit external producer/consumer dependencies.
 * @throws std::exception For missing directed topology, foreign roots or capture-time construction.
 */
PooledImmutableLookupGraphEndpoints appendPooledImmutableLocalLookupGraph(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary> boundary,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,qwen4exp::pooled_native::OriginalRows original,
    qwen4exp::PooledNativeOperandSpans begin,DeviceGenerationControlBinding control,
    qwen4exp::PooledNativeTensorSpan output,const std::string& identity);
}
