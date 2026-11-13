/**
 * @file QwenSparseObservationROCmAdapter.h
 * @brief Setup-only HIP graph identity and static-resource adapter for native events.
 *
 * Retains the original hipKernelNameRefByPtr identity path and queries only
 * captured native function attributes; graph replay and event intervals are separate.
 */
#pragma once
#include "QwenSparseObservationNativeMetadata.h"
namespace llaminar2::qwen4exp::QSA_SUITE::observation115
{
using ObservationNode=hipGraphNode_t;
using ObservationNodeType=hipGraphNodeType;
using ObservationKernelParams=hipKernelNodeParams;
inline constexpr const char* ObservationBackend="ROCm";
/**
 * @brief Read dependency identities using the HIP graph API.
 * @param node Borrowed node in the retained fixture graph.
 * @param parents Optional output array; null queries only its required capacity.
 * @param count In/out array capacity and returned dependency count.
 * @return Native status, checked by the caller before interpreting the graph.
 * @details This setup-only metadata query neither mutates nor submits the graph.
 */
inline hipError_t observationDependencies(ObservationNode node,ObservationNode* parents,std::size_t* count)
{
    return hipGraphNodeGetDependencies(node,parents,count);
}
/**
 * @brief Inspect one retained HIP kernel node without replay or a terminal join.
 * @param node Borrowed native node; its retained graph must remain live.
 * @param stream Exact fixture stream used to resolve the original symbol.
 * @return Captured identity, symbol, native static resources and actual captured block resource-based resident-block limit (theoretical occupancy).
 * @throws std::runtime_error For native metadata failure or an empty symbol.
 */
inline NativeKernelMetadata inspectObservationKernel(ObservationNode node,QSA_STREAM stream)
{
    hipKernelNodeParams params{};check(hipGraphKernelNodeGetParams(node,&params));
    const char* name=hipKernelNameRefByPtr(params.func,stream);
    if(!params.func||!name||!*name)throw std::runtime_error("Observation HIP node lacks its original native identity");
    hipFuncAttributes attributes{};check(hipFuncGetAttributes(&attributes,params.func));
    NativeKernelMetadata result{.identity=params.func,.name=name,.registers=attributes.numRegs,
        .shared_bytes=attributes.sharedSizeBytes,.local_bytes=attributes.localSizeBytes,.max_threads=attributes.maxThreadsPerBlock};
    // Occupancy is a resource limit for the captured block, not achieved usage.
    // Capture separately enforces each role's statically declared exact geometry.
    const int threads=int(params.blockDim.x*params.blockDim.y*params.blockDim.z);
    if(threads<=0)throw std::runtime_error("Observation HIP node has no block threads");
    check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&result.resident_block_limit,params.func,threads,params.sharedMemBytes));return result;
}
}
