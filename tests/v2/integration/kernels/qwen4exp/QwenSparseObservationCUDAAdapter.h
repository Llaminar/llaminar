/**
 * @file QwenSparseObservationCUDAAdapter.h
 * @brief Setup-only CUDA graph identity and static-resource adapter for native events.
 *
 * Uses the established original Driver handles, including CUfunction/CUkernel
 * distinction. No CUPTI, profiler attachment, eager replay or host-stub substitution.
 */
#pragma once
#include "backends/cuda/CUDADriverApi.h"
#include "QwenSparseObservationNativeMetadata.h"
namespace llaminar2::qwen4exp::QSA_SUITE::observation115
{
using ObservationNode=cudaGraphNode_t;
using ObservationNodeType=cudaGraphNodeType;
using ObservationKernelParams=cudaKernelNodeParams;
inline constexpr const char* ObservationBackend="CUDA";
/**
 * @brief Read dependency identities using the CUDA graph-edge API.
 * @param node Borrowed node in the retained fixture graph.
 * @param parents Optional output array; null queries only its required capacity.
 * @param count In/out array capacity and returned dependency count.
 * @return Native status; the caller rejects an incomplete or lossy edge query.
 * @details Fixture captures use ordinary dependency edges. No graph is mutated,
 * launched or synchronized, and no edge metadata is discarded on successful return.
 */
inline cudaError_t observationDependencies(ObservationNode node,ObservationNode* parents,std::size_t* count)
{
    return cudaGraphNodeGetDependencies(node,parents,nullptr,count);
}
/**
 * @brief Inspect one original recorded CUDA node without launching it.
 * @param node Kernel node borrowed from the retained runtime graph.
 * @param stream Exact fixture stream; the graph and context must remain live.
 * @return Original captured identity, symbol, static resources and 256-thread resource-based resident-block limit (theoretical occupancy).
 * @throws std::runtime_error For missing identities or any Driver API failure.
 * @details Inspection occurs after capture and before timing. Handles remain
 * borrowed; this operation performs no synchronization or profiler attachment.
 */
inline NativeKernelMetadata inspectObservationKernel(ObservationNode node,QSA_STREAM stream)
{
    (void)stream;
    const auto& driver=CUDADriverApi::instance();
    const auto checked=[](CUresult status){if(status!=CUDA_SUCCESS)throw std::runtime_error("Observation CUDA Driver metadata query failed, CUresult="+std::to_string(static_cast<int>(status)));};
    CUDA_KERNEL_NODE_PARAMS params{};checked(driver.graphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(node),&params));
    NativeKernelMetadata result;CUfunction function=nullptr;const char* name=nullptr;
    if(params.func){function=params.func;result.identity=reinterpret_cast<void*>(params.func);checked(driver.funcGetName(&name,function));}
#if CUDA_VERSION >= 12000
    else if(params.kern){result.identity=reinterpret_cast<void*>(params.kern);checked(driver.kernelGetName(&name,params.kern));checked(driver.kernelGetFunction(&function,params.kern));}
#endif
    else throw std::runtime_error("Observation CUDA node lacks an original native kernel identity");
    if(!function||!name||!*name)throw std::runtime_error("Observation CUDA native symbol is empty");
    result.name=name;
    int shared=0,local=0;
    checked(driver.funcGetAttribute(&result.registers,CU_FUNC_ATTRIBUTE_NUM_REGS,function));
    checked(driver.funcGetAttribute(&shared,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,function));
    checked(driver.funcGetAttribute(&local,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,function));
    checked(driver.funcGetAttribute(&result.max_threads,CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,function));
    checked(driver.occupancyMaxActiveBlocksPerMultiprocessor(&result.resident_block_limit,function,256,params.sharedMemBytes));
    result.shared_bytes=static_cast<std::size_t>(shared);result.local_bytes=static_cast<std::size_t>(local);return result;
}
}
