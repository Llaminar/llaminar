/**
 * @file PooledACKReadyCancellationCUDA.cu
 * @brief CUDA-only setup graph editing with original Driver kernel identities.
 *
 * Driver identities match CUDAGraphCapture inspection; runtime host-stub addresses
 * are never substituted. The added node is a test-only controller cancellation.
 */
#include "backends/cuda/CUDADriverApi.h"
#include "backends/cuda/CUDAGraphCapture.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
namespace llaminar2::test::pooled_ack::cuda_ack_cancel
{
/** @brief Preserve exact native error status during setup-only graph inspection/edit. */
inline void check(cudaError_t status)
{
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
/** @return Original Driver kernel parameters; handles belong to the captured runtime graph. */
inline CUDA_KERNEL_NODE_PARAMS originalParameters(cudaGraphNode_t node)
{
    CUDA_KERNEL_NODE_PARAMS result{};
    const auto status = CUDADriverApi::instance().graphKernelNodeGetParams(
        reinterpret_cast<CUgraphNode>(node), &result);
    if (status != CUDA_SUCCESS)
        throw std::runtime_error("Cancellation cuGraphKernelNodeGetParams failed");
    return result;
}
/** @return Same function-or-kernel identity published by CUDAGraphCapture::inspectKernelNodes. */
inline std::uintptr_t originalIdentity(const CUDA_KERNEL_NODE_PARAMS &params)
{
    if (params.func)
        return reinterpret_cast<std::uintptr_t>(params.func);
#if CUDA_VERSION >= 12000
    return reinterpret_cast<std::uintptr_t>(params.kern);
#else
    return 0;
#endif
}
} // namespace llaminar2::test::pooled_ack::cuda_ack_cancel
#define ACK_CANCEL_NAMESPACE cuda_ack_cancel
#define ACK_DEVICE_PREDICATE is_cuda
#define ACK_CAPTURE CUDAGraphCapture
#define ACK_NODE cudaGraphNode_t
#define ACK_NODE_TYPE cudaGraphNodeType
#define ACK_KERNEL_PARAMS cudaKernelNodeParams
#define ACK_RT(NAME) cuda##NAME
#define ACK_CHECK(STATUS) check(STATUS)
#define ACK_DEPENDENCIES(NODE, DEPS, COUNT) cudaGraphNodeGetDependencies(NODE, DEPS, nullptr, COUNT)
#define ACK_ADD_DEPENDENCIES(GRAPH, FROM, TO, COUNT) cudaGraphAddDependencies(GRAPH, FROM, TO, nullptr, COUNT)
#include "PooledACKReadyCancellationNative.cuh"
namespace llaminar2::test::pooled_ack
{
void spliceCUDA(IGPUGraphCapture &graph,
                                           const qwen4exp::pooled_source::DeviceArgs &expected, void *stream, AuthorizationReplay replay)
{
    cuda_ack_cancel::splice(graph, expected, stream, replay);
}
} // namespace llaminar2::test::pooled_ack
