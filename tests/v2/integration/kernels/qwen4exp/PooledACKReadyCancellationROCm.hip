/**
 * @file PooledACKReadyCancellationROCm.hip
 * @brief HIP-only original graph identity/edge authentication for cancellation.
 *
 * Native function identity and argument metadata are inspected before the single
 * retained executable is instantiated. No production hook or host race is used.
 */
#include "backends/rocm/HIPGraphCapture.h"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <stdexcept>
namespace llaminar2::test::pooled_ack::rocm_ack_cancel
{
/** @brief Preserve exact native error status during setup-only graph inspection/edit. */
inline void check(hipError_t status)
{
    if (status != hipSuccess)
        throw std::runtime_error(hipGetErrorString(status));
}
/** @return Captured native argument metadata, not live GPU values. */
inline hipKernelNodeParams originalParameters(hipGraphNode_t node)
{
    hipKernelNodeParams result{};
    check(hipGraphKernelNodeGetParams(node, &result));
    return result;
}
/** @return Same function identity published by HIPGraphCapture::inspectKernelNodes. */
inline std::uintptr_t originalIdentity(const hipKernelNodeParams &params)
{
    return reinterpret_cast<std::uintptr_t>(params.func);
}
} // namespace llaminar2::test::pooled_ack::rocm_ack_cancel
#define ACK_CANCEL_NAMESPACE rocm_ack_cancel
#define ACK_DEVICE_PREDICATE is_rocm
#define ACK_CAPTURE HIPGraphCapture
#define ACK_NODE hipGraphNode_t
#define ACK_NODE_TYPE hipGraphNodeType
#define ACK_KERNEL_PARAMS hipKernelNodeParams
#define ACK_RT(NAME) hip##NAME
#define ACK_CHECK(STATUS) check(STATUS)
#define ACK_DEPENDENCIES(NODE, DEPS, COUNT) hipGraphNodeGetDependencies(NODE, DEPS, COUNT)
#define ACK_ADD_DEPENDENCIES(GRAPH, FROM, TO, COUNT) hipGraphAddDependencies(GRAPH, FROM, TO, COUNT)
#include "PooledACKReadyCancellationNative.cuh"
namespace llaminar2::test::pooled_ack
{
void spliceROCm(IGPUGraphCapture &graph,
                                           const qwen4exp::pooled_source::DeviceArgs &expected, void *stream, AuthorizationReplay replay)
{
    rocm_ack_cancel::splice(graph, expected, stream, replay);
}
} // namespace llaminar2::test::pooled_ack
