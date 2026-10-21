/**
 * @file Test__PipelineForwardOnlyFailureCUDA.cpp
 * @brief Dedicated CUDA symbol-interposition test, never linked into production.
 *
 * Every vendor call is forwarded unchanged first. Only one successful real D2H
 * enqueue on an authenticated parent stream receives a synthetic InvalidValue
 * return. No runtime last-error state is read, cleared or manufactured. Native
 * stream/event ownership is observed under the actual calling device.
 */
#include <cuda_runtime_api.h>
#include "../../../utils/PipelineForwardOnlyFaultProbe.h"
using llaminar2::test::pp_fault::Probe;
using llaminar2::test::pp_fault::original;
namespace {
/** @brief Query the actual calling device without changing stream or ownership. */
int callingDevice()
{
    static const auto real = original<decltype(&cudaGetDevice)>("cudaGetDevice");
    int device = -1;
    return real(&device) == cudaSuccess ? device : -1;
}
}
/** @brief Observe real accepted parent launches; effective injection arms after all peers. */
extern "C" cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t graph, cudaStream_t stream)
{
    static const auto real = original<decltype(&cudaGraphLaunch)>("cudaGraphLaunch");
    const auto status = real(graph, stream);
    if (auto *probe = Probe::active()) probe->launch(graph, stream, callingDevice(), status == cudaSuccess);
    return status;
}
/** @brief Forward one real enqueue, then override only its selected successful return. */
extern "C" cudaError_t CUDARTAPI cudaMemcpyAsync(void *destination, const void *source, size_t bytes,
    cudaMemcpyKind kind, cudaStream_t stream)
{
    static const auto real = original<decltype(&cudaMemcpyAsync)>("cudaMemcpyAsync");
    const auto status = real(destination, source, bytes, kind, stream);
    if (kind == cudaMemcpyDeviceToHost)
        if (auto *probe = Probe::active())
            if (probe->copy(destination, source, bytes, stream, callingDevice(), status == cudaSuccess))
                return cudaErrorInvalidValue;
    return status;
}
/** @brief Record the native event and its real exact stream before observing it. */
extern "C" cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    static const auto real = original<decltype(&cudaEventRecord)>("cudaEventRecord");
    const auto status = real(event, stream);
    if (auto *probe = Probe::active()) probe->record(event, stream, callingDevice(), status == cudaSuccess);
    return status;
}
/** @brief Observe only successful real fences; the test adds no fence of its own. */
extern "C" cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event)
{
    static const auto real = original<decltype(&cudaEventSynchronize)>("cudaEventSynchronize");
    const auto status = real(event);
    if (auto *probe = Probe::active()) probe->wait(event, callingDevice(), status == cudaSuccess);
    return status;
}
/** @brief Require exact-owner completion before forwarding real executable destruction. */
extern "C" cudaError_t CUDARTAPI cudaGraphExecDestroy(cudaGraphExec_t graph)
{
    static const auto real = original<decltype(&cudaGraphExecDestroy)>("cudaGraphExecDestroy");
    const int device = callingDevice();
    if (auto *probe = Probe::active()) probe->destroy(graph, device, true);
    const auto status = real(graph);
    if (auto *probe = Probe::active()) probe->destroy(graph, device, false, status == cudaSuccess);
    return status;
}
#define LLAMINAR_PP_FAULT_BACKEND "CUDA"
#include "PipelineForwardOnlyFailureCases.h"
