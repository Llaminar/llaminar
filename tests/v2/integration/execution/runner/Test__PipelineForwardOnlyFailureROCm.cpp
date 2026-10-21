/**
 * @file Test__PipelineForwardOnlyFailureROCm.cpp
 * @brief Dedicated ROCm symbol-interposition test, never linked into production.
 *
 * Every vendor call is forwarded unchanged first. Only one successful real D2H
 * enqueue on an authenticated parent stream receives a synthetic InvalidValue
 * return. No runtime last-error state is read, cleared or manufactured. Native
 * stream/event ownership is observed under the actual calling device.
 */
#include <hip/hip_runtime_api.h>
#include "../../../utils/PipelineForwardOnlyFaultProbe.h"
using llaminar2::test::pp_fault::Probe;
using llaminar2::test::pp_fault::original;
namespace {
/** @brief Query the actual calling device without changing stream or ownership. */
int callingDevice()
{
    static const auto real = original<decltype(&hipGetDevice)>("hipGetDevice");
    int device = -1;
    return real(&device) == hipSuccess ? device : -1;
}
}
/** @brief Observe real accepted parent launches; effective injection arms after all peers. */
extern "C" hipError_t hipGraphLaunch(hipGraphExec_t graph, hipStream_t stream)
{
    static const auto real = original<decltype(&hipGraphLaunch)>("hipGraphLaunch");
    const auto status = real(graph, stream);
    if (auto *probe = Probe::active()) probe->launch(graph, stream, callingDevice(), status == hipSuccess);
    return status;
}
/** @brief Forward one real enqueue, then override only its selected successful return. */
extern "C" hipError_t hipMemcpyAsync(void *destination, const void *source, size_t bytes,
    hipMemcpyKind kind, hipStream_t stream)
{
    static const auto real = original<decltype(&hipMemcpyAsync)>("hipMemcpyAsync");
    const auto status = real(destination, source, bytes, kind, stream);
    if (kind == hipMemcpyDeviceToHost)
        if (auto *probe = Probe::active())
            if (probe->copy(destination, source, bytes, stream, callingDevice(), status == hipSuccess))
                return hipErrorInvalidValue;
    return status;
}
/** @brief Record the native event and its real exact stream before observing it. */
extern "C" hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream)
{
    static const auto real = original<decltype(&hipEventRecord)>("hipEventRecord");
    const auto status = real(event, stream);
    if (auto *probe = Probe::active()) probe->record(event, stream, callingDevice(), status == hipSuccess);
    return status;
}
/** @brief Observe only successful real fences; the test adds no fence of its own. */
extern "C" hipError_t hipEventSynchronize(hipEvent_t event)
{
    static const auto real = original<decltype(&hipEventSynchronize)>("hipEventSynchronize");
    const auto status = real(event);
    if (auto *probe = Probe::active()) probe->wait(event, callingDevice(), status == hipSuccess);
    return status;
}
/** @brief Require exact-owner completion before forwarding real executable destruction. */
extern "C" hipError_t hipGraphExecDestroy(hipGraphExec_t graph)
{
    static const auto real = original<decltype(&hipGraphExecDestroy)>("hipGraphExecDestroy");
    const int device = callingDevice();
    if (auto *probe = Probe::active()) probe->destroy(graph, device, true);
    const auto status = real(graph);
    if (auto *probe = Probe::active()) probe->destroy(graph, device, false, status == hipSuccess);
    return status;
}
#define LLAMINAR_PP_FAULT_BACKEND "ROCm"
#include "PipelineForwardOnlyFailureCases.h"
