/**
 * @file DeviceGenerationGraphProgram.cpp
 * @brief Compile generation algorithms through one capture/controller contract.
 *
 * Construction consumes immutable bindings, not device values. All failures
 * stay on the requested architecture; the caller retains its original error
 * and cannot turn a missing conditional implementation into hosted replay.
 */
#include "DeviceGenerationGraphProgram.h"

#include "backends/IBackend.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"

namespace llaminar2
{
namespace
{
/** @brief Reject incomplete bindings before changing any native graph state. */
bool validateDestination(IGPUGraphCapture &destination,
    const DeviceGenerationGraphControl &control, std::string &error)
{
    error.clear();
    if (!control.valid())
        error = "Generation graph requires a valid algorithm and complete resident controller geometry";
    else if (isGraphCaptureActive())
        error = "Generation graph construction cannot nest inside another capture transaction";
    else if (!destination.executionStream())
        error = "Generation graph requires its owner's exact non-null stream";
    else if (destination.hasExecutable())
        error = "Generation graph construction cannot replace a retained executable; retire it explicitly";
    return error.empty();
}

/** @brief Require successful instantiation without launching or observing request data. */
bool instantiateProgram(IGPUGraphCapture &destination, std::string &error)
{
    if (destination.instantiate() && destination.hasExecutable())
        return true;
    error = "Generation graph instantiation did not produce a complete executable";
    return false;
}
} // namespace

bool DeviceGenerationGraphControl::valid() const noexcept
{
    return policy.valid() && rows &&
        stride >= sampling_math::kDeviceGenerationControlCount && requests > 0;
}

bool DeviceGenerationGraphProgram::native(IGPUGraphCapture &destination,
    const DeviceGenerationGraphControl &control,
    const DeviceControlledLoopProgram &program, std::string &error)
{
    using namespace sampling_math;
    if (!validateDestination(destination, control, error))
        return false;
    const bool dynamic = control.policy.mode == DeviceGenerationPolicyMode::Dynamic;
    if (!destination.supportsDeviceControlledWhileLoop() ||
        (dynamic && !destination.supportsDeviceControlledSelectorWhileLoop()))
    {
        error = "Generation graph backend cannot implement the declared native loop policy";
        return false;
    }
    if (program.iteration.empty())
    {
        error = "Generation graph requires a complete non-empty transaction";
        return false;
    }
    for (const auto &fragment : program.entry)
    {
        if (!fragment.valid() || fragment.capture == &destination ||
            fragment.capture->nodeCount() == 0)
        {
            error = "Generation graph entry requires complete participant-local arrival fragments";
            return false;
        }
    }
    for (const auto &fragment : program.initialization)
    {
        if (!fragment.valid() || fragment.capture == &destination ||
            fragment.capture->nodeCount() == 0 ||
            fragment.execution == DeviceControlledLoopFragmentExecution::IfDeviceSelectorAtLeast)
        {
            error = "Generation graph initialization requires complete non-selector fragments";
            return false;
        }
    }
    for (const auto &fragment : program.iteration)
    {
        if (!fragment.valid() || fragment.capture == &destination ||
            fragment.capture->nodeCount() == 0 ||
            (!dynamic && fragment.execution == DeviceControlledLoopFragmentExecution::IfDeviceSelectorAtLeast))
        {
            error = "Generation graph contains an incomplete, self-referencing or algorithm-incompatible fragment";
            return false;
        }
    }
    // A child supplies its recorded graph, not a separately launched executable.
    // Requiring child instantiation wastes native graph memory and rejects the
    // legitimate parent-only ownership used by ordinary generation. Backend
    // composition validates the recorded native graph and its device identity.
    const DeviceControlledLoopPredicate predicate{
        .control_rows_device = control.rows,
        .control_stride = control.stride,
        .request_count = control.requests,
        .healthy_index = kDeviceGenerationControlOk,
        .complete_index = kDeviceGenerationControlRequestComplete,
    };
    // The predicate is identical for ordinary and speculative generation.
    // Only dynamic speculative work needs an additional bounded selector.
    const bool built = dynamic
        ? destination.buildDeviceControlledSelectorWhileLoop(program, predicate,
            DeviceControlledLoopSelector{
                .control_rows_device = control.rows,
                .control_stride = control.stride,
                .request_count = control.requests,
                .healthy_index = kDeviceGenerationControlOk,
                .complete_index = kDeviceGenerationControlRequestComplete,
                .selector_index = kDeviceGenerationControlCurrentDraftDepth,
                .error_index = kDeviceGenerationControlErrorCode,
                .minimum_selector = control.policy.minimum_depth,
                .maximum_selector = control.policy.maximum_depth,
                .invalid_selector_error = static_cast<int>(DeviceGenerationError::InvalidDepthSelector),
            })
        : destination.buildDeviceControlledWhileLoop(program, predicate);
    if (!built)
    {
        error = "Generation graph could not construct its complete native transaction loop";
        return false;
    }
    return instantiateProgram(destination, error);
}

bool DeviceGenerationGraphProgram::ticketPublisher(IGPUGraphCapture &destination,
    IWorkerGPUContext &context, IBackend &backend, DeviceId device,
    const DeviceGenerationGraphControl &control, const uint32_t *maintenance_due,
    sampling_math::DeviceGenerationDispatchTicket *tickets, std::string &error)
{
    if (!validateDestination(destination, control, error))
        return false;
    if (context.isDeviceGraphCaptureActive())
    {
        error = "Generation ticket publisher cannot replace another worker capture transaction";
        return false;
    }
    if (!device.is_gpu() || backend.backendDeviceType() != device.type ||
        context.deviceOrdinal() != device.gpu_ordinal() || !tickets)
    {
        error = "Generation ticket publisher requires the exact GPU backend and persistent ticket storage";
        return false;
    }
    ScopedBackendGraphCapture recording(context, destination, "device_generation_ticket_publisher");
    if (!recording.begin())
    {
        error = "Generation ticket publisher could not begin isolated capture";
        return false;
    }
    const bool enqueued = backend.enqueuePublishDeviceGenerationDispatchTickets(
        control.rows, control.stride, control.requests, maintenance_due,
        tickets, device.gpu_ordinal(), destination.executionStream());
    // Close exactly once even when enqueue reports failure. The scope also
    // closes on an asynchronous-backend exception, preserving its diagnostic.
    recording.finish();
    if (!enqueued)
    {
        error = "Generation ticket publisher could not enqueue its immutable publication";
        return false;
    }
    return instantiateProgram(destination, error);
}
} // namespace llaminar2
