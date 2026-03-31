/**
 * @file CoherenceTracker.cpp
 * @brief Implementation of per-buffer coherence tracking for BufferArena
 */

#include "CoherenceTracker.h"
#include "tensors/TensorClasses.h"
#include "utils/Logger.h"

namespace llaminar2
{

    bool CoherenceTracker::prepareForRead(TensorBase *tensor, CoherenceState &state, DeviceId target)
    {
        if (!tensor)
            return true; // External or null — nothing to do

        // Use tensor's canonical coherence state for transfer decisions,
        // with arena-level UNINITIALIZED check from CoherenceState
        if (!state.needsTransferTo(target, tensor->coherenceState()))
            return true; // Already in the right place

        if (target.is_gpu())
        {
            // Need data on GPU — upload from host
            if (!tensor->ensureOnDevice(target))
            {
                LOG_ERROR("CoherenceTracker: failed to upload tensor to " << target.to_string());
                return false;
            }
        }
        else
        {
            // Need data on CPU — download from GPU
            if (!tensor->ensureOnHost())
            {
                LOG_ERROR("CoherenceTracker: failed to download tensor to host");
                return false;
            }
        }

        return true;
    }

    bool CoherenceTracker::prepareForWrite(TensorBase *tensor, CoherenceState &state, DeviceId target)
    {
        if (!tensor)
            return true;

        if (target.is_gpu())
        {
            // Allocate GPU buffer if not yet allocated (don't transfer data)
            if (!tensor->allocateOnDevice(target))
            {
                LOG_ERROR("CoherenceTracker: failed to allocate device buffer on " << target.to_string());
                return false;
            }
        }
        // CPU writes just use the existing host buffer — nothing to allocate

        return true;
    }

    void CoherenceTracker::markWritten(CoherenceState &state, DeviceId device)
    {
        if (device.is_gpu())
        {
            state.authority = CoherenceState::DEVICE;
            state.authoritative_device = device;
        }
        else
        {
            state.authority = CoherenceState::HOST;
            state.authoritative_device = DeviceId::cpu();
        }
    }

    void CoherenceTracker::markWrittenWithEvent(TensorBase *tensor, CoherenceState &state,
                                                DeviceId device, void *stream)
    {
        markWritten(state, device);

        // Also update the tensor's canonical coherence state and record a GPU event
        // for fine-grained D2H sync when tensor->data() is later called.
        if (device.is_gpu() && tensor)
        {
            tensor->transitionToWithEvent(TensorCoherenceState::DEVICE_AUTHORITATIVE, device, stream);
        }
        else if (device.is_cpu() && tensor)
        {
            tensor->transitionTo(TensorCoherenceState::HOST_AUTHORITATIVE);
        }
    }

    void CoherenceTracker::markWrittenFlagsOnly(TensorBase *tensor, CoherenceState &state,
                                                DeviceId device)
    {
        markWritten(state, device);

        // Lightweight: update tensor coherence state without event recording.
        // The executor synchronizes streams at step boundaries, so per-tensor
        // events are unnecessary overhead during graph replay.
        if (device.is_gpu() && tensor)
        {
            tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        }
        else if (device.is_cpu() && tensor)
        {
            tensor->transitionTo(TensorCoherenceState::HOST_AUTHORITATIVE);
        }
    }

} // namespace llaminar2
