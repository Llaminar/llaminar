/**
 * @file CoherenceTracker.cpp
 * @brief Implementation of per-buffer coherence tracking for BufferArena
 */

#include "CoherenceTracker.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/TensorClasses.h"
#include "transfer/TransferEngine.h"
#include "utils/Logger.h"
#include "utils/Assertions.h"

#include <stdexcept>

namespace llaminar2
{

    bool CoherenceTracker::prepareForRead(TensorBase *tensor, CoherenceState &state, DeviceId target, void *stream)
    {
        if (!tensor)
            return true; // External or null — nothing to do

        /*
         * Native capture is validation-only: allocation, upload, migration, and
         * external event discovery have already happened before beginCapture().
         * TransferEngine consults the scoped dependency ledger to distinguish an
         * earlier recorded producer from a strict graph-external input.
         */
        if (target.is_gpu() && currentGraphCaptureDependencyLedger())
        {
            TransferEngine::requireDeviceInput(tensor, target, stream);
            return true;
        }

#if LLAMINAR_ASSERTIONS_ACTIVE
        // Invariant: if the arena says DEVICE is authoritative, the tensor
        // MUST NOT be HOST_AUTHORITATIVE. If it is, something reset the
        // tensor's coherence state outside arena control (e.g., allocateOnDevice
        // bug from commit 74de820e).
        if (state.authority == CoherenceState::DEVICE &&
            tensor->coherenceState() == TensorCoherenceState::HOST_AUTHORITATIVE)
        {
            LOG_ERROR("[COHERENCE_INVARIANT] Arena says DEVICE authoritative but tensor is "
                      "HOST_AUTHORITATIVE — coherence state was reset outside arena control. "
                      "tensor=" << static_cast<const void *>(tensor)
                                << " target=" << target.to_string());
            tensor->dumpCoherenceAuditLog();
            LLAMINAR_ASSERT(false,
                            "Arena/tensor coherence state mismatch — see audit log. "
                            "Arena: DEVICE, Tensor: HOST_AUTHORITATIVE");
        }
#endif

        // Use tensor's canonical coherence state for transfer decisions,
        // with arena-level UNINITIALIZED check from CoherenceState
        if (!state.needsTransferTo(target, tensor->coherenceState()))
        {
            /*
             * Correct residency does not imply correct stream ordering. The
             * producer may have published device authority on another stream,
             * so every GPU read must join that event to its exact consumer.
             */
            if (target.is_gpu())
                TransferEngine::requireDeviceInput(tensor, target, stream);
            return true;
        }

        if (target.is_gpu())
        {
            // TransferEngine validates the exact stream, performs any upload,
            // and joins an existing producer event before arena consumers run.
            TransferEngine::prepareDeviceInput(tensor, target, stream);
        }
        else
        {
            // CPU materialization is an explicit transfer boundary.
            const auto result = TransferEngine::instance().download(tensor);
            if (!result.success)
            {
                LOG_ERROR("CoherenceTracker: failed to download tensor to host: "
                          << result.error);
                return false;
            }
        }

        return true;
    }

    bool CoherenceTracker::prepareForWrite(TensorBase *tensor, CoherenceState &state, DeviceId target, void *stream)
    {
        if (!tensor)
            return true;

        if (target.is_gpu())
        {
            /*
             * Capture preflight has already established stable output
             * addresses. Recording may validate that allocation but must not
             * call the placement-capable API, even when it would happen to be
             * a no-op for the current process state.
             */
            if (isGraphCaptureActive())
                TransferEngine::requireDeviceOutput(tensor, target, stream);
            else
                TransferEngine::prepareDeviceOutput(tensor, target, stream);
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
        if (!tensor)
        {
            throw std::invalid_argument(
                "CoherenceTracker::markWrittenWithEvent requires a tensor");
        }
        if (device.is_gpu() && !stream)
        {
            throw std::invalid_argument(
                "CoherenceTracker::markWrittenWithEvent requires the exact "
                "non-null GPU producer stream");
        }

        /*
         * Publish the tensor-level contract first. Event creation or recording
         * can fail, and the arena must not expose device authority unless that
         * exact producer dependency was committed successfully.
         */
        if (device.is_gpu())
        {
            TransferEngine::publishDeviceWrite(tensor, device, stream);
            if (currentGraphCaptureDependencyLedger())
            {
                /*
                 * The producer is only recorded.  ScopedGraphCaptureStage owns
                 * the in-transaction edge, and the post-launch graph boundary
                 * will publish real global authority.
                 */
                return;
            }
        }
        else
        {
            TransferEngine::publishHostWrite(tensor);
        }

        markWritten(state, device);
    }

    void CoherenceTracker::markWrittenFlagsOnly(TensorBase *tensor, CoherenceState &state,
                                                DeviceId device)
    {
        if (device.is_gpu() && currentGraphCaptureDependencyLedger())
        {
            /*
             * Do not make a recorded-but-unlaunched write globally visible.
             * Stage completion advances the capture ledger without mutating the
             * arena or TensorBase coherence state.
             */
            return;
        }

        markWritten(state, device);

        // Lightweight graph-replay publication: the graph executor owns the
        // single replay-completion event, so per-tensor events are redundant.
        if (device.is_gpu() && tensor)
        {
            TransferEngine::publishGraphOwnedDeviceWrite(tensor, device);
        }
        else if (device.is_cpu() && tensor)
        {
            TransferEngine::publishHostWrite(tensor);
        }
    }

} // namespace llaminar2
