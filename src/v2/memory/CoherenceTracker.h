/**
 * @file CoherenceTracker.h
 * @brief Per-buffer coherence state tracking for BufferArena
 *
 * Delegates coherence decisions to TensorBase::coherenceState()
 * (the canonical TensorCoherenceState state machine). The local
 * CoherenceState struct tracks only arena-level concerns
 * (e.g., "has this buffer ever been written?").
 *
 * This is internal to BufferArena — stages never interact with it directly.
 */

#pragma once

#include "backends/DeviceId.h"
#include "tensors/CoherenceState.h"
#include <cstddef>
#include <string>

namespace llaminar2
{

    // Forward declarations
    class ITensor;
    class TensorBase;

    /**
     * @brief Per-buffer coherence state for BufferArena.
     *
     * Tracks arena-level write history. Transfer decisions delegate
     * to the tensor's TensorCoherenceState via CoherenceTracker methods.
     */
    struct CoherenceState
    {
        enum Authority : uint8_t
        {
            UNINITIALIZED, ///< Buffer registered but never written
            HOST,          ///< Host copy is authoritative
            DEVICE,        ///< Device copy is authoritative
        };

        Authority authority = UNINITIALIZED;
        DeviceId authoritative_device; ///< Valid when authority == DEVICE (which GPU)

        /// Check if a transfer is needed to make data available on target.
        /// Uses arena-level UNINITIALIZED check first, then delegates
        /// to the tensor's coherence state for HOST/DEVICE decisions.
        bool needsTransferTo(DeviceId target, TensorCoherenceState tensor_state) const
        {
            if (authority == UNINITIALIZED)
                return false; // Write-only, no data to transfer

            // Delegate to the canonical coherence state machine
            if (target.is_gpu())
                return ::llaminar2::needsHostToDeviceUpload(tensor_state);

            if (target.is_cpu())
                return ::llaminar2::needsDeviceToHostSync(tensor_state);

            return false;
        }

        /// Legacy overload — kept for callers that don't have tensor state handy.
        /// @deprecated Prefer needsTransferTo(target, tensor_state).
        bool needsTransferTo(DeviceId target) const
        {
            if (authority == UNINITIALIZED)
                return false;

            if (authority == HOST && target.is_gpu())
                return true;

            if (authority == DEVICE && target.is_cpu())
                return true;

            if (authority == DEVICE && target.is_gpu() && target != authoritative_device)
                return true;

            return false;
        }
    };

    /**
     * @brief Coherence operations for the BufferArena.
     *
     * Wraps the calls to TensorBase::ensureOnDevice / ensureOnHost /
     * transitionTo(DEVICE_AUTHORITATIVE) so that BufferArena doesn't need to know about
     * tensor internals.
     */
    class CoherenceTracker
    {
    public:
        /**
         * @brief Ensure tensor data is available on target device for reading.
         *
         * If the tensor's host copy is authoritative and target is a GPU,
         * performs H2D upload. If device copy is authoritative and target
         * is CPU, performs D2H download.
         *
         * @param tensor  The underlying tensor (must be TensorBase-derived)
         * @param state   Current coherence state (updated on success)
         * @param target  Target device for reading
         * @return true on success
         */
        static bool prepareForRead(TensorBase *tensor, CoherenceState &state, DeviceId target, void *stream = nullptr);

        /**
         * @brief Ensure tensor has allocated storage on target device for writing.
         *
         * Does NOT transfer data (the kernel will overwrite it).
         * Allocates device buffer if not yet allocated.
         *
         * @param tensor  The underlying tensor
         * @param state   Current coherence state (not modified — markWritten does that)
         * @param target  Target device for writing
         * @return true on success
         */
        static bool prepareForWrite(TensorBase *tensor, CoherenceState &state, DeviceId target, void *stream = nullptr);

        /**
         * @brief Mark buffer as written on device.
         *
         * Updates coherence state to reflect that the given device now
         * holds the authoritative copy.
         *
         * @param state   Coherence state to update
         * @param device  Device that now holds authoritative data
         */
        static void markWritten(CoherenceState &state, DeviceId device);

        /**
         * @brief Mark buffer as written on device with stream event recording.
         *
         * Like markWritten() but also records a GPU completion event on the
         * tensor via transitionToWithEvent(DEVICE_AUTHORITATIVE, ..., stream). This enables
         * fine-grained sync: ensureOnHost() can wait on just this event
         * instead of doing a full device synchronize.
         *
         * @param tensor  The underlying tensor
         * @param state   Coherence state to update
         * @param device  Device that now holds authoritative data
         * @param stream  GPU stream where the kernel ran (nullptr = default)
         */
        static void markWrittenWithEvent(TensorBase *tensor, CoherenceState &state,
                                         DeviceId device, void *stream);

        /**
         * @brief Lightweight flags-only mark for graph replay (Phase 3).
         *
         * During graph replay, the executor synchronizes streams at the end
         * of the step, so per-tensor event recording is unnecessary overhead.
         * This variant updates coherence state and tensor flags without
         * recording a GPU event (~100-300µs savings per call on ROCm).
         *
         * @param tensor  The underlying tensor
         * @param state   Coherence state to update
         * @param device  Device that now holds authoritative data
         */
        static void markWrittenFlagsOnly(TensorBase *tensor, CoherenceState &state,
                                         DeviceId device);
    };

} // namespace llaminar2
