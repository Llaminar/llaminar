/**
 * @file IMultiDeviceOrchestrator.h
 * @brief Interface for multi-device orchestration extending IInferenceRunner
 *
 * This interface extends IInferenceRunner to add multi-device specific
 * capabilities, enabling:
 * - Access to per-device inference runners
 * - LOCAL tensor parallelism context management
 * - Device count and coordination
 *
 * Design goals:
 * - Drop-in replacement for IInferenceRunner (extends it)
 * - Testable via dependency injection
 * - Compatible with MockMultiDeviceOrchestrator for unit testing
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "IInferenceRunner.h"

namespace llaminar2
{

    // Forward declarations
    class ILocalTPContext;

    /**
     * @brief Interface for multi-device orchestration
     *
     * Extends IInferenceRunner to provide multi-device coordination capabilities.
     * This is the primary interface for LOCAL tensor parallelism across multiple
     * devices within a single MPI rank.
     *
     * Key concepts:
     * - device_count(): Number of devices participating in LOCAL TP
     * - deviceRunner(): Access individual per-device runners
     * - localTPContext(): Access the LOCAL TP collective context
     *
     * Thread safety: All methods are thread-safe. The multi-device orchestrator
     * coordinates execution across devices internally.
     *
     * Usage:
     * @code
     * // Get orchestrator from factory
     * auto orchestrator = createMultiDeviceOrchestrator(config);
     *
     * // Use as IInferenceRunner (single interface)
     * orchestrator->forward(tokens, seq_len);
     * const float* logits = orchestrator->logits();
     *
     * // Or access multi-device specifics
     * int num_devices = orchestrator->device_count();
     * for (int i = 0; i < num_devices; ++i) {
     *     auto* runner = orchestrator->deviceRunner(i);
     *     // Inspect per-device state...
     * }
     *
     * // Access LOCAL TP context for collective operations
     * auto* tp_ctx = orchestrator->localTPContext();
     * tp_ctx->allreduce(tensor);
     * @endcode
     */
    class IMultiDeviceOrchestrator : public IInferenceRunner
    {
    public:
        ~IMultiDeviceOrchestrator() override = default;

        // =====================================================================
        // Multi-Device Query API
        // =====================================================================

        /**
         * @brief Get the number of devices participating in LOCAL TP
         *
         * This is the LOCAL TP degree - number of devices owned by this rank.
         *
         * @return Number of devices (>= 1)
         */
        virtual int device_count() const = 0;

        /**
         * @brief Get the inference runner for a specific device
         *
         * Each device has its own inference runner with its own KV cache,
         * weight shard, and compute graph instance.
         *
         * @param device_idx 0-based device index (must be < device_count())
         * @return Pointer to the device's inference runner (never nullptr for valid index)
         * @throws std::out_of_range if device_idx >= device_count()
         *
         * @note The returned pointer is valid for the lifetime of the orchestrator.
         *       Do not delete or store long-term.
         */
        virtual IInferenceRunner *deviceRunner(int device_idx) = 0;

        /**
         * @brief Get the inference runner for a specific device (const version)
         *
         * @param device_idx 0-based device index (must be < device_count())
         * @return Const pointer to the device's inference runner
         * @throws std::out_of_range if device_idx >= device_count()
         */
        virtual const IInferenceRunner *deviceRunner(int device_idx) const = 0;

        // =====================================================================
        // LOCAL TP Context API
        // =====================================================================

        /**
         * @brief Get the LOCAL tensor parallelism context
         *
         * The LOCAL TP context manages collective operations (allreduce, allgather)
         * across devices within this rank. It handles:
         * - Device synchronization
         * - Backend selection (NCCL/RCCL/HOST)
         * - Proportional work distribution (weights)
         *
         * @return Pointer to the LOCAL TP context (may be nullptr if device_count() == 1)
         *
         * @note For single-device orchestration, this may return nullptr since
         *       no collective operations are needed.
         */
        virtual ILocalTPContext *localTPContext() = 0;

        /**
         * @brief Get the LOCAL tensor parallelism context (const version)
         *
         * @return Const pointer to the LOCAL TP context
         */
        virtual const ILocalTPContext *localTPContext() const = 0;

        // =====================================================================
        // Device Status API
        // =====================================================================

        /**
         * @brief Check if all devices are ready for inference
         *
         * Verifies that all device runners have:
         * - Loaded weights
         * - Allocated KV caches
         * - Initialized compute graphs
         *
         * @return true if all devices are ready
         */
        virtual bool allDevicesReady() const = 0;

        /**
         * @brief Synchronize all devices
         *
         * Ensures all pending device operations (kernel launches, transfers)
         * have completed. Call this before accessing results that may be
         * computed asynchronously across devices.
         */
        virtual void synchronizeDevices() = 0;
    };

} // namespace llaminar2
