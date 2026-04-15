/**
 * @file DeviceWeightPreparer.h
 * @brief Prepares GEMM repacked weights and uploads non-GEMM weights to target devices
 *
 * DeviceWeightPreparer takes a WeightViewSet (the PP stage's weight subset) and a
 * target DeviceId, then:
 *   1. For GEMM weights: calls KernelFactory::getOrCreatePreparedGemmWeights()
 *      with the correct ordinal guard
 *   2. For non-GEMM weights (norms, biases, embeddings): calls ensureOnDevice()
 *
 * This decouples weight preparation from graph construction, enabling clean
 * heterogeneous multi-device PP where each stage independently prepares its
 * weights for its target device.
 *
 * Thread safety: prepare() may be called from a single thread per device.
 * The underlying KernelFactory is internally synchronized for concurrent access.
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#pragma once

#include "../loaders/WeightViewSet.h"
#include "../backends/DeviceId.h"
#include <cstddef>
#include <string>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Result of a device weight preparation pass
     */
    struct WeightPreparationResult
    {
        size_t gemm_weights_prepared = 0;      ///< Count of GEMM weights repacked
        size_t non_gemm_weights_uploaded = 0;  ///< Count of non-GEMM weights uploaded
        size_t total_device_bytes = 0;         ///< Total device memory allocated
        size_t failures = 0;                   ///< Number of failed preparations
        std::vector<std::string> failed_names; ///< Names of weights that failed
    };

    /**
     * @brief Prepares a PP stage's weights for execution on a specific device
     *
     * Usage:
     * @code
     * auto pool = SharedWeightPool::create();
     * pool->loadFromGGUF("model.gguf");
     *
     * auto views = pool->createViewSet(0, 12, true, false);  // Layers 0-11
     *
     * DeviceWeightPreparer preparer;
     * auto result = preparer.prepare(views, DeviceId::cuda(0));
     * LOG_INFO("Prepared " << result.gemm_weights_prepared << " GEMM + "
     *          << result.non_gemm_weights_uploaded << " non-GEMM weights");
     * @endcode
     */
    class DeviceWeightPreparer
    {
    public:
        DeviceWeightPreparer() = default;
        ~DeviceWeightPreparer() = default;

        // Non-copyable, movable
        DeviceWeightPreparer(const DeviceWeightPreparer &) = delete;
        DeviceWeightPreparer &operator=(const DeviceWeightPreparer &) = delete;
        DeviceWeightPreparer(DeviceWeightPreparer &&) = default;
        DeviceWeightPreparer &operator=(DeviceWeightPreparer &&) = default;

        /**
         * @brief Prepare all weights in the view set for execution on the target device
         *
         * For GPU devices (CUDA/ROCm):
         *   - GEMM weights: Creates PreparedGemmHandle via KernelFactory with ordinal guard
         *   - Non-GEMM weights: Calls ensureOnDevice() for host→device upload
         *
         * For CPU devices:
         *   - GEMM weights: Creates CPU-packed PreparedGemmHandle (VNNI layout)
         *   - Non-GEMM weights: No-op (already on host)
         *
         * @param views The weight view set for this PP stage
         * @param target_device The device to prepare weights for
         * @return Result struct with counts and any failures
         */
        WeightPreparationResult prepare(
            const WeightViewSet &views,
            DeviceId target_device);

    private:
        /**
         * @brief Prepare a single GEMM weight via KernelFactory
         */
        static bool prepareGemmWeight(const WeightView &view, DeviceId target_device);

        /**
         * @brief Upload a non-GEMM weight to the target device via ensureOnDevice
         */
        static bool prepareNonGemmWeight(const WeightView &view, DeviceId target_device);
    };

} // namespace llaminar2
