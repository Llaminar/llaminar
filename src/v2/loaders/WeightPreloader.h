/**
 * @file WeightPreloader.h
 * @brief Pre-packs model weights before graph execution
 *
 * WeightPreloader ensures all model weights are packed for their target devices
 * before inference begins. This eliminates first-use packing overhead during
 * graph execution and allows memory to be released early.
 *
 * Supports:
 * - CPU: VNNI INT8 packing for AVX-512 kernels
 * - CUDA: INT8 + per-column scales for CUTLASS kernels
 * - ROCm: (future) similar to CUDA
 *
 * Usage:
 *   auto preloader = WeightPreloader(weight_manager, placement_map);
 *   preloader.preloadAll();  // Pack all weights before inference
 *
 * @see WeightManager for weight loading
 * @see WeightPlacementMap for device assignment
 * @see KernelFactory::ensurePackedWeightsInTensorCache for packing implementation
 *
 * @author David Sanftenberg
 */

#pragma once

#include "WeightManager.h"
#include "WeightPlacementMap.h"
#include "../backends/DeviceId.h"
#include "../backends/DeviceType.h"
#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace llaminar2
{

    /**
     * @brief Pre-packing progress callback
     *
     * Called during preloading with:
     * @param current Number of weights packed so far
     * @param total Total number of weights to pack
     * @param name Current weight name being packed
     */
    using PreloadProgressCallback = std::function<void(size_t current, size_t total, const std::string &name)>;

    /**
     * @brief Pre-packs model weights before graph execution
     *
     * This class coordinates weight packing across all devices before inference
     * begins, ensuring:
     * 1. No first-use packing overhead during execution
     * 2. Raw GGUF data can be released after packing (memory savings)
     * 3. Progress reporting for large models
     *
     * Design notes:
     * - Uses WeightPlacementMap to determine target device for each weight
     * - Delegates to KernelFactory::ensurePackedWeightsInTensorCache() for actual packing
     * - Thread-safe: can be called from single thread while weights are used elsewhere
     */
    class WeightPreloader
    {
    public:
        /**
         * @brief Construct preloader
         *
         * @param weight_manager WeightManager containing loaded weights
         * @param placement_map Device placement map (nullptr = all CPU)
         */
        WeightPreloader(
            std::shared_ptr<WeightManager> weight_manager,
            std::shared_ptr<WeightPlacementMap> placement_map = nullptr);

        /**
         * @brief Preload all weights in the weight manager
         *
         * Iterates all cached weights and packs them for their target devices.
         * For weights without explicit placement, uses CPU.
         *
         * @param progress_callback Optional callback for progress reporting
         * @param release_raw_data If true, release raw GGUF data after packing (default: true)
         * @return true if all weights were packed successfully
         */
        bool preloadAll(
            PreloadProgressCallback progress_callback = nullptr,
            bool release_raw_data = true);

        /**
         * @brief Preload specific weights by name
         *
         * @param weight_names Names of weights to preload
         * @param progress_callback Optional callback for progress reporting
         * @param release_raw_data If true, release raw GGUF data after packing
         * @return true if all specified weights were packed successfully
         */
        bool preload(
            const std::vector<std::string> &weight_names,
            PreloadProgressCallback progress_callback = nullptr,
            bool release_raw_data = true);

        /**
         * @brief Preload all weights for a specific device type
         *
         * Packs all weights that are targeted for the given device type.
         *
         * @param target_device Device type to preload for
         * @param progress_callback Optional callback for progress reporting
         * @param release_raw_data If true, release raw GGUF data after packing
         * @return true if all matching weights were packed successfully
         */
        bool preloadForDevice(
            DeviceType target_device,
            PreloadProgressCallback progress_callback = nullptr,
            bool release_raw_data = true);

        /**
         * @brief Preload all weights for a specific device (with ordinal)
         *
         * Packs all weights that are targeted for the given device.
         * This overload preserves the device ordinal for multi-GPU setups.
         *
         * @param target_device DeviceId to preload for (includes type and ordinal)
         * @param progress_callback Optional callback for progress reporting
         * @param release_raw_data If true, release raw GGUF data after packing
         * @return true if all matching weights were packed successfully
         */
        bool preloadForDevice(
            DeviceId target_device,
            PreloadProgressCallback progress_callback = nullptr,
            bool release_raw_data = true);

        /**
         * @brief Get statistics about preloaded weights
         *
         * @return Pair of (num_cpu_packed, num_gpu_packed)
         */
        std::pair<size_t, size_t> stats() const { return {num_cpu_packed_, num_gpu_packed_}; }

    private:
        /**
         * @brief Get target device for a weight
         *
         * @param weight_name Name of the weight
         * @return DeviceId for the weight (CPU if no placement map)
         */
        DeviceId getTargetDevice(const std::string &weight_name) const;

        /**
         * @brief Pack a single weight tensor
         *
         * @param tensor Weight tensor to pack
         * @param target_device Target device (type + ordinal for multi-GPU)
         * @param release_raw_data If true, release raw data after packing
         * @return true if packing succeeded
         */
        bool packWeight(
            TensorBase *tensor,
            DeviceId target_device,
            bool release_raw_data);

        /**
         * @brief Upload all non-GEMM weights to GPU
         *
         * Non-GEMM weights (norms, embeddings, biases) don't need GEMM packing
         * but still need to be uploaded to GPU for kernel access.
         * This eliminates lazy upload overhead during inference.
         *
         * @param target_device Target GPU device
         * @return Number of weights uploaded
         */
        size_t uploadNonGemmWeights(DeviceId target_device);

        /**
         * @brief Preload weights with explicit device override
         *
         * Used when no placement map exists - forces all weights to specific device.
         *
         * @param weight_names Names of weights to preload
         * @param override_device Force all weights to this device (with ordinal)
         * @param progress_callback Optional callback for progress reporting
         * @param release_raw_data If true, release raw GGUF data after packing
         * @return true if all specified weights were packed successfully
         */
        bool preloadWithOverrideDevice(
            const std::vector<std::string> &weight_names,
            DeviceId override_device,
            PreloadProgressCallback progress_callback = nullptr,
            bool release_raw_data = true);

        std::shared_ptr<WeightManager> weight_manager_;
        std::shared_ptr<WeightPlacementMap> placement_map_;

        // Statistics
        size_t num_cpu_packed_ = 0;
        size_t num_gpu_packed_ = 0;
    };

} // namespace llaminar2
