/**
 * @file ILocalPPContext.h
 * @brief Interface for LOCAL pipeline parallelism activation transfer
 *
 * LOCAL PP = pipeline parallel stages within a single MPI rank, decoupled from MPI world_size.
 * This enables pipeline parallelism across devices owned by one rank, using high-bandwidth
 * backends like NCCL, RCCL, PCIeBAR, or HOST for activation transfer between stages.
 *
 * Key concepts:
 * - LOCAL PP degree can be different from MPI world_size
 * - Each PP stage maps to a specific device OR a nested TP/PP context
 * - Layer boundaries define which layers execute on which stage
 * - Backend selection based on device types (NCCL for CUDA-only, PCIeBAR for mixed)
 *
 * Pipeline parallel activation flow (simple):
 *   Stage 0 (layers 0-5)    → transfer →    Stage 1 (layers 6-11)    → transfer →    Stage 2 (layers 12-23)
 *        [cuda:0]                                [cuda:1]                                  [rocm:0]
 *
 * Hierarchical topology (PP wrapping TP):
 *   Stage 0 (layers 0-11)        → transfer →    Stage 1 (layers 12-23)    → transfer →    Stage 2 (final)
 *   [TP(rocm:0, rocm:1)]                              [cuda:0]                                  [cpu]
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../backends/GlobalDeviceAddress.h"
#include "../config/OrchestrationConfig.h"
#include "../tensors/ITensor.h"
#include "PPStage.h"
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class TensorBase;
    class ILocalTPContext;

    /**
     * @brief Configuration for LocalPPContext
     *
     * Describes the pipeline parallel topology: which device handles each stage
     * and which layers belong to each stage.
     */
    struct LocalPPConfig
    {
        /**
         * @brief Device assignment for each PP stage
         *
         * stage_devices[i] is the device that executes stage i.
         * Example: [cuda:0, cuda:1, rocm:0] for 3-stage PP
         */
        std::vector<GlobalDeviceAddress> stage_devices;

        /**
         * @brief Layer boundaries for each stage
         *
         * Defines the layer ranges as [start, end) pairs via boundary indices.
         * layer_boundaries[i] is the first layer of stage i.
         * layer_boundaries[i+1] is the first layer of stage i+1 (exclusive end for stage i).
         *
         * Example for 24 layers with 3 stages:
         *   layer_boundaries = [0, 8, 16, 24]
         *   Stage 0: layers [0, 8)   = layers 0-7
         *   Stage 1: layers [8, 16)  = layers 8-15
         *   Stage 2: layers [16, 24) = layers 16-23
         *
         * Size must be stage_devices.size() + 1
         */
        std::vector<int> layer_boundaries;

        /**
         * @brief Validate configuration consistency
         * @return true if configuration is valid
         */
        bool isValid() const
        {
            if (stage_devices.empty())
                return false;
            if (layer_boundaries.size() != stage_devices.size() + 1)
                return false;

            // Boundaries must be monotonically increasing
            for (size_t i = 1; i < layer_boundaries.size(); ++i)
            {
                if (layer_boundaries[i] <= layer_boundaries[i - 1])
                    return false;
            }
            return true;
        }

        /**
         * @brief Get number of PP stages
         */
        int numStages() const { return static_cast<int>(stage_devices.size()); }

        /**
         * @brief Get layer range for a stage
         * @param stage Stage index (0-based)
         * @return Pair of (first_layer, last_layer_exclusive)
         */
        std::pair<int, int> layerRangeForStage(int stage) const
        {
            if (stage < 0 || stage >= numStages())
                return {-1, -1};
            return {layer_boundaries[stage], layer_boundaries[stage + 1]};
        }

        /**
         * @brief Get stage that owns a layer
         * @param layer Layer index
         * @return Stage index, or -1 if layer is out of bounds
         */
        int stageForLayer(int layer) const
        {
            for (int s = 0; s < numStages(); ++s)
            {
                if (layer >= layer_boundaries[s] && layer < layer_boundaries[s + 1])
                    return s;
            }
            return -1;
        }
    };

    /**
     * @brief Hierarchical configuration for LocalPPContext with nested contexts
     *
     * Supports PP stages that are either:
     * - Single devices (traditional PP model)
     * - TP domains (PP wrapping TP - common pattern)
     * - Nested PP contexts (future: PP of PP)
     *
     * ## Example: PP(TP(rocm:0, rocm:1), cuda:0, cpu)
     *
     * ```cpp
     * auto tp_ctx = createLocalTPContext({rocm:0, rocm:1}, {}, AUTO);
     *
     * HierarchicalPPConfig config;
     * config.stages = {
     *     PPStage::fromTPContext(tp_ctx),   // Stage 0: TP domain
     *     PPStage::fromDevice(cuda:0),      // Stage 1: single device
     *     PPStage::fromDevice(cpu),         // Stage 2: single device
     * };
     * config.layer_boundaries = {0, 14, 24, 24};  // Stage 0 gets 14 layers, etc.
     *
     * auto pp_ctx = createLocalPPContext(config);
     * pp_ctx->transfer(tensor, 0, 1);  // Automatically handles TP domain → single device
     * ```
     *
     * ## Transfer Semantics
     *
     * When transferring FROM a TP domain:
     * - After TP allreduce, all devices have identical data
     * - Uses representative device (device 0) as source
     * - TP context may provide BAR-backed buffers for cross-vendor transfer
     *
     * When transferring TO a TP domain:
     * - For activations: broadcast to all TP devices
     * - TP context handles internal distribution
     */
    struct HierarchicalPPConfig
    {
        /**
         * @brief PP stages (can be single devices or nested contexts)
         */
        std::vector<PPStage> stages;

        /**
         * @brief Layer boundaries for each stage
         *
         * Same semantics as LocalPPConfig::layer_boundaries.
         * Size must be stages.size() + 1
         */
        std::vector<int> layer_boundaries;

        /**
         * @brief Validate configuration consistency
         */
        bool isValid() const
        {
            if (stages.empty())
                return false;
            if (layer_boundaries.size() != stages.size() + 1)
                return false;

            // Boundaries must be monotonically increasing
            for (size_t i = 1; i < layer_boundaries.size(); ++i)
            {
                if (layer_boundaries[i] < layer_boundaries[i - 1])
                    return false;
            }
            return true;
        }

        int numStages() const { return static_cast<int>(stages.size()); }

        std::pair<int, int> layerRangeForStage(int stage) const
        {
            if (stage < 0 || stage >= numStages())
                return {-1, -1};
            return {layer_boundaries[stage], layer_boundaries[stage + 1]};
        }

        int stageForLayer(int layer) const
        {
            for (int s = 0; s < numStages(); ++s)
            {
                if (layer >= layer_boundaries[s] && layer < layer_boundaries[s + 1])
                    return s;
            }
            return -1;
        }

        /**
         * @brief Get representative device for a stage
         *
         * For single device: returns the device
         * For TP domain: returns device 0 (after allreduce, all have same data)
         */
        GlobalDeviceAddress deviceForStage(int stage) const
        {
            if (stage < 0 || stage >= numStages())
            {
                throw std::out_of_range("Invalid stage index");
            }
            return stages[stage].representativeDevice();
        }

        /**
         * @brief Get the stage object
         */
        const PPStage &stageAt(int stage) const
        {
            if (stage < 0 || stage >= numStages())
            {
                throw std::out_of_range("Invalid stage index");
            }
            return stages[stage];
        }

        /**
         * @brief Convert to flat LocalPPConfig (for backward compatibility)
         *
         * Returns a config with representative devices for each stage.
         * Use this when you need to interface with code that only understands
         * flat device lists.
         */
        LocalPPConfig toFlatConfig() const
        {
            LocalPPConfig flat;
            flat.layer_boundaries = layer_boundaries;
            for (const auto &stage : stages)
            {
                flat.stage_devices.push_back(stage.representativeDevice());
            }
            return flat;
        }

        /**
         * @brief Check if any PP transfer is cross-vendor (CUDA↔ROCm)
         *
         * Cross-vendor transfers require BAR-backed tensors for efficient
         * data movement via PCIe BAR mapping.
         *
         * @return true if any consecutive stages have different GPU vendors
         */
        bool hasCrossVendorTransfers() const
        {
            for (int i = 0; i + 1 < numStages(); ++i)
            {
                const auto &src = stages[i];
                const auto &dst = stages[i + 1];

                // Get representative devices
                DeviceId src_dev = src.representativeDevice().toLocalDeviceId();
                DeviceId dst_dev = dst.representativeDevice().toLocalDeviceId();

                // Check for cross-vendor GPU transfer
                bool src_cuda = src_dev.is_cuda();
                bool src_rocm = src_dev.is_rocm();
                bool dst_cuda = dst_dev.is_cuda();
                bool dst_rocm = dst_dev.is_rocm();

                if ((src_cuda && dst_rocm) || (src_rocm && dst_cuda))
                {
                    return true;
                }
            }
            return false;
        }

        /**
         * @brief Check if stage N outputs to a cross-vendor stage
         *
         * Used to determine if stage N's hidden state tensor should be BAR-backed.
         *
         * @param stage_idx The stage to check
         * @return true if next stage has a different GPU vendor
         */
        bool stageOutputsToCrossVendor(int stage_idx) const
        {
            if (stage_idx < 0 || stage_idx + 1 >= numStages())
            {
                return false; // Last stage or invalid index
            }

            const auto &src = stages[stage_idx];
            const auto &dst = stages[stage_idx + 1];

            DeviceId src_dev = src.representativeDevice().toLocalDeviceId();
            DeviceId dst_dev = dst.representativeDevice().toLocalDeviceId();

            bool src_cuda = src_dev.is_cuda();
            bool src_rocm = src_dev.is_rocm();
            bool dst_cuda = dst_dev.is_cuda();
            bool dst_rocm = dst_dev.is_rocm();

            return (src_cuda && dst_rocm) || (src_rocm && dst_cuda);
        }
    };

    /**
     * @brief Interface for LOCAL pipeline parallelism activation transfer
     *
     * LOCAL PP = multiple PP stages within a single MPI rank.
     * This decouples PP degree from MPI world_size, enabling:
     * - Single-rank multi-GPU pipeline execution
     * - Heterogeneous GPU PP (CUDA + ROCm stages on same rank)
     * - Fine-grained layer distribution across devices
     *
     * Thread safety: All methods are thread-safe. Transfer operations
     * block until data is available on the destination device.
     *
     * Activation transfer pattern:
     * @code
     * // After stage 0 completes layer 7:
     * pp_ctx->transfer(activations, 0, 1);  // stage_from=0, stage_to=1
     * // Now activations are on stage 1's device, ready for layer 8
     * @endcode
     */
    class ILocalPPContext
    {
    public:
        virtual ~ILocalPPContext() = default;

        // =====================================================================
        // Configuration
        // =====================================================================

        /**
         * @brief Get number of PP stages
         * @return Number of pipeline stages in this context
         */
        virtual int numStages() const = 0;

        /**
         * @brief Get device for a PP stage
         *
         * @param stage Stage index (0 to numStages()-1)
         * @return GlobalDeviceAddress for the stage's device
         * @throws std::out_of_range if stage is invalid
         */
        virtual const GlobalDeviceAddress &deviceForStage(int stage) const = 0;

        /**
         * @brief Get backend type for transfer between stages
         *
         * Backend selection based on device types:
         * - Both CUDA devices → NCCL
         * - Both ROCm devices → RCCL
         * - Mixed types → PCIeBAR or HOST
         *
         * @param stage_from Source stage index
         * @param stage_to Destination stage index
         * @return CollectiveBackendType for this transfer path
         */
        virtual CollectiveBackendType backendForTransfer(int stage_from, int stage_to) const = 0;

        /**
         * @brief Get layer range for a stage
         *
         * @param stage Stage index (0 to numStages()-1)
         * @return Pair of (first_layer, last_layer_exclusive)
         */
        virtual std::pair<int, int> layerRangeForStage(int stage) const = 0;

        /**
         * @brief Get stage that owns a layer
         *
         * @param layer Layer index
         * @return Stage index, or -1 if layer is out of bounds
         */
        virtual int stageForLayer(int layer) const = 0;

        /**
         * @brief Get all stage devices
         * @return Vector of GlobalDeviceAddress for all stages
         */
        virtual const std::vector<GlobalDeviceAddress> &stageDevices() const = 0;

        /**
         * @brief Get layer boundaries
         * @return Vector of layer boundary indices
         */
        virtual const std::vector<int> &layerBoundaries() const = 0;

        // =====================================================================
        // Activation Transfer Operations
        // =====================================================================

        /**
         * @brief Transfer activations between PP stages (synchronous)
         *
         * Copies activation tensor from source stage's device to destination
         * stage's device. Blocks until transfer is complete.
         *
         * The tensor must be resident on stage_from's device before calling.
         * After return, the tensor data is available on stage_to's device.
         *
         * @param activations Tensor to transfer (must be on stage_from's device)
         * @param stage_from Source stage index
         * @param stage_to Destination stage index
         * @param active_bytes If non-zero, transfer only this many bytes instead of
         *        the full tensor buffer. Used during decode to avoid transferring
         *        the entire pre-allocated buffer when only a small active region
         *        (e.g., 1 × d_model × sizeof(float)) is needed.
         * @return true on success, false on error
         *
         * @note For same-device stages (stage_from device == stage_to device),
         *       this is a no-op and returns true immediately.
         */
        virtual bool transfer(TensorBase *activations, int stage_from, int stage_to,
                              size_t active_bytes = 0) = 0;

        /**
         * @brief Transfer activations between PP stages (asynchronous)
         *
         * Initiates async copy from source to destination device.
         * Returns immediately; call synchronize() to wait for completion.
         *
         * @param activations Tensor to transfer (must be on stage_from's device)
         * @param stage_from Source stage index
         * @param stage_to Destination stage index
         * @param stream Device-specific stream handle (cudaStream_t, hipStream_t, or nullptr for default)
         * @return true if transfer was initiated, false on error
         *
         * @note The tensor data must not be modified until synchronize() returns.
         * @note For same-device stages, this is a no-op and returns true.
         */
        virtual bool transferAsync(TensorBase *activations, int stage_from, int stage_to, void *stream = nullptr) = 0;

        // =====================================================================
        // Synchronization
        // =====================================================================

        /**
         * @brief Wait for all pending async transfers to complete
         *
         * Blocks until all previously initiated async transfers have finished.
         * Must be called before accessing transferred activation data.
         */
        virtual void synchronize() = 0;

        /**
         * @brief Synchronize a specific stream
         *
         * @param stream Device-specific stream handle to synchronize
         */
        virtual void synchronizeStream(void *stream) = 0;

        // =====================================================================
        // Utility
        // =====================================================================

        /**
         * @brief Check if two stages are on the same device
         *
         * Useful for optimization: skip transfer for stages sharing a device.
         *
         * @param stage_a First stage index
         * @param stage_b Second stage index
         * @return true if both stages are on the same device
         */
        virtual bool sameDevice(int stage_a, int stage_b) const = 0;

        /**
         * @brief Get total number of layers across all stages
         * @return Total layer count (layer_boundaries.back())
         */
        virtual int totalLayers() const = 0;

        /**
         * @brief Reserve temporary buffer capacity for transfer operations
         *
         * Pre-allocates internal staging buffers to avoid allocation in the hot path.
         * Useful for HOST backend which may need CPU staging memory.
         *
         * @param bytes Minimum buffer capacity in bytes
         * @return true if reservation succeeded
         */
        virtual bool reserveStagingBufferBytes(size_t bytes) = 0;
    };

    /**
     * @brief Factory function to create a LocalPPContext
     *
     * Creates an implementation of ILocalPPContext based on the provided
     * configuration. Backend selection is automatic based on device types:
     * - Homogeneous CUDA → NCCL for all transfers
     * - Homogeneous ROCm → RCCL for all transfers
     * - Heterogeneous → PCIeBAR or HOST depending on topology
     *
     * @param config PP configuration (devices and layer boundaries)
     * @return Unique pointer to ILocalPPContext implementation
     * @throws std::invalid_argument if config.isValid() returns false
     */
    std::unique_ptr<ILocalPPContext> createLocalPPContext(const LocalPPConfig &config);

    /**
     * @brief Factory function to create a hierarchical LocalPPContext
     *
     * Creates an implementation of ILocalPPContext that supports nested
     * parallelism contexts (TP domains, nested PP).
     *
     * Key differences from flat createLocalPPContext():
     * - Stages can be TP domains (multiple devices with shared state)
     * - Transfer operations automatically handle TP context boundaries
     * - BAR-backed tensor handling delegated to TP context
     *
     * ## Usage
     *
     * ```cpp
     * // Create TP context first
     * auto tp_ctx = createLocalTPContext({rocm:0, rocm:1}, {}, AUTO);
     *
     * // Build hierarchical config
     * HierarchicalPPConfig config;
     * config.stages = {
     *     PPStage::fromTPContext(tp_ctx),
     *     PPStage::fromDevice(cuda:0),
     *     PPStage::fromDevice(cpu),
     * };
     * config.layer_boundaries = {0, 14, 24, 24};
     *
     * // Create PP context
     * auto pp_ctx = createLocalPPContext(config);
     *
     * // Transfer from TP domain → single device (handles BAR automatically)
     * pp_ctx->transfer(activations, 0, 1);
     * ```
     *
     * @param config Hierarchical PP configuration
     * @return Unique pointer to ILocalPPContext implementation
     * @throws std::invalid_argument if config.isValid() returns false
     */
    std::unique_ptr<ILocalPPContext> createLocalPPContext(const HierarchicalPPConfig &config);

} // namespace llaminar2
