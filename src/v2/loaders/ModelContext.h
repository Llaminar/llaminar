/**
 * @file ModelContext.h
 * @brief Model context for pipeline initialization
 *
 * Wraps GGUF model metadata and provides convenient access for pipelines.
 * Eliminates need for pipelines to hardcode architecture parameters.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "ModelLoader.h"
#include "WeightManager.h"
#include "WeightPlacementMap.h"
#include "ModelContextConfig.h"
#include "../interfaces/IModelContext.h"
#include "../backends/DeviceId.h"
#include "../utils/MPIContext.h"
#include "../tensors/Tensors.h"
#include <memory>
#include <string>

namespace llaminar2
{

    /**
     * @brief Model context for pipeline initialization
     *
     * Implements IModelContext interface for production use with GGUF files.
     *
     * Contains:
     * - Model file path
     * - Parsed GGUF metadata (architecture, hyperparameters)
     * - ModelLoader for on-demand tensor loading
     * - WeightManager for distributed weight allocation
     *
     * Usage:
     *   auto ctx = ModelContext::create("model.gguf", mpi_ctx);
     *   auto pipeline = PipelineFactory::create(ctx->architecture(), ctx, mpi_ctx, device_idx);
     *   // Pipeline reads hyperparameters from ctx->model()
     *   // Pipeline loads weights via ctx->getWeight(name, device_idx)
     */
    class ModelContext : public IModelContext
    {
    public:
        /**
         * @brief Create model context from GGUF file
        /**
         * @param model_path Path to GGUF model file
         * @param mpi_ctx MPI context for distributed weight management (nullptr = single rank)
         * @param placement_map Fine-grained weight placement decisions (nullptr = default all to device 0)
         * @param factory Optional TensorFactory for NUMA-aware allocation
         * @param strategy Weight distribution strategy (default: REPLICATED)
         * @param weight_precision Weight loading precision (affects dequantization, default: NATIVE)
         * @return Shared pointer to context, or nullptr on error
         */
        static std::shared_ptr<ModelContext> create(
            const std::string &model_path,
            std::shared_ptr<MPIContext> mpi_ctx = nullptr,
            std::shared_ptr<WeightPlacementMap> placement_map = nullptr,
            TensorFactory *factory = nullptr,
            WeightDistributionStrategy strategy = WeightDistributionStrategy::REPLICATED,
            WeightPrecision weight_precision = WeightPrecision::NATIVE);

        /**
         * @brief Create model context with unified configuration
         *
         * This is the preferred factory method. Supports all scenarios:
         * - Single device (default config)
         * - TP (config.total_shards > 1)
         * - PP (config.first_layer != 0 or config.last_layer != -1)
         * - Combined TP + PP
         *
         * @param model_path Path to GGUF model file
         * @param config Configuration for weight loading
         * @return Shared pointer to context, or nullptr on error
         */
        static std::shared_ptr<ModelContext> create(
            const std::string &model_path,
            const ModelContextConfig &config);

        /**
         * @brief Create model context for a Pipeline Parallelism stage
         *
         * Creates a ModelContext with LAYER_PARTITIONED strategy that only loads
         * weights for the specified layer range. This reduces memory usage by
         * ~50% for 2-stage PP.
         *
         * Layer range is [first_layer, last_layer) - first inclusive, last exclusive.
         *
         * Special weights:
         * - has_embedding: if true, token embedding is loaded (typically stage 0)
         * - has_lm_head: if true, output norm and LM head are loaded (typically last stage)
         *
         * @param model_path Path to GGUF model file
         * @param first_layer First layer index (inclusive)
         * @param last_layer Last layer index (exclusive)
         * @param has_embedding True if this stage should load embedding
         * @param has_lm_head True if this stage should load output norm and LM head
         * @param mpi_ctx MPI context (optional)
         * @param weight_precision Weight loading precision (default: NATIVE)
         * @return Shared pointer to context, or nullptr on error
         */
        static std::shared_ptr<ModelContext> createForPPStage(
            const std::string &model_path,
            int first_layer,
            int last_layer,
            bool has_embedding,
            bool has_lm_head,
            std::shared_ptr<MPIContext> mpi_ctx = nullptr,
            WeightPrecision weight_precision = WeightPrecision::NATIVE);

        /**
         * @brief Create test-only model context (doesn't load actual model)
         *
         * For unit tests that need a ModelContext but don't need real model data.
         * Initializes a minimal valid GGUFModel structure to prevent undefined behavior.
         *
         * @param model_path Dummy path (can be anything)
         * @param mpi_ctx Optional MPI context for multi-rank tests (use nullptr for single-rank)
         * @return Shared pointer to context (always succeeds)
         */
        static std::shared_ptr<ModelContext> createForTesting(
            const std::string &model_path = "test.gguf",
            std::shared_ptr<MPIContext> mpi_ctx = nullptr,
            uint32_t block_count = 1);

        // =========================================================================
        // IModelContext Implementation
        // =========================================================================

        /**
         * @brief Get model file path
         */
        const std::string &path() const override { return model_path_; }

        /**
         * @brief Get GGUF model metadata
         */
        const GGUFModel &model() const { return loader_.getModel(); }

        /**
         * @brief Get architecture string
         */
        const std::string &architecture() const override { return loader_.getModel().architecture; }

        /**
         * @brief Get ModelLoader interface for tensor loading
         */
        std::shared_ptr<IModelLoader> loader() override { return loader_interface_; }

        /**
         * @brief Get concrete ModelLoader for internal use
         */
        ModelLoader &concreteLoader() { return loader_; }
        const ModelLoader &concreteLoader() const { return loader_; }

        /**
         * @brief Get weight tensor by name (shared instance)
         *
         * Loads from GGUF with appropriate distribution strategy.
         *
         * @param name GGUF tensor name (e.g., "token_embd.weight", "blk.0.attn_q.weight")
         * @param device Device for tensor placement (default: CPU)
         * @return Shared pointer to tensor, or nullptr on error
         */
        std::shared_ptr<TensorBase> getWeight(const std::string &name, DeviceId device = DeviceId::cpu()) override
        {
            return weight_manager_->getWeight(name, device);
        }

        /**
         * @brief Get weight tensor for a specific device (device-isolated instance)
         *
         * For multi-device scenarios (LOCAL TP), each device needs its own tensor
         * instance to track coherence state independently.
         *
         * @param name GGUF tensor name
         * @param device Target device for this tensor instance
         * @return Device-specific tensor instance, or nullptr on error
         */
        std::shared_ptr<TensorBase> getWeightForDevice(const std::string &name, DeviceId device) override
        {
            return weight_manager_->getWeightForDevice(name, device);
        }

        /**
         * @brief Check if a tensor exists in the model
         * @param name GGUF tensor name to check
         * @return true if tensor exists, false otherwise
         */
        bool hasTensor(const std::string &name) const override
        {
            return loader_.hasTensor(name);
        }

        /**
         * @brief Get weight manager interface (for IModelContext)
         */
        std::shared_ptr<IWeightManager> weightManager() override { return weight_manager_; }

        /**
         * @brief Get concrete weight manager for production code
         *
         * Use this when you need WeightManager-specific methods not on the interface.
         */
        std::shared_ptr<WeightManager> concreteWeightManager() { return weight_manager_; }

        // =========================================================================
        // Convenience Hyperparameter Accessors (IModelContext)
        // =========================================================================

        int blockCount() const override
        {
            // For PP stages, return the layer count for this stage (not total model layers)
            if (pp_block_count_override_ >= 0)
            {
                return pp_block_count_override_;
            }
            return static_cast<int>(loader_.blockCount());
        }
        int embeddingLength() const override { return static_cast<int>(loader_.embeddingLength()); }
        int headCount() const override { return static_cast<int>(loader_.headCount()); }
        int headCountKV() const override { return static_cast<int>(loader_.headCountKV()); }
        int vocabSize() const override { return static_cast<int>(loader_.vocabSize()); }
        int contextLength() const override { return static_cast<int>(loader_.contextLength()); }
        int feedForwardLength() const override { return static_cast<int>(loader_.feedForwardLength()); }

        /**
         * @brief Get weight placement map (Phase 6: Multi-GPU support)
         *
         * Returns the placement map from the weight manager.
         * @return Shared pointer to placement map (may be nullptr if single device)
         */
        std::shared_ptr<WeightPlacementMap> placementMap() const
        {
            return weight_manager_ ? weight_manager_->placementMap() : nullptr;
        }

    private:
        // Private constructor - use create() factory method
    private:
        /**
         * @brief Private constructor (use create() instead)
         */
        ModelContext(const std::string &model_path,
                     std::shared_ptr<MPIContext> mpi_ctx,
                     std::shared_ptr<WeightPlacementMap> placement_map = nullptr,
                     TensorFactory *factory = nullptr,
                     WeightDistributionStrategy strategy = WeightDistributionStrategy::REPLICATED);

        std::string model_path_;
        std::unique_ptr<TensorFactory> owned_test_factory_; // For createForTesting() only - must be declared BEFORE loader_!
        ModelLoader loader_;
        std::shared_ptr<WeightManager> weight_manager_;

        // Interface wrapper for loader (since loader_ is stored by value, not as shared_ptr)
        std::shared_ptr<IModelLoader> loader_interface_;

        // PP stage layer count override (-1 = use loader's blockCount)
        // Set by createForPPStage() to return correct layer count for nested MDO graph building
        int pp_block_count_override_ = -1;
    };

} // namespace llaminar2
