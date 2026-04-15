/**
 * @file SharedWeightPool.h
 * @brief Single-load weight pool that owns all host-resident tensors
 *
 * SharedWeightPool loads a GGUF model exactly once and provides lightweight
 * WeightViewSet accessors for each PP stage. This eliminates the N+1 GGUF
 * parse and independent weight loading that occurs when each PP stage
 * creates its own ModelContext.
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#pragma once

#include "ISharedWeightPool.h"
#include "ModelLoader.h"
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Concrete implementation of ISharedWeightPool
     *
     * Owns a single ModelLoader and all loaded TensorBase instances.
     * After loadFromGGUF() completes, the pool is immutable and thread-safe.
     */
    class SharedWeightPool : public ISharedWeightPool
    {
    public:
        SharedWeightPool();
        ~SharedWeightPool() override;

        // Disable copy, allow move
        SharedWeightPool(const SharedWeightPool &) = delete;
        SharedWeightPool &operator=(const SharedWeightPool &) = delete;
        SharedWeightPool(SharedWeightPool &&) = default;
        SharedWeightPool &operator=(SharedWeightPool &&) = default;

        /**
         * @brief Create a new SharedWeightPool instance
         */
        static std::unique_ptr<SharedWeightPool> create();

        // =========================================================================
        // ISharedWeightPool Implementation
        // =========================================================================

        bool loadFromGGUF(const std::string &model_path, bool use_mmap = true) override;
        bool isLoaded() const override { return loaded_; }

        WeightViewSet createViewSet(
            int first_layer, int last_layer,
            bool has_embedding, bool has_lm_head) const override;

        std::shared_ptr<TensorBase> getTensor(const std::string &name) const override;
        bool hasTensor(const std::string &name) const override;
        std::vector<std::string> tensorNames() const override;
        size_t tensorCount() const override;

        const PoolModelMetadata &metadata() const override { return metadata_; }

        size_t totalHostBytes() const override;

        bool isGemmWeight(const std::string &name) const override;

        // =========================================================================
        // Additional Accessors
        // =========================================================================

        /**
         * @brief Get the underlying ModelLoader (for advanced use cases)
         *
         * Primarily useful for creating ModelContext wrappers during migration.
         */
        ModelLoader &loader() { return loader_; }
        const ModelLoader &loader() const { return loader_; }

    private:
        /**
         * @brief Check if a weight name falls within a layer range
         *
         * Reuses the same logic as WeightManager::isWeightInLayerRange().
         */
        static bool isWeightInLayerRange(
            const std::string &name,
            int first_layer, int last_layer,
            bool has_embedding, bool has_lm_head);

        /**
         * @brief Extract the layer index from a weight name (e.g., "blk.5.attn_q.weight" → 5)
         * @return Layer index, or -1 for non-layer weights
         */
        static int extractLayerIndex(const std::string &name);

        /**
         * @brief Check if a weight is a non-GEMM weight (norm, bias, embedding, etc.)
         *
         * Mirrors GraphSchema::WeightShardingConfig::isNonGemmWeight() logic.
         */
        static bool isNonGemmWeight(const std::string &name);

        /**
         * @brief Populate metadata_ from loaded model
         */
        void extractMetadata();

        /**
         * @brief Load all weight tensors from the model into the pool
         */
        bool loadAllTensors();

        ModelLoader loader_;
        PoolModelMetadata metadata_;
        bool loaded_ = false;

        /// All loaded weight tensors, keyed by GGUF tensor name
        std::unordered_map<std::string, std::shared_ptr<TensorBase>> tensors_;
    };

} // namespace llaminar2
