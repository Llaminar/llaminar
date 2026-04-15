/**
 * @file ISharedWeightPool.h
 * @brief Interface for shared weight storage across PP stages
 *
 * ISharedWeightPool abstracts a single-load, many-view weight storage system.
 * The pool loads a GGUF model once and provides lightweight WeightViewSet
 * accessors for each PP stage, enabling weight sharing without duplication.
 *
 * Key properties:
 * - Loads model weights exactly once from disk
 * - Owns all host-resident tensor data
 * - Provides read-only views filtered by layer range
 * - Thread-safe after initialization (read-only access)
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#pragma once

#include "WeightViewSet.h"
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class TensorBase;

    /**
     * @brief Model metadata extracted from GGUF during loading
     *
     * Provides all hyperparameters needed by graph builders and stage runners
     * without requiring access to the full ModelLoader or GGUF structures.
     */
    struct PoolModelMetadata
    {
        std::string architecture;         ///< e.g., "qwen2", "llama", "qwen3"
        std::string model_path;           ///< Path to the GGUF file

        int n_layers = 0;                 ///< Total transformer layers in the model
        int d_model = 0;                  ///< Hidden dimension (embedding length)
        int n_heads = 0;                  ///< Number of query attention heads
        int n_kv_heads = 0;               ///< Number of KV heads (GQA)
        int head_dim = 0;                 ///< Dimension per head (d_model / n_heads or explicit)
        int d_ff = 0;                     ///< FFN intermediate dimension
        int vocab_size = 0;               ///< Vocabulary size
        int context_length = 0;           ///< Maximum context length

        // GDN-specific dimensions (optional)
        int gdn_n_k_heads = 0;            ///< GDN key heads
        int gdn_n_v_heads = 0;            ///< GDN value heads
        int gdn_d_state = 0;              ///< GDN state dimension

        bool hasGDN() const { return gdn_n_k_heads > 0 && gdn_n_v_heads > 0; }
    };

    /**
     * @brief Interface for shared weight pool — load once, view many
     *
     * Implementations own the GGUF loader and all host-resident weight tensors.
     * PP stages receive WeightViewSet objects that reference (not copy) these tensors.
     *
     * Lifecycle:
     * 1. Create pool via factory
     * 2. loadFromGGUF() — parse + load all weights into host memory
     * 3. createViewSet() — create per-stage views (repeated per PP stage)
     * 4. [Optional] releaseHostData() — free host memory after GPU upload
     * 5. Destructor frees all remaining tensors
     *
     * Thread safety: After loadFromGGUF() completes, all read methods are thread-safe.
     *
     * Usage:
     * @code
     * auto pool = SharedWeightPool::create();
     * pool->loadFromGGUF("model.gguf");
     *
     * // Create views for 2-stage PP
     * auto views0 = pool->createViewSet(0, 12, true, false);   // Stage 0: embedding + layers 0-11
     * auto views1 = pool->createViewSet(12, 24, false, true);  // Stage 1: layers 12-23 + LM head
     *
     * // Both views reference the SAME underlying tensors
     * @endcode
     */
    class ISharedWeightPool
    {
    public:
        virtual ~ISharedWeightPool() = default;

        // =========================================================================
        // Loading
        // =========================================================================

        /**
         * @brief Load a GGUF model file into the pool
         *
         * Parses the GGUF header, metadata, and loads all weight tensors into
         * host memory. Uses mmap by default for efficient I/O.
         *
         * @param model_path Path to the GGUF model file
         * @param use_mmap Whether to use memory-mapped I/O (default: true)
         * @return true on success
         */
        virtual bool loadFromGGUF(const std::string &model_path, bool use_mmap = true) = 0;

        /**
         * @brief Whether the pool has been loaded
         */
        virtual bool isLoaded() const = 0;

        // =========================================================================
        // View Creation
        // =========================================================================

        /**
         * @brief Create a WeightViewSet for a PP stage's layer range
         *
         * Returns a lightweight view containing only the weights assigned to the
         * specified layer range. Views reference (not copy) pool-owned tensors.
         *
         * @param first_layer First layer index (inclusive)
         * @param last_layer Last layer index (exclusive)
         * @param has_embedding Whether this stage owns the embedding table
         * @param has_lm_head Whether this stage owns the output norm + LM head
         * @return WeightViewSet with filtered weight views
         */
        virtual WeightViewSet createViewSet(
            int first_layer, int last_layer,
            bool has_embedding, bool has_lm_head) const = 0;

        // =========================================================================
        // Direct Access
        // =========================================================================

        /**
         * @brief Get a weight tensor by name
         * @return Shared pointer to the tensor, or nullptr if not found
         */
        virtual std::shared_ptr<TensorBase> getTensor(const std::string &name) const = 0;

        /**
         * @brief Check whether a tensor exists in the pool
         */
        virtual bool hasTensor(const std::string &name) const = 0;

        /**
         * @brief Get all weight names in the pool
         */
        virtual std::vector<std::string> tensorNames() const = 0;

        /**
         * @brief Get the total number of loaded tensors
         */
        virtual size_t tensorCount() const = 0;

        // =========================================================================
        // Metadata
        // =========================================================================

        /**
         * @brief Get model metadata extracted from GGUF
         */
        virtual const PoolModelMetadata &metadata() const = 0;

        // =========================================================================
        // Memory Management
        // =========================================================================

        /**
         * @brief Get total host memory used by weight tensors (bytes)
         */
        virtual size_t totalHostBytes() const = 0;

        /**
         * @brief Check if a weight is a GEMM weight (needs device-specific repacking)
         *
         * @param name Weight name
         * @return true if the weight requires GEMM repacking for execution
         */
        virtual bool isGemmWeight(const std::string &name) const = 0;
    };

} // namespace llaminar2
