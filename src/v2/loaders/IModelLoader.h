/**
 * @file IModelLoader.h
 * @brief Interface for loading model weights from storage
 *
 * Abstracts model file access to enable:
 * 1. Unit testing without GGUF files
 * 2. In-memory test models
 * 3. Different model formats (GGUF, SafeTensors, etc.)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../backends/DeviceId.h"
#include "../execution/config/RuntimeConfig.h"
#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <cstdint>
#include <cstddef>

namespace llaminar2
{

    // Forward declaration - TensorBase is the base tensor type
    class TensorBase;

    /**
     * @brief Abstract interface for model loading operations
     *
     * This interface abstracts model file access to enable:
     * - Unit testing without actual GGUF files on disk
     * - In-memory mock models for fast testing
     * - Different model format backends (GGUF, SafeTensors, etc.)
     * - Deterministic testing of weight loading logic
     *
     * Implementations:
     * - ModelLoader: Real GGUF file-backed implementation
     * - MockModelLoader: Test implementation with in-memory tensors
     *
     * Usage:
     * @code
     * // Production code works with interface
     * void loadWeights(IModelLoader& loader) {
     *     auto embd = loader.getTensor("token_embd.weight", DeviceId::cpu());
     *     auto q = loader.getTensor("blk.0.attn_q.weight", DeviceId::cpu());
     * }
     *
     * // Test code uses mock
     * auto mock = MockModelLoaderBuilder()
     *     .setArchitecture("qwen2")
     *     .setBlockCount(24)
     *     .addFP32Tensor("token_embd.weight", {151936, 896})
     *     .addQ4_0Tensor("blk.0.attn_q.weight", {896, 896})
     *     .build();
     * loadWeights(*mock);
     * @endcode
     */
    class IModelLoader
    {
    public:
        virtual ~IModelLoader() = default;

        // =========================================================================
        // Loading / Initialization
        // =========================================================================

        /**
         * @brief Check if model has been successfully loaded
         * @return true if model is ready for tensor access
         */
        virtual bool isLoaded() const = 0;

        // =========================================================================
        // Tensor Access
        // =========================================================================

        /**
         * @brief Load a tensor from the model
         *
         * For real loaders, this reads from disk. For mock loaders, this returns
         * pre-configured in-memory tensors.
         *
         * @param name Tensor name (e.g., "token_embd.weight", "blk.0.attn_q.weight")
         * @param device Target device for the tensor (default: CPU)
         * @param weight_precision How to load the weight (NATIVE keeps original format)
         * @return Loaded tensor or nullptr if not found
         */
        virtual std::shared_ptr<TensorBase> loadTensor(
            const std::string &name,
            DeviceId device = DeviceId::cpu(),
            WeightPrecision weight_precision = WeightPrecision::NATIVE) = 0;

        /**
         * @brief Load a row slice of a tensor (for tensor parallelism)
         *
         * Only reads/returns rows [row_start, row_end) for memory efficiency.
         * Used for row-parallel weight sharding.
         *
         * @param name Tensor name
         * @param row_start First row to load (0-indexed)
         * @param row_end One past the last row to load
         * @param device Target device
         * @param weight_precision Weight precision mode
         * @return Tensor with shape [row_end - row_start, cols] or nullptr
         */
        virtual std::shared_ptr<TensorBase> loadTensorRowSlice(
            const std::string &name,
            size_t row_start, size_t row_end,
            DeviceId device = DeviceId::cpu(),
            WeightPrecision weight_precision = WeightPrecision::NATIVE) = 0;

        /**
         * @brief Load a column slice of a tensor (for tensor parallelism)
         *
         * Only reads/returns columns [col_start, col_end) for memory efficiency.
         * Used for column-parallel weight sharding.
         *
         * @param name Tensor name
         * @param col_start First column to load (0-indexed)
         * @param col_end One past the last column to load
         * @param device Target device
         * @param weight_precision Weight precision mode
         * @return Tensor with shape [rows, col_end - col_start] or nullptr
         */
        virtual std::shared_ptr<TensorBase> loadTensorColumnSlice(
            const std::string &name,
            size_t col_start, size_t col_end,
            DeviceId device = DeviceId::cpu(),
            WeightPrecision weight_precision = WeightPrecision::NATIVE) = 0;

        /**
         * @brief Load an expert slice of a 3D MoE tensor (for expert parallelism)
         *
         * Only reads/returns experts [expert_start, expert_end) from a 3D tensor
         * with shape [ne0, ne1, num_experts]. Used for MoE expert parallelism
         * where each rank loads only its assigned expert subset.
         *
         * @param name Tensor name (must be 3D: [cols, rows_per_expert, num_experts])
         * @param expert_start First expert to load (0-indexed)
         * @param expert_end One past the last expert to load
         * @param device Target device
         * @param weight_precision Weight precision mode
         * @return Tensor with shape [ne0, ne1, expert_end - expert_start] or nullptr
         */
        virtual std::shared_ptr<TensorBase> loadTensorExpertSlice(
            const std::string &name,
            size_t expert_start, size_t expert_end,
            DeviceId device = DeviceId::cpu(),
            WeightPrecision weight_precision = WeightPrecision::NATIVE)
        {
            // Default: not implemented. Override in ModelLoader for GGUF 3D tensors.
            (void)name; (void)expert_start; (void)expert_end; (void)device; (void)weight_precision;
            return nullptr;
        }

        /**
         * @brief Check if a tensor exists in the model
         * @param name Tensor name
         * @return true if tensor exists
         */
        virtual bool hasTensor(const std::string &name) const = 0;

        /**
         * @brief Get tensor shape without loading the full tensor
         *
         * Returns the dimensions of the tensor for slice calculation without
         * loading the entire tensor into memory.
         *
         * @param name Tensor name
         * @return Optional vector of dimensions, or std::nullopt if tensor not found
         */
        virtual std::optional<std::vector<size_t>> getTensorShape(const std::string &name) const = 0;

        /**
         * @brief Get names of all tensors in the model
         * @return Vector of tensor names
         */
        virtual std::vector<std::string> tensorNames() const = 0;

        // =========================================================================
        // Model Metadata
        // =========================================================================

        /**
         * @brief Get model architecture identifier
         * @return Architecture string (e.g., "qwen2", "llama", "gpt2")
         */
        virtual std::string architecture() const = 0;

        /**
         * @brief Get total number of tensors in the model
         */
        virtual size_t tensorCount() const = 0;

        /**
         * @brief Get total model size in bytes (all tensor data)
         */
        virtual size_t totalBytes() const = 0;

        // =========================================================================
        // Model Hyperparameters (Config)
        // =========================================================================

        /**
         * @brief Get integer hyperparameter value
         * @param key Parameter key (e.g., "block_count", "head_count")
         * @param default_val Default value if key not found
         * @return Parameter value
         */
        virtual int getInt(const std::string &key, int default_val = 0) const = 0;

        /**
         * @brief Get 64-bit integer hyperparameter value
         * @param key Parameter key
         * @param default_val Default value if key not found
         * @return Parameter value
         */
        virtual uint64_t getUInt64(const std::string &key, uint64_t default_val = 0) const = 0;

        /**
         * @brief Get float hyperparameter value
         * @param key Parameter key (e.g., "rope_theta", "rms_norm_eps")
         * @param default_val Default value if key not found
         * @return Parameter value
         */
        virtual float getFloat(const std::string &key, float default_val = 0.0f) const = 0;

        /**
         * @brief Get string hyperparameter value
         * @param key Parameter key
         * @param default_val Default value if key not found
         * @return Parameter value
         */
        virtual std::string getString(const std::string &key, const std::string &default_val = "") const = 0;

        // =========================================================================
        // Convenience Accessors (Common Hyperparameters)
        // =========================================================================

        /**
         * @brief Get number of transformer layers
         */
        virtual uint64_t blockCount() const = 0;

        /**
         * @brief Get hidden dimension (d_model)
         */
        virtual uint64_t embeddingLength() const = 0;

        /**
         * @brief Get number of attention heads
         */
        virtual uint64_t headCount() const = 0;

        /**
         * @brief Get number of key-value heads (for GQA)
         */
        virtual uint64_t headCountKV() const = 0;

        /**
         * @brief Get vocabulary size
         */
        virtual uint64_t vocabSize() const = 0;

        /**
         * @brief Get maximum context length
         */
        virtual uint64_t contextLength() const = 0;

        /**
         * @brief Get feed-forward hidden dimension (intermediate_size)
         *
         * For Qwen2/Llama models, this is the FFN intermediate dimension.
         * Read from metadata key "llama.feed_forward_length" or "qwen2.feed_forward_length".
         *
         * @return FFN intermediate dimension, or 0 if not available
         */
        virtual uint64_t feedForwardLength() const = 0;

        /**
         * @brief Get attention head dimension (key_length from GGUF)
         *
         * For models where head_dim != d_model / n_heads (e.g., Qwen3 with head_dim=128
         * but d_model=1024, n_heads=16). Returns 0 if not explicitly set in metadata,
         * in which case the caller should fall back to d_model / n_heads.
         *
         * @return Head dimension, or 0 if not explicitly set
         */
        virtual uint64_t keyLength() const = 0;

        /**
         * @brief Get RoPE base frequency (theta)
         */
        virtual float ropeTheta() const = 0;

        /**
         * @brief Get RMSNorm epsilon
         */
        virtual float rmsNormEps() const = 0;

        /**
         * @brief Release mmap regions to free mapped file memory.
         *
         * Called after all weight data has been consumed (packed into kernel
         * storage or copied into BufferArena). Only the concrete loader
         * implementation needs to act; mock loaders can no-op.
         */
        virtual void releaseMmapRegions() {}

        /**
         * @brief Advise the OS to reclaim physical pages backing the mmap regions.
         *
         * Uses madvise(MADV_DONTNEED) to release physical pages without
         * unmapping the virtual address range. Future reads re-fault from
         * the page cache. Safe to call after all GEMM weights have been
         * packed into interleaved format.
         *
         * @return Total bytes advised
         */
        virtual size_t adviseMmapDontneed() { return 0; }
    };

} // namespace llaminar2
