/**
 * @file ModelLoader.h
 * @brief Streamlined GGUF model loader for Llaminar V2
 * @author David Sanftenberg
 *
 * Loads transformer models from GGUF format files with:
 *  - Essential GGUF parsing (magic, version, metadata, tensor info)
 *  - Core quantization formats (F32, F16, Q4_0, Q6_K, Q8_0, IQ4_NL)
 *  - Simple tensor loading API for pipeline integration
 *  - Device-aware tensor creation (CPU, CUDA, ROCm)
 *  - INT8 dequantization mode (for AVX512-VNNI / CUDA INT8 GEMM)
 *
 * Simplified from V1 ModelLoader (3870 lines) to focus on:
 *  - Essential GGUF features (no MPI column slicing, no caching)
 *  - V2 architecture patterns (device affinity, ITensor interfaces)
 *  - Common quantization formats (excludes rare IQ variants)
 *
 * Key Differences from V1:
 *  - No MPI column sharding (V2 uses different distribution strategy)
 *  - No tensor caching (pipelines manage weight lifecycle)
 *  - Optional dequantization to INT8 (when --weight-precision int8 is set)
 *  - Returns V2 TensorBase (with device affinity, ITensor interfaces)
 */

#pragma once

#include "IModelLoader.h" // Interface
#include "MmapRegion.h"   // RAII mmap wrapper for zero-copy loading
#include "../backends/DeviceId.h"
#include "../execution/config/RuntimeConfig.h" // for WeightPrecision
#include "../tensors/TensorFactory.h"          // for owned_factory_
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class TensorBase;
    class IMPIContext;

    // =============================================================================
    // GGUF TYPE DEFINITIONS
    // =============================================================================

    /**
     * @brief GGUF metadata value types
     */
    enum class GGUFValueType : uint32_t
    {
        UINT8 = 0,
        INT8 = 1,
        UINT16 = 2,
        INT16 = 3,
        UINT32 = 4,
        INT32 = 5,
        FLOAT32 = 6,
        BOOL = 7,
        STRING = 8,
        ARRAY = 9,
        UINT64 = 10,
        INT64 = 11,
        FLOAT64 = 12
    };

    /**
     * @brief GGUF tensor types (quantization formats)
     *
     * V2 supports subset of V1 formats (common formats only):
     *  - F32: Unquantized 32-bit float (baseline)
     *  - F16: Half precision 16-bit float (2× compression)
     *  - Q4_0: 4-bit quantization (8× compression)
     *  - Q6_K: 6-bit K-quant (5.3× compression)
     *  - Q8_0: 8-bit quantization (4× compression)
     *  - IQ4_NL: 4-bit importance-weighted quantization
     *
     * Also defined but not yet implemented (will log warning on load):
     *  - Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q8_K
     *  - IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ3_S, IQ2_S, IQ4_XS, IQ1_M
     */
    enum class GGUFTensorType : uint32_t
    {
        F32 = 0,
        F16 = 1,
        Q4_0 = 2,
        BF16 = 30,
        Q4_1 = 3,
        Q5_0 = 6,
        Q5_1 = 7,
        Q8_0 = 8,
        Q2_K = 10,
        Q3_K = 11,
        Q4_K = 12,
        Q5_K = 13,
        Q6_K = 14,
        Q8_K = 15,
        IQ2_XXS = 16,
        IQ2_XS = 17,
        IQ3_XXS = 18,
        IQ1_S = 19,
        IQ4_NL = 20,
        IQ3_S = 21,
        IQ2_S = 22,
        IQ4_XS = 23,
        IQ1_M = 29
    };

    /**
     * @brief GGUF metadata value (variant type)
     */
    struct GGUFValue
    {
        GGUFValueType type;
        std::vector<uint8_t> data;
        uint64_t array_length = 0; // For ARRAY type, stores number of elements

        // String array storage (for tokenizer vocabulary, merges, etc.)
        std::vector<std::string> string_array_value;

        // Typed accessors
        uint32_t asUInt32() const;
        uint64_t asUInt64() const;
        float asFloat32() const;
        std::string asString() const;
        uint64_t asArrayLength() const { return array_length; }
        const std::vector<std::string> &asStringArray() const { return string_array_value; }
    };

    /**
     * @brief GGUF tensor metadata
     */
    struct GGUFTensorInfo
    {
        std::string name;
        std::vector<uint64_t> dimensions;
        GGUFTensorType type;
        uint64_t offset;     // Byte offset from data_offset (NOT file start!)
        uint64_t size_bytes; // Total size in bytes
        uint16_t split_idx;  // Index of split file containing this tensor (0 = main file)

        // Helper methods
        bool isQuantized() const;
        size_t getTypeSize() const;  // Bytes per block
        size_t getBlockSize() const; // Elements per block
    };

    /**
     * @brief Parsed GGUF model metadata
     */
    struct GGUFModel
    {
        // Header
        uint32_t version = 0;
        uint64_t tensor_count = 0;
        uint64_t metadata_kv_count = 0;
        uint32_t alignment = 32;

        // Metadata
        std::string architecture;      // "qwen2", "llama", etc.
        uint64_t context_length = 0;   // max_position_embeddings
        uint64_t embedding_length = 0; // hidden_size (d_model)
        uint64_t block_count = 0;      // num_hidden_layers
        uint64_t head_count = 0;       // num_attention_heads
        uint64_t head_count_kv = 0;    // num_key_value_heads (GQA)
        uint64_t key_length = 0;       // attention.key_length (head_dim); 0 = use d_model/n_heads
        uint64_t value_length = 0;     // attention.value_length; 0 = use key_length
        uint64_t vocab_size = 0;       // vocabulary size
        float rope_theta = 10000.0f;   // RoPE base frequency
        float rms_norm_eps = 1e-6f;    // RMSNorm epsilon (default 1e-6 for Qwen2/LLaMA)

        std::map<std::string, GGUFValue> metadata;
        std::vector<GGUFTensorInfo> tensors;

        uint64_t data_offset = 0; // Byte offset to tensor data region

        // Multi-part GGUF support
        uint16_t split_count = 1;                 // Number of split files (1 = no split)
        uint16_t split_no = 0;                    // This file's split index (0-based)
        std::vector<std::string> split_paths;     // Paths to all split files
        std::vector<uint64_t> split_data_offsets; // Data offset for each split

        // Helper methods
        bool hasMetadata(const std::string &key) const;
        GGUFTensorInfo *findTensor(const std::string &name);
        const GGUFTensorInfo *findTensor(const std::string &name) const;
    };

    // =============================================================================
    // GGUF LOADER
    // =============================================================================

    /**
     * @brief Streamlined GGUF model loader for V2
     *
     * Implements IModelLoader interface for production GGUF file loading.
     *
     * Usage:
     *   ModelLoader loader;
     *   if (!loader.loadModel("model.gguf")) { error(); }
     *   auto embedding = loader.loadTensor("token_embd.weight", device_idx);
     *   auto wq = loader.loadTensor("blk.0.attn_q.weight", device_idx);
     */
    class ModelLoader : public IModelLoader
    {
    public:
        /**
         * @brief Construct ModelLoader with optional TensorFactory
         * @param factory TensorFactory for NUMA-aware tensor creation (optional)
         */
        explicit ModelLoader(TensorFactory *factory = nullptr);
        ~ModelLoader() override = default;

        /**
         * @brief Enable or disable mmap-based file loading
         *
         * When enabled (default), the GGUF file is memory-mapped with MAP_POPULATE
         * during loadModel(). Subsequent tensor loads use memcpy from the mapped
         * region instead of seekg+read through an ifstream, eliminating file_mutex_
         * serialization and enabling fully parallel tensor loading.
         *
         * When disabled (--no-mmap), falls back to the original ifstream + file_mutex_
         * path. Use this for extremely large models that won't fit in RAM.
         *
         * Must be called BEFORE loadModel().
         */
        void setUseMmap(bool use_mmap) { use_mmap_ = use_mmap; }

        /**
         * @brief Skip page cache eviction during NUMA mmap
         *
         * When enabled, MmapRegion::create() will NOT call POSIX_FADV_DONTNEED
         * before the NUMA first-touch loop. Use this in multi-rank mode after
         * MmapRegion::prepopulatePageCache() has warmed the page cache, so that
         * the first-touch loop faults from cache (fast) instead of disk (slow).
         *
         * Must be called BEFORE loadModel().
         */
        void setSkipMmapCacheEviction(bool skip) { skip_mmap_cache_eviction_ = skip; }

        /**
         * @brief Check if mmap is active (file successfully memory-mapped)
         */
        bool isMmapActive() const { return mmap_region_ != nullptr; }

        /**
         * @brief Get shared_ptr to the mmap region for a given tensor
         *
         * Tensors hold this to keep the mmap alive for zero-copy access.
         * Returns the appropriate region for split files.
         */
        std::shared_ptr<MmapRegion> getMmapRegion(const GGUFTensorInfo *info) const
        {
            if (!mmap_region_)
                return nullptr;
            if (model_.split_count > 1 && info->split_idx > 0)
            {
                size_t idx = info->split_idx - 1;
                return (idx < split_mmap_regions_.size()) ? split_mmap_regions_[idx] : nullptr;
            }
            return mmap_region_;
        }

        /**
         * @brief Load GGUF file and parse metadata
         * @param file_path Path to .gguf model file
         * @return true if successful, false on error
         */
        bool loadModel(const std::string &file_path);

        /**
         * @brief Initialize a minimal valid model structure for testing
         *
         * Creates a dummy GGUFModel with valid defaults to prevent accessing
         * uninitialized memory in unit tests. Does not load any actual file.
         */
        void initializeTestModel(uint32_t block_count = 1);

        // =========================================================================
        // IModelLoader Implementation
        // =========================================================================

        /**
         * @brief Check if model has been loaded
         */
        bool isLoaded() const override { return loaded_; }

        /**
         * @brief Check if a tensor exists in the model
         * @param tensor_name Name of tensor to check
         * @return true if tensor exists, false otherwise
         */
        bool hasTensor(const std::string &tensor_name) const override;

        /**
         * @brief Get tensor shape without loading the full tensor
         * @param name Tensor name
         * @return Optional vector of dimensions, or std::nullopt if tensor not found
         */
        std::optional<std::vector<size_t>> getTensorShape(const std::string &name) const override;

        /**
         * @brief Get parsed model metadata
         */
        const GGUFModel &getModel() const { return model_; }

        /**
         * @brief Load tensor from GGUF file with new weight precision API
         * @param tensor_name Name of tensor (e.g., "token_embd.weight", "blk.0.attn_q.weight")
         * @param device Device for tensor placement (default: CPU)
         * @param weight_precision How to load the weight:
         *       - NATIVE: Keep in original GGUF format (default, dequantize on-the-fly)
         *       - CONVERT_TO_FP32: Dequantize to FP32 at load time
         *       - CONVERT_TO_BF16: Dequantize to BF16 at load time
         *       - CONVERT_TO_FP16: Dequantize to FP16 at load time
         *       - CONVERT_TO_INT8: Dequantize to INT8 at load time
         * @return Tensor with appropriate type or nullptr on error
         */
        std::shared_ptr<TensorBase> loadTensor(const std::string &tensor_name,
                                               DeviceId device = DeviceId::cpu(),
                                               WeightPrecision weight_precision = WeightPrecision::NATIVE) override;

        /**
         * @brief Load a row slice of a tensor from GGUF file (memory-efficient)
         *
         * Only reads the bytes for rows [row_start, row_end) from the GGUF file,
         * creating a tensor with shape [row_end - row_start, cols]. This is used
         * for tensor parallelism to avoid loading full weights on each MPI rank.
         *
         * @param tensor_name Name of tensor (e.g., "blk.0.attn_output.weight")
         * @param row_start First row to load (0-indexed)
         * @param row_end One past the last row to load
         * @param device Device for tensor placement (default: CPU)
         * @param weight_precision How to load the weight (NATIVE recommended)
         * @return Tensor with shape [row_end - row_start, cols] or nullptr on error
         *
         * @note Only works for 2D weight matrices
         * @note row_start must be aligned to block boundaries for quantized tensors
         */
        std::shared_ptr<TensorBase> loadTensorRowSlice(const std::string &tensor_name,
                                                       size_t row_start, size_t row_end,
                                                       DeviceId device = DeviceId::cpu(),
                                                       WeightPrecision weight_precision = WeightPrecision::NATIVE) override;

        /**
         * @brief Load a column slice of a tensor from GGUF file (memory-efficient)
         *
         * Reads only the columns [col_start, col_end) from the GGUF file for each row,
         * creating a tensor with shape [rows, col_end - col_start]. This is used for
         * column-parallel tensor parallelism (e.g., FFN Gate/Up/Down weights).
         *
         * For quantized tensors, col_start and col_end must be aligned to block boundaries.
         * The function reads row-by-row, extracting only the needed blocks per row.
         *
         * @param tensor_name Name of tensor (e.g., "blk.0.ffn_down.weight")
         * @param col_start First column to load (0-indexed, block-aligned for quantized)
         * @param col_end One past the last column to load (block-aligned for quantized)
         * @param device Device for tensor placement (default: CPU)
         * @param weight_precision How to load the weight (NATIVE recommended)
         * @return Tensor with shape [rows, col_end - col_start] or nullptr on error
         *
         * @note Only works for 2D weight matrices
         * @note For quantized tensors, col_start/col_end must be multiples of block_size
         */
        std::shared_ptr<TensorBase> loadTensorColumnSlice(const std::string &tensor_name,
                                                          size_t col_start, size_t col_end,
                                                          DeviceId device = DeviceId::cpu(),
                                                          WeightPrecision weight_precision = WeightPrecision::NATIVE) override;

        /**
         * @brief Load an expert slice of a 3D MoE tensor from GGUF file
         *
         * Reads only experts [expert_start, expert_end) from a 3D tensor with shape
         * [ne0, ne1, num_experts]. Each expert's data is contiguous in memory, so this
         * reads a single contiguous byte range. Returns a 3D tensor with shape
         * [ne0, ne1, expert_end - expert_start].
         *
         * @param tensor_name Name of tensor (e.g., "blk.0.ffn_gate_exps.weight")
         * @param expert_start First expert to load (0-indexed)
         * @param expert_end One past the last expert to load
         * @param device Device for tensor placement (default: CPU)
         * @param weight_precision How to load the weight (NATIVE recommended)
         * @return Tensor with shape [ne0, ne1, local_count] or nullptr on error
         */
        std::shared_ptr<TensorBase> loadTensorExpertSlice(const std::string &tensor_name,
                                                          size_t expert_start, size_t expert_end,
                                                          DeviceId device = DeviceId::cpu(),
                                                          WeightPrecision weight_precision = WeightPrecision::NATIVE) override;

        // =========================================================================
        // IModelLoader - Metadata Accessors
        // =========================================================================

        std::vector<std::string> tensorNames() const override;
        std::string architecture() const override { return model_.architecture; }
        size_t tensorCount() const override { return model_.tensors.size(); }
        size_t totalBytes() const override;

        int getInt(const std::string &key, int default_val = 0) const override;
        uint64_t getUInt64(const std::string &key, uint64_t default_val = 0) const override;
        float getFloat(const std::string &key, float default_val = 0.0f) const override;
        std::string getString(const std::string &key, const std::string &default_val = "") const override;

        // Convenience accessors (common hyperparameters)
        uint64_t blockCount() const override { return model_.block_count; }
        uint64_t embeddingLength() const override { return model_.embedding_length; }
        uint64_t headCount() const override { return model_.head_count; }
        uint64_t headCountKV() const override { return model_.head_count_kv; }
        uint64_t vocabSize() const override { return model_.vocab_size; }
        uint64_t contextLength() const override { return model_.context_length; }
        uint64_t feedForwardLength() const override {
            uint64_t d_ff = getUInt64("feed_forward_length", 0);
            if (d_ff == 0)
            {
                // MoE models may not have feed_forward_length; fall back to shared expert or per-expert FFN
                d_ff = getUInt64("expert_shared_feed_forward_length", 0);
            }
            if (d_ff == 0)
            {
                d_ff = getUInt64("expert_feed_forward_length", 0);
            }
            return d_ff;
        }
        uint64_t keyLength() const override { return model_.key_length; }
        float ropeTheta() const override { return model_.rope_theta; }
        float rmsNormEps() const override { return model_.rms_norm_eps; }

        void releaseMmapRegions() override
        {
            if (mmap_region_)
            {
                LOG_INFO("[ModelLoader] Releasing mmap region ("
                         << (mmap_region_->size() / (1024 * 1024)) << " MB)");
                mmap_region_.reset();
            }
            for (auto &region : split_mmap_regions_)
            {
                if (region)
                {
                    LOG_INFO("[ModelLoader] Releasing split mmap region ("
                             << (region->size() / (1024 * 1024)) << " MB)");
                    region.reset();
                }
            }
            split_mmap_regions_.clear();
        }

        size_t adviseMmapDontneed() override
        {
            size_t total = 0;
            if (mmap_region_)
            {
                total += mmap_region_->adviseDontneed();
            }
            for (auto &region : split_mmap_regions_)
            {
                if (region)
                    total += region->adviseDontneed();
            }
            if (total > 0)
            {
                LOG_INFO("[ModelLoader] Advised DONTNEED on mmap regions ("
                         << (total / (1024 * 1024)) << " MB) — pages reclaimable by OS");
            }
            return total;
        }

    private:
        // Tensor factory (created internally if not provided)
        TensorFactory *factory_;
        std::unique_ptr<TensorFactory> owned_factory_; // Owned factory if created internally
        std::shared_ptr<IMPIContext> owned_mpi_ctx_;    // Owned MPI context for owned factory

        // Parsing helpers
        bool parseHeader();
        bool parseMetadata();
        bool parseTensorInfo();
        void extractModelMetadata();

        // Multi-part GGUF helpers
        bool loadSplitFiles();
        std::string generateSplitPath(const std::string &base_path, uint16_t split_no, uint16_t split_count);
        bool parseSplitPath(const std::string &split_path, std::string &prefix, uint16_t &split_no, uint16_t &split_count);

        // INT8 dequantization helpers
        std::shared_ptr<TensorBase> dequantizeToINT8(
            const GGUFTensorInfo *info,
            const std::vector<size_t> &shape,
            const std::vector<uint8_t> &raw);

        // FP32 dequantization (comprehensive support for all quantized formats)
        std::shared_ptr<TensorBase> dequantizeToFP32(
            const GGUFTensorInfo *info,
            const std::vector<size_t> &shape,
            const std::vector<uint8_t> &raw);

        void dequantizeIQ4_NLToFP32(
            const std::vector<uint8_t> &raw,
            std::vector<float> &fp32_buffer,
            const std::vector<size_t> &shape);
        void dequantizeQ8_0ToFP32(
            const std::vector<uint8_t> &raw,
            std::vector<float> &fp32_buffer,
            const std::vector<size_t> &shape);
        void dequantizeQ4_0ToFP32(
            const std::vector<uint8_t> &raw,
            std::vector<float> &fp32_buffer,
            const std::vector<size_t> &shape);
        void dequantizeQ6_KToFP32(
            const std::vector<uint8_t> &raw,
            std::vector<float> &fp32_buffer,
            const std::vector<size_t> &shape);

        // Low-level readers
        template <typename T>
        bool readValue(T &value);
        bool readString(std::string &str);
        bool readArray(GGUFValue &value);

        // State
        bool loaded_ = false;
        std::string file_path_;
        std::ifstream file_stream_;
        std::vector<std::ifstream> split_streams_; // Additional file streams for multi-part GGUF
        GGUFModel model_;
        mutable std::mutex file_mutex_; // Protects file_stream_ and split_streams_ for thread-safety

        // mmap state (when use_mmap_ is true)
        bool use_mmap_ = true;
        bool skip_mmap_cache_eviction_ = false;
        std::shared_ptr<MmapRegion> mmap_region_;                     // Main file mmap (shared for zero-copy tensors)
        std::vector<std::shared_ptr<MmapRegion>> split_mmap_regions_; // Split file mmaps

        /**
         * @brief Get pointer into mmap'd region for a tensor's data
         * @param info Tensor metadata (contains offset and split_idx)
         * @return Pointer to tensor data in mmap region, or nullptr if not mmap'd
         */
        const uint8_t *getMmapPtr(const GGUFTensorInfo *info) const;

        /**
         * @brief Get the data offset for a tensor's split file
         * @param info Tensor metadata
         * @return Data offset for the file containing this tensor
         */
        uint64_t getDataOffset(const GGUFTensorInfo *info) const;
    };

    // =============================================================================
    // TEMPLATE IMPLEMENTATIONS
    // =============================================================================

    template <typename T>
    bool ModelLoader::readValue(T &value)
    {
        return file_stream_.read(reinterpret_cast<char *>(&value), sizeof(T)).good();
    }

} // namespace llaminar2
