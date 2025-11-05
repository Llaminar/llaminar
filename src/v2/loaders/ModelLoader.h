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
 *  - Optional dequantization to INT8 (when --precision int8 is set)
 *  - Returns V2 TensorBase (with device affinity, ITensor interfaces)
 */

#pragma once

#include "../pipelines/PipelineConfig.h" // for ComputePrecision
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class TensorBase;
    class TensorFactory;

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

        // Typed accessors
        uint32_t asUInt32() const;
        uint64_t asUInt64() const;
        float asFloat32() const;
        std::string asString() const;
        uint64_t asArrayLength() const { return array_length; }
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
        uint64_t vocab_size = 0;       // vocabulary size
        float rope_theta = 10000.0f;   // RoPE base frequency

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
     * Usage:
     *   ModelLoader loader;
     *   if (!loader.loadModel("model.gguf")) { error(); }
     *   auto embedding = loader.loadTensor("token_embd.weight", device_idx);
     *   auto wq = loader.loadTensor("blk.0.attn_q.weight", device_idx);
     */
    class ModelLoader
    {
    public:
        /**
         * @brief Construct ModelLoader with optional TensorFactory
         * @param factory TensorFactory for NUMA-aware tensor creation (optional)
         */
        explicit ModelLoader(TensorFactory *factory = nullptr);
        ~ModelLoader() = default;

        /**
         * @brief Load GGUF file and parse metadata
         * @param file_path Path to .gguf model file
         * @return true if successful, false on error
         */
        bool loadModel(const std::string &file_path);

        /**
         * @brief Check if model has been loaded
         */
        bool isLoaded() const { return loaded_; }

        /**
         * @brief Get parsed model metadata
         */
        const GGUFModel &getModel() const { return model_; }

        /**
         * @brief Load tensor from GGUF file
         * @param tensor_name Name of tensor (e.g., "token_embd.weight", "blk.0.attn_q.weight")
         * @param device_idx Device index for tensor placement (0 = CPU, 1+ = GPU)
         * @param precision Compute precision mode (if INT8, dequantize all formats to INT8)
         * @return Tensor with appropriate type (FP32Tensor, IQ4_NLTensor, INT8Tensor, etc.) or nullptr on error
         *
         * @note When precision is INT8, all quantized tensors are dequantized to INT8 format
         *       for AVX512-VNNI (CPU) or INT8×INT8 GEMM (CUDA/CUTLASS)
         * @note Otherwise, returns tensors in native format (quantized weights stay quantized)
         * @note Tensor type determined by GGUF metadata (F32 → FP32Tensor, IQ4_NL → IQ4_NLTensor)
         */
        std::shared_ptr<TensorBase> loadTensor(const std::string &tensor_name,
                                               int device_idx = 0,
                                               ComputePrecision precision = ComputePrecision::FP32);

    private:
        // Tensor factory (optional, for NUMA-aware allocation)
        TensorFactory *factory_;

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
