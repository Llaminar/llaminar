#pragma once

/**
 * @file model_loader.h
 * @brief GGUF Model Loader - Quantized Model Loading, Dequantization, and Weight Distribution
 * @author David Sanftenberg
 *
 * ============================================================================
 * OVERVIEW: GGUF FORMAT AND MODEL LOADING SYSTEM
 * ============================================================================
 *
 * The ModelLoader system handles loading quantized LLM models from GGUF
 * (GPT-Generated Unified Format) files, performing on-the-fly dequantization,
 * and distributing weights across MPI ranks for distributed inference.
 *
 * GGUF FILE STRUCTURE:
 * ┌─────────────────────────────────────┐
 * │ Magic Number (0x46554747 = "GGUF") │
 * ├─────────────────────────────────────┤
 * │ Version (uint32)                    │
 * ├─────────────────────────────────────┤
 * │ Tensor Count (uint64)               │
 * ├─────────────────────────────────────┤
 * │ Metadata KV Count (uint64)          │
 * ├─────────────────────────────────────┤
 * │ Metadata (architecture, n_layers,   │
 * │           hyperparameters, etc.)    │
 * ├─────────────────────────────────────┤
 * │ Tensor Metadata (name, dims, type,  │
 * │                  quantization, etc.)│
 * ├─────────────────────────────────────┤
 * │ Padding (32-byte alignment)         │
 * ├─────────────────────────────────────┤
 * │ Tensor Data (quantized blobs)       │
 * └─────────────────────────────────────┘
 *
 * SUPPORTED QUANTIZATION FORMATS:
 *   - F32: Full precision (4 bytes/element)
 *   - F16: Half precision (2 bytes/element)
 *   - Q4_0: 4-bit with shared scale per block (32 values/block)
 *   - Q4_1: 4-bit with min+max per block
 *   - Q5_0, Q5_1: 5-bit variants
 *   - Q6_K: 6-bit K-quants (improved quality, 256 values/super-block)
 *   - Q8_0: 8-bit quantization
 *   - Q2_K, Q3_K, Q4_K, Q5_K: K-quants family (high compression)
 *
 * DEQUANTIZATION PROCESS:
 *   Quantized weights are dequantized to FP32 **once at load time**.
 *   This happens before inference begins, not during runtime.
 *
 *   Example (Q4_0 format):
 *   - On-disk: 32 4-bit values + 1 FP16 scale = 18 bytes/block
 *   - Dequantized: 32 FP32 values = 128 bytes
 *   - Formula: value = (quant_nibble - 8) * scale
 *
 *   Performance:
 *   - Dequantization: ~50-200ms per large tensor (CPU-bound)
 *   - Total model load: 1-5 seconds for 0.5B parameter model
 *   - Parallelized with OpenMP for FP16→FP32 conversion
 *
 * WEIGHT TENSOR NAMING CONVENTION (from llama.cpp):
 *
 *   Embedding Layer:
 *     token_embd.weight           [vocab_size, hidden_size]
 *
 *   Transformer Blocks (per-layer prefix: blk.{layer}.):
 *     attn_norm.weight            [hidden_size]  (RMSNorm gamma, no bias)
 *
 *     Attention Weights:
 *       attn_q.weight             [n_head*head_dim, hidden_size]
 *       attn_k.weight             [n_head_kv*head_dim, hidden_size]
 *       attn_v.weight             [n_head_kv*head_dim, hidden_size]
 *       attn_output.weight        [hidden_size, n_head*head_dim]
 *
 *     ⚠️  CRITICAL - Attention Biases (MUST BE LOADED!):
 *       attn_q.bias               [n_head*head_dim]
 *       attn_k.bias               [n_head_kv*head_dim]
 *       attn_v.bias               [n_head_kv*head_dim]
 *
 *       Historical Bug: Llaminar originally only loaded weights, ignoring
 *       bias tensors. This caused 79.9x divergence because some models have
 *       LARGE bias values (e.g., Q bias range [-79.0, +47.75] with 94 out
 *       of 896 values > ±10.0). Always check for and load bias tensors!
 *
 *     FFN Weights:
 *       ffn_norm.weight           [hidden_size]  (RMSNorm gamma, no bias)
 *       ffn_gate.weight           [d_ff, hidden_size]
 *       ffn_up.weight             [d_ff, hidden_size]
 *       ffn_down.weight           [hidden_size, d_ff]
 *
 *   Output Layer:
 *     output_norm.weight          [hidden_size]  (RMSNorm gamma, no bias)
 *     output.weight               [vocab_size, hidden_size]  (LM head)
 *
 * WEIGHT MATRIX STORAGE CONVENTION:
 *   ALL weight matrices stored as [out_features, in_features].
 *   This matches:
 *     - GGUF on-disk format (no conversions needed)
 *     - PyTorch nn.Linear convention
 *     - Industry standard (most ML frameworks)
 *
 *   Usage in kernels: output = input @ weight^T
 *
 *   Example: K projection
 *     input:  [seq_len, 896]
 *     weight: [128, 896]  ← stored as [out, in]
 *     matmul: input @ weight^T (transpose_B=true)
 *     result: [seq_len, 128] ✓
 *
 * BIAS APPLICATION (CRITICAL):
 *   After matrix multiplication, biases are broadcast-added across the
 *   sequence dimension:
 *
 *   output[row, col] += bias[col]  for all rows
 *
 *   Bias shapes:
 *     - Q bias: [n_head * head_dim]  (e.g., [896] for 14 heads * 64 dim)
 *     - K/V bias: [n_head_kv * head_dim]  (e.g., [128] for GQA)
 *
 *   Biases are sliced with weights in MPI tensor parallel mode:
 *     - Each rank gets bias slice for its owned heads
 *     - After local projection + bias, MPI_Allreduce assembles full output
 *
 * MPI WEIGHT DISTRIBUTION (Tensor Parallel):
 *   Single-Rank: All weights/biases used directly (no distribution)
 *
 *   Multi-Rank: Weights sliced by attention heads (column partition)
 *     - Example: 2 ranks, 14 Q heads
 *       Rank 0: local_wq [448, 896] (heads 0-6),  local_bq [448]
 *       Rank 1: local_wq [448, 896] (heads 7-13), local_bq [448]
 *     - After projection + bias, MPI_Allreduce for full [seq_len, 896]
 *
 * SPECIAL TENSOR TYPES:
 *   Token Embeddings (token_embd.weight):
 *     - Shape: [vocab_size, hidden_size] (e.g., [151936, 896])
 *     - Largest tensor in model (100-500 MB typically)
 *     - Usually NOT quantized (stored as F16 or F32)
 *     - Replicated across all MPI ranks
 *     - Accessed via index lookup: embedding[token_id] → [hidden_size]
 *
 *   RMSNorm Gamma Weights (.weight suffix):
 *     - Shape: [hidden_size] (1D vector)
 *     - NO bias term (unlike LayerNorm)
 *     - Applied element-wise: output[i] = (input[i] / rms) * gamma[i]
 *     - Multiplicative scale, not additive bias
 *
 * VALIDATION AND DIAGNOSTICS:
 *   Environment Flags:
 *     LLAMINAR_DEQUANT_STATS=1        Log per-tensor statistics (min/max/mean)
 *     LLAMINAR_DEQUANT_ANOMALIES=1    Warn on NaN/Inf/huge values
 *
 *   Example diagnostic output:
 *     [DEQUANT] blk.0.attn_q.weight: shape=[896,896], quant=Q6_K
 *       min=-0.2415, max=0.2891, mean=0.0001, mean_abs=0.0234
 *       samples: [0.0151, -0.0234, 0.0089, ...]
 *
 * COMMON PITFALLS:
 *   ❌ Forgetting to load biases → Massive divergence (79.9x error)
 *   ❌ Incorrect shape assumptions → Matmul dimension mismatches
 *   ❌ Skipping weight transpose → Wrong orientation for nn.Linear
 *   ❌ Ignoring dequant anomalies → Silent NaN/Inf propagation
 *   ❌ Hardcoded tensor names → Breaks on architecture changes
 *
 *   ✅ Always check for and load bias tensors with hasTensor()
 *   ✅ Validate shapes immediately after loading
 *   ✅ Enable diagnostics during initial model testing
 *   ✅ Follow GGUF naming conventions from llama.cpp
 *   ✅ Test with multiple model sizes to catch dimension bugs
 *
 * See:
 *   - .github/instructions/llaminar-architecture.instructions.md (Section 9)
 *   - docs/WEIGHT_MATRIX_CONVENTIONS.md
 *   - src/kernels/MPIAttentionKernel.cpp (bias application example)
 * ============================================================================
 */

#include "common.h"
#include "tensors/tensor_base.h"
#include "transformer_config.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <fstream>
#include <mutex>
#include <atomic>

// GGUF metadata value types
/**
 * @enum GGUFValueType
 * @brief Type identifier for GGUF metadata values
 *
 * Each metadata key-value pair in GGUF has an associated type tag.
 * These types match the GGUF specification from llama.cpp.
 */
enum class GGUFValueType : uint32_t
{
    UINT8 = 0,   ///< Unsigned 8-bit integer
    INT8 = 1,    ///< Signed 8-bit integer
    UINT16 = 2,  ///< Unsigned 16-bit integer
    INT16 = 3,   ///< Signed 16-bit integer
    UINT32 = 4,  ///< Unsigned 32-bit integer
    INT32 = 5,   ///< Signed 32-bit integer
    FLOAT32 = 6, ///< 32-bit floating point
    BOOL = 7,    ///< Boolean value
    STRING = 8,  ///< UTF-8 string
    ARRAY = 9,   ///< Array of values
    UINT64 = 10, ///< Unsigned 64-bit integer
    INT64 = 11,  ///< Signed 64-bit integer
    FLOAT64 = 12 ///< 64-bit floating point (double)
};

/**
 * @enum GGUFTensorType
 * @brief Quantization format identifier for tensor data
 *
 * Specifies how tensor weights are quantized/stored in the GGUF file.
 * Each format has different compression ratios and quality trade-offs.
 *
 * Compression Ratios (relative to FP32):
 *   - F32: 1.0x (no compression, 4 bytes/element)
 *   - F16: 2.0x (2 bytes/element)
 *   - Q8_0: 4.0x (1 byte/element + scale overhead)
 *   - Q6_K: ~5.3x (high quality, recommended for most weights)
 *   - Q4_0: 8.0x (4 bits/element + scale)
 *   - Q2_K: ~16x (aggressive compression, may degrade quality)
 *
 * K-quants (Q*_K formats):
 *   Use larger super-blocks (256 elements) with multiple scales for
 *   better quality than non-K variants. Recommended for production.
 */
enum class GGUFTensorType : uint32_t
{
    F32 = 0,  ///< Full precision float (4 bytes/element, no quantization)
    F16 = 1,  ///< Half precision float (2 bytes/element)
    Q4_0 = 2, ///< 4-bit quantization, 32 values per block
    Q4_1 = 3, ///< 4-bit with min+max, 32 values per block
    // 4,5 removed upstream (Q4_2/Q4_3)
    Q5_0 = 6,     ///< 5-bit quantization, 32 values per block
    Q5_1 = 7,     ///< 5-bit with min+max, 32 values per block
    Q8_0 = 8,     ///< 8-bit quantization, 32 values per block
    Q8_1 = 9,     ///< 8-bit with min+max (unsupported, parser rejects)
    Q2_K = 10,    ///< 2-bit K-quants, 256 values per super-block
    Q3_K = 11,    ///< 3-bit K-quants, 256 values per super-block
    Q4_K = 12,    ///< 4-bit K-quants, 256 values per super-block
    Q5_K = 13,    ///< 5-bit K-quants, 256 values per super-block
    Q6_K = 14,    ///< 6-bit K-quants, 256 values per super-block (recommended)
    Q8_K = 15,    ///< 8-bit K-quants, 256 values per super-block
    IQ2_XXS = 16, ///< iMatrix 2-bit extra-extra-small
    IQ2_XS = 17,  ///< iMatrix 2-bit extra-small
    IQ3_XXS = 18, ///< iMatrix 3-bit extra-extra-small
    IQ1_S = 19,   ///< iMatrix 1-bit small
    IQ4_NL = 20,  ///< iMatrix 4-bit non-linear
    IQ3_S = 21,   ///< iMatrix 3-bit small
    IQ2_S = 22,   ///< iMatrix 2-bit small
    IQ4_XS = 23,  ///< iMatrix 4-bit extra-small
    IQ1_M = 29,   ///< iMatrix 1-bit medium
    Q4_K_M = Q4_K ///< Alias: Q4_K_M uses Q4_K layout
};

/**
 * @struct GGUFValue
 * @brief Container for a GGUF metadata value
 *
 * Stores metadata as raw bytes with type information.
 * Use as<T>() to extract typed values.
 */
struct GGUFValue
{
    GGUFValueType type;        ///< Type identifier
    std::vector<uint8_t> data; ///< Raw value bytes

    /**
     * @brief Extract typed value from metadata
     * @tparam T Target type (int, float, etc.)
     * @return Extracted value
     * @throws std::runtime_error if type mismatch
     */
    template <typename T>
    T as() const;

    /**
     * @brief Extract string value
     * @return String value (only valid if type == STRING)
     * @throws std::runtime_error if not a string
     */
    std::string asString() const;

    /**
     * @brief Extract string array
     * @return Vector of strings (only valid if type == ARRAY of strings)
     * @throws std::runtime_error if not a string array
     */
    std::vector<std::string> asStringArray() const;
};

/**
 * @struct GGUFTensorInfo
 * @brief Metadata for a single tensor in the GGUF file
 *
 * Contains all information needed to locate and dequantize a tensor:
 * name, shape, quantization type, file offset, and byte size.
 */
struct GGUFTensorInfo
{
    std::string name;                 ///< Tensor name (e.g., "blk.0.attn_q.weight")
    std::vector<uint64_t> dimensions; ///< Shape (e.g., [896, 896] for Q weight)
    GGUFTensorType type;              ///< Quantization format
    uint64_t offset;                  ///< Byte offset in file to tensor data
    size_t size_bytes;                ///< Total bytes for this tensor

    /**
     * @brief Check if tensor is quantized (vs full precision)
     * @return true if quantized (Q4_0, Q6_K, etc.), false if F32/F16
     */
    bool isQuantized() const;

    /**
     * @brief Get size in bytes per element (or per block for quantized)
     * @return Bytes per element/block depending on type
     */
    size_t getTypeSize() const;

    /**
     * @brief Get quantization block size (elements per block)
     * @return Block size (32 for Q4_0, 256 for K-quants, 1 for F32/F16)
     */
    size_t getBlockSize() const;
};

/**
 * @struct GGUFModel
 * @brief Complete GGUF model structure with metadata and tensor information
 *
 * Represents the parsed contents of a GGUF file header.
 * Contains all metadata (hyperparameters, architecture info) and
 * tensor metadata (shapes, types, offsets) but NOT the actual tensor data.
 *
 * Tensor data is loaded on-demand via ModelLoader::loadTensor().
 */
struct GGUFModel
{
    uint32_t version;           ///< GGUF format version (currently 3)
    uint64_t tensor_count;      ///< Number of tensors in file
    uint64_t metadata_kv_count; ///< Number of metadata key-value pairs
    uint64_t data_offset;       ///< File offset where tensor data begins
    uint32_t alignment = 32;    ///< Alignment for tensor data section (GGUF default)

    std::unordered_map<std::string, GGUFValue> metadata; ///< Metadata key-value store
    std::vector<GGUFTensorInfo> tensors;                 ///< Tensor metadata array

    // Model-specific metadata (extracted from metadata map)
    std::string architecture;            ///< Model architecture (e.g., "llama", "qwen2")
    uint32_t context_length;             ///< Maximum sequence length
    uint32_t embedding_length;           ///< Hidden dimension (d_model)
    uint32_t block_count;                ///< Number of transformer layers (n_layers)
    uint32_t feed_forward_length = 0;    ///< FFN dimension (d_ff), 0 if not present
    uint32_t head_count;                 ///< Number of attention heads (n_head)
    uint32_t head_count_kv;              ///< Number of KV heads (n_head_kv, for GQA)
    float rope_freq_base = 10000.0f;     ///< RoPE frequency base
    float rms_norm_eps = 1e-6f;          ///< RMSNorm epsilon (default 1e-6 for PyTorch/HuggingFace)
    std::vector<std::string> token_list; ///< Tokenizer vocabulary (if present)

    /**
     * @brief Check if metadata key exists
     * @param key Metadata key to check
     * @return true if key is present
     */
    bool hasMetadata(const std::string &key) const;

    /**
     * @brief Get metadata value with default fallback
     * @tparam T Value type
     * @param key Metadata key
     * @param default_value Value to return if key not found
     * @return Metadata value or default
     */
    template <typename T>
    T getMetadata(const std::string &key, T default_value = T{}) const;

    /**
     * @brief Find tensor by name (mutable)
     * @param name Tensor name to search for
     * @return Pointer to tensor info, or nullptr if not found
     */
    GGUFTensorInfo *findTensor(const std::string &name);

    /**
     * @brief Find tensor by name (const)
     * @param name Tensor name to search for
     * @return Const pointer to tensor info, or nullptr if not found
     */
    const GGUFTensorInfo *findTensor(const std::string &name) const;
};

/**
 * @class ModelLoader
 * @brief Main interface for loading and dequantizing quantized LLM models from GGUF files
 *
 * ModelLoader provides a complete pipeline for:
 *   1. Parsing GGUF file headers and metadata
 *   2. Loading and dequantizing tensor weights on-demand or in streaming fashion
 *   3. Supporting MPI-based tensor partitioning for distributed execution
 *   4. Validating loaded weights with optional diagnostics
 *
 * ============================================================================
 * BASIC USAGE
 * ============================================================================
 *
 * @code
 * // Initialize and load model metadata
 * ModelLoader loader;
 * if (!loader.loadModel("models/qwen-0.5b-q6_k.gguf")) {
 *     LOG_ERROR("Failed to load model");
 *     return false;
 * }
 *
 * // Load full-precision tensors
 * auto embedding = loader.loadTensor("token_embd.weight");
 * auto q_weight = loader.loadTensor("blk.0.attn_q.weight");
 * auto q_bias = loader.loadTensor("blk.0.attn_q.bias");  // ⚠️ CRITICAL - don't forget biases!
 *
 * // Access model metadata
 * const auto &model = loader.getModel();
 * LOG_INFO("Loaded model: " << model.architecture
 *          << ", layers: " << model.block_count
 *          << ", hidden: " << model.embedding_length);
 * @endcode
 *
 * ============================================================================
 * MPI DISTRIBUTION WITH STREAMING
 * ============================================================================
 *
 * For distributed tensor parallel execution, use streaming shard loaders:
 *
 * @code
 * // Example: 14 heads, 2 ranks → 7 heads per rank (896 dims per shard)
 * // Attention weight shape: [1792, 896] (14 heads * 128 dims per head, hidden size)
 *
 * int world_size = 2;
 * int rank = ...; // from MPI_Comm_rank
 *
 * // Calculate column ranges for this rank
 * int total_heads = 14;
 * int heads_per_rank = total_heads / world_size;  // 7
 * int col_offset = rank * heads_per_rank * 128;   // Rank 0: 0, Rank 1: 896
 * int col_count = heads_per_rank * 128;            // 896
 *
 * // Allocate destination buffer
 * std::vector<float> shard_data(num_rows * col_count);
 *
 * // Stream load column shard (single pass, no intermediate full tensor)
 * std::vector<int> offsets = {col_offset};
 * std::vector<int> counts = {col_count};
 * std::vector<float*> dests = {shard_data.data()};
 *
 * if (!loader.loadTensorColumnShards("blk.0.attn_q.weight", offsets, counts, dests)) {
 *     LOG_ERROR("Streaming load failed, falling back to full load");
 *     auto full = loader.loadTensor("blk.0.attn_q.weight");
 *     // ... manual slicing ...
 * }
 * @endcode
 *
 * **Distribution Strategy**:
 *   - **Attention weights (Q/K/V/O)**: Column-sharded across heads
 *   - **Attention biases (bq/bk/bv)**: Column-sharded (1D tensors, same logic)
 *   - **FFN weights**: gate/up column-sharded, down row-sharded
 *   - **Embeddings**: Replicated on all ranks (too large for efficient sharding)
 *   - **RMSNorm gamma**: Replicated (tiny, [hidden_dim] only)
 *
 * ============================================================================
 * TENSOR NAMING REFERENCE
 * ============================================================================
 *
 * See file header for complete naming conventions. Quick reference:
 *   - Embeddings: `token_embd.weight`
 *   - Attention: `blk.{L}.attn_{q,k,v,output}.{weight,bias}`
 *   - FFN: `blk.{L}.ffn_{gate,up,down}.weight`
 *   - Norms: `blk.{L}.attn_norm.weight`, `blk.{L}.ffn_norm.weight`
 *   - Output: `output_norm.weight`, `output.weight` (LM head)
 *
 * ============================================================================
 * PERFORMANCE CHARACTERISTICS
 * ============================================================================
 *
 * **Dequantization Performance** (CPU-bound, single-threaded):
 *   - Small tensor (896×896 Q6_K): ~5-10ms
 *   - Large tensor (2048×2048 Q6_K): ~50-100ms
 *   - Embedding (151936×896 Q6_K): ~500-1000ms
 *   - Total model load (0.5B params): 1-3 seconds
 *
 * **Streaming vs Full Load**:
 *   - Streaming column shard: Same speed, 1/world_size memory
 *   - Full load + slice: Same speed, full tensor temporary memory
 *   - Streaming preferred for large weights (>10MB) in multi-rank setups
 *
 * **Memory Usage** (0.5B model):
 *   - Quantized file (Q6_K): ~350MB
 *   - Dequantized FP32 weights: ~2GB
 *   - Cache overhead: ~50MB (tensor metadata + file buffer)
 *   - Streaming overhead: ~5MB (shard cache for repeated access)
 *
 * ============================================================================
 * VALIDATION AND DIAGNOSTICS
 * ============================================================================
 *
 * Enable via environment variables:
 *
 * @code{.bash}
 * # Per-tensor statistics (min/max/mean/sample)
 * export LLAMINAR_DEQUANT_STATS=1
 *
 * # Detect anomalous values (NaN/Inf/huge outliers)
 * export LLAMINAR_DEQUANT_ANOMALIES=1
 * @endcode
 *
 * Output example:
 * @verbatim
 * [INFO] Dequant blk.0.attn_q.weight [896,896] Q6_K → F32
 *        Min: -0.523, Max: 0.487, Mean: 0.002, Std: 0.089
 *        Sample[0:5]: [0.012, -0.034, 0.056, -0.078, 0.023]
 * [WARN] Anomaly in blk.0.attn_q.bias: 94/896 values > ±10.0
 *        Max: 47.75, Min: -79.0  ← EXPECTED for biases!
 * @endverbatim
 *
 * ⚠️ **Note**: Large bias values are NORMAL and ESSENTIAL. Do not normalize!
 *
 * ============================================================================
 * QUANTIZATION SHARD CACHE
 * ============================================================================
 *
 * For repeated streaming loads (e.g., loading multiple column shards from same tensor),
 * ModelLoader maintains an LRU cache of raw quantized data blocks to avoid redundant
 * file reads. Cache stats available via `getQuantShardCacheStats()`.
 *
 * Default cache size: 256MB (configurable via internal threshold)
 * Eviction policy: LRU (least recently used)
 *
 * @code
 * // Monitor cache efficiency
 * auto stats = loader.getQuantShardCacheStats();
 * LOG_INFO("Cache: " << stats.cache_hits << " hits, "
 *          << stats.cache_misses << " misses, "
 *          << stats.bytes_resident / 1024 / 1024 << " MB resident");
 *
 * // Clear cache if needed (e.g., after loading phase)
 * loader.clearQuantShardCache();
 * @endcode
 *
 * ============================================================================
 * THREAD SAFETY
 * ============================================================================
 *
 * - **Single-threaded dequantization**: One tensor loaded at a time
 * - **Thread-safe after load**: Multiple threads can access loaded tensors
 * - **MPI-safe**: Each rank independently loads its shard, no collectives needed
 * - **Cache thread-safety**: Internal shard cache is NOT thread-safe (single-threaded use only)
 *
 * ============================================================================
 * SUPPORTED QUANTIZATION FORMATS
 * ============================================================================
 *
 * - **F32**: Full precision (no quantization)
 * - **F16**: Half precision
 * - **Q4_0**, **Q4_1**: 4-bit quantization (32 vals/block)
 * - **Q5_0**, **Q5_1**: 5-bit quantization (32 vals/block)
 * - **Q8_0**: 8-bit quantization (32 vals/block)
 * - **Q2_K** to **Q6_K**: K-quants with 256-element super-blocks (recommended)
 * - **Q8_1**: Unsupported (parser rejects)
 *
 * @see GGUFModel for metadata structure
 * @see GGUFTensorInfo for tensor metadata
 * @see loadTensor() for basic loading
 * @see loadTensorColumnShards() for MPI streaming distribution
 */
class ModelLoader
{
public:
    /**
     * @brief Default constructor
     *
     * Creates an empty ModelLoader. Must call loadModel() before use.
     */
    ModelLoader();

    /**
     * @brief Destructor (default)
     *
     * Closes file streams and cleans up internal caches.
     */
    ~ModelLoader() = default;

    /**
     * @brief Load and parse a GGUF model file
     * @param file_path Path to .gguf model file
     * @return true if successful, false if file corrupt or unsupported format
     *
     * @details Parses:
     *   1. GGUF magic number and version
     *   2. Metadata key-value pairs (hyperparameters, architecture)
     *   3. Tensor information (names, shapes, types, offsets)
     *
     * @note Tensor data is NOT loaded yet - loaded on-demand via loadTensor()
     * @note Call isLoaded() to check if model is ready for use
     *
     * @code
     * ModelLoader loader;
     * if (!loader.loadModel("model.gguf")) {
     *     LOG_ERROR("Failed to load model");
     *     return false;
     * }
     * @endcode
     */
    bool loadModel(const std::string &file_path);

    /**
     * @brief Check if model is successfully loaded
     * @return true if loadModel() completed successfully
     */
    bool isLoaded() const { return loaded_; }

    /**
     * @brief Get model metadata and tensor information
     * @return Const reference to GGUFModel structure
     * @throws May assert if called before loadModel()
     *
     * @code
     * const auto &model = loader.getModel();
     * LOG_INFO("Model: " << model.architecture
     *          << ", layers: " << model.block_count);
     * @endcode
     */
    const GGUFModel &getModel() const { return model_; }

    /**
     * @brief Load and dequantize a tensor by name
     * @param tensor_name Tensor name (e.g., "blk.0.attn_q.weight")
     * @return Shared pointer to dequantized FP32 tensor (SimpleTensor or COSMATensor)
     * @throws std::runtime_error if tensor not found or dequantization fails
     *
     * @note First call dequantizes from file, subsequent calls return cached result
     * @note Use this for embeddings, norms, and single-rank tensors
     * @note For MPI sharding, prefer loadTensorColumnShards() for memory efficiency
     *
     * @code
     * auto weight = loader.loadTensor("blk.0.attn_q.weight");
     * auto bias = loader.loadTensor("blk.0.attn_q.bias");  // ⚠️ Don't forget!
     * @endcode
     */
    std::shared_ptr<llaminar::TensorBase> loadTensor(const std::string &tensor_name);

    /**
     * @brief Load all tensors in the model (for debugging/testing)
     * @return Vector of all dequantized tensors
     *
     * @warning This loads ENTIRE model into memory (GBs for even small models)
     * @note Only use for testing, diagnostics, or single-node execution
     */
    std::vector<std::shared_ptr<llaminar::TensorBase>> loadAllTensors();

    /**
     * @brief Stream load multiple column shards from a 2D tensor (MPI-optimized)
     * @param tensor_name Tensor name
     * @param col_offsets Starting column index for each shard
     * @param col_counts Number of columns in each shard
     * @param dests Output buffers (must be pre-allocated: rows × col_counts[i])
     * @return true if successful, false if unsupported (caller should fallback)
     *
     * @details Extracts disjoint column ranges in a single file pass:
     *   - For tensor [rows, cols]: extracts multiple [rows, col_counts[i]] sub-matrices
     *   - Output written as dense row-major: dests[i][row * col_counts[i] + col]
     *   - Only one file read pass regardless of number of shards
     *
     * **Limitations**:
     *   - 2D tensors only (enforced by implementation)
     *   - F32/F16/Q*_* types supported; exotic types may fallback to false
     *
     * **MPI Use Case** (2 ranks, attention weight [1792, 896]):
     * @code
     * int rank = ...; // 0 or 1
     * int col_offset = rank * 896;  // Rank 0: col 0-895, Rank 1: col 896-1791
     * int col_count = 896;
     *
     * std::vector<float> shard(num_rows * col_count);
     * std::vector<int> offsets = {col_offset};
     * std::vector<int> counts = {col_count};
     * std::vector<float*> dests = {shard.data()};
     *
     * if (!loader.loadTensorColumnShards("blk.0.attn_q.weight",
     *                                     offsets, counts, dests)) {
     *     // Fallback: full load + slice
     *     auto full = loader.loadTensor("blk.0.attn_q.weight");
     *     // ... manual column extraction ...
     * }
     * @endcode
     *
     * @note Uses internal quantized block cache to optimize repeated access
     * @note Returns false for unsupported types - caller must fallback gracefully
     */
    bool loadTensorColumnShards(const std::string &tensor_name,
                                const std::vector<int> &col_offsets,
                                const std::vector<int> &col_counts,
                                const std::vector<float *> &dests);

    /**
     * @brief Stream load a contiguous row shard from a 2D tensor
     * @param tensor_name Tensor name
     * @param row_offset Starting row index
     * @param row_count Number of rows to load
     * @param dest Output buffer (must be pre-allocated: row_count × cols)
     * @return true if successful, false if unsupported
     *
     * @details Extracts [row_offset : row_offset+row_count, :] slice
     *   - Output layout: row-major, original column count preserved
     *   - Use for FFN down projection (row-sharded) or batch splitting
     *
     * **Limitations**: Same as loadTensorColumnShards (2D, supported types only)
     *
     * @note Less common than column sharding (attention uses column distribution)
     */
    bool loadTensorRowShard(const std::string &tensor_name,
                            int row_offset,
                            int row_count,
                            float *dest);

    /**
     * @brief Print model metadata summary to stdout
     *
     * Displays: architecture, layer count, hidden size, head counts, vocab size, etc.
     */
    void printModelInfo() const;

    /**
     * @brief Print all tensor metadata (names, shapes, types) to stdout
     *
     * Useful for understanding model structure and debugging tensor names.
     */
    void printTensorInfo() const;

    /**
     * @brief Get list of all tensor names in the model
     * @return Vector of tensor names (e.g., ["token_embd.weight", "blk.0.attn_q.weight", ...])
     */
    std::vector<std::string> getTensorNames() const;

    // ========================================================================
    // TESTING HOOKS - Expose dequant routines for unit tests
    // ========================================================================
    // These enable constructing synthetic quantized blocks and verifying decode logic.
    // NOT part of stable public API (subject to change).

    /**
     * @brief Dequantize Q4_K block data (testing hook)
     * @param data Raw quantized bytes
     * @param n_elements Number of elements to decode
     * @param type Quantization type (for validation)
     * @param tensor_name Tensor name (for diagnostics)
     * @return Dequantized FP32 values
     */
    std::vector<float> dequantizeQ4_K(const uint8_t *data, size_t n_elements, GGUFTensorType type, const std::string &tensor_name);

    /**
     * @brief Dequantize Q4_0 block data (testing hook)
     */
    std::vector<float> dequantizeQ4_0(const uint8_t *data, size_t n_elements);

    /**
     * @brief Dequantize Q4_1 block data (testing hook)
     */
    std::vector<float> dequantizeQ4_1(const uint8_t *data, const GGUFTensorInfo &info);

    /**
     * @brief Dequantize Q5_1 block data (testing hook)
     */
    std::vector<float> dequantizeQ5_1(const uint8_t *data, const GGUFTensorInfo &info);

    /**
     * @brief Dequantize Q5_0 block data (testing hook)
     */
    std::vector<float> dequantizeQ5_0(const uint8_t *data, const GGUFTensorInfo &info);

    /**
     * @brief Dequantize Q2_K block data (testing hook)
     */
    std::vector<float> dequantizeQ2_K(const uint8_t *data, GGUFTensorType type, const std::string &tensor_name, const GGUFTensorInfo &info);

    /**
     * @brief Dequantize Q3_K block data (testing hook)
     */
    std::vector<float> dequantizeQ3_K(const uint8_t *data, GGUFTensorType type, const std::string &tensor_name, const GGUFTensorInfo &info);

    /**
     * @brief Dequantize Q5_K block data (testing hook)
     */
    std::vector<float> dequantizeQ5_K(const uint8_t *data, GGUFTensorType type, const std::string &tensor_name, const GGUFTensorInfo &info);

    /**
     * @brief Dequantize Q6_K block data (testing hook)
     */
    std::vector<float> dequantizeQ6_K(const uint8_t *data, GGUFTensorType type, const std::string &tensor_name, const GGUFTensorInfo &info);

    /**
     * @brief Check if quantization type is supported
     * @param type Quantization type to check
     * @return true if ModelLoader can dequantize this type
     */
    bool supportsQuantization(GGUFTensorType type) const;

    /**
     * @brief Dequantize raw tensor data (internal dispatch)
     * @param tensor_info Tensor metadata
     * @param quantized_data Raw quantized bytes
     * @param tensor_name Tensor name (for diagnostics)
     * @return Dequantized FP32 values
     *
     * @note Dispatches to appropriate dequantizeQ*() method based on type
     */
    std::vector<float> dequantizeTensor(const GGUFTensorInfo &tensor_info,
                                        const std::vector<uint8_t> &quantized_data,
                                        const std::string &tensor_name);

    /**
     * @brief Create transformer layer configuration from model metadata
     * @return TransformerLayerConfig with hyperparameters
     *
     * @note Used by pipeline factories to initialize kernels
     */
    TransformerLayerConfig createLayerConfig() const;

    /**
     * @brief Clear the internal quantized block cache
     *
     * Frees memory used by cached quantized data blocks.
     * Call after loading phase if you want to reduce memory footprint.
     *
     * @note Does NOT clear dequantized tensor cache (those come from loadTensor calls)
     */
    void clearQuantShardCache();

    /**
     * @struct QuantShardCacheStats
     * @brief Statistics for quantized block cache
     */
    struct QuantShardCacheStats
    {
        size_t loads = 0;          ///< Total load operations attempted
        size_t cache_hits = 0;     ///< Loads served from cache
        size_t cache_misses = 0;   ///< Loads requiring file read
        size_t evictions = 0;      ///< Cache entries evicted (LRU)
        size_t bytes_resident = 0; ///< Current cache memory usage
    };

    /**
     * @brief Get quantized block cache statistics
     * @return Cache performance metrics
     *
     * @code
     * auto stats = loader.getQuantShardCacheStats();
     * double hit_rate = (double)stats.cache_hits / stats.loads;
     * LOG_INFO("Cache hit rate: " << (hit_rate * 100) << "%");
     * @endcode
     */
    QuantShardCacheStats getQuantShardCacheStats() const;

private:
    // ========================================================================
    // MEMBER VARIABLES
    // ========================================================================

    bool loaded_;               ///< Whether model has been successfully loaded
    std::string file_path_;     ///< Path to GGUF file
    GGUFModel model_;           ///< Parsed model metadata and tensor info
    std::ifstream file_stream_; ///< File handle for reading tensor data

    // ========================================================================
    // GGUF PARSING (called by loadModel)
    // ========================================================================

    /**
     * @brief Parse GGUF file header (magic, version, counts)
     * @return true if header is valid
     */
    bool parseHeader();

    /**
     * @brief Parse metadata key-value pairs
     * @return true if metadata is valid
     */
    bool parseMetadata();

    /**
     * @brief Parse tensor information array
     * @return true if tensor info is valid
     */
    bool parseTensorInfo();

    /**
     * @brief Validate parsed model (required fields present, etc.)
     * @return true if model is valid
     */
    bool validateModel();

    // ========================================================================
    // HELPER FUNCTIONS
    // ========================================================================

    /**
     * @brief Read a typed value from file stream
     * @tparam T Value type
     * @param value Output parameter
     * @return true if read succeeded
     */
    template <typename T>
    bool readValue(T &value);

    /**
     * @brief Read a GGUF string (length-prefixed UTF-8)
     * @param str Output string
     * @return true if read succeeded
     */
    bool readString(std::string &str);

    /**
     * @brief Read a GGUF array value
     * @param value Output GGUFValue containing array
     * @return true if read succeeded
     */
    bool readArray(GGUFValue &value);

    /**
     * @brief Get total file size in bytes
     * @return File size
     */
    size_t getFileSize() const;

    // ========================================================================
    // DEQUANTIZATION HELPERS
    // ========================================================================

    /**
     * @brief Dequantize Q8_0 block data
     * @param data Raw quantized bytes
     * @param n_elements Number of elements to decode
     * @param tensor_name Tensor name (for diagnostics)
     * @return Dequantized FP32 values
     *
     * @note Q8_0: 32 int8 values + 1 FP16 scale per block
     */
    std::vector<float> dequantizeQ8_0(const uint8_t *data, size_t n_elements, const std::string &tensor_name);

    /**
     * @brief Dequantize FP16 data to FP32
     * @param data Raw FP16 bytes
     * @param n_elements Number of elements
     * @return Dequantized FP32 values
     */
    std::vector<float> dequantizeF16(const uint8_t *data, size_t n_elements);

    /**
     * @brief Log dequantization statistics (if diagnostics enabled)
     * @param tensor_name Tensor name
     * @param type Quantization type
     * @param values Dequantized output
     * @param max_samples Maximum values to sample for logging
     *
     * @note Only logs if LLAMINAR_DEQUANT_STATS or LLAMINAR_DEQUANT_ANOMALIES set
     */
    void logDequantStats(const std::string &tensor_name, GGUFTensorType type, const std::vector<float> &values, size_t max_samples) const;

    // ========================================================================
    // POLYMORPHIC DEQUANTIZATION INTERFACE
    // ========================================================================
    // Strategy pattern for type-specific dequantization dispatch

    /**
     * @struct IDequantizer
     * @brief Abstract interface for dequantization strategies
     *
     * Allows type-specific dequantization logic without switch statements.
     */
    struct IDequantizer
    {
        virtual ~IDequantizer() = default;

        /**
         * @brief Dequantize tensor data
         * @param info Tensor metadata
         * @param data Raw quantized bytes
         * @param name Tensor name (for diagnostics)
         * @param loader ModelLoader reference (for calling helper methods)
         * @return Dequantized FP32 values
         */
        virtual std::vector<float> run(const GGUFTensorInfo &info, const std::vector<uint8_t> &data, const std::string &name, ModelLoader &loader) const = 0;
    };

    /**
     * @struct Q8_0Dequantizer
     * @brief Dequantizer for Q8_0 format
     */
    struct Q8_0Dequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &info, const std::vector<uint8_t> &data, const std::string &name, ModelLoader &loader) const override;
    };

    /**
     * @struct Q4_0Dequantizer
     * @brief Dequantizer for Q4_0 format
     */
    struct Q4_0Dequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &info, const std::vector<uint8_t> &data, const std::string &name, ModelLoader &loader) const override;
    };

    /**
     * @struct Q4KDequantizer
     * @brief Dequantizer for Q4_K and related K-quant formats
     */
    struct Q4KDequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &info, const std::vector<uint8_t> &data, const std::string &name, ModelLoader &loader) const override;
    };

    /**
     * @brief Select appropriate dequantizer for quantization type
     * @param type Quantization type
     * @return Dequantizer instance, or nullptr if unsupported
     */
    const IDequantizer *selectDequantizer(GGUFTensorType type) const;

    // ========================================================================
    // MODEL-SPECIFIC PARSING
    // ========================================================================

    /**
     * @brief Extract model-specific metadata into GGUFModel fields
     *
     * Parses metadata map to populate architecture, block_count, head_count, etc.
     * Called by parseMetadata() after reading raw key-value pairs.
     */
    void extractModelMetadata();

    // ========================================================================
    // QUANTIZED SHARD CACHE (Phase 1 optimization for streaming loads)
    // ========================================================================
    // LRU cache for full dequantized tensors to avoid redundant file reads
    // when loading multiple shards from the same tensor.

    /**
     * @struct CachedFullTensor
     * @brief Cache entry for a full dequantized tensor
     */
    struct CachedFullTensor
    {
        std::vector<float> data;  ///< Full FP32 tensor data
        std::vector<int> shape;   ///< Original tensor shape
        size_t bytes = 0;         ///< Memory usage (data.size() * sizeof(float))
        uint64_t last_access = 0; ///< Monotonic access counter (for LRU eviction)
    };

    mutable std::mutex quant_cache_mutex_;                                       ///< Protects cache data structures
    mutable std::unordered_map<std::string, CachedFullTensor> quant_full_cache_; ///< Tensor name → cached data
    mutable std::atomic<uint64_t> quant_cache_clock_{0};                         ///< Monotonic counter for LRU
    mutable std::atomic<size_t> quant_cache_bytes_{0};                           ///< Total cache memory usage
    mutable std::atomic<size_t> quant_cache_loads_{0};                           ///< Times cache was queried
    mutable std::atomic<size_t> quant_cache_hits_{0};                            ///< Times tensor found in cache
    mutable std::atomic<size_t> quant_cache_misses_{0};                          ///< Times tensor needed loading
    mutable std::atomic<size_t> quant_cache_evictions_{0};                       ///< Times entry evicted (LRU)

    /**
     * @brief Get or load full tensor into cache
     * @param tensor_name Tensor name
     * @param info Tensor metadata
     * @return Pointer to cached tensor, or nullptr on failure
     *
     * @details Implements LRU cache with configurable size limit:
     *   1. Check cache for existing entry
     *   2. If miss, dequantize full tensor
     *   3. Evict LRU entries if cache exceeds limit
     *   4. Insert new entry and return
     *
     * @note Thread-safe via internal mutex
     */
    const CachedFullTensor *getOrCacheFullQuantTensor(const std::string &tensor_name, const GGUFTensorInfo &info);

    /**
     * @brief Get maximum cache size in bytes
     * @return Max bytes (from env LLAMINAR_SHARD_CACHE_MAX_MB, default 512MB)
     */
    size_t quantShardCacheMaxBytes() const;
};

// ========================================================================
// STATIC ASSERTIONS - Validate ggml block structure sizes
// ========================================================================
// These ensure embedded ggml headers match expected layouts.
// If static_assert fails, ggml version may have diverged.

#ifdef GGML_QKK_MAX
static_assert(sizeof(block_q2_K) == 2 * sizeof(ggml_half) + QK_K / 16 + QK_K / 4,
              "block_q2_K size diverged from ggml");
static_assert(sizeof(block_q3_K) == sizeof(ggml_half) + QK_K / 4 + QK_K / 8 + 12,
              "block_q3_K size diverged from ggml");
static_assert(sizeof(block_q5_K) == 2 * sizeof(ggml_half) + K_SCALE_SIZE + QK_K / 2 + QK_K / 8,
              "block_q5_K size diverged from ggml");
static_assert(sizeof(block_q6_K) == sizeof(ggml_half) + QK_K / 16 + 3 * QK_K / 4,
              "block_q6_K size diverged from ggml");
#endif