#pragma once

#include "TensorBase.h"
#include "SimpleTensor.h"
#include "CosmaTensor.h"
#include "ShardedSimpleTensor.h"
#include "ShardSpec.h"
#include "BF16Tensor.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>
#include <string>

// Forward declarations are now replaced with full includes above

namespace llaminar
{
    // ================= Quantized Tensor Skeleton (Phase 1) =================
    /**
     * @brief Lightweight enumeration of supported quantization formats.
     * Extend as additional formats become supported. F32/F16 act as passthrough.
     */
    enum class QuantFormat
    {
        F32,
        F16,
        Q4_0,
        Q5_0,
        Q8_0,
        Q4_K,
        Q5_K,
        Q6_K,
        Q8_K
    };

    /**
     * @brief Descriptor for a single quantization block.
     * Holds structural parameters used during decode.
     */
    struct QuantBlockDescriptor
    {
        int elements_per_block{0}; ///< Logical elements represented
        int bytes_per_block{0};    ///< Total bytes (scale headers + packed payload)
        int scale_count{1};        ///< Number of scale values per block
        int bits_per_value{0};     ///< Bit width per stored value (4,6,8,...)
        bool is_k_quant{false};    ///< True if K‑quant super-block semantics apply
    };

    /**
     * @brief Layout metadata for an entire quantized tensor.
     */
    struct QuantStorageLayout
    {
        QuantFormat format{QuantFormat::F32};
        std::vector<int> original_shape; ///< 2D matrix shape [rows, cols]
        size_t total_blocks{0};          ///< Number of blocks covering tensor
        QuantBlockDescriptor block_desc; ///< Format-specific descriptor
    };

    /**
     * @brief Quantized tensor retaining raw bytes; dequantization happens on demand inside kernels.
     * No contiguous FP32 buffer is exposed to discourage accidental full expansion.
     */
    class QuantizedTensor : public TensorBase
    {
    public:
        QuantizedTensor(const QuantStorageLayout &layout, std::vector<uint8_t> raw)
            : layout_(layout), raw_(std::move(raw)) {}

        const std::vector<int> &shape() const override { return layout_.original_shape; }

        /**
         * @brief Get FP32 data pointer (uses pull-through cache for dequantization)
         * WARNING: Expensive! Dequantizes entire tensor on cache miss.
         * Prefer using weight_data() for operators that can work with quantized data.
         */
        float *data() override
        {
            // Use pull-through cache for full tensor dequantization
            return const_cast<float *>(data_fp32());
        }

        const float *data() const override
        {
            // Use pull-through cache for full tensor dequantization
            return data_fp32();
        }

        int size() const override { return layout_.original_shape.empty() ? 0 : layout_.original_shape[0] * layout_.original_shape[1]; }
        std::string type_name() const override { return "QuantizedTensor"; }
        int ndim() const override { return static_cast<int>(layout_.original_shape.size()); }
        bool is_distributed() const override { return false; }
        void zero() override { /* no-op: quantized raw stays as-is */ }
        void fill(float) override { /* unsupported for quantized storage */ }
        std::shared_ptr<TensorBase> copy() const override { return std::make_shared<QuantizedTensor>(layout_, raw_); }
        void copy_from(const TensorBase &other) override { (void)other; /* no-op for now */ }

        const QuantStorageLayout &layout() const { return layout_; }
        const uint8_t *raw() const { return raw_.data(); }

        // === Decode Utilities (slow path prototypes) ===
        void decodeBlock(size_t block_index, float *dst) const; ///< Decode one block to FP32
        // Note: BF16 (bfloat16) decoding is handled via decodeTileBF16() in slab allocator (see QuantizedSlabAllocator)

        // ===== Cache Statistics (for benchmarking) =====
        struct CacheStats
        {
            uint64_t block_requests{0};      // total block fetch attempts
            uint64_t block_hits{0};          // served from cache
            uint64_t block_misses{0};        // decoded blocks
            uint64_t full_block_fastpath{0}; // times full-block tile copy path taken
        };

        // Enable/disable statistics globally (thread-safe atomic flag access)
        static void enable_cache_stats(bool v);
        static void reset_cache_stats();
        static CacheStats cache_stats();

        // ========== Pull-Through Cache Interface ==========

        TensorDataType native_type() const override { return TensorDataType::QUANTIZED; }

        size_t element_count() const override
        {
            return layout_.original_shape.empty() ? 0 : static_cast<size_t>(layout_.original_shape[0] * layout_.original_shape[1]);
        }

    protected:
        /**
         * @brief No native FP32 pointer - quantized storage only
         * This forces all accesses through the decode path (cache).
         */
        const float *data_native_fp32() const override { return nullptr; }

        /**
         * @brief No native BF16 pointer - quantized storage only
         */
        const void *data_native_bf16() const override { return nullptr; }

        /**
         * @brief Decode entire tensor to FP32 (for cache miss)
         * Called by TensorBase::data_fp32() via QuantSlabCache when no cached FP32 exists.
         */
        void decode_to_fp32(float *dst) const override
        {
            if (!dst)
                return;

            const auto &desc = layout_.block_desc;
            if (desc.elements_per_block <= 0)
            {
                return;
            }

            size_t total_elements = element_count();
            size_t num_blocks = (total_elements + desc.elements_per_block - 1) / desc.elements_per_block;

#pragma omp parallel for if (num_blocks > 10)
            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
            {
                size_t offset = block_idx * desc.elements_per_block;
                size_t remaining = total_elements - offset;
                size_t block_size = std::min(static_cast<size_t>(desc.elements_per_block), remaining);

                decodeBlock(block_idx, dst + offset);
            }
        }

        /**
         * @brief Decode quantized to BF16 (conversion via FP32 intermediate)
         * Called by TensorBase::data_bf16() via QuantSlabCache when BF16 needed.
         * Note: Could be optimized later to decode directly to BF16 in QuantSlabCache.
         */
        void decode_to_bf16(void *dst) const override
        {
            // Decode to FP32 first, then convert to BF16
            size_t count = element_count();
            std::vector<float> fp32_temp(count);
            decode_to_fp32(fp32_temp.data());

            bfloat16 *bf16_dst = static_cast<bfloat16 *>(dst);
#pragma omp parallel for if (count > 10000)
            for (size_t i = 0; i < count; ++i)
            {
                bf16_dst[i] = bfloat16::from_float(fp32_temp[i]);
            }
        }

    public:
    private:
        QuantStorageLayout layout_;
        std::vector<uint8_t> raw_;
        // Per-process global counters
        static std::atomic<bool> stats_enabled_;
        static std::atomic<uint64_t> stat_block_requests_;
        static std::atomic<uint64_t> stat_block_hits_;
        static std::atomic<uint64_t> stat_block_misses_;
        static std::atomic<uint64_t> stat_full_block_fastpath_;
    };

    /**
     * TensorFactory provides utilities for creating and converting between
     * different tensor types (SimpleTensor, COSMATensor) and the legacy Tensor struct.
     */
    class TensorFactory
    {
    public:
        // Simple tensor creation
        static std::shared_ptr<llaminar::TensorBase> create_simple(const std::vector<int> &shape);
        static std::shared_ptr<llaminar::TensorBase> create_simple(const std::vector<int> &shape,
                                                                   const std::vector<float> &data);

        // COSMA tensor creation
        static std::shared_ptr<llaminar::TensorBase> create_cosma(const std::vector<int> &shape,
                                                                  const std::string &label = "",
                                                                  int mpi_rank = -1);

        // Automatic tensor type selection
        static std::shared_ptr<llaminar::TensorBase> create_auto(const std::vector<int> &shape,
                                                                 bool prefer_distributed = false);
        static std::shared_ptr<llaminar::TensorBase> create_auto(const std::vector<int> &shape,
                                                                 const std::vector<float> &data = {});

        // Iterate currently registered sharded tensors (non-owning). Callback receives const ShardSpec&.
        static void for_each_sharded(const std::function<void(const ShardSpec &, const std::vector<int> &)> &fn);

        // Assign role metadata to last created sharded tensor (helper for loader/pipeline)
        static void set_last_shard_role(const std::string &role);

        // Sharded tensor creation (1D partition along given axis index within shape)
        // axis_kind selects semantic axis (Hidden or Heads); axis_dim_index indicates which dimension in shape is split.
        static std::shared_ptr<llaminar::TensorBase> create_sharded(const std::vector<int> &shape,
                                                                    ShardSpec::Axis axis_kind,
                                                                    int axis_dim_index,
                                                                    int world,
                                                                    int rank);

        // Head-aligned sharded creation: ensure partition boundaries align to whole heads (head_dim elements).
        // global_dim = n_heads * head_dim, axis_dim_index indicates which dimension encodes that product.
        // Distributes heads with near-even strategy (base + remainder) so some ranks may have +1 head.
        static std::shared_ptr<llaminar::TensorBase> create_heads_sharded(const std::vector<int> &shape,
                                                                          int axis_dim_index,
                                                                          int n_heads,
                                                                          int head_dim,
                                                                          int world,
                                                                          int rank);

        // Tensor type conversion
        static std::shared_ptr<llaminar::TensorBase> convert_to_cosma(const std::shared_ptr<llaminar::TensorBase> &tensor,
                                                                      const std::string &label = "",
                                                                      int mpi_rank = -1);
        static std::shared_ptr<llaminar::TensorBase> convert_to_simple(const std::shared_ptr<llaminar::TensorBase> &tensor);

        // Type-specific accessors
        static std::shared_ptr<SimpleTensor> to_simple_tensor(std::shared_ptr<llaminar::TensorBase> tensor);
        static std::shared_ptr<COSMATensor> to_cosma_tensor(std::shared_ptr<llaminar::TensorBase> tensor);

        // ================= BF16 Tensor Creation (Phase 5) =================
        /**
         * @brief Create a BF16 tensor with given shape (zero-initialized)
         * @param shape Tensor dimensions (e.g., {seq_len, d_model})
         * @return Shared pointer to BF16Tensor
         */
        static std::shared_ptr<llaminar::TensorBase> create_bf16(const std::vector<int> &shape);

        /**
         * @brief Create a BF16 tensor from FP32 data
         * @param shape Tensor dimensions
         * @param fp32_data FP32 source data (will be converted to BF16)
         * @return Shared pointer to BF16Tensor
         */
        static std::shared_ptr<llaminar::TensorBase> create_bf16(const std::vector<int> &shape,
                                                                 const std::vector<float> &fp32_data);

        /**
         * @brief Convert existing tensor to BF16
         * @param tensor Source tensor (will be converted to BF16)
         * @return Shared pointer to BF16Tensor
         */
        static std::shared_ptr<llaminar::TensorBase> convert_to_bf16(const std::shared_ptr<llaminar::TensorBase> &tensor);

        // Type-specific BF16 accessor
        static std::shared_ptr<BF16Tensor> to_bf16_tensor(std::shared_ptr<llaminar::TensorBase> tensor);

        // ================= Quantized Tensor Helpers (Phase 1 Skeleton) =================
        /**
         * @brief Create a quantized tensor from raw packed bytes.
         * @param shape Logical 2D shape [rows, cols]
         * @param format Quantization format (Q4_0, Q6_K, Q8_0, etc.)
         * @param raw Raw contiguous byte storage for all blocks
         */
        static std::shared_ptr<llaminar::TensorBase> create_quantized(const std::vector<int> &shape,
                                                                      QuantFormat format,
                                                                      const std::vector<uint8_t> &raw);

        /**
         * @brief Return true if tensor is a QuantizedTensor.
         */
        static bool is_quantized(const std::shared_ptr<llaminar::TensorBase> &t);

        /**
         * @brief Cast to QuantizedTensor (nullptr if not quantized).
         */
        static std::shared_ptr<QuantizedTensor> to_quantized(std::shared_ptr<llaminar::TensorBase> t);

    private:
        TensorFactory() = default; // Static class, no instantiation
    };

} // namespace llaminar