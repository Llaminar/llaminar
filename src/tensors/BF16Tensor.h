/**
 * @file BF16Tensor.h
 * @brief Tensor storing activations in BF16 format for 2× memory reduction
 * @author David Sanftenberg
 * @date October 20, 2025
 *
 * BF16 (bfloat16) format offers:
 * - Same exponent range as FP32 (8 bits) → prevents overflow/underflow
 * - Reduced mantissa precision (7 bits vs 23 bits) → ~3-4 decimal digits
 * - Direct truncation/rounding for FP32↔BF16 conversion
 * - Hardware acceleration on Ice Lake+, Zen 4+
 *
 * Use cases:
 * - Q/K/V projection outputs
 * - Attention context vectors
 * - FFN intermediate activations (gate, up, down)
 *
 * NOT recommended for (use FP32 accumulation):
 * - Softmax computations
 * - RMSNorm denominators
 * - Final logits
 */

#pragma once

#include "TensorBase.h"
#include "../utils/BFloat16.h"
#include "../utils/DebugEnv.h"
#include "../Logger.h"
#include <vector>
#include <memory>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <omp.h>

namespace llaminar
{

    class BF16Tensor : public TensorBase
    {
    private:
        std::vector<bfloat16> data_;
        std::vector<int> shape_;

        /**
         * @brief NUMA-aware first-touch initialization for BF16 tensors
         *
         * Uses OpenMP parallel loops to ensure memory pages are allocated on the NUMA node
         * where they will be accessed by worker threads. This eliminates remote NUMA access
         * penalties (2-3x latency) for large tensors.
         *
         * @param data Pointer to BF16 memory buffer to initialize
         * @param size Number of BF16 elements to initialize
         * @param init_value Value to initialize elements to (default 0.0)
         */
        static void numaFirstTouch(bfloat16 *data, size_t size, bfloat16 init_value = bfloat16(0.0f))
        {
            const auto &env = debugEnv();

            // Skip if disabled via environment
            if (!env.loader.numa_first_touch)
            {
                std::fill(data, data + size, init_value);
                return;
            }

            // Small allocations: single-threaded (overhead not worth it)
            // Threshold: 128KB = 64K bfloat16 elements (2 bytes each)
            constexpr size_t kSmallThreshold = 64 * 1024;
            if (size < kSmallThreshold)
            {
                std::fill(data, data + size, init_value);
                return;
            }

// Large allocations: parallel first-touch for NUMA locality
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < size; ++i)
            {
                data[i] = init_value;
            }

            // Optional: Verify NUMA locality (expensive, debug only)
            if (env.loader.numa_verify_locality && size >= kSmallThreshold)
            {
                LOG_DEBUG("[BF16Tensor-NUMA] First-touch completed for " << size
                                                                         << " BF16 elements (" << (size * sizeof(bfloat16) / 1024.0 / 1024.0)
                                                                         << " MB) using " << omp_get_max_threads() << " threads");
            }
        }

    public:
        // ========== Constructors ==========

        BF16Tensor() = default;

        /**
         * @brief Construct BF16 tensor with given shape (zero-initialized)
         * @param dims Shape dimensions (e.g., {seq_len, d_model})
         */
        explicit BF16Tensor(const std::vector<int> &dims) : shape_(dims)
        {
            size_t total_size = 1;
            for (int dim : dims)
            {
                if (dim < 0)
                {
                    throw std::invalid_argument("BF16Tensor dimensions must be non-negative");
                }
                total_size *= static_cast<size_t>(dim);
            }

            // Allocate memory (without initialization to avoid duplicate work)
            data_.resize(total_size);

            // NUMA-aware first-touch initialization
            numaFirstTouch(data_.data(), total_size, bfloat16(0.0f));
        }

        /**
         * @brief Construct BF16 tensor from FP32 data
         * @param dims Shape dimensions
         * @param fp32_data FP32 source data (will be converted to BF16)
         */
        BF16Tensor(const std::vector<int> &dims, const std::vector<float> &fp32_data)
            : shape_(dims)
        {
            size_t total_size = 1;
            for (int dim : dims)
            {
                total_size *= static_cast<size_t>(dim);
            }

            if (fp32_data.size() != total_size)
            {
                throw std::invalid_argument("BF16Tensor: data size mismatch");
            }

            data_.resize(total_size);
            from_fp32(fp32_data.data(), total_size);
        }

        /**
         * @brief Construct BF16 tensor from BF16 data
         * @param dims Shape dimensions
         * @param bf16_data BF16 source data
         */
        BF16Tensor(const std::vector<int> &dims, const std::vector<bfloat16> &bf16_data)
            : shape_(dims), data_(bf16_data)
        {
            size_t total_size = 1;
            for (int dim : dims)
            {
                total_size *= static_cast<size_t>(dim);
            }

            if (bf16_data.size() != total_size)
            {
                throw std::invalid_argument("BF16Tensor: data size mismatch");
            }
        }

        /**
         * @brief Move constructor - transfer ownership of data
         */
        BF16Tensor(BF16Tensor &&other) noexcept
            : data_(std::move(other.data_)), shape_(std::move(other.shape_))
        {
        }

        /**
         * @brief Copy constructor - create independent copy
         */
        BF16Tensor(const BF16Tensor &other)
            : data_(other.data_), shape_(other.shape_)
        {
        }

        // ========== TensorBase Interface ==========

        /**
         * @brief Get FP32 data pointer (uses pull-through cache)
         * Automatically decoded via shared LRU cache when needed.
         * For best performance, use bf16_data() directly when possible.
         */
        float *data() override
        {
            // Use pull-through cache for FP32 conversion
            return const_cast<float *>(data_fp32());
        }

        const float *data() const override
        {
            // Use pull-through cache for FP32 conversion
            return data_fp32();
        }

        const std::vector<int> &shape() const override { return shape_; }

        int size() const override
        {
            int total = 1;
            for (int dim : shape_)
            {
                total *= dim;
            }
            return total;
        }

        int ndim() const override { return static_cast<int>(shape_.size()); }

        std::string type_name() const override { return "BF16Tensor"; }

        bool is_distributed() const override { return false; }

        void zero() override
        {
            std::fill(data_.begin(), data_.end(), bfloat16(0.0f));
            // Cache invalidation handled automatically by QuantSlabCache
        }

        void fill(float value) override
        {
            bfloat16 bf16_value = bfloat16::from_float(value);
            std::fill(data_.begin(), data_.end(), bf16_value);
            // Cache invalidation handled automatically by QuantSlabCache
        }

        std::shared_ptr<TensorBase> copy() const override
        {
            auto result = std::make_shared<BF16Tensor>(shape_, data_);
            return result;
        }

        void copy_from(const TensorBase &other) override
        {
            if (other.shape() != shape_)
            {
                throw std::invalid_argument("BF16Tensor::copy_from: shape mismatch");
            }

            // Check if source is also BF16 (fast path)
            const BF16Tensor *other_bf16 = dynamic_cast<const BF16Tensor *>(&other);
            if (other_bf16)
            {
                data_ = other_bf16->data_;
            }
            else
            {
                // Convert from FP32
                from_fp32(other.data(), static_cast<size_t>(other.size()));
            }

            // Cache invalidation handled automatically by QuantSlabCache
        }

        // ========== Pull-Through Cache Interface ==========

        TensorDataType native_type() const override { return TensorDataType::BF16; }

        size_t element_count() const override
        {
            return data_.size();
        }

    protected:
        /**
         * @brief Fast path for BF16 access (returns direct pointer)
         * Called by TensorBase::data_bf16() - zero overhead for native type!
         */
        const void *data_native_bf16() const override { return data_.data(); }

        /**
         * @brief Decode BF16 to FP32 (for cache miss)
         * Called by TensorBase::data_fp32() via QuantSlabCache when no cached FP32 exists.
         */
        void decode_to_fp32(float *dst) const override
        {
            if (!dst || data_.empty())
                return;

#pragma omp parallel for if (data_.size() > 10000)
            for (size_t i = 0; i < data_.size(); ++i)
            {
                dst[i] = static_cast<float>(data_[i]);
            }
        }

        /**
         * @brief Decode BF16 to BF16 (trivial copy)
         * Called by TensorBase::data_bf16() via QuantSlabCache (rare - usually fast path used).
         */
        void decode_to_bf16(void *dst) const override
        {
            std::memcpy(dst, data_.data(), data_.size() * sizeof(bfloat16));
        }

    public:
        // ========== BF16-Specific Interface ==========

        /**
         * @brief Get native BF16 data pointer (preferred for BF16-aware operators)
         */
        bfloat16 *bf16_data() { return data_.data(); }
        const bfloat16 *bf16_data() const { return data_.data(); }

        /**
         * @brief Convert FP32 data to BF16 and store
         * @param fp32_data Source FP32 data
         * @param count Number of elements to convert
         */
        void from_fp32(const float *fp32_data, size_t count)
        {
            if (count != data_.size())
            {
                throw std::invalid_argument("BF16Tensor::from_fp32: size mismatch");
            }

#pragma omp parallel for if (count > 10000)
            for (size_t i = 0; i < count; ++i)
            {
                data_[i] = bfloat16::from_float(fp32_data[i]);
            }

            // Cache invalidation handled automatically by QuantSlabCache
        }

        /**
         * @brief Convert BF16 data to FP32
         * @param fp32_data Destination FP32 buffer
         * @param count Number of elements to convert
         */
        void to_fp32(float *fp32_data, size_t count) const
        {
            if (count != data_.size())
            {
                throw std::invalid_argument("BF16Tensor::to_fp32: size mismatch");
            }

#pragma omp parallel for if (count > 10000)
            for (size_t i = 0; i < count; ++i)
            {
                fp32_data[i] = static_cast<float>(data_[i]);
            }
        }

        // ========== Batch Operations ==========

        /**
         * @brief Get batch size (first dimension for 3D tensors, 1 otherwise)
         */
        int batch_size() const
        {
            if (shape_.size() >= 3)
            {
                return shape_[0]; // [batch, seq, hidden]
            }
            return 1; // [seq, hidden] or [hidden] → implicit batch=1
        }

        /**
         * @brief Extract single sequence from batch tensor
         * @param batch_idx Batch index to extract
         * @return New BF16Tensor containing the extracted sequence
         */
        std::shared_ptr<BF16Tensor> get_batch(int batch_idx) const
        {
            if (shape_.size() < 3)
            {
                throw std::runtime_error("BF16Tensor::get_batch requires 3D tensor [batch, seq, hidden]");
            }

            const int batch = shape_[0];
            const int seq = shape_[1];
            const int hidden = shape_[2];

            if (batch_idx < 0 || batch_idx >= batch)
            {
                throw std::out_of_range("BF16Tensor::get_batch: batch_idx out of range");
            }

            // Create new 2D tensor [seq, hidden]
            std::vector<int> new_shape = {seq, hidden};
            auto result = std::make_shared<BF16Tensor>(new_shape);

            // Copy data for this batch
            const size_t seq_size = static_cast<size_t>(seq * hidden);
            const size_t offset = static_cast<size_t>(batch_idx) * seq_size;

            std::copy(data_.begin() + offset,
                      data_.begin() + offset + seq_size,
                      result->data_.begin());

            return result;
        }

        /**
         * @brief Stack multiple sequence tensors into a batch tensor
         * @param sequences Vector of 2D tensors [seq, hidden] to stack
         * @return New 3D BF16Tensor [batch, seq, hidden]
         */
        static std::shared_ptr<BF16Tensor> stack_batch(
            const std::vector<std::shared_ptr<BF16Tensor>> &sequences)
        {
            if (sequences.empty())
            {
                return nullptr;
            }

            // Validate all sequences have same shape
            const auto &ref_shape = sequences[0]->shape();
            for (size_t i = 1; i < sequences.size(); ++i)
            {
                if (sequences[i]->shape() != ref_shape)
                {
                    throw std::invalid_argument("BF16Tensor::stack_batch: all sequences must have same shape");
                }
            }

            // Create batched tensor [batch, seq, hidden]
            const int batch = static_cast<int>(sequences.size());
            std::vector<int> batched_shape = {batch};
            batched_shape.insert(batched_shape.end(), ref_shape.begin(), ref_shape.end());

            auto result = std::make_shared<BF16Tensor>(batched_shape);

            // Copy each sequence into batch
            const size_t seq_size = sequences[0]->data_.size();
            for (int b = 0; b < batch; ++b)
            {
                const size_t offset = static_cast<size_t>(b) * seq_size;
                std::copy(sequences[b]->data_.begin(),
                          sequences[b]->data_.end(),
                          result->data_.begin() + offset);
            }

            return result;
        }
    };

} // namespace llaminar
