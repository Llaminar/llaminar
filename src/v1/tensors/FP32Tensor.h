/**
 * @file FP32Tensor.h
 * @brief Standard FP32 tensor with BLAS-based GEMM support.
 *
 * This is the new standard FP32 tensor type that replaces SimpleTensor.
 * It implements ITensorGemm for efficient matrix multiplications via BLAS.
 *
 * Migration from SimpleTensor:
 * - FP32Tensor has the same API as SimpleTensor
 * - Adds createGemm() support for unified GEMM interface
 * - NUMA-aware allocation (same as SimpleTensor)
 * - Zero-copy data access
 *
 * @author David Sanftenberg
 */

#pragma once

#include "TensorBase.h"
#include "../Logger.h"
#include "../utils/DebugEnv.h"
#include "../utils/BFloat16.h"
#include "../ITensorGemm.h"
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <omp.h>
#include <cblas.h>

namespace llaminar
{

    // Forward declaration
    class FP32Gemm;

    /**
     * @brief Standard FP32 in-memory tensor implementation.
     *
     * Replaces SimpleTensor with unified GEMM interface support.
     * Provides NUMA-aware allocation and efficient BLAS-based matrix operations.
     */
    class FP32Tensor : public TensorBase
    {
    private:
        std::vector<float> data_;
        std::vector<int> shape_;

        /**
         * @brief Perform NUMA-aware first-touch initialization on allocated memory
         *
         * Uses OpenMP parallel loops to ensure memory pages are allocated on the NUMA node
         * where they will be accessed by worker threads. This eliminates remote NUMA access
         * penalties (2-3x latency) for large tensors like K/V caches.
         *
         * @param data Pointer to memory buffer to initialize
         * @param size Number of elements to initialize
         * @param init_value Value to initialize elements to (default 0.0f)
         */
        static void numaFirstTouch(float *data, size_t size, float init_value = 0.0f)
        {
            const auto &env = debugEnv();

            // Skip if disabled via environment
            if (!env.loader.numa_first_touch)
            {
                std::fill(data, data + size, init_value);
                return;
            }

            // Small allocations: single-threaded (overhead not worth it)
            constexpr size_t kSmallThreshold = 32 * 1024; // 128KB (32K floats)
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
                LOG_DEBUG("[FP32Tensor-NUMA] First-touch completed for " << size
                                                                         << " elements (" << (size * sizeof(float) / 1024.0 / 1024.0)
                                                                         << " MB) using " << omp_get_max_threads() << " threads");
            }
        }

    public:
        // Constructors
        FP32Tensor() = default;

        explicit FP32Tensor(const std::vector<int> &dims) : shape_(dims)
        {
            int total_size = 1;
            for (int dim : dims)
            {
                if (dim < 0)
                {
                    throw std::invalid_argument("Tensor dimensions must be non-negative");
                }
                total_size *= dim;
            }

            // Allocate memory (without initialization to avoid duplicate work)
            data_.resize(total_size);

            // NUMA-aware first-touch initialization
            numaFirstTouch(data_.data(), total_size, 0.0f);
        }

        FP32Tensor(const std::vector<int> &dims, const std::vector<float> &values)
            : shape_(dims), data_(values)
        {
            if (total_elements() != static_cast<int>(values.size()))
            {
                throw std::invalid_argument("Data size does not match tensor shape");
            }
        }

        // TensorBase interface implementation
        const std::vector<int> &shape() const override { return shape_; }

        int size() const override
        {
            return data_.size();
        }

        int ndim() const override { return shape_.size(); }

        float *data() override { return data_.data(); }
        const float *data() const override { return data_.data(); }

        // Pull-through cache interface (fast path for FP32)
        const float *data_fp32() const override { return data_.data(); }

        TensorDataType native_type() const override { return TensorDataType::FP32; }

        size_t element_count() const override { return data_.size(); }

        std::string type_name() const override { return "FP32Tensor"; }
        bool is_distributed() const override { return false; }

        void zero() override
        {
            std::fill(data_.begin(), data_.end(), 0.0f);
        }

        void fill(float value) override
        {
            std::fill(data_.begin(), data_.end(), value);
        }

        // Basic tensor operations
        std::shared_ptr<TensorBase> copy() const override
        {
            return std::make_shared<FP32Tensor>(shape_, data_);
        }

        void copy_from(const TensorBase &other) override
        {
            if (!is_compatible_shape(other))
            {
                throw std::invalid_argument("Incompatible tensor shapes for copy");
            }

            // If other is also an FP32Tensor, we can do efficient copy
            if (auto fp32_other = dynamic_cast<const FP32Tensor *>(&other))
            {
                data_ = fp32_other->data_;
            }
            else
            {
                // Generic copy using data() accessor
                const float *src = other.data();
                std::copy(src, src + size(), data_.data());
            }
        }

    protected:
        // Fast path: FP32Tensor is natively FP32, return direct pointer
        const float *data_native_fp32() const override
        {
            return data_.data();
        }

        // Decode methods (required by TensorBase)
        void decode_to_fp32(float *dst) const override
        {
            // Already FP32, just copy
            std::memcpy(dst, data_.data(), data_.size() * sizeof(float));
        }

        void decode_to_bf16(void *dst) const override
        {
            // Convert FP32 → BF16
            bfloat16 *bf16_dst = static_cast<bfloat16 *>(dst);

#pragma omp parallel for if (data_.size() > 10000)
            for (size_t i = 0; i < data_.size(); ++i)
            {
                bf16_dst[i] = bfloat16::from_float(data_[i]);
            }
        }

    public:
        // Utility methods
        int total_elements() const
        {
            int total = 1;
            for (int dim : shape_)
            {
                total *= dim;
            }
            return total;
        }

        void resize(const std::vector<int> &new_shape)
        {
            int new_size = 1;
            for (int dim : new_shape)
            {
                new_size *= dim;
            }

            data_.resize(new_size);
            shape_ = new_shape;

            // NUMA first-touch for large tensors
            numaFirstTouch(data_.data(), new_size, 0.0f);
        }

        // Element access
        float &operator()(int i) { return data_[i]; }
        const float &operator()(int i) const { return data_[i]; }

        float &operator()(int i, int j)
        {
            return data_[i * shape_[1] + j];
        }

        const float &operator()(int i, int j) const
        {
            return data_[i * shape_[1] + j];
        }

        // ITensorGemm interface - create FP32-specific GEMM implementation
        ITensorGemm *createGemmRaw() const override;
    };

    /**
     * @brief FP32-specific GEMM implementation using BLAS.
     *
     * Provides efficient matrix multiplication for FP32 tensors via OpenBLAS.
     */
    class FP32Gemm : public ITensorGemm
    {
    private:
        const FP32Tensor *tensor_;

    public:
        explicit FP32Gemm(const FP32Tensor *tensor) : tensor_(tensor) {}

        bool multiply(const float *A, float *C,
                      int m, int n, int k,
                      bool transpose_B,
                      float alpha,
                      float beta,
                      int row_offset = 0,
                      int row_count = -1) override
        {
            // Default row_count to n if not specified
            if (row_count < 0)
                row_count = n;

            // Validate row range
            const auto &shape = tensor_->shape();
            if (row_offset < 0 || row_offset + row_count > shape[0])
            {
                LOG_ERROR("FP32Gemm: Invalid row range [" << row_offset << ", "
                                                          << (row_offset + row_count) << ") for tensor with "
                                                          << shape[0] << " rows");
                return false;
            }

            const float *B = tensor_->data();

            // For FP32 tensors with row_offset, adjust B pointer to start at the offset row
            // B is [total_rows, k], we want to use rows [row_offset : row_offset+row_count]
            const float *B_subset = B + row_offset * shape[1];

            if (shape.size() != 2)
            {
                LOG_ERROR("FP32Gemm: Weight tensor must be 2D, got " << shape.size() << "D");
                return false;
            }

            // Validate column dimension only (rows are handled by row_offset/row_count)
            int expected_cols = transpose_B ? k : n;
            if (shape[1] != expected_cols)
            {
                LOG_ERROR("FP32Gemm: Weight column mismatch: expected " << expected_cols
                                                                        << ", got " << shape[1]);
                return false;
            }

            // Perform GEMM: C = alpha * A @ B_subset^T + beta * C
            // A: [m, k], B_subset: [row_count, k] (if transpose_B=true), C: [m, row_count]
            if (transpose_B)
            {
                // B_subset is stored as [row_count, k], we want B_subset^T which is [k, row_count]
                // cblas_sgemm computes: C = alpha * op(A) * op(B) + beta * C
                // We want: C = alpha * A * B_subset^T + beta * C
                // So: op(A) = A (no transpose), op(B) = B_subset^T (transpose)
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            m, row_count, k,
                            alpha,
                            A, k,        // A: [m, k], lda = k
                            B_subset, k, // B_subset: [row_count, k], ldb = k (will be transposed)
                            beta,
                            C, row_count); // C: [m, row_count], ldc = row_count
            }
            else
            {
                // B_subset is stored as [k, row_count], use as-is
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            m, row_count, k,
                            alpha,
                            A, k,                // A: [m, k], lda = k
                            B_subset, row_count, // B_subset: [k, row_count], ldb = row_count
                            beta,
                            C, n); // C: [m, n], ldc = n
            }

            return true;
        }

        bool supports(int m, int n, int k) const override
        {
            // BLAS supports all reasonable sizes
            return m > 0 && n > 0 && k > 0;
        }

        const char *name() const override
        {
            return "FP32_BLAS_Gemm";
        }

        bool supports_bf16() const override
        {
            // FP32Gemm only supports FP32 activations
            // BF16 support would require a separate BF16Gemm class
            return false;
        }
    };

    // Implementation of createGemmRaw
    inline ITensorGemm *FP32Tensor::createGemmRaw() const
    {
        return new FP32Gemm(this);
    }

} // namespace llaminar
