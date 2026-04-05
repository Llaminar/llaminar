/**
 * @file ActivationRotation.h
 * @brief Block-diagonal orthogonal rotation for activation kurtosis reduction
 *
 * Applies a deterministic block-diagonal rotation to FP32 activation vectors
 * before Q8_1 quantization. The rotation spreads outlier energy uniformly
 * across quantization blocks, dramatically reducing kurtosis (e.g., 1191→28
 * for Qwen3.5-4B) and improving int8 fidelity.
 *
 * Mathematical basis:
 *   For GEMM y = X @ W^T:
 *     X' = X @ R     (rotate activations before Q8_1 quantization)
 *     W' = W @ R     (pre-rotate weights once at load time)
 *     X' @ W'^T = XR(WR)^T = XRR^TW^T = XW^T = y  (R orthogonal ⟹ RR^T = I)
 *
 * The rotation is block-diagonal: R = diag(R₁, R₂, ..., Rₙ) where each Rᵢ
 * is a block_dim × block_dim orthogonal matrix. This keeps the cost low
 * (~1M FLOPs for 2560-dim with block_dim=128) while capturing most of the
 * kurtosis reduction benefit vs a full-dimension rotation.
 *
 * Two rotation contexts are needed per model:
 *   - hidden_dim rotation: for QKV, Wo, Gate, Up, LM Head projections
 *   - ffn_dim rotation: for FFN Down projection
 *
 * Usage:
 *   // At model load time:
 *   auto rot = std::make_shared<ActivationRotation>(hidden_dim, 128);
 *
 *   // Before Q8_1 quantization (activation path):
 *   rot->rotate_inplace(fp32_row, hidden_dim);
 *
 *   // During weight preparation (one-time):
 *   rot->rotate_weight_rows(weight_data, n_rows, k_dim);
 *
 *   // Create rotated Q8_0 weight copy:
 *   auto rotated = rot->create_rotated_weight(original_q8_0_tensor);
 */

#pragma once

#include "kernels/cpu/turboquant/TurboQuantRotation.h"
#include "tensors/TensorClasses.h"
#include "tensors/FP16Utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace llaminar2
{

    class ActivationRotation
    {
    public:
        /**
         * @brief Construct a block-diagonal rotation for the given total dimension.
         *
         * @param total_dim   Total vector dimension (e.g., hidden_dim=2560 or ffn_dim=9216)
         * @param block_dim   Block size for the rotation (e.g., 128). Must divide total_dim.
         * @param seed        Seed for rotation matrix generation (default: 31)
         */
        explicit ActivationRotation(int total_dim, int block_dim, uint64_t seed = 31)
            : total_dim_(total_dim),
              block_dim_(block_dim),
              n_blocks_(total_dim / block_dim),
              rotation_(generate_rotation_matrix(block_dim, seed))
        {
            assert(total_dim % block_dim == 0 &&
                   "total_dim must be divisible by block_dim");
        }

        /// Total vector dimension
        int total_dim() const { return total_dim_; }

        /// Block dimension for rotation
        int block_dim() const { return block_dim_; }

        /// Number of blocks
        int n_blocks() const { return n_blocks_; }

        /// Access the underlying rotation matrix (block_dim × block_dim)
        const TurboQuantRotation &rotation() const { return rotation_; }

        /**
         * @brief Apply block-diagonal rotation to a single FP32 row in-place.
         *
         * Rotates each block_dim-sized chunk of the row independently using
         * the same rotation matrix: row[i*d:(i+1)*d] = R @ row[i*d:(i+1)*d]
         *
         * @param row   FP32 data (length >= total_dim)
         * @param dim   Row dimension (must equal total_dim)
         */
        void rotate_inplace(float *row, int dim) const
        {
            assert(dim == total_dim_);
            const int d = block_dim_;

            // Stack-allocate scratch for one block (up to 256 elements)
            alignas(64) float scratch[256];
            assert(d <= 256 && "block_dim must be <= 256 for stack allocation");

            for (int b = 0; b < n_blocks_; ++b)
            {
                float *chunk = row + b * d;
                apply_rotation(rotation_, chunk, scratch);
                // Copy back
                copy_block(scratch, chunk, d);
            }
        }

        /**
         * @brief Apply block-diagonal rotation to multiple FP32 rows in-place.
         *
         * Processes each row independently. Thread-safe when called on
         * different rows. For parallel usage, call rotate_inplace() per-row
         * from within an OpenMP workshare region.
         *
         * @param data  FP32 matrix (rows × dim, row-major)
         * @param rows  Number of rows
         * @param dim   Row dimension (must equal total_dim)
         */
        void rotate_rows_inplace(float *data, int rows, int dim) const
        {
            for (int r = 0; r < rows; ++r)
            {
                rotate_inplace(data + r * dim, dim);
            }
        }

        /**
         * @brief Apply inverse block-diagonal rotation to a single FP32 row in-place.
         *
         * Uses R^T (transpose) to undo the rotation.
         *
         * @param row   FP32 data (length >= total_dim)
         * @param dim   Row dimension (must equal total_dim)
         */
        void inverse_rotate_inplace(float *row, int dim) const
        {
            assert(dim == total_dim_);
            const int d = block_dim_;
            alignas(64) float scratch[256];
            assert(d <= 256);

            for (int b = 0; b < n_blocks_; ++b)
            {
                float *chunk = row + b * d;
                apply_rotation_transpose(rotation_, chunk, scratch);
                copy_block(scratch, chunk, d);
            }
        }

        /**
         * @brief Pre-rotate weight matrix rows for GEMM compatibility.
         *
         * For y = X @ W^T to remain correct after rotating X:
         *   X' = X @ R  →  W' = W @ R  →  X' @ W'^T = X @ W^T
         *
         * This applies R to each row of W (along the K dimension).
         * Called once during weight preparation.
         *
         * @param weight_fp32    FP32 dequantized weight data (N × K, row-major)
         * @param n_rows         Number of output rows (N)
         * @param k_dim          Input dimension (K), must equal total_dim
         */
        void rotate_weight_rows(float *weight_fp32, int n_rows, int k_dim) const
        {
            assert(k_dim == total_dim_);
            for (int r = 0; r < n_rows; ++r)
            {
                rotate_inplace(weight_fp32 + r * k_dim, k_dim);
            }
        }

        /**
         * @brief Create a rotated copy of a Q8_0 weight tensor.
         *
         * Dequantizes each row to FP32, applies block-diagonal rotation,
         * re-quantizes to Q8_0, and returns a new Q8_0Tensor.
         *
         * This is a one-time cost at model load time.
         *
         * @param src  Source Q8_0 weight tensor [N, K]. K must equal total_dim.
         * @return New Q8_0Tensor with rotated rows, or nullptr on error.
         */
        std::shared_ptr<Q8_0Tensor> create_rotated_weight(const Q8_0Tensor *src) const
        {
            if (!src || src->shape().size() != 2)
                return nullptr;

            const size_t N = src->shape()[0];
            const size_t K = src->shape()[1];
            if (static_cast<int>(K) != total_dim_)
                return nullptr;

            const size_t blocks_per_row = (K + 31) / 32;
            const size_t total_blocks = N * blocks_per_row;
            const size_t total_bytes = total_blocks * sizeof(Q8_0Block);

            // Allocate raw block storage
            std::vector<uint8_t> raw_data(total_bytes);
            Q8_0Block *dst_blocks = reinterpret_cast<Q8_0Block *>(raw_data.data());

            // Thread-local FP32 buffer for dequant + rotation
            #pragma omp parallel
            {
                std::vector<float> row_fp32(K);

                #pragma omp for schedule(static)
                for (size_t r = 0; r < N; ++r)
                {
                    // 1. Dequantize row to FP32
                    src->to_fp32_row(r, row_fp32.data());

                    // 2. Apply block-diagonal rotation
                    rotate_inplace(row_fp32.data(), static_cast<int>(K));

                    // 3. Re-quantize to Q8_0 blocks
                    Q8_0Block *row_dst = dst_blocks + r * blocks_per_row;
                    quantize_fp32_row_to_q8_0(row_fp32.data(), row_dst, K);
                }
            }

            return std::make_shared<Q8_0Tensor>(
                src->shape(), raw_data);
        }

        /**
         * @brief Create a rotated copy of any TensorBase weight (generic path).
         *
         * Uses data() to dequantize to FP32, then rotates and re-quantizes to Q8_0.
         * Works for any weight format (Q4_0, IQ4_NL, Q8_0, etc.) but always
         * outputs Q8_0 since post-rotation values don't fit original formats.
         *
         * @param src  Source weight tensor [N, K]. K must equal total_dim.
         * @return New Q8_0Tensor with rotated rows, or nullptr on error.
         */
        std::shared_ptr<Q8_0Tensor> create_rotated_weight_generic(const TensorBase *src) const
        {
            if (!src || src->shape().size() != 2)
                return nullptr;

            const size_t N = src->shape()[0];
            const size_t K = src->shape()[1];
            if (static_cast<int>(K) != total_dim_)
                return nullptr;

            // Dequantize entire tensor to FP32
            const float *fp32_data = src->data();
            if (!fp32_data)
                return nullptr;

            const size_t blocks_per_row = (K + 31) / 32;
            const size_t total_blocks = N * blocks_per_row;
            const size_t total_bytes = total_blocks * sizeof(Q8_0Block);

            std::vector<uint8_t> raw_data(total_bytes);
            Q8_0Block *dst_blocks = reinterpret_cast<Q8_0Block *>(raw_data.data());

            #pragma omp parallel
            {
                std::vector<float> row_fp32(K);

                #pragma omp for schedule(static)
                for (size_t r = 0; r < N; ++r)
                {
                    // Copy and rotate the row
                    std::memcpy(row_fp32.data(), fp32_data + r * K, K * sizeof(float));
                    rotate_inplace(row_fp32.data(), static_cast<int>(K));

                    // Quantize to Q8_0
                    Q8_0Block *row_dst = dst_blocks + r * blocks_per_row;
                    quantize_fp32_row_to_q8_0(row_fp32.data(), row_dst, K);
                }
            }

            return std::make_shared<Q8_0Tensor>(
                std::vector<size_t>{N, K}, raw_data);
        }

    private:
        static void copy_block(const float *src, float *dst, int n)
        {
#if defined(__AVX512F__)
            int i = 0;
            for (; i + 16 <= n; i += 16)
                _mm512_storeu_ps(dst + i, _mm512_loadu_ps(src + i));
            for (; i < n; ++i)
                dst[i] = src[i];
#else
            for (int i = 0; i < n; ++i)
                dst[i] = src[i];
#endif
        }

        /// Quantize a row of FP32 values to Q8_0 blocks
        static void quantize_fp32_row_to_q8_0(const float *src, Q8_0Block *dst, size_t count)
        {
            const size_t n_blocks = (count + 31) / 32;
            for (size_t b = 0; b < n_blocks; ++b)
            {
                const size_t offset = b * 32;
                const size_t block_len = std::min<size_t>(32, count - offset);
                const float *block_src = src + offset;
                Q8_0Block &block = dst[b];

                // Find max absolute value
                float max_abs = 0.0f;
                for (size_t i = 0; i < block_len; ++i)
                    max_abs = std::max(max_abs, std::abs(block_src[i]));

                if (max_abs < 1e-30f)
                {
                    block.d = 0;
                    std::memset(block.qs, 0, 32);
                    continue;
                }

                const float scale = max_abs / 127.0f;
                block.d = fp32_to_fp16(scale);
                const float inv_scale = 127.0f / max_abs;

                for (size_t i = 0; i < block_len; ++i)
                {
                    float v = block_src[i] * inv_scale;
                    v = std::max(-127.0f, std::min(127.0f, v));
                    block.qs[i] = static_cast<int8_t>(std::round(v));
                }
                // Zero-fill any padding (partial last block)
                for (size_t i = block_len; i < 32; ++i)
                    block.qs[i] = 0;
            }
        }

        int total_dim_;
        int block_dim_;
        int n_blocks_;
        TurboQuantRotation rotation_;
    };

} // namespace llaminar2
