/**
 * @file IQ4_NL_BlockDecoder.h
 * @brief IQ4_NL block decoder for CUDA kernels
 *
 * Provides device-side dequantization for IQ4_NL quantized tensors.
 * Implements the IBlockDecoder interface using compile-time polymorphism.
 *
 * @author David Sanftenberg
 * @date October 31, 2025
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h> // BF16 support (CUDA 11.0+, Compute Capability 8.0+)
#include <cstdint>

namespace llaminar2
{
    namespace cuda
    {

        /**
         * @brief IQ4_NL quantized block structure (matches CPU implementation)
         *
         * Each block encodes 32 float values in 18 bytes:
         * - 2 bytes: FP16 scale factor (d)
         * - 16 bytes: Packed 4-bit indices (2 per byte)
         *
         * Effective compression: 4.5 bits/value (~7.1× vs FP32)
         */
        struct IQ4_NLBlock
        {
            static constexpr int BLOCK_SIZE = 32;

            uint16_t d;     ///< FP16 scale factor
            uint8_t qs[16]; ///< Packed 4-bit indices (low/high nibbles)
        };

        /**
         * @brief IQ4_NL lookup table in constant memory
         *
         * Maps 4-bit indices (0-15) to quantized values (-127 to 113).
         * Defined in IQ4_NL_BlockDecoder.cu.
         */
        extern __constant__ int8_t kvalues_iq4nl[16];

        /**
         * @brief IQ4_NL block decoder for CUDA kernels
         *
         * Decodes IQ4_NL quantized blocks on-the-fly during GEMM.
         * Uses constant memory for kvalues lookup table.
         *
         * Block format:
         * - 16-bit FP16 scale factor
         * - 16 bytes containing 32 4-bit quantized values
         * - Total: 32 elements per block
         */
        template <typename BlockType = IQ4_NLBlock>
        class IQ4_NL_Decoder
        {
        public:
            /**
             * @brief Construct decoder with pointer to quantized weight tensor
             *
             * @param blocks Pointer to quantized blocks [n_rows × k_blocks]
             * @param n_rows Number of rows in weight matrix
             * @param k_blocks Number of blocks along K dimension
             */
            __device__ __host__ IQ4_NL_Decoder(
                const BlockType *blocks,
                int n_rows,
                int k_blocks)
                : blocks_(blocks), n_rows_(n_rows), k_blocks_(k_blocks)
            {
            }

            /**
             * @brief Decode one IQ4_NL block to FP32
             *
             * Dequantizes 32 4-bit values using FP16 scale factor and kvalues LUT.
             * This method is inlined directly into the GEMM kernel for zero overhead.
             *
             * @param block Pointer to quantized block
             * @param output Output buffer (must have 32 floats)
             */
            __device__ inline void decode_block(const BlockType *block, float *output) const
            {
                const float d = __half2float(*reinterpret_cast<const __half *>(&block->d));

#pragma unroll
                for (int j = 0; j < 16; ++j)
                {
                    const uint8_t qbyte = block->qs[j];
                    const uint8_t idx_low = qbyte & 0x0F;
                    output[j] = d * static_cast<float>(kvalues_iq4nl[idx_low]);
                    const uint8_t idx_high = qbyte >> 4;
                    output[j + 16] = d * static_cast<float>(kvalues_iq4nl[idx_high]);
                }
            }

            /**
             * @brief Decode one IQ4_NL block to FP16 (for Tensor Cores)
             *
             * Phase 2 optimization: Dequantizes directly to FP16 to avoid FP32→FP16 conversion.
             * This is critical for Tensor Core performance (wmma requires FP16 inputs).
             *
             * @param block Pointer to quantized block
             * @param output Output buffer (must have 32 __half elements)
             */
            __device__ inline void decode_block_fp16(const BlockType *block, __half *output) const
            {
                const __half d = *reinterpret_cast<const __half *>(&block->d);

#pragma unroll
                for (int j = 0; j < 16; ++j)
                {
                    const uint8_t qbyte = block->qs[j];
                    const uint8_t idx_low = qbyte & 0x0F;
                    output[j] = __hmul(d, __int2half_rn(kvalues_iq4nl[idx_low]));
                    const uint8_t idx_high = qbyte >> 4;
                    output[j + 16] = __hmul(d, __int2half_rn(kvalues_iq4nl[idx_high]));
                }
            }

            /**
             * @brief Decode one IQ4_NL block to BF16 (for Tensor Cores with BF16 support)
             *
             * Phase 3 optimization: Dequantizes directly to BF16 for Ampere+ GPUs (SM 8.0+).
             * BF16 has wider dynamic range than FP16 (same exponent as FP32) which can
             * improve numerical stability while maintaining Tensor Core performance.
             *
             * Requirements: CUDA Compute Capability ≥ 8.0 (Ampere: RTX 3090, A100, etc.)
             *
             * @param block Pointer to quantized block
             * @param output Output buffer (must have 32 __nv_bfloat16 elements)
             */
            __device__ inline void decode_block_bf16(const BlockType *block, __nv_bfloat16 *output) const
            {
                // Extract FP16 scale factor
                const __half d_fp16 = *reinterpret_cast<const __half *>(&block->d);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
                // Manual FP16→BF16 conversion (truncation of mantissa)
                // FP16: 1 sign, 5 exp, 10 mantissa → BF16: 1 sign, 8 exp, 7 mantissa
                // Extract exponent and mantissa, then convert to BF16 via FP32
                const float d_fp32 = __half2float(d_fp16);
                const __nv_bfloat16 d = __float2bfloat16(d_fp32);

#pragma unroll
                for (int j = 0; j < 16; ++j)
                {
                    const uint8_t qbyte = block->qs[j];
                    const uint8_t idx_low = qbyte & 0x0F;
                    // Convert int8 quantized value to BF16 and multiply by scale
                    output[j] = __hmul(d, __int2bfloat16_rn(kvalues_iq4nl[idx_low]));
                    const uint8_t idx_high = qbyte >> 4;
                    output[j + 16] = __hmul(d, __int2bfloat16_rn(kvalues_iq4nl[idx_high]));
                }
#else
                // Fallback for pre-Ampere (should not reach here)
                const float d_fp32 = __half2float(d_fp16);

#pragma unroll
                for (int j = 0; j < 16; ++j)
                {
                    const uint8_t qbyte = block->qs[j];
                    const uint8_t idx_low = qbyte & 0x0F;
                    output[j] = __float2bfloat16(d_fp32 * static_cast<float>(kvalues_iq4nl[idx_low]));
                    const uint8_t idx_high = qbyte >> 4;
                    output[j + 16] = __float2bfloat16(d_fp32 * static_cast<float>(kvalues_iq4nl[idx_high]));
                }
#endif
            }

            /**
             * @brief Get block at specific position
             *
             * @param row Row index in weight matrix
             * @param k_block Block index along K dimension
             * @return Pointer to quantized block
             */
            __device__ inline const BlockType *get_block_at(int row, int k_block) const
            {
                return &blocks_[row * k_blocks_ + k_block];
            }

            /**
             * @brief Get block size (number of elements per block)
             */
            __device__ __host__ inline int block_size() const { return 32; }

            /**
             * @brief Get tensor dimensions
             */
            __device__ __host__ inline int rows() const { return n_rows_; }
            __device__ __host__ inline int k_blocks() const { return k_blocks_; }

        private:
            const BlockType *blocks_;
            int n_rows_;
            int k_blocks_;
        };

    } // namespace cuda
} // namespace llaminar2
