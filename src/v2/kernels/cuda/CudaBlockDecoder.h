/**
 * @file CudaBlockDecoder.h
 * @brief CUDA block decoder interface for quantized tensors
 *
 * Provides device-side dequantization interface for block-quantized weight formats.
 * Uses compile-time polymorphism (templates) for zero-overhead abstraction.
 *
 * This file defines ONLY the interface contract. Actual decoder implementations
 * live in separate files (IQ4_NL_BlockDecoder.h, Q6_K_BlockDecoder.h, etc.).
 *
 * Usage:
 *   #include "IQ4_NL_BlockDecoder.h"
 *
 *   template<typename Decoder>
 *   __global__ void gemm_kernel(..., const Decoder decoder) {
 *       float decoded[BLOCK_SIZE];
 *       const auto* block = decoder.get_block_at(row, k_block);
 *       decoder.decode_block(block, decoded);
 *   }
 *
 * @author David Sanftenberg
 * @date October 31, 2025
 */

#pragma once

#include <cuda_runtime.h>

namespace llaminar2
{
    namespace cuda
    {

        /**
         * @brief Block decoder interface (compile-time polymorphism)
         *
         * All decoder implementations must provide:
         *
         * 1. Constructor:
         *    __device__ __host__ Decoder(const BlockType* blocks, int n_rows, int k_blocks)
         *
         * 2. Decode method (MUST be inlined for performance):
         *    __device__ inline void decode_block(const BlockType* block, float* output) const
         *
         * 3. Block accessor:
         *    __device__ inline const BlockType* get_block_at(int row, int k_block) const
         *
         * 4. Metadata:
         *    __device__ __host__ inline int block_size() const
         *    __device__ __host__ inline int rows() const
         *    __device__ __host__ inline int k_blocks() const
         *
         * Note: This is not a virtual interface (no vtable overhead on device).
         * Template-based dispatch provides zero-overhead polymorphism.
         *
         * Performance requirements:
         * - decode_block() is called in GEMM hot path (thousands of times per matmul)
         * - MUST be marked __device__ inline and implemented in header
         * - Should use #pragma unroll for decode loops
         * - Avoid branches in decode logic when possible
         *
         * Implementations:
         * - IQ4_NL_BlockDecoder.h - IQ4_NL format (32 elements/block, 4-bit + FP16 scale)
         * - Q6_K_BlockDecoder.h   - Q6_K format (future)
         * - Q8_0_BlockDecoder.h   - Q8_0 format (future)
         */

        // Interface documented via concept (C++20) or informal contract (C++17)
        // See individual decoder files for implementations

    } // namespace cuda
} // namespace llaminar2
