/**
 * @file QuantLinearKernel.h
 * @brief Prototype fused decode + GEMM kernel for quantized weight matrices.
 *
 * This is a baseline correctness-oriented implementation that decodes quant blocks on the fly
 * while accumulating A * W into an FP32 output buffer (optionally downcasting later).
 *
 * Layout assumptions:
 *  - Weight tensor W is stored as QuantizedTensor with logical shape [K, N] (rows=K, cols=N).
 *  - Activation matrix A provided as row-major float/FP16 pointer with shape [M, K].
 *  - Output matrix C is row-major with shape [M, N].
 *  - Quant blocks cover contiguous column spans within a single row (as produced by TensorFactory::create_quantized).
 *
 * Performance notes:
 *  - Decodes one block (elements_per_block values) at a time and immediately applies it across all M rows for the given k row.
 *  - Avoids materializing full dequantized weight row (memory saving) but does not yet vectorize inner loops.
 *  - Future optimizations: tile over K and N, SIMD FMA, batch decode for multiple k, prefetch next block, fused accumulation into FP16 if allowed.
 */

#pragma once

#include <cstddef>
#include <vector>
#include <memory>
#include "../tensors/TensorFactory.h"

namespace llaminar
{

    struct QuantLinearParams
    {
        const float *A = nullptr; // (M x K) activation matrix (row-major)
        int M = 0;                // rows of A / C
        int K = 0;                // shared dimension (rows of W, cols of A)
        int N = 0;                // columns of W / C
        float *C = nullptr;       // (M x N) output accumulator (row-major, FP32)
        bool zero_C = true;       // if true, zero initialize C before accumulation
        int tile_n = 0;           // optional N tile size (columns per macro-block); 0 => auto (multiple of block_elems or 256)
        int tile_k = 0;           // optional K tile size (decode multiple rows together); 0 => auto heuristic (1 or 4 for large M)
    };

    /**
     * @brief Fused quant decode + matmul: C = A * W  (W quantized as QuantizedTensor)
     * @param Wq Quantized weight tensor with shape [K, N].
     * @param p  Parameter struct (M,K,N pointers & flags).
     * @return true on success, false on validation failure.
     */
    bool quant_linear_fused(const QuantizedTensor &Wq, const QuantLinearParams &p);

} // namespace llaminar
