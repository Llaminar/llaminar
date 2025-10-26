/**
 * @file QuantizedGemmL1Opt.h
 * @brief L1 cache-optimized quantized GEMM kernel
 *
 * Reduces L1 cache misses through explicit register blocking and micro-kernel optimization.
 * Based on BLIS/GotoBLAS design principles.
 *
 * Target: Reduce L1 miss rate from 41.7% to <10% while maintaining throughput.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../tensors/TensorKernels.h"
#include "../../utils/DebugEnv.h"
#include <algorithm>

// SIMD intrinsics
#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2
{

    /**
     * @brief L1-optimized quantized GEMM kernel with explicit register blocking
     *
     * Key optimizations for L1 cache efficiency:
     * 1. Micro-kernel (MR × NR): Fits in registers (8×6 for AVX512 = 48 floats = 192 bytes)
     * 2. Explicit register blocking: Minimizes memory traffic
     * 3. Panel-panel multiplication: Better spatial/temporal locality
     * 4. Aligned loads: Reduces cache line splits
     * 5. Software prefetching: Hides memory latency
     *
     * Design:
     * - L1 panel: MC × KC (processes MC rows of A, KC columns at a time)
     * - L2 tile: KC × NC (decodes NC columns of B, reuses across MC rows)
     * - Micro-kernel: MR × NR (innermost loop, stays in registers)
     *
     * Cache hierarchy utilization:
     * - Registers: MR × NR accumulator (8×6 = 48 floats)
     * - L1 cache: MC × KC panel of A + NR decoded B columns (~20-24 KB)
     * - L2 cache: KC × NC tile of decoded B (~100-400 KB)
     */
    class QuantizedGemmL1Opt : public ITensorGemm
    {
    public:
        explicit QuantizedGemmL1Opt(const IBlockDecoder *decoder)
            : decoder_(decoder) {}

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only
        }

        bool multiply(
            const float *A, float *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

    private:
        const IBlockDecoder *decoder_;

        // Micro-kernel dimensions (optimized for AVX512)
        static constexpr int MR = 8; // Rows processed per micro-kernel (8 AVX512 accumulators)
        static constexpr int NR = 6; // Cols processed per micro-kernel (6 columns fit in registers)

        // Panel dimensions (optimized for L1/L2 cache)
        static constexpr int MC = 256; // M panel size (L1: 256×896×4 = 896 KB for A panel)
        static constexpr int NC = 64;  // N panel size (OPTIMIZED: 896×64×4 = 224 KB for B tile)
        static constexpr int KC = 256; // K panel size (balance between reuse and working set)
        static constexpr int PREFETCH_DISTANCE = 64; // Prefetch 64 floats ahead

        // Micro-kernel: C[MR×NR] += A[MR×KC] * B[KC×NR]
        void micro_kernel(
            const float *A_panel, const float *B_panel,
            float *C, int ldc, int k_panel,
            float alpha, float beta, int mr, int nr);

        // Pack panels for better cache utilization
        void pack_A_panel(const float *A, float *A_packed, int m_panel, int k_panel, int lda);
        void pack_B_panel(const float *B_decoded, float *B_packed, int k_panel, int n_panel);
    };

} // namespace llaminar2
