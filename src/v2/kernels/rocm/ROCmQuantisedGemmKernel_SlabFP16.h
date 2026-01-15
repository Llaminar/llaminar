/**
 * @file ROCmQuantisedGemmKernel_SlabFP16.h
 * @brief Header for slab-based FP16 GEMM functions
 *
 * This header declares the extern "C" functions implemented in
 * ROCmQuantisedGemmKernel_SlabFP16.hip for use from C++ code.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "kernels/SlabGemmConfig.h"

#include <cstdint>
#include <cstddef>

// ============================================================================
// Extern "C" declarations for HIP-compiled functions
// ============================================================================

extern "C"
{
    /**
     * @brief Execute slab-based INT8→FP16 GEMM with fixed workspace
     *
     * Converts INT8 data to FP16 in slabs and accumulates FP32 result.
     * Uses fixed workspace buffers - no allocations during execution.
     *
     * @param d_A_int8     INT8 activations [M × K] on device
     * @param d_B_int8     INT8 weights [K × N] on device (transposed)
     * @param d_C_fp32     FP32 output [M × N] on device
     * @param d_scaleA     Per-row scales [M] on device
     * @param d_scaleB     Per-column scales [N] on device
     * @param M, N, K      Full GEMM dimensions
     * @param slab_a_fp16  Workspace for A slab [slab_m × slab_k] (__half*)
     * @param slab_b_fp16  Workspace for B slab [slab_k × slab_n] (__half*)
     * @param slab_c_fp16  Workspace for C slab [slab_m × slab_n] (__half*)
     * @param config       Slab configuration (dimensions)
     * @param device_id    ROCm device ID
     * @param stream       HIP stream (nullptr for default)
     * @return true on success
     */
    bool rocmQuantGemm_executeSlabFP16(
        const int8_t *d_A_int8,
        const int8_t *d_B_int8,
        float *d_C_fp32,
        const float *d_scaleA,
        const float *d_scaleB,
        int M, int N, int K,
        void *slab_a_fp16,
        void *slab_b_fp16,
        void *slab_c_fp16,
        const llaminar2::SlabGemmConfig *config,
        int device_id,
        void *stream);

    /**
     * @brief Check if slab FP16 GEMM should be used instead of full FP16
     *
     * Decision criteria:
     *   1. Full FP16 would exceed available workspace budget
     *   2. User forced slab mode via LLAMINAR_ROCM_GEMM_FORCE_SLAB=1
     *   3. Matrix is large enough to benefit from chunking
     *
     * @param M, N, K          Full GEMM dimensions
     * @param workspace_budget Available workspace bytes
     * @return true if slab GEMM should be used
     */
    bool rocmQuantGemm_shouldUseSlabFP16(
        int M, int N, int K,
        size_t workspace_budget);

    /**
     * @brief Get optimal slab configuration for given dimensions and budget
     *
     * @param M, N, K          Full GEMM dimensions
     * @param workspace_budget Available workspace bytes (0 = auto 64MB)
     * @param out_config       Output configuration
     */
    void rocmQuantGemm_getSlabConfig(
        int M, int N, int K,
        size_t workspace_budget,
        llaminar2::SlabGemmConfig *out_config);
}
