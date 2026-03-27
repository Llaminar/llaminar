/**
 * @file CUDADeviceWorkspace.h
 * @brief Per-device GPU workspace for CUDA kernel dispatch.
 *
 * Replaces process-global static state (g_rm_cache, getKparPartials,
 * g_streamk_fixup_buf, g_ws) with properly scoped, lifecycle-managed
 * objects.  Owned by KernelFactory, one per device, freed on clearCache().
 *
 * Thread safety: Each workspace is single-device.  The row-major cache
 * has its own mutex for concurrent GEMV dispatch from graph replay.
 */

#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C"
{
#endif

    // =====================================================================
    // Opaque handles — .cu files define the concrete structs.
    // =====================================================================

    /** Per-device GEMV workspace (KPAR partials, SM count cache). */
    typedef struct CUDAGemvContext_ CUDAGemvContext;

    /** Per-weight row-major transpose (ROWPAR acceleration). */
    typedef struct CUDARowMajorWeights_ CUDARowMajorWeights;

    /** Per-device prefill workspace (stream-K fixup buffer). */
    typedef struct CUDAPrefillContext_ CUDAPrefillContext;

    /** Per-device cuBLAS workspace (handle + FP16 staging). */
    typedef struct CUDACuBLASContext_ CUDACuBLASContext;

    // -----------------------------------------------------------------
    // GEMV context lifetime
    // -----------------------------------------------------------------
    CUDAGemvContext *cudaGemvContext_create(int cuda_device_id);
    void cudaGemvContext_destroy(CUDAGemvContext *ctx);

    // -----------------------------------------------------------------
    // Row-major weight transpose lifetime (per CUDAPackedWeights)
    // -----------------------------------------------------------------
    CUDARowMajorWeights *cudaRowMajorWeights_create(
        const uint8_t *d_payload_col,
        const uint16_t *d_scales_col,
        const uint16_t *d_mins_col,
        const uint32_t *d_emins_col,
        int N, int K,
        int payload_bytes,
        int cuda_device_id,
        void *stream);
    void cudaRowMajorWeights_destroy(CUDARowMajorWeights *rm);

    // -----------------------------------------------------------------
    // Prefill context lifetime
    // -----------------------------------------------------------------
    CUDAPrefillContext *cudaPrefillContext_create(int cuda_device_id);
    void cudaPrefillContext_destroy(CUDAPrefillContext *ctx);

    // -----------------------------------------------------------------
    // cuBLAS context lifetime
    // -----------------------------------------------------------------
    CUDACuBLASContext *cudaCuBLASContext_create(int cuda_device_id);
    void cudaCuBLASContext_destroy(CUDACuBLASContext *ctx);

#ifdef __cplusplus
}
#endif
