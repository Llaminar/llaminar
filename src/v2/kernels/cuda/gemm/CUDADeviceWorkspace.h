/**
 * @file CUDADeviceWorkspace.h
 * @brief Per-device GPU workspace for CUDA kernel dispatch.
 *
 * Provides stable, lifecycle-managed arenas for canonical decode and prefill
 * K-partition reductions. Owned by KernelFactory, one per device, and released
 * when its prepared-kernel cache is cleared.
 *
 * Thread safety: Each workspace is single-device.  The row-major cache
 * has its own mutex for concurrent GEMV dispatch from graph replay.
 */

#pragma once

#include <cstddef>
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

    /** Per-device prefill workspace for canonical public-M1 K partials. */
    typedef struct CUDAPrefillContext_ CUDAPrefillContext;

    /** Per-device cuBLAS workspace (handle + FP16 staging). */

    // -----------------------------------------------------------------
    // GEMV context lifetime
    // -----------------------------------------------------------------
    CUDAGemvContext *cudaGemvContext_create(int cuda_device_id);
    void cudaGemvContext_destroy(CUDAGemvContext *ctx);
    void cudaGemvContext_bindWorkspace(
        CUDAGemvContext *ctx,
        float *kpar_partials,
        size_t kpar_partials_bytes);

    // -----------------------------------------------------------------
    // Row-major weight transpose lifetime (per quantized GEMM kernel weight)
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
    CUDARowMajorWeights *cudaRowMajorWeights_createForCodebook(
        const uint8_t *d_payload_col,
        const uint16_t *d_scales_col,
        const uint16_t *d_mins_col,
        const uint32_t *d_emins_col,
        int N,
        int K,
        uint8_t codebook_id,
        int cuda_device_id,
        void *stream);
    void cudaRowMajorWeights_destroy(CUDARowMajorWeights *rm);
    bool cudaNativeVNNIGemvTuned_policyRequiresRowMajor(
        uint8_t codebook_id,
        int N,
        int K);

    // -----------------------------------------------------------------
    // Prefill context lifetime
    // -----------------------------------------------------------------
    CUDAPrefillContext *cudaPrefillContext_create(int cuda_device_id);
    void cudaPrefillContext_destroy(CUDAPrefillContext *ctx);
    void cudaPrefillContext_bindWorkspace(
        CUDAPrefillContext *ctx,
        float *canonical_kpart_partials,
        size_t canonical_kpart_partials_bytes);
    bool cudaNativeVNNIPrefill_getWorkspacePlan(
        uint8_t codebook_id,
        int M,
        int N,
        int K,
        int cuda_device_id,
        size_t *canonical_kpart_partials_bytes,
        int *planned_k_partitions);

    // -----------------------------------------------------------------
    // cuBLAS context lifetime
    // -----------------------------------------------------------------

#ifdef __cplusplus
}
#endif
