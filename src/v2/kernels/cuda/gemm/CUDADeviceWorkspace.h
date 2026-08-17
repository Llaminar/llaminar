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

    /**
     * @brief Plan one prefill launch with distinct decoder and policy IDs.
     *
     * The second identifier is the original source codebook retained across
     * CPU-tier representation normalization. It controls only the serial-M1
     * arithmetic tree and therefore belongs in workspace identity.
     */
    bool cudaNativeVNNIPrefill_getWorkspacePlanWithPolicy(
        uint8_t codebook_id,
        uint8_t arithmetic_policy_codebook_id,
        int M,
        int N,
        int K,
        int cuda_device_id,
        size_t *canonical_kpart_partials_bytes,
        int *planned_k_partitions);

    /**
     * @brief Compute the persistent scratch envelope for a prefill graph family.
     *
     * A captured prefill family can replay any installed exact-overlay bucket
     * through `max_M`. Exact dispatch is not monotonic in row count: a smaller
     * bucket may use the canonical public-M1 K-partition schedule while the
     * largest bucket uses a direct full-K tile. This query examines the complete
     * active launch-policy range and returns the largest canonical requirement
     * that any replay in the family can publish.
     *
     * The function performs policy planning only. It launches no kernels,
     * allocates no device memory, and leaves 128-row arena padding to the owning
     * workspace planner.
     *
     * @param codebook_id Canonical NativeVNNI codebook identifier.
     * @param max_M Inclusive maximum row count represented by the graph family.
     * @param N Projection output width.
     * @param K Projection reduction width; must be divisible by 32.
     * @param cuda_device_id CUDA device whose public-M1 schedule is authoritative.
     * @param canonical_kpart_partials_bytes Receives the unpadded maximum bytes.
     * @param planned_k_partitions Receives the public-M1 partition count, or one
     *        when no member of the family uses canonical K partitioning.
     * @param planned_rows Receives the row count responsible for the envelope,
     *        or zero when no member requires canonical reduction scratch.
     * @return `true` for a complete valid policy envelope; `false` for invalid
     *         geometry, unsupported codebook, malformed overlay, or unavailable
     *         canonical schedule.
     */
    bool cudaNativeVNNIPrefill_getWorkspaceEnvelope(
        uint8_t codebook_id,
        int max_M,
        int N,
        int K,
        int cuda_device_id,
        size_t *canonical_kpart_partials_bytes,
        int *planned_k_partitions,
        int *planned_rows);

    /** @brief Compute a graph-family envelope for an explicit source policy. */
    bool cudaNativeVNNIPrefill_getWorkspaceEnvelopeWithPolicy(
        uint8_t codebook_id,
        uint8_t arithmetic_policy_codebook_id,
        int max_M,
        int N,
        int K,
        int cuda_device_id,
        size_t *canonical_kpart_partials_bytes,
        int *planned_k_partitions,
        int *planned_rows);

    // -----------------------------------------------------------------
    // cuBLAS context lifetime
    // -----------------------------------------------------------------

#ifdef __cplusplus
}
#endif
