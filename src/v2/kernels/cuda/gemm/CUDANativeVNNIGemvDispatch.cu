/**
 * @file CUDANativeVNNIGemvDispatch.cu
 * @brief Public CUDA NativeVNNI GEMV dispatch across physical format shards.
 *
 * This translation unit intentionally owns no expensive device templates.  A
 * runtime codebook maps directly to exactly one two-format shard, preserving
 * the former public ABI while allowing Ninja to compile all eight template
 * groups concurrently.  Trainer controls fan out to every shard because they
 * are semantic execution state, not per-format preferences.
 */

#include "CUDANativeVNNIGemvShard.h"
#include "CUDAGroupedVerifierLaunch.h"
#include "tensors/NativeVnniFormatInfo.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace
{
/** Map every supported runtime codebook to exactly one physical source TU. */
constexpr int shardIndexForCodebook(uint8_t codebook_id)
{
    switch (codebook_id)
    {
    case 0:
    case 4:
        return 0;
    case 5:
    case 6:
        return 1;
    case 7:
    case 8:
        return 2;
    case 9:
    case 10:
        return 3;
    case 11:
    case 12:
        return 4;
    case 13:
    case 14:
        return 5;
    case 15:
    case 16:
        return 6;
    case 17:
    case 19:
    case llaminar2::kNativeVnniExpandedInt8MinCodebook:
        return 7;
    default:
        return -1;
    }
}

/*
 * Keep hot dispatch as a direct call selected by a compact switch.  An array
 * of function pointers would be shorter, but would add an avoidable indirect
 * branch to eager launches and trainer timing. Graph replay does not execute
 * this host code, yet setup and uncaptured diagnostics should remain cheap.
 */
#define LLAMINAR_ROUTE_CUDA_NVNNI_SHARD(SYMBOL, FALLBACK, ...)          \
    do                                                                  \
    {                                                                   \
        switch (shardIndexForCodebook(codebook_id))                     \
        {                                                               \
        case 0:                                                         \
            return LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(SYMBOL, 0)(         \
                __VA_ARGS__);                                           \
        case 1:                                                         \
            return LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(SYMBOL, 1)(         \
                __VA_ARGS__);                                           \
        case 2:                                                         \
            return LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(SYMBOL, 2)(         \
                __VA_ARGS__);                                           \
        case 3:                                                         \
            return LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(SYMBOL, 3)(         \
                __VA_ARGS__);                                           \
        case 4:                                                         \
            return LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(SYMBOL, 4)(         \
                __VA_ARGS__);                                           \
        case 5:                                                         \
            return LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(SYMBOL, 5)(         \
                __VA_ARGS__);                                           \
        case 6:                                                         \
            return LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(SYMBOL, 6)(         \
                __VA_ARGS__);                                           \
        case 7:                                                         \
            return LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(SYMBOL, 7)(         \
                __VA_ARGS__);                                           \
        default:                                                        \
            return FALLBACK;                                            \
        }                                                               \
    } while (false)

#define LLAMINAR_FOR_EACH_CUDA_NVNNI_SHARD(SYMBOL, ...)                \
    LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(SYMBOL, 0)(__VA_ARGS__);           \
    LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(SYMBOL, 1)(__VA_ARGS__);           \
    LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(SYMBOL, 2)(__VA_ARGS__);           \
    LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(SYMBOL, 3)(__VA_ARGS__);           \
    LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(SYMBOL, 4)(__VA_ARGS__);           \
    LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(SYMBOL, 5)(__VA_ARGS__);           \
    LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(SYMBOL, 6)(__VA_ARGS__);           \
    LLAMINAR_CUDA_NVNNI_SHARD_SYMBOL(SYMBOL, 7)(__VA_ARGS__)
} // namespace

extern "C"
{
/**
 * @brief Validate capture-safe stream ownership once, then route the exact candidate.
 * @details The full buffer/epilogue contract is declared in CUDACanonicalKpartFold.h.
 */
bool cudaNativeVNNIGemvTuned_fusedKpar_fp32(
    const int8_t *activations, const uint8_t *payload, const uint16_t *scales,
    const uint16_t *secondary, const uint32_t *extended_minima, float *output,
    const float *activation_scales, int n, float alpha, float beta,
    const float *existing, const float *bias, uint8_t codebook_id, int device,
    void *raw_stream, const llaminar2::CUDACanonicalKpartFoldPlan &plan)
{
    const auto stream = static_cast<cudaStream_t>(raw_stream);
    if (!stream || stream == cudaStreamLegacy || stream == cudaStreamPerThread ||
        !activations || !payload || !scales || !output || !activation_scales ||
        n <= 0 || device < 0 || (beta != 0.0f && !existing) ||
        reinterpret_cast<uintptr_t>(activations) % alignof(int4) != 0)
        return false;
    int current_device = -1;
    CUcontext current_context = nullptr, stream_context = nullptr;
    // cudaStreamGetDevice invalidates active capture. These driver metadata
    // queries are capture-safe and prove stronger, exact context ownership.
    // Nothing is downloaded, synchronized, or remembered in a shadow cache.
    if (cudaGetDevice(&current_device) != cudaSuccess || current_device != device ||
        cuCtxGetCurrent(&current_context) != CUDA_SUCCESS || !current_context ||
        cuStreamGetCtx(stream, &stream_context) != CUDA_SUCCESS ||
        stream_context != current_context)
        return false;
    LLAMINAR_ROUTE_CUDA_NVNNI_SHARD(
        cudaNativeVNNIGemvTuned_fusedKpar_fp32, false,
        activations, payload, scales, secondary, extended_minima, output,
        activation_scales, n, alpha, beta, existing, bias, codebook_id,
        device, stream, plan);
}

/** @brief Route the header-declared resource query to its one template owner. */
bool cudaNativeVNNIGemvTuned_fusedKpar_resources(
    uint8_t codebook_id, const llaminar2::CUDACanonicalKpartFoldPlan &plan,
    llaminar2::CUDACanonicalKpartFoldResources &resources)
{
    LLAMINAR_ROUTE_CUDA_NVNNI_SHARD(
        cudaNativeVNNIGemvTuned_fusedKpar_resources, false,
        codebook_id, plan, resources);
}

bool cudaNativeVNNIGemvTuned_supportsCodebook(uint8_t codebook_id)
{
    return shardIndexForCodebook(codebook_id) >= 0;
}

bool cudaNativeVNNIGemvTuned_queryGeneratedDispatch(
    uint8_t codebook_id,
    int graph_captured,
    int m,
    int n,
    int k,
    int *shape_id,
    int *tile_n,
    int *cpt,
    int *exact_kb)
{
    LLAMINAR_ROUTE_CUDA_NVNNI_SHARD(
        cudaNativeVNNIGemvTuned_queryGeneratedDispatch,
        false,
        codebook_id,
        graph_captured,
        m,
        n,
        k,
        shape_id,
        tile_n,
        cpt,
        exact_kb);
}

/**
 * @brief Resolve a complete source arithmetic policy on its owning shard.
 *
 * Physical codebook 19/23 shards call this only when CPU residency changed the
 * execution representation.  Routing policy lookup back to the source shard
 * avoids multiplying generated selector instantiations across decoder shards.
 */
bool cudaNativeVNNIGemvTuned_resolveArithmeticPolicy(
    uint8_t codebook_id,
    int graph_captured,
    int m,
    int n,
    int k,
    int *shape_id,
    int *tile_n,
    int *cpt,
    int *target_waves,
    int *mkg,
    int *max_kb,
    int *force_two_phase,
    int *exact_kb)
{
    LLAMINAR_ROUTE_CUDA_NVNNI_SHARD(
        cudaNativeVNNIGemvTuned_queryCompleteGeneratedDispatch,
        false,
        codebook_id,
        graph_captured,
        m,
        n,
        k,
        shape_id,
        tile_n,
        cpt,
        target_waves,
        mkg,
        max_kb,
        force_two_phase,
        exact_kb);
}

/** Resolve a source codebook's grouped-row policy on its owning shard. */
bool cudaNativeVNNIGemvTuned_resolveGroupedArithmeticPolicy(
    uint8_t codebook_id,
    int graph_captured,
    int m,
    int n,
    int k,
    int *kernel,
    int *grouped_rows)
{
    LLAMINAR_ROUTE_CUDA_NVNNI_SHARD(
        cudaNativeVNNIGemvTuned_queryGeneratedGroupedDispatch,
        false,
        codebook_id,
        graph_captured,
        m,
        n,
        k,
        kernel,
        grouped_rows);
}

bool cudaNativeVNNIGemvTuned_queryCanonicalM1Schedule(
    uint8_t codebook_id,
    int n,
    int k,
    int sm_count,
    int *uses_ordered_reducer,
    int *k_partitions)
{
    LLAMINAR_ROUTE_CUDA_NVNNI_SHARD(
        cudaNativeVNNIGemvTuned_queryCanonicalM1Schedule,
        false,
        codebook_id,
        n,
        k,
        sm_count,
        uses_ordered_reducer,
        k_partitions);
}

double cudaNativeVNNIGemvTuned_measureGeneratedDispatchNs(
    uint8_t codebook_id,
    int graph_captured,
    int m,
    const int *ns,
    const int *ks,
    int query_count,
    int iterations,
    uint64_t *out_checksum)
{
    LLAMINAR_ROUTE_CUDA_NVNNI_SHARD(
        cudaNativeVNNIGemvTuned_measureGeneratedDispatchNs,
        -1.0,
        codebook_id,
        graph_captured,
        m,
        ns,
        ks,
        query_count,
        iterations,
        out_checksum);
}

bool cudaNativeVNNIGemvTuned_fp32(
    const int8_t *d_A_int8,
    const uint8_t *d_payload,
    const uint16_t *d_scales,
    const uint16_t *d_mins,
    const uint32_t *d_emins,
    float *d_C_fp32,
    const float *d_scales_A_block,
    int N,
    int K,
    float alpha,
    float beta,
    const float *d_C_existing,
    const float *d_bias,
    uint8_t codebook_id,
    int cuda_device_id,
    void *stream,
    CUDAGemvContext *gemv_ctx,
    CUDARowMajorWeights **rm_slot)
{
    LLAMINAR_ROUTE_CUDA_NVNNI_SHARD(
        cudaNativeVNNIGemvTuned_fp32,
        false,
        d_A_int8,
        d_payload,
        d_scales,
        d_mins,
        d_emins,
        d_C_fp32,
        d_scales_A_block,
        N,
        K,
        alpha,
        beta,
        d_C_existing,
        d_bias,
        codebook_id,
        cuda_device_id,
        stream,
        gemv_ctx,
        rm_slot);
}

/**
 * @brief Route a physical CUDA decoder with an independent arithmetic policy.
 *
 * ExpertOverlay CPU residency may normalize compact source bytes into codebook
 * 19 or 23.  The physical codebook selects the decoder shard; the policy
 * codebook selects the source model's generated launch and reduction tree.
 */
bool cudaNativeVNNIGemvTuned_fp32_withPolicy(
    const int8_t *d_A_int8,
    const uint8_t *d_payload,
    const uint16_t *d_scales,
    const uint16_t *d_mins,
    const uint32_t *d_emins,
    float *d_C_fp32,
    const float *d_scales_A_block,
    int N,
    int K,
    float alpha,
    float beta,
    const float *d_C_existing,
    const float *d_bias,
    uint8_t codebook_id,
    uint8_t arithmetic_policy_codebook_id,
    int cuda_device_id,
    void *stream,
    CUDAGemvContext *gemv_ctx,
    CUDARowMajorWeights **rm_slot)
{
    LLAMINAR_ROUTE_CUDA_NVNNI_SHARD(
        cudaNativeVNNIGemvTuned_fp32_withPolicy,
        false,
        d_A_int8,
        d_payload,
        d_scales,
        d_mins,
        d_emins,
        d_C_fp32,
        d_scales_A_block,
        N,
        K,
        alpha,
        beta,
        d_C_existing,
        d_bias,
        codebook_id,
        arithmetic_policy_codebook_id,
        cuda_device_id,
        stream,
        gemv_ctx,
        rm_slot);
}

bool cudaNativeVNNIGemvTuned_small_m_fp32(
    const int8_t *d_A_int8,
    const uint8_t *d_payload,
    const uint16_t *d_scales,
    const uint16_t *d_mins,
    const uint32_t *d_emins,
    float *d_C_fp32,
    const float *d_scales_A_block,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    const float *d_C_existing,
    const float *d_bias,
    uint8_t codebook_id,
    int cuda_device_id,
    void *stream,
    CUDAGemvContext *gemv_ctx,
    CUDARowMajorWeights **rm_slot)
{
    LLAMINAR_ROUTE_CUDA_NVNNI_SHARD(
        cudaNativeVNNIGemvTuned_small_m_fp32,
        false,
        d_A_int8,
        d_payload,
        d_scales,
        d_mins,
        d_emins,
        d_C_fp32,
        d_scales_A_block,
        M,
        N,
        K,
        alpha,
        beta,
        d_C_existing,
        d_bias,
        codebook_id,
        cuda_device_id,
        stream,
        gemv_ctx,
        rm_slot);
}

/**
 * @brief Route grouped CUDA decode with source-format arithmetic identity.
 * @see CUDAGroupedVerifierLaunch.h for the canonical operand/count contract.
 *
 * The dispatcher forwards immutable launch metadata; it never reads the live
 * device count or changes the selected physical codebook shard.
 */
bool cudaNativeVNNIGemvTuned_small_m_fp32_withPolicy(
    const int8_t *d_A_int8,
    const uint8_t *d_payload,
    const uint16_t *d_scales,
    const uint16_t *d_mins,
    const uint32_t *d_emins,
    float *d_C_fp32,
    const float *d_scales_A_block,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    const float *d_C_existing,
    const float *d_bias,
    uint8_t codebook_id,
    uint8_t arithmetic_policy_codebook_id,
    int cuda_device_id,
    void *stream,
    CUDAGemvContext *gemv_ctx,
    CUDARowMajorWeights **rm_slot,
    const llaminar2::DeviceRowRange *row_range)
{
    LLAMINAR_ROUTE_CUDA_NVNNI_SHARD(
        cudaNativeVNNIGemvTuned_small_m_fp32_withPolicy,
        false,
        d_A_int8,
        d_payload,
        d_scales,
        d_mins,
        d_emins,
        d_C_fp32,
        d_scales_A_block,
        M,
        N,
        K,
        alpha,
        beta,
        d_C_existing,
        d_bias,
        codebook_id,
        arithmetic_policy_codebook_id,
        cuda_device_id,
        stream,
        gemv_ctx,
        rm_slot,
        row_range);
}

bool cudaNativeVNNIInitIQGridTables_tuned()
{
    /* Device symbols are shared across the linked CUDA module. */
    return cudaNativeVNNIInitIQGridTables_tuned_shard_0();
}

void cudaNativeVNNIGemvSweep_setConfig(
    int kernel_family,
    int tile_n,
    int cpt,
    int target_waves,
    int mkg,
    int max_kb,
    int exact_kb,
    int force_two_phase)
{
    LLAMINAR_FOR_EACH_CUDA_NVNNI_SHARD(
        cudaNativeVNNIGemvSweep_setConfig,
        kernel_family,
        tile_n,
        cpt,
        target_waves,
        mkg,
        max_kb,
        exact_kb,
        force_two_phase);
}

void cudaNativeVNNIGemvSweep_clearConfig()
{
    LLAMINAR_FOR_EACH_CUDA_NVNNI_SHARD(
        cudaNativeVNNIGemvSweep_clearConfig);
}

void cudaNativeVNNIGemvSweep_setGroupedRows(int rows_per_tile)
{
    LLAMINAR_FOR_EACH_CUDA_NVNNI_SHARD(
        cudaNativeVNNIGemvSweep_setGroupedRows,
        rows_per_tile);
}

void cudaNativeVNNIGroupedVerifier_setTensorCoreOverride(int enabled)
{
    LLAMINAR_FOR_EACH_CUDA_NVNNI_SHARD(
        cudaNativeVNNIGroupedVerifier_setTensorCoreOverride,
        enabled);
}

bool cudaNativeVNNIGemvSweep_isActive()
{
    /* Every public mutation fans out before returning. */
    return cudaNativeVNNIGemvSweep_isActive_shard_0();
}

void cudaNativeVNNIGemvTuned_clearStaticState()
{
    LLAMINAR_FOR_EACH_CUDA_NVNNI_SHARD(
        cudaNativeVNNIGemvTuned_clearStaticState);
}

CUDAGemvContext *cudaGemvContext_create(int cuda_device_id)
{
    return cudaGemvContext_create_shard_0(cuda_device_id);
}

void cudaGemvContext_destroy(CUDAGemvContext *ctx)
{
    cudaGemvContext_destroy_shard_0(ctx);
}

void cudaGemvContext_bindWorkspace(
    CUDAGemvContext *ctx,
    float *kpar_partials,
    size_t kpar_partials_bytes)
{
    cudaGemvContext_bindWorkspace_shard_0(
        ctx,
        kpar_partials,
        kpar_partials_bytes);
}

CUDARowMajorWeights *cudaRowMajorWeights_create(
    const uint8_t *d_payload_col,
    const uint16_t *d_scales_col,
    const uint16_t *d_mins_col,
    const uint32_t *d_emins_col,
    int N,
    int K,
    int payload_bytes,
    int cuda_device_id,
    void *stream)
{
    return cudaRowMajorWeights_create_shard_0(
        d_payload_col,
        d_scales_col,
        d_mins_col,
        d_emins_col,
        N,
        K,
        payload_bytes,
        cuda_device_id,
        stream);
}

CUDARowMajorWeights *cudaRowMajorWeights_createForCodebook(
    const uint8_t *d_payload_col,
    const uint16_t *d_scales_col,
    const uint16_t *d_mins_col,
    const uint32_t *d_emins_col,
    int N,
    int K,
    uint8_t codebook_id,
    int cuda_device_id,
    void *stream)
{
    LLAMINAR_ROUTE_CUDA_NVNNI_SHARD(
        cudaRowMajorWeights_createForCodebook,
        nullptr,
        d_payload_col,
        d_scales_col,
        d_mins_col,
        d_emins_col,
        N,
        K,
        codebook_id,
        cuda_device_id,
        stream);
}

bool cudaNativeVNNIGemvTuned_policyRequiresRowMajor(
    uint8_t codebook_id,
    int N,
    int K)
{
    LLAMINAR_ROUTE_CUDA_NVNNI_SHARD(
        cudaNativeVNNIGemvTuned_policyRequiresRowMajor,
        false,
        codebook_id,
        N,
        K);
}

void cudaRowMajorWeights_destroy(CUDARowMajorWeights *rm)
{
    cudaRowMajorWeights_destroy_shard_0(rm);
}

void cudaNativeVNNIGemvTuned_setDecodeEquivalentM1Config(int enabled)
{
    LLAMINAR_FOR_EACH_CUDA_NVNNI_SHARD(
        cudaNativeVNNIGemvTuned_setDecodeEquivalentM1Config,
        enabled);
}

int cudaNativeVNNIGemvTuned_getDecodeEquivalentM1Config()
{
    return cudaNativeVNNIGemvTuned_getDecodeEquivalentM1Config_shard_0();
}

void cudaNativeVNNIGemvTuned_setSerialPartitionN(int n)
{
    LLAMINAR_FOR_EACH_CUDA_NVNNI_SHARD(
        cudaNativeVNNIGemvTuned_setSerialPartitionN,
        n);
}

int cudaNativeVNNIGemvTuned_getSerialPartitionN()
{
    return cudaNativeVNNIGemvTuned_getSerialPartitionN_shard_0();
}
} // extern "C"

#undef LLAMINAR_FOR_EACH_CUDA_NVNNI_SHARD
#undef LLAMINAR_ROUTE_CUDA_NVNNI_SHARD
