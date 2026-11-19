/**
 * @file CUDANativeQuantisedPrefillDispatch.cu
 * @brief Public CUDA NativeQuantised prefill ABI across eight physical codebook shards.
 *
 * This host-only TU owns every public cudaNativeQuantisedPrefill_* entrypoint,
 * the per-device prefill context, workspace planning and the single
 * definitions of the profiler/sweep controls and thread-local launch records
 * declared by CUDANativeQuantisedPrefillHostPolicy.h. Launches and resource
 * queries route by codebook to exactly one shard
 * (CUDANativeQuantisedPrefillShardN.cu); codebook-independent controls are
 * process state here, not per-shard copies, so a setter is observed by every
 * shard at once. Planning reads the generated exact-overlay table only here
 * and in the shard that launches. This TU contains no device code.
 */

#include <cuda_runtime.h>

#include "CUDANativeQuantisedPrefillHostPolicy.h"
#include "CUDANativeQuantisedPrefillShard.h"
#include "CUDANativeQuantisedPrefillDiagnostics.h"
#include "CUDANativeQuantisedWorkspaceHardware.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <cstdint>
#include <new>
#include <string>

using namespace llaminar2::cuda::prefill;
using namespace llaminar2::cuda::prefill_host;

namespace llaminar2::cuda::prefill_host
{
    PrefillHostControls &hostState()
    {
        // Seeded once from the typed diagnostics environment. Production never
        // changes the defaults; only isolated sweep/profiler harnesses do.
        static PrefillHostControls controls{
            .force_tile_id = llaminar2::debugEnv().gemm.cuda_force_prefill_tile,
            .force_canonical_kpart = false,
            .exact_overlay_enabled = true,
            .bk256_force_mode = llaminar2::debugEnv().gemm.cuda_bk256_mode,
        };
        return controls;
    }

    PrefillStagingSchedule &stagingSchedule()
    {
        // A diagnostic scope may change operand delivery on its own host
        // thread. Normal inference retains the certified default.
        thread_local PrefillStagingSchedule schedule = PrefillStagingSchedule::RegisterDecode;
        return schedule;
    }

    LastLaunchSelection &lastLaunchSelection()
    {
        // Thread-local because tile sweep benchmarks launch from several worker
        // threads; diagnostics read back the selection on the same thread.
        thread_local LastLaunchSelection selection;
        return selection;
    }
} // namespace llaminar2::cuda::prefill_host

namespace
{
/*
 * Hot dispatch is a direct call selected by a compact switch rather than an
 * indirect function-pointer table, matching the decode shard dispatcher.
 */
#define LLAMINAR_ROUTE_CUDA_NATIVE_QUANTISED_PREFILL_SHARD(SYMBOL, FALLBACK, ...)        \
    do                                                                              \
    {                                                                               \
        switch (prefillShardIndexForCodebook(codebook_id))                          \
        {                                                                           \
        case 0: return LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_INDEXED_SYMBOL(SYMBOL, 0)(__VA_ARGS__); \
        case 1: return LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_INDEXED_SYMBOL(SYMBOL, 1)(__VA_ARGS__); \
        case 2: return LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_INDEXED_SYMBOL(SYMBOL, 2)(__VA_ARGS__); \
        case 3: return LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_INDEXED_SYMBOL(SYMBOL, 3)(__VA_ARGS__); \
        case 4: return LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_INDEXED_SYMBOL(SYMBOL, 4)(__VA_ARGS__); \
        case 5: return LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_INDEXED_SYMBOL(SYMBOL, 5)(__VA_ARGS__); \
        case 6: return LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_INDEXED_SYMBOL(SYMBOL, 6)(__VA_ARGS__); \
        case 7: return LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_INDEXED_SYMBOL(SYMBOL, 7)(__VA_ARGS__); \
        default: return FALLBACK;                                                   \
        }                                                                           \
    } while (false)
} // namespace

extern "C"
{
    bool cudaNativeQuantisedPrefill_setStagingSchedule(PrefillStagingSchedule schedule)
    {
        if (!isKnownPrefillStagingSchedule(schedule))
            return false;
        stagingSchedule() = schedule;
        return true;
    }

    PrefillStagingSchedule cudaNativeQuantisedPrefill_getStagingSchedule()
    {
        return stagingSchedule();
    }

    PrefillStagingSchedule cudaNativeQuantisedPrefill_getLastLaunchStagingSchedule()
    {
        return lastLaunchSelection().staging;
    }

    // BK256 force mode: 0=auto(heuristic), 1=force ON, -1=force OFF
    void cudaNativeQuantisedPrefill_setBK256Mode(int mode)
    {
        hostState().bk256_force_mode = mode;
    }

    int cudaNativeQuantisedPrefill_getBK256Mode()
    {
        return hostState().bk256_force_mode;
    }

    /** Select the exact public-M1 ordered K-partition candidate for a sweep. */
    void cudaNativeQuantisedPrefill_setCanonicalKPartitionMode(bool enabled)
    {
        hostState().force_canonical_kpart = enabled;
    }

    /** Return whether the exact public-M1 K-partition candidate is forced. */
    bool cudaNativeQuantisedPrefill_getCanonicalKPartitionMode()
    {
        return hostState().force_canonical_kpart;
    }

    /** Enable or bypass installed exact cells for an isolated policy sweep. */
    void cudaNativeQuantisedPrefill_setExactOverlayEnabled(bool enabled)
    {
        hostState().exact_overlay_enabled = enabled;
    }

    /** Return whether ordinary production may consult installed exact cells. */
    bool cudaNativeQuantisedPrefill_getExactOverlayEnabled()
    {
        return hostState().exact_overlay_enabled;
    }

    void cudaNativeQuantisedPrefill_getLastLaunchSelection(
        int *tile_id,
        int *k_partitions,
        int *used_bk256,
        int *used_canonical_kpart)
    {
        if (tile_id)
            *tile_id = lastLaunchSelection().tile_id;
        if (k_partitions)
            *k_partitions = lastLaunchSelection().k_partitions;
        if (used_bk256)
            *used_bk256 = lastLaunchSelection().used_bk256;
        if (used_canonical_kpart)
            *used_canonical_kpart =
                lastLaunchSelection().used_canonical_kpart;
    }

    /**
     * @brief Report whether the last launch came from an installed exact cell.
     *
     * The value is thread-local for the same reason as the remaining launch
     * diagnostics: sweep workers configure and inspect their own launch
     * transaction. Production PerfStats uses this provenance to distinguish a
     * measured overlay from the total generic policy.
     */
    int cudaNativeQuantisedPrefill_getLastLaunchUsedExactOverlay()
    {
        return lastLaunchSelection().used_exact_overlay;
    }

    /**
     * @brief Query compiler resources for the exact last-launched prefill path.
     *
     * Routed to the one shard owning @p codebook_id, which holds the launched
     * specialization. The thread-local launch record is shared, so the shard
     * sees the identity published by its own previous launch on this thread.
     * Setup-only: no allocation, transfer, launch or synchronization.
     * @return False for a stale/unsupported identity or an unowned codebook.
     */
    bool cudaNativeQuantisedPrefill_queryLastLaunchResources(
        uint8_t codebook_id,
        int n,
        int k,
        int cuda_device_id,
        CUDADensePrefillKernelResources *primary,
        CUDADensePrefillKernelResources *auxiliary)
    {
        LLAMINAR_ROUTE_CUDA_NATIVE_QUANTISED_PREFILL_SHARD(
            cudaNativeQuantisedPrefill_queryLastLaunchResources, false,
            codebook_id, n, k, cuda_device_id, primary, auxiliary);
    }

    /**
     * @brief Query compiler resources of one explicit prefill candidate.
     *
     * Routed to the one shard owning @p codebook_id; BK256 candidates exist
     * only for codebook 0 and therefore only in its shard.
     * @return False for an invalid candidate or an unowned codebook.
     */
    bool cudaNativeQuantisedPrefill_queryCandidateResources(
        uint8_t codebook_id,
        int tile_id,
        int canonical_kpart,
        int ordered_bk256,
        int cuda_device_id,
        CUDADensePrefillKernelResources *primary,
        CUDADensePrefillKernelResources *auxiliary,
        PrefillStagingSchedule staging)
    {
        LLAMINAR_ROUTE_CUDA_NATIVE_QUANTISED_PREFILL_SHARD(
            cudaNativeQuantisedPrefill_queryCandidateResources, false,
            codebook_id, tile_id, canonical_kpart, ordered_bk256,
            cuda_device_id, primary, auxiliary, staging);
    }

    // Force output-tile geometry for sweep benchmarks. K partitioning remains
    // bound to the public M=1 arithmetic contract.
    void cudaNativeQuantisedPrefill_setForceTile(int tile_id)
    {
        hostState().force_tile_id = tile_id;
    }

    void cudaNativeQuantisedPrefill_getForceTile(int *tile_id)
    {
        if (tile_id)
            *tile_id = hostState().force_tile_id;
    }

    // Return the number of tiles for a given tile config + shape
    int cudaNativeQuantisedPrefill_getTileCount(int tile_id, int M, int N)
    {
        int bm = 64, bn = 64;
        switch (static_cast<TileId>(tile_id))
        {
        case TileId::T64x64_w2x2:
            bm = 64;
            bn = 64;
            break;
        case TileId::T64x128_w2x2:
        case TileId::T64x128_w4x2:
        case TileId::T64x128_w2x4:
            bm = 64;
            bn = 128;
            break;
        case TileId::T128x128_w4x2:
        case TileId::T128x128_w4x4:
            bm = 128;
            bn = 128;
            break;
        case TileId::Count:
            return 0;
        }
        return ((M + bm - 1) / bm) * ((N + bn - 1) / bn);
    }
    // -----------------------------------------------------------------
    // Prefill context lifetime
    // -----------------------------------------------------------------
    CUDAPrefillContext *cudaPrefillContext_create(int device_id)
    {
        auto *ctx = new (std::nothrow) CUDAPrefillContext_();
        if (!ctx)
            return nullptr;
        ctx->device_id = device_id;
        return ctx;
    }

    void cudaPrefillContext_destroy(CUDAPrefillContext *ctx)
    {
        if (!ctx)
            return;
        delete ctx;
    }

    void cudaPrefillContext_bindWorkspace(
        CUDAPrefillContext *ctx,
        float *canonical_kpart_partials,
        size_t canonical_kpart_partials_bytes)
    {
        if (!ctx)
            return;
        ctx->workspace_canonical_kpart_partials = canonical_kpart_partials;
        ctx->workspace_canonical_kpart_partials_size =
            canonical_kpart_partials_bytes;
    }

    /**
     * @brief Plan prefill workspace using distinct decode and arithmetic formats.
     *
     * The execution codebook selects the physical decoder and any learned
     * prefill overlay. The arithmetic-policy codebook selects the serial-M1
     * reduction tree that every row must reproduce. They differ after a cold
     * CPU tier promotes a compact asymmetric matrix into codebook 23.
     */
    bool cudaNativeQuantisedPrefill_getWorkspacePlanWithPolicy(
        uint8_t codebook_id,
        uint8_t arithmetic_policy_codebook_id,
        int M,
        int N,
        int K,
        int cuda_device_id,
        size_t *canonical_kpart_partials_bytes,
        int *planned_k_partitions)
    {
        if (canonical_kpart_partials_bytes)
            *canonical_kpart_partials_bytes = 0;
        if (planned_k_partitions)
            *planned_k_partitions = 1;
        if (M <= 0 || N <= 0 || K <= 0 || (K % 32) != 0)
            return false;
        if (!isSupportedNativeQuantisedPrefillCodebook(codebook_id))
            return false;
        if (!cudaNativeQuantisedGemvTuned_supportsCodebook(
                arithmetic_policy_codebook_id))
            return false;

        llaminar2::cuda::generated::CUDADensePrefillOverlayConfig
            exact_overlay{};
        const DensePrefillOverlayStatus overlay_status =
            selectDensePrefillOverlay(
                codebook_id, M, N, K, exact_overlay);
        if (overlay_status == DensePrefillOverlayStatus::Invalid)
            return false;
        const bool needs_canonical_kpart =
            hostState().force_canonical_kpart ||
            (overlay_status == DensePrefillOverlayStatus::Selected &&
             exact_overlay.canonical_kpart);
        if (!needs_canonical_kpart)
            return true;
        return queryCanonicalPrefillWorkspaceForRows(
            arithmetic_policy_codebook_id,
            M,
            N,
            K,
            llaminar2::cudaHardwareMetadata().at(cuda_device_id).sm_count,
            canonical_kpart_partials_bytes,
            planned_k_partitions);
    }

    /** @brief Plan a self-describing matrix whose policy equals its decoder. */
    bool cudaNativeQuantisedPrefill_getWorkspacePlan(
        uint8_t codebook_id,
        int M,
        int N,
        int K,
        int cuda_device_id,
        size_t *canonical_kpart_partials_bytes,
        int *planned_k_partitions)
    {
        return cudaNativeQuantisedPrefill_getWorkspacePlanWithPolicy(
            codebook_id,
            codebook_id == llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook ? uint8_t{4} : codebook_id,
            M,
            N,
            K,
            cuda_device_id,
            canonical_kpart_partials_bytes,
            planned_k_partitions);
    }

    /** @brief Plan the maximum workspace for an explicit arithmetic policy. */
    static bool workspaceEnvelopeWithHardwareBinding(
        uint8_t codebook_id,
        uint8_t arithmetic_policy_codebook_id,
        int max_M,
        int N,
        int K,
        int cuda_device_id,
        const llaminar2::cuda::NativeQuantisedWorkspaceHardware *hardware,
        size_t *canonical_kpart_partials_bytes,
        int *planned_k_partitions,
        int *planned_rows)
    {
        if (canonical_kpart_partials_bytes)
            *canonical_kpart_partials_bytes = 0;
        if (planned_k_partitions)
            *planned_k_partitions = 1;
        if (planned_rows)
            *planned_rows = 0;
        if (max_M <= 0 || N <= 0 || K <= 0 || (K % 32) != 0)
            return false;
        if (!isSupportedNativeQuantisedPrefillCodebook(codebook_id))
            return false;
        if (!cudaNativeQuantisedGemvTuned_supportsCodebook(
                arithmetic_policy_codebook_id))
            return false;

        int canonical_rows = 0;
        if (hostState().force_canonical_kpart)
        {
            canonical_rows = max_M;
        }
        else
        {
            const bool exact_overlay_controls_launch =
                hostState().exact_overlay_enabled &&
                hostState().force_tile_id < 0 &&
                hostState().bk256_force_mode == 0;
            if (exact_overlay_controls_launch)
            {
                /*
                 * Generated cells are sparse in geometry and non-monotonic in
                 * launch family. Scan the complete installed policy rather
                 * than assuming the largest M represents every smaller replay.
                 * This occurs during graph/workspace planning and is cached by
                 * the C++ adapter; no scan occurs in captured execution.
                 */
                for (const auto &entry :
                     llaminar2::cuda::generated::kCUDADensePrefillOverlayEntries)
                {
                    if (entry.codebook != (codebook_id == llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook ? uint8_t{4} : codebook_id) ||
                        entry.n != N || entry.k != K ||
                        entry.m <= 1 || entry.m > max_M)
                    {
                        continue;
                    }
                    if (!isValidDensePrefillOverlayConfig(
                            codebook_id, entry.config))
                    {
                        return false;
                    }
                    if (entry.config.canonical_kpart)
                        canonical_rows = std::max(canonical_rows, entry.m);
                }
            }
        }

        if (canonical_rows == 0)
            return true;
        if (!queryCanonicalPrefillWorkspaceForRows(
                arithmetic_policy_codebook_id,
                canonical_rows,
                N,
                K,
                hardware ? hardware->smCount() : llaminar2::cudaHardwareMetadata().at(cuda_device_id).sm_count,
                canonical_kpart_partials_bytes,
                planned_k_partitions))
        {
            return false;
        }
        if (planned_rows)
            *planned_rows = canonical_rows;
        return true;
    }

    /** @brief Runtime ABI binds the admitted participant only when installed policy consumes its SM count. */
    bool cudaNativeQuantisedPrefill_getWorkspaceEnvelopeWithPolicy(
        uint8_t codebook_id, uint8_t arithmetic_policy_codebook_id, int max_M, int N, int K,
        int cuda_device_id, size_t *bytes, int *partitions, int *rows)
    {
        return workspaceEnvelopeWithHardwareBinding(codebook_id, arithmetic_policy_codebook_id,
            max_M, N, K, cuda_device_id, nullptr, bytes, partitions, rows);
    }

    /** @brief Plan a self-describing matrix whose policy equals its decoder. */
    bool cudaNativeQuantisedPrefill_getWorkspaceEnvelope(
        uint8_t codebook_id,
        int max_M,
        int N,
        int K,
        int cuda_device_id,
        size_t *canonical_kpart_partials_bytes,
        int *planned_k_partitions,
        int *planned_rows)
    {
        return cudaNativeQuantisedPrefill_getWorkspaceEnvelopeWithPolicy(
            codebook_id,
            codebook_id == llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook ? uint8_t{4} : codebook_id,
            max_M,
            N,
            K,
            cuda_device_id,
            canonical_kpart_partials_bytes,
            planned_k_partitions,
            planned_rows);
    }
} // extern "C"

bool llaminar2::cuda::nativeQuantisedPrefillWorkspaceEnvelopeForHardware(
    std::uint8_t codebook, std::uint8_t arithmetic_codebook, int max_rows, int n, int k,
    const NativeQuantisedWorkspaceHardware &hardware, std::size_t *bytes, int *partitions, int *rows)
{
    return workspaceEnvelopeWithHardwareBinding(codebook, arithmetic_codebook, max_rows, n, k,
        hardware.deviceOrdinal(), &hardware, bytes, partitions, rows);
}


// Profitability gate removed: NativeQuantised is now the only CUDA GEMM path.
// TC/CUTLASS expanded fallback has been sunset. All codebooks always use
// the NativeQuantised prefill kernel regardless of shape.

/**
 * @brief Execute NativeQuantised prefill with explicit decoder and arithmetic IDs.
 *
 * `codebook_id` describes the bytes read by the templated kernel, while
 * `arithmetic_policy_codebook_id` owns the serial-M1 split tree. Keeping both
 * values explicit prevents representation normalization during expert
 * migration from changing FP32 publication order.
 */
extern "C" bool cudaNativeQuantisedPrefill_fp32_withPolicy(
    const int8_t *d_A_int8,
    const uint8_t *d_payload,
    const uint16_t *d_scales,
    const uint16_t *d_mins,
    const uint32_t *d_emins,
    float *d_C_fp32,
    const float *d_scales_A_block,
    const int32_t *d_sums_A_block,
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
    CUDAPrefillContext *prefill_ctx)
{
    if (!d_A_int8 || !d_payload || !d_scales || !d_C_fp32 || !d_scales_A_block)
        return false;
    if (M <= 0 || N <= 0 || K <= 0 || (K % 32) != 0)
        return false;
    if (!prefill_ctx)
        return false;
    if (!cudaNativeQuantisedGemvTuned_supportsCodebook(
            arithmetic_policy_codebook_id))
        return false;
    if (!isAmperePlus(cuda_device_id))
        return false;
    if (cudaSetDevice(cuda_device_id) != cudaSuccess)
        return false;

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    int serial_m1_uses_ordered_reducer = 0;
    int serial_m1_k_partitions = 0;
    if (!cudaNativeQuantisedGemvTuned_queryCanonicalM1Schedule(
            arithmetic_policy_codebook_id,
            N,
            K,
            querySmCount(prefill_ctx),
            &serial_m1_uses_ordered_reducer,
            &serial_m1_k_partitions))
    {
        return false;
    }

    bool ok = false;
    switch (llaminar2::cuda::prefill_host::prefillShardIndexForCodebook(codebook_id))
    {
#define LLAMINAR_PREFILL_LAUNCH_SHARD(INDEX)                                          \
    case INDEX:                                                                      \
        ok = LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_INDEXED_SYMBOL(             \
            cudaNativeQuantisedPrefill_launch, INDEX)(                               \
            d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C_fp32,                \
            d_scales_A_block, d_sums_A_block, M, N, K, alpha, beta, d_C_existing,    \
            d_bias, codebook_id, serial_m1_k_partitions,                             \
            serial_m1_uses_ordered_reducer, cuda_stream, prefill_ctx);               \
        break
        LLAMINAR_PREFILL_LAUNCH_SHARD(0);
        LLAMINAR_PREFILL_LAUNCH_SHARD(1);
        LLAMINAR_PREFILL_LAUNCH_SHARD(2);
        LLAMINAR_PREFILL_LAUNCH_SHARD(3);
        LLAMINAR_PREFILL_LAUNCH_SHARD(4);
        LLAMINAR_PREFILL_LAUNCH_SHARD(5);
        LLAMINAR_PREFILL_LAUNCH_SHARD(6);
        LLAMINAR_PREFILL_LAUNCH_SHARD(7);
#undef LLAMINAR_PREFILL_LAUNCH_SHARD
    default:
        return false;
    }

    if (ok && llaminar2::PerfStatsCollector::isDomainEnabled("kernel"))
    {
        int tile_id = -1;
        int k_partitions = 1;
        int used_bk256 = 0;
        int used_canonical_kpart = 0;
        int used_exact_overlay = 0;
        cudaNativeQuantisedPrefill_getLastLaunchSelection(
            &tile_id,
            &k_partitions,
            &used_bk256,
            &used_canonical_kpart);
        used_exact_overlay =
            cudaNativeQuantisedPrefill_getLastLaunchUsedExactOverlay();
        llaminar2::PerfStatsCollector::addCounter(
            "kernel",
            "cuda_native_quantised_prefill_calls",
            1.0,
            "gemm",
            "cuda:" + std::to_string(cuda_device_id),
            llaminar2::PerfStatsCollector::Tags{
                {"codebook", std::to_string(static_cast<int>(codebook_id))},
                {"arithmetic_policy_codebook",
                 std::to_string(
                     static_cast<int>(arithmetic_policy_codebook_id))},
                {"m", std::to_string(M)},
                {"n", std::to_string(N)},
                {"k", std::to_string(K)},
                {"tile_id", std::to_string(tile_id)},
                {"k_partitions", std::to_string(k_partitions)},
                {"bk256", used_bk256 ? "1" : "0"},
                {"canonical_kpart", used_canonical_kpart ? "1" : "0"},
                {"staging_schedule", std::to_string(static_cast<int>(
                    cudaNativeQuantisedPrefill_getLastLaunchStagingSchedule()))},
                {"exact_overlay", used_exact_overlay ? "1" : "0"},
                {"sums_a", d_sums_A_block ? "1" : "0"}});
    }
    return ok;
}

/** @brief Execute a self-describing matrix with its own arithmetic policy. */
extern "C" bool cudaNativeQuantisedPrefill_fp32(
    const int8_t *d_A_int8,
    const uint8_t *d_payload,
    const uint16_t *d_scales,
    const uint16_t *d_mins,
    const uint32_t *d_emins,
    float *d_C_fp32,
    const float *d_scales_A_block,
    const int32_t *d_sums_A_block,
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
    CUDAPrefillContext *prefill_ctx)
{
    return cudaNativeQuantisedPrefill_fp32_withPolicy(
        d_A_int8,
        d_payload,
        d_scales,
        d_mins,
        d_emins,
        d_C_fp32,
        d_scales_A_block,
        d_sums_A_block,
        M,
        N,
        K,
        alpha,
        beta,
        d_C_existing,
        d_bias,
        codebook_id,
        codebook_id == llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook ? uint8_t{4} : codebook_id,
        cuda_device_id,
        stream,
        prefill_ctx);
}
