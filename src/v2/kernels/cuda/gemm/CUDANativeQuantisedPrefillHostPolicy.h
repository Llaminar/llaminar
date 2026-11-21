/**
 * @file CUDANativeQuantisedPrefillHostPolicy.h
 * @brief Host-only policy and process state shared by CUDA prefill dispatch and shards.
 *
 * CUDA NativeQuantised prefill is compiled as eight physical codebook shards
 * (CUDANativeQuantisedPrefillShard0..7.cu) behind one host dispatcher
 * (CUDANativeQuantisedPrefillDispatch.cu). Each shard is a complete,
 * deterministic NVVM module: nvcc's compiler-internal --split-compile is not
 * used, because its partitioning varies between identical compiles and with
 * it the PTX, register allocation and spill-guard verdict of the same kernels.
 *
 * Everything that is policy rather than device code lives here, once:
 * - the prefill context and the thread-local last-launch record;
 * - the profiler/sweep controls (forced tile, canonical K partitions, exact
 *   overlay enable, BK256 mode) and the thread-local staging schedule, which
 *   have exactly one definition in the dispatcher TU, so every shard reads and
 *   records through the same authority instead of private per-TU copies;
 * - tile identities, the supported-codebook inventory, overlay validation and
 *   selection, and the canonical public-M1 workspace size;
 * - the codebook to shard map, used by the dispatcher to route and by each
 *   shard to instantiate only its own codebooks.
 *
 * No function here launches a kernel, allocates, transfers or synchronizes.
 */

#pragma once

#include "CUDADenseProductionPrefillOverlayGenerated.inc"
#include "CUDANativeQuantisedPrefillSchedule.h"
#include "kernels/cuda/gemm/CUDADeviceWorkspace.h"
#include "backends/cuda/CUDAHardwareMetadata.h"
#include "tensors/NativeQuantisedFormatInfo.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

extern "C" bool cudaNativeQuantisedGemvTuned_queryCanonicalM1Schedule(
    uint8_t codebook_id,
    int n,
    int k,
    int sm_count,
    int *uses_ordered_reducer,
    int *k_partitions);

/** @brief Return whether the generated serial policy owns this codebook. */
extern "C" bool cudaNativeQuantisedGemvTuned_supportsCodebook(
    uint8_t codebook_id);

// =========================================================================
// Per-device prefill context. Owned by KernelFactory, one per device.
// =========================================================================
struct CUDAPrefillContext_
{
    int device_id = -1;
    float *workspace_canonical_kpart_partials = nullptr;
    size_t workspace_canonical_kpart_partials_size = 0;
};

namespace llaminar2::cuda::prefill_host
{
    using llaminar2::cuda::prefill::PrefillStagingSchedule;
    using llaminar2::cuda::prefill::isKnownPrefillStagingSchedule;

    /** @brief Physical identity of the most recent launch on this host thread. */
    struct LastLaunchSelection
    {
        int tile_id = -1;
        int k_partitions = 1;
        int used_bk256 = 0;
        int used_canonical_kpart = 0;
        int used_exact_overlay = 0;
        PrefillStagingSchedule staging = PrefillStagingSchedule::RegisterDecode;
    };

    /**
     * @brief Process-wide profiler/sweep controls for CUDA prefill launch policy.
     *
     * Production keeps the defaults for the whole process; only isolated sweep
     * and profiler harnesses change them. The forced tile and BK256 mode are
     * seeded once from DebugEnv when the dispatcher first creates the state.
     */
    struct PrefillHostControls
    {
        int force_tile_id = -1;              ///< Forced BK64 tile, or -1.
        bool force_canonical_kpart = false;  ///< Force the public-M1 K partitions.
        bool exact_overlay_enabled = true;   ///< Consult installed exact cells.
        int bk256_force_mode = 0;            ///< 0 auto, 1 force on, -1 force off.
    };

    /** @return The single process-wide control set (defined by the dispatcher). */
    PrefillHostControls &hostState();
    /** @return This host thread's operand-staging schedule (defined by the dispatcher). */
    PrefillStagingSchedule &stagingSchedule();
    /** @return This host thread's last-launch record (defined by the dispatcher). */
    LastLaunchSelection &lastLaunchSelection();

    /**
     * @brief Publish the physical identity of a launch on this host thread.
     * @param tile_id Launched tile, -2/-3 for BK256.
     * @param k_partitions Public-M1 K partitions used by the launch.
     * @param used_bk256 Whether the BK256 family ran.
     * @param used_canonical_kpart Whether the canonical K-partition tree ran.
     * @param used_exact_overlay Whether an installed exact cell chose the launch.
     * @param staging Operand-staging schedule of the launch.
     */
    inline void recordLastLaunchSelection(
        int tile_id,
        int k_partitions,
        bool used_bk256,
        bool used_canonical_kpart = false,
        bool used_exact_overlay = false,
        PrefillStagingSchedule staging = PrefillStagingSchedule::RegisterDecode)
    {
        auto &selection = lastLaunchSelection();
        selection.tile_id = tile_id;
        selection.k_partitions = k_partitions;
        selection.used_bk256 = used_bk256 ? 1 : 0;
        selection.used_canonical_kpart = used_canonical_kpart ? 1 : 0;
        selection.used_exact_overlay = used_exact_overlay ? 1 : 0;
        selection.staging = staging;
    }

    /**
     * @brief Borrow physical SM geometry from the one admitted CUDA inventory.
     * @param ctx Prepared or planning-only context with its exact participant ordinal.
     * @return Real physical count; query failure cannot manufacture an 82-SM device.
     * @throws std::invalid_argument On an absent projection owner.
     */
    inline int querySmCount(const CUDAPrefillContext_ *ctx)
    {
        if (!ctx)
            throw std::invalid_argument("CUDA prefill requires its prepared device owner");
        return llaminar2::cudaHardwareMetadata().at(ctx->device_id).sm_count;
    }

    /**
     * @brief Read the selected device's admitted architecture without another query.
     * @param device_id CUDA ordinal whose prefill capability is being admitted.
     * @return True only for an admitted Ampere-or-newer device.
     *
     * Parallel participant capture can inspect different ordinals at once.
     * Caching an ordinal and result in separate statics races their publication.
     * Physical facts instead share the immutable startup authority; this family
     * predicate is independent of its prepared operand and arithmetic policy.
     */
    inline bool isAmperePlus(int device_id)
    {
        return llaminar2::cudaHardwareMetadata().at(device_id).sm_major >= 8;
    }

    // ─── Sweep-derived tile dispatch ───────────────────────────────────
    // Tile configurations validated across production geometries. K reduction
    // is not a free tuning dimension: only full-K and canonical public-M1
    // partition schedules are admissible.

    enum class TileId : uint8_t
    {
        T64x64_w2x2,   // BM=64  BN=64  WM=2 WN=2  (128 threads)
        T64x128_w2x2,  // BM=64  BN=128 WM=2 WN=2  (128 threads)
        T64x128_w4x2,  // BM=64  BN=128 WM=4 WN=2  (256 threads)
        T64x128_w2x4,  // BM=64  BN=128 WM=2 WN=4  (256 threads)
        T128x128_w4x2, // BM=128 BN=128 WM=4 WN=2  (256 threads)
        T128x128_w4x4, // BM=128 BN=128 WM=4 WN=4  (512 threads)
        Count,
    };

    constexpr int kDensePrefillTileCount = static_cast<int>(TileId::Count);

    /** Return whether an integer names one registered BK64 tile geometry. */
    constexpr bool isDensePrefillTileId(int tile_id)
    {
        return tile_id >= 0 && tile_id < kDensePrefillTileCount;
    }

    struct TileChoice
    {
        TileId tile;
    };

    /**
     * @brief Result of consulting the installed dense-prefill exact overlay.
     *
     * `Invalid` is deliberately distinct from `NotSelected`: a malformed
     * generated entry is a production-policy failure and must never silently
     * continue through the generic heuristic. `NotSelected` means either that
     * the runtime key was not explicitly swept or that a profiler force control
     * owns this process-local launch.
     */
    enum class DensePrefillOverlayStatus : uint8_t
    {
        NotSelected,
        Selected,
        Invalid,
    };

    /**
     * @brief Return whether an identifier has a compiled NativeQuantised prefill path.
     *
     * Workspace planning and launch dispatch must accept exactly the same
     * codebook set. Centralizing the inventory prevents a newly implemented
     * format from becoming launchable without also receiving a valid persistent
     * workspace contract.
     */
    constexpr bool isSupportedNativeQuantisedPrefillCodebook(uint8_t codebook)
    {
        switch (codebook)
        {
        case 0:
        case 4:
        case llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook:
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
        case 15:
        case 16:
        case 17:
        case 19:
        case llaminar2::kNativeQuantisedExpandedInt8MinCodebook:
            return true;
        default:
            return false;
        }
    }

    /**
     * @brief Validate one generated overlay tuple before it controls execution.
     *
     * BK256 launch identifiers exist only for Q4_0 and cannot preserve the
     * canonical public-M1 partition tree. Ordinary BK64 tiles cover every
     * supported codebook and may select either direct or canonical arithmetic.
     */
    constexpr bool isValidDensePrefillOverlayConfig(
        uint8_t codebook,
        const llaminar2::cuda::generated::CUDADensePrefillOverlayConfig &config)
    {
        if (!isKnownPrefillStagingSchedule(config.staging))
            return false;
        const bool ordinary_tile = isDensePrefillTileId(config.tile_id);
        const bool valid_bk256 =
            codebook == 0 && config.bk256 && !config.canonical_kpart &&
            config.staging == PrefillStagingSchedule::RegisterDecode &&
            (config.tile_id == -3 || config.tile_id == -2);
        const bool valid_bk64 = !config.bk256 && ordinary_tile;
        return valid_bk256 || valid_bk64;
    }

    /**
     * @brief Resolve one authenticated dense-prefill launch before capture.
     *
     * Exact overlays use source4 for the explicit IQ4 tuple family and otherwise
     * the runtime-visible codebook and matrix
     * geometry `(M,N,K)`. Profiler force controls have higher precedence so the
     * sweep can continue to measure every physical candidate after an overlay
     * has been installed. Ordinary production has no such controls and sees the
     * exact overlay before the total generic policy.
     *
     * @param codebook Runtime NativeQuantised execution codebook.
     * @param M Number of prefill rows in the captured bucket.
     * @param N Projection output width.
     * @param K Projection reduction width.
     * @param config Receives the concrete launch tuple on `Selected`.
     * @return Selection status, retaining malformed generated state as fatal.
     */
    inline DensePrefillOverlayStatus selectDensePrefillOverlay(
        uint8_t codebook,
        int M,
        int N,
        int K,
        llaminar2::cuda::generated::CUDADensePrefillOverlayConfig &config)
    {
        const bool profiler_override =
            !hostState().exact_overlay_enabled ||
            hostState().force_canonical_kpart ||
            stagingSchedule() != PrefillStagingSchedule::RegisterDecode ||
            hostState().force_tile_id >= 0 ||
            hostState().bk256_force_mode != 0;
        if (profiler_override)
            return DensePrefillOverlayStatus::NotSelected;

        // IQ4_XS owns physical24 kernels and resources, but retains exactly
        // the source4 tile/staging/reduction schedule rather than losing its
        // admitted cells when the physical metadata identity changes.
        const uint8_t policy_codebook = codebook == llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook
            ? uint8_t{4} : codebook;
        if (!llaminar2::cuda::generated::selectCUDADensePrefillOverlay(
                policy_codebook, M, N, K, config))
        {
            return DensePrefillOverlayStatus::NotSelected;
        }

        return isValidDensePrefillOverlayConfig(codebook, config)
                   ? DensePrefillOverlayStatus::Selected
                   : DensePrefillOverlayStatus::Invalid;
    }

    /**
     * @brief Query bytes for one row count using the public M=1 reduction tree.
     *
     * The partition boundaries are owned by the serial-decode policy. Prefill
     * may reuse that arithmetic only after this query proves an ordered reducer
     * with more than one partition on the selected device.
     */
    inline bool queryCanonicalPrefillWorkspaceForRows(
        uint8_t codebook_id,
        int rows,
        int N,
        int K,
        int sm_count,
        size_t *canonical_kpart_partials_bytes,
        int *planned_k_partitions)
    {
        int uses_ordered_reducer = 0;
        int k_partitions = 0;
        if (!cudaNativeQuantisedGemvTuned_queryCanonicalM1Schedule(
                codebook_id,
                N,
                K,
                sm_count,
                &uses_ordered_reducer,
                &k_partitions) ||
            uses_ordered_reducer == 0 || k_partitions <= 1)
        {
            return false;
        }

        const size_t partitions = static_cast<size_t>(k_partitions);
        const size_t row_count = static_cast<size_t>(rows);
        const size_t columns = static_cast<size_t>(N);
        constexpr size_t element_bytes = sizeof(float);
        if (row_count > std::numeric_limits<size_t>::max() / partitions ||
            row_count * partitions >
                std::numeric_limits<size_t>::max() / columns ||
            row_count * partitions * columns >
                std::numeric_limits<size_t>::max() / element_bytes)
        {
            return false;
        }

        if (canonical_kpart_partials_bytes)
        {
            *canonical_kpart_partials_bytes =
                partitions * row_count * columns * element_bytes;
        }
        if (planned_k_partitions)
            *planned_k_partitions = k_partitions;
        return true;
    }


    /**
     * @brief Map every supported prefill codebook to its one physical shard.
     *
     * The groups match the decode GEMV shards. Unsupported identifiers map to
     * -1; the inventory and the map are asserted equal below, so a newly
     * supported format cannot become launchable without an owning shard.
     * @param codebook Runtime execution codebook.
     * @return Shard index in [0, 7], or -1.
     */
    constexpr int prefillShardIndexForCodebook(uint8_t codebook)
    {
        switch (codebook)
        {
        case 0:
        case 4:
        case llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook:
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
        case llaminar2::kNativeQuantisedExpandedInt8MinCodebook:
            return 7;
        default:
            return -1;
        }
    }

    /** @return Whether the inventory and the shard map agree on every identifier. */
    constexpr bool prefillShardMapCoversInventory()
    {
        for (int codebook = 0; codebook < 256; ++codebook)
        {
            const auto id = static_cast<uint8_t>(codebook);
            if (isSupportedNativeQuantisedPrefillCodebook(id) !=
                (prefillShardIndexForCodebook(id) >= 0))
                return false;
        }
        return true;
    }
    static_assert(prefillShardMapCoversInventory(),
                  "every supported CUDA prefill codebook needs exactly one shard");
} // namespace llaminar2::cuda::prefill_host
