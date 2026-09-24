/**
 * @file CUDANativeVNNIPrefillDiagnostics.h
 * @brief Setup-time resource inspection for exact CUDA NativeVNNI prefill kernels.
 *
 * Dense prefill policy is allowed to choose only compiler specializations that
 * preserve the arithmetic contract and remain economical. This interface makes
 * the compiler-resource half of that contract explicit: trainers can inspect a
 * concrete tile before timing it, while production-route tests can inspect the
 * exact specialization selected by one untimed launch. These functions inspect
 * CUDA metadata or select thread-local diagnostic staging; they allocate no
 * memory, transfer no data, launch no kernel, and synchronize no stream/device.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include "CUDANativeVNNIPrefillSchedule.h"

/** Compiler and theoretical-occupancy evidence for one CUDA kernel symbol. */
struct CUDADensePrefillKernelResources
{
    /** Exact core-owned CUDA symbol; valid while the queried module is loaded.
     * Diagnostic fixtures may capture this symbol instead of recompiling the
     * device header under different flags. It is never serialized as an ID.
     */
    const void *kernel_symbol = nullptr;
    int registers_per_thread = 0;
    std::size_t local_memory_bytes_per_thread = 0;
    std::size_t static_shared_memory_bytes = 0;
    std::size_t dynamic_shared_memory_bytes = 0;
    int threads_per_block = 0;
    int max_threads_per_block = 0;
    int max_active_blocks_per_sm = 0;
};

extern "C"
{
    /** @brief Set the calling thread's explicit diagnostic staging selection.
     * @param schedule Physical operand-delivery schedule; no automatic substitute.
     * @return False for an unknown value, leaving the prior selection unchanged.
     * A structurally incompatible tile/codebook is rejected by launch admission.
     * Production leaves RegisterDecode selected until certified policy owns it.
     */
    bool cudaNativeVNNIPrefill_setStagingSchedule(
        llaminar2::cuda::prefill::PrefillStagingSchedule schedule);

    /** @brief Return the calling thread's current diagnostic staging choice. */
    llaminar2::cuda::prefill::PrefillStagingSchedule
    cudaNativeVNNIPrefill_getStagingSchedule();

    /** @brief Return the staging identity published by the last probed launch. */
    llaminar2::cuda::prefill::PrefillStagingSchedule
    cudaNativeVNNIPrefill_getLastLaunchStagingSchedule();

    /**
     * @brief Inspect the exact primary and auxiliary kernels from the last launch.
     *
     * The caller must first issue one untimed launch on the same host thread.
     * The launch publishes its tile, staging, BK256 mode, and reducer identity
     * into thread-local diagnostics; this query resolves those diagnostics to
     * concrete symbols and returns their compiler resources.
     *
     * @param codebook_id Runtime NativeVNNI execution codebook.
     * @param n Output-column extent from the probed launch.
     * @param k Reduction extent from the probed launch.
     * @param cuda_device_id CUDA ordinal owning the compiled specialization.
     * @param primary Receives the main GEMM kernel resources.
     * @param auxiliary Receives the ordered reducer resources for canonical
     *        K-partition launches, or an all-zero record otherwise.
     * @return `true` only when the published route is valid and every involved
     *         symbol has non-zero theoretical occupancy.
     */
    bool cudaNativeVNNIPrefill_queryLastLaunchResources(
        std::uint8_t codebook_id,
        int n,
        int k,
        int cuda_device_id,
        CUDADensePrefillKernelResources *primary,
        CUDADensePrefillKernelResources *auxiliary);

    /**
     * @brief Inspect one forceable candidate without launching it.
     *
     * This is the exhaustive inventory surface used before a sweep constructs
     * tasks. BK64 candidates use tile IDs 0..5 and select direct or canonical
     * reduction with `canonical_kpart`. Q4_0 BK256 candidates use tile ID -3
     * (128x64) or -2 (128x128), and `ordered_bk256` selects the exact compile-
     * time arithmetic specialization used by the public M=1 schedule.
     *
     * @param codebook_id Runtime NativeVNNI execution codebook.
     * @param tile_id Physical BK64 or BK256 tile identifier.
     * @param canonical_kpart Non-zero for the ordered BK64 partition kernel.
     * @param ordered_bk256 Non-zero for the ordered Q4_0 BK256 specialization.
     * @param cuda_device_id CUDA ordinal used for occupancy calculation.
     * @param primary Receives the main GEMM kernel resources.
     * @param auxiliary Receives canonical reducer resources or an all-zero row.
     * @param staging Exact operand-delivery specialization. BK256 supports only
     *        RegisterDecode. Unsupported BK64 ownership geometry is rejected.
     * @return `true` when the candidate identity is launchable and queryable.
     */
    bool cudaNativeVNNIPrefill_queryCandidateResources(
        std::uint8_t codebook_id,
        int tile_id,
        int canonical_kpart,
        int ordered_bk256,
        int cuda_device_id,
        CUDADensePrefillKernelResources *primary,
        CUDADensePrefillKernelResources *auxiliary,
        llaminar2::cuda::prefill::PrefillStagingSchedule staging =
            llaminar2::cuda::prefill::PrefillStagingSchedule::RegisterDecode);
}
