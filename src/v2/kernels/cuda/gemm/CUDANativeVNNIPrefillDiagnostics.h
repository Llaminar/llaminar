/**
 * @file CUDANativeVNNIPrefillDiagnostics.h
 * @brief Setup-time resource inspection for exact CUDA NativeVNNI prefill kernels.
 *
 * Dense prefill policy is allowed to choose only compiler specializations that
 * preserve the arithmetic contract and remain economical. This interface makes
 * the compiler-resource half of that contract explicit: trainers can inspect a
 * concrete tile before timing it, while production-route tests can inspect the
 * exact specialization selected by one untimed launch. These functions query
 * CUDA metadata only; they allocate no memory, transfer no data, launch no
 * kernel, and synchronize no stream or device.
 */

#pragma once

#include <cstddef>
#include <cstdint>

/** Compiler and theoretical-occupancy evidence for one CUDA kernel symbol. */
struct CUDADensePrefillKernelResources
{
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
    /**
     * @brief Inspect the exact primary and auxiliary kernels from the last launch.
     *
     * The caller must first issue one untimed launch on the same host thread.
     * The launch publishes its tile, BK256 mode, and canonical-reducer identity
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
     * @return `true` when the candidate identity is launchable and queryable.
     */
    bool cudaNativeVNNIPrefill_queryCandidateResources(
        std::uint8_t codebook_id,
        int tile_id,
        int canonical_kpart,
        int ordered_bk256,
        int cuda_device_id,
        CUDADensePrefillKernelResources *primary,
        CUDADensePrefillKernelResources *auxiliary);
}
