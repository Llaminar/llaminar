/**
 * @file DeviceIntegerBlockScan.h
 * @brief Cooperative, exact integer prefixes for CUDA and HIP route metadata.
 *
 * Every lane participates, including lanes outside a short input prefix. Wave
 * scans and a first-wave scan of wave totals replace data-sized lane-zero loops.
 * This is only for bounded nonnegative integer counts: floating-point reduction
 * order is a different contract and must never use this helper implicitly.
 * Scratch is block-owned shared memory, not an allocation or a host mirror.
 */
#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
namespace llaminar2::device_integer_scan
{
    /** @brief Per-lane exclusive prefix and the identical complete block total. */
    struct Result
    {
        int exclusive;
        int total;
    };

    /**
     * @brief Shared totals for one fixed-size workgroup, reusable after each call.
     * @tparam Threads Full workgroup size; whole wave32/wave64 groups are required.
     */
    template<int Threads>
    struct Scratch
    {
        static_assert(Threads >= 64 && Threads <= 1024 && Threads % 64 == 0);
        // HIP can compile either wave width. CUDA uses the wave32 capacity.
        int wave_totals[Threads / 32];
    };

    /**
     * @brief Exact inclusive integer sum in one full, converged hardware wave.
     * @param value This lane's nonnegative count; all partial sums must fit int.
     * @return Sum through this lane in ascending lane order.
     */
    __device__ __forceinline__ int waveInclusive(int value)
    {
        const int lane = static_cast<int>(threadIdx.x) % warpSize;
        for (int distance = 1; distance < warpSize; distance *= 2)
        {
#if defined(__HIPCC__)
            const int previous = __shfl_up(value, distance, warpSize);
#else
            const int previous = __shfl_up_sync(0xffffffffu, value, distance);
#endif
            if (lane >= distance)
                value += previous;
        }
        return value;
    }

    /**
     * @brief Scan one count per lane with no serial data-sized loop.
     * @tparam Threads Exact launched workgroup width.
     * @param value Count owned by this lane; inactive input lanes supply zero.
     * @param scratch Block-owned storage, shared by all callers in this block.
     * @return This lane's exclusive prefix and the block total.
     * @details All Threads lanes must call together. Three block barriers
     * publish wave sums, publish their prefixes, then protect scratch reuse.
     * That final edge matters when one wave reaches the next chunk before its
     * peers have read the current totals. Valid route capacity bounds the sum.
     */
    template<int Threads>
    __device__ __forceinline__ Result exclusive(int value, Scratch<Threads> &scratch)
    {
        const int lane = static_cast<int>(threadIdx.x) % warpSize;
        const int wave = static_cast<int>(threadIdx.x) / warpSize;
        const int waves = Threads / warpSize;
        const int inclusive = waveInclusive(value);
        if (lane == warpSize - 1)
            scratch.wave_totals[wave] = inclusive;
        __syncthreads();

        // The first wave cooperatively scans at most 32 wave totals. Other
        // waves wait at the publication barrier, not behind a scalar loop.
        if (wave == 0)
        {
            const int total = waveInclusive(lane < waves ? scratch.wave_totals[lane] : 0);
            if (lane < waves)
                scratch.wave_totals[lane] = total;
        }
        __syncthreads();
        const Result result{
            inclusive - value + (wave == 0 ? 0 : scratch.wave_totals[wave - 1]),
            scratch.wave_totals[waves - 1]};
        __syncthreads();
        return result;
    }

    /**
     * @brief Publish an arbitrary-length exclusive count scan in ascending chunks.
     * @tparam Threads Exact launched workgroup width.
     * @param counts Immutable input counts; their complete sum must fit int.
     * @param offsets Nonaliasing output of at least count entries.
     * @param count Number of experts; the caller admits a positive capacity.
     * @param scratch Block-owned reusable wave totals.
     * @details Each chunk has coalesced loads/stores. Only the number of chunks
     * is sequential; one lane never walks the complete expert array alone.
     */
    template<int Threads>
    __device__ __forceinline__ void arrayExclusive(
        const int *counts, int *offsets, int count, Scratch<Threads> &scratch)
    {
        int previous_chunks = 0;
        for (int base = 0; base < count; base += Threads)
        {
            const int index = base + static_cast<int>(threadIdx.x);
            const auto prefix = exclusive(index < count ? counts[index] : 0, scratch);
            if (index < count)
                offsets[index] = previous_chunks + prefix.exclusive;
            previous_chunks += prefix.total;
        }
    }
}
#endif
