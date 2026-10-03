/**
 * @file MoEGroupingInitialization.h
 * @brief One stream-ordered initialization contract for GPU expert grouping.
 *
 * The workspace owns every pointer. A captured initialization resets the whole
 * admitted grouping extent before counting/scattering, including slots made
 * inactive by a smaller request. Invalid maps use -1; unused compact rows and
 * their weights use exact zero. One parallel launch replaces independent native
 * memsets without host state, new storage, or a change to routing arithmetic.
 */
#pragma once

namespace llaminar2
{
    /** @brief Borrowed, nonaliasing grouping arrays and their admitted extents. */
    struct MoEGroupingInitialization final
    {
        int *expert_counts = nullptr;
        int *write_heads = nullptr;             ///< Optional scatter cursor bank.
        int *original_to_grouped = nullptr;     ///< Optional original-slot map.
        int *original_expert_ids = nullptr;     ///< Optional original expert map.
        int *grouped_token_indices = nullptr;
        float *grouped_weights = nullptr;
        int num_experts = 0;
        int total_slots = 0;

        /** @brief Reject missing required publications or empty/negative extents. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return expert_counts && grouped_token_indices && grouped_weights &&
                num_experts > 0 && total_slots > 0;
        }

        /** @brief Number of independent lanes needed to cover both array families. */
        [[nodiscard]] constexpr int workItems() const noexcept
        {
            return num_experts > total_slots ? num_experts : total_slots;
        }

#if defined(__CUDACC__) || defined(__HIPCC__)
        /**
         * @brief Reset one coordinate without touching neighbouring guards.
         * @param index Nonnegative lane index; padded workgroup lanes are legal.
         * @details The next count/scatter launch is ordered on the same stream.
         * No lane reads a peer's writes, so this operation needs no block barrier.
         */
        __device__ __forceinline__ void resetIndex(unsigned int index) const
        {
            if (index < static_cast<unsigned int>(num_experts))
            {
                expert_counts[index] = 0;
                if (write_heads) write_heads[index] = 0;
            }
            if (index < static_cast<unsigned int>(total_slots))
            {
                if (original_to_grouped) original_to_grouped[index] = -1;
                if (original_expert_ids) original_expert_ids[index] = -1;
                grouped_token_indices[index] = 0;
                grouped_weights[index] = 0.0f;
            }
        }
#endif
    };
}

extern "C"
{
#ifdef HAVE_CUDA
    /**
     * @brief Enqueue the CUDA reset on an exact non-default stream.
     * @param binding Borrowed persistent arrays; validated before launch.
     * @param device_ordinal Owning CUDA device.
     * @param stream Non-null CUDA stream ordering reset before grouping.
     * @return False on invalid binding, device admission, or launch failure.
     */
    bool cudaMoE_initialize_grouping(
        llaminar2::MoEGroupingInitialization binding, int device_ordinal, void *stream);
#endif
#ifdef HAVE_ROCM
    /**
     * @brief Enqueue the HIP reset using the identical array/sentinel contract.
     * @param binding Borrowed persistent arrays; validated before launch.
     * @param device_ordinal Owning ROCm device.
     * @param stream Non-null HIP stream ordering reset before grouping.
     * @return False on invalid binding, device admission, or launch failure.
     */
    bool hipMoE_initialize_grouping(
        llaminar2::MoEGroupingInitialization binding, int device_ordinal, void *stream);
#endif
}
