/**
 * @file SamplingWideTopKDevice.inl
 * @brief Captured batched K=65..256 selection with bounded shared-memory lists.
 *
 * CUDA and HIP instantiate the same exact value/token ordering. Sixteen lanes
 * keep sorted candidates in shared memory; one complete warp/wave merges their
 * heads without lane-private K-entry arrays. The second kernel merges sorted
 * vocabulary partitions and uses the canonical SamplingMath normalization.
 * Scratch is a bounded view of already admitted Top-K storage, never a new
 * allocation. Arbitrarily large vocabularies are streamed through those lists.
 */
#pragma once

#include "SamplingMath.h"

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <optional>

namespace llaminar2::sampling_wide
{
    constexpr int kLaneLists = 16;
    constexpr int kPartialLists = 32;

    /** @brief Immutable launch geometry constrained by admitted scratch bytes. */
    struct Geometry
    {
        int partial_blocks;
        std::size_t shared_bytes;
    };

    /**
     * @brief Resolve wide Top-K partitions without increasing persistent storage.
     * @param vocabulary Positive input row width.
     * @param rows Positive captured row capacity, not a device live-row readback.
     * @param k Requested compact width in the supported wide range.
     * @param scratch_entries Available entries in each admitted value/id buffer.
     * @return No geometry if the policy or minimum scratch footprint is invalid.
     */
    inline std::optional<Geometry> geometry(
        int vocabulary, int rows, int k, int scratch_entries) noexcept
    {
        if (vocabulary <= 0 || rows <= 0 || k <= 64 ||
            k > sampling_math::kMaxTopK || k > vocabulary || scratch_entries <= 0)
            return std::nullopt;
        const int available_lists = scratch_entries / rows / k;
        const int vocabulary_lists = 1 + (vocabulary - 1) / kLaneLists;
        const int partitions = std::min({kPartialLists, available_lists, vocabulary_lists});
        if (partitions <= 0)
            return std::nullopt;
        return Geometry{partitions, static_cast<std::size_t>(kLaneLists) * k *
                                        (sizeof(float) + sizeof(int))};
    }

#if defined(__CUDACC__) || defined(__HIPCC__)
    /** @brief One candidate; negative ids represent empty sorted lists. */
    struct Candidate { float value; int index; };

    /** @brief The same total order as existing Top-K: value descending, id ascending. */
    __device__ __forceinline__ bool better(Candidate candidate, Candidate current)
    {
        return candidate.index >= 0 && (current.index < 0 ||
            candidate.value > current.value ||
            (candidate.value == current.value && candidate.index < current.index));
    }

    /** @brief Broadcast one field within the kernel's complete warp or wave. */
    template <int Width, class T>
    __device__ __forceinline__ T broadcast(T value)
    {
#if defined(__HIPCC__)
        return __shfl(value, 0, Width);
#else
        return __shfl_sync(0xffffffffu, value, 0, Width);
#endif
    }

    /** @brief Select the exact best head using a fixed warp/wave reduction tree. */
    template <int Width>
    __device__ __forceinline__ Candidate bestHead(Candidate head)
    {
        const int lane = static_cast<int>(threadIdx.x);
        for (int offset = Width / 2; offset > 0; offset >>= 1)
        {
#if defined(__HIPCC__)
            const Candidate peer{__shfl_down(head.value, offset, Width),
                                 __shfl_down(head.index, offset, Width)};
#else
            const Candidate peer{__shfl_down_sync(0xffffffffu, head.value, offset, Width),
                                 __shfl_down_sync(0xffffffffu, head.index, offset, Width)};
#endif
            if (lane + offset < Width && better(peer, head))
                head = peer;
        }
        return {broadcast<Width>(head.value), broadcast<Width>(head.index)};
    }

    /**
     * @brief Merge immutable sorted lists using one register cursor per lane.
     * @param values Calling lane's candidate values, or null for an empty lane.
     * @param indices Corresponding globally unique vocabulary token ids.
     * @param count Number of valid entries in that lane's list.
     * @param k Number of ranks to publish.
     * @param output_values Shared or global output; lane zero is its only writer.
     * @param output_indices Matching output token ids.
     */
    template <int Width>
    __device__ __forceinline__ void merge(
        const float *values, const int *indices, int count, int k,
        float *output_values, int *output_indices)
    {
        int cursor = 0;
        for (int rank = 0; rank < k; ++rank)
        {
            const Candidate head = cursor < count
                ? Candidate{values[cursor], indices[cursor]} : Candidate{-FLT_MAX, -1};
            const Candidate winner = bestHead<Width>(head);
            if (threadIdx.x == 0)
            {
                output_values[rank] = winner.value;
                output_indices[rank] = winner.index;
            }
            // Vocabulary ids are unique: only the list owning this head moves.
            if (head.index >= 0 && head.index == winner.index)
                ++cursor;
        }
    }

    /**
     * @brief Stream one vocabulary partition into bounded lane-owned sorted lists.
     *
     * All lanes participate in the final merge, including the lanes with no
     * list. Large K therefore never needs a partially active hardware wave or
     * more than 32 KiB of dynamic shared memory on either backend.
     */
    template <int Width>
    __global__ __launch_bounds__(Width, 4) void partials(
        const float *data, int vocabulary, int row_stride, int k, int partitions,
        float *partial_values, int *partial_indices, const int *active_rows)
    {
        const int row = static_cast<int>(blockIdx.y);
        const int live = active_rows ? *active_rows : static_cast<int>(gridDim.y);
        if (live <= 0 || live > static_cast<int>(gridDim.y) || row >= live)
            return;
        const int lane = static_cast<int>(threadIdx.x);
        const int partition = static_cast<int>(blockIdx.x);
        const int begin = static_cast<int>(static_cast<long long>(partition) * vocabulary / partitions);
        const int end = static_cast<int>(static_cast<long long>(partition + 1) * vocabulary / partitions);
        extern __shared__ unsigned char scratch[];
        auto *lists = reinterpret_cast<float *>(scratch);
        auto *ids = reinterpret_cast<int *>(lists + kLaneLists * k);
        float *values = lane < kLaneLists ? lists + lane * k : nullptr;
        int *indices = lane < kLaneLists ? ids + lane * k : nullptr;
        int count = 0;
        if (lane < kLaneLists)
        {
            const float *input = data + static_cast<std::size_t>(row) * row_stride;
            for (int token = begin + lane; token < end; token += kLaneLists)
            {
                const Candidate candidate{input[token], token};
                if (count == k && !better(candidate, {values[k - 1], indices[k - 1]}))
                    continue;
                int position = count < k ? count : k - 1;
                for (; position > 0 && better(candidate, {values[position - 1], indices[position - 1]});
                     --position)
                {
                    values[position] = values[position - 1];
                    indices[position] = indices[position - 1];
                }
                values[position] = candidate.value;
                indices[position] = candidate.index;
                if (count < k)
                    ++count;
            }
        }
        __syncthreads();
        const auto output = (static_cast<std::size_t>(row) * partitions + partition) * k;
        merge<Width>(values, indices, count, k, partial_values + output, partial_indices + output);
    }

    /** @brief Merge sorted partitions and normalize with unchanged SamplingMath arithmetic. */
    template <int Width>
    __global__ __launch_bounds__(Width, 4) void distributions(
        const float *partial_values, const int *partial_indices, int partitions,
        int k, float top_p, float temperature, int *output_ids, int output_stride,
        float *output_probs, const int *active_rows)
    {
        const int row = static_cast<int>(blockIdx.x);
        const int live = active_rows ? *active_rows : static_cast<int>(gridDim.x);
        if (live <= 0 || live > static_cast<int>(gridDim.x) || row >= live)
            return;
        const int lane = static_cast<int>(threadIdx.x);
        const auto input = (static_cast<std::size_t>(row) * partitions + lane) * k;
        __shared__ float selected_values[sampling_math::kMaxTopK];
        __shared__ int selected_indices[sampling_math::kMaxTopK];
        __shared__ float weights[sampling_math::kMaxTopK];
        merge<Width>(lane < partitions ? partial_values + input : nullptr,
                     lane < partitions ? partial_indices + input : nullptr,
                     lane < partitions ? k : 0, k, selected_values, selected_indices);
        if (lane == 0)
        {
            const auto output = static_cast<std::size_t>(row) * output_stride;
            sampling_math::build_topk_topp_distribution_from_sorted(
                selected_values, selected_indices, k, top_p, temperature,
                output_ids + output, output_probs + output, weights);
        }
    }
#endif
} // namespace llaminar2::sampling_wide
