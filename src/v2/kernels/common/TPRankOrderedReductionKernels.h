/**
 * @file TPRankOrderedReductionKernels.h
 * @brief Device launch ABI for fixed-rank tensor-parallel reductions.
 *
 * Native NCCL/RCCL all-reduce is free to choose a different reduction tree
 * when the message contains one decode row versus several grouped verifier
 * rows.  The launchers declared here perform only the arithmetic half of the
 * batch-invariant protocol: an earlier raw allgather publishes immutable
 * rank-major FP32 banks, then one device kernel folds those banks in ascending
 * participant order.  Transport therefore moves bytes without arithmetic and
 * the visible sum has one stable order for every runtime row count. Live-prefix
 * overloads share the native row authority and preserve inactive in-place bytes.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include "collective/NativeCollectiveRows.h"

namespace llaminar2
{
    /**
     * @brief Fold rank-major FP32 banks on a CUDA device.
     *
     * @param rank_banks Device input shaped `[rank_count, element_count]`.
     * @param output Device destination with `element_count` FP32 values.
     * @param element_count Number of values contributed by every participant.
     * @param rank_count Number of participant banks to fold in ascending order.
     * @param device_index CUDA ordinal that owns both allocations.
     * @param stream Exact non-default CUDA stream ordered after the allgather.
     * @return true when the kernel was enqueued successfully.
     */
    bool launchCUDATPRankOrderedSumFP32(
        const float *rank_banks,
        float *output,
        std::size_t element_count,
        int rank_count,
        int device_index,
        void *stream);

    /**
     * @brief Fold rank-major FP32 banks on a ROCm device.
     *
     * @param rank_banks Device input shaped `[rank_count, element_count]`.
     * @param output Device destination with `element_count` FP32 values.
     * @param element_count Number of values contributed by every participant.
     * @param rank_count Number of participant banks to fold in ascending order.
     * @param device_index HIP ordinal that owns both allocations.
     * @param stream Exact non-default HIP stream ordered after the allgather.
     * @return true when the kernel was enqueued successfully.
     */
    bool launchROCmTPRankOrderedSumFP32(
        const float *rank_banks,
        float *output,
        std::size_t element_count,
        int rank_count,
        int device_index,
        void *stream);

    /**
     * @brief Fold only a native collective's live FP32 row prefix on CUDA.
     * @param rank_banks Immutable rank-major banks with admitted capacity strides.
     * @param output Original in-place FP32 bank; inactive bytes are untouched.
     * @param rows Same whole-prefix descriptor consumed by the preceding allgather.
     * @param rank_count Physical participants folded in ascending rank order.
     * @param device_index Exact participant ordinal owning both banks.
     * @param stream Non-null ordered stream consuming the native publication.
     * @return Whether the native kernel was admitted and enqueued successfully.
     */
    bool launchCUDATPRankOrderedSumFP32(
        const float *rank_banks, float *output, const NativeCollectiveRows &rows,
        int rank_count, int device_index, void *stream);

    /**
     * @brief Fold only a native collective's live FP32 row prefix on ROCm.
     * @param rank_banks Immutable rank-major banks with admitted capacity strides.
     * @param output Original in-place FP32 bank; inactive bytes are untouched.
     * @param rows Same whole-prefix descriptor consumed by the preceding allgather.
     * @param rank_count Physical participants folded in ascending rank order.
     * @param device_index Exact participant ordinal owning both banks.
     * @param stream Non-null ordered stream consuming the native publication.
     * @return Whether the native kernel was admitted and enqueued successfully.
     */
    bool launchROCmTPRankOrderedSumFP32(
        const float *rank_banks, float *output, const NativeCollectiveRows &rows,
        int rank_count, int device_index, void *stream);

    /**
     * @brief Borrow the original local partial and its communicator position.
     *
     * The index is the position in the original ordered communicator, never a
     * device ordinal. The enclosing retained graph owns these immutable bytes;
     * this value creates neither storage nor a host execution-state authority.
     */
    struct TPRankOrderedLocalContributionFP32 final
    {
        const float *data; ///< Original participant-local FP32 contribution.
        std::size_t capacity_elements; ///< Complete admitted source range.
        int participant_index; ///< Explicit position in the original membership.
    };

    /**
     * @brief Validate a fixed-message fold with a separately retained local partial.
     *
     * Peer banks retain original source-indexed capacity strides; the bank at
     * local.participant_index is deliberately not an operand. All other banks
     * contain exactly element_count values published by their original channel.
     * The local value replaces only its own rank's load. Additions still start
     * from rank zero and proceed in the exact original ascending order.
     *
     * This is a borrowed launch contract, not a collective or an allocation
     * owner. The component caller must retain the original PMA/TransferEngine
     * owners, order all sends before a possible in-place fold, and include every
     * pointer, extent, stride and membership index in retained graph identity.
     * No production transport selects this diagnostic leaf automatically.
     */
    class TPRankOrderedLocalSumFP32 final
    {
    public:
        /**
         * @brief Freeze complete ranges before any device selection or enqueue.
         * @param local Original local contribution and explicit communicator index.
         * @param peer_banks Original source-indexed peer storage; own slot is unused.
         * @param peer_capacity_elements Complete admitted peer allocation range.
         * @param bank_stride_elements Capacity stride of each original rank slot.
         * @param output FP32 destination, disjoint from all peer storage.
         * @param output_capacity_elements Complete admitted destination range.
         * @param element_count Positive exact fixed-message scalar extent.
         * @param rank_count Physical membership size, at least two.
         * @throws std::invalid_argument For null/misaligned/short ranges, invalid
         *         membership, or aliasing other than exact local/output identity.
         * @throws std::overflow_error For unrepresentable byte/address geometry.
         */
        TPRankOrderedLocalSumFP32(TPRankOrderedLocalContributionFP32 local,
            const float *peer_banks, std::size_t peer_capacity_elements,
            std::size_t bank_stride_elements, float *output,
            std::size_t output_capacity_elements, std::size_t element_count,
            int rank_count)
            : local_(local), peers_(peer_banks), stride_(bank_stride_elements),
              output_(output), elements_(element_count), ranks_(rank_count)
        {
            if (rank_count < 2 || local.participant_index < 0 ||
                local.participant_index >= rank_count || element_count == 0 ||
                element_count > local.capacity_elements ||
                element_count > output_capacity_elements ||
                bank_stride_elements < element_count)
                throw std::invalid_argument("Local rank fold requires complete fixed extent and original membership");
            const auto maximum = std::numeric_limits<std::size_t>::max();
            if (bank_stride_elements > maximum / static_cast<std::size_t>(rank_count))
                throw std::overflow_error("Local rank fold bank geometry overflows");
            if (peer_capacity_elements < bank_stride_elements * static_cast<std::size_t>(rank_count))
                throw std::invalid_argument("Local rank fold peer storage is smaller than its complete bank geometry");
            const auto source = range(local.data, local.capacity_elements);
            const auto peers = range(peer_banks, peer_capacity_elements);
            const auto destination = range(output, output_capacity_elements);
            if (overlap(peers, source) || overlap(peers, destination) ||
                (overlap(source, destination) && source != destination))
                throw std::invalid_argument("Local rank fold permits only exact local/output alias and disjoint peer storage");
        }

        /** @return Original participant-local contribution and communicator index. */
        [[nodiscard]] TPRankOrderedLocalContributionFP32 local() const noexcept { return local_; }
        /** @return Immutable source-indexed peer banks, excluding the unused local slot. */
        [[nodiscard]] const float *peerBanks() const noexcept { return peers_; }
        /** @return Admitted rank-slot stride; it never becomes a communication extent. */
        [[nodiscard]] std::size_t bankStride() const noexcept { return stride_; }
        /** @return Original exact destination pointer. */
        [[nodiscard]] float *output() const noexcept { return output_; }
        /** @return Positive fixed scalar extent consumed by this captured fold. */
        [[nodiscard]] std::size_t elements() const noexcept { return elements_; }
        /** @return Original ordered physical membership size. */
        [[nodiscard]] int rankCount() const noexcept { return ranks_; }

    private:
        /** @brief Inclusive start and exclusive end of one checked borrowed range. */
        struct Range
        {
            std::uintptr_t first, end;
            bool operator==(const Range &) const = default;
        };
        /**
         * @brief Bound the whole admitted range without pointer arithmetic.
         * @param data Non-null naturally aligned FP32 address.
         * @param elements Positive complete allocation/view extent.
         * @return Checked half-open byte address range.
         * @throws std::invalid_argument For absent/misaligned storage.
         * @throws std::overflow_error For unrepresentable bytes or address end.
         */
        static Range range(const float *data, std::size_t elements)
        {
            const auto first = reinterpret_cast<std::uintptr_t>(data);
            if (!data || first % alignof(float) || elements == 0)
                throw std::invalid_argument("Local rank fold requires aligned nonempty FP32 storage");
            if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
                elements * sizeof(float) > std::numeric_limits<std::uintptr_t>::max() - first)
                throw std::overflow_error("Local rank fold byte address span overflows");
            return {first, first + elements * sizeof(float)};
        }
        /** @return Whether two already checked complete borrowed ranges intersect. */
        static bool overlap(Range a, Range b) noexcept { return a.first < b.end && b.first < a.end; }
        TPRankOrderedLocalContributionFP32 local_; ///< Original local operand, never mirrored.
        const float *peers_; ///< Original admitted peer-bank address.
        std::size_t stride_; ///< Original peer capacity stride.
        float *output_; ///< Original destination, optionally exactly the local operand.
        std::size_t elements_; ///< Exact fixed-message scalar extent.
        int ranks_; ///< Ordered physical communicator degree.
    };

    /**
     * @brief Reject default-stream sentinels before any native device operation.
     * @param device_index Explicit nonnegative physical device ordinal.
     * @param stream Exact native stream, excluding null/legacy/per-thread defaults.
     * @return Whether launch identity is explicit; no GPU API is invoked.
     *
     * CUDA and HIP reserve native stream handles 1 and 2 for their legacy and
     * per-thread default sentinels. This matches the existing diagnostic-copy
     * exact-stream guard and is shared with device-free admission regressions.
     */
    [[nodiscard]] inline bool tpRankOrderedLocalLaunchValid(int device_index, void *stream) noexcept
    {
        return device_index >= 0 && reinterpret_cast<std::uintptr_t>(stream) > 2u;
    }

    /**
     * @brief Enqueue the validated local-aware fixed-rank fold on CUDA.
     * @param launch Complete immutable original operand/rank geometry.
     * @param device_index Actual CUDA ordinal, independent of communicator index.
     * @param stream Exact non-null ordered stream after complete sends/receives.
     * @return Whether native device selection and enqueue succeeded; no retry.
     */
    bool launchCUDATPRankOrderedLocalSumFP32(
        const TPRankOrderedLocalSumFP32 &launch, int device_index, void *stream);

    /**
     * @brief Enqueue the same validated local-aware fixed-rank fold on ROCm.
     * @param launch Complete immutable original operand/rank geometry.
     * @param device_index Actual HIP ordinal, independent of communicator index.
     * @param stream Exact non-null ordered stream after complete sends/receives.
     * @return Whether native device selection and enqueue succeeded; no retry.
     */
    bool launchROCmTPRankOrderedLocalSumFP32(
        const TPRankOrderedLocalSumFP32 &launch, int device_index, void *stream);
} // namespace llaminar2
