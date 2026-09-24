/**
 * @file NativeVNNIPrefillProbePlan.h
 * @brief Pure planning helpers for deduplicating CPU prefill route probes.
 *
 * The CPU NativeVNNI prefill launcher preserves the accumulation order of the
 * production serial M=1 route.  A shape whose serial route partitions K must
 * therefore execute every M>1 request through the decode-equivalent K-part
 * family.  Conversely, a shape whose serial route covers full K cannot enter
 * a K-part candidate merely because the trainer requested one.
 *
 * The performance harness still emits one evidence row for every logical
 * candidate.  This helper predicts which physical route a mismatched request
 * will normalize to, allowing the harness to exercise that physical route
 * once and reuse the resulting correctness probe for unsupported aliases.
 * Canonical timing remains exclusive to candidates whose requested and
 * observed route IDs agree.
 */

#pragma once

#include <stdexcept>
#include <string_view>

namespace llaminar2::test::trainer
{
    inline constexpr std::string_view kCPUPrefillRowChunkCandidate =
        "cpu.nvnni.prefill.row_chunk_grid.full_k";
    inline constexpr std::string_view kCPUPrefillTwoRowNbc1Candidate =
        "cpu.nvnni.prefill.two_row_tiles.nbc1.full_k";
    inline constexpr std::string_view kCPUPrefillKPartPairwiseCandidate =
        "cpu.nvnni.prefill.decode_equivalent_kpart.pairwise";
    inline constexpr std::string_view kCPUPrefillKPartWideRowsCandidate =
        "cpu.nvnni.prefill.decode_equivalent_kpart.wide_rows";

    /**
     * @brief Predict the effective production route for one trainer request.
     *
     * @param requested_candidate_id Registry ID requested by the trainer.
     * @param serial_m1_uses_kpart Whether production M=1 partitions K.
     * @param candidate_uses_kpart Whether the requested candidate belongs to
     *        the decode-equivalent K-part family.
     * @param candidate_requests_wide_rows Whether the candidate asks for the
     *        AVX-512 wide-row K-part implementation.
     * @param effective_runtime_is_avx512 Whether runtime dispatch selected
     *        AVX-512 rather than AVX2.
     * @param M Number of grouped input rows. Ordinary prefill requires M >= 2.
     * @param N Number of output columns.
     * @param threads Number of threads visible to the production launcher.
     * @return Registry ID of the physical route production will execute.
     *
     * @throws std::invalid_argument when the supplied geometry is not an
     *         ordinary-prefill launch.
     *
     * @note This function intentionally models only arithmetic-family
     * normalization. Compatible full-K candidates retain their exact registry
     * ID because their explicit schedule override remains authoritative.
     */
    inline std::string_view expectedCPUPrefillPhysicalCandidateId(
        std::string_view requested_candidate_id,
        bool serial_m1_uses_kpart,
        bool candidate_uses_kpart,
        bool candidate_requests_wide_rows,
        bool effective_runtime_is_avx512,
        int M,
        int N,
        int threads)
    {
        if (M < 2 || N <= 0 || threads <= 0)
        {
            throw std::invalid_argument(
                "CPU prefill probe planning requires M >= 2, N > 0, and "
                "threads > 0");
        }

        if (serial_m1_uses_kpart)
        {
            // Production ignores every full-K schedule request in this
            // regime. Wide rows are a distinct physical implementation only
            // when AVX-512 can process at least three rows together.
            return candidate_requests_wide_rows &&
                           effective_runtime_is_avx512 && M >= 3
                       ? kCPUPrefillKPartWideRowsCandidate
                       : kCPUPrefillKPartPairwiseCandidate;
        }

        if (!candidate_uses_kpart)
            return requested_candidate_id;

        // A K-part request on a serial-full-K shape carries an Auto schedule
        // and nbc=1 into the production launcher. Auto selects the row/chunk
        // grid when there are too few N chunks for the thread team, or when a
        // sub-64 output stride makes the two-row microkernel inapplicable.
        const int n_chunks = (N + 63) / 64;
        const int row_chunk_task_limit = threads / 4;
        return N < 64 || n_chunks <= row_chunk_task_limit
                   ? kCPUPrefillRowChunkCandidate
                   : kCPUPrefillTwoRowNbc1Candidate;
    }
} // namespace llaminar2::test::trainer
