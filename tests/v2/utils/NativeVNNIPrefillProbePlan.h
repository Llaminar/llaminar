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
#include "kernels/cpu/gemm/CPUNativeVNNIPrefillSchedule.h"

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
     * @param encoding Actual prepared encoding; required for Auto row reuse.
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
        int threads,
        cpu::native_vnni::CPUNativeVNNIEncoding encoding =
            cpu::native_vnni::CPUNativeVNNIEncoding::ExpandedInt8)
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

        // Keep unsupported ISA/M candidates in the complete research matrix.
        // Reuse an independently measured pairwise probe, but its different
        // observed identity prevents timing or promoting the four-row label.
        // This is diagnostic evidence reuse, never production dispatch.
        if (!effective_runtime_is_avx512 || M < 3)
        {
            if (requested_candidate_id == "cpu.nvnni.prefill.four_row_grid.nbc1.full_k")
                return "cpu.nvnni.prefill.two_row_pair_grid.nbc1.full_k";
            if (requested_candidate_id == "cpu.nvnni.prefill.four_row_grid.nbc2.full_k")
                return "cpu.nvnni.prefill.two_row_pair_grid.nbc2.full_k";
            if (requested_candidate_id == "cpu.nvnni.prefill.four_row_grid.nbc4.full_k")
                return "cpu.nvnni.prefill.two_row_pair_grid.nbc4.full_k";
            if (requested_candidate_id == "cpu.nvnni.prefill.four_row_grid.nbc8.full_k")
                return "cpu.nvnni.prefill.two_row_pair_grid.nbc8.full_k";
        }
        if (!candidate_uses_kpart)
            return requested_candidate_id;

        // A K-part request on a serial-full-K shape carries Auto and nbc=1.
        // Reuse the production policy; a second heuristic here previously
        // disagreed with the real grid once wave balancing was introduced.
        using namespace cpu::native_vnni;
        switch (resolvePrefillSchedule(PrefillSchedulePolicy::Auto,
            {M, N, N, 1, threads, encoding,
             effective_runtime_is_avx512 ? PrefillRowKernelSet::WideRows
                                        : PrefillRowKernelSet::Pairwise}))
        {
        case PrefillSchedulePolicy::RowChunkGrid:
            return kCPUPrefillRowChunkCandidate;
        case PrefillSchedulePolicy::TwoRowNMajor:
            return kCPUPrefillTwoRowNbc1Candidate;
        case PrefillSchedulePolicy::TwoRowPairGrid:
            return "cpu.nvnni.prefill.two_row_pair_grid.nbc1.full_k";
        case PrefillSchedulePolicy::FourRowGrid:
            return "cpu.nvnni.prefill.four_row_grid.nbc1.full_k";
        case PrefillSchedulePolicy::Auto:
            throw std::logic_error("Prefill policy failed to resolve Auto");
        }
        throw std::logic_error("Unknown resolved CPU prefill schedule");
    }
} // namespace llaminar2::test::trainer
