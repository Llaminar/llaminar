/**
 * @file DeviceMoERuntimeABI.h
 * @brief Lightweight ABI contract for graph-resident MoE layer state.
 *
 * GPU translation units intentionally use a device-friendly view of
 * DeviceMoELayerRuntime instead of including MoERuntimeTable.h, whose host
 * tensor dependencies contain x86 SIMD code. These constants make that split
 * mechanical rather than conventional: the host definition and every CUDA/HIP
 * view must independently prove the same 64-bit size and critical offsets.
 */
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MOE_RUNTIME_HD __host__ __device__
#else
#define LLAMINAR_MOE_RUNTIME_HD
#endif

namespace llaminar2::moe_runtime_abi
{
    /** Number of independently retained runtime routing phases. */
    inline constexpr std::size_t kHistogramSourceCount = 3;

    /** Stable source ordinals used by host, CUDA, and ROCm histogram policy. */
    enum class HistogramSource : std::uint32_t
    {
        Decode = 0u,          ///< Ordinary serial decode rows.
        Prefill = 1u,         ///< Prompt-ingestion rows.
        GroupedVerifier = 2u, ///< Accepted grouped-verifier rows.
    };

    /** @return One policy-mask bit for a validated histogram source. */
    [[nodiscard]] LLAMINAR_MOE_RUNTIME_HD constexpr std::uint32_t
    histogramSourceBit(
        HistogramSource source) noexcept
    {
        return 1u << static_cast<std::uint32_t>(source);
    }

    /** Mask selecting every retained runtime routing phase. */
    inline constexpr std::uint32_t kAllHistogramSourcesMask =
        (1u << kHistogramSourceCount) - 1u;

    /** Low bit selecting one of the two persistent histogram banks. */
    inline constexpr std::uint32_t kHistogramWriterBankMask = 1u;
    /**
     * Bit that quarantines route rows while economy activation crosses a
     * request boundary. Raw states 0/1 intentionally remain admitted so
     * existing model-setup initialization is the economical fast path.
     */
    inline constexpr std::uint32_t kHistogramWriterQuarantineBit = 2u;
    /** Complete set of bits accepted by the host/device writer-state ABI. */
    inline constexpr std::uint32_t kHistogramWriterStateMask =
        kHistogramWriterBankMask | kHistogramWriterQuarantineBit;

    /** @return A complete writer state for one bank and admission decision. */
    [[nodiscard]] LLAMINAR_MOE_RUNTIME_HD constexpr std::uint32_t
    makeHistogramWriterState(
        std::uint32_t bank,
        bool admit_rows) noexcept
    {
        return (bank & kHistogramWriterBankMask) |
               (admit_rows ? 0u : kHistogramWriterQuarantineBit);
    }

    /** @return Whether a device-published writer state is structurally valid. */
    [[nodiscard]] LLAMINAR_MOE_RUNTIME_HD constexpr bool
    validHistogramWriterState(
        std::uint32_t state) noexcept
    {
        return (state & ~kHistogramWriterStateMask) == 0u;
    }

    /** @return Physical bank selected by a validated writer state. */
    [[nodiscard]] LLAMINAR_MOE_RUNTIME_HD constexpr std::uint32_t
    histogramWriterBank(
        std::uint32_t state) noexcept
    {
        return state & kHistogramWriterBankMask;
    }

    /** @return Whether a validated writer state admits route rows. */
    [[nodiscard]] LLAMINAR_MOE_RUNTIME_HD constexpr bool
    histogramWriterAdmitsRows(
        std::uint32_t state) noexcept
    {
        return (state & kHistogramWriterQuarantineBit) == 0u;
    }

    /** @return Whether a policy mask selects at least one known source only. */
    [[nodiscard]] LLAMINAR_MOE_RUNTIME_HD constexpr bool
    validHistogramSourceMask(
        std::uint32_t mask) noexcept
    {
        return mask != 0u &&
               (mask & ~kAllHistogramSourcesMask) == 0u;
    }

    /** Maximum expert geometry shared by the host and GPU runtime views. */
    inline constexpr std::uint32_t kMaxExperts = 512u;

    /** ABI generation for graph-resident records; old captured addresses are incompatible. */
    inline constexpr std::uint32_t kRuntimeLayoutVersion = 2u;

    /**
     * @brief Integer grouping assigns one lane to each live expert.
     * @param experts Positive admitted expert count, at most kMaxExperts.
     * @return A complete 256/512-lane block, or zero for invalid geometry.
     */
    [[nodiscard]] LLAMINAR_MOE_RUNTIME_HD constexpr std::uint32_t
    expertGroupingThreads(std::uint32_t experts) noexcept
    {
        return experts == 0u || experts > kMaxExperts ? 0u
             : experts <= 256u ? 256u : 512u;
    }

    /**
     * @brief Keep ROCm serial and grouped routing on the same reduction tree.
     * @param experts Positive admitted expert count, at most kMaxExperts.
     * @return Populated wave64 lanes, capped at the canonical 256 partitions.
     *
     * Above 256 experts each lane accumulates its stride-256 terms. Expanding
     * the block instead would change arithmetic order relative to grouped rows.
     */
    [[nodiscard]] LLAMINAR_MOE_RUNTIME_HD constexpr std::uint32_t
    wave64RouterThreads(std::uint32_t experts) noexcept
    {
        return experts == 0u || experts > kMaxExperts ? 0u
             : experts > 256u ? 256u : ((experts + 63u) / 64u) * 64u;
    }

    /** Histogram storage uses the same logical expert ceiling as placement. */
    inline constexpr std::size_t kHistogramMaxExperts = kMaxExperts;

    /**
     * @brief One persistent, device-owned routing-evidence generation.
     *
     * The first dimension is decode, ordinary prefill, then grouped verifier.
     * Selected and locally executed routes remain separate so placement policy
     * and execution diagnostics can consume the same immutable generation.
     * Two model-lifetime instances are allocated outside request-reset state;
     * inference writes one while maintenance drains and clears the other.
     */
    struct DeviceMoERuntimeHistogramBank
    {
        std::uint64_t selected[kHistogramSourceCount]
                              [kHistogramMaxExperts] = {};
        std::uint64_t local[kHistogramSourceCount]
                           [kHistogramMaxExperts] = {};
    };

    inline constexpr std::size_t kHistogramBankBytes =
        2 * kHistogramSourceCount * kHistogramMaxExperts *
        sizeof(std::uint64_t);

    static_assert(sizeof(DeviceMoERuntimeHistogramBank) ==
                  kHistogramBankBytes);

    /*
     * Each of the two 512-expert placement banks owns a complete union-like
     * descriptor record: three NativeVNNI views plus three compact contiguous
     * floating views and one format discriminator.  Both families retain
     * stable addresses because a captured runtime table may be republished by
     * asynchronous expert movement without changing its graph arguments.
     * Keeping these reviewed constants beside that rationale makes any future
     * descriptor expansion fail every host/device ABI assertion together.
     */
    /** Bytes in one complete 512-expert placement generation. */
    inline constexpr std::size_t kPlacementBankBytes = 128016;
    /** Offset of the overlay-wide sparse-packet target array inside a bank. */
    inline constexpr std::size_t kOverlayRouteParticipantOffset = 125952;

    inline constexpr std::size_t kLayerRuntimeBytes = 281064;
    /** First device-local decode execution counter in the inline ABI. */
    inline constexpr std::size_t kDecodeLocalHistogramOffset = 260272;
    /** First device-local ordinary-prefill execution counter. */
    inline constexpr std::size_t kPrefillLocalHistogramOffset = 268464;
    /** First device-local grouped-verifier execution counter. */
    inline constexpr std::size_t kGroupedVerifierLocalHistogramOffset = 276656;
    inline constexpr std::size_t kRouteParticipantIdsOffset = 280848;
    inline constexpr std::size_t kDeferredVerifierExpertIdsOffset = 280856;
    inline constexpr std::size_t kDeferredVerifierParticipantIdsOffset = 280864;
    inline constexpr std::size_t kExpertCountsOffset = 280872;
    inline constexpr std::size_t kDeferredVerifierRouteCapacityOffset = 281000;
    inline constexpr std::size_t kParticipantIdOffset = 281004;
    inline constexpr std::size_t kParticipantCountOffset = 281008;
    inline constexpr std::size_t kCurrentBatchLLEPMovementObservedOffset = 281012;
    inline constexpr std::size_t
        kCurrentBatchLLEPNonOwnerAssignmentObservedOffset = 281016;
    inline constexpr std::size_t
        kCurrentBatchLLEPTransientBankActiveOffset = 281020;
    inline constexpr std::size_t kRuntimeHistogramBanksOffset = 281024;
    inline constexpr std::size_t kRuntimeHistogramActiveBankOffset = 281032;
    inline constexpr std::size_t kOverlayEpochTicketOffset = 281040;
    inline constexpr std::size_t kOverlayPlacementBanksOffset = 281048;
    /** Failure-only provenance for the last device epoch-boundary operation. */
    inline constexpr std::size_t kOverlayEpochStatusOffset = 281056;

    static_assert(sizeof(void *) == 8,
                  "DeviceMoELayerRuntime ABI requires 64-bit pointers");
} // namespace llaminar2::moe_runtime_abi

#undef LLAMINAR_MOE_RUNTIME_HD
