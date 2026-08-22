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
    [[nodiscard]] constexpr std::uint32_t histogramSourceBit(
        HistogramSource source) noexcept
    {
        return 1u << static_cast<std::uint32_t>(source);
    }

    /** Mask selecting every retained runtime routing phase. */
    inline constexpr std::uint32_t kAllHistogramSourcesMask =
        (1u << kHistogramSourceCount) - 1u;

    /** @return Whether a policy mask selects at least one known source only. */
    [[nodiscard]] constexpr bool validHistogramSourceMask(
        std::uint32_t mask) noexcept
    {
        return mask != 0u &&
               (mask & ~kAllHistogramSourcesMask) == 0u;
    }

    /** Maximum expert geometry shared by the host and GPU runtime views. */
    inline constexpr std::size_t kHistogramMaxExperts = 256;

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
     * Each of the two 256-expert placement banks owns a complete union-like
     * descriptor record: three NativeVNNI views plus three compact contiguous
     * floating views and one format discriminator.  Both families retain
     * stable addresses because a captured runtime table may be republished by
     * asynchronous expert movement without changing its graph arguments.
     * Keeping these reviewed constants beside that rationale makes any future
     * descriptor expansion fail every host/device ABI assertion together.
     */
    /** Bytes in one complete 256-expert placement generation. */
    inline constexpr std::size_t kPlacementBankBytes = 64016;
    /** Offset of the overlay-wide sparse-packet target array inside a bank. */
    inline constexpr std::size_t kOverlayRouteParticipantOffset = 62976;

    inline constexpr std::size_t kLayerRuntimeBytes = 140768;
    /** First device-local decode execution counter in the inline ABI. */
    inline constexpr std::size_t kDecodeLocalHistogramOffset = 130224;
    /** First device-local ordinary-prefill execution counter. */
    inline constexpr std::size_t kPrefillLocalHistogramOffset = 134320;
    /** First device-local grouped-verifier execution counter. */
    inline constexpr std::size_t kGroupedVerifierLocalHistogramOffset = 138416;
    inline constexpr std::size_t kRouteParticipantIdsOffset = 140560;
    inline constexpr std::size_t kDeferredVerifierExpertIdsOffset = 140568;
    inline constexpr std::size_t kDeferredVerifierParticipantIdsOffset = 140576;
    inline constexpr std::size_t kExpertCountsOffset = 140584;
    inline constexpr std::size_t kDeferredVerifierRouteCapacityOffset = 140712;
    inline constexpr std::size_t kParticipantIdOffset = 140716;
    inline constexpr std::size_t kParticipantCountOffset = 140720;
    inline constexpr std::size_t kCurrentBatchLLEPMovementObservedOffset = 140724;
    inline constexpr std::size_t
        kCurrentBatchLLEPNonOwnerAssignmentObservedOffset = 140728;
    inline constexpr std::size_t
        kCurrentBatchLLEPTransientBankActiveOffset = 140732;
    inline constexpr std::size_t kRuntimeHistogramBanksOffset = 140736;
    inline constexpr std::size_t kRuntimeHistogramActiveBankOffset = 140744;
    inline constexpr std::size_t kOverlayEpochTicketOffset = 140752;
    inline constexpr std::size_t kOverlayPlacementBanksOffset = 140760;

    static_assert(sizeof(void *) == 8,
                  "DeviceMoELayerRuntime ABI requires 64-bit pointers");
} // namespace llaminar2::moe_runtime_abi
