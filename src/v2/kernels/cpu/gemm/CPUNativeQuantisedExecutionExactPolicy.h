/**
 * @file CPUNativeQuantisedExecutionExactPolicy.h
 * @brief Complete scalar identity for measured fused CPU expert row policies.
 *
 * Exact observations precede the existing generic selector only in their
 * admitted execution class. Source labels and descriptor vectors are evidence
 * provenance; pointers, payloads and lossy packed hashes never enter this key.
 */
#pragma once

#include <array>
#include <cstdint>
#include "kernels/common/MoEProjectionNumericalContract.h"

namespace llaminar2::cpu::native_quantised
{
    /** @brief Actual typed caller phase, never inferred from expert row count. */
    enum class CPUExecutionExactOperation : std::uint32_t
    {
        GPUAlignedExpertFusedDecode = 0,
        GPUAlignedExpertFusedMTPVerifier = 1,
        GPUAlignedExpertFusedPrefill = 2,
    };
    /** @brief Independent compiled and runtime CPU algorithm regimes. */
    enum class CPUExecutionExactISA : std::uint32_t { AVX2 = 0, AVX512 = 1 };
    /** @brief Output layout proved from immutable admitted addresses/strides. */
    enum class CPUExecutionExactLayout : std::uint32_t
    {
        Other = 0,
        ContiguousFP32Aligned64 = 1,
    };
    /** @brief Real call entry; an existing region does not assert its lifetime. */
    enum class CPUExecutionExactTeamEntry : std::uint32_t { Standalone = 0, ExistingRegion = 1 };
    /** @brief Policy-invariant ownership class admitted by the fused planner. */
    enum class CPUExecutionExactBundleClass : std::uint32_t
    {
        None = 0,
        FilledKPartOutputTileLocal = 1,
    };
    /** @brief Existing byte-equivalent physical rows; no arithmetic-policy choice. */
    enum class CPUExecutionExactChoice : std::uint32_t { Pairwise = 0, WideRows = 1 };

    /**
     * @brief ABI v1 full-width key shared by runtime and the canonical emitter.
     *
     * A prepared matrix preserves its source codebook/superblock even when its
     * physical encoding changes. Both identities and the numerical contract
     * therefore participate. Every integer is an actual extent, not a bucket.
     */
    struct CPUExecutionExactKey
    {
        CPUExecutionExactOperation operation{};
        CPUProjectionNumericalPolicy numerical_policy{};
        std::uint32_t source_codebook = 0;
        std::uint32_t source_superblock = 0;
        std::uint32_t prepared_encoding = 0;
        std::uint32_t asymmetric = 0;
        CPUExecutionExactISA build_isa{};
        CPUExecutionExactISA runtime_isa{};
        std::uint32_t workers = 0;
        std::uint32_t rows = 0;
        std::uint32_t physical_n = 0;
        std::uint32_t policy_n = 0;
        std::uint32_t k = 0;
        std::uint32_t k_tiles = 0;
        std::uint32_t ambient_n_block_chunks = 0;
        CPUExecutionExactLayout layout{};
        CPUExecutionExactTeamEntry team_entry{};
        CPUExecutionExactBundleClass bundle_class{};

        /** @return Exact ordered scalar ABI; no field is truncated or hashed. */
        [[nodiscard]] constexpr std::array<std::uint32_t, 18> scalars() const noexcept
        {
            return {static_cast<std::uint32_t>(operation), static_cast<std::uint32_t>(numerical_policy),
                source_codebook, source_superblock, prepared_encoding, asymmetric,
                static_cast<std::uint32_t>(build_isa), static_cast<std::uint32_t>(runtime_isa),
                workers, rows, physical_n, policy_n, k, k_tiles, ambient_n_block_chunks,
                static_cast<std::uint32_t>(layout), static_cast<std::uint32_t>(team_entry),
                static_cast<std::uint32_t>(bundle_class)};
        }
        /** @brief Compare the entire typed execution identity. */
        bool operator==(const CPUExecutionExactKey &) const = default;
    };
}
