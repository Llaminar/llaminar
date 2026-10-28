/**
 * @file MTPVerifierGraphFamily.h
 * @brief One immutable verifier-family authority for capture and physical admission.
 *
 * Hosted scheduler tickets name exact pre-captured complete transactions. Their
 * verifier widths therefore cover every admitted depth. Native conditional and
 * standalone execution retain their existing bounded family; mutable active-row
 * state never changes this setup-owned inventory. Ordinary M1 is owned elsewhere.
 */
#pragma once
#include <bit>
#include <cstddef>
#include <stdexcept>

namespace llaminar2
{
/** @brief Declared execution ownership, independent of backend names. */
enum class MTPVerifierExecutionRole
{
    Standalone,
    NativeConditionalParent,
    HostedCapturedTransactions,
};

/** @brief Complete-parent owner whose retained native transaction family is admitted. */
enum class MTPGenerationParentRole
{
    Controller,
    PipelineFollower,
};

/** @brief Immutable model-forward identities admitted before graph capture. */
enum class MTPVerifierGraphFamily
{
    BoundedBuckets,
    ExactRows,
};

/**
 * @brief Project declared execution ownership into its retained verifier family.
 * @param role Complete parent or standalone execution ownership.
 * @return The sole family permitted for that role.
 * @throws std::invalid_argument for an unknown role; admission has no fallback.
 */
[[nodiscard]] constexpr MTPVerifierGraphFamily resolveMTPVerifierGraphFamily(
    MTPVerifierExecutionRole role)
{
    switch (role)
    {
    case MTPVerifierExecutionRole::Standalone:
    case MTPVerifierExecutionRole::NativeConditionalParent:
        return MTPVerifierGraphFamily::BoundedBuckets;
    case MTPVerifierExecutionRole::HostedCapturedTransactions:
        return MTPVerifierGraphFamily::ExactRows;
    }
    throw std::invalid_argument("MTP verifier execution role is unknown");
}

/**
 * @brief Count grouped-verifier widths without counting ordinary M1 twice.
 * @param maximum_rows Admitted per-request verifier capacity.
 * @param family Immutable family selected from declared execution ownership.
 * @return Zero below two rows, otherwise the complete number of identities.
 * @throws std::invalid_argument for an unknown family.
 */
[[nodiscard]] constexpr std::size_t mtpVerifierGraphFamilyWidthCount(
    int maximum_rows, MTPVerifierGraphFamily family)
{
    switch (family)
    {
    case MTPVerifierGraphFamily::BoundedBuckets:
        return maximum_rows < 2 ? 0u :
            std::bit_width(static_cast<unsigned>(maximum_rows - 1));
    case MTPVerifierGraphFamily::ExactRows:
        return maximum_rows < 2 ? 0u : static_cast<std::size_t>(maximum_rows - 1);
    }
    throw std::invalid_argument("MTP verifier graph family is unknown");
}

/**
 * @brief Enumerate one family largest first, freezing shared storage at capacity.
 * @param index Zero-based largest-first member index.
 * @param maximum_rows Admitted per-request verifier capacity.
 * @param family Immutable admitted family.
 * @return Exact physical width, or zero for an index outside the family.
 * @throws std::invalid_argument for an unknown family.
 */
[[nodiscard]] constexpr int mtpVerifierGraphFamilyWidthAt(
    std::size_t index, int maximum_rows, MTPVerifierGraphFamily family)
{
    if (index >= mtpVerifierGraphFamilyWidthCount(maximum_rows, family)) return 0;
    if (family == MTPVerifierGraphFamily::ExactRows)
        return maximum_rows - static_cast<int>(index);
    return index == 0 ? maximum_rows : static_cast<int>(
        std::bit_floor(static_cast<unsigned>(maximum_rows - 1)) >> (index - 1));
}

/**
 * @brief Authenticate a physical graph width against its admitted family.
 * @param rows Candidate physical grouped-verifier width.
 * @param maximum_rows Admitted per-request capacity.
 * @param family Immutable family selected before allocation.
 * @return Whether the exact identity is present; unknown families are rejected.
 */
[[nodiscard]] constexpr bool mtpVerifierGraphFamilyContains(
    int rows, int maximum_rows, MTPVerifierGraphFamily family) noexcept
{
    if (rows < 2 || rows > maximum_rows) return false;
    switch (family)
    {
    case MTPVerifierGraphFamily::BoundedBuckets:
        return rows == maximum_rows || std::has_single_bit(static_cast<unsigned>(rows));
    case MTPVerifierGraphFamily::ExactRows:
        return true;
    }
    return false;
}
}
