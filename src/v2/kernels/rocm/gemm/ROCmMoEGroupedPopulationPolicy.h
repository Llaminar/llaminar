/**
 * @file ROCmMoEGroupedPopulationPolicy.h
 * @brief Immutable grouped-operation admission and device-owned dot-family selection.
 *
 * Capture binds an exact operation key; replay reduces the authoritative expert
 * counts into one arena-owned decision. Every projection borrows that decision,
 * so gate/up and down cannot independently choose writers. Route-owned admission
 * precedes this policy. These integer decisions change scheduling, never source
 * partitions, quantization or floating-point reduction order.
 */
#pragma once
#include "kernels/common/MoEPrefillProjectionExecution.h"
#include "kernels/common/DeviceNativeQuantisedMatrixDesc.h"
#include "tensors/NativeQuantisedFormatInfo.h"
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <span>
#include <type_traits>

#if defined(__HIPCC__)
#define LLAMINAR_POPULATION_HD __host__ __device__
#else
#define LLAMINAR_POPULATION_HD
#endif

namespace llaminar2::rocm
{
/** @brief Device publication values; all other bit patterns are invalid. */
enum class MoEGroupedPopulationFamily : uint32_t { Dot4, N32, WideFloor16, Invalid };
/** @brief Exact immutable geometry families with separately measured economics. */
enum class MoEGroupedPopulationDomain : uint8_t { Unmeasured, Qwen, Compact };
/** @brief Activation addressing and publication are independent admission facts. */
enum class MoEGroupedPopulationInput : uint8_t { OriginalTokens, OriginalRouteSlots };
/** @brief Quantization ownership retained by the complete operation. */
enum class MoEGroupedPopulationPreparation : uint8_t { Internal, PublishedQ8 };

/** @brief Original immutable singleton sources, separate from normalized arithmetic masks. */
struct MoEGroupedSourceTuple
{
    NativeQuantisedSourceIdentity gate{}, up{}, down{};
};

/** @brief Complete immutable key; exact singleton sources preserve arithmetic-mask aliases. */
struct MoEGroupedPopulationKey
{
    MoEPrefillMeasuredOperation operation = MoEPrefillMeasuredOperation::Unmeasured;
    MoEGroupedPopulationInput input = MoEGroupedPopulationInput::OriginalTokens;
    MoEGroupedPopulationPreparation preparation = MoEGroupedPopulationPreparation::Internal;
    int rows = 0, hidden = 0, intermediate = 0, experts = 0, top_k = 0;
    uint32_t gateup_physical = 0, down_physical = 0, gateup_source = 0, down_source = 0;
    MoEGroupedSourceTuple singleton_sources{};
    MoEGroupedSourceTuple routed_sources{}; ///< All original descriptors authenticated before static grouping.
};

/** @brief Authenticate every member of the bounded original routed descriptor tables.
 * @param gate Original gate descriptors. @param up Original up descriptors.
 * @param down Original down descriptors.
 * @return Exact uniform sources, or an empty identity outside the measured domain.
 * Mask equality cannot prove these identities: promoted payloads and arithmetic
 * aliases deliberately retain different original source metadata. */
[[nodiscard]] inline MoEGroupedSourceTuple routedVerifierOriginalSources(
    std::span<const DeviceNativeQuantisedMatrixDesc> gate,
    std::span<const DeviceNativeQuantisedMatrixDesc> up,
    std::span<const DeviceNativeQuantisedMatrixDesc> down) noexcept
{
    if (gate.size() != 256 || up.size() != 256 || down.size() != 256) return {};
    const auto matches = [](const DeviceNativeQuantisedMatrixDesc &d, int n, int k,
                            unsigned physical, unsigned source) {
        return d.valid() && d.mins && d.n == n && d.k == k && d.blocks_per_row == unsigned(k / 32) &&
            d.codebook_id == physical && d.source_identity_present == 1 &&
            d.source_is_superblock == 1 && d.source_codebook_id == source;
    };
    for (std::size_t e = 0; e < gate.size(); ++e)
        if (!matches(gate[e], 512, 2048, 13, 13) || !matches(up[e], 512, 2048, 13, 13) ||
            !matches(down[e], 2048, 512, 24, 4)) return {};
    return {{13, true, true}, {13, true, true}, {4, true, true}};
}

/** @param key Complete original static transaction, resolved before grouping.
 * @return Whether the measured column-directory producer is qualified.
 * Device counts choose cold Dot4 versus hot WMMA inside its shared consumers;
 * the host admits only this immutable physical work size and source tuple. */
[[nodiscard]] constexpr bool routedVerifierColumnQualified(const MoEGroupedPopulationKey &key) noexcept
{
    return key.operation == MoEPrefillMeasuredOperation::CompleteFullWidthTokenRowsAllRoutes &&
        key.input == MoEGroupedPopulationInput::OriginalTokens &&
        key.preparation == MoEGroupedPopulationPreparation::PublishedQ8 &&
        key.rows == 16 && key.hidden == 2048 && key.intermediate == 512 && key.experts == 256 && key.top_k == 8 &&
        key.gateup_physical == (uint32_t{1} << 13) && key.down_physical == (uint32_t{1} << 24) &&
        key.gateup_source == (uint32_t{1} << 13) && key.down_source == (uint32_t{1} << 4) &&
        key.routed_sources.gate == NativeQuantisedSourceIdentity{13, true, true} &&
        key.routed_sources.up == NativeQuantisedSourceIdentity{13, true, true} &&
        key.routed_sources.down == NativeQuantisedSourceIdentity{4, true, true};
}

/** @brief Admit the measured complete shared-verifier transaction without reading live rows.
 * @param key Original operation, source tuple, geometry and activation ownership.
 * @return Whether the existing N32 producer is qualified for this immutable key.
 * Source masks encode arithmetic and may merge aliases; exact original sources
 * and canonical physical masks must agree independently. Existing RouteOwned
 * admission still selects the dot producer for live prefixes through eight. */
[[nodiscard]] constexpr bool sharedVerifierWmmaQualified(const MoEGroupedPopulationKey &key) noexcept
{
    static_assert(native_quantised_formats::kAllSourceFormats.size() == 21,
        "Requalify shared verifier WMMA before extending the source inventory");
    if (key.operation != MoEPrefillMeasuredOperation::CompleteFullWidthSharedVerifierTokenRows ||
        key.input != MoEGroupedPopulationInput::OriginalTokens ||
        key.preparation != MoEGroupedPopulationPreparation::PublishedQ8 ||
        key.hidden != 2048 || key.intermediate != 512 || key.experts != 1 || key.top_k != 1)
        return false;
    const auto &sources = key.singleton_sources;
    if (sources.gate != sources.up) return false;
    const auto gu = canonicalDeviceVnniCodebookId(sources.gate);
    const auto down = canonicalDeviceVnniCodebookId(sources.down);
    if (gu >= 32 || down >= 32 || key.gateup_physical != (uint32_t{1} << gu) ||
        key.down_physical != (uint32_t{1} << down) ||
        key.gateup_source != (uint32_t{1} << canonicalDeviceVnniCodebookId(sources.gate.codebook_id)) ||
        key.down_source != (uint32_t{1} << canonicalDeviceVnniCodebookId(sources.down.codebook_id)))
        return false;
    if (key.rows == 16 && sources.gate == sources.down) return true;
    return (key.rows == 9 || key.rows == 16) &&
        sources.gate == NativeQuantisedSourceIdentity{13, true, true} &&
        sources.down == NativeQuantisedSourceIdentity{4, true, true};
}

/** @param key Actual validated capture operation. @return Recognized candidate domain; the native selector separately admits qualified implementations. */
[[nodiscard]] constexpr MoEGroupedPopulationDomain groupedPopulationDomain(
    const MoEGroupedPopulationKey &key) noexcept
{
    if (key.operation != MoEPrefillMeasuredOperation::CompleteFullWidthTokenRowsAllRoutes ||
        key.input != MoEGroupedPopulationInput::OriginalTokens ||
        key.preparation != MoEGroupedPopulationPreparation::PublishedQ8 ||
        key.rows != 512 || key.top_k != 8)
        return MoEGroupedPopulationDomain::Unmeasured;
    if (key.hidden == 2048 && key.intermediate == 512 && key.experts == 256 &&
        key.gateup_physical == (uint32_t{1} << 13) && key.gateup_source == (uint32_t{1} << 13) &&
        key.down_physical == (uint32_t{1} << 24) && key.down_source == (uint32_t{1} << 4))
        return MoEGroupedPopulationDomain::Qwen;
    if (key.hidden == 512 && key.intermediate == 256 && key.experts == 8 &&
        (key.gateup_physical == 1u || key.gateup_physical == (uint32_t{1} << 13)) &&
        key.down_physical == key.gateup_physical && key.gateup_source == key.gateup_physical &&
        key.down_source == key.gateup_physical)
        return MoEGroupedPopulationDomain::Compact;
    return MoEGroupedPopulationDomain::Unmeasured;
}

/** @brief Non-null borrowed device publication; copying it never copies the live decision. */
class MoEGroupedPopulationBinding final
{
public:
    /** @param publication Canonical arena-owned word, retained through graph retirement.
     * @throws std::invalid_argument For missing publication storage. */
    explicit MoEGroupedPopulationBinding(const uint32_t *publication) : publication_(publication)
    { if (!publication) throw std::invalid_argument("population binding requires device publication storage"); }
    /** @return Immutable capture binding; this does not read device state. */
    [[nodiscard]] LLAMINAR_POPULATION_HD constexpr const uint32_t *address() const noexcept { return publication_; }
#if defined(__HIPCC__)
    /** @return The current device-owned family, trapping on uninitialized or malformed data. */
    [[nodiscard]] __device__ __forceinline__ MoEGroupedPopulationFamily load() const
    {
        const auto value = __builtin_amdgcn_readfirstlane(*publication_);
        if (value >= static_cast<uint32_t>(MoEGroupedPopulationFamily::Invalid)) __builtin_trap();
        return static_cast<MoEGroupedPopulationFamily>(value);
    }
#endif
private:
    const uint32_t *publication_;
};
static_assert(std::is_trivially_copyable_v<MoEGroupedPopulationBinding>);
}
#undef LLAMINAR_POPULATION_HD
