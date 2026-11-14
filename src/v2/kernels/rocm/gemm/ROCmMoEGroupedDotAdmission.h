/**
 * @file ROCmMoEGroupedDotAdmission.h
 * @brief Exclusive per-expert projection ownership inside the tiled MoE family.
 *
 * Route admission still selects RouteOwned versus ExpertTiled. This separate
 * captured value chooses the dot producer for each complete expert projection.
 * Both tile families read the same published whole-expert count, so different
 * row-tile boundaries cannot create overlapping writers or leave output holes.
 */
#pragma once
#include "kernels/common/DeviceNativeQuantisedMatrixDesc.h"
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#if defined(__HIPCC__)
#define LLAMINAR_MOE_DOT_HD __host__ __device__
#else
#define LLAMINAR_MOE_DOT_HD
#endif

namespace llaminar2::rocm
{
/** @brief Projection role shared by directory, tile and dot-engine policies. */
enum class MoEPrefillProjectionRole : uint8_t { GateUp, Down };

/** @brief Explicit experimental whole-expert row floors; none is an installed policy. */
enum class MoEGroupedWmmaRowFloor : uint8_t
{
    EveryExpert = 0, Four = 4, Eight = 8, Twelve = 12, Sixteen = 16
};

/** @param floor Proposed recording choice. @return Whether it names a supported experiment. */
[[nodiscard]] constexpr bool validMoEGroupedWmmaRowFloor(MoEGroupedWmmaRowFloor floor) noexcept
{
    switch (floor)
    {
    case MoEGroupedWmmaRowFloor::EveryExpert:
    case MoEGroupedWmmaRowFloor::Four:
    case MoEGroupedWmmaRowFloor::Eight:
    case MoEGroupedWmmaRowFloor::Twelve:
    case MoEGroupedWmmaRowFloor::Sixteen: return true;
    }
    return false;
}

/** @brief One writer family, or a malformed descriptor rejected by both families. */
enum class MoEGroupedDotProducer : uint8_t { OrderedDot4, Wmma, Invalid };

/** @brief Immutable projection identity and one shared device admission predicate. */
class MoEGroupedDotAdmission final
{
public:
    /** @brief Default admission retains the existing ordered implementation. */
    constexpr MoEGroupedDotAdmission() noexcept = default;

    /**
     * @brief Bind a native candidate to its complete projection and source identity.
     * @param role Gate/up or down; both native and vector leaves borrow this value.
     * @param source_n Original unsliced output width owning arithmetic policy.
     * @param physical_n Prepared projection width, including compact down slices.
     * @param k Shared contraction width, divisible by 32.
     * @param floor Whole-expert count needed for the candidate; zero forces native.
     * @param physical_codebooks Physical decoder identities eligible for native work.
     * @param source_codebooks Source identities eligible for native work.
     * @return A trivially copyable, capture-stable admission value.
     * @throws std::invalid_argument For incomplete geometry, masks or an unknown floor.
     */
    static MoEGroupedDotAdmission nativeCandidate(
        MoEPrefillProjectionRole role, int source_n, int physical_n, int k,
        MoEGroupedWmmaRowFloor floor, uint32_t physical_codebooks = UINT32_MAX,
        uint32_t source_codebooks = UINT32_MAX)
    {
        if ((role != MoEPrefillProjectionRole::GateUp && role != MoEPrefillProjectionRole::Down) ||
            source_n <= 0 || physical_n <= 0 || physical_n > source_n || k <= 0 || k % 32 ||
            (role == MoEPrefillProjectionRole::GateUp && physical_n != source_n) ||
            !physical_codebooks || !source_codebooks)
            throw std::invalid_argument("incomplete grouped dot admission geometry or codebooks");
        if (!validMoEGroupedWmmaRowFloor(floor))
            throw std::invalid_argument("unknown grouped WMMA expert-row floor");
        MoEGroupedDotAdmission result;
        result.native_ = true;
        result.role_ = role;
        result.floor_ = floor;
        result.source_n_ = source_n;
        result.physical_n_ = physical_n;
        result.k_ = k;
        result.physical_codebooks_ = physical_codebooks;
        // These source aliases retain the canonical Q8_0 arithmetic identity.
        constexpr uint32_t aliases = (uint32_t{1} << 20) | (uint32_t{1} << 21);
        result.source_codebooks_ = source_codebooks & aliases
            ? (source_codebooks & ~aliases) | (uint32_t{1} << 19) : source_codebooks;
        return result;
    }

    /** @return Whether recording must include a native projection leaf. */
    [[nodiscard]] constexpr bool mayUseNative() const noexcept { return native_; }

    /** @param role Submitted role. @param n Prepared width. @param k Contraction width.
     * @return Whether a native launch borrows this exact immutable binding. */
    [[nodiscard]] constexpr bool binds(MoEPrefillProjectionRole role, int n, int k) const noexcept
    { return native_ && role_ == role && physical_n_ == n && k_ == k; }

    /** @return Original unsliced width retained for admission/evidence identity. */
    [[nodiscard]] constexpr int sourceOutputColumns() const noexcept { return source_n_; }

    /** @return Whether recording must retain ordered leaves for some experts. */
    [[nodiscard]] constexpr bool mayUseOrdered() const noexcept
    {
        constexpr uint32_t all_sources = UINT32_MAX & ~((uint32_t{1} << 20) | (uint32_t{1} << 21));
        return !native_ || floor_ != MoEGroupedWmmaRowFloor::EveryExpert ||
            physical_codebooks_ != UINT32_MAX || source_codebooks_ != all_sources;
    }

    /**
     * @brief Select the sole dot writer for an entire expert projection.
     * @param desc Actual descriptor, including promoted source provenance.
     * @param whole_expert_rows Authoritative group_counts[expert], never tile-local rows.
     * @return The one selected producer, or Invalid for malformed live metadata.
     */
    [[nodiscard]] LLAMINAR_MOE_DOT_HD constexpr MoEGroupedDotProducer select(
        const DeviceNativeQuantisedMatrixDesc &desc, int whole_expert_rows) const noexcept
    {
        if (!native_) return MoEGroupedDotProducer::OrderedDot4;
        uint8_t source = desc.source_identity_present ? desc.source_codebook_id : desc.codebook_id;
        if (source == 20 || source == 21) source = 19;
        if (whole_expert_rows <= 0 || desc.n != physical_n_ || desc.k != k_ ||
            desc.codebook_id >= 32 || source >= 32)
            return MoEGroupedDotProducer::Invalid;
        if (whole_expert_rows < static_cast<int>(floor_) ||
            !(physical_codebooks_ & (uint32_t{1} << desc.codebook_id)) ||
            !(source_codebooks_ & (uint32_t{1} << source)))
            return MoEGroupedDotProducer::OrderedDot4;
        return MoEGroupedDotProducer::Wmma;
    }

private:
    bool native_ = false;
    MoEPrefillProjectionRole role_ = MoEPrefillProjectionRole::GateUp;
    MoEGroupedWmmaRowFloor floor_ = MoEGroupedWmmaRowFloor::EveryExpert;
    int source_n_ = 0;
    int physical_n_ = 0;
    int k_ = 0;
    uint32_t physical_codebooks_ = 0;
    uint32_t source_codebooks_ = 0;
};

static_assert(std::is_trivially_copyable_v<MoEGroupedDotAdmission>);
static_assert(std::is_standard_layout_v<MoEGroupedDotAdmission>);
}
#undef LLAMINAR_MOE_DOT_HD
