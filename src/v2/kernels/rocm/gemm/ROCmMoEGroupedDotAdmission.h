/**
 * @file ROCmMoEGroupedDotAdmission.h
 * @brief Exclusive per-expert projection ownership inside the tiled MoE family.
 *
 * Route admission still selects RouteOwned versus ExpertTiled. This separate
 * captured value chooses the dot producer for each complete expert projection.
 * The adaptive native consumer reads the published whole-expert count once,
 * so its final partial row tile cannot select a different arithmetic branch.
 * Wide consumers and their directory builder borrow the same immutable row
 * plan. Shared gate/up planning admits every supported descriptor together;
 * independently filtered roles cannot reinterpret one another's row offsets.
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

/** @brief One arithmetic branch, or a malformed descriptor rejected before consumption. */
enum class MoEGroupedDotProducer : uint8_t { OrderedDot4, Wmma, Invalid };

/** @brief Complete immutable directory identity, including count-dependent row steps. */
class MoEGroupedRowPlan final
{
public:
    /** @brief Default plan names the ordinary sixteen-row native directory. */
    constexpr MoEGroupedRowPlan() noexcept = default;

    /** @param rows Fixed rows per directory entry. @return A validated fixed plan.
     * @throws std::invalid_argument For an unsupported row tile. */
    static constexpr MoEGroupedRowPlan fixed(int rows)
    {
        if (rows != 4 && rows != 8 && rows != 12 && rows != 16)
            throw std::invalid_argument("unsupported grouped directory row step");
        MoEGroupedRowPlan result;
        result.rows_and_kind_ = static_cast<uint8_t>(rows);
        return result;
    }

    /** @param role Projection role. @param floor Arithmetic cutoff; rows remain sixteen for both branches.
     * @return A fixed native microtile plan owning the supplied cutoff. */
    static constexpr MoEGroupedRowPlan nativeMicro(MoEPrefillProjectionRole role, MoEGroupedWmmaRowFloor floor)
    {
        if (!validMoEGroupedWmmaRowFloor(floor))
            throw std::invalid_argument("unknown grouped WMMA expert-row floor");
        if (role != MoEPrefillProjectionRole::GateUp && role != MoEPrefillProjectionRole::Down)
            throw std::invalid_argument("unknown grouped projection role");
        MoEGroupedRowPlan result;
        result.role_ = role;
        result.floor_ = floor;
        return result;
    }

    /** @param role Shared gate/up or down projection. @param floor Whole-expert cutoff.
     * @return The complete count-only plan for the explicit wide candidate.
     * @throws std::invalid_argument For an unknown role or cutoff. */
    static constexpr MoEGroupedRowPlan wide(MoEPrefillProjectionRole role, MoEGroupedWmmaRowFloor floor)
    {
        auto result = nativeMicro(role,floor);
        result.rows_and_kind_ = 0x80u | (role == MoEPrefillProjectionRole::GateUp ? 12u : 16u);
        return result;
    }

    /** @param role Original shared GU or down role.
     * @return Bounded column-directory plan, retaining original GU4/down12 cold arithmetic. */
    static constexpr MoEGroupedRowPlan column(MoEPrefillProjectionRole role)
    {
        auto result=nativeMicro(role,MoEGroupedWmmaRowFloor::Eight);
        result.rows_and_kind_=0xc0u | (role==MoEPrefillProjectionRole::GateUp ? 4u : 12u);
        return result;
    }
    /** @return Whether compact directory words also encode native column ownership. */
    [[nodiscard]] LLAMINAR_MOE_DOT_HD constexpr bool isColumn() const noexcept { return rows_and_kind_ & 0x40u; }
    /** @return Whether this plan belongs to a wide Dot4/native macro tile. */
    [[nodiscard]] LLAMINAR_MOE_DOT_HD constexpr bool isWide() const noexcept { return rows_and_kind_ & 0x80u; }
    /** @return The projection role sealed with its row policy. */
    [[nodiscard]] LLAMINAR_MOE_DOT_HD constexpr MoEPrefillProjectionRole role() const noexcept { return role_; }
    /** @return The sole whole-expert threshold used by both builder and consumer. */
    [[nodiscard]] LLAMINAR_MOE_DOT_HD constexpr MoEGroupedWmmaRowFloor floor() const noexcept { return floor_; }
    /** @return Smallest reachable row step, for a conservative persistent capacity. */
    [[nodiscard]] constexpr int minimumRows() const noexcept
    { return isWide() && floor_ == MoEGroupedWmmaRowFloor::EveryExpert ? 16 : rows_and_kind_ & 0x1fu; }
    /** @param count Whole-expert count. @return Its row step, or zero for invalid/empty work. */
    [[nodiscard]] LLAMINAR_MOE_DOT_HD constexpr int rows(int count) const noexcept
    {
        return count <= 0 ? 0 : isWide() && count >= static_cast<int>(floor_) ? 16 : rows_and_kind_ & 0x1fu;
    }
    /** @param other Another recording plan. @return Exact directory reuse compatibility. */
    [[nodiscard]] constexpr bool operator==(const MoEGroupedRowPlan &other) const noexcept = default;

private:
    MoEPrefillProjectionRole role_ = MoEPrefillProjectionRole::GateUp;
    MoEGroupedWmmaRowFloor floor_ = MoEGroupedWmmaRowFloor::EveryExpert;
    // One private byte retains the row step, wide kind and column-ownership
    // bit. Keeping all three in the row plan prevents directory aliasing. This keeps the
    // complete admission value in its original capture-packet footprint.
    uint8_t rows_and_kind_ = 16;
};

/** @brief Immutable projection identity and one shared device admission predicate. */
class MoEGroupedDotAdmission final
{
public:
    /** @brief Default admission retains the existing ordered implementation. */
    constexpr MoEGroupedDotAdmission() noexcept = default;

    /**
     * @brief Bind a native candidate to its complete projection and source identity.
     * @param role Gate/up or down; both arithmetic branches borrow this value.
     * @param source_n Original unsliced output width owning arithmetic policy.
     * @param physical_n Prepared projection width, including compact down slices.
     * @param k Shared contraction width, divisible by 32.
     * @param floor Whole-expert count needed for the candidate; zero removes the row cutoff.
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
        result.row_plan_ = MoEGroupedRowPlan::nativeMicro(role,floor);
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

    /** @brief Bind a wide consumer with one count-only producer choice for every admitted codebook.
     * Shared gate/up directory rows cannot depend on separate descriptor masks.
     * @param role Shared gate/up or down. @param source_n Unsliced output width.
     * @param physical_n Prepared output width. @param k Contraction width.
     * @param floor Whole-expert cutoff. @return A sealed geometry and row-plan binding. */
    static MoEGroupedDotAdmission wideCandidate(MoEPrefillProjectionRole role, int source_n,
        int physical_n, int k, MoEGroupedWmmaRowFloor floor)
    {
        auto result = nativeCandidate(role,source_n,physical_n,k,floor);
        result.row_plan_ = MoEGroupedRowPlan::wide(role,floor);
        return result;
    }

    /** @brief Bind the bounded original-source column-directory candidate.
     * @param role Projection role. @param source_n Unsliced width. @param physical_n Prepared width.
     * @param k Contraction width. @return Exact sealed source/geometry/row identity.
     * @throws std::invalid_argument When the original routed geometry is absent. */
    static MoEGroupedDotAdmission columnCandidate(MoEPrefillProjectionRole role,int source_n,int physical_n,int k)
    {
        const bool gateup=role==MoEPrefillProjectionRole::GateUp;
        if(source_n!=physical_n || source_n!=(gateup ? 512 : 2048) || k!=(gateup ? 2048 : 512))
            throw std::invalid_argument("column candidate requires original full-width routed projections");
        auto result=nativeCandidate(role,source_n,physical_n,k,MoEGroupedWmmaRowFloor::Eight);
        result.row_plan_=MoEGroupedRowPlan::column(role);
        return result;
    }

    /** @return The exact plan borrowed by the directory builder and every projection role. */
    [[nodiscard]] LLAMINAR_MOE_DOT_HD constexpr MoEGroupedRowPlan rowPlan() const noexcept { return row_plan_; }

    /** @return Whether recording must include a native projection leaf. */
    [[nodiscard]] constexpr bool mayUseNative() const noexcept { return native_; }

    /** @param role Submitted role. @param n Prepared width. @param k Contraction width.
     * @return Whether a native launch borrows this exact immutable binding. */
    [[nodiscard]] constexpr bool binds(MoEPrefillProjectionRole role, int n, int k) const noexcept
    { return native_ && row_plan_.role() == role && physical_n_ == n && k_ == k; }

    /** @return Original unsliced width retained for admission/evidence identity. */
    [[nodiscard]] constexpr int sourceOutputColumns() const noexcept { return source_n_; }

    /** @return Whether the ordered branch is reachable and needs an adaptive consumer. */
    [[nodiscard]] constexpr bool mayUseOrdered() const noexcept
    {
        constexpr uint32_t all_sources = UINT32_MAX & ~((uint32_t{1} << 20) | (uint32_t{1} << 21));
        return !native_ || row_plan_.floor() != MoEGroupedWmmaRowFloor::EveryExpert ||
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
        if (row_plan_.isColumn() && (!desc.source_identity_present || !desc.source_is_superblock ||
            desc.source_codebook_id != (row_plan_.role()==MoEPrefillProjectionRole::GateUp ? 13 : 4) ||
            desc.codebook_id != (row_plan_.role()==MoEPrefillProjectionRole::GateUp ? 13 : 24)))
            return MoEGroupedDotProducer::Invalid;
        if (whole_expert_rows < static_cast<int>(row_plan_.floor()) ||
            !(physical_codebooks_ & (uint32_t{1} << desc.codebook_id)) ||
            !(source_codebooks_ & (uint32_t{1} << source)))
            return MoEGroupedDotProducer::OrderedDot4;
        return MoEGroupedDotProducer::Wmma;
    }

private:
    bool native_ = false;
    MoEGroupedRowPlan row_plan_;
    int source_n_ = 0;
    int physical_n_ = 0;
    int k_ = 0;
    uint32_t physical_codebooks_ = 0;
    uint32_t source_codebooks_ = 0;
};

static_assert(std::is_trivially_copyable_v<MoEGroupedDotAdmission>);
static_assert(std::is_standard_layout_v<MoEGroupedDotAdmission>);
static_assert(std::is_trivially_copyable_v<MoEGroupedRowPlan>);
static_assert(std::is_standard_layout_v<MoEGroupedRowPlan>);
static_assert(sizeof(MoEGroupedRowPlan) == 3);
static_assert(sizeof(MoEGroupedDotAdmission) == 24);
}
#undef LLAMINAR_MOE_DOT_HD
