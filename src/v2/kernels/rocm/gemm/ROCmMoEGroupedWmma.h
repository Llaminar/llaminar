/**
 * @file ROCmMoEGroupedWmma.h
 * @brief Native matrix leaves for the retained grouped MoE projection transaction.
 *
 * Dot-engine recording is independent of the device-owned route strategy. WMMA
 * replaces only ExpertTiled gate/up and down leaves. Nonzero row floors select
 * Dot4 or WMMA inside one consumer with a shared directory; quantization, routing,
 * SwiGLU and ordered publication retain their existing owners and allocations.
 * The explicit wide candidate keeps a column per Dot4 thread and computes one
 * native column chunk per block, with cold experts using every fourth block. Its count-only row plan also owns directory
 * generation, including the twelve-row gate/up versus sixteen-row native step.
 */
#pragma once
#include "ROCmMoEPrefillArithmetic.h"
#include "kernels/common/MoEPrefillProjectionExecution.h"
#include "ROCmMoEGroupedPrefillRoutePolicy.h"
#include "ROCmMoEGroupedDotAdmission.h"
#include "ROCmMoEGroupedPopulationPolicy.h"
#include <array>

namespace llaminar2::rocm
{
/** @brief Explicit recording choices; Automatic retains installed measured policy. */
enum class MoEGroupedPrefillEngine { Automatic, PopulationAdaptive, OrderedDot4, WmmaM16N32, WmmaM16N64, WideDot4Wmma };

/** @brief Directory producer embedded in the captured operation, independent of arithmetic. */
enum class MoEGroupedDirectorySource { RolePlanner, GroupingOffsetSlots, GroupingColumnSlots };

/** @brief Nestable worker-local selection, embedded in the captured native nodes. */
class ScopedMoEGroupedPrefillEngine final
{
public:
    /** @brief Set the recording engine and its explicit whole-expert row floor.
     * @param engine Explicit candidate or installed policy.
     * @param floor Native eligibility threshold; only native candidates accept a nonzero floor.
     * @param directory Directory producer; offset slots require the explicit wide floor-eight candidate. */
    explicit ScopedMoEGroupedPrefillEngine(MoEGroupedPrefillEngine engine,
        MoEGroupedWmmaRowFloor floor = MoEGroupedWmmaRowFloor::EveryExpert,
        MoEGroupedDirectorySource directory = MoEGroupedDirectorySource::RolePlanner);
    /** @brief Restore the enclosing recording policy without changing retained graphs. */
    ~ScopedMoEGroupedPrefillEngine();
    ScopedMoEGroupedPrefillEngine(const ScopedMoEGroupedPrefillEngine &) = delete;
    ScopedMoEGroupedPrefillEngine &operator=(const ScopedMoEGroupedPrefillEngine &) = delete;
private:
    MoEGroupedPrefillEngine previous_;
    MoEGroupedWmmaRowFloor previous_floor_;
    MoEGroupedDirectorySource previous_directory_;
};

/** @brief Resolved recording policy; Automatic never survives native admission. */
struct MoEGroupedPrefillPolicy
{
    MoEGroupedPrefillEngine engine;
    MoEGroupedWmmaRowFloor minimum_expert_rows;
    MoEGroupedPopulationDomain population = MoEGroupedPopulationDomain::Unmeasured;
    MoEGroupedDirectorySource directory = MoEGroupedDirectorySource::RolePlanner;
};

/** @brief One distinct successfully submitted tiled launch geometry, including dormant leaves. */
struct MoEGroupedPrefillTileSubmission
{
    MoEGroupedPrefillEngine engine = MoEGroupedPrefillEngine::OrderedDot4;
    MoEGroupedWmmaRowFloor floor = MoEGroupedWmmaRowFloor::EveryExpert;
    int rows = 0, columns = 0, threads = 0, ordered_rows = 0;
    uint32_t count = 0;
    int ordered_columns = 0; ///< Cold branch width, distinct from native columns.
};

/** @brief Bounded capture-time inventory; no container allocation or device-state mirror. */
struct MoEGroupedPrefillRoleSubmission
{
    std::array<MoEGroupedPrefillTileSubmission,16> tiles{};
    size_t size = 0;
    /** @param tile Actual submitted geometry and number of physical-codebook leaves.
     * @throws std::logic_error If an unsupported geometry roster exceeds its declared bound. */
    void record(MoEGroupedPrefillTileSubmission tile)
    {
        for (size_t i = 0; i < size; ++i)
        {
            auto &old = tiles[i];
            if (old.engine == tile.engine && old.floor == tile.floor && old.rows == tile.rows &&
                old.columns == tile.columns && old.threads == tile.threads && old.ordered_rows == tile.ordered_rows &&
                old.ordered_columns == tile.ordered_columns)
            { old.count += tile.count; return; }
        }
        if (size == tiles.size()) throw std::logic_error("grouped submission geometry inventory overflow");
        tiles[size++] = tile;
    }
    /** @return Number of submitted tiled kernels, not replay executions. */
    [[nodiscard]] uint32_t leaves() const noexcept
    { uint32_t result = 0; for (size_t i = 0; i < size; ++i) result += tiles[i].count; return result; }
};

/** @brief Caller-owned construction receipt; published only after complete launch success. */
struct MoEGroupedPrefillSubmission
{
    enum class State : uint8_t { Empty, Complete };
    State state = State::Empty;
    MoEGroupedPrefillPolicy policy{MoEGroupedPrefillEngine::OrderedDot4,MoEGroupedWmmaRowFloor::EveryExpert};
    MoEGroupedPrefillRoleSubmission gateup, down;
    bool gateup_route_owned = false, down_route_owned = false;
    bool exact_ordered_overlay = false;
};

/** @brief Resolve recording against immutable native hardware metadata.
 * @param device Visible HIP ordinal owning the complete transaction.
 * @param key Actual immutable operation, including source aliases and Q8 ownership.
 * @return Concrete Dot4 or native WMMA engine and whole-expert row floor.
 * @throws std::invalid_argument When an explicit native candidate is unsupported. */
MoEGroupedPrefillPolicy selectedMoEGroupedPrefillPolicy(int device, const MoEGroupedPopulationKey &key);

/** @brief Which authoritative row index addresses the quantized activation bank. */
enum class MoEGroupedWmmaInput { OriginalTokens, OriginalRouteSlots, GroupedRows };

/** @brief Fault injection accepted only by direct test mutation of a retained native node.
 * Production submission requires None. A test may suppress one selected branch
 * to prove its actual writer ownership while checking that the other survives. */
enum class MoEGroupedBranchSuppression : uint8_t { None, Ordered, Wmma };

/** @brief Frozen projection bindings; live extents come only from the group plan. */
struct MoEGroupedWmmaArguments
{
    const int8_t *activations = nullptr;
    const float *activation_scales = nullptr;
    const int *group_token_indices = nullptr;
    const int *group_counts = nullptr;
    const int *group_offsets = nullptr;
    const uint32_t *directory = nullptr;
    const DeviceNativeQuantisedMatrixDesc *descriptors = nullptr;
    const DeviceNativeQuantisedMatrixDesc *second_descriptors = nullptr;
    float *output = nullptr;
    float *second_output = nullptr;
    int directory_capacity = 0;
    int rows = 0;
    int top_k = 0;
    int n = 0;
    int k = 0;
    int experts = 0;
    int device = -1;
    MoEGroupedWmmaInput input = MoEGroupedWmmaInput::GroupedRows;
    MoEPrefillSerialM1KPartitionPolicy partitions;
    ROCmMoEGroupedRouteAdmission admission; ///< Required existing group-plan authority.
    MoEGroupedDotAdmission dot_admission; ///< One whole-expert choice shared by both arithmetic branches.
    MoEGroupingPackedLayout directory_layout = MoEGroupingPackedLayout::WideFloor8; ///< Exact published work encoding.
    MoEGroupedBranchSuppression branch_suppression = MoEGroupedBranchSuppression::None; ///< Rejected by production when non-None.
};

/** @brief Authenticate the required live publication before any direct column submission.
 * @param args Actual frozen packet. @param publication Borrow from the current grouping generation.
 * @param stream Exact producer stream retained by the caller.
 * @throws std::logic_error For an absent, revoked, mismatched or falsely tagged column layout. */
inline void requireMoEGroupedColumnPublication(const MoEGroupedWmmaArguments &args,
    const MoEPrefillGroupingProvenance::DirectoryBorrow *publication, void *stream)
{
    if(args.directory_layout!=MoEGroupingPackedLayout::WideFloor8 && args.directory_layout!=MoEGroupingPackedLayout::ColumnFloor8)
        throw std::logic_error("unknown grouped directory encoding");
    const bool column=args.dot_admission.rowPlan().isColumn();
    if(column!=(args.directory_layout==MoEGroupingPackedLayout::ColumnFloor8))
        throw std::logic_error("column row plan and directory encoding disagree");
    if(!column)return;
    if(!publication)throw std::logic_error("column projection requires its live grouping publication");
    (void)publication->requireProjection(args.group_counts,args.group_offsets,args.directory,
        args.rows,args.top_k,args.experts,DeviceId::rocm(args.device),stream,
        MoEGroupingPackedLayout::ColumnFloor8);
}

/** @brief Record all physical-codebook leaves for one gate/up or down projection.
 * @param engine Concrete native tile selected for this device at recording time.
 * @param codebooks Actual physical descriptor mask, preserving separate IQ4 ids.
 * @param args Frozen bindings and original full-width arithmetic partition policy.
 * @param stream Explicit non-default producer stream owned by args.device.
 * @param submitted Optional caller-owned role inventory, cleared on entry and published on full success.
 * @param publication Required current grouping publication for the column-layout candidate.
 * @return Whether every required native launch was accepted. */
bool submitMoEGroupedWmma(MoEGroupedPrefillEngine engine, uint32_t codebooks,
                         const MoEGroupedWmmaArguments &args, void *stream,
                         MoEGroupedPrefillRoleSubmission *submitted = nullptr,
                         const MoEPrefillGroupingProvenance::DirectoryBorrow *publication = nullptr);

/** @brief Publish this TU's immutable IQ grids during backend initialization.
 * @param device Owning native device. @param iq3s IQ3_S uint32 table.
 * @param iq3xxs IQ3_XXS uint32 table. @param iq2s IQ2_S uint64 table.
 * @param iq2xs IQ2_XS uint64 table. @param iq2xxs IQ2_XXS uint64 table.
 * @param iq1s Shared IQ1 uint64 table. @return Whether all tables were published. */
bool initializeMoEGroupedWmmaGrids(int device, const void *iq3s, const void *iq3xxs,
    const void *iq2s, const void *iq2xs, const void *iq2xxs, const void *iq1s);
}
