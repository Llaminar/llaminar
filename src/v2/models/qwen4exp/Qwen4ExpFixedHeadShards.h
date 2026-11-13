/**
 * @file Qwen4ExpFixedHeadShards.h
 * @brief Source-owned QFN16-head fixed native partitions, without selectable distributed serving.
 *
 * Equal query head counts do not imply equal stored native bytes. Boundaries
 * follow original head offsets; all leading/inter-head gaps and trailing rows
 * belong exactly once to contiguous primary ranges. PMA remains byte authority.
 */
#pragma once
#include "loaders/ResolvedImmutableRowTablePlacement.h"
#include <memory>
namespace llaminar2 {
/** @brief Immutable derived head coordinate plus original range; no stored routing map. */
struct Qwen4ExpFixedHeadShardAssignment final {
    std::size_t participant; ///< Coordinate in the original declared collective/cohort order.
    std::size_t first_head; ///< First original head, never inferred from sorted physical ordinals.
    std::size_t head_count; ///< Equal logical head workload; native storage lengths may differ.
    ImmutableRowTableSourceRange source; ///< Exact original range from the sole layout inventory.
};
/** @brief Model-owned constructor for the existing resolved fixed-range alternative. */
class Qwen4ExpFixedHeadShards final {
public:
    /**
     * @brief Check original QFN head metadata and declare one fixed native range per actual TP consumer.
     * @param source Original full codec/rows/native stride.
     * @param offsets Exact ordered16 head offsets from authenticated model metadata.
     * @param vocab_sizes Exact positive16 original head vocabularies, allowing gaps/trailing source rows.
     * @param participants Ordered existing homogeneous rank-local GPU TP cohort, degree2/4/8.
     * The caller must use this exact order for future collective ranks. This
     * component cannot authenticate request replication or invent TP membership.
     * @return Sole checked resolved layout authority; no capacity admission, upload or graph route is implied.
     * @throws std::invalid_argument For missing/overlapping/outside heads or unsupported/foreign/duplicate cohort.
     */
    static std::shared_ptr<const ResolvedImmutableRowTablePlacement> declare(
        ImmutableRowTableExtent source,std::span<const std::uint64_t> offsets,
        std::span<const std::uint64_t> vocab_sizes,std::span<const PhysicalMemoryAllocatorIdentity> participants);
    /**
     * @brief Derive head coordinates from the sole source-ordered range inventory.
     * @param placement Checked declaration from declare(), never a whole-table placement.
     * @param owner Exact actual owner to project.
     * @return Immutable coordinate/range value; no parallel retained assignment list.
     * @throws std::exception For wrong placement or foreign owner.
     */
    static Qwen4ExpFixedHeadShardAssignment assignmentFor(
        const ResolvedImmutableRowTablePlacement &placement,PhysicalMemoryAllocatorIdentity owner);
};
} // namespace llaminar2
