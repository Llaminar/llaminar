/**
 * @file MoEOverlayDynamicPolicyWorkspace.h
 * @brief Exact persistent scratch geometry for deterministic device policy.
 *
 * The immutable header and typed arrays share one admitted allocation. Only
 * the authority's serialized maintenance stream uses its mutable tail; decode,
 * prefill and terminal restoration reuse those bytes after their graph edges.
 * Shapes, offsets, generation and byte extent are captured identity. No field
 * count depends on a hardware ceiling or native shared-memory capacity.
 */
#pragma once
#include "DeviceMoERuntimeABI.h"
#include "MoEOverlayExtent.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MOE_POLICY_HD __host__ __device__
#else
#define LLAMINAR_MOE_POLICY_HD
#endif

// One field catalog owns scalar type and logical extent for sizing and views.
#define LLAMINAR_MOE_POLICY_FIELDS(X) \
 X(participant_state_base, const std::uint64_t *, Participants) \
 X(expert_counts, std::uint64_t, Experts) \
 X(phase_expert_counts, std::uint64_t, PhaseExperts) \
 X(phase_target_load, std::uint64_t, PhaseTargets) \
 X(target_load, std::uint64_t, Targets) \
 X(service_cost, std::uint64_t, TargetPhases) \
 X(migration_transfer_and_repack_ns, std::uint64_t, TargetPairs) \
 X(migration_inference_interference_ns, std::uint64_t, TargetPairs) \
 X(last_moved, std::uint64_t, Experts) \
 X(minimum_residency_generations, std::uint64_t, Scalar) \
 X(payoff_horizon_tokens, std::uint64_t, Scalar) \
 X(minimum_net_benefit_ns, std::uint64_t, Scalar) \
 X(current_owner, std::int32_t, Experts) \
 X(desired_owner, std::int32_t, Experts) \
 X(candidate_owner, std::int32_t, Experts) \
 X(target_priority, std::int32_t, Targets) \
 X(target_tier, std::int32_t, Targets) \
 X(priorities, std::int32_t, Targets) \
 X(ordered_experts, std::uint32_t, Experts) \
 X(quotas, std::uint32_t, Targets) \
 X(remaining, std::uint32_t, Targets) \
 X(priority_targets, std::uint32_t, Targets) \
 X(cycle, std::uint32_t, Targets) \
 X(best_cycle, std::uint32_t, Targets) \
 X(fallback_cycle, std::uint32_t, Targets) \
 X(dfs_source, std::uint32_t, Targets) \
 X(dfs_next_expert, std::uint32_t, Targets) \
 X(excluded, std::uint8_t, Experts) \
 X(enumerated, std::uint8_t, Experts) \
 X(visited, std::uint8_t, Targets)

namespace llaminar2
{
    /** @brief Immutable physical-member and logical-target extents; they are never aliases. */
    struct MoEOverlayDynamicPolicyShape
    {
        std::uint32_t participants = 0u;
        std::uint32_t targets = 0u;
        std::uint32_t tiers = 0u;
        std::uint32_t experts = 0u;
        /** @return Whether the source runtime can address all declared coordinates. */
        LLAMINAR_MOE_POLICY_HD constexpr bool valid() const noexcept
        {
            return participants && participants <= INT32_MAX && targets && targets <= participants && tiers && tiers <= targets &&
                experts && experts <= moe_runtime_abi::kMaxExperts;
        }
        LLAMINAR_MOE_POLICY_HD constexpr bool operator==(const MoEOverlayDynamicPolicyShape &) const = default;
    };

    /** @brief Named independently typed arrays in the single admitted workspace. */
    enum class MoEOverlayDynamicPolicyField : std::uint32_t
    {
#define LLAMINAR_MOE_POLICY_ENUM(name, type, extent) name,
        LLAMINAR_MOE_POLICY_FIELDS(LLAMINAR_MOE_POLICY_ENUM)
#undef LLAMINAR_MOE_POLICY_ENUM
        Count,
    };

    /** @brief Immutable allocation header; mutable algorithm state lives only in its tail. */
    struct alignas(8) MoEOverlayDynamicPolicyWorkspaceHeader
    {
        static constexpr std::uint32_t kMagic = 0x57504f4du; // MOPW
        static constexpr std::uint32_t kVersion = 2u;
        static constexpr std::uint32_t kPhases = moe_runtime_abi::kHistogramSourceCount;
        static constexpr std::uint32_t kFields = static_cast<std::uint32_t>(MoEOverlayDynamicPolicyField::Count);
        std::uint32_t magic = kMagic;
        std::uint32_t version = kVersion;
        MoEOverlayDynamicPolicyShape shape;
        std::uint32_t reserved = 0u;
        std::uint64_t bytes = 0u;
        std::uint64_t generation = 0u;
        std::uint64_t offsets[kFields] = {};

        /** @brief Logical field extents used by the shared catalog. */
        enum class Extent { Participants, Targets, Experts, PhaseExperts, PhaseTargets, TargetPhases, TargetPairs, Scalar };

        /** @brief Resolve one logical element count without allocating or narrowing.
         * @param shape Original admitted cardinalities. @param extent Named array geometry.
         * @return Exact element count. */
        LLAMINAR_MOE_POLICY_HD static constexpr std::uint64_t countFor(
            MoEOverlayDynamicPolicyShape shape, Extent extent) noexcept
        {
            switch (extent)
            {
            case Extent::Participants: return shape.participants;
            case Extent::Targets: return shape.targets;
            case Extent::Experts: return shape.experts;
            case Extent::PhaseExperts: return std::uint64_t(kPhases) * shape.experts;
            case Extent::PhaseTargets: return std::uint64_t(kPhases) * shape.targets;
            case Extent::TargetPhases: return std::uint64_t(shape.targets) * kPhases;
            case Extent::TargetPairs: return std::uint64_t(shape.targets) * shape.targets;
            case Extent::Scalar: return 1u;
            }
            return 0u;
        }

        /** @brief Resolve exact array cardinality from the single field catalog.
         * @param shape Original cardinalities. @param field Named typed array.
         * @return Element count, or zero for a foreign field. */
        LLAMINAR_MOE_POLICY_HD static constexpr std::uint64_t count(
            MoEOverlayDynamicPolicyShape shape, MoEOverlayDynamicPolicyField field) noexcept
        {
            switch (field)
            {
#define LLAMINAR_MOE_POLICY_COUNT(name, type, extent) case MoEOverlayDynamicPolicyField::name: return countFor(shape, Extent::extent);
                LLAMINAR_MOE_POLICY_FIELDS(LLAMINAR_MOE_POLICY_COUNT)
#undef LLAMINAR_MOE_POLICY_COUNT
            default: return 0u;
            }
        }

        /** @brief Advance one checked typed field in canonical storage order.
         * @param cursor Current byte frontier. @param shape Admitted cardinalities.
         * @param field Next named field. @param offset Receives its aligned start.
         * @return False before an overflowing product, sum or foreign field. */
        LLAMINAR_MOE_POLICY_HD static constexpr bool append(
            std::uint64_t &cursor, MoEOverlayDynamicPolicyShape shape,
            MoEOverlayDynamicPolicyField field, std::uint64_t &offset) noexcept
        {
            std::uint64_t alignment = 0u, scalar_bytes = 0u;
            switch (field)
            {
#define LLAMINAR_MOE_POLICY_SIZE(name, type, extent) case MoEOverlayDynamicPolicyField::name: alignment = alignof(type); scalar_bytes = sizeof(type); break;
                LLAMINAR_MOE_POLICY_FIELDS(LLAMINAR_MOE_POLICY_SIZE)
#undef LLAMINAR_MOE_POLICY_SIZE
            default: return false;
            }
            if (cursor > UINT64_MAX - (alignment - 1u)) return false;
            offset = (cursor + alignment - 1u) & ~(alignment - 1u);
            const auto elements = count(shape, field);
            if (elements > (UINT64_MAX - offset) / scalar_bytes) return false;
            cursor = offset + elements * scalar_bytes;
            return true;
        }

        /** @brief Compute the exact BOM including immutable layout header.
         * @param shape Original participant/tier/expert counts.
         * @return Byte extent, or zero for invalid/overflowing geometry. */
        LLAMINAR_MOE_POLICY_HD static constexpr std::uint64_t bytesFor(MoEOverlayDynamicPolicyShape shape) noexcept
        {
            if (!shape.valid()) return 0u;
            std::uint64_t cursor = sizeof(MoEOverlayDynamicPolicyWorkspaceHeader), offset = 0u;
            for (std::uint32_t field = 0u; field < kFields; ++field)
                if (!append(cursor, shape, static_cast<MoEOverlayDynamicPolicyField>(field), offset)) return 0u;
            return cursor;
        }

        /** @brief Create the sole immutable setup header for an admitted allocation.
         * @param shape Original cardinalities. @param generation Positive model-lifetime identity.
         * @return Complete exact layout. @throws std::invalid_argument For invalid/overflowing geometry. */
        static MoEOverlayDynamicPolicyWorkspaceHeader checked(MoEOverlayDynamicPolicyShape shape,
            std::uint64_t generation)
        {
            MoEOverlayDynamicPolicyWorkspaceHeader result;
            result.shape = shape;
            result.bytes = bytesFor(shape);
            result.generation = generation;
            if (!result.bytes || !generation) throw std::invalid_argument("Dynamic policy workspace has invalid or overflowing admitted geometry");
            std::uint64_t cursor = sizeof(result);
            for (std::uint32_t field = 0u; field < kFields; ++field)
                if (!append(cursor, shape, static_cast<MoEOverlayDynamicPolicyField>(field), result.offsets[field]))
                    throw std::invalid_argument("Dynamic policy workspace field extent overflows");
            return result;
        }

        /** @brief Authenticate every header field before exposing any mutable array.
         * @param expected_shape Captured immutable geometry. @param expected_bytes Admitted bytes.
         * @param expected_generation Captured model-lifetime workspace identity.
         * @return True only for the exact canonical layout and generation. */
        LLAMINAR_MOE_POLICY_HD bool validFor(MoEOverlayDynamicPolicyShape expected_shape,
            std::uint64_t expected_bytes, std::uint64_t expected_generation) const noexcept
        {
            if (magic != kMagic || version != kVersion || reserved || !shape.valid() ||
                !(shape == expected_shape) || !generation || generation != expected_generation || bytes != expected_bytes)
                return false;
            std::uint64_t cursor = sizeof(*this), offset = 0u;
            for (std::uint32_t field = 0u; field < kFields; ++field)
                if (!append(cursor, shape, static_cast<MoEOverlayDynamicPolicyField>(field), offset) || offsets[field] != offset)
                    return false;
            return cursor == bytes;
        }
    };

    /** @brief Captured exact allocation identity; host validation never reads its device pointer. */
    struct MoEOverlayDynamicPolicyWorkspaceBinding
    {
        void *data = nullptr;
        std::uint64_t bytes = 0u;
        std::uint64_t generation = 0u;
        MoEOverlayDynamicPolicyShape shape;
        /** @return Whether immutable pointer/geometry/generation form one complete binding. */
        LLAMINAR_MOE_POLICY_HD bool valid() const noexcept
        {
            return data && generation && bytes && bytes == MoEOverlayDynamicPolicyWorkspaceHeader::bytesFor(shape);
        }
        /** @return Capture equality of every pointer and policy-layout scalar. */
        bool operator==(const MoEOverlayDynamicPolicyWorkspaceBinding &) const = default;
    };

    /** @brief Zero-storage row-major view retaining the original matrix arithmetic order. */
    template <typename T> struct MoEOverlayDynamicPolicyRows
    {
        T *data;
        std::uint32_t columns;
        /** @param row Original first-axis index. @return Exact contiguous row. */
        LLAMINAR_MOE_POLICY_HD T *operator[](std::uint32_t row) const noexcept
        { return data + std::uint64_t(row) * columns; }
    };

    /** @brief Borrowed typed views into the original persistent scratch, never a second owner. */
    class MoEOverlayDynamicPolicyWorkspaceView
    {
    public:
        /** @param binding Exact captured storage identity; validate before accessing arrays. */
        LLAMINAR_MOE_POLICY_HD explicit MoEOverlayDynamicPolicyWorkspaceView(
            MoEOverlayDynamicPolicyWorkspaceBinding binding) noexcept : binding_(binding) {}
        /** @brief Authenticate device-owned immutable header against captured geometry.
         * @param shape Runtime controller geometry. @return False before any scratch write on mismatch. */
        LLAMINAR_MOE_POLICY_HD bool validFor(MoEOverlayDynamicPolicyShape shape) const noexcept
        {
            return binding_.valid() && binding_.shape == shape &&
                header()->validFor(shape, binding_.bytes, binding_.generation);
        }
#define LLAMINAR_MOE_POLICY_ACCESS(name, type, extent) \
        /** @return Exact typed array from the authenticated original workspace. */ \
        LLAMINAR_MOE_POLICY_HD type *name() const noexcept { return array<type>(MoEOverlayDynamicPolicyField::name); }
        LLAMINAR_MOE_POLICY_FIELDS(LLAMINAR_MOE_POLICY_ACCESS)
#undef LLAMINAR_MOE_POLICY_ACCESS
        /** @return Exact immutable allocation header; no mutable execution state is mirrored. */
        LLAMINAR_MOE_POLICY_HD const MoEOverlayDynamicPolicyWorkspaceHeader *header() const noexcept
        { return static_cast<const MoEOverlayDynamicPolicyWorkspaceHeader *>(binding_.data); }
        /** @brief Resolve one typed mutable field after validFor established its layout.
         * @tparam T Catalog scalar type. @param field Original named array.
         * @return Contained alias, or null for a malformed range. */
        template <typename T>
        LLAMINAR_MOE_POLICY_HD T *array(MoEOverlayDynamicPolicyField field) const noexcept
        {
            return moeOverlayExtentArray<T>(binding_.data, binding_.bytes,
                header()->offsets[static_cast<std::uint32_t>(field)],
                MoEOverlayDynamicPolicyWorkspaceHeader::count(binding_.shape, field));
        }
        /** @return Phase-major expert rows with the exact admitted expert stride. */
        LLAMINAR_MOE_POLICY_HD MoEOverlayDynamicPolicyRows<std::uint64_t> phaseExpertRows() const noexcept
        { return {phase_expert_counts(), binding_.shape.experts}; }
        /** @return Phase-major logical-target rows, without fixed participant padding. */
        LLAMINAR_MOE_POLICY_HD MoEOverlayDynamicPolicyRows<std::uint64_t> phaseTargetRows() const noexcept
        { return {phase_target_load(), binding_.shape.targets}; }
        /** @return Target-major measured-phase rows. */
        LLAMINAR_MOE_POLICY_HD MoEOverlayDynamicPolicyRows<std::uint64_t> serviceRows() const noexcept
        { return {service_cost(), MoEOverlayDynamicPolicyWorkspaceHeader::kPhases}; }
        /** @return Directed source/destination rows in immutable logical-target order. */
        LLAMINAR_MOE_POLICY_HD MoEOverlayDynamicPolicyRows<std::uint64_t> transferRows() const noexcept
        { return {migration_transfer_and_repack_ns(), binding_.shape.targets}; }
        /** @return Directed interference rows in the same exact order. */
        LLAMINAR_MOE_POLICY_HD MoEOverlayDynamicPolicyRows<std::uint64_t> interferenceRows() const noexcept
        { return {migration_inference_interference_ns(), binding_.shape.targets}; }
    private:
        MoEOverlayDynamicPolicyWorkspaceBinding binding_;
    };
    static_assert(sizeof(void *) == sizeof(std::uint64_t));
    static_assert(std::is_trivially_copyable_v<MoEOverlayDynamicPolicyWorkspaceBinding>);
}
#undef LLAMINAR_MOE_POLICY_FIELDS
#undef LLAMINAR_MOE_POLICY_HD
