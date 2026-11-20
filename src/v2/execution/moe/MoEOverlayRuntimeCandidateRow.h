/**
 * @file MoEOverlayRuntimeCandidateRow.h
 * @brief Shared host/device lowering of one complete logical movement destination.
 *
 * This pure operation grants no epoch, arrival or publication permission. The
 * controller first acquires its original physical transport receipt, validates
 * every row, then reserves/clones an inactive bank. Whole owners retain their
 * domain-local replica metadata; fragments retain original family geometry,
 * owner=-1 and zero replica metadata. Publication still waits for every original
 * participant/group ACK through the existing durable epoch protocol.
 */
#pragma once
#include "DeviceMoEIntermediateShardBinding.h"
#include "kernels/common/DeviceMoEFloatingMatrixDesc.h"
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define LLAMINAR_CANDIDATE_ROW_HD __host__ __device__ __forceinline__
#elif defined(__HIPCC__)
#define LLAMINAR_CANDIDATE_ROW_HD __host__ __device__ inline __attribute__((always_inline))
#else
#define LLAMINAR_CANDIDATE_ROW_HD inline
#endif

namespace llaminar2
{
    /** @brief Immutable local runtime/member geometry; never a mutable owner snapshot. */
    struct MoEOverlayRuntimeCandidateContext
    {
        int model_layer = -1;
        std::uint32_t expert_count = 0;
        int global_participant = -1;
        std::uint32_t domain_participant = 0;
        std::uint32_t domain_participants = 0;
        /** SAME original group's role, independent of whether a runtime family is missing. */
        MoEOverlayRouteTarget original_cohort_target;
        DeviceMoEIntermediateShardBinding fragment;
    };

    /** @brief One validated bank row, identical for host and native descriptor views. */
    template <typename Descriptor>
    struct MoEOverlayRuntimeCandidateRow
    {
        Descriptor descriptor{};
        std::uint8_t local_compute = 0;
        std::uint8_t replica_role = 0;
        std::uint32_t resident_mask = 0;
        std::int32_t route = -1;
    };

    /**
     * @brief Build one exact destination row without mutating a runtime bank.
     * @param context Original runtime-bound family and both member namespaces.
     * @param expert Original global expert ID.
     * @param target Authenticated complete destination route.
     * @param whole_domain_member Whole destination's domain index, or -1 outside this group/cohort.
     * @param arrival Exact local completed descriptor; null for nonlocal destinations.
     * @param output Validated result, unchanged on rejection.
     * @return False for missing arrivals, fake whole cohort owners, malformed
     *         namespace/geometry, replica fragments or foreign descriptors.
     */
    template <typename Descriptor>
    LLAMINAR_CANDIDATE_ROW_HD bool buildMoEOverlayRuntimeCandidateRow(
        const MoEOverlayRuntimeCandidateContext &context, std::uint32_t expert,
        MoEOverlayRouteTarget target, int whole_domain_member,
        const Descriptor *arrival, MoEOverlayRuntimeCandidateRow<Descriptor> &output) noexcept
    {
        if (context.model_layer < 0 || context.expert_count == 0 || expert >= context.expert_count ||
            context.global_participant < 0 || context.domain_participants == 0 ||
            context.domain_participant >= context.domain_participants || target.isAbsent() ||
            !context.fragment.target().validFor(context.expert_count, context.global_participant, context.model_layer))
            return false;
        if (context.original_cohort_target.isCohort())
        {
            if (!context.fragment.enabled() ||
                context.fragment.target().cohortTarget() != context.original_cohort_target.word())
                return false;
        }
        else if (!context.original_cohort_target.isAbsent() || context.fragment.enabled())
            return false;
        MoEOverlayRuntimeCandidateRow<Descriptor> row;
        row.route = target.word();
        row.descriptor.logical_expert_id = static_cast<std::int32_t>(expert);
        row.descriptor.owner_participant = -1;
        row.descriptor.local_slot = -1;
        if (target.isCohort())
        {
            if (whole_domain_member != -1)
                return false;
            const bool local = context.original_cohort_target.word() == target.word();
            if (!local)
            {
                if (arrival)
                    return false;
            }
            else
            {
                if (!arrival || arrival->logical_expert_id != static_cast<std::int32_t>(expert) ||
                    arrival->local_slot < 0 || arrival->owner_participant != -1 ||
                    arrival->flags != toMoEExpertFlags(DeviceMoEExpertFlags::Valid | DeviceMoEExpertFlags::Resident) ||
                    !deviceMoEExpertProjectionReady(*arrival, DeviceMoEProjectionSet::IntermediateShard))
                    return false;
                const auto h = static_cast<std::int32_t>(context.fragment.modelColumns());
                const auto f = static_cast<std::int32_t>(context.fragment.localIntermediateColumns());
                if (deviceMoEWeightFormatIsFloating(arrival->weight_format))
                {
                    if (arrival->floating_gate.n != f || arrival->floating_gate.k != h)
                        return false;
                }
                else if (h % 32 || f % 32 || arrival->gate.n != f || arrival->gate.k != h ||
                    arrival->gate.blocks_per_row != static_cast<std::uint32_t>(h / 32) ||
                    arrival->up.blocks_per_row != static_cast<std::uint32_t>(h / 32) ||
                    arrival->down.blocks_per_row != static_cast<std::uint32_t>(f / 32))
                    return false;
                row.descriptor = *arrival;
                row.descriptor.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                    DeviceMoEExpertFlags::Resident | DeviceMoEExpertFlags::LocalCompute);
                row.local_compute = 1;
            }
        }
        else
        {
            if (whole_domain_member < -1 ||
                (whole_domain_member >= 0 &&
                 (static_cast<std::uint32_t>(whole_domain_member) >= context.domain_participants ||
                  whole_domain_member >= std::numeric_limits<std::uint32_t>::digits || context.fragment.enabled())))
                return false;
            const bool local = target.word() == context.global_participant;
            if (local)
            {
                if (whole_domain_member != static_cast<int>(context.domain_participant) || !arrival ||
                    arrival->logical_expert_id != static_cast<std::int32_t>(expert) || arrival->local_slot < 0 ||
                    (arrival->projection_set != DeviceMoEProjectionSet::CompleteExpert &&
                     arrival->projection_set != DeviceMoEProjectionSet::GateUp) ||
                    !deviceMoEExpertProjectionReady(*arrival, arrival->projection_set))
                    return false;
                // Retain the existing complete gate/up/down compatibility law;
                // graph-family readiness alone does not compare whole shapes.
                const bool pair = arrival->projection_set == DeviceMoEProjectionSet::GateUp;
                if (deviceMoEWeightFormatIsFloating(arrival->weight_format))
                {
                    if (arrival->floating_gate.n != arrival->floating_up.n ||
                        arrival->floating_gate.k != arrival->floating_up.k ||
                        (!pair && (arrival->floating_down.n != arrival->floating_gate.k ||
                                   arrival->floating_down.k != arrival->floating_gate.n)))
                        return false;
                }
                else if (arrival->gate.n != arrival->up.n || arrival->gate.k != arrival->up.k ||
                    (!pair && (arrival->down.n != arrival->gate.k || arrival->down.k != arrival->gate.n)))
                    return false;
                row.descriptor = *arrival;
                row.descriptor.flags = (arrival->flags & toMoEExpertFlags(DeviceMoEExpertFlags::TransferSlot)) |
                    toMoEExpertFlags(DeviceMoEExpertFlags::Valid | DeviceMoEExpertFlags::Resident |
                        DeviceMoEExpertFlags::PreferredOwner | DeviceMoEExpertFlags::LocalCompute);
                row.local_compute = 1;
                row.replica_role = 1; // Shared ABI Primary; fragments never use it.
            }
            else if (arrival)
                return false;
            if (whole_domain_member >= 0)
            {
                row.descriptor.owner_participant = whole_domain_member;
                row.resident_mask = std::uint32_t{1} << whole_domain_member;
            }
        }
        output = row;
        return true;
    }
} // namespace llaminar2
#undef LLAMINAR_CANDIDATE_ROW_HD
