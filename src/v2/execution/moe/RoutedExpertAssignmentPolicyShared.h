/**
 * @file RoutedExpertAssignmentPolicyShared.h
 * @brief Backend-neutral routed expert assignment policy dispatch.
 *
 * This header is the stable CPU/CUDA/ROCm-facing policy surface for assigning
 * already-routed expert rows to MoE domain participants. Individual algorithms
 * live behind this layer so future policies can be added without teaching every
 * caller about each algorithm's private helper types.
 */

#pragma once

#include "execution/moe/LeastLoadedExpertAssignment.h"

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_ROUTED_EXPERT_ASSIGNMENT_HD __host__ __device__ __forceinline__
#else
#define LLAMINAR_ROUTED_EXPERT_ASSIGNMENT_HD inline
#endif

namespace llaminar2::routed_expert_assignment
{
    enum class Algorithm : uint32_t
    {
        StaticOwner = 0,
        LeastLoadedResident = 1,
    };

    using AssignmentWorkspace = least_loaded_ep::LeastLoadedExpertAssignmentWorkspace;
    using AssignmentSpan = least_loaded_ep::LeastLoadedExpertAssignmentSpan;
    using WeightTransfer = least_loaded_ep::LeastLoadedExpertWeightTransfer;
    using AssignmentStatus = least_loaded_ep::LeastLoadedExpertAssignmentStatus;

    struct PolicyConfig
    {
        Algorithm algorithm = Algorithm::StaticOwner;
        least_loaded_ep::LeastLoadedExpertAssignmentConfig least_loaded;
    };

    LLAMINAR_ROUTED_EXPERT_ASSIGNMENT_HD bool planStaticOwnerAssignment(
        const uint64_t *expert_loads,
        const uint32_t *expert_owner_participants,
        const least_loaded_ep::LeastLoadedExpertAssignmentConfig &config,
        const AssignmentWorkspace &workspace,
        AssignmentSpan *spans,
        uint32_t span_capacity,
        AssignmentStatus *status_out) noexcept
    {
        AssignmentStatus status{};
        if (status_out)
            *status_out = status;

        if (expert_loads == nullptr ||
            expert_owner_participants == nullptr ||
            config.expert_count == 0u ||
            config.participant_count == 0u)
        {
            status.invalid_config = 1u;
            if (status_out)
                *status_out = status;
            return false;
        }

        if (workspace.assigned_load != nullptr)
        {
            for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                workspace.assigned_load[participant] = 0ULL;
        }

        for (uint32_t expert = 0; expert < config.expert_count; ++expert)
        {
            const uint32_t owner = expert_owner_participants[expert];
            if (owner >= config.participant_count)
            {
                status.invalid_config = 1u;
                if (status_out)
                    *status_out = status;
                return false;
            }

            const uint64_t load = expert_loads[expert];
            status.total_load += load;
            if (load > status.max_expert_load)
                status.max_expert_load = load;
            if (workspace.assigned_load != nullptr)
                workspace.assigned_load[owner] += load;

            if (load == 0ULL)
                continue;

            if (!least_loaded_ep::appendAssignmentSpan(
                    spans,
                    span_capacity,
                    nullptr,
                    0u,
                    status,
                    expert,
                    owner,
                    owner,
                    owner < 32u ? (1u << owner) : 0u,
                    0ULL,
                    load,
                    false,
                    config.max_non_owner_experts_per_participant,
                    config.participant_count))
            {
                if (status_out)
                    *status_out = status;
                return false;
            }
        }

        status.standard_ep_selected = 1u;
        if (config.participant_count > 0u)
        {
            status.capacity_per_participant =
                least_loaded_ep::ceilDiv(status.total_load, config.participant_count);
        }
        least_loaded_ep::summarizeParticipantLoads(
            workspace.assigned_load,
            config.participant_count,
            &status.standard_load_min,
            &status.standard_load_max,
            &status.standard_load_spread);
        status.assigned_load_min = status.standard_load_min;
        status.assigned_load_max = status.standard_load_max;
        status.assigned_load_spread = status.standard_load_spread;

        if (status_out)
            *status_out = status;
        return status.overflow == 0u;
    }

    LLAMINAR_ROUTED_EXPERT_ASSIGNMENT_HD bool planRoutedExpertAssignment(
        const uint64_t *expert_loads,
        const uint32_t *expert_owner_participants,
        const PolicyConfig &config,
        const AssignmentWorkspace &workspace,
        AssignmentSpan *spans,
        uint32_t span_capacity,
        WeightTransfer *transfers,
        uint32_t transfer_capacity,
        AssignmentStatus *status_out) noexcept
    {
        switch (config.algorithm)
        {
        case Algorithm::StaticOwner:
            return planStaticOwnerAssignment(
                expert_loads,
                expert_owner_participants,
                config.least_loaded,
                workspace,
                spans,
                span_capacity,
                status_out);
        case Algorithm::LeastLoadedResident:
            return least_loaded_ep::planLeastLoadedExpertAssignment(
                expert_loads,
                expert_owner_participants,
                config.least_loaded,
                workspace,
                spans,
                span_capacity,
                transfers,
                transfer_capacity,
                status_out);
        }

        AssignmentStatus status{};
        status.invalid_config = 1u;
        if (status_out)
            *status_out = status;
        return false;
    }

} // namespace llaminar2::routed_expert_assignment
