/**
 * @file OrdinaryGenerationGraphPlan.h
 * @brief Exact retained ordinary-generation owners shared by setup and PMA input.
 *
 * Sampler recordings imported into a native parent own no executable. Native
 * ownership needs the resident-input forward and one conditional parent;
 * hosted controllers additionally instantiate their two actual tail children.
 * Hosted ownership follows declared execution policy on either GPU vendor.
 * Nonterminal pipeline participants never own the tail's sampler children.
 * A retained speculative family already accounts for the shared parent/ticket
 * owner, so it must not be charged twice. This is a BOM, never a live ledger.
 */
#pragma once
#include "backends/DeviceId.h"
#include "execution/mtp/DeviceGenerationExecutionPolicy.h"
#include "kernels/common/SparseQueryExecutionProgram.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
/** @brief Physical PP membership authorizes a sampler-free retained parent. */
enum class PipelineForwardOnlyGraphOwner : uint8_t
{
    None, ///< No declared PP participant; partial layer ranges do not imply ownership.
    Participant, ///< One original PP participant, including every TP sibling.
};

/** @brief Immutable native-owner geometry for scalar ordinary generation. */
struct OrdinaryGenerationGraphPlan
{
    static constexpr std::size_t prefill_sampler = 0;
    static constexpr std::size_t decode_sampler = 1;
    static constexpr std::size_t hosted_transaction = 2;
    static constexpr std::size_t child_recordings = 3;

    /**
     * @brief Count sampler-free parents for the authored physical PP member.
     * @param query_programs Exact immutable main-model program family.
     * @param owner Membership projected by the frozen rank/stage declaration.
     * @return One parent per program for a participant, zero without ownership.
     * @throws std::invalid_argument for an unknown program family or owner.
     *
     * Native and captured-domain parents import existing model/transport
     * recordings; neither creates independently instantiated child owners.
     */
    static std::size_t pipelineForwardOnlyExecutableCount(std::size_t query_programs,
        PipelineForwardOnlyGraphOwner owner)
    {
        if (query_programs == 0 || query_programs > kSparseQueryExecutionProgramCount)
            throw std::invalid_argument("Pipeline forced graph inventory has an unknown query-program family");
        switch (owner)
        {
        case PipelineForwardOnlyGraphOwner::None: return 0u;
        case PipelineForwardOnlyGraphOwner::Participant: return query_programs;
        }
        throw std::invalid_argument("Pipeline forced graph inventory has an unknown physical owner");
    }

    /**
     * @brief Price actual ordinary owners outside the common forward/MTP BOM.
     * @param device Exact native GPU participant.
     * @param retained_mtp Whether admitted MTP capacity covers the shared parent.
     * @param execution Declared native/hosted policy; vendor cannot infer a mixed boundary.
     * @param parent_role Sampling authority or nonterminal pipeline follower.
     * @param query_programs Whole-main-model immutable query-program count.
     * @return Independently resident general executable slots.
     * @throws std::invalid_argument for unsupported policy, owner or device.
     */
    static std::size_t additionalExecutableCount(DeviceId device, bool retained_mtp,
        DeviceGenerationExecutionPolicy execution,
        MTPGenerationParentRole parent_role,
        std::size_t query_programs = 1u)
    {
        if (!device.is_gpu())
            throw std::invalid_argument("Ordinary generation graph inventory requires a GPU");
        if (query_programs == 0 || query_programs > kSparseQueryExecutionProgramCount)
            throw std::invalid_argument("Ordinary graph inventory has an unknown query-program family");
        if (execution != DeviceGenerationExecutionPolicy::NativeConditionalGraph &&
            execution != DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions)
            throw std::invalid_argument("Ordinary graph inventory requires a supported declared execution policy");
        if (execution != declaredDeviceGenerationExecutionPolicy(device.type, DeviceGenerationExecutionDomain::NativeDomain) &&
            execution != declaredDeviceGenerationExecutionPolicy(device.type, DeviceGenerationExecutionDomain::HostedBoundary))
            throw std::invalid_argument("Ordinary graph inventory execution policy contradicts its device declaration");
        if (parent_role != MTPGenerationParentRole::Controller &&
            parent_role != MTPGenerationParentRole::PipelineFollower)
            throw std::invalid_argument("Ordinary graph inventory has an unknown parent owner");
        // Each program owns its resident-input forward and parent; the MTP
        // inventory covers the mutually exclusive parent's retained capacity.
        // Only a hosted sampling authority instantiates the two tail children:
        // prefill+decode samplers for overlays, prefill+transaction otherwise.
        // Hosted PP followers own one complete local parent, with no sampler.
        const bool owns_hosted_children =
            execution == DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions &&
            parent_role == MTPGenerationParentRole::Controller;
        return query_programs * (1u + (retained_mtp ? 0u : 1u) + (owns_hosted_children ? 2u : 0u));
    }
};
}
