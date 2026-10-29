/**
 * @file CPUExpertStageTestAdmission.h
 * @brief Exact packed-storage admission retained by real CPU expert-stage fixtures.
 *
 * Tests retain a model store for each preparation lifetime. Native quantized
 * view extents use the production packer geometry; owning floating expert
 * copies use their original native byte extent. No live byte estimate or
 * process-global admission state participates in these device-free fixtures.
 */
#pragma once
#include "CPUPreparedStoreTestAdmission.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"

namespace llaminar2::test
{
/**
 * @brief Admit the actual active expert projection views for one fixture.
 * @tparam Params Production compute-stage or local-expert-stage parameter type.
 * @param params Positive preparation fixture, updated with its retained store.
 * @param owner Explicit persistent owner exercised by the fixture.
 * @return Owned store whose scope must include every stage using these params;
 *         empty when the caller already retains an external store.
 */
template <typename Params>
std::unique_ptr<PreparedWeightStore> admitCPUExpertStageForTest(
    Params &params, PhysicalMemoryOwner owner)
{
    if (!params.device_id.is_cpu() || (owner != PhysicalMemoryOwner::PrimaryModelWeights &&
        owner != PhysicalMemoryOwner::AdditionalModelWeights && owner != PhysicalMemoryOwner::RoutedExpertWeights))
        throw std::invalid_argument("CPU expert fixture requires an explicit CPU/persistent owner");
    if constexpr (requires { params.persistent_weight_owner; })
        params.persistent_weight_owner = owner;
    if (params.layer_idx < 0)
        params.layer_idx = 0;
    std::unique_ptr<PreparedWeightStore> retained;
    if (!params.prepared_store)
    {
        retained = std::make_unique<PreparedWeightStore>();
        params.prepared_store = retained.get();
    }
    if (params.prepared_store->physicalMemoryAuthority())
        return retained;
    if (params.expert_gate_views.empty())
    {
        // Use the production view extractor rather than duplicating its GGUF
        // orientation, expert slicing or shared-parent lifetime rules.
        MoEExpertComputeStage::Params extraction;
        extraction.device_id = params.device_id;
        extraction.num_experts = params.num_experts;
        extraction.d_model = params.d_model;
        extraction.expert_intermediate = params.expert_intermediate;
        extraction.gate_exps = params.gate_exps;
        extraction.up_exps = params.up_exps;
        extraction.down_exps = params.down_exps;
        extraction.expert_mask = params.expert_mask;
        if (!MoEExpertComputeStage::extractExpertViews(extraction))
            throw std::invalid_argument("CPU expert fixture cannot extract its real source views");
        params.expert_gate_views = std::move(extraction.expert_gate_views);
        params.expert_up_views = std::move(extraction.expert_up_views);
        params.expert_down_views = std::move(extraction.expert_down_views);
    }
    size_t bytes = 0;
    for (int expert = 0; expert < params.num_experts; ++expert)
    {
        if (!params.expert_mask.empty() && !params.expert_mask.at(expert))
            continue;
        if constexpr (requires { params.local_expert_count; params.local_expert_start; })
        {
            if (params.expert_mask.empty() && params.local_expert_count >= 0 &&
                (expert < params.local_expert_start ||
                 expert >= params.local_expert_start + params.local_expert_count))
                continue;
        }
        for (const auto *views : {&params.expert_gate_views, &params.expert_up_views, &params.expert_down_views})
        {
            const auto &view = views->at(expert);
            if (!view)
                throw std::invalid_argument("CPU expert fixture lacks an active projection view");
            size_t extent = cpuPreparedStoreTestBytes(*view);
            const auto *native = IINT8Unpackable::fromTensor(view.get());
            if (!native || !native->vnniFormatInfo())
                extent = CPUWeightStoragePlacement::local().requiredAllocationBytes<uint8_t>(view->size_bytes());
            if (extent > std::numeric_limits<size_t>::max() - bytes)
                throw std::overflow_error("CPU expert fixture byte count overflows");
            bytes += extent;
        }
    }
    const size_t capacity = std::max(size_t{1}, bytes);
    PhysicalMemoryPlanBuilder plan;
    plan.add({0, params.device_id, capacity, capacity}, owner, bytes);
    params.prepared_store->installPhysicalMemoryAuthority(
        std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0));
    return retained;
}
}
