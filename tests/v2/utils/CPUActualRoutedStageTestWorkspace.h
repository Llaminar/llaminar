/**
 * @file CPUActualRoutedStageTestWorkspace.h
 * @brief Finalized production router/expert ownership for device-free fixtures.
 *
 * Setup oracles may expose routes for sparse fixture edits. Actual paired
 * execution collects the full manifest, binds both stages to the same backing,
 * then executes the producer and validates its original routes before edits.
 */
#pragma once
#include "CPUProjectionTestWorkspace.h"
#include "execution/compute_stages/stages/MoERoutingStage.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include <cstring>
#include <vector>
namespace llaminar2::test
{
/** @brief Authenticated setup routing retained without owning an expired graph. */
struct CPUPreparedRouterTestSource
{
    std::weak_ptr<MoERoutedPipelineKernelOwner> owner;
    MoERoutingStage::Params source;
    std::vector<float> indices, weights;
};

/**
 * @brief Own the finalized real router/expert manifest and ordered binding.
 *
 * Existing numerical fixtures may edit explicit sparse sentinels after
 * setup routing. Actual paired replay first proves the router reproduces
 * its source results in the final backing, then reapplies those edits.
 */
class CPUActualRoutedStageTestWorkspace final
{
public:
    /** @brief Bind complete roots before executing either paired stage. */
    CPUActualRoutedStageTestWorkspace(MoEExpertComputeStage &expert,
        const MoEExpertComputeStage::Params &params,
        const CPUPreparedRouterTestSource *seed, IDeviceContext *context)
        : expert_(expert)
    {
        try
        {
            auto requirements=expert.getWorkspaceRequirements(params.seq_len);
            if(seed)
            {
                auto source=seed->source;
                source.routed_pipeline_kernel_owner=params.routed_pipeline_kernel_owner;
                source.output_indices=params.routing_indices;
                source.output_weights=params.routing_weights;
                router_=std::make_unique<MoERoutingStage>(source);
                requirements.merge(router_->getWorkspaceRequirements(params.seq_len));
            }
            arena_=std::make_unique<CPUProjectionTestWorkspace>(requirements);
            if(router_) router_->bindWorkspace(arena_->get());
            expert_.bindWorkspace(arena_->get());
            if(router_)
            {
                auto *indices=dynamic_cast<FP32Tensor *>(params.routing_indices);
                auto *weights=dynamic_cast<FP32Tensor *>(params.routing_weights);
                if(!indices || !weights) throw std::logic_error("Paired router fixture requires FP32 route tensors");
                const size_t count=seed->indices.size();
                std::vector<float> edited_indices(indices->data(),indices->data()+count);
                std::vector<float> edited_weights(weights->data(),weights->data()+count);
                if(!router_->execute(context) ||
                    std::memcmp(indices->data(),seed->indices.data(),count*sizeof(float)) ||
                    std::memcmp(weights->data(),seed->weights.data(),count*sizeof(float)))
                    throw std::runtime_error("Finalized real router changed its authenticated setup routes");
                std::copy(edited_indices.begin(),edited_indices.end(),indices->mutable_data());
                std::copy(edited_weights.begin(),edited_weights.end(),weights->mutable_data());
            }
        }
        catch(...)
        {
            expert_.unbindWorkspace();
            if(router_) router_->unbindWorkspace();
            throw;
        }
    }
    /** @brief Revoke stage borrows before the exact arena retires. */
    ~CPUActualRoutedStageTestWorkspace()
    {
        expert_.unbindWorkspace();
        if(router_) router_->unbindWorkspace();
    }
private:
    MoEExpertComputeStage &expert_;
    std::unique_ptr<MoERoutingStage> router_;
    std::unique_ptr<CPUProjectionTestWorkspace> arena_;
};

}
