/** @file ImmutableRowTableAdmissionWireReplayAccess.h
 * @brief Closed original service/ordering inputs for the actual production planner and wire parser.
 * No source/certificate/cap bytes are added to the wire or production catalog.
 */
#pragma once
#include "planning/ImmutableRowTableAdmissionWire.h"
#include "planning/MemoryPlanner.h"
#include "utils/PlanningHostServiceFixture.h"
#include <algorithm>
namespace llaminar2::test {
/** @brief Declare the fixture's explicitly authored full-main lookup role before planning.
 * @param config Original fixture participant; whole VRAM and CPU consumers have no IO worker.
 */
inline void declareRecordedNgramHostServices(DevicePlanConfig& config){
    std::vector<RetainedHostServiceInstance> roles;
    if(config.device.is_gpu() && config.execution_role==DeviceExecutionMemoryRole::ContinuationGraph &&
       config.ngram_execution_policy && config.ngram_execution_policy->storage!=ImmutableRowStoragePreference::VRAM)
        roles.push_back({config.world_rank,config.ngram_execution_policy->lookupIOConsumer(config.device,
            static_cast<std::size_t>(std::max(1,config.total_shards))),RetainedHostServiceRole::ImmutableLookupIO});
    config.retained_host_services=planningHostServiceDeclarations(config.world_rank,roles);
}
/** @brief Preserve the original recorded programs while joining explicitly authored participant declarations.
 * @param configs Original source copies, before request construction invokes the production planner.
 * Missing inputs are deliberately left for the actual production validator.
 */
inline void shareRecordedNgramHostServices(std::vector<DevicePlanConfig>& configs){
    if(configs.empty() || std::any_of(configs.begin(),configs.end(),[](const auto& c){return !c.retained_host_services;}))return;
    const auto original=configs.front().retained_host_services;
    if(std::all_of(configs.begin(),configs.end(),[&](const auto& c){return c.retained_host_services==original;}))return;
    RetainedHostServiceDeclarations::Builder builder(original->worldRank());std::vector<RetainedHostServiceInstance> roles;
    std::vector<const RetainedHostServiceProgram*> originalPrograms;
    for(const auto& config:configs){
        if(config.retained_host_services->worldRank()!=original->worldRank())throw std::invalid_argument("Recorded lookup declarations mixed original ranks");
        for(const auto& role:config.retained_host_services->instances()){
            const auto& program=config.retained_host_services->program(role);
            const auto previous=std::find(roles.begin(),roles.end(),role);
            if(previous!=roles.end()){
                const auto* retained=originalPrograms[static_cast<std::size_t>(previous-roles.begin())];
                if(retained->identity()!=program.identity() || retained->resourceIdentity()!=program.resourceIdentity())
                    throw std::invalid_argument("Recorded lookup role has conflicting original program evidence");
                continue;
            }
            builder.add(role,program);roles.push_back(role);originalPrograms.push_back(&program);
        }
    }
    auto shared=std::move(builder).freeze();for(auto& config:configs)config.retained_host_services=shared;
}

/** @brief Closed local-input replay of the actual receiver reconstruction, preserving its precise validation. */
struct MemoryPlannerOrderingReplayAccess {
    /** @return Same production provision after original metadata/arithmetic/owner validation. */
    static std::shared_ptr<const ImmutableRowTableExecutionProvision> restore(
        const ModelMemoryProfile& profile,const ImmutableRowTablePolicy& requested,
        const ImmutableRowTablePlacementInput& input,std::vector<ImmutableRowTableExecutionConsumer> consumers,
        PhysicalMemoryPlan execution,PhysicalMemoryPlan required,const NativeOrderingEventBudgetProjection& projection,
        const std::shared_ptr<const RetainedHostServiceDeclarations>& originalServices={}){
        return MemoryPlanner::restoreImmutableRowTableExecutionProvisionWithOrderingProjection(profile,requested,input,
            std::move(consumers),std::move(execution),std::move(required),projection,originalServices);
    }
};
struct ImmutableRowTableAdmissionWireReplayAccess {
    /** @return Actual parser output using one sealed receiver-owned recorded input. */
    static std::vector<ImmutableRowTableAdmissionRequest> parse(std::span<const std::uint8_t> bytes,
        int rank,const ModelMemoryProfile* source,const NativeOrderingEventBudgetProjection& projection,
        const std::shared_ptr<const RetainedHostServiceDeclarations>& originalServices={}){
        return ImmutableRowTableAdmissionParser::parse(bytes,rank,source,projection,originalServices);
    }
};
}
