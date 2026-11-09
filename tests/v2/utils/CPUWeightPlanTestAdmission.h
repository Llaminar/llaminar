/**
 * @file CPUWeightPlanTestAdmission.h
 * @brief Genuine immutable admission for fresh CPU model-plan Unit fixtures.
 *
 * The fixture declares all final source projections before materialization.
 * It calls the same source projector, representation policy, native final-root
 * geometry and bounded consumer quote as production. This helper owns no live
 * ledger; actual preparation and release remain exclusively PMA-owned.
 */
#pragma once
#include "loaders/WeightManager.h"
#include "loaders/CPUWeightRequirementOwner.h"
#include "tensors/NativeWeightSourceRange.h"
#include "loaders/PreparedWeightRepresentationContract.h"
#include "loaders/PreparedWeightIdentityContract.h"
#include "kernels/cpu/gemm/CPUWeightPreparationMemory.h"
#include "kernels/cpu/gemm/CPUWeightPreparationGeometry.h"
#include <map>
#include <set>
#include <tuple>
#include <initializer_list>

namespace llaminar2::test {
/**
 * @brief Admit the complete concurrently retained CPU source set for this fixture.
 * @param manager Actual original weight manager, before any fresh materialization.
 * @param loader Genuine tensor-backed reader supplying original metadata/placement.
 * @param plans Complete retained plan set, including any later independent Store.
 * @param original_world_rank Original declared source/participant rank, including non-root Unit metadata fixtures.
 * @throws std::logic_error If admission would replace an existing live authority.
 */
inline void admitCPUWeightPlansForTest(WeightManager& manager,IModelLoader& loader,
    std::initializer_list<const WeightPlan*> plans,int original_world_rank=0)
{
    if(original_world_rank<0)throw std::invalid_argument("CPU plan fixture requires its original nonnegative rank");
    if(manager.physicalMemoryAuthority())
        throw std::logic_error("CPU plan fixture must declare its complete source set before first materialization");
    using Key=std::pair<DeviceId,PhysicalMemoryOwner>;
    std::map<Key,size_t> persistent;
    std::map<DeviceId,size_t> staging;
    // A tied floating resident/engine aliases the same final root. Native raw
    // residency and NativeQuantised packing are distinct physical representations.
    // Original participant scopes also distinguish actual replicated roots;
    // equal source bytes on different TP/overlay participants are not one owner.
    std::set<std::tuple<uint64_t,DeviceId,PhysicalMemoryOwner,std::string,size_t,size_t,size_t,size_t,size_t,int,bool,
        int,int,int,std::string,int,int,size_t,std::vector<size_t>>> roots;
    const auto add=[](size_t& total,size_t bytes) {
        if(bytes>std::numeric_limits<size_t>::max()-total)
            throw std::overflow_error("CPU plan fixture physical extent overflows");
        total+=bytes;
    };
    for(const auto* plan:plans) {
        if(!plan)throw std::invalid_argument("CPU plan fixture received absent original plan");
        for(const auto& requirement:plan->requirements()) {
            if(!requirement.target_device.is_cpu())continue;
            const auto& name=requirement.source_name.empty()?requirement.canonical_name:requirement.source_name;
            if(!loader.hasTensor(name))continue; // Original required/missing behavior remains materializer-owned.
            const auto original=loader.nativeWeightSourceGeometry(name);
            const auto placement=loader.nativeWeightSourcePlacement();
            if(placement!=CPUStoragePlacement::local())
                throw std::invalid_argument("This device-free source fixture requires its actual local heap placement");
            const auto role=requirement.role==WeightRole::Other?inferWeightRole(requirement.canonical_name):requirement.role;
            const auto owner=resolveCPUWeightRequirementOwner(requirement,plan->physicalMemoryOwner());
            const auto execution=PreparedWeightRepresentationContract::resolve(role,original.original_type,
                requirement.canonical_name,loader.architecture(),debugEnv().gemm.gdn_projection_arithmetic,
                debugEnv().gemm.moe_router_arithmetic);
            const auto operand=PreparedWeightRepresentationContract::operandPrecision(role,requirement.canonical_name,
                loader.architecture(),debugEnv().gemm.gdn_projection_arithmetic,debugEnv().gemm.moe_router_arithmetic);
            const auto precision=PreparedWeightRepresentationContract::resolveCPUWeightPrecision(
                role,original.original_type,WeightPrecision::NATIVE,execution,operand);
            const bool expert=original.original_shape.size()==3 && isRoutedExpertRole(role);
            const auto classification=PreparedWeightIdentityContract::classifyOriginalMatrix(role,DeviceId::cpu());
            const bool gemm=expert || (original.original_shape.size()==2 &&
                (classification==PreparedGemmClassification::Gemm ||
                (classification==PreparedGemmClassification::RequiresSchema && manager.isGemmWeight(requirement.canonical_name))));
            for(const auto& projection:projectNativeWeightSource(requirement,original)) {
                const auto& range=projection.range;
                auto quote=quoteCPUWeightPreparation(original.original_type,projection.finalRows(),projection.finalColumns(),
                    precision,gemm?CPUWeightConsumer::PreparedGemm:CPUWeightConsumer::ResidentTensor,operand);
                std::vector<size_t> assembly_key;
                if(projection.assembly) {
                    const auto& assembly=*projection.assembly;
                    assembly_key={size_t(assembly.axis),assembly.final_rows,assembly.final_columns};
                    size_t largest=0;
                    forEachNativeWeightSourceStripe(original,range,projection.assembly,
                        [&](const auto& stripe,size_t,size_t) {
                            largest=std::max(largest,stripe.nativeBytes());
                            assembly_key.insert(assembly_key.end(),{stripe.original_expert,stripe.selected_expert_ordinal,
                                stripe.first_row,stripe.rows,stripe.first_column,stripe.columns,stripe.native_row_bytes});
                        });
                    if(quote) {
                        if(largest%quote->native_bytes_per_unit)
                            throw std::invalid_argument("CPU test assembly is not aligned to the actual writer unit");
                        quote->maximum_units=largest/quote->native_bytes_per_unit;
                    }
                }
                if(quote)staging[requirement.target_device]=std::max(staging[requirement.target_device],
                    resolveNativeWeightLoadStaging(*quote,placement,configuredNativeWeightLoadStagingBudgetBytes()).physical_bytes);
                const bool packed=gemm && precision==WeightPrecision::NATIVE &&
                    native_quantised_formats::forQuantType(tensorTypeName(original.original_type));
                auto final_type=original.original_type;
                if(precision==WeightPrecision::CONVERT_TO_FP32)final_type=TensorType::FP32;
                if(precision==WeightPrecision::CONVERT_TO_FP16)final_type=TensorType::FP16;
                if(precision==WeightPrecision::CONVERT_TO_BF16)final_type=TensorType::BF16;
                if(precision==WeightPrecision::CONVERT_TO_INT8)
                    throw std::invalid_argument("Native-policy plan fixture cannot invent an INT8 conversion request");
                const auto key=std::tuple{plan->strategy().model_id.value,requirement.target_device,owner,
                    name,range.original_expert,range.first_row,range.rows,range.first_column,range.columns,int(final_type),packed,
                    requirement.pp_stage,requirement.tp_domain,requirement.tp_rank_or_device_index,
                    requirement.overlay_domain,requirement.overlay_participant_world_rank,requirement.overlay_participant_index,
                    projection.matrix_count,assembly_key};
                if(!roots.insert(key).second)continue;
                size_t bytes=0;
                if(packed)bytes=CPUWeightPreparationMemory::sourceNative(tensorTypeName(final_type),projection.finalRows(),projection.finalColumns()).persistent_bytes;
                else {
                    size_t payload=projection.assembly ? projection.finalRows()*(projection.finalColumns()/original.native_group_elements)*original.native_group_bytes : range.nativeBytes();
                    if(final_type==TensorType::FP32 || final_type==TensorType::FP16 || final_type==TensorType::BF16) {
                        const size_t width=final_type==TensorType::FP32?sizeof(float):sizeof(uint16_t);
                        if(projection.finalRows()>std::numeric_limits<size_t>::max()/projection.finalColumns()/width)
                            throw std::overflow_error("CPU plan fixture final scalar extent overflows");
                        payload=projection.finalRows()*projection.finalColumns()*width;
                    }
                    if(projection.matrix_count>std::numeric_limits<size_t>::max()/payload)
                        throw std::overflow_error("CPU plan fixture final outer extent overflows");
                    bytes=placement.physicalBytes<uint8_t>(payload*projection.matrix_count);
                }
                add(persistent[{requirement.target_device,owner}],bytes);
            }
        }
    }
    if(persistent.empty() && staging.empty())return; // A GPU-only metadata fixture declares no CPU reader owner.
    std::map<DeviceId,size_t> totals;
    for(const auto& [key,bytes]:persistent)add(totals[key.first],bytes);
    for(const auto& [device,bytes]:staging)add(totals[device],bytes);
    PhysicalMemoryPlanBuilder budget;
    for(const auto& [device,total]:totals) {
        const PhysicalMemoryResource resource{original_world_rank,device,std::max(size_t{1},total),std::max(size_t{1},total)};
        for(const auto& [key,bytes]:persistent)if(key.first==device)budget.add(resource,key.second,bytes);
        if(staging[device])budget.add(resource,PhysicalMemoryOwner::WeightLoadStaging,staging[device]);
    }
    manager.installPhysicalMemoryAuthority(std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(budget.build()),original_world_rank));
}
}
