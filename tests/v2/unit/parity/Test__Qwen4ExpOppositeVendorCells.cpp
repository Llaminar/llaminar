/** @file Test__Qwen4ExpOppositeVendorCells.cpp
 * @brief Canonical opposite-vendor storage declaration/discovery/retained15 regression without inference.
 */
#include "integration/parity/qwen4exp/Qwen4ExpSourceAdmissionCI.h"
#include "integration/parity/ModelParityRuntimeExport.h"
#include "integration/parity/qwen4exp/Qwen4ExpSourceAdmissionEvidence.h"
#include <gtest/gtest.h>
#include <map>
#include <sstream>
#include <set>
#include <tuple>
namespace llaminar2::test::parity {
using namespace qwen4exp;
TEST(Qwen4ExpOppositeVendorCells, FourOrdinaryCasesKeepOriginalSourcesAndNumericalGates){
    EXPECT_EQ(qwen4ExpProductionCases().size(),20u);EXPECT_EQ(qwen4ExpGPUAdaptiveProductionCases().size(),4u);
    EXPECT_EQ(qwen4ExpGPUAdaptiveSourceAdmissionSeeds().size(),16u);
    const auto& cells=qwen4ExpOppositeVendorOrdinaryRetained15Cases();ASSERT_EQ(cells.size(),4u);
    const auto& reference=qwen4ExpProductionCases().front();std::set<std::string> names;
    for(const auto& old:qwen4ExpGPUAdaptiveProductionCases())names.insert(old.testName());
    for(const auto& cell:cells){
        EXPECT_TRUE(names.insert(cell.testName()).second);EXPECT_LT(cell.testName().size(),256u);
        EXPECT_EQ(cell.mtp,ModelParityMTP::Off);EXPECT_EQ(cell.retained_mtp_draft_capacity,15);
        ASSERT_TRUE(cell.model.shared_predictor);EXPECT_EQ(cell.model.shared_predictor->path(),kPredictorModelPath);
        EXPECT_EQ(cell.model.model_path,kMainModelPath);EXPECT_EQ(cell.model.transformer_layers,48);
        EXPECT_EQ(cell.model.reference_directory,reference.model.reference_directory);EXPECT_EQ(cell.model.token_ids,reference.model.token_ids);
        EXPECT_EQ(cell.activation_precision,reference.activation_precision);EXPECT_EQ(cell.kv_cache_precision,reference.kv_cache_precision);
        EXPECT_EQ(cell.prefix_restore_geometry,reference.prefix_restore_geometry);
        EXPECT_EQ(cell.graph_snapshot_policy.required_prefill_snapshot_keys,reference.graph_snapshot_policy.required_prefill_snapshot_keys);
        EXPECT_EQ(cell.graph_snapshot_policy.required_decode_snapshot_keys,reference.graph_snapshot_policy.required_decode_snapshot_keys);
        EXPECT_EQ(cell.thresholds.cosine_threshold,reference.thresholds.cosine_threshold);
        EXPECT_EQ(cell.thresholds.kl_threshold,reference.thresholds.kl_threshold);
        EXPECT_EQ(cell.thresholds.prefill_kl_threshold,reference.thresholds.prefill_kl_threshold);
        const auto& a=cell.thresholds;const auto& b=reference.thresholds;
        EXPECT_EQ(std::tie(a.cosine_threshold,a.decode_cosine_threshold,a.early_layers_count,a.min_early_layers_passed,
            a.kl_threshold,a.prefill_kl_threshold,a.mtp_kl_threshold,a.excluded_stages,a.allreduce_stages,
            a.min_top1_accuracy,a.min_top5_accuracy,a.min_decode_pass_rate,a.pytorch_top1_in_topk),
            std::tie(b.cosine_threshold,b.decode_cosine_threshold,b.early_layers_count,b.min_early_layers_passed,
            b.kl_threshold,b.prefill_kl_threshold,b.mtp_kl_threshold,b.excluded_stages,b.allreduce_stages,
            b.min_top1_accuracy,b.min_top5_accuracy,b.min_decode_pass_rate,b.pytorch_top1_in_topk));
        EXPECT_TRUE(cell.graph_snapshot_policy.require_graph_execution_on_gpu);EXPECT_TRUE(cell.graph_snapshot_policy.require_snapshot_publication);
        EXPECT_EQ(cell.expert_overlay,std::optional{ModelParityExpertOverlayPolicy{}});
        EXPECT_FALSE(cell.requiresPhysicalExpertMovement());EXPECT_FALSE(cell.e2e_certification);
    }
}
TEST(Qwen4ExpOppositeVendorCells, DiscoveryKeepsStorageSeparateFromComputeAndExpertDomains){
    std::set<std::pair<DeviceType,std::uint64_t>> axes;
    for(const auto& cell:qwen4ExpOppositeVendorOrdinaryRetained15Cases()){
        ASSERT_EQ(cell.topology.participants.size(),2u);ASSERT_TRUE(cell.topology.expert_overlay_plan);
        const auto& plan=*cell.topology.expert_overlay_plan;ASSERT_EQ(plan.domains.size(),2u);
        const auto gpu=plan.domains.front().participants.front();const auto local=gpu.isCUDA()?DeviceType::CUDA:DeviceType::ROCm;
        const auto remote=local==DeviceType::CUDA?DeviceType::ROCm:DeviceType::CUDA;
        EXPECT_EQ(plan.domains.front().participants.size(),1u);EXPECT_EQ(plan.domains.back().participants,(std::vector{GlobalDeviceAddress::cpu(0)}));
        ASSERT_TRUE(cell.ngram_table_policy);const auto& placement=cell.ngram_table_policy->gpu_adaptive.placement;
        ASSERT_EQ(placement.owners.size(),2u);EXPECT_EQ(placement.owners[0].participant.backend,local);EXPECT_EQ(placement.owners[1].participant.backend,remote);
        ASSERT_TRUE(placement.coordinator);EXPECT_EQ(placement.coordinator->backend,local);
        axes.emplace(local,placement.owners.front().maximum_cache_rows);
        std::ostringstream out;PrintTo(Qwen4ExpSourceAdmissionCase(cell),&out);const auto metadata=nlohmann::json::parse(out.str());
        EXPECT_EQ(metadata["participants"].size(),2u);EXPECT_EQ(metadata["storage_participants"].size(),2u);
        EXPECT_FALSE(metadata["mtp_enabled"].get<bool>());EXPECT_EQ(metadata["retained_mtp_draft_capacity"],15);
        EXPECT_NE(cell.testName().find("CUDA"),std::string::npos);EXPECT_NE(cell.testName().find("ROCm"),std::string::npos);
    }
    EXPECT_EQ(axes.size(),4u);
}
TEST(Qwen4ExpOppositeVendorCells, SourceAndNumericalProjectionKeepInactiveFullFifteen){
    for(const auto& cell:qwen4ExpOppositeVendorOrdinaryRetained15Cases()){
        const auto source=qwen4ExpSourceAdmissionConfig(cell,cell.model.model_path,0);
        const auto numerical=cell.makeOrchestrationConfig(cell.model.model_path,0);
        EXPECT_FALSE(source.mtp.enabled);EXPECT_EQ(source.mtp.graph_capacity_draft_tokens,15);
        EXPECT_EQ(source.mtp.graph_capacity_draft_tokens,numerical.mtp.graph_capacity_draft_tokens);
        EXPECT_EQ(source.shared_predictor,cell.model.shared_predictor);
        EXPECT_EQ(source.ngram_table_policy,cell.ngram_table_policy);
        EXPECT_NO_THROW(requireQwen4ExpSourceMaintenancePolicy(numerical.moe_rebalance,source.moe_rebalance));
        EXPECT_THROW(qwen4ExpExpandSourceAdmissionMTP(std::span{&cell,std::size_t{1}}),std::invalid_argument);
    }
}
TEST(Qwen4ExpOppositeVendorCells, ActivePoliciesRetainOriginalStorageAndSourceCapacity){
    const auto& cells=qwen4ExpOppositeVendorMTPRetained15Cases();ASSERT_EQ(cells.size(),12u);
    std::set<std::tuple<DeviceType,std::uint64_t,ModelParityMTP>> axes;
    for(const auto& cell:cells){
        EXPECT_TRUE(cell.mtp==ModelParityMTP::Depth1 || cell.mtp==ModelParityMTP::Depth3 ||
                    cell.mtp==ModelParityMTP::DynamicDepth);
        EXPECT_EQ(cell.retained_mtp_draft_capacity,15);EXPECT_EQ(cell.model.maximum_mtp_draft_depth,15);
        ASSERT_TRUE(cell.model.shared_predictor);EXPECT_EQ(cell.model.shared_predictor->path(),kPredictorModelPath);
        EXPECT_EQ(cell.model.model_path,kMainModelPath);EXPECT_EQ(cell.model.transformer_layers,48);
        EXPECT_EQ(cell.expert_overlay,std::optional{ModelParityExpertOverlayPolicy{}});
        EXPECT_FALSE(cell.requiresPhysicalExpertMovement());EXPECT_FALSE(cell.e2e_certification);
        const auto config=cell.makeOrchestrationConfig(cell.model.model_path,0);
        const auto source=qwen4ExpSourceAdmissionConfig(cell,cell.model.model_path,0);
        EXPECT_TRUE(config.mtp.enabled);EXPECT_TRUE(source.mtp.enabled);
        EXPECT_EQ(config.mtp.graph_capacity_draft_tokens,15);EXPECT_EQ(source.mtp.graph_capacity_draft_tokens,15);
        EXPECT_EQ(config.mtp.depth_policy.mode,cell.mtp==ModelParityMTP::DynamicDepth?
            MTPDepthPolicyMode::Dynamic:MTPDepthPolicyMode::Fixed);
        if(cell.mtp==ModelParityMTP::DynamicDepth){
            // Numerical/generation diagnostics deliberately start at full
            // depth; source admission retains production automatic intent.
            EXPECT_EQ(config.mtp.depth_policy.initial_depth,15);EXPECT_EQ(config.mtp.depth_policy.max_depth,15);
            EXPECT_EQ(source.mtp.depth_policy.initial_depth,0);EXPECT_EQ(source.mtp.depth_policy.max_depth,15);
        }else EXPECT_EQ(config.mtp.draft_tokens,cell.mtp==ModelParityMTP::Depth1?1:3);
        ASSERT_TRUE(cell.ngram_table_policy);const auto& placement=cell.ngram_table_policy->gpu_adaptive.placement;
        ASSERT_EQ(placement.owners.size(),2u);ASSERT_TRUE(placement.coordinator);
        EXPECT_TRUE(axes.emplace(placement.coordinator->backend,placement.owners[0].maximum_cache_rows,cell.mtp).second);
    }
    EXPECT_EQ(axes.size(),12u);
}
TEST(Qwen4ExpOppositeVendorCells, ActiveRuntimeExportsNameOriginalOrdinaryControls){
    std::map<std::string,ModelParityCase> controls;
    for(const auto& cell:qwen4ExpOppositeVendorOrdinaryRetained15Cases())controls.emplace(cell.testName(),cell);
    for(const auto& cell:qwen4ExpOppositeVendorMTPRetained15Cases()){
        std::ostringstream out;PrintTo(cell,&out);const auto record=nlohmann::json::parse(out.str());
        const auto& generation=record.at("runtime").at("generation");
        EXPECT_EQ(generation.at("mtp_policy"),modelParityMTPPolicyName(cell.mtp));
        EXPECT_EQ(generation.at("mtp_verify_mode"),"speculative-sampling");
        const auto found=controls.find(generation.at("serial_control_id").get<std::string>());
        ASSERT_NE(found,controls.end());
        EXPECT_EQ(cell.ngram_table_policy,found->second.ngram_table_policy);
        EXPECT_EQ(cell.retained_mtp_draft_capacity,found->second.retained_mtp_draft_capacity);
        EXPECT_EQ(cell.topology.test_id,found->second.topology.test_id);
        EXPECT_EQ(cell.activation_precision,found->second.activation_precision);
        EXPECT_EQ(cell.kv_cache_precision,found->second.kv_cache_precision);
        EXPECT_EQ(cell.graph_snapshot_policy.required_decode_snapshot_keys,found->second.graph_snapshot_policy.required_decode_snapshot_keys);
        EXPECT_TRUE(record.at("e2e").is_null());
    }
}
/** @test Constructing separate movement families cannot mutate any of the original sixteen static case bytes. */
TEST(Qwen4ExpOppositeVendorCells,DynamicExpertFamiliesPreserveOriginalStaticSixteen){
    const auto staticRecords=[](){std::vector<std::string> records;
        for(const auto* family:{&qwen4ExpOppositeVendorOrdinaryRetained15Cases(),&qwen4ExpOppositeVendorMTPRetained15Cases()})
            for(const auto& cell:*family){std::ostringstream out;PrintTo(cell,&out);records.push_back(out.str());}
        return records;};
    const auto before=staticRecords();ASSERT_EQ(before.size(),16u);
    for(const auto order:{RoutedExpertOwnerOrder::Ordinal,RoutedExpertOwnerOrder::Random}){
        EXPECT_EQ(qwen4ExpOppositeVendorDynamicExpertRetained15Cases(order).size(),16u);
        EXPECT_EQ(qwen4ExpOppositeVendorDynamicExpertMTPRetained15Cases(order).size(),12u);
    }
    EXPECT_EQ(staticRecords(),before);
    EXPECT_THROW((void)qwen4ExpOppositeVendorDynamicExpertRetained15Cases(static_cast<RoutedExpertOwnerOrder>(99)),std::invalid_argument);
}
/** @test Actual required expert movement is independent of predictor depth and opposite-vendor storage membership. */
TEST(Qwen4ExpOppositeVendorCells,DynamicExpertProductRetainsMovementAndOriginalStorageRoles){
    std::set<std::tuple<RoutedExpertOwnerOrder,DeviceType,std::uint64_t,ModelParityMTP>> axes;
    std::size_t off=0,active=0,dynamicDepth=0;
    const auto& reference=qwen4ExpOppositeVendorOrdinaryRetained15Cases().front();
    for(const auto order:{RoutedExpertOwnerOrder::Ordinal,RoutedExpertOwnerOrder::Random})
    for(const auto& cell:qwen4ExpOppositeVendorDynamicExpertRetained15Cases(order)){
        ASSERT_TRUE(cell.expert_overlay);EXPECT_EQ(cell.expert_overlay->owner_order,order);
        EXPECT_EQ(cell.expert_overlay->movement,ModelParityExpertMovement::Dynamic);
        EXPECT_TRUE(cell.requiresPhysicalExpertMovement());EXPECT_FALSE(cell.requiresNoExpertMovement());
        EXPECT_EQ(cell.dynamic_evidence,ModelParityDynamicEvidence::EconomicMovement);
        EXPECT_FALSE(cell.requiresObservedConvergenceSpeedup());EXPECT_FALSE(cell.e2e_certification);
        EXPECT_EQ(cell.retained_mtp_draft_capacity,15);EXPECT_EQ(cell.model.maximum_mtp_draft_depth,15);
        ASSERT_TRUE(cell.model.shared_predictor);EXPECT_EQ(cell.model.shared_predictor->path(),kPredictorModelPath);
        EXPECT_EQ(cell.model.model_path,kMainModelPath);EXPECT_EQ(cell.model.token_ids,reference.model.token_ids);
        EXPECT_EQ(cell.graph_snapshot_policy.required_decode_snapshot_keys,reference.graph_snapshot_policy.required_decode_snapshot_keys);
        const auto& a=cell.thresholds;const auto& b=reference.thresholds;
        EXPECT_EQ(std::tie(a.cosine_threshold,a.decode_cosine_threshold,a.early_layers_count,a.min_early_layers_passed,
            a.kl_threshold,a.prefill_kl_threshold,a.mtp_kl_threshold,a.excluded_stages,a.allreduce_stages,
            a.min_top1_accuracy,a.min_top5_accuracy,a.min_decode_pass_rate,a.pytorch_top1_in_topk),
            std::tie(b.cosine_threshold,b.decode_cosine_threshold,b.early_layers_count,b.min_early_layers_passed,
            b.kl_threshold,b.prefill_kl_threshold,b.mtp_kl_threshold,b.excluded_stages,b.allreduce_stages,
            b.min_top1_accuracy,b.min_top5_accuracy,b.min_decode_pass_rate,b.pytorch_top1_in_topk));
        ASSERT_TRUE(cell.ngram_table_policy);const auto& placement=cell.ngram_table_policy->gpu_adaptive.placement;
        ASSERT_TRUE(placement.coordinator);ASSERT_EQ(placement.owners.size(),2u);
        const auto local=placement.coordinator->backend,remote=placement.owners[1].participant.backend;
        EXPECT_NE(local,remote);EXPECT_EQ(placement.owners[0].participant.backend,local);
        ASSERT_TRUE(cell.topology.expert_overlay_plan);const auto& expert=*cell.topology.expert_overlay_plan;
        ASSERT_EQ(expert.domains.size(),2u);EXPECT_EQ(expert.routed_tiers.front().max_experts_per_layer,8u);
        EXPECT_EQ(expert.domains.front().participants.size(),1u);
        EXPECT_EQ(expert.domains.back().participants,(std::vector{GlobalDeviceAddress::cpu(0)}));
        EXPECT_EQ(cell.topology.participants.size(),2u);
        for(const auto& domain:expert.domains)for(const auto& participant:domain.participants)
            EXPECT_NE(participant.device_type,remote); // Storage-only GPU is never a movement destination.
        const auto config=cell.makeOrchestrationConfig(cell.model.model_path,0);
        EXPECT_EQ(config.moe_rebalance.mode,MoERebalanceRuntimeMode::Dynamic);
        EXPECT_EQ(config.routed_expert_owner_order,order);
        ASSERT_TRUE(config.moe_routed_expert_plan);
        EXPECT_EQ(config.moe_routed_expert_plan->residency_policy,RoutedExpertResidencyPolicy::RoutedTierRebalanced);
        EXPECT_EQ(config.moe_rebalance.device_min_maintenance_period_tokens,16);
        EXPECT_EQ(config.moe_rebalance.device_initial_maintenance_period_tokens,16);
        EXPECT_EQ(config.mtp.graph_capacity_draft_tokens,15);EXPECT_EQ(config.mtp.enabled,cell.mtp!=ModelParityMTP::Off);
        if(cell.mtp==ModelParityMTP::Off)++off;else ++active;
        if(cell.mtp==ModelParityMTP::DynamicDepth){++dynamicDepth;EXPECT_EQ(config.mtp.depth_policy.mode,MTPDepthPolicyMode::Dynamic);EXPECT_EQ(config.mtp.depth_policy.initial_depth,15);}
        else if(cell.mtp!=ModelParityMTP::Off)EXPECT_EQ(config.mtp.draft_tokens,cell.mtp==ModelParityMTP::Depth1?1:3);
        EXPECT_TRUE(axes.emplace(order,local,placement.owners[0].maximum_cache_rows,cell.mtp).second);
    }
    EXPECT_EQ(axes.size(),32u);EXPECT_EQ(off,8u);EXPECT_EQ(active,24u);EXPECT_EQ(dynamicDepth,8u);
}
/** @test Existing runtime export pairs active lanes with the SAME Dynamic expert/owner-order Off control. */
TEST(Qwen4ExpOppositeVendorCells,DynamicExpertRuntimeExportKeepsMatchedMovementOffControls){
    for(const auto order:{RoutedExpertOwnerOrder::Ordinal,RoutedExpertOwnerOrder::Random}){
        std::map<std::string,ModelParityCase> controls;
        for(const auto& cell:qwen4ExpOppositeVendorDynamicExpertRetained15Cases(order))
            if(cell.mtp==ModelParityMTP::Off)controls.emplace(cell.testName(),cell);
        ASSERT_EQ(controls.size(),4u);
        for(const auto& cell:qwen4ExpOppositeVendorDynamicExpertMTPRetained15Cases(order)){
            std::ostringstream out;PrintTo(cell,&out);const auto record=nlohmann::json::parse(out.str());
            const auto found=controls.find(record.at("runtime").at("generation").at("serial_control_id").get<std::string>());
            ASSERT_NE(found,controls.end());const auto& control=found->second;
            EXPECT_EQ(control.expert_overlay,cell.expert_overlay);EXPECT_TRUE(control.requiresPhysicalExpertMovement());
            EXPECT_EQ(control.ngram_table_policy,cell.ngram_table_policy);EXPECT_EQ(control.retained_mtp_draft_capacity,15);
            auto expected=cell;expected.mtp=ModelParityMTP::Off;
            EXPECT_EQ(serializeOrchestrationConfig(expected.makeOrchestrationConfig(cell.model.model_path,0)),
                      serializeOrchestrationConfig(control.makeOrchestrationConfig(control.model.model_path,0)));
            EXPECT_FALSE(control.makeOrchestrationConfig(control.model.model_path,0).mtp.enabled);
            EXPECT_EQ(control.makeOrchestrationConfig(control.model.model_path,0).moe_rebalance.mode,MoERebalanceRuntimeMode::Dynamic);
            EXPECT_TRUE(record.at("e2e").is_null());
        }
    }
}

}
