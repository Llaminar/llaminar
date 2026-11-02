/** @file Test__Qwen4ExpConsumerLocalTP2Execution.cpp
 * @brief Device-free canonical TP2 execution, movement and economy obligations.
 * Declarations and exporter metadata are inspected without a model/backend.
 * No absent hardware, numerical, movement or performance result is certified.
 */
#include "integration/parity/qwen4exp/Qwen4ExpSourceAdmissionCases.h"
#include "integration/parity/ModelParityRuntimeExport.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <map>
#include <set>
#include <sstream>
namespace llaminar2::test::parity {
using namespace qwen4exp;
/** @return Original complete exported record, without a runtime or file query. */
static nlohmann::json exported(const ModelParityCase& cell){std::ostringstream out;PrintTo(cell,&out);return nlohmann::json::parse(out.str());}

TEST(Qwen4ExpConsumerLocalTP2Execution, SourceProductRemainsOriginalStaticSixteen)
{
    const auto& seeds=qwen4ExpConsumerLocalTP2AdaptiveSourceAdmissionSeeds();
    ASSERT_EQ(seeds.size(),4u);
    EXPECT_EQ(qwen4ExpExpandSourceAdmissionMTP(seeds).size(),16u);
    for(const auto& seed:seeds){
        const auto backend=seed.topology.expert_overlay_plan->domains.front().participants.front().isCUDA()?DeviceType::CUDA:DeviceType::ROCm;
        const std::array ordinals{0,1};
        auto previous=qwen4ExpProductionDefinition(Qwen4ExpParityTopology::CPU);
        previous.topology=qwen4ExpFixedHeadParityTopology(backend,ordinals);
        previous.topology.test_id+="_ConsumerLocal";
        previous.features.ngram_storage=qwen4ExpGPUAdaptiveStorageAxes();
        bool found=false;
        for(const auto& cell:expandModelParityDefinition(previous))if(cell.testName()==seed.testName()){
            EXPECT_EQ(exported(cell),exported(seed));found=true;
        }
        EXPECT_TRUE(found);
    }
}

TEST(Qwen4ExpConsumerLocalTP2Execution, ActualExpertMovementAndEconomyRemainSeparateFromMTP)
{
    const std::array policies{ModelParityExpertOverlayPolicy{},
        ModelParityExpertOverlayPolicy{.owner_order=RoutedExpertOwnerOrder::Ordinal,.movement=ModelParityExpertMovement::Dynamic},
        ModelParityExpertOverlayPolicy{.owner_order=RoutedExpertOwnerOrder::Random,.movement=ModelParityExpertMovement::Dynamic}};
    for(const auto policy:policies){
        const auto& cells=qwen4ExpConsumerLocalTP2ExecutionCases(policy);
        ASSERT_EQ(cells.size(),16u);
        EXPECT_EQ(qwen4ExpConsumerLocalTP2NativeExactCases(policy).size(),12u);
        std::set<std::string> names;
        std::map<std::string,ModelParityCase> controls;
        for(const auto& cell:cells)if(cell.mtp==ModelParityMTP::Off)controls.emplace(cell.testName(),cell);
        ASSERT_EQ(controls.size(),4u);
        for(const auto& cell:cells){
            EXPECT_TRUE(names.insert(cell.testName()).second);
            EXPECT_EQ(cell.expert_overlay,std::optional{policy});
            EXPECT_EQ(cell.retained_mtp_draft_capacity,15);
            ASSERT_TRUE(cell.ngram_table_policy);
            EXPECT_EQ(cell.ngram_table_policy->gpu_adaptive.placement.mode,AdaptiveGPUCachePlacementMode::ConsumerLocal);
            const auto& plan=*cell.topology.expert_overlay_plan;
            ASSERT_EQ(plan.domains.front().participants.size(),2u);
            EXPECT_EQ(plan.domains.back().participants,(std::vector{GlobalDeviceAddress::cpu(0)}));
            EXPECT_EQ(plan.routed_tiers.front().max_experts_per_layer,8);
            const bool dynamic=policy.movement==ModelParityExpertMovement::Dynamic;
            EXPECT_EQ(cell.requiresPhysicalExpertMovement(),dynamic);
            EXPECT_EQ(cell.requiresObservedConvergenceSpeedup(),dynamic&&cell.mtp==ModelParityMTP::Off);
            EXPECT_FALSE(cell.e2e_certification);
            EXPECT_FALSE(cell.paired_workload); // No singleton panel/profile can certify this new TP2 case.
            const auto config=cell.makeOrchestrationConfig(cell.model.model_path,0);
            if(dynamic){
                EXPECT_EQ(config.moe_rebalance.device_min_maintenance_period_tokens,16u);
                EXPECT_EQ(config.moe_rebalance.device_initial_maintenance_period_tokens,16u);
            }
            const auto record=exported(cell);
            const auto id=record.at("runtime").at("generation").at("serial_control_id").get<std::string>();
            ASSERT_TRUE(controls.contains(id))<<id;
            const auto& control=controls.at(id);
            EXPECT_EQ(cell.topology.test_id,control.topology.test_id);
            EXPECT_EQ(cell.ngram_table_policy,control.ngram_table_policy);
            EXPECT_EQ(cell.expert_overlay,control.expert_overlay);
            EXPECT_EQ(cell.model.token_ids,control.model.token_ids);
            EXPECT_EQ(cell.thresholds.cosine_threshold,control.thresholds.cosine_threshold);
            EXPECT_EQ(cell.thresholds.kl_threshold,control.thresholds.kl_threshold);
            EXPECT_FALSE(modelParityDiagnosticBenchmarkArguments(cell).empty());
            EXPECT_FALSE(modelParityServerArguments(cell).empty());
            EXPECT_FALSE(record.at("diagnostic_benchmark").at("certification_eligible").get<bool>());
        }
    }
}
}
