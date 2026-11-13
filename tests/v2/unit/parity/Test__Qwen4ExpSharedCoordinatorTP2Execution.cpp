/**
 * @file Test__Qwen4ExpSharedCoordinatorTP2Execution.cpp
 * @brief Device-free shared-source execution, serial-control and public CLI contracts.
 *
 * The canonical source definition owns every participant and storage axis.
 * These tests prevent source-only admission or an unrelated panel from being
 * mistaken for execution qualification. They load no model or device runtime.
 */
#include "integration/parity/qwen4exp/Qwen4ExpSourceAdmissionCases.h"
#include "integration/parity/qwen4exp/Qwen4ExpLocalPairedCases.h"
#include "integration/parity/ModelParityRuntimeExport.h"
#include "config/OrchestrationConfigParser.h"
#include <gtest/gtest.h>
#include <map>
#include <set>
#include <sstream>
#include <tuple>

namespace llaminar2::test::parity
{
namespace
{
using namespace qwen4exp;
constexpr std::array kSharedExpertPolicies{
    ModelParityExpertOverlayPolicy{},
    ModelParityExpertOverlayPolicy{.owner_order=RoutedExpertOwnerOrder::Ordinal,.movement=ModelParityExpertMovement::Dynamic},
    ModelParityExpertOverlayPolicy{.owner_order=RoutedExpertOwnerOrder::Random,.movement=ModelParityExpertMovement::Dynamic}};

/** @param cell Original canonical declaration.
 * @return The actual runtime export without accessing source or native state. */
nlohmann::json exportedShared(const ModelParityCase& cell)
{
    std::ostringstream out;
    PrintTo(cell,&out);
    return nlohmann::json::parse(out.str());
}

/** @param arguments Original exported argv vector, without an executable name.
 * @return A public CLI configuration without initializing the inference runtime. */
OrchestrationConfig parseSharedArguments(std::vector<std::string> arguments)
{
    arguments.insert(arguments.begin(),"llaminar2");
    std::vector<char*> argv;
    for(auto& argument:arguments)argv.push_back(argument.data());
    return OrchestrationConfigParser{}.parseArgs(static_cast<int>(argv.size()),argv.data());
}
}

/** @test Source backing expansion preserves original execution families and Off/active capacity. */
TEST(Qwen4ExpSharedCoordinatorTP2Execution, OriginalSourceAdmissionKeepsOffZeroAndActiveFifteen)
{
    const auto& seeds=qwen4ExpSharedCoordinatorTP2AdaptiveSourceAdmissionSeeds();
    ASSERT_EQ(seeds.size(),4u);
    const auto original=qwen4ExpExpandSourceAdmissionMTP(seeds);
    ASSERT_EQ(original.size(),16u);
    std::vector<nlohmann::json> before;
    for(const auto& cell:original){
        before.push_back(exportedShared(cell));
        EXPECT_EQ(cell.retained_mtp_draft_capacity,cell.mtp==ModelParityMTP::Off?0:15);
        EXPECT_EQ(cell.expert_overlay,std::optional{ModelParityExpertOverlayPolicy{}});
    }
    for(const auto policy:kSharedExpertPolicies)
        EXPECT_EQ(qwen4ExpSharedCoordinatorTP2ExecutionCases(policy).size(),8u);
    const auto after=qwen4ExpExpandSourceAdmissionMTP(seeds);
    ASSERT_EQ(after.size(),before.size());
    for(std::size_t i=0;i<after.size();++i)EXPECT_EQ(exportedShared(after[i]),before[i]);
}

/** @test One exact compute domain shares GPU2 storage without turning that owner into a model rank. */
TEST(Qwen4ExpSharedCoordinatorTP2Execution, AllAxesRetainOriginalTopologyMathAndMovementObligations)
{
    const auto& reference=qwen4ExpProductionCases().front();
    std::set<std::string> names;
    for(const auto policy:kSharedExpertPolicies){
        std::set<std::tuple<std::uint64_t,ModelParityMTP>> axes;
        for(const auto& cell:qwen4ExpSharedCoordinatorTP2ExecutionCases(policy)){
            SCOPED_TRACE(cell.testName());
            EXPECT_TRUE(names.insert(cell.testName()).second);
            EXPECT_EQ(cell.model.model_path,kMainModelPath);
            EXPECT_EQ(cell.model.transformer_layers,48);
            ASSERT_TRUE(cell.model.shared_predictor);
            EXPECT_EQ(cell.model.shared_predictor->path(),kPredictorModelPath);
            EXPECT_EQ(cell.retained_mtp_draft_capacity,15);
            EXPECT_EQ(cell.activation_precision,reference.activation_precision);
            EXPECT_EQ(cell.kv_cache_precision,reference.kv_cache_precision);
            EXPECT_EQ(cell.model.token_ids,reference.model.token_ids);
            EXPECT_EQ(cell.graph_snapshot_policy.required_prefill_snapshot_keys,reference.graph_snapshot_policy.required_prefill_snapshot_keys);
            EXPECT_EQ(cell.graph_snapshot_policy.required_decode_snapshot_keys,reference.graph_snapshot_policy.required_decode_snapshot_keys);
            EXPECT_EQ(cell.thresholds.kl_threshold,reference.thresholds.kl_threshold);
            EXPECT_EQ(cell.thresholds.prefill_kl_threshold,reference.thresholds.prefill_kl_threshold);
            EXPECT_EQ(cell.thresholds.min_top1_accuracy,reference.thresholds.min_top1_accuracy);
            EXPECT_EQ(cell.thresholds.min_top5_accuracy,reference.thresholds.min_top5_accuracy);
            ASSERT_TRUE(cell.topology.expert_overlay_plan);
            const auto& plan=*cell.topology.expert_overlay_plan;
            EXPECT_EQ(plan.domains.front().participants,(std::vector{GlobalDeviceAddress::rocm(0),GlobalDeviceAddress::rocm(1)}));
            EXPECT_EQ(plan.domains.back().participants,(std::vector{GlobalDeviceAddress::cpu(0)}));
            EXPECT_EQ(plan.routed_tiers.front().max_experts_per_layer,8);
            ASSERT_TRUE(cell.ngram_table_policy);
            const auto& placement=cell.ngram_table_policy->gpu_adaptive.placement;
            EXPECT_EQ(placement.mode,AdaptiveGPUCachePlacementMode::DeclaredRankLocal);
            EXPECT_EQ(placement.coordinator,(std::optional{AdaptiveGPUCacheParticipant{0,DeviceType::ROCm,0}}));
            ASSERT_EQ(placement.owners.size(),1u);
            EXPECT_EQ(placement.owners.front().participant,(AdaptiveGPUCacheParticipant{0,DeviceType::ROCm,2}));
            const auto rows=placement.owners.front().maximum_cache_rows;
            EXPECT_TRUE(rows==0 || rows==64);
            EXPECT_TRUE(axes.emplace(rows,cell.mtp).second);
            EXPECT_EQ(cell.expert_overlay,std::optional{policy});
            const bool dynamic=policy.movement==ModelParityExpertMovement::Dynamic;
            EXPECT_EQ(cell.requiresPhysicalExpertMovement(),dynamic);
            EXPECT_EQ(cell.requiresObservedConvergenceSpeedup(),dynamic&&cell.mtp==ModelParityMTP::Off);
            EXPECT_FALSE(cell.e2e_certification);
            EXPECT_FALSE(cell.paired_workload); // A different topology's panel cannot certify this source group.
        }
        EXPECT_EQ(axes.size(),8u);
    }
    EXPECT_EQ(names.size(),24u);
    EXPECT_THROW((void)qwen4ExpSharedCoordinatorTP2ExecutionCases(
        ModelParityExpertOverlayPolicy{.owner_order=RoutedExpertOwnerOrder::Random}),std::invalid_argument);
}

/** @test Exact replay and HTTP exports retain their own Off control and never borrow another storage family. */
TEST(Qwen4ExpSharedCoordinatorTP2Execution, ExactReplayAndPublicRuntimeKeepMatchingSerialControls)
{
    for(const auto policy:kSharedExpertPolicies){
        const auto& cells=qwen4ExpSharedCoordinatorTP2ExecutionCases(policy);
        std::map<std::string,const ModelParityCase*> originals;
        std::map<std::string,const ModelParityCase*> controls;
        for(const auto& cell:cells){
            originals.emplace(cell.testName(),&cell);
            if(cell.mtp==ModelParityMTP::Off)controls.emplace(cell.testName(),&cell);
        }
        ASSERT_EQ(controls.size(),2u);
        const auto exact=qwen4ExpSharedCoordinatorTP2NativeExactCases(policy);
        ASSERT_EQ(exact.size(),6u);
        for(const auto& cell:exact){
            ASSERT_TRUE(originals.contains(cell.testName()));
            EXPECT_EQ(exportedShared(cell),exportedShared(*originals.at(cell.testName())));
            EXPECT_NE(cell.mtp,ModelParityMTP::Off);
        }
        for(const auto& cell:cells){
            const auto record=exportedShared(cell);
            const auto id=record.at("runtime").at("generation").at("serial_control_id").get<std::string>();
            ASSERT_TRUE(controls.contains(id))<<id;
            const auto& control=*controls.at(id);
            EXPECT_EQ(cell.topology.test_id,control.topology.test_id);
            EXPECT_EQ(cell.ngram_table_policy,control.ngram_table_policy);
            EXPECT_EQ(cell.expert_overlay,control.expert_overlay);
            EXPECT_EQ(cell.retained_mtp_draft_capacity,control.retained_mtp_draft_capacity);
            EXPECT_FALSE(record.at("diagnostic_benchmark").at("certification_eligible").get<bool>());
            auto expected=cell.makeOrchestrationConfig(cell.model.model_path,0);
            ASSERT_TRUE(normalizeMoERoutedExpertPlacementDomains(expected).empty());
            for(const auto& arguments:{modelParityServerArguments(cell),modelParityDiagnosticBenchmarkArguments(cell)}){
                const auto actual=parseSharedArguments(arguments);
                EXPECT_EQ(actual.ngram_table_policy,expected.ngram_table_policy);
                EXPECT_EQ(actual.shared_predictor,expected.shared_predictor);
                EXPECT_EQ(actual.mtp.enabled,expected.mtp.enabled);
                EXPECT_EQ(actual.mtp.graph_capacity_draft_tokens,15);
                EXPECT_EQ(actual.mtp.depth_policy.mode,expected.mtp.depth_policy.mode);
                ASSERT_TRUE(actual.moe_routed_expert_plan);
                EXPECT_EQ(actual.moe_routed_expert_plan->domains.front().participants,
                          expected.moe_routed_expert_plan->domains.front().participants);
                EXPECT_EQ(actual.moe_routed_expert_plan->owner_order,expected.moe_routed_expert_plan->owner_order);
                EXPECT_EQ(actual.moe_rebalance.mode,expected.moe_rebalance.mode);
            }
        }
    }
    const auto families=qfnOriginalCaseFamilies();
    for(const auto name:{"Qwen4ExpSharedCoordinatorTP2StaticRetained15",
                         "Qwen4ExpSharedCoordinatorTP2DynamicOrdinalRetained15",
                         "Qwen4ExpSharedCoordinatorTP2DynamicRandomRetained15"}){
        ASSERT_TRUE(families.contains(name));
        EXPECT_EQ(families.at(name).size(),8u);
    }
}
} // namespace llaminar2::test::parity

namespace llaminar2::test::parity
{
/** @test Exactly six hardware rows retain real worker roles, explicit tier capacities and original public workload identity. */
TEST(Qwen4ExpSharedCoordinatorTP2Execution, CanonicalSixHardwareRowsAndTierPolicies)
{
    using namespace qwen4exp;
    constexpr std::array targets{Qwen4ExpHardwareTarget::ROCm1, Qwen4ExpHardwareTarget::ROCm2,
        Qwen4ExpHardwareTarget::ROCm4, Qwen4ExpHardwareTarget::CUDA1,
        Qwen4ExpHardwareTarget::CUDA2, Qwen4ExpHardwareTarget::CUDA2ROCm4};
    std::set<std::string> names;
    std::size_t configurations = 0;
    for (const auto target : targets) {
        const bool mixed = target == Qwen4ExpHardwareTarget::CUDA2ROCm4;
        for (const auto owners : {Qwen4ExpHardwareNgramOwners::Continuation,
                                 Qwen4ExpHardwareNgramOwners::Follower, Qwen4ExpHardwareNgramOwners::Both}) {
            if (!mixed && owners != Qwen4ExpHardwareNgramOwners::Continuation) {
                EXPECT_THROW(qwen4ExpHardwareTargetDefinition(target, owners), std::invalid_argument);
                continue;
            }
            ++configurations;
            const auto definition = qwen4ExpHardwareTargetDefinition(target, owners);
            ASSERT_EQ(definition.features.ngram_storage.size(), 7u);
            ASSERT_TRUE(definition.topology.expert_overlay_plan);
            const auto& plan = *definition.topology.expert_overlay_plan;
            EXPECT_EQ(plan.routed_tiers.front().max_experts_per_layer, 8);
            EXPECT_TRUE(plan.routed_tiers.back().fallback);
            if (mixed) {
                ASSERT_EQ(plan.domains.size(), 3u);
                EXPECT_EQ(plan.domains[1].name, "rocm_followers");
                EXPECT_EQ(plan.domains[1].participants, (std::vector{GlobalDeviceAddress::rocm(0),
                    GlobalDeviceAddress::rocm(1), GlobalDeviceAddress::rocm(2), GlobalDeviceAddress::rocm(3)}));
                EXPECT_EQ(plan.domains[1].scope, ExecutionDomainScope::RANK_LOCAL);
                EXPECT_EQ(plan.routed_tiers[1].max_experts_per_layer, 8);
            }
            std::size_t observed = 0;
            for (const auto& cell : expandModelParityDefinition(definition)) {
                if (cell.expert_overlay != std::optional{ModelParityExpertOverlayPolicy{}} ||
                    (cell.mtp != ModelParityMTP::Off && cell.mtp != ModelParityMTP::Depth1 &&
                     cell.mtp != ModelParityMTP::Depth3 && cell.mtp != ModelParityMTP::DynamicDepth)) continue;
                ++observed;
                SCOPED_TRACE(cell.testName());
                EXPECT_TRUE(names.insert(cell.testName()).second);
                EXPECT_EQ(cell.model.model_path, kMainModelPath);
                EXPECT_EQ(cell.model.token_ids, qwen4ExpProductionCases().front().model.token_ids);
                EXPECT_EQ(cell.retained_mtp_draft_capacity, 15);
                ASSERT_TRUE(cell.ngram_table_policy);
                const auto& storage = *cell.ngram_table_policy;
                if (storage.storage == ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD) {
                    const auto& placement = storage.gpu_adaptive.placement;
                    const std::size_t degree = target == Qwen4ExpHardwareTarget::ROCm4 ? 4 :
                        (target == Qwen4ExpHardwareTarget::ROCm1 || target == Qwen4ExpHardwareTarget::CUDA1 ? 1 : 2);
                    const auto count = !mixed ? degree : owners == Qwen4ExpHardwareNgramOwners::Continuation ? 2u :
                        owners == Qwen4ExpHardwareNgramOwners::Follower ? 4u : 6u;
                    ASSERT_EQ(placement.owners.size(), count);
                    const auto rows = placement.owners.front().maximum_cache_rows;
                    EXPECT_TRUE(rows == 0 || rows == 64);
                    EXPECT_TRUE(storage.gpu_adaptive.host.maximum_cache_rows == 0 ||
                                storage.gpu_adaptive.host.maximum_cache_rows == 64);
                    if (rows == 0) EXPECT_EQ(storage.gpu_adaptive.host.maximum_cache_rows, 0u);
                    for (const auto& owner : placement.owners) {
                        EXPECT_EQ(owner.maximum_cache_rows, rows);
                        EXPECT_EQ(owner.cache_physical_byte_limit, rows ? 1u << 20 : 0u);
                    }
                }
                const auto hardware = cell.toTestConfig().local_hardware_requirements;
                ASSERT_TRUE(hardware);
                EXPECT_EQ(hardware->cuda_devices, target == Qwen4ExpHardwareTarget::CUDA1 ? 1 :
                    (target == Qwen4ExpHardwareTarget::CUDA2 || mixed ? 2 : 0));
                EXPECT_EQ(hardware->rocm_devices, target == Qwen4ExpHardwareTarget::ROCm1 ? 1 :
                    target == Qwen4ExpHardwareTarget::ROCm2 ? 2 :
                    (target == Qwen4ExpHardwareTarget::ROCm4 || mixed ? 4 : 0));
                auto expected = cell.makeOrchestrationConfig(cell.model.model_path, 0);
                ASSERT_TRUE(normalizeMoERoutedExpertPlacementDomains(expected).empty());
                const auto actual = parseSharedArguments(modelParityServerArguments(cell));
                EXPECT_EQ(actual.ngram_table_policy, expected.ngram_table_policy);
                EXPECT_EQ(actual.mtp.enabled, expected.mtp.enabled);
                EXPECT_EQ(actual.mtp.graph_capacity_draft_tokens, 15);
                ASSERT_TRUE(actual.moe_routed_expert_plan);
                if (mixed) EXPECT_EQ(actual.moe_routed_expert_plan->domains[1].participants,
                                    expected.moe_routed_expert_plan->domains[1].participants);
            }
            EXPECT_EQ(observed, 28u);
        }
    }
    EXPECT_EQ(configurations, 8u);
    EXPECT_EQ(names.size(), 224u);
}
} // namespace llaminar2::test::parity

namespace llaminar2::test::parity
{
/** @test Performance candidates retain source identity but not correctness quota8, retained15 or forced migration. */
TEST(Qwen4ExpSharedCoordinatorTP2Execution, SixHardwareProductionDefaultsUseOriginalPMAResidency)
{
    using namespace qwen4exp;
    const auto& cases = qwen4ExpHardwareProductionSourceCases();
    ASSERT_EQ(cases.size(), 12u);
    EXPECT_EQ(qwen4ExpHardwareProductionSourceAdmissionCases().size(), cases.size());
    for (const auto& cell : cases) {
        SCOPED_TRACE(cell.testName());
        EXPECT_EQ(cell.topology.test_id.find("GpuExperts8"), std::string::npos);
        EXPECT_NE(cell.topology.test_id.find("PMAExpertResidency"), std::string::npos);
        EXPECT_EQ(cell.mtp, ModelParityMTP::Off);
        EXPECT_EQ(cell.retained_mtp_draft_capacity, 0);
        EXPECT_FALSE(cell.model.shared_predictor);
        EXPECT_FALSE(cell.requiresPhysicalExpertMovement());
        ASSERT_TRUE(cell.ngram_table_policy);
        EXPECT_TRUE(cell.ngram_table_policy->storage == ImmutableRowStoragePreference::RAM ||
                    cell.ngram_table_policy->storage == ImmutableRowStoragePreference::MemoryMapped);
        ASSERT_TRUE(cell.topology.expert_overlay_plan);
        for (const auto& tier : cell.topology.expert_overlay_plan->routed_tiers) {
            EXPECT_EQ(tier.max_experts_per_layer, 0);
            EXPECT_EQ(tier.memory_budget_bytes, 0u);
        }
        auto expected = cell.makeOrchestrationConfig(cell.model.model_path, 0);
        ASSERT_TRUE(normalizeMoERoutedExpertPlacementDomains(expected).empty());
        const auto actual = parseSharedArguments(modelParityDiagnosticBenchmarkArguments(cell));
        EXPECT_EQ(actual.ngram_table_policy, expected.ngram_table_policy);
        EXPECT_FALSE(actual.mtp.enabled);
        EXPECT_EQ(actual.mtp.graph_capacity_draft_tokens, 0);
        ASSERT_TRUE(actual.moe_routed_expert_plan);
        for (const auto& tier : actual.moe_routed_expert_plan->routed_tiers)
            EXPECT_EQ(tier.max_experts_per_layer, 0);
        EXPECT_EQ(actual.moe_rebalance.mode, expected.moe_rebalance.mode);
    }
    EXPECT_TRUE(qfnOriginalCaseFamilies().contains("Qwen4ExpHardwareProductionDefaults"));
}
} // namespace llaminar2::test::parity

namespace llaminar2::test::parity {
/** @test Callable cache experiments share exact explicit expert work and preserve native-source identity. */
TEST(Qwen4ExpSharedCoordinatorTP2Execution, MatchedCacheEconomyKeepsExplicitResidencyAndPMAByteCeilings)
{
 using namespace qwen4exp;
 std::vector<std::string> original;
 for(const auto& cell:qwen4ExpHardwareProductionSourceCases())original.push_back(cell.testName());
 const auto& cases=qwen4ExpHardwareMatchedCacheEconomyCases();ASSERT_EQ(cases.size(),12u);
 std::set<std::string> names;std::map<DeviceType,std::size_t> hardwareRows;std::set<std::tuple<DeviceType,int,std::size_t>> cacheAxes;
 for(const auto& cell:cases){
  SCOPED_TRACE(cell.testName());EXPECT_TRUE(names.insert(cell.testName()).second);
  EXPECT_EQ(cell.model.model_path,kMainModelPath);EXPECT_EQ(cell.model.token_ids,qwen4ExpProductionCases().front().model.token_ids);
  EXPECT_FALSE(cell.model.shared_predictor);EXPECT_EQ(cell.mtp,ModelParityMTP::Off);EXPECT_EQ(cell.retained_mtp_draft_capacity,0);
  ASSERT_TRUE(cell.topology.expert_overlay_plan);const auto& plan=*cell.topology.expert_overlay_plan;
  ASSERT_EQ(plan.routed_tiers.size(),2u);EXPECT_EQ(plan.routed_tiers.front().max_experts_per_layer,128);
  EXPECT_EQ(plan.routed_tiers.back().max_experts_per_layer,384);EXPECT_TRUE(plan.routed_tiers.back().fallback);
  for(const auto& tier:plan.routed_tiers)EXPECT_EQ(tier.memory_budget_bytes,0u);
  ASSERT_EQ(plan.domains.front().participants.size(),2u);const auto backend=plan.domains.front().participants.front().toLocalDeviceId().type;
  ++hardwareRows[backend];EXPECT_EQ(cell.topology.participants.size(),3u);
  ASSERT_TRUE(cell.ngram_table_policy);const auto& policy=*cell.ngram_table_policy;
  if(policy.isGPUAdaptive()){
   const auto& request=policy.gpu_adaptive;EXPECT_EQ(request.host.maximum_cache_rows,0u);EXPECT_EQ(request.host.cache_physical_byte_limit,0u);
   std::size_t bytes=0;
   if(request.placement.mode==AdaptiveGPUCachePlacementMode::ConsumerLocal){
    EXPECT_TRUE(request.placement.owners.empty());EXPECT_FALSE(request.placement.coordinator);
    EXPECT_EQ(request.maximum_gpu_cache_rows,0x7fffffffULL);bytes=request.gpu_cache_physical_byte_limit;
   }else{
    ASSERT_EQ(request.placement.mode,AdaptiveGPUCachePlacementMode::DeclaredRankLocal);ASSERT_EQ(request.placement.owners.size(),2u);
    ASSERT_TRUE(request.placement.coordinator);EXPECT_EQ(request.placement.coordinator->device(),(DeviceId{backend,0}));
    EXPECT_EQ(request.maximum_gpu_cache_rows,0u);EXPECT_EQ(request.gpu_cache_physical_byte_limit,0u);
    for(int i=0;i<2;++i){EXPECT_EQ(request.placement.owners[i].participant.device(),(DeviceId{backend,i}));
     EXPECT_EQ(request.placement.owners[i].maximum_cache_rows,0x7fffffffULL);}
    bytes=request.placement.owners.front().cache_physical_byte_limit;
    EXPECT_EQ(request.placement.owners.back().cache_physical_byte_limit,bytes);
   }
   EXPECT_TRUE(bytes==(std::size_t{256}<<20)||bytes==(std::size_t{1}<<30));
   EXPECT_TRUE(cacheAxes.emplace(backend,static_cast<int>(request.placement.mode),bytes).second);
  }else EXPECT_TRUE(policy.storage==ImmutableRowStoragePreference::RAM||policy.storage==ImmutableRowStoragePreference::MemoryMapped);
  auto expected=cell.makeOrchestrationConfig(cell.model.model_path,0);ASSERT_TRUE(normalizeMoERoutedExpertPlacementDomains(expected).empty());
  const auto parsed=parseSharedArguments(modelParityServerArguments(cell));
  EXPECT_EQ(parsed.ngram_table_policy,expected.ngram_table_policy);EXPECT_FALSE(parsed.mtp.enabled);EXPECT_EQ(parsed.mtp.graph_capacity_draft_tokens,0);
  ASSERT_TRUE(parsed.moe_routed_expert_plan);ASSERT_EQ(parsed.moe_routed_expert_plan->routed_tiers.size(),2u);
  EXPECT_EQ(parsed.moe_routed_expert_plan->routed_tiers[0].max_experts_per_layer,128);
  EXPECT_EQ(parsed.moe_routed_expert_plan->routed_tiers[1].max_experts_per_layer,384);
  const auto exported=exportedShared(cell);EXPECT_EQ(exported["diagnostic_benchmark"]["policy"],"diagnostic_declared");
  EXPECT_FALSE(exported.contains("paired_workload"))<<"No declared profile may invent accuracy/performance evidence";
  const auto hardware=cell.toTestConfig().local_hardware_requirements;ASSERT_TRUE(hardware);
  EXPECT_EQ(hardware->cuda_devices,backend==DeviceType::CUDA?2:0);EXPECT_EQ(hardware->rocm_devices,backend==DeviceType::ROCm?2:0);
 }
 EXPECT_EQ(hardwareRows[DeviceType::CUDA],6u);EXPECT_EQ(hardwareRows[DeviceType::ROCm],6u);EXPECT_EQ(cacheAxes.size(),8u);
 std::vector<std::string> after;for(const auto& cell:qwen4ExpHardwareProductionSourceCases())after.push_back(cell.testName());EXPECT_EQ(after,original);
 EXPECT_TRUE(qfnOriginalCaseFamilies().contains("Qwen4ExpHardwareMatchedCacheEconomy"));
}
/** @test The same definition retains functional MTP and dynamic expert axes, while unsupported owners fail explicitly. */
TEST(Qwen4ExpSharedCoordinatorTP2Execution, MatchedCacheDefinitionsRetainFunctionalAxesAndRejectForeignOwners)
{
 using namespace qwen4exp;
 for(const auto target:{Qwen4ExpHardwareTarget::CUDA2,Qwen4ExpHardwareTarget::ROCm2}){
  const auto policy=qwen4ExpMatchedGPUCachePolicy(target,AdaptiveGPUCachePlacementMode::DeclaredRankLocal,std::size_t{1}<<30);
  const auto definition=qwen4ExpMatchedCacheDefinition(target,policy);
  std::set<ModelParityMTP> mtp;std::vector<ModelParityExpertOverlayPolicy> movement;
  for(const auto& cell:expandModelParityDefinition(definition)){
   mtp.insert(cell.mtp);if(cell.expert_overlay)movement.push_back(*cell.expert_overlay);
   EXPECT_EQ(cell.retained_mtp_draft_capacity,15);EXPECT_EQ(cell.ngram_table_policy,std::optional{policy});
  }
  for(const auto mode:{ModelParityMTP::Off,ModelParityMTP::Depth1,ModelParityMTP::Depth3,ModelParityMTP::DynamicDepth})EXPECT_TRUE(mtp.contains(mode));
  for(const auto& expected:kSharedExpertPolicies)EXPECT_NE(std::find(movement.begin(),movement.end(),expected),movement.end());
  auto foreign=policy;foreign.gpu_adaptive.placement.owners[1].participant.ordinal=7;
  EXPECT_THROW(qwen4ExpMatchedCacheDefinition(target,foreign),std::invalid_argument);
  EXPECT_THROW(qwen4ExpMatchedCacheDefinition(target,policy,0,384),std::invalid_argument);
 }
 EXPECT_THROW(qwen4ExpMatchedGPUCachePolicy(Qwen4ExpHardwareTarget::ROCm4,AdaptiveGPUCachePlacementMode::ConsumerLocal,1u<<20),std::invalid_argument);
 EXPECT_THROW(qwen4ExpMatchedCacheDefinition(Qwen4ExpHardwareTarget::CUDA2ROCm4,ImmutableRowTablePolicy{.storage=ImmutableRowStoragePreference::RAM}),std::invalid_argument);
 EXPECT_THROW(qwen4ExpMatchedCacheDefinition(Qwen4ExpHardwareTarget::CUDA2,ImmutableRowTablePolicy{.storage=ImmutableRowStoragePreference::VRAM}),std::invalid_argument);
}
} // namespace llaminar2::test::parity

namespace llaminar2::test::parity {
/** @test RAM source comparisons derive exact original cells rather than changing topology, workload or acceptance. */
TEST(Qwen4ExpSharedCoordinatorTP2Execution, FullRAMMatchedCacheSiblingsPreserveOriginalDeclaredComparisons){
    using namespace qwen4exp;const auto& original=qwen4ExpHardwareMatchedCacheEconomyCases();ASSERT_EQ(original.size(),12u);
    const auto& ram=qwen4ExpHardwareMatchedCacheRAMBackingEconomyCases();ASSERT_EQ(ram.size(),8u);
    std::set<std::string> originalNames;for(const auto& cell:original)originalNames.insert(cell.testName());
    for(auto cell:ram){
        ASSERT_TRUE(cell.ngram_table_policy);EXPECT_EQ(cell.ngram_table_policy->gpu_adaptive.source_backing,AdaptiveImmutableRowBacking::RAM);
        EXPECT_EQ(cell.ngram_table_policy->gpu_adaptive.host.maximum_cache_rows,0u);EXPECT_EQ(cell.ngram_table_policy->gpu_adaptive.host.cache_physical_byte_limit,0u);
        const auto ramName=cell.testName();EXPECT_FALSE(originalNames.contains(ramName));
        const auto args=modelParityServerArguments(cell);const auto flag=std::find(args.begin(),args.end(),"--ngram-source-backing");ASSERT_NE(flag,args.end());ASSERT_NE(std::next(flag),args.end());EXPECT_EQ(*std::next(flag),"ram");
        EXPECT_EQ(cell.mtp,ModelParityMTP::Off);EXPECT_EQ(cell.retained_mtp_draft_capacity,0);EXPECT_FALSE(cell.e2e_certification);
        cell.ngram_table_policy->gpu_adaptive.source_backing=AdaptiveImmutableRowBacking::SSD;
        ASSERT_TRUE(originalNames.contains(cell.testName()));const auto match=std::find_if(original.begin(),original.end(),[&](const auto& before){return before.testName()==cell.testName();});ASSERT_NE(match,original.end());
        EXPECT_EQ(modelParityServerArguments(cell),modelParityServerArguments(*match));
    }
    EXPECT_TRUE(qfnOriginalCaseFamilies().contains("Qwen4ExpHardwareMatchedCacheRAMBackingEconomy"));
}
}
