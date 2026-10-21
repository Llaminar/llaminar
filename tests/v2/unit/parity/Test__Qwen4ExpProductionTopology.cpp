/**
 * @file Test__Qwen4ExpProductionTopology.cpp
 * @brief Device-free QFN topology ownership and public CLI projection regressions.
 *
 * A mixed-vendor pipeline needs three participant-local graphs. Declaring a
 * foreign GPU as an expert tier inside one continuation graph does not create
 * that executor. These tests inspect the actual canonical production config
 * and exported CLI, without admitting weights or initializing accelerators.
 */
#include "integration/parity/qwen4exp/Qwen4ExpProductionParityCases.h"
#include "integration/parity/ModelParityRuntimeExport.h"
#include "config/OrchestrationConfigParser.h"

#include <gtest/gtest.h>
#include <array>
#include <set>
#include <map>
#include <tuple>

namespace llaminar2::test::parity
{
namespace
{
/** @brief Check exact graph ownership and a contiguous, complete main-layer partition. */
void expectMixedPipeline(const OrchestrationConfig &config)
{
    EXPECT_FALSE(config.moe_routed_expert_plan);
    // Named domains/stages own this topology; the legacy flat PP scalar is
    // intentionally not a second authority in the public CLI declaration.
    ASSERT_EQ(config.domain_definitions.size(),3u);
    ASSERT_EQ(config.pp_stage_definitions.size(),3u);
    const std::array devices{GlobalDeviceAddress::cuda(0),GlobalDeviceAddress::rocm(0),GlobalDeviceAddress::cpu(0)};
    constexpr std::array first{0,8,20};
    constexpr std::array last{7,19,47};
    for (std::size_t stage=0;stage<devices.size();++stage)
    {
        const auto &domain=config.domain_definitions[stage];
        const auto &placement=config.pp_stage_definitions[stage];
        ASSERT_EQ(domain.devices.size(),1u);
        EXPECT_EQ(domain.devices.front(),devices[stage]);
        EXPECT_EQ(domain.owner_rank,0);
        EXPECT_EQ(domain.scope,TPScope::RANK_LOCAL);
        EXPECT_EQ(placement.domain_name,domain.name);
        EXPECT_EQ(placement.stage_id,static_cast<int>(stage));
        EXPECT_EQ(placement.first_layer,first[stage]);
        EXPECT_EQ(placement.last_layer,last[stage]);
    }
}
} // namespace

/** @test Both ngram modes retain independent CUDA, ROCm and CPU graph ownership. */
TEST(Qwen4ExpProductionTopology, MixedVendorUsesExplicitParticipantLocalPipeline)
{
    const auto cells=expandModelParityDefinition(qwen4exp::qwen4ExpProductionDefinition(
        qwen4exp::Qwen4ExpParityTopology::CUDAROCmCPU));
    EXPECT_EQ(cells.size(),2u);
    std::set<ImmutableRowStoragePreference> storage;
    for (const auto &cell:cells)
    {
        EXPECT_EQ(cell.topology.kind,ModelParityTopologyKind::RankLocalPipelineParallel);
        EXPECT_EQ(cell.topology.test_id,"CUDA0_ROCm0_CPU0_PP");
        EXPECT_FALSE(cell.expert_overlay);
        EXPECT_FALSE(cell.topology.isExpertOverlay());
        ASSERT_TRUE(cell.ngram_table_policy);
        storage.insert(cell.ngram_table_policy->storage);
        expectMixedPipeline(cell.makeOrchestrationConfig(cell.model.model_path,0));
    }
    EXPECT_EQ(storage,(std::set{ImmutableRowStoragePreference::RAM,ImmutableRowStoragePreference::MemoryMapped}));
}

/** @test Public server command projection cannot flatten the pipeline into a foreign expert tier. */
TEST(Qwen4ExpProductionTopology, PublicCliRoundTripPreservesPipelineAndNgramPolicy)
{
    for (const auto &cell:expandModelParityDefinition(qwen4exp::qwen4ExpProductionDefinition(
             qwen4exp::Qwen4ExpParityTopology::CUDAROCmCPU)))
    {
        auto args=modelParityServerArguments(cell);
        args.insert(args.begin(),"llaminar2");
        std::vector<char *> argv;
        for (auto &arg:args) argv.push_back(arg.data());
        const auto config=OrchestrationConfigParser{}.parseArgs(static_cast<int>(argv.size()),argv.data());
        expectMixedPipeline(config);
        EXPECT_EQ(config.ngram_table_policy,*cell.ngram_table_policy);
    }
}

/** @test Single-vendor overlays keep their four owner/movement policies and one GPU graph. */
TEST(Qwen4ExpProductionTopology, SingleVendorOverlaysNeverContainForeignGpuParticipants)
{
    for (const auto mode:{qwen4exp::Qwen4ExpParityTopology::CUDACPU,qwen4exp::Qwen4ExpParityTopology::ROCmCPU})
    {
        const auto cells=expandModelParityDefinition(qwen4exp::qwen4ExpProductionDefinition(mode));
        ASSERT_EQ(cells.size(),8u);
        for (const auto &cell:cells)
        {
            const auto config=cell.makeOrchestrationConfig(cell.model.model_path,0);
            ASSERT_TRUE(config.moe_routed_expert_plan);
            EXPECT_TRUE(config.pp_stage_definitions.empty());
            const auto &plan=*config.moe_routed_expert_plan;
            ASSERT_EQ(plan.domains.size(),2u);
            EXPECT_EQ(plan.continuation_domain,plan.domains.front().name);
            ASSERT_EQ(plan.domains.front().participants.size(),1u);
            EXPECT_EQ(plan.domains.front().participants.front(),mode==qwen4exp::Qwen4ExpParityTopology::CUDACPU
                ?GlobalDeviceAddress::cuda(0):GlobalDeviceAddress::rocm(0));
            ASSERT_EQ(plan.domains.back().participants.size(),1u);
            EXPECT_EQ(plan.domains.back().participants.front(),GlobalDeviceAddress::cpu(0));
        }
    }
}

/** @test The main-only matrix covers all requested backend/storage combinations without a stale overlay alias. */
TEST(Qwen4ExpProductionTopology, MatrixContainsAllFourCompositionsAndRejectsUnknownModes)
{
    const auto &cells=qwen4exp::qwen4ExpProductionCases();
    ASSERT_EQ(cells.size(),20u);
    std::set<std::pair<std::string,ImmutableRowStoragePreference>> combinations;
    for (const auto &cell:cells)
    {
        ASSERT_TRUE(cell.ngram_table_policy);
        combinations.emplace(cell.topology.test_id,cell.ngram_table_policy->storage);
        EXPECT_NE(cell.topology.test_id,"CUDA0_ROCm0_CPU0_ExpertOverlay");
    }
    EXPECT_EQ(combinations.size(),8u);
    EXPECT_THROW((void)qwen4exp::qwen4ExpProductionTopology(
        static_cast<qwen4exp::Qwen4ExpParityTopology>(255)),std::invalid_argument);
}
/** @test Explicit zero/bounded adaptive axes use every existing topology and unchanged mathematical/source gates. */
TEST(Qwen4ExpProductionTopology, AdaptiveHostStorageAddsExactCanonicalCellsWithoutReplacingMain20)
{
    const auto &original=qwen4exp::qwen4ExpProductionCases();ASSERT_EQ(original.size(),20u);
    const auto &adaptive=qwen4exp::qwen4ExpHostAdaptiveProductionCases();ASSERT_EQ(adaptive.size(),20u);
    std::set<std::string> names;
    for(const auto &cell:original){ASSERT_TRUE(cell.ngram_table_policy);EXPECT_FALSE(cell.ngram_table_policy->isHostAdaptive());names.insert(cell.testName());}
    std::set<std::tuple<std::string,std::uint64_t,std::size_t>> axes;
    std::map<std::string,std::size_t> topology_counts;
    for(const auto &cell:adaptive)
    {
        ASSERT_TRUE(cell.ngram_table_policy);const auto &p=*cell.ngram_table_policy;EXPECT_TRUE(p.isHostAdaptive());
        EXPECT_TRUE(names.insert(cell.testName()).second);++topology_counts[cell.topology.test_id];
        axes.emplace(cell.topology.test_id,p.host_adaptive.maximum_cache_rows,p.host_adaptive.cache_physical_byte_limit);
        EXPECT_EQ(cell.model.model_path,qwen4exp::kMainModelPath);EXPECT_EQ(cell.model.reference_directory,original.front().model.reference_directory);
        EXPECT_EQ(cell.model.token_ids,original.front().model.token_ids);EXPECT_EQ(cell.model.decode_steps,2);EXPECT_EQ(cell.model.transformer_layers,48);
        EXPECT_EQ(cell.activation_precision,ActivationPrecision::FP32);EXPECT_EQ(cell.kv_cache_precision,KVCachePrecision::FP16);
        EXPECT_EQ(cell.mtp,ModelParityMTP::Off);EXPECT_EQ(cell.retained_mtp_draft_capacity,0);
        EXPECT_EQ(cell.thresholds.early_layers_count,48);EXPECT_EQ(cell.thresholds.min_early_layers_passed,48);
        EXPECT_EQ(cell.graph_snapshot_policy.required_prefill_snapshot_keys,original.front().graph_snapshot_policy.required_prefill_snapshot_keys);
        EXPECT_EQ(cell.graph_snapshot_policy.required_decode_snapshot_keys,original.front().graph_snapshot_policy.required_decode_snapshot_keys);
        EXPECT_TRUE(cell.graph_snapshot_policy.require_graph_execution_on_gpu);EXPECT_TRUE(cell.graph_snapshot_policy.require_snapshot_publication);
        EXPECT_EQ(cell.prefix_restore_geometry,original.front().prefix_restore_geometry);
        EXPECT_EQ(cell.collective_evidence_source,original.front().collective_evidence_source);
        auto args=modelParityServerArguments(cell);args.insert(args.begin(),"llaminar2");std::vector<char *> argv;
        for(auto &arg:args)argv.push_back(arg.data());
        const auto c=OrchestrationConfigParser{}.parseArgs(static_cast<int>(argv.size()),argv.data());EXPECT_EQ(c.ngram_table_policy,p);
        if(cell.topology.test_id=="CUDA0_ROCm0_CPU0_PP")expectMixedPipeline(c);
        if(p.host_adaptive.maximum_cache_rows){EXPECT_EQ(p.host_adaptive.maximum_cache_rows,64u);EXPECT_EQ(p.host_adaptive.cache_physical_byte_limit,1u<<20);}
        else EXPECT_EQ(p.host_adaptive.cache_physical_byte_limit,0u);
    }
    EXPECT_EQ(names.size(),40u);EXPECT_EQ(axes.size(),8u);EXPECT_EQ(topology_counts.size(),4u);
    EXPECT_EQ(topology_counts["CPU0"],2u);EXPECT_EQ(topology_counts["CUDA0_CPU0_ExpertOverlay"],8u);
    EXPECT_EQ(topology_counts["ROCm0_CPU0_ExpertOverlay"],8u);EXPECT_EQ(topology_counts["CUDA0_ROCm0_CPU0_PP"],2u);
}
} // namespace llaminar2::test::parity
