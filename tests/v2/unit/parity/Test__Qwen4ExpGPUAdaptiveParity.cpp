/**
 * @file Test__Qwen4ExpGPUAdaptiveParity.cpp
 * @brief Exact device-free model inventory and complete mixed-request identity.
 *
 * This suite expands the production definitions but loads no model and claims
 * no native/admission/numerical certificate. The actual model cells retain all
 * inherited checkpoint, prefix, precision and captured-path obligations.
 */
#include "integration/parity/qwen4exp/Qwen4ExpGPUAdaptiveParityCases.h"
#include "integration/parity/qwen4exp/Qwen4ExpFixedHeadParityCases.h"
#include "integration/parity/ModelParityRuntimeExport.h"
#include "config/OrchestrationConfigDocument.h"
#include "config/OrchestrationConfigParser.h"
#include <sstream>
#include <gtest/gtest.h>
#include <set>
#include <tuple>

namespace llaminar2::test::parity
{
/** @test Only four new cells appear, with exact original source and complete ordinary mathematical gates. */
TEST(Qwen4ExpGPUAdaptiveParity, FourCanonicalCellsKeepOriginalFamiliesAndAllNumericalGates)
{
    const auto &main=qwen4exp::qwen4ExpProductionCases(),&host=qwen4exp::qwen4ExpHostAdaptiveProductionCases();
    const auto &fixed=qwen4exp::qwen4ExpFixedHeadProductionCases(),&adaptive=qwen4exp::qwen4ExpGPUAdaptiveProductionCases();
    ASSERT_EQ(main.size(),20u);ASSERT_EQ(host.size(),20u);ASSERT_EQ(fixed.size(),64u);ASSERT_EQ(adaptive.size(),4u);
    std::set<std::string> names;
    for(const auto *family:{&main,&host,&fixed})for(const auto &cell:*family)ASSERT_TRUE(names.insert(cell.testName()).second);
    std::set<std::tuple<DeviceType,std::uint64_t,std::uint64_t>> axes;
    const auto &reference=main.front();
    for(const auto &cell:adaptive)
    {
        ASSERT_TRUE(names.insert(cell.testName()).second);ASSERT_TRUE(cell.ngram_table_policy);
        const auto &p=*cell.ngram_table_policy;ASSERT_TRUE(p.isGPUAdaptive());
        ASSERT_TRUE(cell.expert_overlay);EXPECT_EQ(*cell.expert_overlay,ModelParityExpertOverlayPolicy{});
        EXPECT_EQ(cell.mtp,ModelParityMTP::Off);EXPECT_EQ(cell.retained_mtp_draft_capacity,0);EXPECT_FALSE(cell.e2e_certification);
        EXPECT_EQ(cell.model.model_path,reference.model.model_path);EXPECT_EQ(cell.model.model_path,qwen4exp::kMainModelPath);
        EXPECT_EQ(cell.model.reference_directory,reference.model.reference_directory);EXPECT_EQ(cell.model.token_ids,reference.model.token_ids);
        EXPECT_EQ(cell.model.transformer_layers,48);EXPECT_EQ(cell.model.decode_steps,2);EXPECT_EQ(cell.model.max_seq_len,reference.model.max_seq_len);
        EXPECT_EQ(cell.activation_precision,reference.activation_precision);EXPECT_EQ(cell.kv_cache_precision,reference.kv_cache_precision);
        EXPECT_EQ(cell.thresholds.early_layers_count,48);EXPECT_EQ(cell.thresholds.min_early_layers_passed,48);
        const auto &a=cell.thresholds,&b=reference.thresholds;
        EXPECT_EQ(std::tie(a.cosine_threshold,a.decode_cosine_threshold,a.early_layers_count,a.min_early_layers_passed,
            a.kl_threshold,a.prefill_kl_threshold,a.mtp_kl_threshold,a.excluded_stages,a.allreduce_stages,
            a.min_top1_accuracy,a.min_top5_accuracy,a.min_decode_pass_rate,a.pytorch_top1_in_topk),
            std::tie(b.cosine_threshold,b.decode_cosine_threshold,b.early_layers_count,b.min_early_layers_passed,
            b.kl_threshold,b.prefill_kl_threshold,b.mtp_kl_threshold,b.excluded_stages,b.allreduce_stages,
            b.min_top1_accuracy,b.min_top5_accuracy,b.min_decode_pass_rate,b.pytorch_top1_in_topk));
        EXPECT_EQ(cell.graph_snapshot_policy.required_prefill_snapshot_keys,reference.graph_snapshot_policy.required_prefill_snapshot_keys);
        EXPECT_EQ(cell.graph_snapshot_policy.required_decode_snapshot_keys,reference.graph_snapshot_policy.required_decode_snapshot_keys);
        EXPECT_TRUE(cell.graph_snapshot_policy.require_graph_execution_on_gpu);EXPECT_TRUE(cell.graph_snapshot_policy.require_snapshot_publication);
        EXPECT_EQ(cell.prefix_restore_geometry,reference.prefix_restore_geometry);EXPECT_EQ(cell.collective_evidence_source,reference.collective_evidence_source);
        ASSERT_TRUE(cell.topology.expert_overlay_plan);const auto &plan=*cell.topology.expert_overlay_plan;
        ASSERT_EQ(plan.domains.size(),2u);ASSERT_EQ(plan.domains.front().participants.size(),1u);
        const auto gpu=plan.domains.front().participants.front();EXPECT_TRUE(gpu.isCUDA() || gpu.isROCm());
        EXPECT_EQ(plan.domains.front().scope,ExecutionDomainScope::SINGLE);
        EXPECT_EQ(plan.domains.back().participants,(std::vector{GlobalDeviceAddress::cpu(0)}));
        EXPECT_EQ(plan.continuation_domain_spec.effectiveDensePolicy(),DenseParallelPolicy::Replicated);
        EXPECT_EQ(plan.routed_tiers.front().max_experts_per_layer,8);EXPECT_EQ(plan.routed_tiers.back().max_experts_per_layer,0);
        EXPECT_EQ(plan.routed_tiers.front().memory_budget_bytes,0u);
        const auto backend=gpu.isCUDA()?DeviceType::CUDA:DeviceType::ROCm;
        axes.emplace(backend,p.gpu_adaptive.maximum_gpu_cache_rows,p.gpu_adaptive.host.maximum_cache_rows);
        const auto config=cell.makeOrchestrationConfig(cell.model.model_path,0);
        EXPECT_EQ(config.ngram_table_policy,p);EXPECT_TRUE(config.pp_stage_definitions.empty());EXPECT_TRUE(config.tp_devices.empty());
        const auto parsed=deserializeOrchestrationConfig(serializeOrchestrationConfig(config));
        EXPECT_EQ(parsed.ngram_table_policy,p);ASSERT_TRUE(parsed.moe_routed_expert_plan);
        EXPECT_EQ(parsed.moe_routed_expert_plan->routed_tiers.front().max_experts_per_layer,8);
        auto args=modelParityServerArguments(cell);args.insert(args.begin(),"llaminar2");std::vector<char *> argv;
        for(auto &arg:args)argv.push_back(arg.data());
        const auto cli=OrchestrationConfigParser{}.parseArgs(static_cast<int>(argv.size()),argv.data());
        EXPECT_EQ(cli.ngram_table_policy,p);ASSERT_TRUE(cli.moe_routed_expert_plan);
        EXPECT_EQ(cli.moe_routed_expert_plan->routed_tiers.front().max_experts_per_layer,8);
        EXPECT_EQ(cli.executionDomainDefinitions().front().participants,plan.domains.front().participants);
        std::ostringstream discovery;
        ASSERT_NO_THROW(PrintTo(cell,&discovery));
        EXPECT_NE(discovery.str().find("\"server_args\":["),std::string::npos);
        EXPECT_NE(discovery.str().find("gpu-adaptive-vram-ram-ssd"),std::string::npos);
        EXPECT_NE(discovery.str().find("\"e2e\":null,\"benchmark\":null"),std::string::npos);
        if(p.gpu_adaptive.maximum_gpu_cache_rows)
        {
            EXPECT_EQ(p.gpu_adaptive.maximum_gpu_cache_rows,64u);EXPECT_EQ(p.gpu_adaptive.host.maximum_cache_rows,64u);
            EXPECT_EQ(p.gpu_adaptive.gpu_cache_physical_byte_limit,1u<<20);EXPECT_EQ(p.gpu_adaptive.host.cache_physical_byte_limit,1u<<20);
        }
        else
        {
            EXPECT_EQ(p.gpu_adaptive.gpu_cache_physical_byte_limit,0u);EXPECT_EQ(p.gpu_adaptive.host.maximum_cache_rows,0u);
            EXPECT_EQ(p.gpu_adaptive.host.cache_physical_byte_limit,0u);
        }
    }
    EXPECT_EQ(axes,(std::set<std::tuple<DeviceType,std::uint64_t,std::uint64_t>>{
        {DeviceType::CUDA,0,0},{DeviceType::CUDA,64,64},{DeviceType::ROCm,0,0},{DeviceType::ROCm,64,64}}));
    EXPECT_EQ(names.size(),108u);
}
/** @test Every mutable request field is part of a unique name without changing historical whole-tier names. */
TEST(Qwen4ExpGPUAdaptiveParity, CompleteRequestedCeilingsAndPoliciesCannotShareCellIdentity)
{
    auto original=qwen4exp::qwen4ExpGPUAdaptiveStorageAxes().back();const auto reference=ngramStorageTestName(original);
    std::set<std::string> names{reference};
    for(int mutation=0;mutation<12;++mutation)
    {
        auto p=original;
        switch(mutation)
        {
        case 0:++p.gpu_adaptive.maximum_gpu_cache_rows;break;
        case 1:++p.gpu_adaptive.gpu_cache_physical_byte_limit;break;
        case 2:++p.gpu_adaptive.gpu_policy.aging_accesses;break;
        case 3:++p.gpu_adaptive.gpu_policy.protected_denominator;break;
        case 4:p.gpu_adaptive.gpu_policy.protected_numerator=0;break;
        case 5:++p.gpu_adaptive.host.maximum_cache_rows;break;
        case 6:++p.gpu_adaptive.host.cache_physical_byte_limit;break;
        case 7:++p.gpu_adaptive.host.policy.aging_accesses;break;
        case 8:++p.gpu_adaptive.host.policy.protected_denominator;break;
        case 9:p.gpu_adaptive.host.policy.protected_numerator=0;break;
        case 10:p.window_bytes*=2;break;
        case 11:++p.window_count;break;
        }
        EXPECT_NE(p,original);EXPECT_TRUE(names.insert(ngramStorageTestName(p)).second);
        auto cell=qwen4exp::qwen4ExpGPUAdaptiveProductionCases().back();cell.ngram_table_policy=p;
        auto args=modelParityNgramStorageArguments(cell);args.insert(args.begin(),"llaminar2");
        std::vector<char *> argv;for(auto &arg:args)argv.push_back(arg.data());
        EXPECT_EQ(OrchestrationConfigParser{}.parseArgs(static_cast<int>(argv.size()),argv.data()).ngram_table_policy,p);
    }
    EXPECT_EQ(ngramStorageTestName(ImmutableRowTablePolicy{ImmutableRowStoragePreference::RAM}),"NgramRAM_");
    EXPECT_EQ(ngramStorageTestName(ImmutableRowTablePolicy{ImmutableRowStoragePreference::MemoryMapped}),"NgramMmap_");
}
/** @test Every actual expanded cell fits one canonical artifact component for both registered test bodies. */
TEST(Qwen4ExpGPUAdaptiveParity, CompleteCanonicalArtifactComponentsFitNameMaxWithoutDroppingRequestFields)
{
    const auto &cases=qwen4exp::qwen4ExpGPUAdaptiveProductionCases();ASSERT_EQ(cases.size(),4u);
    std::set<std::string> components;
    for(const auto &cell:cases)for(const auto *body:{"ProductionParity","SnapshotAdmission"})
    {
        // The canonical writer joins suite/test with '/', then replaces each
        // forbidden ASCII byte with one '_'. Therefore this original complete
        // ASCII identity has exactly the final component's byte length. This
        // check does not implement a second sanitizer or omit any typed field.
        const auto raw=std::string("Qwen4ExpGPUAdaptive/Qwen4ExpMainParityTest/")+body+"/"+cell.testName();
        EXPECT_TRUE(std::all_of(raw.begin(),raw.end(),[](unsigned char c){return c<128;}));
        EXPECT_LE(raw.size(),255u) << raw;
        EXPECT_TRUE(components.insert(raw).second);
        const auto old=std::string("Qwen4ExpGPUAdaptiveVRAMRAMSSD/Qwen4ExpMainParityTest/")+body+"/"+cell.testName();
        if(cell.ngram_table_policy->gpu_adaptive.maximum_gpu_cache_rows)
        {
            EXPECT_GT(old.size(),255u) << "The old complete positive identity is the original-code negative control";
        }
    }
    EXPECT_EQ(components.size(),8u);
}
/** @test Unsupported continuation topologies and mixed policy arms cannot enter by inventory declaration. */
TEST(Qwen4ExpGPUAdaptiveParity, CPUAndPipelineDefinitionsRemainUnsupportedRatherThanSubstituted)
{
    EXPECT_THROW((void)qwen4exp::qwen4ExpGPUAdaptiveParityDefinition(qwen4exp::Qwen4ExpParityTopology::CPU),std::invalid_argument);
    EXPECT_THROW((void)qwen4exp::qwen4ExpGPUAdaptiveParityDefinition(qwen4exp::Qwen4ExpParityTopology::CUDAROCmCPU),std::invalid_argument);
    auto definition=qwen4exp::qwen4ExpGPUAdaptiveParityDefinition(qwen4exp::Qwen4ExpParityTopology::CUDACPU);
    definition.features.ngram_storage.front().distribution=ImmutableRowTableDistribution::FixedHeadSharded;
    EXPECT_THROW((void)expandModelParityDefinition(definition),std::invalid_argument);
    definition=qwen4exp::qwen4ExpGPUAdaptiveParityDefinition(qwen4exp::Qwen4ExpParityTopology::CUDACPU);
    definition.features.ngram_storage.front().host_adaptive.maximum_cache_rows=1;
    EXPECT_THROW((void)expandModelParityDefinition(definition),std::invalid_argument);
}
} // namespace llaminar2::test::parity
