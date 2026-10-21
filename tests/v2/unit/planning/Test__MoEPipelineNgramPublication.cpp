/**
 * @file Test__MoEPipelineNgramPublication.cpp
 * @brief Keep admitted native table placement attached to PP runtime geometry.
 *
 * These tests compile complete QFN metadata through the public topology grammar
 * and PMA transaction. Sparse GGUF extents have no loaded payload. Publication
 * must retain the exact sealed table owner without rewriting request policies.
 */
#include "planning/OrchestrationCandidateAdmission.h"
#include "config/OrchestrationConfigParser.h"
#include "loaders/ResolvedImmutableRowTablePlacement.h"
#include "../../utils/Qwen4ExpPlanningGGUFFixture.h"
#include "../../utils/CPUExecutionTestGeometry.h"
#include <gtest/gtest.h>

using namespace llaminar2;
namespace
{
/** @return Distinct observed GPU allocators and one CPU owner; no backend discovery. */
ClusterInventory publicationInventory()
{
    ClusterInventory inventory;
    inventory.world_size = 1;
    RankInventory rank;
    rank.rank = rank.node_id = rank.local_rank = 0;
    rank.hostname = "publication-host";
    rank.cpu_cores = rank.cpu_worker_threads = 8;
    rank.cpu_execution = test::kSyntheticCPUExecutionGeometry;
    rank.cpu.memory_bytes = rank.cpu.free_memory_bytes = rank.cpu_memory_bytes = 512ULL << 30;
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        rank.gpus.push_back({.type = backend, .local_device_id = 0,
            .memory_bytes = 512ULL << 30, .free_memory_bytes = 500ULL << 30,
            .compute_units = 64, .uuid = std::string(deviceTypeToString(backend)) + "-publication"});
    inventory.ranks.push_back(std::move(rank));
    inventory.buildNodeAggregations();
    return inventory;
}

/** @return A two-stage source-complete PP candidate with the table in either stage. */
AutomaticOrchestrationCandidate publicationCandidate(const PlanningModelSource &source,
    ImmutableRowStoragePreference preference, bool table_in_first_stage)
{
    const int boundary = table_in_first_stage ? 8 : 1;
    std::vector<std::string> args{"llaminar2", "--define-domain",
        "first=cuda:0;scope=rank_local;backend=auto;owner=0", "--define-domain",
        "second=rocm:0;scope=rank_local;backend=auto;owner=0", "--pp-stage",
        "0=first:0-" + std::to_string(boundary - 1), "--pp-stage",
        "1=second:" + std::to_string(boundary) + "-47"};
    std::vector<char *> argv;
    for (auto &arg : args) argv.push_back(arg.data());
    auto config = OrchestrationConfigParser{}.parseArgs(argv.size(), argv.data());
    config.model_path = source.path();
    config.max_seq_len = 64;
    config.moe_routed_prefill.overlay_segment_rows = 32;
    config.moe_rebalance.mode = MoERebalanceRuntimeMode::Off;
    config.mtp_activation_policy = MTPActivationPolicy::Disabled;
    config.prefix_cache.enabled = false;
    config.ngram_table_policy.storage = preference;
    return {OrchestrationStrategy::PipelineParallel,
        ExecutionRankMembership(publicationInventory(), {0}), std::move(config)};
}
}

/** @test Every table tier survives parent admission publication before graph construction. */
TEST(MoEPipelineNgramPublication, PublishesExactPlacementWithTheAdmittedRows)
{
    test::Qwen4ExpPlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    for (const bool first_stage : {false, true})
        for (const auto preference : {ImmutableRowStoragePreference::RAM,
            ImmutableRowStoragePreference::MemoryMapped, ImmutableRowStoragePreference::VRAM})
        {
            SCOPED_TRACE(::testing::Message() << first_stage << '/' << int(preference));
            const auto candidate = publicationCandidate(source, preference, first_stage);
            const auto admitted = AdmittedOrchestrationCandidate::admit(candidate, source,
                {.prefill_bucket_rows = {1, 32}, .minimum_prefill_sequence_rows = 1,
                 .maximum_cached_prefill_buckets = 8});
            ASSERT_TRUE(admitted.pipelineMemory());
            const auto &pipeline = *admitted.pipelineMemory();
            const auto &sealed = pipeline.rankPlan().runtime;
            ASSERT_TRUE(sealed.ngram_table_placement);
            ASSERT_EQ(pipeline.stages().size(), 2u);
            RuntimeConfig current = sealed;
            current.resident_graph_rows = 0;
            current.moe_routed_prefill.overlay_segment_rows = 1;
            current.ngram_table_placement.reset();
            pipeline.publishRuntimeCapacity(current);
            EXPECT_EQ(current.resident_graph_rows, sealed.resident_graph_rows);
            EXPECT_EQ(current.moe_routed_prefill.overlay_segment_rows,
                sealed.moe_routed_prefill.overlay_segment_rows);
            ASSERT_EQ(current.ngram_table_placement, sealed.ngram_table_placement);
            ASSERT_EQ(current.ngram_table_placement->consumers().size(), 1u);
            EXPECT_EQ(current.ngram_table_placement->consumers().front().device,
                first_stage ? DeviceId::cuda(0) : DeviceId(DeviceType::ROCm, 0));
            for (const auto &stage : pipeline.stages())
            {
                EXPECT_EQ(stage.rankPlan().runtime.ngram_table_placement, current.ngram_table_placement);
                EXPECT_EQ(stage.capacity()->physical_memory_admission, pipeline.physicalAdmission());
            }
            // Repeated setup publication retains the same immutable identity.
            pipeline.publishRuntimeCapacity(current);
            EXPECT_EQ(current.ngram_table_placement, sealed.ngram_table_placement);
        }
}
