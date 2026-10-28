/**
 * @file PipelineForwardOnlyFailureCases.h
 * @brief Real admitted PP parents with one forwarded-enqueue synthetic error return.
 *
 * Dedicated vendor executables keep symbol overrides out of all other suites.
 * Construction, prefill and the first pending teacher run unarmed. Observing
 * all original accepted parent launches is the post-admission injection gate.
 * No test callback submits work, fences a stream or changes device state.
 */
#pragma once
#include "PipelineGenerationTestSupport.h"
#include "../../../utils/PipelineForwardOnlyFaultProbe.h"

namespace llaminar2::test
{
/** @brief Reuse the actual model-free graph/arena fixture, with explicit native backend. */
class PipelineForwardOnlyFailure : public PipelineMTPStateOwnership {};

/** @test Accepted terminal writes retire before failure return and parent destruction. */
TEST_P(PipelineForwardOnlyFailure, ForwardedTerminalEnqueueErrorRetiresBeforeReturn)
{
    const std::string backend_name = GetParam();
    initCPUBackend(-1);
    const auto first_device = backend_name == "CUDA" ? DeviceId::cuda(0) : DeviceId::rocm(0);
    auto *backend = getBackendFor(first_device);
    ASSERT_NE(backend, nullptr);
    if (backend->deviceCount() < 2) GTEST_SKIP() << "Requires two " << backend_name << " devices";
    struct RouterLifetime {
        bool owned = GlobalBackendRouter::get() == nullptr;
        /** @brief Install canonical transport before any admitted model-free owners. */
        RouterLifetime() { if (owned) GlobalBackendRouter::initForTests(); }
        /** @brief All actual graph/collective owners retire before the router. */
        ~RouterLifetime() { if (owned) GlobalBackendRouter::shutdown(); }
    } router;
    OrdinaryGraphEvidence evidence("generation,forward_graph,sampling,mtp");
    const auto logical_writes = [](DeviceId device) {
        std::array<double, 2> counts{};
        for (const auto &record : PerfStatsCollector::snapshot())
        {
            if (record.domain != "mtp" || record.device != device.toString()) continue;
            if (record.name == "device_resident_logical_state_mailboxes" &&
                record.tags.contains("publication_kind") && record.tags.at("publication_kind") == "forward_only_terminal")
                counts[0] += record.value;
            if (record.name == "device_resident_logical_state_partial_write_completions") counts[1] += record.value;
        }
        return counts;
    };
    const auto host_publications = [] {
        double total = 0;
        for (const auto &record : PerfStatsCollector::snapshot())
            if (record.domain == "sampling" && record.name == "rank_pipeline_condition_host_logits_publications") total += record.value;
        return total;
    };
    for (const int participant_count : {2, 4})
    {
        if (backend->deviceCount() < participant_count) continue;
        for (const bool reverse : {false, true})
        for (const size_t failure_copy : {size_t{1}, size_t{2}, size_t(3 * participant_count + 2)})
        {
            SCOPED_TRACE(::testing::Message() << "participants=" << participant_count << " reverse=" << reverse << " copy=" << failure_copy);
            ModelMemoryProfile profile;
            profile.architecture = "qwen35";
            profile.n_layers = participant_count;
            profile.d_model = profile.head_dim = profile.vocab_size = 32;
            profile.d_ff = 64;
            profile.n_heads = profile.n_kv_heads = 1;
            profile.max_seq_len = 1024;
            std::vector<DevicePlanConfig> memory_inputs;
            for (int ordinal = 0; ordinal < participant_count; ++ordinal)
            {
                DevicePlanConfig input;
                input.world_rank = 0;
                input.device = backend_name == "CUDA" ? DeviceId::cuda(ordinal) : DeviceId::rocm(ordinal);
                input.first_layer = input.last_layer = reverse ? participant_count - 1 - ordinal : ordinal;
                input.device_total_bytes = input.device_free_bytes = 1ull << 30;
                input.associated_host_memory = PhysicalMemoryResource{.world_rank = 0,
                    .device = DeviceId::cpu(), .total_bytes = 1ull << 30, .admission_available_bytes = 1ull << 30};
                input.device_compute_units = 128;
                input.max_seq_len = input.activation_seq_len = 1024;
                input.local_pipeline_backend = backend_name == "CUDA" ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
                input.prefix_cache.enabled = true;
                input.prefix_cache.storage_mode = PrefixCacheStorageMode::Ram;
                input.prefix_cache.ram_budget_bytes = 4u << 20;
                input.prefix_cache.device_budget_bytes = input.prefix_cache.disk_budget_bytes = 0;
                input.captured_serving_graphs = resolveCapturedServingGraphMemoryInventory(
                    {256, 512},
                    {},
                    resolveMTPVerifierGraphFamily(mtpVerifierExecutionRole(declaredDeviceGenerationExecutionPolicy(
                    input.device.type, DeviceGenerationExecutionDomain::NativeDomain))),
                    input.first_layer + 1 == participant_count
                        ? MTPGenerationParentRole::Controller : MTPGenerationParentRole::PipelineFollower,
                    DeviceGenerationExecutionDomain::NativeDomain);
                memory_inputs.push_back(input);
            }
            auto memory = std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(MemoryPlanner::plan(profile, memory_inputs).physicalPlan()), 0);
            std::vector<std::shared_ptr<OrdinaryDGOGraph>> graphs;
            std::vector<std::unique_ptr<IInferenceRunner>> owners;
            std::vector<DeviceGraphOrchestrator *> stages(participant_count);
            for (int i = 0; i < participant_count; ++i)
            {
                const int ordinal = reverse ? participant_count - 1 - i : i;
                const auto device = backend_name == "CUDA" ? DeviceId::cuda(ordinal) : DeviceId::rocm(ordinal);
                auto config = stateConfig(device);
                config.n_layers = 1;
                config.total_n_layers = participant_count;
                config.pp_layer_offset = i;
                config.layer_types.assign(participant_count, "full_attention");
                config.max_seq_len = 1024;
                config.kv_cache_precision = KVCachePrecision::FP16;
                // This explicit model-free constructor has no file-backed model
                // admission. Exercise the real RAM prefix tier; device-hot/disk
                // admission belongs to the production-factory model tests.
                config.prefix_cache.enabled = true;
                config.prefix_cache.storage_mode = PrefixCacheStorageMode::Ram;
                config.prefix_cache.ram_budget_bytes = 4u << 20;
                config.prefix_cache.device_budget_bytes = 0;
                config.prefix_cache.disk_budget_bytes = 0;
                auto graph = std::make_shared<OrdinaryDGOGraph>(config, i == 0, i + 1 == participant_count);
                auto owner = DeviceGraphOrchestrator::createForTest({
                    .model_ctx = MockModelContext::createMinimal(), .graph_builder = graph,
                    .physical_memory_authority = memory});
                owner->setPPStageConfig({.first_layer = i, .last_layer = i + 1,
                    .has_embedding = i == 0, .has_lm_head = i + 1 == participant_count});
                auto &context = GPUDeviceContextPool::instance().getContext(device);
                context.submitAndWait([&] {
                    ASSERT_TRUE(graph->table.ensureOnDevice(device, context.defaultStream()));
                    TransferEngine::publishDeviceWrite(&graph->table, device, context.defaultStream());
                    ASSERT_TRUE(owner->initializeInferenceStateFromArena(1, 1024, device));
                });
                ASSERT_FALSE(HasFatalFailure());
                stages[i] = owner.get();
                graphs.push_back(std::move(graph));
                owners.push_back(std::move(owner));
            }
            RankOrchestrator::Config rank_config;
            rank_config.mode = RankOrchestrator::ParallelismMode::PP;
            rank_config.max_seq_len = 1024;
            for (int i = 0; i < participant_count; ++i)
                rank_config.pp_stages.push_back({.first_layer = i, .last_layer = i + 1,
                    .has_embedding = i == 0, .has_lm_head = i + 1 == participant_count,
                    .stage_devices = {GlobalDeviceAddress::fromLocalDeviceId(stages[i]->primaryDeviceId())}});
            auto model = MockModelContext::createMinimal();
            model->setVocabSize(32);
            auto rank = RankOrchestrator::createForTestWithPipelineStages(
                std::move(model), std::move(owners), rank_config);
            ASSERT_TRUE(rank->materializeServingGraphFamilyWithoutLaunch({
                .prefill_bucket_rows = {256, 512}, .prefill_pad_token_id = 0,
                .main_decode_graph = ServingMainDecodeGraphKind::HistoryBearingSerial}));
            for (auto *stage : stages)
            {
                EXPECT_EQ(memory->claimedBytes(stage->primaryDeviceId(), PhysicalMemoryOwner::LocalCollective,
                    PhysicalMemoryMaterializationKind::NewAllocation), sizeof(int32_t));
            }

            auto *pipeline = rank.get();
            OrchestrationConfig config;
            config.max_seq_len = 1024;
            config.prefill_max_bucket_size = 256;
            config.prefix_cache.enabled = false;
            config.prefix_cache.storage_mode = PrefixCacheStorageMode::Disabled;
            RankExecutionPlan plan;
            plan.primary_device = GlobalDeviceAddress::fromLocalDeviceId(stages.back()->primaryDeviceId());
            plan.runtime.max_seq_len = 1024;
            plan.runtime.resident_graph_rows = 512;
            plan.runtime.prefix_cache = config.prefix_cache;
            std::unique_ptr<pp_fault::Probe> probe; // Declared first: runner always retires before probe.
            auto runner = std::make_unique<OrchestrationRunner>(config, plan, std::move(rank));

            runner->clearCache();
            SamplingParams law;
            law.temperature = 1.0f;
            law.top_k = 4;
            law.seed = 1700;
            law.presence_penalty = 0.75f;
            law.frequency_penalty = 0.25f;
            law.repetition_penalty = 1.2f;
            runner->setSamplingParams(law);
            runner->setStopTokens({});
            ASSERT_TRUE(runner->prefill(std::vector<int>(256, 7))) << runner->lastError();
            ASSERT_NE(runner->lastLogits(), nullptr) << "Establish a real prefill host cache before failure";
            ASSERT_TRUE(runner->forceDecodeToken(13).success()) << runner->lastError();
            std::vector<int> devices;
            for (auto *stage : stages) devices.push_back(stage->primaryDeviceId().ordinal);
            const double publications_before = host_publications();
            probe = std::make_unique<pp_fault::Probe>(devices, failure_copy);
            std::vector<std::array<double, 2>> writes_before;
            for (auto *stage : stages) writes_before.push_back(logical_writes(stage->primaryDeviceId()));
            const auto result = runner->forceDecodeToken(17);
            probe->publicReturned();
            for (size_t i = 0; i < stages.size(); ++i)
            {
                const auto writes = logical_writes(stages[i]->primaryDeviceId());
                EXPECT_EQ(writes[0], writes_before[i][0] + (i + 1 == stages.size() ? 1 : 0));
                EXPECT_EQ(writes[1], writes_before[i][1] + (i + 1 == stages.size() ? 0 : 1));
                EXPECT_EQ(stages[i]->deviceResidentLogicalSequenceState().coversRequest(0), i + 1 == stages.size())
                    << "Before failed terminal-copy cleanup, only the actual tail owns a complete logical outcome";
            }
            EXPECT_FALSE(result.success()) << "Synthetic native return must reach the actual public failure boundary";
            const size_t expected_bytes = failure_copy == size_t(3 * participant_count + 2)
                ? sampling_math::kDeviceGenerationControlCount * sizeof(int32_t) : sizeof(int);
            EXPECT_TRUE(probe->returnedSafely(expected_bytes)) << "All real parents precede injection; exact-stream event covers every accepted host write before return";
            EXPECT_THROW(pipeline->logits(), std::logic_error) << "Failed forward must invalidate the previous prefill rank cache";
            EXPECT_EQ(host_publications(), publications_before);
            runner.reset(); // Actual destructor must fence even peers with no terminal copy.
            EXPECT_TRUE(probe->retiredSafely()) << "All original parent executables require exact-owner event completion before destruction";
            probe.reset();
            for (const auto device : devices)
            {
                EXPECT_EQ(memory->claimedBytes(backend_name == "CUDA" ? DeviceId::cuda(device) : DeviceId::rocm(device),
                    PhysicalMemoryOwner::LocalCollective, PhysicalMemoryMaterializationKind::NewAllocation), 0u);
            }
        }
    }
}
INSTANTIATE_TEST_SUITE_P(Native, PipelineForwardOnlyFailure,
    ::testing::Values(std::string(LLAMINAR_PP_FAULT_BACKEND)));
} // namespace llaminar2::test
