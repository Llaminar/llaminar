/**
 * @file Test__Qwen35MoE_GraphNative_CudaHotRocmWarmCpuCold_Parity.cpp
 * @brief Phase 17: Production-path PyTorch parity gate for Qwen3.5 MoE graph-native
 *        overlay with `cuda_hot`, `rocm_warm`, and `cpu_cold` fallback tiers.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>

#include "Qwen35MoEParityTestBase.h"
#include "backends/ComputeBackend.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "execution/moe/MoEExpertOverlayProfiler.h"
#include "execution/moe/MoEExpertOwnerMap.h"
#include "execution/moe/MoEExpertParallelPlanner.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35moe;

namespace
{
    constexpr const char *kModelPath = "/opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf";
    constexpr const char *kSnapshotDir = "pytorch_qwen35_moe_snapshots";
    constexpr const char *kCudaHotDomain = "cuda_hot";
    constexpr const char *kRocmWarmDomain = "rocm_warm";
    constexpr const char *kCpuColdDomain = "cpu_cold";
    constexpr const char *kLegacyEnvVar = "LLAMINAR_MOE_LEGACY_OVERLAY_DOMAIN_RUNTIME";

    constexpr int kCudaHotTierIndex = 0;
    constexpr int kRocmWarmTierIndex = 1;
    constexpr int kCpuColdTierIndex = 2;

    // Approximate Qwen3.5-35B-A3B metadata for topology-only, model-free planning.
    constexpr int kQwen35MoENumExperts = 256;
    constexpr int kQwen35MoENumLayers = 94;

    bool isLegacyOverlayRuntimeEnabled()
    {
        const char *value = std::getenv(kLegacyEnvVar);
        return value != nullptr && std::string(value) == "1";
    }

    bool modelAvailable()
    {
        return std::filesystem::exists(kModelPath);
    }

    ExpertComputeDomain cudaHotDomain()
    {
        ExpertComputeDomain domain;
        domain.name = kCudaHotDomain;
        domain.kind = ExpertDomainKind::SingleDevice;
        domain.backend = CollectiveBackendType::NCCL;
        domain.participants = {GlobalDeviceAddress::cuda(0)};
        domain.world_ranks = {0};
        domain.owner_rank = 0;
        domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
        return domain;
    }

    ExpertComputeDomain rocmWarmDomain()
    {
        ExpertComputeDomain domain;
        domain.name = kRocmWarmDomain;
        domain.kind = ExpertDomainKind::SingleDevice;
        domain.backend = CollectiveBackendType::RCCL;
        domain.participants = {GlobalDeviceAddress::rocm(0)};
        domain.world_ranks = {1};
        domain.owner_rank = 1;
        domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
        return domain;
    }

    ExpertComputeDomain cpuColdDomain()
    {
        ExpertComputeDomain domain;
        domain.name = kCpuColdDomain;
        domain.kind = ExpertDomainKind::SingleDevice;
        domain.backend = CollectiveBackendType::HOST;
        domain.participants = {GlobalDeviceAddress::cpu(0)};
        domain.world_ranks = {2};
        domain.owner_rank = 2;
        domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
        return domain;
    }

    ExpertRoutedTier makeTier(
        const std::string &name,
        const std::string &domain,
        int priority,
        int max_experts_per_layer,
        bool fallback = false)
    {
        ExpertRoutedTier t;
        t.name = name;
        t.domain = domain;
        t.priority = priority;
        t.max_experts_per_layer = max_experts_per_layer;
        t.memory_budget_bytes = 0;
        t.fallback = fallback;
        return t;
    }

    MoEExpertModelMetadata topologyOnlyMetadata()
    {
        MoEExpertModelMetadata metadata;
        metadata.num_experts = kQwen35MoENumExperts;
        metadata.num_layers = kQwen35MoENumLayers;
        metadata.d_model = 4096;
        metadata.routed_intermediate_size = 1536;
        metadata.has_shared_expert = true;
        metadata.shared_intermediate_size = 1536;
        metadata.routed_quant_type = "Q4_K";
        metadata.shared_quant_type = "Q4_K";
        return metadata;
    }

    MoEExpertModelMetadata metadataFromModel(const ModelContext &ctx)
    {
        const auto &loader = ctx.concreteLoader();
        const std::string &arch = ctx.architecture();

        MoEExpertModelMetadata metadata;
        metadata.num_layers = ctx.totalBlockCount();
        metadata.num_experts = loader.getInt(arch + ".expert_count", 0);
        metadata.d_model = ctx.embeddingLength();
        metadata.routed_intermediate_size = loader.getInt(arch + ".expert_feed_forward_length", 0);
        if (metadata.routed_intermediate_size == 0)
            metadata.routed_intermediate_size = ctx.feedForwardLength();
        metadata.has_shared_expert = loader.getInt(arch + ".expert_shared_count", 0) > 0;
        metadata.shared_intermediate_size = metadata.has_shared_expert
                                                ? metadata.routed_intermediate_size
                                                : 0;
        metadata.routed_quant_type = "Q4_K";
        metadata.shared_quant_type = "Q4_K";
        return metadata;
    }

    MoEExpertParallelPlan requestedPlan(const MoEExpertModelMetadata &metadata)
    {
        int cuda_capacity = std::max(1, metadata.num_experts / 2);
        int rocm_capacity = std::max(1, metadata.num_experts / 4);
        if (metadata.num_experts >= 3 && cuda_capacity + rocm_capacity >= metadata.num_experts)
            rocm_capacity = std::max(1, metadata.num_experts - cuda_capacity - 1);

        MoEExpertParallelPlan plan;
        plan.enabled = true;
        plan.execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
        plan.continuation_domain = kCudaHotDomain;
        plan.shared_expert_domain = kCudaHotDomain;
        plan.residency_policy = ExpertResidencyPolicy::StaticById;
        plan.domains = {
            cudaHotDomain(),
            rocmWarmDomain(),
            cpuColdDomain(),
        };
        plan.routed_tiers = {
            makeTier("hot", kCudaHotDomain, 0, cuda_capacity),
            makeTier("warm", kRocmWarmDomain, 1, rocm_capacity),
            makeTier("cold", kCpuColdDomain, 2, 0, /*fallback=*/true),
        };
        return plan;
    }

    std::string planValidationErrors(const MoEExpertParallelValidationResult &validation)
    {
        std::ostringstream message;
        for (const auto &error : validation.errors)
            message << "\n - " << error;
        return message.str();
    }

    std::shared_ptr<MoEExpertParallelPlan> makePlannedOverlayPlan(
        const MoEExpertModelMetadata &metadata)
    {
        auto result = MoEExpertParallelPlanner::plan(requestedPlan(metadata), metadata);
        auto planned = result.planned_plan;

        MoEExpertParallelValidationOptions options;
        options.layer_count = metadata.num_layers;
        options.routed_expert_count = metadata.num_experts;
        auto validation = validateMoEExpertParallelPlan(planned, options);
        if (!validation.ok())
        {
            throw std::invalid_argument(
                "Graph-native CudaHot/RocmWarm/CpuCold plan is invalid:" +
                planValidationErrors(validation));
        }
        return std::make_shared<MoEExpertParallelPlan>(std::move(planned));
    }

    std::vector<size_t> tierExpertCounts(const MoEExpertParallelPlan &plan)
    {
        std::vector<size_t> counts(plan.routed_tiers.size(), 0);
        for (const auto &placement : plan.placements)
        {
            for (int tier_index : placement.routed_expert_tier)
            {
                if (tier_index >= 0 && tier_index < static_cast<int>(counts.size()))
                    ++counts[static_cast<size_t>(tier_index)];
            }
        }
        return counts;
    }

    std::optional<std::string> acceleratorHardwareBlocker()
    {
        const int cuda_count = getCudaDeviceCount();
        if (cuda_count < 1)
            return "GraphNative CudaHot/RocmWarm/CpuCold parity requires >=1 CUDA device, found " +
                   std::to_string(cuda_count);

        const int rocm_count = getRocmDeviceCount();
        if (rocm_count < 1)
            return "GraphNative CudaHot/RocmWarm/CpuCold parity requires >=1 ROCm device, found " +
                   std::to_string(rocm_count);

        return std::nullopt;
    }

    bool isTopologySmokeTest()
    {
        const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
        return info && std::string(info->name()) == "TopologySmoke";
    }

    bool isProfilerTest()
    {
        const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
        return info && std::string(info->name()) == "ProfilerCpuFallbackRows";
    }

    TestConfig makeGraphNativeTestConfig()
    {
        TestConfig config;
        config.name = "GraphNative_CudaHot_RocmWarm_CpuCold";
        config.devices = {ParityDeviceType::CUDA, ParityDeviceType::ROCm, ParityDeviceType::CPU};
        config.parallelism = Parallelism::None;
        config.collective = Collective::None;
        config.thresholds = {
            .cosine_threshold = 0.90f,
            .decode_cosine_threshold = 0.80f,
            .early_layers_count = 6,
            .min_early_layers_passed = 5,
            .kl_threshold = 0.05f,
            .min_top1_accuracy = 0.80f,
            .min_top5_accuracy = 0.60f,
            .pytorch_top1_in_topk = 4,
        };
        config.mpi_ranks = 3;
        config.model_path = kModelPath;
        config.snapshot_dir = kSnapshotDir;
        config.activation_precision = ActivationPrecision::FP32;
        config.kv_cache_precision = KVCachePrecision::FP16;
        return config;
    }

} // namespace

class Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold
    : public Qwen35MoEConfigDrivenParityTest<Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold>
{
public:
    static const TestConfig &staticConfig()
    {
        static TestConfig kConfig = makeGraphNativeTestConfig();
        return kConfig;
    }

    const TestConfig &getTestConfig() const { return staticConfig(); }

protected:
    using Base = Qwen35MoEConfigDrivenParityTest<Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold>;

    void SetUp() override
    {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized)
        {
            GTEST_SKIP() << "GraphNative CudaHot/RocmWarm/CpuCold parity requires MPI initialization";
        }

        int rank = 0;
        int world_size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (world_size < cfg().mpi_ranks)
        {
            GTEST_SKIP() << "GraphNative CudaHot/RocmWarm/CpuCold parity requires "
                         << cfg().mpi_ranks << " MPI ranks (got " << world_size << ")";
        }

        mpi_ctx_ = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);

        if (isTopologySmokeTest())
            return;

        if (isProfilerTest())
        {
            ASSERT_TRUE(MoEExpertOverlayProfiler::isEnabled())
                << "Profiler parity tests must run with LLAMINAR_PROFILING=1; "
                   "CTest discovery should inject and mpirun-forward it.";
        }

        Base::SetUp();
    }

    void applyModelOverrides() override
    {
        if (config_.prompt.empty())
            config_.prompt = "The quick brown fox jumps over the lazy dog";
        if (config_.token_ids.empty())
            config_.token_ids = {785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562};

        if (!cfg().model_path.empty())
            config_.model_path = cfg().model_path;
        if (!cfg().snapshot_dir.empty())
            config_.snapshot_dir = cfg().snapshot_dir;

        if (!modelAvailable())
            return;

        const auto metadata_path = std::filesystem::path(config_.snapshot_dir) / "metadata.txt";
        const bool metadata_missing = !std::filesystem::exists(metadata_path);
        const bool metadata_stale = !metadata_missing &&
                                    readSnapshotVersion(metadata_path) < kRequiredSnapshotVersion;
        const int local_needs_regen = (metadata_missing || metadata_stale) ? 1 : 0;
        int global_needs_regen = 0;
        MPI_Allreduce(&local_needs_regen, &global_needs_regen, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        int local_regen_failed = 0;
        if (global_needs_regen && isRank0())
        {
            LOG_INFO("[Qwen3.5 MoE GraphNative] Regenerating PyTorch snapshots on rank 0");
            local_regen_failed = regeneratePyTorchSnapshots() ? 0 : 1;
        }
        int global_regen_failed = 0;
        MPI_Allreduce(&local_regen_failed, &global_regen_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (global_regen_failed)
        {
            ADD_FAILURE() << "Qwen3.5 MoE snapshot regeneration failed";
            return;
        }

        auto prefill_tokens = readPrefillTokensFromMetadata();
        if (!prefill_tokens.empty())
        {
            config_.token_ids = std::move(prefill_tokens);
            LOG_INFO("[Qwen3.5 MoE GraphNative] Loaded " << config_.token_ids.size()
                                                         << " prefill token IDs from metadata");
        }
    }

    bool broadcastRootFlag(bool root_value) const
    {
        int flag = isRootParityRank() && root_value ? 1 : 0;
        MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        return flag != 0;
    }

    bool synchronizeRanksOk(bool local_ok) const
    {
        int ok = local_ok ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        return ok == 1;
    }

    bool collectivelyCheckHardwareAndModel() const
    {
        bool available = false;
        if (isRootParityRank())
            available = !acceleratorHardwareBlocker().has_value() && modelAvailable();
        return broadcastRootFlag(available);
    }

    bool setupPipeline()
    {
        if (!isRootParityRank())
        {
            try
            {
                overlay_plan_ = makePlannedOverlayPlan(topologyOnlyMetadata());
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[Qwen3.5 MoE GraphNative rank" << mpiRank() << "] Plan failed: " << e.what());
                return false;
            }
            return true;
        }

        DeviceManager::instance().initialize(-1);
        model_ctx_ = ModelContext::create(
            config_.model_path,
            nullptr,
            nullptr,
            nullptr,
            WeightDistributionStrategy::REPLICATED);
        if (!model_ctx_)
        {
            LOG_ERROR("[Qwen3.5 MoE GraphNative] Failed to load model from " << config_.model_path);
            return false;
        }

        configureModel(model_ctx_);

        const auto full_metadata = metadataFromModel(*model_ctx_);
        try
        {
            overlay_plan_ = makePlannedOverlayPlan(full_metadata);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[Qwen3.5 MoE GraphNative] Overlay plan construction failed: " << e.what());
            return false;
        }

        InferenceRunnerConfig inf_config;
        inf_config.max_seq_len = 4096;
        inf_config.batch_size = 1;
        inf_config.force_graph = true;
        inf_config.activation_precision = cfg().activation_precision;
        inf_config.kv_cache_precision = cfg().kv_cache_precision;
        inf_config.moe_expert_parallel_plan = overlay_plan_;
        inf_config.moe_expert_overlay_mpi_ctx = mpi_ctx_;

        runner_ = createInferenceRunner(model_ctx_, nullptr, DeviceId::cuda(0), inf_config);
        if (!runner_)
        {
            LOG_ERROR("[Qwen3.5 MoE GraphNative] Failed to create inference runner on cuda:0");
            return false;
        }

        runner_->enableSnapshotCapture();
        return true;
    }

    bool isRootParityRank() const { return isRank0(); }

    bool synchronizedDecodeWorkAvailable()
    {
        bool available = true;
        if (isRootParityRank())
        {
            available = !loadPyTorchSnapshot("decode_step0_LM_HEAD").empty() &&
                        !readDecodeTokensFromMetadata().empty();
        }
        return broadcastRootFlag(available);
    }

    bool producedPrefillSummary(const ParityTestSummary &summary) const
    {
        return summary.embedding_passed ||
               !summary.layer_stats.empty() ||
               summary.lm_head_passed ||
               summary.lm_head_cosine != 0.0f ||
               summary.total_layers_passed > 0;
    }

    bool producedDecodeSummary(const DecodeParitySummary &summary) const
    {
        return !summary.step_stats.empty() ||
               summary.steps_total > 0 ||
               summary.top1_matches > 0;
    }

    [[noreturn]] void abortGraphNativeWorld(const std::string &reason) const
    {
        const std::string message =
            "[Qwen3.5 MoE GraphNative] " + reason +
            "; aborting MPI world to avoid stranding rocm_warm/cpu_cold participants";
        LOG_ERROR(message);
        std::cerr << message << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
        std::abort();
    }

    [[noreturn]] void abortAfterRootThrow(const char *phase, const std::string &what) const
    {
        abortGraphNativeWorld(std::string("root rank threw during ") + phase + ": " + what);
    }

    void runGraphNativePrefillParityBody()
    {
        if (isLegacyOverlayRuntimeEnabled())
        {
            FAIL() << kLegacyEnvVar
                   << " is set in the environment. This test requires graph-native overlay lowering.";
        }

        const bool hardware_and_model_ok = collectivelyCheckHardwareAndModel();
        if (!hardware_and_model_ok)
        {
            if (isRootParityRank())
            {
                const auto blocker = acceleratorHardwareBlocker();
                if (blocker)
                    GTEST_SKIP() << *blocker;
                else
                    GTEST_SKIP() << "Model not found at " << kModelPath;
            }
            return;
        }

        const bool setup_ok = setupPipeline();
        ASSERT_TRUE(synchronizeRanksOk(setup_ok)) << "Pipeline setup failed on one or more ranks";

        if (!isRootParityRank())
            return;

        ParityTestSummary summary;
        try
        {
            summary = runPrefillParity();
        }
        catch (const std::exception &e)
        {
            abortAfterRootThrow("prefill parity", e.what());
        }
        catch (...)
        {
            abortAfterRootThrow("prefill parity", "unknown exception");
        }

        if (!producedPrefillSummary(summary))
        {
            abortGraphNativeWorld(
                "root produced no prefill parity summary - forward likely failed before "
                "all overlay tiers completed");
        }

        assertParity(summary);
    }

    void runGraphNativeDecodeParityBody()
    {
        if (isLegacyOverlayRuntimeEnabled())
        {
            FAIL() << kLegacyEnvVar
                   << " is set. Unset for graph-native decode parity.";
        }

        const bool hardware_and_model_ok = collectivelyCheckHardwareAndModel();
        if (!hardware_and_model_ok)
        {
            if (isRootParityRank())
            {
                const auto blocker = acceleratorHardwareBlocker();
                if (blocker)
                    GTEST_SKIP() << *blocker;
                else
                    GTEST_SKIP() << "Model not found at " << kModelPath;
            }
            return;
        }

        if (!synchronizedDecodeWorkAvailable())
            GTEST_SKIP() << "Decode snapshots or decode token metadata are unavailable";

        const bool setup_ok = setupPipeline();
        ASSERT_TRUE(synchronizeRanksOk(setup_ok)) << "Pipeline setup failed";

        if (!isRootParityRank())
            return;

        DecodeParitySummary summary;
        try
        {
            summary = runDecodeParity();
        }
        catch (const std::exception &e)
        {
            abortAfterRootThrow("decode parity", e.what());
        }
        catch (...)
        {
            abortAfterRootThrow("decode parity", "unknown exception");
        }

        if (!producedDecodeSummary(summary))
        {
            abortGraphNativeWorld(
                "root produced no decode parity summary - forward likely failed before "
                "all overlay tiers completed");
        }

        assertDecodeParity(summary);
    }

    std::shared_ptr<MoEExpertParallelPlan> overlay_plan_;
};

TEST_F(Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold, TopologySmoke)
{
    if (!isRootParityRank())
    {
        GTEST_SKIP() << "TopologySmoke is rank-0 only; rank " << mpiRank()
                     << " participates in MPI barriers but does not own topology assertions";
    }

    EXPECT_FALSE(isLegacyOverlayRuntimeEnabled())
        << kLegacyEnvVar << " must NOT be set for graph-native tests";

    const auto metadata = topologyOnlyMetadata();

    std::shared_ptr<MoEExpertParallelPlan> plan;
    ASSERT_NO_THROW(plan = makePlannedOverlayPlan(metadata))
        << "Plan construction threw for the mixed cuda_hot / rocm_warm / cpu_cold layout";
    ASSERT_NE(plan, nullptr);

    EXPECT_TRUE(plan->isTieredOverlay());
    EXPECT_EQ(plan->continuation_domain, kCudaHotDomain);
    EXPECT_EQ(plan->shared_expert_domain, kCudaHotDomain);
    EXPECT_FALSE(plan->continuation_domain_spec.dense_tp_enabled);
    ASSERT_EQ(plan->domains.size(), 3u);
    ASSERT_EQ(plan->routed_tiers.size(), 3u);

    const auto &cuda_domain = plan->domains[0];
    EXPECT_EQ(cuda_domain.name, kCudaHotDomain);
    EXPECT_EQ(cuda_domain.kind, ExpertDomainKind::SingleDevice);
    EXPECT_EQ(cuda_domain.compute_kind, ExpertDomainComputeKind::ReplicatedExperts);
    ASSERT_EQ(cuda_domain.participants.size(), 1u);
    EXPECT_TRUE(cuda_domain.participants[0].isCUDA());
    EXPECT_EQ(cuda_domain.owner_rank, 0);
    ASSERT_EQ(cuda_domain.world_ranks.size(), 1u);
    EXPECT_EQ(cuda_domain.world_ranks[0], 0);

    const auto &rocm_domain = plan->domains[1];
    EXPECT_EQ(rocm_domain.name, kRocmWarmDomain);
    EXPECT_EQ(rocm_domain.kind, ExpertDomainKind::SingleDevice);
    EXPECT_EQ(rocm_domain.compute_kind, ExpertDomainComputeKind::ReplicatedExperts);
    ASSERT_EQ(rocm_domain.participants.size(), 1u);
    EXPECT_TRUE(rocm_domain.participants[0].isROCm());
    EXPECT_EQ(rocm_domain.owner_rank, 1);
    ASSERT_EQ(rocm_domain.world_ranks.size(), 1u);
    EXPECT_EQ(rocm_domain.world_ranks[0], 1);

    const auto &cpu_domain = plan->domains[2];
    EXPECT_EQ(cpu_domain.name, kCpuColdDomain);
    EXPECT_EQ(cpu_domain.kind, ExpertDomainKind::SingleDevice);
    EXPECT_EQ(cpu_domain.compute_kind, ExpertDomainComputeKind::ReplicatedExperts);
    ASSERT_EQ(cpu_domain.participants.size(), 1u);
    EXPECT_TRUE(cpu_domain.participants[0].isCPU());
    EXPECT_EQ(cpu_domain.owner_rank, 2);
    ASSERT_EQ(cpu_domain.world_ranks.size(), 1u);
    EXPECT_EQ(cpu_domain.world_ranks[0], 2);

    for (const auto &domain : plan->domains)
    {
        EXPECT_EQ(domain.kind, ExpertDomainKind::SingleDevice)
            << "Domain '" << domain.name << "' must remain a single graph-native participant";
        EXPECT_EQ(domain.compute_kind, ExpertDomainComputeKind::ReplicatedExperts)
            << "Domain '" << domain.name << "' must use whole-expert graph-native ownership";
    }

    for (size_t tier_index = 0; tier_index < plan->routed_tiers.size(); ++tier_index)
    {
        const auto &tier = plan->routed_tiers[tier_index];
        EXPECT_EQ(tier.fallback, tier_index == kCpuColdTierIndex)
            << "Only the cpu_cold tier may be fallback=true; tier=" << tier.name;
    }
    EXPECT_EQ(plan->routed_tiers[kCudaHotTierIndex].domain, kCudaHotDomain);
    EXPECT_EQ(plan->routed_tiers[kRocmWarmTierIndex].domain, kRocmWarmDomain);
    EXPECT_EQ(plan->routed_tiers[kCpuColdTierIndex].domain, kCpuColdDomain);

    ASSERT_EQ(plan->placements.size(), static_cast<size_t>(metadata.num_layers));

    const auto counts = tierExpertCounts(*plan);
    ASSERT_EQ(counts.size(), 3u);
    EXPECT_GT(counts[kCudaHotTierIndex], 0u);
    EXPECT_GT(counts[kRocmWarmTierIndex], 0u);
    EXPECT_GT(counts[kCpuColdTierIndex], 0u);
    EXPECT_EQ(counts[kCudaHotTierIndex] + counts[kRocmWarmTierIndex] + counts[kCpuColdTierIndex],
              static_cast<size_t>(metadata.num_experts) * static_cast<size_t>(metadata.num_layers));

    const int expected_cuda_experts = metadata.num_experts / 2;
    const int expected_rocm_experts = metadata.num_experts / 4;
    for (const auto &placement : plan->placements)
    {
        ASSERT_EQ(placement.routed_expert_tier.size(), static_cast<size_t>(metadata.num_experts));
        for (int expert = 0; expert < metadata.num_experts; ++expert)
        {
            const int expected_tier = expert < expected_cuda_experts
                                          ? kCudaHotTierIndex
                                          : (expert < expected_cuda_experts + expected_rocm_experts
                                                 ? kRocmWarmTierIndex
                                                 : kCpuColdTierIndex);
            EXPECT_EQ(placement.routed_expert_tier[static_cast<size_t>(expert)], expected_tier)
                << "StaticById three-tier split changed at layer=" << placement.layer
                << " expert=" << expert;
        }
    }

    const auto owner_map = MoEExpertOwnerMap::build(*plan);
    ASSERT_EQ(owner_map.participants().size(), 3u);

    bool has_cuda_owner = false;
    bool has_rocm_owner = false;
    bool has_cpu_owner = false;
    for (const auto &participant : owner_map.participants())
    {
        has_cuda_owner = has_cuda_owner || participant.device.is_cuda();
        has_rocm_owner = has_rocm_owner || participant.device.is_rocm();
        has_cpu_owner = has_cpu_owner || participant.device.is_cpu();
    }
    EXPECT_TRUE(has_cuda_owner) << "Owner map must include CUDA expert owners";
    EXPECT_TRUE(has_rocm_owner) << "Owner map must include ROCm expert owners";
    EXPECT_TRUE(has_cpu_owner) << "Owner map must include CPU fallback expert owners";

    for (int layer = 0; layer < metadata.num_layers; ++layer)
    {
        for (int expert = 0; expert < metadata.num_experts; ++expert)
        {
            ASSERT_EQ(owner_map.ownerCountForExpert(layer, expert), 1u)
                << "layer=" << layer << " expert=" << expert;
            const auto *owner = owner_map.ownerFor(layer, expert);
            ASSERT_NE(owner, nullptr) << "layer=" << layer << " expert=" << expert;
            if (expert < expected_cuda_experts)
            {
                EXPECT_EQ(owner->domain_name, kCudaHotDomain);
                EXPECT_TRUE(owner->device.is_cuda());
            }
            else if (expert < expected_cuda_experts + expected_rocm_experts)
            {
                EXPECT_EQ(owner->domain_name, kRocmWarmDomain);
                EXPECT_TRUE(owner->device.is_rocm());
            }
            else
            {
                EXPECT_EQ(owner->domain_name, kCpuColdDomain);
                EXPECT_TRUE(owner->device.is_cpu());
            }
        }
    }

    LOG_INFO("[TopologySmoke] cuda_hot experts/layer: " << expected_cuda_experts
                                                        << ", rocm_warm experts/layer: "
                                                        << expected_rocm_experts
                                                        << ", cpu_cold experts/layer: "
                                                        << metadata.num_experts - expected_cuda_experts - expected_rocm_experts);
}

TEST_F(Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold, PrefillParity)
{
    runGraphNativePrefillParityBody();
}

TEST_F(Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold, DecodeParity)
{
    runGraphNativeDecodeParityBody();
}

TEST_F(Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold, ProfilerCpuFallbackRows)
{
    if (isLegacyOverlayRuntimeEnabled())
    {
        FAIL() << kLegacyEnvVar << " must NOT be set for graph-native profiler test.";
    }

    ASSERT_TRUE(MoEExpertOverlayProfiler::isEnabled())
        << "Profiler parity tests must run with LLAMINAR_PROFILING=1; "
           "CTest discovery should inject and mpirun-forward it.";

    const bool hardware_and_model_ok = collectivelyCheckHardwareAndModel();
    if (!hardware_and_model_ok)
    {
        if (isRootParityRank())
        {
            const auto blocker = acceleratorHardwareBlocker();
            if (blocker)
                GTEST_SKIP() << *blocker;
            else
                GTEST_SKIP() << "Model not found at " << kModelPath;
        }
        return;
    }

    const bool setup_ok = setupPipeline();
    ASSERT_TRUE(synchronizeRanksOk(setup_ok)) << "Pipeline setup failed";

    if (!isRootParityRank())
        return;

    MoEExpertOverlayProfiler::reset();

    try
    {
        runner_->forward(config_.token_ids.data(), config_.token_ids.size());
    }
    catch (const std::exception &e)
    {
        abortAfterRootThrow("profiler forward", e.what());
    }
    catch (...)
    {
        abortAfterRootThrow("profiler forward", "unknown exception");
    }

    const auto rows = MoEExpertOverlayProfiler::rows();
    ASSERT_FALSE(rows.empty())
        << "No profiler rows emitted after forward pass with LLAMINAR_PROFILING=1";

    size_t cpu_fallback_rows = 0;
    size_t gpu_cached_rows = 0;
    size_t compact_dispatch_bytes = 0;
    size_t compact_return_bytes = 0;
    size_t dense_bytes_avoided = 0;
    std::map<std::string, size_t> rows_by_domain;
    std::map<std::string, size_t> cpu_fallback_by_domain;
    for (const auto &row : rows)
    {
        cpu_fallback_rows += row.cpu_fallback_rows;
        gpu_cached_rows += row.gpu_cached_rows;
        compact_dispatch_bytes += row.compact_dispatch_bytes;
        compact_return_bytes += row.compact_return_bytes;
        dense_bytes_avoided += row.dense_bytes_avoided;
        rows_by_domain[row.domain] += row.selected_rows + row.inbound_rows;
        if (row.cpu_fallback_rows > 0)
            cpu_fallback_by_domain[row.domain] += row.cpu_fallback_rows;
    }

    EXPECT_GT(cpu_fallback_rows, 0u)
        << "Expected cpu_fallback_rows > 0 for the configured cpu_cold fallback tier";
    EXPECT_GT(compact_dispatch_bytes, 0u)
        << "Expected graph-native sparse dispatch compact bytes > 0";
    EXPECT_GT(compact_return_bytes, 0u)
        << "Expected graph-native return/reduce compact bytes > 0";
    EXPECT_GT(compact_dispatch_bytes + compact_return_bytes, 0u)
        << "Expected graph-native compact bytes > 0";

    std::ostringstream domain_summary;
    for (const auto &[domain, row_count] : rows_by_domain)
        domain_summary << " " << domain << "=" << row_count;
    std::ostringstream fallback_summary;
    for (const auto &[domain, row_count] : cpu_fallback_by_domain)
        fallback_summary << " " << domain << "=" << row_count;

    LOG_INFO("[ProfilerCpuFallbackRows] rows=" << rows.size()
                                               << " cpu_fallback_rows=" << cpu_fallback_rows
                                               << " gpu_cached_rows=" << gpu_cached_rows
                                               << " compact_dispatch_bytes=" << compact_dispatch_bytes
                                               << " compact_return_bytes=" << compact_return_bytes
                                               << " dense_bytes_avoided=" << dense_bytes_avoided
                                               << " rows_by_domain:" << domain_summary.str()
                                               << " cpu_fallback_by_domain:" << fallback_summary.str());
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "[Rank " << rank << "] Qwen3.5 MoE GraphNative CudaHot/RocmWarm/CpuCold parity test\n";

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();

    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
