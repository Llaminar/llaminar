/**
 * @file Test__Qwen35MoE_GraphNative_CudaHotRocmWarm_Parity.cpp
 * @brief Phase 16: Production-path PyTorch parity gate for Qwen3.5 MoE graph-native
 *        overlay with an all-GPU `cuda_hot` / `rocm_warm` layout and no CPU fallback.
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
    constexpr const char *kLegacyEnvVar = "LLAMINAR_MOE_LEGACY_OVERLAY_DOMAIN_RUNTIME";

    constexpr int kCudaHotTierIndex = 0;
    constexpr int kRocmWarmTierIndex = 1;

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

    ExpertRoutedTier makeTier(
        const std::string &name,
        const std::string &domain,
        int priority,
        int max_experts_per_layer)
    {
        ExpertRoutedTier t;
        t.name = name;
        t.domain = domain;
        t.priority = priority;
        t.max_experts_per_layer = max_experts_per_layer;
        t.memory_budget_bytes = 0;
        t.fallback = false;
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
        const int cuda_capacity = std::max(1, metadata.num_experts / 2);
        const int rocm_capacity = std::max(1, metadata.num_experts - cuda_capacity);

        MoEExpertParallelPlan plan;
        plan.enabled = true;
        plan.execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
        plan.continuation_domain = kCudaHotDomain;
        plan.shared_expert_domain = kCudaHotDomain;
        plan.residency_policy = ExpertResidencyPolicy::StaticById;
        plan.domains = {
            cudaHotDomain(),
            rocmWarmDomain(),
        };
        plan.routed_tiers = {
            makeTier("hot", kCudaHotDomain, 0, cuda_capacity),
            makeTier("warm", kRocmWarmDomain, 1, rocm_capacity),
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
                "Graph-native CudaHot/RocmWarm plan is invalid:" +
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
            return "GraphNative CudaHot/RocmWarm parity requires >=1 CUDA device, found " +
                   std::to_string(cuda_count);

        const int rocm_count = getRocmDeviceCount();
        if (rocm_count < 1)
            return "GraphNative CudaHot/RocmWarm parity requires >=1 ROCm device, found " +
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
        return info && std::string(info->name()) == "ProfilerAllGpuNoFallbackRows";
    }

    std::optional<int> profilerTierIndex(const MoEExpertOverlayProfileRow &row)
    {
        std::vector<std::string> parts;
        std::stringstream stream(row.domain);
        std::string part;
        while (std::getline(stream, part, ':'))
            parts.push_back(part);
        if (parts.size() < 4)
            return std::nullopt;
        try
        {
            return std::stoi(parts[3]);
        }
        catch (...)
        {
            return std::nullopt;
        }
    }

    TestConfig makeGraphNativeTestConfig()
    {
        TestConfig config;
        config.name = "GraphNative_CudaHot_RocmWarm";
        config.devices = {ParityDeviceType::CUDA, ParityDeviceType::ROCm};
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
        config.mpi_ranks = 2;
        config.model_path = kModelPath;
        config.snapshot_dir = kSnapshotDir;
        config.activation_precision = ActivationPrecision::FP32;
        config.kv_cache_precision = KVCachePrecision::FP16;
        return config;
    }

} // namespace

class Qwen35MoEGraphNativeCudaHotRocmWarm
    : public Qwen35MoEConfigDrivenParityTest<Qwen35MoEGraphNativeCudaHotRocmWarm>
{
public:
    static const TestConfig &staticConfig()
    {
        static TestConfig kConfig = makeGraphNativeTestConfig();
        return kConfig;
    }

    const TestConfig &getTestConfig() const { return staticConfig(); }

protected:
    using Base = Qwen35MoEConfigDrivenParityTest<Qwen35MoEGraphNativeCudaHotRocmWarm>;

    void SetUp() override
    {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized)
        {
            GTEST_SKIP() << "GraphNative CudaHot/RocmWarm parity requires MPI initialization";
        }

        int rank = 0;
        int world_size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (world_size < cfg().mpi_ranks)
        {
            GTEST_SKIP() << "GraphNative CudaHot/RocmWarm parity requires "
                         << cfg().mpi_ranks << " MPI ranks (got " << world_size << ")";
        }

        mpi_ctx_ = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);

        if (isTopologySmokeTest())
            return;

        if (isProfilerTest() && !MoEExpertOverlayProfiler::isEnabled())
        {
            GTEST_SKIP() << "Set LLAMINAR_PROFILING=1 at process startup to test "
                            "all-GPU graph-native profiler rows";
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
            "; aborting MPI world to avoid stranding rocm_warm participant";
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
                "all overlay GPU tiers completed");
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
                "all overlay GPU tiers completed");
        }

        assertDecodeParity(summary);
    }

    std::shared_ptr<MoEExpertParallelPlan> overlay_plan_;
};

TEST_F(Qwen35MoEGraphNativeCudaHotRocmWarm, TopologySmoke)
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
        << "Plan construction threw for the all-GPU cuda_hot / rocm_warm layout";
    ASSERT_NE(plan, nullptr);

    EXPECT_TRUE(plan->isTieredOverlay());
    EXPECT_EQ(plan->continuation_domain, kCudaHotDomain);
    EXPECT_EQ(plan->shared_expert_domain, kCudaHotDomain);
    EXPECT_FALSE(plan->continuation_domain_spec.dense_tp_enabled);
    ASSERT_EQ(plan->domains.size(), 2u);
    ASSERT_EQ(plan->routed_tiers.size(), 2u);

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

    for (const auto &domain : plan->domains)
    {
        EXPECT_EQ(domain.compute_kind, ExpertDomainComputeKind::ReplicatedExperts)
            << "Domain '" << domain.name << "' must use whole-expert graph-native ownership";
        for (const auto &participant : domain.participants)
        {
            EXPECT_TRUE(participant.isCUDA() || participant.isROCm())
                << "All-GPU plan must not include CPU participants; got "
                << participant.toShortString();
        }
    }

    for (const auto &tier : plan->routed_tiers)
    {
        EXPECT_FALSE(tier.fallback)
            << "All-GPU cuda_hot / rocm_warm plan must not configure fallback=true tiers";
    }
    EXPECT_EQ(plan->routed_tiers[kCudaHotTierIndex].domain, kCudaHotDomain);
    EXPECT_EQ(plan->routed_tiers[kRocmWarmTierIndex].domain, kRocmWarmDomain);

    ASSERT_EQ(plan->placements.size(), static_cast<size_t>(metadata.num_layers));

    const auto counts = tierExpertCounts(*plan);
    ASSERT_EQ(counts.size(), 2u);
    const size_t expected_cuda_total =
        static_cast<size_t>(metadata.num_layers) * static_cast<size_t>(metadata.num_experts / 2);
    const size_t expected_rocm_total =
        static_cast<size_t>(metadata.num_layers) *
        static_cast<size_t>(metadata.num_experts - metadata.num_experts / 2);
    EXPECT_EQ(counts[kCudaHotTierIndex], expected_cuda_total);
    EXPECT_EQ(counts[kRocmWarmTierIndex], expected_rocm_total);
    EXPECT_EQ(counts[kCudaHotTierIndex] + counts[kRocmWarmTierIndex],
              static_cast<size_t>(metadata.num_experts) * static_cast<size_t>(metadata.num_layers));

    for (const auto &placement : plan->placements)
    {
        ASSERT_EQ(placement.routed_expert_tier.size(), static_cast<size_t>(metadata.num_experts));
        for (int expert = 0; expert < metadata.num_experts; ++expert)
        {
            const int expected_tier = expert < metadata.num_experts / 2
                                          ? kCudaHotTierIndex
                                          : kRocmWarmTierIndex;
            EXPECT_EQ(placement.routed_expert_tier[static_cast<size_t>(expert)], expected_tier)
                << "StaticById split changed at layer=" << placement.layer
                << " expert=" << expert;
        }
    }

    const auto owner_map = MoEExpertOwnerMap::build(*plan);
    ASSERT_EQ(owner_map.participants().size(), 2u);
    for (const auto &participant : owner_map.participants())
    {
        EXPECT_FALSE(participant.device.is_cpu())
            << "Production graph-native owner map must not include CPU expert participants";
        EXPECT_TRUE(participant.device.is_cuda() || participant.device.is_rocm())
            << "Unexpected participant device: " << participant.device.to_string();
    }

    for (int layer = 0; layer < metadata.num_layers; ++layer)
    {
        for (int expert = 0; expert < metadata.num_experts; ++expert)
        {
            ASSERT_EQ(owner_map.ownerCountForExpert(layer, expert), 1u)
                << "layer=" << layer << " expert=" << expert;
            const auto *owner = owner_map.ownerFor(layer, expert);
            ASSERT_NE(owner, nullptr) << "layer=" << layer << " expert=" << expert;
            EXPECT_FALSE(owner->device.is_cpu())
                << "CPU owner found in all-GPU plan at layer=" << layer
                << " expert=" << expert;
            EXPECT_TRUE(owner->device.is_cuda() || owner->device.is_rocm())
                << "Unexpected owner device at layer=" << layer
                << " expert=" << expert << ": " << owner->device.to_string();
            if (expert < metadata.num_experts / 2)
            {
                EXPECT_EQ(owner->domain_name, kCudaHotDomain);
                EXPECT_TRUE(owner->device.is_cuda());
            }
            else
            {
                EXPECT_EQ(owner->domain_name, kRocmWarmDomain);
                EXPECT_TRUE(owner->device.is_rocm());
            }
        }
    }

    LOG_INFO("[TopologySmoke] cuda_hot experts/layer: " << metadata.num_experts / 2
                                                        << ", rocm_warm experts/layer: "
                                                        << metadata.num_experts - metadata.num_experts / 2);
}

TEST_F(Qwen35MoEGraphNativeCudaHotRocmWarm, PrefillParity)
{
    runGraphNativePrefillParityBody();
}

TEST_F(Qwen35MoEGraphNativeCudaHotRocmWarm, DecodeParity)
{
    runGraphNativeDecodeParityBody();
}

TEST_F(Qwen35MoEGraphNativeCudaHotRocmWarm, ProfilerAllGpuNoFallbackRows)
{
    if (isLegacyOverlayRuntimeEnabled())
    {
        FAIL() << kLegacyEnvVar << " must NOT be set for graph-native profiler test.";
    }

    if (!MoEExpertOverlayProfiler::isEnabled())
    {
        GTEST_SKIP() << "Set LLAMINAR_PROFILING=1 at process startup to test "
                        "all-GPU graph-native profiler rows";
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

    for (const auto &row : rows)
    {
        EXPECT_EQ(row.cpu_fallback_rows, 0u)
            << "All-GPU graph-native overlay must not emit CPU fallback rows, phase="
            << row.phase << " domain=" << row.domain;
        if (row.phase == "gn_local_expert")
        {
            EXPECT_NE(row.domain_kind, "CPU")
                << "All local expert rows must execute on CUDA or ROCm participants";
        }
    }

    const bool has_sparse_compact = std::any_of(
        rows.begin(), rows.end(), [](const MoEExpertOverlayProfileRow &row)
        { return row.phase == "gn_sparse_dispatch" && row.compact_dispatch_bytes > 0; });
    EXPECT_TRUE(has_sparse_compact)
        << "Expected graph-native sparse dispatch profiler rows with compact bytes > 0";

    struct DispatchAggregate
    {
        size_t selected_rows = 0;
        size_t inbound_rows = 0;
        size_t compact_bytes = 0;
    };

    std::map<std::string, DispatchAggregate> warm_dispatch;
    for (const auto &row : rows)
    {
        if (row.phase != "gn_sparse_dispatch")
            continue;
        const auto tier_index = profilerTierIndex(row);
        if (!tier_index || *tier_index != kRocmWarmTierIndex)
            continue;

        auto &aggregate = warm_dispatch[row.domain];
        aggregate.selected_rows += row.selected_rows;
        aggregate.inbound_rows += row.inbound_rows;
        aggregate.compact_bytes += row.compact_dispatch_bytes;
    }

    bool checked_warm_dispatch = false;
    for (const auto &[domain, aggregate] : warm_dispatch)
    {
        if (aggregate.selected_rows == 0 && aggregate.inbound_rows == 0 && aggregate.compact_bytes == 0)
            continue;
        checked_warm_dispatch = true;
        EXPECT_EQ(aggregate.selected_rows, aggregate.inbound_rows)
            << "Warm-tier compact sparse dispatch row counts must not be multiplied by "
               "continuation TP degree; domain="
            << domain;
        EXPECT_GT(aggregate.compact_bytes, 0u)
            << "Warm-tier dispatch should use compact sparse payload bytes; domain=" << domain;
    }

    EXPECT_TRUE(checked_warm_dispatch)
        << "Expected at least one rocm_warm sparse dispatch aggregate to validate row counts";

    LOG_INFO("[ProfilerAllGpuNoFallbackRows] Total profiler rows: " << rows.size()
                                                                    << "; warm dispatch groups: "
                                                                    << warm_dispatch.size());
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "[Rank " << rank << "] Qwen3.5 MoE GraphNative CudaHot/RocmWarm parity test\n";

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