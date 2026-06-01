/**
 * @file Test__Qwen35MoE_ExpertOverlay_Parity.cpp
 * @brief Qwen3.5 MoE parity tests for same-layer expert overlay plans.
 *
 * This is a V2 parity-harness test: it loads the real Qwen3.5 MoE GGUF,
 * regenerates/loads PyTorch snapshots through Qwen35MoEParityTestBase, injects
 * a planned MoEExpertParallelPlan into the production graph config, and compares
 * Llaminar prefill/decode snapshots against the PyTorch reference.
 *
 * Bridge Phase 5A audit contract: OverlayPlanTopology_* tests only prove that the
 * planned same-layer tier assignments are non-empty and stable. PrefillParity_*
 * and DecodeParity_* are the real V2 inference parity bodies and use the
 * production overlay domain-worker protocol for non-root CPU fallback ranks.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>

#include "Qwen35MoEParityTestBase.h"
#include "backends/ComputeBackend.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "execution/moe/MoEExpertParallelPlanner.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35moe;

namespace
{
    constexpr const char *kModelPath = "/opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf";
    constexpr const char *kSnapshotDir = "pytorch_qwen35_moe_snapshots";
    constexpr const char *kRocmSharedHotDomain = "rocm_shared_hot";
    constexpr const char *kCudaSharedHotDomain = "cuda_shared_hot";
    constexpr const char *kRocmHotDomain = "rocm_hot";
    constexpr const char *kCpuColdDomain = "cpu_cold";

    enum class OverlayTopologyKind
    {
        RocmSharedHotCpuCold,
        CudaSharedHotRocmHotCpuCold,
    };

    struct OverlayParityCase
    {
        std::string name;
        TestConfig config;
        OverlayTopologyKind topology;
    };

    size_t gib(size_t value)
    {
        return value * 1024ULL * 1024ULL * 1024ULL;
    }

    ExpertComputeDomain rocmLocalTPDomain(const std::string &name)
    {
        ExpertComputeDomain domain;
        domain.name = name;
        domain.kind = ExpertDomainKind::LocalTP;
        domain.backend = CollectiveBackendType::RCCL;
        domain.participants = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
        domain.owner_rank = 0;
        domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
        return domain;
    }

    ExpertComputeDomain cudaSharedHotDomain()
    {
        ExpertComputeDomain domain;
        domain.name = kCudaSharedHotDomain;
        domain.kind = ExpertDomainKind::SingleDevice;
        domain.backend = CollectiveBackendType::NCCL;
        domain.participants = {GlobalDeviceAddress::cuda(0)};
        domain.world_ranks = {0};
        domain.owner_rank = 0;
        domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
        return domain;
    }

    ExpertComputeDomain cpuNodeLocalTPDomain()
    {
        ExpertComputeDomain domain;
        domain.name = kCpuColdDomain;
        domain.kind = ExpertDomainKind::NodeLocalTP;
        domain.backend = CollectiveBackendType::UPI;
        domain.participants = {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)};
        domain.world_ranks = {0, 1};
        domain.owner_rank = 0;
        domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
        return domain;
    }

    ExpertRoutedTier tier(
        const std::string &name,
        const std::string &domain,
        int priority,
        int max_experts_per_layer,
        size_t memory_budget_bytes,
        bool fallback = false)
    {
        ExpertRoutedTier result;
        result.name = name;
        result.domain = domain;
        result.priority = priority;
        result.max_experts_per_layer = max_experts_per_layer;
        result.memory_budget_bytes = memory_budget_bytes;
        result.fallback = fallback;
        return result;
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

    std::string validationErrors(const MoEExpertParallelValidationResult &validation)
    {
        std::ostringstream message;
        for (const auto &error : validation.errors)
            message << "\n - " << error;
        return message.str();
    }

    MoEExpertParallelPlan requestedPlan(OverlayTopologyKind topology, const MoEExpertModelMetadata &metadata)
    {
        const int small_hot_capacity = std::max(1, metadata.num_experts / 8);
        const int medium_hot_capacity = std::max(1, metadata.num_experts / 4);

        MoEExpertParallelPlan plan;
        plan.enabled = true;
        plan.execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
        plan.residency_policy = ExpertResidencyPolicy::StaticById;

        switch (topology)
        {
        case OverlayTopologyKind::RocmSharedHotCpuCold:
            plan.continuation_domain = kRocmSharedHotDomain;
            plan.shared_expert_domain = kRocmSharedHotDomain;
            plan.domains = {
                rocmLocalTPDomain(kRocmSharedHotDomain),
                cpuNodeLocalTPDomain(),
            };
            plan.routed_tiers = {
                tier("shared_hot", kRocmSharedHotDomain, 0, medium_hot_capacity, gib(4)),
                tier("cold", kCpuColdDomain, 1, 0, 0, true),
            };
            break;
        case OverlayTopologyKind::CudaSharedHotRocmHotCpuCold:
            plan.continuation_domain = kCudaSharedHotDomain;
            plan.shared_expert_domain = kCudaSharedHotDomain;
            plan.domains = {
                cudaSharedHotDomain(),
                rocmLocalTPDomain(kRocmHotDomain),
                cpuNodeLocalTPDomain(),
            };
            plan.routed_tiers = {
                tier("shared_hottest", kCudaSharedHotDomain, 0, small_hot_capacity, gib(2)),
                tier("hot", kRocmHotDomain, 1, medium_hot_capacity, gib(4)),
                tier("cold", kCpuColdDomain, 2, 0, 0, true),
            };
            break;
        }

        return plan;
    }

    std::shared_ptr<MoEExpertParallelPlan> makeOverlayPlan(
        OverlayTopologyKind topology,
        const ModelContext &ctx)
    {
        const auto metadata = metadataFromModel(ctx);
        auto planned = MoEExpertParallelPlanner::plan(
                           requestedPlan(topology, metadata),
                           metadata)
                           .planned_plan;

        MoEExpertParallelValidationOptions options;
        options.layer_count = metadata.num_layers;
        options.routed_expert_count = metadata.num_experts;
        auto validation = validateMoEExpertParallelPlan(planned, options);
        if (!validation.ok())
        {
            throw std::invalid_argument("Invalid planned MoE expert overlay:" + validationErrors(validation));
        }

        return std::make_shared<MoEExpertParallelPlan>(std::move(planned));
    }

    TestConfig makeBaseConfig(const std::string &name)
    {
        TestConfig config;
        config.name = name;
        config.devices = {ParityDeviceType::CPU};
        config.parallelism = Parallelism::None;
        config.collective = Collective::None;
        config.thresholds = {
            .cosine_threshold = 0.90f,
            .decode_cosine_threshold = 0.80f,
            .early_layers_count = 6,
            .min_early_layers_passed = 5,
            .kl_threshold = 0.05f,
            .min_top1_accuracy = 0.80f,
            .min_top5_accuracy = 0.80f,
            .pytorch_top1_in_topk = 4,
        };
        config.mpi_ranks = 2;
        config.model_path = kModelPath;
        config.snapshot_dir = kSnapshotDir;
        config.activation_precision = ActivationPrecision::FP32;
        config.kv_cache_precision = KVCachePrecision::FP16;
        return config;
    }

    const std::vector<OverlayParityCase> kOverlayParityCases = {
        {
            .name = "ROCm2TP_SharedHot_CPU2NodeLocalTP_Cold",
            .config = makeBaseConfig("ExpertOverlay_ROCm2TP_SharedHot_CPU2NodeLocalTP_Cold"),
            .topology = OverlayTopologyKind::RocmSharedHotCpuCold,
        },
        {
            .name = "CUDA1_SharedHot_ROCm2TP_Hot_CPU2NodeLocalTP_Cold",
            .config = makeBaseConfig("ExpertOverlay_CUDA1_SharedHot_ROCm2TP_Hot_CPU2NodeLocalTP_Cold"),
            .topology = OverlayTopologyKind::CudaSharedHotRocmHotCpuCold,
        },
    };

    const OverlayParityCase &caseForCurrentTest()
    {
        const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        if (!test_info)
            return kOverlayParityCases.front();

        const std::string test_name = test_info->name();
        for (const auto &test_case : kOverlayParityCases)
        {
            if (test_name.find(test_case.name) != std::string::npos)
                return test_case;
        }

        return kOverlayParityCases.front();
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

    const ExpertComputeDomain *findDomain(
        const MoEExpertParallelPlan &plan,
        const std::string &domain_name)
    {
        auto it = std::find_if(plan.domains.begin(), plan.domains.end(),
                               [&domain_name](const ExpertComputeDomain &domain)
                               {
                                   return domain.name == domain_name;
                               });
        return it == plan.domains.end() ? nullptr : &(*it);
    }

    MoEExpertModelMetadata topologyOnlyMetadata()
    {
        MoEExpertModelMetadata metadata;
        metadata.num_experts = 256;
        return metadata;
    }

    std::shared_ptr<MoEExpertParallelPlan> makeTopologyOnlyOverlayPlan(OverlayTopologyKind topology)
    {
        return std::make_shared<MoEExpertParallelPlan>(requestedPlan(topology, topologyOnlyMetadata()));
    }

    std::optional<std::string> overlayHardwareBlocker(OverlayTopologyKind topology)
    {
        const int cuda_count = getCudaDeviceCount();
        const int rocm_count = getRocmDeviceCount();
        switch (topology)
        {
        case OverlayTopologyKind::RocmSharedHotCpuCold:
            if (rocm_count < 2)
                return "ExpertOverlay parity requires 2 ROCm devices, found " + std::to_string(rocm_count);
            break;
        case OverlayTopologyKind::CudaSharedHotRocmHotCpuCold:
            if (cuda_count < 1)
                return "ExpertOverlay parity requires 1 CUDA device, found " + std::to_string(cuda_count);
            if (rocm_count < 2)
                return "ExpertOverlay parity requires 2 ROCm devices, found " + std::to_string(rocm_count);
            break;
        }
        return std::nullopt;
    }

    std::string activationPrecisionConfigValue(ActivationPrecision precision)
    {
        std::string value = activationPrecisionToString(precision);
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        return value;
    }

    std::string kvCachePrecisionConfigValue(KVCachePrecision precision)
    {
        switch (precision)
        {
        case KVCachePrecision::AUTO:
            return "auto";
        case KVCachePrecision::FP32:
            return "fp32";
        case KVCachePrecision::FP16:
            return "fp16";
        case KVCachePrecision::Q8_1:
            return "q8_1";
        case KVCachePrecision::Q16_1:
            return "q16_1";
        case KVCachePrecision::TQ4:
            return "tq4";
        case KVCachePrecision::TQ:
            return "tq";
        }
        return "auto";
    }
} // namespace

class Qwen35MoEExpertOverlay
    : public Qwen35MoEConfigDrivenParityTest<Qwen35MoEExpertOverlay>
{
public:
    const TestConfig &getTestConfig() const { return caseForCurrentTest().config; }

protected:
    using Base = Qwen35MoEConfigDrivenParityTest<Qwen35MoEExpertOverlay>;

    void SetUp() override
    {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized)
        {
            // Harness precondition: these tests are discovered as MPI CTests, and
            // running outside MPI would not exercise the overlay audit surface.
            GTEST_SKIP() << "ExpertOverlay parity requires MPI initialization";
        }

        int rank = 0;
        int world_size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (world_size < cfg().mpi_ranks)
        {
            // Harness precondition: the CPU fallback tier is expressed as a
            // two-rank NodeLocalTP domain.
            GTEST_SKIP() << "ExpertOverlay parity requires " << cfg().mpi_ranks
                         << " MPI ranks (got " << world_size << ")";
        }

        if (auto blocker = overlayHardwareBlocker(caseForCurrentTest().topology))
            GTEST_SKIP() << *blocker;

        mpi_ctx_ = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);
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
        if (cfg().decode_steps > 0)
            config_.decode_steps = cfg().decode_steps;

        const auto metadata_path = std::filesystem::path(config_.snapshot_dir) / "metadata.txt";
        const bool metadata_missing = !std::filesystem::exists(metadata_path);
        const bool metadata_stale = !metadata_missing &&
                                    readSnapshotVersion(metadata_path) < kRequiredSnapshotVersion;
        const bool model_available = std::filesystem::exists(config_.model_path);
        const int local_needs_regen = (metadata_missing || metadata_stale) && model_available ? 1 : 0;
        int global_needs_regen = 0;
        MPI_Allreduce(&local_needs_regen, &global_needs_regen, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        int local_regen_failed = 0;
        if (global_needs_regen && isRank0())
        {
            LOG_INFO("[Qwen3.5 MoE ExpertOverlay] Regenerating snapshots once on rank 0");
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
            LOG_INFO("[Qwen3.5 MoE ExpertOverlay] Loaded " << config_.token_ids.size()
                                                           << " prefill token IDs from metadata");
        }
    }

    bool setupPipeline()
    {
        if (!isRootParityRank())
        {
            overlay_plan_ = makeTopologyOnlyOverlayPlan(caseForCurrentTest().topology);
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
            LOG_ERROR("[Qwen3.5 MoE ExpertOverlay] Failed to load model");
            return false;
        }

        configureModel(model_ctx_);

        try
        {
            overlay_plan_ = makeOverlayPlan(caseForCurrentTest().topology, *model_ctx_);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[Qwen3.5 MoE ExpertOverlay] " << e.what());
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

        runner_ = createInferenceRunner(model_ctx_, nullptr, DeviceId::cpu(), inf_config);
        if (!runner_)
        {
            LOG_ERROR("[Qwen3.5 MoE ExpertOverlay] Failed to create inference runner");
            return false;
        }

        runner_->enableSnapshotCapture();
        return true;
    }

    bool isRootParityRank() const
    {
        return isRank0();
    }

    bool synchronizeRanksOk(bool local_ok) const
    {
        int ok = local_ok ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        return ok == 1;
    }

    bool broadcastRootFlag(bool root_value) const
    {
        int flag = isRootParityRank() && root_value ? 1 : 0;
        MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        return flag != 0;
    }

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
               summary.top1_matches > 0 ||
               summary.top3_matches > 0 ||
               summary.top5_matches > 0;
    }

    [[noreturn]] void abortOverlayWorld(const std::string &reason) const
    {
        const std::string message = "[Qwen3.5 MoE ExpertOverlay] " + reason +
                                    "; aborting MPI world to avoid stranding CPU NodeLocalTP participants";
        LOG_ERROR(message);
        std::cerr << message << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
        std::abort();
    }

    [[noreturn]] void abortOverlayWorldAfterRootThrow(const char *phase, const std::string &what) const
    {
        abortOverlayWorld(std::string("root rank threw during ") + phase + ": " + what);
    }

    void runOverlayPrefillParityBody()
    {
        const bool setup_ok = setupPipeline();
        ASSERT_TRUE(synchronizeRanksOk(setup_ok)) << "Pipeline setup failed";

        if (!isRootParityRank())
        {
            return;
        }

        ParityTestSummary summary;
        try
        {
            summary = runPrefillParity();
        }
        catch (const std::exception &e)
        {
            abortOverlayWorldAfterRootThrow("prefill parity", e.what());
        }
        catch (...)
        {
            abortOverlayWorldAfterRootThrow("prefill parity", "unknown non-std exception");
        }
        if (!producedPrefillSummary(summary))
        {
            abortOverlayWorld("root rank produced no prefill parity summary, likely because forward failed before all overlay tiers completed");
        }
        assertParity(summary);
    }

    void runOverlayDecodeParityBody()
    {
        const bool setup_ok = setupPipeline();
        ASSERT_TRUE(synchronizeRanksOk(setup_ok)) << "Pipeline setup failed";

        if (!synchronizedDecodeWorkAvailable())
            GTEST_SKIP() << "Decode snapshots or decode token metadata are unavailable";

        if (!isRootParityRank())
        {
            return;
        }

        DecodeParitySummary summary;
        try
        {
            summary = runDecodeParity();
        }
        catch (const std::exception &e)
        {
            abortOverlayWorldAfterRootThrow("decode parity", e.what());
        }
        catch (...)
        {
            abortOverlayWorldAfterRootThrow("decode parity", "unknown non-std exception");
        }
        if (!producedDecodeSummary(summary))
        {
            abortOverlayWorld("root rank produced no decode parity summary, likely because forward failed before all overlay tiers completed");
        }
        assertDecodeParity(summary);
    }

    std::shared_ptr<MoEExpertParallelPlan> overlay_plan_;
};

TEST_F(Qwen35MoEExpertOverlay, OverlayPlanTopology_ROCm2TP_SharedHot_CPU2NodeLocalTP_Cold)
{
    if (!isRootParityRank())
        return;

    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    ASSERT_NE(overlay_plan_, nullptr);
    EXPECT_TRUE(overlay_plan_->isTieredOverlay());
    EXPECT_EQ(overlay_plan_->placements.size(), static_cast<size_t>(model_ctx_->totalBlockCount()));
    EXPECT_EQ(overlay_plan_->continuation_domain, kRocmSharedHotDomain);
    EXPECT_EQ(overlay_plan_->shared_expert_domain, kRocmSharedHotDomain);
    ASSERT_EQ(overlay_plan_->routed_tiers.size(), 2u);
    EXPECT_EQ(overlay_plan_->routed_tiers[0].memory_budget_bytes, gib(4));
    EXPECT_TRUE(overlay_plan_->routed_tiers.back().fallback);

    const auto counts = tierExpertCounts(*overlay_plan_);
    ASSERT_EQ(counts.size(), 2u);
    EXPECT_GT(counts[0], 0u);
    EXPECT_GT(counts[1], 0u);
}

TEST_F(Qwen35MoEExpertOverlay, OverlayPlanTopology_CUDA1_SharedHot_ROCm2TP_Hot_CPU2NodeLocalTP_Cold)
{
    if (!isRootParityRank())
        return;

    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    ASSERT_NE(overlay_plan_, nullptr);
    EXPECT_TRUE(overlay_plan_->isTieredOverlay());
    EXPECT_EQ(overlay_plan_->placements.size(), static_cast<size_t>(model_ctx_->totalBlockCount()));
    EXPECT_EQ(overlay_plan_->continuation_domain, kCudaSharedHotDomain);
    EXPECT_EQ(overlay_plan_->shared_expert_domain, kCudaSharedHotDomain);
    ASSERT_EQ(overlay_plan_->routed_tiers.size(), 3u);
    EXPECT_EQ(overlay_plan_->routed_tiers[0].memory_budget_bytes, gib(2));
    EXPECT_EQ(overlay_plan_->routed_tiers[1].memory_budget_bytes, gib(4));
    EXPECT_TRUE(overlay_plan_->routed_tiers.back().fallback);

    const auto counts = tierExpertCounts(*overlay_plan_);
    ASSERT_EQ(counts.size(), 3u);
    EXPECT_GT(counts[0], 0u);
    EXPECT_GT(counts[1], 0u);
    EXPECT_GT(counts[2], 0u);
}

TEST_F(Qwen35MoEExpertOverlay, PrefillParity_ROCm2TP_SharedHot_CPU2NodeLocalTP_Cold)
{
    runOverlayPrefillParityBody();
}

TEST_F(Qwen35MoEExpertOverlay, DecodeParity_ROCm2TP_SharedHot_CPU2NodeLocalTP_Cold)
{
    runOverlayDecodeParityBody();
}

TEST_F(Qwen35MoEExpertOverlay, PrefillParity_CUDA1_SharedHot_ROCm2TP_Hot_CPU2NodeLocalTP_Cold)
{
    runOverlayPrefillParityBody();
}

TEST_F(Qwen35MoEExpertOverlay, DecodeParity_CUDA1_SharedHot_ROCm2TP_Hot_CPU2NodeLocalTP_Cold)
{
    runOverlayDecodeParityBody();
}

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (rank == 0)
    {
        std::cout << "Qwen3.5 MoE ExpertOverlay V2 parity suite: MPI world size="
                  << world_size << ", thread support="
                  << (provided >= MPI_THREAD_MULTIPLE ? "MPI_THREAD_MULTIPLE" : "limited")
                  << std::endl;
    }

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    int global_result = 0;
    MPI_Allreduce(&result, &global_result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();

    std::cout.flush();
    std::cerr.flush();
    _exit(global_result);
}
