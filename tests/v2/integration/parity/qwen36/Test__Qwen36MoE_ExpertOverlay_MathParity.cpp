/**
 * @file Test__Qwen36MoE_ExpertOverlay_MathParity.cpp
 * @brief PyTorch snapshot parity for Qwen3.6 MoE homogeneous 2-GPU expert overlays.
 *
 * These tests are the quality gate for the GPU MoE rebalancing sprint target:
 * one LocalTP routed expert domain on either 2x CUDA/NCCL or 2x ROCm/RCCL,
 * phase-split dense policy, and fp16 TP allreduce transport.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>

#include "../qwen35moe/Qwen35MoEParityTestBase.h"
#include "Qwen36MoEParityTestBase.h"
#include "backends/DeviceAddressAdapter.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"
#include "collective/LocalTPContext.h"
#include "execution/moe/MoEExpertParallelPlanner.h"
#include "execution/mtp/MTPWeightManifest.h"
#include "utils/Logger.h"

#include <algorithm>
#include <filesystem>
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
using namespace llaminar2::test::parity::qwen36;

namespace
{
    using PlanFactory = std::shared_ptr<MoEExpertParallelPlan> (*)();

    const std::vector<std::string> kOverlayExcludedStages = {
        "Q_PROJECTION",
        "K_PROJECTION",
        "V_PROJECTION",
        "Q_NORM",
        "K_NORM",
        "Q_ROPE",
        "K_ROPE",
        "ATTENTION_CONTEXT",
        "FA_GATE",
        "ATTENTION_CONTEXT_GATED",
        "FFN_GATE",
        "FFN_UP",
        "FFN_SWIGLU",
        "QKV_PROJECTION",
        "GDN_CONV1D_OUTPUT",
        "GDN_Z_PROJECTION",
        "GDN_DELTA_RULE_OUTPUT",
        "GDN_NORM_GATE_OUTPUT",
        "MOE_EXPERT_OUTPUT",
        "MOE_SHARED_EXPERT_OUTPUT",
        "MOE_SHARED_GATE_OUTPUT",
    };

    const std::vector<std::string> kOverlayAllreduceStages = {
        "MOE_COMBINED_OUTPUT",
    };

    BackendThresholds qwen36MoEOverlayThresholds()
    {
        return {
            .cosine_threshold = 0.90f,
            .decode_cosine_threshold = 0.80f,
            .early_layers_count = 6,
            .min_early_layers_passed = 5,
            .kl_threshold = 0.05f,
            .excluded_stages = kOverlayExcludedStages,
            .allreduce_stages = kOverlayAllreduceStages,
            .min_top1_accuracy = 80.0f,
            .min_top5_accuracy = 60.0f,
            .pytorch_top1_in_topk = 4,
        };
    }

    TestConfig baseConfig(
        const std::string &name,
        std::vector<ParityDeviceType> devices,
        Collective backend,
        const std::string &snapshot_dir)
    {
        TestConfig config;
        config.name = name;
        config.devices = std::move(devices);
        config.parallelism = Parallelism::None;
        config.collective = backend;
        config.thresholds = qwen36MoEOverlayThresholds();
        config.model_path = "/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf";
        config.snapshot_dir = snapshot_dir;
        config.activation_precision = ActivationPrecision::FP32;
        config.kv_cache_precision = KVCachePrecision::FP16;
        config.decode_steps = 3;
        return config;
    }

    const std::vector<TestConfig> kQwen36MoEExpertOverlayConfigs = {
        baseConfig(
            "Qwen36MoE_ExpertOverlay_CUDA2TP_HotOnly_PhaseSplit_FP16Transport",
            {ParityDeviceType::CUDA, ParityDeviceType::CUDA},
            Collective::NCCL,
            "pytorch_qwen36_moe_singledevice_cuda_snapshots"),
        baseConfig(
            "Qwen36MoE_ExpertOverlay_ROCm2TP_HotOnly_PhaseSplit_FP16Transport",
            {ParityDeviceType::ROCm, ParityDeviceType::ROCm},
            Collective::RCCL,
            "pytorch_qwen36_moe_singledevice_rocm_snapshots"),
    };

    PlanFactory planFactoryForConfig(const TestConfig &config)
    {
        if (config.name.find("CUDA2TP") != std::string::npos)
            return qwen36MoEOverlayPlanCuda2TPHotOnly;
        if (config.name.find("ROCm2TP") != std::string::npos)
            return qwen36MoEOverlayPlanRocm2TPHotOnly;
        throw std::invalid_argument(
            "no Qwen3.6 MoE expert overlay plan factory for config '" +
            config.name + "'");
    }

    std::optional<std::string> expertOverlayHardwareBlocker(const TestConfig &config)
    {
        if (auto blocker = checkHardwareAvailability(config))
            return blocker;
#ifndef HAVE_NCCL
        if (config.collective == Collective::NCCL)
            return "NCCL not available";
#endif
#ifndef HAVE_RCCL
        if (config.collective == Collective::RCCL)
            return "RCCL not available";
#endif
        return std::nullopt;
    }

    MoEExpertModelMetadata metadataFromModel(const ModelContext &ctx)
    {
        const auto &loader = ctx.concreteLoader();
        const std::string &arch = ctx.architecture();

        MoEExpertModelMetadata metadata;
        metadata.num_layers = mainLayerCountExcludingMTP(
            loader,
            arch,
            ctx.blockCount());
        metadata.num_experts = loader.getInt(arch + ".expert_count", 0);
        metadata.d_model = ctx.embeddingLength();
        metadata.routed_intermediate_size =
            loader.getInt(arch + ".expert_feed_forward_length", 0);
        if (metadata.routed_intermediate_size == 0)
            metadata.routed_intermediate_size = ctx.feedForwardLength();
        metadata.has_shared_expert = loader.getInt(arch + ".expert_shared_count", 0) > 0;
        metadata.shared_intermediate_size = metadata.has_shared_expert
                                                ? metadata.routed_intermediate_size
                                                : 0;
        metadata.routed_quant_type = "IQ3";
        metadata.shared_quant_type = "IQ3";
        return metadata;
    }

    std::string validationErrors(const MoEExpertParallelValidationResult &validation)
    {
        std::ostringstream message;
        for (const auto &error : validation.errors)
            message << "\n - " << error;
        return message.str();
    }

    std::shared_ptr<MoEExpertParallelPlan> makePlannedOverlayPlan(
        const TestConfig &config,
        const ModelContext &ctx)
    {
        const auto metadata = metadataFromModel(ctx);
        auto requested = planFactoryForConfig(config)();
        if (!requested)
            throw std::invalid_argument("overlay parity plan factory returned null");

        auto planned = MoEExpertParallelPlanner::plan(*requested, metadata).planned_plan;

        MoEExpertParallelValidationOptions options;
        options.layer_count = metadata.num_layers;
        options.routed_expert_count = metadata.num_experts;
        auto validation = validateMoEExpertParallelPlan(planned, options);
        if (!validation.ok())
        {
            throw std::invalid_argument(
                "invalid planned Qwen3.6 MoE expert overlay:" +
                validationErrors(validation));
        }

        return std::make_shared<MoEExpertParallelPlan>(std::move(planned));
    }

    DeviceId continuationRootDevice(const MoEExpertParallelPlan &plan)
    {
        for (const auto &domain : plan.domains)
        {
            if (domain.name != plan.continuation_domain)
                continue;
            if (domain.participants.empty())
            {
                throw std::runtime_error(
                    "continuation domain '" + plan.continuation_domain +
                    "' has no participants");
            }

            const int root = std::clamp(
                plan.continuation_domain_spec.logical_root_participant,
                0,
                static_cast<int>(domain.participants.size()) - 1);
            return DeviceAddressAdapter::toDeviceId(
                domain.participants[static_cast<size_t>(root)]);
        }

        throw std::runtime_error(
            "continuation domain '" + plan.continuation_domain +
            "' was not found in overlay plan");
    }

    const ExpertComputeDomain &continuationDomain(const MoEExpertParallelPlan &plan)
    {
        for (const auto &domain : plan.domains)
        {
            if (domain.name == plan.continuation_domain)
                return domain;
        }

        throw std::runtime_error(
            "continuation domain '" + plan.continuation_domain +
            "' was not found in overlay plan");
    }

    std::vector<float> equalWeights(size_t count)
    {
        if (count == 0)
            return {};
        return std::vector<float>(count, 1.0f / static_cast<float>(count));
    }
} // namespace

class Qwen36MoEExpertOverlayParityTest
    : public Qwen35MoEConfigDrivenParityTest<Qwen36MoEExpertOverlayParityTest>,
      public ::testing::WithParamInterface<TestConfig>
{
public:
    const TestConfig &getTestConfig() const { return GetParam(); }

protected:
    using Base = Qwen35MoEConfigDrivenParityTest<Qwen36MoEExpertOverlayParityTest>;

    void SetUp() override
    {
        if (auto blocker = expertOverlayHardwareBlocker(GetParam()))
            GTEST_SKIP() << GetParam().name << " " << *blocker;
        Base::SetUp();
    }

    bool setupPipeline()
    {
        DeviceManager::instance().initialize(-1);

        model_ctx_ = ModelContext::create(
            config_.model_path,
            nullptr,
            nullptr,
            nullptr,
            WeightDistributionStrategy::SHARDED);
        if (!model_ctx_)
        {
            LOG_ERROR("[Qwen3.6 MoE ExpertOverlay MathParity] Failed to load model");
            return false;
        }

        configureModel(model_ctx_);

        try
        {
            overlay_plan_ = makePlannedOverlayPlan(GetParam(), *model_ctx_);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[Qwen3.6 MoE ExpertOverlay MathParity] " << e.what());
            return false;
        }

        InferenceRunnerConfig inf_config;
        inf_config.max_seq_len = 4096;
        inf_config.batch_size = 1;
        inf_config.force_graph = true;
        inf_config.activation_precision = GetParam().activation_precision;
        inf_config.kv_cache_precision = GetParam().kv_cache_precision;
        // Preserve Qwen35Graph's hybrid schema policy: early GDN layers and all
        // FA layers use FP32 allreduce, later GDN layers use FP16 transport.
        inf_config.tp_allreduce_precision_override = "schema";
        inf_config.use_mapped_memory = true;
        inf_config.moe_expert_parallel_plan = overlay_plan_;
        inf_config.moe_expert_overlay_mpi_ctx = mpi_ctx_;

        const ExpertComputeDomain *domain = nullptr;
        try
        {
            (void)continuationRootDevice(*overlay_plan_);
            domain = &continuationDomain(*overlay_plan_);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[Qwen3.6 MoE ExpertOverlay MathParity] " << e.what());
            return false;
        }

        if (!domain || domain->kind != ExpertDomainKind::LocalTP || domain->participants.size() < 2)
        {
            LOG_ERROR("[Qwen3.6 MoE ExpertOverlay MathParity] continuation domain must be a multi-device LocalTP domain");
            return false;
        }

        const std::vector<float> weights =
            domain->weights.empty() ? equalWeights(domain->participants.size()) : domain->weights;
        auto tp_ctx = createLocalTPContext(
            domain->participants,
            weights,
            domain->backend);
        if (!tp_ctx)
        {
            LOG_ERROR("[Qwen3.6 MoE ExpertOverlay MathParity] Failed to create LocalTP context");
            return false;
        }

        RankOrchestrator::Config rank_config;
        rank_config.devices = domain->participants;
        rank_config.weights = weights;
        rank_config.backend = domain->backend;
        rank_config.max_seq_len = inf_config.max_seq_len;
        rank_config.batch_size = inf_config.batch_size;
        rank_config.activation_precision = inf_config.activation_precision;
        rank_config.kv_cache_precision = inf_config.kv_cache_precision;
        rank_config.tp_allreduce_precision_override = inf_config.tp_allreduce_precision_override;
        rank_config.use_mapped_memory = inf_config.use_mapped_memory;
        rank_config.moe_expert_parallel_plan = overlay_plan_;
        rank_config.moe_expert_overlay_mpi_ctx = mpi_ctx_;

        runner_ = createRankOrchestrator(
            model_ctx_,
            std::move(tp_ctx),
            rank_config);
        if (!runner_)
        {
            LOG_ERROR("[Qwen3.6 MoE ExpertOverlay MathParity] Failed to create runner");
            return false;
        }

        runner_->enableSnapshotCapture();
        return true;
    }

    bool decodeWorkAvailable()
    {
        return !loadPyTorchSnapshot("decode_step0_LM_HEAD").empty() &&
               !readDecodeTokensFromMetadata().empty();
    }

    std::shared_ptr<MoEExpertParallelPlan> overlay_plan_;
};

TEST_P(Qwen36MoEExpertOverlayParityTest, PrefillParity)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    auto summary = runPrefillParity();
    assertParity(summary);
}

TEST_P(Qwen36MoEExpertOverlayParityTest, DecodeParity)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    if (!decodeWorkAvailable())
        GTEST_SKIP() << "Decode snapshots or decode token metadata are unavailable";
    auto summary = runDecodeParity();
    assertDecodeParity(summary);
}

TEST_P(Qwen36MoEExpertOverlayParityTest, SnapshotInfrastructure)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";

    auto embedding = loadPyTorchSnapshot("EMBEDDING");
    ASSERT_FALSE(embedding.empty()) << "Failed to load EMBEDDING snapshot";

    ASSERT_TRUE(runner_ != nullptr);
    runner_->forward(config_.token_ids.data(), config_.token_ids.size());

    auto keys = runner_->getSnapshotKeys();
    EXPECT_GT(keys.size(), 0) << "No snapshots captured";

    EXPECT_NE(std::find(keys.begin(), keys.end(), "EMBEDDING"), keys.end())
        << "Missing EMBEDDING snapshot";
    EXPECT_NE(std::find(keys.begin(), keys.end(), "LM_HEAD"), keys.end())
        << "Missing LM_HEAD snapshot";

    bool has_ffn_residual = false;
    for (const auto &key : keys)
    {
        if (key.find("FFN_RESIDUAL") != std::string::npos)
        {
            has_ffn_residual = true;
            break;
        }
    }
    EXPECT_TRUE(has_ffn_residual) << "Missing FFN_RESIDUAL snapshot";
}

INSTANTIATE_TEST_SUITE_P(
    Qwen36MoEExpertOverlay,
    Qwen36MoEExpertOverlayParityTest,
    ::testing::ValuesIn(kQwen36MoEExpertOverlayConfigs),
    [](const ::testing::TestParamInfo<TestConfig> &info)
    {
        return info.param.name;
    });

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    initializeLogging();
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
