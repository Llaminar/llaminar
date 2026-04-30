/**
 * @file Test__Qwen35MoE_HybridPPTP_Parity.cpp
 * @brief Hybrid PP+TP parity coverage for Qwen3.5 MoE expert GPU-cache migration.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>
#include <cstdlib>
#include "Qwen35MoEParityTestBase.h"
#include "collective/BackendRouter.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/orchestrators/RankOrchestrator.h"
#include "utils/DebugEnv.h"

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35moe;

namespace
{
    const std::vector<std::string> kHybridPPTPMoEExcludedStages = {
        "Q_PROJECTION",
        "K_PROJECTION",
        "V_PROJECTION",
        "Q_NORM",
        "K_NORM",
        "Q_ROPE",
        "K_ROPE",
        "ATTENTION_CONTEXT",
        "FFN_GATE",
        "FFN_UP",
        "FFN_SWIGLU",
        "QKV_PROJECTION",
        "GDN_Z_PROJECTION",
        "GDN_DELTA_RULE_OUTPUT",
        "GDN_NORM_GATE_OUTPUT",
    };

    const std::vector<std::string> kHybridPPTPMoEAllreduceStages = {
        "MOE_EXPERT_OUTPUT",
        "MOE_SHARED_EXPERT_OUTPUT",
        "MOE_SHARED_GATE_OUTPUT",
        "MOE_COMBINED_OUTPUT",
    };

    const std::vector<TestConfig> kHybridPPTPMoEConfigs = {
        {
            .name = "LocalPP_TP_ROCm_CPUCache_TP_ROCm_CPU_35B_MoE",
            .devices = {ParityDeviceType::ROCm, ParityDeviceType::CPU, ParityDeviceType::ROCm, ParityDeviceType::CPU},
            .parallelism = Parallelism::LocalPP,
            .thresholds = {
                .cosine_threshold = 0.88f,
                .decode_cosine_threshold = 0.76f,
                .early_layers_count = 6,
                .min_early_layers_passed = 4,
                .kl_threshold = 0.06f,
                .excluded_stages = kHybridPPTPMoEExcludedStages,
                .allreduce_stages = kHybridPPTPMoEAllreduceStages,
                .min_top1_accuracy = 0.60f,
                .min_top5_accuracy = 0.80f,
                .pytorch_top1_in_topk = 5,
            },
            .skip_reason = "Blocked: mixed CPU/GPU HybridPPTP for Qwen3.5 MoE still needs generic GPU DeviceLoadPipeline coverage for dense FFN GEMMs before MoE expert-cache parity can run.",
            .model_path = "/opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
            .snapshot_dir = "pytorch_qwen35_moe_snapshots",
            .pp_stage_sizes = {2, 2},
            .tp_collective = Collective::HOST,
            .activation_precision = ActivationPrecision::FP32,
            .kv_cache_precision = KVCachePrecision::FP16,
            .decode_steps = 2,
        },
    };
}

class Qwen35MoEHybridPPTPParityTest
    : public Qwen35MoEConfigDrivenParityTest<Qwen35MoEHybridPPTPParityTest>,
      public ::testing::WithParamInterface<TestConfig>
{
public:
    const TestConfig &getTestConfig() const { return GetParam(); }

protected:
    using Base = Qwen35MoEConfigDrivenParityTest<Qwen35MoEHybridPPTPParityTest>;

    void SetUp() override
    {
        Base::SetUp();
        setenv("LLAMINAR_MOE_REBALANCE", "dynamic", 1);
        setenv("LLAMINAR_MOE_REBALANCE_WINDOW", "2", 1);
        setenv("LLAMINAR_MOE_GPU_EXPERT_CACHE", "8", 1);
        mutableDebugEnv().reload();
    }

    void TearDown() override
    {
        Base::TearDown();
        unsetenv("LLAMINAR_MOE_GPU_EXPERT_CACHE");
        unsetenv("LLAMINAR_MOE_REBALANCE_WINDOW");
        unsetenv("LLAMINAR_MOE_REBALANCE");
        mutableDebugEnv().reload();
    }

    bool warmGraphAndApplyStaticGpuExpertCache()
    {
        auto *rank = dynamic_cast<RankOrchestrator *>(this->runner_.get());
        if (!rank)
        {
            ADD_FAILURE() << "HybridPPTP test expected a RankOrchestrator root";
            return false;
        }

        if (!this->runner_->forward(this->config_.token_ids.data(), this->config_.token_ids.size()))
        {
            ADD_FAILURE() << "HybridPPTP graph warmup forward failed";
            return false;
        }

        constexpr int num_sockets = 2;
        constexpr int num_layers = 40;
        constexpr int num_experts = 256;
        const int cache_experts = mutableDebugEnv().moe_rebalance.gpu_cache_experts_per_layer;

        std::vector<std::vector<std::vector<bool>>> masks(
            num_sockets,
            std::vector<std::vector<bool>>(num_layers, std::vector<bool>(num_experts, false)));

        for (int layer = 0; layer < num_layers; ++layer)
        {
            for (int expert = 0; expert < num_experts; ++expert)
            {
                const bool cached_on_gpu = expert < cache_experts;
                masks[cached_on_gpu ? 0 : 1][layer][expert] = true;
            }
        }

        rank->applyMoEExpertMasksForAllDevices(masks);
        this->runner_->clear_cache();
        this->runner_->clearSnapshots();
        return true;
    }
};

TEST_P(Qwen35MoEHybridPPTPParityTest, PrefillParityWithGpuExpertCache)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    ASSERT_TRUE(warmGraphAndApplyStaticGpuExpertCache()) << "GPU expert cache mask application failed";

    auto summary = runPrefillParity();
    assertParity(summary);
}

TEST_P(Qwen35MoEHybridPPTPParityTest, DecodeParityWithGpuExpertCache)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    ASSERT_TRUE(warmGraphAndApplyStaticGpuExpertCache()) << "GPU expert cache mask application failed";

    auto summary = runDecodeParity();
    assertDecodeParity(summary);
}

TEST_P(Qwen35MoEHybridPPTPParityTest, SnapshotInfrastructureWithGpuExpertCache)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    ASSERT_TRUE(warmGraphAndApplyStaticGpuExpertCache()) << "GPU expert cache mask application failed";

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
}

INSTANTIATE_TEST_SUITE_P(
    Qwen35MoEHybridPPTP,
    Qwen35MoEHybridPPTPParityTest,
    ::testing::ValuesIn(kHybridPPTPMoEConfigs),
    [](const ::testing::TestParamInfo<TestConfig> &info)
    {
        return info.param.name;
    });

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();

    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
