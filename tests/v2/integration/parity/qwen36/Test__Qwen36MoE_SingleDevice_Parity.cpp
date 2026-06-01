/**
 * @file Test__Qwen36MoE_SingleDevice_Parity.cpp
 * @brief Single-device Qwen3.6 MoE parity tests (CPU, CUDA, ROCm)
 *
 * Mirrors the Qwen3.5 MoE layer-by-layer parity harness so Qwen3.6 MoE
 * divergences produce the usual snapshot CSV diagnostics for prefill/decode.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>

#include "../qwen35moe/Qwen35MoEParityTestBase.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35moe;

namespace
{
    static const auto kQwen36MoE35BStrictSingleDeviceThresholds = BackendThresholds{
        .cosine_threshold = 0.96f,
        .decode_cosine_threshold = 0.98f,
        .early_layers_count = 6,
        .min_early_layers_passed = 5,
        .kl_threshold = 0.03f,
        .min_top1_accuracy = 80.0f,
        .min_top5_accuracy = 60.0f,
        .pytorch_top1_in_topk = 3,
    };

    static const std::vector<TestConfig> kQwen36MoESingleDeviceConfigs = {
        {
            .name = "Qwen36MoE_35B_CPU_KV_FP16",
            .devices = {ParityDeviceType::CPU},
            .parallelism = Parallelism::None,
            .collective = Collective::None,
            .thresholds = kQwen36MoE35BStrictSingleDeviceThresholds,
            .model_path = "/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf",
            .snapshot_dir = "pytorch_qwen36_moe_singledevice_cpu_snapshots",
            .activation_precision = ActivationPrecision::FP32,
            .kv_cache_precision = KVCachePrecision::FP16,
        },
        {
            .name = "Qwen36MoE_35B_CUDA_KV_FP16",
            .devices = {ParityDeviceType::CUDA},
            .parallelism = Parallelism::None,
            .collective = Collective::None,
            .thresholds = kQwen36MoE35BStrictSingleDeviceThresholds,
            .model_path = "/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf",
            .snapshot_dir = "pytorch_qwen36_moe_singledevice_cuda_snapshots",
            .activation_precision = ActivationPrecision::FP32,
            .kv_cache_precision = KVCachePrecision::FP16,
        },
        {
            .name = "Qwen36MoE_35B_ROCm_KV_FP16",
            .devices = {ParityDeviceType::ROCm},
            .parallelism = Parallelism::None,
            .collective = Collective::None,
            .thresholds = kQwen36MoE35BStrictSingleDeviceThresholds,
            .model_path = "/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf",
            .snapshot_dir = "pytorch_qwen36_moe_singledevice_rocm_snapshots",
            .activation_precision = ActivationPrecision::FP32,
            .kv_cache_precision = KVCachePrecision::FP16,
        },
    };
}

class Qwen36MoESingleDeviceParityTest
    : public Qwen35MoEConfigDrivenParityTest<Qwen36MoESingleDeviceParityTest>,
      public ::testing::WithParamInterface<TestConfig>
{
public:
    const TestConfig &getTestConfig() const { return GetParam(); }
};

TEST_P(Qwen36MoESingleDeviceParityTest, PrefillParity)
{
    auto summary = runSingleDevicePrefillParity();
    assertParity(summary);
}

TEST_P(Qwen36MoESingleDeviceParityTest, DecodeParity)
{
    auto summary = runSingleDeviceDecodeParity();
    assertDecodeParity(summary);
}

TEST_P(Qwen36MoESingleDeviceParityTest, SnapshotInfrastructure)
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
    Qwen36MoE,
    Qwen36MoESingleDeviceParityTest,
    ::testing::ValuesIn(kQwen36MoESingleDeviceConfigs),
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
