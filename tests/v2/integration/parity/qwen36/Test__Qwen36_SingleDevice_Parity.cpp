/**
 * @file Test__Qwen36_SingleDevice_Parity.cpp
 * @brief Classic single-device Qwen3.6 dense math parity tests.
 *
 * These tests mirror the Qwen3.5 and Qwen3.6 MoE layer-by-layer parity
 * harnesses.  The prefix/MTP parity suite proves request-level behavior; this
 * suite proves the underlying dense graph math by comparing PyTorch snapshots
 * against Llaminar snapshots for prefill, decode, and snapshot availability on
 * CPU, CUDA, and ROCm.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>

#include "../qwen35/Qwen35ParityTestBase.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"

#include <algorithm>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35;

namespace
{
    /**
     * @brief Tight first-pass thresholds for the 27B Q4_K_S dense fixture.
     *
     * The PyTorch generator dequantizes the same GGUF weights, so large drift
     * is a real implementation signal.  The tolerances still leave a modest
     * buffer for backend reduction order and quantized-kernel accumulation
     * differences, especially through recurrent GDN decode state.
     */
    static const auto kQwen36Dense27BStrictSingleDeviceThresholds = BackendThresholds{
        .cosine_threshold = 0.96f,
        .decode_cosine_threshold = 0.93f,
        .early_layers_count = 8,
        .min_early_layers_passed = 8,
        .kl_threshold = 0.08f,
        .min_top1_accuracy = 80.0f,
        .min_top5_accuracy = 80.0f,
        .pytorch_top1_in_topk = 3,
    };

    /**
     * @brief Symmetric backend matrix for dense Qwen3.6 classic parity.
     *
     * Keep CPU, CUDA, and ROCm entries structurally identical unless a measured
     * backend difference has a documented reason.  This suite is intended to
     * prevent hidden backend drift while MTP depth policies are tuned.
     */
    static const std::vector<TestConfig> kQwen36DenseSingleDeviceConfigs = {
        {
            .name = "Qwen36Dense_27B_CPU_KV_FP16",
            .devices = {ParityDeviceType::CPU},
            .parallelism = Parallelism::None,
            .collective = Collective::None,
            .thresholds = kQwen36Dense27BStrictSingleDeviceThresholds,
            .model_path = "/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf",
            .snapshot_dir = "pytorch_qwen36_dense_singledevice_snapshots",
            .activation_precision = ActivationPrecision::FP32,
            .kv_cache_precision = KVCachePrecision::FP16,
        },
        {
            .name = "Qwen36Dense_27B_CUDA_KV_FP16",
            .devices = {ParityDeviceType::CUDA},
            .parallelism = Parallelism::None,
            .collective = Collective::None,
            .thresholds = kQwen36Dense27BStrictSingleDeviceThresholds,
            .model_path = "/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf",
            .snapshot_dir = "pytorch_qwen36_dense_singledevice_snapshots",
            .activation_precision = ActivationPrecision::FP32,
            .kv_cache_precision = KVCachePrecision::FP16,
        },
        {
            .name = "Qwen36Dense_27B_ROCm_KV_FP16",
            .devices = {ParityDeviceType::ROCm},
            .parallelism = Parallelism::None,
            .collective = Collective::None,
            .thresholds = kQwen36Dense27BStrictSingleDeviceThresholds,
            .model_path = "/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf",
            .snapshot_dir = "pytorch_qwen36_dense_singledevice_snapshots",
            .activation_precision = ActivationPrecision::FP32,
            .kv_cache_precision = KVCachePrecision::FP16,
        },
    };
} // namespace

/**
 * @brief Parameterized wrapper around the Qwen3.5 hybrid-GDN parity base.
 *
 * Qwen3.6 dense GGUFs use the same hybrid GDN/full-attention graph family as
 * Qwen3.5, with trailing MTP sidecar tensors.  The classic math suite disables
 * MTP and validates the main graph through the existing Qwen3.5 reference
 * generator and CSV-export machinery.
 */
class Qwen36DenseSingleDeviceParityTest
    : public Qwen35ConfigDrivenParityTest<Qwen36DenseSingleDeviceParityTest>,
      public ::testing::WithParamInterface<TestConfig>
{
public:
    const TestConfig &getTestConfig() const { return GetParam(); }
};

TEST_P(Qwen36DenseSingleDeviceParityTest, PrefillParity)
{
    auto summary = runSingleDevicePrefillParity();
    assertParity(summary);
}

TEST_P(Qwen36DenseSingleDeviceParityTest, DecodeParity)
{
    auto summary = runSingleDeviceDecodeParity();
    assertDecodeParity(summary);
}

TEST_P(Qwen36DenseSingleDeviceParityTest, SnapshotInfrastructure)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";

    auto embedding = loadPyTorchSnapshot("EMBEDDING");
    ASSERT_FALSE(embedding.empty()) << "Failed to load EMBEDDING snapshot";

    ASSERT_NE(runner_, nullptr);
    ASSERT_TRUE(runner_->forward(config_.token_ids.data(), config_.token_ids.size()));

    auto keys = runner_->getSnapshotKeys();
    EXPECT_GT(keys.size(), 0) << "No snapshots captured";

    EXPECT_NE(std::find(keys.begin(), keys.end(), "EMBEDDING"), keys.end())
        << "Missing EMBEDDING snapshot";
    EXPECT_NE(std::find(keys.begin(), keys.end(), "LM_HEAD"), keys.end())
        << "Missing LM_HEAD snapshot";

    bool has_gdn_state = false;
    for (const auto &key : keys)
    {
        if (key.find("GDN_DELTA_RULE_OUTPUT") != std::string::npos ||
            key.find("GDN_NORM_GATE_OUTPUT") != std::string::npos)
        {
            has_gdn_state = true;
            break;
        }
    }
    EXPECT_TRUE(has_gdn_state)
        << "Dense Qwen3.6 snapshots should include GDN stage outputs";
}

INSTANTIATE_TEST_SUITE_P(
    Qwen36Dense,
    Qwen36DenseSingleDeviceParityTest,
    ::testing::ValuesIn(kQwen36DenseSingleDeviceConfigs),
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
