/**
 * @file Test__Qwen35MoE_LocalTP_Parity.cpp
 * @brief Local Tensor Parallelism parity tests for Qwen3.5 MoE on ROCm.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>
#include "Qwen35MoEParityTestBase.h"
#include "collective/BackendRouter.h"
#include "backends/GPUDeviceContextPool.h"

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35moe;

namespace
{
    const std::vector<std::string> kLocalTPMoEExcludedStages = {
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
        /*
         * ExpertOverlay reduces the routed and shared branches through typed
         * rooted collectives, then publishes one replicated canonical
         * MOE_COMBINED_OUTPUT. Branch-local buffers are deliberately not
         * semantic post-collective checkpoints in that production DAG.
         */
        "MOE_EXPERT_OUTPUT",
        "MOE_SHARED_EXPERT_OUTPUT",
        "MOE_SHARED_GATE_OUTPUT",
    };

    const std::vector<TestConfig> kLocalTPMoEConfigs = {
        {
            .name = "LocalTP_RCCL_2xROCm_35B_MoE",
            .devices = {ParityDeviceType::ROCm, ParityDeviceType::ROCm},
            .parallelism = Parallelism::LocalTP,
            .collective = Collective::RCCL,
            .thresholds = {
                .cosine_threshold = 0.90f,
                .decode_cosine_threshold = 0.80f,
                .early_layers_count = 6,
                .min_early_layers_passed = 5,
                .kl_threshold = 0.05f,
                .excluded_stages = kLocalTPMoEExcludedStages,
                /*
                 * MOE_COMBINED_OUTPUT is already the post-publication tensor.
                 * Requiring a second `_ALLREDUCED` alias would describe the
                 * retired ordinary-TP graph rather than the live sparse path.
                 */
                .allreduce_stages = {},
                .min_top1_accuracy = 0.80f,
                .min_top5_accuracy = 0.80f,
                .pytorch_top1_in_topk = 4,
            },
            .model_path = "/opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
            .snapshot_dir = "pytorch_qwen35_moe_snapshots",
            .activation_precision = ActivationPrecision::FP32,
            .kv_cache_precision = KVCachePrecision::FP16,
        },
    };
}


class Qwen35MoELocalTPParityTest
    : public Qwen35MoEConfigDrivenParityTest<Qwen35MoELocalTPParityTest>,
      public ::testing::WithParamInterface<TestConfig>
{
public:
    const TestConfig &getTestConfig() const { return GetParam(); }
};

TEST_P(Qwen35MoELocalTPParityTest, ProductionParity)
{
    runProductionParityCampaign();
}

INSTANTIATE_TEST_SUITE_P(
    Qwen35MoELocalTP,
    Qwen35MoELocalTPParityTest,
    ::testing::ValuesIn(kLocalTPMoEConfigs),
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
