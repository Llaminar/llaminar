/**
 * @file Test__Qwen2_SingleDevice_Parity.cpp
 * @brief Canonically generated single-device Qwen2 real-weight parity matrix.
 *
 * Model artifacts, physical topology, activation/KV precision, and numerical
 * contracts are declared as typed data. The shared model-parity generator
 * expands those declarations and the ordinary production fixture retains every
 * prefill/decode checkpoint comparison and CSV artifact.
 */

#include "../ModelParityDefinition.h"
#include "Qwen2ParityTestBase.h"
#include "Qwen2ModelParityDefinitions.h"

#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>

#include <iostream>
#include <iterator>
#include <utility>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen2;

namespace
{
    /** @return Shared Qwen2 numerical contract with explicit quality limits. */
    BackendThresholds qwen2Thresholds(
        float kl_threshold,
        float min_top1_accuracy,
        float min_top5_accuracy,
        int pytorch_top1_in_topk = 3)
    {
        return BackendThresholds{
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.95f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = kl_threshold,
            .min_top1_accuracy = min_top1_accuracy,
            .min_top5_accuracy = min_top5_accuracy,
            .pytorch_top1_in_topk = pytorch_top1_in_topk,
        };
    }

    /** @return Exact one-participant topology for one backend. */
    ModelParityTopologyDefinition singleDeviceTopology(
        std::string test_id,
        GlobalDeviceAddress address)
    {
        return ModelParityTopologyDefinition{
            .test_id = std::move(test_id),
            .kind = ModelParityTopologyKind::SingleDevice,
            .participants = {
                ModelParityParticipant{
                    .address = std::move(address),
                    .world_rank = 0,
                },
            },
            .collective = Collective::None,
            .mpi_ranks = 1,
        };
    }

    /** @return One model/topology definition with a generated KV axis. */
    ModelParityDefinition singleDeviceDefinition(
        ModelParityModelDefinition model,
        ModelParityTopologyDefinition topology,
        BackendThresholds thresholds,
        std::vector<KVCachePrecision> kv_precisions,
        std::vector<ModelParityPrecisionThresholdOverride> overrides = {})
    {
        ModelParityDefinition definition;
        definition.model = std::move(model);
        definition.topology = std::move(topology);
        definition.thresholds = std::move(thresholds);
        definition.precisions.activation = {ActivationPrecision::FP32};
        definition.precisions.kv_cache = std::move(kv_precisions);
        definition.precisions.threshold_overrides = std::move(overrides);
        return definition;
    }

    /** @return Exact typed declarations whose expansion replaces 18 records. */
    std::vector<ModelParityDefinition> qwen2SingleDeviceDefinitions()
    {
        const auto q4_model = qwen2Q40ParityModel();
        const auto q8_model = qwen2ParityModel(
            "Qwen2_Q8_0",
            "models/qwen2.5-0.5b-instruct-q8_0.gguf",
            "pytorch_qwen2_snapshots_q8_0");

        const auto cpu = singleDeviceTopology(
            "CPU0", GlobalDeviceAddress::cpu());
        const auto cuda = singleDeviceTopology(
            "CUDA0", GlobalDeviceAddress::cuda(0));
        const auto rocm = singleDeviceTopology(
            "ROCm0", GlobalDeviceAddress::rocm(0));

        const auto cpu_q4_default = qwen2Thresholds(0.005f, 90.0f, 95.0f);
        // This exact CPU/Q4_0/Q16_1 cell admits four of the five HF leaders:
        // independently audited scratch/cache approximation exchanges the fifth
        // and sixth logits. Keep every other distribution/stage gate unchanged,
        // and do not extend this allowance to another backend or KV format.
        const auto cpu_q4_q16 = qwen2Thresholds(0.006f, 90.0f, 80.0f);
        const auto cuda_q4_default = qwen2Thresholds(0.009f, 80.0f, 80.0f);
        const auto cuda_q4_q8 = qwen2Thresholds(0.008f, 80.0f, 95.0f, 5);
        const auto cuda_q4_tq = qwen2Thresholds(0.008f, 80.0f, 80.0f, 5);
        const auto rocm_q4_default = qwen2Thresholds(0.005f, 80.0f, 95.0f);
        const auto rocm_q4_fp16 = qwen2Thresholds(0.008f, 80.0f, 95.0f);
        const auto rocm_q4_quantized =
            qwen2Thresholds(0.005f, 80.0f, 95.0f, 5);
        const auto q8_default = qwen2Thresholds(0.01f, 80.0f, 95.0f);
        const auto q8_quantized = qwen2Thresholds(0.01f, 80.0f, 95.0f, 5);

        std::vector<ModelParityDefinition> definitions;
        definitions.push_back(singleDeviceDefinition(
            q4_model,
            cpu,
            cpu_q4_default,
            {
                KVCachePrecision::FP16,
                KVCachePrecision::Q8_1,
                KVCachePrecision::Q16_1,
                KVCachePrecision::TQ,
            },
            {
                {
                    .activation = ActivationPrecision::FP32,
                    .kv_cache = KVCachePrecision::Q16_1,
                    .thresholds = cpu_q4_q16,
                },
            }));
        definitions.push_back(singleDeviceDefinition(
            q4_model,
            cuda,
            cuda_q4_default,
            {
                KVCachePrecision::FP16,
                KVCachePrecision::Q8_1,
                KVCachePrecision::TQ,
            },
            {
                {
                    .activation = ActivationPrecision::FP32,
                    .kv_cache = KVCachePrecision::Q8_1,
                    .thresholds = cuda_q4_q8,
                },
                {
                    .activation = ActivationPrecision::FP32,
                    .kv_cache = KVCachePrecision::TQ,
                    .thresholds = cuda_q4_tq,
                },
            }));
        definitions.push_back(singleDeviceDefinition(
            q4_model,
            rocm,
            rocm_q4_default,
            {
                KVCachePrecision::FP16,
                KVCachePrecision::FP32,
                KVCachePrecision::Q8_1,
                KVCachePrecision::TQ,
            },
            {
                {
                    .activation = ActivationPrecision::FP32,
                    .kv_cache = KVCachePrecision::FP16,
                    .thresholds = rocm_q4_fp16,
                },
                {
                    .activation = ActivationPrecision::FP32,
                    .kv_cache = KVCachePrecision::Q8_1,
                    .thresholds = rocm_q4_quantized,
                },
                {
                    .activation = ActivationPrecision::FP32,
                    .kv_cache = KVCachePrecision::TQ,
                    .thresholds = rocm_q4_quantized,
                },
            }));

        const auto q8_definition =
            [&](const ModelParityTopologyDefinition &topology)
        {
            return singleDeviceDefinition(
                q8_model,
                topology,
                q8_default,
                {KVCachePrecision::FP16, KVCachePrecision::Q8_1},
                {
                    {
                        .activation = ActivationPrecision::FP32,
                        .kv_cache = KVCachePrecision::Q8_1,
                        .thresholds = q8_quantized,
                    },
                });
        };
        definitions.push_back(q8_definition(cpu));
        definitions.push_back(q8_definition(cuda));
        definitions.push_back(q8_definition(rocm));

        auto chat_model = qwen2ParityModel(
            "Qwen2_Q8_0_ChatCalc",
            "models/qwen2.5-0.5b-instruct-q8_0.gguf",
            "pytorch_qwen2_snapshots_q8_0_chat_calc");
        chat_model.prompt = R"(<|im_start|>system
You are a calculator. Reply with only the numeric answer, no explanation.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
)";
        chat_model.token_ids = {
            151644, 8948, 198, 2610, 525, 264, 29952, 13, 17841, 448,
            1172, 279, 24064, 4226, 11, 902, 16148, 13, 151645, 198,
            151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645,
            198, 151644, 77091, 198,
        };
        chat_model.decode_steps = 2;
        definitions.push_back(singleDeviceDefinition(
            std::move(chat_model),
            cuda,
            qwen2Thresholds(0.01f, 80.0f, 80.0f),
            {KVCachePrecision::FP16}));
        return definitions;
    }

    /** @return One ordered generated case vector for GoogleTest discovery. */
    const std::vector<ModelParityCase> &qwen2SingleDeviceCases()
    {
        static const auto cases = []
        {
            std::vector<ModelParityCase> expanded;
            for (const auto &definition : qwen2SingleDeviceDefinitions())
            {
                auto definition_cases =
                    expandModelParityDefinition(definition);
                expanded.insert(
                    expanded.end(),
                    std::make_move_iterator(definition_cases.begin()),
                    std::make_move_iterator(definition_cases.end()));
            }
            return expanded;
        }();
        return cases;
    }
} // namespace

class Qwen2SingleDeviceParityTest
    : public ConfigDrivenParityTest<Qwen2SingleDeviceParityTest>,
      public ModelParityCaseParameter
{
};

TEST_P(Qwen2SingleDeviceParityTest, ProductionParity)
{
    runProductionParityCampaign();
}

INSTANTIATE_TEST_SUITE_P(
    Qwen2,
    Qwen2SingleDeviceParityTest,
    ::testing::ValuesIn(qwen2SingleDeviceCases()),
    [](const ::testing::TestParamInfo<ModelParityCase> &info)
    { return info.param.testName(); });

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
