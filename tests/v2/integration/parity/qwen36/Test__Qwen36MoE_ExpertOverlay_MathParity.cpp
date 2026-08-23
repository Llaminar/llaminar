/**
 * @file Test__Qwen36MoE_ExpertOverlay_MathParity.cpp
 * @brief Typed production parity matrix for homogeneous Qwen3.6 MoE overlays.
 *
 * Each declaration names only a real model/reference pack and one physical
 * topology. The shared matrix expands Ordinal/Random placement,
 * Static/Dynamic residency, MTP off/fixed 1/2/3/15/dynamic-depth, activation
 * and KV precision, and the mandatory fresh/full/partial prefix lifecycle.
 * Every cell enters through OrchestrationRunner and emits the common numerical
 * and production-path CSV artifacts.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>

#include "../qwen35moe/Qwen35MoEParityTestBase.h"
#include "Qwen36ModelParityDefinitions.h"
#include "Qwen36MoEParityTestBase.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"
#include "utils/Logger.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35moe;
using namespace llaminar2::test::parity::qwen36;

namespace
{
    /** @return Canonically expanded two-CUDA and two-ROCm overlay cases. */
    const std::vector<ModelParityCase> &qwen36MoEExpertOverlayCases()
    {
        static const auto cases = []
        {
            const std::array definitions = {
                qwen36MoEParityDefinition(
                    qwen36MoECuda2ExpertOverlayTopology(),
                    "pytorch_qwen36_moe_singledevice_cuda_snapshots",
                    qwen36MoEExpertOverlayThresholds()),
                qwen36MoEParityDefinition(
                    qwen36MoERocm2ExpertOverlayTopology(),
                    "pytorch_qwen36_moe_singledevice_rocm_snapshots",
                    qwen36MoEExpertOverlayThresholds()),
            };

            std::vector<ModelParityCase> expanded;
            for (const auto &definition : definitions)
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

/** Production Qwen3.6 MoE ExpertOverlay parity over one generated case. */
class Qwen36MoEExpertOverlayParityTest
    : public Qwen35MoEConfigDrivenParityTest<
          Qwen36MoEExpertOverlayParityTest>,
      public ModelParityCaseParameter
{
protected:
    using Base = Qwen35MoEConfigDrivenParityTest<
        Qwen36MoEExpertOverlayParityTest>;

    /**
     * @brief Require final routed/shared publication for every MoE layer.
     *
     * Branch-local tensors are intentionally excluded because they are
     * sharded. MOE_COMBINED_OUTPUT is the semantic post-collective boundary
     * and must remain visible on every captured layer.
     */
    ParityGraphSnapshotPolicy parityGraphSnapshotPolicy(
        ParityForwardPhase phase) const override
    {
        auto policy = Base::parityGraphSnapshotPolicy(phase);
        if (phase != ParityForwardPhase::Prefill)
            return policy;

        for (int layer = 0; layer < parityLayerCount(); ++layer)
        {
            const std::string key =
                "layer" + std::to_string(layer) +
                "_MOE_COMBINED_OUTPUT";
            if (std::find(
                    policy.required_prefill_snapshot_keys.begin(),
                    policy.required_prefill_snapshot_keys.end(),
                    key) == policy.required_prefill_snapshot_keys.end())
            {
                policy.required_prefill_snapshot_keys.push_back(key);
            }
        }
        return policy;
    }
};

TEST_P(Qwen36MoEExpertOverlayParityTest, ProductionParity)
{
    runProductionParityCampaign();
}

/** Lock down accepted-placement accounting used by movement evidence. */
TEST(Qwen36MoEExpertOverlayPerfStats,
     DynamicMovementAcceptsEitherProductionDecisionForm)
{
    const PerfStatRecord accepted_replica{
        .kind = PerfStatRecord::Kind::Counter,
        .domain = "moe_rebalance",
        .name = "device_rebalance_selected_replicas",
        .value = 2.0,
    };
    const PerfStatRecord accepted_ownership_swap{
        .kind = PerfStatRecord::Kind::Counter,
        .domain = "moe_rebalance",
        .name = "device_rebalance_dynamic_ownership_swap_accepts",
        .value = 1.0,
    };
    const PerfStatRecord rejected_attempt{
        .kind = PerfStatRecord::Kind::Counter,
        .domain = "moe_rebalance",
        .name = "device_rebalance_dynamic_ownership_swap_attempts",
        .value = 7.0,
    };

    EXPECT_EQ(
        dynamicRebalanceAcceptedPlacementCount({accepted_replica}), 2.0);
    EXPECT_EQ(
        dynamicRebalanceAcceptedPlacementCount({accepted_ownership_swap}),
        1.0);
    EXPECT_EQ(
        dynamicRebalanceAcceptedPlacementCount(
            {accepted_replica, accepted_ownership_swap}),
        3.0);
    EXPECT_EQ(
        dynamicRebalanceAcceptedPlacementCount({rejected_attempt}), 0.0);
}

/** Prove multiline reference prompts authenticate without regeneration. */
TEST(Qwen36MoEExpertOverlayMetadata,
     MultilinePromptAuthenticatesWithoutRegeneration)
{
    const std::filesystem::path metadata_path =
        std::filesystem::temp_directory_path() /
        ("llaminar_qwen36_multiline_metadata_" +
         std::to_string(static_cast<long long>(::getpid())) + ".txt");
    const std::string prompt =
        "Read the ledger exactly.\n"
        "Ledger item 0001: alpha.\n"
        "Return one JSON object.";

    {
        std::ofstream metadata(metadata_path, std::ios::trunc);
        ASSERT_TRUE(metadata.is_open()) << "failed to create " << metadata_path;
        metadata << "snapshot_version: 4\n"
                 << "prompt: Read the ledger exactly.\n"
                 << "Ledger item 0001: alpha.\n"
                 << "Return one JSON object.\n"
                 << "token_ids: 1,2,3\n"
                 << "decode_steps: 2\n"
                 << "decode_tokens: 4,5\n";
    }

    EXPECT_EQ(
        readMultilineStringFromMetadata(
            metadata_path, "prompt", "token_ids"),
        std::optional<std::string>{prompt});
    EXPECT_TRUE(metadataLooksUsable(metadata_path, prompt, 2));
    EXPECT_FALSE(metadataLooksUsable(metadata_path, prompt + "\nchanged", 2));

    std::error_code remove_error;
    std::filesystem::remove(metadata_path, remove_error);
    EXPECT_FALSE(remove_error) << "failed to remove " << metadata_path
                               << ": " << remove_error.message();
}

INSTANTIATE_TEST_SUITE_P(
    Qwen36MoEExpertOverlay,
    Qwen36MoEExpertOverlayParityTest,
    ::testing::ValuesIn(qwen36MoEExpertOverlayCases()),
    [](const ::testing::TestParamInfo<ModelParityCase> &info)
    {
        return info.param.testName();
    });

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    initializeLogging();
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();

    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
