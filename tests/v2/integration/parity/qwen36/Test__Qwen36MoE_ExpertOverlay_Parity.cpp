/**
 * @file Test__Qwen36MoE_ExpertOverlay_Parity.cpp
 * @brief Typed production parity for Qwen3.6 MoE and Ornith fine-tune overlays.
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
#include "Ornith15ModelParityDefinitions.h"
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
    /** @return Canonical overlay cases, including the Ornith four-ROCm reproducer. */
    const std::vector<ModelParityCase> &qwen36MoEExpertOverlayCases()
    {
        static const auto cases = []
        {
            const std::array definitions = {
                qwen36MoECPU2NodeTPParityDefinition(),
                qwen36MoEParityDefinition(
                    qwen36MoECuda2ExpertOverlayTopology(),
                    "pytorch_qwen36_moe_singledevice_cuda_snapshots",
                    qwen36MoEExpertOverlayThresholds()),
                qwen36MoEParityDefinition(
                    qwen36MoERocm2ExpertOverlayTopology(),
                    "pytorch_qwen36_moe_singledevice_rocm_snapshots",
                    qwen36MoEExpertOverlayThresholds()),
            };

            auto all_definitions = withOrnith15CertificationModels(definitions);
            // Keep the observed HTTP wrong-answer case in the same typed
            // expander as every other real-weight diagnostic.  Its standard
            // Static/Ordinal MTP-off cell gives checkpoint CSV evidence for
            // the native graph forward before changing any numerical gate.
            all_definitions.push_back(
                ornith15MoEQ4ChatNoThinkingForwardParityDefinition(
                    qwen36MoECuda2ExpertOverlayTopology(),
                    "pytorch_ornith15_q4_chat_nonthinking_cuda2_snapshots"));
            all_definitions.push_back(ornith15MoEQ8AccuracyParityDefinition(
                ornith15MoERocm4ExpertOverlayTopology(),
                "pytorch_ornith15_q8_natural_decode_rocm4_snapshots"));
            std::vector<ModelParityCase> expanded;
            for (const auto &definition : all_definitions)
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
     * sharded. The root-owned routed fold, gated shared contribution, and
     * final combine are all complete semantic values. Requiring each boundary
     * lets a real graph-captured failure distinguish sparse exchange/folding
     * from the subsequent shared-expert merge without an eager replay.
     */
    ParityGraphSnapshotPolicy parityGraphSnapshotPolicy(
        ParityForwardPhase phase) const override
    {
        auto policy = Base::parityGraphSnapshotPolicy(phase);

        /*
         * The routed-output checkpoint alone cannot prove that the captured
         * mapped-sparse prefill fold consumed the request-pinned ownership
         * schedule produced by grouped routing. Retain that typed route
         * evidence before graph construction, and make a missing prefill
         * ledger a precise failure instead of silently weakening the
         * ExpertOverlay proof.
         *
         * Ordinary LocalTP decode has a different rooted route-slot lowering:
         * it consumes the immutable static ownership contract directly and
         * does not materialize the grouped-prefill assignment scratch. Asking
         * that graph to publish the prefill-only scratch would test a stale
         * buffer, not live production state. Decode remains rigorously checked
         * at its actual expert/output checkpoints; this additional route
         * evidence is scoped to the one graph that consumes it.
         */
        const int declared_main_layer_count =
            this->modelParityCase().model.transformer_layers;
        const auto route_snapshot_inventory =
            modelParityExpertOverlayRouteSnapshotInventory(
                declared_main_layer_count,
                this->modelParityCase().mtp);
        if (phase == ParityForwardPhase::Prefill)
        {
            auto &required_route_keys =
                policy.required_prefill_snapshot_keys;
            for (const std::string &key : route_snapshot_inventory.main_model)
            {
                if (std::find(
                        required_route_keys.begin(),
                        required_route_keys.end(),
                        key) == required_route_keys.end())
                {
                    required_route_keys.push_back(key);
                }
            }
        }

        if (phase != ParityForwardPhase::Prefill)
            return policy;

        constexpr std::array<std::string_view, 3> kRequiredMoEOutputs = {
            "MOE_EXPERT_OUTPUT",
            "MOE_SHARED_GATE_OUTPUT",
            "MOE_COMBINED_OUTPUT",
        };
        /*
         * This policy becomes graph identity before ModelContext loading, so
         * runtime metadata is unavailable at this point. The typed cell's
         * declared main-layer count is immutable and is cross-checked against
         * the GGUF before the production graph is admitted.
        */
        for (int layer = 0; layer < declared_main_layer_count; ++layer)
        {
            for (const std::string_view suffix : kRequiredMoEOutputs)
            {
                const std::string key =
                    "layer" + std::to_string(layer) + "_" +
                    std::string(suffix);
                if (std::find(
                        policy.required_prefill_snapshot_keys.begin(),
                        policy.required_prefill_snapshot_keys.end(),
                        key) == policy.required_prefill_snapshot_keys.end())
                {
                    policy.required_prefill_snapshot_keys.push_back(key);
                }
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
