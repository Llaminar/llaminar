/**
 * @file Test__ModelParityOverlayCertification.cpp
 * @brief Device-free certification coverage for GPU overlay controls and Ornith Q8.
 *
 * Eligibility must originate in the typed model/topology definition. These
 * regressions prove that adding HTTP tags preserves every mathematical cell,
 * keeps whole-expert/projection A/B workloads matched, and exports production
 * auto intent for both HTTP and benchmark callers. No model or GPU is loaded;
 * real HTTP lifetimes and measured benchmarks remain separate obligations.
 */

#include "integration/parity/qwen36/Ornith15ModelParityDefinitions.h"
#include "config/OrchestrationConfigParser.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace llaminar2::test::parity
{
    namespace
    {
        /**
         * @brief Parse an exported public argument vector without starting inference.
         * @param arguments Owned strings whose storage outlives the CLI parser.
         * @return The same typed configuration consumed by the production frontend.
         */
        OrchestrationConfig parsePublicArguments(std::vector<std::string> arguments)
        {
            arguments.insert(arguments.begin(), "llaminar2");
            std::vector<char *> argv;
            for (auto &argument : arguments)
                argv.push_back(argument.data());
            return OrchestrationConfigParser{}.parseArgs(
                static_cast<int>(argv.size()), argv.data());
        }

        /**
         * @brief Serialize through the same discovery boundary used by CI.
         * @param cell Complete typed cell, including its optional HTTP profile.
         * @return The structured model/runtime/HTTP/benchmark discovery record.
         */
        nlohmann::json discoveryRecord(const ModelParityCase &cell)
        {
            std::ostringstream output;
            PrintTo(cell, &output);
            return nlohmann::json::parse(output.str());
        }
    }

    /** Both vendors need a same-model whole-expert control, not another model's score. */
    TEST(ModelParityOverlayCertification, WholeExpertAndProjectionHaveMatchedPublicControls)
    {
        for (const auto &topology : {qwen36::qwen36MoECuda2ExpertOverlayTopology(),
                                    qwen36::qwen36MoERocm2ExpertOverlayTopology()})
        {
            const auto control = expandModelParityDefinition(
                qwen36::qwen36MoEExpertOverlayCertificationDefinition(topology, "/reference"));
            const auto projection = expandModelParityDefinition(
                qwen36::qwen36MoEProjectionParityDefinition(topology, "/reference"));
            ASSERT_EQ(control.size(), 24u);
            ASSERT_EQ(projection.size(), control.size());
            std::size_t tagged = 0;
            for (std::size_t index = 0; index < control.size(); ++index)
            {
                const auto &before = control[index];
                const auto &after = projection[index];
                SCOPED_TRACE(before.testName());
                EXPECT_EQ(before.model.model_path, after.model.model_path);
                EXPECT_EQ(before.model.reference_directory, after.model.reference_directory);
                EXPECT_EQ(before.model.token_ids, after.model.token_ids);
                EXPECT_EQ(before.model.decode_steps, after.model.decode_steps);
                EXPECT_EQ(before.mtp, after.mtp);
                EXPECT_EQ(before.activation_precision, after.activation_precision);
                EXPECT_EQ(before.kv_cache_precision, after.kv_cache_precision);
                EXPECT_EQ(before.prefix_restore_geometry, after.prefix_restore_geometry);
                ASSERT_EQ(before.e2e_certification.has_value(), after.e2e_certification.has_value());
                if (!before.e2e_certification)
                    continue;
                ++tagged;
                EXPECT_EQ(*before.e2e_certification, *after.e2e_certification);
                EXPECT_EQ(before.mtp, ModelParityMTP::DynamicDepth);
                EXPECT_EQ(before.movementEvidence(), ModelParityMovementEvidence::MovementRequired);
                EXPECT_EQ(after.movementEvidence(), before.movementEvidence());

                for (const auto *cell : {&before, &after})
                {
                    const auto expected = cell == &before
                        ? RoutedExpertComputePolicy::Apportioned
                        : RoutedExpertComputePolicy::GateUpOwnedDownColumns;
                    const auto record = discoveryRecord(*cell);
                    for (const auto &arguments : {
                             record.at("e2e").at("server_args").get<std::vector<std::string>>(),
                             modelParityBenchmarkArguments(*cell)})
                    {
                        const auto runtime = parsePublicArguments(arguments);
                        EXPECT_TRUE(std::holds_alternative<AutomaticOrchestrationRequest>(
                            resolveOrchestrationIntent(runtime)));
                        EXPECT_EQ(runtime.routed_expert_compute_policy, expected);
                        EXPECT_TRUE(runtime.mtp.enabled);
                        EXPECT_EQ(runtime.mtp.depth_policy.mode, MTPDepthPolicyMode::Dynamic);
                        EXPECT_EQ(runtime.moe_rebalance.mode, MoERebalanceRuntimeMode::Dynamic);
                        EXPECT_TRUE(runtime.tp_devices.empty());
                        EXPECT_FALSE(runtime.moe_routed_expert_plan);
                    }
                }
            }
            EXPECT_EQ(tagged, 1u);
        }
        EXPECT_THROW(qwen36::qwen36MoEExpertOverlayCertificationDefinition(
            qwen36::qwen36SingleDeviceTopology("CUDA0", GlobalDeviceAddress::cuda(0)),
            "/reference"), std::invalid_argument);
    }

    /** Tag the reported Q8/ROCm4 identity without replacing its independent HF diagnosis. */
    TEST(ModelParityOverlayCertification, OrnithQ8Rocm4RetainsMathAndAddsOneHTTPBenchmarkTag)
    {
        const auto diagnostic = qwen36::ornith15MoEQ8AccuracyParityDefinition(
            qwen36::ornith15MoERocm4ExpertOverlayTopology(),
            "pytorch_ornith15_q8_natural_decode_rocm4_snapshots");
        const auto certificate = qwen36::ornith15MoEQ8Rocm4CertificationDefinition();
        const auto before = expandModelParityDefinition(diagnostic);
        const auto after = expandModelParityDefinition(certificate);
        ASSERT_EQ(before.size(), 24u);
        ASSERT_EQ(after.size(), before.size());
        std::size_t tagged = 0;
        for (std::size_t index = 0; index < before.size(); ++index)
        {
            const auto &cell = after[index];
            SCOPED_TRACE(cell.testName());
            EXPECT_EQ(cell.testName(), before[index].testName());
            EXPECT_EQ(cell.model.model_path, before[index].model.model_path);
            EXPECT_EQ(cell.model.reference_directory, before[index].model.reference_directory);
            EXPECT_EQ(cell.model.prompt, before[index].model.prompt);
            EXPECT_EQ(cell.model.token_ids, before[index].model.token_ids);
            EXPECT_EQ(cell.model.token_ids.size(), 424u);
            EXPECT_EQ(cell.model.decode_steps, 89);
            EXPECT_EQ(cell.prefix_restore_geometry, before[index].prefix_restore_geometry);
            EXPECT_FALSE(before[index].e2e_certification);
            if (!cell.e2e_certification)
                continue;
            ++tagged;
            ASSERT_TRUE(cell.expert_overlay);
            EXPECT_EQ(cell.expert_overlay->movement, ModelParityExpertMovement::Dynamic);
            EXPECT_EQ(cell.expert_overlay->owner_order, RoutedExpertOwnerOrder::Ordinal);
            EXPECT_EQ(cell.mtp, ModelParityMTP::DynamicDepth);
            EXPECT_EQ(cell.e2e_certification->context_length, 8192);
            EXPECT_EQ(cell.e2e_certification->generation_tokens, 2048);
            EXPECT_EQ(cell.e2e_certification->minimum_prompt_tokens, 4096);
            EXPECT_EQ(cell.movementEvidence(), ModelParityMovementEvidence::MovementRequired);
            EXPECT_EQ(cell.topology.collective, Collective::RCCL);
            EXPECT_EQ(cell.topology.participants.size(), 4u);
            const auto record = discoveryRecord(cell);
            EXPECT_EQ(record.at("benchmark").at("policy"), "production_defaults");
            for (const auto &arguments : {
                     record.at("e2e").at("server_args").get<std::vector<std::string>>(),
                     modelParityBenchmarkArguments(cell)})
            {
                EXPECT_NE(std::find(arguments.begin(), arguments.end(), "rocm=4"), arguments.end());
                const auto runtime = parsePublicArguments(arguments);
                EXPECT_TRUE(std::holds_alternative<AutomaticOrchestrationRequest>(
                    resolveOrchestrationIntent(runtime)));
                EXPECT_EQ(runtime.routed_expert_compute_policy, RoutedExpertComputePolicy::Apportioned);
                EXPECT_TRUE(runtime.mtp.enabled);
                EXPECT_EQ(runtime.mtp.depth_policy.mode, MTPDepthPolicyMode::Dynamic);
                EXPECT_EQ(runtime.moe_rebalance.mode, MoERebalanceRuntimeMode::Dynamic);
                EXPECT_TRUE(runtime.tp_devices.empty());
                EXPECT_FALSE(runtime.moe_routed_expert_plan);
            }
        }
        EXPECT_EQ(tagged, 1u);
    }
}
