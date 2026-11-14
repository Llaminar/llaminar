/**
 * @file Test__ModelParityOverlayCertification.cpp
 * @brief Device-free diagnostic eligibility and placement for GPU overlay controls.
 *
 * Eligibility must originate in the typed model/topology definition. These
 * regressions preserve all mathematical cells and matched whole-expert/projection
 * workloads while proving that non-release models remain diagnostic-only. Both
 * public runtime and diagnostic benchmark exports must retain declared placement,
 * arithmetic and movement intent. No model or accelerator is loaded.
 */

#include "integration/parity/qwen36/Ornith15ModelParityDefinitions.h"
#include "config/OrchestrationConfigParser.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
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

        /**
         * @brief Prove diagnostic-only discovery and lossless overlay placement.
         * @param cell Real non-release cell, without a fabricated HTTP profile.
         * @param compute Physical expert policy shared by every declared routed domain.
         */
        void requireDiagnosticOverlay(const ModelParityCase &cell, RoutedExpertComputePolicy compute)
        {
            ASSERT_FALSE(cell.e2e_certification);
            ASSERT_TRUE(cell.expert_overlay);
            const auto record = discoveryRecord(cell);
            EXPECT_TRUE(record.at("e2e").is_null());
            EXPECT_TRUE(record.at("benchmark").is_null());
            EXPECT_TRUE(record.at("cross_host_e2e").empty());
            const auto diagnostic_args = modelParityDiagnosticBenchmarkArguments(cell);
            const auto &diagnostic = record.at("diagnostic_benchmark");
            EXPECT_EQ(diagnostic.at("policy"), "diagnostic_declared");
            EXPECT_EQ(diagnostic.at("certification_eligible"), false);
            EXPECT_EQ(diagnostic.at("context_length"), cell.model.max_seq_len);
            EXPECT_EQ(diagnostic.at("args"), diagnostic_args);
            EXPECT_THROW(modelParityBenchmarkArguments(cell), std::invalid_argument);

            auto expected = cell.makeOrchestrationConfig(cell.model.model_path, 0);
            ASSERT_TRUE(normalizeMoERoutedExpertPlacementDomains(expected).empty());
            ASSERT_TRUE(expected.moe_routed_expert_plan);
            const auto &plan = *expected.moe_routed_expert_plan;
            ASSERT_FALSE(plan.domains.empty());
            for (const auto &domain : plan.domains)
                EXPECT_EQ(domain.toExecutionDomainDefinition().routed_compute_policy, compute);
            for (const auto &arguments : {
                     record.at("runtime").at("server_args").get<std::vector<std::string>>(),
                     diagnostic_args})
            {
                const auto runtime = parsePublicArguments(arguments);
                EXPECT_FALSE(std::holds_alternative<AutomaticOrchestrationRequest>(
                    resolveOrchestrationIntent(runtime)));
                EXPECT_EQ(runtime.activation_precision, expected.activation_precision);
                EXPECT_EQ(runtime.kv_cache_precision, expected.kv_cache_precision);
                EXPECT_EQ(runtime.default_backend, expected.default_backend);
                EXPECT_EQ(runtime.tp_devices, expected.tp_devices);
                EXPECT_EQ(runtime.device_map, expected.device_map);
                EXPECT_EQ(runtime.device_for_this_rank, expected.device_for_this_rank);
                EXPECT_EQ(runtime.mtp.enabled, cell.mtpEnabled());
                EXPECT_EQ(runtime.mtp.depth_policy.mode, cell.usesDynamicMTPDepth()
                    ? MTPDepthPolicyMode::Dynamic : MTPDepthPolicyMode::Fixed);
                if (cell.mtpEnabled() && !cell.usesDynamicMTPDepth())
                    EXPECT_EQ(runtime.mtp.draft_tokens, cell.requestedMTPDraftDepth());
                EXPECT_EQ(runtime.moe_rebalance.mode,
                          cell.expert_overlay->movement == ModelParityExpertMovement::Dynamic
                              ? MoERebalanceRuntimeMode::Dynamic : MoERebalanceRuntimeMode::Off);
                ASSERT_TRUE(runtime.moe_routed_expert_plan);
                const auto &actual = *runtime.moe_routed_expert_plan;
                EXPECT_EQ(actual.owner_order, plan.owner_order);
                EXPECT_EQ(actual.topology, plan.topology);
                EXPECT_EQ(actual.residency_policy, plan.residency_policy);
                EXPECT_EQ(actual.continuation_domain, plan.continuation_domain);
                EXPECT_EQ(actual.base_model_domain, plan.base_model_domain);
                EXPECT_EQ(actual.shared_expert_domain, plan.shared_expert_domain);
                ASSERT_EQ(actual.domains.size(), plan.domains.size());
                for (std::size_t i = 0; i < actual.domains.size(); ++i)
                    EXPECT_EQ(modelParityDomainArgument(actual.domains[i].toExecutionDomainDefinition()),
                              modelParityDomainArgument(plan.domains[i].toExecutionDomainDefinition()));
                ASSERT_EQ(actual.dense_domains.size(), plan.dense_domains.size());
                for (std::size_t i = 0; i < actual.dense_domains.size(); ++i)
                    EXPECT_EQ(modelParityDomainArgument(actual.dense_domains[i]),
                              modelParityDomainArgument(plan.dense_domains[i]));
                ASSERT_EQ(actual.routed_tiers.size(), plan.routed_tiers.size());
                for (std::size_t i = 0; i < actual.routed_tiers.size(); ++i)
                {
                    EXPECT_EQ(actual.routed_tiers[i].name, plan.routed_tiers[i].name);
                    EXPECT_EQ(actual.routed_tiers[i].domain, plan.routed_tiers[i].domain);
                    EXPECT_EQ(actual.routed_tiers[i].priority, plan.routed_tiers[i].priority);
                    EXPECT_EQ(actual.routed_tiers[i].max_experts_per_layer,
                              plan.routed_tiers[i].max_experts_per_layer);
                    EXPECT_EQ(actual.routed_tiers[i].memory_budget_bytes,
                              plan.routed_tiers[i].memory_budget_bytes);
                }
            }
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
            std::size_t checked = 0;
            for (std::size_t index = 0; index < control.size(); ++index)
            {
                const auto &before = control[index];
                const auto &after = projection[index];
                SCOPED_TRACE(before.testName());
                EXPECT_EQ(before.model.model_path, after.model.model_path);
                EXPECT_EQ(before.model.reference_directory, after.model.reference_directory);
                EXPECT_EQ(before.model.prompt, after.model.prompt);
                EXPECT_EQ(before.model.token_ids, after.model.token_ids);
                EXPECT_EQ(before.model.decode_steps, after.model.decode_steps);
                EXPECT_EQ(before.mtp, after.mtp);
                EXPECT_EQ(before.activation_precision, after.activation_precision);
                EXPECT_EQ(before.kv_cache_precision, after.kv_cache_precision);
                EXPECT_EQ(before.prefix_restore_geometry, after.prefix_restore_geometry);
                EXPECT_EQ(after.movementEvidence(), before.movementEvidence());
                requireDiagnosticOverlay(before, RoutedExpertComputePolicy::Apportioned);
                requireDiagnosticOverlay(after, RoutedExpertComputePolicy::GateUpOwnedDownColumns);
                ++checked;
            }
            EXPECT_EQ(checked, 24u);
        }
        EXPECT_THROW(qwen36::qwen36MoEExpertOverlayCertificationDefinition(
            qwen36::qwen36SingleDeviceTopology("CUDA0", GlobalDeviceAddress::cuda(0)),
            "/reference"), std::invalid_argument);
    }

    /** Preserve the Q8/ROCm4 numerical identity without extending release eligibility. */
    TEST(ModelParityOverlayCertification, OrnithQ8Rocm4RetainsMathAndDiagnosticEligibility)
    {
        const auto diagnostic = qwen36::ornith15MoEQ8AccuracyParityDefinition(
            qwen36::ornith15MoERocm4ExpertOverlayTopology(),
            "pytorch_ornith15_q8_natural_decode_rocm4_snapshots");
        const auto certificate = qwen36::ornith15MoEQ8Rocm4CertificationDefinition();
        const auto before = expandModelParityDefinition(diagnostic);
        const auto after = expandModelParityDefinition(certificate);
        ASSERT_EQ(before.size(), 24u);
        ASSERT_EQ(after.size(), before.size());
        std::size_t checked = 0;
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
            EXPECT_EQ(cell.mtp, before[index].mtp);
            EXPECT_EQ(cell.activation_precision, before[index].activation_precision);
            EXPECT_EQ(cell.kv_cache_precision, before[index].kv_cache_precision);
            EXPECT_EQ(cell.movementEvidence(), before[index].movementEvidence());
            EXPECT_EQ(cell.topology.collective, Collective::RCCL);
            EXPECT_EQ(cell.topology.participants.size(), 4u);
            requireDiagnosticOverlay(cell, RoutedExpertComputePolicy::Apportioned);
            ++checked;
        }
        EXPECT_EQ(checked, 24u);
    }

    /** Fine-tune projection diagnostics retain matched math and declared placement. */
    TEST(ModelParityOverlayCertification, OrnithProjectionRetainsMatchedModelAndDiagnosticContracts)
    {
        std::vector<ModelParityDefinition> controls;
        std::vector<ModelParityDefinition> projections;
        for (const auto &topology : {qwen36::qwen36MoECuda2ExpertOverlayTopology(),
                                    qwen36::qwen36MoERocm2ExpertOverlayTopology()})
        {
            const std::array parent_controls = {
                qwen36::qwen36MoEExpertOverlayCertificationDefinition(topology, "/reference")};
            const auto models = qwen36::withOrnith15CertificationModels(parent_controls);
            const auto variants = qwen36::qwen36MoEAutomaticProjectionVariants(models);
            ASSERT_EQ(variants.size(), models.size());
            controls.push_back(models.back());
            projections.push_back(variants.back());
        }
        controls.push_back(qwen36::ornith15MoEQ8Rocm4CertificationDefinition());
        projections.push_back(qwen36::ornith15MoEQ8Rocm4ProjectionCertificationDefinition());
        for (std::size_t variant = 0; variant < controls.size(); ++variant)
        {
            const auto before = expandModelParityDefinition(controls[variant]);
            const auto after = expandModelParityDefinition(projections[variant]);
            ASSERT_EQ(before.size(), 24u);
            ASSERT_EQ(after.size(), before.size());
            std::size_t checked = 0;
            for (std::size_t index = 0; index < before.size(); ++index)
            {
                const auto &control = before[index];
                const auto &projection = after[index];
                SCOPED_TRACE(projection.testName());
                EXPECT_NE(projection.testName(), control.testName());
                EXPECT_EQ(projection.model.model_path, control.model.model_path);
                EXPECT_EQ(projection.model.reference_directory, control.model.reference_directory);
                EXPECT_EQ(projection.model.prompt, control.model.prompt);
                EXPECT_EQ(projection.model.token_ids, control.model.token_ids);
                EXPECT_EQ(projection.model.decode_steps, control.model.decode_steps);
                EXPECT_EQ(projection.mtp, control.mtp);
                EXPECT_EQ(projection.activation_precision, control.activation_precision);
                EXPECT_EQ(projection.kv_cache_precision, control.kv_cache_precision);
                EXPECT_EQ(projection.prefix_restore_geometry, control.prefix_restore_geometry);
                EXPECT_EQ(projection.movementEvidence(), control.movementEvidence());
                EXPECT_EQ(projection.e2e_certification, control.e2e_certification);
                requireDiagnosticOverlay(control, RoutedExpertComputePolicy::Apportioned);
                requireDiagnosticOverlay(projection, RoutedExpertComputePolicy::GateUpOwnedDownColumns);
                ++checked;
            }
            EXPECT_EQ(checked, 24u);
        }
    }
}
