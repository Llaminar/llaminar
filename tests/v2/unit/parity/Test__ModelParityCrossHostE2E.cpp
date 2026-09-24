/**
 * @file Test__ModelParityCrossHostE2E.cpp
 * @brief Device-free remote certification expansion and frontend-policy proofs.
 *
 * No VM, model payload, MPI rank or accelerator is started here. These tests
 * prove that canonical definitions generate distinct remote E2E identities,
 * preserve the mathematical matrix, and cannot accidentally emit fixed local
 * placement while claiming to exercise default automatic orchestration.
 */
#include "integration/parity/qwen36/Qwen36ModelParityDefinitions.h"
#include "integration/parity/qwen36/Ornith15ModelParityDefinitions.h"
#include "config/OrchestrationConfigParser.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <limits>
#include <sstream>

namespace llaminar2::test::parity
{
    namespace
    {
        /** @return Model-owned canonical definition, without reading its GGUF. */
        ModelParityDefinition remoteSource(GlobalDeviceAddress device = GlobalDeviceAddress::rocm(0))
        {
            return qwen36::qwen36MoEParityDefinition(
                qwen36::qwen36SingleDeviceTopology("SourceGPU", std::move(device)),
                "/reference", qwen36::qwen36MoESingleDeviceThresholds());
        }

        /** @return The real GoogleTest JSON boundary consumed by inventory discovery. */
        nlohmann::json record(const ModelParityCase &cell)
        {
            std::ostringstream out;
            PrintTo(cell, &out);
            return nlohmann::json::parse(out.str());
        }

        /** @return Shared frontend parser result; argv storage outlives parsing. */
        OrchestrationConfig parsePolicy(const nlohmann::json &arguments)
        {
            auto args = arguments.get<std::vector<std::string>>();
            args.insert(args.begin(), "llaminar2");
            std::vector<char *> argv;
            for (auto &arg : args) argv.push_back(arg.data());
            return OrchestrationConfigParser{}.parseArgs(static_cast<int>(argv.size()), argv.data());
        }
    }

    TEST(ModelParityCrossHostE2E, PositiveHostCountRetainsTheContinuationRank)
    {
        for (const int invalid : {-1, 0, std::numeric_limits<int>::max()})
            EXPECT_THROW((void)ModelParityRemoteCPUHosts{invalid}, std::invalid_argument);
        for (const int count : {1, 2, 3, 8, std::numeric_limits<int>::max() - 1})
        {
            const ModelParityRemoteCPUHosts hosts(count);
            EXPECT_EQ(hosts.count(), count);
            EXPECT_EQ(hosts.executionRanks(), count + 1);
        }
        EXPECT_THROW((void)modelParityCrossHostFrontendName(
            static_cast<ModelParityCrossHostFrontend>(255)), std::invalid_argument);
    }

    TEST(ModelParityCrossHostE2E, RequestedModelExpandsFourRoutesWithoutChangingParityCells)
    {
        auto definition = remoteSource();
        const auto tagged = expandModelParityDefinition(definition);
        definition.e2e_certifiable.front().remote_cpu_overlays.clear();
        const auto ordinary = expandModelParityDefinition(definition);
        ASSERT_EQ(tagged.size(), ordinary.size());
        std::set<std::string> identities;
        std::set<std::pair<int, std::string>> routes;
        for (std::size_t i = 0; i < tagged.size(); ++i)
        {
            EXPECT_EQ(tagged[i].testName(), ordinary[i].testName());
            auto actual = record(tagged[i]);
            for (const auto &remote : actual.at("cross_host_e2e"))
            {
                ASSERT_EQ(tagged[i].mtp, ModelParityMTP::DynamicDepth);
                ASSERT_TRUE(tagged[i].e2e_certification);
                EXPECT_TRUE(identities.insert(remote.at("id").get<std::string>()).second);
                routes.emplace(remote.at("topology").at("remote_cpu_hosts").get<int>(),
                               remote.at("frontend").get<std::string>());
                EXPECT_EQ(remote.at("topology").at("continuation_backend"), "rocm");
                EXPECT_EQ(remote.at("topology").at("execution_ranks"),
                          remote.at("topology").at("remote_cpu_hosts").get<int>() + 1);
                EXPECT_EQ(remote.at("movement_evidence"), "required");
                EXPECT_EQ(remote.at("owner_order"), "ordinal");
                // The model path belongs to the parent record, so a remount
                // cannot leave a second stale path inside remote metadata.
                EXPECT_FALSE(remote.contains("model"));
            }
            actual["cross_host_e2e"] = nlohmann::json::array();
            EXPECT_EQ(actual, record(ordinary[i]));
        }
        EXPECT_EQ(identities.size(), 4u);
        const std::set<std::pair<int, std::string>> expected{
            {1, "plan-apply"}, {1, "auto-serve"}, {2, "plan-apply"}, {2, "auto-serve"}};
        EXPECT_EQ(routes, expected);
    }

    TEST(ModelParityCrossHostE2E, ProjectionUsesDefaultAutoAndPreservesPoliciesOnBothGpuVendors)
    {
        for (const auto address : {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0)})
        {
            auto definition = remoteSource(address);
            definition.e2e_certifiable.front().remote_cpu_overlays = {ModelParityRemoteCPUHosts{3}};
            for (const auto &cell : expandModelParityDefinition(definition))
            {
                const auto exported = record(cell);
                for (const auto &remote : exported.at("cross_host_e2e"))
                {
                    const auto config = parsePolicy(remote.at("server_policy_args"));
                    EXPECT_EQ(config.planning_mode, OrchestrationPlanningMode::InferFromPlacement);
                    EXPECT_FALSE(config.device_for_this_rank);
                    EXPECT_TRUE(config.device_map.empty());
                    EXPECT_TRUE(config.domain_definitions.empty());
                    EXPECT_FALSE(config.moe_routed_expert_plan);
                    const auto intent = resolveOrchestrationIntent(config);
                    ASSERT_TRUE(std::holds_alternative<AutomaticOrchestrationRequest>(intent));
                    const auto &automatic = std::get<AutomaticOrchestrationRequest>(intent);
                    EXPECT_TRUE(automatic.allows(DeviceType::CPU));
                    EXPECT_TRUE(automatic.allows(address.isCUDA() ? DeviceType::CUDA : DeviceType::ROCm));
                    EXPECT_FALSE(automatic.allows(address.isCUDA() ? DeviceType::ROCm : DeviceType::CUDA));
                    EXPECT_TRUE(automatic.allows(OrchestrationStrategy::ExpertOverlay));
                    EXPECT_FALSE(automatic.allows(OrchestrationStrategy::SingleDevice));
                    EXPECT_EQ(automatic.options().host_participation, AutomaticHostParticipation::AllDiscovered);
                    EXPECT_EQ(config.activation_precision, "fp32");
                    EXPECT_EQ(config.kv_cache_precision, "fp16");
                    EXPECT_TRUE(config.mtp.enabled);
                    EXPECT_EQ(config.mtp.verify_mode, MTPVerifyMode::SpeculativeSampling);
                    EXPECT_EQ(config.moe_rebalance.mode, MoERebalanceRuntimeMode::Dynamic);
                    EXPECT_EQ(config.moe_rebalance.migration_transfer_slots,
                              cell.dynamic_rebalance.migration_transfer_slots);
                }
            }
        }
    }

    TEST(ModelParityCrossHostE2E, EligibilityRejectsDuplicateAndNonGpuSourceTopologies)
    {
        auto definition = remoteSource();
        definition.e2e_certifiable.front().remote_cpu_overlays = {
            ModelParityRemoteCPUHosts{1}, ModelParityRemoteCPUHosts{1}};
        EXPECT_THROW((void)expandModelParityDefinition(definition), std::invalid_argument);
        auto cpu = qwen36::qwen36MoECPU2NodeTPParityDefinition();
        cpu.e2e_certifiable.front().remote_cpu_overlays = {ModelParityRemoteCPUHosts{1}};
        EXPECT_THROW((void)expandModelParityDefinition(cpu), std::invalid_argument);
        auto local_cpu = remoteSource(GlobalDeviceAddress::cpu());
        local_cpu.e2e_certifiable = {{.mtp = ModelParityMTP::DynamicDepth,
            .remote_cpu_overlays = {ModelParityRemoteCPUHosts{1}}}};
        EXPECT_THROW((void)expandModelParityDefinition(local_cpu), std::invalid_argument);
    }

    TEST(ModelParityCrossHostE2E, FineTuneDoesNotInheritRemoteCertificationEligibility)
    {
        const std::array definitions{remoteSource()};
        const auto variants = qwen36::withOrnith15CertificationModels(definitions);
        ASSERT_EQ(variants.size(), 2u);
        EXPECT_EQ(variants.front().e2e_certifiable.front().remote_cpu_overlays.size(), 2u);
        EXPECT_TRUE(variants.back().e2e_certifiable.front().remote_cpu_overlays.empty());
        for (const auto &cell : expandModelParityDefinition(variants.back()))
            EXPECT_TRUE(record(cell).at("cross_host_e2e").empty());
    }

    TEST(ModelParityCrossHostE2E, UntaggedAndInvalidProjectionCannotProduceAutomaticArguments)
    {
        const auto cell = expandModelParityDefinition(remoteSource()).front();
        EXPECT_THROW((void)modelParityServerArguments(cell,
            ModelParityRuntimePlacementProjection::AutomaticRemoteCPUOverlay), std::invalid_argument);
        EXPECT_THROW((void)modelParityServerArguments(cell,
            static_cast<ModelParityRuntimePlacementProjection>(255)), std::invalid_argument);
    }
}
