/**
 * @file Test__Qwen35MoE_122B_NodeExpertOverlay_Parity.cpp
 * @brief 122B model registration for the generated node ExpertOverlay matrix.
 *
 * Shared by the 35B and 122B generated node ExpertOverlay matrices. This
 * translation-unit boundary does not change request ownership, reference
 * mathematics, graph execution, or evidence publication ordering.
 */
#include "node_overlay/NodeExpertOverlayParityFixture.h"
#include "config/OrchestrationConfigParser.h"

namespace llaminar2::test::parity::qwen35moe::node_overlay
{

/**
 * @brief Model-wide physical and active wave envelopes are typed separately.
 *
 * This prevents the regression where the fixture derived a 49-cycle target
 * but `applyRuntimePolicy()` later replaced it with an unrelated two-cycle
 * value.  A two-CPU topology has one independent participant axis; a single-
 * CPU topology has only the 48 layer-parallel tier cycles.
 */
TEST(Qwen122DynamicWaveGeometry,
     DerivesCyclesAndCommandEnvelopeFromModelAndTopology)
{
    const Qwen122OverlayTopologySpec two_cpu{
        .test_id = "typed_two_cpu",
        .cuda_participants = 0,
        .rocm_participants = 1,
        .cpu_participants = 2,
        .mpi_ranks = 2,
        .continuation = Qwen122ContinuationBackend::ROCm,
        .dynamic_speedup_witness =
            ModelParityDynamicSpeedupWitness::Ordinal,
    };
    const auto two_cpu_policy = qwen122DynamicParityEconomics(
        two_cpu,
        /*transformer_layers=*/48);
    EXPECT_EQ(two_cpu_policy.migration_transfer_slots, 49u);
    EXPECT_EQ(two_cpu_policy.resolvedMigrationCyclesPerWave(), 49u);
    EXPECT_EQ(two_cpu_policy.dynamic_max_swaps_per_layer, 2u);
    EXPECT_EQ(two_cpu_policy.dynamic_max_plan_entries_per_wave, 147u);

    auto one_cpu = two_cpu;
    one_cpu.test_id = "typed_one_cpu";
    one_cpu.cpu_participants = 1;
    one_cpu.mpi_ranks = 1;
    const auto one_cpu_policy = qwen122DynamicParityEconomics(
        one_cpu,
        /*transformer_layers=*/48);
    EXPECT_EQ(one_cpu_policy.migration_transfer_slots, 48u);
    EXPECT_EQ(one_cpu_policy.resolvedMigrationCyclesPerWave(), 48u);
    EXPECT_EQ(one_cpu_policy.dynamic_max_swaps_per_layer, 1u);
    EXPECT_EQ(one_cpu_policy.dynamic_max_plan_entries_per_wave, 96u);
}

/**
 * @brief Matrix expansion selects evidence geometry before fixture setup.
 *
 * This regression separates the speed witness's timing window from the five
 * MTP movement-only cells. Both use the same allocated transfer lanes rather
 * than imposing a second, smaller active-cycle cap. It also proves
 * that the fixture receives a complete immutable policy rather than deriving
 * controller settings from the GoogleTest name.
 */
TEST(Qwen122DynamicWaveGeometry,
     ExpansionSeparatesMovementProofFromObservedSpeedup)
{
    const Qwen122OverlayTopologySpec spec{
        .test_id = "typed_runtime_policy",
        .cuda_participants = 0,
        .rocm_participants = 1,
        .cpu_participants = 2,
        .mpi_ranks = 2,
        .continuation = Qwen122ContinuationBackend::ROCm,
        .dynamic_speedup_witness =
            ModelParityDynamicSpeedupWitness::Ordinal,
    };
    const auto cases = expandModelParityDefinition(
        qwen122ExpertOverlayParityDefinition(spec));

    const auto find_dynamic = [&](ModelParityMTP mtp)
        -> const ModelParityCase &
    {
        const auto found = std::find_if(
            cases.begin(), cases.end(),
            [&](const ModelParityCase &test_case)
            {
                return test_case.expert_overlay.has_value() &&
                       test_case.expert_overlay->owner_order ==
                           RoutedExpertOwnerOrder::Ordinal &&
                       test_case.expert_overlay->movement ==
                           ModelParityExpertMovement::Dynamic &&
                       test_case.mtp == mtp;
            });
        if (found == cases.end())
            throw std::logic_error("Generated Qwen122 Dynamic cell is missing");
        return *found;
    };

    const auto &speedup = find_dynamic(ModelParityMTP::Off);
    EXPECT_TRUE(speedup.requiresObservedConvergenceSpeedup());
    EXPECT_EQ(kQwen35MoEConvergenceTimingWarmupRequests, 0);
    EXPECT_EQ(kQwen35MoEConvergenceTimingMeasuredRequests, 3);
    EXPECT_EQ(kQwen35MoEConvergenceTimingDecodeForwards, 2);
    EXPECT_EQ(kQwen35MoEConvergenceTimingPromptRows, 17u)
        << "The speed witness must still traverse captured 16+1 prefill segments";
    EXPECT_EQ(qwen35MoEConvergenceTimingRequestRoutedRows(), 19u);
    EXPECT_EQ(qwen35MoEConvergenceTimingCohortRoutedRows(), 57u);
    EXPECT_EQ(qwen35MoEMaximumNumericalParityRoutedRows(), 24u);
    EXPECT_EQ(qwen35MoEConvergenceProtectedRoutedRows(), 81u);
    EXPECT_EQ(qwen35MoEConvergenceTrainingMaximumRoutedRows(), 36u);
    EXPECT_EQ(kQwen35MoEConvergenceHistogramWindowRows, 118);
    EXPECT_GT(
        static_cast<std::uint64_t>(
            kQwen35MoEConvergenceHistogramWindowRows),
        qwen35MoEConvergenceProtectedRoutedRows() +
            qwen35MoEConvergenceTrainingMaximumRoutedRows())
        << "One admitted request, the matched cohort, and canonical parity must remain in one immutable epoch";
    EXPECT_EQ(
        speedup.dynamic_rebalance.window_size,
        kQwen35MoEConvergenceHistogramWindowRows);
    EXPECT_EQ(
        speedup.dynamic_rebalance.max_window_size,
        kQwen35MoEConvergenceHistogramWindowRows);
    EXPECT_FLOAT_EQ(speedup.dynamic_rebalance.window_growth_factor, 1.0F);
    EXPECT_EQ(speedup.dynamic_rebalance.migration_transfer_slots, 49u);
    EXPECT_EQ(
        speedup.dynamic_rebalance.resolvedMigrationCyclesPerWave(),
        49u);

    const auto &movement = find_dynamic(ModelParityMTP::Depth1);
    EXPECT_FALSE(movement.requiresObservedConvergenceSpeedup());
    EXPECT_EQ(
        movement.dynamic_rebalance.window_size,
        kQwen35MoEMovementProofInitialWindowRows);
    EXPECT_EQ(movement.dynamic_rebalance.max_window_size, 4096);
    EXPECT_FLOAT_EQ(
        movement.dynamic_rebalance.window_growth_factor,
        4096.0F /
            static_cast<float>(
                kQwen35MoEMovementProofInitialWindowRows));
    EXPECT_EQ(movement.dynamic_rebalance.migration_transfer_slots, 49u);
    EXPECT_FALSE(movement.dynamic_rebalance.migration_cycles_per_wave.has_value());
    EXPECT_EQ(
        movement.dynamic_rebalance.resolvedMigrationCyclesPerWave(),
        movement.dynamic_rebalance.migration_transfer_slots);
    EXPECT_EQ(
        movement.dynamic_rebalance.device_min_maintenance_period_tokens,
        4096);
}

/**
 * @brief Apply each depth-15 drift budget only to policies that can reach it.
 *
 * The ROCm/CPU Dynamic branch repeatedly measured 0.06695 KL after fourteen
 * quantized recurrent round trips. This typed expansion proof prevents a
 * future matrix rewrite from either restoring false deep-recursion rejections
 * or weakening ordinary/depth-three MTP evidence along with the fixed and
 * adaptive policies whose admitted ceiling is depth fifteen.
 */
TEST(Qwen122NumericalThresholdGeometry,
     Depth15KLBudgetCoversFixedAndAdaptiveDepthWithoutWeakeningShallowPolicies)
{
    const Qwen122OverlayTopologySpec spec{
        .test_id = "typed_numerical_threshold",
        .cuda_participants = 0,
        .rocm_participants = 1,
        .cpu_participants = 2,
        .mpi_ranks = 2,
        .continuation = Qwen122ContinuationBackend::ROCm,
        .dynamic_speedup_witness =
            ModelParityDynamicSpeedupWitness::Disabled,
    };
    const auto cases = expandModelParityDefinition(
        qwen122ExpertOverlayParityDefinition(spec));

    const auto threshold_for = [&](ModelParityMTP mtp) -> float
    {
        const auto found = std::find_if(
            cases.begin(), cases.end(),
            [&](const ModelParityCase &test_case)
            {
                return test_case.expert_overlay.has_value() &&
                       test_case.expert_overlay->owner_order ==
                           RoutedExpertOwnerOrder::Ordinal &&
                       test_case.expert_overlay->movement ==
                           ModelParityExpertMovement::Dynamic &&
                       test_case.mtp == mtp;
            });
        if (found == cases.end() ||
            !found->thresholds.mtp_kl_threshold.has_value())
        {
            throw std::logic_error(
                "Generated Qwen122 MTP threshold cell is missing");
        }
        return *found->thresholds.mtp_kl_threshold;
    };

    EXPECT_FLOAT_EQ(threshold_for(ModelParityMTP::Depth3), 0.05f);
    EXPECT_FLOAT_EQ(threshold_for(ModelParityMTP::DynamicDepth), 0.07f);
    EXPECT_FLOAT_EQ(threshold_for(ModelParityMTP::Depth15), 0.07f);

    const auto aggregate_floor_for =
        [&](ModelParityMTP mtp) -> std::optional<float>
    {
        const auto found = std::find_if(
            cases.begin(), cases.end(),
            [&](const ModelParityCase &test_case)
            {
                return test_case.expert_overlay.has_value() &&
                       test_case.expert_overlay->owner_order ==
                           RoutedExpertOwnerOrder::Ordinal &&
                       test_case.expert_overlay->movement ==
                           ModelParityExpertMovement::Dynamic &&
                       test_case.mtp == mtp;
            });
        if (found == cases.end())
            throw std::logic_error(
                "Generated Qwen122 recursive aggregate cell is missing");
        return found->mtp_recursive_aggregate_cosine_floor;
    };

    EXPECT_FALSE(
        aggregate_floor_for(ModelParityMTP::Depth3).has_value());
    ASSERT_TRUE(
        aggregate_floor_for(ModelParityMTP::DynamicDepth).has_value());
    ASSERT_TRUE(
        aggregate_floor_for(ModelParityMTP::Depth15).has_value());
    EXPECT_FLOAT_EQ(
        *aggregate_floor_for(ModelParityMTP::DynamicDepth), 0.98f);
    EXPECT_FLOAT_EQ(
        *aggregate_floor_for(ModelParityMTP::Depth15), 0.98f);
}

/**
 * @brief The complete 122B matrix owns two representative timing cohorts.
 *
 * Every Dynamic cell still proves physical movement. Matched throughput is
 * intentionally sampled once for CUDA/CPU and once for ROCm/CPU rather than
 * multiplied over all nine topologies, both owner orders, and six MTP modes.
 */
TEST(Qwen122DynamicWaveGeometry,
     ProductionMatrixSelectsOnlyTypedTransportSpeedupWitnesses)
{
    std::size_t dynamic_cells = 0u;
    std::size_t movement_only_cells = 0u;
    std::set<std::string> witness_topologies;

    for (const auto &spec : qwen122OverlayTopologySpecs())
    {
        const auto cases = expandModelParityDefinition(
            qwen122ExpertOverlayParityDefinition(spec));
        ASSERT_EQ(cases.size(), 24u);
        for (const auto &test_case : cases)
        {
            // Every cell must report the schema's actual activation storage;
            // FP16 here refers only to the independent KV-cache axis.
            EXPECT_EQ(test_case.activation_precision, ActivationPrecision::FP32);
            EXPECT_NE(test_case.testName().find("ActFP32"), std::string::npos);
            if (!test_case.requiresPhysicalExpertMovement())
                continue;
            ++dynamic_cells;
            if (!test_case.requiresObservedConvergenceSpeedup())
            {
                ++movement_only_cells;
                continue;
            }

            witness_topologies.insert(test_case.topology.test_id);
            ASSERT_TRUE(test_case.expert_overlay.has_value());
            EXPECT_EQ(
                test_case.expert_overlay->owner_order,
                RoutedExpertOwnerOrder::Random);
            EXPECT_EQ(test_case.mtp, ModelParityMTP::Off);
            EXPECT_EQ(
                test_case.activation_precision,
                ActivationPrecision::FP32);
            EXPECT_EQ(
                test_case.kv_cache_precision,
                KVCachePrecision::FP16);
        }
    }

    EXPECT_EQ(dynamic_cells, 108u);
    EXPECT_EQ(movement_only_cells, 106u);
    EXPECT_EQ(
        witness_topologies,
        (std::set<std::string>{
            "ROCm4_CPU2_2xMPI_NodeExpertOverlay",
            "CUDA2_CPU2_2xMPI_NodeExpertOverlay",
        }));
}
/** @brief HTTP certification tags select four existing Dynamic/adaptive cells only. */
TEST(Qwen122E2ECertification, SelectsInitialTopologySubsetWithoutExpandingParity)
{
    std::set<std::string> selected;
    for (const auto &spec : qwen122OverlayTopologySpecs())
    {
        const auto cases = expandModelParityDefinition(qwen122ExpertOverlayParityDefinition(spec));
        ASSERT_EQ(cases.size(), 24u);
        for (const auto &cell : cases)
        {
            if (!cell.e2e_certification) continue;
            EXPECT_EQ(cell.e2e_certification->readiness_timeout_seconds, 180);
            EXPECT_TRUE(selected.insert(cell.topology.test_id).second);
            EXPECT_EQ(cell.mtp, ModelParityMTP::DynamicDepth);
            ASSERT_TRUE(cell.expert_overlay);
            EXPECT_EQ(cell.expert_overlay->movement, ModelParityExpertMovement::Dynamic);
            EXPECT_EQ(cell.expert_overlay->owner_order, RoutedExpertOwnerOrder::Ordinal);
            const auto config = cell.makeOrchestrationConfig(cell.model.model_path, 0);
            EXPECT_EQ(config.moe_rebalance.mode, MoERebalanceRuntimeMode::Dynamic);
            EXPECT_TRUE(config.prefix_cache.enabled);
            auto arguments = modelParityServerArguments(cell);
            arguments.insert(arguments.begin(), "llaminar2");
            std::vector<char *> argv;
            for (auto &argument : arguments) argv.push_back(argument.data());
            const auto parsed = OrchestrationConfigParser{}.parseArgs(
                static_cast<int>(argv.size()), argv.data());
            ASSERT_TRUE(parsed.moe_routed_expert_plan);
            const auto &expected = *config.moe_routed_expert_plan;
            const auto &actual = *parsed.moe_routed_expert_plan;
            EXPECT_EQ(actual.continuation_domain, expected.continuation_domain);
            ASSERT_EQ(actual.domains.size(), expected.domains.size());
            ASSERT_EQ(actual.routed_tiers.size(), expected.routed_tiers.size());
            for (std::size_t i = 0; i < actual.domains.size(); ++i)
            {
                EXPECT_EQ(actual.domains[i].participants, expected.domains[i].participants);
                EXPECT_EQ(actual.domains[i].scope, expected.domains[i].scope);
                EXPECT_EQ(actual.domains[i].toExecutionDomainDefinition().ranks,
                          expected.domains[i].toExecutionDomainDefinition().ranks);
                EXPECT_EQ(actual.routed_tiers[i].priority, expected.routed_tiers[i].priority);
                EXPECT_EQ(actual.routed_tiers[i].max_experts_per_layer, 0u);
            }
            EXPECT_EQ(parsed.moe_rebalance.mode, MoERebalanceRuntimeMode::Dynamic);
            EXPECT_EQ(parsed.mtp.depth_policy.mode, config.mtp.depth_policy.mode);
            EXPECT_EQ(parsed.mtp.depth_policy.max_depth, 15);
        }
    }
    EXPECT_EQ(selected, (std::set<std::string>{
        "CUDA2_CPU2_2xMPI_NodeExpertOverlay",
        "ROCm2_CPU2_2xMPI_NodeExpertOverlay",
        "ROCm4_CPU2_2xMPI_NodeExpertOverlay",
        "CUDA2_ROCm4_2xMPI_NodeExpertOverlay",
    }));
}

INSTANTIATE_TEST_SUITE_P(
    Qwen35_122B_ExpertOverlay,
    Qwen35MoENodeExpertOverlayParityTest,
    ::testing::ValuesIn(qwen122ExpertOverlayParityCases()),
    [](const ::testing::TestParamInfo<ModelParityCase> &info)
    { return info.param.testName(); });

}
