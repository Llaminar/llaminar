/**
 * @file Test__ModelParityDefinition.cpp
 * @brief Device-free contracts for canonical model-parity matrix expansion.
 *
 * These tests prove that a model/topology declaration expands into the exact
 * standard matrix and that every generated policy reaches the production
 * configuration without parsing its GoogleTest name.  They intentionally load
 * no model and touch no accelerator; real-weight campaigns certify execution.
 */

#include "integration/parity/ModelParityDefinition.h"
#include "integration/parity/qwen36/Qwen36ModelParityDefinitions.h"

#include "config/OrchestrationConfig.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <tuple>
#include <utility>

namespace llaminar2::test::parity
{
    namespace
    {
        /** @return Minimal valid real-model identity for generator tests. */
        ModelParityModelDefinition makeModelDefinition(int maximum_mtp_depth = 0)
        {
            return ModelParityModelDefinition{
                .test_id = "QwenTest",
                .model_path = "/models/qwen-test.gguf",
                .reference_directory = "/references/qwen-test",
                .prompt = "typed parity",
                .token_ids = {1, 2, 3},
                .decode_steps = 5,
                .max_seq_len = 8192,
                .transformer_layers = 24,
                .attention_heads = 14,
                .kv_heads = 2,
                .maximum_mtp_draft_depth = maximum_mtp_depth,
            };
        }

        /** @return Minimal single-CUDA topology with no ExpertOverlay. */
        ModelParityTopologyDefinition makeSingleDeviceTopology()
        {
            return ModelParityTopologyDefinition{
                .test_id = "CUDA0",
                .kind = ModelParityTopologyKind::SingleDevice,
                .participants = {
                    ModelParityParticipant{
                        .address = GlobalDeviceAddress::cuda(0),
                        .world_rank = 0,
                    },
                },
            };
        }

        /** @return Enabled immutable blueprint sufficient for policy projection. */
        std::shared_ptr<const MoERoutedExpertPlacementPlan> makeOverlayPlan()
        {
            auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
            plan->enabled = true;
            plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
            plan->owner_order = RoutedExpertOwnerOrder::Ordinal;
            return plan;
        }

        /**
         * @return Two-rank heterogeneous ExpertOverlay topology.
         *
         * Rank zero deliberately owns two CUDA participants while rank one
         * owns four ROCm participants.  This catches accidental rank inference
         * from participant order or a presumed one-device-per-rank topology.
         */
        ModelParityTopologyDefinition makeOverlayTopology()
        {
            return ModelParityTopologyDefinition{
                .test_id = "CUDA2_ROCm4_NodeOverlay",
                .kind = ModelParityTopologyKind::NodeMultiDomain,
                .participants = {
                    {GlobalDeviceAddress::cuda(0), 0},
                    {GlobalDeviceAddress::cuda(1), 0},
                    {GlobalDeviceAddress::rocm(0), 1},
                    {GlobalDeviceAddress::rocm(1), 1},
                    {GlobalDeviceAddress::rocm(2), 1},
                    {GlobalDeviceAddress::rocm(3), 1},
                },
                .collective = Collective::HETEROGENEOUS,
                .mpi_ranks = 2,
                .expert_overlay_plan = makeOverlayPlan(),
            };
        }

        /** @return Definition with deterministic threshold/config sentinels. */
        ModelParityDefinition makeDefinition(
            ModelParityTopologyDefinition topology,
            int maximum_mtp_depth = 0)
        {
            ModelParityDefinition definition;
            definition.model = makeModelDefinition(maximum_mtp_depth);
            definition.topology = std::move(topology);
            definition.thresholds.cosine_threshold = 0.9999f;
            definition.thresholds.decode_cosine_threshold = 0.999f;
            definition.dynamic_rebalance.window_size = 17;
            definition.dynamic_rebalance.migration_max_cycles_per_wave = 3;
            definition.dynamic_rebalance.device_maintenance_slack_tokens = 0;
            definition.dynamic_rebalance
                .device_min_maintenance_period_tokens = 1;
            definition.dynamic_rebalance
                .device_initial_maintenance_period_tokens = 1;
            return definition;
        }

        /** @return Canonical case matching an exact typed policy tuple. */
        const ModelParityCase &findCase(
            const std::vector<ModelParityCase> &cases,
            RoutedExpertOwnerOrder owner_order,
            ModelParityExpertMovement movement,
            ModelParityMTP mtp)
        {
            const auto found = std::find_if(
                cases.begin(), cases.end(),
                [&](const ModelParityCase &test_case)
                {
                    return test_case.expert_overlay.has_value() &&
                           test_case.expert_overlay->owner_order == owner_order &&
                           test_case.expert_overlay->movement == movement &&
                           test_case.mtp == mtp;
                });
            EXPECT_NE(found, cases.end());
            return *found;
        }

        /**
         * @brief Device-free access to the parity graph inventory policy.
         *
         * The production fixture normally evaluates this policy during runner
         * construction.  The probe deliberately does not invoke fixture setup,
         * load a model, or create a backend context.
         */
        class SnapshotInventoryProbe final : public ParityTestBase
        {
        public:
            void TestBody() override
            {
            }

            std::string getBackendName() override
            {
                return "InventoryProbe";
            }

            void configure(
                const std::filesystem::path &reference_directory,
                ParitySnapshotCaptureInventory inventory)
            {
                config_.snapshot_dir = reference_directory.string();
                config_.graph_snapshot_policy.enabled = true;
                config_.graph_snapshot_policy.capture_inventory = inventory;
                config_.graph_snapshot_policy
                    .prefill_snapshot_capture_filter = {"LM_HEAD"};
            }

            std::vector<std::string> captureFilter() const
            {
                return paritySnapshotSetupCaptureFilter();
            }

            /** Declare stages whose oracle values require live collectives. */
            void requirePostCollectiveSnapshots(
                std::vector<std::string> semantic_stages)
            {
                config_.collective_evidence_source =
                    ParityCollectiveEvidenceSource::PostCollectiveSnapshot;
                config_.allreduce_stages = std::move(semantic_stages);
            }
        };

        /** @brief Unique temporary reference directory removed at scope exit. */
        class ScopedReferenceDirectory final
        {
        public:
            ScopedReferenceDirectory()
                : path_(
                      std::filesystem::temp_directory_path() /
                      ("llaminar_parity_inventory_" +
                       std::to_string(
                           std::chrono::steady_clock::now()
                               .time_since_epoch()
                               .count())))
            {
                std::filesystem::create_directories(path_);
            }

            ~ScopedReferenceDirectory()
            {
                std::error_code ignored;
                std::filesystem::remove_all(path_, ignored);
            }

            const std::filesystem::path &path() const
            {
                return path_;
            }

            void add(const std::string &filename) const
            {
                std::ofstream output(path_ / filename, std::ios::binary);
                ASSERT_TRUE(output.is_open());
                output.put('\0');
                ASSERT_TRUE(output.good());
            }

        private:
            std::filesystem::path path_;
        };
    } // namespace

    TEST(ModelParityDefinition, SnapshotInventoryDefaultsToAuthenticatedCheckpoints)
    {
        ScopedReferenceDirectory reference;
        reference.add("EMBEDDING.npy");
        reference.add("decode_step3_MTP2_FINAL_NORM.npy");
        reference.add("decode_step_bad.npy");
        reference.add("untrusted_scratch.txt");

        SnapshotInventoryProbe probe;
        probe.configure(
            reference.path(),
            ParitySnapshotCaptureInventory::AuthenticatedReferenceCheckpoints);

        EXPECT_EQ(
            probe.captureFilter(),
            (std::vector<std::string>{
                "EMBEDDING",
                "LM_HEAD",
                "MTP0_FINAL_NORM",
                "MTP2_FINAL_NORM",
            }));
    }

    TEST(ModelParityDefinition, EveryPublishedOutputIsAnExplicitEmptyFilter)
    {
        SnapshotInventoryProbe probe;
        probe.configure(
            "/reference-pack-is-deliberately-not-read",
            ParitySnapshotCaptureInventory::EveryPublishedOutput);

        EXPECT_TRUE(probe.captureFilter().empty());
    }

    TEST(ModelParityDefinition, PostCollectiveEvidenceInstallsCompletedGraphKeys)
    {
        ScopedReferenceDirectory reference;
        reference.add("layer0_ATTENTION_OUTPUT.npy");
        reference.add("layer0_FFN_NORM.npy");
        reference.add("layer0_MOE_EXPERT_OUTPUT.npy");

        SnapshotInventoryProbe probe;
        probe.configure(
            reference.path(),
            ParitySnapshotCaptureInventory::AuthenticatedReferenceCheckpoints);
        probe.requirePostCollectiveSnapshots(
            {"ATTENTION_OUTPUT", "MOE_EXPERT_OUTPUT"});

        const auto filter = probe.captureFilter();
        EXPECT_NE(
            std::find(
                filter.begin(), filter.end(),
                "layer0_ATTENTION_OUTPUT_ALLREDUCED"),
            filter.end());
        EXPECT_NE(
            std::find(
                filter.begin(), filter.end(),
                "layer0_MOE_EXPERT_OUTPUT_ALLREDUCED"),
            filter.end());
        EXPECT_EQ(
            std::find(
                filter.begin(), filter.end(),
                "layer0_FFN_NORM_ALLREDUCED"),
            filter.end());
    }

    TEST(ModelParityDefinition, DenseQwen36DeclaresExactOptimizedMTPCheckpointSurface)
    {
        const auto definition = qwen36::qwen36DenseParityDefinition(
            qwen36::qwen36SingleDeviceTopology(
                "ROCm0", GlobalDeviceAddress::rocm(0)),
            "/references/qwen36-dense-rocm");

        const auto &surface = definition.model.mtp_checkpoint_surface;
        EXPECT_EQ(
            surface.size(),
            qwen36::kQwen36DenseMTPModelStageSuffixes.size() + 1u);
        EXPECT_EQ(surface.front(), "TERMINAL_HIDDEN_ROW_SELECT");
        EXPECT_EQ(
            std::find(surface.begin(), surface.end(), "FFN_SWIGLU"),
            surface.end())
            << "the optimized dense graph must not manufacture a fused-away "
               "temporary merely for parity";
        EXPECT_NE(
            std::find(surface.begin(), surface.end(), "FFN_DOWN"),
            surface.end());

        const std::set<std::string> unique_surface(
            surface.begin(), surface.end());
        EXPECT_EQ(unique_surface.size(), surface.size());

        const auto cases = expandModelParityDefinition(definition);
        ASSERT_EQ(cases.size(), 6u);
        for (const auto &test_case : cases)
            EXPECT_EQ(test_case.model.mtp_checkpoint_surface, surface);
    }

    TEST(ModelParityDefinition, PlainDefinitionHasOneMandatoryPrefixLifecycleCell)
    {
        const auto cases = expandModelParityDefinition(
            makeDefinition(makeSingleDeviceTopology()));

        ASSERT_EQ(cases.size(), 1u);
        EXPECT_FALSE(cases.front().expert_overlay.has_value());
        EXPECT_EQ(cases.front().mtp, ModelParityMTP::Off);
        EXPECT_EQ(
            cases.front().prefix_cache_block_size,
            kModelParityPrefixRestoreProofBlockSize);
        EXPECT_EQ(
            cases.front().movementEvidence(),
            ModelParityMovementEvidence::NotApplicable);
        EXPECT_EQ(
            cases.front().testName(),
            "QwenTest_CUDA0_ActFP32_KVFP16_MTPOff");
    }

    TEST(ModelParityDefinition, ExpertOverlayStandardMTPExpandsExactTwentyFourCells)
    {
        auto definition = makeDefinition(
            makeOverlayTopology(), kModelParityRequiredMaximumMTPDepth);
        definition.features.mtp = ModelParityAxisProfile::Standard;

        const auto cases = expandModelParityDefinition(definition);

        ASSERT_EQ(cases.size(), 24u);
        std::set<std::string> names;
        std::set<std::tuple<int, int, int>> policies;
        for (const auto &test_case : cases)
        {
            ASSERT_TRUE(test_case.expert_overlay.has_value());
            EXPECT_EQ(
                test_case.prefix_cache_block_size,
                kModelParityPrefixRestoreProofBlockSize);
            names.insert(test_case.testName());
            policies.emplace(
                static_cast<int>(test_case.expert_overlay->owner_order),
                static_cast<int>(test_case.expert_overlay->movement),
                static_cast<int>(test_case.mtp));
        }
        EXPECT_EQ(names.size(), 24u);
        EXPECT_EQ(policies.size(), 24u);
    }

    TEST(ModelParityDefinition, PrefixRestoreDoesNotMultiplyOverlayMTPMatrix)
    {
        auto definition = makeDefinition(
            makeOverlayTopology(), kModelParityRequiredMaximumMTPDepth);
        definition.features.mtp = ModelParityAxisProfile::Standard;

        const auto cases = expandModelParityDefinition(definition);

        ASSERT_EQ(cases.size(), 24u);
        for (const auto &test_case : cases)
        {
            EXPECT_EQ(
                test_case.prefix_cache_block_size,
                kModelParityPrefixRestoreProofBlockSize);
            EXPECT_EQ(
                test_case.testName().find("Prefix"),
                std::string::npos)
                << "mandatory prefix restore must not be encoded as a case axis";
        }
    }

    TEST(ModelParityDefinition, PrecisionAxesUseTheSameCentralCrossProduct)
    {
        auto definition = makeDefinition(makeSingleDeviceTopology());
        definition.precisions.activation = {
            ActivationPrecision::FP32,
            ActivationPrecision::BF16,
        };
        definition.precisions.kv_cache = {
            KVCachePrecision::FP32,
            KVCachePrecision::FP16,
            KVCachePrecision::Q8_1,
        };
        auto q8_thresholds = definition.thresholds;
        q8_thresholds.kl_threshold = 0.123f;
        definition.precisions.threshold_overrides = {
            {
                .activation = ActivationPrecision::BF16,
                .kv_cache = KVCachePrecision::Q8_1,
                .thresholds = q8_thresholds,
            },
        };

        const auto cases = expandModelParityDefinition(definition);

        ASSERT_EQ(cases.size(), 6u);
        std::set<std::pair<int, int>> precision_pairs;
        for (const auto &test_case : cases)
        {
            precision_pairs.emplace(
                static_cast<int>(test_case.activation_precision),
                static_cast<int>(test_case.kv_cache_precision));
        }
        EXPECT_EQ(precision_pairs.size(), 6u);
        const auto overridden = std::find_if(
            cases.begin(), cases.end(),
            [](const ModelParityCase &test_case)
            {
                return test_case.activation_precision ==
                           ActivationPrecision::BF16 &&
                       test_case.kv_cache_precision ==
                           KVCachePrecision::Q8_1;
            });
        ASSERT_NE(overridden, cases.end());
        EXPECT_FLOAT_EQ(overridden->thresholds.kl_threshold, 0.123f);
    }

    TEST(ModelParityDefinition, TypedCaseProjectsStaticPolicyWithoutMovement)
    {
        auto definition = makeDefinition(
            makeOverlayTopology(), kModelParityRequiredMaximumMTPDepth);
        definition.features.mtp = ModelParityAxisProfile::Standard;
        const auto cases = expandModelParityDefinition(definition);
        const auto &test_case = findCase(
            cases,
            RoutedExpertOwnerOrder::Random,
            ModelParityExpertMovement::Static,
            ModelParityMTP::Off);

        const TestConfig legacy = test_case.toTestConfig();
        ASSERT_EQ(legacy.devices.size(), 6u);
        EXPECT_EQ(legacy.devices[0], ParityDeviceType::CUDA);
        EXPECT_EQ(legacy.devices[2], ParityDeviceType::ROCm);
        EXPECT_EQ(legacy.parallelism, Parallelism::None);
        EXPECT_EQ(legacy.mpi_ranks, 2);
        EXPECT_EQ(legacy.moe_rebalance.mode, MoERebalanceRuntimeMode::Off);
        EXPECT_EQ(
            legacy.moe_movement_expectation,
            ParityMoEMovementExpectation::NoMovement);
        EXPECT_EQ(
            legacy.mtp_expectation,
            ParityMTPExpectation::Disabled);
        EXPECT_EQ(legacy.mtp_expected_draft_depth, 0);
        EXPECT_EQ(legacy.mtp_expected_graph_capacity, 0);
        EXPECT_FALSE(legacy.moe_rebalance_exercise.enabled);
        EXPECT_EQ(
            legacy.routed_expert_owner_order,
            RoutedExpertOwnerOrder::Random);
        ASSERT_NE(legacy.moe_routed_expert_plan, nullptr);
        EXPECT_NE(
            legacy.moe_routed_expert_plan.get(),
            test_case.topology.expert_overlay_plan.get());
        EXPECT_EQ(
            legacy.moe_routed_expert_plan->residency_policy,
            RoutedExpertResidencyPolicy::StaticById);
        EXPECT_EQ(
            test_case.movementEvidence(),
            ModelParityMovementEvidence::NoMovement);

        OrchestrationConfig runtime;
        test_case.applyRuntimePolicy(runtime);
        EXPECT_FALSE(runtime.mtp.enabled);
        EXPECT_EQ(runtime.mtp.graph_capacity_draft_tokens, 0);
        EXPECT_TRUE(runtime.prefix_cache.enabled);
        EXPECT_EQ(
            runtime.prefix_cache.storage_mode,
            PrefixCacheStorageMode::Tiered);
        EXPECT_EQ(
            runtime.prefix_cache.block_size,
            kModelParityPrefixRestoreProofBlockSize);
        EXPECT_GT(runtime.prefix_cache.ram_budget_bytes, 0u);
        EXPECT_GT(runtime.prefix_cache.device_budget_bytes, 0u);
        EXPECT_GT(runtime.prefix_cache.disk_budget_bytes, 0u);
        EXPECT_EQ(runtime.moe_rebalance.mode, MoERebalanceRuntimeMode::Off);
        ASSERT_NE(runtime.moe_routed_expert_plan, nullptr);
        EXPECT_EQ(
            runtime.moe_routed_expert_plan->owner_order,
            RoutedExpertOwnerOrder::Random);
    }

    TEST(ModelParityDefinition, TypedCaseProjectsDynamicDepthPrefixAndMovement)
    {
        auto definition = makeDefinition(
            makeOverlayTopology(), kModelParityRequiredMaximumMTPDepth);
        definition.features.mtp = ModelParityAxisProfile::Standard;
        definition.precisions.activation = {ActivationPrecision::BF16};
        definition.precisions.kv_cache = {KVCachePrecision::Q8_1};
        const auto cases = expandModelParityDefinition(definition);
        const auto &test_case = findCase(
            cases,
            RoutedExpertOwnerOrder::Ordinal,
            ModelParityExpertMovement::Dynamic,
            ModelParityMTP::DynamicDepth);

        OrchestrationConfig runtime;
        test_case.applyRuntimePolicy(runtime);

        EXPECT_EQ(runtime.activation_precision, "bf16");
        EXPECT_EQ(runtime.kv_cache_precision, "q8_1");
        EXPECT_TRUE(runtime.prefix_cache.enabled);
        EXPECT_EQ(runtime.prefix_cache.storage_mode, PrefixCacheStorageMode::Tiered);
        EXPECT_EQ(
            runtime.prefix_cache.block_size,
            kModelParityPrefixRestoreProofBlockSize);
        EXPECT_TRUE(runtime.mtp.enabled);
        EXPECT_EQ(runtime.mtp.draft_tokens, 15);
        EXPECT_EQ(runtime.mtp.graph_capacity_draft_tokens, 15);
        EXPECT_EQ(runtime.mtp.depth_policy.mode, MTPDepthPolicyMode::Dynamic);
        EXPECT_EQ(runtime.mtp.depth_policy.min_depth, 1);
        EXPECT_EQ(runtime.mtp.depth_policy.max_depth, 15);
        EXPECT_EQ(runtime.mtp.depth_policy.initial_depth, 15);
        EXPECT_EQ(runtime.mtp.depth_policy.window_size, 1);
        EXPECT_EQ(runtime.moe_rebalance.mode, MoERebalanceRuntimeMode::Dynamic);
        EXPECT_EQ(runtime.moe_rebalance.window_size, 17);
        EXPECT_EQ(runtime.moe_rebalance.migration_max_cycles_per_wave, 3u);
        EXPECT_EQ(runtime.moe_rebalance.device_maintenance_slack_tokens, 0);
        EXPECT_EQ(
            runtime.moe_rebalance.device_min_maintenance_period_tokens,
            1);
        EXPECT_EQ(
            runtime.moe_rebalance.device_initial_maintenance_period_tokens,
            1);
        ASSERT_NE(runtime.moe_routed_expert_plan, nullptr);
        EXPECT_EQ(
            runtime.moe_routed_expert_plan->residency_policy,
            RoutedExpertResidencyPolicy::RoutedTierRebalanced);
        EXPECT_EQ(
            test_case.movementEvidence(),
            ModelParityMovementEvidence::MovementRequired);
        EXPECT_EQ(
            test_case.mtpEvidence(),
            ModelParityMTPEvidence::DynamicDepth);

        const TestConfig legacy = test_case.toTestConfig();
        EXPECT_EQ(
            legacy.moe_movement_expectation,
            ParityMoEMovementExpectation::PhysicalMovement);
        EXPECT_TRUE(legacy.moe_rebalance_exercise.enabled);
        EXPECT_TRUE(
            legacy.moe_rebalance_exercise
                .require_production_overlay_authority);
        EXPECT_TRUE(legacy.moe_rebalance_exercise.request_after_prefill);
        EXPECT_EQ(
            legacy.moe_rebalance_exercise.request_every_decode_steps,
            1);
        EXPECT_EQ(legacy.moe_rebalance_exercise.min_decode_steps, 2);
        EXPECT_FALSE(
            legacy.moe_rebalance_exercise.require_movement_epoch_advance);
        EXPECT_EQ(
            legacy.mtp_expectation,
            ParityMTPExpectation::DynamicDepth);
        EXPECT_EQ(legacy.mtp_expected_draft_depth, 15);
        EXPECT_EQ(legacy.mtp_expected_graph_capacity, 15);
    }

    TEST(ModelParityDefinition, ExpertOverlayRejectsEnvironmentOwnedMovementCadence)
    {
        auto definition = makeDefinition(makeOverlayTopology());
        definition.dynamic_rebalance.device_initial_maintenance_period_tokens =
            -1;

        EXPECT_THROW(
            (void)expandModelParityDefinition(definition),
            std::invalid_argument);
    }

    TEST(ModelParityDefinition, ServerConfigProjectsRankLocalHybridPipelineExactly)
    {
        auto topology = ModelParityTopologyDefinition{
            .test_id = "CUDA2_ROCm1_LocalPipeline",
            .kind = ModelParityTopologyKind::RankLocalPipelineParallel,
            .participants = {
                {GlobalDeviceAddress::cuda(0), 0},
                {GlobalDeviceAddress::cuda(1), 0},
                {GlobalDeviceAddress::rocm(0), 0},
            },
            .mpi_ranks = 1,
            .pipeline_stage_sizes = {2, 1},
            .pipeline_weights = {0.25f, 0.75f},
            .tensor_parallel_collective = Collective::NCCL,
        };
        const auto cases = expandModelParityDefinition(
            makeDefinition(std::move(topology)));
        ASSERT_EQ(cases.size(), 1u);

        const auto config = cases.front().makeOrchestrationConfig(
            "/tmp/staged-model.gguf", 0);

        EXPECT_EQ(config.model_path, "/tmp/staged-model.gguf");
        EXPECT_EQ(config.max_seq_len, 8192);
        EXPECT_EQ(config.tp_degree, 1);
        EXPECT_EQ(config.pp_degree, 2);
        EXPECT_EQ(config.pp_split, PPSplitMode::MANUAL);
        ASSERT_EQ(config.domain_definitions.size(), 2u);
        ASSERT_EQ(config.pp_stage_definitions.size(), 2u);
        EXPECT_EQ(config.domain_definitions[0].devices.size(), 2u);
        EXPECT_EQ(config.domain_definitions[0].scope, TPScope::RANK_LOCAL);
        EXPECT_EQ(config.domain_definitions[0].owner_rank, 0);
        EXPECT_EQ(
            config.domain_definitions[0].backend,
            CollectiveBackendType::NCCL);
        EXPECT_EQ(config.domain_definitions[1].devices.size(), 1u);
        EXPECT_EQ(config.pp_stage_definitions[0].first_layer, 0);
        EXPECT_EQ(config.pp_stage_definitions[0].last_layer, 5);
        EXPECT_EQ(config.pp_stage_definitions[1].first_layer, 6);
        EXPECT_EQ(config.pp_stage_definitions[1].last_layer, 23);
        EXPECT_TRUE(config.prefix_cache.enabled);
    }

    TEST(ModelParityDefinition, ServerConfigProjectsNodeTPRankOwnershipExactly)
    {
        auto topology = ModelParityTopologyDefinition{
            .test_id = "CPU2_NodeTP",
            .kind = ModelParityTopologyKind::NodeTensorParallel,
            .participants = {
                {GlobalDeviceAddress::cpu(0), 0},
                {GlobalDeviceAddress::cpu(1), 1},
            },
            .collective = Collective::MPI,
            .mpi_ranks = 2,
        };
        const auto cases = expandModelParityDefinition(
            makeDefinition(std::move(topology)));
        ASSERT_EQ(cases.size(), 1u);

        const auto config = cases.front().makeOrchestrationConfig(
            "/tmp/staged-model.gguf", 1);

        EXPECT_EQ(config.device_mode, DeviceAssignmentMode::EXPLICIT);
        EXPECT_EQ(config.tp_scope, TPScope::NODE_LOCAL);
        EXPECT_EQ(config.tp_degree, 2);
        EXPECT_EQ(config.pp_degree, 1);
        EXPECT_EQ(config.default_backend, CollectiveBackendType::MPI);
        ASSERT_EQ(config.device_map.size(), 2u);
        EXPECT_EQ(config.device_map[0].first, 0);
        EXPECT_EQ(config.device_map[0].second, GlobalDeviceAddress::cpu(0));
        EXPECT_EQ(config.device_map[1].first, 1);
        EXPECT_EQ(config.device_map[1].second, GlobalDeviceAddress::cpu(1));
    }

    TEST(ModelParityDefinition, ServerConfigLeavesOverlayAsSoleTopologyAuthority)
    {
        auto definition = makeDefinition(makeOverlayTopology());
        const auto cases = expandModelParityDefinition(definition);
        ASSERT_EQ(cases.size(), 4u);

        const auto config = cases.front().makeOrchestrationConfig(
            "/tmp/staged-model.gguf", 0);

        EXPECT_EQ(config.tp_degree, 1);
        EXPECT_EQ(config.pp_degree, 1);
        EXPECT_TRUE(config.device_map.empty());
        EXPECT_TRUE(config.tp_devices.empty());
        EXPECT_TRUE(config.domain_definitions.empty());
        ASSERT_NE(config.moe_routed_expert_plan, nullptr);
    }

    TEST(ModelParityDefinition, RejectsIncompleteOrContradictoryDeclarations)
    {
        auto insufficient_mtp = makeDefinition(
            makeSingleDeviceTopology(),
            kModelParityRequiredMaximumMTPDepth - 1);
        insufficient_mtp.features.mtp = ModelParityAxisProfile::Standard;
        EXPECT_THROW(
            static_cast<void>(expandModelParityDefinition(insufficient_mtp)),
            std::invalid_argument);

        auto empty_precision = makeDefinition(makeSingleDeviceTopology());
        empty_precision.precisions.activation.clear();
        EXPECT_THROW(
            static_cast<void>(expandModelParityDefinition(empty_precision)),
            std::invalid_argument);

        auto invalid_identifier = makeDefinition(makeSingleDeviceTopology());
        invalid_identifier.topology.test_id = "CUDA:0";
        EXPECT_THROW(
            static_cast<void>(expandModelParityDefinition(invalid_identifier)),
            std::invalid_argument);

        auto duplicate_precision =
            makeDefinition(makeSingleDeviceTopology());
        duplicate_precision.precisions.kv_cache = {
            KVCachePrecision::FP16,
            KVCachePrecision::FP16,
        };
        EXPECT_THROW(
            static_cast<void>(expandModelParityDefinition(
                duplicate_precision)),
            std::invalid_argument);

        auto invalid_precision_override =
            makeDefinition(makeSingleDeviceTopology());
        invalid_precision_override.precisions.threshold_overrides = {
            {
                .activation = ActivationPrecision::BF16,
                .kv_cache = KVCachePrecision::FP16,
            },
        };
        EXPECT_THROW(
            static_cast<void>(expandModelParityDefinition(
                invalid_precision_override)),
            std::invalid_argument);

        auto invalid_rank = makeDefinition(makeOverlayTopology());
        invalid_rank.topology.participants.back().world_rank = 2;
        EXPECT_THROW(
            static_cast<void>(expandModelParityDefinition(invalid_rank)),
            std::invalid_argument);

        auto duplicate_participant = makeDefinition(makeOverlayTopology());
        duplicate_participant.topology.participants.back().address =
            duplicate_participant.topology.participants.front().address;
        EXPECT_THROW(
            static_cast<void>(expandModelParityDefinition(
                duplicate_participant)),
            std::invalid_argument);

        auto uncovered_rank = makeDefinition(makeOverlayTopology());
        for (auto &participant : uncovered_rank.topology.participants)
            participant.world_rank = 0;
        EXPECT_THROW(
            static_cast<void>(expandModelParityDefinition(uncovered_rank)),
            std::invalid_argument);

        auto inventory_bound = makeDefinition(makeOverlayTopology());
        for (auto &participant : inventory_bound.topology.participants)
            participant.world_rank = std::nullopt;
        EXPECT_NO_THROW(
            static_cast<void>(expandModelParityDefinition(inventory_bound)));

        auto unresolved_local = makeDefinition(makeSingleDeviceTopology());
        unresolved_local.topology.participants.front().world_rank =
            std::nullopt;
        EXPECT_THROW(
            static_cast<void>(expandModelParityDefinition(unresolved_local)),
            std::invalid_argument);

        auto bad_pipeline = makeDefinition(makeSingleDeviceTopology());
        bad_pipeline.topology.pipeline_stage_sizes = {1};
        EXPECT_THROW(
            static_cast<void>(expandModelParityDefinition(bad_pipeline)),
            std::invalid_argument);

        auto missing_pipeline_layers = makeDefinition(
            ModelParityTopologyDefinition{
                .test_id = "CUDA_ROCm_LocalPipeline",
                .kind = ModelParityTopologyKind::RankLocalPipelineParallel,
                .participants = {
                    {GlobalDeviceAddress::cuda(0), 0},
                    {GlobalDeviceAddress::rocm(0), 0},
                },
                .pipeline_stage_sizes = {1, 1},
            });
        missing_pipeline_layers.model.transformer_layers = 0;
        EXPECT_THROW(
            static_cast<void>(expandModelParityDefinition(
                missing_pipeline_layers)),
            std::invalid_argument);

        auto invalid_uniform_node_tp = makeDefinition(
            ModelParityTopologyDefinition{
                .test_id = "CPU4_NodeTP",
                .kind = ModelParityTopologyKind::NodeTensorParallel,
                .participants = {
                    {GlobalDeviceAddress::cpu(0), 0},
                    {GlobalDeviceAddress::cpu(1), 1},
                    {GlobalDeviceAddress::cpu(2), 2},
                    {GlobalDeviceAddress::cpu(3), 3},
                },
                .collective = Collective::MPI,
                .mpi_ranks = 4,
            });
        EXPECT_THROW(
            static_cast<void>(expandModelParityDefinition(
                invalid_uniform_node_tp)),
            std::invalid_argument);

        auto disabled_overlay = makeDefinition(makeOverlayTopology());
        auto disabled_plan =
            std::make_shared<MoERoutedExpertPlacementPlan>();
        disabled_overlay.topology.expert_overlay_plan = disabled_plan;
        EXPECT_THROW(
            static_cast<void>(expandModelParityDefinition(disabled_overlay)),
            std::invalid_argument);
    }

} // namespace llaminar2::test::parity
