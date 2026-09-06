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
#include "integration/parity/ProductionParityModelPath.h"
#include "integration/parity/qwen35moe/Qwen35MoEModelParityDefinitions.h"
#include "integration/parity/qwen36/Qwen36ModelParityDefinitions.h"
#include "integration/parity/qwen36/Ornith15ModelParityDefinitions.h"
#include "integration/parity/qwen38/Qwen38ModelParityDefinitions.h"
#include "config/OrchestrationConfigParser.h"

#include "config/OrchestrationConfig.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <set>
#include <stdexcept>
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

        /** @return Overlay plan with one explicitly typed continuation domain. */
        std::shared_ptr<const MoERoutedExpertPlacementPlan>
        makeContinuationOverlayPlan(
            std::vector<GlobalDeviceAddress> participants)
        {
            auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
            plan->enabled = true;
            plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
            plan->owner_order = RoutedExpertOwnerOrder::Ordinal;
            plan->continuation_domain = "continuation";
            RoutedExpertDomain continuation;
            continuation.name = plan->continuation_domain;
            continuation.participants = std::move(participants);
            plan->domains.push_back(std::move(continuation));
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
            MoERebalanceRuntimeConfig dynamic_policy;
            dynamic_policy.window_size = 17;
            dynamic_policy.migration_transfer_slots = 3;
            dynamic_policy.device_maintenance_slack_tokens = 0;
            dynamic_policy
                .device_min_maintenance_period_tokens =
                    maximum_mtp_depth >=
                            kModelParityRequiredMaximumMTPDepth
                        ? kModelParityRequiredMaximumMTPDepth + 1
                        : 1;
            dynamic_policy
                .device_initial_maintenance_period_tokens = 1;
            definition.dynamic_rebalance = {
                .economic_movement = dynamic_policy,
                .economic_movement_and_observed_speedup = dynamic_policy,
            };
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

        /** @brief Restore one process environment variable at scope exit. */
        class ScopedEnvironmentVariable final
        {
        public:
            /**
             * @brief Install or remove an environment value for one test.
             * @param name Environment variable name.
             * @param value Replacement value, or nullopt to remove it.
             */
            ScopedEnvironmentVariable(
                std::string name,
                std::optional<std::string> value)
                : name_(std::move(name))
            {
                if (const char *previous = std::getenv(name_.c_str()))
                    previous_ = previous;
                if (value.has_value())
                {
                    if (setenv(name_.c_str(), value->c_str(), 1) != 0)
                    {
                        throw std::runtime_error(
                            "failed to install unit-test environment variable " +
                            name_);
                    }
                }
                else
                {
                    if (unsetenv(name_.c_str()) != 0)
                    {
                        throw std::runtime_error(
                            "failed to remove unit-test environment variable " +
                            name_);
                    }
                }
            }

            /** Restore the exact environment state observed on construction. */
            ~ScopedEnvironmentVariable()
            {
                if (previous_.has_value())
                    static_cast<void>(
                        setenv(name_.c_str(), previous_->c_str(), 1));
                else
                    static_cast<void>(unsetenv(name_.c_str()));
            }

            ScopedEnvironmentVariable(
                const ScopedEnvironmentVariable &) = delete;
            ScopedEnvironmentVariable &operator=(
                const ScopedEnvironmentVariable &) = delete;

        private:
            std::string name_;                    ///< Variable being guarded.
            std::optional<std::string> previous_; ///< Original optional value.
        };
    } // namespace

    TEST(ModelParityDefinition, ProcessCampaignRejectsMissingTmpfsAuthority)
    {
        ScopedEnvironmentVariable process_campaign(
            std::string(kProductionParityProcessCampaignEnvironment),
            std::string("1"));
        ScopedEnvironmentVariable no_ramdisk(
            std::string(kProductionParityModelRamdiskEnvironment),
            std::nullopt);

        try
        {
            static_cast<void>(productionParityResolvedModelPath(
                "/models/must-not-be-opened.gguf"));
            FAIL() << "A process campaign used its source GGUF without tmpfs";
        }
        catch (const std::runtime_error &error)
        {
            EXPECT_NE(
                std::string(error.what()).find(
                    "bypassed authenticated tmpfs model staging"),
                std::string::npos);
        }
    }

    /**
     * @brief MTP movement proof must retain every sidecar route projection.
     *
     * The executor filters each named stage output independently. This test
     * prevents a numerical `MTP0_MOE_EXPERT_OUTPUT` key from being mistaken
     * for implicit capture of the placement banks and domain assignment.
     */
    TEST(ModelParitySnapshotInventory,
         ExpertOverlayMTPIncludesPinnedSidecarRouteEvidence)
    {
        EXPECT_THROW(
            static_cast<void>(
                modelParityExpertOverlayRouteSnapshotInventory(
                    /*main_layer_count=*/0,
                    ModelParityMTP::Off)),
            std::invalid_argument);

        const auto without_mtp =
            modelParityExpertOverlayRouteSnapshotInventory(
                /*main_layer_count=*/2,
                ModelParityMTP::Off);
        const auto with_mtp =
            modelParityExpertOverlayRouteSnapshotInventory(
                /*main_layer_count=*/2,
                ModelParityMTP::Depth1);

        EXPECT_EQ(without_mtp.main_model.size(), 16u);
        EXPECT_TRUE(without_mtp.mtp_sidecar.empty());
        EXPECT_EQ(with_mtp.main_model.size(), 16u);
        EXPECT_EQ(with_mtp.mtp_sidecar.size(), 8u);
        for (const std::string_view suffix : {
                 "MOE_DOMAIN_ROUTE_PARTICIPANT_IDS",
                 "MOE_RUNTIME_ROUTE_WEIGHTS",
                 "MOE_ROUTE_CONTRIBUTIONS",
                 "MOE_OVERLAY_ROUTE_PARTICIPANTS_BANK0",
                 "MOE_OVERLAY_ROUTE_BANK0_EPOCH",
                 "MOE_OVERLAY_ROUTE_PARTICIPANTS_BANK1",
                 "MOE_OVERLAY_ROUTE_BANK1_EPOCH",
                 "MOE_OVERLAY_ROUTE_SELECTED_BANK",
             })
        {
            EXPECT_NE(
                std::find(
                    with_mtp.mtp_sidecar.begin(),
                    with_mtp.mtp_sidecar.end(),
                    "MTP0_" + std::string(suffix)),
                with_mtp.mtp_sidecar.end())
                << suffix;
        }
    }

    TEST(ModelParityDefinition, FocusedDiagnosticMayUseExplicitModelPath)
    {
        ScopedEnvironmentVariable no_process_campaign(
            std::string(kProductionParityProcessCampaignEnvironment),
            std::nullopt);
        ScopedEnvironmentVariable no_ramdisk(
            std::string(kProductionParityModelRamdiskEnvironment),
            std::nullopt);
        const std::string configured = "/models/focused-diagnostic.gguf";

        EXPECT_EQ(productionParityResolvedModelPath(configured), configured);
    }

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

    TEST(ModelParityDefinition, Qwen36DynamicEconomicsAreProductionValid)
    {
        const auto economics = qwen36::qwen36MoEDynamicParityEconomics();

        EXPECT_EQ(economics.mode, MoERebalanceRuntimeMode::Dynamic);
        EXPECT_GE(economics.dynamic_imbalance_threshold_per_mille, 1000u);
        EXPECT_EQ(
            economics.device_min_maintenance_period_tokens,
            kModelParityRequiredMaximumMTPDepth + 1);
    }

    TEST(ModelParityDefinition, PlainDefinitionHasOneMandatoryPrefixLifecycleCell)
    {
        const auto cases = expandModelParityDefinition(
            makeDefinition(makeSingleDeviceTopology()));

        ASSERT_EQ(cases.size(), 1u);
        EXPECT_FALSE(cases.front().expert_overlay.has_value());
        EXPECT_EQ(cases.front().mtp, ModelParityMTP::Off);
        EXPECT_EQ(cases.front().retained_mtp_draft_capacity, 0);
        EXPECT_EQ(
            cases.front().prefix_restore_geometry,
            ModelParityPrefixRestoreGeometry::AuthenticatedPromptBlock);
        EXPECT_EQ(
            cases.front().movementEvidence(),
            ModelParityMovementEvidence::NotApplicable);
        EXPECT_EQ(
            cases.front().testName(),
            "QwenTest_CUDA0_ActFP32_KVFP16_MTPOff");
    }

    TEST(ModelParityDefinition,
         MandatoryPrefixGeometryArchivesOneAuthenticatedPromptBlock)
    {
        EXPECT_EQ(productionParityPrefixRestoreProofBlockSize(1u), 1);
        EXPECT_EQ(productionParityPrefixRestoreProofBlockSize(9u), 9);
        EXPECT_THROW(
            (void)productionParityPrefixRestoreProofBlockSize(0u),
            std::invalid_argument);
    }

    TEST(ModelParityDefinition,
         DynamicSpeedupWitnessRejectsNonOverlayTopology)
    {
        auto definition = makeDefinition(makeSingleDeviceTopology());
        definition.features.dynamic_speedup_witness =
            ModelParityDynamicSpeedupWitness::Ordinal;

        try
        {
            (void)expandModelParityDefinition(definition);
            FAIL() << "A speed witness without ExpertOverlay must be rejected";
        }
        catch (const std::invalid_argument &error)
        {
            EXPECT_NE(
                std::string(error.what()).find(
                    "requires an ExpertOverlay topology"),
                std::string::npos);
        }
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
        std::size_t economic_movement_cells = 0;
        for (const auto &test_case : cases)
        {
            ASSERT_TRUE(test_case.expert_overlay.has_value());
            EXPECT_EQ(
                test_case.retained_mtp_draft_capacity,
                kModelParityRequiredMaximumMTPDepth)
                << "every active and control cell must share one setup-capacity identity";
            EXPECT_EQ(
                test_case.prefix_restore_geometry,
                ModelParityPrefixRestoreGeometry::AuthenticatedPromptBlock);
            names.insert(test_case.testName());
            policies.emplace(
                static_cast<int>(test_case.expert_overlay->owner_order),
                static_cast<int>(test_case.expert_overlay->movement),
                static_cast<int>(test_case.mtp));
            economic_movement_cells +=
                test_case.dynamic_evidence ==
                        ModelParityDynamicEvidence::EconomicMovement
                    ? 1u
                    : 0u;
            EXPECT_FALSE(test_case.requiresObservedConvergenceSpeedup())
                << "a GPU-only topology has no CPU-tier convergence witness";
        }
        EXPECT_EQ(names.size(), 24u);
        EXPECT_EQ(policies.size(), 24u);
        EXPECT_EQ(economic_movement_cells, 12u);
    }

    TEST(ModelParityDefinition,
         CanonicalTPAllreduceDiagnosticsRequireMTPAndHomogeneousTPAboveTwo)
    {
        const auto make_case = [](
                                   std::vector<GlobalDeviceAddress> participants,
                                   ModelParityMTP mtp)
        {
            ModelParityCase test_case;
            test_case.topology.test_id = "TypedContinuation";
            test_case.topology.expert_overlay_plan =
                makeContinuationOverlayPlan(std::move(participants));
            test_case.expert_overlay = ModelParityExpertOverlayPolicy{};
            test_case.mtp = mtp;
            return test_case;
        };

        EXPECT_FALSE(
            make_case(
                {GlobalDeviceAddress::cuda(0)},
                ModelParityMTP::Depth1)
                .requiresCanonicalTPAllreduceMTPDiagnostics());
        EXPECT_FALSE(
            make_case(
                {GlobalDeviceAddress::cuda(0),
                 GlobalDeviceAddress::cuda(1),
                 GlobalDeviceAddress::cuda(2),
                 GlobalDeviceAddress::cuda(3)},
                ModelParityMTP::Off)
                .requiresCanonicalTPAllreduceMTPDiagnostics());
        EXPECT_TRUE(
            make_case(
                {GlobalDeviceAddress::cuda(0),
                 GlobalDeviceAddress::cuda(1),
                 GlobalDeviceAddress::cuda(2),
                 GlobalDeviceAddress::cuda(3)},
                ModelParityMTP::Depth1)
                .requiresCanonicalTPAllreduceMTPDiagnostics());
        EXPECT_TRUE(
            make_case(
                {GlobalDeviceAddress::rocm(0),
                 GlobalDeviceAddress::rocm(1),
                 GlobalDeviceAddress::rocm(2),
                 GlobalDeviceAddress::rocm(3)},
                ModelParityMTP::DynamicDepth)
                .requiresCanonicalTPAllreduceMTPDiagnostics());
        EXPECT_FALSE(
            make_case(
                {GlobalDeviceAddress::cuda(0),
                 GlobalDeviceAddress::cuda(1),
                 GlobalDeviceAddress::rocm(0),
                 GlobalDeviceAddress::rocm(1)},
                ModelParityMTP::Depth1)
                .requiresCanonicalTPAllreduceMTPDiagnostics());
    }

    TEST(ModelParityDefinition,
         ExplicitDynamicWitnessSelectsOneOwnerAndOnePrecisionPair)
    {
        auto topology = makeOverlayTopology();
        topology.test_id = "CUDA2_CPU2_NodeOverlay";
        topology.participants = {
            {GlobalDeviceAddress::cuda(0), 0},
            {GlobalDeviceAddress::cuda(1), 0},
            {GlobalDeviceAddress::cpu(0), 1},
            {GlobalDeviceAddress::cpu(1), 1},
        };
        auto definition = makeDefinition(
            std::move(topology), kModelParityRequiredMaximumMTPDepth);
        definition.features.mtp = ModelParityAxisProfile::Standard;
        definition.features.dynamic_speedup_witness =
            ModelParityDynamicSpeedupWitness::Random;
        definition.precisions.activation = {
            ActivationPrecision::FP32,
        };
        definition.precisions.kv_cache = {
            KVCachePrecision::FP16,
            KVCachePrecision::Q8_1,
        };
        definition.dynamic_rebalance.economic_movement.window_size = 19;
        definition.dynamic_rebalance.economic_movement.max_window_size = 4096;
        definition.dynamic_rebalance.economic_movement
            .migration_transfer_slots = 49;
        definition.dynamic_rebalance.economic_movement
            .migration_cycles_per_wave = 2;
        definition.dynamic_rebalance
            .economic_movement_and_observed_speedup.window_size = 640;
        definition.dynamic_rebalance
            .economic_movement_and_observed_speedup.max_window_size = 640;
        definition.dynamic_rebalance
            .economic_movement_and_observed_speedup
            .migration_transfer_slots = 49;

        const auto cases = expandModelParityDefinition(definition);

        ASSERT_EQ(cases.size(), 48u);
        std::size_t witnesses = 0;
        std::size_t movement_only = 0;
        for (const auto &test_case : cases)
        {
            if (!test_case.requiresPhysicalExpertMovement())
            {
                EXPECT_EQ(
                    test_case.dynamic_evidence,
                    ModelParityDynamicEvidence::NotApplicable);
                EXPECT_EQ(test_case.dynamic_rebalance.window_size, 19);
                EXPECT_EQ(
                    test_case.dynamic_rebalance.migration_transfer_slots,
                    49u);
                EXPECT_EQ(
                    test_case.dynamic_rebalance
                        .resolvedMigrationExecutionStreams(),
                    moe_rebalance_policy::
                        kDefaultMigrationExecutionStreams);
                EXPECT_EQ(
                    test_case.dynamic_rebalance
                        .resolvedMigrationCyclesPerWave(),
                    2u);
                continue;
            }
            if (test_case.requiresObservedConvergenceSpeedup())
            {
                ++witnesses;
                EXPECT_EQ(test_case.mtp, ModelParityMTP::Off);
                EXPECT_EQ(test_case.dynamic_rebalance.window_size, 640);
                EXPECT_EQ(
                    test_case.dynamic_rebalance.migration_transfer_slots,
                    49u);
                EXPECT_EQ(
                    test_case.dynamic_rebalance
                        .resolvedMigrationExecutionStreams(),
                    moe_rebalance_policy::
                        kDefaultMigrationExecutionStreams);
                EXPECT_EQ(
                    test_case.dynamic_rebalance
                        .resolvedMigrationCyclesPerWave(),
                    49u);
            }
            else
            {
                ++movement_only;
                EXPECT_EQ(
                    test_case.dynamic_evidence,
                    ModelParityDynamicEvidence::EconomicMovement);
                EXPECT_EQ(test_case.dynamic_rebalance.window_size, 19);
                EXPECT_EQ(
                    test_case.dynamic_rebalance.migration_transfer_slots,
                    49u);
                EXPECT_EQ(
                    test_case.dynamic_rebalance
                        .resolvedMigrationExecutionStreams(),
                    moe_rebalance_policy::
                        kDefaultMigrationExecutionStreams);
                EXPECT_EQ(
                    test_case.dynamic_rebalance
                        .resolvedMigrationCyclesPerWave(),
                    2u);
            }
        }
        EXPECT_EQ(witnesses, 1u);
        EXPECT_EQ(movement_only, 23u);
    }

    TEST(ModelParityDefinition,
         Qwen35MoE35BSeparatesMovementAndObservedSpeedupHistogramPolicies)
    {
        const auto &spec =
            qwen35moe::qwen35MoE35BOverlayTopologySpecs().front();
        const auto cases = expandModelParityDefinition(
            qwen35moe::qwen35MoE35BGraphNativeParityDefinition(spec));

        const auto find_dynamic = [&cases](
                                      RoutedExpertOwnerOrder owner_order,
                                      ModelParityPrefillGraphMode prefill_mode)
            -> const ModelParityCase &
        {
            const auto found = std::find_if(
                cases.begin(), cases.end(),
                [&](const ModelParityCase &test_case)
                {
                    return test_case.expert_overlay.has_value() &&
                           test_case.expert_overlay->owner_order ==
                               owner_order &&
                           test_case.expert_overlay->movement ==
                               ModelParityExpertMovement::Dynamic &&
                           test_case.prefill_graph.mode == prefill_mode;
                });
            if (found == cases.end())
            {
                throw std::logic_error(
                    "Generated Qwen3.5 MoE 35B Dynamic cell is missing");
            }
            return *found;
        };

        for (const auto owner_order : {
                 RoutedExpertOwnerOrder::Ordinal,
                 RoutedExpertOwnerOrder::Random})
        {
            const auto &speedup = find_dynamic(
                owner_order, ModelParityPrefillGraphMode::Standard);
            const bool selected =
                owner_order == RoutedExpertOwnerOrder::Random;
            EXPECT_EQ(
                speedup.requiresObservedConvergenceSpeedup(), selected);
            EXPECT_EQ(
                speedup.dynamic_rebalance.window_size,
                selected
                    ? qwen35moe::kQwen35MoEConvergenceHistogramWindowRows
                    : 256);
            EXPECT_EQ(
                speedup.dynamic_rebalance.max_window_size,
                selected
                    ? qwen35moe::kQwen35MoEConvergenceHistogramWindowRows
                    : 256);
            if (selected)
            {
                EXPECT_GT(
                    static_cast<std::uint64_t>(
                        speedup.dynamic_rebalance.window_size),
                    qwen35moe::qwen35MoEConvergenceTimingCohortRoutedRows());
            }

            const auto &movement = find_dynamic(
                owner_order,
                ModelParityPrefillGraphMode::SegmentedCaptured);
            EXPECT_FALSE(movement.requiresObservedConvergenceSpeedup());
            EXPECT_EQ(movement.dynamic_rebalance.window_size, 256);
            EXPECT_EQ(movement.dynamic_rebalance.max_window_size, 256);
        }
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
                test_case.prefix_restore_geometry,
                ModelParityPrefixRestoreGeometry::AuthenticatedPromptBlock);
            EXPECT_EQ(
                test_case.testName().find("Prefix"),
                std::string::npos)
                << "mandatory prefix restore must not be encoded as a case axis";
        }
    }

    TEST(ModelParityDefinition, PrefillGraphProfilesUseTheCentralCrossProduct)
    {
        auto definition = makeDefinition(
            makeOverlayTopology(), kModelParityRequiredMaximumMTPDepth);
        definition.features.mtp = ModelParityAxisProfile::Standard;
        definition.features.prefill_graph = {
            ModelParityPrefillGraphPolicy{},
            ModelParityPrefillGraphPolicy{
                .mode = ModelParityPrefillGraphMode::SegmentedCaptured,
                .captured_rows = 4,
            },
        };

        const auto cases = expandModelParityDefinition(definition);

        ASSERT_EQ(cases.size(), 48u);
        EXPECT_EQ(
            std::count_if(
                cases.begin(), cases.end(),
                [](const ModelParityCase &test_case)
                { return test_case.prefill_graph.isSegmentedCaptured(); }),
            24);
        const auto segmented = std::find_if(
            cases.begin(), cases.end(),
            [](const ModelParityCase &test_case)
            { return test_case.prefill_graph.isSegmentedCaptured(); });
        ASSERT_NE(segmented, cases.end());
        EXPECT_EQ(segmented->prefill_graph.captured_rows, 4);
        EXPECT_NE(
            segmented->testName().find("SegmentedPrefillRows4"),
            std::string::npos);
    }

    TEST(ModelParityDefinition, Qwen35GraphNativeHasNoPrivatePolicyCells)
    {
        const auto generic_multi_device =
            qwen35moe::qwen35MoEMultiDeviceThresholds();
        EXPECT_NE(
            std::find(
                generic_multi_device.excluded_stages.begin(),
                generic_multi_device.excluded_stages.end(),
                "MOE_EXPERT_OUTPUT"),
            generic_multi_device.excluded_stages.end())
            << "ordinary tensor-parallel snapshots remain branch-local";

        std::vector<ModelParityCase> cases;
        for (const auto &spec :
             qwen35moe::qwen35MoE35BOverlayTopologySpecs())
        {
            auto topology_cases = expandModelParityDefinition(
                qwen35moe::qwen35MoE35BGraphNativeParityDefinition(spec));
            cases.insert(
                cases.end(),
                std::make_move_iterator(topology_cases.begin()),
                std::make_move_iterator(topology_cases.end()));
        }

        ASSERT_EQ(cases.size(), 20u);
        std::set<std::string> names;
        std::set<std::string> topology_ids;
        std::size_t segmented = 0;
        std::size_t static_ordinal = 0;
        std::size_t dynamic_ordinal = 0;
        std::size_t static_random = 0;
        std::size_t dynamic_random = 0;
        for (const auto &test_case : cases)
        {
            EXPECT_EQ(
                std::find(
                    test_case.thresholds.excluded_stages.begin(),
                    test_case.thresholds.excluded_stages.end(),
                    "MOE_EXPERT_OUTPUT"),
                test_case.thresholds.excluded_stages.end())
                << "ExpertOverlay must compare its canonical sparse-return sum";
            names.insert(test_case.testName());
            topology_ids.insert(test_case.topology.test_id);
            segmented +=
                test_case.prefill_graph.isSegmentedCaptured() ? 1u : 0u;
            ASSERT_TRUE(test_case.expert_overlay.has_value());
            const bool dynamic =
                test_case.expert_overlay->movement ==
                ModelParityExpertMovement::Dynamic;
            const bool random =
                test_case.expert_overlay->owner_order ==
                RoutedExpertOwnerOrder::Random;
            if (dynamic && random)
                ++dynamic_random;
            else if (dynamic)
                ++dynamic_ordinal;
            else if (random)
                ++static_random;
            else
                ++static_ordinal;
        }
        EXPECT_EQ(names.size(), cases.size());
        EXPECT_EQ(topology_ids.size(), 4u);
        EXPECT_EQ(segmented, 4u);
        EXPECT_EQ(static_ordinal, 5u);
        EXPECT_EQ(dynamic_ordinal, 5u);
        EXPECT_EQ(static_random, 5u);
        EXPECT_EQ(dynamic_random, 5u);
    }

    TEST(ModelParityDefinition, UnimplementedActivationAxesFailBeforeExpansion)
    {
        for (const auto activation : {
                 ActivationPrecision::FP16, ActivationPrecision::BF16,
                 ActivationPrecision::Q8_1, ActivationPrecision::Q16_1,
                 ActivationPrecision::Hybrid, ActivationPrecision::HybridQ16,
                 ActivationPrecision::TQ4, ActivationPrecision::TQ8,
                 ActivationPrecision::AQ8})
        {
            SCOPED_TRACE(activationPrecisionToString(activation));
            auto definition = makeDefinition(makeSingleDeviceTopology());
            definition.precisions.activation = {activation};
            EXPECT_THROW((void)expandModelParityDefinition(definition), std::invalid_argument);
        }
    }

    TEST(ModelParityDefinition, PrecisionAxesUseTheSameCentralCrossProduct)
    {
        auto definition = makeDefinition(makeSingleDeviceTopology());
        definition.precisions.activation = {
            ActivationPrecision::FP32,
        };
        definition.precisions.kv_cache = {
            KVCachePrecision::FP32,
            KVCachePrecision::FP16,
            KVCachePrecision::Q8_1,
        };
        auto q8_thresholds = definition.thresholds;
        q8_thresholds.kl_threshold = 0.123f;
        q8_thresholds.mtp_kl_threshold = 0.234f;
        definition.precisions.threshold_overrides = {
            {
                .activation = ActivationPrecision::FP32,
                .kv_cache = KVCachePrecision::Q8_1,
                .thresholds = q8_thresholds,
            },
        };

        const auto cases = expandModelParityDefinition(definition);

        ASSERT_EQ(cases.size(), 3u);
        std::set<std::pair<int, int>> precision_pairs;
        for (const auto &test_case : cases)
        {
            precision_pairs.emplace(
                static_cast<int>(test_case.activation_precision),
                static_cast<int>(test_case.kv_cache_precision));
        }
        EXPECT_EQ(precision_pairs.size(), 3u);
        const auto overridden = std::find_if(
            cases.begin(), cases.end(),
            [](const ModelParityCase &test_case)
            {
                return test_case.activation_precision ==
                           ActivationPrecision::FP32 &&
                       test_case.kv_cache_precision ==
                           KVCachePrecision::Q8_1;
            });
        ASSERT_NE(overridden, cases.end());
        EXPECT_FLOAT_EQ(overridden->thresholds.kl_threshold, 0.123f);
        ASSERT_TRUE(overridden->thresholds.mtp_kl_threshold.has_value());
        EXPECT_FLOAT_EQ(
            *overridden->thresholds.mtp_kl_threshold,
            0.234f);
    }

    TEST(ModelParityDefinition, MTPKLOverrideChangesOnlyItsTypedPolicy)
    {
        auto definition = makeDefinition(
            makeOverlayTopology(), kModelParityRequiredMaximumMTPDepth);
        definition.thresholds.mtp_kl_threshold = 0.05f;
        definition.features.mtp = ModelParityAxisProfile::Standard;
        definition.features.mtp_kl_threshold_overrides = {
            {
                .policy = ModelParityMTP::Depth15,
                .maximum_kl_divergence = 0.06f,
            },
        };

        const auto cases = expandModelParityDefinition(definition);
        const auto &depth_three = findCase(
            cases,
            RoutedExpertOwnerOrder::Ordinal,
            ModelParityExpertMovement::Static,
            ModelParityMTP::Depth3);
        const auto &depth_fifteen = findCase(
            cases,
            RoutedExpertOwnerOrder::Ordinal,
            ModelParityExpertMovement::Static,
            ModelParityMTP::Depth15);
        const auto &dynamic_depth = findCase(
            cases,
            RoutedExpertOwnerOrder::Ordinal,
            ModelParityExpertMovement::Static,
            ModelParityMTP::DynamicDepth);

        ASSERT_TRUE(depth_three.thresholds.mtp_kl_threshold.has_value());
        ASSERT_TRUE(depth_fifteen.thresholds.mtp_kl_threshold.has_value());
        ASSERT_TRUE(dynamic_depth.thresholds.mtp_kl_threshold.has_value());
        EXPECT_FLOAT_EQ(*depth_three.thresholds.mtp_kl_threshold, 0.05f);
        EXPECT_FLOAT_EQ(*depth_fifteen.thresholds.mtp_kl_threshold, 0.06f);
        EXPECT_FLOAT_EQ(*dynamic_depth.thresholds.mtp_kl_threshold, 0.05f);
        EXPECT_FLOAT_EQ(
            *depth_fifteen.toTestConfig().thresholds.mtp_kl_threshold,
            0.06f);
    }

    /**
     * @brief Recursive/grouped LM-head proofs consume the canonical typed K.
     *
     * The reference argmax is rank four in production, while production's
     * argmax is absent from the reference top four. This catches both a
     * hardcoded top-three gate and an accidental reverse-direction gate that
     * the canonical PyTorch-top1-in-production-topK policy does not declare.
     */
    TEST(ModelParityDefinition, LMHeadContainmentUsesConfiguredReferenceDirection)
    {
        constexpr std::array<float, 6> production{
            10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f};
        constexpr std::array<float, 6> reference{
            1.0f, 8.0f, 9.0f, 10.0f, 7.0f, 6.0f};

        const ReferenceTopKContainmentResult top3 =
            evaluateReferenceTopKContainment(
                production.data(),
                reference.data(),
                production.size(),
                production.size(),
                3);
        EXPECT_TRUE(top3.enabled);
        EXPECT_FALSE(top3.passed);
        EXPECT_FLOAT_EQ(top3.reference_top1_in_production, 0.0f);
        EXPECT_FLOAT_EQ(top3.production_top1_in_reference, 0.0f);

        const ReferenceTopKContainmentResult top4 =
            evaluateReferenceTopKContainment(
                production.data(),
                reference.data(),
                production.size(),
                production.size(),
                4);
        EXPECT_TRUE(top4.enabled);
        EXPECT_TRUE(top4.passed);
        EXPECT_FLOAT_EQ(top4.reference_top1_in_production, 1.0f);
        EXPECT_FLOAT_EQ(top4.production_top1_in_reference, 0.0f);

        const ReferenceTopKContainmentResult disabled =
            evaluateReferenceTopKContainment(
                production.data(),
                reference.data(),
                production.size(),
                production.size(),
                0);
        EXPECT_FALSE(disabled.enabled);
        EXPECT_TRUE(disabled.passed);
    }

    TEST(ModelParityDefinition,
         RecursiveMTPCosineOverrideChangesOnlyDeepTypedPolicies)
    {
        auto definition = makeDefinition(
            makeOverlayTopology(), kModelParityRequiredMaximumMTPDepth);
        definition.features.mtp = ModelParityAxisProfile::Standard;
        definition.features
            .mtp_recursive_aggregate_cosine_threshold_overrides = {
            {
                .policy = ModelParityMTP::Depth15,
                .minimum_cosine_similarity = 0.98f,
            },
            {
                .policy = ModelParityMTP::DynamicDepth,
                .minimum_cosine_similarity = 0.981f,
            },
        };

        const auto cases = expandModelParityDefinition(definition);
        const auto &depth_three = findCase(
            cases,
            RoutedExpertOwnerOrder::Ordinal,
            ModelParityExpertMovement::Static,
            ModelParityMTP::Depth3);
        const auto &depth_fifteen = findCase(
            cases,
            RoutedExpertOwnerOrder::Ordinal,
            ModelParityExpertMovement::Static,
            ModelParityMTP::Depth15);
        const auto &dynamic_depth = findCase(
            cases,
            RoutedExpertOwnerOrder::Ordinal,
            ModelParityExpertMovement::Static,
            ModelParityMTP::DynamicDepth);

        EXPECT_FALSE(
            depth_three.mtp_recursive_aggregate_cosine_floor.has_value());
        ASSERT_TRUE(
            depth_fifteen.mtp_recursive_aggregate_cosine_floor.has_value());
        ASSERT_TRUE(
            dynamic_depth.mtp_recursive_aggregate_cosine_floor.has_value());
        EXPECT_FLOAT_EQ(
            *depth_fifteen.mtp_recursive_aggregate_cosine_floor,
            0.98f);
        EXPECT_FLOAT_EQ(
            *dynamic_depth.mtp_recursive_aggregate_cosine_floor,
            0.981f);
        ASSERT_TRUE(
            depth_fifteen.toTestConfig()
                .mtp_recursive_aggregate_cosine_floor.has_value());
        EXPECT_FLOAT_EQ(
            *depth_fifteen.toTestConfig()
                 .mtp_recursive_aggregate_cosine_floor,
            0.98f);
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
        EXPECT_EQ(
            legacy.mtp_expected_graph_capacity,
            kModelParityRequiredMaximumMTPDepth)
            << "The adapter must expose physical capacity independently of active MTP execution";
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
        EXPECT_EQ(
            runtime.mtp.graph_capacity_draft_tokens,
            kModelParityRequiredMaximumMTPDepth)
            << "The execution-off control must retain the same setup envelope as enabled cells";
        EXPECT_TRUE(runtime.prefix_cache.enabled);
        EXPECT_EQ(
            runtime.prefix_cache.storage_mode,
            PrefixCacheStorageMode::Tiered);
        EXPECT_EQ(
            test_case.prefix_restore_geometry,
            ModelParityPrefixRestoreGeometry::AuthenticatedPromptBlock);
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
        definition.precisions.activation = {ActivationPrecision::FP32};
        definition.precisions.kv_cache = {KVCachePrecision::Q8_1};
        const auto cases = expandModelParityDefinition(definition);
        const auto &test_case = findCase(
            cases,
            RoutedExpertOwnerOrder::Ordinal,
            ModelParityExpertMovement::Dynamic,
            ModelParityMTP::DynamicDepth);

        OrchestrationConfig runtime;
        test_case.applyRuntimePolicy(runtime);

        EXPECT_EQ(runtime.activation_precision, "fp32");
        EXPECT_EQ(runtime.kv_cache_precision, "q8_1");
        EXPECT_TRUE(runtime.prefix_cache.enabled);
        EXPECT_EQ(runtime.prefix_cache.storage_mode, PrefixCacheStorageMode::Tiered);
        EXPECT_EQ(
            test_case.prefix_restore_geometry,
            ModelParityPrefixRestoreGeometry::AuthenticatedPromptBlock);
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
        EXPECT_EQ(runtime.moe_rebalance.migration_transfer_slots, 3u);
        EXPECT_EQ(runtime.moe_rebalance.device_maintenance_slack_tokens, 0);
        EXPECT_EQ(
            runtime.moe_rebalance.device_min_maintenance_period_tokens,
            kModelParityRequiredMaximumMTPDepth + 1);
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
        definition.dynamic_rebalance.economic_movement
            .device_initial_maintenance_period_tokens = -1;

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

        auto disabled_mtp_override =
            makeDefinition(makeSingleDeviceTopology());
        disabled_mtp_override.features.mtp_kl_threshold_overrides = {
            {
                .policy = ModelParityMTP::Depth15,
                .maximum_kl_divergence = 0.06f,
            },
        };
        EXPECT_THROW(
            static_cast<void>(expandModelParityDefinition(
                disabled_mtp_override)),
            std::invalid_argument);

        auto shallow_recursive_cosine_override = makeDefinition(
            makeOverlayTopology(),
            kModelParityRequiredMaximumMTPDepth);
        shallow_recursive_cosine_override.features.mtp =
            ModelParityAxisProfile::Standard;
        shallow_recursive_cosine_override.features
            .mtp_recursive_aggregate_cosine_threshold_overrides = {
            {
                .policy = ModelParityMTP::Depth3,
                .minimum_cosine_similarity = 0.98f,
            },
        };
        EXPECT_THROW(
            static_cast<void>(expandModelParityDefinition(
                shallow_recursive_cosine_override)),
            std::invalid_argument);

        auto clipped_adaptive_mtp = makeDefinition(
            makeOverlayTopology(),
            kModelParityRequiredMaximumMTPDepth);
        clipped_adaptive_mtp.features.mtp =
            ModelParityAxisProfile::Standard;
        clipped_adaptive_mtp.dynamic_rebalance.economic_movement
            .device_min_maintenance_period_tokens =
            kModelParityRequiredMaximumMTPDepth;
        EXPECT_THROW(
            static_cast<void>(expandModelParityDefinition(
                clipped_adaptive_mtp)),
            std::invalid_argument);

        auto empty_precision = makeDefinition(makeSingleDeviceTopology());
        empty_precision.precisions.activation.clear();
        EXPECT_THROW(
            static_cast<void>(expandModelParityDefinition(empty_precision)),
            std::invalid_argument);

        auto empty_prefill = makeDefinition(makeSingleDeviceTopology());
        empty_prefill.features.prefill_graph.clear();
        EXPECT_THROW(
            static_cast<void>(expandModelParityDefinition(empty_prefill)),
            std::invalid_argument);

        auto invalid_segmented = makeDefinition(makeSingleDeviceTopology());
        invalid_segmented.features.prefill_graph = {
            ModelParityPrefillGraphPolicy{
                .mode = ModelParityPrefillGraphMode::SegmentedCaptured,
                .captured_rows = 0,
            },
        };
        EXPECT_THROW(
            static_cast<void>(expandModelParityDefinition(invalid_segmented)),
            std::invalid_argument);

        auto duplicate_prefill = makeDefinition(makeSingleDeviceTopology());
        duplicate_prefill.features.prefill_graph = {
            ModelParityPrefillGraphPolicy{},
            ModelParityPrefillGraphPolicy{},
        };
        EXPECT_THROW(
            static_cast<void>(expandModelParityDefinition(duplicate_prefill)),
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

        auto missing_overlay_layers = makeDefinition(makeOverlayTopology());
        missing_overlay_layers.model.transformer_layers = 0;
        EXPECT_THROW(
            static_cast<void>(expandModelParityDefinition(
                missing_overlay_layers)),
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

    TEST(ModelParityDefinition, E2ETagsSelectExistingCellsWithoutChangingMatrix)
    {
        auto definition = makeDefinition(makeOverlayTopology(), 15);
        definition.features.mtp = ModelParityAxisProfile::Standard;
        const auto before = expandModelParityDefinition(definition);
        definition.e2e_certifiable = {{
            .mtp = ModelParityMTP::Depth2,
            .owner_order = RoutedExpertOwnerOrder::Random,
            .movement = ModelParityExpertMovement::Dynamic,
            .profile = {.readiness_timeout_seconds = 180},
        }};
        const auto after = expandModelParityDefinition(definition);
        ASSERT_EQ(before.size(), after.size());
        int tags = 0;
        for (std::size_t i = 0; i < after.size(); ++i)
        {
            EXPECT_EQ(before[i].testName(), after[i].testName());
            if (after[i].e2e_certification)
            {
                ++tags;
                EXPECT_EQ(after[i].mtp, ModelParityMTP::Depth2);
                EXPECT_EQ(after[i].expert_overlay->movement, ModelParityExpertMovement::Dynamic);
                EXPECT_EQ(after[i].expert_overlay->owner_order, RoutedExpertOwnerOrder::Random);
                EXPECT_EQ(after[i].e2e_certification->readiness_timeout_seconds, 180);
                std::ostringstream exported;
                PrintTo(after[i], &exported);
                EXPECT_NE(exported.str().find("\"readiness_timeout_seconds\":180"), std::string::npos);
            }
        }
        EXPECT_EQ(tags, 1);
    }

    /** @brief Discovery carries the matrix's obligation, not CLI default inference. */
    TEST(ModelParityDefinition, E2EExportsTypedMovementObligations)
    {
        const auto expect_export = [](const ModelParityDefinition &definition,
                                      const std::string &expected)
        {
            std::size_t tagged = 0;
            for (const auto &cell : expandModelParityDefinition(definition))
            {
                if (!cell.e2e_certification) continue;
                ++tagged;
                std::ostringstream exported;
                PrintTo(cell, &exported);
                EXPECT_NE(exported.str().find(
                              "\"movement_evidence\":\"" + expected + "\""),
                          std::string::npos);
            }
            EXPECT_GT(tagged, 0u);
        };
        auto single = makeDefinition(makeSingleDeviceTopology());
        single.e2e_certifiable = {{}};
        expect_export(single, "not_applicable");
        for (const auto movement : {ModelParityExpertMovement::Static,
                                    ModelParityExpertMovement::Dynamic})
        {
            auto overlay = makeDefinition(makeOverlayTopology());
            overlay.e2e_certifiable = {{
                .owner_order = RoutedExpertOwnerOrder::Ordinal,
                .movement = movement,
            }};
            expect_export(overlay, movement == ModelParityExpertMovement::Static
                                       ? "forbidden" : "required");
        }
        EXPECT_THROW(modelParityE2EMovementEvidenceName(
                         static_cast<ModelParityMovementEvidence>(255)),
                     std::invalid_argument);
    }

    TEST(ModelParityDefinition, E2ETagsRejectUnmatchedOverlappingAndInvalidProfiles)
    {
        auto definition = makeDefinition(makeSingleDeviceTopology());
        definition.e2e_certifiable = {{.mtp = ModelParityMTP::Depth15}};
        EXPECT_THROW((void)expandModelParityDefinition(definition), std::invalid_argument);
        definition.e2e_certifiable = {{}, {}};
        EXPECT_THROW((void)expandModelParityDefinition(definition), std::invalid_argument);
        definition.e2e_certifiable = {{.owner_order = RoutedExpertOwnerOrder::Ordinal}};
        EXPECT_THROW((void)expandModelParityDefinition(definition), std::invalid_argument);
        definition.e2e_certifiable = {{.profile = {.context_length = 100}}};
        EXPECT_THROW((void)expandModelParityDefinition(definition), std::invalid_argument);
        for (int timeout : {0, -1})
        {
            definition.e2e_certifiable = {{.profile = {.readiness_timeout_seconds = timeout}}};
            EXPECT_THROW((void)expandModelParityDefinition(definition), std::invalid_argument);
        }
        definition.e2e_certifiable = {{}};
        const auto cells = expandModelParityDefinition(definition);
        ASSERT_TRUE(cells.front().e2e_certification);
        EXPECT_EQ(cells.front().e2e_certification->readiness_timeout_seconds, 60);
    }

    /** @return Public CLI round trip of one exported certification cell. */
    OrchestrationConfig parseE2EArguments(const ModelParityCase &cell)
    {
        auto args = modelParityE2EServerArguments(cell);
        args.insert(args.begin(), "llaminar2");
        std::vector<char *> argv;
        for (auto &arg : args) argv.push_back(arg.data());
        return OrchestrationConfigParser{}.parseArgs(static_cast<int>(argv.size()), argv.data());
    }

    TEST(ModelParityDefinition, E2EThinkingModesAreTypedAndExportedWithoutFilenameInference)
    {
        auto definition = qwen36::qwen36MoECPU2NodeTPParityDefinition();
        for (const auto mode : {ModelParityE2EThinkingModes::NonThinkingOnly,
                                ModelParityE2EThinkingModes::ThinkingAndNonThinking})
        {
            definition.e2e_certifiable.front().profile.thinking_modes = mode;
            for (const auto &cell : expandModelParityDefinition(definition))
            {
                if (!cell.e2e_certification) continue;
                std::ostringstream exported;
                PrintTo(cell, &exported);
                EXPECT_NE(exported.str().find(std::string("\"thinking_modes\":\"") +
                    modelParityE2EThinkingModesName(mode) + "\""), std::string::npos);
            }
        }
        definition.e2e_certifiable.front().profile.thinking_modes =
            static_cast<ModelParityE2EThinkingModes>(-1);
        EXPECT_THROW(expandModelParityDefinition(definition), std::invalid_argument);
    }

    TEST(ModelParityDefinition, OrnithCertificationInheritsTopologyButNotReferenceIdentity)
    {
        std::vector<ModelParityDefinition> originals;
        for (const auto address : {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0)})
            originals.push_back(qwen36::qwen36MoEParityDefinition(
                qwen36::qwen36SingleDeviceTopology(address.toString(), address),
                "/references/qwen36", qwen36::qwen36MoESingleDeviceThresholds()));
        // Test identifiers are separate from the CLI's punctuation-bearing addresses.
        originals[0].topology.test_id = "CPU0";
        originals[1].topology.test_id = "CUDA0";
        originals[2].topology.test_id = "ROCm0";
        originals.push_back(qwen36::qwen36MoECPU2NodeTPParityDefinition());
        for (const auto &topology : {qwen36::qwen36MoECuda2ExpertOverlayTopology(),
                                    qwen36::qwen36MoERocm2ExpertOverlayTopology()})
            originals.push_back(qwen36::qwen36MoEParityDefinition(
                topology, "/references/qwen36", qwen36::qwen36MoEExpertOverlayThresholds()));

        const auto definitions = qwen36::withOrnith15CertificationModels(originals);
        ASSERT_EQ(definitions.size(), originals.size() + 4);
        std::set<std::string> names;
        std::set<std::string> references;
        std::size_t added_cells = 0, added_tags = 0;
        for (std::size_t i = 0; i < originals.size(); ++i)
        {
            const auto before = expandModelParityDefinition(originals[i]);
            const auto after = expandModelParityDefinition(definitions[i]);
            ASSERT_EQ(before.size(), after.size());
            for (std::size_t j = 0; j < before.size(); ++j)
            {
                EXPECT_EQ(before[j].testName(), after[j].testName());
                EXPECT_EQ(before[j].model.model_path, after[j].model.model_path);
                EXPECT_TRUE(names.insert(after[j].testName()).second);
            }
        }
        for (std::size_t i = originals.size(); i < definitions.size(); ++i)
        {
            const auto &variant = definitions[i];
            const auto source = std::find_if(originals.begin(), originals.end(),
                [&](const auto &d) { return d.topology.test_id == variant.topology.test_id; });
            ASSERT_NE(source, originals.end());
            EXPECT_NE(variant.model.reference_directory, source->model.reference_directory);
            EXPECT_TRUE(references.insert(variant.model.reference_directory).second);
            auto expected_definition = *source;
            if (source->topology.expert_overlay_plan)
                expected_definition.e2e_certifiable = {{
                    .mtp = ModelParityMTP::DynamicDepth,
                    .owner_order = RoutedExpertOwnerOrder::Ordinal,
                    .movement = ModelParityExpertMovement::Dynamic,
                }};
            const auto before = expandModelParityDefinition(expected_definition);
            const auto after = expandModelParityDefinition(variant);
            ASSERT_EQ(before.size(), after.size());
            added_cells += after.size();
            for (std::size_t j = 0; j < after.size(); ++j)
            {
                EXPECT_TRUE(names.insert(after[j].testName()).second);
                EXPECT_EQ(after[j].testName().substr(variant.model.test_id.size()),
                          before[j].testName().substr(source->model.test_id.size()));
                ASSERT_EQ(before[j].e2e_certification.has_value(), after[j].e2e_certification.has_value());
                if (!after[j].e2e_certification) continue;
                ++added_tags;
                auto expected = modelParityE2EServerArguments(before[j]);
                std::replace(expected.begin(), expected.end(), source->model.model_path, variant.model.model_path);
                EXPECT_EQ(expected, modelParityE2EServerArguments(after[j]));
                EXPECT_EQ(after[j].e2e_certification->readiness_timeout_seconds,
                          before[j].e2e_certification->readiness_timeout_seconds);
                EXPECT_EQ(after[j].e2e_certification->thinking_modes,
                          ModelParityE2EThinkingModes::ThinkingAndNonThinking);
                const auto config = parseE2EArguments(after[j]);
                EXPECT_EQ(config.mtp.depth_policy.max_depth, 15);
                EXPECT_TRUE(config.prefix_cache.enabled);
            }
        }
        EXPECT_EQ(added_cells, 78u);
        EXPECT_EQ(added_tags, 4u);
    }

    TEST(ModelParityDefinition, CPUNodeCertificationRetainsFullMatrixAndOneAuthority)
    {
        const auto definition = qwen36::qwen36MoECPU2NodeTPParityDefinition();
        const auto cells = expandModelParityDefinition(definition);
        ASSERT_EQ(cells.size(), 24u);
        std::size_t tagged = 0;
        for (const auto &cell : cells)
        {
            ASSERT_EQ(cell.topology.mpi_ranks, 2);
            if (!cell.e2e_certification) continue;
            ++tagged;
            EXPECT_EQ(cell.mtp, ModelParityMTP::DynamicDepth);
            ASSERT_TRUE(cell.expert_overlay);
            EXPECT_EQ(cell.expert_overlay->movement, ModelParityExpertMovement::Dynamic);
            EXPECT_EQ(cell.expert_overlay->owner_order, RoutedExpertOwnerOrder::Ordinal);
            const auto config = parseE2EArguments(cell);
            ASSERT_TRUE(config.moe_routed_expert_plan);
            const auto &plan = *config.moe_routed_expert_plan;
            ASSERT_EQ(plan.domains.size(), 1u);
            ASSERT_EQ(plan.routed_tiers.size(), 1u);
            EXPECT_EQ(plan.topology, RoutedExpertPlacementTopology::SingleDomain);
            EXPECT_EQ(plan.domains[0].scope, ExecutionDomainScope::NODE_LOCAL);
            EXPECT_EQ(plan.domains[0].participants,
                      (std::vector<GlobalDeviceAddress>{GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)}));
            EXPECT_EQ(config.moe_rebalance.mode, MoERebalanceRuntimeMode::Dynamic);
            EXPECT_EQ(config.mtp.depth_policy.max_depth, 15);
            EXPECT_TRUE(config.prefix_cache.enabled);
        }
        EXPECT_EQ(tagged, 1u);
    }

    TEST(ModelParityDefinition, InitialSingleGPUCertificationSelectsOnlyAdaptiveMTP)
    {
        for (const auto address : {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0)})
        {
            const auto topology = qwen36::qwen36SingleDeviceTopology("single", address);
            for (const auto &definition : {
                     qwen38::qwen38DenseParityDefinition(topology, "/references/dense"),
                     qwen36::qwen36MoEParityDefinition(topology, "/references/moe", {})})
            {
                const auto cases = expandModelParityDefinition(definition);
                ASSERT_EQ(cases.size(), 6u);
                std::size_t tagged = 0;
                for (const auto &cell : cases)
                {
                    if (!cell.e2e_certification) continue;
                    ++tagged;
                    EXPECT_EQ(cell.mtp, ModelParityMTP::DynamicDepth);
                    EXPECT_EQ(cell.retained_mtp_draft_capacity, 15);
                    EXPECT_FALSE(cell.expert_overlay.has_value());
                }
                EXPECT_EQ(tagged, address.isCPU() ? 0u : 1u);
            }
        }
        for (const auto &topology : {qwen36::qwen36MoECuda2ExpertOverlayTopology(),
                                    qwen36::qwen36MoERocm2ExpertOverlayTopology()})
            EXPECT_TRUE(qwen36::qwen36MoEParityDefinition(topology, "/references/moe", {}).e2e_certifiable.empty());
    }

    TEST(ModelParityDefinition, Qwen38E2ERoundTripsAllMTPPoliciesOnEachBackend)
    {
        for (const auto address : {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0)})
        {
            auto topology = makeSingleDeviceTopology();
            topology.participants = {{address, 0}};
            auto definition = qwen38::qwen38DenseParityDefinition(topology, "/references/qwen38");
            definition.e2e_certifiable.clear();
            for (const auto mtp : kCanonicalModelParityMTPPolicies)
                definition.e2e_certifiable.push_back({.mtp = mtp});
            for (const auto &cell : expandModelParityDefinition(definition))
            {
                SCOPED_TRACE(cell.testName());
                const auto expected = cell.makeOrchestrationConfig(cell.model.model_path, 0);
                const auto actual = parseE2EArguments(cell);
                ASSERT_TRUE(actual.device_for_this_rank);
                EXPECT_EQ(actual.device_for_this_rank, expected.device_for_this_rank);
                EXPECT_EQ(actual.activation_precision, "fp32");
                EXPECT_EQ(actual.kv_cache_precision, "fp16");
                EXPECT_EQ(actual.mtp.enabled, cell.mtpEnabled());
                EXPECT_EQ(actual.mtp.graph_capacity_draft_tokens,
                          expected.mtp.graph_capacity_draft_tokens);
                if (cell.mtpEnabled())
                    EXPECT_EQ(actual.mtp.draft_tokens, expected.mtp.draft_tokens);
                EXPECT_EQ(actual.mtp.depth_policy.mode, expected.mtp.depth_policy.mode);
                EXPECT_TRUE(actual.prefix_cache.enabled);
                EXPECT_EQ(actual.prefix_cache.storage_mode, PrefixCacheStorageMode::Tiered);
                EXPECT_EQ(cell.model.transformer_layers, 64);
                EXPECT_NE(cell.model.model_path.find("Qwen3.8-27B-IQ4_XS"), std::string::npos);
            }
        }
    }

    TEST(ModelParityDefinition, E2EOverlayRoundTripPreservesBothPlacementAxesAndEconomics)
    {
        for (auto topology : {qwen36::qwen36MoECuda2ExpertOverlayTopology(), qwen36::qwen36MoERocm2ExpertOverlayTopology()})
        {
            auto definition = qwen36::qwen36MoEParityDefinition(topology, "/references/moe", {});
            definition.e2e_certifiable.clear();
            for (const auto &policy : kCanonicalModelParityExpertOverlayPolicies)
                definition.e2e_certifiable.push_back({
                    .owner_order = policy.owner_order, .movement = policy.movement});
            for (const auto &cell : expandModelParityDefinition(definition))
            {
                if (!cell.e2e_certification) continue;
                SCOPED_TRACE(cell.testName());
                const auto expected = cell.makeOrchestrationConfig(cell.model.model_path, 0);
                const auto actual = parseE2EArguments(cell);
                ASSERT_TRUE(actual.moe_routed_expert_plan);
                const auto &plan = *actual.moe_routed_expert_plan;
                EXPECT_EQ(plan.owner_order, cell.expert_overlay->owner_order);
                EXPECT_EQ(plan.residency_policy, expected.moe_routed_expert_plan->residency_policy);
                EXPECT_EQ(plan.domains.size(), topology.expert_overlay_plan->domains.size());
                EXPECT_EQ(plan.domains[0].participants, topology.expert_overlay_plan->domains[0].participants);
                EXPECT_EQ(plan.domains[0].backend, topology.expert_overlay_plan->domains[0].backend);
                EXPECT_EQ(actual.moe_rebalance.mode, expected.moe_rebalance.mode);
                EXPECT_EQ(actual.moe_rebalance.window_size, expected.moe_rebalance.window_size);
                EXPECT_EQ(actual.moe_rebalance.migration_transfer_slots, expected.moe_rebalance.migration_transfer_slots);
                EXPECT_EQ(actual.moe_rebalance.dynamic_min_improvement_per_mille, expected.moe_rebalance.dynamic_min_improvement_per_mille);
            }
        }
    }

} // namespace llaminar2::test::parity
