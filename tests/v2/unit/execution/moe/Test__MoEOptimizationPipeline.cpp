/**
 * @file Test__MoEOptimizationPipeline.cpp
 * @brief Device-free adversarial publication checks for four/eight-device PP(TP).
 *
 * Real rank composition consumes injected receipt-only children. The native
 * journal writer and archive supply actual production evidence values; no
 * model, GPU context, transport, or payload allocation participates. Colliding
 * transaction IDs deliberately exercise independent stage namespaces.
 */
#include <gtest/gtest.h>
#include "execution/moe/MoEOptimizationPipeline.h"
#include "execution/moe/NativeMoEMovementArchive.h"
#include "execution/moe/NativeMoEMovementPerfStats.h"
#include "execution/local_execution/orchestrators/RankOrchestrator.h"
#include "app/modes/PrefixMovementJson.h"
#include "app/modes/MoEMovementTransportJson.h"
#include "mocks/MockModelContext.h"
#include "mocks/MockLocalTPContext.h"
#include <array>
#include <limits>
#include <cstdlib>

namespace llaminar2
{
    namespace
    {
        /** @brief Borrowed publication views require a retained value owner. */
        template<class T> concept RvalueParticipantView = requires(T value) { std::move(value).participants(); };
        /** @brief A composed snapshot cannot lend a view from a temporary either. */
        template<class T> concept RvalueStageView = requires(T value) { std::move(value).entries(); };
        static_assert(!RvalueParticipantView<MoEOptimizationStageIdentity>);
        static_assert(!RvalueStageView<MoEOptimizationStages<MoEOptimizationStatus>>);

        /** @brief Restore the process-local collector policy after a wire fixture. */
        class PipelineTelemetryScope final
        {
        public:
            /** @brief Enable metadata capture without opening any payload or device. */
            PipelineTelemetryScope()
            {
                for (const char *name : {"LLAMINAR_PERF_STATS_JSON", "LLAMINAR_PERF_STATS_FILTER"})
                    prior_.push_back(std::getenv(name) ? std::optional<std::string>{std::getenv(name)} : std::nullopt);
                ::setenv("LLAMINAR_PERF_STATS_JSON", "1", 1);
                ::unsetenv("LLAMINAR_PERF_STATS_FILTER");
                PerfStatsCollector::reset();
            }
            /** @brief Restore both collector environment values and retire captured metadata. */
            ~PipelineTelemetryScope()
            {
                std::size_t index = 0;
                for (const char *name : {"LLAMINAR_PERF_STATS_JSON", "LLAMINAR_PERF_STATS_FILTER"})
                {
                    const auto &prior = prior_[index++];
                    if (prior) ::setenv(name, prior->c_str(), 1);
                    else ::unsetenv(name);
                }
                PerfStatsCollector::reset();
            }
        private:
            std::vector<std::optional<std::string>> prior_;
        };

        /** @brief Receipt-only runner; all device execution entry points stay unused. */
        class PipelineReceiptRunner final : public IInferenceRunner
        {
        public:
            MoEOptimizationStatus status;
            MoEOptimizationMovementLedger ledger;
            DeviceId device = DeviceId::cpu();
            mutable int observations = 0;
            uint64_t lookup_epoch = 0;
            uint64_t live_epoch = 0;
            mutable int epoch_observations = 0;
            int harvests = 0;
            /** @return False; this fixture cannot execute inference. */
            bool forward(const int *, int) override { return false; }
            /** @return No tensor payload is needed for publication. */
            const float *logits() const override { return nullptr; }
            /** @return Minimal metadata-only vocabulary. */
            int vocab_size() const override { return 1; }
            /** @brief Request reset preserves already-completed receipts. */
            void clear_cache() override {}
            /** @return No request has advanced. */
            int get_position() const override { return 0; }
            /** @return Declarative runner kind, never a capture certificate. */
            ExecutionPath executionPath() const override { return ExecutionPath::GRAPH; }
            /** @return Injected prepared-owner metadata; no graph is captured here. */
            ServingGraphPreparationKind servingGraphPreparationKind() const noexcept override
            { return ServingGraphPreparationKind::NativeDeviceExecutableFamily; }
            /** @return Stable model-independent fixture identity. */
            const char *architecture() const override { return "pipeline-receipts"; }
            /** @return Metadata only; no backend is opened. */
            DeviceId primaryDeviceId() const override { return device; }
            /** @return One counted immutable observation for freshness assertions. */
            MoEOptimizationStatus moeOptimizationStatus() const override { ++observations; return status; }
            /** @return Already-completed owner history. */
            MoEOptimizationMovementLedger moeOptimizationMovementLedger() const override { return ledger; }
            /** @return Passive scalar from this injected participant's epoch authority. */
            uint64_t moeRuntimeMovementEpoch() const override { ++epoch_observations; return live_epoch; }
            /** @return Admission metadata only; no cache allocation or payload restore. */
            PrefixLookupResult lookupPrefix(const std::vector<int32_t> &) override
            {
                PrefixLookupResult result;
                result.supported = result.cache_enabled = true;
                result.block_size = 16;
                result.placement_epochs = PrefixPlacementEpochSpan::at(lookup_epoch);
                result.requires_terminal_hidden = result.requires_terminal_logits = false;
                return result;
            }
            /** @return Whether the actual nested harvest retained this participant's admission. */
            bool harvestPrefix(const PrefixLookupResult &admission,
                const std::vector<int32_t> &, int) override
            {
                ++harvests;
                return admission.placement_epochs == PrefixPlacementEpochSpan::at(lookup_epoch);
            }
        };

        /** @brief One standard-window scope for each declared topology width. */
        class MoEOptimizationPipelineTest : public ::testing::TestWithParam<int>
        {
        protected:
            /** @brief Complete nested PP/TP tree with borrowed pointers owned by the returned rank. */
            struct PrefixRanks
            {
                std::unique_ptr<RankOrchestrator> rank;
                std::vector<PipelineReceiptRunner *> participants;
            };

            /** @return Actual orchestration code around device-free epoch authorities. */
            PrefixRanks prefixRanks(bool reverse = false) const
            {
                PrefixRanks result;
                RankOrchestrator::Config parent;
                parent.mode = RankOrchestrator::ParallelismMode::TP_PP;
                parent.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
                std::vector<std::unique_ptr<IInferenceRunner>> stages;
                for (std::size_t i = 0; i < 2; ++i)
                {
                    const auto scope = identity(i, reverse);
                    auto child = parent;
                    child.mode = RankOrchestrator::ParallelismMode::TP;
                    child.devices = scope.participants();
                    std::vector<std::unique_ptr<IInferenceRunner>> participants;
                    for (const auto &device : child.devices)
                    {
                        auto participant = std::make_unique<PipelineReceiptRunner>();
                        participant->device = device.toLocalDeviceId();
                        participant->lookup_epoch = participant->live_epoch = i == 0 ? 100 : 1;
                        result.participants.push_back(participant.get());
                        participants.push_back(std::move(participant));
                    }
                    auto collective = std::make_unique<test::MockLocalTPContext>();
                    collective->setDevices(child.devices);
                    stages.push_back(RankOrchestrator::createForTest(test::MockModelContext::createMinimal(),
                        std::move(participants), std::move(collective), child));
                    parent.pp_stages.push_back({.first_layer = scope.firstLayer(), .last_layer = scope.mainLastLayer(),
                        .has_embedding = i == 0, .has_lm_head = i == 1, .stage_devices = scope.participants(),
                        .tp_backend = child.devices.front().toLocalDeviceId().is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL});
                }
                result.rank = RankOrchestrator::createForTestWithPipelineStages(
                    test::MockModelContext::createMinimal(), std::move(stages), parent);
                return result;
            }

            /** @return Authored vendor order and exact stage-global interval. */
            MoEOptimizationStageIdentity identity(std::size_t stage, bool reverse = false, bool mtp = false) const
            {
                std::vector<GlobalDeviceAddress> devices;
                for (int i = 0; i < GetParam() / 2; ++i)
                    devices.push_back((stage == 0) != reverse ? GlobalDeviceAddress::rocm(i) : GlobalDeviceAddress::cuda(i));
                return {stage, static_cast<int>(stage * 2), static_cast<int>(stage * 2 + 2),
                    static_cast<int>(stage * 2 + 2 + (stage == 1 && mtp)), std::move(devices), stage == 1};
            }

            /** @return A leaf whose counters intentionally match its sibling's. */
            static MoEOptimizationStatus status()
            {
                return {.authority = MoEOptimizationAuthority::Device,
                    .state = MoEOptimizationLifecycleState::Active,
                    .activity = MoEOptimizationActivityState::DeviceOwned,
                    .published_movement_waves = 1,
                    .completed_movement = {.transactions = 1, .commands = 2, .physical_bytes = 6000, .same_priority_moves = 2},
                    .completed_decision_windows = 3,
                    .last_decision = MoEOptimizationDecisionReceipt{.transaction = (1ULL << 53) + 1, .candidate_epoch = (1ULL << 53) + 2}};
            }

            /** @return Native writer/archive proof with compact rows translated once. */
            static MoEOptimizationMovementLedger ledger(const MoEOptimizationStageIdentity &scope)
            {
                NativeMoEMovementArchiveConfig config{.workspace_generation = 7,
                    .layers = static_cast<std::uint32_t>(scope.routedLastLayer() - scope.firstLayer()),
                    .first_model_layer = static_cast<std::uint32_t>(scope.firstLayer()),
                    .experts = 8, .wave_capacity = 2, .edge_capacity = 4};
                for (const auto &device : scope.participants())
                    config.participants.push_back({device.toLocalDeviceId(), 0, 17});
                NativeMoEMovementArchive archive(std::move(config));
                DeviceMoERebalanceMovementJournalState state{};
                std::array<DeviceMoERebalanceMovementWave, 2> waves{};
                std::array<DeviceMoERebalanceMovementEdge, 4> edges{};
                const DeviceMoERebalanceLoadSpreadProof proof{
                    .accepted_spread_improvement = 40, .pre_wave_spread = 100, .post_wave_spread = 60,
                    .pre_wave_total = 200, .post_wave_total = 200,
                    .pre_participant_spread = 0, .post_participant_spread = 20,
                    .pre_participant_total = 200, .post_participant_total = 200,
                    .requested_payload_slots = 1, .minimum_improvement_per_slot = 40,
                    .maximum_post_spread_per_mille = 300, .ownership_swap_accepts = 1};
                DeviceMoERebalanceMovementJournalView view{&state, waves.data(), edges.data(), 2, 4};
                const auto append = prepareDeviceMoEMovementJournalAppend(view, (1ULL << 53) + 1, 1, 2, proof, 6000);
                if (append.disposition != DeviceMoEMovementJournalDisposition::Record)
                    throw std::logic_error("Fixture native writer rejected its bounded publication");
                const auto last = static_cast<std::uint32_t>(scope.participants().size() - 1);
                const auto row = static_cast<std::uint32_t>(scope.routedLastLayer() - scope.firstLayer() - 1);
                edges[0] = {row, 3, 0, last, 90, 8192};
                edges[1] = {row, 4, last, 0, 10, 8192};
                if (!commitDeviceMoEMovementJournalAppend(view, append))
                    throw std::logic_error("Fixture native writer failed publication");
                archive.observe(1, 7, state, std::span(waves).first(1), std::span(edges).first(2));
                return archive.ledger();
            }

            /** @return Checked composition with independently mutable test inputs. */
            MoEOptimizationStatus compose(MoEOptimizationStatus first, MoEOptimizationStatus second) const
            {
                return composeMoEOptimizationStatus(MoEOptimizationStages<MoEOptimizationStatus>::seal(
                    {{identity(0), std::move(first)}, {identity(1), std::move(second)}}));
            }
        };

        TEST_P(MoEOptimizationPipelineTest, IndependentPublishersPreserveCollidingTransactionsAndGlobalLayers)
        {
            for (bool reverse : {false, true})
                for (bool mtp : {false, true})
                {
                    auto first = identity(0, reverse, mtp), second = identity(1, reverse, mtp);
                    auto combined = composeMoEOptimizationMovementLedger(MoEOptimizationStages<MoEOptimizationMovementLedger>::seal(
                        {{first, ledger(first)}, {second, ledger(second)}}));
                    EXPECT_TRUE(combined.complete());
                    EXPECT_TRUE(combined.edges.empty());
                    const auto wire = moeMovementTransportJson(combined);
                    ASSERT_EQ(wire.at("schema"), 2);
                    ASSERT_EQ(wire.at("stages").size(), 2);
                    for (int stage = 0; stage < 2; ++stage)
                    {
                        const auto &transport = wire["stages"][stage]["transport"];
                        EXPECT_EQ(transport["movement"]["edges"][0]["layer"], stage * 2 + 1 + (stage == 1 && mtp));
                        EXPECT_EQ(transport["device_publications"][0]["transaction"], (1ULL << 53) + 1);
                        EXPECT_EQ(transport["device_publications"][0]["physical_payload_bytes"], 6000);
                        EXPECT_EQ(wire["stages"][stage]["identity"]["participants"].size(), GetParam() / 2);
                    }
                    const auto observed = compose(status(), status());
                    EXPECT_EQ(observed.completed_movement.transactions, 2);
                    EXPECT_EQ(observed.completed_movement.physical_bytes, 12000);
                    EXPECT_EQ(observed.completed_decision_windows, 6);
                    EXPECT_FALSE(observed.last_decision);
                    EXPECT_FALSE(observed.demand_window.valid());
                    EXPECT_FALSE(observed.quiescentBetweenWaves());
                    const auto status_wire = moeOptimizationStatusJson(observed);
                    EXPECT_TRUE(status_wire["demand_window"].is_null());
                    EXPECT_EQ(status_wire["stages"][1]["status"]["last_decision"]["candidate_epoch"], (1ULL << 53) + 2);
                    if (reverse && mtp)
                    {
                        PipelineTelemetryScope telemetry;
                        std::vector<MoEOptimizationStages<MoEOptimizationMovementTopology>::Entry> geometry;
                        for (const auto &stage : combined.stages.entries())
                        {
                            recordNativeMoECompletedMovement(stage.value.device_publications.front(), stage.value.edges,
                                stage.identity.participants().front().toLocalDeviceId().toString(),
                                stage.identity.firstLayer(), stage.identity.routedLastLayer() - stage.identity.firstLayer());
                            geometry.push_back({stage.identity, {.authority = MoEOptimizationAuthority::Device,
                                .axes = MoEOptimizationMovementAxes::ParticipantPlacement}});
                        }
                        auto records = nlohmann::json::parse(PerfStatsCollector::jsonString()).at("records");
                        for (auto &record : records) record["rank"] = 0;
                        auto terminal = wire;
                        terminal["rank"] = 0;
                        const auto topology = composeMoEOptimizationMovementTopology(
                            MoEOptimizationStages<MoEOptimizationMovementTopology>::seal(std::move(geometry)));
                        RecordProperty("pipeline_movement_fixture", nlohmann::json{
                            {"records", records}, {"terminal_movement", nlohmann::json::array({terminal})},
                            {"http_movement", moeMovementLedgerJson(combined)},
                            {"http_topology", moeMovementTopologyJson(topology)}}.dump());
                    }
                }
        }

        TEST_P(MoEOptimizationPipelineTest, OneStageCannotHideRegressionWithSiblingProgress)
        {
            auto first = status(), second = status();
            first.demand_window = {.generation = 5, .collected_routed_rows = 3, .capacity_routed_rows = 10};
            second.demand_window = {.generation = 8, .collected_routed_rows = 4, .capacity_routed_rows = 10};
            const auto before = compose(first, second).progressStamp();
            EXPECT_EQ(before.relationTo(before), MoEOptimizationProgressRelation::Unchanged);
            --first.completed_movement.transactions;
            second.completed_movement.transactions += 2;
            EXPECT_EQ(compose(first, second).progressStamp().relationTo(before), MoEOptimizationProgressRelation::Regressed);
            first = status(); first.demand_window = {.generation = 5, .collected_routed_rows = 2, .capacity_routed_rows = 10};
            EXPECT_EQ(compose(first, second).progressStamp().relationTo(before), MoEOptimizationProgressRelation::Regressed);
            ++first.demand_window.generation; first.demand_window.collected_routed_rows = 0;
            EXPECT_EQ(compose(first, second).progressStamp().relationTo(before), MoEOptimizationProgressRelation::Advanced);
            first = status(); second = status();
            const auto decision_before = compose(first, second).progressStamp();
            ++first.completed_decision_windows;
            EXPECT_EQ(compose(first, second).progressStamp().relationTo(decision_before), MoEOptimizationProgressRelation::Advanced);
            --first.completed_decision_windows;
            const auto decision_base = compose(first, second).progressStamp();
            --first.completed_decision_windows;
            second.completed_decision_windows += 2;
            EXPECT_EQ(compose(first, second).progressStamp().relationTo(decision_base), MoEOptimizationProgressRelation::Regressed);
            auto rebound = composeMoEOptimizationStatus(MoEOptimizationStages<MoEOptimizationStatus>::seal(
                {{identity(0, true), first}, {identity(1, true), second}}));
            EXPECT_EQ(rebound.progressStamp().relationTo(decision_before), MoEOptimizationProgressRelation::Regressed);
            EXPECT_EQ(first.progressStamp().relationTo(decision_before), MoEOptimizationProgressRelation::Regressed);
        }

        TEST_P(MoEOptimizationPipelineTest, EveryStageMustRetainItsTerminalTransportReceipts)
        {
            auto first = ledger(identity(0)), second = ledger(identity(1));
            second.device_publications.clear();
            auto missing = composeMoEOptimizationMovementLedger(MoEOptimizationStages<MoEOptimizationMovementLedger>::seal(
                {{identity(0), first}, {identity(1), second}}));
            EXPECT_THROW(moeMovementTransportJson(missing), std::invalid_argument);
            second = ledger(identity(1)); second.discarded_device_publications = 1;
            auto truncated = composeMoEOptimizationMovementLedger(MoEOptimizationStages<MoEOptimizationMovementLedger>::seal(
                {{identity(0), first}, {identity(1), second}}));
            EXPECT_FALSE(truncated.complete());
            EXPECT_THROW(moeMovementLedgerJson(truncated), std::invalid_argument);
            EXPECT_THROW(moeMovementTransportJson(truncated), std::invalid_argument);
            second = ledger(identity(1)); second.edges[0].layer = 0;
            EXPECT_THROW(composeMoEOptimizationMovementLedger(MoEOptimizationStages<MoEOptimizationMovementLedger>::seal(
                {{identity(0), first}, {identity(1), second}})), std::invalid_argument);
            second = ledger(identity(1)); second.edges[0].destination_device = DeviceId::rocm(0);
            EXPECT_THROW(composeMoEOptimizationMovementLedger(MoEOptimizationStages<MoEOptimizationMovementLedger>::seal(
                {{identity(0), first}, {identity(1), second}})), std::invalid_argument);
        }

        TEST_P(MoEOptimizationPipelineTest, ScopeCannotBeOmittedReorderedOverlappedOrNested)
        {
            using Stages = MoEOptimizationStages<MoEOptimizationStatus>;
            EXPECT_THROW(Stages::seal({}), std::invalid_argument);
            EXPECT_THROW(Stages::seal({{identity(0), status()}}), std::invalid_argument);
            EXPECT_THROW(Stages::seal({{identity(1), status()}}), std::invalid_argument);
            EXPECT_THROW(Stages::seal({{identity(1), status()}, {identity(0), status()}}), std::invalid_argument);
            EXPECT_THROW(Stages::seal({{identity(0), status()}, {identity(0), status()}}), std::invalid_argument);
            const auto first_scope = identity(0), second_scope = identity(1);
            auto overlap = MoEOptimizationStageIdentity{1, 1, 4, 4, second_scope.participants(), true};
            EXPECT_THROW(Stages::seal({{identity(0), status()}, {overlap, status()}}), std::invalid_argument);
            auto premature_mtp = MoEOptimizationStageIdentity{0, 0, 2, 3, first_scope.participants(), false};
            EXPECT_THROW(Stages::seal({{premature_mtp, status()}, {identity(1), status()}}), std::invalid_argument);
            const auto nested = compose(status(), status());
            EXPECT_THROW(Stages::seal({{identity(0), nested}, {identity(1), status()}}), std::invalid_argument);
            auto invalid = status(); invalid.authority = MoEOptimizationAuthority::None;
            EXPECT_THROW(compose(invalid, status()), std::invalid_argument);
            invalid.authority = MoEOptimizationAuthority::Pipeline;
            EXPECT_THROW(compose(invalid, status()), std::invalid_argument);
        }

        TEST_P(MoEOptimizationPipelineTest, DemandWindowsAndDrainRemainPerOwner)
        {
            auto first = status(), second = status();
            first.authority = MoEOptimizationAuthority::Host;
            first.activity = second.activity = MoEOptimizationActivityState::CollectingDemand;
            first.demand_window = {.generation = 2, .collected_routed_rows = 7, .capacity_routed_rows = 10};
            second.demand_window = {.generation = 9, .collected_routed_rows = 1, .capacity_routed_rows = 10};
            EXPECT_TRUE(compose(first, second).canBeginExclusiveCohort(2));
            EXPECT_FALSE(compose(first, second).canBeginExclusiveCohort(3));
            first.published_progress_generation = 1;
            EXPECT_FALSE(compose(first, second).quiescentBetweenWaves());
            auto terminal = drainedMoEOptimizationStatus(compose(first, second));
            EXPECT_TRUE(terminal.drained());
            for (const auto &stage : terminal.stages.entries())
                EXPECT_TRUE(stage.value.drained());
            first.state = MoEOptimizationLifecycleState::Failed;
            EXPECT_TRUE(drainedMoEOptimizationStatus(compose(first, second)).failed());
            first = status(); first.state = MoEOptimizationLifecycleState::LearningEconomy;
            EXPECT_TRUE(compose(first, second).learning());
        }

        TEST_P(MoEOptimizationPipelineTest, CounterOverflowCannotPublishWrappedTotals)
        {
            for (int field = 0; field < 8; ++field)
            {
                auto first = status(), second = status();
                const std::array<std::uint64_t *, 8> fields{&first.published_movement_waves,
                    &first.completed_movement.transactions, &first.completed_movement.commands,
                    &first.completed_movement.physical_bytes, &first.completed_movement.promotions,
                    &first.completed_movement.demotions, &first.completed_movement.same_priority_moves,
                    &first.completed_decision_windows};
                *fields[field] = UINT64_MAX;
                second.completed_movement.promotions = second.completed_movement.demotions = 1;
                EXPECT_THROW(compose(first, second), std::overflow_error) << field;
            }
        }

        TEST_P(MoEOptimizationPipelineTest, NativeArchiveRejectsGlobalLayerOverflow)
        {
            NativeMoEMovementArchiveConfig config{.workspace_generation = 7, .layers = 2,
                .first_model_layer = static_cast<std::uint32_t>(INT32_MAX) - 2,
                .experts = 8, .wave_capacity = 2, .edge_capacity = 4};
            const auto scope = identity(1);
            for (const auto &device : scope.participants())
                config.participants.push_back({device.toLocalDeviceId(), 0, 0});
            EXPECT_NO_THROW((void)NativeMoEMovementArchive{config});
            ++config.first_model_layer;
            EXPECT_THROW((void)NativeMoEMovementArchive{config}, std::invalid_argument);
            config.first_model_layer = UINT32_MAX;
            EXPECT_THROW((void)NativeMoEMovementArchive{config}, std::invalid_argument);
        }

        TEST_P(MoEOptimizationPipelineTest, DisabledAndAbsentOwnersDoNotInventMovement)
        {
            auto disabled = MoEOptimizationStatus{.authority = MoEOptimizationAuthority::Device,
                .state = MoEOptimizationLifecycleState::MovementDisabled, .activity = MoEOptimizationActivityState::Dormant};
            auto orphan = disabled;
            orphan.authority = MoEOptimizationAuthority::Pipeline;
            EXPECT_THROW(moeOptimizationStatusJson(orphan), std::invalid_argument);
            auto observation = compose(disabled, disabled);
            EXPECT_EQ(observation.state, MoEOptimizationLifecycleState::MovementDisabled);
            EXPECT_EQ(observation.completed_movement.transactions, 0);
            EXPECT_FALSE(observation.active());
            EXPECT_FALSE(observation.canBeginExclusiveCohort(1));
            EXPECT_TRUE(drainedMoEOptimizationStatus(observation).drained());
            auto absent = compose({}, {});
            EXPECT_EQ(absent.state, MoEOptimizationLifecycleState::NotApplicable);
            EXPECT_FALSE(absent.active());
            EXPECT_EQ(absent.progressStamp().relationTo(absent.progressStamp()), MoEOptimizationProgressRelation::Unchanged);
        }

        TEST_P(MoEOptimizationPipelineTest, FrozenTopologyRetainsIndependentStageAxes)
        {
            auto topology = composeMoEOptimizationMovementTopology(MoEOptimizationStages<MoEOptimizationMovementTopology>::seal({
                {identity(0), {.authority = MoEOptimizationAuthority::Host, .axes = MoEOptimizationMovementAxes::TierResidency}},
                {identity(1), {.authority = MoEOptimizationAuthority::Device, .axes = MoEOptimizationMovementAxes::ParticipantPlacement}}}));
            EXPECT_TRUE(topology.valid());
            EXPECT_EQ(topology.axes, MoEOptimizationMovementAxes::Both);
            const auto json = moeMovementTopologyJson(topology);
            EXPECT_EQ(json["schema"], 2);
            EXPECT_EQ(json["stages"][0]["topology"]["authority"], "host");
            EXPECT_EQ(json["stages"][1]["topology"]["available_axes"], nlohmann::json::array({"participant_placement"}));
            topology.axes = MoEOptimizationMovementAxes::TierResidency;
            EXPECT_FALSE(topology.valid());
            EXPECT_THROW(moeMovementTopologyJson(topology), std::invalid_argument);
        }

        TEST_P(MoEOptimizationPipelineTest, ProductionRanksElectOneRootPerStageAndRejectDuplicateTPRoots)
        {
            for (bool reverse : {false, true})
            {
                RankOrchestrator::Config parent;
                parent.mode = RankOrchestrator::ParallelismMode::TP_PP;
                parent.prefix_cache.enabled = false;
                parent.prefix_cache.storage_mode = PrefixCacheStorageMode::Disabled;
                std::vector<std::unique_ptr<IInferenceRunner>> stages;
                std::vector<PipelineReceiptRunner *> observations;
                for (std::size_t i = 0; i < 2; ++i)
                {
                    auto scope = identity(i, reverse);
                    RankOrchestrator::Config child = parent;
                    child.mode = RankOrchestrator::ParallelismMode::TP;
                    child.devices = scope.participants();
                    std::vector<std::unique_ptr<IInferenceRunner>> participants;
                    for (std::size_t j = 0; j < child.devices.size(); ++j)
                    {
                        auto participant = std::make_unique<PipelineReceiptRunner>();
                        participant->device = child.devices[j].toLocalDeviceId();
                        if (j + 1 == child.devices.size())
                        {
                            participant->status = status();
                            participant->ledger = ledger(scope);
                        }
                        observations.push_back(participant.get());
                        participants.push_back(std::move(participant));
                    }
                    auto collective = std::make_unique<test::MockLocalTPContext>();
                    collective->setDevices(child.devices);
                    stages.push_back(RankOrchestrator::createForTest(test::MockModelContext::createMinimal(),
                        std::move(participants), std::move(collective), child));
                    parent.pp_stages.push_back({.first_layer = scope.firstLayer(), .last_layer = scope.mainLastLayer(),
                        .has_embedding = i == 0, .has_lm_head = i == 1, .stage_devices = scope.participants(),
                        .tp_backend = child.devices.front().toLocalDeviceId().is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL});
                }
                auto rank = RankOrchestrator::createForTestWithPipelineStages(test::MockModelContext::createMinimal(), std::move(stages), parent);
                const auto value = rank->moeOptimizationStatus();
                EXPECT_EQ(value.completed_movement.transactions, 2);
                EXPECT_EQ(value.stages.entries().size(), 2);
                for (const auto *participant : observations)
                    EXPECT_EQ(participant->observations, 1);
                EXPECT_EQ(moeMovementTransportJson(rank->moeOptimizationMovementLedger())["stages"].size(), 2);
                observations.front()->status = status();
                EXPECT_THROW(rank->moeOptimizationStatus(), std::logic_error);
                EXPECT_THROW(rank->moeOptimizationMovementLedger(), std::logic_error);
                observations.front()->status = {};
                for (auto *participant : observations)
                {
                    participant->status = {};
                    participant->ledger = {};
                }
                EXPECT_EQ(rank->moeOptimizationStatus().authority, MoEOptimizationAuthority::None);
                EXPECT_TRUE(rank->moeOptimizationMovementLedger().stages.empty());
            }
        }

        TEST_P(MoEOptimizationPipelineTest, PrefixMovementComparesIndependentStageEpochs)
        {
            for (const bool reverse : {false, true})
            {
                auto fixture = prefixRanks(reverse);
                EXPECT_THROW(fixture.rank->prefixMovementStages(), std::logic_error);
                const auto admission = fixture.rank->lookupPrefix({1, 2, 3});
                // The old flattened comparison reports movement for 100/1 even
                // when neither owner changed. The real per-child lookups are retained.
                EXPECT_EQ(admission.placement_epochs, PrefixPlacementEpochSpan::covering(1, 100));
                ASSERT_TRUE(fixture.rank->harvestPrefix(admission, {1, 2, 3}, 3));
                const PrefixCacheRequestSummary stable{.movement_stages = fixture.rank->prefixMovementStages()};
                EXPECT_FALSE(stable.crossedMovementEpoch());
                EXPECT_FALSE(stable.movementPrecededAdmissionOf(stable));
                for (const auto *participant : fixture.participants)
                {
                    EXPECT_EQ(participant->epoch_observations, 1);
                    EXPECT_EQ(participant->harvests, 1);
                }

                fixture.participants.back()->live_epoch = 2;
                const PrefixCacheRequestSummary moved{.movement_stages = fixture.rank->prefixMovementStages()};
                EXPECT_TRUE(moved.crossedMovementEpoch());
                EXPECT_FALSE(stable.movementPrecededAdmissionOf(moved)); // The later lookup preceded movement.
                EXPECT_EQ(fixture.rank->moeRuntimeMovementEpoch(), 100); // A maximum alone hides the change.
                for (std::size_t i = GetParam() / 2; i < fixture.participants.size(); ++i)
                    fixture.participants[i]->lookup_epoch = fixture.participants[i]->live_epoch = 2;
                (void)fixture.rank->lookupPrefix({1, 2, 3, 4});
                const PrefixCacheRequestSummary next{.movement_stages = fixture.rank->prefixMovementStages()};
                EXPECT_FALSE(next.crossedMovementEpoch());
                EXPECT_TRUE(stable.movementPrecededAdmissionOf(next));
                EXPECT_TRUE(moved.movementPrecededAdmissionOf(next));
                const auto wire = prefixMovementStagesJson(next.movement_stages);
                EXPECT_EQ(wire["stages"][0]["epochs"]["completion_movement_epoch"], 100);
                EXPECT_EQ(wire["stages"][1]["epochs"]["completion_movement_epoch"], 2);
                RecordProperty(reverse ? "prefix_epochs_reverse" : "prefix_epochs_forward", wire.dump());
            }
        }

        TEST_P(MoEOptimizationPipelineTest, PrefixMovementRejectsStaleAndForeignScopes)
        {
            auto fixture = prefixRanks();
            (void)fixture.rank->lookupPrefix({1});
            const PrefixCacheRequestSummary stable{.movement_stages = fixture.rank->prefixMovementStages()};
            for (std::size_t i = GetParam() / 2; i < fixture.participants.size(); ++i)
                fixture.participants[i]->live_epoch = 0;
            EXPECT_THROW(fixture.rank->prefixMovementStages(), std::logic_error);
            auto foreign = prefixRanks(true);
            (void)foreign.rank->lookupPrefix({1});
            const PrefixCacheRequestSummary changed{.movement_stages = foreign.rank->prefixMovementStages()};
            EXPECT_THROW((void)stable.movementPrecededAdmissionOf(changed), std::logic_error);
            EXPECT_THROW((void)stable.movementPrecededAdmissionOf({}), std::logic_error);
            EXPECT_THROW((void)PrefixCacheRequestSummary{}.movementPrecededAdmissionOf(stable), std::logic_error);
        }

        INSTANTIATE_TEST_SUITE_P(DeviceCounts, MoEOptimizationPipelineTest, ::testing::Values(4, 8));
    }
}
