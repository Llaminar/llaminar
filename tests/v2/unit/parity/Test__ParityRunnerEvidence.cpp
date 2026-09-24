/**
 * @file Test__ParityRunnerEvidence.cpp
 * @brief Device-free regression for evidence across retained runner cells.
 *
 * Reproduce the depth-one/depth-two aggregate boundary with real collector
 * records. Construction must remain authentic, but an idle later cell must
 * never inherit graph replay, MTP, movement, or prefix execution from its peer.
 */
#include "integration/parity/ParityRunnerEvidence.h"
#include "utils/ProductionParityEvidence.h"
#include <gtest/gtest.h>
#include <array>
#include <cstdlib>
#include <optional>
#include <string>

namespace llaminar2::test::parity
{
    /** Isolate collector configuration and records without initializing devices. */
    class ParityRunnerEvidenceTest : public ::testing::Test
    {
    protected:
        /** Enable only this test's bounded evidence and remember the environment. */
        void SetUp() override
        {
            for (std::size_t i = 0; i < names_.size(); ++i)
            {
                if (const char *value = std::getenv(names_[i]))
                    previous_[i] = value;
                setenv(names_[i], i == 0 ? "1" : "", 1);
            }
            PerfStatsCollector::reset();
        }

        /** Drop this cell's records and restore configuration for later tests. */
        void TearDown() override
        {
            for (std::size_t i = 0; i < names_.size(); ++i)
            {
                if (previous_[i])
                    setenv(names_[i], previous_[i]->c_str(), 1);
                else
                    unsetenv(names_[i]);
            }
            PerfStatsCollector::reset();
        }

        /** Publish the construction families checked by the overlay fixture. */
        void publishSetup(const std::string &device)
        {
            for (const auto &family : setup_)
                PerfStatsCollector::addCounter(
                    family.domain, family.name, 17.0, "model_setup", device,
                    {{"identity", "same_retained_runner"}});
        }

        /** Publish request work that must be independently redone in every cell. */
        void publishRuntime(const std::string &device)
        {
            for (const auto &family : runtime_)
                PerfStatsCollector::addCounter(
                    family.domain, family.name, 9.0, "decode", device);
            PerfStatsCollector::recordTimingNs(
                "forward_graph", "full_graph_replay_graph", 100, "decode", device);
            PerfStatsCollector::recordOrderedSequenceStep(
                "forward_graph", "moe_overlay_collective_transaction_sequence",
                {1, 2, 3, 4}, "prefill", device);
        }

        /** Independent producer/consumer contract, not the retention implementation. */
        const std::array<PerfStatsCollector::RecordFamily, 14> setup_{{
            {"mtp", "sidecar_terminal_hidden_read_only_contracts"},
            {"memory", "moe_serial_local_expert_buffer_arena_allocations"},
            {"memory", "moe_serial_local_expert_buffer_arena_bytes"},
            {"moe_placement", "routed_expert_weight_selection"},
            {"moe_overlay_controller", "static_no_movement_transactions"},
            {"moe_overlay_residency", "static_no_movement_checks"},
            {"moe_overlay_controller", "follower_runtime_tables_materialized"},
            {"moe_overlay_participant_graph", "materialized_mapped_follower_families"},
            {"moe_overlay_participant_graph", "materialized_rank_local_canonical_ticket_consumers"},
            {"moe_overlay_participant_graph", "rank_local_canonical_ticket_consumer_launches"},
            {"forward_graph", "heterogeneous_ticket_transactions"},
            {"forward_graph", "segmented_plan_segments"},
            {"forward_graph", "segmented_graph_capture_segments"},
            {"forward_graph", "retained_parent_transaction_zero_launches"},
        }};
        /** Same domains, including similar names: broad/prefix retention is unsafe. */
        const std::array<PerfStatsCollector::RecordFamily, 13> runtime_{{
            {"mtp", "sidecar_terminal_hidden_read_leases"},
            {"mtp", "draft_steps"},
            {"mtp", "device_generation_verifier_calls"},
            {"mtp", "sidecar_terminal_hidden_read_only_contracts_unrelated"},
            {"memory", "allocations"},
            {"moe_overlay_participant_graph", "ticket_selected_graphs"},
            {"moe_overlay_controller", "dynamic_movement_transactions"},
            {"moe_overlay_residency", "committed_expert_migrations"},
            {"moe_overlay_residency", "committed_waves"},
            {"moe_overlay_residency", "dynamic_no_movement_checks"},
            {"forward_graph", "retained_parent_replays"},
            {"prefix_cache", "restores"},
            {"transfer", "packed_weight_bytes"},
        }};

    private:
        const std::array<const char *, 2> names_{
            "LLAMINAR_PERF_STATS_JSON", "LLAMINAR_PERF_STATS_FILTER"};
        std::array<std::optional<std::string>, 2> previous_;
    };

    TEST_F(ParityRunnerEvidenceTest, RetainedSetupSurvivesButEveryCellMustExecute)
    {
        for (const char *device : {"CPU:0", "CUDA:0", "ROCm:0"})
        {
            resetParityRunnerEvidence(ParityRunnerEvidenceLifetime::FreshRunner);
            publishSetup(device);
            const auto setup_json = PerfStatsCollector::jsonString();
            // Repeated compatible MTP policies must not accumulate prior work
            // or inflate once-per-runner allocation/materialization counts.
            for (int cell = 0; cell < 20; ++cell)
            {
                publishRuntime(device);
                EXPECT_TRUE(productionParityHasRetainedParentDecodeReplay(PerfStatsCollector::snapshot()));
                resetParityRunnerEvidence(ParityRunnerEvidenceLifetime::RetainedRunner);
                EXPECT_EQ(PerfStatsCollector::snapshot().size(), setup_.size());
                EXPECT_EQ(PerfStatsCollector::jsonString(), setup_json);
                EXPECT_FALSE(productionParityHasRetainedParentDecodeReplay(PerfStatsCollector::snapshot()));
            }
        }
    }

    TEST_F(ParityRunnerEvidenceTest, FreshRunnerCannotInheritAnyPriorEvidence)
    {
        publishSetup("CUDA:0");
        publishRuntime("CUDA:0");
        resetParityRunnerEvidence(ParityRunnerEvidenceLifetime::FreshRunner);
        EXPECT_TRUE(PerfStatsCollector::snapshot().empty());
    }

    TEST_F(ParityRunnerEvidenceTest, RetentionCannotManufactureMissingConstruction)
    {
        publishRuntime("ROCm:0");
        resetParityRunnerEvidence(ParityRunnerEvidenceLifetime::RetainedRunner);
        EXPECT_TRUE(PerfStatsCollector::snapshot().empty());
        // Fresh work after reset is still collected: retaining setup must not
        // disable observability in the next request window.
        PerfStatsCollector::addCounter("mtp", "sidecar_terminal_hidden_read_leases");
        const auto records = PerfStatsCollector::snapshot();
        ASSERT_EQ(records.size(), 1u);
        EXPECT_EQ(records.front().count, 1u);
    }

    TEST_F(ParityRunnerEvidenceTest, InvalidLifetimeFailsClosed)
    {
        EXPECT_THROW(resetParityRunnerEvidence(
            static_cast<ParityRunnerEvidenceLifetime>(73)), std::logic_error);
    }

    TEST_F(ParityRunnerEvidenceTest, FullGraphCertificateRequiresFreshReplayAfterRetainedCapture)
    {
        for (const char *device : {"CUDA:0", "ROCm:0"})
        {
            resetParityRunnerEvidence(ParityRunnerEvidenceLifetime::FreshRunner);
            // Mirror the production transaction plan and retained parent's
            // one-time zero launch, not a made-up already-certified result.
            PerfStatsCollector::addCounter("forward_graph", "segmented_plan_segments", 4, "setup", device);
            PerfStatsCollector::addCounter("forward_graph", "segmented_graph_capture_segments", 3, "setup", device);
            PerfStatsCollector::addCounter("forward_graph", "retained_parent_transaction_zero_launches", 1, "decode", device);
            const auto certify = []
            {
                return certifyProductionParityGraphExecution(collectProductionParityEvidence(
                    PerfStatsCollector::snapshot(), true,
                    ProductionParityExecutionTopology::HeterogeneousAccelerator,
                    ProductionParityGraphContract::HeterogeneousCoordinatorSegmented,
                    true, 1.0, 4500.0));
            };
            for (int cell = 0; cell < 20; ++cell)
            {
                resetParityRunnerEvidence(ParityRunnerEvidenceLifetime::RetainedRunner);
                EXPECT_EQ(certify(), ProductionParityGraphCertification::MissingHeterogeneousDecodeReplay);
                PerfStatsCollector::addCounter("forward_graph", "retained_parent_replays", 1, "decode", device);
                EXPECT_EQ(certify(), ProductionParityGraphCertification::Certified);
            }
            resetParityRunnerEvidence(ParityRunnerEvidenceLifetime::FreshRunner);
            EXPECT_EQ(certify(), ProductionParityGraphCertification::MissingHeterogeneousSegmentPlan);
        }
    }
}
