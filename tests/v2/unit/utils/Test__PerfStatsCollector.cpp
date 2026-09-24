#include "utils/PerfStatsCollector.h"
#include "utils/DebugEnv.h"
#include "utils/KernelProfiler.h"
#include "utils/KVCacheProfiler.h"
#include "utils/Logger.h"
#include "utils/WeightLoadingProfiler.h"
#include "utils/ProductionParityEvidence.h"
#include "kernels/common/SamplingMath.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test::parity;

namespace
{
    class ScopedEnv
    {
    public:
        ScopedEnv(const char *name, const char *value)
            : name_(name)
        {
            const char *old = std::getenv(name);
            if (old)
            {
                had_old_ = true;
                old_value_ = old;
            }
            if (value)
                setenv(name, value, 1);
            else
                unsetenv(name);
            mutableDebugEnv().reload();
            PerfStatsCollector::reloadConfigurationFromEnvironment();
        }

        ~ScopedEnv()
        {
            if (had_old_)
                setenv(name_.c_str(), old_value_.c_str(), 1);
            else
                unsetenv(name_.c_str());
            mutableDebugEnv().reload();
            PerfStatsCollector::reloadConfigurationFromEnvironment();
        }

    private:
        std::string name_;
        bool had_old_ = false;
        std::string old_value_;
    };

    class ScopedLoggerRank
    {
    public:
        explicit ScopedLoggerRank(int rank)
            : previous_rank_(Logger::getInstance().getRank())
        {
            Logger::getInstance().setRank(rank);
        }

        ~ScopedLoggerRank()
        {
            Logger::getInstance().setRank(previous_rank_);
        }

        ScopedLoggerRank(const ScopedLoggerRank &) = delete;
        ScopedLoggerRank &operator=(const ScopedLoggerRank &) = delete;

    private:
        int previous_rank_ = -1;
    };

    std::filesystem::path uniqueTempPath(const std::string &suffix)
    {
        const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
        return std::filesystem::temp_directory_path() /
               ("llaminar_perf_stats_test_" + std::to_string(ticks) + suffix);
    }

    std::string readFile(const std::filesystem::path &path)
    {
        std::ifstream in(path);
        std::stringstream buffer;
        buffer << in.rdbuf();
        return buffer.str();
    }

    /**
     * @brief Append one complete participant ledger for HIP ticket dispatch.
     *
     * The fixture intentionally uses three controller transactions: two
     * ticket-selected captured continuations followed by one terminal
     * submission. That mirrors the equality the production gate proves.
     */
    void appendCertifiedHostedTicketParticipant(
        std::vector<PerfStatRecord> &records,
        const std::string &device)
    {
        records.push_back(PerfStatRecord{
            .kind = PerfStatRecord::Kind::Counter,
            .domain = "mtp",
            .name = "device_generation_loop_graph_materializations",
            .device = device,
            .tags = {
                {"backend", "HIP"},
                {"execution",
                 "hosted_captured_transactions_with_ticket_only_dispatch"},
                {"fragments", "5"},
                {"conditional_fragments", "0"},
                {"ticket_conditioned_fragments", "1"}},
            .value = 1.0,
        });
        records.push_back(PerfStatRecord{
            .kind = PerfStatRecord::Kind::Counter,
            .domain = "mtp",
            .name = "device_generation_loop_graph_launches",
            .device = device,
            .tags = {
                {"backend", "HIP"},
                {"execution",
                 "hosted_ticket_selected_captured_transactions"},
                {"fragments", "5"},
                {"conditional_fragments", "0"},
                {"ticket_conditioned_fragments", "1"}},
            .value = 1.0,
        });
        records.push_back(PerfStatRecord{
            .kind = PerfStatRecord::Kind::Counter,
            .domain = "mtp",
            .name = "device_generation_dispatch_ticket_d2h_submissions",
            .device = device,
            .tags = {
                {"bytes",
                 std::to_string(
                     sampling_math::DeviceGenerationDispatchTicket::
                         kWireBytes)},
                {"abi_version",
                 std::to_string(
                     sampling_math::DeviceGenerationDispatchTicket::
                         kABIVersion)},
                {"word_count",
                 std::to_string(
                     sampling_math::DeviceGenerationDispatchTicket::
                         kWordCount)},
                {"authority", "immutable_scheduler_snapshot"},
                {"state_payload", "false"}},
            .value = 3.0,
        });
        for (int transaction = 1; transaction <= 3; ++transaction)
        {
            records.push_back(PerfStatRecord{
                .kind = PerfStatRecord::Kind::Counter,
                .domain = "mtp",
                .name = "device_generation_dispatch_tickets_observed",
                .device = device,
                .tags = {
                    {"transaction", std::to_string(transaction)},
                    {"next_depth", "2"},
                    {"complete", transaction == 3 ? "true" : "false"},
                    {"maintenance_due", "false"}},
                .value = 1.0,
            });
        }
        records.push_back(PerfStatRecord{
            .kind = PerfStatRecord::Kind::Counter,
            .domain = "mtp",
            .name = "hosted_device_generation_transaction_submissions",
            .device = device,
            .tags = {
                {"depth", "2"},
                {"fragments", "5"},
                {"dynamic_depth_source", "device_controller_ticket"}},
            .value = 2.0,
        });
        records.push_back(PerfStatRecord{
            .kind = PerfStatRecord::Kind::Counter,
            .domain = "mtp",
            .name = "hosted_device_generation_terminal_submissions",
            .device = device,
            .tags = {{"transactions", "3"}},
            .value = 1.0,
        });
        records.push_back(PerfStatRecord{
            .kind = PerfStatRecord::Kind::Counter,
            .domain = "mtp",
            .name = "device_generation_terminal_transactions",
            .device = device,
            .value = 3.0,
        });
        records.push_back(PerfStatRecord{
            .kind = PerfStatRecord::Kind::Counter,
            .domain = "mtp",
            .name =
                "device_generation_terminal_compact_outcome_reductions",
            .device = device,
            .tags = {
                {"authority", "device_generation_controller"},
                {"accounting_role", "captured_graph_replay_multiplier"},
                {"source", "captured_stochastic_compact_outcome"},
                {"execution", "host_scheduled_captured_transactions"}},
            .value = 3.0,
        });
        records.push_back(PerfStatRecord{
            .kind = PerfStatRecord::Kind::Counter,
            .domain = "mtp",
            .name = "device_generation_terminal_response_bridges",
            .device = device,
            .value = 1.0,
        });
    }

    /** @brief Build a complete mirrored two-participant HIP ticket fixture. */
    std::vector<PerfStatRecord> certifiedHostedTicketRecords()
    {
        std::vector<PerfStatRecord> records{
            PerfStatRecord{
                .kind = PerfStatRecord::Kind::Counter,
                .domain = "mtp",
                .name = "device_generation_execution_policy_selections",
                .device = "ROCm:0",
                .tags = {
                    {"policy", "host_scheduled_captured_transactions"},
                    {"topology", "fixed_depth"},
                    {"selection_boundary", "pre_first_draft"}},
                .value = 1.0,
            }};
        appendCertifiedHostedTicketParticipant(records, "ROCm:0");
        appendCertifiedHostedTicketParticipant(records, "ROCm:1");
        return records;
    }

    /**
     * @brief Build the complete ledger for a request terminal on ticket one.
     *
     * There is deliberately no continuation-submission record: no such graph
     * was selected.  Every other counter still proves the one captured
     * transaction from immutable ticket publication through terminal response.
     */
    std::vector<PerfStatRecord> certifiedTerminalOnlyHostedTicketRecords()
    {
        auto records = certifiedHostedTicketRecords();
        std::erase_if(
            records,
            [](const PerfStatRecord &record)
            {
                return record.name ==
                           "hosted_device_generation_transaction_submissions" ||
                       record.name ==
                           "device_generation_dispatch_tickets_observed";
            });

        for (PerfStatRecord &record : records)
        {
            if (record.name ==
                    "device_generation_dispatch_ticket_d2h_submissions" ||
                record.name == "device_generation_terminal_transactions" ||
                record.name ==
                    "device_generation_terminal_compact_outcome_reductions")
            {
                record.value = 1.0;
            }
            else if (record.name ==
                     "hosted_device_generation_terminal_submissions")
            {
                record.tags["transactions"] = "1";
            }
        }

        for (const std::string device : {"ROCm:0", "ROCm:1"})
        {
            records.push_back(PerfStatRecord{
                .kind = PerfStatRecord::Kind::Counter,
                .domain = "mtp",
                .name = "device_generation_dispatch_tickets_observed",
                .device = device,
                .tags = {
                    {"transaction", "1"},
                    {"next_depth", "2"},
                    {"complete", "true"},
                    {"maintenance_due", "false"}},
                .value = 1.0,
            });
        }
        return records;
    }
}

TEST(Test__PerfStatsCollector, AggregatesCountersAndTimers)
{
    ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    PerfStatsCollector::addCounter("mtp", "draft_steps", 1.0, "decode");
    PerfStatsCollector::addCounter("mtp", "draft_steps", 2.0, "decode");
    PerfStatsCollector::recordTimingNs("mtp", "sidecar_forward", 1000, "decode", "rocm:0");
    PerfStatsCollector::recordTimingNs("mtp", "sidecar_forward", 3000, "decode", "rocm:0");

    const auto records = PerfStatsCollector::snapshot();
    ASSERT_EQ(records.size(), 2u);

    const auto counter_it = std::find_if(records.begin(), records.end(), [](const auto &record)
                                         {
                                             return record.kind == PerfStatRecord::Kind::Counter &&
                                                    record.domain == "mtp" &&
                                                    record.name == "draft_steps";
                                         });
    ASSERT_NE(counter_it, records.end());
    EXPECT_EQ(counter_it->count, 2u);
    EXPECT_DOUBLE_EQ(counter_it->value, 3.0);

    const auto timer_it = std::find_if(records.begin(), records.end(), [](const auto &record)
                                       {
                                           return record.kind == PerfStatRecord::Kind::Timer &&
                                                  record.domain == "mtp" &&
                                                  record.name == "sidecar_forward";
                                       });
    ASSERT_NE(timer_it, records.end());
    EXPECT_EQ(timer_it->count, 2u);
    EXPECT_EQ(timer_it->total_ns, 4000u);
    EXPECT_EQ(timer_it->min_ns, 1000u);
    EXPECT_EQ(timer_it->max_ns, 3000u);
    EXPECT_EQ(timer_it->device, "rocm:0");
}

TEST(Test__PerfStatsCollector, OrderedSequenceEvidenceIsBoundedAndRoleComparable)
{
    ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    constexpr uint64_t kSteps = 2048u;
    for (uint64_t step = 1u; step <= kSteps; ++step)
    {
        const uint64_t generation = 17u + step / 31u;
        PerfStatsCollector::recordOrderedSequenceStep(
            "forward_graph",
            "collective_sequence",
            {generation, step},
            "decode",
            "node",
            {{"role", "continuation"}});
        PerfStatsCollector::recordOrderedSequenceStep(
            "forward_graph",
            "collective_sequence",
            {generation, step},
            "decode",
            "node",
            {{"role", "participant"}});
    }
    // One reordered pair must produce distinct evidence even though the word
    // count and the multiset of identifiers are unchanged.
    PerfStatsCollector::recordOrderedSequenceStep(
        "forward_graph",
        "collective_sequence",
        {2u, 1u},
        "decode",
        "node",
        {{"role", "reordered"}});
    PerfStatsCollector::recordOrderedSequenceStep(
        "forward_graph",
        "collective_sequence",
        {1u, 2u},
        "decode",
        "node",
        {{"role", "reordered"}});

    const auto records = PerfStatsCollector::snapshot({"forward_graph"});
    ASSERT_EQ(records.size(), 3u)
        << "Sequence cardinality must be bounded by stable roles, not steps";
    const auto by_role = [&](const char *role)
    {
        return std::find_if(
            records.begin(),
            records.end(),
            [role](const PerfStatRecord &record)
            {
                const auto tag = record.tags.find("role");
                return record.kind == PerfStatRecord::Kind::OrderedSequence &&
                       tag != record.tags.end() && tag->second == role;
            });
    };

    const auto continuation_it = by_role("continuation");
    const auto participant_it = by_role("participant");
    const auto reordered_it = by_role("reordered");
    ASSERT_NE(continuation_it, records.end());
    ASSERT_NE(participant_it, records.end());
    ASSERT_NE(reordered_it, records.end());
    const auto &continuation = *continuation_it;
    const auto &participant = *participant_it;
    const auto &reordered = *reordered_it;
    EXPECT_EQ(continuation.count, kSteps);
    EXPECT_EQ(continuation.sequence_word_count, kSteps * 2u);
    EXPECT_EQ(continuation.sequence_digest_lo, participant.sequence_digest_lo);
    EXPECT_EQ(continuation.sequence_digest_hi, participant.sequence_digest_hi);
    EXPECT_NE(reordered.sequence_digest_lo, continuation.sequence_digest_lo);
    EXPECT_NE(reordered.sequence_digest_hi, continuation.sequence_digest_hi);

    const std::string csv =
        PerfStatsCollector::csvString({"forward_graph"});
    EXPECT_NE(csv.find("sequence_word_count"), std::string::npos);
    EXPECT_NE(csv.find("ordered_sequence,forward_graph,collective_sequence"),
              std::string::npos);
}

TEST(Test__ProductionParityEvidence, RecognizesNativeConditionalGenerationParent)
{
    const std::vector<PerfStatRecord> records = {
        PerfStatRecord{
            .kind = PerfStatRecord::Kind::Counter,
            .domain = "mtp",
            .name = "device_generation_loop_graph_materializations",
            .tags = {{"execution", "native_device_controlled_selector_while"}},
            .value = 1.0,
        },
        PerfStatRecord{
            .kind = PerfStatRecord::Kind::Counter,
            .domain = "mtp",
            .name = "device_generation_loop_graph_launches",
            .tags = {{"execution", "single_async_native_selector_while_launch"}},
            .value = 1.0,
        },
        PerfStatRecord{
            .kind = PerfStatRecord::Kind::Counter,
            .domain = "mtp",
            .name = "device_generation_terminal_compact_outcome_reductions",
            .tags = {{"execution", "native_conditional_graph"}},
            .value = 4.0,
        },
    };

    const auto evidence = collectProductionDeviceGenerationEvidence(records);
    EXPECT_TRUE(evidence.controller_observed);
    EXPECT_TRUE(evidence.hasNativeParent());
    EXPECT_TRUE(evidence.hasCertifiedGenerationLoop());
    EXPECT_FALSE(evidence.hosted_ticket_boundary_certified);
    EXPECT_EQ(
        evidence.policy,
        ProductionDeviceGenerationPolicy::NativeConditionalParent);
    EXPECT_STREQ(
        productionDeviceGenerationPolicyName(evidence.policy),
        "native_conditional_parent");
}

TEST(Test__ProductionParityEvidence,
     ClassifiesEveryAcceleratorBearingTopologyExplicitly)
{
    EXPECT_EQ(
        classifyProductionParityExecutionTopology(2, 0, 0),
        ProductionParityExecutionTopology::CPUOnly);
    EXPECT_EQ(
        classifyProductionParityExecutionTopology(0, 2, 0),
        ProductionParityExecutionTopology::HomogeneousGPU);
    EXPECT_EQ(
        classifyProductionParityExecutionTopology(0, 0, 4),
        ProductionParityExecutionTopology::HomogeneousGPU);
    EXPECT_EQ(
        classifyProductionParityExecutionTopology(2, 1, 0),
        ProductionParityExecutionTopology::HeterogeneousAccelerator);
    EXPECT_EQ(
        classifyProductionParityExecutionTopology(0, 1, 1),
        ProductionParityExecutionTopology::HeterogeneousAccelerator);
    EXPECT_EQ(
        resolveProductionParityGraphContract(
            ProductionParityExecutionTopology::HeterogeneousAccelerator,
            /*is_campaign_authority=*/true,
            /*local_accelerator=*/true),
        ProductionParityGraphContract::HeterogeneousCoordinatorSegmented);
    EXPECT_EQ(
        resolveProductionParityGraphContract(
            ProductionParityExecutionTopology::HeterogeneousAccelerator,
            /*is_campaign_authority=*/false,
            /*local_accelerator=*/true),
        ProductionParityGraphContract::
            HeterogeneousGPUParticipantCaptured);
    EXPECT_EQ(
        resolveProductionParityGraphContract(
            ProductionParityExecutionTopology::HeterogeneousAccelerator,
            /*is_campaign_authority=*/false,
            /*local_accelerator=*/false),
        ProductionParityGraphContract::
            HeterogeneousCPUParticipantDeclarative);
}

TEST(Test__ProductionParityEvidence,
     RejectsUncapturedHeterogeneousGraphExecution)
{
    ProductionParityEvidence evidence;
    evidence.graph_execution = true;
    evidence.execution_topology =
        ProductionParityExecutionTopology::HeterogeneousAccelerator;
    evidence.graph_contract =
        ProductionParityGraphContract::HeterogeneousCoordinatorSegmented;

    EXPECT_EQ(
        certifyProductionParityGraphExecution(evidence),
        ProductionParityGraphCertification::
            MissingHeterogeneousSegmentPlan)
        << "Declarative graph entry alone must never certify a GPU-bearing "
           "heterogeneous production cell";

    evidence.segmented_plan = true;
    evidence.segmented_capture = true;
    evidence.decode_graph_capture = true;
    evidence.decode_graph_replay = true;
    evidence.segmented_replay = true;
    EXPECT_EQ(
        certifyProductionParityGraphExecution(evidence),
        ProductionParityGraphCertification::Certified);
}

TEST(Test__ProductionParityEvidence,
     DistinguishesNativeAndSegmentedCaptureContracts)
{
    ProductionParityEvidence homogeneous;
    homogeneous.graph_execution = true;
    homogeneous.execution_topology =
        ProductionParityExecutionTopology::HomogeneousGPU;
    homogeneous.graph_contract =
        ProductionParityGraphContract::HomogeneousGPUCaptured;
    homogeneous.full_graph_replay = true;
    homogeneous.decode_graph_replay = true;
    EXPECT_EQ(
        certifyProductionParityGraphExecution(homogeneous),
        ProductionParityGraphCertification::Certified);

    homogeneous.segmented_plan = true;
    EXPECT_EQ(
        certifyProductionParityGraphExecution(homogeneous),
        ProductionParityGraphCertification::
            UnexpectedNativeSegmentation);

    ProductionParityEvidence cpu;
    cpu.graph_execution = true;
    cpu.execution_topology = ProductionParityExecutionTopology::CPUOnly;
    EXPECT_EQ(
        certifyProductionParityGraphExecution(cpu),
        ProductionParityGraphCertification::Certified);
}

TEST(Test__ProductionParityEvidence,
     HeterogeneousFollowersProveTheirRankLocalGraphContract)
{
    ProductionParityEvidence gpu_follower;
    gpu_follower.graph_execution = true;
    gpu_follower.execution_topology =
        ProductionParityExecutionTopology::HeterogeneousAccelerator;
    gpu_follower.graph_contract = ProductionParityGraphContract::
        HeterogeneousGPUParticipantCaptured;
    gpu_follower.full_graph_replay = true;
    gpu_follower.decode_graph_replay = true;
    EXPECT_EQ(
        certifyProductionParityGraphExecution(gpu_follower),
        ProductionParityGraphCertification::Certified);

    ProductionParityEvidence cpu_follower;
    cpu_follower.graph_execution = true;
    cpu_follower.execution_topology =
        ProductionParityExecutionTopology::HeterogeneousAccelerator;
    cpu_follower.graph_contract = ProductionParityGraphContract::
        HeterogeneousCPUParticipantDeclarative;
    EXPECT_EQ(
        certifyProductionParityGraphExecution(cpu_follower),
        ProductionParityGraphCertification::Certified);

    gpu_follower.graph_contract =
        ProductionParityGraphContract::HeterogeneousCoordinatorSegmented;
    EXPECT_EQ(
        certifyProductionParityGraphExecution(gpu_follower),
        ProductionParityGraphCertification::
            MissingHeterogeneousSegmentPlan)
        << "Only the inventory-resolved authority may satisfy the campaign's "
           "mandatory segmented-boundary proof";
}

TEST(Test__ProductionParityEvidence,
     RetainedParentDecodeReplayAcceptsEveryProductionCounterSpelling)
{
    const auto record = [](std::string name,
                           std::string phase = "decode",
                           double value = 1.0)
    {
        return PerfStatRecord{
            .kind = PerfStatRecord::Kind::Counter,
            .domain = "forward_graph",
            .name = std::move(name),
            .phase = std::move(phase),
            .device = "ROCm:0",
            .value = value,
        };
    };

    EXPECT_TRUE(productionParityHasRetainedParentDecodeReplay(
        {record("retained_parent_replays")}));
    EXPECT_TRUE(productionParityHasRetainedParentDecodeReplay(
        {record("retained_parent_transaction_replays")}));
    EXPECT_FALSE(productionParityHasRetainedParentDecodeReplay(
        {record("retained_parent_replays", "prefill")}));
    EXPECT_FALSE(productionParityHasRetainedParentDecodeReplay(
        {record("retained_parent_replays", "decode", 0.0)}));
}

TEST(Test__ProductionParityEvidence,
     DormantGenerationInfrastructureDoesNotImplyControllerExecution)
{
    const std::vector<PerfStatRecord> records = {
        PerfStatRecord{
            .kind = PerfStatRecord::Kind::Counter,
            .domain = "mtp",
            .name = "device_generation_loop_stream_initializations",
            .phase = "initialization",
            .device = "ROCm:0",
            .tags = {{"backend", "HIP"}},
            .value = 1.0,
        },
        PerfStatRecord{
            .kind = PerfStatRecord::Kind::Counter,
            .domain = "mtp",
            .name = "device_generation_loop_stream_initializations",
            .phase = "initialization",
            .device = "ROCm:1",
            .tags = {{"backend", "HIP"}},
            .value = 1.0,
        },
    };

    const auto evidence = collectProductionDeviceGenerationEvidence(records);
    EXPECT_FALSE(evidence.controller_observed);
    EXPECT_EQ(
        evidence.policy,
        ProductionDeviceGenerationPolicy::NotObserved);
    EXPECT_FALSE(evidence.hasCertifiedGenerationLoop());
    EXPECT_EQ(evidence.certification_detail, "not_observed");
}

TEST(Test__ProductionParityEvidence, DistinguishesHostedCapturedTransactions)
{
    const std::vector<PerfStatRecord> records = {
        PerfStatRecord{
            .kind = PerfStatRecord::Kind::Counter,
            .domain = "mtp",
            .name = "device_generation_loop_graph_materializations",
            .tags = {
                {"execution",
                 "hosted_captured_transactions_with_ticket_only_dispatch"}},
            .value = 1.0,
        },
        PerfStatRecord{
            .kind = PerfStatRecord::Kind::Counter,
            .domain = "mtp",
            .name = "device_generation_loop_graph_launches",
            .tags = {
                {"execution",
                 "hosted_ticket_selected_captured_transactions"}},
            .value = 1.0,
        },
        PerfStatRecord{
            .kind = PerfStatRecord::Kind::Counter,
            .domain = "mtp",
            .name = "device_generation_terminal_compact_outcome_reductions",
            .tags = {
                {"execution", "host_scheduled_captured_transactions"}},
            .value = 3.0,
        },
    };

    const auto evidence = collectProductionDeviceGenerationEvidence(records);
    EXPECT_TRUE(evidence.controller_observed);
    EXPECT_FALSE(evidence.hasNativeParent());
    EXPECT_FALSE(evidence.hosted_ticket_boundary_certified);
    EXPECT_FALSE(evidence.hasCertifiedGenerationLoop());
    EXPECT_EQ(
        evidence.policy,
        ProductionDeviceGenerationPolicy::HostScheduledCapturedTransactions);
}

TEST(Test__ProductionParityEvidence, CertifiesMirroredHostedTicketBoundary)
{
    const auto evidence = collectProductionDeviceGenerationEvidence(
        certifiedHostedTicketRecords());
    EXPECT_TRUE(evidence.controller_observed);
    EXPECT_FALSE(evidence.hasNativeParent());
    EXPECT_TRUE(evidence.hosted_ticket_boundary_certified);
    EXPECT_TRUE(evidence.hasCertifiedGenerationLoop());
    EXPECT_EQ(
        evidence.policy,
        ProductionDeviceGenerationPolicy::HostScheduledCapturedTransactions);
}

TEST(Test__ProductionParityEvidence, CertifiesTerminalOnlyHostedTicketBoundary)
{
    const auto evidence = collectProductionDeviceGenerationEvidence(
        certifiedTerminalOnlyHostedTicketRecords());
    EXPECT_TRUE(evidence.controller_observed);
    EXPECT_FALSE(evidence.hasNativeParent());
    EXPECT_TRUE(evidence.hosted_ticket_boundary_certified)
        << evidence.certification_detail;
    EXPECT_TRUE(evidence.hasCertifiedGenerationLoop());
    EXPECT_EQ(
        evidence.policy,
        ProductionDeviceGenerationPolicy::HostScheduledCapturedTransactions);
}

TEST(Test__ProductionParityEvidence, CertifiesRankCoordinatedHostedLaunch)
{
    auto records = certifiedHostedTicketRecords();
    std::erase_if(
        records,
        [](const PerfStatRecord &record)
        {
            return record.name == "device_generation_loop_graph_launches";
        });
    records.push_back(PerfStatRecord{
        .kind = PerfStatRecord::Kind::Counter,
        .domain = "mtp",
        .name = "rank_device_generation_parent_launches",
        .device = "rank",
        .tags = {
            {"execution", "hosted_ticket_selected_captured_transactions"},
            {"launch_order", "all_participants_before_ticket_wait"},
            {"participants", "2"}},
        .value = 1.0,
    });

    const auto evidence = collectProductionDeviceGenerationEvidence(records);
    EXPECT_TRUE(evidence.hosted_ticket_boundary_certified)
        << evidence.certification_detail;
    EXPECT_TRUE(evidence.hasCertifiedGenerationLoop());

    records.back().tags["participants"] = "3";
    const auto malformed =
        collectProductionDeviceGenerationEvidence(records);
    EXPECT_FALSE(malformed.hosted_ticket_boundary_certified);
    EXPECT_FALSE(malformed.hasCertifiedGenerationLoop());
}

TEST(Test__ProductionParityEvidence, CertifiesExactRetainedHostedGraphReuse)
{
    auto records = certifiedHostedTicketRecords();
    for (PerfStatRecord &record : records)
    {
        if (record.name !=
            "device_generation_loop_graph_materializations")
        {
            continue;
        }

        record.name = "device_generation_loop_graph_reuses";
        record.tags.erase("execution");
        record.tags.insert({
            {"execution_policy", "host_scheduled_captured_transactions"},
            {"requests", "1"},
            {"draft_depth", "2"},
            {"minimum_draft_depth", "2"},
            {"maximum_draft_depth", "2"},
            {"depth_policy", "fixed_width"},
            {"sampling_mode", "stochastic"},
            {"conditional_fragments", "0"},
            {"ticket_conditioned_fragments", "1"},
            {"workspace_generation", "7"},
        });
    }

    const auto evidence = collectProductionDeviceGenerationEvidence(records);
    EXPECT_TRUE(evidence.hosted_ticket_boundary_certified)
        << evidence.certification_detail;
    EXPECT_TRUE(evidence.hasCertifiedGenerationLoop());

    auto malformed = records;
    auto reuse = std::find_if(
        malformed.begin(),
        malformed.end(),
        [](const PerfStatRecord &record)
        {
            return record.name == "device_generation_loop_graph_reuses" &&
                   record.device == "ROCm:1";
        });
    ASSERT_NE(reuse, malformed.end());
    reuse->tags["workspace_generation"] = "0";
    const auto malformed_evidence =
        collectProductionDeviceGenerationEvidence(malformed);
    EXPECT_FALSE(malformed_evidence.hosted_ticket_boundary_certified);
    EXPECT_FALSE(malformed_evidence.hasCertifiedGenerationLoop());
}

TEST(Test__ProductionParityEvidence, RejectsMalformedHostedTicketEvidence)
{
    auto native_conditionals = certifiedHostedTicketRecords();
    auto materialization = std::find_if(
        native_conditionals.begin(),
        native_conditionals.end(),
        [](const PerfStatRecord &record)
        {
            return record.name ==
                       "device_generation_loop_graph_materializations" &&
                   record.device == "ROCm:0";
        });
    ASSERT_NE(materialization, native_conditionals.end());
    materialization->tags["conditional_fragments"] = "1";
    EXPECT_FALSE(collectProductionDeviceGenerationEvidence(native_conditionals)
                     .hosted_ticket_boundary_certified)
        << "HIP hosted execution must not claim native conditional graph nodes.";

    auto malformed_ticket = certifiedHostedTicketRecords();
    auto ticket = std::find_if(
        malformed_ticket.begin(),
        malformed_ticket.end(),
        [](const PerfStatRecord &record)
        {
            return record.name ==
                       "device_generation_dispatch_ticket_d2h_submissions" &&
                   record.device == "ROCm:0";
        });
    ASSERT_NE(ticket, malformed_ticket.end());
    ticket->tags["bytes"] = "64";
    EXPECT_FALSE(collectProductionDeviceGenerationEvidence(malformed_ticket)
                     .hosted_ticket_boundary_certified);

    auto divergent_ledger = certifiedHostedTicketRecords();
    auto transactions = std::find_if(
        divergent_ledger.begin(),
        divergent_ledger.end(),
        [](const PerfStatRecord &record)
        {
            return record.name ==
                       "device_generation_terminal_transactions" &&
                   record.device == "ROCm:1";
        });
    ASSERT_NE(transactions, divergent_ledger.end());
    transactions->value = 4.0;
    EXPECT_FALSE(collectProductionDeviceGenerationEvidence(divergent_ledger)
                     .hosted_ticket_boundary_certified);

    auto mutable_host_read = certifiedHostedTicketRecords();
    mutable_host_read.push_back(PerfStatRecord{
        .kind = PerfStatRecord::Kind::Counter,
        .domain = "mtp",
        .name = "grouped_outcome_stochastic_device_outcome_host_bridge",
        .device = "ROCm:0",
        .value = 1.0,
    });
    EXPECT_FALSE(collectProductionDeviceGenerationEvidence(mutable_host_read)
                     .hosted_ticket_boundary_certified);
}

TEST(Test__ProductionParityEvidence, RejectsMissingOrConflictingAuthorityTags)
{
    const PerfStatRecord unclassified{
        .kind = PerfStatRecord::Kind::Counter,
        .domain = "mtp",
        .name = "device_generation_terminal_transactions",
        .value = 2.0,
    };
    auto evidence =
        collectProductionDeviceGenerationEvidence({unclassified});
    EXPECT_TRUE(evidence.controller_observed);
    EXPECT_FALSE(evidence.hasNativeParent());
    EXPECT_EQ(
        evidence.policy,
        ProductionDeviceGenerationPolicy::Unclassified);

    PerfStatRecord native = unclassified;
    native.name = "device_generation_loop_graph_launches";
    native.tags = {{"execution", "single_async_native_while_launch"}};
    PerfStatRecord hosted = native;
    hosted.tags = {
        {"execution", "hosted_ticket_selected_captured_transactions"}};
    evidence = collectProductionDeviceGenerationEvidence({native, hosted});
    EXPECT_TRUE(evidence.controller_observed);
    EXPECT_FALSE(evidence.hasNativeParent());
    EXPECT_EQ(
        evidence.policy,
        ProductionDeviceGenerationPolicy::Inconsistent);
}

TEST(Test__PerfStatsCollector, ExportsFilteredJsonAndCsv)
{
    ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    PerfStatsCollector::addCounter("mtp", "accepted_tokens", 2.0, "decode");
    PerfStatsCollector::addCounter("prefix_cache", "hits", 1.0, "prefill");

    const auto json = nlohmann::json::parse(PerfStatsCollector::jsonString({"mtp"}));
    ASSERT_EQ(json.at("schema"), "llaminar.perf_stats.v1");
    ASSERT_EQ(json.at("records").size(), 1u);
    EXPECT_EQ(json.at("records")[0].at("domain"), "mtp");
    EXPECT_EQ(json.at("records")[0].at("name"), "accepted_tokens");
    EXPECT_DOUBLE_EQ(json.at("records")[0].at("value").get<double>(), 2.0);

    const std::string csv = PerfStatsCollector::csvString({"mtp"});
    EXPECT_NE(csv.find("kind,domain,name,phase,device,tags,count,value"), std::string::npos);
    EXPECT_NE(csv.find("counter,mtp,accepted_tokens,decode"), std::string::npos);
    EXPECT_EQ(csv.find("prefix_cache"), std::string::npos);
}

TEST(Test__PerfStatsCollector, SummaryTableCanBeRequestedByEnv)
{
    ScopedEnv enable("LLAMINAR_PERF_STATS_TABLE", "1");
    PerfStatsCollector::reset();

    ASSERT_TRUE(PerfStatsCollector::isEnabled());

    PerfStatsCollector::addCounter("mtp", "draft_steps", 3.0, "decode", "rocm:0");
    PerfStatsCollector::recordTimingNs("mtp", "sidecar_forward", 5000, "decode", "rocm:0");

    const std::string summary = PerfStatsCollector::summaryString({"mtp"});
    EXPECT_NE(summary.find("UNIFIED PERF STATS"), std::string::npos);
    EXPECT_NE(summary.find("mtp.draft_steps"), std::string::npos);
    EXPECT_NE(summary.find("mtp.sidecar_forward"), std::string::npos);
}

TEST(Test__PerfStatsCollector, ResetCanPreserveSelectedDomains)
{
    ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    PerfStatsCollector::addCounter("moe_rebalance", "apply_calls", 1.0, "rebalance");
    PerfStatsCollector::recordTimingNs("moe_rebalance", "gpu_direct_transfer", 2000, "rebalance");
    PerfStatsCollector::addCounter("stage_gpu", "graph_replay.total", 3.0, "decode");
    PerfStatsCollector::addCounter("forward_pass", "decode", 1.0, "decode");

    PerfStatsCollector::resetPreservingDomains({"moe_rebalance"});

    const auto records = PerfStatsCollector::snapshot();
    ASSERT_EQ(records.size(), 2u);
    EXPECT_TRUE(std::all_of(records.begin(), records.end(), [](const auto &record)
                            { return record.domain == "moe_rebalance"; }));

    const auto has_apply = std::any_of(records.begin(), records.end(), [](const auto &record)
                                       {
                                           return record.kind == PerfStatRecord::Kind::Counter &&
                                                  record.name == "apply_calls";
                                       });
    const auto has_transfer = std::any_of(records.begin(), records.end(), [](const auto &record)
                                          {
                                              return record.kind == PerfStatRecord::Kind::Timer &&
                                                     record.name == "gpu_direct_transfer";
                                          });
    EXPECT_TRUE(has_apply);
    EXPECT_TRUE(has_transfer);
}

/**
 * @brief Mixed domains retain only explicitly named setup record families.
 *
 * Benchmark graph evidence shares `forward_graph` with measured replay and
 * host-wall timing records. This test prevents a future reset from either
 * discarding capture certification or retaining the entire warmup runtime.
 */
TEST(Test__PerfStatsCollector, ResetCanPreserveExactFamiliesFromMixedDomains)
{
    ScopedEnv env("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    PerfStatsCollector::addCounter(
        "forward_graph", "full_graph_plan_graphs", 1.0, "setup");
    PerfStatsCollector::addCounter(
        "forward_graph", "decode_graph_phase", 1.0, "setup");
    PerfStatsCollector::recordTimingNs(
        "forward_graph", "full_graph_replay_graph", 9000, "warmup");
    PerfStatsCollector::addCounter(
        "mtp", "draft_steps", 3.0, "warmup");
    PerfStatsCollector::addCounter(
        "gpu_graph_inventory", "kernel_nodes", 7.0, "graph_setup");

    PerfStatsCollector::resetPreserving(
        {"gpu_graph_inventory"},
        {{"forward_graph", "full_graph_plan_graphs"},
         {"forward_graph", "decode_graph_phase"}});

    const auto records = PerfStatsCollector::snapshot();
    ASSERT_EQ(records.size(), 3u);
    EXPECT_TRUE(std::any_of(
        records.begin(), records.end(), [](const PerfStatRecord &record)
        {
            return record.domain == "forward_graph" &&
                   record.name == "full_graph_plan_graphs";
        }));
    EXPECT_TRUE(std::any_of(
        records.begin(), records.end(), [](const PerfStatRecord &record)
        {
            return record.domain == "forward_graph" &&
                   record.name == "decode_graph_phase";
        }));
    EXPECT_TRUE(std::any_of(
        records.begin(), records.end(), [](const PerfStatRecord &record)
        {
            return record.domain == "gpu_graph_inventory" &&
                   record.name == "kernel_nodes";
        }));
    EXPECT_FALSE(std::any_of(
        records.begin(), records.end(), [](const PerfStatRecord &record)
        {
            return record.name == "full_graph_replay_graph" ||
                   record.domain == "mtp";
        }));

    PerfStatsCollector::reset();
}

TEST(Test__PerfStatsCollector, PerfStatsExportAloneDoesNotEnableGpuStageEventTiming)
{
    ScopedEnv profiling("LLAMINAR_PROFILING", nullptr);
    ScopedEnv stage_timing("LLAMINAR_GPU_STAGE_TIMING", nullptr);
    ScopedEnv stage_detail("LLAMINAR_GPU_STAGE_TIMING_DETAIL", nullptr);
    ScopedEnv perf_stage_timing("LLAMINAR_PERF_STATS_GPU_STAGE_TIMING", nullptr);
    ScopedEnv filter("LLAMINAR_PERF_STATS_FILTER", nullptr);
    ScopedEnv json("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    EXPECT_TRUE(PerfStatsCollector::isEnabled());
    EXPECT_FALSE(PerfStatsCollector::gpuStageEventTimingEnabled());
}

TEST(Test__PerfStatsCollector, UnrelatedExportDoesNotEnableCpuStageTiming)
{
    ScopedEnv profiling("LLAMINAR_PROFILING", nullptr);
    ScopedEnv kernel_profiling("LLAMINAR_PROFILE_KERNELS", nullptr);
    ScopedEnv cpu_stage_timing("LLAMINAR_PERF_STATS_CPU_STAGE_TIMING", nullptr);
    ScopedEnv filter("LLAMINAR_PERF_STATS_FILTER", "moe_canonical_publication");
    ScopedEnv json("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    EXPECT_TRUE(PerfStatsCollector::isEnabled());
    EXPECT_FALSE(PerfStatsCollector::cpuStageTimingEnabled());
}

TEST(Test__PerfStatsCollector, ExportFilterIsACollectionDomainGate)
{
    ScopedEnv profiling("LLAMINAR_PROFILING", nullptr);
    ScopedEnv kernel_profiling("LLAMINAR_PROFILE_KERNELS", nullptr);
    ScopedEnv filter("LLAMINAR_PERF_STATS_FILTER", "mtp");
    ScopedEnv json("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    EXPECT_TRUE(PerfStatsCollector::isDomainEnabled("mtp"));
    EXPECT_FALSE(PerfStatsCollector::isDomainEnabled("kernel"));
    EXPECT_FALSE(PerfStatsCollector::isDomainEnabled("stage_cpu_detail"));

    PerfStatsCollector::addCounter("mtp", "kept", 1.0);
    PerfStatsCollector::addCounter("kernel", "discarded", 1.0);
    {
        PerfStatsCollector::ScopedTimer discarded_timer(
            "kernel", "discarded_timer");
    }

    const auto records = PerfStatsCollector::snapshot();
    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records.front().domain, "mtp");
    EXPECT_EQ(records.front().name, "kept");
}

/**
 * @brief Environment parsing is a setup transition, never dormant hot work.
 *
 * The raw mutation between the two observations deliberately omits a reload.
 * If `isDomainEnabled()` starts consulting `getenv()` again, the middle
 * assertion fails and protects the disabled inference fast path that the 122B
 * Dynamic benchmark exposed.
 */
TEST(Test__PerfStatsCollector, EnvironmentIsParsedOnlyAtExplicitPolicyBoundaries)
{
    ScopedEnv profiling("LLAMINAR_PROFILING", nullptr);
    ScopedEnv stage_timing("LLAMINAR_GPU_STAGE_TIMING", nullptr);
    ScopedEnv stage_detail("LLAMINAR_GPU_STAGE_TIMING_DETAIL", nullptr);
    ScopedEnv perf_gpu_timing(
        "LLAMINAR_PERF_STATS_GPU_STAGE_TIMING", nullptr);
    ScopedEnv perf_cpu_timing(
        "LLAMINAR_PERF_STATS_CPU_STAGE_TIMING", nullptr);
    ScopedEnv table("LLAMINAR_PERF_STATS_TABLE", nullptr);
    ScopedEnv summary("LLAMINAR_PERF_STATS_SUMMARY", nullptr);
    ScopedEnv csv("LLAMINAR_PERF_STATS_CSV", nullptr);
    ScopedEnv json("LLAMINAR_PERF_STATS_JSON", nullptr);
    PerfStatsCollector::reset();

    ASSERT_FALSE(PerfStatsCollector::isEnabled());
    ASSERT_FALSE(PerfStatsCollector::isDomainEnabled("moe_overlay_residency"));

    ASSERT_EQ(::setenv("LLAMINAR_PERF_STATS_JSON", "1", 1), 0);
    EXPECT_FALSE(PerfStatsCollector::isEnabled());
    EXPECT_FALSE(PerfStatsCollector::isDomainEnabled("moe_overlay_residency"));

    PerfStatsCollector::reloadConfigurationFromEnvironment();
    EXPECT_TRUE(PerfStatsCollector::isEnabled());
    EXPECT_TRUE(PerfStatsCollector::isDomainEnabled("moe_overlay_residency"));

    ASSERT_EQ(::unsetenv("LLAMINAR_PERF_STATS_JSON"), 0);
    PerfStatsCollector::reloadConfigurationFromEnvironment();
    EXPECT_FALSE(PerfStatsCollector::isEnabled());
}

TEST(Test__PerfStatsCollector, QualifiedFilterEnablesItsOwningDomain)
{
    ScopedEnv profiling("LLAMINAR_PROFILING", nullptr);
    ScopedEnv kernel_profiling("LLAMINAR_PROFILE_KERNELS", nullptr);
    ScopedEnv filter(
        "LLAMINAR_PERF_STATS_FILTER",
        "mtp.verifier_forward");
    ScopedEnv json("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    EXPECT_TRUE(PerfStatsCollector::isDomainEnabled("mtp"));
    EXPECT_FALSE(PerfStatsCollector::isDomainEnabled("moe"));
}

TEST(Test__PerfStatsCollector, CpuStageFilterEnablesCpuStageTiming)
{
    ScopedEnv profiling("LLAMINAR_PROFILING", nullptr);
    ScopedEnv kernel_profiling("LLAMINAR_PROFILE_KERNELS", nullptr);
    ScopedEnv cpu_stage_timing("LLAMINAR_PERF_STATS_CPU_STAGE_TIMING", nullptr);
    ScopedEnv filter("LLAMINAR_PERF_STATS_FILTER", "stage_cpu_detail.verifier");
    ScopedEnv json("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    EXPECT_TRUE(PerfStatsCollector::isEnabled());
    EXPECT_TRUE(PerfStatsCollector::cpuStageTimingEnabled());

    PerfStatsCollector::addCounter("stage_cpu", "verifier", 1.0);
    PerfStatsCollector::addCounter("stage_cpu_detail", "verifier", 1.0);
    const auto records = PerfStatsCollector::snapshot();
    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records.front().domain, "stage_cpu_detail");
}

TEST(Test__PerfStatsCollector, ExplicitCpuStageTimingEnablesStructuredCollection)
{
    ScopedEnv profiling("LLAMINAR_PROFILING", nullptr);
    ScopedEnv kernel_profiling("LLAMINAR_PROFILE_KERNELS", nullptr);
    ScopedEnv filter("LLAMINAR_PERF_STATS_FILTER", nullptr);
    ScopedEnv json("LLAMINAR_PERF_STATS_JSON", nullptr);
    ScopedEnv csv("LLAMINAR_PERF_STATS_CSV", nullptr);
    ScopedEnv cpu_stage_timing("LLAMINAR_PERF_STATS_CPU_STAGE_TIMING", "1");
    PerfStatsCollector::reset();

    EXPECT_TRUE(PerfStatsCollector::isEnabled());
    EXPECT_TRUE(PerfStatsCollector::cpuStageTimingEnabled());
}

TEST(Test__PerfStatsCollector, GraphKernelInventoryIsExplicitAndReloadable)
{
    ScopedEnv inventory("LLAMINAR_GPU_GRAPH_KERNEL_INVENTORY", nullptr);
    EXPECT_FALSE(debugEnv().runtime_debug.gpu_graph_kernel_inventory);

    {
        ScopedEnv enable("LLAMINAR_GPU_GRAPH_KERNEL_INVENTORY", "1");
        EXPECT_TRUE(debugEnv().runtime_debug.gpu_graph_kernel_inventory);
    }

    EXPECT_FALSE(debugEnv().runtime_debug.gpu_graph_kernel_inventory);
}

TEST(Test__PerfStatsCollector, GpuStageTimingEnablesStructuredCollection)
{
    ScopedEnv profiling("LLAMINAR_PROFILING", nullptr);
    ScopedEnv stage_timing("LLAMINAR_GPU_STAGE_TIMING", "1");
    ScopedEnv perf_stage_timing("LLAMINAR_PERF_STATS_GPU_STAGE_TIMING", nullptr);
    ScopedEnv cpu_stage_timing("LLAMINAR_PERF_STATS_CPU_STAGE_TIMING", nullptr);
    ScopedEnv filter("LLAMINAR_PERF_STATS_FILTER", nullptr);
    ScopedEnv json("LLAMINAR_PERF_STATS_JSON", nullptr);
    ScopedEnv csv("LLAMINAR_PERF_STATS_CSV", nullptr);
    PerfStatsCollector::reset();

    EXPECT_TRUE(PerfStatsCollector::isEnabled());
    EXPECT_FALSE(PerfStatsCollector::cpuStageTimingEnabled());
    EXPECT_TRUE(PerfStatsCollector::gpuStageEventTimingEnabled());

    PerfStatsCollector::recordTimingNs(
        "stage_gpu",
        "graph_replay.total",
        1000,
        "decode",
        "cuda:0",
        {{"attribution", "gpu_event"}});
    EXPECT_EQ(PerfStatsCollector::snapshot({"stage_gpu"}).size(), 1u);
}

TEST(Test__PerfStatsCollector, PerfStatsStageGpuRequestsEnableGpuStageEventTiming)
{
    {
        ScopedEnv profiling("LLAMINAR_PROFILING", nullptr);
        ScopedEnv stage_timing("LLAMINAR_GPU_STAGE_TIMING", nullptr);
        ScopedEnv stage_detail("LLAMINAR_GPU_STAGE_TIMING_DETAIL", nullptr);
        ScopedEnv json("LLAMINAR_PERF_STATS_JSON", "1");
        ScopedEnv filter("LLAMINAR_PERF_STATS_FILTER", "stage_gpu");
        ScopedEnv perf_stage_timing("LLAMINAR_PERF_STATS_GPU_STAGE_TIMING", nullptr);
        PerfStatsCollector::reset();

        EXPECT_TRUE(PerfStatsCollector::gpuStageEventTimingEnabled());
    }

    {
        ScopedEnv profiling("LLAMINAR_PROFILING", nullptr);
        ScopedEnv stage_timing("LLAMINAR_GPU_STAGE_TIMING", nullptr);
        ScopedEnv stage_detail("LLAMINAR_GPU_STAGE_TIMING_DETAIL", nullptr);
        ScopedEnv json("LLAMINAR_PERF_STATS_JSON", "1");
        ScopedEnv filter("LLAMINAR_PERF_STATS_FILTER", nullptr);
        ScopedEnv perf_stage_timing("LLAMINAR_PERF_STATS_GPU_STAGE_TIMING", "1");
        PerfStatsCollector::reset();

        EXPECT_TRUE(PerfStatsCollector::gpuStageEventTimingEnabled());
    }
}

TEST(Test__PerfStatsCollector, DeprecatedUnifiedProfilingAliasesGraphSafePerfStatsOnly)
{
    ScopedEnv kernel_profiling("LLAMINAR_PROFILE_KERNELS", nullptr);
    ScopedEnv executor_profiling("LLAMINAR_EXECUTOR_PROFILING", nullptr);
    ScopedEnv stage_timing("LLAMINAR_GPU_STAGE_TIMING", nullptr);
    ScopedEnv perf_stage_timing("LLAMINAR_PERF_STATS_GPU_STAGE_TIMING", nullptr);
    ScopedEnv cpu_stage_timing("LLAMINAR_PERF_STATS_CPU_STAGE_TIMING", nullptr);
    ScopedEnv summary("LLAMINAR_PERF_STATS_SUMMARY", nullptr);
    ScopedEnv profiling("LLAMINAR_PROFILING", "1");

    EXPECT_TRUE(PerfStatsCollector::isEnabled());
    EXPECT_TRUE(PerfStatsCollector::cpuStageTimingEnabled());
    EXPECT_TRUE(PerfStatsCollector::gpuStageEventTimingEnabled());
    EXPECT_FALSE(debugEnv().profile.enabled);
    EXPECT_FALSE(debugEnv().execution.executor_profiling);
}

TEST(Test__PerfStatsCollector, DeprecatedUnifiedProfilingFlushesSummaryWithoutExportPath)
{
    ScopedEnv json("LLAMINAR_PERF_STATS_JSON", nullptr);
    ScopedEnv csv("LLAMINAR_PERF_STATS_CSV", nullptr);
    ScopedEnv table("LLAMINAR_PERF_STATS_TABLE", nullptr);
    ScopedEnv summary("LLAMINAR_PERF_STATS_SUMMARY", nullptr);
    ScopedEnv profiling("LLAMINAR_PROFILING", "1");
    PerfStatsCollector::reset();

    PerfStatsCollector::addCounter("profiling", "summary_only_flush", 1.0);
    testing::internal::CaptureStdout();
    EXPECT_TRUE(PerfStatsCollector::flushFromEnv());
    const std::string output = testing::internal::GetCapturedStdout();

    EXPECT_NE(output.find("UNIFIED PERF STATS"), std::string::npos);
    EXPECT_NE(output.find("profiling.summary_only_flush"), std::string::npos);
}

TEST(Test__PerfStatsCollector, RankQualifiedExportWritesParticipantLocalEvidence)
{
    ScopedLoggerRank rank(7);

    const std::filesystem::path path_template =
        uniqueTempPath(".rank-{rank}.json");
    std::string expected_path_text = path_template.string();
    const size_t token = expected_path_text.find("{rank}");
    ASSERT_NE(token, std::string::npos);
    expected_path_text.replace(token, std::string("{rank}").size(), "7");
    const std::filesystem::path expected_path(expected_path_text);

    {
        ScopedEnv json("LLAMINAR_PERF_STATS_JSON", path_template.c_str());
        ScopedEnv csv("LLAMINAR_PERF_STATS_CSV", nullptr);
        ScopedEnv summary("LLAMINAR_PERF_STATS_SUMMARY", nullptr);
        PerfStatsCollector::reset();
        PerfStatsCollector::addCounter(
            "stage_cpu", "rank_local_probe", 1.0, "prefill", "cpu");

        ASSERT_TRUE(PerfStatsCollector::flushFromEnv());
        ASSERT_TRUE(std::filesystem::exists(expected_path));
        const auto document = nlohmann::json::parse(readFile(expected_path));
        ASSERT_EQ(document.at("records").size(), 1u);
        EXPECT_EQ(document.at("records")[0].at("name"), "rank_local_probe");
    }

    std::error_code error;
    std::filesystem::remove(expected_path, error);
}

TEST(Test__PerfStatsCollector, ExistingProfilersPublishStructuredRecords)
{
    ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", "1");
    ScopedEnv kernel_timing("LLAMINAR_PROFILE_KERNELS", "1");
    PerfStatsCollector::reset();
    KernelProfiler::resetAll();
    KVCacheProfiler::reset();
    WeightLoadingProfiler::reset();

    KernelProfiler::setCurrentPhase(KernelProfiler::Phase::DECODE);
    KernelProfiler::record(KernelType::LM_HEAD, 2000, "rocm:0");

    KVCacheProfiler::setCurrentPhase(KVCacheProfiler::Phase::DECODE);
    KVCacheProfiler::record(KVCacheOpType::APPEND, 3000, 4, 128);

    WeightLoadingProfiler::addDetail("weights.gemm_pack.test", 0.25);

    const auto records = PerfStatsCollector::snapshot();
    auto has_record = [&](const std::string &domain, const std::string &name)
    {
        return std::any_of(records.begin(), records.end(), [&](const auto &record)
                           {
                               return record.domain == domain && record.name == name;
                           });
    };

    EXPECT_TRUE(has_record("kernel", "LM_HEAD"));
    EXPECT_TRUE(has_record("kv_cache", "KV_APPEND"));
    EXPECT_TRUE(has_record("kv_cache", "tokens"));
    EXPECT_TRUE(has_record("kv_cache", "bytes"));
    EXPECT_TRUE(has_record("weight_loading", "weights.gemm_pack.test"));
}

TEST(Test__PerfStatsCollector, WeightLoadingTimersRetainConcurrentParticipantScopes)
{
    ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();
    WeightLoadingProfiler::reset();

    constexpr int kParticipants = 4;
    std::mutex mutex;
    std::condition_variable condition;
    int arrived = 0;
    bool release = false;
    std::vector<std::thread> workers;
    workers.reserve(kParticipants);
    for (int participant = 0; participant < kParticipants; ++participant)
    {
        workers.emplace_back(
            [&]
            {
                ScopedWeightLoadTimer phase(WeightLoadPhase::GRAPH_BUILD);
                ScopedWeightLoadDetailTimer detail(
                    "graph.build.concurrent_participant");
                std::unique_lock<std::mutex> lock(mutex);
                ++arrived;
                condition.notify_all();
                condition.wait(lock, [&] { return release; });
            });
    }

    {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [&] { return arrived == kParticipants; });
        release = true;
    }
    condition.notify_all();
    for (auto &worker : workers)
        worker.join();

    const auto records = PerfStatsCollector::snapshot({"weight_loading"});
    const auto find_record = [&](const char *name) -> const PerfStatRecord *
    {
        const auto it = std::find_if(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record) { return record.name == name; });
        return it == records.end() ? nullptr : &*it;
    };
    const auto *phase_record = find_record("graph_build");
    const auto *detail_record =
        find_record("graph.build.concurrent_participant");
    ASSERT_NE(phase_record, nullptr);
    ASSERT_NE(detail_record, nullptr);
    EXPECT_EQ(phase_record->count, kParticipants);
    EXPECT_EQ(detail_record->count, kParticipants);

    WeightLoadingProfiler::reset();
    PerfStatsCollector::reset();
}

TEST(Test__PerfStatsCollector, GraphReplayTimersCarrySyncScopeTags)
{
    ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    PerfStatsCollector::recordTimingNs(
        "forward_graph",
        "segmented_replay_total",
        1000,
        "decode",
        "cuda:0",
        {{"context", "main_decode"}, {"sync_scope", "stream_synchronized"}});
    PerfStatsCollector::recordTimingNs(
        "forward_graph",
        "segmented_replay_total",
        2000,
        "decode",
        "cuda:0",
        {{"context", "mtp_decode_sidecar"}, {"sync_scope", "launch_only_deferred"}});

    const auto records = PerfStatsCollector::snapshot({"forward_graph"});
    ASSERT_EQ(records.size(), 2u);

    auto has_sync_scope = [&](const std::string &scope) {
        return std::any_of(records.begin(), records.end(), [&](const PerfStatRecord &record) {
            const auto it = record.tags.find("sync_scope");
            return record.domain == "forward_graph" &&
                   record.name == "segmented_replay_total" &&
                   it != record.tags.end() &&
                   it->second == scope;
        });
    };

    EXPECT_TRUE(has_sync_scope("stream_synchronized"));
    EXPECT_TRUE(has_sync_scope("launch_only_deferred"));

    const std::string csv = PerfStatsCollector::csvString({"forward_graph"});
    EXPECT_NE(csv.find("sync_scope=stream_synchronized"), std::string::npos);
    EXPECT_NE(csv.find("sync_scope=launch_only_deferred"), std::string::npos);
}

TEST(Test__PerfStatsCollector, FlushFromEnvWritesMachineReadableFiles)
{
    const auto json_path = uniqueTempPath(".json");
    const auto csv_path = uniqueTempPath(".csv");
    ScopedEnv json_env("LLAMINAR_PERF_STATS_JSON", json_path.string().c_str());
    ScopedEnv csv_env("LLAMINAR_PERF_STATS_CSV", csv_path.string().c_str());
    ScopedEnv filter_env("LLAMINAR_PERF_STATS_FILTER", "mtp");
    PerfStatsCollector::reset();

    PerfStatsCollector::recordTimingNs(
        "mtp",
        "verifier_forward",
        123456,
        "decode",
        "cpu",
        {{"depth", "0"}});
    PerfStatsCollector::addCounter("kernel", "gemm_calls", 9.0, "decode");

    ASSERT_TRUE(PerfStatsCollector::flushFromEnv());
    ASSERT_TRUE(std::filesystem::exists(json_path));
    ASSERT_TRUE(std::filesystem::exists(csv_path));

    const auto json = nlohmann::json::parse(readFile(json_path));
    ASSERT_EQ(json.at("records").size(), 1u);
    EXPECT_EQ(json.at("records")[0].at("domain"), "mtp");
    EXPECT_EQ(json.at("records")[0].at("name"), "verifier_forward");
    EXPECT_EQ(json.at("records")[0].at("tags").at("depth"), "0");

    const std::string csv = readFile(csv_path);
    EXPECT_NE(csv.find("timer,mtp,verifier_forward,decode,cpu,depth=0"), std::string::npos);
    EXPECT_EQ(csv.find("kernel"), std::string::npos);

    std::filesystem::remove(json_path);
    std::filesystem::remove(csv_path);
}

TEST(Test__PerfStatsCollector, FlushFromEnvWritesMultipleFilteredDomainsForMTPGraphEvidence)
{
    const auto json_path = uniqueTempPath(".json");
    const auto csv_path = uniqueTempPath(".csv");
    ScopedEnv json_env("LLAMINAR_PERF_STATS_JSON", json_path.string().c_str());
    ScopedEnv csv_env("LLAMINAR_PERF_STATS_CSV", csv_path.string().c_str());
    ScopedEnv filter_env("LLAMINAR_PERF_STATS_FILTER", "mtp,forward_graph");
    PerfStatsCollector::reset();

    PerfStatsCollector::recordTimingNs(
        "mtp",
        "verifier_forward",
        2000000,
        "decode",
        "cuda:0");
    PerfStatsCollector::recordTimingNs(
        "forward_graph",
        "segmented_replay_total",
        750000,
        "decode",
        "cuda:0",
        {{"context", "mtp_decode_sidecar"}});
    PerfStatsCollector::addCounter("kernel", "gemm_calls", 3.0, "decode");

    ASSERT_TRUE(PerfStatsCollector::flushFromEnv());

    const auto json = nlohmann::json::parse(readFile(json_path));
    ASSERT_EQ(json.at("records").size(), 2u);

    bool saw_mtp_verifier = false;
    bool saw_graph_replay = false;
    for (const auto &record : json.at("records"))
    {
        const std::string domain = record.at("domain").get<std::string>();
        const std::string name = record.at("name").get<std::string>();
        EXPECT_NE(domain, "kernel");
        saw_mtp_verifier = saw_mtp_verifier ||
                           (domain == "mtp" && name == "verifier_forward");
        saw_graph_replay = saw_graph_replay ||
                           (domain == "forward_graph" && name == "segmented_replay_total");
    }
    EXPECT_TRUE(saw_mtp_verifier);
    EXPECT_TRUE(saw_graph_replay);

    const std::string csv = readFile(csv_path);
    EXPECT_NE(csv.find("timer,mtp,verifier_forward,decode,cuda:0"), std::string::npos);
    EXPECT_NE(csv.find("timer,forward_graph,segmented_replay_total,decode,cuda:0"), std::string::npos);
    EXPECT_EQ(csv.find("kernel"), std::string::npos);

    std::filesystem::remove(json_path);
    std::filesystem::remove(csv_path);
}
