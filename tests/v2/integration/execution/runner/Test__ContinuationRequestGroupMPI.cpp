/**
 * @file Test__ContinuationRequestGroupMPI.cpp
 * @brief Real-MPI proofs of consumer-only prompt extents and ticket-command order.
 *
 * PMPI observes the actual collective count, datatype, root and communicator
 * size, independently of PerfStats. Tests cover root-only continuation, dense
 * world consumers, proper subsets, nonzero root remapping and large-to-small
 * prompts. Expert followers retain real sparse ticket completion across fresh
 * commands without receiving token storage or inserting a barrier.
 * No model weights or accelerator initialization are required.
 */
#include "execution/runner/ContinuationRequestGroup.h"
#include "execution/runner/OrchestrationRunner.h"
#include "execution/moe/MoEOverlayInferenceTransactionService.h"
#include "utils/MPIContext.h"
#include "utils/PerfStatsCollector.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <string>

namespace
{
    /** Exact endpoint extent observed at the real MPI broadcast boundary. */
    struct BroadcastExtent
    {
        int elements;
        int element_bytes;
        int peers;
        int group_root;
    };

    thread_local bool observe_broadcasts = false;
    thread_local std::vector<BroadcastExtent> observed_broadcasts;

    /** @brief Limit passive MPI observation to the intended production calls. */
    class ScopedBroadcastObservation final
    {
    public:
        /** @brief Start one trace, excluding setup and teardown collectives. */
        ScopedBroadcastObservation()
        {
            observed_broadcasts.clear();
            observe_broadcasts = true;
        }
        /** @brief Leave MPI profiling dormant for the next test's admission. */
        ~ScopedBroadcastObservation() { observe_broadcasts = false; }
        ScopedBroadcastObservation(const ScopedBroadcastObservation &) = delete;
        ScopedBroadcastObservation &operator=(const ScopedBroadcastObservation &) = delete;
    };
}

/**
 * @brief Observe actual bytes/participants through MPI's public profiling ABI.
 * @param buffer Actual send/receive storage passed by production transport.
 * @param count Live element count, not backing capacity.
 * @param datatype Wire element representation.
 * @param root Root in this communicator, which may differ from execution rank.
 * @param comm Actual recipient communicator.
 * @return Unchanged native transport status.
 */
extern "C" int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
                         int root, MPI_Comm comm)
{
    if (observe_broadcasts)
    {
        int peers = 0, bytes = 0;
        PMPI_Comm_size(comm, &peers);
        PMPI_Type_size(datatype, &bytes);
        observed_broadcasts.push_back({count, bytes, peers, root});
    }
    return PMPI_Bcast(buffer, count, datatype, root, comm);
}

namespace llaminar2::test
{
    namespace
    {
        /** @return Borrowed real execution membership for this model-free proof. */
        std::shared_ptr<MPIContext> context()
        {
            int rank = -1, size = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);
            return std::make_shared<MPIContext>(rank, size, MPI_COMM_WORLD);
        }

        /**
         * @brief Prove actual prompt extents and next-command order for one frozen group.
         * @param consumers Exact resolved continuation ranks in execution order.
         * @param root Root selected from the continuation membership.
         */
        void provePayloads(const std::vector<int> &consumers, int root)
        {
            const auto mpi = context();
            ContinuationRequestGroup group(mpi, consumers);
            const bool consumer = group.role() == ContinuationRequestGroup::Role::TokenConsumer;
            if (consumer)
            {
                const auto comm = group.coordinationCommunicator();
                if (consumers.size() == 1u)
                    EXPECT_EQ(comm, MPI_COMM_NULL);
                else
                {
                    int size = 0;
                    ASSERT_EQ(MPI_Comm_size(comm, &size), MPI_SUCCESS);
                    EXPECT_EQ(size, static_cast<int>(consumers.size()));
                }
            }
            else
                EXPECT_THROW((void)group.coordinationCommunicator(), std::logic_error);

            PerfStatsCollector::reset();
            const ScopedBroadcastObservation observation;
            std::size_t trace_cursor = 0;
            std::size_t total_tokens = 0;
            for (int cycle = 0; cycle != 20; ++cycle)
                for (int count : {8192, 1, 31, 65})
                {
                    total_tokens += static_cast<std::size_t>(count);
                    int32_t command = static_cast<int32_t>(OrchestrationRunner::MPICommand::PREFILL);
                    mpi->broadcast_int32(&command, 1, root);
                    EXPECT_EQ(command, static_cast<int32_t>(OrchestrationRunner::MPICommand::PREFILL));
                    std::array<int32_t, 3> header{count, cycle, count - 1};
                    mpi->broadcast_int32(header.data(), header.size(), root);
                    EXPECT_EQ(header, (std::array<int32_t, 3>{count, cycle, count - 1}));
                    std::vector<int32_t> expected(static_cast<std::size_t>(count));
                    std::iota(expected.begin(), expected.end(), cycle * 10000);
                    if (mpi->rank() == root)
                        group.publishPrompt(expected, root);
                    else
                    {
                        const auto received = group.receivePrompt(static_cast<std::size_t>(count), root);
                        if (consumer)
                            EXPECT_EQ(received, expected);
                        else
                        {
                            EXPECT_TRUE(received.empty());
                            EXPECT_EQ(received.capacity(), 0u);
                        }
                    }

                    // The very next world exchange must still be DECODE_STEP,
                    // never a leftover token receive or a repair barrier.
                    command = static_cast<int32_t>(OrchestrationRunner::MPICommand::DECODE_STEP);
                    mpi->broadcast_int32(&command, 1, root);
                    EXPECT_EQ(command, static_cast<int32_t>(OrchestrationRunner::MPICommand::DECODE_STEP));
                    header = {15, cycle, count};
                    mpi->broadcast_int32(header.data(), header.size(), root);
                    EXPECT_EQ(header, (std::array<int32_t, 3>{15, cycle, count}));

                    const bool payload = consumer && consumers.size() > 1u;
                    const std::size_t calls = payload ? 5u : 4u;
                    ASSERT_EQ(observed_broadcasts.size() - trace_cursor, calls);
                    for (std::size_t call = 0; call != calls; ++call)
                    {
                        const auto &extent = observed_broadcasts[trace_cursor + call];
                        const bool token_call = payload && call == 2u;
                        const auto metadata_call = payload && call > 2u ? call - 1u : call;
                        EXPECT_EQ(extent.elements, token_call ? count : (metadata_call % 2u == 0u ? 1 : 3));
                        EXPECT_EQ(extent.element_bytes, sizeof(int32_t));
                        EXPECT_EQ(extent.peers, token_call ? static_cast<int>(consumers.size()) : mpi->world_size());
                        EXPECT_EQ(extent.group_root, token_call
                            ? static_cast<int>(std::lower_bound(consumers.begin(), consumers.end(), root) - consumers.begin())
                            : root);
                    }
                    trace_cursor += calls;
                }

            // Observation mirrors, rather than determines, the actual transport.
            const auto records = PerfStatsCollector::snapshot({"orchestration_command"});
            ASSERT_EQ(records.size(), 1u);
            EXPECT_EQ(records.front().name, "prompt_payload_bytes");
            EXPECT_EQ(records.front().count, 80u);
            const std::size_t copies = !consumer ? 0u
                : (mpi->rank() == root ? consumers.size() - 1u : 1u);
            EXPECT_EQ(records.front().value, static_cast<double>(total_tokens * sizeof(int32_t) * copies));
        }

        /** @brief CPU completion witness for the unchanged production sparse ticket protocol. */
        class TicketExecutor final : public IMoEOverlayInferenceTransactionExecutor
        {
        public:
            /** @brief Authenticate and count one execution, without model state. */
            bool executeMoEOverlayInferenceTransaction(
                const MoEOverlayInferenceTransactionTicket &ticket, std::string *error) override
            {
                if (!ticket.valid() || ticket.command_id < last_command)
                {
                    if (error) *error = "Out-of-order or malformed ticket";
                    return false;
                }
                last_command = ticket.command_id;
                ++executions;
                return true;
            }
            std::uint64_t last_command = 0;
            std::uint64_t executions = 0;
        };
    }

    TEST(ContinuationRequestGroupMPI, SingleConsumerExcludesExpertFollowers)
    {
        ASSERT_EQ(context()->world_size(), 2);
        for (int root : {0, 1}) provePayloads({root}, root);
    }

    TEST(ContinuationRequestGroupMPI, OrdinaryWorldConsumersRetainExactLivePayload)
    {
        const auto mpi = context();
        ASSERT_EQ(mpi->world_size(), 2);
        for (int root : {0, 1}) provePayloads({0, 1}, root);
    }

    TEST(ContinuationRequestGroupMPI, SubgroupExcludesExpertsAndRemapsNonzeroRoot)
    {
        ASSERT_EQ(context()->world_size(), 3);
        for (int root : {0, 2}) provePayloads({0, 2}, root);
    }

    TEST(ContinuationRequestGroupMPI, MetadataOnlyFollowersCompleteUnbarrieredTicketCommands)
    {
        const auto mpi = context();
        ASSERT_EQ(mpi->world_size(), 2);
        for (int root : {0, 1})
        {
            ContinuationRequestGroup group(mpi, {root});
            auto channel = std::make_shared<MoEOverlayMPIInferenceTransactionChannel>(
                MoEOverlayMPIInferenceTransactionChannel::Config{
                    .mpi_ctx = mpi, .source_world_rank = root, .target_world_rank = 1 - root,
                    .send_slot_count = 2});
            const MoEOverlayInferenceTransactionProtocol::Config protocol{
                .topology = {.workspace_generation = 73, .topology_fingerprint_low = 123,
                             .topology_fingerprint_high = 456, .source_world_rank = root,
                             .target_world_rank = 1 - root},
                .slot_count = 2, .max_request_count = 1,
                .max_rows_per_request = 16, .max_mtp_draft_depth = 15};
            TicketExecutor executor;
            std::unique_ptr<MoEOverlayInferenceTransactionPublisher> publisher;
            std::unique_ptr<MoEOverlayInferenceTransactionFollower> follower;
            if (mpi->rank() == root)
                publisher = std::make_unique<MoEOverlayInferenceTransactionPublisher>(
                    MoEOverlayInferenceTransactionPublisher::Config{.channel = channel, .protocol = protocol});
            else
                follower = std::make_unique<MoEOverlayInferenceTransactionFollower>(
                    MoEOverlayInferenceTransactionFollower::Config{.channel = channel, .executor = &executor,
                                                                   .protocol = protocol});
            const ScopedBroadcastObservation observation;
            for (std::uint64_t iteration = 1; iteration <= 256; ++iteration)
            {
                const int executions = static_cast<int>(iteration % 4u);
                const bool prefill = iteration % 2u == 1u;
                int32_t command = static_cast<int32_t>(prefill
                    ? OrchestrationRunner::MPICommand::PREFILL : OrchestrationRunner::MPICommand::DECODE_STEP);
                mpi->broadcast_int32(&command, 1, root);
                const int32_t count = iteration % 3u == 0u ? 8192 : 1;
                std::array<int32_t, 3> header{count, static_cast<int32_t>(iteration), executions};
                mpi->broadcast_int32(header.data(), header.size(), root);
                EXPECT_EQ(header[1], iteration);
                if (prefill)
                {
                    if (publisher)
                        group.publishPrompt(std::vector<int32_t>(static_cast<std::size_t>(count), 42), root);
                    else
                    {
                        const auto tokens = group.receivePrompt(static_cast<std::size_t>(count), root);
                        EXPECT_TRUE(tokens.empty());
                        EXPECT_EQ(tokens.capacity(), 0u);
                    }
                }
                if (publisher)
                {
                    std::string error;
                    ASSERT_TRUE(publisher->beginCommand({iteration, iteration, iteration}, &error)) << error;
                    for (int index = 0; index != executions; ++index)
                    {
                        const auto ticket = publisher->publish({
                            .graph_role = MoEOverlayInferenceGraphRole::MTPGroupedVerifier,
                            .logical_step_id = iteration * 4u + static_cast<unsigned>(index),
                            .placement_epoch = iteration, .request_count = 1,
                            .logical_rows_per_request = 16, .physical_rows_per_request = 16,
                            .draft_depth = 15, .sidecar_depth = -1});
                        ASSERT_TRUE(ticket.ok) << ticket.error;
                        ASSERT_TRUE(publisher->retire(ticket, &error)) << error;
                    }
                    ASSERT_TRUE(publisher->complete(iteration, 0u, &error)) << error;
                }
                else
                {
                    const auto result = follower->runOneCommand();
                    ASSERT_TRUE(result.ok) << result.error;
                    EXPECT_EQ(result.executed_transactions, executions);
                }
            }
            // Exactly tag+header per command: no prompt payload, artificial
            // barrier or world-wide terminal collective was added by the fix.
            EXPECT_EQ(observed_broadcasts.size(), 512u);
            if (follower) EXPECT_EQ(executor.executions, 384u);
        }
    }
}
