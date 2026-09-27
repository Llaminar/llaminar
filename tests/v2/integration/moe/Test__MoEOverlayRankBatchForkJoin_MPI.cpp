/**
 * @file Test__MoEOverlayRankBatchForkJoin_MPI.cpp
 * @brief Real-MPI witness that independent sparse peers start before return waits.
 *
 * Two followers exchange a test-only receipt after receiving their dispatches,
 * then wait for a receipt from the root's local expert compute before returning.
 * A serialized root deadlocks at the first return; the production graph fork/join
 * submits both peers and completes local work before waiting, while preserving
 * exact ordered output consumption. No model, GPU, sleep, or performance gate.
 */
#include "execution/moe/MoEOverlayRankBatchGraphSchedule.h"
#include "execution/moe/MoEOverlayRankBatchTransport.h"
#include "../../mocks/MockComputeStage.h"
#include "utils/MPIContext.h"

#include <gtest/gtest.h>
#include <mpi.h>
#include <algorithm>
#include <array>
#include <memory>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        /** @brief Independent packet, wire and asynchronous-send ownership for one peer. */
        struct PeerLane
        {
            MoEOverlayCollectiveWorkspace packets;
            std::unique_ptr<MoEOverlayMPIRankBatchTransport> transport;
            MoEOverlaySparseRows dispatch;
            MoEOverlayReturnRows returned;

            /** @brief Allocate stable CPU storage before graph execution. */
            PeerLane(const std::shared_ptr<MPIContext> &mpi, int peer)
            {
                packets.ensureCapacity(1, 1, 4, 1, DeviceId::cpu());
                dispatch = mpi->rank() == 0 ? packets.localExpertInput(0, 0) : packets.dispatchReceive(0, 0);
                returned = mpi->rank() == 0 ? packets.returnReceive(0, 0) : packets.localExpertOutput(0, 0);
                auto wire = std::make_shared<MoEOverlayRankBatchWireWorkspace>(
                    MoEOverlayRankBatchWireWorkspace::Config{
                        .participant_ids = {peer}, .max_total_rows = 1,
                        .max_total_entries = 1, .d_model = 4, .top_k = 1});
                transport = std::make_unique<MoEOverlayMPIRankBatchTransport>(
                    MoEOverlayMPIRankBatchTransport::Config{
                        .mpi_ctx = mpi, .source_world_rank = 0, .target_world_rank = peer,
                        .workspace = std::move(wire), .transaction_slot_count = 16,
                        .asynchronous_send_slot_count = 1});
            }
        };
    }

    /** @test Remote dispatch and local compute precede all waits, including slot reuse/MTP. */
    TEST(Test__MoEOverlayRankBatchForkJoin_MPI, AllPeersAndLocalWorkStartBeforeAnyReturn)
    {
        int rank = -1;
        int world_size = 0;
        ASSERT_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &rank), MPI_SUCCESS);
        ASSERT_EQ(MPI_Comm_size(MPI_COMM_WORLD, &world_size), MPI_SUCCESS);
        ASSERT_EQ(world_size, 3);
        auto mpi = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);
        std::array<std::unique_ptr<PeerLane>, 3> lanes;
        for (int peer = 1; peer <= 2; ++peer)
            if (rank == 0 || rank == peer)
                lanes[peer] = std::make_unique<PeerLane>(mpi, peer);

        for (int round = 0; round < 24; ++round)
        {
            SCOPED_TRACE(round);
            const auto key = [&](int peer, MoEOverlayCollectiveDirection direction) {
                return round % 3 == 0
                    ? makeMoEOverlayRankBatchKey(91, round + 1, ExpertHistogramSource::PrefillChunk,
                                                0, 0, 0, 0, peer, direction)
                    : makeMTPMoEOverlayRankBatchKey(91, round + 1, round % 3 == 1 ? 2 : 15,
                                                   0, 0, 0, 0, peer, direction);
            };
            if (rank == 0)
            {
                ComputeGraph graph;
                std::vector<MoEOverlayRankBatchGraphLane> nodes;
                std::vector<int> consumed;
                for (int peer = 1; peer <= 2; ++peer)
                {
                    auto &packet = lanes[peer]->dispatch;
                    packet.source_participant = 0;
                    packet.target_participant = peer;
                    packet.residency_epoch = round + 1;
                    packet.live_row_count = packet.live_entry_count = 1;
                    packet.row_ids_host[0] = 0;
                    packet.entry_offsets_host[0] = 0;
                    packet.entry_offsets_host[1] = 1;
                    packet.expert_ids_host[0] = peer;
                    packet.route_weights_host[0] = 1.0f;
                    std::fill_n(packet.hidden_rows_fp32, 4, static_cast<float>(round + peer));
                    auto send = std::make_unique<testing::MockComputeStage>(ComputeStageType::MOE_RANK_BATCH_DISPATCH);
                    send->setOnExecute([&, peer](IDeviceContext *) {
                        const std::array<const MoEOverlaySparseRows *, 1> rows{&lanes[peer]->dispatch};
                        const auto result = lanes[peer]->transport->exchangeDispatch(
                            key(peer, MoEOverlayCollectiveDirection::Dispatch), rows, {});
                        ASSERT_TRUE(result.ok) << result.error;
                    });
                    auto receive = std::make_unique<testing::MockComputeStage>(ComputeStageType::MOE_RANK_BATCH_RETURN_REDUCE);
                    receive->setOnExecute([&, peer](IDeviceContext *) {
                        const std::array<MoEOverlayReturnRows *, 1> rows{&lanes[peer]->returned};
                        const auto result = lanes[peer]->transport->exchangeReturn(
                            key(peer, MoEOverlayCollectiveDirection::ReturnReduce), {}, rows);
                        ASSERT_TRUE(result.ok) << result.error;
                        EXPECT_FLOAT_EQ(rows[0]->output_rows_fp32[3], static_cast<float>(round + peer));
                        consumed.push_back(peer);
                    });
                    const auto suffix = std::to_string(peer);
                    nodes.push_back({"dispatch" + suffix, "return" + suffix});
                    graph.addNode(nodes.back().dispatch, std::move(send), DeviceId::cpu());
                    graph.addNode(nodes.back().returned, std::move(receive), DeviceId::cpu());
                    graph.addDependency(nodes.back().returned, nodes.back().dispatch);
                }
                auto local_dispatch = std::make_unique<testing::MockComputeStage>(ComputeStageType::MOE_SPARSE_DISPATCH);
                auto local_compute = std::make_unique<testing::MockComputeStage>(ComputeStageType::MOE_LOCAL_EXPERT);
                local_compute->setOnExecute([&](IDeviceContext *) {
                    // Each remote return depends on this local-compute receipt.
                    // Reintroducing return-before-local-work creates a real MPI
                    // deadlock, not a throughput/noisy-clock assertion.
                    for (int peer = 1; peer <= 2; ++peer)
                        ASSERT_EQ(MPI_Send(&round, 1, MPI_INT, peer, 19008, MPI_COMM_WORLD), MPI_SUCCESS);
                });
                auto local_return = std::make_unique<testing::MockComputeStage>(ComputeStageType::MOE_SPARSE_RETURN_REDUCE);
                local_return->setOnExecute([&](IDeviceContext *) { consumed.push_back(0); });
                graph.addNode("local_dispatch", std::move(local_dispatch), DeviceId::cpu());
                graph.addNode("local_compute", std::move(local_compute), DeviceId::cpu());
                graph.addNode("local_return", std::move(local_return), DeviceId::cpu());
                const std::array<MoEOverlayLocalExpertGraphLane, 1> local{{
                    {"local_dispatch", "local_compute", "local_return"}}};
                wireMoEOverlayRankBatchForkJoin(graph, nodes, local);
                for (auto *stage : graph.getExecutionStages())
                    ASSERT_TRUE(stage->execute(nullptr)); // Test stage bodies invoke the real transport.
                EXPECT_EQ(consumed, (std::vector<int>{1, 2, 0}));
            }
            else
            {
                auto &lane = *lanes[rank];
                const std::array<MoEOverlaySparseRows *, 1> incoming{&lane.dispatch};
                auto result = lane.transport->exchangeDispatch(
                    key(rank, MoEOverlayCollectiveDirection::Dispatch), {}, incoming);
                ASSERT_TRUE(result.ok) << result.error;
                // Neither follower may return until both have received work.
                // This is an adversarial test edge, not an inference collective.
                int observed = -1;
                ASSERT_EQ(MPI_Sendrecv(&round, 1, MPI_INT, 3 - rank, 19007,
                    &observed, 1, MPI_INT, 3 - rank, 19007, MPI_COMM_WORLD, MPI_STATUS_IGNORE), MPI_SUCCESS);
                ASSERT_EQ(observed, round);
                ASSERT_EQ(MPI_Recv(&observed, 1, MPI_INT, 0, 19008,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE), MPI_SUCCESS);
                ASSERT_EQ(observed, round);
                lane.returned.source_participant = rank;
                lane.returned.target_participant = 0;
                lane.returned.residency_epoch = lane.dispatch.residency_epoch;
                lane.returned.live_row_count = 1;
                lane.returned.row_ids_host[0] = 0;
                std::copy_n(lane.dispatch.hidden_rows_fp32, 4, lane.returned.output_rows_fp32);
                const std::array<const MoEOverlayReturnRows *, 1> outgoing{&lane.returned};
                result = lane.transport->exchangeReturn(
                    key(rank, MoEOverlayCollectiveDirection::ReturnReduce), outgoing, {});
                ASSERT_TRUE(result.ok) << result.error;
            }
        }
    }
}
