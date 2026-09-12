/**
 * @file Test__MoEOverlaySparseTransport_MPI.cpp
 * @brief Real-MPI integration tests for scalar and rank-batched sparse MoE transport.
 *
 * The rank-batch case models several accelerator participants on one socket
 * and proves they cross the MPI boundary in one authenticated envelope in each
 * direction while retaining exact row, weight, epoch, and participant identity.
 */

#include "execution/moe/MoEOverlaySparseCollective.h"
#include "execution/moe/MoEOverlayCanonicalHostReturn.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/compute_stages/stages/MoESparseReturnReduceStage.h"
#include "tensors/Tensors.h"
#include "../../mocks/MockComputeStage.h"
#include "execution/moe/MoEOverlayInferenceTransactionService.h"
#include "execution/moe/MoEOverlayRankBatchTransport.h"
#include "execution/moe/MoESparseRequestIdentity.h"
#include "utils/MPIContext.h"
#include "utils/PerfStatsCollector.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>

namespace llaminar2::test
{
    namespace
    {
        /** @brief Enable one PerfStats domain without leaking environment into later tests. */
        class ScopedTransportPerfStats final
        {
        public:
            /** @brief Enable only the requested production evidence family. */
            explicit ScopedTransportPerfStats(
                const char *filter = "moe_overlay_transport")
            {
                remember("LLAMINAR_PERF_STATS_SUMMARY", &old_summary_);
                remember("LLAMINAR_PERF_STATS_FILTER", &old_filter_);
                ::setenv("LLAMINAR_PERF_STATS_SUMMARY", "1", 1);
                ::setenv(
                    "LLAMINAR_PERF_STATS_FILTER",
                    filter,
                    1);
                PerfStatsCollector::reset();
            }

            ~ScopedTransportPerfStats()
            {
                PerfStatsCollector::reset();
                restore("LLAMINAR_PERF_STATS_SUMMARY", old_summary_);
                restore("LLAMINAR_PERF_STATS_FILTER", old_filter_);
            }

            ScopedTransportPerfStats(const ScopedTransportPerfStats &) = delete;
            ScopedTransportPerfStats &operator=(
                const ScopedTransportPerfStats &) = delete;

        private:
            /** @brief Copy an optional process environment value before mutation. */
            static void remember(
                const char *name,
                std::optional<std::string> *destination)
            {
                if (const char *value = ::getenv(name))
                    *destination = value;
            }

            /** @brief Restore an original environment value or remove the test value. */
            static void restore(
                const char *name,
                const std::optional<std::string> &value)
            {
                if (value)
                    ::setenv(name, value->c_str(), 1);
                else
                    ::unsetenv(name);
            }

            std::optional<std::string> old_summary_;
            std::optional<std::string> old_filter_;
        };

        /** @brief Device-free retained-graph witness used by the MPI control test. */
        class RecordingTransactionExecutor final
            : public IMoEOverlayInferenceTransactionExecutor
        {
        public:
            /** @brief Record the already-authenticated ticket in receive order. */
            bool executeMoEOverlayInferenceTransaction(
                const MoEOverlayInferenceTransactionTicket &ticket,
                std::string *error) override
            {
                if (!ticket.valid() ||
                    ticket.action !=
                        MoEOverlayInferenceTransactionAction::Execute)
                {
                    if (error)
                        *error = "Executor received a malformed control ticket";
                    return false;
                }
                tickets.push_back(ticket);
                return true;
            }

            std::vector<MoEOverlayInferenceTransactionTicket> tickets;
        };
    } // namespace

    class Test__MoEOverlaySparseTransport_MPI : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

            if (world_size_ < 2)
                GTEST_SKIP() << "Test requires at least 2 MPI ranks";

            mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);
            collective_ = std::make_unique<MoEOverlayMPISparseCollectiveContext>(
                MoEOverlayMPISparseCollectiveContext::Config{
                    .mpi_ctx = mpi_ctx_,
                    .local_participant_ids = {rank_},
                });

            workspace_.ensureCapacity(16, 32, 4, 2, DeviceId::cpu());
            workspace_.resetForStep(1, 0);
        }

        MoEOverlayCollectiveKey dispatchKey() const
        {
            MoEOverlayCollectiveKey key;
            key.generation_id = 1;
            key.step_id = 0;
            key.layer_idx = 3;
            key.tier_idx = 1;
            key.domain_id = 9;
            key.direction = MoEOverlayCollectiveDirection::Dispatch;
            key.sequence = 41;
            return key;
        }

        MoEOverlayCollectiveKey returnKey() const
        {
            auto key = dispatchKey();
            key.direction = MoEOverlayCollectiveDirection::ReturnReduce;
            key.sequence = 42;
            return key;
        }

        MoEOverlayCollectiveKey mtpDispatchKey() const
        {
            return makeMTPMoEOverlayCollectiveKey(
                1,
                7,
                2,
                3,
                1,
                9,
                1,
                MoEOverlayCollectiveDirection::Dispatch);
        }

        MoEOverlayCollectiveKey mtpReturnKey() const
        {
            return makeMTPMoEOverlayCollectiveKey(
                1,
                7,
                2,
                3,
                1,
                9,
                0,
                MoEOverlayCollectiveDirection::ReturnReduce);
        }

        int rank_ = -1;
        int world_size_ = 0;
        std::shared_ptr<IMPIContext> mpi_ctx_;
        MoEOverlayCollectiveWorkspace workspace_;
        std::unique_ptr<MoEOverlayMPISparseCollectiveContext> collective_;
    };

    TEST_F(Test__MoEOverlaySparseTransport_MPI, DispatchAndReturnMoveCompactRowsByKey)
    {
        auto dispatch_key = dispatchKey();

        auto outbound_dispatch = workspace_.localExpertInput(dispatch_key.layer_idx, dispatch_key.tier_idx);
        outbound_dispatch.key = dispatch_key;
        outbound_dispatch.source_participant = rank_;
        outbound_dispatch.target_participant = (rank_ == 0) ? 1 : 0;

        if (rank_ == 0)
        {
            outbound_dispatch.residency_epoch = 7;
            outbound_dispatch.live_row_count = 2;
            outbound_dispatch.live_entry_count = 4;
            outbound_dispatch.row_ids_host[0] = 10;
            outbound_dispatch.row_ids_host[1] = 12;
            outbound_dispatch.entry_offsets_host[0] = 0;
            outbound_dispatch.entry_offsets_host[1] = 2;
            outbound_dispatch.entry_offsets_host[2] = 4;
            outbound_dispatch.expert_ids_host[0] = 3;
            outbound_dispatch.expert_ids_host[1] = 5;
            outbound_dispatch.expert_ids_host[2] = 7;
            outbound_dispatch.expert_ids_host[3] = 9;
            outbound_dispatch.route_weights_host[0] = 0.6f;
            outbound_dispatch.route_weights_host[1] = 0.4f;
            outbound_dispatch.route_weights_host[2] = 0.3f;
            outbound_dispatch.route_weights_host[3] = 0.7f;
            for (size_t index = 0; index < outbound_dispatch.live_row_count * static_cast<size_t>(outbound_dispatch.d_model); ++index)
                outbound_dispatch.hidden_rows_fp32[index] = static_cast<float>(100 + index);
        }
        else
        {
            outbound_dispatch.live_row_count = 0;
            outbound_dispatch.live_entry_count = 0;
            outbound_dispatch.entry_offsets_host[0] = 0;
        }

        auto inbound_dispatch = workspace_.dispatchReceive(dispatch_key.layer_idx, dispatch_key.tier_idx);
        auto dispatch_result = collective_->dispatch(dispatch_key, outbound_dispatch, &inbound_dispatch, nullptr);
        ASSERT_TRUE(dispatch_result.ok) << dispatch_result.error;
        EXPECT_TRUE(dispatch_result.collective_complete);
        EXPECT_EQ(inbound_dispatch.key, dispatch_key);

        if (rank_ == 1)
        {
            EXPECT_EQ(inbound_dispatch.live_row_count, 2u);
            EXPECT_EQ(inbound_dispatch.live_entry_count, 4u);
            EXPECT_EQ(inbound_dispatch.row_ids_host[0], 10);
            EXPECT_EQ(inbound_dispatch.row_ids_host[1], 12);
            EXPECT_EQ(inbound_dispatch.entry_offsets_host[0], 0);
            EXPECT_EQ(inbound_dispatch.entry_offsets_host[1], 2);
            EXPECT_EQ(inbound_dispatch.entry_offsets_host[2], 4);
            EXPECT_EQ(inbound_dispatch.expert_ids_host[3], 9);
            EXPECT_FLOAT_EQ(inbound_dispatch.route_weights_host[2], 0.3f);
            EXPECT_FLOAT_EQ(inbound_dispatch.hidden_rows_fp32[7], 107.0f);
        }
        else
        {
            EXPECT_EQ(inbound_dispatch.live_row_count, 0u);
            EXPECT_EQ(inbound_dispatch.live_entry_count, 0u);
        }

        auto stale_dispatch = collective_->dispatch(dispatch_key, outbound_dispatch, &inbound_dispatch, nullptr);
        EXPECT_FALSE(stale_dispatch.ok);

        auto return_key = returnKey();
        auto outbound_return = workspace_.localExpertOutput(return_key.layer_idx, return_key.tier_idx);
        outbound_return.key = return_key;
        outbound_return.source_participant = rank_;
        outbound_return.target_participant = (rank_ == 1) ? 0 : 1;

        if (rank_ == 1)
        {
            outbound_return.residency_epoch = 7;
            outbound_return.live_row_count = 2;
            outbound_return.row_ids_host[0] = 10;
            outbound_return.row_ids_host[1] = 12;
            for (size_t index = 0; index < outbound_return.live_row_count * static_cast<size_t>(outbound_return.d_model); ++index)
                outbound_return.output_rows_fp32[index] = static_cast<float>(200 + index);
        }
        else
        {
            outbound_return.live_row_count = 0;
        }

        auto inbound_return = workspace_.returnReceive(return_key.layer_idx, return_key.tier_idx);
        auto return_result = collective_->returnReduce(return_key, outbound_return, &inbound_return, nullptr);
        ASSERT_TRUE(return_result.ok) << return_result.error;
        EXPECT_TRUE(return_result.collective_complete);
        EXPECT_EQ(inbound_return.key, return_key);

        if (rank_ == 0)
        {
            EXPECT_EQ(inbound_return.live_row_count, 2u);
            EXPECT_EQ(inbound_return.row_ids_host[0], 10);
            EXPECT_EQ(inbound_return.row_ids_host[1], 12);
            EXPECT_FLOAT_EQ(inbound_return.output_rows_fp32[0], 200.0f);
            EXPECT_FLOAT_EQ(inbound_return.output_rows_fp32[7], 207.0f);
        }
        else
        {
            EXPECT_EQ(inbound_return.live_row_count, 0u);
        }

        auto stale_return = collective_->returnReduce(return_key, outbound_return, &inbound_return, nullptr);
        EXPECT_FALSE(stale_return.ok);
    }

    /** @brief Every two-rank expert placement preserves the serial FP32 fold. */
    TEST_F(Test__MoEOverlaySparseTransport_MPI, CanonicalReturnIsPlacementInvariantAcrossRanks)
    {
        constexpr int columns = 4;
        const std::array<float, 4> values{0x1p24f, 1.0f, -0x1p24f, 1.0f};
        FP32Tensor bank(std::vector<size_t>{4 * (columns + 1) + 1});
        FP32Tensor weights(std::vector<size_t>{1, 4});
        FP32Tensor output(std::vector<size_t>{1, columns});
        std::fill_n(weights.mutable_data(), 4, 1.0f);
        MoECanonicalRouteReduceStage::Params params;
        params.device_id = DeviceId::cpu();
        params.canonical_route_contributions = &bank;
        params.routing_weights = &weights;
        params.output = &output;
        params.seq_len = 1;
        params.top_k = 4;
        params.d_model = columns;
        params.canonical_route_arithmetic = MoECanonicalRouteArithmeticPolicy::UnweightedExpertRowThenOrderedFMA;
        params.canonical_route_layout = MoECanonicalRoutePublicationLayout::PackedIndexedRouteRows;
        params.reduction_role = MoECanonicalRouteReductionRole::RootOwner;
        MoECanonicalRouteReduceStage reducer(params);
        llaminar2::testing::MockDeviceContext context(DeviceId::cpu(), ComputeBackendType::CPU);

        for (int placement = 0; placement < 16; ++placement)
        {
            const auto key = makeMoEOverlayCollectiveKey(
                71, static_cast<uint64_t>(placement + 1), 0, 0, 0, 0,
                MoEOverlayCollectiveDirection::ReturnReduce);
            auto outbound = workspace_.localExpertOutput(0, 0);
            outbound.key = key;
            outbound.residency_epoch = static_cast<uint64_t>(placement + 1);
            outbound.source_participant = rank_;
            outbound.target_participant = 0;
            outbound.layout = MoEOverlayReturnLayout::CanonicalExpertRoutes;
            for (int slot = 0; slot < 4; ++slot)
            {
                if (((placement >> slot) & 1) != rank_) continue;
                const size_t row = outbound.live_row_count++;
                outbound.row_ids_host[row] = slot;
                std::fill_n(outbound.output_rows_fp32 + row * columns, columns, values[slot]);
            }
            auto inbound = workspace_.returnReceive(0, 0);
            MoESparseReturnReduceStage::Params returned;
            returned.device_id = DeviceId::cpu();
            returned.collective_context = collective_.get();
            returned.key = key;
            returned.source_participant = rank_;
            returned.target_participant = 0;
            returned.outbound_rows = &outbound;
            returned.inbound_rows = &inbound;
            returned.dense_output = rank_ == 0 ? &bank : nullptr;
            returned.return_layout = MoEOverlayReturnLayout::CanonicalExpertRoutes;
            returned.inbound_consumer_role = rank_ == 0
                ? MoESparseReturnReduceStage::InboundConsumerRole::ContinuationAccumulator
                : MoESparseReturnReduceStage::InboundConsumerRole::ProtocolParticipant;
            returned.seq_len = 1;
            returned.d_model = columns;
            returned.clear_output_before_scatter = rank_ == 0;
            returned.require_explicit_transaction_identity = true;
            MoESparseReturnReduceStage return_stage(returned);
            return_stage.updateMoEOverlayCollectiveRuntimeParams({
                .generation_id = 71,
                .step_id = static_cast<uint64_t>(placement + 1),
            });
            ASSERT_TRUE(return_stage.execute(&context));
            if (rank_ != 0) continue;
            ASSERT_EQ(inbound.layout, MoEOverlayReturnLayout::CanonicalExpertRoutes);
            ASSERT_EQ(inbound.live_row_count, 4u);
            ASSERT_TRUE(reducer.execute(&context));
            for (int column = 0; column < columns; ++column)
                EXPECT_EQ(output.data()[column], 1.0f) << "placement=" << placement;
        }
    }

    TEST_F(Test__MoEOverlaySparseTransport_MPI, MTPNamespacedDispatchAndReturnPreserveKeyAcrossRanks)
    {
        auto dispatch_key = mtpDispatchKey();
        ASSERT_TRUE(dispatch_key.isValid());

        auto outbound_dispatch = workspace_.localExpertInput(dispatch_key.layer_idx, dispatch_key.tier_idx);
        outbound_dispatch.key = dispatch_key;
        outbound_dispatch.source_participant = rank_;
        outbound_dispatch.target_participant = (rank_ == 0) ? 1 : 0;

        if (rank_ == 0)
        {
            outbound_dispatch.residency_epoch = 13;
            outbound_dispatch.live_row_count = 1;
            outbound_dispatch.live_entry_count = 2;
            outbound_dispatch.row_ids_host[0] = 17;
            outbound_dispatch.entry_offsets_host[0] = 0;
            outbound_dispatch.entry_offsets_host[1] = 2;
            outbound_dispatch.expert_ids_host[0] = 4;
            outbound_dispatch.expert_ids_host[1] = 8;
            outbound_dispatch.route_weights_host[0] = 0.25f;
            outbound_dispatch.route_weights_host[1] = 0.75f;
            for (int col = 0; col < outbound_dispatch.d_model; ++col)
                outbound_dispatch.hidden_rows_fp32[col] = static_cast<float>(300 + col);
        }
        else
        {
            outbound_dispatch.live_row_count = 0;
            outbound_dispatch.live_entry_count = 0;
            outbound_dispatch.entry_offsets_host[0] = 0;
        }

        auto inbound_dispatch = workspace_.dispatchReceive(dispatch_key.layer_idx, dispatch_key.tier_idx);
        const auto dispatch_result = collective_->dispatch(dispatch_key, outbound_dispatch, &inbound_dispatch, nullptr);
        ASSERT_TRUE(dispatch_result.ok) << dispatch_result.error;
        EXPECT_TRUE(dispatch_result.collective_complete);
        EXPECT_EQ(inbound_dispatch.key, dispatch_key);
        EXPECT_EQ(inbound_dispatch.key.key_namespace, MoEOverlayCollectiveNamespace::MTP);
        EXPECT_EQ(inbound_dispatch.key.mtp_depth, 2);
        EXPECT_EQ(inbound_dispatch.key.participant_id, 1);

        if (rank_ == 1)
        {
            EXPECT_EQ(inbound_dispatch.live_row_count, 1u);
            EXPECT_EQ(inbound_dispatch.live_entry_count, 2u);
            EXPECT_EQ(inbound_dispatch.row_ids_host[0], 17);
            EXPECT_EQ(inbound_dispatch.expert_ids_host[1], 8);
            EXPECT_FLOAT_EQ(inbound_dispatch.route_weights_host[1], 0.75f);
            EXPECT_FLOAT_EQ(inbound_dispatch.hidden_rows_fp32[3], 303.0f);
        }
        else
        {
            EXPECT_EQ(inbound_dispatch.live_row_count, 0u);
            EXPECT_EQ(inbound_dispatch.live_entry_count, 0u);
        }

        auto return_key = mtpReturnKey();
        ASSERT_TRUE(return_key.isValid());

        auto outbound_return = workspace_.localExpertOutput(return_key.layer_idx, return_key.tier_idx);
        outbound_return.key = return_key;
        outbound_return.source_participant = rank_;
        outbound_return.target_participant = (rank_ == 1) ? 0 : 1;

        if (rank_ == 1)
        {
            outbound_return.residency_epoch = 13;
            outbound_return.live_row_count = 1;
            outbound_return.row_ids_host[0] = 17;
            for (int col = 0; col < outbound_return.d_model; ++col)
                outbound_return.output_rows_fp32[col] = static_cast<float>(400 + col);
        }
        else
        {
            outbound_return.live_row_count = 0;
        }

        auto inbound_return = workspace_.returnReceive(return_key.layer_idx, return_key.tier_idx);
        const auto return_result = collective_->returnReduce(return_key, outbound_return, &inbound_return, nullptr);
        ASSERT_TRUE(return_result.ok) << return_result.error;
        EXPECT_TRUE(return_result.collective_complete);
        EXPECT_EQ(inbound_return.key, return_key);
        EXPECT_EQ(inbound_return.key.key_namespace, MoEOverlayCollectiveNamespace::MTP);
        EXPECT_EQ(inbound_return.key.mtp_depth, 2);
        EXPECT_EQ(inbound_return.key.participant_id, 0);

        if (rank_ == 0)
        {
            EXPECT_EQ(inbound_return.live_row_count, 1u);
            EXPECT_EQ(inbound_return.row_ids_host[0], 17);
            EXPECT_FLOAT_EQ(inbound_return.output_rows_fp32[3], 403.0f);
        }
        else
        {
            EXPECT_EQ(inbound_return.live_row_count, 0u);
        }
    }

    TEST_F(Test__MoEOverlaySparseTransport_MPI,
           FourParticipantsOnOneRemoteRankUseOneExactDispatchAndReturnEnvelope)
    {
        if (world_size_ != 2)
            GTEST_SKIP() << "Rank-batch protocol fixture requires exactly two MPI ranks";

        constexpr int kLayer = 5;
        constexpr int kTier = 1;
        constexpr int kDModel = 4;
        constexpr int kTopK = 1;
        const std::vector<int> participant_ids{10, 11, 12, 13};
        auto wire = std::make_shared<MoEOverlayRankBatchWireWorkspace>(
            MoEOverlayRankBatchWireWorkspace::Config{
                .participant_ids = participant_ids,
                .max_total_rows = participant_ids.size(),
                .max_total_entries = participant_ids.size(),
                .d_model = kDModel,
                .top_k = kTopK,
            });
        MoEOverlayMPIRankBatchTransport transport(
            {.mpi_ctx = mpi_ctx_,
             .source_world_rank = 0,
             .target_world_rank = 1,
             .workspace = wire,
             .transaction_slot_count = 64});

        std::array<MoEOverlayCollectiveWorkspace, 4> packet_storage;
        for (auto &workspace : packet_storage)
            workspace.ensureCapacity(1, 1, kDModel, kTopK, DeviceId::cpu());

        std::array<MoEOverlaySparseRows, 4> dispatch_outbound{
            packet_storage[0].localExpertInput(kLayer, kTier),
            packet_storage[1].localExpertInput(kLayer, kTier),
            packet_storage[2].localExpertInput(kLayer, kTier),
            packet_storage[3].localExpertInput(kLayer, kTier),
        };
        std::array<MoEOverlaySparseRows, 4> dispatch_inbound{
            packet_storage[0].dispatchReceive(kLayer, kTier),
            packet_storage[1].dispatchReceive(kLayer, kTier),
            packet_storage[2].dispatchReceive(kLayer, kTier),
            packet_storage[3].dispatchReceive(kLayer, kTier),
        };
        std::array<const MoEOverlaySparseRows *, 4> dispatch_outbound_views{};
        std::array<MoEOverlaySparseRows *, 4> dispatch_inbound_views{};
        for (size_t index = 0; index < participant_ids.size(); ++index)
        {
            dispatch_outbound_views[index] = &dispatch_outbound[index];
            dispatch_inbound_views[index] = &dispatch_inbound[index];
            dispatch_outbound[index].source_participant = 2;
            dispatch_outbound[index].target_participant = participant_ids[index];
            dispatch_outbound[index].residency_epoch = 100u + index;
            if (rank_ == 0)
            {
                dispatch_outbound[index].live_row_count = 1;
                dispatch_outbound[index].live_entry_count = 1;
                dispatch_outbound[index].row_ids_host[0] =
                    static_cast<int32_t>(20u + index);
                dispatch_outbound[index].entry_offsets_host[0] = 0;
                dispatch_outbound[index].entry_offsets_host[1] = 1;
                dispatch_outbound[index].expert_ids_host[0] =
                    static_cast<int32_t>(30u + index);
                dispatch_outbound[index].route_weights_host[0] =
                    0.125f * static_cast<float>(index + 1u);
                for (int column = 0; column < kDModel; ++column)
                {
                    dispatch_outbound[index].hidden_rows_fp32[column] =
                        static_cast<float>(1000u + index * 10u + column);
                }
            }
        }

        const auto dispatch_key = makeMoEOverlayRankBatchKey(
            /*generation_id=*/9,
            /*step_id=*/4,
            ExpertHistogramSource::DecodeToken,
            kLayer,
            kTier,
            /*domain_ordinal=*/3,
            /*source_world_rank=*/0,
            /*target_world_rank=*/1,
            MoEOverlayCollectiveDirection::Dispatch);
        const auto dispatch_result = rank_ == 0
                                         ? transport.exchangeDispatch(
                                               dispatch_key,
                                               dispatch_outbound_views,
                                               {})
                                         : transport.exchangeDispatch(
                                               dispatch_key,
                                               {},
                                               dispatch_inbound_views);
        ASSERT_TRUE(dispatch_result.ok) << dispatch_result.error;
        ASSERT_TRUE(dispatch_result.collective_complete);

        if (rank_ == 1)
        {
            for (size_t index = 0; index < participant_ids.size(); ++index)
            {
                const auto &packet = dispatch_inbound[index];
                EXPECT_EQ(packet.key.participant_id, participant_ids[index]);
                EXPECT_EQ(packet.key.histogram_source,
                          ExpertHistogramSource::DecodeToken);
                EXPECT_EQ(packet.residency_epoch, 100u + index);
                EXPECT_EQ(packet.source_participant, 2);
                EXPECT_EQ(packet.target_participant, participant_ids[index]);
                EXPECT_EQ(packet.live_row_count, 1u);
                EXPECT_EQ(packet.live_entry_count, 1u);
                EXPECT_EQ(packet.row_ids_host[0],
                          static_cast<int32_t>(20u + index));
                EXPECT_EQ(packet.expert_ids_host[0],
                          static_cast<int32_t>(30u + index));
                EXPECT_FLOAT_EQ(
                    packet.route_weights_host[0],
                    0.125f * static_cast<float>(index + 1u));
                EXPECT_FLOAT_EQ(
                    packet.hidden_rows_fp32[3],
                    static_cast<float>(1003u + index * 10u));
            }
        }

        std::array<MoEOverlayReturnRows, 4> return_outbound{
            packet_storage[0].localExpertOutput(kLayer, kTier),
            packet_storage[1].localExpertOutput(kLayer, kTier),
            packet_storage[2].localExpertOutput(kLayer, kTier),
            packet_storage[3].localExpertOutput(kLayer, kTier),
        };
        std::array<MoEOverlayReturnRows, 4> return_inbound{
            packet_storage[0].returnReceive(kLayer, kTier),
            packet_storage[1].returnReceive(kLayer, kTier),
            packet_storage[2].returnReceive(kLayer, kTier),
            packet_storage[3].returnReceive(kLayer, kTier),
        };
        std::array<const MoEOverlayReturnRows *, 4> return_outbound_views{};
        std::array<MoEOverlayReturnRows *, 4> return_inbound_views{};
        for (size_t index = 0; index < participant_ids.size(); ++index)
        {
            return_outbound_views[index] = &return_outbound[index];
            return_inbound_views[index] = &return_inbound[index];
            return_outbound[index].source_participant = participant_ids[index];
            return_outbound[index].target_participant = 2;
            return_outbound[index].residency_epoch = 100u + index;
            if (rank_ == 1)
            {
                return_outbound[index].live_row_count = 1;
                return_outbound[index].row_ids_host[0] =
                    static_cast<int32_t>(20u + index);
                for (int column = 0; column < kDModel; ++column)
                {
                    return_outbound[index].output_rows_fp32[column] =
                        static_cast<float>(2000u + index * 10u + column);
                }
            }
        }

        const auto return_key = makeMoEOverlayRankBatchKey(
            /*generation_id=*/9,
            /*step_id=*/4,
            ExpertHistogramSource::DecodeToken,
            kLayer,
            kTier,
            /*domain_ordinal=*/3,
            /*source_world_rank=*/0,
            /*target_world_rank=*/1,
            MoEOverlayCollectiveDirection::ReturnReduce);
        const auto return_result = rank_ == 1
                                       ? transport.exchangeReturn(
                                             return_key,
                                             return_outbound_views,
                                             {})
                                       : transport.exchangeReturn(
                                             return_key,
                                             {},
                                             return_inbound_views);
        ASSERT_TRUE(return_result.ok) << return_result.error;
        ASSERT_TRUE(return_result.collective_complete);

        if (rank_ == 0)
        {
            for (size_t index = 0; index < participant_ids.size(); ++index)
            {
                const auto &packet = return_inbound[index];
                EXPECT_EQ(packet.key.participant_id, participant_ids[index]);
                EXPECT_EQ(packet.residency_epoch, 100u + index);
                EXPECT_EQ(packet.source_participant, participant_ids[index]);
                EXPECT_EQ(packet.target_participant, 2);
                EXPECT_EQ(packet.live_row_count, 1u);
                EXPECT_EQ(packet.row_ids_host[0],
                          static_cast<int32_t>(20u + index));
                EXPECT_FLOAT_EQ(
                    packet.output_rows_fp32[3],
                    static_cast<float>(2003u + index * 10u));
            }
        }

        /* Both ranks reject an exact replay before issuing another MPI call. */
        const auto stale = rank_ == 1
                               ? transport.exchangeReturn(
                                     return_key,
                                     return_outbound_views,
                                     {})
                               : transport.exchangeReturn(
                                     return_key,
                                     {},
                                     return_inbound_views);
        EXPECT_FALSE(stale.ok);
        EXPECT_NE(stale.error.find("stale"), std::string::npos);
    }

    TEST_F(Test__MoEOverlaySparseTransport_MPI,
           AsyncSendRingWrapsAcrossSidecarAndVerifierTransactions)
    {
        if (world_size_ != 2)
            GTEST_SKIP() << "Rank-batch protocol fixture requires exactly two MPI ranks";

        ScopedTransportPerfStats perf_stats;
        constexpr int kLayer = 7;
        constexpr int kTier = 2;
        constexpr int kDModel = 4;
        constexpr int kTopK = 1;
        constexpr size_t kSendSlots = 2;
        constexpr int kRounds = 36;
        const std::vector<int> participant_ids{21};

        auto wire = std::make_shared<MoEOverlayRankBatchWireWorkspace>(
            MoEOverlayRankBatchWireWorkspace::Config{
                .participant_ids = participant_ids,
                .max_total_rows = 1,
                .max_total_entries = 1,
                .d_model = kDModel,
                .top_k = kTopK,
            });
        MoEOverlayMPIRankBatchTransport transport(
            {.mpi_ctx = mpi_ctx_,
             .source_world_rank = 0,
             .target_world_rank = 1,
             .workspace = wire,
             .transaction_slot_count = 16,
             .asynchronous_send_slot_count = kSendSlots});

        MoEOverlayCollectiveWorkspace packet_storage;
        packet_storage.ensureCapacity(
            1, 1, kDModel, kTopK, DeviceId::cpu());
        auto dispatch_outbound =
            packet_storage.localExpertInput(kLayer, kTier);
        auto dispatch_inbound =
            packet_storage.dispatchReceive(kLayer, kTier);
        auto return_outbound =
            packet_storage.localExpertOutput(kLayer, kTier);
        auto return_inbound =
            packet_storage.returnReceive(kLayer, kTier);
        const std::array<const MoEOverlaySparseRows *, 1>
            dispatch_outbound_views{&dispatch_outbound};
        const std::array<MoEOverlaySparseRows *, 1>
            dispatch_inbound_views{&dispatch_inbound};
        const std::array<const MoEOverlayReturnRows *, 1>
            return_outbound_views{&return_outbound};
        const std::array<MoEOverlayReturnRows *, 1>
            return_inbound_views{&return_inbound};

        MoESparseHostOperationSequence operations;
        constexpr std::array<int, 9> depths{0, 0, 0, 1, 2, 3, 15, 0, 0};
        for (int round = 0; round < kRounds; ++round)
        {
            // The learned sidecar stays in namespace depth zero across every
            // speculative row; only its host-issued operation changes. Mix
            // verifier depths and repeated sidecars while wrapping send slots.
            const int depth = depths[static_cast<std::size_t>(round) % depths.size()];
            const auto operation = operations.issue();
            ASSERT_TRUE(operation.has_value());
            const uint64_t step = *operation;
            const int32_t row_id = 500 + round;

            dispatch_outbound.source_participant = 4;
            dispatch_outbound.target_participant = participant_ids.front();
            dispatch_outbound.residency_epoch = 77;
            dispatch_outbound.live_row_count = 1;
            dispatch_outbound.live_entry_count = 1;
            dispatch_outbound.row_ids_host[0] = row_id;
            dispatch_outbound.entry_offsets_host[0] = 0;
            dispatch_outbound.entry_offsets_host[1] = 1;
            dispatch_outbound.expert_ids_host[0] = 30 + round;
            dispatch_outbound.route_weights_host[0] = 0.5f;
            for (int column = 0; column < kDModel; ++column)
            {
                dispatch_outbound.hidden_rows_fp32[column] =
                    static_cast<float>(10000 + round * 10 + column);
            }

            const auto dispatch_key = makeMTPMoEOverlayRankBatchKey(
                /*generation_id=*/19,
                step,
                depth,
                kLayer,
                kTier,
                /*domain_ordinal=*/5,
                /*source_world_rank=*/0,
                /*target_world_rank=*/1,
                MoEOverlayCollectiveDirection::Dispatch);
            const auto dispatch_result =
                rank_ == 0
                    ? transport.exchangeDispatch(
                          dispatch_key, dispatch_outbound_views, {})
                    : transport.exchangeDispatch(
                          dispatch_key, {}, dispatch_inbound_views);
            ASSERT_TRUE(dispatch_result.ok) << dispatch_result.error;
            ASSERT_TRUE(dispatch_result.collective_complete);

            if (rank_ == 1)
            {
                EXPECT_EQ(dispatch_inbound.key.key_namespace,
                          MoEOverlayCollectiveNamespace::MTP);
                EXPECT_EQ(dispatch_inbound.key.mtp_depth, depth);
                EXPECT_EQ(dispatch_inbound.row_ids_host[0], row_id);
                EXPECT_EQ(dispatch_inbound.expert_ids_host[0], 30 + round);
                EXPECT_FLOAT_EQ(
                    dispatch_inbound.hidden_rows_fp32[3],
                    static_cast<float>(10003 + round * 10));
            }

            return_outbound.source_participant = participant_ids.front();
            return_outbound.target_participant = 4;
            return_outbound.residency_epoch = 77;
            return_outbound.live_row_count = 1;
            return_outbound.row_ids_host[0] = row_id;
            for (int column = 0; column < kDModel; ++column)
            {
                return_outbound.output_rows_fp32[column] =
                    static_cast<float>(20000 + round * 10 + column);
            }

            const auto return_key = makeMTPMoEOverlayRankBatchKey(
                /*generation_id=*/19,
                step,
                depth,
                kLayer,
                kTier,
                /*domain_ordinal=*/5,
                /*source_world_rank=*/0,
                /*target_world_rank=*/1,
                MoEOverlayCollectiveDirection::ReturnReduce);
            const auto return_result =
                rank_ == 1
                    ? transport.exchangeReturn(
                          return_key, return_outbound_views, {})
                    : transport.exchangeReturn(
                          return_key, {}, return_inbound_views);
            ASSERT_TRUE(return_result.ok) << return_result.error;
            ASSERT_TRUE(return_result.collective_complete);

            if (rank_ == 0)
            {
                EXPECT_EQ(return_inbound.key.key_namespace,
                          MoEOverlayCollectiveNamespace::MTP);
                EXPECT_EQ(return_inbound.key.mtp_depth, depth);
                EXPECT_EQ(return_inbound.row_ids_host[0], row_id);
                EXPECT_FLOAT_EQ(
                    return_inbound.output_rows_fp32[3],
                    static_cast<float>(20003 + round * 10));
            }
            const auto stale = rank_ == 1
                ? transport.exchangeReturn(return_key, return_outbound_views, {})
                : transport.exchangeReturn(return_key, {}, return_inbound_views);
            EXPECT_FALSE(stale.ok);
            EXPECT_NE(stale.error.find("stale"), std::string::npos);
        }

        uint64_t asynchronous_submissions = 0;
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"moe_overlay_transport.rank_batch_async_send_submissions"}))
        {
            asynchronous_submissions += record.count;
            EXPECT_FALSE(record.tags.contains("logical_step"));
            EXPECT_FALSE(record.tags.contains("generation"));
            EXPECT_FALSE(record.tags.contains("bytes"));
        }
        EXPECT_EQ(asynchronous_submissions, static_cast<uint64_t>(kRounds))
            << "Each rank owns exactly one asynchronous direction per round";
    }

    TEST_F(Test__MoEOverlaySparseTransport_MPI,
           TransactionFollowerSelectsMainAndDynamicDepthMTPGraphsFromFixedRing)
    {
        if (world_size_ != 2)
            GTEST_SKIP() << "Transaction protocol fixture requires exactly two MPI ranks";

        ScopedTransportPerfStats perf_stats("moe_overlay_transaction");
        constexpr std::size_t kControlSlots = 2;
        const MoEOverlayInferenceTopologyIdentity topology{
            .workspace_generation = 71,
            .topology_fingerprint_low = 0x12345678u,
            .topology_fingerprint_high = 0x9abcdef0u,
            .source_world_rank = 0,
            .target_world_rank = 1,
        };
        const MoEOverlayInferenceCommandIdentity command{
            .request_generation = 5,
            .command_id = 9,
            .initial_placement_epoch = 17,
        };
        auto channel =
            std::make_shared<MoEOverlayMPIInferenceTransactionChannel>(
                MoEOverlayMPIInferenceTransactionChannel::Config{
                    .mpi_ctx = mpi_ctx_,
                    .source_world_rank = 0,
                    .target_world_rank = 1,
                    .send_slot_count = kControlSlots,
                });

        std::vector<MoEOverlayInferenceTransactionTicket> expected;
        std::uint64_t ordinal = 1;
        const auto append = [&](MoEOverlayInferenceGraphRole role,
                                int logical_rows,
                                int physical_rows,
                                int draft_depth = -1,
                                int sidecar_depth = -1,
                                std::uint64_t retired_decode_tokens = 0u)
        {
            const std::uint64_t ticket_ordinal = ordinal++;
            expected.push_back(makeMoEOverlayInferenceExecutionTicket(
                topology,
                command,
                ticket_ordinal,
                /*logical_step_id=*/100u + ticket_ordinal,
                /*placement_epoch=*/17,
                role,
                /*request_count=*/2,
                logical_rows,
                physical_rows,
                draft_depth,
                sidecar_depth,
                /*prefill_schedule_workload=*/{},
                retired_decode_tokens));
        };

        /* One bucketed prefill and serial decode precede depth-2/depth-3 cycles. */
        append(MoEOverlayInferenceGraphRole::MainPrefill, 4, 4);
        append(MoEOverlayInferenceGraphRole::MainDecode, 1, 1);
        append(MoEOverlayInferenceGraphRole::MTPDraft, 1, 1, 2, 0, 1u);
        append(MoEOverlayInferenceGraphRole::MTPDraft, 1, 1, 2, 1, 1u);
        append(
            MoEOverlayInferenceGraphRole::MTPGroupedVerifier,
            3,
            4,
            2,
            -1,
            1u);
        append(MoEOverlayInferenceGraphRole::MainDecode, 1, 1, -1, -1, 4u);
        append(MoEOverlayInferenceGraphRole::MTPDraft, 1, 1, 3, 0, 5u);
        append(MoEOverlayInferenceGraphRole::MTPDraft, 1, 1, 3, 1, 5u);
        append(MoEOverlayInferenceGraphRole::MTPDraft, 1, 1, 3, 2, 5u);
        append(
            MoEOverlayInferenceGraphRole::MTPGroupedVerifier,
            4,
            4,
            3,
            -1,
            5u);
        RecordingTransactionExecutor executor;
        std::uint64_t retired_prefill_tokens = 0u;
        std::uint64_t retired_decode_tokens = 0u;
        std::uint64_t retired_decode_notifications = 0u;
        if (rank_ == 0)
        {
            MoEOverlayInferenceTransactionPublisher publisher({
                .channel = channel,
                .protocol = {
                    .topology = topology,
                    .slot_count = kControlSlots,
                    .max_request_count = 2,
                    .max_rows_per_request = 4,
                    .max_mtp_draft_depth = 3,
                },
            });
            std::string error;
            ASSERT_TRUE(publisher.beginCommand(command, &error)) << error;
            for (size_t index = 0; index < expected.size(); ++index)
            {
                const auto &ticket = expected[index];
                const auto published = publisher.publish({
                    .graph_role = ticket.graph_role,
                    .logical_step_id = ticket.logical_step_id,
                    .placement_epoch = ticket.placement_epoch,
                    .request_count = ticket.request_count,
                    .logical_rows_per_request =
                        ticket.logical_rows_per_request,
                    .physical_rows_per_request =
                        ticket.physical_rows_per_request,
                    .draft_depth = ticket.draft_depth,
                    .sidecar_depth = ticket.sidecar_depth,
                    .retired_decode_progress_tokens =
                        ticket.retired_decode_progress_tokens,
                });
                ASSERT_TRUE(published.ok) << published.error;
                EXPECT_EQ(published.ticket, ticket);
                if (index == 0u)
                {
                    EXPECT_FALSE(publisher.complete(17, 0u, &error))
                        << "Complete must not overtake a live data-plane return";
                }
                ASSERT_TRUE(publisher.retire(published, &error)) << error;
            }
            ASSERT_TRUE(publisher.complete(17, 9u, &error)) << error;
            EXPECT_EQ(publisher.state(),
                      MoEOverlayInferenceProtocolState::Complete);
        }
        else
        {
            MoEOverlayInferenceTransactionFollower follower({
                .channel = channel,
                .executor = &executor,
                .protocol = {
                    .topology = topology,
                    .slot_count = kControlSlots,
                    .max_request_count = 2,
                    .max_rows_per_request = 4,
                    .max_mtp_draft_depth = 3,
                },
                .retired_prefill_progress_sink =
                    [&retired_prefill_tokens](
                        std::uint64_t completed_tokens,
                        std::string *)
                {
                    retired_prefill_tokens += completed_tokens;
                    return true;
                },
                .retired_decode_progress_sink =
                    [&retired_decode_tokens,
                     &retired_decode_notifications](
                        std::uint64_t completed_tokens,
                        std::string *)
                {
                    retired_decode_tokens += completed_tokens;
                    ++retired_decode_notifications;
                    return true;
                },
            });
            const auto result = follower.runOneCommand();
            ASSERT_TRUE(result.ok) << result.error;
            EXPECT_FALSE(result.aborted);
            EXPECT_EQ(result.executed_transactions, expected.size());
            EXPECT_EQ(executor.tickets, expected);
            EXPECT_EQ(retired_prefill_tokens, 8u)
                << "The follower must sideband request_count times real rows";
            EXPECT_EQ(retired_decode_tokens, 9u);
            EXPECT_EQ(retired_decode_notifications, 4u)
                << "Repeated graph tickets in one sequence must not duplicate progress";
            EXPECT_EQ(result.retired_decode_progress_tokens, 9u);
            EXPECT_EQ(follower.state(),
                      MoEOverlayInferenceProtocolState::Complete);
        }

        /* Receipt of the terminal proves every earlier fixed slot is reusable. */
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank_ == 0)
            EXPECT_EQ(channel->inFlightSendCount(), 0u);

        std::uint64_t control_ticket_witnesses = 0;
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"moe_overlay_transaction.control_tickets"}))
        {
            control_ticket_witnesses += record.count;
        }
        const auto sent_ticket_count = expected.size() + 1u;
        EXPECT_EQ(
            control_ticket_witnesses,
            rank_ == 0 ? expected.size() * 3u + 2u
                       : sent_ticket_count + expected.size());
    }

} // namespace llaminar2::test
