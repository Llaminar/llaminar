#include "execution/moe/MoEOverlaySparseCollective.h"
#include "utils/MPIContext.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include <memory>

namespace llaminar2::test
{
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
                MoEOverlayMPISparseCollectiveContext::Config{.mpi_ctx = mpi_ctx_, .local_participant_id = rank_});

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

} // namespace llaminar2::test
