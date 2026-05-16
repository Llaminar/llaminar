/**
 * @file Test__MoEGraphNative_AbortPropagation.cpp
 * @brief Phase 21 sparse collective abort propagation tests for graph-native MoE overlay.
 */

#include "execution/compute_stages/stages/MoESparseDispatchStage.h"
#include "execution/compute_stages/stages/MoESparseReturnReduceStage.h"
#include "execution/moe/MoEOverlaySparseCollective.h"
#include "mocks/MockComputeStage.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include <memory>
#include <string>

namespace llaminar2::test
{
    namespace
    {
        constexpr int kLayer = 2;
        constexpr int kTier = 1;
        constexpr int kDomain = 42;
        constexpr int kSeqLen = 2;
        constexpr int kDModel = 4;
        constexpr int kTopK = 2;

        MoEOverlayCollectiveKey keyFor(MoEOverlayCollectiveDirection direction, uint64_t sequence)
        {
            MoEOverlayCollectiveKey key;
            key.generation_id = 7;
            key.step_id = 0;
            key.layer_idx = kLayer;
            key.tier_idx = kTier;
            key.domain_id = kDomain;
            key.direction = direction;
            key.sequence = sequence;
            return key;
        }

        void expectAborted(const MoEOverlayCollectiveResult &result, int reason_code)
        {
            EXPECT_FALSE(result.ok);
            EXPECT_EQ(result.error_code, reason_code);
            EXPECT_NE(result.error.find("aborted"), std::string::npos);
        }

        MoEOverlayCollectiveWorkspace makeWorkspace()
        {
            MoEOverlayCollectiveWorkspace workspace;
            workspace.ensureCapacity(kSeqLen, kSeqLen * kTopK, kDModel, kTopK, DeviceId::cpu());
            workspace.resetForStep(7, 0);
            return workspace;
        }

    } // namespace

    TEST(Test__MoEGraphNativeAbortPropagation, LocalSparseCollectiveAbortRejectsDispatchAndReturnKeys)
    {
        MoEOverlayLocalSparseCollectiveContext collective({.participant_count = 1, .slot_count = 4});
        auto workspace = makeWorkspace();

        const auto dispatch_key = keyFor(MoEOverlayCollectiveDirection::Dispatch, 10);
        auto outbound_dispatch = workspace.localExpertInput(kLayer, kTier);
        outbound_dispatch.key = dispatch_key;
        outbound_dispatch.source_participant = 0;
        outbound_dispatch.target_participant = 0;
        outbound_dispatch.live_row_count = 0;
        outbound_dispatch.live_entry_count = 0;
        outbound_dispatch.entry_offsets_host[0] = 0;
        auto inbound_dispatch = workspace.dispatchReceive(kLayer, kTier);

        collective.abort(dispatch_key, 77);
        const auto dispatch_result = collective.dispatch(dispatch_key, outbound_dispatch, &inbound_dispatch, nullptr);
        expectAborted(dispatch_result, 77);

        const auto return_key = keyFor(MoEOverlayCollectiveDirection::ReturnReduce, 11);
        auto outbound_return = workspace.localExpertOutput(kLayer, kTier);
        outbound_return.key = return_key;
        outbound_return.source_participant = 0;
        outbound_return.target_participant = 0;
        outbound_return.live_row_count = 0;
        auto inbound_return = workspace.returnReceive(kLayer, kTier);

        collective.abort(return_key, 88);
        const auto return_result = collective.returnReduce(return_key, outbound_return, &inbound_return, nullptr);
        expectAborted(return_result, 88);
    }

    TEST(Test__MoEGraphNativeAbortPropagation, SparseStagesReturnFalseAfterParticipantAbort)
    {
        auto cpu_ctx = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
        MoEOverlayLocalSparseCollectiveContext collective({.participant_count = 1, .slot_count = 4});
        auto workspace = makeWorkspace();

        const auto dispatch_key = keyFor(MoEOverlayCollectiveDirection::Dispatch, 20);
        auto inbound_dispatch = workspace.dispatchReceive(kLayer, kTier);
        collective.abort(dispatch_key, 91);

        MoESparseDispatchStage::Params dispatch_params;
        dispatch_params.device_id = DeviceId::cpu();
        dispatch_params.collective_context = &collective;
        dispatch_params.workspace = &workspace;
        dispatch_params.key = dispatch_key;
        dispatch_params.source_participant = 0;
        dispatch_params.target_participant = 0;
        dispatch_params.seq_len = kSeqLen;
        dispatch_params.top_k = kTopK;
        dispatch_params.d_model = kDModel;
        dispatch_params.inbound_rows = &inbound_dispatch;
        MoESparseDispatchStage dispatch_stage(std::move(dispatch_params));

        EXPECT_FALSE(dispatch_stage.execute(cpu_ctx.get()));

        const auto return_key = keyFor(MoEOverlayCollectiveDirection::ReturnReduce, 21);
        auto outbound_return = workspace.localExpertOutput(kLayer, kTier);
        auto inbound_return = workspace.returnReceive(kLayer, kTier);
        auto dense_output = std::make_shared<FP32Tensor>(std::vector<size_t>{kSeqLen, kDModel});
        collective.abort(return_key, 92);

        MoESparseReturnReduceStage::Params return_params;
        return_params.device_id = DeviceId::cpu();
        return_params.collective_context = &collective;
        return_params.key = return_key;
        return_params.source_participant = 0;
        return_params.target_participant = 0;
        return_params.outbound_rows = &outbound_return;
        return_params.inbound_rows = &inbound_return;
        return_params.dense_output = dense_output.get();
        return_params.seq_len = kSeqLen;
        return_params.d_model = kDModel;
        MoESparseReturnReduceStage return_stage(std::move(return_params));

        EXPECT_FALSE(return_stage.execute(cpu_ctx.get()));
    }

    TEST(Test__MoEGraphNativeAbortPropagation, MPISparseCollectiveAbortRejectsBeforeDispatchAndReturn)
    {
        int rank = -1;
        int world_size = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (world_size < 2)
            GTEST_SKIP() << "MPI abort propagation check requires at least 2 ranks";

        auto mpi_ctx = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);
        MoEOverlayMPISparseCollectiveContext collective(
            MoEOverlayMPISparseCollectiveContext::Config{.mpi_ctx = mpi_ctx, .local_participant_id = rank});
        auto workspace = makeWorkspace();

        const auto dispatch_key = keyFor(MoEOverlayCollectiveDirection::Dispatch, 30);
        auto outbound_dispatch = workspace.localExpertInput(kLayer, kTier);
        outbound_dispatch.key = dispatch_key;
        outbound_dispatch.source_participant = rank;
        outbound_dispatch.target_participant = (rank + 1) % world_size;
        outbound_dispatch.live_row_count = 0;
        outbound_dispatch.live_entry_count = 0;
        outbound_dispatch.entry_offsets_host[0] = 0;
        auto inbound_dispatch = workspace.dispatchReceive(kLayer, kTier);

        collective.abort(dispatch_key, 101);
        const auto dispatch_result = collective.dispatch(dispatch_key, outbound_dispatch, &inbound_dispatch, nullptr);
        expectAborted(dispatch_result, 101);

        const auto return_key = keyFor(MoEOverlayCollectiveDirection::ReturnReduce, 31);
        auto outbound_return = workspace.localExpertOutput(kLayer, kTier);
        outbound_return.key = return_key;
        outbound_return.source_participant = rank;
        outbound_return.target_participant = (rank + 1) % world_size;
        outbound_return.live_row_count = 0;
        auto inbound_return = workspace.returnReceive(kLayer, kTier);

        collective.abort(return_key, 102);
        const auto return_result = collective.returnReduce(return_key, outbound_return, &inbound_return, nullptr);
        expectAborted(return_result, 102);
    }

} // namespace llaminar2::test