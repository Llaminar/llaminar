/**
 * @file Test__Qwen4ExpCheckpointGeometry.cpp
 * @brief Exact opaque rollback capacities and checked arithmetic without devices.
 *
 * PP ownership controls capacity: a PLE-only participant has no sparse tails,
 * predictor caches have no lexical history, and full main-cache capacity stays
 * independent of context length because complete pools are append-only state.
 */
#include "kernels/qwen4exp/Qwen4ExpCheckpointGeometry.h"
#include "kernels/DeviceSequenceStateGeometry.h"
#include "planning/PersistentStateMemoryEstimator.h"
#include "planning/ModelMemoryProfile.h"
#include <gtest/gtest.h>
#include <limits>

using llaminar2::qwen4exp::Qwen4ExpCheckpointGeometry;
/** @test Main, predictor and PLE-only owners price distinct exact live-state families. */
TEST(Qwen4ExpCheckpointGeometry, CapacitiesFollowExactLayerAndLexicalOwnership)
{
    using G = Qwen4ExpCheckpointGeometry;
    EXPECT_EQ(G::sparseLayerBytes(), 1552u);
    EXPECT_EQ(G::capacity(0, 0), 0u);
    EXPECT_EQ(G::capacity(1, 0), 1552u);
    EXPECT_EQ(G::capacity(0, 10240), 368648u);
    EXPECT_EQ(G::capacity(12, 10240), 387272u);
}
/** @test Overflow cannot admit a wrapped allocation smaller than its opaque payload. */
TEST(Qwen4ExpCheckpointGeometry, RejectsUnrepresentableStateCapacity)
{
    using G = Qwen4ExpCheckpointGeometry;
    EXPECT_THROW((void)G::capacity(std::numeric_limits<size_t>::max(), 0), std::overflow_error);
    EXPECT_THROW((void)G::capacity(0, std::numeric_limits<size_t>::max()), std::overflow_error);
}

/** @test Metadata admission and opaque cache materialization price the same PP-local tails. */
TEST(Qwen4ExpCheckpointGeometry, MetadataOnlyGPUCheckpointBOMIncludesExactAuxiliaryAndTrunk)
{
    using namespace llaminar2;
    ModelMemoryProfile profile;
    profile.architecture = "qwen4exp";
    profile.n_layers = 48;
    profile.mtp_layer_count = 1;
    profile.d_model = 2560;
    profile.n_heads = 24;
    profile.n_kv_heads = 2;
    profile.head_dim = 256;
    profile.gdn_conv_kernel_size = 4;
    profile.gdn_state_size = 128;
    profile.gdn_inner_size = 6144;
    profile.gdn_group_count = 16;
    profile.gdn_time_step_rank = 48;
    profile.full_attention_interval = 4;
    profile.flash_next.emplace();
    profile.flash_next->residual_width = 10240;
    profile.flash_next->ple_decoder_layer = 1;
    for (uint32_t layer = 3; layer < 48; layer += 4)
        profile.flash_next->sparse_decoder_layers.push_back(layer);
    profile.flash_next->sparse_decoder_layers.push_back(48);
    // This fixture intentionally describes an inline Sparse4 predictor. The
    // published separate Dense0 artifact is covered by PredictorProfile tests.
    profile.flash_next_predictor = Qwen4ExpPredictorMetadata{
        .source_layer = 48, .attention = Qwen4ExpPredictorAttention::Sparse,
        .hidden_norm = Qwen4ExpPredictorHiddenNorm::PerBranch,
        .source = Qwen4ExpPredictorSource::Inline};
    auto missing_policy = profile;
    missing_policy.flash_next_predictor.reset();
    EXPECT_THROW((void)PersistentStateMemoryEstimator::isFullAttentionLayer(missing_policy, 48),
        std::invalid_argument);
    EXPECT_EQ(profile.mainLayerCount(), 48);
    EXPECT_EQ(profile.totalLayerCount(), 49);
    for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
        for (const auto interval : {std::pair{0, 1}, std::pair{3, 3}, std::pair{0, 47}})
            for (const bool predictor : {false, true})
            {
                SCOPED_TRACE(device.toString() + ":" + std::to_string(interval.first) + ":" +
                             std::to_string(interval.second) + ":" + std::to_string(predictor));
                const auto estimate = PersistentStateMemoryEstimator::estimate(profile, device,
                    3, 16, 2, 2, 0, 24, interval.first, interval.second, "fp16",
                    predictor ? MTPStateRole::PredictorOwner : MTPStateRole::MainModelFollower);
                const size_t sparse = interval.second == 1 ? 0 : interval.second == 3 ? 1 : 12;
                const auto main_bytes = DeviceSequenceStateGeometry(sparse).checkpointBytes() +
                    Qwen4ExpCheckpointGeometry::capacity(sparse, interval.first == 0 ? 10240 : 0);
                const auto predictor_bytes = predictor
                    ? DeviceSequenceStateGeometry(1).checkpointBytes() + Qwen4ExpCheckpointGeometry::capacity(1, 0) : 0;
                EXPECT_EQ(estimate.main_full_attention_layers, sparse);
                EXPECT_EQ(estimate.mtp_full_attention_layers, predictor ? 1 : 0);
                EXPECT_EQ(estimate.sequence_metadata_bytes,
                    kMTPConcurrentLiveCheckpointSets * (main_bytes + predictor_bytes) + 3 * main_bytes);
                // An isolated sparse layer has no recurrent payload: its
                // terminal rollback still retains the full four-branch trunk.
                if (interval.first == 3)
                    EXPECT_EQ(estimate.checkpoint_state_bytes,
                        kMTPConcurrentLiveCheckpointSets * 10240u * sizeof(float));
            }
}

/** @test Legacy total-block profiles and main-only Flash Next profiles have explicit distinct contracts. */
TEST(Qwen4ExpCheckpointGeometry, RejectsInvalidAndOverflowingLayerCounts)
{
    llaminar2::ModelMemoryProfile profile;
    profile.n_layers = 49;
    profile.mtp_layer_count = 1;
    EXPECT_EQ(profile.mainLayerCount(), 48);
    EXPECT_EQ(profile.totalLayerCount(), 49);
    profile.n_layers = 0;
    EXPECT_THROW((void)profile.mainLayerCount(), std::invalid_argument);
    profile.flash_next.emplace();
    profile.n_layers = std::numeric_limits<int>::max();
    EXPECT_THROW((void)profile.totalLayerCount(), std::overflow_error);
}
