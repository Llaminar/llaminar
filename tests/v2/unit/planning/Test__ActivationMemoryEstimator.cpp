/**
 * @file Test__ActivationMemoryEstimator.cpp
 * @brief Verifies generic and production model graph-arena byte accounting.
 */

#include <gtest/gtest.h>
#include "planning/ActivationMemoryEstimator.h"
#include "backends/DeviceId.h"

#include <algorithm>
#include <string>

using namespace llaminar2;

TEST(Test__ActivationMemoryEstimator, ReturnsNonZeroForValidInput)
{
    size_t bytes = ActivationMemoryEstimator::estimate(
        1, 4096, 896, 4864, 14, 2, 64, 151936, DeviceId::cuda(0));

    EXPECT_GT(bytes, 0u);
}

TEST(Test__ActivationMemoryEstimator, ReturnsZeroForInvalidInput)
{
    EXPECT_EQ(ActivationMemoryEstimator::estimate(0, 4096, 896, 4864, 14, 2, 64, 151936, DeviceId::cuda(0)), 0u);
    EXPECT_EQ(ActivationMemoryEstimator::estimate(1, 0, 896, 4864, 14, 2, 64, 151936, DeviceId::cuda(0)), 0u);
    EXPECT_EQ(ActivationMemoryEstimator::estimate(1, 4096, 0, 4864, 14, 2, 64, 151936, DeviceId::cuda(0)), 0u);
}

TEST(Test__ActivationMemoryEstimator, ScalesWithSeqLen)
{
    size_t bytes_2k = ActivationMemoryEstimator::estimate(
        1, 2048, 896, 4864, 14, 2, 64, 151936, DeviceId::cuda(0));
    size_t bytes_4k = ActivationMemoryEstimator::estimate(
        1, 4096, 896, 4864, 14, 2, 64, 151936, DeviceId::cuda(0));

    // Larger seq_len should mean more activation memory
    EXPECT_GT(bytes_4k, bytes_2k);
}

TEST(Test__ActivationMemoryEstimator, LargeVocabDoesNotReserveAllPositionLogits)
{
    constexpr size_t B = 1, S = 4096, D = 896, F = 4864;
    constexpr size_t H = 14, HK = 2, HD = 64, V = 151936;
    constexpr size_t FP32 = 4;

    const size_t actual = ActivationMemoryEstimator::estimate(
        static_cast<int>(B), static_cast<int>(S), static_cast<int>(D), static_cast<int>(F),
        static_cast<int>(H), static_cast<int>(HK), static_cast<int>(HD), static_cast<int>(V),
        DeviceId::cuda(0));

    const size_t hidden_state = B * S * D * FP32;
    const size_t terminal_logits = B * V * FP32;
    const size_t all_position_logits = B * S * V * FP32;

    EXPECT_LT(actual, hidden_state + all_position_logits);
    EXPECT_GE(actual, hidden_state + terminal_logits);
}

TEST(Test__ActivationMemoryEstimator, CPUAndGPUSameEstimate)
{
    // Activation estimate should be similar for CPU and GPU (same buffers needed)
    size_t gpu = ActivationMemoryEstimator::estimate(
        1, 4096, 896, 4864, 14, 2, 64, 151936, DeviceId::cuda(0));
    size_t cpu = ActivationMemoryEstimator::estimate(
        1, 4096, 896, 4864, 14, 2, 64, 151936, DeviceId::cpu());

    EXPECT_EQ(gpu, cpu);
}

TEST(Test__ActivationMemoryEstimator, LogicalArenaOwnership_MatchesManualComputation)
{
    // Manually compute the physical BufferId owners for known inputs:
    // B=1, S=512, D=256, F=1024, H=4, HK=2, HD=64, V=1000
    constexpr size_t B = 1, S = 512, D = 256, F = 1024;
    constexpr size_t H = 4, HK = 2, HD = 64, V = 1000;
    constexpr size_t FP32 = 4;

    const size_t Q = H * HD;
    const size_t KV = HK * HD;
    const size_t expected_owned =
        B * S * (5 * D + 2 * Q + 2 * KV + 2 * F + S) * FP32 +
        B * V * FP32;

    size_t actual = ActivationMemoryEstimator::estimate(
        1, 512, 256, 1024, 4, 2, 64, 1000, DeviceId::cuda(0));

    EXPECT_EQ(actual, expected_owned);
}

TEST(Test__ActivationMemoryEstimator, OneRowPrefill_StillOwnsEveryRegisteredBuffer)
{
    // A large vocabulary dominates this case, but registered one-row graph
    // buffers remain physically owned and must not disappear from the estimate.
    constexpr size_t B = 1, S = 1, D = 256, F = 1024;
    constexpr size_t H = 4, HK = 2, HD = 64, V = 500000;
    constexpr size_t FP32 = 4;

    const size_t Q = H * HD;
    const size_t KV = HK * HD;
    const size_t expected =
        B * S * (5 * D + 2 * Q + 2 * KV + 2 * F + S) * FP32 +
        B * V * FP32;

    size_t actual = ActivationMemoryEstimator::estimate(
        static_cast<int>(B), static_cast<int>(S), static_cast<int>(D), static_cast<int>(F),
        static_cast<int>(H), static_cast<int>(HK), static_cast<int>(HD), static_cast<int>(V),
        DeviceId::cuda(0));

    EXPECT_EQ(actual, expected);
}

TEST(Test__ActivationMemoryEstimator,
     Qwen35MoE122BLocalTP_MatchesDeclarativeArenaByteForByte)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen35moe";
    profile.n_layers = 49;
    profile.mtp_layer_count = 1;
    profile.d_model = 3072;
    profile.d_ff = 1024;
    profile.n_heads = 32;
    profile.n_kv_heads = 2;
    profile.head_dim = 256;
    profile.vocab_size = 248320;
    profile.max_seq_len = 4096;
    profile.expert_count = 256;
    profile.expert_used_count = 8;
    profile.expert_feed_forward_length = 1024;

    const auto add_projection = [&](
        std::string name,
        size_t output_columns,
        size_t input_columns)
    {
        TensorSizeInfo tensor;
        tensor.name = std::move(name);
        tensor.K = input_columns;
        tensor.elements = output_columns * input_columns;
        tensor.layer_index = 0;
        profile.tensors.push_back(std::move(tensor));
    };
    add_projection("blk.0.attn_q.weight", 16384, 3072);
    add_projection("blk.0.attn_gate.weight", 8192, 3072);
    add_projection("blk.0.attn_qkv.weight", 16384, 3072);
    add_projection("blk.0.ssm_out.weight", 3072, 8192);
    add_projection("blk.0.ssm_alpha.weight", 64, 3072);
    add_projection("blk.0.ssm_beta.weight", 64, 3072);

    const size_t actual = ActivationMemoryEstimator::estimate(
        profile,
        ActivationGraphMemoryGeometry{
            .batch_size = 1,
            .resident_graph_rows = 4096,
            .local_d_ff = 512,
            .local_n_heads = 16,
            .local_n_kv_heads = 1,
            .first_layer = 0,
            .last_layer = 47,
            .total_shards = 2,
            .mtp_target_query_rows = 2,
            .mtp_terminal_logits_layout =
                MTPTerminalLogitsLayout::FullVocabularyPerParticipant,
        },
        DeviceId::cuda(0));

    /*
     * This constant is the independent sum of all 49 layer/model BufferSpecs
     * resolved by Qwen35MoESchema for the real 122B local-TP geometry.  It is
     * also the byte sum printed by BufferArena's address map in the production
     * model probe. In particular it includes the 503,316,480-byte canonical
     * route tensor that exposed the original capacity-admission defect.
     */
    constexpr size_t kExpectedDeclarativeArenaBytes = 2253085700ULL;
    EXPECT_EQ(actual, kExpectedDeclarativeArenaBytes);
}
