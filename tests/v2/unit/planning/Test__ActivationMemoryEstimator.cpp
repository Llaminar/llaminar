#include <gtest/gtest.h>
#include "planning/ActivationMemoryEstimator.h"
#include "backends/DeviceId.h"

#include <algorithm>

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

TEST(Test__ActivationMemoryEstimator, PeakFormula_MatchesManualComputation)
{
    // Manually compute the three-phase peak for known inputs:
    // B=1, S=512, D=256, F=1024, H=4, HK=2, HD=64, V=1000
    constexpr size_t B = 1, S = 512, D = 256, F = 1024;
    constexpr size_t H = 4, HK = 2, HD = 64, V = 1000;
    constexpr size_t FP32 = 4;

    size_t hidden_state = B * S * D * FP32;
    size_t residual = B * S * D * FP32;
    size_t q_proj = B * S * H * HD * FP32;
    size_t k_proj = B * S * HK * HD * FP32;
    size_t v_proj = B * S * HK * HD * FP32;
    size_t attn_output = B * S * D * FP32;
    size_t norm_scratch = B * S * D * FP32;
    size_t ffn_gate = B * S * F * FP32;
    size_t ffn_up = B * S * F * FP32;
    size_t ffn_down = B * S * D * FP32;
    size_t logits = B * V * FP32;

    size_t attn_phase = hidden_state + residual + q_proj + k_proj + v_proj + attn_output + norm_scratch;
    size_t ffn_phase = hidden_state + residual + ffn_gate + ffn_up + ffn_down + norm_scratch;
    size_t lm_head_phase = hidden_state + logits;
    size_t expected_peak = std::max({attn_phase, ffn_phase, lm_head_phase});

    size_t actual = ActivationMemoryEstimator::estimate(
        1, 512, 256, 1024, 4, 2, 64, 1000, DeviceId::cuda(0));

    EXPECT_EQ(actual, expected_peak);
}

TEST(Test__ActivationMemoryEstimator, OneRowPrefill_LargeVocabLMHeadDominates)
{
    // With S=1 and vocab=500000, lm_head_phase = B*(S*D+V)*4
    // dominates. This keeps the estimator honest about terminal-row logits.
    constexpr size_t B = 1, S = 1, D = 256, F = 1024;
    constexpr size_t H = 4, HK = 2, HD = 64, V = 500000;
    constexpr size_t FP32 = 4;

    size_t lm_head_phase = B * S * D * FP32 + B * V * FP32;

    size_t actual = ActivationMemoryEstimator::estimate(
        static_cast<int>(B), static_cast<int>(S), static_cast<int>(D), static_cast<int>(F),
        static_cast<int>(H), static_cast<int>(HK), static_cast<int>(HD), static_cast<int>(V),
        DeviceId::cuda(0));

    EXPECT_EQ(actual, lm_head_phase);
}
