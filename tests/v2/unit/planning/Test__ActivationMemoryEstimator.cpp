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

TEST(Test__ActivationMemoryEstimator, LargeVocabDominatedByLogits)
{
    // With large vocab, logits buffer (B×S×V×4) dominates
    size_t bytes_small_vocab = ActivationMemoryEstimator::estimate(
        1, 4096, 896, 4864, 14, 2, 64, 1000, DeviceId::cuda(0));
    size_t bytes_large_vocab = ActivationMemoryEstimator::estimate(
        1, 4096, 896, 4864, 14, 2, 64, 151936, DeviceId::cuda(0));

    EXPECT_GT(bytes_large_vocab, bytes_small_vocab);
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
    size_t logits = B * S * V * FP32;

    size_t attn_phase = hidden_state + residual + q_proj + k_proj + v_proj + attn_output + norm_scratch;
    size_t ffn_phase = hidden_state + residual + ffn_gate + ffn_up + ffn_down + norm_scratch;
    size_t lm_head_phase = hidden_state + logits;
    size_t expected_peak = std::max({attn_phase, ffn_phase, lm_head_phase});

    size_t actual = ActivationMemoryEstimator::estimate(
        1, 512, 256, 1024, 4, 2, 64, 1000, DeviceId::cuda(0));

    EXPECT_EQ(actual, expected_peak);
}

TEST(Test__ActivationMemoryEstimator, LargeVocab_LMHeadDominates)
{
    // With vocab=500000, lm_head_phase = B*S*(D+V)*4 should dominate
    // Verify the peak comes from the lm_head phase
    constexpr size_t B = 1, S = 4096, D = 256, F = 1024;
    constexpr size_t H = 4, HK = 2, HD = 64, V = 500000;
    constexpr size_t FP32 = 4;

    size_t lm_head_phase = B * S * D * FP32 + B * S * V * FP32;

    size_t actual = ActivationMemoryEstimator::estimate(
        1, 4096, 256, 1024, 4, 2, 64, 500000, DeviceId::cuda(0));

    // lm_head_phase should be the peak when vocab is huge
    EXPECT_EQ(actual, lm_head_phase);
}
