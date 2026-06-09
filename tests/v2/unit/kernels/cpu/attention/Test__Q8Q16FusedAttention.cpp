/**
 * @file Test__Q8Q16FusedAttention.cpp
 * @brief Tests for the fused Q8_1 and Q16_1 attention decode paths.
 *
 * Validates that passing raw Q8_1/Q16_1 tensors directly to the attention
 * kernel (bypassing FP32 shadow buffers) produces results matching the
 * dequant-first path within acceptable tolerance.
 *
 * Q16_1: VNNI int16 dot product (VPDPWSSD) + int16 V accumulation
 * Q8_1:  Inline int8→float dequant in attention inner loop
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "kernels/cpu/attention/CPUFlashAttentionKernelT.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"

using namespace llaminar2;

// ─────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────
static double cosine_similarity(const float *a, const float *b, size_t n)
{
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        dot += double(a[i]) * double(b[i]);
        na += double(a[i]) * double(a[i]);
        nb += double(b[i]) * double(b[i]);
    }
    if (na < 1e-30 || nb < 1e-30)
        return 0.0;
    return dot / std::sqrt(na * nb);
}

static float max_abs_diff(const float *a, const float *b, size_t n)
{
    float max_diff = 0.0f;
    for (size_t i = 0; i < n; ++i)
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    return max_diff;
}

// ─────────────────────────────────────────────────────────────────────
// Test fixture
// ─────────────────────────────────────────────────────────────────────
class Test__Q8Q16FusedAttention : public ::testing::Test
{
protected:
    int head_dim_ = 0;
    int n_heads_ = 0;
    int n_kv_heads_ = 0;
    int kv_dim_ = 0;
    int q_dim_ = 0;

    void SetUpDims(int head_dim, int n_heads, int n_kv_heads)
    {
        head_dim_ = head_dim;
        n_heads_ = n_heads;
        n_kv_heads_ = n_kv_heads;
        kv_dim_ = n_kv_heads * head_dim;
        q_dim_ = n_heads * head_dim;
    }

    static std::shared_ptr<FP32Tensor> makeRandomFP32(int rows, int cols, unsigned seed)
    {
        auto t = std::make_shared<FP32Tensor>(std::vector<size_t>{
            static_cast<size_t>(rows), static_cast<size_t>(cols)});
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        float *d = t->mutable_data();
        for (size_t i = 0; i < t->numel(); ++i)
            d[i] = dist(rng);
        return t;
    }

    /**
     * Run attention via the dequant-first path:
     *   FP32 Q, dequant(quantized K) → FP32, dequant(quantized V) → FP32 → kernel.
     */
    std::vector<float> runDequantPath(
        FP32Tensor *Q,
        const float *K_fp32, const float *V_fp32,
        int kv_len, bool causal)
    {
        auto K_t = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(kv_len), static_cast<size_t>(kv_dim_)});
        auto V_t = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(kv_len), static_cast<size_t>(kv_dim_)});
        std::memcpy(K_t->mutable_data(), K_fp32, static_cast<size_t>(kv_len) * kv_dim_ * sizeof(float));
        std::memcpy(V_t->mutable_data(), V_fp32, static_cast<size_t>(kv_len) * kv_dim_ * sizeof(float));

        auto output = std::make_shared<FP32Tensor>(
            std::vector<size_t>{1, static_cast<size_t>(q_dim_)});

        auto kernel = Q->createAttention();
        EXPECT_NE(kernel, nullptr);

        auto ws = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n_heads_ * kv_len)});

        bool ok = kernel->compute_tensor(
            Q, K_t.get(), V_t.get(), output.get(),
            /*batch_size=*/1, /*seq_len=*/1, kv_len,
            n_heads_, n_kv_heads_, head_dim_,
            causal, /*window_size=*/-1,
            ws.get(), nullptr, nullptr, -1);
        EXPECT_TRUE(ok);

        std::vector<float> result(q_dim_);
        std::memcpy(result.data(), output->data(), q_dim_ * sizeof(float));
        return result;
    }

    /**
     * Run attention with raw Q8_1 tensors (fused path).
     * The kernel should detect Q8_1 native_type and dispatch to compute_decode_q8kv.
     */
    std::vector<float> runFusedQ8Path(
        FP32Tensor *Q,
        const Q8_1Tensor *K_q8, const Q8_1Tensor *V_q8,
        int kv_len, bool causal)
    {
        auto output = std::make_shared<FP32Tensor>(
            std::vector<size_t>{1, static_cast<size_t>(q_dim_)});

        auto kernel = Q->createAttention();
        EXPECT_NE(kernel, nullptr);

        auto ws = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n_heads_ * kv_len)});

        bool ok = kernel->compute_tensor(
            Q, K_q8, V_q8, output.get(),
            /*batch_size=*/1, /*seq_len=*/1, kv_len,
            n_heads_, n_kv_heads_, head_dim_,
            causal, /*window_size=*/-1,
            ws.get(), nullptr, nullptr, -1);
        EXPECT_TRUE(ok);

        std::vector<float> result(q_dim_);
        std::memcpy(result.data(), output->data(), q_dim_ * sizeof(float));
        return result;
    }

    std::vector<float> runFusedQ8PathRows(
        FP32Tensor *Q,
        const Q8_1Tensor *K_q8, const Q8_1Tensor *V_q8,
        int seq_len, int kv_len, bool causal)
    {
        auto output = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(q_dim_)});

        auto kernel = Q->createAttention();
        EXPECT_NE(kernel, nullptr);

        auto ws = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n_heads_ * kv_len)});

        bool ok = kernel->compute_tensor(
            Q, K_q8, V_q8, output.get(),
            /*batch_size=*/1, seq_len, kv_len,
            n_heads_, n_kv_heads_, head_dim_,
            causal, /*window_size=*/-1,
            ws.get(), nullptr, nullptr, -1);
        EXPECT_TRUE(ok);

        std::vector<float> result(static_cast<size_t>(seq_len) * q_dim_);
        std::memcpy(result.data(), output->data(), result.size() * sizeof(float));
        return result;
    }

    /**
     * Run attention with raw Q16_1 tensors (fused path).
     * The kernel should detect Q16_1 native_type and dispatch to compute_decode_q16kv.
     */
    std::vector<float> runFusedQ16Path(
        FP32Tensor *Q,
        const Q16_1Tensor *K_q16, const Q16_1Tensor *V_q16,
        int kv_len, bool causal)
    {
        auto output = std::make_shared<FP32Tensor>(
            std::vector<size_t>{1, static_cast<size_t>(q_dim_)});

        auto kernel = Q->createAttention();
        EXPECT_NE(kernel, nullptr);

        auto ws = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n_heads_ * kv_len)});

        bool ok = kernel->compute_tensor(
            Q, K_q16, V_q16, output.get(),
            /*batch_size=*/1, /*seq_len=*/1, kv_len,
            n_heads_, n_kv_heads_, head_dim_,
            causal, /*window_size=*/-1,
            ws.get(), nullptr, nullptr, -1);
        EXPECT_TRUE(ok);

        std::vector<float> result(q_dim_);
        std::memcpy(result.data(), output->data(), q_dim_ * sizeof(float));
        return result;
    }

    std::vector<float> runFusedQ16PathRows(
        FP32Tensor *Q,
        const Q16_1Tensor *K_q16, const Q16_1Tensor *V_q16,
        int seq_len, int kv_len, bool causal)
    {
        auto output = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(q_dim_)});

        auto kernel = Q->createAttention();
        EXPECT_NE(kernel, nullptr);

        auto ws = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n_heads_ * kv_len)});

        bool ok = kernel->compute_tensor(
            Q, K_q16, V_q16, output.get(),
            /*batch_size=*/1, seq_len, kv_len,
            n_heads_, n_kv_heads_, head_dim_,
            causal, /*window_size=*/-1,
            ws.get(), nullptr, nullptr, -1);
        EXPECT_TRUE(ok);

        std::vector<float> result(static_cast<size_t>(seq_len) * q_dim_);
        std::memcpy(result.data(), output->data(), result.size() * sizeof(float));
        return result;
    }
};

// =================================================================
// Q8_1 Fused Attention Tests
// =================================================================

// Test 1: Q8_1 fused vs dequant — head_dim=128, short context
TEST_F(Test__Q8Q16FusedAttention, Q8_1_FusedVsDequant_HeadDim128_ShortContext)
{
    SetUpDims(/*head_dim=*/128, /*n_heads=*/16, /*n_kv_heads=*/8);
    constexpr int KV_LEN = 10;

    auto Q = makeRandomFP32(1, q_dim_, 1000);
    auto K_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 1001);
    auto V_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 1002);

    auto K_q8 = Q8_1Tensor::quantize_from_fp32(K_fp32->data(), K_fp32->shape());
    auto V_q8 = Q8_1Tensor::quantize_from_fp32(V_fp32->data(), V_fp32->shape());

    // Dequant reference: dequant Q8_1 → FP32, then FP32 attention
    auto dequant_result = runDequantPath(Q.get(), K_q8->fp32_data(), V_q8->fp32_data(), KV_LEN, false);

    // Fused path: pass raw Q8_1 tensors
    auto fused_result = runFusedQ8Path(Q.get(), K_q8.get(), V_q8.get(), KV_LEN, false);

    double cos = cosine_similarity(dequant_result.data(), fused_result.data(), q_dim_);
    std::cout << "Q8_1 Fused vs Dequant (h128, kv=10): cosine=" << cos << std::endl;
    EXPECT_GT(cos, 0.999) << "Q8_1 fused path should closely match dequant path";

    // Per-head analysis
    double worst = 1.0;
    int worst_h = -1;
    for (int h = 0; h < n_heads_; ++h)
    {
        double hcos = cosine_similarity(
            dequant_result.data() + h * head_dim_,
            fused_result.data() + h * head_dim_,
            head_dim_);
        if (hcos < worst)
        {
            worst = hcos;
            worst_h = h;
        }
    }
    std::cout << "  Worst per-head cosine: head " << worst_h << " = " << worst << std::endl;
    EXPECT_GT(worst, 0.99) << "Per-head cosine too low";
}

// Test 2: Q8_1 fused vs dequant — head_dim=128, longer context
TEST_F(Test__Q8Q16FusedAttention, Q8_1_FusedVsDequant_HeadDim128_LongContext)
{
    SetUpDims(128, 16, 8);
    constexpr int KV_LEN = 200;

    auto Q = makeRandomFP32(1, q_dim_, 2000);
    auto K_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 2001);
    auto V_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 2002);

    auto K_q8 = Q8_1Tensor::quantize_from_fp32(K_fp32->data(), K_fp32->shape());
    auto V_q8 = Q8_1Tensor::quantize_from_fp32(V_fp32->data(), V_fp32->shape());

    auto dequant_result = runDequantPath(Q.get(), K_q8->fp32_data(), V_q8->fp32_data(), KV_LEN, false);
    auto fused_result = runFusedQ8Path(Q.get(), K_q8.get(), V_q8.get(), KV_LEN, false);

    double cos = cosine_similarity(dequant_result.data(), fused_result.data(), q_dim_);
    std::cout << "Q8_1 Fused vs Dequant (h128, kv=200): cosine=" << cos << std::endl;
    EXPECT_GT(cos, 0.999) << "Q8_1 fused should match for long context";
}

// Test 3: Q8_1 fused — head_dim=64 (Qwen2.5-0.5B-like)
TEST_F(Test__Q8Q16FusedAttention, Q8_1_FusedVsDequant_HeadDim64)
{
    SetUpDims(/*head_dim=*/64, /*n_heads=*/16, /*n_kv_heads=*/4);
    constexpr int KV_LEN = 50;

    auto Q = makeRandomFP32(1, q_dim_, 3000);
    auto K_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 3001);
    auto V_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 3002);

    auto K_q8 = Q8_1Tensor::quantize_from_fp32(K_fp32->data(), K_fp32->shape());
    auto V_q8 = Q8_1Tensor::quantize_from_fp32(V_fp32->data(), V_fp32->shape());

    auto dequant_result = runDequantPath(Q.get(), K_q8->fp32_data(), V_q8->fp32_data(), KV_LEN, false);
    auto fused_result = runFusedQ8Path(Q.get(), K_q8.get(), V_q8.get(), KV_LEN, false);

    double cos = cosine_similarity(dequant_result.data(), fused_result.data(), q_dim_);
    std::cout << "Q8_1 Fused vs Dequant (h64, kv=50): cosine=" << cos << std::endl;
    EXPECT_GT(cos, 0.999) << "Q8_1 fused should match for head_dim=64";
}

// Test 4: Q8_1 causal masking
TEST_F(Test__Q8Q16FusedAttention, Q8_1_FusedVsDequant_CausalMask)
{
    SetUpDims(128, 16, 8);
    constexpr int KV_LEN = 30;

    auto Q = makeRandomFP32(1, q_dim_, 4000);
    auto K_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 4001);
    auto V_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 4002);

    auto K_q8 = Q8_1Tensor::quantize_from_fp32(K_fp32->data(), K_fp32->shape());
    auto V_q8 = Q8_1Tensor::quantize_from_fp32(V_fp32->data(), V_fp32->shape());

    auto dequant_result = runDequantPath(Q.get(), K_q8->fp32_data(), V_q8->fp32_data(), KV_LEN, /*causal=*/true);
    auto fused_result = runFusedQ8Path(Q.get(), K_q8.get(), V_q8.get(), KV_LEN, /*causal=*/true);

    double cos = cosine_similarity(dequant_result.data(), fused_result.data(), q_dim_);
    std::cout << "Q8_1 Fused vs Dequant (causal, h128, kv=30): cosine=" << cos << std::endl;
    EXPECT_GT(cos, 0.999) << "Q8_1 fused with causal mask should match";
}

// =================================================================
// Q16_1 Fused Attention Tests
// =================================================================

// Test 5: Q16_1 fused vs dequant — head_dim=128, short context
TEST_F(Test__Q8Q16FusedAttention, Q16_1_FusedVsDequant_HeadDim128_ShortContext)
{
    SetUpDims(/*head_dim=*/128, /*n_heads=*/16, /*n_kv_heads=*/8);
    constexpr int KV_LEN = 10;

    auto Q = makeRandomFP32(1, q_dim_, 5000);
    auto K_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 5001);
    auto V_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 5002);

    auto K_q16 = Q16_1Tensor::quantize_from_fp32(K_fp32->data(), K_fp32->shape());
    auto V_q16 = Q16_1Tensor::quantize_from_fp32(V_fp32->data(), V_fp32->shape());

    // Dequant Q16_1 to FP32 for reference (fp32_data() returns lazy-dequantized FP32)
    auto dequant_result = runDequantPath(Q.get(), K_q16->fp32_data(), V_q16->fp32_data(), KV_LEN, false);
    auto fused_result = runFusedQ16Path(Q.get(), K_q16.get(), V_q16.get(), KV_LEN, false);

    double cos = cosine_similarity(dequant_result.data(), fused_result.data(), q_dim_);
    std::cout << "Q16_1 Fused vs Dequant (h128, kv=10): cosine=" << cos << std::endl;
    EXPECT_GT(cos, 0.999) << "Q16_1 fused should closely match dequant path";

    // Per-head analysis
    double worst = 1.0;
    int worst_h = -1;
    for (int h = 0; h < n_heads_; ++h)
    {
        double hcos = cosine_similarity(
            dequant_result.data() + h * head_dim_,
            fused_result.data() + h * head_dim_,
            head_dim_);
        if (hcos < worst)
        {
            worst = hcos;
            worst_h = h;
        }
    }
    std::cout << "  Worst per-head cosine: head " << worst_h << " = " << worst << std::endl;
    EXPECT_GT(worst, 0.99) << "Per-head cosine too low";
}

// Test 6: Q16_1 fused — longer context
TEST_F(Test__Q8Q16FusedAttention, Q16_1_FusedVsDequant_HeadDim128_LongContext)
{
    SetUpDims(128, 16, 8);
    constexpr int KV_LEN = 200;

    auto Q = makeRandomFP32(1, q_dim_, 6000);
    auto K_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 6001);
    auto V_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 6002);

    auto K_q16 = Q16_1Tensor::quantize_from_fp32(K_fp32->data(), K_fp32->shape());
    auto V_q16 = Q16_1Tensor::quantize_from_fp32(V_fp32->data(), V_fp32->shape());

    auto dequant_result = runDequantPath(Q.get(), K_q16->fp32_data(), V_q16->fp32_data(), KV_LEN, false);
    auto fused_result = runFusedQ16Path(Q.get(), K_q16.get(), V_q16.get(), KV_LEN, false);

    double cos = cosine_similarity(dequant_result.data(), fused_result.data(), q_dim_);
    std::cout << "Q16_1 Fused vs Dequant (h128, kv=200): cosine=" << cos << std::endl;
    EXPECT_GT(cos, 0.999) << "Q16_1 fused should match for long context";
}

// Test 7: Q16_1 fused — head_dim=64
TEST_F(Test__Q8Q16FusedAttention, Q16_1_FusedVsDequant_HeadDim64)
{
    SetUpDims(/*head_dim=*/64, /*n_heads=*/16, /*n_kv_heads=*/4);
    constexpr int KV_LEN = 50;

    auto Q = makeRandomFP32(1, q_dim_, 7000);
    auto K_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 7001);
    auto V_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 7002);

    auto K_q16 = Q16_1Tensor::quantize_from_fp32(K_fp32->data(), K_fp32->shape());
    auto V_q16 = Q16_1Tensor::quantize_from_fp32(V_fp32->data(), V_fp32->shape());

    auto dequant_result = runDequantPath(Q.get(), K_q16->fp32_data(), V_q16->fp32_data(), KV_LEN, false);
    auto fused_result = runFusedQ16Path(Q.get(), K_q16.get(), V_q16.get(), KV_LEN, false);

    double cos = cosine_similarity(dequant_result.data(), fused_result.data(), q_dim_);
    std::cout << "Q16_1 Fused vs Dequant (h64, kv=50): cosine=" << cos << std::endl;
    EXPECT_GT(cos, 0.999) << "Q16_1 fused should match for head_dim=64";
}

// Test 8: Q16_1 causal masking
TEST_F(Test__Q8Q16FusedAttention, Q16_1_FusedVsDequant_CausalMask)
{
    SetUpDims(128, 16, 8);
    constexpr int KV_LEN = 30;

    auto Q = makeRandomFP32(1, q_dim_, 8000);
    auto K_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 8001);
    auto V_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 8002);

    auto K_q16 = Q16_1Tensor::quantize_from_fp32(K_fp32->data(), K_fp32->shape());
    auto V_q16 = Q16_1Tensor::quantize_from_fp32(V_fp32->data(), V_fp32->shape());

    auto dequant_result = runDequantPath(Q.get(), K_q16->fp32_data(), V_q16->fp32_data(), KV_LEN, /*causal=*/true);
    auto fused_result = runFusedQ16Path(Q.get(), K_q16.get(), V_q16.get(), KV_LEN, /*causal=*/true);

    double cos = cosine_similarity(dequant_result.data(), fused_result.data(), q_dim_);
    std::cout << "Q16_1 Fused vs Dequant (causal, h128, kv=30): cosine=" << cos << std::endl;
    EXPECT_GT(cos, 0.999) << "Q16_1 fused with causal mask should match";
}

TEST_F(Test__Q8Q16FusedAttention, Q8_1_MultiRowDecodeMatchesSerialPrefixDecode_M2ToM4)
{
    SetUpDims(/*head_dim=*/64, /*n_heads=*/8, /*n_kv_heads=*/4);
    constexpr int BASE_KV = 19;
    constexpr int MAX_M = 4;
    constexpr int FULL_KV = BASE_KV + MAX_M;

    auto Q_all = makeRandomFP32(MAX_M, q_dim_, 9000);
    auto K_fp32 = makeRandomFP32(FULL_KV, kv_dim_, 9001);
    auto V_fp32 = makeRandomFP32(FULL_KV, kv_dim_, 9002);
    auto K_q8 = Q8_1Tensor::quantize_from_fp32(K_fp32->data(), K_fp32->shape());
    auto V_q8 = Q8_1Tensor::quantize_from_fp32(V_fp32->data(), V_fp32->shape());

    for (int m = 2; m <= MAX_M; ++m)
    {
        FP32Tensor Q_multi({static_cast<size_t>(m), static_cast<size_t>(q_dim_)});
        std::copy_n(Q_all->data(), static_cast<size_t>(m) * q_dim_, Q_multi.mutable_data());
        const int kv_len = BASE_KV + m;
        const std::vector<float> multi =
            runFusedQ8PathRows(&Q_multi, K_q8.get(), V_q8.get(), m, kv_len, /*causal=*/true);

        for (int row = 0; row < m; ++row)
        {
            FP32Tensor Q_row({1, static_cast<size_t>(q_dim_)});
            std::copy_n(Q_all->data() + static_cast<size_t>(row) * q_dim_,
                        q_dim_,
                        Q_row.mutable_data());
            const std::vector<float> serial =
                runFusedQ8Path(&Q_row, K_q8.get(), V_q8.get(), BASE_KV + row + 1, /*causal=*/true);
            const float *multi_row = multi.data() + static_cast<size_t>(row) * q_dim_;
            const double cos = cosine_similarity(multi_row, serial.data(), q_dim_);
            const float max_diff = max_abs_diff(multi_row, serial.data(), q_dim_);
            EXPECT_GT(cos, 0.999f) << "M=" << m << " row=" << row;
            EXPECT_LT(max_diff, 2e-3f) << "M=" << m << " row=" << row;
        }
    }
}

TEST_F(Test__Q8Q16FusedAttention, Q16_1_MultiRowDecodeMatchesSerialPrefixDecode_M2ToM4)
{
    SetUpDims(/*head_dim=*/64, /*n_heads=*/8, /*n_kv_heads=*/4);
    constexpr int BASE_KV = 19;
    constexpr int MAX_M = 4;
    constexpr int FULL_KV = BASE_KV + MAX_M;

    auto Q_all = makeRandomFP32(MAX_M, q_dim_, 9100);
    auto K_fp32 = makeRandomFP32(FULL_KV, kv_dim_, 9101);
    auto V_fp32 = makeRandomFP32(FULL_KV, kv_dim_, 9102);
    auto K_q16 = Q16_1Tensor::quantize_from_fp32(K_fp32->data(), K_fp32->shape());
    auto V_q16 = Q16_1Tensor::quantize_from_fp32(V_fp32->data(), V_fp32->shape());

    for (int m = 2; m <= MAX_M; ++m)
    {
        FP32Tensor Q_multi({static_cast<size_t>(m), static_cast<size_t>(q_dim_)});
        std::copy_n(Q_all->data(), static_cast<size_t>(m) * q_dim_, Q_multi.mutable_data());
        const int kv_len = BASE_KV + m;
        const std::vector<float> multi =
            runFusedQ16PathRows(&Q_multi, K_q16.get(), V_q16.get(), m, kv_len, /*causal=*/true);

        for (int row = 0; row < m; ++row)
        {
            FP32Tensor Q_row({1, static_cast<size_t>(q_dim_)});
            std::copy_n(Q_all->data() + static_cast<size_t>(row) * q_dim_,
                        q_dim_,
                        Q_row.mutable_data());
            const std::vector<float> serial =
                runFusedQ16Path(&Q_row, K_q16.get(), V_q16.get(), BASE_KV + row + 1, /*causal=*/true);
            const float *multi_row = multi.data() + static_cast<size_t>(row) * q_dim_;
            const double cos = cosine_similarity(multi_row, serial.data(), q_dim_);
            const float max_diff = max_abs_diff(multi_row, serial.data(), q_dim_);
            EXPECT_GT(cos, 0.999f) << "M=" << m << " row=" << row;
            EXPECT_LT(max_diff, 2e-3f) << "M=" << m << " row=" << row;
        }
    }
}
