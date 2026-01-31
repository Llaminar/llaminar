#include <gtest/gtest.h>
#include "v2/kernels/cpu/gemm_v4/QuantisedGemmJit_Q8_1_Strided_Softmax.h"
#include "v2/tensors/BlockStructures.h"
#include "v2/tensors/FP16Utils.h"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

using namespace llaminar2;
using namespace llaminar2::gemm_v4;

// Helper to quantize FP32 to Q8_1
void quantize_q8_1(const float *src, Q8_1Block *dst, int count)
{
    int blocks = count / 32;
    for (int b = 0; b < blocks; ++b)
    {
        float max_abs = 0.0f;
        for (int i = 0; i < 32; ++i)
        {
            max_abs = std::max(max_abs, std::abs(src[b * 32 + i]));
        }

        float scale = max_abs / 127.0f;
        if (scale == 0.0f)
            scale = 1.0f; // Avoid div by zero

        dst[b].d = fp32_to_fp16(scale);
        dst[b].sum_qs = 0;

        for (int i = 0; i < 32; ++i)
        {
            float val = src[b * 32 + i];
            int8_t q = static_cast<int8_t>(std::round(val / scale));
            dst[b].qs[i] = q;
            dst[b].sum_qs += q;
        }
    }
}

TEST(Test__QuantisedGemmJit_Q8_1_Strided_Softmax, Correctness)
{
    std::srand(42);
    int seq_len = 1 + (std::rand() % 8);
    int kv_len = 16 * (1 + (std::rand() % 8));
    int head_dim = 32 * (1 + (std::rand() % 4));
    float scale = 1.0f / std::sqrt((float)head_dim);

    std::vector<float> Q(seq_len * head_dim);
    std::vector<float> K(kv_len * head_dim);
    std::vector<float> C(seq_len * kv_len);
    std::vector<float> C_ref(seq_len * kv_len);

    // Init random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : Q)
        x = dist(gen);
    for (auto &x : K)
        x = dist(gen);

    // Quantize
    int k_blocks = head_dim / 32;
    std::vector<Q8_1Block> Q_quant(seq_len * k_blocks);
    std::vector<Q8_1Block> K_quant(kv_len * k_blocks);

    quantize_q8_1(Q.data(), Q_quant.data(), seq_len * head_dim);
    quantize_q8_1(K.data(), K_quant.data(), kv_len * head_dim);

    // Reference implementation (FP32 GEMM + Softmax)
    std::vector<float> Q_deq(seq_len * head_dim);
    std::vector<float> K_deq(kv_len * head_dim);

    for (int i = 0; i < seq_len * k_blocks; ++i)
    {
        float s = fp16_to_fp32(Q_quant[i].d);
        for (int j = 0; j < 32; ++j)
        {
            Q_deq[i * 32 + j] = Q_quant[i].qs[j] * s;
        }
    }
    for (int i = 0; i < kv_len * k_blocks; ++i)
    {
        float s = fp16_to_fp32(K_quant[i].d);
        for (int j = 0; j < 32; ++j)
        {
            K_deq[i * 32 + j] = K_quant[i].qs[j] * s;
        }
    }

    // Compute Ref
    for (int i = 0; i < seq_len; ++i)
    {
        float max_val = -1e9f;
        for (int j = 0; j < kv_len; ++j)
        {
            float dot = 0.0f;
            for (int k = 0; k < head_dim; ++k)
            {
                dot += Q_deq[i * head_dim + k] * K_deq[j * head_dim + k];
            }
            C_ref[i * kv_len + j] = dot * scale;
            max_val = std::max(max_val, C_ref[i * kv_len + j]);
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < kv_len; ++j)
        {
            C_ref[i * kv_len + j] = std::exp(C_ref[i * kv_len + j] - max_val);
            sum_exp += C_ref[i * kv_len + j];
        }
        for (int j = 0; j < kv_len; ++j)
        {
            C_ref[i * kv_len + j] /= sum_exp;
        }
    }

    QuantisedGemmJit_Q8_1_Strided_Softmax kernel;
    StridedGemmSoftmaxParams params;
    params.Q = Q_quant.data();
    params.K = K_quant.data();
    params.C = C.data();
    params.seq_len = seq_len;
    params.kv_len = kv_len;
    params.head_dim = head_dim;
    params.stride_q_bytes = k_blocks * sizeof(Q8_1Block);
    params.stride_k_bytes = k_blocks * sizeof(Q8_1Block);
    params.stride_c = kv_len;
    params.scale = scale;
    params.mask = nullptr;
    params.is_causal = false;
    params.causal_offset = 0;

    kernel.execute(params);

    // Compare
    for (int i = 0; i < seq_len * kv_len; ++i)
    {
        EXPECT_NEAR(C[i], C_ref[i], 1e-2f) << "Mismatch at index " << i;
    }
}
