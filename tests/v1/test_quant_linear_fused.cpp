#include <gtest/gtest.h>
#include "../src/tensors/TensorFactory.h"
#include "../src/operators/QuantLinearKernel.h"
#include "../src/Logger.h"
#include <random>

using namespace llaminar;

static std::shared_ptr<QuantizedTensor> make_simple_q4_0(int K, int N)
{
    QuantBlockDescriptor desc;
    desc.elements_per_block = 32;
    desc.bytes_per_block = 18;
    desc.bits_per_value = 4;
    desc.scale_count = 1;
    desc.is_k_quant = false;
    int blocks_per_row = (N + desc.elements_per_block - 1) / desc.elements_per_block;
    size_t total_blocks = (size_t)K * blocks_per_row;
    std::vector<uint8_t> raw(total_blocks * desc.bytes_per_block, 0);
    std::mt19937 rng(123);
    for (size_t b = 0; b < total_blocks; ++b)
    {
        uint8_t *ptr = raw.data() + b * desc.bytes_per_block;
        uint16_t h = 0x3C00;
        std::memcpy(ptr, &h, 2); // scale=1
        for (int i = 0; i < 16; ++i)
            ptr[2 + i] = (uint8_t)rng();
    }
    QuantStorageLayout layout{QuantFormat::Q4_0, {K, N}, total_blocks, desc};
    return std::make_shared<QuantizedTensor>(layout, raw);
}

TEST(QuantLinearFused, Q4_0BaselineParity)
{
    int M = 8, K = 64, N = 96; // small sizes
    auto Wq = make_simple_q4_0(K, N);
    std::vector<float> A((size_t)M * K);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto &v : A)
        v = dist(rng);
    std::vector<float> C_fused((size_t)M * N);
    QuantLinearParams params;
    params.A = A.data();
    params.M = M;
    params.K = K;
    params.N = N;
    params.C = C_fused.data();
    params.zero_C = true;
    ASSERT_TRUE(quant_linear_fused(*Wq, params));

    // Reference: fully dequantize W then do naive GEMM
    std::vector<float> W_full((size_t)K * N);
    int block_elems = Wq->layout().block_desc.elements_per_block;
    int blocks_per_row = (N + block_elems - 1) / block_elems;
    std::vector<float> tmp(block_elems);
    for (int k = 0; k < K; ++k)
    {
        for (int b = 0; b < blocks_per_row; ++b)
        {
            int col0 = b * block_elems;
            int span = std::min(block_elems, N - col0);
            size_t bi = (size_t)k * blocks_per_row + b;
            Wq->decodeBlock(bi, tmp.data());
            std::memcpy(W_full.data() + (size_t)k * N + col0, tmp.data(), span * sizeof(float));
        }
    }
    std::vector<float> C_ref((size_t)M * N, 0.f);
    for (int m = 0; m < M; ++m)
    {
        for (int k = 0; k < K; ++k)
        {
            float a = A[(size_t)m * K + k];
            const float *w_row = W_full.data() + (size_t)k * N;
            float *c_row = C_ref.data() + (size_t)m * N;
            for (int n = 0; n < N; ++n)
                c_row[n] += a * w_row[n];
        }
    }
    // Compare
    double max_abs = 0, rel_l2_num = 0, rel_l2_den = 0;
    for (size_t i = 0; i < C_ref.size(); ++i)
    {
        double d = std::abs((double)C_ref[i] - (double)C_fused[i]);
        if (d > max_abs)
            max_abs = d;
        rel_l2_num += d * d;
        rel_l2_den += (double)C_ref[i] * C_ref[i];
    }
    double rel_l2 = std::sqrt(rel_l2_num / std::max(1e-12, rel_l2_den));
    EXPECT_LT(max_abs, 1e-4) << "max_abs=" << max_abs;
    EXPECT_LT(rel_l2, 1e-5) << "rel_l2=" << rel_l2;
}
