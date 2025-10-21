// Parametric parity tests for fused quant linear kernel across formats & tile sizes.
#include <gtest/gtest.h>
#include "../src/tensors/TensorFactory.h"
#include "../src/operators/QuantLinearKernel.h"
#include "../src/QuantDequant.h"               // for block_q*_K structs & QK_K
#include "../llama.cpp/ggml/src/ggml-quants.h" // reference quantize_row_q*_K_ref
#include <random>

using namespace llaminar;

struct FusedParamCfg
{
    QuantFormat format;
    int M, K, N;
    int tile_n; // 0 => auto
    int tile_k; // 0 => auto
};

static std::string name_for(const FusedParamCfg &c)
{
    std::string fmt;
    switch (c.format)
    {
    case QuantFormat::Q4_0:
        fmt = "Q4_0";
        break;
    case QuantFormat::Q8_0:
        fmt = "Q8_0";
        break;
    case QuantFormat::Q4_K:
        fmt = "Q4_K";
        break;
    case QuantFormat::Q5_K:
        fmt = "Q5_K";
        break;
    case QuantFormat::Q6_K:
        fmt = "Q6_K";
        break;
    case QuantFormat::Q8_K:
        fmt = "Q8_K";
        break;
    default:
        fmt = "UNK";
        break;
    }
    return fmt + std::string("_M") + std::to_string(c.M) + "K" + std::to_string(c.K) + "N" + std::to_string(c.N) +
           "_tn" + std::to_string(c.tile_n) + "_tk" + std::to_string(c.tile_k);
}

class QuantLinearFusedParamTest : public ::testing::TestWithParam<FusedParamCfg>
{
protected:
    std::mt19937 rng{42};
    std::uniform_real_distribution<float> dist{-1.f, 1.f};

    std::shared_ptr<QuantizedTensor> make_tensor(int K, int N, QuantFormat fmt)
    {
        // Construct a synthetic quantized tensor using block descriptor heuristics similar to earlier test.
        QuantBlockDescriptor desc;
        switch (fmt)
        {
        case QuantFormat::Q4_0:
            desc.elements_per_block = 32;
            desc.bytes_per_block = 18;
            desc.bits_per_value = 4;
            desc.scale_count = 1;
            desc.is_k_quant = false;
            break;
        case QuantFormat::Q8_0:
            desc.elements_per_block = 32;
            desc.bytes_per_block = 34;
            desc.bits_per_value = 8;
            desc.scale_count = 1;
            desc.is_k_quant = false;
            break; // (scale fp16 + 32 bytes values)
        case QuantFormat::Q4_K:
            desc.elements_per_block = QK_K;
            desc.bytes_per_block = sizeof(block_q4_K);
            desc.bits_per_value = 4;
            desc.scale_count = 0;
            desc.is_k_quant = true;
            break;
        case QuantFormat::Q5_K:
            desc.elements_per_block = QK_K;
            desc.bytes_per_block = sizeof(block_q5_K);
            desc.bits_per_value = 5;
            desc.scale_count = 0;
            desc.is_k_quant = true;
            break;
        case QuantFormat::Q6_K:
            desc.elements_per_block = QK_K;
            desc.bytes_per_block = sizeof(block_q6_K);
            desc.bits_per_value = 6;
            desc.scale_count = 0;
            desc.is_k_quant = true;
            break;
        case QuantFormat::Q8_K:
            desc.elements_per_block = QK_K;
            desc.bytes_per_block = sizeof(block_q8_K);
            desc.bits_per_value = 8;
            desc.scale_count = 0;
            desc.is_k_quant = true;
            break;
        default:
            desc.elements_per_block = 32;
            desc.bytes_per_block = 18;
            desc.bits_per_value = 4;
            desc.scale_count = 1;
            desc.is_k_quant = false;
            break; // fallback
        }
        int blocks_per_row = (N + desc.elements_per_block - 1) / desc.elements_per_block;
        size_t total_blocks = (size_t)K * blocks_per_row;
        std::vector<uint8_t> raw(total_blocks * desc.bytes_per_block, 0);
        std::mt19937 lrng(123);
        if (!desc.is_k_quant)
        {
            for (size_t b = 0; b < total_blocks; ++b)
            {
                uint8_t *ptr = raw.data() + b * desc.bytes_per_block;
                uint16_t h = 0x3C00;
                std::memcpy(ptr, &h, 2);
                for (size_t i = 2; i < desc.bytes_per_block; ++i)
                    ptr[i] = (uint8_t)lrng();
            }
        }
        else
        {
            // Realistic K-format: generate FP32 data per block, reference quantize into block struct, copy bytes.
            std::uniform_real_distribution<float> qdist(-4.f, 4.f);
            for (size_t k_row = 0; k_row < (size_t)K; ++k_row)
            {
                for (int bcol = 0; bcol < blocks_per_row; ++bcol)
                {
                    size_t block_index = k_row * (size_t)blocks_per_row + bcol;
                    uint8_t *ptr = raw.data() + block_index * desc.bytes_per_block;
                    int elem_base = bcol * desc.elements_per_block;
                    int span = std::min(desc.elements_per_block, N - elem_base);
                    // Prepare input buffer of size desc.elements_per_block (pad with zeros if tail)
                    std::vector<float> buf(desc.elements_per_block, 0.f);
                    for (int i = 0; i < span; ++i)
                        buf[i] = qdist(lrng);
                    // Quantize according to format
                    switch (fmt)
                    {
                    case QuantFormat::Q4_K:
                    {
                        block_q4_K blk{};
                        quantize_row_q4_K_ref(buf.data(), &blk, desc.elements_per_block);
                        std::memcpy(ptr, &blk, sizeof(blk));
                        break;
                    }
                    case QuantFormat::Q5_K:
                    {
                        block_q5_K blk{};
                        quantize_row_q5_K_ref(buf.data(), &blk, desc.elements_per_block);
                        std::memcpy(ptr, &blk, sizeof(blk));
                        break;
                    }
                    case QuantFormat::Q6_K:
                    {
                        block_q6_K blk{};
                        quantize_row_q6_K_ref(buf.data(), &blk, desc.elements_per_block);
                        std::memcpy(ptr, &blk, sizeof(blk));
                        break;
                    }
                    case QuantFormat::Q8_K:
                    {
                        block_q8_K blk{};
                        quantize_row_q8_K_ref(buf.data(), &blk, desc.elements_per_block);
                        std::memcpy(ptr, &blk, sizeof(blk));
                        break;
                    }
                    default:
                        break; // unreachable for K formats
                    }
                }
            }
        }
        QuantStorageLayout layout{fmt, {K, N}, total_blocks, desc};
        return std::make_shared<QuantizedTensor>(layout, raw);
    }
};

TEST_P(QuantLinearFusedParamTest, Parity)
{
    const auto cfg = GetParam();
    auto Wq = make_tensor(cfg.K, cfg.N, cfg.format);
    std::vector<float> A((size_t)cfg.M * cfg.K);
    std::mt19937 rngA(7);
    std::uniform_real_distribution<float> distA(-1.f, 1.f);
    for (auto &v : A)
        v = distA(rngA);
    std::vector<float> C_fused((size_t)cfg.M * cfg.N, 0.f);
    QuantLinearParams params;
    params.A = A.data();
    params.M = cfg.M;
    params.K = cfg.K;
    params.N = cfg.N;
    params.C = C_fused.data();
    params.zero_C = true;
    params.tile_n = cfg.tile_n;
    params.tile_k = cfg.tile_k;
    ASSERT_TRUE(quant_linear_fused(*Wq, params));

    // Reference: full dequant row-by-row (reusing block decode)
    std::vector<float> W_full((size_t)cfg.K * cfg.N, 0.f);
    int block_elems = Wq->layout().block_desc.elements_per_block;
    int blocks_per_row = (cfg.N + block_elems - 1) / block_elems;
    std::vector<float> tmp(block_elems);
    for (int k = 0; k < cfg.K; ++k)
    {
        for (int b = 0; b < blocks_per_row; ++b)
        {
            int col0 = b * block_elems;
            int span = std::min(block_elems, cfg.N - col0);
            size_t bi = (size_t)k * blocks_per_row + b;
            Wq->decodeBlock(bi, tmp.data());
            std::memcpy(W_full.data() + (size_t)k * cfg.N + col0, tmp.data(), span * sizeof(float));
        }
    }
    std::vector<float> C_ref((size_t)cfg.M * cfg.N, 0.f);
    for (int m = 0; m < cfg.M; ++m)
    {
        for (int k = 0; k < cfg.K; ++k)
        {
            float a = A[(size_t)m * cfg.K + k];
            const float *w_row = W_full.data() + (size_t)k * cfg.N;
            float *c_row = C_ref.data() + (size_t)m * cfg.N;
            for (int n = 0; n < cfg.N; ++n)
                c_row[n] += a * w_row[n];
        }
    }
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
    EXPECT_LT(max_abs, 1e-4) << name_for(cfg) << " max_abs=" << max_abs;
    EXPECT_LT(rel_l2, 1e-5) << name_for(cfg) << " rel_l2=" << rel_l2;
}

// Parameter space: a few moderate shapes + varying tiles (auto / explicit) for Q4_0, Q8_0, Q4_K
static std::vector<FusedParamCfg> build_params()
{
    std::vector<FusedParamCfg> v;
    std::vector<std::tuple<int, int, int>> shapes = {{8, 64, 96}, {4, 48, 80}, {16, 96, 160}, {32, 256, 384}};
    std::vector<int> tile_ns = {0, 128};
    std::vector<int> tile_ks = {0, 2};
    std::vector<QuantFormat> fmts = {QuantFormat::Q4_0, QuantFormat::Q8_0, QuantFormat::Q4_K, QuantFormat::Q5_K, QuantFormat::Q6_K, QuantFormat::Q8_K};
    for (auto f : fmts)
    {
        for (auto s : shapes)
        {
            for (int tn : tile_ns)
            {
                for (int tk : tile_ks)
                {
                    v.push_back({f, std::get<0>(s), std::get<1>(s), std::get<2>(s), tn, tk});
                }
            }
        }
    }
    return v;
}

INSTANTIATE_TEST_SUITE_P(FusedParity, QuantLinearFusedParamTest, ::testing::ValuesIn(build_params()),
                         [](const ::testing::TestParamInfo<FusedParamCfg> &info)
                         { return name_for(info.param); });
