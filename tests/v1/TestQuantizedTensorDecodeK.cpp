// Tests for K-format block decode parity with upstream ggml dequantize_row_q*_K helpers.
// We construct a single random block worth of FP32 data, quantize via *_ref where available,
// then feed raw block bytes through QuantizedTensor::decodeBlock and compare outputs.

#include <gtest/gtest.h>
#include "../src/tensors/TensorFactory.h"
#include "../src/QuantDequant.h" // for qd_fp16_to_fp32 if needed
#include "../llama.cpp/ggml/src/ggml-quants.h"
#include <random>

using namespace llaminar;

namespace
{
    // Helper to make reproducible random floats in a moderate range.
    std::vector<float> make_random(int n)
    {
        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
        std::vector<float> v(n);
        for (int i = 0; i < n; ++i)
            v[i] = dist(rng);
        return v;
    }

    template <typename BlockT>
    std::vector<uint8_t> block_to_bytes(const BlockT &blk)
    {
        std::vector<uint8_t> bytes(sizeof(BlockT));
        std::memcpy(bytes.data(), &blk, sizeof(BlockT));
        return bytes;
    }
}

// Generic parity checker for one K-format: quantize -> decodeBlock vs dequantize_row
template <typename BlockT, typename QuantizeFn, typename DequantRowFn>
void run_k_block_test(QuantFormat fmt, QuantizeFn q_fn, DequantRowFn d_fn)
{
    constexpr int QK = QK_K; // 256
    auto input = make_random(QK);
    BlockT blk{};
    q_fn(input.data(), &blk, QK); // reference quantize
    // Raw bytes for a single block
    auto raw = block_to_bytes(blk);

    // Build quantized tensor layout: 1 block only (256 elements)
    QuantBlockDescriptor desc;
    desc.elements_per_block = QK;
    desc.bytes_per_block = (int)raw.size();
    desc.bits_per_value = 0;
    desc.is_k_quant = true;
    QuantStorageLayout layout{fmt, {1, QK}, 1, desc};
    auto qt = std::make_shared<QuantizedTensor>(layout, raw);

    float decoded[QK];
    qt->decodeBlock(0, decoded);

    // Ground truth dequant
    float truth[QK];
    d_fn(&blk, truth, QK);

    // Compare
    double max_abs = 0.0, rel_l2_num = 0.0, rel_l2_den = 0.0;
    for (int i = 0; i < QK; ++i)
    {
        double diff = (double)decoded[i] - (double)truth[i];
        max_abs = std::max(max_abs, std::abs(diff));
        rel_l2_num += diff * diff;
        rel_l2_den += (double)truth[i] * (double)truth[i];
    }
    double rel_l2 = rel_l2_den > 0 ? std::sqrt(rel_l2_num / rel_l2_den) : 0.0;
    EXPECT_LT(max_abs, 1e-3) << "max_abs divergence too large";
    EXPECT_LT(rel_l2, 1e-4) << "relative L2 divergence too large";
}

TEST(QuantizedTensorDecodeK, Q4_K_BlockParity)
{
    run_k_block_test<block_q4_K>(QuantFormat::Q4_K, quantize_row_q4_K_ref, dequantize_row_q4_K);
}

TEST(QuantizedTensorDecodeK, Q5_K_BlockParity)
{
    run_k_block_test<block_q5_K>(QuantFormat::Q5_K, quantize_row_q5_K_ref, dequantize_row_q5_K);
}

TEST(QuantizedTensorDecodeK, Q6_K_BlockParity)
{
    run_k_block_test<block_q6_K>(QuantFormat::Q6_K, quantize_row_q6_K_ref, dequantize_row_q6_K);
}

// Q8_K: upstream provides dequantize but quantize reference may differ; we can build a synthetic block by round-tripping through quantize_row_q8_0 then adapting, but simplest is to skip if no ref quantize fn available.
// Provide a smoke test calling decodeBlock to ensure it doesn't crash when fed zeros.
TEST(QuantizedTensorDecodeK, Q8_K_Smoke)
{
    constexpr int QK = QK_K;
    std::vector<uint8_t> raw(sizeof(block_q8_K));
    std::fill(raw.begin(), raw.end(), 0);
    QuantBlockDescriptor desc;
    desc.elements_per_block = QK;
    desc.bytes_per_block = (int)raw.size();
    desc.bits_per_value = 8;
    desc.is_k_quant = true;
    QuantStorageLayout layout{QuantFormat::Q8_K, {1, QK}, 1, desc};
    auto qt = std::make_shared<QuantizedTensor>(layout, raw);
    float decoded[QK];
    qt->decodeBlock(0, decoded);
    // All zeros expected with zeroed block
    for (int i = 0; i < QK; ++i)
    {
        EXPECT_FLOAT_EQ(decoded[i], 0.0f);
    }
}
