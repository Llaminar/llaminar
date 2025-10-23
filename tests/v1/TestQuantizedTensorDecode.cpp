#include "tensors/TensorFactory.h"
#include "QuantDequant.h"
#include <gtest/gtest.h>

using namespace llaminar;

// Helper to build a single Q4_0 block for predictable values:
// scale = 0.5 (fp16 representation), values 0..31 encoded with pattern.
static std::vector<uint8_t> make_q4_0_raw_one_block(float scale)
{
    // Build fp16 from ggml helper (approx); reuse ggml_fp32_to_fp16 in ggml-common? For test keep manual cast via QuantDequant helper path.
    // We'll approximate: use native half via _Float16 then reinterpret.
    _Float16 h = (_Float16)scale; // not exact but decodeBlock uses ggml_fp16_to_fp32 too.
    uint16_t hbits;
    std::memcpy(&hbits, &h, 2);
    std::vector<uint8_t> raw;
    raw.resize(2 + 16);
    std::memcpy(raw.data(), &hbits, 2);
    // Pack nibbles: produce a repeating ramp [-8..7]
    for (int i = 0; i < 16; ++i)
    {
        uint8_t lo = (uint8_t)((i % 16) & 0x0F); // 0..15
        uint8_t hi = (uint8_t)(((i + 1) % 16) & 0x0F);
        raw[2 + i] = (hi << 4) | lo;
    }
    return raw;
}

TEST(QuantizedTensorDecode, Q4_0DecodeBlockBasic)
{
    auto raw = make_q4_0_raw_one_block(0.5f);
    std::vector<int> shape{1, 32};
    auto qt = TensorFactory::create_quantized(shape, QuantFormat::Q4_0, raw);
    auto qcast = TensorFactory::to_quantized(qt);
    ASSERT_TRUE(qcast);
    std::vector<float> out(32, 0.0f);
    qcast->decodeBlock(0, out.data());
    // Just verify range within plausible bounds and non-zero variety
    float minv = out[0], maxv = out[0];
    for (float v : out)
    {
        minv = std::min(minv, v);
        maxv = std::max(maxv, v);
    }
    EXPECT_LT(minv, 0.0f);
    EXPECT_GT(maxv, 0.0f);
}

TEST(QuantizedTensorDecode, Q8_0DecodeBlockBasic)
{
    // Build Q8_0 block: scale half, bytes = 2 + 32 int8
    _Float16 h = (_Float16)1.0f;
    uint16_t hbits;
    std::memcpy(&hbits, &h, 2);
    std::vector<uint8_t> raw(2 + 32);
    std::memcpy(raw.data(), &hbits, 2);
    for (int i = 0; i < 32; ++i)
        raw[2 + i] = (uint8_t)((int8_t)(i - 16));
    std::vector<int> shape{1, 32};
    auto qt = TensorFactory::create_quantized(shape, QuantFormat::Q8_0, raw);
    auto qcast = TensorFactory::to_quantized(qt);
    ASSERT_TRUE(qcast);
    std::vector<float> out(32, 0.0f);
    qcast->decodeBlock(0, out.data());
    EXPECT_NE(out[0], out[1]);
    EXPECT_LT(out[0], 0.0f);
    EXPECT_GT(out[31], 0.0f);
}

// TileFP16CoversRegion test removed - decodeTileFP16() was dead code using wrong _Float16 type.
// BF16 decoding is tested via QuantizedSlabAllocator and GEMM integration tests.
