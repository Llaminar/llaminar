// Micro test: verify Q5_1 block storage -> dequantizeQ5_1 ordering matches expected row-major
// layout: blocks laid out row-contiguously (all blocks of row0, then row1, ...).
// This isolates whether the full dequant path matches our assumed layout; if this passes
// persistent parity failures likely stem from partial column shard slicing logic.

#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include "ModelLoader.h"

using namespace llaminar;

namespace
{
    // Encode a single Q5_1 block (32 values) with scale=1.0, min=0.0, qh high bits all zero.
    // values must be in range [0,31]; with qh=0 we can only represent 0..15 directly, so we keep
    // values <16 to avoid needing high bits; for upper half we also keep <16. This keeps test simple.
    static void encode_q5_1_block(const std::vector<uint8_t> &low16, const std::vector<uint8_t> &high16, std::vector<uint8_t> &out)
    {
        ASSERT_EQ(low16.size(), 16u);
        ASSERT_EQ(high16.size(), 16u);
        uint16_t d_fp16 = 0x3C00; // 1.0
        uint16_t m_fp16 = 0x0000; // 0.0
        uint32_t qh = 0;          // no high bits set
        // layout: d(2) m(2) qh(4) qs(16)
        out.resize(2 + 2 + 4 + 16);
        std::memcpy(out.data(), &d_fp16, 2);
        std::memcpy(out.data() + 2, &m_fp16, 2);
        std::memcpy(out.data() + 4, &qh, 4);
        uint8_t *qs = out.data() + 8;
        for (int j = 0; j < 16; ++j)
        {
            // pack high half value in high nibble, low half value in low nibble
            uint8_t lo = low16[j] & 0x0F; // ensure <16
            uint8_t hi = high16[j] & 0x0F;
            qs[j] = (hi << 4) | lo;
        }
    }
}

TEST(Q5_1_LayoutMicro, FullDecodeOrdering)
{
    const int rows = 3;
    const int cols = 64;                  // 2 blocks per row
    const int blocks_per_row = cols / 32; // 2
    std::vector<uint8_t> raw;
    raw.reserve(rows * blocks_per_row * (2 + 2 + 4 + 16));
    // Build deterministic pattern: For block index B (row-major), low half values = j, high half values = j+1 (mod 16)
    // ensure <16 so that high bits unused path suffices.
    for (int r = 0; r < rows; ++r)
    {
        for (int b = 0; b < blocks_per_row; ++b)
        {
            std::vector<uint8_t> lo(16), hi(16);
            for (int j = 0; j < 16; ++j)
            {
                lo[j] = (uint8_t)(j & 0x0F);
                hi[j] = (uint8_t)((j + 1) & 0x0F);
            }
            std::vector<uint8_t> block_bytes;
            encode_q5_1_block(lo, hi, block_bytes);
            raw.insert(raw.end(), block_bytes.begin(), block_bytes.end());
        }
    }
    // Fake tensor info
    GGUFTensorInfo info;
    info.name = "synthetic_q5_1";
    info.type = GGUFTensorType::Q5_1;
    info.dimensions = {(uint64_t)rows, (uint64_t)cols};
    info.size_bytes = raw.size();
    // Use loader decode function directly
    ModelLoader loader; // not loading a model; only using decode helper
    std::vector<float> decoded = loader.dequantizeQ5_1(raw.data(), info);
    ASSERT_EQ(decoded.size(), (size_t)rows * cols);
    // Validate ordering: each row has 2 blocks; block structure: first 16 are low nibble values (0..15), next 16 are high nibble (1..16 mod 16)
    for (int r = 0; r < rows; ++r)
    {
        for (int b = 0; b < blocks_per_row; ++b)
        {
            for (int j = 0; j < 16; ++j)
            {
                float expect_lo = (float)j;                // low half
                float expect_hi = (float)((j + 1) & 0x0F); // high half wrapped
                float got_lo = decoded[r * cols + b * 32 + j];
                float got_hi = decoded[r * cols + b * 32 + 16 + j];
                ASSERT_FLOAT_EQ(got_lo, expect_lo) << "row=" << r << " block=" << b << " j=" << j;
                ASSERT_FLOAT_EQ(got_hi, expect_hi) << "row=" << r << " block=" << b << " j=" << j;
            }
        }
    }
}
