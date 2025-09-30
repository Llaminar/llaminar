#include "model_loader.h"
#include "logger.h"
#include "../src/quant_dequant.h"
#include <gtest/gtest.h>
#include <random>
#include <cstring>
#include <cmath>
#include <mpi.h>

// This test samples a random Q4_0 block payload, decodes it via ModelLoader::dequantizeQ4_0
// and independently via the fused helper (dequant_q4_0_rows) to ensure identical fp32 vectors.
// Guards against future drift between reference and fused paths.

extern "C"
{
#include "ggml.h"
}

using namespace llaminar;

namespace
{
    std::vector<uint8_t> make_random_q4_0_payload(size_t n_elements, uint32_t seed)
    {
        const size_t qk = 32;                             // values per block
        const size_t block_bytes = sizeof(uint16_t) + 16; // fp16 scale + 16 packed nibbles
        size_t n_blocks = (n_elements + qk - 1) / qk;
        std::vector<uint8_t> data(n_blocks * block_bytes, 0);
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> nib(0, 15);
        std::uniform_int_distribution<int> scale_bits(0, 0xFFFF);
        for (size_t b = 0; b < n_blocks; ++b)
        {
            uint16_t scale = static_cast<uint16_t>(scale_bits(rng));
            std::memcpy(&data[b * block_bytes], &scale, sizeof(uint16_t));
            uint8_t *vals = &data[b * block_bytes + sizeof(uint16_t)];
            for (int i = 0; i < 16; ++i)
            {
                uint8_t lo = static_cast<uint8_t>(nib(rng));
                uint8_t hi = static_cast<uint8_t>(nib(rng));
                vals[i] = static_cast<uint8_t>(lo | (hi << 4));
            }
        }
        return data;
    }
}

TEST(QuantDequantParity, Q4_0ReferenceMatchesFusedHelper)
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
    {
        int argc = 0;
        char **argv = nullptr;
        MPI_Init(&argc, &argv);
    }
    const size_t n = 320; // 10 blocks
    auto payload = make_random_q4_0_payload(n, 12345);

    ModelLoader loader; // Not loading a full model; we just want the helper path.
    auto ref = loader.dequantizeQ4_0(payload.data(), n);
    ASSERT_EQ(ref.size(), n);

    std::vector<float> fused(n, 0.0f);
    llaminar::dequant_q4_0_rows(payload.data(), fused.data(), n);

    double max_abs = 0.0;
    double diff_sq = 0.0;
    double ref_sq = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        double d = static_cast<double>(ref[i]) - fused[i];
        max_abs = std::max(max_abs, std::fabs(d));
        diff_sq += d * d;
        ref_sq += static_cast<double>(ref[i]) * ref[i];
    }
    double rel_l2 = (ref_sq > 0.0) ? std::sqrt(diff_sq) / std::sqrt(ref_sq) : 0.0;
    EXPECT_LT(max_abs, 1e-6) << "Q4_0 dequant mismatch max_abs=" << max_abs;
    EXPECT_LT(rel_l2, 1e-7) << "Q4_0 dequant rel_l2 drift=" << rel_l2;
}
