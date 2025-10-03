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
    // Build a legitimate q4_0 payload by quantizing random float data via ggml's reference quantizer.
    std::vector<uint8_t> make_quantized_q4_0_payload(size_t n_elements, uint32_t seed)
    {
        const int qk = 32;
        if (n_elements % qk != 0)
        {
            n_elements = (n_elements / qk) * qk; // enforce multiple of block for reference path
        }
        const size_t n_blocks = n_elements / qk;
        std::vector<float> src(n_elements);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : src)
            v = dist(rng);

        // Allocate block storage and invoke ggml reference quantizer
        std::vector<block_q4_0> blocks(n_blocks);
        quantize_row_q4_0_ref(src.data(), blocks.data(), (int64_t)n_elements);

        // Serialize contiguous bytes (struct is tightly packed: 2 + 16 = 18 bytes)
        std::vector<uint8_t> bytes(n_blocks * (sizeof(uint16_t) + 16));
        std::memcpy(bytes.data(), blocks.data(), bytes.size());
        return bytes;
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
    const size_t n = 320; // 10 blocks (multiple of 32)
    auto payload = make_quantized_q4_0_payload(n, 12345);

    ModelLoader loader; // Not loading a full model; we just want the helper path.
    auto ref = loader.dequantizeQ4_0(payload.data(), n);
    ASSERT_EQ(ref.size(), n);

    std::vector<float> fused(n, 0.0f);
    llaminar::dequant_q4_0_rows(payload.data(), fused.data(), n);

    double max_abs = 0.0, diff_sq = 0.0, ref_sq = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        double d = static_cast<double>(ref[i]) - fused[i];
        max_abs = std::max(max_abs, std::fabs(d));
        diff_sq += d * d;
        ref_sq += static_cast<double>(ref[i]) * ref[i];
    }
    double rel_l2 = ref_sq > 0.0 ? std::sqrt(diff_sq) / std::sqrt(ref_sq) : 0.0;
    if (!(max_abs < 1e-7 && rel_l2 < 1e-8))
    {
        // Dump detailed diagnostics for the first block to help isolate structural mismatch.
        const size_t qk = 32;
        const block_q4_0 *blocks = reinterpret_cast<const block_q4_0 *>(payload.data());
        float ref_block[32];
        float fused_block[32];
        // Use upstream reference to dequantize only first block
        dequantize_row_q4_0(blocks, ref_block, qk);
        // Use fused helper on just the first block bytes
        llaminar::dequant_block_q4_0(reinterpret_cast<const uint8_t *>(blocks), fused_block, 32);
        uint16_t scale_bits = blocks[0].d;
        float scale_ref = ggml_fp16_to_fp32(scale_bits);
        fprintf(stderr, "Q4_0 parity diag: scale_bits=0x%04x scale_ref=%g\n", scale_bits, scale_ref);
        fprintf(stderr, " ref_block[0..15]:");
        for (int i = 0; i < 16; ++i)
            fprintf(stderr, " %g", ref_block[i]);
        fprintf(stderr, "\n fused_block[0..15]:");
        for (int i = 0; i < 16; ++i)
            fprintf(stderr, " %g", fused_block[i]);
        fprintf(stderr, "\n ref_block[16..31]:");
        for (int i = 16; i < 32; ++i)
            fprintf(stderr, " %g", ref_block[i]);
        fprintf(stderr, "\n fused_block[16..31]:");
        for (int i = 16; i < 32; ++i)
            fprintf(stderr, " %g", fused_block[i]);
        fprintf(stderr, "\n first 8 element diffs (ref - fused):");
        for (int i = 0; i < 8; ++i)
            fprintf(stderr, " %g", ref_block[i] - fused_block[i]);
        fprintf(stderr, "\n");
    }
    // Allow exact bitwise parity threshold (values should be identical); relax slightly for safety.
    EXPECT_LT(max_abs, 1e-7) << "Q4_0 dequant mismatch max_abs=" << max_abs;
    EXPECT_LT(rel_l2, 1e-8) << "Q4_0 dequant rel_l2 drift=" << rel_l2;
}
