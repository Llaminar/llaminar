/**
 * @file Test__MXFP4Tensor.cpp
 * @brief Device-free regressions for the OCP MXFP4 (GGUF type 39) weight format.
 *
 * MXFP4 weights (DeepSeek-V4 / gpt-oss routed experts) must reproduce their
 * original FP4 E2M1 + E8M0 values exactly. These tests pin:
 *  - the shared format constants (doubled E2M1 lookup, exact FP16 block scale);
 *  - MXFP4Tensor decode, Q8_0 re-encoding, rank-three expert views;
 *  - NativeVNNI preparation (verbatim codes, exact FP16 scale, hard rejection
 *    of exponents without an exact FP16 scale, except all-zero blocks such as
 *    the e = 0 blocks ggml's quantizer writes, which prepare with scale +0);
 *  - the CPU NativeVNNI GEMV/GEMM on MXFP4 weights against an exact FP64
 *    reference, using activations whose Q8 quantization is lossless so the
 *    only admissible error is FP32 accumulation rounding.
 */
#include <gtest/gtest.h>

#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include "tensors/MXFP4Format.h"
#include "tensors/NativeVnniFormatInfo.h"
#include "tensors/Tensors.h"
#include "tensors/VnniPackContext.h"
#include "loaders/gpu_pipeline/RepackFormat.h"
#include "utils/CPUProjectionTestWorkspace.h"

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @brief Exact reference value of element @p j of @p block. */
    double referenceValue(const MXFP4Block &block, int j)
    {
        const uint8_t code = j < 16 ? (block.qs[j] & 0x0F) : (block.qs[j - 16] >> 4);
        // E2M1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} times 2^(e-127).
        static constexpr double kE2M1[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
        const double magnitude = kE2M1[code & 7];
        return (code & 8 ? -magnitude : magnitude) * std::ldexp(1.0, static_cast<int>(block.e) - 127);
    }

    /** @brief Random blocks with exponents in [e_lo, e_hi]. */
    std::vector<uint8_t> randomBlocks(size_t blocks, uint32_t seed, int e_lo = 112, int e_hi = 130)
    {
        std::mt19937 rng(seed);
        std::vector<uint8_t> raw(blocks * sizeof(MXFP4Block));
        auto *b = reinterpret_cast<MXFP4Block *>(raw.data());
        for (size_t i = 0; i < blocks; ++i)
        {
            b[i].e = static_cast<uint8_t>(e_lo + static_cast<int>(rng() % static_cast<uint32_t>(e_hi - e_lo + 1)));
            for (auto &q : b[i].qs) q = static_cast<uint8_t>(rng());
        }
        return raw;
    }

    /**
     * @brief Turn block @p index into an all-zero block the way ggml's quantizer writes it (e = 0).
     * @param negative_zero Use code 8 (-0) for every element instead of code 0.
     */
    void makeZeroBlock(std::vector<uint8_t> &raw, size_t index, uint8_t e = 0, bool negative_zero = false)
    {
        auto &block = reinterpret_cast<MXFP4Block *>(raw.data())[index];
        block.e = e;
        std::memset(block.qs, negative_zero ? 0x88 : 0x00, sizeof(block.qs));
    }

    uint32_t bits(float v)
    {
        uint32_t out;
        std::memcpy(&out, &v, sizeof(out));
        return out;
    }
} // namespace

TEST(MXFP4Format, LookupAndScalesAreExact)
{
    static constexpr double kE2M1[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    for (int code = 0; code < 16; ++code)
        EXPECT_EQ(static_cast<double>(mxfp4::kValues[code]), 2.0 * (code & 8 ? -kE2M1[code & 7] : kE2M1[code & 7]));
    for (int e = 0; e < 256; ++e)
    {
        const auto byte = static_cast<uint8_t>(e);
        EXPECT_EQ(mxfp4::scaleFP32(byte), static_cast<float>(std::ldexp(1.0, e - 128))) << e;
        if (!mxfp4::exponentFitsFP16(byte))
            continue;
        EXPECT_EQ(bits(fp16_to_fp32(mxfp4::scaleFP16Bits(byte))), bits(mxfp4::scaleFP32(byte))) << e;
    }
    EXPECT_FALSE(mxfp4::exponentFitsFP16(mxfp4::kMinExactFP16Exponent - 1));
    EXPECT_FALSE(mxfp4::exponentFitsFP16(mxfp4::kMaxExactFP16Exponent + 1));

    // Only blocks whose every code is +-0 escape the exponent window.
    uint8_t qs[16] = {};
    EXPECT_TRUE(mxfp4::blockFitsFP16(0, qs));
    std::memset(qs, 0x88, sizeof(qs));
    EXPECT_TRUE(mxfp4::blockFitsFP16(255, qs));
    qs[15] = 0x18; // one element with code 1
    EXPECT_FALSE(mxfp4::blockFitsFP16(0, qs));
    EXPECT_TRUE(mxfp4::blockFitsFP16(128, qs));
    EXPECT_EQ(mxfp4::preparedScaleFP16Bits(0), 0u);
    EXPECT_EQ(mxfp4::preparedScaleFP16Bits(128), mxfp4::scaleFP16Bits(128));
}

TEST(MXFP4Format, CatalogAndRepackIdentity)
{
    const auto *format = native_vnni_formats::forQuantType("MXFP4");
    ASSERT_EQ(format, &native_vnni_formats::MXFP4);
    EXPECT_EQ(format->codebook_id, kNativeVnniMXFP4Codebook);
    EXPECT_EQ(format->payload_bytes, 16);
    EXPECT_FALSE(format->is_asymmetric);
    EXPECT_EQ(native_vnni_formats::forSourceIdentity(kNativeVnniMXFP4Codebook, false), format);
    // Own physical decoder, IQ4_NL arithmetic policy, verbatim CPU-tier round trip.
    EXPECT_EQ(canonicalDeviceVnniCodebookId(*format), kNativeVnniMXFP4Codebook);
    EXPECT_EQ(canonicalDeviceVnniCodebookId(kNativeVnniMXFP4Codebook), 4);
    EXPECT_EQ(nativeVnniPolicyCodebook(kNativeVnniMXFP4Codebook), 4);
    EXPECT_EQ(migrationStableDeviceVnniFormat(*format).codebook_id, kNativeVnniMXFP4Codebook);
    EXPECT_FALSE(areNativeVnniGateUpCodebooksCompatible(4, kNativeVnniMXFP4Codebook));
    EXPECT_EQ(codebookIdToRepackFormat(kNativeVnniMXFP4Codebook, false), RepackFormat::MXFP4);
    EXPECT_EQ(repackPayloadBytesPerBlock(RepackFormat::MXFP4), 16);
    const auto regions = nativeVnniPackedRegionSizes(64, 128, *format);
    EXPECT_EQ(regions.payload_bytes, 64u * 4u * 16u);
    EXPECT_EQ(regions.scales_bytes, 64u * 4u * 2u);
    EXPECT_EQ(regions.mins_bytes, 0u);
    EXPECT_TRUE(isNativeVnniFormat(TensorType::MXFP4));
    EXPECT_STREQ(tensorTypeName(TensorType::MXFP4), "MXFP4");
}

TEST(MXFP4Tensor, DecodeQ8AndExpertViewsAreExact)
{
    // GGUF rank-three expert tensor [K, N, E] = [64, 3, 4] -> 12 rows of K=64.
    constexpr size_t K = 64, N = 3, E = 4, rows = N * E, bpr = K / 32;
    auto raw = randomBlocks(rows * bpr, 7);
    auto tensor = std::make_shared<MXFP4Tensor>(std::vector<size_t>{K, N, E}, raw);
    const auto *blocks = reinterpret_cast<const MXFP4Block *>(raw.data());

    std::vector<float> row(K);
    for (size_t r = 0; r < rows; ++r)
    {
        tensor->to_fp32_row(r, row.data());
        for (size_t k = 0; k < K; ++k)
            ASSERT_EQ(static_cast<double>(row[k]), referenceValue(blocks[r * bpr + k / 32], static_cast<int>(k % 32)))
                << "row " << r << " col " << k;
        for (size_t kb = 0; kb < bpr; ++kb)
        {
            Q8_0Block q8{};
            tensor->decode_to_q8_0(r, kb, &q8);
            const float d = fp16_to_fp32(q8.d);
            for (int j = 0; j < 32; ++j)
                ASSERT_EQ(static_cast<double>(d * static_cast<float>(q8.qs[j])),
                          referenceValue(blocks[r * bpr + kb], j));
        }
    }

    // A row view of expert 2 (rows 6..8) decodes the same bytes.
    auto view = tensor->create_view({N, K}, 2 * N * K);
    ASSERT_NE(view, nullptr);
    std::vector<float> expected(K), actual(K);
    for (size_t r = 0; r < N; ++r)
    {
        tensor->to_fp32_row(2 * N + r, expected.data());
        view->to_fp32_row(r, actual.data());
        EXPECT_EQ(std::memcmp(expected.data(), actual.data(), K * sizeof(float)), 0);
    }
    EXPECT_THROW(MXFP4Tensor({K, N, E}, std::vector<uint8_t>(raw.size() - 1)), std::invalid_argument);
}

TEST(MXFP4Tensor, NativeVnniPackingIsVerbatimAndRejectsInexactScales)
{
    constexpr int N = 2, K = 64, bpr = K / 32;
    auto raw = randomBlocks(N * bpr, 11);
    MXFP4Tensor tensor({N, K}, raw);
    ASSERT_EQ(tensor.vnniFormatInfo(), &native_vnni_formats::MXFP4);

    std::vector<uint8_t> payload(N * bpr * 16);
    std::vector<uint16_t> scales(N * bpr);
    VnniPackContext ctx{};
    ctx.N = N;
    ctx.K = K;
    ctx.blocks_per_row = bpr;
    ctx.payload_bytes = 16;
    ctx.payload_array = payload.data();
    ctx.scales_array = scales.data();
    const auto *blocks = reinterpret_cast<const MXFP4Block *>(raw.data());
    for (int n = 0; n < N; ++n)
        for (int b = 0; b < bpr; ++b)
        {
            tensor.packVnniBlock(ctx, n, n, b);
            const size_t linear = vnniLinearIdx(ctx, n, b);
            EXPECT_EQ(std::memcmp(payload.data() + linear * 16, blocks[n * bpr + b].qs, 16), 0);
            EXPECT_EQ(scales[linear], mxfp4::scaleFP16Bits(blocks[n * bpr + b].e));
        }

    // An exponent whose scale is not an exact FP16 value never packs.
    auto bad = raw;
    reinterpret_cast<MXFP4Block *>(bad.data())[1].e = 90;
    MXFP4Tensor bad_tensor({N, K}, bad);
    EXPECT_THROW(bad_tensor.packVnniBlock(ctx, 0, 0, 1), std::runtime_error);
    EXPECT_THROW(
        {
            Q8_0Block q8{};
            bad_tensor.decode_to_q8_0(0, 1, &q8);
        },
        std::runtime_error);
    // A nonzero block keeps its rejection even at ggml's zero-block exponent.
    reinterpret_cast<MXFP4Block *>(bad.data())[1].e = 0;
    MXFP4Tensor nonzero_e0({N, K}, bad);
    EXPECT_THROW(nonzero_e0.packVnniBlock(ctx, 0, 0, 1), std::runtime_error);
}

TEST(MXFP4Tensor, GgmlAllZeroBlocksPrepareExactlyWithZeroScale)
{
    // ggml quantize_row_mxfp4_ref writes e = 0 (2^-128, no FP16 form) for an
    // all-zero block; llama.cpp loads such files, so preparation must as well.
    constexpr int N = 2, K = 64, bpr = K / 32;
    auto raw = randomBlocks(N * bpr, 13);
    makeZeroBlock(raw, 1);
    makeZeroBlock(raw, 2, 255, true);
    MXFP4Tensor tensor({N, K}, raw);

    std::vector<uint8_t> payload(N * bpr * 16);
    std::vector<uint16_t> scales(N * bpr);
    VnniPackContext ctx{};
    ctx.N = N;
    ctx.K = K;
    ctx.blocks_per_row = bpr;
    ctx.payload_bytes = 16;
    ctx.payload_array = payload.data();
    ctx.scales_array = scales.data();
    const auto *blocks = reinterpret_cast<const MXFP4Block *>(raw.data());
    for (int n = 0; n < N; ++n)
        for (int b = 0; b < bpr; ++b)
        {
            ASSERT_NO_THROW(tensor.packVnniBlock(ctx, n, n, b));
            const size_t linear = vnniLinearIdx(ctx, n, b);
            const size_t source = static_cast<size_t>(n) * bpr + b;
            EXPECT_EQ(std::memcmp(payload.data() + linear * 16, blocks[source].qs, 16), 0);
            EXPECT_EQ(scales[linear], source == 1 || source == 2 ? 0u : mxfp4::scaleFP16Bits(blocks[source].e));
        }

    for (size_t source : {size_t{1}, size_t{2}})
    {
        Q8_0Block q8{};
        ASSERT_NO_THROW(tensor.decode_to_q8_0(source / bpr, source % bpr, &q8));
        EXPECT_EQ(q8.d, 0u);
        for (int j = 0; j < 32; ++j)
            EXPECT_EQ(q8.qs[j], 0) << "block " << source << " element " << j;
    }
    std::vector<float> row(K);
    for (size_t r = 0; r < N; ++r)
    {
        tensor.to_fp32_row(r, row.data());
        for (size_t k = 0; k < K; ++k)
            ASSERT_EQ(static_cast<double>(row[k]), referenceValue(blocks[r * bpr + k / 32], static_cast<int>(k % 32)));
    }
}

namespace
{
    /**
     * @brief Activations whose per-32 Q8 quantization is lossless.
     *
     * Every block holds integers in [-127, 127] times 2^-7 and contains one
     * magnitude-127 entry, so the activation scale is exactly 2^-7 and the
     * integer codes are the integers themselves.
     */
    std::vector<float> losslessQ8Activations(int M, int K, uint32_t seed)
    {
        std::mt19937 rng(seed);
        std::vector<float> a(static_cast<size_t>(M) * K);
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
            {
                const int q = (k % 32 == 0) ? ((rng() & 1) ? 127 : -127)
                                            : static_cast<int>(rng() % 255) - 127;
                a[static_cast<size_t>(m) * K + k] = std::ldexp(static_cast<float>(q), -7);
            }
        return a;
    }

    /**
     * @brief CPU NativeVNNI projection against the exact FP64 reference.
     * @param ggml_zero_blocks Replace every seventh block with a ggml all-zero block (e = 0 or 255).
     */
    void runCpuProjection(int M, int N, int K, uint32_t seed, bool ggml_zero_blocks = false)
    {
        auto raw = randomBlocks(static_cast<size_t>(N) * (K / 32), seed);
        if (ggml_zero_blocks)
            for (size_t block = 0; block < raw.size() / sizeof(MXFP4Block); block += 7)
                makeZeroBlock(raw, block, block % 2 ? 255 : 0, block % 3 == 0);
        MXFP4Tensor weights({static_cast<size_t>(N), static_cast<size_t>(K)}, raw);
        cpu::native_vnni::CPUNativeVNNIGemmKernel kernel(&weights);
        ASSERT_TRUE(kernel.isValid());

        const auto a = losslessQ8Activations(M, K, seed + 1);
        FP32Tensor A({static_cast<size_t>(M), static_cast<size_t>(K)});
        std::memcpy(A.mutable_data(), a.data(), a.size() * sizeof(float));
        FP32Tensor C({static_cast<size_t>(M), static_cast<size_t>(N)});
        test::CPUProjectionTestWorkspace workspace(M, K, test::cpuProjectionTestRequirements(M, {&kernel}));
        ASSERT_TRUE(kernel.multiply_tensor(&A, &C, M, N, K, true, 1.0f, 0.0f, nullptr, nullptr, -1,
                                           workspace.get()));

        std::vector<float> w(K);
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n)
            {
                weights.to_fp32_row(static_cast<size_t>(n), w.data());
                double exact = 0.0, magnitude = 0.0;
                for (int k = 0; k < K; ++k)
                {
                    const double term = static_cast<double>(w[k]) * a[static_cast<size_t>(m) * K + k];
                    exact += term;
                    magnitude += std::abs(term);
                }
                // Only FP32 rounding of per-block products and partial sums remains.
                const double got = C.data()[static_cast<size_t>(m) * N + n];
                EXPECT_LE(std::abs(got - exact), 1e-6 * magnitude + 1e-30)
                    << "M=" << M << " m=" << m << " n=" << n << " got=" << got << " exact=" << exact;
            }
    }
} // namespace

TEST(MXFP4CPUNativeVNNI, GemvMatchesExactReference) { runCpuProjection(1, 192, 512, 21); }

TEST(MXFP4CPUNativeVNNI, GemmMatchesExactReference)
{
    for (int M : {2, 3, 5, 17})
        runCpuProjection(M, 128, 256, 31u + static_cast<uint32_t>(M));
}

TEST(MXFP4CPUNativeVNNI, GgmlAllZeroBlocksMatchExactReference)
{
    for (int M : {1, 5})
        runCpuProjection(M, 128, 256, 41u + static_cast<uint32_t>(M), true);
}
