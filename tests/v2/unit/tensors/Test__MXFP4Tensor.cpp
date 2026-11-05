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
#include "kernels/common/NativeVnniScaleLoad.h"
#include "tensors/MXFP4Format.h"
#include "tensors/NativeVnniFormatInfo.h"
#include "tensors/Tensors.h"
#include "tensors/VnniPackContext.h"
#include "loaders/gpu_pipeline/RepackFormat.h"
#include "utils/CPUProjectionTestWorkspace.h"

#include <array>
#include <cmath>
#include <immintrin.h>
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

TEST(MXFP4Format, NativeE8M0ScaleBitsAreExactNormalFP32)
{
    // The native E8M0 plane (issue #26) builds the scale from bits on device.
    // Over its whole domain the result must equal the reference scale and be a
    // normal FP32 value (CUDA flushes subnormals), and it must agree with the
    // current FP16 plane wherever that plane is defined.
    for (int e = 0; e < 256; ++e)
    {
        const auto byte = static_cast<uint8_t>(e);
        if (!mxfp4::exponentFitsNativeE8M0(byte))
        {
            EXPECT_TRUE(e < mxfp4::kMinNormalFP32Exponent || e > mxfp4::kMaxNativeE8M0Exponent) << e;
            continue;
        }
        const uint32_t built = mxfp4::normalScaleFP32Bits(byte);
        EXPECT_EQ(built, bits(mxfp4::scaleFP32(byte))) << e;
        EXPECT_NE(built & 0x7f800000u, 0u) << "subnormal scale for e=" << e;
        if (mxfp4::exponentFitsFP16(byte))
            EXPECT_EQ(built, bits(fp16_to_fp32(mxfp4::scaleFP16Bits(byte)))) << e;
    }
    // Every exponent the FP16 plane accepts stays accepted by the native plane.
    for (int e = mxfp4::kMinExactFP16Exponent; e <= mxfp4::kMaxExactFP16Exponent; ++e)
        EXPECT_TRUE(mxfp4::exponentFitsNativeE8M0(static_cast<uint8_t>(e))) << e;
    // Only MXFP4 stores the 1-byte native plane; every other codebook keeps FP16.
    for (int codebook = 0; codebook < 32; ++codebook)
        EXPECT_EQ(nativeVnniScalePlaneBytes(static_cast<uint8_t>(codebook)),
                  codebook == kNativeVnniMXFP4Codebook ? 1u : 2u) << codebook;
    // Prepared MXFP4 is 17 bytes per 32 weights, exactly the GGUF source size.
    const auto regions = nativeVnniPackedRegionSizes(64, 128, native_vnni_formats::MXFP4);
    EXPECT_EQ(regions.logicalBytes(), 64u * 4u * 17u);
    EXPECT_EQ(regions.scales_bytes, 64u * 4u);
}

TEST(MXFP4Format, PowerOfTwoBF16WordIsExactAndValidated)
{
    // CPU units keep a 16-bit scale slot; native E8M0 scales are stored there
    // as BF16 powers of two that widen by placement (word << 16) alone.
    for (int e = 0; e < 256; ++e)
    {
        const auto byte = static_cast<uint8_t>(e);
        if (e != 0 && !mxfp4::exponentFitsNativeE8M0(byte))
            continue;
        const uint16_t word = mxfp4::powerOfTwoBF16Word(byte);
        EXPECT_EQ(uint32_t{word} << 16, e == 0 ? 0u : mxfp4::normalScaleFP32Bits(byte)) << e;
        uint8_t back = 0xff;
        ASSERT_TRUE(mxfp4::e8m0FromPowerOfTwoBF16Word(word, back)) << e;
        EXPECT_EQ(back, byte);
    }
    // Every other word is rejected: signs, mantissa bits, and exponents outside [2, 252].
    size_t accepted = 0;
    for (uint32_t word = 0; word < 65536u; ++word)
    {
        uint8_t e = 0;
        if (!mxfp4::e8m0FromPowerOfTwoBF16Word(static_cast<uint16_t>(word), e))
            continue;
        ++accepted;
        EXPECT_EQ(mxfp4::powerOfTwoBF16Word(e), word);
    }
    EXPECT_EQ(accepted, 1u + 251u); // zero marker plus e in [2, 252]
    // Every admitted exponent keeps the largest decodable weight finite.
    EXPECT_TRUE(std::isfinite(12.0f * mxfp4::scaleFP32(mxfp4::kMaxNativeE8M0Exponent)));
    EXPECT_FALSE(std::isfinite(12.0f * mxfp4::scaleFP32(mxfp4::kMaxNativeE8M0Exponent + 1)));
}

namespace
{
    /**
     * @brief One 64-column CPU unit whose primary words are @p words.
     * @param codebook Source codebook; the scale encoding derives from it and @p encoding.
     */
    cpu::native_vnni::CPUNativeVNNIPackedWeights cpuUnitWithScales(
        const std::array<uint16_t, 64> &words, uint8_t codebook,
        cpu::native_vnni::CPUNativeVNNIEncoding encoding = cpu::native_vnni::CPUNativeVNNIEncoding::NibbleLUT)
    {
        using namespace cpu::native_vnni;
        CPUNativeVNNIPackedWeights packed;
        packed.N = packed.N_padded = 64;
        packed.K = 32;
        packed.blocks_per_row = 1;
        packed.payload_bytes = encoding == CPUNativeVNNIEncoding::NibbleLUT ? 16 : 32;
        packed.codebook_id = codebook;
        packed.encoding = encoding;
        packed.data_stride = static_cast<int>(preparedDataStride(packed.encoding));
        packed.interleaved_block_stride =
            static_cast<int>(preparedInterleavedBlockStride(packed.encoding, false));
        packed.native_interleaved.resize(static_cast<size_t>(packed.interleaved_block_stride), uint8_t{0});
        std::memcpy(const_cast<uint16_t *>(packed.chunkScales(0, 0)), words.data(), sizeof(words));
        return packed;
    }

    /** @brief Expected FP32 bits of every column through the loader under test. */
    template <class Load>
    void expectColumns(const cpu::native_vnni::CPUNativeVNNIPackedWeights &packed,
                       const std::array<uint32_t, 64> &expected, Load load)
    {
        for (int column = 0; column < 64; ++column)
            EXPECT_EQ(bits(packed.blockScale(0, 0, column)), expected[column]) << "blockScale col " << column;
        load(packed.chunkScales(0, 0), expected);
    }
} // namespace

TEST(MXFP4CPUScaleLoaders, PowerOfTwoBF16AndFP16WordsWidenExactlyOnEveryISA)
{
    using namespace cpu::native_vnni;
    // Every admitted native exponent plus the zero marker, 64 per unit.
    std::vector<uint8_t> exponents{0};
    for (int e = mxfp4::kMinNormalFP32Exponent; e <= mxfp4::kMaxNativeE8M0Exponent; ++e)
        exponents.push_back(static_cast<uint8_t>(e));
    auto vector_check = [](const uint16_t *primary, const std::array<uint32_t, 64> &expected,
                           const CPUNativeVNNIPackedWeights &packed)
    {
#if defined(__AVX512F__)
        for (int group = 0; group < 4; ++group)
        {
            alignas(64) float out[16];
            _mm512_store_ps(out, packed.loadScales512(primary + group * 16));
            for (int lane = 0; lane < 16; ++lane)
                EXPECT_EQ(bits(out[lane]), expected[group * 16 + lane]) << "512 col " << group * 16 + lane;
        }
#endif
#if defined(__AVX2__)
        for (int group = 0; group < 8; ++group)
        {
            alignas(32) float out[8];
            _mm256_store_ps(out, packed.loadScales256(primary + group * 8));
            for (int lane = 0; lane < 8; ++lane)
                EXPECT_EQ(bits(out[lane]), expected[group * 8 + lane]) << "256 col " << group * 8 + lane;
        }
#endif
        (void)primary; (void)expected; (void)packed;
    };
    for (size_t first = 0; first < exponents.size(); first += 64)
    {
        std::array<uint16_t, 64> words{};
        std::array<uint32_t, 64> expected{};
        for (int column = 0; column < 64; ++column)
        {
            const uint8_t e = exponents[(first + column) % exponents.size()];
            words[column] = mxfp4::powerOfTwoBF16Word(e);
            expected[column] = e == 0 ? 0u : mxfp4::normalScaleFP32Bits(e);
        }
        const auto packed = cpuUnitWithScales(words, kNativeVnniMXFP4Codebook);
        ASSERT_EQ(packed.scaleEncoding(), CPUNativeVNNIScaleEncoding::PowerOfTwoBF16);
        ASSERT_EQ(packed.primaryScaleLayout(), VnniPrimaryScaleLayout::PowerOfTwoBF16Words);
        expectColumns(packed, expected, [&](const uint16_t *p, const auto &x) { vector_check(p, x, packed); });
    }
    // FP16 (every other format, and rotated/expanded source-25 weights) is unchanged.
    std::array<uint16_t, 64> half_words{};
    std::array<uint32_t, 64> half_expected{};
    for (int column = 0; column < 64; ++column)
    {
        half_words[column] = static_cast<uint16_t>(0x0001u + column * 0x01f3u); // subnormals through 2^15 range
        half_expected[column] = bits(fp16_to_fp32(half_words[column]));
    }
    const auto fp16 = cpuUnitWithScales(half_words, 4);
    ASSERT_EQ(fp16.scaleEncoding(), CPUNativeVNNIScaleEncoding::FP16);
    expectColumns(fp16, half_expected, [&](const uint16_t *p, const auto &x) { vector_check(p, x, fp16); });
    // Rotated MXFP4 keeps source codebook 25 but is expanded INT8 with FP16 scales.
    const auto rotated = cpuUnitWithScales(half_words, kNativeVnniMXFP4Codebook, CPUNativeVNNIEncoding::ExpandedInt8);
    EXPECT_EQ(rotated.scaleEncoding(), CPUNativeVNNIScaleEncoding::FP16);
    EXPECT_EQ(bits(rotated.blockScale(0, 0, 3)), half_expected[3]);
}

TEST(MXFP4ScaleCarrier, StagedWordsDecodeExactlyAndFP16CallbacksAreUntouched)
{
    // Direct and staged decode agree for every admitted byte and the zero marker.
    for (int e : {0, 2, 103, 104, 128, 129, 143, 144, 252})
    {
        const auto byte = static_cast<uint8_t>(e);
        const uint16_t staged = mxfp4::powerOfTwoBF16Word(byte);
        const float decoded = native_vnni_scale::floatFromBits(uint32_t{staged} << 16);
        EXPECT_EQ(bits(decoded), e == 0 ? 0u : mxfp4::normalScaleFP32Bits(byte)) << e;
        EXPECT_EQ(bits(native_vnni_scale::e8m0Scale(byte)), bits(decoded)) << e;
    }
    // e = 129 (scale 2) stages as 0x4000: a staged word is never re-read as a raw byte.
    EXPECT_EQ(mxfp4::powerOfTwoBF16Word(129), 0x4000u);
    // 2-byte planes pass the original FP16 word to the caller's widening, unchanged.
    const uint16_t plane[3] = {0x3c00u, 0x0001u, 0x7bffu};
    int calls = 0;
    const auto widen = [&](uint16_t word) { ++calls; return fp16_to_fp32(word); };
    for (size_t linear = 0; linear < 3; ++linear)
    {
        EXPECT_EQ(native_vnni_scale::loadStagedPrimary<4>(plane, linear), plane[linear]);
        EXPECT_EQ(bits(native_vnni_scale::primaryScale<4>(plane, linear, widen)),
                  bits(fp16_to_fp32(plane[linear])));
        EXPECT_EQ(bits(native_vnni_scale::primaryScale(uint8_t{4}, plane, linear, widen)),
                  bits(fp16_to_fp32(plane[linear])));
    }
    EXPECT_EQ(calls, 6);
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
    EXPECT_EQ(regions.scales_bytes, 64u * 4u * 1u);
    EXPECT_EQ(regions.mins_bytes, 0u);
    EXPECT_TRUE(isNativeVnniFormat(TensorType::MXFP4));
    EXPECT_STREQ(tensorTypeName(TensorType::MXFP4), "MXFP4");
    // Planning resolves GGUF inventory labels through the canonical enum walk.
    EXPECT_EQ(tensorTypeFromName("MXFP4"), TensorType::MXFP4);
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

TEST(MXFP4Tensor, NativeE8M0LayoutsCoverBoundariesAndRejectUnadmittedExponents)
{
    // One row of K = 32 * 13 blocks: admitted boundaries, a zero block, and nonzero codes.
    const std::vector<int> admitted = {2, 103, 104, 128, 143, 144, 251, 252};
    const int blocks = static_cast<int>(admitted.size()) + 2;
    std::vector<uint8_t> raw(static_cast<size_t>(blocks) * sizeof(MXFP4Block), 0);
    auto *b = reinterpret_cast<MXFP4Block *>(raw.data());
    for (size_t i = 0; i < admitted.size(); ++i)
    {
        b[i].e = static_cast<uint8_t>(admitted[i]);
        for (int j = 0; j < 16; ++j)
            b[i].qs[j] = static_cast<uint8_t>(0x71 + i + j);
    }
    makeZeroBlock(raw, admitted.size(), 0);         // ggml zero block: marker 0
    makeZeroBlock(raw, admitted.size() + 1, 255, true); // zero block with unadmitted e
    MXFP4Tensor tensor({1, static_cast<size_t>(blocks) * 32}, raw);

    std::vector<uint8_t> payload(static_cast<size_t>(blocks) * 16);
    std::vector<uint16_t> words(static_cast<size_t>(blocks));
    VnniPackContext ctx{};
    ctx.N = 1;
    ctx.K = blocks * 32;
    ctx.blocks_per_row = blocks;
    ctx.payload_bytes = 16;
    ctx.payload_array = payload.data();
    ctx.scales_array = words.data();
    for (const auto layout : {VnniPrimaryScaleLayout::NativeE8M0Bytes, VnniPrimaryScaleLayout::PowerOfTwoBF16Words})
    {
        SCOPED_TRACE(static_cast<int>(layout));
        ctx.primary_layout = layout;
        std::fill(words.begin(), words.end(), uint16_t{0xa5a5});
        for (int block = 0; block < blocks; ++block)
            ASSERT_NO_THROW(tensor.packVnniBlock(ctx, 0, 0, block));
        const auto *bytes = reinterpret_cast<const uint8_t *>(words.data());
        for (int block = 0; block < blocks; ++block)
        {
            const uint8_t expected = block < static_cast<int>(admitted.size()) ? b[block].e : uint8_t{0};
            if (layout == VnniPrimaryScaleLayout::NativeE8M0Bytes)
                EXPECT_EQ(bytes[block], expected) << block;
            else
                EXPECT_EQ(words[block], mxfp4::powerOfTwoBF16Word(expected)) << block;
            EXPECT_EQ(std::memcmp(payload.data() + block * 16, b[block].qs, 16), 0) << block;
            // The staged decode equals the exact source scale for every admitted block.
            const uint16_t staged = layout == VnniPrimaryScaleLayout::NativeE8M0Bytes
                ? native_vnni_scale::loadStagedPrimary<kNativeVnniMXFP4Codebook>(words.data(), block)
                : words[block];
            const float scale = native_vnni_scale::decodeStagedPrimary<kNativeVnniMXFP4Codebook>(
                staged, [](uint16_t) { return std::nanf(""); });
            EXPECT_EQ(bits(scale), expected == 0 ? 0u : bits(mxfp4::scaleFP32(expected))) << block;
        }
        // The byte plane writes exactly one byte per block, never the neighbouring storage.
        if (layout == VnniPrimaryScaleLayout::NativeE8M0Bytes)
            for (size_t byte = static_cast<size_t>(blocks); byte < words.size() * 2; ++byte)
                EXPECT_EQ(bytes[byte], 0xa5) << byte;
    }
    // Nonzero blocks outside [2, 252] never pack under a native layout.
    for (const int e : {0, 1, 253, 254, 255})
    {
        auto bad = raw;
        reinterpret_cast<MXFP4Block *>(bad.data())[0].e = static_cast<uint8_t>(e);
        MXFP4Tensor bad_tensor({1, static_cast<size_t>(blocks) * 32}, bad);
        ctx.primary_layout = VnniPrimaryScaleLayout::NativeE8M0Bytes;
        EXPECT_THROW(bad_tensor.packVnniBlock(ctx, 0, 0, 0), std::runtime_error) << e;
        ctx.primary_layout = VnniPrimaryScaleLayout::PowerOfTwoBF16Words;
        EXPECT_THROW(bad_tensor.packVnniBlock(ctx, 0, 0, 0), std::runtime_error) << e;
    }
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

/**
 * @test Every CUDA/ROCm packed-weight uploader executes this plan. It must
 * transfer exactly the live primary-plane bytes, never the rounded storage
 * byte that follows an odd one-byte E8M0 plane, while keeping word-aligned
 * per-chunk strides. Controls: even 1-byte planes and 2-byte (FP16) planes
 * collapse to one exact copy.
 */
TEST(MXFP4ScalePlaneUpload, PlanTransfersExactLiveBytesForOddEvenAndWordPlanes)
{
    using llaminar2::kNativeVnniMXFP4Codebook;
    using llaminar2::nativeVnniPrimaryPlaneUploadPlan;
    struct Case { uint8_t codebook; size_t blocks; size_t chunks; };
    const Case cases[] = {
        {kNativeVnniMXFP4Codebook, 207, 1}, {kNativeVnniMXFP4Codebook, 207, 5},
        {kNativeVnniMXFP4Codebook, 1, 3},   {kNativeVnniMXFP4Codebook, 208, 1},
        {kNativeVnniMXFP4Codebook, 208, 5}, {4, 207, 5}, {0, 207, 1}, {kNativeVnniMXFP4Codebook, 0, 4}};
    for (const Case &c : cases)
    {
        SCOPED_TRACE(::testing::Message() << "codebook=" << int(c.codebook) << " blocks=" << c.blocks
                                          << " chunks=" << c.chunks);
        const size_t width = llaminar2::nativeVnniScalePlaneBytes(c.codebook);
        const size_t live = c.blocks * width;
        const size_t stride = llaminar2::nativeVnniScalePlaneStorageWords(c.codebook, c.blocks) * 2;
        const auto plan = nativeVnniPrimaryPlaneUploadPlan(c.codebook, c.blocks, c.chunks);
        EXPECT_EQ(plan.allocation_bytes, c.chunks * stride);
        EXPECT_EQ(plan.stride_bytes, stride);
        EXPECT_EQ(plan.transferredBytes(), c.chunks * live) << "only live bytes cross the bus";
        // Byte-level coverage: every live byte is copied exactly once and no
        // unused storage byte is ever copied.
        std::vector<int> copied(plan.allocation_bytes, 0);
        for (size_t copy = 0; copy < plan.copies; ++copy)
        {
            ASSERT_LE(plan.copyOffset(copy) + plan.copy_bytes, plan.allocation_bytes);
            for (size_t b = 0; b < plan.copy_bytes; ++b)
                ++copied[plan.copyOffset(copy) + b];
        }
        for (size_t chunk = 0; chunk < c.chunks; ++chunk)
            for (size_t b = 0; b < stride; ++b)
                EXPECT_EQ(copied[chunk * stride + b], b < live ? 1 : 0) << "chunk " << chunk << " byte " << b;
        // Exact strides need no per-chunk copies.
        if (live == stride && live != 0)
            EXPECT_EQ(plan.copies, 1u);
    }
    // The odd MXFP4 single matrix keeps one unused capacity byte that is not sent.
    const auto odd = nativeVnniPrimaryPlaneUploadPlan(kNativeVnniMXFP4Codebook, 207, 1);
    EXPECT_EQ(odd.allocation_bytes, 208u);
    EXPECT_EQ(odd.transferredBytes(), 207u);
}
