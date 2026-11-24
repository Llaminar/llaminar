/**
 * @file Test__CPUNativeExtremeByteProjection.cpp
 * @brief All 22 GGUF source formats project signed-byte endpoints exactly.
 *
 * Author original GGUF blocks, not CPU interleaved bytes. The scalar oracle
 * stages the source codec's per-block words and widens its signed products to
 * int64, independently of the production interleave, bias, and SIMD dot.
 */
#include "kernels/cpu/gemm/CPUNativeQuantisedGemv.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedWeightPacker.h"
#include "tensors/Tensors.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace llaminar2::test
{
namespace
{
    using namespace cpu::native_quantised;
    namespace formats = native_quantised_formats;
    namespace exact = device_fp32_contract;
    using Policy = CPUProjectionNumericalPolicy;
    constexpr int kN = 67, kK = 288, kBlocks = kK / 32, kRows = 4;
    constexpr int kStride = kN + 5;

    /** @brief Build a live-N ragged-K matrix of original source blocks.
     * @tparam Tensor GGUF storage owner that runs the original source packer.
     * @tparam Block Original on-disk block structure.
     * @param fill Author endpoint payload and nonzero factors for one block.
     * @return CPU tensor with a final partial superblock where applicable.
     */
    template<class Tensor, class Block, class Fill>
    std::unique_ptr<TensorBase> ggufSource(Fill fill)
    {
        constexpr int per_row = (kK + Block::BLOCK_SIZE - 1) / Block::BLOCK_SIZE;
        std::vector<std::uint8_t> raw(kN * per_row * sizeof(Block));
        for (int n = 0; n < kN; ++n)
            for (int b = 0; b < per_row; ++b)
            {
                Block block{};
                // Both endpoints occur in every K block across adjacent columns;
                // the next source block interchanges them without changing the grid.
                const bool positive = (n + b) % 2 != 0;
                const auto d = static_cast<std::uint16_t>(0x1d35 + 13 * n + 17 * b);
                fill(block, positive, d);
                std::memcpy(raw.data() + (n * per_row + b) * sizeof(Block), &block, sizeof(Block));
            }
        return std::make_unique<Tensor>(std::vector<std::size_t>{kN, kK}, raw);
    }

    /** @brief Generate GGUF payload codes at each format's integer endpoints.
     * @param format Exact catalog identity, including source aliases.
     * @return Original tensor, or null for an uncovered catalog entry (test failure).
     * @details Affine formats have unsigned dots; nonzero minima put the low
     * endpoint below zero. IQ grid indices select at least one maximum-magnitude
     * lane, with both legal endpoint signs. IQ1 uses all -1/+1 grids and deltas.
     */
    std::unique_ptr<TensorBase> extremeSource(const NativeQuantisedFormatInfo &format)
    {
        using namespace formats;
        const auto nibble = [](auto &w, bool pos, std::uint16_t d) {
            w.d = d;
            std::memset(w.qs, pos ? 0xff : 0, sizeof(w.qs));
        };
        const auto affine = [&](auto &w, bool pos, std::uint16_t d) {
            nibble(w, pos, d);
            w.m = fp32_to_fp16(-3 * fp16_to_fp32(d));
        };
        const auto q8 = [](auto &w, bool pos, std::uint16_t d) {
            w.d = d;
            std::fill_n(w.qs, std::size(w.qs), pos ? 127 : -127);
        };
        const auto kAffine = [&](auto &w, bool pos, std::uint16_t d) {
            nibble(w, pos, d);
            w.dmin = fp32_to_fp16(0.5f * fp16_to_fp32(d));
            std::memset(w.scales, 0xff, sizeof(w.scales)); // both 6-bit selectors = 63
        };
        if (&format == &IQ4_NL) return ggufSource<IQ4_NLTensor, IQ4_NLBlock>(nibble);
        if (&format == &Q4_0) return ggufSource<Q4_0Tensor, Q4_0Block>(nibble);
        if (&format == &Q4_1) return ggufSource<Q4_1Tensor, Q4_1Block>(affine);
        if (&format == &Q5_0) return ggufSource<Q5_0Tensor, Q5_0Block>([&](auto &w, bool pos, auto d) {
            nibble(w, pos, d);
            std::memset(w.qh, pos ? 0xff : 0, sizeof(w.qh));
        });
        if (&format == &Q5_1) return ggufSource<Q5_1Tensor, Q5_1Block>([&](auto &w, bool pos, auto d) {
            affine(w, pos, d);
            std::memset(w.qh, pos ? 0xff : 0, sizeof(w.qh));
        });
        if (&format == &Q8_0) return ggufSource<Q8_0Tensor, Q8_0Block>(q8);
        if (&format == &Q8_1) return ggufSource<Q8_1Tensor, Q8_1Block>([&](auto &w, bool pos, auto d) {
            q8(w, pos, d);
            w.sum_qs = (pos ? 127 : -127) * 32;
        });
        if (&format == &Q8_K) return ggufSource<Q8_KTensor, Q8_KBlock>([](auto &w, bool pos, auto d) {
            const int q = pos ? 127 : -127;
            w.d = fp16_to_fp32(d);
            std::fill_n(w.qs, std::size(w.qs), q);
            std::fill_n(w.bsums, std::size(w.bsums), 16 * q);
        });
        if (&format == &MXFP4) return ggufSource<MXFP4Tensor, MXFP4Block>([](auto &w, bool pos, auto d) {
            w.e = 118 + (d % 5);
            std::memset(w.qs, pos ? 0x77 : 0xff, sizeof(w.qs));
        });
        if (&format == &Q4_K) return ggufSource<Q4_KTensor, Q4_KBlock>(kAffine);
        if (&format == &Q5_K) return ggufSource<Q5_KTensor, Q5_KBlock>([&](auto &w, bool pos, auto d) {
            kAffine(w, pos, d);
            std::memset(w.qh, pos ? 0xff : 0, sizeof(w.qh));
        });
        if (&format == &Q2_K) return ggufSource<Q2_KTensor, Q2_KBlock>([&](auto &w, bool pos, auto d) {
            nibble(w, pos, d);
            w.dmin = fp32_to_fp16(0.5f * fp16_to_fp32(d));
            for (int i = 0; i < 16; ++i) w.scales[i] = i % 2 ? 0x7e : 0xf7;
        });
        if (&format == &Q3_K) return ggufSource<Q3_KTensor, Q3_KBlock>([&](auto &w, bool pos, auto d) {
            nibble(w, pos, d);
            std::memset(w.hmask, pos ? 0xff : 0, sizeof(w.hmask));
            std::memset(w.scales, 0xff, sizeof(w.scales)); // original signed coefficient +31
        });
        if (&format == &Q6_K) return ggufSource<Q6_KTensor, Q6_KBlock>([](auto &w, bool pos, auto d) {
            w.d = d;
            std::memset(w.ql, pos ? 0xff : 0, sizeof(w.ql));
            std::memset(w.qh, pos ? 0xff : 0, sizeof(w.qh));
            for (int i = 0; i < 16; ++i) w.scales[i] = i % 2 ? 127 : 31;
        });
        if (&format == &IQ4_XS) return ggufSource<IQ4_XSTensor, IQ4_XSBlock>([&](auto &w, bool pos, auto d) {
            nibble(w, pos, d);
            w.scales_h = 0xffff;
            std::memset(w.scales_l, 0x7f, sizeof(w.scales_l)); // alternating positive coefficients
        });
        if (&format == &IQ3_S) return ggufSource<IQ3_STensor, IQ3_SBlock>([](auto &w, bool pos, auto d) {
            w.d = d;
            std::memset(w.qs, 261 & 255, sizeof(w.qs)); // grid 261 contains 15
            std::memset(w.qh, 0xff, sizeof(w.qh));
            std::memset(w.signs, pos ? 0 : 0xff, sizeof(w.signs));
            std::memset(w.scales, 0xf5, sizeof(w.scales));
        });
        if (&format == &IQ3_XXS) return ggufSource<IQ3_XXSTensor, IQ3_XXSBlock>([](auto &w, bool pos, auto d) {
            w.d = d;
            std::memset(w.qs, 135, 64); // grid 135 contains 62
            // Sign code 127 expands to all eight sign bits set (even parity).
            const std::uint32_t signs = pos ? 0xf0000000u : 0xffffffffu;
            for (int i = 0; i < 8; ++i) std::memcpy(w.qs + 64 + 4 * i, &signs, 4);
        });
        if (&format == &IQ2_S) return ggufSource<IQ2_STensor, IQ2_SBlock>([](auto &w, bool pos, auto d) {
            w.d = d;
            std::memset(w.qs, 0xff, 32); // grid 1023 is all 43
            std::memset(w.qh, 0xff, sizeof(w.qh));
            std::memset(w.qs + 32, pos ? 0 : 0xff, 32);
            std::memset(w.scales, 0xf7, sizeof(w.scales));
        });
        if (&format == &IQ2_XS) return ggufSource<IQ2_XSTensor, IQ2_XSBlock>([](auto &w, bool pos, auto d) {
            w.d = d;
            const std::uint16_t entry = 511 | ((pos ? 0 : 127) << 9); // grid 511 is all 43
            for (int i = 0; i < 32; ++i) std::memcpy(&w.qs[i], &entry, 2);
            std::memset(w.scales, 0xf7, sizeof(w.scales));
        });
        if (&format == &IQ2_XXS) return ggufSource<IQ2_XXSTensor, IQ2_XXSBlock>([](auto &w, bool pos, auto d) {
            w.d = d;
            const std::uint32_t signs = pos ? 0xf0000000u : 0xffffffffu;
            auto *bytes = reinterpret_cast<std::uint8_t *>(w.qs);
            for (int i = 0; i < 8; ++i)
            {
                std::memset(bytes + 8 * i, 57, 4); // grid 57 contains 43
                std::memcpy(bytes + 8 * i + 4, &signs, 4);
            }
        });
        if (&format == &IQ1_S) return ggufSource<IQ1_STensor, IQ1_SBlock>([](auto &w, bool pos, auto d) {
            w.d = d;
            std::memset(w.qs, pos ? 0xff : 0, sizeof(w.qs)); // grid 2047 or 0
            const std::uint16_t word = pos ? 0x7fff : 0xf000;
            for (int i = 0; i < 8; ++i) std::memcpy(&w.qh[i], &word, 2);
        });
        if (&format == &IQ1_M) return ggufSource<IQ1_MTensor, IQ1_MBlock>([](auto &w, bool pos, auto d) {
            std::memset(w.qs, pos ? 0xff : 0, sizeof(w.qs)); // grid 2047 or 0
            std::memset(w.qh, pos ? 0x77 : 0x88, sizeof(w.qh)); // +/- delta accompanies +/- grid
            for (int i = 0; i < 4; ++i)
            {
                const std::uint16_t word = ((d >> (4 * i)) & 15) << 12 | 0x0f3c;
                std::memcpy(w.scales + 2 * i, &word, 2);
            }
        });
        return {};
    }

    /** @brief Decode one original source block through the codec, without CPU prepared addressing. */
    struct ScalarBlock
    {
        std::array<std::int8_t, 32> bytes{};
        std::array<std::uint8_t, 32> payload{};
        std::uint16_t primary = 0, secondary = 0;
        std::uint32_t minima = 0;
    };

    /** @param source Original tensor's source packer.
     * @param column Live output column. @param kb Logical 32-value block.
     * @return Source-ordered signed bytes and original staged codec metadata.
     */
    ScalarBlock stageSource(const IINT8Unpackable &source, int column, int kb)
    {
        ScalarBlock w;
        const auto &format = *source.vnniFormatInfo();
        source.packVnniBlock({.N = 1, .K = kK, .blocks_per_row = kBlocks,
            .payload_bytes = format.payload_bytes, .payload_array = w.payload.data(),
            .scales_array = &w.primary, .mins_array = &w.secondary, .emins_array = &w.minima,
            .destination_block_origin = kb, .destination_block_count = 1}, column, 0, kb);
        if (format.payload_bytes == 32)
            std::memcpy(w.bytes.data(), w.payload.data(), w.bytes.size()); // Q8 source aliases
        else if (&format == &formats::Q6_K)
            for (int i = 0; i < 32; ++i)
                w.bytes[i] = ((w.payload[i % 16] >> (i / 16 * 4)) & 15) +
                    16 * ((w.payload[16 + i / 4] >> (i % 4 * 2)) & 3) - 32;
        else
            decode_native_block(format.codebook_id, w.payload.data(), w.bytes.data());
        return w;
    }

    /** @return Signed dot endpoints, before the affine offset or source scale. */
    std::pair<int, int> endpoints(const NativeQuantisedFormatInfo &format)
    {
        using namespace formats;
        if (format.payload_bytes == 32) return {-127, 127};
        if (&format == &Q4_0) return {-8, 7};
        if (&format == &Q4_1 || &format == &Q4_K) return {0, 15};
        if (&format == &Q5_0) return {-16, 15};
        if (&format == &Q5_1 || &format == &Q5_K) return {0, 31};
        if (&format == &Q6_K) return {-32, 31};
        if (&format == &Q3_K) return {-4, 3};
        if (&format == &Q2_K) return {0, 3};
        if (&format == &IQ4_NL || &format == &IQ4_XS) return {-127, 113};
        if (&format == &IQ2_S || &format == &IQ2_XS || &format == &IQ2_XXS) return {-43, 43};
        if (&format == &IQ3_S) return {-15, 15};
        if (&format == &IQ3_XXS) return {-62, 62};
        if (&format == &IQ1_S || &format == &IQ1_M) return {-1, 1};
        if (&format == &MXFP4) return {-12, 12};
        throw std::invalid_argument("No signed dot endpoints for this source format");
    }

    /** @brief Append one source-codec block using int64 products and explicit FP rounding edges.
     * @param format Source identity, not the prepared CPU encoding.
     * @param w Original codec words and signed weights. @param a Published activation block.
     * @param acc Prior accumulator in this K partition. @param policy Numerical contract under test.
     * @return FP32 accumulator under the format's original contribution program.
     */
    float scalarAppend(const NativeQuantisedFormatInfo &format, const ScalarBlock &w,
        const Q8_1Block &a, float acc, Policy policy)
    {
        std::array<std::int64_t, 2> dot{};
        std::array<int, 4> sum{};
        for (int i = 0; i < 32; ++i)
        {
            dot[i / 16] += std::int64_t(a.qs[i]) * w.bytes[i];
            sum[i / 8] += a.qs[i];
        }
        const float ad = nativeQuantisedFP16ScaleToFP32(a.d);
        const float d = fp16_to_fp32(w.primary);
        float low = d, high = fp16_to_fp32(w.secondary);
        using namespace formats;
        using exact::add;
        using exact::multiply;
        if (&format == &Q6_K)
        {
            low = multiply(d, native_quantised_q6_scale::low(w.secondary));
            high = multiply(d, native_quantised_q6_scale::high(w.secondary));
        }
        if (&format == &IQ2_S)
        {
            low = multiply(multiply(d, native_quantised_iq2s_scale::low(w.secondary) + 0.5f), 0.25f);
            high = multiply(multiply(d, native_quantised_iq2s_scale::high(w.secondary) + 0.5f), 0.25f);
        }
        if (&format == &Q6_K || &format == &Q2_K || &format == &Q3_K ||
            &format == &IQ2_S || &format == &IQ2_XS || &format == &IQ1_M)
        {
            float term = multiply(ad, add(multiply(low, float(dot[0])), multiply(high, float(dot[1]))));
            if (&format == &Q2_K)
                term = add(term, multiply(ad, add(
                    multiply(fp16_to_fp32(std::uint16_t(w.minima)), float(sum[0] + sum[1])),
                    multiply(fp16_to_fp32(std::uint16_t(w.minima >> 16)), float(sum[2] + sum[3])))));
            if (&format == &IQ1_M)
            {
                const auto delta = [&](int i) {
                    return multiply((w.payload[4 + i / 2] & (8 << (4 * (i % 2)))) ? -0.125f : 0.125f, float(sum[i]));
                };
                term = add(term, multiply(ad, add(multiply(add(delta(0), delta(1)), low),
                                                  multiply(add(delta(2), delta(3)), high))));
            }
            return add(acc, term);
        }
        if (&format == &IQ3_S) low = multiply(d, native_quantised_iq3s_scale::coefficient(w.secondary));
        if (&format == &IQ4_XS) low = multiply(d, native_quantised_iq4xs_scale::coefficient(w.secondary));
        const float scaled_weight = multiply(ad, low);
        const float signed_dot = float(dot[0] + dot[1]);
        const float whole_sum = float(sum[0] + sum[1] + sum[2] + sum[3]);
        const bool corrected = &format == &IQ1_S || format.is_asymmetric;
        // Legacy compact IQ1_S/IQ2_XXS/IQ3_XXS use the rounded block on both policies.
        if (policy == Policy::BackendNative && &format != &IQ1_S && &format != &IQ2_XXS && &format != &IQ3_XXS)
        {
            const float result = std::fma(signed_dot, scaled_weight, acc);
            return corrected ? std::fma(multiply(ad, whole_sum), high, result) : result;
        }
        const float term = multiply(scaled_weight, signed_dot);
        return add(acc, corrected ? add(term, multiply(multiply(ad, high), whole_sum)) : term);
    }

    /** @brief Start with real Q8 publication, then extend its negative endpoint to full int8.
     * @param policy Activation publication policy exercised before injecting -128.
     * @param q8 Retained prequantized rows for the projection leaves.
     * @details Normal FP32 Q8 quantization uses +/-127; -128 is legal in the
     * prequantized ABI. Recompute the raw sum after injecting it rather than
     * asking the production quantizer to change its numerical contract.
     */
    void makeActivations(Policy policy, std::array<Q8_1Block, kRows * kBlocks> &q8)
    {
        std::array<float, kRows * kK> input{};
        const auto negative = [](int row, int b, int i) {
            const int pattern = (row + b) % 3;
            return pattern == 0 || (pattern == 2 && (i + row) % 2 == 0);
        };
        for (int r = 0; r < kRows; ++r)
            for (int b = 0; b < kBlocks; ++b)
                for (int i = 0; i < 32; ++i)
                    input[r * kK + 32 * b + i] = (negative(r, b, i) ? -1.0f : 1.0f) * (1.25f + r + b * 0.375f);
        quantize_activations_to_q8_1(input.data(), q8.data(), kRows, kK, kBlocks, policy);
        for (int r = 0; r < kRows; ++r)
            for (int b = 0; b < kBlocks; ++b)
            {
                auto &a = q8[r * kBlocks + b];
                ASSERT_GT(nativeQuantisedFP16ScaleToFP32(a.d), 0.0f);
                ASSERT_EQ(a.sum_qs, std::accumulate(std::begin(a.qs), std::end(a.qs), 0));
                for (int i = 0; i < 32; ++i)
                {
                    ASSERT_EQ(a.qs[i], negative(r, b, i) ? -127 : 127);
                    if (negative(r, b, i)) a.qs[i] = -128;
                }
                a.sum_qs = std::accumulate(std::begin(a.qs), std::end(a.qs), 0);
                EXPECT_EQ(a.sum_qs, (std::array{-4096, 4064, -16})[(r + b) % 3]);
            }
    }

    /** @return Explicit ISA requests for every projection family in this compiler envelope. */
    constexpr auto compiledPaths()
    {
#if defined(__aarch64__) && defined(__ARM_NEON)
        return std::array{std::pair{ISAPath::NEON, "ASIMD"},
#if defined(__ARM_FEATURE_DOTPROD)
            std::pair{ISAPath::NEON_DOTPROD, "SDOT"},
#endif
#if LLAMINAR_COMPILED_WITH_NEON_I8MM
            std::pair{ISAPath::NEON_I8MM, "I8MM"},
#endif
        };
#else
        return std::array{std::pair{ISAPath::AVX2, "AVX2"},
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
            std::pair{ISAPath::AVX512, "AVX512VNNI"},
#endif
        };
#endif
    }

    /** @test Every live output must match a scalar int64-dot codec oracle; dirty row/N padding must survive. */
    TEST(CPUNativeExtremeByteProjection, AllCataloguedFormatsMatchScalarCodec)
    {
        ASSERT_EQ(formats::kAllSourceFormats.size(), 22u);
        for (const auto policy : {Policy::BackendNative, Policy::GPUAlignedExpert})
        {
            SCOPED_TRACE(static_cast<int>(policy));
            std::array<Q8_1Block, kRows * kBlocks> activation{};
            ASSERT_NO_FATAL_FAILURE(makeActivations(policy, activation));
            const int parts = policy == Policy::BackendNative ? 1 : MoEProjectionNumericalContract::orderedKPartitionsForWidth(kK);
            const int width = (kBlocks + parts - 1) / parts;
            for (const auto &[name, format] : formats::kAllSourceFormats)
            {
                SCOPED_TRACE(name);
                const auto tensor = extremeSource(*format);
                ASSERT_NE(tensor, nullptr);
                const auto *source = IINT8Unpackable::fromTensor(tensor.get());
                ASSERT_NE(source, nullptr);
                ASSERT_EQ(source->vnniFormatInfo(), format);
                CPUNativeQuantisedPackedWeights packed;
                packed.numerical_policy = policy;
                ASSERT_TRUE(packWeightsCPUNativeQuantised(tensor.get(), packed));
                const auto [lo, hi] = endpoints(*format);
                std::array<float, kRows * kStride> expected{};
                for (int n = 0; n < kN; ++n)
                {
                    std::array<ScalarBlock, kBlocks> weights;
                    for (int b = 0; b < kBlocks; ++b)
                    {
                        weights[b] = stageSource(*source, n, b);
                        const bool pos = (n + (format->is_superblock ? b / 8 : b)) % 2 != 0;
                        const auto &q = weights[b].bytes;
                        ASSERT_EQ(pos ? *std::max_element(q.begin(), q.end()) : *std::min_element(q.begin(), q.end()), pos ? hi : lo)
                            << "column=" << n << " block=" << b;
                    }
                    for (int r = 0; r < kRows; ++r)
                    {
                        auto &result = expected[r * kStride + n];
                        for (int p = 0; p < parts; ++p)
                        {
                            float partial = 0.0f;
                            for (int b = p * width; b < std::min(kBlocks, (p + 1) * width); ++b)
                                partial = scalarAppend(*format, weights[b], activation[r * kBlocks + b], partial, policy);
                            result = p == 0 ? partial : exact::add(result, partial);
                        }
                        ASSERT_TRUE(std::isfinite(result));
                    }
                }
                for (const auto &[isa, isa_name] : compiledPaths())
                {
                    SCOPED_TRACE(isa_name);
                    for (int m = 1; m <= kRows; ++m)
                    {
                        SCOPED_TRACE(m);
                        constexpr float dirty = -1234567.0f;
                        std::array<float, kRows * kStride> actual;
                        actual.fill(dirty);
                        std::vector<float> scratch(nativeQuantisedProjectionPartialEnvelopeFloats(packed, m), dirty);
                        if (m == 1)
                            gemv_native_quantised_preq(packed, activation.data(), actual.data(), scratch,
                                isa, DecodeSchedulePolicy::Nbc1);
                        else
                        {
                            // Select shared-row production leaves explicitly: at small N,
                            // the automatic scheduler could instead choose single-row tasks.
                            const bool wide = isa == ISAPath::NEON || isa == ISAPath::NEON_DOTPROD ||
                                isa == ISAPath::NEON_I8MM || (isa == ISAPath::AVX512 && m >= 3);
                            gemm_native_quantised_preq_decode_equivalent_rows(packed,
                                activation.data(), actual.data(), scratch, m, kStride, isa,
                                wide ? VerifierRowsPolicy::WideRows : VerifierRowsPolicy::Pairwise,
                                /*publish_verifier_route=*/false, /*chunks=*/0);
                        }
                        for (int r = 0; r < kRows; ++r)
                            for (int n = 0; n < kStride; ++n)
                            {
                                const int i = r * kStride + n;
                                const float want = r < m && n < kN ? expected[i] : dirty;
                                ASSERT_EQ(std::bit_cast<std::uint32_t>(actual[i]), std::bit_cast<std::uint32_t>(want))
                                    << "row=" << r << " column=" << n << " actual=" << actual[i] << " expected=" << want;
                            }
                    }
                }
            }
        }
    }
} // namespace
} // namespace llaminar2::test
