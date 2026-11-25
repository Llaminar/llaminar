/**
 * @file Test__CPUCodedCompactUnits.cpp
 * @brief Superblock-coded compact CPU units: exact size, exact metadata, exact outputs.
 *
 * Compact multi-scale CPU units (Q5_K, Q3_K, Q2_K, IQ3_S, IQ3_XXS, IQ2_S, IQ2_XS, IQ2_XXS,
 * IQ1_S, IQ1_M) store the exact source codes and payload and one `d`/`dmin` word per (column, source
 * superblock). This suite proves, device-free and on the build's native ISA
 * (NEON, AVX2 or AVX-512):
 *
 * - the SIMD code/word readers equal the scalar plane readers for every
 *   code width and every aligned column vector;
 * - the NEON Q4_K staging (generic reconstruct<5> and, in FP16-arithmetic
 *   envelopes, the fused binary16 reader) equals the codec's staged words bit
 *   for bit, NaN sign and payload included, for every binary16 d/dmin and
 *   every six-bit code, and for mixed per-column d/dmin and 12-bit codes;
 * - the SIMD factor reconstruction equals widening the codec's staged word
 *   (decodeSuperblockMetadata, itself proven against the per-block packer by
 *   V2_Unit_VnniCodedPacking) for every code and a broad d/dmin sweep,
 *   including zero, subnormal, negative, infinite and NaN half words (IEEE
 *   754 leaves the sign and payload of an arithmetic NaN unspecified, and
 *   compilers may fold `-dmin * c` into `dmin * -c`, so a NaN result only
 *   has to be a NaN; every other result is compared bit for bit);
 * - a prepared matrix occupies exactly the GGUF bytes of its 64-padded columns;
 * - the production SIMD projection equals the scalar codec reference for
 *   1..4 rows, ragged N/K tails and both contiguous and ordered K reductions.
 * - compensation-free nibble and Q8_0 layouts preserve the previous signed
 *   integer dot and per-block output bytes, including coded Q4_K ragged K;
 * - every production activation quantizer publishes `sum_qs` equal to the sum
 *   of its 32 published bytes, which the AVX2/AVX-512 zero-point corrections
 *   of the compensation-free nibble dots rely on;
 * - on x86, the decode-once packed prefill panel is selected exactly for
 *   buckets with at least 16 live rows, reproduces independent serial rows
 *   at team widths 1-4 (including backend-native K partitions with carry),
 *   each worker writes only its own slab slot, and a team narrower than the
 *   admitted width never needs more than the admitted borrow.
 */
#include <gtest/gtest.h>

#include "../../../utils/TestTensorFactory.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedCodedMetadata.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedMultiScaleKernels.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedGemv.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedWeightPacker.h"
#if defined(__aarch64__)
#include "kernels/cpu/gemm/CPUNativeNEONPreparedProjection.h"
#endif
#include "kernels/cpu/gemm/CPUWeightPreparationMemory.h"
#include "tensors/NativeQuantisedSuperblockMetadata.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

#include <omp.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        using namespace cpu::native_quantised;
        namespace sb = native_quantised_superblock;
        using Factory = TestTensorFactory;

#if defined(__aarch64__)
        constexpr int kLanes = 4;
#elif defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        constexpr int kLanes = 16;
#else
        constexpr int kLanes = 8;
#endif
        using Ops = multi_scale::SIMD<kLanes>;
        using L = coded_metadata::Lanes<Ops>;
        using I = Ops::I;
        using F = Ops::F;

        /** @return Bits of lane @p lane of @p value. */
        template <typename V> std::uint32_t laneBits(V value, int lane)
        {
            std::uint32_t bits = 0;
            std::memcpy(&bits, reinterpret_cast<const char *>(&value) + 4 * lane, 4);
            return bits;
        }

        /** @return The binary32 bits the per-block unit would have widened @p half to, on this ISA. */
        std::uint32_t widenedBits(std::uint16_t half)
        {
            I words{};
            for (int lane = 0; lane < kLanes; ++lane)
                std::memcpy(reinterpret_cast<char *>(&words) + 4 * lane, &half, 2);
            return laneBits(Ops::halfWords(words), 0);
        }

        /** @return Whether two widened factors agree: identical bits, or both NaN. */
        bool sameFactor(std::uint32_t actual, std::uint32_t expected)
        {
            const auto nan = [](std::uint32_t bits) { return (bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu); };
            return actual == expected || (nan(actual) && nan(expected));
        }

        /** @brief One compact source format and its random-tensor factory. */
        struct CompactSource
        {
            std::string name;
            std::uint8_t codebook = 0;
            double prepared_bytes_per_32 = 0;
            std::function<std::unique_ptr<TensorBase>(const std::vector<size_t> &, uint32_t)> make;
        };

        std::vector<CompactSource> compactSources()
        {
            return {
                {"Q5_K", 7, 22.0, [](auto s, auto r) {
                    const auto tensor = Factory::createQ5_KRandom(s, r);
                    const size_t blocks = s[0] * ((s[1] + 255) / 256);
                    std::vector<uint8_t> bytes(blocks * sizeof(Q5_KBlock));
                    for (size_t i = 0; i < blocks; ++i)
                    {
                        auto block = tensor->typed_data()[i];
                        block.dmin = static_cast<uint16_t>(0x1800 + i % 0x1300);
                        std::memcpy(bytes.data() + i * sizeof(block), &block, sizeof(block));
                    }
                    return std::make_unique<Q5_KTensor>(s, bytes);
                }},
                {"Q3_K", 9, 13.75, [](auto s, auto r) { return Factory::createQ3_KRandom(s, r); }},
                {"Q2_K", 10, 10.5, [](auto s, auto r) { return Factory::createQ2_KRandom(s, r); }},
                {"IQ3_S", 11, 13.75, [](auto s, auto r) { return Factory::createIQ3_SRandom(s, r); }},
                {"IQ3_XXS", 12, 12.25, [](auto s, auto r) { return Factory::createIQ3_XXSRandom(s, r); }},
                {"IQ2_S", 13, 10.25, [](auto s, auto r) { return Factory::createIQ2_SRandom(s, r); }},
                {"IQ2_XS", 14, 9.25, [](auto s, auto r) { return Factory::createIQ2_XSRandom(s, r); }},
                {"IQ2_XXS", 15, 8.25, [](auto s, auto r) { return Factory::createIQ2_XXSRandom(s, r); }},
                {"IQ1_S", 16, 6.25, [](auto s, auto r) { return Factory::createIQ1_SRandom(s, r); }},
                {"IQ1_M", 17, 7.0, [](auto s, auto r) { return Factory::createIQ1_MRandom(s, r); }},
            };
        }

        /** @brief One block's original per-block publication and explicit FP32 accumulation. */
        float expandedReferenceBlock(const IINT8Unpackable &source,
            const Q8_1Block &activation, int k, int column, int kb,
            CPUProjectionNumericalPolicy policy, float acc)
        {
            const auto &format = *source.vnniFormatInfo();
            uint8_t payload[20];
            uint16_t scale = 0, minimum = 0;
            source.packVnniBlock({.N = 1, .K = k, .blocks_per_row = (k + 31) / 32,
                .payload_bytes = format.payload_bytes, .payload_array = payload,
                .scales_array = &scale, .mins_array = &minimum,
                .destination_block_origin = kb, .destination_block_count = 1}, column, 0, kb);
            int8_t values[32];
            decode_native_block(format.codebook_id, payload, values);
            int dot = 0;
            for (int i = 0; i < 32; ++i)
                dot += int(values[i]) * activation.qs[i];
            using device_fp32_contract::multiply;
            using device_fp32_contract::add;
            const bool iq = format.codebook_id == 11;
            const float weight = iq ? multiply(fp16_to_fp32(scale), float(2 * minimum + 1)) : fp16_to_fp32(scale);
            const float a = nativeQuantisedFP16ScaleToFP32(activation.d);
            const float combined = multiply(a, weight);
            if (policy == CPUProjectionNumericalPolicy::BackendNative)
            {
                const float value = std::fma(float(dot), combined, acc);
                return iq ? value : std::fma(multiply(a, float(activation.sum_qs)), fp16_to_fp32(minimum), value);
            }
            const float value = multiply(combined, float(dot));
            return add(acc, iq ? value : add(value, multiply(multiply(a, fp16_to_fp32(minimum)), float(activation.sum_qs))));
        }

        /** @return Interesting binary16 words: zeros, subnormals, normals of both signs, extremes, inf, NaN. */
        std::vector<std::uint16_t> halfSweep()
        {
            std::vector<std::uint16_t> words = {0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF, 0x0400, 0x8400,
                                                0x3C00, 0xBC00, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0x7C01,
                                                0xFE00, 0x1400, 0x2E66, 0xA9C3};
            std::mt19937 rng(17);
            for (int i = 0; i < 300; ++i)
                words.push_back(static_cast<std::uint16_t>(rng()));
            return words;
        }

        template <std::uint8_t Codebook> void expectReconstructionMatchesCodec()
        {
            SCOPED_TRACE("codebook=" + std::to_string(Codebook));
            const std::uint32_t codes = 1u << sb::superblockCodeBits(Codebook);
            const auto halves = halfSweep();
            std::mt19937 rng(Codebook);
            for (std::uint16_t d : halves)
                for (std::uint32_t first = 0; first < codes; first += kLanes)
                {
                    // Q2_K's 16-bit codes x every d would be slow; sample its dmin.
                    const std::uint16_t dmin = sb::superblockHasDmin(Codebook) ? halves[rng() % halves.size()] : 0;
                    I code{}, word{};
                    for (int lane = 0; lane < kLanes; ++lane)
                    {
                        const std::int32_t c = static_cast<std::int32_t>((first + lane) % codes);
                        const std::int32_t w = static_cast<std::int32_t>(sb::packSuperblockWord(d, dmin));
                        std::memcpy(reinterpret_cast<char *>(&code) + 4 * lane, &c, 4);
                        std::memcpy(reinterpret_cast<char *>(&word) + 4 * lane, &w, 4);
                    }
                    const auto factors = coded_metadata::reconstruct<Codebook, Ops>(code, word);
                    for (int lane = 0; lane < kLanes; ++lane)
                    {
                        const auto staged =
                            sb::decodeSuperblockMetadata<Codebook>((first + lane) % codes, sb::packSuperblockWord(d, dmin));
                        ASSERT_PRED2(sameFactor, laneBits(factors.primary, lane), widenedBits(staged.primary))
                            << "code=" << (first + lane) % codes << " d=" << d;
                        if constexpr (Codebook != 12 && Codebook != 15)
                            ASSERT_PRED2(sameFactor, laneBits(factors.secondary, lane), widenedBits(staged.secondary))
                                << "code=" << (first + lane) % codes << " d=" << d;
                        if constexpr (Codebook == 10)
                        {
                            ASSERT_PRED2(sameFactor, laneBits(factors.emin_low, lane),
                                         widenedBits(static_cast<std::uint16_t>(staged.emin)));
                            ASSERT_PRED2(sameFactor, laneBits(factors.emin_high, lane),
                                         widenedBits(static_cast<std::uint16_t>(staged.emin >> 16)));
                        }
                    }
                    if (Codebook == 10 && first > 4096)
                        break;
                }
        }
    } // namespace

    /** SIMD plane readers equal the scalar readers for every code width and column vector. */
    TEST(CPUCodedCompactUnits, SimdCodeReadersEqualScalarPlaneReaders)
    {
        std::mt19937 rng(5);
        for (std::uint8_t codebook : {5, 7, 9, 10, 11, 13, 17})
        {
            std::vector<std::uint8_t> planes(coded_metadata::codeBlockBytes(codebook) + 16, 0);
            std::vector<std::uint32_t> expected(64);
            for (int column = 0; column < 64; ++column)
            {
                expected[column] = rng() & ((1u << sb::superblockCodeBits(codebook)) - 1u);
                coded_metadata::storeCode(planes.data(), codebook, column, expected[column]);
            }
            for (int column = 0; column < 64; ++column)
                ASSERT_EQ(coded_metadata::loadCode(planes.data(), codebook, column), expected[column]);
            for (int column = 0; column < 64; column += kLanes)
            {
                I code{};
                switch (codebook)
                {
                case 7: code = L::code<7>(planes.data(), column); break;
                case 5: code = L::code<5>(planes.data(), column); break;
                case 9: code = L::code<9>(planes.data(), column); break;
                case 10: code = L::code<10>(planes.data(), column); break;
                case 11: code = L::code<11>(planes.data(), column); break;
                case 13: code = L::code<13>(planes.data(), column); break;
                case 17: code = L::code<17>(planes.data(), column); break;
                }
                for (int lane = 0; lane < kLanes; ++lane)
                    ASSERT_EQ(laneBits(code, lane), expected[column + lane])
                        << "codebook=" << int(codebook) << " column=" << column + lane;
            }
        }
    }

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    /** @test The AVX-512 Q5_K mask addresses every weight byte, including column and group boundaries. */
    TEST(CPUCodedCompactUnits, AVX512Q5KFifthBitMaskMatchesScalar)
    {
        using Wide = multi_scale::SIMD<16>;
        constexpr int columns = 64, groups = 8;
        for (int first = 0; first < columns; first += 16)
            for (int group = 0; group < groups; ++group)
                for (int bit = 0; bit < 64; ++bit)
                {
                    std::array<std::uint8_t, groups * columns / 2> planes{};
                    // Exercise both halves of each source high-bit byte through
                    // the real publisher, with exactly one bit set per probe.
                    const int column = first + bit / 4;
                    uint8_t qh[4]{};
                    qh[group / 2] = 1u << (4 * (group % 2) + bit % 4);
                    coded_metadata::storeQ5HighBits(planes.data(), column, qh);
                    const auto *mask = planes.data() + group * columns / 2 + first / 2;
                    const auto result = Wide::insertQ5High(Wide::I{} + 0x0F070300, mask);
                    for (int lane = 0; lane < 16; ++lane)
                    {
                        const unsigned expected = 0x0F070300u |
                            (lane == bit / 4 ? 16u << (8 * (bit % 4)) : 0u);
                        ASSERT_EQ(laneBits(result, lane), expected) << "group=" << group << " bit=" << bit;
                    }
                }
    }
#endif

    /** SIMD reconstruction equals widening the codec's staged words for every derived-scale compact format. */
    TEST(CPUCodedCompactUnits, SimdReconstructionEqualsCodecWordsExhaustively)
    {
        expectReconstructionMatchesCodec<5>();
        expectReconstructionMatchesCodec<7>();
        expectReconstructionMatchesCodec<9>();
        expectReconstructionMatchesCodec<10>();
        expectReconstructionMatchesCodec<12>();
        expectReconstructionMatchesCodec<14>();
        expectReconstructionMatchesCodec<15>();
        expectReconstructionMatchesCodec<16>();
        expectReconstructionMatchesCodec<17>();
        for (const auto d : halfSweep())
            for (unsigned code = 0; code < 16; ++code)
            {
                const auto input = sb::decodeSuperblockMetadata<11>(code, d);
                const auto factor = multi_scale::iq3sSourceScale<Ops>(I{} + int(code), I{} + int(d));
                const auto expected = device_fp32_contract::multiply(fp16_to_fp32(input.primary), float(2 * input.secondary + 1));
                ASSERT_PRED2(sameFactor, laneBits(factor, 0), std::bit_cast<uint32_t>(expected));
            }
    }

#if defined(__aarch64__) && defined(__ARM_NEON)
    namespace
    {
        constexpr std::size_t kQ4KSweepLanes = std::size_t(65536) * 64 * 2;

        /** @brief Every binary16 word in 64-column headers, each column with every six-bit code. */
        struct Q4KStagingSweep
        {
            std::vector<std::uint8_t> headers; ///< 1024 headers: column c of header h holds d = dmin = 64h + c.
            std::vector<std::uint8_t> planes;  ///< 64 code blocks: block c holds code c | c << 6 in every column.
        };

        Q4KStagingSweep q4KStagingSweep()
        {
            Q4KStagingSweep sweep;
            const std::size_t header_bytes = coded_metadata::headerBytes(5), plane_bytes = coded_metadata::codeBlockBytes(5);
            sweep.headers.assign(1024 * header_bytes, 0);
            sweep.planes.assign(64 * plane_bytes, 0);
            for (std::uint32_t h = 0; h < 65536; ++h)
                coded_metadata::storeSuperblockWord(sweep.headers.data() + (h / 64) * header_bytes, 5, int(h % 64),
                    sb::packSuperblockWord(static_cast<std::uint16_t>(h), static_cast<std::uint16_t>(h)));
            for (std::uint32_t c = 0; c < 64; ++c)
                for (int column = 0; column < 64; ++column)
                    coded_metadata::storeCode(sweep.planes.data() + c * plane_bytes, 5, column, c | (c << 6));
            return sweep;
        }

        /**
         * @return The codec's staged words, widened exactly (raw bits), for every
         *         (binary16, code) pair: four consecutive halves per eight lanes,
         *         scale lanes then minimum lanes, codes innermost.
         */
        std::vector<std::uint32_t> q4KCodecBits()
        {
            std::vector<std::uint32_t> bits(kQ4KSweepLanes);
            std::size_t out = 0;
            for (std::uint32_t h = 0; h < 65536; h += 4)
                for (std::uint32_t c = 0; c < 64; ++c, out += 8)
                    for (int lane = 0; lane < 4; ++lane)
                    {
                        const auto half = static_cast<std::uint16_t>(h + lane);
                        const auto staged = sb::decodeSuperblockMetadata<5>(c | (c << 6), sb::packSuperblockWord(half, half));
                        bits[out + lane] = widenedBits(staged.primary);
                        bits[out + 4 + lane] = widenedBits(staged.secondary);
                    }
            return bits;
        }

        /** @brief Assert raw bit equality and name the first differing lane. */
        void expectSameBits(const std::vector<std::uint32_t> &actual, const std::vector<std::uint32_t> &expected,
            const char *reader)
        {
            ASSERT_EQ(actual.size(), expected.size());
            for (std::size_t i = 0; i < expected.size(); ++i)
                ASSERT_EQ(actual[i], expected[i]) << reader << ": half 0x" << std::hex << (i / 512) * 4 + i % 4
                    << " code " << std::dec << (i / 8) % 64 << (i % 8 < 4 ? " scale" : " minimum");
        }
    } // namespace

    /**
     * The generic NEON reconstruct<5> (Q5_K, and Q4_K outside FP16-arithmetic
     * envelopes) equals widening the codec's staged words bit for bit, NaN sign
     * and payload included, for every binary16 d/dmin and every six-bit code.
     */
    TEST(CPUCodedCompactUnits, NeonQ4KReconstructionEqualsCodecBitsExhaustively)
    {
        std::vector<std::uint32_t> actual(kQ4KSweepLanes);
        std::size_t out = 0;
        for (std::uint32_t h = 0; h < 65536; h += 4)
            for (std::uint32_t c = 0; c < 64; ++c, out += 8)
            {
                I word{}, code{};
                for (int lane = 0; lane < 4; ++lane)
                {
                    const auto half = static_cast<std::uint16_t>(h + lane);
                    word[lane] = static_cast<int>(sb::packSuperblockWord(half, half));
                    code[lane] = static_cast<int>(c | (c << 6));
                }
                const auto factors = coded_metadata::reconstruct<5, Ops>(code, word);
                std::memcpy(actual.data() + out, &factors.primary, sizeof(factors.primary));
                std::memcpy(actual.data() + out + 4, &factors.secondary, sizeof(factors.secondary));
            }
        expectSameBits(actual, q4KCodecBits(), "reconstruct<5>");
    }

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    /**
     * The fused binary16 Q4_K reader (one half multiply per block for scale and
     * minimum, integer sign flip and -0 select) equals widening the codec's
     * staged words bit for bit, NaN sign and payload included, for every
     * binary16 d/dmin, every six-bit code and every column lane of real code
     * planes and headers. Runs under the admitted FPCR profile; the projection
     * entries reject any other (CPUNativeNEONPreparedProjection tests).
     */
    TEST(CPUCodedCompactUnits, NeonHalfQ4KReconstructionEqualsCodecBitsExhaustively)
    {
        const auto sweep = q4KStagingSweep();
        const std::size_t header_bytes = coded_metadata::headerBytes(5), plane_bytes = coded_metadata::codeBlockBytes(5);
        std::vector<std::uint32_t> actual(kQ4KSweepLanes);
        for (std::size_t header = 0; header < 1024; ++header)
            for (int column = 0; column < 64; column += 4)
            {
                const auto base = coded_metadata::neon_half::q4KHalfBase(sweep.headers.data() + header * header_bytes, column);
                // Lane order of q4KCodecBits(): first half = 64 * header + column.
                std::size_t out = (header * 16 + std::size_t(column) / 4) * 64 * 8;
                for (std::size_t c = 0; c < 64; ++c, out += 8)
                {
                    float32x4_t scale, minimum;
                    coded_metadata::neon_half::q4KFactors(sweep.planes.data() + c * plane_bytes, column, base, scale, minimum);
                    vst1q_u32(actual.data() + out, vreinterpretq_u32_f32(scale));
                    vst1q_u32(actual.data() + out + 4, vreinterpretq_u32_f32(minimum));
                }
            }
        expectSameBits(actual, q4KCodecBits(), "neon_half::q4KFactors");
    }
#endif

    /**
     * Mixed operands, complementing the diagonal sweeps above (which use d ==
     * dmin and scale code == minimum code, so they cannot see a swapped or
     * duplicated input): every column of a real header carries its own d and
     * an unrelated dmin, and every column of a real code plane its own 12-bit
     * code whose scale and minimum fields differ, covering all 4096 codes.
     * Both the fused reader (FP16-arithmetic envelopes) and the generic
     * reconstruct<5> must equal the codec's widened staged words in raw bits,
     * NaN sign and payload included, lane by lane.
     */
    TEST(CPUCodedCompactUnits, NeonQ4KMixedHeaderAndCodeGatherEqualsCodecBits)
    {
        const auto halves = halfSweep();
        std::vector<std::uint8_t> planes(coded_metadata::codeBlockBytes(5), 0);
        std::vector<std::uint8_t> header(coded_metadata::headerBytes(5), 0);
        for (std::size_t h = 0; h < halves.size(); ++h)
        {
            for (int column = 0; column < 64; ++column)
                coded_metadata::storeSuperblockWord(header.data(), 5, column,
                    sb::packSuperblockWord(halves[(h + column * 5) % halves.size()],
                                           halves[(h * 7 + column * 13 + 1) % halves.size()]));
            for (std::uint32_t first = 0; first < 4096; first += 64)
            {
                std::fill(planes.begin(), planes.end(), 0);
                for (int column = 0; column < 64; ++column)
                    coded_metadata::storeCode(planes.data(), 5, column, (first + column * 37) % 4096);
                for (int column = 0; column < 64; column += 4)
                {
                    const auto generic = coded_metadata::reconstruct<5, Ops>(
                        L::code<5>(planes.data(), column), L::superblockWords<5>(header.data(), column));
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
                    float32x4_t scale, minimum;
                    coded_metadata::neon_half::q4KFactors(planes.data(), column,
                        coded_metadata::neon_half::q4KHalfBase(header.data(), column), scale, minimum);
#endif
                    for (int lane = 0; lane < 4; ++lane)
                    {
                        const std::uint32_t code = (first + (column + lane) * 37) % 4096;
                        const auto word = coded_metadata::loadSuperblockWord(header.data(), 5, column + lane);
                        const auto staged = sb::decodeSuperblockMetadata<5>(code, word);
                        const auto expected_scale = widenedBits(staged.primary);
                        const auto expected_minimum = widenedBits(staged.secondary);
                        ASSERT_EQ(laneBits(generic.primary, lane), expected_scale)
                            << "reconstruct<5> code=" << code << " word=0x" << std::hex << word;
                        ASSERT_EQ(laneBits(generic.secondary, lane), expected_minimum)
                            << "reconstruct<5> code=" << code << " word=0x" << std::hex << word;
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
                        ASSERT_EQ(laneBits(scale, lane), expected_scale)
                            << "neon_half code=" << code << " word=0x" << std::hex << word;
                        ASSERT_EQ(laneBits(minimum, lane), expected_minimum)
                            << "neon_half code=" << code << " word=0x" << std::hex << word;
#endif
                    }
                }
            }
        }
    }
#endif

    /** Compact matrices retain the exact file density of each original format. */
    TEST(CPUCodedCompactUnits, PreparedBytesEqualFileDensity)
    {
        for (const auto &source : compactSources())
        {
            SCOPED_TRACE(source.name);
            const auto tensor = source.make({128, 1024}, 3u);
            CPUNativeQuantisedPackedWeights packed;
            ASSERT_TRUE(packWeightsCPUNativeQuantised(tensor.get(), packed));
            ASSERT_TRUE(packed.usesCompactMultiScale());
            const double bytes = source.prepared_bytes_per_32 * 128.0 * (1024 / 32);
            EXPECT_EQ(static_cast<double>(packed.native_interleaved.size()), bytes);
            EXPECT_EQ(packed.preparedBytes(), packed.native_interleaved.size());
            EXPECT_EQ(CPUWeightPreparationMemory::sourceNative(source.name, 128, 1024).persistent_payload_bytes,
                packed.native_interleaved.size());
            EXPECT_EQ(static_cast<double>(packed.preparedFootprint().weight_bytes_per_n_chunk_k_block),
                      source.prepared_bytes_per_32 * 64.0);
            if (source.codebook == 7 || source.codebook == 11)
                EXPECT_EQ(preparedFootprintForFormat(*native_quantised_formats::forSourceIdentity(source.codebook, true))
                              .weight_bytes_per_n_chunk_k_block, source.prepared_bytes_per_32 * 64.0);
        }
        EXPECT_EQ(preparedEncodingForSource(native_quantised_formats::Q5_1), CPUNativeQuantisedEncoding::Q5NativeSingleScale);
    }

    /** The production SIMD projection equals the scalar codec reference for every compact format. */
    TEST(CPUCodedCompactUnits, SimdProjectionEqualsScalarCodecReference)
    {
        constexpr int kN = 67, kK = 544, kBlocks = kK / 32;
        constexpr int kPartitions = MoEProjectionNumericalContract::orderedKPartitionsForWidth(kK);
        constexpr int kBlocksPerPartition = (kBlocks + kPartitions - 1) / kPartitions;
        static_assert(kBlocks % sb::kBlocksPerSuperblock != 0);
        static_assert(kBlocksPerPartition > 1 && kBlocks % kBlocksPerPartition != 0);
        std::mt19937 rng(11);
        std::array<std::vector<Q8_1Block>, 4> activation;
        for (auto &row : activation)
        {
            row.resize(kBlocks);
            for (auto &block : row)
            {
                block.d = static_cast<std::uint16_t>(0x2000 + rng() % 0x1800);
                int sum = 0;
                for (int i = 0; i < 32; ++i)
                {
                    block.qs[i] = static_cast<std::int8_t>(static_cast<int>(rng() % 255) - 127);
                    sum += block.qs[i];
                }
                block.sum_qs = static_cast<decltype(block.sum_qs)>(sum);
            }
        }
        for (const auto &source : compactSources())
        {
            SCOPED_TRACE(source.name);
            const auto tensor = source.make({static_cast<size_t>(kN), static_cast<size_t>(kK)}, 23u + source.codebook);
            CPUNativeQuantisedPackedWeights packed;
            ASSERT_TRUE(packWeightsCPUNativeQuantised(tensor.get(), packed));
            packed.numerical_policy = CPUProjectionNumericalPolicy::GPUAlignedExpert;
            const auto view = packed.readView();
            for (int chunk = 0; chunk < packed.N_padded / 64; ++chunk)
            {
                const int columns = std::min(64, kN - chunk * 64);
                std::array<std::array<float, 64>, 4> actual{}, expected{}, expected_ordered{};
                for (int row = 0; row < 4; ++row)
                    for (int column = 0; column < columns; ++column)
                    {
                        float acc = 0.0f, reduced = 0.0f;
                        for (int partition = 0; partition < kPartitions; ++partition)
                        {
                            float partial = 0.0f;
                            for (int kb = partition * kBlocksPerPartition;
                                 kb < std::min(kBlocks, (partition + 1) * kBlocksPerPartition); ++kb)
                            {
                                const float contribution = multi_scale::scalarReferenceBlock(
                                    view, activation[row][kb], chunk, kb, column);
                                acc += contribution;
                                partial += contribution;
                            }
                            reduced = partition == 0 ? partial : reduced + partial;
                        }
                        expected[row][column] = acc;
                        expected_ordered[row][column] = reduced;
                    }
                const auto run = [&]<std::size_t Rows>() {
                    std::array<const Q8_1Block *, Rows> input;
                    std::array<float *, Rows> output;
                    for (std::size_t row = 0; row < Rows; ++row)
                    {
                        input[row] = activation[row].data();
                        output[row] = actual[row].data();
                        actual[row].fill(-1.0f);
                    }
                    const auto compare = [&](const auto &reference) {
                        for (std::size_t row = 0; row < Rows; ++row)
                            for (int column = 0; column < columns; ++column)
                                ASSERT_EQ(std::bit_cast<std::uint32_t>(actual[row][column]),
                                          std::bit_cast<std::uint32_t>(reference[row][column]))
                                    << "rows=" << Rows << " row=" << row << " column=" << chunk * 64 + column;
                    };
                    const int live = kLanes == 4 ? columns : 64;
                    multi_scale::dispatch<kLanes, Rows, Ops>(view, input, output, chunk, 0, kBlocks, false, live);
                    compare(expected);
                    multi_scale::dispatchOrdered<kLanes, Rows, Ops>(view, input, output, chunk, kBlocks,
                                                                   kPartitions, kBlocksPerPartition, live);
                    compare(expected_ordered);
                };
                run.template operator()<1>();
                run.template operator()<2>();
                run.template operator()<3>();
                run.template operator()<4>();
            }
        }
    }

    /** @brief Check both CPU policies against independent expanded weights across source and K-tree boundaries.
     * @param k Logical K, with a ragged final source superblock.
     */
    static void verifyQ5KIQ3SExpandedOutputs(int k)
    {
        constexpr int n = 67;
        const int blocks = k / 32;
        const int partitions = MoEProjectionNumericalContract::orderedKPartitionsForWidth(k);
        SCOPED_TRACE(k);
        std::mt19937 rng(1493);
        std::array<std::vector<Q8_1Block>, 4> activation;
        for (auto &row : activation)
        {
            row.resize(blocks);
            for (auto &a : row)
            {
                a.d = static_cast<uint16_t>(0x2000 + rng() % 0x1800);
                for (auto &q : a.qs)
                {
                    q = static_cast<int8_t>(int(rng() % 255) - 127);
                    a.sum_qs += q;
                }
            }
        }
        for (const auto &format : compactSources())
        {
            if (format.codebook != 7 && format.codebook != 11)
                continue;
            SCOPED_TRACE(format.name);
            const auto source = format.make({n, static_cast<std::size_t>(k)}, 37);
            const auto &unpackable = dynamic_cast<const IINT8Unpackable &>(*source);
            for (const auto policy : {CPUProjectionNumericalPolicy::BackendNative, CPUProjectionNumericalPolicy::GPUAlignedExpert})
            {
                SCOPED_TRACE(static_cast<int>(policy));
                CPUNativeQuantisedPackedWeights packed;
                packed.numerical_policy = policy;
                ASSERT_TRUE(packWeightsCPUNativeQuantised(source.get(), packed));
                const auto view = packed.readView();
                const bool ordered = policy == CPUProjectionNumericalPolicy::GPUAlignedExpert;
                const int parts = ordered ? partitions : 1;
                const int blocks_per_part = (blocks + parts - 1) / parts;
                for (int chunk = 0; chunk < 2; ++chunk)
                {
                    const int columns = std::min(64, n - chunk * 64);
                    const auto run = [&]<size_t Rows>() {
                        std::array<std::array<float, 64>, Rows> actual{};
#if defined(__aarch64__)
                        std::array<std::span<const Q8_1Block>, Rows> input;
                        std::array<std::span<float>, Rows> output;
                        for (size_t r = 0; r < Rows; ++r)
                        {
                            input[r] = activation[r];
                            output[r] = std::span(actual[r]).first(columns);
                        }
                        const auto tile = CPUNativeNEONProjectionTile<Rows>::bind(view, input, output, chunk);
#if defined(__ARM_FEATURE_MATMUL_INT8)
                        tile.template execute<multi_scale::NEONByteDot::I8MM>();
#elif defined(__ARM_FEATURE_DOTPROD)
                        tile.template execute<multi_scale::NEONByteDot::DotProd>();
#else
                        tile.template execute<>();
#endif
#else
                        std::array<const Q8_1Block *, Rows> input;
                        std::array<float *, Rows> output;
                        for (size_t r = 0; r < Rows; ++r)
                        {
                            input[r] = activation[r].data();
                            output[r] = actual[r].data();
                        }
                        if (ordered)
                            multi_scale::dispatchOrdered<kLanes, Rows, Ops>(view, input, output, chunk,
                                blocks, parts, blocks_per_part);
                        else
                            multi_scale::dispatch<kLanes, Rows, Ops>(view, input, output, chunk, 0, blocks, false);
#endif
                        for (size_t row = 0; row < Rows; ++row)
                            for (int column = 0; column < columns; ++column)
                            {
                                float expected = 0.0f;
                                for (int p = 0; p < parts; ++p)
                                {
                                    float partial = 0.0f;
                                    for (int kb = p * blocks_per_part; kb < std::min(blocks, (p + 1) * blocks_per_part); ++kb)
                                        partial = expandedReferenceBlock(unpackable, activation[row][kb], k, chunk * 64 + column,
                                            kb, policy, partial);
                                    expected = p == 0 ? partial : expected + partial;
                                }
                                ASSERT_EQ(std::bit_cast<uint32_t>(actual[row][column]), std::bit_cast<uint32_t>(expected))
                                    << "rows=" << Rows << " row=" << row << " column=" << chunk * 64 + column;
                            }
                    };
                    run.template operator()<1>();
                    run.template operator()<2>();
                    run.template operator()<3>();
                    run.template operator()<4>();
                }
            }
        }
    }

    /** @test Hoisted header factors survive a new K partition inside a superblock and a new superblock inside a partition. */
    TEST(CPUCodedCompactUnits, Q5KIQ3SPreserveExpandedOutputs)
    {
        verifyQ5KIQ3SExpandedOutputs(544);
        // 98 blocks gives a seven-block partition and a two-block final
        // superblock. This exercises both kinds of interior boundary.
        verifyQ5KIQ3SExpandedOutputs(3136);
    }

    namespace
    {
        /** @brief Build source Q4_K with nonzero minima and all 12-bit scale/minimum fields varied. */
        std::unique_ptr<TensorBase> q4KSource(const std::vector<std::size_t> &shape, std::uint32_t seed)
        {
            const auto superblocks = shape[0] * ((shape[1] + 255) / 256);
            std::vector<std::uint8_t> bytes(superblocks * sizeof(Q4_KBlock));
            std::mt19937 rng(seed);
            for (std::size_t index = 0; index < superblocks; ++index)
            {
                Q4_KBlock block{};
                block.d = static_cast<std::uint16_t>(0x1400 + rng() % 1024);
                block.dmin = static_cast<std::uint16_t>(0x1000 + rng() % 1024);
                for (auto &value : block.scales) value = static_cast<std::uint8_t>(rng());
                for (auto &value : block.qs) value = static_cast<std::uint8_t>(rng());
                std::memcpy(bytes.data() + index * sizeof(block), &block, sizeof(block));
            }
            return std::make_unique<Q4_KTensor>(shape, bytes);
        }

        /** @return Test factories for every compensation-free source, including IQ4_XS's independent metadata. */
        std::vector<CompactSource> compensationFreeSources()
        {
            return {
                {"Q4_0", 0, 18, [](auto s, auto r) { return Factory::createQ4_0Random(s, r); }},
                {"Q4_1", 5, 20, [](auto s, auto r) { return Factory::createQ4_1Random(s, r); }},
                {"Q4_K", 5, 18, q4KSource},
                {"IQ4_NL", 4, 18, [](auto s, auto r) { return Factory::createIQ4_NLRandom(s, r); }},
                {"IQ4_XS", 4, 17, [](auto s, auto r) { return Factory::createIQ4_XSRandom(s, r); }},
                {"MXFP4", 25, 17, [](auto s, auto r) { return Factory::createMXFP4Random(s, r); }},
                {"Q8_0", 19, 34, [](auto s, auto r) { return Factory::createQ8_0Random(s, r); }},
                {"Q8_1", 20, 34, [](auto s, auto r) { return Factory::createQ8_1Random(s, -1.0f, 1.0f, r); }},
                {"Q8_K", 21, 34, [](auto s, auto r) { return Factory::createQ8_KRandom(s, r); }},
            };
        }

        /** @brief Replay the previous per-block source factors and signed-dot program, independent of resident bytes. */
        float previousBlock(const IINT8Unpackable &source, const CPUNativeQuantisedPackedWeights &packed,
            const Q8_1Block &a, int column, int block, float acc)
        {
            std::uint8_t payload[32]{};
            std::uint16_t primary{}, secondary{};
            source.packVnniBlock(VnniPackContext{.N=1, .K=packed.K,
                .blocks_per_row=packed.blocks_per_row, .payload_bytes=packed.payload_bytes,
                .payload_array=payload, .scales_array=&primary, .mins_array=&secondary,
                .destination_block_origin=block, .destination_block_count=1,
                .primary_layout=packed.codebook_id == 25 ? VnniPrimaryScaleLayout::PowerOfTwoBF16Words
                                                        : VnniPrimaryScaleLayout::FP16Words}, column, 0, block);
            std::int8_t weight[32];
            if (packed.usesSignedInt8SingleScale())
                std::memcpy(weight, payload, sizeof(weight));
            else
                decode_native_block(packed.codebook_id, payload, weight);
            // Old x86 formula, even for formats whose optimized leaf already used unsigned B.
            int biased_dot = 0, sum_weights = 0;
            for (int k = 0; k < 32; ++k)
            {
                biased_dot += (128 + int(a.qs[k])) * weight[k];
                sum_weights += weight[k];
            }
            const float dot = float(biased_dot - 128 * sum_weights);
            const float as = nativeQuantisedFP16ScaleToFP32(a.d);
            const float ws = packed.codebook_id == 25 ? std::bit_cast<float>(std::uint32_t{primary} << 16)
                : packed.usesNibbleLUTSourceScale() ? device_fp32_contract::multiply(fp16_to_fp32(primary), float(int(secondary) - 32))
                : fp16_to_fp32(primary);
            using device_fp32_contract::multiply;
            using device_fp32_contract::add;
            const float scaled = multiply(as, ws);
            const float minimum = fp16_to_fp32(secondary);
            if (packed.numerical_policy == CPUProjectionNumericalPolicy::BackendNative)
            {
                const float result = std::fma(scaled, dot, acc);
                return packed.is_asymmetric ? std::fma(multiply(as, float(a.sum_qs)), minimum, result) : result;
            }
            const float main = multiply(scaled, dot);
            return add(acc, packed.is_asymmetric ? add(main, multiply(multiply(as, minimum), float(a.sum_qs))) : main);
        }
    } // namespace

    /** The resident layout pays only for payload, source/staged factors and live superblock headers. */
#if defined(__AVX2__)
    namespace
    {
        /** @brief Bind one exact OpenMP team width and restore the caller's policy on exit. */
        class ScopedPanelTeam final
        {
        public:
            /** @param workers Positive team width selected for this replay. */
            explicit ScopedPanelTeam(int workers)
                : previous_threads_(omp_get_max_threads()), previous_dynamic_(omp_get_dynamic())
            {
                omp_set_dynamic(0);
                omp_set_num_threads(workers);
            }
            ~ScopedPanelTeam()
            {
                omp_set_num_threads(previous_threads_);
                omp_set_dynamic(previous_dynamic_);
            }
            ScopedPanelTeam(const ScopedPanelTeam &) = delete;
            ScopedPanelTeam &operator=(const ScopedPanelTeam &) = delete;

        private:
            int previous_threads_;
            int previous_dynamic_;
        };

        /** @brief Collect only the production prefill launch family; restore the collection policy on exit. */
        class ScopedPrefillLaunchObservation final
        {
        public:
            ScopedPrefillLaunchObservation()
                : json_(prior("LLAMINAR_PERF_STATS_JSON")), filter_(prior("LLAMINAR_PERF_STATS_FILTER"))
            {
                setenv("LLAMINAR_PERF_STATS_JSON", "1", 1);
                setenv("LLAMINAR_PERF_STATS_FILTER", "kernel.cpu_native_quantised_prefill_gemm_launch", 1);
                PerfStatsCollector::reloadConfigurationFromEnvironment();
                PerfStatsCollector::reset();
            }
            ~ScopedPrefillLaunchObservation()
            {
                PerfStatsCollector::reset();
                restore("LLAMINAR_PERF_STATS_JSON", json_);
                restore("LLAMINAR_PERF_STATS_FILTER", filter_);
                PerfStatsCollector::reloadConfigurationFromEnvironment();
            }
            ScopedPrefillLaunchObservation(const ScopedPrefillLaunchObservation &) = delete;
            ScopedPrefillLaunchObservation &operator=(const ScopedPrefillLaunchObservation &) = delete;

        private:
            /** @return Original optional environment value, preserving absence. */
            static std::optional<std::string> prior(const char *name)
            {
                const char *value = std::getenv(name);
                return value ? std::optional<std::string>{value} : std::nullopt;
            }
            /** @brief Restore one original environment value exactly. */
            static void restore(const char *name, const std::optional<std::string> &value)
            {
                if (value)
                    setenv(name, value->c_str(), 1);
                else
                    unsetenv(name);
            }
            std::optional<std::string> json_, filter_;
        };

        /** @brief Route evidence published by one production prefill launch. */
        struct PrefillLaunch
        {
            std::uint64_t count = 0; ///< Matching launches since the last reset.
            std::string route;       ///< Physical route name.
            std::string threads;     ///< Worker count the route was admitted for.
            std::string k_tiles;     ///< Serial K partitions folded by the route.
        };

        /** @return The prefill launch recorded since the last collector reset. */
        PrefillLaunch observedPrefillLaunch()
        {
            PrefillLaunch launch;
            for (const auto &record : PerfStatsCollector::snapshot({"kernel.cpu_native_quantised_prefill_gemm_launch"}))
            {
                if (record.domain != "kernel" || record.name != "cpu_native_quantised_prefill_gemm_launch" ||
                    record.kind != PerfStatRecord::Kind::Counter)
                    continue;
                launch.count += record.count;
                launch.route = record.tags.at("route");
                launch.threads = record.tags.at("threads");
                launch.k_tiles = record.tags.at("k_tiles");
            }
            return launch;
        }

        /** @brief One bucket of original requests and the live request lengths replayed through it in order. */
        struct PanelReplay
        {
            int n = 0;                ///< Physical output columns, including a masked SIMD tail.
            int k = 0;                ///< Logical K, a multiple of 32.
            int requests = 1;         ///< Independent original requests sharing the bucket.
            int rows_per_request = 0; ///< Physical rows owned by each request.
            std::vector<std::vector<std::int32_t>> live; ///< One length per request for each replayed cell.
        };

        /**
         * @brief Replay one source through production Auto prefill at one exact team width.
         * @param source GGUF-layout source tensor of shape {n, k}.
         * @param policy Arithmetic policy of the prepared matrix.
         * @param workers OpenMP team width; the serial reference uses the same width.
         * @param replay Bucket geometry and live cells.
         * @param require_k_partitions Whether the serial tree must fold several K partitions.
         * @param admitted_workers Participant width the slab is admitted for; at least @p workers.
         *
         * Every cell is compared bitwise with independent serial M=1 rows at the
         * same width (the backend-native K tree depends on it); unauthored rows
         * and the N padding stay dirty. A bucket with at least 16 live rows must
         * select the packed panel when every worker owns an N strip; a sparser
         * bucket keeps the ordinary route even though its storage was admitted
         * for the panel. The scratch is exactly the borrow admitted at
         * @p admitted_workers, so a narrower team proves the admitted quote covers
         * it. On the panel route each worker writes only its own slot: its first
         * panel, live carry rows and nothing past the slot's panel/carry extent
         * or the whole invocation demand.
         */
        void replayPackedPrefill(const TensorBase &source, CPUProjectionNumericalPolicy policy, int workers,
            const PanelReplay &replay, bool require_k_partitions, int admitted_workers = 0)
        {
            admitted_workers = admitted_workers ? admitted_workers : workers;
            constexpr float dirty = -12345.5f;
            constexpr auto dirty_bits = std::bit_cast<std::uint32_t>(dirty);
            const auto clean = [](float value) { return std::bit_cast<std::uint32_t>(value) == dirty_bits; };
            const int n = replay.n, k = replay.k, blocks = k / 32, stride = n + 5;
            const int rows = replay.requests * replay.rows_per_request;
            SCOPED_TRACE("workers=" + std::to_string(workers) + " admitted=" + std::to_string(admitted_workers) +
                " policy=" + std::to_string(int(policy)));
            ASSERT_GE(admitted_workers, workers);
            ScopedPanelTeam team(workers);
            ASSERT_EQ(nativeQuantisedInvocationWorkerCount(), workers);

            CPUNativeQuantisedPackedWeights packed;
            packed.numerical_policy = policy;
            ASSERT_TRUE(packWeightsCPUNativeQuantised(&source, packed));
            const auto format = x86_panel::formatFor(packed.encoding, packed.codebook_id);
            ASSERT_NE(format, x86_panel::Format::None);
            const int lanes = activeISALevel() >= ISALevel::AVX512 ? 16 : 8;
            const auto config = serialTileConfigForPackedMatrix(packed, n, k, workers);
            if (require_k_partitions)
            {
                // Restate the serial M=1 split (CPUNativeQuantisedTileConfig.h)
                // for this cell instead of trusting the host's cache regime:
                // one N task, fewer than the team; at least 256 blocks and 16M
                // FLOPs admit the split in both regimes. A memory-bound host
                // targets one partition per worker and a compute-bound host
                // eight, both far below blocks/4, so both fold several.
                const auto &tile = debugEnv().cpu_vnni;
                ASSERT_TRUE(tile.k_tiles <= 0 && tile.min_bpr_k_parallel <= 0 && tile.n_block_chunks <= 0)
                    << "LLAMINAR_CPU_VNNI_* overrides replace the serial K tree under test";
                ASSERT_GE(workers, 2);
                ASSERT_LE(n, 64);
                ASSERT_GE(blocks, 256);
                ASSERT_GE(2LL * n * k, 16'000'000LL);
                ASSERT_GE(blocks / 4, 8 * workers);
                ASSERT_TRUE(config.k_tiles == workers || config.k_tiles == 8 * workers)
                    << "k_tiles=" << config.k_tiles << " workers=" << workers;
            }
            const int partitions = std::max(1, config.k_tiles);

            // Independent slot geometry: a worker's strip is a whole 64-column
            // chunk on AVX-512 when every worker of this team can own one, else
            // one SIMD vector wide. Up to eight K blocks of eight dot-group
            // planes plus a scale plane (and a minimum plane for Q4_K/Q5_K),
            // then one carry partial of the strip width per original row when a
            // later partition spans several panels; each slot is padded to a
            // 64-byte line.
            const int strip = lanes == 16 && (n + 63) / 64 >= workers ? 64 : lanes;
            const bool affine = format == x86_panel::Format::Q4K || format == x86_panel::Format::Q5K;
            const std::size_t panel_floats = std::size_t(std::min(blocks, 8)) * (affine ? 10 : 9) * strip;
            const bool carry = partitions > 1 && (blocks + partitions - 1) / partitions > 8;
            const std::size_t carry_floats = carry ? std::size_t(rows) * strip : 0;
            const std::size_t pitch = (panel_floats + carry_floats + 15) / 16 * 16;
            // The panel needs one N strip of at least one vector per worker of this exact team.
            const bool eligible = (n + lanes - 1) / lanes >= workers;
            const auto quoted = x86_panel::invocationFloats(format, lanes, rows, n, blocks, config.k_tiles, workers);
            ASSERT_EQ(quoted, eligible ? pitch * workers : 0u);
            ASSERT_GE(nativeQuantisedProjectionPartialFloats(packed, rows), quoted);
            std::size_t borrow = 0, envelope = 0;
            {
                // Admission happens once, at the participant's width.
                ScopedPanelTeam admitted(admitted_workers);
                ASSERT_EQ(nativeQuantisedInvocationWorkerCount(), admitted_workers);
                borrow = nativeQuantisedProjectionPartialFloats(packed, rows);
                envelope = nativeQuantisedProjectionPartialEnvelopeFloats(packed, rows);
            }
            ASSERT_EQ(nativeQuantisedInvocationWorkerCount(), workers);
            ASSERT_GE(borrow, quoted) << "the admitted quote under-borrows a narrower team";
            ASSERT_GE(envelope, quoted);
            constexpr std::size_t guard = 64;
            std::vector<float> scratch(borrow + guard, dirty);

            std::mt19937 rng(2323u + static_cast<std::uint32_t>(k));
            std::vector<Q8_1Block> input(std::size_t(rows) * blocks);
            for (auto &block : input)
            {
                block.d = static_cast<std::uint16_t>(0x2000 + rng() % 0x1000);
                for (auto &byte : block.qs)
                {
                    byte = static_cast<std::int8_t>(int(rng() % 256) - 128);
                    block.sum_qs += byte;
                }
            }
            std::vector<float> serial(std::size_t(rows) * stride, dirty), actual(serial.size(), dirty);
            std::vector<float> serial_scratch(nativeQuantisedProjectionPartialFloats(packed, 1));
            for (int row = 0; row < rows; ++row)
                gemv_native_quantised_preq(packed, input.data() + std::size_t(row) * blocks,
                    serial.data() + std::size_t(row) * stride, serial_scratch);

            // The exact panel prefix, including carry, is validated before any write.
            if (eligible)
                EXPECT_THROW(gemm_native_quantised_preq(packed, input.data(), actual.data(),
                    std::span<float>(scratch).first(quoted - 1), rows, stride), std::invalid_argument);
            ASSERT_TRUE(std::all_of(actual.begin(), actual.end(), clean));

            for (const auto &lengths : replay.live)
            {
                ASSERT_EQ(lengths.size(), std::size_t(replay.requests));
                const int live = std::accumulate(lengths.begin(), lengths.end(), 0);
                std::string cell = "live=";
                for (const auto length : lengths)
                    cell += std::to_string(length) + ",";
                SCOPED_TRACE(cell);
                const auto work = replay.requests == 1
                    ? ProjectionRowWork::fromPrefix(DeviceRowRange::deviceCounted(rows, lengths.data()))
                    : ProjectionRowWork::fromRequests(
                          DeviceRequestRowRanges::deviceCounted(replay.requests, replay.rows_per_request, lengths.data()));
                std::fill(actual.begin(), actual.end(), dirty);
                std::fill(scratch.begin(), scratch.end(), dirty);
                PerfStatsCollector::reset();
                gemm_native_quantised_preq(packed, input.data(), actual.data(), scratch, rows, stride, ISAPath::AUTO,
                    VerifierRowsPolicy::Auto, PrefillSchedulePolicy::Auto, 0, &work);
                const auto launch = observedPrefillLaunch();

                const auto authored = [&](int row) {
                    return row % replay.rows_per_request < lengths[std::size_t(row / replay.rows_per_request)];
                };
                for (int row = 0; row < rows; ++row)
                    for (int col = 0; col < stride; ++col)
                    {
                        const auto index = std::size_t(row) * stride + col;
                        const auto expected = authored(row) ? serial[index] : dirty;
                        ASSERT_EQ(std::bit_cast<std::uint32_t>(actual[index]), std::bit_cast<std::uint32_t>(expected))
                            << "row=" << row << " column=" << col;
                    }
                for (std::size_t index = borrow; index < scratch.size(); ++index)
                    ASSERT_TRUE(clean(scratch[index])) << "write past the admitted projection borrow at " << index;

                if (live == 0)
                {
                    EXPECT_EQ(launch.count, 0u) << "an empty bucket must not launch";
                    continue;
                }
                ASSERT_EQ(launch.count, 1u);
                EXPECT_EQ(launch.threads, std::to_string(workers));
                const bool panel = eligible && live >= x86_panel::min_rows;
                ASSERT_EQ(launch.route == "packed_panel", panel) << "route=" << launch.route;
                if (!panel)
                {
                    if (require_k_partitions)
                        EXPECT_EQ(launch.route, "decode_equivalent_kpart_rows");
                    continue;
                }
                EXPECT_EQ(launch.k_tiles, std::to_string(partitions));
                for (std::size_t index = quoted; index < borrow; ++index)
                    ASSERT_TRUE(clean(scratch[index])) << "panel write past its quote at " << index;
                for (int worker = 0; worker < workers; ++worker)
                {
                    const auto base = std::size_t(worker) * pitch;
                    // Every worker owns at least one N strip and packs its first panel here.
                    EXPECT_FALSE(clean(scratch[base])) << "worker " << worker << " did not pack into its own slot";
                    for (int row = 0; row < rows; ++row)
                        if (carry && !authored(row))
                            for (int lane = 0; lane < strip; ++lane)
                                ASSERT_TRUE(clean(scratch[base + panel_floats + std::size_t(row) * strip + lane]))
                                    << "worker " << worker << " carried unauthored row " << row;
                    for (auto index = base + panel_floats + carry_floats; index < base + pitch; ++index)
                        ASSERT_TRUE(clean(scratch[index])) << "worker " << worker << " wrote slot padding " << index;
                }
            }
        }

        /** @return One source of every x86 panel format (Q5_K, IQ3_S, Q4_K, IQ4_XS, Q5_0) with shape {n, k}. */
        std::vector<std::pair<std::string, std::unique_ptr<TensorBase>>> panelSources(int n, int k)
        {
            const std::vector<std::size_t> shape{static_cast<std::size_t>(n), static_cast<std::size_t>(k)};
            std::vector<std::pair<std::string, std::unique_ptr<TensorBase>>> sources;
            for (const auto &format : compactSources())
                if (format.codebook == 7 || format.codebook == 11)
                    sources.emplace_back(format.name, format.make(shape, 121));
            sources.emplace_back("Q4_K", q4KSource(shape, 321));
            sources.emplace_back("IQ4_XS", Factory::createIQ4_XSRandom(shape, 789));
            sources.emplace_back("Q5_0", Factory::createQ5_0Random(shape, 201));
            return sources;
        }
    } // namespace

    /** @test Replay the installed x86 packed prefill route against independent serial rows at team widths 1-4. */
    TEST(CPUCodedCompactUnits, X86PackedPrefillKeepsSerialAndTailBytes)
    {
        ScopedPrefillLaunchObservation observation;
        // Two original requests of 17 rows share one 34-row bucket. Live totals
        // 34, 16, 17 and 33 select the panel, including a 9+7 split across the
        // request boundary; 3 and 15 are sparse prefixes of a bucket admitted for
        // the panel and keep the ordinary route; 0 launches nothing. The
        // GPU-aligned tree always has 16 partitions: K=1120 (35 blocks) leaves
        // empty trailing partitions and K=4768 (149 blocks, ten per partition)
        // carries partials across panels. On AVX-512, teams of one and two
        // workers own whole 64-column strips and teams of three and four own
        // sixteen-column strips, so both strip widths are replayed.
        for (const int k : {1120, 4768})
        {
            const PanelReplay replay{.n = 67, .k = k, .requests = 2, .rows_per_request = 17,
                .live = {{17, 17}, {0, 0}, {3, 0}, {9, 6}, {9, 7}, {16, 0}, {0, 17}, {17, 16}}};
            for (const auto &[name, source] : panelSources(replay.n, k))
            {
                SCOPED_TRACE(name + " k=" + std::to_string(k));
                for (const auto policy : {CPUProjectionNumericalPolicy::BackendNative, CPUProjectionNumericalPolicy::GPUAlignedExpert})
                    for (int workers = 1; workers <= 4; ++workers)
                        replayPackedPrefill(*source, policy, workers, replay, false);
            }
        }
        // A participant admitted for four workers where N=17 has fewer strips
        // (two sixteen-column AVX-512 strips, or one whole chunk for a single
        // worker; three AVX2 strips): four workers cannot select the panel, but
        // narrower teams do and borrow from the four-worker admission. A quote
        // of the admitted width alone was zero, so backend-native teams of one
        // to three workers under-borrowed here.
        const PanelReplay narrower{.n = 17, .k = 1120, .requests = 1, .rows_per_request = 17,
            .live = {{17}, {3}}};
        for (const auto &[name, source] : panelSources(narrower.n, narrower.k))
        {
            SCOPED_TRACE(name + " narrower than admitted");
            for (const auto policy : {CPUProjectionNumericalPolicy::BackendNative, CPUProjectionNumericalPolicy::GPUAlignedExpert})
                for (int workers = 1; workers <= 4; ++workers)
                    replayPackedPrefill(*source, policy, workers, narrower, false, 4);
        }
        // The backend-native tree splits K only for at least 16M FLOPs with fewer
        // N tasks than workers. One 64-column chunk with 4104 K blocks gives
        // exactly one or eight partitions per worker at every width >= 2 (memory-
        // or compute-bound host), each longer than one panel, so partials fold
        // and carry across panels in either cache regime.
        const PanelReplay partitioned{.n = 61, .k = 4104 * 32, .requests = 1, .rows_per_request = 17,
            .live = {{17}, {3}}};
        for (const auto &[name, source] : panelSources(partitioned.n, partitioned.k))
        {
            SCOPED_TRACE(name + " backend-native K partitions");
            for (int workers = 2; workers <= 4; ++workers)
                replayPackedPrefill(*source, CPUProjectionNumericalPolicy::BackendNative, workers, partitioned, true);
        }
    }
#endif

    TEST(CPUCompensationFreeUnits, PreparedBytesMatchCompensationFreeLayout)
    {
        for (const auto &source : compensationFreeSources())
        {
            const auto tensor = source.make({67, 512}, 9);
            SCOPED_TRACE(source.name);
            CPUNativeQuantisedPackedWeights packed;
            ASSERT_TRUE(packWeightsCPUNativeQuantised(tensor.get(), packed));
            EXPECT_EQ(packed.native_interleaved.size(), source.prepared_bytes_per_32 * 128 * 16);
            if (source.name == "Q4_K")
                EXPECT_TRUE(packed.superblockCoded());
            if (source.name == "Q4_1")
                EXPECT_FALSE(packed.superblockCoded());
        }
    }

    /** A ragged final K superblock keeps one header for only its live payload/code blocks. */
    TEST(CPUCompensationFreeUnits, Q4KRaggedTailKeepsOneSourceHeader)
    {
        constexpr int N = 67, K = 288, Blocks = K / 32;
        const auto tensor = q4KSource({N, K}, 9);
        const auto *source = IINT8Unpackable::fromTensor(tensor.get());
        ASSERT_NE(source, nullptr);
        CPUNativeQuantisedPackedWeights packed;
        ASSERT_TRUE(packWeightsCPUNativeQuantised(tensor.get(), packed));
        const auto view = packed.readView();
        EXPECT_EQ(view.geometry().rowBytes(), (16 * 64 + 96) * Blocks + 2 * 256);
        EXPECT_EQ(view.bytes().size(), view.geometry().rowBytes() * 2);
        for (int column = 0; column < N; ++column)
            for (int block = 0; block < Blocks; ++block)
            {
                EXPECT_EQ(view.blockScale(column / 64, block, column % 64),
                    fp16_to_fp32(fp32_to_fp16(source->get_block_scale(column, block))));
                EXPECT_EQ(view.blockMin(column / 64, block, column % 64),
                    fp16_to_fp32(fp32_to_fp16(source->get_block_min(column, block))));
            }
    }

    /** The two unsigned dot programs reconstruct every signed product, including signed minima. */
    TEST(CPUCompensationFreeUnits, WeightBiasAndActivationBiasProduceIdenticalIntegerDots)
    {
        for (int a = -128; a <= 127; ++a)
            for (int w = -128; w <= 127; ++w)
            {
                const int previous = (a + 128) * w - 128 * w;
                const int next = a * (w + 128) - 128 * a;
                const auto product = std::int64_t(a) * w;
                ASSERT_EQ(previous, product) << "a=" << a << " w=" << w;
                ASSERT_EQ(next, product) << "a=" << a << " w=" << w;
#if defined(__AVX2__)
                const auto dot = dotUnsignedWeightsAVX2(_mm256_setzero_si256(), _mm256_set1_epi8(static_cast<char>(w + 128)),
                    _mm256_set1_epi8(static_cast<char>(a)));
                ASSERT_EQ(_mm256_extract_epi32(dot, 0) - 4 * 128 * a, 4 * product) << "a=" << a << " w=" << w;
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
                const auto dot512 = dotSignedQ8Weights512(_mm512_setzero_si512(),
                    _mm512_set1_epi8(static_cast<char>(a)), _mm512_set1_epi8(static_cast<char>(w)));
                ASSERT_EQ(_mm_cvtsi128_si32(_mm512_castsi512_si128(dot512)) - 4 * 128 * a, 4 * product)
                    << "a=" << a << " w=" << w;
#endif
            }
    }

    namespace
    {
        /**
         * @brief Require a published Q8 block's sum to equal its 32 published bytes.
         * @param block Publication under test.
         * @param valid Live source values; every later byte must be zero padding.
         * @param quantizer Production entry point named in a failure.
         */
        void expectExactActivationSum(const Q8_1Block &block, int valid, const std::string &quantizer)
        {
            int sum = 0;
            for (int index = 0; index < 32; ++index)
            {
                if (index >= valid)
                    ASSERT_EQ(block.qs[index], 0) << quantizer << " left padding byte " << index;
                sum += block.qs[index];
            }
            ASSERT_EQ(int(block.sum_qs), sum) << quantizer << " valid=" << valid;
        }

        /** @return FP32 blocks covering random magnitudes, saturation endpoints, ties, signed zeros and subnormals. */
        std::vector<std::array<float, 32>> activationSumBlocks()
        {
            std::vector<std::array<float, 32>> blocks;
            std::mt19937 rng(4242);
            std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
            for (const float magnitude : {1e-12f, 1e-6f, 1.0f, 3.0f, 1e4f, 1e30f})
                for (int sample = 0; sample < 64; ++sample)
                {
                    std::array<float, 32> block{};
                    for (auto &value : block)
                        value = magnitude * unit(rng);
                    blocks.push_back(block);
                }
            const auto add = [&](auto value) {
                std::array<float, 32> block{};
                for (int index = 0; index < 32; ++index)
                    block[index] = value(index);
                blocks.push_back(block);
            };
            add([](int) { return 1.0f; });
            add([](int) { return -1.0f; });
            add([](int index) { return index % 2 ? -2.5f : 2.5f; });
            add([](int index) { return index ? (float(index) - 0.5f) / 127.0f : 1.0f; }); // Near-ties after scaling.
            add([](int index) { return index < 16 ? 127.5f / 127.0f : -126.5f / 127.0f; });
            add([](int index) { return index == 7 ? -3.0f : 0.0f; });
            add([](int index) { return index % 3 ? -0.0f : 0.0f; });
            add([](int index) { return std::ldexp(float(index - 16), -140); });
            return blocks;
        }
    } // namespace

    /**
     * @test Every production activation quantizer publishes `sum_qs` equal to its 32 published bytes.
     *
     * Production activations are FP32, so CPU NativeQuantised projections
     * receive Q8_1 rows only from these entry points: the backend-native SIMD
     * block quantizer and its two-block form, the GPU-aligned expert program,
     * the row driver and the fused SwiGLU publication. The AVX2 nibble dots use
     * zero points 12 (MXFP4) and 0 (IQ4_NL/IQ4_XS) where AVX-512 uses 128; the
     * corrections agree, and equal the signed dot, only for an exact sum. Every
     * ISA variant compiled into this build is also called directly.
     */
    TEST(CPUCompensationFreeUnits, ProductionActivationQuantizersPublishExactSums)
    {
        using Policy = CPUProjectionNumericalPolicy;
        constexpr std::array policies{Policy::BackendNative, Policy::GPUAlignedExpert};
        const auto blocks = activationSumBlocks();
        Q8_1Block first{}, second{};
        const auto dirty = [](Q8_1Block &block) { std::memset(&block, 0x5a, sizeof(block)); };
        for (const auto &values : blocks)
        {
            for (int valid = 0; valid <= 32; ++valid)
            {
                dirty(first);
                simd::quantize_single_block(values.data(), first, valid);
                expectExactActivationSum(first, valid, "simd::quantize_single_block");
                dirty(first);
                simd::quantize_single_block_scalar(values.data(), first, valid);
                expectExactActivationSum(first, valid, "simd::quantize_single_block_scalar");
                for (const auto policy : policies)
                {
                    // The GPU-aligned program admits only nonempty blocks.
                    if (valid == 0 && policy == Policy::GPUAlignedExpert)
                        continue;
                    dirty(first);
                    quantizeActivationBlock(values.data(), first, valid, policy);
                    expectExactActivationSum(first, valid, "quantizeActivationBlock policy=" + std::to_string(int(policy)));
                }
                if (valid > 0)
                {
                    dirty(first);
                    cpu::gpu_aligned_expert_q8::quantizeBlock(values.data(), first, valid);
                    expectExactActivationSum(first, valid, "gpu_aligned_expert_q8::quantizeBlock");
                }
            }
#if defined(__AVX2__)
            if (cpu_supports_avx2())
            {
                dirty(first);
                simd::quantize_single_block_avx2(values.data(), first);
                expectExactActivationSum(first, 32, "simd::quantize_single_block_avx2");
            }
#endif
#if defined(__AVX512F__)
            if (cpu_supports_avx512())
            {
                dirty(first);
                simd::quantize_single_block_avx512(values.data(), first);
                expectExactActivationSum(first, 32, "simd::quantize_single_block_avx512");
            }
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
            dirty(first);
            simd::quantize_single_block_neon(values.data(), first);
            expectExactActivationSum(first, 32, "simd::quantize_single_block_neon");
#endif
        }
        // Two-block forms over adjacent pairs, including a zero block beside a live one.
        for (std::size_t index = 0; index + 1 < blocks.size(); ++index)
        {
            std::array<float, 64> pair{};
            std::copy(blocks[index].begin(), blocks[index].end(), pair.begin());
            std::copy(blocks[index + 1].begin(), blocks[index + 1].end(), pair.begin() + 32);
            for (const auto policy : policies)
            {
                dirty(first);
                dirty(second);
                quantizeTwoActivationBlocks(pair.data(), first, second, policy);
                expectExactActivationSum(first, 32, "quantizeTwoActivationBlocks first");
                expectExactActivationSum(second, 32, "quantizeTwoActivationBlocks second");
            }
            dirty(first);
            dirty(second);
            cpu::gpu_aligned_expert_q8::quantizeTwoBlocks(pair.data(), first, second);
            expectExactActivationSum(first, 32, "gpu_aligned_expert_q8::quantizeTwoBlocks first");
            expectExactActivationSum(second, 32, "gpu_aligned_expert_q8::quantizeTwoBlocks second");
#if defined(__AVX512F__)
            if (cpu_supports_avx512())
            {
                dirty(first);
                dirty(second);
                simd::quantize_two_blocks_avx512(pair.data(), first, second);
                expectExactActivationSum(first, 32, "simd::quantize_two_blocks_avx512 first");
                expectExactActivationSum(second, 32, "simd::quantize_two_blocks_avx512 second");
            }
#endif
        }
        // Row drivers over ragged K and the fused SwiGLU publication.
        constexpr int rows = 3;
        std::mt19937 rng(77);
        std::uniform_real_distribution<float> unit(-4.0f, 4.0f);
        for (const int k : {32, 64, 97, 1000, 1024})
        {
            const int k_blocks = (k + 31) / 32;
            std::vector<float> activation(std::size_t(rows) * k), gate(activation.size()), up(activation.size());
            for (std::size_t index = 0; index < activation.size(); ++index)
            {
                activation[index] = blocks[index / 32 % blocks.size()][index % 32];
                gate[index] = unit(rng);
                up[index] = unit(rng);
            }
            std::vector<Q8_1Block> published(std::size_t(rows) * k_blocks);
            for (const auto policy : policies)
            {
                SCOPED_TRACE("k=" + std::to_string(k) + " policy=" + std::to_string(int(policy)));
                std::memset(published.data(), 0x5a, published.size() * sizeof(Q8_1Block));
                cpu::native_quantised::quantize_activations_to_q8_1(
                    activation.data(), published.data(), rows, k, k_blocks, policy);
                for (std::size_t block = 0; block < published.size(); ++block)
                    expectExactActivationSum(published[block], std::min(32, k - int(block % k_blocks) * 32),
                        "quantize_activations_to_q8_1");
                if (k % 32)
                    continue;
                std::memset(published.data(), 0x5a, published.size() * sizeof(Q8_1Block));
                cpu::native_quantised::swiglu_quantize_activations_to_q8_1(
                    gate.data(), up.data(), published.data(), rows, k, k_blocks, policy);
                for (const auto &block : published)
                    expectExactActivationSum(block, 32, "swiglu_quantize_activations_to_q8_1");
            }
        }
    }

#if defined(__AVX2__)
    /** @test The AVX2 nibble domains preserve signed-byte endpoints for every adjacent pair. */
    TEST(CPUCompensationFreeUnits, AVX2NibbleDotsMatchEveryAdjacentSelectorPair)
    {
        const auto check = []<NibbleDotAVX2 Domain>(std::uint8_t cb) {
            const auto lut = build_decode_lut_avx2_for_codebook(cb);
            const int bias = nibbleWeightBiasAVX2<Domain>(cb);
            for (int lo = 0; lo < 16; ++lo)
                for (int hi = 0; hi < 16; ++hi)
                {
                    const std::uint8_t selector = lo | (hi << 4);
                    std::uint8_t payload[16];
                    std::fill_n(payload, 16, selector);
                    std::int8_t reference[32];
                    decode_native_block(cb, payload, reference);
                    const auto indices = _mm256_set1_epi32(lo | (hi << 8) | (hi << 16) | (lo << 24));
                    const auto weight = decodeNibbleAVX2<Domain>(indices, lut);
                    for (const int a : {-128, -127, 0, 127})
                        for (const int b : {-128, -127, 0, 127})
                        {
                            const auto ab = static_cast<std::int8_t>(a), bb = static_cast<std::int8_t>(b);
                            const auto word = _mm256_set1_epi32(std::bit_cast<std::int32_t>(std::array{ab, bb, bb, ab}));
                            auto plane = _mm256_setzero_si256();
                            for (int group = 0; group < 4; ++group)
                                plane = dotNibbleAVX2<Domain>(plane, weight, word);
                            // Identical planes exercise the worst-case intermediate range
                            // (including the MXFP4 domain that cannot merge in INT16).
                            const auto result = finishNibbleDotAVX2<Domain>(plane, plane);
                            ASSERT_EQ(_mm256_extract_epi32(result, 0) - bias * 16 * (a + b),
                                16 * (a * reference[0] + b * reference[16]));
                        }
                }
        };
        check.template operator()<NibbleDotAVX2::Linear>(native_quantised_formats::Q4_0.codebook_id);
        check.template operator()<NibbleDotAVX2::Linear>(native_quantised_formats::Q4_1.codebook_id);
        check.template operator()<NibbleDotAVX2::SmallLUT>(kNativeQuantisedMXFP4Codebook);
        check.template operator()<NibbleDotAVX2::SignedLUT>(native_quantised_formats::IQ4_NL.codebook_id);
    }

    /** @test Deferred widening stays exact for signed bytes even at the INT16 chain bounds. */
    TEST(CPUCompensationFreeUnits, AVX2Q8DotsWidenAfterEightGroups)
    {
        alignas(32) std::array<std::uint8_t, 8 * 256> weights{};
        Q8_1Block row0{}, row1{};
        for (int w = -128; w <= 127; ++w)
        {
            for (int group = 0; group < 8; ++group)
                for (int column = 0; column < 8; ++column)
                    std::fill_n(weights.data() + group * 256 + column * 4, 4,
                        static_cast<std::uint8_t>(column % 2 ? -1 - w : w));
            for (int a = -128; a <= 127; ++a)
            {
                const int b = -1 - a;
                std::fill_n(row0.qs, 32, static_cast<std::int8_t>(a));
                std::fill_n(row1.qs, 32, static_cast<std::int8_t>(b));
                row0.sum_qs = 32 * a;
                row1.sum_qs = 32 * b;
                const auto one = dotSignedQ8BlockAVX2(weights.data(), std::array<const Q8_1Block *, 1>{&row0});
                const auto two = dotSignedQ8BlockAVX2(weights.data(), std::array<const Q8_1Block *, 2>{&row0, &row1});
                ASSERT_EQ(_mm256_extract_epi32(one.values[0], 0), 32 * w * a);
                ASSERT_EQ(_mm256_extract_epi32(two.values[0], 1), 32 * (-1 - w) * a);
                ASSERT_EQ(_mm256_extract_epi32(two.values[1], 0), 32 * w * b);
            }
        }
    }
#endif

    /** Real SIMD leaves preserve old projection bytes at M=1..4, including a partial N tile and ordered K folds. */
    TEST(CPUCompensationFreeUnits, SimdProjectionMatchesPreviousPerBlockProgram)
    {
        constexpr int N = 67, K = 512, Blocks = K / 32;
        std::array<Q8_1Block, 4 * Blocks> activation{};
        std::mt19937 rng(352);
        for (auto &a : activation)
        {
            a.d = 0x2800 + (rng() % 1024);
            for (auto &v : a.qs)
            {
                v = static_cast<std::int8_t>(int(rng() % 256) - 128);
                a.sum_qs += v;
            }
        }
        for (const auto &source : compensationFreeSources())
        {
            const auto tensor = source.make({N, K}, 23 + source.codebook);
            const auto *unpackable = IINT8Unpackable::fromTensor(tensor.get());
            ASSERT_NE(unpackable, nullptr);
            CPUNativeQuantisedPackedWeights packed;
            ASSERT_TRUE(packWeightsCPUNativeQuantised(tensor.get(), packed));
            for (const auto policy : {CPUProjectionNumericalPolicy::BackendNative, CPUProjectionNumericalPolicy::GPUAlignedExpert})
            {
                packed.numerical_policy = policy;
                const int partitions = policy == CPUProjectionNumericalPolicy::GPUAlignedExpert
                    ? MoEProjectionNumericalContract::orderedKPartitionsForWidth(K) : 1;
                const int span = (Blocks + partitions - 1) / partitions;
                for (int rows = 1; rows <= 4; ++rows)
                {
                    std::array<float, 4 * N> output{};
                    SCOPED_TRACE(source.name + " rows=" + std::to_string(rows) + " policy=" + std::to_string(int(policy)));
#if defined(__aarch64__) && defined(__ARM_NEON)
                    constexpr auto instruction =
#if defined(__ARM_FEATURE_MATMUL_INT8)
                        multi_scale::NEONByteDot::I8MM;
#elif defined(__ARM_FEATURE_DOTPROD)
                        multi_scale::NEONByteDot::DotProd;
#else
                        multi_scale::NEONByteDot::ASIMD;
#endif
                    const auto run = [&]<std::size_t Rows>() {
                        const auto view = packed.readView();
                        std::array<std::span<const Q8_1Block>, Rows> inputs{};
                        for (std::size_t row = 0; row < Rows; ++row)
                            inputs[row] = std::span(activation).subspan(row * Blocks, Blocks);
                        for (int chunk = 0; chunk < 2; ++chunk)
                        {
                            std::array<std::span<float>, Rows> outputs{};
                            for (std::size_t row = 0; row < Rows; ++row)
                                outputs[row] = std::span(output).subspan(row * N + chunk * 64, std::min(64, N - chunk * 64));
                            CPUNativeNEONProjectionTile<Rows>::bind(view, inputs, outputs, chunk).template execute<instruction>();
                        }
                    };
                    switch (rows)
                    {
                    case 1: run.template operator()<1>(); break;
                    case 2: run.template operator()<2>(); break;
                    case 3: run.template operator()<3>(); break;
                    case 4: run.template operator()<4>(); break;
                    }
#else
                    std::array<float, 4 * Blocks * 128> scratch{};
                    // The ordered expert K tree remains active at M=1, but
                    // only the serial GEMV entry admits a single live row.
                    if (rows == 1)
                        gemv_native_quantised_preq(packed, activation.data(), output.data(), scratch);
                    else
                        gemm_native_quantised_preq(packed, activation.data(), output.data(), scratch, rows, N);
#endif
                    for (int row = 0; row < rows; ++row)
                        for (int column = 0; column < N; ++column)
                        {
                            float expected = 0.0f;
                            for (int part = 0; part < partitions; ++part)
                            {
                                float partial = 0.0f;
                                for (int block = part * span; block < std::min((part + 1) * span, Blocks); ++block)
                                    partial = previousBlock(*unpackable, packed, activation[row * Blocks + block], column, block, partial);
                                expected = part == 0 ? partial : device_fp32_contract::add(expected, partial);
                            }
                            EXPECT_EQ(std::bit_cast<std::uint32_t>(output[row * N + column]), std::bit_cast<std::uint32_t>(expected))
                                << "row=" << row << " column=" << column;
                        }
                }
            }
        }
    }
} // namespace llaminar2::test
