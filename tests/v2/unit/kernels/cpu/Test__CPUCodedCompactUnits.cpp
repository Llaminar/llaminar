/**
 * @file Test__CPUCodedCompactUnits.cpp
 * @brief Superblock-coded compact CPU units: exact size, exact metadata, exact outputs.
 *
 * Compact multi-scale CPU units (Q3_K, Q2_K, IQ3_XXS, IQ2_S, IQ2_XS, IQ2_XXS,
 * IQ1_S, IQ1_M) store the source payload at its exact width, the source
 * scale code at its exact width and one `d`/`dmin` word per (column, source
 * superblock). This suite proves, device-free and on the build's native ISA
 * (NEON, AVX2 or AVX-512):
 *
 * - the SIMD code/word readers equal the scalar plane readers for every
 *   code width and every aligned column vector;
 * - the SIMD factor reconstruction equals widening the codec's staged word
 *   (decodeSuperblockMetadata, itself proven against the per-block packer by
 *   V2_Unit_VnniCodedPacking) for every code and a broad d/dmin sweep,
 *   including zero, subnormal, negative, infinite and NaN half words (IEEE
 *   754 leaves the sign and payload of an arithmetic NaN unspecified, and
 *   compilers may fold `-dmin * c` into `dmin * -c`, so a NaN result only
 *   has to be a NaN; every other result is compared bit for bit);
 * - a prepared matrix occupies exactly the GGUF bytes of its 64-padded
 *   columns, nothing else;
 * - the production SIMD projection equals the scalar codec reference for
 *   1..4 rows, a ragged final column chunk and every compact format.
 */
#include <gtest/gtest.h>

#include "../../../utils/TestTensorFactory.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedCodedMetadata.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedMultiScaleKernels.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedWeightPacker.h"
#include "tensors/NativeQuantisedSuperblockMetadata.h"

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
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
            double file_bytes_per_32 = 0;
            std::function<std::unique_ptr<TensorBase>(const std::vector<size_t> &, uint32_t)> make;
        };

        std::vector<CompactSource> compactSources()
        {
            return {
                {"Q3_K", 9, 13.75, [](auto s, auto r) { return Factory::createQ3_KRandom(s, r); }},
                {"Q2_K", 10, 10.5, [](auto s, auto r) { return Factory::createQ2_KRandom(s, r); }},
                {"IQ3_XXS", 12, 12.25, [](auto s, auto r) { return Factory::createIQ3_XXSRandom(s, r); }},
                {"IQ2_S", 13, 10.25, [](auto s, auto r) { return Factory::createIQ2_SRandom(s, r); }},
                {"IQ2_XS", 14, 9.25, [](auto s, auto r) { return Factory::createIQ2_XSRandom(s, r); }},
                {"IQ2_XXS", 15, 8.25, [](auto s, auto r) { return Factory::createIQ2_XXSRandom(s, r); }},
                {"IQ1_S", 16, 6.25, [](auto s, auto r) { return Factory::createIQ1_SRandom(s, r); }},
                {"IQ1_M", 17, 7.0, [](auto s, auto r) { return Factory::createIQ1_MRandom(s, r); }},
            };
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
        for (std::uint8_t codebook : {9, 10, 11, 13, 17})
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

    /** SIMD reconstruction equals widening the codec's staged words for every derived-scale compact format. */
    TEST(CPUCodedCompactUnits, SimdReconstructionEqualsCodecWordsExhaustively)
    {
        expectReconstructionMatchesCodec<9>();
        expectReconstructionMatchesCodec<10>();
        expectReconstructionMatchesCodec<12>();
        expectReconstructionMatchesCodec<14>();
        expectReconstructionMatchesCodec<15>();
        expectReconstructionMatchesCodec<16>();
        expectReconstructionMatchesCodec<17>();
    }

    /** A prepared compact matrix holds exactly the file bytes of its 64-padded columns. */
    TEST(CPUCodedCompactUnits, PreparedBytesEqualFileDensity)
    {
        for (const auto &source : compactSources())
        {
            SCOPED_TRACE(source.name);
            const auto tensor = source.make({128, 1024}, 3u);
            CPUNativeQuantisedPackedWeights packed;
            ASSERT_TRUE(packWeightsCPUNativeQuantised(tensor.get(), packed));
            ASSERT_TRUE(packed.usesCompactMultiScale());
            const double file = source.file_bytes_per_32 * 128.0 * (1024 / 32);
            EXPECT_EQ(static_cast<double>(packed.native_interleaved.size()), file);
            EXPECT_EQ(packed.preparedBytes(), packed.native_interleaved.size());
            EXPECT_EQ(static_cast<double>(packed.preparedFootprint().weight_bytes_per_n_chunk_k_block),
                      source.file_bytes_per_32 * 64.0);
        }
    }

    /** The production SIMD projection equals the scalar codec reference for every compact format. */
    TEST(CPUCodedCompactUnits, SimdProjectionEqualsScalarCodecReference)
    {
        constexpr int kN = 67, kK = 512, kBlocks = kK / 32;
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
            const auto view = packed.readView();
            for (int chunk = 0; chunk < packed.N_padded / 64; ++chunk)
            {
                const int columns = std::min(64, kN - chunk * 64);
                std::array<std::array<float, 64>, 4> actual{}, expected{};
                for (int row = 0; row < 4; ++row)
                    for (int column = 0; column < columns; ++column)
                    {
                        float acc = 0.0f;
                        for (int kb = 0; kb < kBlocks; ++kb)
                            acc += multi_scale::scalarReferenceBlock(view, activation[row][kb], chunk, kb, column);
                        expected[row][column] = acc;
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
                    multi_scale::dispatch<kLanes, Rows, Ops>(view, input, output, chunk, 0, kBlocks, false,
                                                             kLanes == 4 ? columns : 64);
                    for (std::size_t row = 0; row < Rows; ++row)
                        for (int column = 0; column < columns; ++column)
                            ASSERT_EQ(std::bit_cast<std::uint32_t>(actual[row][column]),
                                      std::bit_cast<std::uint32_t>(expected[row][column]))
                                << "rows=" << Rows << " row=" << row << " column=" << chunk * 64 + column;
                };
                run.template operator()<1>();
                run.template operator()<2>();
                run.template operator()<3>();
                run.template operator()<4>();
            }
        }
    }
} // namespace llaminar2::test
