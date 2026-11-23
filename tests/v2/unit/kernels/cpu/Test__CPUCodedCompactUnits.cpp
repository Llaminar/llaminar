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
 *   integer dot and per-block output bytes, including coded Q4_K ragged K.
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

#include <array>
#include <bit>
#include <cstdint>
#include <cmath>
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
        EXPECT_EQ(preparedEncodingForSource(native_quantised_formats::Q5_1, true), CPUNativeQuantisedEncoding::ExpandedInt8);
        EXPECT_EQ(preparedEncodingForSource(native_quantised_formats::Q5_K, true), CPUNativeQuantisedEncoding::ExpandedInt8);
        EXPECT_EQ(preparedEncodingForSource(native_quantised_formats::IQ3_S, true), CPUNativeQuantisedEncoding::ExpandedInt8);
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

    /** Both CPU arithmetic contracts stay byte-identical to the prior expanded CPU publication. */
    TEST(CPUCodedCompactUnits, Q5KIQ3SPreserveExpandedOutputs)
    {
        constexpr int n = 67, k = 544, blocks = k / 32;
        constexpr int partitions = MoEProjectionNumericalContract::orderedKPartitionsForWidth(k);
        std::mt19937 rng(1493);
        std::array<std::array<Q8_1Block, blocks>, 4> activation{};
        for (auto &row : activation)
            for (auto &a : row)
            {
                a.d = static_cast<uint16_t>(0x2000 + rng() % 0x1800);
                for (auto &q : a.qs)
                {
                    q = static_cast<int8_t>(int(rng() % 255) - 127);
                    a.sum_qs += q;
                }
            }
        for (const auto &format : compactSources())
        {
            if (format.codebook != 7 && format.codebook != 11)
                continue;
            SCOPED_TRACE(format.name);
            const auto source = format.make({n, k}, 37);
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
            if (packed.codebook_id == 19)
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
    TEST(CPUCompensationFreeUnits, PreparedBytesMatchCompensationFreeLayout)
    {
        for (const auto &source : compensationFreeSources())
        {
            const auto tensor = source.make({67, 512}, 9);
            SCOPED_TRACE(source.name);
            CPUNativeQuantisedPackedWeights packed;
            ASSERT_TRUE(packWeightsCPUNativeQuantised(tensor.get(), packed));
            EXPECT_FALSE(packed.usesInlineCompensation());
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
                ASSERT_EQ(previous, next);
#if defined(__AVX2__)
                const auto dot = dotUnsignedWeightsAVX2(_mm256_setzero_si256(), _mm256_set1_epi8(static_cast<char>(w + 128)),
                    _mm256_set1_epi8(static_cast<char>(a)));
                ASSERT_EQ(_mm256_extract_epi32(dot, 0) - 4 * 128 * a, 4 * previous);
#endif
            }
    }

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
