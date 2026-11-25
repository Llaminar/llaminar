/**
 * @file Test__CPUX86CodedLeaves.cpp
 * @brief Dedicated x86 coded decode leaves against the generic traversals they replace.
 *
 * The x86 decode round replaced generic traversals with dedicated AVX2 and
 * AVX-512 leaves that keep each column's exact INT32 dot and per-policy FP32
 * block program but change the physical walk:
 *
 * - ordinary Q5_0/Q5_1 (q5::dispatch): one unrolled pass per register tile,
 *   with 16-block K cache tiles for the AVX2 one-row tile;
 * - Q5_K and IQ3_S (multi_scale::x86::singleScaleRows): unrolled groups,
 *   superblock-addressed units and headers, contiguous and ordered K trees;
 * - AVX-512 Q4_K and IQ4_XS (multi_scale::x86::nibbleSingleScaleRows): per
 *   superblock bases, routed from the one- to four-row nibble chunks and the
 *   ordered GPU-aligned trees;
 * - AVX-512 packed prefill panels: four rows per decoded panel vector.
 *
 * Each case builds sealed prepared publications directly (random payloads and
 * codes, controlled scale words) and compares the production entry bitwise
 * with the generic traversal on the same bytes: the original per-strip Q5
 * loop, multi_scale::rows(), the templated generic nibble chunk and ordered
 * kernels, or independent serial M=1 rows. Every column carries at most one
 * non-finite scale source (a NaN with its own payload, a signalling NaN, or an
 * infinity), so its result does not depend on the order in which commutative
 * operations meet. The only tolerated difference is the sign of the default
 * NaN generated from infinities, which IEEE 754 leaves unspecified (x86
 * hardware always produces 0xffc00000; translators may not).
 *
 * The ordered partition fold has its own regression: two superblocks whose
 * scales are NaNs with different payloads, one per partition, must publish the
 * earlier partition's NaN for every leaf, row tile, lane width and policy
 * (the generic multi_scale::rows() fold of the other compact codebooks is not
 * pinned and is not covered here).
 * The x86 leaves fold through addOrderedAVX2()/addOrderedAVX512() because a
 * plain vector `+` lets the compiler swap VADDPS operands, which GCC 14 did in
 * three of the four AVX2 one-row ordered instantiations.
 */
#include <gtest/gtest.h>

#include "kernels/cpu/gemm/CPUNativeQuantisedGemv.h"

#include <omp.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#if defined(__AVX2__)
namespace llaminar2::test
{
    namespace
    {
        using namespace cpu::native_quantised;
        using Policy = CPUProjectionNumericalPolicy;
        using Encoding = CPUNativeQuantisedEncoding;
        using multi_scale::Reduction;
        constexpr Policy kPolicies[] = {Policy::BackendNative, Policy::GPUAlignedExpert};

        /** @brief Physical identity of one prepared format, as the packer publishes it. */
        struct Layout
        {
            const char *name;
            Encoding encoding;
            std::uint8_t codebook;
            bool asymmetric;
            bool superblock;
            int payload_bytes;
        };
        constexpr Layout kQ50{"Q5_0", Encoding::Q5NativeSingleScale, 6, false, false, 20};
        constexpr Layout kQ51{"Q5_1", Encoding::Q5NativeSingleScale, 7, true, false, 20};
        constexpr Layout kQ5K{"Q5_K", Encoding::CompactMultiScale, 7, true, true, 20};
        constexpr Layout kIQ3S{"IQ3_S", Encoding::CompactMultiScale, 11, false, true, 13};
        constexpr Layout kQ4K{"Q4_K", Encoding::Q4KSuperblockCoded, 5, true, true, 16};
        constexpr Layout kIQ4XS{"IQ4_XS", Encoding::NibbleLUTSourceScale, 4, false, true, 16};

        /** @brief Scale-word content of one prepared matrix. */
        enum class Scales
        {
            Finite,        ///< Every scale word finite.
            OneNonFinite,  ///< At most one non-finite scale source per column.
            TwoNaNSources, ///< Superblock 0 and 1 hold differently signed, different-payload NaNs.
        };

        /** @return A finite binary16 scale of either sign, between 2^-7 and 2^-1. */
        std::uint16_t finiteHalf(std::mt19937 &rng)
        {
            return static_cast<std::uint16_t>((rng() & 0x8000u) | (0x2000u + rng() % 0x1800u));
        }

        /**
         * @return The non-finite binary16 assigned to @p column, or zero for a finite column.
         *
         * Quiet NaNs of both signs and a signalling NaN carry column-specific
         * payloads; infinities of both signs generate only the default NaN.
         */
        std::uint16_t nonFiniteHalf(int column)
        {
            const auto payload = static_cast<std::uint16_t>(1 + (column * 37) % 0x1FF);
            switch (column % 6)
            {
            case 1: return static_cast<std::uint16_t>(0x7E00u | payload);
            case 2: return 0x7C00u;
            case 3: return 0xFC00u;
            case 4: return static_cast<std::uint16_t>(0xFE00u | payload);
            case 5: return static_cast<std::uint16_t>(0x7C00u | payload);
            default: return 0;
            }
        }

        /** @brief Write one little-endian binary16 word. */
        void putHalf(std::uint8_t *where, std::uint16_t value) { std::memcpy(where, &value, sizeof(value)); }

        /**
         * @brief Build and seal one prepared publication of random payloads and codes.
         * @param packed Receives the matrix; it owns its bytes.
         * @param layout Physical format.
         * @param n Columns, a multiple of 64.
         * @param blocks Logical K blocks; ragged final superblocks are allowed.
         * @param policy Arithmetic policy.
         * @param scales Scale-word content.
         * @param seed Payload and scale seed.
         */
        void makeWeights(CPUNativeQuantisedPackedWeights &packed, const Layout &layout, int n, int blocks,
                         Policy policy, Scales scales, std::uint32_t seed)
        {
            ASSERT_EQ(n % 64, 0);
            packed.N = n;
            packed.N_padded = n;
            packed.K = blocks * 32;
            packed.blocks_per_row = blocks;
            packed.encoding = layout.encoding;
            packed.codebook_id = layout.codebook;
            packed.is_asymmetric = layout.asymmetric;
            packed.is_superblock = layout.superblock;
            packed.numerical_policy = policy;
            packed.payload_bytes = layout.payload_bytes;
            packed.data_stride = preparedDataStride(layout.encoding, layout.codebook);
            packed.interleaved_block_stride =
                preparedInterleavedBlockStride(layout.encoding, layout.asymmetric, layout.codebook);
            packed.native_interleaved.resize(packed.preparedBytes());
            std::mt19937 rng(seed);
            {
                const auto bytes = packed.native_interleaved.beginRewrite();
                for (auto &byte : bytes)
                    byte = static_cast<std::uint8_t>(rng());
                const int chunks = n / 64;
                if (packed.superblockCoded())
                {
                    const auto field = preparedSuperblockHeaderBytes(layout.encoding, layout.codebook) / 64;
                    const int superblocks = (blocks + 7) / 8;
                    for (int chunk = 0; chunk < chunks; ++chunk)
                        for (int superblock = 0; superblock < superblocks; ++superblock)
                        {
                            auto *header = bytes.data() + packed.superblockHeaderOffset(chunk, superblock * 8);
                            for (int column = 0; column < 64; ++column)
                            {
                                const int global = chunk * 64 + column;
                                std::uint16_t d = finiteHalf(rng), dmin = finiteHalf(rng);
                                if (scales == Scales::TwoNaNSources && superblock < 2)
                                    d = superblock == 0 ? static_cast<std::uint16_t>(0x7E00u | (1 + global % 0x1FF))
                                                        : static_cast<std::uint16_t>(0xFE00u | (0x1FF - global % 0x1FF));
                                if (scales == Scales::OneNonFinite && superblock == (global / 6) % superblocks &&
                                    nonFiniteHalf(global))
                                {
                                    // An affine format's sixth kind poisons its minimum instead.
                                    if (field == 4 && global % 6 == 5)
                                        dmin = 0x7C00u;
                                    else
                                        d = nonFiniteHalf(global);
                                }
                                putHalf(header + column * field, d);
                                if (field == 4)
                                    putHalf(header + column * field + 2, dmin);
                            }
                        }
                }
                else
                {
                    // Per-block FP16 d (and Q5_1 m) planes follow the payload.
                    for (int chunk = 0; chunk < chunks; ++chunk)
                        for (int kb = 0; kb < blocks; ++kb)
                        {
                            auto *plane = bytes.data() + packed.blockOffset(chunk, kb) + packed.data_stride;
                            for (int column = 0; column < 64; ++column)
                            {
                                const int global = chunk * 64 + column;
                                std::uint16_t d = finiteHalf(rng), m = finiteHalf(rng);
                                if (scales == Scales::OneNonFinite && kb == (global * 5) % blocks &&
                                    nonFiniteHalf(global))
                                {
                                    if (layout.asymmetric && global % 6 == 5)
                                        m = 0x7C00u;
                                    else
                                        d = nonFiniteHalf(global);
                                }
                                putHalf(plane + 2 * column, d);
                                if (layout.asymmetric)
                                    putHalf(plane + 128 + 2 * column, m);
                            }
                        }
                }
            }
            packed.sealPublication();
            ASSERT_TRUE(packed.publicationSealed());
        }

        /** @return @p rows x @p blocks signed Q8_1 blocks with exact sums and finite scales, including -128. */
        std::vector<Q8_1Block> makeActivations(int rows, int blocks, std::uint32_t seed)
        {
            std::mt19937 rng(seed);
            std::vector<Q8_1Block> activation(std::size_t(rows) * blocks);
            for (auto &block : activation)
            {
                block.d = finiteHalf(rng);
                int sum = 0;
                const bool extreme = rng() % 7 == 0;
                for (auto &q : block.qs)
                {
                    q = static_cast<std::int8_t>(extreme ? -128 : int(rng() % 256) - 128);
                    sum += q;
                }
                block.sum_qs = static_cast<std::int16_t>(sum);
            }
            return activation;
        }

        /** @return @p bits as 0x-prefixed hexadecimal (AssertionResult ignores stream manipulators). */
        std::string hexBits(std::uint32_t bits)
        {
            char text[16];
            std::snprintf(text, sizeof text, "0x%08x", bits);
            return text;
        }

        /** @return Whether two outputs agree: identical bits, or both the (sign-free) default NaN. */
        bool sameOutput(std::uint32_t actual, std::uint32_t expected)
        {
            const auto default_nan = [](std::uint32_t bits) { return (bits & 0x7FFFFFFFu) == 0x7FC00000u; };
            return actual == expected || (default_nan(actual) && default_nan(expected));
        }

        /** @brief Compare @p rows 64-float rows, reporting the first mismatch. */
        ::testing::AssertionResult sameRows(const std::vector<std::array<float, 64>> &actual,
                                            const std::vector<std::array<float, 64>> &expected)
        {
            for (std::size_t row = 0; row < actual.size(); ++row)
                for (int column = 0; column < 64; ++column)
                {
                    const auto a = std::bit_cast<std::uint32_t>(actual[row][column]);
                    const auto e = std::bit_cast<std::uint32_t>(expected[row][column]);
                    if (!sameOutput(a, e))
                        return ::testing::AssertionFailure() << "row=" << row << " column=" << column
                                                             << " actual=" << hexBits(a) << " expected=" << hexBits(e);
                }
            return ::testing::AssertionSuccess();
        }

        /** @return Whether @p bits is any NaN. */
        bool isNaN(std::uint32_t bits) { return (bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu); }

        /** @brief Row pointers into persistent activations and 64-float outputs. */
        template <std::size_t Rows> struct Tile
        {
            std::array<const Q8_1Block *, Rows> input{};
            std::array<float *, Rows> output{};
        };

        /** @return A tile over rows [0, Rows) of @p activation and @p outputs. */
        template <std::size_t Rows>
        Tile<Rows> tile(const std::vector<Q8_1Block> &activation, int blocks, std::vector<std::array<float, 64>> &outputs)
        {
            Tile<Rows> result;
            for (std::size_t row = 0; row < Rows; ++row)
            {
                result.input[row] = activation.data() + row * std::size_t(blocks);
                result.output[row] = outputs[row].data();
            }
            return result;
        }

        /** @return @p rows output rows holding @p seed-derived finite values (an accumulated K prefix). */
        std::vector<std::array<float, 64>> initialRows(std::size_t rows, std::uint32_t seed)
        {
            std::mt19937 rng(seed);
            std::vector<std::array<float, 64>> result(rows);
            for (auto &row : result)
                for (auto &value : row)
                    value = static_cast<float>(int(rng() % 2001) - 1000) / 64.0f;
            return result;
        }

        // ---------------------------------------------------------------------
        // Ordinary Q5: the per-strip traversal of 4d5a0c5e7 as the oracle.
        // ---------------------------------------------------------------------

        /**
         * @brief The pre-round Q5 projection: one K walk per strip with a rolled four-group loop.
         * @details Uses the retained q5::decode8()/decode16(), q5::dot() and
         * q5::contribute() building blocks in their original order.
         */
        template <int Lanes, bool SumPairs, std::size_t Rows>
        void referenceQ5(const CPUNativeQuantisedPackedWeights &packed, const Tile<Rows> &t, int chunk, int first,
                         int last, bool accumulate)
        {
            using Ops = multi_scale::SIMD<Lanes>;
            using I = typename Ops::I;
            using F = typename Ops::F;
            for (int column = 0; column < 64; column += Lanes)
            {
                F partial[Rows]{};
                if (accumulate)
                    for (std::size_t row = 0; row < Rows; ++row)
                        partial[row] = multi_scale::load<F>(t.output[row] + column);
                for (int kb = first; kb < last; ++kb)
                {
                    const auto *unit = packed.blockUnit(chunk, kb);
                    I dots[Rows]{};
                    for (int group = 0; group < 4; ++group)
                    {
                        const auto [low, high] = [&] {
                            if constexpr (Lanes == 8)
                                return q5::decode8(unit, column, group);
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
                            else
                                return q5::decode16(unit, column, group);
#endif
                        }();
                        for (std::size_t row = 0; row < Rows; ++row)
                        {
                            dots[row] = q5::dot<Lanes>(dots[row], low, t.input[row][kb], group);
                            dots[row] = q5::dot<Lanes>(dots[row], high, t.input[row][kb], group + 4);
                        }
                    }
                    const auto *d = reinterpret_cast<const std::uint16_t *>(unit + Q5NativePayload::bytes) + column;
                    for (std::size_t row = 0; row < Rows; ++row)
                        partial[row] = packed.is_asymmetric
                            ? q5::contribute<Lanes, true>(partial[row], dots[row], d, d + 64, t.input[row][kb],
                                                         packed.numerical_policy)
                            : q5::contribute<Lanes, false>(partial[row], dots[row], d, nullptr, t.input[row][kb],
                                                          packed.numerical_policy);
                }
                if constexpr (SumPairs)
                    for (std::size_t row = 0; row < Rows; row += 2)
                    {
                        const F sum = partial[row] + partial[row + 1];
                        std::memcpy(t.output[row] + column, &sum, sizeof(F));
                    }
                else
                    for (std::size_t row = 0; row < Rows; ++row)
                        std::memcpy(t.output[row] + column, &partial[row], sizeof(F));
            }
        }

        /** @brief Compare the production Q5 leaf with the per-strip oracle for one tile shape. */
        template <int Lanes, bool SumPairs, std::size_t Rows>
        void expectQ5Tile(const CPUNativeQuantisedPackedWeights &packed, const std::vector<Q8_1Block> &activation,
                          int blocks)
        {
            SCOPED_TRACE("lanes=" + std::to_string(Lanes) + " rows=" + std::to_string(Rows) +
                         " sum_pairs=" + std::to_string(SumPairs));
            struct Range { int first, last; bool accumulate; };
            // Full K (several 16-block AVX2 tiles plus a remainder), an
            // accumulated interior range, an exact two-tile range and an empty one.
            for (const Range range : {Range{0, blocks, false}, Range{3, blocks - 2, true}, Range{0, 32, false},
                                      Range{7, 7, true}})
                for (int chunk = 0; chunk < packed.N_padded / 64; ++chunk)
                {
                    SCOPED_TRACE("first=" + std::to_string(range.first) + " last=" + std::to_string(range.last) +
                                 " chunk=" + std::to_string(chunk));
                    auto actual = initialRows(Rows, 91u + chunk), expected = actual;
                    q5::dispatch<Lanes, SumPairs>(packed, tile<Rows>(activation, blocks, actual).input,
                                                  tile<Rows>(activation, blocks, actual).output, chunk, range.first,
                                                  range.last, range.accumulate);
                    referenceQ5<Lanes, SumPairs>(packed, tile<Rows>(activation, blocks, expected), chunk, range.first,
                                                 range.last, range.accumulate);
                    ASSERT_TRUE(sameRows(actual, expected));
                }
        }

        // ---------------------------------------------------------------------
        // Q5_K and IQ3_S: multi_scale::rows() as the oracle.
        // ---------------------------------------------------------------------

        /** @brief Compare dispatch()/dispatchOrdered() with the generic rows() traversal for one tile shape. */
        template <int Codebook, int Lanes, std::size_t Rows>
        void expectSingleScaleTile(const CPUNativeQuantisedPackedWeights &packed,
                                   const std::vector<Q8_1Block> &activation, int blocks)
        {
            SCOPED_TRACE("lanes=" + std::to_string(Lanes) + " rows=" + std::to_string(Rows));
            for (int chunk = 0; chunk < packed.N_padded / 64; ++chunk)
            {
                // Contiguous: full K and an accumulated range that starts and
                // ends inside source superblocks.
                for (const auto [first, last, accumulate] : {std::tuple{0, blocks, false}, std::tuple{3, blocks - 3, true}})
                {
                    auto actual = initialRows(Rows, 17u + chunk), expected = actual;
                    multi_scale::dispatch<Lanes, Rows>(packed, tile<Rows>(activation, blocks, actual).input,
                                                       tile<Rows>(activation, blocks, actual).output, chunk, first,
                                                       last, accumulate);
                    multi_scale::rows<Codebook, Lanes, Rows, Reduction::Contiguous>(
                        packed, tile<Rows>(activation, blocks, expected).input,
                        tile<Rows>(activation, blocks, expected).output, chunk, first, last, accumulate);
                    ASSERT_TRUE(sameRows(actual, expected)) << "contiguous first=" << first << " chunk=" << chunk;
                }
                // Ordered: superblock-aligned, misaligned, and with empty trailing partitions.
                for (const auto [partitions, per_partition] : {std::pair{(blocks + 7) / 8, 8}, std::pair{8, 5},
                                                               std::pair{16, 3}})
                {
                    auto actual = initialRows(Rows, 29u), expected = initialRows(Rows, 31u);
                    multi_scale::dispatchOrdered<Lanes, Rows>(packed, tile<Rows>(activation, blocks, actual).input,
                                                              tile<Rows>(activation, blocks, actual).output, chunk,
                                                              blocks, partitions, per_partition);
                    multi_scale::rows<Codebook, Lanes, Rows, Reduction::OrderedPartitions>(
                        packed, tile<Rows>(activation, blocks, expected).input,
                        tile<Rows>(activation, blocks, expected).output, chunk, 0, blocks, false, partitions,
                        per_partition);
                    ASSERT_TRUE(sameRows(actual, expected)) << "ordered partitions=" << partitions << " chunk=" << chunk;
                }
            }
        }

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        // ---------------------------------------------------------------------
        // AVX-512 Q4_K and IQ4_XS: the templated generic nibble kernels as the oracle.
        // ---------------------------------------------------------------------

        /** @brief One generic single-row nibble chunk, bypassing the coded-nibble routing. */
        template <NibbleDecodeKind Kind, bool Asymmetric>
        void genericNibbleRow(const CPUNativeQuantisedPackedWeights &packed, const Q8_1Block *activation,
                              float *output, int chunk, int first, int last, __m512i lut, bool accumulate)
        {
            if (packed.numerical_policy == Policy::GPUAlignedExpert)
                gemv_native_quantised_avx512_chunk_native<Kind, Policy::GPUAlignedExpert, Asymmetric>(
                    packed, activation, output, chunk, first, last, lut, accumulate);
            else
                gemv_native_quantised_avx512_chunk_native<Kind, Policy::BackendNative, Asymmetric>(
                    packed, activation, output, chunk, first, last, lut, accumulate);
        }

        /**
         * @brief Compare every routed AVX-512 coded-nibble entry with independent generic rows.
         * @tparam Kind Generic decoder of the format.
         * @tparam Asymmetric Whether the format carries a minimum.
         */
        template <NibbleDecodeKind Kind, bool Asymmetric>
        void expectCodedNibbleEntries(const CPUNativeQuantisedPackedWeights &packed,
                                      const std::vector<Q8_1Block> &activation, int blocks)
        {
            ASSERT_EQ(nibbleDecodeKind(packed.codebook_id), Kind);
            const __m512i lut = build_decode_lut(packed.codebook_id);
            const auto row = [&](int r) { return activation.data() + std::size_t(r) * blocks; };
            for (int chunk = 0; chunk < packed.N_padded / 64; ++chunk)
            {
                for (const auto [first, last, accumulate] : {std::tuple{0, blocks, false}, std::tuple{3, blocks - 3, true}})
                {
                    SCOPED_TRACE("contiguous first=" + std::to_string(first) + " chunk=" + std::to_string(chunk));
                    auto expected = initialRows(4, 5u + chunk);
                    for (int r = 0; r < 4; ++r)
                        genericNibbleRow<Kind, Asymmetric>(packed, row(r), expected[r].data(), chunk, first, last, lut,
                                                           accumulate);
                    for (int rows = 1; rows <= 4; ++rows)
                    {
                        auto actual = initialRows(4, 5u + chunk);
                        auto *c = actual.data();
                        if (rows == 1)
                            gemv_native_quantised_avx512_chunk_native(packed, row(0), c[0].data(), chunk, first, last,
                                                                      lut, accumulate);
                        else if (rows == 2)
                            gemm_2row_native_chunk(packed, row(0), row(1), c[0].data(), c[1].data(), chunk, first,
                                                   last, lut, accumulate);
                        else if (rows == 3)
                            gemm_3row_native_2z_chunk(packed, row(0), row(1), row(2), c[0].data(), c[1].data(),
                                                      c[2].data(), chunk, first, last, lut, accumulate);
                        else
                            gemm_4row_native_1z_chunk(packed, row(0), row(1), row(2), row(3), c[0].data(),
                                                      c[1].data(), c[2].data(), c[3].data(), chunk, first, last, lut,
                                                      accumulate);
                        actual.resize(rows);
                        auto reference = expected;
                        reference.resize(rows);
                        ASSERT_TRUE(sameRows(actual, reference)) << "rows=" << rows;
                    }
                }
                // The ordered trees are the GPU-aligned program for either prepared policy.
                for (const auto [partitions, per_partition] : {std::pair{(blocks + 7) / 8, 8}, std::pair{8, 5},
                                                               std::pair{16, 3}})
                {
                    SCOPED_TRACE("ordered partitions=" + std::to_string(partitions) + " chunk=" + std::to_string(chunk));
                    auto expected = initialRows(4, 3u);
                    for (int r = 0; r < 4; ++r)
                        gemv_native_avx512_ordered_partitions<Kind, Asymmetric>(packed, row(r), expected[r].data(),
                                                                                chunk, blocks, partitions,
                                                                                per_partition, lut);
                    for (int rows = 1; rows <= 4; ++rows)
                    {
                        auto actual = initialRows(4, 7u);
                        auto *c = actual.data();
                        if (rows == 1)
                            dispatch_native_avx512_ordered_partitions(packed, row(0), c[0].data(), chunk, blocks,
                                                                      partitions, per_partition, lut);
                        else if (rows == 2)
                            dispatch_2row_native_avx512_ordered_partitions(packed, row(0), row(1), c[0].data(),
                                                                           c[1].data(), chunk, blocks, partitions,
                                                                           per_partition, lut);
                        else if (rows == 3)
                            dispatch_3row_native_avx512_ordered_partitions(packed, row(0), row(1), row(2), c[0].data(),
                                                                           c[1].data(), c[2].data(), chunk, blocks,
                                                                           partitions, per_partition, lut);
                        else
                        {
                            const auto t = tile<4>(activation, blocks, actual);
                            if (packed.encoding == Encoding::Q4KSuperblockCoded)
                                multi_scale::x86::nibbleSingleScaleRows<5, 4, Reduction::OrderedPartitions>(
                                    packed, t.input, t.output, chunk, 0, blocks, false, lut,
                                    nibbleWeightBias(packed.codebook_id), partitions, per_partition);
                            else
                                multi_scale::x86::nibbleSingleScaleRows<4, 4, Reduction::OrderedPartitions>(
                                    packed, t.input, t.output, chunk, 0, blocks, false, lut,
                                    nibbleWeightBias(packed.codebook_id), partitions, per_partition);
                        }
                        actual.resize(rows);
                        auto reference = expected;
                        reference.resize(rows);
                        ASSERT_TRUE(sameRows(actual, reference)) << "rows=" << rows;
                    }
                }
            }
        }
#endif

        // ---------------------------------------------------------------------
        // Ordered fold NaN precedence.
        // ---------------------------------------------------------------------

        /**
         * @brief Require the ordered tree to publish its first partition's NaN.
         * @param earlier The first partition alone, through the same arithmetic program.
         * @param later The second partition alone.
         * @param tree The complete ordered tree.
         */
        ::testing::AssertionResult earlierNaNWins(const std::vector<std::array<float, 64>> &earlier,
                                                  const std::vector<std::array<float, 64>> &later,
                                                  const std::vector<std::array<float, 64>> &tree)
        {
            for (std::size_t row = 0; row < tree.size(); ++row)
                for (int column = 0; column < 64; ++column)
                {
                    const auto first = std::bit_cast<std::uint32_t>(earlier[row][column]);
                    const auto second = std::bit_cast<std::uint32_t>(later[row][column]);
                    const auto folded = std::bit_cast<std::uint32_t>(tree[row][column]);
                    if (!isNaN(first) || !isNaN(second) || first == second)
                        return ::testing::AssertionFailure() << "fixture lost its distinct NaNs at row=" << row
                                                             << " column=" << column;
                    if (folded != first)
                        return ::testing::AssertionFailure() << "row=" << row << " column=" << column
                                                             << " folded=" << hexBits(folded) << " earlier="
                                                             << hexBits(first) << " later=" << hexBits(second);
                }
            return ::testing::AssertionSuccess();
        }

        /** @brief The ordered Q5_K/IQ3_S leaves keep the earlier partition's NaN. */
        template <int Lanes, std::size_t Rows>
        void expectCompactOrderedFold(const CPUNativeQuantisedPackedWeights &packed,
                                      const std::vector<Q8_1Block> &activation)
        {
            SCOPED_TRACE("lanes=" + std::to_string(Lanes) + " rows=" + std::to_string(Rows));
            constexpr int blocks = 16;
            std::vector<std::array<float, 64>> earlier(Rows), later(Rows);
            // A partition starts at positive zero: one contiguous range is the same program.
            multi_scale::dispatch<Lanes, Rows>(packed, tile<Rows>(activation, blocks, earlier).input,
                                               tile<Rows>(activation, blocks, earlier).output, 0, 0, 8, false);
            multi_scale::dispatch<Lanes, Rows>(packed, tile<Rows>(activation, blocks, later).input,
                                               tile<Rows>(activation, blocks, later).output, 0, 8, 16, false);
            for (const auto [partitions, per_partition] : {std::pair{2, 8}, std::pair{4, 4}})
            {
                std::vector<std::array<float, 64>> tree(Rows);
                multi_scale::dispatchOrdered<Lanes, Rows>(packed, tile<Rows>(activation, blocks, tree).input,
                                                          tile<Rows>(activation, blocks, tree).output, 0, blocks,
                                                          partitions, per_partition);
                ASSERT_TRUE(earlierNaNWins(earlier, later, tree)) << "partitions=" << partitions;
            }
        }
    } // namespace

    /** @test Ordered x86 trees publish the earlier partition's NaN for every leaf, tile, width and policy. */
    TEST(CPUX86CodedLeaves, OrderedFoldKeepsEarlierPartitionNaN)
    {
        constexpr int n = 64, blocks = 16;
        const auto activation = makeActivations(4, blocks, 404u);
        for (const Layout &layout : {kQ5K, kIQ3S})
            for (const Policy policy : kPolicies)
            {
                SCOPED_TRACE(std::string(layout.name) + " policy=" + std::to_string(int(policy)));
                CPUNativeQuantisedPackedWeights packed;
                makeWeights(packed, layout, n, blocks, policy, Scales::TwoNaNSources, 61u + layout.codebook);
                // Only one-row AVX2 tiles use the leaf; wider AVX2 tiles keep rows().
                expectCompactOrderedFold<8, 1>(packed, activation);
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
                expectCompactOrderedFold<16, 1>(packed, activation);
                expectCompactOrderedFold<16, 2>(packed, activation);
                expectCompactOrderedFold<16, 3>(packed, activation);
                expectCompactOrderedFold<16, 4>(packed, activation);
#endif
            }
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        for (const Layout &layout : {kQ4K, kIQ4XS})
            for (const Policy policy : kPolicies)
            {
                SCOPED_TRACE(std::string(layout.name) + " policy=" + std::to_string(int(policy)));
                CPUNativeQuantisedPackedWeights packed;
                makeWeights(packed, layout, n, blocks, policy, Scales::TwoNaNSources, 71u + layout.codebook);
                const __m512i lut = build_decode_lut(packed.codebook_id);
                const int bias = nibbleWeightBias(packed.codebook_id);
                const auto row = [&](int r) { return activation.data() + std::size_t(r) * blocks; };
                // Single-partition ordered trees over each half: the GPU-aligned
                // program the full tree applies to each partition.
                const auto leaf = [&]<std::size_t Rows>(std::vector<std::array<float, 64>> &out, int first, int last,
                                                       int partitions, int per_partition) {
                    const auto t = tile<Rows>(activation, blocks, out);
                    if (layout.codebook == 5)
                        multi_scale::x86::nibbleSingleScaleRows<5, Rows, Reduction::OrderedPartitions>(
                            packed, t.input, t.output, 0, first, last, false, lut, bias, partitions, per_partition);
                    else
                        multi_scale::x86::nibbleSingleScaleRows<4, Rows, Reduction::OrderedPartitions>(
                            packed, t.input, t.output, 0, first, last, false, lut, bias, partitions, per_partition);
                };
                std::vector<std::array<float, 64>> earlier(4), later(4);
                leaf.template operator()<4>(earlier, 0, 8, 1, 8);
                leaf.template operator()<4>(later, 8, 16, 1, 8);
                for (const auto [partitions, per_partition] : {std::pair{2, 8}, std::pair{4, 4}})
                {
                    SCOPED_TRACE("partitions=" + std::to_string(partitions));
                    for (int rows = 1; rows <= 4; ++rows)
                    {
                        std::vector<std::array<float, 64>> tree(4);
                        auto *c = tree.data();
                        if (rows == 1)
                            dispatch_native_avx512_ordered_partitions(packed, row(0), c[0].data(), 0, blocks,
                                                                      partitions, per_partition, lut);
                        else if (rows == 2)
                            dispatch_2row_native_avx512_ordered_partitions(packed, row(0), row(1), c[0].data(),
                                                                           c[1].data(), 0, blocks, partitions,
                                                                           per_partition, lut);
                        else if (rows == 3)
                            dispatch_3row_native_avx512_ordered_partitions(packed, row(0), row(1), row(2), c[0].data(),
                                                                           c[1].data(), c[2].data(), 0, blocks,
                                                                           partitions, per_partition, lut);
                        else
                            leaf.template operator()<4>(tree, 0, blocks, partitions, per_partition);
                        tree.resize(rows);
                        auto first = earlier, second = later;
                        first.resize(rows);
                        second.resize(rows);
                        ASSERT_TRUE(earlierNaNWins(first, second, tree)) << "rows=" << rows;
                    }
                }
            }
#endif
    }

    /** @test The unrolled Q5_0/Q5_1 leaf (with AVX2 K cache tiles) equals the per-strip traversal. */
    TEST(CPUX86CodedLeaves, Q5LeafEqualsPerStripTraversal)
    {
        constexpr int n = 128, blocks = 37;
        const auto activation = makeActivations(4, blocks, 505u);
        for (const Layout &layout : {kQ50, kQ51})
            for (const Policy policy : kPolicies)
                for (const Scales scales : {Scales::Finite, Scales::OneNonFinite})
                {
                    SCOPED_TRACE(std::string(layout.name) + " policy=" + std::to_string(int(policy)) +
                                 " non_finite=" + std::to_string(scales == Scales::OneNonFinite));
                    CPUNativeQuantisedPackedWeights packed;
                    makeWeights(packed, layout, n, blocks, policy, scales, 81u + layout.codebook);
                    expectQ5Tile<8, false, 1>(packed, activation, blocks);
                    expectQ5Tile<8, false, 2>(packed, activation, blocks);
                    expectQ5Tile<8, true, 2>(packed, activation, blocks);
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
                    expectQ5Tile<16, false, 1>(packed, activation, blocks);
                    expectQ5Tile<16, false, 2>(packed, activation, blocks);
                    expectQ5Tile<16, true, 2>(packed, activation, blocks);
                    expectQ5Tile<16, false, 3>(packed, activation, blocks);
                    expectQ5Tile<16, true, 4>(packed, activation, blocks);
#endif
                }
    }

    /** @test Q5_K and IQ3_S leaves equal the generic rows() traversal, contiguous and ordered. */
    TEST(CPUX86CodedLeaves, Q5KIQ3SLeavesEqualGenericRows)
    {
        constexpr int n = 128, blocks = 37;
        const auto activation = makeActivations(4, blocks, 606u);
        for (const Layout &layout : {kQ5K, kIQ3S})
            for (const Policy policy : kPolicies)
                for (const Scales scales : {Scales::Finite, Scales::OneNonFinite})
                {
                    SCOPED_TRACE(std::string(layout.name) + " policy=" + std::to_string(int(policy)) +
                                 " non_finite=" + std::to_string(scales == Scales::OneNonFinite));
                    CPUNativeQuantisedPackedWeights packed;
                    makeWeights(packed, layout, n, blocks, policy, scales, 91u + layout.codebook);
                    const auto run = [&]<int Codebook>() {
                        expectSingleScaleTile<Codebook, 8, 1>(packed, activation, blocks);
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
                        expectSingleScaleTile<Codebook, 16, 1>(packed, activation, blocks);
                        expectSingleScaleTile<Codebook, 16, 2>(packed, activation, blocks);
                        expectSingleScaleTile<Codebook, 16, 3>(packed, activation, blocks);
                        expectSingleScaleTile<Codebook, 16, 4>(packed, activation, blocks);
#endif
                    };
                    if (layout.codebook == 7)
                        run.template operator()<7>();
                    else
                        run.template operator()<11>();
                }
    }

    /** @test The Q5_K/IQ3_S leaves own complete 64-column chunks and reject a partial one before writing. */
    TEST(CPUX86CodedLeaves, SingleScaleLeavesRejectPartialChunks)
    {
        constexpr int n = 64, blocks = 9;
        const auto activation = makeActivations(1, blocks, 707u);
        for (const Layout &layout : {kQ5K, kIQ3S})
        {
            SCOPED_TRACE(layout.name);
            CPUNativeQuantisedPackedWeights packed;
            makeWeights(packed, layout, n, blocks, Policy::GPUAlignedExpert, Scales::Finite, 13u);
            const auto expectRejected = [&]<int Lanes>() {
                std::vector<std::array<float, 64>> output(1);
                output[0].fill(-7.5f);
                const auto untouched = output;
                const auto t = tile<1>(activation, blocks, output);
                EXPECT_THROW(multi_scale::dispatch<Lanes>(packed, t.input, t.output, 0, 0, blocks, false, 63),
                             std::invalid_argument);
                EXPECT_THROW(multi_scale::dispatchOrdered<Lanes>(packed, t.input, t.output, 0, blocks, 3, 3, 63),
                             std::invalid_argument);
                EXPECT_TRUE(std::memcmp(output.data(), untouched.data(), sizeof(float) * 64) == 0);
            };
            expectRejected.template operator()<8>();
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
            expectRejected.template operator()<16>();
#endif
        }
    }

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    /** @test Routed AVX-512 Q4_K/IQ4_XS entries (1-4 rows, contiguous and ordered) equal generic rows. */
    TEST(CPUX86CodedLeaves, AVX512CodedNibbleLeafEqualsGenericChunks)
    {
        constexpr int n = 128, blocks = 37;
        const auto activation = makeActivations(4, blocks, 808u);
        for (const Policy policy : kPolicies)
            for (const Scales scales : {Scales::Finite, Scales::OneNonFinite})
            {
                SCOPED_TRACE("policy=" + std::to_string(int(policy)) +
                             " non_finite=" + std::to_string(scales == Scales::OneNonFinite));
                {
                    SCOPED_TRACE("Q4_K");
                    CPUNativeQuantisedPackedWeights packed;
                    makeWeights(packed, kQ4K, n, blocks, policy, scales, 101u);
                    expectCodedNibbleEntries<NibbleDecodeKind::Q4_1_IDENTITY, true>(packed, activation, blocks);
                }
                {
                    SCOPED_TRACE("IQ4_XS");
                    CPUNativeQuantisedPackedWeights packed;
                    makeWeights(packed, kIQ4XS, n, blocks, policy, scales, 103u);
                    expectCodedNibbleEntries<NibbleDecodeKind::LUT, false>(packed, activation, blocks);
                }
            }
    }
#endif

    /** @test Packed prefill panels (four-row AVX-512 tiles, two-row AVX2 tiles) reproduce serial rows. */
    TEST(CPUX86CodedLeaves, PanelRowTilesEqualSerialRows)
    {
        struct Team
        {
            int previous_threads = omp_get_max_threads();
            int previous_dynamic = omp_get_dynamic();
            Team() { omp_set_dynamic(0); omp_set_num_threads(1); }
            ~Team() { omp_set_num_threads(previous_threads); omp_set_dynamic(previous_dynamic); }
        } team;
        constexpr int n = 128, stride = n + 5;
        constexpr float dirty = -12345.5f;
        const int lanes = activeISALevel() >= ISALevel::AVX512 ? 16 : 8;
        // 37 blocks: contiguous and 16 short ordered partitions; 149 blocks:
        // GPU-aligned partitions of ten carry partials across eight-block panels.
        for (const int blocks : {37, 149})
            for (const Layout &layout : {kQ5K, kIQ3S, kQ4K, kIQ4XS, kQ50})
                for (const Policy policy : kPolicies)
                    for (const Scales scales : {Scales::Finite, Scales::OneNonFinite})
                    {
                        SCOPED_TRACE(std::string(layout.name) + " blocks=" + std::to_string(blocks) + " policy=" +
                                     std::to_string(int(policy)) +
                                     " non_finite=" + std::to_string(scales == Scales::OneNonFinite));
                        CPUNativeQuantisedPackedWeights packed;
                        makeWeights(packed, layout, n, blocks, policy, scales, 111u + layout.codebook);
                        const auto format = x86_panel::formatFor(packed.encoding, packed.codebook_id);
                        const auto activation = makeActivations(19, blocks, 909u + blocks);
                        std::vector<float> serial(std::size_t(19) * stride, dirty);
                        std::vector<float> serial_scratch(nativeQuantisedProjectionPartialFloats(packed, 1));
                        for (int row = 0; row < 19; ++row)
                            gemv_native_quantised_preq(packed, activation.data() + std::size_t(row) * blocks,
                                                       serial.data() + std::size_t(row) * stride, serial_scratch);
                        // Sixteen to nineteen rows: full four-row tiles and tails of one to three rows.
                        for (int rows = 16; rows <= 19; ++rows)
                        {
                            SCOPED_TRACE("rows=" + std::to_string(rows));
                            ASSERT_TRUE(x86_panel::supportsGeometry(format, lanes, rows, n, 1))
                                << "the production Auto route must select the packed panel";
                            std::vector<float> actual(std::size_t(rows) * stride, dirty);
                            std::vector<float> scratch(nativeQuantisedProjectionPartialFloats(packed, rows), dirty);
                            gemm_native_quantised_preq(packed, activation.data(), actual.data(), scratch, rows, stride);
                            for (int row = 0; row < rows; ++row)
                                for (int column = 0; column < stride; ++column)
                                {
                                    const auto index = std::size_t(row) * stride + column;
                                    ASSERT_PRED2(sameOutput, std::bit_cast<std::uint32_t>(actual[index]),
                                                 std::bit_cast<std::uint32_t>(serial[index]))
                                        << "row=" << row << " column=" << column;
                                }
                        }
                    }
    }
} // namespace llaminar2::test
#endif // __AVX2__
