/**
 * @file Test__CPUX86CodedLeaves.cpp
 * @brief Dedicated x86 coded decode leaves against the generic traversals they replace.
 *
 * The x86 decode round replaced generic traversals with dedicated AVX2 and
 * AVX-512 leaves that keep each column's exact INT32 dot and per-policy FP32
 * block program but change the physical walk:
 *
 * - ordinary Q5_0/Q5_1 (q5::dispatch): one unrolled pass per register tile,
 *   with L2-derived K cache tiles for the AVX2 one-row tile (any tile length,
 *   including mid-range starts, gives the same bytes);
 * - Q5_K and IQ3_S (multi_scale::x86::singleScaleRows): unrolled groups,
 *   superblock-addressed units and headers, contiguous and ordered K trees;
 * - AVX-512 Q4_K and IQ4_XS (multi_scale::x86::nibbleSingleScaleRows): per
 *   superblock bases, routed from the one- to four-row nibble chunks and the
 *   ordered GPU-aligned trees;
 * - packed prefill panels: whole 64-column AVX-512 strips (two rows by four
 *   vectors) when every worker owns one, sixteen-column strips (four rows)
 *   otherwise, and two-row AVX2 strips, including explicit backend-native and
 *   GPU-aligned K partition trees that carry partials across panels, and
 *   trees whose whole partitions fit one panel and fold in registers;
 * - pooled packed panels of the batched expert route: ordinary-prefill members
 *   of a multi-projection launch from two rows stream one 64-column chunk per
 *   task through a uniformly strided worker slot of the invocation's slab.
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
 * earlier partition's NaN for every leaf, row tile, lane width and policy.
 * The x86 leaves fold through addOrderedAVX2()/addOrderedAVX512() because a
 * plain vector `+` lets the compiler swap VADDPS operands, which GCC 14 did in
 * three of the four AVX2 one-row ordered instantiations.
 *
 * The GPU-aligned program of the generic multi_scale::rows() (the other
 * compact codebooks: IQ2_XXS, IQ1_S, IQ2_XS, IQ1_M, ...) pins both its
 * per-block accumulation and its partition fold the same way
 * (accumulateBlockOrdered(), accumulateOrdered()); backend-native rows keep their unchanged plain `+`
 * program. FusedOrderedPartitionNaNsKeepEarlierPartitionAndSerialBytes drives
 * it through the fused batched entry with two NaN activation scales per row,
 * in different partitions or inside one partition, and checks the earlier
 * NaN analytically as well as serial-row bytes.
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
        constexpr Layout kIQ2XS{"IQ2_XS", Encoding::CompactMultiScale, 14, true, true, 9};
        constexpr Layout kIQ2XXS{"IQ2_XXS", Encoding::CompactMultiScale, 15, false, true, 8};
        constexpr Layout kIQ1S{"IQ1_S", Encoding::CompactMultiScale, 16, true, true, 6};
        constexpr Layout kIQ1M{"IQ1_M", Encoding::CompactMultiScale, 17, true, true, 6};

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
         * @param live_columns Logical N (at most @p n); padding metadata is normalized before the seal.
         */
        void makeWeights(CPUNativeQuantisedPackedWeights &packed, const Layout &layout, int n, int blocks,
                         Policy policy, Scales scales, std::uint32_t seed, int live_columns = 0)
        {
            ASSERT_EQ(n % 64, 0);
            ASSERT_LE(live_columns, n);
            packed.N = live_columns > 0 ? live_columns : n;
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
            packed.normalizePaddingMetadata();
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
                    auto expected = initialRows(Rows, 91u + chunk);
                    referenceQ5<Lanes, SumPairs>(packed, tile<Rows>(activation, blocks, expected), chunk, range.first,
                                                 range.last, range.accumulate);
                    // The cache-derived AVX2 K tile (0), the L1 floor and a short
                    // tile that starts mid-range: the tile length never changes a byte.
                    for (const int tile_blocks : {0, 16, 5})
                    {
                        auto actual = initialRows(Rows, 91u + chunk);
                        q5::dispatch<Lanes, SumPairs>(packed, tile<Rows>(activation, blocks, actual).input,
                                                      tile<Rows>(activation, blocks, actual).output, chunk,
                                                      range.first, range.last, range.accumulate, tile_blocks);
                        ASSERT_TRUE(sameRows(actual, expected)) << "tile_blocks=" << tile_blocks;
                    }
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

    /** @test The AVX2 one-row Q5 K tile is half the associativity-safe private L2, with an L1 floor. */
    TEST(CPUX86CodedLeaves, Q5AVX2KTileFollowsObservedL2)
    {
        // 1 MiB 8-way: 7/8 of 1 MiB = 917504 B, half for the tile.
        EXPECT_EQ(q5::avx2OneRowKTileBlocks({1u << 20, 32u << 20, 8, 16}, false), 458752 / 1408);
        EXPECT_EQ(q5::avx2OneRowKTileBlocks({1u << 20, 32u << 20, 8, 16}, true), 458752 / 1536);
        EXPECT_EQ(q5::avx2OneRowKTileBlocks({512u << 10, 32u << 20, 8, 16}, false), 229376 / 1408);
        // A one-way cache keeps its full capacity.
        EXPECT_EQ(q5::avx2OneRowKTileBlocks({256u << 10, 8u << 20, 1, 16}, false), 131072 / 1408);
        // Small caches keep the sixteen-block L1 tile.
        EXPECT_EQ(q5::avx2OneRowKTileBlocks({32u << 10, 8u << 20, 8, 16}, false), 16);
        CPUNativeQuantisedPackedWeights packed;
        makeWeights(packed, kQ50, 64, 4, Policy::BackendNative, Scales::Finite, 7u);
        const auto activation = makeActivations(1, 4, 3u);
        std::vector<std::array<float, 64>> output(1);
        const auto t = tile<1>(activation, 4, output);
        EXPECT_THROW(q5::dispatch<8>(packed, t.input, t.output, 0, 0, 4, false, -1), std::invalid_argument);
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

    namespace
    {
        /** @brief Bind one exact OpenMP team width and restore the caller's policy on exit. */
        class ScopedTeam final
        {
        public:
            explicit ScopedTeam(int workers)
                : previous_threads_(omp_get_max_threads()), previous_dynamic_(omp_get_dynamic())
            {
                omp_set_dynamic(0);
                omp_set_num_threads(workers);
            }
            ~ScopedTeam()
            {
                omp_set_num_threads(previous_threads_);
                omp_set_dynamic(previous_dynamic_);
            }
            ScopedTeam(const ScopedTeam &) = delete;
            ScopedTeam &operator=(const ScopedTeam &) = delete;

        private:
            int previous_threads_;
            int previous_dynamic_;
        };

        /**
         * @brief Serial M=1 rows through an explicit K-partition tree, independent of the panel.
         * @details Every partition is one production chunk kernel call from
         * positive zero, folded by reduceNativeQuantisedKTilePartialsExact():
         * the same partial bank and left fold serial decode uses for a K split.
         */
        std::vector<float> partitionedSerialRows(const CPUNativeQuantisedPackedWeights &packed,
                                                 const std::vector<Q8_1Block> &activation, int rows, int stride,
                                                 int partitions, float dirty)
        {
            const int blocks = packed.blocks_per_row, width = (blocks - 1) / partitions + 1;
            const bool wide = activeISALevel() >= ISALevel::AVX512;
            std::vector<float> result(std::size_t(rows) * stride, dirty);
            std::vector<float> bank(std::size_t(partitions) * 64);
            for (int row = 0; row < rows; ++row)
            {
                const Q8_1Block *a = activation.data() + std::size_t(row) * blocks;
                for (int chunk = 0; chunk < packed.N_padded / 64; ++chunk)
                {
                    for (int part = 0; part < partitions; ++part)
                    {
                        const int first = std::min(part * width, blocks), last = std::min(first + width, blocks);
                        float *partial = bank.data() + std::size_t(part) * 64;
                        if (wide)
                        {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
                            if (packed.usesNibbleLUT())
                                gemv_native_quantised_avx512_chunk_native(packed, a, partial, chunk, first, last,
                                                                          build_decode_lut(packed.codebook_id), false);
                            else
                                gemv_native_quantised_avx512_chunk_non_nibble(packed, a, partial, chunk, first, last,
                                                                              false);
#endif
                        }
                        else if (packed.usesNibbleLUT())
                            gemv_avx2_chunk_native(packed, a, partial, chunk, first, last,
                                                   build_decode_lut_avx2_for_codebook(packed.codebook_id), false);
                        else
                            gemv_avx2_chunk_non_nibble(packed, a, partial, chunk, first, last, false);
                    }
                    const int columns = std::min(64, packed.N - chunk * 64);
                    reduceNativeQuantisedKTilePartialsExact(bank.data(),
                        result.data() + std::size_t(row) * stride + chunk * 64, columns, partitions, wide,
                        packed.numerical_policy);
                }
            }
            return result;
        }
    } // namespace

    /** @test Packed prefill panels reproduce serial rows for every strip width and row tile. */
    TEST(CPUX86CodedLeaves, PanelRowTilesEqualSerialRows)
    {
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
                        {
                            ScopedTeam team(1);
                            std::vector<float> serial_scratch(nativeQuantisedProjectionPartialFloats(packed, 1));
                            for (int row = 0; row < 19; ++row)
                                gemv_native_quantised_preq(packed, activation.data() + std::size_t(row) * blocks,
                                                           serial.data() + std::size_t(row) * stride, serial_scratch);
                        }
                        // One worker owns whole 64-column AVX-512 strips (two rows by
                        // four vectors); three workers own sixteen-column strips (four
                        // rows). Sixteen to nineteen rows cover every row-tile tail.
                        for (const int workers : {1, 3})
                        {
                            ScopedTeam team(workers);
                            SCOPED_TRACE("workers=" + std::to_string(workers) + " strip=" +
                                         std::to_string(x86_panel::stripColumns(lanes, n, workers)));
                            for (int rows = 16; rows <= 19; ++rows)
                            {
                                SCOPED_TRACE("rows=" + std::to_string(rows));
                                ASSERT_TRUE(x86_panel::supportsGeometry(format, lanes, rows, n, workers))
                                    << "the production Auto route must select the packed panel";
                                std::vector<float> actual(std::size_t(rows) * stride, dirty);
                                std::vector<float> scratch(nativeQuantisedProjectionPartialFloats(packed, rows), dirty);
                                gemm_native_quantised_preq(packed, activation.data(), actual.data(), scratch, rows,
                                                           stride);
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
    }

    /** @test Panels fold explicit backend-native and GPU-aligned K trees exactly like serial partial banks. */
    TEST(CPUX86CodedLeaves, PanelPartitionedTreesEqualSerialPartials)
    {
        // A ragged N tail (122 = 64 + 58) on both strip widths. 149 blocks in
        // three partitions of 50 or sixteen of 10 carry partials across panels.
        // Partitions that fit one eight-block panel pack whole partitions and
        // fold them in registers: 37 blocks in five of eight (one per panel),
        // sixteen of three (two per panel, three empty trailing partitions) or
        // 37 of one; 17 blocks in sixteen of two (seven empty trailing); the
        // register fold applies to GPU-aligned trees only. Two NaN superblocks
        // of different payloads prove that every GPU-aligned fold keeps the
        // earlier partition's NaN (its ordered-add contract), as the serial
        // partial bank does. Backend-native NaN cells are not compared here:
        // its fold is a plain FP32 add whose NaN operand order the compiler
        // picks, and the unchanged per-partition panel program already
        // publishes other NaN payloads than the serial bank at 67dac67f7
        // (pre-existing, see the PR's x86 expert-panel evidence). This change
        // keeps that program byte for byte; finite cells cover both policies.
        constexpr int n = 128, live_n = 122, rows = 17, stride = live_n + 3;
        constexpr float dirty = -12345.5f;
        const int lanes = activeISALevel() >= ISALevel::AVX512 ? 16 : 8;
        struct Tree
        {
            int blocks, partitions;
        };
        for (const Tree tree : {Tree{149, 3}, Tree{149, 16}, Tree{37, 5}, Tree{37, 16}, Tree{37, 37}, Tree{17, 16}})
        {
            const int blocks = tree.blocks, partitions = tree.partitions;
            const auto activation = makeActivations(rows, blocks, 1313u + blocks);
            for (const Layout &layout : {kQ5K, kIQ3S, kQ4K, kIQ4XS, kQ50})
                for (const Policy policy : kPolicies)
                for (const Scales scales : {Scales::Finite, Scales::TwoNaNSources})
                {
                    if (scales == Scales::TwoNaNSources && policy != Policy::GPUAlignedExpert)
                        continue;
                    SCOPED_TRACE(std::string(layout.name) + " policy=" + std::to_string(int(policy)) + " blocks=" +
                                 std::to_string(blocks) + " partitions=" + std::to_string(partitions) +
                                 " nan=" + std::to_string(scales == Scales::TwoNaNSources));
                    CPUNativeQuantisedPackedWeights packed;
                    makeWeights(packed, layout, n, blocks, policy, scales, 131u + layout.codebook, live_n);
                    const auto format = x86_panel::formatFor(packed.encoding, packed.codebook_id);
                    const auto view = packed.readView();
                    const CPUProjectionRowGrid grid(rows, nullptr);
                    const auto serial = partitionedSerialRows(packed, activation, rows, stride, partitions, dirty);
                    // Carry exactly when a partition is longer than one panel.
                    ASSERT_EQ(x86_panel::needsCarry(blocks, partitions), (blocks - 1) / partitions + 1 > x86_panel::depth);
                    for (const int workers : {1, 2, 3})
                    {
                        SCOPED_TRACE("workers=" + std::to_string(workers) + " strip=" +
                                     std::to_string(x86_panel::stripColumns(lanes, live_n, workers)));
                        ScopedTeam team(workers);
                        std::vector<float> actual(std::size_t(rows) * stride, dirty);
                        std::vector<float> scratch(
                            x86_panel::invocationFloats(format, lanes, rows, live_n, blocks, partitions, workers), dirty);
                        ASSERT_FALSE(scratch.empty());
                        x86_panel::project(view, format, activation.data(), actual.data(), rows, stride, partitions,
                                           workers, lanes, grid, nullptr, scratch);
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
    }

    /**
     * @test Ordinary-prefill fused members stream pooled packed panels from the admitted slab and keep serial bytes.
     * @details The batched MoE expert route pools many small expert projections
     * in one team. A Prefill member with automatic schedules, an x86 panel
     * source and at least two rows takes one task per 64-column chunk that
     * decodes each weight strip once for all of its rows, instead of register
     * row tiles that re-decode every weight per tile (the file-density IQ3_S
     * expert down-projection prefill regression). Every byte still equals
     * independent serial M=1 decode, with and without bias, for both policies,
     * one and three workers, a new and a retained team, 16-, 17- and 40-block
     * K (one-block, two-block-with-empty-tail and three-block GPU-aligned
     * partitions), a masked N tail and single-row members beside panel
     * members. The launch borrows one uniformly strided slot per worker of the
     * invocation's slab, never more than the members' admitted envelope; a
     * one-float-short slab, a team wider than the slab was planned for, and a
     * slab overlapping a sibling's output, activation or bias or the
     * descriptor table are rejected before any write; a slab base off a cache
     * line keeps every byte (relative slot contract). The reported row tile
     * is the executed one. Decode members, explicit ISA and explicit ownership
     * keep their register tiles.
     */
    TEST(CPUX86CodedLeaves, FusedPrefillMembersStreamPanelsAndRetainSerialBytes)
    {
        constexpr std::array<int, 5> member_rows{17, 3, 33, 16, 1};
        constexpr int members = static_cast<int>(member_rows.size());
        constexpr float dirty = 123456.0f;
        const int lanes = activeISALevel() >= ISALevel::AVX512 ? 16 : 8;
        // N = 65: one complete 64-column strip and a one-column tail strip;
        // N = 40: only a tail strip (AVX-512 runs every chunk as a tail).
        for (const int live_n : {65, 40})
        for (const Layout &layout : {kIQ3S, kQ5K, kQ50, kQ4K, kIQ4XS})
            for (const Policy policy : kPolicies)
                for (const int blocks : {16, 17, 40})
                    for (const int workers : {1, 3})
                    {
                        const int n = live_n > 64 ? 128 : 64;
                        SCOPED_TRACE(std::string(layout.name) + " policy=" + std::to_string(int(policy)) + " blocks=" +
                                     std::to_string(blocks) + " workers=" + std::to_string(workers) +
                                     " N=" + std::to_string(live_n));
                        ScopedTeam team(workers);
                        std::array<CPUNativeQuantisedPackedWeights, members> packed;
                        std::array<std::vector<Q8_1Block>, members> input;
                        std::array<std::vector<float>, members> actual, expected, bias;
                        std::array<FusedVerifierRowsDesc, members> descriptors;
                        for (int p = 0; p < members; ++p)
                        {
                            makeWeights(packed[p], layout, n, blocks, policy, Scales::Finite, 401u + 17u * p + blocks,
                                        live_n);
                            const int m = member_rows[p], stride = live_n + 3;
                            input[p] = makeActivations(m, blocks, 777u + 29u * p + blocks);
                            expected[p].assign(std::size_t(m - 1) * stride + live_n + 4, dirty);
                            {
                                std::vector<float> scratch(nativeQuantisedProjectionPartialFloats(packed[p], 1));
                                for (int row = 0; row < m; ++row)
                                    gemv_native_quantised_preq(packed[p], input[p].data() + std::size_t(row) * blocks,
                                                               expected[p].data() + std::size_t(row) * stride, scratch);
                            }
                            if (p % 2 == 0)
                            {
                                // The fused epilogue adds bias after each row's complete K result.
                                bias[p].resize(live_n);
                                for (int column = 0; column < live_n; ++column)
                                    bias[p][column] = float(column - 31) * 0.0193f;
                                for (int row = 0; row < m; ++row)
                                    for (int column = 0; column < live_n; ++column)
                                    {
                                        volatile float sum = expected[p][std::size_t(row) * stride + column] + bias[p][column];
                                        expected[p][std::size_t(row) * stride + column] = sum;
                                    }
                            }
                            actual[p].assign(expected[p].size(), dirty);
                            descriptors[p] = {.packed = &packed[p], .output = actual[p].data(),
                                .bias = bias[p].empty() ? nullptr : bias[p].data(), .N = live_n, .ldc = stride,
                                .rows = m, .input = input[p].data(), .purpose = ProjectionRowsPurpose::Prefill};
                        }
                        // Independent slot geometry: up to eight K blocks of eight
                        // dot-group planes plus a scale plane (and a minimum plane
                        // for Q4_K/Q5_K) of a whole 64-column chunk (AVX-512) or
                        // eight columns (AVX2); no partition here spans a panel, so
                        // no carry; one slot per worker padded to a cache line.
                        const auto plan = planNativeQuantisedFusedRows(descriptors, 1, workers);
                        const bool affine = layout.codebook == 5 || layout.codebook == 7;
                        const std::size_t slot = (std::size_t(std::min(blocks, 8)) * (affine ? 10 : 9) *
                                                  (lanes == 16 ? 64 : 8) + 15) / 16 * 16;
                        std::size_t envelope = 0;
                        for (int p = 0; p < members; ++p)
                        {
                            EXPECT_EQ(plan.projections[p].packed_panel, member_rows[p] >= x86_panel::min_pooled_rows) << p;
                            envelope = std::max(envelope, nativeQuantisedProjectionPartialEnvelopeFloats(packed[p], member_rows[p]));
                        }
                        ASSERT_EQ(plan.panel_workers, workers);
                        EXPECT_EQ(plan.panel_slot_floats, slot);
                        const std::size_t demand = plan.sharedPartialFloats();
                        EXPECT_EQ(demand, std::size_t(workers) * slot);
                        EXPECT_LE(demand, envelope) << "the members' admitted envelope must cover the launch";
                        // The reported tile is the one complete strips execute:
                        // one row for wide multi-block GPU-aligned Q4_K/Q5_K/Q5_0
                        // register folds, otherwise two; N < 64 has no complete
                        // wide strip, only a tail strip of two-row tiles.
                        const bool gpu = policy == Policy::GPUAlignedExpert;
                        const int partition_width = (blocks - 1) / std::max(1, plan.projections[0].k_tiles) + 1;
                        const bool split = lanes == 16 && live_n >= 64 && gpu && plan.projections[0].k_tiles > 1 &&
                            partition_width > 1 && partition_width <= x86_panel::depth &&
                            (layout.codebook == 5 || layout.codebook == 7 || layout.codebook == 6);
                        for (int p = 0; p < members; ++p)
                        {
                            if (!plan.projections[p].packed_panel)
                                continue;
                            const auto dims = FusedProjectionEvidenceDimensions::from(descriptors[p], plan.projections[p],
                                lanes == 16 ? ISAPath::AVX512 : ISAPath::AVX2, true);
                            EXPECT_EQ(dims.panel_row_tile, split ? 1 : 2) << p;
                            const auto tags = dims.tags();
                            EXPECT_EQ(tags.at("physical_row_tile"), split ? "1" : "2") << p;
                            EXPECT_EQ(tags.at("route"), "prefill_x86_packed_panel_rows") << p;
                        }
                        const auto reset = [&] { for (auto &bank : actual) std::fill(bank.begin(), bank.end(), dirty); };
                        const auto check = [&]
                        {
                            for (int p = 0; p < members; ++p)
                                for (std::size_t i = 0; i < actual[p].size(); ++i)
                                    ASSERT_PRED2(sameOutput, std::bit_cast<std::uint32_t>(actual[p][i]),
                                                 std::bit_cast<std::uint32_t>(expected[p][i]))
                                        << "member=" << p << " index=" << i;
                        };
                        constexpr std::size_t guard = 64;
                        AlignedVector<float> slab;
                        slab.resize_uninitialized(demand + guard);
                        const std::span<float> exact(slab.data(), demand);
                        const auto poison = [&] { std::fill(slab.begin(), slab.end(), std::bit_cast<float>(0x7fa00ee1u)); };
                        const auto guarded = [&]
                        {
                            for (std::size_t i = demand; i < slab.size(); ++i)
                                ASSERT_EQ(std::bit_cast<std::uint32_t>(slab[i]), 0x7fa00ee1u) << "slab overrun at " << i;
                        };
                        // A short slab is rejected before any output write.
                        reset();
                        EXPECT_THROW((void)gemm_native_quantised_fused_verifier_rows_preq(nullptr, descriptors.data(),
                            exact.first(demand - 1), members, 1, blocks), std::invalid_argument);
                        for (const auto &bank : actual)
                            for (const float value : bank)
                                ASSERT_EQ(value, dirty);
                        reset();
                        poison();
                        ASSERT_TRUE(gemm_native_quantised_fused_verifier_rows_preq(nullptr, descriptors.data(), exact,
                                                                                   members, 1, blocks));
                        check();
                        guarded();
                        // A retained team indexes the same slots by its member ids.
                        reset();
                        poison();
#pragma omp parallel
                        {
                            (void)gemm_native_quantised_fused_verifier_rows_preq(nullptr, descriptors.data(), exact,
                                                                                 members, 1, blocks);
                        }
                        check();
                        guarded();
                        // The slab contract is relative: worker w owns floats
                        // [w*slot, (w+1)*slot), slots being whole 64-byte lines
                        // apart; every panel access is unaligned-safe, so a slab
                        // base off a cache line keeps every byte (production
                        // workspaces are 64-byte aligned, so slots never share
                        // a line there).
                        {
                            AlignedVector<float> offset_slab;
                            offset_slab.resize_uninitialized(demand + 1);
                            reset();
                            ASSERT_TRUE(gemm_native_quantised_fused_verifier_rows_preq(nullptr, descriptors.data(),
                                std::span<float>(offset_slab.data() + 1, demand), members, 1, blocks));
                            check();
                        }
                        // Adversarial admission, all before any output write: a
                        // team wider than the slab was sized for, and a slab
                        // that overlaps a sibling's output, activation or bias,
                        // or the descriptor table.
                        const auto rejectedBeforeWrite = [&](const char *what, const FusedVerifierRowsDesc *bundle,
                                                             std::span<float> bank)
                        {
                            SCOPED_TRACE(what);
                            reset();
                            EXPECT_THROW((void)gemm_native_quantised_fused_verifier_rows_preq(nullptr, bundle, bank,
                                members, 1, blocks), std::invalid_argument);
                            for (const auto &out : actual)
                                for (const float value : out)
                                    ASSERT_EQ(value, dirty);
                        };
                        {
                            ScopedTeam wider(workers + 1);
                            rejectedBeforeWrite("team wider than the planned slab", descriptors.data(), exact);
                        }
                        // One buffer holds the slab and, inside its used prefix,
                        // the aliased object; the single-row member 4 (a tile
                        // member running concurrently with panel tasks) and the
                        // biased panel member 0 are the victims.
                        const auto aliased = [&](const char *what, auto bind)
                        {
                            AlignedVector<float> shared;
                            shared.resize_uninitialized(demand + 4096);
                            std::fill(shared.data(), shared.data() + shared.size(), dirty);
                            auto bundle = descriptors;
                            bind(bundle, shared.data());
                            rejectedBeforeWrite(what, bundle.data(), std::span<float>(shared.data(), demand));
                            // Nothing inside the shared buffer was written either.
                            for (std::size_t i = 0; i < 64; ++i)
                                ASSERT_EQ(shared.data()[i], dirty) << what;
                        };
                        aliased("sibling output", [&](auto &bundle, float *base) { bundle[4].output = base; });
                        aliased("sibling activation", [&](auto &bundle, float *base) {
                            std::memcpy(base + 64, input[4].data(), input[4].size() * sizeof(Q8_1Block));
                            bundle[4].input = reinterpret_cast<const Q8_1Block *>(base + 64);
                        });
                        aliased("panel member bias", [&](auto &bundle, float *base) {
                            std::memcpy(base + 64, bias[0].data(), bias[0].size() * sizeof(float));
                            bundle[0].bias = base + 64;
                        });
                        {
                            AlignedVector<float> shared;
                            shared.resize_uninitialized(demand + 4096);
                            std::fill(shared.data(), shared.data() + shared.size(), dirty);
                            auto *table = reinterpret_cast<FusedVerifierRowsDesc *>(shared.data() + 64);
                            std::copy(descriptors.begin(), descriptors.end(), table);
                            rejectedBeforeWrite("descriptor table", table, std::span<float>(shared.data(), demand));
                        }
                        // Decode members, an explicit ISA and explicit ownership keep their tiles.
                        auto decode = descriptors;
                        for (auto &d : decode)
                            d.purpose = ProjectionRowsPurpose::Decode;
                        const auto unpooled = [&](std::span<const FusedVerifierRowsDesc> bundle, ISAPath path,
                                                  FusedOutputJobOwnership ownership)
                        {
                            const auto other = planNativeQuantisedFusedRows(bundle, 1, workers, path, ownership);
                            for (int p = 0; p < members; ++p)
                                EXPECT_FALSE(other.projections[p].packed_panel) << p;
                            EXPECT_EQ(other.panel_workers, 0);
                            std::vector<float> bank(std::max<std::size_t>(other.sharedPartialFloats(), 1));
                            reset();
                            ASSERT_TRUE(gemm_native_quantised_fused_verifier_rows_preq(nullptr, bundle.data(), bank,
                                                                                       members, 1, blocks, path, ownership));
                            check();
                        };
                        unpooled(decode, ISAPath::AUTO, FusedOutputJobOwnership::Auto);
                        unpooled(descriptors, ISAPath::AVX2, FusedOutputJobOwnership::Auto);
                        unpooled(descriptors, ISAPath::AUTO, FusedOutputJobOwnership::Current);
                    }
    }
    /**
     * @test The fused batched route keeps the earlier ordered partition's NaN for nonpanel and panel members.
     * @details GPU-aligned ordered trees publish the earlier partition's NaN
     * (the addOrdered contract of every x86 ordered fold). IQ2_XS and IQ1_M
     * (two-scale blocks) and IQ2_XXS and IQ1_S (whole-block single-scale
     * blocks, the other per-block accumulation site) take no packed panel;
     * their tiles run the GPU-aligned program of the generic
     * multi_scale::rows(), whose partition fold and per-block accumulation
     * are pinned to that contract, so the operand order no longer depends on
     * how serial decode, row tiles or the fused entry inline them. At
     * 67dac67f7 both failed: two NaNs inside one partition kept the later NaN
     * (AVX2 GCC 14 accumulated the contribution as VADDPS source one) and the
     * fused entry's IQ2_XS/IQ1_M rows differed from serial decode. Rows r % 3 == 1 carry distinct NaN activation
     * scales (positive and negative, different payloads) in two partitions;
     * rows r % 3 == 2 carry two in one partition. Every NaN row must publish
     * the earlier NaN in every live lane, and every row must equal independent
     * serial M=1 decode, for nonpanel-only bundles and for mixed bundles whose
     * IQ3_S members stream pooled panels, at one and three workers, standalone
     * and in a retained team, K of 17 and 40 blocks.
     */
    TEST(CPUX86CodedLeaves, FusedOrderedPartitionNaNsKeepEarlierPartitionAndSerialBytes)
    {
        constexpr int n = 128, live_n = 67;
        constexpr float dirty = 123456.0f;
        struct Member
        {
            const Layout *layout;
            int rows;
        };
        const std::array<std::vector<Member>, 2> bundles{{
            {{&kIQ2XS, 5}, {&kIQ1M, 17}, {&kIQ2XS, 1}, {&kIQ1M, 2}, {&kIQ2XS, 3}, {&kIQ2XXS, 5}, {&kIQ1S, 4}},
            {{&kIQ3S, 4}, {&kIQ2XS, 5}, {&kIQ1M, 17}, {&kIQ3S, 1}, {&kIQ1M, 2}, {&kIQ2XXS, 3}, {&kIQ1S, 1}}}};
        for (const int blocks : {17, 40})
            for (std::size_t bundle = 0; bundle < bundles.size(); ++bundle)
                for (const int workers : {1, 3})
                {
                    const auto &members = bundles[bundle];
                    const int count = static_cast<int>(members.size());
                    SCOPED_TRACE("blocks=" + std::to_string(blocks) + " bundle=" +
                                 (bundle ? "mixed" : "nonpanel") + " workers=" + std::to_string(workers));
                    ScopedTeam team(workers);
                    // GPU-aligned trees: sixteen partitions of ceil(K/16) blocks.
                    const int width = (blocks - 1) / MoEProjectionNumericalContract::ordered_k_partitions + 1;
                    std::vector<CPUNativeQuantisedPackedWeights> packed(count);
                    std::vector<std::vector<Q8_1Block>> input(count);
                    std::vector<std::vector<float>> actual(count), expected(count);
                    std::vector<std::vector<std::uint32_t>> earlier(count);
                    std::vector<FusedVerifierRowsDesc> descriptors(count);
                    const int stride = live_n + 3;
                    for (int p = 0; p < count; ++p)
                    {
                        const int m = members[p].rows;
                        makeWeights(packed[p], *members[p].layout, n, blocks, Policy::GPUAlignedExpert, Scales::Finite,
                                    907u + 13u * p + blocks, live_n);
                        input[p] = makeActivations(m, blocks, 1907u + 31u * p + blocks);
                        earlier[p].assign(m, 0);
                        for (int row = 0; row < m; ++row)
                        {
                            const auto nan = [&](int block, std::uint16_t half) {
                                input[p][std::size_t(row) * blocks + block].d = half;
                            };
                            std::uint16_t first = 0;
                            if (row % 3 == 1)
                            {
                                // Partition 0 and the last nonempty partition.
                                first = static_cast<std::uint16_t>(0x7E00u | (1u + row));
                                nan(1, first);
                                nan(blocks - 2, static_cast<std::uint16_t>(0xFE00u | (0x1F0u - row)));
                            }
                            else if (row % 3 == 2)
                            {
                                // Two blocks of partition 1.
                                first = static_cast<std::uint16_t>(0xFE00u | (3u + row));
                                nan(width, first);
                                nan(width + 1, static_cast<std::uint16_t>(0x7E00u | (0x1E0u - row)));
                            }
                            if (first)
                                earlier[p][row] = std::bit_cast<std::uint32_t>(nativeQuantisedFP16ScaleToFP32(first));
                        }
                        expected[p].assign(std::size_t(m - 1) * stride + live_n + 4, dirty);
                        {
                            std::vector<float> scratch(nativeQuantisedProjectionPartialFloats(packed[p], 1));
                            for (int row = 0; row < m; ++row)
                                gemv_native_quantised_preq(packed[p], input[p].data() + std::size_t(row) * blocks,
                                                           expected[p].data() + std::size_t(row) * stride, scratch);
                        }
                        actual[p].assign(expected[p].size(), dirty);
                        descriptors[p] = {.packed = &packed[p], .output = actual[p].data(), .N = live_n, .ldc = stride,
                            .rows = m, .input = input[p].data(), .purpose = ProjectionRowsPurpose::Prefill};
                    }
                    const auto plan = planNativeQuantisedFusedRows(descriptors, 1, workers);
                    for (int p = 0; p < count; ++p)
                        ASSERT_EQ(plan.projections[p].packed_panel, members[p].layout == &kIQ3S && members[p].rows >= 2)
                            << "member " << p;
                    std::vector<float> slab(std::max<std::size_t>(plan.sharedPartialFloats(), 1));
                    const auto check = [&](const char *mode)
                    {
                        for (int p = 0; p < count; ++p)
                            for (int row = 0; row < members[p].rows; ++row)
                                for (int column = 0; column < stride; ++column)
                                {
                                    const auto index = std::size_t(row) * stride + column;
                                    const auto got = std::bit_cast<std::uint32_t>(actual[p][index]);
                                    ASSERT_EQ(got, std::bit_cast<std::uint32_t>(expected[p][index]))
                                        << mode << " member=" << p << " " << members[p].layout->name << " row=" << row
                                        << " column=" << column << " got=" << hexBits(got)
                                        << " serial=" << hexBits(std::bit_cast<std::uint32_t>(expected[p][index]));
                                    if (earlier[p][row] && column < live_n)
                                        ASSERT_EQ(got, earlier[p][row])
                                            << mode << " member=" << p << " " << members[p].layout->name << " row=" << row
                                            << " column=" << column << " got=" << hexBits(got)
                                            << " earlier=" << hexBits(earlier[p][row]);
                                }
                    };
                    for (auto &bank : actual)
                        std::fill(bank.begin(), bank.end(), dirty);
                    ASSERT_TRUE(gemm_native_quantised_fused_verifier_rows_preq(nullptr, descriptors.data(), slab, count,
                                                                               1, blocks));
                    check("standalone");
                    for (auto &bank : actual)
                        std::fill(bank.begin(), bank.end(), dirty);
#pragma omp parallel
                    {
                        (void)gemm_native_quantised_fused_verifier_rows_preq(nullptr, descriptors.data(), slab, count, 1,
                                                                             blocks);
                    }
                    check("retained");
                }
    }
} // namespace llaminar2::test
#endif // __AVX2__
