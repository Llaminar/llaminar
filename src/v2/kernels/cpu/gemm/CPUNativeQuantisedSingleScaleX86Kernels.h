/**
 * @file CPUNativeQuantisedSingleScaleX86Kernels.h
 * @brief AVX2/AVX-512 contiguous leaves of the whole-block single-scale coded formats.
 *
 * These leaves replace the generic multi_scale::rows() traversal on x86 for
 * the two compact codebooks whose 32 values share one source scale, Q5_K and
 * IQ3_S, and (AVX-512) the generic nibble chunks for the superblock-coded
 * nibble formats Q4_K and IQ4_XS. They read the same file-density units
 * (Q5_K: four nibble words, eight bit planes, a 12-bit code and a d|dmin
 * superblock word; IQ3_S: two index words, a qh and sign word, a sign tail
 * byte, a 4-bit code and a d superblock word; Q4_K: four nibble words, a
 * 12-bit code and a d|dmin word; IQ4_XS: four nibble words, a 6-bit code and
 * a d word) and keep the exact arithmetic of the kernels they replace: an
 * exact INT32 dot per column and block, then the unchanged per-policy FP32
 * block contribution in ascending K.
 *
 * Only the physical traversal and integer formulation differ:
 * - register-bounded groups of column strips share one sequential walk of the
 *   chunk row, with all eight K groups unrolled (fixed-stride loads);
 * - superblock headers and block units are addressed once per superblock,
 *   without per-block geometry branches;
 * - the policy and the physical layout are fixed before the K traversal, so
 *   the K loop holds no format, encoding or policy branch;
 * - weights enter the dot unsigned. IQ3_S grid bytes are biased by sixteen
 *   (an immutable derived copy of the source grid, [17,31] or [1,15] after
 *   the sign), and sixteen times the activation sum is removed again. This
 *   requires Q8_1Block::sum_qs to equal the sum of the 32 published bytes on
 *   both widths (AVX2 formerly used a signed dot that never read it), which
 *   every production quantizer guarantees and
 *   CPUCompensationFreeUnits.ProductionActivationQuantizersPublishExactSums
 *   (V2_Integration_CPUCompensationFreeUnits, ProductionTestPreflight) proves;
 *   AVX2 keeps four groups' bounded INT16 pair sums before widening;
 * - ordered K partitions fold through addOrderedAVX2()/addOrderedAVX512(),
 *   so the earlier partition's NaN payload wins as in every other x86 tree.
 * No decoded weights are stored, nothing is allocated and NEON is unaffected.
 */
#pragma once

#if defined(__AVX2__)

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <utility>

#include "CPUNativeQuantisedMultiScaleKernels.h"

namespace llaminar2::cpu::native_quantised::multi_scale::x86
{
    /** @return Whether every byte of every grid entry is a positive value below sixteen. */
    consteval bool iq3sGridBytesBelowSixteen()
    {
        for (const std::uint32_t entry : iq3s_grid)
            for (int byte = 0; byte < 4; ++byte)
            {
                const auto value = (entry >> (8 * byte)) & 255u;
                if (value == 0 || value > 15)
                    return false;
            }
        return true;
    }
    static_assert(iq3sGridBytesBelowSixteen(), "IQ3_S biased grid requires source magnitudes in [1,15]");

    /**
     * @brief The immutable IQ3_S source grid with sixteen added to every byte.
     *
     * A positive value w becomes 16 + w in [17,31]; its negation is 32 minus
     * that byte, 16 - w in [1,15]. Both are unsigned dot operands whose INT16
     * pair sums stay below 2*31*128 = 7936, and the dot over a block equals
     * the signed dot plus sixteen times the activation sum. This is a codebook
     * constant (2 KiB), not a prepared or expanded weight representation.
     */
    alignas(64) inline constexpr std::array<std::uint32_t, 512> kIQ3SGridPlus16 = [] {
        std::array<std::uint32_t, 512> table{};
        for (std::size_t index = 0; index < table.size(); ++index)
            table[index] = iq3s_grid[index] + 0x10101010u;
        return table;
    }();

    /**
     * @brief Column strips that share one K traversal for a tile.
     * @tparam Lanes Eight AVX2 or sixteen AVX-512 output columns per strip.
     * @tparam Rows Live activation rows sharing each decode.
     *
     * Bounded by accumulators, superblock factors and decode temporaries:
     * AVX-512 has 32 vector registers, AVX2 sixteen.
     */
    template <int Codebook, int Lanes, std::size_t Rows>
    inline constexpr int kSingleScaleStrips = Lanes == 16 && Rows == 1 ? (Codebook == 11 ? 2 : 4) : 1;

    /**
     * @brief Physical identity of a single-scale coded leaf.
     * @tparam Codebook 4 (IQ4_XS), 5 (Q4_K), 7 (Q5_K) or 11 (IQ3_S).
     */
    template <int Codebook> struct SingleScaleLayout
    {
        static_assert(Codebook == 4 || Codebook == 5 || Codebook == 7 || Codebook == 11);
        static constexpr auto encoding = Codebook == 5 ? CPUNativeQuantisedEncoding::Q4KSuperblockCoded
            : Codebook == 4 ? CPUNativeQuantisedEncoding::NibbleLUTSourceScale
                            : CPUNativeQuantisedEncoding::CompactMultiScale;
        /// Whether the block contribution carries an additive minimum.
        static constexpr bool affine = Codebook == 5 || Codebook == 7;
        /// Whether the payload is four plain nibble words (low: groups 0..3, high: 4..7).
        static constexpr bool nibbles = Codebook == 4 || Codebook == 5;
    };

    /** @return One signed Q8 activation word, replicated to every lane. */
    template <typename I>
    [[nodiscard, gnu::always_inline]] inline I activationWord(const Q8_1Block &block, int group) noexcept
    {
        int word;
        std::memcpy(&word, block.qs + group * 4, sizeof(word));
        return I{} + word;
    }

    /** @return kLanes contiguous unsigned bytes zero-extended to INT32 lanes (VPMOVZXBD). */
    template <int Lanes>
    [[nodiscard, gnu::always_inline]] inline typename SIMD<Lanes>::I zeroExtendBytes(const std::uint8_t *source) noexcept
    {
        using I = typename SIMD<Lanes>::I;
        if constexpr (Lanes == 8)
            return std::bit_cast<I>(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(source))));
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        else
            return std::bit_cast<I>(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(source))));
#endif
    }

    /**
     * @brief Replicate one byte of each column word into all four of its bytes.
     * @tparam Byte Source byte in [0,3].
     */
    template <int Lanes, int Byte>
    [[nodiscard, gnu::always_inline]] inline typename SIMD<Lanes>::I replicateByte(typename SIMD<Lanes>::I value) noexcept
    {
        using I = typename SIMD<Lanes>::I;
        constexpr int base = Byte * 0x01010101;
        if constexpr (Lanes == 8)
            return std::bit_cast<I>(_mm256_shuffle_epi8(std::bit_cast<__m256i>(value),
                _mm256_setr_epi32(base, base + 0x04040404, base + 0x08080808, base + 0x0c0c0c0c,
                                  base, base + 0x04040404, base + 0x08080808, base + 0x0c0c0c0c)));
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        else
            return std::bit_cast<I>(_mm512_shuffle_epi8(std::bit_cast<__m512i>(value),
                _mm512_set4_epi32(base + 0x0c0c0c0c, base + 0x08080808, base + 0x04040404, base)));
#endif
    }

    /**
     * @brief Decode one IQ3_S group to unsigned dot bytes 16 +- |grid|.
     * @tparam Group Four-value group in [0,7].
     * @param unit First byte of the 64-column block: index words 0 and 1, the
     *        qh/sign word 2 at byte 512 and the sign-tail plane at byte 768.
     * @param column First column of the strip.
     * @return Bytes equal to sixteen plus the signed source values.
     *
     * The index and sign bits are exactly decodeIQ3SGroup()'s: byte Group%4
     * of index word Group/4 plus qh bit Group as bit eight, and sign byte
     * Group/2 tested in its low (even group) or high (odd group) nibble.
     * Loading a word plane B bytes late puts byte B of every column word at
     * byte 0, so one replicating shuffle serves every sign byte and no source
     * word needs to stay live across groups. Late loads end inside the unit.
     */
    template <int Lanes, int Group>
    [[nodiscard, gnu::always_inline]] inline typename SIMD<Lanes>::I decodeIQ3S(const std::uint8_t *unit, int column) noexcept
    {
        using Ops = SIMD<Lanes>;
        using I = typename Ops::I;
        const I index = (load<I>(unit + (Group / 4) * 256 + column * 4 + Group % 4) & 255) |
                        ((load<I>(unit + 512 + column * 4) << (8 - Group)) & 256);
        const I sign_source = Group < 6 ? load<I>(unit + 512 + column * 4 + 1 + Group / 2)
                                        : zeroExtendBytes<Lanes>(unit + 768 + column);
        const I signs = replicateByte<Lanes, 0>(sign_source);
        constexpr int bits = 0x08040201 << (4 * (Group % 2));
        if constexpr (Lanes == 8)
        {
            const __m256i grid = _mm256_i32gather_epi32(reinterpret_cast<const int *>(kIQ3SGridPlus16.data()),
                                                        std::bit_cast<__m256i>(index), 4);
            // Negated bytes: 32 - u == (u ^ 0xff) + 33 (mod 256) where the sign is set.
            const __m256i mask = _mm256_set1_epi32(bits);
            const __m256i negative = _mm256_cmpeq_epi8(_mm256_and_si256(std::bit_cast<__m256i>(signs), mask), mask);
            return std::bit_cast<I>(_mm256_add_epi8(_mm256_xor_si256(grid, negative),
                                                    _mm256_and_si256(negative, _mm256_set1_epi8(33))));
        }
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        else
        {
            const __m512i grid = _mm512_i32gather_epi32(std::bit_cast<__m512i>(index), kIQ3SGridPlus16.data(), 4);
            const __mmask64 negative = _mm512_test_epi8_mask(std::bit_cast<__m512i>(signs), _mm512_set1_epi32(bits));
            return std::bit_cast<I>(_mm512_mask_sub_epi8(grid, negative, _mm512_set1_epi8(32), grid));
        }
#endif
    }

    /**
     * @brief Decode one Q5_K group to unsigned five-bit dot bytes.
     * @tparam Group Four-value group in [0,7].
     * @param unit First byte of the 64-column block.
     * @param column First column of the strip.
     * @param raw Low-nibble word Group%4 of the strip.
     * @return Exactly dotSingleScale()'s low | 16*fifth-bit bytes.
     */
    template <int Lanes, int Group>
    [[nodiscard, gnu::always_inline]] inline typename SIMD<Lanes>::I decodeQ5K(
        const std::uint8_t *unit, int column, typename SIMD<Lanes>::I raw) noexcept
    {
        using Ops = SIMD<Lanes>;
        using I = typename Ops::I;
        const auto *fifth = unit + 1024 + 32 * Group + column / 2;
        if constexpr (Lanes == 8)
        {
            const I low = (Group < 4 ? raw : raw >> 4) & 0x0F0F0F0F;
            return low | Ops::template q5High<false>(fifth);
        }
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        else
        {
            // (nibble & 15) | (bit ? 16 : 0): a zero-masked byte broadcast of
            // sixteen from a general register (no vector constant) and one
            // ternary-logic merge (A | (B & C), imm 0xF8).
            std::uint64_t mask;
            std::memcpy(&mask, fifth, sizeof(mask));
            const __m512i nibbles = Group < 4 ? std::bit_cast<__m512i>(raw) : _mm512_srli_epi32(std::bit_cast<__m512i>(raw), 4);
            const __m512i high = _mm512_maskz_set1_epi8(mask, 16);
            return std::bit_cast<I>(_mm512_ternarylogic_epi32(high, nibbles, _mm512_set1_epi8(15), 0xF8));
        }
#endif
    }

    /**
     * @brief Accumulate one strip's eight unsigned groups into one exact INT32 dot per row.
     * @tparam Order The eight groups in visiting order; integer sums are order-free.
     * @param decode Group decoder, called once per group and shared by every row.
     *
     * AVX-512 uses VPDPBUSD per group. AVX2 sums four groups' VPMADDUBSW
     * pair sums in INT16 (each at most 2*31*128 = 7936, four at most 31744)
     * and widens once; the INT32 dot is identical in every case. Visiting groups
     * that share a source word back to back keeps that word's live range short.
     */
    template <int Lanes, std::size_t Rows, int... Order, typename Decode>
    [[gnu::always_inline]] inline void dotGroups(std::integer_sequence<int, Order...>,
                                                 const std::array<const Q8_1Block *, Rows> &activation, int kb,
                                                 Decode &&decode, typename SIMD<Lanes>::I (&dots)[Rows]) noexcept
    {
        static_assert(sizeof...(Order) == 8);
        using I = typename SIMD<Lanes>::I;
        constexpr int order[8] = {Order...};
        if constexpr (Lanes == 8)
        {
            const auto quad = [&]<int First>() {
                __m256i sums[Rows];
                const auto group = [&]<int Slot>() {
                    constexpr int Group = order[Slot];
                    const auto weights = std::bit_cast<__m256i>(decode.template operator()<Group>());
#pragma GCC unroll 4
                    for (std::size_t row = 0; row < Rows; ++row)
                    {
                        const auto pairs = _mm256_maddubs_epi16(weights,
                            std::bit_cast<__m256i>(activationWord<I>(activation[row][kb], Group)));
                        sums[row] = Slot == First ? pairs : _mm256_add_epi16(sums[row], pairs);
                    }
                };
                group.template operator()<First>();
                group.template operator()<First + 1>();
                group.template operator()<First + 2>();
                group.template operator()<First + 3>();
#pragma GCC unroll 4
                for (std::size_t row = 0; row < Rows; ++row)
                    dots[row] += std::bit_cast<I>(_mm256_madd_epi16(sums[row], _mm256_set1_epi16(1)));
            };
            quad.template operator()<0>();
            quad.template operator()<4>();
        }
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        else
        {
            const auto group = [&]<int Group>() {
                const auto weights = decode.template operator()<Group>();
#pragma GCC unroll 4
                for (std::size_t row = 0; row < Rows; ++row)
                    dots[row] = SIMD<16>::dot(dots[row], weights, activationWord<I>(activation[row][kb], Group));
            };
            (group.template operator()<Order>(), ...);
        }
#endif
    }

    /**
     * @brief Contiguous Q5_K/IQ3_S tile under one fixed FP32 policy.
     * @tparam Codebook 7 (Q5_K) or 11 (IQ3_S).
     * @tparam Lanes Eight AVX2 or sixteen AVX-512 columns per strip.
     * @tparam Rows Rows sharing every decode, in [1,4].
     * @tparam Policy Prepared numerical policy.
     * @param packed Admitted coded compact matrix (CompactMultiScale, phase 0).
     * @param activation Persistent Q8_1 rows.
     * @param output Sixty-four writable FP32 columns per row.
     * @param chunk Output-column chunk.
     * @param begin First logical K block; may start inside a superblock.
     * @param end One-past-last logical K block.
     * @param accumulate Start from the existing output instead of positive zero (Contiguous only).
     * @param partitions Exact ordered partition count; one for Contiguous.
     * @param blocks_per_partition Consecutive K blocks per ordered partition; unused for Contiguous.
     */
    template <int Codebook, int Lanes, std::size_t Rows, Reduction Tree, CPUProjectionNumericalPolicy Policy,
              typename Packed>
    LLAMINAR_MULTISCALE_EXACT inline void singleScalePolicy(const Packed &packed,
        const std::array<const Q8_1Block *, Rows> &activation, const std::array<float *, Rows> &output, int chunk,
        int begin, int end, bool accumulate, int partitions, int blocks_per_partition)
    {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
        using Layout = SingleScaleLayout<Codebook>;
        static_assert(!Layout::nibbles, "Q4_K and IQ4_XS use nibblePolicy()");
        using Ops = SIMD<Lanes>;
        using I = typename Ops::I;
        using F = typename Ops::F;
        using L = coded_metadata::Lanes<Ops>;
        constexpr int strips = kSingleScaleStrips<Codebook, Lanes, Rows>;
        constexpr auto encoding = Layout::encoding;
        constexpr std::size_t data_stride = preparedDataStride(encoding, Codebook);
        constexpr std::size_t block_stride = preparedInterleavedBlockStride(encoding, Layout::affine, Codebook);
        // IQ3_S weights enter the dot as 16 + w; Q5_K values are unsigned already.
        constexpr int center = Codebook == 11 ? 16 : 0;
        const auto geometry = preparedBlockGeometry(encoding, Layout::affine, Codebook,
                                                    static_cast<std::size_t>(packed.blocks_per_row));
        const std::uint8_t *const row_base = packed.interleavedBase() + geometry.blockOffset(chunk, 0);
        for (int pass = 0; pass < 64; pass += Lanes * strips)
        {
            F reduced[Rows][strips];
            // Ordered trees start every partition at positive zero and fold it
            // into the earlier ones in ascending order, exactly as rows() and
            // the nibble ordered kernels do. Contiguous K is one partition.
            const int partition_count = Tree == Reduction::OrderedPartitions ? partitions : 1;
            for (int partition = 0; partition < partition_count; ++partition)
            {
                F acc[Rows][strips];
#pragma GCC unroll 4
                for (std::size_t row = 0; row < Rows; ++row)
#pragma GCC unroll 4
                    for (int strip = 0; strip < strips; ++strip)
                        acc[row][strip] = Tree == Reduction::Contiguous && accumulate
                            ? load<F>(output[row] + pass + strip * Lanes) : F{};
                const int first = Tree == Reduction::OrderedPartitions ? begin + partition * blocks_per_partition : begin;
                const int last = Tree == Reduction::OrderedPartitions ? std::min(first + blocks_per_partition, end) : end;
                for (int kb = first; kb < last;)
                {
                    // One superblock (or its ragged/partial intersection) at a time:
                    // its header stays resident in L1 and is widened per block, so
                    // no factor vector has to survive the block's decode.
                    const int superblock_end = std::min(last, static_cast<int>(geometry.superblockEnd(kb)));
                    const std::uint8_t *const header = row_base + geometry.headerOffset(0, kb);
                    const std::uint8_t *unit = row_base + geometry.blockOffset(0, kb);
                    for (; kb < superblock_end; ++kb, unit += block_stride)
                    {
                        // Re-read the L1-resident header in every block: hoisting its
                        // widened factors out of the block loop would only spill them.
                        const std::uint8_t *block_header = header;
                        asm("" : "+r"(block_header));
    #pragma GCC unroll 4
                        for (int strip = 0; strip < strips; ++strip)
                        {
                            const int column = pass + strip * Lanes;
                            I dots[Rows]{};
                            [[maybe_unused]] I code;
                            if constexpr (Codebook == 11)
                            {
                                dotGroups<Lanes, Rows>(std::integer_sequence<int, 0, 1, 2, 3, 4, 5, 6, 7>{}, activation, kb,
                                    [&]<int Group>() { return decodeIQ3S<Lanes, Group>(unit, column); }, dots);
                                code = L::nibbles(unit + data_stride, column);
                            }
                            else
                            {
                                // Groups g and g + 4 share nibble word g: visit them together.
                                dotGroups<Lanes, Rows>(std::integer_sequence<int, 0, 4, 1, 5, 2, 6, 3, 7>{}, activation, kb,
                                    [&]<int Group>() {
                                        return decodeQ5K<Lanes, Group>(unit, column, load<I>(unit + (Group % 4) * 256 + column * 4));
                                    }, dots);
                                code = zeroExtendBytes<Lanes>(unit + data_stride + column) |
                                       (L::nibbles(unit + data_stride + 64, column) << 8);
                            }
                            // The codec's widened factors: iq3sSourceScale() and reconstruct<7>().
                            F scale0, scale1{};
                            if constexpr (Codebook == 11)
                            {
                                const I word = L::template superblockWords<11>(block_header, column);
                                scale0 = L::half(word) * L::toFloat(2 * (code & 15) + 1);
                            }
                            else
                            {
                                const I word = L::template superblockWords<7>(block_header, column);
                                const F dmin = L::half(word >> 16);
                                scale0 = L::staged(L::half(word) * L::toFloat(code & 63));
                                scale1 = L::staged(-dmin * L::toFloat((code >> 6) & 63));
                            }
    #pragma GCC unroll 4
                            for (std::size_t row = 0; row < Rows; ++row)
                            {
                                const Q8_1Block &a = activation[row][kb];
                                const auto dot = dots[row] - center * static_cast<int>(a.sum_qs);
                                const float scale_a = nativeQuantisedFP16ScaleToFP32(a.d);
                                if constexpr (Lanes == 8)
                                {
                                    if constexpr (Layout::affine)
                                        acc[row][strip] = std::bit_cast<F>(accumulateCorrectedSingleScaleBlockAVX2(
                                            std::bit_cast<__m256>(acc[row][strip]), std::bit_cast<__m256i>(dot),
                                            std::bit_cast<__m256>(scale0), a.sum_qs, std::bit_cast<__m256>(scale1), scale_a, Policy));
                                    else
                                        acc[row][strip] = std::bit_cast<F>(accumulateSingleScaleBlockAVX2(
                                            std::bit_cast<__m256>(acc[row][strip]), std::bit_cast<__m256i>(dot),
                                            std::bit_cast<__m256>(scale0), scale_a, Policy));
                                }
    #if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
                                else
                                {
                                    if constexpr (Layout::affine)
                                        acc[row][strip] = std::bit_cast<F>(accumulateCorrectedSingleScaleBlockAVX512(
                                            std::bit_cast<__m512>(acc[row][strip]), std::bit_cast<__m512i>(dot),
                                            std::bit_cast<__m512>(scale0), a.sum_qs, std::bit_cast<__m512>(scale1), scale_a, Policy));
                                    else
                                        acc[row][strip] = std::bit_cast<F>(accumulateSingleScaleBlockAVX512(
                                            std::bit_cast<__m512>(acc[row][strip]), std::bit_cast<__m512i>(dot),
                                            std::bit_cast<__m512>(scale0), scale_a, Policy));
                                }
#endif
                            }
                        }
                    }
                }
#pragma GCC unroll 4
                for (std::size_t row = 0; row < Rows; ++row)
#pragma GCC unroll 4
                    for (int strip = 0; strip < strips; ++strip)
                        reduced[row][strip] = partition == 0 ? acc[row][strip] : foldOrdered<Lanes>(reduced[row][strip], acc[row][strip]);
            }
#pragma GCC unroll 4
            for (std::size_t row = 0; row < Rows; ++row)
#pragma GCC unroll 4
                for (int strip = 0; strip < strips; ++strip)
                    std::memcpy(output[row] + pass + strip * Lanes, &reduced[row][strip], sizeof(F));
        }
    }

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    /**
     * @brief AVX-512 Q4_K / IQ4_XS tile under one fixed FP32 policy and K tree.
     * @tparam Codebook 5 (Q4_K) or 4 (IQ4_XS).
     * @tparam Rows Rows sharing each decoded weight vector.
     * @tparam Policy Prepared numerical policy.
     *
     * Traversal: chunk row -> source superblock -> block -> 16-column strip, so
     * the chunk row streams once in order. Each strip widens its superblock d
     * once and keeps it in registers for the superblock's blocks; the codec
     * then combines it with each block's code exactly as reconstruct<5>()
     * (with q4KBase()) and sourceScaleFactors() do. Every column still visits
     * ascending K with its own accumulator, so the bytes are unchanged.
     */
    template <int Codebook, std::size_t Rows, Reduction Tree, CPUProjectionNumericalPolicy Policy, typename Packed>
    LLAMINAR_MULTISCALE_EXACT inline void nibblePolicy(const Packed &packed,
        const std::array<const Q8_1Block *, Rows> &activation, const std::array<float *, Rows> &output, int chunk,
        int begin, int end, bool accumulate, int partitions, int blocks_per_partition, __m512i decode_lut, int lut_bias)
    {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
        using Layout = SingleScaleLayout<Codebook>;
        static_assert(Layout::nibbles);
        using Ops = SIMD<16>;
        using I = typename Ops::I;
        using F = typename Ops::F;
        using L = coded_metadata::Lanes<Ops>;
        constexpr int strips = Rows == 1 ? 4 : 1;
        constexpr auto encoding = Layout::encoding;
        constexpr std::size_t data_stride = preparedDataStride(encoding, Codebook);
        constexpr std::size_t block_stride = preparedInterleavedBlockStride(encoding, Layout::affine, Codebook);
        // IQ4_XS LUT bytes are lut_bias + w; Q4_K nibbles are unsigned already.
        const int center = Codebook == 4 ? lut_bias : 0;
        const auto geometry = preparedBlockGeometry(encoding, Layout::affine, Codebook,
                                                    static_cast<std::size_t>(packed.blocks_per_row));
        const std::uint8_t *const row_base = packed.interleavedBase() + geometry.blockOffset(chunk, 0);
        for (int pass = 0; pass < 64; pass += 16 * strips)
        {
            F reduced[Rows][strips];
            // Ordered trees start every partition at positive zero and fold it
            // into the earlier ones in ascending order, exactly as rows() and
            // the nibble ordered kernels do. Contiguous K is one partition.
            const int partition_count = Tree == Reduction::OrderedPartitions ? partitions : 1;
            for (int partition = 0; partition < partition_count; ++partition)
            {
                F acc[Rows][strips];
#pragma GCC unroll 4
                for (std::size_t row = 0; row < Rows; ++row)
#pragma GCC unroll 4
                    for (int strip = 0; strip < strips; ++strip)
                        acc[row][strip] = Tree == Reduction::Contiguous && accumulate
                            ? load<F>(output[row] + pass + strip * 16) : F{};
                const int first = Tree == Reduction::OrderedPartitions ? begin + partition * blocks_per_partition : begin;
                const int last = Tree == Reduction::OrderedPartitions ? std::min(first + blocks_per_partition, end) : end;
                for (int kb = first; kb < last;)
                {
                    const int superblock_end = std::min(last, static_cast<int>(geometry.superblockEnd(kb)));
                    const std::uint8_t *const header = row_base + geometry.headerOffset(0, kb);
                    const std::uint8_t *const first_unit = row_base + geometry.blockOffset(0, kb);
                    // Widen every strip's superblock d once; it stays in registers
                    // for the superblock's blocks. Q4_K's dmin is re-widened per
                    // block from the L1-resident header: four more hoisted vectors
                    // would not fit beside the accumulators and would spill.
                    F base[strips];
    #pragma GCC unroll 4
                    for (int strip = 0; strip < strips; ++strip)
                    {
                        if constexpr (Codebook == 5)
                            base[strip] = L::half(L::template superblockWords<5>(header, pass + strip * 16));
                        else
                            base[strip] = Packed::template iq4xsBases<16>(header, pass + strip * 16);
                    }
                    const std::uint8_t *unit = first_unit;
                    for (int k = kb; k < superblock_end; ++k, unit += block_stride)
                    {
    #pragma GCC unroll 4
                        for (int strip = 0; strip < strips; ++strip)
                        {
                            const int column = pass + strip * 16;
                            I dots[Rows]{};
                            // Groups g and g + 4 share nibble word g: visit them together.
                            dotGroups<16, Rows>(std::integer_sequence<int, 0, 4, 1, 5, 2, 6, 3, 7>{}, activation, k,
                                [&]<int Group>() {
                                    const I raw = load<I>(unit + (Group % 4) * 256 + column * 4);
                                    const I nibble = (Group < 4 ? raw : raw >> 4) & 0x0F0F0F0F;
                                    if constexpr (Codebook == 4)
                                        return std::bit_cast<I>(_mm512_shuffle_epi8(decode_lut, std::bit_cast<__m512i>(nibble)));
                                    else
                                        return nibble;
                                }, dots);
                            const std::uint8_t *const planes = unit + data_stride;
                            F scale0, scale1{};
                            if constexpr (Codebook == 5)
                            {
                                const std::uint8_t *block_header = header;
                                asm("" : "+r"(block_header));
                                const coded_metadata::Q4KBaseFactors<F> q4k{
                                    base[strip], L::half(L::template superblockWords<5>(block_header, column) >> 16)};
                                const auto factors = coded_metadata::reconstruct<5, Ops>(
                                    Packed::template q4kCodes<16>(planes, column), q4k);
                                scale0 = factors.primary;
                                scale1 = factors.secondary;
                            }
                            else
                                scale0 = coded_metadata::sourceScaleFactors<kNativeQuantisedIQ4XSSourceScaleCodebook, Ops>(
                                    Packed::template iq4xsCodes<16>(planes, column), base[strip]).primary;
    #pragma GCC unroll 4
                            for (std::size_t row = 0; row < Rows; ++row)
                            {
                                const Q8_1Block &a = activation[row][k];
                                const auto dot = dots[row] - center * static_cast<int>(a.sum_qs);
                                const float scale_a = nativeQuantisedFP16ScaleToFP32(a.d);
                                if constexpr (Layout::affine)
                                    acc[row][strip] = std::bit_cast<F>(accumulateCorrectedSingleScaleBlockAVX512(
                                        std::bit_cast<__m512>(acc[row][strip]), std::bit_cast<__m512i>(dot),
                                        std::bit_cast<__m512>(scale0), a.sum_qs, std::bit_cast<__m512>(scale1), scale_a, Policy));
                                else
                                    acc[row][strip] = std::bit_cast<F>(accumulateSingleScaleBlockAVX512(
                                        std::bit_cast<__m512>(acc[row][strip]), std::bit_cast<__m512i>(dot),
                                        std::bit_cast<__m512>(scale0), scale_a, Policy));
                            }
                        }
                    }
                    kb = superblock_end;
                }
#pragma GCC unroll 4
                for (std::size_t row = 0; row < Rows; ++row)
#pragma GCC unroll 4
                    for (int strip = 0; strip < strips; ++strip)
                        reduced[row][strip] = partition == 0 ? acc[row][strip] : foldOrdered<16>(reduced[row][strip], acc[row][strip]);
            }
#pragma GCC unroll 4
            for (std::size_t row = 0; row < Rows; ++row)
#pragma GCC unroll 4
                for (int strip = 0; strip < strips; ++strip)
                    std::memcpy(output[row] + pass + strip * 16, &reduced[row][strip], sizeof(F));
        }
    }

#endif

    /**
     * @brief Validate a requested K interval and, for ordered trees, its exact partition geometry.
     * @throws std::invalid_argument For a chunk, K interval or partition tree outside the admitted matrix.
     *
     * The same admission as rows(): empty trailing partitions of a short K are
     * legal and contribute positive zero.
     */
    template <Reduction Tree, typename Packed>
    inline void requireSingleScaleTile(const Packed &packed, int chunk, int begin, int end, int partitions,
                                       int blocks_per_partition)
    {
        if (chunk < 0 || chunk >= packed.N_padded / 64 || (end > begin && (begin < 0 || end > packed.blocks_per_row)))
            throw std::invalid_argument("Multi-scale CPU tile is outside its admitted geometry");
        if constexpr (Tree == Reduction::OrderedPartitions)
            if (partitions <= 0 || blocks_per_partition <= 0 ||
                static_cast<std::int64_t>(partitions) * blocks_per_partition < end - begin ||
                static_cast<std::int64_t>(begin) + static_cast<std::int64_t>(partitions) * blocks_per_partition > INT32_MAX)
                throw std::invalid_argument("Multi-scale CPU tile requires an exact bounded ordered K tree");
    }

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    /**
     * @brief AVX-512 Q4_K (5) or IQ4_XS (4) superblock-coded nibble tile.
     * @tparam Tree Contiguous K or the admitted ordered partition tree.
     * @param packed Admitted Q4KSuperblockCoded or NibbleLUTSourceScale matrix.
     * @param activation One persistent Q8_1 row per tile row.
     * @param output Sixty-four writable FP32 columns per row.
     * @param chunk Output-column chunk.
     * @param begin,end Logical K-block interval.
     * @param accumulate Start from the existing output instead of positive zero (Contiguous only).
     * @param partitions,blocks_per_partition Exact ordered tree; one and zero for Contiguous.
     * @param decode_lut The caller's broadcast nibble table, biased by @p lut_bias.
     * @param lut_bias Unsigned zero point of @p decode_lut, removed via the activation sum.
     * @throws std::invalid_argument For a mismatched encoding, geometry or tree.
     *
     * Contiguous tiles follow the prepared policy; the ordered tree always
     * applies the GPU-aligned expert program, as the generic ordered kernels do.
     */
    template <int Codebook, std::size_t Rows, Reduction Tree = Reduction::Contiguous, typename Packed>
    inline void nibbleSingleScaleRows(const Packed &packed, const std::array<const Q8_1Block *, Rows> &activation,
                                      const std::array<float *, Rows> &output, int chunk, int begin, int end,
                                      bool accumulate, __m512i decode_lut, int lut_bias, int partitions = 1,
                                      int blocks_per_partition = 0)
    {
        static_assert(Codebook == 4 || Codebook == 5);
        static_assert(Rows >= 1 && Rows <= 4);
        if (packed.encoding != SingleScaleLayout<Codebook>::encoding)
            throw std::invalid_argument("x86 coded nibble tile requires its admitted encoding");
        requireSingleScaleTile<Tree>(packed, chunk, begin, end, partitions, blocks_per_partition);
        // The ordered nibble tree is by definition the GPU-aligned expert
        // program, exactly like gemv_native_avx512_ordered_partitions().
        if (Tree == Reduction::OrderedPartitions ||
            packed.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
            return nibblePolicy<Codebook, Rows, Tree, CPUProjectionNumericalPolicy::GPUAlignedExpert>(
                packed, activation, output, chunk, begin, end, accumulate, partitions, blocks_per_partition,
                decode_lut, lut_bias);
        return nibblePolicy<Codebook, Rows, Tree, CPUProjectionNumericalPolicy::BackendNative>(
            packed, activation, output, chunk, begin, end, accumulate, partitions, blocks_per_partition,
            decode_lut, lut_bias);
    }
#endif

    /**
     * @brief x86 Q5_K (7) or IQ3_S (11) tile; fixes the policy before the traversal.
     * @tparam Tree Contiguous K or the admitted ordered partition tree.
     * @param partitions,blocks_per_partition Exact ordered tree; one and zero for Contiguous.
     * @throws std::invalid_argument For a chunk, nonempty K interval or tree outside the matrix.
     */
    template <int Codebook, int Lanes, std::size_t Rows, Reduction Tree, typename Packed>
    inline void singleScaleRows(const Packed &packed, const std::array<const Q8_1Block *, Rows> &activation,
                                const std::array<float *, Rows> &output, int chunk, int begin, int end, bool accumulate,
                                int partitions, int blocks_per_partition)
    {
        static_assert(Rows >= 1 && Rows <= 4);
        requireSingleScaleTile<Tree>(packed, chunk, begin, end, partitions, blocks_per_partition);
        if (packed.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
            return singleScalePolicy<Codebook, Lanes, Rows, Tree, CPUProjectionNumericalPolicy::GPUAlignedExpert>(
                packed, activation, output, chunk, begin, end, accumulate, partitions, blocks_per_partition);
        return singleScalePolicy<Codebook, Lanes, Rows, Tree, CPUProjectionNumericalPolicy::BackendNative>(
            packed, activation, output, chunk, begin, end, accumulate, partitions, blocks_per_partition);
    }
}

#endif // __AVX2__
