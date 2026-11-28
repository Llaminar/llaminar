/**
 * @file CPUNativeQuantisedQ5X86Kernels.h
 * @brief AVX2 and AVX-512 projection of 20-byte Q5_0/Q5_1 payloads.
 *
 * Both ISAs decode the same four paired K groups into unsigned [0,31]
 * bytes and dot against signed Q8 bytes. Q5_0 then subtracts sixteen times
 * the actual activation sum in INT32; neither format stores weight sums.
 * The surrounding FP32 block program is the existing per-policy contract.
 * Projection walks each chunk row once per register-bounded group of column
 * strips with all K groups unrolled, so the 22-byte-per-32-value stream stays
 * sequential and stride-prefetchable at file density. The AVX2 one-row tile
 * walks L2-sized K tiles per strip (avx2OneRowKTileBlocks()).
 */
#pragma once

#include <algorithm>
#include <bit>
#include <utility>

#include <stdexcept>

#include "CPUNativeQuantisedContributionContract.h"
#include "CPUNativeQuantisedMultiScaleKernels.h"
#include "backends/CPUExecutionGeometry.h"
#include "CPUNativeQuantisedPackedWeights.h"

namespace llaminar2::cpu::native_quantised::q5
{
    /** @brief Load low nibbles and expand the one-byte fifth-bit pairs for eight columns. */
    [[nodiscard]] inline auto decode8(const std::uint8_t *unit, int column, int group) noexcept
    {
        using Ops = multi_scale::SIMD<8>;
        using I = Ops::I;
        const auto nibbles = multi_scale::load<__m256i>(unit + Q5NativePayload::nibbleOffset(group, column));
        const auto fifth = _mm256_srl_epi32(
            multi_scale::load<__m256i>(unit + Q5NativePayload::fifthOffset(0, column)), _mm_cvtsi32_si128(group * 8));
        const auto bits = std::bit_cast<__m256i>(Ops::repeatByte<0>(std::bit_cast<I>(fifth)));
        const auto unpack = [](__m256i raw, __m256i fifth_bits) {
            const auto mask = _mm256_set1_epi32(0x08040201);
            const auto missing = _mm256_cmpeq_epi8(_mm256_and_si256(fifth_bits, mask), _mm256_setzero_si256());
            return _mm256_or_si256(_mm256_and_si256(raw, _mm256_set1_epi8(15)),
                                   _mm256_andnot_si256(missing, _mm256_set1_epi8(16)));
        };
        // Shift the second set of fifth bits into the same mask positions,
        // reducing the constants competing with the live row accumulators.
        return std::pair{std::bit_cast<I>(unpack(nibbles, bits)),
                         std::bit_cast<I>(unpack(_mm256_srli_epi16(nibbles, 4), _mm256_srli_epi16(bits, 4)))};
    }

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    /** @brief Load low nibbles and expand the one-byte fifth-bit pairs for sixteen columns. */
    [[nodiscard]] inline auto decode16(const std::uint8_t *unit, int column, int group) noexcept
    {
        using Ops = multi_scale::SIMD<16>;
        using I = Ops::I;
        const auto nibbles = multi_scale::load<__m512i>(unit + Q5NativePayload::nibbleOffset(group, column));
        const auto fifth = _mm512_srl_epi32(
            multi_scale::load<__m512i>(unit + Q5NativePayload::fifthOffset(0, column)), _mm_cvtsi32_si128(group * 8));
        const auto bits = std::bit_cast<__m512i>(Ops::repeatByte<0>(std::bit_cast<I>(fifth)));
        const auto unpack = [&]<int Shift>(__m512i raw) {
            const auto mask = _mm512_set1_epi32(static_cast<int>(0x08040201u << Shift));
            const auto low = _mm512_and_si512(raw, _mm512_set1_epi8(15));
            return _mm512_mask_add_epi8(low, _mm512_test_epi8_mask(bits, mask), low, _mm512_set1_epi8(16));
        };
        return std::pair{std::bit_cast<I>(unpack.template operator()<0>(nibbles)),
                         std::bit_cast<I>(unpack.template operator()<4>(_mm512_srli_epi16(nibbles, 4)))};
    }
#endif

    /** @brief Broadcast one signed activation four-byte group and append its exact dot. */
    template <int Lanes>
    [[nodiscard]] inline typename multi_scale::SIMD<Lanes>::I dot(
        typename multi_scale::SIMD<Lanes>::I acc, typename multi_scale::SIMD<Lanes>::I weights,
        const Q8_1Block &activation, int group) noexcept
    {
        const auto word = multi_scale::load<int>(activation.qs + group * 4);
        using I = typename multi_scale::SIMD<Lanes>::I;
        if constexpr (Lanes == 8)
        {
            // 2*31*128 = 7936, so VPMADDUBSW never saturates, including for -128 activations.
            const auto pair = _mm256_maddubs_epi16(std::bit_cast<__m256i>(weights), _mm256_set1_epi32(word));
            return acc + std::bit_cast<I>(_mm256_madd_epi16(pair, _mm256_set1_epi16(1)));
        }
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        else
            return multi_scale::SIMD<16>::dot(acc, weights, std::bit_cast<I>(_mm512_set1_epi32(word)));
#endif
    }

    /** @brief Sum all original Q8 bytes for the Q5_0 zero-point correction.
     * @param qs Full 32-byte signed activation payload (including -128).
     * @return The exact signed sum, independently of any staged FP32 metadata.
     *
     * The unsigned byte sum counts every negative byte as value + 256, so the
     * signed sum subtracts 256 per set sign bit. Only a zero register is
     * needed, which leaves the vector registers to the live accumulators.
     */
    [[nodiscard]] inline int sumQ8(const std::int8_t *qs) noexcept
    {
        const auto input = multi_scale::load<__m256i>(qs);
        const auto sums = _mm256_sad_epu8(input, _mm256_setzero_si256());
        const auto pairs = _mm_add_epi64(_mm256_castsi256_si128(sums), _mm256_extracti128_si256(sums, 1));
        const int unsigned_sum = _mm_cvtsi128_si32(_mm_add_epi64(pairs, _mm_srli_si128(pairs, 8)));
        return unsigned_sum - 256 * std::popcount(static_cast<std::uint32_t>(_mm256_movemask_epi8(input)));
    }

    /** @brief Apply the unchanged FP32 contribution program on either x86 width. */
    template <int Lanes, bool Asymmetric>
    [[nodiscard]] inline typename multi_scale::SIMD<Lanes>::F contribute(
        typename multi_scale::SIMD<Lanes>::F acc, typename multi_scale::SIMD<Lanes>::I dot,
        const std::uint16_t *d, const std::uint16_t *m, const Q8_1Block &a,
        CPUProjectionNumericalPolicy policy) noexcept
    {
        using F = typename multi_scale::SIMD<Lanes>::F;
        const float a_scale = nativeQuantisedFP16ScaleToFP32(a.d);
        if constexpr (!Asymmetric)
            dot -= 16 * sumQ8(a.qs);
        if constexpr (Lanes == 8)
        {
            const auto scale = _mm256_cvtph_ps(multi_scale::load<__m128i>(d));
            if constexpr (Asymmetric)
                return std::bit_cast<F>(accumulateCorrectedSingleScaleBlockAVX2(std::bit_cast<__m256>(acc),
                    std::bit_cast<__m256i>(dot), scale, a.sum_qs, _mm256_cvtph_ps(multi_scale::load<__m128i>(m)),
                    a_scale, policy));
            return std::bit_cast<F>(accumulateSingleScaleBlockAVX2(std::bit_cast<__m256>(acc),
                std::bit_cast<__m256i>(dot), scale, a_scale, policy));
        }
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        else
        {
            const auto scale = _mm512_cvtph_ps(multi_scale::load<__m256i>(d));
            if constexpr (Asymmetric)
                return std::bit_cast<F>(accumulateCorrectedSingleScaleBlockAVX512(std::bit_cast<__m512>(acc),
                    std::bit_cast<__m512i>(dot), scale, a.sum_qs, _mm512_cvtph_ps(multi_scale::load<__m256i>(m)),
                    a_scale, policy));
            return std::bit_cast<F>(accumulateSingleScaleBlockAVX512(std::bit_cast<__m512>(acc),
                std::bit_cast<__m512i>(dot), scale, a_scale, policy));
        }
#endif
    }

    /**
     * @brief Number of SIMD column strips that share one K traversal.
     * @tparam Lanes Eight AVX2 or sixteen AVX-512 output columns per strip.
     * @tparam Rows Live activation rows of the tile.
     *
     * Visiting several strips per block streams each 64-column unit
     * contiguously and decodes each activation block once for all of them,
     * instead of re-walking the whole K range once per strip. The count is
     * bounded by the integer and FP32 accumulators that must stay in
     * registers: AVX-512 has 32 vector registers, AVX2 only 16.
     */
    template <int Lanes, std::size_t Rows>
    inline constexpr int kStripsPerPass = Lanes == 16 ? (Rows == 1 ? 4 : Rows == 2 ? 2 : 1) : 1;

    /**
     * @brief Expand one compile-time K group of an eight-column strip.
     * @tparam Group Low-nibble group in [0,3]; its high nibbles form group + 4.
     * @param unit First byte of the 64-column Q5 block.
     * @tparam Aligned Select the group's byte from the strip's aligned fifth-bit
     *         vector @p fifth instead of a shifted load.
     * @param column First column of the strip.
     * @param fifth The strip's aligned fifth-bit plane (Aligned only).
     * @return Exact unsigned five-bit weights of groups @p Group and @p Group + 4.
     *
     * The same bytes as decode8(). Loading the fifth-bit plane @p Group
     * bytes later puts this group's byte first in every column word, so one
     * replicating shuffle serves all groups and no fifth-bit vector stays
     * live across groups; the load stays inside the unit (its scale plane
     * follows). This keeps the sixteen AVX2 registers for the accumulators.
     */
    template <int Group, bool Aligned = false>
    [[nodiscard, gnu::always_inline]] inline std::pair<__m256i, __m256i> decodeGroup8(
        const std::uint8_t *unit, int column, [[maybe_unused]] __m256i fifth = {}) noexcept
    {
        using Ops = multi_scale::SIMD<8>;
        const auto nibbles = multi_scale::load<__m256i>(unit + Q5NativePayload::nibbleOffset(Group, column));
        const auto bits = [&] {
            if constexpr (Aligned)
                return std::bit_cast<__m256i>(Ops::repeatByte<Group>(std::bit_cast<Ops::I>(fifth)));
            else
                return std::bit_cast<__m256i>(Ops::repeatByte<0>(
                    multi_scale::load<Ops::I>(unit + Q5NativePayload::fifthOffset(Group, column))));
        }();
        // A byte whose tested fifth bit is set gets sixteen, exactly as in
        // decode8(). The high half reuses the low mask after a word shift
        // (the bits shifted across bytes land outside the tested low nibble).
        // Byte-typed vector operations keep the selector in one register:
        // mixing 64-bit and byte intrinsic modes makes GCC copy it.
        using Bytes = std::int8_t __attribute__((vector_size(32)));
        const auto selector = std::bit_cast<Bytes>(_mm256_set1_epi32(0x08040201));
        const auto unpack = [&](__m256i raw, __m256i fifth_bits) {
            const Bytes set = (std::bit_cast<Bytes>(fifth_bits) & selector) == selector;
            return std::bit_cast<__m256i>((std::bit_cast<Bytes>(raw) & 15) | (set & 16));
        };
        return {unpack(nibbles, bits), unpack(_mm256_srli_epi16(nibbles, 4), _mm256_srli_epi16(bits, 4))};
    }

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    /**
     * @brief Expand one compile-time K group of a sixteen-column strip.
     * @tparam Group Low-nibble group in [0,3]; its high nibbles form group + 4.
     * @param unit First byte of the 64-column Q5 block.
     * @param column First column of the strip.
     * @param fifth The strip's fifth-bit bytes, loaded once per block.
     * @return Exact unsigned five-bit weights of groups @p Group and @p Group + 4, as decode16().
     */
    template <int Group>
    [[nodiscard, gnu::always_inline]] inline std::pair<__m512i, __m512i> decodeGroup16(
        const std::uint8_t *unit, int column, __m512i fifth) noexcept
    {
        const auto nibbles = multi_scale::load<__m512i>(unit + Q5NativePayload::nibbleOffset(Group, column));
        const auto bits = _mm512_shuffle_epi8(fifth, _mm512_add_epi8(
            _mm512_set1_epi8(static_cast<char>(Group)), _mm512_set4_epi32(0x0c0c0c0c, 0x08080808, 0x04040404, 0)));
        // Each half is (nibble & 15) | (fifth bit ? 16 : 0), the bytes of
        // decode16(): one zero-masked byte move materializes the sixteens
        // and one ternary-logic op merges them (A | (B & C), imm 0xF8),
        // with no register copy for a merge-masked destination.
        const auto sixteen = _mm512_set1_epi8(16);
        const auto fifteen = _mm512_set1_epi8(15);
        const auto merge = [&](__m512i raw, __m512i mask) {
            const auto high_bit = _mm512_maskz_mov_epi8(_mm512_test_epi8_mask(bits, mask), sixteen);
            return _mm512_ternarylogic_epi32(high_bit, raw, fifteen, 0xF8);
        };
        return {merge(nibbles, _mm512_set1_epi32(0x08040201)),
                merge(_mm512_srli_epi16(nibbles, 4), _mm512_set1_epi32(static_cast<int>(0x80402010u)))};
    }
#endif

    /**
     * @brief Accumulate a bounded K range while sharing one weight decode across live rows.
     * @tparam Lanes Physical AVX2 or AVX-512 output width.
     * @tparam Asymmetric Q5_1 minimum term, instead of Q5_0 centering.
     * @tparam SumPairs Publish the existing two-row fused epilogue sum in the corresponding even row.
     * @tparam Policy Prepared FP32 contribution program, fixed before the K traversal.
     * @tparam Rows Number of live activation and output rows.
     * @param packed Admitted native Q5 publication.
     * @param input Activation rows with complete Q8 blocks.
     * @param output Writable physical chunk per row; callers own any N-tail handling.
     * @param chunk Physical 64-column output chunk.
     * @param first First logical K block in this range.
     * @param last One past the last block in this range.
     * @param accumulate Read existing output accumulators instead of starting from positive zero.
     *
     * Each column keeps the original program: an exact INT32 dot over the
     * block's 32 values, then the unchanged per-policy FP32 contribution in
     * ascending K. Only the physical traversal changes. kStripsPerPass strips
     * share one contiguous walk of the chunk row with all four groups
     * unrolled, so every load has a fixed stride across blocks (the hardware
     * stride prefetchers track it) and the activation scale and Q5_0 sum
     * are computed once per row and block instead of once per strip.
     */
    template <int Lanes, bool Asymmetric, bool SumPairs, CPUProjectionNumericalPolicy Policy, std::size_t Rows>
    inline void projectRange(const CPUNativeQuantisedPackedWeights &packed,
        const std::array<const Q8_1Block *, Rows> &input, const std::array<float *, Rows> &output,
        int chunk, int first, int last, bool accumulate)
    {
        static_assert(!SumPairs || Rows % 2 == 0);
        using Ops = multi_scale::SIMD<Lanes>;
        using I = typename Ops::I;
        using F = typename Ops::F;
        constexpr int strips = kStripsPerPass<Lanes, Rows>;
        constexpr auto codebook = Asymmetric ? native_quantised_formats::Q5_1.codebook_id
                                             : native_quantised_formats::Q5_0.codebook_id;
        // The leaf is bound to Q5NativeSingleScale: one fixed per-block unit
        // (payload, FP16 d, optional FP16 m) and no superblock headers.
        const auto geometry = preparedBlockGeometry(CPUNativeQuantisedEncoding::Q5NativeSingleScale, Asymmetric,
                                                    codebook, static_cast<std::size_t>(packed.blocks_per_row));
        constexpr std::size_t stride =
            preparedInterleavedBlockStride(CPUNativeQuantisedEncoding::Q5NativeSingleScale, Asymmetric, codebook);
        const std::uint8_t *const chunk_base = packed.interleavedBase() + geometry.blockOffset(chunk, 0);
        for (int pass = 0; pass < 64; pass += Lanes * strips)
        {
            F partial[Rows][strips]{};
            // Fully unrolled: a runtime row index would keep the array in memory.
            if (accumulate)
#pragma GCC unroll 4
                for (std::size_t row = 0; row < Rows; ++row)
#pragma GCC unroll 8
                    for (int strip = 0; strip < strips; ++strip)
                        partial[row][strip] = multi_scale::load<F>(output[row] + pass + strip * Lanes);
            const std::uint8_t *unit = chunk_base + static_cast<std::size_t>(first) * stride;
            for (int kb = first; kb < last; ++kb, unit += stride)
            {
                I dots[Rows][strips]{};
                if constexpr (Lanes == 8)
                {
                    // Two groups' four unsigned-by-signed pair sums stay in
                    // INT16: each is at most 2*31*128 = 7936 in magnitude, so
                    // four sum to at most 31744 and never wrap. One widening
                    // per group pair then feeds the unchanged INT32 dot. Wider
                    // row tiles widen every group: their extra live INT16 sums
                    // would not fit the sixteen AVX2 registers.
                    constexpr bool defer = Rows == 1;
                    const auto pair = [&]<int G>() {
#pragma GCC unroll 8
                        for (int strip = 0; strip < strips; ++strip)
                        {
                            const int column = pass + strip * Lanes;
                            [[maybe_unused]] __m256i sums[Rows];
                            // One-row tiles cover the whole block: load the strip's
                            // fifth-bit plane once, aligned, without line splits.
                            [[maybe_unused]] __m256i fifth{};
                            if constexpr (Rows == 1)
                                fifth = multi_scale::load<__m256i>(unit + Q5NativePayload::fifthOffset(0, column));
                            // Decode one group at a time; only its two vectors are live.
                            const auto group = [&]<int Group>() {
                                const auto [low, high] = decodeGroup8<Group, Rows == 1>(unit, column, fifth);
#pragma GCC unroll 4
                                for (std::size_t row = 0; row < Rows; ++row)
                                {
                                    const auto word = [&](int g) {
                                        return _mm256_set1_epi32(multi_scale::load<int>(input[row][kb].qs + g * 4));
                                    };
                                    const auto sum = _mm256_add_epi16(_mm256_maddubs_epi16(low, word(Group)),
                                                                      _mm256_maddubs_epi16(high, word(Group + 4)));
                                    if constexpr (!defer)
                                        dots[row][strip] += std::bit_cast<I>(_mm256_madd_epi16(sum, _mm256_set1_epi16(1)));
                                    else
                                        sums[row] = Group == G ? sum : _mm256_add_epi16(sums[row], sum);
                                }
                            };
                            group.template operator()<G>();
                            group.template operator()<G + 1>();
                            if constexpr (defer)
#pragma GCC unroll 4
                                for (std::size_t row = 0; row < Rows; ++row)
                                    dots[row][strip] += std::bit_cast<I>(_mm256_madd_epi16(sums[row], _mm256_set1_epi16(1)));
                        }
                    };
                    pair.template operator()<0>();
                    pair.template operator()<2>();
                }
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
                else
                {
                    const auto group = [&]<int G>() {
#pragma GCC unroll 8
                        for (int strip = 0; strip < strips; ++strip)
                        {
                            const int column = pass + strip * Lanes;
                            const auto fifth = multi_scale::load<__m512i>(unit + Q5NativePayload::fifthOffset(0, column));
                            const auto [low, high] = decodeGroup16<G>(unit, column, fifth);
#pragma GCC unroll 4
                            for (std::size_t row = 0; row < Rows; ++row)
                            {
                                dots[row][strip] = dot<Lanes>(dots[row][strip], std::bit_cast<I>(low), input[row][kb], G);
                                dots[row][strip] = dot<Lanes>(dots[row][strip], std::bit_cast<I>(high), input[row][kb], G + 4);
                            }
                        }
                    };
                    group.template operator()<0>();
                    group.template operator()<1>();
                    group.template operator()<2>();
                    group.template operator()<3>();
                }
#endif
                // Inline FP16 d (and Q5_1 m) planes follow the payload.
                const auto *d = reinterpret_cast<const std::uint16_t *>(unit + Q5NativePayload::bytes) + pass;
#pragma GCC unroll 4
                for (std::size_t row = 0; row < Rows; ++row)
                {
                    const Q8_1Block &a = input[row][kb];
                    const float a_scale = nativeQuantisedFP16ScaleToFP32(a.d);
                    // Q5_0 centering: subtract sixteen times the exact signed
                    // activation sum from the unsigned-weight dot, in INT32.
                    [[maybe_unused]] int center = 0;
                    if constexpr (!Asymmetric)
                        center = 16 * sumQ8(a.qs);
#pragma GCC unroll 8
                    for (int strip = 0; strip < strips; ++strip)
                    {
                        const auto *scale_words = d + strip * Lanes;
                        if constexpr (Lanes == 8)
                        {
                            const auto scale = _mm256_cvtph_ps(multi_scale::load<__m128i>(scale_words));
                            if constexpr (Asymmetric)
                                partial[row][strip] = std::bit_cast<F>(accumulateCorrectedSingleScaleBlockAVX2(
                                    std::bit_cast<__m256>(partial[row][strip]), std::bit_cast<__m256i>(dots[row][strip]), scale,
                                    a.sum_qs, _mm256_cvtph_ps(multi_scale::load<__m128i>(scale_words + 64)), a_scale, Policy));
                            else
                                partial[row][strip] = std::bit_cast<F>(accumulateSingleScaleBlockAVX2(
                                    std::bit_cast<__m256>(partial[row][strip]),
                                    std::bit_cast<__m256i>(dots[row][strip] - center), scale, a_scale, Policy));
                        }
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
                        else
                        {
                            const auto scale = _mm512_cvtph_ps(multi_scale::load<__m256i>(scale_words));
                            if constexpr (Asymmetric)
                                partial[row][strip] = std::bit_cast<F>(accumulateCorrectedSingleScaleBlockAVX512(
                                    std::bit_cast<__m512>(partial[row][strip]), std::bit_cast<__m512i>(dots[row][strip]), scale,
                                    a.sum_qs, _mm512_cvtph_ps(multi_scale::load<__m256i>(scale_words + 64)), a_scale, Policy));
                            else
                                partial[row][strip] = std::bit_cast<F>(accumulateSingleScaleBlockAVX512(
                                    std::bit_cast<__m512>(partial[row][strip]),
                                    std::bit_cast<__m512i>(dots[row][strip] - center), scale, a_scale, Policy));
                        }
#endif
                    }
                }
            }
#pragma GCC unroll 8
            for (int strip = 0; strip < strips; ++strip)
            {
                const int column = pass + strip * Lanes;
                if constexpr (SumPairs)
                {
#pragma GCC unroll 4
                    for (std::size_t row = 0; row < Rows; row += 2)
                    {
                        const F sum = partial[row][strip] + partial[row + 1][strip];
                        std::memcpy(output[row] + column, &sum, sizeof(F));
                    }
                }
                else
#pragma GCC unroll 4
                    for (std::size_t row = 0; row < Rows; ++row)
                        std::memcpy(output[row] + column, &partial[row][strip], sizeof(F));
            }
        }
    }

    /**
     * @brief K blocks of one AVX2 one-row Q5 cache tile, derived from the observed private L2.
     * @param cache The participant's cache geometry (CPUExecutionGeometry::local().cache in production).
     * @param asymmetric Q5_1 units carry a minimum plane (1536 instead of 1408 bytes).
     * @return Half of the associativity-safe L2 in whole units, at least 16 blocks.
     *
     * The eight strip passes of a tile re-read lines the first pass brought in.
     * An L1-sized tile (16 blocks) kept those re-reads in L1 but cut each
     * pass's strided stream to sixteen accesses, which measured slower from
     * DRAM (AVX2 Q5_0 M1 t1: 15.7 ms with 16-block tiles and 17.4 ms with the
     * 32-block GPU-aligned partitions, against 13.2 ms for whole-K passes at
     * 29c6e906a). An L2-resident tile keeps the long streams the hardware
     * prefetchers track while still bounding the re-read set when a chunk row
     * exceeds L2. The other half of L2 is left to activations, outputs and
     * the other strips' lines; an N-way L2 contributes N-1 ways
     * (associativitySafeBytes). The sixteen-block floor is the L1 tile.
     */
    [[nodiscard]] constexpr int avx2OneRowKTileBlocks(const CPUCacheGeometry &cache, bool asymmetric) noexcept
    {
        const std::uint64_t ways = cache.private_l2_ways;
        const std::uint64_t safe = ways > 1 ? cache.private_l2_bytes - cache.private_l2_bytes / ways
                                            : cache.private_l2_bytes;
        const std::uint64_t unit = preparedInterleavedBlockStride(CPUNativeQuantisedEncoding::Q5NativeSingleScale,
            asymmetric, asymmetric ? native_quantised_formats::Q5_1.codebook_id : native_quantised_formats::Q5_0.codebook_id);
        return static_cast<int>(std::clamp<std::uint64_t>(safe / 2 / unit, 16, 1u << 20));
    }

    /**
     * @brief Run projectRange(), cache-tiling K for the AVX2 one-row decode tile.
     * @tparam Lanes Physical AVX2 or AVX-512 output width.
     * @tparam Asymmetric Q5_1 minimum term, instead of Q5_0 centering.
     * @tparam SumPairs Publish the two-row fused epilogue sum in the corresponding even row.
     * @tparam Policy Prepared FP32 contribution program, fixed before the K traversal.
     * @tparam Rows Number of live activation and output rows.
     * @param packed,input,output,chunk,first,last,accumulate As projectRange().
     * @param tile_blocks Positive K blocks per AVX2 one-row tile (avx2OneRowKTileBlocks()).
     *
     * AVX-512 one-row tiles already cover all 64 columns of a block in one
     * pass. Sixteen YMM registers cannot hold eight strips' accumulators, so
     * the AVX2 one-row tile instead visits @p tile_blocks blocks for each of
     * its eight strips before moving on: the first pass streams the tile's
     * lines in order and the others re-read them from cache. Each strip
     * carries its FP32 accumulator across tiles in its own output columns (an
     * exact binary32 store and reload), so every column still accumulates
     * the same contributions in ascending K with unchanged rounding; the tile
     * length never changes a byte. Paired-sum tiles publish only a row sum
     * and keep the single walk.
     */
    template <int Lanes, bool Asymmetric, bool SumPairs, CPUProjectionNumericalPolicy Policy, std::size_t Rows>
    inline void projectPolicy(const CPUNativeQuantisedPackedWeights &packed,
        const std::array<const Q8_1Block *, Rows> &input, const std::array<float *, Rows> &output,
        int chunk, int first, int last, bool accumulate, int tile_blocks)
    {
        if constexpr (Lanes == 8 && Rows == 1 && !SumPairs)
        {
            if (tile_blocks <= 0)
                throw std::invalid_argument("AVX2 Q5 K tiles require a positive block count");
            for (int begin = first; begin < last; begin += tile_blocks)
                projectRange<Lanes, Asymmetric, SumPairs, Policy>(packed, input, output, chunk, begin,
                    std::min(last, begin + tile_blocks), accumulate || begin != first);
            if (first < last)
                return;
        }
        projectRange<Lanes, Asymmetric, SumPairs, Policy>(packed, input, output, chunk, first, last, accumulate);
    }

    /**
     * @brief Select symmetric or asymmetric Q5 and the FP32 policy once before the K traversal.
     * @param tile_blocks AVX2 one-row K tile; zero derives it from the participant's observed cache.
     */
    template <int Lanes, bool SumPairs = false, std::size_t Rows>
    inline void dispatch(const CPUNativeQuantisedPackedWeights &packed,
        const std::array<const Q8_1Block *, Rows> &input, const std::array<float *, Rows> &output,
        int chunk, int first, int last, bool accumulate, int tile_blocks = 0)
    {
        using Policy = CPUProjectionNumericalPolicy;
        if (Lanes == 8 && Rows == 1 && !SumPairs && tile_blocks == 0)
            tile_blocks = avx2OneRowKTileBlocks(CPUExecutionGeometry::local().cache, packed.is_asymmetric);
        const bool aligned = packed.numerical_policy == Policy::GPUAlignedExpert;
        if (packed.is_asymmetric)
        {
            if (aligned)
                return projectPolicy<Lanes, true, SumPairs, Policy::GPUAlignedExpert>(packed, input, output, chunk,
                    first, last, accumulate, tile_blocks);
            return projectPolicy<Lanes, true, SumPairs, Policy::BackendNative>(packed, input, output, chunk, first,
                last, accumulate, tile_blocks);
        }
        if (aligned)
            return projectPolicy<Lanes, false, SumPairs, Policy::GPUAlignedExpert>(packed, input, output, chunk, first,
                last, accumulate, tile_blocks);
        return projectPolicy<Lanes, false, SumPairs, Policy::BackendNative>(packed, input, output, chunk, first, last,
            accumulate, tile_blocks);
    }
}
