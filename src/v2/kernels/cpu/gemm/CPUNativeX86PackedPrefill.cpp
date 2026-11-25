/**
 * @file CPUNativeX86PackedPrefill.cpp
 * @brief Decode native weights into a short worker panel and reuse across live rows.
 *
 * Every worker visits increasing K inside the exact M=1 partition tree. Pack
 * writes directly to the worker-owned slab. The next stage only reads that
 * bounded panel and writes final output (or an unfinished ordered partial).
 * No expanded weight publication or per-invocation allocation exists.
 *
 * Strip widths follow x86_panel::stripColumns(): AVX2 strips are eight
 * columns consumed two rows at a time; AVX-512 strips are a whole 64-column
 * chunk consumed two rows by four vectors (the shape of the expanded INT8
 * kernels, one broadcast and one per-row scale/correction per four dots)
 * whenever every worker owns one, else sixteen columns four rows at a time.
 *
 * GPU-aligned partitioned trees of complete strips whose partitions fit one panel pack whole
 * partitions and fold them in registers (consumeOrdered / consumeWideOrdered)
 * with the serial tree's exact ordered operations. Pooled members of the batched
 * expert route run the same strips as one task per 64-column chunk on the
 * calling worker's slot (projectPooledChunk).
 */
#if defined(__AVX2__)
#include "CPUNativeX86PackedPrefill.h"

#include <algorithm>
#include "CPUNativeAVX2Gemv.h"
#include "CPUProjectionStorageProof.h"

namespace llaminar2::cpu::native_quantised::x86_panel
{
    namespace
    {
        using Policy = CPUProjectionNumericalPolicy;
        template <int CB>
        inline constexpr CPUNativeQuantisedEncoding encoding =
            CB == 5   ? CPUNativeQuantisedEncoding::Q4KSuperblockCoded
            : CB == 6 ? CPUNativeQuantisedEncoding::Q5NativeSingleScale
            : CB == 4 ? CPUNativeQuantisedEncoding::NibbleLUTSourceScale
                      : CPUNativeQuantisedEncoding::CompactMultiScale;

        /**
         * @brief Decode one SIMD-wide sub-strip directly into the current worker's reserved panel.
         * @tparam Lanes Eight AVX2 or sixteen AVX-512 output columns per vector.
         * @tparam Subs Vectors per panel strip (one, or four for a 64-column AVX-512 strip).
         * @tparam Sub This call's vector within the strip.
         * @tparam CB Source factor and payload decoder.
         * @param w Admitted file-density publication, retained for the invocation.
         * @param chunk,col Physical original coordinates of this sub-strip's first column.
         * @param begin,end Increasing logical K-block interval inside one arithmetic partition.
         * @param out Worker slot, large enough for all intervals up to @ref depth.
         *
         * A block of the panel holds ten (or nine) strip-wide planes: the eight
         * dot groups, the scale and the minimum. Plane i of sub-strip Sub starts
         * at (i * Subs + Sub) * Lanes, so a one-vector strip keeps its original
         * i * Lanes layout and each consume load reads one contiguous vector.
         */
        template <int Lanes, int Subs, int Sub, int CB>
        [[gnu::noinline]]
        void pack(const CPUNativeQuantisedWeightView &w, int chunk, int col, int begin, int end, float *out)
        {
            using Ops = multi_scale::SIMD<Lanes>;
            using I = typename Ops::I;
            using L = coded_metadata::Lanes<Ops>;
            static_assert(Sub >= 0 && Sub < Subs);
            constexpr int stride = blockFloats(static_cast<Format>(CB), Lanes * Subs);
            constexpr auto plane = [](int index) { return (index * Subs + Sub) * Lanes; };
            const auto storeVector = [](float *where, auto value) { std::memcpy(where, &value, sizeof(value)); };
            constexpr auto geom = preparedBlockGeometry(encoding<CB>, affine(static_cast<Format>(CB)), CB, 0);
            constexpr auto data = preparedDataStride(encoding<CB>, CB);
            const auto geo = w.geometry();
            const auto *root = w.interleavedBase() + geo.rowBytes() * chunk;
            for (int k = begin; k < end;)
            {
                const auto next = CB == 6 ? end : std::min(end, int(geo.superblockEnd(k)));
                const auto *u = root + geo.blockOffset(0, k);
                typename Ops::F d{}, min{};
                if constexpr (CB != 6)
                {
                    constexpr int meta_cb = CB == 4 ? kNativeQuantisedIQ4XSSourceScaleCodebook : CB;
                    const auto header = L::template superblockWords<meta_cb>(root + geo.headerOffset(0, k), col);
                    d = L::half(header);
                    if constexpr (affine(static_cast<Format>(CB)))
                        min = L::half(header >> 16);
                }
                for (; k < next; ++k, u += geom.block_stride, out += stride)
                {
                    if constexpr (CB == 11)
                    {
#pragma GCC unroll 1
                        for (int g = 0; g < 8; ++g)
                            storeVector(out + plane(g), multi_scale::decodeIQ3SGroup<Lanes, Ops>(u, col, g));
                    }
                    else
                    {
#pragma GCC unroll 1
                        for (int g = 0; g < 4; ++g)
                        {
                            if constexpr (CB == 6)
                            {
                                if constexpr (Lanes == 8)
                                {
                                    auto [l, h] = q5::decode8(u, col, g);
                                    storeVector(out + plane(g), l);
                                    storeVector(out + plane(g + 4), h);
                                }
#if defined(__AVX512VNNI__)
                                else
                                {
                                    auto [l, h] = q5::decode16(u, col, g);
                                    storeVector(out + plane(g), l);
                                    storeVector(out + plane(g + 4), h);
                                }
#endif
                            }
                            else
                            {
                                const I raw = multi_scale::load<I>(u + g * 256 + col * 4);
                                auto store = [&](I v, int group)
                                {
                                    I low = v & 0x0f0f0f0f;
                                    if constexpr (CB == 4)
                                    {
                                        if constexpr (Lanes == 8)
                                            low = std::bit_cast<I>(_mm256_shuffle_epi8(
                                                build_decode_lut_avx2_for_codebook(4), std::bit_cast<__m256i>(low)));
#if defined(__AVX512VNNI__)
                                        else
                                            low = std::bit_cast<I>(_mm512_shuffle_epi8(
                                                _mm512_xor_si512(_mm512_broadcast_i32x4(
                                                                     _mm_load_si128(reinterpret_cast<const __m128i *>(
                                                                         avx2_luts::IQ4_NL_DECODE_LUT))),
                                                                 _mm512_set1_epi8(-128)),
                                                std::bit_cast<__m512i>(low)));
#endif
                                    }
                                    else if constexpr (CB == 7)
                                    {
                                        const auto *fifth = u + 1024 + group * 32 + col / 2;
                                        if constexpr (Lanes == 8)
                                            low |= Ops::template q5High<false>(fifth);
#if defined(__AVX512VNNI__)
                                        else
                                        {
                                            // The panel packer does not hold dot accumulators.
                                            // This form avoids spilling a wide add constant.
                                            const auto bits = multi_scale::load<std::uint64_t>(fifth);
                                            low |= std::bit_cast<I>(_mm512_maskz_set1_epi8(bits, 16));
                                        }
#endif
                                    }
                                    storeVector(out + plane(group), low);
                                };
                                store(raw, g);
                                store(raw >> 4, g + 4);
                            }
                        }
                    }
                    if constexpr (CB == 6)
                        storeVector(out + plane(8), Ops::half(reinterpret_cast<const uint16_t *>(u + data) + col));
                    else if constexpr (CB == 11)
                        storeVector(out + plane(8), d * L::toFloat(2 * L::nibbles(u + data, col) + 1));
                    else if constexpr (CB == 4)
                    {
                        constexpr auto key = kNativeQuantisedIQ4XSSourceScaleCodebook;
                        storeVector(out + plane(8), coded_metadata::sourceScaleFactors<key, Ops>(
                                                         L::template code<key>(u + data, col), d)
                                                         .primary);
                    }
                    else
                    {
                        const auto factors =
                            coded_metadata::reconstruct<5, Ops>(L::template code<CB>(u + data, col), {d, min});
                        storeVector(out + plane(8), factors.primary);
                        storeVector(out + plane(9), factors.secondary);
                    }
                }
            }
        }

        /** Load only live N columns. A partial N tail never borrows output padding. */
        template <int Lanes> inline typename multi_scale::SIMD<Lanes>::F loadOutput(const float *out, int live)
        {
            using F = typename multi_scale::SIMD<Lanes>::F;
            if constexpr (Lanes == 8)
            {
                const auto mask =
                    _mm256_cmpgt_epi32(_mm256_set1_epi32(live), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
                return std::bit_cast<F>(live == Lanes ? _mm256_loadu_ps(out) : _mm256_maskload_ps(out, mask));
            }
#if defined(__AVX512VNNI__)
            else
                return std::bit_cast<F>(live == Lanes ? _mm512_loadu_ps(out)
                                                      : _mm512_maskz_loadu_ps((1u << live) - 1u, out));
#endif
        }

        /** Publish live destination lanes directly, including when N is not a vector multiple. */
        template <int Lanes> inline void writeOutput(float *out, int live, typename multi_scale::SIMD<Lanes>::F value)
        {
            if constexpr (Lanes == 8)
            {
                const auto v = std::bit_cast<__m256>(value);
                if (live == Lanes)
                    _mm256_storeu_ps(out, v);
                else
                    _mm256_maskstore_ps(
                        out, _mm256_cmpgt_epi32(_mm256_set1_epi32(live), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)), v);
            }
#if defined(__AVX512VNNI__)
            else
            {
                const auto v = std::bit_cast<__m512>(value);
                if (live == Lanes)
                    _mm512_storeu_ps(out, v);
                else
                    _mm512_mask_storeu_ps(out, (1u << live) - 1u, v);
            }
#endif
        }

        /** Fold one completed K partition using the established per-policy arithmetic. */
        template <int Lanes, Policy P>
        inline typename multi_scale::SIMD<Lanes>::F fold(typename multi_scale::SIMD<Lanes>::F left,
                                                         typename multi_scale::SIMD<Lanes>::F right)
        {
            using F = typename multi_scale::SIMD<Lanes>::F;
            if constexpr (P == Policy::BackendNative)
                return left + right;
            else if constexpr (Lanes == 8)
                return std::bit_cast<F>(addOrderedAVX2(std::bit_cast<__m256>(left), std::bit_cast<__m256>(right)));
#if defined(__AVX512VNNI__)
            else
                return std::bit_cast<F>(addOrderedAVX512(std::bit_cast<__m512>(left), std::bit_cast<__m512>(right)));
#endif
        }

        /**
         * @brief Accumulate one panel block into every row of a register tile.
         * @tparam Lanes Physical SIMD columns per row.
         * @tparam CB File-density source decoder that wrote the panel.
         * @tparam Rows Independently accumulated rows of the tile.
         * @tparam P Original FP32 block policy.
         * @param acc Each row's running FP32 lanes, updated in place.
         * @param p This block's dot-group planes and widened factors in the worker's panel.
         * @param a First activation block of each row's panel interval.
         * @param i Block index inside the panel interval.
         * @details The per-block program of the register-folded GPU-aligned
         * leaf consumeOrdered(), the same source as consume()'s loop body:
         * exact INT32 dot, zero-point correction, then the policy's FP32 block
         * contribution. consume() keeps its own unchanged copy because a
         * backend-native block is a fused multiply-add whose NaN operand
         * (and so payload) the compiler's register allocation picks; moving
         * its code would change those payloads. The GPU-aligned contribution
         * is a fixed chain of rounded, operand-ordered operations.
         */
        template <int Lanes, int CB, int Rows, Policy P>
        [[gnu::always_inline]] inline void consumeBlock(std::array<typename multi_scale::SIMD<Lanes>::F, Rows> &acc,
                                                        const float *p, const std::array<const Q8_1Block *, Rows> &a,
                                                        int i)
        {
            using Ops = multi_scale::SIMD<Lanes>;
            using F = typename Ops::F;
            using I = typename Ops::I;
                I dots[Rows]{};
                if constexpr (Lanes == 8)
                {
#pragma GCC unroll 1
                    for (int g = 0; g < 8; ++g)
                    {
                        const I w = multi_scale::load<I>(p + g * Lanes);
#pragma GCC unroll 4
                        for (int r = 0; r < Rows; ++r)
                        {
                            const int word = multi_scale::load<int>(a[r][i].qs + g * 4);
                            if constexpr (CB == 5 || CB == 6 || CB == 7)
                                dots[r] = Ops::dotUnsignedQ5(dots[r], w, I{} + word);
                            else
                                dots[r] = Ops::dot(dots[r], w, I{} + word);
                        }
                    }
                }
                else
                {
                    // All eight groups unrolled, with even and odd groups in
                    // separate INT32 chains: each row's VPDPBUSD sequence is
                    // no longer one serial dependency, and every panel vector
                    // feeds all rows of the tile. The exact integer sum is
                    // order-free, so the dot equals the single-chain result.
                    I odd[Rows]{};
                    const auto group = [&]<int G>() {
                        const I w = multi_scale::load<I>(p + G * Lanes);
#pragma GCC unroll 4
                        for (int r = 0; r < Rows; ++r)
                        {
                            const I word = I{} + multi_scale::load<int>(a[r][i].qs + G * 4);
                            if constexpr (G % 2 == 0)
                                dots[r] = Ops::dot(dots[r], w, word);
                            else
                                odd[r] = Ops::dot(odd[r], w, word);
                        }
                    };
                    group.template operator()<0>();
                    group.template operator()<1>();
                    group.template operator()<2>();
                    group.template operator()<3>();
                    group.template operator()<4>();
                    group.template operator()<5>();
                    group.template operator()<6>();
                    group.template operator()<7>();
#pragma GCC unroll 4
                    for (int r = 0; r < Rows; ++r)
                        dots[r] += odd[r];
                }
#pragma GCC unroll 4
                for (int r = 0; r < Rows; ++r)
                {
                    const auto &act = a[r][i];
                    if constexpr (CB == 6)
                        dots[r] -= 16 * q5::sumQ8(act.qs);
                    if constexpr (Lanes == 16 && (CB == 4 || CB == 11))
                        dots[r] -= 128 * int(act.sum_qs);
                    const auto s = nativeQuantisedFP16ScaleToFP32(act.d);
                    if constexpr (Lanes == 8)
                    {
                        if constexpr (affine(static_cast<Format>(CB)))
                            acc[r] = std::bit_cast<F>(accumulateCorrectedSingleScaleBlockAVX2(
                                std::bit_cast<__m256>(acc[r]), std::bit_cast<__m256i>(dots[r]),
                                std::bit_cast<__m256>(multi_scale::load<F>(p + 8 * Lanes)), act.sum_qs,
                                std::bit_cast<__m256>(multi_scale::load<F>(p + 9 * Lanes)), s, P));
                        else
                            acc[r] = std::bit_cast<F>(accumulateSingleScaleBlockAVX2(
                                std::bit_cast<__m256>(acc[r]), std::bit_cast<__m256i>(dots[r]),
                                std::bit_cast<__m256>(multi_scale::load<F>(p + 8 * Lanes)), s, P));
                    }
#if defined(__AVX512VNNI__)
                    else
                    {
                        if constexpr (affine(static_cast<Format>(CB)))
                            acc[r] = std::bit_cast<F>(accumulateCorrectedSingleScaleBlockAVX512(
                                std::bit_cast<__m512>(acc[r]), std::bit_cast<__m512i>(dots[r]),
                                std::bit_cast<__m512>(multi_scale::load<F>(p + 8 * Lanes)), act.sum_qs,
                                std::bit_cast<__m512>(multi_scale::load<F>(p + 9 * Lanes)), s, P));
                        else
                            acc[r] = std::bit_cast<F>(accumulateSingleScaleBlockAVX512(
                                std::bit_cast<__m512>(acc[r]), std::bit_cast<__m512i>(dots[r]),
                                std::bit_cast<__m512>(multi_scale::load<F>(p + 8 * Lanes)), s, P));
                    }
#endif
                }
        }

        /**
         * @brief Consume a reusable panel with independent row dots and unchanged FP rounding.
         * @tparam Lanes Physical SIMD columns per row.
         * @tparam CB File-density source decoder that wrote the panel.
         * @tparam Rows Independently accumulated rows: one or two (AVX2), up to four (AVX-512).
         * @tparam P Original FP32 block and partition policy.
         * @param p Worker-local panel of decoded dot groups and widened factors.
         * @param n K blocks in this physical panel; zero for an empty serial partition.
         * @param a First activation block of each row's K interval.
         * @param out Live destinations, including their original N tail.
         * @param carry Unfinished noninitial partials, unused for a single-panel partition.
         * @param live Valid output lanes in each destination.
         * @param first_panel,last_panel Whether this panel starts/ends its arithmetic partition.
         * @param fold_earlier Whether the destination already owns a preceding partition.
         */
        template <int Lanes, int CB, int Rows, Policy P>
        [[gnu::noinline]]
        void consume(const float *p, int n, const std::array<const Q8_1Block *, Rows> &a,
                     const std::array<float *, Rows> &out, const std::array<float *, Rows> &carry, int live,
                     bool first_panel, bool last_panel, bool fold_earlier)
        {
            using Ops = multi_scale::SIMD<Lanes>;
            using F = typename Ops::F;
            using I = typename Ops::I;
            constexpr int stride = blockFloats(static_cast<Format>(CB), Lanes);
            std::array<F, Rows> acc{};
#pragma GCC unroll 4
            for (int r = 0; r < Rows; ++r)
                if (!first_panel)
                    acc[r] = fold_earlier ? loadOutput<Lanes>(carry[r], Lanes) : loadOutput<Lanes>(out[r], live);
            for (int i = 0; i < n; ++i, p += stride)
            {
                I dots[Rows]{};
                if constexpr (Lanes == 8)
                {
#pragma GCC unroll 1
                    for (int g = 0; g < 8; ++g)
                    {
                        const I w = multi_scale::load<I>(p + g * Lanes);
#pragma GCC unroll 4
                        for (int r = 0; r < Rows; ++r)
                        {
                            const int word = multi_scale::load<int>(a[r][i].qs + g * 4);
                            if constexpr (CB == 5 || CB == 6 || CB == 7)
                                dots[r] = Ops::dotUnsignedQ5(dots[r], w, I{} + word);
                            else
                                dots[r] = Ops::dot(dots[r], w, I{} + word);
                        }
                    }
                }
                else
                {
                    // All eight groups unrolled, with even and odd groups in
                    // separate INT32 chains: each row's VPDPBUSD sequence is
                    // no longer one serial dependency, and every panel vector
                    // feeds all rows of the tile. The exact integer sum is
                    // order-free, so the dot equals the single-chain result.
                    I odd[Rows]{};
                    const auto group = [&]<int G>() {
                        const I w = multi_scale::load<I>(p + G * Lanes);
#pragma GCC unroll 4
                        for (int r = 0; r < Rows; ++r)
                        {
                            const I word = I{} + multi_scale::load<int>(a[r][i].qs + G * 4);
                            if constexpr (G % 2 == 0)
                                dots[r] = Ops::dot(dots[r], w, word);
                            else
                                odd[r] = Ops::dot(odd[r], w, word);
                        }
                    };
                    group.template operator()<0>();
                    group.template operator()<1>();
                    group.template operator()<2>();
                    group.template operator()<3>();
                    group.template operator()<4>();
                    group.template operator()<5>();
                    group.template operator()<6>();
                    group.template operator()<7>();
#pragma GCC unroll 4
                    for (int r = 0; r < Rows; ++r)
                        dots[r] += odd[r];
                }
#pragma GCC unroll 4
                for (int r = 0; r < Rows; ++r)
                {
                    const auto &act = a[r][i];
                    if constexpr (CB == 6)
                        dots[r] -= 16 * q5::sumQ8(act.qs);
                    if constexpr (Lanes == 16 && (CB == 4 || CB == 11))
                        dots[r] -= 128 * int(act.sum_qs);
                    const auto s = nativeQuantisedFP16ScaleToFP32(act.d);
                    if constexpr (Lanes == 8)
                    {
                        if constexpr (affine(static_cast<Format>(CB)))
                            acc[r] = std::bit_cast<F>(accumulateCorrectedSingleScaleBlockAVX2(
                                std::bit_cast<__m256>(acc[r]), std::bit_cast<__m256i>(dots[r]),
                                std::bit_cast<__m256>(multi_scale::load<F>(p + 8 * Lanes)), act.sum_qs,
                                std::bit_cast<__m256>(multi_scale::load<F>(p + 9 * Lanes)), s, P));
                        else
                            acc[r] = std::bit_cast<F>(accumulateSingleScaleBlockAVX2(
                                std::bit_cast<__m256>(acc[r]), std::bit_cast<__m256i>(dots[r]),
                                std::bit_cast<__m256>(multi_scale::load<F>(p + 8 * Lanes)), s, P));
                    }
#if defined(__AVX512VNNI__)
                    else
                    {
                        if constexpr (affine(static_cast<Format>(CB)))
                            acc[r] = std::bit_cast<F>(accumulateCorrectedSingleScaleBlockAVX512(
                                std::bit_cast<__m512>(acc[r]), std::bit_cast<__m512i>(dots[r]),
                                std::bit_cast<__m512>(multi_scale::load<F>(p + 8 * Lanes)), act.sum_qs,
                                std::bit_cast<__m512>(multi_scale::load<F>(p + 9 * Lanes)), s, P));
                        else
                            acc[r] = std::bit_cast<F>(accumulateSingleScaleBlockAVX512(
                                std::bit_cast<__m512>(acc[r]), std::bit_cast<__m512i>(dots[r]),
                                std::bit_cast<__m512>(multi_scale::load<F>(p + 8 * Lanes)), s, P));
                    }
#endif
                }
            }
#pragma GCC unroll 4
            for (int r = 0; r < Rows; ++r)
            {
                if (!fold_earlier)
                    writeOutput<Lanes>(out[r], live, acc[r]);
                else if (!last_panel)
                    writeOutput<Lanes>(carry[r], Lanes, acc[r]);
                else
                    writeOutput<Lanes>(out[r], live, fold<Lanes, P>(loadOutput<Lanes>(out[r], live), acc[r]));
            }
        }
        /**
         * @brief Scalar geometry of one register-folded panel, passed in two registers.
         * @details Sixteen bytes of integers travel in two general registers
         * (SysV x86-64), so the folded leaves take five register arguments and
         * no outgoing stack argument.
         */
        struct PanelFold
        {
            int width;      ///< Blocks per arithmetic partition (the last partition of K may be shorter).
            int n;          ///< K blocks in this panel; whole partitions' blocks.
            int first;      ///< Nonzero when this panel starts with partition 0 of the serial tree.
            int empty_tail; ///< Empty trailing partitions folded after the last nonempty one (last panel only).
        };
        static_assert(sizeof(PanelFold) == 16);

        /**
         * @brief Consume whole GPU-aligned arithmetic partitions of one panel and fold them in registers.
         * @tparam Lanes Physical SIMD columns per row.
         * @tparam CB File-density source decoder that wrote the panel.
         * @tparam Rows Independently accumulated rows: one or two (AVX2), up to four (AVX-512).
         * @tparam P Original FP32 block and partition policy.
         * @param p Worker-local panel holding the blocks of consecutive whole partitions.
         * @param fold Partition width, panel blocks, first-panel flag and empty trailing partitions.
         * @param a First activation block of each row's panel interval.
         * @param out Destinations of complete strips (every lane live); they hold the
         *        folded total of every earlier panel.
         * @details The serial M=1 tree forms every partition from +0, seeds the
         * total with partition 0, then folds each later partition and each
         * empty trailing partition (+0) as total + partial. consume() ran that
         * tree one partition per call with a destination load/fold/store per
         * partition; here each partition is formed in registers and folded at
         * once, with the identical operations, operands and order. The total
         * is published once per panel, and a later panel resumes from that
         * FP32 value, which round-trips exactly. Only complete strips take
         * this leaf: an N-tail strip keeps consume(), so no tail count or lane
         * mask is live here (it was reloaded from the stack in the epilogue).
         */
        template <int Lanes, int CB, int Rows, Policy P>
        [[gnu::noinline]]
        void consumeOrdered(const float *p, PanelFold fold_geometry, const std::array<const Q8_1Block *, Rows> &a,
                            const std::array<float *, Rows> &out)
        {
            const int width = fold_geometry.width, n = fold_geometry.n, empty_tail = fold_geometry.empty_tail;
            bool first = fold_geometry.first != 0;
            static_assert(P == Policy::GPUAlignedExpert, "Only operand-ordered folds may move into registers");
            using F = typename multi_scale::SIMD<Lanes>::F;
            constexpr int stride = blockFloats(static_cast<Format>(CB), Lanes);
            std::array<F, Rows> total{};
#pragma GCC unroll 4
            for (int r = 0; r < Rows; ++r)
                if (!first)
                    total[r] = multi_scale::load<F>(out[r]);
            for (int i = 0; i < n;)
            {
                const int end = std::min(n, i + width);
                std::array<F, Rows> acc{};
                for (; i < end; ++i, p += stride)
                    consumeBlock<Lanes, CB, Rows, P>(acc, p, a, i);
#pragma GCC unroll 4
                for (int r = 0; r < Rows; ++r)
                    total[r] = first ? acc[r] : fold<Lanes, P>(total[r], acc[r]);
                first = false;
            }
            for (int e = 0; e < empty_tail; ++e)
#pragma GCC unroll 4
                for (int r = 0; r < Rows; ++r)
                    total[r] = fold<Lanes, P>(total[r], F{});
#pragma GCC unroll 4
            for (int r = 0; r < Rows; ++r)
                std::memcpy(out[r], &total[r], sizeof(F));
        }
#if defined(__AVX512VNNI__)
        /**
         * @brief Accumulate one 64-column panel block into both rows of a wide register tile.
         * @tparam CB File-density source decoder that wrote the panel.
         * @tparam Rows One or two rows sharing every panel vector.
         * @tparam P Original FP32 block policy.
         * @param acc Each row's four running FP32 vectors, updated in place.
         * @param p This block's interleaved dot-group planes and widened factors.
         * @param a First activation block of each row's panel interval.
         * @param i Block index inside the panel interval.
         * @details consumeWide()'s per-block program (same source; consumeWide()
         * keeps its own copy for the reason given at consumeBlock()) for the
         * GPU-aligned consumeWideOrdered(): each activation broadcast feeds four
         * VPDPBUSD, each row's activation scale and correction serve all four
         * vectors, every column keeps consume()'s exact INT32 dot and FP32
         * block program.
         */
        template <int CB, int Rows, Policy P>
        [[gnu::always_inline]] inline void consumeWideBlock(
            typename multi_scale::SIMD<16>::F (&acc)[Rows][wide_strip / 16], const float *p,
            const std::array<const Q8_1Block *, Rows> &a, int i)
        {
            constexpr int Lanes = 16, Subs = wide_strip / Lanes;
            using Ops = multi_scale::SIMD<Lanes>;
            using F = typename Ops::F;
            using I = typename Ops::I;
            I dots[Rows][Subs]{};
            const auto group = [&]<int G>() {
                I w[Subs];
#pragma GCC unroll 4
                for (int s = 0; s < Subs; ++s)
                    w[s] = multi_scale::load<I>(p + (G * Subs + s) * Lanes);
#pragma GCC unroll 2
                for (int r = 0; r < Rows; ++r)
                {
                    const I word = I{} + multi_scale::load<int>(a[r][i].qs + G * 4);
#pragma GCC unroll 4
                    for (int s = 0; s < Subs; ++s)
                        dots[r][s] = Ops::dot(dots[r][s], w[s], word);
                }
            };
            group.template operator()<0>();
            group.template operator()<1>();
            group.template operator()<2>();
            group.template operator()<3>();
            group.template operator()<4>();
            group.template operator()<5>();
            group.template operator()<6>();
            group.template operator()<7>();
#pragma GCC unroll 2
            for (int r = 0; r < Rows; ++r)
            {
                const auto &act = a[r][i];
                int correction = 0;
                if constexpr (CB == 6)
                    correction = 16 * q5::sumQ8(act.qs);
                if constexpr (CB == 4 || CB == 11)
                    correction = 128 * int(act.sum_qs);
                const auto s_a = nativeQuantisedFP16ScaleToFP32(act.d);
#pragma GCC unroll 4
                for (int s = 0; s < Subs; ++s)
                {
                    const auto dot = dots[r][s] - correction;
                    const auto scale = std::bit_cast<__m512>(multi_scale::load<F>(p + (8 * Subs + s) * Lanes));
                    if constexpr (affine(static_cast<Format>(CB)))
                        acc[r][s] = std::bit_cast<F>(accumulateCorrectedSingleScaleBlockAVX512(
                            std::bit_cast<__m512>(acc[r][s]), std::bit_cast<__m512i>(dot), scale, act.sum_qs,
                            std::bit_cast<__m512>(multi_scale::load<F>(p + (9 * Subs + s) * Lanes)), s_a, P));
                    else
                        acc[r][s] = std::bit_cast<F>(accumulateSingleScaleBlockAVX512(
                            std::bit_cast<__m512>(acc[r][s]), std::bit_cast<__m512i>(dot), scale, s_a, P));
                }
            }
        }

        /**
         * @brief Consume a 64-column AVX-512 panel with two rows of four independent vectors.
         * @tparam CB File-density source decoder that wrote the panel.
         * @tparam Rows One or two rows sharing every panel vector.
         * @tparam P Original FP32 block and partition policy.
         * @param p,n,a,out,carry,live,first_panel,last_panel,fold_earlier As consume().
         *
         * The two-row, four-vector tile of the expanded INT8 kernels: each
         * activation broadcast feeds four VPDPBUSD, and each row's activation
         * scale and zero-point correction are computed once per block for all
         * 64 columns instead of once per 16. Four vectors per row are already
         * independent dependency chains. Every column keeps consume()'s exact
         * INT32 dot, correction, FP32 block program and partition fold.
         */
        template <int CB, int Rows, Policy P>
        [[gnu::noinline]]
        void consumeWide(const float *p, int n, const std::array<const Q8_1Block *, Rows> &a,
                         const std::array<float *, Rows> &out, const std::array<float *, Rows> &carry, int live,
                         bool first_panel, bool last_panel, bool fold_earlier)
        {
            constexpr int Lanes = 16, Subs = wide_strip / Lanes;
            using Ops = multi_scale::SIMD<Lanes>;
            using F = typename Ops::F;
            using I = typename Ops::I;
            constexpr int stride = blockFloats(static_cast<Format>(CB), wide_strip);
            // Live lanes of each 16-column vector of the strip's (possibly partial) tail.
            int lives[Subs];
#pragma GCC unroll 4
            for (int s = 0; s < Subs; ++s)
                lives[s] = std::clamp(live - s * Lanes, 0, Lanes);
            F acc[Rows][Subs]{};
#pragma GCC unroll 2
            for (int r = 0; r < Rows; ++r)
#pragma GCC unroll 4
                for (int s = 0; s < Subs; ++s)
                    if (!first_panel)
                        acc[r][s] = fold_earlier ? loadOutput<Lanes>(carry[r] + s * Lanes, Lanes)
                                                 : loadOutput<Lanes>(out[r] + s * Lanes, lives[s]);
            for (int i = 0; i < n; ++i, p += stride)
            {
                I dots[Rows][Subs]{};
                const auto group = [&]<int G>() {
                    I w[Subs];
#pragma GCC unroll 4
                    for (int s = 0; s < Subs; ++s)
                        w[s] = multi_scale::load<I>(p + (G * Subs + s) * Lanes);
#pragma GCC unroll 2
                    for (int r = 0; r < Rows; ++r)
                    {
                        const I word = I{} + multi_scale::load<int>(a[r][i].qs + G * 4);
#pragma GCC unroll 4
                        for (int s = 0; s < Subs; ++s)
                            dots[r][s] = Ops::dot(dots[r][s], w[s], word);
                    }
                };
                group.template operator()<0>();
                group.template operator()<1>();
                group.template operator()<2>();
                group.template operator()<3>();
                group.template operator()<4>();
                group.template operator()<5>();
                group.template operator()<6>();
                group.template operator()<7>();
#pragma GCC unroll 2
                for (int r = 0; r < Rows; ++r)
                {
                    const auto &act = a[r][i];
                    int correction = 0;
                    if constexpr (CB == 6)
                        correction = 16 * q5::sumQ8(act.qs);
                    if constexpr (CB == 4 || CB == 11)
                        correction = 128 * int(act.sum_qs);
                    const auto s_a = nativeQuantisedFP16ScaleToFP32(act.d);
#pragma GCC unroll 4
                    for (int s = 0; s < Subs; ++s)
                    {
                        const auto dot = dots[r][s] - correction;
                        const auto scale = std::bit_cast<__m512>(multi_scale::load<F>(p + (8 * Subs + s) * Lanes));
                        if constexpr (affine(static_cast<Format>(CB)))
                            acc[r][s] = std::bit_cast<F>(accumulateCorrectedSingleScaleBlockAVX512(
                                std::bit_cast<__m512>(acc[r][s]), std::bit_cast<__m512i>(dot), scale, act.sum_qs,
                                std::bit_cast<__m512>(multi_scale::load<F>(p + (9 * Subs + s) * Lanes)), s_a, P));
                        else
                            acc[r][s] = std::bit_cast<F>(accumulateSingleScaleBlockAVX512(
                                std::bit_cast<__m512>(acc[r][s]), std::bit_cast<__m512i>(dot), scale, s_a, P));
                    }
                }
            }
#pragma GCC unroll 2
            for (int r = 0; r < Rows; ++r)
#pragma GCC unroll 4
                for (int s = 0; s < Subs; ++s)
                {
                    if (!fold_earlier)
                        writeOutput<Lanes>(out[r] + s * Lanes, lives[s], acc[r][s]);
                    else if (!last_panel)
                        writeOutput<Lanes>(carry[r] + s * Lanes, Lanes, acc[r][s]);
                    else
                        writeOutput<Lanes>(out[r] + s * Lanes, lives[s],
                                           fold<Lanes, P>(loadOutput<Lanes>(out[r] + s * Lanes, lives[s]), acc[r][s]));
                }
        }
        /**
         * @brief Wide-strip consumeOrdered(): whole partitions of a 64-column panel folded in registers.
         * @tparam CB File-density source decoder that wrote the panel.
         * @tparam Rows One or two rows sharing every panel vector.
         * @tparam P Original FP32 block and partition policy.
         * @tparam Single Whether every partition is one block (width == 1, GPU-aligned K <= 512).
         * @param p,fold_geometry,a,out As consumeOrdered(): a complete 64-column strip.
         * @details consumeWideBlock()'s exact block program; partitions start
         * from +0, partition 0 seeds the total, later and empty trailing
         * partitions fold as total + partial in increasing order.
         */
        template <int CB, int Rows, Policy P, bool Single>
        [[gnu::noinline]]
        void consumeWideOrdered(const float *p, PanelFold fold_geometry, const std::array<const Q8_1Block *, Rows> &a,
                                const std::array<float *, Rows> &out)
        {
            const int width = fold_geometry.width, n = fold_geometry.n, empty_tail = fold_geometry.empty_tail;
            bool first = fold_geometry.first != 0;
            static_assert(P == Policy::GPUAlignedExpert, "Only operand-ordered folds may move into registers");
            constexpr int Lanes = 16, Subs = wide_strip / Lanes;
            using F = typename multi_scale::SIMD<Lanes>::F;
            constexpr int stride = blockFloats(static_cast<Format>(CB), wide_strip);
            F total[Rows][Subs]{};
#pragma GCC unroll 2
            for (int r = 0; r < Rows; ++r)
#pragma GCC unroll 4
                for (int s = 0; s < Subs; ++s)
                    if (!first)
                        total[r][s] = multi_scale::load<F>(out[r] + s * Lanes);
            for (int i = 0; i < n;)
            {
                // A one-block partition (Single) is folded as soon as it is
                // formed; its partial is never live beside the totals across a
                // dot phase.
                const int end = Single ? i + 1 : std::min(n, i + width);
                F acc[Rows][Subs]{};
                for (; i < end; ++i, p += stride)
                    consumeWideBlock<CB, Rows, P>(acc, p, a, i);
#pragma GCC unroll 2
                for (int r = 0; r < Rows; ++r)
#pragma GCC unroll 4
                    for (int s = 0; s < Subs; ++s)
                        total[r][s] = first ? acc[r][s] : fold<Lanes, P>(total[r][s], acc[r][s]);
                first = false;
            }
            for (int e = 0; e < empty_tail; ++e)
#pragma GCC unroll 2
                for (int r = 0; r < Rows; ++r)
#pragma GCC unroll 4
                    for (int s = 0; s < Subs; ++s)
                        total[r][s] = fold<Lanes, P>(total[r][s], F{});
#pragma GCC unroll 2
            for (int r = 0; r < Rows; ++r)
#pragma GCC unroll 4
                for (int s = 0; s < Subs; ++s)
                    std::memcpy(out[r] + s * Lanes, &total[r][s], sizeof(F));
        }
#endif

        /**
         * @brief Visit every live row tile of a strip.
         * @tparam TileRows Register-tile height (rowTile() of the strip, or one).
         * @param grid Read-only live row ownership, including request boundaries.
         * @param update Generic callable invoked as update.template operator()<Rows>(first_row).
         * @details A sixteen-column AVX-512 strip reuses each panel vector
         * across four rows; AVX2 and the four-vector wide strip keep two rows
         * within their register budgets.
         */
        template <int TileRows, class Update>
        [[gnu::always_inline]] inline void eachRowTile(const CPUProjectionRowGrid &grid, Update &&update)
        {
            constexpr int tile_rows = TileRows;
            for (int tile = 0; tile < grid.tiles(tile_rows); ++tile)
            {
                const auto span = grid.tileSpan(tile, tile_rows);
                switch (span.size())
                {
                case 4:
                    if constexpr (tile_rows >= 4)
                        update.template operator()<4>(span.first);
                    break;
                case 3:
                    if constexpr (tile_rows >= 3)
                        update.template operator()<3>(span.first);
                    break;
                case 2:
                    if constexpr (tile_rows >= 2)
                        update.template operator()<2>(span.first);
                    break;
                case 1:
                    update.template operator()<1>(span.first);
                    break;
                default:
                    break;
                }
            }
        }

        /** @return Whether a strip takes the register-folded GPU-aligned program (stripeOrdered()). */
        template <int Strip, Policy P>
        inline bool foldsInRegisters(const CPUNativeQuantisedWeightView &w, int n, int parts) noexcept
        {
            // Only the GPU-aligned tree has an operand-ordered fold
            // (addOrdered*), so only it may move its folds into registers and
            // keep every byte, NaN payloads included; a backend-native fold is
            // a plain FP32 add whose NaN operand the compiler chooses. Only
            // complete strips: an N-tail strip keeps stripe()'s per-partition
            // program, whose masked tail publication is unchanged.
            const int width = (w.blocks_per_row - 1) / parts + 1;
            return P == Policy::GPUAlignedExpert && parts > 1 && width <= depth && w.N - n >= Strip;
        }

        /**
         * @brief stripe() for GPU-aligned trees whose partitions fit one panel, on a complete strip.
         * @tparam Lanes,Strip,CB,P As stripe(); P is GPU-aligned and foldsInRegisters() holds.
         * @param w,a,c,ldc,n,parts,grid,panel As stripe(); no carry is needed.
         * @details Packs as many whole partitions as fit one panel and folds
         * them in registers (consumeOrdered / consumeWideOrdered) with the
         * serial tree's exact ordered operations. A separate function, so that
         * stripe()'s per-partition program and its frame stay those of
         * 67dac67f7. Its loop control is scalar: superword vectorization of
         * the per-row-tile pointer arrays only moved pointers through vector
         * registers and stack slots (no arithmetic is affected).
         */
        template <int Lanes, int Strip, int CB, Policy P>
        [[gnu::noinline, gnu::optimize("no-tree-slp-vectorize")]]
        void stripeOrdered(const CPUNativeQuantisedWeightView &w, const Q8_1Block *a, float *c, int ldc, int n,
                           int parts, const CPUProjectionRowGrid &grid, float *panel)
        {
            constexpr int Subs = Strip / Lanes;
            static_assert(Subs == 1 || (Lanes == 16 && Strip == wide_strip));
            static_assert(P == Policy::GPUAlignedExpert);
            const int blocks = w.blocks_per_row;
            const int width = (blocks - 1) / parts + 1;
            // A wide strip is one whole chunk: decode its four sixteen-column
            // vectors into their interleaved planes.
            const auto packPanel = [&](int begin, int next)
            {
                pack<Lanes, Subs, 0, CB>(w, n / 64, n % 64, begin, next, panel);
                if constexpr (Subs == 4)
                {
                    pack<Lanes, Subs, 1, CB>(w, n / 64, Lanes, begin, next, panel);
                    pack<Lanes, Subs, 2, CB>(w, n / 64, 2 * Lanes, begin, next, panel);
                    pack<Lanes, Subs, 3, CB>(w, n / 64, 3 * Lanes, begin, next, panel);
                }
            };
            // Partitions [0, nonempty) hold blocks; the remaining ones are
            // empty and fold +0 after the last nonempty partition.
            const int per_panel = depth / width, nonempty = (blocks - 1) / width + 1;
            for (int first = 0; first < nonempty; first += per_panel)
            {
                const int last = std::min(nonempty, first + per_panel);
                const int begin = first * width, next = std::min(last * width, blocks);
                const int empty_tail = last == nonempty ? parts - nonempty : 0;
                packPanel(begin, next);
                const auto consumeTiles = [&]<int TileRows, bool Single>()
                {
                    eachRowTile<TileRows>(grid, [&]<int Rows>(int row)
                    {
                        std::array<const Q8_1Block *, Rows> in{};
                        std::array<float *, Rows> out{};
                        for (int r = 0; r < Rows; ++r)
                        {
                            out[r] = c + std::size_t(row + r) * ldc + n;
                            in[r] = a + std::size_t(row + r) * blocks + begin;
                        }
#if defined(__AVX512VNNI__)
                        if constexpr (Subs == 4)
                            consumeWideOrdered<CB, Rows, P, Single>(panel, {width, next - begin, first == 0, empty_tail},
                                                                    in, out);
                        else
#endif
                            consumeOrdered<Lanes, CB, Rows, P>(panel, {width, next - begin, first == 0, empty_tail}, in,
                                                               out);
                    });
                };
                // The tile height is panelRowTile(), which the launch evidence reports.
                constexpr int tile_rows = rowTile(Strip);
                if constexpr (Subs == 4)
                {
                    constexpr bool split = wideOrderedSplitsRows(static_cast<Format>(CB));
                    if (width == 1)
                        consumeTiles.template operator()<tile_rows, true>();
                    else
                        consumeTiles.template operator()<split ? 1 : tile_rows, false>();
                }
                else
                    consumeTiles.template operator()<tile_rows, false>();
            }
        }

        /**
         * @brief Own one original N strip across all live rows and the complete K tree.
         * @tparam Lanes Physical SIMD columns per row.
         * @tparam CB Admitted source decoder.
         * @tparam P Original FP32 block and partition policy.
         * @param w Admitted weight view.
         * @param a,c Original row-major input and output roots.
         * @param ldc Original output row stride.
         * @param n First original N coordinate in this strip.
         * @param parts Exact serial K partition count.
         * @param grid Read-only live row ownership, including request boundaries.
         * @param panel,carry Exclusive worker slot (carry present only when needed).
         *
         * The per-partition program of 67dac67f7, unchanged. Callers route
         * complete strips of GPU-aligned trees whose partitions fit one panel
         * (at most @ref depth blocks each: every expert tree up to K = 4096) to
         * stripeOrdered() instead (foldsInRegisters()).
         */
        template <int Lanes, int Strip, int CB, Policy P>
        void stripe(const CPUNativeQuantisedWeightView &w, const Q8_1Block *a, float *c, int ldc, int n, int parts,
                    const CPUProjectionRowGrid &grid, float *panel, float *carry)
        {
            constexpr int Subs = Strip / Lanes;
            static_assert(Subs == 1 || (Lanes == 16 && Strip == wide_strip));
            const int live = std::min(Strip, w.N - n);
            const int blocks = w.blocks_per_row;
            const int width = (blocks - 1) / parts + 1;
            for (int kpart = 0; kpart < parts; ++kpart)
            {
                const int first = static_cast<int>(std::min(std::int64_t(kpart) * width, std::int64_t(blocks))),
                          end = first + std::min(width, blocks - first);
                for (int begin = first;;)
                {
                    const int next = std::min(end, begin + depth);
                    if (next > begin)
                    {
                        // A wide strip is one whole chunk: decode its four
                        // sixteen-column vectors into their interleaved planes.
                        pack<Lanes, Subs, 0, CB>(w, n / 64, n % 64, begin, next, panel);
                        if constexpr (Subs == 4)
                        {
                            pack<Lanes, Subs, 1, CB>(w, n / 64, Lanes, begin, next, panel);
                            pack<Lanes, Subs, 2, CB>(w, n / 64, 2 * Lanes, begin, next, panel);
                            pack<Lanes, Subs, 3, CB>(w, n / 64, 3 * Lanes, begin, next, panel);
                        }
                    }
                    auto update = [&]<int Rows>(int row)
                    {
                        std::array<const Q8_1Block *, Rows> in{};
                        std::array<float *, Rows> out{}, partial{};
                        for (int r = 0; r < Rows; ++r)
                        {
                            out[r] = c + std::size_t(row + r) * ldc + n;
                            in[r] = a + std::size_t(row + r) * blocks + begin;
                            if (carry)
                                partial[r] = carry + std::size_t(row + r) * Strip;
                        }
#if defined(__AVX512VNNI__)
                        if constexpr (Subs == 4)
                            consumeWide<CB, Rows, P>(panel, next - begin, in, out, partial, live, begin == first,
                                                     next == end, kpart != 0);
                        else
#endif
                            consume<Lanes, CB, Rows, P>(panel, next - begin, in, out, partial, live, begin == first,
                                                        next == end, kpart != 0);
                    };
                    // A sixteen-column AVX-512 strip reuses each panel vector
                    // across four rows; AVX2 and the four-vector wide strip
                    // keep two rows within their register budgets.
                    constexpr int width = rowTile(Strip);
                    for (int tile = 0; tile < grid.tiles(width); ++tile)
                    {
                        const auto span = grid.tileSpan(tile, width);
                        switch (span.size())
                        {
                        case 4:
                            if constexpr (width >= 4)
                                update.template operator()<4>(span.first);
                            break;
                        case 3:
                            if constexpr (width >= 3)
                                update.template operator()<3>(span.first);
                            break;
                        case 2:
                            update.template operator()<2>(span.first);
                            break;
                        case 1:
                            update.template operator()<1>(span.first);
                            break;
                        default:
                            break;
                        }
                    }
                    if (next == end)
                        break;
                    begin = next;
                }
            }
        }

        /** Workshare over stripes, not row replay: each packed panel is consumed across the row grid. */
        template <int Lanes, int Strip, int CB, Policy P>
        void run(const CPUNativeQuantisedWeightView &w, const Q8_1Block *a, float *c, int ldc, int rows, int parts,
                 const CPUProjectionRowGrid &grid, float *scratch)
        {
            auto task = [&]
            {
                constexpr auto format = static_cast<Format>(CB);
                const auto pitch = workerFloats(format, Strip, rows, w.blocks_per_row, parts);
                auto *panel = scratch + static_cast<std::size_t>(omp_get_thread_num()) * pitch;
                auto *carry = needsCarry(w.blocks_per_row, parts)
                                  ? panel + std::min(w.blocks_per_row, depth) * blockFloats(format, Strip)
                                  : nullptr;
                const int strips = columnStrips(w.N, Strip);
#pragma omp for schedule(static)
                for (int strip = 0; strip < strips; ++strip)
                {
                    if (foldsInRegisters<Strip, P>(w, strip * Strip, parts))
                    {
                        if constexpr (P == Policy::GPUAlignedExpert)
                            stripeOrdered<Lanes, Strip, CB, P>(w, a, c, ldc, strip * Strip, parts, grid, panel);
                    }
                    else
                        stripe<Lanes, Strip, CB, P>(w, a, c, ldc, strip * Strip, parts, grid, panel, carry);
                }
            };
            OMP_WORKSHARE_REGION(task);
        }

        /**
         * @brief Resolve a validated panel format and FP policy into one compiled program.
         * @param format Admitted panel format.
         * @param policy Immutable projection arithmetic identity.
         * @param body Generic callable invoked as body.template operator()<CB, P>().
         * @throws std::invalid_argument For Format::None.
         */
        template <class Body> void visitProgram(Format format, Policy policy, Body &&body)
        {
            const auto call = [&]<Format F>()
            {
                constexpr auto cb = static_cast<int>(F);
                if (policy == Policy::GPUAlignedExpert)
                    return body.template operator()<cb, Policy::GPUAlignedExpert>();
                return body.template operator()<cb, Policy::BackendNative>();
            };
            switch (format)
            {
            case Format::Q4K:
                return call.template operator()<Format::Q4K>();
            case Format::Q5K:
                return call.template operator()<Format::Q5K>();
            case Format::IQ3S:
                return call.template operator()<Format::IQ3S>();
            case Format::IQ4XS:
                return call.template operator()<Format::IQ4XS>();
            case Format::Q50:
                return call.template operator()<Format::Q50>();
            case Format::None:
                break;
            }
            throw std::invalid_argument("Native x86 prefill requires a coded panel format");
        }

        /** @brief Dispatch a validated panel format and FP policy before entering the team. */
        template <int Lanes, int Strip>
        void dispatch(Format format, const CPUNativeQuantisedWeightView &w, const Q8_1Block *a, float *out, int rows,
                      int stride, int partitions, const CPUProjectionRowGrid &grid, float *scratch)
        {
            visitProgram(format, w.numerical_policy, [&]<int CB, Policy P>()
                         { run<Lanes, Strip, CB, P>(w, a, out, stride, rows, partitions, grid, scratch); });
        }

        /**
         * @brief Stream one 64-column chunk of a pooled member through the calling worker's slot.
         * @tparam Lanes Physical SIMD columns per row.
         * @tparam Strip Columns per strip: the whole chunk on AVX-512, eight on AVX2.
         * @tparam CB Admitted source decoder.
         * @tparam P Original FP32 block and partition policy.
         * @param w Admitted weight view.
         * @param a,c Member's row-major input and output roots.
         * @param ldc Member's output row stride.
         * @param rows Member's rows; every row is live.
         * @param parts Exact serial K partition count.
         * @param chunk Physical 64-column chunk owned by this task.
         * @param panel Calling worker's slot: panel, then (only when needed) one carry per row.
         */
        template <int Lanes, int Strip, int CB, Policy P>
        void pooledChunk(const CPUNativeQuantisedWeightView &w, const Q8_1Block *a, float *c, int ldc, int rows,
                         int parts, int chunk, float *panel)
        {
            constexpr auto format = static_cast<Format>(CB);
            const CPUProjectionRowGrid grid(rows, nullptr);
            auto *carry = needsCarry(w.blocks_per_row, parts)
                              ? panel + std::min(w.blocks_per_row, depth) * blockFloats(format, Strip)
                              : nullptr;
            const int columns = std::min(64, w.N - 64 * chunk);
            for (int column = 0; column < columns; column += Strip)
            {
                if (foldsInRegisters<Strip, P>(w, 64 * chunk + column, parts))
                {
                    if constexpr (P == Policy::GPUAlignedExpert)
                        stripeOrdered<Lanes, Strip, CB, P>(w, a, c, ldc, 64 * chunk + column, parts, grid, panel);
                }
                else
                    stripe<Lanes, Strip, CB, P>(w, a, c, ldc, 64 * chunk + column, parts, grid, panel, carry);
            }
        }
    } // namespace

    void project(const CPUNativeQuantisedWeightView &weights, Format format, const Q8_1Block *activation, float *output,
                 int rows, int output_stride, int partitions, int workers, int lanes, const CPUProjectionRowGrid &grid,
                 const ProjectionRowWork *work, std::span<float> scratch)
    {
        if (!supportsGeometry(format, lanes, rows, weights.N, workers) || partitions <= 0 || output_stride < weights.N ||
            format != formatFor(weights.encoding, weights.codebook_id))
            throw std::invalid_argument("Native x86 prefill received invalid panel geometry");
        // This invocation's own width; the admitted borrow covers every width up to the participant's.
        const auto size = invocationFloats(format, lanes, rows, weights.N, weights.blocks_per_row, partitions, workers);
        if (scratch.size() < size)
            throw std::invalid_argument("Native x86 prefill requires its admitted per-worker panel storage");
        const storage_proof::Bytes writes{scratch.data(), size * sizeof(float)};
        storage_proof::requireDisjoint(writes,
                                       {activation, std::size_t(rows) * weights.blocks_per_row * sizeof(Q8_1Block)});
        storage_proof::requireDisjoint(writes,
                                       {output, (std::size_t(rows - 1) * output_stride + weights.N) * sizeof(float)});
        storage_proof::requireDisjoint(writes, {weights.interleavedBase(), weights.bytes().size()});
        storage_proof::requireOriginalWorkReadOnly(writes, work);
        if (lanes == 8)
            return dispatch<8, 8>(format, weights, activation, output, rows, output_stride, partitions, grid,
                                  scratch.data());
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        // The same rule as the borrowed quote: a whole chunk per worker when possible.
        if (stripColumns(lanes, weights.N, workers) == wide_strip)
            return dispatch<16, wide_strip>(format, weights, activation, output, rows, output_stride, partitions, grid,
                                            scratch.data());
        return dispatch<16, 16>(format, weights, activation, output, rows, output_stride, partitions, grid,
                                scratch.data());
#else
        throw std::invalid_argument("Native x86 prefill cannot select AVX-512 in its AVX2 envelope");
#endif
    }

    void projectPooledChunk(const CPUNativeQuantisedWeightView &weights, Format format, const Q8_1Block *activation,
                            float *output, int rows, int output_stride, int partitions, int lanes, int chunk,
                            float *slot)
    {
        if (lanes == 8)
            return visitProgram(format, weights.numerical_policy, [&]<int CB, Policy P>()
                { pooledChunk<8, 8, CB, P>(weights, activation, output, output_stride, rows, partitions, chunk, slot); });
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        return visitProgram(format, weights.numerical_policy, [&]<int CB, Policy P>()
            { pooledChunk<16, wide_strip, CB, P>(weights, activation, output, output_stride, rows, partitions, chunk,
                                                  slot); });
#else
        throw std::invalid_argument("Native x86 pooled prefill cannot select AVX-512 in its AVX2 envelope");
#endif
    }
} // namespace llaminar2::cpu::native_quantised::x86_panel
#endif
