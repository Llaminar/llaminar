/**
 * @file CPUNativeX86PackedPrefill.cpp
 * @brief Decode native weights into a short worker panel and reuse across live rows.
 *
 * Every worker visits increasing K inside the exact M=1 partition tree. Pack
 * writes directly to the worker-owned slab. The next stage only reads that
 * bounded panel and writes final output (or an unfinished ordered partial).
 * No expanded weight publication or per-invocation allocation exists.
 */
#if defined(__AVX2__)
#include "CPUNativeX86PackedPrefill.h"
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
         * @brief Decode directly into the current worker's reserved panel.
         * @tparam Lanes Eight AVX2 or sixteen AVX-512 output columns.
         * @tparam CB Source factor and payload decoder.
         * @param w Admitted file-density publication, retained for the invocation.
         * @param chunk,col Physical original coordinates of the column strip.
         * @param begin,end Increasing logical K-block interval inside one arithmetic partition.
         * @param out Worker slot, large enough for all intervals up to @ref depth.
         */
        template <int Lanes, int CB>
        [[gnu::noinline]]
        void pack(const CPUNativeQuantisedWeightView &w, int chunk, int col, int begin, int end, float *out)
        {
            using Ops = multi_scale::SIMD<Lanes>;
            using I = typename Ops::I;
            using L = coded_metadata::Lanes<Ops>;
            constexpr int stride = blockFloats(static_cast<Format>(CB), Lanes);
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
                            storeVector(out + Lanes * g, multi_scale::decodeIQ3SGroup<Lanes, Ops>(u, col, g));
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
                                    storeVector(out + Lanes * g, l);
                                    storeVector(out + Lanes * (g + 4), h);
                                }
#if defined(__AVX512VNNI__)
                                else
                                {
                                    auto [l, h] = q5::decode16(u, col, g);
                                    storeVector(out + Lanes * g, l);
                                    storeVector(out + Lanes * (g + 4), h);
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
                                    storeVector(out + Lanes * group, low);
                                };
                                store(raw, g);
                                store(raw >> 4, g + 4);
                            }
                        }
                    }
                    if constexpr (CB == 6)
                        storeVector(out + 8 * Lanes, Ops::half(reinterpret_cast<const uint16_t *>(u + data) + col));
                    else if constexpr (CB == 11)
                        storeVector(out + 8 * Lanes, d * L::toFloat(2 * L::nibbles(u + data, col) + 1));
                    else if constexpr (CB == 4)
                    {
                        constexpr auto key = kNativeQuantisedIQ4XSSourceScaleCodebook;
                        storeVector(out + 8 * Lanes, coded_metadata::sourceScaleFactors<key, Ops>(
                                                         L::template code<key>(u + data, col), d)
                                                         .primary);
                    }
                    else
                    {
                        const auto factors =
                            coded_metadata::reconstruct<5, Ops>(L::template code<CB>(u + data, col), {d, min});
                        storeVector(out + 8 * Lanes, factors.primary);
                        storeVector(out + 9 * Lanes, factors.secondary);
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
         */
        template <int Lanes, int CB, Policy P>
        void stripe(const CPUNativeQuantisedWeightView &w, const Q8_1Block *a, float *c, int ldc, int n, int parts,
                    const CPUProjectionRowGrid &grid, float *panel, float *carry)
        {
            const int live = std::min(Lanes, w.N - n);
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
                        pack<Lanes, CB>(w, n / 64, n % 64, begin, next, panel);
                    auto update = [&]<int Rows>(int row)
                    {
                        std::array<const Q8_1Block *, Rows> in{};
                        std::array<float *, Rows> out{}, partial{};
                        for (int r = 0; r < Rows; ++r)
                        {
                            out[r] = c + std::size_t(row + r) * ldc + n;
                            in[r] = a + std::size_t(row + r) * blocks + begin;
                            if (carry)
                                partial[r] = carry + std::size_t(row + r) * Lanes;
                        }
                        consume<Lanes, CB, Rows, P>(panel, next - begin, in, out, partial, live, begin == first,
                                                    next == end, kpart != 0);
                    };
                    // AVX-512 reuses each panel vector across four rows; the
                    // sixteen AVX2 registers keep the two-row tile.
                    constexpr int width = Lanes == 16 ? 4 : 2;
                    for (int tile = 0; tile < grid.tiles(width); ++tile)
                    {
                        const auto span = grid.tileSpan(tile, width);
                        switch (span.size())
                        {
                        case 4:
                            if constexpr (width == 4)
                                update.template operator()<4>(span.first);
                            break;
                        case 3:
                            if constexpr (width == 4)
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
        template <int Lanes, int CB, Policy P>
        void run(const CPUNativeQuantisedWeightView &w, const Q8_1Block *a, float *c, int ldc, int rows, int parts,
                 const CPUProjectionRowGrid &grid, float *scratch)
        {
            auto task = [&]
            {
                constexpr auto format = static_cast<Format>(CB);
                const auto pitch = workerFloats(format, Lanes, rows, w.blocks_per_row, parts);
                auto *panel = scratch + static_cast<std::size_t>(omp_get_thread_num()) * pitch;
                auto *carry = needsCarry(w.blocks_per_row, parts)
                                  ? panel + std::min(w.blocks_per_row, depth) * blockFloats(format, Lanes)
                                  : nullptr;
                const int strips = columnStrips(w.N, Lanes);
#pragma omp for schedule(static)
                for (int strip = 0; strip < strips; ++strip)
                    stripe<Lanes, CB, P>(w, a, c, ldc, strip * Lanes, parts, grid, panel, carry);
            };
            OMP_WORKSHARE_REGION(task);
        }

        /** @brief Dispatch a validated panel format and FP policy before entering the team. */
        template <int Lanes>
        void dispatch(Format format, const CPUNativeQuantisedWeightView &w, const Q8_1Block *a, float *out, int rows,
                      int stride, int partitions, const CPUProjectionRowGrid &grid, float *scratch)
        {
            const auto call = [&]<Format F>()
            {
                constexpr auto cb = static_cast<int>(F);
                if (w.numerical_policy == Policy::GPUAlignedExpert)
                    return run<Lanes, cb, Policy::GPUAlignedExpert>(w, a, out, stride, rows, partitions, grid, scratch);
                return run<Lanes, cb, Policy::BackendNative>(w, a, out, stride, rows, partitions, grid, scratch);
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
            return dispatch<8>(format, weights, activation, output, rows, output_stride, partitions, grid,
                               scratch.data());
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        return dispatch<16>(format, weights, activation, output, rows, output_stride, partitions, grid, scratch.data());
#else
        throw std::invalid_argument("Native x86 prefill cannot select AVX-512 in its AVX2 envelope");
#endif
    }
} // namespace llaminar2::cpu::native_quantised::x86_panel
#endif
