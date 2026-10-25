/**
 * @file CPUFixedEpochGDN.h
 * @brief Grouped CPU query contraction and shared-prefix checkpoint advancement.
 *
 * Up to four queries share each checkpoint load. Every query independently
 * traverses its canonical raw history backwards, then applies value FMAs in
 * forward order. Rounded prefix advancement is shared at fixed eight-position
 * boundaries. Serial decode uses this same program with one live row. Snapshot
 * publication copies the complete phase/checkpoint/history value without adding
 * an arithmetic boundary. All temporary arrays are bounded thread-private
 * scratch; the caller owns state and snapshot storage.
 */
#pragma once

#include "kernels/CPUGDNStateLayout.h"
#include "CPUAdjustedGDNReadout.h"
#include "utils/CPUFeatures.h"
#include "utils/PerfStatsCollector.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace llaminar2::cpu_gdn_epoch
{
#if defined(__AVX2__)
    /** @brief AVX2 lanes own independent values; key reductions remain ordered. */
    struct SIMD256
    {
        using Value = __m256;
        static constexpr int width = 8;
        /** @return A zero accumulator. */
        static Value zero() { return _mm256_setzero_ps(); }
        /** @return One scalar broadcast to independent value lanes. */
        static Value broadcast(float v) { return _mm256_set1_ps(v); }
        /** @return Unaligned live value lanes. */
        static Value load(const float *p) { return _mm256_loadu_ps(p); }
        /** @brief Publish exactly eight live values. */
        static void store(float *p, Value v) { _mm256_storeu_ps(p, v); }
        /** @return One explicitly rounded multiply. */
        static Value mul(Value a, Value b) { return _mm256_mul_ps(a, b); }
        /** @return One explicitly rounded subtraction. */
        static Value sub(Value a, Value b) { return _mm256_sub_ps(a, b); }
        /** @return One correctly rounded fused multiply-add per lane. */
        static Value fma(Value a, Value b, Value c) { return _mm256_fmadd_ps(a, b, c); }
    };
#endif
#if defined(__AVX512F__)
    /** @brief AVX-512 implements the identical per-value FP32 program. */
    struct SIMD512
    {
        using Value = __m512;
        static constexpr int width = 16;
        /** @return A zero accumulator. */
        static Value zero() { return _mm512_setzero_ps(); }
        /** @return One scalar broadcast to independent value lanes. */
        static Value broadcast(float v) { return _mm512_set1_ps(v); }
        /** @return Unaligned live value lanes. */
        static Value load(const float *p) { return _mm512_loadu_ps(p); }
        /** @brief Publish exactly sixteen live values. */
        static void store(float *p, Value v) { _mm512_storeu_ps(p, v); }
        /** @return One explicitly rounded multiply. */
        static Value mul(Value a, Value b) { return _mm512_mul_ps(a, b); }
        /** @return One explicitly rounded subtraction. */
        static Value sub(Value a, Value b) { return _mm512_sub_ps(a, b); }
        /** @return One correctly rounded fused multiply-add per lane. */
        static Value fma(Value a, Value b, Value c) { return _mm512_fmadd_ps(a, b, c); }
    };
#endif

    /**
     * @brief Project independent queries while loading each checkpoint tile once.
     * @tparam SIMD ISA-specific independent value lanes.
     * @tparam Rows Number of live queries, fixed at 1..4 for register allocation.
     * @tparam Vectors Independent value vectors per query.
     * @param layout Validated per-head state geometry.
     * @param state Complete immutable checkpoint/history state.
     * @param x Contracted queries with a fixed 512-float row stride.
     * @param c Forward-indexed coefficients with an eight-float row stride.
     * @param phase History length preceding this group.
     * @param first First value column in this physical tile.
     * @param output First live output row.
     * @param stride Physical output row stride in floats.
     */
    template<class SIMD, int Rows, int Vectors>
    inline void projectTile(const CPUGDNStateLayout &layout, const float *state,
        const float *x, const float *c, int phase, int first, float *output, int stride)
    {
        typename SIMD::Value sums[Rows][Vectors];
#pragma GCC unroll 4
        for (int r = 0; r < Rows; ++r)
#pragma GCC unroll 8
            for (int z = 0; z < Vectors; ++z) sums[r][z] = SIMD::zero();
        for (int j = 0; j < layout.keyWidth(); ++j)
        {
#pragma GCC unroll 8
            for (int z = 0; z < Vectors; ++z)
            {
                const auto s = SIMD::load(state + 1 + j * layout.valueWidth() + first + z * SIMD::width);
#pragma GCC unroll 4
                for (int r = 0; r < Rows; ++r)
                    sums[r][z] = SIMD::fma(s, SIMD::broadcast(x[r * 512 + j]), sums[r][z]);
            }
        }
#pragma GCC unroll 4
        for (int r = 0; r < Rows; ++r)
        {
            for (int p = 0; p <= phase + r; ++p)
            {
                const auto coefficient = SIMD::broadcast(c[r * CPUGDNStateLayout::kPeriod + p]);
                const float *value = layout.record(state, p) + layout.keyWidth() + first;
#pragma GCC unroll 8
                for (int z = 0; z < Vectors; ++z)
                    sums[r][z] = SIMD::fma(coefficient, SIMD::load(value + z * SIMD::width), sums[r][z]);
            }
#pragma GCC unroll 8
            for (int z = 0; z < Vectors; ++z)
                SIMD::store(output + r * stride + first + z * SIMD::width, sums[r][z]);
        }
    }

    /**
     * @brief Dispatch a complete fixed-row readout without changing arithmetic.
     * @tparam Rows Live query count, 1..4.
     * @param layout Validated per-head geometry.
     * @param state Complete checkpoint/history value, borrowed read-only.
     * @param x Contracted queries, stride 512.
     * @param c Ordered coefficients, stride eight.
     * @param phase Pending history before this group.
     * @param output First live output row.
     * @param stride Output row stride in floats.
     */
    template<int Rows>
    __attribute__((noinline)) inline void project(const CPUGDNStateLayout &layout,
        const float *state, const float *x, const float *c, int phase, float *output, int stride)
    {
        int vi = 0;
        constexpr int vectors = Rows == 1 ? 8 : Rows == 2 ? 4 : 2;
#if defined(__AVX512F__)
        if (activeISALevel() == ISALevel::AVX512)
        {
            for (; vi + vectors * 16 <= layout.valueWidth(); vi += vectors * 16)
                projectTile<SIMD512, Rows, vectors>(layout, state, x, c, phase, vi, output, stride);
            for (; vi + 16 <= layout.valueWidth(); vi += 16)
                projectTile<SIMD512, Rows, 1>(layout, state, x, c, phase, vi, output, stride);
        }
#endif
#if defined(__AVX2__)
        if (activeISALevel() >= ISALevel::AVX2)
        {
            for (; vi + vectors * 8 <= layout.valueWidth(); vi += vectors * 8)
                projectTile<SIMD256, Rows, vectors>(layout, state, x, c, phase, vi, output, stride);
            for (; vi + 8 <= layout.valueWidth(); vi += 8)
                projectTile<SIMD256, Rows, 1>(layout, state, x, c, phase, vi, output, stride);
        }
#endif
        for (; vi < layout.valueWidth(); ++vi)
            for (int r = 0; r < Rows; ++r)
            {
                float sum = 0;
                for (int j = 0; j < layout.keyWidth(); ++j)
                    sum = std::fma(state[1 + j * layout.valueWidth() + vi], x[r * 512 + j], sum);
                for (int p = 0; p <= phase + r; ++p)
                    sum = std::fma(c[r * CPUGDNStateLayout::kPeriod + p],
                        layout.record(state, p)[layout.keyWidth() + vi], sum);
                output[r * stride + vi] = sum;
            }
    }

    /**
     * @brief Share the ordered rounded prefix computation for one checkpoint.
     * @tparam SIMD Independent value lanes.
     * @tparam Vectors Number of independent value vectors retained in registers.
     * @param layout Validated head geometry.
     * @param state Complete mutable state with eight live raw updates.
     * @param first First value column in this tile.
     */
    template<class SIMD, int Vectors>
    inline void flushTile(const CPUGDNStateLayout &layout, float *state, int first)
    {
        for (int p = 0; p < CPUGDNStateLayout::kPeriod; ++p)
        {
            const float *raw = layout.record(state, p);
            const auto decay = SIMD::broadcast(raw[layout.keyWidth() + layout.valueWidth()]);
            const auto beta = SIMD::broadcast(raw[layout.keyWidth() + layout.valueWidth() + 1]);
            typename SIMD::Value sums[Vectors];
#pragma GCC unroll 4
            for (int z = 0; z < Vectors; ++z) sums[z] = SIMD::zero();
            for (int j = 0; j < layout.keyWidth(); ++j)
            {
                const auto key = SIMD::broadcast(raw[j]);
#pragma GCC unroll 4
                for (int z = 0; z < Vectors; ++z)
                    sums[z] = SIMD::fma(SIMD::mul(SIMD::load(state + 1 + j * layout.valueWidth() + first + z * SIMD::width), decay), key, sums[z]);
            }
#pragma GCC unroll 4
            for (int z = 0; z < Vectors; ++z)
                sums[z] = SIMD::mul(SIMD::sub(SIMD::load(raw + layout.keyWidth() + first + z * SIMD::width), sums[z]), beta);
            for (int j = 0; j < layout.keyWidth(); ++j)
            {
                const auto key = SIMD::broadcast(raw[j]);
#pragma GCC unroll 4
                for (int z = 0; z < Vectors; ++z)
                {
                    float *target = state + 1 + j * layout.valueWidth() + first + z * SIMD::width;
                    SIMD::store(target, SIMD::fma(key, sums[z], SIMD::mul(SIMD::load(target), decay)));
                }
            }
        }
    }

    /**
     * @brief Advance exactly eight updates, then retire all history to zero.
     * @param layout Validated physical head geometry.
     * @param state Complete mutable state with eight live raw records.
     */
    inline void flush(const CPUGDNStateLayout &layout, float *state)
    {
        int vi = 0;
#if defined(__AVX512F__)
        if (activeISALevel() == ISALevel::AVX512)
        {
            for (; vi + 64 <= layout.valueWidth(); vi += 64) flushTile<SIMD512, 4>(layout, state, vi);
            for (; vi + 16 <= layout.valueWidth(); vi += 16) flushTile<SIMD512, 1>(layout, state, vi);
        }
#endif
#if defined(__AVX2__)
        if (activeISALevel() >= ISALevel::AVX2)
        {
            for (; vi + 32 <= layout.valueWidth(); vi += 32) flushTile<SIMD256, 4>(layout, state, vi);
            for (; vi + 8 <= layout.valueWidth(); vi += 8) flushTile<SIMD256, 1>(layout, state, vi);
        }
#endif
        for (; vi < layout.valueWidth(); ++vi)
            for (int p = 0; p < CPUGDNStateLayout::kPeriod; ++p)
            {
                const float *raw = layout.record(state, p);
                const float decay = raw[layout.keyWidth() + layout.valueWidth()];
                float projection = 0;
                for (int j = 0; j < layout.keyWidth(); ++j)
                {
                    float &s = state[1 + j * layout.valueWidth() + vi];
                    s *= decay;
                    projection = std::fma(s, raw[j], projection);
                }
                const float delta = (raw[layout.keyWidth() + vi] - projection) * raw[layout.keyWidth() + layout.valueWidth() + 1];
                for (int j = 0; j < layout.keyWidth(); ++j)
                {
                    float &s = state[1 + j * layout.valueWidth() + vi];
                    s = std::fma(raw[j], delta, s);
                }
            }
        std::fill(layout.record(state, 0), state + layout.headFloats(), 0.0f);
        state[0] = 0;
    }

    /**
     * @brief Publish one complete state, clearing inactive raw history.
     * @param layout Validated physical layout.
     * @param source Immutable source checkpoint/history with at least phase records.
     * @param target Snapshot destination, or the exact source for the terminal row.
     * @param phase Live history length in the published value.
     */
    inline void publish(const CPUGDNStateLayout &layout, const float *source, float *target, int phase)
    {
        if (target != source)
        {
            const int live = layout.matrixFloats() + phase * layout.recordFloats();
            std::memcpy(target + 1, source + 1, static_cast<std::size_t>(live) * sizeof(float));
            std::fill(target + 1 + live, target + layout.headFloats(), 0.0f);
        }
        target[0] = static_cast<float>(phase);
    }

    /**
     * @brief Execute a complete head's live rows with common serial arithmetic.
     * @param layout Validated geometry; all spans were admitted by the caller.
     * @param initial Immutable initial head, possibly equal to working.
     * @param working Mutable complete head; the last snapshot may own this bank.
     * @param q Prepared queries, with qk_stride between rows.
     * @param k Prepared keys, with qk_stride between rows.
     * @param v Raw values, with v_stride between rows.
     * @param decay Rounded decay factors, with gate_stride between rows.
     * @param beta Rounded write strengths, with gate_stride between rows.
     * @param rows Positive number of live candidate rows.
     * @param qk_stride Physical query/key row stride.
     * @param v_stride Physical raw-value row stride.
     * @param gate_stride Physical gate row stride.
     * @param output First output row.
     * @param output_stride Physical output row stride.
     * @param snapshots Optional first post-row snapshot for this head.
     * @param snapshot_stride Complete bank stride between snapshot rows.
     */
    inline void run(const CPUGDNStateLayout &layout, const float *initial, float *working,
        const float *q, const float *k, const float *v, const float *decay, const float *beta,
        int rows, int qk_stride, int v_stride, int gate_stride, float *output, int output_stride,
        float *snapshots = nullptr, int snapshot_stride = 0)
    {
        if (initial != working)
            std::memcpy(working, initial, static_cast<std::size_t>(layout.headFloats()) * sizeof(float));
        alignas(64) float x[4 * 512], coefficients[4 * CPUGDNStateLayout::kPeriod];
        int projection_groups = 0, grouped_rows = 0, checkpoint_flushes = 0;
        for (int first = 0; first < rows;)
        {
            const int phase = static_cast<int>(working[0]);
            const int count = std::min({4, rows - first, CPUGDNStateLayout::kPeriod - phase});
            for (int r = 0; r < count; ++r)
            {
                const int row = first + r;
                float *raw = layout.record(working, phase + r);
                std::memcpy(raw, k + static_cast<std::size_t>(row) * qk_stride, layout.keyWidth() * sizeof(float));
                std::memcpy(raw + layout.keyWidth(), v + static_cast<std::size_t>(row) * v_stride, layout.valueWidth() * sizeof(float));
                raw[layout.keyWidth() + layout.valueWidth()] = decay[row * gate_stride];
                raw[layout.keyWidth() + layout.valueWidth() + 1] = beta[row * gate_stride];
                float *query = x + r * 512;
                std::memcpy(query, q + static_cast<std::size_t>(row) * qk_stride, layout.keyWidth() * sizeof(float));
                for (int p = phase + r; p >= 0; --p)
                {
                    const float *u = layout.record(working, p);
                    coefficients[r * CPUGDNStateLayout::kPeriod + p] = prepareAdjustedGDNReadout(
                        query, u, u[layout.keyWidth() + layout.valueWidth()],
                        u[layout.keyWidth() + layout.valueWidth() + 1], layout.keyWidth(), query);
                }
            }
            float *out = output + static_cast<std::size_t>(first) * output_stride;
            switch (count)
            {
                case 1: project<1>(layout, working, x, coefficients, phase, out, output_stride); break;
                case 2: project<2>(layout, working, x, coefficients, phase, out, output_stride); break;
                case 3: project<3>(layout, working, x, coefficients, phase, out, output_stride); break;
                case 4: project<4>(layout, working, x, coefficients, phase, out, output_stride); break;
            }
            ++projection_groups;
            if (count > 1) grouped_rows += count;
            // Publish before flushing: every non-boundary row owns the old
            // checkpoint plus exactly its own prefix of raw history.
            if (snapshots)
                for (int r = 0; r < count; ++r)
                    if (phase + r + 1 < CPUGDNStateLayout::kPeriod)
                        publish(layout, working, snapshots + static_cast<std::size_t>(first + r) * snapshot_stride, phase + r + 1);
            if (phase + count == CPUGDNStateLayout::kPeriod)
            {
                flush(layout, working);
                ++checkpoint_flushes;
                if (snapshots)
                    publish(layout, working, snapshots + static_cast<std::size_t>(first + count - 1) * snapshot_stride, 0);
            }
            else working[0] = static_cast<float>(phase + count);
            first += count;
        }
        if (PerfStatsCollector::isDomainEnabled("kernel"))
        {
            PerfStatsCollector::addCounter("kernel", "cpu_gdn_epoch_projection_groups",
                projection_groups, "recurrence", "cpu");
            PerfStatsCollector::addCounter("kernel", "cpu_gdn_epoch_grouped_projection_rows",
                grouped_rows, "recurrence", "cpu");
            PerfStatsCollector::addCounter("kernel", "cpu_gdn_epoch_checkpoint_flushes",
                checkpoint_flushes, "recurrence", "cpu");
        }
    }
}
