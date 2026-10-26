/**
 * @file CPUFixedEpochGDN.h
 * @brief Grouped CPU query contraction and shared-prefix checkpoint advancement.
 *
 * Up to four queries share each checkpoint load. Every query independently
 * traverses its canonical raw history backwards, then applies value FMAs in
 * forward order. Rounded prefix advancement is shared at fixed eight-position
 * boundaries. Serial decode uses this same program with one live row. Snapshot
 * publication retains immutable checkpoints once per crossed epoch and appends
 * each raw update once; acceptance copies the selected canonical prefix without
 * adding an arithmetic boundary. Compact capture executes directly in those
 * retained checkpoints and the shared raw log, so no mutable bank is cloned.
 * Initial retention reuses each first-projection checkpoint load. Epoch-zero
 * source-to-destination advancement similarly avoids a separate clone pass.
 * Epoch flushes pipeline the next rounded decay/key sum through the current
 * update; intermediate decayed words remain exclusive until the final update.
 * Numeric helpers return completed integer work; only the CPU caller team
 * publishes collector records after its numeric workshare has completed.
 * Explicit dense serialization remains an oracle. All temporary arrays are
 * bounded thread-private scratch; the caller owns state and snapshot storage.
 */
#pragma once

#include "kernels/CPUGDNStateLayout.h"
#include "kernels/CPUSharedGDNSnapshots.h"
#include "CPUAdjustedGDNReadout.h"
#include "CPUFixedEpochQueryBlock.h"
#include "utils/CPUFeatures.h"
#include "CPUFixedEpochObservations.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
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
     * @tparam Retain Whether this first readout also owns initial matrix retention.
     * @param layout Validated per-head state geometry.
     * @param matrix Immutable key-major checkpoint with matrixFloats live elements.
     * @param history Immutable contiguous epoch raw records.
     * @param x Contracted queries with a fixed 512-float row stride.
     * @param c Forward-indexed coefficients with an eight-float row stride.
     * @param phase History length preceding this group.
     * @param first First value column in this physical tile.
     * @param output First live output row.
     * @param stride Physical output row stride in floats.
     * @param retained Exclusive retained matrix destination when Retain is true;
     *        unused for ordinary readouts. Each loaded value is stored once.
     */
    template<class SIMD, int Rows, int Vectors, bool Retain = false>
    inline void projectTile(const CPUGDNStateLayout &layout, const float *matrix, const float *history,
        const float *x, const float *c, int phase, int first, float *output, int stride, float *retained = nullptr)
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
                const auto s = SIMD::load(matrix + j * layout.valueWidth() + first + z * SIMD::width);
                if constexpr (Retain)
                    SIMD::store(retained + j * layout.valueWidth() + first + z * SIMD::width, s);
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
                const float *value = (history + static_cast<size_t>(p) * layout.recordFloats()) + layout.keyWidth() + first;
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
     * @tparam Retain Retain the initial matrix through this complete readout.
     * @param layout Validated per-head geometry.
     * @param matrix Immutable key-major checkpoint.
     * @param history Immutable contiguous epoch raw records.
     * @param x Contracted queries, stride 512.
     * @param c Ordered coefficients, stride eight.
     * @param phase Pending history before this group.
     * @param output First live output row.
     * @param stride Output row stride in floats.
     * @param retained Complete matrix destination required by Retain=true;
     *        scalar tails store only for the first row to retain each value once.
     */
    template<int Rows, bool Retain = false>
    __attribute__((noinline)) inline void project(const CPUGDNStateLayout &layout,
        const float *matrix, const float *history, const float *x, const float *c, int phase, float *output, int stride, float *retained = nullptr)
    {
        if constexpr (Retain)
            if (!retained) throw std::invalid_argument("CPU GDN initial readout requires retained checkpoint storage");
        int vi = 0;
        constexpr int vectors = Rows == 1 ? 8 : Rows == 2 ? 4 : 2;
#if defined(__AVX512F__)
        if (activeISALevel() == ISALevel::AVX512)
        {
            // Sixteen or fewer projection accumulators fit the AVX-512 file.
            // Only independent value columns are regrouped; every output keeps
            // exactly the same ascending-key and forward-history FMA chain.
            constexpr int vectors512 = Rows <= 2 ? 8 : 4;
            for (; vi + vectors512 * 16 <= layout.valueWidth(); vi += vectors512 * 16)
                projectTile<SIMD512, Rows, vectors512, Retain>(layout, matrix, history, x, c, phase, vi, output, stride, retained);
            // Keep the old blocked remainder for smaller admitted widths;
            // a wider full tile must not turn its old64/32-column tile into
            // several single-vector passes.
            if constexpr (vectors512 != vectors)
                for (; vi + vectors * 16 <= layout.valueWidth(); vi += vectors * 16)
                    projectTile<SIMD512, Rows, vectors, Retain>(layout, matrix, history, x, c, phase, vi, output, stride, retained);
            for (; vi + 16 <= layout.valueWidth(); vi += 16)
                projectTile<SIMD512, Rows, 1, Retain>(layout, matrix, history, x, c, phase, vi, output, stride, retained);
        }
#endif
#if defined(__AVX2__)
        // Three rows need more than six FMA chains to hide dependency latency.
        // Four vectors retain twelve accumulators; three query broadcasts and
        // one consumed matrix vector fill the sixteen-register AVX2 budget.
        // Serial rows and all other row counts retain their proven geometry.
        constexpr int vectors256 = Rows == 3 ? 4 : vectors;
        if (activeISALevel() >= ISALevel::AVX2)
        {
            for (; vi + vectors256 * 8 <= layout.valueWidth(); vi += vectors256 * 8)
                projectTile<SIMD256, Rows, vectors256, Retain>(layout, matrix, history, x, c, phase, vi, output, stride, retained);
            for (; vi + 8 <= layout.valueWidth(); vi += 8)
                projectTile<SIMD256, Rows, 1, Retain>(layout, matrix, history, x, c, phase, vi, output, stride, retained);
        }
#endif
        for (; vi < layout.valueWidth(); ++vi)
            for (int r = 0; r < Rows; ++r)
            {
                float sum = 0;
                for (int j = 0; j < layout.keyWidth(); ++j)
                {
                    const float checkpoint = matrix[j * layout.valueWidth() + vi];
                    if constexpr (Retain)
                        if (r == 0) retained[j * layout.valueWidth() + vi] = checkpoint;
                    sum = std::fma(checkpoint, x[r * 512 + j], sum);
                }
                for (int p = 0; p <= phase + r; ++p)
                    sum = std::fma(c[r * CPUGDNStateLayout::kPeriod + p],
                        (history + static_cast<size_t>(p) * layout.recordFloats())[layout.keyWidth() + vi], sum);
                output[r * stride + vi] = sum;
            }
    }

    /**
     * @brief Apply the ordered rounded epoch program to one exclusive checkpoint tile.
     * @tparam SIMD Independent value lanes.
     * @tparam Vectors Number of independent value vectors retained in registers.
     * @param layout Validated head geometry.
     * @param matrix Exclusive mutable next checkpoint.
     * @param history Eight immutable raw records in canonical forward order.
     * @param first First value column in this tile.
     * @param initial_source Immutable predecessor for update zero. Subsequent
     *        updates borrow the newly computed destination in place. It may alias
     *        matrix for ordinary encoded-state advancement.
     *
     * The first projection seeds rounded decayed state in matrix. Each later
     * projection is formed during the preceding update in increasing key order.
     * matrix temporarily holds the next rounded decayed state and is never
     * published until update seven leaves the final canonical checkpoint.
     */
    template<class SIMD, int Vectors>
    inline void flushTile(const CPUGDNStateLayout &layout, float *matrix, const float *history, int first, const float *initial_source)
    {
        // Seed the first ordered projection while storing its explicitly
        // rounded decayed words in the exclusive destination. Exact aliasing
        // is safe: this pass never reads an earlier key after writing it.
        const float *raw = history;
        const auto first_decay = SIMD::broadcast(raw[layout.keyWidth() + layout.valueWidth()]);
        typename SIMD::Value sums[Vectors];
#pragma GCC unroll 4
        for (int z = 0; z < Vectors; ++z) sums[z] = SIMD::zero();
        for (int j = 0; j < layout.keyWidth(); ++j)
        {
            const auto key = SIMD::broadcast(raw[j]);
#pragma GCC unroll 4
            for (int z = 0; z < Vectors; ++z)
            {
                const auto decayed = SIMD::mul(SIMD::load(
                    initial_source + j * layout.valueWidth() + first + z * SIMD::width), first_decay);
                SIMD::store(matrix + j * layout.valueWidth() + first + z * SIMD::width, decayed);
                sums[z] = SIMD::fma(decayed, key, sums[z]);
            }
        }
        for (int p = 0; p < CPUGDNStateLayout::kPeriod; ++p)
        {
            raw = history + static_cast<size_t>(p) * layout.recordFloats();
            const auto beta = SIMD::broadcast(raw[layout.keyWidth() + layout.valueWidth() + 1]);
#pragma GCC unroll 4
            for (int z = 0; z < Vectors; ++z)
                sums[z] = SIMD::mul(SIMD::sub(SIMD::load(
                    raw + layout.keyWidth() + first + z * SIMD::width), sums[z]), beta);
            if (p + 1 < CPUGDNStateLayout::kPeriod)
            {
                const float *next = raw + layout.recordFloats();
                const auto next_decay = SIMD::broadcast(next[layout.keyWidth() + layout.valueWidth()]);
                typename SIMD::Value next_sums[Vectors];
#pragma GCC unroll 4
                for (int z = 0; z < Vectors; ++z) next_sums[z] = SIMD::zero();
                for (int j = 0; j < layout.keyWidth(); ++j)
                {
                    const auto key = SIMD::broadcast(raw[j]);
                    const auto next_key = SIMD::broadcast(next[j]);
#pragma GCC unroll 4
                    for (int z = 0; z < Vectors; ++z)
                    {
                        float *target = matrix + j * layout.valueWidth() + first + z * SIMD::width;
                        const auto updated = SIMD::fma(key, sums[z], SIMD::load(target));
                        const auto decayed = SIMD::mul(updated, next_decay);
                        next_sums[z] = SIMD::fma(decayed, next_key, next_sums[z]);
                        SIMD::store(target, decayed);
                    }
                }
#pragma GCC unroll 4
                for (int z = 0; z < Vectors; ++z) sums[z] = next_sums[z];
            }
            else
            {
                // Only the terminal update becomes a canonical checkpoint.
                // No reader may observe this exclusive tile before completion.
                for (int j = 0; j < layout.keyWidth(); ++j)
                {
                    const auto key = SIMD::broadcast(raw[j]);
#pragma GCC unroll 4
                    for (int z = 0; z < Vectors; ++z)
                    {
                        float *target = matrix + j * layout.valueWidth() + first + z * SIMD::width;
                        SIMD::store(target, SIMD::fma(key, sums[z], SIMD::load(target)));
                    }
                }
            }
        }
    }

    /**
     * @brief Advance exactly eight raw updates into an exclusively owned checkpoint.
     * @param layout Validated physical head geometry.
     * @param matrix Mutable key-major checkpoint; may be a retained next-epoch slot.
     * @param history Eight immutable raw records in increasing position order.
     * @param initial_source Predecessor matrix, possibly equal to matrix. The
     *        first projection reads this source and seeds matrix. Pipelined
     *        updates retain the next rounded decayed state in that exclusive
     *        destination; only update seven publishes canonical checkpoint bits.
     *        SIMD lanes and scalar value tails execute the same FP32 sequence.
     */
    inline void flushCheckpoint(const CPUGDNStateLayout &layout, float *matrix, const float *history, const float *initial_source)
    {
        int vi = 0;
#if defined(__AVX512F__)
        if (activeISALevel() == ISALevel::AVX512)
        {
            for (; vi + 64 <= layout.valueWidth(); vi += 64) flushTile<SIMD512, 4>(layout, matrix, history, vi, initial_source);
            for (; vi + 16 <= layout.valueWidth(); vi += 16) flushTile<SIMD512, 1>(layout, matrix, history, vi, initial_source);
        }
#endif
#if defined(__AVX2__)
        if (activeISALevel() >= ISALevel::AVX2)
        {
            for (; vi + 32 <= layout.valueWidth(); vi += 32) flushTile<SIMD256, 4>(layout, matrix, history, vi, initial_source);
            for (; vi + 8 <= layout.valueWidth(); vi += 8) flushTile<SIMD256, 1>(layout, matrix, history, vi, initial_source);
        }
#endif
        for (; vi < layout.valueWidth(); ++vi)
        {
            const float *raw = history;
            const float first_decay = raw[layout.keyWidth() + layout.valueWidth()];
            float projection = 0.0f;
            for (int j = 0; j < layout.keyWidth(); ++j)
            {
                const float decayed = initial_source[j * layout.valueWidth() + vi] * first_decay;
                matrix[j * layout.valueWidth() + vi] = decayed;
                projection = std::fma(decayed, raw[j], projection);
            }
            for (int p = 0; p < CPUGDNStateLayout::kPeriod; ++p)
            {
                raw = history + static_cast<size_t>(p) * layout.recordFloats();
                const float delta = (raw[layout.keyWidth() + vi] - projection) *
                    raw[layout.keyWidth() + layout.valueWidth() + 1];
                if (p + 1 < CPUGDNStateLayout::kPeriod)
                {
                    const float *next = raw + layout.recordFloats();
                    const float next_decay = next[layout.keyWidth() + layout.valueWidth()];
                    float next_projection = 0.0f;
                    for (int j = 0; j < layout.keyWidth(); ++j)
                    {
                        float *target = matrix + j * layout.valueWidth() + vi;
                        const float updated = std::fma(raw[j], delta, *target);
                        const float decayed = updated * next_decay;
                        next_projection = std::fma(decayed, next[j], next_projection);
                        *target = decayed;
                    }
                    projection = next_projection;
                }
                else
                    for (int j = 0; j < layout.keyWidth(); ++j)
                    {
                        float *target = matrix + j * layout.valueWidth() + vi;
                        *target = std::fma(raw[j], delta, *target);
                    }
            }
        }
    }

    /**
     * @brief Advance an encoded ordinary state and retire its completed history.
     * @param layout Validated complete-state geometry.
     * @param state Exclusive encoded head with exactly eight pending records.
     */
    inline void flush(const CPUGDNStateLayout &layout, float *state)
    {
        flushCheckpoint(layout, state + 1, layout.record(state, 0), state + 1);
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
     * @param snapshot_stride Complete bank stride between dense oracle rows.
     * @return Actual completed head work for the existing caller team to aggregate.
     */
    inline CPUFixedEpochObservations run(const CPUGDNStateLayout &layout, const float *initial, float *working,
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
            }
            const float *queries=q+static_cast<size_t>(first)*qk_stride;
            switch(count) {
                case 1: contractQueries<1>(layout,layout.record(working,0),queries,qk_stride,phase,x,coefficients); break;
                case 2: contractQueries<2>(layout,layout.record(working,0),queries,qk_stride,phase,x,coefficients); break;
                case 3: contractQueries<3>(layout,layout.record(working,0),queries,qk_stride,phase,x,coefficients); break;
                case 4: contractQueries<4>(layout,layout.record(working,0),queries,qk_stride,phase,x,coefficients); break;
            }
            float *out = output + static_cast<std::size_t>(first) * output_stride;
            switch (count)
            {
                case 1: project<1>(layout, working+1, layout.record(working,0), x, coefficients, phase, out, output_stride); break;
                case 2: project<2>(layout, working+1, layout.record(working,0), x, coefficients, phase, out, output_stride); break;
                case 3: project<3>(layout, working+1, layout.record(working,0), x, coefficients, phase, out, output_stride); break;
                case 4: project<4>(layout, working+1, layout.record(working,0), x, coefficients, phase, out, output_stride); break;
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
        return {.heads=1u,
            .projection_groups=static_cast<std::uint64_t>(projection_groups),
            .grouped_rows=static_cast<std::uint64_t>(grouped_rows),
            .checkpoint_flushes=static_cast<std::uint64_t>(checkpoint_flushes)};
    }
    /**
     * @brief Execute compact capture directly in retained checkpoint/log slots.
     * @param layout Validated per-head physical and arithmetic geometry.
     * @param initial Immutable canonical input head; never modified or cloned by a separate pass.
     * @param q Prepared query rows, separated by qk_stride floats.
     * @param k Prepared key rows, separated by qk_stride floats.
     * @param v Raw value rows, separated by v_stride floats.
     * @param decay Rounded decay factors separated by gate_stride floats.
     * @param beta Rounded write strengths separated by gate_stride floats.
     * @param rows Positive live candidate count admitted by shared storage.
     * @param qk_stride Prepared query/key physical row stride in floats.
     * @param v_stride Raw value physical row stride in floats.
     * @param gate_stride Rounded gate physical row stride in floats.
     * @param output First live output row, overwritten completely.
     * @param output_stride Output row stride in floats.
     * @param shared Exclusive unpublished retained transaction; begin retains the
     *        initial history once; the synchronous first readout retains cp0
     *        before complete can publish any candidate.
     * @return Actual completed retained work; no observation escapes on failure.
     *
     * Current checkpoints are immutable. A fixed epoch boundary derives its
     * next retained slot directly from the predecessor using exactly eight raw
     * updates, without a separate matrix clone. Queries and projections borrow
     * the same numeric helpers as ordinary encoded-state execution. No mutable canonical bank,
     * row serialization, inactive-history clearing, or partial flush is needed.
     */
    inline CPUFixedEpochObservations runShared(const CPUGDNStateLayout &layout, const float *initial,
        const float *q, const float *k, const float *v, const float *decay, const float *beta,
        int rows, int qk_stride, int v_stride, int gate_stride, float *output, int output_stride,
        CPUSharedGDNSnapshots &shared)
    {
        if (rows <= 0)
            throw std::invalid_argument("CPU GDN retained readout requires live rows; empty participants retire metadata");
        alignas(64) float x[4 * 512], coefficients[4 * CPUGDNStateLayout::kPeriod];
        int initial_phase = 0, first = 0;
        int projection_groups = 0, grouped_rows = 0, checkpoint_flushes = 0;
        // Both first-retaining and later readouts execute this one FP32 program.
        // Retain is compile-time so ordinary readouts carry no conditional store
        // inside the key loop. Scratch remains bounded and thread-private.
        auto readout_group = [&]<bool Retain>(const float *matrix, float *retained)
        {
            const int prefix = initial_phase + first;
            const int epoch = prefix / CPUGDNStateLayout::kPeriod;
            const int phase = prefix % CPUGDNStateLayout::kPeriod;
            const int count = std::min({4, rows - first, CPUGDNStateLayout::kPeriod - phase});
            const float *const history = shared.record(epoch * CPUGDNStateLayout::kPeriod);
            for (int r = 0; r < count; ++r)
            {
                const int row = first + r;
                float *const raw = shared.record(initial_phase + row);
                std::memcpy(raw, k + static_cast<size_t>(row) * qk_stride, layout.keyWidth() * sizeof(float));
                std::memcpy(raw + layout.keyWidth(), v + static_cast<size_t>(row) * v_stride, layout.valueWidth() * sizeof(float));
                raw[layout.keyWidth() + layout.valueWidth()] = decay[row * gate_stride];
                raw[layout.keyWidth() + layout.valueWidth() + 1] = beta[row * gate_stride];
            }
            const float *const queries = q + static_cast<size_t>(first) * qk_stride;
            switch (count)
            {
                case 1: contractQueries<1>(layout, history, queries, qk_stride, phase, x, coefficients); break;
                case 2: contractQueries<2>(layout, history, queries, qk_stride, phase, x, coefficients); break;
                case 3: contractQueries<3>(layout, history, queries, qk_stride, phase, x, coefficients); break;
                case 4: contractQueries<4>(layout, history, queries, qk_stride, phase, x, coefficients); break;
            }
            float *const out = output + static_cast<size_t>(first) * output_stride;
            switch (count)
            {
                case 1: project<1, Retain>(layout, matrix, history, x, coefficients, phase, out, output_stride, retained); break;
                case 2: project<2, Retain>(layout, matrix, history, x, coefficients, phase, out, output_stride, retained); break;
                case 3: project<3, Retain>(layout, matrix, history, x, coefficients, phase, out, output_stride, retained); break;
                case 4: project<4, Retain>(layout, matrix, history, x, coefficients, phase, out, output_stride, retained); break;
            }
            ++projection_groups;
            if (count > 1) grouped_rows += count;
            first += count;
        };
        // The callback runs after phase/span validation. Its successful return
        // proves every initial matrix value was retained through the first full
        // projection, including scalar tails, before publication is permitted.
        shared.beginWithInitialReadout(initial, rows,
            [&](const float *source, float *retained)
            {
                initial_phase = static_cast<int>(initial[0]);
                readout_group.template operator()<true>(source, retained);
            });
        auto advance_completed_epoch = [&]()
        {
            const int prefix = initial_phase + first;
            if (prefix % CPUGDNStateLayout::kPeriod != 0) return;
            const int epoch = prefix / CPUGDNStateLayout::kPeriod;
            // Update zero reads the immutable predecessor and writes the next
            // checkpoint directly. Updates one through seven are in place on
            // that next slot, with the same rounded products and FMA order.
            flushCheckpoint(layout, shared.checkpoint(epoch),
                shared.record((epoch - 1) * CPUGDNStateLayout::kPeriod), shared.checkpoint(epoch - 1));
            ++checkpoint_flushes;
        };
        advance_completed_epoch();
        while (first < rows)
        {
            const int epoch = (initial_phase + first) / CPUGDNStateLayout::kPeriod;
            readout_group.template operator()<false>(shared.checkpoint(epoch), nullptr);
            advance_completed_epoch();
        }
        shared.complete();
        return {.heads=1u, .shared_heads=1u,
            .projection_groups=static_cast<std::uint64_t>(projection_groups),
            .grouped_rows=static_cast<std::uint64_t>(grouped_rows),
            .checkpoint_flushes=static_cast<std::uint64_t>(checkpoint_flushes),
            .shared_flushes=static_cast<std::uint64_t>(checkpoint_flushes),
            .raw_records=static_cast<std::uint64_t>(initial_phase + rows),
            .candidate_rows=static_cast<std::uint64_t>(rows)};
    }
}
