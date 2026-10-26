/**
 * @file ExactLeafCPU.cpp
 * @brief Research-only grouped, byte-exact output-only GDN leaf evaluation.
 *
 * Preserve every FP32 multiplication, FMA, and increasing-key reduction from
 * native CPU decode. Four sibling rows share parent-state loads; each row keeps
 * its own accumulator. Dense leaf states are never written. Corrections remain
 * live until acceptance selects a compact (parent, decay, key, correction)
 * record. A pending parent is materialized during the next first key pass,
 * without a separate state read or a repeated key projection for that update.
 * This is a memory-traffic experiment, not the proposal's projection-saving
 * adjusted-query algebra and not an installed production execution path.
 */

#include <immintrin.h>

namespace
{
    /** @brief Dense snapshots are an explicit matched traffic control. */
    enum class Publication { CompactCorrections, DenseSnapshots };
    /** @brief A parent is resident or one exactly resolved update away. */
    enum class ParentEncoding { Dense, Pending };

    /**
     * @brief Evaluate a fixed sibling-row tile over eight independent values.
     * @tparam Rows Number of live rows, bounded by the physical four-row tile.
     * @tparam Publish Whether the matched control also writes dense states.
     * @tparam Parent Whether this first tile resolves a pending parent in place
     *         into the distinct persistent @p resolved_parent destination.
     * @param dk Key width; all reductions visit exactly [0,dk) in order.
     * @param dv Full value stride; this candidate requires a multiple of eight.
     * @param vi First value column owned by this tile.
     * @param state Immutable parent or pending update's immutable parent.
     * @param q Scaled queries, contiguous [Rows,dk].
     * @param k Keys, contiguous [Rows,dk].
     * @param v Values, contiguous [Rows,dv].
     * @param decay Already-computed FP32 decay gates, [Rows].
     * @param beta Already-computed FP32 write gates, [Rows].
     * @param output Destination [Rows,dv], overwritten without reading it.
     * @param corrections Destination [Rows,dv], retained through acceptance.
     * @param snapshots Dense-control destination [Rows,dk,dv], otherwise null.
     * @param pending_decay Decay of the accepted pending update.
     * @param pending_key Key of the accepted pending update, [dk].
     * @param pending_u Exact correction of the accepted pending update, [dv].
     * @param resolved_parent Destination of the old pending parent, [dk,dv].
     */
    template<int Rows, Publication Publish, ParentEncoding Parent>
    inline void tile(int dk, int dv, int vi, const float *state,
                     const float *q, const float *k, const float *v,
                     const float *decay, const float *beta,
                     float *output, float *corrections, float *snapshots,
                     float pending_decay, const float *pending_key,
                     const float *pending_u, float *resolved_parent)
    {
        __m256 accum[Rows];
        for (int row = 0; row < Rows; ++row)
            accum[row] = _mm256_setzero_ps();
        for (int j = 0; j < dk; ++j)
        {
            __m256 parent = _mm256_loadu_ps(state + j * dv + vi);
            if constexpr (Parent == ParentEncoding::Pending)
            {
                parent = _mm256_fmadd_ps(_mm256_set1_ps(pending_key[j]),
                    _mm256_loadu_ps(pending_u + vi),
                    _mm256_mul_ps(parent, _mm256_set1_ps(pending_decay)));
                _mm256_storeu_ps(resolved_parent + j * dv + vi, parent);
            }
            for (int row = 0; row < Rows; ++row)
            {
                const __m256 decayed = _mm256_mul_ps(parent, _mm256_set1_ps(decay[row]));
                accum[row] = _mm256_fmadd_ps(decayed, _mm256_set1_ps(k[row * dk + j]), accum[row]);
            }
        }
        __m256 delta[Rows];
        for (int row = 0; row < Rows; ++row)
        {
            delta[row] = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(v + row * dv + vi), accum[row]),
                                     _mm256_set1_ps(beta[row]));
            _mm256_storeu_ps(corrections + row * dv + vi, delta[row]);
            accum[row] = _mm256_setzero_ps();
        }
        const float *read_parent = Parent == ParentEncoding::Pending ? resolved_parent : state;
        for (int j = 0; j < dk; ++j)
        {
            const __m256 parent = _mm256_loadu_ps(read_parent + j * dv + vi);
            for (int row = 0; row < Rows; ++row)
            {
                const __m256 decayed = _mm256_mul_ps(parent, _mm256_set1_ps(decay[row]));
                const __m256 updated = _mm256_fmadd_ps(_mm256_set1_ps(k[row * dk + j]), delta[row], decayed);
                if constexpr (Publish == Publication::DenseSnapshots)
                    _mm256_storeu_ps(snapshots + (row * dk + j) * dv + vi, updated);
                accum[row] = _mm256_fmadd_ps(updated, _mm256_set1_ps(q[row * dk + j]), accum[row]);
            }
        }
        for (int row = 0; row < Rows; ++row)
            _mm256_storeu_ps(output + row * dv + vi, accum[row]);
    }

    /**
     * @brief Tile arbitrary live sibling counts without capacity-sized work.
     * @tparam Publish Compact output or matched dense-state traffic control.
     * @tparam Parent Encoding of the supplied parent.
     * @return 1 after completing every live row, -1 for an invalid contract.
     * All pointers and dimensions have the same meaning as tile(); rows is the
     * total live sibling count. Pending storage must be disjoint from inputs.
     */
    template<Publication Publish, ParentEncoding Parent>
    int run(int rows, int dk, int dv, const float *state,
            const float *q, const float *k, const float *v,
            const float *decay, const float *beta,
            float *output, float *corrections, float *snapshots,
            float pending_decay = 0, const float *pending_key = nullptr,
            const float *pending_u = nullptr, float *resolved_parent = nullptr)
    {
        if (rows < 0 || dk <= 0 || dk > 512 || dv <= 0 || dv > 512 || dv % 8 ||
            !state || !q || !k || !v || !decay || !beta || !output || !corrections)
            return -1;
        if constexpr (Publish == Publication::DenseSnapshots)
            if (!snapshots) return -1;
        if constexpr (Parent == ParentEncoding::Pending)
            if (!pending_key || !pending_u || !resolved_parent || resolved_parent == state) return -1;
        if constexpr (Publish == Publication::DenseSnapshots && Parent == ParentEncoding::Dense)
        {
            // The control completes one small row group before touching the
            // next. A value-first scan across every full snapshot repeatedly
            // evicts partially written cache lines and inflates the apparent
            // benefit of suppressing stores. Independent value columns permit
            // this cache-local schedule without changing any FP32 reduction.
            int first = 0;
            for (; first + 4 <= rows; first += 4)
                for (int vi = 0; vi < dv; vi += 8)
                    tile<4, Publish, ParentEncoding::Dense>(dk, dv, vi, state,
                        q + first * dk, k + first * dk, v + first * dv, decay + first, beta + first,
                        output + first * dv, corrections + first * dv, snapshots + first * dk * dv,
                        0, nullptr, nullptr, nullptr);
            for (; first < rows; ++first)
                for (int vi = 0; vi < dv; vi += 8)
                    tile<1, Publish, ParentEncoding::Dense>(dk, dv, vi, state,
                        q + first * dk, k + first * dk, v + first * dv, decay + first, beta + first,
                        output + first * dv, corrections + first * dv, snapshots + first * dk * dv,
                        0, nullptr, nullptr, nullptr);
            return 1;
        }
        for (int vi = 0; vi < dv; vi += 8)
        {
            int first = 0;
            if constexpr (Parent == ParentEncoding::Pending)
            {
                if (rows == 0) continue;
                // Resolving exactly the first live tile publishes these eight
                // columns before all later tiles consume them. No empty pass
                // can publish an uninitialized parent.
                tile<1, Publish, ParentEncoding::Pending>(dk, dv, vi, state, q, k, v, decay, beta,
                    output, corrections, snapshots, pending_decay, pending_key, pending_u, resolved_parent);
                first = 1;
            }
            const float *parent = Parent == ParentEncoding::Pending ? resolved_parent : state;
            for (; first + 4 <= rows; first += 4)
                tile<4, Publish, ParentEncoding::Dense>(dk, dv, vi, parent,
                    q + first * dk, k + first * dk, v + first * dv, decay + first, beta + first,
                    output + first * dv, corrections + first * dv,
                    Publish == Publication::DenseSnapshots ? snapshots + first * dk * dv : nullptr,
                    0, nullptr, nullptr, nullptr);
            for (; first < rows; ++first)
                tile<1, Publish, ParentEncoding::Dense>(dk, dv, vi, parent,
                    q + first * dk, k + first * dk, v + first * dv, decay + first, beta + first,
                    output + first * dv, corrections + first * dv,
                    Publish == Publication::DenseSnapshots ? snapshots + first * dk * dv : nullptr,
                    0, nullptr, nullptr, nullptr);
        }
        return 1;
    }
}

/** @brief Compute all sibling outputs and exact corrections, writing no dense leaf state.
 * @return 1 on success; -1 on invalid pointers or unsupported reference geometry.
 * @see run for the common row, geometry, operand, and destination contracts.
 */
extern "C" int gdn_leaf_exact_outputs(int rows, int dk, int dv, const float *state,
    const float *q, const float *k, const float *v, const float *decay, const float *beta,
    float *output, float *corrections)
{
    return run<Publication::CompactCorrections, ParentEncoding::Dense>(
        rows, dk, dv, state, q, k, v, decay, beta, output, corrections, nullptr);
}

/** @brief Matched dense-snapshot control with identical projection arithmetic.
 * @return 1 on success; -1 on invalid pointers or unsupported reference geometry.
 * @see run for common arguments; snapshots is [rows,dk,dv].
 */
extern "C" int gdn_leaf_exact_snapshots(int rows, int dk, int dv, const float *state,
    const float *q, const float *k, const float *v, const float *decay, const float *beta,
    float *output, float *corrections, float *snapshots)
{
    return run<Publication::DenseSnapshots, ParentEncoding::Dense>(
        rows, dk, dv, state, q, k, v, decay, beta, output, corrections, snapshots);
}

/** @brief Fuse old-pending materialization with the first new leaf's key pass.
 * @return 1 on success; -1 on invalid pointers, alias, or unsupported geometry.
 * @see run for common and pending arguments; resolved_parent is published only
 *      when at least one live row executes, and remains unchanged for rows=0.
 */
extern "C" int gdn_leaf_exact_after_pending(int rows, int dk, int dv, const float *state,
    const float *q, const float *k, const float *v, const float *decay, const float *beta,
    float *output, float *corrections, float pending_decay, const float *pending_key,
    const float *pending_u, float *resolved_parent)
{
    return run<Publication::CompactCorrections, ParentEncoding::Pending>(
        rows, dk, dv, state, q, k, v, decay, beta, output, corrections, nullptr,
        pending_decay, pending_key, pending_u, resolved_parent);
}

/** @brief Fuse pending resolution and publish the internal children's dense states.
 * @return 1 on success; -1 for an invalid contract.
 * @see gdn_leaf_exact_after_pending for common arguments; snapshots is a
 *      distinct [rows,dk,dv] destination for internal nodes consumed this round.
 */
extern "C" int gdn_leaf_exact_pending_snapshots(int rows, int dk, int dv, const float *state,
    const float *q, const float *k, const float *v, const float *decay, const float *beta,
    float *output, float *corrections, float pending_decay, const float *pending_key,
    const float *pending_u, float *resolved_parent, float *snapshots)
{
    return run<Publication::DenseSnapshots, ParentEncoding::Pending>(
        rows, dk, dv, state, q, k, v, decay, beta, output, corrections, snapshots,
        pending_decay, pending_key, pending_u, resolved_parent);
}

/**
 * @brief Resolve one accepted compact update for an explicit dense consumer.
 * @param dk Key width in [1,512].
 * @param dv Value width, a positive multiple of eight through 512.
 * @param parent Immutable parent in [key,value] layout.
 * @param decay Exact FP32 decay gate used by verification.
 * @param key Retained key, [dk].
 * @param correction Retained byte-exact correction, [dv].
 * @param state Distinct dense destination [dk,dv].
 * @return 1 on success, -1 on invalid dimensions, pointers or alias.
 */
extern "C" int gdn_leaf_exact_materialize(int dk, int dv, const float *parent,
    float decay, const float *key, const float *correction, float *state)
{
    if (dk <= 0 || dk > 512 || dv <= 0 || dv > 512 || dv % 8 ||
        !parent || !key || !correction || !state || state == parent) return -1;
    for (int j = 0; j < dk; ++j)
        for (int vi = 0; vi < dv; vi += 8)
            _mm256_storeu_ps(state + j * dv + vi,
                _mm256_fmadd_ps(_mm256_set1_ps(key[j]), _mm256_loadu_ps(correction + vi),
                    _mm256_mul_ps(_mm256_loadu_ps(parent + j * dv + vi), _mm256_set1_ps(decay))));
    return 1;
}
