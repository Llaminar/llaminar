/**
 * @file SharedReadoutCPU.cpp
 * @brief Projection-eliding sibling evaluator for the new canonical CPU readout.
 *
 * Four sibling rows reuse each parent load. Leaf instantiations perform only
 * the adjusted-query action; no key projection, correction or dense child is
 * computed. Internal nodes additionally advance ordinary dense state. Accepted
 * leaves retain raw update operands and resolve only when a dense consumer
 * needs their state. Every buffer is caller-owned and allocated before timing.
 * The separate production M=1 bridge checks an independent FP32 interpreter.
 */
#include "kernels/cpu/gdn/CPUAdjustedGDNReadout.h"
#include "kernels/cpu/gdn/CPUGatedDeltaNet.h"
#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <vector>

namespace
{
    /** @brief Dense states are required only for nodes with immediate consumers. */
    enum class Publication { Leaf, Internal };

    /**
     * @brief Evaluate one physical tile with independent ordered row reductions.
     * @tparam Rows Live siblings, either one or four.
     * @tparam Publish Whether each child has an immediate dense-state consumer.
     * @param dk Key width; reductions visit [0,dk) in order.
     * @param dv Value width and key-major parent stride.
     * @param vi First of eight independent value columns.
     * @param state Immutable dense parent, [dk,dv].
     * @param adjusted Prepared adjusted queries, [Rows,dk].
     * @param eta Prepared readout coefficients, [Rows].
     * @param k Keys, [Rows,dk].
     * @param v Values, [Rows,dv].
     * @param decay Decay gates, [Rows].
     * @param beta Write gates, [Rows].
     * @param output Readouts, [Rows,dv].
     * @param snapshots Internal-node destinations [Rows,dk,dv], otherwise null.
     */
    template<int Rows, Publication Publish>
    void tile(int dk, int dv, int vi, const float *state,
        const float *adjusted, const float *eta, const float *k, const float *v,
        const float *decay, const float *beta, float *output, float *snapshots)
    {
        __m256 out[Rows], projection[Rows];
        for (int row = 0; row < Rows; ++row)
        {
            out[row] = _mm256_setzero_ps();
            if constexpr (Publish == Publication::Internal)
                projection[row] = _mm256_setzero_ps();
        }
        for (int j = 0; j < dk; ++j)
        {
            const __m256 parent = _mm256_loadu_ps(state + j*dv + vi);
            for (int row = 0; row < Rows; ++row)
            {
                out[row] = _mm256_fmadd_ps(parent, _mm256_set1_ps(adjusted[row*dk + j]), out[row]);
                if constexpr (Publish == Publication::Internal)
                    projection[row] = _mm256_fmadd_ps(
                        _mm256_mul_ps(parent, _mm256_set1_ps(decay[row])),
                        _mm256_set1_ps(k[row*dk + j]), projection[row]);
            }
        }
        for (int row = 0; row < Rows; ++row)
        {
            const __m256 value = _mm256_loadu_ps(v + row*dv + vi);
            _mm256_storeu_ps(output + row*dv + vi,
                _mm256_fmadd_ps(_mm256_set1_ps(eta[row]), value, out[row]));
            if constexpr (Publish == Publication::Internal)
                projection[row] = _mm256_mul_ps(_mm256_sub_ps(value, projection[row]), _mm256_set1_ps(beta[row]));
        }
        if constexpr (Publish == Publication::Internal)
            for (int j = 0; j < dk; ++j)
            {
                const __m256 parent = _mm256_loadu_ps(state + j*dv + vi);
                for (int row = 0; row < Rows; ++row)
                    _mm256_storeu_ps(snapshots + (row*dk + j)*dv + vi,
                        _mm256_fmadd_ps(_mm256_set1_ps(k[row*dk + j]), projection[row],
                            _mm256_mul_ps(parent, _mm256_set1_ps(decay[row]))));
            }
    }

    /**
     * @brief Prepare live queries, then evaluate fixed physical sibling tiles.
     * @tparam Publish Explicit leaf or internal-node state publication.
     * @return 1 on success, -1 for invalid pointers or unsupported geometry.
     * @see gdn_shared_group for all common argument contracts.
     */
    template<Publication Publish>
    int group(int rows, int dk, int dv, const float *state, const float *q,
        const float *k, const float *v, const float *decay, const float *beta,
        float *adjusted, float *eta, float *output, float *snapshots)
    {
        if (rows < 0 || rows > 4096 || dk <= 0 || dk > 512 || dv <= 0 || dv > 512 || dv%8 ||
            !state || !q || !k || !v || !decay || !beta || !adjusted || !eta || !output)
            return -1;
        if constexpr (Publish == Publication::Internal)
            if (!snapshots) return -1;
        for (int row = 0; row < rows; ++row)
            eta[row] = llaminar2::prepareAdjustedGDNReadout(q + row*dk, k + row*dk,
                decay[row], beta[row], dk, adjusted + row*dk);
        // Finish each small group before advancing, retaining dense-control
        // cache locality as well as the leaf kernel's parent-load reuse.
        int first = 0;
        for (; first + 4 <= rows; first += 4)
            for (int vi = 0; vi < dv; vi += 8)
                tile<4, Publish>(dk, dv, vi, state, adjusted + first*dk, eta + first,
                    k + first*dk, v + first*dv, decay + first, beta + first, output + first*dv,
                    Publish == Publication::Internal ? snapshots + first*dk*dv : nullptr);
        for (; first < rows; ++first)
            for (int vi = 0; vi < dv; vi += 8)
                tile<1, Publish>(dk, dv, vi, state, adjusted + first*dk, eta + first,
                    k + first*dk, v + first*dv, decay + first, beta + first, output + first*dv,
                    Publish == Publication::Internal ? snapshots + first*dk*dv : nullptr);
        return 1;
    }
}

/**
 * @brief Evaluate actual live siblings under the new shared readout definition.
 * @param internal 0 for leaves, 1 for nodes whose states have immediate consumers.
 * @param rows Live sibling count in [0,4096].
 * @param dk Key dimension in [1,512].
 * @param dv Value dimension, a multiple of eight in [8,512].
 * @param state Immutable parent [dk,dv].
 * @param q Already normalized/scaled queries [rows,dk].
 * @param k Normalized keys [rows,dk].
 * @param v Values [rows,dv].
 * @param decay Rounded decay gates [rows].
 * @param beta Rounded write gates [rows].
 * @param adjusted Persistent prepared-query workspace [rows,dk].
 * @param eta Persistent coefficient workspace [rows].
 * @param output Readout destination [rows,dv].
 * @param snapshots Dense internal-state destination [rows,dk,dv], untouched for leaves.
 * @return 1 on success, -1 for an invalid contract. Empty calls write nothing.
 */
extern "C" int gdn_shared_group(int internal, int rows, int dk, int dv,
    const float *state, const float *q, const float *k, const float *v,
    const float *decay, const float *beta, float *adjusted, float *eta,
    float *output, float *snapshots)
{
    if (internal == 0)
        return group<Publication::Leaf>(rows, dk, dv, state, q, k, v, decay, beta, adjusted, eta, output, snapshots);
    if (internal == 1)
        return group<Publication::Internal>(rows, dk, dv, state, q, k, v, decay, beta, adjusted, eta, output, snapshots);
    return -1;
}

/**
 * @brief Resolve only an accepted raw leaf update without repeating its readout.
 * @param dk Key dimension in [1,512].
 * @param dv Value dimension, a multiple of eight in [8,512].
 * @param parent Immutable accepted leaf's parent [dk,dv].
 * @param key Retained raw normalized key [dk].
 * @param value Retained raw value [dv].
 * @param decay Retained rounded decay gate.
 * @param beta Retained rounded write gate.
 * @param state Distinct dense destination [dk,dv].
 * @return 1 on success, -1 for invalid pointers, geometry or alias.
 */
extern "C" int gdn_shared_resolve(int dk, int dv, const float *parent,
    const float *key, const float *value, float decay, float beta, float *state)
{
    if (dk <= 0 || dk > 512 || dv <= 0 || dv > 512 || dv%8 ||
        !parent || !key || !value || !state || parent == state) return -1;
    for (int vi = 0; vi < dv; vi += 8)
    {
        __m256 projection = _mm256_setzero_ps();
        for (int j = 0; j < dk; ++j)
            projection = _mm256_fmadd_ps(
                _mm256_mul_ps(_mm256_loadu_ps(parent + j*dv + vi), _mm256_set1_ps(decay)),
                _mm256_set1_ps(key[j]), projection);
        const __m256 correction = _mm256_mul_ps(
            _mm256_sub_ps(_mm256_loadu_ps(value + vi), projection), _mm256_set1_ps(beta));
        for (int j = 0; j < dk; ++j)
            _mm256_storeu_ps(state + j*dv + vi,
                _mm256_fmadd_ps(_mm256_set1_ps(key[j]), correction,
                    _mm256_mul_ps(_mm256_loadu_ps(parent + j*dv + vi), _mm256_set1_ps(decay))));
    }
    return 1;
}

/**
 * @brief Run the actual new production CPU M=1 implementation with raw gates.
 * @param dk Key dimension in [1,512].
 * @param dv Value dimension in [1,512].
 * @param q Raw query before production head scaling [dk].
 * @param k Supplied normalized key [dk].
 * @param v Raw value [dv].
 * @param gates Raw alpha, beta, negative-exp-A and dt-bias [4].
 * @param state In/out production state [dk,dv].
 * @param output New production readout [dv].
 * @param legacy_output Old readout interpreted from the updated state [dv].
 * @return 1 when output and state authenticate against an independent interpreter;
 *         -1 for invalid arguments, -2 for native failure, -3 for control drift,
 *         and -4 for a caught C++ exception. This is outside timing intervals.
 */
extern "C" int gdn_shared_serial(int dk, int dv, const float *q, const float *k,
    const float *v, const float *gates, float *state, float *output, float *legacy_output)
{
    if (dk <= 0 || dk > 512 || dv <= 0 || dv > 512 || !q || !k || !v || !gates || !state || !output || !legacy_output)
        return -1;
    try
    {
        std::vector<float> control(state, state + dk*dv), scaled(dk), adjusted(dk), expected(dv);
        const float scale = 1.0f/std::sqrt(static_cast<float>(dk));
        const float x = gates[0] + gates[3];
        const float decay = std::exp(gates[2]*(x > 20 ? x : std::log1p(std::exp(x))));
        const float beta = 1.0f/(1.0f + std::exp(-gates[1]));
        float dot = 0;
        for (int j = 0; j < dk; ++j)
        {
            scaled[j] = q[j]*scale;
            dot = std::fma(k[j], scaled[j], dot);
        }
        const float eta = beta*dot;
        for (int j = 0; j < dk; ++j) adjusted[j] = decay*std::fma(-eta, k[j], scaled[j]);
        for (int vi = 0; vi < dv; ++vi)
        {
            float out = 0, projection = 0, legacy = 0;
            for (int j = 0; j < dk; ++j)
            {
                out = std::fma(control[j*dv + vi], adjusted[j], out);
                control[j*dv + vi] *= decay;
                projection = std::fma(control[j*dv + vi], k[j], projection);
            }
            expected[vi] = std::fma(eta, v[vi], out);
            const float delta = (v[vi] - projection)*beta;
            for (int j = 0; j < dk; ++j)
            {
                control[j*dv + vi] = std::fma(k[j], delta, control[j*dv + vi]);
                legacy = std::fma(control[j*dv + vi], scaled[j], legacy);
            }
            legacy_output[vi] = legacy;
        }
        llaminar2::CPUGatedDeltaNet kernel;
        if (!kernel.recurrent_step(q, k, v, gates, gates+1, gates+2, gates+3, output, state, 1, dk, dv, false)) return -2;
        if (std::memcmp(state, control.data(), control.size()*sizeof(float)) ||
            std::memcmp(output, expected.data(), dv*sizeof(float))) return -3;
        return 1;
    }
    catch (...) { return -4; }
}

/**
 * @brief Expose core input preparation, authenticated by gdn_shared_serial.
 * @param dk Key width in [1,512].
 * @param q Raw query [dk].
 * @param gates Raw alpha, beta, negative-exp-A and dt-bias [4].
 * @param scaled Destination for the head-scaled query [dk].
 * @param decay_beta Destination for rounded decay and write gates [2].
 * @return 1 on success, -1 for an invalid contract.
 */
extern "C" int gdn_shared_prepare(int dk, const float *q, const float *gates,
    float *scaled, float *decay_beta)
{
    if (dk <= 0 || dk > 512 || !q || !gates || !scaled || !decay_beta) return -1;
    const float scale = 1.0f/std::sqrt(static_cast<float>(dk));
    const float x = gates[0] + gates[3];
    decay_beta[0] = std::exp(gates[2]*(x > 20 ? x : std::log1p(std::exp(x))));
    decay_beta[1] = 1.0f/(1.0f + std::exp(-gates[1]));
    for (int j = 0; j < dk; ++j) scaled[j] = q[j]*scale;
    return 1;
}
