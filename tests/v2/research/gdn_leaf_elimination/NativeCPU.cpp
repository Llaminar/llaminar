/**
 * @file NativeCPU.cpp
 * @brief Test-only C ABI for comparing leaf elimination with actual CPU decode.
 *
 * Calls CPUGatedDeltaNet::recurrent_step from the linked production core, and
 * authenticates supplied preprocessing/arithmetic with an independent FMA
 * control before interpreting any difference. Q/K are supplied pre-normalized;
 * the public kernel still owns query scaling and gate conversion. Candidate
 * readouts are research references, never dispatchable production kernels.
 * No GPU, model, sampling, or performance qualification is implied.
 */

#include "kernels/cpu/gdn/CPUGatedDeltaNet.h"

#include <cmath>
#include <cstring>
#include <vector>

/**
 * @brief Expose supplied-core preprocessing for the independent exact trial.
 * @param dk Key width in [1,512].
 * @param q Raw query, [dk].
 * @param gates Raw alpha, beta_raw, negative-exp-A and dt_bias.
 * @param scaled_q Destination [dk].
 * @param decay_beta Destination [2], containing decay then write strength.
 * @return 1 on success or -1 for invalid input. The companion native compare
 *         authenticates this arithmetic against the actual production kernel.
 */
extern "C" int gdn_leaf_cpu_prepare(int dk, const float *q, const float *gates,
                                   float *scaled_q, float *decay_beta)
{
    if (dk <= 0 || dk > 512 || !q || !gates || !scaled_q || !decay_beta) return -1;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dk));
    const float x = gates[0] + gates[3];
    const float softplus = x > 20.0f ? x : std::log1p(std::exp(x));
    decay_beta[0] = std::exp(gates[2] * softplus);
    decay_beta[1] = 1.0f / (1.0f + std::exp(-gates[1]));
    for (int j = 0; j < dk; ++j) scaled_q[j] = q[j] * scale;
    return 1;
}

/**
 * @brief Compare one real CPU decode step with two leaf-only readouts.
 * @param dk Key width in [1,512].
 * @param dv Value width, a positive multiple of 16 no larger than 512.
 * @param q Input query before the production 1/sqrt(dk) scaling.
 * @param k Supplied key, with no additional L2 normalization requested.
 * @param v Input value vector.
 * @param gates Four raw gates: alpha, beta_raw, negative-exp-A, dt_bias.
 * @param state In/out production recurrence state in [key,value] layout.
 * @param output Production readout of the updated state.
 * @param fused Candidate FP32 readout with explicit FMA reductions.
 * @param wide Candidate FP64 readout rounded to FP32 at publication.
 * @return 1 for an authenticated comparison, -1 for invalid input, -2 for
 *         kernel failure, -3 for mismatched preprocessing/FMA control, and
 *         -4 if a C++ exception occurred. Exceptions never cross the C ABI.
 */
extern "C" int gdn_leaf_cpu_compare(
    int dk, int dv, const float *q, const float *k, const float *v,
    const float *gates, float *state, float *output, float *fused, float *wide)
{
    if (dk <= 0 || dk > 512 || dv <= 0 || dv > 512 || dv % 16 ||
        !q || !k || !v || !gates || !state || !output || !fused || !wide)
        return -1;
    try
    {
        const std::vector<float> initial(state, state + dk * dv);
        const float scale = 1.0f / std::sqrt(static_cast<float>(dk));
        const float x = gates[0] + gates[3];
        const float softplus = x > 20.0f ? x : std::log1p(std::exp(x));
        const float decay = std::exp(gates[2] * softplus);
        const float beta = 1.0f / (1.0f + std::exp(-gates[1]));
        std::vector<float> scaled_q(dk);
        for (int j = 0; j < dk; ++j)
            scaled_q[j] = q[j] * scale;

        llaminar2::CPUGatedDeltaNet kernel;
        if (!kernel.recurrent_step(q, k, v, gates, gates + 1, gates + 2,
                                   gates + 3, output, state, 1, dk, dv, false))
            return -2;

        // Authenticate the exact native arithmetic first. The value widths
        // deliberately have no scalar tail on either AVX2 or AVX512.
        std::vector<float> control = initial;
        std::vector<float> control_output(dv, 0.0f);
        for (int vi = 0; vi < dv; ++vi)
        {
            float projection = 0.0f;
            for (int j = 0; j < dk; ++j)
            {
                control[j * dv + vi] *= decay;
                projection = std::fma(control[j * dv + vi], k[j], projection);
            }
            const float correction = (v[vi] - projection) * beta;
            for (int j = 0; j < dk; ++j)
            {
                control[j * dv + vi] = std::fma(k[j], correction, control[j * dv + vi]);
                control_output[vi] = std::fma(control[j * dv + vi], scaled_q[j], control_output[vi]);
            }
        }
        if (std::memcmp(control.data(), state, control.size() * sizeof(float)) ||
            std::memcmp(control_output.data(), output, dv * sizeof(float)))
            return -3;

        float dot = 0.0f;
        double dot_wide = 0.0;
        for (int j = 0; j < dk; ++j)
        {
            dot = std::fma(k[j], scaled_q[j], dot);
            dot_wide += static_cast<double>(k[j]) * scaled_q[j];
        }
        const float eta = beta * dot;
        const double eta_wide = static_cast<double>(beta) * dot_wide;
        for (int vi = 0; vi < dv; ++vi)
        {
            float value = 0.0f;
            double value_wide = 0.0;
            for (int j = 0; j < dk; ++j)
            {
                const float adjusted = decay * std::fma(-eta, k[j], scaled_q[j]);
                value = std::fma(initial[j * dv + vi], adjusted, value);
                const double adjusted_wide = static_cast<double>(decay) *
                    (static_cast<double>(scaled_q[j]) - eta_wide * k[j]);
                value_wide += static_cast<double>(initial[j * dv + vi]) * adjusted_wide;
            }
            fused[vi] = std::fma(eta, v[vi], value);
            wide[vi] = static_cast<float>(value_wide + eta_wide * v[vi]);
        }
        return 1;
    }
    catch (...)
    {
        return -4;
    }
}
