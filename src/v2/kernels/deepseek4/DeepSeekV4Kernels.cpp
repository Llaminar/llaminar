/**
 * @file DeepSeekV4Kernels.cpp
 * @brief Host-side policy construction shared by every DeepSeek-V4 backend.
 *
 * The YaRN ramp bounds are integers in DeepSeek's reference (floor/ceil of the
 * correction dimensions); they are resolved once here so CPU and GPU kernels
 * evaluate identical per-pair frequencies from the same immutable policy.
 */
#include "DeepSeekV4Kernels.h"

#include <cmath>
#include <stdexcept>

namespace llaminar2::deepseek4
{
RopePolicy RopePolicy::make(int ropeDim, float base, int originalContext, float factor,
                            float betaFast, float betaSlow)
{
    if (ropeDim <= 0 || ropeDim % 2 != 0 || !(base > 1.0f))
        throw std::invalid_argument("DeepSeekV4 RoPE requires a positive even dimension and base > 1");
    RopePolicy policy;
    policy.ropeDim = ropeDim;
    policy.base = base;
    if (originalContext <= 0)
        return policy;
    if (!(factor > 0.0f) || !(betaFast > 0.0f) || !(betaSlow > 0.0f))
        throw std::invalid_argument("DeepSeekV4 YaRN requires positive factor and correction rotations");
    // find_correction_dim(rot) = d * ln(L / (rot * 2pi)) / (2 ln b)
    auto correction = [&](double rotations) {
        return ropeDim * std::log(originalContext / (rotations * 2.0 * M_PI)) / (2.0 * std::log(base));
    };
    double low = std::floor(correction(betaFast));
    double high = std::ceil(correction(betaSlow));
    low = std::max(low, 0.0);
    high = std::min(high, static_cast<double>(ropeDim - 1));
    if (low == high)
        high += 0.001;
    policy.yarn = true;
    policy.factor = factor;
    policy.rampLow = static_cast<float>(low);
    policy.rampHigh = static_cast<float>(high);
    return policy;
}
} // namespace llaminar2::deepseek4
