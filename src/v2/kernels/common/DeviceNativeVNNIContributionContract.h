/**
 * @file DeviceNativeVNNIContributionContract.h
 * @brief One FP32 contribution program for every movable expert codebook.
 *
 * NativeVNNI payload decoding is codebook-specific, while its persisted scale
 * application is not backend-specific. ExpertOverlay can move one prepared
 * expert between CUDA and ROCm, so the exact multiplication, correction, and
 * addition tree must travel with the expert. This contract consumes exact
 * integer decoder terms and defines the shared binary32 publication program.
 */

#pragma once

#include "kernels/common/DeviceFP32NumericalContract.h"

#include <cstdint>

namespace llaminar2::device_native_vnni_contract
{
    /**
     * @brief Publish one single-scale 32-value block contribution.
     * @param dot Exact signed INT8 dot product.
     * @param weight_scale FP16-derived weight scale.
     * @param activation_scale FP32 activation-block scale.
     * @return Canonically rounded contribution, not yet accumulated.
     */
    __device__ __forceinline__ float singleScaleBlock(
        std::int32_t dot,
        float weight_scale,
        float activation_scale) noexcept
    {
        using namespace device_fp32_contract;
        return multiply(
            multiply(activation_scale, weight_scale),
            static_cast<float>(dot));
    }

    /**
     * @brief Publish one single-scale asymmetric-minimum correction.
     * @param activation_sum Exact sum of the 32 signed activation bytes.
     * @param weight_min FP16-derived additive weight minimum.
     * @param activation_scale FP32 activation-block scale.
     * @return Canonically rounded correction, not yet accumulated.
     */
    __device__ __forceinline__ float singleScaleCorrection(
        std::int32_t activation_sum,
        float weight_min,
        float activation_scale) noexcept
    {
        using namespace device_fp32_contract;
        return multiply(
            multiply(activation_scale, weight_min),
            static_cast<float>(activation_sum));
    }

    /**
     * @brief Publish one dual-scale 32-value block contribution.
     * @param low_dot Exact dot product for values 0 through 15.
     * @param high_dot Exact dot product for values 16 through 31.
     * @param low_weight_scale FP16-derived low-half scale.
     * @param high_weight_scale FP16-derived high-half scale.
     * @param activation_scale FP32 activation-block scale.
     * @return Canonically rounded contribution, not yet accumulated.
     */
    __device__ __forceinline__ float dualScaleBlock(
        std::int32_t low_dot,
        std::int32_t high_dot,
        float low_weight_scale,
        float high_weight_scale,
        float activation_scale) noexcept
    {
        using namespace device_fp32_contract;
        const float paired = add(
            multiply(low_weight_scale, static_cast<float>(low_dot)),
            multiply(high_weight_scale, static_cast<float>(high_dot)));
        return multiply(activation_scale, paired);
    }

    /**
     * @brief Publish a dual-half asymmetric-minimum correction.
     * @param low_activation_sum Sum of activation values 0 through 15.
     * @param high_activation_sum Sum of activation values 16 through 31.
     * @param low_weight_min FP16-derived low-half minimum.
     * @param high_weight_min FP16-derived high-half minimum.
     * @param activation_scale FP32 activation-block scale.
     * @return Canonically rounded correction, not yet accumulated.
     */
    __device__ __forceinline__ float dualScaleCorrection(
        std::int32_t low_activation_sum,
        std::int32_t high_activation_sum,
        float low_weight_min,
        float high_weight_min,
        float activation_scale) noexcept
    {
        using namespace device_fp32_contract;
        const float paired = add(
            multiply(
                low_weight_min,
                static_cast<float>(low_activation_sum)),
            multiply(
                high_weight_min,
                static_cast<float>(high_activation_sum)));
        return multiply(activation_scale, paired);
    }

    /**
     * @brief Publish the IQ1_M signed one-eighth delta correction.
     * @param sum0 Activation sum for values 0 through 7.
     * @param sum1 Activation sum for values 8 through 15.
     * @param sum2 Activation sum for values 16 through 23.
     * @param sum3 Activation sum for values 24 through 31.
     * @param delta0 Signed delta for group zero.
     * @param delta1 Signed delta for group one.
     * @param delta2 Signed delta for group two.
     * @param delta3 Signed delta for group three.
     * @param low_weight_scale Weight scale shared by groups zero and one.
     * @param high_weight_scale Weight scale shared by groups two and three.
     * @param activation_scale FP32 activation-block scale.
     * @return Canonically rounded correction, not yet accumulated.
     */
    __device__ __forceinline__ float iq1MDeltaCorrection(
        std::int32_t sum0,
        std::int32_t sum1,
        std::int32_t sum2,
        std::int32_t sum3,
        float delta0,
        float delta1,
        float delta2,
        float delta3,
        float low_weight_scale,
        float high_weight_scale,
        float activation_scale) noexcept
    {
        using namespace device_fp32_contract;
        const float low_delta = multiply(
            add(
                multiply(delta0, static_cast<float>(sum0)),
                multiply(delta1, static_cast<float>(sum1))),
            low_weight_scale);
        const float high_delta = multiply(
            add(
                multiply(delta2, static_cast<float>(sum2)),
                multiply(delta3, static_cast<float>(sum3))),
            high_weight_scale);
        return multiply(activation_scale, add(low_delta, high_delta));
    }

    /**
     * @brief Add one independently rounded contribution in traversal order.
     * @param accumulator Running row or route accumulator.
     * @param contribution Previously rounded block, correction, or route term.
     * @return Canonically rounded updated accumulator.
     */
    __device__ __forceinline__ float accumulate(
        float accumulator,
        float contribution) noexcept
    {
        return device_fp32_contract::add(accumulator, contribution);
    }

    /**
     * @brief Publish one independently rounded router-weighted expert value.
     * @param route_weight Router probability for the original top-k slot.
     * @param expert_value Complete expert down-projection value.
     * @return Canonically rounded weighted route contribution.
     */
    __device__ __forceinline__ float weightRoute(
        float route_weight,
        float expert_value) noexcept
    {
        return device_fp32_contract::multiply(route_weight, expert_value);
    }
} // namespace llaminar2::device_native_vnni_contract
