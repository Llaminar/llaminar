/**
 * @file ROCmNativeVNNIDecodeEquivalentMath.h
 * @brief Canonical FP32 update operations shared by ROCm NativeVNNI kernels.
 *
 * NativeVNNI stores one INT8 dot-product payload with one or two FP16 weight
 * scales per 32-value K block.  A grouped verifier or prefill kernel may use a
 * different output tile from serial M=1 decode, but it must not change the
 * floating-point expression tree that publishes the row.  Even algebraically
 * equivalent rewrites can round differently and make speculative verification
 * depend on the grouped row count.
 *
 * Keep the operations in this file small, explicit, and shared by every ROCm
 * NativeVNNI execution family.  Callers own the integer dot products and K
 * traversal; these helpers own the exact FP32 parenthesization at the point
 * where one K block is committed to the running accumulator.
 */

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

/**
 * @brief Commit one single-scale NativeVNNI K block in serial-decode order.
 *
 * Serial decode first applies the weight scale to the exact INT32 dot product,
 * then applies the activation scale while adding the block to the running FP32
 * accumulator. Keeping this order explicit avoids the tile-dependent rewrite
 * `(dot * activation_scale) * weight_scale`.
 *
 * @param accumulator Running FP32 dot-product accumulator.
 * @param dot INT32 dot product for the complete 32-value K block.
 * @param weight_scale FP16-derived weight scale for the K block.
 * @param activation_scale FP32 scale for the quantized activation K block.
 * @return The updated FP32 accumulator.
 */
__device__ __forceinline__ float rocm_native_vnni_commit_single_scale_block(
    float accumulator,
    int32_t dot,
    float weight_scale,
    float activation_scale)
{
    const float weight_scaled_dot =
        static_cast<float>(dot) * weight_scale;
    return __fmaf_rn(
        weight_scaled_dot,
        activation_scale,
        accumulator);
}

/**
 * @brief Commit one asymmetric minimum correction in serial-decode order.
 *
 * @param accumulator Running FP32 dot-product accumulator.
 * @param activation_sum Exact INT32 sum of the quantized activation block.
 * @param weight_min FP16-derived asymmetric minimum for the weight block.
 * @param activation_scale FP32 scale for the quantized activation K block.
 * @return The updated FP32 accumulator.
 */
__device__ __forceinline__ float rocm_native_vnni_commit_single_scale_correction(
    float accumulator,
    int32_t activation_sum,
    float weight_min,
    float activation_scale)
{
    const float weight_scaled_sum =
        static_cast<float>(activation_sum) * weight_min;
    return __fmaf_rn(
        weight_scaled_sum,
        activation_scale,
        accumulator);
}

/**
 * @brief Commit one dual-scale NativeVNNI K block in serial-decode order.
 *
 * The low and high 16-value half-blocks are accumulated exactly in INT32 by the
 * caller.  This helper then:
 *
 * 1. rounds the low scaled contribution to FP32,
 * 2. fuses the high scaled contribution into that low contribution,
 * 3. fuses the activation-scaled block contribution into the running result.
 *
 * Using explicit round-to-nearest FMAs prevents a surrounding tile kernel from
 * reassociating these operations according to its register layout.  Serial
 * decode, grouped verification, dense prefill, and grouped MoE prefill must all
 * call this helper rather than reproducing the formula locally.
 *
 * @param accumulator Running FP32 dot-product accumulator.
 * @param low_dot INT32 dot product for elements 0..15 of the K block.
 * @param high_dot INT32 dot product for elements 16..31 of the K block.
 * @param low_weight_scale FP16-derived weight scale for elements 0..15.
 * @param high_weight_scale FP16-derived weight scale for elements 16..31.
 * @param activation_scale FP32 scale for the quantized activation K block.
 * @return The updated FP32 accumulator.
 */
__device__ __forceinline__ float rocm_native_vnni_commit_dual_scale_block(
    float accumulator,
    int32_t low_dot,
    int32_t high_dot,
    float low_weight_scale,
    float high_weight_scale,
    float activation_scale)
{
    const float low_contribution =
        static_cast<float>(low_dot) * low_weight_scale;
    const float paired_contribution = __fmaf_rn(
        static_cast<float>(high_dot),
        high_weight_scale,
        low_contribution);
    return __fmaf_rn(
        paired_contribution,
        activation_scale,
        accumulator);
}

/**
 * @brief Commit a two-half asymmetric correction in serial-decode order.
 *
 * Q2_K stores a separate minimum for each 16-value half-block. The correction
 * must combine those two weighted activation sums before applying the shared
 * activation scale, matching serial M=1 decode.
 *
 * @param accumulator Running FP32 dot-product accumulator.
 * @param low_activation_sum INT32 activation sum for elements 0..15.
 * @param high_activation_sum INT32 activation sum for elements 16..31.
 * @param low_weight_min FP16-derived minimum for elements 0..15.
 * @param high_weight_min FP16-derived minimum for elements 16..31.
 * @param activation_scale FP32 scale for the quantized activation K block.
 * @return The updated FP32 accumulator.
 */
__device__ __forceinline__ float rocm_native_vnni_commit_dual_scale_correction(
    float accumulator,
    int32_t low_activation_sum,
    int32_t high_activation_sum,
    float low_weight_min,
    float high_weight_min,
    float activation_scale)
{
    const float low_correction =
        static_cast<float>(low_activation_sum) * low_weight_min;
    const float paired_correction = __fmaf_rn(
        static_cast<float>(high_activation_sum),
        high_weight_min,
        low_correction);
    return __fmaf_rn(
        paired_correction,
        activation_scale,
        accumulator);
}

/**
 * @brief Commit the IQ1_M signed one-eighth delta correction.
 *
 * IQ1_M uses four independently signed eight-value grid groups. The group
 * activation sums and signs are combined in the same nested order used by
 * serial decode, then scaled by the two half-block scales and the activation
 * scale.
 *
 * @param accumulator Running FP32 dot-product accumulator.
 * @param sum0 Activation sum for values 0..7.
 * @param sum1 Activation sum for values 8..15.
 * @param sum2 Activation sum for values 16..23.
 * @param sum3 Activation sum for values 24..31.
 * @param delta0 Signed delta for group 0.
 * @param delta1 Signed delta for group 1.
 * @param delta2 Signed delta for group 2.
 * @param delta3 Signed delta for group 3.
 * @param low_weight_scale Weight scale for groups 0 and 1.
 * @param high_weight_scale Weight scale for groups 2 and 3.
 * @param activation_scale FP32 scale for the quantized activation K block.
 * @return The updated FP32 accumulator.
 */
__device__ __forceinline__ float rocm_native_vnni_commit_iq1_m_delta_correction(
    float accumulator,
    int32_t sum0,
    int32_t sum1,
    int32_t sum2,
    int32_t sum3,
    float delta0,
    float delta1,
    float delta2,
    float delta3,
    float low_weight_scale,
    float high_weight_scale,
    float activation_scale)
{
    const float low_group = __fmaf_rn(
        delta1,
        static_cast<float>(sum1),
        delta0 * static_cast<float>(sum0));
    const float high_group = __fmaf_rn(
        delta3,
        static_cast<float>(sum3),
        delta2 * static_cast<float>(sum2));
    const float low_scaled = low_group * low_weight_scale;
    const float paired_scaled = __fmaf_rn(
        high_group,
        high_weight_scale,
        low_scaled);
    return __fmaf_rn(
        paired_scaled,
        activation_scale,
        accumulator);
}

/**
 * @brief Apply a router weight with one explicit round-to-nearest operation.
 *
 * Collective publication stores each weighted route row before reducing it,
 * while a single-device kernel may reduce the row immediately.  Expressing the
 * multiplication as an FMA with an exact zero addend gives both forms the same
 * FP32 rounding boundary and prevents the immediate form from contracting the
 * multiplication into its following accumulation.
 *
 * @param route_weight Router probability for one original top-k slot.
 * @param expert_value Complete or split-K-reduced expert down value.
 * @return One independently rounded weighted route contribution.
 */
__device__ __forceinline__ float rocm_native_vnni_weight_route_rn(
    float route_weight,
    float expert_value)
{
    return __fmaf_rn(route_weight, expert_value, 0.0f);
}

/**
 * @brief Add one rounded route or K-part contribution without FMA contraction.
 *
 * @param accumulator Running FP32 result in canonical traversal order.
 * @param contribution Previously rounded contribution to append.
 * @return The round-to-nearest FP32 sum.
 */
__device__ __forceinline__ float rocm_native_vnni_accumulate_rn(
    float accumulator,
    float contribution)
{
    return __fadd_rn(accumulator, contribution);
}
