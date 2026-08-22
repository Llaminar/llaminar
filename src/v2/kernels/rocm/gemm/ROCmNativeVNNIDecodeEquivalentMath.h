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
 * This file is the ROCm-facing adapter used by existing kernels. It delegates
 * every contribution to the backend-neutral NativeVNNI contract; callers own
 * integer dot products and K traversal, while the shared contract owns exact
 * FP32 parenthesization.
 */

#pragma once

#include <hip/hip_runtime.h>

#include "kernels/common/DeviceNativeVNNIContributionContract.h"

#include <cstdint>

/**
 * @brief Commit one single-scale NativeVNNI K block in serial-decode order.
 *
 * Movable-expert projection first rounds the activation/weight scale product,
 * then its multiplication by the exact INT32 dot product, and finally adds the
 * persisted contribution to the running accumulator. The shared contract owns
 * the HIP dependency barriers required to retain those rounding edges.
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
    return llaminar2::device_native_vnni_contract::accumulate(
        accumulator,
        llaminar2::device_native_vnni_contract::singleScaleBlock(
            dot,
            weight_scale,
            activation_scale));
}

/**
 * @brief Commit one corrected single-scale block as one publication unit.
 *
 * @param accumulator Running FP32 dot-product accumulator.
 * @param dot INT32 dot product for the complete 32-value K block.
 * @param weight_scale FP16-derived scale for the weight block.
 * @param activation_sum Exact INT32 sum of the quantized activation block.
 * @param weight_min FP16-derived asymmetric minimum for the weight block.
 * @param activation_scale FP32 scale for the quantized activation K block.
 * @return The updated FP32 accumulator.
 */
__device__ __forceinline__ float
rocm_native_vnni_commit_corrected_single_scale_block(
    float accumulator,
    int32_t dot,
    float weight_scale,
    int32_t activation_sum,
    float weight_min,
    float activation_scale)
{
    float contribution =
        llaminar2::device_native_vnni_contract::singleScaleBlock(
            dot,
            weight_scale,
            activation_scale);
    contribution = llaminar2::device_native_vnni_contract::accumulate(
        contribution,
        llaminar2::device_native_vnni_contract::singleScaleCorrection(
            activation_sum,
            weight_min,
            activation_scale));
    return llaminar2::device_native_vnni_contract::accumulate(
        accumulator,
        contribution);
}

/**
 * @brief Commit one dual-scale NativeVNNI K block in serial-decode order.
 *
 * The low and high 16-value half-blocks are accumulated exactly in INT32 by the
 * caller.  This helper then:
 *
 * 1. rounds each half-block scale multiplication independently,
 * 2. adds those persisted values,
 * 3. multiplies by the activation scale, and
 * 4. adds the persisted block contribution to the running result.
 *
 * Using explicit retained round-to-nearest operations prevents a surrounding
 * tile kernel from reassociating according to its register layout. Serial
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
    return llaminar2::device_native_vnni_contract::accumulate(
        accumulator,
        llaminar2::device_native_vnni_contract::dualScaleBlock(
            low_dot,
            high_dot,
            low_weight_scale,
            high_weight_scale,
            activation_scale));
}

/**
 * @brief Commit one corrected dual-scale block as one publication unit.
 *
 * Q2_K stores a separate minimum for each 16-value half-block. The correction
 * is first combined with the dual-scale dot contribution, then that completed
 * block is appended once to the row accumulator.
 *
 * @param accumulator Running FP32 dot-product accumulator.
 * @param low_dot INT32 dot product for elements 0..15 of the K block.
 * @param high_dot INT32 dot product for elements 16..31 of the K block.
 * @param low_weight_scale FP16-derived scale for elements 0..15.
 * @param high_weight_scale FP16-derived scale for elements 16..31.
 * @param low_activation_sum INT32 activation sum for elements 0..15.
 * @param high_activation_sum INT32 activation sum for elements 16..31.
 * @param low_weight_min FP16-derived minimum for elements 0..15.
 * @param high_weight_min FP16-derived minimum for elements 16..31.
 * @param activation_scale FP32 scale for the quantized activation K block.
 * @return The updated FP32 accumulator.
 */
__device__ __forceinline__ float
rocm_native_vnni_commit_corrected_dual_scale_block(
    float accumulator,
    int32_t low_dot,
    int32_t high_dot,
    float low_weight_scale,
    float high_weight_scale,
    int32_t low_activation_sum,
    int32_t high_activation_sum,
    float low_weight_min,
    float high_weight_min,
    float activation_scale)
{
    float contribution =
        llaminar2::device_native_vnni_contract::dualScaleBlock(
            low_dot,
            high_dot,
            low_weight_scale,
            high_weight_scale,
            activation_scale);
    contribution = llaminar2::device_native_vnni_contract::accumulate(
        contribution,
        llaminar2::device_native_vnni_contract::dualScaleCorrection(
            low_activation_sum,
            high_activation_sum,
            low_weight_min,
            high_weight_min,
            activation_scale));
    return llaminar2::device_native_vnni_contract::accumulate(
        accumulator,
        contribution);
}

/**
 * @brief Commit one IQ1_M dual-scale block and delta as one publication unit.
 *
 * IQ1_M uses four independently signed eight-value grid groups. The group
 * activation sums and signs are combined in the same nested order used by
 * serial decode, then scaled by the two half-block scales and the activation
 * scale.
 *
 * @param accumulator Running FP32 dot-product accumulator.
 * @param low_dot INT32 dot product for elements 0..15 of the K block.
 * @param high_dot INT32 dot product for elements 16..31 of the K block.
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
__device__ __forceinline__ float rocm_native_vnni_commit_iq1_m_block(
    float accumulator,
    int32_t low_dot,
    int32_t high_dot,
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
    float contribution =
        llaminar2::device_native_vnni_contract::dualScaleBlock(
            low_dot,
            high_dot,
            low_weight_scale,
            high_weight_scale,
            activation_scale);
    contribution = llaminar2::device_native_vnni_contract::accumulate(
        contribution,
        llaminar2::device_native_vnni_contract::iq1MDeltaCorrection(
            sum0,
            sum1,
            sum2,
            sum3,
            delta0,
            delta1,
            delta2,
            delta3,
            low_weight_scale,
            high_weight_scale,
            activation_scale));
    return llaminar2::device_native_vnni_contract::accumulate(
        accumulator,
        contribution);
}

/**
 * @brief Apply a router weight with one explicit round-to-nearest operation.
 *
 * Collective publication stores each weighted route row before reducing it,
 * while a single-device kernel may reduce the row immediately.  Expressing the
 * multiplication through the shared retained-rounding primitive gives both
 * forms the same FP32 boundary and prevents contraction into its accumulation.
 *
 * @param route_weight Router probability for one original top-k slot.
 * @param expert_value Complete or split-K-reduced expert down value.
 * @return One independently rounded weighted route contribution.
 */
__device__ __forceinline__ float rocm_native_vnni_weight_route_rn(
    float route_weight,
    float expert_value)
{
    return llaminar2::device_native_vnni_contract::weightRoute(
        route_weight,
        expert_value);
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
    return llaminar2::device_native_vnni_contract::accumulate(
        accumulator,
        contribution);
}
