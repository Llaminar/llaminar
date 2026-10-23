/**
 * @file ROCmQuantisedActivationProducers.h
 * @brief Canonical ordinary and count-owned single-Q8 producer bridge ABI.
 *
 * Old C symbols retain their full-active calling convention. Explicit rows
 * bridges copy immutable descriptors by value; host code never reads a live
 * count. Native count-owned specializations reject inactive rows before all
 * source and scratch access. Streams, scale/sum pitches and numeric programs
 * remain original. No bridge owns allocations or chooses another arithmetic.
 */
#pragma once
#include "kernels/common/DeviceRowRange.h"
#include <cstdint>
extern "C" {
/**
 * @brief Publish full-active rows through the retained C ABI.
 * @param d_A_fp32 Borrowed row-major [M,K] FP32 source.
 * @param d_A_int8 Row-major [M,K] signed-byte destination; inactive rows untouched.
 * @param d_scales_blockwise Row-major FP32 scales [M,ceil(K/block_size)].
 * @param M Positive physical rows, retained as the original scratch pitch.
 * @param K Positive source width; original block/tail admission is unchanged.
 * @param rocm_device_id Exact owning backend device ordinal.
 * @param stream Exact non-null, non-legacy, non-per-thread-default stream.
 * @param block_size Original HIP block width32/64/128; WithSums requires32.
 * @return Original successful-submission result; native errors remain visible.
 */
bool rocmQuantGemm_quantizeActivationsBlockwise(const float *d_A_fp32,
    int8_t *d_A_int8,
    float *d_scales_blockwise,
    int M, int K,
    int rocm_device_id, void *stream,
    int block_size = 32);
/**
 * @brief Publish device-count-owned rows on the original numeric program.
 * @param d_A_fp32 Borrowed row-major [M,K] FP32 source.
 * @param d_A_int8 Row-major [M,K] signed-byte destination; inactive rows untouched.
 * @param d_scales_blockwise Row-major FP32 scales [M,ceil(K/block_size)].
 * @param M Positive physical rows, retained as the original scratch pitch.
 * @param K Positive source width; original block/tail admission is unchanged.
 * @param rocm_device_id Exact owning backend device ordinal.
 * @param stream Exact non-null, non-legacy, non-per-thread-default stream.
 * @param block_size Original HIP block width32/64/128; WithSums requires32.
 * @param rowRange Optional immutable row authority copied by value; no host live-count read.
 * @return Original successful-submission result; native errors remain visible.
 */
bool rocmQuantGemm_quantizeActivationsBlockwise_rows(const float *d_A_fp32,
    int8_t *d_A_int8,
    float *d_scales_blockwise,
    int M, int K,
    int rocm_device_id, void *stream,
    int block_size, const llaminar2::DeviceRowRange* rowRange);
/**
 * @brief Publish full-active rows through the retained C ABI.
 * @param d_A_fp32 Borrowed row-major [M,K] FP32 source.
 * @param d_A_int8 Row-major [M,K] signed-byte destination; inactive rows untouched.
 * @param d_scales_blockwise Row-major FP32 scales [M,ceil(K/block_size)].
 * @param d_sums_blockwise Optional INT32 block sums [M,K/32], row-major.
 * @param M Positive physical rows, retained as the original scratch pitch.
 * @param K Positive source width; original block/tail admission is unchanged.
 * @param rocm_device_id Exact owning backend device ordinal.
 * @param stream Exact non-null, non-legacy, non-per-thread-default stream.
 * @param block_size Original HIP block width32/64/128; WithSums requires32.
 * @return Original successful-submission result; native errors remain visible.
 */
bool rocmQuantGemm_quantizeActivationsBlockwiseWithSums(const float *d_A_fp32,
    int8_t *d_A_int8,
    float *d_scales_blockwise,
    int32_t *d_sums_blockwise,
    int M, int K,
    int rocm_device_id, void *stream,
    int block_size = 32);
/**
 * @brief Publish device-count-owned rows on the original numeric program.
 * @param d_A_fp32 Borrowed row-major [M,K] FP32 source.
 * @param d_A_int8 Row-major [M,K] signed-byte destination; inactive rows untouched.
 * @param d_scales_blockwise Row-major FP32 scales [M,ceil(K/block_size)].
 * @param d_sums_blockwise Optional INT32 block sums [M,K/32], row-major.
 * @param M Positive physical rows, retained as the original scratch pitch.
 * @param K Positive source width; original block/tail admission is unchanged.
 * @param rocm_device_id Exact owning backend device ordinal.
 * @param stream Exact non-null, non-legacy, non-per-thread-default stream.
 * @param block_size Original HIP block width32/64/128; WithSums requires32.
 * @param rowRange Optional immutable row authority copied by value; no host live-count read.
 * @return Original successful-submission result; native errors remain visible.
 */
bool rocmQuantGemm_quantizeActivationsBlockwiseWithSums_rows(const float *d_A_fp32,
    int8_t *d_A_int8,
    float *d_scales_blockwise,
    int32_t *d_sums_blockwise,
    int M, int K,
    int rocm_device_id, void *stream,
    int block_size, const llaminar2::DeviceRowRange* rowRange);
/**
 * @brief Publish full-active rows through the retained C ABI.
 * @param gate Borrowed row-major [M,K] FP32 gate operand.
 * @param up Borrowed row-major [M,K] FP32 up operand.
 * @param A_int8 Row-major [M,K] signed-byte destination; inactive rows untouched.
 * @param scales_A_blockwise Row-major FP32 scales [M,K/32].
 * @param M Positive physical rows, retained as the original scratch pitch.
 * @param K Positive source width; original block/tail admission is unchanged.
 * @param device_idx Exact owning backend device ordinal.
 * @param stream Exact non-null, non-legacy, non-per-thread-default stream.
 * @return Original successful-submission result; native errors remain visible.
 */
bool hipOps_fused_swiglu_quantize_blockwise(const float *gate,
    const float *up,
    int8_t *A_int8,
    float *scales_A_blockwise,
    int M, int K,
    int device_idx,
    void *stream);
/**
 * @brief Publish device-count-owned rows on the original numeric program.
 * @param gate Borrowed row-major [M,K] FP32 gate operand.
 * @param up Borrowed row-major [M,K] FP32 up operand.
 * @param A_int8 Row-major [M,K] signed-byte destination; inactive rows untouched.
 * @param scales_A_blockwise Row-major FP32 scales [M,K/32].
 * @param M Positive physical rows, retained as the original scratch pitch.
 * @param K Positive source width; original block/tail admission is unchanged.
 * @param device_idx Exact owning backend device ordinal.
 * @param stream Exact non-null, non-legacy, non-per-thread-default stream.
 * @param rowRange Optional immutable row authority copied by value; no host live-count read.
 * @return Original successful-submission result; native errors remain visible.
 */
bool hipOps_fused_swiglu_quantize_blockwise_rows(const float *gate,
    const float *up,
    int8_t *A_int8,
    float *scales_A_blockwise,
    int M, int K,
    int device_idx,
    void *stream, const llaminar2::DeviceRowRange* rowRange);
/**
 * @brief Publish full-active rows through the retained C ABI.
 * @param gate Borrowed row-major [M,K] FP32 gate operand.
 * @param up Borrowed row-major [M,K] FP32 up operand.
 * @param A_int8 Row-major [M,K] signed-byte destination; inactive rows untouched.
 * @param scales_A_blockwise Row-major FP32 scales [M,K/32].
 * @param sums_A_blockwise Optional INT32 block sums [M,K/32], row-major.
 * @param M Positive physical rows, retained as the original scratch pitch.
 * @param K Positive source width; original block/tail admission is unchanged.
 * @param device_idx Exact owning backend device ordinal.
 * @param stream Exact non-null, non-legacy, non-per-thread-default stream.
 * @return Original successful-submission result; native errors remain visible.
 */
bool hipOps_fused_swiglu_quantize_blockwise_with_sums(const float *gate,
    const float *up,
    int8_t *A_int8,
    float *scales_A_blockwise,
    int32_t *sums_A_blockwise,
    int M, int K,
    int device_idx,
    void *stream);
/**
 * @brief Publish device-count-owned rows on the original numeric program.
 * @param gate Borrowed row-major [M,K] FP32 gate operand.
 * @param up Borrowed row-major [M,K] FP32 up operand.
 * @param A_int8 Row-major [M,K] signed-byte destination; inactive rows untouched.
 * @param scales_A_blockwise Row-major FP32 scales [M,K/32].
 * @param sums_A_blockwise Optional INT32 block sums [M,K/32], row-major.
 * @param M Positive physical rows, retained as the original scratch pitch.
 * @param K Positive source width; original block/tail admission is unchanged.
 * @param device_idx Exact owning backend device ordinal.
 * @param stream Exact non-null, non-legacy, non-per-thread-default stream.
 * @param rowRange Optional immutable row authority copied by value; no host live-count read.
 * @return Original successful-submission result; native errors remain visible.
 */
bool hipOps_fused_swiglu_quantize_blockwise_with_sums_rows(const float *gate,
    const float *up,
    int8_t *A_int8,
    float *scales_A_blockwise,
    int32_t *sums_A_blockwise,
    int M, int K,
    int device_idx,
    void *stream, const llaminar2::DeviceRowRange* rowRange);
}
