/**
 * @file FloatingPointVerifierLaunch.h
 * @brief Canonical CUDA/HIP launch ABI for fixed-order floating verifier rows.
 *
 * All bridges borrow persistent matrices and exact non-null streams. Optional
 * row metadata is host recording data copied into GPU arguments; its device
 * count remains controller-owned and must be ordered before the launch.
 * Physical matrices and scratch never change shape when the count changes.
 */
#pragma once

#include "kernels/common/DeviceRowRange.h"

extern "C"
{
    /**
     * @brief Enqueue fixed-order floating projections.
     * @param d_A_array Persistent device pointers to FP32 input matrices [M,K].
     * @param d_B_array Persistent device pointers to native weight matrices [N,K].
     * @param d_C_array Persistent device pointers to FP32 result matrices [M,N].
     * @param M Physical matrix rows; independent of the device live count.
     * @param N Output columns.
     * @param K Reduction width.
     * @param batch_count Number of independent projections.
     * @param device_id Owning backend device ordinal.
     * @param stream Exact non-null producer/consumer stream.
     * @param row_range Borrowed launch metadata, or null for fully active physical rows.
     * @return Whether validation and asynchronous submission succeeded.
     */
    bool cudaFp32_tiny_batched_projection(
        const float *const *d_A_array,
        const float *const *d_B_array,
        float *const *d_C_array,
        int M,
        int N,
        int K,
        int batch_count,
        int device_id,
        void *stream,
        const llaminar2::DeviceRowRange *row_range = nullptr);

    /**
     * @brief Enqueue fixed-order floating projections.
     * @param d_A_array Persistent device pointers to FP32 input matrices [M,K].
     * @param d_B_array Persistent device pointers to native weight matrices [N,K].
     * @param d_C_array Persistent device pointers to FP32 result matrices [M,N].
     * @param M Physical matrix rows; independent of the device live count.
     * @param N Output columns.
     * @param K Reduction width.
     * @param batch_count Number of independent projections.
     * @param weight_dtype Native weight encoding: FP16=0, BF16=1.
     * @param device_id Owning backend device ordinal.
     * @param stream Exact non-null producer/consumer stream.
     * @param row_range Borrowed launch metadata, or null for fully active physical rows.
     * @return Whether validation and asynchronous submission succeeded.
     */
    bool cudaFp32x16_tiny_batched_projection(
        const float *const *d_A_array,
        const float *const *d_B_array,
        float *const *d_C_array,
        int M,
        int N,
        int K,
        int batch_count,
        int weight_dtype,
        int device_id,
        void *stream,
        const llaminar2::DeviceRowRange *row_range = nullptr);

    /**
     * @brief Enqueue fixed-order SwiGLU/down arithmetic.
     * @param d_gate FP32 gate activation matrix [M,K].
     * @param d_up FP32 up activation matrix [M,K].
     * @param d_weights Native weight matrix [N,K].
     * @param d_output FP32 result matrix [M,N].
     * @param M Physical matrix rows; independent of the device live count.
     * @param N Output columns.
     * @param K Reduction width.
     * @param weight_dtype Native weight encoding: FP32=0, FP16=1, BF16=2.
     * @param device_id Owning backend device ordinal.
     * @param stream Exact non-null producer/consumer stream.
     * @param row_range Borrowed launch metadata, or null for fully active physical rows.
     * @return Whether validation and asynchronous submission succeeded.
     */
    bool cudaFloating_swiglu_down_projection(
        const float *d_gate,
        const float *d_up,
        const void *d_weights,
        float *d_output,
        int M,
        int N,
        int K,
        int weight_dtype,
        int device_id,
        void *stream,
        const llaminar2::DeviceRowRange *row_range = nullptr);

    /**
     * @brief Enqueue fixed-order floating projections.
     * @param d_A_array Persistent device pointers to FP32 input matrices [M,K].
     * @param d_B_array Persistent device pointers to native weight matrices [N,K].
     * @param d_C_array Persistent device pointers to FP32 result matrices [M,N].
     * @param M Physical matrix rows; independent of the device live count.
     * @param N Output columns.
     * @param K Reduction width.
     * @param batch_count Number of independent projections.
     * @param device_id Owning backend device ordinal.
     * @param stream Exact non-null producer/consumer stream.
     * @param row_range Borrowed launch metadata, or null for fully active physical rows.
     * @return Whether validation and asynchronous submission succeeded.
     */
    bool rocmFp32_small_n_batched_projection(
        const float *const *d_A_array,
        const float *const *d_B_array,
        float *const *d_C_array,
        int M,
        int N,
        int K,
        int batch_count,
        int device_id,
        void *stream,
        const llaminar2::DeviceRowRange *row_range = nullptr);

    /**
     * @brief Enqueue fixed-order floating projections.
     * @param d_A_array Persistent device pointers to FP32 input matrices [M,K].
     * @param d_B_array Persistent device pointers to native weight matrices [N,K].
     * @param d_C_array Persistent device pointers to FP32 result matrices [M,N].
     * @param M Physical matrix rows; independent of the device live count.
     * @param N Output columns.
     * @param K Reduction width.
     * @param batch_count Number of independent projections.
     * @param weight_dtype Native weight encoding: FP16=0, BF16=1.
     * @param device_id Owning backend device ordinal.
     * @param stream Exact non-null producer/consumer stream.
     * @param row_range Borrowed launch metadata, or null for fully active physical rows.
     * @return Whether validation and asynchronous submission succeeded.
     */
    bool rocmFp32x16_batched_projection(
        const float *const *d_A_array,
        const float *const *d_B_array,
        float *const *d_C_array,
        int M,
        int N,
        int K,
        int batch_count,
        int weight_dtype,
        int device_id,
        void *stream,
        const llaminar2::DeviceRowRange *row_range = nullptr);

    /**
     * @brief Enqueue fixed-order SwiGLU/down arithmetic.
     * @param d_gate FP32 gate activation matrix [M,K].
     * @param d_up FP32 up activation matrix [M,K].
     * @param d_weights Native weight matrix [N,K].
     * @param d_output FP32 result matrix [M,N].
     * @param M Physical matrix rows; independent of the device live count.
     * @param N Output columns.
     * @param K Reduction width.
     * @param weight_dtype Native weight encoding: FP32=0, FP16=1, BF16=2.
     * @param device_id Owning backend device ordinal.
     * @param stream Exact non-null producer/consumer stream.
     * @param row_range Borrowed launch metadata, or null for fully active physical rows.
     * @return Whether validation and asynchronous submission succeeded.
     */
    bool rocmFloating_swiglu_down_projection(
        const float *d_gate,
        const float *d_up,
        const void *d_weights,
        float *d_output,
        int M,
        int N,
        int K,
        int weight_dtype,
        int device_id,
        void *stream,
        const llaminar2::DeviceRowRange *row_range = nullptr);

}
