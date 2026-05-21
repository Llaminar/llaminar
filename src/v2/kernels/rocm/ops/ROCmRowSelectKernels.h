/**
 * @file ROCmRowSelectKernels.h
 * @brief ROCm host wrappers for graph-capturable hidden-state row selection.
 *
 * This is the HIP counterpart to the CUDA row-select helper. The stage owns a
 * pinned host scalar plus a device scalar; HIP graph capture records the scalar
 * H2D copy and a fixed row-copy kernel so replay can change selected rows by
 * updating the pinned host value before graph launch.
 *
 * Lifecycle: allocation/free are owned by HiddenStateRowSelectStage and should
 * occur during warmup before stream capture begins.
 */

#pragma once

namespace llaminar2::rocm
{

    /** @brief Allocate pinned host and device scalar storage for selected row. */
    bool allocateRowSelectParam(
        int device_ordinal,
        int **host_selected_row,
        int **device_selected_row);

    /** @brief Free scalar storage allocated by allocateRowSelectParam(). */
    void freeRowSelectParam(
        int device_ordinal,
        int *host_selected_row,
        int *device_selected_row);

    /** @brief Upload selected-row scalar to its stable device address. */
    bool uploadRowSelectParam(
        int *device_selected_row,
        const int *host_selected_row,
        void *stream);

    /** @brief Launch FP32 row-select copy: output[0, :] = input[selected_row, :]. */
    bool launchRowSelectFP32(
        const float *input,
        float *output,
        const int *device_selected_row,
        int seq_len,
        int d_model,
        void *stream);

} // namespace llaminar2::rocm