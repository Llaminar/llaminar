/**
 * @file CUDARowSelectKernels.h
 * @brief CUDA host wrappers for graph-capturable hidden-state row selection.
 *
 * Provides the tiny CUDA runtime surface used by HiddenStateRowSelectStage:
 * pinned host/device scalar allocation, scalar upload, and one fixed-grid row
 * copy kernel. The captured graph records the scalar upload and kernel launch;
 * later replays read the current value from the same pinned host address.
 *
 * Lifecycle: allocation/free are owned by the stage. Kernel launches run on the
 * stream supplied by the graph executor or on the default stream when null.
 */

#pragma once

#include <cstddef>

namespace llaminar2::cuda
{

    /**
     * @brief Allocate pinned host scalar storage for selected row.
     *
     * Device scalar storage should normally be graph workspace. This helper
     * owns only the stable host replay source for captured H2D copies.
     */
    bool allocateRowSelectHostParam(
        int device_ordinal,
        int **host_selected_row);

    /**
     * @brief Free pinned host scalar storage allocated by allocateRowSelectHostParam().
     */
    void freeRowSelectHostParam(
        int device_ordinal,
        int *host_selected_row);

    /**
     * @brief Allocate pinned host and device scalar storage for selected row.
     *
     * @param device_ordinal CUDA device ordinal that owns the device scalar.
     * @param host_selected_row Receives pinned host int pointer.
     * @param device_selected_row Receives device int pointer.
     * @return true when both allocations succeeded.
     */
    bool allocateRowSelectParam(
        int device_ordinal,
        int **host_selected_row,
        int **device_selected_row);

    /**
     * @brief Free scalar storage allocated by allocateRowSelectParam().
     *
     * @param device_ordinal CUDA device ordinal that owns the device scalar.
     * @param host_selected_row Pinned host int pointer, may be null.
     * @param device_selected_row Device int pointer, may be null.
     */
    void freeRowSelectParam(
        int device_ordinal,
        int *host_selected_row,
        int *device_selected_row);

    /**
     * @brief Upload selected-row scalar to its stable device address.
     *
     * @param device_selected_row Device int destination.
     * @param host_selected_row Pinned host int source.
     * @param stream Opaque CUDA stream pointer used for the async copy.
     * @return true when the copy was enqueued successfully.
     */
    bool uploadRowSelectParam(
        int *device_selected_row,
        const int *host_selected_row,
        void *stream);

    /**
     * @brief Launch FP32 row-select copy: output[0, :] = input[selected_row, :].
     *
     * @param input Device pointer to [seq_len, d_model] FP32 hidden states.
     * @param output Device pointer to [1, d_model] FP32 scratch row.
     * @param device_selected_row Device scalar containing selected row index.
     * @param seq_len Number of rows in input, used for defensive clamping.
     * @param d_model Number of columns to copy.
     * @param stream Opaque CUDA stream pointer for the kernel launch.
     * @return true when launch was successful.
     */
    bool launchRowSelectFP32(
        const float *input,
        float *output,
        const int *device_selected_row,
        int seq_len,
        int d_model,
        void *stream);

    /**
     * @brief Launch FP32 MTP concat: output[row] = [embedding[row], hidden[row]].
     *
     * @param hidden Device pointer to [rows, hidden_dim].
     * @param embedding Device pointer to [rows, hidden_dim].
     * @param output Device pointer to [rows, hidden_dim * 2].
     * @param rows Number of token rows.
     * @param hidden_dim Hidden width of each input.
     * @param stream Opaque CUDA stream pointer for the kernel launch.
     * @return true when launch was successful.
     */
    bool launchMTPConcatFP32(
        const float *hidden,
        const float *embedding,
        float *output,
        int rows,
        int hidden_dim,
        void *stream);

} // namespace llaminar2::cuda
