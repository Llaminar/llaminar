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

#include <cstdint>

namespace llaminar2::rocm
{

    /** @brief Allocate pinned host scalar storage for selected row. */
    bool allocateRowSelectHostParam(
        int device_ordinal,
        int **host_selected_row);

    /** @brief Allocate pinned host storage for several selected rows. */
    bool allocateRowSelectHostParams(
        int device_ordinal,
        int **host_selected_rows,
        int row_count);

    /** @brief Free pinned host scalar storage allocated by allocateRowSelectHostParam(). */
    void freeRowSelectHostParam(
        int device_ordinal,
        int *host_selected_row);

    /** @brief Upload selected-row scalar to its stable device address. */
    bool uploadRowSelectParam(
        int *device_selected_row,
        const int *host_selected_row,
        void *stream);

    /** @brief Upload selected-row indices to their stable device address. */
    bool uploadRowSelectParams(
        int *device_selected_rows,
        const int *host_selected_rows,
        int row_count,
        void *stream);

    /** @brief Launch FP32 row-select copy: output[0, :] = input[selected_row, :]. */
    bool launchRowSelectFP32(
        const float *input,
        float *output,
        const int *device_selected_row,
        int seq_len,
        int d_model,
        void *stream);

    /**
     * @brief Copy one graph-immutable FP32 source row without replay metadata.
     *
     * The selected row is encoded in the captured kernel arguments. This path
     * is intended for graph-native diagnostic checkpoints whose row cannot
     * change without rebuilding the graph itself.
     *
     * @param input Device pointer to [seq_len, d_model] FP32 hidden states.
     * @param output Device pointer to [1, d_model] FP32 checkpoint storage.
     * @param selected_row Immutable row index encoded in the captured D2D source address.
     * @param seq_len Number of source rows, used for validation.
     * @param d_model Number of columns copied.
     * @param stream Explicit HIP stream.
     * @return true when the contiguous device-to-device copy was accepted.
     */
    bool launchFixedRowSelectFP32(
        const float *input,
        float *output,
        int selected_row,
        int seq_len,
        int d_model,
        void *stream);

    /**
     * @brief Copy one graph-immutable contiguous FP32 row range.
     *
     * The source offset and byte count are capture-time geometry. No pinned
     * host row plan or H2D metadata upload is required.
     *
     * @param input Device pointer to [seq_len, d_model] FP32 hidden states.
     * @param output Device pointer to [selected_row_count, d_model] FP32 rows.
     * @param first_selected_row First immutable source row.
     * @param selected_row_count Number of contiguous rows copied.
     * @param seq_len Number of source rows, used for validation.
     * @param d_model Number of columns per row.
     * @param stream Explicit non-null HIP stream.
     * @return true when the contiguous D2D copy was accepted.
     */
    bool launchFixedRowsSelectFP32(
        const float *input,
        float *output,
        int first_selected_row,
        int selected_row_count,
        int seq_len,
        int d_model,
        void *stream);

    /** @brief Launch FP32 multi-row select: output[row, :] = input[selected_rows[row], :]. */
    bool launchRowsSelectFP32(
        const float *input,
        float *output,
        const int *device_selected_rows,
        int seq_len,
        int d_model,
        int selected_row_count,
        void *stream);

    /**
     * @brief Pack one terminal hidden row per padded request from resident lengths.
     *
     * For request `r`, the HIP kernel reads
     * `r * request_row_stride + request_sequence_lengths[r] - 1`. This makes
     * unequal request geometry a graph-resident input instead of a host replay
     * parameter or a stale verifier-workspace alias.
     *
     * @param input Device pointer to flattened [requests * stride, d_model] rows.
     * @param output Device pointer to compact [request_count, d_model] rows.
     * @param request_sequence_lengths Device INT32 array with one real length per request.
     * @param seq_len Total number of flattened source rows.
     * @param request_row_stride Padded number of rows reserved for each request.
     * @param d_model Number of FP32 columns in each hidden row.
     * @param request_count Number of terminal rows to pack.
     * @param stream Explicit non-null HIP stream.
     * @return true when the graph-capturable kernel launch succeeds.
     */
    bool launchRequestTerminalRowsSelectFP32(
        const float *input,
        float *output,
        const int32_t *request_sequence_lengths,
        int seq_len,
        int request_row_stride,
        int d_model,
        int request_count,
        void *stream);

    /** @brief Launch FP32 MTP concat: output[row] = [embedding[row], hidden[row]]. */
    bool launchMTPConcatFP32(
        const float *hidden,
        const float *embedding,
        float *output,
        int rows,
        int hidden_dim,
        void *stream);

} // namespace llaminar2::rocm
