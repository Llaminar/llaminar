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
#include <cstdint>

namespace llaminar2::cuda
{

    /**
     * @brief Allocate pinned host scalar storage for selected row.
     *
     * Device scalar storage should normally be graph workspace. This helper
     * owns only the stable host source for pre-capture scalar uploads.
     */
    bool allocateRowSelectHostParam(
        int device_ordinal,
        int **host_selected_row);

    /**
     * @brief Allocate pinned host storage for several selected rows.
     *
     * Device storage should be graph workspace. This helper owns only the
     * stable host source used for pre-replay uploads.
     */
    bool allocateRowSelectHostParams(
        int device_ordinal,
        int **host_selected_rows,
        int row_count);

    /**
     * @brief Free pinned host scalar storage allocated by allocateRowSelectHostParam().
     */
    void freeRowSelectHostParam(
        int device_ordinal,
        int *host_selected_row);

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
     * @brief Upload selected-row indices to their stable device address.
     *
     * @param device_selected_rows Device int array destination.
     * @param host_selected_rows Pinned host int array source.
     * @param row_count Number of selected rows to upload.
     * @param stream Opaque CUDA stream pointer used for the async copy.
     * @return true when the copy was enqueued successfully.
     */
    bool uploadRowSelectParams(
        int *device_selected_rows,
        const int *host_selected_rows,
        int row_count,
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
     * @param stream Explicit CUDA stream.
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
     * This is the multi-row counterpart of launchFixedRowSelectFP32(). The
     * source offset and byte count are capture-time geometry, so no row-index
     * array, pinned host parameter, or H2D metadata upload participates in the
     * launch. It is used by MTP catchup graphs whose verifier suffix is known to
     * be contiguous by construction.
     *
     * @param input Device pointer to [seq_len, d_model] FP32 hidden states.
     * @param output Device pointer to [selected_row_count, d_model] FP32 rows.
     * @param first_selected_row First immutable source row.
     * @param selected_row_count Number of contiguous rows copied.
     * @param seq_len Number of source rows, used for validation.
     * @param d_model Number of columns per row.
     * @param stream Explicit non-null CUDA stream.
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

    /**
     * @brief Launch FP32 multi-row select: output[row, :] = input[selected_rows[row], :].
     *
     * @param input Device pointer to [seq_len, d_model] FP32 hidden states.
     * @param output Device pointer to [selected_row_count, d_model] FP32 scratch rows.
     * @param device_selected_rows Device array containing selected row indices.
     * @param seq_len Number of rows in input, used for defensive clamping.
     * @param d_model Number of columns to copy.
     * @param selected_row_count Number of output rows to produce.
     * @param stream Opaque CUDA stream pointer for the kernel launch.
     * @return true when launch was successful.
     */
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
     * For request `r`, the source row is
     * `r * request_row_stride + request_sequence_lengths[r] - 1`. The lengths
     * pointer is read by the CUDA kernel itself, so graph replay observes the
     * current device-owned request geometry without an H2D row-index update.
     *
     * @param input Device pointer to flattened [requests * stride, d_model] rows.
     * @param output Device pointer to compact [request_count, d_model] rows.
     * @param request_sequence_lengths Device INT32 array with one real length per request.
     * @param seq_len Total number of flattened source rows.
     * @param request_row_stride Padded number of rows reserved for each request.
     * @param d_model Number of FP32 columns in each hidden row.
     * @param request_count Number of terminal rows to pack.
     * @param stream Explicit non-null CUDA stream.
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
