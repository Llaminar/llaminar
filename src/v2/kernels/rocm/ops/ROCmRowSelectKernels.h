/**
 * @file ROCmRowSelectKernels.h
 * @brief ROCm launch contracts for graph-capturable row and prefill selection.
 *
 * Hidden-state helpers retain their explicitly bounded pre-capture parameter
 * storage, while long-context prefill consumes only device-owned request banks,
 * device geometry, and canonical KV progress. Allocation belongs to stage or
 * arena setup. Every launch and transfer requires the exact non-null stream
 * supplied by its producer; null never aliases HIP's default stream.
 */

#pragma once

#include <cstdint>

namespace llaminar2::rocm
{

    /**
     * @brief Materialize one graph-stable prefill chunk from a resident request bank.
     *
     * The kernel reads canonical KV progress and admitted request geometry on
     * device. It performs no allocation, transfer, callback, or synchronization.
     * See IBackend::enqueuePreparePrefillChunkView() for the full contract.
     */
    bool launchPreparePrefillChunkView(
        const int32_t *request_token_ids,
        const int32_t *request_position_ids,
        const int32_t *request_total_rows,
        const int32_t *cached_tokens,
        int request_row_capacity,
        int bucket_seq_len,
        int32_t pad_token_id,
        int32_t *out_token_ids,
        int32_t *out_position_ids,
        int32_t *out_real_rows,
        int32_t *out_row_stride,
        void *stream);

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

    /**
     * @brief Pack request-terminal rows using a device-owned padded stride.
     *
     * Real lengths and stride are consumed from one event-published geometry
     * record. A captured HIP launch can therefore replay at any prompt width
     * within @p seq_capacity without host-authored row metadata or recapture.
     *
     * @param input Device pointer to maximum-capacity flattened hidden rows.
     * @param output Device pointer to compact [request_count, d_model] rows.
     * @param request_sequence_lengths Device INT32 real-length array.
     * @param request_row_stride Device INT32 scalar containing current padded width.
     * @param seq_capacity Maximum flattened source rows available at input.
     * @param d_model Number of FP32 columns in each hidden row.
     * @param request_count Fixed captured request count.
     * @param stream Explicit non-null HIP stream.
     * @return true when the graph-capturable kernel launch succeeds.
     */
    bool launchDeviceGeometryRequestTerminalRowsSelectFP32(
        const float *input,
        float *output,
        const int32_t *request_sequence_lengths,
        const int32_t *request_row_stride,
        int seq_capacity,
        int d_model,
        int request_count,
        void *stream);

    /**
     * @brief Pack the next shifted-prefill hidden range from canonical KV progress.
     *
     * The captured kernel derives the current segment-relative row as
     * `shifted_count - (main_count - admitted_length)`. Request index and row
     * count are immutable graph geometry; all progress values remain resident
     * at stable device addresses across replay.
     *
     * @param input Maximum-capacity flattened hidden-state rows.
     * @param output Compact contiguous hidden rows consumed by the MTP sidecar.
     * @param main_cached_tokens Canonical main-cache count for @p request_index.
     * @param shifted_cached_tokens Canonical shifted-MTP count for the request.
     * @param request_sequence_lengths Resident current-segment lengths.
     * @param request_row_stride Resident padded request stride.
     * @param request_index Immutable request row owned by this graph.
     * @param seq_capacity Flattened hidden-state capacity.
     * @param d_model FP32 columns per hidden row.
     * @param selected_row_count Number of consecutive rows to pack.
     * @param stream Explicit non-null HIP stream.
     * @return true when the graph-capturable launch succeeds.
     */
    bool launchDeviceKVProgressRowsSelectFP32(
        const float *input,
        float *output,
        const int32_t *main_cached_tokens,
        const int32_t *shifted_cached_tokens,
        const int32_t *request_sequence_lengths,
        const int32_t *request_row_stride,
        int request_index,
        int seq_capacity,
        int d_model,
        int selected_row_count,
        void *stream);

    /**
     * @brief Prepare one request's graph-resident shifted-MTP prefill payload.
     *
     * This is the HIP counterpart of the CUDA transaction preparer. Canonical
     * device KV counts select initial-segment or continuation-bridge arithmetic;
     * malformed progress traps on device instead of producing a partial append.
     * All outputs are persistent arena allocations consumed later in the same
     * captured graph.
     */
    bool launchShiftedMTPPrefillPrepareFP32(
        const float *input_hidden,
        float *terminal_hidden_archive,
        float *packed_hidden_out,
        const int32_t *input_token_ids,
        const int32_t *input_position_ids,
        int32_t *shifted_token_ids_out,
        int32_t *shifted_position_ids_out,
        int32_t *append_lengths_out,
        const int32_t *main_cached_tokens,
        const int32_t *shifted_cached_tokens,
        const int32_t *request_sequence_lengths,
        const int32_t *request_row_stride_device,
        int request_index,
        int request_count,
        int captured_row_stride,
        int seq_capacity,
        int d_model,
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
