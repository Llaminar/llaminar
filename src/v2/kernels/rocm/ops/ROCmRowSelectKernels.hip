/**
 * @file ROCmRowSelectKernels.hip
 * @brief HIP kernels for graph-capturable row and prefill-window selection.
 *
 * Row-selection entry points compact hidden-state rows into persistent graph
 * buffers. The prefill materializer derives each request window entirely from
 * admitted device tensors and canonical device KV progress, allowing one fixed
 * graph executable to serve every chunk of a long prompt. Every launch requires
 * the exact non-null producer stream; no kernel in this file may infer ordering
 * from HIP's default stream.
 */

#include "ROCmRowSelectKernels.h"
#include "../../../utils/Logger.h"

#ifdef HAVE_ROCM

#include <algorithm>
#include <hip/hip_runtime.h>

namespace llaminar2::rocm
{
    namespace
    {
        /**
         * @brief Build one resident prefill bucket from canonical request/KV state.
         *
         * Every work-item independently derives the immutable source window
         * from device scalars. No inter-workgroup barrier or atomic is needed
         * because the later captured KV append is the sole progress writer.
         * Full four-row groups use 128-bit token and position transfers when
         * all participating addresses are naturally aligned.
         */
        __global__ void preparePrefillChunkViewKernel(
            const int32_t *__restrict__ request_token_ids,
            const int32_t *__restrict__ request_position_ids,
            const int32_t *__restrict__ request_total_rows,
            const int32_t *__restrict__ cached_tokens,
            int request_row_capacity,
            int bucket_seq_len,
            int32_t pad_token_id,
            int32_t *__restrict__ out_token_ids,
            int32_t *__restrict__ out_position_ids,
            int32_t *__restrict__ out_real_rows,
            int32_t *__restrict__ out_row_stride)
        {
            const int total_rows = *request_total_rows;
            const int request_first_position = request_position_ids[0];
            const int cached_row_count = *cached_tokens;
            const int source_start = cached_row_count - request_first_position;
            const bool geometry_valid =
                total_rows > 0 && total_rows <= request_row_capacity &&
                source_start >= 0 && source_start < total_rows &&
                bucket_seq_len > 0 && bucket_seq_len <= request_row_capacity;
            const int global_thread =
                static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            if (!geometry_valid)
            {
                if (global_thread == 0)
                    __builtin_trap();
                return;
            }

            const int real_rows = min(bucket_seq_len, total_rows - source_start);
            if (global_thread == 0)
            {
                *out_real_rows = real_rows;
                *out_row_stride = bucket_seq_len;
            }

            const int vector_count = (bucket_seq_len + 3) / 4;
            const int grid_stride =
                static_cast<int>(blockDim.x * gridDim.x);
            const bool aligned_vector_path =
                ((reinterpret_cast<uintptr_t>(request_token_ids + source_start) |
                  reinterpret_cast<uintptr_t>(request_position_ids + source_start) |
                  reinterpret_cast<uintptr_t>(out_token_ids) |
                  reinterpret_cast<uintptr_t>(out_position_ids)) &
                 uintptr_t{0x0f}) == 0;

            for (int vector_index = global_thread;
                 vector_index < vector_count;
                 vector_index += grid_stride)
            {
                const int row = vector_index * 4;
                if (row + 3 < real_rows && aligned_vector_path)
                {
                    reinterpret_cast<int4 *>(out_token_ids)[vector_index] =
                        reinterpret_cast<const int4 *>(
                            request_token_ids + source_start)[vector_index];
                    reinterpret_cast<int4 *>(out_position_ids)[vector_index] =
                        reinterpret_cast<const int4 *>(
                            request_position_ids + source_start)[vector_index];
                    continue;
                }

#pragma unroll
                for (int lane = 0; lane < 4; ++lane)
                {
                    const int output_row = row + lane;
                    if (output_row >= bucket_seq_len)
                        break;
                    const bool active = output_row < real_rows;
                    out_token_ids[output_row] =
                        active
                            ? request_token_ids[source_start + output_row]
                            : pad_token_id;
                    out_position_ids[output_row] =
                        active
                            ? request_position_ids[source_start + output_row]
                            : request_first_position + source_start + output_row;
                }
            }
        }

        /// @brief Copy one selected FP32 row using a grid-stride loop over columns.
        __global__ void rowSelectFP32Kernel(
            const float *__restrict__ input,
            float *__restrict__ output,
            const int *__restrict__ selected_row_ptr,
            int seq_len,
            int d_model)
        {
            const int raw_selected_row = selected_row_ptr ? *selected_row_ptr : 0;
            const int upper_bound_row = seq_len - 1;
            const int selected_row = raw_selected_row < 0
                                         ? 0
                                         : (raw_selected_row > upper_bound_row ? upper_bound_row : raw_selected_row);
            const size_t source_base = static_cast<size_t>(selected_row) * static_cast<size_t>(d_model);

            const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = blockDim.x * gridDim.x;
            for (int column = thread_index; column < d_model; column += stride)
            {
                output[column] = input[source_base + static_cast<size_t>(column)];
            }
        }

        /// @brief Copy several selected FP32 rows using a grid-stride loop over the compact output.
        __global__ void rowsSelectFP32Kernel(
            const float *__restrict__ input,
            float *__restrict__ output,
            const int *__restrict__ selected_rows,
            int seq_len,
            int d_model,
            int selected_row_count)
        {
            const int total = selected_row_count * d_model;
            const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = blockDim.x * gridDim.x;
            const int upper_bound_row = seq_len - 1;
            for (int idx = thread_index; idx < total; idx += stride)
            {
                const int output_row = idx / d_model;
                const int column = idx - output_row * d_model;
                const int raw_selected_row = selected_rows ? selected_rows[output_row] : 0;
                const int selected_row = raw_selected_row < 0
                                             ? 0
                                             : (raw_selected_row > upper_bound_row ? upper_bound_row : raw_selected_row);
                const size_t source_offset =
                    static_cast<size_t>(selected_row) * static_cast<size_t>(d_model) +
                    static_cast<size_t>(column);
                output[static_cast<size_t>(idx)] = input[source_offset];
            }
        }

        /**
         * @brief Pack request-terminal rows using device-resident real lengths.
         *
         * The request length and source-row arithmetic remain in the captured
         * HIP graph. This lets one executable replay with new request geometry
         * without adopting host row indices or verifier metadata from another
         * transaction.
         */
        __global__ void requestTerminalRowsSelectFP32Kernel(
            const float *__restrict__ input,
            float *__restrict__ output,
            const int32_t *__restrict__ request_sequence_lengths,
            int seq_len,
            int request_row_stride,
            int d_model,
            int request_count)
        {
            const int total = request_count * d_model;
            const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = blockDim.x * gridDim.x;
            for (int idx = thread_index; idx < total; idx += stride)
            {
                const int request = idx / d_model;
                const int column = idx - request * d_model;

                /*
                 * The serving boundary rejects empty or oversized requests.
                 * Defensive device clamping protects memory if a malformed
                 * direct caller bypasses that validation.
                 */
                const int raw_length = request_sequence_lengths[request];
                const int request_length = raw_length < 1
                                               ? 1
                                               : (raw_length > request_row_stride
                                                      ? request_row_stride
                                                      : raw_length);
                const int raw_source_row =
                    request * request_row_stride + request_length - 1;
                const int source_row = raw_source_row < seq_len
                                           ? raw_source_row
                                           : seq_len - 1;
                const size_t source_offset =
                    static_cast<size_t>(source_row) * static_cast<size_t>(d_model) +
                    static_cast<size_t>(column);
                output[static_cast<size_t>(idx)] = input[source_offset];
            }
        }

        /**
         * @brief Device-geometry variant used by prompt-width-total graphs.
         *
         * The stride and lengths belong to one request-admission publication.
         * One cooperative scalar load per workgroup keeps the row derivation
         * inexpensive without moving any geometry back through the host.
         */
        __global__ void deviceGeometryRequestTerminalRowsSelectFP32Kernel(
            const float *__restrict__ input,
            float *__restrict__ output,
            const int32_t *__restrict__ request_sequence_lengths,
            const int32_t *__restrict__ request_row_stride_device,
            int seq_capacity,
            int d_model,
            int request_count)
        {
            __shared__ int request_row_stride;
            if (threadIdx.x == 0)
                request_row_stride = *request_row_stride_device;
            __syncthreads();

            const int total = request_count * d_model;
            const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
            const int grid_stride = blockDim.x * gridDim.x;
            for (int idx = thread_index; idx < total; idx += grid_stride)
            {
                const int request = idx / d_model;
                const int column = idx - request * d_model;
                const int maximum_row_stride =
                    seq_capacity / request_count;
                const int bounded_row_stride =
                    request_row_stride < 1
                        ? 1
                        : (request_row_stride > maximum_row_stride
                               ? maximum_row_stride
                               : request_row_stride);
                const int raw_length = request_sequence_lengths[request];
                const int request_length = raw_length < 1
                                               ? 1
                                               : (raw_length > bounded_row_stride
                                                      ? bounded_row_stride
                                                      : raw_length);
                const int raw_source_row =
                    request * bounded_row_stride + request_length - 1;
                const int source_row = raw_source_row < seq_capacity
                                           ? raw_source_row
                                           : seq_capacity - 1;
                const size_t source_offset =
                    static_cast<size_t>(source_row) *
                        static_cast<size_t>(d_model) +
                    static_cast<size_t>(column);
                output[static_cast<size_t>(idx)] = input[source_offset];
            }
        }

        /**
         * @brief Copy the next shifted-prefill range named by live KV progress.
         *
         * Main KV has already consumed the complete current request segment.
         * Subtracting its admitted segment length recovers that segment's
         * absolute base. Shifted KV advances after each sidecar replay, so its
         * canonical count identifies the next unconsumed hidden row. This
         * arithmetic is deliberately inside the captured kernel: the host never
         * owns or uploads a replay cursor.
         */
        __global__ void deviceKVProgressRowsSelectFP32Kernel(
            const float *__restrict__ input,
            float *__restrict__ output,
            const int32_t *__restrict__ main_cached_tokens,
            const int32_t *__restrict__ shifted_cached_tokens,
            const int32_t *__restrict__ request_sequence_lengths,
            const int32_t *__restrict__ request_row_stride_device,
            int request_index,
            int seq_capacity,
            int d_model,
            int selected_row_count)
        {
            __shared__ int source_row_start;
            __shared__ int progress_valid;
            if (threadIdx.x == 0)
            {
                const int request_row_stride = *request_row_stride_device;
                const int request_length =
                    request_sequence_lengths[request_index];
                const int main_count = *main_cached_tokens;
                const int shifted_count = *shifted_cached_tokens;
                const int segment_base = main_count - request_length;
                const int segment_row = shifted_count - segment_base;
                const int request_base = request_index * request_row_stride;

                progress_valid =
                    request_row_stride > 0 &&
                    request_length > 0 &&
                    request_length <= request_row_stride &&
                    segment_base >= 0 &&
                    segment_row >= 0 &&
                    selected_row_count <= request_length - segment_row &&
                    request_base >= 0 &&
                    request_base <= seq_capacity - selected_row_count &&
                    segment_row <=
                        seq_capacity - request_base - selected_row_count;
                source_row_start = request_base + segment_row;
            }
            __syncthreads();

            const int total = selected_row_count * d_model;
            const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
            const int grid_stride = blockDim.x * gridDim.x;
            for (int idx = thread_index; idx < total; idx += grid_stride)
            {
                if (!progress_valid)
                {
                    output[static_cast<size_t>(idx)] =
                        __int_as_float(0x7fc00000);
                    continue;
                }
                const int output_row = idx / d_model;
                const int column = idx - output_row * d_model;
                const size_t source_offset =
                    static_cast<size_t>(source_row_start + output_row) *
                        static_cast<size_t>(d_model) +
                    static_cast<size_t>(column);
                output[static_cast<size_t>(idx)] = input[source_offset];
            }
        }

        /** @brief HIP implementation of the captured shifted-prefill transaction preparer. */
        template <bool RowsAreFloat4Aligned>
        __global__ void shiftedMTPPrefillPrepareFP32Kernel(
            const float *__restrict__ input_hidden,
            float *__restrict__ terminal_hidden_archive,
            float *__restrict__ packed_hidden_out,
            const int32_t *__restrict__ input_token_ids,
            const int32_t *__restrict__ input_position_ids,
            int32_t *__restrict__ shifted_token_ids_out,
            int32_t *__restrict__ shifted_position_ids_out,
            int32_t *__restrict__ append_lengths_out,
            const int32_t *__restrict__ main_cached_tokens,
            const int32_t *__restrict__ shifted_cached_tokens,
            const int32_t *__restrict__ request_sequence_lengths,
            const int32_t *__restrict__ request_row_stride_device,
            int request_index,
            int request_count,
            int captured_row_stride,
            int seq_capacity,
            int d_model)
        {
            const int request_length = request_sequence_lengths[request_index];
            const int live_row_stride = *request_row_stride_device;
            const int main_count = *main_cached_tokens;
            const int shifted_count = *shifted_cached_tokens;
            const int segment_base = main_count - request_length;
            const bool initial_segment = shifted_count == segment_base;
            const bool continuation_bridge =
                shifted_count >= 0 && shifted_count + 1 == segment_base;
            const bool valid =
                request_count > 0 && request_index >= 0 &&
                request_index < request_count && captured_row_stride > 0 &&
                live_row_stride > 0 &&
                (request_count == 1 ||
                 live_row_stride == captured_row_stride) &&
                request_length > 0 &&
                request_length <= live_row_stride &&
                request_length <= captured_row_stride && segment_base >= 0 &&
                (initial_segment || continuation_bridge) &&
                request_count <= seq_capacity / captured_row_stride;
            if (!valid)
                __builtin_trap();

            const int append_length = continuation_bridge
                                          ? request_length
                                          : request_length - 1;
            if (blockIdx.x == 0 && threadIdx.x == 0)
                append_lengths_out[request_index] = append_length;

            const int output_request_base =
                request_index * captured_row_stride;
            const int hidden_request_base = output_request_base;
            const int input_request_base = request_index * live_row_stride;
            /*
             * Mirror CUDA's four-column work item so both GPU backends share
             * one arithmetic contract. Aligned model widths use native
             * 128-bit accesses; arbitrary widths use the compile-time scalar
             * tail specialization without introducing another transaction.
             */
            const int vector_columns = (d_model + 3) / 4;
            const int flattened_vectors =
                captured_row_stride * vector_columns;
            const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
            const int grid_stride = blockDim.x * gridDim.x;
            for (int vector_idx = thread_index;
                 vector_idx < flattened_vectors;
                 vector_idx += grid_stride)
            {
                const int output_row_in_request =
                    vector_idx / vector_columns;
                const int vector_column =
                    vector_idx - output_row_in_request * vector_columns;
                const int column = vector_column * 4;
                const int output_row =
                    output_request_base + output_row_in_request;
                const bool real_row = output_row_in_request < append_length;

                const float *source_hidden = nullptr;
                int source_token_row = input_request_base;
                if (real_row)
                {
                    if (continuation_bridge && output_row_in_request == 0)
                    {
                        source_hidden = terminal_hidden_archive +
                            static_cast<size_t>(request_index) * d_model;
                    }
                    else
                    {
                        const int source_hidden_row =
                            hidden_request_base + output_row_in_request -
                            (continuation_bridge ? 1 : 0);
                        source_hidden = input_hidden +
                            static_cast<size_t>(source_hidden_row) * d_model;
                    }
                    source_token_row =
                        input_request_base + output_row_in_request +
                        (initial_segment ? 1 : 0);
                }

                float *destination_hidden = packed_hidden_out +
                    static_cast<size_t>(output_row) * d_model;
                if constexpr (RowsAreFloat4Aligned)
                {
                    const float4 hidden_value = real_row
                        ? *reinterpret_cast<const float4 *>(
                              source_hidden + column)
                        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    *reinterpret_cast<float4 *>(destination_hidden + column) =
                        hidden_value;
                }
                else
                {
#pragma unroll
                    for (int lane = 0; lane < 4; ++lane)
                    {
                        const int scalar_column = column + lane;
                        if (scalar_column < d_model)
                        {
                            destination_hidden[scalar_column] = real_row
                                ? source_hidden[scalar_column]
                                : 0.0f;
                        }
                    }
                }

                /*
                 * Row-zero work items consume their prior bridge columns
                 * before publishing the replacement terminal columns. Since
                 * the same thread owns both operations, the fused transaction
                 * needs neither a second launch nor a grid-wide barrier.
                 */
                if (output_row_in_request == 0)
                {
                    const float *terminal_source = input_hidden +
                        static_cast<size_t>(
                            hidden_request_base + request_length - 1) *
                            d_model;
                    float *terminal_destination = terminal_hidden_archive +
                        static_cast<size_t>(request_index) * d_model;
                    if constexpr (RowsAreFloat4Aligned)
                    {
                        *reinterpret_cast<float4 *>(
                            terminal_destination + column) =
                            *reinterpret_cast<const float4 *>(
                                terminal_source + column);
                    }
                    else
                    {
#pragma unroll
                        for (int lane = 0; lane < 4; ++lane)
                        {
                            const int scalar_column = column + lane;
                            if (scalar_column < d_model)
                            {
                                terminal_destination[scalar_column] =
                                    terminal_source[scalar_column];
                            }
                        }
                    }
                }

                if (vector_column == 0)
                {
                    shifted_token_ids_out[output_row] =
                        real_row ? input_token_ids[source_token_row] : int32_t{0};
                    shifted_position_ids_out[output_row] =
                        real_row ? input_position_ids[source_token_row] : int32_t{0};
                }
            }
        }

        /// @brief Concatenate two [rows, hidden_dim] matrices row-wise as [embedding, hidden].
        __global__ void mtpConcatFP32Kernel(
            const float *__restrict__ hidden,
            const float *__restrict__ embedding,
            float *__restrict__ output,
            int rows,
            int hidden_dim)
        {
            const int total = rows * hidden_dim;
            const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = blockDim.x * gridDim.x;
            for (int idx = thread_index; idx < total; idx += stride)
            {
                const int row = idx / hidden_dim;
                const int col = idx - row * hidden_dim;
                const size_t src_offset = static_cast<size_t>(idx);
                const size_t dst_offset =
                    (static_cast<size_t>(row) * static_cast<size_t>(hidden_dim) * 2) +
                    static_cast<size_t>(col);
                output[dst_offset] = embedding[src_offset];
                output[dst_offset + static_cast<size_t>(hidden_dim)] = hidden[src_offset];
            }
        }

        /// @brief Return true if a HIP status is successful.
        bool ok(hipError_t status)
        {
            return status == hipSuccess;
        }
    }

    bool allocateRowSelectHostParam(
        int device_ordinal,
        int **host_selected_row)
    {
        if (!host_selected_row)
            return false;

        *host_selected_row = nullptr;

        // Pin the host scalar so the captured hipMemcpyAsync replays from a
        // stable address and avoids pageable-memory staging inside capture.
        if (!ok(hipSetDevice(device_ordinal)))
            return false;
        return ok(hipHostMalloc(reinterpret_cast<void **>(host_selected_row), sizeof(int), hipHostMallocDefault));
    }

    bool allocateRowSelectHostParams(
        int device_ordinal,
        int **host_selected_rows,
        int row_count)
    {
        if (!host_selected_rows || row_count <= 0)
            return false;

        *host_selected_rows = nullptr;

        if (!ok(hipSetDevice(device_ordinal)))
            return false;
        return ok(hipHostMalloc(
            reinterpret_cast<void **>(host_selected_rows),
            static_cast<size_t>(row_count) * sizeof(int),
            hipHostMallocDefault));
    }

    void freeRowSelectHostParam(
        int device_ordinal,
        int *host_selected_row)
    {
        (void)hipSetDevice(device_ordinal);
        if (host_selected_row)
            (void)hipHostFree(host_selected_row);
    }

    bool uploadRowSelectParam(
        int *device_selected_row,
        const int *host_selected_row,
        void *stream)
    {
        if (!device_selected_row || !host_selected_row || !stream)
            return false;
        auto hip_stream = reinterpret_cast<hipStream_t>(stream);
        return ok(hipMemcpyAsync(
            device_selected_row,
            host_selected_row,
            sizeof(int),
            hipMemcpyHostToDevice,
            hip_stream));
    }

    bool uploadRowSelectParams(
        int *device_selected_rows,
        const int *host_selected_rows,
        int row_count,
        void *stream)
    {
        if (!device_selected_rows || !host_selected_rows || row_count <= 0 || !stream)
            return false;
        auto hip_stream = reinterpret_cast<hipStream_t>(stream);
        return ok(hipMemcpyAsync(
            device_selected_rows,
            host_selected_rows,
            static_cast<size_t>(row_count) * sizeof(int),
            hipMemcpyHostToDevice,
            hip_stream));
    }

    bool launchRowSelectFP32(
        const float *input,
        float *output,
        const int *device_selected_row,
        int seq_len,
        int d_model,
        void *stream)
    {
        if (!input || !output || !device_selected_row || seq_len <= 0 ||
            d_model <= 0 || !stream)
            return false;

        constexpr int threads_per_block = 256;
        const int blocks = std::max(1, std::min(1024, (d_model + threads_per_block - 1) / threads_per_block));
        auto hip_stream = reinterpret_cast<hipStream_t>(stream);
        hipLaunchKernelGGL(
            rowSelectFP32Kernel,
            dim3(blocks),
            dim3(threads_per_block),
            0,
            hip_stream,
            input,
            output,
            device_selected_row,
            seq_len,
            d_model);
        return ok(hipGetLastError());
    }

    bool launchFixedRowSelectFP32(
        const float *input,
        float *output,
        int selected_row,
        int seq_len,
        int d_model,
        void *stream)
    {
        if (!input || !output || selected_row < 0 ||
            selected_row >= seq_len || seq_len <= 0 || d_model <= 0 ||
            !stream)
        {
            return false;
        }

        /*
         * The graph geometry fixes both row and width. A contiguous D2D copy
         * therefore gives the runtime's copy engine the complete transfer,
         * avoids per-thread bounds work, and captures no host-owned replay
         * scalar. Rebuilding for another geometry naturally records another
         * immutable source address.
         */
        const size_t source_offset =
            static_cast<size_t>(selected_row) *
            static_cast<size_t>(d_model);
        return ok(hipMemcpyAsync(
            output,
            input + source_offset,
            static_cast<size_t>(d_model) * sizeof(float),
            hipMemcpyDeviceToDevice,
            reinterpret_cast<hipStream_t>(stream)));
    }

    bool launchFixedRowsSelectFP32(
        const float *input,
        float *output,
        int first_selected_row,
        int selected_row_count,
        int seq_len,
        int d_model,
        void *stream)
    {
        if (!input || !output || !stream || first_selected_row < 0 ||
            selected_row_count <= 0 || seq_len <= 0 || d_model <= 0 ||
            first_selected_row > seq_len - selected_row_count)
        {
            return false;
        }

        const size_t source_offset =
            static_cast<size_t>(first_selected_row) *
            static_cast<size_t>(d_model);
        const size_t element_count =
            static_cast<size_t>(selected_row_count) *
            static_cast<size_t>(d_model);
        return ok(hipMemcpyAsync(
            output,
            input + source_offset,
            element_count * sizeof(float),
            hipMemcpyDeviceToDevice,
            reinterpret_cast<hipStream_t>(stream)));
    }

    bool launchRowsSelectFP32(
        const float *input,
        float *output,
        const int *device_selected_rows,
        int seq_len,
        int d_model,
        int selected_row_count,
        void *stream)
    {
        if (!input || !output || !device_selected_rows || seq_len <= 0 || d_model <= 0 ||
            selected_row_count <= 0 || !stream)
            return false;

        constexpr int threads_per_block = 256;
        const int total = selected_row_count * d_model;
        const int blocks = std::max(1, std::min(1024, (total + threads_per_block - 1) / threads_per_block));
        auto hip_stream = reinterpret_cast<hipStream_t>(stream);
        hipLaunchKernelGGL(
            rowsSelectFP32Kernel,
            dim3(blocks),
            dim3(threads_per_block),
            0,
            hip_stream,
            input,
            output,
            device_selected_rows,
            seq_len,
            d_model,
            selected_row_count);
        return ok(hipGetLastError());
    }

    bool launchRequestTerminalRowsSelectFP32(
        const float *input,
        float *output,
        const int32_t *request_sequence_lengths,
        int seq_len,
        int request_row_stride,
        int d_model,
        int request_count,
        void *stream)
    {
        if (!input || !output || !request_sequence_lengths || !stream ||
            seq_len <= 0 || request_row_stride <= 0 || d_model <= 0 ||
            request_count <= 0 ||
            request_count * request_row_stride != seq_len)
        {
            LOG_ERROR(
                "[ROCmRowSelectKernels] Request-terminal row selection "
                "rejected an incomplete launch contract"
                << " input=" << static_cast<const void *>(input)
                << " output=" << static_cast<void *>(output)
                << " lengths="
                << static_cast<const void *>(request_sequence_lengths)
                << " stream=" << stream
                << " seq_len=" << seq_len
                << " request_row_stride=" << request_row_stride
                << " d_model=" << d_model
                << " request_count=" << request_count);
            return false;
        }

        constexpr int threads_per_block = 256;
        const int total = request_count * d_model;
        const int blocks = std::max(
            1,
            std::min(
                1024,
                (total + threads_per_block - 1) / threads_per_block));
        auto hip_stream = reinterpret_cast<hipStream_t>(stream);
        hipLaunchKernelGGL(
            requestTerminalRowsSelectFP32Kernel,
            dim3(blocks),
            dim3(threads_per_block),
            0,
            hip_stream,
            input,
            output,
            request_sequence_lengths,
            seq_len,
            request_row_stride,
            d_model,
            request_count);
        const hipError_t launch_status = hipGetLastError();
        if (launch_status != hipSuccess)
        {
            int current_device = -1;
            int stream_device = -1;
            const hipError_t current_device_status =
                hipGetDevice(&current_device);
            const hipError_t stream_device_status =
                hipStreamGetDevice(hip_stream, &stream_device);
            LOG_ERROR(
                "[ROCmRowSelectKernels] Request-terminal row-selection launch "
                "failed: "
                << hipGetErrorString(launch_status)
                << " current_device=" << current_device
                << " current_device_status="
                << hipGetErrorString(current_device_status)
                << " stream_device=" << stream_device
                << " stream_device_status="
                << hipGetErrorString(stream_device_status)
                << " stream=" << stream
                << " input=" << static_cast<const void *>(input)
                << " output=" << static_cast<void *>(output)
                << " lengths="
                << static_cast<const void *>(request_sequence_lengths)
                << " seq_len=" << seq_len
                << " request_row_stride=" << request_row_stride
                << " d_model=" << d_model
                << " request_count=" << request_count);
            return false;
        }
        return true;
    }

    bool launchMTPConcatFP32(
        const float *hidden,
        const float *embedding,
        float *output,
        int rows,
        int hidden_dim,
        void *stream)
    {
        if (!hidden || !embedding || !output || rows <= 0 ||
            hidden_dim <= 0 || !stream)
            return false;

        constexpr int threads_per_block = 256;
        const int total = rows * hidden_dim;
        const int blocks = std::max(1, std::min(1024, (total + threads_per_block - 1) / threads_per_block));
        auto hip_stream = reinterpret_cast<hipStream_t>(stream);
        hipLaunchKernelGGL(
            mtpConcatFP32Kernel,
            dim3(blocks),
            dim3(threads_per_block),
            0,
            hip_stream,
            hidden,
            embedding,
            output,
            rows,
            hidden_dim);
        return ok(hipGetLastError());
    }

    bool launchDeviceGeometryRequestTerminalRowsSelectFP32(
        const float *input,
        float *output,
        const int32_t *request_sequence_lengths,
        const int32_t *request_row_stride,
        int seq_capacity,
        int d_model,
        int request_count,
        void *stream)
    {
        if (!input || !output || !request_sequence_lengths ||
            !request_row_stride || !stream || seq_capacity <= 0 ||
            d_model <= 0 || request_count <= 0 ||
            request_count > seq_capacity)
        {
            LOG_ERROR(
                "[ROCmRowSelectKernels] Device-geometry request-terminal "
                "selection rejected an incomplete launch contract"
                << " input=" << static_cast<const void *>(input)
                << " output=" << static_cast<void *>(output)
                << " lengths="
                << static_cast<const void *>(request_sequence_lengths)
                << " row_stride="
                << static_cast<const void *>(request_row_stride)
                << " stream=" << stream
                << " seq_capacity=" << seq_capacity
                << " d_model=" << d_model
                << " request_count=" << request_count);
            return false;
        }

        constexpr int threads_per_block = 256;
        const int total = request_count * d_model;
        const int blocks = std::max(
            1,
            std::min(
                1024,
                (total + threads_per_block - 1) / threads_per_block));
        hipLaunchKernelGGL(
            deviceGeometryRequestTerminalRowsSelectFP32Kernel,
            dim3(blocks),
            dim3(threads_per_block),
            0,
            reinterpret_cast<hipStream_t>(stream),
            input,
            output,
            request_sequence_lengths,
            request_row_stride,
            seq_capacity,
            d_model,
            request_count);
        const hipError_t launch_status = hipGetLastError();
        if (launch_status != hipSuccess)
        {
            LOG_ERROR(
                "[ROCmRowSelectKernels] Device-geometry request-terminal "
                "row-selection launch failed: "
                << hipGetErrorString(launch_status)
                << " stream=" << stream
                << " seq_capacity=" << seq_capacity
                << " d_model=" << d_model
                << " request_count=" << request_count);
            return false;
        }
        return true;
    }

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
        void *stream)
    {
        if (!input || !output || !main_cached_tokens ||
            !shifted_cached_tokens || !request_sequence_lengths ||
            !request_row_stride || !stream || request_index < 0 ||
            seq_capacity <= 0 || d_model <= 0 || selected_row_count <= 0 ||
            selected_row_count > seq_capacity)
        {
            LOG_ERROR("[ROCmRowSelectKernels] Device-KV-progress row selection rejected an incomplete launch contract"
                      << " request=" << request_index
                      << " rows=" << selected_row_count
                      << " seq_capacity=" << seq_capacity
                      << " main_count=" << main_cached_tokens
                      << " shifted_count=" << shifted_cached_tokens
                      << " lengths=" << request_sequence_lengths
                      << " stride=" << request_row_stride
                      << " stream=" << stream);
            return false;
        }

        constexpr int threads_per_block = 256;
        const int total = selected_row_count * d_model;
        const int blocks = std::max(
            1,
            std::min(
                1024,
                (total + threads_per_block - 1) / threads_per_block));
        hipLaunchKernelGGL(
            deviceKVProgressRowsSelectFP32Kernel,
            dim3(blocks),
            dim3(threads_per_block),
            0,
            reinterpret_cast<hipStream_t>(stream),
            input,
            output,
            main_cached_tokens,
            shifted_cached_tokens,
            request_sequence_lengths,
            request_row_stride,
            request_index,
            seq_capacity,
            d_model,
            selected_row_count);
        const hipError_t launch_status = hipGetLastError();
        if (launch_status != hipSuccess)
        {
            LOG_ERROR("[ROCmRowSelectKernels] Device-KV-progress row-selection launch failed: "
                      << hipGetErrorString(launch_status)
                      << " request=" << request_index
                      << " rows=" << selected_row_count
                      << " stream=" << stream);
            return false;
        }
        return true;
    }

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
        void *stream)
    {
        if (!input_hidden || !terminal_hidden_archive || !packed_hidden_out ||
            !input_token_ids || !input_position_ids || !shifted_token_ids_out ||
            !shifted_position_ids_out || !append_lengths_out ||
            !main_cached_tokens || !shifted_cached_tokens ||
            !request_sequence_lengths || !request_row_stride_device || !stream ||
            request_index < 0 || request_index >= request_count ||
            request_count <= 0 || captured_row_stride <= 0 ||
            seq_capacity < request_count * captured_row_stride || d_model <= 0)
        {
            LOG_ERROR("[ROCmRowSelectKernels] Shifted-MTP prefill preparation rejected an incomplete launch contract"
                      << " request=" << request_index << "/" << request_count
                      << " row_stride=" << captured_row_stride
                      << " seq_capacity=" << seq_capacity
                      << " d_model=" << d_model
                      << " stream=" << stream);
            return false;
        }

        constexpr int threads_per_block = 256;
        const int vector_columns = (d_model + 3) / 4;
        const int vectors = captured_row_stride * vector_columns;
        const int blocks = std::max(
            1,
            std::min(1024, (vectors + threads_per_block - 1) / threads_per_block));

        const bool rows_are_float4_aligned = d_model % 4 == 0;
        const auto is_float4_aligned = [](const void *pointer)
        {
            return (reinterpret_cast<std::uintptr_t>(pointer) & 0x0fU) == 0U;
        };
        if (rows_are_float4_aligned &&
            (!is_float4_aligned(input_hidden) ||
             !is_float4_aligned(terminal_hidden_archive) ||
             !is_float4_aligned(packed_hidden_out)))
        {
            LOG_ERROR("[ROCmRowSelectKernels] Shifted-MTP vector preparation requires 16-byte-aligned hidden tensors"
                      << " input=" << input_hidden
                      << " archive=" << terminal_hidden_archive
                      << " output=" << packed_hidden_out);
            return false;
        }

        const auto launch = [&]<bool RowsAreFloat4Aligned>()
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    shiftedMTPPrefillPrepareFP32Kernel<RowsAreFloat4Aligned>),
                dim3(blocks),
                dim3(threads_per_block),
                0,
                reinterpret_cast<hipStream_t>(stream),
                input_hidden,
                terminal_hidden_archive,
                packed_hidden_out,
                input_token_ids,
                input_position_ids,
                shifted_token_ids_out,
                shifted_position_ids_out,
                append_lengths_out,
                main_cached_tokens,
                shifted_cached_tokens,
                request_sequence_lengths,
                request_row_stride_device,
                request_index,
                request_count,
                captured_row_stride,
                seq_capacity,
                d_model);
        };
        if (rows_are_float4_aligned)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
        const hipError_t launch_status = hipGetLastError();
        if (launch_status != hipSuccess)
        {
            LOG_ERROR("[ROCmRowSelectKernels] Shifted-MTP prefill preparation launch failed: "
                      << hipGetErrorString(launch_status)
                      << " request=" << request_index
                      << " stream=" << stream);
            return false;
        }
        return true;
    }

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
        void *stream)
    {
        if (!request_token_ids || !request_position_ids ||
            !request_total_rows || !cached_tokens ||
            request_row_capacity <= 0 || bucket_seq_len <= 0 ||
            bucket_seq_len > request_row_capacity || !out_token_ids ||
            !out_position_ids || !out_real_rows || !out_row_stride || !stream)
        {
            LOG_ERROR("[ROCmRowSelectKernels] Prefill chunk materialization received an incomplete launch contract");
            return false;
        }

        /*
         * A 4096-row bucket contains only 1024 int4 groups. On gfx906, a
         * conventional 256-thread block concentrates those groups into four
         * workgroups and four CUs. Pooled rocprof A/B samples across
         * 16/32/64/128/256-thread variants place the optimum at 32 active lanes
         * per wave: 32 independent workgroups expose enough CUs to hide this
         * tiny prelude's launch and memory latency, while 16 threads lose more
         * to inactive wave64 lanes than they gain from additional distribution.
         */
        constexpr int threads_per_block = 32;
        const int vector_count = (bucket_seq_len + 3) / 4;
        const int blocks = std::max(
            1,
            std::min(128, (vector_count + threads_per_block - 1) /
                              threads_per_block));
        hipLaunchKernelGGL(
            preparePrefillChunkViewKernel,
            dim3(blocks),
            dim3(threads_per_block),
            0,
            reinterpret_cast<hipStream_t>(stream),
            request_token_ids,
            request_position_ids,
            request_total_rows,
            cached_tokens,
            request_row_capacity,
            bucket_seq_len,
            pad_token_id,
            out_token_ids,
            out_position_ids,
            out_real_rows,
            out_row_stride);
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess)
        {
            LOG_ERROR("[ROCmRowSelectKernels] Prefill chunk materialization launch failed: "
                      << hipGetErrorString(status));
            return false;
        }
        return true;
    }

} // namespace llaminar2::rocm

#endif // HAVE_ROCM
