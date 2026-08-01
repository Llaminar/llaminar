/**
 * @file ROCmRowSelectKernels.hip
 * @brief HIP implementation of graph-capturable hidden-state row selection.
 *
 * The graph captures a host-to-device scalar copy followed by a fixed-grid copy
 * kernel. Replaying the same graph executable with a different pinned scalar
 * value selects a different final hidden-state row without changing kernel
 * pointers, grid dimensions, or source pointer offsets.
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
        if (!device_selected_row || !host_selected_row)
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
        if (!input || !output || !device_selected_row || seq_len <= 0 || d_model <= 0)
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
        if (!hidden || !embedding || !output || rows <= 0 || hidden_dim <= 0)
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

} // namespace llaminar2::rocm

#endif // HAVE_ROCM
