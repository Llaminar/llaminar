/**
 * @file CUDARowSelectKernels.cu
 * @brief CUDA implementation of graph-capturable hidden-state row selection.
 *
 * The row copy uses a fixed kernel signature and launch shape for a fixed
 * bucket. The selected row is read from a device scalar that is updated by a
 * captured host-to-device copy, allowing one captured graph executable to replay
 * with different real prompt lengths inside the same bucket.
 */

#include "CUDARowSelectKernels.h"
#include "../../../utils/Logger.h"

#ifdef HAVE_CUDA

#include <algorithm>
#include <cuda_runtime.h>

namespace llaminar2::cuda
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
         * A request-batched prefill tensor is flattened as contiguous padded
         * request rows. Every output row therefore has an independent source
         * index even when requests have unequal lengths. Reading lengths in the
         * kernel keeps this derivation ordered with the graph and avoids sharing
         * verifier-row metadata that belongs to a different lifecycle.
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
                 * Admission rejects invalid lengths on the host API boundary.
                 * Clamp again in device code solely to make a malformed launch
                 * memory-safe; valid production requests take the straight path.
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
         * The stride scalar shares the request-admission publication with the
         * length row. One cooperative load per block avoids redundant global
         * traffic while preserving a fixed launch geometry for graph replay.
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
                    /*
                     * Invalid canonical state is poisoned, never clamped into
                     * a plausible row. The downstream exactness gate therefore
                     * fails loudly without granting malformed metadata an OOB
                     * access or silently turning it into a fallback selection.
                     */
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

        /// @brief Return true if a CUDA status is successful.
        bool ok(cudaError_t status)
        {
            return status == cudaSuccess;
        }
    }

    bool allocateRowSelectHostParam(
        int device_ordinal,
        int **host_selected_row)
    {
        if (!host_selected_row)
            return false;

        *host_selected_row = nullptr;

        // Pin the host scalar so the captured cudaMemcpyAsync is legal and
        // replays from a stable address without pageable-memory staging.
        if (!ok(cudaSetDevice(device_ordinal)))
            return false;
        return ok(cudaHostAlloc(reinterpret_cast<void **>(host_selected_row), sizeof(int), cudaHostAllocDefault));
    }

    bool allocateRowSelectHostParams(
        int device_ordinal,
        int **host_selected_rows,
        int row_count)
    {
        if (!host_selected_rows || row_count <= 0)
            return false;

        *host_selected_rows = nullptr;

        if (!ok(cudaSetDevice(device_ordinal)))
            return false;
        return ok(cudaHostAlloc(
            reinterpret_cast<void **>(host_selected_rows),
            static_cast<size_t>(row_count) * sizeof(int),
            cudaHostAllocDefault));
    }

    void freeRowSelectHostParam(
        int device_ordinal,
        int *host_selected_row)
    {
        cudaSetDevice(device_ordinal);
        if (host_selected_row)
            cudaFreeHost(host_selected_row);
    }

    bool uploadRowSelectParam(
        int *device_selected_row,
        const int *host_selected_row,
        void *stream)
    {
        if (!device_selected_row || !host_selected_row)
            return false;
        auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        return ok(cudaMemcpyAsync(
            device_selected_row,
            host_selected_row,
            sizeof(int),
            cudaMemcpyHostToDevice,
            cuda_stream));
    }

    bool uploadRowSelectParams(
        int *device_selected_rows,
        const int *host_selected_rows,
        int row_count,
        void *stream)
    {
        if (!device_selected_rows || !host_selected_rows || row_count <= 0 || !stream)
            return false;
        auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        return ok(cudaMemcpyAsync(
            device_selected_rows,
            host_selected_rows,
            static_cast<size_t>(row_count) * sizeof(int),
            cudaMemcpyHostToDevice,
            cuda_stream));
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
        auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        rowSelectFP32Kernel<<<blocks, threads_per_block, 0, cuda_stream>>>(
            input,
            output,
            device_selected_row,
            seq_len,
            d_model);
        return ok(cudaGetLastError());
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
        return ok(cudaMemcpyAsync(
            output,
            input + source_offset,
            static_cast<size_t>(d_model) * sizeof(float),
            cudaMemcpyDeviceToDevice,
            reinterpret_cast<cudaStream_t>(stream)));
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
        return ok(cudaMemcpyAsync(
            output,
            input + source_offset,
            element_count * sizeof(float),
            cudaMemcpyDeviceToDevice,
            reinterpret_cast<cudaStream_t>(stream)));
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
        auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        rowsSelectFP32Kernel<<<blocks, threads_per_block, 0, cuda_stream>>>(
            input,
            output,
            device_selected_rows,
            seq_len,
            d_model,
            selected_row_count);
        return ok(cudaGetLastError());
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
                "[CUDARowSelectKernels] Request-terminal row selection "
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
        auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        requestTerminalRowsSelectFP32Kernel<<<
            blocks,
            threads_per_block,
            0,
            cuda_stream>>>(
            input,
            output,
            request_sequence_lengths,
            seq_len,
            request_row_stride,
            d_model,
            request_count);
        const cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess)
        {
            LOG_ERROR(
                "[CUDARowSelectKernels] Request-terminal row-selection launch "
                "failed: "
                << cudaGetErrorString(launch_status)
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
                "[CUDARowSelectKernels] Device-geometry request-terminal "
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
        deviceGeometryRequestTerminalRowsSelectFP32Kernel<<<
            blocks,
            threads_per_block,
            0,
            reinterpret_cast<cudaStream_t>(stream)>>>(
            input,
            output,
            request_sequence_lengths,
            request_row_stride,
            seq_capacity,
            d_model,
            request_count);
        const cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess)
        {
            LOG_ERROR(
                "[CUDARowSelectKernels] Device-geometry request-terminal "
                "row-selection launch failed: "
                << cudaGetErrorString(launch_status)
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
            LOG_ERROR("[CUDARowSelectKernels] Device-KV-progress row selection rejected an incomplete launch contract"
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
        deviceKVProgressRowsSelectFP32Kernel<<<
            blocks,
            threads_per_block,
            0,
            reinterpret_cast<cudaStream_t>(stream)>>>(
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
        const cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess)
        {
            LOG_ERROR("[CUDARowSelectKernels] Device-KV-progress row-selection launch failed: "
                      << cudaGetErrorString(launch_status)
                      << " request=" << request_index
                      << " rows=" << selected_row_count
                      << " stream=" << stream);
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
        auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        mtpConcatFP32Kernel<<<blocks, threads_per_block, 0, cuda_stream>>>(
            hidden,
            embedding,
            output,
            rows,
            hidden_dim);
        return ok(cudaGetLastError());
    }

} // namespace llaminar2::cuda

#endif // HAVE_CUDA
