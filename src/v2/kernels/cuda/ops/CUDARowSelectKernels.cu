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

    bool allocateRowSelectParam(
        int device_ordinal,
        int **host_selected_row,
        int **device_selected_row)
    {
        if (!host_selected_row || !device_selected_row)
            return false;

        *host_selected_row = nullptr;
        *device_selected_row = nullptr;

        if (!allocateRowSelectHostParam(device_ordinal, host_selected_row))
            return false;
        if (!ok(cudaMalloc(reinterpret_cast<void **>(device_selected_row), sizeof(int))))
        {
            freeRowSelectHostParam(device_ordinal, *host_selected_row);
            *host_selected_row = nullptr;
            return false;
        }
        return true;
    }

    void freeRowSelectParam(
        int device_ordinal,
        int *host_selected_row,
        int *device_selected_row)
    {
        cudaSetDevice(device_ordinal);
        if (device_selected_row)
            cudaFree(device_selected_row);
        freeRowSelectHostParam(device_ordinal, host_selected_row);
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
        return ok(cudaGetLastError());
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
