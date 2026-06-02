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

    void freeRowSelectHostParam(
        int device_ordinal,
        int *host_selected_row)
    {
        hipSetDevice(device_ordinal);
        if (host_selected_row)
            hipHostFree(host_selected_row);
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
        if (!ok(hipMalloc(reinterpret_cast<void **>(device_selected_row), sizeof(int))))
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
        hipSetDevice(device_ordinal);
        if (device_selected_row)
            hipFree(device_selected_row);
        freeRowSelectHostParam(device_ordinal, host_selected_row);
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
