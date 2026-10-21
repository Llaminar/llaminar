/**
 * @file CUDANativeVNNIDecodeCommon.cuh
 * @brief Shared byte-exact CUDA NativeVNNI decode and contribution primitives.
 *
 * CUDA GEMV, grouped verifier, and prefill kernels include this header so that
 * every execution regime decodes quantized weights and publishes FP32 block
 * contributions with the same arithmetic order. IQ lookup tables are copied
 * once during backend initialization and remain immutable device-owned state
 * throughout graph capture and replay. Q6_K metadata preserves the original
 * half base and signed coefficients; IQ2_S retains its half base and scale
 * nibble byte. Every execution family reconstructs these source scales before
 * applying its established contribution tree.
 * IQ4_XS physical format 24 preserves the original half base and raw six-bit
 * coefficient in separate 16-bit planes. Its LUT payload and source4 reduction
 * schedule are unchanged; the coefficient is storage metadata, never a minimum.
 * IQ3_S physical11 similarly widens its original base and multiplies the raw
 * source odd coefficient in FP32 before the unchanged single-dot contribution.
 */

#pragma once

#include "tensors/IQQuantTables.h"
#include "tensors/NativeVnniFormatInfo.h"
#include "tensors/NativeVnniQ6Scale.h"
#include "tensors/NativeVnniIQ2SScale.h"
#include "tensors/NativeVnniIQ4XSScale.h"
#include "tensors/NativeVnniIQ3SScale.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/common/DeviceNativeVNNIContributionContract.h"
#include "kernels/common/NativeVNNIPackedBits.h"

#include <cstdint>
#include <iterator>

namespace llaminar2::cuda_native_vnni
{
    extern __device__ __constant__ int8_t d_iq4nl_values[16];

    extern __device__ uint32_t d_iq3s_grid[512];
    extern __device__ uint32_t d_iq3xxs_grid[256];
    extern __device__ uint64_t d_iq2s_grid[1024];
    extern __device__ uint64_t d_iq2xs_grid[512];
    extern __device__ uint64_t d_iq2xxs_grid[256];
    extern __device__ uint64_t d_iq1s_grid[2048];

    template <typename T, size_t N>
    inline bool copyHostArrayToDeviceSymbol(T (&symbol)[N], const T *src, size_t count)
    {
        void *dst = nullptr;
        size_t bytes = 0;
        cudaError_t err = cudaGetSymbolAddress(&dst, symbol);
        if (err != cudaSuccess)
            return false;
        err = cudaGetSymbolSize(&bytes, symbol);
        if (err != cudaSuccess || bytes < count * sizeof(T))
            return false;
        err = cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice);
        return err == cudaSuccess;
    }

    inline bool initIQGridTables()
    {
        if (!copyHostArrayToDeviceSymbol(d_iq3s_grid, iq3s_grid, std::size(iq3s_grid)))
            return false;
        if (!copyHostArrayToDeviceSymbol(d_iq3xxs_grid, iq3xxs_grid, std::size(iq3xxs_grid)))
            return false;
        if (!copyHostArrayToDeviceSymbol(d_iq2s_grid, iq2s_grid, std::size(iq2s_grid)))
            return false;

        if (!copyHostArrayToDeviceSymbol(d_iq2xs_grid, iq2xs_grid, std::size(iq2xs_grid)))
            return false;
        if (!copyHostArrayToDeviceSymbol(d_iq2xxs_grid, iq2xxs_grid, std::size(iq2xxs_grid)))
            return false;
        if (!copyHostArrayToDeviceSymbol(d_iq1s_grid, iq1s_grid, std::size(iq1s_grid)))
            return false;
        return true;
    }

#include "kernels/common/DeviceNativeVNNIDecodeBody.cuh"

}
