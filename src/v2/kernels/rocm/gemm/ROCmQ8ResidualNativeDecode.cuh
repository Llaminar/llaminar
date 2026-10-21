/**
 * @file ROCmQ8ResidualNativeDecode.cuh
 * @brief HIP adapter for the canonical native decoder using borrowed original grids.
 *
 * Grid pointers are fixed cold prepared-engine operands, not new device tables.
 * Native packed arithmetic preserves CUDA byte semantics with AMD instructions;
 * the included decoder and contribution body is the same production authority.
 */
#pragma once
#include "kernels/common/DeviceNativeVNNIContributionContract.h"
#include "kernels/common/GPUQ8ResidualNative.h"
#include "kernels/common/NativeVNNIPackedBits.h"
#include "tensors/NativeVnniIQ2SScale.h"
#include "tensors/NativeVnniIQ3SScale.h"
#include "tensors/NativeVnniIQ4XSScale.h"
#include "tensors/NativeVnniQ6Scale.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
namespace llaminar2::rocm_q8_residual
{
    /** @brief Per-launch immutable view; no copied grid storage or new owner. */
    struct NativeDecoder
    {
        const std::uint32_t *d_iq3s_grid, *d_iq3xxs_grid;
        const std::uint64_t *d_iq2s_grid, *d_iq2xs_grid, *d_iq2xxs_grid, *d_iq1s_grid;
        /** @brief Bind the exact cold prepared-engine source-table pointers. */
        __device__ explicit NativeDecoder(NativeEmbeddingIQTables t)
            : d_iq3s_grid(t.iq3s), d_iq3xxs_grid(t.iq3xxs), d_iq2s_grid(t.iq2s), d_iq2xs_grid(t.iq2xs),
              d_iq2xxs_grid(t.iq2xxs), d_iq1s_grid(t.iq1s)
        {
        }
        /** @brief Subtract four independent bytes without cross-byte borrow. */
        __device__ __forceinline__ static std::uint32_t __vsub4(std::uint32_t a, std::uint32_t b)
        {
            const auto low = ((a & 0x7f7f7f7fu) | 0x80808080u) - (b & 0x7f7f7f7fu);
            return low ^ ((a ^ ~b) & 0x80808080u);
        }
        /** @brief Map CUDA's byte selector to AMD's native source order. */
        __device__ __forceinline__ static std::uint32_t __byte_perm(std::uint32_t a, std::uint32_t b,
                                                                    std::uint32_t s)
        {
            const auto select = (s & 7u) | ((s & 0x70u) << 4) | ((s & 0x700u) << 8) | ((s & 0x7000u) << 12);
            return __builtin_amdgcn_perm(b, a, select);
        }
#include "kernels/common/DeviceNativeVNNIDecodeBody.cuh"
    };
} // namespace llaminar2::rocm_q8_residual
