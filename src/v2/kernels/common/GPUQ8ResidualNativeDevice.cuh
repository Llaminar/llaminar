/**
 * @file GPUQ8ResidualNativeDevice.cuh
 * @brief Exact paired native contribution adapter shared by vector and cooperative tiles.
 *
 * Decoder reuse preserves original source factor/minimum/sign semantics. Per-row
 * activation subgroup sums are integer metadata computed from admitted byte banks;
 * no residual tensor, global partial output, or extra persistent bank is formed.
 */
#pragma once
#if defined(Q8_RESIDUAL_CUDA)
#include "kernels/cuda/gemm/CUDANativeVNNIDecodeCommon.cuh"
namespace llaminar2::cuda_q8_residual
{
    /** @brief Compile-time adapter to the unchanged installed CUDA native primitives. */
    struct NativeDecoder
    {
        const std::uint32_t *d_iq3s_grid, *d_iq3xxs_grid;
        const std::uint64_t *d_iq2s_grid, *d_iq2xs_grid, *d_iq2xxs_grid, *d_iq1s_grid;
        /** @brief Retain the exact original cold codebook pointer operands. */
        __device__ explicit NativeDecoder(NativeEmbeddingIQTables tables)
            : d_iq3s_grid(tables.iq3s), d_iq3xxs_grid(tables.iq3xxs),
              d_iq2s_grid(tables.iq2s), d_iq2xs_grid(tables.iq2xs),
              d_iq2xxs_grid(tables.iq2xxs), d_iq1s_grid(tables.iq1s)
        {
        }
#include "kernels/common/DeviceNativeVNNIDecodeBody.cuh"
    };
} // namespace llaminar2::cuda_q8_residual
#else
#include "kernels/rocm/gemm/ROCmQ8ResidualNativeDecode.cuh"
#endif
namespace llaminar2::Q8R_NAMESPACE
{
    /** @brief Bind only original metadata consumed by this compile-time execution format. */
    template <std::uint8_t C> struct NativeBlock
    {
        using Traits = NativeDecoder::CodebookTraits<C>;
        std::uint16_t scale, secondary, qh;
        std::uint32_t emin;
        /** @brief Explicit inactive local tile metadata; no global source access. */
        __device__ NativeBlock() : scale(0), secondary(0), qh(0), emin(0) {}
        /** @brief Load exact original scale/raw-factor planes and IQ1_M delta signs. */
        __device__ NativeBlock(const DeviceNativeVNNIMatrixDesc &w, std::uint64_t block,
                               const std::uint8_t *payload)
            : scale(static_cast<const std::uint16_t *>(w.scales)[block]), secondary(0), qh(0), emin(0)
        {
            if constexpr (Traits::requires_secondary_metadata)
                secondary = static_cast<const std::uint16_t *>(w.mins)[block];
            if constexpr (Traits::is_dual_scale_asym)
                emin = static_cast<const std::uint32_t *>(w.emins)[block];
            if constexpr (Traits::is_iq1_m)
                qh = static_cast<std::uint16_t>(payload[4]) | (static_cast<std::uint16_t>(payload[5]) << 8);
        }
    };
    /** @brief Publish one full32 contribution, with exact low/high/subgroup corrections. */
    template <std::uint8_t C>
    __device__ __forceinline__ float nativeContribution(NativeDecoder &decoder, const NativeBlock<C> &b,
                                                        int lo, int hi, float activation, const int (&s)[4])
    {
        const int low = s[0] + s[1], high = s[2] + s[3];
        return decoder.native_vnni_block_contribution_from_reduced_terms_rn<C>(
            lo, hi, activation, b.scale, b.secondary, b.emin, low + high, low, high, b.qh, s[0], s[1], s[2], s[3]);
    }
} // namespace llaminar2::Q8R_NAMESPACE
