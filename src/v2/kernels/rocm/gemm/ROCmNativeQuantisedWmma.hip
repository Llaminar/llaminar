/**
 * @file ROCmNativeQuantisedWmma.hip
 * @brief Native wave64 compilation and typed submission for gfx1100 INT8 WMMA.
 *
 * This file uses the backend's native ordered wave64 contract. Both shipping
 * architectures are still compiled and spill-audited; gfx906 emits no WMMA
 * entry. Shared compact decoders and exact FP32 contribution helpers prevent
 * the wave policy from becoming a second quantization or arithmetic contract.
 */
#include "ROCmNativeQuantisedWmmaLaunch.h"
#include "ROCmNativeQuantisedDecodeEquivalentMath.h"
#include "kernels/rocm/common/ROCmSignedDot4.h"
#include <cstdio>
#include <stdexcept>
#include <type_traits>
#include "ROCmNativeQuantisedGemmDecode.hip.inc"
#include "ROCmNativeQuantisedWmma.hip.inc"

namespace llaminar2::rocm
{
VNNIPrefillLaunch submitNativeQuantisedWmma(
    NativeQuantisedWmmaTile tile, const NativeQuantisedWmmaArguments &args, void *stream)
{
    const auto hip_stream = static_cast<hipStream_t>(stream);
    if (!hip_stream || hip_stream == hipStreamLegacy || hip_stream == hipStreamPerThread ||
        !args.activations || !args.payload || !args.scales || !args.output ||
        (!args.row_scales && !args.block_scales) || args.m <= 0 || args.n <= 0 ||
        args.k <= 0 || args.k % 32 || args.serial_partitions <= 0 ||
        args.serial_partitions > args.k / 32)
        throw std::invalid_argument("incomplete native WMMA operands, stream or serial policy");
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1100__)
    auto format = [&]<NativeQuantisedGemmFormat FMT>() {
        using Traits = NATIVE_QUANTISEDGemmTraits<FMT>;
        if ((Traits::needs_aux_compute && !args.auxiliary) ||
            (Traits::is_dual_scale_asym && !args.minima_pairs))
            throw std::invalid_argument("native WMMA source correction metadata is missing");
        auto scale_mode = [&]<NativeWmmaActivationScale ScaleMode>() {
            auto submit = [&]<int MT, int NT>() {
                const void *function = reinterpret_cast<const void *>(
                    HIP_KERNEL_NAME(native_quantised_wmma_gemm<FMT, MT, NT, ScaleMode>));
                native_quantised_wmma_gemm<FMT, MT, NT, ScaleMode><<<
                    dim3((args.n + NT - 1) / NT, (args.m + MT - 1) / MT), 256, 0, hip_stream>>>(
                    args.activations, args.payload, args.scales, args.auxiliary,
                    args.minima_pairs, args.output, args.row_scales, args.block_scales,
                    args.m, args.n, args.k, args.serial_partitions);
                return VNNIPrefillLaunch{.producer = VNNIPrefillProducer::NativeWmma,
                    .execution_codebook = args.codebook, .n_tile = NT, .m_tile = MT,
                    .min_blocks = 1, .unroll = 0, .full_tiles = false,
                    .function = function, .block_threads = 256, .dynamic_shared_memory_bytes = 0};
            };
            switch (tile)
            {
            case NativeQuantisedWmmaTile::M32N32: return submit.template operator()<32, 32>();
            case NativeQuantisedWmmaTile::M32N64: return submit.template operator()<32, 64>();
            case NativeQuantisedWmmaTile::M64N64: return submit.template operator()<64, 64>();
            case NativeQuantisedWmmaTile::M64N128: return submit.template operator()<64, 128>();
            }
            throw std::invalid_argument("unknown native WMMA tile");
        };
        return args.block_scales
            ? scale_mode.template operator()<NativeWmmaActivationScale::Blockwise>()
            : scale_mode.template operator()<NativeWmmaActivationScale::WholeRow>();
    };
    switch (args.codebook)
    {
#define LLAMINAR_SUBMIT_WMMA(FMT) case FMT: return format.template operator()<FMT>()
    LLAMINAR_SUBMIT_WMMA(NATIVE_QUANTISED_GEMM_Q4_0);
    LLAMINAR_SUBMIT_WMMA(NATIVE_QUANTISED_GEMM_IQ4_NL);
    LLAMINAR_SUBMIT_WMMA(NATIVE_QUANTISED_GEMM_IQ4_XS_SOURCE);
    LLAMINAR_SUBMIT_WMMA(NATIVE_QUANTISED_GEMM_Q4_1);
    LLAMINAR_SUBMIT_WMMA(NATIVE_QUANTISED_GEMM_Q5_0);
    LLAMINAR_SUBMIT_WMMA(NATIVE_QUANTISED_GEMM_Q5_1);
    LLAMINAR_SUBMIT_WMMA(NATIVE_QUANTISED_GEMM_Q6_K);
    LLAMINAR_SUBMIT_WMMA(NATIVE_QUANTISED_GEMM_Q3_K);
    LLAMINAR_SUBMIT_WMMA(NATIVE_QUANTISED_GEMM_Q2_K);
    LLAMINAR_SUBMIT_WMMA(NATIVE_QUANTISED_GEMM_IQ3_S);
    LLAMINAR_SUBMIT_WMMA(NATIVE_QUANTISED_GEMM_IQ3_XXS);
    LLAMINAR_SUBMIT_WMMA(NATIVE_QUANTISED_GEMM_IQ2_S);
    LLAMINAR_SUBMIT_WMMA(NATIVE_QUANTISED_GEMM_IQ2_XS);
    LLAMINAR_SUBMIT_WMMA(NATIVE_QUANTISED_GEMM_IQ2_XXS);
    LLAMINAR_SUBMIT_WMMA(NATIVE_QUANTISED_GEMM_IQ1_S);
    LLAMINAR_SUBMIT_WMMA(NATIVE_QUANTISED_GEMM_IQ1_M);
    LLAMINAR_SUBMIT_WMMA(NATIVE_QUANTISED_GEMM_Q8_0);
    LLAMINAR_SUBMIT_WMMA(NATIVE_QUANTISED_GEMM_EXPANDED_INT8_MIN);
#undef LLAMINAR_SUBMIT_WMMA
    }
#endif
    throw std::invalid_argument("unsupported native WMMA physical codebook");
}

bool initializeNativeQuantisedWmmaGrids(int device_id, const void *iq3s,
    const void *iq3xxs, const void *iq2s, const void *iq2xs,
    const void *iq2xxs, const void *iq1s)
{
    return initialize_native_quantised_gemm_grids(
        device_id, iq3s, iq3xxs, iq2s, iq2xs, iq2xxs, iq1s);
}
}
