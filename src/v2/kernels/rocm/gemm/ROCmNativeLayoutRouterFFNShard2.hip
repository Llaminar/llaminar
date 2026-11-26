/**
 * @file ROCmNativeLayoutRouterFFNShard2.hip
 * @brief Native-layout router FFN shard 2 (groups 11..15).
 */
#include "ROCmNativeQuantisedMoEDeviceHelpers.hip.inc"
#include "ROCmNativeLayoutRouterFFNBody.hip.inc"

LLAMINAR_DEFINE_ROCM_IQ_GRID_INIT(native_layout_2)

namespace llaminar2::rocm {
const void* nativeLayoutRouterFFNKernelFunctionShard2(ResidentMoEFFNDecoderProgram program)
{
#define LLAMINAR_LAYOUT_FAMILY(G) case G:return reinterpret_cast<const void*>(&nativeLayoutRouterFFNKernel<ResidentMoEFFNDecoderProgram::familyMask<G>()>);
    switch(program.groups()) { LLAMINAR_LAYOUT_FAMILY(11) LLAMINAR_LAYOUT_FAMILY(12) LLAMINAR_LAYOUT_FAMILY(13) LLAMINAR_LAYOUT_FAMILY(14) LLAMINAR_LAYOUT_FAMILY(15) }
#undef LLAMINAR_LAYOUT_FAMILY
    throw std::invalid_argument("Native-layout FFN source envelope is invalid");
}
}
