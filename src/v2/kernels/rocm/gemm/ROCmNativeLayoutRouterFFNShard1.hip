/**
 * @file ROCmNativeLayoutRouterFFNShard1.hip
 * @brief Native-layout router FFN shard 1 (groups 6..10).
 */
#include "ROCmNativeQuantisedMoEDeviceHelpers.hip.inc"
#include "ROCmNativeLayoutRouterFFNBody.hip.inc"

LLAMINAR_DEFINE_ROCM_IQ_GRID_INIT(native_layout_1)

namespace llaminar2::rocm {
const void* nativeLayoutRouterFFNKernelFunctionShard1(ResidentMoEFFNDecoderProgram program)
{
#define LLAMINAR_LAYOUT_FAMILY(G) case G:return reinterpret_cast<const void*>(&nativeLayoutRouterFFNKernel<ResidentMoEFFNDecoderProgram::familyMask<G>()>);
    switch(program.groups()) { LLAMINAR_LAYOUT_FAMILY(6) LLAMINAR_LAYOUT_FAMILY(7) LLAMINAR_LAYOUT_FAMILY(8) LLAMINAR_LAYOUT_FAMILY(9) LLAMINAR_LAYOUT_FAMILY(10) }
#undef LLAMINAR_LAYOUT_FAMILY
    throw std::invalid_argument("Native-layout FFN source envelope is invalid");
}
}
