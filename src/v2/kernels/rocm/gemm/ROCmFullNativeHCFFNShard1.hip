/**
 * @file ROCmFullNativeHCFFNShard1.hip
 * @brief Full native HC/FFN shard 1 (groups 6..10).
 */
#include "ROCmNativeQuantisedMoEDeviceHelpers.hip.inc"
#include "ROCmPersistentHCReadBody.hip.inc"
#include "ROCmNativeLayoutRouterFFNBody.hip.inc"
#include "ROCmFullNativeHCFFNBody.hip.inc"

LLAMINAR_DEFINE_ROCM_IQ_GRID_INIT(full_hc_1)

namespace llaminar2::rocm {
const void* fullNativeHCFFNKernelFunctionShard1(ResidentMoEFFNDecoderProgram program)
{
#define LLAMINAR_FULL_HC_FAMILY(G) case G:return reinterpret_cast<const void*>(&fullNativeHCFFNKernel<ResidentMoEFFNDecoderProgram::familyMask<G>()>);
    switch(program.groups()) { LLAMINAR_FULL_HC_FAMILY(6) LLAMINAR_FULL_HC_FAMILY(7) LLAMINAR_FULL_HC_FAMILY(8) LLAMINAR_FULL_HC_FAMILY(9) LLAMINAR_FULL_HC_FAMILY(10) }
#undef LLAMINAR_FULL_HC_FAMILY
    throw std::invalid_argument("Full native HC/FFN source envelope is invalid");
}
}
