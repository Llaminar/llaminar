/**
 * @file ROCmFullNativeHCFFNShard2.hip
 * @brief Full native HC/FFN shard 2 (groups 11..15).
 */
#include "ROCmNativeQuantisedMoEDeviceHelpers.hip.inc"
#include "ROCmPersistentHCReadBody.hip.inc"
#include "ROCmNativeLayoutRouterFFNBody.hip.inc"
#include "ROCmFullNativeHCFFNBody.hip.inc"

LLAMINAR_DEFINE_ROCM_IQ_GRID_INIT(full_hc_2)

namespace llaminar2::rocm {
const void* fullNativeHCFFNKernelFunctionShard2(ResidentMoEFFNDecoderProgram program)
{
#define LLAMINAR_FULL_HC_FAMILY(G) case G:return reinterpret_cast<const void*>(&fullNativeHCFFNKernel<ResidentMoEFFNDecoderProgram::familyMask<G>()>);
    switch(program.groups()) { LLAMINAR_FULL_HC_FAMILY(11) LLAMINAR_FULL_HC_FAMILY(12) LLAMINAR_FULL_HC_FAMILY(13) LLAMINAR_FULL_HC_FAMILY(14) LLAMINAR_FULL_HC_FAMILY(15) }
#undef LLAMINAR_FULL_HC_FAMILY
    throw std::invalid_argument("Full native HC/FFN source envelope is invalid");
}
}
