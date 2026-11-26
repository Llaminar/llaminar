/**
 * @file ROCmFullNativeHCFFNShard0.hip
 * @brief Full native HC/FFN shard 0 (groups 1..5 and prepare/launch).
 */
#include "ROCmNativeQuantisedMoEDeviceHelpers.hip.inc"
#include "ROCmPersistentHCReadBody.hip.inc"
#include "ROCmNativeLayoutRouterFFNBody.hip.inc"
#include "ROCmFullNativeHCFFNBody.hip.inc"
#include "ROCmFullNativeHCFFNLaunch.hip.inc"

LLAMINAR_DEFINE_ROCM_IQ_GRID_INIT(full_hc_0)
