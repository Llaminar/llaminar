/**
 * @file ROCmNativeLayoutRouterFFNShard0.hip
 * @brief Native-layout router FFN shard 0 (groups 1..5 and prepare/launch).
 */
#include "ROCmNativeQuantisedMoEDeviceHelpers.hip.inc"
#include "ROCmNativeLayoutRouterFFNBody.hip.inc"
#include "ROCmNativeLayoutRouterFFNLaunch.hip.inc"

LLAMINAR_DEFINE_ROCM_IQ_GRID_INIT(native_layout_0)
