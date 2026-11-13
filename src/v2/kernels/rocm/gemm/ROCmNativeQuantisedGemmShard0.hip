/**
 * @file ROCmNativeQuantisedGemmShard0.hip
 * @brief NativeQuantised GEMM instantiations for Q4_0, IQ4_NL and source-tuple IQ4_XS.
 * All three formats keep the existing integer-dot and source-policy trees.
 * Physical24 loads an original half base and encoded6 word before the same
 * single-scale contribution; the shared include owns every instantiation.
 */
#define LLAMINAR_ROCM_NATIVE_QUANTISED_SHARD 0
#include "ROCmNativeQuantisedGemmShardImpl.hip.inc"
