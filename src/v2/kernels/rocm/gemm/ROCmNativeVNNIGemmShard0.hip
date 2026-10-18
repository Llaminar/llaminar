/**
 * @file ROCmNativeVNNIGemmShard0.hip
 * @brief NativeVNNI GEMM instantiations for Q4_0, IQ4_NL and source-tuple IQ4_XS.
 * All three formats keep the existing integer-dot and source-policy trees.
 * Physical24 loads an original half base and encoded6 word before the same
 * single-scale contribution; the shared include owns every instantiation.
 */
#define LLAMINAR_ROCM_NATIVE_VNNI_SHARD 0
#include "ROCmNativeVNNIGemmShardImpl.hip.inc"
