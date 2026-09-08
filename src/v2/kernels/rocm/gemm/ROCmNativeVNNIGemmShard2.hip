/**
 * @file ROCmNativeVNNIGemmShard2.hip
 * @brief NativeVNNI GEMM instantiations for Q5_1 and Q6_K.
 */
#define LLAMINAR_ROCM_NATIVE_VNNI_SHARD 2
#include "ROCmNativeVNNIGemmShardImpl.hip.inc"
