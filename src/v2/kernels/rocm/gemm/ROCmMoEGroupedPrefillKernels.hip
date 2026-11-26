/**
 * @file ROCmMoEGroupedPrefillKernels.hip
 * @brief Coordinator and non-template kernels for sharded ROCm grouped MoE prefill.
 *
 * Per-codebook prefill kernels are physically sharded across six companion TUs
 * (ROCmMoEGroupedPrefillShard0..5.hip, three codebooks each) so LLVM compiles
 * all 18 quantization formats concurrently.
 */
#define LLAMINAR_ROCM_MOE_PREFILL_SHARD -1
#include "ROCmMoEGroupedPrefillShardImpl.hip.inc"
