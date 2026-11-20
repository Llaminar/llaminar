/**
 * @file CUDANativeQuantisedPrefillShard0.cu
 * @brief CUDA NativeQuantised prefill instantiations for codebooks 0, 4 and 24 (IQ4_XS original factors).
 */
#define LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD 0
#define LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_SYMBOL(NAME) \
    LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_INDEXED_SYMBOL(NAME, 0)
#include "CUDANativeQuantisedPrefillShardImpl.cu.inc"
