/**
 * @file CUDANativeQuantisedPrefillShard1.cu
 * @brief CUDA NativeQuantised prefill instantiations for codebooks 5 and 6.
 */
#define LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD 1
#define LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_SYMBOL(NAME) \
    LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_INDEXED_SYMBOL(NAME, 1)
#include "CUDANativeQuantisedPrefillShardImpl.cu.inc"
