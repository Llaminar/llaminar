/**
 * @file CUDANativeQuantisedPrefillShard6.cu
 * @brief CUDA NativeQuantised prefill instantiations for codebooks 15 and 16.
 */
#define LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD 6
#define LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_SYMBOL(NAME) \
    LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_INDEXED_SYMBOL(NAME, 6)
#include "CUDANativeQuantisedPrefillShardImpl.cu.inc"
