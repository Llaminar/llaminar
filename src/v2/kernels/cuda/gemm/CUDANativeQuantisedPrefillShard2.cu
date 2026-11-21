/**
 * @file CUDANativeQuantisedPrefillShard2.cu
 * @brief CUDA NativeQuantised prefill instantiations for codebooks 7 and 8.
 */
#define LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD 2
#define LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_SYMBOL(NAME) \
    LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_INDEXED_SYMBOL(NAME, 2)
#include "CUDANativeQuantisedPrefillShardImpl.cu.inc"
