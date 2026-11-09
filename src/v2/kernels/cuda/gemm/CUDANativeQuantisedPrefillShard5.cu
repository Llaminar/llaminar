/**
 * @file CUDANativeQuantisedPrefillShard5.cu
 * @brief CUDA NativeQuantised prefill instantiations for codebooks 13 and 14.
 */
#define LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD 5
#define LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_SYMBOL(NAME) \
    LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_INDEXED_SYMBOL(NAME, 5)
#include "CUDANativeQuantisedPrefillShardImpl.cu.inc"
