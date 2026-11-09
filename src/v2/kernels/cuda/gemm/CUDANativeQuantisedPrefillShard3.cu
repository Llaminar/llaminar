/**
 * @file CUDANativeQuantisedPrefillShard3.cu
 * @brief CUDA NativeQuantised prefill instantiations for codebooks 9 and 10.
 */
#define LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD 3
#define LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_SYMBOL(NAME) \
    LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_INDEXED_SYMBOL(NAME, 3)
#include "CUDANativeQuantisedPrefillShardImpl.cu.inc"
