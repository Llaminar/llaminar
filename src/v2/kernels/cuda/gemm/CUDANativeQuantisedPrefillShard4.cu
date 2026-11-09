/**
 * @file CUDANativeQuantisedPrefillShard4.cu
 * @brief CUDA NativeQuantised prefill instantiations for codebooks 11 and 12.
 */
#define LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD 4
#define LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_SYMBOL(NAME) \
    LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_INDEXED_SYMBOL(NAME, 4)
#include "CUDANativeQuantisedPrefillShardImpl.cu.inc"
