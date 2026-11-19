/**
 * @file CUDANativeQuantisedPrefillShard7.cu
 * @brief CUDA NativeQuantised prefill instantiations for codebooks 17, 19 and 23 (expanded INT8 minimum).
 */
#define LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD 7
#define LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_SYMBOL(NAME) \
    LLAMINAR_CUDA_NATIVE_QUANTISED_PREFILL_SHARD_INDEXED_SYMBOL(NAME, 7)
#include "CUDANativeQuantisedPrefillShardImpl.cu.inc"
