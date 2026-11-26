/**
 * @file CUDANativeQuantisedGemvShard2.cu
 * @brief CUDA NativeQuantised GEMV instantiations for IQ4_XS (original source-scale).
 */
#define LLAMINAR_CUDA_NATIVE_QUANTISED_GEMV_SHARD 2
#define LLAMINAR_CUDA_NATIVE_QUANTISED_GEMV_SHARD_SYMBOL_IMPL(NAME) NAME##_shard_2
#define LLAMINAR_CUDA_NATIVE_QUANTISED_GEMV_SHARD_SYMBOL(NAME) \
    LLAMINAR_CUDA_NATIVE_QUANTISED_GEMV_SHARD_SYMBOL_IMPL(NAME)
#define LLAMINAR_CUDA_NATIVE_QUANTISED_GEMV_FOR_EACH_CODEBOOK(MACRO) \
    MACRO(llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook)
#include "CUDANativeQuantisedGemvShardImpl.cu.inc"
