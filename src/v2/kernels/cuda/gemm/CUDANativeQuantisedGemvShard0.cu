/**
 * @file CUDANativeQuantisedGemvShard0.cu
 * @brief CUDA NativeQuantised GEMV instantiations for Q4_0, IQ4_NL and original-factor IQ4_XS.
 * IQ4_XS physical format 24 preserves the original half base and raw six-bit
 * coefficient in separate 16-bit planes. Its LUT payload and source4 reduction
 * schedule are unchanged; the coefficient is storage metadata, never a minimum.
 */
#define LLAMINAR_CUDA_NATIVE_QUANTISED_GEMV_SHARD 0
#define LLAMINAR_CUDA_NATIVE_QUANTISED_GEMV_SHARD_SYMBOL_IMPL(NAME) NAME##_shard_0
#define LLAMINAR_CUDA_NATIVE_QUANTISED_GEMV_SHARD_SYMBOL(NAME) \
    LLAMINAR_CUDA_NATIVE_QUANTISED_GEMV_SHARD_SYMBOL_IMPL(NAME)
#define LLAMINAR_CUDA_NATIVE_QUANTISED_GEMV_FOR_EACH_CODEBOOK(MACRO) \
    MACRO(0); MACRO(4); MACRO(llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook)
#include "CUDANativeQuantisedGemvShardImpl.cu.inc"
