/**
 * @file CUDAImmutableFixedHeadLookupKernels.cu
 * @brief CUDA boundary for original-native fixed-head shard primitives.
 *
 * Shared operations retain the exact CUDA stream and existing source codebooks;
 * collective progress remains with explicit existing native collective stages.
 */
#define QFN_FIXED_HEAD_CUDA 1
#include "../../qwen4exp/GPUImmutableFixedHeadLookupKernels.cuh"
