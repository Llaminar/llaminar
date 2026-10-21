/**
 * @file ROCmImmutableFixedHeadLookupKernels.hip
 * @brief HIP boundary for original-native fixed-head shard primitives.
 *
 * Shared operations retain the exact HIP stream and existing source codebooks;
 * collective progress remains with explicit existing native collective stages.
 */
#define QFN_FIXED_HEAD_ROCM 1
#include "../../qwen4exp/GPUImmutableFixedHeadLookupKernels.cuh"
