/**
 * @file CUDAQwen4ExpKernels.cu
 * @brief Native CUDA compilation boundary for shared capture-safe GR/PLE kernels.
 *
 * The shared implementation fixes arithmetic partitions across backends and
 * keeps host submission independent of GPU-owned live/acceptance state.
 */
#define QFN_CUDA 1
#include "../../qwen4exp/GPUQwen4ExpKernels.cuh"
