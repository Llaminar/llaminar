/**
 * @file CUDADeepSeekV4Kernels.cu
 * @brief Native CUDA compilation boundary for the shared DeepSeek-V4 primitive kernels.
 *
 * The shared implementation lives in kernels/deepseek4/GPUDeepSeekV4Kernels.cuh
 * so a future HIP translation unit compiles the same arithmetic.
 */
#define DSV4_CUDA 1
#include "../../deepseek4/GPUDeepSeekV4Kernels.cuh"
