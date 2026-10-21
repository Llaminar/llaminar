/** @file CUDAImmutableAdaptiveLookupKernels.cu
 * @brief CUDA compilation boundary for the checked single-GPU adaptive native component.
 * Original codebooks, system-scope protocol ordering and shared integer policy
 * remain identical to the HIP peer; no serving capability is advertised here.
 */
#define QFN_ADAPTIVE_CUDA 1
#include "../../qwen4exp/GPUImmutableAdaptiveLookupKernels.cuh"
