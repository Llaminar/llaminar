/** @file ROCmImmutableAdaptiveLookupKernels.hip
 * @brief HIP compilation boundary for the checked single-GPU adaptive native component.
 * Original codebooks, system-scope protocol ordering and shared integer policy
 * remain identical to the CUDA peer; no serving capability is advertised here.
 */
#include "../../qwen4exp/GPUImmutableAdaptiveLookupKernels.cuh"
