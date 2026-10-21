/**
 * @file ROCmQwen4ExpKernels.hip
 * @brief Native HIP compilation boundary for shared capture-safe GR/PLE kernels.
 *
 * A canonical 256-lane tree preserves grouped-row byte equivalence on wave32
 * and wave64 hardware without atomics, mutable host state or row replay.
 */
#define QFN_ROCM 1
#include "../../qwen4exp/GPUQwen4ExpKernels.cuh"
