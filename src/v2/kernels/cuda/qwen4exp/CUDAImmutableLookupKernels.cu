/**
 * @file CUDAImmutableLookupKernels.cu
 * @brief CUDA compilation boundary for exact retained immutable-row I/O.
 *
 * Shared protocol kernels use native system-scope publication and fresh mapped
 * payload loads while preserving the backend's existing source decoder.
 * The direct resident-table companion bypasses that service protocol entirely
 * and decodes immutable original VRAM rows on the same explicit stream ABI.
 */
#define QFN_LOOKUP_CUDA 1
#include "../../qwen4exp/GPUImmutableLookupKernels.cuh"

#include "../../qwen4exp/GPUImmutableResidentLookupKernels.cuh"
