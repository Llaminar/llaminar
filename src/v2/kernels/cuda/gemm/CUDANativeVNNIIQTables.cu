/**
 * @file CUDANativeVNNIIQTables.cu
 * @brief Device storage for immutable CUDA NativeVNNI IQ decode tables.
 *
 * Backend initialization publishes canonical host codebooks into these device
 * arrays before any graph is captured. Kernels subsequently treat the arrays
 * as immutable read-only state; no inference replay performs allocation,
 * transfer, mutation, or host synchronization for table access.
 */

#include <cstdint>

namespace llaminar2::cuda_native_vnni
{
    __device__ __constant__ int8_t d_iq4nl_values[16] = {
        -127, -104, -83, -65,
        -49, -35, -22, -10,
        1, 13, 25, 38,
        53, 69, 89, 113};

    __device__ uint32_t d_iq3s_grid[512];
    __device__ uint32_t d_iq3xxs_grid[256];
    __device__ uint64_t d_iq2s_grid[1024];
    __device__ uint64_t d_iq2xs_grid[512];
    __device__ uint64_t d_iq2xxs_grid[256];
    __device__ uint64_t d_iq1s_grid[2048];
}
