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
