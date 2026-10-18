/**
 * @file CUDANativeVNNIIQTables.cu
 * @brief Device storage for immutable CUDA NativeVNNI IQ decode tables.
 *
 * Backend initialization publishes canonical host codebooks into these device
 * arrays before any graph is captured. Kernels subsequently treat the arrays
 * as immutable read-only state; no inference replay performs allocation,
 * transfer, mutation, or host synchronization for table access.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include "kernels/common/NativeEmbeddingIQTables.h"

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

/**
 * @brief Borrow existing backend grid addresses after their generation owner published them.
 * @param ordinal Exact device selected during source-weight preparation.
 * @param tables Destination non-owning immutable view; no cache or allocation is created.
 * @return True when all six existing source codebooks have valid device addresses.
 */
extern "C" bool cudaNativeEmbeddingIQTableView(
    int ordinal, llaminar2::NativeEmbeddingIQTables *tables)
{
    if (!tables || ordinal < 0) return false;
    *tables = {};
    if (cudaSetDevice(ordinal) != cudaSuccess) return false;
    void *address = nullptr;
    if (cudaGetSymbolAddress(&address, llaminar2::cuda_native_vnni::d_iq3s_grid) != cudaSuccess) return false;
    tables->iq3s = static_cast<const uint32_t *>(address);
    if (cudaGetSymbolAddress(&address, llaminar2::cuda_native_vnni::d_iq3xxs_grid) != cudaSuccess) return false;
    tables->iq3xxs = static_cast<const uint32_t *>(address);
    if (cudaGetSymbolAddress(&address, llaminar2::cuda_native_vnni::d_iq2s_grid) != cudaSuccess) return false;
    tables->iq2s = static_cast<const uint64_t *>(address);
    if (cudaGetSymbolAddress(&address, llaminar2::cuda_native_vnni::d_iq2xs_grid) != cudaSuccess) return false;
    tables->iq2xs = static_cast<const uint64_t *>(address);
    if (cudaGetSymbolAddress(&address, llaminar2::cuda_native_vnni::d_iq2xxs_grid) != cudaSuccess) return false;
    tables->iq2xxs = static_cast<const uint64_t *>(address);
    if (cudaGetSymbolAddress(&address, llaminar2::cuda_native_vnni::d_iq1s_grid) != cudaSuccess) return false;
    tables->iq1s = static_cast<const uint64_t *>(address);
    return true;
}
