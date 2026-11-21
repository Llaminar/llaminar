/**
 * @file probe.hip
 * @brief Compile-only resource witness for HIP device-conditional dependencies.
 *
 * The host preprocessor never sees the dimensions header. Its changed value
 * must still invalidate Ninja and ccache and alter every target's actual LDS
 * allocation. No accelerator is required and this kernel is never launched.
 */
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include <device_dimensions.inc>
#endif

/**
 * @brief Retain the full device-only array in final native metadata.
 * @param output Borrowed diagnostic result, unused by the compile-only replay.
 * @param index Runtime index preventing scalar replacement of the LDS array.
 */
extern "C" __global__ void device_input_probe(int *output, unsigned index)
{
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ volatile int values[kDeviceWords];
    values[threadIdx.x] = static_cast<int>(threadIdx.x);
    __syncthreads();
    output[threadIdx.x] = values[index % kDeviceWords];
#endif
}
