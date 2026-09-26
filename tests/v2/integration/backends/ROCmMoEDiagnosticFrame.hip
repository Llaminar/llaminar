/**
 * @file ROCmMoEDiagnosticFrame.hip
 * @brief Measure the compiler's fatal-diagnostic call frame without launching it.
 *
 * ROCm's symbol-enabled build can retain an OCKL printf call frame even for a
 * spill-free kernel. This minimal reference separates that compiler-owned ABI
 * requirement from register spills without a hard-coded byte allowance. It is
 * compiled with the same build-type/debug flags as the production HIP kernels.
 */
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdio>

namespace
{
    /** @brief Keep a diagnostic reachable without adding arithmetic or arrays. */
    __global__ __launch_bounds__(256) void diagnosticFrameReference(int *value)
    {
        if (!value)
        {
            printf("MoE resource test invalid pointer: %p\n", value);
            __builtin_trap();
        }
        value[threadIdx.x] = threadIdx.x;
    }
}

/**
 * @brief Query the compiler-owned diagnostic frame on the current HIP device.
 * @param bytes Receives the compiled per-thread private allocation.
 * @return Whether the query succeeded; never launches the reference kernel.
 */
bool rocmMoEDiagnosticFrameBytes(std::size_t &bytes)
{
    hipFuncAttributes attributes{};
    if (hipFuncGetAttributes(&attributes,
            reinterpret_cast<const void *>(diagnosticFrameReference)) != hipSuccess)
        return false;
    bytes = attributes.localSizeBytes;
    return true;
}
