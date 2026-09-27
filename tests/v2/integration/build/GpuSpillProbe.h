/**
 * @file GpuSpillProbe.h
 * @brief Compile-only witnesses for the optimized GPU register-spill contract.
 *
 * No probe is launched. The spilling witness keeps 128 input values live across
 * an opaque compiler boundary under a 1024-thread resource bound. The private
 * array witness instead requires deliberate addressable local storage without
 * register pressure, proving that the guard does not ban all scratch memory.
 */
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#if PROBE_KIND == 6
/** @brief Symbols must not manufacture debug-only scratch saves in optimized helpers. */
__device__ __noinline__ float register_only_helper(float value)
{
    return value * 1.25f;
}
/** @brief Exercise a real out-of-line call without allocator pressure. */
__global__ void clean_helper_probe(const float *input, float *output)
{
    output[threadIdx.x] = register_only_helper(input[threadIdx.x]);
}
#elif (PROBE_KIND == 4 || PROBE_KIND == 5) && defined(__HIPCC__)
/** @brief SGPR pressure is legal when the final compiler keeps it in VGPR lanes. */
__global__ __launch_bounds__(256) void scalar_probe(const unsigned *input, unsigned *output)
{
    unsigned values[96];
#pragma unroll
    for (int i = 0; i < 96; ++i)
        values[i] = __builtin_amdgcn_readfirstlane(input[i]);
#if PROBE_KIND == 5
    // This intentional private array must not turn the independent register
    // moves into a false-positive memory-spill failure.
    volatile unsigned private_values[16];
    for (int i = 0; i < 16; ++i)
        private_values[i] = input[i];
#endif
    asm volatile("" ::: "memory");
#pragma unroll
    for (int i = 0; i < 96; ++i)
    {
        // Require a scalar source after the lifetime boundary; without this
        // witness LLVM may legally move every value into ordinary VGPRs early.
        unsigned scalar_value = __builtin_amdgcn_readfirstlane(values[i]);
        asm volatile("" : "+s"(scalar_value));
        output[i * blockDim.x + threadIdx.x] = scalar_value;
    }
#if PROBE_KIND == 5
    output[threadIdx.x] += private_values[threadIdx.x & 15];
#endif
}
#elif PROBE_KIND == 3
/** @brief Out-of-line device spills must not hide behind a clean kernel entry. */
__device__ __noinline__ void spilling_helper(const float *input, float *output)
{
    float values[512];
#pragma unroll
    for (int i = 0; i < 512; ++i)
        values[i] = input[i * 1024 + threadIdx.x];
    asm volatile("" ::: "memory");
#pragma unroll
    for (int i = 0; i < 512; ++i)
        output[i * 1024 + threadIdx.x] = values[i];
}
/** @brief The callee, not this register-light entry point, violates the rule. */
__global__ void helper_probe(const float *input, float *output)
{
    spilling_helper(input, output);
}
#elif PROBE_KIND == 1
/** @brief Force register pressure; optimized compilation must fail the guard. */
__global__ __launch_bounds__(1024) void spill_probe(const float *input, float *output)
{
    float values[128];
#pragma unroll
    for (int i = 0; i < 128; ++i)
        values[i] = input[i * 1024 + threadIdx.x];
    // The opaque memory side effect prevents moving input loads past this point.
    asm volatile("" ::: "memory");
#pragma unroll
    for (int i = 0; i < 128; ++i)
        output[i * 1024 + threadIdx.x] = values[i];
}
#elif PROBE_KIND == 2
/** @brief Intentional addressable private memory is not a register spill. */
__global__ void private_probe(const float *input, float *output, int index)
{
    volatile float values[128];
    for (int i = 0; i < 128; ++i)
        values[i] = input[i + threadIdx.x];
    output[threadIdx.x] = values[index & 127];
}
#else
/** @brief A register-light kernel must remain buildable under enforcement. */
__global__ void clean_probe(const float *input, float *output)
{
    output[threadIdx.x] = input[threadIdx.x] + 1.0f;
}
#endif
