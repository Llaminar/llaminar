/**
 * @file Test__ROCmQ4ScaleContribution.hip
 * @brief Captured replay of actual Q4_0 scale publication and a labeled
 * widening diagnostic.
 *
 * Recorded finite operands isolate arithmetic from routing and packed-weight
 * transport. All storage precedes capture. Partial, empty and reset epochs
 * poison outputs before replay; both programs must match scalar RN arithmetic.
 * The diagnostic boundary is test-only and is not a proposed production fix.
 */
#include "backends/GPUDeviceProcessLifetime.h"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "backends/rocm/ROCmRuntimeStartup.h"
#include "kernels/rocm/gemm/ROCmNativeVNNIDecodeEquivalentMath.h"
#include <array>
#include <bit>
#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif

namespace llaminar2::test
{
namespace
{
constexpr unsigned kCases = 256;
/** @brief Fixed recorded operands, with device-read live extent for retained
 * replay. */
struct Inputs
{
    std::array<uint16_t, kCases> half;
    std::array<float, kCases> activation, initial;
    std::array<int32_t, kCases> dot;
    unsigned live;
};
/** @brief Observe each rounding edge, not only the final cancellation-sensitive
 * sum. */
struct Result
{
    float widened, product, contribution, committed;
};
/** @brief Actual production helper path; Boundary adds only a diagnostic
 * widening barrier.
 * @tparam Boundary Whether to expose an opaque FP32 widening boundary.
 * @param in Persistent recorded operands and device-read live extent.
 * @param out Persistent per-case observations, untouched beyond the live prefix.
 */
template <bool Boundary> __global__ void replayScale(const Inputs *in, Result *out)
{
    unsigned i = threadIdx.x;
    if (i >= in->live)
        return;
    float weight = rocm_native_vnni_load_single_weight_scale<0>(in->half[i], 0);
    if constexpr (Boundary)
        weight = device_fp32_contract::persistRounded(weight);
    out[i] = {
        weight, device_fp32_contract::multiply(in->activation[i], weight),
        device_native_vnni_contract::singleScaleBlock(in->dot[i], weight, in->activation[i]),
        rocm_native_vnni_commit_single_scale_block(in->initial[i], in->dot[i], weight, in->activation[i])};
}
/** @brief Decode every finite binary16 word using exact integer
 * exponent/significand arithmetic.
 * @param h Recorded finite binary16 bit pattern.
 * @return Exactly widened binary32 value, preserving signed zero.
 */
float widen(uint16_t h)
{
    uint32_t sign = uint32_t(h & 0x8000) << 16, exponent = (h >> 10) & 31, mantissa = h & 1023;
    if (exponent)
        return std::bit_cast<float>(sign | ((exponent + 112) << 23) | (mantissa << 13));
    if (!mantissa)
        return std::bit_cast<float>(sign);
    int shift = 0;
    while (!(mantissa & 1024))
    {
        mantissa <<= 1;
        ++shift;
    }
    return std::bit_cast<float>(sign | (uint32_t(113 - shift) << 23) | ((mantissa & 1023) << 13));
}
/** @brief Binary64 holds the exact product of two binary32 significands before
 * RN narrowing.
 * @param a First binary32 operand.
 * @param b Second binary32 operand.
 * @return Binary32 round-to-nearest product.
 */
float rnProduct(float a, float b)
{
    volatile double exact = double(a) * double(b);
    volatile float r = static_cast<float>(exact);
    return r;
}
/** @brief A binary64 sum preserves enough precision for correctly rounded
 * binary32 addition.
 * @param a First binary32 operand.
 * @param b Second binary32 operand.
 * @return Binary32 round-to-nearest sum.
 */
float rnAdd(float a, float b)
{
    volatile double exact = double(a) + double(b);
    volatile float r = static_cast<float>(exact);
    return r;
}
/**
 * @brief Native failures retain the operation name and status.
 * @param s Original native operation status.
 * @param op Exact operation name.
 * @throws std::runtime_error When the native operation fails.
 */
void checked(hipError_t s, const char *op)
{
    if (s != hipSuccess)
        throw std::runtime_error(std::string(op) + " status=" + std::to_string(static_cast<int>(s)) + ": " +
                                 hipGetErrorString(s));
}
/**
 * @brief Own every native handle on its exact admitted device.
 * The process lease is first and therefore released last, after native cleanup.
 */
struct Resources
{
    GPUDeviceProcessLifetime::NativeContextLease process_lease;
    int ordinal;
    hipStream_t stream = nullptr;
    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    Inputs *input = nullptr;
    Result *actual = nullptr;
    Result *diagnostic = nullptr;
    bool capturing = false;

    /** @brief Acquire process admission before any test-native resource exists.
     * @param device Exact visible ordinal owning all handles in this scope.
     */
    explicit Resources(int device)
        : process_lease(GPUDeviceProcessLifetime::acquireNativeContext()), ordinal(device)
    {
    }

    /** @brief Fail fatally with exact operation, device and native error status.
     * @param status Original native release result.
     * @param operation Exact attempted release operation.
     */
    void release(hipError_t status, const char *operation) noexcept
    {
        if (status == hipSuccess)
            return;
        std::fprintf(stderr,
                     "[ERROR] Q4 scale replay cleanup device=%d operation=%s "
                     "status=%d error=%s\n",
                     ordinal, operation, static_cast<int>(status), hipGetErrorString(status));
        std::terminate();
    }

    /** @brief Release the retained graph before storage and stream, on its exact
     * device. */
    ~Resources() noexcept
    {
        if (!stream && !graph && !exec && !input && !actual && !diagnostic)
            return;
        release(hipSetDevice(ordinal), "hipSetDevice");
        if (capturing)
        {
            hipGraph_t abandoned = nullptr;
            const auto status = hipStreamEndCapture(stream, &abandoned);
            capturing = false;
            release(status, "abort incomplete capture");
            if (abandoned)
                release(hipGraphDestroy(abandoned), "destroy abandoned graph");
        }
        if (exec)
            release(hipGraphExecDestroy(exec), "hipGraphExecDestroy");
        if (graph)
            release(hipGraphDestroy(graph), "hipGraphDestroy");
        if (input)
            release(hipFree(input), "hipFree(input)");
        if (actual)
            release(hipFree(actual), "hipFree(actual)");
        if (diagnostic)
            release(hipFree(diagnostic), "hipFree(diagnostic)");
        if (stream)
            release(hipStreamDestroy(stream), "hipStreamDestroy");
    }
};
} // namespace
/** @test Actual production Q4_0 rounding edges equal scalar RN across captured
 * reset epochs. */
TEST(ROCmQ4ScaleContribution, CapturedProductionAndExplicitWideningDiagnostic)
{
    ASSERT_EQ(std::fegetround(), FE_TONEAREST);
#if defined(__SSE__)
    ASSERT_EQ(_mm_getcsr() & ((1u << 15) | (1u << 6)), 0u) << "Host oracle requires FTZ/DAZ disabled";
#endif
    requireROCmRuntimeStartup();
    initializeROCmHardwareMetadata();
    const auto &inventory = rocmHardwareMetadata();
    ASSERT_GT(inventory.size(), 0u);
    constexpr std::array<uint16_t, 16> halves = {0,      0x8000, 1,      0x8001, 0x3ff,  0x83ff,
                                                 0x400,  0x8400, 0x3555, 0xb555, 0x3c01, 0xbc01,
                                                 0x7bff, 0xfbff, 0x1001, 0x9001};
    constexpr std::array<uint32_t, 16> activations = {
        0,          0x80000000, 1,          0x80000001, 0x007fffff, 0x00800000, 0x33800001, 0xb3800001,
        0x3f000001, 0x3f7fffff, 0x3f800001, 0xbf800001, 0x40490fdb, 0xc0490fdb, 0x47800001, 0xc7800001};
    constexpr std::array<int32_t, 8> dots = {0, 1, -1, 32767, -32768, 32512, -32512, 17};
    Inputs input{};
    for (unsigned i = 0; i < kCases; ++i)
    {
        input.half[i] = halves[i % 16];
        input.activation[i] = std::bit_cast<float>(activations[i / 16]);
        input.dot[i] = dots[(i / 4) % 8];
        input.initial[i] = (i & 1) ? -0.125f : 0.125f;
    }
    for (size_t device = 0; device < inventory.size(); ++device)
    {
        SCOPED_TRACE(device);
        Resources r(static_cast<int>(device));
        checked(hipSetDevice(static_cast<int>(device)), "select device");
        checked(hipStreamCreateWithFlags(&r.stream, hipStreamNonBlocking), "create stream");
        checked(hipMalloc(&r.input, sizeof(Inputs)), "allocate input");
        checked(hipMalloc(&r.actual, kCases * sizeof(Result)), "allocate output");
        checked(hipMalloc(&r.diagnostic, kCases * sizeof(Result)), "allocate diagnostic");
        for (bool boundary : {false, true})
        {
            const auto function = boundary ? reinterpret_cast<const void *>(replayScale<true>)
                                           : reinterpret_cast<const void *>(replayScale<false>);
            hipFuncAttributes attributes{};
            checked(hipFuncGetAttributes(&attributes, function), "hipFuncGetAttributes");
            int active_blocks = 0;
            checked(hipOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, function, kCases, 0),
                    "hipOccupancyMaxActiveBlocksPerMultiprocessor");
            std::printf("Q4_SCALE_RESOURCE device=%zu isa=%s variant=%s regs=%d "
                        "shared=%zu local=%zu threads=%d active_blocks_per_cu=%d\n",
                        device, inventory.at(device).isa_target.c_str(),
                        boundary ? "explicit_widening_diagnostic" : "production", attributes.numRegs,
                        attributes.sharedSizeBytes, attributes.localSizeBytes, attributes.maxThreadsPerBlock,
                        active_blocks);
        }
        checked(hipStreamBeginCapture(r.stream, hipStreamCaptureModeThreadLocal), "begin capture");
        r.capturing = true;
        hipLaunchKernelGGL(replayScale<false>, dim3(1), dim3(kCases), 0, r.stream, r.input, r.actual);
        hipLaunchKernelGGL(replayScale<true>, dim3(1), dim3(kCases), 0, r.stream, r.input, r.diagnostic);
        checked(hipGetLastError(), "record kernels");
        const auto end_status = hipStreamEndCapture(r.stream, &r.graph);
        // EndCapture closes the native interval even on failure. Do not call it
        // twice during unwinding.
        r.capturing = false;
        checked(end_status, "hipStreamEndCapture");
        checked(hipGraphInstantiate(&r.exec, r.graph, nullptr, nullptr, 0), "instantiate");
        for (unsigned repeat = 0; repeat < 3; ++repeat)
            for (unsigned live : {256u, 17u, 1u, 0u, 256u})
            {
                input.live = live;
                checked(hipMemcpyAsync(r.input, &input, sizeof(input), hipMemcpyHostToDevice, r.stream),
                        "publish input");
                checked(hipMemsetAsync(r.actual, 0xa5, kCases * sizeof(Result), r.stream), "poison output");
                checked(hipMemsetAsync(r.diagnostic, 0xa5, kCases * sizeof(Result), r.stream),
                        "poison diagnostic");
                checked(hipGraphLaunch(r.exec, r.stream), "replay");
                std::array<Result, kCases> actual{}, diagnostic{};
                checked(
                    hipMemcpyAsync(actual.data(), r.actual, sizeof(actual), hipMemcpyDeviceToHost, r.stream),
                    "collect output");
                checked(hipMemcpyAsync(diagnostic.data(), r.diagnostic, sizeof(diagnostic),
                                       hipMemcpyDeviceToHost, r.stream),
                        "collect diagnostic");
                checked(hipStreamSynchronize(r.stream), "terminal collection");
                for (unsigned i = 0; i < kCases; ++i)
                {
                    SCOPED_TRACE(i);
                    SCOPED_TRACE(live);
                    Result expected{};
                    if (i < live)
                    {
                        expected.widened = widen(input.half[i]);
                        expected.product = rnProduct(input.activation[i], expected.widened);
                        expected.contribution = rnProduct(expected.product, float(input.dot[i]));
                        expected.committed = rnAdd(input.initial[i], expected.contribution);
                    }
                    else
                        for (float *v : {&expected.widened, &expected.product, &expected.contribution,
                                         &expected.committed})
                            *v = std::bit_cast<float>(0xa5a5a5a5u);
                    EXPECT_EQ((std::bit_cast<std::array<uint32_t, 4>>(actual[i])),
                              (std::bit_cast<std::array<uint32_t, 4>>(expected)))
                        << "production";
                    EXPECT_EQ((std::bit_cast<std::array<uint32_t, 4>>(diagnostic[i])),
                              (std::bit_cast<std::array<uint32_t, 4>>(expected)))
                        << "explicit widening diagnostic";
                }
            }
    }
}
} // namespace llaminar2::test
