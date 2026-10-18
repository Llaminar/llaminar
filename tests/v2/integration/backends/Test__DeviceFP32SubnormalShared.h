/**
 * @file Test__DeviceFP32SubnormalShared.h
 * @brief Captured actual canonical FP32 operations on independent fixed words.
 *
 * CUDA/HIP wrappers compile the same dynamic-operand kernel under ordinary
 * production flags. Exact unsigned output words make floating comparison FTZ
 * irrelevant. Native capture and named workspace storage remain owned through
 * the last terminal observation; poison and guards expose stale or oversized writes.
 */
#pragma once
#if defined(LLAMINAR_FP32_TEST_CUDA)
#include <cuda_runtime.h>
#define LLAMINAR_FP32_TEST_KERNEL deviceFP32SubnormalCUDA
#else
#include <hip/hip_runtime.h>
#define LLAMINAR_FP32_TEST_KERNEL deviceFP32SubnormalROCm
#endif
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/common/DeviceFP32NumericalContract.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "utils/FP32SubnormalContractCases.h"
#include <gtest/gtest.h>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace
{
using namespace llaminar2;
using namespace llaminar2::test;
/**
 * @brief Publish exact words from actual canonical operations with live operands.
 * @param source Three original binary32 input words per independent case.
 * @param output Three unsigned result words per case, without float conversion.
 * @param count Number of cases in the exact bound source/output extents.
 */
__global__ void LLAMINAR_FP32_TEST_KERNEL(const std::uint32_t *source,
                                         std::uint32_t *output, int count)
{
    const int row = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= count) return;
    namespace math = device_fp32_contract;
    const float x = math::floatFromBits(source[row * 3]);
    const float y = math::floatFromBits(source[row * 3 + 1]);
    const float z = math::floatFromBits(source[row * 3 + 2]);
    output[row * 3] = math::bitsFromFloat(math::multiply(x, y));
    output[row * 3 + 1] = math::bitsFromFloat(math::add(x, y));
    output[row * 3 + 2] = math::bitsFromFloat(math::fusedMultiplyAdd(x, y, z));
}
/**
 * @brief Reject failed lifecycle operations before reading any output words.
 * @param result Success from the exact requested backend operation.
 * @param operation Stable diagnostic identifying the failed operation.
 * @throws std::runtime_error when the operation fails.
 */
void require(bool result, const char *operation)
{
    if (!result) throw std::runtime_error(operation);
}
/** @brief Last-declared owner joins work before graph/workspace retirement. */
struct ObservationJoin
{
    IBackend *backend;
    DeviceId device;
    void *stream;
    /** @brief Retire every submitted observation, including failure unwinding. */
    ~ObservationJoin() { (void)backend->synchronizeStream(stream, device.ordinal); }
};
/**
 * @brief Compare every actual captured result and guard word through four replays.
 * @param device Explicit endpoint owned by this backend-specific test.
 */
void prove(DeviceId device)
{
    auto &context = GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&]
    {
        auto *backend = getBackendFor(device);
        void *stream = context.defaultStream();
        ASSERT_NE(backend, nullptr); ASSERT_NE(stream, nullptr);
        const size_t words = fp32SubnormalCases.size() * 3;
        constexpr size_t guards = 8;
        constexpr const char *source_name = "fp32_contract_source";
        constexpr const char *output_name = "fp32_contract_output";
        WorkspaceRequirements requirements;
        requirements.buffers = {
            {source_name, words * sizeof(std::uint32_t), 256, true},
            {output_name, (words + guards) * sizeof(std::uint32_t), 256, true}};
        // The explicit test constructor retains both exact named buffers under
        // the canonical workspace owner. No tensor/SIMD host headers enter the
        // GPU compiler, and no allocation or rebinding occurs during capture.
        DeviceWorkspaceManager workspace(device, requirements.total_bytes_with_alignment());
        require(workspace.allocate(requirements), "FP32 fixture workspace allocation");
        void *source = workspace.getBuffer(source_name);
        void *output = workspace.getBuffer(output_name);
        ASSERT_NE(source, nullptr); ASSERT_NE(output, nullptr);
        std::vector<std::uint32_t> input;
        input.reserve(words);
        for (const auto &row : fp32SubnormalCases)
        {
            input.push_back(row.lhs); input.push_back(row.rhs);
            input.push_back(row.accumulator);
        }
        require(backend->hostToDevice(source, input.data(),
            words * sizeof(std::uint32_t), device.ordinal, stream), "FP32 input publication");
        require(backend->synchronizeStream(stream, device.ordinal), "FP32 capture frontier join");
        auto graph = context.createGraphCapture(stream);
        ASSERT_NE(graph, nullptr);
        ObservationJoin join{backend, device, stream};
        {
            ScopedBackendGraphCapture recording(context, *graph, "FP32 subnormal byte contract");
            require(recording.begin(), "FP32 capture begin");
            require(backend->memset(output, 0x5a,
                (words + guards) * sizeof(std::uint32_t), device.ordinal, stream), "FP32 output poison");
#if defined(LLAMINAR_FP32_TEST_CUDA)
            LLAMINAR_FP32_TEST_KERNEL<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
                static_cast<const std::uint32_t *>(source),
                static_cast<std::uint32_t *>(output), int(fp32SubnormalCases.size()));
            require(cudaGetLastError() == cudaSuccess, "FP32 CUDA kernel launch");
#else
            hipLaunchKernelGGL(LLAMINAR_FP32_TEST_KERNEL, dim3(1), dim3(32), 0,
                static_cast<hipStream_t>(stream),
                static_cast<const std::uint32_t *>(source),
                static_cast<std::uint32_t *>(output), int(fp32SubnormalCases.size()));
            require(hipGetLastError() == hipSuccess, "FP32 ROCm kernel launch");
#endif
            recording.finish();
        }
        require(graph->instantiate(), "FP32 graph instantiate");
        std::vector<std::uint32_t> actual(words + guards);
        for (int replay = 0; replay < 4; ++replay)
        {
            // The graph retains addresses, not operand values. Rotate all cases
            // between joined replays and publish only the same fixed input extent.
            for (size_t row = 0; row < fp32SubnormalCases.size(); ++row)
            {
                const auto &test = fp32SubnormalCases[(row + replay) % fp32SubnormalCases.size()];
                input[row * 3] = test.lhs; input[row * 3 + 1] = test.rhs;
                input[row * 3 + 2] = test.accumulator;
            }
            require(backend->hostToDevice(source, input.data(),
                words * sizeof(std::uint32_t), device.ordinal, stream), "FP32 changed input publication");
            require(graph->launch(), "FP32 graph replay");
            require(backend->deviceToHost(actual.data(), output,
                actual.size() * sizeof(std::uint32_t), device.ordinal, stream) &&
                backend->synchronizeStream(stream, device.ordinal), "FP32 terminal observation");
            for (size_t row = 0; row < fp32SubnormalCases.size(); ++row)
                for (size_t op = 0; op < 3; ++op)
                    ASSERT_EQ(actual[row * 3 + op],
                        fp32SubnormalCases[(row + replay) % fp32SubnormalCases.size()].expected[op])
                        << "case=" << row << " operation=" << op << " replay=" << replay;
            for (size_t guard = words; guard < actual.size(); ++guard)
                ASSERT_EQ(actual[guard], 0x5a5a5a5au) << "guard=" << guard;
        }
    });
}
#if defined(LLAMINAR_FP32_TEST_CUDA)
/** @test CUDA retains exact subnormal inputs/results under production fast math. */
TEST(DeviceFP32SubnormalContract, CUDA) { prove(DeviceId::cuda(0)); }
#else
/** @test ROCm executes the identical fixed-word arithmetic contract. */
TEST(DeviceFP32SubnormalContract, ROCm) { prove(DeviceId::rocm(0)); }
#endif
} // namespace
#undef LLAMINAR_FP32_TEST_KERNEL
