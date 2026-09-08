/**
 * @file DeviceCountedFloatingRowsProof.h
 * @brief Shared CUDA/ROCm public-adapter proof for floating live-row admission.
 *
 * Every native floating format is compared byte-for-byte with public serial
 * decode, then replayed through one retained graph while its live extent grows,
 * shrinks and becomes empty. Transfers and waits here are fixture diagnostics,
 * never part of the captured production transaction.
 */
#pragma once

#include <gtest/gtest.h>
#include "GpuPreparedGemmHarness.h"
#include "ScopedGPUStream.h"
#include "TestTensorFactory.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "transfer/TransferEngine.h"
#include <array>
#include <cstring>

namespace llaminar2::test
{
/** @brief The two independently optimized floating verifier operations. */
enum class FloatingVerifierOperation { Projection, SwiGLUDown };

/**
 * @brief Certify one native format/operation through the real tensor interface.
 * @tparam GraphCapture Backend's retained native graph implementation.
 * @tparam Stream Native stream handle accepted by that graph implementation.
 * @param device Explicit device owning all operands and publication state.
 * @param type Native FP32, FP16 or BF16 weight representation.
 * @param operation Projection or fused SwiGLU/down transaction to capture.
 */
template<class GraphCapture, class Stream>
void proveDeviceCountedFloatingRows(DeviceId device, TensorType type,
                                   FloatingVerifierOperation operation)
{
    constexpr int M = 31, N = 32, K = 257;
    SCOPED_TRACE(static_cast<int>(type));
    SCOPED_TRACE(static_cast<int>(operation));
    ScopedGPUStream stream_owner(device);
    auto stream = static_cast<Stream>(stream_owner.get());
    auto *backend = getBackendFor(device);
    ASSERT_NE(backend, nullptr);
    std::unique_ptr<TensorBase> weights;
    if (type == TensorType::FP32)
        weights = TestTensorFactory::createFP32Random({N, K}, -0.5f, 0.5f, 83041);
    else if (type == TensorType::FP16)
        weights = TestTensorFactory::createFP16Random({N, K}, -0.5f, 0.5f, 83041);
    else
        weights = TestTensorFactory::createBF16Random({N, K}, -0.5f, 0.5f, 83041);
    auto prepared = makeGpuPreparedFloatingPointGemm(
        weights.get(), device, "test.device_counted_floating",
        ModelContextId{static_cast<uint64_t>(830410 + static_cast<int>(type))});
    auto *kernel = prepared.kernel;
    ASSERT_NE(kernel, nullptr);
    kernel->setGPUStream(stream);
    auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
    ASSERT_NE(consumer, nullptr);
    auto requirements = consumer->getWorkspaceRequirements(M, N, K);
    const std::array<int, 1> columns{N};
    consumer->appendFusedProjectionWorkspaceRequirements(requirements, M, columns, K);
    auto workspace = std::make_unique<DeviceWorkspaceManager>(
        device, requirements.total_bytes_with_alignment());
    ASSERT_TRUE(workspace->allocate(requirements));
    consumer->bindWorkspace(workspace.get());
    ASSERT_TRUE(kernel->prepareFusedProjectionGraphCapture(1));
    auto input = TestTensorFactory::createFP32Random({M, K}, -0.5f, 0.5f, 83042);
    auto output = TestTensorFactory::createFP32({M, N});
    auto serial_input = TestTensorFactory::createFP32({1, K});
    auto serial_output = TestTensorFactory::createFP32({1, N});
    auto count = TestTensorFactory::createINT32({1});
    ASSERT_TRUE(input->ensureOnDevice(device, stream));
    ASSERT_TRUE(output->allocateOnDevice(device, stream));
    ASSERT_TRUE(serial_input->allocateOnDevice(device, stream));
    ASSERT_TRUE(serial_output->allocateOnDevice(device, stream));
    ASSERT_TRUE(count->allocateOnDevice(device, stream));
    std::vector<float> expected(M * N);
    for (int row = 0; row < M; ++row)
    {
        ASSERT_TRUE(backend->deviceToDevice(serial_input->gpu_data_ptr(),
            static_cast<const float *>(input->gpu_data_ptr()) + row * K,
            K * sizeof(float), device.ordinal, stream));
        TransferEngine::publishCurrentDeviceWrite(serial_input.get(), stream);
        if (operation == FloatingVerifierOperation::Projection)
            ASSERT_TRUE(kernel->multiply_tensor(serial_input.get(), serial_output.get(), 1, N, K));
        else
            ASSERT_TRUE(kernel->multiply_tensor_with_fused_swiglu(
                serial_input.get(), serial_input.get(), serial_output.get(), 1, N, K));
        ASSERT_TRUE(backend->deviceToHost(expected.data() + row * N,
            serial_output->gpu_data_ptr(), N * sizeof(float), device.ordinal, stream));
    }
    TransferEngine::requireDeviceInput(input.get(), device, stream);
    auto graph = std::make_unique<GraphCapture>(stream, device.ordinal);
    {
        auto rows = kernel->beginVerifierDecodeEquivalentScope(
            DeviceRowRange::deviceCounted(M, static_cast<const int32_t *>(count->gpu_data_ptr())));
        ScopedBackendGraphCapture recording(*graph, "floating_device_counted_verifier");
        ASSERT_TRUE(recording.begin());
        const bool ok = operation == FloatingVerifierOperation::Projection
            ? kernel->multiply_fused_verifier_rows_decode_equivalent(
                input.get(), {{kernel, output.get(), N, nullptr, "projection"}}, M, K)
            : kernel->multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                input.get(), input.get(), output.get(), M, N, K);
        recording.finish();
        ASSERT_TRUE(ok);
        ASSERT_TRUE(graph->instantiate());
    }
    std::vector<uint32_t> actual(M * N);
    for (int32_t active : {3, M, 0, 16, 2, 1, 17, M, 3})
    {
        SCOPED_TRACE(active);
        ASSERT_TRUE(backend->memset(output->gpu_data_ptr(), 0xa5,
            actual.size() * sizeof(uint32_t), device.ordinal, stream));
        ASSERT_TRUE(backend->hostToDevice(count->gpu_data_ptr(), &active,
            sizeof(active), device.ordinal, stream));
        ASSERT_TRUE(graph->launch());
        ASSERT_TRUE(backend->deviceToHost(actual.data(), output->gpu_data_ptr(),
            actual.size() * sizeof(uint32_t), device.ordinal, stream));
        EXPECT_EQ(std::memcmp(actual.data(), expected.data(), active * N * sizeof(float)), 0);
        for (size_t i = active * N; i < actual.size(); ++i)
            ASSERT_EQ(actual[i], 0xa5a5a5a5u) << "inactive=" << i;
    }
    graph.reset();
    consumer->unbindWorkspace();
    kernel->clearGPUStreamBinding();
}
}
