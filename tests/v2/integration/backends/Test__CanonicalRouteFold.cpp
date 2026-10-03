/**
 * @file Test__CanonicalRouteFold.cpp
 * @brief Captured serial-order route folding across GPU geometry and replay.
 *
 * This format-independent boundary receives FP32 contributions from every
 * supported quantized/floating expert. Real graph replay changes the tensor
 * contents through dense, all-zero, sparse and new dense inputs, without
 * rebinding storage or recapturing. The independent CPU oracle requires byte
 * equality, including cancellation and signed zero. Poisoned input capacity
 * and output guards expose incorrect live extents. Performance is measured
 * separately and never controls this functional production-preflight gate.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "utils/TestTensorFactory.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using namespace llaminar2;
    using namespace llaminar2::test;

    /** @brief Require an explicitly selected backend; unavailable devices fail. */
    class CanonicalRouteFold : public ::testing::TestWithParam<std::string> {};

    /** @brief Reject an incomplete fixture operation instead of observing stale bytes. */
    void require(bool success, const char *operation)
    {
        if (!success) throw std::runtime_error(operation);
    }

    /** @brief Keep asynchronous fixture storage alive through observation or failure. */
    struct ObservationJoin
    {
        IBackend *backend;
        DeviceId device;
        void *stream;
        /** @brief Test-only terminal join before graph and tensor destruction. */
        ~ObservationJoin() { (void)backend->synchronizeStream(stream, device.ordinal); }
    };

    /**
     * @brief Replay one production launch geometry against a serial FP32 oracle.
     * @param context Owner of the real endpoint and its explicit capture stream.
     * @param device Endpoint used for every operation and final observation.
     * @param rows Retained graph's exact row extent, including verifier tails.
     * @param columns Local output width, including partial thread blocks.
     * @param routes Live router-order contributions, not a padded window capacity.
     */
    void prove(IWorkerGPUContext &context, DeviceId device, int rows, int columns, int routes)
    {
        SCOPED_TRACE(::testing::Message() << device.to_string() << " rows=" << rows
            << " columns=" << columns << " routes=" << routes);
        constexpr std::size_t guard = 16;
        const std::size_t input_count = std::size_t(rows) * columns * routes;
        const std::size_t output_count = std::size_t(rows) * columns;
        auto *backend = getBackendFor(device);
        void *stream = context.defaultStream();
        require(backend && stream, "route fold requires an explicit real backend stream");
        auto kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
        require(kernel != nullptr, "route fold kernel factory");
        kernel->setGPUStream(stream);
        auto input = TestTensorFactory::createFP32Zeros({input_count + guard});
        auto output = TestTensorFactory::createFP32Zeros({output_count + guard});
        auto graph = context.createGraphCapture(stream);
        ObservationJoin join{backend, device, stream};
        require(input->ensureOnDevice(device, stream) && output->ensureOnDevice(device, stream),
            "route fold fixture admission");
        const auto enqueue = [&] {
            return kernel->reduceCanonicalRouteContributions(input.get(), output.get(), rows, routes, columns);
        };
        require(enqueue(), "route fold warmup");
        {
            ScopedBackendGraphCapture recording(context, *graph, "route fold serial-order regression");
            require(recording.begin(), "route fold capture begin");
            require(enqueue(), "route fold captured production launch");
            recording.finish();
        }
        require(graph->instantiate(), "route fold graph instantiate");
        std::vector<GPUGraphKernelNodeInfo> nodes;
        std::string inspection_error;
        require(graph->inspectKernelNodes(nodes, &inspection_error), "route fold native node inspection");
        ASSERT_EQ(nodes.size(), 1u) << inspection_error;
        ASSERT_TRUE(nodes.front().valid());
        EXPECT_EQ(nodes.front().local_memory_bytes_per_thread, 0u);

        constexpr std::array<float, 10> pattern{
            16777216.0f, 1.0f, -16777216.0f, 0.25f, -0.5f,
            3.0f, -0.125f, 0.0625f, 0.0f, -0.0f};
        constexpr float sentinel = -98765.5f;
        std::vector<float> source(input_count + guard);
        std::vector<float> expected(output_count + guard);
        std::vector<float> actual(output_count + guard);
        for (int replay = 0; replay < 4; ++replay)
        {
            SCOPED_TRACE(::testing::Message() << "replay=" << replay);
            std::fill(source.begin(), source.end(), std::numeric_limits<float>::quiet_NaN());
            std::fill(expected.begin(), expected.end(), sentinel);
            std::fill(actual.begin(), actual.end(), sentinel);
            for (int row = 0; row < rows; ++row)
                for (int column = 0; column < columns; ++column)
                {
                    volatile float sum = 0.0f;
                    for (int route = 0; route < routes; ++route)
                    {
                        // Inactive slots remain legitimate zero contributions;
                        // only capacity beyond the declared tensor is poison.
                        const bool inactive = replay == 1 ||
                            (replay == 2 && (row % 3 != 0 || route % 3 != 0));
                        const float value = inactive ? -0.0f
                            : pattern[(row + column + route + replay) % pattern.size()];
                        source[(std::size_t(row) * routes + route) * columns + column] = value;
                        sum = sum + value;
                    }
                    expected[std::size_t(row) * columns + column] = sum;
                }
            require(backend->hostToDevice(input->gpu_data_ptr(), source.data(),
                source.size() * sizeof(float), device.ordinal, stream), "route fold input publication");
            require(backend->hostToDevice(output->gpu_data_ptr(), actual.data(),
                actual.size() * sizeof(float), device.ordinal, stream), "route fold output guards");
            require(graph->launchOnStream(stream), "route fold retained replay");
            require(backend->deviceToHost(actual.data(), output->gpu_data_ptr(),
                actual.size() * sizeof(float), device.ordinal, stream) &&
                backend->synchronizeStream(stream, device.ordinal), "route fold terminal observation");
            if (std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) != 0)
                for (std::size_t index = 0; index < actual.size(); ++index)
                    ASSERT_EQ(std::bit_cast<std::uint32_t>(actual[index]),
                        std::bit_cast<std::uint32_t>(expected[index])) << "output/guard index=" << index;
        }
    }

    /** @test Every verifier M plus representative prefill shapes preserve exact replay. */
    TEST_P(CanonicalRouteFold, CapturedSerialOrderAndReplay)
    {
        const DeviceId device = GetParam() == "CUDA" ? DeviceId::cuda(0) : DeviceId::rocm(0);
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        context.submitAndWait([&] {
            for (int rows = 1; rows <= 16; ++rows)
                for (int routes : {1, 3, 7, 8, 9, 15, 16, 17, 33})
                    prove(context, device, rows, 65, routes);
            for (int rows : {17, 64, 129, 512})
                for (int columns : {256, 513, 2048})
                    for (int routes : {7, 8, 9, 17})
                        prove(context, device, rows, columns, routes);
        });
    }

    INSTANTIATE_TEST_SUITE_P(Backends, CanonicalRouteFold, ::testing::ValuesIn(std::vector<std::string>{
#ifdef HAVE_CUDA
        "CUDA",
#endif
#ifdef HAVE_ROCM
        "ROCm",
#endif
    }), [](const auto &info) { return info.param; });
}
