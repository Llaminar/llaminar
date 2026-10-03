/**
 * @file Perf__SharedExpertColumnGate.cpp
 * @brief Isolated full-row/column gate economics for the real 512-row prefill bucket.
 *
 * Each test profiles one backend, kernel and geometry. Forty identical launches
 * in a retained graph amortize launch overhead. Inputs keep sigmoid exactly one
 * so repeated in-place gating does not decay into subnormal work. The explicit
 * performance suite is not part of ProductionTestPreflight; functional sweeps
 * independently prove changing inputs, masks, ownership and arithmetic.
 */
#include <gtest/gtest.h>
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "kernels/common/SharedExpertColumnGateKernels.h"
#include "transfer/TransferEngine.h"
#include "utils/TestTensorFactory.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace llaminar2::test
{
namespace
{
/** @brief Fail explicitly while retaining fixture resource unwinding. */
void require(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }

/** @brief Own timing handles and join before borrowed graph storage can retire. */
struct Measurement
{
    IBackend &backend;
    IWorkerGPUContext &gpu;
    int ordinal;
    void *stream;
    void *begin = nullptr;
    void *end = nullptr;
    /** @brief Allocate diagnostic timing events before capture. */
    Measurement(IBackend &backend, IWorkerGPUContext &gpu, int ordinal, void *stream)
        : backend(backend), gpu(gpu), ordinal(ordinal), stream(stream)
    {
        begin = backend.createTimingEvent(ordinal);
        end = backend.createTimingEvent(ordinal);
        if (!begin || !end)
        {
            backend.destroyEvent(begin, ordinal); backend.destroyEvent(end, ordinal);
            throw std::runtime_error("shared column gate timing admission");
        }
    }
    /** @brief Finish diagnostic work before releasing timing resources. */
    ~Measurement()
    {
        (void)gpu.synchronizeStreamChecked(stream);
        backend.destroyEvent(end, ordinal); backend.destroyEvent(begin, ordinal);
    }
    /** @return Device elapsed time for one retained graph, excluding host observations. */
    double sample(IGPUGraphCapture &graph)
    {
        require(backend.recordEvent(begin, ordinal, stream) && graph.launch() &&
            backend.recordEvent(end, ordinal, stream) && backend.waitForEvent(end, ordinal), "gate timing execution");
        float milliseconds = 0;
        require(backend.eventElapsedTimeMs(begin, end, ordinal, &milliseconds), "gate elapsed interval");
        return double(milliseconds) * 1000.;
    }
};

/** @brief Measure exactly one kernel; profiler filtering never mixes candidate and control. */
void measure(DeviceId device, bool columns)
{
    constexpr int rows = 512, width = 2048, live = 448, launches = 40, samples = 41;
    const int output_width = columns ? width / 2 : width;
    auto *backend = getBackendFor(device);
    ASSERT_NE(backend, nullptr);
    auto &gpu = GPUDeviceContextPool::instance().getContext(device);
    gpu.submitAndWait([&] {
        auto *stream = gpu.getOrCreateAuxiliaryStream("shared_column_gate_economy");
        const auto tensor = [&](std::size_t count, float initial) {
            auto value = TestTensorFactory::createFP32({count});
            std::fill_n(value->mutable_data(), count, initial);
            require(value->ensureOnDevice(device, stream), "gate timing tensor admission");
            return value;
        };
        auto input = tensor(rows * width, .03125f), gate = tensor(width, .5f);
        auto shared = tensor(rows * output_width, 1.f), routed = tensor(rows * output_width, 1.f);
        auto combined = tensor(rows * output_width, -1234.5f);
        auto active = TestTensorFactory::createINT32({1});
        require(active->ensureOnDevice(device, stream), "gate timing row admission");
        // INT32 tensors do not expose a mutable float view. Publish the small
        // diagnostic input before capture on the same explicit execution stream.
        require(backend->hostToDeviceOnStream(active->gpu_data_ptr(), &live, sizeof(live),
            device.ordinal, stream), "gate timing live row publication");
        auto oracle = llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
        oracle->setGPUStream(stream);
        require(gpu.synchronizeStreamChecked(stream), "gate timing setup join");
        for (auto *value : {input.get(), gate.get(), shared.get(), routed.get()})
            TransferEngine::requireDeviceInput(value, device, stream);
        auto capture = gpu.createGraphCapture(stream);
        Measurement timer(*backend, gpu, device.ordinal, stream);
        {
            ScopedBackendGraphCapture recording(gpu, *capture, "shared_column_gate_economy");
            require(recording.begin(), "gate timing capture begin");
            for (int launch = 0; launch < launches; ++launch)
            {
                if (columns)
                    require(gateSharedExpertColumnsFP32(device, {
                        .input = static_cast<const float *>(input->gpu_data_ptr()),
                        .gate = static_cast<const float *>(gate->gpu_data_ptr()),
                        .shared = static_cast<float *>(shared->gpu_data_ptr()),
                        .routed = static_cast<const float *>(routed->gpu_data_ptr()),
                        .combined = static_cast<float *>(combined->gpu_data_ptr()),
                        .rows = rows, .model_columns = width, .local_columns = output_width,
                        .active_rows = static_cast<const int *>(active->gpu_data_ptr())}, stream), "gate timing columns");
                else
                    require(oracle->sharedExpertGateAddFromTensorsEffectiveSeqLen(input.get(), gate.get(), shared.get(),
                        routed.get(), combined.get(), rows, width, static_cast<const int *>(active->gpu_data_ptr())),
                        "gate timing full row");
            }
            recording.finish();
        }
        require(capture->instantiate(), "gate timing instantiate");
        std::vector<double> values;
        for (int iteration = -8; iteration < samples; ++iteration)
        {
            const auto us = timer.sample(*capture) / launches;
            if (iteration >= 0) values.push_back(us);
        }
        std::vector<float> output(combined->numel());
        require(backend->deviceToHostOnStream(output.data(), combined->gpu_data_ptr(), combined->size_bytes(),
            device.ordinal, stream) && gpu.synchronizeStreamChecked(stream), "gate timing terminal observation");
        for (std::size_t i = 0; i < output.size(); ++i)
            ASSERT_EQ(output[i], i < std::size_t(live) * output_width ? 2.f : 0.f);
        std::sort(values.begin(), values.end());
        // Logical traffic excludes the shared gate vector's cache reuse.
        const double bytes = sizeof(float) * (double(live) * width + 4. * live * output_width +
            2. * (rows - live) * output_width);
        std::printf("SHARED_COLUMN_GATE,backend=%s,rows=%d,live=%d,width=%d,output_width=%d,"
            "median_us=%.3f,p10_us=%.3f,p90_us=%.3f,logical_GBps=%.3f,exact=1\n",
            device.is_cuda() ? "CUDA" : "ROCm", rows, live, width, output_width,
            values[samples / 2], values[samples / 10], values[samples * 9 / 10], bytes / values[samples / 2] / 1000.);
    });
}
}
#ifdef HAVE_CUDA
TEST(SharedExpertColumnGatePerf, CUDA_Columns) { measure(DeviceId::cuda(0), true); }
TEST(SharedExpertColumnGatePerf, CUDA_Full) { measure(DeviceId::cuda(0), false); }
#endif
#ifdef HAVE_ROCM
TEST(SharedExpertColumnGatePerf, ROCm_Columns) { measure(DeviceId::rocm(0), true); }
TEST(SharedExpertColumnGatePerf, ROCm_Full) { measure(DeviceId::rocm(0), false); }
#endif
}
