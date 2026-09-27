/**
 * @file Perf__GPUExpertPipeline.cpp
 * @brief Captured all-format routed-expert pipeline economy on both GPUs.
 *
 * Each independently selectable case has the real 122B projection geometry.
 * Timing includes device grouping, gate/up/SwiGLU, canonical down
 * publication and reduction. Weight preparation, graph construction and result
 * downloads are outside event timing. Numerical certification belongs to the
 * independent serial oracle in MoELiveRows; this fixture additionally requires
 * the timed replay sequence to leave the same finite output bits. No performance
 * threshold or benchmark case is added to the production-preflight gate.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/DeviceMoEExpertDescriptorBuilder.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "utils/GpuPreparedGemmHarness.h"
#include "utils/TestTensorFactory.h"
#include "utils/QuantizedVerifierFormats.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using namespace llaminar2;
    using namespace llaminar2::test;

    /** One backend/storage/physical-row shape; no weight conversion is timed. */
    struct Case
    {
        DeviceId device;
        TensorType format;
        int rows;
        int live_rows;
        int local_routes;
        std::string name;
    };

    /** @return Symmetric storage, prompt/verifier geometry and sparse live-work cases. */
    std::vector<Case> cases()
    {
        std::vector<Case> result;
        for (auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
        {
            for (auto format : {TensorType::FP16, TensorType::BF16, TensorType::FP32})
                for (int rows : {1, 16, 128, 512})
                    result.push_back({device, format, rows, rows, 8,
                        std::string(device.is_cuda() ? "CUDA_" : "ROCm_") +
                        (format == TensorType::FP16 ? "FP16_" : format == TensorType::BF16 ? "BF16_" : "FP32_") +
                        "Rows" + std::to_string(rows)});
            // Capacity and live work are independent under graph capture. Sparse
            // cases model a participant that owns only some of a token's routes;
            // the full case guards against optimizing only mostly-empty graphs.
            for (const auto &format : quantizedMoEVerifierFormats())
                for (int live : {0, 1, 3, 16})
                    result.push_back({device, format.tensor_type, 16, live, live == 16 ? 8 : 3,
                        std::string(device.is_cuda() ? "CUDA_" : "ROCm_") + format.label +
                        "Rows16Live" + std::to_string(live)});
        }
        return result;
    }

    /** @return Source bytes rounded once to the requested weight storage. */
    std::unique_ptr<TensorBase> weight(TensorType type, const std::vector<size_t> &shape, uint32_t seed)
    {
        if (type == TensorType::FP16) return TestTensorFactory::createFP16Random(shape, -0.125f, 0.125f, seed);
        if (type == TensorType::BF16) return TestTensorFactory::createBF16Random(shape, -0.125f, 0.125f, seed);
        if (type == TensorType::FP32) return TestTensorFactory::createFP32Random(shape, -0.125f, 0.125f, seed);
        for (const auto &format : quantizedMoEVerifierFormats())
            if (format.tensor_type == type) return format.create(shape, seed);
        throw std::invalid_argument("unregistered expert benchmark weight format");
    }

    /** Captured, event-timed whole-operation fixture; supports one-case profiling. */
    class GPUExpertPipeline : public ::testing::TestWithParam<Case> {};

    TEST_P(GPUExpertPipeline, CapturedPipeline)
    {
        const auto parameter = GetParam();
        auto *backend = getBackendFor(parameter.device);
        if (!backend) GTEST_SKIP() << "Backend is not compiled into this build";
        if (backend->deviceCount() <= parameter.device.ordinal) GTEST_SKIP();
        auto &context = GPUDeviceContextPool::instance().getContext(parameter.device);
        context.submitAndWait([&] {
            constexpr int hidden = 3072, intermediate = 1024, experts = 8, top_k = 8;
            auto *stream = context.defaultStream();
            ASSERT_NE(stream, nullptr);
            auto tensor = [&](int m, int n) {
                auto result = TestTensorFactory::createFP32Random(
                    {static_cast<size_t>(m), static_cast<size_t>(n)}, -0.125f, 0.125f, 721);
                if (!result->ensureOnDevice(parameter.device, stream))
                    throw std::runtime_error("floating economy tensor preparation failed");
                return result;
            };
            std::vector<std::unique_ptr<TensorBase>> sources;
            std::vector<GpuPreparedGemm> prepared;
            std::array<DeviceMoEFloatingMatrixDesc, experts> gates{}, ups{}, downs{};
            std::array<DeviceNativeVNNIMatrixDesc, experts> quant_gates{}, quant_ups{}, quant_downs{};
            const bool floating = parameter.format == TensorType::FP16 ||
                parameter.format == TensorType::BF16 || parameter.format == TensorType::FP32;
            DeviceMoEWeightFormat format{};
            for (int expert = 0; expert < experts; ++expert)
            {
                for (int projection = 0; projection < 3; ++projection)
                {
                    const auto shape = projection == 2
                        ? std::vector<size_t>{hidden, intermediate}
                        : std::vector<size_t>{intermediate, hidden};
                    sources.push_back(weight(parameter.format, shape, 1000 + expert * 3 + projection));
                    const auto name = "expert.economy." + std::to_string(expert) + "." + std::to_string(projection);
                    prepared.push_back(floating
                        ? makeGpuPreparedFloatingPointGemm(sources.back().get(), parameter.device, name)
                        : makeGpuPreparedGemm(sources.back().get(), parameter.device, name));
                }
                DeviceMoEExpertDescriptor descriptor{};
                ASSERT_TRUE(exportDeviceMoEExpertWeightDescriptors(prepared[expert * 3].kernel,
                    prepared[expert * 3 + 1].kernel, prepared[expert * 3 + 2].kernel,
                    hidden, intermediate, descriptor));
                gates[expert] = descriptor.floating_gate;
                ups[expert] = descriptor.floating_up;
                downs[expert] = descriptor.floating_down;
                quant_gates[expert] = descriptor.gate;
                quant_ups[expert] = descriptor.up;
                quant_downs[expert] = descriptor.down;
                format = descriptor.weight_format;
            }
            const auto requirements = parameter.device.is_cuda()
                ? MoEWorkspaceBuffers::cudaMoE(parameter.rows, hidden, intermediate, experts, top_k)
                : MoEWorkspaceBuffers::rocmMoE(parameter.rows, hidden, intermediate, experts, top_k);
            DeviceWorkspaceManager workspace(parameter.device,
                requirements.total_bytes_with_alignment() + 4 * 1024 * 1024);
            ASSERT_TRUE(workspace.allocate(requirements));
            auto kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(parameter.device);
            ASSERT_NE(kernel, nullptr);
            kernel->setGPUStream(stream);
            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel.get());
            ASSERT_NE(consumer, nullptr);
            consumer->bindWorkspace(&workspace);
            const int gate_table = floating
                ? kernel->uploadGroupedExpertFloatingGateUpDescriptorTables(
                    gates.data(), ups.data(), format, experts, hidden, intermediate)
                : kernel->uploadGroupedExpertGateUpDescriptorTables(
                    quant_gates.data(), quant_ups.data(), experts, hidden, intermediate);
            const int down_table = floating
                ? kernel->uploadGroupedExpertFloatingDownDescriptorTable(
                    downs.data(), format, experts, hidden, intermediate)
                : kernel->uploadGroupedExpertDownDescriptorTable(
                    quant_downs.data(), experts, hidden, intermediate);
            ASSERT_GE(gate_table, 0);
            ASSERT_GE(down_table, 0);
            auto input = tensor(parameter.rows, hidden);
            auto ids = tensor(parameter.rows, top_k), probabilities = tensor(parameter.rows, top_k);
            auto output = tensor(parameter.rows, hidden), routes = tensor(parameter.rows * top_k, hidden);
            std::vector<float> route_ids(parameter.rows * top_k), route_weights(route_ids.size(), 0.125f);
            for (size_t index = 0; index < route_ids.size(); ++index)
            {
                const bool local = index / top_k < static_cast<size_t>(parameter.live_rows) &&
                    index % top_k < static_cast<size_t>(parameter.local_routes);
                route_ids[index] = local ? static_cast<float>(index % experts) : -1.0f;
                if (!local) route_weights[index] = 0.0f;
            }
            ASSERT_TRUE(backend->hostToDevice(ids->gpu_data_ptr(), route_ids.data(), route_ids.size() * sizeof(float), parameter.device.ordinal, stream));
            ASSERT_TRUE(backend->hostToDevice(probabilities->gpu_data_ptr(), route_weights.data(), route_weights.size() * sizeof(float), parameter.device.ordinal, stream));
            auto enqueue = [&] {
                return kernel->prepareExpertGroupsAsync(ids.get(), probabilities.get(), parameter.rows, experts, top_k) &&
                    kernel->executeGroupedPrefillPipeline(input.get(), output.get(), gate_table, down_table,
                        parameter.rows, hidden, intermediate, experts, top_k, routes.get()) &&
                    kernel->reduceCanonicalRouteContributions(routes.get(), output.get(), parameter.rows, top_k, hidden);
            };
            ASSERT_TRUE(enqueue());
            std::vector<float> expected(parameter.rows * hidden), actual(expected.size());
            ASSERT_TRUE(backend->deviceToHost(expected.data(), output->gpu_data_ptr(), expected.size() * sizeof(float), parameter.device.ordinal, stream));
            ASSERT_TRUE(backend->synchronizeStream(stream, parameter.device.ordinal));
            ASSERT_TRUE(std::all_of(expected.begin(), expected.end(), [](float value) { return std::isfinite(value); }));
            if (parameter.live_rows > 0)
                ASSERT_TRUE(std::any_of(expected.begin(), expected.end(), [](float value) { return value != 0.0f; }));
            else
                ASSERT_TRUE(std::all_of(expected.begin(), expected.end(), [](float value) { return value == 0.0f; }));
            auto graph = context.createGraphCapture(stream);
            {
                GraphCaptureGuard recording;
                ASSERT_TRUE(graph->beginCapture());
                ASSERT_TRUE(enqueue());
                ASSERT_TRUE(graph->endCapture());
            }
            ASSERT_TRUE(graph->instantiate());
            void *start = context.createEvent(), *end = context.createEvent();
            ASSERT_NE(start, nullptr);
            ASSERT_NE(end, nullptr);
            auto retire = [&](void *) {
                (void)backend->synchronizeStream(stream, parameter.device.ordinal);
                context.destroyEvent(end);
                context.destroyEvent(start);
            };
            std::unique_ptr<void, decltype(retire)> timing_owner(&context, retire);
            std::array<float, 9> samples{};
            for (size_t sample = 0; sample < samples.size() + 3u; ++sample)
            {
                ASSERT_TRUE(context.recordEventChecked(start, stream));
                ASSERT_TRUE(graph->launchOnStream(stream));
                ASSERT_TRUE(context.recordEventChecked(end, stream));
                ASSERT_TRUE(context.synchronizeEventChecked(end));
                if (sample >= 3u) samples[sample - 3u] = context.eventElapsedTime(start, end);
            }
            std::sort(samples.begin(), samples.end());
            std::printf("MOE_EXPERT_ECONOMY,%s,%.3f\n", parameter.name.c_str(), samples[samples.size() / 2u] * 1000.0f);
            ASSERT_TRUE(backend->deviceToHost(actual.data(), output->gpu_data_ptr(), actual.size() * sizeof(float), parameter.device.ordinal, stream));
            ASSERT_TRUE(backend->synchronizeStream(stream, parameter.device.ordinal));
            EXPECT_EQ(std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)), 0);
        });
    }

    INSTANTIATE_TEST_SUITE_P(ProductionShape, GPUExpertPipeline, ::testing::ValuesIn(cases()),
        [](const ::testing::TestParamInfo<Case> &info) { return info.param.name; });
}
