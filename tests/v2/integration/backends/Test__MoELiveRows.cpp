/**
 * @file Test__MoELiveRows.cpp
 * @brief Retained CUDA/ROCm MoE graphs consume live rows, not capture capacity.
 *
 * Every quantized and floating weight format exercises shared and routed experts. One graph
 * is replayed across empty, short, and full publications without rebinding or
 * recapture. Active output bytes must equal independent serial decode; padded
 * output must be zero. This detects stale scratch, double publication by two
 * dispatch families, and shared experts accidentally evaluating padded rows.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "execution/moe/DeviceMoEExpertDescriptorBuilder.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "utils/GpuPreparedGemmHarness.h"
#include "utils/QuantizedVerifierFormats.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace
{
    using namespace llaminar2;
    using namespace llaminar2::test;

    /** @brief Identical lifecycle and byte gates for both GPU backends. */
    class MoELiveRows : public ::testing::TestWithParam<std::string> {};

    /** @brief A source factory plus its production descriptor family. */
    struct WeightCase
    {
        const char *label;
        DeviceMoEWeightFormat format;
        std::function<std::unique_ptr<TensorBase>(const std::vector<size_t> &, uint32_t)> create;
    };

    /** @return Canonical quantized inventory plus all three floating formats. */
    std::vector<WeightCase> liveRowsFormats()
    {
        std::vector<WeightCase> cases;
        for (const auto &format : quantizedMoEVerifierFormats())
            cases.push_back({format.label, DeviceMoEWeightFormat::NativeVNNI, format.create});
        cases.push_back({"FP16", DeviceMoEWeightFormat::FP16,
            [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
            { return TestTensorFactory::createFP16Random(shape, -0.125f, 0.125f, seed); }});
        cases.push_back({"BF16", DeviceMoEWeightFormat::BF16,
            [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
            { return TestTensorFactory::createBF16Random(shape, -0.125f, 0.125f, seed); }});
        cases.push_back({"FP32", DeviceMoEWeightFormat::FP32,
            [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
            { return TestTensorFactory::createFP32Random(shape, -0.125f, 0.125f, seed); }});
        return cases;
    }

    /**
     * @brief Exercise one endpoint on its worker's explicit non-default stream.
     * @param device Endpoint whose prepared weights and scratch own the graph.
     * @param context Worker that orders input publication, capture, and replay.
     */
    void proveLiveRows(DeviceId device, IWorkerGPUContext &context)
    {
        context.submitAndWait([&]()
        {
            constexpr int rows = 16, hidden_size = 256, width = 256;
            auto *backend = getBackendFor(device);
            auto *stream = context.defaultStream();
            ASSERT_NE(backend, nullptr);
            ASSERT_NE(stream, nullptr);

            // Test-owned mutable publications are written only between complete
            // transactions. No copy, allocation or host decision enters capture.
            auto tensor = [&](int m, int n)
            {
                auto result = TestTensorFactory::createFP32(
                    {static_cast<size_t>(m), static_cast<size_t>(n)});
                std::fill_n(result->mutable_data(), result->numel(), 0.0f);
                if (!result->ensureOnDevice(device, stream))
                    throw std::runtime_error("live-row fixture materialization failed");
                return result;
            };
            auto upload = [&](TensorBase *target, const void *data, size_t bytes)
            {
                return backend->hostToDevice(target->gpu_data_ptr(), data, bytes,
                                             0, stream);
            };
            auto observe = [&](TensorBase *source, std::vector<float> &values)
            {
                return backend->deviceToHost(values.data(), source->gpu_data_ptr(),
                           values.size() * sizeof(float), 0, stream) &&
                       backend->synchronizeStream(stream, 0);
            };

            std::vector<float> hidden_values(rows * hidden_size);
            for (size_t index = 0; index < hidden_values.size(); ++index)
                hidden_values[index] = 0.019f * static_cast<float>(
                    static_cast<int>((index * 17 + 3) % 47) - 23);

            for (const auto &format : liveRowsFormats())
            {
                SCOPED_TRACE(format.label);
                // Distinct experts prevent wrong expert-id publication from
                // passing merely because every descriptor has equal weights.
                std::vector<std::unique_ptr<TensorBase>> source_weights;
                std::vector<GpuPreparedGemm> prepared;
                std::array<DeviceMoEExpertDescriptor, 2> descriptors{};
                const bool floating = deviceMoEWeightFormatIsFloating(format.format);
                for (size_t expert = 0; expert < descriptors.size(); ++expert)
                {
                    for (int role = 0; role < 3; ++role)
                    {
                        const std::vector<size_t> shape = role == 2
                            ? std::vector<size_t>{hidden_size, width}
                            : std::vector<size_t>{width, hidden_size};
                        source_weights.push_back(format.create(shape, 68001 + expert * 10 + role));
                        const std::string name = std::string("test.live_rows.") + format.label +
                            "." + std::to_string(expert) + "." + std::to_string(role);
                        prepared.push_back(floating
                            ? makeGpuPreparedFloatingPointGemm(source_weights.back().get(), device, name)
                            : makeGpuPreparedGemm(source_weights.back().get(), device, name));
                    }
                    ASSERT_TRUE(exportDeviceMoEExpertWeightDescriptors(
                        prepared[expert * 3].kernel, prepared[expert * 3 + 1].kernel,
                        prepared[expert * 3 + 2].kernel, hidden_size, width, descriptors[expert]));
                    ASSERT_EQ(descriptors[expert].weight_format, format.format);
                }

                for (const bool shared : {false, true})
                {
                    SCOPED_TRACE(shared ? "shared" : "routed");
                    const int experts = shared ? 1 : 16;
                    const int top_k = shared ? 1 : 4;
                    const auto requirements = device.is_cuda()
                        ? MoEWorkspaceBuffers::cudaMoE(rows, hidden_size, width, experts, top_k)
                        : MoEWorkspaceBuffers::rocmMoE(rows, hidden_size, width, experts, top_k);
                    DeviceWorkspaceManager workspace(device,
                        requirements.total_bytes_with_alignment() + 4 * 1024 * 1024);
                    ASSERT_TRUE(workspace.allocate(requirements));
                    auto kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
                    ASSERT_NE(kernel, nullptr);
                    kernel->setGPUStream(stream);
                    auto *workspace_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel.get());
                    ASSERT_NE(workspace_consumer, nullptr);
                    workspace_consumer->bindWorkspace(&workspace);
                    std::vector<DeviceNativeVNNIMatrixDesc> gates(experts), ups(experts), downs(experts);
                    std::vector<DeviceMoEFloatingMatrixDesc> fp_gates(experts), fp_ups(experts), fp_downs(experts);
                    for (int expert = 0; expert < experts; ++expert)
                    {
                        const auto &descriptor = descriptors[expert % descriptors.size()];
                        gates[expert] = descriptor.gate;
                        ups[expert] = descriptor.up;
                        downs[expert] = descriptor.down;
                        fp_gates[expert] = descriptor.floating_gate;
                        fp_ups[expert] = descriptor.floating_up;
                        fp_downs[expert] = descriptor.floating_down;
                    }
                    const int gateup_table = floating
                        ? kernel->uploadGroupedExpertFloatingGateUpDescriptorTables(
                            fp_gates.data(), fp_ups.data(), format.format, experts, hidden_size, width)
                        : kernel->uploadGroupedExpertGateUpDescriptorTables(
                            gates.data(), ups.data(), experts, hidden_size, width);
                    const int down_table = floating
                        ? kernel->uploadGroupedExpertFloatingDownDescriptorTable(
                            fp_downs.data(), format.format, experts, hidden_size, width)
                        : kernel->uploadGroupedExpertDownDescriptorTable(
                            downs.data(), experts, hidden_size, width);
                    ASSERT_GE(gateup_table, 0);
                    ASSERT_GE(down_table, 0);

                    auto hidden = tensor(rows, hidden_size);
                    auto indices = tensor(rows, top_k);
                    auto weights = tensor(rows, top_k);
                    auto output = tensor(rows, hidden_size);
                    auto contributions = tensor(rows * top_k, hidden_size);
                    auto reduced_output = tensor(rows, hidden_size);
                    auto live_count = tensor(1, 1); // INT32 publication, opaque to TensorBase.
                    auto *count = static_cast<int32_t *>(live_count->gpu_data_ptr());
                    ASSERT_TRUE(upload(hidden.get(), hidden_values.data(), hidden_values.size() * sizeof(float)));
                    std::vector<float> route_ids(rows * top_k), route_weights(rows * top_k);
                    for (int row = 0; row < rows; ++row)
                        for (int route = 0; route < top_k; ++route)
                        {
                            route_ids[row * top_k + route] = (row * 3 + route) % experts;
                            route_weights[row * top_k + route] = shared ? 1.0f :
                                static_cast<float>(route + 1) / 10.0f;
                        }

                    // The independent M=1 oracle uses the ordinary decode API,
                    // not the grouped implementation that is under test.
                    auto row_hidden = tensor(1, hidden_size);
                    auto row_indices = tensor(1, top_k);
                    auto row_weights = tensor(1, top_k);
                    auto row_output = tensor(1, hidden_size);
                    std::vector<std::unique_ptr<FP32Tensor>> gate_storage, up_storage;
                    std::vector<ITensor *> gate_outputs, up_outputs;
                    for (int route = 0; route < top_k; ++route)
                    {
                        gate_storage.push_back(tensor(1, width));
                        up_storage.push_back(tensor(1, width));
                        gate_outputs.push_back(gate_storage.back().get());
                        up_outputs.push_back(up_storage.back().get());
                    }
                    std::vector<float> serial(rows * hidden_size), one_row(hidden_size);
                    for (int row = 0; row < rows; ++row)
                    {
                        ASSERT_TRUE(upload(row_hidden.get(), hidden_values.data() + row * hidden_size,
                                           hidden_size * sizeof(float)));
                        ASSERT_TRUE(upload(row_indices.get(), route_ids.data() + row * top_k, top_k * sizeof(float)));
                        ASSERT_TRUE(upload(row_weights.get(), route_weights.data() + row * top_k, top_k * sizeof(float)));
                        // Keep host table metadata alive through terminal
                        // observation, including any asynchronous setup copy.
                        std::vector<int> ids(top_k);
                        if (floating)
                        {
                            // Floating serial decode uses the typed table API;
                            // the FromRouting split-projection API is quantized.
                            // Its expert IDs belong to an immutable capture
                            // owner. Give each oracle row its own owner instead
                            // of illegally rewriting a previously prepared one.
                            // This host row replay is solely the test oracle.
                            const auto serial_requirements = device.is_cuda()
                                ? MoEWorkspaceBuffers::cudaMoE(1, hidden_size, width, experts, top_k)
                                : MoEWorkspaceBuffers::rocmMoE(1, hidden_size, width, experts, top_k);
                            DeviceWorkspaceManager serial_workspace(device,
                                serial_requirements.total_bytes_with_alignment() + 4 * 1024 * 1024);
                            ASSERT_TRUE(serial_workspace.allocate(serial_requirements));
                            auto serial_kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
                            ASSERT_NE(serial_kernel, nullptr);
                            serial_kernel->setGPUStream(stream);
                            auto *serial_consumer = dynamic_cast<IWorkspaceConsumer *>(serial_kernel.get());
                            ASSERT_NE(serial_consumer, nullptr);
                            serial_consumer->bindWorkspace(&serial_workspace);
                            const int serial_gateup = serial_kernel->uploadGroupedExpertFloatingGateUpDescriptorTables(
                                fp_gates.data(), fp_ups.data(), format.format, experts, hidden_size, width);
                            const int serial_down = serial_kernel->uploadGroupedExpertFloatingDownDescriptorTable(
                                fp_downs.data(), format.format, experts, hidden_size, width);
                            ASSERT_GE(serial_gateup, 0);
                            ASSERT_GE(serial_down, 0);
                            std::transform(route_ids.begin() + row * top_k,
                                route_ids.begin() + (row + 1) * top_k, ids.begin(),
                                [](float value) { return static_cast<int>(value); });
                            const float *probabilities = route_weights.data() + row * top_k;
                            ASSERT_TRUE(serial_kernel->prepareGroupedTableDecodeLaunchState(ids.data(),
                                probabilities, serial_gateup, serial_down, top_k, gate_outputs.data(),
                                up_outputs.data(), row_output.get(), hidden_size, width));
                            ASSERT_TRUE(serial_kernel->groupedExpertGateUpDecodeFromTable(row_hidden.get(),
                                ids.data(), serial_gateup, top_k, gate_outputs.data(), up_outputs.data(),
                                hidden_size, width));
                            ASSERT_TRUE(serial_kernel->groupedExpertDownDecodeFromTable(gate_outputs.data(),
                                up_outputs.data(), ids.data(), probabilities, serial_down, top_k,
                                row_output.get(), hidden_size, width));
                            // The independent owner cannot retire its scratch
                            // until this oracle row has finished using it.
                            ASSERT_TRUE(observe(row_output.get(), one_row));
                        }
                        else
                        {
                            ASSERT_TRUE(kernel->groupedExpertGateUpDecodeFromRouting(row_hidden.get(),
                                row_indices.get(), gateup_table, top_k, gate_outputs.data(), up_outputs.data(),
                                hidden_size, width));
                            ASSERT_TRUE(kernel->groupedExpertDownDecodeFromRouting(gate_outputs.data(),
                                up_outputs.data(), row_indices.get(), row_weights.get(), down_table, top_k,
                                row_output.get(), hidden_size, width));
                            ASSERT_TRUE(observe(row_output.get(), one_row));
                        }
                        std::copy(one_row.begin(), one_row.end(), serial.begin() + row * hidden_size);
                    }
                    ASSERT_TRUE(std::all_of(serial.begin(), serial.end(), [](float x) { return std::isfinite(x); }));
                    ASSERT_TRUE(std::any_of(serial.begin(), serial.end(), [](float x) { return x != 0.0f; }));

                    ASSERT_TRUE(upload(indices.get(), route_ids.data(), route_ids.size() * sizeof(float)));
                    ASSERT_TRUE(upload(weights.get(), route_weights.data(), route_weights.size() * sizeof(float)));
                    ASSERT_TRUE(upload(live_count.get(), &rows, sizeof(rows)));
                    auto enqueue = [&]()
                    {
                        const bool grouped = shared
                            ? kernel->prepareSharedExpertPrefillGroup(DeviceRowRange::deviceCounted(rows, count))
                            : kernel->prepareExpertGroupsAsync(indices.get(), weights.get(), rows, experts, top_k);
                        return grouped && kernel->executeGroupedPrefillPipeline(hidden.get(), output.get(),
                            gateup_table, down_table, rows, hidden_size, width, experts, top_k) &&
                            kernel->executeGroupedPrefillPipeline(hidden.get(), reduced_output.get(),
                                gateup_table, down_table, rows, hidden_size, width, experts, top_k,
                                contributions.get()) &&
                            kernel->reduceCanonicalRouteContributions(contributions.get(), reduced_output.get(),
                                rows, top_k, hidden_size);
                    };
                    // Materialize every persistent scratch view before recording.
                    ASSERT_TRUE(enqueue());
                    ASSERT_TRUE(backend->synchronizeStream(stream, 0));
                    auto capture = context.createGraphCapture(stream);
                    ASSERT_NE(capture, nullptr);
                    {
                        GraphCaptureGuard recording;
                        ASSERT_TRUE(capture->beginCapture());
                        ASSERT_TRUE(enqueue());
                        ASSERT_TRUE(capture->endCapture());
                    }
                    ASSERT_TRUE(capture->instantiate());
                    std::vector<float> actual(serial.size());
                    std::vector<float> reduced(serial.size());
                    const std::vector<float> poison(serial.size(), std::numeric_limits<float>::quiet_NaN());
                    // This permutation crosses the small-M/tiled boundary in
                    // both directions and revisits empty/full publications.
                    for (int replay = 0; replay < 35; ++replay)
                    {
                        const int active = (replay * 7) % 17;
                        SCOPED_TRACE(::testing::Message() << "replay=" << replay << " live=" << active);
                        auto active_ids = route_ids, active_weights = route_weights;
                        std::fill(active_ids.begin() + active * top_k, active_ids.end(), -1.0f);
                        std::fill(active_weights.begin() + active * top_k, active_weights.end(), 0.0f);
                        ASSERT_TRUE(upload(indices.get(), active_ids.data(), active_ids.size() * sizeof(float)));
                        ASSERT_TRUE(upload(weights.get(), active_weights.data(), active_weights.size() * sizeof(float)));
                        ASSERT_TRUE(upload(live_count.get(), &active, sizeof(active)));
                        ASSERT_TRUE(upload(output.get(), poison.data(), poison.size() * sizeof(float)));
                        ASSERT_TRUE(capture->launch());
                        ASSERT_TRUE(observe(output.get(), actual));
                        ASSERT_TRUE(observe(reduced_output.get(), reduced));
                        for (size_t index = 0; index < actual.size(); ++index)
                        {
                            const float expected = index < static_cast<size_t>(active * hidden_size)
                                ? serial[index] : 0.0f;
                            ASSERT_EQ(std::bit_cast<uint32_t>(actual[index]), std::bit_cast<uint32_t>(expected))
                                << "row=" << index / hidden_size << " col=" << index % hidden_size;
                            ASSERT_EQ(std::bit_cast<uint32_t>(reduced[index]), std::bit_cast<uint32_t>(expected))
                                << "canonical publication row=" << index / hidden_size
                                << " col=" << index % hidden_size;
                        }
                        if (shared)
                        {
                            int published_count = -1;
                            ASSERT_TRUE(backend->deviceToHost(&published_count,
                                workspace.getBuffer(MoEWorkspaceBuffers::GROUP_COUNTS),
                                sizeof(published_count), 0, stream));
                            ASSERT_TRUE(backend->synchronizeStream(stream, 0));
                            ASSERT_EQ(published_count, active) << "padded shared rows must never become work";
                        }
                    }
                }
            }
        });
    }

    /** @test One retained graph follows all live counts for every codebook. */
    TEST_P(MoELiveRows, CapturedSharedAndRoutedAllFormats)
    {
        if (GetParam() == "CUDA")
            proveLiveRows(DeviceId::cuda(0), GPUDeviceContextPool::instance().getNvidiaContext(0));
        else
            proveLiveRows(DeviceId::rocm(0), GPUDeviceContextPool::instance().getAMDContext(0));
    }

    INSTANTIATE_TEST_SUITE_P(Backends, MoELiveRows, ::testing::Values("CUDA", "ROCm"),
        [](const ::testing::TestParamInfo<std::string> &info) { return info.param; });
}
