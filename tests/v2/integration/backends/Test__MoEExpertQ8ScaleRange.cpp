/**
 * @file Test__MoEExpertQ8ScaleRange.cpp
 * @brief Observe production expert Q8 metadata at finite-source boundaries.
 *
 * Public table decode and retained shared-expert prefill execute the actual
 * private hidden and post-SwiGLU producers. Named graph workspace publications
 * are inspected only after an explicit terminal observation join. The router
 * is deliberately absent so its reusable hidden sidecar cannot mask a broken
 * expert producer. Source quantized weights retain their original format.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "kernels/common/DeviceQ8ActivationNumericalContract.h"
#include "kernels/common/DeviceSwiGLUNumericalContract.h"
#include "utils/GpuPreparedGemmHarness.h"
#include "utils/Q8ScaleUnderflowFixture.h"
#include "utils/TestTensorFactory.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

namespace
{
using namespace llaminar2;
using namespace llaminar2::test;
constexpr int width = 256;
constexpr int blocks = width / 32;

/**
 * @brief Reject a failed setup or publication before interpreting its bytes.
 * @param result Success returned by the exact production operation.
 * @param operation Stable diagnostic identifying that operation.
 * @throws std::runtime_error when the operation did not succeed.
 */
void require(bool result, const char *operation)
{
    if (!result) throw std::runtime_error(operation);
}

/** @brief Retire submitted work before its retained graph and storage owners. */
struct ObservationJoin
{
    IBackend *backend;
    DeviceId device;
    void *stream;
    /** @brief Also retire a partially submitted failed observation. */
    ~ObservationJoin() { (void)backend->synchronizeStream(stream, device.ordinal); }
};

/**
 * @brief Inspect exact byte/scale publications against finite original values.
 * @param backend Backend owning the published workspace.
 * @param device Exact device of the captured producer.
 * @param stream Non-default terminal observation stream.
 * @param workspace Stable producer-owned scratch, retained through observation.
 * @param bytes_name Typed workspace key for contiguous Q8 rows.
 * @param scales_name Typed workspace key for contiguous block scales.
 * @param expected Unquantized original rows or exact canonical SwiGLU results.
 */
void checkPublication(IBackend *backend, DeviceId device, void *stream,
                      DeviceWorkspaceManager &workspace, const char *bytes_name,
                      const char *scales_name, const std::vector<float> &expected)
{
    std::vector<int8_t> bytes(expected.size());
    std::vector<float> scales(expected.size() / 32);
    ASSERT_LE(bytes.size(), workspace.getBufferSize(bytes_name));
    ASSERT_LE(scales.size() * sizeof(float), workspace.getBufferSize(scales_name));
    ASSERT_NE(workspace.getBuffer(bytes_name), nullptr);
    ASSERT_NE(workspace.getBuffer(scales_name), nullptr);
    require(backend->deviceToHost(bytes.data(), workspace.getBuffer(bytes_name),
        bytes.size(), device.ordinal, stream), "expert bytes observation");
    require(backend->deviceToHost(scales.data(), workspace.getBuffer(scales_name),
        scales.size() * sizeof(float), device.ordinal, stream), "expert scales observation");
    require(backend->synchronizeStream(stream, device.ordinal), "expert observation join");
    for (size_t block = 0; block < scales.size(); ++block)
    {
        float maximum = 0.f;
        for (size_t lane = 0; lane < 32; ++lane)
            maximum = std::max(maximum, std::fabs(expected[block * 32 + lane]));
        // This independent branch is evaluated on the host without GPU FTZ.
        const float scale = maximum == 0.f ? 1.f : canonicalPreparedHalfValue(
            std::clamp(device_fp32_contract::multiply(maximum,
                MoEProjectionNumericalContract::q8_scale_multiplier), 0x1p-24f, 65504.f));
        ASSERT_EQ(std::bit_cast<uint32_t>(scales[block]), std::bit_cast<uint32_t>(scale))
            << bytes_name << " block=" << block;
        ASSERT_TRUE(std::isfinite(scales[block]) && scales[block] > 0.f);
        for (size_t lane = 0; lane < 32; ++lane)
            ASSERT_EQ(int(bytes[block * 32 + lane]),
                device_q8_activation_contract::quantize(expected[block * 32 + lane], scale))
                << bytes_name << " block=" << block << " lane=" << lane;
    }
}

/**
 * @brief Exercise actual expert producers with controlled and computed intermediates.
 * @param device Explicit CUDA or ROCm endpoint, never a silent alternate mode.
 */
void proveExpertScaleRange(DeviceId device)
{
    auto &context = GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&]
    {
        auto *backend = getBackendFor(device);
        void *stream = context.defaultStream();
        ASSERT_NE(backend, nullptr);
        ASSERT_NE(stream, nullptr);
        constexpr int capacity = 512;
        const auto requirements = device.is_cuda()
            ? MoEWorkspaceBuffers::cudaMoE(capacity, width, width, 1, 1)
            : MoEWorkspaceBuffers::rocmMoE(capacity, width, width, 1, 1);
        DeviceWorkspaceManager workspace(device, requirements.total_bytes_with_alignment() + 4 * 1024 * 1024);
        ASSERT_TRUE(workspace.allocate(requirements));
        auto kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
        ASSERT_NE(kernel, nullptr);
        kernel->setGPUStream(stream);
        auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel.get());
        ASSERT_NE(consumer, nullptr);
        consumer->bindWorkspace(&workspace);
        std::vector<std::unique_ptr<TensorBase>> sources;
        std::vector<GpuPreparedGemm> prepared;
        std::array<DeviceNativeVNNIMatrixDesc, 3> descriptors{};
        for (int role = 0; role < 3; ++role)
        {
            sources.push_back(TestTensorFactory::createIQ4_NLRandom({width, width}, 7310 + role));
            prepared.push_back(makeGpuPreparedGemm(sources.back().get(), device,
                "test.expert.q8.scale." + std::to_string(role)));
            ASSERT_TRUE(prepared.back().kernel->exportNativeVNNIMatrixDesc(descriptors[role]));
        }
        const int gateup = kernel->uploadGroupedExpertGateUpDescriptorTables(
            &descriptors[0], &descriptors[1], 1, width, width);
        const int down = kernel->uploadGroupedExpertDownDescriptorTable(&descriptors[2], 1, width, width);
        ASSERT_GE(gateup, 0); ASSERT_GE(down, 0);
        const auto tensor = [&](int rows)
        {
            auto result = TestTensorFactory::createFP32({size_t(rows), size_t(width)});
            std::fill_n(result->mutable_data(), result->numel(), 0.f);
            require(result->ensureOnDevice(device, stream), "expert test tensor setup");
            return result;
        };
        auto hidden = tensor(capacity), output = tensor(capacity);
        auto gate = tensor(1), up = tensor(1), serial_output = tensor(1);
        auto count_storage = tensor(1);
        auto *count = static_cast<int32_t *>(count_storage->gpu_data_ptr());
        ITensor *gates[]{gate.get()}, *ups[]{up.get()};
        const int ids[]{0}; const float weights[]{1.f};
        ASSERT_TRUE(kernel->prepareGroupedTableDecodeLaunchState(ids, weights,
            gateup, down, 1, gates, ups, serial_output.get(), width, width));
        std::vector<float> input(capacity * width);
        const std::array<float, 8> extrema{0.f, std::numeric_limits<float>::denorm_min(),
            std::numeric_limits<float>::min(), 0x1p-24f, 0x1p-20f, .125f, -0.f, 1.f};
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = (i % 2 ? -1.f : 1.f) * extrema[(i / 32) % blocks];
        for (int lane = 0; lane < 32; ++lane)
            for (int row = 0; row < capacity; ++row)
                input[row * width + 3 * 32 + lane] = q8ScaleUnderflowSourceValue(lane);
        require(backend->hostToDevice(hidden->gpu_data_ptr(), input.data(), input.size() * sizeof(float),
            device.ordinal, stream), "expert original input upload");
        const auto poison = [&](const char *bytes_name, const char *scales_name)
        {
            return backend->memset(workspace.getBuffer(bytes_name), 0x5a,
                workspace.getBufferSize(bytes_name), device.ordinal, stream) &&
                backend->memset(workspace.getBuffer(scales_name), 0x5a,
                workspace.getBufferSize(scales_name), device.ordinal, stream);
        };
        const auto serial = [&]
        {
            return poison(MoEWorkspaceBuffers::DECODE_HIDDEN_INT8, MoEWorkspaceBuffers::DECODE_HIDDEN_SCALES) &&
                poison(MoEWorkspaceBuffers::DECODE_SWIGLU_INT8, MoEWorkspaceBuffers::DECODE_SWIGLU_SCALES) &&
                kernel->groupedExpertGateUpDecodeFromTable(hidden.get(), ids, gateup, 1,
                gates, ups, width, width) && kernel->groupedExpertDownDecodeFromTable(
                gates, ups, ids, weights, down, 1, serial_output.get(), width, width);
        };
        auto serial_graph = context.createGraphCapture(stream);
        ASSERT_NE(serial_graph, nullptr);
        ObservationJoin serial_join{backend, device, stream};
        require(serial(), "expert serial warmup");
        require(backend->synchronizeStream(stream, device.ordinal), "expert serial setup join");
        {
            ScopedBackendGraphCapture recording(context, *serial_graph, "expert serial Q8 range");
            require(recording.begin(), "expert serial capture begin");
            require(serial(), "expert serial capture body");
            recording.finish();
        }
        require(serial_graph->instantiate() && serial_graph->launch(), "expert serial replay");
        ASSERT_NO_FATAL_FAILURE(checkPublication(backend, device, stream, workspace,
            MoEWorkspaceBuffers::DECODE_HIDDEN_INT8, MoEWorkspaceBuffers::DECODE_HIDDEN_SCALES,
            std::vector<float>(input.begin(), input.begin() + width)));
        std::vector<float> serial_gate(width), serial_up(width), serial_mid(width), serial_result(width);
        require(backend->deviceToHost(serial_result.data(), serial_output->gpu_data_ptr(), width * sizeof(float), device.ordinal, stream) &&
            backend->deviceToHost(serial_gate.data(), gate->gpu_data_ptr(), width * sizeof(float), device.ordinal, stream) &&
            backend->deviceToHost(serial_up.data(), up->gpu_data_ptr(), width * sizeof(float), device.ordinal, stream) &&
            backend->synchronizeStream(stream, device.ordinal), "expert serial projections observation");
        for (int i = 0; i < width; ++i)
            serial_mid[i] = device_swiglu_contract::swigluValue(serial_gate[i], serial_up[i]);
        ASSERT_NO_FATAL_FAILURE(checkPublication(backend, device, stream, workspace,
            MoEWorkspaceBuffers::DECODE_SWIGLU_INT8, MoEWorkspaceBuffers::DECODE_SWIGLU_SCALES, serial_mid));
        // Shared prefill has one route per source row. The actual retained
        // pipeline must preserve each original row and the serial SwiGLU bytes.
        for (const int rows : {65, capacity})
        {
            const auto grouped = [&]
            {
                return poison(MoEWorkspaceBuffers::PREFILL_A_INT8, MoEWorkspaceBuffers::PREFILL_A_SCALES) &&
                    poison(MoEWorkspaceBuffers::PREFILL_SWIGLU_INT8, MoEWorkspaceBuffers::PREFILL_SWIGLU_SCALES) &&
                    kernel->prepareSharedExpertPrefillGroup(DeviceRowRange::deviceCounted(rows, count)) &&
                    kernel->executeGroupedPrefillPipeline(hidden.get(), output.get(), gateup, down,
                        rows, width, width, 1, 1);
            };
            require(backend->hostToDevice(count, &rows, sizeof(rows), device.ordinal, stream), "prefill count setup");
            require(grouped() && backend->synchronizeStream(stream, device.ordinal), "expert prefill warmup");
            auto graph = context.createGraphCapture(stream);
            ASSERT_NE(graph, nullptr);
            ObservationJoin join{backend, device, stream};
            {
                ScopedBackendGraphCapture recording(context, *graph, "expert prefill Q8 range");
                require(recording.begin(), "expert prefill capture begin");
                require(grouped(), "expert prefill capture body");
                recording.finish();
            }
            require(graph->instantiate(), "expert prefill instantiate");
            for (const int live : {rows, 1, 0, rows})
            {
                require(backend->hostToDevice(count, &live, sizeof(live), device.ordinal, stream), "prefill live input");
                require(graph->launch() && backend->synchronizeStream(stream, device.ordinal), "expert prefill replay");
                std::vector<float> actual_output(rows * width);
                require(backend->deviceToHost(actual_output.data(), output->gpu_data_ptr(),
                    actual_output.size() * sizeof(float), device.ordinal, stream) &&
                    backend->synchronizeStream(stream, device.ordinal), "prefill result observation");
                for (int row = 0; row < rows; ++row)
                    for (int col = 0; col < width; ++col)
                    {
                        ASSERT_TRUE(std::isfinite(actual_output[row * width + col]));
                        ASSERT_EQ(std::bit_cast<uint32_t>(actual_output[row * width + col]),
                            std::bit_cast<uint32_t>(row < live ? serial_result[col] : 0.f))
                            << "row=" << row << " col=" << col << " live=" << live;
                    }
                if (live == 0) continue;
                ASSERT_NO_FATAL_FAILURE(checkPublication(backend, device, stream, workspace,
                    MoEWorkspaceBuffers::PREFILL_A_INT8, MoEWorkspaceBuffers::PREFILL_A_SCALES,
                    std::vector<float>(input.begin(), input.begin() + live * width)));
                std::vector<float> expected_mid(live * width);
                for (int row = 0; row < live; ++row)
                    std::copy(serial_mid.begin(), serial_mid.end(), expected_mid.begin() + row * width);
                ASSERT_NO_FATAL_FAILURE(checkPublication(backend, device, stream, workspace,
                    MoEWorkspaceBuffers::PREFILL_SWIGLU_INT8, MoEWorkspaceBuffers::PREFILL_SWIGLU_SCALES, expected_mid));
            }
        }
        // A separate captured public down producer receives controlled gate/up
        // projections. gate=128 has exact sigmoid1; no transcendental tolerance
        // can hide loss of a subnormal post-SwiGLU maximum.
        std::vector<float> controlled_gate(width, 128.f), controlled_up(input.begin(), input.begin() + width);
        for (int i = 0; i < width; ++i)
            serial_mid[i] = device_swiglu_contract::swigluValue(controlled_gate[i], controlled_up[i]);
        require(backend->hostToDevice(gate->gpu_data_ptr(), controlled_gate.data(), width * sizeof(float), device.ordinal, stream) &&
            backend->hostToDevice(up->gpu_data_ptr(), controlled_up.data(), width * sizeof(float), device.ordinal, stream), "controlled projections upload");
        auto controlled_graph = context.createGraphCapture(stream);
        ASSERT_NE(controlled_graph, nullptr);
        ObservationJoin controlled_join{backend, device, stream};
        {
            ScopedBackendGraphCapture recording(context, *controlled_graph, "controlled SwiGLU Q8 range");
            require(recording.begin(), "controlled SwiGLU capture begin");
            require(poison(MoEWorkspaceBuffers::DECODE_SWIGLU_INT8, MoEWorkspaceBuffers::DECODE_SWIGLU_SCALES) &&
                kernel->groupedExpertDownDecodeFromTable(
                    gates, ups, ids, weights, down, 1, serial_output.get(), width, width),
                "controlled SwiGLU capture body");
            recording.finish();
        }
        require(controlled_graph->instantiate(), "controlled SwiGLU instantiate");
        for (int replay = 0; replay < 4; ++replay)
        {
            require(controlled_graph->launch(), "controlled SwiGLU replay");
            ASSERT_NO_FATAL_FAILURE(checkPublication(backend, device, stream, workspace,
                MoEWorkspaceBuffers::DECODE_SWIGLU_INT8, MoEWorkspaceBuffers::DECODE_SWIGLU_SCALES, serial_mid));
        }
    });
}
#ifdef HAVE_CUDA
/** @test CUDA expert-owned subnormal publications survive captured decode/prefill. */
TEST(MoEExpertQ8ScaleRange, CUDA) { proveExpertScaleRange(DeviceId::cuda(0)); }
#endif
#ifdef HAVE_ROCM
/** @test ROCm expert-owned publications retain the identical source-value contract. */
TEST(MoEExpertQ8ScaleRange, ROCm) { proveExpertScaleRange(DeviceId::rocm(0)); }
#endif
} // namespace
