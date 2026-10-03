/**
 * @file Test__SharedExpertColumnGate.cpp
 * @brief Captured byte-equivalence of column gating and the existing separate gate/add path.
 *
 * Every small grouped row count, scalar/vector output geometry, and degree up
 * to eight sees twenty changing-input replays. Negative/empty/partial/oversized
 * live counts exercise the existing clamping contract; poisoned output tails
 * detect writes beyond the declared geometry. This functional preflight has
 * no timing assertions or synthetic performance policy.
 */
#include <gtest/gtest.h>
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/KernelFactory.h"
#include "kernels/IMoEKernel.h"
#include "kernels/common/SharedExpertColumnGateKernels.h"
#include "transfer/TransferEngine.h"
#include "utils/TestTensorFactory.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace llaminar2::test
{
namespace
{
/** @brief Preserve unwinding on a failed GPU operation. */
void require(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }

/** @brief Join only at fixture retirement, never from the kernel under test. */
struct Retire
{
    IWorkerGPUContext &gpu;
    void *stream;
    /** @brief Keep borrowed graph inputs alive until queued work completes. */
    ~Retire() { (void)gpu.synchronizeStreamChecked(stream); }
};

/** @brief Compare one geometry with the backend's unchanged full-width arithmetic. */
void proveGeometry(DeviceId device, IWorkerGPUContext &gpu, void *stream, int rows, int width, int degree)
{
    const int columns = width / degree;
    auto *backend = getBackendFor(device);
    const auto tensor = [&](std::size_t count) {
        auto value = TestTensorFactory::createFP32({count + 16});
        std::fill_n(value->mutable_data(), value->numel(), 0.f);
        require(value->ensureOnDevice(device, stream), "column gate admission");
        return value;
    };
    const std::size_t full = std::size_t(rows) * width, local = std::size_t(rows) * columns;
    auto input = tensor(full), gate = tensor(width), shared = tensor(local), routed = tensor(local), output = tensor(local);
    auto reference_shared = tensor(full);
    auto active = TestTensorFactory::createINT32({1});
    require(active->ensureOnDevice(device, stream), "column gate live count admission");
    auto oracle = llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
    oracle->setGPUStream(stream);
    WorkspaceRequirements requirements;
    MoEWorkspaceBuffers::add(requirements, MoEWorkspaceBuffers::ROCM_SHARED_GATE, rows * sizeof(float));
    DeviceWorkspaceManager workspace(device, requirements.total_bytes_with_alignment());
    require(workspace.allocate(requirements), "standalone gate oracle workspace");
    dynamic_cast<IWorkspaceConsumer *>(oracle.get())->bindWorkspace(&workspace);
    SharedExpertColumnGate binding{.input = static_cast<const float *>(input->gpu_data_ptr()),
        .gate = static_cast<const float *>(gate->gpu_data_ptr()),
        .shared = static_cast<float *>(shared->gpu_data_ptr()),
        .routed = static_cast<const float *>(routed->gpu_data_ptr()),
        .combined = static_cast<float *>(output->gpu_data_ptr()),
        .rows = rows, .model_columns = width, .local_columns = columns,
        .active_rows = static_cast<const int *>(active->gpu_data_ptr())};
    // Rejections must not enqueue work or consume a default stream.
    require(!gateSharedExpertColumnsFP32(device, binding, nullptr), "column gate accepted null stream");
    auto invalid = binding; invalid.combined = invalid.shared;
    require(!gateSharedExpertColumnsFP32(device, invalid, stream), "column gate accepted aliased output");
    if (width % 4 == 0)
    {
        invalid = binding; ++invalid.input;
        require(!gateSharedExpertColumnsFP32(device, invalid, stream), "column gate accepted unaligned vector input");
    }
    require(gpu.synchronizeStreamChecked(stream), "column gate setup join");
    for (auto *tensor : {input.get(), gate.get(), reference_shared.get()})
        TransferEngine::requireDeviceInput(tensor, device, stream);
    auto capture = gpu.createGraphCapture(stream);
    Retire retire{gpu, stream};
    {
        ScopedBackendGraphCapture recording(gpu, *capture, "shared_column_gate_oracle");
        require(recording.begin(), "column gate begin capture");
        require(gateSharedExpertColumnsFP32(device, binding, stream), "column gate capture");
        require(oracle->sharedExpertGateFromTensorsEffectiveSeqLen(input.get(), gate.get(), reference_shared.get(),
            rows, width, binding.active_rows), "column gate standalone full-row oracle capture");
        recording.finish();
    }
    require(capture->instantiate(), "column gate instantiate");
    std::vector<float> x(input->numel()), g(gate->numel()), s(reference_shared->numel()), r(reference_shared->numel());
    std::vector<float> local_s(shared->numel()), local_r(routed->numel()), observed(output->numel());
    std::vector<float> observed_s(shared->numel()), expected_s(reference_shared->numel());
    constexpr float poison = -1234.5f;
    const auto upload = [&](ITensor *t, const std::vector<float> &values) {
        require(backend->hostToDeviceOnStream(t->gpu_data_ptr(), values.data(), values.size() * sizeof(float),
            device.ordinal, stream), "column gate input upload");
    };
    for (int replay = 0; replay < 20; ++replay)
    {
        SCOPED_TRACE("rows=" + std::to_string(rows) + "/width=" + std::to_string(width) +
            "/degree=" + std::to_string(degree) + "/replay=" + std::to_string(replay));
        const int owner = replay % degree;
        const int live = replay % 5 == 0 ? -1 : replay % 5 == 1 ? 0 : replay % 5 == 2 ? (rows + 1) / 2
            : replay % 5 == 3 ? rows : rows + 7;
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] = .03125f * (int((i * 31 + replay * 7) % 53) - 26);
        for (std::size_t i = 0; i < g.size(); ++i)
            g[i] = (replay % 4 == 0 ? 8.f : .0625f) * (int((i * 17 + replay * 3) % 19) - 9);
        for (std::size_t i = 0; i < s.size(); ++i)
        {
            s[i] = .0078125f * (int((i * 11 + replay * 19) % 137) - 68);
            r[i] = .015625f * (int((i * 13 + replay * 23) % 107) - 53);
        }
        std::fill(local_s.begin(), local_s.end(), poison);
        std::fill(local_r.begin(), local_r.end(), poison);
        for (int row = 0; row < rows; ++row)
            for (int col = 0; col < columns; ++col)
            {
                local_s[std::size_t(row) * columns + col] = s[std::size_t(row) * width + owner * columns + col];
                local_r[std::size_t(row) * columns + col] = r[std::size_t(row) * width + owner * columns + col];
            }
        std::fill(observed.begin(), observed.end(), poison);
        upload(input.get(), x); upload(gate.get(), g);
        upload(shared.get(), local_s); upload(routed.get(), local_r); upload(output.get(), observed);
        upload(reference_shared.get(), s);
        require(backend->hostToDeviceOnStream(active->gpu_data_ptr(), &live, sizeof(live), device.ordinal, stream)
            && capture->launch(), "column gate retained replay");
        const auto read = [&](ITensor *t, std::vector<float> &values) {
            require(backend->deviceToHostOnStream(values.data(), t->gpu_data_ptr(), values.size() * sizeof(float),
                device.ordinal, stream), "column gate observation");
        };
        read(output.get(), observed);
        read(shared.get(), observed_s); read(reference_shared.get(), expected_s);
        require(gpu.synchronizeStreamChecked(stream), "column gate terminal join");
        for (int row = 0; row < rows; ++row)
            for (int col = 0; col < columns; ++col)
            {
                const auto a = std::size_t(row) * columns + col;
                const auto b = std::size_t(row) * width + owner * columns + col;
                // The existing TP path rounds the product before a distinct
                // residual-add kernel. Host addition of its observed product
                // is the diagnostic oracle for that exact FP32 boundary. A
                // fused-gate/add oracle would miss accidental FMA contraction.
                const bool valid = row < std::clamp(live, 0, rows);
                const float gated = valid ? expected_s[b] : 0.f;
                const float combined = valid ? gated + r[b] : 0.f;
                ASSERT_TRUE(std::isfinite(observed[a]));
                ASSERT_EQ(std::bit_cast<uint32_t>(observed_s[a]), std::bit_cast<uint32_t>(gated));
                ASSERT_EQ(std::bit_cast<uint32_t>(observed[a]), std::bit_cast<uint32_t>(combined));
            }
        for (std::size_t i = local; i < observed.size(); ++i)
        {
            ASSERT_EQ(observed[i], poison);
            ASSERT_EQ(observed_s[i], poison);
        }
    }
}

/** @brief Sweep all grouped rows and representative scalar/vector column geometries. */
void run(DeviceId device)
{
    ASSERT_NE(getBackendFor(device), nullptr);
    auto &gpu = GPUDeviceContextPool::instance().getContext(device);
    gpu.submitAndWait([&] {
        void *stream = gpu.getOrCreateAuxiliaryStream("shared_column_gate_regression");
        for (int width : {6, 258, 512, 2048, 4096})
            for (int degree : {2, 4, 8})
            {
                if (width % degree) continue;
                for (int rows = 1; rows <= 16; ++rows) proveGeometry(device, gpu, stream, rows, width, degree);
                if (width == 2048)
                    for (int rows : {64, 129, 448, 512}) proveGeometry(device, gpu, stream, rows, width, degree);
            }
    });
}
}
#ifdef HAVE_CUDA
TEST(SharedExpertColumnGate, CUDA) { run(DeviceId::cuda(0)); }
#endif
#ifdef HAVE_ROCM
TEST(SharedExpertColumnGate, ROCm) { run(DeviceId::rocm(0)); }
#endif
}
