/**
 * @file Test__SharedExpertColumnGate.cpp
 * @brief Captured byte-equivalence of column gating and the existing separate gate/add path.
 *
 * Every small grouped row count, scalar/vector output geometry, and degree up
 * to eight sees twenty changing-input replays. Negative/empty/partial/oversized
 * live counts exercise the existing clamping contract; poisoned output tails
 * detect writes beyond the declared geometry. This functional preflight has
 * no timing assertions or synthetic performance policy.
 * Exact zero-dot witnesses independently prove subnormal products and sums;
 * neither optimized implementation supplies the other's expected bit words.
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
#include "kernels/common/DeviceFP32NumericalContract.h"
#include "tensors/TensorKernels.h"
#include "transfer/TransferEngine.h"
#include "utils/TestTensorFactory.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
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
    bool pending = true;
    /** @brief Keep borrowed graph inputs alive until queued work completes. */
    ~Retire() { if (pending) (void)gpu.synchronizeStreamChecked(stream); }
    /** @brief Disarm only after the caller's checked terminal join succeeds. */
    void completed() noexcept { pending = false; }
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

/**
 * @brief Exact binary32 witnesses for sigmoid(0) * shared followed by addition.
 *
 * Zero dot makes the gate exactly 0.5 without depending on vendor exp accuracy.
 * Constants distinguish subnormal inputs/results, signed zero and RNE ties.
 * Products and sums are specified independently of every production helper.
 */
struct HalfGateWords
{
    std::uint32_t shared;
    std::uint32_t routed;
    std::uint32_t product;
    std::uint32_t sum;
};
constexpr std::array kHalfGateWords{
    HalfGateWords{0x00800000u, 0x00400000u, 0x00400000u, 0x00800000u},
    HalfGateWords{0x80800000u, 0x80400000u, 0x80400000u, 0x80800000u},
    HalfGateWords{0x00000002u, 0x00000001u, 0x00000001u, 0x00000002u},
    HalfGateWords{0x80000002u, 0x80000001u, 0x80000001u, 0x80000002u},
    HalfGateWords{0x00000003u, 0x80000001u, 0x00000002u, 0x00000001u},
    HalfGateWords{0x80000003u, 0x00000001u, 0x80000002u, 0x80000001u},
    HalfGateWords{0x00000001u, 0x00000003u, 0x00000000u, 0x00000003u},
    HalfGateWords{0x80000001u, 0x80000003u, 0x80000000u, 0x80000003u},
    HalfGateWords{0x01000000u, 0x807fffffu, 0x00800000u, 0x00000001u}};

/**
 * @brief Prove standalone/full-fused/column products and real ResidualAdd.
 * @param device Explicit selected GPU backend and ordinal.
 * @param gpu Worker owning capture and retirement.
 * @param stream Exact non-null producer/consumer stream.
 * @param rows Retained row count, including partial/empty replay coverage.
 * @param width Full input width; the fixture owns half its output columns.
 * @param normal Cancellation witness with an observed unit-input gate factor.
 * @throws std::runtime_error on an admission, transfer or capture failure.
 */
void proveHalfGate(DeviceId device, IWorkerGPUContext &gpu, void *stream, int rows, int width, bool normal = false)
{
    constexpr float poison = -1234.5f;
    const int columns = width / 2;
    const std::size_t full = std::size_t(rows) * width, local = std::size_t(rows) * columns;
    // The separate ROCm odd-extent regression owns ResidualAdd scalar tails.
    // These exact full-width witnesses isolate arithmetic with a vector-aligned
    // full extent, while width6/columns3 exercises scalar gate addressing.
    ASSERT_EQ(full % 4, 0u);
    auto *backend = getBackendFor(device);
    const auto tensor = [&](std::size_t count) {
        auto value = TestTensorFactory::createFP32({count + 16});
        std::fill_n(value->mutable_data(), value->numel(), 0.f);
        require(value->ensureOnDevice(device, stream), "half gate admission");
        return value;
    };
    auto input = tensor(full), gate = tensor(width), shared = tensor(local), routed = tensor(local), combined = tensor(local);
    auto full_shared = tensor(full), full_routed = tensor(full), full_combined = tensor(full);
    auto fused_shared = tensor(full), fused_combined = tensor(full);
    auto active = TestTensorFactory::createINT32({1});
    require(active->ensureOnDevice(device, stream), "half gate active admission");
    auto moe = llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
    auto residual = llaminar::v2::kernels::KernelFactory::createResidualAdd(full_shared.get(), device.type, device.ordinal);
    moe->setGPUStream(stream); residual->setGPUStream(stream);
    WorkspaceRequirements requirements;
    MoEWorkspaceBuffers::add(requirements, MoEWorkspaceBuffers::ROCM_SHARED_GATE, rows * sizeof(float));
    DeviceWorkspaceManager workspace(device, requirements.total_bytes_with_alignment());
    require(workspace.allocate(requirements), "half gate workspace");
    dynamic_cast<IWorkspaceConsumer *>(moe.get())->bindWorkspace(&workspace);
    SharedExpertColumnGate binding{.input = static_cast<const float *>(input->gpu_data_ptr()),
        .gate = static_cast<const float *>(gate->gpu_data_ptr()), .shared = static_cast<float *>(shared->gpu_data_ptr()),
        .routed = static_cast<const float *>(routed->gpu_data_ptr()), .combined = static_cast<float *>(combined->gpu_data_ptr()),
        .rows = rows, .model_columns = width, .local_columns = columns,
        .active_rows = static_cast<const int *>(active->gpu_data_ptr())};
    require(gpu.synchronizeStreamChecked(stream), "half gate setup join");
    for (auto *value : {input.get(), gate.get(), full_shared.get(), full_routed.get(), fused_shared.get()})
        TransferEngine::requireDeviceInput(value, device, stream);
    float normal_gate = 0.f;
    if (normal)
    {
        // Measure only the common scalar sigmoid coefficient using shared=1.
        // The subsequent product/sum oracle is independent binary32 arithmetic;
        // no vendor exp approximation is assumed to equal the host exp program.
        std::vector<float> x(full + 16, 0.f), g(width + 16, 0.f), unit(full + 16, 1.f), factors(full + 16);
        for (int row = 0; row < rows; ++row) x[std::size_t(row) * width] = 1.f;
        g[0] = 1.f;
        Retire setup_retire{gpu, stream};
        for (const auto &[value, data] : std::array{
            std::pair<ITensor *, const std::vector<float> *>{input.get(), &x},
            std::pair<ITensor *, const std::vector<float> *>{gate.get(), &g},
            std::pair<ITensor *, const std::vector<float> *>{full_shared.get(), &unit}})
            require(backend->hostToDeviceOnStream(value->gpu_data_ptr(), data->data(), data->size() * sizeof(float),
                device.ordinal, stream), "normal gate coefficient input");
        moe->sharedExpertGateFromTensors(input.get(), gate.get(), full_shared.get(), rows, width);
        require(backend->deviceToHostOnStream(factors.data(), full_shared->gpu_data_ptr(), factors.size() * sizeof(float),
            device.ordinal, stream), "normal gate coefficient observation");
        require(gpu.synchronizeStreamChecked(stream), "normal gate coefficient terminal join");
        setup_retire.completed();
        normal_gate = factors[0];
        ASSERT_TRUE(std::isnormal(normal_gate));
        ASSERT_GT(normal_gate, .5f); ASSERT_LT(normal_gate, 1.f);
        ASSERT_NE(std::bit_cast<std::uint32_t>(normal_gate) & 0x007fffffu, 0u);
        for (std::size_t i = 0; i < full; ++i) ASSERT_EQ(factors[i], normal_gate);
        // The eager coefficient probe publishes a new producer event for this
        // tensor. Join that publication on the exact consumer stream before
        // recording; the checked host observation alone does not publish this
        // event edge into the transfer authority's capture contract.
        TransferEngine::requireDeviceInput(full_shared.get(), device, stream);
    }
    auto capture = gpu.createGraphCapture(stream);
    Retire retire{gpu, stream};
    {
        ScopedBackendGraphCapture recording(gpu, *capture, "shared_half_gate_independent_words");
        require(recording.begin(), "half gate capture begin");
        require(gateSharedExpertColumnsFP32(device, binding, stream), "half column gate capture");
        require(moe->sharedExpertGateFromTensorsEffectiveSeqLen(input.get(), gate.get(), full_shared.get(),
            rows, width, binding.active_rows), "half standalone gate capture");
        require(residual->apply_tensor(full_shared.get(), full_routed.get(), full_combined.get(), full,
            nullptr, device.ordinal), "half separate residual capture");
        require(moe->sharedExpertGateAddFromTensorsEffectiveSeqLen(input.get(), gate.get(), fused_shared.get(),
            full_routed.get(), fused_combined.get(), rows, width, binding.active_rows), "half full fusion capture");
        recording.finish();
    }
    require(capture->instantiate(), "half gate instantiate");
    const auto upload = [&](ITensor *value, const std::vector<float> &data) {
        require(backend->hostToDeviceOnStream(value->gpu_data_ptr(), data.data(), data.size() * sizeof(float),
            device.ordinal, stream), "half gate upload");
    };
    for (int replay = 0; replay < 20; ++replay)
    {
        SCOPED_TRACE("independent half gate rows=" + std::to_string(rows) + "/width=" +
            std::to_string(width) + "/replay=" + std::to_string(replay));
        const int live = replay % 5 == 0 ? -1 : replay % 5 == 1 ? 0 : replay % 5 == 2 ? (rows + 1) / 2
            : replay % 5 == 3 ? rows : rows + 7;
        const int owner = replay % 2;
        std::vector<float> values(full + 16, poison), residuals(full + 16, poison);
        std::vector<float> local_values(local + 16, poison), local_residuals(local + 16, poison);
        const std::vector<float> full_poison(full + 16, poison), local_poison(local + 16, poison);
        std::vector<float> got_shared(local + 16), got_combined(local + 16), got_full(full + 16), got_sum(full + 16);
        std::vector<float> got_fused(full + 16), got_fused_sum(full + 16);
        for (int row = 0; row < rows; ++row)
            for (int col = 0; col < width; ++col)
            {
                const auto index = std::size_t(row) * width + col;
                const auto &words = kHalfGateWords[(index + replay) % kHalfGateWords.size()];
                values[index] = normal ? std::bit_cast<float>((index + replay) % 2 ? 0xbf800001u : 0x3f800001u)
                    : std::bit_cast<float>(words.shared);
                // ResidualAdd has no active-row scalar. Its preceding gate
                // clears inactive rows, and this fixture supplies zero residual
                // there so the composed inactive result is independently zero.
                residuals[index] = row < std::clamp(live, 0, rows)
                    ? (normal ? -device_fp32_contract::multiply(normal_gate, values[index])
                        : std::bit_cast<float>(words.routed)) : 0.f;
            }
        for (int row = 0; row < rows; ++row)
            for (int col = 0; col < columns; ++col)
            {
                local_values[std::size_t(row) * columns + col] = values[std::size_t(row) * width + owner * columns + col];
                local_residuals[std::size_t(row) * columns + col] = residuals[std::size_t(row) * width + owner * columns + col];
            }
        // Declared after every borrowed DMA host buffer. On any enqueue/read
        // failure this joins before those vectors unwind, not merely at the
        // outer graph retirement boundary.
        Retire replay_retire{gpu, stream};
        upload(shared.get(), local_values); upload(routed.get(), local_residuals);
        upload(full_shared.get(), values); upload(fused_shared.get(), values); upload(full_routed.get(), residuals);
        // Borrowed H2D sources remain alive until this replay's terminal join.
        upload(combined.get(), local_poison);
        upload(full_combined.get(), full_poison);
        upload(fused_combined.get(), full_poison);
        require(backend->hostToDeviceOnStream(active->gpu_data_ptr(), &live, sizeof(live), device.ordinal, stream)
            && capture->launch(), "half gate retained replay");
        const auto read = [&](ITensor *value, std::vector<float> &data) {
            require(backend->deviceToHostOnStream(data.data(), value->gpu_data_ptr(), data.size() * sizeof(float),
                device.ordinal, stream), "half gate terminal observation");
        };
        read(shared.get(), got_shared); read(combined.get(), got_combined);
        read(full_shared.get(), got_full); read(full_combined.get(), got_sum);
        read(fused_shared.get(), got_fused); read(fused_combined.get(), got_fused_sum);
        require(gpu.synchronizeStreamChecked(stream), "half gate terminal join");
        replay_retire.completed();
        for (int row = 0; row < rows; ++row)
            for (int col = 0; col < width; ++col)
            {
                const auto index = std::size_t(row) * width + col;
                const auto &words = kHalfGateWords[(index + replay) % kHalfGateWords.size()];
                const bool valid = row < std::clamp(live, 0, rows);
                const auto product = valid ? (normal ? std::bit_cast<std::uint32_t>(
                    device_fp32_contract::multiply(normal_gate, values[index])) : words.product) : 0u;
                const auto sum = valid && !normal ? words.sum : 0u;
                if (valid && normal)
                {
                    ASSERT_TRUE(std::isnormal(values[index]));
                    ASSERT_TRUE(std::isnormal(residuals[index]));
                    ASSERT_NE(std::fma(normal_gate, values[index], residuals[index]), 0.f)
                        << "normal witness must distinguish one FMA from two RNE operations";
                }
                ASSERT_EQ(std::bit_cast<std::uint32_t>(got_full[index]), product);
                ASSERT_EQ(std::bit_cast<std::uint32_t>(got_sum[index]), sum);
                ASSERT_EQ(std::bit_cast<std::uint32_t>(got_fused[index]), product);
                ASSERT_EQ(std::bit_cast<std::uint32_t>(got_fused_sum[index]), sum);
                if (col >= owner * columns && col < (owner + 1) * columns)
                {
                    const auto local_index = std::size_t(row) * columns + col - owner * columns;
                    ASSERT_EQ(std::bit_cast<std::uint32_t>(got_shared[local_index]), product);
                    ASSERT_EQ(std::bit_cast<std::uint32_t>(got_combined[local_index]), sum);
                }
            }
        for (const auto *data : {&got_shared, &got_combined, &got_full, &got_sum, &got_fused, &got_fused_sum})
            for (std::size_t index = data->size() - 16; index < data->size(); ++index)
                ASSERT_EQ((*data)[index], poison);
    }
}

/** @brief Execute the exact-word captured witnesses in scalar/vector branches. */
void runHalfGate(DeviceId device)
{
    auto &gpu = GPUDeviceContextPool::instance().getContext(device);
    gpu.submitAndWait([&] {
        void *stream = gpu.getOrCreateAuxiliaryStream("shared_half_gate_regression");
        proveHalfGate(device, gpu, stream, 2, 6);
        proveHalfGate(device, gpu, stream, 3, 12);
        proveHalfGate(device, gpu, stream, 9, 16);
        proveHalfGate(device, gpu, stream, 2, 6, true);
        proveHalfGate(device, gpu, stream, 9, 16, true);
    });
}

/** @brief Prove CPU standalone and fused products against the independent words. */
void runHalfGateCPU()
{
    for (bool normal : {false, true})
    for (int rows : {1, 9})
    {
        constexpr int width = 16;
        auto input = TestTensorFactory::createFP32Zeros({std::size_t(rows), width});
        auto gate = TestTensorFactory::createFP32Zeros({width});
        auto shared = TestTensorFactory::createFP32({std::size_t(rows), width});
        auto fused = TestTensorFactory::createFP32({std::size_t(rows), width});
        auto routed = TestTensorFactory::createFP32({std::size_t(rows), width});
        auto combined = TestTensorFactory::createFP32({std::size_t(rows), width});
        auto fused_combined = TestTensorFactory::createFP32({std::size_t(rows), width});
        auto moe = llaminar::v2::kernels::KernelFactory::createMoEKernel(DeviceId::cpu());
        auto residual = llaminar::v2::kernels::KernelFactory::createResidualAdd(shared.get(), DeviceType::CPU);
        float normal_gate = 0.f;
        if (normal)
        {
            for (int row = 0; row < rows; ++row) input->mutable_data()[std::size_t(row) * width] = 1.f;
            gate->mutable_data()[0] = 1.f;
            std::fill_n(shared->mutable_data(), shared->numel(), 1.f);
            moe->sharedExpertGateFromTensors(input.get(), gate.get(), shared.get(), rows, width);
            normal_gate = shared->data()[0];
            ASSERT_GT(normal_gate, .5f); ASSERT_LT(normal_gate, 1.f);
            ASSERT_NE(std::bit_cast<std::uint32_t>(normal_gate) & 0x007fffffu, 0u);
        }
        for (std::size_t i = 0; i < shared->numel(); ++i)
        {
            const auto &words = kHalfGateWords[i % kHalfGateWords.size()];
            shared->mutable_data()[i] = fused->mutable_data()[i] = normal
                ? std::bit_cast<float>(i % 2 ? 0xbf800001u : 0x3f800001u) : std::bit_cast<float>(words.shared);
            routed->mutable_data()[i] = normal ? -device_fp32_contract::multiply(normal_gate, shared->data()[i])
                : std::bit_cast<float>(words.routed);
            if (normal) ASSERT_NE(std::fma(normal_gate, shared->data()[i], routed->data()[i]), 0.f);
        }
        moe->sharedExpertGateFromTensors(input.get(), gate.get(), shared.get(), rows, width);
        require(residual->apply_tensor(shared.get(), routed.get(), combined.get(), shared->numel()), "CPU half residual");
        moe->sharedExpertGateAddFromTensors(input.get(), gate.get(), fused.get(), routed.get(), fused_combined.get(), rows, width);
        for (std::size_t i = 0; i < shared->numel(); ++i)
        {
            const auto &words = kHalfGateWords[i % kHalfGateWords.size()];
            const auto product = normal ? std::bit_cast<std::uint32_t>(-routed->data()[i]) : words.product;
            const auto sum = normal ? 0u : words.sum;
            ASSERT_EQ(std::bit_cast<std::uint32_t>(shared->data()[i]), product);
            ASSERT_EQ(std::bit_cast<std::uint32_t>(combined->data()[i]), sum);
            ASSERT_EQ(std::bit_cast<std::uint32_t>(fused->data()[i]), product);
            ASSERT_EQ(std::bit_cast<std::uint32_t>(fused_combined->data()[i]), sum);
        }
    }
}
}
TEST(SharedExpertColumnGate, SubnormalPublicationCPU) { runHalfGateCPU(); }
#ifdef HAVE_CUDA
TEST(SharedExpertColumnGate, CUDA) { run(DeviceId::cuda(0)); }
TEST(SharedExpertColumnGate, SubnormalPublicationCUDA) { runHalfGate(DeviceId::cuda(0)); }
#endif
#ifdef HAVE_ROCM
TEST(SharedExpertColumnGate, ROCm) { run(DeviceId::rocm(0)); }
TEST(SharedExpertColumnGate, SubnormalPublicationROCm) { runHalfGate(DeviceId::rocm(0)); }
#endif
}
