/**
 * @file Test__NativeQ8SerialGeometry.cpp
 * @brief Captured full-output and partial-plane proof for fixed-arithmetic Q8 geometry.
 *
 * Real production preparation supplies each native-Q8 descriptor. The existing
 * serial entrypoint then consumes exactly represented Q8 activations and its
 * ordinary generated arithmetic policy. Only independent column ownership is
 * changed. Every captured replay poisons all live partial/output bytes first;
 * full-buffer comparison and untouched guards expose missing writes, CPT tails
 * and K640's six empty partition planes. Resources and captured native symbols
 * are authenticated independently of the host construction receipt. No timing
 * result or installed dispatch winner is produced by this focused regression.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/rocm/HIPGraphCapture.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/common/DeviceNativeVNNIMatrixDesc.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
#include "kernels/rocm/gemm/ROCmNativeVNNISerialGeometry.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "tensors/TensorClasses.h"
#include "transfer/TransferEngine.h"
#include "utils/GpuPreparedGemmHarness.h"
#include "utils/TestTensorFactory.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

/** @brief Existing source-aware production serial entrypoint; no test implementation. */
extern "C" bool rocmGemv_native_vnni_fp32_with_policy(
    const int8_t *, const uint8_t *, const void *, const void *, const void *,
    float *, const float *, float *, int, int, uint8_t, uint8_t, int, void *, const float *);
/** @brief Existing thread-local arithmetic-width lifecycle, separate from physical N. */
extern "C" void rocmGemv_native_vnni_set_serial_partition_n(int);
/** @brief Read the prior arithmetic-width binding for exact scope restoration. */
extern "C" int rocmGemv_native_vnni_get_serial_partition_n();

namespace llaminar2::test
{
namespace
{
using namespace llaminar2::rocm;
constexpr std::uint32_t kGuardBits = 0x7fc12345u;
constexpr int kGuardElements = 16;

/**
 * @brief Fail a fixture prerequisite before any later output can masquerade as a pass.
 * @param success Required operation result.
 * @param operation Stable diagnostic context.
 * @throws std::runtime_error when the operation failed.
 */
void require(bool success, const char *operation)
{
    if (!success) throw std::runtime_error(operation);
}

/**
 * @brief Prove a rejected public call recorded no native node of any kind.
 * @param graph Completed capture of only the rejected operation.
 * @throws std::runtime_error if native graph enumeration fails or finds work.
 */
void requireNoCapturedWork(const HIPGraphCapture &graph)
{
    std::size_t count = 1;
    require(hipGraphGetNodes(graph.graph(), nullptr, &count) == hipSuccess,
        "native-Q8 rejected-call native graph enumeration failed");
    require(count == 0 && graph.nodeCount() == 0,
        "native-Q8 rejected public call enqueued work");
}

/** @brief Restore the arithmetic-only N binding even if construction fails. */
class ArithmeticWidthScope final
{
public:
    /** @brief Set an intentionally different policy N without changing physical storage. */
    explicit ArithmeticWidthScope(int n) : previous_(rocmGemv_native_vnni_get_serial_partition_n())
    { rocmGemv_native_vnni_set_serial_partition_n(n); }
    /** @brief Restore the exact prior thread-local arithmetic binding. */
    ~ArithmeticWidthScope() { rocmGemv_native_vnni_set_serial_partition_n(previous_); }
    ArithmeticWidthScope(const ArithmeticWidthScope &) = delete;
    ArithmeticWidthScope &operator=(const ArithmeticWidthScope &) = delete;
private:
    int previous_;
};

/** @brief Retain fixture device work until graph/storage owners can retire safely. */
struct ObservationJoin
{
    IBackend *backend;
    DeviceId device;
    void *stream;
    /** @brief Join only at fixture teardown, never in captured execution. */
    ~ObservationJoin() { (void)backend->synchronizeStream(stream, device.ordinal); }
};

/** @brief Entire live output/partial bytes plus both allocation guards. */
struct Snapshot
{
    NativeQ8SerialLaunch launch;
    std::vector<std::uint32_t> output;
    std::vector<std::uint32_t> partials;
};

/**
 * @brief Compare a construction receipt with independent native graph inspection.
 * @param graph Completed native capture, inspected outside recording/timing.
 * @param receipt Actual typed launch observation from the same construction thread.
 * @param only_native Whether capture deliberately contains no activation preparation.
 * @throws std::runtime_error for a missing or mismatched symbol/grid/block/resource.
 */
void authenticateGraph(const HIPGraphCapture &graph, const NativeQ8SerialLaunch &receipt,
    bool only_native = true)
{
    std::vector<GPUGraphKernelNodeInfo> nodes;
    std::string error;
    require(graph.inspectKernelNodes(nodes, &error), "native-Q8 graph inspection failed");
    if (only_native)
        require(nodes.size() == (receipt.reducer.function ? 2u : 1u), "native-Q8 capture lost a kernel");
    for (const auto *launch : {&receipt.producer, &receipt.reducer})
    {
        if (!launch->function) continue;
        const auto match = std::find_if(nodes.begin(), nodes.end(), [&](const auto &node)
        { return node.function_identity == reinterpret_cast<std::uintptr_t>(launch->function); });
        require(match != nodes.end(), "native-Q8 captured symbol differs from launch receipt");
        require(match->valid() && match->local_memory_bytes_per_thread == 0 &&
            match->grid_x == static_cast<std::uint32_t>(launch->grid_x) &&
            match->grid_y == static_cast<std::uint32_t>(launch->grid_y) &&
            match->grid_z == static_cast<std::uint32_t>(launch->grid_z) &&
            match->block_x == static_cast<std::uint32_t>(launch->block_x) &&
            match->block_y == static_cast<std::uint32_t>(launch->block_y) &&
            match->block_z == static_cast<std::uint32_t>(launch->block_z),
            "native-Q8 captured geometry/resources differ from receipt");
    }
}

/**
 * @brief Prove the canonical public prepared call accepts scoped geometry/default flags.
 * @param device Actual device owning source weights and workspace.
 * @param stream Exact producer stream retained until fixture retirement.
 * @param admission Already authenticated nonbaseline geometry.
 * @throws std::runtime_error for preparation/capture/complete-byte failures.
 */
void provePreparedDefault(DeviceId device, void *stream,
    const NativeQ8SerialGeometryAdmission &admission)
{
    constexpr int n = 129, k = 640;
    auto *backend = getBackendFor(device);
    auto source = TestTensorFactory::createQ8_0Random({n, k}, 8282);
    auto prepared = makeGpuPreparedGemm(source.get(), device, "native_q8.serial_geometry.public_default");
    auto *kernel = dynamic_cast<ROCmQuantisedGemmKernel *>(prepared.kernel);
    require(kernel != nullptr, "native-Q8 public fixture needs the real prepared ROCm kernel");
    const auto requirements = kernel->getWorkspaceRequirements(1, n, k);
    DeviceWorkspaceManager workspace(device, requirements.total_bytes_with_alignment());
    require(workspace.allocate(requirements), "native-Q8 public workspace admission");
    auto input = TestTensorFactory::createFP32Random({1, k}, -0.25f, 0.25f, 82);
    auto output = TestTensorFactory::createFP32({1, n});
    TransferEngine::prepareDeviceInput(input.get(), device, stream);
    TransferEngine::requireDeviceInput(input.get(), device, stream);
    TransferEngine::prepareDeviceInput(output.get(), device, stream);
    TransferEngine::requireDeviceInput(output.get(), device, stream);
    kernel->setGPUStream(stream);
    kernel->bindWorkspace(&workspace);
    /** Join before borrowed workspace/tensors retire, including exception paths. */
    struct Binding
    {
        ROCmQuantisedGemmKernel *kernel;
        IBackend *backend;
        DeviceId device;
        void *stream;
        ~Binding()
        {
            (void)backend->synchronizeStream(stream, device.ordinal);
            kernel->unbindWorkspace();
            kernel->clearGPUStreamBinding();
        }
    } binding{kernel, backend, device, stream};
    // Both public overloads default transpose_B=true; the implementation
    // ignores that historical flag. A geometry scope must preserve that ABI.
    require(kernel->multiply_tensor(input.get(), output.get(), 1, n, k), "native-Q8 public unscoped primer");
    std::vector<std::uint32_t> baseline(n), candidate(n);
    require(backend->deviceToHost(baseline.data(), output->gpu_data_ptr(), n * sizeof(float),
        device.ordinal, stream) && backend->synchronizeStream(stream, device.ordinal),
        "native-Q8 public baseline observation");
    HIPGraphCapture graph(static_cast<hipStream_t>(stream), device.ordinal);
    NativeQ8SerialLaunch receipt;
    {
        ScopedNativeQ8SerialGeometry scope(admission);
        for (const auto [rows, alpha] : std::array{std::pair{2, 1.0f}, std::pair{1, 2.0f}})
        {
            // Seed a real successful receipt before EACH rejection. The
            // rejected transaction must clear it and record zero native nodes.
            require(kernel->multiply_tensor(input.get(), output.get(), 1, n, k) &&
                backend->synchronizeStream(stream, device.ordinal), "native-Q8 public rejection primer");
            require(lastNativeQ8SerialLaunch().has_value(), "native-Q8 rejection primer omitted receipt");
            HIPGraphCapture rejected_graph(static_cast<hipStream_t>(stream), device.ordinal);
            ScopedBackendGraphCapture rejected_capture(rejected_graph,
                "native-Q8 incompatible public call records no work");
            require(rejected_capture.begin(), "native-Q8 rejected-call capture begin");
            EXPECT_THROW((void)kernel->multiply_tensor(input.get(), output.get(), rows, n, k, true, alpha),
                std::logic_error);
            EXPECT_FALSE(lastNativeQ8SerialLaunch().has_value());
            rejected_capture.finish();
            requireNoCapturedWork(rejected_graph);
        }
        for (void *bad_stream : {static_cast<void *>(hipStreamLegacy),
                 static_cast<void *>(hipStreamPerThread)})
        {
            require(kernel->multiply_tensor(input.get(), output.get(), 1, n, k) &&
                backend->synchronizeStream(stream, device.ordinal), "native-Q8 public bad-stream rejection primer");
            require(lastNativeQ8SerialLaunch().has_value(), "native-Q8 bad-stream primer omitted receipt");
            kernel->setGPUStream(bad_stream);
            HIPGraphCapture rejected_graph(static_cast<hipStream_t>(stream), device.ordinal);
            ScopedBackendGraphCapture rejected_capture(rejected_graph,
                "native-Q8 default stream rejection precedes activation quantization");
            require(rejected_capture.begin(), "native-Q8 bad-stream rejection capture begin");
            EXPECT_THROW((void)kernel->multiply_tensor(input.get(), output.get(), 1, n, k), std::logic_error);
            EXPECT_FALSE(lastNativeQ8SerialLaunch().has_value());
            rejected_capture.finish();
            requireNoCapturedWork(rejected_graph);
            kernel->setGPUStream(stream);
        }
        ScopedBackendGraphCapture capture(graph, "native-Q8 canonical prepared M1 geometry");
        require(capture.begin(), "native-Q8 public capture begin");
        require(hipMemsetAsync(output->gpu_data_ptr(), 0xff, n * sizeof(float),
            static_cast<hipStream_t>(stream)) == hipSuccess, "native-Q8 public captured output poison");
        const bool enqueued = kernel->multiply_tensor(input.get(), output.get(), 1, n, k);
        capture.finish();
        require(enqueued, "native-Q8 public scoped default-flag capture");
        const auto observed = lastNativeQ8SerialLaunch();
        require(observed.has_value(), "native-Q8 public scoped call omitted receipt");
        receipt = *observed;
    }
    authenticateGraph(graph, receipt, false);
    require(graph.instantiate() && graph.launch(), "native-Q8 public replay after scope exit");
    require(backend->deviceToHost(candidate.data(), output->gpu_data_ptr(), n * sizeof(float),
        device.ordinal, stream) && backend->synchronizeStream(stream, device.ordinal),
        "native-Q8 public full-buffer observation");
    EXPECT_EQ(candidate, baseline);
    EXPECT_EQ(receipt.geometry, admission.geometry());
}

/**
 * @brief Prove one physical N/K using unchanged prepared native-Q8 bytes.
 * @param device Exact ROCm endpoint executing this fixture.
 * @param stream Exact non-null endpoint stream retained through graph retirement.
 * @param n Physical output columns, including deliberately odd CPT tails.
 * @param k Full reduction width; 640 exercises empty KB16 planes.
 * @param admissions Pre-capture native resource proofs for all five geometries.
 * @throws std::runtime_error for setup/capture/publication failures.
 */
void proveShape(DeviceId device, void *stream, int n, int k,
    const std::array<NativeQ8SerialGeometryAdmission, 5> &admissions)
{
    auto *backend = getBackendFor(device);
    require(backend && stream, "native-Q8 fixture requires an explicit-stream ROCm endpoint");
    auto source = TestTensorFactory::createQ8_0Random({std::size_t(n), std::size_t(k)}, 821u + n + k);
    auto prepared = makeGpuPreparedGemm(source.get(), device,
        "native_q8.serial_geometry." + std::to_string(n) + "." + std::to_string(k));
    DeviceNativeVNNIMatrixDesc descriptor;
    require(prepared.kernel && prepared.kernel->exportNativeVNNIMatrixDesc(descriptor) &&
        descriptor.valid() && descriptor.codebook_id == 19 && descriptor.n == n && descriptor.k == k,
        "native-Q8 fixture did not prepare the actual native19 descriptor");

    const int kb = MoEProjectionNumericalContract::orderedKPartitionsForWidth(k);
    const std::size_t partial_live = kb > 1 ? std::size_t(kb) * n : 0u;
    const auto floating = [&](std::size_t elements)
    {
        auto value = TestTensorFactory::createFP32({elements});
        std::fill_n(value->mutable_data(), elements, std::bit_cast<float>(kGuardBits));
        TransferEngine::prepareDeviceInput(value.get(), device, stream);
        TransferEngine::requireDeviceInput(value.get(), device, stream);
        return value;
    };
    std::vector<int8_t> activation_bytes(k);
    for (int index = 0; index < k; ++index)
        activation_bytes[index] = static_cast<int8_t>((index * 37 + 11) % 255 - 127);
    auto activations = std::make_unique<INT8Tensor>(std::vector<std::size_t>{std::size_t(k)},
        activation_bytes, 1.0f);
    TransferEngine::prepareDeviceInput(activations.get(), device, stream);
    TransferEngine::requireDeviceInput(activations.get(), device, stream);
    auto scales = floating(k / 32);
    std::vector<float> scale_values(k / 32);
    for (std::size_t index = 0; index < scale_values.size(); ++index)
        scale_values[index] = (index & 1u) ? 0.03125f : 0.0625f;
    require(backend->hostToDevice(scales->gpu_data_ptr(), scale_values.data(),
        scale_values.size() * sizeof(float), device.ordinal, stream), "native-Q8 activation-scale upload");
    auto output = floating(std::size_t(n) + kGuardElements);
    auto partials = floating(partial_live + kGuardElements);
    ObservationJoin shape_join{backend, device, stream};
    // No source topology change: the arithmetic selector sees a synthetic
    // original width, while every actual symbol/grid/stride retains physical n.
    ArithmeticWidthScope policy_n(4096);

    const auto captureAndObserve = [&](const NativeQ8SerialGeometryAdmission *admission) -> Snapshot
    {
        std::optional<ScopedNativeQ8SerialGeometry> geometry_scope;
        if (admission) geometry_scope.emplace(*admission);
        clearNativeQ8SerialLaunch();
        HIPGraphCapture graph(static_cast<hipStream_t>(stream), device.ordinal);
        ScopedBackendGraphCapture capture(graph, "native-Q8 fixed-arithmetic geometry regression");
        require(capture.begin(), "native-Q8 capture begin");
        // Poison live storage on EVERY replay. Guards outside those exact
        // extents keep their distinct payload and must never be overwritten.
        require(hipMemsetAsync(output->gpu_data_ptr(), 0xff, std::size_t(n) * sizeof(float),
            static_cast<hipStream_t>(stream)) == hipSuccess, "native-Q8 captured output poison");
        if (partial_live)
            require(hipMemsetAsync(partials->gpu_data_ptr(), 0xff, partial_live * sizeof(float),
                static_cast<hipStream_t>(stream)) == hipSuccess, "native-Q8 captured partial poison");
        const bool enqueued = rocmGemv_native_vnni_fp32_with_policy(
            static_cast<const int8_t *>(activations->gpu_data_ptr()), descriptor.payload,
            descriptor.scales, descriptor.mins, descriptor.emins,
            static_cast<float *>(output->gpu_data_ptr()), nullptr,
            static_cast<float *>(partials->gpu_data_ptr()), n, k, descriptor.codebook_id,
            descriptor.arithmeticPolicyCodebookId(), device.ordinal, stream,
            static_cast<const float *>(scales->gpu_data_ptr()));
        capture.finish();
        require(enqueued, "native-Q8 captured production launch");
        const auto receipt = lastNativeQ8SerialLaunch();
        require(receipt.has_value(), "native-Q8 capture did not publish an actual launch receipt");
        geometry_scope.reset();
        authenticateGraph(graph, *receipt);
        require(graph.instantiate(), "native-Q8 graph instantiate");
        Snapshot observed{*receipt, std::vector<std::uint32_t>(n + kGuardElements),
            std::vector<std::uint32_t>(partial_live + kGuardElements)};
        std::vector<std::uint32_t> first_output, first_partials;
        for (int replay = 0; replay < 3; ++replay)
        {
            require(graph.launch(), "native-Q8 retained graph replay");
            require(backend->synchronizeStream(stream, device.ordinal), "native-Q8 fixture observation join");
            require(backend->deviceToHost(observed.output.data(), output->gpu_data_ptr(),
                observed.output.size() * sizeof(float), device.ordinal, stream) &&
                backend->deviceToHost(observed.partials.data(), partials->gpu_data_ptr(),
                observed.partials.size() * sizeof(float), device.ordinal, stream) &&
                backend->synchronizeStream(stream, device.ordinal), "native-Q8 full-buffer observation");
            if (replay == 0)
            {
                first_output = observed.output;
                first_partials = observed.partials;
            }
            else require(observed.output == first_output && observed.partials == first_partials,
                "native-Q8 retained replay changed output/partial bytes");
            for (int column = 0; column < n; ++column)
                require(std::isfinite(std::bit_cast<float>(observed.output[column])), "native-Q8 output poison survived");
            for (std::size_t index = 0; index < partial_live; ++index)
                require(std::isfinite(std::bit_cast<float>(observed.partials[index])), "native-Q8 partial poison survived");
            for (std::size_t index = n; index < observed.output.size(); ++index)
                require(observed.output[index] == kGuardBits, "native-Q8 output guard overwritten");
            for (std::size_t index = partial_live; index < observed.partials.size(); ++index)
                require(observed.partials[index] == kGuardBits, "native-Q8 partial guard overwritten");
        }
        return observed;
    };

    const Snapshot baseline = captureAndObserve(nullptr);
    ASSERT_EQ(baseline.launch.geometry, NativeQ8SerialGeometry::TN128CPT1);
    for (const auto &admission : admissions)
    {
        const auto candidate = captureAndObserve(&admission);
        const auto spec = nativeQ8SerialGeometrySpec(admission.geometry());
        EXPECT_EQ(candidate.output, baseline.output);
        EXPECT_EQ(candidate.partials, baseline.partials);
        EXPECT_EQ(candidate.launch.geometry, admission.geometry());
        EXPECT_EQ(candidate.launch.physical_n, n);
        EXPECT_EQ(candidate.launch.arithmetic_policy_n, 4096);
        EXPECT_EQ(candidate.launch.k_partitions, kb);
        EXPECT_EQ(candidate.launch.blocks_per_partition, (k / 32 + kb - 1) / kb);
        EXPECT_EQ(candidate.launch.partial_elements, partial_live);
        EXPECT_EQ(candidate.launch.stream, stream);
        EXPECT_EQ(candidate.launch.producer.function, admission.producerResources().function);
        EXPECT_EQ(candidate.launch.producer.grid_x, (n + spec.tile_columns - 1) / spec.tile_columns);
        EXPECT_EQ(candidate.launch.producer.grid_y, kb);
        EXPECT_EQ(candidate.launch.producer.block_x, spec.block_threads);
        if (kb > 1)
            EXPECT_EQ(candidate.launch.reducer.function, admission.reducerResources().function);
        else EXPECT_EQ(candidate.launch.reducer.function, nullptr);
        if (k == 640)
        {
            ASSERT_EQ(candidate.launch.k_partitions, 16);
            ASSERT_EQ(candidate.launch.blocks_per_partition, 2);
            ASSERT_EQ(candidate.launch.nonempty_partitions, 10);
            for (std::size_t index = std::size_t(10) * n; index < partial_live; ++index)
                EXPECT_EQ(candidate.partials[index], 0u) << "Empty planes must publish positive zero";
        }
    }
}
} // namespace

/** @test Real captured Q8 symbols preserve full outputs and all partial planes. */
TEST(NativeQ8SerialGeometry, CapturedTailsAndEmptyPartitionsAreByteExact)
{
    GPUDeviceContextPool::instance().getAMDContext(0).submitAndWait([]
    {
        const auto device = DeviceId::rocm(0);
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        void *stream = context.defaultStream();
        auto *backend = getBackendFor(device);
        ASSERT_NE(backend, nullptr);
        ASSERT_NE(stream, nullptr);
        ObservationJoin join{backend, device, stream};
        const std::array admissions{
            admitNativeQ8SerialGeometry(NativeQ8SerialGeometry::TN64CPT1, 0),
            admitNativeQ8SerialGeometry(NativeQ8SerialGeometry::TN128CPT1, 0),
            admitNativeQ8SerialGeometry(NativeQ8SerialGeometry::TN256CPT1, 0),
            admitNativeQ8SerialGeometry(NativeQ8SerialGeometry::TN128CPT2, 0),
            admitNativeQ8SerialGeometry(NativeQ8SerialGeometry::TN256CPT2, 0)};
        EXPECT_THROW((void)admitNativeQ8SerialGeometry(static_cast<NativeQ8SerialGeometry>(255), 0),
            std::invalid_argument);
        ASSERT_FALSE(currentNativeQ8SerialGeometry().has_value());
        {
            ScopedNativeQ8SerialGeometry outer(admissions[0]);
            EXPECT_EQ(currentNativeQ8SerialGeometry(), admissions[0].geometry());
            {
                ScopedNativeQ8SerialGeometry inner(admissions[4]);
                EXPECT_EQ(currentNativeQ8SerialGeometry(), admissions[4].geometry());
            }
            EXPECT_EQ(currentNativeQ8SerialGeometry(), admissions[0].geometry());
            bool child_started_unscoped = false;
            bool child_restored = false;
            std::thread child([&]
            {
                child_started_unscoped = !currentNativeQ8SerialGeometry();
                { ScopedNativeQ8SerialGeometry local(admissions[2]); }
                child_restored = !currentNativeQ8SerialGeometry();
            });
            child.join();
            EXPECT_TRUE(child_started_unscoped);
            EXPECT_TRUE(child_restored);
            EXPECT_EQ(currentNativeQ8SerialGeometry(), admissions[0].geometry());
        }
        EXPECT_FALSE(currentNativeQ8SerialGeometry().has_value());
        {
            ScopedNativeQ8SerialGeometry scope(admissions[0]);
            EXPECT_THROW((void)rocmGemv_native_vnni_fp32_with_policy(
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                1, 640, 5, 5, 0, stream, nullptr), std::logic_error);
            EXPECT_FALSE(lastNativeQ8SerialLaunch().has_value());
            EXPECT_THROW((void)rocmGemv_native_vnni_fp32_with_policy(
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                1, 640, 19, 19, 1, stream, nullptr), std::logic_error);
            EXPECT_FALSE(lastNativeQ8SerialLaunch().has_value());
            EXPECT_THROW(validateNativeQ8SerialScopeBinding(1, stream), std::logic_error);
            for (void *bad_stream : {static_cast<void *>(nullptr),
                     static_cast<void *>(hipStreamLegacy), static_cast<void *>(hipStreamPerThread)})
            {
                EXPECT_THROW((void)rocmGemv_native_vnni_fp32_with_policy(
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                    1, 640, 19, 19, 0, bad_stream, nullptr), std::logic_error);
                EXPECT_FALSE(lastNativeQ8SerialLaunch().has_value());
            }
        }
        {
            HIPGraphCapture graph(static_cast<hipStream_t>(stream), 0);
            ScopedBackendGraphCapture capture(graph, "native-Q8 rejects capture-time admission");
            ASSERT_TRUE(capture.begin());
            EXPECT_THROW((void)admitNativeQ8SerialGeometry(NativeQ8SerialGeometry::TN64CPT1, 0),
                std::logic_error);
            EXPECT_THROW((void)ScopedNativeQ8SerialGeometry(admissions[0]), std::logic_error);
            capture.finish();
        }
        ASSERT_NO_FATAL_FAILURE(provePreparedDefault(device, stream, admissions[4]));
        for (const int k : {32, 160, 640, 2560})
            for (const int n : {1, 63, 65, 127, 129, 160, 257})
            {
                SCOPED_TRACE(::testing::Message() << "N=" << n << " K=" << k);
                ASSERT_NO_FATAL_FAILURE(proveShape(device, stream, n, k, admissions));
            }
    });
}
} // namespace llaminar2::test
