/**
 * @file Test__ROCmVerifierEpilogueReplay.cpp
 * @brief Retained grouped projection proof for beta and mapped destinations.
 *
 * Independent production M1 calls own the numerical oracle. The grouped graph
 * must contain a grouped producer and retain the original device count through
 * its final combine/copy, including empty work and repeated dirty replay.
 */
#include <gtest/gtest.h>
#include "kernels/rocm/gemm/ROCmNativeQuantisedVerifierLaunch.h"
#include "kernels/rocm/gemm/ROCmNativeQuantisedArchitecture.h"

#include <hip/hip_runtime.h>
#include "backends/GPUDeviceContextPool.h"
#include "backends/rocm/HIPGraphCapture.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "kernels/rocm/gemm/ROCmQuantisedEpilogues.h"
#include "kernels/common/ProjectionRowWork.h"
#include "transfer/TransferEngine.h"
#include "../../../utils/GpuPreparedGemmHarness.h"
#include "../../../utils/QuantizedVerifierFormats.h"
#include "../../../utils/ScopedGPUStream.h"
#include "../../../utils/TestTensorFactory.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;

extern "C" bool rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
    const int8_t *, const int8_t *, float *, const float *, const float *,
    int, int, float, float, const float *, const float *, int, void *);

namespace
{
/** @brief Physical output placement is independent of logical live rows. */
enum class OutputPlacement { Device, Mapped };
/** @brief Exercise both existing verifier scope constructors. */
enum class RowAdmission { Fixed, Counted, Requests };

/** @brief Retire queued fixture work before graphs, operands and host staging. */
struct FixtureStreamDrain
{
    hipStream_t stream;
    /** @brief Failure paths must preserve the same owner lifetime as successful replay. */
    ~FixtureStreamDrain()
    {
        if (hipStreamSynchronize(stream) != hipSuccess) std::terminate();
    }
};

/**
 * @brief Require one grouped producer and an exact row-admitted epilogue.
 * @tparam M Physical rows of the original matrix subview.
 * @param placement Persistent output storage contract, retained across replays.
 * @param admission Whether a device word owns the current live prefix.
 * @param beta Existing-output contribution, including the mapped-copy case.
 * @tparam M Physical view rows. @tparam RequestRows Original request capacity.
 * @tparam Origin Original view origin.
 * @param candidate Captured grouped producer; independent M1 oracle is unchanged.
 */
template<int M = 3, int RequestRows = 3, int Origin = 1>
void proveVerifierEpilogue(OutputPlacement placement, RowAdmission admission, float beta,
    rocm::NativeQuantisedVerifierCandidate candidate = rocm::NativeQuantisedVerifierCandidate::Automatic)
{
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0)
        GTEST_SKIP() << "No ROCm device available";
    constexpr int N = 17, K = 32;
    constexpr float alpha = 0.75f;
    const auto device = DeviceId::rocm(0);
    ScopedGPUStream stream_owner(device);
    const auto stream = static_cast<hipStream_t>(stream_owner.get());
    if((candidate==rocm::NativeQuantisedVerifierCandidate::WmmaM16N64) &&
        rocm::nativeQuantisedHardwareForDevice(0).architecture!=native_quantised::NativeQuantisedArchitecture::rocm(11,0,0))
        GTEST_SKIP() << "Native verifier WMMA requires actual gfx1100";

    const auto &formats = quantizedVerifierFormats();
    const auto format = std::find_if(formats.begin(), formats.end(),
        [](const auto &entry) { return entry.tensor_type == TensorType::Q4_0; });
    ASSERT_NE(format, formats.end());
    auto weights = format->create({N, K}, 73101);
    auto prepared = makeGpuPreparedGemm(weights.get(), device,
        "test.verifier.epilogue", ModelContextId{73102});
    auto *kernel = dynamic_cast<rocm::ROCmQuantisedGemmKernel *>(prepared.kernel);
    ASSERT_NE(kernel, nullptr);
    kernel->setGPUStream(stream);
    auto requirements = kernel->getWorkspaceRequirements(M, N, K);
    auto workspace = std::make_unique<DeviceWorkspaceManager>(
        device, requirements.total_bytes_with_alignment());
    ASSERT_TRUE(workspace->allocate(requirements));
    kernel->bindWorkspace(workspace.get());
    ASSERT_TRUE(kernel->prepareFusedProjectionGraphCapture(1));

    auto input = TestTensorFactory::createFP32Random({M, K}, -0.75f, 0.75f, 73103);
    auto output = placement == OutputPlacement::Mapped
        ? FP32Tensor::createMapped({M, N}, device)
        : TestTensorFactory::createFP32({M, N});
    auto serial_input = TestTensorFactory::createFP32({1, K});
    auto serial_output = TestTensorFactory::createFP32({1, N});
    auto bias = TestTensorFactory::createFP32Random({N}, -0.125f, 0.125f, 73104);
    auto count = TestTensorFactory::createINT32({admission == RowAdmission::Requests ? 2u : 1u});
    std::array<float, M * N> initial{}, oracle{};
    std::array<std::uint32_t, M * N> actual{};
    std::array<std::int32_t, 2> published_counts{M, 0};
    std::unique_ptr<IGPUGraphCapture> graph;
    const FixtureStreamDrain drain{stream};
    ASSERT_NE(output, nullptr);
    ASSERT_TRUE(input->ensureOnDevice(device, stream));
    ASSERT_TRUE(bias->ensureOnDevice(device, stream));
    if (placement == OutputPlacement::Device)
        ASSERT_TRUE(output->allocateOnDevice(device, stream));
    ASSERT_TRUE(serial_input->allocateOnDevice(device, stream));
    ASSERT_TRUE(serial_output->allocateOnDevice(device, stream));
    ASSERT_TRUE(count->allocateOnDevice(device, stream));

    for (int i = 0; i < M * N; ++i)
        initial[i] = 0.03125f * static_cast<float>((i % 19) - 9);
    // The oracle never enters a verifier scope and never calls a grouped API.
    for (int row = 0; row < M; ++row)
    {
        ASSERT_EQ(hipMemcpyAsync(serial_input->gpu_data_ptr(),
            static_cast<const float *>(input->gpu_data_ptr()) + row * K,
            K * sizeof(float), hipMemcpyDeviceToDevice, stream), hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(serial_output->gpu_data_ptr(), initial.data() + row * N,
            N * sizeof(float), hipMemcpyHostToDevice, stream), hipSuccess);
        TransferEngine::publishCurrentDeviceWrite(serial_input.get(), stream);
        TransferEngine::publishCurrentDeviceWrite(serial_output.get(), stream);
        ASSERT_TRUE(kernel->multiply_tensor(serial_input.get(), serial_output.get(),
            1, N, K, true, alpha, beta, bias.get()));
        ASSERT_EQ(hipMemcpyAsync(oracle.data() + row * N, serial_output->gpu_data_ptr(),
            N * sizeof(float), hipMemcpyDeviceToHost, stream), hipSuccess);
    }
    ASSERT_EQ(hipMemcpyAsync(output->gpu_data_ptr(), initial.data(), sizeof(initial),
        hipMemcpyHostToDevice, stream), hipSuccess);
    TransferEngine::publishCurrentDeviceWrite(output.get(), stream);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    TransferEngine::requireDeviceInput(input.get(), device, stream);
    TransferEngine::requireDeviceInput(bias.get(), device, stream);
    TransferEngine::requireDeviceInput(output.get(), device, stream);
    const auto rows = DeviceRowRange::deviceCounted(M,
        static_cast<const std::int32_t *>(count->gpu_data_ptr()));
    const auto work = admission == RowAdmission::Requests
        ? ProjectionRowWork::fromRequests(DeviceRequestRowRanges::deviceCounted(
              2, RequestRows, static_cast<const std::int32_t *>(count->gpu_data_ptr()))).slice(Origin, M)
        : ProjectionRowWork::fromPrefix(rows);
    graph = GPUDeviceContextPool::instance().retainAMDContext(0)->createGraphCapture(stream);
    {
        std::optional<ProjectionRowWorkScope> work_scope;
        if (admission == RowAdmission::Requests) work_scope.emplace(work);
        auto arithmetic = admission == RowAdmission::Counted
            ? kernel->beginVerifierDecodeEquivalentScope(rows)
            : kernel->beginVerifierDecodeEquivalentScope();
        rocm::ScopedNativeQuantisedVerifierCandidate candidate_scope(candidate,0);
        ScopedBackendGraphCapture capture(*graph, "verifier_epilogue_replay");
        ASSERT_TRUE(capture.begin());
        ASSERT_TRUE(kernel->multiply_tensor(input.get(), output.get(),
            M, N, K, true, alpha, beta, bias.get()));
        capture.finish();
        std::vector<GPUGraphKernelNodeInfo> nodes;
        std::string error;
        ASSERT_TRUE(graph->inspectKernelNodes(nodes, &error)) << error;
        EXPECT_EQ(std::count_if(nodes.begin(), nodes.end(), [&](const auto &node) {
            return node.name.find((candidate==rocm::NativeQuantisedVerifierCandidate::WmmaM16N64)
                ? "native_quantised_verifier_wmma_m16n64" : "gemv_native_quantised_small_m_kernel_t") != std::string::npos;
        }), 1) << "A verifier projection must record its grouped producer exactly once";
        EXPECT_EQ(std::count_if(nodes.begin(), nodes.end(), [](const auto &node) {
            return node.name.find("gemv_native_quantised_kernel_t") != std::string::npos;
        }), 0) << "Serial M1 repair must never coexist with the grouped producer";
        if((candidate==rocm::NativeQuantisedVerifierCandidate::WmmaM16N64))
            for(const auto &node:nodes) {
                EXPECT_EQ(node.name.find("gemv_native_quantised_small_m_kernel_t"),std::string::npos);
                if(node.name.find("native_quantised_verifier_wmma_m16n64")!=std::string::npos) {
                    ASSERT_TRUE(node.valid()); EXPECT_EQ(node.block_x,128u);
                    EXPECT_EQ(node.block_y,1u); EXPECT_EQ(node.block_z,1u);
                    EXPECT_EQ(node.grid_x,1u); EXPECT_EQ(node.grid_y,1u); EXPECT_EQ(node.grid_z,1u);
                }
            }
        EXPECT_EQ(std::count_if(nodes.begin(),nodes.end(),[](const auto &node) {
            return node.name.find("native_quantised_verifier_ordered_m4n64")!=std::string::npos;
        }), 0);
        if (admission != RowAdmission::Fixed)
        {
            auto *native = dynamic_cast<HIPGraphCapture *>(graph.get());
            ASSERT_NE(native, nullptr);
            std::size_t count_nodes = 0;
            ASSERT_EQ(hipGraphGetNodes(native->graph(), nullptr, &count_nodes), hipSuccess);
            std::vector<hipGraphNode_t> handles(count_nodes);
            ASSERT_EQ(hipGraphGetNodes(native->graph(), handles.data(), &count_nodes), hipSuccess);
            int combines = 0;
            for (auto node : handles)
            {
                hipGraphNodeType type{};
                ASSERT_EQ(hipGraphNodeGetType(node, &type), hipSuccess);
                if (type != hipGraphNodeTypeKernel) continue;
                hipKernelNodeParams params{};
                ASSERT_EQ(hipGraphKernelNodeGetParams(node, &params), hipSuccess);
                const char *name = hipKernelNameRefByPtr(params.func, stream);
                ASSERT_NE(name, nullptr);
                if (std::string(name).find("applyFp32CombineEpilogue_kernel") == std::string::npos) continue;
                ++combines;
                ASSERT_NE(params.kernelParams, nullptr);
                ASSERT_EQ(params.extra, nullptr);
                ASSERT_NE(params.kernelParams[4], nullptr);
                ASSERT_NE(params.kernelParams[8], nullptr);
                int captured_m = 0;
                std::memcpy(&captured_m, params.kernelParams[4], sizeof(captured_m));
                EXPECT_EQ(captured_m, M);
                if (admission == RowAdmission::Requests)
                {
                    ASSERT_NE(std::string(name).find("ProjectionRowWork"), std::string::npos);
                    std::array<std::byte, sizeof(ProjectionRowWork)> bytes;
                    std::memcpy(bytes.data(), params.kernelParams[8], bytes.size());
                    EXPECT_EQ(std::bit_cast<ProjectionRowWork>(bytes).captureIdentity(), work.captureIdentity());
                }
                else
                {
                    ASSERT_NE(std::string(name).find("DeviceRowRange"), std::string::npos);
                    std::array<std::byte, sizeof(DeviceRowRange)> bytes;
                    std::memcpy(bytes.data(), params.kernelParams[8], bytes.size());
                    EXPECT_EQ(std::bit_cast<DeviceRowRange>(bytes).captureIdentity(), rows.captureIdentity());
                }
            }
            EXPECT_EQ(combines, 1);
        }
        ASSERT_TRUE(graph->instantiate());
    }
    const std::vector<std::array<std::int32_t, 2>> publications = admission == RowAdmission::Requests
        ? (RequestRows == 3
            ? std::vector<std::array<std::int32_t, 2>>{{3,3},{1,2},{0,0},{2,1},{0,3},{3,0},{3,3}}
            : std::vector<std::array<std::int32_t, 2>>{{5,4},{6,5},{5,4},{5,5},{6,4},{0,0},{8,8},{5,4}})
        : std::vector<std::array<std::int32_t, 2>>{{M,0},{1,0},{0,0},{2,0},{M,0},{0,0},{M,0}};
    for (const auto &publication : publications)
    {
        if (admission == RowAdmission::Fixed && publication[0] != M) continue;
        SCOPED_TRACE("live=" + std::to_string(publication[0]) + "," + std::to_string(publication[1]));
        for (const auto *buffer : {GemmWorkspaceBuffers::TEMP_C_FP32,
                 GemmWorkspaceBuffers::QUANT_A, GemmWorkspaceBuffers::SCALES_A_BLOCKWISE,
                 GemmWorkspaceBuffers::SUMS_A_BLOCKWISE, GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL})
            ASSERT_EQ(hipMemsetAsync(workspace->getBuffer(buffer), 0xa5,
                workspace->getBufferSize(buffer), stream), hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(output->gpu_data_ptr(), initial.data(), sizeof(initial),
            hipMemcpyHostToDevice, stream), hipSuccess);
        published_counts = publication;
        const auto count_bytes = (admission == RowAdmission::Requests ? 2 : 1) * sizeof(std::int32_t);
        ASSERT_EQ(hipMemcpyAsync(count->gpu_data_ptr(), published_counts.data(), count_bytes,
            hipMemcpyHostToDevice, stream), hipSuccess);
        ASSERT_TRUE(graph->launch());
        ASSERT_EQ(hipMemcpyAsync(actual.data(), output->gpu_data_ptr(), sizeof(actual),
            hipMemcpyDeviceToHost, stream), hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        for (int i = 0; i < M * N; ++i)
        {
            // Resolve the original request coordinate independently from the
            // view origin and request capacity. This oracle derives directly from
            // authored counts rather than reusing the producer's row cursor.
            const int original_row = Origin + i / N;
            const bool active = admission == RowAdmission::Requests
                ? original_row % RequestRows < publication[original_row / RequestRows]
                : i / N < publication[0];
            ASSERT_EQ(actual[i], std::bit_cast<std::uint32_t>(active ? oracle[i] : initial[i]))
                << "output index=" << i;
        }
    }
    graph.reset();
    kernel->unbindWorkspace();
    kernel->clearGPUStreamBinding();
}
}

/** @test Retained grouped beta/bias must preserve device-owned inactive rows. */
TEST(ROCmVerifierEpilogueReplay, DeviceOutput)
{
    for (auto admission : {RowAdmission::Fixed, RowAdmission::Counted})
        ASSERT_NO_FATAL_FAILURE(proveVerifierEpilogue(OutputPlacement::Device, admission, -0.25f));
}

/** @test Mapped output requires the same grouped arithmetic and live-byte extent. */
TEST(ROCmVerifierEpilogueReplay, MappedOutput)
{
    for (auto admission : {RowAdmission::Fixed, RowAdmission::Counted})
        for (float beta : {0.0f, -0.25f})
            ASSERT_NO_FATAL_FAILURE(proveVerifierEpilogue(OutputPlacement::Mapped, admission, beta));
}

/** @test Beta combines preserve request-local holes and a nonzero original view origin. */
TEST(ROCmVerifierEpilogueReplay, RequestHolesAndSubview)
{
    for (auto placement : {OutputPlacement::Device, OutputPlacement::Mapped})
        ASSERT_NO_FATAL_FAILURE(proveVerifierEpilogue<4>(placement, RowAdmission::Requests, -0.25f));
}

/** @test Copy-only mapped publication preserves signed zero, subnormals and NaN payload bits. */
TEST(ROCmVerifierEpilogueReplay, MappedCopyBits)
{
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0)
        GTEST_SKIP() << "No ROCm device available";
    constexpr int M = 3, N = 17;
    constexpr std::array<std::uint32_t, 8> patterns{
        0u, 0x80000000u, 1u, 0x80000001u, 0x007fffffu, 0x807fffffu,
        0x3f800000u, 0x7fc01234u};
    const auto device = DeviceId::rocm(0);
    ScopedGPUStream stream_owner(device);
    const auto stream = static_cast<hipStream_t>(stream_owner.get());
    auto computed = TestTensorFactory::createFP32({M, N});
    auto output = FP32Tensor::createMapped({M, N}, device);
    auto count = TestTensorFactory::createINT32({1});
    std::array<std::uint32_t, M * N> words{}, actual{};
    std::int32_t published_live = M;
    std::unique_ptr<IGPUGraphCapture> graph;
    const FixtureStreamDrain drain{stream};
    ASSERT_NE(output, nullptr);
    ASSERT_TRUE(computed->allocateOnDevice(device, stream));
    ASSERT_TRUE(count->allocateOnDevice(device, stream));
    const auto rows = DeviceRowRange::deviceCounted(M,
        static_cast<const std::int32_t *>(count->gpu_data_ptr()));
    graph = GPUDeviceContextPool::instance().retainAMDContext(0)->createGraphCapture(stream);
    {
        ScopedBackendGraphCapture capture(*graph, "mapped_copy_bits");
        ASSERT_TRUE(capture.begin());
        ASSERT_TRUE(rocmQuantGemm_applyFp32CombineEpilogue(
            static_cast<const float *>(computed->gpu_data_ptr()),
            static_cast<float *>(output->gpu_data_ptr()), nullptr, nullptr,
            M, N, 1.0f, 0.0f, 0, stream, &rows));
        capture.finish();
        ASSERT_TRUE(graph->instantiate());
    }
    int epoch = 0;
    for (const std::int32_t live : {3, 1, 0, 2, 3, 0, 3})
    {
        for (std::size_t i = 0; i < words.size(); ++i)
            words[i] = patterns[(i + epoch) % patterns.size()];
        ++epoch;
        published_live = live;
        ASSERT_EQ(hipMemcpyAsync(computed->gpu_data_ptr(), words.data(), sizeof(words),
            hipMemcpyHostToDevice, stream), hipSuccess);
        ASSERT_EQ(hipMemsetAsync(output->gpu_data_ptr(), 0xa5, sizeof(actual), stream), hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(count->gpu_data_ptr(), &published_live, sizeof(published_live),
            hipMemcpyHostToDevice, stream), hipSuccess);
        ASSERT_TRUE(graph->launch());
        ASSERT_EQ(hipMemcpyAsync(actual.data(), output->gpu_data_ptr(), sizeof(actual),
            hipMemcpyDeviceToHost, stream), hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        for (int i = 0; i < M * N; ++i)
            ASSERT_EQ(actual[i], i < live * N ? words[i] : 0xa5a5a5a5u)
                << "live=" << live << " index=" << i;
    }
    graph.reset();
}

/** @brief Native row cursors retain request holes, subview origin and mapped beta bytes. */
TEST(ROCmVerifierWmmaEpilogueReplay, RequestHolesAndSubview)
{
    for(auto placement:{OutputPlacement::Device,OutputPlacement::Mapped})
        ASSERT_NO_FATAL_FAILURE(proveVerifierEpilogue<4>(placement,RowAdmission::Requests,-0.25f,
            rocm::NativeQuantisedVerifierCandidate::WmmaM16N64));
}
/** @brief Native P1 standalone publication retains fixed/counted alpha-beta arithmetic. */
TEST(ROCmVerifierWmmaEpilogueReplay, FixedAndCounted)
{
    for(auto admission:{RowAdmission::Fixed,RowAdmission::Counted})
        for(auto placement:{OutputPlacement::Device,OutputPlacement::Mapped})
            ASSERT_NO_FATAL_FAILURE(proveVerifierEpilogue(placement,admission,-0.25f,
                rocm::NativeQuantisedVerifierCandidate::WmmaM16N64));
}


/** @brief Native replay preserves four/five-row intervals across independent request origins. */
TEST(ROCmVerifierWmmaEpilogueReplay, NativeRequestIntervalInterleaving)
{
    for (auto placement : {OutputPlacement::Device, OutputPlacement::Mapped})
        ASSERT_NO_FATAL_FAILURE((proveVerifierEpilogue<14,8,1>(placement,
            RowAdmission::Requests, -0.25f, rocm::NativeQuantisedVerifierCandidate::WmmaM16N64)));
}

/** @brief Replay scalar beta publication into retained device C for every source. */
TEST(ROCmScalarGEMVEpilogueReplay, DeviceDestinationOwnsBetaPublication)
{
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0)
        GTEST_SKIP() << "No ROCm device available";
    constexpr int N = 17, K = 256;
    constexpr float beta = 0.25f;
    const auto device = DeviceId::rocm(0);
    ScopedGPUStream owner(device);
    const auto stream = static_cast<hipStream_t>(owner.get());
    for (const auto &format : quantizedVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        auto weights = format.create({N, K}, 73201);
        auto prepared = makeGpuPreparedGemm(weights.get(), device,
            "test.scalar.gemv.beta", ModelContextId{73202});
        auto *kernel = dynamic_cast<rocm::ROCmQuantisedGemmKernel *>(prepared.kernel);
        ASSERT_NE(kernel, nullptr);
        kernel->setGPUStream(stream);
        auto requirements = kernel->getWorkspaceRequirements(1, N, K);
        DeviceWorkspaceManager workspace(device, requirements.total_bytes_with_alignment());
        ASSERT_TRUE(workspace.allocate(requirements));
        kernel->bindWorkspace(&workspace);
        const auto release_binding = [](rocm::ROCmQuantisedGemmKernel *bound) { bound->unbindWorkspace(); };
        const std::unique_ptr<rocm::ROCmQuantisedGemmKernel, decltype(release_binding)> binding(kernel, release_binding);
        ASSERT_TRUE(kernel->prepareFusedProjectionGraphCapture(1));

        auto input = TestTensorFactory::createFP32Random({1, K}, -0.75f, 0.75f, 73203);
        auto output = TestTensorFactory::createFP32({1, N});
        std::array<float, N> projection{}, initial{}, actual{};
        HIPGraphCapture graph(stream, 0);
        const FixtureStreamDrain drain{stream};
        ASSERT_TRUE(input->ensureOnDevice(device, stream));
        ASSERT_TRUE(output->allocateOnDevice(device, stream));
        ASSERT_FALSE(output->isMapped());
        ASSERT_TRUE(kernel->multiply_tensor(input.get(), output.get(), 1, N, K));
        ASSERT_EQ(hipMemcpyAsync(projection.data(), output->gpu_data_ptr(), sizeof(projection),
            hipMemcpyDeviceToHost, stream), hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        // The oracle's publication and the input upload both precede the
        // captured consumer. Join their retained events on its exact stream.
        TransferEngine::requireDeviceInput(input.get(), device, stream);
        TransferEngine::requireDeviceInput(output.get(), device, stream);
        {
            ScopedBackendGraphCapture capture(graph, "scalar_gemv_beta_destination");
            ASSERT_TRUE(capture.begin());
            ASSERT_TRUE(kernel->multiply_tensor(input.get(), output.get(), 1, N, K, true, 1.0f, beta));
            capture.finish();
        }

        std::size_t count = 0;
        ASSERT_EQ(hipGraphGetNodes(graph.graph(), nullptr, &count), hipSuccess);
        std::vector<hipGraphNode_t> nodes(count);
        ASSERT_EQ(hipGraphGetNodes(graph.graph(), nodes.data(), &count), hipSuccess);
        int combinations = 0;
        for (auto node : nodes)
        {
            hipGraphNodeType type{};
            ASSERT_EQ(hipGraphNodeGetType(node, &type), hipSuccess);
            EXPECT_NE(type, hipGraphNodeTypeMemcpy);
            if (type != hipGraphNodeTypeKernel)
                continue;
            hipKernelNodeParams parameters{};
            ASSERT_EQ(hipGraphKernelNodeGetParams(node, &parameters), hipSuccess);
            const char *name = hipKernelNameRefByPtr(parameters.func, stream);
            ASSERT_NE(name, nullptr);
            if (std::string_view(name).find("applyFp32CombineEpilogue_kernel") == std::string_view::npos)
                continue;
            ++combinations;
            ASSERT_NE(parameters.kernelParams, nullptr);
            void *destination = nullptr;
            std::memcpy(&destination, parameters.kernelParams[1], sizeof(destination));
            EXPECT_EQ(destination, output->gpu_data_ptr());
        }
        EXPECT_EQ(combinations, 1);
        ASSERT_TRUE(graph.instantiate());
        for (const float sign : {1.0f, -1.0f, 1.0f})
        {
            for (int column = 0; column < N; ++column)
                initial[column] = sign * static_cast<float>(column - N / 2) / 32.0f;
            ASSERT_EQ(hipMemcpyAsync(output->gpu_data_ptr(), initial.data(), sizeof(initial),
                hipMemcpyHostToDevice, stream), hipSuccess);
            ASSERT_TRUE(graph.launch());
            ASSERT_EQ(hipMemcpyAsync(actual.data(), output->gpu_data_ptr(), sizeof(actual),
                hipMemcpyDeviceToHost, stream), hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            for (int column = 0; column < N; ++column)
                EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[column]),
                    std::bit_cast<std::uint32_t>(projection[column] + beta * initial[column]));
        }
    }
}

/** @test Execute captured INT8 beta publication against the same disjoint original state. */
TEST(ROCmScalarVNNIBetaPublicationReplay, ExistingValuesAndFinalDeviceRowCanAlias)
{
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0)
        GTEST_SKIP() << "No ROCm device available";
    const auto device = DeviceId::rocm(0);
    ScopedGPUStream owner(device);
    const auto stream = static_cast<hipStream_t>(owner.get());
    const auto free_device = [](void *pointer) { (void)hipFree(pointer); };
    const auto upload = [&]<typename T>(const std::vector<T> &values) {
        T *pointer = nullptr;
        const auto bytes = values.size() * sizeof(T);
        if (hipMalloc(reinterpret_cast<void **>(&pointer), bytes) != hipSuccess)
            throw std::runtime_error("INT8 GEMV replay could not allocate its fixture buffer");
        std::unique_ptr<T, decltype(free_device)> buffer(pointer, free_device);
        if (hipMemcpyAsync(pointer, values.data(), bytes, hipMemcpyHostToDevice, stream) != hipSuccess)
            throw std::runtime_error("INT8 GEMV replay could not load its recorded fixture");
        return buffer;
    };
    // Odd N, square, LDS split-K, and wide dispatches. Exact binary
    // coefficients make the old split contribution and its beta seed exact.
    const std::array geometries{std::pair{17, 256}, std::pair{257, 256},
        std::pair{269, 768}, std::pair{384, 512}, std::pair{512, 32}};
    for (const auto &[n, k] : geometries)
    {
        SCOPED_TRACE("N=" + std::to_string(n) + " K=" + std::to_string(k));
        std::vector<std::int8_t> activation(k), weights(static_cast<std::size_t>(n) * k);
        std::vector<float> scales_a(k / 32, 0.25f), scales_b(n, 0.125f), bias(n);
        std::vector<float> old(n + 8, -782.0f), control(old.size()), actual(old.size());
        for (int j = 0; j < k; ++j)
            activation[j] = static_cast<std::int8_t>((j * 5 + 1) % 11 - 5);
        for (std::size_t j = 0; j < weights.size(); ++j)
            weights[j] = static_cast<std::int8_t>(static_cast<int>((j * 3 + 7) % 13) - 6);
        for (int j = 0; j < n; ++j)
            bias[j] = static_cast<float>(j % 7 - 3) * 0.0625f;

        const auto d_a = upload(activation);
        const auto d_b = upload(weights);
        const auto d_a_scale = upload(scales_a);
        const auto d_b_scale = upload(scales_b);
        const auto d_bias = upload(bias);
        const auto d_existing = upload(old);
        const auto d_control = upload(old);
        const auto d_alias = upload(old);

        HIPGraphCapture graph(stream, 0);
        const FixtureStreamDrain drain{stream};
        const auto launch = [&](float *destination, const float *existing) {
            return rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                d_a.get(), d_b.get(), destination, d_a_scale.get(), d_b_scale.get(),
                n, k, 0.5f, 0.25f, existing, d_bias.get(), 0, stream);
        };
        {
            ScopedBackendGraphCapture capture(graph, "int8_scalar_beta_on_final_device_row");
            ASSERT_TRUE(capture.begin());
            ASSERT_TRUE(launch(d_alias.get(), d_alias.get()));
            capture.finish();
        }
        std::size_t count = 0;
        ASSERT_EQ(hipGraphGetNodes(graph.graph(), nullptr, &count), hipSuccess);
        ASSERT_GT(count, 0u);
        std::vector<hipGraphNode_t> nodes(count);
        ASSERT_EQ(hipGraphGetNodes(graph.graph(), nodes.data(), &count), hipSuccess);
        for (auto node : nodes)
        {
            hipGraphNodeType type{};
            ASSERT_EQ(hipGraphNodeGetType(node, &type), hipSuccess);
            EXPECT_NE(type, hipGraphNodeTypeMemcpy);
        }
        ASSERT_TRUE(graph.instantiate());
        const auto bytes = old.size() * sizeof(float);
        for (const float sign : {1.0f, -1.0f, 1.0f})
        {
            for (int column = 0; column < n; ++column)
                old[column] = sign * static_cast<float>(column % 23 - 11) * 0.0625f;
            ASSERT_EQ(hipMemcpyAsync(d_existing.get(), old.data(), bytes, hipMemcpyHostToDevice, stream), hipSuccess);
            ASSERT_EQ(hipMemcpyAsync(d_alias.get(), old.data(), bytes, hipMemcpyHostToDevice, stream), hipSuccess);
            ASSERT_TRUE(launch(d_control.get(), d_existing.get()));
            ASSERT_TRUE(graph.launch());
            ASSERT_EQ(hipMemcpyAsync(control.data(), d_control.get(), bytes, hipMemcpyDeviceToHost, stream), hipSuccess);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), d_alias.get(), bytes, hipMemcpyDeviceToHost, stream), hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            EXPECT_EQ(std::memcmp(actual.data(), control.data(), bytes), 0);
            // Two device executions could both miss columns in a partial N
            // tile. Also check every live column against the recorded weights.
            for (int column = 0; column < n; ++column)
            {
                int dot = 0;
                for (int inner = 0; inner < k; ++inner)
                {
                    const auto packed = (static_cast<std::size_t>(inner / 4) * n + column) * 4 + inner % 4;
                    dot += activation[inner] * weights[packed];
                }
                const float expected = static_cast<float>(dot) * (0.25f * 0.125f * 0.5f)
                    + 0.25f * old[column] + bias[column];
                EXPECT_EQ(actual[column], expected) << "column " << column;
            }
            for (std::size_t column = n; column < old.size(); ++column)
                EXPECT_EQ(actual[column], old[column]);
        }
    }
}
