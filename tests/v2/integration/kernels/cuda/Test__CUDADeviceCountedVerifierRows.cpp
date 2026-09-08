/**
 * @file Test__CUDADeviceCountedVerifierRows.cpp
 * @brief Captured all-codebook proof of device-owned logical row admission.
 *
 * Production preparation and the public serial M1 operation supply the byte
 * oracle. One retained graph exercises shallow, deep, empty, and cross-scratch-
 * tile publications while poisoned inactive outputs must remain untouched.
 * Keeping this focused fixture separate avoids rebuilding the large historical
 * small-M test translation unit during row-lifecycle work.
 * An explicitly selected Perf suite reuses these byte oracles at model shapes;
 * its timing/dispatch sweep is excluded from the preflight registration.
 */
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "backends/cuda/CUDAGraphCapture.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/cuda/gemm/CUDAGroupedVerifierLaunch.h"
#include "kernels/cuda/gemm/CUDAQuantisedGemmKernel.h"
#include "tensors/NativeVnniFormatInfo.h"
#include "transfer/TransferEngine.h"
#include "../../../utils/GpuPreparedGemmHarness.h"
#include "../../../utils/QuantizedVerifierFormats.h"
#include "../../../utils/ScopedGPUStream.h"
#include "../../../utils/TestTensorFactory.h"
#include "../../../utils/DeviceCountedFloatingRowsProof.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <iostream>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

extern "C" bool cudaQuantGemm_quantizeActivationsBlockwise(
    const float *, int8_t *, float *, int, int, int, void *);
extern "C" void cudaNativeVNNIGemvSweep_setGroupedRows(int);
extern "C" void cudaNativeVNNIGroupedVerifier_setTensorCoreOverride(int);

using namespace llaminar2;
using namespace llaminar2::cuda;
using namespace llaminar2::test;

/** @brief One source format per case, also isolatable for kernel profiling. */
class CUDADeviceCountedVerifierRows
    : public ::testing::TestWithParam<QuantizedVerifierFormatCase> {};

/** @brief Exercise raw kernels and both public tensor adapter transactions. */
enum class VerifierRowRoute { Raw, TensorProjection, TensorSwiGLU };

/** @brief Explicit performance experiment; never inferred by functional tests. */
struct VerifierRowEconomyProbe
{
    int n; ///< Exact output columns of the production-shaped projection.
    int k; ///< Exact reduction width, retaining the installed M1 arithmetic.
    int grouped_rows; ///< Candidate reuse width; zero selects installed policy.
};

/** @brief Bound a trainer-only row-reuse override to one recording scope. */
class ScopedVerifierRowCandidate final
{
public:
    /** @brief Select only row reuse; the serial arithmetic policy stays intact. */
    explicit ScopedVerifierRowCandidate(int rows)
    {
        cudaNativeVNNIGemvSweep_setGroupedRows(rows);
    }
    /** @brief Retire the diagnostic selector before another graph is recorded. */
    ~ScopedVerifierRowCandidate() { cudaNativeVNNIGemvSweep_setGroupedRows(0); }
    ScopedVerifierRowCandidate(const ScopedVerifierRowCandidate &) = delete;
    ScopedVerifierRowCandidate &operator=(const ScopedVerifierRowCandidate &) = delete;
};

/** @brief Force the production integer tensor-core branch in a focused proof. */
class ScopedVerifierTensorCoreCandidate final
{
public:
    /** @brief Select tensor cores without changing the inherited M1 arithmetic. */
    ScopedVerifierTensorCoreCandidate()
    {
        cudaNativeVNNIGroupedVerifier_setTensorCoreOverride(1);
    }
    /** @brief Restore generated selection before the next independent test. */
    ~ScopedVerifierTensorCoreCandidate()
    {
        cudaNativeVNNIGroupedVerifier_setTensorCoreOverride(0);
    }
    ScopedVerifierTensorCoreCandidate(const ScopedVerifierTensorCoreCandidate &) = delete;
    ScopedVerifierTensorCoreCandidate &operator=(const ScopedVerifierTensorCoreCandidate &) = delete;
};

/**
 * @brief Compare changing live extents with public serial arithmetic.
 * @param format Complete source-codebook identity and nondegenerate fixture.
 * @param route Capture surface under test; all routes use the same byte oracle.
 * @param probe Optional isolated economy experiment, outside functional gates.
 */
static void proveDeviceCountedRows(
    const QuantizedVerifierFormatCase &format, VerifierRowRoute route,
    const std::optional<VerifierRowEconomyProbe> &probe = std::nullopt)
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        GTEST_SKIP() << "No CUDA device available";
    const int M = probe ? 16 : 31; // Functional cases cross the scratch tile.
    const int N = probe ? probe->n : 1024;
    const int K = probe ? probe->k : 1024;
    const auto device = DeviceId::cuda(0);
    {
        SCOPED_TRACE(format.label);
        ScopedGPUStream stream_owner(device);
        const auto stream = static_cast<cudaStream_t>(stream_owner.get());
        auto weights = format.create({static_cast<size_t>(N), static_cast<size_t>(K)}, 72041);
        auto prepared = makeGpuPreparedGemm(
            weights.get(), device, std::string("test.device_rows.") + format.label,
            ModelContextId{static_cast<uint64_t>(720410 + format.source_codebook_id)});
        auto *kernel = dynamic_cast<CUDAQuantisedGemmKernel *>(prepared.kernel);
        ASSERT_NE(kernel, nullptr);
        kernel->setGPUStream(stream);
        const auto requirements = kernel->getWorkspaceRequirements(M, N, K);
        auto workspace = std::make_unique<DeviceWorkspaceManager>(
            device, requirements.total_bytes_with_alignment());
        ASSERT_TRUE(workspace->allocate(requirements));
        kernel->bindWorkspace(workspace.get());
        ASSERT_TRUE(kernel->prepareFusedProjectionGraphCapture(1));
        ASSERT_NE(workspace, nullptr);
        DeviceNativeVNNIMatrixDesc matrix;
        ASSERT_TRUE(kernel->exportNativeVNNIMatrixDesc(matrix));

        auto input = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)}, -0.75f, 0.75f, 72042);
        auto output = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});
        auto serial_input = TestTensorFactory::createFP32({1, static_cast<size_t>(K)});
        auto serial_output = TestTensorFactory::createFP32({1, static_cast<size_t>(N)});
        auto count = TestTensorFactory::createINT32({1});
        ASSERT_TRUE(input->ensureOnDevice(device, stream));
        ASSERT_TRUE(output->allocateOnDevice(device, stream));
        ASSERT_TRUE(serial_input->allocateOnDevice(device, stream));
        ASSERT_TRUE(serial_output->allocateOnDevice(device, stream));
        ASSERT_TRUE(count->allocateOnDevice(device, stream));

        std::vector<float> oracle(static_cast<size_t>(M) * N);
        for (int row = 0; row < M; ++row)
        {
            ASSERT_EQ(cudaMemcpyAsync(serial_input->gpu_data_ptr(),
                static_cast<const float *>(input->gpu_data_ptr()) + row * K,
                K * sizeof(float), cudaMemcpyDeviceToDevice, stream), cudaSuccess);
            TransferEngine::publishCurrentDeviceWrite(serial_input.get(), stream);
            if (route == VerifierRowRoute::TensorSwiGLU)
                ASSERT_TRUE(kernel->multiply_tensor_with_fused_swiglu(
                    serial_input.get(), serial_input.get(), serial_output.get(), 1, N, K));
            else
                ASSERT_TRUE(kernel->multiply_tensor(serial_input.get(), serial_output.get(), 1, N, K));
            ASSERT_EQ(cudaMemcpyAsync(oracle.data() + row * N,
                serial_output->gpu_data_ptr(), N * sizeof(float),
                cudaMemcpyDeviceToHost, stream), cudaSuccess);
        }
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        ASSERT_TRUE(std::any_of(oracle.begin(), oracle.end(),
            [](float value) { return std::isfinite(value) && value != 0.0f; }));

        auto *quant = static_cast<int8_t *>(workspace->getBuffer(GemmWorkspaceBuffers::QUANT_A));
        auto *scales = static_cast<float *>(workspace->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE));
        auto *partials = static_cast<float *>(workspace->getBuffer(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS));
        ASSERT_TRUE(quant && scales && partials);
        // This adapter borrows the same declared arena; it owns no GPU bytes.
        std::unique_ptr<CUDAGemvContext, decltype(&cudaGemvContext_destroy)> context(
            cudaGemvContext_create(0), cudaGemvContext_destroy);
        ASSERT_NE(context, nullptr);
        cudaGemvContext_bindWorkspace(context.get(), partials,
            workspace->getBufferSize(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS));
        ASSERT_TRUE(cudaQuantGemm_quantizeActivationsBlockwise(
            static_cast<const float *>(input->gpu_data_ptr()), quant, scales,
            M, K, 0, stream));
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        // Join initial tensor publication before recording, just as production
        // graph preparation does. A completed stream is not a coherence receipt.
        TransferEngine::requireDeviceInput(input.get(), device, stream);
        const auto rows = DeviceRowRange::deviceCounted(
            M, static_cast<const int32_t *>(count->gpu_data_ptr()));
        auto graph = std::make_unique<CUDAGraphCapture>(stream, 0);
        {
            ScopedVerifierRowCandidate candidate(probe ? probe->grouped_rows : 0);
            auto arithmetic = kernel->beginVerifierDecodeEquivalentScope(rows);
            ScopedBackendGraphCapture capture(*graph, "device_counted_verifier_rows");
            ASSERT_TRUE(capture.begin());
            bool launched = false;
            if (route == VerifierRowRoute::TensorProjection)
                launched = kernel->multiply_fused_verifier_rows_decode_equivalent(
                    input.get(), {{kernel, output.get(), N, nullptr, "projection"}}, M, K);
            else if (route == VerifierRowRoute::TensorSwiGLU)
                launched = kernel->multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                    input.get(), input.get(), output.get(), M, N, K);
            else
                launched = cudaNativeVNNIGemvTuned_small_m_fp32_withPolicy(
                quant, matrix.payload, static_cast<const uint16_t *>(matrix.scales),
                static_cast<const uint16_t *>(matrix.mins),
                static_cast<const uint32_t *>(matrix.emins),
                static_cast<float *>(output->gpu_data_ptr()), scales,
                M, N, K, 1.0f, 0.0f, nullptr, nullptr, matrix.codebook_id,
                canonicalDeviceVnniCodebookId(matrix.arithmeticPolicyCodebookId()),
                0, stream, context.get(), nullptr, &rows);
            capture.finish();
            ASSERT_TRUE(launched);
            ASSERT_TRUE(graph->instantiate());
        }

        // One graph crosses shallow/deep/empty states without changing its
        // embedded pointers, partial strides, or allocation generation.
        std::vector<int32_t> publications{3, M, 2, 0, 16, std::min(17, M), 1, M, 3};
        for (int active = 2; active <= 16; ++active)
            publications.push_back(active);
        std::vector<uint32_t> actual(static_cast<size_t>(M) * N);
        for (int32_t active : publications)
        {
            SCOPED_TRACE("active=" + std::to_string(active));
            ASSERT_EQ(cudaMemsetAsync(output->gpu_data_ptr(), 0xa5,
                actual.size() * sizeof(uint32_t), stream), cudaSuccess);
            ASSERT_EQ(cudaMemsetAsync(partials, 0xa5,
                workspace->getBufferSize(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS),
                stream), cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(count->gpu_data_ptr(), &active, sizeof(active),
                cudaMemcpyHostToDevice, stream), cudaSuccess);
            ASSERT_TRUE(graph->launch());
            ASSERT_EQ(cudaMemcpyAsync(actual.data(), output->gpu_data_ptr(),
                actual.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            if (active > 0)
                EXPECT_EQ(std::memcmp(actual.data(), oracle.data(),
                    static_cast<size_t>(active) * N * sizeof(float)), 0);
            for (size_t i = static_cast<size_t>(active) * N; i < actual.size(); ++i)
                ASSERT_EQ(actual[i], 0xa5a5a5a5u) << "inactive index=" << i;
        }
        if (probe)
        {
            // All candidate rows already passed the serial-byte and poisoned
            // suffix oracle above. Timing uses the same retained executable;
            // count publication, warmup and D2H validation are outside events.
            const auto destroy_event = [](cudaEvent_t event) { (void)cudaEventDestroy(event); };
            using Event = std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, decltype(destroy_event)>;
            cudaEvent_t raw_start = nullptr;
            ASSERT_EQ(cudaEventCreate(&raw_start), cudaSuccess);
            Event start(raw_start, destroy_event);
            cudaEvent_t raw_stop = nullptr;
            ASSERT_EQ(cudaEventCreate(&raw_stop), cudaSuccess);
            Event stop(raw_stop, destroy_event);
            constexpr int replays = 128;
            for (int32_t active : {2, 3, 4, 8, 16})
            {
                ASSERT_EQ(cudaMemcpyAsync(count->gpu_data_ptr(), &active, sizeof(active),
                    cudaMemcpyHostToDevice, stream), cudaSuccess);
                for (int warmup = 0; warmup < 8; ++warmup)
                    ASSERT_TRUE(graph->launch());
                std::vector<float> timings;
                for (int sample = 0; sample < 5; ++sample)
                {
                    ASSERT_EQ(cudaEventRecord(start.get(), stream), cudaSuccess);
                    for (int repeat = 0; repeat < replays; ++repeat)
                        ASSERT_TRUE(graph->launch());
                    ASSERT_EQ(cudaEventRecord(stop.get(), stream), cudaSuccess);
                    ASSERT_EQ(cudaEventSynchronize(stop.get()), cudaSuccess);
                    float elapsed_ms = 0.0f;
                    ASSERT_EQ(cudaEventElapsedTime(&elapsed_ms, start.get(), stop.get()), cudaSuccess);
                    timings.push_back(elapsed_ms * 1000.0f / replays);
                }
                std::sort(timings.begin(), timings.end());
                std::cout << "DEVICE_ROW_ECONOMY," << format.label << ',' << M << ','
                          << N << ',' << K << ',' << probe->grouped_rows << ','
                          << active << ',' << timings[timings.size() / 2] << '\n';
            }
        }
        graph.reset(); // Retire captured operands before their workspace owner.
        kernel->unbindWorkspace();
        kernel->clearGPUStreamBinding();
    }
}

/** @brief Compare retained capacity with live extent without changing weight bytes. */
TEST(CUDADeviceCountedVerifierRowsEconomy, Qwen38FFNReuseWidths)
{
    const auto &formats = quantizedMoEVerifierFormats();
    const auto format = std::find_if(formats.begin(), formats.end(),
        [](const auto &value) { return value.tensor_type == TensorType::IQ4_XS; });
    ASSERT_NE(format, formats.end());
    // Interleave candidate widths in a fixed nonmonotonic order, bracketed by
    // installed policy, so clock drift is visible rather than a hidden winner.
    for (const auto shape : {std::pair{17408, 5120}, std::pair{5120, 17408}})
        for (int rows : {0, 4, 16, 2, 8, 0})
            proveDeviceCountedRows(*format, VerifierRowRoute::Raw,
                VerifierRowEconomyProbe{shape.first, shape.second, rows});
}

TEST_P(CUDADeviceCountedVerifierRows, RetainedGraphPreservesPhysicalScratch)
{
    proveDeviceCountedRows(GetParam(), VerifierRowRoute::Raw);
}

TEST_P(CUDADeviceCountedVerifierRows, TensorCoreRetainsInactiveRowsAndPhysicalScratch)
{
    const ScopedVerifierTensorCoreCandidate candidate;
    proveDeviceCountedRows(GetParam(), VerifierRowRoute::Raw);
}

TEST_P(CUDADeviceCountedVerifierRows, TensorProjectionInheritsDeviceRows)
{
    proveDeviceCountedRows(GetParam(), VerifierRowRoute::TensorProjection);
}

TEST_P(CUDADeviceCountedVerifierRows, TensorSwiGLUInheritsDeviceRows)
{
    proveDeviceCountedRows(GetParam(), VerifierRowRoute::TensorSwiGLU);
}

/** @brief Native floating weights share the same retained count contract. */
TEST(CUDADeviceCountedVerifierRows, AllFloatingFormatsThroughTensorAdapters)
{
    for (auto type : {TensorType::FP32, TensorType::FP16, TensorType::BF16})
        for (auto operation : {FloatingVerifierOperation::Projection, FloatingVerifierOperation::SwiGLUDown})
            proveDeviceCountedFloatingRows<CUDAGraphCapture, cudaStream_t>(
                DeviceId::cuda(0), type, operation);
}

INSTANTIATE_TEST_SUITE_P(AllFormats, CUDADeviceCountedVerifierRows,
    // The same complete registry with non-degenerate IQ3_S/IQ4_XS payloads.
    ::testing::ValuesIn(quantizedMoEVerifierFormats()),
    [](const ::testing::TestParamInfo<QuantizedVerifierFormatCase> &info)
    {
        return std::string(info.param.label);
    });
