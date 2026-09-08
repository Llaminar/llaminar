/**
 * @file Test__ROCmDeviceCountedVerifierRows.cpp
 * @brief Captured all-codebook proof of device-owned logical row admission.
 *
 * Production preparation and the public serial M1 operation supply the byte
 * oracle. One retained graph exercises shallow, deep, empty, and cross-scratch-
 * tile publications while poisoned inactive outputs must remain untouched.
 * Keeping this focused fixture separate avoids rebuilding the large historical
 * small-M test translation unit during row-lifecycle work.
 */
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "backends/rocm/HIPGraphCapture.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/rocm/gemm/ROCmGroupedVerifierLaunch.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "tensors/NativeVnniFormatInfo.h"
#include "transfer/TransferEngine.h"
#include "../../../utils/GpuPreparedGemmHarness.h"
#include "../../../utils/QuantizedVerifierFormats.h"
#include "../../../utils/ScopedGPUStream.h"
#include "../../../utils/TestTensorFactory.h"
#include "../../../utils/DeviceCountedFloatingRowsProof.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <iostream>
#include <vector>

extern "C" bool rocmQuantGemm_quantizeActivationsBlockwiseWithSums(
    const float *, int8_t *, float *, int32_t *, int, int, int, void *, int);

using namespace llaminar2;
using namespace llaminar2::rocm;
using namespace llaminar2::test;

/** @brief One source format per case, also isolatable for kernel profiling. */
class ROCmDeviceCountedVerifierRows
    : public ::testing::TestWithParam<QuantizedVerifierFormatCase> {};

/** @brief Native launch families sharing the same device count contract. */
enum class VerifierProjectionBundle { Single, Homogeneous, MixedDecoder, TensorProjection, TensorSwiGLU };

/** @brief Profiler replays never repeat the independently measured byte/timing corpus. */
enum class VerifierProbePurpose { CorrectnessOnly, CorrectnessAndTiming, IsolatedProfile };

/** @brief Learned bias is independent of row admission and must survive replay. */
enum class VerifierBias { None, PerColumn };

/** @brief Perf-only physical geometry; live counts remain device-owned inputs. */
struct VerifierCapacityProbe
{
    int rows;
    int columns;
    int width;
    VerifierProbePurpose purpose = VerifierProbePurpose::CorrectnessAndTiming;
};

/**
 * @brief Stress one physical launch family against an independent serial oracle.
 * @param format Source-format creator and arithmetic identity.
 * @param bundle Native launch family, not an alternate execution policy.
 * @param probe Optional captured timing geometry; absent in the functional gate.
 * @param bias_policy Include a nonzero learned bias in fused serial/graph proofs.
 *
 * Fused cases use three independent outputs and partial slices. MixedDecoder
 * exercises the per-projection decoder dispatcher for every codebook; the
 * established mixed-weight fixtures separately prove heterogeneous bundles.
 */
static void proveDeviceCountedRows(
    const QuantizedVerifierFormatCase &format, VerifierProjectionBundle bundle,
    std::optional<VerifierCapacityProbe> probe = std::nullopt,
    VerifierBias bias_policy = VerifierBias::None)
{
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0)
        GTEST_SKIP() << "No ROCm device available";
    const int M = probe ? probe->rows : 31; // Functional proof crosses the scratch tile.
    const int N = probe ? probe->columns : 1024;
    const int K = probe ? probe->width : 1024;
    const bool profile_only = probe && probe->purpose == VerifierProbePurpose::IsolatedProfile;
    ASSERT_GE(M, 3);
    ASSERT_GT(N, 0);
    ASSERT_GT(K, 0);
    const size_t matrix_rows = static_cast<size_t>(M);
    const size_t matrix_columns = static_cast<size_t>(N);
    const size_t matrix_width = static_cast<size_t>(K);
    const auto device = DeviceId::rocm(0);
    {
        const int projection_count =
            bundle == VerifierProjectionBundle::Homogeneous ||
            bundle == VerifierProjectionBundle::MixedDecoder ? 3 : 1;
        ASSERT_TRUE(bias_policy == VerifierBias::None || projection_count > 1);
        SCOPED_TRACE(format.label);
        ScopedGPUStream stream_owner(device);
        const auto stream = static_cast<hipStream_t>(stream_owner.get());
        auto weights = format.create({matrix_columns, matrix_width}, 72041);
        auto prepared = makeGpuPreparedGemm(
            weights.get(), device, std::string("test.device_rows.") + format.label,
            ModelContextId{static_cast<uint64_t>(720410 + format.source_codebook_id)});
        auto *kernel = dynamic_cast<ROCmQuantisedGemmKernel *>(prepared.kernel);
        ASSERT_NE(kernel, nullptr);
        kernel->setGPUStream(stream);
        auto requirements = kernel->getWorkspaceRequirements(M, N, K);
        if (projection_count > 1)
        {
            const std::array<int, 3> columns{N, N, N};
            kernel->appendFusedProjectionWorkspaceRequirements(requirements, M, columns, K);
        }
        auto workspace = std::make_unique<DeviceWorkspaceManager>(
            device, requirements.total_bytes_with_alignment());
        ASSERT_TRUE(workspace->allocate(requirements));
        kernel->bindWorkspace(workspace.get());
        ASSERT_TRUE(kernel->prepareFusedProjectionGraphCapture(projection_count));
        ASSERT_NE(workspace, nullptr);
        DeviceNativeVNNIMatrixDesc matrix;
        ASSERT_TRUE(kernel->exportNativeVNNIMatrixDesc(matrix));

        auto input = TestTensorFactory::createFP32Random({matrix_rows, matrix_width}, -0.75f, 0.75f, 72042);
        auto output = TestTensorFactory::createFP32({static_cast<size_t>(projection_count) * matrix_rows, matrix_columns});
        auto serial_input = TestTensorFactory::createFP32({1, matrix_width});
        auto serial_output = TestTensorFactory::createFP32({1, matrix_columns});
        auto count = TestTensorFactory::createINT32({1});
        auto bias = TestTensorFactory::createFP32({matrix_columns});
        if (bias_policy == VerifierBias::PerColumn)
        {
            for (int column = 0; column < N; ++column)
                bias->mutable_data()[column] = 0.03125f * static_cast<float>((column % 9) - 4);
            TransferEngine::prepareDeviceInput(bias.get(), device, stream);
        }
        ASSERT_TRUE(input->ensureOnDevice(device, stream));
        ASSERT_TRUE(output->allocateOnDevice(device, stream));
        ASSERT_TRUE(serial_input->allocateOnDevice(device, stream));
        ASSERT_TRUE(serial_output->allocateOnDevice(device, stream));
        ASSERT_TRUE(count->allocateOnDevice(device, stream));

        std::vector<float> oracle(static_cast<size_t>(M) * N);
        for (int row = 0; !profile_only && row < M; ++row)
        {
            ASSERT_EQ(hipMemcpyAsync(serial_input->gpu_data_ptr(),
                static_cast<const float *>(input->gpu_data_ptr()) + row * K,
                K * sizeof(float), hipMemcpyDeviceToDevice, stream), hipSuccess);
            TransferEngine::publishCurrentDeviceWrite(serial_input.get(), stream);
            if (bundle == VerifierProjectionBundle::TensorSwiGLU)
                ASSERT_TRUE(kernel->multiply_tensor_with_fused_swiglu(
                    serial_input.get(), serial_input.get(), serial_output.get(), 1, N, K));
            else
                ASSERT_TRUE(kernel->multiply_tensor(serial_input.get(), serial_output.get(), 1, N, K,
                    true, 1.0f, 0.0f, bias_policy == VerifierBias::PerColumn ? bias.get() : nullptr));
            ASSERT_EQ(hipMemcpyAsync(oracle.data() + row * N,
                serial_output->gpu_data_ptr(), N * sizeof(float),
                hipMemcpyDeviceToHost, stream), hipSuccess);
        }
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        if (!profile_only)
            ASSERT_TRUE(std::any_of(oracle.begin(), oracle.end(),
                [](float value) { return std::isfinite(value) && value != 0.0f; }));

        auto *quant = static_cast<int8_t *>(workspace->getBuffer(GemmWorkspaceBuffers::QUANT_A));
        auto *scales = static_cast<float *>(workspace->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE));
        auto *sums = static_cast<int32_t *>(workspace->getBuffer(GemmWorkspaceBuffers::SUMS_A_BLOCKWISE));
        const auto partial_id = projection_count == 1
            ? GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL
            : GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL_BATCHED;
        auto *partials = static_cast<float *>(workspace->getBuffer(partial_id));
        const size_t partial_slice = workspace->getBufferSize(partial_id) /
            (sizeof(float) * projection_count);
        ASSERT_TRUE(quant && scales && sums && partials);
        ASSERT_TRUE(rocmQuantGemm_quantizeActivationsBlockwiseWithSums(
            static_cast<const float *>(input->gpu_data_ptr()), quant, scales, sums,
            M, K, 0, stream, 32));
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        // Join initial tensor publication before recording, just as production
        // graph preparation does. A completed stream is not a coherence receipt.
        TransferEngine::requireDeviceInput(input.get(), device, stream);
        const auto rows = DeviceRowRange::deviceCounted(
            M, static_cast<const int32_t *>(count->gpu_data_ptr()));
        auto graph = std::make_unique<HIPGraphCapture>(stream, 0);
        {
            auto arithmetic = kernel->beginVerifierDecodeEquivalentScope(rows);
            ScopedBackendGraphCapture capture(*graph, "device_counted_verifier_rows");
            ASSERT_TRUE(capture.begin());
            bool launched = false;
            if (bundle == VerifierProjectionBundle::TensorProjection)
                launched = kernel->multiply_fused_verifier_rows_decode_equivalent(
                    input.get(), {{kernel, output.get(), N, nullptr, "projection"}}, M, K);
            else if (bundle == VerifierProjectionBundle::TensorSwiGLU)
                launched = kernel->multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                    input.get(), input.get(), output.get(), M, N, K);
            else if (bundle == VerifierProjectionBundle::Single)
            {
                launched = rocmGemv_native_vnni_small_m_fp32_with_sums_policy(
                    quant, matrix.payload, matrix.scales, matrix.mins, matrix.emins,
                    static_cast<float *>(output->gpu_data_ptr()), scales, sums, partials,
                    M, N, K, matrix.codebook_id,
                    canonicalDeviceVnniCodebookId(matrix.arithmeticPolicyCodebookId()),
                    0, stream, &rows);
            }
            else
            {
                // Launch descriptors are copied during capture, while every
                // mutable payload and count remains device-owned.
                std::array<const uint8_t *, 3> payloads{};
                std::array<const uint16_t *, 3> weight_scales{}, mins{};
                std::array<const uint32_t *, 3> emins{};
                std::array<float *, 3> outputs{}, scratch{};
                std::array<const float *, 3> biases{};
                std::array<int, 3> columns{N, N, N};
                std::array<uint8_t, 3> codebooks{};
                for (int p = 0; p < projection_count; ++p)
                {
                    payloads[p] = matrix.payload;
                    weight_scales[p] = static_cast<const uint16_t *>(matrix.scales);
                    mins[p] = static_cast<const uint16_t *>(matrix.mins);
                    emins[p] = static_cast<const uint32_t *>(matrix.emins);
                    outputs[p] = static_cast<float *>(output->gpu_data_ptr()) + p * M * N;
                    scratch[p] = partials + p * partial_slice;
                    codebooks[p] = matrix.codebook_id;
                    biases[p] = bias_policy == VerifierBias::PerColumn
                        ? static_cast<const float *>(bias->gpu_data_ptr()) : nullptr;
                }
                const auto launch_fused = [&](const int32_t *activation_sums)
                {
                    if (bundle == VerifierProjectionBundle::Homogeneous)
                        return rocmGemv_native_vnni_small_m_batched_fp32_with_sums_policy(
                            quant, payloads.data(), weight_scales.data(), mins.data(), emins.data(),
                            biases.data(), outputs.data(), scales, activation_sums, scratch.data(), columns.data(),
                            projection_count, M, K, matrix.codebook_id, 0, stream, -1, -1, &rows);
                    return rocmGemv_native_vnni_small_m_batched_mixed_fp32_with_sums_policy(
                            quant, payloads.data(), weight_scales.data(), mins.data(), emins.data(),
                            biases.data(), outputs.data(), scales, activation_sums, scratch.data(), columns.data(),
                            codebooks.data(), projection_count, M, K, 0, stream, -1, -1, &rows);
                };
                // Invalid bindings must reject before adding any graph node;
                // the immediately following valid recording must still work.
                if (!profile_only) EXPECT_FALSE(launch_fused(nullptr));
                launched = launch_fused(sums);
            }
            capture.finish();
            ASSERT_TRUE(launched);
            ASSERT_TRUE(graph->instantiate());
        }

        if (profile_only)
        {
            // One terminal replay contains only this exact candidate's
            // producer/reducer. Setup dispatches have distinct kernel names;
            // no oracle, warmup, or timing replay contaminates this range.
            const int32_t active = 3;
            ASSERT_EQ(hipMemcpyAsync(count->gpu_data_ptr(), &active, sizeof(active),
                hipMemcpyHostToDevice, stream), hipSuccess);
            ASSERT_TRUE(graph->launch());
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            graph.reset();
            kernel->unbindWorkspace();
            kernel->clearGPUStreamBinding();
            return;
        }

        // One graph crosses shallow/deep/empty states without changing its
        // embedded pointers, partial strides, or allocation generation.
        std::vector<int32_t> publications{3, M, 2, 0, 16, 17, 1, M, 3};
        for (int active = 2; active <= 16; ++active)
            publications.push_back(active);
        std::erase_if(publications, [M](int active) { return active > M; });
        std::vector<uint32_t> actual(static_cast<size_t>(projection_count) * M * N);
        for (int32_t active : publications)
        {
            SCOPED_TRACE("active=" + std::to_string(active));
            ASSERT_EQ(hipMemsetAsync(output->gpu_data_ptr(), 0xa5,
                actual.size() * sizeof(uint32_t), stream), hipSuccess);
            ASSERT_EQ(hipMemsetAsync(partials, 0xa5,
                workspace->getBufferSize(partial_id),
                stream), hipSuccess);
            ASSERT_EQ(hipMemcpyAsync(count->gpu_data_ptr(), &active, sizeof(active),
                hipMemcpyHostToDevice, stream), hipSuccess);
            ASSERT_TRUE(graph->launch());
            ASSERT_EQ(hipMemcpyAsync(actual.data(), output->gpu_data_ptr(),
                actual.size() * sizeof(uint32_t), hipMemcpyDeviceToHost, stream), hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            for (int p = 0; p < projection_count; ++p)
            {
                const auto *actual_projection = actual.data() + p * M * N;
                if (active > 0)
                    EXPECT_EQ(std::memcmp(actual_projection, oracle.data(),
                        static_cast<size_t>(active) * N * sizeof(float)), 0) << "projection=" << p;
                for (size_t i = static_cast<size_t>(active) * N; i < M * N; ++i)
                    ASSERT_EQ(actual_projection[i], 0xa5a5a5a5u)
                        << "projection=" << p << " inactive index=" << i;
            }
        }
        if (probe && probe->purpose == VerifierProbePurpose::CorrectnessAndTiming)
        {
            // The same retained executable has already passed every live-row
            // byte check and inactive poison check. Upload and warmup are not
            // timed; repeated launches contain no host data or row decisions.
            const auto destroy_event = [](hipEvent_t event) { (void)hipEventDestroy(event); };
            using Event = std::unique_ptr<std::remove_pointer_t<hipEvent_t>, decltype(destroy_event)>;
            hipEvent_t raw_start = nullptr;
            ASSERT_EQ(hipEventCreate(&raw_start), hipSuccess);
            Event start(raw_start, destroy_event);
            hipEvent_t raw_stop = nullptr;
            ASSERT_EQ(hipEventCreate(&raw_stop), hipSuccess);
            Event stop(raw_stop, destroy_event);
            constexpr int replays = 64;
            std::vector<int32_t> timed_counts{2, 3};
            if (M > 3) timed_counts.push_back(M);
            for (int32_t active : timed_counts)
            {
                ASSERT_LE(active, M);
                ASSERT_EQ(hipMemcpyAsync(count->gpu_data_ptr(), &active, sizeof(active),
                    hipMemcpyHostToDevice, stream), hipSuccess);
                for (int warmup = 0; warmup < 8; ++warmup) ASSERT_TRUE(graph->launch());
                std::vector<float> timings;
                for (int sample = 0; sample < 7; ++sample)
                {
                    ASSERT_EQ(hipEventRecord(start.get(), stream), hipSuccess);
                    for (int repeat = 0; repeat < replays; ++repeat) ASSERT_TRUE(graph->launch());
                    ASSERT_EQ(hipEventRecord(stop.get(), stream), hipSuccess);
                    ASSERT_EQ(hipEventSynchronize(stop.get()), hipSuccess);
                    float elapsed_ms = 0.0f;
                    ASSERT_EQ(hipEventElapsedTime(&elapsed_ms, start.get(), stop.get()), hipSuccess);
                    timings.push_back(elapsed_ms * 1000.0f / replays);
                }
                std::sort(timings.begin(), timings.end());
                std::cout << "ROCM_DEVICE_ROW_ECONOMY," << format.label << ','
                          << static_cast<int>(bundle) << ',' << M << ',' << N << ',' << K
                          << ',' << active << ',' << timings[timings.size() / 2] << '\n';
            }
        }
        graph.reset(); // Retire captured operands before their workspace owner.
        kernel->unbindWorkspace();
        kernel->clearGPUStreamBinding();
    }
}

/** @brief Attribute capacity overhead in production tensor GEMM and fused down paths. */
TEST(ROCmDeviceCountedVerifierRowsEconomy, Qwen38ProjectionCapacity)
{
    const auto &formats = quantizedMoEVerifierFormats();
    const auto format = std::find_if(formats.begin(), formats.end(),
        [](const auto &value) { return value.tensor_type == TensorType::IQ4_XS; });
    ASSERT_NE(format, formats.end());
    for (auto shape : {std::pair{17408, 5120}, std::pair{5120, 17408}, std::pair{10240, 5120}})
        for (auto bundle : {VerifierProjectionBundle::TensorProjection, VerifierProjectionBundle::TensorSwiGLU})
            for (int capacity : {3, 16, 3})
                proveDeviceCountedRows(*format, bundle,
                    VerifierCapacityProbe{capacity, shape.first, shape.second});
}

/** @brief Compare both fused decoder launch families at the retained row ceiling. */
TEST(ROCmDeviceCountedVerifierRowsEconomy, Qwen38FusedCapacity)
{
    const auto &formats = quantizedMoEVerifierFormats();
    const auto format = std::find_if(formats.begin(), formats.end(),
        [](const auto &value) { return value.tensor_type == TensorType::IQ4_XS; });
    ASSERT_NE(format, formats.end());
    for (auto shape : {std::pair{17408, 5120}, std::pair{10240, 5120}})
        for (auto bundle : {VerifierProjectionBundle::Homogeneous, VerifierProjectionBundle::MixedDecoder})
            for (int capacity : {3, 16, 3})
                proveDeviceCountedRows(*format, bundle,
                    VerifierCapacityProbe{capacity, shape.first, shape.second});
}

/** @brief Isolate one capacity/decoder pair; canonical timing is a separate invocation. */
TEST(ROCmDeviceCountedVerifierRowsEconomy, Qwen38FusedProfile)
{
    const auto &formats = quantizedMoEVerifierFormats();
    const auto format = std::find_if(formats.begin(), formats.end(),
        [](const auto &value) { return value.tensor_type == TensorType::IQ4_XS; });
    ASSERT_NE(format, formats.end());
    const auto *capacity_text = std::getenv("LLAMINAR_ROCM_VERIFIER_PROFILE_CAPACITY");
    const int capacity = capacity_text ? std::stoi(capacity_text) : 16;
    ASSERT_TRUE(capacity == 3 || capacity == 16);
    const auto *bundle_text = std::getenv("LLAMINAR_ROCM_VERIFIER_PROFILE_BUNDLE");
    const std::string bundle_name = bundle_text ? bundle_text : "mixed";
    ASSERT_TRUE(bundle_name == "mixed" || bundle_name == "homogeneous");
    proveDeviceCountedRows(*format,
        bundle_name == "mixed" ? VerifierProjectionBundle::MixedDecoder : VerifierProjectionBundle::Homogeneous,
        VerifierCapacityProbe{capacity, 17408, 5120, VerifierProbePurpose::IsolatedProfile});
}

TEST_P(ROCmDeviceCountedVerifierRows, RetainedGraphPreservesPhysicalScratch)
{
    proveDeviceCountedRows(GetParam(), VerifierProjectionBundle::Single);
}

TEST_P(ROCmDeviceCountedVerifierRows, FusedBundlePreservesIndependentScratch)
{
    proveDeviceCountedRows(GetParam(), VerifierProjectionBundle::Homogeneous);
}

TEST_P(ROCmDeviceCountedVerifierRows, MixedDecoderBundlePreservesIndependentScratch)
{
    proveDeviceCountedRows(GetParam(), VerifierProjectionBundle::MixedDecoder);
}

/** @brief Tail lanes must not publish and learned bias must be added exactly once. */
TEST_P(ROCmDeviceCountedVerifierRows, FusedColumnTailAndBiasAcrossLiveRows)
{
    for (auto bundle : {VerifierProjectionBundle::Homogeneous, VerifierProjectionBundle::MixedDecoder})
        proveDeviceCountedRows(GetParam(), bundle,
            VerifierCapacityProbe{31, 1009, 1024, VerifierProbePurpose::CorrectnessOnly},
            VerifierBias::PerColumn);
}

/** @brief One-block formats exercise direct publication rather than split-K reduction. */
TEST(ROCmDeviceCountedVerifierRows, DirectPublicationColumnTailAndBias)
{
    for (const auto &format : quantizedMoEVerifierFormats())
    {
        // A 32-value reduction fits one native block. Superblock formats need
        // wider source geometry and are covered by the ordered-partial sweep.
        if (format.tensor_type != TensorType::Q4_0 && format.tensor_type != TensorType::Q4_1 &&
            format.tensor_type != TensorType::Q5_0 && format.tensor_type != TensorType::Q5_1 &&
            format.tensor_type != TensorType::Q8_0 && format.tensor_type != TensorType::Q8_1 &&
            format.tensor_type != TensorType::IQ4_NL)
            continue;
        for (auto bundle : {VerifierProjectionBundle::Homogeneous, VerifierProjectionBundle::MixedDecoder})
            proveDeviceCountedRows(format, bundle,
                VerifierCapacityProbe{31, 1009, 32, VerifierProbePurpose::CorrectnessOnly},
                VerifierBias::PerColumn);
    }
}

TEST_P(ROCmDeviceCountedVerifierRows, TensorProjectionInheritsDeviceRows)
{
    proveDeviceCountedRows(GetParam(), VerifierProjectionBundle::TensorProjection);
}

TEST_P(ROCmDeviceCountedVerifierRows, TensorSwiGLUInheritsDeviceRows)
{
    proveDeviceCountedRows(GetParam(), VerifierProjectionBundle::TensorSwiGLU);
}

/** @brief Native floating weights share the same retained count contract. */
TEST(ROCmDeviceCountedVerifierRows, AllFloatingFormatsThroughTensorAdapters)
{
    for (auto type : {TensorType::FP32, TensorType::FP16, TensorType::BF16})
        for (auto operation : {FloatingVerifierOperation::Projection, FloatingVerifierOperation::SwiGLUDown})
            proveDeviceCountedFloatingRows<HIPGraphCapture, hipStream_t>(
                DeviceId::rocm(0), type, operation);
}

INSTANTIATE_TEST_SUITE_P(AllFormats, ROCmDeviceCountedVerifierRows,
    // The same complete registry with non-degenerate IQ3_S/IQ4_XS payloads.
    ::testing::ValuesIn(quantizedMoEVerifierFormats()),
    [](const ::testing::TestParamInfo<QuantizedVerifierFormatCase> &info)
    {
        return std::string(info.param.label);
    });
