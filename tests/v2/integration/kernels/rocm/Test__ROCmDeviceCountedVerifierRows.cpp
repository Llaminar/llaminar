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
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "kernels/rocm/gemm/ROCmGroupedVerifierLaunch.h"
#include "kernels/rocm/gemm/ROCmNativeQuantisedVerifierLaunch.h"
#include "kernels/rocm/gemm/ROCmNativeQuantisedArchitecture.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "tensors/NativeQuantisedFormatInfo.h"
#include "utils/PrefillGraphBucketDefaults.h"
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
#include <map>
#include <tuple>
#include <iomanip>
#include <optional>
#include <span>
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

/** @brief Drain the exact stream before borrowed fixture storage can retire on failure. */
struct VerifierFixtureStreamDrain {
    hipStream_t stream; ///< Fixture-owned nondefault stream.
    /** @brief Preserve asynchronous owners; inability to retire is fatal. */
    ~VerifierFixtureStreamDrain() { if(hipStreamSynchronize(stream)!=hipSuccess)std::terminate(); }
};

/**
 * @brief Retain one immutable production-prepared source across sequential widths.
 *
 * The source tensor outlives the prepared handle, which borrows its metadata.
 * No workspace, captured executable, activation or mutable row count is shared.
 */
class VerifierPreparedSource final
{
public:
    /** @brief Prepare the original seeded source once for one exact matrix.
     * @param format Complete source identity and creator. @param device Original owner.
     * @param columns Full output width. @param width Full reduction width. */
    VerifierPreparedSource(const QuantizedVerifierFormatCase &format,
        DeviceId device, int columns, int width)
        : source_(format.create({static_cast<size_t>(columns),static_cast<size_t>(width)},72041)),
          prepared_(makeGpuPreparedGemm(source_.get(),device,
              std::string("test.device_rows.")+format.label,
              ModelContextId{static_cast<uint64_t>(720410+format.source_codebook_id)})),
          device_(device),columns_(columns),width_(width),type_(format.tensor_type),
          source_id_(format.source_codebook_id),superblock_(format.source_is_superblock) {}

    /** @brief Authenticate an immutable matrix before lending its kernel to a case.
     * @param format Requested complete source identity. @param device Original owner.
     * @param columns Full output width. @param width Full reduction width.
     * @return The retained ROCm kernel, with no case workspace or stream binding.
     * @throws std::invalid_argument For a different source, owner or geometry.
     * @throws std::runtime_error For an unexpected prepared kernel type. */
    ROCmQuantisedGemmKernel &kernel(const QuantizedVerifierFormatCase &format,
        DeviceId device, int columns, int width) const
    {
        if(device!=device_ || columns!=columns_ || width!=width_ ||
            format.tensor_type!=type_ || format.source_codebook_id!=source_id_ ||
            format.source_is_superblock!=superblock_)
            throw std::invalid_argument("Verifier prepared source cannot cross source/device/geometry");
        auto *result=dynamic_cast<ROCmQuantisedGemmKernel *>(prepared_.kernel);
        if(!result)throw std::runtime_error("Verifier source requires a prepared ROCm quantized kernel");
        return *result;
    }
private:
    std::unique_ptr<TensorBase> source_;
    GpuPreparedGemm prepared_;
    DeviceId device_;
    int columns_,width_;
    TensorType type_;
    uint8_t source_id_;
    bool superblock_;
};

/** @brief Retire a case binding before its workspace, including fatal test exits. */
class VerifierFixtureKernelBinding final
{
public:
    /** @brief Bind fresh case storage on its exact stream.
     * @param kernel Retained immutable source kernel. @param workspace Case-owned storage.
     * @param stream Nondefault case stream, retained until this scope retires. */
    VerifierFixtureKernelBinding(ROCmQuantisedGemmKernel &kernel,
        DeviceWorkspaceManager &workspace,hipStream_t stream):kernel_(kernel),stream_(stream)
    {
        try {kernel_.setGPUStream(stream_);kernel_.bindWorkspace(&workspace);}
        catch(...) {kernel_.unbindWorkspace();kernel_.clearGPUStreamBinding();throw;}
    }
    VerifierFixtureKernelBinding(const VerifierFixtureKernelBinding &)=delete;
    VerifierFixtureKernelBinding &operator=(const VerifierFixtureKernelBinding &)=delete;
    /** @brief Graphs and observations retire first; clear the borrowed case binding next. */
    ~VerifierFixtureKernelBinding()
    {
        if(hipStreamSynchronize(stream_)!=hipSuccess)std::terminate();
        kernel_.unbindWorkspace();kernel_.clearGPUStreamBinding();
    }
private:
    ROCmQuantisedGemmKernel &kernel_;
    hipStream_t stream_;
};

/** @brief Native launch families sharing the same device count contract. */
enum class VerifierProjectionBundle { Single, Homogeneous, MixedDecoder, TensorProjection, TensorSwiGLU };

/** @brief Profiler replays never repeat the independently measured byte/timing corpus. */
enum class VerifierProbePurpose { CorrectnessOnly, CorrectnessAndTiming, IsolatedProfile, PairedCandidates, PairedAllCounts, PairedExactWidth, PairedAutomaticExactWidth, IsolatedAutomaticProfile };

/** @brief Learned bias is independent of row admission and must survive replay. */
enum class VerifierBias { None, PerColumn };

/** @brief Optional native-grid proof, independent of correctness and timing. */
enum class VerifierGridCertificate { NotRequested, BoundedSingleProjection, EmptyKPartitions };

/** @brief Captured physical geometry; live counts remain device-owned inputs. */
struct VerifierCapacityProbe
{
    int rows;
    int columns;
    int width;
    VerifierProbePurpose purpose = VerifierProbePurpose::CorrectnessAndTiming;
    VerifierGridCertificate grid_certificate = VerifierGridCertificate::NotRequested;
    int profile_live = 3; ///< Explicit original live count for one isolated profiler replay.
};

/**
 * @brief Stress one physical launch family against an independent serial oracle.
 * @param format Source-format creator and arithmetic identity.
 * @param bundle Native launch family, not an alternate execution policy.
 * @param probe Optional captured geometry and native scheduling certificate.
 * @param bias_policy Include a nonzero learned bias in fused serial/graph proofs.
 * @param candidate Exact construction-time producer; the serial oracle stays independent.
 * @param reused_source Optional immutable same-format matrix; every mutable case owner stays fresh.
 *
 * Fused cases use three independent outputs and partial slices. MixedDecoder
 * exercises the per-projection decoder dispatcher for every codebook; the
 * established mixed-weight fixtures separately prove heterogeneous bundles.
 */
static void proveDeviceCountedRows(
    const QuantizedVerifierFormatCase &format, VerifierProjectionBundle bundle,
    std::optional<VerifierCapacityProbe> probe = std::nullopt,
    VerifierBias bias_policy = VerifierBias::None,
    NativeQuantisedVerifierCandidate candidate = NativeQuantisedVerifierCandidate::Automatic,
    VerifierPreparedSource *reused_source = nullptr)
{
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0)
        GTEST_SKIP() << "No ROCm device available";
    const int M = probe ? probe->rows : 31; // Functional proof crosses the scratch tile.
    const int N = probe ? probe->columns : 1024;
    const int K = probe ? probe->width : 1024;
    const bool profile_only = probe && (probe->purpose == VerifierProbePurpose::IsolatedProfile ||
        probe->purpose == VerifierProbePurpose::IsolatedAutomaticProfile);
    const bool paired = probe && (probe->purpose == VerifierProbePurpose::PairedCandidates ||
        probe->purpose == VerifierProbePurpose::PairedAllCounts ||
        probe->purpose == VerifierProbePurpose::PairedExactWidth ||
        probe->purpose == VerifierProbePurpose::PairedAutomaticExactWidth);
    const bool exact_auto = probe && (probe->purpose == VerifierProbePurpose::PairedAutomaticExactWidth ||
        probe->purpose == VerifierProbePurpose::IsolatedAutomaticProfile);
    ASSERT_TRUE(!paired || ((candidate==NativeQuantisedVerifierCandidate::WmmaM16N64 || exact_auto) &&
        (bundle==VerifierProjectionBundle::TensorProjection || bundle==VerifierProjectionBundle::TensorSwiGLU)));

    ASSERT_GE(M, 1);
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
        if((candidate==NativeQuantisedVerifierCandidate::WmmaM16N64 || exact_auto) &&
            nativeQuantisedHardwareForDevice(0).architecture!=native_quantised::NativeQuantisedArchitecture::rocm(11,0,0))
            GTEST_SKIP() << "Native verifier WMMA requires actual gfx1100";

        std::unique_ptr<VerifierPreparedSource> local_source;
        if(!reused_source)
            local_source=std::make_unique<VerifierPreparedSource>(format,device,N,K);
        auto *kernel=&(reused_source?*reused_source:*local_source).kernel(format,device,N,K);
        auto requirements = kernel->getWorkspaceRequirements(M, N, K);
        if (projection_count > 1)
        {
            const std::array<int, 3> columns{N, N, N};
            kernel->appendFusedProjectionWorkspaceRequirements(requirements, M, columns, K);
        }
        auto workspace = std::make_unique<DeviceWorkspaceManager>(
            device, requirements.total_bytes_with_alignment());
        ASSERT_TRUE(workspace->allocate(requirements));
        const VerifierFixtureKernelBinding binding(*kernel,*workspace,stream);
        ASSERT_TRUE(kernel->prepareFusedProjectionGraphCapture(projection_count));
        ASSERT_NE(workspace, nullptr);
        DeviceNativeQuantisedMatrixDesc matrix;
        ASSERT_TRUE(kernel->exportNativeQuantisedMatrixDesc(matrix));

        auto input = TestTensorFactory::createFP32Random({matrix_rows, matrix_width}, -0.75f, 0.75f, 72042);
        auto output = TestTensorFactory::createFP32({static_cast<size_t>(projection_count) * matrix_rows, matrix_columns});
        auto serial_input = TestTensorFactory::createFP32({1, matrix_width});
        auto serial_output = TestTensorFactory::createFP32({1, matrix_columns});
        auto count = TestTensorFactory::createINT32({1});
        auto bias = TestTensorFactory::createFP32({matrix_columns});
        const auto partial_id = projection_count == 1
            ? GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL
            : GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL_BATCHED;
        std::vector<float> oracle(static_cast<size_t>(M)*N);
        std::vector<uint32_t> actual(static_cast<size_t>(projection_count)*M*N);
        const bool prove_empty_partitions=probe && probe->grid_certificate==VerifierGridCertificate::EmptyKPartitions;
        std::vector<uint32_t> partial_words(prove_empty_partitions
            ? workspace->getBufferSize(partial_id)/sizeof(float) : 0);
        int captured_partitions=0;
        std::int32_t published_live=M;
        std::unique_ptr<IGPUGraphCapture> graph, dot_control;
        std::array<std::unique_ptr<IGPUGraphCapture>,2> timing_graphs;
        const VerifierFixtureStreamDrain drain{stream};
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
        graph = GPUDeviceContextPool::instance().retainAMDContext(0)->createGraphCapture(stream);
        {
            auto arithmetic = kernel->beginVerifierDecodeEquivalentScope(rows);
            ScopedCapturedVerifierExtent extent_scope(exact_auto
                ? CapturedVerifierExtent::exactHostedRequest(device, rows) : CapturedVerifierExtent{});
            ScopedNativeQuantisedVerifierCandidate candidate_scope(candidate,0);
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
                launched = rocmGemv_native_quantised_small_m_fp32_with_sums_policy(
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
                        return rocmGemv_native_quantised_small_m_batched_fp32_with_sums_policy(
                            quant, payloads.data(), weight_scales.data(), mins.data(), emins.data(),
                            biases.data(), outputs.data(), scales, activation_sums, scratch.data(), columns.data(),
                            projection_count, M, K, matrix.codebook_id, 0, stream, -1, -1, &rows);
                    return rocmGemv_native_quantised_small_m_batched_mixed_fp32_with_sums_policy(
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
            if(candidate==NativeQuantisedVerifierCandidate::WmmaM16N64 || (exact_auto && M>=5)) {
                std::vector<GPUGraphKernelNodeInfo> nodes; std::string error;
                ASSERT_TRUE(graph->inspectKernelNodes(nodes,&error)) << error;
                int producers=0;
                for(const auto &node:nodes) {
                    if(node.name.find("native_quantised_verifier_wmma_m16n64")!=std::string::npos) {
                        ++producers; ASSERT_TRUE(node.valid()); EXPECT_EQ(node.block_x,128u);
                        EXPECT_EQ(node.block_y,1u); EXPECT_EQ(node.block_z,1u);
                        EXPECT_EQ(node.grid_x,static_cast<uint32_t>(projection_count*((N+63)/64)));
                        EXPECT_GE(node.grid_y,1u); EXPECT_EQ(node.grid_z,1u);
                        if(captured_partitions)EXPECT_EQ(node.grid_y,static_cast<unsigned>(captured_partitions));
                        captured_partitions=node.grid_y;
                    }
                    EXPECT_EQ(node.name.find("native_quantised_verifier_ordered"),std::string::npos);
                    EXPECT_EQ(node.name.find("gemv_native_quantised_kernel_t"),std::string::npos);
                    EXPECT_EQ(node.name.find("gemv_native_quantised_small_m_kernel_t"),std::string::npos);
                    EXPECT_EQ(node.name.find("gemv_native_quantised_small_m_batched_shared"),std::string::npos);
                }
                // One fused packet per existing persistent-scratch row tile;
                // M31 intentionally crosses the default capacity16 boundary.
                ASSERT_EQ(producers,1+(M-1)/kDefaultNativeQuantisedVerifierRowCapacity);
                if(prove_empty_partitions) {
                    ASSERT_LE(M,kDefaultNativeQuantisedVerifierRowCapacity);
                    const int span=(K/32+captured_partitions-1)/captured_partitions;
                    ASSERT_LT((K/32+span-1)/span,captured_partitions)
                        << "The actual recorded serial policy must contain empty tail planes";
                }
            }
            if(exact_auto && M<=4) {
                std::vector<GPUGraphKernelNodeInfo> nodes;std::string error;
                ASSERT_TRUE(graph->inspectKernelNodes(nodes,&error))<<error;
                int grouped=0;
                for(const auto &node:nodes) {
                    EXPECT_EQ(node.name.find("native_quantised_verifier_wmma"),std::string::npos);
                    EXPECT_EQ(node.name.find("native_quantised_verifier_ordered"),std::string::npos);
                    if(node.name.find("gemv_native_quantised_small_m_")!=std::string::npos)++grouped;
                }
                ASSERT_GT(grouped,0)<<"Shallow Automatic must record the original grouped Dot4 producer";
            }
            if (probe && probe->grid_certificate == VerifierGridCertificate::BoundedSingleProjection)
            {
                ASSERT_EQ(bundle, VerifierProjectionBundle::Single);
                // Byte equivalence alone cannot detect a return to rectangular
                // capacity scheduling. Inspect the real captured launch, not
                // a launch-policy mock or a PerfStats-derived authority.
                std::vector<GPUGraphKernelNodeInfo> nodes;
                std::string inspection_error;
                ASSERT_TRUE(graph->inspectKernelNodes(nodes, &inspection_error))
                    << inspection_error;
                const auto producer = std::find_if(nodes.begin(), nodes.end(), [](const auto &node)
                {
                    return node.name.find("gemv_native_quantised_small_m_kernel_t") != std::string::npos;
                });
                ASSERT_NE(producer, nodes.end());
                ASSERT_TRUE(producer->valid());
                // The producer retains fixed four-row arithmetic tiles. This
                // wide shape already has ample column/K parallelism; admitted
                // depth-15 capacity must not schedule every physical row tile.
                EXPECT_LT(producer->grid_z, static_cast<uint32_t>((M + 3) / 4));
                for (const auto &node : nodes)
                    if (node.name.find("gemv_native_quantised_reduce_small_m_kernel_t") != std::string::npos)
                    {
                        ASSERT_TRUE(node.valid());
                        EXPECT_LT(node.grid_y, static_cast<uint32_t>(M));
                    }
            }
            ASSERT_TRUE(graph->instantiate());
        }

        if (profile_only)
        {
            // One terminal replay contains only this exact candidate's
            // producer/reducer. Setup dispatches have distinct kernel names;
            // no oracle, warmup, or timing replay contaminates this range.
            const int32_t active = probe->profile_live;
            ASSERT_GE(active,1); ASSERT_LE(active,M);
            published_live=active;
            ASSERT_EQ(hipMemcpyAsync(count->gpu_data_ptr(), &published_live, sizeof(published_live),
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
        std::vector<int32_t> publications{3, M, 2, 0, 16, 17, 1, M, 3, 4, 5, 4};
        for (int active = 2; active <= 16; ++active)
            publications.push_back(active);
        std::erase_if(publications, [M](int active) { return active > M; });
        for (int32_t active : publications)
        {
            SCOPED_TRACE("active=" + std::to_string(active));
            ASSERT_EQ(hipMemsetAsync(output->gpu_data_ptr(), 0xa5,
                actual.size() * sizeof(uint32_t), stream), hipSuccess);
            ASSERT_EQ(hipMemsetAsync(partials, 0xa5,
                workspace->getBufferSize(partial_id),
                stream), hipSuccess);
            published_live=active;
            ASSERT_EQ(hipMemcpyAsync(count->gpu_data_ptr(), &published_live, sizeof(published_live),
                hipMemcpyHostToDevice, stream), hipSuccess);
            ASSERT_TRUE(graph->launch());
            ASSERT_EQ(hipMemcpyAsync(actual.data(), output->gpu_data_ptr(),
                actual.size() * sizeof(uint32_t), hipMemcpyDeviceToHost, stream), hipSuccess);
            if(prove_empty_partitions && active==M)
                ASSERT_EQ(hipMemcpyAsync(partial_words.data(),partials,partial_words.size()*sizeof(uint32_t),
                    hipMemcpyDeviceToHost,stream),hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            if(prove_empty_partitions && active==M) {
                const int span=(K/32+captured_partitions-1)/captured_partitions;
                const int nonempty=(K/32+span-1)/span;
                for(int p=0;p<projection_count;++p)
                    for(int partition=nonempty;partition<captured_partitions;++partition)
                        for(int i=0;i<M*N;++i)
                            ASSERT_EQ(partial_words[p*partial_slice+static_cast<size_t>(partition)*M*N+i],0u)
                                << "Empty plane must replace poisoned scratch with positive zero";
            }
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
                published_live=active;
            ASSERT_EQ(hipMemcpyAsync(count->gpu_data_ptr(), &published_live, sizeof(published_live),
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
        if(paired) {
            constexpr int repetitions=8, samples=31;
            // Both candidates consume the same prepared operands, persistent
            // workspace, row publication and stream. Every timed repetition
            // records the complete public operation, including quantization.
            const auto enqueue_complete=[&] {
                if(bundle==VerifierProjectionBundle::TensorProjection)
                    return kernel->multiply_fused_verifier_rows_decode_equivalent(
                        input.get(),{{kernel,output.get(),N,nullptr,"projection"}},M,K);
                return kernel->multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                    input.get(),input.get(),output.get(),M,N,K);
            };
            const auto record=[&](IGPUGraphCapture &recording,NativeQuantisedVerifierCandidate path,int repeat) {
                auto arithmetic=kernel->beginVerifierDecodeEquivalentScope(rows);
                ScopedCapturedVerifierExtent extent_scope(exact_auto
                    ? CapturedVerifierExtent::exactHostedRequest(device, rows) : CapturedVerifierExtent{});
                ScopedNativeQuantisedVerifierCandidate selected(path,0);
                ScopedBackendGraphCapture capture(recording,"paired_complete_verifier_operation");
                ASSERT_TRUE(capture.begin());
                for(int i=0;i<repeat;++i)ASSERT_TRUE(enqueue_complete());
                capture.finish();ASSERT_TRUE(recording.instantiate());
            };
            const auto validate=[&](IGPUGraphCapture &executable) {
                for(int active:publications) {
                    published_live=active;
                    ASSERT_EQ(hipMemsetAsync(output->gpu_data_ptr(),0xa5,actual.size()*sizeof(uint32_t),stream),hipSuccess);
                    ASSERT_EQ(hipMemsetAsync(partials,0xa5,workspace->getBufferSize(partial_id),stream),hipSuccess);
                    ASSERT_EQ(hipMemcpyAsync(count->gpu_data_ptr(),&published_live,sizeof(published_live),hipMemcpyHostToDevice,stream),hipSuccess);
                    ASSERT_TRUE(executable.launch());
                    ASSERT_EQ(hipMemcpyAsync(actual.data(),output->gpu_data_ptr(),actual.size()*sizeof(uint32_t),hipMemcpyDeviceToHost,stream),hipSuccess);
                    ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);
                    ASSERT_EQ(std::memcmp(actual.data(),oracle.data(),static_cast<size_t>(active)*N*sizeof(float)),0);
                    for(size_t i=static_cast<size_t>(active)*N;i<actual.size();++i)ASSERT_EQ(actual[i],0xa5a5a5a5u);
                }
            };
            dot_control=GPUDeviceContextPool::instance().retainAMDContext(0)->createGraphCapture(stream);
            ASSERT_NO_FATAL_FAILURE(record(*dot_control,NativeQuantisedVerifierCandidate::Dot4,1));
            ASSERT_NO_FATAL_FAILURE(validate(*dot_control));
            using NodeKey=std::tuple<uintptr_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t>;
            const auto histogram=[](IGPUGraphCapture &capture,std::map<NodeKey,size_t> &result) {
                std::vector<GPUGraphKernelNodeInfo> nodes;std::string error;
                ASSERT_TRUE(capture.inspectKernelNodes(nodes,&error))<<error;
                for(const auto &node:nodes) {
                    ASSERT_TRUE(node.valid());
                    ++result[{node.function_identity,node.grid_x,node.grid_y,node.grid_z,
                        node.block_x,node.block_y,node.block_z}];
                }
                ASSERT_FALSE(result.empty());
            };
            for(int path=0;path<2;++path) {
                timing_graphs[path]=GPUDeviceContextPool::instance().retainAMDContext(0)->createGraphCapture(stream);
                ASSERT_NO_FATAL_FAILURE(record(*timing_graphs[path],path==0?NativeQuantisedVerifierCandidate::Dot4:
                    candidate,repetitions));
                std::map<NodeKey,size_t> single,batch;
                ASSERT_NO_FATAL_FAILURE(histogram(path==0?*dot_control:*graph,single));
                ASSERT_NO_FATAL_FAILURE(histogram(*timing_graphs[path],batch));
                for(auto &[key,number]:single)number*=repetitions;
                ASSERT_EQ(batch,single)<<"Timing must retain eight complete authenticated operations";
                ASSERT_NO_FATAL_FAILURE(validate(*timing_graphs[path]));
            }
            const auto destroy_event=[](hipEvent_t event){(void)hipEventDestroy(event);};
            using Event=std::unique_ptr<std::remove_pointer_t<hipEvent_t>,decltype(destroy_event)>;
            hipEvent_t raw_start=nullptr,raw_stop=nullptr;
            ASSERT_EQ(hipEventCreate(&raw_start),hipSuccess);Event start(raw_start,destroy_event);
            ASSERT_EQ(hipEventCreate(&raw_stop),hipSuccess);Event stop(raw_stop,destroy_event);
            const VerifierFixtureStreamDrain timing_drain{stream};
            std::vector<int32_t> timed_counts;
            if(probe->purpose==VerifierProbePurpose::PairedAllCounts)
                for(int active=1;active<=M;++active)timed_counts.push_back(active);
            else if(probe->purpose==VerifierProbePurpose::PairedExactWidth || exact_auto) timed_counts={M};
            else {timed_counts={2,3};if(M>3)timed_counts.push_back(M);}
            for(int active:timed_counts) {
                published_live=active;
                ASSERT_EQ(hipMemcpyAsync(count->gpu_data_ptr(),&published_live,sizeof(published_live),hipMemcpyHostToDevice,stream),hipSuccess);
                for(int warm=0;warm<8;++warm)for(auto &executable:timing_graphs)ASSERT_TRUE(executable->launch());
                for(int sample=0;sample<samples;++sample)
                    for(int order=0;order<2;++order) {
                        // For two candidates this is AB/BA. A simultaneous
                        // rotation and reversal would cancel and always run AB.
                        const int path=(sample+order)%2;
                        ASSERT_EQ(hipEventRecord(start.get(),stream),hipSuccess);
                        ASSERT_TRUE(timing_graphs[path]->launch());
                        ASSERT_EQ(hipEventRecord(stop.get(),stream),hipSuccess);
                        ASSERT_EQ(hipEventSynchronize(stop.get()),hipSuccess);
                        float elapsed=0;ASSERT_EQ(hipEventElapsedTime(&elapsed,start.get(),stop.get()),hipSuccess);
                        ASSERT_TRUE(std::isfinite(elapsed));ASSERT_GT(elapsed,0.0f);
                        std::cout<<std::setprecision(10)<<"ROCM_MTP_WMMA_PAIRED,"<<format.label<<','
                            <<static_cast<int>(bundle)<<','<<M<<','<<N<<','<<K<<','<<active<<','
                            <<sample<<','<<order<<','<<(path==0?"dot4":exact_auto?"automatic":"wmma_m16n64")<<','
                            <<repetitions<<','<<elapsed*1000.0/repetitions<<'\n';
                    }
            }
            // Each candidate is independently checked after timing; another
            // candidate never repairs its output before the byte observation.
            for(auto &executable:timing_graphs)ASSERT_NO_FATAL_FAILURE(validate(*executable));
        }
        // Reverse declaration order drains the exact stream, retires graphs and
        // buffers, then clears the kernel binding before the workspace retires.
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

/**
 * @brief Wide single projections must iterate live rows with bounded workers.
 *
 * This geometry supplies enough independent columns before the depth-15 row
 * envelope is considered. Replay sweeps every live width, including zero and
 * large-to-small transitions, against serial bytes and poisoned inactive rows.
 * Every source codebook uses the same public production launch. Timing is
 * deliberately absent from this preflight regression.
 */
TEST_P(ROCmDeviceCountedVerifierRows, WideSingleProjectionUsesLiveRowWorkers)
{
    proveDeviceCountedRows(GetParam(), VerifierProjectionBundle::Single,
        VerifierCapacityProbe{16, 16384, 256, VerifierProbePurpose::CorrectnessOnly,
            VerifierGridCertificate::BoundedSingleProjection});
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

/** @brief Actual native candidate replay, separate from the established Automatic gate. */
class ROCmDeviceCountedVerifierWmma : public ::testing::TestWithParam<QuantizedVerifierFormatCase> {};
TEST_P(ROCmDeviceCountedVerifierWmma, SerialBytesStandalone)
{
    proveDeviceCountedRows(GetParam(),VerifierProjectionBundle::Single,std::nullopt,
        VerifierBias::None,NativeQuantisedVerifierCandidate::WmmaM16N64);
}
TEST_P(ROCmDeviceCountedVerifierWmma, SerialBytesHomogeneous)
{
    proveDeviceCountedRows(GetParam(),VerifierProjectionBundle::Homogeneous,std::nullopt,
        VerifierBias::PerColumn,NativeQuantisedVerifierCandidate::WmmaM16N64);
}
TEST_P(ROCmDeviceCountedVerifierWmma, SerialBytesMixedDispatcher)
{
    proveDeviceCountedRows(GetParam(),VerifierProjectionBundle::MixedDecoder,std::nullopt,
        VerifierBias::PerColumn,NativeQuantisedVerifierCandidate::WmmaM16N64);
}
/** @brief Both complete public operations preserve serial bytes for every source format. */
TEST_P(ROCmDeviceCountedVerifierWmma, SerialBytesPublicOperations)
{
    for(auto operation:{VerifierProjectionBundle::TensorProjection,VerifierProjectionBundle::TensorSwiGLU})
        ASSERT_NO_FATAL_FAILURE(proveDeviceCountedRows(GetParam(),operation,std::nullopt,
            VerifierBias::None,NativeQuantisedVerifierCandidate::WmmaM16N64));
}

/** @brief Sweep shallow physical buckets and tails. @param format Source identity. @param candidate Native implementation. */
static void proveWmmaPhysicalBuckets(const QuantizedVerifierFormatCase &format,NativeQuantisedVerifierCandidate candidate)
{
    for(int rows=1;rows<=16;++rows)
        for(auto bundle:{VerifierProjectionBundle::Single,VerifierProjectionBundle::Homogeneous,
                VerifierProjectionBundle::MixedDecoder})
            for(auto bias:{VerifierBias::None,VerifierBias::PerColumn}) {
                if(bundle==VerifierProjectionBundle::Single && bias==VerifierBias::PerColumn)continue;
                ASSERT_NO_FATAL_FAILURE(proveDeviceCountedRows(format,bundle,
                    VerifierCapacityProbe{rows,65,256,VerifierProbePurpose::CorrectnessOnly},bias,
                    candidate));
            }
}
/** @brief Prove actual empty canonical planes. @param format Source identity. @param candidate Native implementation. */
static void proveWmmaEmptyPartitions(const QuantizedVerifierFormatCase &format,NativeQuantisedVerifierCandidate candidate)
{
    for(auto bundle:{VerifierProjectionBundle::Single,VerifierProjectionBundle::Homogeneous,
            VerifierProjectionBundle::MixedDecoder})
        ASSERT_NO_FATAL_FAILURE(proveDeviceCountedRows(format,bundle,
            VerifierCapacityProbe{16,65,768,VerifierProbePurpose::CorrectnessOnly,
                VerifierGridCertificate::EmptyKPartitions},
            bundle==VerifierProjectionBundle::Single?VerifierBias::None:VerifierBias::PerColumn,
            candidate));
}
TEST_P(ROCmDeviceCountedVerifierWmma, PhysicalBucketsAndColumnTails)
{ ASSERT_NO_FATAL_FAILURE(proveWmmaPhysicalBuckets(GetParam(),NativeQuantisedVerifierCandidate::WmmaM16N64)); }
TEST_P(ROCmDeviceCountedVerifierWmma, EmptyCanonicalPartitions)
{ ASSERT_NO_FATAL_FAILURE(proveWmmaEmptyPartitions(GetParam(),NativeQuantisedVerifierCandidate::WmmaM16N64)); }


/**
 * @brief Independent prepared matrices prove mixed codebooks, widths and request origins.
 * @param primary Source format rotated through the complete loader registry.
 * @param candidate Native construction identity; serial oracles never enter it.
 * @param ragged Distinct physical column tails when true, common serial-policy widths otherwise.
 */
static void proveHeterogeneousVerifierRows(const QuantizedVerifierFormatCase &primary,
    NativeQuantisedVerifierCandidate candidate,bool ragged)
{
    int devices=0;
    if(hipGetDeviceCount(&devices)!=hipSuccess || devices==0)GTEST_SKIP()<<"No ROCm device available";
    constexpr int M=14,K=768;
    const std::array<int,3> columns=ragged?std::array<int,3>{17,65,129}:std::array<int,3>{65,65,65};
    const auto device=DeviceId::rocm(0); ScopedGPUStream stream_owner(device);
    const auto stream=static_cast<hipStream_t>(stream_owner.get());
    if(nativeQuantisedHardwareForDevice(0).architecture!=native_quantised::NativeQuantisedArchitecture::rocm(11,0,0))
        GTEST_SKIP()<<"Native verifier WMMA requires actual gfx1100";
    const auto &formats=quantizedMoEVerifierFormats();
    std::array<const QuantizedVerifierFormatCase*,3> sources{&primary,nullptr,nullptr};
    for(int p=1;p<3;++p) {
        auto f=std::find_if(formats.begin(),formats.end(),[&](const auto &entry) {
            return entry.tensor_type==(p==1?TensorType::IQ2_S:TensorType::Q4_1);
        });
        ASSERT_NE(f,formats.end());sources[p]=&*f;
    }
    std::vector<std::unique_ptr<TensorBase>> weights;
    std::vector<TensorBase*> borrowed;
    for(int p=0;p<3;++p) {
        weights.push_back(sources[p]->create({static_cast<size_t>(columns[p]),K},74001+p*101));
        borrowed.push_back(weights.back().get());
    }
    auto prepared=makeGpuPreparedGemmBatch(borrowed,device,"test.verifier.heterogeneous",ModelContextId{74011});
    std::array<ROCmQuantisedGemmKernel*,3> kernels{};
    std::array<std::unique_ptr<DeviceWorkspaceManager>,3> workspaces;
    std::array<std::unique_ptr<FP32Tensor>,3> outputs,serial_outputs,biases;
    std::array<std::vector<float>,3> oracle;
    std::array<std::vector<uint32_t>,3> actual;
    auto input=TestTensorFactory::createFP32Random({M,K},-0.75f,0.75f,74021);
    auto serial_input=TestTensorFactory::createFP32({1,K});
    auto counts=TestTensorFactory::createINT32({2});
    std::array<int32_t,2> published{M,0};
    std::vector<ITensorGemm::TensorProjectionDesc> projections;
    constexpr std::array<const char*,3> labels{"mixed_0","mixed_1","mixed_2"};
    std::unique_ptr<IGPUGraphCapture> graph;
    const VerifierFixtureStreamDrain drain{stream};
    ASSERT_TRUE(input->ensureOnDevice(device,stream));
    ASSERT_TRUE(serial_input->allocateOnDevice(device,stream));
    ASSERT_TRUE(counts->allocateOnDevice(device,stream));
    for(int p=0;p<3;++p) {
        kernels[p]=dynamic_cast<ROCmQuantisedGemmKernel*>(prepared[p].kernel);ASSERT_NE(kernels[p],nullptr);
        kernels[p]->setGPUStream(stream);
        auto req=kernels[p]->getWorkspaceRequirements(M,columns[p],K);
        if(p==0)kernels[p]->appendFusedProjectionWorkspaceRequirements(req,M,columns,K);
        workspaces[p]=std::make_unique<DeviceWorkspaceManager>(device,req.total_bytes_with_alignment());
        ASSERT_TRUE(workspaces[p]->allocate(req));kernels[p]->bindWorkspace(workspaces[p].get());
        outputs[p]=TestTensorFactory::createFP32({M,static_cast<size_t>(columns[p])});
        serial_outputs[p]=TestTensorFactory::createFP32({1,static_cast<size_t>(columns[p])});
        biases[p]=TestTensorFactory::createFP32Random({static_cast<size_t>(columns[p])},-0.125f,0.125f,74031+p);
        ASSERT_TRUE(outputs[p]->allocateOnDevice(device,stream));
        ASSERT_TRUE(serial_outputs[p]->allocateOnDevice(device,stream));
        ASSERT_TRUE(biases[p]->ensureOnDevice(device,stream));
        oracle[p].resize(M*columns[p]);actual[p].resize(M*columns[p]);
        projections.push_back({kernels[p],outputs[p].get(),columns[p],biases[p].get(),labels[p]});
        // Each projection has its own source, output, bias and production M1 oracle.
        for(int row=0;row<M;++row) {
            ASSERT_EQ(hipMemcpyAsync(serial_input->gpu_data_ptr(),
                static_cast<const float*>(input->gpu_data_ptr())+row*K,K*sizeof(float),hipMemcpyDeviceToDevice,stream),hipSuccess);
            TransferEngine::publishCurrentDeviceWrite(serial_input.get(),stream);
            ASSERT_TRUE(kernels[p]->multiply_tensor(serial_input.get(),serial_outputs[p].get(),1,columns[p],K,
                true,1.0f,0.0f,biases[p].get()));
            ASSERT_EQ(hipMemcpyAsync(oracle[p].data()+row*columns[p],serial_outputs[p]->gpu_data_ptr(),
                columns[p]*sizeof(float),hipMemcpyDeviceToHost,stream),hipSuccess);
        }
        ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);
        ASSERT_TRUE(std::any_of(oracle[p].begin(),oracle[p].end(),[](float x){return std::isfinite(x)&&x!=0.0f;}));
    }
    ASSERT_TRUE(kernels[0]->prepareFusedProjectionGraphCapture(3));
    TransferEngine::requireDeviceInput(input.get(),device,stream);
    for(auto &bias:biases)TransferEngine::requireDeviceInput(bias.get(),device,stream);
    for(bool requests:{false,true}) {
        const auto rows=DeviceRowRange::deviceCounted(M,static_cast<const int32_t*>(counts->gpu_data_ptr()));
        const auto work=ProjectionRowWork::fromRequests(DeviceRequestRowRanges::deviceCounted(
            2,8,static_cast<const int32_t*>(counts->gpu_data_ptr()))).slice(1,M);
        graph=GPUDeviceContextPool::instance().retainAMDContext(0)->createGraphCapture(stream);
        {
            std::optional<ProjectionRowWorkScope> work_scope;
            if(requests)work_scope.emplace(work);
            auto arithmetic=requests?kernels[0]->beginVerifierDecodeEquivalentScope():kernels[0]->beginVerifierDecodeEquivalentScope(rows);
            ScopedNativeQuantisedVerifierCandidate selection(candidate,0);
            ScopedBackendGraphCapture capture(*graph,"heterogeneous_verifier_rows");ASSERT_TRUE(capture.begin());
            ASSERT_TRUE(kernels[0]->multiply_fused_verifier_rows_decode_equivalent(input.get(),projections,M,K));
            capture.finish();
            std::vector<GPUGraphKernelNodeInfo> nodes;std::string error;
            ASSERT_TRUE(graph->inspectKernelNodes(nodes,&error))<<error;
            int matrix_packets=0;
            for(const auto &node:nodes) {
                ASSERT_TRUE(node.valid());
                if(node.name.find("native_quantised_verifier_wmma_m16n64")!=std::string::npos)++matrix_packets;
                EXPECT_EQ(node.name.find("native_quantised_verifier_ordered"),std::string::npos);
                EXPECT_EQ(node.name.find("gemv_native_quantised_kernel_t"),std::string::npos);
                EXPECT_EQ(node.name.find("gemv_native_quantised_small_m_kernel_t"),std::string::npos);
            }
            ASSERT_GE(matrix_packets,1);ASSERT_LE(matrix_packets,3);

            ASSERT_TRUE(graph->instantiate());
        }
        const std::vector<std::array<int32_t,2>> publications=requests
            ?std::vector<std::array<int32_t,2>>{{5,4},{6,5},{5,4},{5,5},{6,4},{0,0},{8,8},{5,4}}
            :std::vector<std::array<int32_t,2>>{{4,0},{5,0},{4,0},{M,0},{0,0},{1,0},{M,0},{4,0}};
        for(auto next:publications) {
            published=next;
            for(auto &workspace:workspaces)
                for(const auto *id:{GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL,GemmWorkspaceBuffers::QUANT_A,
                    GemmWorkspaceBuffers::SCALES_A_BLOCKWISE,GemmWorkspaceBuffers::SUMS_A_BLOCKWISE})
                    ASSERT_EQ(hipMemsetAsync(workspace->getBuffer(id),0xa5,workspace->getBufferSize(id),stream),hipSuccess);
            ASSERT_EQ(hipMemsetAsync(workspaces[0]->getBuffer(GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL_BATCHED),0xa5,
                workspaces[0]->getBufferSize(GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL_BATCHED),stream),hipSuccess);
            for(int p=0;p<3;++p)ASSERT_EQ(hipMemsetAsync(outputs[p]->gpu_data_ptr(),0xa5,actual[p].size()*sizeof(uint32_t),stream),hipSuccess);
            ASSERT_EQ(hipMemcpyAsync(counts->gpu_data_ptr(),published.data(),sizeof(published),hipMemcpyHostToDevice,stream),hipSuccess);
            ASSERT_TRUE(graph->launch());
            for(int p=0;p<3;++p)ASSERT_EQ(hipMemcpyAsync(actual[p].data(),outputs[p]->gpu_data_ptr(),actual[p].size()*sizeof(uint32_t),hipMemcpyDeviceToHost,stream),hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);
            for(int row=0;row<M;++row) {
                const bool active=requests?(1+row)%8<published[(1+row)/8]:row<published[0];
                for(int p=0;p<3;++p)for(int n=0;n<columns[p];++n) {
                    uint32_t expected=0xa5a5a5a5u;
                    if(active)std::memcpy(&expected,&oracle[p][row*columns[p]+n],sizeof(expected));
                    ASSERT_EQ(actual[p][row*columns[p]+n],expected)<<"projection="<<p<<" row="<<row<<" column="<<n;
                }
            }
        }
        graph.reset();
    }
    for(auto *kernel:kernels){kernel->unbindWorkspace();kernel->clearGPUStreamBinding();}
}


/** @brief Forced native mixed bundles retain independent source bytes and request holes. */
TEST_P(ROCmDeviceCountedVerifierWmma, SerialBytesHeterogeneousInputs)
{
    for(bool ragged:{false,true})
        ASSERT_NO_FATAL_FAILURE(proveHeterogeneousVerifierRows(GetParam(),NativeQuantisedVerifierCandidate::WmmaM16N64,ragged));
}

INSTANTIATE_TEST_SUITE_P(AllFormats,ROCmDeviceCountedVerifierWmma,
    ::testing::ValuesIn(quantizedMoEVerifierFormats()),
    [](const ::testing::TestParamInfo<QuantizedVerifierFormatCase> &info){return std::string(info.param.label);});

/**
 * @brief Measure complete-operation economy on the first production IQ4_XS verifier shapes.
 * @param purpose Endpoint panel or every original live-prefix count.
 * @param candidate Forced matrix or installed Automatic producer compared against original Dot4.
 */
static void proveQwen38PairedWmma(VerifierProbePurpose purpose,
    NativeQuantisedVerifierCandidate candidate=NativeQuantisedVerifierCandidate::WmmaM16N64)
{
    const auto &formats=quantizedMoEVerifierFormats();
    const auto format=std::find_if(formats.begin(),formats.end(),[](const auto &entry){return entry.tensor_type==TensorType::IQ4_XS;});
    ASSERT_NE(format,formats.end());
    for(const auto [columns,width]:std::array<std::pair<int,int>,3>{{{17408,5120},{5120,17408},{10240,5120}}})
        for(int capacity : (purpose==VerifierProbePurpose::PairedExactWidth || purpose==VerifierProbePurpose::PairedAutomaticExactWidth)
            ? std::vector<int>{2,3,4,5,6,7,8,9,10,11,12,13,14,15,16} : std::vector<int>{3,16})
            for(auto operation:{VerifierProjectionBundle::TensorProjection,VerifierProjectionBundle::TensorSwiGLU})
                ASSERT_NO_FATAL_FAILURE(proveDeviceCountedRows(*format,operation,
                    VerifierCapacityProbe{capacity,columns,width,purpose},
                    VerifierBias::None,candidate));
}

/** @brief Endpoint panel preserves the initial paired comparison identity. */
TEST(ROCmDeviceCountedVerifierRowsEconomy, Qwen38PairedWmmaCapacity)
{ ASSERT_NO_FATAL_FAILURE(proveQwen38PairedWmma(VerifierProbePurpose::PairedCandidates)); }
/** @brief Every positive live prefix resolves the measured device-side crossover. */
TEST(ROCmDeviceCountedVerifierRowsEconomy, Qwen38PairedWmmaAllCounts)
{ ASSERT_NO_FATAL_FAILURE(proveQwen38PairedWmma(VerifierProbePurpose::PairedAllCounts)); }

/** @brief Exact hosted physical widths are measured independently of a capacity-16 prefix. */
TEST(ROCmDeviceCountedVerifierRowsEconomy, Qwen38PairedWmmaExactWidths)
{ ASSERT_NO_FATAL_FAILURE(proveQwen38PairedWmma(VerifierProbePurpose::PairedExactWidth)); }

/** @brief Installed Automatic uses original Dot4 or native WMMA in exact hosted graphs. */
TEST(ROCmDeviceCountedVerifierRowsEconomy, Qwen38AutomaticExactWidths)
{ ASSERT_NO_FATAL_FAILURE(proveQwen38PairedWmma(VerifierProbePurpose::PairedAutomaticExactWidth,
    NativeQuantisedVerifierCandidate::Automatic)); }

/** @brief Independent source-codebook economy cells use the same complete public byte oracle. */
class ROCmVerifierWmmaFormatEconomy : public ::testing::TestWithParam<QuantizedVerifierFormatCase> {};

/**
 * @brief Measure one original codebook without borrowing another format's crossover.
 * @param format Complete source-format registry entry, including normalized aliases.
 * @param widths Exact physical transaction widths, never a mutable host count.
 * @details Every cell retains the existing31 alternating pairs of eight public
 * projection/SwiGLU operations, authenticated native inventory, serial bytes
 * and poisoned partial/empty/regrown replay. This is performance-only work.
 */
static void proveCodebookExactWidths(const QuantizedVerifierFormatCase &format,
    std::span<const int> widths)
{
    initializeROCmHardwareMetadata();
    int device_count=0;
    if(hipGetDeviceCount(&device_count)!=hipSuccess || device_count==0)
        GTEST_SKIP()<<"No ROCm device available";
    if(nativeQuantisedHardwareForDevice(0).architecture!=
        native_quantised::NativeQuantisedArchitecture::rocm(11,0,0))
        GTEST_SKIP()<<"Native verifier WMMA requires actual gfx1100";
    for(const auto [columns,width]:std::array<std::pair<int,int>,3>{{{17408,5120},{5120,17408},{10240,5120}}}) {
        VerifierPreparedSource source(format,DeviceId::rocm(0),columns,width);
        for(int rows:widths)
            for(auto operation:{VerifierProjectionBundle::TensorProjection,VerifierProjectionBundle::TensorSwiGLU})
                ASSERT_NO_FATAL_FAILURE(proveDeviceCountedRows(format,operation,
                    VerifierCapacityProbe{rows,columns,width,VerifierProbePurpose::PairedExactWidth},
                    VerifierBias::None,NativeQuantisedVerifierCandidate::WmmaM16N64,&source));
    }
}

/** @test Prepared sources survive grow/shrink/API reuse while every case binding retires. */
TEST(ROCmVerifierWmmaDispatch, PreparedSourceReuse)
{
    initializeROCmHardwareMetadata();
    int device_count=0;
    if(hipGetDeviceCount(&device_count)!=hipSuccess || device_count==0)
        GTEST_SKIP()<<"No ROCm device available";
    if(nativeQuantisedHardwareForDevice(0).architecture!=
        native_quantised::NativeQuantisedArchitecture::rocm(11,0,0))
        GTEST_SKIP()<<"Native verifier WMMA requires actual gfx1100";
    const auto &formats=quantizedMoEVerifierFormats();
    const auto found=std::find_if(formats.begin(),formats.end(),
        [](const auto &format){return format.tensor_type==TensorType::IQ4_XS;});
    ASSERT_NE(found,formats.end());
    const auto &format=*found;
    const auto device=DeviceId::rocm(0);
    constexpr int n=65,k=768;
    VerifierPreparedSource source(format,device,n,k);
    auto *kernel=&source.kernel(format,device,n,k);
    DeviceNativeQuantisedMatrixDesc original;
    ASSERT_TRUE(kernel->exportNativeQuantisedMatrixDesc(original));
    for(const auto [rows,operation]:std::array<std::pair<int,VerifierProjectionBundle>,4>{{
        {5,VerifierProjectionBundle::TensorProjection},{16,VerifierProjectionBundle::TensorSwiGLU},
        {2,VerifierProjectionBundle::TensorProjection},{5,VerifierProjectionBundle::TensorSwiGLU}}}) {
        ASSERT_NO_FATAL_FAILURE(proveDeviceCountedRows(format,operation,
            VerifierCapacityProbe{rows,n,k,VerifierProbePurpose::CorrectnessOnly},
            VerifierBias::None,NativeQuantisedVerifierCandidate::WmmaM16N64,&source));
        EXPECT_EQ(&source.kernel(format,device,n,k),kernel);
        EXPECT_FALSE(kernel->hasWorkspace());
        EXPECT_THROW((void)kernel->requireGPUStream(),std::runtime_error);
        DeviceNativeQuantisedMatrixDesc current;
        ASSERT_TRUE(kernel->exportNativeQuantisedMatrixDesc(current));
        EXPECT_EQ(current.payload,original.payload);
    }
    auto alias=format;alias.tensor_type=TensorType::IQ4_NL;alias.source_is_superblock=false;
    EXPECT_THROW((void)source.kernel(alias,device,n,k),std::invalid_argument);
    EXPECT_THROW((void)source.kernel(format,device,n+1,k),std::invalid_argument);
    EXPECT_THROW((void)source.kernel(format,device,n,k+256),std::invalid_argument);
    EXPECT_THROW((void)source.kernel(format,DeviceId::rocm(1),n,k),std::invalid_argument);
}

/** @brief First bracket shallow, crossover and full verifier widths for every source format. */
TEST_P(ROCmVerifierWmmaFormatEconomy, DISABLED_ExactWidthEndpoints)
{
    constexpr std::array widths{4,5,8,16};
    ASSERT_NO_FATAL_FAILURE(proveCodebookExactWidths(GetParam(),widths));
}

/** @brief Resolve every admitted physical width before installing an extended measured policy. */
TEST_P(ROCmVerifierWmmaFormatEconomy, DISABLED_ExactWidthSweep)
{
    constexpr std::array widths{2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    ASSERT_NO_FATAL_FAILURE(proveCodebookExactWidths(GetParam(),widths));
}

INSTANTIATE_TEST_SUITE_P(AllFormats,ROCmVerifierWmmaFormatEconomy,
    ::testing::ValuesIn(quantizedMoEVerifierFormats()),
    [](const ::testing::TestParamInfo<QuantizedVerifierFormatCase> &info){return std::string(info.param.label);});


/**
 * @brief Profile one exact public projection producer without byte/timing replays.
 * @param candidate Exact construction identity. @param live Original device-owned count.
 */
static void profileVerifierWmma(NativeQuantisedVerifierCandidate candidate,int live)
{
    const auto &formats=quantizedMoEVerifierFormats();
    const auto format=std::find_if(formats.begin(),formats.end(),[](const auto &entry){return entry.tensor_type==TensorType::IQ4_XS;});
    ASSERT_NE(format,formats.end());
    ASSERT_NO_FATAL_FAILURE(proveDeviceCountedRows(*format,VerifierProjectionBundle::TensorProjection,
        VerifierCapacityProbe{candidate==NativeQuantisedVerifierCandidate::Automatic?live:16,5120,17408,
            candidate==NativeQuantisedVerifierCandidate::Automatic?VerifierProbePurpose::IsolatedAutomaticProfile:VerifierProbePurpose::IsolatedProfile,
            VerifierGridCertificate::NotRequested,live},VerifierBias::None,candidate));
}


/** @brief Original forced matrix control at the same shape. */
TEST(ROCmVerifierWmmaProfile, WmmaLive16)
{ ASSERT_NO_FATAL_FAILURE(profileVerifierWmma(NativeQuantisedVerifierCandidate::WmmaM16N64,16)); }
/** @brief Original four-row ordered control at the same shape. */
TEST(ROCmVerifierWmmaProfile, Dot4Live4)
{ ASSERT_NO_FATAL_FAILURE(profileVerifierWmma(NativeQuantisedVerifierCandidate::Dot4,4)); }

/** @brief Isolate the installed shallow Dot4 producer in its exact hosted graph. */
TEST(ROCmVerifierWmmaProfile, AutomaticExact4)
{ ASSERT_NO_FATAL_FAILURE(profileVerifierWmma(NativeQuantisedVerifierCandidate::Automatic,4)); }
/** @brief Isolate installed WMMA at the first measured winning width. */
TEST(ROCmVerifierWmmaProfile, AutomaticExact5)
{ ASSERT_NO_FATAL_FAILURE(profileVerifierWmma(NativeQuantisedVerifierCandidate::Automatic,5)); }
/** @brief Isolate installed WMMA at the largest admitted single-request verifier width. */
TEST(ROCmVerifierWmmaProfile, AutomaticExact16)
{ ASSERT_NO_FATAL_FAILURE(profileVerifierWmma(NativeQuantisedVerifierCandidate::Automatic,16)); }

/** @test Installed dispatch is total over widths/provenance and refuses borrowed or incomplete work. */
TEST(ROCmVerifierWmmaDispatch, ExactHostedKeysAndExclusions)
{
    initializeROCmHardwareMetadata();
    if(rocmHardwareMetadata().size()==0 || nativeQuantisedHardwareForDevice(0).architecture !=
        native_quantised::NativeQuantisedArchitecture::rocm(11,0,0))
        GTEST_SKIP()<<"UNQUALIFIED: native verifier policy requires actual gfx1100";
    int32_t count=0, foreign_count=0;
    uint8_t payload=0, foreign_payload=0;
    const auto device=DeviceId::rocm(0);
    const auto select=[&](CapturedVerifierExtent extent,
                          std::optional<NativeQuantisedVerifierOperation> operation,
                          const NativeQuantisedVerifierWork &work,bool decode=true) {
        ScopedCapturedVerifierExtent admitted(extent);
        ScopedNativeQuantisedVerifierOperation original(operation);
        return nativeQuantisedVerifierProducer(0,decode,work);
    };
    const auto dot=NativeQuantisedVerifierCandidate::Dot4;
    const auto matrix=NativeQuantisedVerifierCandidate::WmmaM16N64;
    for(int m=2;m<=33;++m)
        for(const auto [n,k]:std::array<std::pair<int,int>,6>{{{17408,5120},{5120,17408},{10240,5120},{17408,5088},{5121,17408},{8192,2048}}})
            for(auto kind:{NativeQuantisedVerifierOperationKind::Projection,NativeQuantisedVerifierOperationKind::SwiGLUDown}) {
                const auto rows=DeviceRowRange::deviceCounted(m,&count);
                const auto extent=CapturedVerifierExtent::exactHostedRequest(device,rows);
                const NativeQuantisedVerifierOperation operation{kind,0,m,n,k,&payload,{4,true,true}};
                const NativeQuantisedVerifierWork work{rows,&payload,n,k,1,24,NativeQuantisedVerifierPolicyMatch::CanonicalSerial};
                const bool measured=m>=5 && m<=16 && ((n==17408 && k==5120)||(n==5120 && k==17408)||(n==10240 && k==5120));
                EXPECT_EQ(select(extent,operation,work),measured?matrix:dot);
                EXPECT_EQ(select({},operation,work),dot);
                EXPECT_EQ(select(extent,std::nullopt,work),dot);
            }
    const auto rows=DeviceRowRange::deviceCounted(5,&count);
    const auto extent=CapturedVerifierExtent::exactHostedRequest(device,rows);
    const NativeQuantisedVerifierOperation original{NativeQuantisedVerifierOperationKind::Projection,0,5,17408,5120,&payload,{4,true,true}};
    const NativeQuantisedVerifierWork original_work{rows,&payload,17408,5120,1,24,NativeQuantisedVerifierPolicyMatch::CanonicalSerial};
    for(int source=0;source<=25;++source)for(bool superblock:{false,true})for(bool present:{false,true}) {
        auto operation=original;operation.source={static_cast<uint8_t>(source),superblock,present};
        EXPECT_EQ(select(extent,operation,original_work),source==4 && superblock && present?matrix:dot);
    }
    for(int physical=0;physical<=32;++physical) {
        auto work=original_work;work.physical_codebook=static_cast<uint8_t>(physical);
        EXPECT_EQ(select(extent,original,work),physical==24?matrix:dot);
    }
    for(int mutation=0;mutation<13;++mutation) {
        SCOPED_TRACE(mutation);
        auto operation=original;auto work=original_work;
        switch(mutation) {
        case 0:work.rows=DeviceRowRange::deviceCounted(5,&foreign_count);break;
        case 1:work.rows=DeviceRowRange::deviceCounted(6,&count).slice(0,5);break;
        case 2:work.rows=DeviceRowRange::deviceCounted(6,&count).slice(1,5);break;
        case 3:work.rows=DeviceRowRange::deviceCounted(5,&count).repeatEachRow(2).slice(0,5);break;
        case 4:work.rows=DeviceRowRange::fullyActive(5);break;
        case 5:work.weights=&foreign_payload;break;
        case 6:work.projections=2;break;
        case 7:work.policy=NativeQuantisedVerifierPolicyMatch::Unmeasured;break;
        case 8:operation.weights=nullptr;break;
        case 9:operation.device=1;break;
        case 10:operation.m=6;break;
        case 11:operation.n=10240;break;
        case 12:operation.k=17408;break;
        }
        EXPECT_EQ(select(extent,operation,work),dot);
    }
    EXPECT_EQ(select(extent,original,original_work,false),dot);
    {
        ProjectionRowWorkScope generic(ProjectionRowWork::fromPrefix(rows));
        EXPECT_EQ(select(extent,original,original_work),dot);
    }
    for(auto forced:{dot,matrix}) {
        ScopedNativeQuantisedVerifierCandidate diagnostic(forced,0);
        EXPECT_EQ(select(extent,original,original_work),forced);
    }
    EXPECT_FALSE(ScopedNativeQuantisedVerifierOperation::current());
    EXPECT_FALSE(ScopedCapturedVerifierExtent::current().isExact());
}

/** @brief Record the public operation through the actual production capture controller. */
class VerifierPublicOperationStage final : public IComputeStage
{
public:
    /** @param device Original participant. @param input Original input publication. @param output Declared output owner.
     * @param rows Physical output rows. @param columns Physical output columns. @param submit Real public API. */
    VerifierPublicOperationStage(DeviceId device,TensorBase *input,TensorBase *output,int rows,int columns,
        std::function<bool(StageGPUExecution)> submit)
        : IComputeStage(device),input_(input),output_(output),rows_(rows),columns_(columns),submit_(std::move(submit)) {}
    /** @brief Use the executor's exact native stream and worker for the public operation. */
    bool execute(IDeviceContext*) override { return submit_(gpuExecution()); }
    /** @brief Join the original input publication before capture on its exact worker/stream. */
    bool prepareGraphLaunch(IDeviceContext *ctx,void *stream) override {
        if(!IComputeStage::prepareGraphLaunch(ctx,stream))return false;
        TransferEngine::requireDeviceInput(input_,gpuExecution().device(),stream);
        return true;
    }
    /** @return Only construction joins inputs; native replay preserves its captured ownership. */
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override { return GraphLaunchPreparationPolicy::CaptureOnly; }
    /** @return This stage performs one real grouped projection. */
    ComputeStageType type() const override { return ComputeStageType::GEMM; }
    /** @return Only the native backend under qualification. */
    bool supportsBackend(ComputeBackendType backend) const override { return backend==ComputeBackendType::GPU_ROCM; }
    /** @return Original input and preallocated output names consumed by the real capture ledger. */
    StageBufferContract bufferContract() const override {
        return StageBufferContract::build().addInput(BufferId::HIDDEN_STATE,"FP32")
            .addPreallocatedOutput(BufferId::FFN_OUTPUT,"FP32");
    }
private:
    /** @return Original output publication contract used by the production capture ledger. */
    StageDumpInfo buildDumpInfoImpl() const override {
        StageDumpInfo info;
        info.outputs.push_back({.name="verifier_output",.data=nullptr,.rows=rows_,.cols=columns_,
            .dtype="FP32",.element_size=sizeof(float),.byte_size=static_cast<size_t>(rows_)*columns_*sizeof(float),
            .tensor=output_});
        return info;
    }
    TensorBase *input_; ///< Original published tensor, outliving graph preparation and replay.
    TensorBase *output_; ///< Original tensor whose publication belongs to this stage.
    int rows_,columns_; ///< Immutable physical output geometry.
    std::function<bool(StageGPUExecution)> submit_; ///< Outlived by original tensors, engine and workspace.
};

/** @test Production worker capture installs Automatic, retains row guards and never re-records on replay. */
TEST(ROCmVerifierWmmaDispatch, NativeWorkerCaptureAndDirtyReplay)
{
    initializeROCmHardwareMetadata();
    if(rocmHardwareMetadata().size()==0 || nativeQuantisedHardwareForDevice(0).architecture !=
        native_quantised::NativeQuantisedArchitecture::rocm(11,0,0))
        GTEST_SKIP()<<"UNQUALIFIED: native verifier worker proof requires actual gfx1100";
    const auto &formats=quantizedMoEVerifierFormats();
    const auto format=std::find_if(formats.begin(),formats.end(),[](const auto &entry){return entry.tensor_type==TensorType::IQ4_XS;});
    ASSERT_NE(format,formats.end());
    const auto device=DeviceId::rocm(0);
    auto worker=GPUDeviceContextPool::instance().retainAMDContext(0);
    ROCmDeviceContext context(device,0);
    for(int m:{4,5,16})for(bool swiglu:{false,true}) {
        SCOPED_TRACE(::testing::Message()<<"M="<<m<<" swiglu="<<swiglu);
        constexpr int n=5120,k=17408;
        ScopedGPUStream stream_owner(device);const auto stream=static_cast<hipStream_t>(stream_owner.get());
        auto weights=format->create({n,k},72061);
        auto prepared=makeGpuPreparedGemm(weights.get(),device,"test.verifier.worker",ModelContextId{720610});
        auto *kernel=dynamic_cast<ROCmQuantisedGemmKernel*>(prepared.kernel);ASSERT_NE(kernel,nullptr);
        kernel->setGPUStream(stream);
        const auto requirements=kernel->getWorkspaceRequirements(m,n,k);
        auto workspace=std::make_unique<DeviceWorkspaceManager>(device,requirements.total_bytes_with_alignment());
        ASSERT_TRUE(workspace->allocate(requirements));kernel->bindWorkspace(workspace.get());
        ASSERT_TRUE(kernel->prepareFusedProjectionGraphCapture(1));
        auto input=TestTensorFactory::createFP32Random({static_cast<size_t>(m),k},-.75f,.75f,72062);
        auto output=TestTensorFactory::createFP32({static_cast<size_t>(m),n});
        auto serial_input=TestTensorFactory::createFP32({1,k});
        auto serial_output=TestTensorFactory::createFP32({1,n});
        auto count=TestTensorFactory::createINT32({1});
        std::vector<uint32_t> oracle(static_cast<size_t>(m)*n),actual(oracle.size());
        int32_t live=m;
        const VerifierFixtureStreamDrain operands_drain{stream};
        TransferEngine::prepareDeviceInput(input.get(),device,stream);
        ASSERT_TRUE(output->allocateOnDevice(device,stream));
        ASSERT_TRUE(serial_input->allocateOnDevice(device,stream));
        ASSERT_TRUE(serial_output->allocateOnDevice(device,stream));
        ASSERT_TRUE(count->allocateOnDevice(device,stream));
        for(int row=0;row<m;++row) {
            ASSERT_EQ(hipMemcpyAsync(serial_input->gpu_data_ptr(),static_cast<const float*>(input->gpu_data_ptr())+row*k,
                k*sizeof(float),hipMemcpyDeviceToDevice,stream),hipSuccess);
            TransferEngine::publishCurrentDeviceWrite(serial_input.get(),stream);
            ASSERT_TRUE(swiglu ? kernel->multiply_tensor_with_fused_swiglu(serial_input.get(),serial_input.get(),serial_output.get(),1,n,k)
                : kernel->multiply_tensor(serial_input.get(),serial_output.get(),1,n,k,true,1.f,0.f,nullptr));
            ASSERT_EQ(hipMemcpyAsync(oracle.data()+row*n,serial_output->gpu_data_ptr(),n*sizeof(float),hipMemcpyDeviceToHost,stream),hipSuccess);
        }
        ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);
        const auto rows=DeviceRowRange::deviceCounted(m,static_cast<const int32_t*>(count->gpu_data_ptr()));
        const auto exact=CapturedVerifierExtent::exactHostedRequest(device,rows);
        for(bool admitted:{false,true}) {
            SCOPED_TRACE(admitted);
            const auto expected=admitted?exact:CapturedVerifierExtent{};
            int stage_calls=0;
            ComputeGraph graph;
            graph.addNode("original_verifier",std::make_unique<VerifierPublicOperationStage>(device,input.get(),output.get(),m,n,[&](StageGPUExecution execution) {
                if(!worker->ownsCurrentThread() || execution.nativeStream()!=stream || ScopedCapturedVerifierExtent::current()!=expected)
                    throw std::logic_error("Verifier capture lost original worker, stream or extent");
                ++stage_calls;execution.bind(kernel);
                auto arithmetic=kernel->beginVerifierDecodeEquivalentScope(rows);
                return swiglu ? kernel->multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(input.get(),input.get(),output.get(),m,n,k)
                    : kernel->multiply_fused_verifier_rows_decode_equivalent(input.get(),{{kernel,output.get(),n,nullptr,"original"}},m,k);
            }),device);
            graph.setTerminalNode("original_verifier");
            BufferArena arena;
            ASSERT_TRUE(arena.registerExternalBuffer(BufferId::HIDDEN_STATE,input.get()));
            ASSERT_TRUE(arena.registerExternalBuffer(BufferId::FFN_OUTPUT,output.get()));
            DeviceGraphExecutor executor;executor.setArena(&arena);
            executor.setWorkerGPUContextResolver([&](DeviceId owner)->IWorkerGPUContext* {return owner==device?worker.get():nullptr;});
            DeviceGraphExecutor::GraphSegmentCache cache;
            ASSERT_TRUE(cache.bindBorrowedCaptureStream(worker.get(),stream,device));
            const VerifierFixtureStreamDrain graph_drain{stream};
            DeviceGraphExecutor::DecodeCapturePolicy policy;
            policy.allow_cached_graph_replay=true;policy.defer_final_sync=true;policy.captured_verifier_extent=expected;
            const auto execute=[&] {
                graph.reset();
                return executor.executeDecodeWithCapturePolicy(graph,&context,&cache,stream,worker.get(),nullptr,policy);
            };
            live=m;ASSERT_EQ(hipMemcpyAsync(count->gpu_data_ptr(),&live,sizeof(live),hipMemcpyHostToDevice,stream),hipSuccess);
            ASSERT_TRUE(execute());ASSERT_EQ(cache.segments.size(),1u);
            const auto record_count=stage_calls;ASSERT_GT(record_count,0);
            std::vector<GPUGraphKernelNodeInfo> nodes;std::string error;
            ASSERT_TRUE(cache.segments.front().capture->inspectKernelNodes(nodes,&error))<<error;
            int matrices=0,ordered=0,dots=0;
            for(const auto &node:nodes) {
                EXPECT_EQ(node.name.find("gemv_native_quantised_kernel_t"),std::string::npos)
                    << "Native worker capture must not record serial-row repair";
                matrices+=node.name.find("native_quantised_verifier_wmma_m16n64")!=std::string::npos;
                ordered+=node.name.find("native_quantised_verifier_ordered_m4n64")!=std::string::npos;
                dots+=node.name.find("gemv_native_quantised_small_m_")!=std::string::npos;
            }
            EXPECT_EQ(matrices,admitted && m>=5?1:0);EXPECT_EQ(ordered,0);
            EXPECT_EQ(dots>0,!(admitted && m>=5));
            for(int replay=0;replay<2;++replay)for(int active:{m,0,4,1,m}) {
                live=active;
                ASSERT_EQ(hipMemsetAsync(output->gpu_data_ptr(),0xa5,actual.size()*sizeof(uint32_t),stream),hipSuccess);
                ASSERT_EQ(hipMemsetAsync(workspace->getBuffer(GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL),0xa5,
                    workspace->getBufferSize(GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL),stream),hipSuccess);
                ASSERT_EQ(hipMemcpyAsync(count->gpu_data_ptr(),&live,sizeof(live),hipMemcpyHostToDevice,stream),hipSuccess);
                ASSERT_TRUE(execute());
                ASSERT_EQ(hipMemcpyAsync(actual.data(),output->gpu_data_ptr(),actual.size()*sizeof(uint32_t),hipMemcpyDeviceToHost,stream),hipSuccess);
                ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);
                ASSERT_EQ(std::memcmp(actual.data(),oracle.data(),static_cast<size_t>(active)*n*sizeof(uint32_t)),0);
                ASSERT_TRUE(std::all_of(actual.begin()+static_cast<size_t>(active)*n,actual.end(),[](uint32_t word){return word==0xa5a5a5a5;}));
                EXPECT_EQ(stage_calls,record_count)<<"Native replay must not reconstruct the operation";
                EXPECT_FALSE(ScopedCapturedVerifierExtent::current().isExact());
            }
        }
    }
}
