/**
 * @file Test__FixedKPrefillProjection.cpp
 * @brief Chunk- and schedule-invariance regressions for explicit-operand GPU prefill (#74).
 *
 * Ordinary half-RNE GDN prefill projections used vendor BLAS whose reduction
 * order followed the physical row count, so prefix-cache checkpoints that split
 * a cold prompt into different chunks changed its tokens. The fixed-K contract
 * makes every prompt row independent of chunking, bucket and row offset.
 *
 * Two layers are proven on the real backend:
 *  - kernel schedules: every compiled fixed-K schedule produces identical row
 *    bytes for every M and row offset, against one reference launch;
 *  - stage replay: the production GDNProjectionStage, captured once per bucket,
 *    replays a 778-row prompt under the issue's chunkings (512+256+10 versus
 *    512+266) and others, with identical bytes for every explicit projection.
 *
 * CUDA and ROCm compile this file separately so each process owns one runtime.
 */
#include "backends/GPUDeviceProcessLifetime.h"
#include "utils/MPISessionOwner.h"
#include "../../utils/FixedKPrefillChunkHarness.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/ExplicitGPUStream.h"
#ifdef LLAMINAR_FIXED_K_CUDA
#include "backends/cuda/CUDAGraphCapture.h"
#include "kernels/cuda/gemm/CUDAFixedKProjectionLaunch.h"
#include <cuda_runtime.h>
#else
#include "backends/rocm/HIPGraphCapture.h"
#endif
#include "backends/ComputeBackend.h"
#include "utils/DebugEnv.h"
#include "utils/MPIContext.h"

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
#ifdef LLAMINAR_FIXED_K_CUDA
    constexpr const char *kRouteCounter = "gemm.cuda_fixed_k_projection_calls";
    /** @return The test's single physical participant. */
    DeviceId testDevice() { return DeviceId::cuda(0); }
#else
    constexpr const char *kRouteCounter = "gemm.rocm_fixed_k_projection_calls";
    DeviceId testDevice() { return DeviceId::rocm(0); }
#endif

    /** @brief The issue's prompt and its prefix-cache ON/OFF chunkings, plus adversarial splits. */
    const std::vector<FixedKChunkSchedule> &issueSchedules()
    {
        static const std::vector<FixedKChunkSchedule> schedules{
            {"prefix_off_512_266", {512, 266}},
            {"prefix_on_512_256_10", {512, 256, 10}},
            {"single_778", {778}},
            {"ragged_1_63_65_127_129_393", {1, 63, 65, 127, 129, 393}},
            {"buckets_64x12_10", {64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10}},
            {"odd_448_330", {448, 330}},
        };
        return schedules;
    }

    /**
     * @brief Run one stage replay on the real worker of @p device.
     * @param wide_format Native QKV/Z format in alpha/beta-only mode.
     * @param k Reduction width. @param columns QKV, Z, alpha, beta widths.
     * @param full_half All four projections half-RNE (explicit full-FP16 mode).
     */
    void replayOnDevice(TensorType wide_format, int k, std::array<int, 4> columns, bool full_half)
    {
        const DeviceId device = testDevice();
        DeviceManager::instance().initialize(-1);
        auto &worker = GPUDeviceContextPool::instance().getContext(device);
        const auto &formats = quantizedVerifierFormats();
        const auto format = std::find_if(formats.begin(), formats.end(),
            [&](const auto &candidate) { return candidate.tensor_type == wide_format; });
        ASSERT_NE(format, formats.end());
        worker.submitAndWait([&]
        {
#ifdef LLAMINAR_FIXED_K_CUDA
            CUDADeviceContext context(device, device.ordinal);
#else
            ROCmDeviceContext context(device, device.ordinal);
#endif
            FixedKGraphFactory factory = [device](void *stream)
            { return GPUDeviceContextPool::instance().retainContext(device)->createGraphCapture(stream); };
            ASSERT_NO_FATAL_FAILURE(runFixedKChunkReplay(device, context, factory, &worker, *format,
                k, columns, full_half, issueSchedules(), kRouteCounter));
        });
    }
}

/** @test Qwen3.6-35B-A3B default policy: IQ3_S QKV/Z with half alpha/beta, full and TP2 widths. */
TEST(FixedKPrefillProjection, HalfAlphaBetaChunkReplayQwen36)
{
    ASSERT_NO_FATAL_FAILURE(replayOnDevice(TensorType::IQ3_S, 2048, {512, 256, 32, 32}, false));
    ASSERT_NO_FATAL_FAILURE(replayOnDevice(TensorType::IQ3_S, 2048, {256, 128, 16, 16}, false));
}

/** @test QFN geometry: K=2560 half alpha/beta at full and tensor-parallel shard widths. */
TEST(FixedKPrefillProjection, HalfAlphaBetaChunkReplayQFN)
{
    ASSERT_NO_FATAL_FAILURE(replayOnDevice(TensorType::Q5_K, 2560, {512, 256, 48, 48}, false));
    ASSERT_NO_FATAL_FAILURE(replayOnDevice(TensorType::Q5_K, 2560, {256, 128, 12, 12}, false));
}

/** @test Explicit full-FP16 GDN: every projection half-RNE, exercising the wide tiled schedules. */
TEST(FixedKPrefillProjection, FullHalfGDNChunkReplay)
{
    ASSERT_NO_FATAL_FAILURE(replayOnDevice(TensorType::Q5_K, 2048, {2048, 1024, 32, 32}, true));
}

#ifdef LLAMINAR_FIXED_K_CUDA
/**
 * @test Every compiled CUDA half schedule gives identical rows for every M and offset.
 *
 * One reference launch at M=1024 fixes the expected bytes of each input row.
 * Each schedule then evaluates windows of the same rows at M in 1..1024 and row
 * offsets that move rows within and across tile boundaries. Ragged groups cover
 * narrow and wide widths, tails that are not multiples of eight and K tails.
 */
TEST(FixedKPrefillProjection, CUDAHalfSchedulesByteIdentical)
{
    DeviceManager::instance().initialize(-1);
    auto &worker = GPUDeviceContextPool::instance().getContext(DeviceId::cuda(0));
    worker.submitAndWait([&]
    {
        struct Shape { int k; std::vector<int> widths; };
        const std::vector<Shape> shapes{{2048, {32, 32}}, {2560, {48, 48}}, {2056, {13, 21}},
                                        {2048, {520, 264, 8, 8}}, {192, {100, 36}}, {65, {16, 300}}, {2044, {32, 32}}};
        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
        for (const auto &shape : shapes)
        {
            SCOPED_TRACE(::testing::Message() << "K=" << shape.k << " widths=" << shape.widths.size());
            constexpr int reference_rows = 1024, padding = 600;
            int total = 0;
            for (int w : shape.widths) total += w;
            std::mt19937 rng(9074);
            std::uniform_real_distribution<float> a(-0.75f, 0.75f), b(-0.125f, 0.125f);
            std::vector<uint16_t> host_a(size_t(reference_rows + padding) * shape.k), host_b(size_t(total) * shape.k);
            for (auto &v : host_a) v = canonicalPreparedHalfBits(a(rng));
            for (auto &v : host_b) v = canonicalPreparedHalfBits(b(rng));
            uint16_t *d_a = nullptr, *d_b = nullptr; float *d_c = nullptr, *d_p = nullptr;
            const size_t c_bytes = size_t(reference_rows + padding) * total * sizeof(float);
            const size_t p_bytes = cuda::fixedKHalfPartialBytes(reference_rows + padding, total);
            ASSERT_EQ(cudaMalloc(&d_a, host_a.size() * 2), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&d_b, host_b.size() * 2), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&d_c, c_bytes), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&d_p, p_bytes), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(d_a, host_a.data(), host_a.size() * 2, cudaMemcpyHostToDevice), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(d_b, host_b.data(), host_b.size() * 2, cudaMemcpyHostToDevice), cudaSuccess);
            auto group = [&](int rows, int offset) {
                const uint16_t *weight = d_b; float *out = d_c; size_t written = 0;
                auto g = FixedHalfProjectionGroup::single(d_a + size_t(offset) * shape.k, weight, out, shape.widths[0]);
                written += size_t(rows) * shape.widths[0]; weight += size_t(shape.widths[0]) * shape.k;
                for (size_t p = 1; p < shape.widths.size(); ++p)
                {
                    g.append(weight, d_c + written, shape.widths[p]);
                    written += size_t(rows) * shape.widths[p]; weight += size_t(shape.widths[p]) * shape.k;
                }
                return g;
            };
            auto fetch = [&](int rows) {
                std::vector<float> v(size_t(rows) * total);
                EXPECT_EQ(cudaMemcpyAsync(v.data(), d_c, v.size() * sizeof(float), cudaMemcpyDeviceToHost, stream), cudaSuccess);
                EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
                return v;
            };
            auto row = [&](const std::vector<float> &v, int rows, size_t p, int r) {
                size_t offset = 0;
                for (size_t q = 0; q < p; ++q) offset += size_t(rows) * shape.widths[q];
                return v.data() + offset + size_t(r) * shape.widths[p];
            };
            ASSERT_TRUE(cuda::launchFixedKHalfProjection(group(reference_rows, 0), reference_rows, shape.k, 0, stream,
                cuda::FixedKHalfSchedule::Tiled64x64, d_p, p_bytes));
            const auto reference = fetch(reference_rows);
            for (auto schedule : cuda::fixedKHalfSchedules())
                for (int rows : {1, 2, 7, 16, 17, 31, 33, 63, 64, 65, 127, 129, 255, 256, 384, 448, 512, 513, 1024})
                    for (int offset : {0, 1, 7, 15, 16, 127, 257})
                    {
                        if (offset + rows > reference_rows) continue;
                        ASSERT_EQ(cudaMemsetAsync(d_c, 0xff, c_bytes, stream), cudaSuccess);
                        ASSERT_TRUE(cuda::launchFixedKHalfProjection(group(rows, offset), rows, shape.k, 0, stream,
                            schedule, d_p, p_bytes));
                        const auto actual = fetch(rows);
                        for (size_t p = 0; p < shape.widths.size(); ++p)
                            for (int r = 0; r < rows; ++r)
                                ASSERT_EQ(std::memcmp(row(actual, rows, p, r), row(reference, reference_rows, p, offset + r),
                                    size_t(shape.widths[p]) * sizeof(float)), 0)
                                    << cuda::fixedKHalfScheduleName(schedule) << " M=" << rows << " offset=" << offset
                                    << " projection=" << p << " row=" << r;
                    }
            EXPECT_EQ(cudaFree(d_a), cudaSuccess);
            EXPECT_EQ(cudaFree(d_b), cudaSuccess);
            EXPECT_EQ(cudaFree(d_c), cudaSuccess);
            EXPECT_EQ(cudaFree(d_p), cudaSuccess);
        }
        EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    });
}

/** @test Partitioned launches reject a missing or undersized partial bank before any enqueue. */
TEST(FixedKPrefillProjection, CUDAPartitionedRejectsUnadmittedPartials)
{
    DeviceManager::instance().initialize(-1);
    auto &worker = GPUDeviceContextPool::instance().getContext(DeviceId::cuda(0));
    worker.submitAndWait([&]
    {
        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
        uint16_t *d_a = nullptr; float *d_c = nullptr;
        ASSERT_EQ(cudaMalloc(&d_a, 64 * 2048 * 2), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_c, 64 * 32 * sizeof(float)), cudaSuccess);
        const auto group = FixedHalfProjectionGroup::single(d_a, d_a, d_c, 32);
        ASSERT_EQ(cuda::selectFixedKHalfSchedule(64, 32, 2048), cuda::FixedKHalfSchedule::Partitioned64x64);
        EXPECT_THROW(cuda::launchFixedKHalfProjection(group, 64, 2048, 0, stream, nullptr, 0), std::invalid_argument);
        EXPECT_THROW(cuda::launchFixedKHalfProjection(group, 64, 2048, 0, stream, d_c,
            cuda::fixedKHalfPartialBytes(64, 32) - 1), std::invalid_argument);
        EXPECT_THROW(cuda::launchFixedKHalfProjection(group, 64, 2048, 0, nullptr, d_c,
            cuda::fixedKHalfPartialBytes(64, 32)), std::invalid_argument);
        EXPECT_THROW(cuda::launchFixedKHalfProjection(group, 0, 2048, 0, stream, d_c,
            cuda::fixedKHalfPartialBytes(64, 32)), std::invalid_argument);
        // The policy bound covers every partitioned group at a row envelope.
        for (int m : {1, 64, 256, 512, 513, 1024, 4096})
            for (int columns : {1, 8, 64, 96, 128, 129, 4096})
                if (cuda::fixedKHalfScheduleNeedsPartials(cuda::selectFixedKHalfSchedule(m, columns, 2048)))
                    EXPECT_LE(cuda::fixedKHalfPartialBytes(m, columns), cuda::fixedKHalfPartialBound(m))
                        << "m=" << m << " columns=" << columns;
        EXPECT_EQ(cudaFree(d_a), cudaSuccess);
        EXPECT_EQ(cudaFree(d_c), cudaSuccess);
        EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    });
}
#endif

/** @brief Initialize MPI and the native runtime before graph tests. */
int main(int argc, char **argv)
{
    if (auto terminal = llaminar2::MPISessionOwner::enterProcess(argc, argv))
        return *terminal;
    llaminar2::GPUDeviceProcessLifetime gpu_process_lifetime;

    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) != MPI_SUCCESS)
        return 1;
    if (provided < MPI_THREAD_MULTIPLE)
    {
        llaminar2::GPUDeviceProcessLifetime::retireActive();
        MPI_Finalize();
        return 1;
    }
    setenv("LLAMINAR_PERF_STATS_JSON", "1", 1);
    mutableDebugEnv().reload();
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    llaminar2::GPUDeviceProcessLifetime::retireActive();
    MPI_Finalize();
    return result;
}
