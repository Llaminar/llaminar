/**
 * @file Test__FixedKPrefillProjection.cpp
 * @brief Chunk- and schedule-invariance regressions for explicit-operand GPU prefill (#74).
 *
 * Ordinary explicit-operand GDN prefill projections (half-RNE alpha/beta by
 * default, full FP16 and strict FP32 by explicit policy) used vendor BLAS whose
 * reduction order followed the physical row count, so prefix-cache checkpoints
 * that split a cold prompt into different chunks changed its tokens. The
 * fixed-K contract makes every prompt row independent of chunking, bucket and
 * row offset.
 *
 * Two layers are proven on the real backend:
 *  - kernel schedules: every compiled fixed-K schedule produces identical row
 *    bytes for every M and row offset against one reference launch; strict-FP32
 *    rows additionally equal a host IEEE-FMA evaluation of the contract;
 *  - stage replay: the production GDNProjectionStage, captured once per bucket,
 *    replays a 778-row prompt under the issue's chunkings (512+256+10 versus
 *    512+266) and others, with identical bytes for every explicit projection.
 *
 * CUDA and ROCm compile this file separately so each process owns one runtime.
 * gfx1100 WMMA schedules are exercised only on native gfx1100 devices; the
 * schedule inventory follows the device's actual half arithmetic.
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
#include "kernels/rocm/gemm/ROCmFixedKProjectionLaunch.h"
#include <hip/hip_runtime.h>
#endif
#include "backends/ComputeBackend.h"
#include "utils/DebugEnv.h"
#include "utils/MPIContext.h"

#include <bit>
#include <cmath>
#include <functional>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
#ifdef LLAMINAR_FIXED_K_CUDA
    constexpr const char *kRouteCounter = "gemm.cuda_fixed_k_projection_calls";
    /** @return The test's single physical participant. */
    DeviceId testDevice() { return DeviceId::cuda(0); }
    using NativeStream = cudaStream_t;
    /** @brief Thin native wrappers so the kernel-level sweep is backend neutral. */
    struct Native
    {
        static bool alloc(void **p, size_t bytes) { return cudaMalloc(p, bytes) == cudaSuccess; }
        static bool free(void *p) { return cudaFree(p) == cudaSuccess; }
        static bool upload(void *d, const void *h, size_t bytes) { return cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice) == cudaSuccess; }
        static bool download(void *h, const void *d, size_t bytes, NativeStream s)
        { return cudaMemcpyAsync(h, d, bytes, cudaMemcpyDeviceToHost, s) == cudaSuccess && cudaStreamSynchronize(s) == cudaSuccess; }
        static bool poison(void *d, size_t bytes, NativeStream s) { return cudaMemsetAsync(d, 0xff, bytes, s) == cudaSuccess; }
        static bool create(NativeStream *s) { return cudaStreamCreateWithFlags(s, cudaStreamNonBlocking) == cudaSuccess; }
        static bool destroy(NativeStream s) { return cudaStreamDestroy(s) == cudaSuccess; }
    };
#else
    constexpr const char *kRouteCounter = "gemm.rocm_fixed_k_projection_calls";
    DeviceId testDevice() { return DeviceId::rocm(0); }
    using NativeStream = hipStream_t;
    struct Native
    {
        static bool alloc(void **p, size_t bytes) { return hipMalloc(p, bytes) == hipSuccess; }
        static bool free(void *p) { return hipFree(p) == hipSuccess; }
        static bool upload(void *d, const void *h, size_t bytes) { return hipMemcpy(d, h, bytes, hipMemcpyHostToDevice) == hipSuccess; }
        static bool download(void *h, const void *d, size_t bytes, NativeStream s)
        { return hipMemcpyAsync(h, d, bytes, hipMemcpyDeviceToHost, s) == hipSuccess && hipStreamSynchronize(s) == hipSuccess; }
        static bool poison(void *d, size_t bytes, NativeStream s) { return hipMemsetAsync(d, 0xff, bytes, s) == hipSuccess; }
        static bool create(NativeStream *s) { return hipStreamCreateWithFlags(s, hipStreamNonBlocking) == hipSuccess; }
        static bool destroy(NativeStream s) { return hipStreamDestroy(s) == hipSuccess; }
    };
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
     * @brief Run one stage replay on the real worker of the test device.
     * @param wide_format Native QKV/Z format in alpha/beta-only mode.
     * @param k Reduction width. @param columns QKV, Z, alpha, beta widths.
     * @param full_explicit All four projections explicit (full FP16/FP32 modes).
     * @param precision FP16RNE or FP32 operands.
     */
    void replayOnDevice(TensorType wide_format, int k, std::array<int, 4> columns, bool full_explicit,
        FloatingActivationOperandPrecision precision)
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
                k, columns, full_explicit, precision, issueSchedules(), kRouteCounter));
        });
    }

    /** @brief Host evaluation of the scalar contract: eight ascending FMA chains, left fold. */
    float hostScalarContract(const float *a, const float *b, int k)
    {
        float result = 0.0f;
        for (int part = 0; part < fixed_k_projection::kPartitions; ++part)
        {
            float chain = 0.0f;
            for (int e = fixed_k_projection::partitionBegin(part, k); e < fixed_k_projection::partitionBegin(part + 1, k); ++e)
                chain = std::fma(a[e], b[e], chain);
            result = part ? result + chain : chain;
        }
        return result;
    }

    /** @brief One kernel-level geometry: reduction width and ragged group widths. */
    struct Shape { int k; std::vector<int> widths; };

    /**
     * @brief Every schedule gives identical rows for every M and row offset.
     * @tparam Operand uint16_t half words or float strict FP32.
     * @param shapes Geometries to sweep. @param schedule_count Compiled schedule count.
     * @param launch Submit schedule index s for (group, m, k, stream, partials, bytes).
     * @param host_contract For FP32: also compare each reference row with the host contract.
     */
    template<class Operand>
    void sweepSchedules(const std::vector<Shape> &shapes, size_t schedule_count,
        const std::function<bool(size_t, const FixedRaggedProjectionGroup<Operand> &, int, int, NativeStream, float *, size_t)> &launch,
        bool host_contract)
    {
        NativeStream stream{};
        ASSERT_TRUE(Native::create(&stream));
        for (const auto &shape : shapes)
        {
            SCOPED_TRACE(::testing::Message() << "K=" << shape.k << " projections=" << shape.widths.size());
            constexpr int reference_rows = 1024, padding = 600;
            int total = 0;
            for (int w : shape.widths) total += w;
            std::mt19937 rng(9074);
            std::uniform_real_distribution<float> a(-0.75f, 0.75f), b(-0.125f, 0.125f);
            std::vector<Operand> host_a(size_t(reference_rows + padding) * shape.k), host_b(size_t(total) * shape.k);
            std::vector<float> exact_a(host_a.size()), exact_b(host_b.size());
            for (size_t i = 0; i < host_a.size(); ++i)
            {
                if constexpr (std::is_same_v<Operand, float>) host_a[i] = exact_a[i] = a(rng);
                else { host_a[i] = canonicalPreparedHalfBits(a(rng)); }
            }
            for (size_t i = 0; i < host_b.size(); ++i)
            {
                if constexpr (std::is_same_v<Operand, float>) host_b[i] = exact_b[i] = b(rng);
                else { host_b[i] = canonicalPreparedHalfBits(b(rng)); }
            }
            void *d_a = nullptr, *d_b = nullptr, *d_c = nullptr, *d_p = nullptr;
            const size_t c_bytes = size_t(reference_rows + padding) * total * sizeof(float);
            const size_t p_bytes = fixed_k_projection::partialBytes(reference_rows + padding, total);
            ASSERT_TRUE(Native::alloc(&d_a, host_a.size() * sizeof(Operand)));
            ASSERT_TRUE(Native::alloc(&d_b, host_b.size() * sizeof(Operand)));
            ASSERT_TRUE(Native::alloc(&d_c, c_bytes));
            ASSERT_TRUE(Native::alloc(&d_p, p_bytes));
            ASSERT_TRUE(Native::upload(d_a, host_a.data(), host_a.size() * sizeof(Operand)));
            ASSERT_TRUE(Native::upload(d_b, host_b.data(), host_b.size() * sizeof(Operand)));
            const auto *A = static_cast<const Operand *>(d_a);
            const auto *B = static_cast<const Operand *>(d_b);
            auto *C = static_cast<float *>(d_c);
            auto group = [&](int rows, int offset) {
                const Operand *weight = B; size_t written = 0;
                auto g = FixedRaggedProjectionGroup<Operand>::single(A + size_t(offset) * shape.k, weight, C, shape.widths[0]);
                written += size_t(rows) * shape.widths[0]; weight += size_t(shape.widths[0]) * shape.k;
                for (size_t p = 1; p < shape.widths.size(); ++p)
                {
                    g.append(weight, C + written, shape.widths[p]);
                    written += size_t(rows) * shape.widths[p]; weight += size_t(shape.widths[p]) * shape.k;
                }
                return g;
            };
            auto fetch = [&](int rows) {
                std::vector<float> v(size_t(rows) * total);
                EXPECT_TRUE(Native::download(v.data(), d_c, v.size() * sizeof(float), stream));
                return v;
            };
            auto row = [&](const std::vector<float> &v, int rows, size_t p, int r) {
                size_t offset = 0;
                for (size_t q = 0; q < p; ++q) offset += size_t(rows) * shape.widths[q];
                return v.data() + offset + size_t(r) * shape.widths[p];
            };
            ASSERT_TRUE(launch(schedule_count - 1, group(reference_rows, 0), reference_rows, shape.k, stream,
                static_cast<float *>(d_p), p_bytes));
            const auto reference = fetch(reference_rows);
            if (host_contract)
            {
                size_t weight_offset = 0;
                for (size_t p = 0; p < shape.widths.size(); ++p)
                {
                    for (int r : {0, 1, 511, reference_rows - 1})
                        for (int n = 0; n < shape.widths[p]; ++n)
                        {
                            const float expected = hostScalarContract(exact_a.data() + size_t(r) * shape.k,
                                exact_b.data() + weight_offset + size_t(n) * shape.k, shape.k);
                            ASSERT_EQ(std::bit_cast<uint32_t>(row(reference, reference_rows, p, r)[n]),
                                      std::bit_cast<uint32_t>(expected)) << "host contract projection=" << p << " row=" << r;
                        }
                    weight_offset += size_t(shape.widths[p]) * shape.k;
                }
            }
            for (size_t schedule = 0; schedule < schedule_count; ++schedule)
                for (int rows : {1, 2, 7, 16, 17, 31, 33, 63, 64, 65, 127, 129, 255, 256, 384, 448, 512, 513, 1024})
                    for (int offset : {0, 1, 7, 15, 16, 127, 257})
                    {
                        if (offset + rows > reference_rows) continue;
                        ASSERT_TRUE(Native::poison(d_c, c_bytes, stream));
                        ASSERT_TRUE(launch(schedule, group(rows, offset), rows, shape.k, stream,
                            static_cast<float *>(d_p), p_bytes));
                        const auto actual = fetch(rows);
                        for (size_t p = 0; p < shape.widths.size(); ++p)
                            for (int r = 0; r < rows; ++r)
                                ASSERT_EQ(std::memcmp(row(actual, rows, p, r), row(reference, reference_rows, p, offset + r),
                                    size_t(shape.widths[p]) * sizeof(float)), 0)
                                    << "schedule=" << schedule << " M=" << rows << " offset=" << offset
                                    << " projection=" << p << " row=" << r;
                    }
            EXPECT_TRUE(Native::free(d_a));
            EXPECT_TRUE(Native::free(d_b));
            EXPECT_TRUE(Native::free(d_c));
            EXPECT_TRUE(Native::free(d_p));
        }
        EXPECT_TRUE(Native::destroy(stream));
    }

    /**
     * @brief Narrow, QFN, ragged, K-tail, empty-partition and wide geometries.
     * @details 48+48 and 120+8 exceed one bounded partitioned slab at M=1024
     * (slabs of 640 and 512 rows), so slab boundaries are compared too.
     */
    const std::vector<Shape> &sweepShapes()
    {
        static const std::vector<Shape> shapes{{2048, {32, 32}}, {2560, {48, 48}}, {2056, {13, 21}},
            {2048, {520, 264, 8, 8}}, {192, {100, 36}}, {65, {16, 300}}, {2044, {32, 32}}, {1024, {120, 8}}};
        return shapes;
    }

    /** @brief Run @p body on the test device's real worker. */
    void onWorker(const std::function<void()> &body)
    {
        DeviceManager::instance().initialize(-1);
        GPUDeviceContextPool::instance().getContext(testDevice()).submitAndWait(body);
    }
}

/** @test Qwen3.6-35B-A3B default policy: IQ3_S QKV/Z with half alpha/beta, full and TP2 widths. */
TEST(FixedKPrefillProjection, HalfAlphaBetaChunkReplayQwen36)
{
    const auto half = FloatingActivationOperandPrecision::FP16RNE;
    ASSERT_NO_FATAL_FAILURE(replayOnDevice(TensorType::IQ3_S, 2048, {512, 256, 32, 32}, false, half));
    ASSERT_NO_FATAL_FAILURE(replayOnDevice(TensorType::IQ3_S, 2048, {256, 128, 16, 16}, false, half));
}

/** @test QFN geometry: K=2560 half alpha/beta at full and tensor-parallel shard widths. */
TEST(FixedKPrefillProjection, HalfAlphaBetaChunkReplayQFN)
{
    const auto half = FloatingActivationOperandPrecision::FP16RNE;
    ASSERT_NO_FATAL_FAILURE(replayOnDevice(TensorType::Q5_K, 2560, {512, 256, 48, 48}, false, half));
    ASSERT_NO_FATAL_FAILURE(replayOnDevice(TensorType::Q5_K, 2560, {256, 128, 12, 12}, false, half));
}

/** @test Explicit full-FP16 GDN: every projection half-RNE, exercising the wide schedules. */
TEST(FixedKPrefillProjection, FullHalfGDNChunkReplay)
{
    ASSERT_NO_FATAL_FAILURE(replayOnDevice(TensorType::Q5_K, 2048, {2048, 1024, 32, 32}, true,
        FloatingActivationOperandPrecision::FP16RNE));
}

/** @test Explicit strict-FP32 GDN: narrow alpha/beta and every projection, all chunkings identical. */
TEST(FixedKPrefillProjection, StrictFP32GDNChunkReplay)
{
    const auto fp32 = FloatingActivationOperandPrecision::FP32;
    ASSERT_NO_FATAL_FAILURE(replayOnDevice(TensorType::IQ3_S, 2048, {512, 256, 32, 32}, false, fp32));
    ASSERT_NO_FATAL_FAILURE(replayOnDevice(TensorType::Q5_K, 2048, {2048, 1024, 32, 32}, true, fp32));
}

#ifdef LLAMINAR_FIXED_K_CUDA
/** @test Every compiled CUDA half schedule gives identical rows for every M and offset. */
TEST(FixedKPrefillProjection, HalfSchedulesByteIdentical)
{
    onWorker([&]
    {
        const auto schedules = cuda::fixedKHalfSchedules();
        sweepSchedules<uint16_t>(sweepShapes(), schedules.size(),
            [&](size_t s, const FixedHalfProjectionGroup &g, int m, int k, NativeStream stream, float *p, size_t bytes)
            { return cuda::launchFixedKHalfProjection(g, m, k, 0, stream, schedules[s], p, bytes); }, false);
    });
}

/** @test Every CUDA strict-FP32 schedule gives identical rows equal to the host FMA contract. */
TEST(FixedKPrefillProjection, FP32SchedulesMatchHostContract)
{
    onWorker([&]
    {
        const auto schedules = cuda::fixedKFP32Schedules();
        sweepSchedules<float>(sweepShapes(), schedules.size(),
            [&](size_t s, const FixedStrictFP32ProjectionGroup &g, int m, int k, NativeStream stream, float *p, size_t bytes)
            { return cuda::launchFixedKFP32Projection(g, m, k, 0, stream, schedules[s], p, bytes); }, true);
    });
}

/** @test Partitioned launches reject a missing or undersized partial bank before any enqueue. */
TEST(FixedKPrefillProjection, PartitionedRejectsUnadmittedPartials)
{
    onWorker([&]
    {
        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
        uint16_t *d_a = nullptr; float *d_c = nullptr;
        ASSERT_EQ(cudaMalloc(&d_a, 64 * 2048 * 2), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_c, 64 * 32 * sizeof(float)), cudaSuccess);
        const auto group = FixedHalfProjectionGroup::single(d_a, d_a, d_c, 32);
        const auto partitioned = cuda::FixedKHalfSchedule::Partitioned64x64;
        ASSERT_EQ(cuda::selectFixedKHalfSchedule(64, 32, 2048), partitioned);
        const size_t exact = cuda::fixedKHalfPartialBytes(64, 32);
        EXPECT_THROW(cuda::launchFixedKHalfProjection(group, 64, 2048, 0, stream, partitioned, nullptr, 0), std::invalid_argument);
        EXPECT_THROW(cuda::launchFixedKHalfProjection(group, 64, 2048, 0, stream, partitioned, d_c, exact - 1), std::invalid_argument);
        EXPECT_THROW(cuda::launchFixedKHalfProjection(group, 64, 2048, 0, nullptr, partitioned, d_c, exact), std::invalid_argument);
        EXPECT_THROW(cuda::launchFixedKHalfProjection(group, 0, 2048, 0, stream, partitioned, d_c, exact), std::invalid_argument);
        // The shared bound covers every partitioned group at a row envelope.
        for (int m : {1, 64, 256, 512, 513, 1024, 4096})
            for (int columns : {1, 8, 64, 96, 128, 129, 4096})
                if (fixed_k_projection::usesPartitionedSchedule(m, columns))
                    EXPECT_LE(fixed_k_projection::partialBytes(m, columns), fixed_k_projection::partialBound(m))
                        << "m=" << m << " columns=" << columns;
        EXPECT_EQ(cudaFree(d_a), cudaSuccess);
        EXPECT_EQ(cudaFree(d_c), cudaSuccess);
        EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    });
}
#else
/** @test Every ROCm half schedule valid on this device gives identical rows for every M and offset. */
TEST(FixedKPrefillProjection, HalfSchedulesByteIdentical)
{
    onWorker([&]
    {
        const int device = testDevice().ordinal;
        const auto schedules = rocm::fixedKSchedules(rocm::fixedKHalfOperandForDevice(device));
        sweepSchedules<uint16_t>(sweepShapes(), schedules.size(),
            [&](size_t s, const FixedHalfProjectionGroup &g, int m, int k, NativeStream stream, float *p, size_t bytes)
            { return rocm::launchFixedKHalfProjection(g, m, k, device, stream, schedules[s], p, bytes); }, false);
    });
}

/** @test Every ROCm strict-FP32 schedule gives identical rows equal to the host FMA contract. */
TEST(FixedKPrefillProjection, FP32SchedulesMatchHostContract)
{
    onWorker([&]
    {
        const int device = testDevice().ordinal;
        const auto schedules = rocm::fixedKSchedules(rocm::FixedKOperand::StrictFP32);
        sweepSchedules<float>(sweepShapes(), schedules.size(),
            [&](size_t s, const FixedStrictFP32ProjectionGroup &g, int m, int k, NativeStream stream, float *p, size_t bytes)
            { return rocm::launchFixedKFP32Projection(g, m, k, device, stream, schedules[s], p, bytes); }, true);
    });
}

/** @test Half schedules reject the other half arithmetic and partitioned launches an unadmitted bank. */
TEST(FixedKPrefillProjection, RejectsMismatchedArithmeticAndUnadmittedPartials)
{
    onWorker([&]
    {
        const int device = testDevice().ordinal;
        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
        uint16_t *d_a = nullptr; float *d_c = nullptr;
        ASSERT_EQ(hipMalloc(&d_a, 64 * 2048 * 2), hipSuccess);
        ASSERT_EQ(hipMalloc(&d_c, 64 * 32 * sizeof(float)), hipSuccess);
        const auto group = FixedHalfProjectionGroup::single(d_a, d_a, d_c, 32);
        const auto operand = rocm::fixedKHalfOperandForDevice(device);
        const auto other = operand == rocm::FixedKOperand::HalfWmma
            ? rocm::FixedKSchedule::ScalarTiled64x64 : rocm::FixedKSchedule::WmmaTiled64x64;
        EXPECT_THROW(rocm::launchFixedKHalfProjection(group, 64, 2048, device, stream, other, nullptr, 0), std::invalid_argument);
        const auto partitioned = rocm::selectFixedKSchedule(operand, 64, 32);
        ASSERT_TRUE(rocm::fixedKScheduleNeedsPartials(partitioned));
        const size_t exact = fixed_k_projection::partialBytes(64, 32);
        EXPECT_THROW(rocm::launchFixedKHalfProjection(group, 64, 2048, device, stream, partitioned, nullptr, 0), std::invalid_argument);
        EXPECT_THROW(rocm::launchFixedKHalfProjection(group, 64, 2048, device, stream, partitioned, d_c, exact - 1), std::invalid_argument);
        EXPECT_THROW(rocm::launchFixedKHalfProjection(group, 64, 2048, device, nullptr, partitioned, d_c, exact), std::invalid_argument);
        // Scalar partitions every narrow group in bounded slabs that fit the admitted bank.
        for (int m : {1, 64, 511, 512, 513, 640, 1024, 4096, 16384})
            for (int columns : {1, 8, 64, 96, 120, 127, 128})
            {
                for (auto arithmetic : {rocm::FixedKOperand::HalfScalar, rocm::FixedKOperand::StrictFP32})
                    EXPECT_EQ(rocm::selectFixedKSchedule(arithmetic, m, columns), rocm::FixedKSchedule::ScalarPartitioned64x64);
                const int rows = rocm::fixedKPartialRows(rocm::FixedKSchedule::ScalarPartitioned64x64, m, columns);
                EXPECT_TRUE(rows >= 1 && rows <= m && (rows == m || rows % 64 == 0)) << "m=" << m << " columns=" << columns;
                EXPECT_LE(fixed_k_projection::partialBytes(rows, columns), fixed_k_projection::partialBound(m))
                    << "m=" << m << " columns=" << columns;
            }
        EXPECT_EQ(rocm::selectFixedKSchedule(rocm::FixedKOperand::StrictFP32, 1024, 129), rocm::FixedKSchedule::ScalarTiled64x64);
        EXPECT_EQ(hipFree(d_a), hipSuccess);
        EXPECT_EQ(hipFree(d_c), hipSuccess);
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
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
