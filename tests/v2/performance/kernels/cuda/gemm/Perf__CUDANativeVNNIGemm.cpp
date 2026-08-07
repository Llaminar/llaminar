/**
 * @file Perf__CUDANativeVNNIGemm.cpp
 * @brief Correctness and performance harness for tensor-core native-vnni prefill GEMM.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>
#include <set>
#include <span>
#include <thread>
#include <tuple>

#ifdef HAVE_CUDA

#include <cuda_runtime.h>

#include "../../native_vnni_dispatch/GPUTrainerVerification.h"
#include "CUDANativeVNNIGemmPerfCommon.h"
#include "../../../../utils/ScopedGPUStream.h"
#include "kernels/cuda/gemm/CUDANativeVNNIPrefillDiagnostics.h"
#include "fort.hpp"

using namespace llaminar2::test::native_vnni_gemm_perf;
using llaminar2::DeviceId;
using llaminar2::DeviceWorkspaceManager;
using llaminar2::FP32Tensor;
using llaminar2::ITensorGemm;
using llaminar2::IWorkspaceConsumer;
using llaminar2::test::TestTensorFactory;
using llaminar2::TensorBase;
using llaminar2::WorkspaceDescriptor;
using llaminar2::WorkspaceRequirements;

extern "C"
{
    void cudaNativeVNNIPrefill_setBK256Mode(int mode);
    int cudaNativeVNNIPrefill_getBK256Mode();
    void cudaNativeVNNIPrefill_setCanonicalKPartitionMode(bool enabled);
    bool cudaNativeVNNIPrefill_getCanonicalKPartitionMode();
    void cudaNativeVNNIPrefill_setExactOverlayEnabled(bool enabled);
    bool cudaNativeVNNIPrefill_getExactOverlayEnabled();
    void cudaNativeVNNIPrefill_setForceTile(int tile_id);
    void cudaNativeVNNIPrefill_getForceTile(int *tile_id);
    void cudaNativeVNNIPrefill_getLastLaunchSelection(
        int *tile_id,
        int *k_partitions,
        int *used_bk256,
        int *used_canonical_kpart);
    int cudaNativeVNNIPrefill_getTileCount(int tile_id, int M, int N);
    bool cudaNativeVNNIGemvTuned_queryGeneratedDispatch(
        uint8_t codebook_id,
        int graph_captured,
        int m,
        int n,
        int k,
        int *shape_id,
        int *tile_n,
        int *cpt,
        int *exact_kb);
    double cudaNativeVNNIGemvTuned_measureGeneratedDispatchNs(
        uint8_t codebook_id,
        int graph_captured,
        int m,
        const int *ns,
        const int *ks,
        int query_count,
        int iterations,
        uint64_t *out_checksum);
}

namespace
{
    /**
     * @brief Temporarily expose the generic prefill policy to a tournament.
     *
     * Incremental collection often runs after older exact overlays have been
     * installed. The AUTO evidence row must still mean the generic policy,
     * otherwise an old winner can masquerade as AUTO and contaminate both
     * candidate identity and workspace publication. The guard restores the
     * process's production setting even when a GoogleTest assertion unwinds.
     */
    class ScopedDensePrefillOverlayBypass
    {
    public:
        ScopedDensePrefillOverlayBypass()
            : previous_(cudaNativeVNNIPrefill_getExactOverlayEnabled())
        {
            cudaNativeVNNIPrefill_setExactOverlayEnabled(false);
        }

        ~ScopedDensePrefillOverlayBypass()
        {
            cudaNativeVNNIPrefill_setExactOverlayEnabled(previous_);
        }

        ScopedDensePrefillOverlayBypass(
            const ScopedDensePrefillOverlayBypass &) = delete;
        ScopedDensePrefillOverlayBypass &operator=(
            const ScopedDensePrefillOverlayBypass &) = delete;

    private:
        bool previous_ = true;
    };

    const llaminar2::NativeVnniFormatInfo &requireNativeVnniInfo(
        const TensorBase *weights,
        const std::string &format_name)
    {
        const auto *unpackable = dynamic_cast<const llaminar2::IINT8Unpackable *>(weights);
        const llaminar2::NativeVnniFormatInfo *info = unpackable ? unpackable->vnniFormatInfo() : nullptr;
        if (!info)
            throw std::runtime_error("CUDA NativeVNNI sweep format " + format_name + " did not expose vnniFormatInfo()");
        return *info;
    }

    TEST(CUDANativeVNNIGemmPerfOffline, FormatListCodebookIdsMatchTensorMetadata)
    {
        const std::set<std::string> expected_formats = {
            "Q4_0", "IQ4_NL", "IQ4_XS", "Q4_1", "Q4_K", "Q5_0", "Q5_1",
            "Q5_K", "Q6_K", "Q3_K", "Q2_K", "IQ3_S", "IQ3_XXS", "IQ2_S",
            "IQ2_XS", "IQ2_XXS", "IQ1_S", "IQ1_M", "Q8_0", "Q8_1", "Q8_K"};
        std::set<std::string> observed_formats;
        for (const auto &format : kFormats)
        {
            auto weights = format.create(/*n=*/2, /*k=*/256);
            ASSERT_NE(weights, nullptr) << format.name;

            const auto &info = requireNativeVnniInfo(weights.get(), format.name);
            EXPECT_EQ(info.codebook_id, format.codebook_id) << format.name;
            EXPECT_TRUE(observed_formats.insert(format.name).second)
                << "duplicate CUDA NativeVNNI source format " << format.name;
        }
        EXPECT_EQ(observed_formats, expected_formats);
    }

    TEST(CUDANativeVNNIGemmPerfOffline, ShapeListMatchesCanonicalExactOverlays)
    {
        std::set<std::tuple<std::string, int, int>> expected;
        for (const auto &shape :
             llaminar2::test::native_vnni_dispatch::nativeVnniShapeManifest())
        {
            if (shape.role == "production" && shape.exact_overlay)
                expected.emplace(shape.name, shape.N, shape.K);
        }

        std::set<std::tuple<std::string, int, int>> observed;
        for (const auto &shape : kQwenShapes)
            EXPECT_TRUE(observed.emplace(shape.name, shape.n, shape.k).second)
                << "duplicate CUDA NativeVNNI exact geometry " << shape.name;
        EXPECT_EQ(observed, expected);
    }

    TEST(CUDANativeVNNIGemmPerfOffline, PrefillRowsExcludeDecodeAndVerifierDepths)
    {
        EXPECT_EQ(kPrefillMValues, llaminar2::defaultPrefillGraphBucketSizes());
        EXPECT_EQ(std::count(kPrefillMValues.begin(), kPrefillMValues.end(), 1), 0);
        for (int m = 2; m <= llaminar2::kDefaultNativeVNNIVerifierRowCapacity;
             ++m)
            EXPECT_EQ(std::count(kPrefillMValues.begin(), kPrefillMValues.end(), m), 0)
                << "M=" << m << " belongs to grouped verifier training";
    }

    /**
     * @test Keep CUDA generated decode dispatch within a few host nanoseconds.
     *
     * A decode or grouped-verifier projection can be small enough for host
     * policy overhead to become visible. This test calls the same cached
     * generated selector as production while deliberately avoiding every CUDA
     * runtime API. It measures one repeatedly hot graph-capture/eager-launch
     * Q8 key and a rotating set of production Qwen geometries. Graph replay
     * itself owns fixed kernel nodes and does not call this host selector. A
     * loop-only control performs the same indexing, checksum update, and
     * compiler fence; its median cost is subtracted from the selector median.
     *
     * Q8 source aliases are normalized to execution codebook 19 before this
     * selector. The test therefore verifies NativeVNNI dispatch itself rather
     * than reviving a format-specific Q8 route.
     */
    TEST(CUDANativeVNNIGemmPerfOffline, GeneratedDecodeDispatchCacheLatency)
    {
        struct Query
        {
            int n;
            int k;
        };

        constexpr uint8_t codebook_id = 19;
        constexpr int graph_captured = 1;
        constexpr int iterations = 100000;
        constexpr int samples = 7;
        constexpr std::array<Query, 1> hot = {{{512, 2048}}};
        constexpr std::array<Query, 8> working_set = {{
            {512, 2048},
            {2048, 512},
            {2048, 2048},
            {4096, 2048},
            {8192, 2048},
            {1024, 5120},
            {6144, 5120},
            {5120, 17408},
        }};

        for (const Query &query : working_set)
        {
            int shape_id = 0;
            int tile_n = 0;
            int cpt = 0;
            int exact_kb = 0;
            ASSERT_TRUE(cudaNativeVNNIGemvTuned_queryGeneratedDispatch(
                codebook_id,
                graph_captured,
                1,
                query.n,
                query.k,
                &shape_id,
                &tile_n,
                &cpt,
                &exact_kb))
                << "missing CUDA Q8 NativeVNNI policy for "
                << query.n << 'x' << query.k;
        }

        const auto measure = [&](const auto &queries)
        {
            std::array<double, samples> net_ns{};
            std::array<int, queries.size()> ns{};
            std::array<int, queries.size()> ks{};
            for (size_t index = 0; index < queries.size(); ++index)
            {
                ns[index] = queries[index].n;
                ks[index] = queries[index].k;
            }

            for (int sample = 0; sample < samples; ++sample)
            {
                uint64_t checksum = 0;
                net_ns[static_cast<size_t>(sample)] =
                    cudaNativeVNNIGemvTuned_measureGeneratedDispatchNs(
                        codebook_id,
                        graph_captured,
                        1,
                        ns.data(),
                        ks.data(),
                        static_cast<int>(queries.size()),
                        iterations,
                        &checksum);
                EXPECT_NE(checksum, 0u);
            }

            std::sort(net_ns.begin(), net_ns.end());
            return net_ns[net_ns.size() / 2];
        };

        const double hot_ns = measure(hot);
        const double working_set_ns = measure(working_set);

        EXPECT_LT(hot_ns, 10.0);
        EXPECT_LT(working_set_ns, 20.0);
        std::cout
            << "[CUDANativeVNNI][DISPATCH_CACHE_LATENCY] hot_ns="
            << hot_ns
            << " working_set_ns=" << working_set_ns
            << std::endl;
    }

    class CUDANativeVNNIGemmPerf : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            int device_count = 0;
            if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
                GTEST_SKIP() << "No CUDA devices available";

            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
            device_count_ = device_count;
            peak_tc_tops_ = estimatePeakInt8TensorCoreTops();
            if (peak_tc_tops_ > 0.0)
                std::fprintf(stderr, "[CUDANativeVNNIGemm] Peak dense INT8 tensor-core throughput: %.1f TOPS\n", peak_tc_tops_);
        }

        double peak_tc_tops_ = 0.0;
        int device_count_ = 0;
    };

    TEST_F(CUDANativeVNNIGemmPerf, CompilerResources_AllCodebooksAndCandidates)
    {
        static constexpr uint8_t codebooks[] = {
            0, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 19};
        const std::string csv_path =
            getEnvString("LLAMINAR_TILE_RESOURCE_CSV");
        FILE *csv = nullptr;
        if (!csv_path.empty())
        {
            csv = std::fopen(csv_path.c_str(), "w");
            ASSERT_NE(csv, nullptr) << "Failed to open " << csv_path;
            std::fprintf(
                csv,
                "codebook,tile_id,canonical_kpart,ordered_bk256,"
                "primary_registers_per_thread,"
                "primary_local_memory_bytes_per_thread,"
                "primary_static_shared_memory_bytes,"
                "primary_dynamic_shared_memory_bytes,"
                "primary_threads_per_block,"
                "primary_max_threads_per_block,"
                "primary_max_active_blocks_per_sm,"
                "auxiliary_registers_per_thread,"
                "auxiliary_local_memory_bytes_per_thread,"
                "auxiliary_static_shared_memory_bytes,"
                "auxiliary_dynamic_shared_memory_bytes,"
                "auxiliary_threads_per_block,"
                "auxiliary_max_threads_per_block,"
                "auxiliary_max_active_blocks_per_sm,spill_free\n");
        }

        size_t queried = 0;
        size_t spilling = 0;
        std::set<std::tuple<int, int, int, int>> observed_spilling;
        const auto inspect = [&](uint8_t codebook,
                                 int tile_id,
                                 int canonical_kpart,
                                 int ordered_bk256)
        {
            CUDADensePrefillKernelResources primary{};
            CUDADensePrefillKernelResources auxiliary{};
            ASSERT_TRUE(cudaNativeVNNIPrefill_queryCandidateResources(
                codebook,
                tile_id,
                canonical_kpart,
                ordered_bk256,
                /*cuda_device_id=*/0,
                &primary,
                &auxiliary));
            EXPECT_GT(primary.registers_per_thread, 0);
            EXPECT_GT(primary.threads_per_block, 0);
            EXPECT_GE(
                primary.max_threads_per_block,
                primary.threads_per_block);
            EXPECT_GT(primary.max_active_blocks_per_sm, 0);
            if (canonical_kpart)
            {
                EXPECT_GT(auxiliary.registers_per_thread, 0);
                EXPECT_EQ(auxiliary.local_memory_bytes_per_thread, 0u);
                EXPECT_GT(auxiliary.threads_per_block, 0);
                EXPECT_GE(
                    auxiliary.max_threads_per_block,
                    auxiliary.threads_per_block);
                EXPECT_GT(auxiliary.max_active_blocks_per_sm, 0);
            }
            else
            {
                EXPECT_EQ(auxiliary.registers_per_thread, 0);
                EXPECT_EQ(auxiliary.local_memory_bytes_per_thread, 0u);
                EXPECT_EQ(auxiliary.threads_per_block, 0);
                EXPECT_EQ(auxiliary.max_active_blocks_per_sm, 0);
            }

            const bool spill_free =
                primary.local_memory_bytes_per_thread == 0 &&
                auxiliary.local_memory_bytes_per_thread == 0;
            ++queried;
            spilling += spill_free ? 0 : 1;
            if (!spill_free)
            {
                observed_spilling.emplace(
                    static_cast<int>(codebook),
                    tile_id,
                    canonical_kpart,
                    ordered_bk256);
            }
            if (csv)
            {
                std::fprintf(
                    csv,
                    "%u,%d,%d,%d,%d,%zu,%zu,%zu,%d,%d,%d,"
                    "%d,%zu,%zu,%zu,%d,%d,%d,%d\n",
                    static_cast<unsigned>(codebook),
                    tile_id,
                    canonical_kpart,
                    ordered_bk256,
                    primary.registers_per_thread,
                    primary.local_memory_bytes_per_thread,
                    primary.static_shared_memory_bytes,
                    primary.dynamic_shared_memory_bytes,
                    primary.threads_per_block,
                    primary.max_threads_per_block,
                    primary.max_active_blocks_per_sm,
                    auxiliary.registers_per_thread,
                    auxiliary.local_memory_bytes_per_thread,
                    auxiliary.static_shared_memory_bytes,
                    auxiliary.dynamic_shared_memory_bytes,
                    auxiliary.threads_per_block,
                    auxiliary.max_threads_per_block,
                    auxiliary.max_active_blocks_per_sm,
                    spill_free ? 1 : 0);
            }
        };

        for (uint8_t codebook : codebooks)
        {
            for (int tile_id = 0; tile_id < 6; ++tile_id)
            {
                inspect(codebook, tile_id, 0, 0);
                inspect(codebook, tile_id, 1, 0);
            }
        }
        for (int tile_id : {-3, -2})
        {
            inspect(/*Q4_0=*/0, tile_id, 0, 0);
            inspect(/*Q4_0=*/0, tile_id, 0, 1);
        }
        if (csv)
        {
            std::fflush(csv);
            std::fclose(csv);
        }
        std::fprintf(
            stderr,
            "[TileSweep][resources] queried=%zu spilling=%zu spill_free=%zu\n",
            queried,
            spilling,
            queried - spilling);
        EXPECT_EQ(queried, 196u);
        std::set<std::tuple<int, int, int, int>> expected_spilling;
        for (int codebook : {8, 9, 13, 14})
        {
            for (int tile_id : {1, 4})
                expected_spilling.emplace(codebook, tile_id, 0, 0);
        }
        for (int codebook : {10, 17})
        {
            for (int tile_id : {1, 4, 5})
            {
                expected_spilling.emplace(codebook, tile_id, 0, 0);
                expected_spilling.emplace(codebook, tile_id, 1, 0);
            }
        }
        EXPECT_EQ(observed_spilling, expected_spilling)
            << "Compiler resource drift changed the launchable candidate set";
    }

    TEST_F(CUDANativeVNNIGemmPerf, Correctness_AllFormats_KeyShapes)
    {
        const RunConfig cfg = loadRunConfig();

        struct CorrectnessTask
        {
            const FormatSpec *format = nullptr;
            const Shape *shape = nullptr;
        };

        struct CorrectnessResult
        {
            std::string label;
            double cosine = 0.0;
            bool passed = false;
        };

        std::vector<CorrectnessTask> tasks;
        tasks.reserve(static_cast<size_t>(cfg.max_cases));
        for (const auto &format : kFormats)
        {
            if (!shouldRunName(cfg.format_filters, format.name))
                continue;
            for (const auto &shape : kQwenShapes)
            {
                if (!shouldRunName(cfg.shape_filters, shape.name))
                    continue;
                if ((shape.k % 32) != 0)
                    continue;
                if (static_cast<int>(tasks.size()) >= cfg.max_cases)
                    break;
                tasks.push_back(CorrectnessTask{&format, &shape});
            }
            if (static_cast<int>(tasks.size()) >= cfg.max_cases)
                break;
        }

        ASSERT_FALSE(tasks.empty()) << "No correctness cases selected. Check LLAMINAR_CUDA_NATIVE_GEMM_FORMATS / LLAMINAR_CUDA_NATIVE_GEMM_SHAPES.";

        int worker_count = std::min(2, std::max(1, device_count_));
        if (cfg.performance_workers > 0)
            worker_count = std::min(worker_count, cfg.performance_workers);

        std::fprintf(stderr,
                     "[CUDANativeVNNIGemm][Correctness] using %d worker thread(s) across %d CUDA device(s), %zu cases\n",
                     worker_count, worker_count, tasks.size());

        std::vector<CorrectnessResult> results(tasks.size());
        std::atomic<size_t> next_task{0};
        std::mutex log_mutex;

        auto worker = [&](int worker_index)
        {
            const int device_id = worker_index;
            while (true)
            {
                const size_t task_index = next_task.fetch_add(1, std::memory_order_relaxed);
                if (task_index >= tasks.size())
                    break;

                const auto &format = *tasks[task_index].format;
                const auto &shape = *tasks[task_index].shape;

                {
                    std::lock_guard<std::mutex> lock(log_mutex);
                    std::fprintf(stderr,
                                 "[CUDANativeVNNIGemm][Correctness][gpu=%d] format=%s shape=%s prefill_m=%d\n",
                                 device_id, format.name.c_str(), shape.name.c_str(), cfg.correctness_prefill_m);
                }

                // Create shared input so both cuBLAS and NativeVNNI see identical data
                auto h_input = TestTensorFactory::createFP32Random(
                    {static_cast<size_t>(cfg.correctness_prefill_m), static_cast<size_t>(shape.k)}, -0.25f, 0.25f, 7);
                const float *input_ptr = h_input->data();

                // cuBLAS FP32 reference (ground truth)
                auto ref_weights = format.create(static_cast<size_t>(shape.n), static_cast<size_t>(shape.k));
                const RunResult cublas_ref = runCuBLASReference(ref_weights.get(), input_ptr,
                    cfg.correctness_prefill_m, shape.n, shape.k, device_id);

                // NativeVNNI quantized GEMM
                auto native_vnni_weights = format.create(static_cast<size_t>(shape.n), static_cast<size_t>(shape.k));
                const RunResult native_vnni = runKernel(native_vnni_weights.get(),
                    cfg.correctness_prefill_m, shape.n, shape.k, RunPath::NativeVNNITensorCore,
                    0, 1, device_id, input_ptr);

                const double cosine = cosineSimilarity(cublas_ref.output, native_vnni.output);
                results[task_index] = {format.name + " " + shape.name, cosine, cosine >= kCosineGate};
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (int i = 0; i < worker_count; ++i)
            workers.emplace_back(worker, i);
        for (auto &t : workers)
            t.join();

        for (const auto &r : results)
        {
            EXPECT_TRUE(r.passed) << r.label << " cosine=" << r.cosine << " < " << kCosineGate;
        }
    }

    TEST_F(CUDANativeVNNIGemmPerf, Performance_AllFormats_AllShapes)
    {
        const RunConfig cfg = loadRunConfig();

        struct PerfTask
        {
            const FormatSpec *format = nullptr;
            const Shape *shape = nullptr;
        };

        struct PerfRow
        {
            std::string format_name;
            uint8_t codebook_id = 0;
            std::string shape_name;
            int m = 0;
            int n = 0;
            int k = 0;
            int warmup_runs = 0;
            int bench_runs = 0;
            size_t weight_bytes = 0;
            std::string native_family;
            double min_us = 0.0;
            double mean_us = 0.0;
            double tops = 0.0;
            double pct_tc_peak = 0.0;
        };

        std::vector<PerfTask> tasks;
        tasks.reserve(static_cast<size_t>(cfg.max_cases));
        for (const auto &format : kFormats)
        {
            if (!shouldRunName(cfg.format_filters, format.name))
                continue;

            for (const auto &shape : kQwenShapes)
            {
                if (!shouldRunName(cfg.shape_filters, shape.name))
                    continue;
                if ((shape.k % 32) != 0)
                    continue;
                if (static_cast<int>(tasks.size()) >= cfg.max_cases)
                    break;
                tasks.push_back(PerfTask{&format, &shape});
            }

            if (static_cast<int>(tasks.size()) >= cfg.max_cases)
                break;
        }

        ASSERT_FALSE(tasks.empty()) << "No performance cases selected. Check LLAMINAR_CUDA_NATIVE_GEMM_FORMATS / LLAMINAR_CUDA_NATIVE_GEMM_SHAPES.";

        int worker_count = std::min(2, std::max(1, device_count_));
        if (cfg.performance_workers > 0)
            worker_count = std::min(worker_count, cfg.performance_workers);

        std::fprintf(stderr,
                     "[CUDANativeVNNIGemm][Perf] using %d worker thread(s) across %d CUDA device(s)\n",
                     worker_count,
                     worker_count);

        std::vector<std::vector<PerfRow>> task_rows(tasks.size());
        std::atomic<size_t> next_task{0};
        std::mutex log_mutex;

        auto worker = [&](int worker_index)
        {
            const int device_id = worker_index;

            while (true)
            {
                const size_t task_index = next_task.fetch_add(1, std::memory_order_relaxed);
                if (task_index >= tasks.size())
                    break;

                const PerfTask &task = tasks[task_index];
                const auto &format = *task.format;
                const auto &shape = *task.shape;

                std::vector<PerfRow> rows;
                rows.reserve(cfg.performance_prefill_m.size());

                for (int m : cfg.performance_prefill_m)
                {
                    auto weights = format.create(static_cast<size_t>(shape.n), static_cast<size_t>(shape.k));
                    const uint8_t codebook_id = requireNativeVnniInfo(weights.get(), format.name).codebook_id;
                    const size_t weight_bytes = weights->size_bytes();
                    const RunResult result = runKernel(weights.get(), m, shape.n, shape.k, RunPath::NativeVNNITensorCore, cfg.warmup_runs, cfg.bench_runs, device_id);

                    const auto metrics = computeGemmThroughputMetrics(m, shape.n, shape.k, result.min_us, peak_tc_tops_);

                    {
                        std::lock_guard<std::mutex> lock(log_mutex);
                        std::fprintf(stderr,
                                     "[CUDANativeVNNIGemm][Perf][gpu=%d] format=%s codebook=%u shape=%s M=%d N=%d K=%d "
                                     "family=%s min_us=%.3f tops=%.3f pct_tc_peak=%.1f%%\n",
                                     device_id,
                                     format.name.c_str(),
                                     static_cast<unsigned>(codebook_id),
                                     shape.name.c_str(),
                                     m,
                                     shape.n,
                                     shape.k,
                                     result.native_family.c_str(),
                                     result.min_us,
                                     metrics.achieved_tops,
                                     metrics.pct_tc_peak);
                    }

                    rows.push_back(PerfRow{
                        format.name,
                        codebook_id,
                        shape.name,
                        m,
                        shape.n,
                        shape.k,
                        cfg.warmup_runs,
                        cfg.bench_runs,
                        weight_bytes,
                        result.native_family,
                        result.min_us,
                        result.mean_us,
                        metrics.achieved_tops,
                        metrics.pct_tc_peak,
                    });
                }

                task_rows[task_index] = std::move(rows);
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (int worker_index = 0; worker_index < worker_count; ++worker_index)
            workers.emplace_back(worker, worker_index);
        for (auto &thread : workers)
            thread.join();

        std::FILE *csv = nullptr;
        if (!cfg.csv_path.empty())
        {
            csv = std::fopen(cfg.csv_path.c_str(), "w");
            ASSERT_NE(csv, nullptr) << "Failed to open CSV: " << cfg.csv_path;
            std::fprintf(
                csv,
                "format,codebook,shape,m,n,k,warmup_runs,bench_runs,weight_bytes,family,min_us,mean_us,tops,pct_tc_peak\n");
        }

        int executed_cases = 0;
        for (const auto &rows : task_rows)
        {
            if (rows.empty())
                continue;

            for (const auto &row : rows)
            {
                if (csv)
                {
                    std::fprintf(
                        csv,
                        "%s,%u,%s,%d,%d,%d,%d,%d,%zu,%s,%.3f,%.3f,%.3f,%.1f\n",
                        row.format_name.c_str(),
                        static_cast<unsigned>(row.codebook_id),
                        row.shape_name.c_str(),
                        row.m,
                        row.n,
                        row.k,
                        row.warmup_runs,
                        row.bench_runs,
                        row.weight_bytes,
                        row.native_family.c_str(),
                        row.min_us,
                        row.mean_us,
                        row.tops,
                        row.pct_tc_peak);
                }
                ++executed_cases;
            }
        }

        if (csv)
        {
            std::fclose(csv);
            std::fprintf(stderr,
                         "[CUDANativeVNNIGemm][Perf] wrote sweep CSV to %s\n",
                         cfg.csv_path.c_str());
        }

        ASSERT_GT(executed_cases, 0) << "No performance cases selected. Check LLAMINAR_CUDA_NATIVE_GEMM_FORMATS / LLAMINAR_CUDA_NATIVE_GEMM_SHAPES.";
    }

    // =========================================================================
    // Tile + Strategy Sweep
    //
    // Sweeps all tile configs × strategies × shapes × M to compare:
    //   - AUTO:     production output-tile heuristic
    //   - STD:      BK64 with a forced output tile and direct ownership
    //   - KPART:    BK64 with the exact public-M1 ordered K partitions
    //   - BK256:    BK256 forced with the public-M=1 reduction schedule
    //
    // Environment variables:
    //   LLAMINAR_TILE_SWEEP_SHAPES       - Comma-separated shape names (default: all kQwenShapes)
    //   LLAMINAR_TILE_SWEEP_PREFILL_M    - Comma-separated M values
    //                                     (default: MTP rows + canonical prefill buckets)
    //   LLAMINAR_TILE_SWEEP_TILES        - Comma-separated tile IDs 0..5 for BK64 (default: all)
    //   LLAMINAR_TILE_SWEEP_STRATEGIES   - Comma-separated: auto,std,kpart,bk256
    //   LLAMINAR_TILE_SWEEP_WARMUP       - Warmup runs (default: 3)
    //   LLAMINAR_TILE_SWEEP_BENCH        - Benchmark runs (default: 10)
    //   LLAMINAR_TILE_SWEEP_CSV          - CSV output path
    //   LLAMINAR_TILE_SWEEP_TIMING_CSV   - Raw native-event timing sidecar
    //   LLAMINAR_TILE_SWEEP_FORMAT       - Quant format: Q4_0 (default), IQ4_NL, etc.
    // =========================================================================

    struct TileSpec
    {
        int tile_id;
        const char *name;
        int bm, bn;
        int warps_m, warps_n;
        int block_size;
    };

    static constexpr TileSpec kAllTiles[] = {
        {0, "T64x64_w2x2", 64, 64, 2, 2, 128},
        {1, "T64x128_w2x2", 64, 128, 2, 2, 128},
        {2, "T64x128_w4x2", 64, 128, 4, 2, 256},
        {3, "T64x128_w2x4", 64, 128, 2, 4, 256},
        {4, "T128x128_w4x2", 128, 128, 4, 2, 256},
        {5, "T128x128_w4x4", 128, 128, 4, 4, 512},
    };

    enum class Strategy
    {
        Auto,               // Production heuristic
        Standard,           // BK64 with one complete canonical K walk
        CanonicalKpart,     // BK64 with public-M1 partitions + ordered reducer
        BK256,              // BK256 with one complete canonical K walk
    };

    struct SweepConfig
    {
        int warmup_runs = 3;
        int bench_runs = 10;
        std::vector<int> prefill_m = llaminar2::defaultPrefillGraphBucketSizes();
        std::set<std::string> shape_filters;
        std::vector<int> tile_ids = {0, 1, 2, 3, 4, 5};
        /*
         * Every candidate in this inventory is byte-eligible by construction.
         * KPART obtains its partition geometry from public M=1; no independent
         * split count is accepted by this sweep.
         */
        std::vector<Strategy> strategies = {
            Strategy::Auto,
            Strategy::Standard,
            Strategy::CanonicalKpart,
            Strategy::BK256};
        std::string csv_path;
        std::string timing_csv_path;
        std::string format_name = "Q4_0";
    };

    static SweepConfig loadSweepConfig()
    {
        SweepConfig cfg;
        if (const auto v = getEnvInt("LLAMINAR_TILE_SWEEP_WARMUP"))
            cfg.warmup_runs = std::max(1, *v);
        if (const auto v = getEnvInt("LLAMINAR_TILE_SWEEP_BENCH"))
            cfg.bench_runs = std::max(1, *v);

        const auto m_vals = getEnvCsvInts("LLAMINAR_TILE_SWEEP_PREFILL_M");
        if (!m_vals.empty())
            cfg.prefill_m = m_vals;

        cfg.shape_filters = getEnvCsvSet("LLAMINAR_TILE_SWEEP_SHAPES");

        const auto tile_vals = getEnvCsvInts("LLAMINAR_TILE_SWEEP_TILES");
        if (!tile_vals.empty())
        {
            cfg.tile_ids.clear();
            for (int t : tile_vals)
                if (t >= 0 && t <= 5)
                    cfg.tile_ids.push_back(t);
        }

        const auto strat_set = getEnvCsvSet("LLAMINAR_TILE_SWEEP_STRATEGIES");
        if (!strat_set.empty())
        {
            cfg.strategies.clear();
            if (strat_set.count("auto"))
                cfg.strategies.push_back(Strategy::Auto);
            if (strat_set.count("std"))
                cfg.strategies.push_back(Strategy::Standard);
            if (strat_set.count("kpart"))
                cfg.strategies.push_back(Strategy::CanonicalKpart);
            if (strat_set.count("bk256"))
                cfg.strategies.push_back(Strategy::BK256);
        }

        const std::string csv = getEnvString("LLAMINAR_TILE_SWEEP_CSV");
        if (!csv.empty())
            cfg.csv_path = csv;

        const std::string timing_csv =
            getEnvString("LLAMINAR_TILE_SWEEP_TIMING_CSV");
        if (!timing_csv.empty())
            cfg.timing_csv_path = timing_csv;

        const std::string fmt = getEnvString("LLAMINAR_TILE_SWEEP_FORMAT");
        if (!fmt.empty())
            cfg.format_name = fmt;

        return cfg;
    }

    static const char *strategyName(Strategy s)
    {
        switch (s)
        {
        case Strategy::Auto:
            return "AUTO";
        case Strategy::Standard:
            return "STD";
        case Strategy::CanonicalKpart:
            return "KPART";
        case Strategy::BK256:
            return "BK256";
        }
        return "???";
    }

    struct SweepRow
    {
        std::string format_name;
        uint8_t codebook_id = 0;
        std::string shape_name;
        int m, n, k;
        std::string tile_name;
        int tile_id;
        int tiles; // tile count for this config
        std::string strategy;
        int requested_k_partitions; // 0=inherit public-M1; 1=full-K
        double min_us;
        double mean_us;
        double tops;
        double pct_peak;
        int gpu_id;
        size_t byte_mismatches;
        uint64_t first_byte_mismatch;
        bool correctness_pass;
        bool canonical_kpart_available;
        int observed_tile_id;
        int observed_k_partitions;
        int observed_bk256;
        int observed_canonical_kpart;
        int primary_registers_per_thread;
        size_t primary_local_memory_bytes_per_thread;
        size_t primary_static_shared_memory_bytes;
        size_t primary_dynamic_shared_memory_bytes;
        int primary_threads_per_block;
        int primary_max_threads_per_block;
        int primary_max_active_blocks_per_sm;
        int auxiliary_registers_per_thread;
        size_t auxiliary_local_memory_bytes_per_thread;
        size_t auxiliary_static_shared_memory_bytes;
        size_t auxiliary_dynamic_shared_memory_bytes;
        int auxiliary_threads_per_block;
        int auxiliary_max_threads_per_block;
        int auxiliary_max_active_blocks_per_sm;
    };

    struct SweepTask
    {
        const Shape *shape;
        int m;
        Strategy strat;
        int tile_id; // -1=auto, -2=BK256, 0..5=BK64 tile
        int requested_k_partitions;
        std::string tile_name;
        int tiles;
    };

    /**
     * @brief Apply one forceable prefill candidate to the production launcher.
     *
     * These controls are process-wide debug state. The turnkey collector masks
     * one GPU into each process, so a complete setup/launch transaction remains
     * isolated from every other device lane.
     */
    static void configureSweepTask(const SweepTask &task)
    {
        if (task.strat == Strategy::Auto)
        {
            cudaNativeVNNIPrefill_setBK256Mode(0);
            cudaNativeVNNIPrefill_setForceTile(-1);
            cudaNativeVNNIPrefill_setCanonicalKPartitionMode(false);
        }
        else if (task.tile_id == -2)
        {
            cudaNativeVNNIPrefill_setBK256Mode(1);
            cudaNativeVNNIPrefill_setForceTile(-1);
            cudaNativeVNNIPrefill_setCanonicalKPartitionMode(false);
        }
        else
        {
            cudaNativeVNNIPrefill_setBK256Mode(-1);
            cudaNativeVNNIPrefill_setForceTile(task.tile_id);
            cudaNativeVNNIPrefill_setCanonicalKPartitionMode(
                task.strat == Strategy::CanonicalKpart);
        }
    }

    /**
     * @brief Merge candidate-specific workspace plans into one persistent plan.
     *
     * A shape/M tournament changes tile and reduction policies while retaining
     * the same production kernel. Every named buffer is allocated once at the
     * largest requested size so candidate timing cannot include allocator work
     * or observe a missing buffer after a force-mode transition.
     */
    static WorkspaceRequirements tournamentWorkspaceRequirements(
        IWorkspaceConsumer *consumer,
        const std::vector<SweepTask> &tasks,
        const Shape *shape,
        int m)
    {
        WorkspaceRequirements merged;
        if (!consumer)
            return merged;

        const auto merge = [&merged](const WorkspaceRequirements &requirements)
        {
            for (const auto &buffer : requirements.buffers)
            {
                auto existing = std::find_if(
                    merged.buffers.begin(), merged.buffers.end(),
                    [&buffer](const WorkspaceDescriptor &candidate)
                    { return candidate.name == buffer.name; });
                if (existing == merged.buffers.end())
                {
                    merged.buffers.push_back(buffer);
                    continue;
                }
                existing->size_bytes =
                    std::max(existing->size_bytes, buffer.size_bytes);
                existing->alignment =
                    std::max(existing->alignment, buffer.alignment);
                existing->required = existing->required || buffer.required;
            }
        };

        for (const auto &task : tasks)
        {
            if (task.shape != shape || task.m != m)
                continue;
            configureSweepTask(task);
            merge(consumer->getWorkspaceRequirements(m, shape->n, shape->k));
        }

        /*
         * The ordinary-prefill tournament has a different arithmetic contract
         * from grouped verifier decode.  Verifier M=2..16 must reproduce the
         * public M=1 K-partition tree exactly; prefill candidates must reproduce
         * the production exact-M prefill result while changing only physical
         * tile geometry.  Include that AUTO route even when a caller requests
         * only forced candidates so the immutable workspace can always build
         * the correct exact-M oracle.
         */
        cudaNativeVNNIPrefill_setBK256Mode(0);
        cudaNativeVNNIPrefill_setForceTile(-1);
        cudaNativeVNNIPrefill_setCanonicalKPartitionMode(false);
        merge(consumer->getWorkspaceRequirements(m, shape->n, shape->k));
        return merged;
    }

    /** @brief Move-only ownership for one CUDA timing event. */
    class ScopedCUDATimingEvent
    {
    public:
        ScopedCUDATimingEvent()
        {
            const cudaError_t error = cudaEventCreate(&event_);
            if (error != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string("failed to create CUDA timing event: ") +
                    cudaGetErrorString(error));
            }
        }

        ~ScopedCUDATimingEvent()
        {
            if (event_)
                (void)cudaEventDestroy(event_);
        }

        ScopedCUDATimingEvent(const ScopedCUDATimingEvent &) = delete;
        ScopedCUDATimingEvent &operator=(const ScopedCUDATimingEvent &) = delete;

        ScopedCUDATimingEvent(ScopedCUDATimingEvent &&other) noexcept
            : event_(other.event_)
        {
            other.event_ = nullptr;
        }

        ScopedCUDATimingEvent &operator=(ScopedCUDATimingEvent &&other) noexcept
        {
            if (this != &other)
            {
                if (event_)
                    (void)cudaEventDestroy(event_);
                event_ = other.event_;
                other.event_ = nullptr;
            }
            return *this;
        }

        [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

    private:
        cudaEvent_t event_ = nullptr;
    };

    /**
     * @brief Persistent device state for one shape/M candidate tournament.
     *
     * Construction performs every allocation and transfer shared by the
     * candidates. `measure()` contains only production kernel launches and
     * stream-local event timing. `byteMismatches()` compares every output byte
     * on device after timing and downloads only mismatch metadata. Dedicated
     * grouped-verifier integration suites retain the stricter M=1
     * serial-decode oracle for M=2..16.
     */
    class PreparedSweepExecution
    {
    public:
        PreparedSweepExecution(
            ITensorGemm *kernel,
            int m,
            int n,
            int k,
            int device_id,
            const WorkspaceRequirements &requirements,
            int maximum_bench_runs)
            : kernel_(kernel), m_(m), n_(n), k_(k), device_id_(device_id)
        {
            if (!kernel_ || maximum_bench_runs <= 0 ||
                cudaSetDevice(device_id_) != cudaSuccess)
                throw std::runtime_error("invalid CUDA sweep execution device");

            workspace_consumer_ = dynamic_cast<IWorkspaceConsumer *>(kernel_);
            if (workspace_consumer_)
            {
                const size_t required = requirements.total_bytes_with_alignment();
                const size_t budget = std::max(
                    required + required / 10, size_t{64} * 1024 * 1024);
                workspace_ = std::make_unique<DeviceWorkspaceManager>(
                    DeviceId::cuda(device_id_), budget);
                if (!workspace_->allocate(requirements))
                    throw std::runtime_error(
                        "failed to allocate persistent CUDA sweep workspace");
                workspace_consumer_->bindWorkspace(workspace_.get());
            }

            host_input_ = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(m_), static_cast<size_t>(k_)},
                -0.25f, 0.25f, 7);
            input_ = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(m_), static_cast<size_t>(k_)});
            std::memcpy(
                input_->mutable_data(), host_input_->data(),
                static_cast<size_t>(m_) * k_ * sizeof(float));
            output_ = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(m_), static_cast<size_t>(n_)});
            oracle_output_ = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(m_), static_cast<size_t>(n_)});
            comparison_state_ = std::make_unique<FP32Tensor>(
                std::vector<size_t>{4});

            stream_owner_ = std::make_unique<llaminar2::test::ScopedGPUStream>(
                DeviceId::cuda(device_id_));
            stream_ = static_cast<cudaStream_t>(stream_owner_->get());
            kernel_->setGPUStream(static_cast<void *>(stream_));

            /*
             * Upload and publish every persistent tensor on the same exact
             * stream consumed by the measured kernel.  A null-stream setup
             * publication would violate the production ownership contract and
             * make this harness blind to stream-ordering regressions.
             */
            const DeviceId device = DeviceId::cuda(device_id_);
            if (!input_->ensureOnDevice(device, stream_) ||
                !output_->ensureOnDevice(device, stream_) ||
                !oracle_output_->ensureOnDevice(device, stream_) ||
                !comparison_state_->ensureOnDevice(device, stream_))
            {
                throw std::runtime_error(
                    "failed to allocate persistent CUDA sweep tensors");
            }

            start_events_.reserve(static_cast<size_t>(maximum_bench_runs));
            stop_events_.reserve(static_cast<size_t>(maximum_bench_runs));
            for (int index = 0; index < maximum_bench_runs; ++index)
            {
                start_events_.emplace_back();
                stop_events_.emplace_back();
            }
            times_us_.resize(static_cast<size_t>(maximum_bench_runs));

            buildExactMPrefillOracle();
        }

        ~PreparedSweepExecution()
        {
            if (stream_)
                (void)cudaStreamSynchronize(stream_);
            if (kernel_)
                kernel_->clearGPUStreamBinding();
            if (workspace_consumer_)
                workspace_consumer_->unbindWorkspace();
        }

        PreparedSweepExecution(const PreparedSweepExecution &) = delete;
        PreparedSweepExecution &operator=(const PreparedSweepExecution &) = delete;

        RunResult measure(int warmup_runs, int bench_runs)
        {
            if (warmup_runs <= 0 || bench_runs <= 0 ||
                static_cast<size_t>(bench_runs) > start_events_.size())
            {
                throw std::invalid_argument(
                    "invalid CUDA NativeVNNI timing repetition count");
            }
            for (int iteration = 0; iteration < warmup_runs; ++iteration)
                launch(m_, input_.get(), output_.get());
            checkStream("warmup");

            for (int iteration = 0; iteration < bench_runs; ++iteration)
            {
                const size_t index = static_cast<size_t>(iteration);
                if (cudaEventRecord(start_events_[index].get(), stream_) !=
                    cudaSuccess)
                    throw std::runtime_error(
                        "CUDA sweep start-event recording failed");
                launch(m_, input_.get(), output_.get());
                if (cudaEventRecord(stop_events_[index].get(), stream_) !=
                    cudaSuccess)
                    throw std::runtime_error(
                        "CUDA sweep stop-event recording failed");
            }
            if (cudaEventSynchronize(
                    stop_events_[static_cast<size_t>(bench_runs - 1)].get()) !=
                cudaSuccess)
            {
                throw std::runtime_error(
                    "CUDA sweep candidate event-batch synchronization failed");
            }
            for (int iteration = 0; iteration < bench_runs; ++iteration)
            {
                float elapsed_ms = 0.0f;
                if (cudaEventElapsedTime(
                        &elapsed_ms,
                        start_events_[static_cast<size_t>(iteration)].get(),
                        stop_events_[static_cast<size_t>(iteration)].get()) !=
                    cudaSuccess)
                    throw std::runtime_error(
                        "CUDA sweep elapsed-time query failed");
                times_us_[static_cast<size_t>(iteration)] =
                    static_cast<double>(elapsed_ms) * 1000.0;
            }
            if (const cudaError_t error = cudaGetLastError();
                error != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string("CUDA sweep candidate failed: ") +
                    cudaGetErrorString(error));
            }

            RunResult result;
            result.min_us = *std::min_element(
                times_us_.begin(),
                times_us_.begin() + bench_runs);
            result.mean_us = std::accumulate(
                                 times_us_.begin(),
                                 times_us_.begin() + bench_runs,
                                 0.0) /
                             static_cast<double>(bench_runs);
            result.native_family = "native_vnni_tc";
            last_timing_sample_count_ = static_cast<size_t>(bench_runs);
            return result;
        }

        /**
         * @brief Return the exact native-event samples from the last candidate.
         *
         * The view aliases persistent host storage owned by this prepared
         * tournament and remains valid until the next `measure()` call. This
         * avoids a per-candidate allocation while allowing the durable corpus
         * writer to retain every observation used by the aggregate statistics.
         */
        [[nodiscard]] std::span<const double> timingSamples() const noexcept
        {
            return std::span<const double>(
                times_us_.data(), last_timing_sample_count_);
        }

        /**
         * @brief Resolve one candidate before querying compiler resources.
         *
         * AUTO is a shape-dependent policy rather than a kernel symbol. One
         * untimed launch is therefore required to publish the exact physical
         * specialization selected by production. The stream wait is setup
         * work and occurs before any canonical timing event is recorded.
         */
        void probeLaunch()
        {
            launch(m_, input_.get(), output_.get());
            checkStream("resource-identity probe");
        }

        size_t byteMismatches()
        {
            auto *comparison_words = reinterpret_cast<uint64_t *>(
                comparison_state_->gpu_data_ptr());
            if (!llaminar2::test::enqueueCudaFP32ByteComparison(
                    reinterpret_cast<const float *>(output_->gpu_data_ptr()),
                    reinterpret_cast<const float *>(
                        oracle_output_->gpu_data_ptr()),
                    static_cast<size_t>(m_) * static_cast<size_t>(n_),
                    comparison_words,
                    comparison_words + 1,
                    stream_))
            {
                throw std::runtime_error(
                    "CUDA sweep full-buffer byte comparison launch failed");
            }

            std::array<uint64_t, 2> result{};
            if (cudaMemcpyAsync(
                    result.data(),
                    comparison_words,
                    sizeof(result),
                    cudaMemcpyDeviceToHost,
                    stream_) != cudaSuccess)
            {
                throw std::runtime_error(
                    "CUDA sweep mismatch metadata download failed");
            }
            checkStream("full-buffer byte certificate");
            first_byte_mismatch_ = result[1];
            return static_cast<size_t>(result[0]);
        }

        /** @return Least mismatched byte offset or `UINT64_MAX`. */
        uint64_t firstByteMismatch() const
        {
            return first_byte_mismatch_;
        }

    private:
        void launch(int m, FP32Tensor *input, FP32Tensor *output)
        {
            if (!kernel_->multiply_tensor(input, output, m, n_, k_))
                throw std::runtime_error("CUDA NativeVNNI sweep launch failed");
        }

        void checkStream(const char *operation)
        {
            if (const cudaError_t error = cudaStreamSynchronize(stream_);
                error != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string("CUDA sweep ") + operation + " failed: " +
                    cudaGetErrorString(error));
            }
        }

        void buildExactMPrefillOracle()
        {
            cudaNativeVNNIPrefill_setBK256Mode(0);
            cudaNativeVNNIPrefill_setForceTile(-1);
            cudaNativeVNNIPrefill_setCanonicalKPartitionMode(false);
            launch(m_, input_.get(), oracle_output_.get());
            checkStream("exact-M device oracle construction");
        }

        ITensorGemm *kernel_ = nullptr;
        IWorkspaceConsumer *workspace_consumer_ = nullptr;
        int m_ = 0;
        int n_ = 0;
        int k_ = 0;
        int device_id_ = 0;
        std::unique_ptr<DeviceWorkspaceManager> workspace_;
        std::unique_ptr<FP32Tensor> host_input_;
        std::unique_ptr<FP32Tensor> input_;
        std::unique_ptr<FP32Tensor> output_;
        std::unique_ptr<FP32Tensor> oracle_output_;
        std::unique_ptr<FP32Tensor> comparison_state_;
        std::unique_ptr<llaminar2::test::ScopedGPUStream> stream_owner_;
        cudaStream_t stream_ = nullptr;
        std::vector<ScopedCUDATimingEvent> start_events_;
        std::vector<ScopedCUDATimingEvent> stop_events_;
        std::vector<double> times_us_;
        size_t last_timing_sample_count_ = 0;
        uint64_t first_byte_mismatch_ = std::numeric_limits<uint64_t>::max();
    };

    static size_t estimateVramBytes(int m, int n, int k)
    {
        const size_t a_bytes = static_cast<size_t>(m) * k * sizeof(float);
        const size_t c_bytes = static_cast<size_t>(m) * n * sizeof(float);
        // Q4_0 weight: n*k / 2 (4 bits per element) + scale overhead
        const size_t w_bytes = static_cast<size_t>(n) * k / 2 + static_cast<size_t>(n) * (k / 32) * 2;
        // Workspace: quant_a(M*K) + scales_a(M*4) + acc_int32(M*N*4) +
        //            concurrent prefill extra acc slots(2*M*N*4) +
        //            scales_a_blockwise(M*(K/32)*4) + temp_c_fp32(M*N*4)
        const size_t workspace_bytes = static_cast<size_t>(m) * k                     // quant_a (int8)
                                       + static_cast<size_t>(m) * 4                   // scales_a
                                       + static_cast<size_t>(m) * n * 4               // acc_int32
                                       + 2 * static_cast<size_t>(m) * n * 4           // concurrent prefill extra acc
                                       + static_cast<size_t>(m) * ((k + 31) / 32) * 4 // scales_a_blockwise
                                       + static_cast<size_t>(m) * n * 4;              // temp_c_fp32
        return a_bytes + c_bytes + w_bytes + workspace_bytes;
    }

    TEST_F(CUDANativeVNNIGemmPerf, TileSweep_AllStrategies)
    {
        ScopedDensePrefillOverlayBypass overlay_bypass;
        const SweepConfig cfg = loadSweepConfig();
        const double peak_tops = peak_tc_tops_;
        // Force-mode controls are process-wide. The turnkey collector obtains
        // parallelism by launching one masked process per physical GPU, keeping
        // each candidate transaction isolated and allowing persistent state to
        // remain bound for the complete shape/M tournament.
        constexpr int worker_count = 1;

        // Query VRAM on device 0
        size_t vram_free = 0, vram_total = 0;
        cudaMemGetInfo(&vram_free, &vram_total);
        // Use 85% of free VRAM as budget (leave headroom for driver allocations)
        const size_t vram_budget = static_cast<size_t>(static_cast<double>(vram_free) * 0.85);

        std::fprintf(stderr,
                     "[TileSweep] %d GPU(s) detected, VRAM free=%.0f MB, budget=%.0f MB\n",
                     worker_count,
                     static_cast<double>(vram_free) / (1024.0 * 1024.0),
                     static_cast<double>(vram_budget) / (1024.0 * 1024.0));

        // Find the format factory
        std::function<std::unique_ptr<TensorBase>(size_t, size_t)> create_weights;
        std::string resolved_format_name;
        uint8_t resolved_codebook_id = 0;
        uint8_t resolved_execution_codebook_id = 0;
        {
            std::string fmt_lower = toLower(cfg.format_name);
            bool found = false;
            for (const auto &f : kFormats)
            {
                if (toLower(f.name) == fmt_lower)
                {
                    create_weights = f.create;
                    resolved_format_name = f.name;
                    resolved_codebook_id = f.codebook_id;
                    resolved_execution_codebook_id = f.runtimeCodebook();
                    found = true;
                    break;
                }
            }
            ASSERT_TRUE(found) << "Unknown format: " << cfg.format_name;
        }

        // ── Build flat task list ──
        std::vector<SweepTask> tasks;
        const uint8_t execution_codebook_id =
            resolved_execution_codebook_id;
        const auto has_canonical_kpart =
            [execution_codebook_id](int n, int k)
        {
            int shape_id = -1;
            int tile_n = 0;
            int cpt = 0;
            int exact_kb = 0;
            if (!cudaNativeVNNIGemvTuned_queryGeneratedDispatch(
                    execution_codebook_id,
                    /*graph_captured=*/1,
                    /*m=*/1,
                    n,
                    k,
                    &shape_id,
                    &tile_n,
                    &cpt,
                    &exact_kb))
            {
                throw std::runtime_error(
                    "generated M=1 dispatch is incomplete for dense sweep");
            }
            // NativeGemvShape::KPAR is the second scoped-enum value.
            return shape_id == 1;
        };
        const auto resource_eligible =
            [execution_codebook_id](
                int tile_id,
                bool canonical_kpart,
                bool ordered_bk256)
        {
            CUDADensePrefillKernelResources primary{};
            CUDADensePrefillKernelResources auxiliary{};
            if (!cudaNativeVNNIPrefill_queryCandidateResources(
                    execution_codebook_id,
                    tile_id,
                    canonical_kpart ? 1 : 0,
                    ordered_bk256 ? 1 : 0,
                    /*cuda_device_id=*/0,
                    &primary,
                    &auxiliary))
            {
                throw std::runtime_error(
                    "CUDA dense prefill candidate inventory query failed");
            }
            return primary.local_memory_bytes_per_thread == 0 &&
                   auxiliary.local_memory_bytes_per_thread == 0;
        };

        for (const auto &shape : kQwenShapes)
        {
            if (!shouldRunName(cfg.shape_filters, shape.name))
                continue;

            for (int m : cfg.prefill_m)
            {
                // Check VRAM budget before adding tasks for this (shape, M)
                const size_t estimated_vram = estimateVramBytes(m, shape.n, shape.k);
                if (estimated_vram > vram_budget)
                {
                    std::fprintf(stderr,
                                 "[TileSweep] SKIPPED %s M=%d: estimated %.0f MB > budget %.0f MB\n",
                                 shape.name.c_str(), m,
                                 static_cast<double>(estimated_vram) / (1024.0 * 1024.0),
                                 static_cast<double>(vram_budget) / (1024.0 * 1024.0));
                    continue;
                }

                for (Strategy strat : cfg.strategies)
                {
                    /*
                     * BK256 is implemented only by the Q4_0 execution family.
                     * Advertising it for another codebook merely relaunches
                     * AUTO under a false candidate name and corrupts the
                     * tournament, so do not create that task.
                     */
                    if (strat == Strategy::BK256 &&
                        resolved_format_name != "Q4_0")
                    {
                        continue;
                    }
                    if (strat == Strategy::BK256 &&
                        (!resource_eligible(-3, false, false) ||
                         !resource_eligible(-3, false, true) ||
                         !resource_eligible(-2, false, false) ||
                         !resource_eligible(-2, false, true)))
                    {
                        continue;
                    }
                    if (strat == Strategy::CanonicalKpart &&
                        !has_canonical_kpart(shape.n, shape.k))
                    {
                        continue;
                    }
                    if (strat == Strategy::Auto)
                    {
                        tasks.push_back({&shape, m, strat, -1, 0, "AUTO", 0});
                    }
                    else if (strat == Strategy::BK256)
                    {
                        const int bk256_tiles = ((m + 127) / 128) * ((shape.n + 127) / 128);
                        tasks.push_back(
                            {&shape, m, strat, -2, 1,
                             "BK256_128x128", bk256_tiles});
                    }
                    else
                    {
                        // BK64 strategies: iterate tiles
                        for (int tile_id : cfg.tile_ids)
                        {
                            if (!resource_eligible(
                                    tile_id,
                                    strat == Strategy::CanonicalKpart,
                                    false))
                            {
                                continue;
                            }
                            const int tile_count = cudaNativeVNNIPrefill_getTileCount(tile_id, m, shape.n);
                            const int candidate_k_partitions =
                                strat == Strategy::CanonicalKpart ? 0 : 1;
                            tasks.push_back(
                                {&shape, m, strat, tile_id,
                                 candidate_k_partitions,
                                 kAllTiles[tile_id].name, tile_count});
                        }
                    }
                }
            }
        }

        ASSERT_FALSE(tasks.empty()) << "No tile sweep cases selected. Check env var filters.";

        std::fprintf(stderr,
                     "[TileSweep] %zu tasks to run across %d GPU(s)\n",
                     tasks.size(), worker_count);

        // ── Open CSV for incremental output ──
        FILE *csv_fp = nullptr;
        if (!cfg.csv_path.empty())
        {
            csv_fp = std::fopen(cfg.csv_path.c_str(), "w");
            ASSERT_NE(csv_fp, nullptr) << "Failed to open CSV: " << cfg.csv_path;
            std::fprintf(csv_fp,
                         "format,codebook,shape,m,n,k,tile,tile_id,strategy,requested_k_partitions,tiles,"
                         "min_us,mean_us,tops,pct_peak,gpu,byte_mismatches,"
                         "first_byte_mismatch,correctness_pass,observed_tile_id,"
                         "observed_k_partitions,observed_bk256,"
                         "observed_canonical_kpart,canonical_kpart_available,"
                         "primary_registers_per_thread,"
                         "primary_local_memory_bytes_per_thread,"
                         "primary_static_shared_memory_bytes,"
                         "primary_dynamic_shared_memory_bytes,"
                         "primary_threads_per_block,"
                         "primary_max_threads_per_block,"
                         "primary_max_active_blocks_per_sm,"
                         "auxiliary_registers_per_thread,"
                         "auxiliary_local_memory_bytes_per_thread,"
                         "auxiliary_static_shared_memory_bytes,"
                         "auxiliary_dynamic_shared_memory_bytes,"
                         "auxiliary_threads_per_block,"
                         "auxiliary_max_threads_per_block,"
                         "auxiliary_max_active_blocks_per_sm\n");
            std::fflush(csv_fp);
        }

        FILE *timing_csv_fp = nullptr;
        if (!cfg.timing_csv_path.empty())
        {
            timing_csv_fp = std::fopen(cfg.timing_csv_path.c_str(), "w");
            ASSERT_NE(timing_csv_fp, nullptr)
                << "Failed to open raw timing CSV: " << cfg.timing_csv_path;
            std::fprintf(
                timing_csv_fp,
                "backend,phase,format,codebook,shape,m,n,k,tile,tile_id,"
                "strategy,requested_k_partitions,sample_index,timed_replays,latency_us,"
                "latency_us_hex\n");
            std::fflush(timing_csv_fp);
        }

        // ── Run one allocation-free candidate tournament on the masked GPU ──
        std::vector<SweepRow> rows(tasks.size());
        std::vector<bool> row_valid(tasks.size(), false);
        constexpr int gpu_id = 0;
        const Shape *prepared_shape = nullptr;
        int prepared_m = 0;
        std::unique_ptr<TensorBase> weights;
        std::unique_ptr<llaminar2::test::GpuPreparedGemm> prepared;
        std::unique_ptr<PreparedSweepExecution> execution;

        for (size_t task_idx = 0; task_idx < tasks.size(); ++task_idx)
        {
            const SweepTask &task = tasks[task_idx];
            const auto &shape = *task.shape;
            std::fprintf(stderr,
                         "[TileSweep][gpu=%d] %s M=%d %s %s kpart=%d tiles=%d\n",
                         gpu_id, shape.name.c_str(), task.m,
                         task.tile_name.c_str(), strategyName(task.strat),
                         task.requested_k_partitions, task.tiles);

            if (prepared_shape != task.shape)
            {
                execution.reset();
                prepared.reset();
                weights = create_weights(
                    static_cast<size_t>(shape.n),
                    static_cast<size_t>(shape.k));
                prepared = std::make_unique<llaminar2::test::GpuPreparedGemm>(
                    llaminar2::test::makeGpuPreparedGemm(
                        weights.get(), DeviceId::cuda(gpu_id),
                        "perf.cuda_native_vnni_prefill." + shape.name));
                prepared_shape = task.shape;
                prepared_m = 0;
            }
            if (prepared_m != task.m)
            {
                execution.reset();
                auto *consumer =
                    dynamic_cast<IWorkspaceConsumer *>(prepared->kernel);
                const WorkspaceRequirements requirements =
                    tournamentWorkspaceRequirements(
                        consumer, tasks, task.shape, task.m);
                execution = std::make_unique<PreparedSweepExecution>(
                    prepared->kernel, task.m, shape.n, shape.k, gpu_id,
                    requirements, cfg.bench_runs);
                prepared_m = task.m;
            }

            configureSweepTask(task);
            const uint8_t source_codebook_id =
                requireNativeVnniInfo(weights.get(), resolved_format_name)
                    .codebook_id;
            if (source_codebook_id != resolved_codebook_id)
            {
                throw std::runtime_error(
                    "CUDA dense prefill source codebook identity changed");
            }

            /*
             * Resolve AUTO through the real production launcher, then inspect
             * that exact symbol before timing it. A spilling specialization
             * is an invalid dispatch target, so it fails the transaction here
             * rather than contributing even one canonical timing sample.
             */
            execution->probeLaunch();
            int probed_tile_id = -99;
            int probed_k_partitions = -99;
            int probed_bk256 = -99;
            int probed_canonical_kpart = -99;
            cudaNativeVNNIPrefill_getLastLaunchSelection(
                &probed_tile_id,
                &probed_k_partitions,
                &probed_bk256,
                &probed_canonical_kpart);

            CUDADensePrefillKernelResources primary_resources{};
            CUDADensePrefillKernelResources auxiliary_resources{};
            if (!cudaNativeVNNIPrefill_queryLastLaunchResources(
                    execution_codebook_id,
                    shape.n,
                    shape.k,
                    gpu_id,
                    &primary_resources,
                    &auxiliary_resources))
            {
                throw std::runtime_error(
                    "CUDA dense prefill candidate resource query failed");
            }
            const int primary_registers =
                primary_resources.registers_per_thread;
            const size_t primary_local_bytes =
                primary_resources.local_memory_bytes_per_thread;
            const size_t primary_static_shared_bytes =
                primary_resources.static_shared_memory_bytes;
            const size_t primary_dynamic_shared_bytes =
                primary_resources.dynamic_shared_memory_bytes;
            const int primary_threads = primary_resources.threads_per_block;
            const int primary_max_threads =
                primary_resources.max_threads_per_block;
            const int primary_active_blocks =
                primary_resources.max_active_blocks_per_sm;
            const int auxiliary_registers =
                auxiliary_resources.registers_per_thread;
            const size_t auxiliary_local_bytes =
                auxiliary_resources.local_memory_bytes_per_thread;
            const size_t auxiliary_static_shared_bytes =
                auxiliary_resources.static_shared_memory_bytes;
            const size_t auxiliary_dynamic_shared_bytes =
                auxiliary_resources.dynamic_shared_memory_bytes;
            const int auxiliary_threads =
                auxiliary_resources.threads_per_block;
            const int auxiliary_max_threads =
                auxiliary_resources.max_threads_per_block;
            const int auxiliary_active_blocks =
                auxiliary_resources.max_active_blocks_per_sm;
            if (primary_registers <= 0 || primary_threads <= 0 ||
                primary_max_threads < primary_threads ||
                primary_active_blocks <= 0 || primary_local_bytes != 0 ||
                auxiliary_local_bytes != 0 ||
                (auxiliary_threads > 0 &&
                 (auxiliary_registers <= 0 ||
                  auxiliary_max_threads < auxiliary_threads ||
                  auxiliary_active_blocks <= 0)))
            {
                std::fprintf(
                    stderr,
                    "[TileSweep][resources] rejected %s/%s/M=%d/%s/%s "
                    "primary={regs=%d,local=%zu,static_smem=%zu,"
                    "dynamic_smem=%zu,threads=%d,max_threads=%d,"
                    "active_blocks=%d} auxiliary={regs=%d,local=%zu,"
                    "static_smem=%zu,dynamic_smem=%zu,threads=%d,"
                    "max_threads=%d,active_blocks=%d}\n",
                    resolved_format_name.c_str(),
                    shape.name.c_str(),
                    task.m,
                    task.tile_name.c_str(),
                    strategyName(task.strat),
                    primary_registers,
                    primary_local_bytes,
                    primary_static_shared_bytes,
                    primary_dynamic_shared_bytes,
                    primary_threads,
                    primary_max_threads,
                    primary_active_blocks,
                    auxiliary_registers,
                    auxiliary_local_bytes,
                    auxiliary_static_shared_bytes,
                    auxiliary_dynamic_shared_bytes,
                    auxiliary_threads,
                    auxiliary_max_threads,
                    auxiliary_active_blocks);
                throw std::runtime_error(
                    "CUDA dense prefill candidate failed the zero-spill/"
                    "occupancy eligibility gate");
            }

            const RunResult rr = execution->measure(
                cfg.warmup_runs, cfg.bench_runs);
            int observed_tile_id = -99;
            int observed_k_partitions = -99;
            int observed_bk256 = -99;
            int observed_canonical_kpart = -99;
            cudaNativeVNNIPrefill_getLastLaunchSelection(
                &observed_tile_id,
                &observed_k_partitions,
                &observed_bk256,
                &observed_canonical_kpart);
            if (std::tie(
                    observed_tile_id,
                    observed_k_partitions,
                    observed_bk256,
                    observed_canonical_kpart) !=
                std::tie(
                    probed_tile_id,
                    probed_k_partitions,
                    probed_bk256,
                    probed_canonical_kpart))
            {
                throw std::runtime_error(
                    "CUDA dense prefill route changed after resource proof");
            }
            const size_t byte_mismatches = execution->byteMismatches();
            const auto metrics = computeGemmThroughputMetrics(
                task.m, shape.n, shape.k, rr.min_us, peak_tops);
            if (timing_csv_fp)
            {
                const auto samples = execution->timingSamples();
                for (size_t sample_index = 0;
                     sample_index < samples.size(); ++sample_index)
                {
                    const double latency_us = samples[sample_index];
                    std::fprintf(
                        timing_csv_fp,
                        "cuda,prefill,%s,%u,%s,%d,%d,%d,%s,%d,%s,%d,"
                        "%zu,1,%.9f,%a\n",
                        resolved_format_name.c_str(),
                        static_cast<unsigned>(source_codebook_id),
                        shape.name.c_str(), task.m, shape.n, shape.k,
                        task.tile_name.c_str(), task.tile_id,
                        strategyName(task.strat), task.requested_k_partitions,
                        sample_index, latency_us, latency_us);
                }
                std::fflush(timing_csv_fp);
            }

            SweepRow &row = rows[task_idx];
            row.format_name = resolved_format_name;
            row.codebook_id = source_codebook_id;
            row.shape_name = shape.name;
            row.m = task.m;
            row.n = shape.n;
            row.k = shape.k;
            row.tile_name = task.tile_name;
            row.tile_id = task.tile_id;
            row.tiles = task.tiles;
            row.strategy = strategyName(task.strat);
            row.requested_k_partitions = task.requested_k_partitions;
            row.min_us = rr.min_us;
            row.mean_us = rr.mean_us;
            row.tops = metrics.achieved_tops;
            row.pct_peak = metrics.pct_tc_peak;
            row.gpu_id = gpu_id;
            row.byte_mismatches = byte_mismatches;
            row.first_byte_mismatch = execution->firstByteMismatch();
            row.correctness_pass = byte_mismatches == 0;
            row.canonical_kpart_available =
                has_canonical_kpart(shape.n, shape.k);
            row.observed_tile_id = observed_tile_id;
            row.observed_k_partitions = observed_k_partitions;
            row.observed_bk256 = observed_bk256;
            row.observed_canonical_kpart = observed_canonical_kpart;
            row.primary_registers_per_thread = primary_registers;
            row.primary_local_memory_bytes_per_thread = primary_local_bytes;
            row.primary_static_shared_memory_bytes =
                primary_static_shared_bytes;
            row.primary_dynamic_shared_memory_bytes =
                primary_dynamic_shared_bytes;
            row.primary_threads_per_block = primary_threads;
            row.primary_max_threads_per_block = primary_max_threads;
            row.primary_max_active_blocks_per_sm = primary_active_blocks;
            row.auxiliary_registers_per_thread = auxiliary_registers;
            row.auxiliary_local_memory_bytes_per_thread =
                auxiliary_local_bytes;
            row.auxiliary_static_shared_memory_bytes =
                auxiliary_static_shared_bytes;
            row.auxiliary_dynamic_shared_memory_bytes =
                auxiliary_dynamic_shared_bytes;
            row.auxiliary_threads_per_block = auxiliary_threads;
            row.auxiliary_max_threads_per_block = auxiliary_max_threads;
            row.auxiliary_max_active_blocks_per_sm = auxiliary_active_blocks;
            row_valid[task_idx] = true;

            if (csv_fp)
            {
                std::fprintf(csv_fp,
                             "%s,%u,%s,%d,%d,%d,%s,%d,%s,%d,%d,"
                             "%.3f,%.3f,%.9f,%.9f,%d,%zu,%llu,%d,%d,%d,%d,%d,%d,"
                             "%d,%zu,%zu,%zu,%d,%d,%d,%d,%zu,%zu,%zu,%d,%d,%d\n",
                             row.format_name.c_str(),
                             static_cast<unsigned>(row.codebook_id),
                             row.shape_name.c_str(), row.m, row.n, row.k,
                             row.tile_name.c_str(), row.tile_id,
                             row.strategy.c_str(), row.requested_k_partitions, row.tiles,
                             row.min_us, row.mean_us, row.tops, row.pct_peak,
                             row.gpu_id, row.byte_mismatches,
                             static_cast<unsigned long long>(
                                 row.first_byte_mismatch),
                             row.correctness_pass ? 1 : 0,
                             row.observed_tile_id,
                             row.observed_k_partitions,
                             row.observed_bk256,
                             row.observed_canonical_kpart,
                             row.canonical_kpart_available ? 1 : 0,
                             row.primary_registers_per_thread,
                             row.primary_local_memory_bytes_per_thread,
                             row.primary_static_shared_memory_bytes,
                             row.primary_dynamic_shared_memory_bytes,
                             row.primary_threads_per_block,
                             row.primary_max_threads_per_block,
                             row.primary_max_active_blocks_per_sm,
                             row.auxiliary_registers_per_thread,
                             row.auxiliary_local_memory_bytes_per_thread,
                             row.auxiliary_static_shared_memory_bytes,
                             row.auxiliary_dynamic_shared_memory_bytes,
                             row.auxiliary_threads_per_block,
                             row.auxiliary_max_threads_per_block,
                             row.auxiliary_max_active_blocks_per_sm);
                std::fflush(csv_fp);
            }
        }

        execution.reset();
        prepared.reset();

        // Restore original modes
        cudaNativeVNNIPrefill_setBK256Mode(0);
        cudaNativeVNNIPrefill_setForceTile(-1);
        cudaNativeVNNIPrefill_setCanonicalKPartitionMode(false);

        // Collect valid rows
        std::vector<SweepRow> valid_rows;
        valid_rows.reserve(tasks.size());
        for (size_t i = 0; i < tasks.size(); ++i)
            if (row_valid[i])
                valid_rows.push_back(std::move(rows[i]));

        ASSERT_FALSE(valid_rows.empty()) << "No tile sweep cases ran.";

        // ── Render summary table ──
        // Group by (shape, M), find best tile+strategy per group
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Shape" << "M" << "N" << "K" << "Tile" << "Strat"
              << "SK" << "Tiles" << "Min (us)" << "TOPS" << "%Peak" << "Best?"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(4).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 11; ++c)
            if (c != 4)
                table.column(c).set_cell_text_align(fort::text_align::right);

        // Find best per (shape, M)
        std::map<std::string, double> best_for_group;
        for (const auto &r : valid_rows)
        {
            if (!r.correctness_pass)
                continue;
            std::string key = r.shape_name + "_" + std::to_string(r.m);
            auto it = best_for_group.find(key);
            if (it == best_for_group.end() || r.min_us < it->second)
                best_for_group[key] = r.min_us;
        }

        for (const auto &r : valid_rows)
        {
            char min_us[16], tops[16], pct[16];
            std::snprintf(min_us, sizeof(min_us), "%.1f", r.min_us);
            std::snprintf(tops, sizeof(tops), "%.2f", r.tops);
            std::snprintf(pct, sizeof(pct), "%.1f%%", r.pct_peak);

            std::string key = r.shape_name + "_" + std::to_string(r.m);
            const auto best = best_for_group.find(key);
            const bool is_best = r.correctness_pass &&
                                 best != best_for_group.end() &&
                                 r.min_us <= best->second * 1.001;

            table << r.shape_name << r.m << r.n << r.k
                  << r.tile_name << r.strategy << r.requested_k_partitions << r.tiles
                  << min_us << tops << pct << (is_best ? "<< BEST" : "")
                  << fort::endr;
        }

        std::fprintf(stderr, "\n%s\n", table.to_string().c_str());

        // ── Close CSV ──
        if (csv_fp)
        {
            std::fclose(csv_fp);
            std::fprintf(stderr,
                         "[TileSweep] Results written to %s\n",
                         cfg.csv_path.c_str());
        }
        if (timing_csv_fp)
        {
            ASSERT_EQ(std::fclose(timing_csv_fp), 0);
            std::fprintf(
                stderr,
                "[TileSweep] Raw timings written to %s\n",
                cfg.timing_csv_path.c_str());
        }

        // Summary: for each (shape, M), print winner
        std::fprintf(stderr, "\n[TileSweep] Winners per (shape, M):\n");
        std::string prev_key;
        for (const auto &r : valid_rows)
        {
            std::string key = r.shape_name + "_" + std::to_string(r.m);
            if (key == prev_key)
                continue;
            const auto best = best_for_group.find(key);
            const bool is_best = r.correctness_pass &&
                                 best != best_for_group.end() &&
                                 r.min_us <= best->second * 1.001;
            if (is_best)
            {
                std::fprintf(stderr, "  %s M=%d: %s %s sk=%d → %.1f us\n",
                             r.shape_name.c_str(), r.m, r.tile_name.c_str(),
                             r.strategy.c_str(), r.requested_k_partitions, r.min_us);
                prev_key = key;
            }
        }

        std::set<std::string> selected_groups;
        for (const auto &task : tasks)
            selected_groups.insert(
                task.shape->name + "_" + std::to_string(task.m));
        EXPECT_EQ(best_for_group.size(), selected_groups.size())
            << "Every exact-overlay shape/M tournament requires at least one "
               "byte-exact candidate";
    }
}

#endif
