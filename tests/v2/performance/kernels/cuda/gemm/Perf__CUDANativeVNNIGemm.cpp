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
#include <mutex>
#include <set>
#include <thread>
#include <tuple>

#ifdef HAVE_CUDA

#include <cuda_runtime.h>

#include "CUDANativeVNNIGemmPerfCommon.h"
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
    void cudaNativeVNNIPrefill_setStreamKMode(int mode);
    int cudaNativeVNNIPrefill_getStreamKMode();
    void cudaNativeVNNIPrefill_setBK256Mode(int mode);
    int cudaNativeVNNIPrefill_getBK256Mode();
    void cudaNativeVNNIPrefill_setForceTile(int tile_id, int split_k);
    void cudaNativeVNNIPrefill_getForceTile(int *tile_id, int *split_k);
    void cudaNativeVNNIPrefill_freeStreamKFixup();
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
    //   - AUTO:     production heuristic (BK256/BK64/StreamK all auto-selected)
    //   - STD:      BK64 with forced tile + split-K, stream-K off
    //   - SK1:      BK64 with one-pass stream-K (atomicAdd, diagnostic only)
    //   - SK2:      BK64 with two-pass stream-K (fixup buffer)
    //   - BK256:    BK256 forced, split-K=1
    //   - BK256_SK: BK256 forced, with split-K sweep (1,2,4,8)
    //
    // Environment variables:
    //   LLAMINAR_TILE_SWEEP_SHAPES       - Comma-separated shape names (default: all kQwenShapes)
    //   LLAMINAR_TILE_SWEEP_PREFILL_M    - Comma-separated M values
    //                                     (default: MTP rows + canonical prefill buckets)
    //   LLAMINAR_TILE_SWEEP_TILES        - Comma-separated tile IDs 0..5 for BK64 (default: all)
    //   LLAMINAR_TILE_SWEEP_STRATEGIES   - Comma-separated: auto,std,sk1,sk2,bk256,bk256_sk (default: all)
    //   LLAMINAR_TILE_SWEEP_WARMUP       - Warmup runs (default: 3)
    //   LLAMINAR_TILE_SWEEP_BENCH        - Benchmark runs (default: 10)
    //   LLAMINAR_TILE_SWEEP_CSV          - CSV output path
    //   LLAMINAR_TILE_SWEEP_SPLIT_K      - Comma-separated BK64 split-K values (default: 1)
    //   LLAMINAR_TILE_SWEEP_BK256_SPLIT_K- Comma-separated BK256 split-K values (default: 1,2,4,8)
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
        Auto,     // Production heuristic (BK256/BK64/StreamK all auto)
        Standard, // BK64 only, Stream-K OFF, heuristic split-K or forced split-K
        SK1,      // BK64 only, one-pass stream-K (atomicAdd)
        SK2,      // BK64 only, two-pass stream-K (fixup buffer)
        BK256,    // BK256 forced, split-K=1
        BK256_SK, // BK256 forced, with split-K sweep (2,4,8)
    };

    struct SweepConfig
    {
        int warmup_runs = 3;
        int bench_runs = 10;
        std::vector<int> prefill_m = llaminar2::defaultPrefillGraphBucketSizes();
        std::set<std::string> shape_filters;
        std::vector<int> tile_ids = {0, 1, 2, 3, 4, 5};
        // SK1 performs an atomic accumulation whose ordering is not invariant
        // across row grouping. It remains forceable for diagnostics, but it is
        // deliberately absent from the production candidate tournament.
        std::vector<Strategy> strategies = {Strategy::Auto, Strategy::Standard, Strategy::SK2, Strategy::BK256, Strategy::BK256_SK};
        std::vector<int> split_k_values = {1};
        std::vector<int> bk256_split_k_values = {1, 2, 4, 8};
        std::string csv_path;
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
            if (strat_set.count("sk1"))
                cfg.strategies.push_back(Strategy::SK1);
            if (strat_set.count("sk2"))
                cfg.strategies.push_back(Strategy::SK2);
            if (strat_set.count("bk256"))
                cfg.strategies.push_back(Strategy::BK256);
            if (strat_set.count("bk256_sk"))
                cfg.strategies.push_back(Strategy::BK256_SK);
        }

        const auto sk_vals = getEnvCsvInts("LLAMINAR_TILE_SWEEP_SPLIT_K");
        if (!sk_vals.empty())
            cfg.split_k_values = sk_vals;

        const auto bk256_sk_vals = getEnvCsvInts("LLAMINAR_TILE_SWEEP_BK256_SPLIT_K");
        if (!bk256_sk_vals.empty())
            cfg.bk256_split_k_values = bk256_sk_vals;

        const std::string csv = getEnvString("LLAMINAR_TILE_SWEEP_CSV");
        if (!csv.empty())
            cfg.csv_path = csv;

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
        case Strategy::SK1:
            return "SK1";
        case Strategy::SK2:
            return "SK2";
        case Strategy::BK256:
            return "BK256";
        case Strategy::BK256_SK:
            return "BK256_SK";
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
        int split_k; // 1 for SK, actual value for standard
        double min_us;
        double mean_us;
        double tops;
        double pct_peak;
        int gpu_id;
        size_t bit_mismatches;
        bool correctness_pass;
    };

    struct SweepTask
    {
        const Shape *shape;
        int m;
        Strategy strat;
        int tile_id; // -1=auto, -2=BK256, 0..5=BK64 tile
        int split_k;
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
            cudaNativeVNNIPrefill_setStreamKMode(0);
            cudaNativeVNNIPrefill_setForceTile(-1, 0);
        }
        else if (task.tile_id == -2)
        {
            cudaNativeVNNIPrefill_setBK256Mode(1);
            cudaNativeVNNIPrefill_setStreamKMode(-1);
            cudaNativeVNNIPrefill_setForceTile(-1, task.split_k);
        }
        else
        {
            cudaNativeVNNIPrefill_setBK256Mode(-1);
            cudaNativeVNNIPrefill_setForceTile(task.tile_id, task.split_k);
            cudaNativeVNNIPrefill_setStreamKMode(
                task.strat == Strategy::SK1 ? 1
                : task.strat == Strategy::SK2 ? 2
                                              : -1);
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

        // The bitwise oracle executes production serial decode through M=1.
        // Include that route before binding the immutable tournament workspace.
        cudaNativeVNNIPrefill_setBK256Mode(0);
        cudaNativeVNNIPrefill_setStreamKMode(0);
        cudaNativeVNNIPrefill_setForceTile(-1, 0);
        merge(consumer->getWorkspaceRequirements(1, shape->n, shape->k));
        return merged;
    }

    /**
     * @brief Persistent device state for one shape/M candidate tournament.
     *
     * Construction performs every allocation and transfer shared by the
     * candidates. `measure()` contains only production kernel launches and
     * stream-local event timing. `bitMismatches()` performs the small D2H row
     * sample after timing and compares it with production serial M=1 results.
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
            const WorkspaceRequirements &requirements)
            : kernel_(kernel), m_(m), n_(n), k_(k), device_id_(device_id)
        {
            if (!kernel_ || cudaSetDevice(device_id_) != cudaSuccess)
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
            serial_input_ = std::make_unique<FP32Tensor>(
                std::vector<size_t>{1, static_cast<size_t>(k_)});
            serial_output_ = std::make_unique<FP32Tensor>(
                std::vector<size_t>{1, static_cast<size_t>(n_)});

            const DeviceId device = DeviceId::cuda(device_id_);
            if (!input_->ensureOnDevice(device) ||
                !output_->ensureOnDevice(device) ||
                !serial_input_->ensureOnDevice(device) ||
                !serial_output_->ensureOnDevice(device))
            {
                throw std::runtime_error(
                    "failed to allocate persistent CUDA sweep tensors");
            }
            if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) !=
                cudaSuccess)
                throw std::runtime_error(
                    "failed to create persistent CUDA sweep stream");
            if (cudaEventCreate(&start_) != cudaSuccess)
            {
                (void)cudaStreamDestroy(stream_);
                stream_ = nullptr;
                throw std::runtime_error(
                    "failed to create persistent CUDA sweep start event");
            }
            if (cudaEventCreate(&stop_) != cudaSuccess)
            {
                (void)cudaEventDestroy(start_);
                start_ = nullptr;
                (void)cudaStreamDestroy(stream_);
                stream_ = nullptr;
                throw std::runtime_error(
                    "failed to create persistent CUDA sweep stop event");
            }
            kernel_->setGPUStream(static_cast<void *>(stream_));
            oracle_rows_ = serialOracleRows(m_);
            serial_oracle_.resize(oracle_rows_.size() * static_cast<size_t>(n_));
            candidate_rows_.resize(serial_oracle_.size());
            buildSerialOracle();
        }

        ~PreparedSweepExecution()
        {
            if (stream_)
                (void)cudaStreamSynchronize(stream_);
            if (kernel_)
                kernel_->setGPUStream(nullptr);
            if (workspace_consumer_)
                workspace_consumer_->unbindWorkspace();
            if (start_)
                (void)cudaEventDestroy(start_);
            if (stop_)
                (void)cudaEventDestroy(stop_);
            if (stream_)
                (void)cudaStreamDestroy(stream_);
        }

        PreparedSweepExecution(const PreparedSweepExecution &) = delete;
        PreparedSweepExecution &operator=(const PreparedSweepExecution &) = delete;

        RunResult measure(int warmup_runs, int bench_runs)
        {
            for (int iteration = 0; iteration < warmup_runs; ++iteration)
                launch(m_, input_.get(), output_.get());
            checkStream("warmup");

            std::vector<double> times_us;
            times_us.reserve(static_cast<size_t>(bench_runs));
            for (int iteration = 0; iteration < bench_runs; ++iteration)
            {
                if (cudaEventRecord(start_, stream_) != cudaSuccess)
                    throw std::runtime_error(
                        "CUDA sweep start-event recording failed");
                launch(m_, input_.get(), output_.get());
                if (cudaEventRecord(stop_, stream_) != cudaSuccess)
                    throw std::runtime_error(
                        "CUDA sweep stop-event recording failed");
                if (cudaEventSynchronize(stop_) != cudaSuccess)
                    throw std::runtime_error(
                        "CUDA sweep candidate event synchronization failed");
                float elapsed_ms = 0.0f;
                if (cudaEventElapsedTime(&elapsed_ms, start_, stop_) !=
                    cudaSuccess)
                    throw std::runtime_error(
                        "CUDA sweep elapsed-time query failed");
                times_us.push_back(static_cast<double>(elapsed_ms) * 1000.0);
            }
            if (const cudaError_t error = cudaGetLastError();
                error != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string("CUDA sweep candidate failed: ") +
                    cudaGetErrorString(error));
            }

            RunResult result;
            result.min_us = *std::min_element(times_us.begin(), times_us.end());
            result.mean_us = std::accumulate(times_us.begin(), times_us.end(), 0.0) /
                             static_cast<double>(times_us.size());
            result.native_family = "native_vnni_tc";
            return result;
        }

        size_t bitMismatches()
        {
            const float *device_output = reinterpret_cast<const float *>(
                output_->gpu_data_ptr());
            for (size_t index = 0; index < oracle_rows_.size(); ++index)
            {
                const size_t row = static_cast<size_t>(oracle_rows_[index]);
                if (cudaMemcpyAsync(
                        candidate_rows_.data() + index * static_cast<size_t>(n_),
                        device_output + row * static_cast<size_t>(n_),
                        static_cast<size_t>(n_) * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_) != cudaSuccess)
                {
                    throw std::runtime_error(
                        "CUDA sweep candidate row download failed");
                }
            }
            checkStream("candidate row download");

            size_t mismatches = 0;
            for (size_t index = 0; index < serial_oracle_.size(); ++index)
            {
                if (std::memcmp(
                        &serial_oracle_[index], &candidate_rows_[index],
                        sizeof(float)) != 0)
                {
                    ++mismatches;
                }
            }
            return mismatches;
        }

    private:
        static std::vector<int> serialOracleRows(int m)
        {
            std::set<int> rows;
            for (int row = 0; row < std::min(m, 4); ++row)
                rows.insert(row);
            for (const int anchor : {m / 4, m / 2, (3 * m) / 4})
            {
                if (anchor > 0)
                    rows.insert(anchor - 1);
                if (anchor < m)
                    rows.insert(anchor);
            }
            for (int row = std::max(0, m - 6); row < m; ++row)
                rows.insert(row);
            return {rows.begin(), rows.end()};
        }

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

        void buildSerialOracle()
        {
            cudaNativeVNNIPrefill_setBK256Mode(0);
            cudaNativeVNNIPrefill_setStreamKMode(0);
            cudaNativeVNNIPrefill_setForceTile(-1, 0);
            const float *host_input = host_input_->data();
            float *device_input = reinterpret_cast<float *>(
                serial_input_->gpu_data_ptr());
            const float *device_output = reinterpret_cast<const float *>(
                serial_output_->gpu_data_ptr());
            for (size_t index = 0; index < oracle_rows_.size(); ++index)
            {
                const size_t row = static_cast<size_t>(oracle_rows_[index]);
                if (cudaMemcpyAsync(
                        device_input,
                        host_input + row * static_cast<size_t>(k_),
                        static_cast<size_t>(k_) * sizeof(float),
                        cudaMemcpyHostToDevice, stream_) != cudaSuccess)
                    throw std::runtime_error(
                        "CUDA sweep serial-oracle row upload failed");
                launch(1, serial_input_.get(), serial_output_.get());
                if (cudaMemcpyAsync(
                        serial_oracle_.data() +
                            index * static_cast<size_t>(n_),
                        device_output,
                        static_cast<size_t>(n_) * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_) != cudaSuccess)
                    throw std::runtime_error(
                        "CUDA sweep serial-oracle row download failed");
            }
            checkStream("serial oracle construction");
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
        std::unique_ptr<FP32Tensor> serial_input_;
        std::unique_ptr<FP32Tensor> serial_output_;
        cudaStream_t stream_ = nullptr;
        cudaEvent_t start_ = nullptr;
        cudaEvent_t stop_ = nullptr;
        std::vector<int> oracle_rows_;
        std::vector<float> serial_oracle_;
        std::vector<float> candidate_rows_;
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
        {
            std::string fmt_lower = toLower(cfg.format_name);
            bool found = false;
            for (const auto &f : kFormats)
            {
                if (toLower(f.name) == fmt_lower)
                {
                    create_weights = f.create;
                    resolved_format_name = f.name;
                    found = true;
                    break;
                }
            }
            ASSERT_TRUE(found) << "Unknown format: " << cfg.format_name;
        }

        // ── Build flat task list ──
        std::vector<SweepTask> tasks;

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
                    if (strat == Strategy::Auto)
                    {
                        tasks.push_back({&shape, m, strat, -1, 0, "AUTO", 0});
                    }
                    else if (strat == Strategy::BK256 || strat == Strategy::BK256_SK)
                    {
                        std::vector<int> sk_for_bk256;
                        if (strat == Strategy::BK256)
                            sk_for_bk256 = {1};
                        else
                            sk_for_bk256 = cfg.bk256_split_k_values;

                        const int bk256_tiles = ((m + 127) / 128) * ((shape.n + 127) / 128);
                        for (int sk : sk_for_bk256)
                            tasks.push_back({&shape, m, strat, -2, sk, "BK256_128x128", bk256_tiles});
                    }
                    else
                    {
                        // BK64 strategies: iterate tiles
                        for (int tile_id : cfg.tile_ids)
                        {
                            const int tile_count = cudaNativeVNNIPrefill_getTileCount(tile_id, m, shape.n);
                            std::vector<int> sk_for_strat;
                            if (strat == Strategy::Standard)
                                sk_for_strat = cfg.split_k_values;
                            else
                                sk_for_strat = {1};

                            for (int sk : sk_for_strat)
                                tasks.push_back({&shape, m, strat, tile_id, sk,
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
                         "format,codebook,shape,m,n,k,tile,tile_id,strategy,split_k,tiles,"
                         "min_us,mean_us,tops,pct_peak,gpu,bit_mismatches,correctness_pass\n");
            std::fflush(csv_fp);
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
                         "[TileSweep][gpu=%d] %s M=%d %s %s sk=%d tiles=%d\n",
                         gpu_id, shape.name.c_str(), task.m,
                         task.tile_name.c_str(), strategyName(task.strat),
                         task.split_k, task.tiles);

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
                    requirements);
                prepared_m = task.m;
            }

            configureSweepTask(task);
            const RunResult rr = execution->measure(
                cfg.warmup_runs, cfg.bench_runs);
            const size_t bit_mismatches = execution->bitMismatches();
            const auto metrics = computeGemmThroughputMetrics(
                task.m, shape.n, shape.k, rr.min_us, peak_tops);
            const uint8_t codebook_id =
                requireNativeVnniInfo(weights.get(), resolved_format_name)
                    .codebook_id;

            SweepRow &row = rows[task_idx];
            row.format_name = resolved_format_name;
            row.codebook_id = codebook_id;
            row.shape_name = shape.name;
            row.m = task.m;
            row.n = shape.n;
            row.k = shape.k;
            row.tile_name = task.tile_name;
            row.tile_id = task.tile_id;
            row.tiles = task.tiles;
            row.strategy = strategyName(task.strat);
            row.split_k = task.split_k;
            row.min_us = rr.min_us;
            row.mean_us = rr.mean_us;
            row.tops = metrics.achieved_tops;
            row.pct_peak = metrics.pct_tc_peak;
            row.gpu_id = gpu_id;
            row.bit_mismatches = bit_mismatches;
            row.correctness_pass = bit_mismatches == 0;
            row_valid[task_idx] = true;

            if (csv_fp)
            {
                std::fprintf(csv_fp,
                             "%s,%u,%s,%d,%d,%d,%s,%d,%s,%d,%d,%.3f,%.3f,%.4f,%.2f,%d,%zu,%d\n",
                             row.format_name.c_str(),
                             static_cast<unsigned>(row.codebook_id),
                             row.shape_name.c_str(), row.m, row.n, row.k,
                             row.tile_name.c_str(), row.tile_id,
                             row.strategy.c_str(), row.split_k, row.tiles,
                             row.min_us, row.mean_us, row.tops, row.pct_peak,
                             row.gpu_id, row.bit_mismatches,
                             row.correctness_pass ? 1 : 0);
                std::fflush(csv_fp);
            }
        }

        execution.reset();
        prepared.reset();

        // Restore original modes
        cudaNativeVNNIPrefill_setBK256Mode(0);
        cudaNativeVNNIPrefill_setStreamKMode(0);
        cudaNativeVNNIPrefill_setForceTile(-1, 0);
        cudaNativeVNNIPrefill_freeStreamKFixup();

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
                  << r.tile_name << r.strategy << r.split_k << r.tiles
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
                             r.strategy.c_str(), r.split_k, r.min_us);
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
