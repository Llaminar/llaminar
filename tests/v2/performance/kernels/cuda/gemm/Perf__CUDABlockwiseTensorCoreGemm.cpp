/**
 * @file Perf__CUDABlockwiseTensorCoreGemm.cpp
 * @brief Correctness and performance sweep for the CUDA blockwise tensor-core GEMM/GEMV scaffold.
 */

#include <gtest/gtest.h>

#ifdef HAVE_CUDA

#include <cuda_runtime.h>

#include "kernels/cuda/CUDAQuantisedGemmKernel.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/coherence/GpuCoherence.h"
#include "backends/DeviceId.h"
#include "../../../../utils/TestTensorFactory.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::cuda;
using namespace llaminar2::test;

extern "C"
{
    void cudaTCGemm_setTuningOverrides(int force_small_m, int force_wide_n, int force_balanced, int unused0, int unused1);
}

namespace
{
    constexpr int kDefaultWarmupRuns = 3;
    constexpr int kDefaultBenchRuns = 10;
    constexpr float kCosineGate = 0.9990f;

    struct FormatSpec
    {
        std::string name;
        std::function<std::unique_ptr<TensorBase>(size_t, size_t)> create;
    };

    const std::vector<FormatSpec> kFormats = {
        {"Q4_0", [](size_t n, size_t k) { return TestTensorFactory::createQ4_0Random({n, k}); }},
        {"IQ4_NL", [](size_t n, size_t k) { return TestTensorFactory::createIQ4_NLRandom({n, k}); }},
        {"Q4_1", [](size_t n, size_t k) { return TestTensorFactory::createQ4_1Random({n, k}); }},
        {"Q5_0", [](size_t n, size_t k) { return TestTensorFactory::createQ5_0Random({n, k}); }},
        {"Q5_1", [](size_t n, size_t k) { return TestTensorFactory::createQ5_1Random({n, k}); }},
        {"Q6_K", [](size_t n, size_t k) { return TestTensorFactory::createQ6_KRandom({n, k}); }},
        {"Q3_K", [](size_t n, size_t k) { return TestTensorFactory::createQ3_KRandom({n, k}); }},
        {"Q2_K", [](size_t n, size_t k) { return TestTensorFactory::createQ2_KRandom({n, k}); }},
        {"IQ3_S", [](size_t n, size_t k) { return TestTensorFactory::createIQ3_SRandom({n, k}); }},
        {"IQ3_XXS", [](size_t n, size_t k) { return TestTensorFactory::createIQ3_XXSRandom({n, k}); }},
        {"IQ2_S", [](size_t n, size_t k) { return TestTensorFactory::createIQ2_SRandom({n, k}); }},
        {"IQ2_XS", [](size_t n, size_t k) { return TestTensorFactory::createIQ2_XSRandom({n, k}); }},
        {"IQ2_XXS", [](size_t n, size_t k) { return TestTensorFactory::createIQ2_XXSRandom({n, k}); }},
        {"IQ1_S", [](size_t n, size_t k) { return TestTensorFactory::createIQ1_SRandom({n, k}); }},
        {"IQ1_M", [](size_t n, size_t k) { return TestTensorFactory::createIQ1_MRandom({n, k}); }},
    };

    struct Shape
    {
        std::string name;
        int n;
        int k;
    };

    const std::vector<Shape> kQwenShapes = {
        {"0.5B_Attn", 896, 896},
        {"0.5B_FFN_Up", 4864, 896},
        {"0.5B_FFN_Down", 896, 4864},
        {"0.5B_LM_Head", 151936, 896},
        {"3B_Attn", 2048, 2048},
        {"3B_FFN_Up", 11008, 2048},
        {"3B_FFN_Down", 2048, 11008},
        {"3B_LM_Head", 151936, 2048},
        {"7B_Attn", 3584, 3584},
        {"7B_FFN_Up", 18944, 3584},
        {"7B_FFN_Down", 3584, 18944},
        {"7B_LM_Head", 152064, 3584},
    };

    const std::vector<int> kPrefillMValues = {32, 64, 128};

    struct SweepConfig
    {
        int warmup_runs = kDefaultWarmupRuns;
        int bench_runs = kDefaultBenchRuns;
        int correctness_prefill_m = 128;
        std::vector<int> performance_prefill_m = kPrefillMValues;
        std::set<std::string> format_filters;
        std::set<std::string> shape_filters;
        int max_cases = std::numeric_limits<int>::max();
        bool smoke = false;
        std::string gemm_dispatch = "auto";
    };

    std::string toLower(std::string value)
    {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        return value;
    }

    std::string trim(std::string value)
    {
        const auto begin = value.find_first_not_of(" \t\n\r");
        if (begin == std::string::npos)
            return {};
        const auto end = value.find_last_not_of(" \t\n\r");
        return value.substr(begin, end - begin + 1);
    }

    std::optional<int> getEnvInt(const char *name)
    {
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return std::nullopt;
        return std::atoi(raw);
    }

    bool getEnvFlag(const char *name)
    {
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return false;
        return std::atoi(raw) != 0;
    }

    std::set<std::string> getEnvCsvSet(const char *name)
    {
        std::set<std::string> values;
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return values;

        std::stringstream ss(raw);
        std::string item;
        while (std::getline(ss, item, ','))
        {
            item = toLower(trim(item));
            if (!item.empty())
            {
                values.insert(item);
            }
        }
        return values;
    }

    std::vector<int> getEnvCsvInts(const char *name)
    {
        std::vector<int> values;
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return values;

        std::stringstream ss(raw);
        std::string item;
        while (std::getline(ss, item, ','))
        {
            item = trim(item);
            if (!item.empty())
            {
                values.push_back(std::atoi(item.c_str()));
            }
        }
        return values;
    }

    std::string getEnvString(const char *name)
    {
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return {};
        return toLower(trim(raw));
    }

    SweepConfig loadSweepConfig()
    {
        SweepConfig cfg;
        cfg.smoke = getEnvFlag("LLAMINAR_CUDA_TC_SMOKE");

        if (cfg.smoke)
        {
            cfg.warmup_runs = 1;
            cfg.bench_runs = 2;
            cfg.correctness_prefill_m = 32;
            cfg.performance_prefill_m = {32};
            cfg.format_filters = {"q4_0", "iq4_nl"};
            cfg.shape_filters = {"0.5b_attn", "0.5b_ffn_up", "0.5b_lm_head"};
            cfg.max_cases = 4;
        }

        if (const auto value = getEnvInt("LLAMINAR_CUDA_TC_WARMUP_RUNS"))
            cfg.warmup_runs = std::max(0, *value);
        if (const auto value = getEnvInt("LLAMINAR_CUDA_TC_BENCH_RUNS"))
            cfg.bench_runs = std::max(1, *value);
        if (const auto value = getEnvInt("LLAMINAR_CUDA_TC_CORRECTNESS_PREFILL_M"))
            cfg.correctness_prefill_m = std::max(1, *value);
        if (const auto value = getEnvInt("LLAMINAR_CUDA_TC_MAX_CASES"))
            cfg.max_cases = std::max(1, *value);

        const auto format_filters = getEnvCsvSet("LLAMINAR_CUDA_TC_FORMATS");
        if (!format_filters.empty())
            cfg.format_filters = format_filters;

        const auto shape_filters = getEnvCsvSet("LLAMINAR_CUDA_TC_SHAPES");
        if (!shape_filters.empty())
            cfg.shape_filters = shape_filters;

        const auto prefill_m = getEnvCsvInts("LLAMINAR_CUDA_TC_PREFILL_M");
        if (!prefill_m.empty())
            cfg.performance_prefill_m = prefill_m;

        const auto gemm_dispatch = getEnvString("LLAMINAR_CUDA_TC_GEMM_DISPATCH");
        if (!gemm_dispatch.empty())
            cfg.gemm_dispatch = gemm_dispatch;

        return cfg;
    }

    void applySpecializedDispatchOverride(const SweepConfig &cfg, int m)
    {
        const bool is_gemm = (m > 1);
        if (!is_gemm)
        {
            cudaTCGemm_setTuningOverrides(0, 0, 0, 0, 0);
            return;
        }

        if (cfg.gemm_dispatch == "small_m")
        {
            cudaTCGemm_setTuningOverrides(1, 0, 0, 0, 0);
        }
        else if (cfg.gemm_dispatch == "wide_n")
        {
            cudaTCGemm_setTuningOverrides(0, 1, 0, 0, 0);
        }
        else if (cfg.gemm_dispatch == "balanced")
        {
            cudaTCGemm_setTuningOverrides(0, 0, 1, 0, 0);
        }
        else
        {
            cudaTCGemm_setTuningOverrides(0, 0, 0, 0, 0);
        }
    }

    bool shouldRunName(const std::set<std::string> &filters, const std::string &name)
    {
        return filters.empty() || filters.count(toLower(name)) > 0;
    }

    struct RunResult
    {
        std::vector<float> output;
        double min_us = 0.0;
        double mean_us = 0.0;
    };

    float cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b)
    {
        double dot = 0.0;
        double norm_a = 0.0;
        double norm_b = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        if (norm_a == 0.0 || norm_b == 0.0)
            return 0.0f;
        return static_cast<float>(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
    }

    class CUDABlockwiseTensorCorePerf : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            int device_count = 0;
            const cudaError_t err = cudaGetDeviceCount(&device_count);
            if (err != cudaSuccess || device_count == 0)
            {
                GTEST_SKIP() << "No CUDA devices available";
            }

            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
            device_ = DeviceId::cuda(0);
        }

        RunResult runKernel(
            TensorBase *weights,
            int m,
            int n,
            int k,
            CUDABlockwiseExecutionBackend backend,
            const SweepConfig &cfg)
        {
            CUDAQuantisedGemmKernel::setBlockwiseExecutionBackend(backend);
            if (backend == CUDABlockwiseExecutionBackend::SpecializedBlockwise)
                applySpecializedDispatchOverride(cfg, m);
            else
                cudaTCGemm_setTuningOverrides(0, 0, 0, 0, 0);

            auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(m), static_cast<size_t>(k)}, -0.25f, 0.25f, 7);
            auto output = TestTensorFactory::createFP32Zeros({static_cast<size_t>(m), static_cast<size_t>(n)});

            CUDAQuantisedGemmKernel kernel(weights, 0);
            DeviceWorkspaceManager workspace(device_, 512ull * 1024ull * 1024ull);
            const auto reqs = kernel.getWorkspaceRequirements(m, n, k);
            if (!workspace.allocate(reqs))
            {
                throw std::runtime_error("Failed to allocate CUDA GEMM workspace");
            }
            kernel.bindWorkspace(&workspace);

            std::vector<double> times_us;
            times_us.reserve(static_cast<size_t>(cfg.bench_runs));

            const bool ok = with_gpu_coherence(
                device_,
                {input.get()},
                {output.get()},
                [&]() {
                    for (int i = 0; i < cfg.warmup_runs; ++i)
                    {
                        if (!kernel.multiply_tensor(input.get(), output.get(), m, n, k, true, 1.0f, 0.0f, nullptr, nullptr, -1, &workspace))
                        {
                            return false;
                        }
                    }

                    for (int i = 0; i < cfg.bench_runs; ++i)
                    {
                        cudaEvent_t start = nullptr;
                        cudaEvent_t stop = nullptr;
                        if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess)
                        {
                            return false;
                        }

                        cudaEventRecord(start);
                        const bool run_ok = kernel.multiply_tensor(input.get(), output.get(), m, n, k, true, 1.0f, 0.0f, nullptr, nullptr, -1, &workspace);
                        cudaEventRecord(stop);
                        cudaEventSynchronize(stop);

                        float elapsed_ms = 0.0f;
                        cudaEventElapsedTime(&elapsed_ms, start, stop);
                        cudaEventDestroy(start);
                        cudaEventDestroy(stop);

                        if (!run_ok)
                        {
                            return false;
                        }

                        times_us.push_back(static_cast<double>(elapsed_ms) * 1000.0);
                    }

                    return true;
                });
            if (!ok)
            {
                throw std::runtime_error("CUDA blockwise GEMM run failed");
            }

            const float *host = output->data();
            if (!host)
            {
                throw std::runtime_error("CUDA blockwise GEMM output sync failed");
            }

            RunResult result;
            result.output.assign(host, host + static_cast<size_t>(m) * n);
            result.min_us = *std::min_element(times_us.begin(), times_us.end());
            result.mean_us = std::accumulate(times_us.begin(), times_us.end(), 0.0) / static_cast<double>(times_us.size());
            return result;
        }

        DeviceId device_ = DeviceId::cpu();
    };

    TEST_F(CUDABlockwiseTensorCorePerf, Correctness_AllFormats_KeyShapes)
    {
        const SweepConfig cfg = loadSweepConfig();
        int executed_cases = 0;

        for (const auto &format : kFormats)
        {
            if (!shouldRunName(cfg.format_filters, format.name))
            {
                continue;
            }

            for (const auto &shape : kQwenShapes)
            {
                if (!shouldRunName(cfg.shape_filters, shape.name))
                {
                    continue;
                }
                if ((shape.k % 32) != 0)
                {
                    continue;
                }
                if (executed_cases >= cfg.max_cases)
                {
                    return;
                }

                std::fprintf(stderr,
                             "[CUDABlockwiseTC][Correctness] format=%s shape=%s decode_m=1 prefill_m=%d warmup=%d bench=%d gemm_dispatch=%s\n",
                             format.name.c_str(), shape.name.c_str(), cfg.correctness_prefill_m, cfg.warmup_runs, cfg.bench_runs, cfg.gemm_dispatch.c_str());

                auto weights_decode = format.create(static_cast<size_t>(shape.n), static_cast<size_t>(shape.k));
                const RunResult legacy_decode = runKernel(weights_decode.get(), 1, shape.n, shape.k, CUDABlockwiseExecutionBackend::LegacyDP4A, cfg);
                const RunResult tc_decode = runKernel(weights_decode.get(), 1, shape.n, shape.k, CUDABlockwiseExecutionBackend::TensorCoreScaffold, cfg);
                EXPECT_GE(cosineSimilarity(legacy_decode.output, tc_decode.output), kCosineGate) << format.name << " decode " << shape.name;

                const RunResult tck_decode = runKernel(weights_decode.get(), 1, shape.n, shape.k, CUDABlockwiseExecutionBackend::SpecializedBlockwise, cfg);
                EXPECT_GE(cosineSimilarity(legacy_decode.output, tck_decode.output), kCosineGate) << format.name << " decode(kernels) " << shape.name;

                auto weights_prefill = format.create(static_cast<size_t>(shape.n), static_cast<size_t>(shape.k));
                const RunResult legacy_prefill = runKernel(weights_prefill.get(), cfg.correctness_prefill_m, shape.n, shape.k, CUDABlockwiseExecutionBackend::LegacyDP4A, cfg);
                const RunResult tc_prefill = runKernel(weights_prefill.get(), cfg.correctness_prefill_m, shape.n, shape.k, CUDABlockwiseExecutionBackend::TensorCoreScaffold, cfg);
                EXPECT_GE(cosineSimilarity(legacy_prefill.output, tc_prefill.output), kCosineGate) << format.name << " prefill " << shape.name;

                const RunResult tck_prefill = runKernel(weights_prefill.get(), cfg.correctness_prefill_m, shape.n, shape.k, CUDABlockwiseExecutionBackend::SpecializedBlockwise, cfg);
                EXPECT_GE(cosineSimilarity(legacy_prefill.output, tck_prefill.output), kCosineGate) << format.name << " prefill(kernels) " << shape.name;
                ++executed_cases;
            }
        }

        ASSERT_GT(executed_cases, 0) << "No correctness cases selected. Check LLAMINAR_CUDA_TC_FORMATS / LLAMINAR_CUDA_TC_SHAPES.";
    }

    TEST_F(CUDABlockwiseTensorCorePerf, Performance_AllFormats_AllShapes)
    {
        const SweepConfig cfg = loadSweepConfig();
        int executed_cases = 0;

        for (const auto &format : kFormats)
        {
            if (!shouldRunName(cfg.format_filters, format.name))
            {
                continue;
            }

            for (const auto &shape : kQwenShapes)
            {
                if (!shouldRunName(cfg.shape_filters, shape.name))
                {
                    continue;
                }
                if ((shape.k % 32) != 0)
                {
                    continue;
                }
                if (executed_cases >= cfg.max_cases)
                {
                    return;
                }

                auto weights_gemv = format.create(static_cast<size_t>(shape.n), static_cast<size_t>(shape.k));
                const RunResult legacy_gemv = runKernel(weights_gemv.get(), 1, shape.n, shape.k, CUDABlockwiseExecutionBackend::LegacyDP4A, cfg);
                const RunResult tc_gemv = runKernel(weights_gemv.get(), 1, shape.n, shape.k, CUDABlockwiseExecutionBackend::TensorCoreScaffold, cfg);
                const RunResult tck_gemv = runKernel(weights_gemv.get(), 1, shape.n, shape.k, CUDABlockwiseExecutionBackend::SpecializedBlockwise, cfg);

                std::fprintf(stderr,
                             "[CUDABlockwiseTC][GEMV] format=%s shape=%s M=1 N=%d K=%d warmup=%d bench=%d "
                             "legacy_min_us=%.3f tc_min_us=%.3f tck_min_us=%.3f "
                             "scaffold_speedup=%.3fx kernels_speedup=%.3fx\n",
                             format.name.c_str(), shape.name.c_str(), shape.n, shape.k,
                             cfg.warmup_runs, cfg.bench_runs,
                             legacy_gemv.min_us, tc_gemv.min_us, tck_gemv.min_us,
                             legacy_gemv.min_us / tc_gemv.min_us,
                             legacy_gemv.min_us / tck_gemv.min_us);

                for (int m : cfg.performance_prefill_m)
                {
                    auto weights_gemm = format.create(static_cast<size_t>(shape.n), static_cast<size_t>(shape.k));
                    const RunResult legacy_gemm = runKernel(weights_gemm.get(), m, shape.n, shape.k, CUDABlockwiseExecutionBackend::LegacyDP4A, cfg);
                    const RunResult tc_gemm = runKernel(weights_gemm.get(), m, shape.n, shape.k, CUDABlockwiseExecutionBackend::TensorCoreScaffold, cfg);
                    const RunResult tck_gemm = runKernel(weights_gemm.get(), m, shape.n, shape.k, CUDABlockwiseExecutionBackend::SpecializedBlockwise, cfg);

                    std::fprintf(stderr,
                                 "[CUDABlockwiseTC][GEMM] format=%s shape=%s M=%d N=%d K=%d warmup=%d bench=%d "
                                 "gemm_dispatch=%s "
                                 "legacy_min_us=%.3f tc_min_us=%.3f tck_min_us=%.3f "
                                 "scaffold_speedup=%.3fx kernels_speedup=%.3fx\n",
                                 format.name.c_str(), shape.name.c_str(), m, shape.n, shape.k,
                                 cfg.warmup_runs, cfg.bench_runs, cfg.gemm_dispatch.c_str(),
                                 legacy_gemm.min_us, tc_gemm.min_us, tck_gemm.min_us,
                                 legacy_gemm.min_us / tc_gemm.min_us,
                                 legacy_gemm.min_us / tck_gemm.min_us);
                }

                ++executed_cases;
            }
        }

        ASSERT_GT(executed_cases, 0) << "No performance cases selected. Check LLAMINAR_CUDA_TC_FORMATS / LLAMINAR_CUDA_TC_SHAPES.";
    }
}

#endif