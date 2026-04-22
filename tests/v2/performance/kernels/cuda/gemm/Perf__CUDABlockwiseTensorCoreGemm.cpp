/**
 * @file Perf__CUDABlockwiseTensorCoreGemm.cpp
 * @brief Correctness and performance sweep for the CUDA blockwise tensor-core GEMM/GEMV scaffold.
 */

#include <gtest/gtest.h>

#ifdef HAVE_CUDA

#include <cuda_runtime.h>

#include "kernels/cuda/gemm/CUDAQuantisedGemmKernel.h"
#include "kernels/KernelFactory.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/coherence/GpuCoherence.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "backends/DeviceId.h"
#include "../../../../utils/TestTensorFactory.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
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
using llaminar::v2::kernels::KernelFactory;

// TC tuning extern removed — NativeVNNI is the sole CUDA GEMM path

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
        {"Q4_0", [](size_t n, size_t k)
         { return TestTensorFactory::createQ4_0Random({n, k}); }},
        {"IQ4_NL", [](size_t n, size_t k)
         { return TestTensorFactory::createIQ4_NLRandom({n, k}); }},
        {"Q4_1", [](size_t n, size_t k)
         { return TestTensorFactory::createQ4_1Random({n, k}); }},
        {"Q5_0", [](size_t n, size_t k)
         { return TestTensorFactory::createQ5_0Random({n, k}); }},
        {"Q5_1", [](size_t n, size_t k)
         { return TestTensorFactory::createQ5_1Random({n, k}); }},
        {"Q6_K", [](size_t n, size_t k)
         { return TestTensorFactory::createQ6_KRandom({n, k}); }},
        {"Q3_K", [](size_t n, size_t k)
         { return TestTensorFactory::createQ3_KRandom({n, k}); }},
        {"Q2_K", [](size_t n, size_t k)
         { return TestTensorFactory::createQ2_KRandom({n, k}); }},
        {"IQ3_S", [](size_t n, size_t k)
         { return TestTensorFactory::createIQ3_SRandom({n, k}); }},
        {"IQ3_XXS", [](size_t n, size_t k)
         { return TestTensorFactory::createIQ3_XXSRandom({n, k}); }},
        {"IQ2_S", [](size_t n, size_t k)
         { return TestTensorFactory::createIQ2_SRandom({n, k}); }},
        {"IQ2_XS", [](size_t n, size_t k)
         { return TestTensorFactory::createIQ2_XSRandom({n, k}); }},
        {"IQ2_XXS", [](size_t n, size_t k)
         { return TestTensorFactory::createIQ2_XXSRandom({n, k}); }},
        {"IQ1_S", [](size_t n, size_t k)
         { return TestTensorFactory::createIQ1_SRandom({n, k}); }},
        {"IQ1_M", [](size_t n, size_t k)
         { return TestTensorFactory::createIQ1_MRandom({n, k}); }},
        {"Q8_0", [](size_t n, size_t k)
         { return TestTensorFactory::createQ8_0Random({n, k}); }},
    };

    struct Shape
    {
        std::string name;
        int n;
        int k;
    };

    const std::vector<Shape> kQwenShapes = {
        // ================================================================
        // Qwen2.5 0.5B  (hidden=896, intermediate=4864, vocab=151936)
        // ================================================================
        {"0.5B_Attn", 896, 896},
        {"0.5B_FFN_Up", 4864, 896},
        {"0.5B_FFN_Down", 896, 4864},
        {"0.5B_LM_Head", 151936, 896},
        // TP=2: ColPar N/2, RowPar K/2
        {"0.5B_TP2_Attn_QKV", 448, 896}, // column-parallel
        {"0.5B_TP2_Attn_Wo", 896, 448},  // row-parallel
        {"0.5B_TP2_FFN_Up", 2432, 896},
        {"0.5B_TP2_FFN_Down", 896, 2432},
        {"0.5B_TP2_LM_Head", 75968, 896},
        // TP=4
        {"0.5B_TP4_Attn_QKV", 224, 896},
        {"0.5B_TP4_Attn_Wo", 896, 224},
        {"0.5B_TP4_FFN_Up", 1216, 896},
        {"0.5B_TP4_FFN_Down", 896, 1216},
        {"0.5B_TP4_LM_Head", 37984, 896},

        // ================================================================
        // Qwen2.5 3B  (hidden=2048, intermediate=11008, vocab=151936)
        // ================================================================
        {"3B_Attn", 2048, 2048},
        {"3B_FFN_Up", 11008, 2048},
        {"3B_FFN_Down", 2048, 11008},
        {"3B_LM_Head", 151936, 2048},
        // TP=2
        {"3B_TP2_Attn_QKV", 1024, 2048},
        {"3B_TP2_Attn_Wo", 2048, 1024},
        {"3B_TP2_FFN_Up", 5504, 2048},
        {"3B_TP2_FFN_Down", 2048, 5504},
        {"3B_TP2_LM_Head", 75968, 2048},
        // TP=4
        {"3B_TP4_Attn_QKV", 512, 2048},
        {"3B_TP4_Attn_Wo", 2048, 512},
        {"3B_TP4_FFN_Up", 2752, 2048},
        {"3B_TP4_FFN_Down", 2048, 2752},
        {"3B_TP4_LM_Head", 37984, 2048},

        // ================================================================
        // Qwen2.5 7B  (hidden=3584, intermediate=18944, vocab=152064)
        // ================================================================
        {"7B_Attn", 3584, 3584},
        {"7B_FFN_Up", 18944, 3584},
        {"7B_FFN_Down", 3584, 18944},
        {"7B_LM_Head", 152064, 3584},
        // TP=2
        {"7B_TP2_Attn_QKV", 1792, 3584},
        {"7B_TP2_Attn_Wo", 3584, 1792},
        {"7B_TP2_FFN_Up", 9472, 3584},
        {"7B_TP2_FFN_Down", 3584, 9472},
        {"7B_TP2_LM_Head", 76032, 3584},
        // TP=4
        {"7B_TP4_Attn_QKV", 896, 3584},
        {"7B_TP4_Attn_Wo", 3584, 896},
        {"7B_TP4_FFN_Up", 4736, 3584},
        {"7B_TP4_FFN_Down", 3584, 4736},
        {"7B_TP4_LM_Head", 38016, 3584},

        // ================================================================
        // Qwen2.5 14B  (hidden=5120, intermediate=13824, vocab=152064)
        // ================================================================
        {"14B_Attn", 5120, 5120},
        {"14B_FFN_Up", 13824, 5120},
        {"14B_FFN_Down", 5120, 13824},
        {"14B_LM_Head", 152064, 5120},
        // TP=2
        {"14B_TP2_Attn_QKV", 2560, 5120},
        {"14B_TP2_Attn_Wo", 5120, 2560},
        {"14B_TP2_FFN_Up", 6912, 5120},
        {"14B_TP2_FFN_Down", 5120, 6912},
        {"14B_TP2_LM_Head", 76032, 5120},
        // TP=4
        {"14B_TP4_Attn_QKV", 1280, 5120},
        {"14B_TP4_Attn_Wo", 5120, 1280},
        {"14B_TP4_FFN_Up", 3456, 5120},
        {"14B_TP4_FFN_Down", 5120, 3456},
        {"14B_TP4_LM_Head", 38016, 5120},
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
        bool gemv_only = false;
        bool tuned_gemv_enabled = true;
        bool native_vnni_enabled = true;
        std::string gemm_dispatch = "auto";
    };

    // KernelModeGuard is a no-op — NativeVNNI is always enabled, CUTLASS removed.
    struct KernelModeGuard
    {
        KernelModeGuard(bool /*native_vnni_enabled*/, bool /*force_cutlass_fallback*/) {}
        ~KernelModeGuard() = default;
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

        cfg.gemv_only = getEnvFlag("LLAMINAR_CUDA_TC_GEMV_ONLY");
        if (const auto value = getEnvInt("LLAMINAR_CUDA_TC_TUNED_GEMV"))
            cfg.tuned_gemv_enabled = (*value != 0);
        if (const auto value = getEnvInt("LLAMINAR_CUDA_TC_NATIVE_VNNI"))
            cfg.native_vnni_enabled = (*value != 0);

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

    // TC tuning overrides removed — NativeVNNI handles dispatch internally.
    void applyGemmTuningOverride(const SweepConfig & /*cfg*/, int /*m*/)
    {
    }

    bool shouldRunName(const std::set<std::string> &filters, const std::string &name)
    {
        return filters.empty() || filters.count(toLower(name)) > 0;
    }

    int parseTPDegree(const std::string &name)
    {
        if (name.find("_TP2_") != std::string::npos)
            return 2;
        if (name.find("_TP4_") != std::string::npos)
            return 4;
        return 1;
    }

    std::string tpBaselineKey(const std::string &name)
    {
        std::string model;
        std::string layer;

        const auto tp_pos = name.find("_TP");
        if (tp_pos != std::string::npos)
        {
            model = name.substr(0, tp_pos);
            const auto after_tp = name.find('_', tp_pos + 3);
            layer = (after_tp != std::string::npos) ? name.substr(after_tp + 1) : "";
        }
        else
        {
            const auto first = name.find('_');
            model = (first != std::string::npos) ? name.substr(0, first) : name;
            layer = (first != std::string::npos) ? name.substr(first + 1) : "";
        }

        // Map TP-sharded layer names back to their TP=1 base
        if (layer == "Attn_QKV" || layer == "Attn_Wo")
            layer = "Attn";

        return model + "_" + layer;
    }

    // Theoretical peak HBM bandwidth from device properties.
    // Uses: 2 * memoryClockRate (DDR) * busWidth_bytes.
    // Example: RTX 3090 → 2 * 9.751 GHz * 48 bytes = 936 GB/s.
    double measurePeakBandwidth()
    {
        int device = 0;
        cudaGetDevice(&device);
        int clockKHz = 0, busWidthBits = 0;
        cudaDeviceGetAttribute(&clockKHz, cudaDevAttrMemoryClockRate, device);
        cudaDeviceGetAttribute(&busWidthBits, cudaDevAttrGlobalMemoryBusWidth, device);
        if (clockKHz <= 0 || busWidthBits <= 0)
            return 0.0;
        // memoryClockRate is in kHz, busWidth in bits; DDR factor of 2
        return 2.0 * static_cast<double>(clockKHz) * 1e3 * static_cast<double>(busWidthBits / 8) / 1e9;
    }

    struct RunResult
    {
        std::vector<float> output;
        double min_us = 0.0;
        double mean_us = 0.0;
        std::string backend_family;
        int split_k = 1;
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

            peak_bw_GB_s_ = measurePeakBandwidth();
            std::fprintf(stderr, "[CUDABlockwiseTC] Measured peak HBM bandwidth: %.1f GB/s\n", peak_bw_GB_s_);
        }

        RunResult runKernel(
            TensorBase *weights,
            int m,
            int n,
            int k,
            const SweepConfig &cfg)
        {
            applyGemmTuningOverride(cfg, m);
            KernelModeGuard mode_guard(cfg.native_vnni_enabled, false);

            // Create kernel via KernelFactory (same path as parity tests)
            EXPECT_TRUE(weights->ensureOnDevice(device_));
            auto kernel = KernelFactory::createGemm(weights, DeviceType::CUDA);
            if (!kernel)
            {
                throw std::runtime_error("KernelFactory::createGemm returned null");
            }

            // Set up workspace via IWorkspaceConsumer interface
            auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel.get());
            std::unique_ptr<DeviceWorkspaceManager> workspace;
            if (ws_consumer)
            {
                workspace = std::make_unique<DeviceWorkspaceManager>(device_, 512ull * 1024ull * 1024ull);
                const auto reqs = ws_consumer->getWorkspaceRequirements(m, n, k);
                if (!workspace->allocate(reqs))
                {
                    throw std::runtime_error("Failed to allocate CUDA GEMM workspace");
                }
                ws_consumer->bindWorkspace(workspace.get());
            }

            // Create tensor wrappers and upload to GPU
            auto h_input = TestTensorFactory::createFP32Random({static_cast<size_t>(m), static_cast<size_t>(k)}, -0.25f, 0.25f, 7);

            auto A_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(k)});
            std::memcpy(A_tensor->mutable_data(), h_input->data(), static_cast<size_t>(m) * k * sizeof(float));
            auto C_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(n)});

            if (!A_tensor->ensureOnDevice(device_))
                throw std::runtime_error("ensureOnDevice A failed");
            if (!C_tensor->ensureOnDevice(device_))
                throw std::runtime_error("ensureOnDevice C failed");

            std::vector<double> times_us;
            times_us.reserve(static_cast<size_t>(cfg.bench_runs));

            // Warmup
            for (int i = 0; i < cfg.warmup_runs; ++i)
            {
                if (!kernel->multiply_tensor(A_tensor.get(), C_tensor.get(), m, n, k))
                {
                    throw std::runtime_error("CUDA blockwise GEMM warmup failed");
                }
            }

            // Benchmark
            for (int i = 0; i < cfg.bench_runs; ++i)
            {
                cudaEvent_t start = nullptr;
                cudaEvent_t stop = nullptr;
                if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess)
                {
                    throw std::runtime_error("cudaEventCreate failed");
                }

                cudaEventRecord(start);
                const bool run_ok = kernel->multiply_tensor(A_tensor.get(), C_tensor.get(), m, n, k);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                float elapsed_ms = 0.0f;
                cudaEventElapsedTime(&elapsed_ms, start, stop);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);

                if (!run_ok)
                {
                    throw std::runtime_error("CUDA blockwise GEMM bench run failed");
                }

                times_us.push_back(static_cast<double>(elapsed_ms) * 1000.0);
            }

            // Download result
            RunResult result;
            result.output.resize(static_cast<size_t>(m) * n);
            C_tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
            std::memcpy(result.output.data(), C_tensor->data(), static_cast<size_t>(m) * n * sizeof(float));

            if (ws_consumer)
            {
                ws_consumer->unbindWorkspace();
            }

            result.min_us = *std::min_element(times_us.begin(), times_us.end());
            result.mean_us = std::accumulate(times_us.begin(), times_us.end(), 0.0) / static_cast<double>(times_us.size());
            result.backend_family = "native_vnni_tc";
            return result;
        }

        DeviceId device_ = DeviceId::cpu();
        double peak_bw_GB_s_ = 0;
    };

    TEST_F(CUDABlockwiseTensorCorePerf, Correctness_AllFormats_KeyShapes)
    {
        const SweepConfig cfg = loadSweepConfig();

        // Correctness only needs one run per backend (no warmup, no timing)
        SweepConfig correctness_cfg = cfg;
        correctness_cfg.warmup_runs = 0;
        correctness_cfg.bench_runs = 1;

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
                             "[CUDABlockwiseTC][Correctness] format=%s shape=%s decode_m=1 prefill_m=%d\n",
                             format.name.c_str(), shape.name.c_str(), correctness_cfg.correctness_prefill_m);

                // Run decode (M=1) and prefill (M>1) —verify both produce valid output
                auto weights_decode = format.create(static_cast<size_t>(shape.n), static_cast<size_t>(shape.k));
                const RunResult decode_result = runKernel(weights_decode.get(), 1, shape.n, shape.k, correctness_cfg);

                auto weights_prefill = format.create(static_cast<size_t>(shape.n), static_cast<size_t>(shape.k));
                const RunResult prefill_result = runKernel(weights_prefill.get(), correctness_cfg.correctness_prefill_m, shape.n, shape.k, correctness_cfg);

                // Verify decode and prefill first rows are consistent
                const int compare_n = shape.n;
                std::vector<float> decode_first_row(decode_result.output.begin(), decode_result.output.begin() + compare_n);
                std::vector<float> prefill_first_row(prefill_result.output.begin(), prefill_result.output.begin() + compare_n);
                EXPECT_GE(cosineSimilarity(decode_first_row, prefill_first_row), kCosineGate) << format.name << " decode_vs_prefill " << shape.name;
                ++executed_cases;
            }
        }

        ASSERT_GT(executed_cases, 0) << "No correctness cases selected. Check LLAMINAR_CUDA_TC_FORMATS / LLAMINAR_CUDA_TC_SHAPES.";
    }

    TEST_F(CUDABlockwiseTensorCorePerf, Performance_AllFormats_AllShapes)
    {
        const SweepConfig cfg = loadSweepConfig();
        int executed_cases = 0;
        std::map<std::string, double> baseline_times; // "format_model_layer" -> tuned time_us at TP=1

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
                const size_t weight_bytes = weights_gemv->size_bytes();

                const RunResult np_tuned_gemv = runKernel(weights_gemv.get(), 1, shape.n, shape.k, cfg);

                // -- HBM bandwidth metrics --
                const size_t activation_bytes = static_cast<size_t>(shape.k) * sizeof(float);
                const size_t output_bytes = static_cast<size_t>(shape.n) * sizeof(float);
                const size_t total_bytes = weight_bytes + activation_bytes + output_bytes;
                const double eff_bw = static_cast<double>(total_bytes) / (np_tuned_gemv.min_us * 1e-6) / 1e9;
                const double pct_peak = (peak_bw_GB_s_ > 0) ? (eff_bw / peak_bw_GB_s_ * 100.0) : 0.0;

                // -- TP scaling efficiency --
                const int tp = parseTPDegree(shape.name);
                const std::string base_key = format.name + "_" + tpBaselineKey(shape.name);
                double tp_eff = -1.0;
                if (tp == 1)
                {
                    baseline_times[base_key] = np_tuned_gemv.min_us;
                }
                else
                {
                    const auto it = baseline_times.find(base_key);
                    if (it != baseline_times.end() && it->second > 0)
                    {
                        tp_eff = (it->second / np_tuned_gemv.min_us) / tp;
                    }
                }

                // Build optional TP efficiency suffix
                char tp_str[64] = "";
                if (tp_eff >= 0)
                {
                    std::snprintf(tp_str, sizeof(tp_str), " tp_eff=%.3f", tp_eff);
                }

                std::fprintf(stderr,
                             "[CUDABlockwiseTC][GEMV] format=%s shape=%s M=1 N=%d K=%d warmup=%d bench=%d tuned=%d "
                             "time_us=%.3f eff_bw=%.1fGB/s pct_peak=%.1f%%%s\n",
                             format.name.c_str(), shape.name.c_str(), shape.n, shape.k,
                             cfg.warmup_runs, cfg.bench_runs, cfg.tuned_gemv_enabled ? 1 : 0,
                             np_tuned_gemv.min_us, eff_bw, pct_peak, tp_str);

                if (cfg.gemv_only)
                {
                    ++executed_cases;
                    continue;
                }

                for (int m : cfg.performance_prefill_m)
                {
                    auto weights_gemm = format.create(static_cast<size_t>(shape.n), static_cast<size_t>(shape.k));
                    const RunResult gemm_result = runKernel(weights_gemm.get(), m, shape.n, shape.k, cfg);

                    std::fprintf(stderr,
                                 "[CUDABlockwiseTC][GEMM] format=%s shape=%s M=%d N=%d K=%d warmup=%d bench=%d "
                                 "gemm_dispatch=%s native_vnni=%d backend=%s split_k=%d "
                                 "min_us=%.3f\n",
                                 format.name.c_str(), shape.name.c_str(), m, shape.n, shape.k,
                                 cfg.warmup_runs, cfg.bench_runs, cfg.gemm_dispatch.c_str(),
                                 cfg.native_vnni_enabled ? 1 : 0,
                                 gemm_result.backend_family.empty() ? "native_vnni" : gemm_result.backend_family.c_str(),
                                 gemm_result.split_k,
                                 gemm_result.min_us);
                }

                ++executed_cases;
            }
        }

        ASSERT_GT(executed_cases, 0) << "No performance cases selected. Check LLAMINAR_CUDA_TC_FORMATS / LLAMINAR_CUDA_TC_SHAPES.";
    }
}

#endif