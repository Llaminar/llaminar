/**
 * @file CUDANativeVNNIGemmPerfCommon.h
 * @brief Shared utilities for CUDA native-vnni GEMM perf and sweep harnesses.
 */

#pragma once

#include <cuda_runtime.h>

#include "backends/DeviceId.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/KernelFactory.h"
#include "kernels/cuda/gemm/CUDAQuantisedGemmKernel.h"
#include "../../../../utils/TestTensorFactory.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2::test::native_vnni_gemm_perf
{
    using llaminar::v2::kernels::KernelFactory;

    extern "C" const char *cudaFusedTCGemm_lastSelectedFamily();
    extern "C" const char *cudaFusedTCGemmV2_lastSelectedFamily();

    constexpr int kDefaultWarmupRuns = 3;
    constexpr int kDefaultBenchRuns = 10;
    constexpr float kCosineGate = 0.9990f;

    struct FormatSpec
    {
        std::string name;
        uint8_t codebook_id;
        std::function<std::unique_ptr<TensorBase>(size_t, size_t)> create;
    };

    inline const std::vector<FormatSpec> kFormats = {
        {"Q4_0", 0, [](size_t n, size_t k)
         { return TestTensorFactory::createQ4_0Random({n, k}); }},
        {"IQ4_NL", 4, [](size_t n, size_t k)
         { return TestTensorFactory::createIQ4_NLRandom({n, k}); }},
        {"Q4_1", 5, [](size_t n, size_t k)
         { return TestTensorFactory::createQ4_1Random({n, k}); }},
        {"Q5_0", 6, [](size_t n, size_t k)
         { return TestTensorFactory::createQ5_0Random({n, k}); }},
        {"Q5_1", 7, [](size_t n, size_t k)
         { return TestTensorFactory::createQ5_1Random({n, k}); }},
        {"Q6_K", 8, [](size_t n, size_t k)
         { return TestTensorFactory::createQ6_KRandom({n, k}); }},
        {"Q3_K", 9, [](size_t n, size_t k)
         { return TestTensorFactory::createQ3_KRandom({n, k}); }},
        {"Q2_K", 10, [](size_t n, size_t k)
         { return TestTensorFactory::createQ2_KRandom({n, k}); }},
        {"IQ3_S", 11, [](size_t n, size_t k)
         { return TestTensorFactory::createIQ3_SRandom({n, k}); }},
        {"IQ3_XXS", 12, [](size_t n, size_t k)
         { return TestTensorFactory::createIQ3_XXSRandom({n, k}); }},
        {"IQ2_S", 13, [](size_t n, size_t k)
         { return TestTensorFactory::createIQ2_SRandom({n, k}); }},
        {"IQ2_XS", 14, [](size_t n, size_t k)
         { return TestTensorFactory::createIQ2_XSRandom({n, k}); }},
        {"IQ2_XXS", 15, [](size_t n, size_t k)
         { return TestTensorFactory::createIQ2_XXSRandom({n, k}); }},
        {"IQ1_S", 16, [](size_t n, size_t k)
         { return TestTensorFactory::createIQ1_SRandom({n, k}); }},
        {"IQ1_M", 17, [](size_t n, size_t k)
         { return TestTensorFactory::createIQ1_MRandom({n, k}); }},
        {"Q8_0", 18, [](size_t n, size_t k)
         { return TestTensorFactory::createQ8_0Random({n, k}); }},
    };

    struct Shape
    {
        std::string name;
        int n;
        int k;
    };

    inline const std::vector<Shape> kQwenShapes = {
        {"0.5B_Attn", 896, 896},
        {"0.5B_FFN_Up", 4864, 896},
        {"0.5B_FFN_Down", 896, 4864},
        {"0.5B_LM_Head", 151936, 896},
        {"0.5B_TP2_Attn_QKV", 448, 896},
        {"0.5B_TP2_Attn_Wo", 896, 448},
        {"0.5B_TP2_FFN_Up", 2432, 896},
        {"0.5B_TP2_FFN_Down", 896, 2432},
        {"0.5B_TP2_LM_Head", 75968, 896},
        {"0.5B_TP4_Attn_QKV", 224, 896},
        {"0.5B_TP4_Attn_Wo", 896, 224},
        {"0.5B_TP4_FFN_Up", 1216, 896},
        {"0.5B_TP4_FFN_Down", 896, 1216},
        {"0.5B_TP4_LM_Head", 37984, 896},
        {"3B_Attn", 2048, 2048},
        {"3B_FFN_Up", 11008, 2048},
        {"3B_FFN_Down", 2048, 11008},
        {"3B_LM_Head", 151936, 2048},
        {"3B_TP2_Attn_QKV", 1024, 2048},
        {"3B_TP2_Attn_Wo", 2048, 1024},
        {"3B_TP2_FFN_Up", 5504, 2048},
        {"3B_TP2_FFN_Down", 2048, 5504},
        {"3B_TP2_LM_Head", 75968, 2048},
        {"3B_TP4_Attn_QKV", 512, 2048},
        {"3B_TP4_Attn_Wo", 2048, 512},
        {"3B_TP4_FFN_Up", 2752, 2048},
        {"3B_TP4_FFN_Down", 2048, 2752},
        {"3B_TP4_LM_Head", 37984, 2048},
        {"7B_Attn", 3584, 3584},
        {"7B_FFN_Up", 18944, 3584},
        {"7B_FFN_Down", 3584, 18944},
        {"7B_LM_Head", 152064, 3584},
        {"7B_TP2_Attn_QKV", 1792, 3584},
        {"7B_TP2_Attn_Wo", 3584, 1792},
        {"7B_TP2_FFN_Up", 9472, 3584},
        {"7B_TP2_FFN_Down", 3584, 9472},
        {"7B_TP2_LM_Head", 76032, 3584},
        {"7B_TP4_Attn_QKV", 896, 3584},
        {"7B_TP4_Attn_Wo", 3584, 896},
        {"7B_TP4_FFN_Up", 4736, 3584},
        {"7B_TP4_FFN_Down", 3584, 4736},
        {"7B_TP4_LM_Head", 38016, 3584},
        {"14B_Attn", 5120, 5120},
        {"14B_FFN_Up", 13824, 5120},
        {"14B_FFN_Down", 5120, 13824},
        {"14B_LM_Head", 152064, 5120},
        {"14B_TP2_Attn_QKV", 2560, 5120},
        {"14B_TP2_Attn_Wo", 5120, 2560},
        {"14B_TP2_FFN_Up", 6912, 5120},
        {"14B_TP2_FFN_Down", 5120, 6912},
        {"14B_TP2_LM_Head", 76032, 5120},
        {"14B_TP4_Attn_QKV", 1280, 5120},
        {"14B_TP4_Attn_Wo", 5120, 1280},
        {"14B_TP4_FFN_Up", 3456, 5120},
        {"14B_TP4_FFN_Down", 5120, 3456},
        {"14B_TP4_LM_Head", 38016, 5120},
    };

    inline const std::vector<int> kPrefillMValues = {32, 64, 128};

    struct RunConfig
    {
        int warmup_runs = kDefaultWarmupRuns;
        int bench_runs = kDefaultBenchRuns;
        int correctness_prefill_m = 128;
        std::vector<int> performance_prefill_m = kPrefillMValues;
        std::set<std::string> format_filters;
        std::set<std::string> shape_filters;
        int max_cases = std::numeric_limits<int>::max();
        bool smoke = false;
        std::string csv_path;
        int performance_workers = 0;
    };

    struct RunResult
    {
        std::vector<float> output;
        double min_us = 0.0;
        double mean_us = 0.0;
        std::string native_family;
    };

    enum class RunPath
    {
        NativeVNNITensorCore,
        CutlassFallback,
    };

    struct KernelModeGuard
    {
        KernelModeGuard(bool native_vnni_enabled, bool force_cutlass_fallback)
            : native_vnni_enabled_(llaminar2::cuda::CUDAQuantisedGemmKernel::isNativeVNNIEnabled()),
              force_cutlass_fallback_(llaminar2::cuda::CUDAQuantisedGemmKernel::isForceCutlassFallback())
        {
            llaminar2::cuda::CUDAQuantisedGemmKernel::setNativeVNNIEnabled(native_vnni_enabled);
            llaminar2::cuda::CUDAQuantisedGemmKernel::setForceCutlassFallback(force_cutlass_fallback);
        }

        ~KernelModeGuard()
        {
            llaminar2::cuda::CUDAQuantisedGemmKernel::setNativeVNNIEnabled(native_vnni_enabled_);
            llaminar2::cuda::CUDAQuantisedGemmKernel::setForceCutlassFallback(force_cutlass_fallback_);
        }

        bool native_vnni_enabled_;
        bool force_cutlass_fallback_;
    };

    inline std::string toLower(std::string value)
    {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        return value;
    }

    inline std::string trim(std::string value)
    {
        const auto begin = value.find_first_not_of(" \t\n\r");
        if (begin == std::string::npos)
            return {};
        const auto end = value.find_last_not_of(" \t\n\r");
        return value.substr(begin, end - begin + 1);
    }

    inline std::optional<int> getEnvInt(const char *name)
    {
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return std::nullopt;
        return std::atoi(raw);
    }

    inline std::optional<double> getEnvDouble(const char *name)
    {
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return std::nullopt;
        return std::atof(raw);
    }

    inline std::string getEnvString(const char *name)
    {
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return {};
        return trim(raw);
    }

    inline bool getEnvFlag(const char *name)
    {
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return false;
        return std::atoi(raw) != 0;
    }

    inline std::set<std::string> getEnvCsvSet(const char *name)
    {
        std::set<std::string> values;
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return values;

        std::stringstream stream(raw);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            token = trim(token);
            if (!token.empty())
                values.insert(toLower(token));
        }
        return values;
    }

    inline std::vector<int> getEnvCsvInts(const char *name)
    {
        std::vector<int> values;
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return values;

        std::stringstream stream(raw);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            token = trim(token);
            if (!token.empty())
                values.push_back(std::atoi(token.c_str()));
        }
        return values;
    }

    inline RunConfig loadRunConfig()
    {
        RunConfig cfg;
        if (const auto value = getEnvInt("LLAMINAR_CUDA_NATIVE_GEMM_WARMUP"))
            cfg.warmup_runs = std::max(0, *value);
        if (const auto value = getEnvInt("LLAMINAR_CUDA_NATIVE_GEMM_BENCH"))
            cfg.bench_runs = std::max(1, *value);
        if (const auto value = getEnvInt("LLAMINAR_CUDA_NATIVE_GEMM_CORRECTNESS_PREFILL_M"))
            cfg.correctness_prefill_m = std::max(2, *value);
        if (const auto value = getEnvInt("LLAMINAR_CUDA_NATIVE_GEMM_MAX_CASES"))
            cfg.max_cases = std::max(1, *value);

        const auto prefill_m = getEnvCsvInts("LLAMINAR_CUDA_NATIVE_GEMM_PREFILL_M");
        if (!prefill_m.empty())
            cfg.performance_prefill_m = prefill_m;

        cfg.format_filters = getEnvCsvSet("LLAMINAR_CUDA_NATIVE_GEMM_FORMATS");
        cfg.shape_filters = getEnvCsvSet("LLAMINAR_CUDA_NATIVE_GEMM_SHAPES");
        cfg.smoke = getEnvFlag("LLAMINAR_CUDA_NATIVE_GEMM_SMOKE");
        if (const auto value = getEnvInt("LLAMINAR_CUDA_NATIVE_GEMM_WORKERS"))
            cfg.performance_workers = std::max(0, *value);

        const std::string sweep_out = getEnvString("LLAMINAR_CUDA_NATIVE_GEMM_SWEEP_OUT");
        if (!sweep_out.empty())
            cfg.csv_path = sweep_out;

        const std::string sweep_csv = getEnvString("LLAMINAR_CUDA_NATIVE_GEMM_SWEEP_CSV");
        if (!sweep_csv.empty())
            cfg.csv_path = sweep_csv;

        if (cfg.smoke)
        {
            cfg.warmup_runs = 1;
            cfg.bench_runs = 2;
            cfg.correctness_prefill_m = 32;
            cfg.performance_prefill_m = {32};
            cfg.max_cases = std::min(cfg.max_cases, 12);
        }

        return cfg;
    }

    inline bool shouldRunName(const std::set<std::string> &filters, const std::string &name)
    {
        return filters.empty() || filters.count(toLower(name)) > 0;
    }

    inline double estimatePeakInt8TensorCoreTops()
    {
        if (const auto value = getEnvDouble("LLAMINAR_CUDA_NATIVE_GEMM_PEAK_TC_TOPS"))
            return std::max(0.0, *value);

        int device = 0;
        cudaGetDevice(&device);
        int clock_khz = 0;
        int sm_count = 0;
        cudaDeviceProp props{};
        cudaGetDeviceProperties(&props, device);
        cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
        if (clock_khz <= 0 || sm_count <= 0)
            return 0.0;

        double int8_ops_per_sm_per_cycle = 0.0;
        switch (props.major * 10 + props.minor)
        {
        case 75: // Turing
        case 80: // A100 / Ampere datacenter
        case 89: // Ada Lovelace
            int8_ops_per_sm_per_cycle = 2048.0;
            break;
        case 86: // GA10x Ampere, e.g. RTX 3090
            int8_ops_per_sm_per_cycle = 2304.0;
            break;
        case 90: // Hopper
            int8_ops_per_sm_per_cycle = 4096.0;
            break;
        default:
            return 0.0;
        }

        const double clock_ghz = static_cast<double>(clock_khz) / 1e6;
        return static_cast<double>(sm_count) * int8_ops_per_sm_per_cycle * clock_ghz / 1e3;
    }

    struct GemmThroughputMetrics
    {
        double achieved_tops = 0.0;
        double pct_tc_peak = 0.0;
    };

    inline GemmThroughputMetrics computeGemmThroughputMetrics(int m, int n, int k, double time_us, double peak_tc_tops)
    {
        if (time_us <= 0.0)
            return {};

        const double total_ops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
        const double achieved_tops = total_ops / (time_us * 1e-6) / 1e12;
        const double pct_tc_peak = (peak_tc_tops > 0.0) ? (achieved_tops / peak_tc_tops * 100.0) : 0.0;
        return {achieved_tops, pct_tc_peak};
    }

    inline float cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b)
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

    inline RunResult runKernel(TensorBase *weights, int m, int n, int k, RunPath path, int warmup_runs, int bench_runs, int cuda_device_id = 0)
    {
        const bool native_vnni_enabled = (path == RunPath::NativeVNNITensorCore);
        const bool force_cutlass_fallback = (path == RunPath::CutlassFallback);
        KernelModeGuard mode_guard(native_vnni_enabled, force_cutlass_fallback);

        if (cudaSetDevice(cuda_device_id) != cudaSuccess)
            throw std::runtime_error("cudaSetDevice failed");

        DeviceId device = DeviceId::cuda(cuda_device_id);
        if (!weights->ensureOnDevice(device))
            throw std::runtime_error("Failed to upload weights to CUDA device");

        auto kernel = KernelFactory::createGemm(weights, DeviceType::CUDA);
        if (!kernel)
            throw std::runtime_error("KernelFactory::createGemm returned null");

        auto *workspace_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel.get());
        std::unique_ptr<DeviceWorkspaceManager> workspace;
        if (workspace_consumer)
        {
            const auto reqs = workspace_consumer->getWorkspaceRequirements(m, n, k);
            // Compute actual budget from requirements instead of hardcoding
            size_t needed = 0;
            for (const auto &buf : reqs.buffers)
            {
                needed = (needed + buf.alignment - 1) & ~(buf.alignment - 1);
                needed += buf.size_bytes;
            }
            // Add 10% headroom for alignment padding
            const size_t budget = std::max(needed + needed / 10, size_t{64} * 1024 * 1024);
            workspace = std::make_unique<DeviceWorkspaceManager>(device, budget);
            if (!workspace->allocate(reqs))
                throw std::runtime_error("Failed to allocate CUDA GEMM workspace");
            workspace_consumer->bindWorkspace(workspace.get());
        }

        auto h_input = TestTensorFactory::createFP32Random({static_cast<size_t>(m), static_cast<size_t>(k)}, -0.25f, 0.25f, 7);

        // Create tensor wrappers and upload to GPU
        auto A_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(k)});
        std::memcpy(A_tensor->mutable_data(), h_input->data(), static_cast<size_t>(m) * k * sizeof(float));
        auto C_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(n)});

        DeviceId gpu_device(DeviceType::CUDA, 0);
        if (!A_tensor->ensureOnDevice(gpu_device))
            throw std::runtime_error("ensureOnDevice A failed");
        if (!C_tensor->ensureOnDevice(gpu_device))
            throw std::runtime_error("ensureOnDevice C failed");

        std::vector<double> times_us;
        times_us.reserve(static_cast<size_t>(bench_runs));

        for (int i = 0; i < warmup_runs; ++i)
        {
            if (!kernel->multiply_tensor(A_tensor.get(), C_tensor.get(), m, n, k))
            {
                throw std::runtime_error("CUDA native-vnni GEMM warmup failed");
            }
        }

        for (int i = 0; i < bench_runs; ++i)
        {
            cudaEvent_t start = nullptr;
            cudaEvent_t stop = nullptr;
            if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess)
            {
                throw std::runtime_error("cudaEventCreate failed");
            }

            cudaEventRecord(start);
            const bool ok = kernel->multiply_tensor(A_tensor.get(), C_tensor.get(), m, n, k);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float elapsed_ms = 0.0f;
            cudaEventElapsedTime(&elapsed_ms, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            if (!ok)
            {
                throw std::runtime_error("CUDA native-vnni GEMM bench run failed");
            }

            times_us.push_back(static_cast<double>(elapsed_ms) * 1000.0);
        }

        RunResult result;
        result.output.resize(static_cast<size_t>(m) * n);
        C_tensor->mark_device_dirty();
        std::memcpy(result.output.data(), C_tensor->data(), static_cast<size_t>(m) * n * sizeof(float));

        if (workspace_consumer)
            workspace_consumer->unbindWorkspace();

        result.min_us = *std::min_element(times_us.begin(), times_us.end());
        result.mean_us = std::accumulate(times_us.begin(), times_us.end(), 0.0) / static_cast<double>(times_us.size());
        if (path == RunPath::NativeVNNITensorCore)
        {
            // Prefer V2 fused TC → V1 fused TC
            const char *fused_v2 = cudaFusedTCGemmV2_lastSelectedFamily();
            const char *fused_v1 = cudaFusedTCGemm_lastSelectedFamily();
            if (fused_v2 && std::string(fused_v2).substr(0, 2) == "v2")
                result.native_family = fused_v2;
            else if (fused_v1 && std::string(fused_v1) != "unknown")
                result.native_family = fused_v1;
            else
                result.native_family = "native_vnni_tc";
        }
        return result;
    }
}