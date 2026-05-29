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
#include "kernels/cuda/gemm/CuBLASGemmKernel.h"
#include "../../../../utils/GpuPreparedGemmHarness.h"
#include "../../../../utils/PreparedWeightTestHarness.h"
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
        {"Q8_0", 19, [](size_t n, size_t k)
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
        {"9B_Attn", 4096, 4096},
        {"9B_FFN_Up", 12288, 4096},
        {"9B_FFN_Down", 4096, 12288},
        {"9B_LM_Head", 248320, 4096},
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
        // CutlassFallback removed — NativeVNNI is the sole CUDA GEMM path.
    };

    // KernelModeGuard is no longer needed — NativeVNNI is always enabled
    // and CUTLASS fallback no longer exists. Kept as a no-op struct for
    // compatibility with test call sites.
    struct KernelModeGuard
    {
        KernelModeGuard(bool /*native_vnni_enabled*/, bool /*force_cutlass_fallback*/) {}
        ~KernelModeGuard() = default;
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

    /**
     * @brief Run cuBLAS FP32 GEMM as a ground-truth reference.
     *
     * Dequantizes quantized weights to FP32 on host, uploads A and B_dequant
     * to GPU, then runs cuBLAS sgemm. The result serves as a numerically
     * exact (FP32) reference for validating quantized NativeVNNI output.
     *
     * Weight layout: quantized tensors store weights as [N × K] (row per
     * output neuron). cuBLAS expects B in row-major [N × K] with transB=true,
     * which matches the dequantized layout directly.
     *
     * @param weights     Quantized weight tensor (any supported format)
     * @param h_input     Host FP32 input tensor [M × K] (shared with NativeVNNI run)
     * @param m           Number of input rows (sequence length)
     * @param n           Output columns (hidden dim)
     * @param k           Inner dimension
     * @param cuda_device_id  CUDA device ordinal
     * @return RunResult with FP32 output vector (timing fields unused)
     */
    inline RunResult runCuBLASReference(TensorBase *weights, const float *h_input, int m, int n, int k, int cuda_device_id = 0)
    {
        if (cudaSetDevice(cuda_device_id) != cudaSuccess)
            throw std::runtime_error("cudaSetDevice failed");

        // Dequantize weights to FP32 on host — data() returns [N × K] row-major
        const float *h_weights_fp32 = weights->data();
        if (!h_weights_fp32)
            throw std::runtime_error("Failed to dequantize weights to FP32");

        // Allocate GPU buffers
        float *d_A = nullptr;
        float *d_B = nullptr;
        float *d_C = nullptr;
        const size_t a_bytes = static_cast<size_t>(m) * k * sizeof(float);
        const size_t b_bytes = static_cast<size_t>(n) * k * sizeof(float);
        const size_t c_bytes = static_cast<size_t>(m) * n * sizeof(float);

        if (cudaMalloc(&d_A, a_bytes) != cudaSuccess)
            throw std::runtime_error("cudaMalloc d_A failed");
        if (cudaMalloc(&d_B, b_bytes) != cudaSuccess)
        {
            cudaFree(d_A);
            throw std::runtime_error("cudaMalloc d_B failed");
        }
        if (cudaMalloc(&d_C, c_bytes) != cudaSuccess)
        {
            cudaFree(d_A);
            cudaFree(d_B);
            throw std::runtime_error("cudaMalloc d_C failed");
        }

        // Upload A [M×K] and B [N×K] to GPU
        cudaMemcpy(d_A, h_input, a_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_weights_fp32, b_bytes, cudaMemcpyHostToDevice);
        cudaMemset(d_C, 0, c_bytes);

        // Run cuBLAS sgemm: C[M×N] = A[M×K] × B^T[K×N]  (B stored as [N×K], transB=true)
        auto cublas_kernel = llaminar2::cuda::createCuBLASGemm(cuda_device_id);
        if (!cublas_kernel)
        {
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            throw std::runtime_error("createCuBLASGemm returned null");
        }

        bool ok = cublas_kernel->execute(d_A, d_B, d_C, m, n, k,
                                         /*transA=*/false, /*transB=*/true);
        if (!ok)
        {
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            throw std::runtime_error("cuBLAS GEMM execute failed");
        }

        cudaDeviceSynchronize();

        // Download result
        RunResult result;
        result.output.resize(static_cast<size_t>(m) * n);
        cudaMemcpy(result.output.data(), d_C, c_bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        result.native_family = "cublas_fp32";
        return result;
    }

    /**
     * @brief Run NativeVNNI quantized GEMM kernel.
     *
     * @param weights       Quantized weight tensor
     * @param m,n,k         GEMM dimensions
     * @param path          RunPath (only NativeVNNITensorCore supported)
     * @param warmup_runs   Number of warmup iterations
     * @param bench_runs    Number of timed iterations
     * @param cuda_device_id CUDA device ordinal
     * @param shared_input  Optional pre-created FP32 input [M×K] for deterministic
     *                      comparison with cuBLAS reference. If nullptr, creates
     *                      random input internally (seed 7).
     */
    inline RunResult runKernel(TensorBase *weights, int m, int n, int k, RunPath path, int warmup_runs, int bench_runs, int cuda_device_id = 0, const float *shared_input = nullptr)
    {
        if (cudaSetDevice(cuda_device_id) != cudaSuccess)
            throw std::runtime_error("cudaSetDevice failed");

        DeviceId device = DeviceId::cuda(cuda_device_id);

        // Build the GEMM kernel via the production GPU weight load pipeline.
        //
        // NOTE on the API: KernelFactory::prepareGemmHandleLocal() (and therefore
        // PreparedWeightStore::prepareGemm()) intentionally REJECT GPU INT8-packed
        // weights — those device kernels must be constructed from VRAM-resident,
        // VNNI-repacked payloads owned by a WeightVRAMPool. The shared helper
        // makeGpuPreparedGemm() drives that exact pipeline (upload + GPU repack +
        // pool-slot kernel construction) and registers the handle through the
        // un-guarded registerPreparedGemmHandle() path. The returned struct owns
        // the orchestrator + store and must stay alive while `kernel` is used.
        auto prepared = makeGpuPreparedGemm(weights, device, "perf.cuda_native_vnni_gemm.weight");
        ITensorGemm *kernel = prepared.kernel;

        auto *workspace_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
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

        // Use shared input if provided, otherwise create random input
        std::unique_ptr<FP32Tensor> h_input;
        const float *input_ptr = shared_input;
        if (!input_ptr)
        {
            h_input = TestTensorFactory::createFP32Random({static_cast<size_t>(m), static_cast<size_t>(k)}, -0.25f, 0.25f, 7);
            input_ptr = h_input->data();
        }

        // Create tensor wrappers and upload to GPU
        auto A_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(k)});
        std::memcpy(A_tensor->mutable_data(), input_ptr, static_cast<size_t>(m) * k * sizeof(float));
        auto C_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(n)});

        DeviceId gpu_device = DeviceId::cuda(cuda_device_id);
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

        // Surface any sticky CUDA error from warmup before timing so that an
        // illegal access cannot silently serialize as a 0.000 us row.
        if (cudaError_t err = cudaDeviceSynchronize(); err != cudaSuccess)
        {
            throw std::runtime_error(std::string("CUDA error after warmup: ") + cudaGetErrorString(err));
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

            // Detect kernel launch / execution errors that would otherwise produce
            // a bogus ~0 us timing instead of a hard failure.
            if (cudaError_t err = cudaGetLastError(); err != cudaSuccess)
            {
                throw std::runtime_error(std::string("CUDA error during bench run: ") + cudaGetErrorString(err));
            }

            times_us.push_back(static_cast<double>(elapsed_ms) * 1000.0);
        }

        RunResult result;
        result.output.resize(static_cast<size_t>(m) * n);
        C_tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        std::memcpy(result.output.data(), C_tensor->data(), static_cast<size_t>(m) * n * sizeof(float));

        if (workspace_consumer)
            workspace_consumer->unbindWorkspace();

        result.min_us = *std::min_element(times_us.begin(), times_us.end());
        result.mean_us = std::accumulate(times_us.begin(), times_us.end(), 0.0) / static_cast<double>(times_us.size());
        if (path == RunPath::NativeVNNITensorCore)
        {
            result.native_family = "native_vnni_tc";
        }
        return result;
    }
}