/**
 * @file Perf__NativeVNNI_GEMM.cpp
 * @brief Performance and correctness benchmark for native-VNNI GEMM kernels
 *
 * Measures M>1 (prefill) throughput for Q4_0 and IQ4_NL formats using the
 * native-VNNI GEMM kernel, comparing against the INT8 VNNI GEMM (V3/V7)
 * baseline on identical shapes.
 *
 * Benchmarked dimensions are drawn from Qwen2.5-0.5B, 3B, and 7B layer
 * shapes at multiple sequence lengths (M=32, 64, 128, 256).
 *
 * Metrics reported:
 *   - Kernel time (μs): min/mean across benchmark runs
 *   - Speedup vs INT8: int8_min_us / native_vnni_min_us (>1 = faster)
 *   - Theoretical speedup: 8.0 / bpw (from streaming fewer weight bytes)
 *   - Kernel efficiency: actual_speedup / theoretical_speedup × 100%
 *   - Cosine similarity: output vs FP32 reference (correctness gate: >0.9999)
 *
 * The benchmark uses multiply_tensor() which includes:
 *   1. FP32→INT8 activation quantization on GPU
 *   2. Native-VNNI GEMM kernel dispatch (Step 1b) or INT8 GEMM (V3/V7)
 *   3. Per-block FP16 scale application (native-VNNI) or scaling epilogue (INT8)
 *
 * @note Requires ROCm device. Tests skip if no GPU is available.
 * @note Run with build_v2_release for representative timing.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <omp.h>

#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"
#include "../../../utils/TestTensorFactory.h"
#include "fort.hpp"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include "GpuVerification.h"

extern "C"
{
    bool rocmQuantGemm_quantizeActivationsBlockwise(
        const float *d_A_fp32,
        int8_t *d_A_int8,
        float *d_scales_blockwise,
        int M, int K,
        int rocm_device_id, void *stream,
        int block_size);

    bool rocmQuantGemm_quantizeActivationsBlockwiseWithSums(
        const float *d_A_fp32,
        int8_t *d_A_int8,
        float *d_scales_blockwise,
        int32_t *d_sums_blockwise,
        int M, int K,
        int rocm_device_id, void *stream,
        int block_size);

    bool rocmGemm_native_vnni_fp32(
        const int8_t *d_A_int8,
        const uint8_t *d_payload,
        const void *d_block_scales,
        const void *d_block_mins,
        const void *d_block_emins,
        float *d_output,
        const float *d_scales_A,
        const float *d_scales_A_blockwise,
        int M, int N, int K,
        uint8_t codebook_id,
        int device_id, void *stream);

    bool rocmGemv_native_vnni_small_m_batched_fp32(
        const int8_t *d_A_int8,
        const uint8_t *const *d_payloads,
        const uint16_t *const *d_block_scales,
        const uint16_t *const *d_block_mins,
        const uint32_t *const *d_block_emins,
        const float *const *d_biases,
        float *const *d_outputs,
        const float *d_scale_A_blockwise,
        float *const *d_partials,
        const int *Ns,
        int num_projections,
        int M, int K,
        uint8_t codebook_id,
        int device_id,
        void *stream);

    bool rocmGemv_native_vnni_small_m_batched_fp32_with_sums(
        const int8_t *d_A_int8,
        const uint8_t *const *d_payloads,
        const uint16_t *const *d_block_scales,
        const uint16_t *const *d_block_mins,
        const uint32_t *const *d_block_emins,
        const float *const *d_biases,
        float *const *d_outputs,
        const float *d_scale_A_blockwise,
        const int32_t *d_sum_A_blockwise,
        float *const *d_partials,
        const int *Ns,
        int num_projections,
        int M, int K,
        uint8_t codebook_id,
        int device_id,
        void *stream);

    void rocmGemv_native_vnni_set_tuning_overrides(int kb, int target_waves_per_cu);
    void rocmGemv_native_vnni_reset_tuning_overrides();
}
#endif

using namespace llaminar2;
using namespace llaminar2::rocm;
using namespace llaminar2::test;

namespace
{
#ifdef HAVE_ROCM
    using gpu_verify::destroyAllHipBLAS;
    using gpu_verify::gpuCosineSimilarity;
    using gpu_verify::gpuReferenceFP32Gemm;
    using gpu_verify::GpuWeightsCache;
#endif

    // =============================================================================
    // Constants
    // =============================================================================

    constexpr int WARMUP_RUNS = 5;
    constexpr int BENCH_RUNS = 20;

    /// Correctness gate: cosine similarity between native-VNNI and FP32 reference
    constexpr float COSINE_SIM_GATE = 0.9999f;

    /// Small-M verifier paths compare against quantized-activation output, so use
    /// the same practical gate as focused integration regressions.
    constexpr float MTP_SMALL_M_COSINE_GATE = 0.985f;

    /// Performance gate: speedup over INT8 GEMM baseline (grand total average)
    constexpr float SPEEDUP_GATE = 1.0f;

    /// Number of GPUs to use (auto-detected, capped at available)
    static int NUM_GPUS = 1;

    // =============================================================================
    // Format descriptors covered by the native-VNNI small-M perf matrix.
    // =============================================================================

    struct GEMMFormatSpec
    {
        std::string name;
        double bpw;

        std::function<std::unique_ptr<TensorBase>(size_t N, size_t K)> create;
    };

    static const std::vector<GEMMFormatSpec> GEMM_FORMATS = {
        {"Q4_0", 4.5, [](size_t N, size_t K)
         { return TestTensorFactory::createQ4_0Random({N, K}); }},
        {"IQ4_NL", 4.5, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ4_NLRandom({N, K}); }},
        {"Q4_1", 5.0, [](size_t N, size_t K)
         { return TestTensorFactory::createQ4_1Random({N, K}); }},
        {"Q5_0", 5.5, [](size_t N, size_t K)
         { return TestTensorFactory::createQ5_0Random({N, K}); }},
        {"Q5_1", 6.0, [](size_t N, size_t K)
         { return TestTensorFactory::createQ5_1Random({N, K}); }},
        {"Q4_K", 4.5, [](size_t N, size_t K)
         { return TestTensorFactory::createQ4_KRandom({N, K}); }},
        {"Q5_K", 5.5, [](size_t N, size_t K)
         { return TestTensorFactory::createQ5_KRandom({N, K}); }},
        {"Q6_K", 6.5625, [](size_t N, size_t K)
         { return TestTensorFactory::createQ6_KRandom({N, K}); }},
        {"Q3_K", 3.4375, [](size_t N, size_t K)
         { return TestTensorFactory::createQ3_KRandom({N, K}); }},
        {"Q2_K", 2.5625, [](size_t N, size_t K)
         { return TestTensorFactory::createQ2_KRandom({N, K}); }},
        {"IQ3_S", 3.4375, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ3_SRandom({N, K}); }},
        {"IQ3_XXS", 3.0625, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ3_XXSRandom({N, K}); }},
        {"IQ2_S", 2.5, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ2_SRandom({N, K}); }},
        {"IQ2_XS", 2.3125, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ2_XSRandom({N, K}); }},
        {"IQ2_XXS", 2.0625, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ2_XXSRandom({N, K}); }},
        {"IQ1_S", 1.5625, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ1_SRandom({N, K}); }},
        {"IQ1_M", 1.75, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ1_MRandom({N, K}); }},
    };

    // =============================================================================
    // GEMM shape definitions (N×K with variable M)
    // =============================================================================

    struct GEMMShape
    {
        std::string name;
        int N;
        int K;
    };

    /// Sequence lengths to benchmark (typical prefill sizes)
    static const std::vector<int> M_VALUES = {32, 64, 128, 256};

    /// MTP verifier batch sizes exercised by the Phase 13.5 small-M route.
    static const std::vector<int> MTP_SMALL_M_VALUES = {2, 3, 4};

    // Qwen2.5-0.5B:  hidden=896,  intermediate=4864
    // Qwen2.5-3B:    hidden=2048, intermediate=11008
    // Qwen2.5-7B:    hidden=3584, intermediate=18944
    static const std::vector<GEMMShape> GEMM_SHAPES = {
        // Qwen2.5-0.5B
        {"0.5B_AttnOut", 896, 896},
        {"0.5B_FFN_Up", 4864, 896},
        {"0.5B_FFN_Dn", 896, 4864},
        {"0.5B_LM_Head", 151936, 896},
        // Qwen2.5-3B
        {"3B_AttnOut", 2048, 2048},
        {"3B_FFN_Up", 11008, 2048},
        {"3B_FFN_Dn", 2048, 11008},
        {"3B_LM_Head", 151936, 2048},
        // Qwen2.5-7B
        {"7B_AttnOut", 3584, 3584},
        {"7B_FFN_Up", 18944, 3584},
        {"7B_FFN_Dn", 3584, 18944},
        {"7B_LM_Head", 151936, 3584},
    };

    static const std::vector<GEMMShape> MTP_SMALL_M_SHAPES = {
        {"Qwen36_MTP_HiddenProjection", 5120, 5120},
        {"Qwen36_FFN_DownProjection", 5120, 17408},
        {"Qwen36_GDN_InnerProjection", 10240, 5120},
        {"Qwen36_GDN_TimeProjection", 1024, 5120},
        {"Qwen36_GDN_OutputProjection", 5120, 6144},
    };

    // =============================================================================
    // Benchmark result
    // =============================================================================

    struct GEMMBenchResult
    {
        std::string format_name;
        double bpw;
        std::string shape_name;
        int M, N, K;

        double min_us;
        double mean_us;
        double stddev_us;

        double native_weight_bytes; // Native-VNNI payload + scales bytes read

        // INT8 reference comparison
        double int8_min_us = 0.0;
        double speedup_vs_int8 = 0.0;     // int8_min_us / min_us
        double theoretical_speedup = 0.0; // 8.0 / bpw
        double kernel_efficiency = 0.0;   // (speedup / theoretical) × 100

        // Correctness
        float cosine_sim = 0.0f;
        bool correctness_pass = false;

        // GFLOPS
        double gflops = 0.0;
    };

    struct DirectPrefillComparisonResult
    {
        std::string format_name;
        std::string shape_name;
        int M = 0;
        int N = 0;
        int K = 0;

        double small_m_min_us = 0.0;
        double direct_prefill_min_us = 0.0;
        double direct_prefill_quant_min_us = 0.0;
        double prefill_quant_speedup = 0.0;

        float cosine_sim = 0.0f;
        bool correctness_pass = false;
        bool valid = false;
    };

    struct BatchedGDNProjectionResult
    {
        std::string group_name;
        std::string format_name;
        std::string variant_name;
        int M = 0;
        int K = 0;
        int projections = 0;
        int total_N = 0;
        int kb = 0;
        int target_waves = 0;
        uint8_t codebook_id = 0;
        double min_us = 0.0;
        double mean_us = 0.0;
        double stddev_us = 0.0;
        float min_cosine_sim = 0.0f;
        bool correctness_pass = false;
        bool valid = false;
    };

#ifdef HAVE_ROCM
    struct HipBuffer
    {
        void *ptr = nullptr;

        ~HipBuffer()
        {
            if (ptr)
                (void)hipFree(ptr);
        }

        bool allocate(size_t bytes)
        {
            return hipMalloc(&ptr, bytes) == hipSuccess;
        }

        template <typename T>
        T *as()
        {
            return reinterpret_cast<T *>(ptr);
        }

        template <typename T>
        const T *as() const
        {
            return reinterpret_cast<const T *>(ptr);
        }
    };

    struct HipStream
    {
        hipStream_t stream = nullptr;

        ~HipStream()
        {
            if (stream)
                (void)hipStreamDestroy(stream);
        }

        bool create()
        {
            return hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) == hipSuccess && stream != nullptr;
        }

        void *asVoid() const
        {
            return reinterpret_cast<void *>(stream);
        }
    };

    struct BatchedDecodeVariant
    {
        std::string name;
        int kb = 0;
        int target_waves = 0;
        bool automatic = true;
    };

    class BatchedDecodeTuningOverrideGuard
    {
    public:
        explicit BatchedDecodeTuningOverrideGuard(const BatchedDecodeVariant &variant)
        {
            if (variant.automatic)
                rocmGemv_native_vnni_reset_tuning_overrides();
            else
                rocmGemv_native_vnni_set_tuning_overrides(variant.kb, variant.target_waves);
        }

        ~BatchedDecodeTuningOverrideGuard()
        {
            rocmGemv_native_vnni_reset_tuning_overrides();
        }
    };
#endif

    static std::string trim(std::string value)
    {
        const auto begin = value.find_first_not_of(" \t\n\r");
        if (begin == std::string::npos)
            return {};
        const auto end = value.find_last_not_of(" \t\n\r");
        return value.substr(begin, end - begin + 1);
    }

    static std::string toLower(std::string value)
    {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        return value;
    }

    static std::string getEnvString(const char *name)
    {
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return {};
        return trim(raw);
    }

    static std::optional<int> getEnvInt(const char *name)
    {
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return std::nullopt;
        return std::atoi(raw);
    }

    static std::set<std::string> getEnvCsvSet(const char *name)
    {
        std::set<std::string> values;
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return values;
        std::stringstream stream(raw);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            token = toLower(trim(token));
            if (!token.empty())
                values.insert(token);
        }
        return values;
    }

    static bool shouldRunName(const std::set<std::string> &filters, const std::string &name)
    {
        return filters.empty() || filters.count(toLower(name)) != 0;
    }

#ifdef HAVE_ROCM
    static BatchedDecodeVariant parseBatchedDecodeVariantToken(std::string token)
    {
        token = trim(token);
        std::string compact;
        compact.reserve(token.size());
        for (char c : token)
        {
            if (c != '/' && c != '_' && c != '-' && c != ' ')
                compact.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
        }

        if (compact == "AUTO")
            return BatchedDecodeVariant{"AUTO", 0, 0, true};

        const size_t kb_pos = compact.find("KB");
        const size_t tw_pos = compact.find("TW");
        if (kb_pos == std::string::npos || tw_pos == std::string::npos || tw_pos <= kb_pos + 2)
            throw std::runtime_error("Invalid ROCm NativeVNNI batched decode variant token: " + token);

        const int kb = std::atoi(compact.substr(kb_pos + 2, tw_pos - (kb_pos + 2)).c_str());
        const int tw = std::atoi(compact.substr(tw_pos + 2).c_str());
        if (kb <= 0 || tw <= 0 || kb > 64)
            throw std::runtime_error("Invalid ROCm NativeVNNI batched decode variant values: " + token);

        return BatchedDecodeVariant{"KB" + std::to_string(kb) + "/TW" + std::to_string(tw), kb, tw, false};
    }

    static std::vector<BatchedDecodeVariant> getBatchedDecodeVariants()
    {
        const char *raw = std::getenv("LLAMINAR_ROCM_NVNNI_BATCHED_DECODE_VARIANTS");
        const std::string spec = (raw && *raw) ? raw : "auto,kb1tw8,kb2tw12,kb4tw12,kb8tw12,kb8tw24";

        std::vector<BatchedDecodeVariant> variants;
        std::stringstream stream(spec);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            token = trim(token);
            if (!token.empty())
                variants.push_back(parseBatchedDecodeVariantToken(token));
        }
        if (variants.empty())
            variants.push_back(BatchedDecodeVariant{"AUTO", 0, 0, true});
        return variants;
    }
#endif

    // =============================================================================
    // Statistics helper
    // =============================================================================

    static void computeStats(const std::vector<double> &times_us,
                             double &mean, double &min_val,
                             double &max_val, double &stddev)
    {
        mean = std::accumulate(times_us.begin(), times_us.end(), 0.0) /
               static_cast<double>(times_us.size());
        min_val = *std::min_element(times_us.begin(), times_us.end());
        max_val = *std::max_element(times_us.begin(), times_us.end());

        double sq_sum = 0.0;
        for (double t : times_us)
            sq_sum += (t - mean) * (t - mean);
        stddev = std::sqrt(sq_sum / static_cast<double>(times_us.size()));
    }

    // =============================================================================
    // Test fixture
    // =============================================================================

    class NativeVNNIGEMMPerfTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
#ifdef HAVE_ROCM
            int device_count = 0;
            hipError_t err = hipGetDeviceCount(&device_count);
            has_device_ = (err == hipSuccess && device_count > 0);
            if (has_device_)
            {
                NUM_GPUS = std::min(device_count, 3); // Use up to 3 GPUs
                (void)hipSetDevice(0);
                hipDeviceProp_t props;
                hipGetDeviceProperties(&props, 0);
                device_name_ = std::string(props.name) + " (" + props.gcnArchName + ")";
            }
#else
            has_device_ = false;
#endif
        }

        void TearDown() override
        {
#ifdef HAVE_ROCM
            destroyAllHipBLAS();
#endif
        }

        bool has_device_ = false;
        std::string device_name_;

#ifdef HAVE_ROCM

        /// Time a GEMM kernel call on a specific device. Returns sorted timing vector in μs.
        static std::vector<double> timeGEMMKernel(ROCmQuantisedGemmKernel &kernel,
                                                  TensorBase *input, TensorBase *output,
                                                  int M, int N, int K, int device_id)
        {
            (void)hipSetDevice(device_id);

            // Warmup
            for (int i = 0; i < WARMUP_RUNS; ++i)
                kernel.multiply_tensor(input, output, M, N, K);
            (void)hipDeviceSynchronize();

            // Timed runs
            hipEvent_t start = nullptr, stop = nullptr;
            (void)hipEventCreate(&start);
            (void)hipEventCreate(&stop);

            std::vector<double> times_us;
            times_us.reserve(BENCH_RUNS);

            for (int i = 0; i < BENCH_RUNS; ++i)
            {
                (void)hipDeviceSynchronize();
                (void)hipEventRecord(start);
                kernel.multiply_tensor(input, output, M, N, K);
                (void)hipEventRecord(stop);
                (void)hipEventSynchronize(stop);

                float ms = 0.0f;
                (void)hipEventElapsedTime(&ms, start, stop);
                times_us.push_back(static_cast<double>(ms) * 1000.0);
            }

            (void)hipEventDestroy(start);
            (void)hipEventDestroy(stop);

            std::sort(times_us.begin(), times_us.end());
            return times_us;
        }

        /// Benchmark INT8 VNNI GEMM reference on a specific device.
        /// Returns min kernel time in μs.
        static double benchmarkINT8GEMMReference(int M, int N, int K, int device_id)
        {
            (void)hipSetDevice(device_id);

            auto weights = TestTensorFactory::createQ8_0Random(
                {static_cast<size_t>(N), static_cast<size_t>(K)});
            if (!weights)
                return 0.0;

            ROCmPackedWeights packed;
            if (!packWeightsToROCm(weights.get(), packed))
                return 0.0;

            ROCmQuantisedGemmKernel kernel(&packed, device_id);
            auto reqs = kernel.getWorkspaceRequirements(M, N, K);
            const size_t budget = reqs.total_bytes_with_alignment() + (8 * 1024 * 1024);
            auto workspace = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::rocm(device_id), budget);
            if (!workspace->allocate(reqs))
                return 0.0;
            kernel.bindWorkspace(workspace.get());

            auto input = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(M), static_cast<size_t>(K)});
            auto output = TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(N)});
            if (!input->ensureOnDevice(DeviceId::rocm(device_id)))
                return 0.0;
            if (!output->allocateOnDevice(DeviceId::rocm(device_id)))
                return 0.0;

            auto times = timeGEMMKernel(kernel, input.get(), output.get(), M, N, K, device_id);
            kernel.unbindWorkspace();

            return times.empty() ? 0.0 : times.front(); // min
        }

        /// Run a single format+shape+M benchmark on a specific device.
        /// Thread-safe: does not use gtest assertions (caller validates results).
        static GEMMBenchResult benchmarkGEMM(const GEMMFormatSpec &fmt,
                                             const GEMMShape &shape, int M,
                                             double int8_ref_us,
                                             TensorBase *weights,
                                             const GpuWeightsCache *gpu_weights,
                                             int device_id)
        {
            (void)hipSetDevice(device_id);

            GEMMBenchResult result{};
            result.format_name = fmt.name;
            result.bpw = fmt.bpw;
            result.shape_name = shape.name;
            result.M = M;
            result.N = shape.N;
            result.K = shape.K;

            if (!weights)
                return result;

            // 1. Pack pre-created quantized weights
            ROCmPackedWeights packed;
            if (!packWeightsToROCm(weights, packed))
                return result;
            if (packed.native_vnni_payload.empty())
                return result;

            // Native-VNNI weight bytes: payload + scales
            result.native_weight_bytes =
                static_cast<double>(packed.native_vnni_payload.size()) +
                static_cast<double>(packed.native_vnni_scales.size() * sizeof(uint16_t));

            // 2. Create kernel + workspace
            ROCmQuantisedGemmKernel kernel(&packed, device_id);
            auto reqs = kernel.getWorkspaceRequirements(M, shape.N, shape.K);
            const size_t budget = reqs.total_bytes_with_alignment() + (8 * 1024 * 1024);
            auto workspace = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::rocm(device_id), budget);
            if (!workspace->allocate(reqs))
                return result;
            kernel.bindWorkspace(workspace.get());

            // 3. Create input/output tensors and upload
            auto input = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(M), static_cast<size_t>(shape.K)});
            auto output = TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(shape.N)});
            if (!input->ensureOnDevice(DeviceId::rocm(device_id)))
            {
                kernel.unbindWorkspace();
                return result;
            }
            if (!output->allocateOnDevice(DeviceId::rocm(device_id)))
            {
                kernel.unbindWorkspace();
                return result;
            }

            // 4. Correctness: GPU-based FP32 reference via hipBLAS
            {
                kernel.multiply_tensor(input.get(), output.get(), M, shape.N, shape.K);
                (void)hipDeviceSynchronize();
                output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

                if (gpu_weights && gpu_weights->d_weights)
                {
                    auto *in_fp32 = dynamic_cast<FP32Tensor *>(input.get());
                    const float *d_input = reinterpret_cast<const float *>(in_fp32->gpu_data_ptr());
                    if (d_input)
                    {
                        const size_t out_elems = static_cast<size_t>(M) * shape.N;
                        float *d_ref_output = nullptr;
                        auto hip_err = hipMalloc(&d_ref_output, out_elems * sizeof(float));
                        if (hip_err == hipSuccess)
                        {
                            bool gemm_ok = gpuReferenceFP32Gemm(
                                d_input, gpu_weights->d_weights,
                                d_ref_output, M, shape.N, shape.K, device_id);
                            (void)hipDeviceSynchronize();

                            if (gemm_ok)
                            {
                                const float *d_gpu_output = reinterpret_cast<const float *>(
                                    dynamic_cast<FP32Tensor *>(output.get())->gpu_data_ptr());
                                result.cosine_sim = gpuCosineSimilarity(
                                    d_gpu_output, d_ref_output, out_elems, device_id);
                                result.correctness_pass = (result.cosine_sim >= COSINE_SIM_GATE);
                            }
                            (void)hipFree(d_ref_output);
                        }
                    }
                }
                else
                {
                    // Fallback: CPU reference (slow)
                    const float *w_fp32 = weights->data();
                    if (w_fp32)
                    {
                        auto *out_fp32 = dynamic_cast<FP32Tensor *>(output.get());
                        const float *gpu_data = out_fp32->data();
                        const float *a_data = dynamic_cast<const FP32Tensor *>(input.get())->data();
                        const size_t out_elems = static_cast<size_t>(M) * shape.N;
                        std::vector<float> ref_output(out_elems, 0.0f);

#pragma omp parallel for collapse(2) schedule(static)
                        for (int m = 0; m < M; ++m)
                            for (int n = 0; n < shape.N; ++n)
                            {
                                float acc = 0.0f;
                                for (int ki = 0; ki < shape.K; ++ki)
                                    acc += a_data[m * shape.K + ki] * w_fp32[n * shape.K + ki];
                                ref_output[m * shape.N + n] = acc;
                            }

                        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
#pragma omp parallel for reduction(+ : dot, norm_a, norm_b)
                        for (size_t i = 0; i < out_elems; ++i)
                        {
                            dot += (double)gpu_data[i] * ref_output[i];
                            norm_a += (double)gpu_data[i] * gpu_data[i];
                            norm_b += (double)ref_output[i] * ref_output[i];
                        }
                        double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
                        result.cosine_sim = denom > 0 ? static_cast<float>(dot / denom) : 0.0f;
                        result.correctness_pass = (result.cosine_sim >= COSINE_SIM_GATE);

                        output->ensureOnDevice(DeviceId::rocm(device_id));
                    }
                }
            }

            // 5. Timed runs
            auto times = timeGEMMKernel(kernel, input.get(), output.get(),
                                        M, shape.N, shape.K, device_id);

            double max_us;
            computeStats(times, result.mean_us, result.min_us, max_us, result.stddev_us);

            // 6. Compute derived metrics
            double flops = 2.0 * M * static_cast<double>(shape.N) * shape.K;
            result.gflops = flops / (result.min_us * 1e-6) / 1e9;
            result.theoretical_speedup = 8.0 / result.bpw;

            if (int8_ref_us > 0.0 && result.min_us > 0.0)
            {
                result.int8_min_us = int8_ref_us;
                result.speedup_vs_int8 = int8_ref_us / result.min_us;
                result.kernel_efficiency =
                    (result.speedup_vs_int8 / result.theoretical_speedup) * 100.0;
            }

            kernel.unbindWorkspace();
            return result;
        }

        static std::vector<double> timeDirectNativePrefill(
            const DeviceNativeVNNIMatrixDesc &desc,
            const float *d_input,
            int8_t *d_A_int8,
            float *d_scales_A,
            float *d_scales_A_blockwise,
            float *d_output,
            int M, int N, int K,
            int device_id,
            bool include_quantization)
        {
            (void)hipSetDevice(device_id);

            const auto run_once = [&]() -> bool
            {
                if (include_quantization)
                {
                    if (!rocmQuantGemm_quantizeActivationsBlockwise(
                            d_input,
                            d_A_int8,
                            d_scales_A_blockwise,
                            M, K,
                            device_id,
                            nullptr,
                            32))
                    {
                        return false;
                    }
                }

                return rocmGemm_native_vnni_fp32(
                    d_A_int8,
                    desc.payload,
                    desc.scales,
                    desc.mins,
                    desc.emins,
                    d_output,
                    d_scales_A,
                    d_scales_A_blockwise,
                    M, N, K,
                    desc.codebook_id,
                    device_id,
                    nullptr);
            };

            if (!include_quantization)
            {
                if (!rocmQuantGemm_quantizeActivationsBlockwise(
                        d_input,
                        d_A_int8,
                        d_scales_A_blockwise,
                        M, K,
                        device_id,
                        nullptr,
                        32))
                {
                    return {};
                }
            }

            for (int i = 0; i < WARMUP_RUNS; ++i)
            {
                if (!run_once())
                    return {};
            }
            (void)hipDeviceSynchronize();

            hipEvent_t start = nullptr, stop = nullptr;
            (void)hipEventCreate(&start);
            (void)hipEventCreate(&stop);

            std::vector<double> times_us;
            times_us.reserve(BENCH_RUNS);

            for (int i = 0; i < BENCH_RUNS; ++i)
            {
                (void)hipDeviceSynchronize();
                (void)hipEventRecord(start);
                if (!run_once())
                {
                    (void)hipEventDestroy(start);
                    (void)hipEventDestroy(stop);
                    return {};
                }
                (void)hipEventRecord(stop);
                (void)hipEventSynchronize(stop);

                float ms = 0.0f;
                (void)hipEventElapsedTime(&ms, start, stop);
                times_us.push_back(static_cast<double>(ms) * 1000.0);
            }

            (void)hipEventDestroy(start);
            (void)hipEventDestroy(stop);

            std::sort(times_us.begin(), times_us.end());
            return times_us;
        }

        static DirectPrefillComparisonResult compareSmallMToDirectPrefill(
            const GEMMFormatSpec &fmt,
            const GEMMShape &shape,
            int M,
            int device_id)
        {
            (void)hipSetDevice(device_id);

            DirectPrefillComparisonResult result{};
            result.format_name = fmt.name;
            result.shape_name = shape.name;
            result.M = M;
            result.N = shape.N;
            result.K = shape.K;

            auto weights = fmt.create(
                static_cast<size_t>(shape.N),
                static_cast<size_t>(shape.K));
            if (!weights)
                return result;

            ROCmPackedWeights packed;
            if (!packWeightsToROCm(weights.get(), packed))
                return result;
            if (packed.native_vnni_payload.empty())
                return result;

            ROCmQuantisedGemmKernel kernel(&packed, device_id);
            auto reqs = kernel.getWorkspaceRequirements(M, shape.N, shape.K);
            const size_t budget = reqs.total_bytes_with_alignment() + (8 * 1024 * 1024);
            auto workspace = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::rocm(device_id), budget);
            if (!workspace->allocate(reqs))
                return result;
            kernel.bindWorkspace(workspace.get());

            DeviceNativeVNNIMatrixDesc desc{};
            if (!kernel.exportNativeVNNIMatrixDesc(desc))
            {
                kernel.unbindWorkspace();
                return result;
            }

            auto input = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(M), static_cast<size_t>(shape.K)});
            auto small_m_output = TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(shape.N)});
            auto direct_output = TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(shape.N)});

            if (!input->ensureOnDevice(DeviceId::rocm(device_id)) ||
                !small_m_output->allocateOnDevice(DeviceId::rocm(device_id)) ||
                !direct_output->allocateOnDevice(DeviceId::rocm(device_id)))
            {
                kernel.unbindWorkspace();
                return result;
            }

            auto *input_fp32 = dynamic_cast<FP32Tensor *>(input.get());
            auto *small_fp32 = dynamic_cast<FP32Tensor *>(small_m_output.get());
            auto *direct_fp32 = dynamic_cast<FP32Tensor *>(direct_output.get());
            if (!input_fp32 || !small_fp32 || !direct_fp32)
            {
                kernel.unbindWorkspace();
                return result;
            }

            const float *d_input = reinterpret_cast<const float *>(input_fp32->gpu_data_ptr());
            float *d_small_output = reinterpret_cast<float *>(small_fp32->gpu_data_ptr());
            float *d_direct_output = reinterpret_cast<float *>(direct_fp32->gpu_data_ptr());
            if (!d_input || !d_small_output || !d_direct_output)
            {
                kernel.unbindWorkspace();
                return result;
            }

            HipBuffer d_A_int8;
            HipBuffer d_scales_A;
            HipBuffer d_scales_A_blockwise;

            const size_t activation_count = static_cast<size_t>(M) * shape.K;
            const size_t scale_block_count =
                static_cast<size_t>(M) * static_cast<size_t>(shape.K / 32);
            if (!d_A_int8.allocate(activation_count * sizeof(int8_t)) ||
                !d_scales_A.allocate(static_cast<size_t>(M) * sizeof(float)) ||
                !d_scales_A_blockwise.allocate(scale_block_count * sizeof(float)))
            {
                kernel.unbindWorkspace();
                return result;
            }

            std::vector<float> row_scales(static_cast<size_t>(M), 1.0f);
            if (hipMemcpy(
                    d_scales_A.ptr,
                    row_scales.data(),
                    row_scales.size() * sizeof(float),
                    hipMemcpyHostToDevice) != hipSuccess)
            {
                kernel.unbindWorkspace();
                return result;
            }

            auto small_m_times = timeGEMMKernel(
                kernel,
                input.get(),
                small_m_output.get(),
                M, shape.N, shape.K,
                device_id);

            auto prefill_times = timeDirectNativePrefill(
                desc,
                d_input,
                d_A_int8.as<int8_t>(),
                d_scales_A.as<float>(),
                d_scales_A_blockwise.as<float>(),
                d_direct_output,
                M, shape.N, shape.K,
                device_id,
                false);

            auto prefill_quant_times = timeDirectNativePrefill(
                desc,
                d_input,
                d_A_int8.as<int8_t>(),
                d_scales_A.as<float>(),
                d_scales_A_blockwise.as<float>(),
                d_direct_output,
                M, shape.N, shape.K,
                device_id,
                true);

            kernel.unbindWorkspace();

            if (small_m_times.empty() || prefill_times.empty() || prefill_quant_times.empty())
                return result;

            result.small_m_min_us = small_m_times.front();
            result.direct_prefill_min_us = prefill_times.front();
            result.direct_prefill_quant_min_us = prefill_quant_times.front();
            result.prefill_quant_speedup =
                result.direct_prefill_quant_min_us > 0.0
                    ? result.small_m_min_us / result.direct_prefill_quant_min_us
                    : 0.0;

            const size_t out_elems = static_cast<size_t>(M) * shape.N;
            result.cosine_sim = gpuCosineSimilarity(
                d_small_output,
                d_direct_output,
                out_elems,
                device_id);
            result.correctness_pass = (result.cosine_sim >= MTP_SMALL_M_COSINE_GATE);
            result.valid = true;
            return result;
        }

        static BatchedGDNProjectionResult benchmarkBatchedGDNProjection(
            const std::string &group_name,
            const GEMMFormatSpec &fmt,
            const std::vector<int> &projection_Ns,
            int M,
            int K,
            int device_id,
            const BatchedDecodeVariant &variant = BatchedDecodeVariant{"AUTO", 0, 0, true})
        {
            (void)hipSetDevice(device_id);
            BatchedGDNProjectionResult result{};
            HipStream stream;
            if (!stream.create())
                return result;

            result.group_name = group_name;
            result.format_name = fmt.name;
            result.variant_name = variant.name;
            result.M = M;
            result.K = K;
            result.projections = static_cast<int>(projection_Ns.size());
            result.total_N = std::accumulate(projection_Ns.begin(), projection_Ns.end(), 0);
            result.kb = variant.kb;
            result.target_waves = variant.target_waves;

            if (projection_Ns.empty() ||
                projection_Ns.size() > 8 ||
                M < 2 || M > 4 ||
                K <= 0 || (K % 32) != 0)
            {
                return result;
            }

            std::vector<std::unique_ptr<TensorBase>> weights;
            std::vector<ROCmPackedWeights> packed;
            std::vector<std::unique_ptr<ROCmQuantisedGemmKernel>> kernels;
            std::vector<DeviceNativeVNNIMatrixDesc> descs;
            weights.reserve(projection_Ns.size());
            packed.reserve(projection_Ns.size());
            kernels.reserve(projection_Ns.size());
            descs.reserve(projection_Ns.size());

            for (int N : projection_Ns)
            {
                weights.push_back(fmt.create(static_cast<size_t>(N), static_cast<size_t>(K)));
                if (!weights.back())
                    return result;

                packed.emplace_back();
                if (!packWeightsToROCm(weights.back().get(), packed.back()))
                    return result;
                if (packed.back().native_vnni_payload.empty())
                    return result;

                kernels.push_back(std::make_unique<ROCmQuantisedGemmKernel>(&packed.back(), device_id));
                DeviceNativeVNNIMatrixDesc desc{};
                if (!kernels.back()->exportNativeVNNIMatrixDesc(desc))
                    return result;
                if (!desc.valid())
                    return result;
                if (!descs.empty() && desc.codebook_id != descs.front().codebook_id)
                    return result;
                descs.push_back(desc);
            }

            result.codebook_id = descs.front().codebook_id;

            auto input = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(M), static_cast<size_t>(K)});
            if (!input || !input->ensureOnDevice(DeviceId::rocm(device_id)))
                return result;
            auto *input_fp32 = dynamic_cast<FP32Tensor *>(input.get());
            if (!input_fp32 || !input_fp32->gpu_data_ptr())
                return result;

            HipBuffer d_A_int8;
            HipBuffer d_scales_A;
            HipBuffer d_scales_A_blockwise;
            HipBuffer d_sums_A_blockwise;
            if (!d_A_int8.allocate(static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(int8_t)) ||
                !d_scales_A.allocate(static_cast<size_t>(M) * sizeof(float)) ||
                !d_scales_A_blockwise.allocate(static_cast<size_t>(M) * static_cast<size_t>(K / 32) * sizeof(float)) ||
                !d_sums_A_blockwise.allocate(static_cast<size_t>(M) * static_cast<size_t>(K / 32) * sizeof(int32_t)))
            {
                return result;
            }

            std::vector<float> row_scales(static_cast<size_t>(M), 1.0f);
            if (hipMemcpyAsync(
                    d_scales_A.ptr,
                    row_scales.data(),
                    row_scales.size() * sizeof(float),
                    hipMemcpyHostToDevice,
                    stream.stream) != hipSuccess ||
                hipStreamSynchronize(stream.stream) != hipSuccess)
            {
                return result;
            }

            std::array<HipBuffer, 8> outputs;
            std::array<HipBuffer, 8> refs;
            std::array<HipBuffer, 8> partials;
            std::array<const uint8_t *, 8> payload_ptrs{};
            std::array<const uint16_t *, 8> scale_ptrs{};
            std::array<const uint16_t *, 8> min_ptrs{};
            std::array<const uint32_t *, 8> emin_ptrs{};
            std::array<const float *, 8> bias_ptrs{};
            std::array<float *, 8> output_ptrs{};
            std::array<float *, 8> partial_ptrs{};
            std::array<int, 8> Ns{};

            for (size_t i = 0; i < projection_Ns.size(); ++i)
            {
                const int N = projection_Ns[i];
                const size_t output_bytes =
                    static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float);
                const size_t partial_bytes =
                    static_cast<size_t>(8) * static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float);
                if (!outputs[i].allocate(output_bytes) ||
                    !refs[i].allocate(output_bytes) ||
                    !partials[i].allocate(partial_bytes))
                {
                    return result;
                }

                payload_ptrs[i] = descs[i].payload;
                scale_ptrs[i] = static_cast<const uint16_t *>(descs[i].scales);
                min_ptrs[i] = static_cast<const uint16_t *>(descs[i].mins);
                emin_ptrs[i] = static_cast<const uint32_t *>(descs[i].emins);
                bias_ptrs[i] = nullptr;
                output_ptrs[i] = outputs[i].as<float>();
                partial_ptrs[i] = partials[i].as<float>();
                Ns[i] = N;
            }

            const float *d_input = reinterpret_cast<const float *>(input_fp32->gpu_data_ptr());
            auto run_once = [&]() -> bool
            {
                if (!rocmQuantGemm_quantizeActivationsBlockwiseWithSums(
                        d_input,
                        d_A_int8.as<int8_t>(),
                        d_scales_A_blockwise.as<float>(),
                        d_sums_A_blockwise.as<int32_t>(),
                        M, K,
                        device_id,
                        stream.asVoid(),
                        32))
                {
                    return false;
                }

                return rocmGemv_native_vnni_small_m_batched_fp32_with_sums(
                    d_A_int8.as<int8_t>(),
                    payload_ptrs.data(),
                    scale_ptrs.data(),
                    min_ptrs.data(),
                    emin_ptrs.data(),
                    bias_ptrs.data(),
                    output_ptrs.data(),
                    d_scales_A_blockwise.as<float>(),
                    d_sums_A_blockwise.as<int32_t>(),
                    partial_ptrs.data(),
                    Ns.data(),
                    static_cast<int>(projection_Ns.size()),
                    M, K,
                    result.codebook_id,
                    device_id,
                    stream.asVoid());
            };

            for (int i = 0; i < WARMUP_RUNS; ++i)
            {
                if (!run_once())
                    return result;
            }
            (void)hipStreamSynchronize(stream.stream);

            hipEvent_t start = nullptr;
            hipEvent_t stop = nullptr;
            (void)hipEventCreate(&start);
            (void)hipEventCreate(&stop);
            std::vector<double> times_us;
            times_us.reserve(BENCH_RUNS);
            for (int i = 0; i < BENCH_RUNS; ++i)
            {
                (void)hipStreamSynchronize(stream.stream);
                (void)hipEventRecord(start, stream.stream);
                if (!run_once())
                {
                    (void)hipEventDestroy(start);
                    (void)hipEventDestroy(stop);
                    return result;
                }
                (void)hipEventRecord(stop, stream.stream);
                (void)hipEventSynchronize(stop);

                float ms = 0.0f;
                (void)hipEventElapsedTime(&ms, start, stop);
                times_us.push_back(static_cast<double>(ms) * 1000.0);
            }
            (void)hipEventDestroy(start);
            (void)hipEventDestroy(stop);
            std::sort(times_us.begin(), times_us.end());

            if (times_us.empty())
                return result;
            double max_us = 0.0;
            computeStats(times_us, result.mean_us, result.min_us, max_us, result.stddev_us);

            if (!rocmQuantGemm_quantizeActivationsBlockwiseWithSums(
                    d_input,
                    d_A_int8.as<int8_t>(),
                    d_scales_A_blockwise.as<float>(),
                    d_sums_A_blockwise.as<int32_t>(),
                    M, K,
                    device_id,
                    stream.asVoid(),
                    32))
            {
                return result;
            }
            if (!rocmGemv_native_vnni_small_m_batched_fp32_with_sums(
                    d_A_int8.as<int8_t>(),
                    payload_ptrs.data(),
                    scale_ptrs.data(),
                    min_ptrs.data(),
                    emin_ptrs.data(),
                    bias_ptrs.data(),
                    output_ptrs.data(),
                    d_scales_A_blockwise.as<float>(),
                    d_sums_A_blockwise.as<int32_t>(),
                    partial_ptrs.data(),
                    Ns.data(),
                    static_cast<int>(projection_Ns.size()),
                    M, K,
                    result.codebook_id,
                    device_id,
                    stream.asVoid()))
            {
                return result;
            }
            (void)hipStreamSynchronize(stream.stream);

            // Candidate variants are timed above under their explicit KB/TW
            // override. Correctness must be judged against the canonical
            // single-projection path, not against a reference polluted by the
            // same override. Otherwise the trainer can learn a fast variant
            // that is self-consistent but not verifier-equivalent to the model
            // path used outside the sweep.
            rocmGemv_native_vnni_reset_tuning_overrides();

            float min_cosine = 1.0f;
            for (size_t i = 0; i < projection_Ns.size(); ++i)
            {
                const int N = projection_Ns[i];
                if (!rocmGemm_native_vnni_fp32(
                        d_A_int8.as<int8_t>(),
                        descs[i].payload,
                        descs[i].scales,
                        descs[i].mins,
                        descs[i].emins,
                        refs[i].as<float>(),
                        d_scales_A.as<float>(),
                        d_scales_A_blockwise.as<float>(),
                        M, N, K,
                        result.codebook_id,
                        device_id,
                        stream.asVoid()))
                {
                    return result;
                }
                (void)hipStreamSynchronize(stream.stream);
                const float cosine = gpuCosineSimilarity(
                    outputs[i].as<float>(),
                    refs[i].as<float>(),
                    static_cast<size_t>(M) * static_cast<size_t>(N),
                    device_id);
                min_cosine = std::min(min_cosine, cosine);
            }

            result.min_cosine_sim = min_cosine;
            result.correctness_pass = (min_cosine >= MTP_SMALL_M_COSINE_GATE);
            result.valid = true;
            return result;
        }
#endif
    };

    // =============================================================================
    // Test 1: Correctness validation (Q4_0 and IQ4_NL, all shapes, M=128)
    //
    // Gate: cosine similarity > 0.9999 vs FP32 reference
    // =============================================================================

    TEST_F(NativeVNNIGEMMPerfTest, Correctness_AllFormats_AllShapes)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        const int M = 128;

        fprintf(stderr, "\n[NativeVNNI GEMM] Correctness Test (M=%d) using %d GPU(s)\n",
                M, NUM_GPUS);
        fprintf(stderr, "[NativeVNNI GEMM] Device: %s\n", device_name_.c_str());
        fprintf(stderr, "[NativeVNNI GEMM] Gate: cosine similarity >= %.4f\n",
                COSINE_SIM_GATE);

        // Build work items: (format_idx, shape_idx) pairs
        // Sort by estimated cost (N*K descending) then interleave across GPUs
        struct WorkGroup
        {
            int format_idx;
            int shape_idx;
            double cost;
        };
        std::vector<WorkGroup> groups;
        for (int fi = 0; fi < (int)GEMM_FORMATS.size(); ++fi)
            for (int si = 0; si < (int)GEMM_SHAPES.size(); ++si)
                groups.push_back({fi, si,
                                  (double)GEMM_SHAPES[si].N * GEMM_SHAPES[si].K});

        // Sort by cost descending so round-robin distributes heavy shapes evenly
        std::sort(groups.begin(), groups.end(),
                  [](const WorkGroup &a, const WorkGroup &b)
                  { return a.cost > b.cost; });

        // Round-robin assign to GPUs
        std::vector<std::vector<WorkGroup>> per_gpu(NUM_GPUS);
        for (size_t i = 0; i < groups.size(); ++i)
            per_gpu[i % NUM_GPUS].push_back(groups[i]);

        // Results array indexed by (format_idx * num_shapes + shape_idx)
        const size_t total = GEMM_FORMATS.size() * GEMM_SHAPES.size();
        std::vector<GEMMBenchResult> results(total);
        std::atomic<int> completed{0};

        // Worker function for each GPU
        auto worker = [&](int gpu_id)
        {
            (void)hipSetDevice(gpu_id);
            for (const auto &wg : per_gpu[gpu_id])
            {
                const auto &fmt = GEMM_FORMATS[wg.format_idx];
                const auto &shape = GEMM_SHAPES[wg.shape_idx];

                auto weights = fmt.create(
                    static_cast<size_t>(shape.N), static_cast<size_t>(shape.K));
                GpuWeightsCache gpu_w;
                if (weights)
                {
                    const float *w_fp32 = weights->data();
                    if (w_fp32)
                        gpu_w.upload(w_fp32, shape.N, shape.K, gpu_id);
                }

                auto r = benchmarkGEMM(fmt, shape, M, 0.0, weights.get(), &gpu_w, gpu_id);

                size_t idx = wg.format_idx * GEMM_SHAPES.size() + wg.shape_idx;
                results[idx] = std::move(r);

                int done = ++completed;
                fprintf(stderr, "  [GPU %d] %s/%s cos=%.6f %s  (%d/%zu)\n",
                        gpu_id, fmt.name.c_str(), shape.name.c_str(),
                        results[idx].cosine_sim,
                        results[idx].correctness_pass ? "✓" : "✗",
                        done, total);
            }
        };

        // Launch workers
        std::vector<std::thread> threads;
        for (int g = 0; g < NUM_GPUS; ++g)
            threads.emplace_back(worker, g);
        for (auto &t : threads)
            t.join();

        // Render table and validate
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Format" << "Shape" << "M" << "N" << "K"
              << "Cosine Sim" << "Status" << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(1).set_cell_text_align(fort::text_align::left);
        for (int c = 2; c <= 6; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        int pass_count = 0;
        for (int fi = 0; fi < (int)GEMM_FORMATS.size(); ++fi)
        {
            for (int si = 0; si < (int)GEMM_SHAPES.size(); ++si)
            {
                const auto &r = results[fi * GEMM_SHAPES.size() + si];
                char buf_cos[16];
                snprintf(buf_cos, sizeof(buf_cos), "%.6f", r.cosine_sim);
                const char *status = r.correctness_pass ? "PASS ✓" : "FAIL ✗";
                if (r.correctness_pass)
                    ++pass_count;

                table << r.format_name << r.shape_name
                      << std::to_string(M)
                      << std::to_string(r.N) << std::to_string(r.K)
                      << buf_cos << status << fort::endr;

                EXPECT_GE(r.cosine_sim, COSINE_SIM_GATE)
                    << r.format_name << " " << r.shape_name
                    << " cosine=" << r.cosine_sim;
            }
        }

        table << fort::separator;
        char summary[64];
        snprintf(summary, sizeof(summary), "%d/%zu passed", pass_count, total);
        table << "" << summary << "" << "" << "" << "" << "" << fort::endr;

        fprintf(stderr, "\n%s\n", table.to_string().c_str());
#endif
    }

    // =============================================================================
    // Test 2: Performance matrix — all formats × shapes × M values
    //
    // Gate: 1.8x speedup over INT8 GEMM baseline
    // =============================================================================

    TEST_F(NativeVNNIGEMMPerfTest, Performance_AllFormats_AllShapes)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        fprintf(stderr, "\n[NativeVNNI GEMM] Performance Benchmark using %d GPU(s)\n",
                NUM_GPUS);
        fprintf(stderr, "[NativeVNNI GEMM] Device: %s\n", device_name_.c_str());
        fprintf(stderr, "[NativeVNNI GEMM] %zu formats × %zu shapes × %zu M values\n",
                GEMM_FORMATS.size(), GEMM_SHAPES.size(), M_VALUES.size());
        fprintf(stderr, "[NativeVNNI GEMM] %d warmup + %d timed runs each\n",
                WARMUP_RUNS, BENCH_RUNS);
        fprintf(stderr, "[NativeVNNI GEMM] Performance gate: %.1fx speedup vs INT8 GEMM\n",
                SPEEDUP_GATE);

        // =====================================================================
        // Phase 1: Benchmark INT8 GEMM reference for each (shape, M) — multi-GPU
        // =====================================================================
        fprintf(stderr, "\n[Phase 1] Benchmarking INT8 GEMM reference (V3/V7) on %d GPU(s)...\n",
                NUM_GPUS);

        // Build INT8 work items: (shape_idx, M) pairs
        struct Int8Work
        {
            int shape_idx;
            int M;
            double cost;
        };
        std::vector<Int8Work> int8_work;
        for (int si = 0; si < (int)GEMM_SHAPES.size(); ++si)
            for (int M : M_VALUES)
                int8_work.push_back({si, M, (double)GEMM_SHAPES[si].N * GEMM_SHAPES[si].K * M});

        // Sort by cost descending, then round-robin
        std::sort(int8_work.begin(), int8_work.end(),
                  [](const Int8Work &a, const Int8Work &b)
                  { return a.cost > b.cost; });

        std::vector<std::vector<Int8Work>> int8_per_gpu(NUM_GPUS);
        for (size_t i = 0; i < int8_work.size(); ++i)
            int8_per_gpu[i % NUM_GPUS].push_back(int8_work[i]);

        // Key: "shape_name:M" → min time in μs
        // Pre-allocate indexed by (shape_idx * M_VALUES.size() + m_idx)
        const size_t num_shapes = GEMM_SHAPES.size();
        const size_t num_m = M_VALUES.size();
        std::vector<double> int8_times(num_shapes * num_m, 0.0);
        std::atomic<int> int8_done{0};

        {
            std::vector<std::thread> threads;
            for (int g = 0; g < NUM_GPUS; ++g)
            {
                threads.emplace_back([&, g]()
                                     {
                    (void)hipSetDevice(g);
                    for (const auto &w : int8_per_gpu[g])
                    {
                        double ref_us = benchmarkINT8GEMMReference(
                            w.M, GEMM_SHAPES[w.shape_idx].N,
                            GEMM_SHAPES[w.shape_idx].K, g);

                        // Find m_idx
                        int m_idx = 0;
                        for (int mi = 0; mi < (int)M_VALUES.size(); ++mi)
                            if (M_VALUES[mi] == w.M) { m_idx = mi; break; }

                        int8_times[w.shape_idx * num_m + m_idx] = ref_us;
                        int done = ++int8_done;
                        fprintf(stderr, "  [GPU %d] INT8 %s M=%d: %.1f μs  (%d/%zu)\n",
                                g, GEMM_SHAPES[w.shape_idx].name.c_str(),
                                w.M, ref_us, done, int8_work.size());
                    } });
            }
            for (auto &t : threads)
                t.join();
        }

        // Build lookup map from int8_times
        std::unordered_map<std::string, double> int8_ref;
        for (int si = 0; si < (int)num_shapes; ++si)
            for (int mi = 0; mi < (int)num_m; ++mi)
            {
                std::string key = GEMM_SHAPES[si].name + ":" + std::to_string(M_VALUES[mi]);
                int8_ref[key] = int8_times[si * num_m + mi];
            }

        // =====================================================================
        // Phase 2: Benchmark native-VNNI GEMM — multi-GPU
        // =====================================================================
        fprintf(stderr, "\n[Phase 2] Benchmarking native-VNNI GEMM on %d GPU(s)...\n",
                NUM_GPUS);

        // Work groups: (format_idx, shape_idx) — each group runs 4 M values
        // The GPU cache is per format+shape, so keeping M values together avoids
        // duplicate weight uploads
        struct WorkGroup
        {
            int format_idx;
            int shape_idx;
            double cost; // sum of N*K*M for all M values
        };
        std::vector<WorkGroup> groups;
        for (int fi = 0; fi < (int)GEMM_FORMATS.size(); ++fi)
            for (int si = 0; si < (int)GEMM_SHAPES.size(); ++si)
            {
                double c = 0;
                for (int M : M_VALUES)
                    c += (double)GEMM_SHAPES[si].N * GEMM_SHAPES[si].K * M;
                groups.push_back({fi, si, c});
            }

        std::sort(groups.begin(), groups.end(),
                  [](const WorkGroup &a, const WorkGroup &b)
                  { return a.cost > b.cost; });

        std::vector<std::vector<WorkGroup>> per_gpu(NUM_GPUS);
        for (size_t i = 0; i < groups.size(); ++i)
            per_gpu[i % NUM_GPUS].push_back(groups[i]);

        // Results indexed by (format_idx * num_shapes * num_m + shape_idx * num_m + m_idx)
        const size_t total_results = GEMM_FORMATS.size() * num_shapes * num_m;
        std::vector<GEMMBenchResult> results(total_results);
        std::atomic<int> phase2_done{0};
        const size_t total_groups = groups.size();

        {
            std::vector<std::thread> threads;
            for (int g = 0; g < NUM_GPUS; ++g)
            {
                threads.emplace_back([&, g]()
                                     {
                    (void)hipSetDevice(g);
                    for (const auto &wg : per_gpu[g])
                    {
                        const auto &fmt = GEMM_FORMATS[wg.format_idx];
                        const auto &shape = GEMM_SHAPES[wg.shape_idx];

                        auto weights = fmt.create(
                            static_cast<size_t>(shape.N),
                            static_cast<size_t>(shape.K));
                        GpuWeightsCache gpu_w;
                        if (weights)
                        {
                            const float *w_fp32 = weights->data();
                            if (w_fp32)
                                gpu_w.upload(w_fp32, shape.N, shape.K, g);
                        }

                        for (int mi = 0; mi < (int)num_m; ++mi)
                        {
                            int M = M_VALUES[mi];
                            std::string key = shape.name + ":" + std::to_string(M);
                            double ref_us = int8_ref.count(key) ? int8_ref[key] : 0.0;

                            auto r = benchmarkGEMM(fmt, shape, M, ref_us,
                                                   weights.get(), &gpu_w, g);

                            size_t idx = wg.format_idx * num_shapes * num_m
                                       + wg.shape_idx * num_m + mi;
                            results[idx] = std::move(r);
                        }

                        int done = ++phase2_done;
                        fprintf(stderr, "  [GPU %d] %s/%s done  (%d/%zu groups)\n",
                                g, fmt.name.c_str(), shape.name.c_str(),
                                done, total_groups);
                    } });
            }
            for (auto &t : threads)
                t.join();
        }

        // =====================================================================
        // Phase 3: Per-format summary tables (grouped by M)
        // =====================================================================
        for (int fi = 0; fi < (int)GEMM_FORMATS.size(); ++fi)
        {
            const auto &fmt = GEMM_FORMATS[fi];
            for (int mi = 0; mi < (int)num_m; ++mi)
            {
                int M = M_VALUES[mi];
                fort::utf8_table table;
                table.set_border_style(FT_DOUBLE2_STYLE);

                char title[256];
                snprintf(title, sizeof(title),
                         "%s (%.1f bpw) | M=%d | %d warmup + %d runs",
                         fmt.name.c_str(), fmt.bpw, M, WARMUP_RUNS, BENCH_RUNS);

                table << fort::header
                      << "Shape" << "N" << "K"
                      << "NVNNI μs" << "INT8 μs" << "Speedup"
                      << "Theoret." << "Kern Eff"
                      << "GFLOPS" << "Cosine" << "Gate" << fort::endr;

                table.column(0).set_cell_text_align(fort::text_align::left);
                for (int c = 1; c <= 10; ++c)
                    table.column(c).set_cell_text_align(fort::text_align::right);

                for (int si = 0; si < (int)num_shapes; ++si)
                {
                    size_t idx = fi * num_shapes * num_m + si * num_m + mi;
                    const auto &r = results[idx];

                    char b_min[16], b_int8[16], b_speedup[16];
                    char b_theo[16], b_keff[16], b_gflops[16];
                    char b_cos[16];

                    snprintf(b_min, sizeof(b_min), "%.1f", r.min_us);
                    snprintf(b_int8, sizeof(b_int8), "%.1f", r.int8_min_us);
                    snprintf(b_speedup, sizeof(b_speedup), "%.2fx", r.speedup_vs_int8);
                    snprintf(b_theo, sizeof(b_theo), "%.2fx", r.theoretical_speedup);
                    snprintf(b_keff, sizeof(b_keff), "%.0f%%", r.kernel_efficiency);
                    snprintf(b_gflops, sizeof(b_gflops), "%.0f", r.gflops);
                    snprintf(b_cos, sizeof(b_cos), "%.6f", r.cosine_sim);

                    const char *gate_str = "";
                    if (r.speedup_vs_int8 >= SPEEDUP_GATE && r.correctness_pass)
                        gate_str = "PASS ✓";
                    else if (!r.correctness_pass)
                        gate_str = "COS ✗";
                    else if (r.speedup_vs_int8 < SPEEDUP_GATE)
                        gate_str = "PERF ✗";

                    table << r.shape_name
                          << std::to_string(r.N) << std::to_string(r.K)
                          << b_min << b_int8 << b_speedup
                          << b_theo << b_keff
                          << b_gflops << b_cos << gate_str << fort::endr;
                }

                fprintf(stderr, "\n%s\n%s\n", title, table.to_string().c_str());
            }
        }

        // =====================================================================
        // Phase 4: Grand summary — average across all shapes per format per M
        // =====================================================================
        fprintf(stderr, "\n");
        fort::utf8_table grand;
        grand.set_border_style(FT_DOUBLE2_STYLE);
        grand << fort::header
              << "Format" << "BPW" << "M"
              << "Avg NVNNI μs" << "Avg INT8 μs" << "Avg Speedup"
              << "Theoretical" << "Avg Kern Eff" << "Avg Cosine"
              << fort::endr;

        grand.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 8; ++c)
            grand.column(c).set_cell_text_align(fort::text_align::right);

        for (int fi = 0; fi < (int)GEMM_FORMATS.size(); ++fi)
        {
            const auto &fmt = GEMM_FORMATS[fi];
            for (int mi = 0; mi < (int)num_m; ++mi)
            {
                int M = M_VALUES[mi];
                double tot_min = 0, tot_int8 = 0, tot_speedup = 0;
                double tot_keff = 0, tot_cos = 0;
                int count = 0;

                for (int si = 0; si < (int)num_shapes; ++si)
                {
                    size_t idx = fi * num_shapes * num_m + si * num_m + mi;
                    const auto &r = results[idx];
                    tot_min += r.min_us;
                    tot_int8 += r.int8_min_us;
                    tot_speedup += r.speedup_vs_int8;
                    tot_keff += r.kernel_efficiency;
                    tot_cos += r.cosine_sim;
                    ++count;
                }
                if (count == 0)
                    continue;

                char b_bpw[8], b_m[8], b_min[16], b_int8[16];
                char b_speedup[16], b_theo[16], b_keff[16], b_cos[16];

                snprintf(b_bpw, sizeof(b_bpw), "%.1f", fmt.bpw);
                snprintf(b_m, sizeof(b_m), "%d", M);
                snprintf(b_min, sizeof(b_min), "%.1f", tot_min / count);
                snprintf(b_int8, sizeof(b_int8), "%.1f", tot_int8 / count);
                snprintf(b_speedup, sizeof(b_speedup), "%.2fx", tot_speedup / count);
                snprintf(b_theo, sizeof(b_theo), "%.2fx", 8.0 / fmt.bpw);
                snprintf(b_keff, sizeof(b_keff), "%.0f%%", tot_keff / count);
                snprintf(b_cos, sizeof(b_cos), "%.6f", tot_cos / count);

                grand << fmt.name << b_bpw << b_m
                      << b_min << b_int8 << b_speedup
                      << b_theo << b_keff << b_cos << fort::endr;
            }
        }

        fprintf(stderr,
                "GRAND SUMMARY: NativeVNNI GEMM vs INT8 GEMM (V3/V7)\n%s\n",
                grand.to_string().c_str());
        fprintf(stderr, "Speedup = INT8_time / NativeVNNI_time (>1x = faster)\n");
        fprintf(stderr, "Theoretical = 8.0/BPW = %.2fx (bandwidth-optimal for 4.5 bpw)\n",
                8.0 / 4.5);
        fprintf(stderr, "Performance gate: %.1fx | Correctness gate: cosine >= %.4f\n",
                SPEEDUP_GATE, COSINE_SIM_GATE);
#endif
    }

    // =============================================================================
    // Test 3: Focused benchmark — single shape, all M values, both formats
    //
    // Quick iteration test for kernel tuning on a representative shape.
    // =============================================================================

    TEST_F(NativeVNNIGEMMPerfTest, Focused_3B_FFN_Up)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        const GEMMShape shape{"3B_FFN_Up", 11008, 2048};

        fprintf(stderr, "\n[NativeVNNI GEMM] Focused: %s (N=%d K=%d) using %d GPU(s)\n",
                shape.name.c_str(), shape.N, shape.K, NUM_GPUS);
        fprintf(stderr, "[NativeVNNI GEMM] Device: %s\n", device_name_.c_str());

        // INT8 references for each M — parallelize across GPUs
        std::vector<double> int8_refs(M_VALUES.size(), 0.0);
        {
            std::vector<std::thread> threads;
            for (int mi = 0; mi < (int)M_VALUES.size(); ++mi)
            {
                int g = mi % NUM_GPUS;
                threads.emplace_back([&, mi, g]()
                                     {
                    (void)hipSetDevice(g);
                    int8_refs[mi] = benchmarkINT8GEMMReference(
                        M_VALUES[mi], shape.N, shape.K, g);
                    fprintf(stderr, "  [GPU %d] INT8 ref M=%d: %.1f μs\n",
                            g, M_VALUES[mi], int8_refs[mi]); });
            }
            for (auto &t : threads)
                t.join();
        }

        // Work items: (format_idx, m_idx) — round-robin by cost
        struct WorkItem
        {
            int format_idx;
            int m_idx;
            double cost;
        };
        std::vector<WorkItem> items;
        for (int fi = 0; fi < (int)GEMM_FORMATS.size(); ++fi)
            for (int mi = 0; mi < (int)M_VALUES.size(); ++mi)
                items.push_back({fi, mi, (double)shape.N * shape.K * M_VALUES[mi]});

        std::sort(items.begin(), items.end(),
                  [](const WorkItem &a, const WorkItem &b)
                  { return a.cost > b.cost; });

        std::vector<std::vector<WorkItem>> per_gpu(NUM_GPUS);
        for (size_t i = 0; i < items.size(); ++i)
            per_gpu[i % NUM_GPUS].push_back(items[i]);

        // Results indexed by (format_idx * num_m + m_idx)
        const size_t num_m = M_VALUES.size();
        const size_t total = GEMM_FORMATS.size() * num_m;
        std::vector<GEMMBenchResult> results(total);
        std::atomic<int> done_count{0};

        {
            std::vector<std::thread> threads;
            for (int g = 0; g < NUM_GPUS; ++g)
            {
                threads.emplace_back([&, g]()
                                     {
                    (void)hipSetDevice(g);

                    // Group items by format to share weight cache
                    // Sort this GPU's items by format_idx
                    auto my_items = per_gpu[g];
                    std::sort(my_items.begin(), my_items.end(),
                              [](const WorkItem &a, const WorkItem &b) {
                                  return a.format_idx < b.format_idx;
                              });

                    int cur_fi = -1;
                    std::unique_ptr<TensorBase> weights;
                    GpuWeightsCache gpu_w;

                    for (const auto &wi : my_items)
                    {
                        // Load weights when format changes
                        if (wi.format_idx != cur_fi)
                        {
                            cur_fi = wi.format_idx;
                            gpu_w = GpuWeightsCache(); // release old
                            const auto &fmt = GEMM_FORMATS[cur_fi];
                            weights = fmt.create(
                                static_cast<size_t>(shape.N),
                                static_cast<size_t>(shape.K));
                            if (weights)
                            {
                                const float *w_fp32 = weights->data();
                                if (w_fp32)
                                    gpu_w.upload(w_fp32, shape.N, shape.K, g);
                            }
                        }

                        const auto &fmt = GEMM_FORMATS[wi.format_idx];
                        int M = M_VALUES[wi.m_idx];
                        double ref_us = int8_refs[wi.m_idx];

                        auto r = benchmarkGEMM(fmt, shape, M, ref_us,
                                               weights.get(), &gpu_w, g);

                        size_t idx = wi.format_idx * num_m + wi.m_idx;
                        results[idx] = std::move(r);

                        int d = ++done_count;
                        fprintf(stderr, "  [GPU %d] %s M=%d: %.1f μs (%.2fx) (%d/%zu)\n",
                                g, fmt.name.c_str(), M,
                                results[idx].min_us,
                                results[idx].speedup_vs_int8,
                                d, total);
                    } });
            }
            for (auto &t : threads)
                t.join();
        }

        // Render table
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Format" << "M"
              << "NVNNI μs" << "INT8 μs" << "Speedup"
              << "Theoret." << "Kern Eff"
              << "GFLOPS" << "Cosine" << "Gate" << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 9; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (int fi = 0; fi < (int)GEMM_FORMATS.size(); ++fi)
        {
            for (int mi = 0; mi < (int)num_m; ++mi)
            {
                size_t idx = fi * num_m + mi;
                const auto &r = results[idx];

                char b_m[8], b_min[16], b_int8[16], b_speedup[16];
                char b_theo[16], b_keff[16], b_gflops[16], b_cos[16];

                snprintf(b_m, sizeof(b_m), "%d", r.M);
                snprintf(b_min, sizeof(b_min), "%.1f", r.min_us);
                snprintf(b_int8, sizeof(b_int8), "%.1f", r.int8_min_us);
                snprintf(b_speedup, sizeof(b_speedup), "%.2fx", r.speedup_vs_int8);
                snprintf(b_theo, sizeof(b_theo), "%.2fx", r.theoretical_speedup);
                snprintf(b_keff, sizeof(b_keff), "%.0f%%", r.kernel_efficiency);
                snprintf(b_gflops, sizeof(b_gflops), "%.0f", r.gflops);
                snprintf(b_cos, sizeof(b_cos), "%.6f", r.cosine_sim);

                const char *gate = "";
                if (r.speedup_vs_int8 >= SPEEDUP_GATE && r.correctness_pass)
                    gate = "PASS ✓";
                else if (!r.correctness_pass)
                    gate = "COS ✗";
                else
                    gate = "PERF ✗";

                table << r.format_name << b_m
                      << b_min << b_int8 << b_speedup
                      << b_theo << b_keff
                      << b_gflops << b_cos << gate << fort::endr;

                EXPECT_GE(r.cosine_sim, COSINE_SIM_GATE)
                    << GEMM_FORMATS[fi].name << " M=" << r.M
                    << " cosine=" << r.cosine_sim;
            }
            table << fort::separator;
        }

        fprintf(stderr, "\n%s\n", table.to_string().c_str());
#endif
    }

    TEST_F(NativeVNNIGEMMPerfTest, MTP_SmallM_VerifierShapes_AllFormats)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        fprintf(stderr, "\n[NativeVNNI GEMM] Phase 13.5 MTP small-M verifier benchmark\n");
        fprintf(stderr, "[NativeVNNI GEMM] Device: %s\n", device_name_.c_str());
        fprintf(stderr, "[NativeVNNI GEMM] Formats: %zu, Shapes: %zu, M values: %zu\n",
                GEMM_FORMATS.size(), MTP_SMALL_M_SHAPES.size(), MTP_SMALL_M_VALUES.size());
        fprintf(stderr, "[NativeVNNI GEMM] Correctness gate: cosine >= %.4f\n",
                MTP_SMALL_M_COSINE_GATE);

        struct SmallMResult
        {
            GEMMBenchResult result;
        };

        std::vector<SmallMResult> results;
        results.reserve(GEMM_FORMATS.size() * MTP_SMALL_M_SHAPES.size() * MTP_SMALL_M_VALUES.size());

        for (const auto &shape : MTP_SMALL_M_SHAPES)
        {
            std::unordered_map<int, double> int8_refs;
            for (int M : MTP_SMALL_M_VALUES)
            {
                const double ref_us = benchmarkINT8GEMMReference(M, shape.N, shape.K, 0);
                int8_refs[M] = ref_us;
                fprintf(stderr, "  INT8 ref %s M=%d: %.1f μs\n",
                        shape.name.c_str(), M, ref_us);
            }

            for (const auto &fmt : GEMM_FORMATS)
            {
                auto weights = fmt.create(
                    static_cast<size_t>(shape.N),
                    static_cast<size_t>(shape.K));

                GpuWeightsCache gpu_w;
                if (weights)
                {
                    const float *w_fp32 = weights->data();
                    if (w_fp32)
                        gpu_w.upload(w_fp32, shape.N, shape.K, 0);
                }

                for (int M : MTP_SMALL_M_VALUES)
                {
                    auto r = benchmarkGEMM(
                        fmt, shape, M, int8_refs[M], weights.get(), &gpu_w, 0);
                    r.correctness_pass = (r.cosine_sim >= MTP_SMALL_M_COSINE_GATE);
                    results.push_back({std::move(r)});

                    const auto &last = results.back().result;
                    fprintf(stderr, "  %s/%s M=%d: %.1f μs speedup=%.2fx cosine=%.6f\n",
                            last.format_name.c_str(), last.shape_name.c_str(), last.M,
                            last.min_us, last.speedup_vs_int8, last.cosine_sim);
                }
            }
        }

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Format" << "Shape" << "M"
              << "NVNNI μs" << "INT8 μs" << "Speedup"
              << "GFLOPS" << "Cosine" << "Gate" << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(1).set_cell_text_align(fort::text_align::left);
        for (int c = 2; c <= 8; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &entry : results)
        {
            const auto &r = entry.result;
            char b_m[8], b_min[16], b_int8[16], b_speedup[16], b_gflops[16], b_cos[16];
            snprintf(b_m, sizeof(b_m), "%d", r.M);
            snprintf(b_min, sizeof(b_min), "%.1f", r.min_us);
            snprintf(b_int8, sizeof(b_int8), "%.1f", r.int8_min_us);
            snprintf(b_speedup, sizeof(b_speedup), "%.2fx", r.speedup_vs_int8);
            snprintf(b_gflops, sizeof(b_gflops), "%.0f", r.gflops);
            snprintf(b_cos, sizeof(b_cos), "%.6f", r.cosine_sim);

            table << r.format_name << r.shape_name << b_m
                  << b_min << b_int8 << b_speedup
                  << b_gflops << b_cos
                  << (r.correctness_pass ? "PASS" : "COS FAIL")
                  << fort::endr;

            EXPECT_GE(r.cosine_sim, MTP_SMALL_M_COSINE_GATE)
                << r.format_name << " " << r.shape_name << " M=" << r.M
                << " cosine=" << r.cosine_sim;
        }

        fprintf(stderr, "\n%s\n", table.to_string().c_str());
#endif
    }

    TEST_F(NativeVNNIGEMMPerfTest, MTP_SmallM_BatchedGDNProjectionShapes)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        struct GDNProjectionGroup
        {
            std::string name;
            std::string format;
            std::vector<int> projection_Ns;
        };

        const std::vector<GDNProjectionGroup> groups = {
            {"Qwen36_FFN_Q4K_gate_up", "Q4_K", {17408, 17408}},
            {"Qwen36_FFN_Q5K_gate_up", "Q5_K", {17408, 17408}},
            {"Qwen36_GDN_Q4K_qkv_z", "Q4_K", {10240, 6144}},
            {"Qwen36_GDN_Q5K_qkv_z", "Q5_K", {10240, 6144}},
            {"Qwen36_GDN_Q4_1_qkv_z_a", "Q4_1", {12288, 6144, 1024}},
            {"Qwen36_GDN_Q5_1_z_a", "Q5_1", {10240, 1024}},
        };

        fprintf(stderr, "\n[NativeVNNI GEMM] Phase 13.5 batched GDN projection verifier benchmark\n");
        fprintf(stderr, "[NativeVNNI GEMM] Device: %s\n", device_name_.c_str());
        fprintf(stderr, "[NativeVNNI GEMM] K=5120, heterogeneous projection widths, M in {2,3,4}\n");

        std::vector<BatchedGDNProjectionResult> results;
        for (const auto &group : groups)
        {
            auto fmt_it = std::find_if(
                GEMM_FORMATS.begin(),
                GEMM_FORMATS.end(),
                [&](const GEMMFormatSpec &fmt)
                {
                    return fmt.name == group.format;
                });
            ASSERT_NE(fmt_it, GEMM_FORMATS.end()) << group.format;

            for (int M : MTP_SMALL_M_VALUES)
            {
                auto r = benchmarkBatchedGDNProjection(
                    group.name,
                    *fmt_it,
                    group.projection_Ns,
                    M,
                    5120,
                    0);
                ASSERT_TRUE(r.valid) << group.name << " M=" << M;
                EXPECT_TRUE(r.correctness_pass)
                    << group.name << " M=" << M
                    << " cosine=" << r.min_cosine_sim;
                results.push_back(std::move(r));

                const auto &last = results.back();
                fprintf(stderr,
                        "  %s/%s M=%d projections=%d total_N=%d codebook=%u: %.1f μs cosine=%.6f\n",
                        group.name.c_str(),
                        last.format_name.c_str(),
                        last.M,
                        last.projections,
                        last.total_N,
                        static_cast<unsigned>(last.codebook_id),
                        last.min_us,
                        last.min_cosine_sim);
            }
        }

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Format" << "M" << "Proj" << "Total N"
              << "Codebook" << "Min μs" << "Mean μs"
              << "Cosine" << "Gate" << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 8; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &r : results)
        {
            char b_m[8], b_proj[8], b_total_n[16], b_codebook[8];
            char b_min[16], b_mean[16], b_cos[16];
            snprintf(b_m, sizeof(b_m), "%d", r.M);
            snprintf(b_proj, sizeof(b_proj), "%d", r.projections);
            snprintf(b_total_n, sizeof(b_total_n), "%d", r.total_N);
            snprintf(b_codebook, sizeof(b_codebook), "%u", static_cast<unsigned>(r.codebook_id));
            snprintf(b_min, sizeof(b_min), "%.1f", r.min_us);
            snprintf(b_mean, sizeof(b_mean), "%.1f", r.mean_us);
            snprintf(b_cos, sizeof(b_cos), "%.6f", r.min_cosine_sim);

            table << r.format_name << b_m << b_proj << b_total_n
                  << b_codebook << b_min << b_mean << b_cos
                  << (r.correctness_pass ? "PASS" : "COS FAIL")
                  << fort::endr;
        }

        fprintf(stderr, "\n%s\n", table.to_string().c_str());
#endif
    }

    TEST_F(NativeVNNIGEMMPerfTest, TrainerCsv_BatchedProjectionCodebookTagged)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        struct BatchedProjectionGroup
        {
            std::string name;
            std::string format;
            std::vector<int> projection_Ns;
            int K = 5120;
        };

        const std::vector<BatchedProjectionGroup> groups = {
            {"Qwen36_FFN_Q4K_gate_up", "Q4_K", {17408, 17408}, 5120},
            {"Qwen36_FFN_Q5K_gate_up", "Q5_K", {17408, 17408}, 5120},
            {"Qwen36_GDN_Q4K_qkv_z", "Q4_K", {10240, 6144}, 5120},
            {"Qwen36_GDN_Q5K_qkv_z", "Q5_K", {10240, 6144}, 5120},
            {"Qwen36_GDN_Q4_1_qkv_z_a", "Q4_1", {12288, 6144, 1024}, 5120},
            {"Qwen36_GDN_Q5_1_z_a", "Q5_1", {10240, 1024}, 5120},
        };

        std::set<std::string> group_filters =
            getEnvCsvSet("LLAMINAR_ROCM_NVNNI_BATCHED_DECODE_GROUPS");
        if (group_filters.empty())
            group_filters.insert("qwen36_ffn_q4k_gate_up");
        std::set<std::string> format_filters =
            getEnvCsvSet("LLAMINAR_ROCM_NVNNI_BATCHED_DECODE_FORMATS");
        const std::set<std::string> m_filters =
            getEnvCsvSet("LLAMINAR_ROCM_NVNNI_BATCHED_DECODE_M");
        const int max_cases =
            std::max(1, getEnvInt("LLAMINAR_ROCM_NVNNI_BATCHED_DECODE_MAX_CASES").value_or(1));
        const std::string csv_path =
            getEnvString("LLAMINAR_ROCM_NVNNI_BATCHED_DECODE_CSV");
        const std::vector<BatchedDecodeVariant> variants = getBatchedDecodeVariants();

        std::FILE *csv = nullptr;
        if (!csv_path.empty())
        {
            csv = std::fopen(csv_path.c_str(), "w");
            ASSERT_NE(csv, nullptr)
                << "Failed to open ROCm NativeVNNI batched decode CSV: " << csv_path;
            std::fprintf(
                csv,
                "backend,phase,format,codebook,shape,n,k,m,projections,total_n,projection_ns,codebooks,variant,kb,target_waves,min_us,mean_us,stddev_us,cosine,correctness_pass,is_best\n");
        }

        int executed_cases = 0;
        int executed_rows = 0;
        for (const auto &group : groups)
        {
            if (!shouldRunName(group_filters, group.name))
                continue;
            if (!shouldRunName(format_filters, group.format))
                continue;

            auto fmt_it = std::find_if(
                GEMM_FORMATS.begin(),
                GEMM_FORMATS.end(),
                [&](const GEMMFormatSpec &fmt)
                {
                    return fmt.name == group.format;
                });
            ASSERT_NE(fmt_it, GEMM_FORMATS.end()) << group.format;

            for (int M : MTP_SMALL_M_VALUES)
            {
                if (!m_filters.empty() && m_filters.count(std::to_string(M)) == 0)
                    continue;
                if (executed_cases >= max_cases)
                    break;

                struct VariantRow
                {
                    BatchedDecodeVariant variant;
                    BatchedGDNProjectionResult result;
                };
                std::vector<VariantRow> rows;
                rows.reserve(variants.size());
                for (const auto &variant : variants)
                {
                    BatchedDecodeTuningOverrideGuard guard(variant);
                    rows.push_back(VariantRow{
                        variant,
                        benchmarkBatchedGDNProjection(
                            group.name,
                            *fmt_it,
                            group.projection_Ns,
                            M,
                            group.K,
                            0,
                            variant)});
                }

                int best_index = -1;
                for (size_t row_index = 0; row_index < rows.size(); ++row_index)
                {
                    const auto &candidate = rows[row_index].result;
                    if (!candidate.valid || !candidate.correctness_pass || candidate.min_us <= 0.0)
                        continue;
                    if (best_index < 0 ||
                        candidate.min_us < rows[static_cast<size_t>(best_index)].result.min_us)
                    {
                        best_index = static_cast<int>(row_index);
                    }
                }

                ASSERT_GE(best_index, 0)
                    << "No correct ROCm NativeVNNI batched decode variant for "
                    << group.name << " M=" << M;

                std::ostringstream ns_joined;
                std::ostringstream codebooks_joined;
                for (size_t i = 0; i < group.projection_Ns.size(); ++i)
                {
                    if (i > 0)
                    {
                        ns_joined << '+';
                        codebooks_joined << '+';
                    }
                    ns_joined << group.projection_Ns[i];
                    codebooks_joined << static_cast<unsigned>(rows[static_cast<size_t>(best_index)].result.codebook_id);
                }

                if (csv)
                {
                    for (size_t row_index = 0; row_index < rows.size(); ++row_index)
                    {
                        const auto &row = rows[row_index];
                        const auto &r = row.result;
                        std::fprintf(
                            csv,
                            "rocm,batched_decode,%s,%u,%s,%d,%d,%d,%d,%d,%s,%s,%s,%d,%d,%.3f,%.3f,%.3f,%.6f,%d,%d\n",
                            group.format.c_str(),
                            static_cast<unsigned>(r.codebook_id),
                            group.name.c_str(),
                            r.total_N,
                            r.K,
                            r.M,
                            r.projections,
                            r.total_N,
                            ns_joined.str().c_str(),
                            codebooks_joined.str().c_str(),
                            row.variant.name.c_str(),
                            row.variant.kb,
                            row.variant.target_waves,
                            r.min_us,
                            r.mean_us,
                            r.stddev_us,
                            r.min_cosine_sim,
                            r.correctness_pass ? 1 : 0,
                            best_index >= 0 && static_cast<int>(row_index) == best_index ? 1 : 0);
                        ++executed_rows;
                    }
                    std::fflush(csv);
                }

                const auto &best = rows[static_cast<size_t>(best_index)];
                std::fprintf(
                    stderr,
                    "[ROCmNativeVNNI][BATCHED_DECODE][TRAINER] group=%s format=%s codebook=%u M=%d best=%s time_us=%.3f cosine=%.6f\n",
                    group.name.c_str(),
                    group.format.c_str(),
                    static_cast<unsigned>(best.result.codebook_id),
                    M,
                    best.variant.name.c_str(),
                    best.result.min_us,
                    best.result.min_cosine_sim);
                ASSERT_TRUE(best.result.correctness_pass)
                    << group.name << " M=" << M
                    << " cosine=" << best.result.min_cosine_sim;
                ++executed_cases;
            }
        }

        if (csv)
        {
            std::fclose(csv);
            ASSERT_GT(executed_rows, 0)
                << "ROCm NativeVNNI batched decode trainer CSV had no rows.";
        }
        ASSERT_GT(executed_cases, 0)
            << "No ROCm NativeVNNI batched decode trainer cases selected.";
#endif
    }

    TEST_F(NativeVNNIGEMMPerfTest, MTP_SmallM_DirectPrefillRouteComparison)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        fprintf(stderr, "\n[NativeVNNI GEMM] Phase 13.5 small-M route comparison\n");
        fprintf(stderr, "[NativeVNNI GEMM] Device: %s\n", device_name_.c_str());
        fprintf(stderr, "[NativeVNNI GEMM] Signal: speedup = small-M route / direct prefill+quant route (>1 means direct prefill is faster)\n");

        const std::vector<std::string> selected_formats = {"Q4_1", "Q5_1", "Q4_K", "Q5_K"};
        const std::vector<std::string> selected_shapes = {
            "Qwen36_FFN_DownProjection",
            "Qwen36_GDN_OutputProjection",
        };
        const std::vector<int> selected_m_values = {2, 4};

        std::vector<DirectPrefillComparisonResult> results;
        results.reserve(selected_formats.size() * selected_shapes.size() * selected_m_values.size());

        for (const auto &shape_name : selected_shapes)
        {
            auto shape_it = std::find_if(
                MTP_SMALL_M_SHAPES.begin(),
                MTP_SMALL_M_SHAPES.end(),
                [&](const GEMMShape &shape)
                {
                    return shape.name == shape_name;
                });
            ASSERT_NE(shape_it, MTP_SMALL_M_SHAPES.end()) << shape_name;

            for (const auto &format_name : selected_formats)
            {
                auto fmt_it = std::find_if(
                    GEMM_FORMATS.begin(),
                    GEMM_FORMATS.end(),
                    [&](const GEMMFormatSpec &fmt)
                    {
                        return fmt.name == format_name;
                    });
                ASSERT_NE(fmt_it, GEMM_FORMATS.end()) << format_name;

                for (int M : selected_m_values)
                {
                    auto r = compareSmallMToDirectPrefill(*fmt_it, *shape_it, M, 0);
                    results.push_back(r);

                    fprintf(stderr,
                            "  %s/%s M=%d: small-M=%.1f μs direct=%.1f μs direct+quant=%.1f μs signal=%.2fx cosine=%.6f\n",
                            r.format_name.c_str(),
                            r.shape_name.c_str(),
                            r.M,
                            r.small_m_min_us,
                            r.direct_prefill_min_us,
                            r.direct_prefill_quant_min_us,
                            r.prefill_quant_speedup,
                            r.cosine_sim);
                }
            }
        }

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Format" << "Shape" << "M"
              << "Small-M μs" << "Prefill μs" << "Prefill+Quant μs"
              << "Signal" << "Cosine" << "Gate" << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(1).set_cell_text_align(fort::text_align::left);
        for (int c = 2; c <= 8; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &r : results)
        {
            char b_m[8], b_small[16], b_prefill[16], b_prefill_quant[16], b_signal[16], b_cos[16];
            snprintf(b_m, sizeof(b_m), "%d", r.M);
            snprintf(b_small, sizeof(b_small), "%.1f", r.small_m_min_us);
            snprintf(b_prefill, sizeof(b_prefill), "%.1f", r.direct_prefill_min_us);
            snprintf(b_prefill_quant, sizeof(b_prefill_quant), "%.1f", r.direct_prefill_quant_min_us);
            snprintf(b_signal, sizeof(b_signal), "%.2fx", r.prefill_quant_speedup);
            snprintf(b_cos, sizeof(b_cos), "%.6f", r.cosine_sim);

            table << r.format_name << r.shape_name << b_m
                  << b_small << b_prefill << b_prefill_quant
                  << b_signal << b_cos
                  << (r.correctness_pass ? "PASS" : "COS FAIL")
                  << fort::endr;

            EXPECT_TRUE(r.valid)
                << r.format_name << " " << r.shape_name << " M=" << r.M;
            EXPECT_GE(r.cosine_sim, MTP_SMALL_M_COSINE_GATE)
                << r.format_name << " " << r.shape_name << " M=" << r.M
                << " cosine=" << r.cosine_sim;
        }

        fprintf(stderr, "\n%s\n", table.to_string().c_str());
#endif
    }

} // anonymous namespace
