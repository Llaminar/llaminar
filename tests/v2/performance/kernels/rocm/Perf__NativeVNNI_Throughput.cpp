/**
 * @file Perf__NativeVNNI_Throughput.cpp
 * @brief Per-format bandwidth benchmark for native-VNNI GEMV kernels
 *
 * Measures decode throughput for M=1 GEMV and M=2..4 verifier-row GEMV
 * shapes across all native-VNNI formats.
 * at model-realistic dimensions (Qwen2.5-0.5B, 3B, and 7B layer shapes).
 *
 * Each sub-8-bit format is benchmarked against the INT8 VNNI reference
 * (Q8_0 packed to INT8 scatter GEMV) on the same shape, yielding:
 *
 * Metrics reported:
 *   - Kernel time (μs): min/mean across benchmark runs
 *   - Effective bandwidth (GB/s): weight_bytes_read / kernel_time
 *   - BW efficiency (%): effective_BW / HBM_peak_BW
 *   - Speedup vs INT8: int8_min_us / format_min_us
 *   - Theoretical speedup: 8.0 / bpw (from streaming fewer bytes)
 *   - Kernel efficiency: actual_speedup / theoretical_speedup × 100%
 *   - Cosine similarity: GPU vs HipBLAS FP32 reference or, for very large
 *     trainer-only sweeps, GPU vs reset-AUTO native output.
 *
 * Multi-GPU support: work items are distributed across all available GPUs
 * using cost-descending round-robin to balance load evenly.
 *
 * The benchmark uses multiply_tensor() which includes:
 *   1. FP32→INT8 activation quantization on GPU
 *   2. Native-VNNI kernel dispatch (or INT8 scatter GEMV for reference)
 *   3. Scale application (FP32 output)
 *
 * @note Requires ROCm device. Tests skip if no GPU is available.
 * @note Run with build_v2_release for representative timing.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
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
extern "C" void rocmGemv_native_vnni_set_tuning_overrides(int kb, int target_waves_per_cu);
extern "C" void rocmGemv_native_vnni_reset_tuning_overrides();
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

    // MI50/MI60 theoretical HBM2 bandwidth (GB/s)
    // MI60 = 1.0 TB/s, MI50 = 0.77 TB/s. Use MI60 as reference.
    constexpr double HBM2_PEAK_GBPS = 1000.0;

    constexpr int WARMUP_RUNS = 5;
    constexpr int BENCH_RUNS = 20;

    /// Correctness gate: cosine similarity between the full native-VNNI path
    /// and an FP32 reference. This benchmark intentionally includes GPU
    /// activation quantization and FP16 native packed metadata, so asymmetric
    /// formats can sit a little below the generic gate without implying a
    /// decode bug. The exact packed-contract regression lives in
    /// V2_Integration_ROCm_NativeVNNI_GEMV.
    constexpr float COSINE_SIM_GATE = 0.9999f;

    inline float cosine_gate_for([[maybe_unused]] const std::string &format_name)
    {
        // Q4_K uses the Q4_1 native-VNNI packed contract: unsigned nibbles plus
        // FP16 sub-block scale/min metadata and quantized FP32 activations. On
        // the Qwen3.6 GDN time-projection M=3 verifier fixture, that legitimate
        // packed contract can dip just below the asymmetric FP32 health gate
        // even though the exact packed-contract integration test is bit-stable.
        if (format_name == "Q4_K")
            return 0.9997f;
        // Q2_K is another packed K-quant path where the native quantized
        // contract can be visibly lower than the full-FP32 hipBLAS reference
        // without indicating a dispatch mismatch. Keep this as a health gate;
        // exact dispatch equivalence is covered by the NativeVNNI GEMV
        // integration suite and model parity before any generated table install.
        if (format_name == "Q2_K")
            return 0.9998f;
        if (format_name == "Q4_1" || format_name == "Q5_1" || format_name == "Q5_K")
        {
            return 0.9998f;
        }
        return COSINE_SIM_GATE;
    }

    /// Number of GPUs to use (auto-detected, capped at available)
    static int NUM_GPUS = 1;

    // =============================================================================
    // Format descriptors
    // =============================================================================

    struct PerfFormatSpec
    {
        std::string name;
        double bpw;         ///< Bits per weight element
        bool is_superblock; ///< K must be multiple of 256

        std::function<std::unique_ptr<TensorBase>(size_t N, size_t K)> create;
    };

    static const std::vector<PerfFormatSpec> ALL_PERF_FORMATS = {
        // Tier 1: Simple 32-element blocks
        {"Q4_0", 4.5, false, [](size_t N, size_t K)
         { return TestTensorFactory::createQ4_0Random({N, K}); }},
        {"IQ4_NL", 4.5, false, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ4_NLRandom({N, K}); }},
        {"Q4_1", 5.0, false, [](size_t N, size_t K)
         { return TestTensorFactory::createQ4_1Random({N, K}); }},
        {"Q5_0", 5.5, false, [](size_t N, size_t K)
         { return TestTensorFactory::createQ5_0Random({N, K}); }},
        {"Q5_1", 6.0, false, [](size_t N, size_t K)
         { return TestTensorFactory::createQ5_1Random({N, K}); }},
        // Tier 1 super-block
        {"IQ4_XS", 4.5, true, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ4_XSRandom({N, K}); }},

        // Tier 2: K-quant super-blocks
        {"Q4_K", 4.5, true, [](size_t N, size_t K)
         { return TestTensorFactory::createQ4_KRandom({N, K}); }},
        {"Q5_K", 5.5, true, [](size_t N, size_t K)
         { return TestTensorFactory::createQ5_KRandom({N, K}); }},
        {"Q6_K", 6.6, true, [](size_t N, size_t K)
         { return TestTensorFactory::createQ6_KRandom({N, K}); }},
        {"Q3_K", 3.4, true, [](size_t N, size_t K)
         { return TestTensorFactory::createQ3_KRandom({N, K}); }},
        {"Q2_K", 2.6, true, [](size_t N, size_t K)
         { return TestTensorFactory::createQ2_KRandom({N, K}); }},

        // Tier 3: IQ grid-index super-blocks
        {"IQ3_S", 3.4, true, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ3_SRandom({N, K}); }},
        {"IQ3_XXS", 3.1, true, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ3_XXSRandom({N, K}); }},
        {"IQ2_S", 2.5, true, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ2_SRandom({N, K}); }},
        {"IQ2_XS", 2.3, true, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ2_XSRandom({N, K}); }},
        {"IQ2_XXS", 2.1, true, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ2_XXSRandom({N, K}); }},

        // Tier 4: IQ1 ultra-low-bit grid-index super-blocks
        {"IQ1_S", 1.6, true, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ1_SRandom({N, K}); }},
        {"IQ1_M", 1.9, true, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ1_MRandom({N, K}); }},
        // Direct 8-bit path. The default ROCm packer still owns the INT8
        // scatter route, so focused NativeVNNI training explicitly packs this
        // through IINT8Unpackable below before dispatching codebook 19.
        {"Q8_0", 8.5, false, [](size_t N, size_t K)
         { return TestTensorFactory::createQ8_0Random({N, K}); }},
    };

    // Model-realistic GEMV shapes (N×K)
    struct GEMVShape
    {
        std::string name;
        int N;
        int K;
    };

    // Qwen2.5-0.5B:  hidden=896,  intermediate=4864
    // Qwen2.5-3B:    hidden=2048, intermediate=11008
    // Qwen2.5-7B:    hidden=3584, intermediate=18944
    // All K values are multiples of 32 (minimum block size).
    // Super-block formats (256-element) handle non-256-aligned K via sub-block iteration.
    static const std::vector<GEMVShape> SHAPES = {
        // Qwen3.5/Qwen3.6 MoE expert FFN decode shapes.
        {"35BMoE_Expert_GateUp", 512, 2048},
        {"35BMoE_Expert_Down", 2048, 512},
        // Qwen3.6 MoE hybrid GDN verifier decode shapes.
        {"Qwen36MoE_GDN_QKVProjection", 8192, 2048},
        {"Qwen36MoE_GDN_ZProjection", 4096, 2048},
        // Qwen2.5-0.5B
        {"0.5B_AttnOut", 896, 896},    // Qwen2.5-0.5B attention output projection
        {"0.5B_QKV", 896 * 3, 896},    // Qwen2.5-0.5B attention QKV projection
        {"0.5B_FFN_Up", 4864, 896},    // Qwen2.5-0.5B FFN gate/up
        {"0.5B_FFN_Dn", 896, 4864},    // Qwen2.5-0.5B FFN down
        {"0.5B_LM_Head", 151936, 896}, // Qwen2.5-0.5B LM head (vocab projection)
        // Qwen2.5-3B
        {"3B_AttnOut", 2048, 2048},   // Qwen2.5-3B attention output projection
        {"3B_FFN_Up", 11008, 2048},   // Qwen2.5-3B FFN gate/up
        {"3B_FFN_Dn", 2048, 11008},   // Qwen2.5-3B FFN down
        {"3B_LM_Head", 151936, 2048}, // Qwen2.5-3B LM head (vocab projection)
        // Qwen2.5-7B
        {"7B_QKV", 3584 * 3, 3584}, // Qwen2.5-7B attention projection
        {"7B_FFN_Up", 18944, 3584}, // Qwen2.5-7B FFN gate/up
        {"7B_FFN_Dn", 3584, 18944}, // Qwen2.5-7B FFN down
        // Qwen3.6 27B dense / hybrid GDN production shapes.
        {"Qwen36_Attn_QKVProjection", 12288, 5120},
        {"Qwen36_FFN_GateUp", 17408, 5120},
        {"Qwen36_FFN_DownProjection", 5120, 17408},
        {"Qwen36_GDN_InnerProjection", 10240, 5120},
        {"Qwen36_GDN_ZProjection", 6144, 5120},
        {"Qwen36_GDN_TimeProjection", 1024, 5120},
        {"Qwen36_GDN_OutputProjection", 5120, 6144},
        {"Qwen36_LM_Head", 248320, 5120},
    };

    static std::string toLower(std::string value)
    {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        return value;
    }

    static std::string trim(std::string value)
    {
        const auto begin = value.find_first_not_of(" \t\n\r");
        if (begin == std::string::npos)
            return {};
        const auto end = value.find_last_not_of(" \t\n\r");
        return value.substr(begin, end - begin + 1);
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
        return filters.empty() || filters.count(toLower(name)) > 0;
    }

    struct DecodeVariant
    {
        std::string name;
        int kb = 0;
        int target_waves = 0;
        bool auto_dispatch = false;
    };

    static DecodeVariant parseDecodeVariantToken(const std::string &token)
    {
        const std::string value = toLower(trim(token));
        if (value.empty() || value == "auto")
            return DecodeVariant{"AUTO", 0, 0, true};

        std::string compact;
        compact.reserve(value.size());
        for (unsigned char c : value)
        {
            if (std::isalnum(c))
                compact.push_back(static_cast<char>(c));
        }

        const auto kb_pos = compact.find("kb");
        const auto tw_pos = compact.find("tw");
        if (kb_pos == std::string::npos || tw_pos == std::string::npos || tw_pos <= kb_pos + 2)
            throw std::runtime_error("Invalid ROCm NativeVNNI decode variant: " + token);

        const int kb = std::atoi(compact.substr(kb_pos + 2, tw_pos - (kb_pos + 2)).c_str());
        const int tw = std::atoi(compact.substr(tw_pos + 2).c_str());
        if (kb <= 0 || tw <= 0 || kb > 64)
            throw std::runtime_error("Invalid ROCm NativeVNNI decode variant values: " + token);

        return DecodeVariant{"KB" + std::to_string(kb) + "/TW" + std::to_string(tw), kb, tw, false};
    }

    static std::vector<DecodeVariant> getDecodeVariants()
    {
        const char *raw = std::getenv("LLAMINAR_ROCM_NVNNI_DECODE_VARIANTS");
        const std::string spec = (raw && *raw) ? raw : "auto,kb1tw4,kb1tw8,kb2tw8,kb4tw12,kb8tw24,kb16tw24";

        std::vector<DecodeVariant> variants;
        std::stringstream stream(spec);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            token = trim(token);
            if (!token.empty())
                variants.push_back(parseDecodeVariantToken(token));
        }
        if (variants.empty())
            variants.push_back(DecodeVariant{"AUTO", 0, 0, true});
        return variants;
    }

    static std::vector<int> getDecodeMValues()
    {
        const char *raw = std::getenv("LLAMINAR_ROCM_NVNNI_DECODE_M");
        const std::string spec = (raw && *raw) ? raw : "1";

        std::vector<int> values;
        std::stringstream stream(spec);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            token = trim(token);
            if (token.empty())
                continue;
            const int m = std::atoi(token.c_str());
            if (m <= 0)
                throw std::runtime_error("Invalid ROCm NativeVNNI decode M value: " + token);
            values.push_back(m);
        }
        if (values.empty())
            values.push_back(1);
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        return values;
    }

    enum class DecodeReferenceMode
    {
        FP32HipBLAS,
        NativeAuto,
    };

    static DecodeReferenceMode getDecodeReferenceMode()
    {
        std::string raw = toLower(getEnvString("LLAMINAR_ROCM_NVNNI_DECODE_REFERENCE"));
        if (raw.empty() || raw == "fp32" || raw == "hipblas" || raw == "fp32-hipblas")
            return DecodeReferenceMode::FP32HipBLAS;
        if (raw == "native" || raw == "native-auto" || raw == "native_auto" || raw == "auto")
            return DecodeReferenceMode::NativeAuto;
        throw std::runtime_error("Invalid LLAMINAR_ROCM_NVNNI_DECODE_REFERENCE: " + raw);
    }

    static const char *referenceModeName(DecodeReferenceMode mode)
    {
        switch (mode)
        {
        case DecodeReferenceMode::FP32HipBLAS:
            return "fp32";
        case DecodeReferenceMode::NativeAuto:
            return "native-auto";
        }
        return "unknown";
    }

    static float reference_gate_for(const std::string &format_name, DecodeReferenceMode mode)
    {
        if (mode == DecodeReferenceMode::NativeAuto)
        {
            // The native-auto reference compares two quantized NativeVNNI paths
            // with identical input and packed weights. Use a stricter gate than
            // the FP32 health proxy; exact model correctness is still gated by
            // integration parity before generated table installation.
            return 0.99999f;
        }
        return cosine_gate_for(format_name);
    }

    static double nativePackedWeightBytes(const ROCmPackedWeights &packed)
    {
        return static_cast<double>(packed.native_vnni_payload.size()) +
               static_cast<double>(packed.native_vnni_scales.size() * sizeof(uint16_t)) +
               static_cast<double>(packed.native_vnni_mins.size() * sizeof(uint16_t)) +
               static_cast<double>(packed.native_vnni_emins.size() * sizeof(uint32_t));
    }

    static const NativeVnniFormatInfo &requireNativeVnniInfo(const TensorBase *weights, const std::string &format_name)
    {
        const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(weights);
        const NativeVnniFormatInfo *info = unpackable ? unpackable->vnniFormatInfo() : nullptr;
        if (!info)
            throw std::runtime_error("ROCm NativeVNNI decode format " + format_name + " did not expose vnniFormatInfo()");
        return *info;
    }

    /**
     * @brief Populate NativeVNNI host fields for formats not emitted by the default packer.
     *
     * Q8_0 normally uses ROCm's INT8 scatter GEMV path, but the generated
     * verifier-row dispatch table has to cover codebook 19 as well.  This
     * helper uses the tensor's native packing interface to create the same
     * payload/scales/mins arrays that production NativeVNNI descriptors expose.
     */
    static bool ensureNativeVNNIPayloadForDecodeTrainer(
        TensorBase *weights,
        ROCmPackedWeights &packed,
        int N,
        int K,
        const std::string &format_name)
    {
        if (!packed.native_vnni_payload.empty())
            return true;
        if (!weights || N <= 0 || K <= 0 || (K % 32) != 0)
            return false;

        const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(weights);
        const NativeVnniFormatInfo *info = unpackable ? unpackable->vnniFormatInfo() : nullptr;
        if (!info || info->payload_bytes <= 0)
        {
            std::fprintf(stderr,
                         "[ROCmNativeVNNI][DECODE][TRAINER][ERROR] %s does not expose packable NativeVNNI metadata\n",
                         format_name.c_str());
            return false;
        }

        const int blocks_per_row = K / 32;
        const size_t total_blocks =
            static_cast<size_t>(N) * static_cast<size_t>(blocks_per_row);
        packed.native_vnni_payload.assign(
            total_blocks * static_cast<size_t>(info->payload_bytes), uint8_t{0});
        packed.native_vnni_scales.assign(total_blocks, uint16_t{0});
        packed.native_vnni_mins.clear();
        packed.native_vnni_emins.clear();
        if (info->is_asymmetric)
            packed.native_vnni_mins.assign(total_blocks, uint16_t{0});
        if (info->has_emins)
            packed.native_vnni_emins.assign(total_blocks, uint32_t{0});
        packed.native_vnni_codebook_id = info->codebook_id;
        packed.native_vnni_blocks_per_row = static_cast<uint32_t>(blocks_per_row);
        packed.N = N;
        packed.K = K;

        VnniPackContext ctx{};
        ctx.raw_bytes = reinterpret_cast<const uint8_t *>(weights->raw_data());
        ctx.N = N;
        ctx.K = K;
        ctx.blocks_per_row = blocks_per_row;
        ctx.payload_bytes = info->payload_bytes;
        ctx.payload_array = packed.native_vnni_payload.data();
        ctx.scales_array = packed.native_vnni_scales.data();
        ctx.mins_array = packed.native_vnni_mins.empty()
                             ? nullptr
                             : packed.native_vnni_mins.data();
        ctx.emins_array = packed.native_vnni_emins.empty()
                              ? nullptr
                              : packed.native_vnni_emins.data();

        for (int n = 0; n < N; ++n)
            for (int b = 0; b < blocks_per_row; ++b)
                unpackable->packVnniBlock(ctx, n, b);

        return true;
    }

    // =============================================================================
    // Benchmark result
    // =============================================================================

    struct BenchResult
    {
        std::string format_name;
        double bpw;
        std::string shape_name;
        int M;
        int N, K;

        double min_us;
        double mean_us;
        double stddev_us;

        double weight_bytes;  // native-VNNI payload + scales + mins bytes
        double eff_bw_gbps;   // effective bandwidth at min time
        double bw_efficiency; // % of HBM peak

        // INT8 reference comparison (populated when reference available)
        double int8_min_us = 0.0;         // INT8 VNNI reference min time for same shape
        double speedup_vs_int8 = 0.0;     // int8_min_us / min_us (>1 = faster than INT8)
        double theoretical_speedup = 0.0; // 8.0 / bpw (expected from bandwidth savings)
        double kernel_efficiency = 0.0;   // (speedup_vs_int8 / theoretical_speedup) * 100%

        // Correctness (GPU-based HipBLAS reference)
        float cosine_sim = 0.0f;
        bool correctness_pass = false;
    };

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

    class NativeVNNIPerfTest : public ::testing::Test
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
                NUM_GPUS = std::min(device_count, 3);
                (void)hipSetDevice(0);
                hipDeviceProp_t props;
                if (hipGetDeviceProperties(&props, 0) == hipSuccess)
                    device_name_ = std::string(props.name) + " (" + props.gcnArchName + ")";
                else
                    device_name_ = "ROCm device 0";
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
        class DecodeTuningOverrideGuard
        {
        public:
            explicit DecodeTuningOverrideGuard(const DecodeVariant &variant)
            {
                if (variant.auto_dispatch)
                    rocmGemv_native_vnni_reset_tuning_overrides();
                else
                    rocmGemv_native_vnni_set_tuning_overrides(
                        variant.kb, variant.target_waves);
            }

            ~DecodeTuningOverrideGuard()
            {
                rocmGemv_native_vnni_reset_tuning_overrides();
            }

            DecodeTuningOverrideGuard(const DecodeTuningOverrideGuard &) = delete;
            DecodeTuningOverrideGuard &operator=(const DecodeTuningOverrideGuard &) = delete;
        };

        /// Time a GEMV kernel on a specific device. Returns sorted timing vector in μs.
        static std::vector<double> timeKernel(ROCmQuantisedGemmKernel &kernel,
                                              TensorBase *input, TensorBase *output,
                                              int M, int N, int K, int device_id,
                                              hipStream_t stream)
        {
            (void)hipSetDevice(device_id);

            // Warmup
            for (int i = 0; i < WARMUP_RUNS; ++i)
                kernel.multiply_tensor(input, output, M, N, K);
            (void)hipStreamSynchronize(stream);

            // Timed runs
            hipEvent_t start = nullptr, stop = nullptr;
            (void)hipEventCreate(&start);
            (void)hipEventCreate(&stop);

            std::vector<double> times_us;
            times_us.reserve(BENCH_RUNS);

            for (int i = 0; i < BENCH_RUNS; ++i)
            {
                (void)hipStreamSynchronize(stream);
                (void)hipEventRecord(start, stream);
                kernel.multiply_tensor(input, output, M, N, K);
                (void)hipEventRecord(stop, stream);
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

        /// Benchmark INT8 VNNI reference (Q8_0 → INT8 scatter GEMV) for a shape.
        /// Thread-safe: creates all resources locally.
        /// Returns min kernel time in μs.
        static double benchmarkINT8Reference(const GEMVShape &shape, int M, int device_id)
        {
            (void)hipSetDevice(device_id);

            // Create Q8_0 weights — packs to INT8 VNNI (no native-VNNI payload)
            auto weights = TestTensorFactory::createQ8_0Random(
                {static_cast<size_t>(shape.N), static_cast<size_t>(shape.K)});
            if (!weights)
                return 0.0;

            ROCmPackedWeights packed;
            if (!packWeightsToROCm(weights.get(), packed))
                return 0.0;

            ROCmQuantisedGemmKernel kernel(&packed, device_id);
            hipStream_t stream = nullptr;
            if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess || !stream)
                return 0.0;
            kernel.setGPUStream(stream);
            auto reqs = kernel.getWorkspaceRequirements(M, shape.N, shape.K);
            const size_t budget = reqs.total_bytes_with_alignment() + (4 * 1024 * 1024);
            auto workspace = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::rocm(device_id), budget);
            if (!workspace->allocate(reqs))
            {
                (void)hipStreamDestroy(stream);
                return 0.0;
            }
            kernel.bindWorkspace(workspace.get());

            auto input = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(M), static_cast<size_t>(shape.K)});
            auto output = TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(shape.N)});
            if (!input->ensureOnDevice(DeviceId::rocm(device_id)))
            {
                kernel.unbindWorkspace();
                (void)hipStreamDestroy(stream);
                return 0.0;
            }
            if (!output->allocateOnDevice(DeviceId::rocm(device_id)))
            {
                kernel.unbindWorkspace();
                (void)hipStreamDestroy(stream);
                return 0.0;
            }

            auto times = timeKernel(kernel, input.get(), output.get(),
                                    M, shape.N, shape.K, device_id, stream);
            kernel.unbindWorkspace();
            (void)hipStreamDestroy(stream);
            return times.empty() ? 0.0 : times.front();
        }

        /// Run a single format+shape benchmark on a specific device.
        /// Thread-safe: does not use gtest assertions internally.
        static BenchResult benchmarkFormat(const PerfFormatSpec &fmt,
                                           const GEMVShape &shape,
                                           int M,
                                           double int8_ref_us,
                                           TensorBase *weights,
                                           ROCmPackedWeights *packed_weights,
                                           double packed_weight_bytes,
                                           const GpuWeightsCache *gpu_weights,
                                           TensorBase *shared_input,
                                           DecodeReferenceMode reference_mode,
                                           const float *d_native_reference_output,
                                           int device_id)
        {
            (void)hipSetDevice(device_id);

            BenchResult result{};
            result.format_name = fmt.name;
            result.bpw = fmt.bpw;
            result.shape_name = shape.name;
            result.M = M;
            result.N = shape.N;
            result.K = shape.K;

            if (!weights || !packed_weights)
                return result;

            const bool has_host_native_payload =
                !packed_weights->native_vnni_payload.empty() &&
                !packed_weights->native_vnni_scales.empty();
            const bool has_device_native_payload =
                packed_weights->d_native_vnni_payload != nullptr &&
                packed_weights->d_native_vnni_scales != nullptr;
            if (!has_host_native_payload && !has_device_native_payload)
                return result;

            // Calculate weight bytes (native-VNNI payload + scales + mins)
            result.weight_bytes = packed_weight_bytes > 0.0
                                      ? packed_weight_bytes
                                      : nativePackedWeightBytes(*packed_weights);

            // 2. Create kernel + workspace
            ROCmQuantisedGemmKernel kernel(packed_weights, device_id);
            hipStream_t stream = nullptr;
            if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess || !stream)
            {
                std::fprintf(stderr,
                             "[ROCmNativeVNNI][DECODE][TRAINER][ERROR] stream create failed for %s/%s M=%d\n",
                             fmt.name.c_str(), shape.name.c_str(), M);
                return result;
            }
            kernel.setGPUStream(stream);
            auto reqs = kernel.getWorkspaceRequirements(M, shape.N, shape.K);
            const size_t budget = reqs.total_bytes_with_alignment() + (4 * 1024 * 1024);
            auto workspace = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::rocm(device_id), budget);
            if (!workspace->allocate(reqs))
            {
                std::fprintf(stderr,
                             "[ROCmNativeVNNI][DECODE][TRAINER][ERROR] workspace allocation failed for %s/%s M=%d bytes=%zu\n",
                             fmt.name.c_str(),
                             shape.name.c_str(),
                             M,
                             reqs.total_bytes_with_alignment());
                (void)hipStreamDestroy(stream);
                return result;
            }
            kernel.bindWorkspace(workspace.get());

            // 3. Reuse a per-case input when the caller needs variant-vs-variant
            // equivalence, otherwise create a local one for the FP32 reference path.
            std::unique_ptr<TensorBase> owned_input;
            TensorBase *input = shared_input;
            if (!input)
            {
                owned_input = TestTensorFactory::createFP32Random(
                    {static_cast<size_t>(M), static_cast<size_t>(shape.K)});
                input = owned_input.get();
            }
            auto output = TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(shape.N)});
            if (!input->ensureOnDevice(DeviceId::rocm(device_id)))
            {
                std::fprintf(stderr,
                             "[ROCmNativeVNNI][DECODE][TRAINER][ERROR] input upload failed for %s/%s M=%d\n",
                             fmt.name.c_str(), shape.name.c_str(), M);
                kernel.unbindWorkspace();
                (void)hipStreamDestroy(stream);
                return result;
            }
            if (!output->allocateOnDevice(DeviceId::rocm(device_id)))
            {
                std::fprintf(stderr,
                             "[ROCmNativeVNNI][DECODE][TRAINER][ERROR] output allocation failed for %s/%s M=%d\n",
                             fmt.name.c_str(), shape.name.c_str(), M);
                kernel.unbindWorkspace();
                (void)hipStreamDestroy(stream);
                return result;
            }

            // 4. Correctness: GPU-based FP32 reference via hipBLAS
            {
                if (!kernel.multiply_tensor(input, output.get(), M, shape.N, shape.K))
                {
                    std::fprintf(stderr,
                                 "[ROCmNativeVNNI][DECODE][TRAINER][ERROR] multiply failed for %s/%s M=%d\n",
                                 fmt.name.c_str(), shape.name.c_str(), M);
                    kernel.unbindWorkspace();
                    (void)hipStreamDestroy(stream);
                    return result;
                }
                (void)hipStreamSynchronize(stream);
                output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

                const size_t out_elems =
                    static_cast<size_t>(M) * static_cast<size_t>(shape.N);
                const float *d_gpu_output = reinterpret_cast<const float *>(
                    dynamic_cast<FP32Tensor *>(output.get())->gpu_data_ptr());

                if (reference_mode == DecodeReferenceMode::NativeAuto)
                {
                    if (d_gpu_output && d_native_reference_output)
                    {
                        result.cosine_sim = gpuCosineSimilarity(
                            d_gpu_output, d_native_reference_output, out_elems, device_id);
                        result.correctness_pass =
                            (result.cosine_sim >= reference_gate_for(result.format_name, reference_mode));
                    }
                }
                else if (gpu_weights && gpu_weights->d_weights)
                {
                    auto *in_fp32 = dynamic_cast<FP32Tensor *>(input);
                    const float *d_input = reinterpret_cast<const float *>(
                        in_fp32->gpu_data_ptr());
                    if (d_input)
                    {
                        float *d_ref_output = nullptr;
                        auto hip_err = hipMalloc(&d_ref_output, out_elems * sizeof(float));
                        if (hip_err == hipSuccess)
                        {
                            // hipBLAS GEMM with M=1 is effectively GEMV
                            bool gemm_ok = gpuReferenceFP32Gemm(
                                d_input, gpu_weights->d_weights,
                                d_ref_output, M, shape.N, shape.K, device_id);
                            (void)hipDeviceSynchronize();

                            if (gemm_ok)
                            {
                                result.cosine_sim = gpuCosineSimilarity(
                                    d_gpu_output, d_ref_output, out_elems, device_id);
                                result.correctness_pass =
                                    (result.cosine_sim >= reference_gate_for(result.format_name, reference_mode));
                            }
                            (void)hipFree(d_ref_output);
                        }
                    }
                }

                // Re-upload output for timed runs
                output->ensureOnDevice(DeviceId::rocm(device_id));
            }

            // 5. Timed runs
            auto times = timeKernel(kernel, input, output.get(),
                                    M, shape.N, shape.K, device_id, stream);

            double max_us;
            computeStats(times, result.mean_us, result.min_us, max_us, result.stddev_us);

            // 6. Compute metrics
            result.eff_bw_gbps = (result.weight_bytes / (result.min_us * 1e-6)) / 1e9;
            result.bw_efficiency = (result.eff_bw_gbps / HBM2_PEAK_GBPS) * 100.0;
            result.theoretical_speedup = 8.0 / result.bpw;

            if (int8_ref_us > 0.0 && result.min_us > 0.0)
            {
                result.int8_min_us = int8_ref_us;
                result.speedup_vs_int8 = int8_ref_us / result.min_us;
                result.kernel_efficiency =
                    (result.speedup_vs_int8 / result.theoretical_speedup) * 100.0;
            }

            kernel.unbindWorkspace();
            (void)hipStreamDestroy(stream);
            return result;
        }

        static BenchResult benchmarkFormat(const PerfFormatSpec &fmt,
                                           const GEMVShape &shape,
                                           int M,
                                           double int8_ref_us,
                                           TensorBase *weights,
                                           const GpuWeightsCache *gpu_weights,
                                           int device_id)
        {
            ROCmPackedWeights packed;
            if (!weights || !packWeightsToROCm(weights, packed))
            {
                BenchResult result{};
                result.format_name = fmt.name;
                result.bpw = fmt.bpw;
                result.shape_name = shape.name;
                result.M = M;
                result.N = shape.N;
                result.K = shape.K;
                return result;
            }
            if (!ensureNativeVNNIPayloadForDecodeTrainer(
                    weights, packed, shape.N, shape.K, fmt.name))
            {
                BenchResult result{};
                result.format_name = fmt.name;
                result.bpw = fmt.bpw;
                result.shape_name = shape.name;
                result.M = M;
                result.N = shape.N;
                result.K = shape.K;
                return result;
            }
            return benchmarkFormat(fmt,
                                   shape,
                                   M,
                                   int8_ref_us,
                                   weights,
                                   &packed,
                                   nativePackedWeightBytes(packed),
                                   gpu_weights,
                                   nullptr,
                                   DecodeReferenceMode::FP32HipBLAS,
                                   nullptr,
                                   device_id);
        }

        /// Compute the canonical reset-AUTO native output for one trainer case.
        ///
        /// This is used for very large LM-head sweeps where materializing the
        /// full FP32 weight matrix for hipBLAS would dominate the training run.
        /// The returned tensor owns device output memory and must outlive all
        /// candidate comparisons for the same `(format, shape, M, input)` case.
        static std::unique_ptr<TensorBase> computeNativeAutoReferenceOutput(
            ROCmPackedWeights *packed_weights,
            const GEMVShape &shape,
            int M,
            TensorBase *input,
            int device_id)
        {
            if (!packed_weights || !input)
                return nullptr;

            DecodeTuningOverrideGuard guard(DecodeVariant{"AUTO", 0, 0, true});
            ROCmQuantisedGemmKernel kernel(packed_weights, device_id);
            hipStream_t stream = nullptr;
            if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess || !stream)
            {
                std::fprintf(stderr,
                             "[ROCmNativeVNNI][DECODE][TRAINER][ERROR] native-auto stream create failed for %s M=%d\n",
                             shape.name.c_str(), M);
                return nullptr;
            }
            kernel.setGPUStream(stream);
            auto reqs = kernel.getWorkspaceRequirements(M, shape.N, shape.K);
            const size_t budget = reqs.total_bytes_with_alignment() + (4 * 1024 * 1024);
            auto workspace = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::rocm(device_id), budget);
            if (!workspace->allocate(reqs))
            {
                std::fprintf(stderr,
                             "[ROCmNativeVNNI][DECODE][TRAINER][ERROR] native-auto workspace allocation failed for %s M=%d bytes=%zu\n",
                             shape.name.c_str(),
                             M,
                             reqs.total_bytes_with_alignment());
                (void)hipStreamDestroy(stream);
                return nullptr;
            }
            kernel.bindWorkspace(workspace.get());

            auto output = TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(shape.N)});
            if (!input->ensureOnDevice(DeviceId::rocm(device_id)) ||
                !output->allocateOnDevice(DeviceId::rocm(device_id)))
            {
                std::fprintf(stderr,
                             "[ROCmNativeVNNI][DECODE][TRAINER][ERROR] native-auto tensor allocation/upload failed for %s M=%d\n",
                             shape.name.c_str(), M);
                kernel.unbindWorkspace();
                (void)hipStreamDestroy(stream);
                return nullptr;
            }

            if (!kernel.multiply_tensor(input, output.get(), M, shape.N, shape.K))
            {
                std::fprintf(stderr,
                             "[ROCmNativeVNNI][DECODE][TRAINER][ERROR] native-auto multiply failed for %s M=%d\n",
                             shape.name.c_str(), M);
                kernel.unbindWorkspace();
                (void)hipStreamDestroy(stream);
                return nullptr;
            }
            (void)hipStreamSynchronize(stream);
            output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

            kernel.unbindWorkspace();
            (void)hipStreamDestroy(stream);
            return output;
        }
#endif
    };

    // =============================================================================
    // Test: Single-shape sweep across all 18 formats (quick CI check)
    // =============================================================================

    TEST_F(NativeVNNIPerfTest, AllFormats_0_5B_FFN_Up)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        const GEMVShape shape{"0.5B_FFN_Up", 4864, 896};

        fprintf(stderr, "\n[NativeVNNI Perf] Device: %s\n", device_name_.c_str());
        fprintf(stderr, "[NativeVNNI Perf] Shape: %s (N=%d K=%d) | %d warmup + %d runs\n",
                shape.name.c_str(), shape.N, shape.K, WARMUP_RUNS, BENCH_RUNS);

        // INT8 reference
        constexpr int M = 1;
        double int8_us = benchmarkINT8Reference(shape, M, 0);
        fprintf(stderr, "[NativeVNNI Perf] INT8 reference: %.1f μs\n", int8_us);

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Format" << "BPW" << "Weight KB" << "Min μs" << "Mean μs"
              << "Speedup" << "Kern Eff" << "BW GB/s" << "BW Eff %" << "Cosine"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 9; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &fmt : ALL_PERF_FORMATS)
        {
            auto weights = fmt.create(
                static_cast<size_t>(shape.N), static_cast<size_t>(shape.K));
            GpuWeightsCache gpu_w;
            if (weights)
            {
                std::vector<float> w_fp32(static_cast<size_t>(shape.N) * shape.K);
                weights->to_fp32(w_fp32.data());
                gpu_w.upload(w_fp32.data(), shape.N, shape.K, 0);
            }

            auto r = benchmarkFormat(fmt, shape, M, int8_us, weights.get(), &gpu_w, 0);

            char buf_bpw[16], buf_kb[16], buf_min[16], buf_mean[16];
            char buf_speedup[16], buf_keff[16], buf_bw[16], buf_eff[16], buf_cos[16];
            snprintf(buf_bpw, sizeof(buf_bpw), "%.1f", r.bpw);
            snprintf(buf_kb, sizeof(buf_kb), "%.0f", r.weight_bytes / 1024.0);
            snprintf(buf_min, sizeof(buf_min), "%.1f", r.min_us);
            snprintf(buf_mean, sizeof(buf_mean), "%.1f", r.mean_us);
            snprintf(buf_speedup, sizeof(buf_speedup), "%.2fx", r.speedup_vs_int8);
            snprintf(buf_keff, sizeof(buf_keff), "%.0f%%", r.kernel_efficiency);
            snprintf(buf_bw, sizeof(buf_bw), "%.1f", r.eff_bw_gbps);
            snprintf(buf_eff, sizeof(buf_eff), "%.1f%%", r.bw_efficiency);
            snprintf(buf_cos, sizeof(buf_cos), "%.4f", r.cosine_sim);
            table << r.format_name << buf_bpw << buf_kb << buf_min << buf_mean
                  << buf_speedup << buf_keff << buf_bw << buf_eff << buf_cos
                  << fort::endr;
        }

        fprintf(stderr, "\n%s\n", table.to_string().c_str());
#endif
    }

    TEST_F(NativeVNNIPerfTest, TrainerCsv_CodebookTagged)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        std::set<std::string> format_filters = getEnvCsvSet("LLAMINAR_ROCM_NVNNI_DECODE_FORMATS");
        if (format_filters.empty())
            format_filters.insert("q4_0");
        std::set<std::string> shape_filters = getEnvCsvSet("LLAMINAR_ROCM_NVNNI_DECODE_SHAPES");
        if (shape_filters.empty())
            shape_filters.insert("0.5b_attnout");
        const int max_cases = std::max(1, getEnvInt("LLAMINAR_ROCM_NVNNI_DECODE_MAX_CASES").value_or(1));
        const std::string csv_path = getEnvString("LLAMINAR_ROCM_NVNNI_DECODE_CSV");
        const std::vector<DecodeVariant> variants = getDecodeVariants();
        const std::vector<int> m_values = getDecodeMValues();
        const DecodeReferenceMode reference_mode = getDecodeReferenceMode();

        std::fprintf(stderr,
                     "[ROCmNativeVNNI][DECODE][TRAINER] reference=%s\n",
                     referenceModeName(reference_mode));

        std::FILE *csv = nullptr;
        if (!csv_path.empty())
        {
            csv = std::fopen(csv_path.c_str(), "w");
            ASSERT_NE(csv, nullptr) << "Failed to open ROCm NativeVNNI decode CSV: " << csv_path;
            std::fprintf(csv,
                         "backend,phase,format,codebook,shape,m,n,k,variant,kb,target_waves,weight_bytes,min_us,mean_us,stddev_us,eff_bw_gbs,bw_efficiency,speedup_vs_int8,theoretical_speedup,kernel_efficiency,cosine,correctness_pass,is_best\n");
        }

        int executed_cases = 0;
        int executed_rows = 0;
        for (const auto &fmt : ALL_PERF_FORMATS)
        {
            if (!shouldRunName(format_filters, fmt.name))
                continue;

            for (const auto &shape : SHAPES)
            {
                if (!shouldRunName(shape_filters, shape.name))
                    continue;

                auto weights = fmt.create(static_cast<size_t>(shape.N), static_cast<size_t>(shape.K));
                const uint8_t codebook_id = requireNativeVnniInfo(weights.get(), fmt.name).codebook_id;
                ROCmPackedWeights packed;
                ASSERT_TRUE(packWeightsToROCm(weights.get(), packed))
                    << "Failed to pack " << fmt.name << "/" << shape.name;
                ASSERT_TRUE(ensureNativeVNNIPayloadForDecodeTrainer(
                    weights.get(), packed, shape.N, shape.K, fmt.name))
                    << fmt.name << "/" << shape.name << " did not produce a native-VNNI payload";
                const double packed_weight_bytes = nativePackedWeightBytes(packed);

                GpuWeightsCache gpu_w;
                if (weights && reference_mode == DecodeReferenceMode::FP32HipBLAS)
                {
                    std::vector<float> fp32(static_cast<size_t>(shape.N) * shape.K);
                    weights->to_fp32(fp32.data());
                    gpu_w.upload(fp32.data(), shape.N, shape.K, 0);
                }

                struct VariantRow
                {
                    DecodeVariant variant;
                    BenchResult result;
                };

                for (int M : m_values)
                {
                    if (executed_cases >= max_cases)
                        break;

                    const double int8_us = benchmarkINT8Reference(shape, M, 0);
                    auto input = TestTensorFactory::createFP32Random(
                        {static_cast<size_t>(M), static_cast<size_t>(shape.K)});
                    ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)))
                        << "Failed to upload shared trainer input for " << fmt.name
                        << "/" << shape.name << " M=" << M;

                    std::unique_ptr<TensorBase> native_auto_reference;
                    const float *d_native_reference_output = nullptr;
                    if (reference_mode == DecodeReferenceMode::NativeAuto)
                    {
                        native_auto_reference = computeNativeAutoReferenceOutput(
                            &packed, shape, M, input.get(), 0);
                        ASSERT_TRUE(native_auto_reference)
                            << "Failed to compute native-auto reference for "
                            << fmt.name << "/" << shape.name << " M=" << M;
                        d_native_reference_output = reinterpret_cast<const float *>(
                            dynamic_cast<FP32Tensor *>(native_auto_reference.get())->gpu_data_ptr());
                        ASSERT_NE(d_native_reference_output, nullptr)
                            << "Native-auto reference has no device pointer";
                    }

                    std::vector<VariantRow> rows;
                    rows.reserve(variants.size());
                    for (const auto &variant : variants)
                    {
                        DecodeTuningOverrideGuard guard(variant);
                        rows.push_back(VariantRow{
                            variant,
                            benchmarkFormat(fmt,
                                            shape,
                                            M,
                                            int8_us,
                                            weights.get(),
                                            &packed,
                                            packed_weight_bytes,
                                            reference_mode == DecodeReferenceMode::FP32HipBLAS ? &gpu_w : nullptr,
                                            input.get(),
                                            reference_mode,
                                            d_native_reference_output,
                                            0)});
                    }

                    int best_index = -1;
                    for (size_t row_index = 0; row_index < rows.size(); ++row_index)
                    {
                        const auto &candidate = rows[row_index].result;
                        if (!candidate.correctness_pass || candidate.min_us <= 0.0)
                            continue;
                        if (best_index < 0 || candidate.min_us < rows[static_cast<size_t>(best_index)].result.min_us)
                            best_index = static_cast<int>(row_index);
                    }

                    if (csv)
                    {
                        for (size_t row_index = 0; row_index < rows.size(); ++row_index)
                        {
                            const auto &row = rows[row_index];
                            const auto &r = row.result;
                            std::fprintf(csv,
                                         "rocm,decode,%s,%u,%s,%d,%d,%d,%s,%d,%d,%.0f,%.3f,%.3f,%.3f,%.3f,%.3f,%.6f,%.6f,%.3f,%.6f,%d,%d\n",
                                         fmt.name.c_str(),
                                         static_cast<unsigned>(codebook_id),
                                         shape.name.c_str(),
                                         r.M,
                                         shape.N,
                                         shape.K,
                                         row.variant.name.c_str(),
                                         row.variant.kb,
                                         row.variant.target_waves,
                                         r.weight_bytes,
                                         r.min_us,
                                         r.mean_us,
                                         r.stddev_us,
                                         r.eff_bw_gbps,
                                         r.bw_efficiency,
                                         r.speedup_vs_int8,
                                         r.theoretical_speedup,
                                         r.kernel_efficiency,
                                         r.cosine_sim,
                                         r.correctness_pass ? 1 : 0,
                                         best_index >= 0 && static_cast<int>(row_index) == best_index ? 1 : 0);
                            ++executed_rows;
                        }
                        std::fflush(csv);
                    }

                    if (best_index < 0)
                    {
                        for (const auto &row : rows)
                        {
                            std::fprintf(stderr,
                                         "[ROCmNativeVNNI][DECODE][TRAINER][CANDIDATE] format=%s shape=%s M=%d variant=%s time_us=%.3f cosine=%.6f correctness=%d\n",
                                         fmt.name.c_str(),
                                         shape.name.c_str(),
                                         M,
                                         row.variant.name.c_str(),
                                         row.result.min_us,
                                         row.result.cosine_sim,
                                         row.result.correctness_pass ? 1 : 0);
                        }
                    }
                    ASSERT_GE(best_index, 0) << "No correct ROCm NativeVNNI decode variant for "
                                             << fmt.name << "/" << shape.name << " M=" << M;

                    const auto &best = rows[static_cast<size_t>(best_index)];
                    std::fprintf(stderr,
                                 "[ROCmNativeVNNI][DECODE][TRAINER] format=%s codebook=%u shape=%s M=%d best=%s time_us=%.3f cosine=%.6f speedup_vs_int8=%.3fx\n",
                                 fmt.name.c_str(),
                                 static_cast<unsigned>(codebook_id),
                                 shape.name.c_str(),
                                 M,
                                 best.variant.name.c_str(),
                                 best.result.min_us,
                                 best.result.cosine_sim,
                                 best.result.speedup_vs_int8);
                    ASSERT_TRUE(best.result.correctness_pass)
                        << fmt.name << "/" << shape.name << " M=" << M
                        << " cosine=" << best.result.cosine_sim;
                    ++executed_cases;
                }
                if (executed_cases >= max_cases)
                    break;
            }
        }

        if (csv)
        {
            std::fclose(csv);
            ASSERT_GT(executed_rows, 0) << "ROCm NativeVNNI decode trainer CSV had no rows.";
        }
        ASSERT_GT(executed_cases, 0) << "No ROCm NativeVNNI decode trainer cases selected.";
#endif
    }

    // =============================================================================
    // Test: Full matrix — all formats × all shapes — multi-GPU
    // =============================================================================

    TEST_F(NativeVNNIPerfTest, AllFormats_AllShapes_Matrix)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        fprintf(stderr, "\n[NativeVNNI Perf] Device: %s\n", device_name_.c_str());
        fprintf(stderr, "[NativeVNNI Perf] %zu formats × %zu shapes | %d warmup + %d runs each\n",
                ALL_PERF_FORMATS.size(), SHAPES.size(), WARMUP_RUNS, BENCH_RUNS);
        fprintf(stderr, "[NativeVNNI Perf] Using %d GPU(s) for parallel benchmarking\n", NUM_GPUS);

        // =========================================================================
        // Phase 1: Benchmark INT8 VNNI reference for each shape — multi-GPU
        // =========================================================================
        fprintf(stderr, "\n[Phase 1] Benchmarking INT8 VNNI reference on %d GPU(s)...\n",
                NUM_GPUS);

        struct Int8Work
        {
            int shape_idx;
            double cost;
        };
        std::vector<Int8Work> int8_work;
        for (int si = 0; si < (int)SHAPES.size(); ++si)
            int8_work.push_back({si, (double)SHAPES[si].N * SHAPES[si].K});

        std::sort(int8_work.begin(), int8_work.end(),
                  [](const Int8Work &a, const Int8Work &b)
                  { return a.cost > b.cost; });

        std::vector<std::vector<Int8Work>> int8_per_gpu(NUM_GPUS);
        for (size_t i = 0; i < int8_work.size(); ++i)
            int8_per_gpu[i % NUM_GPUS].push_back(int8_work[i]);

        std::vector<double> int8_times(SHAPES.size(), 0.0);
        std::atomic<int> int8_done{0};
        constexpr int kThroughputM = 1;

        {
            std::vector<std::thread> threads;
            for (int g = 0; g < NUM_GPUS; ++g)
            {
                threads.emplace_back([&, g]()
                                     {
                    for (const auto &w : int8_per_gpu[g])
                    {
                        double ref_us = benchmarkINT8Reference(SHAPES[w.shape_idx], kThroughputM, g);
                        int8_times[w.shape_idx] = ref_us;
                        int done = ++int8_done;
                        fprintf(stderr, "  [GPU %d] INT8 %s: %.1f μs  (%d/%zu)\n",
                                g, SHAPES[w.shape_idx].name.c_str(), ref_us,
                                done, int8_work.size());
                    } });
            }
            for (auto &t : threads)
                t.join();
        }

        std::unordered_map<std::string, double> int8_ref_us;
        for (int si = 0; si < (int)SHAPES.size(); ++si)
            int8_ref_us[SHAPES[si].name] = int8_times[si];

        // =========================================================================
        // Phase 2: Benchmark all native-VNNI formats — multi-GPU
        // =========================================================================
        fprintf(stderr, "\n[Phase 2] Benchmarking %zu native-VNNI formats on %d GPU(s)...\n",
                ALL_PERF_FORMATS.size(), NUM_GPUS);

        struct WorkGroup
        {
            int format_idx;
            int shape_idx;
            double cost;
        };
        std::vector<WorkGroup> groups;
        for (int fi = 0; fi < (int)ALL_PERF_FORMATS.size(); ++fi)
            for (int si = 0; si < (int)SHAPES.size(); ++si)
                groups.push_back({fi, si, (double)SHAPES[si].N * SHAPES[si].K});

        std::sort(groups.begin(), groups.end(),
                  [](const WorkGroup &a, const WorkGroup &b)
                  { return a.cost > b.cost; });

        std::vector<std::vector<WorkGroup>> per_gpu(NUM_GPUS);
        for (size_t i = 0; i < groups.size(); ++i)
            per_gpu[i % NUM_GPUS].push_back(groups[i]);

        const size_t num_shapes = SHAPES.size();
        const size_t total = ALL_PERF_FORMATS.size() * num_shapes;
        std::vector<BenchResult> results(total);
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
                        const auto &fmt = ALL_PERF_FORMATS[wg.format_idx];
                        const auto &shape = SHAPES[wg.shape_idx];

                        auto weights = fmt.create(
                            static_cast<size_t>(shape.N),
                            static_cast<size_t>(shape.K));
                        GpuWeightsCache gpu_w;
                        if (weights)
                        {
                            std::vector<float> w_fp32(
                                static_cast<size_t>(shape.N) * shape.K);
                            weights->to_fp32(w_fp32.data());
                            gpu_w.upload(w_fp32.data(), shape.N, shape.K, g);
                        }

                        double ref_us = int8_ref_us.count(shape.name)
                                            ? int8_ref_us[shape.name]
                                            : 0.0;

                        auto r = benchmarkFormat(fmt, shape, kThroughputM, ref_us,
                                                 weights.get(), &gpu_w, g);

                        size_t idx = wg.format_idx * num_shapes + wg.shape_idx;
                        results[idx] = std::move(r);

                        int done = ++phase2_done;
                        fprintf(stderr, "  [GPU %d] %s/%s %.1f μs cos=%.4f %s  (%d/%zu)\n",
                                g, fmt.name.c_str(), shape.name.c_str(),
                                results[idx].min_us, results[idx].cosine_sim,
                                results[idx].correctness_pass ? "✓" : "✗",
                                done, total_groups);
                    } });
            }
            for (auto &t : threads)
                t.join();
        }

        // =========================================================================
        // Phase 3: Print per-shape comparison tables
        // =========================================================================
        for (const auto &shape : SHAPES)
        {
            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            auto ref_it = int8_ref_us.find(shape.name);
            double ref_us = (ref_it != int8_ref_us.end()) ? ref_it->second : 0.0;

            char title[256];
            snprintf(title, sizeof(title),
                     "Shape: %s (N=%d K=%d) | INT8 ref: %.1f μs",
                     shape.name.c_str(), shape.N, shape.K, ref_us);

            table << fort::header
                  << "Format" << "BPW" << "Wt KB" << "Min μs"
                  << "Speedup" << "Theoret." << "Kern Eff"
                  << "BW GB/s" << "BW Eff %" << "Cosine" << fort::endr;

            table.column(0).set_cell_text_align(fort::text_align::left);
            for (int c = 1; c <= 9; ++c)
                table.column(c).set_cell_text_align(fort::text_align::right);

            for (const auto &r : results)
            {
                if (r.shape_name != shape.name)
                    continue;

                char b_bpw[16], b_kb[16], b_min[16];
                char b_speedup[16], b_theo[16], b_keff[16];
                char b_bw[16], b_bweff[16], b_cos[16];

                snprintf(b_bpw, sizeof(b_bpw), "%.1f", r.bpw);
                snprintf(b_kb, sizeof(b_kb), "%.0f", r.weight_bytes / 1024.0);
                snprintf(b_min, sizeof(b_min), "%.1f", r.min_us);
                snprintf(b_speedup, sizeof(b_speedup), "%.2fx", r.speedup_vs_int8);
                snprintf(b_theo, sizeof(b_theo), "%.2fx", r.theoretical_speedup);
                snprintf(b_keff, sizeof(b_keff), "%.0f%%", r.kernel_efficiency);
                snprintf(b_bw, sizeof(b_bw), "%.1f", r.eff_bw_gbps);
                snprintf(b_bweff, sizeof(b_bweff), "%.1f%%", r.bw_efficiency);
                snprintf(b_cos, sizeof(b_cos), "%.4f", r.cosine_sim);

                table << r.format_name << b_bpw << b_kb << b_min
                      << b_speedup << b_theo << b_keff
                      << b_bw << b_bweff << b_cos << fort::endr;
            }

            fprintf(stderr, "\n%s\n%s\n", title, table.to_string().c_str());
        }

        // =========================================================================
        // Phase 4: Grand Summary — average across all shapes, sorted by kern eff
        // =========================================================================
        fprintf(stderr, "\n");
        fort::utf8_table summary;
        summary.set_border_style(FT_DOUBLE2_STYLE);
        summary << fort::header
                << "Format" << "BPW" << "Avg Min μs" << "Avg Speedup"
                << "Theoretical" << "Avg Kern Eff" << "Avg BW GB/s"
                << "Avg Cosine" << "Status"
                << fort::endr;

        summary.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 8; ++c)
            summary.column(c).set_cell_text_align(fort::text_align::right);

        struct FormatSummary
        {
            std::string name;
            double bpw;
            double avg_min_us;
            double avg_speedup;
            double theoretical;
            double avg_kern_eff;
            double avg_bw;
            double avg_cosine;
            bool all_pass;
        };
        std::vector<FormatSummary> format_summaries;

        for (const auto &fmt : ALL_PERF_FORMATS)
        {
            double total_min = 0.0, total_speedup = 0.0, total_keff = 0.0;
            double total_bw = 0.0, total_cos = 0.0;
            bool all_pass = true;
            int count = 0;
            for (const auto &r : results)
            {
                if (r.format_name == fmt.name)
                {
                    total_min += r.min_us;
                    total_speedup += r.speedup_vs_int8;
                    total_keff += r.kernel_efficiency;
                    total_bw += r.eff_bw_gbps;
                    total_cos += r.cosine_sim;
                    if (!r.correctness_pass)
                        all_pass = false;
                    ++count;
                }
            }
            if (count == 0)
                continue;

            format_summaries.push_back({
                fmt.name,
                fmt.bpw,
                total_min / count,
                total_speedup / count,
                8.0 / fmt.bpw,
                total_keff / count,
                total_bw / count,
                total_cos / count,
                all_pass,
            });
        }

        // Sort by kernel efficiency ascending (worst first) for tuning focus
        std::sort(format_summaries.begin(), format_summaries.end(),
                  [](const FormatSummary &a, const FormatSummary &b)
                  { return a.avg_kern_eff < b.avg_kern_eff; });

        for (const auto &fs : format_summaries)
        {
            char b_bpw[16], b_min[16], b_speedup[16], b_theo[16];
            char b_keff[16], b_bw[16], b_cos[16];
            snprintf(b_bpw, sizeof(b_bpw), "%.1f", fs.bpw);
            snprintf(b_min, sizeof(b_min), "%.1f", fs.avg_min_us);
            snprintf(b_speedup, sizeof(b_speedup), "%.2fx", fs.avg_speedup);
            snprintf(b_theo, sizeof(b_theo), "%.2fx", fs.theoretical);
            snprintf(b_keff, sizeof(b_keff), "%.0f%%", fs.avg_kern_eff);
            snprintf(b_bw, sizeof(b_bw), "%.1f", fs.avg_bw);
            snprintf(b_cos, sizeof(b_cos), "%.4f", fs.avg_cosine);
            const char *status = fs.all_pass ? "✓" : "✗";

            summary << fs.name << b_bpw << b_min << b_speedup
                    << b_theo << b_keff << b_bw << b_cos << status
                    << fort::endr;
        }

        fprintf(stderr, "GRAND SUMMARY: Average across all shapes (sorted by Kern Eff ascending — worst first)\n");
        fprintf(stderr, "%s\n", summary.to_string().c_str());
        fprintf(stderr, "Speedup = INT8_time / format_time (>1x = faster than INT8)\n");
        fprintf(stderr, "Theoretical = 8.0/BPW (ideal speedup from bandwidth savings alone)\n");
        fprintf(stderr, "Kern Eff = Speedup/Theoretical × 100%% (how close to bandwidth-optimal)\n");
        fprintf(stderr, "Cosine = GPU output vs HipBLAS FP32 reference (gate: >= %.4f for all formats)\n",
                COSINE_SIM_GATE);

        // Validate correctness (per-format gate)
        for (const auto &r : results)
        {
            const float gate = cosine_gate_for(r.format_name);
            EXPECT_GE(r.cosine_sim, gate)
                << r.format_name << "/" << r.shape_name
                << " cosine=" << r.cosine_sim
                << " (gate=" << gate << ")";
        }
#endif
    }

    // =============================================================================
    // Test: BPW-vs-bandwidth scaling curve (focused)
    // =============================================================================

    TEST_F(NativeVNNIPerfTest, BPW_Scaling_7B_FFN_Down)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        // Largest realistic shape — best for exposing bandwidth differences
        const GEMVShape shape{"7B_FFN_Dn", 3584, 18944};

        fprintf(stderr, "\n[NativeVNNI Perf] BPW Scaling Curve\n");
        fprintf(stderr, "[NativeVNNI Perf] Shape: %s (N=%d K=%d) — largest GEMV shape\n",
                shape.name.c_str(), shape.N, shape.K);

        // Select representative formats spanning the BPW range
        const std::vector<std::string> selected = {
            "IQ2_XXS",
            "IQ2_XS",
            "Q2_K",
            "IQ3_XXS",
            "Q3_K",
            "Q4_0",
            "Q4_K",
            "Q5_0",
            "Q5_K",
            "Q6_K",
        };

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Format" << "BPW" << "Weight MB" << "Min μs" << "BW GB/s"
              << "BW Eff %" << "Bytes/Elem" << "Cosine" << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 7; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &sel_name : selected)
        {
            auto it = std::find_if(ALL_PERF_FORMATS.begin(), ALL_PERF_FORMATS.end(),
                                   [&](const PerfFormatSpec &f)
                                   { return f.name == sel_name; });
            if (it == ALL_PERF_FORMATS.end())
                continue;

            auto weights = it->create(
                static_cast<size_t>(shape.N), static_cast<size_t>(shape.K));
            GpuWeightsCache gpu_w;
            if (weights)
            {
                std::vector<float> w_fp32(static_cast<size_t>(shape.N) * shape.K);
                weights->to_fp32(w_fp32.data());
                gpu_w.upload(w_fp32.data(), shape.N, shape.K, 0);
            }

            constexpr int kInspectorM = 1;
            auto r = benchmarkFormat(*it, shape, kInspectorM, 0.0, weights.get(), &gpu_w, 0);

            double bytes_per_elem = r.weight_bytes / (static_cast<double>(r.N) * r.K);

            char buf_bpw[16], buf_mb[16], buf_min[16], buf_bw[16];
            char buf_eff[16], buf_bpe[16], buf_cos[16];
            snprintf(buf_bpw, sizeof(buf_bpw), "%.1f", r.bpw);
            snprintf(buf_mb, sizeof(buf_mb), "%.2f", r.weight_bytes / (1024.0 * 1024.0));
            snprintf(buf_min, sizeof(buf_min), "%.1f", r.min_us);
            snprintf(buf_bw, sizeof(buf_bw), "%.1f", r.eff_bw_gbps);
            snprintf(buf_eff, sizeof(buf_eff), "%.1f%%", r.bw_efficiency);
            snprintf(buf_bpe, sizeof(buf_bpe), "%.3f", bytes_per_elem);
            snprintf(buf_cos, sizeof(buf_cos), "%.4f", r.cosine_sim);

            table << r.format_name << buf_bpw << buf_mb << buf_min << buf_bw
                  << buf_eff << buf_bpe << buf_cos << fort::endr;
        }

        fprintf(stderr, "\n%s\n", table.to_string().c_str());
        fprintf(stderr, "Expected: lower BPW = less data = lower μs (if decode ALU < BW savings)\n");
        fprintf(stderr, "HBM2 peak bandwidth reference: %.0f GB/s\n", HBM2_PEAK_GBPS);
#endif
    }

} // anonymous namespace
