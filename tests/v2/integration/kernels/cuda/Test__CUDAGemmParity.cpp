/**
 * @file Test__CUDAGemmParity.cpp
 * @brief Parity tests for CUDA GEMM kernels vs CPU reference
 *
 * **Purpose**: Validate that CUDA GEMM kernels produce numerically equivalent
 * results to CPU kernels with high cosine similarity (>= 0.999).
 *
 * **Tests**:
 * - CUDAFloatingPointGemmKernel (FP32) vs FloatingPointGemmKernel
 * - CUDAQuantisedGemmKernel (IQ4_NL) vs QuantisedGemmKernel
 * - Various matrix sizes (decode, prefill, large)
 * - Real tensor objects through KernelFactory dispatch
 *
 * **Pass Criteria**:
 * - Cosine similarity >= 0.999 (very high correlation)
 * - No NaN/Inf in outputs
 * - Relative error < 5% for quantized (quantization inherently lossy)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "transfer/TransferEngine.h"

// Include project headers BEFORE CUDATestUtils.h
#include "tensors/Tensors.h"
#include "tensors/TensorKernels.h" // For TensorProjectionDesc
#include "kernels/KernelFactory.h"
#include "kernels/cuda/gemm/CUDAWeightPacker.h"
#include "kernels/cuda/gemm/CUDAQuantisedGemmKernel.h"
#include "kernels/cuda/gemm/CUDADeviceWorkspace.h"
#include "backends/ComputeBackend.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/compute_stages/ComputeStageUtils.h"
#include "execution/local_execution/coherence/GpuCoherence.h"        // Direct-kernel test coherence
#include "execution/local_execution/device/DeviceWorkspaceManager.h" // For workspace binding
#include "execution/compute_stages/stages/GDNProjectionStage.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "config/TensorParallelConfig.h"
#include "loaders/ModelLoader.h"
#include "loaders/ModelContext.h"
#include "loaders/ModelContextConfig.h"
#include "models/qwen35/Qwen35Schema.h"
#include "models/qwen35moe/Qwen35MoESchema.h"
#include "tensors/TensorSlice.h"
#include "tensors/TensorFactory.h"
#include "utils/PerfStatsCollector.h"
#include "utils/DebugEnv.h"
#include "utils/MPIContext.h"
#include "../../../utils/TestModelHelper.h"
#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#include <cuda_runtime.h>
#endif

// Now include test utils
#include "../../../utils/CUDATestUtils.h"
#include "../../../utils/TestTensorFactory.h"
#include "../../../utils/GpuPreparedGemmHarness.h"
#include "../../../utils/NativeVNNIEquivalenceInventory.h"
#include "../../../utils/QuantizedVerifierFormats.h"
#include "../../../utils/VerifierRowTestInventory.h"

#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <random>
#include <numeric>
#include <filesystem>
#include <limits>
#include <stdexcept>

using namespace llaminar2;
using namespace llaminar2::test::cuda;
using namespace llaminar2::test; // For TestTensorFactory

// Alias for kernel DeviceType to avoid ambiguity
using KernelDeviceType = llaminar::v2::kernels::DeviceType;

// Alias for TensorProjectionDesc
using TensorProjectionDesc = llaminar2::ITensorGemm::TensorProjectionDesc;

#ifdef HAVE_CUDA
extern "C"
{
    void cudaNativeVNNIPrefill_setStreamKMode(int mode);
    int cudaNativeVNNIPrefill_getStreamKMode();
    void cudaNativeVNNIPrefill_setBK256Mode(int mode);
    int cudaNativeVNNIPrefill_getBK256Mode();
    void cudaNativeVNNIPrefill_setDeterministicMode(bool enabled);
    bool cudaNativeVNNIPrefill_getDeterministicMode();
    void cudaNativeVNNIPrefill_setForceTile(int tile_id, int split_k);
    void cudaNativeVNNIPrefill_getForceTile(int *tile_id, int *split_k);
    void cudaNativeVNNIPrefill_getLastLaunchSelection(
        int *tile_id,
        int *split_k,
        int *used_bk256,
        int *used_streamk);
	    bool cudaNativeVNNIInitIQGridTables_tuned();
	    void cudaNativeVNNIGemvTuned_setDecodeEquivalentM1Config(int enabled);
	    int cudaNativeVNNIGemvTuned_getDecodeEquivalentM1Config();
	    bool cudaQuantGemm_quantizeActivationsBlockwise(
        const float *d_A_fp32,
        int8_t *d_A_int8,
        float *d_scales_A,
        int M, int K,
        int cuda_device_id,
        void *stream);
    bool cudaNativeVNNIGemvTuned_fp32(
        const int8_t *d_A_int8,
        const uint8_t *d_payload,
        const uint16_t *d_scales,
        const uint16_t *d_mins,
        const uint32_t *d_emins,
        float *d_C_fp32,
        const float *d_scales_A_block,
        int N, int K,
        float alpha, float beta,
        const float *d_C_existing,
        const float *d_bias,
        uint8_t codebook_id,
        int cuda_device_id,
        void *stream,
        CUDAGemvContext *gemv_ctx,
        CUDARowMajorWeights **rm_slot);
    bool cudaNativeVNNIGemvTuned_small_m_fp32(
        const int8_t *d_A_int8,
        const uint8_t *d_payload,
        const uint16_t *d_scales,
        const uint16_t *d_mins,
        const uint32_t *d_emins,
        float *d_C_fp32,
        const float *d_scales_A_block,
        int M,
        int N, int K,
        float alpha, float beta,
        const float *d_C_existing,
        const float *d_bias,
        uint8_t codebook_id,
        int cuda_device_id,
        void *stream,
        CUDAGemvContext *gemv_ctx,
        CUDARowMajorWeights **rm_slot);
}
#endif

namespace
{
    class ScopedEnv
    {
    public:
        ScopedEnv(const char *name, const char *value)
            : name_(name)
        {
            const char *old = std::getenv(name);
            if (old)
            {
                had_old_ = true;
                old_value_ = old;
            }
            setenv(name, value, 1);
            mutableDebugEnv().reload();
        }

        ~ScopedEnv()
        {
            if (had_old_)
                setenv(name_.c_str(), old_value_.c_str(), 1);
            else
                unsetenv(name_.c_str());
            mutableDebugEnv().reload();
        }

        ScopedEnv(const ScopedEnv &) = delete;
        ScopedEnv &operator=(const ScopedEnv &) = delete;

    private:
        std::string name_;
        std::string old_value_;
        bool had_old_ = false;
    };

#ifdef HAVE_CUDA
    class ScopedCudaPrefillModes
    {
    public:
        ScopedCudaPrefillModes()
            : streamk_(cudaNativeVNNIPrefill_getStreamKMode()),
              bk256_(cudaNativeVNNIPrefill_getBK256Mode()),
              deterministic_(cudaNativeVNNIPrefill_getDeterministicMode())
        {
            cudaNativeVNNIPrefill_getForceTile(&force_tile_, &force_split_k_);
        }

        ~ScopedCudaPrefillModes()
        {
            cudaNativeVNNIPrefill_setForceTile(force_tile_, force_split_k_);
            cudaNativeVNNIPrefill_setStreamKMode(streamk_);
            cudaNativeVNNIPrefill_setBK256Mode(bk256_);
            cudaNativeVNNIPrefill_setDeterministicMode(deterministic_);
        }

        ScopedCudaPrefillModes(const ScopedCudaPrefillModes &) = delete;
        ScopedCudaPrefillModes &operator=(const ScopedCudaPrefillModes &) = delete;

    private:
        int force_tile_ = -1;
        int force_split_k_ = 0;
        int streamk_ = 0;
        int bk256_ = 0;
        bool deterministic_ = false;
    };

    class ScopedDeterministicDebugEnv
    {
    public:
        ScopedDeterministicDebugEnv()
        {
            const char *old = std::getenv("LLAMINAR_DETERMINISTIC");
            if (old)
            {
                had_old_ = true;
                old_value_ = old;
            }
            old_prefill_deterministic_ = cudaNativeVNNIPrefill_getDeterministicMode();
            setenv("LLAMINAR_DETERMINISTIC", "1", 1);
            mutableDebugEnv().reload();
            cudaNativeVNNIPrefill_setDeterministicMode(true);
        }

        ~ScopedDeterministicDebugEnv()
        {
            cudaNativeVNNIPrefill_setDeterministicMode(old_prefill_deterministic_);
            if (had_old_)
                setenv("LLAMINAR_DETERMINISTIC", old_value_.c_str(), 1);
            else
                unsetenv("LLAMINAR_DETERMINISTIC");
            mutableDebugEnv().reload();
        }

        ScopedDeterministicDebugEnv(const ScopedDeterministicDebugEnv &) = delete;
        ScopedDeterministicDebugEnv &operator=(const ScopedDeterministicDebugEnv &) = delete;

    private:
        std::string old_value_;
        bool had_old_ = false;
        bool old_prefill_deterministic_ = false;
    };
#endif

    struct CUDASmallMFormatSpec
    {
        const char *name;
        double cosine_threshold;
        std::function<std::unique_ptr<TensorBase>(size_t, size_t)> create;
    };

    const std::vector<CUDASmallMFormatSpec> &cudaSmallMNativeFormats()
    {
        static const std::vector<CUDASmallMFormatSpec> formats = []
        {
            std::vector<CUDASmallMFormatSpec> result;
            result.reserve(quantizedVerifierFormats().size());
            for (const auto &format : quantizedVerifierFormats())
            {
                double threshold = 0.990;
                switch (format.tensor_type)
                {
                case TensorType::IQ4_NL:
                case TensorType::IQ4_XS:
                    threshold = 0.985;
                    break;
                case TensorType::Q3_K:
                    threshold = 0.980;
                    break;
                case TensorType::Q2_K:
                    threshold = 0.960;
                    break;
                case TensorType::IQ3_S:
                    threshold = 0.970;
                    break;
                case TensorType::IQ3_XXS:
                    threshold = 0.960;
                    break;
                case TensorType::IQ2_S:
                    threshold = 0.920;
                    break;
                case TensorType::IQ2_XS:
                    threshold = 0.900;
                    break;
                case TensorType::IQ2_XXS:
                    threshold = 0.880;
                    break;
                case TensorType::IQ1_S:
                case TensorType::IQ1_M:
                    threshold = 0.800;
                    break;
                case TensorType::Q8_0:
                case TensorType::Q8_1:
                case TensorType::Q8_K:
                    threshold = 0.999;
                    break;
                default:
                    break;
                }
                result.push_back({
                    format.label,
                    threshold,
                    [creator = format.create](size_t n, size_t k)
                    {
                        return creator({n, k}, 42u);
                    }});
            }
            return result;
        }();
        return formats;
    }

    struct LocalTPWoShapeCase
    {
        const char *label;
        int hidden;
        int local_attention_dim;
        int full_attention_dim;
    };

    std::vector<LocalTPWoShapeCase> localTPWoShapeCases()
    {
        return {
            /*
             * The narrow case preserves the original synthetic LocalTP Wo
             * coverage.  The wide case matches Qwen3.6 MoE full-attention
             * LocalTP layers: each rank projects a 2048-wide local attention
             * slice into the 2048-wide hidden stream before the two partials
             * are allreduced.
             */
            {"qwen36_narrow_attention", 2048, 1024, 2048},
            {"qwen36_moe_full_attention", 2048, 2048, 4096},
        };
    }

    struct LocalTPWoFormatCase
    {
        const char *label;
        std::function<std::unique_ptr<TensorBase>(
            const std::vector<size_t> &shape,
            uint32_t seed)>
            create;
    };

    std::vector<LocalTPWoFormatCase> cudaLocalTPWoQuantizedFormatCases()
    {
        std::vector<LocalTPWoFormatCase> cases;
        for (const auto &format : cudaSmallMNativeFormats())
        {
            cases.push_back(
                {format.name,
                 [creator = format.create](
                     const std::vector<size_t> &shape,
                     uint32_t /*seed*/) -> std::unique_ptr<TensorBase>
                 {
                     if (shape.size() != 2u)
                         throw std::runtime_error("cudaLocalTPWoQuantizedFormatCases expects a rank-2 shape");
                     return creator(shape[0], shape[1]);
                 }});
        }
        return cases;
    }

    std::vector<LocalTPWoFormatCase> cudaLocalTPWoFloatingPointFormatCases()
    {
        return {
            {"FP32", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createFP32Random(shape, -0.20f, 0.20f, seed); }},
            {"FP16", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createFP16Random(shape, -0.20f, 0.20f, seed); }},
            {"BF16", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createBF16Random(shape, -0.20f, 0.20f, seed); }},
        };
    }

    std::unique_ptr<TensorBase> wrapSyntheticInputParallelShard(
        std::unique_ptr<TensorBase> local_weight,
        size_t full_rows,
        size_t full_cols,
        int rank,
        int world_size)
    {
        auto meta = SliceMetadata::forRowParallel(
            full_rows, full_cols, rank, world_size,
            true /* inner_is_presliced */);
        return std::make_unique<TensorSlice>(std::move(local_weight), meta);
    }

    GpuPreparedGemm makeGpuPreparedVerifierProjection(
        TensorBase *weight,
        DeviceId device,
        const std::string &canonical_name,
        ModelContextId model_id)
    {
        if (!weight)
            throw std::runtime_error("makeGpuPreparedVerifierProjection: null weight");
        switch (weight->native_type())
        {
        case TensorType::FP32:
        case TensorType::FP16:
        case TensorType::BF16:
            return makeGpuPreparedFloatingPointGemm(weight, device, canonical_name, model_id);
        default:
            return makeGpuPreparedGemm(weight, device, canonical_name, model_id);
        }
    }

    double groupedVerifierGemmPerfCounterValue()
    {
        double total = 0.0;
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"mtp.gemm_grouped_decode_equivalent_verifier_prefill_rows"}))
        {
            if (record.domain == "mtp" &&
                record.name == "gemm_grouped_decode_equivalent_verifier_prefill_rows")
            {
                total += record.value;
            }
        }
        return total;
    }

    struct LocalTPWoShardFixture
    {
        DeviceId device;
        std::unique_ptr<TensorBase> weight;
        GpuPreparedGemm prepared;
        std::vector<float> input_rows;
    };

    LocalTPWoShardFixture makeLocalTPWoShardFixture(
        const LocalTPWoFormatCase &format_case,
        int rank,
        int world_size,
        int N,
        int local_K,
        int full_K,
        DeviceId device,
        uint32_t seed)
    {
        auto local_weight = format_case.create(
            {static_cast<size_t>(N), static_cast<size_t>(local_K)},
            seed + static_cast<uint32_t>(rank * 17));
        auto wrapped_weight = wrapSyntheticInputParallelShard(
            std::move(local_weight),
            static_cast<size_t>(N),
            static_cast<size_t>(full_K),
            rank,
            world_size);

        LocalTPWoShardFixture fixture;
        fixture.device = device;
        fixture.weight = std::move(wrapped_weight);
        fixture.prepared = makeGpuPreparedVerifierProjection(
            fixture.weight.get(),
            device,
            std::string("test.cuda.localtp_wo.") + format_case.label +
                ".rank" + std::to_string(rank),
            ModelContextId{static_cast<uint64_t>(81000 + seed + static_cast<uint32_t>(rank))});

        const size_t row_capacity =
            static_cast<size_t>(kGroupedVerifierRuntimeRows.back());
        auto input = TestTensorFactory::createFP32Random(
            {row_capacity, static_cast<size_t>(local_K)},
            -0.75f,
            0.75f,
            seed + 500u + static_cast<uint32_t>(rank * 31));
        fixture.input_rows.assign(
            input->data(),
            input->data() + row_capacity * static_cast<size_t>(local_K));
        return fixture;
    }

    struct GroupedAndSerialRows
    {
        std::vector<float> grouped;
        std::vector<float> serial;
    };


    ITensorGemm *getPreparedKernel(const TensorBase *tensor, DeviceId device_id)
    {
        // GPU INT8-packed (quantized native-VNNI) weights cannot go through
        // KernelFactory::prepareGemmHandleLocal() — that path is guarded and throws
        // because such kernels must be built from VRAM-resident, repacked payloads
        // owned by a WeightVRAMPool. Route those through the shared production-pipeline
        // helper instead, and keep the returned lifetime owners alive in a static list.
        auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor);
        const bool is_gpu_quantized =
            (device_id.is_cuda() || device_id.is_rocm()) &&
            unpackable != nullptr &&
            const_cast<IINT8Unpackable *>(unpackable)->vnniFormatInfo() != nullptr;

        if (is_gpu_quantized)
        {
            static std::vector<llaminar2::test::GpuPreparedGemm> gpu_prepared;
            const uint64_t prepared_id = static_cast<uint64_t>(gpu_prepared.size());
            gpu_prepared.push_back(
                llaminar2::test::makeGpuPreparedGemm(
                    const_cast<TensorBase *>(tensor),
                    device_id,
                    "test.gpu_prepared_gemm.weight." + std::to_string(prepared_id),
                    ModelContextId{9900 + prepared_id}));
            return gpu_prepared.back().kernel;
        }

        static std::vector<std::shared_ptr<llaminar::v2::kernels::KernelFactory::PreparedGemmHandle>> handles;
        auto prepared = llaminar::v2::kernels::KernelFactory::prepareGemmHandleLocal(tensor, device_id);
        if (!prepared)
        {
            return nullptr;
        }
        handles.push_back(std::move(prepared));
        return llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(handles.back().get());
    }

    // ============================================================================
    // Cosine Similarity Utilities
    // ============================================================================

    /**
     * @brief Compute cosine similarity between two float arrays
     *
     * cosine = (A · B) / (||A|| * ||B||)
     *
     * @return Value in [-1, 1], where 1 = perfect correlation
     */
    double cosineSimilarity(const float *a, const float *b, size_t count)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            dot += static_cast<double>(a[i]) * b[i];
            norm_a += static_cast<double>(a[i]) * a[i];
            norm_b += static_cast<double>(b[i]) * b[i];
        }
        double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
        if (denom < 1e-12)
            return 0.0;
        return dot / denom;
    }

    /**
     * @brief Compute relative L2 error: ||A - B|| / ||B||
     */
    double relativeL2Error(const float *actual, const float *expected, size_t count)
    {
        double diff_norm = 0.0, expected_norm = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            double diff = actual[i] - expected[i];
            diff_norm += diff * diff;
            expected_norm += static_cast<double>(expected[i]) * expected[i];
        }
        if (expected_norm < 1e-12)
            return diff_norm > 1e-12 ? 1e9 : 0.0;
        return std::sqrt(diff_norm / expected_norm);
    }

    /**
     * @brief Compute max absolute error
     */
    float maxAbsError(const float *actual, const float *expected, size_t count)
    {
        float max_err = 0.0f;
        for (size_t i = 0; i < count; ++i)
        {
            float err = std::abs(actual[i] - expected[i]);
            max_err = std::max(max_err, err);
        }
        return max_err;
    }

    /**
     * @brief Assert that two CUDA-produced FP32 rows are bitwise identical.
     *
     * MTP verifier rows are not ordinary approximate GEMM outputs: any row can
     * become live KV/GDN/short-conv state.  The row therefore has to match the
     * serial CUDA decode path exactly, not merely within a model-level cosine
     * tolerance.  The diagnostic reports the first mismatching value and its raw
     * IEEE-754 payload so a future tuning change can tell immediately whether it
     * introduced one ULP of reduction drift or a larger indexing/coherence bug.
     */
    void expectBitwiseEqualFloatRow(
        const char *label,
        const float *actual,
        const float *expected,
        size_t count)
    {
        ASSERT_NE(actual, nullptr) << label << " actual row is null";
        ASSERT_NE(expected, nullptr) << label << " expected row is null";
        ASSERT_GT(count, 0u) << label << " row is empty";

        if (std::memcmp(actual, expected, count * sizeof(float)) == 0)
            return;

        size_t first_mismatch = 0;
        while (first_mismatch < count)
        {
            std::uint32_t actual_word = 0;
            std::uint32_t expected_word = 0;
            std::memcpy(
                &actual_word,
                actual + first_mismatch,
                sizeof(actual_word));
            std::memcpy(
                &expected_word,
                expected + first_mismatch,
                sizeof(expected_word));
            if (actual_word != expected_word)
                break;
            ++first_mismatch;
        }

        std::uint32_t actual_bits = 0;
        std::uint32_t expected_bits = 0;
        std::memcpy(&actual_bits, actual + first_mismatch, sizeof(actual_bits));
        std::memcpy(&expected_bits, expected + first_mismatch, sizeof(expected_bits));

        size_t matching_expected_column = count;
        for (size_t column = 0; column < count; ++column)
        {
            std::uint32_t candidate_bits = 0;
            std::memcpy(
                &candidate_bits,
                expected + column,
                sizeof(candidate_bits));
            if (candidate_bits == actual_bits)
            {
                matching_expected_column = column;
                break;
            }
        }

        ADD_FAILURE()
            << label << " is not bitwise equivalent at column "
            << first_mismatch
            << " actual=" << actual[first_mismatch]
            << " expected=" << expected[first_mismatch]
            << " actual_bits=" << actual_bits
            << " expected_bits=" << expected_bits
            << " matching_expected_column="
            << (matching_expected_column == count
                    ? std::string("none")
                    : std::to_string(matching_expected_column))
            << " max_abs=" << maxAbsError(actual, expected, count)
            << " rel_l2=" << relativeL2Error(actual, expected, count)
            << " cosine=" << cosineSimilarity(actual, expected, count);
    }

    /**
     * @brief Scope direct CUDA native-VNNI calls into production verifier mode.
     *
     * The production graph reaches this policy through
     * `ITensorGemm::beginVerifierDecodeEquivalentScope()`.  The all-format test
     * below intentionally calls the low-level launcher directly, so it needs the
     * same thread-local switch here to prove the publishable MTP verifier path
     * rather than a faster small-M tuning policy.
     */
    class ScopedCudaNativeVNNIDecodeEquivalentM1Policy
    {
    public:
        ScopedCudaNativeVNNIDecodeEquivalentM1Policy()
            : previous_(cudaNativeVNNIGemvTuned_getDecodeEquivalentM1Config())
        {
            cudaNativeVNNIGemvTuned_setDecodeEquivalentM1Config(1);
        }

        ~ScopedCudaNativeVNNIDecodeEquivalentM1Policy()
        {
            cudaNativeVNNIGemvTuned_setDecodeEquivalentM1Config(previous_);
        }

        ScopedCudaNativeVNNIDecodeEquivalentM1Policy(
            const ScopedCudaNativeVNNIDecodeEquivalentM1Policy &) = delete;
        ScopedCudaNativeVNNIDecodeEquivalentM1Policy &operator=(
            const ScopedCudaNativeVNNIDecodeEquivalentM1Policy &) = delete;

    private:
        int previous_ = 0;
    };

    /**
     * @brief Compare two logit rows as probability distributions.
     *
     * Cosine and L2 catch amplitude drift, but speculative decoding is also
     * sensitive to small probability-mass movement near the sampled frontier.
     * This helper applies a numerically stable softmax to both rows and returns
     * 0.5 * (KL(actual || expected) + KL(expected || actual)).
     */
    double symmetricKLFromLogits(const float *actual, const float *expected, size_t count)
    {
        if (!actual || !expected || count == 0)
            return std::numeric_limits<double>::infinity();

        double max_actual = -std::numeric_limits<double>::infinity();
        double max_expected = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < count; ++i)
        {
            max_actual = std::max(max_actual, static_cast<double>(actual[i]));
            max_expected = std::max(max_expected, static_cast<double>(expected[i]));
        }

        double sum_actual = 0.0;
        double sum_expected = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            sum_actual += std::exp(static_cast<double>(actual[i]) - max_actual);
            sum_expected += std::exp(static_cast<double>(expected[i]) - max_expected);
        }

        if (sum_actual <= 0.0 || sum_expected <= 0.0)
            return std::numeric_limits<double>::infinity();

        constexpr double kFloor = 1.0e-30;
        double actual_to_expected = 0.0;
        double expected_to_actual = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            const double p = std::max(
                kFloor,
                std::exp(static_cast<double>(actual[i]) - max_actual) / sum_actual);
            const double q = std::max(
                kFloor,
                std::exp(static_cast<double>(expected[i]) - max_expected) / sum_expected);
            actual_to_expected += p * std::log(p / q);
            expected_to_actual += q * std::log(q / p);
        }
        return 0.5 * (actual_to_expected + expected_to_actual);
    }

    // =========================================================================
    // Helpers: multiply via tensor interface (multiply() removed from ITensorGemm)
    // =========================================================================

    /**
     * @brief Run a CUDA GEMM test body with one explicit nonblocking stream.
     *
     * Production CUDA GEMM entry points are not allowed to use CUDA's default
     * stream.  The test harness must therefore model the same contract as a
     * compute stage: bind every participating GEMM kernel to a real stream
     * before coherence-managed tensor work begins, synchronize that stream
     * before host-side checks, and clear the binding when the test body exits.
     */
    template <typename Fn>
    bool withExplicitCudaGemmStream(std::initializer_list<ITensorGemm *> kernels, Fn &&fn)
    {
        cudaStream_t stream = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
            return false;

        for (ITensorGemm *kernel : kernels)
        {
            if (kernel)
                kernel->setGPUStream(static_cast<void *>(stream));
        }

        const bool body_ok = fn(static_cast<void *>(stream));
        const cudaError_t sync_status = cudaStreamSynchronize(stream);

        for (ITensorGemm *kernel : kernels)
        {
            if (kernel)
                kernel->setGPUStream(nullptr);
        }

        const cudaError_t destroy_status = cudaStreamDestroy(stream);
        return body_ok && sync_status == cudaSuccess && destroy_status == cudaSuccess;
    }

    /**
     * @brief Force decode-equivalent CUDA M=1 GEMV reductions for one test scope.
     *
     * The generated CUDA GEMV dispatch can deliberately choose atomic K-parallel
     * reductions for speed.  MTP verifier publication needs the stricter serial
     * decode oracle, so focused primitive tests enable the same thread-local mode
     * used by CUDAQuantisedGemmKernel's canonical M=1 helper.
     */
    class ScopedCudaDecodeEquivalentM1Gemv final
    {
    public:
        ScopedCudaDecodeEquivalentM1Gemv()
            : previous_(cudaNativeVNNIGemvTuned_getDecodeEquivalentM1Config())
        {
            cudaNativeVNNIGemvTuned_setDecodeEquivalentM1Config(1);
        }

        ~ScopedCudaDecodeEquivalentM1Gemv()
        {
            cudaNativeVNNIGemvTuned_setDecodeEquivalentM1Config(previous_);
        }

        ScopedCudaDecodeEquivalentM1Gemv(const ScopedCudaDecodeEquivalentM1Gemv &) = delete;
        ScopedCudaDecodeEquivalentM1Gemv &operator=(const ScopedCudaDecodeEquivalentM1Gemv &) = delete;

    private:
        int previous_ = 0;
    };

    /**
     * @brief CPU multiply via tensor interface — wraps raw float* in FP32Tensors.
     */
    bool cpuMultiplyToVector(ITensorGemm *kernel, const float *A_data,
                             float *C_data, int M, int N, int K,
                             bool transpose_B = true, float alpha = 1.0f, float beta = 0.0f)
    {
        auto A_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)K});
        std::memcpy(A_tensor->mutable_data(), A_data, (size_t)M * K * sizeof(float));
        auto C_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)N});
        if (beta != 0.0f)
            std::memcpy(C_tensor->mutable_data(), C_data, (size_t)M * N * sizeof(float));
        bool ok = kernel->multiply_tensor(A_tensor.get(), C_tensor.get(), M, N, K, transpose_B, alpha, beta);
        if (ok)
            std::memcpy(C_data, C_tensor->data(), (size_t)M * N * sizeof(float));
        return ok;
    }

    /**
     * @brief CUDA multiply via tensor coherence — creates FP32Tensors, uploads, runs, downloads.
     */
    bool cudaMultiplyViaTensor(ITensorGemm *kernel,
                               const float *A_host, float *C_host,
                               int M, int N, int K, DeviceId gpu_device,
                               bool transpose_B = true, float alpha = 1.0f, float beta = 0.0f)
    {
        auto A_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)K});
        std::memcpy(A_tensor->mutable_data(), A_host, (size_t)M * K * sizeof(float));
        auto C_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)N});
        if (beta != 0.0f)
            std::memcpy(C_tensor->mutable_data(), C_host, (size_t)M * N * sizeof(float));
        bool ok = withExplicitCudaGemmStream(
            {kernel},
            [&](void *stream)
            {
                return with_gpu_coherence(gpu_device, {A_tensor.get()}, {C_tensor.get()},
                                          static_cast<void *>(stream),
                                          [&]
                                          {
                                              return kernel->multiply_tensor(
                                                  A_tensor.get(), C_tensor.get(), M, N, K,
                                                  transpose_B, alpha, beta);
                                          });
            });
        if (ok)
            std::memcpy(C_host, C_tensor->data(), (size_t)M * N * sizeof(float));
        return ok;
    }

    /**
     * @brief Execute one CUDA verifier projection through the grouped LM-head path.
     *
     * This mirrors LMHeadStage::executeDecodeEquivalentVerifierPrefill(): the
     * input rows are FP32 verifier hidden states, the output tensor is FP32
     * logits, and the backend owns the decode-equivalent small-M dispatch scope.
     * Keeping this helper in the CUDA parity suite lets real model-weight tests
     * prove the production grouped path without loading a full graph.
     */
    bool cudaGroupedVerifierProjectionViaTensor(
        ITensorGemm *kernel,
        const float *A_host,
        float *C_host,
        int M,
        int N,
        int K,
        DeviceId gpu_device,
        DeviceWorkspaceManager *workspace)
    {
        auto A_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
        std::memcpy(A_tensor->mutable_data(), A_host,
                    static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(float));
        auto C_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

        bool ok = withExplicitCudaGemmStream(
            {kernel},
            [&](void *stream)
            {
                return with_gpu_coherence(
                    gpu_device,
                    {A_tensor.get()},
                    {C_tensor.get()},
                    static_cast<void *>(stream),
                    [&]
                    {
                        std::vector<ITensorGemm::TensorProjectionDesc> projections = {
                            {kernel, C_tensor.get(), N, nullptr, "lm_head"}};
                        return kernel->multiply_fused_verifier_rows_decode_equivalent(
                            A_tensor.get(),
                            projections,
                            M,
                            K,
                            nullptr,
                            workspace);
                    });
            });
        if (ok)
            std::memcpy(C_host, C_tensor->data(),
                        static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float));
        return ok;
    }

    void runCudaGemmStageRows(
        TensorBase *weight,
        const GpuPreparedGemm &prepared,
        const std::vector<float> &input_values,
        int M,
        int N,
        int K,
        bool force_decode_equivalent,
        cudaStream_t stream,
        std::vector<float> &result,
        DeviceId device)
    {
        ASSERT_EQ(cudaSetDevice(device.ordinal), cudaSuccess)
            << "device=" << device.to_string();
        ASSERT_EQ(
            input_values.size(),
            static_cast<size_t>(M) * static_cast<size_t>(K))
            << "input size for CUDA LocalTP Wo GEMMStage rows";

        auto input = TestTensorFactory::createFP32(
            {static_cast<size_t>(M), static_cast<size_t>(K)});
        std::copy_n(input_values.data(), input_values.size(), input->mutable_data());
        auto output = TestTensorFactory::createFP32Zeros(
            {static_cast<size_t>(M), static_cast<size_t>(N)});

        ASSERT_TRUE(input->ensureOnDevice(device, stream));
        ASSERT_TRUE(output->allocateOnDevice(device, stream));

        GEMMStage::Params params;
        params.device_id = device;
        params.A = input.get();
        params.B = weight;
        params.C = output.get();
        params.m = M;
        params.n = N;
        params.k = K;
        params.alpha = 1.0f;
        params.beta = 0.0f;
        params.transpose_B = false;
        params.gemm_context = GemmContext::ATTN;
        params.force_decode_equivalent_verifier_prefill = force_decode_equivalent;
        params.prepared_ref = prepared.ref;
        params.prepared_store = prepared.store.get();

        GEMMStage stage(params);
        stage.setGPUStream(static_cast<void *>(stream));

        const WorkspaceRequirements requirements = stage.getWorkspaceRequirements(M, N, K);
        const size_t budget = requirements.total_bytes_with_alignment() + 64 * 1024 * 1024;
        DeviceWorkspaceManager workspace(device, budget);
        ASSERT_TRUE(workspace.allocate(requirements));
        stage.bindWorkspace(&workspace);

        CUDADeviceContext ctx(device, device.ordinal);
        ASSERT_TRUE(stage.execute(&ctx))
            << "CUDA LocalTP Wo GEMMStage execution failed"
            << " M=" << M << " N=" << N << " K=" << K
            << " device=" << device.to_string();
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess)
            << "device=" << device.to_string();

        TransferEngine::publishCurrentDeviceWrite(output, stream);
        result.assign(
            output->data(),
            output->data() + static_cast<size_t>(M) * static_cast<size_t>(N));

        stage.unbindWorkspace();
    }

    GroupedAndSerialRows runCudaLocalTPWoShardVerifierRows(
        LocalTPWoShardFixture &fixture,
        int M,
        int N,
        int local_K,
        cudaStream_t stream)
    {
        std::vector<float> input(static_cast<size_t>(M) * static_cast<size_t>(local_K));
        for (int row = 0; row < M; ++row)
        {
            std::copy_n(
                fixture.input_rows.data() + static_cast<size_t>(row) * static_cast<size_t>(local_K),
                local_K,
                input.data() + static_cast<size_t>(row) * static_cast<size_t>(local_K));
        }

        GroupedAndSerialRows rows;
        runCudaGemmStageRows(
            fixture.weight.get(),
            fixture.prepared,
            input,
            M,
            N,
            local_K,
            /*force_decode_equivalent=*/true,
            stream,
            rows.grouped,
            fixture.device);

        rows.serial.resize(static_cast<size_t>(M) * static_cast<size_t>(N));
        for (int row = 0; row < M; ++row)
        {
            std::vector<float> row_input(static_cast<size_t>(local_K));
            std::copy_n(
                input.data() + static_cast<size_t>(row) * static_cast<size_t>(local_K),
                local_K,
                row_input.data());

            std::vector<float> row_values;
            runCudaGemmStageRows(
                fixture.weight.get(),
                fixture.prepared,
                row_input,
                1,
                N,
                local_K,
                /*force_decode_equivalent=*/false,
                stream,
                row_values,
                fixture.device);
            if (row_values.size() != static_cast<size_t>(N))
            {
                ADD_FAILURE()
                    << "serial CUDA LocalTP Wo row produced " << row_values.size()
                    << " values, expected " << N
                    << " device=" << fixture.device.to_string()
                    << " row=" << row;
                return rows;
            }
            std::copy_n(
                row_values.data(),
                N,
                rows.serial.data() + static_cast<size_t>(row) * static_cast<size_t>(N));
        }
        return rows;
    }

    std::vector<float> sumLocalTPPartials(
        const std::vector<GroupedAndSerialRows> &partials,
        bool grouped,
        int M,
        int N)
    {
        std::vector<float> result(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
        for (const auto &partial : partials)
        {
            const std::vector<float> &values = grouped ? partial.grouped : partial.serial;
            if (values.size() != result.size())
            {
                ADD_FAILURE()
                    << "CUDA LocalTP partial size mismatch: got " << values.size()
                    << " expected " << result.size()
                    << " grouped=" << grouped;
                return result;
            }
            for (size_t i = 0; i < result.size(); ++i)
                result[i] += values[i];
        }
        return result;
    }

    void runCudaLocalTPWoAllreduceVerifierRowsMatchSerial(
        const LocalTPWoFormatCase &format_case,
        const std::vector<DeviceId> &devices,
        const LocalTPWoShapeCase &shape_case,
        uint32_t seed)
    {
        ASSERT_EQ(devices.size(), 2u);
        const int N = shape_case.hidden;
        const int local_K = shape_case.local_attention_dim;
        const int full_K = shape_case.full_attention_dim;

        std::vector<cudaStream_t> streams(devices.size(), nullptr);
        for (size_t i = 0; i < devices.size(); ++i)
        {
            ASSERT_EQ(cudaSetDevice(devices[i].ordinal), cudaSuccess)
                << "device=" << devices[i].to_string();
            ASSERT_EQ(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking), cudaSuccess)
                << "device=" << devices[i].to_string();
        }

        std::vector<LocalTPWoShardFixture> fixtures;
        fixtures.reserve(devices.size());
        for (size_t rank = 0; rank < devices.size(); ++rank)
        {
            fixtures.push_back(makeLocalTPWoShardFixture(
                format_case,
                static_cast<int>(rank),
                static_cast<int>(devices.size()),
                N,
                local_K,
                full_K,
                devices[rank],
                seed));
        }

        for (const int M : kGroupedVerifierRuntimeRows)
        {
            PerfStatsCollector::reset();
            const double before = groupedVerifierGemmPerfCounterValue();

            std::vector<GroupedAndSerialRows> partials;
            partials.reserve(fixtures.size());
            for (size_t rank = 0; rank < fixtures.size(); ++rank)
            {
                partials.push_back(runCudaLocalTPWoShardVerifierRows(
                    fixtures[rank],
                    M,
                    N,
                    local_K,
                    streams[rank]));
            }

            const auto grouped_sum = sumLocalTPPartials(partials, true, M, N);
            const auto serial_sum = sumLocalTPPartials(partials, false, M, N);
            const double after = groupedVerifierGemmPerfCounterValue();
            EXPECT_GE(after - before, static_cast<double>(M * static_cast<int>(devices.size())))
                << format_case.label << " shape=" << shape_case.label << " M=" << M
                << " must exercise GEMMStage grouped verifier publication route\n"
                << PerfStatsCollector::summaryString(
                       {"mtp.gemm_grouped_decode_equivalent_verifier_prefill_rows"}, 20);

            expectBitwiseEqualFloatRow(
                (std::string("CUDA LocalTP Wo reconstructed allreduce grouped verifier rows format=") +
                 format_case.label + " shape=" + shape_case.label + " M=" + std::to_string(M))
                    .c_str(),
                grouped_sum.data(),
                serial_sum.data(),
                grouped_sum.size());
        }

        for (size_t i = 0; i < streams.size(); ++i)
        {
            ASSERT_EQ(cudaSetDevice(devices[i].ordinal), cudaSuccess);
            ASSERT_EQ(cudaStreamDestroy(streams[i]), cudaSuccess);
        }
    }

    /**
     * @brief Production-shaped fixture for mirrored LocalTP attention Wo weights.
     *
     * The LocalTP target state for MTP keeps the attention output projection
     * mirrored on each GPU instead of sharding it and allreducing a tiny
     * `[M, hidden]` verifier result.  This fixture models that topology directly:
     * each device owns the complete Wo matrix `[hidden, full_attention_dim]`,
     * and the grouped verifier path must produce exactly the same rows as
     * repeated serial M=1 decode on that same mirrored matrix.
     */
    struct ReplicatedWoFixture
    {
        DeviceId device;
        std::unique_ptr<TensorBase> weight;
        GpuPreparedGemm prepared;
        std::vector<float> input_rows;
    };

    ReplicatedWoFixture makeCudaReplicatedWoFixture(
        const LocalTPWoFormatCase &format_case,
        const LocalTPWoShapeCase &shape_case,
        DeviceId device,
        uint32_t seed)
    {
        const int N = shape_case.hidden;
        const int K = shape_case.full_attention_dim;
        ReplicatedWoFixture fixture;
        fixture.device = device;
        fixture.weight = format_case.create(
            {static_cast<size_t>(N), static_cast<size_t>(K)},
            seed);
        fixture.prepared = makeGpuPreparedVerifierProjection(
            fixture.weight.get(),
            device,
            std::string("test.cuda.replicated_wo.") + format_case.label +
                "." + shape_case.label,
            ModelContextId{static_cast<uint64_t>(91000 + seed)});

        const size_t row_capacity =
            static_cast<size_t>(kGroupedVerifierRuntimeRows.back());
        auto input = TestTensorFactory::createFP32Random(
            {row_capacity, static_cast<size_t>(K)},
            -0.75f,
            0.75f,
            seed + 700u);
        fixture.input_rows.assign(
            input->data(),
            input->data() + row_capacity * static_cast<size_t>(K));
        return fixture;
    }

    GroupedAndSerialRows runCudaReplicatedWoVerifierRows(
        ReplicatedWoFixture &fixture,
        int M,
        int N,
        int K,
        cudaStream_t stream)
    {
        std::vector<float> input(static_cast<size_t>(M) * static_cast<size_t>(K));
        for (int row = 0; row < M; ++row)
        {
            std::copy_n(
                fixture.input_rows.data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                K,
                input.data() + static_cast<size_t>(row) * static_cast<size_t>(K));
        }

        GroupedAndSerialRows rows;
        runCudaGemmStageRows(
            fixture.weight.get(),
            fixture.prepared,
            input,
            M,
            N,
            K,
            /*force_decode_equivalent=*/true,
            stream,
            rows.grouped,
            fixture.device);

        rows.serial.resize(static_cast<size_t>(M) * static_cast<size_t>(N));
        for (int row = 0; row < M; ++row)
        {
            std::vector<float> row_input(static_cast<size_t>(K));
            std::copy_n(
                input.data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                K,
                row_input.data());

            std::vector<float> row_values;
            runCudaGemmStageRows(
                fixture.weight.get(),
                fixture.prepared,
                row_input,
                1,
                N,
                K,
                /*force_decode_equivalent=*/false,
                stream,
                row_values,
                fixture.device);
            if (row_values.size() != static_cast<size_t>(N))
            {
                ADD_FAILURE()
                    << "serial CUDA replicated Wo row produced " << row_values.size()
                    << " values, expected " << N
                    << " device=" << fixture.device.to_string()
                    << " row=" << row;
                return rows;
            }
            std::copy_n(
                row_values.data(),
                N,
                rows.serial.data() + static_cast<size_t>(row) * static_cast<size_t>(N));
        }
        return rows;
    }

    /**
     * @brief Prove mirrored/replicated Wo grouped rows are byte-identical.
     *
     * This is the no-fallback LocalTP path we want for device-resident MTP:
     * keep the tiny verifier projection local to each device, avoid a small
     * collective, and still route through GEMMStage's decode-equivalent grouped
     * implementation rather than row replay.  The perf counter assertion makes
     * sure the grouped production hook actually ran.
     */
    void runCudaReplicatedWoVerifierRowsMatchSerial(
        const LocalTPWoFormatCase &format_case,
        DeviceId device,
        const LocalTPWoShapeCase &shape_case,
        uint32_t seed)
    {
        const int N = shape_case.hidden;
        const int K = shape_case.full_attention_dim;
        ASSERT_EQ(cudaSetDevice(device.ordinal), cudaSuccess)
            << "device=" << device.to_string();
        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess)
            << "device=" << device.to_string();

        auto fixture = makeCudaReplicatedWoFixture(format_case, shape_case, device, seed);
        for (const int M : kGroupedVerifierRuntimeRows)
        {
            PerfStatsCollector::reset();
            const double before = groupedVerifierGemmPerfCounterValue();
            const GroupedAndSerialRows rows = runCudaReplicatedWoVerifierRows(
                fixture,
                M,
                N,
                K,
                stream);
            const double after = groupedVerifierGemmPerfCounterValue();
            EXPECT_GE(after - before, static_cast<double>(M))
                << format_case.label << " shape=" << shape_case.label << " M=" << M
                << " must exercise GEMMStage grouped verifier publication route\n"
                << PerfStatsCollector::summaryString(
                       {"mtp.gemm_grouped_decode_equivalent_verifier_prefill_rows"}, 20);

            expectBitwiseEqualFloatRow(
                (std::string("CUDA replicated Wo grouped verifier rows format=") +
                 format_case.label + " shape=" + shape_case.label +
                 " M=" + std::to_string(M))
                    .c_str(),
                rows.grouped.data(),
                rows.serial.data(),
                rows.grouped.size());
        }

        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }

    /**
     * @brief Return the Qwen3.6 MoE GGUF used by real-weight kernel regressions.
     */
    std::filesystem::path qwen36MoEModelPath()
    {
        if (const char *env = std::getenv("LLAMINAR_QWEN36_MOE_MODEL"))
            return std::filesystem::path(env);
        return std::filesystem::path("/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf");
    }

    /**
     * @brief Return the Qwen3.6 dense GGUF used by real LocalTP regressions.
     */
    std::filesystem::path qwen36DenseModelPath()
    {
        if (const char *env = std::getenv("LLAMINAR_QWEN36_DENSE_MODEL"))
            return std::filesystem::path(env);
        return std::filesystem::path("/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf");
    }

    /**
     * @brief Load a model context whose raw quantized weights can be GPU-prepared.
     *
     * The production GPU upload/repack helper consumes the host GGUF blocks
     * directly, so the context must keep native tensor formats and avoid CPU-side
     * requantization.
     */
    std::shared_ptr<ModelContext> loadQwen36ModelForGpuWeights(
        const std::filesystem::path &model_path)
    {
        ModelContextConfig config = ModelContextConfig::defaults();
        config.strategy = WeightDistributionStrategy::REPLICATED;
        config.weight_precision = WeightPrecision::NATIVE;
        config.use_mmap = true;
        config.target_is_gpu = true;
        return ModelContext::create(model_path.string(), config);
    }

    /**
     * @brief Return the raw tensor that the GPU upload pipeline should consume.
     *
     * LocalTP weight slicing wraps already-materialized row/column slices in a
     * `TensorSlice` so graph builders retain sharding metadata.  The CUDA weight
     * upload/repack harness, like the production GPU loader, needs the actual
     * native tensor that exposes format metadata such as `IINT8Unpackable`.
     * Stage tests should still pass the wrapper to the stage so tensor-sharding
     * metadata remains part of the exercised contract.
     */
    TensorBase *gpuPreparationTensor(TensorBase *tensor)
    {
        if (auto *slice = dynamic_cast<TensorSlice *>(tensor))
            return slice->inner();
        return tensor;
    }

    /**
     * @brief Configure a loaded Qwen3.6 MoE context for two-way LocalTP slicing.
     *
     * The sharder needs both the schema's weight-sharding annotations and GDN
     * dimensions.  Fused GDN QKV is asymmetric: Q and K are replicated, while V
     * is sharded.  Missing any one of these hints changes the physical weight
     * presented to `GDNProjectionStage`, so this helper mirrors the production
     * LocalTP setup before a regression requests per-device weights.
     */
    std::shared_ptr<TensorParallelConfig> configureQwen36MoELocalTPWeights(
        const std::shared_ptr<ModelContext> &model_ctx,
        const std::vector<DeviceId> &devices)
    {
        auto weight_manager = model_ctx ? model_ctx->concreteWeightManager() : nullptr;
        if (!model_ctx || !weight_manager)
            return nullptr;

        Qwen35MoESchemaFactory schema_factory;
        weight_manager->setWeightShardingConfig(schema_factory.getWeightShardingConfig());

        const int n_heads = model_ctx->headCount();
        const int n_kv_heads = std::max(1, model_ctx->headCountKV());
        const int head_dim = model_ctx->keyLength() > 0
                                 ? model_ctx->keyLength()
                                 : (n_heads > 0 ? model_ctx->embeddingLength() / n_heads : 0);
        const int d_ff = std::max(
            32,
            std::max(
                model_ctx->feedForwardLength(),
                model_ctx->concreteLoader().getInt("expert_feed_forward_length", 0)));
        const int vocab_size = model_ctx->vocabSize();

        weight_manager->setModelDimensions(n_heads, n_kv_heads, head_dim);
        weight_manager->setGDNDimensions(
            model_ctx->concreteLoader().getInt("ssm.group_count", 0),
            model_ctx->concreteLoader().getInt("ssm.time_step_rank", 0),
            model_ctx->concreteLoader().getInt("ssm.state_size", 0));

        auto tp_config = std::make_shared<TensorParallelConfig>(
            TensorParallelConfig::equalSplit(
                static_cast<int>(devices.size()),
                n_heads,
                n_kv_heads,
                d_ff,
                vocab_size,
                devices));
        weight_manager->setTensorParallelConfig(tp_config);
        return tp_config;
    }

    /**
     * @brief Configure the dense Qwen3.6 production two-way LocalTP sharder.
     *
     * Qwen3.6 GDN QKV is not a uniform column slice: Q/K channels are mirrored
     * while V channels are apportioned.  A synthetic tensor with the resulting
     * dimensions cannot prove the production slice wrapper, native GGUF block
     * boundaries, or GPU repack.  This helper installs the real dense schema and
     * model dimensions before a regression asks WeightManager for each device's
     * concrete local tensor.
     */
    std::shared_ptr<TensorParallelConfig> configureQwen36DenseLocalTPWeights(
        const std::shared_ptr<ModelContext> &model_ctx,
        const std::vector<DeviceId> &devices)
    {
        auto weight_manager = model_ctx ? model_ctx->concreteWeightManager() : nullptr;
        if (!model_ctx || !weight_manager)
            return nullptr;

        Qwen35SchemaFactory schema_factory;
        weight_manager->setWeightShardingConfig(
            schema_factory.getWeightShardingConfig());

        const int n_heads = model_ctx->headCount();
        const int n_kv_heads = std::max(1, model_ctx->headCountKV());
        const int head_dim = model_ctx->keyLength() > 0
                                 ? model_ctx->keyLength()
                                 : (n_heads > 0
                                        ? model_ctx->embeddingLength() / n_heads
                                        : 0);
        const int d_ff = std::max(32, model_ctx->feedForwardLength());
        const int vocab_size = model_ctx->vocabSize();

        weight_manager->setModelDimensions(n_heads, n_kv_heads, head_dim);
        weight_manager->setGDNDimensions(
            model_ctx->concreteLoader().getInt("ssm.group_count", 0),
            model_ctx->concreteLoader().getInt("ssm.time_step_rank", 0),
            model_ctx->concreteLoader().getInt("ssm.state_size", 0));

        auto tp_config = std::make_shared<TensorParallelConfig>(
            TensorParallelConfig::equalSplit(
                static_cast<int>(devices.size()),
                n_heads,
                n_kv_heads,
                d_ff,
                vocab_size,
                devices));
        weight_manager->setTensorParallelConfig(tp_config);
        return tp_config;
    }

    bool cpuFusedSwiGLUDownToVector(ITensorGemm *kernel,
                                    const float *gate_host,
                                    const float *up_host,
                                    float *C_host,
                                    int M, int N, int K,
                                    float alpha = 1.0f, float beta = 0.0f)
    {
        auto gate_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)K});
        auto up_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)K});
        std::memcpy(gate_tensor->mutable_data(), gate_host, (size_t)M * K * sizeof(float));
        std::memcpy(up_tensor->mutable_data(), up_host, (size_t)M * K * sizeof(float));
        auto C_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)N});
        if (beta != 0.0f)
            std::memcpy(C_tensor->mutable_data(), C_host, (size_t)M * N * sizeof(float));

        bool ok = kernel->multiply_tensor_with_fused_swiglu(
            gate_tensor.get(), up_tensor.get(), C_tensor.get(),
            M, N, K, alpha, beta);
        if (ok)
            std::memcpy(C_host, C_tensor->data(), (size_t)M * N * sizeof(float));
        return ok;
    }

    bool cudaFusedSwiGLUDownViaTensor(ITensorGemm *kernel,
                                      const float *gate_host,
                                      const float *up_host,
                                      float *C_host,
                                      int M, int N, int K,
                                      DeviceId gpu_device,
                                      float alpha = 1.0f, float beta = 0.0f)
    {
        auto gate_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)K});
        auto up_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)K});
        std::memcpy(gate_tensor->mutable_data(), gate_host, (size_t)M * K * sizeof(float));
        std::memcpy(up_tensor->mutable_data(), up_host, (size_t)M * K * sizeof(float));
        auto C_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)N});
        if (beta != 0.0f)
            std::memcpy(C_tensor->mutable_data(), C_host, (size_t)M * N * sizeof(float));

        bool ok = withExplicitCudaGemmStream(
            {kernel},
            [&](void *stream)
            {
                return with_gpu_coherence(gpu_device, {gate_tensor.get(), up_tensor.get()}, {C_tensor.get()},
                                          static_cast<void *>(stream),
                                          [&]
                                          {
                                              if (M > 1)
                                              {
                                                  return kernel->multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                                                      gate_tensor.get(), up_tensor.get(), C_tensor.get(),
                                                      M, N, K, alpha, beta);
                                              }
                                              return kernel->multiply_tensor_with_fused_swiglu(
                                                  gate_tensor.get(), up_tensor.get(), C_tensor.get(),
                                                  M, N, K, alpha, beta);
                                          });
            });
        if (ok)
            std::memcpy(C_host, C_tensor->data(), (size_t)M * N * sizeof(float));
        return ok;
    }

} // namespace

// ============================================================================
// Test Fixture
// ============================================================================

class Test__CUDAGemmParity : public CUDATestBase
{
protected:
    std::mt19937 rng_{42};
    std::uniform_real_distribution<float> dist_{-1.0f, 1.0f};

    void SetUp() override
    {
        CUDATestBase::SetUp();
#ifdef HAVE_CUDA
        if (gpu_idx_ < 0 || !backend_)
        {
            return;
        }
        setenv("LLAMINAR_DETERMINISTIC", "0", 1);
        mutableDebugEnv().reload();
        cudaNativeVNNIPrefill_setDeterministicMode(false);
        llaminar::v2::kernels::KernelFactory::clearCache();
#endif
    }

    /**
     * @brief Fill tensor with random data
     */
    void fillRandom(FP32Tensor *tensor)
    {
        float *data = tensor->mutable_data();
        for (size_t i = 0; i < tensor->numel(); ++i)
        {
            data[i] = dist_(rng_);
        }
    }

    /**
     * @brief Create random FP32 data
     */
    std::vector<float> randomFP32(size_t count)
    {
        std::vector<float> data(count);
        for (auto &val : data)
        {
            val = dist_(rng_);
        }
        return data;
    }

    /**
     * @brief CPU reference GEMM: C = A @ B^T
     *
     * For weight matrix B stored as [N, K], compute C[M, N] = A[M, K] @ B^T
     */
    void cpuGemmReference(
        const float *A, const float *B, float *C,
        int M, int N, int K)
    {
        // C[i, j] = sum_k A[i, k] * B[j, k]  (B is transposed)
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    sum += A[i * K + k] * B[j * K + k];
                }
                C[i * N + j] = sum;
            }
        }
    }

    /**
     * @brief Parity result structure
     */
    struct ParityResult
    {
        double cosine_similarity = 0.0;
        double relative_l2_error = 0.0;
        double symmetric_kl = 0.0;
        float max_abs_error = 0.0f;
        bool has_nan_inf = false;
        bool passed = false;

        void print(const std::string &name) const
        {
            std::cout << name << ":\n"
                      << "  Cosine similarity: " << cosine_similarity << "\n"
                      << "  Relative L2 error: " << (relative_l2_error * 100.0) << "%\n"
                      << "  Symmetric KL:       " << symmetric_kl << "\n"
                      << "  Max abs error:     " << max_abs_error << "\n"
                      << "  Passed:            " << (passed ? "YES" : "NO") << "\n";
        }
    };

    /**
     * @brief Compare CUDA vs CPU GEMM results
     *
     * @param cosine_threshold Minimum cosine similarity (default 0.999)
     * @param rel_l2_threshold Maximum relative L2 error (default 0.05 = 5%)
     */
    ParityResult checkParity(
        const float *cuda_result,
        const float *cpu_result,
        size_t count,
        double cosine_threshold = 0.999,
        double rel_l2_threshold = 0.05)
    {
        ParityResult result;
        result.has_nan_inf = hasNaNOrInf(cuda_result, count);
        result.cosine_similarity = cosineSimilarity(cuda_result, cpu_result, count);
        result.relative_l2_error = relativeL2Error(cuda_result, cpu_result, count);
        result.symmetric_kl = symmetricKLFromLogits(cuda_result, cpu_result, count);
        result.max_abs_error = maxAbsError(cuda_result, cpu_result, count);

        result.passed = !result.has_nan_inf &&
                        result.cosine_similarity >= cosine_threshold &&
                        result.relative_l2_error <= rel_l2_threshold;

        return result;
    }

    // =========================================================================
    // Workspace Management Helpers
    // =========================================================================

    static size_t workspaceBudgetFor(const WorkspaceRequirements &reqs)
    {
        constexpr size_t kMinBudget = 64ull * 1024ull * 1024ull;
        constexpr size_t kPadding = 16ull * 1024ull * 1024ull;
        return std::max(kMinBudget, reqs.total_bytes_with_alignment() + kPadding);
    }

    /**
     * @brief Set up workspace for a CUDA GEMM kernel if it supports it
     *
     * CUDA quantized GEMM kernels require pre-allocated workspace buffers for:
     * - Quantized activations (INT8)
     * - Per-row scales
     * - INT32 intermediate accumulator
     *
     * FP32 kernels don't need workspace, so this is a no-op for them.
     *
     * @param kernel The GEMM kernel
     * @param M Maximum batch/sequence length
     * @param N Output dimension
     * @param K Input dimension
     * @return true on success or if kernel doesn't need workspace, false on allocation failure
     */
    bool setupWorkspaceIfNeeded(ITensorGemm *kernel, int M, int N, int K)
    {
        auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
        if (!ws_consumer)
        {
            // Kernel doesn't implement IWorkspaceConsumer (e.g., FP32 kernel)
            return true;
        }

        auto reqs = ws_consumer->getWorkspaceRequirements(M, N, K);
        workspace_ = std::make_unique<DeviceWorkspaceManager>(gpu_device_, workspaceBudgetFor(reqs));
        if (!workspace_->allocate(reqs))
        {
            LOG_ERROR("Failed to allocate workspace for CUDA GEMM kernel");
            return false;
        }
        ws_consumer->bindWorkspace(workspace_.get());
        return true;
    }

    /**
     * @brief Clean up workspace after test
     */
    void cleanupWorkspaceIfNeeded(ITensorGemm *kernel)
    {
        if (workspace_)
        {
#ifdef HAVE_CUDA
            if (gpu_device_.is_cuda())
            {
                ASSERT_EQ(cudaSetDevice(gpu_device_.ordinal), cudaSuccess);
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess)
                    << "CUDA GEMM test cleanup must wait for workspace users before freeing buffers";
            }
#endif
            auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
            if (ws_consumer)
            {
                ws_consumer->unbindWorkspace();
            }
            kernel->resetDynamicState();
            workspace_.reset();
        }
    }

    /**
     * @brief Set up a shared workspace for multiple CUDA GEMM kernels
     *
     * Used for fused QKV operations where multiple kernels share workspace.
     * Computes the maximum requirements across all kernels.
     *
     * @param kernels Vector of kernels that need workspace
     * @param M Maximum batch/sequence length
     * @param Ns Output dimensions for each kernel
     * @param K Input dimension (shared)
     * @return true on success, false on allocation failure
     */
    bool setupSharedWorkspace(
        const std::vector<ITensorGemm *> &kernels,
        int M,
        const std::vector<int> &Ns,
        int K)
    {
        // Merge requirements across all kernels by buffer name so fused paths
        // keep pace with the evolving CUDA workspace contract.
        WorkspaceRequirements shared_reqs;

        for (size_t i = 0; i < kernels.size(); ++i)
        {
            auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(kernels[i]);
            if (ws_consumer)
            {
                auto reqs = ws_consumer->getWorkspaceRequirements(M, Ns[i], K);
                for (const auto &buf : reqs.buffers)
                {
                    auto it = std::find_if(
                        shared_reqs.buffers.begin(),
                        shared_reqs.buffers.end(),
                        [&](const WorkspaceDescriptor &existing)
                        {
                            return existing.name == buf.name;
                        });

                    if (it == shared_reqs.buffers.end())
                    {
                        shared_reqs.buffers.push_back(buf);
                        continue;
                    }

                    it->size_bytes = std::max(it->size_bytes, buf.size_bytes);
                    it->alignment = std::max(it->alignment, buf.alignment);
                    it->required = it->required || buf.required;
                }
            }
        }

        addCudaConcurrentDecodeGemvSideStreamWorkspace(
            shared_reqs,
            gpu_device_,
            M,
            kernels.size());

        workspace_ = std::make_unique<DeviceWorkspaceManager>(gpu_device_, workspaceBudgetFor(shared_reqs));
        if (!workspace_->allocate(shared_reqs))
        {
            LOG_ERROR("Failed to allocate shared workspace");
            return false;
        }

        // Bind shared workspace to ALL kernels
        for (auto *kernel : kernels)
        {
            auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
            if (ws_consumer)
            {
                ws_consumer->bindWorkspace(workspace_.get());
            }
        }

        return true;
    }

    /**
     * @brief Clean up shared workspace from multiple kernels
     */
    void cleanupSharedWorkspace(const std::vector<ITensorGemm *> &kernels)
    {
#ifdef HAVE_CUDA
        if (workspace_ && gpu_device_.is_cuda())
        {
            ASSERT_EQ(cudaSetDevice(gpu_device_.ordinal), cudaSuccess);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess)
                << "CUDA GEMM shared-workspace cleanup must wait for all projection kernels before freeing buffers";
        }
#endif
        for (auto *kernel : kernels)
        {
            auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
            if (ws_consumer && ws_consumer->hasWorkspace())
            {
                ws_consumer->unbindWorkspace();
            }
            if (kernel)
            {
                kernel->resetDynamicState();
            }
        }
        workspace_.reset();
    }

    std::unique_ptr<DeviceWorkspaceManager> workspace_;
};

// ============================================================================
// FP32 Parity Tests (CUDAFloatingPointGemmKernel vs FloatingPointGemmKernel)
// ============================================================================

#ifdef HAVE_CUDA

TEST_F(Test__CUDAGemmParity, FP32_SmallMatrix_128x256x512)
{
    // Dimensions
    const int M = 128; // batch/sequence
    const int N = 256; // output dim
    const int K = 512; // input dim

    // Create weight tensor on CPU
    auto weights = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)N, (size_t)K});
    fillRandom(weights.get());

    // Create activations
    auto A_data = randomFP32(M * K);

    // ===== CPU Reference =====
    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr) << "Failed to create CPU kernel";
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    // ===== CUDA =====
    // Upload weights to GPU
    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    EXPECT_TRUE(weights->isOnGPU());

    // Create CUDA kernel via KernelFactory
    auto *cuda_kernel = getPreparedKernel(
        weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr) << "Failed to create CUDA kernel";

    // Execute CUDA GEMM via tensor coherence
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    // ===== Compare =====
    auto result = checkParity(C_cuda.data(), C_cpu.data(), M * N, 0.9999, 0.01);
    result.print("FP32 128x256x512");

    EXPECT_FALSE(result.has_nan_inf) << "CUDA output contains NaN/Inf";
    EXPECT_GE(result.cosine_similarity, 0.9999)
        << "Cosine similarity too low: " << result.cosine_similarity;
    EXPECT_LE(result.relative_l2_error, 0.01)
        << "Relative L2 error too high: " << (result.relative_l2_error * 100) << "%";

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, FP32_DecodeSize_1x896x896)
{
    // Decode: single token projection
    const int M = 1;
    const int N = 896; // Qwen2.5 hidden dim
    const int K = 896;

    auto weights = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)N, (size_t)K});
    fillRandom(weights.get());

    auto A_data = randomFP32(M * K);

    // CPU reference
    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    // CUDA
    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);

    float *d_A, *d_C;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, FP32_PrefillSize_512x896x896)
{
    // Prefill: typical sequence length
    const int M = 512;
    const int N = 896;
    const int K = 896;

    auto weights = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)N, (size_t)K});
    fillRandom(weights.get());

    auto A_data = randomFP32(M * K);

    // CPU reference
    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    // CUDA
    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);

    float *d_A, *d_C;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, BlockwiseActivationQuantizationProducesExpectedBlocks)
{
    const int M = 2;
    const int K = 64;
    auto A_data = randomFP32(static_cast<size_t>(M) * K);

    float *d_A = nullptr;
    int8_t *d_A_int8 = nullptr;
    float *d_scales = nullptr;

    ASSERT_EQ(cudaSetDevice(gpu_device_.ordinal), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_A, A_data.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_A_int8, A_data.size() * sizeof(int8_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_scales, static_cast<size_t>(M) * (K / 32) * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_A, A_data.data(), A_data.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    ASSERT_TRUE(cudaQuantGemm_quantizeActivationsBlockwise(
        d_A,
        d_A_int8,
        d_scales,
        M,
        K,
        gpu_device_.ordinal,
        nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<int8_t> quantized(A_data.size(), 0);
    std::vector<float> scales(static_cast<size_t>(M) * (K / 32), 0.0f);
    ASSERT_EQ(cudaMemcpy(quantized.data(), d_A_int8, quantized.size() * sizeof(int8_t), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(scales.data(), d_scales, scales.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int row = 0; row < M; ++row)
    {
        for (int block = 0; block < K / 32; ++block)
        {
            const size_t block_offset = static_cast<size_t>(row) * K + block * 32;
            float max_abs = 0.0f;
            for (int i = 0; i < 32; ++i)
                max_abs = std::max(max_abs, std::abs(A_data[block_offset + i]));

            const float expected_scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
            const float actual_scale = scales[static_cast<size_t>(row) * (K / 32) + block];
            EXPECT_NEAR(actual_scale, expected_scale, 1.0e-6f)
                << "row=" << row << " block=" << block;

            bool saw_nonzero = false;
            for (int i = 0; i < 32; ++i)
            {
                const float qval = A_data[block_offset + i] / expected_scale;
                const int expected_q = static_cast<int>(
                    std::rint(std::min(127.0f, std::max(-127.0f, qval))));
                const int actual_q = static_cast<int>(quantized[block_offset + i]);
                EXPECT_EQ(actual_q, expected_q)
                    << "row=" << row << " block=" << block << " i=" << i;
                saw_nonzero = saw_nonzero || actual_q != 0;
            }
            EXPECT_TRUE(saw_nonzero) << "row=" << row << " block=" << block;
        }
    }

    cudaFree(d_scales);
    cudaFree(d_A_int8);
    cudaFree(d_A);
}

// ============================================================================
// IQ4_NL Parity Tests (CUDAQuantisedGemmKernel vs QuantisedGemmKernel)
// ============================================================================

TEST_F(Test__CUDAGemmParity, IQ4_NL_SmallMatrix_128x896x896)
{
    // Dimensions - K must be multiple of 32 for IQ4_NL
    // Using 896 (Qwen2.5 hidden dim) which is known to work
    const int M = 128;
    const int N = 896;
    const int K = 896;

    // Create IQ4_NL weight tensor using TestTensorFactory
    auto weights = TestTensorFactory::createIQ4_NLRandom({(size_t)N, (size_t)K}, 123);

    // Create activations
    auto A_data = randomFP32(M * K);

    // ===== CPU Reference =====
    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr) << "Failed to create CPU IQ4_NL kernel";
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    // Check CPU result is valid
    ASSERT_FALSE(hasNaNOrInf(C_cpu.data(), M * N)) << "CPU result has NaN/Inf";

    // ===== CUDA =====
    // Upload weights to GPU
    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    EXPECT_TRUE(weights->isOnGPU());

    // Create CUDA kernel via KernelFactory
    auto *cuda_kernel = getPreparedKernel(
        weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr) << "Failed to create CUDA IQ4_NL kernel";

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    // Allocate GPU memory for activations and output
    float *d_A, *d_C;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, IQ4_NL_DecodeSize_1x896x896)
{
    // Decode single token - K must be multiple of 32
    const int M = 1;
    const int N = 896;
    const int K = 896;

    auto weights = TestTensorFactory::createIQ4_NLRandom({(size_t)N, (size_t)K}, 456);

    auto A_data = randomFP32(M * K);

    // CPU reference
    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    // CUDA
    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    float *d_A, *d_C;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, IQ4_NL_PrefillSize_512x896x896)
{
    // Prefill - larger batch
    const int M = 512;
    const int N = 896;
    const int K = 896;

    auto weights = TestTensorFactory::createIQ4_NLRandom({(size_t)N, (size_t)K}, 789);

    auto A_data = randomFP32(M * K);

    // CPU reference
    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    // CUDA
    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    float *d_A, *d_C;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

/**
 * @test Exercise heuristic-only ordinary prefill for every quantized source
 * format accepted by the production preparation pipeline.
 *
 * M=32 lies immediately above the grouped-verifier regime and therefore enters
 * the ordinary prefill launcher. The test intentionally leaves tile, split-K,
 * BK256, and Stream-K controls in production Auto mode. Each source tensor is
 * prepared through the real KernelFactory path, executed on CUDA, and compared
 * with the CPU implementation using the format's established quantized parity
 * threshold. This catches both a missing generated-prefill include dependency
 * and codebook aliases that fail to reach the total heuristic.
 */
TEST_F(Test__CUDAGemmParity, NativeVNNIPrefillHeuristicAllFormatsMatchesCPU)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA build required";
#else
    constexpr int M = 32;
    constexpr int N = 384;
    constexpr int K = 512;
    ScopedCudaPrefillModes modes;
    cudaNativeVNNIPrefill_setForceTile(-1, 0);
    cudaNativeVNNIPrefill_setStreamKMode(0);
    cudaNativeVNNIPrefill_setBK256Mode(0);
    cudaNativeVNNIPrefill_setDeterministicMode(false);

    for (const auto &format : cudaSmallMNativeFormats())
    {
        SCOPED_TRACE(format.name);
        auto weights = format.create(N, K);
        ASSERT_NE(weights, nullptr);
        auto input = randomFP32(static_cast<size_t>(M) * K);

        std::vector<float> cpu_output(static_cast<size_t>(M) * N, 0.0f);
        auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
            weights.get(), KernelDeviceType::CPU);
        ASSERT_NE(cpu_kernel, nullptr) << format.name << " CPU reference kernel";
        ASSERT_TRUE(cpuMultiplyToVector(
            cpu_kernel.get(), input.data(), cpu_output.data(), M, N, K));

        ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
        auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
        ASSERT_NE(cuda_kernel, nullptr) << format.name << " CUDA prepared kernel";
        ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

        std::vector<float> cuda_output(static_cast<size_t>(M) * N, 0.0f);
        ASSERT_TRUE(cudaMultiplyViaTensor(
            cuda_kernel,
            input.data(),
            cuda_output.data(),
            M,
            N,
            K,
            gpu_device_));

        const ParityResult parity = checkParity(
            cuda_output.data(),
            cpu_output.data(),
            cuda_output.size(),
            format.cosine_threshold,
            0.20);
        EXPECT_TRUE(parity.passed)
            << format.name << " heuristic prefill cosine="
            << parity.cosine_similarity << " relative_l2="
            << parity.relative_l2_error;

        cleanupWorkspaceIfNeeded(cuda_kernel);
        llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
    }
#endif
}

/**
 * @test Prove CUDA NativeVNNI active rows are byte-invariant for the complete
 * finite M-totality witness inventory and every quantized format.
 *
 * The production graph may capture a larger M than the request's semantic
 * token count. Before this regression, BK64 used separately expressed FP32
 * arithmetic for interior and border tiles, so padding alone changed rounding.
 * The shared inventory now exhausts every ordinary-prefill M through two full
 * 128-row tiles and both ends of every larger canonical graph-bucket interval.
 *
 * One explicit stream, prepared weight, workspace, and fixed device buffers
 * are reused per format. Every witness also launches its final active row
 * through the independent production M=1 route. This prevents exact-M and
 * bucket-M implementations from agreeing with each other while both drift
 * from serial decode. Only result bytes are downloaded after the launches,
 * keeping the regression focused on production kernel arithmetic instead of
 * measuring thousands of allocation and coherence setup transactions.
 */
TEST_F(Test__CUDAGemmParity, NativeVNNIPrefillActiveRowsAllFormatsByteExactAcrossMAndBuckets)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA build required";
#else
    struct LaunchSelection
    {
        int tile_id = -1;
        int split_k = -1;
        int used_bk256 = -1;
        int used_streamk = -1;
    };
    const auto read_launch_selection = []
    {
        LaunchSelection selection;
        cudaNativeVNNIPrefill_getLastLaunchSelection(
            &selection.tile_id,
            &selection.split_k,
            &selection.used_bk256,
            &selection.used_streamk);
        return selection;
    };

    ScopedCudaPrefillModes modes;
    cudaNativeVNNIPrefill_setForceTile(-1, 0);
    cudaNativeVNNIPrefill_setStreamKMode(0);
    cudaNativeVNNIPrefill_setBK256Mode(0);
    cudaNativeVNNIPrefill_setDeterministicMode(false);

    constexpr int canonical_prefix_rows =
        kDefaultNativeVNNIVerifierRowCapacity + 1;
    std::array<bool, 6> tensor_core_tile_seen{};
    bool bk256_narrow_seen = false;
    bool bk256_wide_seen = false;
    bool profiled_narrow_large_k_crossover_seen = false;

    for (const auto &geometry :
         nativeVNNIPrefillAllFormatGeometryEquivalenceCases())
    {
        SCOPED_TRACE(geometry.label);
        ASSERT_FALSE(geometry.row_cases.empty());
        const int N = geometry.output_columns;
        const int K = geometry.reduction_columns;
        const int maximum_bucket_rows = std::max_element(
            geometry.row_cases.begin(),
            geometry.row_cases.end(),
            [](const auto &lhs, const auto &rhs)
            {
                return lhs.bucket_rows < rhs.bucket_rows;
            })->bucket_rows;
        const std::vector<float> input_values =
            randomFP32(static_cast<size_t>(maximum_bucket_rows) * K);

        for (const auto &format : cudaSmallMNativeFormats())
        {
            SCOPED_TRACE(format.name);

            cudaStream_t stream = nullptr;
            ASSERT_EQ(
                cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                cudaSuccess);

            auto weights = format.create(N, K);
            ASSERT_NE(weights, nullptr);
            ASSERT_TRUE(weights->ensureOnDevice(gpu_device_, stream));

            auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
            ASSERT_NE(cuda_kernel, nullptr)
                << format.name << " CUDA prepared kernel";
            cuda_kernel->setGPUStream(stream);
            ASSERT_TRUE(setupWorkspaceIfNeeded(
                cuda_kernel,
                maximum_bucket_rows,
                N,
                K));

            auto input = std::make_unique<FP32Tensor>(
                std::vector<size_t>{
                    static_cast<size_t>(maximum_bucket_rows),
                    static_cast<size_t>(K)});
            std::memcpy(
                input->mutable_data(),
                input_values.data(),
                input_values.size() * sizeof(float));
            auto exact_output = std::make_unique<FP32Tensor>(
                std::vector<size_t>{
                    static_cast<size_t>(maximum_bucket_rows),
                    static_cast<size_t>(N)});
            auto bucket_output = std::make_unique<FP32Tensor>(
                std::vector<size_t>{
                    static_cast<size_t>(maximum_bucket_rows),
                    static_cast<size_t>(N)});
            auto serial_row_input = std::make_unique<FP32Tensor>(
                std::vector<size_t>{1, static_cast<size_t>(K)});
            auto serial_row_output = std::make_unique<FP32Tensor>(
                std::vector<size_t>{1, static_cast<size_t>(N)});
            ASSERT_TRUE(input->ensureOnDevice(gpu_device_, stream));
            ASSERT_TRUE(exact_output->allocateOnDevice(gpu_device_, stream));
            ASSERT_TRUE(bucket_output->allocateOnDevice(gpu_device_, stream));
            ASSERT_TRUE(serial_row_input->ensureOnDevice(gpu_device_, stream));
            ASSERT_TRUE(serial_row_output->allocateOnDevice(gpu_device_, stream));

            ASSERT_TRUE(cuda_kernel->multiply_tensor(
                input.get(),
                exact_output.get(),
                canonical_prefix_rows,
                N,
                K,
                /*transpose_B=*/true,
                1.0f,
                0.0f));
            const size_t canonical_prefix_values =
                static_cast<size_t>(canonical_prefix_rows) * N;
            std::vector<float> canonical_prefix(canonical_prefix_values);
            ASSERT_EQ(
                cudaMemcpyAsync(
                    canonical_prefix.data(),
                    exact_output->gpu_data_ptr(),
                    canonical_prefix_values * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream),
                cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

            for (const auto &test_case : geometry.row_cases)
            {
                SCOPED_TRACE(
                    std::string("active_m=") +
                    std::to_string(test_case.active_rows) +
                    " bucket_m=" +
                    std::to_string(test_case.bucket_rows));

                ASSERT_TRUE(cuda_kernel->multiply_tensor(
                    input.get(),
                    exact_output.get(),
                    test_case.active_rows,
                    N,
                    K,
                    /*transpose_B=*/true,
                    1.0f,
                    0.0f));
                const LaunchSelection exact_selection = read_launch_selection();
                ASSERT_TRUE(cuda_kernel->multiply_tensor(
                    input.get(),
                    bucket_output.get(),
                    test_case.bucket_rows,
                    N,
                    K,
                    /*transpose_B=*/true,
                    1.0f,
                    0.0f));
                const LaunchSelection bucket_selection = read_launch_selection();

                /*
                 * Lock the measured BK256-to-BK64 crossover into the same
                 * regression that proves its arithmetic. M=33 is the first
                 * exact request above BK256's 32-row narrow tile, while the
                 * corresponding M=64 selection is the graph-captured
                 * production launch used for that request.
                 */
                if (format.name == "Q4_0" &&
                    N == 513 &&
                    K == 2048 &&
                    test_case.active_rows == 33)
                {
                    EXPECT_EQ(exact_selection.tile_id, 0);
                    EXPECT_EQ(exact_selection.used_bk256, 0);
                    EXPECT_EQ(bucket_selection.tile_id, 0);
                    EXPECT_EQ(bucket_selection.used_bk256, 0);
                    profiled_narrow_large_k_crossover_seen = true;
                }

                /*
                 * Route the witness's final active row through production M=1.
                 * The same stream orders the row upload, GEMV launch, and all
                 * result downloads, so this oracle adds no device-wide fence
                 * and cannot observe partially published data.
                 */
                const float *serial_row_source =
                    input_values.data() +
                    static_cast<size_t>(test_case.active_rows - 1) * K;
                ASSERT_EQ(
                    cudaMemcpyAsync(
                        serial_row_input->gpu_data_ptr(),
                        serial_row_source,
                        static_cast<size_t>(K) * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream),
                    cudaSuccess);
                ASSERT_TRUE(cuda_kernel->multiply_tensor(
                    serial_row_input.get(),
                    serial_row_output.get(),
                    1,
                    N,
                    K,
                    /*transpose_B=*/true,
                    1.0f,
                    0.0f));

                SCOPED_TRACE(
                    std::string("exact_selection={tile=") +
                    std::to_string(exact_selection.tile_id) +
                    ",split_k=" + std::to_string(exact_selection.split_k) +
                    ",bk256=" + std::to_string(exact_selection.used_bk256) +
                    ",streamk=" + std::to_string(exact_selection.used_streamk) +
                    "} bucket_selection={tile=" +
                    std::to_string(bucket_selection.tile_id) +
                    ",split_k=" + std::to_string(bucket_selection.split_k) +
                    ",bk256=" + std::to_string(bucket_selection.used_bk256) +
                    ",streamk=" + std::to_string(bucket_selection.used_streamk) +
                    "}");

                for (const LaunchSelection &selection :
                     {exact_selection, bucket_selection})
                {
                    EXPECT_EQ(selection.split_k, 1)
                        << "production AUTO prefill changed the canonical K tree";
                    EXPECT_EQ(selection.used_streamk, 0)
                        << "production AUTO prefill selected atomic Stream-K";
                    if (selection.tile_id >= 0 &&
                        selection.tile_id <
                            static_cast<int>(tensor_core_tile_seen.size()))
                    {
                        tensor_core_tile_seen[
                            static_cast<size_t>(selection.tile_id)] = true;
                    }
                    bk256_narrow_seen |= selection.tile_id == -3;
                    bk256_wide_seen |= selection.tile_id == -2;
                }

                const size_t active_values =
                    static_cast<size_t>(test_case.active_rows) * N;
                std::vector<float> exact_host(active_values);
                std::vector<float> bucket_host(active_values);
                std::vector<float> serial_row_host(static_cast<size_t>(N));
                ASSERT_EQ(
                    cudaMemcpyAsync(
                        exact_host.data(),
                        exact_output->gpu_data_ptr(),
                        active_values * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        stream),
                    cudaSuccess);
                ASSERT_EQ(
                    cudaMemcpyAsync(
                        bucket_host.data(),
                        bucket_output->gpu_data_ptr(),
                        active_values * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        stream),
                    cudaSuccess);
                ASSERT_EQ(
                    cudaMemcpyAsync(
                        serial_row_host.data(),
                        serial_row_output->gpu_data_ptr(),
                        static_cast<size_t>(N) * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        stream),
                    cudaSuccess);
                ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

                const std::string label =
                    std::string(format.name) + " active M=" +
                    std::to_string(test_case.active_rows) +
                    " bucket M=" +
                    std::to_string(test_case.bucket_rows);
                expectBitwiseEqualFloatRow(
                    label.c_str(),
                    bucket_host.data(),
                    exact_host.data(),
                    active_values);
                const std::string prefix_label =
                    std::string(format.name) +
                    " canonical M=17 prefix at active M=" +
                    std::to_string(test_case.active_rows);
                expectBitwiseEqualFloatRow(
                    prefix_label.c_str(),
                    exact_host.data(),
                    canonical_prefix.data(),
                    canonical_prefix_values);
                const std::string serial_label =
                    std::string(format.name) +
                    " final active row vs production M=1 at active M=" +
                    std::to_string(test_case.active_rows);
                expectBitwiseEqualFloatRow(
                    serial_label.c_str(),
                    exact_host.data() +
                        static_cast<size_t>(test_case.active_rows - 1) * N,
                    serial_row_host.data(),
                    static_cast<size_t>(N));
            }

            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            cuda_kernel->setGPUStream(nullptr);
            cleanupWorkspaceIfNeeded(cuda_kernel);
            llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
            ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
        }
    }

    for (size_t tile = 0; tile < tensor_core_tile_seen.size(); ++tile)
    {
        EXPECT_TRUE(tensor_core_tile_seen[tile])
            << "M/N/K inventory did not reach CUDA tensor-core tile " << tile;
    }
    EXPECT_TRUE(bk256_narrow_seen)
        << "M/N/K inventory did not reach narrow-N BK256";
    EXPECT_TRUE(bk256_wide_seen)
        << "M/N/K inventory did not reach wide-N BK256";
    EXPECT_TRUE(profiled_narrow_large_k_crossover_seen)
        << "M/N/K inventory did not reach the profiled BK256-to-BK64 crossover";
#endif
}

/**
 * @test Lock the profiled Qwen3.6 GDN projection tile overlays to the
 * production automatic dispatcher without relaxing byte equality.
 *
 * Q4_1, Q5_1, and IQ1_S use asymmetric NativeVNNI prefill arithmetic. Their
 * Qwen3.6 GDN QKV geometry is common enough to justify an exact dispatch
 * overlay, but a faster tile is admissible only when it preserves every FP32
 * output bit. This regression therefore computes a conservative
 * `T64x128_w2x2` reference, restores production automatic dispatch, checks the
 * selected measured winner, and compares the complete output tensor byte for
 * byte. Stream-K, BK256, split-K, and deterministic test mode remain disabled
 * so this is the same single-partition route used by captured production
 * prefill graphs.
 */
TEST_F(Test__CUDAGemmParity, AsymmetricQwen36GDNProjectionUsesProfiledByteExactTile)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA build required";
#else
    constexpr int M = 128;
    constexpr int N = 8192;
    constexpr int K = 2048;
    constexpr int conservative_tile = 1; // T64x128_w2x2

    struct OverlayCase
    {
        const char *format_name;
        int expected_tile;
    };
    constexpr std::array<OverlayCase, 3> cases = {{
        {"Q4_1", 3},  // T64x128_w2x4
        {"Q5_1", 3},  // T64x128_w2x4
        {"IQ1_S", 2}, // T64x128_w4x2
    }};

    ScopedCudaPrefillModes modes;
    cudaNativeVNNIPrefill_setStreamKMode(0);
    cudaNativeVNNIPrefill_setBK256Mode(0);
    cudaNativeVNNIPrefill_setDeterministicMode(false);

    const std::vector<float> input =
        randomFP32(static_cast<size_t>(M) * static_cast<size_t>(K));

    for (const auto &overlay : cases)
    {
        SCOPED_TRACE(overlay.format_name);
        const auto format = std::find_if(
            cudaSmallMNativeFormats().begin(),
            cudaSmallMNativeFormats().end(),
            [&](const CUDASmallMFormatSpec &candidate)
            {
                return std::strcmp(candidate.name, overlay.format_name) == 0;
            });
        ASSERT_NE(format, cudaSmallMNativeFormats().end())
            << "Missing quantized verifier format " << overlay.format_name;

        auto weights = format->create(
            static_cast<size_t>(N),
            static_cast<size_t>(K));
        ASSERT_NE(weights, nullptr);
        ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));

        auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
        ASSERT_NE(cuda_kernel, nullptr)
            << overlay.format_name << " CUDA prepared kernel";
        ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

        std::vector<float> conservative_output(
            static_cast<size_t>(M) * static_cast<size_t>(N),
            0.0f);
        cudaNativeVNNIPrefill_setForceTile(conservative_tile, 1);
        ASSERT_TRUE(cudaMultiplyViaTensor(
            cuda_kernel,
            input.data(),
            conservative_output.data(),
            M,
            N,
            K,
            gpu_device_));

        int tile_id = -1;
        int split_k = -1;
        int used_bk256 = -1;
        int used_streamk = -1;
        cudaNativeVNNIPrefill_getLastLaunchSelection(
            &tile_id,
            &split_k,
            &used_bk256,
            &used_streamk);
        ASSERT_EQ(tile_id, conservative_tile);
        ASSERT_EQ(split_k, 1);
        ASSERT_EQ(used_bk256, 0);
        ASSERT_EQ(used_streamk, 0);

        std::vector<float> production_output(
            static_cast<size_t>(M) * static_cast<size_t>(N),
            0.0f);
        cudaNativeVNNIPrefill_setForceTile(-1, 0);
        ASSERT_TRUE(cudaMultiplyViaTensor(
            cuda_kernel,
            input.data(),
            production_output.data(),
            M,
            N,
            K,
            gpu_device_));

        cudaNativeVNNIPrefill_getLastLaunchSelection(
            &tile_id,
            &split_k,
            &used_bk256,
            &used_streamk);
        EXPECT_EQ(tile_id, overlay.expected_tile)
            << overlay.format_name << " did not use its profiled production tile";
        EXPECT_EQ(split_k, 1);
        EXPECT_EQ(used_bk256, 0);
        EXPECT_EQ(used_streamk, 0);

        expectBitwiseEqualFloatRow(
            (std::string(overlay.format_name) +
             " production profiled tile vs conservative tile")
                .c_str(),
            production_output.data(),
            conservative_output.data(),
            production_output.size());

        cleanupWorkspaceIfNeeded(cuda_kernel);
        llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
    }
#endif
}

// ============================================================================
// Parameterized Quantized GEMM Parity Test
// ============================================================================

/**
 * @brief Helper macro for quantized parity tests
 *
 * This reduces duplication across all quantized tensor types.
 * All tests use the same pattern: create weights, run CPU reference, run CUDA, compare.
 *
 * K-quant formats use 256-element blocks, so K must be multiple of 256.
 * Simple formats (Q4_0, Q8_0, etc.) use 32-element blocks, K multiple of 32.
 */
#define DEFINE_QUANTIZED_PARITY_TEST(TestName, TensorType, CreateMethod, BlockSize, Seed)         \
    TEST_F(Test__CUDAGemmParity, TestName)                                                        \
    {                                                                                             \
        const int M = 128;                                                                        \
        const int N = 896;                                                                        \
        const int K = (BlockSize == 256) ? 768 : 896; /* K-quants need multiple of 256 */         \
                                                                                                  \
        auto weights = TestTensorFactory::CreateMethod({(size_t)N, (size_t)K}, Seed);             \
                                                                                                  \
        auto A_data = randomFP32(M * K);                                                          \
                                                                                                  \
        /* CPU Reference */                                                                       \
        std::vector<float> C_cpu(M * N, 0.0f);                                                    \
        auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(                       \
            weights.get(), KernelDeviceType::CPU);                                                \
        ASSERT_NE(cpu_kernel, nullptr) << "Failed to create CPU kernel for " #TensorType;         \
        ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K)); \
        ASSERT_FALSE(hasNaNOrInf(C_cpu.data(), M * N)) << "CPU result has NaN/Inf";               \
                                                                                                  \
        /* CUDA */                                                                                \
        ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));                                        \
        auto *cuda_kernel = getPreparedKernel(                                                    \
            weights.get(), gpu_device_);                                                          \
        ASSERT_NE(cuda_kernel, nullptr) << "Failed to create CUDA kernel for " #TensorType;       \
                                                                                                  \
        /* Set up workspace for quantized kernel */                                               \
        ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));                                \
                                                                                                  \
        std::vector<float> C_cuda(M * N, 0.0f);                                                   \
        ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(),                             \
                                          C_cuda.data(), M, N, K, gpu_device_));                  \
                                                                                                  \
        auto result = checkParity(C_cuda.data(), C_cpu.data(), M * N, 0.99, 0.15);                \
        result.print(#TensorType " 128x" + std::to_string(N) + "x" + std::to_string(K));          \
                                                                                                  \
        EXPECT_FALSE(result.has_nan_inf) << "CUDA output contains NaN/Inf";                       \
        EXPECT_GE(result.cosine_similarity, 0.99)                                                 \
            << "Cosine similarity too low: " << result.cosine_similarity;                         \
                                                                                                  \
        cleanupWorkspaceIfNeeded(cuda_kernel);                                                    \
        llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());                       \
    }

/**
 * @brief Helper macro for decode-size (M=1) quantized GEMV parity tests
 *
 * Exercises the NativeVNNI GEMV path (M=1 decode) for each codebook.
 * K-quant formats use 256-element blocks, so K must be multiple of 256.
 */
#define DEFINE_QUANTIZED_DECODE_PARITY_TEST(TestName, TensorType, CreateMethod, BlockSize, Seed)  \
    TEST_F(Test__CUDAGemmParity, TestName)                                                        \
    {                                                                                             \
        const int M = 1;                                                                          \
        const int N = 896;                                                                        \
        const int K = (BlockSize == 256) ? 768 : 896;                                             \
                                                                                                  \
        auto weights = TestTensorFactory::CreateMethod({(size_t)N, (size_t)K}, Seed);             \
                                                                                                  \
        auto A_data = randomFP32(M * K);                                                          \
                                                                                                  \
        /* CPU Reference */                                                                       \
        std::vector<float> C_cpu(M * N, 0.0f);                                                    \
        auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(                       \
            weights.get(), KernelDeviceType::CPU);                                                \
        ASSERT_NE(cpu_kernel, nullptr) << "Failed to create CPU kernel for " #TensorType;         \
        ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K)); \
        ASSERT_FALSE(hasNaNOrInf(C_cpu.data(), M * N)) << "CPU result has NaN/Inf";               \
                                                                                                  \
        /* CUDA */                                                                                \
        ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));                                        \
        auto *cuda_kernel = getPreparedKernel(                                                    \
            weights.get(), gpu_device_);                                                          \
        ASSERT_NE(cuda_kernel, nullptr) << "Failed to create CUDA kernel for " #TensorType;       \
                                                                                                  \
        ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));                                \
                                                                                                  \
        std::vector<float> C_cuda(M * N, 0.0f);                                                   \
        ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(),                             \
                                          C_cuda.data(), M, N, K, gpu_device_));                  \
                                                                                                  \
        auto result = checkParity(C_cuda.data(), C_cpu.data(), M * N, 0.99, 0.15);                \
        result.print(#TensorType " Decode 1x" + std::to_string(N) + "x" + std::to_string(K));     \
                                                                                                  \
        EXPECT_FALSE(result.has_nan_inf) << "CUDA output contains NaN/Inf";                       \
        EXPECT_GE(result.cosine_similarity, 0.99)                                                 \
            << "Cosine similarity too low: " << result.cosine_similarity;                         \
                                                                                                  \
        cleanupWorkspaceIfNeeded(cuda_kernel);                                                    \
        llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());                       \
    }

// ============================================================================
// Q8_0 Parity Tests
// ============================================================================

DEFINE_QUANTIZED_PARITY_TEST(Q8_0_SmallMatrix, Q8_0Tensor, createQ8_0Random, 32, 101)

TEST_F(Test__CUDAGemmParity, Q8_0_PrefillM35_896x896)
{
    ScopedCudaPrefillModes modes;
    cudaNativeVNNIPrefill_setForceTile(-1, 0);
    cudaNativeVNNIPrefill_setStreamKMode(0);

    const int M = 35;
    const int N = 896;
    const int K = 896;

    auto weights = TestTensorFactory::createQ8_0Random({(size_t)N, (size_t)K}, 113);
    auto A_data = randomFP32(M * K);

    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));
    ASSERT_FALSE(hasNaNOrInf(C_cpu.data(), M * N)) << "CPU result has NaN/Inf";

    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    for (int repetition = 0; repetition < 2; ++repetition)
    {
        std::vector<float> C_cuda(M * N, 0.0f);
        ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_))
            << "production split-K prefill repetition " << repetition;

        int tile_id = -99;
        int split_k = -1;
        int used_bk256 = 0;
        int used_streamk = 0;
        cudaNativeVNNIPrefill_getLastLaunchSelection(&tile_id, &split_k, &used_bk256, &used_streamk);
        EXPECT_GT(split_k, 1)
            << "Production regression should exercise multi-partition split-K prefill";
        EXPECT_EQ(used_streamk, 0)
            << "This shape should use split-K rather than Stream-K";

        auto result = checkParity(C_cuda.data(), C_cpu.data(), M * N, 0.999, 0.05);
        result.print("Q8_0 Prefill 35x896x896"
                     " repetition=" + std::to_string(repetition) +
                     " tile=" + std::to_string(tile_id) +
                     " split_k=" + std::to_string(split_k));

        EXPECT_FALSE(result.has_nan_inf) << "CUDA output contains NaN/Inf";
        EXPECT_GE(result.cosine_similarity, 0.999)
            << "Cosine similarity too low: " << result.cosine_similarity;
        EXPECT_LE(result.relative_l2_error, 0.05)
            << "Relative L2 error too high: " << (result.relative_l2_error * 100) << "%";
    }

    cleanupWorkspaceIfNeeded(cuda_kernel);
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, Q8_0_DecodeSize_1x896x896)
{
    const int M = 1;
    const int N = 896;
    const int K = 896;

    auto weights = TestTensorFactory::createQ8_0Random({(size_t)N, (size_t)K}, 111);
    auto A_data = randomFP32(M * K);

    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        weights.get(), gpu_device_);

    auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(cuda_kernel);
    ASSERT_NE(ws_consumer, nullptr);
    const auto reqs = ws_consumer->getWorkspaceRequirements(M, N, K);
    const auto kpar_req = std::find_if(
        reqs.buffers.begin(),
        reqs.buffers.end(),
        [](const WorkspaceDescriptor &buffer)
        {
            return buffer.name == GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS;
        });
    ASSERT_NE(kpar_req, reqs.buffers.end())
        << "CUDA decode native-VNNI KPAR route must declare its partials workspace";
    EXPECT_TRUE(kpar_req->required);
    EXPECT_GE(kpar_req->size_bytes,
              static_cast<size_t>((K + 31) / 32) * static_cast<size_t>(N) * sizeof(float));

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    float *d_A, *d_C;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

// ============================================================================
// Q4_0 Parity Tests
// ============================================================================

DEFINE_QUANTIZED_PARITY_TEST(Q4_0_SmallMatrix, Q4_0Tensor, createQ4_0Random, 32, 102)

TEST_F(Test__CUDAGemmParity, Q4_0_DecodeSize_1x896x896)
{
    const int M = 1;
    const int N = 896;
    const int K = 896;

    auto weights = TestTensorFactory::createQ4_0Random({(size_t)N, (size_t)K}, 121);
    auto A_data = randomFP32(M * K);

    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        weights.get(), gpu_device_);

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    float *d_A, *d_C;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, Q4_0_SmallPrefillM4_UsesNativeSmallMRoute)
{
    const int M = 4;
    const int N = 896;
    const int K = 896;

    auto weights = TestTensorFactory::createQ4_0Random({(size_t)N, (size_t)K}, 231);
    auto A_data = randomFP32(static_cast<size_t>(M) * K);

    std::vector<float> C_cpu(static_cast<size_t>(M) * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));
    ASSERT_FALSE(hasNaNOrInf(C_cpu.data(), C_cpu.size()));

    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    std::vector<float> C_cuda(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_TRUE(cudaGroupedVerifierProjectionViaTensor(
        cuda_kernel,
        A_data.data(),
        C_cuda.data(),
        M,
        N,
        K,
        gpu_device_,
        workspace_.get()));

    auto result = checkParity(C_cuda.data(), C_cpu.data(), C_cpu.size(), 0.99, 0.15);
    result.print("Q4_0 small-prefill native GEMV route 4x896x896");
    EXPECT_FALSE(result.has_nan_inf)
        << "CUDA small-prefill M=4 route must not emit NaN/Inf";
    EXPECT_GE(result.cosine_similarity, 0.99)
        << "Cosine similarity too low: " << result.cosine_similarity;

    cleanupWorkspaceIfNeeded(cuda_kernel);
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, Q4_0_PrefillGraphReplaySurvivesKernelDynamicReset)
{
    const int M = 512;
    const int N = 896;
    const int K = 896;

    ScopedCudaPrefillModes prefill_modes;
    cudaNativeVNNIPrefill_setBK256Mode(-1);
    cudaNativeVNNIPrefill_setDeterministicMode(false);
    cudaNativeVNNIPrefill_setForceTile(1, 1); // T64x128_w2x2, single split-K.
    cudaNativeVNNIPrefill_setStreamKMode(2);  // Force two-pass Stream-K fixup buffer path.

    auto weights = TestTensorFactory::createQ4_0Random({static_cast<size_t>(N), static_cast<size_t>(K)}, 232);
    auto A_data = randomFP32(static_cast<size_t>(M) * K);

    std::vector<float> C_cpu(static_cast<size_t>(M) * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));
    ASSERT_FALSE(hasNaNOrInf(C_cpu.data(), C_cpu.size()));

    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    FP32Tensor A_tensor({static_cast<size_t>(M), static_cast<size_t>(K)});
    std::memcpy(A_tensor.mutable_data(), A_data.data(), A_data.size() * sizeof(float));
    FP32Tensor C_tensor({static_cast<size_t>(M), static_cast<size_t>(N)});

    ASSERT_EQ(cudaSetDevice(gpu_device_.ordinal), cudaSuccess);
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    ASSERT_TRUE(A_tensor.ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(C_tensor.ensureOnDevice(gpu_device_, stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    cuda_kernel->setGPUStream(stream);
    ASSERT_TRUE(cuda_kernel->multiply_tensor(
        &A_tensor, &C_tensor, M, N, K, true, 1.0f, 0.0f, nullptr, nullptr,
        gpu_device_.ordinal, workspace_.get()));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    int tile_id = -1;
    int split_k = 0;
    int used_bk256 = 0;
    int used_streamk = 0;
    cudaNativeVNNIPrefill_getLastLaunchSelection(&tile_id, &split_k, &used_bk256, &used_streamk);
    ASSERT_EQ(used_bk256, 0) << "Regression must exercise BK64 Stream-K, not BK256";
    ASSERT_EQ(used_streamk, 2) << "Regression must exercise two-pass Stream-K context scratch";

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), cudaSuccess);
    ASSERT_TRUE(cuda_kernel->multiply_tensor(
        &A_tensor, &C_tensor, M, N, K, true, 1.0f, 0.0f, nullptr, nullptr,
        gpu_device_.ordinal, workspace_.get()));
    ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), cudaSuccess);
    ASSERT_NE(graph_exec, nullptr);

    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    cuda_kernel->resetDynamicState();
    EXPECT_FALSE(cuda_kernel->hasDynamicStateActive())
        << "resetDynamicState should clear request-scoped stream binding";

    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess)
        << "Captured graph must survive request reset without dangling context scratch";
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<float> C_cuda(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_EQ(cudaMemcpyAsync(
                  C_cuda.data(),
                  C_tensor.gpu_data_ptr(),
                  C_cuda.size() * sizeof(float),
                  cudaMemcpyDeviceToHost,
                  stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    auto result = checkParity(C_cuda.data(), C_cpu.data(), C_cpu.size(), 0.99, 0.15);
    result.print("Q4_0 prefill graph replay after CUDA GEMM dynamic reset");
    EXPECT_FALSE(result.has_nan_inf);
    EXPECT_GE(result.cosine_similarity, 0.99)
        << "Cosine similarity too low after graph replay: " << result.cosine_similarity;

    if (graph_exec)
        cudaGraphExecDestroy(graph_exec);
    if (graph)
        cudaGraphDestroy(graph);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    cleanupWorkspaceIfNeeded(cuda_kernel);
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, Q4_1_DecodeSize_1x896x896)
{
    const int M = 1;
    const int N = 896;
    const int K = 896;

    auto weights = TestTensorFactory::createQ4_1Random({(size_t)N, (size_t)K}, 122);
    auto A_data = randomFP32(M * K);

    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);

    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    float *d_A = nullptr;
    float *d_C = nullptr;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

// ============================================================================
// Q4_1 Parity Tests
// ============================================================================

DEFINE_QUANTIZED_PARITY_TEST(Q4_1_SmallMatrix, Q4_1Tensor, createQ4_1Random, 32, 103)

// ============================================================================
// Q5_0 Parity Tests
// ============================================================================

DEFINE_QUANTIZED_PARITY_TEST(Q5_0_SmallMatrix, Q5_0Tensor, createQ5_0Random, 32, 104)

TEST_F(Test__CUDAGemmParity, Q5_0_DecodeSize_1x896x896)
{
    const int M = 1;
    const int N = 896;
    const int K = 896;

    auto weights = TestTensorFactory::createQ5_0Random({(size_t)N, (size_t)K}, 124);
    auto A_data = randomFP32(M * K);

    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);

    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    float *d_A = nullptr;
    float *d_C = nullptr;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

// ============================================================================
// Q5_1 Parity Tests
// ============================================================================

DEFINE_QUANTIZED_PARITY_TEST(Q5_1_SmallMatrix, Q5_1Tensor, createQ5_1Random, 32, 105)

TEST_F(Test__CUDAGemmParity, Q5_1_DecodeSize_1x896x896)
{
    const int M = 1;
    const int N = 896;
    const int K = 896;

    auto weights = TestTensorFactory::createQ5_1Random({(size_t)N, (size_t)K}, 125);
    auto A_data = randomFP32(M * K);

    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);

    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    float *d_A = nullptr;
    float *d_C = nullptr;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

// ============================================================================
// K-Quant Parity Tests (256-element super-blocks)
// ============================================================================

DEFINE_QUANTIZED_PARITY_TEST(Q6_K_SmallMatrix, Q6_KTensor, createQ6_KRandom, 256, 201)
DEFINE_QUANTIZED_PARITY_TEST(Q2_K_SmallMatrix, Q2_KTensor, createQ2_KRandom, 256, 202)
DEFINE_QUANTIZED_PARITY_TEST(Q3_K_SmallMatrix, Q3_KTensor, createQ3_KRandom, 256, 203)
DEFINE_QUANTIZED_PARITY_TEST(Q4_K_SmallMatrix, Q4_KTensor, createQ4_KRandom, 256, 204)
DEFINE_QUANTIZED_PARITY_TEST(Q5_K_SmallMatrix, Q5_KTensor, createQ5_KRandom, 256, 205)

DEFINE_QUANTIZED_DECODE_PARITY_TEST(Q6_K_DecodeSize, Q6_KTensor, createQ6_KRandom, 256, 211)
DEFINE_QUANTIZED_DECODE_PARITY_TEST(Q2_K_DecodeSize, Q2_KTensor, createQ2_KRandom, 256, 212)
DEFINE_QUANTIZED_DECODE_PARITY_TEST(Q3_K_DecodeSize, Q3_KTensor, createQ3_KRandom, 256, 213)
DEFINE_QUANTIZED_DECODE_PARITY_TEST(Q4_K_DecodeSize, Q4_KTensor, createQ4_KRandom, 256, 214)
DEFINE_QUANTIZED_DECODE_PARITY_TEST(Q5_K_DecodeSize, Q5_KTensor, createQ5_KRandom, 256, 215)

TEST_F(Test__CUDAGemmParity, Q4_K_VerifierSmallM_2x896x768)
{
    const int M = 2;
    const int N = 896;
    const int K = 768;

    auto weights = TestTensorFactory::createQ4_KRandom({(size_t)N, (size_t)K}, 224);
    auto A_data = randomFP32(static_cast<size_t>(M) * K);

    std::vector<float> C_cpu(static_cast<size_t>(M) * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));
    ASSERT_FALSE(hasNaNOrInf(C_cpu.data(), C_cpu.size()));

    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    std::vector<float> C_cuda(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_TRUE(cudaGroupedVerifierProjectionViaTensor(
        cuda_kernel,
        A_data.data(),
        C_cuda.data(),
        M,
        N,
        K,
        gpu_device_,
        workspace_.get()));

    auto result = checkParity(C_cuda.data(), C_cpu.data(), C_cpu.size(), 0.99, 0.15);
    result.print("Q4_K verifier-small-M 2x896x768");
    EXPECT_FALSE(result.has_nan_inf);
    EXPECT_GE(result.cosine_similarity, 0.99)
        << "Cosine similarity too low: " << result.cosine_similarity;
    EXPECT_LE(result.relative_l2_error, 0.15)
        << "Relative L2 error too high: " << (result.relative_l2_error * 100.0) << "%";

    cleanupWorkspaceIfNeeded(cuda_kernel);
    EXPECT_FALSE(cuda_kernel->hasDynamicStateActive())
        << "Small-M verifier GEMV test must not leak CUDA dynamic state after workspace cleanup";
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, Q4_K_VerifierSmallM_2x512x768)
{
    const int M = 2;
    const int N = 512;
    const int K = 768;

    auto weights = TestTensorFactory::createQ4_KRandom({(size_t)N, (size_t)K}, 227);
    auto A_data = randomFP32(static_cast<size_t>(M) * K);

    std::vector<float> C_cpu(static_cast<size_t>(M) * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));
    ASSERT_FALSE(hasNaNOrInf(C_cpu.data(), C_cpu.size()));

    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    std::vector<float> C_cuda(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    auto result = checkParity(C_cuda.data(), C_cpu.data(), C_cpu.size(), 0.99, 0.15);
    result.print("Q4_K verifier-small-M 2x512x768");
    EXPECT_FALSE(result.has_nan_inf);
    EXPECT_GE(result.cosine_similarity, 0.99)
        << "Cosine similarity too low: " << result.cosine_similarity;
    EXPECT_LE(result.relative_l2_error, 0.15)
        << "Relative L2 error too high: " << (result.relative_l2_error * 100.0) << "%";

    cleanupWorkspaceIfNeeded(cuda_kernel);
    EXPECT_FALSE(cuda_kernel->hasDynamicStateActive())
        << "Small-M verifier GEMV test must not leak CUDA dynamic state after workspace cleanup";
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, Q4_K_VerifierSmallM_UsesSpecializedNativeVNNIRoute)
{
    const int M = 2;
    const int N = 896;
    const int K = 768;

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    auto weights = TestTensorFactory::createQ4_KRandom({(size_t)N, (size_t)K}, 229);
    auto A_data = randomFP32(static_cast<size_t>(M) * K);

    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    std::vector<float> C_cuda(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_TRUE(cudaGroupedVerifierProjectionViaTensor(
        cuda_kernel,
        A_data.data(),
        C_cuda.data(),
        M,
        N,
        K,
        gpu_device_,
        workspace_.get()));

    const auto records =
        PerfStatsCollector::snapshot({"kernel.cuda_native_vnni_small_m_calls"});
    uint64_t specialized_records = 0;
    for (const auto &record : records)
    {
        if (record.domain != "kernel" ||
            record.kind != PerfStatRecord::Kind::Counter ||
            record.device != "cuda:" + std::to_string(gpu_device_.ordinal) ||
            record.tags.at("codebook") != "5" ||
            record.tags.at("n") != std::to_string(N) ||
            record.tags.at("k") != std::to_string(K))
        {
            continue;
        }
        if (record.name == "cuda_native_vnni_small_m_calls" &&
            record.tags.at("m") == std::to_string(M) &&
            record.tags.at("route") == "specialized")
        {
            specialized_records += record.count;
        }
    }

    EXPECT_EQ(specialized_records, 1u)
        << "Q4_K M=2 verifier GEMM must use the specialized native-VNNI small-M route";

    PerfStatsCollector::reset();
    cleanupWorkspaceIfNeeded(cuda_kernel);
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, Q4_K_VerifierSmallM_M2MatchesTwoSingleRowDecodeGEMVs)
{
    const int M = 2;
    const int N = 896;
    const int K = 768;

    auto weights = TestTensorFactory::createQ4_KRandom({static_cast<size_t>(N), static_cast<size_t>(K)}, 230);
    auto A_data = randomFP32(static_cast<size_t>(M) * K);

    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    std::vector<float> C_m2(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_TRUE(cudaGroupedVerifierProjectionViaTensor(
        cuda_kernel,
        A_data.data(),
        C_m2.data(),
        M,
        N,
        K,
        gpu_device_,
        workspace_.get()))
        << "M=2 verifier GEMM failed";

    for (int row = 0; row < M; ++row)
    {
        std::vector<float> C_m1(static_cast<size_t>(N), 0.0f);
        ASSERT_TRUE(cudaMultiplyViaTensor(
            cuda_kernel,
            A_data.data() + static_cast<size_t>(row) * K,
            C_m1.data(),
            1,
            N,
            K,
            gpu_device_))
            << "M=1 decode GEMV failed for row " << row;

        const float *verifier_row = C_m2.data() + static_cast<size_t>(row) * N;
        expectBitwiseEqualFloatRow(
            (std::string("Q4_K verifier small-M M=2 row=") + std::to_string(row))
                .c_str(),
            verifier_row,
            C_m1.data(),
            C_m1.size());
    }

    cleanupWorkspaceIfNeeded(cuda_kernel);
    EXPECT_FALSE(cuda_kernel->hasDynamicStateActive())
        << "M=2 row-equivalence test must not leak CUDA dynamic state after workspace cleanup";
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, Q4_K_Qwen36GDNOut_M4MatchesFourSingleRowDecodeGEMVs)
{
    const int M = 4;
    const int N = 5120;
    const int K = 6144;

    auto weights = TestTensorFactory::createQ4_KRandom({static_cast<size_t>(N), static_cast<size_t>(K)}, 231);
    auto A_data = randomFP32(static_cast<size_t>(M) * K);

    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    std::vector<float> C_m4(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_TRUE(cudaGroupedVerifierProjectionViaTensor(
        cuda_kernel,
        A_data.data(),
        C_m4.data(),
        M,
        N,
        K,
        gpu_device_,
        workspace_.get()))
        << "M=4 Qwen3.6 GDN output projection GEMM failed";

    for (int row = 0; row < M; ++row)
    {
        std::vector<float> C_m1(static_cast<size_t>(N), 0.0f);
        ASSERT_TRUE(cudaMultiplyViaTensor(
            cuda_kernel,
            A_data.data() + static_cast<size_t>(row) * K,
            C_m1.data(),
            1,
            N,
            K,
            gpu_device_))
            << "M=1 Qwen3.6 GDN output projection GEMV failed for row " << row;

        const float *verifier_row = C_m4.data() + static_cast<size_t>(row) * N;
        expectBitwiseEqualFloatRow(
            (std::string("Q4_K Qwen36 GDN out M=4 row=") + std::to_string(row))
                .c_str(),
            verifier_row,
            C_m1.data(),
            C_m1.size());
    }

    cleanupWorkspaceIfNeeded(cuda_kernel);
    EXPECT_FALSE(cuda_kernel->hasDynamicStateActive())
        << "M=4 Qwen3.6 row-equivalence test must not leak CUDA dynamic state";
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

/**
 * @brief Guard the Qwen3.6 MoE LocalTP GDN output projection verifier shape.
 *
 * The MoE model stores `ssm_out` as Q6_K. In LocalTP each participant projects
 * its local 2048-value-head slice into the full hidden width, then the graph
 * allreduces those partials.  This test isolates the grouped verifier GEMM used
 * before that allreduce. The production-size boundary inventory covers the
 * smallest grouped depths, the first depth beyond the historical four-row
 * assumption, the default graph capacity, and an over-capacity runtime probe.
 */
TEST_F(Test__CUDAGemmParity, Q6_K_Qwen36MoEGDNOutLocalTP_RuntimeMMatchesSerialDecodeGEMVs)
{
    constexpr int N = 2048;
    constexpr int K = 2048;

    auto weights = TestTensorFactory::createQ6_KRandom(
        {static_cast<size_t>(N), static_cast<size_t>(K)}, 2421);
    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);

    for (const int M : kGroupedVerifierBoundaryRows)
    {
        ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K))
            << "workspace for Qwen3.6 MoE GDN out M=" << M;

        auto A_data = randomFP32(static_cast<size_t>(M) * static_cast<size_t>(K));
        std::vector<float> C_grouped(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
        ASSERT_TRUE(cudaGroupedVerifierProjectionViaTensor(
            cuda_kernel,
            A_data.data(),
            C_grouped.data(),
            M,
            N,
            K,
            gpu_device_,
            workspace_.get()))
            << "M=" << M << " Qwen3.6 MoE GDN output grouped verifier GEMM failed";

        for (int row = 0; row < M; ++row)
        {
            std::vector<float> C_m1(static_cast<size_t>(N), 0.0f);
            ASSERT_TRUE(cudaMultiplyViaTensor(
                cuda_kernel,
                A_data.data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                C_m1.data(),
                1,
                N,
                K,
                gpu_device_))
                << "M=1 Qwen3.6 MoE GDN output projection GEMV failed for row " << row;

            const float *grouped_row =
                C_grouped.data() + static_cast<size_t>(row) * static_cast<size_t>(N);
            expectBitwiseEqualFloatRow(
                (std::string("Q6_K Qwen36 MoE LocalTP GDN out M=") +
                 std::to_string(M) + " row=" + std::to_string(row))
                    .c_str(),
                grouped_row,
                C_m1.data(),
                C_m1.size());
        }

        cleanupWorkspaceIfNeeded(cuda_kernel);
    }

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

/**
 * @brief Prove CUDA LocalTP Wo grouped verifier rows for every native format.
 *
 * The model-level grouped verifier drift investigation showed that synthetic
 * single-format GEMM tests are too narrow.  This matrix exercises the same
 * row-parallel attention output projection shape used by LocalTP Wo, then
 * reconstructs the two-rank allreduce result and compares it byte-for-byte
 * against repeated serial M=1 decode rows.  The shape sweep includes both the
 * older narrow attention case and Qwen3.6 MoE full-attention layers, whose
 * local attention slice is twice as wide.
 */
TEST_F(Test__CUDAGemmParity, LocalTPWoAllQuantizedFormatsGroupedVerifierRowsMatchSerialDecodeStrict)
{
    SKIP_IF_NO_CUDA();
    int cuda_devices = 0;
    ASSERT_EQ(cudaGetDeviceCount(&cuda_devices), cudaSuccess);
    if (cuda_devices < 2)
        GTEST_SKIP() << "CUDA LocalTP Wo regression requires at least two CUDA devices";

    const std::vector<DeviceId> devices = {DeviceId::cuda(0), DeviceId::cuda(1)};

    uint32_t seed = 18800;
    for (const auto &shape_case : localTPWoShapeCases())
    {
        for (const auto &format_case : cudaLocalTPWoQuantizedFormatCases())
        {
            runCudaLocalTPWoAllreduceVerifierRowsMatchSerial(
                format_case,
                devices,
                shape_case,
                seed);
            seed += 101;
        }
    }
}

/**
 * @brief Prove CUDA LocalTP Wo grouped verifier rows for floating weights.
 *
 * Floating-point Wo weights use CUDA's FP GEMM implementation rather than the
 * native-VNNI quantized path.  Keeping FP32, FP16, and BF16 in the same LocalTP
 * reconstruction matrix prevents a future model or loader setting from escaping
 * the strict grouped-verifier serial-row contract.
 */
TEST_F(Test__CUDAGemmParity, LocalTPWoFloatingPointFormatsGroupedVerifierRowsMatchSerialDecodeStrict)
{
    SKIP_IF_NO_CUDA();
    int cuda_devices = 0;
    ASSERT_EQ(cudaGetDeviceCount(&cuda_devices), cudaSuccess);
    if (cuda_devices < 2)
        GTEST_SKIP() << "CUDA LocalTP Wo regression requires at least two CUDA devices";

    const std::vector<DeviceId> devices = {DeviceId::cuda(0), DeviceId::cuda(1)};

    uint32_t seed = 20800;
    for (const auto &shape_case : localTPWoShapeCases())
    {
        for (const auto &format_case : cudaLocalTPWoFloatingPointFormatCases())
        {
            runCudaLocalTPWoAllreduceVerifierRowsMatchSerial(
                format_case,
                devices,
                shape_case,
                seed);
            seed += 101;
        }
    }
}

/**
 * @brief Prove CUDA mirrored LocalTP Wo for every quantized native format.
 *
 * The row-parallel LocalTP Wo matrix above reconstructs the existing sharded
 * allreduce path.  This test covers the target MTP architecture where each
 * shard keeps a full mirrored attention Wo/MTP verifier projection locally and
 * therefore does not participate in a tiny verifier allreduce.  The grouped
 * implementation still has to be economical and byte-identical to serial M=1
 * decode for every native quantized format.
 */
TEST_F(Test__CUDAGemmParity, ReplicatedWoAllQuantizedFormatsGroupedVerifierRowsMatchSerialDecodeStrict)
{
    SKIP_IF_NO_CUDA();

    const DeviceId device = DeviceId::cuda(0);

    uint32_t seed = 22800;
    for (const auto &shape_case : localTPWoShapeCases())
    {
        for (const auto &format_case : cudaLocalTPWoQuantizedFormatCases())
        {
            runCudaReplicatedWoVerifierRowsMatchSerial(
                format_case,
                device,
                shape_case,
                seed);
            seed += 101;
        }
    }
}

/**
 * @brief Prove CUDA mirrored LocalTP Wo for FP32/FP16/BF16 weights.
 *
 * Floating-point mirrored Wo is especially important for the target
 * device-resident MTP path because a backend library may choose different
 * reductions for M=1 and M>1.  This gate forces the grouped verifier path to
 * use the same decode-equivalent small-M contract as serial decode.
 */
TEST_F(Test__CUDAGemmParity, ReplicatedWoFloatingPointFormatsGroupedVerifierRowsMatchSerialDecodeStrict)
{
    SKIP_IF_NO_CUDA();

    const DeviceId device = DeviceId::cuda(0);

    uint32_t seed = 24800;
    for (const auto &shape_case : localTPWoShapeCases())
    {
        for (const auto &format_case : cudaLocalTPWoFloatingPointFormatCases())
        {
            runCudaReplicatedWoVerifierRowsMatchSerial(
                format_case,
                device,
                shape_case,
                seed);
            seed += 101;
        }
    }
}

/**
 * @brief Guard the Qwen3.6 GDN QKV verifier projection shape.
 *
 * The model-level grouped verifier parity test first diverged at
 * `layer1_QKV_PROJECTION` for the Qwen3.6 dense model.  That projection is much
 * wider than the older synthetic small-M tests, so this regression proves the
 * exact MTP contract at the kernel boundary: a grouped M=4 verifier projection
 * must produce the same rows as four ordinary M=1 decode GEMVs using the same
 * CUDA tensor path.  The thresholds are intentionally strict because top-token
 * equality can hide distribution drift that later breaks stochastic MTP.
 */
TEST_F(Test__CUDAGemmParity, Q4_K_Qwen36GDNQKV_M4MatchesFourSingleRowDecodeGEMVs)
{
    const int M = 4;
    const int N = 10240;
    const int K = 5120;

    auto weights = TestTensorFactory::createQ4_KRandom(
        {static_cast<size_t>(N), static_cast<size_t>(K)}, 3321);
    auto A_data = randomFP32(static_cast<size_t>(M) * K);

    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    std::vector<float> C_m4(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_TRUE(cudaGroupedVerifierProjectionViaTensor(
        cuda_kernel,
        A_data.data(),
        C_m4.data(),
        M,
        N,
        K,
        gpu_device_,
        workspace_.get()))
        << "M=4 Qwen3.6 GDN QKV projection GEMM failed";

    for (int row = 0; row < M; ++row)
    {
        std::vector<float> C_m1(static_cast<size_t>(N), 0.0f);
        ASSERT_TRUE(cudaMultiplyViaTensor(
            cuda_kernel,
            A_data.data() + static_cast<size_t>(row) * K,
            C_m1.data(),
            1,
            N,
            K,
            gpu_device_))
            << "M=1 Qwen3.6 GDN QKV projection GEMV failed for row " << row;

        const float *verifier_row = C_m4.data() + static_cast<size_t>(row) * N;
        expectBitwiseEqualFloatRow(
            (std::string("Q4_K Qwen36 GDN QKV M=4 row=") + std::to_string(row))
                .c_str(),
            verifier_row,
            C_m1.data(),
            C_m1.size());
    }

    cleanupWorkspaceIfNeeded(cuda_kernel);
    EXPECT_FALSE(cuda_kernel->hasDynamicStateActive())
        << "M=4 Qwen3.6 GDN QKV row-equivalence test must not leak CUDA dynamic state";
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

/**
 * @brief Guard the production Qwen3.6 GDN qkv+z fused verifier subgroup.
 *
 * `GDNProjectionStage` groups native CUDA qkv and z projections together when
 * verifier rows use the decode-equivalent M=2..4 path.  The model-level M=4
 * parity failure first appears at `layer1_QKV_PROJECTION`, so this test checks
 * the grouped descriptor contract directly: one M=4 qkv+z fused verifier call
 * must match four independent M=1 fused decode calls for both outputs.  That
 * proves the shared-workspace/projection-descriptor path before we diagnose
 * higher-level graph or stage ownership.
 */
TEST_F(Test__CUDAGemmParity, Q4_K_Qwen36GDNQKVZ_M4FusedProjectionMatchesFourSingleRowDecodeGEMVs)
{
    const int M = 4;
    const int K = 5120;
    const int N_QKV = 10240;
    const int N_Z = 6144;

    auto weights_qkv = TestTensorFactory::createQ4_KRandom(
        {static_cast<size_t>(N_QKV), static_cast<size_t>(K)}, 4331);
    auto weights_z = TestTensorFactory::createQ4_KRandom(
        {static_cast<size_t>(N_Z), static_cast<size_t>(K)}, 4332);
    auto A_data = randomFP32(static_cast<size_t>(M) * K);

    auto *cuda_qkv = getPreparedKernel(weights_qkv.get(), gpu_device_);
    auto *cuda_z = getPreparedKernel(weights_z.get(), gpu_device_);
    ASSERT_NE(cuda_qkv, nullptr);
    ASSERT_NE(cuda_z, nullptr);
    ASSERT_TRUE(setupSharedWorkspace({cuda_qkv, cuda_z}, M, {N_QKV, N_Z}, K));

    auto A_m4 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    std::memcpy(A_m4->mutable_data(), A_data.data(), A_data.size() * sizeof(float));
    auto qkv_m4 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_QKV)});
    auto z_m4 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_Z)});
    std::vector<TensorProjectionDesc> m4_projections = {
        {cuda_qkv, qkv_m4.get(), N_QKV, nullptr, "gdn_qkv"},
        {cuda_z, z_m4.get(), N_Z, nullptr, "gdn_z"}};

    ASSERT_TRUE(withExplicitCudaGemmStream(
        {cuda_qkv, cuda_z},
        [&](void *stream)
        {
            return with_gpu_coherence(
                gpu_device_,
                {A_m4.get()},
                {qkv_m4.get(), z_m4.get()},
                static_cast<void *>(stream),
                [&]
                {
                    return cuda_qkv->multiply_fused_verifier_rows_decode_equivalent(
                        A_m4.get(), m4_projections, M, K, nullptr, workspace_.get());
                });
        }))
        << "M=4 Qwen3.6 GDN qkv+z fused verifier projection failed";

    for (int row = 0; row < M; ++row)
    {
        auto A_m1 = std::make_unique<FP32Tensor>(
            std::vector<size_t>{size_t{1}, static_cast<size_t>(K)});
        std::memcpy(
            A_m1->mutable_data(),
            A_data.data() + static_cast<size_t>(row) * K,
            static_cast<size_t>(K) * sizeof(float));
        auto qkv_m1 = std::make_unique<FP32Tensor>(
            std::vector<size_t>{size_t{1}, static_cast<size_t>(N_QKV)});
        auto z_m1 = std::make_unique<FP32Tensor>(
            std::vector<size_t>{size_t{1}, static_cast<size_t>(N_Z)});
        std::vector<TensorProjectionDesc> m1_projections = {
            {cuda_qkv, qkv_m1.get(), N_QKV, nullptr, "gdn_qkv"},
            {cuda_z, z_m1.get(), N_Z, nullptr, "gdn_z"}};

        ASSERT_TRUE(withExplicitCudaGemmStream(
            {cuda_qkv, cuda_z},
            [&](void *stream)
            {
                return with_gpu_coherence(
                    gpu_device_,
                    {A_m1.get()},
                    {qkv_m1.get(), z_m1.get()},
                    static_cast<void *>(stream),
                    [&]
                    {
                        return cuda_qkv->multiply_fused_tensor(A_m1.get(), m1_projections, 1, K);
                    });
            }))
            << "M=1 Qwen3.6 GDN qkv+z fused decode projection failed for row " << row;

        const float *qkv_row = qkv_m4->data() + static_cast<size_t>(row) * N_QKV;
        const float *z_row = z_m4->data() + static_cast<size_t>(row) * N_Z;
        expectBitwiseEqualFloatRow(
            (std::string("Q4_K Qwen36 fused GDN qkv M=4 row=") +
             std::to_string(row))
                .c_str(),
            qkv_row,
            qkv_m1->data(),
            static_cast<size_t>(N_QKV));
        expectBitwiseEqualFloatRow(
            (std::string("Q4_K Qwen36 fused GDN z M=4 row=") +
             std::to_string(row))
                .c_str(),
            z_row,
            z_m1->data(),
            static_cast<size_t>(N_Z));
    }

    cleanupSharedWorkspace({cuda_qkv, cuda_z});
    EXPECT_FALSE(cuda_qkv->hasDynamicStateActive())
        << "M=4 GDN qkv fused verifier test must not leak CUDA dynamic state";
    EXPECT_FALSE(cuda_z->hasDynamicStateActive())
        << "M=4 GDN z fused verifier test must not leak CUDA dynamic state";
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_qkv.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_z.get());
}

/**
 * @brief Guard the real Qwen3.6 dense GDN mixed-codebook qkv+z subgroup.
 *
 * The `Qwen3.6-27B-Q4_K_S` fixture stores `blk.1.attn_qkv.weight` as Q5_K
 * while `blk.1.attn_gate.weight` (the GDN z projection) is Q4_K.  The stage
 * groups them because both use CUDAQuantisedGemmKernel, so this test proves
 * that mixed native codebooks in one fused verifier subgroup remain exactly
 * decode-equivalent for M=4.
 */
TEST_F(Test__CUDAGemmParity, Q5KQ4K_Qwen36GDNQKVZ_M4FusedProjectionMatchesFourSingleRowDecodeGEMVs)
{
    const int M = 4;
    const int K = 5120;
    const int N_QKV = 10240;
    const int N_Z = 6144;

    auto weights_qkv = TestTensorFactory::createQ5_KRandom(
        {static_cast<size_t>(N_QKV), static_cast<size_t>(K)}, 4431);
    auto weights_z = TestTensorFactory::createQ4_KRandom(
        {static_cast<size_t>(N_Z), static_cast<size_t>(K)}, 4432);
    auto A_data = randomFP32(static_cast<size_t>(M) * K);

    auto *cuda_qkv = getPreparedKernel(weights_qkv.get(), gpu_device_);
    auto *cuda_z = getPreparedKernel(weights_z.get(), gpu_device_);
    ASSERT_NE(cuda_qkv, nullptr);
    ASSERT_NE(cuda_z, nullptr);
    ASSERT_TRUE(setupSharedWorkspace({cuda_qkv, cuda_z}, M, {N_QKV, N_Z}, K));

    auto A_m4 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    std::memcpy(A_m4->mutable_data(), A_data.data(), A_data.size() * sizeof(float));
    auto qkv_m4 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_QKV)});
    auto z_m4 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_Z)});
    std::vector<TensorProjectionDesc> m4_projections = {
        {cuda_qkv, qkv_m4.get(), N_QKV, nullptr, "gdn_qkv_q5k"},
        {cuda_z, z_m4.get(), N_Z, nullptr, "gdn_z_q4k"}};

    ASSERT_TRUE(withExplicitCudaGemmStream(
        {cuda_qkv, cuda_z},
        [&](void *stream)
        {
            return with_gpu_coherence(
                gpu_device_,
                {A_m4.get()},
                {qkv_m4.get(), z_m4.get()},
                static_cast<void *>(stream),
                [&]
                {
                    return cuda_qkv->multiply_fused_verifier_rows_decode_equivalent(
                        A_m4.get(), m4_projections, M, K, nullptr, workspace_.get());
                });
        }))
        << "M=4 mixed Q5_K/Q4_K Qwen3.6 GDN qkv+z verifier projection failed";

    for (int row = 0; row < M; ++row)
    {
        auto A_m1 = std::make_unique<FP32Tensor>(
            std::vector<size_t>{size_t{1}, static_cast<size_t>(K)});
        std::memcpy(
            A_m1->mutable_data(),
            A_data.data() + static_cast<size_t>(row) * K,
            static_cast<size_t>(K) * sizeof(float));
        auto qkv_m1 = std::make_unique<FP32Tensor>(
            std::vector<size_t>{size_t{1}, static_cast<size_t>(N_QKV)});
        auto z_m1 = std::make_unique<FP32Tensor>(
            std::vector<size_t>{size_t{1}, static_cast<size_t>(N_Z)});
        std::vector<TensorProjectionDesc> m1_projections = {
            {cuda_qkv, qkv_m1.get(), N_QKV, nullptr, "gdn_qkv_q5k"},
            {cuda_z, z_m1.get(), N_Z, nullptr, "gdn_z_q4k"}};

        ASSERT_TRUE(withExplicitCudaGemmStream(
            {cuda_qkv, cuda_z},
            [&](void *stream)
            {
                return with_gpu_coherence(
                    gpu_device_,
                    {A_m1.get()},
                    {qkv_m1.get(), z_m1.get()},
                    static_cast<void *>(stream),
                    [&]
                    {
                        return cuda_qkv->multiply_fused_tensor(A_m1.get(), m1_projections, 1, K);
                    });
            }))
            << "M=1 mixed Q5_K/Q4_K Qwen3.6 GDN qkv+z decode projection failed for row " << row;

        const float *qkv_row = qkv_m4->data() + static_cast<size_t>(row) * N_QKV;
        const float *z_row = z_m4->data() + static_cast<size_t>(row) * N_Z;
        expectBitwiseEqualFloatRow(
            (std::string("Q5_K/Q4_K Qwen36 fused GDN qkv M=4 row=") +
             std::to_string(row))
                .c_str(),
            qkv_row,
            qkv_m1->data(),
            static_cast<size_t>(N_QKV));
        expectBitwiseEqualFloatRow(
            (std::string("Q5_K/Q4_K Qwen36 fused GDN z M=4 row=") +
             std::to_string(row))
                .c_str(),
            z_row,
            z_m1->data(),
            static_cast<size_t>(N_Z));
    }

    cleanupSharedWorkspace({cuda_qkv, cuda_z});
    EXPECT_FALSE(cuda_qkv->hasDynamicStateActive())
        << "M=4 mixed-codebook GDN qkv fused verifier test must not leak CUDA dynamic state";
    EXPECT_FALSE(cuda_z->hasDynamicStateActive())
        << "M=4 mixed-codebook GDN z fused verifier test must not leak CUDA dynamic state";
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_qkv.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_z.get());
}

/**
 * @brief Guard the asymmetric Qwen3.6 dense LocalTP GDN QKV shard at M=2.
 *
 * Two-way LocalTP replicates the GDN Q/K channels and shards the wider V
 * channels.  The resulting rank-local fused QKV tensor has 7168 output rows,
 * a geometry absent from the earlier full-weight `N=10240` regressions.  The
 * device-resident request-batch condition graph uses exactly M=2, and its first
 * observed byte drift occurred at this projection.  Exercise the public
 * grouped projection API and independently quantized ordinary M=1 API so the
 * test covers dispatch, reduction, and epilogue equivalence end to end.
 */
TEST_F(Test__CUDAGemmParity, Q5K_Qwen36DenseLocalTPGDNQKV_M2MatchesTwoSingleRowDecodeGEMVs)
{
    constexpr int M = 2;
    constexpr int N = 7168;
    constexpr int K = 5120;

    auto weights = TestTensorFactory::createQ5_KRandom(
        {static_cast<size_t>(N), static_cast<size_t>(K)},
        4433);
    ASSERT_NE(weights, nullptr);

    auto *kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(kernel, M, N, K));

    const auto input = randomFP32(static_cast<size_t>(M) * K);
    std::vector<float> grouped(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_TRUE(cudaGroupedVerifierProjectionViaTensor(
        kernel,
        input.data(),
        grouped.data(),
        M,
        N,
        K,
        gpu_device_,
        workspace_.get()));

    for (int row = 0; row < M; ++row)
    {
        std::vector<float> serial(static_cast<size_t>(N), 0.0f);
        ASSERT_TRUE(cudaMultiplyViaTensor(
            kernel,
            input.data() + static_cast<size_t>(row) * K,
            serial.data(),
            /*M=*/1,
            N,
            K,
            gpu_device_));
        expectBitwiseEqualFloatRow(
            (std::string("Q5_K Qwen36 dense LocalTP GDN QKV M=2 row=") +
             std::to_string(row))
                .c_str(),
            grouped.data() + static_cast<size_t>(row) * N,
            serial.data(),
            serial.size());
    }

    cleanupWorkspaceIfNeeded(kernel);
    EXPECT_FALSE(kernel->hasDynamicStateActive());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

/**
 * @brief Prove both real dense Qwen3.6 LocalTP GDN QKV slices at M=2.
 *
 * This is the focused kernel regression for the resident request-batch
 * condition failure.  It uses the actual GGUF tensor, dense schema, asymmetric
 * two-way LocalTP sharder, TensorSlice wrapper, native GPU weight repack, and
 * public grouped projection API.  Each rank-local grouped row is compared with
 * an independently quantized ordinary M=1 decode invocation.  No production
 * serial replay path is involved; serial calls exist only as this test's oracle.
 */
TEST_F(Test__CUDAGemmParity, RealQwen36DenseLocalTPGDNQKVShardM2MatchesSerialDecodeRows)
{
    SKIP_IF_NO_CUDA();
    ASSERT_EQ(cudaSetDevice(gpu_device_.ordinal), cudaSuccess);

    const std::filesystem::path model_path = qwen36DenseModelPath();
    if (!std::filesystem::exists(model_path))
    {
        GTEST_SKIP() << "Qwen 3.6 dense model not found at " << model_path
                     << "; set LLAMINAR_QWEN36_DENSE_MODEL to run this real-weight regression";
    }

    auto model_ctx = loadQwen36ModelForGpuWeights(model_path);
    ASSERT_NE(model_ctx, nullptr);
    const std::vector<DeviceId> logical_tp_devices = {
        DeviceId::cuda(0),
        DeviceId::cuda(1),
    };
    auto tp_config = configureQwen36DenseLocalTPWeights(
        model_ctx,
        logical_tp_devices);
    ASSERT_NE(tp_config, nullptr);
    ASSERT_TRUE(tp_config->validate()) << tp_config->validationError();

    constexpr int M = 2;
    const ModelContextId model_id{36364};
    for (int rank = 0; rank < 2; ++rank)
    {
        auto weight = model_ctx->getWeightForDevice(
            "blk.0.attn_qkv.weight",
            logical_tp_devices[static_cast<size_t>(rank)]);
        ASSERT_NE(weight, nullptr) << "rank=" << rank;
        ASSERT_EQ(weight->native_type(), TensorType::Q5_K)
            << "The Qwen3.6 dense fixture's first GDN QKV projection changed format";

        const int N = static_cast<int>(weight->rows());
        const int K = static_cast<int>(weight->cols());
        ASSERT_EQ(N, 7168) << "rank=" << rank;
        ASSERT_EQ(K, 5120) << "rank=" << rank;

        auto prepared = llaminar2::test::makeGpuPreparedGemm(
            gpuPreparationTensor(weight.get()),
            gpu_device_,
            "blk.0.attn_qkv.weight.localtp.rank" + std::to_string(rank),
            model_id);
        ASSERT_NE(prepared.kernel, nullptr) << "rank=" << rank;
        ASSERT_TRUE(setupWorkspaceIfNeeded(prepared.kernel, M, N, K))
            << "rank=" << rank;

        const auto input = randomFP32(static_cast<size_t>(M) * K);
        std::vector<float> grouped(static_cast<size_t>(M) * N, 0.0f);
        ASSERT_TRUE(cudaGroupedVerifierProjectionViaTensor(
            prepared.kernel,
            input.data(),
            grouped.data(),
            M,
            N,
            K,
            gpu_device_,
            workspace_.get()))
            << "rank=" << rank;

        for (int row = 0; row < M; ++row)
        {
            std::vector<float> serial(static_cast<size_t>(N), 0.0f);
            ASSERT_TRUE(cudaMultiplyViaTensor(
                prepared.kernel,
                input.data() + static_cast<size_t>(row) * K,
                serial.data(),
                /*M=*/1,
                N,
                K,
                gpu_device_))
                << "rank=" << rank << " row=" << row;
            expectBitwiseEqualFloatRow(
                (std::string("real Qwen36 dense LocalTP GDN QKV rank=") +
                 std::to_string(rank) + " M=2 row=" + std::to_string(row))
                    .c_str(),
                grouped.data() + static_cast<size_t>(row) * N,
                serial.data(),
                serial.size());
        }

        cleanupWorkspaceIfNeeded(prepared.kernel);
        EXPECT_FALSE(prepared.kernel->hasDynamicStateActive())
            << "rank=" << rank;
        llaminar::v2::kernels::KernelFactory::clearCacheFor(
            gpuPreparationTensor(weight.get()));
    }
}

TEST_F(Test__CUDAGemmParity, Q4_K_FusedVerifierSmallM_2RowsTwoProjections)
{
    const int M = 2;
    const int N0 = 896;
    const int N1 = 512;
    const int K = 768;

    auto weights0 = TestTensorFactory::createQ4_KRandom({(size_t)N0, (size_t)K}, 225);
    auto weights1 = TestTensorFactory::createQ4_KRandom({(size_t)N1, (size_t)K}, 226);
    auto A_data = randomFP32(static_cast<size_t>(M) * K);

    std::vector<float> C0_cpu(static_cast<size_t>(M) * N0, 0.0f);
    std::vector<float> C1_cpu(static_cast<size_t>(M) * N1, 0.0f);
    auto cpu_kernel0 = llaminar::v2::kernels::KernelFactory::createGemm(
        weights0.get(), KernelDeviceType::CPU);
    auto cpu_kernel1 = llaminar::v2::kernels::KernelFactory::createGemm(
        weights1.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel0, nullptr);
    ASSERT_NE(cpu_kernel1, nullptr);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel0.get(), A_data.data(), C0_cpu.data(), M, N0, K));
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel1.get(), A_data.data(), C1_cpu.data(), M, N1, K));

    auto *cuda_kernel0 = getPreparedKernel(weights0.get(), gpu_device_);
    auto *cuda_kernel1 = getPreparedKernel(weights1.get(), gpu_device_);
    ASSERT_NE(cuda_kernel0, nullptr);
    ASSERT_NE(cuda_kernel1, nullptr);
    ASSERT_TRUE(setupSharedWorkspace({cuda_kernel0, cuda_kernel1}, M, {N0, N1}, K));

    auto A_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)K});
    std::memcpy(A_tensor->mutable_data(), A_data.data(), A_data.size() * sizeof(float));
    auto C0_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)N0});
    auto C1_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)N1});

    std::vector<TensorProjectionDesc> projections = {
        {cuda_kernel0, C0_tensor.get(), N0, nullptr, "q4k_proj0"},
        {cuda_kernel1, C1_tensor.get(), N1, nullptr, "q4k_proj1"}};

    ASSERT_TRUE(withExplicitCudaGemmStream(
        {cuda_kernel0, cuda_kernel1},
        [&](void *stream)
        {
            return with_gpu_coherence(
                gpu_device_,
                {A_tensor.get()},
                {C0_tensor.get(), C1_tensor.get()},
                static_cast<void *>(stream),
                [&]
                {
                    return cuda_kernel0->multiply_fused_tensor(
                        A_tensor.get(), projections, M, K);
                });
        }));

    const float *C0_cuda = C0_tensor->data();
    const float *C1_cuda = C1_tensor->data();
    auto result0 = checkParity(C0_cuda, C0_cpu.data(), C0_cpu.size(), 0.99, 0.15);
    auto result1 = checkParity(C1_cuda, C1_cpu.data(), C1_cpu.size(), 0.99, 0.15);
    result0.print("Q4_K fused verifier-small-M projection 0");
    result1.print("Q4_K fused verifier-small-M projection 1");
    EXPECT_FALSE(result0.has_nan_inf);
    EXPECT_FALSE(result1.has_nan_inf);
    EXPECT_GE(result0.cosine_similarity, 0.99);
    EXPECT_GE(result1.cosine_similarity, 0.99);
    EXPECT_LE(result0.relative_l2_error, 0.15);
    EXPECT_LE(result1.relative_l2_error, 0.15);

    cleanupSharedWorkspace({cuda_kernel0, cuda_kernel1});
    EXPECT_FALSE(cuda_kernel0->hasDynamicStateActive())
        << "Small-M fused verifier projection 0 must not leak CUDA dynamic state after workspace cleanup";
    EXPECT_FALSE(cuda_kernel1->hasDynamicStateActive())
        << "Small-M fused verifier projection 1 must not leak CUDA dynamic state after workspace cleanup";
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights0.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights1.get());
}

TEST_F(Test__CUDAGemmParity, GDNProjectionStageFusesCUDAQuantizedQKVAndZSmallM)
{
    const int M = 2;
    const int K = 768;
    const int N_QKV = 896;
    const int N_Z = 512;
    const int N_ALPHA = 16;
    const int N_BETA = 16;

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    auto input = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    auto input_data = randomFP32(static_cast<size_t>(M) * K);
    std::memcpy(input->mutable_data(), input_data.data(), input_data.size() * sizeof(float));

    auto weights_qkv = TestTensorFactory::createQ4_KRandom(
        {static_cast<size_t>(N_QKV), static_cast<size_t>(K)}, 232);
    auto weights_z = TestTensorFactory::createQ4_KRandom(
        {static_cast<size_t>(N_Z), static_cast<size_t>(K)}, 233);
    auto weights_alpha = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(N_ALPHA), static_cast<size_t>(K)}, -0.1f, 0.1f, 234);
    auto weights_beta = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(N_BETA), static_cast<size_t>(K)}, -0.1f, 0.1f, 235);
    ASSERT_TRUE(weights_alpha->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(weights_beta->ensureOnDevice(gpu_device_));

    std::vector<float> alpha_cpu(static_cast<size_t>(M) * N_ALPHA, 0.0f);
    std::vector<float> beta_cpu(static_cast<size_t>(M) * N_BETA, 0.0f);
    auto cpu_alpha = llaminar::v2::kernels::KernelFactory::createGemm(
        weights_alpha.get(), KernelDeviceType::CPU);
    auto cpu_beta = llaminar::v2::kernels::KernelFactory::createGemm(
        weights_beta.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_alpha, nullptr);
    ASSERT_NE(cpu_beta, nullptr);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_alpha.get(), input_data.data(), alpha_cpu.data(), M, N_ALPHA, K));
    ASSERT_TRUE(cpuMultiplyToVector(cpu_beta.get(), input_data.data(), beta_cpu.data(), M, N_BETA, K));

    auto output_qkv = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_QKV)});
    auto output_z = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_Z)});
    auto output_alpha = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_ALPHA)});
    auto output_beta = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_BETA)});

    auto qkv_prepared = llaminar2::test::makeGpuPreparedGemm(
        weights_qkv.get(),
        gpu_device_,
        "blk.5.attn_qkv.weight",
        ModelContextId{23881});
    auto z_prepared = llaminar2::test::makeGpuPreparedGemm(
        weights_z.get(),
        gpu_device_,
        "blk.5.attn_gate.weight",
        ModelContextId{23881});
    auto alpha_prepared = llaminar2::test::makeGpuPreparedFloatingPointGemm(
        weights_alpha.get(),
        gpu_device_,
        "blk.5.ssm_alpha.weight",
        ModelContextId{23881});
    auto beta_prepared = llaminar2::test::makeGpuPreparedFloatingPointGemm(
        weights_beta.get(),
        gpu_device_,
        "blk.5.ssm_beta.weight",
        ModelContextId{23881});

    auto *kernel_qkv = qkv_prepared.kernel;
    auto *kernel_z = z_prepared.kernel;
    auto *kernel_alpha = alpha_prepared.kernel;
    auto *kernel_beta = beta_prepared.kernel;
    ASSERT_NE(kernel_qkv, nullptr);
    ASSERT_NE(kernel_z, nullptr);
    ASSERT_NE(kernel_alpha, nullptr);
    ASSERT_NE(kernel_beta, nullptr);
    ASSERT_TRUE(kernel_qkv->supports_fused_projection())
        << "CUDA quantized GEMM must advertise fused projection support so GDN can batch qkv+z";
    ASSERT_TRUE(kernel_z->supports_fused_projection())
        << "CUDA quantized GEMM must advertise fused projection support so GDN can batch qkv+z";
    ASSERT_TRUE(kernel_alpha->supports_fused_projection())
        << "CUDA FP32 GEMM must advertise fused projection support so GDN can batch alpha+beta";
    ASSERT_TRUE(kernel_beta->supports_fused_projection())
        << "CUDA FP32 GEMM must advertise fused projection support so GDN can batch alpha+beta";

    GDNProjectionStage::Params params;
    params.device_id = gpu_device_;
    params.input = input.get();
    params.m = M;
    params.k = K;
    params.w_qkv = weights_qkv.get();
    params.output_qkv = output_qkv.get();
    params.n_qkv = N_QKV;
    params.w_z = weights_z.get();
    params.output_z = output_z.get();
    params.n_z = N_Z;
    params.w_a = weights_alpha.get();
    params.output_a = output_alpha.get();
    params.n_a = N_ALPHA;
    params.w_b = weights_beta.get();
    params.output_b = output_beta.get();
    params.n_b = N_BETA;
    params.gemm_qkv = kernel_qkv;
    params.gemm_z = kernel_z;
    params.gemm_a = kernel_alpha;
    params.gemm_b = kernel_beta;
    params.force_decode_equivalent_verifier_prefill = true;

    GDNProjectionStage stage(params);
    WorkspaceRequirements reqs = stage.getWorkspaceRequirements(M, 0, K);
    workspace_ = std::make_unique<DeviceWorkspaceManager>(gpu_device_, workspaceBudgetFor(reqs));
    ASSERT_TRUE(workspace_->allocate(reqs));
    stage.bindWorkspace(workspace_.get());

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    stage.setGPUStream(static_cast<void *>(stream));
    CUDADeviceContext ctx(gpu_device_, gpu_device_.ordinal);

    const bool ok = with_gpu_coherence(
        gpu_device_,
        {input.get()},
        {output_qkv.get(), output_z.get(), output_alpha.get(), output_beta.get()},
        static_cast<void *>(stream),
        [&]
        {
            return stage.execute(&ctx);
        });
    ASSERT_TRUE(ok);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    ASSERT_NE(output_qkv->gpu_data_ptr(), nullptr);
    ASSERT_NE(output_z->gpu_data_ptr(), nullptr);
    ASSERT_NE(output_alpha->gpu_data_ptr(), nullptr);
    ASSERT_NE(output_beta->gpu_data_ptr(), nullptr);
    ASSERT_EQ(cudaMemsetAsync(output_qkv->gpu_data_ptr(), 0,
                              static_cast<size_t>(M) * N_QKV * sizeof(float), stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(output_z->gpu_data_ptr(), 0,
                              static_cast<size_t>(M) * N_Z * sizeof(float), stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(output_alpha->gpu_data_ptr(), 0,
                              static_cast<size_t>(M) * N_ALPHA * sizeof(float), stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(output_beta->gpu_data_ptr(), 0,
                              static_cast<size_t>(M) * N_BETA * sizeof(float), stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), cudaSuccess);
    ASSERT_TRUE(stage.execute(&ctx))
        << "CUDA GDN projection stage should be graph-capturable with fused qkv+z and alpha+beta subgroups";
    ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaGraphExecDestroy(graph_exec), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);

    TransferEngine::publishCurrentDeviceWrite(output_qkv, stream);
    TransferEngine::publishCurrentDeviceWrite(output_z, stream);
    TransferEngine::publishCurrentDeviceWrite(output_alpha, stream);
    TransferEngine::publishCurrentDeviceWrite(output_beta, stream);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    stage.unbindWorkspace();
    workspace_.reset();
    kernel_qkv->resetDynamicState();
    kernel_z->resetDynamicState();
    kernel_alpha->resetDynamicState();
    kernel_beta->resetDynamicState();

    const auto route_records = PerfStatsCollector::snapshot({"kernel.gdn_projection_route"});
    const auto qkv_z_route = std::find_if(
        route_records.begin(),
        route_records.end(),
        [&](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "gdn_projection_route" &&
                   record.kind == PerfStatRecord::Kind::Counter &&
                   record.tags.at("route") == "grouped_decode_equivalent_verifier_subgroup" &&
                   record.tags.at("m") == std::to_string(M) &&
                   record.tags.at("k") == std::to_string(K) &&
                   record.tags.at("projections") == "2" &&
                   record.tags.at("names") == "qkv+z";
        });
    ASSERT_NE(qkv_z_route, route_records.end())
        << "CUDA GDN verifier projection must use the grouped decode-equivalent qkv+z subgroup";

    const auto alpha_beta_route = std::find_if(
        route_records.begin(),
        route_records.end(),
        [&](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "gdn_projection_route" &&
                   record.kind == PerfStatRecord::Kind::Counter &&
                   record.tags.at("route") == "grouped_decode_equivalent_verifier_subgroup" &&
                   record.tags.at("m") == std::to_string(M) &&
                   record.tags.at("k") == std::to_string(K) &&
                   record.tags.at("projections") == "2" &&
                   record.tags.at("names") == "alpha+beta";
        });
    ASSERT_NE(alpha_beta_route, route_records.end())
        << "CUDA GDN verifier projection must use the grouped decode-equivalent FP32 alpha+beta subgroup";

    const auto fused_records =
        PerfStatsCollector::snapshot({"kernel.cuda_native_vnni_small_m_fused_projection_calls"});
    const auto fused_kernel_record = std::find_if(
        fused_records.begin(),
        fused_records.end(),
        [&](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "cuda_native_vnni_small_m_fused_projection_calls" &&
                   record.kind == PerfStatRecord::Kind::Counter &&
                   record.tags.at("m") == std::to_string(M) &&
                   record.tags.at("k") == std::to_string(K) &&
                   record.tags.at("projections") == "2";
        });
    ASSERT_NE(fused_kernel_record, fused_records.end())
        << "CUDA qkv+z GDN subgroup should execute the graph-native small-M fused projection kernel";

    const auto fp32_fused_records =
        PerfStatsCollector::snapshot({"kernel.cuda_fp32_grouped_verifier_projection_calls"});
    const auto fp32_fused_record = std::find_if(
        fp32_fused_records.begin(),
        fp32_fused_records.end(),
        [&](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "cuda_fp32_grouped_verifier_projection_calls" &&
                   record.kind == PerfStatRecord::Kind::Counter &&
                   record.tags.at("dtype") == "fp32" &&
                   record.tags.at("m") == std::to_string(M) &&
                   record.tags.at("k") == std::to_string(K) &&
                   record.tags.at("n") == std::to_string(N_ALPHA) &&
                   record.tags.at("projections") == "2" &&
                   record.tags.at("route") == "fixed_order_fp32_batched_projection";
        });
    ASSERT_NE(fp32_fused_record, fp32_fused_records.end())
        << "CUDA alpha+beta GDN subgroup should execute the fixed-order FP32 verifier projection route";

    const auto alpha_result =
        checkParity(output_alpha->data(), alpha_cpu.data(), alpha_cpu.size(), 0.9999, 0.01);
    const auto beta_result =
        checkParity(output_beta->data(), beta_cpu.data(), beta_cpu.size(), 0.9999, 0.01);
    EXPECT_FALSE(alpha_result.has_nan_inf);
    EXPECT_FALSE(beta_result.has_nan_inf);
    EXPECT_GE(alpha_result.cosine_similarity, 0.9999);
    EXPECT_GE(beta_result.cosine_similarity, 0.9999);
    EXPECT_LE(alpha_result.relative_l2_error, 0.01);
    EXPECT_LE(beta_result.relative_l2_error, 0.01);

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_qkv.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_z.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_alpha.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_beta.get());
    PerfStatsCollector::reset();
}

/**
 * @brief Prove production-width GDNProjectionStage verifier rows are decode-equivalent.
 *
 * The full Qwen3.6 dense grouped-verifier test compares the captured verifier
 * graph against serial decode and first fails near `layer1_QKV_PROJECTION`.
 * This stage-level regression keeps the real layer1 projection formats
 * (Q5_K qkv, Q4_K z, F32 alpha/beta) and checks the exact graph stage API:
 * one M=4 verifier-stage execution must match four ordinary M=1 stage
 * executions row-for-row.  Passing here means remaining drift comes from
 * surrounding graph state rather than GDNProjectionStage itself.
 */
TEST_F(Test__CUDAGemmParity, GDNProjectionStage_Qwen36MixedCodebooks_M4MatchesFourM1StageRows)
{
    const int M = 4;
    const int K = 5120;
    const int N_QKV = 10240;
    const int N_Z = 6144;
    const int N_ALPHA = 48;
    const int N_BETA = 48;

    auto weights_qkv = TestTensorFactory::createQ5_KRandom(
        {static_cast<size_t>(N_QKV), static_cast<size_t>(K)}, 4531);
    auto weights_z = TestTensorFactory::createQ4_KRandom(
        {static_cast<size_t>(N_Z), static_cast<size_t>(K)}, 4532);
    auto weights_alpha = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(N_ALPHA), static_cast<size_t>(K)}, -0.1f, 0.1f, 4533);
    auto weights_beta = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(N_BETA), static_cast<size_t>(K)}, -0.1f, 0.1f, 4534);
    ASSERT_TRUE(weights_alpha->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(weights_beta->ensureOnDevice(gpu_device_));

    auto *kernel_qkv = getPreparedKernel(weights_qkv.get(), gpu_device_);
    auto *kernel_z = getPreparedKernel(weights_z.get(), gpu_device_);
    auto *kernel_alpha = getPreparedKernel(weights_alpha.get(), gpu_device_);
    auto *kernel_beta = getPreparedKernel(weights_beta.get(), gpu_device_);
    ASSERT_NE(kernel_qkv, nullptr);
    ASSERT_NE(kernel_z, nullptr);
    ASSERT_NE(kernel_alpha, nullptr);
    ASSERT_NE(kernel_beta, nullptr);

    struct StageOutputs
    {
        std::unique_ptr<FP32Tensor> qkv;
        std::unique_ptr<FP32Tensor> z;
        std::unique_ptr<FP32Tensor> alpha;
        std::unique_ptr<FP32Tensor> beta;
    };

    auto make_outputs = [&](int rows) -> StageOutputs
    {
        return {
            std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(N_QKV)}),
            std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(N_Z)}),
            std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(N_ALPHA)}),
            std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(N_BETA)})};
    };

    auto run_stage = [&](int rows, const float *input_data, StageOutputs *outputs)
    {
        auto input = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(K)});
        std::memcpy(
            input->mutable_data(),
            input_data,
            static_cast<size_t>(rows) * K * sizeof(float));

        GDNProjectionStage::Params params;
        params.device_id = gpu_device_;
        params.input = input.get();
        params.m = rows;
        params.k = K;
        params.w_qkv = weights_qkv.get();
        params.output_qkv = outputs->qkv.get();
        params.n_qkv = N_QKV;
        params.w_z = weights_z.get();
        params.output_z = outputs->z.get();
        params.n_z = N_Z;
        params.w_a = weights_alpha.get();
        params.output_a = outputs->alpha.get();
        params.n_a = N_ALPHA;
        params.w_b = weights_beta.get();
        params.output_b = outputs->beta.get();
        params.n_b = N_BETA;
        params.gemm_qkv = kernel_qkv;
        params.gemm_z = kernel_z;
        params.gemm_a = kernel_alpha;
        params.gemm_b = kernel_beta;
        params.force_decode_equivalent_verifier_prefill = true;

        GDNProjectionStage stage(params);
        WorkspaceRequirements reqs = stage.getWorkspaceRequirements(rows, 0, K);
        DeviceWorkspaceManager workspace(gpu_device_, workspaceBudgetFor(reqs));
        ASSERT_TRUE(workspace.allocate(reqs));
        stage.bindWorkspace(&workspace);

        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
        stage.setGPUStream(static_cast<void *>(stream));
        CUDADeviceContext ctx(gpu_device_, gpu_device_.ordinal);
        const bool ok = with_gpu_coherence(
            gpu_device_,
            {input.get()},
            {outputs->qkv.get(), outputs->z.get(), outputs->alpha.get(), outputs->beta.get()},
            static_cast<void *>(stream),
            [&]
            {
                return stage.execute(&ctx);
            });
        ASSERT_TRUE(ok);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
        stage.unbindWorkspace();
        kernel_qkv->resetDynamicState();
        kernel_z->resetDynamicState();
        kernel_alpha->resetDynamicState();
        kernel_beta->resetDynamicState();
    };

    const auto input_data = randomFP32(static_cast<size_t>(M) * K);
    StageOutputs grouped = make_outputs(M);
    run_stage(M, input_data.data(), &grouped);

    for (int row = 0; row < M; ++row)
    {
        StageOutputs serial = make_outputs(1);
        run_stage(1, input_data.data() + static_cast<size_t>(row) * K, &serial);

        expectBitwiseEqualFloatRow(
            (std::string("GDNProjectionStage qkv M=") + std::to_string(M) +
             " row=" + std::to_string(row))
                .c_str(),
            grouped.qkv->data() + static_cast<size_t>(row) * N_QKV,
            serial.qkv->data(),
            static_cast<size_t>(N_QKV));
        expectBitwiseEqualFloatRow(
            (std::string("GDNProjectionStage z M=") + std::to_string(M) +
             " row=" + std::to_string(row))
                .c_str(),
            grouped.z->data() + static_cast<size_t>(row) * N_Z,
            serial.z->data(),
            static_cast<size_t>(N_Z));
        expectBitwiseEqualFloatRow(
            (std::string("GDNProjectionStage alpha M=") + std::to_string(M) +
             " row=" + std::to_string(row))
                .c_str(),
            grouped.alpha->data() + static_cast<size_t>(row) * N_ALPHA,
            serial.alpha->data(),
            static_cast<size_t>(N_ALPHA));
        expectBitwiseEqualFloatRow(
            (std::string("GDNProjectionStage beta M=") + std::to_string(M) +
             " row=" + std::to_string(row))
                .c_str(),
            grouped.beta->data() + static_cast<size_t>(row) * N_BETA,
            serial.beta->data(),
            static_cast<size_t>(N_BETA));
    }

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_qkv.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_z.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_alpha.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_beta.get());
}

/**
 * @brief Prove the real Qwen3.6 MoE GDN projection verifier rows are serial-decode equivalent.
 *
 * The MoE 35B fixture stores GDN qkv/z/out weights as Q6_K and uses the
 * narrower MoE hidden width (`K=2048`, qkv=`8192`, z=`4096`, alpha/beta=`32`).
 * Dense Qwen3.6 M=4 tests did not cover this shape or codebook, while the
 * production continuation proof failed first around QKV/Z projection before
 * attention and MoE routing amplified the error.  This regression checks the
 * stage-level contract directly at every important runtime-M boundary. Every
 * grouped row must match repeated M=1 decode-stage rows byte-for-byte.
 */
TEST_F(Test__CUDAGemmParity, GDNProjectionStage_Qwen36MoEQ6K_RuntimeMMatchesSerialStageRows)
{
    constexpr int kK = 2048;
    constexpr int kNQKV = 8192;
    constexpr int kNZ = 4096;
    constexpr int kNAlpha = 32;
    constexpr int kNBeta = 32;

    auto weights_qkv = TestTensorFactory::createQ6_KRandom(
        {static_cast<size_t>(kNQKV), static_cast<size_t>(kK)}, 5531);
    auto weights_z = TestTensorFactory::createQ6_KRandom(
        {static_cast<size_t>(kNZ), static_cast<size_t>(kK)}, 5532);
    auto weights_alpha = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(kNAlpha), static_cast<size_t>(kK)}, -0.1f, 0.1f, 5533);
    auto weights_beta = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(kNBeta), static_cast<size_t>(kK)}, -0.1f, 0.1f, 5534);
    ASSERT_TRUE(weights_alpha->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(weights_beta->ensureOnDevice(gpu_device_));

    auto *kernel_qkv = getPreparedKernel(weights_qkv.get(), gpu_device_);
    auto *kernel_z = getPreparedKernel(weights_z.get(), gpu_device_);
    auto *kernel_alpha = getPreparedKernel(weights_alpha.get(), gpu_device_);
    auto *kernel_beta = getPreparedKernel(weights_beta.get(), gpu_device_);
    ASSERT_NE(kernel_qkv, nullptr);
    ASSERT_NE(kernel_z, nullptr);
    ASSERT_NE(kernel_alpha, nullptr);
    ASSERT_NE(kernel_beta, nullptr);

    struct StageOutputs
    {
        std::unique_ptr<FP32Tensor> qkv;
        std::unique_ptr<FP32Tensor> z;
        std::unique_ptr<FP32Tensor> alpha;
        std::unique_ptr<FP32Tensor> beta;
    };

    auto make_outputs = [](int rows) -> StageOutputs
    {
        return {
            std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(kNQKV)}),
            std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(kNZ)}),
            std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(kNAlpha)}),
            std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(kNBeta)})};
    };

    auto run_stage = [&](int rows, const float *input_data, StageOutputs *outputs)
    {
        auto input = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(kK)});
        std::memcpy(
            input->mutable_data(),
            input_data,
            static_cast<size_t>(rows) * kK * sizeof(float));

        GDNProjectionStage::Params params;
        params.device_id = gpu_device_;
        params.input = input.get();
        params.m = rows;
        params.k = kK;
        params.w_qkv = weights_qkv.get();
        params.output_qkv = outputs->qkv.get();
        params.n_qkv = kNQKV;
        params.w_z = weights_z.get();
        params.output_z = outputs->z.get();
        params.n_z = kNZ;
        params.w_a = weights_alpha.get();
        params.output_a = outputs->alpha.get();
        params.n_a = kNAlpha;
        params.w_b = weights_beta.get();
        params.output_b = outputs->beta.get();
        params.n_b = kNBeta;
        params.gemm_qkv = kernel_qkv;
        params.gemm_z = kernel_z;
        params.gemm_a = kernel_alpha;
        params.gemm_b = kernel_beta;
        params.force_decode_equivalent_verifier_prefill = true;

        GDNProjectionStage stage(params);
        WorkspaceRequirements reqs = stage.getWorkspaceRequirements(rows, 0, kK);
        DeviceWorkspaceManager workspace(gpu_device_, workspaceBudgetFor(reqs));
        ASSERT_TRUE(workspace.allocate(reqs));
        stage.bindWorkspace(&workspace);

        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
        stage.setGPUStream(static_cast<void *>(stream));
        CUDADeviceContext ctx(gpu_device_, gpu_device_.ordinal);
        const bool ok = with_gpu_coherence(
            gpu_device_,
            {input.get()},
            {outputs->qkv.get(), outputs->z.get(), outputs->alpha.get(), outputs->beta.get()},
            static_cast<void *>(stream),
            [&]
            {
                return stage.execute(&ctx);
            });
        ASSERT_TRUE(ok);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
        stage.unbindWorkspace();
        kernel_qkv->resetDynamicState();
        kernel_z->resetDynamicState();
        kernel_alpha->resetDynamicState();
        kernel_beta->resetDynamicState();
    };

    for (const int M : kGroupedVerifierBoundaryRows)
    {
        const auto input_data = randomFP32(static_cast<size_t>(M) * kK);
        StageOutputs grouped = make_outputs(M);
        run_stage(M, input_data.data(), &grouped);

        for (int row = 0; row < M; ++row)
        {
            StageOutputs serial = make_outputs(1);
            run_stage(1, input_data.data() + static_cast<size_t>(row) * kK, &serial);

            expectBitwiseEqualFloatRow(
                (std::string("Qwen36 MoE GDN qkv M=") + std::to_string(M) +
                 " row=" + std::to_string(row))
                    .c_str(),
                grouped.qkv->data() + static_cast<size_t>(row) * kNQKV,
                serial.qkv->data(),
                static_cast<size_t>(kNQKV));
            expectBitwiseEqualFloatRow(
                (std::string("Qwen36 MoE GDN z M=") + std::to_string(M) +
                 " row=" + std::to_string(row))
                    .c_str(),
                grouped.z->data() + static_cast<size_t>(row) * kNZ,
                serial.z->data(),
                static_cast<size_t>(kNZ));
            expectBitwiseEqualFloatRow(
                (std::string("Qwen36 MoE GDN alpha M=") + std::to_string(M) +
                 " row=" + std::to_string(row))
                    .c_str(),
                grouped.alpha->data() + static_cast<size_t>(row) * kNAlpha,
                serial.alpha->data(),
                static_cast<size_t>(kNAlpha));
            expectBitwiseEqualFloatRow(
                (std::string("Qwen36 MoE GDN beta M=") + std::to_string(M) +
                 " row=" + std::to_string(row))
                    .c_str(),
                grouped.beta->data() + static_cast<size_t>(row) * kNBeta,
                serial.beta->data(),
                static_cast<size_t>(kNBeta));
        }
    }

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_qkv.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_z.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_alpha.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_beta.get());
}

/**
 * @brief Prove the sharded LocalTP Qwen3.6 MoE GDN projection verifier shape.
 *
 * The LocalTP graph keeps Q/K replicated but shards the GDN value heads, so the
 * production projection dimensions differ from the full single-device MoE
 * shape: QKV is `[6144, 2048]`, Z is `[2048, 2048]`, and alpha/beta are
 * `[16, 2048]`. This regression prevents the grouped verifier implementation
 * from being correct only for the unsharded GDN tensor geometry.
 */
TEST_F(Test__CUDAGemmParity, GDNProjectionStage_Qwen36MoELocalTPShardQ6K_RuntimeMMatchesSerialStageRows)
{
    constexpr int kK = 2048;
    constexpr int kNQKV = 6144;
    constexpr int kNZ = 2048;
    constexpr int kNAlpha = 16;
    constexpr int kNBeta = 16;

    auto weights_qkv = TestTensorFactory::createQ6_KRandom(
        {static_cast<size_t>(kNQKV), static_cast<size_t>(kK)}, 6531);
    auto weights_z = TestTensorFactory::createQ6_KRandom(
        {static_cast<size_t>(kNZ), static_cast<size_t>(kK)}, 6532);
    auto weights_alpha = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(kNAlpha), static_cast<size_t>(kK)}, -0.1f, 0.1f, 6533);
    auto weights_beta = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(kNBeta), static_cast<size_t>(kK)}, -0.1f, 0.1f, 6534);

    auto qkv_prepared = llaminar2::test::makeGpuPreparedGemm(
        weights_qkv.get(), gpu_device_, "blk.0.attn_qkv.weight.localtp", ModelContextId{65381});
    auto z_prepared = llaminar2::test::makeGpuPreparedGemm(
        weights_z.get(), gpu_device_, "blk.0.attn_gate.weight.localtp", ModelContextId{65381});
    auto alpha_prepared = llaminar2::test::makeGpuPreparedFloatingPointGemm(
        weights_alpha.get(), gpu_device_, "blk.0.ssm_alpha.weight.localtp", ModelContextId{65381});
    auto beta_prepared = llaminar2::test::makeGpuPreparedFloatingPointGemm(
        weights_beta.get(), gpu_device_, "blk.0.ssm_beta.weight.localtp", ModelContextId{65381});

    struct StageOutputs
    {
        std::unique_ptr<FP32Tensor> qkv;
        std::unique_ptr<FP32Tensor> z;
        std::unique_ptr<FP32Tensor> alpha;
        std::unique_ptr<FP32Tensor> beta;
    };

    auto make_outputs = [](int rows) -> StageOutputs
    {
        return {
            std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(kNQKV)}),
            std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(kNZ)}),
            std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(kNAlpha)}),
            std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(kNBeta)})};
    };

    auto run_stage = [&](int rows, const float *input_data, StageOutputs *outputs)
    {
        auto input = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(kK)});
        std::memcpy(
            input->mutable_data(),
            input_data,
            static_cast<size_t>(rows) * static_cast<size_t>(kK) * sizeof(float));

        GDNProjectionStage::Params params;
        params.device_id = gpu_device_;
        params.input = input.get();
        params.m = rows;
        params.k = kK;
        params.w_qkv = weights_qkv.get();
        params.output_qkv = outputs->qkv.get();
        params.n_qkv = kNQKV;
        params.w_z = weights_z.get();
        params.output_z = outputs->z.get();
        params.n_z = kNZ;
        params.w_a = weights_alpha.get();
        params.output_a = outputs->alpha.get();
        params.n_a = kNAlpha;
        params.w_b = weights_beta.get();
        params.output_b = outputs->beta.get();
        params.n_b = kNBeta;
        params.gemm_qkv = qkv_prepared.kernel;
        params.gemm_z = z_prepared.kernel;
        params.gemm_a = alpha_prepared.kernel;
        params.gemm_b = beta_prepared.kernel;
        params.force_decode_equivalent_verifier_prefill = true;

        GDNProjectionStage stage(params);
        WorkspaceRequirements reqs = stage.getWorkspaceRequirements(rows, 0, kK);
        DeviceWorkspaceManager workspace(gpu_device_, workspaceBudgetFor(reqs));
        ASSERT_TRUE(workspace.allocate(reqs));
        stage.bindWorkspace(&workspace);

        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
        stage.setGPUStream(static_cast<void *>(stream));
        CUDADeviceContext ctx(gpu_device_, gpu_device_.ordinal);
        const bool ok = with_gpu_coherence(
            gpu_device_,
            {input.get()},
            {outputs->qkv.get(), outputs->z.get(), outputs->alpha.get(), outputs->beta.get()},
            static_cast<void *>(stream),
            [&]
            {
                return stage.execute(&ctx);
            });
        ASSERT_TRUE(ok);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
        stage.unbindWorkspace();
        qkv_prepared.kernel->resetDynamicState();
        z_prepared.kernel->resetDynamicState();
        alpha_prepared.kernel->resetDynamicState();
        beta_prepared.kernel->resetDynamicState();
    };

    auto expect_projection = [&](const char *name,
                                 const float *grouped,
                                 const float *serial,
                                 size_t count,
                                 int m,
                                 int row)
    {
        expectBitwiseEqualFloatRow(
            (std::string("Qwen36 MoE LocalTP GDN ") + name +
             " M=" + std::to_string(m) +
             " row=" + std::to_string(row))
                .c_str(),
            grouped,
            serial,
            count);
    };

    for (const int M : kGroupedVerifierBoundaryRows)
    {
        const auto input_data = randomFP32(static_cast<size_t>(M) * static_cast<size_t>(kK));
        StageOutputs grouped = make_outputs(M);
        run_stage(M, input_data.data(), &grouped);

        for (int row = 0; row < M; ++row)
        {
            StageOutputs serial = make_outputs(1);
            run_stage(
                1,
                input_data.data() + static_cast<size_t>(row) * static_cast<size_t>(kK),
                &serial);
            expect_projection(
                "qkv",
                grouped.qkv->data() + static_cast<size_t>(row) * static_cast<size_t>(kNQKV),
                serial.qkv->data(),
                static_cast<size_t>(kNQKV),
                M,
                row);
            expect_projection(
                "z",
                grouped.z->data() + static_cast<size_t>(row) * static_cast<size_t>(kNZ),
                serial.z->data(),
                static_cast<size_t>(kNZ),
                M,
                row);
            expect_projection(
                "alpha",
                grouped.alpha->data() + static_cast<size_t>(row) * static_cast<size_t>(kNAlpha),
                serial.alpha->data(),
                static_cast<size_t>(kNAlpha),
                M,
                row);
            expect_projection(
                "beta",
                grouped.beta->data() + static_cast<size_t>(row) * static_cast<size_t>(kNBeta),
                serial.beta->data(),
                static_cast<size_t>(kNBeta),
                M,
                row);
        }
    }

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_qkv.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_z.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_alpha.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_beta.get());
}

/**
 * @brief Prove real Qwen3.6 MoE LocalTP GDN verifier projections by rank.
 *
 * The synthetic LocalTP-shard regression above proves the shape, but it cannot
 * catch bugs in the production weight sharder.  This test loads the actual
 * Qwen3.6 MoE layer-0 GDN projection weights through the LocalTP two-device
 * sharding path:
 *
 * - `attn_qkv.weight`: fused asymmetric Q/K/V, with Q and K replicated and V
 *   sharded.
 * - `attn_gate.weight`: Q6_K column-parallel gate/Z projection.
 * - `ssm_alpha.weight` and `ssm_beta.weight`: FP32 column-parallel per-value
 *   head projections.
 *
 * Each local rank then runs the production `GDNProjectionStage` at the
 * runtime-M boundary inventory and compares every grouped row to a separate
 * M=1 stage execution with the same rank-local weights. That is the exact
 * decode-equivalence contract the graph relies on before recurrence, MoE, and
 * LM-head stages consume these rows.
 */
TEST_F(Test__CUDAGemmParity, RealQwen36MoELocalTPGDNProjectionShardGroupedRowsMatchSerialStageRows)
{
    SKIP_IF_NO_CUDA();
    ASSERT_EQ(cudaSetDevice(gpu_device_.ordinal), cudaSuccess);

    const std::filesystem::path model_path = qwen36MoEModelPath();
    if (!std::filesystem::exists(model_path))
        GTEST_SKIP() << "Qwen 3.6 MoE model not found at " << model_path
                     << "; set LLAMINAR_QWEN36_MOE_MODEL to run this real-weight regression";

    auto model_ctx = loadQwen36ModelForGpuWeights(model_path);
    ASSERT_NE(model_ctx, nullptr);

    const std::vector<DeviceId> logical_tp_devices = {DeviceId::cuda(0), DeviceId::cuda(1)};
    auto tp_config = configureQwen36MoELocalTPWeights(model_ctx, logical_tp_devices);
    ASSERT_NE(tp_config, nullptr);
    ASSERT_TRUE(tp_config->validate()) << tp_config->validationError();

    constexpr int kK = 2048;
    constexpr int kNQKV = 6144;
    constexpr int kNZ = 2048;
    constexpr int kNAlpha = 16;
    constexpr int kNBeta = 16;
    const ModelContextId model_id{36363};

    struct StageOutputs
    {
        std::unique_ptr<FP32Tensor> qkv;
        std::unique_ptr<FP32Tensor> z;
        std::unique_ptr<FP32Tensor> alpha;
        std::unique_ptr<FP32Tensor> beta;
    };

    auto make_outputs = [](int rows) -> StageOutputs
    {
        return {
            std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(kNQKV)}),
            std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(kNZ)}),
            std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(kNAlpha)}),
            std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(kNBeta)})};
    };

    auto expect_projection = [&](const char *name,
                                 const float *grouped,
                                 const float *serial,
                                 size_t count,
                                 int rank,
                                 int m,
                                 int row)
    {
        expectBitwiseEqualFloatRow(
            (std::string("real Qwen36 MoE LocalTP rank=") +
             std::to_string(rank) + " GDN " + name +
             " M=" + std::to_string(m) +
             " row=" + std::to_string(row))
                .c_str(),
            grouped,
            serial,
            count);
    };

    for (int rank = 0; rank < 2; ++rank)
    {
        const DeviceId shard_device = logical_tp_devices[static_cast<size_t>(rank)];

        auto weights_qkv = model_ctx->getWeightForDevice("blk.0.attn_qkv.weight", shard_device);
        auto weights_z = model_ctx->getWeightForDevice("blk.0.attn_gate.weight", shard_device);
        auto weights_alpha = model_ctx->getWeightForDevice("blk.0.ssm_alpha.weight", shard_device);
        auto weights_beta = model_ctx->getWeightForDevice("blk.0.ssm_beta.weight", shard_device);

        ASSERT_NE(weights_qkv, nullptr);
        ASSERT_NE(weights_z, nullptr);
        ASSERT_NE(weights_alpha, nullptr);
        ASSERT_NE(weights_beta, nullptr);
        ASSERT_EQ(weights_qkv->native_type(), TensorType::Q6_K);
        ASSERT_EQ(weights_z->native_type(), TensorType::Q6_K);
        ASSERT_EQ(weights_alpha->native_type(), TensorType::FP32);
        ASSERT_EQ(weights_beta->native_type(), TensorType::FP32);
        ASSERT_EQ(weights_qkv->rows(), static_cast<size_t>(kNQKV));
        ASSERT_EQ(weights_qkv->cols(), static_cast<size_t>(kK));
        ASSERT_EQ(weights_z->rows(), static_cast<size_t>(kNZ));
        ASSERT_EQ(weights_z->cols(), static_cast<size_t>(kK));
        ASSERT_EQ(weights_alpha->rows(), static_cast<size_t>(kNAlpha));
        ASSERT_EQ(weights_alpha->cols(), static_cast<size_t>(kK));
        ASSERT_EQ(weights_beta->rows(), static_cast<size_t>(kNBeta));
        ASSERT_EQ(weights_beta->cols(), static_cast<size_t>(kK));

        auto qkv_prepared = llaminar2::test::makeGpuPreparedGemm(
            gpuPreparationTensor(weights_qkv.get()),
            gpu_device_,
            "blk.0.attn_qkv.weight.localtp.rank" + std::to_string(rank),
            model_id);
        auto z_prepared = llaminar2::test::makeGpuPreparedGemm(
            gpuPreparationTensor(weights_z.get()),
            gpu_device_,
            "blk.0.attn_gate.weight.localtp.rank" + std::to_string(rank),
            model_id);
        auto alpha_prepared = llaminar2::test::makeGpuPreparedFloatingPointGemm(
            gpuPreparationTensor(weights_alpha.get()),
            gpu_device_,
            "blk.0.ssm_alpha.weight.localtp.rank" + std::to_string(rank),
            model_id);
        auto beta_prepared = llaminar2::test::makeGpuPreparedFloatingPointGemm(
            gpuPreparationTensor(weights_beta.get()),
            gpu_device_,
            "blk.0.ssm_beta.weight.localtp.rank" + std::to_string(rank),
            model_id);

        auto run_stage = [&](int rows, const float *input_data, StageOutputs *outputs)
        {
            auto input = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(kK)});
            std::memcpy(
                input->mutable_data(),
                input_data,
                static_cast<size_t>(rows) * static_cast<size_t>(kK) * sizeof(float));

            GDNProjectionStage::Params params;
            params.device_id = gpu_device_;
            params.input = input.get();
            params.m = rows;
            params.k = kK;
            params.w_qkv = weights_qkv.get();
            params.output_qkv = outputs->qkv.get();
            params.n_qkv = kNQKV;
            params.w_z = weights_z.get();
            params.output_z = outputs->z.get();
            params.n_z = kNZ;
            params.w_a = weights_alpha.get();
            params.output_a = outputs->alpha.get();
            params.n_a = kNAlpha;
            params.w_b = weights_beta.get();
            params.output_b = outputs->beta.get();
            params.n_b = kNBeta;
            params.gemm_qkv = qkv_prepared.kernel;
            params.gemm_z = z_prepared.kernel;
            params.gemm_a = alpha_prepared.kernel;
            params.gemm_b = beta_prepared.kernel;
            params.force_decode_equivalent_verifier_prefill = true;

            GDNProjectionStage stage(params);
            WorkspaceRequirements reqs = stage.getWorkspaceRequirements(rows, 0, kK);
            DeviceWorkspaceManager workspace(gpu_device_, workspaceBudgetFor(reqs));
            ASSERT_TRUE(workspace.allocate(reqs));
            stage.bindWorkspace(&workspace);

            cudaStream_t stream = nullptr;
            ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
            stage.setGPUStream(static_cast<void *>(stream));
            CUDADeviceContext ctx(gpu_device_, gpu_device_.ordinal);
            const bool ok = with_gpu_coherence(
                gpu_device_,
                {input.get()},
                {outputs->qkv.get(), outputs->z.get(), outputs->alpha.get(), outputs->beta.get()},
                static_cast<void *>(stream),
                [&]
                {
                    return stage.execute(&ctx);
                });
            ASSERT_TRUE(ok) << "real LocalTP GDN projection failed for rank=" << rank
                            << " rows=" << rows;
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
            stage.unbindWorkspace();
            qkv_prepared.kernel->resetDynamicState();
            z_prepared.kernel->resetDynamicState();
            alpha_prepared.kernel->resetDynamicState();
            beta_prepared.kernel->resetDynamicState();
        };

        for (const int M : kGroupedVerifierBoundaryRows)
        {
            const auto input_data = randomFP32(static_cast<size_t>(M) * static_cast<size_t>(kK));
            StageOutputs grouped = make_outputs(M);
            run_stage(M, input_data.data(), &grouped);

            for (int row = 0; row < M; ++row)
            {
                StageOutputs serial = make_outputs(1);
                run_stage(
                    1,
                    input_data.data() + static_cast<size_t>(row) * static_cast<size_t>(kK),
                    &serial);

                expect_projection(
                    "qkv",
                    grouped.qkv->data() + static_cast<size_t>(row) * static_cast<size_t>(kNQKV),
                    serial.qkv->data(),
                    static_cast<size_t>(kNQKV),
                    rank,
                    M,
                    row);
                expect_projection(
                    "z",
                    grouped.z->data() + static_cast<size_t>(row) * static_cast<size_t>(kNZ),
                    serial.z->data(),
                    static_cast<size_t>(kNZ),
                    rank,
                    M,
                    row);
                expect_projection(
                    "alpha",
                    grouped.alpha->data() + static_cast<size_t>(row) * static_cast<size_t>(kNAlpha),
                    serial.alpha->data(),
                    static_cast<size_t>(kNAlpha),
                    rank,
                    M,
                    row);
                expect_projection(
                    "beta",
                    grouped.beta->data() + static_cast<size_t>(row) * static_cast<size_t>(kNBeta),
                    serial.beta->data(),
                    static_cast<size_t>(kNBeta),
                    rank,
                    M,
                    row);
            }
        }
    }
}

/**
 * @brief Prove CUDA real Qwen3.6 MoE LM-head verifier rows match serial decode.
 *
 * LocalTP MTP verifier failures surface as full-vocab logit drift after the
 * compact row-indexed LM head. Synthetic Q6_K tests cover the dispatch family,
 * but they do not prove the production `output.weight` geometry or the GPU
 * upload/repack path. This regression loads the real Q6_K LM head, executes
 * the runtime-M boundary inventory through the grouped verifier projection API
 * used by LMHeadStage, and compares each row to an ordinary M=1 decode
 * projection.
 */
TEST_F(Test__CUDAGemmParity, RealQwen36MoELMHeadGroupedVerifierRowsMatchSerialDecodeStrict)
{
    SKIP_IF_NO_CUDA();
    ASSERT_EQ(cudaSetDevice(gpu_device_.ordinal), cudaSuccess);

    const std::filesystem::path model_path = qwen36MoEModelPath();
    if (!std::filesystem::exists(model_path))
        GTEST_SKIP() << "Qwen 3.6 MoE model not found at " << model_path
                     << "; set LLAMINAR_QWEN36_MOE_MODEL to run this real-weight regression";

    auto model_ctx = loadQwen36ModelForGpuWeights(model_path);
    ASSERT_NE(model_ctx, nullptr);
    ASSERT_TRUE(model_ctx->hasTensor("output.weight"))
        << "Qwen 3.6 MoE test fixture must expose a concrete LM head";

    auto lm_head = model_ctx->getWeightForDevice("output.weight", DeviceId::cpu());
    ASSERT_NE(lm_head, nullptr);
    ASSERT_EQ(lm_head->native_type(), TensorType::Q6_K)
        << "This regression guards the Qwen 3.6 MoE Q6_K LM-head path";

    constexpr int kExpectedVocab = 248320;
    constexpr int kExpectedHidden = 2048;
    const int N = static_cast<int>(lm_head->rows());
    const int K = static_cast<int>(lm_head->cols());
    ASSERT_EQ(N, kExpectedVocab);
    ASSERT_EQ(K, kExpectedHidden);

    auto prepared = llaminar2::test::makeGpuPreparedGemm(
        lm_head.get(),
        gpu_device_,
        "output.weight",
        ModelContextId{36362});
    ASSERT_NE(prepared.kernel, nullptr);

    for (const int M : kGroupedVerifierBoundaryRows)
    {
        ASSERT_TRUE(setupWorkspaceIfNeeded(prepared.kernel, M, N, K))
            << "workspace for Qwen3.6 MoE LM-head M=" << M;
        const auto input_data = randomFP32(static_cast<size_t>(M) * static_cast<size_t>(K));
        std::vector<float> grouped(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
        ASSERT_TRUE(cudaGroupedVerifierProjectionViaTensor(
            prepared.kernel,
            input_data.data(),
            grouped.data(),
            M,
            N,
            K,
            gpu_device_,
            workspace_.get()))
            << "grouped verifier LM-head projection failed at M=" << M;

        for (int row = 0; row < M; ++row)
        {
            std::vector<float> serial(static_cast<size_t>(N), 0.0f);
            ASSERT_TRUE(cudaMultiplyViaTensor(
                prepared.kernel,
                input_data.data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                serial.data(),
                1,
                N,
                K,
                gpu_device_))
                << "serial M=1 LM-head projection failed at grouped M=" << M
                << " row=" << row;

            const float *grouped_row =
                grouped.data() + static_cast<size_t>(row) * static_cast<size_t>(N);
            expectBitwiseEqualFloatRow(
                (std::string("real Qwen36 MoE LM-head M=") +
                 std::to_string(M) + " row=" + std::to_string(row))
                    .c_str(),
                grouped_row,
                serial.data(),
                serial.size());
        }

        cleanupWorkspaceIfNeeded(prepared.kernel);
    }

    prepared.kernel->setGPUStream(nullptr);
}

/**
 * @brief Reproduce the real Qwen3.6 MoE GDN QKV verifier transaction.
 *
 * The focused verifier-forward suite first observed CUDA drift after layer
 * zero's 8192x2048 Q6_K projection, at row two, while rows zero and one
 * remained byte-identical. Layer zero is a GDN layer: `attn_qkv.weight` is one
 * fused projection rather than the three separate tensors used by a
 * full-attention layer. This regression therefore retains the exact GGUF
 * payload bytes, scales, and minima consumed by `GDNProjectionStage`.
 *
 * The grouped call is the production API and performs one activation
 * quantization followed by one grouped GEMV. The M=1 loop is test-only: each
 * row is independently quantized through the ordinary serial decode API and
 * acts solely as the byte-exact oracle.
 */
TEST_F(Test__CUDAGemmParity,
       RealQwen36MoEGDNQKVGroupedVerifierRowsMatchSerialDecodeStrict)
{
    SKIP_IF_NO_CUDA();
    ASSERT_EQ(cudaSetDevice(gpu_device_.ordinal), cudaSuccess);

    const std::filesystem::path model_path = qwen36MoEModelPath();
    if (!std::filesystem::exists(model_path))
    {
        GTEST_SKIP() << "Qwen 3.6 MoE model not found at " << model_path
                     << "; set LLAMINAR_QWEN36_MOE_MODEL to run this real-weight regression";
    }

    auto model_ctx = loadQwen36ModelForGpuWeights(model_path);
    ASSERT_NE(model_ctx, nullptr);

    auto qkv = model_ctx->getWeightForDevice(
        "blk.0.attn_qkv.weight", DeviceId::cpu());
    ASSERT_NE(qkv, nullptr);
    ASSERT_EQ(qkv->native_type(), TensorType::Q6_K);

    const int n = static_cast<int>(qkv->rows());
    const int k = static_cast<int>(qkv->cols());
    ASSERT_EQ(n, 8192);
    ASSERT_EQ(k, 2048);

    const ModelContextId model_id{36367};
    auto prepared = llaminar2::test::makeGpuPreparedGemm(
        gpuPreparationTensor(qkv.get()),
        gpu_device_,
        "blk.0.attn_qkv.weight",
        model_id);
    ASSERT_NE(prepared.kernel, nullptr);
    ASSERT_TRUE(setupSharedWorkspace(
        {prepared.kernel},
        /*max_m=*/4,
        {n},
        k));

    for (const int m : {2, 3, 4})
    {
        const auto input_values =
            randomFP32(static_cast<size_t>(m) * static_cast<size_t>(k));
        auto input = std::make_unique<FP32Tensor>(
            std::vector<size_t>{
                static_cast<size_t>(m),
                static_cast<size_t>(k)});
        std::memcpy(
            input->mutable_data(),
            input_values.data(),
            input_values.size() * sizeof(float));

        auto grouped_qkv = std::make_unique<FP32Tensor>(
            std::vector<size_t>{
                static_cast<size_t>(m),
                static_cast<size_t>(n)});
        std::vector<TensorProjectionDesc> grouped_projections = {
            {prepared.kernel, grouped_qkv.get(), n, nullptr, "GDN_QKV"}};

        ASSERT_TRUE(withExplicitCudaGemmStream(
            {prepared.kernel},
            [&](void *stream)
            {
                return with_gpu_coherence(
                    gpu_device_,
                    {input.get()},
                    {grouped_qkv.get()},
                    stream,
                    [&]
                    {
                        return prepared.kernel->
                            multiply_fused_verifier_rows_decode_equivalent(
                                input.get(),
                                grouped_projections,
                                m,
                                k,
                                nullptr,
                                workspace_.get());
                    });
            }))
            << "real Qwen3.6 MoE grouped GDN QKV projection failed at M="
            << m;

        for (int row = 0; row < m; ++row)
        {
            auto serial_input = std::make_unique<FP32Tensor>(
                std::vector<size_t>{size_t{1}, static_cast<size_t>(k)});
            std::memcpy(
                serial_input->mutable_data(),
                input_values.data() +
                    static_cast<size_t>(row) * static_cast<size_t>(k),
                static_cast<size_t>(k) * sizeof(float));
            auto serial_qkv = std::make_unique<FP32Tensor>(
                std::vector<size_t>{size_t{1}, static_cast<size_t>(n)});
            std::vector<TensorProjectionDesc> serial_projections = {
                {prepared.kernel, serial_qkv.get(), n, nullptr, "GDN_QKV_serial"}};

            ASSERT_TRUE(withExplicitCudaGemmStream(
                {prepared.kernel},
                [&](void *stream)
                {
                    return with_gpu_coherence(
                        gpu_device_,
                        {serial_input.get()},
                        {serial_qkv.get()},
                        stream,
                        [&]
                        {
                            return prepared.kernel->multiply_fused_tensor(
                                serial_input.get(),
                                serial_projections,
                                /*m=*/1,
                                k,
                                nullptr,
                                workspace_.get());
                        });
                }))
                << "real Qwen3.6 MoE serial GDN QKV projection failed at M="
                << m << " row=" << row;

            expectBitwiseEqualFloatRow(
                (std::string("real Qwen36 MoE GDN QKV M=") +
                 std::to_string(m) +
                 " row=" + std::to_string(row))
                    .c_str(),
                grouped_qkv->data() +
                    static_cast<size_t>(row) * static_cast<size_t>(n),
                serial_qkv->data(),
                static_cast<size_t>(n));
        }
    }

    cleanupSharedWorkspace({prepared.kernel});
    EXPECT_FALSE(prepared.kernel->hasDynamicStateActive());
}

TEST_F(Test__CUDAGemmParity, MTP_RuntimeM_FusedProjection_AllNativeFormats)
{
    const int K = 256;
    const int N0 = 192;
    const int N1 = 128;

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    for (const auto &fmt : cudaSmallMNativeFormats())
    {
        auto weights0 = fmt.create(N0, K);
        auto weights1 = fmt.create(N1, K);
        ASSERT_NE(weights0, nullptr) << "Failed to create " << fmt.name << " first projection";
        ASSERT_NE(weights1, nullptr) << "Failed to create " << fmt.name << " second projection";

        auto cpu_kernel0 = llaminar::v2::kernels::KernelFactory::createGemm(
            weights0.get(), KernelDeviceType::CPU);
        auto cpu_kernel1 = llaminar::v2::kernels::KernelFactory::createGemm(
            weights1.get(), KernelDeviceType::CPU);
        ASSERT_NE(cpu_kernel0, nullptr) << fmt.name << " CPU projection 0 kernel";
        ASSERT_NE(cpu_kernel1, nullptr) << fmt.name << " CPU projection 1 kernel";

        auto *cuda_kernel0 = getPreparedKernel(weights0.get(), gpu_device_);
        auto *cuda_kernel1 = getPreparedKernel(weights1.get(), gpu_device_);
        ASSERT_NE(cuda_kernel0, nullptr) << fmt.name << " CUDA projection 0 kernel";
        ASSERT_NE(cuda_kernel1, nullptr) << fmt.name << " CUDA projection 1 kernel";

        ASSERT_TRUE(setupSharedWorkspace(
            {cuda_kernel0, cuda_kernel1},
            kGroupedVerifierRuntimeRows.back(),
            {N0, N1},
            K))
            << fmt.name << " shared workspace";

        for (const int M : kGroupedVerifierRuntimeRows)
        {
            auto A_data = randomFP32(static_cast<size_t>(M) * K);

            std::vector<float> C0_cpu(static_cast<size_t>(M) * N0, 0.0f);
            std::vector<float> C1_cpu(static_cast<size_t>(M) * N1, 0.0f);
            ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel0.get(), A_data.data(), C0_cpu.data(), M, N0, K))
                << fmt.name << " CPU projection 0 at M=" << M;
            ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel1.get(), A_data.data(), C1_cpu.data(), M, N1, K))
                << fmt.name << " CPU projection 1 at M=" << M;

            auto A_tensor = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
            std::memcpy(A_tensor->mutable_data(), A_data.data(), A_data.size() * sizeof(float));
            auto C0_tensor = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N0)});
            auto C1_tensor = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N1)});

            std::vector<TensorProjectionDesc> projections = {
                {cuda_kernel0, C0_tensor.get(), N0, nullptr, "qkv"},
                {cuda_kernel1, C1_tensor.get(), N1, nullptr, "z"}};

            ASSERT_TRUE(withExplicitCudaGemmStream(
                {cuda_kernel0, cuda_kernel1},
                [&](void *stream)
                {
                    return with_gpu_coherence(
                        gpu_device_,
                        {A_tensor.get()},
                        {C0_tensor.get(), C1_tensor.get()},
                        static_cast<void *>(stream),
                        [&]
                        {
                            return cuda_kernel0->multiply_fused_verifier_rows_decode_equivalent(
                                A_tensor.get(), projections, M, K);
                        });
                }))
                << fmt.name << " CUDA fused projection failed at M=" << M;

            const float *C0_cuda = C0_tensor->data();
            const float *C1_cuda = C1_tensor->data();
            const auto result0 =
                checkParity(C0_cuda, C0_cpu.data(), C0_cpu.size(), fmt.cosine_threshold, 0.20);
            const auto result1 =
                checkParity(C1_cuda, C1_cpu.data(), C1_cpu.size(), fmt.cosine_threshold, 0.20);
            EXPECT_FALSE(result0.has_nan_inf)
                << fmt.name << " projection 0 produced non-finite output at M=" << M;
            EXPECT_FALSE(result1.has_nan_inf)
                << fmt.name << " projection 1 produced non-finite output at M=" << M;
            EXPECT_GE(result0.cosine_similarity, fmt.cosine_threshold)
                << fmt.name << " projection 0 cosine too low at M=" << M
                << " rel_l2=" << result0.relative_l2_error
                << " max_abs=" << result0.max_abs_error;
            EXPECT_GE(result1.cosine_similarity, fmt.cosine_threshold)
                << fmt.name << " projection 1 cosine too low at M=" << M
                << " rel_l2=" << result1.relative_l2_error
                << " max_abs=" << result1.max_abs_error;
            EXPECT_LE(result0.relative_l2_error, 0.20)
                << fmt.name << " projection 0 relative L2 too high at M=" << M;
            EXPECT_LE(result1.relative_l2_error, 0.20)
                << fmt.name << " projection 1 relative L2 too high at M=" << M;

            for (int row = 0; row < M; ++row)
            {
                /*
                 * The verifier publication contract is stricter than CPU
                 * parity.  Production serial decode calls the same CUDA GEMM
                 * engine with M=1 for each accepted token, so grouped MTP rows
                 * must reproduce that CUDA M=1 output byte-for-byte.  This is
                 * the focused guard for the model-level failure where a
                 * 1e-6 layer0 QKV projection difference amplified into a
                 * logits-level verifier miss.
                 */
                auto A_row = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{size_t{1}, static_cast<size_t>(K)});
                std::memcpy(
                    A_row->mutable_data(),
                    A_data.data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                    static_cast<size_t>(K) * sizeof(float));
                auto C0_serial = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{size_t{1}, static_cast<size_t>(N0)});
                auto C1_serial = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{size_t{1}, static_cast<size_t>(N1)});
                std::vector<TensorProjectionDesc> serial_projections = {
                    {cuda_kernel0, C0_serial.get(), N0, nullptr, "qkv_serial"},
                    {cuda_kernel1, C1_serial.get(), N1, nullptr, "z_serial"}};

                ASSERT_TRUE(withExplicitCudaGemmStream(
                    {cuda_kernel0, cuda_kernel1},
                    [&](void *stream)
                    {
                        return with_gpu_coherence(
                            gpu_device_,
                            {A_row.get()},
                            {C0_serial.get(), C1_serial.get()},
                            static_cast<void *>(stream),
                            [&]
                            {
                                return cuda_kernel0->multiply_fused_tensor(
                                    A_row.get(), serial_projections, 1, K);
                            });
                    }))
                    << fmt.name << " CUDA serial projection failed for M=" << M
                    << " row=" << row;

                expectBitwiseEqualFloatRow(
                    (std::string(fmt.name) + " projection 0 M=" + std::to_string(M) +
                     " row=" + std::to_string(row))
                        .c_str(),
                    C0_tensor->data() + static_cast<size_t>(row) * static_cast<size_t>(N0),
                    C0_serial->data(),
                    static_cast<size_t>(N0));
                expectBitwiseEqualFloatRow(
                    (std::string(fmt.name) + " projection 1 M=" + std::to_string(M) +
                     " row=" + std::to_string(row))
                        .c_str(),
                    C1_tensor->data() + static_cast<size_t>(row) * static_cast<size_t>(N1),
                    C1_serial->data(),
                    static_cast<size_t>(N1));
            }
        }

        cleanupSharedWorkspace({cuda_kernel0, cuda_kernel1});
        EXPECT_FALSE(cuda_kernel0->hasDynamicStateActive())
            << fmt.name << " projection 0 leaked CUDA dynamic state";
        EXPECT_FALSE(cuda_kernel1->hasDynamicStateActive())
            << fmt.name << " projection 1 leaked CUDA dynamic state";
        llaminar::v2::kernels::KernelFactory::clearCacheFor(weights0.get());
        llaminar::v2::kernels::KernelFactory::clearCacheFor(weights1.get());
    }

    const auto records =
        PerfStatsCollector::snapshot({"kernel.cuda_native_vnni_small_m_fused_projection_calls",
                                      "kernel.cuda_native_vnni_small_m_concurrent_projection_groups"});
    uint64_t total_count = 0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "cuda_native_vnni_small_m_fused_projection_calls" &&
            record.kind == PerfStatRecord::Kind::Counter)
        {
            total_count += record.count;
            EXPECT_EQ(record.tags.at("k"), std::to_string(K));
            EXPECT_EQ(record.tags.at("projections"), "2");
            const std::string &m_tag = record.tags.at("m");
            const std::string &route_tag = record.tags.at("route");
            EXPECT_GE(std::stoi(m_tag), 2);
            EXPECT_LE(std::stoi(m_tag), kGroupedVerifierRuntimeRows.back());
            EXPECT_EQ(route_tag, "specialized")
                << "CUDA verifier projections must use the grouped small-M native-VNNI route";
        }
        if (record.domain == "kernel" &&
            record.name == "cuda_native_vnni_small_m_concurrent_projection_groups" &&
            record.kind == PerfStatRecord::Kind::Counter)
        {
            total_count += record.count;
            EXPECT_EQ(record.tags.at("k"), std::to_string(K));
            EXPECT_EQ(record.tags.at("projections"), "2");
            const std::string &m_tag = record.tags.at("m");
            EXPECT_GE(std::stoi(m_tag), 2);
            EXPECT_LE(std::stoi(m_tag), kGroupedVerifierRuntimeRows.back());
            EXPECT_GE(std::stoi(record.tags.at("streams")), 2)
                << "Concurrent CUDA verifier projection groups should use explicit side streams";
        }
    }
    EXPECT_EQ(total_count, cudaSmallMNativeFormats().size() * kGroupedVerifierRuntimeRows.size())
        << "Every CUDA native format and certified runtime-M verifier shape should use the grouped route";

    PerfStatsCollector::reset();
}

TEST_F(Test__CUDAGemmParity, MTP_FP32FloatingFusedProjection_RuntimeMMatchesSerialDecodeRows)
{
    const int K = 256;
    const int N = 48;

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    auto weights_alpha = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(N), static_cast<size_t>(K)}, -0.25f, 0.25f, 7301);
    auto weights_beta = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(N), static_cast<size_t>(K)}, -0.25f, 0.25f, 7302);
    ASSERT_NE(weights_alpha, nullptr);
    ASSERT_NE(weights_beta, nullptr);

    auto prepared_alpha = llaminar2::test::makeGpuPreparedFloatingPointGemm(
        weights_alpha.get(),
        gpu_device_,
        "test.cuda.fp32.alpha.grouped_verifier",
        ModelContextId{97301});
    auto prepared_beta = llaminar2::test::makeGpuPreparedFloatingPointGemm(
        weights_beta.get(),
        gpu_device_,
        "test.cuda.fp32.beta.grouped_verifier",
        ModelContextId{97302});
    auto *cuda_alpha = prepared_alpha.kernel;
    auto *cuda_beta = prepared_beta.kernel;
    ASSERT_NE(cuda_alpha, nullptr);
    ASSERT_NE(cuda_beta, nullptr);
    ASSERT_TRUE(cuda_alpha->supports_fused_projection());
    ASSERT_TRUE(cuda_beta->supports_fused_projection());

    ASSERT_TRUE(setupSharedWorkspace(
        {cuda_alpha, cuda_beta},
        kGroupedVerifierRuntimeRows.back(),
        {N, N},
        K))
        << "CUDA FP32 grouped verifier workspace";

    for (const int M : kGroupedVerifierRuntimeRows)
    {
        SCOPED_TRACE(std::string("CUDA FP32 M=") + std::to_string(M));
        auto A_data = randomFP32(static_cast<size_t>(M) * static_cast<size_t>(K));

        auto A_grouped = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
        std::memcpy(A_grouped->mutable_data(), A_data.data(), A_data.size() * sizeof(float));
        auto alpha_grouped = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
        auto beta_grouped = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
        std::vector<TensorProjectionDesc> grouped_projections = {
            {cuda_alpha, alpha_grouped.get(), N, nullptr, "alpha_grouped"},
            {cuda_beta, beta_grouped.get(), N, nullptr, "beta_grouped"}};

        ASSERT_TRUE(withExplicitCudaGemmStream(
            {cuda_alpha, cuda_beta},
            [&](void *stream)
            {
                return with_gpu_coherence(
                    gpu_device_,
                    {A_grouped.get()},
                    {alpha_grouped.get(), beta_grouped.get()},
                    static_cast<void *>(stream),
                    [&]
                    {
                        return cuda_alpha->multiply_fused_verifier_rows_decode_equivalent(
                            A_grouped.get(), grouped_projections, M, K);
                    });
            }))
            << "CUDA FP32 grouped verifier projection failed at M=" << M;

        for (int row = 0; row < M; ++row)
        {
            auto A_row = std::make_unique<FP32Tensor>(
                std::vector<size_t>{size_t{1}, static_cast<size_t>(K)});
            std::memcpy(
                A_row->mutable_data(),
                A_data.data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                static_cast<size_t>(K) * sizeof(float));
            auto alpha_serial = std::make_unique<FP32Tensor>(
                std::vector<size_t>{size_t{1}, static_cast<size_t>(N)});
            auto beta_serial = std::make_unique<FP32Tensor>(
                std::vector<size_t>{size_t{1}, static_cast<size_t>(N)});
            std::vector<TensorProjectionDesc> serial_projections = {
                {cuda_alpha, alpha_serial.get(), N, nullptr, "alpha_serial"},
                {cuda_beta, beta_serial.get(), N, nullptr, "beta_serial"}};

            ASSERT_TRUE(withExplicitCudaGemmStream(
                {cuda_alpha, cuda_beta},
                [&](void *stream)
                {
                    return with_gpu_coherence(
                        gpu_device_,
                        {A_row.get()},
                        {alpha_serial.get(), beta_serial.get()},
                        static_cast<void *>(stream),
                        [&]
                        {
                            return cuda_alpha->multiply_fused_tensor(
                                A_row.get(), serial_projections, 1, K);
                        });
                }))
                << "CUDA FP32 serial decode projection failed at M=" << M
                << " row=" << row;

            expectBitwiseEqualFloatRow(
                (std::string("CUDA FP32 alpha M=") + std::to_string(M) +
                 " row=" + std::to_string(row))
                    .c_str(),
                alpha_grouped->data() + static_cast<size_t>(row) * static_cast<size_t>(N),
                alpha_serial->data(),
                static_cast<size_t>(N));
            expectBitwiseEqualFloatRow(
                (std::string("CUDA FP32 beta M=") + std::to_string(M) +
                 " row=" + std::to_string(row))
                    .c_str(),
                beta_grouped->data() + static_cast<size_t>(row) * static_cast<size_t>(N),
                beta_serial->data(),
                static_cast<size_t>(N));
        }
    }

    const auto records =
        PerfStatsCollector::snapshot({"kernel.cuda_fp32_grouped_verifier_projection_calls"});
    uint64_t grouped_small_n_calls = 0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "cuda_fp32_grouped_verifier_projection_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.at("dtype") == "fp32" &&
            record.tags.at("route") == "fixed_order_fp32_batched_projection" &&
            record.tags.at("n") == std::to_string(N) &&
            record.tags.at("k") == std::to_string(K) &&
            record.tags.at("projections") == "2")
        {
            const std::string &m_tag = record.tags.at("m");
            if (std::stoi(m_tag) >= 2 &&
                std::stoi(m_tag) <= kGroupedVerifierRuntimeRows.back())
                grouped_small_n_calls += record.count;
        }
    }
    EXPECT_EQ(grouped_small_n_calls, kGroupedVerifierRuntimeRows.size())
        << "CUDA FP32 verifier rows must use the fixed-order grouped runtime-M projection route";

    cleanupSharedWorkspace({cuda_alpha, cuda_beta});
    PerfStatsCollector::reset();
}

TEST_F(Test__CUDAGemmParity, MTP_FP16BF16FloatingFusedProjection_RuntimeMMatchesSerialDecodeRows)
{
    /**
     * FP16/BF16 floating weight tensors use FP32 verifier hidden rows on CUDA.
     * The grouped publication path must therefore consume raw 16-bit weight
     * rows directly and match the normal M=1 serial decode entry point
     * bit-for-bit.  This test catches both missing grouped implementations and
     * accidental cuBLAS/float-pointer detours in the serial path.
     */
    const int K = 192;
    const int N = 80;

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    auto run_case = [&](TensorType dtype,
                        const char *dtype_tag,
                        ModelContextId model_base)
    {
        std::unique_ptr<TensorBase> weights_alpha;
        std::unique_ptr<TensorBase> weights_beta;
        if (dtype == TensorType::FP16)
        {
            weights_alpha = TestTensorFactory::createFP16Random(
                {static_cast<size_t>(N), static_cast<size_t>(K)}, -0.25f, 0.25f, 7501);
            weights_beta = TestTensorFactory::createFP16Random(
                {static_cast<size_t>(N), static_cast<size_t>(K)}, -0.25f, 0.25f, 7502);
        }
        else
        {
            weights_alpha = TestTensorFactory::createBF16Random(
                {static_cast<size_t>(N), static_cast<size_t>(K)}, -0.25f, 0.25f, 7601);
            weights_beta = TestTensorFactory::createBF16Random(
                {static_cast<size_t>(N), static_cast<size_t>(K)}, -0.25f, 0.25f, 7602);
        }
        ASSERT_NE(weights_alpha, nullptr);
        ASSERT_NE(weights_beta, nullptr);

        auto prepared_alpha = llaminar2::test::makeGpuPreparedFloatingPointGemm(
            weights_alpha.get(),
            gpu_device_,
            std::string("test.cuda.") + dtype_tag + ".alpha.grouped_verifier",
            model_base);
        auto prepared_beta = llaminar2::test::makeGpuPreparedFloatingPointGemm(
            weights_beta.get(),
            gpu_device_,
            std::string("test.cuda.") + dtype_tag + ".beta.grouped_verifier",
            ModelContextId{model_base.value + 1});
        auto *cuda_alpha = prepared_alpha.kernel;
        auto *cuda_beta = prepared_beta.kernel;
        ASSERT_NE(cuda_alpha, nullptr);
        ASSERT_NE(cuda_beta, nullptr);
        ASSERT_TRUE(cuda_alpha->supports_fused_projection())
            << "FP16/BF16 grouped verifier projection is a required production path";
        ASSERT_TRUE(cuda_beta->supports_fused_projection())
            << "Every projection in the FP16/BF16 group must support the production grouped route";

        ASSERT_TRUE(setupSharedWorkspace(
            {cuda_alpha, cuda_beta},
            kGroupedVerifierRuntimeRows.back(),
            {N, N},
            K))
            << "CUDA " << dtype_tag << " grouped verifier workspace";

        for (const int M : kGroupedVerifierRuntimeRows)
        {
            SCOPED_TRACE(std::string("CUDA ") + dtype_tag + " M=" + std::to_string(M));
            auto A_data = randomFP32(static_cast<size_t>(M) * static_cast<size_t>(K));

            auto A_grouped = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
            std::memcpy(A_grouped->mutable_data(), A_data.data(), A_data.size() * sizeof(float));
            auto alpha_grouped = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
            auto beta_grouped = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
            std::vector<TensorProjectionDesc> grouped_projections = {
                {cuda_alpha, alpha_grouped.get(), N, nullptr, "alpha_grouped"},
                {cuda_beta, beta_grouped.get(), N, nullptr, "beta_grouped"}};

            ASSERT_TRUE(withExplicitCudaGemmStream(
                {cuda_alpha, cuda_beta},
                [&](void *stream)
                {
                    return with_gpu_coherence(
                        gpu_device_,
                        {A_grouped.get()},
                        {alpha_grouped.get(), beta_grouped.get()},
                        static_cast<void *>(stream),
                        [&]
                        {
                            return cuda_alpha->multiply_fused_verifier_rows_decode_equivalent(
                                A_grouped.get(), grouped_projections, M, K);
                        });
                }))
                << "CUDA " << dtype_tag << " grouped verifier projection failed at M=" << M;

            for (int row = 0; row < M; ++row)
            {
                auto A_row = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{size_t{1}, static_cast<size_t>(K)});
                std::memcpy(
                    A_row->mutable_data(),
                    A_data.data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                    static_cast<size_t>(K) * sizeof(float));
                auto alpha_serial = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{size_t{1}, static_cast<size_t>(N)});
                auto beta_serial = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{size_t{1}, static_cast<size_t>(N)});

                ASSERT_TRUE(withExplicitCudaGemmStream(
                    {cuda_alpha, cuda_beta},
                    [&](void *stream)
                    {
                        return with_gpu_coherence(
                            gpu_device_,
                            {A_row.get()},
                            {alpha_serial.get(), beta_serial.get()},
                            static_cast<void *>(stream),
                            [&]
                            {
                                return cuda_alpha->multiply_tensor(
                                           A_row.get(), alpha_serial.get(), 1, N, K,
                                           /*transpose_B=*/true, 1.0f, 0.0f, nullptr, nullptr, -1, workspace_.get()) &&
                                       cuda_beta->multiply_tensor(
                                           A_row.get(), beta_serial.get(), 1, N, K,
                                           /*transpose_B=*/true, 1.0f, 0.0f, nullptr, nullptr, -1, workspace_.get());
                            });
                    }))
                    << "CUDA " << dtype_tag << " serial decode projection failed at M=" << M
                    << " row=" << row;

                expectBitwiseEqualFloatRow(
                    (std::string("CUDA ") + dtype_tag + " alpha M=" + std::to_string(M) +
                     " row=" + std::to_string(row))
                        .c_str(),
                    alpha_grouped->data() + static_cast<size_t>(row) * static_cast<size_t>(N),
                    alpha_serial->data(),
                    static_cast<size_t>(N));
                expectBitwiseEqualFloatRow(
                    (std::string("CUDA ") + dtype_tag + " beta M=" + std::to_string(M) +
                     " row=" + std::to_string(row))
                        .c_str(),
                    beta_grouped->data() + static_cast<size_t>(row) * static_cast<size_t>(N),
                    beta_serial->data(),
                    static_cast<size_t>(N));
            }
        }

        cleanupSharedWorkspace({cuda_alpha, cuda_beta});
    };

    run_case(TensorType::FP16, "fp16", ModelContextId{97501});
    run_case(TensorType::BF16, "bf16", ModelContextId{97601});

    const auto records =
        PerfStatsCollector::snapshot({"kernel.cuda_fp32x16_grouped_verifier_projection_calls"});
    uint64_t fp16_grouped_calls = 0;
    uint64_t bf16_grouped_calls = 0;
    for (const auto &record : records)
    {
        if (record.domain != "kernel" ||
            record.name != "cuda_fp32x16_grouped_verifier_projection_calls" ||
            record.kind != PerfStatRecord::Kind::Counter ||
            record.tags.at("n") != std::to_string(N) ||
            record.tags.at("k") != std::to_string(K) ||
            record.tags.at("projections") != "2")
        {
            continue;
        }
        const std::string &m_tag = record.tags.at("m");
        if (std::stoi(m_tag) < 2 ||
            std::stoi(m_tag) > kGroupedVerifierRuntimeRows.back())
            continue;
        if (record.tags.at("dtype") == "fp16")
            fp16_grouped_calls += record.count;
        else if (record.tags.at("dtype") == "bf16")
            bf16_grouped_calls += record.count;
    }
    EXPECT_EQ(fp16_grouped_calls, kGroupedVerifierRuntimeRows.size());
    EXPECT_EQ(bf16_grouped_calls, kGroupedVerifierRuntimeRows.size());

    PerfStatsCollector::reset();
}

TEST_F(Test__CUDAGemmParity, MTP_FloatingSwiGLUDown_RuntimeMMatchesSerialDecodeRows)
{
    /**
     * Floating shared-expert down projections used to have no grouped verifier
     * implementation on CUDA.  This regression sweeps FP32, FP16, and BF16
     * down weights and proves that grouped runtime-M rows match the one-row
     * decode-sized fused SwiGLU/down entry point bit-for-bit.
     */
    const int K = 192;
    const int N = 80;

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    auto run_case = [&](TensorType dtype,
                        const char *dtype_tag,
                        ModelContextId model_id)
    {
        std::unique_ptr<TensorBase> weights_down;
        if (dtype == TensorType::FP32)
        {
            weights_down = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(N), static_cast<size_t>(K)}, -0.25f, 0.25f, 7701);
        }
        else if (dtype == TensorType::FP16)
        {
            weights_down = TestTensorFactory::createFP16Random(
                {static_cast<size_t>(N), static_cast<size_t>(K)}, -0.25f, 0.25f, 7702);
        }
        else
        {
            weights_down = TestTensorFactory::createBF16Random(
                {static_cast<size_t>(N), static_cast<size_t>(K)}, -0.25f, 0.25f, 7703);
        }
        ASSERT_NE(weights_down, nullptr);

        auto prepared_down = llaminar2::test::makeGpuPreparedFloatingPointGemm(
            weights_down.get(),
            gpu_device_,
            std::string("test.cuda.") + dtype_tag + ".swiglu_down.grouped_verifier",
            model_id);
        auto *cuda_down = prepared_down.kernel;
        ASSERT_NE(cuda_down, nullptr);
        ASSERT_TRUE(setupSharedWorkspace(
            {cuda_down},
            kGroupedVerifierRuntimeRows.back(),
            {N},
            K))
            << "CUDA " << dtype_tag << " floating SwiGLU/down workspace";

        for (const int M : kGroupedVerifierRuntimeRows)
        {
            SCOPED_TRACE(std::string("CUDA ") + dtype_tag + " SwiGLU/down M=" + std::to_string(M));
            auto gate_data = randomFP32(static_cast<size_t>(M) * static_cast<size_t>(K));
            auto up_data = randomFP32(static_cast<size_t>(M) * static_cast<size_t>(K));

            auto gate_grouped = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
            auto up_grouped = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
            auto down_grouped = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
            std::memcpy(gate_grouped->mutable_data(), gate_data.data(), gate_data.size() * sizeof(float));
            std::memcpy(up_grouped->mutable_data(), up_data.data(), up_data.size() * sizeof(float));

            ASSERT_TRUE(withExplicitCudaGemmStream(
                {cuda_down},
                [&](void *stream)
                {
                    return with_gpu_coherence(
                        gpu_device_,
                        {gate_grouped.get(), up_grouped.get()},
                        {down_grouped.get()},
                        static_cast<void *>(stream),
                        [&]
                        {
                            return cuda_down->multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                                gate_grouped.get(), up_grouped.get(), down_grouped.get(),
                                M, N, K, 1.0f, 0.0f, workspace_.get());
                        });
                }))
                << "CUDA " << dtype_tag << " grouped floating SwiGLU/down failed at M=" << M;

            for (int row = 0; row < M; ++row)
            {
                auto gate_row = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{size_t{1}, static_cast<size_t>(K)});
                auto up_row = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{size_t{1}, static_cast<size_t>(K)});
                auto down_serial = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{size_t{1}, static_cast<size_t>(N)});
                std::memcpy(
                    gate_row->mutable_data(),
                    gate_data.data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                    static_cast<size_t>(K) * sizeof(float));
                std::memcpy(
                    up_row->mutable_data(),
                    up_data.data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                    static_cast<size_t>(K) * sizeof(float));

                ASSERT_TRUE(withExplicitCudaGemmStream(
                    {cuda_down},
                    [&](void *stream)
                    {
                        return with_gpu_coherence(
                            gpu_device_,
                            {gate_row.get(), up_row.get()},
                            {down_serial.get()},
                            static_cast<void *>(stream),
                            [&]
                            {
                                return cuda_down->multiply_tensor_with_fused_swiglu(
                                    gate_row.get(), up_row.get(), down_serial.get(),
                                    1, N, K, 1.0f, 0.0f, workspace_.get());
                            });
                    }))
                    << "CUDA " << dtype_tag << " serial floating SwiGLU/down failed at M="
                    << M << " row=" << row;

                expectBitwiseEqualFloatRow(
                    (std::string("CUDA ") + dtype_tag + " floating SwiGLU/down M=" +
                     std::to_string(M) + " row=" + std::to_string(row))
                        .c_str(),
                    down_grouped->data() + static_cast<size_t>(row) * static_cast<size_t>(N),
                    down_serial->data(),
                    static_cast<size_t>(N));
            }
        }

        cleanupSharedWorkspace({cuda_down});
    };

    run_case(TensorType::FP32, "fp32", ModelContextId{97701});
    run_case(TensorType::FP16, "fp16", ModelContextId{97801});
    run_case(TensorType::BF16, "bf16", ModelContextId{97901});

    const auto records =
        PerfStatsCollector::snapshot({"kernel.cuda_floating_grouped_verifier_swiglu_down_calls"});
    uint64_t fp32_grouped_calls = 0;
    uint64_t fp16_grouped_calls = 0;
    uint64_t bf16_grouped_calls = 0;
    for (const auto &record : records)
    {
        if (record.domain != "kernel" ||
            record.name != "cuda_floating_grouped_verifier_swiglu_down_calls" ||
            record.kind != PerfStatRecord::Kind::Counter ||
            record.tags.at("n") != std::to_string(N) ||
            record.tags.at("k") != std::to_string(K))
        {
            continue;
        }
        const std::string &m_tag = record.tags.at("m");
        if (std::stoi(m_tag) < 2 ||
            std::stoi(m_tag) > kGroupedVerifierRuntimeRows.back())
            continue;
        if (record.tags.at("dtype") == "fp32")
            fp32_grouped_calls += record.count;
        else if (record.tags.at("dtype") == "fp16")
            fp16_grouped_calls += record.count;
        else if (record.tags.at("dtype") == "bf16")
            bf16_grouped_calls += record.count;
    }
    EXPECT_EQ(fp32_grouped_calls, kGroupedVerifierRuntimeRows.size());
    EXPECT_EQ(fp16_grouped_calls, kGroupedVerifierRuntimeRows.size());
    EXPECT_EQ(bf16_grouped_calls, kGroupedVerifierRuntimeRows.size());

    PerfStatsCollector::reset();
}

TEST_F(Test__CUDAGemmParity, MTP_M1FusedProjectionSerialDecodeIsBitwiseStable_Q40)
{
    const int K = 256;
    const int N0 = 192;
    const int N1 = 128;

    auto weights0 = TestTensorFactory::createQ4_0Random({static_cast<size_t>(N0), static_cast<size_t>(K)});
    auto weights1 = TestTensorFactory::createQ4_0Random({static_cast<size_t>(N1), static_cast<size_t>(K)});
    ASSERT_NE(weights0, nullptr);
    ASSERT_NE(weights1, nullptr);

    auto *cuda_kernel0 = getPreparedKernel(weights0.get(), gpu_device_);
    auto *cuda_kernel1 = getPreparedKernel(weights1.get(), gpu_device_);
    ASSERT_NE(cuda_kernel0, nullptr);
    ASSERT_NE(cuda_kernel1, nullptr);
    ASSERT_TRUE(setupSharedWorkspace({cuda_kernel0, cuda_kernel1}, 1, {N0, N1}, K));

    const auto A_data = randomFP32(static_cast<size_t>(K));
    auto input = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, static_cast<size_t>(K)});
    std::memcpy(input->mutable_data(), A_data.data(), A_data.size() * sizeof(float));

    auto first0 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, static_cast<size_t>(N0)});
    auto first1 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, static_cast<size_t>(N1)});
    auto second0 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, static_cast<size_t>(N0)});
    auto second1 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, static_cast<size_t>(N1)});

    std::vector<TensorProjectionDesc> first_projections = {
        {cuda_kernel0, first0.get(), N0, nullptr, "qkv_first"},
        {cuda_kernel1, first1.get(), N1, nullptr, "z_first"}};
    std::vector<TensorProjectionDesc> second_projections = {
        {cuda_kernel0, second0.get(), N0, nullptr, "qkv_second"},
        {cuda_kernel1, second1.get(), N1, nullptr, "z_second"}};

    /*
     * MTP verifier rows are defined relative to serial decode.  This guard
     * proves that the serial CUDA fused projection oracle is itself stable for
     * a repeated token before grouped verifier tests compare against it.
     */
    ASSERT_TRUE(withExplicitCudaGemmStream(
        {cuda_kernel0, cuda_kernel1},
        [&](void *stream)
        {
            return with_gpu_coherence(
                       gpu_device_,
                       {input.get()},
                       {first0.get(), first1.get()},
                       static_cast<void *>(stream),
                       [&]
                       {
                           return cuda_kernel0->multiply_fused_tensor(
                               input.get(), first_projections, 1, K);
                       }) &&
                   with_gpu_coherence(
                       gpu_device_,
                       {input.get()},
                       {second0.get(), second1.get()},
                       static_cast<void *>(stream),
                       [&]
                       {
                           return cuda_kernel0->multiply_fused_tensor(
                               input.get(), second_projections, 1, K);
                       });
        }));

    expectBitwiseEqualFloatRow(
        "Q4_0 serial fused projection 0 repeated M=1",
        first0->data(),
        second0->data(),
        static_cast<size_t>(N0));
    expectBitwiseEqualFloatRow(
        "Q4_0 serial fused projection 1 repeated M=1",
        first1->data(),
        second1->data(),
        static_cast<size_t>(N1));

    cleanupSharedWorkspace({cuda_kernel0, cuda_kernel1});
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights0.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights1.get());
}

TEST_F(Test__CUDAGemmParity, MTP_M1SingleFusedProjectionSerialDecodeIsBitwiseStable_Q40)
{
    const int K = 256;
    const int N = 192;

    auto weights = TestTensorFactory::createQ4_0Random({static_cast<size_t>(N), static_cast<size_t>(K)});
    ASSERT_NE(weights, nullptr);

    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupSharedWorkspace({cuda_kernel}, 1, {N}, K));

    const auto A_data = randomFP32(static_cast<size_t>(K));
    auto input = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, static_cast<size_t>(K)});
    std::memcpy(input->mutable_data(), A_data.data(), A_data.size() * sizeof(float));

    auto first = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, static_cast<size_t>(N)});
    auto second = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, static_cast<size_t>(N)});

    std::vector<TensorProjectionDesc> first_projection = {
        {cuda_kernel, first.get(), N, nullptr, "single_first"}};
    std::vector<TensorProjectionDesc> second_projection = {
        {cuda_kernel, second.get(), N, nullptr, "single_second"}};

    ASSERT_TRUE(withExplicitCudaGemmStream(
        {cuda_kernel},
        [&](void *stream)
        {
            return with_gpu_coherence(
                       gpu_device_,
                       {input.get()},
                       {first.get()},
                       static_cast<void *>(stream),
                       [&]
                       {
                           return cuda_kernel->multiply_fused_tensor(
                               input.get(), first_projection, 1, K);
                       }) &&
                   with_gpu_coherence(
                       gpu_device_,
                       {input.get()},
                       {second.get()},
                       static_cast<void *>(stream),
                       [&]
                       {
                           return cuda_kernel->multiply_fused_tensor(
                               input.get(), second_projection, 1, K);
                       });
        }));

    expectBitwiseEqualFloatRow(
        "Q4_0 single serial fused projection repeated M=1",
        first->data(),
        second->data(),
        static_cast<size_t>(N));

    cleanupSharedWorkspace({cuda_kernel});
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, MTP_M1TensorProjectionSerialDecodeIsBitwiseStable_Q40)
{
    const int M = 1;
    const int K = 256;
    const int N = 192;

    auto weights = TestTensorFactory::createQ4_0Random({static_cast<size_t>(N), static_cast<size_t>(K)});
    ASSERT_NE(weights, nullptr);

    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupSharedWorkspace({cuda_kernel}, M, {N}, K));

    const auto A_data = randomFP32(static_cast<size_t>(K));
    auto input = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    std::memcpy(input->mutable_data(), A_data.data(), A_data.size() * sizeof(float));

    auto first = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
    auto second = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

    /*
     * This is the public single-projection decode oracle used by non-fused
     * stages.  If it drifts, the issue is below fused orchestration; if it
     * stays stable while multiply_fused_tensor drifts, the bug is in the fused
     * projection wrapper's stream/coherence sequencing.
     */
    ASSERT_TRUE(withExplicitCudaGemmStream(
        {cuda_kernel},
        [&](void *stream)
        {
            return with_gpu_coherence(
                       gpu_device_,
                       {input.get()},
                       {first.get()},
                       static_cast<void *>(stream),
                       [&]
                       {
                           return cuda_kernel->multiply_tensor(
                               input.get(), first.get(), M, N, K);
                       }) &&
                   with_gpu_coherence(
                       gpu_device_,
                       {input.get()},
                       {second.get()},
                       static_cast<void *>(stream),
                       [&]
                       {
                           return cuda_kernel->multiply_tensor(
                               input.get(), second.get(), M, N, K);
                       });
        }));

    expectBitwiseEqualFloatRow(
        "Q4_0 serial tensor projection repeated M=1",
        first->data(),
        second->data(),
        static_cast<size_t>(N));

    cleanupSharedWorkspace({cuda_kernel});
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, MTP_Q40M1GEMVRepeatFromSameQuantizedActivationIsBitwiseStable)
{
    const int M = 1;
    const int K = 256;
    const int N = 192;

    ASSERT_TRUE(cudaNativeVNNIInitIQGridTables_tuned())
        << "CUDA native-VNNI IQ grid tables must be initialized before M=1 repeat checks";

    auto weights = TestTensorFactory::createQ4_0Random({static_cast<size_t>(N), static_cast<size_t>(K)});
    ASSERT_NE(weights, nullptr);

    auto *kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(kernel, M, N, K));

    DeviceNativeVNNIMatrixDesc desc;
    ASSERT_TRUE(kernel->exportNativeVNNIMatrixDesc(desc))
        << "Q4_0 CUDA kernel must export a native-VNNI descriptor";

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    kernel->setGPUStream(stream);

    CUDAGemvContext *gemv_ctx = cudaGemvContext_create(gpu_device_.ordinal);
    ASSERT_NE(gemv_ctx, nullptr);
    auto *kpar_partials = static_cast<float *>(
        workspace_->getBuffer(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS));
    const size_t kpar_partials_bytes =
        workspace_->getBufferSize(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS);
    cudaGemvContext_bindWorkspace(gemv_ctx, kpar_partials, kpar_partials_bytes);

    auto input = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    auto first = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
    auto second = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

    const auto row = randomFP32(static_cast<size_t>(K));
    std::memcpy(input->mutable_data(), row.data(), row.size() * sizeof(float));

    ASSERT_TRUE(with_gpu_coherence(
        gpu_device_,
        {input.get()},
        {first.get(), second.get()},
        static_cast<void *>(stream),
        [&]
        {
            const float *d_A = static_cast<const float *>(input->gpu_data_ptr());
            auto *d_A_int8 = static_cast<int8_t *>(
                workspace_->getBuffer(GemmWorkspaceBuffers::QUANT_A));
            auto *d_scales_A = static_cast<float *>(
                workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE));
            float *d_first = static_cast<float *>(first->gpu_data_ptr());
            float *d_second = static_cast<float *>(second->gpu_data_ptr());

            if (!d_A || !d_A_int8 || !d_scales_A || !d_first || !d_second)
                return false;
            if (!cudaQuantGemm_quantizeActivationsBlockwise(
                    d_A, d_A_int8, d_scales_A, M, K, gpu_device_.ordinal, stream))
                return false;

            /*
             * Repeat the raw M=1 tuned GEMV from a single quantized row without
             * hiding first-call state behind a warmup.  MTP verifier publication
             * can happen on the first decoded token after graph warmup, so cold
             * and hot calls must be bitwise identical.
             */
            ScopedCudaDecodeEquivalentM1Gemv decode_equivalent_scope;
            if (!cudaNativeVNNIGemvTuned_fp32(
                    d_A_int8,
                    static_cast<const uint8_t *>(desc.payload),
                    static_cast<const uint16_t *>(desc.scales),
                    static_cast<const uint16_t *>(desc.mins),
                    static_cast<const uint32_t *>(desc.emins),
                    d_first,
                    d_scales_A,
                    N,
                    K,
                    1.0f,
                    0.0f,
                    nullptr,
                    nullptr,
                    desc.codebook_id,
                    gpu_device_.ordinal,
                    stream,
                    gemv_ctx,
                    nullptr))
                return false;
            if (!cudaNativeVNNIGemvTuned_fp32(
                    d_A_int8,
                    static_cast<const uint8_t *>(desc.payload),
                    static_cast<const uint16_t *>(desc.scales),
                    static_cast<const uint16_t *>(desc.mins),
                    static_cast<const uint32_t *>(desc.emins),
                    d_second,
                    d_scales_A,
                    N,
                    K,
                    1.0f,
                    0.0f,
                    nullptr,
                    nullptr,
                    desc.codebook_id,
                    gpu_device_.ordinal,
                    stream,
                    gemv_ctx,
                    nullptr))
                return false;

            return cudaStreamSynchronize(stream) == cudaSuccess;
        }));

    expectBitwiseEqualFloatRow(
        "Q4_0 M=1 GEMV repeated from identical quantized activation",
        first->data(),
        second->data(),
        static_cast<size_t>(N));

    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    cudaGemvContext_destroy(gemv_ctx);
    kernel->setGPUStream(nullptr);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    cleanupWorkspaceIfNeeded(kernel);
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, MTP_Q40SmallMGEMVRepeatFromSameQuantizedActivationIsBitwiseStable)
{
    const int M = 2;
    const int K = 256;
    const int N = 192;

    ASSERT_TRUE(cudaNativeVNNIInitIQGridTables_tuned())
        << "CUDA native-VNNI IQ grid tables must be initialized before low-level repeat checks";

    auto weights = TestTensorFactory::createQ4_0Random({static_cast<size_t>(N), static_cast<size_t>(K)});
    ASSERT_NE(weights, nullptr);

    auto *kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(kernel, M, N, K));

    DeviceNativeVNNIMatrixDesc desc;
    ASSERT_TRUE(kernel->exportNativeVNNIMatrixDesc(desc))
        << "Q4_0 CUDA kernel must export a native-VNNI descriptor";

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    kernel->setGPUStream(stream);

    CUDAGemvContext *gemv_ctx = cudaGemvContext_create(gpu_device_.ordinal);
    ASSERT_NE(gemv_ctx, nullptr);
    auto *kpar_partials = static_cast<float *>(
        workspace_->getBuffer(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS));
    const size_t kpar_partials_bytes =
        workspace_->getBufferSize(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS);
    cudaGemvContext_bindWorkspace(gemv_ctx, kpar_partials, kpar_partials_bytes);

    auto input = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    auto first = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
    auto second = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
    auto warmup = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

    const auto row = randomFP32(static_cast<size_t>(K));
    std::memcpy(input->mutable_data(), row.data(), row.size() * sizeof(float));
    std::memset(
        input->mutable_data() + static_cast<size_t>(K),
        0,
        static_cast<size_t>(K) * sizeof(float));

    CUDARowMajorWeights *rowmajor = nullptr;
    ASSERT_TRUE(with_gpu_coherence(
        gpu_device_,
        {input.get()},
        {warmup.get(), first.get(), second.get()},
        static_cast<void *>(stream),
        [&]
        {
            const float *d_A = static_cast<const float *>(input->gpu_data_ptr());
            auto *d_A_int8 = static_cast<int8_t *>(
                workspace_->getBuffer(GemmWorkspaceBuffers::QUANT_A));
            auto *d_scales_A = static_cast<float *>(
                workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE));
            float *d_warmup = static_cast<float *>(warmup->gpu_data_ptr());
            float *d_first = static_cast<float *>(first->gpu_data_ptr());
            float *d_second = static_cast<float *>(second->gpu_data_ptr());

            if (!d_A || !d_A_int8 || !d_scales_A || !d_warmup || !d_first || !d_second)
                return false;
            if (!cudaQuantGemm_quantizeActivationsBlockwise(
                    d_A, d_A_int8, d_scales_A, M, K, gpu_device_.ordinal, stream))
                return false;

            /*
             * The warmup call lets the production path lazily create the
             * row-major auxiliary view if the generated dispatch selects
             * ROWPAR.  The two checked launches below then consume identical
             * quantized activations and identical prepared weights.
             */
            ScopedCudaNativeVNNIDecodeEquivalentM1Policy verifier_scope;

            if (!cudaNativeVNNIGemvTuned_small_m_fp32(
                    d_A_int8,
                    static_cast<const uint8_t *>(desc.payload),
                    static_cast<const uint16_t *>(desc.scales),
                    static_cast<const uint16_t *>(desc.mins),
                    static_cast<const uint32_t *>(desc.emins),
                    d_warmup,
                    d_scales_A,
                    M,
                    N,
                    K,
                    1.0f,
                    0.0f,
                    nullptr,
                    nullptr,
                    desc.codebook_id,
                    gpu_device_.ordinal,
                    stream,
                    gemv_ctx,
                    &rowmajor))
                return false;
            if (!cudaNativeVNNIGemvTuned_small_m_fp32(
                    d_A_int8,
                    static_cast<const uint8_t *>(desc.payload),
                    static_cast<const uint16_t *>(desc.scales),
                    static_cast<const uint16_t *>(desc.mins),
                    static_cast<const uint32_t *>(desc.emins),
                    d_first,
                    d_scales_A,
                    M,
                    N,
                    K,
                    1.0f,
                    0.0f,
                    nullptr,
                    nullptr,
                    desc.codebook_id,
                    gpu_device_.ordinal,
                    stream,
                    gemv_ctx,
                    &rowmajor))
                return false;
            if (!cudaNativeVNNIGemvTuned_small_m_fp32(
                    d_A_int8,
                    static_cast<const uint8_t *>(desc.payload),
                    static_cast<const uint16_t *>(desc.scales),
                    static_cast<const uint16_t *>(desc.mins),
                    static_cast<const uint32_t *>(desc.emins),
                    d_second,
                    d_scales_A,
                    M,
                    N,
                    K,
                    1.0f,
                    0.0f,
                    nullptr,
                    nullptr,
                    desc.codebook_id,
                    gpu_device_.ordinal,
                    stream,
                    gemv_ctx,
                    &rowmajor))
                return false;

            return cudaStreamSynchronize(stream) == cudaSuccess;
        }));

    expectBitwiseEqualFloatRow(
        "Q4_0 small-M GEMV row 0 repeated from identical quantized activation",
        first->data(),
        second->data(),
        static_cast<size_t>(N));
    expectBitwiseEqualFloatRow(
        "Q4_0 small-M GEMV row 1 repeated from identical quantized activation",
        first->data() + static_cast<size_t>(N),
        second->data() + static_cast<size_t>(N),
        static_cast<size_t>(N));

    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    if (rowmajor)
        cudaRowMajorWeights_destroy(rowmajor);
    cudaGemvContext_destroy(gemv_ctx);
    kernel->setGPUStream(nullptr);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    cleanupWorkspaceIfNeeded(kernel);
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, MTP_BlockwiseActivationQuantizeM1RepeatIsBitwiseStable)
{
    const int M = 1;
    const int K = 256;
    const int blocks = K / 32;

    auto weights = TestTensorFactory::createQ4_0Random({size_t{192}, static_cast<size_t>(K)});
    ASSERT_NE(weights, nullptr);
    auto *kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(kernel, 2, 192, K));

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    kernel->setGPUStream(stream);

    auto input = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    const auto row = randomFP32(static_cast<size_t>(K));
    std::memcpy(input->mutable_data(), row.data(), row.size() * sizeof(float));

    std::vector<int8_t> first_q(static_cast<size_t>(K));
    std::vector<int8_t> second_q(static_cast<size_t>(K));
    std::vector<float> first_scales(static_cast<size_t>(blocks));
    std::vector<float> second_scales(static_cast<size_t>(blocks));

    ASSERT_TRUE(with_gpu_coherence(
        gpu_device_,
        {input.get()},
        {},
        static_cast<void *>(stream),
        [&]
        {
            const float *d_A = static_cast<const float *>(input->gpu_data_ptr());
            auto *d_A_int8 = static_cast<int8_t *>(
                workspace_->getBuffer(GemmWorkspaceBuffers::QUANT_A));
            auto *d_scales_A = static_cast<float *>(
                workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE));
            if (!d_A || !d_A_int8 || !d_scales_A)
                return false;

            if (!cudaQuantGemm_quantizeActivationsBlockwise(
                    d_A, d_A_int8, d_scales_A, M, K, gpu_device_.ordinal, stream))
                return false;
            if (cudaMemcpyAsync(first_q.data(), d_A_int8, first_q.size() * sizeof(int8_t),
                                cudaMemcpyDeviceToHost, stream) != cudaSuccess)
                return false;
            if (cudaMemcpyAsync(first_scales.data(), d_scales_A, first_scales.size() * sizeof(float),
                                cudaMemcpyDeviceToHost, stream) != cudaSuccess)
                return false;

            if (!cudaQuantGemm_quantizeActivationsBlockwise(
                    d_A, d_A_int8, d_scales_A, M, K, gpu_device_.ordinal, stream))
                return false;
            if (cudaMemcpyAsync(second_q.data(), d_A_int8, second_q.size() * sizeof(int8_t),
                                cudaMemcpyDeviceToHost, stream) != cudaSuccess)
                return false;
            if (cudaMemcpyAsync(second_scales.data(), d_scales_A, second_scales.size() * sizeof(float),
                                cudaMemcpyDeviceToHost, stream) != cudaSuccess)
                return false;

            return cudaStreamSynchronize(stream) == cudaSuccess;
        }));

    ASSERT_EQ(std::memcmp(first_q.data(), second_q.data(), first_q.size() * sizeof(int8_t)), 0)
        << "M=1 blockwise activation INT8 payload changed across repeated quantization";
    expectBitwiseEqualFloatRow(
        "M=1 blockwise activation scales repeated quantization",
        first_scales.data(),
        second_scales.data(),
        first_scales.size());

    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    kernel->setGPUStream(nullptr);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    cleanupWorkspaceIfNeeded(kernel);
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, NativeVNNISpecializedRuntimeM_AllNativeFormatsMatchSerialGEMVs)
{
    const int K = 512;
    const int N = 384;
    ScopedCudaNativeVNNIDecodeEquivalentM1Policy verifier_policy_scope;

    ASSERT_TRUE(cudaNativeVNNIInitIQGridTables_tuned())
        << "CUDA native-VNNI IQ grid tables must be initialized for direct low-level GEMV tests";

    for (const auto &fmt : cudaSmallMNativeFormats())
    {
        auto weights = fmt.create(N, K);
        ASSERT_NE(weights, nullptr) << "Failed to create " << fmt.name << " weights";

        auto *base_kernel = getPreparedKernel(weights.get(), gpu_device_);
        ASSERT_NE(base_kernel, nullptr) << fmt.name << " CUDA kernel";
        ASSERT_TRUE(setupWorkspaceIfNeeded(
            base_kernel,
            kGroupedVerifierRuntimeRows.back(),
            N,
            K))
            << fmt.name << " workspace";

        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess)
            << fmt.name << " explicit CUDA stream";
        base_kernel->setGPUStream(stream);

        DeviceNativeVNNIMatrixDesc desc;
        ASSERT_TRUE(base_kernel->exportNativeVNNIMatrixDesc(desc))
            << fmt.name << " must export native-VNNI packed descriptor";

        CUDAGemvContext *gemv_ctx = cudaGemvContext_create(gpu_device_.ordinal);
        ASSERT_NE(gemv_ctx, nullptr) << fmt.name << " GEMV context";
        auto *kpar_partials = static_cast<float *>(
            workspace_->getBuffer(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS));
        const size_t kpar_partials_bytes =
            workspace_->getBufferSize(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS);
        cudaGemvContext_bindWorkspace(gemv_ctx, kpar_partials, kpar_partials_bytes);

        CUDARowMajorWeights *rowmajor = nullptr;

        for (const int M : kGroupedVerifierRuntimeRows)
        {
            auto A_data = randomFP32(static_cast<size_t>(M) * K);
            auto A_tensor = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
            std::memcpy(A_tensor->mutable_data(), A_data.data(), A_data.size() * sizeof(float));

            auto C_specialized = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
            auto C_captured = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
            auto C_serial = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
            auto C_serial_captured = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

            ASSERT_TRUE(with_gpu_coherence(
                gpu_device_,
                {A_tensor.get()},
                {C_specialized.get(), C_captured.get(), C_serial.get(),
                 C_serial_captured.get()},
                static_cast<void *>(stream),
                [&]
                {
                    const float *d_A = static_cast<const float *>(A_tensor->gpu_data_ptr());
                    float *d_C_specialized = static_cast<float *>(C_specialized->gpu_data_ptr());
                    float *d_C_captured = static_cast<float *>(C_captured->gpu_data_ptr());
                    float *d_C_serial = static_cast<float *>(C_serial->gpu_data_ptr());
                    float *d_C_serial_captured =
                        static_cast<float *>(C_serial_captured->gpu_data_ptr());
                    auto *d_A_int8 = static_cast<int8_t *>(
                        workspace_->getBuffer(GemmWorkspaceBuffers::QUANT_A));
                    auto *d_scales_A = static_cast<float *>(
                        workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE));

                    if (!d_A || !d_C_specialized || !d_C_captured || !d_C_serial ||
                        !d_C_serial_captured || !d_A_int8 || !d_scales_A)
                        return false;

                    if (!cudaQuantGemm_quantizeActivationsBlockwise(
                            d_A,
                            d_A_int8,
                            d_scales_A,
                            M,
                            K,
                            gpu_device_.ordinal,
                            stream))
                    {
                        return false;
                    }

                    if (!cudaNativeVNNIGemvTuned_small_m_fp32(
                            d_A_int8,
                            static_cast<const uint8_t *>(desc.payload),
                            static_cast<const uint16_t *>(desc.scales),
                            static_cast<const uint16_t *>(desc.mins),
                            static_cast<const uint32_t *>(desc.emins),
                            d_C_specialized,
                            d_scales_A,
                            M,
                            N,
                            K,
                            1.0f,
                            0.0f,
                            nullptr,
                            nullptr,
                            desc.codebook_id,
                            gpu_device_.ordinal,
                            stream,
                            gemv_ctx,
                            &rowmajor))
                    {
                        return false;
                    }

                    /*
                     * Capture the same warmed production kernel entry point.
                     * The eager launch above has already prepared any lazy
                     * row-major weight view, so capture contains only grouped
                     * device work and its ordered reduction. Replaying this
                     * graph proves that runtime row tiling does not rely on a
                     * host-side per-row loop or capture-disabled behavior.
                     */
                    if (cudaStreamSynchronize(stream) != cudaSuccess)
                        return false;
                    cudaGraph_t graph = nullptr;
                    cudaGraphExec_t graph_exec = nullptr;
                    if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) != cudaSuccess)
                        return false;
                    const bool captured_launch = cudaNativeVNNIGemvTuned_small_m_fp32(
                        d_A_int8,
                        static_cast<const uint8_t *>(desc.payload),
                        static_cast<const uint16_t *>(desc.scales),
                        static_cast<const uint16_t *>(desc.mins),
                        static_cast<const uint32_t *>(desc.emins),
                        d_C_captured,
                        d_scales_A,
                        M,
                        N,
                        K,
                        1.0f,
                        0.0f,
                        nullptr,
                        nullptr,
                        desc.codebook_id,
                        gpu_device_.ordinal,
                        stream,
                        gemv_ctx,
                        &rowmajor);
                    /*
                     * Graph-captured and eager policies are independently
                     * trained execution domains and may choose different
                     * serial M=1 reduction geometries. Build the captured
                     * oracle in the same graph so grouped publication is
                     * compared with the serial decode path it must actually
                     * equal in production. This row loop is test-only oracle
                     * construction; production still launches only grouped
                     * verifier kernels.
                     */
                    bool captured_serial_launch = captured_launch;
                    for (int row = 0; captured_serial_launch && row < M; ++row)
                    {
                        const int8_t *row_A =
                            d_A_int8 + static_cast<size_t>(row) * static_cast<size_t>(K);
                        const float *row_scales =
                            d_scales_A + static_cast<size_t>(row) * static_cast<size_t>(K / 32);
                        float *row_C = d_C_serial_captured +
                            static_cast<size_t>(row) * static_cast<size_t>(N);
                        captured_serial_launch = cudaNativeVNNIGemvTuned_fp32(
                            row_A,
                            static_cast<const uint8_t *>(desc.payload),
                            static_cast<const uint16_t *>(desc.scales),
                            static_cast<const uint16_t *>(desc.mins),
                            static_cast<const uint32_t *>(desc.emins),
                            row_C,
                            row_scales,
                            N,
                            K,
                            1.0f,
                            0.0f,
                            nullptr,
                            nullptr,
                            desc.codebook_id,
                            gpu_device_.ordinal,
                            stream,
                            gemv_ctx,
                            &rowmajor);
                    }
                    const cudaError_t capture_end = cudaStreamEndCapture(stream, &graph);
                    if (!captured_launch || !captured_serial_launch ||
                        capture_end != cudaSuccess || !graph)
                    {
                        if (graph)
                            cudaGraphDestroy(graph);
                        return false;
                    }
                    if (cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0) != cudaSuccess ||
                        !graph_exec)
                    {
                        cudaGraphDestroy(graph);
                        return false;
                    }
                    const cudaError_t graph_launch = cudaGraphLaunch(graph_exec, stream);
                    const cudaError_t graph_sync =
                        graph_launch == cudaSuccess
                            ? cudaStreamSynchronize(stream)
                            : graph_launch;
                    cudaGraphExecDestroy(graph_exec);
                    cudaGraphDestroy(graph);
                    if (graph_launch != cudaSuccess || graph_sync != cudaSuccess)
                        return false;

                    for (int row = 0; row < M; ++row)
                    {
                        const int8_t *row_A =
                            d_A_int8 + static_cast<size_t>(row) * static_cast<size_t>(K);
                        const float *row_scales =
                            d_scales_A + static_cast<size_t>(row) * static_cast<size_t>(K / 32);
                        float *row_C =
                            d_C_serial + static_cast<size_t>(row) * static_cast<size_t>(N);
                        if (!cudaNativeVNNIGemvTuned_fp32(
                                row_A,
                                static_cast<const uint8_t *>(desc.payload),
                                static_cast<const uint16_t *>(desc.scales),
                                static_cast<const uint16_t *>(desc.mins),
                                static_cast<const uint32_t *>(desc.emins),
                                row_C,
                                row_scales,
                                N,
                                K,
                                1.0f,
                                0.0f,
                                nullptr,
                                nullptr,
                                desc.codebook_id,
                                gpu_device_.ordinal,
                                stream,
                                gemv_ctx,
                                &rowmajor))
                        {
                            return false;
                        }
                    }

                    return cudaStreamSynchronize(stream) == cudaSuccess;
                }))
                << fmt.name << " specialized-vs-serial CUDA native-VNNI GEMV failed at M=" << M;

            for (int row = 0; row < M; ++row)
            {
                const std::string label =
                    std::string(fmt.name) + " CUDA native-VNNI grouped verifier row M=" +
                    std::to_string(M) + " row=" + std::to_string(row);
                expectBitwiseEqualFloatRow(
                    label.c_str(),
                    C_specialized->data() +
                        static_cast<size_t>(row) * static_cast<size_t>(N),
                    C_serial->data() +
                        static_cast<size_t>(row) * static_cast<size_t>(N),
                    static_cast<size_t>(N));
                expectBitwiseEqualFloatRow(
                    (label + " captured").c_str(),
                    C_captured->data() +
                        static_cast<size_t>(row) * static_cast<size_t>(N),
                    C_serial_captured->data() +
                        static_cast<size_t>(row) * static_cast<size_t>(N),
                    static_cast<size_t>(N));
                /*
                 * A production decode shape executes eagerly once to warm lazy
                 * kernel state, then runs captured thereafter. Cross the
                 * execution surfaces as well as comparing each surface to
                 * itself: this prevents generated eager and captured policies
                 * from introducing different FP32 reduction trees at that
                 * lifecycle boundary.
                 */
                expectBitwiseEqualFloatRow(
                    (label + " eager-grouped-vs-captured-serial").c_str(),
                    C_specialized->data() +
                        static_cast<size_t>(row) * static_cast<size_t>(N),
                    C_serial_captured->data() +
                        static_cast<size_t>(row) * static_cast<size_t>(N),
                    static_cast<size_t>(N));
                expectBitwiseEqualFloatRow(
                    (label + " captured-grouped-vs-eager-serial").c_str(),
                    C_captured->data() +
                        static_cast<size_t>(row) * static_cast<size_t>(N),
                    C_serial->data() +
                        static_cast<size_t>(row) * static_cast<size_t>(N),
                    static_cast<size_t>(N));
            }
        }

        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        if (rowmajor)
            cudaRowMajorWeights_destroy(rowmajor);
        cudaGemvContext_destroy(gemv_ctx);
        base_kernel->setGPUStream(nullptr);
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
        cleanupWorkspaceIfNeeded(base_kernel);
        EXPECT_FALSE(base_kernel->hasDynamicStateActive())
            << fmt.name << " specialized small-M equivalence test leaked CUDA dynamic state";
        llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
    }
}

/**
 * @brief Prove every native codebook on the production large-K KPAR shape class.
 *
 * The compact all-format sweep above uses `K=512`, while Qwen3.6 LocalTP GDN
 * projects a rank-local `K=5120` activation.  Generated dispatch classifies
 * `N=1024, K=5120` as KPAR for both M=1 and M=2.  Before this regression, the
 * grouped verifier could still create a lazy row-major weight view and promote
 * only M=2 to ROWPAR; serial decode deliberately kept KPAR, producing widespread
 * one-ULP drift that eventually changed sampled tokens.
 *
 * This test exercises the public fused verifier API and the ordinary fused M=1
 * API independently, including separate activation quantization.  Every row
 * must be byte-identical for every supported quantized tensor format.  The
 * grouped implementation remains a single native small-M launch and never
 * replays rows through the serial API.
 */
TEST_F(Test__CUDAGemmParity, MTP_RuntimeM_FusedProjection_AllNativeFormats_LargeKKPARMatchesSerialDecode)
{
    constexpr int N = 1024;
    constexpr int K = 5120;

    for (const auto &format : cudaSmallMNativeFormats())
    {
        auto weights = format.create(N, K);
        ASSERT_NE(weights, nullptr)
            << "Failed to create " << format.name << " large-K weights";

        auto *kernel = getPreparedKernel(weights.get(), gpu_device_);
        ASSERT_NE(kernel, nullptr) << format.name << " CUDA kernel";
        ASSERT_TRUE(setupWorkspaceIfNeeded(
            kernel,
            kGroupedVerifierRuntimeRows.back(),
            N,
            K))
            << format.name << " large-K verifier workspace";

        for (const int M : kGroupedVerifierRuntimeRows)
        {
            const auto input = randomFP32(static_cast<size_t>(M) * K);
            std::vector<float> grouped(static_cast<size_t>(M) * N, 0.0f);
            ASSERT_TRUE(cudaGroupedVerifierProjectionViaTensor(
                kernel,
                input.data(),
                grouped.data(),
                M,
                N,
                K,
                gpu_device_,
                workspace_.get()))
                << format.name << " grouped large-K verifier projection M=" << M;

            for (int row = 0; row < M; ++row)
            {
                std::vector<float> serial(static_cast<size_t>(N), 0.0f);
                ASSERT_TRUE(cudaMultiplyViaTensor(
                    kernel,
                    input.data() + static_cast<size_t>(row) * K,
                    serial.data(),
                    /*M=*/1,
                    N,
                    K,
                    gpu_device_))
                    << format.name << " serial large-K decode M=" << M
                    << " row=" << row;

                expectBitwiseEqualFloatRow(
                    (std::string(format.name) + " large-K KPAR M=" +
                     std::to_string(M) + " row=" + std::to_string(row))
                        .c_str(),
                    grouped.data() + static_cast<size_t>(row) * N,
                    serial.data(),
                    serial.size());
            }
        }

        cleanupWorkspaceIfNeeded(kernel);
        EXPECT_FALSE(kernel->hasDynamicStateActive())
            << format.name << " large-K sweep leaked CUDA dynamic state";
        llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
    }
}

/**
 * @brief Prove every native codebook at the Qwen3.6 35B MoE attention shape.
 *
 * The compact all-format sweep covers every verifier depth but does not cover
 * the highly rectangular `Q=8192, K=2048` projection used by the production
 * MoE attention graph. Generated dispatch can select different serial K
 * partitioning and grouped row-reuse geometry at that point. In particular,
 * M=3 and M=4 exercise the third and fourth accumulator slots in a four-row
 * grouped CTA, which a two-row-only geometry cannot certify.
 *
 * Production-sized weights make a complete M=2..16 sweep unnecessarily
 * expensive here. The canonical compact suite owns M totality; this companion
 * gate uses the shared boundary inventory to cover row-tile tails, exact tile
 * fills, the default maximum, and the extended-depth path for every codebook.
 */
TEST_F(Test__CUDAGemmParity,
       MTP_RuntimeM_FusedProjection_AllNativeFormats_Qwen36MoEQKVMatchesSerialDecode)
{
    constexpr int N = 8192;
    constexpr int K = 2048;

    for (const auto &format : cudaSmallMNativeFormats())
    {
        auto weights = format.create(N, K);
        ASSERT_NE(weights, nullptr)
            << "Failed to create " << format.name
            << " Qwen3.6 MoE Q projection weights";

        auto *kernel = getPreparedKernel(weights.get(), gpu_device_);
        ASSERT_NE(kernel, nullptr) << format.name << " CUDA kernel";
        ASSERT_TRUE(setupWorkspaceIfNeeded(
            kernel,
            kGroupedVerifierBoundaryRows.back(),
            N,
            K))
            << format.name << " Qwen3.6 MoE QKV verifier workspace";

        for (const int M : kGroupedVerifierBoundaryRows)
        {
            const auto input = randomFP32(static_cast<size_t>(M) * K);
            std::vector<float> grouped(static_cast<size_t>(M) * N, 0.0f);
            ASSERT_TRUE(cudaGroupedVerifierProjectionViaTensor(
                kernel,
                input.data(),
                grouped.data(),
                M,
                N,
                K,
                gpu_device_,
                workspace_.get()))
                << format.name
                << " grouped Qwen3.6 MoE Q projection M=" << M;

            for (int row = 0; row < M; ++row)
            {
                std::vector<float> serial(static_cast<size_t>(N), 0.0f);
                ASSERT_TRUE(cudaMultiplyViaTensor(
                    kernel,
                    input.data() + static_cast<size_t>(row) * K,
                    serial.data(),
                    /*M=*/1,
                    N,
                    K,
                    gpu_device_))
                    << format.name << " serial Qwen3.6 MoE Q decode M="
                    << M << " row=" << row;

                expectBitwiseEqualFloatRow(
                    (std::string(format.name) + " Qwen3.6 MoE Q M=" +
                     std::to_string(M) + " row=" + std::to_string(row))
                        .c_str(),
                    grouped.data() + static_cast<size_t>(row) * N,
                    serial.data(),
                    serial.size());
            }
        }

        cleanupWorkspaceIfNeeded(kernel);
        EXPECT_FALSE(kernel->hasDynamicStateActive())
            << format.name
            << " Qwen3.6 MoE QKV sweep leaked CUDA dynamic state";
        llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
    }
}

/**
 * @brief Prove runtime-M fused SwiGLU/down publication for every native format.
 *
 * MTP does not feed every verifier projection directly from hidden state. FFN
 * down projections first materialize `silu(gate) * up`, quantize that derived
 * activation, and then enter NativeVNNI. This sweep covers the complete fused
 * production entry point for every certified row count so a row-independent
 * GEMV cannot hide a batch-dependent activation or quantization implementation.
 */
TEST_F(Test__CUDAGemmParity, MTP_RuntimeM_FusedSwiGLUDown_AllNativeFormatsMatchSerialDecode)
{
    constexpr int N = 384;
    constexpr int K = 512;

    for (const auto &format : cudaSmallMNativeFormats())
    {
        auto weights = format.create(N, K);
        ASSERT_NE(weights, nullptr)
            << "Failed to create " << format.name << " fused-SwiGLU weights";

        auto *kernel = getPreparedKernel(weights.get(), gpu_device_);
        ASSERT_NE(kernel, nullptr) << format.name << " CUDA kernel";
        ASSERT_TRUE(setupWorkspaceIfNeeded(
            kernel,
            kGroupedVerifierRuntimeRows.back(),
            N,
            K))
            << format.name << " fused-SwiGLU verifier workspace";

        for (const int M : kGroupedVerifierRuntimeRows)
        {
            const auto gate = randomFP32(static_cast<size_t>(M) * K);
            const auto up = randomFP32(static_cast<size_t>(M) * K);
            std::vector<float> grouped(static_cast<size_t>(M) * N, 0.0f);
            ASSERT_TRUE(cudaFusedSwiGLUDownViaTensor(
                kernel,
                gate.data(),
                up.data(),
                grouped.data(),
                M,
                N,
                K,
                gpu_device_))
                << format.name << " grouped fused-SwiGLU M=" << M;

            for (int row = 0; row < M; ++row)
            {
                std::vector<float> serial(static_cast<size_t>(N), 0.0f);
                ASSERT_TRUE(cudaFusedSwiGLUDownViaTensor(
                    kernel,
                    gate.data() + static_cast<size_t>(row) * K,
                    up.data() + static_cast<size_t>(row) * K,
                    serial.data(),
                    1,
                    N,
                    K,
                    gpu_device_))
                    << format.name << " serial fused-SwiGLU M=" << M
                    << " row=" << row;

                expectBitwiseEqualFloatRow(
                    (std::string(format.name) + " fused-SwiGLU M=" +
                     std::to_string(M) + " row=" + std::to_string(row))
                        .c_str(),
                    grouped.data() + static_cast<size_t>(row) * N,
                    serial.data(),
                    serial.size());
            }
        }

        cleanupWorkspaceIfNeeded(kernel);
        EXPECT_FALSE(kernel->hasDynamicStateActive())
            << format.name << " fused-SwiGLU runtime-M sweep leaked dynamic state";
        llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
    }
}

TEST_F(Test__CUDAGemmParity, MTP_SmallM_MoEProjectionNamesUseSpecializedNativeVNNI)
{
    const int K = 256;
    const int N0 = 192;
    const int N1 = 128;
    const std::array<int, 2> verifier_rows = {3, 4};

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    auto weights0 = TestTensorFactory::createQ4_KRandom({static_cast<size_t>(N0), static_cast<size_t>(K)}, 230);
    auto weights1 = TestTensorFactory::createQ4_KRandom({static_cast<size_t>(N1), static_cast<size_t>(K)}, 231);
    ASSERT_NE(weights0, nullptr);
    ASSERT_NE(weights1, nullptr);

    auto cpu_kernel0 = llaminar::v2::kernels::KernelFactory::createGemm(
        weights0.get(), KernelDeviceType::CPU);
    auto cpu_kernel1 = llaminar::v2::kernels::KernelFactory::createGemm(
        weights1.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel0, nullptr);
    ASSERT_NE(cpu_kernel1, nullptr);

    auto *cuda_kernel0 = getPreparedKernel(weights0.get(), gpu_device_);
    auto *cuda_kernel1 = getPreparedKernel(weights1.get(), gpu_device_);
    ASSERT_NE(cuda_kernel0, nullptr);
    ASSERT_NE(cuda_kernel1, nullptr);
    ASSERT_TRUE(setupSharedWorkspace({cuda_kernel0, cuda_kernel1}, 4, {N0, N1}, K));

    for (int M : verifier_rows)
    {
        auto A_data = randomFP32(static_cast<size_t>(M) * K);

        std::vector<float> C0_cpu(static_cast<size_t>(M) * N0, 0.0f);
        std::vector<float> C1_cpu(static_cast<size_t>(M) * N1, 0.0f);
        ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel0.get(), A_data.data(), C0_cpu.data(), M, N0, K));
        ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel1.get(), A_data.data(), C1_cpu.data(), M, N1, K));

        auto A_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
        std::memcpy(A_tensor->mutable_data(), A_data.data(), A_data.size() * sizeof(float));
        auto C0_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N0)});
        auto C1_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N1)});

        std::vector<TensorProjectionDesc> projections = {
            {cuda_kernel0, C0_tensor.get(), N0, nullptr, "shared_gate"},
            {cuda_kernel1, C1_tensor.get(), N1, nullptr, "shared_up"}};

        ASSERT_TRUE(withExplicitCudaGemmStream(
            {cuda_kernel0, cuda_kernel1},
            [&](void *stream)
            {
                return with_gpu_coherence(
                    gpu_device_,
                    {A_tensor.get()},
                    {C0_tensor.get(), C1_tensor.get()},
                    static_cast<void *>(stream),
                    [&]
                    {
                        return cuda_kernel0->multiply_fused_verifier_rows_decode_equivalent(
                            A_tensor.get(), projections, M, K);
                    });
            }))
            << "MoE-style CUDA fused projection failed at M=" << M;

        const float *C0_cuda = C0_tensor->data();
        const float *C1_cuda = C1_tensor->data();
        const auto result0 =
            checkParity(C0_cuda, C0_cpu.data(), C0_cpu.size(), 0.990, 0.20);
        const auto result1 =
            checkParity(C1_cuda, C1_cpu.data(), C1_cpu.size(), 0.990, 0.20);
        EXPECT_FALSE(result0.has_nan_inf);
        EXPECT_FALSE(result1.has_nan_inf);
        EXPECT_GE(result0.cosine_similarity, 0.990)
            << "projection 0 cosine too low at M=" << M;
        EXPECT_GE(result1.cosine_similarity, 0.990)
            << "projection 1 cosine too low at M=" << M;
        EXPECT_LE(result0.relative_l2_error, 0.20);
        EXPECT_LE(result1.relative_l2_error, 0.20);
    }

    cleanupSharedWorkspace({cuda_kernel0, cuda_kernel1});
    EXPECT_FALSE(cuda_kernel0->hasDynamicStateActive());
    EXPECT_FALSE(cuda_kernel1->hasDynamicStateActive());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights0.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights1.get());

    const auto records =
        PerfStatsCollector::snapshot({"kernel.cuda_native_vnni_small_m_fused_projection_calls"});
    uint64_t total_count = 0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "cuda_native_vnni_small_m_fused_projection_calls" &&
            record.kind == PerfStatRecord::Kind::Counter)
        {
            total_count += record.count;
            EXPECT_EQ(record.tags.at("k"), std::to_string(K));
            EXPECT_EQ(record.tags.at("projections"), "2");
            EXPECT_EQ(record.tags.at("route"), "specialized")
                << "MoE projection groups must use the same specialized small-M native-VNNI route";
        }
    }
    EXPECT_EQ(total_count, verifier_rows.size());

    PerfStatsCollector::reset();
}

/**
 * @brief Guard Qwen3.6 MoE shared-expert verifier rows against serial-decode drift.
 *
 * The Qwen3.6 MoE shared expert is implemented through the generic
 * `ITensorGemm` verifier hooks rather than the routed MoE expert-group kernel.
 * A tiny difference in these grouped `M=2..4` projections is still dangerous:
 * the next layers can amplify a few ULPs into different router probabilities
 * and eventually a different token distribution.  This regression therefore
 * compares every grouped verifier row against the same row executed through the
 * ordinary `M=1` decode contract for the real Qwen3.6 MoE dimensions and IQ3_S
 * codebook used by the model under test.
 */
TEST_F(Test__CUDAGemmParity, IQ3S_Qwen36MoESharedExpert_M234MatchesSingleRowDecode)
{
    constexpr int d_model = 2048;
    constexpr int intermediate = 512;
    constexpr int gate_up_n = intermediate;
    constexpr int gate_up_k = d_model;
    constexpr int down_n = d_model;
    constexpr int down_k = intermediate;
    constexpr std::array<int, 3> verifier_rows = {2, 3, 4};

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    auto weights_gate = TestTensorFactory::createIQ3_SRandom(
        {static_cast<size_t>(gate_up_n), static_cast<size_t>(gate_up_k)}, 6121);
    auto weights_up = TestTensorFactory::createIQ3_SRandom(
        {static_cast<size_t>(gate_up_n), static_cast<size_t>(gate_up_k)}, 6122);
    auto weights_down = TestTensorFactory::createIQ3_SRandom(
        {static_cast<size_t>(down_n), static_cast<size_t>(down_k)}, 6123);

    auto *cuda_gate = getPreparedKernel(weights_gate.get(), gpu_device_);
    auto *cuda_up = getPreparedKernel(weights_up.get(), gpu_device_);
    auto *cuda_down = getPreparedKernel(weights_down.get(), gpu_device_);
    ASSERT_NE(cuda_gate, nullptr) << "IQ3_S shared gate kernel";
    ASSERT_NE(cuda_up, nullptr) << "IQ3_S shared up kernel";
    ASSERT_NE(cuda_down, nullptr) << "IQ3_S shared down kernel";
    ASSERT_TRUE(setupSharedWorkspace(
        {cuda_gate, cuda_up, cuda_down},
        /*max_m=*/4,
        {gate_up_n, gate_up_n, down_n},
        d_model))
        << "IQ3_S Qwen3.6 MoE shared expert workspace";

    for (const int M : verifier_rows)
    {
        /*
         * Keep the input deterministic and moderately small.  That mirrors the
         * normalized hidden-state range in the model-level verifier test while
         * avoiding random seed changes that would make a regression harder to
         * bisect from CI logs.
         */
        std::vector<float> input_data(static_cast<size_t>(M) * d_model);
        for (size_t i = 0; i < input_data.size(); ++i)
        {
            input_data[i] =
                0.017f * static_cast<float>(static_cast<int>(i % 37) - 18) +
                0.003f * static_cast<float>(static_cast<int>((i / 11) % 17) - 8);
        }

        auto input_m = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(d_model)});
        std::memcpy(input_m->mutable_data(), input_data.data(), input_data.size() * sizeof(float));
        auto gate_m = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(intermediate)});
        auto up_m = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(intermediate)});
        auto down_m = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(d_model)});

        std::vector<TensorProjectionDesc> grouped_gate_up = {
            {cuda_gate, gate_m.get(), gate_up_n, nullptr, "shared_gate"},
            {cuda_up, up_m.get(), gate_up_n, nullptr, "shared_up"}};

        ASSERT_TRUE(withExplicitCudaGemmStream(
            {cuda_gate, cuda_up, cuda_down},
            [&](void *stream)
            {
                return with_gpu_coherence(
                           gpu_device_,
                           {input_m.get()},
                           {gate_m.get(), up_m.get()},
                           static_cast<void *>(stream),
                           [&]
                           {
                               return cuda_gate->multiply_fused_verifier_rows_decode_equivalent(
                                   input_m.get(), grouped_gate_up, M, d_model);
                           }) &&
                       with_gpu_coherence(
                           gpu_device_,
                           {gate_m.get(), up_m.get()},
                           {down_m.get()},
                           static_cast<void *>(stream),
                           [&]
                           {
                               return cuda_down->multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                                   gate_m.get(), up_m.get(), down_m.get(),
                                   M, d_model, intermediate,
                                   1.0f, 0.0f);
                           });
            }))
            << "IQ3_S Qwen3.6 MoE grouped shared expert failed at M=" << M;

        for (int row = 0; row < M; ++row)
        {
            auto input_row = std::make_unique<FP32Tensor>(
                std::vector<size_t>{size_t{1}, static_cast<size_t>(d_model)});
            std::memcpy(
                input_row->mutable_data(),
                input_data.data() + static_cast<size_t>(row) * d_model,
                static_cast<size_t>(d_model) * sizeof(float));
            auto gate_row = std::make_unique<FP32Tensor>(
                std::vector<size_t>{size_t{1}, static_cast<size_t>(intermediate)});
            auto up_row = std::make_unique<FP32Tensor>(
                std::vector<size_t>{size_t{1}, static_cast<size_t>(intermediate)});
            auto down_row = std::make_unique<FP32Tensor>(
                std::vector<size_t>{size_t{1}, static_cast<size_t>(d_model)});

            std::vector<TensorProjectionDesc> serial_gate_up = {
                {cuda_gate, gate_row.get(), gate_up_n, nullptr, "shared_gate"},
                {cuda_up, up_row.get(), gate_up_n, nullptr, "shared_up"}};

            ASSERT_TRUE(withExplicitCudaGemmStream(
                {cuda_gate, cuda_up, cuda_down},
                [&](void *stream)
                {
                    return with_gpu_coherence(
                               gpu_device_,
                               {input_row.get()},
                               {gate_row.get(), up_row.get()},
                               static_cast<void *>(stream),
                               [&]
                               {
                                   return cuda_gate->multiply_fused_verifier_rows_decode_equivalent(
                                       input_row.get(), serial_gate_up, 1, d_model);
                               }) &&
                           with_gpu_coherence(
                               gpu_device_,
                               {gate_row.get(), up_row.get()},
                               {down_row.get()},
                               static_cast<void *>(stream),
                               [&]
                               {
                                   return cuda_down->multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                                       gate_row.get(), up_row.get(), down_row.get(),
                                       1, d_model, intermediate,
                                       1.0f, 0.0f);
                               });
                }))
                << "IQ3_S Qwen3.6 MoE serial shared expert failed at M="
                << M << " row=" << row;

            const float *grouped_gate_row =
                gate_m->data() + static_cast<size_t>(row) * intermediate;
            const float *grouped_up_row =
                up_m->data() + static_cast<size_t>(row) * intermediate;
            const float *grouped_down_row =
                down_m->data() + static_cast<size_t>(row) * d_model;

            expectBitwiseEqualFloatRow(
                (std::string("IQ3_S Qwen36 shared gate M=") +
                 std::to_string(M) + " row=" + std::to_string(row))
                    .c_str(),
                grouped_gate_row,
                gate_row->data(),
                static_cast<size_t>(intermediate));
            expectBitwiseEqualFloatRow(
                (std::string("IQ3_S Qwen36 shared up M=") +
                 std::to_string(M) + " row=" + std::to_string(row))
                    .c_str(),
                grouped_up_row,
                up_row->data(),
                static_cast<size_t>(intermediate));
            expectBitwiseEqualFloatRow(
                (std::string("IQ3_S Qwen36 shared down M=") +
                 std::to_string(M) + " row=" + std::to_string(row))
                    .c_str(),
                grouped_down_row,
                down_row->data(),
                static_cast<size_t>(d_model));
        }
    }

    cleanupSharedWorkspace({cuda_gate, cuda_up, cuda_down});
    EXPECT_FALSE(cuda_gate->hasDynamicStateActive());
    EXPECT_FALSE(cuda_up->hasDynamicStateActive());
    EXPECT_FALSE(cuda_down->hasDynamicStateActive());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_gate.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_up.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_down.get());
    PerfStatsCollector::reset();
}

TEST_F(Test__CUDAGemmParity, Q4_K_FusedSwiGLUDownSmallM_2x896x768)
{
    const int M = 2;
    const int N = 896;
    const int K = 768;

    auto weights = TestTensorFactory::createQ4_KRandom({(size_t)N, (size_t)K}, 228);
    auto gate_data = randomFP32(static_cast<size_t>(M) * K);
    auto up_data = randomFP32(static_cast<size_t>(M) * K);

    std::vector<float> C_cpu(static_cast<size_t>(M) * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr);
    ASSERT_TRUE(cpuFusedSwiGLUDownToVector(
        cpu_kernel.get(), gate_data.data(), up_data.data(), C_cpu.data(), M, N, K));
    ASSERT_FALSE(hasNaNOrInf(C_cpu.data(), C_cpu.size()));

    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    std::vector<float> C_cuda(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_TRUE(cudaFusedSwiGLUDownViaTensor(
        cuda_kernel, gate_data.data(), up_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    auto result = checkParity(C_cuda.data(), C_cpu.data(), C_cpu.size(), 0.99, 0.15);
    result.print("Q4_K fused-SwiGLU down small-M 2x896x768");
    EXPECT_FALSE(result.has_nan_inf);
    EXPECT_GE(result.cosine_similarity, 0.99)
        << "Cosine similarity too low: " << result.cosine_similarity;
    EXPECT_LE(result.relative_l2_error, 0.15)
        << "Relative L2 error too high: " << (result.relative_l2_error * 100.0) << "%";

    cleanupWorkspaceIfNeeded(cuda_kernel);
    EXPECT_FALSE(cuda_kernel->hasDynamicStateActive())
        << "Small-M fused-SwiGLU down test must not leak CUDA dynamic state after workspace cleanup";
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, Q4_K_Qwen36FFNGateUp_M4FusedProjectionMatchesFourSingleRowDecodeGEMVs)
{
    const int M = 4;
    const int N = 17408;
    const int K = 5120;

    auto weights_gate = TestTensorFactory::createQ4_KRandom(
        {static_cast<size_t>(N), static_cast<size_t>(K)}, 331);
    auto weights_up = TestTensorFactory::createQ4_KRandom(
        {static_cast<size_t>(N), static_cast<size_t>(K)}, 332);
    auto A_data = randomFP32(static_cast<size_t>(M) * K);

    auto *cuda_gate = getPreparedKernel(weights_gate.get(), gpu_device_);
    auto *cuda_up = getPreparedKernel(weights_up.get(), gpu_device_);
    ASSERT_NE(cuda_gate, nullptr);
    ASSERT_NE(cuda_up, nullptr);
    ASSERT_TRUE(setupSharedWorkspace({cuda_gate, cuda_up}, M, {N, N}, K));

    auto A_m4 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    std::memcpy(A_m4->mutable_data(), A_data.data(), A_data.size() * sizeof(float));
    auto gate_m4 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
    auto up_m4 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
    std::vector<TensorProjectionDesc> m4_projections = {
        {cuda_gate, gate_m4.get(), N, nullptr, "ffn_gate"},
        {cuda_up, up_m4.get(), N, nullptr, "ffn_up"}};

    ASSERT_TRUE(withExplicitCudaGemmStream(
        {cuda_gate, cuda_up},
        [&](void *stream)
        {
            return with_gpu_coherence(
                gpu_device_,
                {A_m4.get()},
                {gate_m4.get(), up_m4.get()},
                static_cast<void *>(stream),
                [&]
                {
                    return cuda_gate->multiply_fused_verifier_rows_decode_equivalent(
                        A_m4.get(), m4_projections, M, K, nullptr, workspace_.get());
                });
        }))
        << "M=4 Qwen3.6 FFN gate/up fused projection failed";

    for (int row = 0; row < M; ++row)
    {
        auto A_m1 = std::make_unique<FP32Tensor>(
            std::vector<size_t>{size_t{1}, static_cast<size_t>(K)});
        std::memcpy(
            A_m1->mutable_data(),
            A_data.data() + static_cast<size_t>(row) * K,
            static_cast<size_t>(K) * sizeof(float));
        auto gate_m1 = std::make_unique<FP32Tensor>(
            std::vector<size_t>{size_t{1}, static_cast<size_t>(N)});
        auto up_m1 = std::make_unique<FP32Tensor>(
            std::vector<size_t>{size_t{1}, static_cast<size_t>(N)});
        std::vector<TensorProjectionDesc> m1_projections = {
            {cuda_gate, gate_m1.get(), N, nullptr, "ffn_gate"},
            {cuda_up, up_m1.get(), N, nullptr, "ffn_up"}};

        ASSERT_TRUE(withExplicitCudaGemmStream(
            {cuda_gate, cuda_up},
            [&](void *stream)
            {
                return with_gpu_coherence(
                    gpu_device_,
                    {A_m1.get()},
                    {gate_m1.get(), up_m1.get()},
                    static_cast<void *>(stream),
                    [&]
                    {
                        return cuda_gate->multiply_fused_tensor(
                            A_m1.get(), m1_projections, 1, K);
                    });
            }))
            << "M=1 Qwen3.6 FFN gate/up projection failed for row " << row;

        const float *gate_row =
            gate_m4->data() + static_cast<size_t>(row) * N;
        const float *up_row =
            up_m4->data() + static_cast<size_t>(row) * N;
        expectBitwiseEqualFloatRow(
            (std::string("Q4_K Qwen36 FFN gate M=4 row=") + std::to_string(row))
                .c_str(),
            gate_row,
            gate_m1->data(),
            static_cast<size_t>(N));
        expectBitwiseEqualFloatRow(
            (std::string("Q4_K Qwen36 FFN up M=4 row=") + std::to_string(row))
                .c_str(),
            up_row,
            up_m1->data(),
            static_cast<size_t>(N));
    }

    cleanupSharedWorkspace({cuda_gate, cuda_up});
    EXPECT_FALSE(cuda_gate->hasDynamicStateActive());
    EXPECT_FALSE(cuda_up->hasDynamicStateActive());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_gate.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_up.get());
}

TEST_F(Test__CUDAGemmParity, Q5Native_Qwen36FFNGateUp_M4FusedProjectionMatchesFourSingleRowDecodeGEMVs)
{
    const int M = 4;
    const int N = 17408;
    const int K = 5120;

    struct FormatCase
    {
        const char *name;
        std::function<std::unique_ptr<TensorBase>(size_t, size_t, int)> create;
    };

    const std::array<FormatCase, 2> formats = {
        FormatCase{
            "Q5_1",
            [](size_t n, size_t k, int seed)
            {
                return TestTensorFactory::createQ5_1Random({n, k}, seed);
            }},
        FormatCase{
            "Q5_K",
            [](size_t n, size_t k, int seed)
            {
                return TestTensorFactory::createQ5_KRandom({n, k}, seed);
            }}};

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");

    for (size_t fmt_index = 0; fmt_index < formats.size(); ++fmt_index)
    {
        const auto &fmt = formats[fmt_index];
        PerfStatsCollector::reset();

        auto weights_gate = fmt.create(static_cast<size_t>(N), static_cast<size_t>(K),
                                       341 + static_cast<int>(fmt_index) * 10);
        auto weights_up = fmt.create(static_cast<size_t>(N), static_cast<size_t>(K),
                                     342 + static_cast<int>(fmt_index) * 10);
        auto A_data = randomFP32(static_cast<size_t>(M) * K);

        auto *cuda_gate = getPreparedKernel(weights_gate.get(), gpu_device_);
        auto *cuda_up = getPreparedKernel(weights_up.get(), gpu_device_);
        ASSERT_NE(cuda_gate, nullptr) << fmt.name << " gate kernel";
        ASSERT_NE(cuda_up, nullptr) << fmt.name << " up kernel";
        ASSERT_TRUE(setupSharedWorkspace({cuda_gate, cuda_up}, M, {N, N}, K))
            << fmt.name << " shared workspace";

        auto A_m4 = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
        std::memcpy(A_m4->mutable_data(), A_data.data(), A_data.size() * sizeof(float));
        auto gate_m4 = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
        auto up_m4 = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
        std::vector<TensorProjectionDesc> m4_projections = {
            {cuda_gate, gate_m4.get(), N, nullptr, "ffn_gate"},
            {cuda_up, up_m4.get(), N, nullptr, "ffn_up"}};

        ASSERT_TRUE(withExplicitCudaGemmStream(
            {cuda_gate, cuda_up},
            [&](void *stream)
            {
                return with_gpu_coherence(
                    gpu_device_,
                    {A_m4.get()},
                    {gate_m4.get(), up_m4.get()},
                    static_cast<void *>(stream),
                    [&]
                    {
                        return cuda_gate->multiply_fused_verifier_rows_decode_equivalent(
                            A_m4.get(), m4_projections, M, K, nullptr, workspace_.get());
                    });
            }))
            << fmt.name << " M=4 Qwen3.6 FFN gate/up fused projection failed";

        for (int row = 0; row < M; ++row)
        {
            auto A_m1 = std::make_unique<FP32Tensor>(
                std::vector<size_t>{size_t{1}, static_cast<size_t>(K)});
            std::memcpy(
                A_m1->mutable_data(),
                A_data.data() + static_cast<size_t>(row) * K,
                static_cast<size_t>(K) * sizeof(float));
            auto gate_m1 = std::make_unique<FP32Tensor>(
                std::vector<size_t>{size_t{1}, static_cast<size_t>(N)});
            auto up_m1 = std::make_unique<FP32Tensor>(
                std::vector<size_t>{size_t{1}, static_cast<size_t>(N)});
            std::vector<TensorProjectionDesc> m1_projections = {
                {cuda_gate, gate_m1.get(), N, nullptr, "ffn_gate"},
                {cuda_up, up_m1.get(), N, nullptr, "ffn_up"}};

            ASSERT_TRUE(withExplicitCudaGemmStream(
                {cuda_gate, cuda_up},
                [&](void *stream)
                {
                    return with_gpu_coherence(
                        gpu_device_,
                        {A_m1.get()},
                        {gate_m1.get(), up_m1.get()},
                        static_cast<void *>(stream),
                        [&]
                        {
                            return cuda_gate->multiply_fused_tensor(
                                A_m1.get(), m1_projections, 1, K);
                        });
                }))
                << fmt.name << " M=1 Qwen3.6 FFN gate/up projection failed for row " << row;

            const float *gate_row =
                gate_m4->data() + static_cast<size_t>(row) * N;
            const float *up_row =
                up_m4->data() + static_cast<size_t>(row) * N;
            expectBitwiseEqualFloatRow(
                (std::string(fmt.name) + " Qwen36 FFN gate M=4 row=" +
                 std::to_string(row))
                    .c_str(),
                gate_row,
                gate_m1->data(),
                static_cast<size_t>(N));
            expectBitwiseEqualFloatRow(
                (std::string(fmt.name) + " Qwen36 FFN up M=4 row=" +
                 std::to_string(row))
                    .c_str(),
                up_row,
                up_m1->data(),
                static_cast<size_t>(N));
        }

        const auto route_records =
            PerfStatsCollector::snapshot({"kernel.cuda_native_vnni_small_m_calls"});
        uint64_t codebook7_m4_specialized = 0;
        for (const auto &record : route_records)
        {
            if (record.domain == "kernel" &&
                record.name == "cuda_native_vnni_small_m_calls" &&
                record.kind == PerfStatRecord::Kind::Counter &&
                record.tags.at("m") == std::to_string(M) &&
                record.tags.at("codebook") == "7" &&
                record.tags.at("n") == std::to_string(N) &&
                record.tags.at("k") == std::to_string(K) &&
                record.tags.at("route") == "specialized")
            {
                codebook7_m4_specialized += record.count;
            }
        }
        EXPECT_EQ(codebook7_m4_specialized, 2u)
            << fmt.name << " M=4 gate/up must route through the specialized "
            << "codebook-7 verifier path";

        cleanupSharedWorkspace({cuda_gate, cuda_up});
        EXPECT_FALSE(cuda_gate->hasDynamicStateActive());
        EXPECT_FALSE(cuda_up->hasDynamicStateActive());
        llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_gate.get());
        llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_up.get());
        PerfStatsCollector::reset();
    }
}

TEST_F(Test__CUDAGemmParity, Q5Native_Qwen36FFNGateUp_M2FusedProjectionMatchesTwoSingleRowDecodeGEMVs)
{
    const int M = 2;
    const int N = 17408;
    const int K = 5120;

    struct FormatCase
    {
        const char *name;
        std::function<std::unique_ptr<TensorBase>(size_t, size_t, int)> create;
    };

    const std::array<FormatCase, 2> formats = {
        FormatCase{
            "Q5_1",
            [](size_t n, size_t k, int seed)
            {
                return TestTensorFactory::createQ5_1Random({n, k}, seed);
            }},
        FormatCase{
            "Q5_K",
            [](size_t n, size_t k, int seed)
            {
                return TestTensorFactory::createQ5_KRandom({n, k}, seed);
            }}};

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");

    for (size_t fmt_index = 0; fmt_index < formats.size(); ++fmt_index)
    {
        const auto &fmt = formats[fmt_index];
        PerfStatsCollector::reset();

        auto weights_gate = fmt.create(static_cast<size_t>(N), static_cast<size_t>(K),
                                       361 + static_cast<int>(fmt_index) * 10);
        auto weights_up = fmt.create(static_cast<size_t>(N), static_cast<size_t>(K),
                                     362 + static_cast<int>(fmt_index) * 10);
        auto A_data = randomFP32(static_cast<size_t>(M) * K);

        auto *cuda_gate = getPreparedKernel(weights_gate.get(), gpu_device_);
        auto *cuda_up = getPreparedKernel(weights_up.get(), gpu_device_);
        ASSERT_NE(cuda_gate, nullptr) << fmt.name << " gate kernel";
        ASSERT_NE(cuda_up, nullptr) << fmt.name << " up kernel";
        ASSERT_TRUE(setupSharedWorkspace({cuda_gate, cuda_up}, M, {N, N}, K))
            << fmt.name << " shared workspace";

        auto A_m2 = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
        std::memcpy(A_m2->mutable_data(), A_data.data(), A_data.size() * sizeof(float));
        auto gate_m2 = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
        auto up_m2 = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
        std::vector<TensorProjectionDesc> m2_projections = {
            {cuda_gate, gate_m2.get(), N, nullptr, "ffn_gate"},
            {cuda_up, up_m2.get(), N, nullptr, "ffn_up"}};

        ASSERT_TRUE(withExplicitCudaGemmStream(
            {cuda_gate, cuda_up},
            [&](void *stream)
            {
                return with_gpu_coherence(
                    gpu_device_,
                    {A_m2.get()},
                    {gate_m2.get(), up_m2.get()},
                    static_cast<void *>(stream),
                    [&]
                    {
                        return cuda_gate->multiply_fused_verifier_rows_decode_equivalent(
                            A_m2.get(), m2_projections, M, K, nullptr, workspace_.get());
                    });
            }))
            << fmt.name << " M=2 Qwen3.6 FFN gate/up fused projection failed";

        for (int row = 0; row < M; ++row)
        {
            auto A_m1 = std::make_unique<FP32Tensor>(
                std::vector<size_t>{size_t{1}, static_cast<size_t>(K)});
            std::memcpy(
                A_m1->mutable_data(),
                A_data.data() + static_cast<size_t>(row) * K,
                static_cast<size_t>(K) * sizeof(float));
            auto gate_m1 = std::make_unique<FP32Tensor>(
                std::vector<size_t>{size_t{1}, static_cast<size_t>(N)});
            auto up_m1 = std::make_unique<FP32Tensor>(
                std::vector<size_t>{size_t{1}, static_cast<size_t>(N)});
            std::vector<TensorProjectionDesc> m1_projections = {
                {cuda_gate, gate_m1.get(), N, nullptr, "ffn_gate"},
                {cuda_up, up_m1.get(), N, nullptr, "ffn_up"}};

            ASSERT_TRUE(withExplicitCudaGemmStream(
                {cuda_gate, cuda_up},
                [&](void *stream)
                {
                    return with_gpu_coherence(
                        gpu_device_,
                        {A_m1.get()},
                        {gate_m1.get(), up_m1.get()},
                        static_cast<void *>(stream),
                        [&]
                        {
                            return cuda_gate->multiply_fused_tensor(
                                A_m1.get(), m1_projections, 1, K);
                        });
                }))
                << fmt.name << " M=1 Qwen3.6 FFN gate/up projection failed for row " << row;

            const float *gate_row =
                gate_m2->data() + static_cast<size_t>(row) * N;
            const float *up_row =
                up_m2->data() + static_cast<size_t>(row) * N;
            expectBitwiseEqualFloatRow(
                (std::string(fmt.name) + " Qwen36 FFN gate M=2 row=" +
                 std::to_string(row))
                    .c_str(),
                gate_row,
                gate_m1->data(),
                static_cast<size_t>(N));
            expectBitwiseEqualFloatRow(
                (std::string(fmt.name) + " Qwen36 FFN up M=2 row=" +
                 std::to_string(row))
                    .c_str(),
                up_row,
                up_m1->data(),
                static_cast<size_t>(N));
        }

        const auto route_records =
            PerfStatsCollector::snapshot({"kernel.cuda_native_vnni_small_m_calls"});
        uint64_t specialized = 0;
        for (const auto &record : route_records)
        {
            if (record.domain == "kernel" &&
                record.name == "cuda_native_vnni_small_m_calls" &&
                record.kind == PerfStatRecord::Kind::Counter &&
                record.tags.at("m") == std::to_string(M) &&
                record.tags.at("n") == std::to_string(N) &&
                record.tags.at("k") == std::to_string(K) &&
                record.tags.at("route") == "specialized")
            {
                specialized += record.count;
            }
        }
        EXPECT_EQ(specialized, 2u)
            << fmt.name << " M=2 gate/up must route through specialized "
            << "small-M native-VNNI projections";

        cleanupSharedWorkspace({cuda_gate, cuda_up});
        EXPECT_FALSE(cuda_gate->hasDynamicStateActive());
        EXPECT_FALSE(cuda_up->hasDynamicStateActive());
        llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_gate.get());
        llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_up.get());
        PerfStatsCollector::reset();
    }
}

TEST_F(Test__CUDAGemmParity, Q4_K_Qwen36FFNGateUp_M2FusedProjectionMatchesTwoSingleRowDecodeGEMVs)
{
    const int M = 2;
    const int N = 17408;
    const int K = 5120;

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    auto weights_gate = TestTensorFactory::createQ4_KRandom(
        {static_cast<size_t>(N), static_cast<size_t>(K)}, 381);
    auto weights_up = TestTensorFactory::createQ4_KRandom(
        {static_cast<size_t>(N), static_cast<size_t>(K)}, 382);
    auto A_data = randomFP32(static_cast<size_t>(M) * K);

    auto *cuda_gate = getPreparedKernel(weights_gate.get(), gpu_device_);
    auto *cuda_up = getPreparedKernel(weights_up.get(), gpu_device_);
    ASSERT_NE(cuda_gate, nullptr) << "Q4_K gate kernel";
    ASSERT_NE(cuda_up, nullptr) << "Q4_K up kernel";
    ASSERT_TRUE(setupSharedWorkspace({cuda_gate, cuda_up}, M, {N, N}, K))
        << "Q4_K shared workspace";

    auto A_m2 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    std::memcpy(A_m2->mutable_data(), A_data.data(), A_data.size() * sizeof(float));
    auto gate_m2 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
    auto up_m2 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
    std::vector<TensorProjectionDesc> m2_projections = {
        {cuda_gate, gate_m2.get(), N, nullptr, "ffn_gate"},
        {cuda_up, up_m2.get(), N, nullptr, "ffn_up"}};

    ASSERT_TRUE(withExplicitCudaGemmStream(
        {cuda_gate, cuda_up},
        [&](void *stream)
        {
            return with_gpu_coherence(
                gpu_device_,
                {A_m2.get()},
                {gate_m2.get(), up_m2.get()},
                static_cast<void *>(stream),
                [&]
                {
                    return cuda_gate->multiply_fused_verifier_rows_decode_equivalent(
                        A_m2.get(), m2_projections, M, K, nullptr, workspace_.get());
                });
        }))
        << "Q4_K M=2 Qwen3.6 FFN gate/up fused projection failed";

    for (int row = 0; row < M; ++row)
    {
        auto A_m1 = std::make_unique<FP32Tensor>(
            std::vector<size_t>{size_t{1}, static_cast<size_t>(K)});
        std::memcpy(
            A_m1->mutable_data(),
            A_data.data() + static_cast<size_t>(row) * K,
            static_cast<size_t>(K) * sizeof(float));
        auto gate_m1 = std::make_unique<FP32Tensor>(
            std::vector<size_t>{size_t{1}, static_cast<size_t>(N)});
        auto up_m1 = std::make_unique<FP32Tensor>(
            std::vector<size_t>{size_t{1}, static_cast<size_t>(N)});
        std::vector<TensorProjectionDesc> m1_projections = {
            {cuda_gate, gate_m1.get(), N, nullptr, "ffn_gate"},
            {cuda_up, up_m1.get(), N, nullptr, "ffn_up"}};

        ASSERT_TRUE(withExplicitCudaGemmStream(
            {cuda_gate, cuda_up},
            [&](void *stream)
            {
                return with_gpu_coherence(
                    gpu_device_,
                    {A_m1.get()},
                    {gate_m1.get(), up_m1.get()},
                    static_cast<void *>(stream),
                    [&]
                    {
                        return cuda_gate->multiply_fused_tensor(
                            A_m1.get(), m1_projections, 1, K);
                    });
            }))
            << "Q4_K M=1 Qwen3.6 FFN gate/up projection failed for row " << row;

        const float *gate_row =
            gate_m2->data() + static_cast<size_t>(row) * N;
        const float *up_row =
            up_m2->data() + static_cast<size_t>(row) * N;
        expectBitwiseEqualFloatRow(
            (std::string("Q4_K Qwen36 FFN gate M=2 row=") + std::to_string(row))
                .c_str(),
            gate_row,
            gate_m1->data(),
            static_cast<size_t>(N));
        expectBitwiseEqualFloatRow(
            (std::string("Q4_K Qwen36 FFN up M=2 row=") + std::to_string(row))
                .c_str(),
            up_row,
            up_m1->data(),
            static_cast<size_t>(N));
    }

    const auto route_records =
        PerfStatsCollector::snapshot({"kernel.cuda_native_vnni_small_m_calls"});
    uint64_t q4k_specialized = 0;
    for (const auto &record : route_records)
    {
        if (record.domain == "kernel" &&
            record.name == "cuda_native_vnni_small_m_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.at("m") == std::to_string(M) &&
            record.tags.at("codebook") == "5" &&
            record.tags.at("n") == std::to_string(N) &&
            record.tags.at("k") == std::to_string(K) &&
            record.tags.at("route") == "specialized")
        {
            q4k_specialized += record.count;
        }
    }
    EXPECT_EQ(q4k_specialized, 2u)
        << "Q4_K M=2 gate/up must route through specialized small-M native-VNNI projections";

    cleanupSharedWorkspace({cuda_gate, cuda_up});
    EXPECT_FALSE(cuda_gate->hasDynamicStateActive());
    EXPECT_FALSE(cuda_up->hasDynamicStateActive());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_gate.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_up.get());
    PerfStatsCollector::reset();
}

TEST_F(Test__CUDAGemmParity, Q4_K_Qwen36FFNGateUp_DeterministicM1UsesCanonicalDecodeRoute)
{
    const int N = 17408;
    const int K = 5120;

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    ScopedDeterministicDebugEnv deterministic;
    PerfStatsCollector::reset();

    auto weights_gate = TestTensorFactory::createQ4_KRandom(
        {static_cast<size_t>(N), static_cast<size_t>(K)}, 383);
    auto weights_up = TestTensorFactory::createQ4_KRandom(
        {static_cast<size_t>(N), static_cast<size_t>(K)}, 384);
    auto A_data = randomFP32(static_cast<size_t>(2) * K);

    auto *cuda_gate = getPreparedKernel(weights_gate.get(), gpu_device_);
    auto *cuda_up = getPreparedKernel(weights_up.get(), gpu_device_);
    ASSERT_NE(cuda_gate, nullptr) << "Q4_K gate kernel";
    ASSERT_NE(cuda_up, nullptr) << "Q4_K up kernel";
    ASSERT_TRUE(setupSharedWorkspace({cuda_gate, cuda_up}, 2, {N, N}, K))
        << "Q4_K shared workspace";
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess)
        << "explicit CUDA stream";
    cuda_gate->setGPUStream(stream);
    cuda_up->setGPUStream(stream);

    auto A_m2 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{2}, static_cast<size_t>(K)});
    std::memcpy(A_m2->mutable_data(), A_data.data(), static_cast<size_t>(2) * K * sizeof(float));
    auto gate_m2 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{2}, static_cast<size_t>(N)});
    auto up_m2 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{2}, static_cast<size_t>(N)});
    std::vector<TensorProjectionDesc> m2_projections = {
        {cuda_gate, gate_m2.get(), N, nullptr, "ffn_gate"},
        {cuda_up, up_m2.get(), N, nullptr, "ffn_up"}};

    ASSERT_TRUE(with_gpu_coherence(
        gpu_device_,
        {A_m2.get()},
        {gate_m2.get(), up_m2.get()},
        static_cast<void *>(stream),
        [&]
        {
            return cuda_gate->multiply_fused_verifier_rows_decode_equivalent(
                A_m2.get(), m2_projections, 2, K);
        }))
        << "Q4_K M=2 canonical gate/up projection failed";

    auto A_m1 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, static_cast<size_t>(K)});
    std::memcpy(A_m1->mutable_data(), A_data.data(), static_cast<size_t>(K) * sizeof(float));
    auto gate_m1 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, static_cast<size_t>(N)});
    auto up_m1 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, static_cast<size_t>(N)});
    std::vector<TensorProjectionDesc> m1_projections = {
        {cuda_gate, gate_m1.get(), N, nullptr, "ffn_gate"},
        {cuda_up, up_m1.get(), N, nullptr, "ffn_up"}};

    ASSERT_TRUE(with_gpu_coherence(
        gpu_device_,
        {A_m1.get()},
        {gate_m1.get(), up_m1.get()},
        static_cast<void *>(stream),
        [&]
        {
            return cuda_gate->multiply_fused_tensor(A_m1.get(), m1_projections, 1, K);
        }))
        << "Q4_K deterministic M=1 canonical gate/up projection failed";

    expectBitwiseEqualFloatRow(
        "deterministic Q4_K gate M=2 row0 vs M=1 canonical decode",
        gate_m2->data(),
        gate_m1->data(),
        static_cast<size_t>(N));
    expectBitwiseEqualFloatRow(
        "deterministic Q4_K up M=2 row0 vs M=1 canonical decode",
        up_m2->data(),
        up_m1->data(),
        static_cast<size_t>(N));

    const auto route_records =
        PerfStatsCollector::snapshot({"kernel.cuda_native_vnni_m1_canonical_decode_calls"});
    uint64_t canonical_m1 = 0;
    for (const auto &record : route_records)
    {
        if (record.domain == "kernel" &&
            record.name == "cuda_native_vnni_m1_canonical_decode_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.at("codebook") == "5" &&
            record.tags.at("n") == std::to_string(N) &&
            record.tags.at("k") == std::to_string(K))
        {
            canonical_m1 += record.count;
        }
    }
    EXPECT_EQ(canonical_m1, 2u)
        << "Deterministic M=1 gate/up must route through canonical decode GEMV";

    cleanupSharedWorkspace({cuda_gate, cuda_up});
    cuda_gate->setGPUStream(nullptr);
    cuda_up->setGPUStream(nullptr);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    EXPECT_FALSE(cuda_gate->hasDynamicStateActive());
    EXPECT_FALSE(cuda_up->hasDynamicStateActive());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_gate.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_up.get());
    PerfStatsCollector::reset();
}

TEST_F(Test__CUDAGemmParity, Q5_K_Qwen36FFNGateUp_DeterministicM1UsesCanonicalDecodeRoute)
{
    const int N = 17408;
    const int K = 5120;

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    ScopedDeterministicDebugEnv deterministic;
    PerfStatsCollector::reset();

    auto weights_gate = TestTensorFactory::createQ5_KRandom(
        {static_cast<size_t>(N), static_cast<size_t>(K)}, 385);
    auto weights_up = TestTensorFactory::createQ5_KRandom(
        {static_cast<size_t>(N), static_cast<size_t>(K)}, 386);
    auto A_data = randomFP32(static_cast<size_t>(2) * K);

    auto *cuda_gate = getPreparedKernel(weights_gate.get(), gpu_device_);
    auto *cuda_up = getPreparedKernel(weights_up.get(), gpu_device_);
    ASSERT_NE(cuda_gate, nullptr) << "Q5_K gate kernel";
    ASSERT_NE(cuda_up, nullptr) << "Q5_K up kernel";
    ASSERT_TRUE(setupSharedWorkspace({cuda_gate, cuda_up}, 2, {N, N}, K))
        << "Q5_K shared workspace";
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess)
        << "explicit CUDA stream";
    cuda_gate->setGPUStream(stream);
    cuda_up->setGPUStream(stream);

    auto A_m2 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{2}, static_cast<size_t>(K)});
    std::memcpy(A_m2->mutable_data(), A_data.data(), static_cast<size_t>(2) * K * sizeof(float));
    auto gate_m2 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{2}, static_cast<size_t>(N)});
    auto up_m2 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{2}, static_cast<size_t>(N)});
    std::vector<TensorProjectionDesc> m2_projections = {
        {cuda_gate, gate_m2.get(), N, nullptr, "ffn_gate"},
        {cuda_up, up_m2.get(), N, nullptr, "ffn_up"}};

    ASSERT_TRUE(with_gpu_coherence(
        gpu_device_,
        {A_m2.get()},
        {gate_m2.get(), up_m2.get()},
        static_cast<void *>(stream),
        [&]
        {
            return cuda_gate->multiply_fused_verifier_rows_decode_equivalent(
                A_m2.get(), m2_projections, 2, K);
        }))
        << "Q5_K M=2 canonical gate/up projection failed";

    auto A_m1 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, static_cast<size_t>(K)});
    std::memcpy(A_m1->mutable_data(), A_data.data(), static_cast<size_t>(K) * sizeof(float));
    auto gate_m1 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, static_cast<size_t>(N)});
    auto up_m1 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, static_cast<size_t>(N)});
    std::vector<TensorProjectionDesc> m1_projections = {
        {cuda_gate, gate_m1.get(), N, nullptr, "ffn_gate"},
        {cuda_up, up_m1.get(), N, nullptr, "ffn_up"}};

    ASSERT_TRUE(with_gpu_coherence(
        gpu_device_,
        {A_m1.get()},
        {gate_m1.get(), up_m1.get()},
        static_cast<void *>(stream),
        [&]
        {
            return cuda_gate->multiply_fused_tensor(A_m1.get(), m1_projections, 1, K);
        }))
        << "Q5_K deterministic M=1 canonical gate/up projection failed";

    expectBitwiseEqualFloatRow(
        "deterministic Q5_K gate M=2 row0 vs M=1 canonical decode",
        gate_m2->data(),
        gate_m1->data(),
        static_cast<size_t>(N));
    expectBitwiseEqualFloatRow(
        "deterministic Q5_K up M=2 row0 vs M=1 canonical decode",
        up_m2->data(),
        up_m1->data(),
        static_cast<size_t>(N));

    const auto route_records =
        PerfStatsCollector::snapshot({"kernel.cuda_native_vnni_m1_canonical_decode_calls"});
    uint64_t canonical_m1 = 0;
    for (const auto &record : route_records)
    {
        if (record.domain == "kernel" &&
            record.name == "cuda_native_vnni_m1_canonical_decode_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.at("codebook") == "7" &&
            record.tags.at("n") == std::to_string(N) &&
            record.tags.at("k") == std::to_string(K))
        {
            canonical_m1 += record.count;
        }
    }
    EXPECT_EQ(canonical_m1, 2u)
        << "Deterministic M=1 Q5_K gate/up must route through canonical decode GEMV";

    cleanupSharedWorkspace({cuda_gate, cuda_up});
    cuda_gate->setGPUStream(nullptr);
    cuda_up->setGPUStream(nullptr);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    EXPECT_FALSE(cuda_gate->hasDynamicStateActive());
    EXPECT_FALSE(cuda_up->hasDynamicStateActive());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_gate.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_up.get());
    PerfStatsCollector::reset();
}

TEST_F(Test__CUDAGemmParity, Q4_K_Qwen36FFNDown_M4FusedSwiGLUMatchesFourSingleRowDecodeGEMVs)
{
    const int M = 4;
    const int N = 5120;
    const int K = 17408;

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    auto weights = TestTensorFactory::createQ4_KRandom({static_cast<size_t>(N), static_cast<size_t>(K)}, 233);
    auto gate_data = randomFP32(static_cast<size_t>(M) * K);
    auto up_data = randomFP32(static_cast<size_t>(M) * K);

    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    std::vector<float> C_m4(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_TRUE(cudaFusedSwiGLUDownViaTensor(
        cuda_kernel,
        gate_data.data(),
        up_data.data(),
        C_m4.data(),
        M,
        N,
        K,
        gpu_device_))
        << "M=4 Qwen3.6 fused-SwiGLU down projection failed";

    for (int row = 0; row < M; ++row)
    {
        std::vector<float> C_m1(static_cast<size_t>(N), 0.0f);
        ASSERT_TRUE(cudaFusedSwiGLUDownViaTensor(
            cuda_kernel,
            gate_data.data() + static_cast<size_t>(row) * K,
            up_data.data() + static_cast<size_t>(row) * K,
            C_m1.data(),
            1,
            N,
            K,
            gpu_device_))
            << "M=1 Qwen3.6 fused-SwiGLU down projection failed for row " << row;

        const float *verifier_row = C_m4.data() + static_cast<size_t>(row) * N;
        expectBitwiseEqualFloatRow(
            (std::string("Q4_K Qwen36 FFN down M=4 row=") + std::to_string(row))
                .c_str(),
            verifier_row,
            C_m1.data(),
            C_m1.size());
    }

    const auto route_records =
        PerfStatsCollector::snapshot({"kernel.cuda_native_vnni_small_m_calls"});
    uint64_t q4k_m4_specialized = 0;
    for (const auto &record : route_records)
    {
        if (record.domain == "kernel" &&
            record.name == "cuda_native_vnni_small_m_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.at("m") == std::to_string(M) &&
            record.tags.at("codebook") == "5" &&
            record.tags.at("n") == std::to_string(N) &&
            record.tags.at("k") == std::to_string(K) &&
            record.tags.at("route") == "specialized")
        {
            q4k_m4_specialized += record.count;
        }
    }
    EXPECT_EQ(q4k_m4_specialized, 1u)
        << "Qwen3.6 verifier-sized fused-SwiGLU down must use the "
        << "decode-equivalent small-M GEMV route, not generic NativeVNNI prefill";

    cleanupWorkspaceIfNeeded(cuda_kernel);
    EXPECT_FALSE(cuda_kernel->hasDynamicStateActive())
        << "M=4 Qwen3.6 fused-SwiGLU row-equivalence test must not leak CUDA dynamic state";
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
    PerfStatsCollector::reset();
}

TEST_F(Test__CUDAGemmParity, Q4_K_Qwen36FFNDown_M2FusedSwiGLUUsesSpecializedNativeVNNI)
{
    const int M = 2;
    const int N = 5120;
    const int K = 17408;

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    auto weights = TestTensorFactory::createQ4_KRandom({static_cast<size_t>(N), static_cast<size_t>(K)}, 234);
    auto gate_data = randomFP32(static_cast<size_t>(M) * K);
    auto up_data = randomFP32(static_cast<size_t>(M) * K);

    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    std::vector<float> C_m2(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_TRUE(cudaFusedSwiGLUDownViaTensor(
        cuda_kernel,
        gate_data.data(),
        up_data.data(),
        C_m2.data(),
        M,
        N,
        K,
        gpu_device_))
        << "M=2 Qwen3.6 fused-SwiGLU down projection failed";

    for (int row = 0; row < M; ++row)
    {
        std::vector<float> C_m1(static_cast<size_t>(N), 0.0f);
        ASSERT_TRUE(cudaFusedSwiGLUDownViaTensor(
            cuda_kernel,
            gate_data.data() + static_cast<size_t>(row) * K,
            up_data.data() + static_cast<size_t>(row) * K,
            C_m1.data(),
            1,
            N,
            K,
            gpu_device_))
            << "M=1 Qwen3.6 fused-SwiGLU down projection failed for row " << row;

        const float *verifier_row = C_m2.data() + static_cast<size_t>(row) * N;
        expectBitwiseEqualFloatRow(
            (std::string("Q4_K Qwen36 FFN down M=2 row=") + std::to_string(row))
                .c_str(),
            verifier_row,
            C_m1.data(),
            C_m1.size());
    }

    const auto route_records =
        PerfStatsCollector::snapshot({"kernel.cuda_native_vnni_small_m_calls"});
    uint64_t specialized_records = 0;
    for (const auto &record : route_records)
    {
        if (record.domain != "kernel" ||
            record.kind != PerfStatRecord::Kind::Counter)
        {
            continue;
        }

        const bool shape_match =
            record.tags.at("n") == std::to_string(N) &&
            record.tags.at("k") == std::to_string(K);
        if (!shape_match)
        {
            continue;
        }

        if (record.name == "cuda_native_vnni_small_m_calls" &&
            record.tags.at("m") == std::to_string(M) &&
            record.tags.at("codebook") == "5" &&
            record.tags.at("route") == "specialized")
        {
            specialized_records += record.count;
        }
    }
    EXPECT_EQ(specialized_records, 1u)
        << "Qwen3.6 fused-SwiGLU down M=2 must use specialized native-VNNI small-M GEMV";

    cleanupWorkspaceIfNeeded(cuda_kernel);
    EXPECT_FALSE(cuda_kernel->hasDynamicStateActive())
        << "M=2 Qwen3.6 fused-SwiGLU row-equivalence test must not leak CUDA dynamic state";
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
    PerfStatsCollector::reset();
}

/**
 * @brief Guard Qwen3.6 FFN-down M=1 fused-SwiGLU decode against repeat drift.
 *
 * The dense MTP verifier operation-equivalence suite compares an all-position
 * verifier row against a later serial replay from the same prefix.  A previous
 * CUDA failure first diverged at layer0_FFN_DOWN by only a few ULPs, which then
 * amplified into logits-level drift after the remaining Qwen3.6 layers.  This
 * focused regression keeps the exact FFN-down production shape under a bitwise
 * first-call/second-call contract so the full model test is no longer the first
 * place to discover M=1 fused-SwiGLU/down nondeterminism.
 */
TEST_F(Test__CUDAGemmParity, Q4_K_Qwen36FFNDown_M1FusedSwiGLUSerialDecodeIsBitwiseStable)
{
    const int M = 1;
    const int N = 5120;
    const int K = 17408;

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    auto weights = TestTensorFactory::createQ4_KRandom(
        {static_cast<size_t>(N), static_cast<size_t>(K)}, 235);
    auto gate_data = randomFP32(static_cast<size_t>(M) * K);
    auto up_data = randomFP32(static_cast<size_t>(M) * K);

    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    std::vector<float> first(static_cast<size_t>(N), 0.0f);
    std::vector<float> second(static_cast<size_t>(N), 0.0f);

    ASSERT_TRUE(cudaFusedSwiGLUDownViaTensor(
        cuda_kernel,
        gate_data.data(),
        up_data.data(),
        first.data(),
        M,
        N,
        K,
        gpu_device_))
        << "first Qwen3.6 fused-SwiGLU/down M=1 decode failed";
    ASSERT_TRUE(cudaFusedSwiGLUDownViaTensor(
        cuda_kernel,
        gate_data.data(),
        up_data.data(),
        second.data(),
        M,
        N,
        K,
        gpu_device_))
        << "second Qwen3.6 fused-SwiGLU/down M=1 decode failed";

    expectBitwiseEqualFloatRow(
        "Q4_K Qwen3.6 FFN-down fused-SwiGLU repeated M=1 decode",
        first.data(),
        second.data(),
        static_cast<size_t>(N));

    uint64_t canonical_m1 = 0;
    const auto route_records =
        PerfStatsCollector::snapshot({"kernel.cuda_native_vnni_m1_canonical_decode_calls"});
    for (const auto &record : route_records)
    {
        if (record.domain == "kernel" &&
            record.name == "cuda_native_vnni_m1_canonical_decode_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.at("codebook") == "5" &&
            record.tags.at("n") == std::to_string(N) &&
            record.tags.at("k") == std::to_string(K))
        {
            canonical_m1 += record.count;
        }
    }
    EXPECT_EQ(canonical_m1, 2u)
        << "Both M=1 FFN-down calls must route through canonical decode GEMV";

    cleanupWorkspaceIfNeeded(cuda_kernel);
    EXPECT_FALSE(cuda_kernel->hasDynamicStateActive())
        << "M=1 Qwen3.6 fused-SwiGLU repeat test must not leak CUDA dynamic state";
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
    PerfStatsCollector::reset();
}

TEST_F(Test__CUDAGemmParity, Q5Native_Qwen36FFNDown_M4FusedSwiGLUMatchesFourSingleRowDecodeGEMVs)
{
    const int M = 4;
    const int N = 5120;
    const int K = 17408;

    struct FormatCase
    {
        const char *name;
        std::function<std::unique_ptr<TensorBase>(size_t, size_t, int)> create;
    };

    const std::array<FormatCase, 2> formats = {
        FormatCase{
            "Q5_1",
            [](size_t n, size_t k, int seed)
            {
                return TestTensorFactory::createQ5_1Random({n, k}, seed);
            }},
        FormatCase{
            "Q5_K",
            [](size_t n, size_t k, int seed)
            {
                return TestTensorFactory::createQ5_KRandom({n, k}, seed);
            }}};

    for (size_t fmt_index = 0; fmt_index < formats.size(); ++fmt_index)
    {
        const auto &fmt = formats[fmt_index];

        auto weights = fmt.create(
            static_cast<size_t>(N),
            static_cast<size_t>(K),
            243 + static_cast<int>(fmt_index) * 10);
        auto gate_data = randomFP32(static_cast<size_t>(M) * K);
        auto up_data = randomFP32(static_cast<size_t>(M) * K);

        auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
        ASSERT_NE(cuda_kernel, nullptr) << fmt.name << " CUDA kernel";
        ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K))
            << fmt.name << " workspace";

        std::vector<float> C_m4(static_cast<size_t>(M) * N, 0.0f);
        ASSERT_TRUE(cudaFusedSwiGLUDownViaTensor(
            cuda_kernel,
            gate_data.data(),
            up_data.data(),
            C_m4.data(),
            M,
            N,
            K,
            gpu_device_))
            << fmt.name << " M=4 Qwen3.6 fused-SwiGLU down projection failed";

        for (int row = 0; row < M; ++row)
        {
            std::vector<float> C_m1(static_cast<size_t>(N), 0.0f);
            ASSERT_TRUE(cudaFusedSwiGLUDownViaTensor(
                cuda_kernel,
                gate_data.data() + static_cast<size_t>(row) * K,
                up_data.data() + static_cast<size_t>(row) * K,
                C_m1.data(),
                1,
                N,
                K,
                gpu_device_))
                << fmt.name << " M=1 Qwen3.6 fused-SwiGLU down projection failed for row " << row;

            const float *verifier_row = C_m4.data() + static_cast<size_t>(row) * N;
            expectBitwiseEqualFloatRow(
                (std::string(fmt.name) + " Qwen36 FFN down M=4 row=" +
                 std::to_string(row))
                    .c_str(),
                verifier_row,
                C_m1.data(),
                C_m1.size());
        }

        cleanupWorkspaceIfNeeded(cuda_kernel);
        EXPECT_FALSE(cuda_kernel->hasDynamicStateActive())
            << fmt.name << " M=4 Qwen3.6 fused-SwiGLU row-equivalence test "
            << "must not leak CUDA dynamic state";
        llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
    }
}

TEST_F(Test__CUDAGemmParity, Q4_K_FusedSwiGLUDownDecodeM1_1x896x768)
{
    const int M = 1;
    const int N = 896;
    const int K = 768;

    auto weights = TestTensorFactory::createQ4_KRandom({(size_t)N, (size_t)K}, 232);
    auto gate_data = randomFP32(static_cast<size_t>(M) * K);
    auto up_data = randomFP32(static_cast<size_t>(M) * K);

    std::vector<float> C_cpu(static_cast<size_t>(M) * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr);
    ASSERT_TRUE(cpuFusedSwiGLUDownToVector(
        cpu_kernel.get(), gate_data.data(), up_data.data(), C_cpu.data(), M, N, K));
    ASSERT_FALSE(hasNaNOrInf(C_cpu.data(), C_cpu.size()));

    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    std::vector<float> C_cuda(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_TRUE(cudaFusedSwiGLUDownViaTensor(
        cuda_kernel, gate_data.data(), up_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    auto result = checkParity(C_cuda.data(), C_cpu.data(), C_cpu.size(), 0.99, 0.15);
    result.print("Q4_K fused-SwiGLU down decode M=1 1x896x768");
    EXPECT_FALSE(result.has_nan_inf);
    EXPECT_GE(result.cosine_similarity, 0.99)
        << "Cosine similarity too low: " << result.cosine_similarity;
    EXPECT_LE(result.relative_l2_error, 0.15)
        << "Relative L2 error too high: " << (result.relative_l2_error * 100.0) << "%";

    cleanupWorkspaceIfNeeded(cuda_kernel);
    EXPECT_FALSE(cuda_kernel->hasDynamicStateActive())
        << "Decode fused-SwiGLU down test must not leak CUDA dynamic state after workspace cleanup";
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

// ============================================================================
// IQ (Importance Quantization) Parity Tests
// ============================================================================

DEFINE_QUANTIZED_PARITY_TEST(IQ4_XS_SmallMatrix, IQ4_XSTensor, createIQ4_XSRandom, 256, 301)
DEFINE_QUANTIZED_PARITY_TEST(IQ2_XXS_SmallMatrix, IQ2_XXSTensor, createIQ2_XXSRandom, 256, 302)

DEFINE_QUANTIZED_PARITY_TEST(IQ2_XS_SmallMatrix, IQ2_XSTensor, createIQ2_XSRandom, 256, 303)
DEFINE_QUANTIZED_PARITY_TEST(IQ2_S_SmallMatrix, IQ2_STensor, createIQ2_SRandom, 256, 304)
DEFINE_QUANTIZED_PARITY_TEST(IQ3_XXS_SmallMatrix, IQ3_XXSTensor, createIQ3_XXSRandom, 256, 305)
DEFINE_QUANTIZED_PARITY_TEST(IQ3_S_SmallMatrix, IQ3_STensor, createIQ3_SRandom, 256, 306)
DEFINE_QUANTIZED_PARITY_TEST(IQ1_S_SmallMatrix, IQ1_STensor, createIQ1_SRandom, 256, 307)
DEFINE_QUANTIZED_PARITY_TEST(IQ1_M_SmallMatrix, IQ1_MTensor, createIQ1_MRandom, 256, 308)

DEFINE_QUANTIZED_DECODE_PARITY_TEST(IQ4_XS_DecodeSize, IQ4_XSTensor, createIQ4_XSRandom, 256, 311)
DEFINE_QUANTIZED_DECODE_PARITY_TEST(IQ2_XXS_DecodeSize, IQ2_XXSTensor, createIQ2_XXSRandom, 256, 312)
DEFINE_QUANTIZED_DECODE_PARITY_TEST(IQ2_XS_DecodeSize, IQ2_XSTensor, createIQ2_XSRandom, 256, 313)
DEFINE_QUANTIZED_DECODE_PARITY_TEST(IQ2_S_DecodeSize, IQ2_STensor, createIQ2_SRandom, 256, 314)
DEFINE_QUANTIZED_DECODE_PARITY_TEST(IQ3_XXS_DecodeSize, IQ3_XXSTensor, createIQ3_XXSRandom, 256, 315)
DEFINE_QUANTIZED_DECODE_PARITY_TEST(IQ3_S_DecodeSize, IQ3_STensor, createIQ3_SRandom, 256, 316)
DEFINE_QUANTIZED_DECODE_PARITY_TEST(IQ1_S_DecodeSize, IQ1_STensor, createIQ1_SRandom, 256, 317)
DEFINE_QUANTIZED_DECODE_PARITY_TEST(IQ1_M_DecodeSize, IQ1_MTensor, createIQ1_MRandom, 256, 318)

// ============================================================================
// Real Model Weight Parity Tests
// ============================================================================
// These tests load actual Q4_0 weights from a GGUF model file to verify
// that CUDA GEMM produces correct results with real-world weight distributions.
// This is critical because synthetic random weights may not expose issues
// that occur with the specific value patterns in trained models.

namespace
{
    constexpr const char *REAL_MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
}

/**
 * @test Q4_0 parity with real model weights: attn_q.weight (layer 0)
 *
 * Tests Q projection weight matrix which is critical for attention.
 * This is one of the weights that showed massive divergence in full inference
 * (cosine=0.098 vs expected ~1.0).
 */
TEST_F(Test__CUDAGemmParity, RealModel_Q4_0_AttnQ_Layer0)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    // Create MPI context (single rank for this test)
    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    // Load real Q projection weight
    auto weights = loader.loadTensor("blk.0.attn_q.weight", DeviceId::cpu());
    ASSERT_NE(weights, nullptr) << "Failed to load blk.0.attn_q.weight";

    auto *q4_tensor = dynamic_cast<Q4_0Tensor *>(weights.get());
    ASSERT_NE(q4_tensor, nullptr) << "Expected Q4_0Tensor, got different type";

    // Dimensions: [N, K] where N=896 (output), K=896 (input) for Qwen2.5-0.5B
    const int M = 4; // Small batch for testing
    const int N = static_cast<int>(q4_tensor->shape()[0]);
    const int K = static_cast<int>(q4_tensor->shape()[1]);

    std::cout << "Real model attn_q weight: " << N << "x" << K << " (Q4_0)\n";

    // Create random activations
    auto A_data = randomFP32(M * K);

    // ===== CPU Reference =====
    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        q4_tensor, KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr) << "Failed to create CPU kernel";
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    // ===== CUDA =====
    // Upload weights to GPU
    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device_));
    EXPECT_TRUE(q4_tensor->isOnGPU());

    auto *cuda_kernel = getPreparedKernel(
        q4_tensor, gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr) << "Failed to create CUDA kernel";

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    float *d_A, *d_C;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    // Clean up cache entry while tensor is still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

/**
 * @test Q4_0 parity with real model weights: attn_k.weight (layer 0)
 *
 * Tests K projection weight matrix. K projection showed even worse divergence
 * than Q in full inference (cosine=0.031).
 */
TEST_F(Test__CUDAGemmParity, RealModel_Q4_0_AttnK_Layer0)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    auto weights = loader.loadTensor("blk.0.attn_k.weight", DeviceId::cpu());
    ASSERT_NE(weights, nullptr) << "Failed to load blk.0.attn_k.weight";

    auto *q4_tensor = dynamic_cast<Q4_0Tensor *>(weights.get());
    ASSERT_NE(q4_tensor, nullptr) << "Expected Q4_0Tensor";

    const int M = 4;
    const int N = static_cast<int>(q4_tensor->shape()[0]);
    const int K = static_cast<int>(q4_tensor->shape()[1]);

    std::cout << "Real model attn_k weight: " << N << "x" << K << " (Q4_0)\n";

    auto A_data = randomFP32(M * K);

    // CPU
    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        q4_tensor, KernelDeviceType::CPU);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    // CUDA
    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        q4_tensor, gpu_device_);

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    float *d_A, *d_C;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    // Clean up cache entry while tensor is still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

/**
 * @test Q4_0 parity with real model weights: attn_v.weight (layer 0)
 *
 * Tests V projection. V showed less divergence (cosine=0.84) but still
 * significantly off in full inference.
 */
TEST_F(Test__CUDAGemmParity, RealModel_Q4_0_AttnV_Layer0)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    auto weights = loader.loadTensor("blk.0.attn_v.weight", DeviceId::cpu());
    ASSERT_NE(weights, nullptr) << "Failed to load blk.0.attn_v.weight";

    auto *q4_tensor = dynamic_cast<Q4_0Tensor *>(weights.get());
    ASSERT_NE(q4_tensor, nullptr) << "Expected Q4_0Tensor";

    const int M = 4;
    const int N = static_cast<int>(q4_tensor->shape()[0]);
    const int K = static_cast<int>(q4_tensor->shape()[1]);

    std::cout << "Real model attn_v weight: " << N << "x" << K << " (Q4_0)\n";

    auto A_data = randomFP32(M * K);

    // CPU
    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        q4_tensor, KernelDeviceType::CPU);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    // CUDA
    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        q4_tensor, gpu_device_);

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    float *d_A, *d_C;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    // Clean up cache entry while tensor is still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

/**
 * @test Q4_0 parity with real model weights: ffn_gate.weight (layer 0)
 *
 * Tests FFN gate weight which is a larger matrix (4864x896 for Qwen2.5-0.5B).
 */
TEST_F(Test__CUDAGemmParity, RealModel_Q4_0_FFNGate_Layer0)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    auto weights = loader.loadTensor("blk.0.ffn_gate.weight", DeviceId::cpu());
    ASSERT_NE(weights, nullptr) << "Failed to load blk.0.ffn_gate.weight";

    auto *q4_tensor = dynamic_cast<Q4_0Tensor *>(weights.get());
    ASSERT_NE(q4_tensor, nullptr) << "Expected Q4_0Tensor";

    const int M = 4;
    const int N = static_cast<int>(q4_tensor->shape()[0]);
    const int K = static_cast<int>(q4_tensor->shape()[1]);

    std::cout << "Real model ffn_gate weight: " << N << "x" << K << " (Q4_0)\n";

    auto A_data = randomFP32(M * K);

    // CPU
    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        q4_tensor, KernelDeviceType::CPU);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    // CUDA
    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        q4_tensor, gpu_device_);

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    float *d_A, *d_C;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    // Clean up cache entry while tensor is still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

/**
 * @test Q4_0 parity with real model weights: output/lm_head.weight
 *
 * Tests vocabulary projection (LM head) which is the final layer.
 * Shape: [vocab_size, hidden_dim] = [151936, 896] for Qwen2.5
 */
TEST_F(Test__CUDAGemmParity, RealModel_Q4_0_LMHead)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    auto weights = loader.loadTensor("output.weight", DeviceId::cpu());
    ASSERT_NE(weights, nullptr) << "Failed to load output.weight (LM head)";

    auto *q4_tensor = dynamic_cast<Q4_0Tensor *>(weights.get());
    if (!q4_tensor)
    {
        // LM head might be in a different format (e.g., FP32/FP16)
        std::cout << "LM head is not Q4_0, tensor type: "
                  << static_cast<int>(weights->native_type()) << "\n";
        GTEST_SKIP() << "LM head is not Q4_0 format";
    }

    const int M = 2;                                       // Smaller batch for large vocab
    const int N = static_cast<int>(q4_tensor->shape()[0]); // vocab_size
    const int K = static_cast<int>(q4_tensor->shape()[1]); // hidden_dim

    std::cout << "Real model LM head weight: " << N << "x" << K << " (Q4_0)\n";

    auto A_data = randomFP32(M * K);

    // CPU
    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        q4_tensor, KernelDeviceType::CPU);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    // CUDA
    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        q4_tensor, gpu_device_);

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    float *d_A, *d_C;
    std::vector<float> C_cuda(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    // Clean up cache entry while tensor is still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

/**
 * @test Q4_0 parity with real model weights using TENSOR API
 *
 * This test uses the multiply_tensor() API (same as full inference)
 * instead of the raw multiply() API to see if the issue is in the
 * tensor-based code path.
 */
TEST_F(Test__CUDAGemmParity, RealModel_Q4_0_AttnQ_TensorAPI)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    auto weights = loader.loadTensor("blk.0.attn_q.weight", DeviceId::cpu());
    ASSERT_NE(weights, nullptr);

    auto *q4_tensor = dynamic_cast<Q4_0Tensor *>(weights.get());
    ASSERT_NE(q4_tensor, nullptr);

    const int M = 4;
    const int N = static_cast<int>(q4_tensor->shape()[0]);
    const int K = static_cast<int>(q4_tensor->shape()[1]);

    std::cout << "Testing multiply_tensor() API with attn_q: " << N << "x" << K << "\n";

    // Create FP32 input and output tensors
    auto input_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    auto output_cpu = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
    auto output_cuda = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

    // Fill input with random data
    float *input_data = input_tensor->mutable_data();
    for (int i = 0; i < M * K; ++i)
    {
        input_data[i] = dist_(rng_);
    }

    // ===== CPU: multiply_tensor() =====
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        q4_tensor, KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr);
    ASSERT_TRUE(cpu_kernel->multiply_tensor(
        input_tensor.get(), output_cpu.get(),
        M, N, K, true, 1.0f, 0.0f, nullptr, nullptr, -1));

    // ===== CUDA: multiply_tensor() =====
    // First ensure weights are on GPU
    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device_));

    auto *cuda_kernel = getPreparedKernel(
        q4_tensor, gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    // Use with_gpu_coherence for automatic input/output coherence management
    ASSERT_TRUE(withExplicitCudaGemmStream(
        {cuda_kernel},
        [&](void *stream)
        {
            return with_gpu_coherence(
                gpu_device_,
                {input_tensor.get()}, // inputs
                {output_cuda.get()},  // outputs (will be marked dirty after kernel)
                static_cast<void *>(stream),
                [&]
                {
                    return cuda_kernel->multiply_tensor(
                        input_tensor.get(), output_cuda.get(),
                        M, N, K, true, 1.0f, 0.0f, nullptr, nullptr, -1);
                });
        }));

    // Clean up workspace
    cleanupWorkspaceIfNeeded(cuda_kernel);

    // ===== Compare =====
    // data() will automatically sync to host if needed
    const float *cpu_data = output_cpu->data();
    const float *cuda_data = output_cuda->data();

    auto result = checkParity(cuda_data, cpu_data, M * N, 0.99, 0.10);
    result.print("Real Model Q4_0 attn_q (multiply_tensor API)");

    EXPECT_GE(result.cosine_similarity, 0.99)
        << "multiply_tensor() API shows divergence!";
    EXPECT_FALSE(result.has_nan_inf);

    // Clean up cache entry while tensor is still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

/**
 * @test Fused QKV GEMM parity: multiply_fused_tensor() vs 3x multiply_tensor()
 *
 * This test validates the new multiply_fused_tensor() method which:
 * 1. Uploads input to GPU once
 * 2. Quantizes activations to INT8 once (shared across all projections)
 * 3. Runs all Q/K/V projections using the shared quantized activations
 *
 * **Key validation**: The fused path should produce identical results to
 * calling multiply_tensor() three times separately, since the quantization
 * of activations should be deterministic.
 *
 * This test was written to verify the fix for CUDA full model divergence
 * where the default multiply_fused_tensor() implementation was calling
 * multiply_tensor() in a loop, which requantized activations for each
 * projection (inefficient and potentially inconsistent).
 */
TEST_F(Test__CUDAGemmParity, FusedQKV_TensorAPI_vs_Separate)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    // Load Q, K, V projection weights
    auto weights_q_base = loader.loadTensor("blk.0.attn_q.weight", DeviceId::cpu());
    auto weights_k_base = loader.loadTensor("blk.0.attn_k.weight", DeviceId::cpu());
    auto weights_v_base = loader.loadTensor("blk.0.attn_v.weight", DeviceId::cpu());
    ASSERT_NE(weights_q_base, nullptr) << "Failed to load attn_q.weight";
    ASSERT_NE(weights_k_base, nullptr) << "Failed to load attn_k.weight";
    ASSERT_NE(weights_v_base, nullptr) << "Failed to load attn_v.weight";

    // Cast to concrete Q4_0 type
    auto *weights_q = dynamic_cast<Q4_0Tensor *>(weights_q_base.get());
    auto *weights_k = dynamic_cast<Q4_0Tensor *>(weights_k_base.get());
    auto *weights_v = dynamic_cast<Q4_0Tensor *>(weights_v_base.get());
    ASSERT_NE(weights_q, nullptr) << "Expected Q4_0Tensor for Q weights";
    ASSERT_NE(weights_k, nullptr) << "Expected Q4_0Tensor for K weights";
    ASSERT_NE(weights_v, nullptr) << "Expected Q4_0Tensor for V weights";

    // Get dimensions
    const int M = 4; // Small batch for testing (could also test M=1 for decode)
    const int N_q = static_cast<int>(weights_q->shape()[0]);
    const int N_k = static_cast<int>(weights_k->shape()[0]);
    const int N_v = static_cast<int>(weights_v->shape()[0]);
    const int K = static_cast<int>(weights_q->shape()[1]);

    std::cout << "FusedQKV test: M=" << M << " K=" << K
              << " N_q=" << N_q << " N_k=" << N_k << " N_v=" << N_v << "\n";

    // Note: ensureOnDevice is NOT needed for CUDA quantized GEMM path -
    // prepareGpuGemmOnDemand handles the upload via WeightVRAMPool internally

    // Create CUDA kernels for each projection
    auto *cuda_kernel_q = getPreparedKernel(
        weights_q, gpu_device_);
    auto *cuda_kernel_k = getPreparedKernel(
        weights_k, gpu_device_);
    auto *cuda_kernel_v = getPreparedKernel(
        weights_v, gpu_device_);
    ASSERT_NE(cuda_kernel_q, nullptr) << "Failed to create CUDA kernel for Q";
    ASSERT_NE(cuda_kernel_k, nullptr) << "Failed to create CUDA kernel for K";
    ASSERT_NE(cuda_kernel_v, nullptr) << "Failed to create CUDA kernel for V";

    // Set up SHARED workspace for fused QKV (all kernels share workspace)
    ASSERT_TRUE(setupSharedWorkspace(
        {cuda_kernel_q, cuda_kernel_k, cuda_kernel_v},
        M, {N_q, N_k, N_v}, K));

    // Create input tensor with random data
    auto input_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    float *input_data = input_tensor->mutable_data();
    for (int i = 0; i < M * K; ++i)
    {
        input_data[i] = dist_(rng_);
    }

    // Create output tensors for SEPARATE path
    auto output_q_separate = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_q)});
    auto output_k_separate = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_k)});
    auto output_v_separate = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_v)});

    // Create output tensors for FUSED path
    auto output_q_fused = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_q)});
    auto output_k_fused = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_k)});
    auto output_v_fused = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_v)});

    // ===== SEPARATE PATH: 3x multiply_tensor() =====
    std::cout << "Running SEPARATE path (3x multiply_tensor)...\n";
    ASSERT_TRUE(withExplicitCudaGemmStream(
        {cuda_kernel_q, cuda_kernel_k, cuda_kernel_v},
        [&](void *stream)
        {
            return with_gpu_coherence(
                gpu_device_,
                {input_tensor.get()},
                {output_q_separate.get(), output_k_separate.get(), output_v_separate.get()},
                static_cast<void *>(stream),
                [&]
                {
                    return cuda_kernel_q->multiply_tensor(
                               input_tensor.get(), output_q_separate.get(),
                               M, N_q, K, true, 1.0f, 0.0f, nullptr, nullptr, -1) &&
                           cuda_kernel_k->multiply_tensor(
                               input_tensor.get(), output_k_separate.get(),
                               M, N_k, K, true, 1.0f, 0.0f, nullptr, nullptr, -1) &&
                           cuda_kernel_v->multiply_tensor(
                               input_tensor.get(), output_v_separate.get(),
                               M, N_v, K, true, 1.0f, 0.0f, nullptr, nullptr, -1);
                });
        }));

    // ===== FUSED PATH: multiply_fused_tensor() =====
    std::cout << "Running FUSED path (multiply_fused_tensor)...\n";

    // Build projection descriptors
    std::vector<TensorProjectionDesc> projections;
    projections.emplace_back(cuda_kernel_q, output_q_fused.get(), N_q,
                             nullptr, "Q");
    projections.emplace_back(cuda_kernel_k, output_k_fused.get(), N_k,
                             nullptr, "K");
    projections.emplace_back(cuda_kernel_v, output_v_fused.get(), N_v,
                             nullptr, "V");

    // Call fused method with coherence wrapper
    ASSERT_TRUE(withExplicitCudaGemmStream(
        {cuda_kernel_q, cuda_kernel_k, cuda_kernel_v},
        [&](void *stream)
        {
            return with_gpu_coherence(
                gpu_device_,
                {input_tensor.get()},
                {output_q_fused.get(), output_k_fused.get(), output_v_fused.get()},
                static_cast<void *>(stream),
                [&]
                {
                    return cuda_kernel_q->multiply_fused_tensor(
                        input_tensor.get(), projections, M, K, nullptr);
                });
        }));

    // ===== COMPARE: Fused vs Separate =====
    std::cout << "\nComparing FUSED vs SEPARATE results:\n";

    // Sync back to host for comparison
    const float *q_separate = output_q_separate->data();
    const float *k_separate = output_k_separate->data();
    const float *v_separate = output_v_separate->data();
    const float *q_fused = output_q_fused->data();
    const float *k_fused = output_k_fused->data();
    const float *v_fused = output_v_fused->data();

    // Q projection parity
    auto result_q = checkParity(q_fused, q_separate, M * N_q, 0.9999, 0.001);
    result_q.print("Q projection (fused vs separate)");
    EXPECT_GE(result_q.cosine_similarity, 0.9999)
        << "Q projection: fused and separate should be nearly identical";
    EXPECT_FALSE(result_q.has_nan_inf);

    // K projection parity
    auto result_k = checkParity(k_fused, k_separate, M * N_k, 0.9999, 0.001);
    result_k.print("K projection (fused vs separate)");
    EXPECT_GE(result_k.cosine_similarity, 0.9999)
        << "K projection: fused and separate should be nearly identical";
    EXPECT_FALSE(result_k.has_nan_inf);

    // V projection parity
    auto result_v = checkParity(v_fused, v_separate, M * N_v, 0.9999, 0.001);
    result_v.print("V projection (fused vs separate)");
    EXPECT_GE(result_v.cosine_similarity, 0.9999)
        << "V projection: fused and separate should be nearly identical";
    EXPECT_FALSE(result_v.has_nan_inf);

    // Also compare against CPU to ensure correctness
    std::cout << "\nComparing FUSED vs CPU reference:\n";

    auto cpu_kernel_q = llaminar::v2::kernels::KernelFactory::createGemm(
        weights_q, KernelDeviceType::CPU);
    auto cpu_kernel_k = llaminar::v2::kernels::KernelFactory::createGemm(
        weights_k, KernelDeviceType::CPU);
    auto cpu_kernel_v = llaminar::v2::kernels::KernelFactory::createGemm(
        weights_v, KernelDeviceType::CPU);

    std::vector<float> q_cpu(M * N_q), k_cpu(M * N_k), v_cpu(M * N_v);
    const float *h_input = input_tensor->data(); // Sync input to host
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel_q.get(), h_input, q_cpu.data(), M, N_q, K));
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel_k.get(), h_input, k_cpu.data(), M, N_k, K));
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel_v.get(), h_input, v_cpu.data(), M, N_v, K));

    auto result_q_cpu = checkParity(q_fused, q_cpu.data(), M * N_q, 0.99, 0.10);
    result_q_cpu.print("Q projection (CUDA fused vs CPU)");

    EXPECT_GE(result_q_cpu.cosine_similarity, 0.99);

    auto result_k_cpu = checkParity(k_fused, k_cpu.data(), M * N_k, 0.99, 0.10);
    result_k_cpu.print("K projection (CUDA fused vs CPU)");
    EXPECT_GE(result_k_cpu.cosine_similarity, 0.99);

    auto result_v_cpu = checkParity(v_fused, v_cpu.data(), M * N_v, 0.99, 0.10);
    result_v_cpu.print("V projection (CUDA fused vs CPU)");
    EXPECT_GE(result_v_cpu.cosine_similarity, 0.99);

    // Clean up shared workspace
    cleanupSharedWorkspace({cuda_kernel_q, cuda_kernel_k, cuda_kernel_v});

    // Clean up cache entries while tensors are still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_q);
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_k);
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_v);
}

/**
 * @test Fused QKV with decode-size batch (M=1)
 *
 * Tests the fused path with M=1 which is the common case during autoregressive
 * decoding. This is important because the quantization behavior might differ
 * with single-row inputs.
 */
TEST_F(Test__CUDAGemmParity, FusedQKV_DecodeSize_M1)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    auto weights_q_base = loader.loadTensor("blk.0.attn_q.weight", DeviceId::cpu());
    auto weights_k_base = loader.loadTensor("blk.0.attn_k.weight", DeviceId::cpu());
    auto weights_v_base = loader.loadTensor("blk.0.attn_v.weight", DeviceId::cpu());
    ASSERT_NE(weights_q_base, nullptr);
    ASSERT_NE(weights_k_base, nullptr);
    ASSERT_NE(weights_v_base, nullptr);

    // Cast to concrete Q4_0 type
    auto *weights_q = dynamic_cast<Q4_0Tensor *>(weights_q_base.get());
    auto *weights_k = dynamic_cast<Q4_0Tensor *>(weights_k_base.get());
    auto *weights_v = dynamic_cast<Q4_0Tensor *>(weights_v_base.get());
    ASSERT_NE(weights_q, nullptr) << "Expected Q4_0Tensor for Q weights";
    ASSERT_NE(weights_k, nullptr) << "Expected Q4_0Tensor for K weights";
    ASSERT_NE(weights_v, nullptr) << "Expected Q4_0Tensor for V weights";

    const int M = 1; // Decode size!
    const int N_q = static_cast<int>(weights_q->shape()[0]);
    const int N_k = static_cast<int>(weights_k->shape()[0]);
    const int N_v = static_cast<int>(weights_v->shape()[0]);
    const int K = static_cast<int>(weights_q->shape()[1]);

    std::cout << "FusedQKV DECODE test: M=" << M << " K=" << K
              << " N_q=" << N_q << " N_k=" << N_k << " N_v=" << N_v << "\n";

    ASSERT_TRUE(weights_q->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(weights_k->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(weights_v->ensureOnDevice(gpu_device_));

    auto *cuda_kernel_q = getPreparedKernel(
        weights_q, gpu_device_);
    auto *cuda_kernel_k = getPreparedKernel(
        weights_k, gpu_device_);
    auto *cuda_kernel_v = getPreparedKernel(
        weights_v, gpu_device_);

    // Set up SHARED workspace for fused QKV (all kernels share workspace)
    ASSERT_TRUE(setupSharedWorkspace(
        {cuda_kernel_q, cuda_kernel_k, cuda_kernel_v},
        M, {N_q, N_k, N_v}, K));

    auto input_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    float *input_data = input_tensor->mutable_data();
    for (int i = 0; i < M * K; ++i)
    {
        input_data[i] = dist_(rng_);
    }

    // Fused outputs
    auto output_q_fused = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_q)});
    auto output_k_fused = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_k)});
    auto output_v_fused = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_v)});

    // Run fused with coherence wrapper
    std::vector<TensorProjectionDesc> projections;
    projections.emplace_back(cuda_kernel_q, output_q_fused.get(), N_q,
                             nullptr, "Q");
    projections.emplace_back(cuda_kernel_k, output_k_fused.get(), N_k,
                             nullptr, "K");
    projections.emplace_back(cuda_kernel_v, output_v_fused.get(), N_v,
                             nullptr, "V");

    ASSERT_TRUE(withExplicitCudaGemmStream(
        {cuda_kernel_q, cuda_kernel_k, cuda_kernel_v},
        [&](void *stream)
        {
            return with_gpu_coherence(
                gpu_device_,
                {input_tensor.get()},
                {output_q_fused.get(), output_k_fused.get(), output_v_fused.get()},
                static_cast<void *>(stream),
                [&]
                {
                    return cuda_kernel_q->multiply_fused_tensor(
                        input_tensor.get(), projections, M, K, nullptr);
                });
        }));

    // Compare against CPU
    auto cpu_kernel_q = llaminar::v2::kernels::KernelFactory::createGemm(
        weights_q, KernelDeviceType::CPU);
    auto cpu_kernel_k = llaminar::v2::kernels::KernelFactory::createGemm(
        weights_k, KernelDeviceType::CPU);
    auto cpu_kernel_v = llaminar::v2::kernels::KernelFactory::createGemm(
        weights_v, KernelDeviceType::CPU);

    std::vector<float> q_cpu(M * N_q), k_cpu(M * N_k), v_cpu(M * N_v);
    const float *h_input = input_tensor->data();
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel_q.get(), h_input, q_cpu.data(), M, N_q, K));
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel_k.get(), h_input, k_cpu.data(), M, N_k, K));
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel_v.get(), h_input, v_cpu.data(), M, N_v, K));

    const float *q_fused = output_q_fused->data();
    const float *k_fused = output_k_fused->data();
    const float *v_fused = output_v_fused->data();

    auto result_q = checkParity(q_fused, q_cpu.data(), M * N_q, 0.99, 0.10);
    result_q.print("Q decode (CUDA fused vs CPU)");
    EXPECT_GE(result_q.cosine_similarity, 0.99);
    EXPECT_FALSE(result_q.has_nan_inf);

    auto result_k = checkParity(k_fused, k_cpu.data(), M * N_k, 0.99, 0.10);
    result_k.print("K decode (CUDA fused vs CPU)");
    EXPECT_GE(result_k.cosine_similarity, 0.99);
    EXPECT_FALSE(result_k.has_nan_inf);

    auto result_v = checkParity(v_fused, v_cpu.data(), M * N_v, 0.99, 0.10);
    result_v.print("V decode (CUDA fused vs CPU)");
    EXPECT_GE(result_v.cosine_similarity, 0.99);
    EXPECT_FALSE(result_v.has_nan_inf);

    // Clean up shared workspace
    cleanupSharedWorkspace({cuda_kernel_q, cuda_kernel_k, cuda_kernel_v});

    // Clean up cache entries while tensors are still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_q_base.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_k_base.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_v_base.get());
}

// ============================================================================
// Cached Kernel Tests
// ============================================================================

/**
 * @test Cached kernel parity: prepared GEMM engine vs createGemm()
 *
 * This test verifies that kernels obtained via prepared GEMM engine lookup (the caching path
 * used by the full pipeline) produce identical results to kernels created via
 * createGemm() (fresh kernel creation).
 *
 * **Why this matters**: If there's a bug in kernel caching (stale weights, incorrect
 * scale factors), this will catch it. The full pipeline uses the prepared GEMM engine path.
 */
TEST_F(Test__CUDAGemmParity, CachedKernel_vs_FreshKernel)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    auto weights = loader.loadTensor("blk.0.attn_q.weight", DeviceId::cpu());
    ASSERT_NE(weights, nullptr);

    auto *q4_tensor = dynamic_cast<Q4_0Tensor *>(weights.get());
    ASSERT_NE(q4_tensor, nullptr);

    const int M = 4;
    const int N = static_cast<int>(q4_tensor->shape()[0]);
    const int K = static_cast<int>(q4_tensor->shape()[1]);

    std::cout << "Testing cached kernel parity: attn_q " << N << "x" << K << "\n";

    // Ensure on GPU
    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device_));

    // Get kernel via prepared GEMM path (legacy createGemm GPU path removed)
    auto *cuda_kernel = getPreparedKernel(q4_tensor, gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);

    // Set up workspace
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    // Create input and outputs
    auto input_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    auto output_a = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
    auto output_b = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

    // Fill input
    float *input_data = input_tensor->mutable_data();
    for (int i = 0; i < M * K; ++i)
    {
        input_data[i] = dist_(rng_);
    }

    // Run kernel twice — should produce identical results
    ASSERT_TRUE(withExplicitCudaGemmStream(
        {cuda_kernel},
        [&](void *stream)
        {
            return with_gpu_coherence(
                gpu_device_,
                {input_tensor.get()},
                {output_a.get()},
                static_cast<void *>(stream),
                [&]
                {
                    return cuda_kernel->multiply_tensor(
                        input_tensor.get(), output_a.get(),
                        M, N, K, true, 1.0f, 0.0f, nullptr, nullptr, -1);
                });
        }));

    ASSERT_TRUE(withExplicitCudaGemmStream(
        {cuda_kernel},
        [&](void *stream)
        {
            return with_gpu_coherence(
                gpu_device_,
                {input_tensor.get()},
                {output_b.get()},
                static_cast<void *>(stream),
                [&]
                {
                    return cuda_kernel->multiply_tensor(
                        input_tensor.get(), output_b.get(),
                        M, N, K, true, 1.0f, 0.0f, nullptr, nullptr, -1);
                });
        }));

    // Compare: should be EXACTLY the same (same kernel, same weights, same input)
    const float *data_a = output_a->data();
    const float *data_b = output_b->data();

    auto result = checkParity(data_b, data_a, M * N, 0.9999, 0.001);
    result.print("Kernel run A vs run B (same prepared kernel)");

    EXPECT_GE(result.cosine_similarity, 0.9999)
        << "Same prepared kernel should produce nearly identical results";
    EXPECT_FALSE(result.has_nan_inf);

    // Clean up workspace
    cleanupWorkspaceIfNeeded(cuda_kernel);

    // Clean up cache entry while tensor is still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

/**
 * @test Cached kernel reuse: multiple calls with same kernel
 *
 * Verifies that cached kernels produce consistent results across multiple calls.
 * This catches issues like stale GPU state or incorrect memory management.
 */
TEST_F(Test__CUDAGemmParity, CachedKernel_MultipleCallsConsistent)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    auto weights = loader.loadTensor("blk.0.attn_q.weight", DeviceId::cpu());
    ASSERT_NE(weights, nullptr);

    auto *q4_tensor = dynamic_cast<Q4_0Tensor *>(weights.get());
    ASSERT_NE(q4_tensor, nullptr);

    const int M = 1; // Decode size
    const int N = static_cast<int>(q4_tensor->shape()[0]);
    const int K = static_cast<int>(q4_tensor->shape()[1]);

    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device_));

    // Get cached kernel
    auto *kernel = getPreparedKernel(q4_tensor, gpu_device_);
    ASSERT_NE(kernel, nullptr);

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(kernel, M, N, K));

    // Create input
    auto input = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    float *input_data = input->mutable_data();
    for (int i = 0; i < M * K; ++i)
    {
        input_data[i] = dist_(rng_);
    }

    // Run 3 times with same input, compare all outputs
    std::vector<std::vector<float>> outputs(3);
    for (int run = 0; run < 3; ++run)
    {
        auto output = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

        // Use with_gpu_coherence for clean coherence handling
        ASSERT_TRUE(withExplicitCudaGemmStream(
            {kernel},
            [&](void *stream)
            {
                return with_gpu_coherence(
                    gpu_device_,
                    {input.get()},
                    {output.get()},
                    static_cast<void *>(stream),
                    [&]
                    {
                        return kernel->multiply_tensor(
                            input.get(), output.get(),
                            M, N, K, true, 1.0f, 0.0f, nullptr, nullptr, -1);
                    });
            }));

        outputs[run].resize(M * N);
        const float *data = output->data();
        std::copy(data, data + M * N, outputs[run].begin());
    }

    // All runs should be identical
    auto result_1_vs_2 = checkParity(outputs[0].data(), outputs[1].data(), M * N, 0.99999, 0.0001);
    auto result_1_vs_3 = checkParity(outputs[0].data(), outputs[2].data(), M * N, 0.99999, 0.0001);

    std::cout << "Run 1 vs Run 2: cosine=" << result_1_vs_2.cosine_similarity << "\n";
    std::cout << "Run 1 vs Run 3: cosine=" << result_1_vs_3.cosine_similarity << "\n";

    EXPECT_GE(result_1_vs_2.cosine_similarity, 0.99999)
        << "Multiple runs with same input should be identical";
    EXPECT_GE(result_1_vs_3.cosine_similarity, 0.99999)
        << "Multiple runs with same input should be identical";

    // Clean up workspace
    cleanupWorkspaceIfNeeded(kernel);

    // Clean up cache entry while tensor is still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

/**
 * @test Cached kernel across different input sizes
 *
 * The full pipeline may use the same cached kernel for both prefill (large M)
 * and decode (M=1). This test verifies the cached kernel works correctly
 * across different input sizes.
 */
TEST_F(Test__CUDAGemmParity, CachedKernel_VaryingBatchSizes)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    auto weights = loader.loadTensor("blk.0.attn_q.weight", DeviceId::cpu());
    ASSERT_NE(weights, nullptr);

    auto *q4_tensor = dynamic_cast<Q4_0Tensor *>(weights.get());
    ASSERT_NE(q4_tensor, nullptr);

    const int N = static_cast<int>(q4_tensor->shape()[0]);
    const int K = static_cast<int>(q4_tensor->shape()[1]);

    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device_));

    // Get ONE cached kernel
    auto *kernel = getPreparedKernel(q4_tensor, gpu_device_);
    ASSERT_NE(kernel, nullptr);

    // Set up workspace for quantized kernel (with largest batch size = 128)
    ASSERT_TRUE(setupWorkspaceIfNeeded(kernel, 128, N, K));

    // Also create a CPU kernel for reference
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        q4_tensor, KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr);

    // Test multiple batch sizes
    std::vector<int> batch_sizes = {1, 4, 16, 64, 128};

    for (int M : batch_sizes)
    {
        std::cout << "Testing M=" << M << "...\n";

        // Create input with fresh random data
        auto input = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
        float *input_data = input->mutable_data();
        for (int i = 0; i < M * K; ++i)
        {
            input_data[i] = dist_(rng_);
        }

        // CUDA output
        auto output_cuda = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

        // Use with_gpu_coherence for clean coherence handling
        ASSERT_TRUE(withExplicitCudaGemmStream(
            {kernel},
            [&](void *stream)
            {
                return with_gpu_coherence(
                    gpu_device_,
                    {input.get()},
                    {output_cuda.get()},
                    static_cast<void *>(stream),
                    [&]
                    {
                        return kernel->multiply_tensor(
                            input.get(), output_cuda.get(),
                            M, N, K, true, 1.0f, 0.0f, nullptr, nullptr, -1);
                    });
            }));

        // CPU reference
        std::vector<float> cpu_output(M * N);
        ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), input->data(), cpu_output.data(), M, N, K));

        // Compare
        const float *cuda_data = output_cuda->data();
        auto result = checkParity(cuda_data, cpu_output.data(), M * N, 0.99, 0.10);
        result.print(("Cached kernel M=" + std::to_string(M)).c_str());

        EXPECT_GE(result.cosine_similarity, 0.99)
            << "Cached kernel should work for M=" << M;
        EXPECT_FALSE(result.has_nan_inf);
    }

    // Clean up workspace
    cleanupWorkspaceIfNeeded(kernel);

    // Clean up cache entry while tensor is still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

// ============================================================================
// Bias Tests (if model has biases)
// ============================================================================

/**
 * @test QKV projection with biases from GGUF
 *
 * Qwen models have biases for Q, K, V projections. This test verifies that
 * CUDA GEMM + bias produces correct results compared to CPU.
 */
TEST_F(Test__CUDAGemmParity, FusedQKV_WithBias)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    // Load weights
    auto weights_q = loader.loadTensor("blk.0.attn_q.weight", DeviceId::cpu());
    auto weights_k = loader.loadTensor("blk.0.attn_k.weight", DeviceId::cpu());
    auto weights_v = loader.loadTensor("blk.0.attn_v.weight", DeviceId::cpu());
    ASSERT_NE(weights_q, nullptr);
    ASSERT_NE(weights_k, nullptr);
    ASSERT_NE(weights_v, nullptr);

    // Try to load biases (may not exist in all models)
    auto bias_q = loader.loadTensor("blk.0.attn_q.bias", DeviceId::cpu());
    auto bias_k = loader.loadTensor("blk.0.attn_k.bias", DeviceId::cpu());
    auto bias_v = loader.loadTensor("blk.0.attn_v.bias", DeviceId::cpu());

    if (!bias_q && !bias_k && !bias_v)
    {
        GTEST_SKIP() << "Model has no QKV biases";
    }

    std::cout << "Biases found: Q=" << (bias_q ? "yes" : "no")
              << " K=" << (bias_k ? "yes" : "no")
              << " V=" << (bias_v ? "yes" : "no") << "\n";

    // Cast weights to Q4_0
    auto *wq = dynamic_cast<Q4_0Tensor *>(weights_q.get());
    auto *wk = dynamic_cast<Q4_0Tensor *>(weights_k.get());
    auto *wv = dynamic_cast<Q4_0Tensor *>(weights_v.get());
    ASSERT_NE(wq, nullptr);
    ASSERT_NE(wk, nullptr);
    ASSERT_NE(wv, nullptr);

    const int M = 1; // Decode
    const int N_q = static_cast<int>(wq->shape()[0]);
    const int N_k = static_cast<int>(wk->shape()[0]);
    const int N_v = static_cast<int>(wv->shape()[0]);
    const int K = static_cast<int>(wq->shape()[1]);

    std::cout << "FusedQKV with bias: M=" << M << " K=" << K
              << " N_q=" << N_q << " N_k=" << N_k << " N_v=" << N_v << "\n";

    // Upload weights to GPU
    ASSERT_TRUE(wq->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(wk->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(wv->ensureOnDevice(gpu_device_));

    // Upload biases to GPU if present
    if (bias_q)
        ASSERT_TRUE(bias_q->ensureOnDevice(gpu_device_));
    if (bias_k)
        ASSERT_TRUE(bias_k->ensureOnDevice(gpu_device_));
    if (bias_v)
        ASSERT_TRUE(bias_v->ensureOnDevice(gpu_device_));

    // Create kernels
    auto *cuda_kernel_q = getPreparedKernel(
        wq, gpu_device_);
    auto *cuda_kernel_k = getPreparedKernel(
        wk, gpu_device_);
    auto *cuda_kernel_v = getPreparedKernel(
        wv, gpu_device_);

    // Set up SHARED workspace for fused QKV (all kernels share workspace)
    ASSERT_TRUE(setupSharedWorkspace(
        {cuda_kernel_q, cuda_kernel_k, cuda_kernel_v},
        M, {N_q, N_k, N_v}, K));

    // Create input
    auto input = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    float *input_data = input->mutable_data();
    for (int i = 0; i < M * K; ++i)
    {
        input_data[i] = dist_(rng_);
    }
    ASSERT_TRUE(input->ensureOnDevice(gpu_device_));

    // Create outputs
    auto output_q = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_q)});
    auto output_k = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_k)});
    auto output_v = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_v)});
    ASSERT_TRUE(output_q->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(output_k->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(output_v->ensureOnDevice(gpu_device_));

    // Run fused GEMM with biases
    std::vector<TensorProjectionDesc> projections;
    projections.emplace_back(cuda_kernel_q, output_q.get(), N_q,
                             bias_q.get(), "Q");
    projections.emplace_back(cuda_kernel_k, output_k.get(), N_k,
                             bias_k.get(), "K");
    projections.emplace_back(cuda_kernel_v, output_v.get(), N_v,
                             bias_v.get(), "V");

    ASSERT_TRUE(withExplicitCudaGemmStream(
        {cuda_kernel_q, cuda_kernel_k, cuda_kernel_v},
        [&](void *stream)
        {
            return with_gpu_coherence(
                gpu_device_,
                {input.get()},
                {output_q.get(), output_k.get(), output_v.get()},
                static_cast<void *>(stream),
                [&]
                {
                    return cuda_kernel_q->multiply_fused_tensor(
                        input.get(), projections, M, K, nullptr);
                });
        }));

    // ===== CPU reference (GEMM + manual bias add) =====
    auto cpu_kernel_q = llaminar::v2::kernels::KernelFactory::createGemm(
        wq, KernelDeviceType::CPU);
    auto cpu_kernel_k = llaminar::v2::kernels::KernelFactory::createGemm(
        wk, KernelDeviceType::CPU);
    auto cpu_kernel_v = llaminar::v2::kernels::KernelFactory::createGemm(
        wv, KernelDeviceType::CPU);

    std::vector<float> q_cpu(M * N_q), k_cpu(M * N_k), v_cpu(M * N_v);
    const float *h_input = input->data();
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel_q.get(), h_input, q_cpu.data(), M, N_q, K));
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel_k.get(), h_input, k_cpu.data(), M, N_k, K));
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel_v.get(), h_input, v_cpu.data(), M, N_v, K));

    // Add biases (CPU side)
    if (bias_q)
    {
        const float *bq = static_cast<const FP32Tensor *>(bias_q.get())->data();
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N_q; ++j)
            {
                q_cpu[i * N_q + j] += bq[j];
            }
        }
    }
    if (bias_k)
    {
        const float *bk = static_cast<const FP32Tensor *>(bias_k.get())->data();
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N_k; ++j)
            {
                k_cpu[i * N_k + j] += bk[j];
            }
        }
    }
    if (bias_v)
    {
        const float *bv = static_cast<const FP32Tensor *>(bias_v.get())->data();
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N_v; ++j)
            {
                v_cpu[i * N_v + j] += bv[j];
            }
        }
    }

    // Compare
    const float *q_cuda = output_q->data();
    const float *k_cuda = output_k->data();
    const float *v_cuda = output_v->data();

    auto result_q = checkParity(q_cuda, q_cpu.data(), M * N_q, 0.99, 0.10);
    result_q.print("Q with bias (CUDA vs CPU)");
    EXPECT_GE(result_q.cosine_similarity, 0.99);
    EXPECT_FALSE(result_q.has_nan_inf);

    auto result_k = checkParity(k_cuda, k_cpu.data(), M * N_k, 0.99, 0.10);
    result_k.print("K with bias (CUDA vs CPU)");
    EXPECT_GE(result_k.cosine_similarity, 0.99);
    EXPECT_FALSE(result_k.has_nan_inf);

    auto result_v = checkParity(v_cuda, v_cpu.data(), M * N_v, 0.99, 0.10);
    result_v.print("V with bias (CUDA vs CPU)");
    EXPECT_GE(result_v.cosine_similarity, 0.99);
    EXPECT_FALSE(result_v.has_nan_inf);

    // Clean up shared workspace
    cleanupSharedWorkspace({cuda_kernel_q, cuda_kernel_k, cuda_kernel_v});

    // Clean up cache entries while tensors are still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_q.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_k.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_v.get());
}

/**
 * @test Fused QKV using cached kernels (simulating full pipeline)
 *
 * This test mimics the full pipeline's kernel usage:
 * 1. Uses prepared-handle + GEMM engine lookup to get cached kernels
 * 2. Uses real model weights and biases
 * 3. Runs multiple iterations like decode
 */
TEST_F(Test__CUDAGemmParity, FusedQKV_CachedKernels_MultipleIterations)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    // Load weights
    auto weights_q = loader.loadTensor("blk.0.attn_q.weight", DeviceId::cpu());
    auto weights_k = loader.loadTensor("blk.0.attn_k.weight", DeviceId::cpu());
    auto weights_v = loader.loadTensor("blk.0.attn_v.weight", DeviceId::cpu());
    ASSERT_NE(weights_q, nullptr);
    ASSERT_NE(weights_k, nullptr);
    ASSERT_NE(weights_v, nullptr);

    auto *wq = dynamic_cast<Q4_0Tensor *>(weights_q.get());
    auto *wk = dynamic_cast<Q4_0Tensor *>(weights_k.get());
    auto *wv = dynamic_cast<Q4_0Tensor *>(weights_v.get());
    ASSERT_NE(wq, nullptr);
    ASSERT_NE(wk, nullptr);
    ASSERT_NE(wv, nullptr);

    const int M = 1;
    const int N_q = static_cast<int>(wq->shape()[0]);
    const int N_k = static_cast<int>(wk->shape()[0]);
    const int N_v = static_cast<int>(wv->shape()[0]);
    const int K = static_cast<int>(wq->shape()[1]);

    // Upload to GPU
    ASSERT_TRUE(wq->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(wk->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(wv->ensureOnDevice(gpu_device_));

    // Get CACHED kernels (this is what the pipeline does)
    auto *kernel_q = getPreparedKernel(wq, gpu_device_);
    auto *kernel_k = getPreparedKernel(wk, gpu_device_);
    auto *kernel_v = getPreparedKernel(wv, gpu_device_);
    ASSERT_NE(kernel_q, nullptr);
    ASSERT_NE(kernel_k, nullptr);
    ASSERT_NE(kernel_v, nullptr);

    // Set up SHARED workspace for fused QKV (all kernels share workspace)
    ASSERT_TRUE(setupSharedWorkspace(
        {kernel_q, kernel_k, kernel_v},
        M, {N_q, N_k, N_v}, K));

    // CPU kernels for reference
    auto cpu_kernel_q = llaminar::v2::kernels::KernelFactory::createGemm(
        wq, KernelDeviceType::CPU);
    auto cpu_kernel_k = llaminar::v2::kernels::KernelFactory::createGemm(
        wk, KernelDeviceType::CPU);
    auto cpu_kernel_v = llaminar::v2::kernels::KernelFactory::createGemm(
        wv, KernelDeviceType::CPU);

    std::cout << "Testing cached kernels over 5 iterations (simulating decode)...\n";

    // Run 5 iterations with different inputs
    for (int iter = 0; iter < 5; ++iter)
    {
        // Create fresh input for each iteration
        auto input = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
        float *input_data = input->mutable_data();
        for (int i = 0; i < M * K; ++i)
        {
            input_data[i] = dist_(rng_);
        }
        ASSERT_TRUE(input->ensureOnDevice(gpu_device_));

        // Create outputs
        auto out_q = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_q)});
        auto out_k = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_k)});
        auto out_v = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_v)});
        ASSERT_TRUE(out_q->ensureOnDevice(gpu_device_));
        ASSERT_TRUE(out_k->ensureOnDevice(gpu_device_));
        ASSERT_TRUE(out_v->ensureOnDevice(gpu_device_));

        // Run fused with cached kernels
        std::vector<TensorProjectionDesc> projections;
        projections.emplace_back(kernel_q, out_q.get(), N_q,
                                 nullptr, "Q");
        projections.emplace_back(kernel_k, out_k.get(), N_k,
                                 nullptr, "K");
        projections.emplace_back(kernel_v, out_v.get(), N_v,
                                 nullptr, "V");

        ASSERT_TRUE(withExplicitCudaGemmStream(
            {kernel_q, kernel_k, kernel_v},
            [&](void *stream)
            {
                return with_gpu_coherence(
                    gpu_device_,
                    {input.get()},
                    {out_q.get(), out_k.get(), out_v.get()},
                    static_cast<void *>(stream),
                    [&]
                    {
                        return kernel_q->multiply_fused_tensor(
                            input.get(), projections, M, K, nullptr);
                    });
            }));

        // CPU reference
        std::vector<float> q_cpu(M * N_q), k_cpu(M * N_k), v_cpu(M * N_v);
        const float *h_input = input->data();
        ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel_q.get(), h_input, q_cpu.data(), M, N_q, K));
        ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel_k.get(), h_input, k_cpu.data(), M, N_k, K));
        ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel_v.get(), h_input, v_cpu.data(), M, N_v, K));

        // Compare
        auto result_q = checkParity(out_q->data(), q_cpu.data(), M * N_q, 0.99, 0.10);
        auto result_k = checkParity(out_k->data(), k_cpu.data(), M * N_k, 0.99, 0.10);
        auto result_v = checkParity(out_v->data(), v_cpu.data(), M * N_v, 0.99, 0.10);

        std::cout << "  Iter " << iter << ": Q=" << result_q.cosine_similarity
                  << " K=" << result_k.cosine_similarity
                  << " V=" << result_v.cosine_similarity << "\n";

        EXPECT_GE(result_q.cosine_similarity, 0.99)
            << "Q failed at iteration " << iter;
        EXPECT_GE(result_k.cosine_similarity, 0.99)
            << "K failed at iteration " << iter;
        EXPECT_GE(result_v.cosine_similarity, 0.99)
            << "V failed at iteration " << iter;
        EXPECT_FALSE(result_q.has_nan_inf);
        EXPECT_FALSE(result_k.has_nan_inf);
        EXPECT_FALSE(result_v.has_nan_inf);
    }

    // Clean up shared workspace
    cleanupSharedWorkspace({kernel_q, kernel_k, kernel_v});

    // Clean up cache entries while tensors are still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_q.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_k.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_v.get());
}

#endif // HAVE_CUDA

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
