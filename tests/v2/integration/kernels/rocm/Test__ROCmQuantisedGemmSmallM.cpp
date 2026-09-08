/**
 * @file Test__ROCmQuantisedGemmSmallM.cpp
 * @brief ROCm small-M arithmetic and persistent fused-workspace regressions.
 *
 * Verifier tests use the public serial decode byte oracle. Fused fixtures
 * declare the same simultaneous projection workspace as production stages;
 * wide M cases also prove that the fixed scratch tile can be reused safely.
 */

#include <gtest/gtest.h>
#include "transfer/TransferEngine.h"

#include "execution/compute_stages/stages/GDNProjectionStage.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "backends/rocm/HIPGraphCapture.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "kernels/rocm/moe/ROCmMoEKernel.h"
#include "loaders/ModelContext.h"
#include "loaders/ModelContextConfig.h"
#include "tensors/TensorSlice.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"
#include "utils/PrefillGraphBucketDefaults.h"
#include "../../../utils/GpuPreparedGemmHarness.h"
#include "../../../utils/NativeVNNIEquivalenceInventory.h"
#include "../../../utils/QuantizedVerifierFormats.h"
#include "../../../utils/ScopedGPUStream.h"
#include "../../../utils/TestTensorFactory.h"
#include "../../../utils/VerifierRowTestInventory.h"

#include <algorithm>
#include <array>
#include <barrier>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <span>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>

extern "C" bool rocmQuantGemm_quantizeActivationsBlockwiseWithSums(
    const float *d_A_fp32,
    int8_t *d_A_int8,
    float *d_scales_blockwise,
    int32_t *d_sums_blockwise,
    int M, int K,
    int rocm_device_id, void *stream,
    int block_size);

extern "C" void rocmGemv_native_vnni_set_tuning_overrides(
    int kb,
    int target_waves_per_cu);

extern "C" void rocmGemv_native_vnni_reset_tuning_overrides();

extern "C" void rocmGemv_native_vnni_set_serial_partition_n(int n);
extern "C" int rocmGemv_native_vnni_get_serial_partition_n();

extern "C" bool rocmGemv_native_vnni_query_serial_m1_config(
    uint8_t codebook_id,
    int N,
    int K,
    int *kb,
    int *target_waves_per_cu);
#endif

using namespace llaminar2;
using namespace llaminar2::rocm;
using namespace llaminar2::test;

namespace
{
    enum class PackedPath
    {
        INT8VNNI,
        NativeVNNI,
    };

    using WeightCreator = std::function<std::unique_ptr<TensorBase>(
        const std::vector<size_t> &shape,
        uint32_t seed)>;

    struct NativeFormatCase
    {
        const char *label;
        WeightCreator create;
        float min_cosine;
        TensorType tensor_type = TensorType::FP32;
    };

#ifdef HAVE_ROCM
    /**
     * @brief Apply one trainer-only NativeVNNI launch override for a test scope.
     *
     * The ROCm decode trainer evaluates explicit kernel schedules through the
     * same process-wide override API used below.  Always restoring automatic
     * dispatch is important because GoogleTest may continue with another test
     * in the same process after an expectation fails.
     */
    class ScopedROCmNativeVNNITuningOverride final
    {
    public:
        /** Select the explicit K partition and target-wave candidate. */
        ScopedROCmNativeVNNITuningOverride(int kb, int target_waves_per_cu)
        {
            rocmGemv_native_vnni_set_tuning_overrides(kb, target_waves_per_cu);
        }

        /** Restore generated production dispatch when the test scope ends. */
        ~ScopedROCmNativeVNNITuningOverride()
        {
            rocmGemv_native_vnni_reset_tuning_overrides();
        }

        ScopedROCmNativeVNNITuningOverride(
            const ScopedROCmNativeVNNITuningOverride &) = delete;
        ScopedROCmNativeVNNITuningOverride &operator=(
            const ScopedROCmNativeVNNITuningOverride &) = delete;
    };

    /**
     * @brief Restore the calling thread's HIP device after a multi-device test.
     *
     * HIP's current device is thread-local process state. A test that walks
     * participants can otherwise leave the GoogleTest thread on its final
     * device, causing a later device-0 fixture to create a stream on device 1.
     * The guard makes that ownership boundary exception- and assertion-safe.
     */
    class ScopedHIPCurrentDeviceRestore final
    {
    public:
        /** @brief Snapshot the exact HIP device selected on the calling thread. */
        ScopedHIPCurrentDeviceRestore()
        {
            const hipError_t status = hipGetDevice(&saved_device_);
            if (status != hipSuccess)
            {
                throw std::runtime_error(
                    std::string("Could not snapshot current HIP device: ") +
                    hipGetErrorString(status));
            }
        }

        /** @brief Restore the saved device even when a test assertion returns early. */
        ~ScopedHIPCurrentDeviceRestore()
        {
            if (saved_device_ >= 0)
                (void)hipSetDevice(saved_device_);
        }

        ScopedHIPCurrentDeviceRestore(
            const ScopedHIPCurrentDeviceRestore &) = delete;
        ScopedHIPCurrentDeviceRestore &operator=(
            const ScopedHIPCurrentDeviceRestore &) = delete;

    private:
        int saved_device_ = -1; ///< Calling thread's device on scope entry.
    };

    class ScopedROCmSerialPartitionPolicy final
    {
    public:
        explicit ScopedROCmSerialPartitionPolicy(int n)
            : previous_(rocmGemv_native_vnni_get_serial_partition_n())
        {
            rocmGemv_native_vnni_set_serial_partition_n(n);
        }

        ~ScopedROCmSerialPartitionPolicy()
        {
            rocmGemv_native_vnni_set_serial_partition_n(previous_);
        }

        ScopedROCmSerialPartitionPolicy(
            const ScopedROCmSerialPartitionPolicy &) = delete;
        ScopedROCmSerialPartitionPolicy &operator=(
            const ScopedROCmSerialPartitionPolicy &) = delete;

    private:
        int previous_ = 0;
    };
#endif

    /**
     * @brief One production-weight representative for each ROCm MoE native-VNNI codegroup.
     *
     * Grouped MoE verifier prefill dispatches by native-VNNI codebook id, not by
     * the original GGUF tensor class.  Several GGUF formats collapse to the same
     * codegroup after GPU repack, so this table intentionally keeps exactly one
     * representative per advertised grouped-prefill codebook.  The test below
     * proves the codegroup dispatch surface rather than spending time on format
     * aliases that reach the same templated HIP instantiation.
     */
    struct NativeCodegroupCase
    {
        const char *label; ///< Human-readable format name used in failure messages.
        uint8_t codebook_id;
        WeightCreator create;
    };

    class ScopedEnv
    {
    public:
        ScopedEnv(const char *name, const char *value)
            : name_(name)
        {
            const char *old_value = std::getenv(name);
            if (old_value)
            {
                had_old_value_ = true;
                old_value_ = old_value;
            }
            setenv(name_.c_str(), value, 1);
            mutableDebugEnv().reload();
        }

        ~ScopedEnv()
        {
            if (had_old_value_)
                setenv(name_.c_str(), old_value_.c_str(), 1);
            else
                unsetenv(name_.c_str());
            mutableDebugEnv().reload();
        }

        ScopedEnv(const ScopedEnv &) = delete;
        ScopedEnv &operator=(const ScopedEnv &) = delete;

    private:
        std::string name_;
        bool had_old_value_ = false;
        std::string old_value_;
    };

    int rocmDeviceCount()
    {
#ifdef HAVE_ROCM
        int count = 0;
        const hipError_t err = hipGetDeviceCount(&count);
        return err == hipSuccess ? count : 0;
#else
        return 0;
#endif
    }

    bool hasROCmDevice()
    {
        return rocmDeviceCount() > 0;
    }

    void cpuFP32GemmRef(const float *A, const float *W, float *C, int M, int N, int K)
    {
        for (int row = 0; row < M; ++row)
        {
            for (int col = 0; col < N; ++col)
            {
                double acc = 0.0;
                for (int kk = 0; kk < K; ++kk)
                    acc += static_cast<double>(A[row * K + kk]) *
                           static_cast<double>(W[col * K + kk]);
                C[row * N + col] = static_cast<float>(acc);
            }
        }
    }

    float cosineSim(const float *a, const float *b, size_t n)
    {
        double dot = 0.0;
        double na = 0.0;
        double nb = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        if (na == 0.0 || nb == 0.0)
            return 0.0f;
        return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
    }

    float relativeL2(const float *a, const float *b, size_t n)
    {
        double diff2 = 0.0;
        double ref2 = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            diff2 += diff * diff;
            ref2 += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        return ref2 == 0.0 ? static_cast<float>(std::sqrt(diff2)) :
                             static_cast<float>(std::sqrt(diff2 / ref2));
    }

    float maxAbsDiff(const float *a, const float *b, size_t n)
    {
        float max_diff = 0.0f;
        for (size_t i = 0; i < n; ++i)
            max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
        return max_diff;
    }

    /**
     * @brief Return the Euclidean norm of a host float span.
     *
     * The all-codegroup grouped-MoE sweep compares grouped verifier rows against
     * the production rowwise decode entry points.  A shared failure can make both
     * paths return all zeroes, which would make equality metrics look deceptively
     * good.  The norm check below keeps every codegroup honest: the serial path
     * must produce meaningful, nonzero verifier rows before grouped equivalence is
     * accepted.
     */
    double l2Norm(const float *values, size_t n)
    {
        double sum_sq = 0.0;
        for (size_t i = 0; i < n; ++i)
            sum_sq += static_cast<double>(values[i]) * static_cast<double>(values[i]);
        return std::sqrt(sum_sq);
    }

    /**
     * @brief Location and values for the largest grouped-vs-serial difference.
     */
    struct MaxDifference
    {
        size_t index = 0;
        float actual = 0.0f;
        float reference = 0.0f;
        float abs_diff = 0.0f;
    };

    /**
     * @brief Find the single output element with the largest absolute delta.
     *
     * When a codebook fails, the row/column and raw values are more useful than a
     * lone aggregate metric.  The grouped verifier kernels are template-dispatched
     * by codebook id, so a max-difference breadcrumb usually points directly at the
     * faulty decode or publication path.
     */
    MaxDifference maxDifference(const float *actual, const float *reference, size_t n)
    {
        MaxDifference result{};
        for (size_t i = 0; i < n; ++i)
        {
            const float diff = std::fabs(actual[i] - reference[i]);
            if (diff > result.abs_diff)
            {
                result = MaxDifference{i, actual[i], reference[i], diff};
            }
        }
        return result;
    }

    /**
     * @brief Require FP32 verifier rows to be byte-for-byte identical.
     *
     * The grouped MoE verifier path publishes rows that serial decode would
     * otherwise own.  Similarity thresholds are useful diagnostics, but they do
     * not prove that grouped publication is safe: a one-ulp perturbation can be
     * amplified by later residual, GDN, MoE, and sampler stages.  The all-format
     * sweep therefore treats serial M=1 decode as the canonical byte stream and
     * fails on the first differing float bit pattern.
     */
    void expectBitwiseFP32RowsEqual(
        const std::string &label,
        const float *actual,
        const float *reference,
        size_t count,
        size_t row_width)
    {
        ASSERT_NE(actual, nullptr) << label;
        ASSERT_NE(reference, nullptr) << label;
        ASSERT_GT(count, 0u) << label;
        if (std::memcmp(actual, reference, count * sizeof(float)) == 0)
            return;

        size_t first_mismatch = 0;
        uint32_t actual_bits = 0;
        uint32_t reference_bits = 0;
        for (; first_mismatch < count; ++first_mismatch)
        {
            std::memcpy(&actual_bits, actual + first_mismatch, sizeof(actual_bits));
            std::memcpy(&reference_bits, reference + first_mismatch, sizeof(reference_bits));
            if (actual_bits != reference_bits)
                break;
        }

        const auto max_diff = maxDifference(actual, reference, count);
        const float cos = cosineSim(actual, reference, count);
        const float rel_l2 = relativeL2(actual, reference, count);
        const float max_abs = maxAbsDiff(actual, reference, count);
        const size_t safe_row_width = row_width == 0 ? count : row_width;
        const size_t mismatch_row = first_mismatch / safe_row_width;
        const size_t mismatch_col = first_mismatch % safe_row_width;
        ADD_FAILURE()
            << label
            << " first_mismatch=" << first_mismatch
            << " row=" << mismatch_row
            << " col=" << mismatch_col
            << " actual=" << (first_mismatch < count ? actual[first_mismatch] : 0.0f)
            << " reference=" << (first_mismatch < count ? reference[first_mismatch] : 0.0f)
            << " actual_bits=0x" << std::hex << actual_bits
            << " reference_bits=0x" << reference_bits << std::dec
            << " cosine=" << cos
            << " rel_l2=" << rel_l2
            << " max_abs=" << max_abs
            << " max_diff_index=" << max_diff.index
            << " max_diff_actual=" << max_diff.actual
            << " max_diff_reference=" << max_diff.reference
            << " max_diff_abs=" << max_diff.abs_diff;
    }

    /**
     * @brief Symmetric KL divergence after a numerically stable softmax.
     *
     * Grouped verifier rows are accepted only if they match serial decode as a
     * distribution, not merely by top-token or raw L2.  GEMV integration tests
     * use the same softmaxed-row view so kernel drift is caught before it
     * amplifies through attention, GDN recurrence, and LM-head sampling.
     */
    double symmetricSoftmaxKL(const float *a, const float *b, size_t n)
    {
        if (n == 0)
            return 0.0;

        const auto max_a = *std::max_element(a, a + n);
        const auto max_b = *std::max_element(b, b + n);
        std::vector<double> pa(n);
        std::vector<double> pb(n);
        double sum_a = 0.0;
        double sum_b = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            pa[i] = std::exp(static_cast<double>(a[i] - max_a));
            pb[i] = std::exp(static_cast<double>(b[i] - max_b));
            sum_a += pa[i];
            sum_b += pb[i];
        }

        constexpr double eps = 1e-30;
        double kl_ab = 0.0;
        double kl_ba = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            const double p = std::max(pa[i] / sum_a, eps);
            const double q = std::max(pb[i] / sum_b, eps);
            kl_ab += p * std::log(p / q);
            kl_ba += q * std::log(q / p);
        }
        return 0.5 * (kl_ab + kl_ba);
    }

    /**
     * @brief CPU reference for ROCm Q8 blockwise activation quantization.
     *
     * The ROCm small-M NativeVNNI path uses per-32-value activation scales and
     * quantized activation sums. This reference mirrors the GPU rounding and
     * clamp rules so integration tests can catch workgroup/wavefront indexing
     * errors before they become verifier-row numerical drift.
     */
    void cpuBlockwiseQuantizeWithSums(
        const std::vector<float> &input,
        int M,
        int K,
        std::vector<int8_t> &quantized,
        std::vector<float> &scales,
        std::vector<int32_t> &sums)
    {
        constexpr int block_size = 32;
        const int blocks_per_row = (K + block_size - 1) / block_size;
        quantized.assign(static_cast<size_t>(M) * static_cast<size_t>(K), 0);
        scales.assign(static_cast<size_t>(M) * static_cast<size_t>(blocks_per_row), 1.0f);
        sums.assign(static_cast<size_t>(M) * static_cast<size_t>(blocks_per_row), 0);

        for (int row = 0; row < M; ++row)
        {
            for (int block = 0; block < blocks_per_row; ++block)
            {
                const int k_start = block * block_size;
                const int k_end = std::min(k_start + block_size, K);
                float max_abs = 0.0f;
                for (int k = k_start; k < k_end; ++k)
                    max_abs = std::max(max_abs, std::fabs(input[static_cast<size_t>(row) * K + k]));

                const float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
                const float inv_scale = 1.0f / scale;
                int32_t sum = 0;
                for (int k = k_start; k < k_end; ++k)
                {
                    int q = static_cast<int>(std::rint(input[static_cast<size_t>(row) * K + k] * inv_scale));
                    q = std::max(-127, std::min(127, q));
                    quantized[static_cast<size_t>(row) * K + k] = static_cast<int8_t>(q);
                    sum += q;
                }

                const size_t out_idx = static_cast<size_t>(row) * blocks_per_row + block;
                scales[out_idx] = scale;
                sums[out_idx] = sum;
            }
        }
    }

    void expectNearFP32(
        const std::vector<float> &actual,
        const std::vector<float> &expected,
        float abs_tolerance,
        const char *label)
    {
        ASSERT_EQ(actual.size(), expected.size()) << label;

        float max_abs = 0.0f;
        size_t max_index = 0;
        for (size_t i = 0; i < actual.size(); ++i)
        {
            const float diff = std::fabs(actual[i] - expected[i]);
            if (diff > max_abs)
            {
                max_abs = diff;
                max_index = i;
            }
        }

        EXPECT_LE(max_abs, abs_tolerance)
            << label << " max_abs=" << max_abs
            << " index=" << max_index
            << " actual=" << actual[max_index]
            << " expected=" << expected[max_index];
    }

    /**
     * @brief Bind the production workspace BOM for scalar or fused projections.
     * @param kernel Kernel whose declared buffers are installed.
     * @param M Physical activation rows.
     * @param N Anchor projection width.
     * @param K Reduction width.
     * @param fused_columns Optional complete simultaneous projection inventory.
     * @return Allocated owner, or null if allocation fails.
     */
    std::unique_ptr<DeviceWorkspaceManager> bindWorkspace(
        ROCmQuantisedGemmKernel &kernel,
        int M,
        int N,
        int K,
        std::span<const int> fused_columns = {})
    {
        WorkspaceRequirements requirements =
            kernel.getWorkspaceRequirements(M, N, K);
        if (!fused_columns.empty())
            kernel.appendFusedProjectionWorkspaceRequirements(
                requirements, M, fused_columns, K);
        auto workspace = std::make_unique<DeviceWorkspaceManager>(
            DeviceId::rocm(0),
            requirements.total_bytes_with_alignment() + 64 * 1024 * 1024);
        if (!workspace->allocate(requirements))
            return nullptr;
        kernel.bindWorkspace(workspace.get());
        return workspace;
    }

    void expectPackedPath(const ROCmPackedWeights &packed, PackedPath path)
    {
        switch (path)
        {
        case PackedPath::INT8VNNI:
            EXPECT_FALSE(packed.int8_data_vnni.empty());
            EXPECT_TRUE(packed.native_vnni_payload.empty());
            break;
        case PackedPath::NativeVNNI:
            EXPECT_FALSE(packed.native_vnni_payload.empty());
            EXPECT_FALSE(packed.native_vnni_scales.empty());
            EXPECT_TRUE(packed.int8_data_vnni.empty());
            break;
        }
    }

    std::vector<NativeFormatCase> nativeFormatCases()
    {
        std::vector<NativeFormatCase> formats;
        formats.reserve(quantizedVerifierFormats().size());
        for (const auto &format : quantizedVerifierFormats())
            formats.push_back({format.label, format.create, 0.985f, format.tensor_type});
        return formats;
    }

    /**
     * @brief Create deterministic IQ3_S weights that do not collapse to zero.
     *
     * `TestTensorFactory::createIQ3_SRandom()` currently generates legal-looking
     * but low-entropy sign/high-bit fields.  That is fine for constructor smoke
     * tests, but it is too weak for the MoE grouped verifier sweep: we need every
     * codebook representative to drive nonzero gate, up, and down work through the
     * native-VNNI repack and decode kernels.  This local fixture mirrors the ROCm
     * repack test pattern with bounded scales so the regression remains focused on
     * grouped decode equivalence rather than on global test-data policy.
     */
    std::unique_ptr<TensorBase> createNonzeroIQ3SForGroupedMoE(
        const std::vector<size_t> &shape,
        uint32_t seed)
    {
        constexpr size_t block_size = IQ3_SBlock::BLOCK_SIZE;
        const size_t rows = shape.at(0);
        const size_t cols = shape.at(1);
        const size_t blocks_per_row = (cols + block_size - 1) / block_size;
        const size_t total_blocks = rows * blocks_per_row;

        std::vector<uint8_t> raw_data(total_blocks * sizeof(IQ3_SBlock));
        auto *blocks = reinterpret_cast<IQ3_SBlock *>(raw_data.data());
        for (size_t block_idx = 0; block_idx < total_blocks; ++block_idx)
        {
            IQ3_SBlock &block = blocks[block_idx];
            block.d = 0x3000; // FP16 0.125: strong enough to be nonzero, small enough to avoid overflow.

            for (size_t j = 0; j < std::size(block.qs); ++j)
            {
                block.qs[j] = static_cast<uint8_t>(
                    (seed + block_idx * 37u + j * 13u) & 0xffu);
            }
            for (size_t j = 0; j < std::size(block.qh); ++j)
            {
                block.qh[j] = static_cast<uint8_t>(
                    ((seed >> 3) + block_idx * 11u + j * 29u) & 0xffu);
            }
            for (size_t j = 0; j < std::size(block.signs); ++j)
            {
                block.signs[j] = static_cast<uint8_t>(
                    (0x5au ^ seed ^ (block_idx * 17u + j * 7u)) & 0xffu);
            }

            for (size_t j = 0; j < std::size(block.scales); ++j)
            {
                const uint8_t lo = static_cast<uint8_t>((1u + seed + block_idx + j) & 0x3u);
                const uint8_t hi = static_cast<uint8_t>((2u + (seed >> 2) + block_idx + j) & 0x3u);
                block.scales[j] = static_cast<uint8_t>(lo | (hi << 4));
            }
        }

        return std::make_unique<IQ3_STensor>(shape, raw_data);
    }

    /**
     * @brief Create bounded IQ4_XS weights for the grouped-MoE equivalence sweep.
     *
     * IQ4_XS stores each 32-column sub-block scale as `d * (ls - 32)`.  The generic
     * random factory uses a low `ls` value that is legal but creates very large
     * verifier outputs after the gate/up, SiLU, and down chain.  That magnifies
     * harmless single-ULP accumulation-order differences into an absolute delta of
     * roughly one, which is noisy for a regression whose purpose is grouped decode
     * equivalence.  This fixture keeps `ls` just above 32 and varies payload bytes
     * deterministically so codebook 4 still executes real nonzero work.
     */
    std::unique_ptr<TensorBase> createBoundedIQ4XSForGroupedMoE(
        const std::vector<size_t> &shape,
        uint32_t seed)
    {
        constexpr size_t block_size = IQ4_XSBlock::BLOCK_SIZE;
        const size_t rows = shape.at(0);
        const size_t cols = shape.at(1);
        const size_t blocks_per_row = (cols + block_size - 1) / block_size;
        const size_t total_blocks = rows * blocks_per_row;

        std::vector<uint8_t> raw_data(total_blocks * sizeof(IQ4_XSBlock));
        auto *blocks = reinterpret_cast<IQ4_XSBlock *>(raw_data.data());
        for (size_t block_idx = 0; block_idx < total_blocks; ++block_idx)
        {
            IQ4_XSBlock &block = blocks[block_idx];
            block.d = 0x2400; // FP16 0.015625 keeps MoE verifier rows in a stable range.
            block.scales_h = 0;
            std::memset(block.scales_l, 0, sizeof(block.scales_l));

            for (int sub = 0; sub < 8; ++sub)
            {
                const uint16_t ls = static_cast<uint16_t>(
                    33u + ((seed + block_idx + static_cast<size_t>(sub)) & 0x3u));
                block.scales_l[sub / 2] |= static_cast<uint8_t>(
                    (ls & 0x0fu) << (4 * (sub & 1)));
                block.scales_h |= static_cast<uint16_t>(
                    ((ls >> 4) & 0x3u) << (2 * sub));
            }

            for (size_t j = 0; j < std::size(block.qs); ++j)
            {
                const uint8_t lo = static_cast<uint8_t>(
                    (seed + block_idx * 19u + j * 5u) & 0x0fu);
                const uint8_t hi = static_cast<uint8_t>(
                    ((seed >> 4) + block_idx * 23u + j * 7u) & 0x0fu);
                block.qs[j] = static_cast<uint8_t>(lo | (hi << 4));
            }
        }

        return std::make_unique<IQ4_XSTensor>(shape, raw_data);
    }

    /**
     * @brief ROCm grouped-MoE NativeVNNI codegroups that claim verifier prefill support.
     *
     * Keep this table aligned with `groupedDecodeSupportsCodebook()` in
     * `ROCmMoEKernel.cpp` and the `LAUNCH_*_IF_PRESENT` macros in
     * `ROCmMoEGroupedPrefillKernels.hip`.  A missing entry means a codegroup can
     * be advertised to production without ever proving rowwise decode
     * equivalence in the focused GEMM/MTP integration suite.
     */
    std::vector<NativeCodegroupCase> nativeMoECodegroupCases()
    {
        return {
            {"Q4_0", 0, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ4_0Random(shape, seed); }},
            {"IQ4_XS", 4, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return createBoundedIQ4XSForGroupedMoE(shape, seed); }},
            {"Q4_K", 5, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ4_KRandom(shape, seed); }},
            {"Q5_0", 6, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ5_0Random(shape, seed); }},
            {"Q5_K", 7, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ5_KRandom(shape, seed); }},
            {"Q6_K", 8, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ6_KRandom(shape, seed); }},
            {"Q3_K", 9, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ3_KRandom(shape, seed); }},
            {"Q2_K", 10, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ2_KRandom(shape, seed); }},
            {"IQ3_S", 11, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return createNonzeroIQ3SForGroupedMoE(shape, seed); }},
            {"IQ3_XXS", 12, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createIQ3_XXSRandom(shape, seed); }},
            {"IQ2_S", 13, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createIQ2_SRandom(shape, seed); }},
            {"IQ2_XS", 14, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createIQ2_XSRandom(shape, seed); }},
            {"IQ2_XXS", 15, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createIQ2_XXSRandom(shape, seed); }},
            {"IQ1_S", 16, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createIQ1_SRandom(shape, seed); }},
            {"IQ1_M", 17, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createIQ1_MRandom(shape, seed); }},
            {"Q8_0", 19, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ8_0Random(shape, seed); }},
        };
    }

    std::unique_ptr<DeviceWorkspaceManager> bindMoEWorkspace(
        ROCmMoEKernel &kernel,
        int max_seq_len,
        int d_model,
        int intermediate,
        int num_experts,
        int top_k)
    {
        const WorkspaceRequirements requirements =
            MoEWorkspaceBuffers::rocmMoE(
                max_seq_len,
                d_model,
                intermediate,
                num_experts,
                top_k);
        auto workspace = std::make_unique<DeviceWorkspaceManager>(
            DeviceId::rocm(0),
            requirements.total_bytes_with_alignment() + 16 * 1024 * 1024);
        if (!workspace->allocate(requirements))
            return nullptr;
        kernel.bindWorkspace(workspace.get());
        return workspace;
    }

    /**
     * @brief Assert that the last grouped MoE verifier run used the optimized prefill path.
     *
     * Accuracy alone is not enough for these regressions: a rowwise serial path
     * could be correct while proving nothing about the grouped verifier kernels
     * that MTP publishes from.  The perfstats records are therefore part of the
     * contract. They prove that routing used the intended compact or scalable
     * device planner and that gate/up/down used the runtime-M batch-invariant
     * publication path selected by the generated tuning policy.
     */
    void expectMoEGroupedVerifierPerfStats(
        const char *label,
        int seq_len,
        int num_experts,
        int top_k)
    {
        const bool expect_small_grouping = seq_len * top_k <= 64;
        bool saw_expected_grouping = false;
        for (const auto &record : PerfStatsCollector::snapshot(
                 {expect_small_grouping
                      ? "kernel.rocm_moe_small_prefill_grouping_calls"
                      : "kernel.rocm_moe_general_prefill_grouping_calls"}))
        {
            const auto slots_it = record.tags.find("total_slots");
            const auto experts_it = record.tags.find("num_experts");
            const auto topk_it = record.tags.find("top_k");
            saw_expected_grouping =
                saw_expected_grouping ||
                (slots_it != record.tags.end() &&
                 slots_it->second == std::to_string(seq_len * top_k) &&
                 experts_it != record.tags.end() &&
                 experts_it->second == std::to_string(num_experts) &&
                 topk_it != record.tags.end() &&
                 topk_it->second == std::to_string(top_k));
        }

        bool saw_grouped_prefill = false;
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"kernel.rocm_moe_grouped_prefill_batch_invariant_calls"}))
        {
            auto tag_equals = [&](const char *key, const std::string &value)
            {
                const auto it = record.tags.find(key);
                return it != record.tags.end() && it->second == value;
            };
            const auto has_nonempty_tag = [&](const char *key)
            {
                const auto it = record.tags.find(key);
                return it != record.tags.end() && !it->second.empty();
            };
            saw_grouped_prefill =
                saw_grouped_prefill ||
                (tag_equals("seq_len", std::to_string(seq_len)) &&
                 tag_equals("top_k", std::to_string(top_k)) &&
                 tag_equals("active_expert_slots", std::to_string(num_experts)) &&
                 tag_equals("gateup_route",
                            seq_len > 8
                                ? "expert_tiled_original_row_quant"
                                : "route_owned_original_row_quant") &&
                 tag_equals("down_route",
                            seq_len > 8
                                ? "expert_tiled_partials_ordered_publish"
                                : "direct_ordered_publish") &&
                 has_nonempty_tag("gateup_tile_m") &&
                 has_nonempty_tag("gateup_tile_n") &&
                 has_nonempty_tag("down_tile_m") &&
                 has_nonempty_tag("down_tile_n"));
        }

        EXPECT_TRUE(saw_expected_grouping)
            << label << " must exercise the intended ROCm "
            << (expect_small_grouping ? "compact" : "scalable")
            << " device expert-grouping route\n"
            << PerfStatsCollector::summaryString(
                   {expect_small_grouping
                        ? "kernel.rocm_moe_small_prefill_grouping_calls"
                        : "kernel.rocm_moe_general_prefill_grouping_calls"},
                   40);
        EXPECT_TRUE(saw_grouped_prefill)
            << label << " must exercise ROCm grouped decode-equivalent verifier prefill "
            << "with ordered down publication\n"
            << PerfStatsCollector::summaryString(
                   {"kernel.rocm_moe_grouped_prefill_batch_invariant_calls"}, 40);
    }

    /**
     * @brief Prove a ROCm MoE grouped verifier codegroup equals rowwise decode.
     *
     * This helper deliberately compares grouped prefill against the production
     * M=1 device-routed entry points, not against a CPU or FP32 oracle. The
     * one-row routing tensors are republished for every serial row because the
     * immutable table API owns one captured route set and must reject changing
     * expert ids. MTP verifier rows publish device state produced by these
     * exact backend paths; therefore the grouped result must match what serial
     * decode would have produced for the same quantized codebook, route ids,
     * route weights, and hidden rows.
     */
    void runROCmMoECodegroupVerifierRowsMatchSerialDecode(const NativeCodegroupCase &gateup_format,
                                                          const NativeCodegroupCase &down_format,
                                                          std::span<const int> verifier_rows)
    {
#ifndef HAVE_ROCM
        (void)gateup_format;
        (void)down_format;
        (void)verifier_rows;
        GTEST_SKIP() << "HAVE_ROCM not enabled";
#else
        ASSERT_FALSE(verifier_rows.empty());
        const int max_seq_len = *std::max_element(verifier_rows.begin(), verifier_rows.end());
        ASSERT_GT(max_seq_len, 0);

        constexpr int d_model = 256;
        constexpr int intermediate = 256;
        constexpr int num_experts = 4;
        constexpr int top_k = 4;
        const DeviceId device = DeviceId::rocm(0);

        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess)
            << gateup_format.label << "/" << down_format.label << " explicit HIP stream";

        ROCmMoEKernel moe_kernel(0);
        static_cast<ITensorKernel &>(moe_kernel).setGPUStream(stream);
        auto moe_workspace = bindMoEWorkspace(moe_kernel, max_seq_len, d_model, intermediate, num_experts, top_k);
        ASSERT_NE(moe_workspace, nullptr) << gateup_format.label << "/" << down_format.label << " MoE workspace";

        std::vector<std::unique_ptr<TensorBase>> owned_weights;
        std::vector<GpuPreparedGemm> prepared_weights;
        owned_weights.reserve(static_cast<size_t>(num_experts * 3));
        prepared_weights.reserve(static_cast<size_t>(num_experts * 3));

        auto add_prepared = [&](int rows, int cols, const NativeCodegroupCase &codegroup, uint32_t seed,
                                const char *role) -> ITensorGemm *
        {
            auto weight = codegroup.create({static_cast<size_t>(rows), static_cast<size_t>(cols)}, seed);
            auto *weight_ptr = weight.get();
            owned_weights.push_back(std::move(weight));
            prepared_weights.push_back(makeGpuPreparedGemm(weight_ptr, device,
                                                           std::string("test.rocm_moe.all_codegroups.") +
                                                               gateup_format.label + "_gateup." + down_format.label +
                                                               "_down." + role + "." + std::to_string(seed),
                                                           ModelContextId{970000 + static_cast<uint64_t>(seed)}));

            auto *kernel = prepared_weights.back().kernel;
            DeviceNativeVNNIMatrixDesc desc{};
            const bool exported = kernel->exportNativeVNNIMatrixDesc(desc);
            EXPECT_TRUE(exported) << gateup_format.label << "/" << down_format.label << " descriptor export for "
                                  << role;
            if (!exported)
                return nullptr;
            EXPECT_EQ(desc.codebook_id, codegroup.codebook_id)
                << gateup_format.label << "/" << down_format.label
                << " must exercise the intended native-VNNI codegroup for " << role;
            if (desc.codebook_id != codegroup.codebook_id)
                return nullptr;
            return kernel;
        };

        struct ExpertGemmTriplet
        {
            ITensorGemm *gate = nullptr;
            ITensorGemm *up = nullptr;
            ITensorGemm *down = nullptr;
        };

        std::array<ExpertGemmTriplet, num_experts> experts{};
        for (int expert = 0; expert < num_experts; ++expert)
        {
            experts[static_cast<size_t>(expert)].gate =
                add_prepared(intermediate, d_model, gateup_format, 1000u + static_cast<uint32_t>(expert), "gate");
            ASSERT_NE(experts[static_cast<size_t>(expert)].gate, nullptr);
            experts[static_cast<size_t>(expert)].up =
                add_prepared(intermediate, d_model, gateup_format, 2000u + static_cast<uint32_t>(expert), "up");
            ASSERT_NE(experts[static_cast<size_t>(expert)].up, nullptr);
            experts[static_cast<size_t>(expert)].down =
                add_prepared(d_model, intermediate, down_format, 3000u + static_cast<uint32_t>(expert), "down");
            ASSERT_NE(experts[static_cast<size_t>(expert)].down, nullptr);
        }

        std::vector<DeviceNativeVNNIMatrixDesc> gate_descs(num_experts);
        std::vector<DeviceNativeVNNIMatrixDesc> up_descs(num_experts);
        std::vector<DeviceNativeVNNIMatrixDesc> down_descs(num_experts);
        for (int expert = 0; expert < num_experts; ++expert)
        {
            const auto &triplet = experts[static_cast<size_t>(expert)];
            ASSERT_TRUE(triplet.gate->exportNativeVNNIMatrixDesc(gate_descs[static_cast<size_t>(expert)]));
            ASSERT_TRUE(triplet.up->exportNativeVNNIMatrixDesc(up_descs[static_cast<size_t>(expert)]));
            ASSERT_TRUE(triplet.down->exportNativeVNNIMatrixDesc(down_descs[static_cast<size_t>(expert)]));
        }

        const int gateup_table = moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
            gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
        ASSERT_GE(gateup_table, 0) << gateup_format.label << "/" << down_format.label << " gate/up table";
        const int down_table =
            moe_kernel.uploadGroupedExpertDownDescriptorTable(down_descs.data(), num_experts, d_model, intermediate);
        ASSERT_GE(down_table, 0) << gateup_format.label << "/" << down_format.label << " down table";

        for (const int seq_len : verifier_rows)
        {
            SCOPED_TRACE("verifier_rows=" + std::to_string(seq_len));
            auto hidden = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
            for (size_t i = 0; i < hidden->numel(); ++i)
            {
                hidden->mutable_data()[i] = 0.017f * static_cast<float>(static_cast<int>(i % 37) - 18) +
                                            0.003f * static_cast<float>(static_cast<int>((i / 11) % 23) - 11);
            }
            ASSERT_TRUE(hidden->ensureOnDevice(device, stream))
                << gateup_format.label << "/" << down_format.label << " hidden upload M=" << seq_len;

            auto routing_indices =
                TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
            auto routing_weights =
                TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
            for (int row = 0; row < seq_len; ++row)
            {
                float weight_sum = 0.0f;
                for (int route = 0; route < top_k; ++route)
                {
                    const int slot = row * top_k + route;
                    routing_indices->mutable_data()[static_cast<size_t>(slot)] =
                        static_cast<float>((row + route) % num_experts);
                    routing_weights->mutable_data()[static_cast<size_t>(slot)] =
                        0.10f + 0.017f * static_cast<float>((row * 7 + route * 3) % 9);
                    weight_sum += routing_weights->mutable_data()[static_cast<size_t>(slot)];
                }
                for (int route = 0; route < top_k; ++route)
                {
                    routing_weights->mutable_data()[static_cast<size_t>(row * top_k + route)] /= weight_sum;
                }
            }
            ASSERT_TRUE(routing_indices->ensureOnDevice(device, stream));
            ASSERT_TRUE(routing_weights->ensureOnDevice(device, stream));

            std::vector<float> rowwise_expected(static_cast<size_t>(seq_len) * static_cast<size_t>(d_model));
            for (int row = 0; row < seq_len; ++row)
            {
                auto hidden_row = TestTensorFactory::createFP32({1u, static_cast<size_t>(d_model)});
                std::copy(hidden->data() + static_cast<size_t>(row) * d_model,
                          hidden->data() + static_cast<size_t>(row + 1) * d_model, hidden_row->mutable_data());
                ASSERT_TRUE(hidden_row->ensureOnDevice(device, stream));

                auto row_routing_indices = TestTensorFactory::createFP32(
                    {1u, static_cast<size_t>(top_k)});
                auto row_routing_weights = TestTensorFactory::createFP32(
                    {1u, static_cast<size_t>(top_k)});
                for (int route = 0; route < top_k; ++route)
                {
                    const int slot = row * top_k + route;
                    row_routing_indices->mutable_data()[static_cast<size_t>(route)] =
                        routing_indices->data()[static_cast<size_t>(slot)];
                    row_routing_weights->mutable_data()[static_cast<size_t>(route)] =
                        routing_weights->data()[static_cast<size_t>(slot)];
                }
                ASSERT_TRUE(row_routing_indices->ensureOnDevice(device, stream));
                ASSERT_TRUE(row_routing_weights->ensureOnDevice(device, stream));

                std::array<std::shared_ptr<FP32Tensor>, top_k> gate_owned;
                std::array<std::shared_ptr<FP32Tensor>, top_k> up_owned;
                std::array<ITensor *, top_k> gate_outputs = {};
                std::array<ITensor *, top_k> up_outputs = {};
                for (int route = 0; route < top_k; ++route)
                {
                    gate_owned[static_cast<size_t>(route)] =
                        TestTensorFactory::createFP32({1u, static_cast<size_t>(intermediate)});
                    up_owned[static_cast<size_t>(route)] =
                        TestTensorFactory::createFP32({1u, static_cast<size_t>(intermediate)});
                    ASSERT_TRUE(gate_owned[static_cast<size_t>(route)]->ensureOnDevice(device, stream));
                    ASSERT_TRUE(up_owned[static_cast<size_t>(route)]->ensureOnDevice(device, stream));
                    gate_outputs[static_cast<size_t>(route)] = gate_owned[static_cast<size_t>(route)].get();
                    up_outputs[static_cast<size_t>(route)] = up_owned[static_cast<size_t>(route)].get();
                }

                auto decode_output = TestTensorFactory::createFP32({1u, static_cast<size_t>(d_model)});
                ASSERT_TRUE(decode_output->ensureOnDevice(device, stream));
                ASSERT_TRUE(moe_kernel.groupedExpertGateUpDecodeFromRouting(
                    hidden_row.get(), row_routing_indices.get(), gateup_table,
                    top_k, gate_outputs.data(), up_outputs.data(), d_model,
                    intermediate))
                    << gateup_format.label << "/" << down_format.label << " rowwise gate/up row=" << row;
                ASSERT_TRUE(moe_kernel.groupedExpertDownDecodeFromRouting(
                    gate_outputs.data(), up_outputs.data(),
                    row_routing_indices.get(), row_routing_weights.get(),
                    down_table, top_k, decode_output.get(), d_model,
                    intermediate))
                    << gateup_format.label << "/" << down_format.label << " rowwise down row=" << row;
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
                TransferEngine::publishDeviceWrite(decode_output, device, stream);
                std::copy(decode_output->data(), decode_output->data() + d_model,
                          rowwise_expected.begin() + static_cast<size_t>(row) * static_cast<size_t>(d_model));
            }

            auto grouped_output =
                TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
            ASSERT_TRUE(grouped_output->ensureOnDevice(device, stream));

            PerfStatsCollector::reset();
            ASSERT_TRUE(moe_kernel.prepareExpertGroupsAsync(routing_indices.get(), routing_weights.get(), seq_len,
                                                            num_experts, top_k))
                << gateup_format.label << "/" << down_format.label << " grouped expert planning M=" << seq_len;
            ASSERT_TRUE(moe_kernel.executeGroupedPrefillPipeline(hidden.get(), grouped_output.get(), gateup_table,
                                                                 down_table, seq_len, d_model, intermediate,
                                                                 num_experts, top_k))
                << gateup_format.label << "/" << down_format.label << " grouped verifier prefill M=" << seq_len;
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            TransferEngine::publishDeviceWrite(grouped_output, device, stream);

            expectMoEGroupedVerifierPerfStats((std::string(gateup_format.label) + "/" + down_format.label).c_str(),
                                              seq_len, num_experts, top_k);

            const size_t output_count = static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
            const auto label = std::string("ROCm MoE grouped verifier codegroup ") + gateup_format.label + "_gateup/" +
                               down_format.label + "_down M=" + std::to_string(seq_len) +
                               " must match rowwise serial decode";
            const auto metrics = [&]()
            {
                const float *actual = grouped_output->data();
                const float *reference = rowwise_expected.data();
                const float cos = cosineSim(actual, reference, output_count);
                const float rel_l2 = relativeL2(actual, reference, output_count);
                const float max_abs = maxAbsDiff(actual, reference, output_count);
                const double skl = symmetricSoftmaxKL(actual, reference, output_count);
                const double actual_norm = l2Norm(actual, output_count);
                const double reference_norm = l2Norm(reference, output_count);
                const auto max_diff = maxDifference(actual, reference, output_count);
                const size_t max_row = max_diff.index / static_cast<size_t>(d_model);
                const size_t max_col = max_diff.index % static_cast<size_t>(d_model);
                LOG_INFO("[SmallM][MoECodegroup] gateup="
                         << gateup_format.label << " gateup_codebook=" << static_cast<int>(gateup_format.codebook_id)
                         << " down=" << down_format.label
                         << " down_codebook=" << static_cast<int>(down_format.codebook_id) << " M=" << seq_len
                         << " cosine=" << cos << " rel_l2=" << rel_l2 << " symmetric_kl=" << skl
                         << " max_abs=" << max_abs << " actual_norm=" << actual_norm
                         << " reference_norm=" << reference_norm << " max_row=" << max_row << " max_col=" << max_col
                         << " actual_at_max=" << max_diff.actual << " reference_at_max=" << max_diff.reference);
                return std::tuple<float, float, float, double>(cos, rel_l2, max_abs, skl);
            }();
            ASSERT_GT(l2Norm(rowwise_expected.data(), output_count), 1.0e-7)
                << label << " produced an all-zero serial decode witness";
            ASSERT_GT(l2Norm(grouped_output->data(), output_count), 1.0e-7)
                << label << " produced an all-zero grouped decode witness";
            (void)metrics;
            expectBitwiseFP32RowsEqual(label, grouped_output->data(), rowwise_expected.data(), output_count,
                                       static_cast<size_t>(d_model));
        }

        static_cast<ITensorKernel &>(moe_kernel).clearGPUStreamBinding();
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
    }

    template <typename CreateWeights>
    void runDispatchSmallMMatchesReference(const char *label, int M, int N, int K, PackedPath expected_path,
                                           CreateWeights createWeights, float min_cosine)
    {
        auto weights = createWeights({static_cast<size_t>(N), static_cast<size_t>(K)}, 42);
        std::vector<float> W_fp32(static_cast<size_t>(N) * K);
        weights->to_fp32(W_fp32.data());

        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));
        expectPackedPath(packed, expected_path);

        ROCmQuantisedGemmKernel kernel(&packed, 0);
#ifdef HAVE_ROCM
        hipStream_t stream = nullptr;
        ASSERT_EQ(
            hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
            hipSuccess);
        kernel.setGPUStream(stream);
#endif
        auto workspace = bindWorkspace(kernel, M, N, K);
        ASSERT_NE(workspace, nullptr);

        auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
        auto output = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});

        ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(output->allocateOnDevice(DeviceId::rocm(0)));

        ASSERT_TRUE(kernel.multiply_tensor(input.get(), output.get(), M, N, K));
#ifdef HAVE_ROCM
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
#endif
        TransferEngine::publishCurrentDeviceWrite(output, stream);

        std::vector<float> ref(static_cast<size_t>(M) * N);
        cpuFP32GemmRef(input->data(), W_fp32.data(), ref.data(), M, N, K);

        const float cos = cosineSim(output->data(), ref.data(), static_cast<size_t>(M) * N);
        LOG_INFO("[SmallM] " << label << " M=" << M << " cosine=" << cos);
        EXPECT_GT(cos, min_cosine);

        kernel.unbindWorkspace();
#ifdef HAVE_ROCM
        kernel.clearGPUStreamBinding();
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
    }

    /**
     * @brief Prove grouped verifier GEMV is equivalent to serial decode GEMV.
     *
     * This intentionally compares ROCm NativeVNNI grouped runtime-M output
     * against independent M=1 executions using the same packed weights. It does not use
     * a dequantized FP32 reference, because Phase 9.8 needs the grouped verifier
     * path to publish exactly the state/logits serial decode would have produced
     * for the same quantized model path.
     */
    template <typename CreateWeights>
    void runGroupedSmallMMatchesSerialRows(const char *label, int M, int N, int K, CreateWeights createWeights)
    {
        auto weights = createWeights({static_cast<size_t>(N), static_cast<size_t>(K)}, 9898);

        /*
         * Drive the same persistent GPU preparation pipeline used by model
         * loading.  The old version of this helper used packWeightsToROCm(),
         * which could prove a host-built descriptor while leaving device
         * repack, publication metadata, and Q8 source-layout handling untested.
         */
        auto prepared =
            makeGpuPreparedGemm(weights.get(), DeviceId::rocm(0), std::string("test.rocm.grouped_verifier.") + label,
                                ModelContextId{static_cast<uint64_t>(989800 + M)});
        auto *kernel = dynamic_cast<ROCmQuantisedGemmKernel *>(prepared.kernel);
        ASSERT_NE(kernel, nullptr) << label << " production prepared ROCm kernel";

#ifdef HAVE_ROCM
        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
#endif

#ifdef HAVE_ROCM
        kernel->setGPUStream(stream);
#endif
        auto workspace = bindWorkspace(*kernel, M, N, K);
        ASSERT_NE(workspace, nullptr);

        auto grouped_input = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)},
            -0.35f,
            0.35f,
            424242);
        std::vector<float> grouped_host(
            grouped_input->data(),
            grouped_input->data() + static_cast<size_t>(M) * static_cast<size_t>(K));

        auto grouped_output = TestTensorFactory::createFP32(
            {static_cast<size_t>(M), static_cast<size_t>(N)});

        ASSERT_TRUE(grouped_input->ensureOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(grouped_output->allocateOnDevice(DeviceId::rocm(0)));
#ifdef HAVE_ROCM
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
#endif

        /*
         * This helper proves the verifier path, not the generic M-aware prefill
         * route.  Keep the backend verifier scope alive only for the grouped
         * launch so ROCm selects the serial-M1 split policy while still using
         * the economical grouped kernel.
         */
        {
            auto verifier_scope = kernel->beginVerifierDecodeEquivalentScope();
            ASSERT_TRUE(kernel->multiply_tensor(grouped_input.get(), grouped_output.get(), M, N, K));
        }
#ifdef HAVE_ROCM
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
#endif
        TransferEngine::publishCurrentDeviceWrite(grouped_output, stream);
        std::vector<float> grouped_values(
            grouped_output->data(),
            grouped_output->data() + static_cast<size_t>(M) * static_cast<size_t>(N));

        std::vector<float> serial_values(static_cast<size_t>(M) * static_cast<size_t>(N));
        for (int row = 0; row < M; ++row)
        {
            auto row_input = TestTensorFactory::createFP32(
                {1, static_cast<size_t>(K)});
            std::copy_n(
                grouped_host.data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                K,
                row_input->mutable_data());
            auto row_output = TestTensorFactory::createFP32(
                {1, static_cast<size_t>(N)});

            ASSERT_TRUE(row_input->ensureOnDevice(DeviceId::rocm(0)));
            ASSERT_TRUE(row_output->allocateOnDevice(DeviceId::rocm(0)));
#ifdef HAVE_ROCM
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
#endif

            ASSERT_TRUE(kernel->multiply_tensor(row_input.get(), row_output.get(), 1, N, K))
                << "serial row=" << row;
#ifdef HAVE_ROCM
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
#endif
            TransferEngine::publishCurrentDeviceWrite(row_output, stream);
            std::copy_n(
                row_output->data(),
                N,
                serial_values.data() + static_cast<size_t>(row) * static_cast<size_t>(N));
        }

        float minimum_cosine = 1.0f;
        float maximum_relative_l2 = 0.0f;
        float maximum_absolute_error = 0.0f;
        for (int row = 0; row < M; ++row)
        {
            const float *grouped_row = grouped_values.data() + static_cast<size_t>(row) * static_cast<size_t>(N);
            const float *serial_row = serial_values.data() + static_cast<size_t>(row) * static_cast<size_t>(N);
            const float cos = cosineSim(grouped_row, serial_row, static_cast<size_t>(N));
            const float rel_l2 = relativeL2(grouped_row, serial_row, static_cast<size_t>(N));
            const float max_abs = maxAbsDiff(grouped_row, serial_row, static_cast<size_t>(N));
            minimum_cosine = std::min(minimum_cosine, cos);
            maximum_relative_l2 = std::max(maximum_relative_l2, rel_l2);
            maximum_absolute_error = std::max(maximum_absolute_error, max_abs);
        }
        LOG_INFO("[SmallM] " << label << " grouped-vs-serial M=" << M
                             << " minimum_cosine=" << minimum_cosine
                             << " maximum_relative_l2=" << maximum_relative_l2
                             << " maximum_absolute_error=" << maximum_absolute_error);

        expectBitwiseFP32RowsEqual(
            std::string(label) + " grouped verifier rows M=" + std::to_string(M),
            grouped_values.data(),
            serial_values.data(),
            static_cast<size_t>(M) * static_cast<size_t>(N),
            static_cast<size_t>(N));

#ifdef HAVE_ROCM
        kernel->clearGPUStreamBinding();
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
        kernel->unbindWorkspace();
    }

    std::filesystem::path qwen36DenseModelPath()
    {
        if (const char *env = std::getenv("LLAMINAR_QWEN36_DENSE_MODEL"))
            return std::filesystem::path(env);
        return std::filesystem::path("/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf");
    }

    std::filesystem::path qwen36MoEModelPath()
    {
        if (const char *env = std::getenv("LLAMINAR_QWEN36_MOE_MODEL"))
            return std::filesystem::path(env);
        return std::filesystem::path("/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf");
    }

    std::shared_ptr<ModelContext> loadQwen36DenseModelForGpuWeights(
        const std::filesystem::path &model_path)
    {
        ModelContextConfig config = ModelContextConfig::defaults();
        config.strategy = WeightDistributionStrategy::REPLICATED;
        config.weight_precision = WeightPrecision::NATIVE;
        config.use_mmap = true;
        config.payload_access_pattern =
            ModelPayloadAccessPattern::DeviceStaging;
        return ModelContext::create(model_path.string(), config);
    }

    std::optional<std::string> findFirstAvailableTensor(
        ModelContext &model_ctx,
        const std::vector<std::string> &candidates)
    {
        for (const std::string &name : candidates)
        {
            if (model_ctx.hasTensor(name))
                return name;
        }
        return std::nullopt;
    }

    std::optional<std::string> findFirstTensorWithSuffix(
        ModelContext &model_ctx,
        const std::string &suffix)
    {
        for (const auto &name : model_ctx.concreteLoader().tensorNames())
        {
            if (name.size() >= suffix.size() &&
                name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
            {
                return name;
            }
        }
        return std::nullopt;
    }

    void runGemmStageRows(
        TensorBase *weight,
        const GpuPreparedGemm &prepared,
        const std::vector<float> &input_values,
        int M,
        int N,
        int K,
        bool force_decode_equivalent,
        void *stream,
        bool graph_capture,
        std::vector<float> &result,
        DeviceId device = DeviceId::rocm(0))
    {
#ifdef HAVE_ROCM
        if (device.is_rocm())
            ASSERT_EQ(hipSetDevice(device.ordinal), hipSuccess);
#endif
        auto input = TestTensorFactory::createFP32(
            {static_cast<size_t>(M), static_cast<size_t>(K)});
        std::copy_n(input_values.data(), static_cast<size_t>(M) * static_cast<size_t>(K),
                    input->mutable_data());
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
        stage.setGPUStream(stream);

        const WorkspaceRequirements requirements = stage.getWorkspaceRequirements(M, N, K);
        const size_t budget = requirements.total_bytes_with_alignment() + 64 * 1024 * 1024;
        DeviceWorkspaceManager workspace(device, budget);
        ASSERT_TRUE(workspace.allocate(requirements));
        stage.bindWorkspace(&workspace);

        ROCmDeviceContext ctx(device, device.ordinal);
        if (graph_capture)
        {
#ifdef HAVE_ROCM
            /*
             * The activation upload is a graph-external producer.  Close that
             * event edge on the exact capture stream before entering HIP
             * capture, just as DeviceGraphExecutor does for production graph
             * inputs.  Waiting after capture begins would make the graph depend
             * on mutable work owned outside its retained transaction.
             */
            ASSERT_NO_THROW(TransferEngine::requireDeviceInput(
                input.get(), device, stream));
            HIPGraphCapture capture(
                static_cast<hipStream_t>(stream),
                device.ordinal);
            ScopedBackendGraphCapture capture_transaction(
                capture,
                "runGemmStageRows");
            ASSERT_TRUE(capture_transaction.begin());
            const bool capture_launch_ok = stage.execute(&ctx);
            capture_transaction.finish();
            ASSERT_TRUE(capture_launch_ok);
            ASSERT_TRUE(capture.instantiate());
            ASSERT_EQ(hipMemsetAsync(
                          output->gpu_data_ptr(),
                          0,
                          static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float),
                          static_cast<hipStream_t>(stream)),
                      hipSuccess);
            ASSERT_TRUE(capture.launch());
#else
            FAIL() << "graph_capture requested without HAVE_ROCM";
#endif
        }
        else
        {
            ASSERT_TRUE(stage.execute(&ctx));
        }
#ifdef HAVE_ROCM
        ASSERT_EQ(hipStreamSynchronize(static_cast<hipStream_t>(stream)), hipSuccess);
#endif
        TransferEngine::publishCurrentDeviceWrite(
            output,
            static_cast<hipStream_t>(stream));
        result.assign(
            output->data(),
            output->data() + static_cast<size_t>(M) * static_cast<size_t>(N));

        stage.unbindWorkspace();
    }

    void runRealWeightGemmStageGroupedRowsMatchSerial(
        TensorBase *weight,
        const GpuPreparedGemm &prepared,
        int M,
        int N,
        int K,
        float input_scale,
        bool graph_capture,
        void *stream)
    {
        auto grouped_input = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)},
            -input_scale,
            input_scale,
            919191 + M + static_cast<int>(input_scale * 1000.0f));
        std::vector<float> grouped_host(
            grouped_input->data(),
            grouped_input->data() + static_cast<size_t>(M) * static_cast<size_t>(K));

        std::vector<float> grouped_values;
        runGemmStageRows(
            weight, prepared, grouped_host, M, N, K,
            /*force_decode_equivalent=*/true, stream, graph_capture, grouped_values);
        ASSERT_EQ(grouped_values.size(), static_cast<size_t>(M) * static_cast<size_t>(N));

        std::vector<float> serial_values(static_cast<size_t>(M) * static_cast<size_t>(N));
        for (int row = 0; row < M; ++row)
        {
            std::vector<float> row_input(static_cast<size_t>(K));
            std::copy_n(
                grouped_host.data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                K,
                row_input.data());
            std::vector<float> row_values;
            runGemmStageRows(
                weight, prepared, row_input, 1, N, K,
                /*force_decode_equivalent=*/false, stream, /*graph_capture=*/false, row_values);
            ASSERT_EQ(row_values.size(), static_cast<size_t>(N));
            std::copy_n(
                row_values.data(),
                N,
                serial_values.data() + static_cast<size_t>(row) * static_cast<size_t>(N));
        }

        for (int row = 0; row < M; ++row)
        {
            const float *grouped_row = grouped_values.data() + static_cast<size_t>(row) * static_cast<size_t>(N);
            const float *serial_row = serial_values.data() + static_cast<size_t>(row) * static_cast<size_t>(N);
            const float cos = cosineSim(grouped_row, serial_row, static_cast<size_t>(N));
            const float rel_l2 = relativeL2(grouped_row, serial_row, static_cast<size_t>(N));
            const float max_abs = maxAbsDiff(grouped_row, serial_row, static_cast<size_t>(N));
            LOG_INFO("[SmallM] real Qwen3.6 output GEMMStage grouped-vs-serial M=" << M
                                                                                   << " scale=" << input_scale
                                                                                   << " graph_capture=" << graph_capture
                                                                                   << " row=" << row
                                                                                   << " cosine=" << cos
                                                                                   << " rel_l2=" << rel_l2
                                                                                   << " max_abs=" << max_abs);
        }

        expectBitwiseFP32RowsEqual(
            std::string("real Qwen3.6 output GEMMStage grouped verifier rows M=") +
                std::to_string(M) + " scale=" + std::to_string(input_scale),
            grouped_values.data(),
            serial_values.data(),
            static_cast<size_t>(M) * static_cast<size_t>(N),
            static_cast<size_t>(N));
    }

    /**
     * @brief Formats that must prove LocalTP Wo reconstruction equivalence.
     *
     * The model-level MTP failures we are chasing appear after row-parallel
     * attention output projection and allreduce.  A single real IQ3_S model is
     * a useful reproducer, but it is not a complete proof.  This table sweeps
     * every native ROCm quantized tensor format and keeps the high-risk
     * low-bit formats nonzero enough to exercise their actual decode kernels.
     */
    std::vector<NativeFormatCase> localTPWoQuantizedFormatCases()
    {
        std::vector<NativeFormatCase> formats;
        formats.reserve(quantizedVerifierFormats().size());
        for (const auto &format : quantizedVerifierFormats())
        {
            WeightCreator creator = format.create;
            if (format.tensor_type == TensorType::IQ4_XS)
                creator = createBoundedIQ4XSForGroupedMoE;
            else if (format.tensor_type == TensorType::IQ3_S)
                creator = createNonzeroIQ3SForGroupedMoE;
            formats.push_back({format.label, std::move(creator), 0.0f, format.tensor_type});
        }
        return formats;
    }

    std::vector<NativeFormatCase> localTPWoFloatingPointFormatCases()
    {
        return {
            {"FP32", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createFP32Random(shape, -0.20f, 0.20f, seed); }, 0.0f},
            {"FP16", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createFP16Random(shape, -0.20f, 0.20f, seed); }, 0.0f},
            {"BF16", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createBF16Random(shape, -0.20f, 0.20f, seed); }, 0.0f},
        };
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
             * Narrow local-attention rows cover the original synthetic LocalTP
             * Wo regression shape.  The wider case matches Qwen3.6 MoE
             * full-attention layers, where each of two LocalTP ranks owns a
             * 2048-wide attention-output slice before the row-parallel Wo
             * partials are allreduced back to the 2048-wide hidden stream.
             */
            {"qwen36_narrow_attention", 2048, 1024, 2048},
            {"qwen36_moe_full_attention", 2048, 2048, 4096},
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
        const NativeFormatCase &format_case,
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
            std::string("test.localtp_wo.") + format_case.label + ".rank" + std::to_string(rank),
            ModelContextId{static_cast<uint64_t>(71000 + seed + static_cast<uint32_t>(rank))});

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

    GroupedAndSerialRows runLocalTPWoShardVerifierRows(
        LocalTPWoShardFixture &fixture,
        int M,
        int N,
        int local_K,
        void *stream)
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
        runGemmStageRows(
            fixture.weight.get(),
            fixture.prepared,
            input,
            M,
            N,
            local_K,
            /*force_decode_equivalent=*/true,
            stream,
            /*graph_capture=*/false,
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
            runGemmStageRows(
                fixture.weight.get(),
                fixture.prepared,
                row_input,
                1,
                N,
                local_K,
                /*force_decode_equivalent=*/false,
                stream,
                /*graph_capture=*/false,
                row_values,
                fixture.device);
            if (row_values.size() != static_cast<size_t>(N))
            {
                ADD_FAILURE()
                    << "serial LocalTP Wo row produced " << row_values.size()
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
                    << "LocalTP partial size mismatch: got " << values.size()
                    << " expected " << result.size()
                    << " grouped=" << grouped;
                return result;
            }
            for (size_t i = 0; i < result.size(); ++i)
                result[i] += values[i];
        }
        return result;
    }

    void runLocalTPWoAllreduceVerifierRowsMatchSerial(
        const NativeFormatCase &format_case,
        const std::vector<DeviceId> &devices,
        const LocalTPWoShapeCase &shape_case,
        int N,
        int local_K,
        int full_K,
        uint32_t seed)
    {
        ASSERT_EQ(devices.size(), 2u);
#ifdef HAVE_ROCM
        std::vector<hipStream_t> streams(devices.size(), nullptr);
        for (size_t i = 0; i < devices.size(); ++i)
        {
            ASSERT_EQ(hipSetDevice(devices[i].ordinal), hipSuccess);
            ASSERT_EQ(hipStreamCreateWithFlags(&streams[i], hipStreamNonBlocking), hipSuccess)
                << "device=" << devices[i].to_string();
        }
#else
        std::vector<void *> streams(devices.size(), nullptr);
#endif

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
                partials.push_back(runLocalTPWoShardVerifierRows(
                    fixtures[rank],
                    M,
                    N,
                    local_K,
#ifdef HAVE_ROCM
                    streams[rank]
#else
                    streams[rank]
#endif
                    ));
            }

            const auto grouped_sum = sumLocalTPPartials(partials, true, M, N);
            const auto serial_sum = sumLocalTPPartials(partials, false, M, N);
            const double after = groupedVerifierGemmPerfCounterValue();
            EXPECT_GE(after - before, static_cast<double>(M * static_cast<int>(devices.size())))
                << format_case.label << " shape=" << shape_case.label << " M=" << M
                << " must exercise GEMMStage grouped verifier publication route\n"
                << PerfStatsCollector::summaryString(
                       {"mtp.gemm_grouped_decode_equivalent_verifier_prefill_rows"}, 20);

            expectBitwiseFP32RowsEqual(
                std::string("ROCm LocalTP Wo reconstructed allreduce grouped verifier rows format=") +
                    format_case.label + " shape=" + shape_case.label + " M=" + std::to_string(M),
                grouped_sum.data(),
                serial_sum.data(),
                grouped_sum.size(),
                static_cast<size_t>(N));
        }

#ifdef HAVE_ROCM
        for (size_t i = 0; i < streams.size(); ++i)
        {
            ASSERT_EQ(hipSetDevice(devices[i].ordinal), hipSuccess);
            ASSERT_EQ(hipStreamDestroy(streams[i]), hipSuccess);
        }
#endif
    }

    /**
     * @brief Production-shaped fixture for mirrored LocalTP attention Wo weights.
     *
     * The device-resident MTP target keeps small verifier projections local by
     * mirroring the full Wo/MTP-head matrix on each shard.  That avoids an
     * uneconomical allreduce for a handful of verifier rows.  The fixture below
     * models that target state directly with a full `[hidden, full_attention_dim]`
     * weight on one ROCm device.
     */
    struct ReplicatedWoFixture
    {
        DeviceId device;
        std::unique_ptr<TensorBase> weight;
        GpuPreparedGemm prepared;
        std::vector<float> input_rows;
    };

    ReplicatedWoFixture makeReplicatedWoFixture(
        const NativeFormatCase &format_case,
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
            std::string("test.replicated_wo.") + format_case.label + "." + shape_case.label,
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

    GroupedAndSerialRows runReplicatedWoVerifierRows(
        ReplicatedWoFixture &fixture,
        int M,
        int N,
        int K,
        void *stream)
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
        runGemmStageRows(
            fixture.weight.get(),
            fixture.prepared,
            input,
            M,
            N,
            K,
            /*force_decode_equivalent=*/true,
            stream,
            /*graph_capture=*/false,
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
            runGemmStageRows(
                fixture.weight.get(),
                fixture.prepared,
                row_input,
                1,
                N,
                K,
                /*force_decode_equivalent=*/false,
                stream,
                /*graph_capture=*/false,
                row_values,
                fixture.device);
            if (row_values.size() != static_cast<size_t>(N))
            {
                ADD_FAILURE()
                    << "serial ROCm replicated Wo row produced " << row_values.size()
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
     * This gate is intentionally separate from the LocalTP row-parallel
     * reconstruction gate: mirrored Wo is the primary device-resident path we
     * want for tiny verifier projections.  The perfstats assertion prevents an
     * accidental serial-row replay from masquerading as grouped verifier work.
     */
    void runReplicatedWoVerifierRowsMatchSerial(
        const NativeFormatCase &format_case,
        DeviceId device,
        const LocalTPWoShapeCase &shape_case,
        uint32_t seed)
    {
        const int N = shape_case.hidden;
        const int K = shape_case.full_attention_dim;
#ifdef HAVE_ROCM
        ASSERT_EQ(hipSetDevice(device.ordinal), hipSuccess);
        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess)
            << "device=" << device.to_string();
#else
        void *stream = nullptr;
#endif

        auto fixture = makeReplicatedWoFixture(format_case, shape_case, device, seed);
        for (const int M : kGroupedVerifierRuntimeRows)
        {
            PerfStatsCollector::reset();
            const double before = groupedVerifierGemmPerfCounterValue();
            const GroupedAndSerialRows rows = runReplicatedWoVerifierRows(
                fixture,
                M,
                N,
                K,
#ifdef HAVE_ROCM
                stream
#else
                stream
#endif
            );
            const double after = groupedVerifierGemmPerfCounterValue();
            EXPECT_GE(after - before, static_cast<double>(M))
                << format_case.label << " shape=" << shape_case.label << " M=" << M
                << " must exercise GEMMStage grouped verifier publication route\n"
                << PerfStatsCollector::summaryString(
                       {"mtp.gemm_grouped_decode_equivalent_verifier_prefill_rows"}, 20);

            expectBitwiseFP32RowsEqual(
                std::string("ROCm replicated Wo grouped verifier rows format=") +
                    format_case.label + " shape=" + shape_case.label +
                    " M=" + std::to_string(M),
                rows.grouped.data(),
                rows.serial.data(),
                rows.grouped.size(),
                static_cast<size_t>(N));
        }

#ifdef HAVE_ROCM
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
    }

    inline float swigluRef(float gate, float up)
    {
        return gate / (1.0f + std::exp(-gate)) * up;
    }

    template <typename CreateWeights>
    void runFusedSwiGLUDownSmallMMatchesReference(
        const char *label,
        int M,
        int N,
        int K,
        PackedPath expected_path,
        CreateWeights createWeights,
        float min_cosine,
        bool graph_capture = false,
        int graph_replays = 1)
    {
        ASSERT_GE(graph_replays, 1);

        auto weights = createWeights(
            {static_cast<size_t>(N), static_cast<size_t>(K)},
            4242);
        std::vector<float> W_fp32(static_cast<size_t>(N) * K);
        weights->to_fp32(W_fp32.data());

        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));
        expectPackedPath(packed, expected_path);

        ROCmQuantisedGemmKernel kernel(&packed, 0);
        kernel.prepareWeights();
        ASSERT_TRUE(kernel.weights_converted());

#ifdef HAVE_ROCM
        hipStream_t stream = nullptr;
        ASSERT_EQ(
            hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
            hipSuccess);
        kernel.setGPUStream(stream);
#endif

        auto workspace = bindWorkspace(kernel, M, N, K);
        ASSERT_NE(workspace, nullptr);

        auto gate = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)},
            -0.5f,
            0.5f,
            5151);
        auto up = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)},
            -0.5f,
            0.5f,
            6161);
        auto output = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});

        ASSERT_TRUE(gate->ensureOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(up->ensureOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(output->allocateOnDevice(DeviceId::rocm(0)));

#ifdef HAVE_ROCM
        std::unique_ptr<HIPGraphCapture> captured_graph;
        if (graph_capture)
        {
            captured_graph = std::make_unique<HIPGraphCapture>(stream, 0);
            ScopedBackendGraphCapture capture_transaction(
                *captured_graph,
                "runFusedSwiGLUDownSmallMMatchesReference");
            ASSERT_TRUE(capture_transaction.begin());
            const bool capture_launch_ok = kernel.multiply_tensor_with_fused_swiglu(
                gate.get(),
                up.get(),
                output.get(),
                M,
                N,
                K);
            capture_transaction.finish();
            ASSERT_TRUE(capture_launch_ok);
            ASSERT_TRUE(captured_graph->instantiate());

            for (int replay = 0; replay < graph_replays; ++replay)
            {
                ASSERT_EQ(hipMemsetAsync(output->gpu_data_ptr(),
                                         0,
                                         static_cast<size_t>(M) * N * sizeof(float),
                                         stream),
                          hipSuccess);
                ASSERT_TRUE(captured_graph->launch())
                    << "replay=" << replay;
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess)
                    << "replay=" << replay;
            }
        }
        else
#endif
        {
            ASSERT_TRUE(kernel.multiply_tensor_with_fused_swiglu(
                gate.get(),
                up.get(),
                output.get(),
                M,
                N,
                K));
#ifdef HAVE_ROCM
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
#endif
        }
        TransferEngine::publishCurrentDeviceWrite(output, stream);

        std::vector<float> swiglu(static_cast<size_t>(M) * K);
        for (size_t i = 0; i < swiglu.size(); ++i)
            swiglu[i] = swigluRef(gate->data()[i], up->data()[i]);

        std::vector<float> ref(static_cast<size_t>(M) * N);
        cpuFP32GemmRef(swiglu.data(), W_fp32.data(), ref.data(), M, N, K);

        const float cos = cosineSim(output->data(), ref.data(), static_cast<size_t>(M) * N);
        LOG_INFO("[SmallM] " << label << " fused SwiGLU down M=" << M << " cosine=" << cos);
        EXPECT_GT(cos, min_cosine);

#ifdef HAVE_ROCM
        captured_graph.reset();
        if (stream)
        {
            kernel.clearGPUStreamBinding();
            EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
        }
#endif
        kernel.unbindWorkspace();
    }

    template <typename CreateWeights>
    void runFusedSwiGLUDownSmallMMatchesSerialRows(
        const char *label,
        int M,
        int N,
        int K,
        PackedPath expected_path,
        CreateWeights createWeights,
        bool graph_capture = true,
        int graph_replays = 1)
    {
        ASSERT_GE(M, 2);
        ASSERT_GE(graph_replays, 1);

        auto weights = createWeights(
            {static_cast<size_t>(N), static_cast<size_t>(K)},
            777331);

        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));
        expectPackedPath(packed, expected_path);

        ROCmQuantisedGemmKernel kernel(&packed, 0);
        kernel.prepareWeights();
        ASSERT_TRUE(kernel.weights_converted());

#ifdef HAVE_ROCM
        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
        kernel.setGPUStream(stream);
#else
        void *stream = nullptr;
#endif

        auto grouped_workspace = bindWorkspace(kernel, M, N, K);
        ASSERT_NE(grouped_workspace, nullptr);

        auto gate = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)},
            -0.5f,
            0.5f,
            13579);
        auto up = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)},
            -0.5f,
            0.5f,
            24680);
        auto grouped_output = TestTensorFactory::createFP32(
            {static_cast<size_t>(M), static_cast<size_t>(N)});

        std::vector<float> gate_host(
            gate->data(),
            gate->data() + static_cast<size_t>(M) * static_cast<size_t>(K));
        std::vector<float> up_host(
            up->data(),
            up->data() + static_cast<size_t>(M) * static_cast<size_t>(K));

#ifdef HAVE_ROCM
        ASSERT_TRUE(gate->ensureOnDevice(DeviceId::rocm(0), stream));
        ASSERT_TRUE(up->ensureOnDevice(DeviceId::rocm(0), stream));
        ASSERT_TRUE(grouped_output->allocateOnDevice(DeviceId::rocm(0), stream));

        std::unique_ptr<HIPGraphCapture> captured_graph;
        if (graph_capture)
        {
            captured_graph = std::make_unique<HIPGraphCapture>(stream, 0);
            ScopedBackendGraphCapture capture_transaction(
                *captured_graph,
                "runFusedSwiGLUDownSmallMMatchesSerialRows");
            ASSERT_TRUE(capture_transaction.begin());
            /*
             * This helper is the verifier contract proof, not the generic
             * M-aware fused-SwiGLU throughput test.  The generated ROCm table
             * may choose a different split-K policy for M=2..4 than serial
             * decode's M=1 route; accepted-state publication must instead use
             * the decode-equivalent wrapper so every grouped row keeps the
             * serial verifier reduction order while still launching as a
             * graph-capturable grouped kernel.
             */
            const bool capture_launch_ok =
                kernel.multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                gate.get(),
                up.get(),
                grouped_output.get(),
                M,
                N,
                K,
                1.0f,
                0.0f,
                grouped_workspace.get());
            capture_transaction.finish();
            ASSERT_TRUE(capture_launch_ok);
            ASSERT_TRUE(captured_graph->instantiate());

            for (int replay = 0; replay < graph_replays; ++replay)
            {
                ASSERT_EQ(hipMemsetAsync(grouped_output->gpu_data_ptr(),
                                         0,
                                         static_cast<size_t>(M) * N * sizeof(float),
                                         stream),
                          hipSuccess);
                ASSERT_TRUE(captured_graph->launch())
                    << "replay=" << replay;
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess)
                    << "replay=" << replay;
            }
        }
        else
        {
            ASSERT_TRUE(kernel.multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                gate.get(),
                up.get(),
                grouped_output.get(),
                M,
                N,
                K,
                1.0f,
                0.0f,
                grouped_workspace.get()));
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        }
#else
        FAIL() << "ROCm serial-row oracle requested without HAVE_ROCM";
#endif

        TransferEngine::publishCurrentDeviceWrite(grouped_output, stream);
        std::vector<float> grouped_values(
            grouped_output->data(),
            grouped_output->data() + static_cast<size_t>(M) * static_cast<size_t>(N));

        kernel.unbindWorkspace();
        grouped_workspace.reset();

        std::vector<float> serial_values(static_cast<size_t>(M) * static_cast<size_t>(N));
        for (int row = 0; row < M; ++row)
        {
            auto row_gate = TestTensorFactory::createFP32(
                {1, static_cast<size_t>(K)});
            auto row_up = TestTensorFactory::createFP32(
                {1, static_cast<size_t>(K)});
            std::copy_n(
                gate_host.data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                K,
                row_gate->mutable_data());
            std::copy_n(
                up_host.data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                K,
                row_up->mutable_data());
            auto row_output = TestTensorFactory::createFP32(
                {1, static_cast<size_t>(N)});

#ifdef HAVE_ROCM
            ASSERT_TRUE(row_gate->ensureOnDevice(DeviceId::rocm(0), stream));
            ASSERT_TRUE(row_up->ensureOnDevice(DeviceId::rocm(0), stream));
            ASSERT_TRUE(row_output->allocateOnDevice(DeviceId::rocm(0), stream));
#endif

            auto row_workspace = bindWorkspace(kernel, 1, N, K);
            ASSERT_NE(row_workspace, nullptr);
            ASSERT_TRUE(kernel.multiply_tensor_with_fused_swiglu(
                row_gate.get(),
                row_up.get(),
                row_output.get(),
                1,
                N,
                K));
#ifdef HAVE_ROCM
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
#endif
            TransferEngine::publishCurrentDeviceWrite(row_output, stream);
            std::copy_n(
                row_output->data(),
                N,
                serial_values.data() + static_cast<size_t>(row) * static_cast<size_t>(N));
            kernel.unbindWorkspace();
        }

        for (int row = 0; row < M; ++row)
        {
            const float *grouped_row = grouped_values.data() + static_cast<size_t>(row) * static_cast<size_t>(N);
            const float *serial_row = serial_values.data() + static_cast<size_t>(row) * static_cast<size_t>(N);
            const float cos = cosineSim(grouped_row, serial_row, static_cast<size_t>(N));
            const float rel_l2 = relativeL2(grouped_row, serial_row, static_cast<size_t>(N));
            const float max_abs = maxAbsDiff(grouped_row, serial_row, static_cast<size_t>(N));
            const double skl = symmetricSoftmaxKL(grouped_row, serial_row, static_cast<size_t>(N));
            LOG_INFO("[SmallM] " << label << " fused SwiGLU/down grouped-vs-serial M=" << M
                                 << " graph_capture=" << graph_capture
                                 << " row=" << row
                                 << " cosine=" << cos
                                 << " rel_l2=" << rel_l2
                                 << " symmetric_kl=" << skl
                                 << " max_abs=" << max_abs);
        }

        expectBitwiseFP32RowsEqual(
            std::string(label) + " fused SwiGLU/down grouped verifier rows M=" +
                std::to_string(M),
            grouped_values.data(),
            serial_values.data(),
            static_cast<size_t>(M) * static_cast<size_t>(N),
            static_cast<size_t>(N));

#ifdef HAVE_ROCM
        captured_graph.reset();
        kernel.clearGPUStreamBinding();
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
    }

    template <typename CreateWeights>
    void runFusedQKVSmallMMatchesSeparate(
        const char *label,
        int M,
        int K,
        PackedPath expected_path,
        CreateWeights createWeights,
        float min_cosine,
        int Nq = 896,
        int Nk = 128,
        int Nv = 128)
    {
        auto wq = createWeights({static_cast<size_t>(Nq), static_cast<size_t>(K)}, 101);
        auto wk = createWeights({static_cast<size_t>(Nk), static_cast<size_t>(K)}, 102);
        auto wv = createWeights({static_cast<size_t>(Nv), static_cast<size_t>(K)}, 103);

        ROCmPackedWeights packed_q;
        ROCmPackedWeights packed_k;
        ROCmPackedWeights packed_v;
        ASSERT_TRUE(packWeightsToROCm(wq.get(), packed_q));
        ASSERT_TRUE(packWeightsToROCm(wk.get(), packed_k));
        ASSERT_TRUE(packWeightsToROCm(wv.get(), packed_v));
        expectPackedPath(packed_q, expected_path);
        expectPackedPath(packed_k, expected_path);
        expectPackedPath(packed_v, expected_path);

        ROCmQuantisedGemmKernel q_kernel(&packed_q, 0);
        ROCmQuantisedGemmKernel k_kernel(&packed_k, 0);
        ROCmQuantisedGemmKernel v_kernel(&packed_v, 0);

#ifdef HAVE_ROCM
        hipStream_t stream = nullptr;
        ASSERT_EQ(
            hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
            hipSuccess);
        q_kernel.setGPUStream(stream);
        k_kernel.setGPUStream(stream);
        v_kernel.setGPUStream(stream);
#endif

        auto q_workspace = bindWorkspace(q_kernel, M, Nq, K);
        auto k_workspace = bindWorkspace(k_kernel, M, Nk, K);
        auto v_workspace = bindWorkspace(v_kernel, M, Nv, K);
        ASSERT_NE(q_workspace, nullptr);
        ASSERT_NE(k_workspace, nullptr);
        ASSERT_NE(v_workspace, nullptr);

        auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
        auto separate_q = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(Nq)});
        auto separate_k = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(Nk)});
        auto separate_v = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(Nv)});
        auto fused_q = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(Nq)});
        auto fused_k = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(Nk)});
        auto fused_v = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(Nv)});
        auto bias_q = TestTensorFactory::createFP32({static_cast<size_t>(Nq)});
        auto bias_k = TestTensorFactory::createFP32({static_cast<size_t>(Nk)});
        auto bias_v = TestTensorFactory::createFP32({static_cast<size_t>(Nv)});

        for (int col = 0; col < Nq; ++col)
            bias_q->mutable_data()[col] = 0.03125f * static_cast<float>((col % 9) - 4);
        for (int col = 0; col < Nk; ++col)
        {
            bias_k->mutable_data()[col] = 0.03125f * static_cast<float>((col % 7) - 3);
            bias_v->mutable_data()[col] = 0.03125f * static_cast<float>((col % 5) - 2);
        }

        /*
         * Mirror request admission: production kernels consume already-placed
         * activation and bias tensors and write into stable output storage.
         * In particular, fused projection bias joining is not an implicit H2D
         * upload boundary, so every learned bias must be published before the
         * separate and fused production paths are compared.
         */
        const DeviceId device = DeviceId::rocm(0);
        TransferEngine::prepareDeviceInput(input.get(), device, stream);
        TransferEngine::prepareDeviceInput(bias_q.get(), device, stream);
        TransferEngine::prepareDeviceInput(bias_k.get(), device, stream);
        TransferEngine::prepareDeviceInput(bias_v.get(), device, stream);
        TransferEngine::prepareDeviceOutput(separate_q.get(), device, stream);
        TransferEngine::prepareDeviceOutput(separate_k.get(), device, stream);
        TransferEngine::prepareDeviceOutput(separate_v.get(), device, stream);
        TransferEngine::prepareDeviceOutput(fused_q.get(), device, stream);
        TransferEngine::prepareDeviceOutput(fused_k.get(), device, stream);
        TransferEngine::prepareDeviceOutput(fused_v.get(), device, stream);

        ASSERT_TRUE(q_kernel.multiply_tensor(input.get(), separate_q.get(), M, Nq, K,
                                             true, 1.0f, 0.0f, bias_q.get()));
        ASSERT_TRUE(k_kernel.multiply_tensor(input.get(), separate_k.get(), M, Nk, K,
                                             true, 1.0f, 0.0f, bias_k.get()));
        ASSERT_TRUE(v_kernel.multiply_tensor(input.get(), separate_v.get(), M, Nv, K,
                                             true, 1.0f, 0.0f, bias_v.get()));
#ifdef HAVE_ROCM
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
#endif
        TransferEngine::publishCurrentDeviceWrite(separate_q, stream);
        TransferEngine::publishCurrentDeviceWrite(separate_k, stream);
        TransferEngine::publishCurrentDeviceWrite(separate_v, stream);

        std::vector<ITensorGemm::TensorProjectionDesc> projections;
        projections.emplace_back(&q_kernel, fused_q.get(), Nq, bias_q.get(), "q_small_m");
        projections.emplace_back(&k_kernel, fused_k.get(), Nk, bias_k.get(), "k_small_m");
        projections.emplace_back(&v_kernel, fused_v.get(), Nv, bias_v.get(), "v_small_m");

        ASSERT_TRUE(q_kernel.multiply_fused_verifier_rows_decode_equivalent(
            input.get(), projections, M, K, nullptr, q_workspace.get()));
#ifdef HAVE_ROCM
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
#endif
        TransferEngine::publishCurrentDeviceWrite(fused_q, stream);
        TransferEngine::publishCurrentDeviceWrite(fused_k, stream);
        TransferEngine::publishCurrentDeviceWrite(fused_v, stream);

        const float q_cos = cosineSim(fused_q->data(), separate_q->data(), static_cast<size_t>(M) * Nq);
        const float k_cos = cosineSim(fused_k->data(), separate_k->data(), static_cast<size_t>(M) * Nk);
        const float v_cos = cosineSim(fused_v->data(), separate_v->data(), static_cast<size_t>(M) * Nv);

        LOG_INFO("[SmallM] " << label << " fused QKV M=" << M << " cosine q="
                             << q_cos << " k=" << k_cos << " v=" << v_cos);
        EXPECT_GT(q_cos, min_cosine);
        EXPECT_GT(k_cos, min_cosine);
        EXPECT_GT(v_cos, min_cosine);

        q_kernel.unbindWorkspace();
        k_kernel.unbindWorkspace();
        v_kernel.unbindWorkspace();
#ifdef HAVE_ROCM
        q_kernel.clearGPUStreamBinding();
        k_kernel.clearGPUStreamBinding();
        v_kernel.clearGPUStreamBinding();
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
    }

    template <typename CreateWeights>
    void runFusedQKVSmallMMatchesSerialM1DecodeRows(
        const char *label,
        int M,
        int K,
        PackedPath expected_path,
        CreateWeights createWeights,
        int Nq = 896,
        int Nk = 128,
        int Nv = 128)
    {
        ASSERT_GE(M, 2);

        auto wq = createWeights({static_cast<size_t>(Nq), static_cast<size_t>(K)}, 4101);
        auto wk = createWeights({static_cast<size_t>(Nk), static_cast<size_t>(K)}, 4102);
        auto wv = createWeights({static_cast<size_t>(Nv), static_cast<size_t>(K)}, 4103);

        ROCmPackedWeights packed_q;
        ROCmPackedWeights packed_k;
        ROCmPackedWeights packed_v;
        ASSERT_TRUE(packWeightsToROCm(wq.get(), packed_q));
        ASSERT_TRUE(packWeightsToROCm(wk.get(), packed_k));
        ASSERT_TRUE(packWeightsToROCm(wv.get(), packed_v));
        expectPackedPath(packed_q, expected_path);
        expectPackedPath(packed_k, expected_path);
        expectPackedPath(packed_v, expected_path);

        ROCmQuantisedGemmKernel q_kernel(&packed_q, 0);
        ROCmQuantisedGemmKernel k_kernel(&packed_k, 0);
        ROCmQuantisedGemmKernel v_kernel(&packed_v, 0);

#ifdef HAVE_ROCM
        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
        ASSERT_NE(stream, nullptr);
        q_kernel.setGPUStream(stream);
        k_kernel.setGPUStream(stream);
        v_kernel.setGPUStream(stream);
#endif

        const std::array<int, 3> fused_columns{Nq, Nk, Nv};
        auto q_workspace = bindWorkspace(q_kernel, M, Nq, K, fused_columns);
        auto k_workspace = bindWorkspace(k_kernel, M, Nk, K);
        auto v_workspace = bindWorkspace(v_kernel, M, Nv, K);
        ASSERT_NE(q_workspace, nullptr);
        ASSERT_NE(k_workspace, nullptr);
        ASSERT_NE(v_workspace, nullptr);

        auto input = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)},
            -0.75f,
            0.75f,
            9090u + static_cast<uint32_t>(M));
        auto fused_q = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(Nq)});
        auto fused_k = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(Nk)});
        auto fused_v = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(Nv)});
        auto bias_q = TestTensorFactory::createFP32({static_cast<size_t>(Nq)});
        auto bias_k = TestTensorFactory::createFP32({static_cast<size_t>(Nk)});
        auto bias_v = TestTensorFactory::createFP32({static_cast<size_t>(Nv)});

        for (int col = 0; col < Nq; ++col)
            bias_q->mutable_data()[col] = 0.015625f * static_cast<float>((col % 11) - 5);
        for (int col = 0; col < Nk; ++col)
        {
            bias_k->mutable_data()[col] = 0.015625f * static_cast<float>((col % 7) - 3);
            bias_v->mutable_data()[col] = 0.015625f * static_cast<float>((col % 5) - 2);
        }

#ifdef HAVE_ROCM
        const DeviceId device = DeviceId::rocm(0);

        /*
         * Admission owns every placement operation.  The grouped verifier and
         * the serial-row oracle below both exercise production kernels, whose
         * contract is to join already-published inputs and write into stable
         * device storage.  Bias is a real kernel input just like activation;
         * leaving it host-only would test an obsolete implicit-upload path.
         */
        TransferEngine::prepareDeviceInput(input.get(), device, stream);
        TransferEngine::prepareDeviceInput(bias_q.get(), device, stream);
        TransferEngine::prepareDeviceInput(bias_k.get(), device, stream);
        TransferEngine::prepareDeviceInput(bias_v.get(), device, stream);
        TransferEngine::prepareDeviceOutput(fused_q.get(), device, stream);
        TransferEngine::prepareDeviceOutput(fused_k.get(), device, stream);
        TransferEngine::prepareDeviceOutput(fused_v.get(), device, stream);
#else
        ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(bias_q->ensureOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(bias_k->ensureOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(bias_v->ensureOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(fused_q->allocateOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(fused_k->allocateOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(fused_v->allocateOnDevice(DeviceId::rocm(0)));
#endif

        std::vector<ITensorGemm::TensorProjectionDesc> projections;
        projections.emplace_back(&q_kernel, fused_q.get(), Nq, bias_q.get(), "q_serial_m1_oracle");
        projections.emplace_back(&k_kernel, fused_k.get(), Nk, bias_k.get(), "k_serial_m1_oracle");
        projections.emplace_back(&v_kernel, fused_v.get(), Nv, bias_v.get(), "v_serial_m1_oracle");

        ASSERT_TRUE(q_kernel.multiply_fused_verifier_rows_decode_equivalent(
            input.get(), projections, M, K, nullptr, q_workspace.get()));
#ifdef HAVE_ROCM
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
#endif
        TransferEngine::publishCurrentDeviceWrite(fused_q, stream);
        TransferEngine::publishCurrentDeviceWrite(fused_k, stream);
        TransferEngine::publishCurrentDeviceWrite(fused_v, stream);

        const struct ProjectionCase
        {
            ROCmQuantisedGemmKernel *kernel;
            FP32Tensor *fused;
            TensorBase *bias;
            int n;
            const char *name;
        } projection_cases[] = {
            {&q_kernel, fused_q.get(), bias_q.get(), Nq, "q"},
            {&k_kernel, fused_k.get(), bias_k.get(), Nk, "k"},
            {&v_kernel, fused_v.get(), bias_v.get(), Nv, "v"},
        };

        for (int row = 0; row < M; ++row)
        {
            auto row_input = TestTensorFactory::createFP32({1u, static_cast<size_t>(K)});
            std::copy(input->data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                      input->data() + static_cast<size_t>(row + 1) * static_cast<size_t>(K),
                      row_input->mutable_data());
#ifdef HAVE_ROCM
            TransferEngine::prepareDeviceInput(row_input.get(), device, stream);
#else
            ASSERT_TRUE(row_input->ensureOnDevice(DeviceId::rocm(0)));
#endif

            for (const ProjectionCase &projection : projection_cases)
            {
                auto serial = TestTensorFactory::createFP32({1u, static_cast<size_t>(projection.n)});
#ifdef HAVE_ROCM
                TransferEngine::prepareDeviceOutput(serial.get(), device, stream);
#else
                ASSERT_TRUE(serial->allocateOnDevice(DeviceId::rocm(0)));
#endif
                ASSERT_TRUE(projection.kernel->multiply_tensor(
                    row_input.get(),
                    serial.get(),
                    1,
                    projection.n,
                    K,
                    true,
                    1.0f,
                    0.0f,
                    projection.bias))
                    << label << " serial M=1 projection=" << projection.name
                    << " row=" << row;
#ifdef HAVE_ROCM
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
#endif
                TransferEngine::publishCurrentDeviceWrite(serial, stream);

                const float *grouped_row =
                    projection.fused->data() + static_cast<size_t>(row) * static_cast<size_t>(projection.n);
                const float *serial_row = serial->data();
                const size_t count = static_cast<size_t>(projection.n);
                const float cos = cosineSim(grouped_row, serial_row, count);
                const float rel_l2 = relativeL2(grouped_row, serial_row, count);
                const float max_abs = maxAbsDiff(grouped_row, serial_row, count);
                const double skl = symmetricSoftmaxKL(grouped_row, serial_row, count);

                LOG_INFO("[SmallM] " << label
                                     << " projection=" << projection.name
                                     << " M=" << M
                                     << " row=" << row
                                     << " cosine=" << cos
                                     << " rel_l2=" << rel_l2
                                     << " symmetric_kl=" << skl
                                     << " max_abs=" << max_abs);
                expectBitwiseFP32RowsEqual(
                    std::string(label) + " projection=" + projection.name +
                        " grouped verifier row=" + std::to_string(row) +
                        " M=" + std::to_string(M),
                    grouped_row,
                    serial_row,
                    count,
                    count);
            }
        }

        q_kernel.unbindWorkspace();
        k_kernel.unbindWorkspace();
        v_kernel.unbindWorkspace();
#ifdef HAVE_ROCM
        q_kernel.clearGPUStreamBinding();
        k_kernel.clearGPUStreamBinding();
        v_kernel.clearGPUStreamBinding();
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
    }

    template <typename CreateWeights>
    void runFusedProjectionGroupSmallMMatchesSeparate(
        const char *label,
        int M,
        int K,
        PackedPath expected_path,
        CreateWeights createWeights,
        float min_cosine,
        const std::vector<int> &Ns,
        bool graph_capture,
        bool bind_fused_to_shared_workspace = false,
        int graph_replays = 1,
        bool verifier_decode_equivalent = false)
    {
        ASSERT_FALSE(Ns.empty());
        ASSERT_GE(graph_replays, 1);

#ifdef HAVE_ROCM
        hipStream_t stream = nullptr;
        ASSERT_EQ(
            hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
            hipSuccess);
#endif

        std::vector<std::unique_ptr<TensorBase>> weights;
        std::vector<ROCmPackedWeights> packed(Ns.size());
        std::vector<std::unique_ptr<ROCmQuantisedGemmKernel>> kernels;
        std::vector<std::unique_ptr<DeviceWorkspaceManager>> workspaces;
        weights.reserve(Ns.size());
        kernels.reserve(Ns.size());
        workspaces.reserve(Ns.size());

        for (size_t i = 0; i < Ns.size(); ++i)
        {
            weights.push_back(createWeights({static_cast<size_t>(Ns[i]), static_cast<size_t>(K)},
                                            static_cast<uint32_t>(200 + i)));
            ASSERT_TRUE(packWeightsToROCm(weights.back().get(), packed[i]));
            expectPackedPath(packed[i], expected_path);
            kernels.push_back(std::make_unique<ROCmQuantisedGemmKernel>(&packed[i], 0));
#ifdef HAVE_ROCM
            kernels.back()->setGPUStream(stream);
#endif
            workspaces.push_back(bindWorkspace(*kernels.back(), M, Ns[i], K));
            ASSERT_NE(workspaces.back(), nullptr);
        }

        auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
        ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));

        std::vector<std::unique_ptr<FP32Tensor>> separate;
        std::vector<std::unique_ptr<FP32Tensor>> fused;
        separate.reserve(Ns.size());
        fused.reserve(Ns.size());
        for (int n : Ns)
        {
            separate.push_back(TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(n)}));
            fused.push_back(TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(n)}));
            ASSERT_TRUE(separate.back()->allocateOnDevice(DeviceId::rocm(0)));
            ASSERT_TRUE(fused.back()->allocateOnDevice(DeviceId::rocm(0)));
        }

        for (size_t i = 0; i < Ns.size(); ++i)
        {
            ASSERT_TRUE(kernels[i]->multiply_tensor(input.get(), separate[i].get(), M, Ns[i], K));
        }
#ifdef HAVE_ROCM
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
#endif
        for (auto &output : separate)
            TransferEngine::publishCurrentDeviceWrite(output, stream);

        std::unique_ptr<DeviceWorkspaceManager> shared_workspace;
        if (bind_fused_to_shared_workspace)
        {
            WorkspaceRequirements combined;
            for (size_t i = 0; i < Ns.size(); ++i)
                combined.merge(kernels[i]->getWorkspaceRequirements(M, Ns[i], K));
            kernels.front()->appendFusedProjectionWorkspaceRequirements(
                combined, M, Ns, K);

            shared_workspace = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::rocm(0),
                combined.total_bytes_with_alignment() + 64 * 1024 * 1024);
            ASSERT_TRUE(shared_workspace->allocate(combined));

            for (auto &kernel : kernels)
                kernel->bindWorkspace(shared_workspace.get());
        }

        std::vector<ITensorGemm::TensorProjectionDesc> projections;
        projections.reserve(Ns.size());
        for (size_t i = 0; i < Ns.size(); ++i)
        {
            projections.emplace_back(kernels[i].get(),
                                     fused[i].get(),
                                     Ns[i],
                                     nullptr,
                                     "small_m_group");
        }

        /*
         * MTP verifier graph-capture tests must go through the same production
         * entry point as GEMMStage/GDNProjectionStage.  Calling the generic
         * fused prefill API can be numerically close, but it does not enter the
         * decode-equivalent grouped verifier scope or prove the publication
         * path that serial-row parity depends on.
         */
        auto launch_fused_group = [&]() -> bool
        {
            if (verifier_decode_equivalent)
            {
                return kernels.front()->multiply_fused_verifier_rows_decode_equivalent(
                    input.get(),
                    projections,
                    M,
                    K,
                    nullptr,
                    shared_workspace ? shared_workspace.get() : nullptr);
            }
            return kernels.front()->multiply_fused_tensor(input.get(), projections, M, K);
        };

#ifdef HAVE_ROCM
        std::unique_ptr<HIPGraphCapture> captured_graph;
        if (graph_capture)
        {
            ASSERT_TRUE(launch_fused_group());
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

            captured_graph = std::make_unique<HIPGraphCapture>(stream, 0);
            ScopedBackendGraphCapture capture_transaction(
                *captured_graph,
                "runFusedQKVSmallMMatchesSeparate");
            ASSERT_TRUE(capture_transaction.begin());
            const bool capture_launch_ok = launch_fused_group();
            capture_transaction.finish();
            ASSERT_TRUE(capture_launch_ok);
            ASSERT_TRUE(captured_graph->instantiate());
            for (int replay = 0; replay < graph_replays; ++replay)
            {
                for (size_t i = 0; i < Ns.size(); ++i)
                {
                    ASSERT_EQ(hipMemsetAsync(fused[i]->gpu_data_ptr(),
                                             0,
                                             static_cast<size_t>(M) * Ns[i] * sizeof(float),
                                             stream),
                              hipSuccess);
                }
                ASSERT_TRUE(captured_graph->launch())
                    << "replay=" << replay;
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess)
                    << "replay=" << replay;
            }
        }
        else
#endif
        {
            ASSERT_TRUE(launch_fused_group());
#ifdef HAVE_ROCM
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
#endif
        }

        for (size_t i = 0; i < Ns.size(); ++i)
        {
            TransferEngine::publishCurrentDeviceWrite(fused[i], stream);
            const float cos = cosineSim(fused[i]->data(),
                                        separate[i]->data(),
                                        static_cast<size_t>(M) * Ns[i]);
            LOG_INFO("[SmallM] " << label << " projection=" << i
                                 << " M=" << M << " N=" << Ns[i]
                                 << " cosine=" << cos);
            EXPECT_GT(cos, min_cosine);
        }

#ifdef HAVE_ROCM
        captured_graph.reset();
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
        for (auto &kernel : kernels)
            kernel->unbindWorkspace();
    }

    template <typename CreateWeights>
    void runFusedProjectionGroupSmallMMatchesReference(
        const char *label,
        int M,
        int K,
        PackedPath expected_path,
        CreateWeights createWeights,
        float min_cosine,
        const std::vector<int> &Ns,
        const std::vector<const char *> &projection_names,
        bool graph_capture,
        bool bind_fused_to_shared_workspace = false,
        int graph_replays = 1)
    {
        ASSERT_FALSE(Ns.empty());
        ASSERT_EQ(Ns.size(), projection_names.size());
        ASSERT_GE(graph_replays, 1);

#ifdef HAVE_ROCM
        hipStream_t stream = nullptr;
        ASSERT_EQ(
            hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
            hipSuccess);
#endif

        std::vector<std::unique_ptr<TensorBase>> weights;
        std::vector<std::vector<float>> weights_fp32;
        std::vector<ROCmPackedWeights> packed(Ns.size());
        std::vector<std::unique_ptr<ROCmQuantisedGemmKernel>> kernels;
        std::vector<std::unique_ptr<DeviceWorkspaceManager>> workspaces;
        weights.reserve(Ns.size());
        weights_fp32.reserve(Ns.size());
        kernels.reserve(Ns.size());
        workspaces.reserve(Ns.size());

        for (size_t i = 0; i < Ns.size(); ++i)
        {
            weights.push_back(createWeights({static_cast<size_t>(Ns[i]), static_cast<size_t>(K)},
                                            static_cast<uint32_t>(900 + i)));
            weights_fp32.emplace_back(static_cast<size_t>(Ns[i]) * static_cast<size_t>(K));
            weights.back()->to_fp32(weights_fp32.back().data());

            ASSERT_TRUE(packWeightsToROCm(weights.back().get(), packed[i]));
            expectPackedPath(packed[i], expected_path);
            kernels.push_back(std::make_unique<ROCmQuantisedGemmKernel>(&packed[i], 0));
#ifdef HAVE_ROCM
            kernels.back()->setGPUStream(stream);
#endif
            workspaces.push_back(bindWorkspace(*kernels.back(), M, Ns[i], K));
            ASSERT_NE(workspaces.back(), nullptr);
        }

        auto input = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)},
            -0.75f,
            0.75f,
            8080);
        ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));

        std::vector<std::unique_ptr<FP32Tensor>> fused;
        fused.reserve(Ns.size());
        for (int n : Ns)
        {
            fused.push_back(TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(n)}));
            ASSERT_TRUE(fused.back()->allocateOnDevice(DeviceId::rocm(0)));
        }

        std::unique_ptr<DeviceWorkspaceManager> shared_workspace;
        if (bind_fused_to_shared_workspace)
        {
            WorkspaceRequirements combined;
            for (size_t i = 0; i < Ns.size(); ++i)
                combined.merge(kernels[i]->getWorkspaceRequirements(M, Ns[i], K));
            kernels.front()->appendFusedProjectionWorkspaceRequirements(
                combined, M, Ns, K);

            shared_workspace = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::rocm(0),
                combined.total_bytes_with_alignment() + 64 * 1024 * 1024);
            ASSERT_TRUE(shared_workspace->allocate(combined));

            for (auto &kernel : kernels)
                kernel->bindWorkspace(shared_workspace.get());
        }

        std::vector<ITensorGemm::TensorProjectionDesc> projections;
        projections.reserve(Ns.size());
        for (size_t i = 0; i < Ns.size(); ++i)
        {
            projections.emplace_back(kernels[i].get(),
                                     fused[i].get(),
                                     Ns[i],
                                     nullptr,
                                     projection_names[i]);
        }

#ifdef HAVE_ROCM
        std::unique_ptr<HIPGraphCapture> captured_graph;
        if (graph_capture)
        {
            ASSERT_TRUE(kernels.front()->multiply_fused_tensor(input.get(), projections, M, K));
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

            captured_graph = std::make_unique<HIPGraphCapture>(stream, 0);
            ScopedBackendGraphCapture capture_transaction(
                *captured_graph,
                "runFusedProjectionGroupSmallMMatchesReference");
            ASSERT_TRUE(capture_transaction.begin());
            const bool capture_launch_ok = kernels.front()->multiply_fused_tensor(
                input.get(), projections, M, K);
            capture_transaction.finish();
            ASSERT_TRUE(capture_launch_ok);
            ASSERT_TRUE(captured_graph->instantiate());

            for (int replay = 0; replay < graph_replays; ++replay)
            {
                for (size_t i = 0; i < Ns.size(); ++i)
                {
                    ASSERT_EQ(hipMemsetAsync(fused[i]->gpu_data_ptr(),
                                             0,
                                             static_cast<size_t>(M) * static_cast<size_t>(Ns[i]) * sizeof(float),
                                             stream),
                              hipSuccess);
                }
                ASSERT_TRUE(captured_graph->launch())
                    << "replay=" << replay;
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess)
                    << "replay=" << replay;
            }
        }
        else
#endif
        {
            ASSERT_TRUE(kernels.front()->multiply_fused_tensor(input.get(), projections, M, K));
#ifdef HAVE_ROCM
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
#endif
        }

        for (size_t i = 0; i < Ns.size(); ++i)
        {
            TransferEngine::publishCurrentDeviceWrite(fused[i], stream);

            std::vector<float> ref(static_cast<size_t>(M) * static_cast<size_t>(Ns[i]));
            cpuFP32GemmRef(input->data(), weights_fp32[i].data(), ref.data(), M, Ns[i], K);

            const float cos = cosineSim(fused[i]->data(),
                                        ref.data(),
                                        static_cast<size_t>(M) * static_cast<size_t>(Ns[i]));
            LOG_INFO("[SmallM] " << label
                                 << " projection=" << projection_names[i]
                                 << " M=" << M << " N=" << Ns[i]
                                 << " cosine=" << cos);
            EXPECT_GT(cos, min_cosine)
                << "projection=" << projection_names[i]
                << " N=" << Ns[i];
        }

#ifdef HAVE_ROCM
        captured_graph.reset();
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
        for (auto &kernel : kernels)
            kernel->unbindWorkspace();
    }

    void runMixedProjectionGroupSmallMMatchesSeparate(
        const char *label,
        int M,
        int K,
        const std::vector<WeightCreator> &createWeights,
        float min_cosine,
        const std::vector<int> &Ns,
        bool graph_capture,
        int graph_replays = 1,
        bool verifier_decode_equivalent = false)
    {
        ASSERT_FALSE(Ns.empty());
        ASSERT_EQ(createWeights.size(), Ns.size());
        ASSERT_GE(graph_replays, 1);

#ifdef HAVE_ROCM
        hipStream_t stream = nullptr;
        ASSERT_EQ(
            hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
            hipSuccess);
#endif

        std::vector<std::unique_ptr<TensorBase>> weights;
        std::vector<ROCmPackedWeights> packed(Ns.size());
        std::vector<std::unique_ptr<ROCmQuantisedGemmKernel>> kernels;
        std::vector<std::unique_ptr<DeviceWorkspaceManager>> workspaces;
        weights.reserve(Ns.size());
        kernels.reserve(Ns.size());
        workspaces.reserve(Ns.size());

        for (size_t i = 0; i < Ns.size(); ++i)
        {
            weights.push_back(createWeights[i]({static_cast<size_t>(Ns[i]), static_cast<size_t>(K)},
                                               static_cast<uint32_t>(700 + i)));
            ASSERT_TRUE(packWeightsToROCm(weights.back().get(), packed[i]));
            expectPackedPath(packed[i], PackedPath::NativeVNNI);
            kernels.push_back(std::make_unique<ROCmQuantisedGemmKernel>(&packed[i], 0));
#ifdef HAVE_ROCM
            kernels.back()->setGPUStream(stream);
#endif
            workspaces.push_back(bindWorkspace(*kernels.back(), M, Ns[i], K));
            ASSERT_NE(workspaces.back(), nullptr);
        }

        auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
        ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));

        std::vector<std::unique_ptr<FP32Tensor>> separate;
        std::vector<std::unique_ptr<FP32Tensor>> fused;
        separate.reserve(Ns.size());
        fused.reserve(Ns.size());
        for (int n : Ns)
        {
            separate.push_back(TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(n)}));
            fused.push_back(TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(n)}));
            ASSERT_TRUE(separate.back()->allocateOnDevice(DeviceId::rocm(0)));
            ASSERT_TRUE(fused.back()->allocateOnDevice(DeviceId::rocm(0)));
        }

        for (size_t i = 0; i < Ns.size(); ++i)
            ASSERT_TRUE(kernels[i]->multiply_tensor(input.get(), separate[i].get(), M, Ns[i], K));
#ifdef HAVE_ROCM
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
#endif
        for (auto &output : separate)
            TransferEngine::publishCurrentDeviceWrite(output, stream);

        WorkspaceRequirements combined;
        for (size_t i = 0; i < Ns.size(); ++i)
            combined.merge(kernels[i]->getWorkspaceRequirements(M, Ns[i], K));
        kernels.front()->appendFusedProjectionWorkspaceRequirements(
            combined, M, Ns, K);

        auto shared_workspace = std::make_unique<DeviceWorkspaceManager>(
            DeviceId::rocm(0),
            combined.total_bytes_with_alignment() + 64 * 1024 * 1024);
        ASSERT_TRUE(shared_workspace->allocate(combined));
        for (auto &kernel : kernels)
            kernel->bindWorkspace(shared_workspace.get());

        std::vector<ITensorGemm::TensorProjectionDesc> projections;
        projections.reserve(Ns.size());
        for (size_t i = 0; i < Ns.size(); ++i)
        {
            projections.emplace_back(kernels[i].get(),
                                     fused[i].get(),
                                     Ns[i],
                                     nullptr,
                                     "mixed_small_m_group");
        }

        auto launch_fused_group = [&]() -> bool
        {
            if (verifier_decode_equivalent)
            {
                return kernels.front()->multiply_fused_verifier_rows_decode_equivalent(
                    input.get(),
                    projections,
                    M,
                    K,
                    nullptr,
                    shared_workspace.get());
            }
            return kernels.front()->multiply_fused_tensor(input.get(), projections, M, K);
        };

#ifdef HAVE_ROCM
        std::unique_ptr<HIPGraphCapture> captured_graph;
        if (graph_capture)
        {
            captured_graph = std::make_unique<HIPGraphCapture>(stream, 0);
            ScopedBackendGraphCapture capture_transaction(
                *captured_graph,
                "runMixedProjectionGroupSmallMMatchesSeparate");
            ASSERT_TRUE(capture_transaction.begin());
            const bool capture_launch_ok = launch_fused_group();
            capture_transaction.finish();
            ASSERT_TRUE(capture_launch_ok);
            ASSERT_TRUE(captured_graph->instantiate());
            for (int replay = 0; replay < graph_replays; ++replay)
            {
                for (size_t i = 0; i < Ns.size(); ++i)
                {
                    ASSERT_EQ(hipMemsetAsync(fused[i]->gpu_data_ptr(),
                                             0,
                                             static_cast<size_t>(M) * Ns[i] * sizeof(float),
                                             stream),
                              hipSuccess);
                }
                ASSERT_TRUE(captured_graph->launch())
                    << "replay=" << replay;
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess)
                    << "replay=" << replay;
            }
        }
        else
#endif
        {
            ASSERT_TRUE(launch_fused_group());
#ifdef HAVE_ROCM
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
#endif
        }

        for (size_t i = 0; i < Ns.size(); ++i)
        {
            TransferEngine::publishCurrentDeviceWrite(fused[i], stream);
            const float cos = cosineSim(fused[i]->data(),
                                        separate[i]->data(),
                                        static_cast<size_t>(M) * Ns[i]);
            LOG_INFO("[SmallM] " << label << " mixed projection=" << i
                                 << " M=" << M << " N=" << Ns[i]
                                 << " cosine=" << cos);
            EXPECT_GT(cos, min_cosine);
        }

#ifdef HAVE_ROCM
        captured_graph.reset();
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
        for (auto &kernel : kernels)
            kernel->unbindWorkspace();
    }

    void runMixedProjectionGroupSmallMMatchesSerialM1DecodeRows(
        const char *label,
        int M,
        int K,
        const std::vector<WeightCreator> &createWeights,
        const std::vector<int> &Ns,
        const std::vector<const char *> &projection_names)
    {
        ASSERT_GE(M, 2);
        ASSERT_FALSE(Ns.empty());
        ASSERT_EQ(createWeights.size(), Ns.size());
        ASSERT_EQ(projection_names.size(), Ns.size());

#ifdef HAVE_ROCM
        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
        ASSERT_NE(stream, nullptr);
#endif

        std::vector<std::unique_ptr<TensorBase>> weights;
        std::vector<ROCmPackedWeights> packed(Ns.size());
        std::vector<std::unique_ptr<ROCmQuantisedGemmKernel>> kernels;
        weights.reserve(Ns.size());
        kernels.reserve(Ns.size());

        WorkspaceRequirements combined;
        for (size_t i = 0; i < Ns.size(); ++i)
        {
            weights.push_back(createWeights[i]({static_cast<size_t>(Ns[i]), static_cast<size_t>(K)},
                                               static_cast<uint32_t>(8100 + i)));
            ASSERT_TRUE(packWeightsToROCm(weights.back().get(), packed[i]));
            expectPackedPath(packed[i], PackedPath::NativeVNNI);
            kernels.push_back(std::make_unique<ROCmQuantisedGemmKernel>(&packed[i], 0));
#ifdef HAVE_ROCM
            kernels.back()->setGPUStream(stream);
#endif
            combined.merge(kernels.back()->getWorkspaceRequirements(M, Ns[i], K));
        }
        kernels.front()->appendFusedProjectionWorkspaceRequirements(
            combined, M, Ns, K);

        auto shared_workspace = std::make_unique<DeviceWorkspaceManager>(
            DeviceId::rocm(0),
            combined.total_bytes_with_alignment() + 64 * 1024 * 1024);
        ASSERT_TRUE(shared_workspace->allocate(combined));
        for (auto &kernel : kernels)
            kernel->bindWorkspace(shared_workspace.get());

        auto input = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)},
            -0.35f,
            0.35f,
            9190u + static_cast<uint32_t>(M));
#ifdef HAVE_ROCM
        ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0), stream));
#else
        ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
#endif

        std::vector<std::unique_ptr<FP32Tensor>> fused;
        fused.reserve(Ns.size());
        std::vector<ITensorGemm::TensorProjectionDesc> projections;
        projections.reserve(Ns.size());
        for (size_t i = 0; i < Ns.size(); ++i)
        {
            fused.push_back(TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(Ns[i])}));
            ASSERT_TRUE(fused.back()->allocateOnDevice(DeviceId::rocm(0)));
            projections.emplace_back(kernels[i].get(),
                                     fused.back().get(),
                                     Ns[i],
                                     nullptr,
                                     projection_names[i]);
        }

        ASSERT_TRUE(kernels.front()->multiply_fused_verifier_rows_decode_equivalent(
            input.get(), projections, M, K, nullptr, shared_workspace.get()));
#ifdef HAVE_ROCM
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
#endif
        for (auto &output : fused)
            TransferEngine::publishCurrentDeviceWrite(output, stream);

        for (int row = 0; row < M; ++row)
        {
            auto row_input = TestTensorFactory::createFP32({1u, static_cast<size_t>(K)});
            std::copy(input->data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                      input->data() + static_cast<size_t>(row + 1) * static_cast<size_t>(K),
                      row_input->mutable_data());
#ifdef HAVE_ROCM
            ASSERT_TRUE(row_input->ensureOnDevice(DeviceId::rocm(0), stream));
#else
            ASSERT_TRUE(row_input->ensureOnDevice(DeviceId::rocm(0)));
#endif

            for (size_t projection = 0; projection < Ns.size(); ++projection)
            {
                auto serial = TestTensorFactory::createFP32(
                    {1u, static_cast<size_t>(Ns[projection])});
                ASSERT_TRUE(serial->allocateOnDevice(DeviceId::rocm(0)));
                ASSERT_TRUE(kernels[projection]->multiply_tensor(
                    row_input.get(),
                    serial.get(),
                    1,
                    Ns[projection],
                    K))
                    << label << " projection=" << projection_names[projection]
                    << " serial row=" << row;
#ifdef HAVE_ROCM
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
#endif
                TransferEngine::publishCurrentDeviceWrite(serial, stream);

                const float *grouped_row =
                    fused[projection]->data() + static_cast<size_t>(row) * static_cast<size_t>(Ns[projection]);
                const float *serial_row = serial->data();
                const size_t count = static_cast<size_t>(Ns[projection]);
                const float cos = cosineSim(grouped_row, serial_row, count);
                const float rel_l2 = relativeL2(grouped_row, serial_row, count);
                const float max_abs = maxAbsDiff(grouped_row, serial_row, count);
                const double skl = symmetricSoftmaxKL(grouped_row, serial_row, count);

                LOG_INFO("[SmallM] " << label
                                     << " projection=" << projection_names[projection]
                                     << " M=" << M
                                     << " row=" << row
                                     << " cosine=" << cos
                                     << " rel_l2=" << rel_l2
                                     << " symmetric_kl=" << skl
                                     << " max_abs=" << max_abs);
                expectBitwiseFP32RowsEqual(
                    std::string(label) + " projection=" +
                        projection_names[projection] +
                        " grouped verifier row=" + std::to_string(row) +
                        " M=" + std::to_string(M),
                    grouped_row,
                    serial_row,
                    count,
                    count);
            }
        }

#ifdef HAVE_ROCM
        for (auto &kernel : kernels)
            kernel->clearGPUStreamBinding();
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
        for (auto &kernel : kernels)
            kernel->unbindWorkspace();
    }

    /**
     * @brief Compare one padded request-batch projection with an isolated request.
     *
     * Request-batched prefill can move a projection from an M=11 launch to an
     * M=32 launch even though the second request still owns only eleven real
     * rows.  Every GEMM row is mathematically independent, so a dispatch-policy
     * change is not allowed to alter any output byte.  This helper exercises the
     * ordinary production fused-projection entry point at both shapes; it does
     * not force the MTP verifier policy or replay rows through M=1.
     */
    void runMixedProjectionPaddedPrefillMatchesIsolatedRequest(
        const char *label,
        int padded_rows,
        int request_row_offset,
        int request_rows,
        int K,
        const std::vector<WeightCreator> &createWeights,
        const std::vector<int> &Ns,
        const std::vector<const char *> &projection_names)
    {
        ASSERT_GT(padded_rows, 0);
        ASSERT_GE(request_row_offset, 0);
        ASSERT_GT(request_rows, 0);
        ASSERT_LE(request_row_offset + request_rows, padded_rows);
        ASSERT_EQ(createWeights.size(), Ns.size());
        ASSERT_EQ(projection_names.size(), Ns.size());

#ifdef HAVE_ROCM
        hipStream_t stream = nullptr;
        ASSERT_EQ(
            hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
            hipSuccess);
        ASSERT_NE(stream, nullptr);
#endif

        std::vector<std::unique_ptr<TensorBase>> weights;
        std::vector<ROCmPackedWeights> packed(Ns.size());
        std::vector<std::unique_ptr<ROCmQuantisedGemmKernel>> kernels;
        weights.reserve(Ns.size());
        kernels.reserve(Ns.size());

        WorkspaceRequirements combined;
        for (size_t projection = 0; projection < Ns.size(); ++projection)
        {
            weights.push_back(createWeights[projection](
                {static_cast<size_t>(Ns[projection]),
                 static_cast<size_t>(K)},
                static_cast<uint32_t>(12100 + projection)));
            ASSERT_TRUE(packWeightsToROCm(
                weights.back().get(), packed[projection]));
            expectPackedPath(packed[projection], PackedPath::NativeVNNI);
            kernels.push_back(std::make_unique<ROCmQuantisedGemmKernel>(
                &packed[projection], 0));
#ifdef HAVE_ROCM
            kernels.back()->setGPUStream(stream);
#endif
            combined.merge(kernels.back()->getWorkspaceRequirements(
                padded_rows, Ns[projection], K));
            combined.merge(kernels.back()->getWorkspaceRequirements(
                request_rows, Ns[projection], K));
        }
        kernels.front()->appendFusedProjectionWorkspaceRequirements(
            combined, padded_rows, Ns, K);
        kernels.front()->appendFusedProjectionWorkspaceRequirements(
            combined, request_rows, Ns, K);

        auto workspace = std::make_unique<DeviceWorkspaceManager>(
            DeviceId::rocm(0),
            combined.total_bytes_with_alignment() + 64 * 1024 * 1024);
        ASSERT_TRUE(workspace->allocate(combined));
        for (auto &kernel : kernels)
            kernel->bindWorkspace(workspace.get());

        auto padded_input = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(padded_rows), static_cast<size_t>(K)},
            -0.35f,
            0.35f,
            12190u);
        for (int row = request_row_offset + request_rows;
             row < padded_rows;
             ++row)
        {
            std::fill_n(
                padded_input->mutable_data() +
                    static_cast<size_t>(row) * static_cast<size_t>(K),
                K,
                1000.0f + static_cast<float>(row));
        }

        auto isolated_input = TestTensorFactory::createFP32(
            {static_cast<size_t>(request_rows), static_cast<size_t>(K)});
        std::copy_n(
            padded_input->data() +
                static_cast<size_t>(request_row_offset) *
                    static_cast<size_t>(K),
            static_cast<size_t>(request_rows) * static_cast<size_t>(K),
            isolated_input->mutable_data());
#ifdef HAVE_ROCM
        ASSERT_TRUE(padded_input->ensureOnDevice(DeviceId::rocm(0), stream));
        ASSERT_TRUE(isolated_input->ensureOnDevice(DeviceId::rocm(0), stream));
#else
        ASSERT_TRUE(padded_input->ensureOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(isolated_input->ensureOnDevice(DeviceId::rocm(0)));
#endif

        std::vector<std::unique_ptr<FP32Tensor>> padded_outputs;
        std::vector<std::unique_ptr<FP32Tensor>> isolated_outputs;
        std::vector<ITensorGemm::TensorProjectionDesc> padded_projections;
        std::vector<ITensorGemm::TensorProjectionDesc> isolated_projections;
        for (size_t projection = 0; projection < Ns.size(); ++projection)
        {
            padded_outputs.push_back(TestTensorFactory::createFP32(
                {static_cast<size_t>(padded_rows),
                 static_cast<size_t>(Ns[projection])}));
            isolated_outputs.push_back(TestTensorFactory::createFP32(
                {static_cast<size_t>(request_rows),
                 static_cast<size_t>(Ns[projection])}));
            ASSERT_TRUE(padded_outputs.back()->allocateOnDevice(
                DeviceId::rocm(0)));
            ASSERT_TRUE(isolated_outputs.back()->allocateOnDevice(
                DeviceId::rocm(0)));
            padded_projections.emplace_back(
                kernels[projection].get(),
                padded_outputs.back().get(),
                Ns[projection],
                nullptr,
                projection_names[projection]);
            isolated_projections.emplace_back(
                kernels[projection].get(),
                isolated_outputs.back().get(),
                Ns[projection],
                nullptr,
                projection_names[projection]);
        }

        ASSERT_TRUE(kernels.front()->multiply_fused_tensor(
            padded_input.get(), padded_projections, padded_rows, K));
        ASSERT_TRUE(kernels.front()->multiply_fused_tensor(
            isolated_input.get(), isolated_projections, request_rows, K));
#ifdef HAVE_ROCM
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
#endif

        for (size_t projection = 0; projection < Ns.size(); ++projection)
        {
            TransferEngine::publishDeviceWrite(padded_outputs[projection], DeviceId::rocm(0), stream);
            TransferEngine::publishDeviceWrite(isolated_outputs[projection], DeviceId::rocm(0), stream);
            const size_t row_width = static_cast<size_t>(Ns[projection]);
            const float *padded_request =
                padded_outputs[projection]->data() +
                static_cast<size_t>(request_row_offset) * row_width;
            const float *isolated_request =
                isolated_outputs[projection]->data();
            expectBitwiseFP32RowsEqual(
                std::string(label) + " projection=" +
                    projection_names[projection] +
                    " padded-M=" + std::to_string(padded_rows) +
                    " isolated-M=" + std::to_string(request_rows),
                padded_request,
                isolated_request,
                static_cast<size_t>(request_rows) * row_width,
                row_width);
        }

#ifdef HAVE_ROCM
        for (auto &kernel : kernels)
            kernel->clearGPUStreamBinding();
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
        for (auto &kernel : kernels)
            kernel->unbindWorkspace();
    }

#ifdef HAVE_ROCM
    template <typename CreateWeights>
    void runGraphCapturedDispatchSmallMMatchesReference(
        const char *label,
        int M,
        int N,
        int K,
        CreateWeights createWeights,
        float min_cosine)
    {
        auto weights = createWeights(
            {static_cast<size_t>(N), static_cast<size_t>(K)},
            777);
        std::vector<float> W_fp32(static_cast<size_t>(N) * K);
        weights->to_fp32(W_fp32.data());

        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));
        expectPackedPath(packed, PackedPath::NativeVNNI);

        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

        ROCmQuantisedGemmKernel kernel(&packed, 0);
        kernel.setGPUStream(stream);
        auto workspace = bindWorkspace(kernel, M, N, K);
        ASSERT_NE(workspace, nullptr);

        auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
        auto output = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});

        ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(output->allocateOnDevice(DeviceId::rocm(0)));

        ASSERT_TRUE(kernel.multiply_tensor(input.get(), output.get(), M, N, K));
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        auto captured_graph = std::make_unique<HIPGraphCapture>(stream, 0);
        ScopedBackendGraphCapture capture_transaction(
            *captured_graph,
            "runGraphCapturedDispatchSmallMMatchesReference");
        ASSERT_TRUE(capture_transaction.begin());
        const bool capture_launch_ok =
            kernel.multiply_tensor(input.get(), output.get(), M, N, K);
        capture_transaction.finish();
        ASSERT_TRUE(capture_launch_ok);
        ASSERT_TRUE(captured_graph->instantiate());

        ASSERT_EQ(hipMemsetAsync(output->gpu_data_ptr(), 0, static_cast<size_t>(M) * N * sizeof(float), stream),
                  hipSuccess);
        ASSERT_TRUE(captured_graph->launch());
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        TransferEngine::publishCurrentDeviceWrite(output, stream);

        std::vector<float> ref(static_cast<size_t>(M) * N);
        cpuFP32GemmRef(input->data(), W_fp32.data(), ref.data(), M, N, K);

        const float cos = cosineSim(output->data(), ref.data(), static_cast<size_t>(M) * N);
        LOG_INFO("[SmallM] " << label << " graph-captured M=" << M << " cosine=" << cos);
        EXPECT_GT(cos, min_cosine);

        captured_graph.reset();
        kernel.clearGPUStreamBinding();
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
        kernel.unbindWorkspace();
    }
#endif
}

TEST(Test__ROCmQuantisedGemmSmallM, DispatchQ80M2MatchesReference)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const int N = 896;
    const int K = 896;
    runDispatchSmallMMatchesReference(
        "Q8_0 INT8-VNNI",
        2,
        N,
        K,
        PackedPath::INT8VNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ8_0Random(shape, seed); },
        0.985f);
}

/**
 * @test Prove ROCm NativeVNNI active rows are byte-invariant for the complete
 * finite M-totality witness inventory and every quantized format.
 *
 * This is the ROCm half of the cross-backend graph-bucket contract. The shared
 * inventory exhausts every ordinary-prefill M through two complete 128-row
 * tiles and both open ends of every larger canonical bucket interval. Every
 * loader-supported source format travels through persistent preparation and
 * the public production GEMM entry point; no deterministic mode or row-replay
 * implementation substitutes for the optimized grouped implementation.
 *
 * Independent production M=1 launches establish the complete serial-decode
 * oracle through the largest bucket. The oracle remains device-resident while
 * it is assembled, then crosses D2H once. Every active word from every exact-M
 * and padded-bucket launch must match the corresponding serial row, including
 * exact graph-bucket thresholds and all interior rows.
 * Allocations, stream creation, preparation, and workspace binding are reused
 * for all witnesses of one format. Only result bytes are downloaded, keeping
 * the test sensitive to device arithmetic without making setup dominate the
 * integration gate.
 */
TEST(Test__ROCmQuantisedGemmSmallM, NativeVNNIPrefillActiveRowsAllFormatsByteExactAcrossMAndBuckets)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

#ifdef HAVE_ROCM
    const DeviceId device = DeviceId::rocm(0);
    uint64_t geometry_index = 0;

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
        auto input = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(maximum_bucket_rows), static_cast<size_t>(K)},
            -0.75f,
            0.75f,
            static_cast<uint32_t>(77123u + geometry_index));

        for (const auto &format : quantizedVerifierFormats())
        {
            SCOPED_TRACE(format.label);
            auto weights = format.create(
                {static_cast<size_t>(N), static_cast<size_t>(K)},
                77124u);
            ASSERT_NE(weights, nullptr);
            auto prepared = makeGpuPreparedGemm(
                weights.get(),
                device,
                std::string("test.rocm.prefill_m_invariance.") +
                    geometry.label + "." + format.label,
                ModelContextId{
                    static_cast<uint64_t>(
                        771240 + geometry_index * 100 +
                        format.source_codebook_id)});
            auto *kernel =
                dynamic_cast<ROCmQuantisedGemmKernel *>(prepared.kernel);
            ASSERT_NE(kernel, nullptr)
                << format.label << " production prepared ROCm kernel";

            hipStream_t stream = nullptr;
            ASSERT_EQ(
                hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
                hipSuccess);
            kernel->setGPUStream(stream);
            auto workspace = bindWorkspace(
                *kernel,
                maximum_bucket_rows,
                N,
                K);
            ASSERT_NE(workspace, nullptr);

            auto exact_output = TestTensorFactory::createFP32(
                {static_cast<size_t>(maximum_bucket_rows),
                 static_cast<size_t>(N)});
            auto bucket_output = TestTensorFactory::createFP32(
                {static_cast<size_t>(maximum_bucket_rows),
                 static_cast<size_t>(N)});
            auto serial_input = TestTensorFactory::createFP32(
                {1u, static_cast<size_t>(K)});
            auto serial_output = TestTensorFactory::createFP32(
                {1u, static_cast<size_t>(N)});
            auto serial_rows = TestTensorFactory::createFP32(
                {static_cast<size_t>(maximum_bucket_rows),
                 static_cast<size_t>(N)});
            ASSERT_TRUE(input->ensureOnDevice(device, stream));
            ASSERT_TRUE(exact_output->allocateOnDevice(device, stream));
            ASSERT_TRUE(bucket_output->allocateOnDevice(device, stream));
            ASSERT_TRUE(serial_input->allocateOnDevice(device, stream));
            ASSERT_TRUE(serial_output->allocateOnDevice(device, stream));
            ASSERT_TRUE(serial_rows->allocateOnDevice(device, stream));

            /*
             * Build the complete oracle from the real public M=1 dispatch.
             * Device-to-device row copies preserve exact FP32 bytes and keep
             * the entire construction ordered on the kernel stream. The one
             * terminal D2H below cannot perturb any candidate launch.
             */
            const size_t serial_row_input_bytes =
                static_cast<size_t>(K) * sizeof(float);
            const size_t serial_row_output_bytes =
                static_cast<size_t>(N) * sizeof(float);
            std::vector<float> serial_decode_rows(
                static_cast<size_t>(maximum_bucket_rows) * N);
            for (int row = 0; row < maximum_bucket_rows; ++row)
            {
                ASSERT_EQ(
                    hipMemcpyAsync(
                        serial_input->gpu_data_ptr(),
                        static_cast<const float *>(input->gpu_data_ptr()) +
                            static_cast<size_t>(row) * K,
                        serial_row_input_bytes,
                        hipMemcpyDeviceToDevice,
                        stream),
                    hipSuccess);
                TransferEngine::publishCurrentDeviceWrite(
                    serial_input.get(),
                    stream);
                ASSERT_TRUE(kernel->multiply_tensor(
                    serial_input.get(),
                    serial_output.get(),
                    1,
                    N,
                    K))
                    << "serial row=" << row;
                ASSERT_EQ(
                    hipMemcpyAsync(
                        static_cast<float *>(serial_rows->gpu_data_ptr()) +
                            static_cast<size_t>(row) * N,
                        serial_output->gpu_data_ptr(),
                        serial_row_output_bytes,
                        hipMemcpyDeviceToDevice,
                        stream),
                    hipSuccess);
            }
            ASSERT_EQ(
                hipMemcpyAsync(
                    serial_decode_rows.data(),
                    serial_rows->gpu_data_ptr(),
                    serial_decode_rows.size() * sizeof(float),
                    hipMemcpyDeviceToHost,
                    stream),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

            for (const auto &test_case : geometry.row_cases)
            {
                SCOPED_TRACE(
                    std::string("active_m=") +
                    std::to_string(test_case.active_rows) +
                    " bucket_m=" +
                    std::to_string(test_case.bucket_rows));

                ASSERT_TRUE(kernel->multiply_tensor(
                    input.get(),
                    exact_output.get(),
                    test_case.active_rows,
                    N,
                    K));
                ASSERT_TRUE(kernel->multiply_tensor(
                    input.get(),
                    bucket_output.get(),
                    test_case.bucket_rows,
                    N,
                    K));

                const size_t active_values =
                    static_cast<size_t>(test_case.active_rows) * N;
                std::vector<float> exact_host(active_values);
                std::vector<float> bucket_host(active_values);
                ASSERT_EQ(
                    hipMemcpyAsync(
                        exact_host.data(),
                        exact_output->gpu_data_ptr(),
                        active_values * sizeof(float),
                        hipMemcpyDeviceToHost,
                        stream),
                    hipSuccess);
                ASSERT_EQ(
                    hipMemcpyAsync(
                        bucket_host.data(),
                        bucket_output->gpu_data_ptr(),
                        active_values * sizeof(float),
                        hipMemcpyDeviceToHost,
                        stream),
                    hipSuccess);
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

                expectBitwiseFP32RowsEqual(
                    std::string("ROCm NativeVNNI active rows format=") +
                        format.label + " active M=" +
                        std::to_string(test_case.active_rows) +
                        " bucket M=" +
                        std::to_string(test_case.bucket_rows),
                    bucket_host.data(),
                    exact_host.data(),
                    active_values,
                    static_cast<size_t>(N));
                expectBitwiseFP32RowsEqual(
                    std::string("ROCm NativeVNNI exact M versus serial rows format=") +
                        format.label + " active M=" +
                        std::to_string(test_case.active_rows),
                    exact_host.data(),
                    serial_decode_rows.data(),
                    active_values,
                    static_cast<size_t>(N));
                expectBitwiseFP32RowsEqual(
                    std::string("ROCm NativeVNNI bucket active rows versus serial M=1 format=") +
                        format.label + " active M=" +
                        std::to_string(test_case.active_rows),
                    bucket_host.data(),
                    serial_decode_rows.data(),
                    active_values,
                    static_cast<size_t>(N));
            }

            kernel->unbindWorkspace();
            kernel->clearGPUStreamBinding();
            ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
        }
        ++geometry_index;
    }
#endif
}

/**
 * @test Prove a scalar GEMM executes without the fused-projection arena.
 *
 * The public workspace declaration intentionally reserves
 * `ROCM_SCATTER_PARTIAL_BATCHED` only after a graph declares a concurrent
 * projection bundle. The kernel's binding validator must honor that same
 * contract; otherwise standalone decode and LM-head callers receive a plan
 * that the kernel itself rejects before launch.
 */
TEST(Test__ROCmQuantisedGemmSmallM,
     StandaloneWorkspaceContractDoesNotRequireFusedProjectionArena)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    constexpr int M = 1;
    constexpr int N = 128;
    constexpr int K = 256;
    auto weights = TestTensorFactory::createQ4_0Random({N, K}, /*seed=*/2811);
    ROCmPackedWeights packed;
    ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));
    ROCmQuantisedGemmKernel kernel(&packed, /*rocm_device_id=*/0);

#ifdef HAVE_ROCM
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    kernel.setGPUStream(stream);
#endif

    const WorkspaceRequirements requirements =
        kernel.getWorkspaceRequirements(M, N, K);
    EXPECT_EQ(
        requirements.find(GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL_BATCHED),
        nullptr);
    auto workspace = bindWorkspace(kernel, M, N, K);
    ASSERT_NE(workspace, nullptr);

    auto input = TestTensorFactory::createFP32Random({M, K});
    auto output = TestTensorFactory::createFP32({M, N});
    ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
    ASSERT_TRUE(output->allocateOnDevice(DeviceId::rocm(0)));
    ASSERT_TRUE(kernel.multiply_tensor(input.get(), output.get(), M, N, K));
#ifdef HAVE_ROCM
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
#endif

    kernel.unbindWorkspace();
#ifdef HAVE_ROCM
    kernel.clearGPUStreamBinding();
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
}

/**
 * @test Prove every quantized source format through production preparation and dispatch.
 *
 * Each M=2..16 grouped call uses the persistent device repack pipeline and the
 * public ITensorGemm entry point, then compares raw FP32 output bytes against M
 * independent production M=1 calls on the same prepared kernel.  Perfstats are
 * part of the acceptance contract: byte equality without one grouped small-M
 * launch per format/depth could be satisfied by serial replay.
 */
TEST(Test__ROCmQuantisedGemmSmallM, ProductionPreparedAllQuantizedFormatsRuntimeMMatchSerialDecodeBytes)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    constexpr int N = 384;
    constexpr int K = 512;
    for (const auto &format : quantizedVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        for (const int M : kGroupedVerifierRuntimeRows)
        {
            runGroupedSmallMMatchesSerialRows(
                format.label,
                M,
                N,
                K,
                format.create);
        }
    }

    uint64_t grouped_calls = 0;
    uint64_t canonical_raw_int8_calls = 0;
    for (const auto &record : PerfStatsCollector::snapshot(
             {"kernel.rocm_native_vnni_small_m_calls"}))
    {
        if (record.domain != "kernel" ||
            record.name != "rocm_native_vnni_small_m_calls" ||
            record.kind != PerfStatRecord::Kind::Counter)
        {
            continue;
        }
        ASSERT_EQ(record.tags.at("n"), std::to_string(N));
        ASSERT_EQ(record.tags.at("k"), std::to_string(K));
        const int m = std::stoi(record.tags.at("m"));
        ASSERT_GE(m, kGroupedVerifierRuntimeRows.front());
        ASSERT_LE(m, kGroupedVerifierRuntimeRows.back());
        grouped_calls += record.count;
        if (record.tags.at("codebook") == "19")
            canonical_raw_int8_calls += record.count;
    }

    EXPECT_EQ(
        grouped_calls,
        quantizedVerifierFormats().size() * kGroupedVerifierRuntimeRows.size())
        << "Every canonical quantized format and certified runtime M must execute "
           "one economical ROCm grouped small-M kernel\n"
        << PerfStatsCollector::summaryString(
               {"kernel.rocm_native_vnni_small_m_calls"}, 100);
    EXPECT_EQ(
        canonical_raw_int8_calls,
        3u * kGroupedVerifierRuntimeRows.size())
        << "Q8_0, Q8_1, and Q8_K must each use codebook 19 at every certified M";

    PerfStatsCollector::reset();
}

/**
 * @test Prove production serial M=1 decode publishes its resolved route.
 *
 * Strong grouped-decode training uses independent production M=1 launches as
 * the bitwise oracle.  The oracle is admissible only when perfstats proves the
   * launch selected the KB and target-wave policy returned by the production
 * serial resolver.  This focused regression prevents M=1 telemetry from
   * disappearing while grouped runtime-M telemetry remains healthy.
 */
TEST(Test__ROCmQuantisedGemmSmallM, ProductionSerialM1RouteTelemetryMatchesResolver)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    ScopedEnv automatic_kb("LLAMINAR_ROCM_NVNNI_GEMV_KB", "-1");
    ScopedEnv automatic_waves("LLAMINAR_ROCM_NVNNI_GEMV_TARGET_WAVES", "-1");
    rocmGemv_native_vnni_reset_tuning_overrides();
    PerfStatsCollector::reset();

    constexpr uint8_t Q8_CODEBOOK = 19;
    constexpr int M = 2;
    constexpr int N = 512;
    constexpr int K = 2048;
    int expected_kb = 0;
    int expected_target_waves = 0;
    ASSERT_TRUE(rocmGemv_native_vnni_query_serial_m1_config(
        Q8_CODEBOOK,
        N,
        K,
        &expected_kb,
        &expected_target_waves));

    runGroupedSmallMMatchesSerialRows(
        "Q8_0 serial-M1 route telemetry",
        M,
        N,
        K,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ8_0Random(shape, seed); });

    uint64_t matching_serial_launches = 0;
    for (const auto &record : PerfStatsCollector::snapshot(
             {"kernel.rocm_native_vnni_small_m_launch"}))
    {
        if (record.domain != "kernel" ||
            record.name != "rocm_native_vnni_small_m_launch" ||
            record.kind != PerfStatRecord::Kind::Counter ||
            record.tags.count("m") == 0 ||
            record.tags.at("m") != "1" ||
            record.tags.count("n") == 0 ||
            record.tags.at("n") != std::to_string(N) ||
            record.tags.count("k") == 0 ||
            record.tags.at("k") != std::to_string(K) ||
            record.tags.count("codebook") == 0 ||
            record.tags.at("codebook") != std::to_string(Q8_CODEBOOK))
        {
            continue;
        }

        ASSERT_EQ(record.tags.at("kb"), std::to_string(expected_kb));
        ASSERT_EQ(
            record.tags.at("target_waves_per_cu"),
            std::to_string(expected_target_waves));
        ASSERT_EQ(record.tags.at("batched"), "false");
        ASSERT_EQ(record.tags.at("projections"), "1");
        matching_serial_launches += record.count;
    }

    EXPECT_EQ(matching_serial_launches, static_cast<uint64_t>(M))
        << "Each independent production M=1 oracle row must publish the exact "
           "serial resolver route\n"
        << PerfStatsCollector::summaryString(
               {"kernel.rocm_native_vnni_small_m_launch"}, 20);

    PerfStatsCollector::reset();
}

/**
 * @test Mirrored Qwen3.6 LM-head dispatch inherits serial LocalTP geometry.
 *
 * Qwen3.6-35B-A3B uses a 2048-wide hidden state and an IQ3_S LM head.  Keep
 * this sentinel on that exact production geometry.  The current learned table
 * happens to choose the same launch for its full and half-vocabulary widths,
 * so this case proves the production contract without manufacturing a false
 * requirement that those two policy entries differ.
 */
TEST(Test__ROCmQuantisedGemmSmallM,
     MirroredQwen36IQ3SLMHeadUsesSerialShardArithmeticPolicy)
{
    constexpr uint8_t IQ3_S_CODEBOOK = 11;
    constexpr int FULL_N = 248320;
    constexpr int SERIAL_PARTITION_N = 124160;
    constexpr int K = 2048;

    int serial_kb = 0;
    int serial_waves = 0;
    ASSERT_TRUE(rocmGemv_native_vnni_query_serial_m1_config(
        IQ3_S_CODEBOOK,
        SERIAL_PARTITION_N,
        K,
        &serial_kb,
        &serial_waves));

    int mirrored_kb = 0;
    int mirrored_waves = 0;
    {
        ScopedROCmSerialPartitionPolicy serial_partition_scope(
            SERIAL_PARTITION_N);
        ASSERT_TRUE(rocmGemv_native_vnni_query_serial_m1_config(
            IQ3_S_CODEBOOK,
            FULL_N,
            K,
            &mirrored_kb,
            &mirrored_waves));
    }

    EXPECT_EQ(mirrored_kb, serial_kb);
    EXPECT_EQ(mirrored_waves, serial_waves);
    EXPECT_EQ(rocmGemv_native_vnni_get_serial_partition_n(), 0);
}

/**
 * @test The serial-partition scope overrides a genuinely distinct policy key.
 *
 * This companion uses the Qwen3.6 dense hidden width, where the generated
 * table currently distinguishes full- and half-vocabulary geometry.  It proves
 * that the scope changes selector lookup rather than merely restoring a
 * thread-local integer after an otherwise identical production selection.
 */
TEST(Test__ROCmQuantisedGemmSmallM,
     MirroredLMHeadSerialPartitionOverridesDistinctGeneratedPolicy)
{
    constexpr uint8_t IQ3_S_CODEBOOK = 11;
    constexpr int FULL_N = 248320;
    constexpr int SERIAL_PARTITION_N = 124160;
    constexpr int K = 5120;

    int serial_kb = 0;
    int serial_waves = 0;
    ASSERT_TRUE(rocmGemv_native_vnni_query_serial_m1_config(
        IQ3_S_CODEBOOK,
        SERIAL_PARTITION_N,
        K,
        &serial_kb,
        &serial_waves));

    int unconstrained_kb = 0;
    int unconstrained_waves = 0;
    ASSERT_TRUE(rocmGemv_native_vnni_query_serial_m1_config(
        IQ3_S_CODEBOOK,
        FULL_N,
        K,
        &unconstrained_kb,
        &unconstrained_waves));
    ASSERT_TRUE(
        unconstrained_kb != serial_kb ||
        unconstrained_waves != serial_waves)
        << "The distinct-policy sentinel must be updated when generated "
           "dispatch converges for this geometry";

    int mirrored_kb = 0;
    int mirrored_waves = 0;
    {
        ScopedROCmSerialPartitionPolicy serial_partition_scope(
            SERIAL_PARTITION_N);
        ASSERT_TRUE(rocmGemv_native_vnni_query_serial_m1_config(
            IQ3_S_CODEBOOK,
            FULL_N,
            K,
            &mirrored_kb,
            &mirrored_waves));
    }

    EXPECT_EQ(mirrored_kb, serial_kb);
    EXPECT_EQ(mirrored_waves, serial_waves);
    EXPECT_EQ(rocmGemv_native_vnni_get_serial_partition_n(), 0);
}

/**
 * @test Prove explicit trainer candidates control the IQ-grid LDS launch.
 *
 * IQ-grid decode uses a dedicated LDS-cached kernel once geometry provides
 * enough parallelism.  Candidate sweeps must still control that kernel's K
 * partition and target occupancy; otherwise many nominally different corpus
 * rows measure the same hidden heuristic launch.  This test deliberately uses
 * a K-heavy IQ2_S shape that takes the LDS branch and validates the effective
 * production launch through perfstats telemetry.
 */
TEST(Test__ROCmQuantisedGemmSmallM, ExplicitIQGridTrainerCandidateControlsLDSRoute)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

#ifndef HAVE_ROCM
    GTEST_SKIP() << "HAVE_ROCM not enabled";
#else
    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    constexpr uint8_t IQ2_S_CODEBOOK = 13;
    constexpr int M = 1;
    constexpr int N = 1536;
    constexpr int K = 6144;
    constexpr int FORCED_KB = 3;
    constexpr int FORCED_TARGET_WAVES = 12;
    ScopedROCmNativeVNNITuningOverride tuning(
        FORCED_KB,
        FORCED_TARGET_WAVES);

    auto weights = TestTensorFactory::createIQ2_SRandom(
        {static_cast<size_t>(N), static_cast<size_t>(K)},
        7319);
    auto prepared = makeGpuPreparedGemm(
        weights.get(),
        DeviceId::rocm(0),
        "test.rocm.native_vnni.iq_lds_explicit_candidate",
        ModelContextId{7319});
    auto *kernel = dynamic_cast<ROCmQuantisedGemmKernel *>(prepared.kernel);
    ASSERT_NE(kernel, nullptr);

    hipStream_t stream = nullptr;
    ASSERT_EQ(
        hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
        hipSuccess);
    kernel->setGPUStream(stream);
    auto workspace = bindWorkspace(*kernel, M, N, K);
    ASSERT_NE(workspace, nullptr);

    auto input = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(M), static_cast<size_t>(K)},
        -0.35f,
        0.35f,
        7320);
    auto output = TestTensorFactory::createFP32(
        {static_cast<size_t>(M), static_cast<size_t>(N)});
    ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0), stream));
    ASSERT_TRUE(output->allocateOnDevice(DeviceId::rocm(0)));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    PerfStatsCollector::reset();
    ASSERT_TRUE(kernel->multiply_tensor(input.get(), output.get(), M, N, K));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    uint64_t matching_launches = 0;
    for (const auto &record : PerfStatsCollector::snapshot(
             {"kernel.rocm_native_vnni_small_m_launch"}))
    {
        if (record.domain != "kernel" ||
            record.name != "rocm_native_vnni_small_m_launch" ||
            record.kind != PerfStatRecord::Kind::Counter ||
            record.tags.at("m") != std::to_string(M) ||
            record.tags.at("n") != std::to_string(N) ||
            record.tags.at("k") != std::to_string(K) ||
            record.tags.at("codebook") !=
                std::to_string(IQ2_S_CODEBOOK))
        {
            continue;
        }

        EXPECT_EQ(record.tags.at("kb"), std::to_string(FORCED_KB));
        EXPECT_EQ(
            record.tags.at("target_waves_per_cu"),
            std::to_string(FORCED_TARGET_WAVES));
        EXPECT_EQ(record.tags.at("path"), "split_reduce");
        EXPECT_EQ(record.tags.at("grid_n"), "6")
            << "N=1536 must take the 256-column IQ LDS launch geometry";
        matching_launches += record.count;
    }
    EXPECT_EQ(matching_launches, 1u)
        << "The explicit IQ2_S candidate must produce one observable LDS launch\n"
        << PerfStatsCollector::summaryString(
               {"kernel.rocm_native_vnni_small_m_launch"}, 20);

    TransferEngine::publishDeviceWrite(output, DeviceId::rocm(0), stream);
    EXPECT_GT(l2Norm(output->data(), output->numel()), 1.0e-7)
        << "The observed candidate must produce a non-zero output witness";

    kernel->clearGPUStreamBinding();
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
    kernel->unbindWorkspace();
    PerfStatsCollector::reset();
#endif
}

TEST(Test__ROCmQuantisedGemmSmallM, DispatchQ4KM2MatchesReference)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const int N = 896;
    const int K = 1024;
    runDispatchSmallMMatchesReference(
        "Q4_K native-VNNI",
        2,
        N,
        K,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.985f);
}

TEST(Test__ROCmQuantisedGemmSmallM, DispatchQ4KGroupedVerifierRowsMatchSerialDecode)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    /*
     * Qwen3.6 attention output projection shape. This catches grouped verifier
     * drift that loose FP32-reference checks can miss: grouped runtime-M must agree
     * with the same packed ROCm native-VNNI path replayed one row at a time.
     */
    constexpr int N = 5120;
    constexpr int K = 5120;
    for (const int M : kGroupedVerifierBoundaryRows)
    {
        runGroupedSmallMMatchesSerialRows(
            "Q4_K native-VNNI Qwen3.6 attention Wo",
            M,
            N,
            K,
            [](const std::vector<size_t> &shape, uint32_t seed)
            { return TestTensorFactory::createQ4_KRandom(shape, seed); });
    }
}

TEST(Test__ROCmQuantisedGemmSmallM, DispatchQ5KGroupedVerifierRowsMatchSerialDecode)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    /*
     * Qwen3.6 GDN output projection shape. Dense verifier replay promotes this
     * family through the same Wo helper as full-attention Q4_K, so it needs its
     * own grouped-vs-serial proof instead of riding on the Q4_K regression.
     */
    constexpr int N = 5120;
    constexpr int K = 6144;
    for (const int M : kGroupedVerifierBoundaryRows)
    {
        runGroupedSmallMMatchesSerialRows(
            "Q5_K native-VNNI Qwen3.6 GDN output",
            M,
            N,
            K,
            [](const std::vector<size_t> &shape, uint32_t seed)
            { return TestTensorFactory::createQ5_KRandom(shape, seed); });
    }
}

TEST(Test__ROCmQuantisedGemmSmallM, DispatchQ6KGroupedVerifierRowsMatchSerialDecode)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    /*
     * Qwen3.6's LM head is Q6_K with K=d_model.  The production vocab dimension
     * is intentionally huge, so this regression keeps N bounded while still
     * exercising the Q6_K grouped runtime-M NativeVNNI route with the real K.
     */
    constexpr int N = 16384;
    constexpr int K = 5120;
    for (const int M : kGroupedVerifierBoundaryRows)
    {
        runGroupedSmallMMatchesSerialRows(
            "Q6_K native-VNNI Qwen3.6 LM-head-like",
            M,
            N,
            K,
            [](const std::vector<size_t> &shape, uint32_t seed)
            { return TestTensorFactory::createQ6_KRandom(shape, seed); });
    }
}

TEST(Test__ROCmQuantisedGemmSmallM, LocalTPWoAllQuantizedFormatsGroupedVerifierRowsMatchSerialDecodeStrict)
{
    if (rocmDeviceCount() < 2)
        GTEST_SKIP() << "ROCm LocalTP Wo regression requires at least two ROCm devices";
    ScopedEnv perf_stats("LLAMINAR_PERF_STATS_SUMMARY", "1");

    /*
     * Qwen3.6 MoE LocalTP splits the attention output projection along the
     * input/head dimension and allreduces two [M, hidden] partial sums.  This
     * sweep proves the grouped verifier projection plus reconstructed allreduce
     * result is byte-identical to rowwise M=1 decode for every ROCm native
     * quantized tensor format, not just the IQ3_S model file that exposed the
     * latest full-parity drift.
     */
    const std::vector<DeviceId> devices = {DeviceId::rocm(0), DeviceId::rocm(1)};

    uint32_t seed = 8800;
    for (const auto &shape_case : localTPWoShapeCases())
    {
        for (const auto &format_case : localTPWoQuantizedFormatCases())
        {
            runLocalTPWoAllreduceVerifierRowsMatchSerial(
                format_case,
                devices,
                shape_case,
                shape_case.hidden,
                shape_case.local_attention_dim,
                shape_case.full_attention_dim,
                seed);
            seed += 101;
        }
    }
}

TEST(Test__ROCmQuantisedGemmSmallM, LocalTPWoFloatingPointFormatsGroupedVerifierRowsMatchSerialDecodeStrict)
{
    if (rocmDeviceCount() < 2)
        GTEST_SKIP() << "ROCm LocalTP Wo regression requires at least two ROCm devices";
    ScopedEnv perf_stats("LLAMINAR_PERF_STATS_SUMMARY", "1");

    /*
     * Floating-point prepared GEMM goes through rocBLAS/hipBLAS-style kernels
     * rather than the native-VNNI quantized dispatch.  Keep it in the same
     * LocalTP reconstruction gate so FP32, FP16, and BF16 verifier rows cannot
     * silently drift from serial decode when models use floating-point weights.
     */
    const std::vector<DeviceId> devices = {DeviceId::rocm(0), DeviceId::rocm(1)};

    uint32_t seed = 10800;
    for (const auto &shape_case : localTPWoShapeCases())
    {
        for (const auto &format_case : localTPWoFloatingPointFormatCases())
        {
            runLocalTPWoAllreduceVerifierRowsMatchSerial(
                format_case,
                devices,
                shape_case,
                shape_case.hidden,
                shape_case.local_attention_dim,
                shape_case.full_attention_dim,
                seed);
            seed += 101;
        }
    }
}

TEST(Test__ROCmQuantisedGemmSmallM, ReplicatedWoAllQuantizedFormatsGroupedVerifierRowsMatchSerialDecodeStrict)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "ROCm replicated Wo regression requires a ROCm device";
    ScopedEnv perf_stats("LLAMINAR_PERF_STATS_SUMMARY", "1");

    /*
     * Mirroring the complete attention Wo/MTP verifier projection on every
     * LocalTP shard is the target economical path for tiny verifier row groups.
     * This sweep proves that route for every ROCm native quantized format,
     * independently of the existing row-parallel allreduce reconstruction test.
     */
    const DeviceId device = DeviceId::rocm(0);

    uint32_t seed = 12800;
    for (const auto &shape_case : localTPWoShapeCases())
    {
        for (const auto &format_case : localTPWoQuantizedFormatCases())
        {
            runReplicatedWoVerifierRowsMatchSerial(
                format_case,
                device,
                shape_case,
                seed);
            seed += 101;
        }
    }
}

TEST(Test__ROCmQuantisedGemmSmallM, ReplicatedWoFloatingPointFormatsGroupedVerifierRowsMatchSerialDecodeStrict)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "ROCm replicated Wo regression requires a ROCm device";
    ScopedEnv perf_stats("LLAMINAR_PERF_STATS_SUMMARY", "1");

    /*
     * Floating-point mirrored Wo must also use the decode-equivalent grouped
     * verifier implementation.  Backend GEMM libraries are free to choose
     * shape-dependent reductions, so FP32/FP16/BF16 stay in the byte-equality
     * matrix instead of relying on relaxed numeric tolerances.
     */
    const DeviceId device = DeviceId::rocm(0);

    uint32_t seed = 14800;
    for (const auto &shape_case : localTPWoShapeCases())
    {
        for (const auto &format_case : localTPWoFloatingPointFormatCases())
        {
            runReplicatedWoVerifierRowsMatchSerial(
                format_case,
                device,
                shape_case,
                seed);
            seed += 101;
        }
    }
}

/**
 * @test Prove every floating ROCm projection format uses grouped verifier math.
 *
 * FP32, FP16, and BF16 weights do not enter the NativeVNNI codebook kernels.
 * They are uploaded through the production RAW_FP weight path and dispatched to
 * the ROCm floating GEMM implementation.  The verifier API must still execute
 * both projections as one economical device-resident group and preserve the
 * exact per-row accumulation order used by ordinary M=1 decode.
 *
 * This regression intentionally checks the backend route counters in addition
 * to byte equality.  Calling two serial GEMMs into an M-row tensor could produce
 * the same bytes while silently defeating the grouped MTP implementation.
 */
TEST(Test__ROCmQuantisedGemmSmallM, FloatingProjectionAllFormatsRuntimeMMatchSerialDecodeStrict)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

#ifndef HAVE_ROCM
    GTEST_SKIP() << "HAVE_ROCM not enabled";
#else
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    constexpr int K = 192;
    constexpr int N = 80;
    constexpr auto verifier_rows = kGroupedVerifierRuntimeRows;
    const DeviceId device = DeviceId::rocm(0);

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    ASSERT_NE(stream, nullptr);

    uint64_t model_id = 181000;
    for (const auto &format : localTPWoFloatingPointFormatCases())
    {
        SCOPED_TRACE(format.label);

        auto alpha_weights = format.create(
            {static_cast<size_t>(N), static_cast<size_t>(K)},
            static_cast<uint32_t>(model_id + 1));
        auto beta_weights = format.create(
            {static_cast<size_t>(N), static_cast<size_t>(K)},
            static_cast<uint32_t>(model_id + 2));
        ASSERT_NE(alpha_weights, nullptr);
        ASSERT_NE(beta_weights, nullptr);

        const TensorType dtype = alpha_weights->native_type();
        ASSERT_EQ(beta_weights->native_type(), dtype);
        const char *dtype_tag = tensorTypeName(dtype);

        auto alpha_prepared = makeGpuPreparedFloatingPointGemm(
            alpha_weights.get(),
            device,
            std::string("test.rocm.floating_grouped.alpha.") + dtype_tag,
            ModelContextId{model_id++});
        auto beta_prepared = makeGpuPreparedFloatingPointGemm(
            beta_weights.get(),
            device,
            std::string("test.rocm.floating_grouped.beta.") + dtype_tag,
            ModelContextId{model_id++});
        ASSERT_NE(alpha_prepared.kernel, nullptr);
        ASSERT_NE(beta_prepared.kernel, nullptr);

        std::array<ITensorGemm *, 2> kernels = {
            alpha_prepared.kernel,
            beta_prepared.kernel};
        WorkspaceRequirements requirements;
        for (ITensorGemm *kernel : kernels)
        {
            kernel->setGPUStream(stream);
            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
            ASSERT_NE(consumer, nullptr)
                << dtype_tag << " floating GEMM must declare workspace ownership";
            requirements.merge(consumer->getWorkspaceRequirements(
                kGroupedVerifierRuntimeRows.back(),
                N,
                K));
        }
        const std::array<int, 2> grouped_widths{N, N};
        auto *group_consumer =
            dynamic_cast<IWorkspaceConsumer *>(kernels.front());
        ASSERT_NE(group_consumer, nullptr);
        group_consumer->appendFusedProjectionWorkspaceRequirements(
            requirements,
            kGroupedVerifierRuntimeRows.back(),
            grouped_widths,
            K);

        DeviceWorkspaceManager workspace(
            device,
            requirements.total_bytes_with_alignment() + 64 * 1024 * 1024);
        ASSERT_TRUE(workspace.allocate(requirements));
        for (ITensorGemm *kernel : kernels)
        {
            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
            ASSERT_NE(consumer, nullptr);
            consumer->bindWorkspace(&workspace);
        }

        for (const int M : verifier_rows)
        {
            SCOPED_TRACE(std::string(dtype_tag) + " M=" + std::to_string(M));

            auto grouped_input = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(M), static_cast<size_t>(K)},
                -0.35f,
                0.35f,
                static_cast<uint32_t>(182000 + M + model_id));
            const std::vector<float> input_values(
                grouped_input->data(),
                grouped_input->data() + grouped_input->numel());
            auto alpha_grouped = TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(N)});
            auto beta_grouped = TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(N)});
            ASSERT_TRUE(grouped_input->ensureOnDevice(device, stream));
            ASSERT_TRUE(alpha_grouped->allocateOnDevice(device, stream));
            ASSERT_TRUE(beta_grouped->allocateOnDevice(device, stream));

            std::vector<ITensorGemm::TensorProjectionDesc> projections = {
                {alpha_prepared.kernel, alpha_grouped.get(), N, nullptr, "alpha"},
                {beta_prepared.kernel, beta_grouped.get(), N, nullptr, "beta"}};
            ASSERT_TRUE(alpha_prepared.kernel->multiply_fused_verifier_rows_decode_equivalent(
                grouped_input.get(), projections, M, K, nullptr, &workspace))
                << dtype_tag << " grouped floating projection M=" << M;
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            TransferEngine::publishDeviceWrite(alpha_grouped, device, stream);
            TransferEngine::publishDeviceWrite(beta_grouped, device, stream);

            const std::vector<float> alpha_grouped_values(
                alpha_grouped->data(),
                alpha_grouped->data() + alpha_grouped->numel());
            const std::vector<float> beta_grouped_values(
                beta_grouped->data(),
                beta_grouped->data() + beta_grouped->numel());
            std::vector<float> alpha_serial_values(
                static_cast<size_t>(M) * static_cast<size_t>(N));
            std::vector<float> beta_serial_values(
                static_cast<size_t>(M) * static_cast<size_t>(N));

            for (int row = 0; row < M; ++row)
            {
                auto row_input = TestTensorFactory::createFP32(
                    {1u, static_cast<size_t>(K)});
                std::copy_n(
                    input_values.data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                    K,
                    row_input->mutable_data());
                auto alpha_serial = TestTensorFactory::createFP32(
                    {1u, static_cast<size_t>(N)});
                auto beta_serial = TestTensorFactory::createFP32(
                    {1u, static_cast<size_t>(N)});
                ASSERT_TRUE(row_input->ensureOnDevice(device, stream));
                ASSERT_TRUE(alpha_serial->allocateOnDevice(device, stream));
                ASSERT_TRUE(beta_serial->allocateOnDevice(device, stream));

                ASSERT_TRUE(alpha_prepared.kernel->multiply_tensor(
                    row_input.get(), alpha_serial.get(),
                    1, N, K,
                    /*transpose_B=*/true,
                    1.0f,
                    0.0f,
                    nullptr,
                    nullptr,
                    -1,
                    &workspace));
                ASSERT_TRUE(beta_prepared.kernel->multiply_tensor(
                    row_input.get(), beta_serial.get(),
                    1, N, K,
                    /*transpose_B=*/true,
                    1.0f,
                    0.0f,
                    nullptr,
                    nullptr,
                    -1,
                    &workspace));
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
                TransferEngine::publishDeviceWrite(alpha_serial, device, stream);
                TransferEngine::publishDeviceWrite(beta_serial, device, stream);
                std::copy_n(
                    alpha_serial->data(),
                    N,
                    alpha_serial_values.data() + static_cast<size_t>(row) * static_cast<size_t>(N));
                std::copy_n(
                    beta_serial->data(),
                    N,
                    beta_serial_values.data() + static_cast<size_t>(row) * static_cast<size_t>(N));
            }

            expectBitwiseFP32RowsEqual(
                std::string("ROCm ") + dtype_tag +
                    " grouped alpha projection M=" + std::to_string(M),
                alpha_grouped_values.data(),
                alpha_serial_values.data(),
                alpha_grouped_values.size(),
                static_cast<size_t>(N));
            expectBitwiseFP32RowsEqual(
                std::string("ROCm ") + dtype_tag +
                    " grouped beta projection M=" + std::to_string(M),
                beta_grouped_values.data(),
                beta_serial_values.data(),
                beta_grouped_values.size(),
                static_cast<size_t>(N));
        }

        for (ITensorGemm *kernel : kernels)
        {
            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
            ASSERT_NE(consumer, nullptr);
            consumer->unbindWorkspace();
            kernel->clearGPUStreamBinding();
        }
    }

    std::array<uint64_t, kGroupedVerifierRuntimeRows.back() + 1>
        fp32_grouped_calls{};
    std::array<uint64_t, kGroupedVerifierRuntimeRows.back() + 1>
        fp16_grouped_calls{};
    std::array<uint64_t, kGroupedVerifierRuntimeRows.back() + 1>
        bf16_grouped_calls{};
    const auto records = PerfStatsCollector::snapshot(
        {"kernel.rocm_fp32_small_n_batched_projection_calls",
         "kernel.rocm_fp32_batched_projection_calls",
         "kernel.rocm_fp32x16_grouped_verifier_projection_calls"});
    for (const auto &record : records)
    {
        auto tag_equals = [&](const char *name, const std::string &value)
        {
            const auto it = record.tags.find(name);
            return it != record.tags.end() && it->second == value;
        };
        const auto m_it = record.tags.find("m");
        if (m_it == record.tags.end())
        {
            continue;
        }
        const int M = std::stoi(m_it->second);
        if (M < verifier_rows.front() || M > verifier_rows.back())
            continue;

        if ((record.name == "rocm_fp32_small_n_batched_projection_calls" ||
             record.name == "rocm_fp32_batched_projection_calls") &&
            tag_equals("n", std::to_string(N)) &&
            tag_equals("k", std::to_string(K)) &&
            tag_equals("batch", "2"))
        {
            fp32_grouped_calls[static_cast<size_t>(M)] += record.count;
        }
        else if (record.name == "rocm_fp32x16_grouped_verifier_projection_calls" &&
                 tag_equals("n", std::to_string(N)) &&
                 tag_equals("k", std::to_string(K)) &&
                 tag_equals("projections", "2") &&
                 tag_equals("route", "fixed_order_fp32x16_batched_projection"))
        {
            if (tag_equals("dtype", "fp16"))
                fp16_grouped_calls[static_cast<size_t>(M)] += record.count;
            else if (tag_equals("dtype", "bf16"))
                bf16_grouped_calls[static_cast<size_t>(M)] += record.count;
        }
    }
    for (const int M : verifier_rows)
    {
        EXPECT_EQ(fp32_grouped_calls[static_cast<size_t>(M)], 1u)
            << "ROCm FP32 verifier projections must use one batched projection at M=" << M;
        EXPECT_EQ(fp16_grouped_calls[static_cast<size_t>(M)], 1u)
            << "ROCm FP16 verifier projections must use one fixed-order grouped launch at M=" << M;
        EXPECT_EQ(bf16_grouped_calls[static_cast<size_t>(M)], 1u)
            << "ROCm BF16 verifier projections must use one fixed-order grouped launch at M=" << M;
    }

    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
    PerfStatsCollector::reset();
#endif
}

/**
 * @test Prove floating GDN gates are invariant to padded request batching.
 *
 * Dense Qwen 3.6 emits 48 alpha and beta values per token from FP32 weights.
 * A short request therefore moves from M=11 in isolation to M=32 inside a
 * two-request padded prefill.  Exercise the ordinary fused production API for
 * every floating weight storage format and require the short request's eleven
 * rows to retain identical FP32 bytes.
 */
TEST(
    Test__ROCmQuantisedGemmSmallM,
    FloatingGDNProjectionAllFormatsPaddedPrefillMatchesIsolatedRequestBytes)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

#ifndef HAVE_ROCM
    GTEST_SKIP() << "HAVE_ROCM not enabled";
#else
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    constexpr int padded_rows = 32;
    constexpr int request_row_offset = 16;
    constexpr int request_rows = 11;
    constexpr int K = 5120;
    constexpr int N = 48;
    const DeviceId device = DeviceId::rocm(0);

    hipStream_t stream = nullptr;
    ASSERT_EQ(
        hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
        hipSuccess);
    ASSERT_NE(stream, nullptr);

    uint64_t model_id = 187000;
    for (const auto &format : localTPWoFloatingPointFormatCases())
    {
        SCOPED_TRACE(format.label);
        auto alpha_weights = format.create(
            {static_cast<size_t>(N), static_cast<size_t>(K)},
            static_cast<uint32_t>(model_id + 1));
        auto beta_weights = format.create(
            {static_cast<size_t>(N), static_cast<size_t>(K)},
            static_cast<uint32_t>(model_id + 2));
        ASSERT_NE(alpha_weights, nullptr);
        ASSERT_NE(beta_weights, nullptr);

        const TensorType dtype = alpha_weights->native_type();
        ASSERT_EQ(beta_weights->native_type(), dtype);
        const char *dtype_tag = tensorTypeName(dtype);
        auto alpha_prepared = makeGpuPreparedFloatingPointGemm(
            alpha_weights.get(),
            device,
            std::string("test.rocm.padded_gdn.alpha.") + dtype_tag,
            ModelContextId{model_id++});
        auto beta_prepared = makeGpuPreparedFloatingPointGemm(
            beta_weights.get(),
            device,
            std::string("test.rocm.padded_gdn.beta.") + dtype_tag,
            ModelContextId{model_id++});
        ASSERT_NE(alpha_prepared.kernel, nullptr);
        ASSERT_NE(beta_prepared.kernel, nullptr);

        std::array<ITensorGemm *, 2> kernels{
            alpha_prepared.kernel,
            beta_prepared.kernel,
        };
        WorkspaceRequirements requirements;
        for (ITensorGemm *kernel : kernels)
        {
            kernel->setGPUStream(stream);
            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
            ASSERT_NE(consumer, nullptr);
            requirements.merge(consumer->getWorkspaceRequirements(
                padded_rows, N, K));
            requirements.merge(consumer->getWorkspaceRequirements(
                request_rows, N, K));
        }
        const std::array<int, 2> grouped_widths{N, N};
        auto *group_consumer =
            dynamic_cast<IWorkspaceConsumer *>(kernels.front());
        ASSERT_NE(group_consumer, nullptr);
        group_consumer->appendFusedProjectionWorkspaceRequirements(
            requirements, padded_rows, grouped_widths, K);
        group_consumer->appendFusedProjectionWorkspaceRequirements(
            requirements, request_rows, grouped_widths, K);

        DeviceWorkspaceManager workspace(
            device,
            requirements.total_bytes_with_alignment() + 64 * 1024 * 1024);
        ASSERT_TRUE(workspace.allocate(requirements));
        for (ITensorGemm *kernel : kernels)
        {
            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
            ASSERT_NE(consumer, nullptr);
            consumer->bindWorkspace(&workspace);
        }

        auto padded_input = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(padded_rows), static_cast<size_t>(K)},
            -0.35f,
            0.35f,
            static_cast<uint32_t>(188000 + model_id));
        auto isolated_input = TestTensorFactory::createFP32(
            {static_cast<size_t>(request_rows), static_cast<size_t>(K)});
        std::copy_n(
            padded_input->data() +
                static_cast<size_t>(request_row_offset) *
                    static_cast<size_t>(K),
            static_cast<size_t>(request_rows) * static_cast<size_t>(K),
            isolated_input->mutable_data());
        ASSERT_TRUE(padded_input->ensureOnDevice(device, stream));
        ASSERT_TRUE(isolated_input->ensureOnDevice(device, stream));

        auto alpha_padded = TestTensorFactory::createFP32(
            {static_cast<size_t>(padded_rows), static_cast<size_t>(N)});
        auto beta_padded = TestTensorFactory::createFP32(
            {static_cast<size_t>(padded_rows), static_cast<size_t>(N)});
        auto alpha_isolated = TestTensorFactory::createFP32(
            {static_cast<size_t>(request_rows), static_cast<size_t>(N)});
        auto beta_isolated = TestTensorFactory::createFP32(
            {static_cast<size_t>(request_rows), static_cast<size_t>(N)});
        for (TensorBase *output : {
                 alpha_padded.get(),
                 beta_padded.get(),
                 alpha_isolated.get(),
                 beta_isolated.get()})
        {
            ASSERT_TRUE(output->allocateOnDevice(device, stream));
        }

        std::vector<ITensorGemm::TensorProjectionDesc> padded_projections = {
            {alpha_prepared.kernel, alpha_padded.get(), N, nullptr, "alpha"},
            {beta_prepared.kernel, beta_padded.get(), N, nullptr, "beta"},
        };
        std::vector<ITensorGemm::TensorProjectionDesc> isolated_projections = {
            {alpha_prepared.kernel, alpha_isolated.get(), N, nullptr, "alpha"},
            {beta_prepared.kernel, beta_isolated.get(), N, nullptr, "beta"},
        };
        ASSERT_TRUE(alpha_prepared.kernel->multiply_fused_tensor(
            padded_input.get(),
            padded_projections,
            padded_rows,
            K,
            nullptr,
            &workspace));
        ASSERT_TRUE(alpha_prepared.kernel->multiply_fused_tensor(
            isolated_input.get(),
            isolated_projections,
            request_rows,
            K,
            nullptr,
            &workspace));
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        for (TensorBase *output : {
                 alpha_padded.get(),
                 beta_padded.get(),
                 alpha_isolated.get(),
                 beta_isolated.get()})
        {
            TransferEngine::publishDeviceWrite(output, device, stream);
        }

        expectBitwiseFP32RowsEqual(
            std::string("ROCm ") + dtype_tag +
                " padded GDN alpha projection",
            alpha_padded->data() +
                static_cast<size_t>(request_row_offset) * N,
            alpha_isolated->data(),
            static_cast<size_t>(request_rows) * N,
            N);
        expectBitwiseFP32RowsEqual(
            std::string("ROCm ") + dtype_tag +
                " padded GDN beta projection",
            beta_padded->data() +
                static_cast<size_t>(request_row_offset) * N,
            beta_isolated->data(),
            static_cast<size_t>(request_rows) * N,
            N);

        for (ITensorGemm *kernel : kernels)
        {
            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
            ASSERT_NE(consumer, nullptr);
            consumer->unbindWorkspace();
            kernel->clearGPUStreamBinding();
        }
    }

    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
}

/**
 * @test Prove floating ROCm grouped SwiGLU/down matches serial decode by bytes.
 *
 * The down projection is a separate grouped contract from gate/up projection:
 * it fuses SiLU, elementwise multiplication, and the weight projection.  Sweep
 * every floating weight storage type so a backend-library implementation cannot
 * quietly replace the fixed-order production kernel for one dtype.
 */
TEST(Test__ROCmQuantisedGemmSmallM, FloatingSwiGLUDownAllFormatsRuntimeMMatchSerialDecodeStrict)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

#ifndef HAVE_ROCM
    GTEST_SKIP() << "HAVE_ROCM not enabled";
#else
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    constexpr int K = 192;
    constexpr int N = 80;
    constexpr auto verifier_rows = kGroupedVerifierRuntimeRows;
    const DeviceId device = DeviceId::rocm(0);

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    ASSERT_NE(stream, nullptr);

    uint64_t model_id = 183000;
    for (const auto &format : localTPWoFloatingPointFormatCases())
    {
        SCOPED_TRACE(format.label);

        auto down_weights = format.create(
            {static_cast<size_t>(N), static_cast<size_t>(K)},
            static_cast<uint32_t>(model_id + 1));
        ASSERT_NE(down_weights, nullptr);
        const char *dtype_tag = tensorTypeName(down_weights->native_type());
        auto down_prepared = makeGpuPreparedFloatingPointGemm(
            down_weights.get(),
            device,
            std::string("test.rocm.floating_grouped.swiglu_down.") + dtype_tag,
            ModelContextId{model_id++});
        ASSERT_NE(down_prepared.kernel, nullptr);
        down_prepared.kernel->setGPUStream(stream);

        auto *consumer = dynamic_cast<IWorkspaceConsumer *>(down_prepared.kernel);
        ASSERT_NE(consumer, nullptr)
            << dtype_tag << " floating down GEMM must declare workspace ownership";
        const WorkspaceRequirements requirements =
            consumer->getWorkspaceRequirements(
                kGroupedVerifierRuntimeRows.back(),
                N,
                K);
        DeviceWorkspaceManager workspace(
            device,
            requirements.total_bytes_with_alignment() + 64 * 1024 * 1024);
        ASSERT_TRUE(workspace.allocate(requirements));
        consumer->bindWorkspace(&workspace);

        for (const int M : verifier_rows)
        {
            SCOPED_TRACE(std::string(dtype_tag) + " M=" + std::to_string(M));

            auto grouped_gate = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(M), static_cast<size_t>(K)},
                -0.45f,
                0.45f,
                static_cast<uint32_t>(184000 + M + model_id));
            auto grouped_up = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(M), static_cast<size_t>(K)},
                -0.45f,
                0.45f,
                static_cast<uint32_t>(185000 + M + model_id));
            const std::vector<float> gate_values(
                grouped_gate->data(),
                grouped_gate->data() + grouped_gate->numel());
            const std::vector<float> up_values(
                grouped_up->data(),
                grouped_up->data() + grouped_up->numel());
            auto grouped_output = TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(N)});
            ASSERT_TRUE(grouped_gate->ensureOnDevice(device, stream));
            ASSERT_TRUE(grouped_up->ensureOnDevice(device, stream));
            ASSERT_TRUE(grouped_output->allocateOnDevice(device, stream));

            ASSERT_TRUE(down_prepared.kernel->multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                grouped_gate.get(),
                grouped_up.get(),
                grouped_output.get(),
                M,
                N,
                K,
                1.0f,
                0.0f,
                &workspace))
                << dtype_tag << " grouped floating SwiGLU/down M=" << M;
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            TransferEngine::publishDeviceWrite(grouped_output, device, stream);
            const std::vector<float> grouped_values(
                grouped_output->data(),
                grouped_output->data() + grouped_output->numel());

            std::vector<float> serial_values(
                static_cast<size_t>(M) * static_cast<size_t>(N));
            for (int row = 0; row < M; ++row)
            {
                auto row_gate = TestTensorFactory::createFP32(
                    {1u, static_cast<size_t>(K)});
                auto row_up = TestTensorFactory::createFP32(
                    {1u, static_cast<size_t>(K)});
                std::copy_n(
                    gate_values.data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                    K,
                    row_gate->mutable_data());
                std::copy_n(
                    up_values.data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                    K,
                    row_up->mutable_data());
                auto row_output = TestTensorFactory::createFP32(
                    {1u, static_cast<size_t>(N)});
                ASSERT_TRUE(row_gate->ensureOnDevice(device, stream));
                ASSERT_TRUE(row_up->ensureOnDevice(device, stream));
                ASSERT_TRUE(row_output->allocateOnDevice(device, stream));

                ASSERT_TRUE(down_prepared.kernel->multiply_tensor_with_fused_swiglu(
                    row_gate.get(),
                    row_up.get(),
                    row_output.get(),
                    1,
                    N,
                    K,
                    1.0f,
                    0.0f,
                    &workspace));
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
                TransferEngine::publishDeviceWrite(row_output, device, stream);
                std::copy_n(
                    row_output->data(),
                    N,
                    serial_values.data() + static_cast<size_t>(row) * static_cast<size_t>(N));
            }

            expectBitwiseFP32RowsEqual(
                std::string("ROCm ") + dtype_tag +
                    " grouped floating SwiGLU/down M=" + std::to_string(M),
                grouped_values.data(),
                serial_values.data(),
                grouped_values.size(),
                static_cast<size_t>(N));
        }

        consumer->unbindWorkspace();
        down_prepared.kernel->clearGPUStreamBinding();
    }

    std::array<
        std::array<uint64_t, kGroupedVerifierRuntimeRows.back() + 1>,
        3>
        grouped_calls{};
    const auto records = PerfStatsCollector::snapshot(
        {"kernel.rocm_floating_grouped_verifier_swiglu_down_calls"});
    for (const auto &record : records)
    {
        if (record.name != "rocm_floating_grouped_verifier_swiglu_down_calls")
            continue;
        auto tag_equals = [&](const char *name, const std::string &value)
        {
            const auto it = record.tags.find(name);
            return it != record.tags.end() && it->second == value;
        };
        const auto m_it = record.tags.find("m");
        if (m_it == record.tags.end())
            continue;
        const int M = std::stoi(m_it->second);
        if (M < verifier_rows.front() || M > verifier_rows.back() ||
            !tag_equals("n", std::to_string(N)) ||
            !tag_equals("k", std::to_string(K)) ||
            !tag_equals("route", "fixed_order_floating_swiglu_down") ||
            !tag_equals("verifier", "1"))
        {
            continue;
        }
        if (tag_equals("dtype", "fp32"))
            grouped_calls[0][static_cast<size_t>(M)] += record.count;
        else if (tag_equals("dtype", "fp16"))
            grouped_calls[1][static_cast<size_t>(M)] += record.count;
        else if (tag_equals("dtype", "bf16"))
            grouped_calls[2][static_cast<size_t>(M)] += record.count;
    }
    for (const int M : verifier_rows)
    {
        EXPECT_EQ(grouped_calls[0][static_cast<size_t>(M)], 1u)
            << "ROCm FP32 grouped floating SwiGLU/down route missing or duplicated at M=" << M;
        EXPECT_EQ(grouped_calls[1][static_cast<size_t>(M)], 1u)
            << "ROCm FP16 grouped floating SwiGLU/down route missing or duplicated at M=" << M;
        EXPECT_EQ(grouped_calls[2][static_cast<size_t>(M)], 1u)
            << "ROCm BF16 grouped floating SwiGLU/down route missing or duplicated at M=" << M;
    }

    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
    PerfStatsCollector::reset();
#endif
}

TEST(Test__ROCmQuantisedGemmSmallM, RealQwen36OutputGEMMStageGroupedVerifierRowsMatchSerialDecode)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

#ifndef HAVE_ROCM
    GTEST_SKIP() << "HAVE_ROCM not enabled";
#else
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    const std::filesystem::path model_path = qwen36DenseModelPath();
    if (!std::filesystem::exists(model_path))
        GTEST_SKIP() << "Qwen 3.6 dense model not found at " << model_path
                     << "; set LLAMINAR_QWEN36_DENSE_MODEL to run this real-weight regression";

    auto model_ctx = loadQwen36DenseModelForGpuWeights(model_path);
    ASSERT_NE(model_ctx, nullptr);

    std::optional<std::string> weight_name =
        findFirstTensorWithSuffix(*model_ctx, ".attn_output.weight");
    if (!weight_name)
        weight_name = findFirstTensorWithSuffix(*model_ctx, ".self_attn.o_proj.weight");
    if (!weight_name)
    {
        weight_name = findFirstAvailableTensor(
            *model_ctx,
            {"blk.0.ssm_out.weight",
             "blk.1.ssm_out.weight"});
    }
    if (!weight_name)
    {
        std::ostringstream names;
        for (const auto &name : model_ctx->concreteLoader().tensorNames())
        {
            if (name.rfind("blk.0.", 0) == 0)
                names << name << "\n";
        }
        FAIL() << "Could not find a Qwen 3.6 output projection tensor. "
               << "Layer-0 tensors:\n"
               << names.str();
    }

    auto wo_weight = model_ctx->getWeightForDevice(*weight_name, DeviceId::cpu());
    ASSERT_NE(wo_weight, nullptr);

    const int N = static_cast<int>(wo_weight->rows());
    const int K = static_cast<int>(wo_weight->cols());
    ASSERT_GT(N, 0);
    ASSERT_GT(K, 0);

    auto prepared = makeGpuPreparedGemm(
        wo_weight.get(),
        DeviceId::rocm(0),
        *weight_name,
        ModelContextId{3636});

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

    for (float input_scale : {0.35f, 1.0f, 3.0f, 8.0f})
    {
        runRealWeightGemmStageGroupedRowsMatchSerial(
            wo_weight.get(), prepared, 2, N, K, input_scale, /*graph_capture=*/false, stream);
        runRealWeightGemmStageGroupedRowsMatchSerial(
            wo_weight.get(), prepared, 3, N, K, input_scale, /*graph_capture=*/false, stream);
        runRealWeightGemmStageGroupedRowsMatchSerial(
            wo_weight.get(), prepared, 4, N, K, input_scale, /*graph_capture=*/false, stream);
    }
    for (float input_scale : {1.0f, 8.0f})
    {
        runRealWeightGemmStageGroupedRowsMatchSerial(
            wo_weight.get(), prepared, 2, N, K, input_scale, /*graph_capture=*/true, stream);
        runRealWeightGemmStageGroupedRowsMatchSerial(
            wo_weight.get(), prepared, 3, N, K, input_scale, /*graph_capture=*/true, stream);
        runRealWeightGemmStageGroupedRowsMatchSerial(
            wo_weight.get(), prepared, 4, N, K, input_scale, /*graph_capture=*/true, stream);
    }

    prepared.kernel->clearGPUStreamBinding();
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
}

/**
 * @test The Qwen 3.5 122B GDN output-projection shard is batch invariant.
 *
 * A four-way LocalTP participant owns a [3072, 2048] Q8_0 slice of
 * @c blk.0.ssm_out.weight.  The production MTP depth-two verifier presents
 * three activation rows to that projection (one target plus two drafts), while
 * ordinary decode presents the same rows independently at M=1.  Exercise the
 * production prepared-weight and graph-captured GEMMStage route at that exact
 * geometry so a local projection defect cannot be mistaken for collective
 * reduction-order drift.
 */
TEST(Test__ROCmQuantisedGemmSmallM,
     Qwen35MoEGDNOutputProjectionQ80TP4ShardM3MatchesSerialDecodeStrict)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

#ifndef HAVE_ROCM
    GTEST_SKIP() << "HAVE_ROCM not enabled";
#else
    constexpr int M = 3;
    constexpr int N = 3072;
    constexpr int K = 2048;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    auto weight = TestTensorFactory::createQ8_0Random(
        {static_cast<size_t>(N), static_cast<size_t>(K)},
        /*seed=*/351223u);
    auto prepared = makeGpuPreparedGemm(
        weight.get(),
        DeviceId::rocm(0),
        "test.qwen35moe.gdn_output_projection.tp4.rank0",
        ModelContextId{351223});

    hipStream_t stream = nullptr;
    ASSERT_EQ(
        hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
        hipSuccess);

    runRealWeightGemmStageGroupedRowsMatchSerial(
        weight.get(),
        prepared,
        M,
        N,
        K,
        /*input_scale=*/1.0f,
        /*graph_capture=*/true,
        stream);

    prepared.kernel->clearGPUStreamBinding();
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
}

TEST(Test__ROCmQuantisedGemmSmallM, RealQwen36MoELMHeadGroupedVerifierRowsMatchSerialDecodeStrict)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

#ifndef HAVE_ROCM
    GTEST_SKIP() << "HAVE_ROCM not enabled";
#else
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    const std::filesystem::path model_path = qwen36MoEModelPath();
    if (!std::filesystem::exists(model_path))
        GTEST_SKIP() << "Qwen 3.6 MoE model not found at " << model_path
                     << "; set LLAMINAR_QWEN36_MOE_MODEL to run this real-weight regression";

    auto model_ctx = loadQwen36DenseModelForGpuWeights(model_path);
    ASSERT_NE(model_ctx, nullptr);
    ASSERT_TRUE(model_ctx->hasTensor("output.weight"))
        << "Qwen 3.6 MoE test fixture must expose a concrete LM head";

    auto lm_head = model_ctx->getWeightForDevice("output.weight", DeviceId::cpu());
    ASSERT_NE(lm_head, nullptr);
    ASSERT_EQ(lm_head->native_type(), TensorType::Q6_K)
        << "This regression guards the Qwen 3.6 MoE Q6_K LM-head path";

    const int N = static_cast<int>(lm_head->rows());
    const int K = static_cast<int>(lm_head->cols());
    ASSERT_EQ(N, 248320);
    ASSERT_EQ(K, 2048);

    auto prepared = makeGpuPreparedGemm(
        lm_head.get(),
        DeviceId::rocm(0),
        "output.weight",
        ModelContextId{36361});

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    ASSERT_NE(stream, nullptr);

    /*
     * This is the exact LM-head geometry used by Qwen3.6 MoE all-position
     * verifier rows.  The model-level parity test can only tell us that logits
     * drifted; this stage-level gate tells us whether grouped runtime-M
     * projection itself stayed decode-equivalent to serial M=1 GEMV.
     */
    for (const int M : kGroupedVerifierBoundaryRows)
    {
        runRealWeightGemmStageGroupedRowsMatchSerial(
            lm_head.get(),
            prepared,
            M,
            N,
            K,
            /*input_scale=*/0.35f,
            /*graph_capture=*/false,
            stream);
    }
    runRealWeightGemmStageGroupedRowsMatchSerial(
        lm_head.get(),
        prepared,
        2,
        N,
        K,
        /*input_scale=*/0.35f,
        /*graph_capture=*/true,
        stream);

    prepared.kernel->clearGPUStreamBinding();
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
}

TEST(Test__ROCmQuantisedGemmSmallM, DispatchQ80SmallMMatchesReference)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const int N = 512;
    const int K = 1024;
    for (int M : {2, 3, 4})
    {
        runDispatchSmallMMatchesReference(
            "Q8_0 INT8-VNNI small-M sweep",
            M,
            N,
            K,
            PackedPath::INT8VNNI,
            [](const std::vector<size_t> &shape, uint32_t seed)
            { return TestTensorFactory::createQ8_0Random(shape, seed); },
            0.985f);
    }
}

TEST(Test__ROCmQuantisedGemmSmallM, DispatchNativeSmallMAllCodebooksMatchReference)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const int N = 512;
    const int K = 1024;
    for (const auto &format : nativeFormatCases())
    {
        for (int M : {2, 3, 4})
        {
            runDispatchSmallMMatchesReference(
                format.label,
                M,
                N,
                K,
                isInt8VnniFormat(format.tensor_type)
                    ? PackedPath::INT8VNNI
                    : PackedPath::NativeVNNI,
                format.create,
                format.min_cosine);
        }
    }
}

TEST(Test__ROCmQuantisedGemmSmallM, DispatchPlainAsymmetricNativeSmallMUsesFreshBlockSums)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const int N = 384;
    const int K = 768;
    for (int M : {2, 4})
    {
        runDispatchSmallMMatchesReference(
            "Q4_1 native-VNNI asymmetric min correction",
            M,
            N,
            K,
            PackedPath::NativeVNNI,
            [](const std::vector<size_t> &shape, uint32_t seed)
            { return TestTensorFactory::createQ4_1Random(shape, seed); },
            0.985f);

        runDispatchSmallMMatchesReference(
            "Q5_1 native-VNNI asymmetric min correction",
            M,
            N,
            K,
            PackedPath::NativeVNNI,
            [](const std::vector<size_t> &shape, uint32_t seed)
            { return TestTensorFactory::createQ5_1Random(shape, seed); },
            0.985f);
    }
}

TEST(Test__ROCmQuantisedGemmSmallM, BlockwiseQuantizeWithSumsUsesWaveLocalShuffleLanes)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support is not compiled in";
#else
    constexpr int M = 4;
    constexpr int K = 64;
    constexpr int block_size = 32;
    constexpr int blocks_per_row = K / block_size;
    const size_t value_count = static_cast<size_t>(M) * static_cast<size_t>(K);
    const size_t block_count = static_cast<size_t>(M) * static_cast<size_t>(blocks_per_row);

    /*
     * This fixture fills all eight logical 32-thread groups in one v2
     * quantizer workgroup. The last six groups live in later wave64 wavefronts,
     * which used to expose the block-global shuffle-source bug.
     */
    std::vector<float> input(value_count);
    for (int row = 0; row < M; ++row)
    {
        for (int block = 0; block < blocks_per_row; ++block)
        {
            const float block_scale = 0.125f + 0.0375f * static_cast<float>(row * blocks_per_row + block);
            for (int lane = 0; lane < block_size; ++lane)
            {
                const int k = block * block_size + lane;
                const float sign = ((lane + row + block) % 2 == 0) ? 1.0f : -1.0f;
                input[static_cast<size_t>(row) * K + k] =
                    sign * block_scale * static_cast<float>((lane % 17) + 1);
            }
        }
    }

    std::vector<int8_t> expected_q;
    std::vector<float> expected_scales;
    std::vector<int32_t> expected_sums;
    cpuBlockwiseQuantizeWithSums(input, M, K, expected_q, expected_scales, expected_sums);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

    float *d_input = nullptr;
    int8_t *d_quantized = nullptr;
    float *d_scales = nullptr;
    int32_t *d_sums = nullptr;
    ASSERT_EQ(hipMalloc(&d_input, value_count * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_quantized, value_count * sizeof(int8_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_scales, block_count * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_sums, block_count * sizeof(int32_t)), hipSuccess);

    ASSERT_EQ(hipMemcpyAsync(
                  d_input,
                  input.data(),
                  value_count * sizeof(float),
                  hipMemcpyHostToDevice,
                  stream),
              hipSuccess);

    ASSERT_TRUE(rocmQuantGemm_quantizeActivationsBlockwiseWithSums(
        d_input,
        d_quantized,
        d_scales,
        d_sums,
        M,
        K,
        0,
        stream,
        block_size));

    std::vector<int8_t> actual_q(value_count);
    std::vector<float> actual_scales(block_count);
    std::vector<int32_t> actual_sums(block_count);
    ASSERT_EQ(hipMemcpyAsync(
                  actual_q.data(),
                  d_quantized,
                  value_count * sizeof(int8_t),
                  hipMemcpyDeviceToHost,
                  stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(
                  actual_scales.data(),
                  d_scales,
                  block_count * sizeof(float),
                  hipMemcpyDeviceToHost,
                  stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(
                  actual_sums.data(),
                  d_sums,
                  block_count * sizeof(int32_t),
                  hipMemcpyDeviceToHost,
                  stream),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    for (size_t i = 0; i < value_count; ++i)
        ASSERT_EQ(actual_q[i], expected_q[i]) << "quantized byte mismatch at element " << i;
    for (size_t i = 0; i < block_count; ++i)
    {
        ASSERT_NEAR(actual_scales[i], expected_scales[i], 1.0e-7f)
            << "scale mismatch at block " << i;
        ASSERT_EQ(actual_sums[i], expected_sums[i])
            << "sum mismatch at block " << i;
    }

    ASSERT_EQ(hipFree(d_sums), hipSuccess);
    ASSERT_EQ(hipFree(d_scales), hipSuccess);
    ASSERT_EQ(hipFree(d_quantized), hipSuccess);
    ASSERT_EQ(hipFree(d_input), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
}

TEST(Test__ROCmQuantisedGemmSmallM, DispatchQ4KRuntimeM16RecordsNativeRouteCounter)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    const int N = 896;
    const int K = 1024;
    constexpr int M = kDefaultNativeVNNIVerifierRowCapacity;
    runGroupedSmallMMatchesSerialRows(
        "Q4_K native-VNNI counter",
        M,
        N,
        K,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); });

    const auto records = PerfStatsCollector::snapshot({"kernel.rocm_native_vnni_small_m_calls"});
    auto route_record = std::find_if(
        records.begin(),
        records.end(),
        [M](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "rocm_native_vnni_small_m_calls" &&
                   record.kind == PerfStatRecord::Kind::Counter &&
                   record.tags.count("m") != 0 &&
                   record.tags.at("m") == std::to_string(M);
        });

    ASSERT_NE(route_record, records.end())
        << "Q4_K M=16 verifier GEMM must use the graph-native ROCm runtime-row route";
    EXPECT_GE(route_record->value, 1.0);
    EXPECT_EQ(route_record->device, "rocm:0");
    EXPECT_EQ(route_record->tags.at("codebook"), "5");
    EXPECT_EQ(route_record->tags.at("n"), std::to_string(N));
    EXPECT_EQ(route_record->tags.at("k"), std::to_string(K));

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, DispatchQ5KSmallMRecordsNativeRouteCounter)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    const int M = 4;
    const int N = 512;
    const int K = 1024;
    runGroupedSmallMMatchesSerialRows(
        "Q5_K native-VNNI small-M counter",
        M,
        N,
        K,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ5_KRandom(shape, seed); });

    const auto records = PerfStatsCollector::snapshot({"kernel.rocm_native_vnni_small_m_calls"});
    auto route_record = std::find_if(
        records.begin(),
        records.end(),
        [M, N, K](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "rocm_native_vnni_small_m_calls" &&
                   record.kind == PerfStatRecord::Kind::Counter &&
                   record.tags.count("m") != 0 &&
                   record.tags.at("m") == std::to_string(M) &&
                   record.tags.count("n") != 0 &&
                   record.tags.at("n") == std::to_string(N) &&
                   record.tags.count("k") != 0 &&
                   record.tags.at("k") == std::to_string(K);
        });

    ASSERT_NE(route_record, records.end())
        << "Q5_K M=4 verifier GEMM must use the graph-native ROCm small-M native route";
    EXPECT_GE(route_record->value, 1.0);
    EXPECT_EQ(route_record->device, "rocm:0");
    EXPECT_EQ(route_record->tags.at("codebook"), "7");

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, FusedSwiGLUDownQ4KM2RecordsNativeRouteCounter)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    const int M = 2;
    const int N = 512;
    const int K = 1024;
    runFusedSwiGLUDownSmallMMatchesSerialRows(
        "Q4_K native-VNNI fused SwiGLU down counter",
        M,
        N,
        K,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        true,
        1);

    const auto records = PerfStatsCollector::snapshot({"kernel.rocm_native_vnni_small_m_calls"});
    auto route_record = std::find_if(
        records.begin(),
        records.end(),
        [M, N, K](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "rocm_native_vnni_small_m_calls" &&
                   record.kind == PerfStatRecord::Kind::Counter &&
                   record.tags.count("source") != 0 &&
                   record.tags.at("source") == "fused_swiglu" &&
                   record.tags.count("m") != 0 &&
                   record.tags.at("m") == std::to_string(M) &&
                   record.tags.count("n") != 0 &&
                   record.tags.at("n") == std::to_string(N) &&
                   record.tags.count("k") != 0 &&
                   record.tags.at("k") == std::to_string(K);
        });

    ASSERT_NE(route_record, records.end())
        << "Q4_K M=2 fused SwiGLU down must use the graph-native ROCm small-M route";
    EXPECT_GE(route_record->value, 1.0);
    EXPECT_EQ(route_record->device, "rocm:0");
    EXPECT_EQ(route_record->tags.at("codebook"), "5");

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, FusedSwiGLUDownQ5KM4RecordsNativeRouteCounter)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    const int M = 4;
    const int N = 512;
    const int K = 1024;
    runFusedSwiGLUDownSmallMMatchesSerialRows(
        "Q5_K native-VNNI fused SwiGLU down counter",
        M,
        N,
        K,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ5_KRandom(shape, seed); },
        true,
        1);

    const auto records = PerfStatsCollector::snapshot({"kernel.rocm_native_vnni_small_m_calls"});
    auto route_record = std::find_if(
        records.begin(),
        records.end(),
        [M, N, K](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "rocm_native_vnni_small_m_calls" &&
                   record.kind == PerfStatRecord::Kind::Counter &&
                   record.tags.count("source") != 0 &&
                   record.tags.at("source") == "fused_swiglu" &&
                   record.tags.count("m") != 0 &&
                   record.tags.at("m") == std::to_string(M) &&
                   record.tags.count("n") != 0 &&
                   record.tags.at("n") == std::to_string(N) &&
                   record.tags.count("k") != 0 &&
                   record.tags.at("k") == std::to_string(K);
        });

    ASSERT_NE(route_record, records.end())
        << "Q5_K M=4 fused SwiGLU down must use the graph-native ROCm small-M route";
    EXPECT_GE(route_record->value, 1.0);
    EXPECT_EQ(route_record->device, "rocm:0");
    EXPECT_EQ(route_record->tags.at("codebook"), "7");

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedFusedSwiGLUDownQ4KQwen36FFNDownM2MatchesReference)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    constexpr int M = 2;
    constexpr int N = 5120;
    constexpr int K = 17408;
    runFusedSwiGLUDownSmallMMatchesSerialRows(
        "Q4_K native-VNNI Qwen3.6 FFN down verifier shape",
        M,
        N,
        K,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        true,
        4);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    auto route_record = std::find_if(
        records.begin(),
        records.end(),
        [](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "rocm_native_vnni_small_m_calls" &&
                   record.kind == PerfStatRecord::Kind::Counter &&
                   record.tags.count("source") != 0 &&
                   record.tags.at("source") == "fused_swiglu" &&
                   record.tags.count("m") != 0 &&
                   record.tags.at("m") == "2" &&
                   record.tags.count("n") != 0 &&
                   record.tags.at("n") == "5120" &&
                   record.tags.count("k") != 0 &&
                   record.tags.at("k") == "17408";
        });

    ASSERT_NE(route_record, records.end())
        << "Qwen3.6 MTP verifier FFN down must use graph-native ROCm fused SwiGLU/down small-M route";
    EXPECT_GE(route_record->value, 1.0);
    EXPECT_EQ(route_record->device, "rocm:0");
    EXPECT_EQ(route_record->tags.at("codebook"), "5");

    double graph_split_reduce_launches = 0.0;
    for (const auto &record : records)
    {
        if (record.domain != "kernel" ||
            record.name != "rocm_native_vnni_small_m_launch" ||
            record.kind != PerfStatRecord::Kind::Counter ||
            record.tags.count("m") == 0 ||
            record.tags.at("m") != "2" ||
            record.tags.count("n") == 0 ||
            record.tags.at("n") != "5120" ||
            record.tags.count("k") == 0 ||
            record.tags.at("k") != "17408")
        {
            continue;
        }

        if (record.tags.count("path") != 0 &&
            record.tags.at("path") == "split_reduce")
        {
            graph_split_reduce_launches += record.value;
        }
    }

    EXPECT_GE(graph_split_reduce_launches, 1.0)
        << "GPU-graph Qwen3.6 small-M FFN down must use ordered workspace K-partition reduction";

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedFusedSwiGLUDownQ4KQwen36FFNDownM2MatchesSerialM1RowsStrict)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    constexpr int M = 2;
    constexpr int N = 5120;
    constexpr int K = 17408;
    runFusedSwiGLUDownSmallMMatchesSerialRows(
        "Q4_K native-VNNI Qwen3.6 FFN down verifier shape",
        M,
        N,
        K,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        true,
        4);

    const auto records = PerfStatsCollector::snapshot({"kernel.rocm_native_vnni_small_m_calls"});
    auto route_record = std::find_if(
        records.begin(),
        records.end(),
        [](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "rocm_native_vnni_small_m_calls" &&
                   record.kind == PerfStatRecord::Kind::Counter &&
                   record.tags.count("source") != 0 &&
                   record.tags.at("source") == "fused_swiglu" &&
                   record.tags.count("m") != 0 &&
                   record.tags.at("m") == "2" &&
                   record.tags.count("n") != 0 &&
                   record.tags.at("n") == "5120" &&
                   record.tags.count("k") != 0 &&
                   record.tags.at("k") == "17408";
        });

    ASSERT_NE(route_record, records.end())
        << "Qwen3.6 MTP verifier FFN down must use graph-native ROCm fused SwiGLU/down small-M route";
    EXPECT_GE(route_record->value, 1.0);
    EXPECT_EQ(route_record->device, "rocm:0");
    EXPECT_EQ(route_record->tags.at("codebook"), "5");

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedFusedSwiGLUDownQ4KQwen36FFNDownM4MatchesReference)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    constexpr int M = 4;
    constexpr int N = 5120;
    constexpr int K = 17408;
    runFusedSwiGLUDownSmallMMatchesSerialRows(
        "Q4_K native-VNNI Qwen3.6 FFN down shifted-prefill shape",
        M,
        N,
        K,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        true,
        4);

    const auto records = PerfStatsCollector::snapshot({"kernel.rocm_native_vnni_small_m_calls"});
    auto route_record = std::find_if(
        records.begin(),
        records.end(),
        [](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "rocm_native_vnni_small_m_calls" &&
                   record.kind == PerfStatRecord::Kind::Counter &&
                   record.tags.count("source") != 0 &&
                   record.tags.at("source") == "fused_swiglu" &&
                   record.tags.count("m") != 0 &&
                   record.tags.at("m") == "4" &&
                   record.tags.count("n") != 0 &&
                   record.tags.at("n") == "5120" &&
                   record.tags.count("k") != 0 &&
                   record.tags.at("k") == "17408";
        });

    ASSERT_NE(route_record, records.end())
        << "Qwen3.6 MTP shifted prefill FFN down must use graph-native ROCm fused SwiGLU/down small-M route";
    EXPECT_GE(route_record->value, 1.0);
    EXPECT_EQ(route_record->device, "rocm:0");
    EXPECT_EQ(route_record->tags.at("codebook"), "5");

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedDispatchQ4KSmallMMatchesReference)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

#ifdef HAVE_ROCM
    const int N = 896;
    const int K = 1024;
    for (int M : {2, 3, 4})
    {
        runGraphCapturedDispatchSmallMMatchesReference(
            "Q4_K native-VNNI",
            M,
            N,
            K,
            [](const std::vector<size_t> &shape, uint32_t seed)
            { return TestTensorFactory::createQ4_KRandom(shape, seed); },
            0.985f);
    }
#endif
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedDispatchIQ3SSmallMMatchesReference)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

#ifdef HAVE_ROCM
    const int N = 512;
    const int K = 1024;
    for (int M : {3, 4})
    {
        runGraphCapturedDispatchSmallMMatchesReference(
            "IQ3_S native-VNNI",
            M,
            N,
            K,
            [](const std::vector<size_t> &shape, uint32_t seed)
            { return TestTensorFactory::createIQ3_SRandom(shape, seed); },
            0.985f);
    }
#endif
}

TEST(Test__ROCmQuantisedGemmSmallM, FusedQ80QKVM2MatchesSeparate)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const int K = 896;
    runFusedQKVSmallMMatchesSeparate(
        "Q8_0 INT8-VNNI",
        2,
        K,
        PackedPath::INT8VNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ8_0Random(shape, seed); },
        0.9999f);
}

TEST(Test__ROCmQuantisedGemmSmallM, FusedQ4KQKVM2MatchesSeparate)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const int K = 1024;
    runFusedQKVSmallMMatchesSeparate(
        "Q4_K native-VNNI",
        2,
        K,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.9999f);
}

TEST(Test__ROCmQuantisedGemmSmallM, FusedQ4KQKVRuntimeMMatchesSerialM1DecodeRowsStrict)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    constexpr int K = 1024;
    for (const int M : kGroupedVerifierRuntimeRows)
    {
        runFusedQKVSmallMMatchesSerialM1DecodeRows(
            "Q4_K native-VNNI fused QKV strict serial decode",
            M,
            K,
            PackedPath::NativeVNNI,
            [](const std::vector<size_t> &shape, uint32_t seed)
            { return TestTensorFactory::createQ4_KRandom(shape, seed); });
    }
}

TEST(Test__ROCmQuantisedGemmSmallM, FusedIQ3SQwen36QKVRuntimeMMatchesSerialM1DecodeRowsStrict)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    /*
     * Qwen3.6 MoE MTP verifier parity first drifted at layer2_QKV_PROJECTION
     * on the ROCm IQ3_S model.  The smaller synthetic Q4_K QKV regression above
     * proves the algorithm family, but it does not cover the production
     * codebook or the 5120-wide hidden rows used by this model.  Keep this test
     * strict: grouped Q/K/V publication at each runtime-M boundary must match
     * rows one at a time through the ordinary decode GEMV path.
     */
    constexpr int K = 5120;
    constexpr int Nq = 5120;
    constexpr int Nk = 1024;
    constexpr int Nv = 1024;
    for (const int M : kGroupedVerifierBoundaryRows)
    {
        runFusedQKVSmallMMatchesSerialM1DecodeRows(
            "IQ3_S native-VNNI Qwen3.6 fused QKV strict serial decode",
            M,
            K,
            PackedPath::NativeVNNI,
            [](const std::vector<size_t> &shape, uint32_t seed)
            { return TestTensorFactory::createIQ3_SRandom(shape, seed); },
            Nq,
            Nk,
            Nv);
    }
}

TEST(Test__ROCmQuantisedGemmSmallM, FusedQ5KQKVSmallMMatchesSeparate)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const int K = 1024;
    for (int M : {3, 4})
    {
        runFusedQKVSmallMMatchesSeparate(
            "Q5_K native-VNNI",
            M,
            K,
            PackedPath::NativeVNNI,
            [](const std::vector<size_t> &shape, uint32_t seed)
            { return TestTensorFactory::createQ5_KRandom(shape, seed); },
            0.9999f);
    }
}

TEST(Test__ROCmQuantisedGemmSmallM, FusedQ5KQKVSmallMRecordsSharedQuantizedNativeRoute)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    const int M = 4;
    const int K = 1024;
    runFusedQKVSmallMMatchesSeparate(
        "Q5_K native-VNNI shared small-M quant counter",
        M,
        K,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ5_KRandom(shape, seed); },
        0.9999f);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    auto shared_quant_record = std::find_if(
        records.begin(),
        records.end(),
        [M, K](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "rocm_fused_small_m_shared_quant_calls" &&
                   record.kind == PerfStatRecord::Kind::Counter &&
                   record.tags.count("m") != 0 &&
                   record.tags.at("m") == std::to_string(M) &&
                   record.tags.count("k") != 0 &&
                   record.tags.at("k") == std::to_string(K) &&
                   record.tags.count("projections") != 0 &&
                   record.tags.at("projections") == "3";
        });

    ASSERT_NE(shared_quant_record, records.end())
        << "Fused Q5_K M=4 QKV must quantize activations once before projection dispatch";
    EXPECT_GE(shared_quant_record->value, 1.0);
    EXPECT_EQ(shared_quant_record->device, "rocm:0");

    double batched_calls = 0.0;
    double batched_projection_calls = 0.0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == std::to_string(M) &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == std::to_string(K) &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "3")
        {
            batched_calls += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_projection_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == std::to_string(M) &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == std::to_string(K))
        {
            batched_projection_calls += record.value;
        }
    }
    if (batched_calls < 1.0 || batched_projection_calls < 3.0)
    {
        for (const auto &record : records)
        {
            if (record.domain != "kernel" ||
                (record.name.find("rocm_native_vnni") == std::string::npos &&
                 record.name.find("rocm_fused") == std::string::npos))
            {
                continue;
            }
            std::string tags;
            for (const auto &tag : record.tags)
            {
                if (!tags.empty())
                    tags += ",";
                tags += tag.first + "=" + tag.second;
            }
            LOG_INFO("[SmallM] observed counter name=" << record.name
                     << " value=" << record.value
                     << " device=" << record.device
                     << " tags=" << tags);
        }
    }
    EXPECT_GE(batched_calls, 1.0)
        << "Fused QKV M=4 should use one graph-native batched native route";
    EXPECT_GE(batched_projection_calls, 3.0)
        << "Fused QKV M=4 batched route should cover Q, K, and V projections";

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, FusedQ4KQKVM2RecordsSharedQuantizedNativeRoute)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    const int K = 1024;
    runFusedQKVSmallMMatchesSeparate(
        "Q4_K native-VNNI shared quant counter",
        2,
        K,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.9999f);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    auto shared_quant_record = std::find_if(
        records.begin(),
        records.end(),
        [K](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "rocm_fused_m2_shared_quant_calls" &&
                   record.kind == PerfStatRecord::Kind::Counter &&
                   record.tags.count("k") != 0 &&
                   record.tags.at("k") == std::to_string(K) &&
                   record.tags.count("projections") != 0 &&
                   record.tags.at("projections") == "3";
        });

    ASSERT_NE(shared_quant_record, records.end())
        << "Fused Q4_K M=2 QKV must quantize activations once before projection dispatch";
    EXPECT_GE(shared_quant_record->value, 1.0);
    EXPECT_EQ(shared_quant_record->device, "rocm:0");

    double batched_calls = 0.0;
    double batched_projection_calls = 0.0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == std::to_string(K) &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "3")
        {
            batched_calls += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_projection_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == std::to_string(K))
        {
            batched_projection_calls += record.value;
        }
    }
    EXPECT_GE(batched_calls, 1.0)
        << "Fused QKV M=2 should use one graph-native batched native route";
    EXPECT_GE(batched_projection_calls, 3.0)
        << "Fused QKV M=2 batched route should cover Q, K, and V projections";

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, FusedQ4KQwen36QKVM2MatchesSeparate)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    runFusedQKVSmallMMatchesSeparate(
        "Q4_K native-VNNI Qwen3.6 QKV shape",
        2,
        5120,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.9999f,
        5120,
        1024,
        1024);
}

/**
 * @test Captured concurrent prefill preserves asymmetric TP Q/K/V outputs.
 *
 * Qwen2.5-0.5B uses 14 query heads and two grouped-KV heads.  Under TP=2 the
 * production projection widths are therefore Q=448, K=64, and V=64 for a
 * nine-token prefill.  This exact mixed-width transaction exposed an ordering
 * defect in the default ROCm multi-stream prefill route: the graph completed,
 * but Q/K/V were read after shared activation scratch had been reused.  Keep
 * the real graph-captured concurrent entry point and a shared graph workspace
 * here so a serial or per-kernel substitute cannot satisfy the regression.
 */
TEST(Test__ROCmQuantisedGemmSmallM,
     GraphCapturedConcurrentQ40Qwen2TP2QKVM9MatchesSeparate)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

#ifdef HAVE_ROCM
    constexpr int M = 9;
    constexpr int K = 896;
    const std::array<int, 3> widths = {448, 64, 64};

    std::array<std::unique_ptr<Q4_0Tensor>, 3> weights;
    std::array<ROCmPackedWeights, 3> packed;
    std::array<std::unique_ptr<ROCmQuantisedGemmKernel>, 3> kernels;
    for (size_t i = 0; i < widths.size(); ++i)
    {
        weights[i] = TestTensorFactory::createQ4_0Random(
            {static_cast<size_t>(widths[i]), static_cast<size_t>(K)},
            static_cast<uint32_t>(2100 + i));
        ASSERT_TRUE(packWeightsToROCm(weights[i].get(), packed[i]));
        expectPackedPath(packed[i], PackedPath::NativeVNNI);
        kernels[i] = std::make_unique<ROCmQuantisedGemmKernel>(&packed[i], 0);
    }

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    for (auto &kernel : kernels)
        kernel->setGPUStream(stream);

    WorkspaceRequirements combined;
    for (size_t i = 0; i < widths.size(); ++i)
        combined.merge(kernels[i]->getWorkspaceRequirements(M, widths[i], K));
    kernels.front()->appendFusedProjectionWorkspaceRequirements(
        combined, M, widths, K);
    auto workspace = std::make_unique<DeviceWorkspaceManager>(
        DeviceId::rocm(0),
        combined.total_bytes_with_alignment() + 64 * 1024 * 1024);
    ASSERT_TRUE(workspace->allocate(combined));
    for (auto &kernel : kernels)
        kernel->bindWorkspace(workspace.get());

    auto input_a = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(M), static_cast<size_t>(K)}, -0.75f, 0.75f, 3101);
    auto input_b = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(M), static_cast<size_t>(K)}, -0.75f, 0.75f, 3102);
    ASSERT_TRUE(input_a->ensureOnDevice(DeviceId::rocm(0), stream));
    ASSERT_TRUE(input_b->ensureOnDevice(DeviceId::rocm(0), stream));

    std::array<std::unique_ptr<FP32Tensor>, 3> reference_a;
    std::array<std::unique_ptr<FP32Tensor>, 3> reference_b;
    std::array<std::unique_ptr<FP32Tensor>, 3> captured_a;
    std::array<std::unique_ptr<FP32Tensor>, 3> captured_b;
    std::array<std::unique_ptr<FP32Tensor>, 3> observed_a;
    std::array<std::unique_ptr<TensorSlice>, 3> bias_slices;
    for (size_t i = 0; i < widths.size(); ++i)
    {
        const std::vector<size_t> shape = {
            static_cast<size_t>(M), static_cast<size_t>(widths[i])};
        auto local_bias = TestTensorFactory::createFP32(
            {static_cast<size_t>(widths[i])});
        for (int column = 0; column < widths[i]; ++column)
        {
            const float magnitude = i == 1 ? 3.0f : 0.125f;
            local_bias->mutable_data()[column] =
                magnitude * static_cast<float>((column % 11) - 5);
        }
        std::unique_ptr<TensorBase> sliced_bias_storage =
            std::move(local_bias);
        bias_slices[i] = std::make_unique<TensorSlice>(
            std::move(sliced_bias_storage),
            SliceMetadata::forRowParallel(
                static_cast<size_t>(widths[i] * 2),
                1,
                0,
                2,
                true));
        reference_a[i] = TestTensorFactory::createFP32(shape);
        reference_b[i] = TestTensorFactory::createFP32(shape);
        captured_a[i] = TestTensorFactory::createFP32(shape);
        captured_b[i] = TestTensorFactory::createFP32(shape);
        observed_a[i] = TestTensorFactory::createFP32(shape);
        ASSERT_TRUE(reference_a[i]->allocateOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(reference_b[i]->allocateOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(captured_a[i]->allocateOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(captured_b[i]->allocateOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(observed_a[i]->allocateOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(bias_slices[i]->ensureOnDevice(DeviceId::rocm(0), stream));

        ASSERT_TRUE(kernels[i]->multiply_tensor(
            input_a.get(), reference_a[i].get(), M, widths[i], K,
            true, 1.0f, 0.0f, bias_slices[i].get()));
        ASSERT_TRUE(kernels[i]->multiply_tensor(
            input_b.get(), reference_b[i].get(), M, widths[i], K,
            true, 1.0f, 0.0f, bias_slices[i].get()));
    }
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    for (size_t i = 0; i < widths.size(); ++i)
    {
        TransferEngine::publishCurrentDeviceWrite(reference_a[i], stream);
        TransferEngine::publishCurrentDeviceWrite(reference_b[i], stream);
    }

    std::vector<ITensorGemm::TensorProjectionDesc> projections_a;
    std::vector<ITensorGemm::TensorProjectionDesc> projections_b;
    for (size_t i = 0; i < widths.size(); ++i)
    {
        projections_a.emplace_back(
            kernels[i].get(), captured_a[i].get(), widths[i],
            bias_slices[i].get(), "qkv_a");
        projections_b.emplace_back(
            kernels[i].get(), captured_b[i].get(), widths[i],
            bias_slices[i].get(), "qkv_b");
    }

    // Provision the persistent side-stream/event transaction before capture.
    ASSERT_TRUE(kernels.front()->multiply_fused_tensor(
        input_a.get(), projections_a, M, K, nullptr, workspace.get()));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    auto captured_graph = std::make_unique<HIPGraphCapture>(stream, 0);
    {
        ScopedBackendGraphCapture capture_transaction(
            *captured_graph,
            "GraphCapturedConcurrentQ40Qwen2TP2QKVM9MatchesSeparate");
        ASSERT_TRUE(capture_transaction.begin());
        // A Qwen2 forward graph reuses the per-device projection event pool for
        // both QKV and gate/up fan-out in every layer.  Preserve that pressure
        // here: an event dependency that aliases a later record can look sound
        // in a two-transaction micrograph while allowing the first layer's
        // shared activation scratch to be overwritten in the full 24-layer DAG.
        for (int layer = 0; layer < 24; ++layer)
        {
            ASSERT_TRUE(kernels.front()->multiply_fused_tensor(
                input_a.get(), projections_a, M, K, nullptr, workspace.get()));
            if (layer == 0)
            {
                for (size_t i = 0; i < widths.size(); ++i)
                {
                    ASSERT_EQ(
                        hipMemcpyAsync(
                            observed_a[i]->gpu_data_ptr(),
                            captured_a[i]->gpu_data_ptr(),
                            static_cast<size_t>(M) * static_cast<size_t>(widths[i]) *
                                sizeof(float),
                            hipMemcpyDeviceToDevice,
                            stream),
                        hipSuccess);
                }
            }
            ASSERT_TRUE(kernels.front()->multiply_fused_tensor(
                input_b.get(), projections_b, M, K, nullptr, workspace.get()));
        }
        capture_transaction.finish();
    }
    ASSERT_TRUE(captured_graph->instantiate());

    for (int replay = 0; replay < 4; ++replay)
    {
        ASSERT_TRUE(captured_graph->launch()) << "replay=" << replay;
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess) << "replay=" << replay;
    }

    for (size_t i = 0; i < widths.size(); ++i)
    {
        TransferEngine::publishCurrentDeviceWrite(captured_a[i], stream);
        TransferEngine::publishCurrentDeviceWrite(captured_b[i], stream);
        TransferEngine::publishCurrentDeviceWrite(observed_a[i], stream);
        const size_t count = static_cast<size_t>(M) * static_cast<size_t>(widths[i]);
        const float cosine_a = cosineSim(
            captured_a[i]->data(), reference_a[i]->data(), count);
        const float cosine_b = cosineSim(
            captured_b[i]->data(), reference_b[i]->data(), count);
        const float observed_cosine_a = cosineSim(
            observed_a[i]->data(), reference_a[i]->data(), count);
        LOG_INFO("[SmallM] Qwen2 captured sequential QKV projection=" << i
                                                                       << " N=" << widths[i]
                                                                       << " cosine_a=" << cosine_a
                                                                       << " cosine_b=" << cosine_b
                                                                       << " observed_cosine_a="
                                                                       << observed_cosine_a);
        EXPECT_GT(cosine_a, 0.9999f) << "first transaction projection=" << i;
        EXPECT_GT(cosine_b, 0.9999f) << "second transaction projection=" << i;
        EXPECT_GT(observed_cosine_a, 0.9999f)
            << "main-stream consumer observed projection=" << i
            << " before the concurrent side stream completed";
    }

    captured_graph.reset();
    for (auto &kernel : kernels)
    {
        kernel->unbindWorkspace();
        kernel->clearGPUStreamBinding();
    }
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
}

/**
 * @test Concurrent participant capture preserves Qwen2 TP projection outputs.
 *
 * LocalTP captures one complete graph per participant on persistent host
 * workers.  The two workers enter HIP capture together, and each graph joins
 * its device-local Q/K/V side streams through the concurrent projection pool.
 * This fixture reproduces that ownership boundary without RCCL or later model
 * stages obscuring the first projection divergence.
 */
TEST(Test__ROCmQuantisedGemmSmallM,
     ConcurrentTwoDeviceGraphCaptureQ40Qwen2TP2QKVM9MatchesSeparate)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

#ifdef HAVE_ROCM
    ScopedHIPCurrentDeviceRestore restore_calling_device;
    int device_count = 0;
    ASSERT_EQ(hipGetDeviceCount(&device_count), hipSuccess);
    if (device_count < 2)
        GTEST_SKIP() << "Two ROCm devices are required";

    constexpr int M = 9;
    constexpr int K = 896;
    constexpr int kParticipants = 2;
    const std::array<int, 3> widths = {448, 64, 64};

    struct Participant
    {
        int device = -1;
        hipStream_t stream = nullptr;
        std::unique_ptr<HIPGraphCapture> captured_graph;
        std::array<std::unique_ptr<Q4_0Tensor>, 3> weights;
        std::array<ROCmPackedWeights, 3> packed;
        std::array<std::unique_ptr<ROCmQuantisedGemmKernel>, 3> kernels;
        std::unique_ptr<DeviceWorkspaceManager> workspace;
        std::unique_ptr<FP32Tensor> input_a;
        std::unique_ptr<FP32Tensor> input_b;
        std::array<std::unique_ptr<FP32Tensor>, 3> reference_a;
        std::array<std::unique_ptr<FP32Tensor>, 3> reference_b;
        std::array<std::unique_ptr<FP32Tensor>, 3> captured_a;
        std::array<std::unique_ptr<FP32Tensor>, 3> captured_b;
        std::array<std::unique_ptr<FP32Tensor>, 3> observed_a;
        std::vector<ITensorGemm::TensorProjectionDesc> projections_a;
        std::vector<ITensorGemm::TensorProjectionDesc> projections_b;
        std::string failure;
    };

    std::array<Participant, kParticipants> participants;
    for (int device = 0; device < kParticipants; ++device)
    {
        Participant &participant = participants[device];
        participant.device = device;
        ASSERT_EQ(hipSetDevice(device), hipSuccess);
        ASSERT_EQ(
            hipStreamCreateWithFlags(&participant.stream, hipStreamNonBlocking),
            hipSuccess);

        WorkspaceRequirements combined;
        for (size_t i = 0; i < widths.size(); ++i)
        {
            participant.weights[i] = TestTensorFactory::createQ4_0Random(
                {static_cast<size_t>(widths[i]), static_cast<size_t>(K)},
                static_cast<uint32_t>(4100 + device * 10 + i));
            ASSERT_TRUE(packWeightsToROCm(
                participant.weights[i].get(), participant.packed[i]));
            expectPackedPath(participant.packed[i], PackedPath::NativeVNNI);
            participant.kernels[i] =
                std::make_unique<ROCmQuantisedGemmKernel>(
                    &participant.packed[i], device);
            participant.kernels[i]->setGPUStream(participant.stream);
            combined.merge(
                participant.kernels[i]->getWorkspaceRequirements(
                    M, widths[i], K));
        }
        participant.kernels.front()
            ->appendFusedProjectionWorkspaceRequirements(
                combined, M, widths, K);

        participant.workspace = std::make_unique<DeviceWorkspaceManager>(
            DeviceId::rocm(device),
            combined.total_bytes_with_alignment() + 64 * 1024 * 1024);
        ASSERT_TRUE(participant.workspace->allocate(combined));
        for (auto &kernel : participant.kernels)
            kernel->bindWorkspace(participant.workspace.get());

        participant.input_a = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)},
            -0.75f, 0.75f, static_cast<uint32_t>(5101 + device * 2));
        participant.input_b = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)},
            -0.75f, 0.75f, static_cast<uint32_t>(5102 + device * 2));
        ASSERT_TRUE(participant.input_a->ensureOnDevice(
            DeviceId::rocm(device), participant.stream));
        ASSERT_TRUE(participant.input_b->ensureOnDevice(
            DeviceId::rocm(device), participant.stream));

        for (size_t i = 0; i < widths.size(); ++i)
        {
            const std::vector<size_t> shape = {
                static_cast<size_t>(M), static_cast<size_t>(widths[i])};
            participant.reference_a[i] = TestTensorFactory::createFP32(shape);
            participant.reference_b[i] = TestTensorFactory::createFP32(shape);
            participant.captured_a[i] = TestTensorFactory::createFP32(shape);
            participant.captured_b[i] = TestTensorFactory::createFP32(shape);
            participant.observed_a[i] = TestTensorFactory::createFP32(shape);
            ASSERT_TRUE(participant.reference_a[i]->allocateOnDevice(
                DeviceId::rocm(device)));
            ASSERT_TRUE(participant.reference_b[i]->allocateOnDevice(
                DeviceId::rocm(device)));
            ASSERT_TRUE(participant.captured_a[i]->allocateOnDevice(
                DeviceId::rocm(device)));
            ASSERT_TRUE(participant.captured_b[i]->allocateOnDevice(
                DeviceId::rocm(device)));
            ASSERT_TRUE(participant.observed_a[i]->allocateOnDevice(
                DeviceId::rocm(device)));

            ASSERT_TRUE(participant.kernels[i]->multiply_tensor(
                participant.input_a.get(), participant.reference_a[i].get(),
                M, widths[i], K));
            ASSERT_TRUE(participant.kernels[i]->multiply_tensor(
                participant.input_b.get(), participant.reference_b[i].get(),
                M, widths[i], K));

            participant.projections_a.emplace_back(
                participant.kernels[i].get(), participant.captured_a[i].get(),
                widths[i], nullptr, "qkv_a");
            participant.projections_b.emplace_back(
                participant.kernels[i].get(), participant.captured_b[i].get(),
                widths[i], nullptr, "qkv_b");
        }
        ASSERT_EQ(hipStreamSynchronize(participant.stream), hipSuccess);

        // Materialize the device-local side-stream/event pool before either
        // worker begins capture, matching production prepareGraphLaunch().
        ASSERT_TRUE(participant.kernels.front()->multiply_fused_tensor(
            participant.input_a.get(), participant.projections_a,
            M, K, nullptr, participant.workspace.get()));
        ASSERT_EQ(hipStreamSynchronize(participant.stream), hipSuccess);
    }

    std::barrier capture_start(kParticipants);
    std::array<std::thread, kParticipants> workers;
    for (int device = 0; device < kParticipants; ++device)
    {
        workers[device] = std::thread([&, device]()
        {
            Participant &participant = participants[device];
            auto fail = [&](std::string message)
            {
                if (participant.failure.empty())
                    participant.failure = std::move(message);
            };

            if (hipSetDevice(device) != hipSuccess)
            {
                fail("hipSetDevice failed");
                return;
            }
            capture_start.arrive_and_wait();
            participant.captured_graph = std::make_unique<HIPGraphCapture>(
                participant.stream,
                device);
            ScopedBackendGraphCapture capture_transaction(
                *participant.captured_graph,
                "ConcurrentTwoDeviceGraphCaptureQ40Qwen2TP2QKVM9MatchesSeparate");
            if (!capture_transaction.begin())
            {
                fail("backend graph capture begin failed");
                return;
            }
            for (int layer = 0; layer < 24; ++layer)
            {
                if (!participant.kernels.front()->multiply_fused_tensor(
                        participant.input_a.get(), participant.projections_a,
                        M, K, nullptr, participant.workspace.get()))
                {
                    fail("first fused projection launch failed");
                    return;
                }
                if (layer == 0)
                {
                    for (size_t i = 0; i < widths.size(); ++i)
                    {
                        if (hipMemcpyAsync(
                                participant.observed_a[i]->gpu_data_ptr(),
                                participant.captured_a[i]->gpu_data_ptr(),
                                static_cast<size_t>(M) *
                                    static_cast<size_t>(widths[i]) *
                                    sizeof(float),
                                hipMemcpyDeviceToDevice,
                                participant.stream) != hipSuccess)
                        {
                            fail("captured observation copy failed");
                            return;
                        }
                    }
                }
                if (!participant.kernels.front()->multiply_fused_tensor(
                        participant.input_b.get(), participant.projections_b,
                        M, K, nullptr, participant.workspace.get()))
                {
                    fail("second fused projection launch failed");
                    return;
                }
            }
            capture_transaction.finish();
            if (!participant.captured_graph->instantiate())
            {
                fail("graph instantiate failed");
                return;
            }
            for (int replay = 0; replay < 4; ++replay)
            {
                if (!participant.captured_graph->launch() ||
                    hipStreamSynchronize(participant.stream) != hipSuccess)
                {
                    fail("captured graph replay failed");
                    return;
                }
            }
        });
    }
    for (auto &worker : workers)
        worker.join();

    for (Participant &participant : participants)
    {
        ASSERT_TRUE(participant.failure.empty())
            << "device=" << participant.device << " " << participant.failure;
        ASSERT_EQ(hipSetDevice(participant.device), hipSuccess);
        for (size_t i = 0; i < widths.size(); ++i)
        {
            TransferEngine::publishCurrentDeviceWrite(
                participant.reference_a[i], participant.stream);
            TransferEngine::publishCurrentDeviceWrite(
                participant.reference_b[i], participant.stream);
            TransferEngine::publishCurrentDeviceWrite(
                participant.captured_a[i], participant.stream);
            TransferEngine::publishCurrentDeviceWrite(
                participant.captured_b[i], participant.stream);
            TransferEngine::publishCurrentDeviceWrite(
                participant.observed_a[i], participant.stream);
            const size_t count =
                static_cast<size_t>(M) * static_cast<size_t>(widths[i]);
            EXPECT_GT(
                cosineSim(participant.captured_a[i]->data(),
                          participant.reference_a[i]->data(), count),
                0.9999f)
                << "device=" << participant.device << " projection=" << i;
            EXPECT_GT(
                cosineSim(participant.captured_b[i]->data(),
                          participant.reference_b[i]->data(), count),
                0.9999f)
                << "device=" << participant.device << " projection=" << i;
            EXPECT_GT(
                cosineSim(participant.observed_a[i]->data(),
                          participant.reference_a[i]->data(), count),
                0.9999f)
                << "device=" << participant.device
                << " first-layer main-stream consumer projection=" << i;
        }

        participant.captured_graph.reset();
        for (auto &kernel : participant.kernels)
        {
            kernel->unbindWorkspace();
            kernel->clearGPUStreamBinding();
        }
        ASSERT_EQ(hipStreamDestroy(participant.stream), hipSuccess);
    }
#endif
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedFusedQ4KQwen36FFNGateUpM2MatchesSeparate)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    runFusedProjectionGroupSmallMMatchesSeparate(
        "Q4_K native-VNNI Qwen3.6 FFN gate/up MTP verifier group",
        2,
        5120,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.9999f,
        {17408, 17408},
        true,
        true,
        2,
        true);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    double batched_calls = 0.0;
    double batched_projection_calls = 0.0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "2")
        {
            batched_calls += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_projection_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120")
        {
            batched_projection_calls += record.value;
        }
    }

    EXPECT_GE(batched_calls, 1.0)
        << "Graph-captured Qwen3.6 FFN gate/up should use one batched native route";
    EXPECT_GE(batched_projection_calls, 2.0)
        << "Batched FFN gate/up route should cover both projection payloads";

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedFusedQ5KQwen36FFNGateUpM2MatchesSeparate)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    runFusedProjectionGroupSmallMMatchesSeparate(
        "Q5_K native-VNNI Qwen3.6 FFN gate/up MTP verifier group",
        2,
        5120,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ5_KRandom(shape, seed); },
        0.999f,
        {17408, 17408},
        true,
        true,
        2,
        true);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    double batched_calls = 0.0;
    double batched_projection_calls = 0.0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "2")
        {
            batched_calls += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_projection_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120")
        {
            batched_projection_calls += record.value;
        }
    }

    EXPECT_GE(batched_calls, 1.0)
        << "Graph-captured Qwen3.6 Q5_K FFN gate/up should use one batched native route";
    EXPECT_GE(batched_projection_calls, 2.0)
        << "Batched Q5_K FFN gate/up route should cover both projection payloads";

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, FusedQ4KQwen36FFNGateUpM2UsesCanonicalBatchedSplitKWorkspace)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    constexpr int M = 2;
    constexpr int K = 5120;
    constexpr int N = 17408;

    std::vector<std::unique_ptr<TensorBase>> weights;
    std::vector<ROCmPackedWeights> packed(2);
    std::vector<std::unique_ptr<ROCmQuantisedGemmKernel>> kernels;
    WorkspaceRequirements combined;

    for (int i = 0; i < 2; ++i)
    {
        weights.push_back(TestTensorFactory::createQ4_KRandom(
            {static_cast<size_t>(N), static_cast<size_t>(K)},
            static_cast<uint32_t>(501 + i)));
        ASSERT_TRUE(packWeightsToROCm(weights.back().get(), packed[i]));
        expectPackedPath(packed[i], PackedPath::NativeVNNI);
        kernels.push_back(std::make_unique<ROCmQuantisedGemmKernel>(&packed[i], 0));
        combined.merge(kernels.back()->getWorkspaceRequirements(M, N, K));
    }
    const std::array<int, 2> fused_widths{N, N};
    kernels.front()->appendFusedProjectionWorkspaceRequirements(
        combined, M, fused_widths, K);

    int single_partial_buffers = 0;
    int batched_partial_buffers = 0;
    int old_slice_named_buffers = 0;
    const std::string old_partial_prefix =
        std::string(GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL) + "_";
    for (const auto &buf : combined.buffers)
    {
        if (buf.name == GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL)
            ++single_partial_buffers;
        if (buf.name == GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL_BATCHED)
            ++batched_partial_buffers;
        if (buf.name != GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL_BATCHED &&
            buf.name.rfind(old_partial_prefix, 0) == 0)
            ++old_slice_named_buffers;
    }
    EXPECT_EQ(single_partial_buffers, 1)
        << "ROCm split-K single-projection scratch should be canonical across graph instances";
    EXPECT_EQ(batched_partial_buffers, 1)
        << "ROCm split-K GEMV-many scratch should be a single canonical arena";
    EXPECT_EQ(old_slice_named_buffers, 0)
        << "Per-kernel scatter partial names cause decode graph workspace reallocations";

    DeviceWorkspaceManager workspace(
        DeviceId::rocm(0),
        combined.total_bytes_with_alignment() + 64 * 1024 * 1024);
    ASSERT_TRUE(workspace.allocate(combined));
    ScopedGPUStream producer_stream(DeviceId::rocm(0));
    for (auto &kernel : kernels)
    {
        kernel->bindWorkspace(&workspace);
        kernel->setGPUStream(producer_stream.get());
    }

    auto input = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(M), static_cast<size_t>(K)});
    auto gate = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});
    auto up = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});
    ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0), producer_stream.get()));
    ASSERT_TRUE(gate->allocateOnDevice(DeviceId::rocm(0), producer_stream.get()));
    ASSERT_TRUE(up->allocateOnDevice(DeviceId::rocm(0), producer_stream.get()));

    std::vector<ITensorGemm::TensorProjectionDesc> projections;
    projections.emplace_back(kernels[0].get(), gate.get(), N, nullptr, "gate");
    projections.emplace_back(kernels[1].get(), up.get(), N, nullptr, "up");

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    EXPECT_TRUE(kernels.front()->multiply_fused_verifier_rows_decode_equivalent(
        input.get(), projections, M, K, nullptr, &workspace))
        << "Canonical batched split-K arena must still provide distinct per-projection partial slices";
#ifdef HAVE_ROCM
    ASSERT_EQ(
        hipStreamSynchronize(static_cast<hipStream_t>(producer_stream.get())),
        hipSuccess);
#endif

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    double batched_calls = 0.0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "2")
        {
            batched_calls += record.value;
        }
    }
    EXPECT_GE(batched_calls, 1.0)
        << "Canonical batched split-K arena should keep the graph-native GEMV-many route active";
    PerfStatsCollector::reset();

    for (auto &kernel : kernels)
        kernel->unbindWorkspace();
}

TEST(Test__ROCmQuantisedGemmSmallM, FusedQ4KQwen36FFNGateUpM2RejectsUndersizedDeclaredPartialWorkspace)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    constexpr int M = 2;
    constexpr int K = 5120;
    constexpr int N = 17408;

    std::vector<std::unique_ptr<TensorBase>> weights;
    std::vector<ROCmPackedWeights> packed(2);
    std::vector<std::unique_ptr<ROCmQuantisedGemmKernel>> kernels;
    WorkspaceRequirements combined;

    for (int i = 0; i < 2; ++i)
    {
        weights.push_back(TestTensorFactory::createQ4_KRandom(
            {static_cast<size_t>(N), static_cast<size_t>(K)},
            static_cast<uint32_t>(601 + i)));
        ASSERT_TRUE(packWeightsToROCm(weights.back().get(), packed[i]));
        expectPackedPath(packed[i], PackedPath::NativeVNNI);
        kernels.push_back(std::make_unique<ROCmQuantisedGemmKernel>(&packed[i], 0));
        combined.merge(kernels.back()->getWorkspaceRequirements(M, N, K));
    }
    const std::array<int, 2> fused_widths{N, N};
    kernels.front()->appendFusedProjectionWorkspaceRequirements(
        combined, M, fused_widths, K);

    bool shrunk_partial = false;
    for (auto &buf : combined.buffers)
    {
        if (!shrunk_partial && buf.name == GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL_BATCHED)
        {
            buf.size_bytes = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float);
            shrunk_partial = true;
        }
    }
    ASSERT_TRUE(shrunk_partial);

    DeviceWorkspaceManager workspace(
        DeviceId::rocm(0),
        combined.total_bytes_with_alignment() + 64 * 1024 * 1024);
    ASSERT_TRUE(workspace.allocate(combined));
    ScopedGPUStream producer_stream(DeviceId::rocm(0));
    for (auto &kernel : kernels)
    {
        kernel->bindWorkspace(&workspace);
        kernel->setGPUStream(producer_stream.get());
    }

    auto input = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(M), static_cast<size_t>(K)});
    ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0), producer_stream.get()));

    auto gate = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});
    auto up = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});
    ASSERT_TRUE(gate->allocateOnDevice(DeviceId::rocm(0), producer_stream.get()));
    ASSERT_TRUE(up->allocateOnDevice(DeviceId::rocm(0), producer_stream.get()));

    std::vector<ITensorGemm::TensorProjectionDesc> projections;
    projections.emplace_back(kernels[0].get(), gate.get(), N, nullptr, "gate");
    projections.emplace_back(kernels[1].get(), up.get(), N, nullptr, "up");

    EXPECT_FALSE(kernels.front()->multiply_fused_verifier_rows_decode_equivalent(
        input.get(), projections, M, K, nullptr, &workspace))
        << "Batched small-M split-K must reject undersized declared partial workspace before HIP launch";
    for (auto &kernel : kernels)
        kernel->unbindWorkspace();
}

TEST(Test__ROCmQuantisedGemmSmallM, DispatchQ4KQwen36GDNAlphaM1MatchesReference)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    runDispatchSmallMMatchesReference(
        "Q4_K native-VNNI Qwen3.6 GDN alpha M=1",
        1,
        48,
        5120,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.985f);
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedFusedQ4KQwen36GDNDecodeM1MatchesReference)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    runFusedProjectionGroupSmallMMatchesReference(
        "Q4_K native-VNNI Qwen3.6 dense GDN decode projection group",
        1,
        5120,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.985f,
        {10240, 6144, 48, 48},
        {"qkv", "z", "alpha", "beta"},
        true,
        true,
        4);
}

TEST(Test__ROCmQuantisedGemmSmallM, Qwen36GDNProjectionStageMixedQuantizedAndRawFPMatchesReference)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

#ifndef HAVE_ROCM
    GTEST_SKIP() << "HAVE_ROCM not enabled";
#else
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    constexpr int M = 1;
    constexpr int K = 5120;
    constexpr int N_QKV = 10240;
    constexpr int N_Z = 6144;
    constexpr int N_ALPHA = 48;
    constexpr int N_BETA = 48;

    auto input = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(M), static_cast<size_t>(K)},
        -0.5f,
        0.5f,
        13801);
    auto w_qkv = TestTensorFactory::createQ5_KRandom(
        {static_cast<size_t>(N_QKV), static_cast<size_t>(K)}, 13802);
    auto w_z = TestTensorFactory::createQ4_KRandom(
        {static_cast<size_t>(N_Z), static_cast<size_t>(K)}, 13803);
    auto w_a = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(N_ALPHA), static_cast<size_t>(K)},
        -0.1f,
        0.1f,
        13804);
    auto w_b = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(N_BETA), static_cast<size_t>(K)},
        -0.1f,
        0.1f,
        13805);

    auto out_qkv = TestTensorFactory::createFP32Zeros(
        {static_cast<size_t>(M), static_cast<size_t>(N_QKV)});
    auto out_z = TestTensorFactory::createFP32Zeros(
        {static_cast<size_t>(M), static_cast<size_t>(N_Z)});
    auto out_a = TestTensorFactory::createFP32Zeros(
        {static_cast<size_t>(M), static_cast<size_t>(N_ALPHA)});
    auto out_b = TestTensorFactory::createFP32Zeros(
        {static_cast<size_t>(M), static_cast<size_t>(N_BETA)});

    ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
    ASSERT_TRUE(out_qkv->allocateOnDevice(DeviceId::rocm(0)));
    ASSERT_TRUE(out_z->allocateOnDevice(DeviceId::rocm(0)));
    ASSERT_TRUE(out_a->allocateOnDevice(DeviceId::rocm(0)));
    ASSERT_TRUE(out_b->allocateOnDevice(DeviceId::rocm(0)));

    auto qkv_prepared = makeGpuPreparedGemm(
        w_qkv.get(), DeviceId::rocm(0), "blk.0.attn_qkv.weight", ModelContextId{1388});
    auto z_prepared = makeGpuPreparedGemm(
        w_z.get(), DeviceId::rocm(0), "blk.0.attn_gate.weight", ModelContextId{1388});
    auto a_prepared = makeGpuPreparedFloatingPointGemm(
        w_a.get(), DeviceId::rocm(0), "blk.0.ssm_alpha.weight", ModelContextId{1388});
    auto b_prepared = makeGpuPreparedFloatingPointGemm(
        w_b.get(), DeviceId::rocm(0), "blk.0.ssm_beta.weight", ModelContextId{1388});

    GDNProjectionStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.input = input.get();
    params.m = M;
    params.k = K;
    params.w_qkv = w_qkv.get();
    params.output_qkv = out_qkv.get();
    params.n_qkv = N_QKV;
    params.w_z = w_z.get();
    params.output_z = out_z.get();
    params.n_z = N_Z;
    params.w_a = w_a.get();
    params.output_a = out_a.get();
    params.n_a = N_ALPHA;
    params.w_b = w_b.get();
    params.output_b = out_b.get();
    params.n_b = N_BETA;
    params.gemm_qkv = qkv_prepared.kernel;
    params.gemm_z = z_prepared.kernel;
    params.gemm_a = a_prepared.kernel;
    params.gemm_b = b_prepared.kernel;

    GDNProjectionStage stage(params);
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    stage.setGPUStream(stream);

    DeviceWorkspaceManager workspace(DeviceId::rocm(0), 256 * 1024 * 1024);
    ASSERT_TRUE(workspace.allocate(stage.getWorkspaceRequirements(M, 0, K)));
    stage.bindWorkspace(&workspace);

    ROCmDeviceContext ctx(DeviceId::rocm(0), 0);
    ASSERT_TRUE(stage.execute(&ctx));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    TransferEngine::publishCurrentDeviceWrite(out_a, stream);
    TransferEngine::publishCurrentDeviceWrite(out_b, stream);

    std::vector<float> ref_a(static_cast<size_t>(M) * N_ALPHA);
    std::vector<float> ref_b(static_cast<size_t>(M) * N_BETA);
    cpuFP32GemmRef(input->data(), w_a->data(), ref_a.data(), M, N_ALPHA, K);
    cpuFP32GemmRef(input->data(), w_b->data(), ref_b.data(), M, N_BETA, K);

    std::vector<float> actual_a(out_a->data(), out_a->data() + ref_a.size());
    std::vector<float> actual_b(out_b->data(), out_b->data() + ref_b.size());
    expectNearFP32(actual_a, ref_a, 1e-3f, "alpha");
    expectNearFP32(actual_b, ref_b, 1e-3f, "beta");

    stage.unbindWorkspace();
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
}

/**
 * @brief Prove captured ROCm GDN separates BF16 and FP32 floating weights.
 *
 * Qwen3.5 uses BF16 qkv/z matrices and FP32 alpha/beta matrices. The ROCm
 * floating GEMM class represents both formats, so a production grouping key
 * must include the exported physical weight type rather than only the dynamic
 * implementation class. The captured stage must form two fused subgroups and
 * retain CPU-reference numerical agreement without per-projection replay.
 */
TEST(Test__ROCmQuantisedGemmSmallM, GDNProjectionStageCapturedMixedBF16AndFP32UsesPhysicalFormatSubgroups)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

#ifndef HAVE_ROCM
    GTEST_SKIP() << "HAVE_ROCM not enabled";
#else
    constexpr int kM = 1;
    constexpr int kK = 192;
    constexpr int kNQKV = 160;
    constexpr int kNZ = 96;
    constexpr int kNAlpha = 12;
    constexpr int kNBeta = 12;
    const DeviceId device = DeviceId::rocm(0);

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();
    ASSERT_EQ(hipSetDevice(device.ordinal), hipSuccess);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

    auto input = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(kM), static_cast<size_t>(kK)}, -0.75f, 0.75f, 18831);
    auto weights_qkv = TestTensorFactory::createBF16Random(
        {static_cast<size_t>(kNQKV), static_cast<size_t>(kK)}, -0.1f, 0.1f, 18832);
    auto weights_z = TestTensorFactory::createBF16Random(
        {static_cast<size_t>(kNZ), static_cast<size_t>(kK)}, -0.1f, 0.1f, 18833);
    auto weights_alpha = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(kNAlpha), static_cast<size_t>(kK)}, -0.1f, 0.1f, 18834);
    auto weights_beta = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(kNBeta), static_cast<size_t>(kK)}, -0.1f, 0.1f, 18835);

    auto output_qkv = TestTensorFactory::createFP32Zeros(
        {static_cast<size_t>(kM), static_cast<size_t>(kNQKV)});
    auto output_z = TestTensorFactory::createFP32Zeros(
        {static_cast<size_t>(kM), static_cast<size_t>(kNZ)});
    auto output_alpha = TestTensorFactory::createFP32Zeros(
        {static_cast<size_t>(kM), static_cast<size_t>(kNAlpha)});
    auto output_beta = TestTensorFactory::createFP32Zeros(
        {static_cast<size_t>(kM), static_cast<size_t>(kNBeta)});

    ASSERT_TRUE(input->ensureOnDevice(device, stream));
    ASSERT_TRUE(output_qkv->allocateOnDevice(device, stream));
    ASSERT_TRUE(output_z->allocateOnDevice(device, stream));
    ASSERT_TRUE(output_alpha->allocateOnDevice(device, stream));
    ASSERT_TRUE(output_beta->allocateOnDevice(device, stream));

    auto prepared_qkv = makeGpuPreparedFloatingPointGemm(
        weights_qkv.get(), device, "blk.46.attn_qkv.weight", ModelContextId{1883});
    auto prepared_z = makeGpuPreparedFloatingPointGemm(
        weights_z.get(), device, "blk.46.attn_gate.weight", ModelContextId{1883});
    auto prepared_alpha = makeGpuPreparedFloatingPointGemm(
        weights_alpha.get(), device, "blk.46.ssm_alpha.weight", ModelContextId{1883});
    auto prepared_beta = makeGpuPreparedFloatingPointGemm(
        weights_beta.get(), device, "blk.46.ssm_beta.weight", ModelContextId{1883});
    ASSERT_NE(prepared_qkv.kernel, nullptr);
    ASSERT_NE(prepared_z.kernel, nullptr);
    ASSERT_NE(prepared_alpha.kernel, nullptr);
    ASSERT_NE(prepared_beta.kernel, nullptr);

    GDNProjectionStage::Params params;
    params.device_id = device;
    params.input = input.get();
    params.m = kM;
    params.k = kK;
    params.w_qkv = weights_qkv.get();
    params.output_qkv = output_qkv.get();
    params.n_qkv = kNQKV;
    params.w_z = weights_z.get();
    params.output_z = output_z.get();
    params.n_z = kNZ;
    params.w_a = weights_alpha.get();
    params.output_a = output_alpha.get();
    params.n_a = kNAlpha;
    params.w_b = weights_beta.get();
    params.output_b = output_beta.get();
    params.n_b = kNBeta;
    params.gemm_qkv = prepared_qkv.kernel;
    params.gemm_z = prepared_z.kernel;
    params.gemm_a = prepared_alpha.kernel;
    params.gemm_b = prepared_beta.kernel;

    GDNProjectionStage stage(params);
    stage.setGPUStream(stream);
    const WorkspaceRequirements requirements =
        stage.getWorkspaceRequirements(kM, 0, kK);
    DeviceWorkspaceManager workspace(
        device, requirements.total_bytes_with_alignment() + 64 * 1024 * 1024);
    ASSERT_TRUE(workspace.allocate(requirements));
    stage.bindWorkspace(&workspace);
    ROCmDeviceContext context(device, device.ordinal);

    ASSERT_TRUE(stage.execute(&context));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    auto captured_graph = std::make_unique<HIPGraphCapture>(stream, device.ordinal);
    ScopedBackendGraphCapture capture_transaction(
        *captured_graph, "ROCm mixed-floating GDN physical-format subgroups");
    ASSERT_TRUE(capture_transaction.begin());
    const bool capture_launch_ok = stage.execute(&context);
    capture_transaction.finish();
    ASSERT_TRUE(capture_launch_ok);
    ASSERT_TRUE(captured_graph->instantiate());
    ASSERT_TRUE(captured_graph->launch());
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    TransferEngine::publishCurrentDeviceWrite(output_qkv, stream);
    TransferEngine::publishCurrentDeviceWrite(output_z, stream);
    TransferEngine::publishCurrentDeviceWrite(output_alpha, stream);
    TransferEngine::publishCurrentDeviceWrite(output_beta, stream);

    std::vector<float> reference_qkv(static_cast<size_t>(kM) * kNQKV);
    std::vector<float> reference_z(static_cast<size_t>(kM) * kNZ);
    std::vector<float> reference_alpha(static_cast<size_t>(kM) * kNAlpha);
    std::vector<float> reference_beta(static_cast<size_t>(kM) * kNBeta);
    cpuFP32GemmRef(input->data(), weights_qkv->data(), reference_qkv.data(), kM, kNQKV, kK);
    cpuFP32GemmRef(input->data(), weights_z->data(), reference_z.data(), kM, kNZ, kK);
    cpuFP32GemmRef(input->data(), weights_alpha->data(), reference_alpha.data(), kM, kNAlpha, kK);
    cpuFP32GemmRef(input->data(), weights_beta->data(), reference_beta.data(), kM, kNBeta, kK);

    auto check_projection = [&](const char *name,
                                FP32Tensor *actual,
                                const std::vector<float> &reference)
    {
        const float cosine = cosineSim(actual->data(), reference.data(), reference.size());
        const float relative_l2 = relativeL2(actual->data(), reference.data(), reference.size());
        EXPECT_GT(cosine, 0.99999f) << name;
        EXPECT_LT(relative_l2, 0.001f) << name;
    };
    check_projection("captured mixed-float GDN qkv", output_qkv.get(), reference_qkv);
    check_projection("captured mixed-float GDN z", output_z.get(), reference_z);
    check_projection("captured mixed-float GDN alpha", output_alpha.get(), reference_alpha);
    check_projection("captured mixed-float GDN beta", output_beta.get(), reference_beta);

    const auto routes = PerfStatsCollector::snapshot({"kernel.gdn_projection_route"});
    auto has_pair = [&](const char *names)
    {
        return std::any_of(
            routes.begin(), routes.end(),
            [&](const PerfStatRecord &record)
            {
                const auto route = record.tags.find("route");
                const auto projection_names = record.tags.find("names");
                return record.domain == "kernel" &&
                       record.name == "gdn_projection_route" &&
                       route != record.tags.end() &&
                       route->second == "same_kernel_mixed_codebook_subgroup" &&
                       projection_names != record.tags.end() &&
                       projection_names->second == names;
            });
    };
    EXPECT_TRUE(has_pair("qkv+z"));
    EXPECT_TRUE(has_pair("alpha+beta"));

    captured_graph.reset();
    stage.unbindWorkspace();
    prepared_qkv.kernel->clearGPUStreamBinding();
    prepared_z.kernel->clearGPUStreamBinding();
    prepared_alpha.kernel->clearGPUStreamBinding();
    prepared_beta.kernel->clearGPUStreamBinding();
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
    PerfStatsCollector::reset();
#endif
}

TEST(Test__ROCmQuantisedGemmSmallM, Qwen36MoEGDNProjectionStageQ6KVerifierRowsMatchSerialDecodeStrict)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

#ifndef HAVE_ROCM
    GTEST_SKIP() << "HAVE_ROCM not enabled";
#else
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    /*
     * Qwen3.6 MoE GDN layers use a smaller hidden width than the dense model
     * and Q6_K for both the fused QKV projection and the Z projection:
     *
     *   attn_qkv: 8192 x 2048
     *   attn_gate: 4096 x 2048
     *   ssm_alpha/beta: 32 x 2048, raw FP32
     *
     * The all-position MTP verifier accepts runtime row counts, so the grouped
     * stage must match the row-by-row decode contract at each important
     * capacity/tile boundary before it is allowed into the model graph.
     */
    constexpr int K = 2048;
    constexpr int N_QKV = 8192;
    constexpr int N_Z = 4096;
    constexpr int N_ALPHA = 32;
    constexpr int N_BETA = 32;

    auto w_qkv = TestTensorFactory::createQ6_KRandom(
        {static_cast<size_t>(N_QKV), static_cast<size_t>(K)}, 23802);
    auto w_z = TestTensorFactory::createQ6_KRandom(
        {static_cast<size_t>(N_Z), static_cast<size_t>(K)}, 23803);
    auto w_a = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(N_ALPHA), static_cast<size_t>(K)},
        -0.1f,
        0.1f,
        23804);
    auto w_b = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(N_BETA), static_cast<size_t>(K)},
        -0.1f,
        0.1f,
        23805);

    auto qkv_prepared = makeGpuPreparedGemm(
        w_qkv.get(), DeviceId::rocm(0), "blk.5.attn_qkv.weight", ModelContextId{2388});
    auto z_prepared = makeGpuPreparedGemm(
        w_z.get(), DeviceId::rocm(0), "blk.5.attn_gate.weight", ModelContextId{2388});
    auto a_prepared = makeGpuPreparedFloatingPointGemm(
        w_a.get(), DeviceId::rocm(0), "blk.5.ssm_alpha.weight", ModelContextId{2388});
    auto b_prepared = makeGpuPreparedFloatingPointGemm(
        w_b.get(), DeviceId::rocm(0), "blk.5.ssm_beta.weight", ModelContextId{2388});

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    ASSERT_NE(stream, nullptr);
    qkv_prepared.kernel->setGPUStream(stream);
    z_prepared.kernel->setGPUStream(stream);
    a_prepared.kernel->setGPUStream(stream);
    b_prepared.kernel->setGPUStream(stream);

    for (const int M : kGroupedVerifierBoundaryRows)
    {
        auto input = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)},
            -0.35f,
            0.35f,
            23900u + static_cast<uint32_t>(M));
        auto out_qkv = TestTensorFactory::createFP32Zeros(
            {static_cast<size_t>(M), static_cast<size_t>(N_QKV)});
        auto out_z = TestTensorFactory::createFP32Zeros(
            {static_cast<size_t>(M), static_cast<size_t>(N_Z)});
        auto out_a = TestTensorFactory::createFP32Zeros(
            {static_cast<size_t>(M), static_cast<size_t>(N_ALPHA)});
        auto out_b = TestTensorFactory::createFP32Zeros(
            {static_cast<size_t>(M), static_cast<size_t>(N_BETA)});

        ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0), stream));
        ASSERT_TRUE(out_qkv->allocateOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(out_z->allocateOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(out_a->allocateOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(out_b->allocateOnDevice(DeviceId::rocm(0)));

        GDNProjectionStage::Params params;
        params.device_id = DeviceId::rocm(0);
        params.input = input.get();
        params.m = M;
        params.k = K;
        params.w_qkv = w_qkv.get();
        params.output_qkv = out_qkv.get();
        params.n_qkv = N_QKV;
        params.w_z = w_z.get();
        params.output_z = out_z.get();
        params.n_z = N_Z;
        params.w_a = w_a.get();
        params.output_a = out_a.get();
        params.n_a = N_ALPHA;
        params.w_b = w_b.get();
        params.output_b = out_b.get();
        params.n_b = N_BETA;
        params.gemm_qkv = qkv_prepared.kernel;
        params.gemm_z = z_prepared.kernel;
        params.gemm_a = a_prepared.kernel;
        params.gemm_b = b_prepared.kernel;
        params.force_decode_equivalent_verifier_prefill = true;

        GDNProjectionStage stage(params);
        stage.setGPUStream(stream);

        DeviceWorkspaceManager workspace(DeviceId::rocm(0), 256 * 1024 * 1024);
        ASSERT_TRUE(workspace.allocate(stage.getWorkspaceRequirements(M, 0, K)));
        stage.bindWorkspace(&workspace);

        ROCmDeviceContext ctx(DeviceId::rocm(0), 0);
        ASSERT_TRUE(stage.execute(&ctx));
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        TransferEngine::publishCurrentDeviceWrite(out_qkv, stream);
        TransferEngine::publishCurrentDeviceWrite(out_z, stream);
        TransferEngine::publishCurrentDeviceWrite(out_a, stream);
        TransferEngine::publishCurrentDeviceWrite(out_b, stream);

        const struct ProjectionCase
        {
            ITensorGemm *kernel;
            FP32Tensor *grouped;
            int n;
            const char *name;
        } cases[] = {
            {qkv_prepared.kernel, out_qkv.get(), N_QKV, "qkv"},
            {z_prepared.kernel, out_z.get(), N_Z, "z"},
            {a_prepared.kernel, out_a.get(), N_ALPHA, "alpha"},
            {b_prepared.kernel, out_b.get(), N_BETA, "beta"},
        };

        for (int row = 0; row < M; ++row)
        {
            auto row_input = TestTensorFactory::createFP32({1u, static_cast<size_t>(K)});
            std::copy(input->data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                      input->data() + static_cast<size_t>(row + 1) * static_cast<size_t>(K),
                      row_input->mutable_data());
            ASSERT_TRUE(row_input->ensureOnDevice(DeviceId::rocm(0), stream));

            for (const ProjectionCase &projection : cases)
            {
                auto serial = TestTensorFactory::createFP32(
                    {1u, static_cast<size_t>(projection.n)});
                ASSERT_TRUE(serial->allocateOnDevice(DeviceId::rocm(0)));
                ASSERT_TRUE(projection.kernel->multiply_tensor(
                    row_input.get(),
                    serial.get(),
                    1,
                    projection.n,
                    K,
                    true,
                    1.0f,
                    0.0f,
                    nullptr,
                    nullptr,
                    -1,
                    &workspace))
                    << "projection=" << projection.name << " row=" << row;
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
                TransferEngine::publishCurrentDeviceWrite(serial, stream);

                const float *grouped_row =
                    projection.grouped->data() + static_cast<size_t>(row) * static_cast<size_t>(projection.n);
                const float *serial_row = serial->data();
                const size_t count = static_cast<size_t>(projection.n);
                const float cos = cosineSim(grouped_row, serial_row, count);
                const float rel_l2 = relativeL2(grouped_row, serial_row, count);
                const float max_abs = maxAbsDiff(grouped_row, serial_row, count);
                const double skl = symmetricSoftmaxKL(grouped_row, serial_row, count);

                LOG_INFO("[SmallM] Qwen3.6 MoE GDN projection=" << projection.name
                                                                << " M=" << M
                                                                << " row=" << row
                                                                << " cosine=" << cos
                                                                << " rel_l2=" << rel_l2
                                                                << " symmetric_kl=" << skl
                                                                << " max_abs=" << max_abs);
                expectBitwiseFP32RowsEqual(
                    std::string("Qwen3.6 MoE GDN projection=") +
                        projection.name + " grouped verifier row=" +
                        std::to_string(row) + " M=" + std::to_string(M),
                    grouped_row,
                    serial_row,
                    count,
                    count);
            }
        }

        stage.unbindWorkspace();
    }

    qkv_prepared.kernel->clearGPUStreamBinding();
    z_prepared.kernel->clearGPUStreamBinding();
    a_prepared.kernel->clearGPUStreamBinding();
    b_prepared.kernel->clearGPUStreamBinding();
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedFusedQ4KGDNProjectionM2MatchesSeparate)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    runFusedProjectionGroupSmallMMatchesSeparate(
        "Q4_K native-VNNI Qwen3.6 GDN projection group",
        2,
        5120,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.9999f,
        {10240, 10240, 1024, 1024},
        true,
        true,
        4,
        true);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    double batched_calls = 0.0;
    double batched_projection_calls = 0.0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "4")
        {
            batched_calls += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_projection_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120")
        {
            batched_projection_calls += record.value;
        }
    }

    EXPECT_GE(batched_calls, 1.0)
        << "Graph-captured Qwen3.6 GDN projection group should use one batched native route";
    EXPECT_GE(batched_projection_calls, 4.0)
        << "Batched GDN route should cover qkv/z/alpha/beta projection payloads";

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedFusedQ4KGDNProjectionM4MatchesSeparate)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    runFusedProjectionGroupSmallMMatchesSeparate(
        "Q4_K native-VNNI Qwen3.6 GDN projection shifted-prefill group",
        4,
        5120,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.9999f,
        {10240, 10240, 1024, 1024},
        true,
        true,
        4,
        true);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    double batched_calls = 0.0;
    double batched_projection_calls = 0.0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "4" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "4")
        {
            batched_calls += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_projection_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "4" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120")
        {
            batched_projection_calls += record.value;
        }
    }

    EXPECT_GE(batched_calls, 1.0)
        << "Graph-captured Qwen3.6 M=4 GDN projection group should use one batched native route";
    EXPECT_GE(batched_projection_calls, 4.0)
        << "Batched M=4 GDN route should cover qkv/z/alpha/beta projection payloads";

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedFusedQ4KQwen36GDNQkvZPairM2UsesHeterogeneousNBatchedRoute)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    runFusedProjectionGroupSmallMMatchesSeparate(
        "Q4_K native-VNNI Qwen3.6 GDN qkv/z heterogeneous-N pair",
        2,
        5120,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.9999f,
        {10240, 6144},
        true,
        true,
        4,
        true);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    double batched_calls = 0.0;
    double heterogeneous_n_bypasses = 0.0;
    double shared_quant_calls = 0.0;
    double batched_projection_calls = 0.0;
    double graph_direct_launches = 0.0;
    double graph_split_reduce_launches = 0.0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "2")
        {
            batched_calls += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_bypasses" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("reason") != 0 &&
            record.tags.at("reason") == "heterogeneous_n_pair" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "2")
        {
            heterogeneous_n_bypasses += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_fused_small_m_shared_quant_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "2")
        {
            shared_quant_calls += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_projection_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120")
        {
            batched_projection_calls += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_launch" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120" &&
            record.tags.count("batched") != 0 &&
            record.tags.at("batched") == "true")
        {
            if (record.tags.count("path") != 0 &&
                record.tags.at("path") == "direct")
            {
                graph_direct_launches += record.value;
            }
            if (record.tags.count("path") != 0 &&
                record.tags.at("path") == "split_reduce")
            {
                graph_split_reduce_launches += record.value;
            }
        }
    }

    EXPECT_GE(batched_calls, 1.0)
        << "Real Qwen3.6 GDN qkv/z should use the graph-captured heterogeneous-N batched route";
    EXPECT_EQ(heterogeneous_n_bypasses, 0.0)
        << "The heterogeneous-N qkv/z shape should be handled by the generic batched kernel, not bypassed";
    EXPECT_GE(shared_quant_calls, 1.0)
        << "The batched qkv/z subgroup should quantize activations once";
    EXPECT_GE(batched_projection_calls, 2.0)
        << "The batched qkv/z subgroup should cover both heterogeneous-N projection payloads";
    EXPECT_GE(graph_direct_launches + graph_split_reduce_launches, 1.0)
        << "GPU-graph Qwen3.6 batched GDN qkv/z must use a graph-native ordered verifier launch";

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedFusedMixedCodebookGDNProjectionM4UsesMixedBatchedRoute)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    std::vector<WeightCreator> creators = {
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ5_KRandom(shape, seed); },
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ5_KRandom(shape, seed); }};

    runMixedProjectionGroupSmallMMatchesSeparate(
        "mixed Q4_K/Q5_K native-VNNI Qwen3.6 GDN projection group",
        4,
        5120,
        creators,
        0.9999f,
        {10240, 10240, 1024, 1024},
        true,
        4,
        true);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    double mixed_batched_calls = 0.0;
    double full_mixed_bypasses = 0.0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "4" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "4" &&
            record.tags.count("codebook") != 0 &&
            record.tags.at("codebook") == "mixed")
        {
            mixed_batched_calls += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_mixed_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "4" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "4")
        {
            mixed_batched_calls += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_bypasses" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("reason") != 0 &&
            record.tags.at("reason") == "mixed_codebook" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "4")
        {
            full_mixed_bypasses += record.value;
        }
    }

    EXPECT_GE(mixed_batched_calls, 1.0)
        << "Mixed-codebook GDN groups should use the graph-capturable mixed native small-M batched route";
    EXPECT_EQ(full_mixed_bypasses, 0.0)
        << "Mixed-codebook GDN groups should no longer bypass to per-projection small-M GEMV";

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedMixedCodebookQwen36GDNQkvZPairM4MatchesSeparate)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    std::vector<WeightCreator> creators = {
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ5_KRandom(shape, seed); },
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); }};

    runMixedProjectionGroupSmallMMatchesSeparate(
        "mixed Q5_K/Q4_K native-VNNI Qwen3.6 GDN qkv/z heterogeneous-N pair",
        4,
        5120,
        creators,
        0.9999f,
        {10240, 6144},
        true,
        4,
        true);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    double mixed_bypasses = 0.0;
    double batched_calls = 0.0;
    double mixed_batched_calls = 0.0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_bypasses" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("reason") != 0 &&
            record.tags.at("reason") == "mixed_codebook" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "2" &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "4")
        {
            mixed_bypasses += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "4" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "2" &&
            record.tags.count("codebook") != 0 &&
            record.tags.at("codebook") == "mixed")
        {
            batched_calls += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_mixed_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "4" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "2")
        {
            mixed_batched_calls += record.value;
        }
    }

    EXPECT_EQ(mixed_bypasses, 0.0)
        << "Mixed-codebook Qwen3.6 qkv/z M=4 should no longer bypass the batched route";
    EXPECT_GE(batched_calls, 1.0)
        << "Mixed-codebook Qwen3.6 qkv/z M=4 must enter the mixed batched route";
    EXPECT_GE(mixed_batched_calls, 1.0)
        << "Mixed-codebook Qwen3.6 qkv/z M=4 must record the mixed route counter";

    PerfStatsCollector::reset();
}

/**
 * @test Certify the mixed-codebook fused launcher at every default verifier depth.
 *
 * The production mixed launcher selects a codebook independently for each
 * projection while sharing one runtime row grid.  Small dimensions keep this
 * exhaustive depth proof inexpensive; the model-shaped tests below retain
 * their separate large-K coverage.
 */
TEST(Test__ROCmQuantisedGemmSmallM, MixedCodebookProjectionRuntimeMMatchesSerialM1DecodeRowsStrict)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const std::vector<WeightCreator> creators = {
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ5_KRandom(shape, seed); },
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); }};

    constexpr int K = 512;
    for (const int M : kGroupedVerifierRuntimeRows)
    {
        runMixedProjectionGroupSmallMMatchesSerialM1DecodeRows(
            "mixed Q5_K/Q4_K runtime verifier depth sweep",
            M,
            K,
            creators,
            {384, 256},
            {"alpha", "beta"});
    }
}

TEST(Test__ROCmQuantisedGemmSmallM, MixedCodebookQwen36GDNQkvZPairRuntimeMMatchesSerialM1DecodeRowsStrict)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    std::vector<WeightCreator> creators = {
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ5_KRandom(shape, seed); },
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); }};

    constexpr int K = 5120;
    for (const int M : kGroupedVerifierBoundaryRows)
    {
        runMixedProjectionGroupSmallMMatchesSerialM1DecodeRows(
            "mixed Q5_K/Q4_K native-VNNI Qwen3.6 GDN qkv/z strict serial decode",
            M,
            K,
            creators,
            {10240, 6144},
            {"qkv", "z"});
    }
}

TEST(
    Test__ROCmQuantisedGemmSmallM,
    MixedCodebookQwen36GDNPaddedPrefillMatchesIsolatedRequestBytes)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    std::vector<WeightCreator> creators = {
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ5_KRandom(shape, seed); },
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); }};

    runMixedProjectionPaddedPrefillMatchesIsolatedRequest(
        "mixed Q5_K/Q4_K Qwen3.6 GDN request-batched prefill",
        /*padded_rows=*/32,
        /*request_row_offset=*/16,
        /*request_rows=*/11,
        /*K=*/5120,
        creators,
        /*Ns=*/{10240, 6144},
        /*projection_names=*/{"qkv", "z"});
}

TEST(Test__ROCmQuantisedGemmSmallM, GDNIQ4XSQwen36MoEQkvZPairRuntimeMMatchesSerialM1DecodeRowsStrict)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    /*
     * The Qwen3.6 MoE GGUF advertises its GDN projection tensors as:
     *   blk.0.attn_qkv.weight  [2048, 8192] IQ4_XS
     *   blk.0.attn_gate.weight [2048, 4096] IQ4_XS
     *
     * In Llaminar's row-major GEMM convention those become N={8192,4096}
     * projections over K=2048 hidden rows.  The model-level grouped verifier
     * diagnostic first showed a visible drift downstream of layer2 GDN
     * projection, so keep this exact-shape regression strict against ordinary
     * M=1 decode rows.
     */
    std::vector<WeightCreator> creators = {
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createIQ4_XSRandom(shape, seed); },
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createIQ4_XSRandom(shape, seed); }};

    constexpr int K = 2048;
    for (const int M : kGroupedVerifierBoundaryRows)
    {
        runMixedProjectionGroupSmallMMatchesSerialM1DecodeRows(
            "IQ4_XS native-VNNI Qwen3.6 MoE GDN qkv/z strict serial decode",
            M,
            K,
            creators,
            {8192, 4096},
            {"gdn_qkv", "gdn_z"});
    }
}

/**
 * @brief Regress the ROCm grouped-prefill FP32 expression tree at every M.
 *
 * Q4_0 was the first all-codegroup witness that exposed grouped prefill using
 * algebraically similar, but differently rounded, block commits and fused
 * SwiGLU/Q8 publication. Keeping this focused case separate from the Cartesian
 * sweep makes that structural defect cheap to reproduce while still crossing
 * the route-owned/expert-tiled boundary and the extended M=31 probe.
 */
TEST(Test__ROCmQuantisedGemmSmallM,
     ROCmMoERoutedVerifierRuntimeM_Q4_0MatchesSerialDecodeBytes)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    const auto formats = nativeMoECodegroupCases();
    ASSERT_FALSE(formats.empty());
    ASSERT_EQ(formats.front().codebook_id, 0u);
    ASSERT_STREQ(formats.front().label, "Q4_0");

    runROCmMoECodegroupVerifierRowsMatchSerialDecode(
        formats.front(), formats.front(), kGroupedVerifierRuntimeRows);
    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, ROCmMoERoutedVerifierRuntimeM_AllNativeCodegroupsMatchSerialDecode)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");

    /*
     * The grouped routed MoE verifier path is a composition of codebook
     * dispatch, compact route planning, gate/up projection, SwiGLU Q8
     * publication, and down projection publication.  Gate/up descriptors must
     * share one codebook, but down descriptors are independent in production
     * models.  Sweep the full gate/up-codegroup x down-codegroup matrix so mixed
     * GGUF layouts such as IQ2_S gate/up with IQ4_XS down cannot escape behind
     * same-format coverage.
     */
    const auto formats = nativeMoECodegroupCases();
    for (const auto &gateup_format : formats)
    {
        for (const auto &down_format : formats)
        {
            SCOPED_TRACE(std::string(gateup_format.label) + "_gateup/" +
                         down_format.label + "_down");
            /*
             * Every native codebook receives the complete runtime-M sweep when
             * it owns all three expert projections. Mixed gate/up and down
             * codegroups cover every ordered pairing at tile/capacity
             * boundaries, including M=31 beyond the default graph bucket.
             */
            if (gateup_format.codebook_id == down_format.codebook_id)
                runROCmMoECodegroupVerifierRowsMatchSerialDecode(
                    gateup_format,
                    down_format,
                    kGroupedVerifierRuntimeRows);
            else
                runROCmMoECodegroupVerifierRowsMatchSerialDecode(
                    gateup_format,
                    down_format,
                    kGroupedVerifierBoundaryRows);
        }
    }

    PerfStatsCollector::reset();
}
