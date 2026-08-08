#include <gtest/gtest.h>
#include "transfer/TransferEngine.h"

#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

#include "../../../mocks/MockComputeStage.h"
#include "../../../utils/GpuPreparedGemmHarness.h"
#include "../../../utils/NativeVNNITrainerEvidence.h"
#include "../../../utils/QuantizedVerifierFormats.h"
#include "../../../utils/TestTensorFactory.h"
#include "../native_vnni_dispatch/NativeVNNIMoEPrefillManifest.h"
#include "../native_vnni_dispatch/NativeVNNIMoERoutingProfiles.h"
#include "../native_vnni_dispatch/NativeVNNIProfilerControl.h"

#ifdef HAVE_CUDA
#include "backends/cuda/CUDAGraphCapture.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/cuda/gemm/CUDAMoEGroupedPrefillKernels.h"
#include "kernels/cuda/moe/CUDAMoEBatchInvariantPolicy.h"
#include "kernels/cuda/moe/CUDAMoEKernel.h"
#include "../native_vnni_dispatch/GPUTrainerVerification.h"

#include <cuda_runtime.h>
#include <dlfcn.h>

extern "C" bool cudaMoE_grouped_prefill_query_kernel_resources(
    uint8_t codebook_id,
    int component,
    int block_threads,
    int *registers_per_thread,
    size_t *local_memory_bytes_per_thread,
    size_t *static_shared_memory_bytes,
    int *max_threads_per_block,
    int *max_active_blocks_per_sm);
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

/**
 * @file Perf__MoEVerifierPrefill.cpp
 * @brief Focused MoE verifier-prefill parity and timing harness.
 *
 * This target isolates the Qwen3.6 MoE MTP verifier hot path: small verifier
 * batches across the complete production MTP depth range, routed top-k
 * experts, and the always-on shared expert. Each case compares grouped
 * verifier prefill against row-wise decode-equivalent execution byte for byte,
 * then reports eager and graph-replay timing in a compact CSV row.
 *
 * The harness is deliberately narrower than full-model benchmark mode. It gives
 * us a stable speedometer for kernel and grouping changes before we spend time
 * rerunning the expensive dense/MoE iteration matrix.
 */

namespace
{
    using KernelFactory = llaminar::v2::kernels::KernelFactory;
    using llaminar2::TransferEngine;

    struct CloseMetrics
    {
        double cosine = 0.0;
        double relative_l2 = 0.0;
        double max_abs = 0.0;
        double min_row_cosine = 1.0;
        double max_row_relative_l2 = 0.0;
        double max_row_kl = 0.0;
        size_t nonfinite_count = 0;
        size_t nonfinite_actual_count = 0;
        size_t nonfinite_expected_count = 0;
        size_t first_nonfinite_index = 0;
        size_t bit_mismatch_count = 0;
        size_t first_bit_mismatch_index = 0;
        uint32_t first_actual_bits = 0;
        uint32_t first_expected_bits = 0;
        size_t worst_row = 0;
    };

    struct BenchResult
    {
        std::string backend;
        std::string case_name;
        int m = 0;
        int top_k = 0;
        int num_experts = 0;
        int d_model = 0;
        int intermediate = 0;
        double eager_ms = 0.0;
        double prepare_ms = 0.0;
        double pipeline_ms = 0.0;
        double graph_ms = 0.0;
        double rowwise_ms = 0.0;
        CloseMetrics metrics;
    };

    struct QuantFormatCase
    {
        const char *name = "";
        std::function<std::unique_ptr<llaminar2::TensorBase>(
            const std::vector<size_t> &, uint32_t)> create;
    };

    std::vector<QuantFormatCase> sharedExpertPreparedFormatCases()
    {
        using llaminar2::test::TestTensorFactory;
        return {
            {"Q4_0", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<llaminar2::TensorBase>
             { return TestTensorFactory::createQ4_0Random(shape, seed); }},
            {"Q4_1", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<llaminar2::TensorBase>
             { return TestTensorFactory::createQ4_1Random(shape, seed); }},
            {"Q5_0", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<llaminar2::TensorBase>
             { return TestTensorFactory::createQ5_0Random(shape, seed); }},
            {"Q5_1", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<llaminar2::TensorBase>
             { return TestTensorFactory::createQ5_1Random(shape, seed); }},
            {"Q2_K", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<llaminar2::TensorBase>
             { return TestTensorFactory::createQ2_KRandom(shape, seed); }},
            {"Q3_K", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<llaminar2::TensorBase>
             { return TestTensorFactory::createQ3_KRandom(shape, seed); }},
            {"Q4_K", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<llaminar2::TensorBase>
             { return TestTensorFactory::createQ4_KRandom(shape, seed); }},
            {"Q5_K", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<llaminar2::TensorBase>
             { return TestTensorFactory::createQ5_KRandom(shape, seed); }},
            {"Q6_K", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<llaminar2::TensorBase>
             { return TestTensorFactory::createQ6_KRandom(shape, seed); }},
            {"IQ4_NL", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<llaminar2::TensorBase>
             { return TestTensorFactory::createIQ4_NLRandom(shape, seed); }},
            {"IQ4_XS", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<llaminar2::TensorBase>
             { return TestTensorFactory::createIQ4_XSRandom(shape, seed); }},
            {"IQ3_S", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<llaminar2::TensorBase>
             { return TestTensorFactory::createIQ3_SRandom(shape, seed); }},
            {"IQ3_XXS", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<llaminar2::TensorBase>
             { return TestTensorFactory::createIQ3_XXSRandom(shape, seed); }},
            {"IQ2_S", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<llaminar2::TensorBase>
             { return TestTensorFactory::createIQ2_SRandom(shape, seed); }},
            {"IQ2_XS", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<llaminar2::TensorBase>
             { return TestTensorFactory::createIQ2_XSRandom(shape, seed); }},
            {"IQ2_XXS", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<llaminar2::TensorBase>
             { return TestTensorFactory::createIQ2_XXSRandom(shape, seed); }},
            {"IQ1_S", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<llaminar2::TensorBase>
             { return TestTensorFactory::createIQ1_SRandom(shape, seed); }},
            {"IQ1_M", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<llaminar2::TensorBase>
             { return TestTensorFactory::createIQ1_MRandom(shape, seed); }},
            {"Q8_0", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<llaminar2::TensorBase>
             { return TestTensorFactory::createQ8_0Random(shape, seed); }},
        };
    }

    const QuantFormatCase &sharedExpertPreparedFormatCase(const char *name)
    {
        static const std::vector<QuantFormatCase> cases = sharedExpertPreparedFormatCases();
        const auto it = std::find_if(
            cases.begin(), cases.end(),
            [name](const QuantFormatCase &tc)
            {
                return std::string(tc.name) == name;
            });
        if (it == cases.end())
            throw std::runtime_error(std::string("unknown shared expert format case: ") + name);
        return *it;
    }

    /**
     * @brief Temporarily override one environment variable.
     *
     * The MoE kernels read a few debug/tuning toggles through DebugEnv. Keeping
     * the override scoped avoids accidental cross-test policy leakage.
     */
    class ScopedEnvOverride
    {
    public:
        ScopedEnvOverride(const char *name, const char *value)
            : name_(name)
        {
            const char *old = std::getenv(name);
            if (old)
            {
                had_old_ = true;
                old_ = old;
            }
            setenv(name, value, 1);
        }

        ~ScopedEnvOverride()
        {
            if (had_old_)
                setenv(name_.c_str(), old_.c_str(), 1);
            else
                unsetenv(name_.c_str());
        }

        ScopedEnvOverride(const ScopedEnvOverride &) = delete;
        ScopedEnvOverride &operator=(const ScopedEnvOverride &) = delete;

    private:
        std::string name_;
        bool had_old_ = false;
        std::string old_;
    };

    int envInt(const char *name, int fallback)
    {
        const char *value = std::getenv(name);
        if (!value || !*value)
            return fallback;
        char *end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value || parsed <= 0)
            return fallback;
        return static_cast<int>(parsed);
    }

    /** Parse a strictly positive finite environment scalar. */
    double envPositiveDouble(const char *name, double fallback)
    {
        const char *value = std::getenv(name);
        if (!value || !*value)
            return fallback;
        char *end = nullptr;
        const double parsed = std::strtod(value, &end);
        if (end == value || *end != '\0' || !std::isfinite(parsed) ||
            parsed <= 0.0)
        {
            throw std::runtime_error(
                std::string(name) + " must be a positive finite scalar");
        }
        return parsed;
    }

    /** Parse a comma-separated positive integer inventory without duplicates. */
    std::vector<int> envCsvPositiveInts(
        const char *name,
        std::initializer_list<int> fallback)
    {
        const char *value = std::getenv(name);
        if (!value || !*value)
            return std::vector<int>(fallback);

        std::vector<int> result;
        std::string csv(value);
        size_t start = 0;
        while (start <= csv.size())
        {
            const size_t comma = csv.find(',', start);
            const std::string token = csv.substr(
                start,
                comma == std::string::npos
                    ? std::string::npos
                    : comma - start);
            char *end = nullptr;
            const long parsed = std::strtol(token.c_str(), &end, 10);
            if (token.empty() || end == token.c_str() || *end != '\0' ||
                parsed <= 0 || parsed > std::numeric_limits<int>::max())
            {
                throw std::runtime_error(
                    std::string(name) + " contains an invalid positive integer");
            }
            const int integer = static_cast<int>(parsed);
            if (std::find(result.begin(), result.end(), integer) != result.end())
            {
                throw std::runtime_error(
                    std::string(name) + " contains a duplicate value");
            }
            result.push_back(integer);
            if (comma == std::string::npos)
                break;
            start = comma + 1;
        }
        return result;
    }

    bool envCsvContainsOrUnset(const char *name, const std::string &candidate)
    {
        const char *value = std::getenv(name);
        if (!value || !*value)
            return true;

        std::string csv(value);
        size_t start = 0;
        while (start <= csv.size())
        {
            const size_t comma = csv.find(',', start);
            std::string item = csv.substr(
                start,
                comma == std::string::npos ? std::string::npos : comma - start);
            item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char ch)
            {
                return !std::isspace(ch);
            }));
            item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char ch)
            {
                return !std::isspace(ch);
            }).base(), item.end());
            if (item == candidate)
                return true;
            if (comma == std::string::npos)
                break;
            start = comma + 1;
        }
        return false;
    }

    /**
     * @brief Resolve an optional exact runtime-M inventory for CUDA profiling.
     *
     * Full correctness suites own contiguous M-totality. This focused speedometer
     * also needs to isolate one production depth, such as M=5, so Nsight can
     * attribute one launch geometry without measuring every default cell first.
     * An explicitly supplied malformed or non-positive value is a configuration
     * error: silently reverting to defaults would produce convincing timing for
     * the wrong verifier shape.
     *
     * @param defaults Test-specific row inventory used when the environment is
     *        absent.
     * @return Requested positive row counts in caller order, with duplicates
     *         removed.
     * @throws std::runtime_error when the selector contains an invalid token.
     */
    std::vector<int> selectedVerifierRows(
        std::initializer_list<int> defaults)
    {
        constexpr const char *kEnvironment =
            "LLAMINAR_MOE_VERIFIER_PREFILL_ROWS";
        const char *value = std::getenv(kEnvironment);
        if (!value || !*value)
            return std::vector<int>(defaults);

        std::vector<int> rows;
        std::string csv(value);
        size_t start = 0;
        while (start <= csv.size())
        {
            const size_t comma = csv.find(',', start);
            std::string item = csv.substr(
                start,
                comma == std::string::npos
                    ? std::string::npos
                    : comma - start);
            item.erase(item.begin(), std::find_if(
                item.begin(), item.end(), [](unsigned char ch)
                {
                    return !std::isspace(ch);
                }));
            item.erase(std::find_if(
                item.rbegin(), item.rend(), [](unsigned char ch)
                {
                    return !std::isspace(ch);
                }).base(), item.end());

            char *end = nullptr;
            const long parsed = std::strtol(item.c_str(), &end, 10);
            if (item.empty() || end == item.c_str() || *end != '\0' ||
                parsed <= 0 ||
                parsed > std::numeric_limits<int>::max())
            {
                throw std::runtime_error(
                    std::string(kEnvironment) +
                    " requires a comma-separated list of positive integers; got '" +
                    item + "'");
            }
            const int row_count = static_cast<int>(parsed);
            if (std::find(rows.begin(), rows.end(), row_count) == rows.end())
                rows.push_back(row_count);

            if (comma == std::string::npos)
                break;
            start = comma + 1;
        }
        return rows;
    }

    std::shared_ptr<llaminar2::FP32Tensor> makeTensor(
        const std::vector<size_t> &shape,
        const std::vector<float> &values)
    {
        auto tensor = std::make_shared<llaminar2::FP32Tensor>(shape);
        std::copy(values.begin(), values.end(), tensor->mutable_data());
        return tensor;
    }

    std::shared_ptr<llaminar2::FP32Tensor> makeZeros(const std::vector<size_t> &shape)
    {
        auto tensor = std::make_shared<llaminar2::FP32Tensor>(shape);
        std::fill(tensor->mutable_data(), tensor->mutable_data() + tensor->numel(), 0.0f);
        return tensor;
    }

    std::vector<float> makeHiddenValues(int rows, int d_model)
    {
        std::vector<float> values(static_cast<size_t>(rows) * d_model);
        for (size_t i = 0; i < values.size(); ++i)
        {
            values[i] =
                0.013f * static_cast<float>(static_cast<int>(i % 29) - 14) +
                0.002f * static_cast<float>(static_cast<int>((i / 7) % 11) - 5);
        }
        return values;
    }

    std::vector<float> makeRoutingIndices(int rows, int top_k, int num_experts)
    {
        std::vector<float> values(static_cast<size_t>(rows) * top_k);
        for (int row = 0; row < rows; ++row)
        {
            for (int k = 0; k < top_k; ++k)
            {
                // Unique routes within a row match the production router, while
                // repeated routes across rows exercise compact active-expert grids.
                values[static_cast<size_t>(row) * top_k + k] =
                    static_cast<float>((k + ((row & 1) ? 4 : 0)) % num_experts);
            }
        }
        return values;
    }

    std::vector<float> makeUniqueRoutingIndices(int rows, int top_k, int num_experts)
    {
        std::vector<float> values(static_cast<size_t>(rows) * top_k);
        for (int row = 0; row < rows; ++row)
        {
            for (int k = 0; k < top_k; ++k)
            {
                values[static_cast<size_t>(row) * top_k + k] =
                    static_cast<float>((row * top_k + k) % num_experts);
            }
        }
        return values;
    }

    /**
     * @brief Publish the separately numbered shared expert in the final route.
     *
     * A combined Qwen 3.6 verifier table has routed expert ids 0..255 and the
     * shared expert at id 256. Merely allocating 257 descriptors does not test
     * that last entry: this helper makes every row select it explicitly while
     * leaving the first @p top_k - 1 routes on their routed-expert pattern.
     *
     * @param values Mutable row-major route-id matrix.
     * @param rows Number of verifier rows in the matrix.
     * @param top_k Number of routes per verifier row.
     * @param num_experts Width of the descriptor table, including shared.
     */
    void publishTerminalExpertRoute(
        std::vector<float> &values,
        int rows,
        int top_k,
        int num_experts)
    {
        if (rows <= 0 || top_k <= 0 || num_experts <= 1 ||
            values.size() != static_cast<size_t>(rows * top_k))
        {
            throw std::invalid_argument(
                "terminal expert route requires a non-empty row-major route matrix");
        }

        const float terminal_expert = static_cast<float>(num_experts - 1);
        for (int row = 0; row < rows; ++row)
        {
            values[static_cast<size_t>(row * top_k + top_k - 1)] =
                terminal_expert;
        }
    }

    std::vector<float> makeRoutingWeights(int rows, int top_k)
    {
        std::vector<float> values(static_cast<size_t>(rows) * top_k);
        for (int row = 0; row < rows; ++row)
        {
            float sum = 0.0f;
            for (int k = 0; k < top_k; ++k)
            {
                const float weight = 0.05f + 0.01f * static_cast<float>((row + k) % top_k);
                values[static_cast<size_t>(row) * top_k + k] = weight;
                sum += weight;
            }
            for (int k = 0; k < top_k; ++k)
                values[static_cast<size_t>(row) * top_k + k] /= sum;
        }
        return values;
    }

    /**
     * @brief Return sorted unique routed expert IDs present in a route table.
     *
     * The performance target keeps production-sized descriptor tables, but only
     * the routed IDs present in the synthetic verifier rows need distinct backing
     * weights.  This keeps setup proportional to the hot path we measure while
     * preserving hard failures for any active descriptor the kernel consumes.
     */
    std::vector<int> uniqueExpertIdsFromRoutes(
        const std::vector<float> &routing_indices,
        int num_experts)
    {
        std::vector<int> ids;
        ids.reserve(routing_indices.size());
        for (float value : routing_indices)
        {
            const int id = static_cast<int>(value);
            EXPECT_GE(id, 0);
            EXPECT_LT(id, num_experts);
            if (id >= 0 && id < num_experts)
                ids.push_back(id);
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        if (ids.empty())
            ids.push_back(0);
        return ids;
    }

    /**
     * @brief Normalize an explicit materialization list for a descriptor table.
     *
     * Active verifier routes must be represented by real prepared weights.  Slots
     * outside the active set are filled with aliases to a valid descriptor after
     * materialization; they are intentionally not used by the test's routes.
     */
    std::vector<int> sanitizeMaterializedExperts(
        std::vector<int> ids,
        int num_experts)
    {
        ids.erase(
            std::remove_if(
                ids.begin(), ids.end(),
                [num_experts](int id)
                {
                    EXPECT_GE(id, 0);
                    EXPECT_LT(id, num_experts);
                    return id < 0 || id >= num_experts;
                }),
            ids.end());
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        if (ids.empty())
            ids.push_back(0);
        return ids;
    }

    /**
     * @brief KL(reference || actual) after stable row-wise softmax.
     *
     * The verifier prefill speedometer is a performance test, but it also acts
     * as a numerical tripwire.  Aggregate cosine can miss row-local rank drift,
     * so we include a softmax-space check over each hidden row.
     */
    double rowSoftmaxKLDivergence(const float *actual, const float *expected, size_t row_width)
    {
        double max_actual = -std::numeric_limits<double>::infinity();
        double max_expected = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < row_width; ++i)
        {
            max_actual = std::max(max_actual, static_cast<double>(actual[i]));
            max_expected = std::max(max_expected, static_cast<double>(expected[i]));
        }

        double sum_actual = 0.0;
        double sum_expected = 0.0;
        for (size_t i = 0; i < row_width; ++i)
        {
            sum_actual += std::exp(static_cast<double>(actual[i]) - max_actual);
            sum_expected += std::exp(static_cast<double>(expected[i]) - max_expected);
        }

        constexpr double kEps = 1.0e-30;
        double kl = 0.0;
        for (size_t i = 0; i < row_width; ++i)
        {
            const double p = std::exp(static_cast<double>(expected[i]) - max_expected) /
                             std::max(sum_expected, kEps);
            const double q = std::exp(static_cast<double>(actual[i]) - max_actual) /
                             std::max(sum_actual, kEps);
            kl += p * (std::log(std::max(p, kEps)) - std::log(std::max(q, kEps)));
        }
        return kl;
    }

    CloseMetrics compareVectors(
        const std::vector<float> &actual,
        const std::vector<float> &expected,
        size_t row_width)
    {
        EXPECT_EQ(actual.size(), expected.size());
        CloseMetrics metrics;
        double dot = 0.0;
        double norm_actual = 0.0;
        double norm_expected = 0.0;
        double diff2 = 0.0;
        for (size_t i = 0; i < actual.size(); ++i)
        {
            const uint32_t actual_bits = std::bit_cast<uint32_t>(actual[i]);
            const uint32_t expected_bits = std::bit_cast<uint32_t>(expected[i]);
            if (actual_bits != expected_bits)
            {
                if (metrics.bit_mismatch_count == 0)
                {
                    metrics.first_bit_mismatch_index = i;
                    metrics.first_actual_bits = actual_bits;
                    metrics.first_expected_bits = expected_bits;
                }
                ++metrics.bit_mismatch_count;
            }

            const bool actual_finite = std::isfinite(actual[i]);
            const bool expected_finite = std::isfinite(expected[i]);
            if (!actual_finite || !expected_finite)
            {
                if (metrics.nonfinite_count == 0)
                    metrics.first_nonfinite_index = i;
                ++metrics.nonfinite_count;
                if (!actual_finite)
                    ++metrics.nonfinite_actual_count;
                if (!expected_finite)
                    ++metrics.nonfinite_expected_count;
                continue;
            }
            const double a = actual[i];
            const double e = expected[i];
            const double diff = a - e;
            dot += a * e;
            norm_actual += a * a;
            norm_expected += e * e;
            diff2 += diff * diff;
            metrics.max_abs = std::max(metrics.max_abs, std::abs(diff));
        }
        metrics.cosine = (norm_actual < 1.0e-30 && norm_expected < 1.0e-30)
                             ? 1.0
                             : dot / (std::sqrt(norm_actual) * std::sqrt(norm_expected) + 1.0e-30);
        metrics.relative_l2 = (norm_expected < 1.0e-30)
                                  ? ((diff2 < 1.0e-30)
                                         ? 0.0
                                         : std::numeric_limits<double>::infinity())
                                  : std::sqrt(diff2) / std::sqrt(norm_expected);
        if (row_width != 0 && actual.size() % row_width == 0)
        {
            const size_t rows = actual.size() / row_width;
            for (size_t row = 0; row < rows; ++row)
            {
                const float *row_actual = actual.data() + row * row_width;
                const float *row_expected = expected.data() + row * row_width;
                double row_dot = 0.0;
                double row_norm_actual = 0.0;
                double row_norm_expected = 0.0;
                double row_diff2 = 0.0;
                for (size_t i = 0; i < row_width; ++i)
                {
                    if (!std::isfinite(row_actual[i]) || !std::isfinite(row_expected[i]))
                    {
                        row_norm_expected = std::numeric_limits<double>::infinity();
                        row_diff2 = std::numeric_limits<double>::infinity();
                        break;
                    }
                    const double a = row_actual[i];
                    const double e = row_expected[i];
                    const double diff = a - e;
                    row_dot += a * e;
                    row_norm_actual += a * a;
                    row_norm_expected += e * e;
                    row_diff2 += diff * diff;
                }
                const double row_cosine =
                    (row_norm_actual < 1.0e-30 && row_norm_expected < 1.0e-30)
                        ? 1.0
                        : row_dot / (std::sqrt(row_norm_actual) * std::sqrt(row_norm_expected) + 1.0e-30);
                const double row_relative_l2 =
                    (row_norm_expected < 1.0e-30)
                        ? ((row_diff2 < 1.0e-30)
                               ? 0.0
                               : std::numeric_limits<double>::infinity())
                        : std::sqrt(row_diff2) / std::sqrt(row_norm_expected);
                const double row_kl =
                    (std::isfinite(row_norm_actual) && std::isfinite(row_norm_expected))
                        ? rowSoftmaxKLDivergence(row_actual, row_expected, row_width)
                        : std::numeric_limits<double>::infinity();
                if (row_cosine < metrics.min_row_cosine ||
                    row_relative_l2 > metrics.max_row_relative_l2 ||
                    row_kl > metrics.max_row_kl)
                {
                    metrics.worst_row = row;
                }
                metrics.min_row_cosine = std::min(metrics.min_row_cosine, row_cosine);
                metrics.max_row_relative_l2 =
                    std::max(metrics.max_row_relative_l2, row_relative_l2);
                metrics.max_row_kl = std::max(metrics.max_row_kl, row_kl);
            }
        }
        return metrics;
    }

    /**
     * @brief Require exact grouped-versus-serial FP32 output identity.
     *
     * Similarity metrics remain in the diagnostic payload because they make a
     * failure's magnitude immediately visible. They are not acceptance
     * thresholds: one differing output bit is a verifier correctness failure.
     */
    void expectBitwiseEqual(const CloseMetrics &metrics)
    {
        EXPECT_EQ(metrics.bit_mismatch_count, 0u)
            << "first_bit_mismatch_index=" << metrics.first_bit_mismatch_index
            << " actual_bits=0x" << std::hex << metrics.first_actual_bits
            << " expected_bits=0x" << metrics.first_expected_bits << std::dec
            << " cosine=" << metrics.cosine
            << " relative_l2=" << metrics.relative_l2
            << " max_abs=" << metrics.max_abs;
        EXPECT_EQ(metrics.nonfinite_count, 0u)
            << "first_nonfinite_index=" << metrics.first_nonfinite_index
            << " nonfinite_actual=" << metrics.nonfinite_actual_count
            << " nonfinite_expected=" << metrics.nonfinite_expected_count
            << " cosine=" << metrics.cosine << " relative_l2=" << metrics.relative_l2
            << " max_abs=" << metrics.max_abs;
        EXPECT_GE(metrics.cosine, 0.9999)
            << "relative_l2=" << metrics.relative_l2 << " max_abs=" << metrics.max_abs
            << " min_row_cosine=" << metrics.min_row_cosine
            << " max_row_relative_l2=" << metrics.max_row_relative_l2
            << " max_row_kl=" << metrics.max_row_kl
            << " worst_row=" << metrics.worst_row;
        EXPECT_LE(metrics.relative_l2, 0.006)
            << "cosine=" << metrics.cosine << " max_abs=" << metrics.max_abs
            << " min_row_cosine=" << metrics.min_row_cosine
            << " max_row_relative_l2=" << metrics.max_row_relative_l2
            << " max_row_kl=" << metrics.max_row_kl
            << " worst_row=" << metrics.worst_row;
        EXPECT_GE(metrics.min_row_cosine, 0.9998)
            << "cosine=" << metrics.cosine << " relative_l2=" << metrics.relative_l2
            << " max_row_relative_l2=" << metrics.max_row_relative_l2
            << " max_row_kl=" << metrics.max_row_kl
            << " worst_row=" << metrics.worst_row;
        EXPECT_LE(metrics.max_row_relative_l2, 0.008)
            << "cosine=" << metrics.cosine << " relative_l2=" << metrics.relative_l2
            << " min_row_cosine=" << metrics.min_row_cosine
            << " max_row_kl=" << metrics.max_row_kl
            << " worst_row=" << metrics.worst_row;
        EXPECT_LE(metrics.max_row_kl, 1.0e-4)
            << "cosine=" << metrics.cosine << " relative_l2=" << metrics.relative_l2
            << " min_row_cosine=" << metrics.min_row_cosine
            << " max_row_relative_l2=" << metrics.max_row_relative_l2
            << " worst_row=" << metrics.worst_row;
        EXPECT_LE(metrics.max_abs, 5.0)
            << "cosine=" << metrics.cosine << " relative_l2=" << metrics.relative_l2
            << " min_row_cosine=" << metrics.min_row_cosine
            << " max_row_relative_l2=" << metrics.max_row_relative_l2
            << " max_row_kl=" << metrics.max_row_kl
            << " worst_row=" << metrics.worst_row;
    }

    void printResult(const BenchResult &result)
    {
        static bool printed_header = false;
        if (!printed_header)
        {
            std::cout
                << "backend,case,m,top_k,num_experts,d_model,intermediate,"
                   "eager_ms,prepare_ms,pipeline_ms,graph_ms,rowwise_ms,speedup_vs_reference,"
                   "cosine,relative_l2,max_abs,"
                   "min_row_cosine,max_row_relative_l2,max_row_kl,"
                   "nonfinite_count,nonfinite_actual_count,nonfinite_expected_count,"
                   "first_nonfinite_index,bit_mismatch_count,"
                   "first_bit_mismatch_index,first_actual_bits,first_expected_bits,"
                   "worst_row\n";
            printed_header = true;
        }

        const double speedup =
            result.graph_ms > 0.0 ? (result.rowwise_ms / result.graph_ms) : 0.0;
        std::cout << std::fixed << std::setprecision(4)
                  << result.backend << ','
                  << result.case_name << ','
                  << result.m << ','
                  << result.top_k << ','
                  << result.num_experts << ','
                  << result.d_model << ','
                  << result.intermediate << ','
                  << result.eager_ms << ','
                  << result.prepare_ms << ','
                  << result.pipeline_ms << ','
                  << result.graph_ms << ','
                  << result.rowwise_ms << ','
                  << speedup << ','
                  << std::setprecision(8) << result.metrics.cosine << ','
                  << result.metrics.relative_l2 << ','
                  << result.metrics.max_abs << ','
                  << result.metrics.min_row_cosine << ','
                  << result.metrics.max_row_relative_l2 << ','
                  << result.metrics.max_row_kl << ','
                  << result.metrics.nonfinite_count << ','
                  << result.metrics.nonfinite_actual_count << ','
                  << result.metrics.nonfinite_expected_count << ','
                  << result.metrics.first_nonfinite_index << ','
                  << result.metrics.bit_mismatch_count << ','
                  << result.metrics.first_bit_mismatch_index << ','
                  << result.metrics.first_actual_bits << ','
                  << result.metrics.first_expected_bits << ','
                  << result.metrics.worst_row << '\n';
    }

    /**
     * @brief Assert that the graph-captured grouped verifier is economical.
     *
     * The correctness oracle is intentionally the expensive path: row-wise
     * decode for routed/shared rows, or split routed+shared verifier prefill
     * for the production combined-shared case.  Phase 9.8 requires the
     * promoted grouped verifier to be decode-equivalent and faster than that
     * reference, otherwise a "green" correctness test can silently preserve a
     * serial verifier cost in the hot path.
     */
    void expectGraphReplayFasterThanReference(const BenchResult &result)
    {
        ASSERT_GT(result.rowwise_ms, 0.0)
            << result.backend << ' ' << result.case_name << " M=" << result.m;
        ASSERT_GT(result.graph_ms, 0.0)
            << result.backend << ' ' << result.case_name << " M=" << result.m;
        EXPECT_LT(result.graph_ms, result.rowwise_ms)
            << result.backend << ' ' << result.case_name << " M=" << result.m
            << " graph_ms=" << result.graph_ms
            << " reference_ms=" << result.rowwise_ms
            << " speedup=" << (result.rowwise_ms / result.graph_ms);
    }

    /**
     * @brief Shared expert FFN promotion gate for verifier rows.
     *
     * Keep the shared expert path independently measurable before it is fused
     * with routed expert work.  A future CUDA shared-expert FFN kernel must be
     * decode-equivalent under the strict metrics above and materially faster
     * than serial row replay before the graph builder can promote it.
     */
    void expectSharedExpertFfnEconomical(const BenchResult &result)
    {
        expectGraphReplayFasterThanReference(result);
        ASSERT_GT(result.graph_ms, 0.0)
            << result.backend << " shared expert FFN M=" << result.m;
        const double speedup = result.rowwise_ms / result.graph_ms;
        EXPECT_GE(speedup, 2.0)
            << result.backend << " shared expert FFN M=" << result.m
            << " graph_ms=" << result.graph_ms
            << " reference_ms=" << result.rowwise_ms;
    }

    /**
     * @brief Owns prepared expert descriptors and their backing GPU weights.
     *
     * The production grouped MoE kernels consume descriptor tables, but those
     * descriptors point into VRAM owned by prepared GEMM objects. This small
     * owner keeps the generated weight tensors, load orchestrators, and stores
     * alive for the whole benchmark case.
     */
    struct PreparedExpertTables
    {
        std::vector<std::unique_ptr<llaminar2::TensorBase>> weights;
        std::vector<llaminar2::test::GpuPreparedGemm> prepared;
        std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> gate_descs;
        std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> up_descs;
        std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> down_descs;
        int gateup_table_id = -1;
        int down_table_id = -1;
    };

    PreparedExpertTables prepareExpertTables(
        llaminar2::IMoEKernel *moe,
        llaminar2::DeviceId device,
        int num_experts,
        int d_model,
        int intermediate,
        const std::string &backend_name,
        uint64_t model_context_base,
        std::vector<int> materialized_experts,
        const llaminar2::test::QuantizedVerifierFormatCase &gateup_format,
        const llaminar2::test::QuantizedVerifierFormatCase &down_format)
    {
        PreparedExpertTables tables;
        materialized_experts =
            sanitizeMaterializedExperts(std::move(materialized_experts), num_experts);
        tables.weights.reserve(materialized_experts.size() * 3);
        tables.prepared.reserve(materialized_experts.size() * 3);
        tables.gate_descs.resize(num_experts);
        tables.up_descs.resize(num_experts);
        tables.down_descs.resize(num_experts);
        std::vector<bool> has_desc(static_cast<size_t>(num_experts), false);

        auto add_desc = [&](
            int rows,
            int cols,
            int seed,
            const char *role,
            const llaminar2::test::QuantizedVerifierFormatCase &format)
        {
            std::unique_ptr<llaminar2::TensorBase> weight = format.create(
                {static_cast<size_t>(rows), static_cast<size_t>(cols)},
                static_cast<unsigned>(seed));

            auto *weight_ptr = weight.get();
            tables.weights.push_back(std::move(weight));
            tables.prepared.push_back(llaminar2::test::makeGpuPreparedGemm(
                weight_ptr,
                device,
                "perf.moe_verifier." + backend_name + "." + role + "." + std::to_string(seed),
                llaminar2::ModelContextId{model_context_base + static_cast<uint64_t>(seed)}));

            llaminar2::DeviceNativeVNNIMatrixDesc desc{};
            EXPECT_TRUE(tables.prepared.back().kernel->exportNativeVNNIMatrixDesc(desc))
                << "failed to export native descriptor role=" << role;
            EXPECT_EQ(desc.n, rows);
            EXPECT_EQ(desc.k, cols);
            EXPECT_EQ(desc.codebook_id, format.device_execution_codebook_id);
            return desc;
        };

        for (int expert : materialized_experts)
        {
            tables.gate_descs[static_cast<size_t>(expert)] =
                add_desc(intermediate, d_model, 4100 + expert, "gate", gateup_format);
            tables.up_descs[static_cast<size_t>(expert)] =
                add_desc(intermediate, d_model, 4200 + expert, "up", gateup_format);
            tables.down_descs[static_cast<size_t>(expert)] =
                add_desc(d_model, intermediate, 4300 + expert, "down", down_format);
            has_desc[static_cast<size_t>(expert)] = true;
        }

        const int alias = materialized_experts.front();
        for (int expert = 0; expert < num_experts; ++expert)
        {
            if (has_desc[static_cast<size_t>(expert)])
                continue;
            tables.gate_descs[static_cast<size_t>(expert)] =
                tables.gate_descs[static_cast<size_t>(alias)];
            tables.up_descs[static_cast<size_t>(expert)] =
                tables.up_descs[static_cast<size_t>(alias)];
            tables.down_descs[static_cast<size_t>(expert)] =
                tables.down_descs[static_cast<size_t>(alias)];
        }

        tables.gateup_table_id = moe->uploadGroupedExpertGateUpDescriptorTables(
            tables.gate_descs.data(), tables.up_descs.data(), num_experts, d_model, intermediate);
        EXPECT_GE(tables.gateup_table_id, 0);
        tables.down_table_id = moe->uploadGroupedExpertDownDescriptorTable(
            tables.down_descs.data(), num_experts, d_model, intermediate);
        EXPECT_GE(tables.down_table_id, 0);
        return tables;
    }

}

#ifdef HAVE_CUDA
namespace
{
    /** Static compiler and occupancy evidence for one exact CUDA kernel. */
    struct CudaMoEPrefillKernelResources
    {
        int registers_per_thread = 0;
        size_t local_memory_bytes_per_thread = 0;
        size_t static_shared_memory_bytes = 0;
        int max_threads_per_block = 0;
        int max_active_blocks_per_sm = 0;

        /** @return true when ptxas emitted no local-memory scratch. */
        [[nodiscard]] bool spillFree() const noexcept
        {
            return local_memory_bytes_per_thread == 0;
        }
    };

    /**
     * @brief Inspect one exact CUDA grouped-prefill production component.
     *
     * A failed query is fatal to the harness. Timing an uninspected kernel
     * would violate the corpus contract because a spilling specialization
     * could appear to win before profiler evidence later disqualified it.
     */
    CudaMoEPrefillKernelResources queryCudaMoEPrefillKernelResources(
        uint8_t execution_codebook,
        int component,
        int block_threads)
    {
        CudaMoEPrefillKernelResources resources{};
        if (!cudaMoE_grouped_prefill_query_kernel_resources(
                execution_codebook,
                component,
                block_threads,
                &resources.registers_per_thread,
                &resources.local_memory_bytes_per_thread,
                &resources.static_shared_memory_bytes,
                &resources.max_threads_per_block,
                &resources.max_active_blocks_per_sm))
        {
            throw std::runtime_error(
                "failed to inspect compiled CUDA MoE grouped-prefill kernel");
        }
        return resources;
    }

    /** Inspect one exact persistent grouped-IMMA specialization. */
    CudaMoEPrefillKernelResources queryCudaMoEGroupedImmaKernelResources(
        uint8_t execution_codebook,
        llaminar2::cuda::moe::GroupedImmaColumns columns)
    {
        CudaMoEPrefillKernelResources resources{};
        if (!cudaMoEGroupedImma_queryKernelResources(
                execution_codebook,
                columns,
                &resources.registers_per_thread,
                &resources.local_memory_bytes_per_thread,
                &resources.static_shared_memory_bytes,
                &resources.max_threads_per_block,
                &resources.max_active_blocks_per_sm))
        {
            throw std::runtime_error(
                "failed to inspect compiled CUDA grouped-IMMA kernel");
        }
        return resources;
    }

    /** Inspect one fused grouped-IMMA gate/up/SwiGLU specialization. */
    CudaMoEPrefillKernelResources queryCudaMoEGroupedImmaGateUpKernelResources(
        uint8_t execution_codebook,
        llaminar2::cuda::moe::GroupedImmaColumns columns,
        llaminar2::cuda::moe::GroupedImmaGateUpSchedule schedule)
    {
        CudaMoEPrefillKernelResources resources{};
        if (!cudaMoEGroupedImma_queryGateUpKernelResources(
                execution_codebook,
                columns,
                schedule,
                &resources.registers_per_thread,
                &resources.local_memory_bytes_per_thread,
                &resources.static_shared_memory_bytes,
                &resources.max_threads_per_block,
                &resources.max_active_blocks_per_sm))
        {
            throw std::runtime_error(
                "failed to inspect compiled CUDA grouped-IMMA gate/up kernel");
        }
        return resources;
    }

    bool hasCudaDevice()
    {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    }

    /**
     * @brief Invoke one CUDA driver profiler-control entry point by symbol.
     *
     * CUDA 13 exports `cuProfilerStart` and `cuProfilerStop` from the driver,
     * while the minimal development image does not install the historical
     * runtime profiler header. Resolving the stable driver ABI keeps profiler
     * controls inside this performance-only harness and prevents production
     * code from acquiring a dependency on Nsight tooling.
     *
     * @param symbol Exact CUDA driver symbol to invoke.
     * @return true only when the symbol exists and reports success.
     */
    bool invokeCudaMoEProfilerControl(const char *symbol)
    {
        if (!symbol || *symbol == '\0')
            return false;
        using ControlFunction = int (*)();
        static void *driver = []
        {
            return ::dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        }();
        if (!driver)
            return false;
        ::dlerror();
        void *raw = ::dlsym(driver, symbol);
        if (!raw || ::dlerror() != nullptr)
            return false;
        return reinterpret_cast<ControlFunction>(raw)() == 0;
    }

    class ScopedCudaMoEPrefillConfig
    {
    public:
        ScopedCudaMoEPrefillConfig()
            : old_tile_m_(llaminar2::mutableDebugEnv().gemm.cuda_moe_prefill_tile_m),
              old_fuse_swiglu_(llaminar2::mutableDebugEnv().gemm.cuda_moe_prefill_fuse_swiglu)
        {
        }

        ~ScopedCudaMoEPrefillConfig()
        {
            auto &gemm = llaminar2::mutableDebugEnv().gemm;
            gemm.cuda_moe_prefill_tile_m = old_tile_m_;
            gemm.cuda_moe_prefill_fuse_swiglu = old_fuse_swiglu_;
        }

        void set(int tile_m, bool fuse_swiglu)
        {
            auto &gemm = llaminar2::mutableDebugEnv().gemm;
            gemm.cuda_moe_prefill_tile_m = tile_m;
            gemm.cuda_moe_prefill_fuse_swiglu = fuse_swiglu;
        }

    private:
        int old_tile_m_ = 0;
        bool old_fuse_swiglu_ = true;
    };

    /** Restore capture-time CUDA MoE launch geometry after one trainer cell. */
    class ScopedCudaMoEGeometryConfig
    {
    public:
        ScopedCudaMoEGeometryConfig()
            : old_imma_gateup_columns_(
                  llaminar2::mutableDebugEnv().gemm.cuda_moe_imma_gateup_columns),
              old_imma_down_columns_(
                  llaminar2::mutableDebugEnv().gemm.cuda_moe_imma_down_columns),
              old_imma_gateup_schedule_(
                  llaminar2::mutableDebugEnv().gemm
                      .cuda_moe_imma_gateup_schedule),
              old_imma_override_active_(
                  llaminar2::mutableDebugEnv().gemm
                      .cuda_moe_imma_geometry_override_active)
        {
        }

        ~ScopedCudaMoEGeometryConfig()
        {
            auto &gemm = llaminar2::mutableDebugEnv().gemm;
            gemm.cuda_moe_imma_gateup_columns = old_imma_gateup_columns_;
            gemm.cuda_moe_imma_down_columns = old_imma_down_columns_;
            gemm.cuda_moe_imma_gateup_schedule =
                old_imma_gateup_schedule_;
            gemm.cuda_moe_imma_geometry_override_active =
                old_imma_override_active_;
        }

        /** Force one pair of compiled grouped-IMMA CTA widths before capture. */
        void setGroupedImmaGeometry(
            int gateup_columns,
            int down_columns,
            llaminar2::cuda::moe::GroupedImmaGateUpSchedule schedule)
        {
            auto &gemm = llaminar2::mutableDebugEnv().gemm;
            gemm.cuda_moe_imma_gateup_columns = gateup_columns;
            gemm.cuda_moe_imma_down_columns = down_columns;
            gemm.cuda_moe_imma_gateup_schedule =
                static_cast<int>(schedule);
            gemm.cuda_moe_imma_geometry_override_active = true;
        }

    private:
        int old_imma_gateup_columns_ = 32;
        int old_imma_down_columns_ = 32;
        int old_imma_gateup_schedule_ = 0;
        bool old_imma_override_active_ = false;
    };

    /** One arithmetic-neutral CUDA production launch candidate. */
    struct CudaMoEProductionCandidate
    {
        int gateup_columns = 32;
        int down_columns = 32;
        llaminar2::cuda::moe::GroupedImmaGateUpSchedule schedule =
            llaminar2::cuda::moe::GroupedImmaGateUpSchedule::
                ParallelProjections;

        /** @brief Return a stable corpus identity for this launch policy. */
        std::string id() const
        {
            const char *schedule_name =
                schedule == llaminar2::cuda::moe::
                                GroupedImmaGateUpSchedule::PairedProjections
                    ? "paired"
                    : "parallel";
            return std::string("imma_") + schedule_name + "_g" +
                   std::to_string(gateup_columns) + "__d" +
                   std::to_string(down_columns);
        }
    };

    /** @brief Enumerate the complete non-dominated CUDA top-8 candidate space. */
    std::vector<CudaMoEProductionCandidate> cudaMoEProductionCandidates()
    {
        constexpr std::array<int, 3> gateup_columns{32, 64, 128};
        constexpr std::array<int, 4> down_columns{32, 64, 128, 256};
        constexpr std::array<
            llaminar2::cuda::moe::GroupedImmaGateUpSchedule,
            2>
            schedules{
                llaminar2::cuda::moe::GroupedImmaGateUpSchedule::
                    ParallelProjections,
                llaminar2::cuda::moe::GroupedImmaGateUpSchedule::
                    PairedProjections,
            };
        std::vector<CudaMoEProductionCandidate> result;
        result.reserve(
            schedules.size() * gateup_columns.size() * down_columns.size());
        for (const auto schedule : schedules)
        {
            for (const int gate_width : gateup_columns)
            {
                for (const int down_width : down_columns)
                {
                    result.push_back(CudaMoEProductionCandidate{
                        .gateup_columns = gate_width,
                        .down_columns = down_width,
                        .schedule = schedule,
                    });
                }
            }
        }
        return result;
    }

    /**
     * @brief Resolve one exact CUDA candidate identity for isolated profiling.
     *
     * Profiler evidence is meaningful only when the requested corpus identity
     * maps to exactly one launch geometry. Unknown or partial spellings are
     * rejected instead of silently selecting the generic production geometry.
     *
     * @param candidate_id Stable candidate identity from the timing corpus.
     * @return Exact candidate when the identity is in the launch registry.
     */
    std::optional<CudaMoEProductionCandidate> findCudaMoEProductionCandidate(
        const std::string &candidate_id)
    {
        for (const auto &candidate : cudaMoEProductionCandidates())
        {
            if (candidate.id() == candidate_id)
                return candidate;
        }
        return std::nullopt;
    }

    /** Stable two-stage timing policy for one CUDA production sweep cell. */
    struct CudaMoEProductionSweepSettings
    {
        int screening_warmups = 1;
        int screening_trials = 3;
        int screening_replays = 2;
        int robust_warmups = 2;
        int robust_trials = 15;
        int robust_replays = 4;
        int minimum_finalists = 4;
        int maximum_finalists = 8;
        double finalist_margin = 0.05;
        std::string profiler_request_id;
        std::optional<CudaMoEProductionCandidate> profiler_candidate;
        std::optional<CudaMoEProductionCandidate> proof_candidate;

        /** @return true when this process owns one isolated profiler launch. */
        [[nodiscard]] bool profiling() const noexcept
        {
            return !profiler_request_id.empty();
        }
    };

    /** Screening, resource, and correctness evidence for one CUDA geometry. */
    struct CudaMoEProductionEvidence
    {
        CudaMoEProductionCandidate candidate;
        CudaMoEPrefillKernelResources gateup_resources;
        CudaMoEPrefillKernelResources down_resources;
        std::vector<double> screening_samples_ms;
        std::vector<double> robust_samples_ms;
        int screening_replays_per_sample = 0;
        int robust_replays_per_sample = 0;
        uint64_t bit_mismatches = 0;
        uint64_t first_bit_mismatch = std::numeric_limits<uint64_t>::max();
        bool route_counter_ok = false;
        bool finalist = false;
        bool winner = false;

        /** @return Robust samples when available, otherwise screening samples. */
        const std::vector<double> &selectedSamplesMs() const
        {
            return robust_samples_ms.empty()
                       ? screening_samples_ms
                       : robust_samples_ms;
        }

        /** @return Replay cardinality associated with `selectedSamplesMs()`. */
        int selectedReplaysPerSample() const
        {
            return robust_samples_ms.empty()
                       ? screening_replays_per_sample
                       : robust_replays_per_sample;
        }

        /** @return Selected median latency in microseconds. */
        double medianUs() const
        {
            const auto &samples = selectedSamplesMs();
            return samples.empty()
                       ? std::numeric_limits<double>::infinity()
                       : samples[samples.size() / 2] * 1000.0;
        }
    };

    /** Own one non-default CUDA stream for a complete production sweep cell. */
    class ScopedCudaMoEPrefillStream
    {
    public:
        ScopedCudaMoEPrefillStream()
        {
            if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) !=
                cudaSuccess)
            {
                throw std::runtime_error(
                    "failed to create CUDA MoE production sweep stream");
            }
        }

        ScopedCudaMoEPrefillStream(const ScopedCudaMoEPrefillStream &) = delete;
        ScopedCudaMoEPrefillStream &operator=(
            const ScopedCudaMoEPrefillStream &) = delete;

        ~ScopedCudaMoEPrefillStream()
        {
            if (stream_)
                (void)cudaStreamDestroy(stream_);
        }

        /** @return Exact non-default CUDA stream owned by this cell. */
        cudaStream_t get() const noexcept
        {
            return stream_;
        }

    private:
        cudaStream_t stream_ = nullptr;
    };

    /** Unbind graph workspace before its manager leaves scope. */
    class ScopedCudaMoEWorkspaceBinding
    {
    public:
        ScopedCudaMoEWorkspaceBinding(
            llaminar2::IWorkspaceConsumer *consumer,
            llaminar2::DeviceWorkspaceManager *workspace)
            : consumer_(consumer)
        {
            if (!consumer_ || !workspace)
            {
                throw std::invalid_argument(
                    "CUDA MoE production sweep requires a workspace binding");
            }
            consumer_->bindWorkspace(workspace);
        }

        ScopedCudaMoEWorkspaceBinding(
            const ScopedCudaMoEWorkspaceBinding &) = delete;
        ScopedCudaMoEWorkspaceBinding &operator=(
            const ScopedCudaMoEWorkspaceBinding &) = delete;

        ~ScopedCudaMoEWorkspaceBinding()
        {
            consumer_->unbindWorkspace();
        }

    private:
        llaminar2::IWorkspaceConsumer *consumer_ = nullptr;
    };

    /** Close an optional C stream owned by a trainer test scope. */
    struct CudaMoEFileCloser
    {
        void operator()(std::FILE *file) const noexcept
        {
            if (file)
                (void)std::fclose(file);
        }
    };

    /** Persistent CUDA counters for allocation-free byte certificates. */
    class CudaMoEPrefillDeviceByteCertificate
    {
    public:
        CudaMoEPrefillDeviceByteCertificate()
        {
            if (cudaMalloc(&mismatch_count_, sizeof(uint64_t)) != cudaSuccess)
            {
                throw std::runtime_error(
                    "failed to allocate persistent CUDA mismatch counter");
            }
            if (cudaMalloc(&first_mismatch_, sizeof(uint64_t)) != cudaSuccess)
            {
                (void)cudaFree(mismatch_count_);
                mismatch_count_ = nullptr;
                throw std::runtime_error(
                    "failed to allocate persistent CUDA first-mismatch counter");
            }
        }

        CudaMoEPrefillDeviceByteCertificate(
            const CudaMoEPrefillDeviceByteCertificate &) = delete;
        CudaMoEPrefillDeviceByteCertificate &operator=(
            const CudaMoEPrefillDeviceByteCertificate &) = delete;

        ~CudaMoEPrefillDeviceByteCertificate()
        {
            if (first_mismatch_)
                (void)cudaFree(first_mismatch_);
            if (mismatch_count_)
                (void)cudaFree(mismatch_count_);
        }

        /** @brief Compare two device tensors and materialize two terminal words. */
        std::pair<uint64_t, uint64_t> compare(
            const float *actual,
            const float *expected,
            size_t count,
            cudaStream_t stream)
        {
            if (!llaminar2::test::enqueueCudaFP32ByteComparison(
                    actual,
                    expected,
                    count,
                    mismatch_count_,
                    first_mismatch_,
                    stream))
            {
                throw std::runtime_error(
                    "failed to enqueue CUDA MoE byte certificate");
            }
            uint64_t host_count = 0;
            uint64_t host_first = std::numeric_limits<uint64_t>::max();
            if (cudaMemcpyAsync(
                    &host_count,
                    mismatch_count_,
                    sizeof(host_count),
                    cudaMemcpyDeviceToHost,
                    stream) != cudaSuccess ||
                cudaMemcpyAsync(
                    &host_first,
                    first_mismatch_,
                    sizeof(host_first),
                    cudaMemcpyDeviceToHost,
                    stream) != cudaSuccess ||
                cudaStreamSynchronize(stream) != cudaSuccess)
            {
                throw std::runtime_error(
                    "failed to read CUDA MoE byte certificate");
            }
            return {host_count, host_first};
        }

    private:
        uint64_t *mismatch_count_ = nullptr;
        uint64_t *first_mismatch_ = nullptr;
    };

    /**
     * @brief Stop a timing cell immediately when its launch body is rejected.
     *
     * A non-fatal EXPECT inside an iteration loop turns one unsupported route
     * into hundreds of failures and then reports meaningless near-zero timing.
     * Throwing here lets GoogleTest report the first broken phase and prevents
     * failed work from being mistaken for an economical kernel.
     */
    void requireCudaBenchBody(bool ok, const char *phase)
    {
        if (!ok)
        {
            throw std::runtime_error(
                std::string("CUDA MoE verifier benchmark body failed during ") +
                (phase ? phase : "unknown phase"));
        }
    }

    /** Reusable CUDA event pair for allocation-free captured-graph timing. */
    class CudaMoEPrefillEventTimer
    {
    public:
        explicit CudaMoEPrefillEventTimer(cudaStream_t stream)
            : stream_(stream)
        {
            if (!stream_)
            {
                throw std::invalid_argument(
                    "CUDA MoE prefill timer requires a non-null stream");
            }
            if (cudaEventCreate(&start_) != cudaSuccess)
            {
                throw std::runtime_error(
                    "failed to create CUDA MoE prefill start event");
            }
            if (cudaEventCreate(&stop_) != cudaSuccess)
            {
                (void)cudaEventDestroy(start_);
                start_ = nullptr;
                throw std::runtime_error(
                    "failed to create CUDA MoE prefill stop event");
            }
        }

        CudaMoEPrefillEventTimer(const CudaMoEPrefillEventTimer &) = delete;
        CudaMoEPrefillEventTimer &operator=(
            const CudaMoEPrefillEventTimer &) = delete;

        ~CudaMoEPrefillEventTimer()
        {
            if (stop_)
                (void)cudaEventDestroy(stop_);
            if (start_)
                (void)cudaEventDestroy(start_);
        }

        /** @brief Time repeated graph replays and return milliseconds per replay. */
        double sample(int replays, const std::function<bool()> &launch)
        {
            if (replays <= 0 || !launch)
            {
                throw std::invalid_argument(
                    "CUDA MoE timer requires positive replay cardinality");
            }
            if (cudaEventRecord(start_, stream_) != cudaSuccess)
                throw std::runtime_error("failed to record CUDA timer start");
            for (int replay = 0; replay < replays; ++replay)
                requireCudaBenchBody(launch(), "CUDA candidate timing replay");
            if (cudaEventRecord(stop_, stream_) != cudaSuccess ||
                cudaEventSynchronize(stop_) != cudaSuccess)
            {
                throw std::runtime_error("failed to complete CUDA timing sample");
            }
            float elapsed_ms = 0.0f;
            if (cudaEventElapsedTime(&elapsed_ms, start_, stop_) != cudaSuccess)
                throw std::runtime_error("failed to read CUDA timing sample");
            return static_cast<double>(elapsed_ms) /
                   static_cast<double>(replays);
        }

    private:
        cudaStream_t stream_ = nullptr;
        cudaEvent_t start_ = nullptr;
        cudaEvent_t stop_ = nullptr;
    };

    /** @brief Prove PerfStats observed one exact forced CUDA candidate. */
    bool observedCudaMoEProductionCandidate(
        int rows,
        const CudaMoEProductionCandidate &candidate)
    {
        const char *expected_schedule =
            candidate.schedule == llaminar2::cuda::moe::
                                      GroupedImmaGateUpSchedule::
                                          PairedProjections
                ? "paired_projections"
                : "parallel_projections";
        const auto records = llaminar2::PerfStatsCollector::snapshot(
            {"kernel.cuda_moe_grouped_prefill_swiglu_path_calls"});
        for (const auto &record : records)
        {
            if (record.name != "cuda_moe_grouped_prefill_swiglu_path_calls")
                continue;
            const auto tag = [&record](const char *name) -> std::string
            {
                const auto found = record.tags.find(name);
                return found == record.tags.end()
                           ? std::string{}
                           : found->second;
            };
            if (record.count > 0 &&
                tag("seq_len") == std::to_string(rows) &&
                tag("gateup_geometry_contract") == "tensor_core_imma" &&
                tag("imma_gateup_columns") ==
                    std::to_string(candidate.gateup_columns) &&
                tag("imma_down_columns") ==
                    std::to_string(candidate.down_columns) &&
                tag("imma_gateup_schedule") == expected_schedule &&
                tag("down_publication") == "ordered_direct")
            {
                return true;
            }
        }
        return false;
    }

    /** Emit one authenticated aggregate row for a CUDA production candidate. */
    void writeCudaMoEProductionEvidence(
        std::FILE *csv,
        const llaminar2::test::native_vnni_dispatch::MoERoutedPrefillCase &routed_case,
        llaminar2::test::native_vnni_dispatch::MoERoutingProfile route_profile,
        const llaminar2::test::native_vnni_dispatch::MoERoutingProfileStats &route_stats,
        int rows,
        const CudaMoEProductionEvidence &evidence)
    {
        if (!csv)
            throw std::invalid_argument("CUDA production evidence requires a CSV");
        const auto &samples = evidence.selectedSamplesMs();
        const auto timing =
            llaminar2::test::trainer::summarizeSortedTimingSamples(samples);
        const auto &gate_format = llaminar2::test::quantizedMoEVerifierFormat(
            routed_case.routed.gate);
        const auto &up_format = llaminar2::test::quantizedMoEVerifierFormat(
            routed_case.routed.up);
        const auto &down_format = llaminar2::test::quantizedMoEVerifierFormat(
            routed_case.routed.down);
        std::ostringstream row;
        row << std::setprecision(17)
            << "cuda,moe_production_prefill,"
            << routed_case.evidenceId() << ','
            << llaminar2::test::native_vnni_dispatch::moeRoutingProfileName(
                   route_profile) << ','
            << routed_case.routed.gate << ','
            << routed_case.routed.up << ','
            << routed_case.routed.down << ','
            << static_cast<unsigned>(gate_format.device_execution_codebook_id) << ','
            << static_cast<unsigned>(up_format.device_execution_codebook_id) << ','
            << static_cast<unsigned>(down_format.device_execution_codebook_id) << ','
            << routed_case.hidden_size << ','
            << routed_case.routed_expert_width << ','
            << routed_case.expert_count << ','
            << routed_case.experts_per_token << ','
            << route_stats.active_experts << ','
            << route_stats.maximum_assignments << ','
            << route_stats.assignment_cv << ','
            << rows << ','
            << evidence.candidate.id() << ','
            << llaminar2::cuda::moe::kGroupedImmaTileRows << ','
            << evidence.candidate.gateup_columns << ','
            << 16 << ','
            << evidence.candidate.down_columns << ','
            << evidence.gateup_resources.local_memory_bytes_per_thread << ','
            << evidence.down_resources.local_memory_bytes_per_thread << ','
            << evidence.gateup_resources.registers_per_thread << ','
            << evidence.down_resources.registers_per_thread << ','
            << evidence.gateup_resources.static_shared_memory_bytes << ','
            << evidence.down_resources.static_shared_memory_bytes << ','
            << evidence.gateup_resources.max_active_blocks_per_sm << ','
            << evidence.down_resources.max_active_blocks_per_sm << ','
            << evidence.screening_samples_ms.size() << ','
            << evidence.screening_replays_per_sample << ','
            << evidence.robust_samples_ms.size() << ','
            << evidence.robust_replays_per_sample << ','
            << samples.size() << ','
            << evidence.selectedReplaysPerSample() << ','
            << timing.min * 1000.0 << ','
            << timing.median * 1000.0 << ','
            << timing.p95 * 1000.0 << ','
            << timing.mad * 1000.0 << ','
            << timing.cv << ','
            << timing.digest << ','
            << evidence.bit_mismatches << ','
            << evidence.first_bit_mismatch << ','
            << (evidence.route_counter_ok ? 1 : 0) << ','
            << (evidence.finalist ? 1 : 0) << ','
            << (evidence.winner ? 1 : 0);
        const std::string serialized = row.str();
        std::fprintf(csv, "%s\n", serialized.c_str());
        std::fflush(csv);
    }

    /** Retain every native CUDA event sample for later corpus authentication. */
    void writeCudaMoEProductionTimingEvidence(
        std::FILE *timing_csv,
        const llaminar2::test::native_vnni_dispatch::MoERoutedPrefillCase &routed_case,
        llaminar2::test::native_vnni_dispatch::MoERoutingProfile route_profile,
        int rows,
        const CudaMoEProductionEvidence &evidence)
    {
        if (!timing_csv)
            return;
        const auto write_phase = [&] (
            const char *phase,
            const std::vector<double> &samples_ms,
            int replays_per_sample)
        {
            for (size_t index = 0; index < samples_ms.size(); ++index)
            {
                std::fprintf(
                    timing_csv,
                    "cuda,moe_production_prefill,%s,%s,%d,%s,%s,%zu,%d,%.9f,%a\n",
                    routed_case.evidenceId().c_str(),
                    std::string(
                        llaminar2::test::native_vnni_dispatch::
                            moeRoutingProfileName(route_profile)).c_str(),
                    rows,
                    evidence.candidate.id().c_str(),
                    phase,
                    index,
                    replays_per_sample,
                    samples_ms[index] * 1000.0,
                    samples_ms[index]);
            }
        };
        write_phase(
            "screening",
            evidence.screening_samples_ms,
            evidence.screening_replays_per_sample);
        write_phase(
            "robust",
            evidence.robust_samples_ms,
            evidence.robust_replays_per_sample);
        std::fflush(timing_csv);
    }

    double timeCudaEvents(cudaStream_t stream, int iterations, const std::function<bool()> &body)
    {
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        EXPECT_EQ(cudaEventCreate(&start), cudaSuccess);
        EXPECT_EQ(cudaEventCreate(&stop), cudaSuccess);
        EXPECT_EQ(cudaEventRecord(start, stream), cudaSuccess);
        for (int i = 0; i < iterations; ++i)
            requireCudaBenchBody(body(), "timed replay");
        EXPECT_EQ(cudaEventRecord(stop, stream), cudaSuccess);
        EXPECT_EQ(cudaEventSynchronize(stop), cudaSuccess);
        float ms = 0.0f;
        EXPECT_EQ(cudaEventElapsedTime(&ms, start, stop), cudaSuccess);
        EXPECT_EQ(cudaEventDestroy(start), cudaSuccess);
        EXPECT_EQ(cudaEventDestroy(stop), cudaSuccess);
        return static_cast<double>(ms) / static_cast<double>(iterations);
    }

    std::vector<float> runCudaRowwiseDecode(
        llaminar2::IMoEKernel *moe,
        llaminar2::DeviceWorkspaceManager *workspace,
        cudaStream_t stream,
        const std::vector<float> &hidden_values,
        const std::vector<float> &routing_indices,
        const std::vector<float> &routing_weights,
        int rows,
        int top_k,
        int d_model,
        int intermediate,
        int gateup_table,
        int down_table,
        llaminar2::DeviceId device,
        double *avg_ms)
    {
        auto *workspace_consumer =
            dynamic_cast<llaminar2::IWorkspaceConsumer *>(moe);
        EXPECT_NE(workspace_consumer, nullptr);
        EXPECT_NE(workspace, nullptr);
        if (workspace_consumer && workspace)
        {
            /*
             * The CUDA MoE kernel is a process-wide singleton.  Rebinding here
             * makes the row-wise decode reference explicit about ownership of
             * the workspace-backed pointer-array tables instead of relying on
             * whichever grouped-prefill stage happened to run immediately
             * before the reference pass.
             */
            workspace_consumer->bindWorkspace(workspace);
        }

        std::vector<float> decoded;
        decoded.reserve(static_cast<size_t>(rows) * d_model);

        auto decode_once = [&]()
        {
            decoded.clear();
            for (int row = 0; row < rows; ++row)
            {
                const auto row_begin = hidden_values.begin() + static_cast<ptrdiff_t>(row) * d_model;
                std::vector<float> row_hidden_values(row_begin, row_begin + d_model);
                auto row_hidden = makeTensor({1, static_cast<size_t>(d_model)}, row_hidden_values);
                EXPECT_TRUE(row_hidden->ensureOnDevice(device, stream));

                std::vector<int> expert_ids(static_cast<size_t>(top_k));
                std::vector<float> expert_weights(static_cast<size_t>(top_k));
                for (int k = 0; k < top_k; ++k)
                {
                    const size_t slot = static_cast<size_t>(row) * top_k + k;
                    expert_ids[static_cast<size_t>(k)] = static_cast<int>(routing_indices[slot]);
                    expert_weights[static_cast<size_t>(k)] = routing_weights[slot];
                }

                std::vector<std::shared_ptr<llaminar2::FP32Tensor>> gate_owned;
                std::vector<std::shared_ptr<llaminar2::FP32Tensor>> up_owned;
                std::vector<llaminar2::ITensor *> gate_outputs(static_cast<size_t>(top_k));
                std::vector<llaminar2::ITensor *> up_outputs(static_cast<size_t>(top_k));
                gate_owned.reserve(top_k);
                up_owned.reserve(top_k);
                for (int k = 0; k < top_k; ++k)
                {
                    gate_owned.push_back(makeZeros({static_cast<size_t>(intermediate)}));
                    up_owned.push_back(makeZeros({static_cast<size_t>(intermediate)}));
                    EXPECT_TRUE(gate_owned.back()->ensureOnDevice(device, stream));
                    EXPECT_TRUE(up_owned.back()->ensureOnDevice(device, stream));
                    gate_outputs[static_cast<size_t>(k)] = gate_owned.back().get();
                    up_outputs[static_cast<size_t>(k)] = up_owned.back().get();
                }

                auto decode_output = makeZeros({static_cast<size_t>(d_model)});
                EXPECT_TRUE(decode_output->ensureOnDevice(device, stream));
                EXPECT_TRUE(moe->groupedExpertGateUpDecodeFromTable(
                    row_hidden.get(), expert_ids.data(), gateup_table, top_k,
                    gate_outputs.data(), up_outputs.data(), d_model, intermediate));
                EXPECT_TRUE(moe->groupedExpertDownDecodeFromTable(
                    gate_outputs.data(), up_outputs.data(), expert_ids.data(), expert_weights.data(),
                    down_table, top_k, decode_output.get(), d_model, intermediate));
                EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
                TransferEngine::publishDeviceWrite(decode_output, device, stream);
                decoded.insert(
                    decoded.end(),
                    decode_output->data(),
                    decode_output->data() + decode_output->numel());
            }
            return true;
        };

        const int timing_iters = std::max(1, envInt("LLAMINAR_MOE_VERIFIER_PREFILL_ROWWISE_ITERS", 3));
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < timing_iters; ++i)
            EXPECT_TRUE(decode_once());
        const auto stop = std::chrono::steady_clock::now();
        *avg_ms = std::chrono::duration<double, std::milli>(stop - start).count() /
                  static_cast<double>(timing_iters);
        return decoded;
    }

    /**
     * @brief Execute one real CUDA MoE production tournament cell.
     *
     * Every expert owns distinct prepared weights, grouping is published once,
     * and each non-dominated geometry captures the same production grouped
     * pipeline. Compiler resource evidence is queried before launch, every
     * candidate is byte-compared on device against the fixed-arithmetic
     * reference, and timing uses a persistent event pair outside graph capture.
     * Every M and routing profile proves that reference against serial row
     * decode before any timing evidence can be emitted.
     *
     * @param routed_case Pinned GGUF source formats and production geometry.
     * @param rows Exact prefill bucket represented by this cell.
     * @param route_profile Deterministic expert-load distribution under test.
     * @param device_ordinal CUDA device assigned by the external scheduler.
     * @param settings Screening, finalist, and serial-proof policy.
     * @param csv Optional aggregate evidence destination.
     * @param timing_csv Optional raw native-event sidecar destination.
     */
    void runCudaProductionMoERoutedCase(
        const llaminar2::test::native_vnni_dispatch::MoERoutedPrefillCase &routed_case,
        int rows,
        llaminar2::test::native_vnni_dispatch::MoERoutingProfile route_profile,
        int device_ordinal,
        const CudaMoEProductionSweepSettings &settings,
        std::FILE *csv,
        std::FILE *timing_csv)
    {
        if (settings.profiling() != settings.profiler_candidate.has_value())
        {
            throw std::invalid_argument(
                "CUDA profiler request and exact candidate must be supplied together");
        }
        if (settings.profiling() && settings.proof_candidate.has_value())
        {
            throw std::invalid_argument(
                "CUDA profiler and focused-proof candidates are mutually exclusive");
        }
        const std::vector<CudaMoEProductionCandidate> candidates =
            settings.profiling()
                ? std::vector<CudaMoEProductionCandidate>{
                      *settings.profiler_candidate}
                : settings.proof_candidate.has_value()
                      ? std::vector<CudaMoEProductionCandidate>{
                            *settings.proof_candidate}
                      : cudaMoEProductionCandidates();
        const int candidate_count = static_cast<int>(candidates.size());
        if (rows <= 8 || device_ordinal < 0)
        {
            throw std::invalid_argument(
                "CUDA production tournament requires M>8 and a valid device");
        }
        if (routed_case.routed.gate != routed_case.routed.up)
        {
            throw std::runtime_error(
                routed_case.evidenceId() +
                " has distinct gate/up formats, but CUDA production fuses them");
        }
        if (settings.screening_warmups < 0 || settings.screening_trials <= 0 ||
            settings.screening_replays <= 0 || settings.robust_warmups < 0 ||
            settings.robust_trials <= 0 || settings.robust_replays <= 0 ||
            settings.minimum_finalists <= 0 ||
            settings.maximum_finalists < settings.minimum_finalists ||
            (!settings.profiling() &&
             settings.maximum_finalists > candidate_count) ||
            settings.finalist_margin < 0.0)
        {
            throw std::invalid_argument("invalid CUDA production timing policy");
        }
        if (cudaSetDevice(device_ordinal) != cudaSuccess)
            throw std::runtime_error("failed to select CUDA sweep device");

        const auto device = llaminar2::DeviceId::cuda(device_ordinal);
        ScopedCudaMoEPrefillStream owned_stream;
        const cudaStream_t stream = owned_stream.get();
        const auto requirements = llaminar2::MoEWorkspaceBuffers::cudaMoE(
            rows,
            routed_case.hidden_size,
            routed_case.routed_expert_width,
            routed_case.expert_count,
            routed_case.experts_per_token);
        auto workspace = std::make_unique<llaminar2::DeviceWorkspaceManager>(
            device,
            requirements.total_bytes_with_alignment() + 8 * 1024 * 1024);
        if (!workspace->allocate(requirements))
            throw std::runtime_error("failed to allocate CUDA MoE workspace");

        llaminar2::CUDAMoEKernel moe(device_ordinal);
        moe.setGPUStream(stream);
        ScopedCudaMoEWorkspaceBinding workspace_binding(&moe, workspace.get());

        const auto &gateup_format =
            llaminar2::test::quantizedVerifierFormat(routed_case.routed.gate.c_str());
        const auto &down_format =
            llaminar2::test::quantizedVerifierFormat(routed_case.routed.down.c_str());
        std::vector<int> materialized_experts(
            static_cast<size_t>(routed_case.expert_count));
        std::iota(materialized_experts.begin(), materialized_experts.end(), 0);
        auto tables = prepareExpertTables(
            &moe,
            device,
            routed_case.expert_count,
            routed_case.hidden_size,
            routed_case.routed_expert_width,
            "cuda",
            470000,
            std::move(materialized_experts),
            gateup_format,
            down_format);
        if (tables.gateup_table_id < 0 || tables.down_table_id < 0)
            throw std::runtime_error("failed to publish CUDA expert tables");

        const std::vector<float> hidden_values =
            makeHiddenValues(rows, routed_case.hidden_size);
        const std::vector<float> routing_indices =
            llaminar2::test::native_vnni_dispatch::makeMoERoutingIndices(
            route_profile,
            rows,
            routed_case.experts_per_token,
            routed_case.expert_count);
        const auto route_stats =
            llaminar2::test::native_vnni_dispatch::summarizeMoERoutingProfile(
                routing_indices,
                routed_case.expert_count);
        const std::vector<float> routing_weights = makeRoutingWeights(
            rows,
            routed_case.experts_per_token);
        auto hidden = makeTensor(
            {static_cast<size_t>(rows),
             static_cast<size_t>(routed_case.hidden_size)},
            hidden_values);
        auto route_indices_tensor = makeTensor(
            {static_cast<size_t>(rows),
             static_cast<size_t>(routed_case.experts_per_token)},
            routing_indices);
        auto route_weights_tensor = makeTensor(
            {static_cast<size_t>(rows),
             static_cast<size_t>(routed_case.experts_per_token)},
            routing_weights);
        auto reference_output = makeZeros(
            {static_cast<size_t>(rows),
             static_cast<size_t>(routed_case.hidden_size)});
        auto candidate_output = makeZeros(
            {static_cast<size_t>(rows),
             static_cast<size_t>(routed_case.hidden_size)});
        for (const auto &tensor : {
                 hidden,
                 route_indices_tensor,
                 route_weights_tensor,
                 reference_output,
                 candidate_output})
        {
            if (!tensor->ensureOnDevice(device, stream))
                throw std::runtime_error("failed to publish CUDA sweep tensor");
        }

        const auto prepare_groups = [&]()
        {
            return moe.prepareExpertGroupsAsync(
                route_indices_tensor.get(),
                route_weights_tensor.get(),
                rows,
                routed_case.expert_count,
                routed_case.experts_per_token);
        };
        const auto execute_pipeline = [&](llaminar2::ITensor *output)
        {
            return moe.executeGroupedPrefillPipeline(
                hidden.get(),
                output,
                tables.gateup_table_id,
                tables.down_table_id,
                rows,
                routed_case.hidden_size,
                routed_case.routed_expert_width,
                routed_case.expert_count,
                routed_case.experts_per_token);
        };
        requireCudaBenchBody(prepare_groups(), "CUDA production grouping");
        if (cudaStreamSynchronize(stream) != cudaSuccess)
            throw std::runtime_error("failed to publish CUDA production groups");

        ScopedCudaMoEGeometryConfig forced_config;
        const CudaMoEProductionCandidate reference_candidate{
            .gateup_columns = 32,
            .down_columns = 32,
            .schedule = llaminar2::cuda::moe::
                GroupedImmaGateUpSchedule::ParallelProjections,
        };
        forced_config.setGroupedImmaGeometry(
            reference_candidate.gateup_columns,
            reference_candidate.down_columns,
            reference_candidate.schedule);
        llaminar2::PerfStatsCollector::reset();
        {
            llaminar2::CUDAGraphCapture graph(stream, device_ordinal);
            llaminar2::ScopedBackendGraphCapture capture(
                graph,
                "CUDA production MoE reference capture");
            if (!capture.begin())
                throw std::runtime_error("failed to begin CUDA reference capture");
            requireCudaBenchBody(
                execute_pipeline(reference_output.get()),
                "CUDA production reference capture");
            capture.finish();
            if (!graph.instantiate() || !graph.launch() ||
                cudaStreamSynchronize(stream) != cudaSuccess)
            {
                throw std::runtime_error(
                    "failed to instantiate or replay CUDA reference graph");
            }
        }
        if (!observedCudaMoEProductionCandidate(
                rows,
                reference_candidate))
        {
            throw std::runtime_error("PerfStats missed CUDA reference policy");
        }

        std::vector<float> reference_host(reference_output->numel());
        if (cudaMemcpyAsync(
                reference_host.data(),
                reference_output->gpu_data_ptr(),
                reference_host.size() * sizeof(float),
                cudaMemcpyDeviceToHost,
                stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess)
        {
            throw std::runtime_error("failed to materialize CUDA reference");
        }
        double serial_ms = 0.0;
        const std::vector<float> serial = runCudaRowwiseDecode(
            &moe,
            workspace.get(),
            stream,
            hidden_values,
            routing_indices,
            routing_weights,
            rows,
            routed_case.experts_per_token,
            routed_case.hidden_size,
            routed_case.routed_expert_width,
            tables.gateup_table_id,
            tables.down_table_id,
            device,
            &serial_ms);
        const CloseMetrics serial_metrics = compareVectors(
            reference_host,
            serial,
            reference_host.size());
        if (serial_metrics.bit_mismatch_count != 0 ||
            serial_metrics.nonfinite_count != 0)
        {
            throw std::runtime_error(
                "CUDA production reference is not serial-row byte-equivalent");
        }
        requireCudaBenchBody(
            prepare_groups(),
            "CUDA post-oracle production grouping");
        if (cudaStreamSynchronize(stream) != cudaSuccess)
            throw std::runtime_error("failed to restore CUDA production groups");

        CudaMoEPrefillEventTimer timer(stream);
        CudaMoEPrefillDeviceByteCertificate certificate;
        const auto measure_candidate = [&] (
            const CudaMoEProductionCandidate &candidate,
            int warmups,
            int trials,
            int replays,
            bool isolated_profile)
        {
            CudaMoEProductionEvidence evidence{};
            evidence.candidate = candidate;
            evidence.gateup_resources =
                queryCudaMoEGroupedImmaGateUpKernelResources(
                gateup_format.device_execution_codebook_id,
                static_cast<llaminar2::cuda::moe::GroupedImmaColumns>(
                    candidate.gateup_columns),
                candidate.schedule);
            evidence.down_resources = queryCudaMoEGroupedImmaKernelResources(
                down_format.device_execution_codebook_id,
                static_cast<llaminar2::cuda::moe::GroupedImmaColumns>(
                    candidate.down_columns));
            if (!evidence.gateup_resources.spillFree() ||
                !evidence.down_resources.spillFree())
            {
                throw std::runtime_error(
                    "spilling CUDA candidate reached timing: " + candidate.id());
            }

            forced_config.setGroupedImmaGeometry(
                candidate.gateup_columns,
                candidate.down_columns,
                candidate.schedule);
            llaminar2::PerfStatsCollector::reset();
            llaminar2::CUDAGraphCapture graph(stream, device_ordinal);
            llaminar2::ScopedBackendGraphCapture capture(
                graph,
                "CUDA production MoE candidate capture");
            if (!capture.begin())
                throw std::runtime_error("failed to begin CUDA candidate capture");
            requireCudaBenchBody(
                execute_pipeline(candidate_output.get()),
                "CUDA production candidate capture");
            capture.finish();
            if (!graph.instantiate())
            {
                throw std::runtime_error(
                    "failed to instantiate CUDA candidate " + candidate.id());
            }
            evidence.route_counter_ok = observedCudaMoEProductionCandidate(
                rows,
                candidate);
            if (!evidence.route_counter_ok)
            {
                throw std::runtime_error(
                    "PerfStats missed forced CUDA candidate " + candidate.id());
            }
            for (int warmup = 0; warmup < warmups; ++warmup)
                requireCudaBenchBody(graph.launch(), "CUDA candidate warmup");
            requireCudaBenchBody(
                graph.launch(),
                "CUDA candidate byte-certificate replay");
            const auto mismatch = certificate.compare(
                reinterpret_cast<const float *>(candidate_output->gpu_data_ptr()),
                reinterpret_cast<const float *>(reference_output->gpu_data_ptr()),
                candidate_output->numel(),
                stream);
            evidence.bit_mismatches = mismatch.first;
            evidence.first_bit_mismatch = mismatch.second;
            if (evidence.bit_mismatches != 0)
            {
                throw std::runtime_error(
                    "CUDA candidate " + candidate.id() +
                    " changed serial-proven output bytes");
            }

            if (isolated_profile)
            {
                /*
                 * Nsight Compute is launched with collection disabled. All
                 * allocations, transfers, route preparation, graph capture,
                 * warmup, and the device byte certificate above are therefore
                 * outside the report. The only collected transaction is this
                 * one replay of the exact production graph. Its stable stream
                 * ID and immutable request ID let the parser associate every
                 * physical child kernel with this candidate without guessing
                 * from kernel names or process-wide launch order.
                 */
                if (cudaStreamSynchronize(stream) != cudaSuccess)
                {
                    throw std::runtime_error(
                        "failed to settle CUDA preconditioning before profiling");
                }
                unsigned long long stream_id = 0;
                if (cudaStreamGetId(stream, &stream_id) != cudaSuccess)
                {
                    throw std::runtime_error(
                        "failed to identify CUDA MoE profiler stream");
                }
                if (!invokeCudaMoEProfilerControl("cuProfilerStart"))
                {
                    throw std::runtime_error(
                        "failed to start isolated CUDA MoE profiler collection");
                }
                const bool launch_ok = graph.launch();
                const cudaError_t completion = cudaStreamSynchronize(stream);
                const bool stop_ok =
                    invokeCudaMoEProfilerControl("cuProfilerStop");
                if (!launch_ok || completion != cudaSuccess || !stop_ok)
                {
                    throw std::runtime_error(
                        "isolated CUDA MoE profiler replay failed");
                }
                std::fprintf(
                    stderr,
                    "[NativeVNNIProfiler][CUDA-MoE-Prefill] request=%s "
                    "candidate=%s M=%d stream_id=%llu launches=1\n",
                    settings.profiler_request_id.c_str(),
                    candidate.id().c_str(),
                    rows,
                    stream_id);
                return evidence;
            }

            evidence.screening_samples_ms.reserve(static_cast<size_t>(trials));
            for (int trial = 0; trial < trials; ++trial)
            {
                evidence.screening_samples_ms.push_back(timer.sample(
                    replays,
                    [&]() { return graph.launch(); }));
            }
            std::sort(
                evidence.screening_samples_ms.begin(),
                evidence.screening_samples_ms.end());
            evidence.screening_replays_per_sample = replays;
            return evidence;
        };

        if (settings.profiling())
        {
            (void)measure_candidate(
                candidates.front(),
                /*warmups=*/2,
                /*trials=*/1,
                /*replays=*/1,
                /*isolated_profile=*/true);
            return;
        }

        std::vector<CudaMoEProductionEvidence> evidence_rows;
        evidence_rows.reserve(static_cast<size_t>(candidate_count));
        for (const auto &candidate : candidates)
        {
            evidence_rows.push_back(measure_candidate(
                candidate,
                settings.screening_warmups,
                settings.screening_trials,
                settings.screening_replays,
                /*isolated_profile=*/false));
        }

        std::vector<size_t> order(evidence_rows.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(
            order.begin(),
            order.end(),
            [&evidence_rows](size_t left, size_t right)
            {
                return evidence_rows[left].medianUs() <
                       evidence_rows[right].medianUs();
            });
        const double finalist_limit_us =
            evidence_rows[order.front()].medianUs() *
            (1.0 + settings.finalist_margin);
        int finalist_count = 0;
        for (size_t rank = 0; rank < order.size(); ++rank)
        {
            const bool required_rank =
                rank < static_cast<size_t>(settings.minimum_finalists);
            const bool near_best =
                evidence_rows[order[rank]].medianUs() <= finalist_limit_us;
            if ((!required_rank && !near_best) ||
                finalist_count >= settings.maximum_finalists)
            {
                continue;
            }
            evidence_rows[order[rank]].finalist = true;
            ++finalist_count;
        }

        for (auto &evidence : evidence_rows)
        {
            if (!evidence.finalist)
                continue;
            CudaMoEProductionEvidence robust = measure_candidate(
                evidence.candidate,
                settings.robust_warmups,
                settings.robust_trials,
                settings.robust_replays,
                /*isolated_profile=*/false);
            evidence.robust_samples_ms =
                std::move(robust.screening_samples_ms);
            evidence.robust_replays_per_sample =
                robust.screening_replays_per_sample;
            evidence.bit_mismatches = robust.bit_mismatches;
            evidence.first_bit_mismatch = robust.first_bit_mismatch;
            evidence.route_counter_ok = robust.route_counter_ok;
        }

        const auto winner = std::min_element(
            evidence_rows.begin(),
            evidence_rows.end(),
            [](const auto &left, const auto &right)
            {
                if (left.finalist != right.finalist)
                    return left.finalist;
                return left.medianUs() < right.medianUs();
            });
        if (winner == evidence_rows.end() || !winner->finalist)
            throw std::runtime_error("CUDA production tournament found no winner");
        winner->winner = true;

        if (csv)
        {
            for (const auto &evidence : evidence_rows)
            {
                writeCudaMoEProductionEvidence(
                    csv,
                    routed_case,
                    route_profile,
                    route_stats,
                    rows,
                    evidence);
                writeCudaMoEProductionTimingEvidence(
                    timing_csv,
                    routed_case,
                    route_profile,
                    rows,
                    evidence);
            }
        }
    }

    /**
     * @brief Prove CUDA launch geometry through the shared production path.
     *
     * With no candidate, the complete non-dominated launch family is checked.
     * Supplying one exact candidate is reserved for a focused regression whose
     * production identity is already known; setup, graph capture, routing,
     * serial-row reference, and full-buffer device comparison remain identical
     * to the exhaustive certificate.
     */
    void proveCudaProductionCandidateInvariance(
        const llaminar2::test::native_vnni_dispatch::MoERoutedPrefillCase &routed_case,
        int rows,
        int device_ordinal,
        llaminar2::test::native_vnni_dispatch::MoERoutingProfile route_profile =
            llaminar2::test::native_vnni_dispatch::MoERoutingProfile::Uniform,
        std::optional<CudaMoEProductionCandidate> proof_candidate = std::nullopt)
    {
        CudaMoEProductionSweepSettings settings{};
        settings.screening_warmups = 0;
        settings.screening_trials = 1;
        settings.screening_replays = 1;
        settings.robust_warmups = 0;
        settings.robust_trials = 1;
        settings.robust_replays = 1;
        settings.minimum_finalists = 1;
        settings.maximum_finalists = 1;
        settings.proof_candidate = std::move(proof_candidate);
        runCudaProductionMoERoutedCase(
            routed_case,
            rows,
            route_profile,
            device_ordinal,
            settings,
            nullptr,
            nullptr);
    }

    BenchResult runCudaCase(
        bool shared,
        int rows,
        int routed_top_k = 8,
        int routed_num_experts = 256,
        const char *case_name_override = nullptr,
        bool unique_routes = false,
        bool include_terminal_expert = false,
        const llaminar2::test::QuantizedVerifierFormatCase *gateup_format = nullptr,
        const llaminar2::test::QuantizedVerifierFormatCase *down_format = nullptr)
    {
        /*
         * Match the Qwen3.6 MoE production shape used by the benchmark matrix:
         * hidden width 2048, 256 routed experts, top-k 8, and 512-wide expert
         * intermediates.  Earlier smoke coverage used fewer descriptor-table
         * entries, which proved the kernels worked but under-represented the
         * descriptor pressure and grouping scan cost paid by the real model.
         */
        constexpr int shared_top_k = 1;
        constexpr int shared_num_experts = 1;
        constexpr int d_model = 2048;
        constexpr int intermediate = 512;
        const int top_k = shared ? shared_top_k : routed_top_k;
        const int num_experts = shared ? shared_num_experts : routed_num_experts;
        const auto &selected_gateup_format = gateup_format
                                                 ? *gateup_format
                                                 : llaminar2::test::quantizedVerifierFormat("IQ2_S");
        const auto &selected_down_format = down_format
                                               ? *down_format
                                               : llaminar2::test::quantizedVerifierFormat("IQ4_XS");
        const int iterations = envInt("LLAMINAR_MOE_VERIFIER_PREFILL_ITERS", 30);
        const int warmups = envInt("LLAMINAR_MOE_VERIFIER_PREFILL_WARMUPS", 5);
        const auto device = llaminar2::DeviceId::cuda(0);

        EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
        cudaStream_t stream = nullptr;
        EXPECT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

        auto moe = KernelFactory::createMoEKernel(device);
        EXPECT_NE(moe, nullptr);
        moe->setGPUStream(stream);
        auto *workspace_consumer = dynamic_cast<llaminar2::IWorkspaceConsumer *>(moe.get());
        EXPECT_NE(workspace_consumer, nullptr);
        const int workspace_num_experts = std::max(num_experts, routed_num_experts);
        const int workspace_top_k = std::max(top_k, routed_top_k);
        auto reqs = llaminar2::MoEWorkspaceBuffers::cudaMoE(
            /*max_seq_len=*/rows,
            d_model,
            intermediate,
            workspace_num_experts,
            workspace_top_k);
        auto workspace = std::make_unique<llaminar2::DeviceWorkspaceManager>(
            device,
            reqs.total_bytes_with_alignment() + 8 * 1024 * 1024);
        EXPECT_TRUE(workspace->allocate(reqs));
        EXPECT_TRUE(workspace->hasBuffer(
            llaminar2::MoEWorkspaceBuffers::CUDA_DECODE_GATEUP_GATE_PTRS));
        EXPECT_TRUE(workspace->hasBuffer(
            llaminar2::MoEWorkspaceBuffers::CUDA_DECODE_GATEUP_UP_PTRS));
        EXPECT_TRUE(workspace->hasBuffer(
            llaminar2::MoEWorkspaceBuffers::CUDA_DECODE_DOWN_GATE_PTRS));
        EXPECT_TRUE(workspace->hasBuffer(
            llaminar2::MoEWorkspaceBuffers::CUDA_DECODE_DOWN_UP_PTRS));
        workspace_consumer->bindWorkspace(workspace.get());

        ScopedCudaMoEPrefillConfig prefill_config;
        const int tile_m_override =
            envInt("LLAMINAR_MOE_VERIFIER_PREFILL_CUDA_TILE_M", 0);
        /*
         * Production stays on the kernel's auto selector by default.  The
         * optional harness override is a speedometer for tuning M=1..4 verifier
         * tiles without editing source between runs.
         */
        prefill_config.set(/*tile_m=*/tile_m_override, /*fuse_swiglu=*/true);
        const auto hidden_values = makeHiddenValues(rows, d_model);
        auto routing_indices = unique_routes
                                   ? makeUniqueRoutingIndices(rows, top_k, num_experts)
                                   : makeRoutingIndices(rows, top_k, num_experts);
        if (include_terminal_expert)
            publishTerminalExpertRoute(routing_indices, rows, top_k, num_experts);
        const auto routing_weights = makeRoutingWeights(rows, top_k);
        auto tables = prepareExpertTables(
            moe.get(), device, num_experts, d_model, intermediate, "cuda", 270000,
            uniqueExpertIdsFromRoutes(routing_indices, num_experts),
            selected_gateup_format,
            selected_down_format);
        auto hidden = makeTensor({static_cast<size_t>(rows), static_cast<size_t>(d_model)}, hidden_values);
        auto route_indices_tensor = makeTensor({static_cast<size_t>(rows), static_cast<size_t>(top_k)}, routing_indices);
        auto route_weights_tensor = makeTensor({static_cast<size_t>(rows), static_cast<size_t>(top_k)}, routing_weights);
        auto grouped_output = makeZeros({static_cast<size_t>(rows), static_cast<size_t>(d_model)});
        EXPECT_TRUE(hidden->ensureOnDevice(device, stream));
        EXPECT_TRUE(route_indices_tensor->ensureOnDevice(device, stream));
        EXPECT_TRUE(route_weights_tensor->ensureOnDevice(device, stream));
        EXPECT_TRUE(grouped_output->ensureOnDevice(device, stream));

        /**
         * @brief Run only the device-resident route grouping half of the case.
         *
         * This perf split tells future tuning whether verifier time is stuck in
         * route metadata construction or in the grouped GEMV/scatter producer.
         */
        auto run_prepare = [&]()
        {
            if (shared)
            {
                return moe->prepareSharedExpertPrefillGroup(rows);
            }
            return moe->prepareExpertGroupsAsync(
                route_indices_tensor.get(), route_weights_tensor.get(),
                rows, num_experts, top_k);
        };

        auto run_pipeline = [&]()
        {
            return moe->executeGroupedPrefillPipeline(
                hidden.get(), grouped_output.get(),
                tables.gateup_table_id, tables.down_table_id,
                rows, d_model, intermediate, num_experts, top_k);
        };

        auto run_grouped = [&]()
        {
            return run_prepare() && run_pipeline();
        };

        for (int i = 0; i < warmups; ++i)
            requireCudaBenchBody(run_grouped(), "warmup");
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        const double eager_ms = timeCudaEvents(stream, iterations, run_grouped);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        const double prepare_ms = timeCudaEvents(stream, iterations, run_prepare);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        requireCudaBenchBody(run_prepare(), "pipeline preparation");
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        const double pipeline_ms = timeCudaEvents(stream, iterations, run_pipeline);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        /*
         * Use the production capture owner rather than a raw begin/end pair.
         * Its GraphCaptureGuard tells TensorBase that stage-local writes are
         * provisional graph state, so no external completion event is recorded
         * inside the graph.  Its transaction destructor also closes capture if
         * a benchmark launch throws, preventing one failed cell from leaving
         * the CUDA stream in capture mode and poisoning every later cell.
         */
        llaminar2::CUDAGraphCapture graph(stream, /*device_ordinal=*/0);
        {
            llaminar2::ScopedBackendGraphCapture capture_transaction(
                graph,
                "CUDA MoE routed verifier perf capture");
            if (!capture_transaction.begin())
            {
                throw std::runtime_error(
                    "CUDA MoE verifier graph capture begin failed");
            }
            requireCudaBenchBody(run_grouped(), "graph capture");
            capture_transaction.finish();
        }
        if (!graph.instantiate())
        {
            throw std::runtime_error(
                "CUDA MoE verifier graph instantiate failed");
        }
        for (int i = 0; i < warmups; ++i)
            requireCudaBenchBody(graph.launch(), "graph warmup replay");
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        const double graph_ms = timeCudaEvents(
            stream,
            iterations,
            [&]()
            {
                return graph.launch();
            });
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        TransferEngine::publishDeviceWrite(grouped_output, device, stream);
        std::vector<float> grouped(
            grouped_output->data(),
            grouped_output->data() + grouped_output->numel());

        double rowwise_ms = 0.0;
        std::vector<float> rowwise = runCudaRowwiseDecode(
            moe.get(), workspace.get(), stream, hidden_values, routing_indices, routing_weights,
            rows, top_k, d_model, intermediate,
            tables.gateup_table_id, tables.down_table_id, device, &rowwise_ms);
        CloseMetrics metrics = compareVectors(grouped, rowwise, static_cast<size_t>(d_model));

        EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);

        return BenchResult{
            "cuda",
            case_name_override ? case_name_override : (shared ? "shared" : "routed"),
            rows,
            top_k,
            num_experts,
            d_model,
            intermediate,
            eager_ms,
            prepare_ms,
            pipeline_ms,
            graph_ms,
            rowwise_ms,
            metrics};
    }

    /**
     * @brief Exercise the production SharedExpertFFNStage verifier route.
     *
     * This is stricter than the isolated IMoE shared-prefill kernel test.  It
     * measures the graph-captured grouped route actually wired by
     * `SharedExpertFFNStage` and compares it against that same stage's serial
     * decode-equivalent replay.  Passing this gate is required before treating
     * shared expert FFN M=2..4 as production-economical.
     */
    BenchResult runCudaSharedExpertStageCase(int rows, const QuantFormatCase &format)
    {
        constexpr int d_model = 2048;
        constexpr int intermediate = 512;
        const int iterations = envInt("LLAMINAR_MOE_VERIFIER_PREFILL_ITERS", 120);
        const int warmups = envInt("LLAMINAR_MOE_VERIFIER_PREFILL_WARMUPS", 5);
        const auto device = llaminar2::DeviceId::cuda(0);

        EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
        cudaStream_t stream = nullptr;
        EXPECT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

        auto gate_w = format.create(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 6101);
        auto up_w = format.create(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 6102);
        auto down_w = format.create(
            {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)}, 6103);
        auto moe = KernelFactory::createMoEKernel(device);
        EXPECT_NE(moe, nullptr);
        moe->setGPUStream(stream);
        auto *moe_workspace = dynamic_cast<llaminar2::IWorkspaceConsumer *>(moe.get());
        EXPECT_NE(moe_workspace, nullptr);
        auto prepared = llaminar2::test::makeGpuPreparedFFNFixture(
            gate_w.get(),
            up_w.get(),
            down_w.get(),
            device,
            std::string("perf.moe_verifier.cuda.shared_stage.") + format.name,
            llaminar2::ModelContextId{390100});

        const auto hidden_values = makeHiddenValues(rows, d_model);
        auto hidden = makeTensor({static_cast<size_t>(rows), static_cast<size_t>(d_model)}, hidden_values);
        auto grouped_output = makeZeros({static_cast<size_t>(rows), static_cast<size_t>(d_model)});
        /*
         * Reproduce the production graph resolver's stable scratch contract.
         * SharedExpertFFNStage deliberately refuses to manufacture GPU scratch
         * during execute(): both projection destinations must already own fixed
         * device addresses before warmup and graph capture begin. Keeping these
         * tensors alive for the complete benchmark also ensures that replay
         * measures the production grouped kernels rather than an allocation or
         * pointer-rebinding setup path.
         */
        auto graph_gate_scratch = makeZeros(
            {static_cast<size_t>(rows), static_cast<size_t>(intermediate)});
        auto graph_up_scratch = makeZeros(
            {static_cast<size_t>(rows), static_cast<size_t>(intermediate)});
        EXPECT_TRUE(hidden->ensureOnDevice(device, stream));
        EXPECT_TRUE(grouped_output->ensureOnDevice(device, stream));
        EXPECT_TRUE(graph_gate_scratch->ensureOnDevice(device, stream));
        EXPECT_TRUE(graph_up_scratch->ensureOnDevice(device, stream));

        auto make_params = [&](llaminar2::TensorBase *output,
                               bool grouped_verifier)
        {
            llaminar2::SharedExpertFFNStage::Params params;
            params.device_id = device;
            params.input = hidden.get();
            params.gate_w = gate_w.get();
            params.up_w = up_w.get();
            params.down_w = down_w.get();
            params.output = output;
            params.gate_scratch = graph_gate_scratch.get();
            params.up_scratch = graph_up_scratch.get();
            params.seq_len = rows;
            params.d_model = d_model;
            params.intermediate = intermediate;
            params.prepared_ref_gate = prepared.gate_ref;
            params.prepared_ref_up = prepared.up_ref;
            params.prepared_ref_down = prepared.down_ref;
            params.prepared_store = prepared.store.get();
            params.force_grouped_verifier_prefill_for_decode = grouped_verifier;
            params.force_decode_equivalent_verifier_prefill = false;
            return params;
        };

        llaminar2::SharedExpertFFNStage grouped_stage(
            make_params(grouped_output.get(), /*grouped_verifier=*/true));
        grouped_stage.setGPUStream(stream);
        grouped_stage.setMoEKernelForTesting(moe.get());
        EXPECT_TRUE(grouped_stage.usesGroupedVerifierPrefillRouteForTesting());

        auto reqs = grouped_stage.getWorkspaceRequirements(rows, d_model, intermediate);
        reqs.merge(llaminar2::MoEWorkspaceBuffers::cudaMoE(
            rows,
            d_model,
            intermediate,
            /*num_experts=*/256,
            /*top_k=*/8));
        auto workspace = std::make_unique<llaminar2::DeviceWorkspaceManager>(
            device,
            reqs.total_bytes_with_alignment() + 8 * 1024 * 1024);
        EXPECT_TRUE(workspace->allocate(reqs));
        grouped_stage.bindWorkspace(workspace.get());
        if (moe_workspace)
            moe_workspace->bindWorkspace(workspace.get());

        llaminar2::DeviceNativeVNNIMatrixDesc gate_desc{};
        llaminar2::DeviceNativeVNNIMatrixDesc up_desc{};
        llaminar2::DeviceNativeVNNIMatrixDesc down_desc{};
        EXPECT_TRUE(prepared.gate_kernel->exportNativeVNNIMatrixDesc(gate_desc));
        EXPECT_TRUE(prepared.up_kernel->exportNativeVNNIMatrixDesc(up_desc));
        EXPECT_TRUE(prepared.down_kernel->exportNativeVNNIMatrixDesc(down_desc));
        const int gateup_table =
            moe ? moe->uploadGroupedExpertGateUpDescriptorTables(
                      &gate_desc, &up_desc, /*num_experts=*/1, d_model, intermediate)
                : -1;
        const int down_table =
            moe ? moe->uploadGroupedExpertDownDescriptorTable(
                      &down_desc, /*num_experts=*/1, d_model, intermediate)
                : -1;
        EXPECT_GE(gateup_table, 0);
        EXPECT_GE(down_table, 0);

        llaminar2::testing::MockDeviceContext ctx(
            device, llaminar2::ComputeBackendType::GPU_CUDA);
        auto run_grouped = [&]()
        {
            return grouped_stage.execute(&ctx);
        };
        std::vector<float> serial;
        auto run_serial = [&]()
        {
            if (!moe || gateup_table < 0 || down_table < 0)
                return false;
            serial.clear();
            serial.reserve(static_cast<size_t>(rows) * static_cast<size_t>(d_model));
            for (int row = 0; row < rows; ++row)
            {
                const auto row_begin = hidden_values.begin() + static_cast<ptrdiff_t>(row) * d_model;
                std::vector<float> row_hidden_values(row_begin, row_begin + d_model);
                auto row_hidden = makeTensor({1u, static_cast<size_t>(d_model)}, row_hidden_values);
                auto row_gate = makeZeros({1u, static_cast<size_t>(intermediate)});
                auto row_up = makeZeros({1u, static_cast<size_t>(intermediate)});
                auto row_output = makeZeros({1u, static_cast<size_t>(d_model)});
                EXPECT_TRUE(row_hidden->ensureOnDevice(device, stream));
                EXPECT_TRUE(row_gate->ensureOnDevice(device, stream));
                EXPECT_TRUE(row_up->ensureOnDevice(device, stream));
                EXPECT_TRUE(row_output->ensureOnDevice(device, stream));

                constexpr int expert_id = 0;
                constexpr float expert_weight = 1.0f;
                llaminar2::ITensor *gate_outputs[1] = {row_gate.get()};
                llaminar2::ITensor *up_outputs[1] = {row_up.get()};
                if (!moe->groupedExpertGateUpDecodeFromTable(
                        row_hidden.get(),
                        &expert_id,
                        gateup_table,
                        1,
                        gate_outputs,
                        up_outputs,
                        d_model,
                        intermediate))
                {
                    return false;
                }
                if (!moe->groupedExpertDownDecodeFromTable(
                        gate_outputs,
                        up_outputs,
                        &expert_id,
                        &expert_weight,
                        down_table,
                        1,
                        row_output.get(),
                        d_model,
                        intermediate))
                {
                    return false;
                }
                EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
                TransferEngine::publishDeviceWrite(row_output, device, stream);
                serial.insert(
                    serial.end(),
                    row_output->data(),
                    row_output->data() + row_output->numel());
            }
            return true;
        };

        for (int i = 0; i < warmups; ++i)
            EXPECT_TRUE(run_grouped());
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_TRUE(grouped_stage.isGraphCapturable());

        const double eager_ms = timeCudaEvents(stream, iterations, run_grouped);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        llaminar2::CUDAGraphCapture graph(stream, /*device_ordinal=*/0);
        {
            llaminar2::ScopedBackendGraphCapture capture_transaction(
                graph,
                "CUDA MoE shared-expert verifier perf capture");
            if (!capture_transaction.begin())
            {
                throw std::runtime_error(
                    "CUDA shared-expert verifier graph capture begin failed");
            }
            requireCudaBenchBody(run_grouped(), "shared graph capture");
            capture_transaction.finish();
        }
        if (!graph.instantiate())
        {
            throw std::runtime_error(
                "CUDA shared-expert verifier graph instantiate failed");
        }
        for (int i = 0; i < warmups; ++i)
            requireCudaBenchBody(graph.launch(), "shared graph warmup replay");
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        const double graph_ms = timeCudaEvents(
            stream,
            iterations,
            [&]()
            {
                return graph.launch();
            });
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        const double serial_ms = timeCudaEvents(stream, std::max(1, iterations / 4), run_serial);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        TransferEngine::publishDeviceWrite(grouped_output, device, stream);
        std::vector<float> grouped(
            grouped_output->data(),
            grouped_output->data() + grouped_output->numel());
        CloseMetrics metrics = compareVectors(grouped, serial, static_cast<size_t>(d_model));

        grouped_stage.unbindWorkspace();
        if (moe_workspace)
            moe_workspace->unbindWorkspace();
        EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);

        return BenchResult{
            "cuda",
            std::string("shared_stage_ffn_") + format.name,
            rows,
            1,
            1,
            d_model,
            intermediate,
            eager_ms,
            0.0,
            0.0,
            graph_ms,
            serial_ms,
            metrics};
    }
}
#endif

TEST(Perf__MoEVerifierPrefill, CUDA_AllFormatProductionPrefillCandidatesAreSpillFree)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    std::set<uint8_t> execution_codebooks;
    for (const auto &format : llaminar2::test::quantizedVerifierFormats())
        execution_codebooks.insert(format.device_execution_codebook_id);

    constexpr std::array<int, 7> ordered_block_widths{
        64, 96, 128, 160, 192, 224, 256,
    };
    constexpr std::array<llaminar2::cuda::moe::GroupedImmaColumns, 4>
        down_imma_widths{
            llaminar2::cuda::moe::GroupedImmaColumns::Columns32,
            llaminar2::cuda::moe::GroupedImmaColumns::Columns64,
            llaminar2::cuda::moe::GroupedImmaColumns::Columns128,
            llaminar2::cuda::moe::GroupedImmaColumns::Columns256,
        };
    constexpr std::array<llaminar2::cuda::moe::GroupedImmaColumns, 3>
        gateup_imma_widths{
            llaminar2::cuda::moe::GroupedImmaColumns::Columns32,
            llaminar2::cuda::moe::GroupedImmaColumns::Columns64,
            llaminar2::cuda::moe::GroupedImmaColumns::Columns128,
        };
    constexpr std::array<
        llaminar2::cuda::moe::GroupedImmaGateUpSchedule,
        2>
        gateup_schedules{
            llaminar2::cuda::moe::GroupedImmaGateUpSchedule::
                ParallelProjections,
            llaminar2::cuda::moe::GroupedImmaGateUpSchedule::
                PairedProjections,
        };
    for (const uint8_t codebook : execution_codebooks)
    {
        SCOPED_TRACE(static_cast<unsigned>(codebook));

        for (const auto columns : down_imma_widths)
        {
            SCOPED_TRACE(llaminar2::cuda::moe::groupedImmaColumns(columns));
            const auto grouped_imma =
                queryCudaMoEGroupedImmaKernelResources(codebook, columns);
            EXPECT_TRUE(grouped_imma.spillFree())
                << "grouped IMMA registers="
                << grouped_imma.registers_per_thread
                << " local_bytes="
                << grouped_imma.local_memory_bytes_per_thread
                << " active_blocks="
                << grouped_imma.max_active_blocks_per_sm;
            EXPECT_GT(grouped_imma.registers_per_thread, 0);
            EXPECT_GT(grouped_imma.max_active_blocks_per_sm, 0);

            /* The original 32-column geometry retains its stronger baseline. */
            if (columns == llaminar2::cuda::moe::GroupedImmaColumns::Columns32 &&
                (codebook == 4 || codebook == 13))
            {
                EXPECT_LE(grouped_imma.registers_per_thread, 48)
                    << "dominant codebook lost cooperative decode occupancy";
                EXPECT_GE(grouped_imma.max_active_blocks_per_sm, 10)
                    << "dominant codebook admits too few resident CTAs";
            }
        }

        for (const auto columns : gateup_imma_widths)
        {
            for (const auto schedule : gateup_schedules)
            {
                SCOPED_TRACE(
                    llaminar2::cuda::moe::groupedImmaColumns(columns));
                SCOPED_TRACE(
                    schedule == llaminar2::cuda::moe::
                                    GroupedImmaGateUpSchedule::
                                        PairedProjections
                        ? "paired_projections"
                        : "parallel_projections");
                const auto grouped_imma_gateup =
                    queryCudaMoEGroupedImmaGateUpKernelResources(
                        codebook,
                        columns,
                        schedule);
                EXPECT_TRUE(grouped_imma_gateup.spillFree())
                    << "fused gate/up IMMA registers="
                    << grouped_imma_gateup.registers_per_thread
                    << " local_bytes="
                    << grouped_imma_gateup.local_memory_bytes_per_thread
                    << " active_blocks="
                    << grouped_imma_gateup.max_active_blocks_per_sm;
                EXPECT_GT(grouped_imma_gateup.registers_per_thread, 0);
                EXPECT_GT(grouped_imma_gateup.max_active_blocks_per_sm, 0);
            }
        }

        for (const int block_width : ordered_block_widths)
        {
            SCOPED_TRACE(block_width);
            const auto gateup = queryCudaMoEPrefillKernelResources(
                codebook,
                /*component=*/0,
                block_width);
            EXPECT_TRUE(gateup.spillFree());
            EXPECT_GT(gateup.registers_per_thread, 0);
            EXPECT_GT(gateup.max_active_blocks_per_sm, 0);

            const auto canonical_down = queryCudaMoEPrefillKernelResources(
                codebook,
                /*component=*/2,
                block_width);
            EXPECT_TRUE(canonical_down.spillFree());
            EXPECT_GT(canonical_down.registers_per_thread, 0);
            EXPECT_GT(canonical_down.max_active_blocks_per_sm, 0);
        }

        /*
         * Every pinned Qwen MoE geometry routes top-8 experts. A direct-down
         * block therefore needs exactly eight warps: larger blocks add only
         * idle warps, consume more residency, and cannot perform less work.
         * Keeping those dominated widths out of the candidate registry avoids
         * spending corpus time proving a launch geometry that cannot win.
         */
        const auto direct_down = queryCudaMoEPrefillKernelResources(
            codebook,
            /*component=*/1,
            /*block_threads=*/8 * 32);
        EXPECT_TRUE(direct_down.spillFree());
        EXPECT_GT(direct_down.registers_per_thread, 0);
        EXPECT_GT(direct_down.max_active_blocks_per_sm, 0);
    }

    const auto gather = queryCudaMoEPrefillKernelResources(
        /*execution_codebook=*/0,
        /*component=*/3,
        /*block_threads=*/256);
    const auto gateup_reduce = queryCudaMoEPrefillKernelResources(
        /*execution_codebook=*/0,
        /*component=*/4,
        /*block_threads=*/32);
    const auto canonical_down_reduce = queryCudaMoEPrefillKernelResources(
        /*execution_codebook=*/0,
        /*component=*/5,
        /*block_threads=*/64);
    EXPECT_TRUE(gather.spillFree())
        << "gather registers=" << gather.registers_per_thread
        << " local_bytes=" << gather.local_memory_bytes_per_thread
        << " active_blocks=" << gather.max_active_blocks_per_sm;
    EXPECT_TRUE(gateup_reduce.spillFree())
        << "gate/up reduce registers=" << gateup_reduce.registers_per_thread
        << " local_bytes=" << gateup_reduce.local_memory_bytes_per_thread
        << " active_blocks=" << gateup_reduce.max_active_blocks_per_sm;
    EXPECT_TRUE(canonical_down_reduce.spillFree())
        << "down reduce registers=" << canonical_down_reduce.registers_per_thread
        << " local_bytes=" << canonical_down_reduce.local_memory_bytes_per_thread
        << " active_blocks=" << canonical_down_reduce.max_active_blocks_per_sm;
#endif
}

TEST(Perf__MoEVerifierPrefill, CUDA_ProductionGGUFMixtureCandidateTrainer)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";
    if (envInt("LLAMINAR_CUDA_MOE_PRODUCTION_SWEEP", 0) == 0)
    {
        GTEST_SKIP()
            << "Set LLAMINAR_CUDA_MOE_PRODUCTION_SWEEP=1 to run the "
               "GGUF-derived CUDA production tournament";
    }

    CudaMoEProductionSweepSettings settings{};
    settings.screening_warmups = envInt(
        "LLAMINAR_CUDA_MOE_PRODUCTION_SWEEP_SCREEN_WARMUPS", 1);
    settings.screening_trials = envInt(
        "LLAMINAR_CUDA_MOE_PRODUCTION_SWEEP_SCREEN_TRIALS", 3);
    settings.screening_replays = envInt(
        "LLAMINAR_CUDA_MOE_PRODUCTION_SWEEP_SCREEN_REPLAYS", 2);
    settings.robust_warmups = envInt(
        "LLAMINAR_CUDA_MOE_PRODUCTION_SWEEP_ROBUST_WARMUPS", 2);
    settings.robust_trials = envInt(
        "LLAMINAR_CUDA_MOE_PRODUCTION_SWEEP_ROBUST_TRIALS", 15);
    settings.robust_replays = envInt(
        "LLAMINAR_CUDA_MOE_PRODUCTION_SWEEP_ROBUST_REPLAYS", 4);
    settings.minimum_finalists = envInt(
        "LLAMINAR_CUDA_MOE_PRODUCTION_SWEEP_MIN_FINALISTS", 4);
    settings.maximum_finalists = envInt(
        "LLAMINAR_CUDA_MOE_PRODUCTION_SWEEP_MAX_FINALISTS", 8);
    settings.finalist_margin = envPositiveDouble(
        "LLAMINAR_CUDA_MOE_PRODUCTION_SWEEP_FINALIST_MARGIN", 0.05);
    const std::vector<int> m_values = envCsvPositiveInts(
        "LLAMINAR_CUDA_MOE_PRODUCTION_SWEEP_M",
        {64, 256, 1024, 2048, 4096, 8192, 16384});
    const int maximum_cells = envInt(
        "LLAMINAR_CUDA_MOE_PRODUCTION_SWEEP_MAX_CELLS",
        std::numeric_limits<int>::max());
    const int device_ordinal = envInt(
        "LLAMINAR_CUDA_MOE_PRODUCTION_SWEEP_DEVICE", 0);
    const char *route_profile_environment =
        std::getenv("LLAMINAR_CUDA_MOE_PRODUCTION_SWEEP_ROUTE_PROFILE");
    if (!route_profile_environment || !*route_profile_environment)
    {
        throw std::runtime_error(
            "CUDA production MoE sweep requires one explicit route profile");
    }
    const auto route_profile =
        llaminar2::test::native_vnni_dispatch::parseMoERoutingProfile(
            route_profile_environment);

    settings.profiler_request_id =
        llaminar2::test::native_vnni_dispatch::profilerRequestId();
    const std::string profiler_candidate_id =
        llaminar2::test::native_vnni_dispatch::profilerEnvironment(
            "LLAMINAR_CUDA_MOE_PRODUCTION_PROFILE_CANDIDATE");
    if (settings.profiling() != !profiler_candidate_id.empty())
    {
        throw std::runtime_error(
            "isolated CUDA MoE profiling requires both a profiler request ID "
            "and LLAMINAR_CUDA_MOE_PRODUCTION_PROFILE_CANDIDATE");
    }
    if (settings.profiling())
    {
        settings.profiler_candidate =
            findCudaMoEProductionCandidate(profiler_candidate_id);
        if (!settings.profiler_candidate)
        {
            throw std::runtime_error(
                "unknown CUDA MoE production profiler candidate " +
                profiler_candidate_id);
        }
        const std::string selected_cases =
            llaminar2::test::native_vnni_dispatch::profilerEnvironment(
                "LLAMINAR_CUDA_MOE_PRODUCTION_SWEEP_CASES");
        if (selected_cases.empty() ||
            selected_cases.find(',') != std::string::npos ||
            m_values.size() != 1 || maximum_cells != 1)
        {
            throw std::runtime_error(
                "isolated CUDA MoE profiling requires one explicit case, one M, "
                "and LLAMINAR_CUDA_MOE_PRODUCTION_SWEEP_MAX_CELLS=1");
        }
    }

    ScopedEnvOverride rowwise_iterations(
        "LLAMINAR_MOE_VERIFIER_PREFILL_ROWWISE_ITERS", "1");
    ScopedEnvOverride perfstats("LLAMINAR_PERF_STATS_JSON", "1");

    std::unique_ptr<std::FILE, CudaMoEFileCloser> owned_csv;
    std::FILE *csv = settings.profiling() ? nullptr : stdout;
    if (!settings.profiling())
    {
        if (const char *path =
                std::getenv("LLAMINAR_CUDA_MOE_PRODUCTION_SWEEP_CSV");
            path && *path)
        {
            owned_csv.reset(std::fopen(path, "w"));
            ASSERT_NE(owned_csv.get(), nullptr)
                << "failed to open CUDA aggregate CSV " << path;
            csv = owned_csv.get();
        }
        std::fprintf(
            csv,
            "backend,phase,case_id,route_profile,source_gate,source_up,source_down,"
            "gate_execution_codebook,up_execution_codebook,"
            "down_execution_codebook,hidden_size,expert_width,expert_count,"
            "top_k,route_active_experts,route_max_assignments,route_assignment_cv,"
            "m,candidate_id,gate_tile_m,gate_tile_n,down_tile_m,"
            "down_tile_n,gate_local_bytes,down_local_bytes,gate_registers,"
            "down_registers,gate_shared_bytes,down_shared_bytes,"
            "gate_active_blocks_per_sm,down_active_blocks_per_sm,"
            "screening_sample_count,screening_replays,robust_sample_count,"
            "robust_replays,selected_sample_count,selected_replays,min_us,"
            "median_us,p95_us,mad_us,cv,timing_sample_digest,bit_mismatches,"
            "first_bit_mismatch,route_counter_ok,finalist,is_winner\n");
    }

    std::unique_ptr<std::FILE, CudaMoEFileCloser> timing_csv;
    if (!settings.profiling())
    {
        if (const char *path =
                std::getenv("LLAMINAR_CUDA_MOE_PRODUCTION_SWEEP_TIMING_CSV");
            path && *path)
        {
            timing_csv.reset(std::fopen(path, "w"));
            ASSERT_NE(timing_csv.get(), nullptr)
                << "failed to open CUDA timing CSV " << path;
            std::fprintf(
                timing_csv.get(),
                "backend,phase,case_id,route_profile,m,candidate_id,timing_phase,"
                "sample_index,timed_replays,latency_us,latency_ms_hex\n");
        }
    }

    int executed_cells = 0;
    for (const auto &routed_case :
         llaminar2::test::native_vnni_dispatch::nativeVnniMoERoutedPrefillCases())
    {
        if (!envCsvContainsOrUnset(
                "LLAMINAR_CUDA_MOE_PRODUCTION_SWEEP_CASES",
                routed_case.evidenceId()) ||
            !envCsvContainsOrUnset(
                "LLAMINAR_CUDA_MOE_PRODUCTION_SWEEP_GATE_FORMATS",
                routed_case.routed.gate) ||
            !envCsvContainsOrUnset(
                "LLAMINAR_CUDA_MOE_PRODUCTION_SWEEP_DOWN_FORMATS",
                routed_case.routed.down))
        {
            continue;
        }
        for (const int rows : m_values)
        {
            if (executed_cells >= maximum_cells)
                break;
            SCOPED_TRACE(
                routed_case.evidenceId() + "/route=" +
                std::string(
                    llaminar2::test::native_vnni_dispatch::
                        moeRoutingProfileName(route_profile)) +
                "/M" + std::to_string(rows));
            std::fprintf(
                stderr,
                "[CUDA MoE production sweep] case=%s route=%s M=%d gate/up=%s "
                "down=%s hidden=%d expert_width=%d experts=%d top_k=%d\n",
                routed_case.evidenceId().c_str(),
                std::string(
                    llaminar2::test::native_vnni_dispatch::
                        moeRoutingProfileName(route_profile)).c_str(),
                rows,
                routed_case.routed.gate.c_str(),
                routed_case.routed.down.c_str(),
                routed_case.hidden_size,
                routed_case.routed_expert_width,
                routed_case.expert_count,
                routed_case.experts_per_token);
            runCudaProductionMoERoutedCase(
                routed_case,
                rows,
                route_profile,
                device_ordinal,
                settings,
                csv,
                timing_csv.get());
            ++executed_cells;
        }
        if (executed_cells >= maximum_cells)
            break;
    }
    EXPECT_GT(executed_cells, 0)
        << "CUDA production filters selected no GGUF-derived cells";
#endif
}

/**
 * @brief Prove every exact-overlay CUDA geometry against serial row decode.
 *
 * The two M values exercise both compiled gate/up widths. Each format tuple is
 * sourced from the pinned GGUF inventory and every one of the 24 candidate
 * schedules runs through the production captured pipeline. The tournament
 * performs a full-buffer device byte comparison before emitting any timing,
 * so adding a tuple to the runtime exact-overlay table without extending this
 * certificate leaves an immediately visible test gap.
 */
TEST(Perf__MoEVerifierPrefill, CUDA_ProductionCandidatesAreSerialRowByteInvariant)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";
    if (envInt("LLAMINAR_CUDA_MOE_CANDIDATE_INVARIANCE", 0) == 0)
    {
        GTEST_SKIP()
            << "Set LLAMINAR_CUDA_MOE_CANDIDATE_INVARIANCE=1 to run the "
               "complete production launch-policy eligibility proof";
    }

    struct ExactOverlayFormatTuple
    {
        const char *gate_up = nullptr;
        const char *down = nullptr;
    };
    constexpr std::array<ExactOverlayFormatTuple, 4> exact_overlay_formats{{
        {"IQ2_S", "IQ4_XS"},
        {"IQ2_S", "Q6_K"},
        {"IQ3_S", "Q6_K"},
        {"IQ2_S", "IQ3_S"},
    }};
    const std::vector<int> rows_to_prove = envCsvPositiveInts(
        "LLAMINAR_CUDA_MOE_CANDIDATE_INVARIANCE_M", {64, 1024});
    const int device_ordinal = envInt(
        "LLAMINAR_CUDA_MOE_CANDIDATE_INVARIANCE_DEVICE", 0);
    const auto &cases =
        llaminar2::test::native_vnni_dispatch::nativeVnniMoERoutedPrefillCases();

    ScopedEnvOverride rowwise_iterations(
        "LLAMINAR_MOE_VERIFIER_PREFILL_ROWWISE_ITERS", "1");
    ScopedEnvOverride perfstats("LLAMINAR_PERF_STATS_JSON", "1");
    for (const auto &format : exact_overlay_formats)
    {
        const auto selected = std::find_if(
            cases.begin(),
            cases.end(),
            [&](const auto &candidate)
            {
                return candidate.hidden_size == 2048 &&
                       candidate.routed_expert_width == 512 &&
                       candidate.expert_count == 256 &&
                       candidate.experts_per_token == 8 &&
                       candidate.routed.gate == format.gate_up &&
                       candidate.routed.up == format.gate_up &&
                       candidate.routed.down == format.down;
            });
        ASSERT_NE(selected, cases.end())
            << "pinned GGUF manifest lacks exact-overlay tuple gate/up="
            << format.gate_up << " down=" << format.down;
        for (const int rows : rows_to_prove)
        {
            SCOPED_TRACE(
                std::string("gate/up=") + format.gate_up +
                "/down=" + format.down + "/M=" +
                std::to_string(rows));
            proveCudaProductionCandidateInvariance(
                *selected, rows, device_ordinal);
        }
    }
#endif
}

/**
 * @brief Regress the staged Qwen 35B metadata path under uneven expert tails.
 *
 * IQ2_S gate/up and IQ4_XS down map to execution codebooks 13 and 4. At the
 * installed 32-column geometry those are the two specializations that stage
 * immutable scale metadata in shared memory. Power-law routing creates both
 * full and partial sixteen-row expert tiles, while M=512 is the production
 * capture bucket used by the fixed benchmark prompt. The selected graph is
 * compared over its complete output buffer against serial row decode before
 * the test can pass.
 */
TEST(
    Perf__MoEVerifierPrefill,
    CUDA_StagedIQ2SIQ4MetadataIsSerialRowByteInvariant)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    const auto candidate =
        findCudaMoEProductionCandidate("imma_paired_g32__d32");
    ASSERT_TRUE(candidate.has_value());
    const auto &cases =
        llaminar2::test::native_vnni_dispatch::nativeVnniMoERoutedPrefillCases();
    const auto selected = std::find_if(
        cases.begin(),
        cases.end(),
        [](const auto &routed_case)
        {
            return routed_case.hidden_size == 2048 &&
                   routed_case.routed_expert_width == 512 &&
                   routed_case.expert_count == 256 &&
                   routed_case.experts_per_token == 8 &&
                   routed_case.routed.gate == "IQ2_S" &&
                   routed_case.routed.up == "IQ2_S" &&
                   routed_case.routed.down == "IQ4_XS";
        });
    ASSERT_NE(selected, cases.end());

    ScopedEnvOverride rowwise_iterations(
        "LLAMINAR_MOE_VERIFIER_PREFILL_ROWWISE_ITERS", "1");
    ScopedEnvOverride perfstats("LLAMINAR_PERF_STATS_JSON", "1");
    proveCudaProductionCandidateInvariance(
        *selected,
        /*rows=*/512,
        /*device_ordinal=*/0,
        llaminar2::test::native_vnni_dispatch::MoERoutingProfile::PowerLaw,
        *candidate);
#endif
}

TEST(Perf__MoEVerifierPrefill, CUDA_M1234_RoutedExpertFFNDecodeEquivalent)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    for (int rows : selectedVerifierRows({1, 2, 3, 4}))
    {
        auto routed = runCudaCase(/*shared=*/false, rows);
        expectBitwiseEqual(routed.metrics);
        if (rows >= 2)
            expectGraphReplayFasterThanReference(routed);
        printResult(routed);
    }
#endif
}

TEST(Perf__MoEVerifierPrefill, CUDA_M1234_SharedExpertFFNDecodeEquivalent)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    for (int rows : selectedVerifierRows({1, 2, 3, 4}))
    {
        auto shared = runCudaCase(/*shared=*/true, rows);
        expectBitwiseEqual(shared.metrics);
        if (rows >= 2)
            expectSharedExpertFfnEconomical(shared);
        printResult(shared);
    }
#endif
}

TEST(Perf__MoEVerifierPrefill, CUDA_M234_SharedExpertFFNStageDecodeEquivalent)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    const QuantFormatCase &format = sharedExpertPreparedFormatCase("IQ3_S");
    for (int rows : selectedVerifierRows({2, 3, 4}))
    {
        auto shared = runCudaSharedExpertStageCase(rows, format);
        expectBitwiseEqual(shared.metrics);
        expectSharedExpertFfnEconomical(shared);
        printResult(shared);
    }
#endif
}

TEST(Perf__MoEVerifierPrefill, CUDA_M234_SharedExpertFFNStageAllCodebooksDecodeEquivalent)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    ScopedEnvOverride iters_env("LLAMINAR_MOE_VERIFIER_PREFILL_ITERS", "12");
    ScopedEnvOverride warmups_env("LLAMINAR_MOE_VERIFIER_PREFILL_WARMUPS", "2");
    for (const QuantFormatCase &format : sharedExpertPreparedFormatCases())
    {
        if (!envCsvContainsOrUnset("LLAMINAR_MOE_VERIFIER_PREFILL_FORMATS", format.name))
            continue;
        SCOPED_TRACE(format.name);
        for (int rows : selectedVerifierRows({2, 3, 4}))
        {
            SCOPED_TRACE(rows);
            auto shared = runCudaSharedExpertStageCase(rows, format);
            expectBitwiseEqual(shared.metrics);
            printResult(shared);
        }
    }
#endif
}

TEST(Perf__MoEVerifierPrefill, CUDA_M4_CombinedRoutedSharedUpperBound)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    /*
     * Upper-bound experiment for the next production refactor: treat routed
     * top-8 plus the shared expert as one top-9 grouped-prefill batch with an
     * extra descriptor entry.  This does not model the shared gate weight, but
     * it proves whether one combined grouped expert launch is materially
     * cheaper than today's routed+shared separate verifier pipelines.
     */
    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    auto combined = runCudaCase(
        /*shared=*/false,
        /*rows=*/4,
        /*routed_top_k=*/9,
        /*routed_num_experts=*/257,
        /*case_name_override=*/"combined_top9_upper_bound",
        /*unique_routes=*/true,
        /*include_terminal_expert=*/true);
    expectBitwiseEqual(combined.metrics);
    expectGraphReplayFasterThanReference(combined);
    printResult(combined);
#endif
}

TEST(Perf__MoEVerifierPrefill, CUDA_M4M9M31_CombinedTop9AllFormatsDecodeEquivalent)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    /*
     * This is a capacity and arithmetic regression, not a timing campaign.
     * One replay per cell keeps the canonical integration gate compact while
     * still entering the production graph-captured pipeline. M=4 covers the
     * direct verifier policy; M=9 and M=31 prove that runtime depth remains
     * independent of the historical M=2..4 assumption.
     */
    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    ScopedEnvOverride iters_env("LLAMINAR_MOE_VERIFIER_PREFILL_ITERS", "1");
    ScopedEnvOverride warmups_env("LLAMINAR_MOE_VERIFIER_PREFILL_WARMUPS", "0");
    ScopedEnvOverride rowwise_env(
        "LLAMINAR_MOE_VERIFIER_PREFILL_ROWWISE_ITERS", "1");
    for (const auto &format : llaminar2::test::quantizedVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        for (const int rows : selectedVerifierRows({4, 9, 31}))
        {
            SCOPED_TRACE(rows);
            auto combined = runCudaCase(
                /*shared=*/false,
                rows,
                /*routed_top_k=*/9,
                /*routed_num_experts=*/257,
                /*case_name_override=*/"combined_top9_all_formats",
                /*unique_routes=*/false,
                /*include_terminal_expert=*/true,
                &format,
                &format);
            expectBitwiseEqual(combined.metrics);
            expectGraphReplayFasterThanReference(combined);
        }
    }
#endif
}
