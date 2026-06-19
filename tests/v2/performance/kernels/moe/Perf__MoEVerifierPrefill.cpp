#include <gtest/gtest.h>

#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"

#include "../../../mocks/MockComputeStage.h"
#include "../../../utils/GpuPreparedGemmHarness.h"
#include "../../../utils/TestTensorFactory.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

/**
 * @file Perf__MoEVerifierPrefill.cpp
 * @brief Focused MoE verifier-prefill parity and timing harness.
 *
 * This target isolates the Qwen3.6 MoE MTP verifier hot path: small verifier
 * batches with M=2,3,4, routed top-k experts, and the always-on shared expert.
 * Each case compares grouped verifier prefill against row-wise decode-equivalent
 * execution and then reports eager and graph-replay timing in a compact CSV row.
 *
 * The harness is deliberately narrower than full-model benchmark mode. It gives
 * us a stable speedometer for kernel and grouping changes before we spend time
 * rerunning the expensive dense/MoE iteration matrix.
 */

namespace
{
    using KernelFactory = llaminar::v2::kernels::KernelFactory;

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

    void expectClose(const CloseMetrics &metrics)
    {
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
                   "eager_ms,graph_ms,rowwise_ms,speedup_vs_reference,"
                   "cosine,relative_l2,max_abs,"
                   "min_row_cosine,max_row_relative_l2,max_row_kl,"
                   "nonfinite_count,nonfinite_actual_count,nonfinite_expected_count,"
                   "first_nonfinite_index,worst_row\n";
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
        std::vector<int> materialized_experts)
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

        auto add_desc = [&](int rows, int cols, int seed, const char *role, int expected_codebook)
        {
            std::unique_ptr<llaminar2::TensorBase> weight;
            if (expected_codebook == 13)
            {
                weight = llaminar2::test::TestTensorFactory::createIQ2_SRandom(
                    {static_cast<size_t>(rows), static_cast<size_t>(cols)},
                    static_cast<unsigned>(seed));
            }
            else
            {
                weight = llaminar2::test::TestTensorFactory::createIQ4_XSRandom(
                    {static_cast<size_t>(rows), static_cast<size_t>(cols)},
                    static_cast<unsigned>(seed));
            }

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
            EXPECT_EQ(desc.codebook_id, expected_codebook);
            return desc;
        };

        for (int expert : materialized_experts)
        {
            tables.gate_descs[static_cast<size_t>(expert)] =
                add_desc(intermediate, d_model, 4100 + expert, "gate", 13);
            tables.up_descs[static_cast<size_t>(expert)] =
                add_desc(intermediate, d_model, 4200 + expert, "up", 13);
            tables.down_descs[static_cast<size_t>(expert)] =
                add_desc(d_model, intermediate, 4300 + expert, "down", 4);
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
    bool hasCudaDevice()
    {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
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

    class ScopedCudaMoEGemmConfig
    {
    public:
        ScopedCudaMoEGemmConfig()
            : old_gateup_kpart_decode_(llaminar2::mutableDebugEnv().gemm.cuda_moe_gateup_kpart_decode),
              old_gateup_kparts_(llaminar2::mutableDebugEnv().gemm.cuda_moe_gateup_kparts),
              old_down_kpart_decode_(llaminar2::mutableDebugEnv().gemm.cuda_moe_down_kpart_decode),
              old_down_kparts_(llaminar2::mutableDebugEnv().gemm.cuda_moe_down_kparts)
        {
        }

        ~ScopedCudaMoEGemmConfig()
        {
            auto &gemm = llaminar2::mutableDebugEnv().gemm;
            gemm.cuda_moe_gateup_kpart_decode = old_gateup_kpart_decode_;
            gemm.cuda_moe_gateup_kparts = old_gateup_kparts_;
            gemm.cuda_moe_down_kpart_decode = old_down_kpart_decode_;
            gemm.cuda_moe_down_kparts = old_down_kparts_;
        }

        void set(bool gateup_kpart, int gateup_kparts, bool down_kpart, int down_kparts)
        {
            auto &gemm = llaminar2::mutableDebugEnv().gemm;
            gemm.cuda_moe_gateup_kpart_decode = gateup_kpart;
            gemm.cuda_moe_gateup_kparts = gateup_kparts;
            gemm.cuda_moe_down_kpart_decode = down_kpart;
            gemm.cuda_moe_down_kparts = down_kparts;
        }

    private:
        bool old_gateup_kpart_decode_ = true;
        int old_gateup_kparts_ = 16;
        bool old_down_kpart_decode_ = true;
        int old_down_kparts_ = 16;
    };

    class CudaGraphOwner
    {
    public:
        ~CudaGraphOwner()
        {
            if (exec_)
                cudaGraphExecDestroy(exec_);
            if (graph_)
                cudaGraphDestroy(graph_);
        }

        cudaGraph_t *graphPtr() { return &graph_; }
        cudaGraphExec_t *execPtr() { return &exec_; }
        cudaGraphExec_t execHandle() const { return exec_; }

    private:
        cudaGraph_t graph_ = nullptr;
        cudaGraphExec_t exec_ = nullptr;
    };

    double timeCudaEvents(cudaStream_t stream, int iterations, const std::function<bool()> &body)
    {
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        EXPECT_EQ(cudaEventCreate(&start), cudaSuccess);
        EXPECT_EQ(cudaEventCreate(&stop), cudaSuccess);
        EXPECT_EQ(cudaEventRecord(start, stream), cudaSuccess);
        for (int i = 0; i < iterations; ++i)
            EXPECT_TRUE(body());
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

        const auto device = llaminar2::DeviceId::cuda(0);
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
                decode_output->transitionTo(llaminar2::TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
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

    BenchResult runCudaCase(
        bool shared,
        int rows,
        int routed_top_k = 8,
        int routed_num_experts = 256,
        const char *case_name_override = nullptr,
        bool unique_routes = false)
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
        const int iterations = envInt("LLAMINAR_MOE_VERIFIER_PREFILL_ITERS", 30);
        const int warmups = envInt("LLAMINAR_MOE_VERIFIER_PREFILL_WARMUPS", 5);
        const auto device = llaminar2::DeviceId::cuda(0);

        EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
        cudaStream_t stream = nullptr;
        EXPECT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

        auto *moe = KernelFactory::getOrCreateMoEKernel(device);
        EXPECT_NE(moe, nullptr);
        moe->setGPUStream(stream);
        auto *workspace_consumer = dynamic_cast<llaminar2::IWorkspaceConsumer *>(moe);
        EXPECT_NE(workspace_consumer, nullptr);
        auto reqs = llaminar2::MoEWorkspaceBuffers::cudaMoE(
            /*max_seq_len=*/4,
            d_model,
            intermediate,
            num_experts,
            top_k);
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
        ScopedCudaMoEGemmConfig gemm_config;
        const int gateup_kparts =
            envInt("LLAMINAR_MOE_VERIFIER_PREFILL_CUDA_GATEUP_KPARTS",
                   llaminar2::debugEnv().gemm.cuda_moe_gateup_kparts);
        const int down_kparts =
            envInt("LLAMINAR_MOE_VERIFIER_PREFILL_CUDA_DOWN_KPARTS",
                   llaminar2::debugEnv().gemm.cuda_moe_down_kparts);
        /*
         * Production defaults remain in DebugEnv. These optional harness
         * overrides let the verifier-prefill speedometer sweep K-partitioning
         * policy without source edits, then promote only proven policies.
         */
        gemm_config.set(/*gateup_kpart=*/true, gateup_kparts,
                        /*down_kpart=*/true, down_kparts);

        const auto hidden_values = makeHiddenValues(rows, d_model);
        const auto routing_indices = unique_routes
                                         ? makeUniqueRoutingIndices(rows, top_k, num_experts)
                                         : makeRoutingIndices(rows, top_k, num_experts);
        const auto routing_weights = makeRoutingWeights(rows, top_k);
        auto tables = prepareExpertTables(
            moe, device, num_experts, d_model, intermediate, "cuda", 270000,
            uniqueExpertIdsFromRoutes(routing_indices, num_experts));
        auto hidden = makeTensor({static_cast<size_t>(rows), static_cast<size_t>(d_model)}, hidden_values);
        auto route_indices_tensor = makeTensor({static_cast<size_t>(rows), static_cast<size_t>(top_k)}, routing_indices);
        auto route_weights_tensor = makeTensor({static_cast<size_t>(rows), static_cast<size_t>(top_k)}, routing_weights);
        auto grouped_output = makeZeros({static_cast<size_t>(rows), static_cast<size_t>(d_model)});
        EXPECT_TRUE(hidden->ensureOnDevice(device, stream));
        EXPECT_TRUE(route_indices_tensor->ensureOnDevice(device, stream));
        EXPECT_TRUE(route_weights_tensor->ensureOnDevice(device, stream));
        EXPECT_TRUE(grouped_output->ensureOnDevice(device, stream));

        auto run_grouped = [&]()
        {
            if (shared)
            {
                if (!moe->prepareSharedExpertPrefillGroup(rows))
                    return false;
            }
            else if (!moe->prepareExpertGroupsAsync(
                         route_indices_tensor.get(), route_weights_tensor.get(),
                         rows, num_experts, top_k))
            {
                return false;
            }
            return moe->executeGroupedPrefillPipeline(
                hidden.get(), grouped_output.get(),
                tables.gateup_table_id, tables.down_table_id,
                rows, d_model, intermediate, num_experts, top_k);
        };

        for (int i = 0; i < warmups; ++i)
            EXPECT_TRUE(run_grouped());
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        const double eager_ms = timeCudaEvents(stream, iterations, run_grouped);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        CudaGraphOwner graph;
        EXPECT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), cudaSuccess);
        const bool captured = run_grouped();
        const cudaError_t end_status = cudaStreamEndCapture(stream, graph.graphPtr());
        EXPECT_TRUE(captured);
        EXPECT_EQ(end_status, cudaSuccess) << cudaGetErrorString(end_status);
        EXPECT_NE(*graph.graphPtr(), nullptr);
        EXPECT_EQ(cudaGraphInstantiate(graph.execPtr(), *graph.graphPtr(), nullptr, nullptr, 0), cudaSuccess);
        for (int i = 0; i < warmups; ++i)
            EXPECT_EQ(cudaGraphLaunch(graph.execHandle(), stream), cudaSuccess);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        const double graph_ms = timeCudaEvents(
            stream,
            iterations,
            [&]()
            {
                return cudaGraphLaunch(graph.execHandle(), stream) == cudaSuccess;
            });
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        grouped_output->transitionTo(llaminar2::TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        std::vector<float> grouped(
            grouped_output->data(),
            grouped_output->data() + grouped_output->numel());

        double rowwise_ms = 0.0;
        std::vector<float> rowwise = runCudaRowwiseDecode(
            moe, workspace.get(), stream, hidden_values, routing_indices, routing_weights,
            rows, top_k, d_model, intermediate,
            tables.gateup_table_id, tables.down_table_id, &rowwise_ms);
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
        auto serial_output = makeZeros({static_cast<size_t>(rows), static_cast<size_t>(d_model)});
        EXPECT_TRUE(hidden->ensureOnDevice(device, stream));
        EXPECT_TRUE(grouped_output->ensureOnDevice(device, stream));
        EXPECT_TRUE(serial_output->ensureOnDevice(device, stream));

        auto make_params = [&](llaminar2::TensorBase *output,
                               bool grouped_verifier,
                               bool serial_decode_equivalent)
        {
            llaminar2::SharedExpertFFNStage::Params params;
            params.device_id = device;
            params.input = hidden.get();
            params.gate_w = gate_w.get();
            params.up_w = up_w.get();
            params.down_w = down_w.get();
            params.output = output;
            params.seq_len = rows;
            params.d_model = d_model;
            params.intermediate = intermediate;
            params.prepared_ref_gate = prepared.gate_ref;
            params.prepared_ref_up = prepared.up_ref;
            params.prepared_ref_down = prepared.down_ref;
            params.prepared_store = prepared.store.get();
            params.force_grouped_verifier_prefill_for_decode = grouped_verifier;
            params.force_decode_equivalent_verifier_prefill = serial_decode_equivalent;
            return params;
        };

        llaminar2::SharedExpertFFNStage grouped_stage(
            make_params(grouped_output.get(), /*grouped_verifier=*/true,
                        /*serial_decode_equivalent=*/false));
        llaminar2::SharedExpertFFNStage serial_stage(
            make_params(serial_output.get(), /*grouped_verifier=*/false,
                        /*serial_decode_equivalent=*/true));
        grouped_stage.setGPUStream(stream);
        serial_stage.setGPUStream(stream);
        EXPECT_TRUE(grouped_stage.usesGroupedVerifierPrefillRouteForTesting());
        EXPECT_TRUE(serial_stage.usesCPUDecodeEquivalentVerifierPrefillForTesting());

        auto reqs = grouped_stage.getWorkspaceRequirements(rows, d_model, intermediate);
        auto workspace = std::make_unique<llaminar2::DeviceWorkspaceManager>(
            device,
            reqs.total_bytes_with_alignment() + 8 * 1024 * 1024);
        EXPECT_TRUE(workspace->allocate(reqs));
        grouped_stage.bindWorkspace(workspace.get());
        serial_stage.bindWorkspace(workspace.get());

        llaminar2::testing::MockDeviceContext ctx(
            device, llaminar2::ComputeBackendType::GPU_CUDA);
        auto run_grouped = [&]()
        {
            return grouped_stage.execute(&ctx);
        };
        auto run_serial = [&]()
        {
            return serial_stage.execute(&ctx);
        };

        for (int i = 0; i < warmups; ++i)
            EXPECT_TRUE(run_grouped());
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_TRUE(grouped_stage.isGraphCapturable());

        const double eager_ms = timeCudaEvents(stream, iterations, run_grouped);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        CudaGraphOwner graph;
        EXPECT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), cudaSuccess);
        const bool captured = run_grouped();
        const cudaError_t end_status = cudaStreamEndCapture(stream, graph.graphPtr());
        EXPECT_TRUE(captured);
        EXPECT_EQ(end_status, cudaSuccess) << cudaGetErrorString(end_status);
        EXPECT_NE(*graph.graphPtr(), nullptr);
        EXPECT_EQ(cudaGraphInstantiate(graph.execPtr(), *graph.graphPtr(), nullptr, nullptr, 0), cudaSuccess);
        for (int i = 0; i < warmups; ++i)
            EXPECT_EQ(cudaGraphLaunch(graph.execHandle(), stream), cudaSuccess);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        const double graph_ms = timeCudaEvents(
            stream,
            iterations,
            [&]()
            {
                return cudaGraphLaunch(graph.execHandle(), stream) == cudaSuccess;
            });
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        const double serial_ms = timeCudaEvents(stream, std::max(1, iterations / 4), run_serial);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        grouped_output->transitionTo(llaminar2::TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        serial_output->transitionTo(llaminar2::TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        std::vector<float> grouped(
            grouped_output->data(),
            grouped_output->data() + grouped_output->numel());
        std::vector<float> serial(
            serial_output->data(),
            serial_output->data() + serial_output->numel());
        CloseMetrics metrics = compareVectors(grouped, serial, static_cast<size_t>(d_model));

        grouped_stage.unbindWorkspace();
        serial_stage.unbindWorkspace();
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
            graph_ms,
            serial_ms,
            metrics};
    }
}
#endif

TEST(Perf__MoEVerifierPrefill, CUDA_M1234_RoutedExpertFFNDecodeEquivalent)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    for (int rows : {1, 2, 3, 4})
    {
        auto routed = runCudaCase(/*shared=*/false, rows);
        expectClose(routed.metrics);
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
    for (int rows : {1, 2, 3, 4})
    {
        auto shared = runCudaCase(/*shared=*/true, rows);
        expectClose(shared.metrics);
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
    for (int rows : {2, 3, 4})
    {
        auto shared = runCudaSharedExpertStageCase(rows, format);
        expectClose(shared.metrics);
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
        for (int rows : {2, 3, 4})
        {
            SCOPED_TRACE(rows);
            auto shared = runCudaSharedExpertStageCase(rows, format);
            expectClose(shared.metrics);
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
        /*unique_routes=*/true);
    expectClose(combined.metrics);
    expectGraphReplayFasterThanReference(combined);
    printResult(combined);
#endif
}
