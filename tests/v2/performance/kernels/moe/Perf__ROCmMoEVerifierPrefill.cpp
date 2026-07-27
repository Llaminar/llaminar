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

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include "kernels/rocm/moe/ROCmMoEKernel.h"

extern "C" bool rocmMoE_grouped_prefill_query_tile_config(
    uint8_t codebook_id,
    int projection_role,
    int m,
    int n,
    int k,
    int *tile_m,
    int *tile_n);
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
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <stdexcept>
#include <vector>

/**
 * @file Perf__ROCmMoEVerifierPrefill.cpp
 * @brief ROCm MoE verifier and batch-invariant prefill speedometer.
 *
 * CUDA and HIP runtime headers cannot be included in the same translation unit
 * in heterogeneous builds because both define vector types such as `dim3`. This
 * file mirrors the CUDA harness with ROCm-only timing and graph-capture calls.
 * In addition to M=1..4 verifier buckets, it measures production M=8/32/256
 * router and routed-expert paths against serial M=1 oracles. Every measured
 * result must remain native-bit identical before its economy result is accepted.
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
        size_t worst_row = 0;
        size_t bit_mismatch_count = 0;
        size_t first_bit_mismatch_index = 0;
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
            llaminar2::mutableDebugEnv().rocm.reload();
        }

        ~ScopedEnvOverride()
        {
            if (had_old_)
                setenv(name_.c_str(), old_.c_str(), 1);
            else
                unsetenv(name_.c_str());
            llaminar2::mutableDebugEnv().rocm.reload();
        }

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

    std::vector<int> envCsvInts(
        const char *name,
        std::initializer_list<int> defaults)
    {
        const char *value = std::getenv(name);
        if (!value || !*value)
            return std::vector<int>(defaults);

        std::vector<int> parsed;
        std::stringstream stream(value);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            const int number = std::atoi(token.c_str());
            if (number > 0)
                parsed.push_back(number);
        }
        return parsed.empty() ? std::vector<int>(defaults) : parsed;
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
                values[static_cast<size_t>(row) * top_k + k] =
                    static_cast<float>((k + ((row & 1) ? 4 : 0)) % num_experts);
        }
        return values;
    }

    /**
     * @brief Build a route table that maximizes active expert slots.
     *
     * The normal route fixture intentionally reuses experts across rows because
     * that resembles many real prompts.  The combined routed+shared verifier
     * experiment needs the opposite pressure: top-8 routed experts plus one
     * shared expert represented as a top-9 table.  Unique routes make the active
     * slot count deterministic and comparable to the CUDA speedometer.
     */
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
     * Qwen 3.6 exposes 256 routed experts plus a shared expert. A fused
     * descriptor table therefore has 257 entries, with shared expert id 256.
     * Allocating that table alone is a weak test because ordinary routed
     * patterns never select its last descriptor. This helper rewrites the final
     * route of every verifier row so grouped gate/up and down must dereference
     * the 257th entry on the real device path.
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
     * The speedometer keeps production-sized descriptor tables but should not
     * spend minutes preparing inactive expert payloads before the timed GPU work.
     * Active slots still get real prepared descriptors and therefore hard-fail on
     * unsupported formats or broken prepared-weight wiring.
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
     * @brief Normalize the explicit expert materialization set.
     *
     * Inactive descriptor slots are filled by aliasing a valid routed descriptor
     * after materialization. They remain valid table entries, but the synthetic
     * route tensors never select them.
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
     * This performance harness doubles as a drift detector for verifier-sized
     * MoE rows.  Row-wise KL gives us a sharper signal than aggregate cosine
     * when a fast path changes the dominant coordinates of one accepted row.
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
            if (std::bit_cast<uint32_t>(actual[i]) !=
                std::bit_cast<uint32_t>(expected[i]))
            {
                if (metrics.bit_mismatch_count == 0)
                    metrics.first_bit_mismatch_index = i;
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

    void expectClose(const CloseMetrics &metrics)
    {
        EXPECT_EQ(metrics.bit_mismatch_count, 0u)
            << "ROCm grouped prefill must be byte-identical to serial decode; "
            << "first_bit_mismatch_index=" << metrics.first_bit_mismatch_index;
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

    /**
     * @brief Report useful routed-expert INT8 work per timed pipeline second.
     *
     * Gate, up, and down each execute `M * top_k * d_model * intermediate`
     * multiply-accumulates. Counting a multiply and add as two integer
     * operations gives six operations per matrix element across the three
     * projections. The metric intentionally divides by the complete grouped
     * pipeline time, so routing-row quantization, SwiGLU quantization, directory
     * planning, partial publication, and launch overhead all reduce the score.
     */
    double routedExpertPipelineGops(const BenchResult &result)
    {
        if (result.pipeline_ms <= 0.0)
            return 0.0;
        const double operations =
            6.0 * static_cast<double>(result.m) * static_cast<double>(result.top_k) *
            static_cast<double>(result.d_model) * static_cast<double>(result.intermediate);
        return operations / (result.pipeline_ms * 1.0e6);
    }

    void printResult(const BenchResult &result)
    {
        static bool printed_header = false;
        if (!printed_header)
        {
            std::cout
                << "backend,case,m,top_k,num_experts,d_model,intermediate,"
                   "eager_ms,prepare_ms,pipeline_ms,pipeline_int8_gops,"
                   "graph_ms,rowwise_ms,speedup_vs_reference,"
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
                  << result.prepare_ms << ','
                  << result.pipeline_ms << ','
                  << routedExpertPipelineGops(result) << ','
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
     * @brief Assert that grouped graph replay beats its decode-equivalent oracle.
     *
     * The reference time is deliberately conservative: row-wise decode for
     * isolated routed/shared rows, and split routed+shared verifier prefill for
     * the production combined-shared case.  This makes the performance test a
     * Phase 9.8 guard: grouped verifier rows must be both numerically strict
     * and cheaper than the serial/equivalent path they replace.
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
     * The shared expert path is tempting to fold into surrounding MoE work, but
     * previous routed+shared fusion attempts changed full-model continuation
     * math.  Keep this focused gate separate: any future ROCm shared-expert FFN
     * kernel must beat the serial decode-equivalent reference by a useful
     * margin and pass strict cosine/L2/KL/max-abs checks before graph wiring.
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

    /**
     * @brief Prepare production-style native-VNNI expert descriptors.
     *
     * The returned owner must stay alive because every descriptor points into
     * VRAM owned by the corresponding prepared GEMM handle.
     */
    PreparedExpertTables prepareExpertTables(
        llaminar2::IMoEKernel *moe,
        llaminar2::DeviceId device,
        int num_experts,
        int d_model,
        int intermediate,
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
                "perf.moe_verifier.rocm." + std::string(format.label) + "." +
                    role + "." + std::to_string(seed),
                llaminar2::ModelContextId{280000 + static_cast<uint64_t>(seed)}));

            llaminar2::DeviceNativeVNNIMatrixDesc desc{};
            EXPECT_TRUE(tables.prepared.back().kernel->exportNativeVNNIMatrixDesc(desc));
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

#ifdef HAVE_ROCM
namespace
{
    bool hasROCmDevice()
    {
        int count = 0;
        return hipGetDeviceCount(&count) == hipSuccess && count > 0;
    }

    bool isGfx906Device()
    {
        hipDeviceProp_t properties{};
        if (hipGetDeviceProperties(&properties, 0) != hipSuccess)
            return false;
        return std::string(properties.gcnArchName).find("gfx906") == 0;
    }

    /**
     * @brief Enforce useful long-prefill throughput on the MI50/MI60 target.
     *
     * The best dense gfx906 INT8 kernels in this repository sustain roughly
     * 12.7 TOPS on larger FFN shapes. A routed pipeline also pays device grouping,
     * row quantization, SwiGLU, and deterministic publication costs, so the
     * acceptance floors are deliberately below that dense ceiling but far above
     * merely beating serial decode. These bounds reject the register-spilling
     * 24-row experiment and the old per-route weight-decode implementation.
     */
    void expectGfx906LongPrefillThroughput(const BenchResult &result)
    {
        if (!isGfx906Device())
            return;

        const double gops = routedExpertPipelineGops(result);
        if (result.m >= 256)
        {
            EXPECT_GE(gops, 9000.0)
                << "gfx906 M=256 grouped routed prefill must sustain at least 9.0 TOPS";
        }
        else if (result.m >= 32)
        {
            EXPECT_GE(gops, 3500.0)
                << "gfx906 M=32 grouped routed prefill must sustain at least 3.5 TOPS";
        }
    }

    class HipGraphOwner
    {
    public:
        ~HipGraphOwner()
        {
            if (exec_)
                (void)hipGraphExecDestroy(exec_);
            if (graph_)
                (void)hipGraphDestroy(graph_);
        }

        hipGraph_t *graphPtr() { return &graph_; }
        hipGraphExec_t *execPtr() { return &exec_; }
        hipGraphExec_t execHandle() const { return exec_; }

    private:
        hipGraph_t graph_ = nullptr;
        hipGraphExec_t exec_ = nullptr;
    };

    /**
     * @brief Stop a timing cell immediately when its launch body is rejected.
     *
     * Continuing after a false grouped-pipeline return records an empty event
     * interval and emits one assertion per iteration. A thrown test failure
     * identifies the first broken phase and ensures failed launches can never
     * appear as high-throughput benchmark winners.
     */
    void requireHipBenchBody(bool ok, const char *phase)
    {
        if (!ok)
        {
            throw std::runtime_error(
                std::string("ROCm MoE verifier benchmark body failed during ") +
                (phase ? phase : "unknown phase"));
        }
    }

    double timeHipEvents(hipStream_t stream, int iterations, const std::function<bool()> &body)
    {
        hipEvent_t start = nullptr;
        hipEvent_t stop = nullptr;
        EXPECT_EQ(hipEventCreate(&start), hipSuccess);
        EXPECT_EQ(hipEventCreate(&stop), hipSuccess);
        EXPECT_EQ(hipEventRecord(start, stream), hipSuccess);
        for (int i = 0; i < iterations; ++i)
            requireHipBenchBody(body(), "timed replay");
        EXPECT_EQ(hipEventRecord(stop, stream), hipSuccess);
        EXPECT_EQ(hipEventSynchronize(stop), hipSuccess);
        float ms = 0.0f;
        EXPECT_EQ(hipEventElapsedTime(&ms, start, stop), hipSuccess);
        EXPECT_EQ(hipEventDestroy(start), hipSuccess);
        EXPECT_EQ(hipEventDestroy(stop), hipSuccess);
        return static_cast<double>(ms) / static_cast<double>(iterations);
    }

    std::vector<float> runRowwiseDecode(
        llaminar2::IMoEKernel *moe,
        hipStream_t stream,
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
        const auto device = llaminar2::DeviceId::rocm(0);
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
                EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
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

    BenchResult runROCmCase(
        bool shared,
        int rows,
        int routed_top_k = 8,
        int routed_num_experts = 256,
        const char *case_name_override = nullptr,
        bool unique_routes = false,
        bool include_terminal_expert = false,
        int d_model = 2048,
        int intermediate = 512,
        const llaminar2::test::QuantizedVerifierFormatCase *gateup_format = nullptr,
        const llaminar2::test::QuantizedVerifierFormatCase *down_format = nullptr)
    {
        /*
         * Keep this harness aligned with the Qwen3.6 MoE model shape.  The
         * actual benchmark matrix routes across 256 experts, so the focused
         * speedometer should pay the same descriptor-table and grouping setup
         * cost instead of testing only a small proxy table.
         */
        constexpr int shared_top_k = 1;
        constexpr int shared_num_experts = 1;
        const int top_k = shared ? shared_top_k : routed_top_k;
        const int num_experts = shared ? shared_num_experts : routed_num_experts;
        const auto &selected_gateup_format = gateup_format
                                                 ? *gateup_format
                                                 : llaminar2::test::quantizedVerifierFormat("IQ2_S");
        const auto &selected_down_format = down_format
                                               ? *down_format
                                               : llaminar2::test::quantizedVerifierFormat("IQ4_XS");
        /*
         * This mirrors the CUDA production speedometer: the combined path and
         * split reference are both sub-millisecond graph-captured paths, so a
         * robust default timing window is required for CTest to reject only
         * real verifier-economy regressions.  Sweeps may still override it
         * with LLAMINAR_MOE_VERIFIER_PREFILL_ITERS.
         */
        const int default_iterations = rows >= 256 ? 8 : (rows >= 32 ? 24 : 120);
        const int default_warmups = rows >= 32 ? 2 : 5;
        const int iterations = envInt(
            "LLAMINAR_MOE_VERIFIER_PREFILL_ITERS", default_iterations);
        const int warmups = envInt(
            "LLAMINAR_MOE_VERIFIER_PREFILL_WARMUPS", default_warmups);
        const auto device = llaminar2::DeviceId::rocm(0);

        EXPECT_EQ(hipSetDevice(0), hipSuccess);
        hipStream_t stream = nullptr;
        EXPECT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

        auto moe = KernelFactory::createMoEKernel(device);
        EXPECT_NE(moe, nullptr);
        moe->setGPUStream(stream);
        auto *workspace_consumer = dynamic_cast<llaminar2::IWorkspaceConsumer *>(moe.get());
        EXPECT_NE(workspace_consumer, nullptr);
        const int workspace_num_experts = std::max(num_experts, routed_num_experts);
        const int workspace_top_k = std::max(top_k, routed_top_k);
        auto reqs = llaminar2::MoEWorkspaceBuffers::rocmMoE(
            /*max_seq_len=*/rows,
            d_model,
            intermediate,
            workspace_num_experts,
            workspace_top_k);
        auto workspace = std::make_unique<llaminar2::DeviceWorkspaceManager>(
            device,
            reqs.total_bytes_with_alignment() + 8 * 1024 * 1024);
        EXPECT_TRUE(workspace->allocate(reqs));
        workspace_consumer->bindWorkspace(workspace.get());

        const auto hidden_values = makeHiddenValues(rows, d_model);
        auto routing_indices = unique_routes
                                   ? makeUniqueRoutingIndices(rows, top_k, num_experts)
                                   : makeRoutingIndices(rows, top_k, num_experts);
        if (include_terminal_expert)
            publishTerminalExpertRoute(routing_indices, rows, top_k, num_experts);
        const auto routing_weights = makeRoutingWeights(rows, top_k);
        auto tables = prepareExpertTables(
            moe.get(), device, num_experts, d_model, intermediate,
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
         * @brief Run only the device-resident grouping half of the verifier path.
         *
         * Timing this separately keeps the performance proof honest: a slow
         * grouped verifier can be bad because of route grouping or because of
         * GEMV/scatter kernels, and those fixes live in different places.
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
            requireHipBenchBody(run_grouped(), "warmup");
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

        const double eager_ms = timeHipEvents(stream, iterations, run_grouped);
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        const double prepare_ms = timeHipEvents(stream, iterations, run_prepare);
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        requireHipBenchBody(run_prepare(), "pipeline preparation");
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        const double pipeline_ms = timeHipEvents(stream, iterations, run_pipeline);
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

        HipGraphOwner graph;
        const hipError_t begin_status =
            hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
        if (begin_status != hipSuccess)
        {
            throw std::runtime_error(
                std::string("ROCm MoE verifier graph capture begin failed: ") +
                hipGetErrorString(begin_status));
        }
        const bool captured = run_grouped();
        const hipError_t end_status = hipStreamEndCapture(stream, graph.graphPtr());
        if (!captured || end_status != hipSuccess || *graph.graphPtr() == nullptr)
        {
            throw std::runtime_error(
                std::string("ROCm MoE verifier graph capture body failed: ") +
                hipGetErrorString(end_status));
        }
        const hipError_t instantiate_status =
            hipGraphInstantiate(
                graph.execPtr(), *graph.graphPtr(), nullptr, nullptr, 0);
        if (instantiate_status != hipSuccess)
        {
            throw std::runtime_error(
                std::string("ROCm MoE verifier graph instantiate failed: ") +
                hipGetErrorString(instantiate_status));
        }
        for (int i = 0; i < warmups; ++i)
            EXPECT_EQ(hipGraphLaunch(graph.execHandle(), stream), hipSuccess);
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        const double graph_ms = timeHipEvents(
            stream,
            iterations,
            [&]()
            {
                return hipGraphLaunch(graph.execHandle(), stream) == hipSuccess;
            });
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

        TransferEngine::publishDeviceWrite(grouped_output, device, stream);
        std::vector<float> grouped(
            grouped_output->data(),
            grouped_output->data() + grouped_output->numel());

        double rowwise_ms = 0.0;
        std::vector<float> rowwise = runRowwiseDecode(
            moe.get(), stream, hidden_values, routing_indices, routing_weights,
            rows, top_k, d_model, intermediate,
            tables.gateup_table_id, tables.down_table_id, &rowwise_ms);
        CloseMetrics metrics = compareVectors(grouped, rowwise, static_cast<size_t>(d_model));

        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);

        return BenchResult{
            "rocm",
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
     * The lower-level IMoE shared-prefill path is useful for kernel isolation,
     * but the graph builder wires `SharedExpertFFNStage`.  This harness compares
     * the stage's grouped M=2..4 dense-FFN route against the same stage's
     * strict serial decode-equivalent replay, then times graph replay for the
     * grouped route.  That is the acceptance shape needed before promoting any
     * shared-expert FFN kernel in production.
     */
    BenchResult runROCmSharedExpertStageCase(int rows, const QuantFormatCase &format)
    {
        constexpr int d_model = 2048;
        constexpr int intermediate = 512;
        const int iterations = envInt("LLAMINAR_MOE_VERIFIER_PREFILL_ITERS", 120);
        const int warmups = envInt("LLAMINAR_MOE_VERIFIER_PREFILL_WARMUPS", 5);
        const auto device = llaminar2::DeviceId::rocm(0);

        EXPECT_EQ(hipSetDevice(0), hipSuccess);
        hipStream_t stream = nullptr;
        EXPECT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

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
            std::string("perf.moe_verifier.rocm.shared_stage.") + format.name,
            llaminar2::ModelContextId{390000});

        const auto hidden_values = makeHiddenValues(rows, d_model);
        auto hidden = makeTensor({static_cast<size_t>(rows), static_cast<size_t>(d_model)}, hidden_values);
        auto grouped_output = makeZeros({static_cast<size_t>(rows), static_cast<size_t>(d_model)});
        EXPECT_TRUE(hidden->ensureOnDevice(device, stream));
        EXPECT_TRUE(grouped_output->ensureOnDevice(device, stream));

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
        reqs.merge(llaminar2::MoEWorkspaceBuffers::rocmMoE(
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
            device, llaminar2::ComputeBackendType::GPU_ROCM);
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
                EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
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
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        EXPECT_TRUE(grouped_stage.isGraphCapturable());

        const double eager_ms = timeHipEvents(stream, iterations, run_grouped);
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

        HipGraphOwner graph;
        EXPECT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);
        const bool captured = run_grouped();
        const hipError_t end_status = hipStreamEndCapture(stream, graph.graphPtr());
        EXPECT_TRUE(captured);
        EXPECT_EQ(end_status, hipSuccess) << hipGetErrorString(end_status);
        EXPECT_NE(*graph.graphPtr(), nullptr);
        EXPECT_EQ(hipGraphInstantiate(graph.execPtr(), *graph.graphPtr(), nullptr, nullptr, 0), hipSuccess);
        for (int i = 0; i < warmups; ++i)
            EXPECT_EQ(hipGraphLaunch(graph.execHandle(), stream), hipSuccess);
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        const double graph_ms = timeHipEvents(
            stream,
            iterations,
            [&]()
            {
                return hipGraphLaunch(graph.execHandle(), stream) == hipSuccess;
            });
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

        const double serial_ms = timeHipEvents(stream, std::max(1, iterations / 4), run_serial);
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

        TransferEngine::publishDeviceWrite(grouped_output, device, stream);
        std::vector<float> grouped(
            grouped_output->data(),
            grouped_output->data() + grouped_output->numel());
        CloseMetrics metrics = compareVectors(grouped, serial, static_cast<size_t>(d_model));

        grouped_stage.unbindWorkspace();
        if (moe_workspace)
            moe_workspace->unbindWorkspace();
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);

        return BenchResult{
            "rocm",
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

    BenchResult runROCmSharedExpertStageSerialOnlyCase(int rows, const QuantFormatCase &format)
    {
        constexpr int d_model = 2048;
        constexpr int intermediate = 512;
        const int iterations = envInt("LLAMINAR_MOE_VERIFIER_PREFILL_ITERS", 24);
        const auto device = llaminar2::DeviceId::rocm(0);

        EXPECT_EQ(hipSetDevice(0), hipSuccess);
        hipStream_t stream = nullptr;
        EXPECT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

        auto gate_w = format.create(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 7101);
        auto up_w = format.create(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 7102);
        auto down_w = format.create(
            {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)}, 7103);
        auto prepared = llaminar2::test::makeGpuPreparedFFNFixture(
            gate_w.get(),
            up_w.get(),
            down_w.get(),
            device,
            std::string("perf.moe_verifier.rocm.shared_stage.serial.") + format.name,
            llaminar2::ModelContextId{391000});

        const auto hidden_values = makeHiddenValues(rows, d_model);
        auto hidden = makeTensor({static_cast<size_t>(rows), static_cast<size_t>(d_model)}, hidden_values);
        auto serial_output_a = makeZeros({static_cast<size_t>(rows), static_cast<size_t>(d_model)});
        auto serial_output_b = makeZeros({static_cast<size_t>(rows), static_cast<size_t>(d_model)});
        EXPECT_TRUE(hidden->ensureOnDevice(device, stream));
        EXPECT_TRUE(serial_output_a->ensureOnDevice(device, stream));
        EXPECT_TRUE(serial_output_b->ensureOnDevice(device, stream));

        auto make_params = [&](llaminar2::TensorBase *output)
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
            params.force_grouped_verifier_prefill_for_decode = false;
            params.force_decode_equivalent_verifier_prefill = true;
            return params;
        };

        llaminar2::SharedExpertFFNStage serial_stage_a(make_params(serial_output_a.get()));
        llaminar2::SharedExpertFFNStage serial_stage_b(make_params(serial_output_b.get()));
        serial_stage_a.setGPUStream(stream);
        serial_stage_b.setGPUStream(stream);
        EXPECT_TRUE(serial_stage_a.usesCPUDecodeEquivalentVerifierPrefillForTesting());
        EXPECT_TRUE(serial_stage_b.usesCPUDecodeEquivalentVerifierPrefillForTesting());

        auto reqs = serial_stage_a.getWorkspaceRequirements(rows, d_model, intermediate);
        reqs.merge(llaminar2::MoEWorkspaceBuffers::rocmMoE(
            rows,
            d_model,
            intermediate,
            /*num_experts=*/256,
            /*top_k=*/8));
        auto workspace = std::make_unique<llaminar2::DeviceWorkspaceManager>(
            device,
            reqs.total_bytes_with_alignment() + 8 * 1024 * 1024);
        EXPECT_TRUE(workspace->allocate(reqs));
        serial_stage_a.bindWorkspace(workspace.get());
        serial_stage_b.bindWorkspace(workspace.get());

        llaminar2::testing::MockDeviceContext ctx(
            device, llaminar2::ComputeBackendType::GPU_ROCM);
        auto run_a = [&]()
        {
            return serial_stage_a.execute(&ctx);
        };
        auto run_b = [&]()
        {
            return serial_stage_b.execute(&ctx);
        };

        EXPECT_TRUE(run_a());
        EXPECT_TRUE(run_b());
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        const double serial_ms = timeHipEvents(stream, std::max(1, iterations), run_a);
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

        TransferEngine::publishDeviceWrite(serial_output_a, device, stream);
        TransferEngine::publishDeviceWrite(serial_output_b, device, stream);
        std::vector<float> serial_a(
            serial_output_a->data(),
            serial_output_a->data() + serial_output_a->numel());
        std::vector<float> serial_b(
            serial_output_b->data(),
            serial_output_b->data() + serial_output_b->numel());
        CloseMetrics metrics = compareVectors(serial_a, serial_b, static_cast<size_t>(d_model));

        serial_stage_a.unbindWorkspace();
        serial_stage_b.unbindWorkspace();
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);

        return BenchResult{
            "rocm",
            std::string("shared_stage_ffn_serial_only_") + format.name,
            rows,
            1,
            1,
            d_model,
            intermediate,
            serial_ms,
            0.0,
            0.0,
            0.0,
            serial_ms,
            metrics};
    }

    /**
     * @brief Exercise the non-grouped M=1 shared-FFN verifier fallback.
     *
     * Phase-split MTP verifier graphs set
     * `SharedExpertFFNStage::Params::disable_grouped_decode_shortcut` so that
     * the sidecar does not reuse the normal one-token grouped shared-expert
     * shortcut as its verifier oracle.  This focused case keeps the fixture
     * small while proving that the plain one-row prepared-GEMM path remains
     * finite and decode-equivalent for sidecar-style inputs.
     */
    BenchResult runROCmSharedExpertStageM1DisabledShortcutCase(
        const QuantFormatCase &format)
    {
        constexpr int rows = 1;
        constexpr int d_model = 2048;
        constexpr int intermediate = 512;
        const auto device = llaminar2::DeviceId::rocm(0);

        EXPECT_EQ(hipSetDevice(0), hipSuccess);
        hipStream_t stream = nullptr;
        EXPECT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

        auto gate_w = format.create(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 8101);
        auto up_w = format.create(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 8102);
        auto down_w = format.create(
            {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)}, 8103);
        auto prepared = llaminar2::test::makeGpuPreparedFFNFixture(
            gate_w.get(),
            up_w.get(),
            down_w.get(),
            device,
            std::string("perf.moe_verifier.rocm.shared_stage.m1_disabled.") + format.name,
            llaminar2::ModelContextId{392000});

        const auto hidden_values = makeHiddenValues(rows, d_model);
        auto hidden = makeTensor({static_cast<size_t>(rows), static_cast<size_t>(d_model)}, hidden_values);
        auto grouped_output = makeZeros({static_cast<size_t>(rows), static_cast<size_t>(d_model)});
        auto disabled_output = makeZeros({static_cast<size_t>(rows), static_cast<size_t>(d_model)});
        EXPECT_TRUE(hidden->ensureOnDevice(device, stream));
        EXPECT_TRUE(grouped_output->ensureOnDevice(device, stream));
        EXPECT_TRUE(disabled_output->ensureOnDevice(device, stream));

        auto make_params = [&](llaminar2::TensorBase *output,
                               bool disable_grouped_decode_shortcut)
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
            params.disable_grouped_decode_shortcut =
                disable_grouped_decode_shortcut;
            return params;
        };

        llaminar2::SharedExpertFFNStage grouped_stage(
            make_params(grouped_output.get(), false));
        llaminar2::SharedExpertFFNStage disabled_stage(
            make_params(disabled_output.get(), true));
        grouped_stage.setGPUStream(stream);
        disabled_stage.setGPUStream(stream);
        EXPECT_FALSE(disabled_stage.usesGroupedDecodeForTesting());

        auto reqs = grouped_stage.getWorkspaceRequirements(rows, d_model, intermediate);
        reqs.merge(disabled_stage.getWorkspaceRequirements(rows, d_model, intermediate));
        reqs.merge(llaminar2::MoEWorkspaceBuffers::rocmMoE(
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
        disabled_stage.bindWorkspace(workspace.get());

        llaminar2::testing::MockDeviceContext ctx(
            device, llaminar2::ComputeBackendType::GPU_ROCM);
        EXPECT_TRUE(grouped_stage.execute(&ctx));
        EXPECT_TRUE(disabled_stage.execute(&ctx));
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

        TransferEngine::publishDeviceWrite(grouped_output, device, stream);
        TransferEngine::publishDeviceWrite(disabled_output, device, stream);
        std::vector<float> grouped(
            grouped_output->data(),
            grouped_output->data() + grouped_output->numel());
        std::vector<float> disabled(
            disabled_output->data(),
            disabled_output->data() + disabled_output->numel());
        CloseMetrics metrics =
            compareVectors(disabled, grouped, static_cast<size_t>(d_model));

        grouped_stage.unbindWorkspace();
        disabled_stage.unbindWorkspace();
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);

        return BenchResult{
            "rocm",
            std::string("shared_stage_ffn_m1_disabled_shortcut_") + format.name,
            rows,
            1,
            1,
            d_model,
            intermediate,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            metrics};
    }

    /**
     * @brief Timing and native-bit result for the production ROCm MoE router.
     *
     * `graph_ms` measures one captured grouped prefill route. `rowwise_ms`
     * measures the same rows through independent production M=1 routing calls,
     * which remain a test oracle only.
     */
    struct RouterBenchResult
    {
        int rows = 0;
        double graph_ms = 0.0;
        double rowwise_ms = 0.0;
        size_t index_bit_mismatches = 0;
        size_t weight_bit_mismatches = 0;
    };

    /**
     * @brief Compute end-to-end router GOPS for the Qwen3.6 production shape.
     *
     * The denominator includes row quantization, grouped Q8 logits, exact
     * softmax/top-k, integer-to-FP32 publication, and graph launch overhead.
     */
    double routerPipelineGops(const RouterBenchResult &result)
    {
        constexpr double d_model = 2048.0;
        constexpr double num_experts = 256.0;
        if (result.graph_ms <= 0.0)
            return 0.0;
        const double operations =
            2.0 * static_cast<double>(result.rows) * d_model * num_experts;
        return operations / (result.graph_ms * 1.0e6);
    }

    /**
     * @brief Force the production Q8 router policy for one benchmark case.
     *
     * The combined CUDA/ROCm perf binary may initialize DebugEnv before the ROCm
     * tests run. Mutating the parsed policy directly avoids test-order-dependent
     * environment reloads while restoring every field on scope exit.
     */
    class ScopedROCmBatchInvariantRouterPolicy
    {
    public:
        ScopedROCmBatchInvariantRouterPolicy()
            : old_q8_(llaminar2::mutableDebugEnv().rocm.moe_router_q8),
              old_fp16_(llaminar2::mutableDebugEnv().rocm.moe_router_fp16),
              old_reuse_(llaminar2::mutableDebugEnv().rocm.moe_reuse_router_q8_hidden)
        {
            auto &config = llaminar2::mutableDebugEnv().rocm;
            config.moe_router_q8 = true;
            config.moe_router_fp16 = false;
            config.moe_reuse_router_q8_hidden = true;
        }

        ~ScopedROCmBatchInvariantRouterPolicy()
        {
            auto &config = llaminar2::mutableDebugEnv().rocm;
            config.moe_router_q8 = old_q8_;
            config.moe_router_fp16 = old_fp16_;
            config.moe_reuse_router_q8_hidden = old_reuse_;
        }

        ScopedROCmBatchInvariantRouterPolicy(
            const ScopedROCmBatchInvariantRouterPolicy &) = delete;
        ScopedROCmBatchInvariantRouterPolicy &operator=(
            const ScopedROCmBatchInvariantRouterPolicy &) = delete;

    private:
        bool old_q8_ = true;
        bool old_fp16_ = false;
        bool old_reuse_ = true;
    };

    /**
     * @brief Benchmark captured batch-invariant Q8 routing against serial M=1.
     *
     * The grouped kernel owns one expert and one fixed four-row tile. Increasing
     * M adds independent tiles while preserving each row's serial K traversal
     * and reduction tree. The perfstats assertion prevents a batch-shaped GEMM
     * or hidden row-replay substitution from passing on coincidental equality.
     */
    RouterBenchResult runROCmBatchInvariantRouterCase(int rows)
    {
        constexpr int d_model = 2048;
        constexpr int intermediate = 512;
        constexpr int num_experts = 256;
        constexpr int top_k = 8;
        const auto device = llaminar2::DeviceId::rocm(0);
        const int iterations = envInt(
            "LLAMINAR_ROCM_MOE_BATCH_INVARIANT_ROUTER_ITERS",
            rows >= 256 ? 12 : (rows >= 32 ? 40 : 160));
        const int rowwise_iterations = rows >= 32 ? 1 : 3;

        ScopedROCmBatchInvariantRouterPolicy policy;
        EXPECT_EQ(hipSetDevice(0), hipSuccess);
        hipStream_t stream = nullptr;
        EXPECT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

        auto moe = KernelFactory::createMoEKernel(device);
        EXPECT_NE(moe, nullptr);
        moe->setGPUStream(stream);
        auto *workspace_consumer = dynamic_cast<llaminar2::IWorkspaceConsumer *>(moe.get());
        EXPECT_NE(workspace_consumer, nullptr);
        auto requirements = llaminar2::MoEWorkspaceBuffers::rocmMoE(
            rows, d_model, intermediate, num_experts, top_k);
        auto workspace = std::make_unique<llaminar2::DeviceWorkspaceManager>(
            device,
            requirements.total_bytes_with_alignment() + 8 * 1024 * 1024);
        EXPECT_TRUE(workspace->allocate(requirements));
        workspace_consumer->bindWorkspace(workspace.get());

        const std::vector<float> hidden_values = makeHiddenValues(rows, d_model);
        auto hidden = makeTensor(
            {static_cast<size_t>(rows), static_cast<size_t>(d_model)},
            hidden_values);
        std::vector<float> gate_values(
            static_cast<size_t>(num_experts) * static_cast<size_t>(d_model));
        for (size_t i = 0; i < gate_values.size(); ++i)
        {
            gate_values[i] =
                0.021f * std::sin(0.0037f * static_cast<float>(i + 37)) +
                0.014f * std::cos(0.0051f * static_cast<float>(i + 19));
        }
        auto gate = makeTensor(
            {static_cast<size_t>(num_experts), static_cast<size_t>(d_model)},
            gate_values);
        auto grouped_indices = makeZeros(
            {static_cast<size_t>(rows), static_cast<size_t>(top_k)});
        auto grouped_weights = makeZeros(
            {static_cast<size_t>(rows), static_cast<size_t>(top_k)});
        EXPECT_TRUE(hidden->ensureOnDevice(device, stream));
        EXPECT_TRUE(gate->ensureOnDevice(device, stream));
        EXPECT_TRUE(grouped_indices->ensureOnDevice(device, stream));
        EXPECT_TRUE(grouped_weights->ensureOnDevice(device, stream));

        llaminar2::MoERoutingResult ignored_host_result;
        auto run_grouped = [&]()
        {
            return moe->routeWithTensors(
                hidden.get(), gate.get(), rows, d_model, num_experts, top_k,
                /*normalize_weights=*/true,
                grouped_indices.get(), grouped_weights.get(), ignored_host_result);
        };

        EXPECT_TRUE(run_grouped());
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

        HipGraphOwner graph;
        EXPECT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);
        const bool captured = run_grouped();
        const hipError_t capture_status =
            hipStreamEndCapture(stream, graph.graphPtr());
        EXPECT_TRUE(captured);
        EXPECT_EQ(capture_status, hipSuccess) << hipGetErrorString(capture_status);
        EXPECT_NE(*graph.graphPtr(), nullptr);
        EXPECT_EQ(
            hipGraphInstantiate(graph.execPtr(), *graph.graphPtr(), nullptr, nullptr, 0),
            hipSuccess);
        for (int i = 0; i < 3; ++i)
            EXPECT_EQ(hipGraphLaunch(graph.execHandle(), stream), hipSuccess);
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        const double graph_ms = timeHipEvents(
            stream,
            iterations,
            [&]()
            {
                return hipGraphLaunch(graph.execHandle(), stream) == hipSuccess;
            });
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

        llaminar2::PerfStatsCollector::reset();
        EXPECT_TRUE(run_grouped());
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        const auto route_records = llaminar2::PerfStatsCollector::snapshot(
            {"kernel.rocm_moe_batch_invariant_prefill_router_calls"});
        const std::string expected_rows = std::to_string(rows);
        const auto route_record = std::find_if(
            route_records.begin(),
            route_records.end(),
            [&](const llaminar2::PerfStatRecord &record)
            {
                const auto tag_equals = [&](const char *key, const std::string &value)
                {
                    const auto it = record.tags.find(key);
                    return it != record.tags.end() && it->second == value;
                };
                return record.name ==
                           "rocm_moe_batch_invariant_prefill_router_calls" &&
                       tag_equals("seq_len", expected_rows) &&
                       tag_equals("row_tile", "16") &&
                       tag_equals("route", "grouped_q8") &&
                       record.count > 0;
            });
        EXPECT_NE(route_record, route_records.end())
            << llaminar2::PerfStatsCollector::summaryString(
                   {"kernel.rocm_moe_batch_invariant_prefill_router_calls"});

        TransferEngine::publishDeviceWrite(grouped_indices, device, stream);
        TransferEngine::publishDeviceWrite(grouped_weights, device, stream);
        const std::vector<float> grouped_index_values(
            grouped_indices->data(),
            grouped_indices->data() + grouped_indices->numel());
        const std::vector<float> grouped_weight_values(
            grouped_weights->data(),
            grouped_weights->data() + grouped_weights->numel());

        std::vector<std::shared_ptr<llaminar2::FP32Tensor>> row_hidden;
        std::vector<std::shared_ptr<llaminar2::FP32Tensor>> row_indices;
        std::vector<std::shared_ptr<llaminar2::FP32Tensor>> row_weights;
        row_hidden.reserve(rows);
        row_indices.reserve(rows);
        row_weights.reserve(rows);
        for (int row = 0; row < rows; ++row)
        {
            const auto first =
                hidden_values.begin() + static_cast<ptrdiff_t>(row) * d_model;
            row_hidden.push_back(makeTensor(
                {1u, static_cast<size_t>(d_model)},
                std::vector<float>(first, first + d_model)));
            row_indices.push_back(makeZeros({1u, static_cast<size_t>(top_k)}));
            row_weights.push_back(makeZeros({1u, static_cast<size_t>(top_k)}));
            EXPECT_TRUE(row_hidden.back()->ensureOnDevice(device, stream));
            EXPECT_TRUE(row_indices.back()->ensureOnDevice(device, stream));
            EXPECT_TRUE(row_weights.back()->ensureOnDevice(device, stream));
        }

        auto run_rowwise = [&]()
        {
            for (int row = 0; row < rows; ++row)
            {
                if (!moe->routeVerifierRowsDecodeEquivalent(
                        row_hidden[static_cast<size_t>(row)].get(),
                        gate.get(),
                        /*seq_len=*/1,
                        d_model,
                        num_experts,
                        top_k,
                        /*normalize_weights=*/true,
                        row_indices[static_cast<size_t>(row)].get(),
                        row_weights[static_cast<size_t>(row)].get()))
                {
                    return false;
                }
            }
            return true;
        };
        EXPECT_TRUE(run_rowwise());
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        const double rowwise_ms = timeHipEvents(
            stream, rowwise_iterations, run_rowwise);
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

        std::vector<float> serial_index_values;
        std::vector<float> serial_weight_values;
        serial_index_values.reserve(static_cast<size_t>(rows) * top_k);
        serial_weight_values.reserve(static_cast<size_t>(rows) * top_k);
        for (int row = 0; row < rows; ++row)
        {
            auto &indices = row_indices[static_cast<size_t>(row)];
            auto &weights = row_weights[static_cast<size_t>(row)];
            TransferEngine::publishDeviceWrite(indices, device, stream);
            TransferEngine::publishDeviceWrite(weights, device, stream);
            serial_index_values.insert(
                serial_index_values.end(), indices->data(), indices->data() + top_k);
            serial_weight_values.insert(
                serial_weight_values.end(), weights->data(), weights->data() + top_k);
        }

        const CloseMetrics index_metrics = compareVectors(
            grouped_index_values, serial_index_values, static_cast<size_t>(top_k));
        const CloseMetrics weight_metrics = compareVectors(
            grouped_weight_values, serial_weight_values, static_cast<size_t>(top_k));
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);

        return RouterBenchResult{
            rows,
            graph_ms,
            rowwise_ms,
            index_metrics.bit_mismatch_count,
            weight_metrics.bit_mismatch_count};
    }

    /**
     * @brief Exercise production's M=1 grouped-verifier shared-FFN route.
     *
     * The Qwen3.6 MoE MTP sidecar sets both verifier-facing knobs on
     * `SharedExpertFFNStage`: it disables the ordinary grouped decode shortcut
     * and asks for grouped verifier prefill when the backend advertises that
     * capability.  The older stage tests covered M=2..4, but the prefix-cache
     * restore path starts with a single sidecar row.  This regression keeps
     * that exact row count and verifies that the stage refuses to route M=1
     * into the grouped verifier kernel, whose contract starts at M=2.
     */
    BenchResult runROCmSharedExpertStageM1GroupedVerifierCase(
        const QuantFormatCase &format)
    {
        constexpr int rows = 1;
        constexpr int d_model = 2048;
        constexpr int intermediate = 512;
        const auto device = llaminar2::DeviceId::rocm(0);

        EXPECT_EQ(hipSetDevice(0), hipSuccess);
        hipStream_t stream = nullptr;
        EXPECT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

        auto gate_w = format.create(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 8201);
        auto up_w = format.create(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 8202);
        auto down_w = format.create(
            {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)}, 8203);
        auto prepared = llaminar2::test::makeGpuPreparedFFNFixture(
            gate_w.get(),
            up_w.get(),
            down_w.get(),
            device,
            std::string("perf.moe_verifier.rocm.shared_stage.m1_grouped_verifier.") + format.name,
            llaminar2::ModelContextId{393000});

        const auto hidden_values = makeHiddenValues(rows, d_model);
        auto hidden = makeTensor({static_cast<size_t>(rows), static_cast<size_t>(d_model)}, hidden_values);
        auto grouped_output = makeZeros({static_cast<size_t>(rows), static_cast<size_t>(d_model)});
        auto reference_output = makeZeros({static_cast<size_t>(rows), static_cast<size_t>(d_model)});
        EXPECT_TRUE(hidden->ensureOnDevice(device, stream));
        EXPECT_TRUE(grouped_output->ensureOnDevice(device, stream));
        EXPECT_TRUE(reference_output->ensureOnDevice(device, stream));

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
            params.seq_len = rows;
            params.d_model = d_model;
            params.intermediate = intermediate;
            params.prepared_ref_gate = prepared.gate_ref;
            params.prepared_ref_up = prepared.up_ref;
            params.prepared_ref_down = prepared.down_ref;
            params.prepared_store = prepared.store.get();
            params.force_grouped_verifier_prefill_for_decode = grouped_verifier;
            params.disable_grouped_decode_shortcut = true;
            return params;
        };

        llaminar2::SharedExpertFFNStage grouped_stage(
            make_params(grouped_output.get(), true));
        llaminar2::SharedExpertFFNStage reference_stage(
            make_params(reference_output.get(), false));
        grouped_stage.setGPUStream(stream);
        reference_stage.setGPUStream(stream);
        /*
         * M=1 is still a verifier bucket in the MTP transaction.  Keep it on
         * the same grouped table-prefill route as M=2..4 so the publication
         * path has one production implementation and this perf regression
         * proves that even the smallest bucket remains decode-equivalent.
         */
        EXPECT_TRUE(grouped_stage.usesGroupedVerifierPrefillRouteForTesting());
        EXPECT_FALSE(grouped_stage.usesGroupedDecodeForTesting());
        EXPECT_FALSE(reference_stage.usesGroupedVerifierPrefillRouteForTesting());
        EXPECT_FALSE(reference_stage.usesGroupedDecodeForTesting());

        auto reqs = grouped_stage.getWorkspaceRequirements(rows, d_model, intermediate);
        reqs.merge(reference_stage.getWorkspaceRequirements(rows, d_model, intermediate));
        reqs.merge(llaminar2::MoEWorkspaceBuffers::rocmMoE(
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
        reference_stage.bindWorkspace(workspace.get());

        llaminar2::testing::MockDeviceContext ctx(
            device, llaminar2::ComputeBackendType::GPU_ROCM);
        EXPECT_TRUE(grouped_stage.execute(&ctx));
        EXPECT_TRUE(reference_stage.execute(&ctx));
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

        TransferEngine::publishDeviceWrite(grouped_output, device, stream);
        TransferEngine::publishDeviceWrite(reference_output, device, stream);
        std::vector<float> grouped(
            grouped_output->data(),
            grouped_output->data() + grouped_output->numel());
        std::vector<float> reference(
            reference_output->data(),
            reference_output->data() + reference_output->numel());
        CloseMetrics metrics =
            compareVectors(grouped, reference, static_cast<size_t>(d_model));

        grouped_stage.unbindWorkspace();
        reference_stage.unbindWorkspace();
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);

        return BenchResult{
            "rocm",
            std::string("shared_stage_ffn_m1_grouped_verifier_") + format.name,
            rows,
            1,
            1,
            d_model,
            intermediate,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            metrics};
    }

    /** One reciprocal MoE projection shape in the dispatch trainer. */
    struct MoEPrefillSweepShape
    {
        const char *name;
        int d_model;
        int intermediate;
    };

    /** One compile-time candidate exposed by the production HIP launcher. */
    struct MoEPrefillSweepCandidate
    {
        int tile_m;
        int tile_n;
    };

    /** Machine-readable result for one exact candidate graph replay. */
    struct MoEPrefillSweepResult
    {
        std::string source_format;
        uint8_t source_codebook = 0;
        uint8_t execution_codebook = 0;
        std::string shape;
        std::string role;
        std::string candidate_id;
        int m = 0;
        int n = 0;
        int k = 0;
        int tile_m = 0;
        int tile_n = 0;
        int warmup_count = 0;
        int sample_count = 0;
        int timed_replays = 0;
        double min_us = 0.0;
        double graph_us = 0.0;
        double p95_us = 0.0;
        double mad_us = 0.0;
        double cv = 0.0;
        double pipeline_gops = 0.0;
        size_t bit_mismatches = 0;
        size_t first_bit_mismatch = 0;
        size_t repeat_bit_mismatches = 0;
        double max_abs = 0.0;
        double relative_l2 = 0.0;
        double cosine = 0.0;
        double symmetric_kld = 0.0;
        std::string grouped_output_digest;
        std::string serial_output_digest;
        std::string timing_sample_digest;
        bool route_counter_ok = false;
        std::string observed_candidate_id;
        bool is_winner = false;
    };

    constexpr std::array<MoEPrefillSweepCandidate, 12> kMoEPrefillSweepCandidates{{
        {4, 64}, {4, 128}, {4, 256},
        {8, 64}, {8, 128}, {8, 256},
        {12, 64}, {12, 128}, {12, 256},
        {16, 64}, {16, 128}, {16, 256},
    }};

    constexpr std::array<MoEPrefillSweepShape, 7> kMoEPrefillSweepShapes{{
        {"gate_ratio_1_8", 2048, 256},
        {"gate_ratio_1_4", 2048, 512},
        {"gate_ratio_1_2", 2048, 1024},
        {"gate_ratio_1_1", 1024, 1024},
        {"gate_ratio_2_1", 1024, 2048},
        {"gate_ratio_4_1", 512, 2048},
        {"gate_ratio_8_1", 256, 2048},
    }};

    std::string moePrefillCandidateName(const MoEPrefillSweepCandidate &candidate)
    {
        return "tm" + std::to_string(candidate.tile_m) +
               "_tn" + std::to_string(candidate.tile_n);
    }

    void writeMoEPrefillSweepResult(
        std::FILE *csv,
        const MoEPrefillSweepResult &result)
    {
        std::fprintf(
            csv,
            "rocm,grouped_prefill,%s,%u,%u,%s,%s,%s,%d,%d,%d,%d,%d,"
            "%d,%d,%d,%.3f,%.3f,%.3f,%.6f,%.6f,%.3f,%zu,%zu,%zu,"
            "%.9g,%.9g,%.9g,%.9g,%s,%s,%s,%d,%s,%d\n",
            result.source_format.c_str(),
            static_cast<unsigned>(result.source_codebook),
            static_cast<unsigned>(result.execution_codebook),
            result.shape.c_str(),
            result.role.c_str(),
            result.candidate_id.c_str(),
            result.m,
            result.n,
            result.k,
            result.tile_m,
            result.tile_n,
            result.warmup_count,
            result.sample_count,
            result.timed_replays,
            result.min_us,
            result.graph_us,
            result.p95_us,
            result.mad_us,
            result.cv,
            result.pipeline_gops,
            result.bit_mismatches,
            result.first_bit_mismatch,
            result.repeat_bit_mismatches,
            result.max_abs,
            result.relative_l2,
            result.cosine,
            result.symmetric_kld,
            result.grouped_output_digest.c_str(),
            result.serial_output_digest.c_str(),
            result.timing_sample_digest.c_str(),
            result.route_counter_ok ? 1 : 0,
            result.observed_candidate_id.c_str(),
            result.is_winner ? 1 : 0);
        std::fflush(csv);
    }

    /**
     * @brief Sweep every selected projection policy against one prepared shape.
     *
     * Eight real expert matrices are retained for the whole shape sweep. Every
     * token routes once to each expert, so an M-row batch gives each expert M
     * rows and makes row-tile reuse directly observable. Grouping is prepared
     * once per M bucket and remains device-resident while each candidate captures
     * the production projection pipeline. The serial-row oracle is also built
     * once per M bucket, then every candidate is required to match it bitwise.
     */
    void runROCmMoEPrefillFormatShapeSweep(
        const llaminar2::test::QuantizedVerifierFormatCase &format,
        const MoEPrefillSweepShape &shape,
        const std::vector<int> &m_values,
        int warmups,
        int iterations,
        int timing_trials,
        std::FILE *csv,
        std::FILE *timing_csv)
    {
        constexpr int top_k = 8;
        constexpr int num_experts = 8;
        const auto device = llaminar2::DeviceId::rocm(0);
        const int max_rows = *std::max_element(m_values.begin(), m_values.end());

        ASSERT_EQ(hipSetDevice(0), hipSuccess);
        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

        llaminar2::ROCmMoEKernel moe_storage(0);
        llaminar2::IMoEKernel *moe = &moe_storage;
        moe->setGPUStream(stream);
        auto *workspace_consumer =
            dynamic_cast<llaminar2::IWorkspaceConsumer *>(moe);
        ASSERT_NE(workspace_consumer, nullptr);

        auto reqs = llaminar2::MoEWorkspaceBuffers::rocmMoE(
            max_rows,
            shape.d_model,
            shape.intermediate,
            num_experts,
            top_k);
        auto workspace = std::make_unique<llaminar2::DeviceWorkspaceManager>(
            device,
            reqs.total_bytes_with_alignment() + 8 * 1024 * 1024);
        ASSERT_TRUE(workspace->allocate(reqs));
        workspace_consumer->bindWorkspace(workspace.get());

        std::vector<int> materialized_experts(static_cast<size_t>(num_experts));
        for (int expert = 0; expert < num_experts; ++expert)
            materialized_experts[static_cast<size_t>(expert)] = expert;
        auto tables = prepareExpertTables(
            moe,
            device,
            num_experts,
            shape.d_model,
            shape.intermediate,
            materialized_experts,
            format,
            format);

        for (const int rows : m_values)
        {
            if (rows <= 8)
                continue;

            const std::vector<float> hidden_values =
                makeHiddenValues(rows, shape.d_model);
            const std::vector<float> routing_indices =
                makeRoutingIndices(rows, top_k, num_experts);
            const std::vector<float> routing_weights =
                makeRoutingWeights(rows, top_k);
            auto hidden = makeTensor(
                {static_cast<size_t>(rows), static_cast<size_t>(shape.d_model)},
                hidden_values);
            auto route_indices_tensor = makeTensor(
                {static_cast<size_t>(rows), static_cast<size_t>(top_k)},
                routing_indices);
            auto route_weights_tensor = makeTensor(
                {static_cast<size_t>(rows), static_cast<size_t>(top_k)},
                routing_weights);
            auto grouped_output = makeZeros(
                {static_cast<size_t>(rows), static_cast<size_t>(shape.d_model)});
            ASSERT_TRUE(hidden->ensureOnDevice(device, stream));
            ASSERT_TRUE(route_indices_tensor->ensureOnDevice(device, stream));
            ASSERT_TRUE(route_weights_tensor->ensureOnDevice(device, stream));
            ASSERT_TRUE(grouped_output->ensureOnDevice(device, stream));

            const auto run_prepare = [&]()
            {
                return moe->prepareExpertGroupsAsync(
                    route_indices_tensor.get(),
                    route_weights_tensor.get(),
                    rows,
                    num_experts,
                    top_k);
            };
            const auto run_pipeline = [&]()
            {
                return moe->executeGroupedPrefillPipeline(
                    hidden.get(),
                    grouped_output.get(),
                    tables.gateup_table_id,
                    tables.down_table_id,
                    rows,
                    shape.d_model,
                    shape.intermediate,
                    num_experts,
                    top_k);
            };

            ASSERT_TRUE(run_prepare());
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            double serial_ms = 0.0;
            const std::vector<float> serial = runRowwiseDecode(
                moe,
                stream,
                hidden_values,
                routing_indices,
                routing_weights,
                rows,
                top_k,
                shape.d_model,
                shape.intermediate,
                tables.gateup_table_id,
                tables.down_table_id,
                &serial_ms);

            for (const std::string role : {std::string("gateup"), std::string("down")})
            {
                if (!envCsvContainsOrUnset(
                        "LLAMINAR_ROCM_MOE_PREFILL_SWEEP_ROLES", role))
                {
                    continue;
                }

                std::vector<MoEPrefillSweepResult> role_results;
                role_results.reserve(kMoEPrefillSweepCandidates.size());
                for (const auto &candidate : kMoEPrefillSweepCandidates)
                {
                    const std::string candidate_name =
                        moePrefillCandidateName(candidate);
                    if (!envCsvContainsOrUnset(
                            "LLAMINAR_ROCM_MOE_PREFILL_SWEEP_VARIANTS",
                            candidate_name))
                    {
                        continue;
                    }

                    const int gateup_tile_m =
                        role == "gateup" ? candidate.tile_m : 16;
                    const int gateup_tile_n =
                        role == "gateup" ? candidate.tile_n : 128;
                    const int down_tile_m =
                        role == "down" ? candidate.tile_m : 16;
                    const int down_tile_n =
                        role == "down" ? candidate.tile_n : 128;
                    const std::string gateup_tile_m_text = std::to_string(gateup_tile_m);
                    const std::string gateup_tile_n_text = std::to_string(gateup_tile_n);
                    const std::string down_tile_m_text = std::to_string(down_tile_m);
                    const std::string down_tile_n_text = std::to_string(down_tile_n);

                    ScopedEnvOverride gateup_m(
                        "LLAMINAR_ROCM_MOE_PREFILL_GATEUP_TILE_M",
                        gateup_tile_m_text.c_str());
                    ScopedEnvOverride gateup_n(
                        "LLAMINAR_ROCM_MOE_PREFILL_GATEUP_TILE_N",
                        gateup_tile_n_text.c_str());
                    ScopedEnvOverride down_m(
                        "LLAMINAR_ROCM_MOE_PREFILL_DOWN_TILE_M",
                        down_tile_m_text.c_str());
                    ScopedEnvOverride down_n(
                        "LLAMINAR_ROCM_MOE_PREFILL_DOWN_TILE_N",
                        down_tile_n_text.c_str());

                    llaminar2::PerfStatsCollector::reset();
                    HipGraphOwner graph;
                    ASSERT_EQ(
                        hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
                        hipSuccess);
                    const bool captured = run_pipeline();
                    const hipError_t capture_status =
                        hipStreamEndCapture(stream, graph.graphPtr());
                    ASSERT_TRUE(captured)
                        << format.label << ' ' << shape.name << " M=" << rows
                        << ' ' << role << ' ' << candidate_name;
                    ASSERT_EQ(capture_status, hipSuccess)
                        << hipGetErrorString(capture_status);
                    ASSERT_NE(*graph.graphPtr(), nullptr);
                    ASSERT_EQ(
                        hipGraphInstantiate(
                            graph.execPtr(), *graph.graphPtr(), nullptr, nullptr, 0),
                        hipSuccess);

                    const auto route_records =
                        llaminar2::PerfStatsCollector::snapshot(
                            {"kernel.rocm_moe_grouped_prefill_batch_invariant_calls"});
                    bool route_counter_ok = false;
                    std::string observed_candidate_id = "missing";
                    for (const auto &record : route_records)
                    {
                        if (record.name !=
                            "rocm_moe_grouped_prefill_batch_invariant_calls")
                        {
                            continue;
                        }
                        const auto tag = [&](const char *name) -> std::string
                        {
                            const auto iterator = record.tags.find(name);
                            return iterator == record.tags.end()
                                       ? std::string{}
                                       : iterator->second;
                        };
                        const std::string observed_tile_m =
                            tag(role == "gateup" ? "gateup_tile_m" : "down_tile_m");
                        const std::string observed_tile_n =
                            tag(role == "gateup" ? "gateup_tile_n" : "down_tile_n");
                        if (!observed_tile_m.empty() && !observed_tile_n.empty())
                        {
                            observed_candidate_id =
                                "tm" + observed_tile_m + "_tn" + observed_tile_n;
                        }
                        route_counter_ok =
                            tag("seq_len") == std::to_string(rows) &&
                            observed_candidate_id == candidate_name &&
                            record.count > 0;
                        if (route_counter_ok)
                            break;
                    }
                    EXPECT_TRUE(route_counter_ok)
                        << format.label << ' ' << shape.name << " M=" << rows
                        << ' ' << role << ' ' << candidate_name << '\n'
                        << llaminar2::PerfStatsCollector::summaryString(
                               {"kernel.rocm_moe_grouped_prefill_batch_invariant_calls"});

                    for (int warmup = 0; warmup < warmups; ++warmup)
                    {
                        ASSERT_EQ(
                            hipGraphLaunch(graph.execHandle(), stream),
                            hipSuccess);
                    }
                    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

                    const auto download_grouped_output = [&]()
                    {
                        std::vector<float> host(grouped_output->numel());
                        const hipError_t copy_status = hipMemcpyAsync(
                            host.data(),
                            grouped_output->gpu_data_ptr(),
                            host.size() * sizeof(float),
                            hipMemcpyDeviceToHost,
                            stream);
                        EXPECT_EQ(copy_status, hipSuccess);
                        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
                        return host;
                    };
                    const std::vector<float> repeated_output_a =
                        download_grouped_output();
                    ASSERT_EQ(
                        hipGraphLaunch(graph.execHandle(), stream),
                        hipSuccess);
                    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
                    const std::vector<float> repeated_output_b =
                        download_grouped_output();
                    const size_t repeat_bit_mismatches =
                        llaminar2::test::trainer::nativeByteMismatchCount(
                            repeated_output_a,
                            repeated_output_b);
                    EXPECT_EQ(repeat_bit_mismatches, 0u)
                        << format.label << ' ' << shape.name << " M=" << rows
                        << ' ' << role << ' ' << candidate_name
                        << " captured graph replay is not byte-stable";

                    std::vector<double> trial_ms;
                    trial_ms.reserve(static_cast<size_t>(timing_trials));
                    for (int trial = 0; trial < timing_trials; ++trial)
                    {
                        trial_ms.push_back(timeHipEvents(
                            stream,
                            iterations,
                            [&]()
                            {
                                return hipGraphLaunch(graph.execHandle(), stream) == hipSuccess;
                            }));
                    }
                    std::sort(trial_ms.begin(), trial_ms.end());
                    const auto timing_evidence =
                        llaminar2::test::trainer::summarizeSortedTimingSamples(
                            trial_ms);
                    const double min_graph_ms = timing_evidence.min;
                    const double graph_ms = timing_evidence.median;
                    const double p95_graph_ms = timing_evidence.p95;
                    const double mad_graph_ms = timing_evidence.mad;
                    const double timing_cv = timing_evidence.cv;
                    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

                    /*
                     * Aggregate timing rows are insufficient evidence for an
                     * installable learned policy. Retain every sorted trial so
                     * the common adapter can recompute the digest, audit the
                     * sample count, and later perform paired confirmation. The
                     * trial value is already normalized to one graph replay by
                     * timeHipEvents(); timed_replays records how many launches
                     * contributed to that event sample.
                     */
                    if (timing_csv)
                    {
                        for (size_t sample_index = 0;
                             sample_index < trial_ms.size();
                             ++sample_index)
                        {
                            std::fprintf(
                                timing_csv,
                                "rocm,grouped_prefill,%s,%u,%u,%s,%s,%s,"
                                "%d,%d,%d,%d,%d,%zu,%d,%.9f,%a\n",
                                format.label,
                                static_cast<unsigned>(format.source_codebook_id),
                                static_cast<unsigned>(format.device_execution_codebook_id),
                                shape.name,
                                role.c_str(),
                                candidate_name.c_str(),
                                rows,
                                role == "gateup" ? shape.intermediate : shape.d_model,
                                role == "gateup" ? shape.d_model : shape.intermediate,
                                candidate.tile_m,
                                candidate.tile_n,
                                sample_index,
                                iterations,
                                trial_ms[sample_index] * 1000.0,
                                trial_ms[sample_index]);
                        }
                        std::fflush(timing_csv);
                    }

                    TransferEngine::publishDeviceWrite(grouped_output, device, stream);
                    ASSERT_TRUE(grouped_output->ensureOnHost(stream));
                    const std::vector<float> actual(
                        grouped_output->data(),
                        grouped_output->data() + grouped_output->numel());
                    /*
                     * Grouped MoE publication contains one hidden vector per
                     * routed row. The former d_model count inspected only the
                     * first vector; use the complete downloaded tensor for both
                     * diagnostics and the hard NativeVNNI byte certificate.
                     */
                    const CloseMetrics metrics = compareVectors(
                        actual,
                        serial,
                        actual.size());
                    const auto common_evidence =
                        llaminar2::test::trainer::compareFP32(
                            actual,
                            serial,
                            actual.size());
                    EXPECT_EQ(common_evidence.mismatch_count, 0u)
                        << format.label << ' ' << shape.name << " M=" << rows
                        << ' ' << role << ' ' << candidate_name
                        << " first_bit_mismatch="
                        << common_evidence.first_mismatch_index;

                    const double operations =
                        6.0 * static_cast<double>(rows) * top_k *
                        static_cast<double>(shape.d_model) *
                        static_cast<double>(shape.intermediate);
                    role_results.push_back(MoEPrefillSweepResult{
                        format.label,
                        format.source_codebook_id,
                        format.device_execution_codebook_id,
                        shape.name,
                        role,
                        candidate_name,
                        rows,
                        role == "gateup" ? shape.intermediate : shape.d_model,
                        role == "gateup" ? shape.d_model : shape.intermediate,
                        candidate.tile_m,
                        candidate.tile_n,
                        warmups,
                        timing_trials,
                        timing_trials * iterations,
                        min_graph_ms * 1000.0,
                        graph_ms * 1000.0,
                        p95_graph_ms * 1000.0,
                        mad_graph_ms * 1000.0,
                        timing_cv,
                        graph_ms > 0.0 ? operations / (graph_ms * 1.0e6) : 0.0,
                        common_evidence.mismatch_count,
                        common_evidence.first_mismatch_index,
                        repeat_bit_mismatches,
                        common_evidence.max_abs,
                        common_evidence.relative_l2,
                        common_evidence.cosine,
                        common_evidence.symmetric_kld,
                        common_evidence.actual_digest,
                        common_evidence.expected_digest,
                        timing_evidence.digest,
                        route_counter_ok,
                        observed_candidate_id,
                        false});
                }

                auto winner = std::min_element(
                    role_results.begin(),
                    role_results.end(),
                    [](const auto &lhs, const auto &rhs)
                    {
                        const bool lhs_exact = lhs.bit_mismatches == 0;
                        const bool rhs_exact = rhs.bit_mismatches == 0;
                        if (lhs_exact != rhs_exact)
                            return lhs_exact;
                        return lhs.graph_us < rhs.graph_us;
                    });
                if (winner != role_results.end() && winner->bit_mismatches == 0)
                    winner->is_winner = true;
                for (const auto &result : role_results)
                    writeMoEPrefillSweepResult(csv, result);
            }
        }

        workspace_consumer->unbindWorkspace();
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
    }
}
#endif

TEST(Perf__MoEVerifierPrefill, ROCm_M1234_RoutedExpertFFNDecodeEquivalent)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    for (int rows : {1, 2, 3, 4})
    {
        auto routed = runROCmCase(/*shared=*/false, rows);
        expectClose(routed.metrics);
        if (rows >= 2)
            expectGraphReplayFasterThanReference(routed);
        printResult(routed);
    }
#endif
}

TEST(Perf__MoEVerifierPrefill, ROCm_M832256_RoutedExpertBatchInvariantEconomy)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    for (int rows : {8, 32, 256})
    {
        SCOPED_TRACE(rows);
        llaminar2::PerfStatsCollector::reset();
        auto routed = runROCmCase(/*shared=*/false, rows);
        expectClose(routed.metrics);
        expectGraphReplayFasterThanReference(routed);
        expectGfx906LongPrefillThroughput(routed);

        int expected_gateup_tile_m = 1;
        int expected_gateup_tile_n = 64;
        int expected_down_tile_m = 1;
        int expected_down_tile_n = 64;
        if (rows > 8)
        {
            ASSERT_TRUE(rocmMoE_grouped_prefill_query_tile_config(
                /*IQ2_S execution codebook=*/13,
                /*projection_role=*/0,
                rows,
                routed.intermediate,
                routed.d_model,
                &expected_gateup_tile_m,
                &expected_gateup_tile_n));
            ASSERT_TRUE(rocmMoE_grouped_prefill_query_tile_config(
                /*IQ4_XS execution codebook=*/4,
                /*projection_role=*/1,
                rows,
                routed.d_model,
                routed.intermediate,
                &expected_down_tile_m,
                &expected_down_tile_n));
        }
        const std::string expected_gateup_tile_m_text =
            std::to_string(expected_gateup_tile_m);
        const std::string expected_gateup_tile_n_text =
            std::to_string(expected_gateup_tile_n);
        const std::string expected_down_tile_m_text =
            std::to_string(expected_down_tile_m);
        const std::string expected_down_tile_n_text =
            std::to_string(expected_down_tile_n);
        const std::string expected_common_row_tile =
            expected_gateup_tile_m == expected_down_tile_m
                ? expected_gateup_tile_m_text
                : "mixed";

        const auto records = llaminar2::PerfStatsCollector::snapshot(
            {"kernel.rocm_moe_grouped_prefill_batch_invariant_calls"});
        const std::string expected_rows = std::to_string(rows);
        const auto production_route = std::find_if(
            records.begin(),
            records.end(),
            [&](const llaminar2::PerfStatRecord &record)
            {
                const auto tag_equals = [&](const char *key, const char *value)
                {
                    const auto it = record.tags.find(key);
                    return it != record.tags.end() && it->second == value;
                };
                return record.name == "rocm_moe_grouped_prefill_batch_invariant_calls" &&
                       tag_equals("seq_len", expected_rows.c_str()) &&
                       tag_equals(
                           "gateup_route",
                           rows > 8
                               ? "expert_tiled_original_row_quant"
                               : "route_owned_original_row_quant") &&
                       tag_equals(
                           "down_route",
                           rows > 8
                               ? "expert_tiled_partials_ordered_publish"
                               : "direct_ordered_publish") &&
                       tag_equals("gateup_tile_m", expected_gateup_tile_m_text.c_str()) &&
                       tag_equals("gateup_tile_n", expected_gateup_tile_n_text.c_str()) &&
                       tag_equals("down_tile_m", expected_down_tile_m_text.c_str()) &&
                       tag_equals("down_tile_n", expected_down_tile_n_text.c_str()) &&
                       tag_equals("row_tile", expected_common_row_tile.c_str()) &&
                       record.count > 0;
            });
        EXPECT_NE(production_route, records.end())
            << llaminar2::PerfStatsCollector::summaryString(
                   {"kernel.rocm_moe_grouped_prefill_batch_invariant_calls"});
        printResult(routed);
    }
#endif
}

TEST(Perf__MoEVerifierPrefill, ROCm_M256_UniformExpertBatchInvariantEconomy)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    /*
     * The skewed long-prefill case above resembles prompts whose router reuses a
     * compact hot expert set. This companion fixture distributes 2,048 routes
     * evenly across all 256 experts, leaving eight rows per expert. It guards the
     * opposite end of the occupancy surface: every descriptor is live, every
     * expert gets real row-tiled work, and the kernel cannot earn an economy pass
     * solely from unusually high weight reuse.
     */
    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    auto routed = runROCmCase(
        /*shared=*/false,
        /*rows=*/256,
        /*routed_top_k=*/8,
        /*routed_num_experts=*/256,
        /*case_name_override=*/"routed_uniform_experts",
        /*unique_routes=*/true);
    expectClose(routed.metrics);
    expectGraphReplayFasterThanReference(routed);
    if (isGfx906Device())
    {
        EXPECT_GE(routedExpertPipelineGops(routed), 6500.0)
            << "gfx906 uniform-expert M=256 prefill must sustain at least 6.5 TOPS";
    }
    printResult(routed);
#endif
}

TEST(Perf__MoEVerifierPrefill, ROCm_M24832256_BatchInvariantRouterEconomy)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    std::cout << "backend,case,m,graph_ms,router_int8_gops,rowwise_ms,speedup,index_bit_mismatches,"
                 "weight_bit_mismatches\n";
    for (int rows : {2, 4, 8, 32, 256})
    {
        SCOPED_TRACE(rows);
        const RouterBenchResult result = runROCmBatchInvariantRouterCase(rows);
        EXPECT_EQ(result.index_bit_mismatches, 0u);
        EXPECT_EQ(result.weight_bit_mismatches, 0u);
        ASSERT_GT(result.graph_ms, 0.0);
        ASSERT_GT(result.rowwise_ms, 0.0);
        EXPECT_LT(result.graph_ms, result.rowwise_ms)
            << "M=" << rows << " grouped router must beat serial M=1 routing";
        const double router_gops = routerPipelineGops(result);
        if (isGfx906Device() && rows >= 256)
        {
            EXPECT_GE(router_gops, 1300.0)
                << "gfx906 M=256 grouped Q8 router must sustain at least 1.3 TOPS end-to-end";
        }
        else if (isGfx906Device() && rows >= 32)
        {
            EXPECT_GE(router_gops, 500.0)
                << "gfx906 M=32 grouped Q8 router must sustain at least 0.5 TOPS end-to-end";
        }
        std::cout << std::fixed << std::setprecision(6)
                  << "rocm,batch_invariant_q8_router," << rows << ','
                  << result.graph_ms << ',' << router_gops << ',' << result.rowwise_ms << ','
                  << (result.rowwise_ms / result.graph_ms) << ','
                  << result.index_bit_mismatches << ','
                  << result.weight_bit_mismatches << '\n';
    }
#endif
}

TEST(Perf__MoEVerifierPrefill, ROCm_M1234_SharedExpertFFNDecodeEquivalent)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    for (int rows : {1, 2, 3, 4})
    {
        auto shared = runROCmCase(/*shared=*/true, rows);
        expectClose(shared.metrics);
        if (rows >= 2)
            expectSharedExpertFfnEconomical(shared);
        printResult(shared);
    }
#endif
}

TEST(Perf__MoEVerifierPrefill, ROCm_M234_SharedExpertFFNStageDecodeEquivalent)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    const QuantFormatCase &format = sharedExpertPreparedFormatCase("IQ3_S");
    for (int rows : {2, 3, 4})
    {
        auto shared = runROCmSharedExpertStageCase(rows, format);
        expectClose(shared.metrics);
        expectSharedExpertFfnEconomical(shared);
        printResult(shared);
    }
#endif
}

TEST(Perf__MoEVerifierPrefill, ROCm_M234_SharedExpertFFNStageAllCodebooksDecodeEquivalent)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

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
            auto shared = runROCmSharedExpertStageCase(rows, format);
            expectClose(shared.metrics);
            printResult(shared);
        }
    }
#endif
}

TEST(Perf__MoEVerifierPrefill, ROCm_M234_SharedExpertFFNStageSerialOnlyAllCodebooksFinite)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    ScopedEnvOverride iters_env("LLAMINAR_MOE_VERIFIER_PREFILL_ITERS", "8");
    for (const QuantFormatCase &format : sharedExpertPreparedFormatCases())
    {
        if (!envCsvContainsOrUnset("LLAMINAR_MOE_VERIFIER_PREFILL_FORMATS", format.name))
            continue;
        SCOPED_TRACE(format.name);
        for (int rows : {2, 3, 4})
        {
            SCOPED_TRACE(rows);
            auto serial = runROCmSharedExpertStageSerialOnlyCase(rows, format);
            expectClose(serial.metrics);
            printResult(serial);
        }
    }
#endif
}

TEST(Perf__MoEVerifierPrefill, ROCm_M1_SharedExpertFFNStageDisabledGroupedDecodeFinite)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    const QuantFormatCase &format = sharedExpertPreparedFormatCase("IQ3_S");
    auto shared = runROCmSharedExpertStageM1DisabledShortcutCase(format);
    expectClose(shared.metrics);
    printResult(shared);
#endif
}

TEST(Perf__MoEVerifierPrefill, ROCm_M1_SharedExpertFFNStageGroupedVerifierDecodeEquivalent)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    const QuantFormatCase &format = sharedExpertPreparedFormatCase("IQ3_S");
    auto shared = runROCmSharedExpertStageM1GroupedVerifierCase(format);
    expectClose(shared.metrics);
    printResult(shared);
#endif
}

TEST(Perf__MoEVerifierPrefill, ROCm_M4_CombinedRoutedSharedUpperBound)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    /*
     * Keep this paired with CUDA_M4_CombinedRoutedSharedUpperBound.  It models
     * the production verifier trick of folding routed top-8 plus the shared
     * expert into a single top-9 grouped-prefill pipeline, using deterministic
     * unique routes so the active-slot count is stable across runs.
     */
    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    auto combined = runROCmCase(
        /*shared=*/false,
        /*rows=*/4,
        /*routed_top_k=*/9,
        /*routed_num_experts=*/257,
        /*case_name_override=*/"combined_top9_upper_bound",
        /*unique_routes=*/true,
        /*include_terminal_expert=*/true);
    expectClose(combined.metrics);
    expectGraphReplayFasterThanReference(combined);
    printResult(combined);
#endif
}

TEST(Perf__MoEVerifierPrefill, ROCm_M4M9M31_CombinedTop9AllFormatsDecodeEquivalent)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    /*
     * M=4 covers the direct small-verifier projection, while M=9 and M=31
     * enter the expert-row-tiled path and its 512-entry device directory.
     * Keep timing to one replay because the acceptance condition here is native
     * byte equality for every model-loadable format; the dedicated speedometer
     * above retains the statistically useful economy window.
     */
    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    ScopedEnvOverride iters_env("LLAMINAR_MOE_VERIFIER_PREFILL_ITERS", "1");
    ScopedEnvOverride warmups_env("LLAMINAR_MOE_VERIFIER_PREFILL_WARMUPS", "0");
    ScopedEnvOverride rowwise_env(
        "LLAMINAR_MOE_VERIFIER_PREFILL_ROWWISE_ITERS", "1");
    for (const auto &format : llaminar2::test::quantizedVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        for (const int rows : {4, 9, 31})
        {
            SCOPED_TRACE(rows);
            auto combined = runROCmCase(
                /*shared=*/false,
                rows,
                /*routed_top_k=*/9,
                /*routed_num_experts=*/257,
                /*case_name_override=*/"combined_top9_all_formats",
                /*unique_routes=*/false,
                /*include_terminal_expert=*/true,
                /*d_model=*/2048,
                /*intermediate=*/512,
                &format,
                &format);
            expectClose(combined.metrics);
            expectGraphReplayFasterThanReference(combined);
        }
    }
#endif
}

TEST(Perf__MoEVerifierPrefill, ROCm_AllFormatAspectRatioMGroupedPrefillDispatchTrainer)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";
    if (envInt("LLAMINAR_ROCM_MOE_PREFILL_SWEEP", 0) == 0)
    {
        GTEST_SKIP()
            << "Set LLAMINAR_ROCM_MOE_PREFILL_SWEEP=1 to run the exhaustive trainer";
    }

    const std::vector<int> m_values = envCsvInts(
        "LLAMINAR_ROCM_MOE_PREFILL_SWEEP_M",
        {12, 16, 24, 32, 64, 128, 256});
    const int warmups = envInt(
        "LLAMINAR_ROCM_MOE_PREFILL_SWEEP_WARMUPS", 5);
    const int iterations = envInt(
        "LLAMINAR_ROCM_MOE_PREFILL_SWEEP_ITERS", 8);
    const int timing_trials = envInt(
        "LLAMINAR_ROCM_MOE_PREFILL_SWEEP_TRIALS", 30);
    ASSERT_GE(warmups, 0);
    ASSERT_GT(iterations, 0);
    ASSERT_GT(timing_trials, 0);
    const int max_cases = envInt(
        "LLAMINAR_ROCM_MOE_PREFILL_SWEEP_MAX_CASES",
        std::numeric_limits<int>::max());
    ScopedEnvOverride rowwise_iters(
        "LLAMINAR_MOE_VERIFIER_PREFILL_ROWWISE_ITERS", "1");
    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");

    std::FILE *csv = stdout;
    bool owns_csv = false;
    if (const char *path = std::getenv("LLAMINAR_ROCM_MOE_PREFILL_SWEEP_CSV");
        path && *path)
    {
        csv = std::fopen(path, "w");
        ASSERT_NE(csv, nullptr) << "failed to open grouped-prefill trainer CSV " << path;
        owns_csv = true;
    }
    std::FILE *timing_csv = nullptr;
    if (const char *path =
            std::getenv("LLAMINAR_ROCM_MOE_PREFILL_SWEEP_TIMING_CSV");
        path && *path)
    {
        timing_csv = std::fopen(path, "w");
        ASSERT_NE(timing_csv, nullptr)
            << "failed to open grouped-prefill raw timing CSV " << path;
        std::fprintf(
            timing_csv,
            "backend,phase,source_format,source_codebook,execution_codebook,"
            "shape,role,candidate_id,m,n,k,tile_m,tile_n,sample_index,"
            "timed_replays,latency_us,latency_ms_hex\n");
    }
    std::fprintf(
        csv,
        "backend,phase,source_format,source_codebook,execution_codebook,shape,role,"
        "candidate_id,m,n,k,tile_m,tile_n,warmup_count,sample_count,timed_replays,"
        "min_us,graph_us,p95_us,mad_us,cv,pipeline_gops,bit_mismatches,"
        "first_bit_mismatch,repeat_bit_mismatches,max_abs,relative_l2,cosine,"
        "symmetric_kld,grouped_output_digest,"
        "serial_output_digest,timing_sample_digest,route_counter_ok,"
        "observed_candidate_id,is_winner\n");

    int executed_cases = 0;
    for (const auto &format : llaminar2::test::quantizedVerifierFormats())
    {
        if (!envCsvContainsOrUnset(
                "LLAMINAR_ROCM_MOE_PREFILL_SWEEP_FORMATS", format.label))
        {
            continue;
        }
        for (const auto &shape : kMoEPrefillSweepShapes)
        {
            if (!envCsvContainsOrUnset(
                    "LLAMINAR_ROCM_MOE_PREFILL_SWEEP_SHAPES", shape.name))
            {
                continue;
            }
            if (executed_cases >= max_cases)
                break;

            SCOPED_TRACE(std::string(format.label) + "/" + shape.name);
            std::fprintf(
                stderr,
                "[ROCm MoE grouped-prefill sweep] format=%s codebook=%u "
                "shape=%s d_model=%d intermediate=%d\n",
                format.label,
                static_cast<unsigned>(format.device_execution_codebook_id),
                shape.name,
                shape.d_model,
                shape.intermediate);
            runROCmMoEPrefillFormatShapeSweep(
                format,
                shape,
                m_values,
                warmups,
                iterations,
                timing_trials,
                csv,
                timing_csv);
            ++executed_cases;
        }
        if (executed_cases >= max_cases)
            break;
    }

    if (owns_csv)
        ASSERT_EQ(std::fclose(csv), 0);
    if (timing_csv)
        ASSERT_EQ(std::fclose(timing_csv), 0);
    EXPECT_GT(executed_cases, 0)
        << "grouped-prefill trainer filters selected no format/shape cases";
#endif
}
