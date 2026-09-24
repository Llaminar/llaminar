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

#ifdef HAVE_ROCM
#include "backends/rocm/HIPGraphCapture.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#include "kernels/rocm/moe/ROCmMoEKernel.h"
#include "../native_vnni_dispatch/GPUTrainerVerification.h"

extern "C" bool rocmMoE_grouped_prefill_query_tile_config(
    uint8_t codebook_id,
    int projection_role,
    int m,
    int n,
    int k,
    int *tile_m,
    int *tile_n);
extern "C" bool rocmMoE_grouped_prefill_query_kernel_resources(
    uint8_t codebook_id,
    int projection_role,
    int tile_m,
    int tile_n,
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
#include <iostream>
#include <limits>
#include <memory>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <set>
#include <string>
#include <stdexcept>
#include <utility>
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

    /**
     * @brief Select the production ROCm router-Q8 publication policy.
     *
     * Shared-expert verifier performance is meaningful only when its input is
     * the exact Q8 row publication produced by the routed router stage.  This
     * scope keeps that production contract independent of test order while
     * restoring every mutable debug-policy field when a case finishes.
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

    /** @brief Read a finite positive floating-point trainer control. */
    double envPositiveDouble(const char *name, double fallback)
    {
        const char *value = std::getenv(name);
        if (!value || !*value)
            return fallback;
        char *end = nullptr;
        const double parsed = std::strtod(value, &end);
        if (end == value || !std::isfinite(parsed) || parsed <= 0.0)
            return fallback;
        return parsed;
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
     * @brief Build one production-shaped compact overlay-follower route packet.
     *
     * A four-participant secondary tier holding roughly 56 of 256 experts sees
     * 1.75 of each token's global top-eight routes on average. The mapped
     * packet consumer preserves those weights, compacts the locally owned
     * routes into an increasing prefix, and fills every unused slot with the
     * `expert=-1, weight=0` sentinel. This fixture reproduces that contract
     * without materializing experts owned by the other three participants.
     *
     * @param rows Physical prefill rows in the captured follower bucket.
     * @param top_k Fixed global route width; the Qwen 3.5 model uses eight.
     * @param first_expert First globally numbered expert owned by the follower.
     * @param local_expert_count Number of consecutive experts owned locally.
     * @return Row-major compact expert IDs and their unrenormalized weights.
     */
    std::pair<std::vector<float>, std::vector<float>>
    makeCompactOverlayFollowerRoutes(
        int rows,
        int top_k,
        int first_expert,
        int local_expert_count)
    {
        if (rows <= 0 || top_k != 8 || first_expert < 0 ||
            local_expert_count <= 0 ||
            first_expert + local_expert_count > 256)
        {
            throw std::invalid_argument(
                "compact Qwen overlay routes require positive rows, top-8, "
                "and a valid participant-local expert interval");
        }

        std::vector<float> indices(
            static_cast<size_t>(rows * top_k), -1.0f);
        std::vector<float> weights(
            static_cast<size_t>(rows * top_k), 0.0f);
        const std::vector<float> global_weights =
            makeRoutingWeights(rows, top_k);

        for (int row = 0; row < rows; ++row)
        {
            /* Three two-route rows followed by one one-route row gives the
             * exact 7/4 routes-per-token expectation for 56/256 ownership. */
            const int local_routes = (row % 4 == 3) ? 1 : 2;
            for (int local_slot = 0; local_slot < local_routes; ++local_slot)
            {
                const size_t compact =
                    static_cast<size_t>(row * top_k + local_slot);
                const int expert_offset =
                    (row * 5 + local_slot * 17) % local_expert_count;
                const int original_route_slot =
                    (row + local_slot * 3) % top_k;
                indices[compact] =
                    static_cast<float>(first_expert + expert_offset);
                weights[compact] = global_weights[
                    static_cast<size_t>(
                        row * top_k + original_route_slot)];
            }
        }
        return {std::move(indices), std::move(weights)};
    }

    /**
     * @brief Return sorted unique routed expert IDs present in a route table.
     *
     * The speedometer keeps production-sized descriptor tables but should not
     * spend minutes preparing inactive expert payloads before the timed GPU work.
     * Active slots still get real prepared descriptors and therefore hard-fail on
     * unsupported formats or broken prepared-weight wiring. `-1` is the public
     * inactive-route sentinel used by compact overlay follower packets.
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
            EXPECT_TRUE(id == -1 || (id >= 0 && id < num_experts));
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

    /**
     * @brief Own one production-shaped HIP graph-capture transaction.
     *
     * Perf cells exercise the same TensorBase publication protocol as model
     * execution. The scoped backend transaction makes provisional writes part
     * of the graph instead of attempting to publish external completion events
     * while HIP is recording. It also guarantees that an exception closes an
     * opened capture before tensors and workspaces begin destruction.
     */
    class ScopedHipPerfGraph
    {
    public:
        ScopedHipPerfGraph(
            hipStream_t stream,
            int device_ordinal,
            std::string operation)
            : graph_(stream, device_ordinal),
              transaction_(graph_, std::move(operation))
        {
            if (!stream || !transaction_.begin())
            {
                throw std::runtime_error(
                    "ScopedHipPerfGraph failed to begin HIP graph capture");
            }
        }

        ScopedHipPerfGraph(const ScopedHipPerfGraph &) = delete;
        ScopedHipPerfGraph &operator=(const ScopedHipPerfGraph &) = delete;

        /** @brief Finish capture and instantiate its immutable executable. */
        bool finishAndInstantiate()
        {
            transaction_.finish();
            return graph_.instantiate();
        }

        /** @brief Enqueue one replay on the captured non-null stream. */
        bool launch()
        {
            return graph_.launch();
        }

    private:
        llaminar2::HIPGraphCapture graph_;
        llaminar2::ScopedBackendGraphCapture transaction_;
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
        llaminar2::DeviceId device,
        double *avg_ms)
    {
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
                std::vector<float> expert_ids(static_cast<size_t>(top_k));
                std::vector<float> expert_weights(static_cast<size_t>(top_k));
                for (int k = 0; k < top_k; ++k)
                {
                    const size_t slot = static_cast<size_t>(row) * top_k + k;
                    expert_ids[static_cast<size_t>(k)] = routing_indices[slot];
                    expert_weights[static_cast<size_t>(k)] = routing_weights[slot];
                }
                auto row_expert_ids = makeTensor(
                    {1, static_cast<size_t>(top_k)}, expert_ids);
                auto row_expert_weights = makeTensor(
                    {1, static_cast<size_t>(top_k)}, expert_weights);
                auto decode_output = makeZeros({static_cast<size_t>(d_model)});

                TransferEngine::prepareDeviceInput(
                    row_hidden.get(), device, stream);
                TransferEngine::prepareDeviceInput(
                    row_expert_ids.get(), device, stream);
                TransferEngine::prepareDeviceInput(
                    row_expert_weights.get(), device, stream);
                TransferEngine::prepareDeviceOutput(
                    decode_output.get(), device, stream);
                requireHipBenchBody(
                    moe->groupedExpertDecodeFromRouting(
                        row_hidden.get(),
                        row_expert_ids.get(),
                        row_expert_weights.get(),
                        gateup_table,
                        down_table,
                        top_k,
                        decode_output.get(),
                        d_model,
                        intermediate),
                    "serial device-routed row oracle");
                if (hipStreamSynchronize(stream) != hipSuccess ||
                    !decode_output->ensureOnHost(stream))
                {
                    throw std::runtime_error(
                        "failed to materialize serial device-routed row oracle");
                }
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
        const llaminar2::test::QuantizedVerifierFormatCase *down_format = nullptr,
        bool canonical_route_split = false,
        const std::vector<float> *routing_indices_override = nullptr,
        const std::vector<float> *routing_weights_override = nullptr)
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
        /*
         * The grouped side consumes exactly `rows`, but the byte oracle below
         * replays one token through all top-k experts.  Its split-K gate/up
         * publication uses the production four-row verifier slab even when
         * the requested grouped M is one, two, or three.  Size the fixture for
         * that shared contract so a small-M economy case cannot accidentally
         * turn an undersized test arena into zero-valued reference output.
         */
        constexpr int kSerialOracleVerifierCapacity = 4;
        auto reqs = llaminar2::MoEWorkspaceBuffers::rocmMoE(
            /*max_seq_len=*/std::max(rows, kSerialOracleVerifierCapacity),
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
        if ((routing_indices_override == nullptr) !=
            (routing_weights_override == nullptr))
        {
            throw std::invalid_argument(
                "ROCm MoE route overrides must provide IDs and weights together");
        }
        auto routing_indices = routing_indices_override
                                   ? *routing_indices_override
                                   : (unique_routes
                                          ? makeUniqueRoutingIndices(
                                                rows, top_k, num_experts)
                                          : makeRoutingIndices(
                                                rows, top_k, num_experts));
        if (include_terminal_expert)
            publishTerminalExpertRoute(routing_indices, rows, top_k, num_experts);
        auto routing_weights = routing_weights_override
                                   ? *routing_weights_override
                                   : makeRoutingWeights(rows, top_k);
        const size_t expected_route_values =
            static_cast<size_t>(rows * top_k);
        if (routing_indices.size() != expected_route_values ||
            routing_weights.size() != expected_route_values)
        {
            throw std::invalid_argument(
                "ROCm MoE route override geometry does not match rows*top_k");
        }
        auto tables = prepareExpertTables(
            moe.get(), device, num_experts, d_model, intermediate,
            uniqueExpertIdsFromRoutes(routing_indices, num_experts),
            selected_gateup_format,
            selected_down_format);
        auto hidden = makeTensor({static_cast<size_t>(rows), static_cast<size_t>(d_model)}, hidden_values);
        auto route_indices_tensor = makeTensor({static_cast<size_t>(rows), static_cast<size_t>(top_k)}, routing_indices);
        auto route_weights_tensor = makeTensor({static_cast<size_t>(rows), static_cast<size_t>(top_k)}, routing_weights);
        auto grouped_output = makeZeros({static_cast<size_t>(rows), static_cast<size_t>(d_model)});
        std::shared_ptr<llaminar2::FP32Tensor> canonical_route_contributions;
        if (canonical_route_split)
        {
            canonical_route_contributions = makeZeros(
                {static_cast<size_t>(rows * top_k),
                 static_cast<size_t>(d_model)});
        }
        EXPECT_TRUE(hidden->ensureOnDevice(device, stream));
        EXPECT_TRUE(route_indices_tensor->ensureOnDevice(device, stream));
        EXPECT_TRUE(route_weights_tensor->ensureOnDevice(device, stream));
        EXPECT_TRUE(grouped_output->ensureOnDevice(device, stream));
        if (canonical_route_contributions)
        {
            EXPECT_TRUE(canonical_route_contributions->ensureOnDevice(
                device, stream));
        }

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
            if (!moe->executeGroupedPrefillPipeline(
                    hidden.get(), grouped_output.get(),
                    tables.gateup_table_id, tables.down_table_id,
                    rows, d_model, intermediate, num_experts, top_k,
                    canonical_route_contributions.get()))
            {
                return false;
            }
            if (!canonical_route_contributions)
                return true;

            /*
             * This is the same graph-capturable producer/reducer contract used
             * by LocalTP after its rooted collective. For one device there is
             * no collective between the two launches: the experiment isolates
             * whether exposing route-parallel down dots outweighs one compact,
             * deterministic router-order reduction.
             */
            return moe->reduceCanonicalRouteContributions(
                canonical_route_contributions.get(),
                grouped_output.get(),
                rows,
                top_k,
                d_model);
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

        ScopedHipPerfGraph graph(
            stream,
            /*device_ordinal=*/0,
            "ROCm MoE routed verifier perf capture");
        requireHipBenchBody(run_grouped(), "graph capture");
        if (!graph.finishAndInstantiate())
        {
            throw std::runtime_error(
                "ROCm MoE verifier graph instantiate failed");
        }
        for (int i = 0; i < warmups; ++i)
            requireHipBenchBody(graph.launch(), "graph warmup replay");
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        const double graph_ms = timeHipEvents(
            stream,
            iterations,
            [&]()
            {
                return graph.launch();
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
            tables.gateup_table_id, tables.down_table_id, device, &rowwise_ms);
        CloseMetrics metrics = compareVectors(grouped, rowwise, grouped.size());

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
     * @brief Timing and byte-parity evidence for one overlay follower decode.
     *
     * `valid_routes` models the subset of a global top-8 row assigned to one
     * overlay follower.  The mapped dispatch consumer compacts those routes
     * into a valid prefix and publishes `expert=-1, weight=0` in every tail
     * slot.  Keeping that packet geometry here is essential: sparse global
     * route positions are not the tensors consumed by the follower graph.
     */
    struct OverlayFollowerDecodeResult
    {
        int valid_routes = 0;
        double eager_ms = 0.0;
        double graph_ms = 0.0;
        double canonical_oracle_ms = 0.0;
        CloseMetrics metrics;
    };

    /**
     * @brief Measure the exact captured Qwen 122B overlay follower entrypoint.
     *
     * Unlike `runROCmCase()`, this function does not route grouped-prefill rows
     * through `executeGroupedPrefillPipeline()`.  It invokes the same
     * `groupedExpertDecodeFromRouting()` contract used by a production mapped
     * M=1 follower stage, including its fixed top-8 geometry, participant mask,
     * direct output fold, and compact prefix of participant-local route slots.
     *
     * @param valid_routes Number of the eight route slots owned by this
     *                     participant; must be one or two for the canonical
     *                     four-device secondary tier.
     * @return Timings plus byte comparison against canonical per-route
     *         publication followed by the production ordered reducer.
     */
    OverlayFollowerDecodeResult runROCmOverlayFollowerDecodeCase(
        int valid_routes)
    {
        constexpr int d_model = 3072;
        constexpr int intermediate = 1024;
        constexpr int num_experts = 256;
        constexpr int top_k = 8;
        if (valid_routes < 1 || valid_routes > 2)
        {
            throw std::invalid_argument(
                "overlay follower decode requires one or two valid routes");
        }

        const int iterations = envInt(
            "LLAMINAR_MOE_OVERLAY_FOLLOWER_DECODE_ITERS", 400);
        const int warmups = envInt(
            "LLAMINAR_MOE_OVERLAY_FOLLOWER_DECODE_WARMUPS", 40);
        const auto device = llaminar2::DeviceId::rocm(0);
        const auto &q8_0 = llaminar2::test::quantizedVerifierFormat("Q8_0");

        EXPECT_EQ(hipSetDevice(0), hipSuccess);
        hipStream_t stream = nullptr;
        EXPECT_EQ(
            hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
            hipSuccess);

        auto moe = KernelFactory::createMoEKernel(device);
        EXPECT_NE(moe, nullptr);
        moe->setGPUStream(stream);
        auto *workspace_consumer =
            dynamic_cast<llaminar2::IWorkspaceConsumer *>(moe.get());
        EXPECT_NE(workspace_consumer, nullptr);
        auto requirements = llaminar2::MoEWorkspaceBuffers::rocmMoE(
            /*max_seq_len=*/1,
            d_model,
            intermediate,
            num_experts,
            top_k);
        auto workspace = std::make_unique<llaminar2::DeviceWorkspaceManager>(
            device,
            requirements.total_bytes_with_alignment() + 8 * 1024 * 1024);
        EXPECT_TRUE(workspace->allocate(requirements));
        workspace_consumer->bindWorkspace(workspace.get());

        std::vector<int> materialized_experts;
        for (int route = 0; route < valid_routes; ++route)
            materialized_experts.push_back(route);
        auto tables = prepareExpertTables(
            moe.get(),
            device,
            num_experts,
            d_model,
            intermediate,
            materialized_experts,
            q8_0,
            q8_0);

        auto hidden = makeTensor(
            {1, static_cast<size_t>(d_model)},
            makeHiddenValues(/*rows=*/1, d_model));
        auto output = makeZeros({1, static_cast<size_t>(d_model)});
        auto canonical_output = makeZeros(
            {1, static_cast<size_t>(d_model)});
        auto canonical_routes = makeZeros(
            {static_cast<size_t>(top_k), static_cast<size_t>(d_model)});

        std::vector<float> expert_ids(static_cast<size_t>(top_k), -1.0f);
        std::vector<float> expert_weights(static_cast<size_t>(top_k), 0.0f);
        const std::vector<float> global_expert_weights =
            makeRoutingWeights(/*rows=*/1, top_k);
        static constexpr std::array<int, 2> kGlobalRouteSlots = {1, 5};
        std::vector<uint8_t> expert_mask(
            static_cast<size_t>(num_experts), 0u);
        for (int route = 0; route < valid_routes; ++route)
        {
            const size_t compact_slot = static_cast<size_t>(route);
            expert_ids[compact_slot] = static_cast<float>(route);
            expert_weights[compact_slot] = global_expert_weights[
                static_cast<size_t>(kGlobalRouteSlots[route])];
            expert_mask[compact_slot] = 1u;
        }
        auto routing_indices = makeTensor(
            {1, static_cast<size_t>(top_k)}, expert_ids);
        auto routing_weights = makeTensor(
            {1, static_cast<size_t>(top_k)}, expert_weights);

        EXPECT_TRUE(hidden->ensureOnDevice(device, stream));
        EXPECT_TRUE(output->ensureOnDevice(device, stream));
        EXPECT_TRUE(canonical_output->ensureOnDevice(device, stream));
        EXPECT_TRUE(canonical_routes->ensureOnDevice(device, stream));
        EXPECT_TRUE(routing_indices->ensureOnDevice(device, stream));
        EXPECT_TRUE(routing_weights->ensureOnDevice(device, stream));

        EXPECT_TRUE(moe->prepareGroupedRuntimeDecodeLaunchState(
            tables.gateup_table_id,
            tables.down_table_id,
            top_k,
            d_model,
            intermediate,
            llaminar2::MoEDecodeDescriptorSource::StaticDescriptorTable));

        auto run_production = [&]()
        {
            return moe->groupedExpertDecodeFromRouting(
                hidden.get(),
                routing_indices.get(),
                routing_weights.get(),
                tables.gateup_table_id,
                tables.down_table_id,
                top_k,
                output.get(),
                d_model,
                intermediate,
                expert_mask.data(),
                /*canonical_route_contributions=*/nullptr);
        };
        auto run_canonical_oracle = [&]()
        {
            if (!moe->groupedExpertDecodeFromRouting(
                hidden.get(),
                routing_indices.get(),
                routing_weights.get(),
                tables.gateup_table_id,
                tables.down_table_id,
                top_k,
                output.get(),
                d_model,
                intermediate,
                expert_mask.data(),
                canonical_routes.get()))
            {
                return false;
            }
            return moe->reduceCanonicalRouteContributions(
                canonical_routes.get(),
                canonical_output.get(),
                /*seq_len=*/1,
                top_k,
                d_model);
        };

        for (int warmup = 0; warmup < warmups; ++warmup)
            requireHipBenchBody(
                run_production(), "overlay follower warmup");
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        const double eager_ms = timeHipEvents(
            stream, iterations, run_production);
        const double canonical_oracle_ms = timeHipEvents(
            stream, iterations, run_canonical_oracle);

        double graph_ms = 0.0;
        {
            ScopedHipPerfGraph graph(
                stream,
                /*device_ordinal=*/0,
                "ROCm Qwen 122B overlay follower decode capture");
            requireHipBenchBody(
                run_production(), "overlay follower graph capture");
            if (!graph.finishAndInstantiate())
            {
                throw std::runtime_error(
                    "ROCm overlay follower decode graph instantiate failed");
            }
            for (int warmup = 0; warmup < warmups; ++warmup)
                requireHipBenchBody(
                    graph.launch(), "overlay follower graph warmup");
            EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
            graph_ms = timeHipEvents(
                stream,
                iterations,
                [&]() { return graph.launch(); });
            EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        }

        requireHipBenchBody(
            run_canonical_oracle(), "overlay follower byte oracle");
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        TransferEngine::publishDeviceWrite(output, device, stream);
        TransferEngine::publishDeviceWrite(canonical_output, device, stream);
        const std::vector<float> production_values(
            output->data(), output->data() + output->numel());
        const std::vector<float> canonical_values(
            canonical_output->data(),
            canonical_output->data() + canonical_output->numel());
        const CloseMetrics metrics = compareVectors(
            production_values,
            canonical_values,
            static_cast<size_t>(d_model));

        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
        return {
            .valid_routes = valid_routes,
            .eager_ms = eager_ms,
            .graph_ms = graph_ms,
            .canonical_oracle_ms = canonical_oracle_ms,
            .metrics = metrics,
        };
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
        constexpr int routed_experts = 256;
        constexpr int routed_top_k = 8;
        const int iterations = envInt("LLAMINAR_MOE_VERIFIER_PREFILL_ITERS", 120);
        const int warmups = envInt("LLAMINAR_MOE_VERIFIER_PREFILL_WARMUPS", 5);
        const auto device = llaminar2::DeviceId::rocm(0);
        ScopedROCmBatchInvariantRouterPolicy router_policy;

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
        auto router_moe = KernelFactory::createMoEKernel(device);
        EXPECT_NE(router_moe, nullptr);
        router_moe->setGPUStream(stream);
        auto *moe_workspace = dynamic_cast<llaminar2::IWorkspaceConsumer *>(moe.get());
        EXPECT_NE(moe_workspace, nullptr);
        auto *router_workspace =
            dynamic_cast<llaminar2::IWorkspaceConsumer *>(router_moe.get());
        EXPECT_NE(router_workspace, nullptr);

        /*
         * Production gives routed and shared experts independent backend
         * objects. Their only shared execution state is this graph-local Q8
         * publication, whose stable device addresses are populated by the
         * routed producer before the shared consumer is captured.
         */
        auto router_q8_publication =
            std::make_shared<llaminar2::MoERouterQ8HiddenPublication>();
        EXPECT_TRUE(router_moe->bindRouterQ8HiddenPublication(
            router_q8_publication,
            llaminar2::MoERouterQ8PublicationAccess::ProducerAndConsumer));
        EXPECT_TRUE(moe->bindRouterQ8HiddenPublication(
            router_q8_publication,
            llaminar2::MoERouterQ8PublicationAccess::RequiredConsumer));
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
        auto gate_scratch = makeZeros(
            {static_cast<size_t>(rows), static_cast<size_t>(intermediate)});
        auto up_scratch = makeZeros(
            {static_cast<size_t>(rows), static_cast<size_t>(intermediate)});
        EXPECT_TRUE(hidden->ensureOnDevice(device, stream));
        EXPECT_TRUE(grouped_output->ensureOnDevice(device, stream));
        EXPECT_TRUE(gate_scratch->ensureOnDevice(device, stream));
        EXPECT_TRUE(up_scratch->ensureOnDevice(device, stream));

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
            params.gate_scratch = gate_scratch.get();
            params.up_scratch = up_scratch.get();
            params.seq_len = rows;
            params.d_model = d_model;
            params.intermediate = intermediate;
            params.prepared_ref_gate = prepared.gate_ref;
            params.prepared_ref_up = prepared.up_ref;
            params.prepared_ref_down = prepared.down_ref;
            params.prepared_store = prepared.store.get();
            params.force_grouped_verifier_prefill_for_decode = grouped_verifier;
            params.force_decode_equivalent_verifier_prefill = false;
            if (grouped_verifier)
                params.required_router_q8_publication = router_q8_publication;
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
        if (router_workspace)
            router_workspace->bindWorkspace(workspace.get());

        /*
         * Materialize the exact producer payload once, outside the isolated
         * shared-stage timing window. Inputs are immutable in this kernel
         * speedometer, so every graph replay consumes the same bytes and
         * addresses that a routed producer node would publish immediately
         * before it in the complete inference graph.
         */
        std::vector<float> router_gate_values(
            static_cast<size_t>(routed_experts) * d_model);
        for (size_t i = 0; i < router_gate_values.size(); ++i)
        {
            router_gate_values[i] =
                0.017f * std::sin(0.0017f * static_cast<float>(i + 29)) +
                0.011f * std::cos(0.0023f * static_cast<float>(i + 47));
        }
        auto router_gate = makeTensor(
            {static_cast<size_t>(routed_experts), static_cast<size_t>(d_model)},
            router_gate_values);
        auto router_indices = makeZeros(
            {static_cast<size_t>(rows), static_cast<size_t>(routed_top_k)});
        auto router_weights = makeZeros(
            {static_cast<size_t>(rows), static_cast<size_t>(routed_top_k)});
        EXPECT_TRUE(router_gate->ensureOnDevice(device, stream));
        EXPECT_TRUE(router_indices->ensureOnDevice(device, stream));
        EXPECT_TRUE(router_weights->ensureOnDevice(device, stream));
        EXPECT_TRUE(router_moe->routeVerifierRowsDecodeEquivalent(
            hidden.get(),
            router_gate.get(),
            rows,
            d_model,
            routed_experts,
            routed_top_k,
            /*normalize_weights=*/true,
            router_indices.get(),
            router_weights.get()));
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

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

        ScopedHipPerfGraph graph(
            stream,
            /*device_ordinal=*/0,
            "ROCm MoE shared-expert verifier perf capture");
        requireHipBenchBody(run_grouped(), "shared graph capture");
        requireHipBenchBody(
            graph.finishAndInstantiate(),
            "shared graph instantiate");
        for (int i = 0; i < warmups; ++i)
            requireHipBenchBody(graph.launch(), "shared graph warmup replay");
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        const double graph_ms = timeHipEvents(
            stream,
            iterations,
            [&]()
            {
                return graph.launch();
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
        if (router_workspace)
            router_workspace->unbindWorkspace();
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

        ScopedHipPerfGraph graph(
            stream,
            /*device_ordinal=*/0,
            "ROCm MoE router verifier perf capture");
        requireHipBenchBody(run_grouped(), "router graph capture");
        requireHipBenchBody(
            graph.finishAndInstantiate(),
            "router graph instantiate");
        for (int i = 0; i < 3; ++i)
            requireHipBenchBody(graph.launch(), "router graph warmup replay");
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        const double graph_ms = timeHipEvents(
            stream,
            iterations,
            [&]()
            {
                return graph.launch();
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

    constexpr std::array<MoEPrefillSweepCandidate, 13> kMoEPrefillSweepCandidates{{
        {4, 64}, {4, 128}, {4, 256},
        {8, 64}, {8, 128}, {8, 256},
        {12, 64}, {12, 128}, {12, 256},
        {16, 64}, {16, 128}, {16, 256},
        {20, 128},
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

    /** Immutable compiler-resource evidence for one exact HIP specialization. */
    struct MoEPrefillKernelResources
    {
        int registers_per_thread = 0;
        size_t local_memory_bytes_per_thread = 0;
        size_t static_shared_memory_bytes = 0;
        int max_threads_per_block = 0;
        int max_active_blocks_per_sm = 0;

        /** @return true when the compiler emitted no per-thread scratch. */
        [[nodiscard]] bool spillFree() const noexcept
        {
            return local_memory_bytes_per_thread == 0;
        }
    };

    /**
     * @brief Inspect one exact candidate without launching it.
     *
     * A failed query is fatal because silently timing an uninspected candidate
     * would make the spill-free tournament policy unverifiable.
     */
    MoEPrefillKernelResources queryMoEPrefillKernelResources(
        uint8_t execution_codebook,
        int projection_role,
        const MoEPrefillSweepCandidate &candidate)
    {
        MoEPrefillKernelResources resources{};
        const bool queried = rocmMoE_grouped_prefill_query_kernel_resources(
            execution_codebook,
            projection_role,
            candidate.tile_m,
            candidate.tile_n,
            &resources.registers_per_thread,
            &resources.local_memory_bytes_per_thread,
            &resources.static_shared_memory_bytes,
            &resources.max_threads_per_block,
            &resources.max_active_blocks_per_sm);
        if (!queried)
        {
            throw std::runtime_error(
                "failed to inspect compiled ROCm MoE grouped-prefill candidate");
        }
        return resources;
    }

    /** @brief Return the stable registry identity for one tile candidate. */
    std::string moePrefillCandidateName(
        const MoEPrefillSweepCandidate &candidate)
    {
        return "tm" + std::to_string(candidate.tile_m) +
               "_tn" + std::to_string(candidate.tile_n);
    }

    /** One independently forceable gate/up plus down candidate pair. */
    struct MoEPrefillCandidatePair
    {
        MoEPrefillSweepCandidate gateup;
        MoEPrefillSweepCandidate down;

        /** @brief Return a stable, registry-compatible pair identity. */
        std::string id() const
        {
            return "g_" + moePrefillCandidateName(gateup) +
                   "__d_" + moePrefillCandidateName(down);
        }
    };

    /**
     * @brief Force one exact production candidate pair without reparsing env.
     *
     * The selector reads the typed DebugEnv snapshot at capture time. Updating
     * those four reviewed trainer controls directly avoids four `setenv` plus
     * configuration reload transactions for every candidate pair.
     */
    class ScopedROCmMoEPrefillCandidatePair
    {
    public:
        explicit ScopedROCmMoEPrefillCandidatePair(
            const MoEPrefillCandidatePair &candidate)
            : old_gateup_m_(llaminar2::mutableDebugEnv().rocm.moe_prefill_gateup_tile_m),
              old_gateup_n_(llaminar2::mutableDebugEnv().rocm.moe_prefill_gateup_tile_n),
              old_down_m_(llaminar2::mutableDebugEnv().rocm.moe_prefill_down_tile_m),
              old_down_n_(llaminar2::mutableDebugEnv().rocm.moe_prefill_down_tile_n)
        {
            auto &config = llaminar2::mutableDebugEnv().rocm;
            config.moe_prefill_gateup_tile_m = candidate.gateup.tile_m;
            config.moe_prefill_gateup_tile_n = candidate.gateup.tile_n;
            config.moe_prefill_down_tile_m = candidate.down.tile_m;
            config.moe_prefill_down_tile_n = candidate.down.tile_n;
        }

        ScopedROCmMoEPrefillCandidatePair(
            const ScopedROCmMoEPrefillCandidatePair &) = delete;
        ScopedROCmMoEPrefillCandidatePair &operator=(
            const ScopedROCmMoEPrefillCandidatePair &) = delete;

        ~ScopedROCmMoEPrefillCandidatePair()
        {
            auto &config = llaminar2::mutableDebugEnv().rocm;
            config.moe_prefill_gateup_tile_m = old_gateup_m_;
            config.moe_prefill_gateup_tile_n = old_gateup_n_;
            config.moe_prefill_down_tile_m = old_down_m_;
            config.moe_prefill_down_tile_n = old_down_n_;
        }

    private:
        int old_gateup_m_ = 0;
        int old_gateup_n_ = 0;
        int old_down_m_ = 0;
        int old_down_n_ = 0;
    };

    /** Reusable HIP event pair for allocation-free candidate timing. */
    class MoEPrefillEventTimer
    {
    public:
        explicit MoEPrefillEventTimer(hipStream_t stream)
            : stream_(stream)
        {
            if (!stream_)
            {
                throw std::runtime_error(
                    "ROCm MoE prefill timer requires a non-null stream");
            }
            if (hipEventCreate(&start_) != hipSuccess)
            {
                throw std::runtime_error(
                    "failed to create ROCm MoE prefill start event");
            }
            if (hipEventCreate(&stop_) != hipSuccess)
            {
                (void)hipEventDestroy(start_);
                start_ = nullptr;
                throw std::runtime_error(
                    "failed to create ROCm MoE prefill stop event");
            }
        }

        MoEPrefillEventTimer(const MoEPrefillEventTimer &) = delete;
        MoEPrefillEventTimer &operator=(const MoEPrefillEventTimer &) = delete;

        ~MoEPrefillEventTimer()
        {
            if (stop_)
                (void)hipEventDestroy(stop_);
            if (start_)
                (void)hipEventDestroy(start_);
        }

        /** @brief Time repeated graph replay and return milliseconds per replay. */
        double sample(
            int iterations,
            const std::function<bool()> &launch)
        {
            if (iterations <= 0 || !launch)
                throw std::invalid_argument("MoE timer requires positive iterations");
            if (hipEventRecord(start_, stream_) != hipSuccess)
                throw std::runtime_error("failed to record MoE timer start");
            for (int iteration = 0; iteration < iterations; ++iteration)
                requireHipBenchBody(launch(), "candidate timing replay");
            if (hipEventRecord(stop_, stream_) != hipSuccess ||
                hipEventSynchronize(stop_) != hipSuccess)
            {
                throw std::runtime_error("failed to complete MoE timer sample");
            }
            float elapsed_ms = 0.0f;
            if (hipEventElapsedTime(&elapsed_ms, start_, stop_) != hipSuccess)
                throw std::runtime_error("failed to read MoE timer sample");
            return static_cast<double>(elapsed_ms) /
                   static_cast<double>(iterations);
        }

    private:
        hipStream_t stream_ = nullptr;
        hipEvent_t start_ = nullptr;
        hipEvent_t stop_ = nullptr;
    };

    /**
     * @brief Persistent device counters for one sweep cell's byte certificates.
     */
    class MoEPrefillDeviceByteCertificate
    {
    public:
        MoEPrefillDeviceByteCertificate()
        {
            if (hipMalloc(&mismatch_count_, sizeof(uint64_t)) != hipSuccess)
            {
                throw std::runtime_error(
                    "failed to allocate persistent MoE mismatch counter");
            }
            if (hipMalloc(&first_mismatch_, sizeof(uint64_t)) != hipSuccess)
            {
                (void)hipFree(mismatch_count_);
                mismatch_count_ = nullptr;
                throw std::runtime_error(
                    "failed to allocate persistent MoE first-mismatch counter");
            }
        }

        MoEPrefillDeviceByteCertificate(
            const MoEPrefillDeviceByteCertificate &) = delete;
        MoEPrefillDeviceByteCertificate &operator=(
            const MoEPrefillDeviceByteCertificate &) = delete;

        ~MoEPrefillDeviceByteCertificate()
        {
            if (first_mismatch_)
                (void)hipFree(first_mismatch_);
            if (mismatch_count_)
                (void)hipFree(mismatch_count_);
        }

        /**
         * @brief Compare two device tensors and materialize two terminal words.
         */
        std::pair<uint64_t, uint64_t> compare(
            const float *actual,
            const float *expected,
            size_t count,
            hipStream_t stream)
        {
            if (!llaminar2::test::enqueueROCmFP32ByteComparison(
                    actual,
                    expected,
                    count,
                    mismatch_count_,
                    first_mismatch_,
                    stream))
            {
                throw std::runtime_error("failed to enqueue MoE byte certificate");
            }
            uint64_t host_count = 0;
            uint64_t host_first = std::numeric_limits<uint64_t>::max();
            if (hipMemcpyAsync(
                    &host_count,
                    mismatch_count_,
                    sizeof(host_count),
                    hipMemcpyDeviceToHost,
                    stream) != hipSuccess ||
                hipMemcpyAsync(
                    &host_first,
                    first_mismatch_,
                    sizeof(host_first),
                    hipMemcpyDeviceToHost,
                    stream) != hipSuccess ||
                hipStreamSynchronize(stream) != hipSuccess)
            {
                throw std::runtime_error("failed to read MoE byte certificate");
            }
            return {host_count, host_first};
        }

    private:
        uint64_t *mismatch_count_ = nullptr;
        uint64_t *first_mismatch_ = nullptr;
    };

    /** Screening and robust timing evidence for one exact candidate pair. */
    struct MoEProductionPairEvidence
    {
        MoEPrefillCandidatePair candidate;
        MoEPrefillKernelResources gateup_resources;
        MoEPrefillKernelResources down_resources;
        std::vector<double> screening_samples_ms;
        std::vector<double> robust_samples_ms;
        int screening_replays_per_sample = 0;
        int robust_replays_per_sample = 0;
        uint64_t bit_mismatches = 0;
        uint64_t first_bit_mismatch = std::numeric_limits<uint64_t>::max();
        bool route_counter_ok = false;
        bool finalist = false;
        bool winner = false;

        /** @brief Select robust samples when available, otherwise screening. */
        const std::vector<double> &selectedSamplesMs() const
        {
            return robust_samples_ms.empty()
                       ? screening_samples_ms
                       : robust_samples_ms;
        }

        /** @brief Return replay count associated with selected samples. */
        int selectedReplaysPerSample() const
        {
            return robust_samples_ms.empty()
                       ? screening_replays_per_sample
                       : robust_replays_per_sample;
        }

        /** @brief Return the robust or screening median in microseconds. */
        double medianUs() const
        {
            const auto &samples = selectedSamplesMs();
            if (samples.empty())
                return std::numeric_limits<double>::infinity();
            return samples[samples.size() / 2] * 1000.0;
        }
    };

    /** @brief Enumerate the complete 12-by-12 production pair space. */
    std::vector<MoEPrefillCandidatePair> moePrefillCandidatePairs()
    {
        std::vector<MoEPrefillCandidatePair> result;
        result.reserve(
            kMoEPrefillSweepCandidates.size() *
            kMoEPrefillSweepCandidates.size());
        for (const auto &gateup : kMoEPrefillSweepCandidates)
        {
            for (const auto &down : kMoEPrefillSweepCandidates)
                result.push_back({gateup, down});
        }
        return result;
    }

    /**
     * @brief Resolve one exact ROCm candidate-pair corpus identity.
     *
     * An isolated hardware-counter request must identify both projection
     * geometries. Partial or unknown names are rejected so rocprof can never
     * attribute one launch to a different gate/up or down specialization.
     *
     * @param candidate_id Stable pair identity emitted by the timing corpus.
     * @return Exact registered pair when the identity is launchable.
     */
    std::optional<MoEPrefillCandidatePair> findMoEPrefillCandidatePair(
        const std::string &candidate_id)
    {
        for (const auto &candidate : moePrefillCandidatePairs())
        {
            if (candidate.id() == candidate_id)
                return candidate;
        }
        return std::nullopt;
    }

    /**
     * @brief Prove PerfStats observed both members of one forced pair.
     */
    bool observedMoEPrefillCandidatePair(
        int rows,
        const MoEPrefillCandidatePair &candidate)
    {
        const auto records = llaminar2::PerfStatsCollector::snapshot(
            {"kernel.rocm_moe_grouped_prefill_batch_invariant_calls"});
        for (const auto &record : records)
        {
            if (record.name !=
                "rocm_moe_grouped_prefill_batch_invariant_calls")
            {
                continue;
            }
            const auto tag = [&record](const char *name) -> std::string
            {
                const auto found = record.tags.find(name);
                return found == record.tags.end()
                           ? std::string{}
                           : found->second;
            };
            if (record.count > 0 &&
                tag("seq_len") == std::to_string(rows) &&
                tag("gateup_tile_m") ==
                    std::to_string(candidate.gateup.tile_m) &&
                tag("gateup_tile_n") ==
                    std::to_string(candidate.gateup.tile_n) &&
                tag("down_tile_m") ==
                    std::to_string(candidate.down.tile_m) &&
                tag("down_tile_n") ==
                    std::to_string(candidate.down.tile_n))
            {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Emit one machine-readable mixed-production tournament row.
     */
    void writeMoEProductionPairEvidence(
        std::FILE *csv,
        const llaminar2::test::native_vnni_dispatch::MoERoutedPrefillCase &routed_case,
        llaminar2::test::native_vnni_dispatch::MoERoutingProfile route_profile,
        const llaminar2::test::native_vnni_dispatch::MoERoutingProfileStats &route_stats,
        int rows,
        const MoEProductionPairEvidence &evidence)
    {
        if (!csv)
            throw std::invalid_argument("MoE production evidence requires a CSV");
        const auto &selected_samples = evidence.selectedSamplesMs();
        const auto timing =
            llaminar2::test::trainer::summarizeSortedTimingSamples(
                selected_samples);
        const auto &gate_format =
            llaminar2::test::quantizedMoEVerifierFormat(
                routed_case.routed.gate);
        const auto &up_format =
            llaminar2::test::quantizedMoEVerifierFormat(
                routed_case.routed.up);
        const auto &down_format =
            llaminar2::test::quantizedMoEVerifierFormat(
                routed_case.routed.down);
        std::ostringstream row;
        row << std::setprecision(17)
            << "rocm,moe_production_prefill,"
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
            << evidence.candidate.gateup.tile_m << ','
            << evidence.candidate.gateup.tile_n << ','
            << evidence.candidate.down.tile_m << ','
            << evidence.candidate.down.tile_n << ','
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
            << selected_samples.size() << ','
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

    /**
     * @brief Retain every raw event sample from one candidate pair.
     *
     * Screening and finalist confirmation are deliberately separate phases.
     * Keeping both prevents the robust retime from erasing evidence that was
     * already paid for and lets the adapter diagnose selection bias or clock
     * drift without rerunning a device sweep.
     */
    void writeMoEProductionPairTimingEvidence(
        std::FILE *timing_csv,
        const llaminar2::test::native_vnni_dispatch::MoERoutedPrefillCase &routed_case,
        llaminar2::test::native_vnni_dispatch::MoERoutingProfile route_profile,
        int rows,
        const MoEProductionPairEvidence &evidence)
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
                    "rocm,moe_production_prefill,%s,%s,%d,%s,%s,%zu,%d,%.9f,%a\n",
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

    /** Stable two-stage timing policy for one production-shaped sweep cell. */
    struct MoEProductionSweepSettings
    {
        int screening_warmups = 1;
        int screening_trials = 3;
        int screening_replays = 2;
        int robust_warmups = 2;
        int robust_trials = 15;
        int robust_replays = 4;
        int minimum_finalists = 12;
        int maximum_finalists = 24;
        double finalist_margin = 0.05;
        std::string profiler_request_id;
        std::optional<MoEPrefillCandidatePair> profiler_candidate;

        /** @return true when this process owns one isolated rocprof launch. */
        [[nodiscard]] bool profiling() const noexcept
        {
            return !profiler_request_id.empty();
        }
    };

    /** Own one non-default HIP stream for a production sweep cell. */
    class ScopedMoEPrefillHipStream
    {
    public:
        ScopedMoEPrefillHipStream()
        {
            if (hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking) !=
                hipSuccess)
            {
                throw std::runtime_error(
                    "failed to create ROCm MoE production sweep stream");
            }
        }

        ScopedMoEPrefillHipStream(const ScopedMoEPrefillHipStream &) = delete;
        ScopedMoEPrefillHipStream &operator=(
            const ScopedMoEPrefillHipStream &) = delete;

        ~ScopedMoEPrefillHipStream()
        {
            if (stream_)
                (void)hipStreamDestroy(stream_);
        }

        /** @brief Return the exact stream owned by this sweep cell. */
        hipStream_t get() const noexcept
        {
            return stream_;
        }

    private:
        hipStream_t stream_ = nullptr;
    };

    /** Unbind a preallocated workspace before its manager is destroyed. */
    class ScopedMoEPrefillWorkspaceBinding
    {
    public:
        ScopedMoEPrefillWorkspaceBinding(
            llaminar2::IWorkspaceConsumer *consumer,
            llaminar2::DeviceWorkspaceManager *workspace)
            : consumer_(consumer)
        {
            if (!consumer_ || !workspace)
            {
                throw std::invalid_argument(
                    "MoE production sweep requires a valid workspace binding");
            }
            consumer_->bindWorkspace(workspace);
        }

        ScopedMoEPrefillWorkspaceBinding(
            const ScopedMoEPrefillWorkspaceBinding &) = delete;
        ScopedMoEPrefillWorkspaceBinding &operator=(
            const ScopedMoEPrefillWorkspaceBinding &) = delete;

        ~ScopedMoEPrefillWorkspaceBinding()
        {
            consumer_->unbindWorkspace();
        }

    private:
        llaminar2::IWorkspaceConsumer *consumer_ = nullptr;
    };

    /**
     * @brief Train one real routed-MoE source tuple at one prefill bucket.
     *
     * Every expert owns distinct prepared weights so the tournament observes
     * realistic cache pressure instead of repeatedly reading an aliased proxy.
     * Route grouping is published once, then every pair captures the production
     * grouped projection pipeline with an exact forced gate/up and down tile.
     * The first eligible pair establishes a serial-row-proven device reference;
     * every pair must match that reference byte-for-byte before timing.
     *
     * @param routed_case GGUF-derived source formats and production geometry.
     * @param rows Original prompt rows in this exact prefill bucket.
     * @param route_profile Deterministic expert-load distribution under test.
     * @param device_ordinal Explicit ROCm device assigned by the cell scheduler.
     * @param settings Screening and finalist timing policy.
     * @param csv Aggregate candidate evidence destination.
     * @param timing_csv Optional raw event-sample sidecar.
     */
    void runROCmProductionMoERoutedCase(
        const llaminar2::test::native_vnni_dispatch::MoERoutedPrefillCase &routed_case,
        int rows,
        llaminar2::test::native_vnni_dispatch::MoERoutingProfile route_profile,
        int device_ordinal,
        const MoEProductionSweepSettings &settings,
        std::FILE *csv,
        std::FILE *timing_csv)
    {
        if (settings.profiling() != settings.profiler_candidate.has_value())
        {
            throw std::invalid_argument(
                "ROCm profiler request and exact candidate pair must be supplied together");
        }
        const std::vector<MoEPrefillCandidatePair> candidates =
            settings.profiling()
                ? std::vector<MoEPrefillCandidatePair>{
                      *settings.profiler_candidate}
                : moePrefillCandidatePairs();
        if (rows <= 8)
        {
            throw std::invalid_argument(
                "production MoE prefill tournament requires M greater than eight");
        }
        if (routed_case.routed.gate != routed_case.routed.up)
        {
            throw std::runtime_error(
                routed_case.evidenceId() +
                " has distinct routed gate/up formats, but the production "
                "fused gate/up kernel requires one common codebook");
        }
        if (settings.screening_warmups < 0 ||
            settings.screening_trials <= 0 ||
            settings.screening_replays <= 0 ||
            settings.robust_warmups < 0 ||
            settings.robust_trials <= 0 ||
            settings.robust_replays <= 0 ||
            settings.minimum_finalists <= 0 ||
            settings.maximum_finalists < settings.minimum_finalists ||
            (!settings.profiling() &&
             settings.maximum_finalists > static_cast<int>(candidates.size())) ||
            settings.finalist_margin <= 0.0)
        {
            throw std::invalid_argument(
                "invalid ROCm MoE production tournament timing policy");
        }

        if (device_ordinal < 0)
        {
            throw std::invalid_argument(
                "ROCm production sweep device ordinal must be non-negative");
        }
        const auto device = llaminar2::DeviceId::rocm(device_ordinal);
        if (hipSetDevice(device_ordinal) != hipSuccess)
        {
            throw std::runtime_error(
                "failed to select ROCm sweep device " +
                std::to_string(device_ordinal));
        }
        ScopedMoEPrefillHipStream owned_stream;
        const hipStream_t stream = owned_stream.get();

        llaminar2::ROCmMoEKernel moe_storage(device_ordinal);
        llaminar2::IMoEKernel *moe = &moe_storage;
        moe->setGPUStream(stream);
        auto *workspace_consumer =
            dynamic_cast<llaminar2::IWorkspaceConsumer *>(moe);
        if (!workspace_consumer)
        {
            throw std::runtime_error(
                "ROCm MoE production kernel does not expose workspace binding");
        }

        const auto requirements = llaminar2::MoEWorkspaceBuffers::rocmMoE(
            rows,
            routed_case.hidden_size,
            routed_case.routed_expert_width,
            routed_case.expert_count,
            routed_case.experts_per_token);
        auto workspace = std::make_unique<llaminar2::DeviceWorkspaceManager>(
            device,
            requirements.total_bytes_with_alignment() + 8 * 1024 * 1024);
        if (!workspace->allocate(requirements))
        {
            throw std::runtime_error(
                "failed to allocate persistent ROCm MoE production workspace");
        }
        ScopedMoEPrefillWorkspaceBinding workspace_binding(
            workspace_consumer,
            workspace.get());

        const auto &gateup_format =
            llaminar2::test::quantizedMoEVerifierFormat(
                routed_case.routed.gate);
        const auto &down_format =
            llaminar2::test::quantizedMoEVerifierFormat(
                routed_case.routed.down);
        std::vector<int> materialized_experts(
            static_cast<size_t>(routed_case.expert_count));
        std::iota(materialized_experts.begin(), materialized_experts.end(), 0);
        auto tables = prepareExpertTables(
            moe,
            device,
            routed_case.expert_count,
            routed_case.hidden_size,
            routed_case.routed_expert_width,
            std::move(materialized_experts),
            gateup_format,
            down_format);
        if (tables.gateup_table_id < 0 || tables.down_table_id < 0)
        {
            throw std::runtime_error(
                "failed to publish production expert descriptor tables");
        }

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
            {
                throw std::runtime_error(
                    "failed to publish a ROCm MoE production sweep tensor");
            }
        }

        const auto prepare_groups = [&]()
        {
            return moe->prepareExpertGroupsAsync(
                route_indices_tensor.get(),
                route_weights_tensor.get(),
                rows,
                routed_case.expert_count,
                routed_case.experts_per_token);
        };
        const auto execute_pipeline = [&](llaminar2::ITensor *output)
        {
            return moe->executeGroupedPrefillPipeline(
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
        requireHipBenchBody(prepare_groups(), "production route preparation");
        if (hipStreamSynchronize(stream) != hipSuccess)
        {
            throw std::runtime_error(
                "failed to complete production route preparation");
        }

        /*
         * The hidden-state upload is an external producer for every captured
         * candidate graph.  Join its durable completion event before the first
         * begin-capture boundary; importing an eager event from inside HIP
         * capture is illegal and would make the tournament exercise a lifecycle
         * that production explicitly rejects.  The immutable hidden tensor and
         * exact stream are reused by all candidates, so this one edge remains
         * valid throughout the cell.
         */
        TransferEngine::requireDeviceInput(hidden.get(), device, stream);

        const MoEPrefillCandidatePair reference_candidate{
            {4, 64},
            {4, 64},
        };
        const auto reference_gateup_resources =
            queryMoEPrefillKernelResources(
                gateup_format.device_execution_codebook_id,
                /*projection_role=*/0,
                reference_candidate.gateup);
        const auto reference_down_resources =
            queryMoEPrefillKernelResources(
                down_format.device_execution_codebook_id,
                /*projection_role=*/1,
                reference_candidate.down);
        if (!reference_gateup_resources.spillFree() ||
            !reference_down_resources.spillFree())
        {
            throw std::runtime_error(
                "serial-proof reference candidate spills and is ineligible");
        }
        {
            ScopedROCmMoEPrefillCandidatePair forced(reference_candidate);
            llaminar2::PerfStatsCollector::reset();
            ScopedHipPerfGraph graph(
                stream,
                device_ordinal,
                "ROCm production MoE prefill reference capture");
            requireHipBenchBody(
                execute_pipeline(reference_output.get()),
                "production reference graph capture");
            if (!graph.finishAndInstantiate())
            {
                throw std::runtime_error(
                    "failed to instantiate production reference graph");
            }
            if (!observedMoEPrefillCandidatePair(
                    rows,
                    reference_candidate))
            {
                throw std::runtime_error(
                    "PerfStats did not observe the forced reference pair");
            }
            requireHipBenchBody(
                graph.launch(),
                "production reference graph replay");
            if (hipStreamSynchronize(stream) != hipSuccess)
            {
                throw std::runtime_error(
                    "failed to complete production reference replay");
            }
        }

        {
            std::vector<float> reference_host(reference_output->numel());
            if (hipMemcpyAsync(
                    reference_host.data(),
                    reference_output->gpu_data_ptr(),
                    reference_host.size() * sizeof(float),
                    hipMemcpyDeviceToHost,
                    stream) != hipSuccess ||
                hipStreamSynchronize(stream) != hipSuccess)
            {
                throw std::runtime_error(
                    "failed to materialize serial-proof reference output");
            }
            double serial_ms = 0.0;
            const std::vector<float> serial = runRowwiseDecode(
                moe,
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
            const CloseMetrics metrics = compareVectors(
                reference_host,
                serial,
                reference_host.size());
            if (metrics.bit_mismatch_count != 0 ||
                metrics.nonfinite_count != 0)
            {
                throw std::runtime_error(
                    "production reference is not byte-equivalent to serial rows");
            }

            /*
             * Serial diagnostics reuse kernel-owned workspace. Republish the
             * immutable production route before any candidate graph is captured
             * so no diagnostic state can leak into tournament execution.
             */
            requireHipBenchBody(
                prepare_groups(),
                "post-serial production route preparation");
            if (hipStreamSynchronize(stream) != hipSuccess)
            {
                throw std::runtime_error(
                    "failed to republish production groups after serial proof");
            }
        }

        MoEPrefillEventTimer timer(stream);
        MoEPrefillDeviceByteCertificate certificate;
        const auto measure_candidate = [&] (
            const MoEPrefillCandidatePair &candidate,
            int warmups,
            int trials,
            int replays,
            bool isolated_profile)
        {
            MoEProductionPairEvidence evidence{};
            evidence.candidate = candidate;
            evidence.gateup_resources = queryMoEPrefillKernelResources(
                gateup_format.device_execution_codebook_id,
                /*projection_role=*/0,
                candidate.gateup);
            evidence.down_resources = queryMoEPrefillKernelResources(
                down_format.device_execution_codebook_id,
                /*projection_role=*/1,
                candidate.down);
            if (!evidence.gateup_resources.spillFree() ||
                !evidence.down_resources.spillFree())
            {
                return evidence;
            }

            ScopedROCmMoEPrefillCandidatePair forced(candidate);
            llaminar2::PerfStatsCollector::reset();
            ScopedHipPerfGraph graph(
                stream,
                device_ordinal,
                "ROCm production MoE prefill candidate capture");
            requireHipBenchBody(
                execute_pipeline(candidate_output.get()),
                "production candidate graph capture");
            if (!graph.finishAndInstantiate())
            {
                throw std::runtime_error(
                    "failed to instantiate production candidate graph " +
                    candidate.id());
            }
            evidence.route_counter_ok =
                observedMoEPrefillCandidatePair(rows, candidate);
            if (!evidence.route_counter_ok)
            {
                throw std::runtime_error(
                    "PerfStats did not observe forced pair " + candidate.id());
            }
            for (int warmup = 0; warmup < warmups; ++warmup)
            {
                requireHipBenchBody(
                    graph.launch(),
                    "production candidate warmup replay");
            }
            requireHipBenchBody(
                graph.launch(),
                "production candidate byte-certificate replay");
            const auto mismatch = certificate.compare(
                reinterpret_cast<const float *>(
                    candidate_output->gpu_data_ptr()),
                reinterpret_cast<const float *>(
                    reference_output->gpu_data_ptr()),
                candidate_output->numel(),
                stream);
            evidence.bit_mismatches = mismatch.first;
            evidence.first_bit_mismatch = mismatch.second;
            if (evidence.bit_mismatches != 0)
            {
                throw std::runtime_error(
                    "candidate " + candidate.id() +
                    " is not byte-equivalent to the serial-proven reference");
            }

            if (isolated_profile)
            {
                /*
                 * Selected-region rocprof collection remains paused during all
                 * allocation, upload, route preparation, capture, warmup, and
                 * device byte certification. Resume for exactly one terminal
                 * graph replay, then pause before closing the immutable request
                 * range. Every physical dispatch in the report therefore owns
                 * this candidate pair; no tournament or setup kernel can leak
                 * into its feature record.
                 */
                if (hipStreamSynchronize(stream) != hipSuccess)
                {
                    throw std::runtime_error(
                        "failed to settle ROCm preconditioning before profiling");
                }
                const std::string profiler_range =
                    "NativeVNNIProfile::" + settings.profiler_request_id;
                if (roctxRangePushA(profiler_range.c_str()) < 0)
                {
                    throw std::runtime_error(
                        "failed to open isolated ROCm MoE profiler range");
                }
                if (roctxProfilerResume(0) != 0)
                {
                    (void)roctxRangePop();
                    throw std::runtime_error(
                        "failed to resume isolated ROCm MoE profiler collection");
                }
                const bool launch_ok = graph.launch();
                const hipError_t completion = hipStreamSynchronize(stream);
                const int pause_status = roctxProfilerPause(0);
                const int range_level = roctxRangePop();
                if (!launch_ok || completion != hipSuccess || pause_status != 0 ||
                    range_level < 0)
                {
                    throw std::runtime_error(
                        "isolated ROCm MoE profiler replay failed");
                }
                std::fprintf(
                    stderr,
                    "[NativeVNNIProfiler][ROCm-MoE-Prefill] request=%s "
                    "candidate=%s M=%d launches=1\n",
                    settings.profiler_request_id.c_str(),
                    candidate.id().c_str(),
                    rows);
                return evidence;
            }

            evidence.screening_samples_ms.reserve(
                static_cast<size_t>(trials));
            for (int trial = 0; trial < trials; ++trial)
            {
                evidence.screening_samples_ms.push_back(timer.sample(
                    replays,
                    [&]()
                    {
                        return graph.launch();
                    }));
            }
            std::sort(
                evidence.screening_samples_ms.begin(),
                evidence.screening_samples_ms.end());
            evidence.screening_replays_per_sample = replays;
            return evidence;
        };

        if (settings.profiling())
        {
            const auto evidence = measure_candidate(
                candidates.front(),
                /*warmups=*/2,
                /*trials=*/1,
                /*replays=*/1,
                /*isolated_profile=*/true);
            if (!evidence.gateup_resources.spillFree() ||
                !evidence.down_resources.spillFree())
            {
                throw std::runtime_error(
                    "spilling ROCm candidate reached isolated profiling");
            }
            return;
        }

        std::vector<MoEProductionPairEvidence> evidence_rows;
        evidence_rows.reserve(candidates.size());
        for (const auto &candidate : candidates)
        {
            MoEProductionPairEvidence evidence = measure_candidate(
                candidate,
                settings.screening_warmups,
                settings.screening_trials,
                settings.screening_replays,
                /*isolated_profile=*/false);
            if (!evidence.gateup_resources.spillFree() ||
                !evidence.down_resources.spillFree())
            {
                std::fprintf(
                    stderr,
                    "[ROCm MoE production sweep] discard spilling pair %s "
                    "gate_local=%zu down_local=%zu\n",
                    candidate.id().c_str(),
                    evidence.gateup_resources.local_memory_bytes_per_thread,
                    evidence.down_resources.local_memory_bytes_per_thread);
                continue;
            }
            evidence_rows.push_back(std::move(evidence));
        }
        if (evidence_rows.empty())
        {
            throw std::runtime_error(
                "all production MoE candidate pairs were ineligible");
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
            MoEProductionPairEvidence robust = measure_candidate(
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
        {
            throw std::runtime_error(
                "production MoE tournament did not retain a finalist");
        }
        winner->winner = true;
        for (const auto &evidence : evidence_rows)
        {
            writeMoEProductionPairEvidence(
                csv,
                routed_case,
                route_profile,
                route_stats,
                rows,
                evidence);
            writeMoEProductionPairTimingEvidence(
                timing_csv,
                routed_case,
                route_profile,
                rows,
                evidence);
        }
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
                device,
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

                    /*
                     * Static resource inspection happens before capture and
                     * before any timing event is created. A spilling template
                     * specialization is not a tournament contestant: retaining
                     * it as a slow observation wastes sweep time and lets a
                     * future noisy sample accidentally promote an invalid
                     * production launch.
                     */
                    const int projection_role = role == "gateup" ? 0 : 1;
                    const MoEPrefillKernelResources resources =
                        queryMoEPrefillKernelResources(
                            format.device_execution_codebook_id,
                            projection_role,
                            candidate);
                    if (!resources.spillFree())
                    {
                        ADD_FAILURE()
                            << format.label << ' ' << role << ' '
                            << candidate_name << " uses "
                            << resources.local_memory_bytes_per_thread
                            << " local/scratch bytes per thread and was "
                               "discarded before timing";
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
                    ScopedHipPerfGraph graph(
                        stream,
                        /*device_ordinal=*/0,
                        "ROCm MoE grouped-prefill candidate capture");
                    ASSERT_TRUE(run_pipeline())
                        << format.label << ' ' << shape.name << " M=" << rows
                        << ' ' << role << ' ' << candidate_name;
                    ASSERT_TRUE(graph.finishAndInstantiate());

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
                        ASSERT_TRUE(graph.launch());
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
                    ASSERT_TRUE(graph.launch());
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
                                return graph.launch();
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

TEST(Perf__MoEVerifierPrefill, ROCm_AllFormatGroupedPrefillCandidatesAreSpillFree)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    std::set<uint8_t> inspected_codebooks;
    for (const auto &format : llaminar2::test::quantizedVerifierFormats())
    {
        if (!inspected_codebooks.insert(
                format.device_execution_codebook_id).second)
        {
            continue;
        }

        for (int projection_role = 0; projection_role < 2; ++projection_role)
        {
            for (const auto &candidate : kMoEPrefillSweepCandidates)
            {
                SCOPED_TRACE(
                    std::string(format.label) + "/" +
                    (projection_role == 0 ? "gateup/" : "down/") +
                    moePrefillCandidateName(candidate));
                const MoEPrefillKernelResources resources =
                    queryMoEPrefillKernelResources(
                        format.device_execution_codebook_id,
                        projection_role,
                        candidate);
                EXPECT_EQ(resources.local_memory_bytes_per_thread, 0u)
                    << "register-spilling candidates must be removed from the "
                       "compiled launch inventory before a sweep is run";
                EXPECT_GT(resources.registers_per_thread, 0);
                EXPECT_GE(resources.max_threads_per_block, candidate.tile_n);
                EXPECT_GT(resources.max_active_blocks_per_sm, 0);
            }
        }
    }
    EXPECT_FALSE(inspected_codebooks.empty());
#endif
}

TEST(Perf__MoEVerifierPrefill, ROCm_DeviceByteCertificateIsExact)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    hipStream_t stream = nullptr;
    float *actual = nullptr;
    float *expected = nullptr;
    uint64_t *mismatch_count = nullptr;
    uint64_t *first_mismatch = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    ASSERT_EQ(hipMalloc(&actual, 4 * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&expected, 4 * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&mismatch_count, sizeof(uint64_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&first_mismatch, sizeof(uint64_t)), hipSuccess);

    const std::array<uint32_t, 4> actual_bits{
        0x00000000u, 0x80000000u, 0x7fc00001u, 0x3f800000u};
    const std::array<uint32_t, 4> expected_bits{
        0x00000000u, 0x00000000u, 0x7fc00002u, 0x3f800000u};
    ASSERT_EQ(
        hipMemcpyAsync(
            actual,
            actual_bits.data(),
            sizeof(actual_bits),
            hipMemcpyHostToDevice,
            stream),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(
            expected,
            expected_bits.data(),
            sizeof(expected_bits),
            hipMemcpyHostToDevice,
            stream),
        hipSuccess);
    ASSERT_TRUE(llaminar2::test::enqueueROCmFP32ByteComparison(
        actual,
        expected,
        actual_bits.size(),
        mismatch_count,
        first_mismatch,
        stream));

    uint64_t host_mismatch_count = 0;
    uint64_t host_first_mismatch = 0;
    ASSERT_EQ(
        hipMemcpyAsync(
            &host_mismatch_count,
            mismatch_count,
            sizeof(host_mismatch_count),
            hipMemcpyDeviceToHost,
            stream),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(
            &host_first_mismatch,
            first_mismatch,
            sizeof(host_first_mismatch),
            hipMemcpyDeviceToHost,
            stream),
        hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    EXPECT_EQ(host_mismatch_count, 2u);
    EXPECT_EQ(host_first_mismatch, 1u);

    EXPECT_EQ(hipFree(first_mismatch), hipSuccess);
    EXPECT_EQ(hipFree(mismatch_count), hipSuccess);
    EXPECT_EQ(hipFree(expected), hipSuccess);
    EXPECT_EQ(hipFree(actual), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
}

TEST(Perf__MoEVerifierPrefill, ROCm_ProductionMoEMixtureManifestIsCanonical)
{
    using llaminar2::test::native_vnni_dispatch::
        nativeVnniMoEPrefillMixtureManifest;
    using llaminar2::test::native_vnni_dispatch::
        nativeVnniMoERoutedPrefillCases;

    const auto &mixtures = nativeVnniMoEPrefillMixtureManifest();
    ASSERT_FALSE(mixtures.empty());
    std::set<std::string> identities;
    size_t native_vnni_count = 0;
    const auto &format_cases = llaminar2::test::quantizedVerifierFormats();
    const auto has_format = [&format_cases](const std::string &label)
    {
        return std::any_of(
            format_cases.begin(),
            format_cases.end(),
            [&label](const auto &format)
            {
                return label == format.label;
            });
    };

    for (const auto &mixture : mixtures)
    {
        ASSERT_TRUE(identities.insert(mixture.evidenceId()).second)
            << mixture.evidenceId();
        ASSERT_GT(mixture.hidden_size, 0);
        ASSERT_GT(mixture.routed_expert_width, 0);
        ASSERT_GT(mixture.shared_expert_width, 0);
        ASSERT_GT(mixture.expert_count, 0);
        ASSERT_GT(mixture.experts_per_token, 0);
        ASSERT_LE(mixture.experts_per_token, mixture.expert_count);
        ASSERT_FALSE(mixture.uses.empty());
        if (!mixture.native_vnni_sweepable)
            continue;
        ++native_vnni_count;
        for (const auto &format : mixture.routed.ordered())
            EXPECT_TRUE(has_format(format)) << mixture.evidenceId() << ' ' << format;
        for (const auto &format : mixture.shared.ordered())
            EXPECT_TRUE(has_format(format)) << mixture.evidenceId() << ' ' << format;
    }
    EXPECT_GT(native_vnni_count, 0u);
    EXPECT_LT(native_vnni_count, mixtures.size())
        << "all-format inventory unexpectedly lost floating/MXFP4 cases";

    const auto &routed_cases = nativeVnniMoERoutedPrefillCases();
    EXPECT_EQ(routed_cases.size(), 49u)
        << "the pinned GGUF manifest's routed source-key inventory changed";
    for (const auto &routed_case : routed_cases)
    {
        EXPECT_EQ(routed_case.routed.gate, routed_case.routed.up)
            << routed_case.evidenceId()
            << " requires a heterogeneous fused gate/up implementation";
        EXPECT_FALSE(routed_case.mixture_evidence_ids.empty())
            << routed_case.evidenceId();
    }
}

TEST(Perf__MoEVerifierPrefill, ROCm_Qwen36_35B_IQ3SProductionMixtureM64)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    using llaminar2::test::native_vnni_dispatch::
        nativeVnniMoEPrefillMixtureManifest;
    const auto &manifest = nativeVnniMoEPrefillMixtureManifest();
    const auto mixture = std::find_if(
        manifest.begin(),
        manifest.end(),
        [](const auto &candidate)
        {
            if (candidate.routed.gate != "IQ2_S" ||
                candidate.routed.up != "IQ2_S" ||
                candidate.routed.down != "IQ4_XS")
            {
                return false;
            }
            return std::any_of(
                candidate.uses.begin(),
                candidate.uses.end(),
                [](const auto &use)
                {
                    return use.release_id == "Qwen3.6-35B-A3B" &&
                           use.variant_id == "UD-IQ3_S" &&
                           use.layers.size() == 37;
                });
        });
    ASSERT_NE(mixture, manifest.end());
    ASSERT_TRUE(mixture->native_vnni_sweepable);
    ASSERT_EQ(mixture->expert_count, 256);
    ASSERT_EQ(mixture->experts_per_token, 8);

    const auto &gateup_format =
        llaminar2::test::quantizedMoEVerifierFormat(mixture->routed.gate);
    const auto &down_format =
        llaminar2::test::quantizedMoEVerifierFormat(mixture->routed.down);
    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    ScopedEnvOverride iterations_env(
        "LLAMINAR_MOE_VERIFIER_PREFILL_ITERS", "1");
    ScopedEnvOverride warmups_env(
        "LLAMINAR_MOE_VERIFIER_PREFILL_WARMUPS", "0");
    ScopedEnvOverride rowwise_env(
        "LLAMINAR_MOE_VERIFIER_PREFILL_ROWWISE_ITERS", "1");
    const auto result = runROCmCase(
        /*shared=*/false,
        /*rows=*/64,
        mixture->experts_per_token,
        mixture->expert_count,
        /*case_name_override=*/"qwen36_35b_ud_iq3s_production_mixture",
        /*unique_routes=*/true,
        /*include_terminal_expert=*/false,
        mixture->hidden_size,
        mixture->routed_expert_width,
        &gateup_format,
        &down_format,
        /*canonical_route_split=*/false);
    expectClose(result.metrics);
    EXPECT_EQ(result.metrics.bit_mismatch_count, 0u);
    EXPECT_EQ(result.metrics.nonfinite_count, 0u);
    printResult(result);
#endif
}

TEST(Perf__MoEVerifierPrefill, ROCm_ProductionGGUFMixtureCandidatePairTrainer)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";
    if (envInt("LLAMINAR_ROCM_MOE_PRODUCTION_SWEEP", 0) == 0)
    {
        GTEST_SKIP()
            << "Set LLAMINAR_ROCM_MOE_PRODUCTION_SWEEP=1 to run the "
               "GGUF-derived candidate-pair trainer";
    }

    MoEProductionSweepSettings settings{};
    settings.screening_warmups = envInt(
        "LLAMINAR_ROCM_MOE_PRODUCTION_SWEEP_SCREEN_WARMUPS", 1);
    settings.screening_trials = envInt(
        "LLAMINAR_ROCM_MOE_PRODUCTION_SWEEP_SCREEN_TRIALS", 3);
    settings.screening_replays = envInt(
        "LLAMINAR_ROCM_MOE_PRODUCTION_SWEEP_SCREEN_REPLAYS", 2);
    settings.robust_warmups = envInt(
        "LLAMINAR_ROCM_MOE_PRODUCTION_SWEEP_ROBUST_WARMUPS", 2);
    settings.robust_trials = envInt(
        "LLAMINAR_ROCM_MOE_PRODUCTION_SWEEP_ROBUST_TRIALS", 15);
    settings.robust_replays = envInt(
        "LLAMINAR_ROCM_MOE_PRODUCTION_SWEEP_ROBUST_REPLAYS", 4);
    settings.minimum_finalists = envInt(
        "LLAMINAR_ROCM_MOE_PRODUCTION_SWEEP_MIN_FINALISTS", 12);
    settings.maximum_finalists = envInt(
        "LLAMINAR_ROCM_MOE_PRODUCTION_SWEEP_MAX_FINALISTS", 24);
    settings.finalist_margin = envPositiveDouble(
        "LLAMINAR_ROCM_MOE_PRODUCTION_SWEEP_FINALIST_MARGIN", 0.05);
    const std::vector<int> m_values = envCsvInts(
        "LLAMINAR_ROCM_MOE_PRODUCTION_SWEEP_M",
        {64, 256, 1024, 2048, 4096, 8192, 16384});
    const int maximum_cells = envInt(
        "LLAMINAR_ROCM_MOE_PRODUCTION_SWEEP_MAX_CELLS",
        std::numeric_limits<int>::max());
    const int device_ordinal = envInt(
        "LLAMINAR_ROCM_MOE_PRODUCTION_SWEEP_DEVICE", 0);
    const char *route_profile_environment =
        std::getenv("LLAMINAR_ROCM_MOE_PRODUCTION_SWEEP_ROUTE_PROFILE");
    if (!route_profile_environment || !*route_profile_environment)
    {
        throw std::runtime_error(
            "ROCm production MoE sweep requires one explicit route profile");
    }
    const auto route_profile =
        llaminar2::test::native_vnni_dispatch::parseMoERoutingProfile(
            route_profile_environment);

    settings.profiler_request_id =
        llaminar2::test::native_vnni_dispatch::profilerRequestId();
    const std::string profiler_candidate_id =
        llaminar2::test::native_vnni_dispatch::profilerEnvironment(
            "LLAMINAR_ROCM_MOE_PRODUCTION_PROFILE_CANDIDATE");
    if (settings.profiling() != !profiler_candidate_id.empty())
    {
        throw std::runtime_error(
            "isolated ROCm MoE profiling requires both a profiler request ID "
            "and LLAMINAR_ROCM_MOE_PRODUCTION_PROFILE_CANDIDATE");
    }
    if (settings.profiling())
    {
        settings.profiler_candidate =
            findMoEPrefillCandidatePair(profiler_candidate_id);
        if (!settings.profiler_candidate)
        {
            throw std::runtime_error(
                "unknown ROCm MoE production profiler candidate " +
                profiler_candidate_id);
        }
        const std::string selected_cases =
            llaminar2::test::native_vnni_dispatch::profilerEnvironment(
                "LLAMINAR_ROCM_MOE_PRODUCTION_SWEEP_CASES");
        if (selected_cases.empty() ||
            selected_cases.find(',') != std::string::npos ||
            m_values.size() != 1 || maximum_cells != 1)
        {
            throw std::runtime_error(
                "isolated ROCm MoE profiling requires one explicit case, one M, "
                "and LLAMINAR_ROCM_MOE_PRODUCTION_SWEEP_MAX_CELLS=1");
        }
    }

    ScopedEnvOverride rowwise_iters(
        "LLAMINAR_MOE_VERIFIER_PREFILL_ROWWISE_ITERS", "1");
    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");

    std::FILE *csv = settings.profiling() ? nullptr : stdout;
    bool owns_csv = false;
    if (!settings.profiling())
    {
        if (const char *path =
                std::getenv("LLAMINAR_ROCM_MOE_PRODUCTION_SWEEP_CSV");
            path && *path)
        {
            csv = std::fopen(path, "w");
            ASSERT_NE(csv, nullptr)
                << "failed to open production MoE aggregate CSV " << path;
            owns_csv = true;
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

    std::FILE *timing_csv = nullptr;
    if (!settings.profiling())
    {
        if (const char *path =
                std::getenv("LLAMINAR_ROCM_MOE_PRODUCTION_SWEEP_TIMING_CSV");
            path && *path)
        {
            timing_csv = std::fopen(path, "w");
            ASSERT_NE(timing_csv, nullptr)
                << "failed to open production MoE timing CSV " << path;
            std::fprintf(
                timing_csv,
                "backend,phase,case_id,route_profile,m,candidate_id,timing_phase,"
                "sample_index,timed_replays,latency_us,latency_ms_hex\n");
        }
    }

    int executed_cells = 0;
    for (const auto &routed_case :
         llaminar2::test::native_vnni_dispatch::nativeVnniMoERoutedPrefillCases())
    {
        if (!envCsvContainsOrUnset(
                "LLAMINAR_ROCM_MOE_PRODUCTION_SWEEP_CASES",
                routed_case.evidenceId()) ||
            !envCsvContainsOrUnset(
                "LLAMINAR_ROCM_MOE_PRODUCTION_SWEEP_GATE_FORMATS",
                routed_case.routed.gate) ||
            !envCsvContainsOrUnset(
                "LLAMINAR_ROCM_MOE_PRODUCTION_SWEEP_DOWN_FORMATS",
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
                "[ROCm MoE production sweep] case=%s route=%s M=%d gate/up=%s "
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
            runROCmProductionMoERoutedCase(
                routed_case,
                rows,
                route_profile,
                device_ordinal,
                settings,
                csv,
                timing_csv);
            ++executed_cells;
        }
        if (executed_cells >= maximum_cells)
            break;
    }

    if (owns_csv)
        ASSERT_EQ(std::fclose(csv), 0);
    if (timing_csv)
        ASSERT_EQ(std::fclose(timing_csv), 0);
    EXPECT_GT(executed_cells, 0)
        << "production filters selected no GGUF-derived MoE sweep cells";
#endif
}

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

/**
 * @brief Certify the exact routed-expert geometry used by the 122B overlay.
 *
 * The production sparse endpoint retains an eight-route graph family and
 * presents one route per compact row. Its Qwen3.5-122B-A10B expert slabs are
 * Q8_0 for gate, up, and down, with hidden width 3072 and expert width 1024.
 * Keeping this as a named speedometer prevents tuning a smaller proxy shape
 * while the real overlay remains dominated by its ROCm participant kernels.
 */
TEST(Perf__MoEVerifierPrefill, ROCm_Qwen35_122B_Q8_0_OverlayDecodeM8)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const auto &q8_0 = llaminar2::test::quantizedVerifierFormat("Q8_0");
    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    const auto result = runROCmCase(
        /*shared=*/false,
        /*rows=*/8,
        /*routed_top_k=*/1,
        /*routed_num_experts=*/256,
        /*case_name_override=*/"qwen35_122b_q8_0_overlay_decode_m8",
        /*unique_routes=*/true,
        /*include_terminal_expert=*/true,
        /*d_model=*/3072,
        /*intermediate=*/1024,
        &q8_0,
        &q8_0,
        /*canonical_route_split=*/false);
    expectClose(result.metrics);
    EXPECT_EQ(result.metrics.bit_mismatch_count, 0u);
    EXPECT_EQ(result.metrics.nonfinite_count, 0u);
    expectGraphReplayFasterThanReference(result);
    printResult(result);
#endif
}

/**
 * @brief Measure the exact per-participant route counts in 122B overlay decode.
 *
 * The complete token has eight routes, but a four-device secondary domain
 * normally assigns only one or two of them to each follower. M=1/2/4 therefore
 * distinguishes follower expert arithmetic from mapped packet latency while
 * retaining the same Q8_0 descriptors, hidden width, graph capture, and
 * serial-row byte oracle as the aggregate M=8 production speedometer.
 */
TEST(Perf__MoEVerifierPrefill, ROCm_Qwen35_122B_Q8_0_OverlayDecodeM1M2M4)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const auto &q8_0 = llaminar2::test::quantizedVerifierFormat("Q8_0");
    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    for (const int rows : {1, 2, 4})
    {
        SCOPED_TRACE(rows);
        const std::string case_name =
            "qwen35_122b_q8_0_overlay_decode_m" + std::to_string(rows);
        const auto result = runROCmCase(
            /*shared=*/false,
            rows,
            /*routed_top_k=*/1,
            /*routed_num_experts=*/256,
            case_name.c_str(),
            /*unique_routes=*/true,
            /*include_terminal_expert=*/true,
            /*d_model=*/3072,
            /*intermediate=*/1024,
            &q8_0,
            &q8_0,
            /*canonical_route_split=*/false);
        expectClose(result.metrics);
        EXPECT_EQ(result.metrics.bit_mismatch_count, 0u);
        EXPECT_EQ(result.metrics.nonfinite_count, 0u);
        expectGraphReplayFasterThanReference(result);
        printResult(result);
    }
#endif
}

/**
 * @brief Prove and time the real M=1 explicit-routing overlay follower.
 *
 * One and two local routes cover the normal four-way secondary-tier split of
 * a global top-8 row.  The test retains the fixed top-8 tensor geometry after
 * compact packet consumption and requires byte identity between direct output
 * and canonical route publication followed by the ordered reducer.
 */
TEST(Perf__MoEVerifierPrefill,
     ROCm_Qwen35_122B_Q8_0_OverlayFollowerExplicitDecodeTop8)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    std::cout
        << "backend,case,valid_routes,top_k,num_experts,d_model,intermediate,"
           "eager_ms,graph_ms,canonical_oracle_ms,bit_mismatch_count,"
           "nonfinite_count\n";
    for (const int valid_routes : {1, 2})
    {
        SCOPED_TRACE(valid_routes);
        const OverlayFollowerDecodeResult result =
            runROCmOverlayFollowerDecodeCase(valid_routes);
        expectClose(result.metrics);
        EXPECT_EQ(result.metrics.bit_mismatch_count, 0u);
        EXPECT_EQ(result.metrics.nonfinite_count, 0u);
        EXPECT_GT(result.eager_ms, 0.0);
        EXPECT_GT(result.graph_ms, 0.0);
        EXPECT_GT(result.canonical_oracle_ms, 0.0);
        std::cout << std::fixed << std::setprecision(4)
                  << "rocm,qwen35_122b_q8_0_overlay_follower_explicit_decode,"
                  << result.valid_routes
                  << ",8,256,3072,1024,"
                  << result.eager_ms << ','
                  << result.graph_ms << ','
                  << result.canonical_oracle_ms << ','
                  << result.metrics.bit_mismatch_count << ','
                  << result.metrics.nonfinite_count << '\n';
    }
#endif
}

/**
 * @brief Measure the larger retained sparse-endpoint buckets for Qwen 122B.
 *
 * M=16 and M=32 are uncommon during steady-state depth-three verification but
 * are retained production capacities and can be selected by prefill
 * segmentation or a wider future speculative policy.  They also straddle the
 * current route-owned/expert-tiled boundary, making this test the production-
 * shape speedometer for training that dispatch decision instead of inferring
 * it from a smaller model or a dense top-k route profile.
 */
TEST(Perf__MoEVerifierPrefill, ROCm_Qwen35_122B_Q8_0_OverlayDecodeM16M32)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const auto &q8_0 = llaminar2::test::quantizedVerifierFormat("Q8_0");
    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    for (const int rows : {16, 32})
    {
        SCOPED_TRACE(rows);
        const auto result = runROCmCase(
            /*shared=*/false,
            rows,
            /*routed_top_k=*/1,
            /*routed_num_experts=*/256,
            rows == 16
                ? "qwen35_122b_q8_0_overlay_decode_m16"
                : "qwen35_122b_q8_0_overlay_decode_m32",
            /*unique_routes=*/true,
            /*include_terminal_expert=*/true,
            /*d_model=*/3072,
            /*intermediate=*/1024,
            &q8_0,
            &q8_0,
            /*canonical_route_split=*/false);
        expectClose(result.metrics);
        EXPECT_EQ(result.metrics.bit_mismatch_count, 0u);
        EXPECT_EQ(result.metrics.nonfinite_count, 0u);
        expectGraphReplayFasterThanReference(result);
        printResult(result);
    }
#endif
}

/**
 * @brief Certify the exact 600-row secondary-participant prefill workload.
 *
 * The canonical 595-token prompt is padded into the retained 600-row graph
 * family. With 30 experts resident on the continuation tier, each secondary
 * GPU owns about 56 of 256 experts and receives 1.75 compact top-eight routes
 * per physical row. This speedometer therefore times the same Q8_0 geometry,
 * compact packet convention, grouped prefill pipeline, independent route-slot
 * publication, ordered fold, and captured replay used by production.
 */
TEST(Perf__MoEVerifierPrefill,
     ROCm_Qwen35_122B_Q8_0_OverlayFollowerPrefillM600)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    constexpr int rows = 600;
    constexpr int top_k = 8;
    constexpr int num_experts = 256;
    constexpr int first_local_expert = 30;
    constexpr int local_experts = 56;
    const auto &q8_0 =
        llaminar2::test::quantizedVerifierFormat("Q8_0");
    auto [routing_indices, routing_weights] =
        makeCompactOverlayFollowerRoutes(
            rows, top_k, first_local_expert, local_experts);

    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    const auto result = runROCmCase(
        /*shared=*/false,
        rows,
        top_k,
        num_experts,
        "qwen35_122b_q8_0_overlay_follower_prefill_m600",
        /*unique_routes=*/false,
        /*include_terminal_expert=*/false,
        /*d_model=*/3072,
        /*intermediate=*/1024,
        &q8_0,
        &q8_0,
        /*canonical_route_split=*/true,
        &routing_indices,
        &routing_weights);
    expectClose(result.metrics);
    EXPECT_EQ(result.metrics.bit_mismatch_count, 0u);
    EXPECT_EQ(result.metrics.nonfinite_count, 0u);
    expectGraphReplayFasterThanReference(result);
    printResult(result);
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
    auto shared = runROCmSharedExpertStageCase(/*rows=*/1, format);
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

TEST(Perf__MoEVerifierPrefill, ROCm_M4_CanonicalRouteSplitUpperBound)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    /*
     * Compare the route-parallel down producer plus strict router-order
     * reducer using the same top-9 verifier geometry as the direct publication
     * speedometer. The candidate uses only public production APIs and
     * persistent device storage, so a winner can be selected by graph policy
     * without introducing a benchmark-only implementation.
     */
    ScopedEnvOverride stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    auto combined = runROCmCase(
        /*shared=*/false,
        /*rows=*/4,
        /*routed_top_k=*/9,
        /*routed_num_experts=*/257,
        /*case_name_override=*/"canonical_route_split_top9",
        /*unique_routes=*/true,
        /*include_terminal_expert=*/true,
        /*d_model=*/2048,
        /*intermediate=*/512,
        /*gateup_format=*/nullptr,
        /*down_format=*/nullptr,
        /*canonical_route_split=*/true);
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
                /*case_name_override=*/"canonical_route_split_top9_all_formats",
                /*unique_routes=*/false,
                /*include_terminal_expert=*/true,
                /*d_model=*/2048,
                /*intermediate=*/512,
                &format,
                &format,
                /*canonical_route_split=*/true);
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
