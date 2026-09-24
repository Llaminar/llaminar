/**
 * @file Perf__CPUMoERouterVerifierRows.cpp
 * @brief Byte-exact economy speedometer for grouped CPU MoE routing.
 *
 * Qwen3.6-35B routes each hidden row through a 256-by-2048 FP32 gate and
 * selects eight experts. Production CPU routing also publishes a canonical
 * Q8_1 copy of every hidden row for the immediately following NativeVNNI
 * expert projections. This harness times that complete contract rather than
 * an isolated dot product.
 *
 * The grouped transaction is compared with the production M=1 route repeated
 * in serial row order. Every softmax probability, selected expert, normalized
 * top-k weight, and Q8_1 activation byte must match. Input generation, result
 * validation, and capacity warmup are outside the canonical timing samples.
 */

#include <gtest/gtest.h>

#include "kernels/cpu/moe/CPUMoEKernel.h"
#include "utils/VerifierRowTestInventory.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

using namespace llaminar2;

namespace
{
    constexpr int kDModel = 2048;
    constexpr int kNumExperts = 256;
    constexpr int kTopK = 8;

    /** @brief Complete observable output from one CPU routing transaction. */
    struct RouteSnapshot
    {
        std::vector<int> expert_indices;
        std::vector<float> expert_weights;
        std::vector<float> router_probabilities;
        std::vector<Q8_1Block> q8_hidden;
    };

    /** @brief One timing row emitted by the production-geometry tournament. */
    struct RouteTiming
    {
        int rows = 0;
        double grouped_us = 0.0;
        double serial_us = 0.0;
        double speedup = 0.0;
    };

    /** @brief Parse one strictly positive integer environment setting. */
    int envPositiveInt(const char *name, int fallback)
    {
        const char *raw = std::getenv(name);
        if (!raw || !*raw)
            return fallback;

        char *end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end == raw || *end != '\0' || parsed <= 0 ||
            parsed > std::numeric_limits<int>::max())
        {
            throw std::runtime_error(
                std::string(name) + " must be a positive integer");
        }
        return static_cast<int>(parsed);
    }

    /**
     * @brief Select the runtime-M inventory, retaining totality probes by default.
     *
     * A comma-separated override supports focused `perf` runs without changing
     * the canonical CTest inventory.
     */
    std::vector<int> selectedRows()
    {
        const char *raw = std::getenv("LLAMINAR_CPU_MOE_ROUTER_M");
        if (!raw || !*raw)
        {
            std::vector<int> rows = {1};
            rows.insert(
                rows.end(),
                test::kGroupedVerifierRuntimeRows.begin(),
                test::kGroupedVerifierRuntimeRows.end());
            return rows;
        }

        std::vector<int> rows;
        std::stringstream stream(raw);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            char *end = nullptr;
            const long parsed = std::strtol(token.c_str(), &end, 10);
            if (token.empty() || end == token.c_str() || *end != '\0' ||
                parsed <= 0 || parsed > std::numeric_limits<int>::max())
            {
                throw std::runtime_error(
                    "LLAMINAR_CPU_MOE_ROUTER_M requires positive comma-separated rows");
            }
            const int row_count = static_cast<int>(parsed);
            if (std::find(rows.begin(), rows.end(), row_count) == rows.end())
                rows.push_back(row_count);
        }
        if (rows.empty())
            throw std::runtime_error("LLAMINAR_CPU_MOE_ROUTER_M selected no rows");
        return rows;
    }

    /** @brief Build deterministic hidden rows with non-trivial quantization tails. */
    std::vector<float> makeHidden(int rows)
    {
        std::vector<float> hidden(static_cast<size_t>(rows) * kDModel);
        for (size_t index = 0; index < hidden.size(); ++index)
        {
            const int centered = static_cast<int>((index * 17 + index / 11) % 257) - 128;
            hidden[index] =
                static_cast<float>(centered) * 0.003125f +
                static_cast<float>(static_cast<int>(index % 7) - 3) * 0.00031f;
        }
        return hidden;
    }

    /** @brief Build a deterministic production-sized router gate matrix. */
    std::vector<float> makeGateWeights()
    {
        std::vector<float> gate(
            static_cast<size_t>(kNumExperts) * kDModel);
        for (size_t index = 0; index < gate.size(); ++index)
        {
            const int centered =
                static_cast<int>((index * 29 + index / 37) % 193) - 96;
            gate[index] =
                static_cast<float>(centered) * 0.00042f +
                static_cast<float>(static_cast<int>(index % 5) - 2) * 0.000013f;
        }
        return gate;
    }

    /**
     * @brief Materialize all observable bytes from one grouped route.
     */
    RouteSnapshot captureGrouped(
        CPUMoEKernel &kernel,
        const std::vector<float> &hidden,
        const std::vector<float> &gate,
        int rows)
    {
        MoERoutingResult result;
        if (!kernel.route(
                hidden.data(), gate.data(), rows, kDModel, kNumExperts, kTopK,
                true, result))
        {
            throw std::runtime_error("grouped CPU router rejected valid geometry");
        }

        const Q8_1Block *q8 =
            kernel.publishedRouterQ8Hidden(hidden.data(), rows, kDModel);
        if (!q8)
            throw std::runtime_error("grouped CPU router did not publish Q8 hidden rows");

        const int blocks_per_row = kDModel / Q8_1Block::BLOCK_SIZE;
        RouteSnapshot snapshot;
        snapshot.expert_indices = result.expert_indices;
        snapshot.expert_weights = result.expert_weights;
        snapshot.router_probabilities = result.router_logits;
        snapshot.q8_hidden.assign(
            q8,
            q8 + static_cast<size_t>(rows) * blocks_per_row);
        return snapshot;
    }

    /**
     * @brief Materialize the diagnostic serial M=1 oracle in semantic row order.
     */
    RouteSnapshot captureSerial(
        CPUMoEKernel &kernel,
        const std::vector<float> &hidden,
        const std::vector<float> &gate,
        int rows)
    {
        const int blocks_per_row = kDModel / Q8_1Block::BLOCK_SIZE;
        RouteSnapshot snapshot;
        snapshot.expert_indices.resize(static_cast<size_t>(rows) * kTopK);
        snapshot.expert_weights.resize(static_cast<size_t>(rows) * kTopK);
        snapshot.router_probabilities.resize(
            static_cast<size_t>(rows) * kNumExperts);
        snapshot.q8_hidden.resize(
            static_cast<size_t>(rows) * blocks_per_row);

        MoERoutingResult row_result;
        for (int row = 0; row < rows; ++row)
        {
            const float *row_hidden =
                hidden.data() + static_cast<size_t>(row) * kDModel;
            if (!kernel.route(
                    row_hidden, gate.data(), 1, kDModel, kNumExperts, kTopK,
                    true, row_result))
            {
                throw std::runtime_error("serial CPU router rejected valid row");
            }

            std::copy_n(
                row_result.expert_indices.data(),
                kTopK,
                snapshot.expert_indices.data() + static_cast<size_t>(row) * kTopK);
            std::copy_n(
                row_result.expert_weights.data(),
                kTopK,
                snapshot.expert_weights.data() + static_cast<size_t>(row) * kTopK);
            std::copy_n(
                row_result.router_logits.data(),
                kNumExperts,
                snapshot.router_probabilities.data() +
                    static_cast<size_t>(row) * kNumExperts);

            const Q8_1Block *q8 =
                kernel.publishedRouterQ8Hidden(row_hidden, 1, kDModel);
            if (!q8)
                throw std::runtime_error("serial CPU router did not publish Q8 hidden row");
            std::copy_n(
                q8,
                blocks_per_row,
                snapshot.q8_hidden.data() +
                    static_cast<size_t>(row) * blocks_per_row);
        }
        return snapshot;
    }

    /** @brief Require exact equality for a trivially copyable vector. */
    template <typename T>
    void expectByteEqual(
        const std::vector<T> &actual,
        const std::vector<T> &expected,
        const char *label,
        int rows)
    {
        ASSERT_EQ(actual.size(), expected.size()) << label << " M=" << rows;
        ASSERT_EQ(
            std::memcmp(
                actual.data(),
                expected.data(),
                actual.size() * sizeof(T)),
            0)
            << label << " is not serial-row byte-equivalent at M=" << rows;
    }

    /** @brief Return the median per-transaction time for one callable. */
    template <typename Function>
    double medianMicros(Function &&function, int samples, int iterations)
    {
        std::vector<double> measurements;
        measurements.reserve(static_cast<size_t>(samples));
        for (int sample = 0; sample < samples; ++sample)
        {
            const auto begin = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < iterations; ++iteration)
                function();
            const auto end = std::chrono::steady_clock::now();
            measurements.push_back(
                std::chrono::duration<double, std::micro>(end - begin).count() /
                static_cast<double>(iterations));
        }
        std::sort(measurements.begin(), measurements.end());
        return measurements[measurements.size() / 2];
    }

    /**
     * @brief Validate and time one runtime-M production router transaction.
     */
    RouteTiming runCase(
        const std::vector<float> &hidden,
        const std::vector<float> &gate,
        int rows,
        int samples,
        int iterations)
    {
        CPUMoEKernel grouped_kernel;
        CPUMoEKernel serial_kernel;

        const RouteSnapshot grouped =
            captureGrouped(grouped_kernel, hidden, gate, rows);
        const RouteSnapshot serial =
            captureSerial(serial_kernel, hidden, gate, rows);
        expectByteEqual(
            grouped.expert_indices, serial.expert_indices,
            "expert indices", rows);
        expectByteEqual(
            grouped.expert_weights, serial.expert_weights,
            "expert weights", rows);
        expectByteEqual(
            grouped.router_probabilities, serial.router_probabilities,
            "router probabilities", rows);
        expectByteEqual(
            grouped.q8_hidden, serial.q8_hidden,
            "router Q8 hidden publication", rows);

        MoERoutingResult grouped_result;
        MoERoutingResult serial_result;
        if (!grouped_kernel.route(
                hidden.data(), gate.data(), rows, kDModel, kNumExperts, kTopK,
                true, grouped_result))
        {
            throw std::runtime_error("grouped CPU router warmup failed");
        }
        if (!serial_kernel.route(
                hidden.data(), gate.data(), 1, kDModel, kNumExperts, kTopK,
                true, serial_result))
        {
            throw std::runtime_error("serial CPU router warmup failed");
        }

        auto grouped_call = [&]()
        {
            if (!grouped_kernel.route(
                    hidden.data(), gate.data(), rows, kDModel, kNumExperts,
                    kTopK, true, grouped_result))
            {
                throw std::runtime_error("timed grouped CPU router failed");
            }
        };
        auto serial_call = [&]()
        {
            for (int row = 0; row < rows; ++row)
            {
                if (!serial_kernel.route(
                        hidden.data() + static_cast<size_t>(row) * kDModel,
                        gate.data(), 1, kDModel, kNumExperts, kTopK,
                        true, serial_result))
                {
                    throw std::runtime_error("timed serial CPU router failed");
                }
            }
        };

        grouped_call();
        serial_call();

        RouteTiming timing;
        timing.rows = rows;
        timing.grouped_us = medianMicros(grouped_call, samples, iterations);
        timing.serial_us = medianMicros(serial_call, samples, iterations);
        timing.speedup = timing.serial_us / timing.grouped_us;
        return timing;
    }
} // namespace

TEST(Perf__CPUMoERouterVerifierRows,
     Qwen36GroupedRouteIsSerialRowByteExactAndEconomicalForRuntimeM)
{
    const std::vector<int> rows = selectedRows();
    const int max_rows = *std::max_element(rows.begin(), rows.end());
    const int samples = envPositiveInt("LLAMINAR_CPU_MOE_ROUTER_SAMPLES", 5);
    const int iterations = envPositiveInt("LLAMINAR_CPU_MOE_ROUTER_ITERATIONS", 30);
    const std::vector<float> hidden = makeHidden(max_rows);
    const std::vector<float> gate = makeGateWeights();

    std::cout << "backend,threads,m,d_model,num_experts,top_k,grouped_us,serial_us,speedup,byte_exact\n";
    for (const int row_count : rows)
    {
        const RouteTiming timing =
            runCase(hidden, gate, row_count, samples, iterations);
        std::cout << "cpu," << omp_get_max_threads() << ','
                  << timing.rows << ',' << kDModel << ',' << kNumExperts << ','
                  << kTopK << ',' << timing.grouped_us << ',' << timing.serial_us
                  << ',' << timing.speedup << ",1\n";

        if (row_count > 1)
        {
            EXPECT_GT(timing.speedup, 1.0)
                << "grouped CPU router must beat repeated production M=1 routing at M="
                << row_count;
        }
    }
}
