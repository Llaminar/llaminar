/**
 * @file Perf__CPUMoEExpertVerifierRows.cpp
 * @brief Production-shaped CPU MoE verifier-FFN correctness and economy gate.
 *
 * Qwen3.6-35B-A3B routes a verifier transaction through expert matrices with
 * `d_model=2048` and `intermediate=512`. On one apportioned dual-socket rank,
 * a depth-three verifier commonly owns about sixteen route rows distributed
 * across fourteen experts: most experts receive one row and a small minority
 * receive two. This harness reproduces that sparse expert-major geometry
 * without loading a model, then measures the exact production primitives:
 *
 * 1. one layer-batched NativeVNNI gate/up projection;
 * 2. fused byte-exact SwiGLU-to-Q8_1 publication; and
 * 3. one layer-batched NativeVNNI down projection.
 *
 * Every grouped intermediate is compared byte-for-byte with independent M=1
 * serial decode before timing begins. The diagnostic serial route is never a
 * production implementation; it is solely the arithmetic oracle required to
 * tune grouped execution safely. IQ2_S is the default because production
 * PerfStats identify codebook 13 for the routed experts in the active
 * Qwen3.6-35B UD-IQ3_S artifact; the file-level quantization label does not
 * imply that every tensor has the same codebook. Set
 * `LLAMINAR_CPU_MOE_EXPERT_FORMATS=all` to run the canonical all-format
 * registry, or provide a comma-separated subset.
 */

#include <gtest/gtest.h>

#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include "kernels/cpu/primitives/SwiGLUPrimitives.h"
#include "utils/QuantizedVerifierFormats.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

using namespace llaminar2;
using llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel;
using llaminar2::cpu::native_vnni::gemv_native_vnni_preq;
using llaminar2::cpu::native_vnni::quantize_activations_to_q8_1;
using llaminar2::cpu::native_vnni::swiglu_quantize_activations_to_q8_1;

namespace
{
    constexpr int kDModel = 2048;
    constexpr int kIntermediate = 512;
    constexpr int kDefaultActiveExperts = 14;
    constexpr int kDefaultLocalRouteRows = 16;

    /** @brief One prepared expert and its immutable packed-weight owners. */
    struct PreparedExpert
    {
        int rows = 0;
        int first_row = 0;
        std::unique_ptr<TensorBase> gate_weights;
        std::unique_ptr<TensorBase> up_weights;
        std::unique_ptr<TensorBase> down_weights;
        std::unique_ptr<CPUNativeVNNIGemmKernel> gate;
        std::unique_ptr<CPUNativeVNNIGemmKernel> up;
        std::unique_ptr<CPUNativeVNNIGemmKernel> down;
    };

    /** @brief Timings for the three production phases and complete pipeline. */
    struct PipelineTiming
    {
        double gate_up_us = 0.0;
        double swiglu_q8_us = 0.0;
        double down_us = 0.0;
        double complete_us = 0.0;
        double persistent_complete_us = 0.0;
        double serial_us = 0.0;
    };

    /** @brief Parse a positive integer environment setting. */
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
            throw std::invalid_argument(
                std::string(name) + " must be a positive integer");
        }
        return static_cast<int>(parsed);
    }

    /** @brief Parse a non-negative integer environment setting. */
    int envNonNegativeInt(const char *name, int fallback)
    {
        const char *raw = std::getenv(name);
        if (!raw || !*raw)
            return fallback;
        char *end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end == raw || *end != '\0' || parsed < 0 ||
            parsed > std::numeric_limits<int>::max())
        {
            throw std::invalid_argument(
                std::string(name) + " must be a non-negative integer");
        }
        return static_cast<int>(parsed);
    }

    /** @brief Parse the optional comma-separated format selection. */
    std::set<std::string> selectedFormats()
    {
        const char *raw = std::getenv("LLAMINAR_CPU_MOE_EXPERT_FORMATS");
        if (!raw || !*raw)
            return {"IQ2_S"};

        std::set<std::string> formats;
        std::stringstream stream(raw);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            const auto first = token.find_first_not_of(" \t\r\n");
            const auto last = token.find_last_not_of(" \t\r\n");
            if (first == std::string::npos)
                continue;
            formats.insert(token.substr(first, last - first + 1u));
        }
        if (formats.empty())
        {
            throw std::invalid_argument(
                "LLAMINAR_CPU_MOE_EXPERT_FORMATS selected no formats");
        }
        return formats;
    }

    /**
     * @brief Build a sparse expert-major row distribution.
     *
     * Every expert receives one row first. Remaining rows are assigned from
     * the start of the expert list, producing the common verifier pattern of a
     * few two-row experts and many one-row experts without manufacturing an
     * unrealistically dense expert batch.
     */
    std::vector<int> routeRowsPerExpert(int active_experts, int total_rows)
    {
        if (active_experts <= 0 || total_rows < active_experts)
        {
            throw std::invalid_argument(
                "CPU MoE expert perf geometry requires rows >= active experts");
        }
        std::vector<int> rows(static_cast<size_t>(active_experts), 1);
        for (int row = active_experts; row < total_rows; ++row)
            ++rows[static_cast<size_t>((row - active_experts) % active_experts)];
        return rows;
    }

    /** @brief Produce deterministic, non-trivial router-published hidden rows. */
    std::vector<float> makeHiddenRows(int rows)
    {
        std::vector<float> hidden(
            static_cast<size_t>(rows) * static_cast<size_t>(kDModel));
        for (size_t index = 0; index < hidden.size(); ++index)
        {
            const int centered =
                static_cast<int>((index * 37u + index / 13u) % 257u) - 128;
            hidden[index] =
                static_cast<float>(centered) * 0.004125f +
                static_cast<float>(static_cast<int>(index % 11u) - 5) *
                    0.000071f;
        }
        return hidden;
    }

    /** @brief Require byte equality and report the first divergent byte. */
    template <typename T>
    void expectByteEqual(
        const std::vector<T> &actual,
        const std::vector<T> &expected,
        const std::string &label)
    {
        ASSERT_EQ(actual.size(), expected.size()) << label;
        const size_t bytes = actual.size() * sizeof(T);
        if (std::memcmp(actual.data(), expected.data(), bytes) == 0)
            return;

        const auto *actual_bytes =
            reinterpret_cast<const uint8_t *>(actual.data());
        const auto *expected_bytes =
            reinterpret_cast<const uint8_t *>(expected.data());
        size_t first = 0;
        while (first < bytes && actual_bytes[first] == expected_bytes[first])
            ++first;
        ADD_FAILURE()
            << label << " differs from serial decode at byte " << first
            << " actual=" << static_cast<unsigned>(actual_bytes[first])
            << " expected=" << static_cast<unsigned>(expected_bytes[first]);
    }

    /** @brief Return median latency after untimed warmup. */
    template <typename Function>
    double medianMicros(Function &&function, int warmup, int samples)
    {
        for (int iteration = 0; iteration < warmup; ++iteration)
            function();

        std::vector<double> measurements;
        measurements.reserve(static_cast<size_t>(samples));
        for (int sample = 0; sample < samples; ++sample)
        {
            const auto begin = std::chrono::steady_clock::now();
            function();
            const auto end = std::chrono::steady_clock::now();
            measurements.push_back(
                std::chrono::duration<double, std::micro>(end - begin).count());
        }
        std::sort(measurements.begin(), measurements.end());
        return measurements[measurements.size() / 2u];
    }

    /**
     * @brief Prepare all matrices and kernels for one canonical codebook.
     */
    std::vector<PreparedExpert> prepareExperts(
        const test::QuantizedVerifierFormatCase &format,
        const std::vector<int> &rows_per_expert)
    {
        std::vector<PreparedExpert> experts;
        experts.reserve(rows_per_expert.size());
        int first_row = 0;
        for (size_t expert_index = 0;
             expert_index < rows_per_expert.size();
             ++expert_index)
        {
            PreparedExpert expert;
            expert.rows = rows_per_expert[expert_index];
            expert.first_row = first_row;
            first_row += expert.rows;

            const uint32_t seed = static_cast<uint32_t>(
                41000u + expert_index * 101u + format.source_codebook_id);
            expert.gate_weights = format.create(
                {static_cast<size_t>(kIntermediate),
                 static_cast<size_t>(kDModel)},
                seed + 1u);
            expert.up_weights = format.create(
                {static_cast<size_t>(kIntermediate),
                 static_cast<size_t>(kDModel)},
                seed + 2u);
            expert.down_weights = format.create(
                {static_cast<size_t>(kDModel),
                 static_cast<size_t>(kIntermediate)},
                seed + 3u);
            if (!expert.gate_weights || !expert.up_weights ||
                !expert.down_weights)
            {
                throw std::runtime_error(
                    std::string("Could not create CPU MoE weights for ") +
                    format.label);
            }

            expert.gate = std::make_unique<CPUNativeVNNIGemmKernel>(
                expert.gate_weights.get());
            expert.up = std::make_unique<CPUNativeVNNIGemmKernel>(
                expert.up_weights.get());
            expert.down = std::make_unique<CPUNativeVNNIGemmKernel>(
                expert.down_weights.get());
            if (!expert.gate->isValid() || !expert.up->isValid() ||
                !expert.down->isValid())
            {
                throw std::runtime_error(
                    std::string("NativeVNNI preparation failed for ") +
                    format.label);
            }
            experts.push_back(std::move(expert));
        }
        return experts;
    }

    /**
     * @brief Validate and time one production-shaped grouped expert pipeline.
     */
    PipelineTiming runFormat(
        const test::QuantizedVerifierFormatCase &format,
        int active_experts,
        int local_route_rows,
        int warmup,
        int samples,
        int profile_iterations,
        const std::string &profile_phase)
    {
        const std::vector<int> rows_per_expert =
            routeRowsPerExpert(active_experts, local_route_rows);
        std::vector<PreparedExpert> experts =
            prepareExperts(format, rows_per_expert);

        const int hidden_blocks =
            kDModel / static_cast<int>(Q8_1Block::BLOCK_SIZE);
        const int activation_blocks =
            kIntermediate / static_cast<int>(Q8_1Block::BLOCK_SIZE);
        const std::vector<float> hidden = makeHiddenRows(local_route_rows);
        std::vector<Q8_1Block> router_q8(
            static_cast<size_t>(local_route_rows) * hidden_blocks);
        quantize_activations_to_q8_1(
            hidden.data(),
            router_q8.data(),
            local_route_rows,
            kDModel,
            hidden_blocks);

        std::vector<float> grouped_gate(
            static_cast<size_t>(local_route_rows) * kIntermediate);
        std::vector<float> grouped_up(grouped_gate.size());
        std::vector<Q8_1Block> grouped_activation(
            static_cast<size_t>(local_route_rows) * activation_blocks);
        std::vector<float> grouped_down(
            static_cast<size_t>(local_route_rows) * kDModel);

        std::vector<float> serial_gate(grouped_gate.size());
        std::vector<float> serial_up(grouped_up.size());
        std::vector<float> serial_activated(grouped_gate.size());
        std::vector<Q8_1Block> serial_activation(grouped_activation.size());
        std::vector<float> serial_down(grouped_down.size());

        using Descriptor =
            CPUNativeVNNIGemmKernel::BatchedPrequantizedProjectionDesc;
        std::vector<Descriptor> gate_up_descriptors;
        std::vector<Descriptor> down_descriptors;
        gate_up_descriptors.reserve(static_cast<size_t>(active_experts) * 2u);
        down_descriptors.reserve(static_cast<size_t>(active_experts));
        for (PreparedExpert &expert : experts)
        {
            const size_t row = static_cast<size_t>(expert.first_row);
            const Q8_1Block *expert_input =
                router_q8.data() + row * hidden_blocks;
            gate_up_descriptors.push_back({
                .kernel = expert.gate.get(),
                .input_q8 = expert_input,
                .output = grouped_gate.data() + row * kIntermediate,
                .bias = nullptr,
                .rows = expert.rows,
                .n = kIntermediate,
                .ldc = kIntermediate,
            });
            gate_up_descriptors.push_back({
                .kernel = expert.up.get(),
                .input_q8 = expert_input,
                .output = grouped_up.data() + row * kIntermediate,
                .bias = nullptr,
                .rows = expert.rows,
                .n = kIntermediate,
                .ldc = kIntermediate,
            });
            down_descriptors.push_back({
                .kernel = expert.down.get(),
                .input_q8 = grouped_activation.data() +
                    row * activation_blocks,
                .output = grouped_down.data() + row * kDModel,
                .bias = nullptr,
                .rows = expert.rows,
                .n = kDModel,
                .ldc = kDModel,
            });
        }

        auto grouped_gate_up = [&]()
        {
            if (!CPUNativeVNNIGemmKernel::
                    multiply_batched_preq_decode_equivalent(
                        gate_up_descriptors.data(),
                        static_cast<int>(gate_up_descriptors.size()),
                        kDModel))
            {
                throw std::runtime_error("Grouped CPU MoE gate/up failed");
            }
        };
        auto grouped_swiglu_q8 = [&]()
        {
            swiglu_quantize_activations_to_q8_1(
                grouped_gate.data(),
                grouped_up.data(),
                grouped_activation.data(),
                local_route_rows,
                kIntermediate,
                activation_blocks);
        };
        auto grouped_down_projection = [&]()
        {
            if (!CPUNativeVNNIGemmKernel::
                    multiply_batched_preq_decode_equivalent(
                        down_descriptors.data(),
                        static_cast<int>(down_descriptors.size()),
                        kIntermediate))
            {
                throw std::runtime_error("Grouped CPU MoE down failed");
            }
        };
        auto grouped_complete = [&]()
        {
            grouped_gate_up();
            grouped_swiglu_q8();
            grouped_down_projection();
        };
        auto grouped_complete_persistent = [&]()
        {
            if (!CPUNativeVNNIGemmKernel::
                    execute_moe_grouped_ffn_transaction_preq_decode_equivalent(
                        gate_up_descriptors.data(),
                        static_cast<int>(gate_up_descriptors.size()),
                        kDModel,
                        grouped_gate.data(),
                        grouped_up.data(),
                        grouped_activation.data(),
                        local_route_rows,
                        kIntermediate,
                        activation_blocks,
                        down_descriptors.data(),
                        static_cast<int>(down_descriptors.size())))
            {
                throw std::runtime_error(
                    "Persistent-team grouped CPU MoE transaction failed");
            }
        };

        auto serial_complete = [&]()
        {
            for (const PreparedExpert &expert : experts)
            {
                for (int local_row = 0; local_row < expert.rows; ++local_row)
                {
                    const int row = expert.first_row + local_row;
                    const Q8_1Block *hidden_row =
                        router_q8.data() +
                        static_cast<size_t>(row) * hidden_blocks;
                    float *gate_row = serial_gate.data() +
                        static_cast<size_t>(row) * kIntermediate;
                    float *up_row = serial_up.data() +
                        static_cast<size_t>(row) * kIntermediate;
                    float *activated_row = serial_activated.data() +
                        static_cast<size_t>(row) * kIntermediate;
                    Q8_1Block *activation_row = serial_activation.data() +
                        static_cast<size_t>(row) * activation_blocks;
                    float *down_row = serial_down.data() +
                        static_cast<size_t>(row) * kDModel;

                    gemv_native_vnni_preq(
                        expert.gate->packedWeights(), hidden_row, gate_row);
                    gemv_native_vnni_preq(
                        expert.up->packedWeights(), hidden_row, up_row);
                    primitives::compute_swiglu_serial(
                        gate_row,
                        up_row,
                        activated_row,
                        kIntermediate);
                    quantize_activations_to_q8_1(
                        activated_row,
                        activation_row,
                        1,
                        kIntermediate,
                        activation_blocks);
                    gemv_native_vnni_preq(
                        expert.down->packedWeights(),
                        activation_row,
                        down_row);
                }
            }
        };

        grouped_complete();
        serial_complete();
        expectByteEqual(
            grouped_gate,
            serial_gate,
            std::string(format.label) + " gate projection");
        expectByteEqual(
            grouped_up,
            serial_up,
            std::string(format.label) + " up projection");
        expectByteEqual(
            grouped_activation,
            serial_activation,
            std::string(format.label) + " SwiGLU Q8_1 publication");
        expectByteEqual(
            grouped_down,
            serial_down,
            std::string(format.label) + " down projection");

        grouped_complete_persistent();
        expectByteEqual(
            grouped_gate,
            serial_gate,
            std::string(format.label) + " persistent-team gate projection");
        expectByteEqual(
            grouped_up,
            serial_up,
            std::string(format.label) + " persistent-team up projection");
        expectByteEqual(
            grouped_activation,
            serial_activation,
            std::string(format.label) +
                " persistent-team SwiGLU Q8_1 publication");
        expectByteEqual(
            grouped_down,
            serial_down,
            std::string(format.label) + " persistent-team down projection");

        if (profile_iterations > 0)
        {
            /*
             * Profiler mode deliberately repeats one already-prepared
             * production phase. Weight construction, packing, and the serial
             * oracle remain one-time setup, so `perf stat` and `perf record`
             * attribute nearly all samples to the selected hot kernel family.
             */
            for (int iteration = 0;
                 iteration < profile_iterations;
                 ++iteration)
            {
                if (profile_phase == "gate_up")
                    grouped_gate_up();
                else if (profile_phase == "swiglu_q8")
                    grouped_swiglu_q8();
                else if (profile_phase == "down")
                    grouped_down_projection();
                else if (profile_phase == "complete")
                    grouped_complete();
                else if (profile_phase == "persistent")
                    grouped_complete_persistent();
                else
                    throw std::invalid_argument(
                        "LLAMINAR_CPU_MOE_EXPERT_PROFILE_PHASE must be "
                        "gate_up, swiglu_q8, down, complete, or persistent");
            }
            const float checksum =
                profile_phase == "gate_up"
                    ? grouped_gate.front() + grouped_up.back()
                    : profile_phase == "swiglu_q8"
                        ? grouped_activation.front().d +
                              grouped_activation.back().d
                        : grouped_down.front() + grouped_down.back();
            std::cerr
                << "[CPUMoEExpertProfiler] format=" << format.label
                << " phase=" << profile_phase
                << " iterations=" << profile_iterations
                << " checksum=" << checksum << '\n';
        }

        PipelineTiming timing;
        timing.gate_up_us =
            medianMicros(grouped_gate_up, warmup, samples);
        timing.swiglu_q8_us =
            medianMicros(grouped_swiglu_q8, warmup, samples);
        timing.down_us =
            medianMicros(grouped_down_projection, warmup, samples);
        timing.complete_us =
            medianMicros(grouped_complete, warmup, samples);
        timing.persistent_complete_us =
            medianMicros(grouped_complete_persistent, warmup, samples);
        timing.serial_us =
            medianMicros(serial_complete, /*warmup=*/1, samples);
        return timing;
    }
} // namespace

/**
 * @brief Prove and time the production-shaped grouped CPU expert transaction.
 */
TEST(Perf_CPUMoEExpertVerifierRows, ProductionGeometryByteExactAndEconomical)
{
    const int active_experts = envPositiveInt(
        "LLAMINAR_CPU_MOE_EXPERT_ACTIVE_EXPERTS",
        kDefaultActiveExperts);
    const int local_route_rows = envPositiveInt(
        "LLAMINAR_CPU_MOE_EXPERT_ROUTE_ROWS",
        kDefaultLocalRouteRows);
    const int warmup = envPositiveInt(
        "LLAMINAR_CPU_MOE_EXPERT_WARMUP",
        3);
    const int samples = envPositiveInt(
        "LLAMINAR_CPU_MOE_EXPERT_SAMPLES",
        9);
    const int profile_iterations = envNonNegativeInt(
        "LLAMINAR_CPU_MOE_EXPERT_PROFILE_ITERATIONS",
        0);
    const char *profile_phase_raw =
        std::getenv("LLAMINAR_CPU_MOE_EXPERT_PROFILE_PHASE");
    const std::string profile_phase =
        profile_phase_raw && *profile_phase_raw
            ? profile_phase_raw
            : "complete";
    ASSERT_LE(active_experts, 256);
    ASSERT_GE(local_route_rows, active_experts);

    const std::set<std::string> requested = selectedFormats();
    const bool all_formats = requested.count("all") != 0u;
    int executed = 0;
    std::cout
        << "\nCPU MoE verifier expert speedometer"
        << " (threads=" << omp_get_max_threads()
        << ", active_experts=" << active_experts
        << ", local_route_rows=" << local_route_rows << ")\n"
        << "format gate_up_us swiglu_q8_us down_us complete_us "
           "persistent_us serial_us speedup persistent_speedup\n";

    for (const auto &format : test::quantizedMoEVerifierFormats())
    {
        if (!all_formats && requested.count(format.label) == 0u)
            continue;

        const PipelineTiming timing = runFormat(
            format,
            active_experts,
            local_route_rows,
            warmup,
            samples,
            profile_iterations,
            profile_phase);
        const double speedup = timing.complete_us > 0.0
            ? timing.serial_us / timing.complete_us
            : 0.0;
        const double persistent_speedup = timing.persistent_complete_us > 0.0
            ? timing.serial_us / timing.persistent_complete_us
            : 0.0;
        std::cout
            << format.label << ' '
            << timing.gate_up_us << ' '
            << timing.swiglu_q8_us << ' '
            << timing.down_us << ' '
            << timing.complete_us << ' '
            << timing.persistent_complete_us << ' '
            << timing.serial_us << ' '
            << speedup << ' '
            << persistent_speedup << '\n';
        EXPECT_GT(speedup, 1.0)
            << format.label
            << " grouped verifier expert execution must beat serial row replay";
        ++executed;
    }
    ASSERT_GT(executed, 0)
        << "No canonical CPU MoE expert formats matched the selection";
}
