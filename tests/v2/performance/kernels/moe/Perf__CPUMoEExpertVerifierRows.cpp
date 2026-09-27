/**
 * @file Perf__CPUMoEExpertVerifierRows.cpp
 * @brief Production-shaped CPU MoE verifier-FFN correctness and economy gate.
 *
 * By default Qwen3.6-35B-A3B routes a verifier transaction through expert
 * matrices with `d_model=2048` and `intermediate=512`. On one apportioned
 * dual-socket rank,
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
 * tune grouped execution safely. The benchmark measures both the GPU-aligned
 * production expert contract and the otherwise-identical backend-native CPU
 * contract so parity-preserving arithmetic changes carry a visible economy
 * cost. IQ2_S remains the default uniform-format diagnostic. The real
 * Qwen3.6-35B UD-IQ3_S artifact mostly pairs IQ2_S gate/up with IQ4_XS down
 * weights; its file-level quantization label does not describe each tensor.
 * Set `LLAMINAR_CPU_MOE_EXPERT_DOWN_FORMAT=IQ4_XS` to measure that mixed
 * transaction. Both formats are recorded in the timing evidence, and every
 * intermediate still uses its own independent serial-row oracle. Set
 * `LLAMINAR_CPU_MOE_EXPERT_FORMATS=all` to run the canonical all-format
 * registry, or provide a comma-separated subset. Timed samples execute a
 * configurable transaction batch (`LLAMINAR_CPU_MOE_EXPERT_BATCH_ITERATIONS`)
 * so sub-millisecond OpenMP wake-up jitter cannot masquerade as a kernel
 * regression. After setup and byte validation, profiler launches repeat only
 * one format and phase under the policy selected by
 * `LLAMINAR_CPU_MOE_EXPERT_PROFILE_POLICY=native|aligned`. They never collect
 * timing samples or run the other policy afterward; profiler overhead must
 * not enter a canonical timing CSV.
 */

#include "../../../utils/NativeVNNITestPartialStorage.h"
#include <gtest/gtest.h>

#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include "kernels/cpu/primitives/SwiGLUPrimitives.h"
#include "utils/CPUFeatures.h"
#include "utils/QuantizedVerifierFormats.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <omp.h>

using namespace llaminar2;
using llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel;
using llaminar2::cpu::native_vnni::VerifierRowsPolicy;
using llaminar2::cpu::native_vnni::gemv_native_vnni_preq;
using llaminar2::cpu::native_vnni::quantize_activations_to_q8_1;
using llaminar2::cpu::native_vnni::swiglu_quantize_activations_to_q8_1;

namespace
{
    constexpr int kDefaultDModel = 2048;
    constexpr int kDefaultIntermediate = 512;
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

    /** @brief Numerical policy isolated by an external profiler launch. */
    enum class ProfilePolicy : uint8_t
    {
        Native,
        Aligned,
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

    /** @brief Parse a forceable grouped-row policy for exact perf comparison. */
    VerifierRowsPolicy selectedVerifierPolicy()
    {
        const char *raw =
            std::getenv("LLAMINAR_CPU_MOE_EXPERT_VERIFIER_POLICY");
        const std::string policy = raw && *raw ? raw : "Auto";
        if (policy == "Auto")
            return VerifierRowsPolicy::Auto;
        if (policy == "Pairwise")
            return VerifierRowsPolicy::Pairwise;
        if (policy == "WideRows")
            return VerifierRowsPolicy::WideRows;
        if (policy == "FullKRowChunkGrid")
            return VerifierRowsPolicy::FullKRowChunkGrid;
        if (policy == "FullKTwoRowNbc1")
            return VerifierRowsPolicy::FullKTwoRowNbc1;
        if (policy == "FullKTwoRowNbc2")
            return VerifierRowsPolicy::FullKTwoRowNbc2;
        if (policy == "FullKTwoRowPairGridNbc1")
            return VerifierRowsPolicy::FullKTwoRowPairGridNbc1;
        if (policy == "FullKTwoRowPairGridNbc2")
            return VerifierRowsPolicy::FullKTwoRowPairGridNbc2;
        if (policy == "FullKTwoRowPairGridNbc4")
            return VerifierRowsPolicy::FullKTwoRowPairGridNbc4;
        if (policy == "FullKTwoRowPairGridNbc8")
            return VerifierRowsPolicy::FullKTwoRowPairGridNbc8;
        throw std::invalid_argument(
            "LLAMINAR_CPU_MOE_EXPERT_VERIFIER_POLICY names an unknown policy");
    }

    /** @brief Parse the one policy that a profiler launch may exercise. */
    ProfilePolicy selectedProfilePolicy()
    {
        const char *raw =
            std::getenv("LLAMINAR_CPU_MOE_EXPERT_PROFILE_POLICY");
        const std::string policy = raw && *raw ? raw : "aligned";
        if (policy == "native")
            return ProfilePolicy::Native;
        if (policy == "aligned")
            return ProfilePolicy::Aligned;
        throw std::invalid_argument(
            "LLAMINAR_CPU_MOE_EXPERT_PROFILE_POLICY must be native or aligned");
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
     * @brief Resolve an explicit down format from the same canonical inventory.
     * @param gate_up Default format when the diagnostic requests uniform weights.
     * @return Stable registry entry; unknown names fail before preparing weights.
     */
    const test::QuantizedVerifierFormatCase &selectedDownFormat(
        const test::QuantizedVerifierFormatCase &gate_up)
    {
        const char *raw = std::getenv("LLAMINAR_CPU_MOE_EXPERT_DOWN_FORMAT");
        if (!raw || !*raw)
            return gate_up;
        return test::quantizedMoEVerifierFormat(raw);
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
    std::vector<float> makeHiddenRows(int rows, int d_model)
    {
        std::vector<float> hidden(
            static_cast<size_t>(rows) * static_cast<size_t>(d_model));
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

    /**
     * @brief Return median per-transaction latency from batched samples.
     *
     * One production transaction is shorter than an OpenMP scheduling quantum
     * on this geometry. Repeating it inside one clock interval makes timing
     * proportional to kernel work while retaining the same prepared weights,
     * worker team, and rotating expert working set.
     */
    template <typename Function>
    double medianMicros(
        Function &&function,
        int warmup,
        int samples,
        int batch_iterations)
    {
        for (int iteration = 0; iteration < warmup; ++iteration)
        {
            for (int batch = 0; batch < batch_iterations; ++batch)
                function();
        }

        std::vector<double> measurements;
        measurements.reserve(static_cast<size_t>(samples));
        for (int sample = 0; sample < samples; ++sample)
        {
            const auto begin = std::chrono::steady_clock::now();
            for (int batch = 0; batch < batch_iterations; ++batch)
                function();
            const auto end = std::chrono::steady_clock::now();
            measurements.push_back(
                std::chrono::duration<double, std::micro>(end - begin).count() /
                static_cast<double>(batch_iterations));
        }
        std::sort(measurements.begin(), measurements.end());
        return measurements[measurements.size() / 2u];
    }

    /**
     * @brief Prepare each projection with its explicitly selected source format.
     * @param format Gate/up source-format owner from the canonical registry.
     * @param down_format Independently selected down source-format owner.
     * @param rows_per_expert Active expert-major row counts.
     * @param d_model Logical hidden width.
     * @param intermediate Logical FFN width.
     * @param numerical_policy Shared arithmetic contract, not a weight format.
     * @return Immutable weights and prepared engines for every active expert.
     */
    std::vector<PreparedExpert> prepareExperts(
        const test::QuantizedVerifierFormatCase &format,
        const test::QuantizedVerifierFormatCase &down_format,
        const std::vector<int> &rows_per_expert,
        int d_model,
        int intermediate,
        CPUProjectionNumericalPolicy numerical_policy)
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
                {static_cast<size_t>(intermediate),
                 static_cast<size_t>(d_model)},
                seed + 1u);
            expert.up_weights = format.create(
                {static_cast<size_t>(intermediate),
                 static_cast<size_t>(d_model)},
                seed + 2u);
            expert.down_weights = down_format.create(
                {static_cast<size_t>(d_model),
                 static_cast<size_t>(intermediate)},
                seed + 3u);
            if (!expert.gate_weights || !expert.up_weights ||
                !expert.down_weights)
            {
                throw std::runtime_error(
                    std::string("Could not create CPU MoE weights for ") +
                    format.label);
            }

            expert.gate = std::make_unique<CPUNativeVNNIGemmKernel>(
                expert.gate_weights.get(), 0, -1, numerical_policy);
            expert.up = std::make_unique<CPUNativeVNNIGemmKernel>(
                expert.up_weights.get(), 0, -1, numerical_policy);
            expert.down = std::make_unique<CPUNativeVNNIGemmKernel>(
                expert.down_weights.get(), 0, -1, numerical_policy);
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
     * @param format Source format for gate/up weights.
     * @param down_format Source format for down weights, independently prepared.
     * @param active_experts Number of expert members in the transaction.
     * @param local_route_rows Total expert-major activation rows.
     * @param d_model Logical hidden width.
     * @param intermediate Logical FFN width.
     * @param numerical_policy Independent native or GPU-aligned arithmetic contract.
     * @param verifier_schedule Explicit diagnostic candidate, or production Auto.
     * @param warmup Untimed batches before measurements.
     * @param samples Number of independently timed batches.
     * @param batch_iterations Transactions per timing interval.
     * @param profile_iterations Isolated profiler iterations; zero selects timing.
     * @param profile_phase Exactly one operation to repeat in profiler mode.
     * @return Phase/transaction medians, or an empty record for profiler-only work.
     */
    PipelineTiming runFormat(
        const test::QuantizedVerifierFormatCase &format,
        const test::QuantizedVerifierFormatCase &down_format,
        int active_experts,
        int local_route_rows,
        int d_model,
        int intermediate,
        CPUProjectionNumericalPolicy numerical_policy,
        VerifierRowsPolicy verifier_schedule,
        int warmup,
        int samples,
        int batch_iterations,
        int profile_iterations,
        const std::string &profile_phase)
    {
        const std::vector<int> rows_per_expert =
            routeRowsPerExpert(active_experts, local_route_rows);
        std::vector<PreparedExpert> experts =
            prepareExperts(
                format,
                down_format,
                rows_per_expert,
                d_model,
                intermediate,
                numerical_policy);

        const int hidden_blocks =
            d_model / static_cast<int>(Q8_1Block::BLOCK_SIZE);
        const int activation_blocks =
            intermediate / static_cast<int>(Q8_1Block::BLOCK_SIZE);
        const std::vector<float> hidden =
            makeHiddenRows(local_route_rows, d_model);
        std::vector<Q8_1Block> router_q8(
            static_cast<size_t>(local_route_rows) * hidden_blocks);
        quantize_activations_to_q8_1(
            hidden.data(),
            router_q8.data(),
            local_route_rows,
            d_model,
            hidden_blocks,
            numerical_policy);

        std::vector<float> grouped_gate(
            static_cast<size_t>(local_route_rows) * intermediate);
        std::vector<float> grouped_up(grouped_gate.size());
        std::vector<Q8_1Block> grouped_activation(
            static_cast<size_t>(local_route_rows) * activation_blocks);
        std::vector<float> grouped_down(
            static_cast<size_t>(local_route_rows) * d_model);

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
                .output = grouped_gate.data() + row * intermediate,
                .bias = nullptr,
                .rows = expert.rows,
                .n = intermediate,
                .ldc = intermediate,
                .verifier_schedule = verifier_schedule,
            });
            gate_up_descriptors.push_back({
                .kernel = expert.up.get(),
                .input_q8 = expert_input,
                .output = grouped_up.data() + row * intermediate,
                .bias = nullptr,
                .rows = expert.rows,
                .n = intermediate,
                .ldc = intermediate,
                .verifier_schedule = verifier_schedule,
            });
            down_descriptors.push_back({
                .kernel = expert.down.get(),
                .input_q8 = grouped_activation.data() +
                    row * activation_blocks,
                .output = grouped_down.data() + row * d_model,
                .bias = nullptr,
                .rows = expert.rows,
                .n = d_model,
                .ldc = d_model,
                .verifier_schedule = verifier_schedule,
            });
        }

        using PartialStorage = llaminar2::test::NativeVNNITestPartialStorage;
        PartialStorage grouped_partials(std::max(
            PartialStorage::bundleFloats(gate_up_descriptors.data(), gate_up_descriptors.size(), 1),
            PartialStorage::bundleFloats(down_descriptors.data(), down_descriptors.size(), 1)));
        auto grouped_gate_up = [&]()
        {
            if (!CPUNativeVNNIGemmKernel::
                    multiply_batched_preq_decode_equivalent(
                        gate_up_descriptors.data(),
                        static_cast<int>(gate_up_descriptors.size()),
                        d_model, grouped_partials.span()))
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
                intermediate,
                activation_blocks,
                numerical_policy);
        };
        auto grouped_down_projection = [&]()
        {
            if (!CPUNativeVNNIGemmKernel::
                    multiply_batched_preq_decode_equivalent(
                        down_descriptors.data(),
                        static_cast<int>(down_descriptors.size()),
                        intermediate, grouped_partials.span()))
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
                        d_model,
                        grouped_gate.data(),
                        grouped_up.data(),
                        grouped_activation.data(),
                        local_route_rows,
                        intermediate,
                        activation_blocks,
                        down_descriptors.data(),
                        static_cast<int>(down_descriptors.size()), grouped_partials.span()))
            {
                throw std::runtime_error(
                    "Persistent-team grouped CPU MoE transaction failed");
            }
        };

        // Serial oracle projections share the largest exact demand, prepared
        // outside every warmup, timer and isolated profiler interval.
        size_t partial_floats = 0;
        for (const PreparedExpert &expert : experts)
            for (const auto *kernel : {expert.gate.get(), expert.up.get(), expert.down.get()})
                partial_floats = std::max(partial_floats,
                    cpu::native_vnni::nativeVNNIProjectionPartialFloats(kernel->packedWeights(), 1));
        test::NativeVNNITestPartialStorage partial_storage(partial_floats);
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
                        static_cast<size_t>(row) * intermediate;
                    float *up_row = serial_up.data() +
                        static_cast<size_t>(row) * intermediate;
                    float *activated_row = serial_activated.data() +
                        static_cast<size_t>(row) * intermediate;
                    Q8_1Block *activation_row = serial_activation.data() +
                        static_cast<size_t>(row) * activation_blocks;
                    float *down_row = serial_down.data() +
                        static_cast<size_t>(row) * d_model;

                    gemv_native_vnni_preq(
                        expert.gate->packedWeights(), hidden_row, gate_row, partial_storage.span());
                    gemv_native_vnni_preq(
                        expert.up->packedWeights(), hidden_row, up_row, partial_storage.span());
                    if (numerical_policy ==
                        CPUProjectionNumericalPolicy::GPUAlignedExpert)
                    {
                        primitives::compute_swiglu_gpu_aligned_expert_serial(
                            gate_row,
                            up_row,
                            activated_row,
                            intermediate);
                    }
                    else
                    {
                        primitives::compute_swiglu_serial(
                            gate_row,
                            up_row,
                            activated_row,
                            intermediate);
                    }
                    quantize_activations_to_q8_1(
                        activated_row,
                        activation_row,
                        1,
                        intermediate,
                        activation_blocks,
                        numerical_policy);
                    gemv_native_vnni_preq(
                        expert.down->packedWeights(),
                        activation_row,
                        down_row, partial_storage.span());
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
                << " down_format=" << down_format.label
                << " phase=" << profile_phase
                << " iterations=" << profile_iterations
                << " checksum=" << checksum << '\n';
            // Profiling and paired timing are disjoint executions. Returning
            // here also prevents later phases from polluting the hot samples.
            return {};
        }

        PipelineTiming timing;
        timing.gate_up_us =
            medianMicros(grouped_gate_up, warmup, samples, batch_iterations);
        timing.swiglu_q8_us =
            medianMicros(grouped_swiglu_q8, warmup, samples, batch_iterations);
        timing.down_us =
            medianMicros(
                grouped_down_projection, warmup, samples, batch_iterations);
        timing.complete_us =
            medianMicros(grouped_complete, warmup, samples, batch_iterations);
        timing.persistent_complete_us =
            medianMicros(
                grouped_complete_persistent, warmup, samples, batch_iterations);
        timing.serial_us =
            medianMicros(
                serial_complete, /*warmup=*/1, samples, batch_iterations);
        return timing;
    }
} // namespace

/**
 * @brief Prove and time the production-shaped grouped CPU expert transaction.
 */
TEST(Perf_CPUMoEExpertVerifierRows, ProductionGeometryByteExactAndEconomical)
{
    const int d_model = envPositiveInt(
        "LLAMINAR_CPU_MOE_EXPERT_D_MODEL",
        kDefaultDModel);
    const int intermediate = envPositiveInt(
        "LLAMINAR_CPU_MOE_EXPERT_INTERMEDIATE",
        kDefaultIntermediate);
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
    const int batch_iterations = envPositiveInt(
        "LLAMINAR_CPU_MOE_EXPERT_BATCH_ITERATIONS",
        32);
    const int profile_iterations = envNonNegativeInt(
        "LLAMINAR_CPU_MOE_EXPERT_PROFILE_ITERATIONS",
        0);
    const char *profile_phase_raw =
        std::getenv("LLAMINAR_CPU_MOE_EXPERT_PROFILE_PHASE");
    const std::string profile_phase =
        profile_phase_raw && *profile_phase_raw
            ? profile_phase_raw
            : "complete";
    const ProfilePolicy profile_policy = selectedProfilePolicy();
    const VerifierRowsPolicy verifier_schedule = selectedVerifierPolicy();
    ASSERT_LE(active_experts, 256);
    ASSERT_GE(local_route_rows, active_experts);
    ASSERT_EQ(d_model % static_cast<int>(Q8_1Block::BLOCK_SIZE), 0)
        << "d_model must contain complete Q8_1 blocks";
    ASSERT_EQ(intermediate % static_cast<int>(Q8_1Block::BLOCK_SIZE), 0)
        << "intermediate must contain complete Q8_1 blocks";

    const std::set<std::string> requested = selectedFormats();
    const bool all_formats = requested.count("all") != 0u;
    const char *csv_path = std::getenv("LLAMINAR_CPU_MOE_EXPERT_POLICY_CSV");
    if (profile_iterations > 0)
    {
        // One process is one physical profiling experiment. Reject ambiguous
        // collection before creating weights or opening an output file.
        ASSERT_FALSE(all_formats);
        ASSERT_EQ(requested.size(), 1u)
            << "Profiler mode requires exactly one source format";
        ASSERT_FALSE(csv_path && *csv_path)
            << "Profiler mode cannot publish canonical timing CSV evidence";
    }
    std::ofstream csv;
    if (csv_path && *csv_path)
    {
        csv.open(csv_path, std::ios::trunc);
        ASSERT_TRUE(csv.is_open()) << "Could not open CSV output " << csv_path;
        csv << "format,down_format,isa,d_model,intermediate,active_experts,route_rows,"
               "threads,native_gate_up_us,native_swiglu_q8_us,native_down_us,"
               "native_complete_us,native_persistent_us,"
               "aligned_gate_up_us,aligned_swiglu_q8_us,aligned_down_us,"
               "aligned_complete_us,aligned_persistent_us,"
               "aligned_over_native_persistent\n";
    }
    int executed = 0;
    std::cout
        << "\nCPU MoE expert numerical-policy speedometer"
        << " (threads=" << omp_get_max_threads()
        << ", d_model=" << d_model
        << ", intermediate=" << intermediate
        << ", active_experts=" << active_experts
        << ", local_route_rows=" << local_route_rows
        << ", verifier_policy="
        << cpu::native_vnni::verifierRowsPolicyName(verifier_schedule)
        << ")\n"
        << "gate_up_format/down_format native_persistent_us aligned_persistent_us "
           "aligned/native native_speedup aligned_speedup\n";

    for (const auto &format : test::quantizedMoEVerifierFormats())
    {
        if (!all_formats && requested.count(format.label) == 0u)
            continue;

        const auto &down_format = selectedDownFormat(format);

        if (profile_iterations > 0)
        {
            // Do not even prepare the unselected numerical policy: its oracle
            // and setup kernels would belong to a different profiler launch.
            (void)runFormat(
                format, down_format, active_experts, local_route_rows, d_model, intermediate,
                profile_policy == ProfilePolicy::Aligned
                    ? CPUProjectionNumericalPolicy::GPUAlignedExpert
                    : CPUProjectionNumericalPolicy::BackendNative,
                verifier_schedule, warmup, samples, batch_iterations,
                profile_iterations, profile_phase);
            ++executed;
            continue;
        }

        PipelineTiming native;
        PipelineTiming aligned;
        auto measure_native = [&]
        {
            native = runFormat(
                format,
                down_format,
                active_experts,
                local_route_rows,
                d_model,
                intermediate,
                CPUProjectionNumericalPolicy::BackendNative,
                verifier_schedule,
                warmup,
                samples,
                batch_iterations,
                profile_policy == ProfilePolicy::Native
                    ? profile_iterations
                    : 0,
                profile_phase);
        };
        auto measure_aligned = [&]
        {
            aligned = runFormat(
                format,
                down_format,
                active_experts,
                local_route_rows,
                d_model,
                intermediate,
                CPUProjectionNumericalPolicy::GPUAlignedExpert,
                verifier_schedule,
                warmup,
                samples,
                batch_iterations,
                profile_policy == ProfilePolicy::Aligned
                    ? profile_iterations
                    : 0,
                profile_phase);
        };
        /* Alternate first policy to avoid systematic cache-temperature bias. */
        if ((executed & 1) != 0)
        {
            measure_aligned();
            measure_native();
        }
        else
        {
            measure_native();
            measure_aligned();
        }
        const double native_speedup = native.persistent_complete_us > 0.0
            ? native.serial_us / native.persistent_complete_us
            : 0.0;
        const double aligned_speedup = aligned.persistent_complete_us > 0.0
            ? aligned.serial_us / aligned.persistent_complete_us
            : 0.0;
        const double slowdown = aligned.persistent_complete_us /
            native.persistent_complete_us;
        std::cout
            << format.label << '/' << down_format.label << ' '
            << native.persistent_complete_us << ' '
            << aligned.persistent_complete_us << ' '
            << slowdown << ' '
            << native_speedup << ' '
            << aligned_speedup << '\n';
        if (csv)
        {
            csv << format.label << ',' << down_format.label << ','
                << isaLevelName(activeISALevel()) << ','
                << d_model << ',' << intermediate << ','
                << active_experts << ',' << local_route_rows << ','
                << omp_get_max_threads() << ','
                << native.gate_up_us << ',' << native.swiglu_q8_us << ','
                << native.down_us << ',' << native.complete_us << ','
                << native.persistent_complete_us << ','
                << aligned.gate_up_us << ',' << aligned.swiglu_q8_us << ','
                << aligned.down_us << ',' << aligned.complete_us << ','
                << aligned.persistent_complete_us << ',' << slowdown << '\n';
        }
        EXPECT_GT(native_speedup, 1.0)
            << format.label
            << " backend-native grouped execution must beat serial row replay";
        EXPECT_GT(aligned_speedup, 1.0)
            << format.label
            << " GPU-aligned grouped execution must beat serial row replay";
        ++executed;
    }
    ASSERT_GT(executed, 0)
        << "No canonical CPU MoE expert formats matched the selection";
}
