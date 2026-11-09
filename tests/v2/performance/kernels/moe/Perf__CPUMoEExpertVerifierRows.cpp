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
 * 1. one layer-batched NativeQuantised gate/up projection;
 * 2. fused byte-exact SwiGLU-to-Q8_1 publication; and
 * 3. one layer-batched NativeQuantised down projection.
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
 * `LLAMINAR_CPU_MOE_EXPERT_PURPOSE=decode|mtp|prefill` selects the
 * explicit scheduling purpose for grouped descriptors only (default Decode).
 * The independent one-row arithmetic oracle remains Decode.
 * Optional GATE_UP_POLICY/DOWN_POLICY independently name explicit existing
 * kernel policies; they inherit the old shared VERIFIER_POLICY by default.
 * GATE_UP_OWNERSHIP/DOWN_OWNERSHIP=Current|Contiguous independently select
 * diagnostic output-job ownership. Current preserves all original receipts;
 * Contiguous requires aligned Pairwise ordinary Prefill and records a distinct
 * ownership identity, never a new row policy or installed exact observation.
 * MEASURE_POLICY=both|native|aligned makes timing scope explicit (default both).
 * Missing numerical-domain columns are NaN, never invented costs. Clocked
 * warmups/samples reuse the same quiescent PerfStats suppression as exact batch;
 * fresh actual phase authentication and isolated profiler work remain outside.
 * Existing PROFILE_REQUEST_ID/PERF_STATS_PATH opt into one isolated gate/up or
 * down invocation on warmed retained workers. Preparation, byte checks, two
 * warmups and fresh actual route/ownership proof remain outside that interval.
 * Phase buffers use the production64B-aligned storage type in both ownership
 * controls, preserving original tight FP32/Q8 strides. Metadata and raw clocks
 * identify this layout revision; older std::vector clocks are not relabeled.
 * Geometry is authored without reconstructing a recorded model route vector.
 */

#include "../../../utils/NativeQuantisedTestPartialStorage.h"
#include "../../../utils/CPUPerfStatsTimingCollectionScope.h"
#include "../native_quantised_dispatch/NativeQuantisedProfilerControl.h"
#include <gtest/gtest.h>

#include "kernels/cpu/gemm/CPUNativeQuantisedGemmKernel.h"
#include "kernels/cpu/primitives/SwiGLUPrimitives.h"
#include "utils/CPUFeatures.h"
#include "tensors/AlignedVector.h"
#include "utils/QuantizedVerifierFormats.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <map>
#include <optional>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <omp.h>

using namespace llaminar2;
using llaminar2::cpu::native_quantised::CPUNativeQuantisedGemmKernel;
using llaminar2::cpu::native_quantised::VerifierRowsPolicy;
using llaminar2::cpu::native_quantised::FusedOutputJobOwnership;
using llaminar2::cpu::native_quantised::fusedOutputJobOwnershipName;
using llaminar2::cpu::native_quantised::ProjectionRowsPurpose;
using llaminar2::cpu::native_quantised::projectionRowsPurposeName;
using llaminar2::cpu::native_quantised::cpuNativeQuantisedSerialEquivalentPolicyN;
using llaminar2::cpu::native_quantised::gemv_native_quantised_preq;
using llaminar2::cpu::native_quantised::quantize_activations_to_q8_1;
using llaminar2::cpu::native_quantised::swiglu_quantize_activations_to_q8_1;

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
        std::unique_ptr<CPUNativeQuantisedGemmKernel> gate;
        std::unique_ptr<CPUNativeQuantisedGemmKernel> up;
        std::unique_ptr<CPUNativeQuantisedGemmKernel> down;
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

    /**
     * @brief Parse the explicit grouped scheduling intent without changing math.
     * @param purpose Public diagnostic token: decode, mtp, or prefill.
     * @return Existing typed purpose forwarded to every grouped projection.
     * @throws std::invalid_argument For every unknown or empty explicit token.
     */
    ProjectionRowsPurpose parseProjectionPurpose(std::string_view purpose)
    {
        if (purpose == "decode") return ProjectionRowsPurpose::Decode;
        if (purpose == "mtp") return ProjectionRowsPurpose::MTPVerifier;
        if (purpose == "prefill") return ProjectionRowsPurpose::Prefill;
        throw std::invalid_argument(
            "LLAMINAR_CPU_MOE_EXPERT_PURPOSE must be decode, mtp, or prefill");
    }

    /** @return Explicit purpose, retaining the owner's original Decode default. */
    ProjectionRowsPurpose selectedProjectionPurpose()
    {
        const char *raw = std::getenv("LLAMINAR_CPU_MOE_EXPERT_PURPOSE");
        return parseProjectionPurpose(raw ? raw : "decode");
    }

    /** @brief Parse a forceable grouped-row policy for exact perf comparison. */
    VerifierRowsPolicy parseVerifierPolicy(std::string_view policy)
    {
        if (policy == "Auto")
            return VerifierRowsPolicy::Auto;
        if (policy == "Pairwise")
            return VerifierRowsPolicy::Pairwise;
        if (policy == "WideRows")
            return VerifierRowsPolicy::WideRows;
        if (policy == "OrderedThreeRows")
            return VerifierRowsPolicy::OrderedThreeRows;
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

    /**
     * @brief Read one explicit role policy, retaining the old shared default.
     * @param name Exact environment field owned by this fixture.
     * @param fallback Original shared policy when the optional field is unset/empty.
     * @return Parsed physical policy; unknown names are rejected, never substituted.
     */
    VerifierRowsPolicy selectedVerifierPolicy(const char* name,
        VerifierRowsPolicy fallback=VerifierRowsPolicy::Auto)
    {
        const char* raw=std::getenv(name);
        if(!raw || !*raw)return fallback;
        try{return parseVerifierPolicy(raw);}
        catch(const std::invalid_argument& error){throw std::invalid_argument(std::string(name)+": "+error.what());}
    }

    /**
     * @brief Parse independent job ownership without selecting another row kernel.
     * @param value Explicit case-sensitive diagnostic token.
     * @return Original or contiguous independent job ownership.
     * @throws std::invalid_argument For an unknown or empty token.
     */
    FusedOutputJobOwnership parseJobOwnership(std::string_view value)
    {
        if (value == "Current") return FusedOutputJobOwnership::Current;
        if (value == "Contiguous") return FusedOutputJobOwnership::Contiguous;
        throw std::invalid_argument("CPU FFN ownership must be Current or Contiguous");
    }

    /**
     * @brief Read one phase's explicit diagnostic from the existing fixture controls.
     * @param name Exact fixture-owned environment key.
     * @return Explicit phase ownership; absence preserves the original schedule.
     * @throws std::invalid_argument For an unknown or empty explicit token.
     */
    FusedOutputJobOwnership selectedJobOwnership(const char *name)
    {
        const char *raw = std::getenv(name);
        return parseJobOwnership(raw ? raw : "Current");
    }

    /** @brief Explicit timing scope; no absent numerical policy becomes a zero cost. */
    enum class MeasurementPolicy { Both, Native, Aligned };
    /** @brief Parse the existing numerical domains without changing their arithmetic. */
    MeasurementPolicy parseMeasurementPolicy(std::string_view value)
    {
        if(value=="both")return MeasurementPolicy::Both;
        if(value=="native")return MeasurementPolicy::Native;
        if(value=="aligned")return MeasurementPolicy::Aligned;
        throw std::invalid_argument("LLAMINAR_CPU_MOE_EXPERT_MEASURE_POLICY must be both, native or aligned");
    }
    /** @brief Preserve original both-domain timing unless the caller explicitly narrows it. */
    MeasurementPolicy selectedMeasurementPolicy()
    {
        const char* raw=std::getenv("LLAMINAR_CPU_MOE_EXPERT_MEASURE_POLICY");
        return parseMeasurementPolicy(raw?raw:"both");
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

    /**
     * @brief Validate isolated profiler scope before any weights or output are created.
     * @param request Existing immutable LinuxPerfControl request ID.
     * @param output Existing per-worker counter output path, paired with request.
     * @param batch Existing batch key; unsupported by this single-phase owner.
     * @param calls Exactly one profiled call; legacy repeated profiling is separate.
     * @param phase Exactly gate_up or down, never a composed FFN transaction.
     * @param formats Number of selected source-format transactions.
     * @param all_formats Whether selection requests the broad canonical matrix.
     * @param policy The one concrete numerical domain this process executes.
     * @throws std::invalid_argument For incomplete or ambiguous isolated scope.
     */
    void requireIsolatedFFNProfileOptions(std::string_view request, std::string_view output,
        std::string_view batch, int calls, std::string_view phase, size_t formats,
        bool all_formats, ProfilePolicy policy)
    {
        if (!batch.empty()) throw std::invalid_argument("FFN isolated profiling does not support a profile batch");
        if (request.empty() && output.empty()) return; // Preserve all legacy timing/profile options.
        if (request.empty() || output.empty() || calls != 1 || formats != 1 || all_formats ||
            (phase != "gate_up" && phase != "down") ||
            (policy != ProfilePolicy::Native && policy != ProfilePolicy::Aligned))
            throw std::invalid_argument("FFN isolated profiling requires paired request/output, one format, numerical domain, gate_up/down phase and exactly one call");
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

    /**
     * @brief Validate the original contiguous FP32 row layout without reading payloads.
     * @param base Actual output/expert-slice address from the owning aligned buffer.
     * @param ldc Original physical row stride; never padded to manufacture alignment.
     * @return True only for a non-null64B base and positive64B row stride.
     */
    bool alignedFFNFloatRows(const float* base,int ldc)
    {
        return base && ldc>0 && reinterpret_cast<std::uintptr_t>(base)%AlignedVector<float>::ALIGNMENT==0 &&
            (static_cast<size_t>(ldc)*sizeof(float))%AlignedVector<float>::ALIGNMENT==0;
    }

    /** @brief Produce deterministic, non-trivial router-published hidden rows. */
    AlignedVector<float> makeHiddenRows(int rows, int d_model)
    {
        AlignedVector<float> hidden(
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
        const AlignedVector<T> &actual,
        const AlignedVector<T> &expected,
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
     * @param function Original complete phase invocation, including its normal team lifetime.
     * @param warmup Unmeasured invocation batches before the sample clocks.
     * @param samples Number of independent native clock intervals.
     * @param batch_iterations Calls per clock interval; normalization occurs once.
     * @param raw_samples Optional chronological native doubles, copied after clocks.
     * @return Original upper median in microseconds per invocation.
     */
    template <typename Function>
    double medianMicros(
        Function &&function,
        int warmup,
        int samples,
        int batch_iterations,
        std::vector<double>* raw_samples=nullptr)
    {
        std::vector<double> measurements;
        measurements.reserve(static_cast<size_t>(samples));
        {
        test::CPUExactTimingCollectionScope recording_pause;
        for (int iteration = 0; iteration < warmup; ++iteration)
        {
            for (int batch = 0; batch < batch_iterations; ++batch)
                function();
        }
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
        } // All worker teams have joined before restoring recording.
        // Retain chronological native doubles only after every clock and team
        // has retired. Sorting below preserves the original median definition.
        if(raw_samples)*raw_samples=measurements;
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

            expert.gate = std::make_unique<CPUNativeQuantisedGemmKernel>(
                expert.gate_weights.get(), 0, -1, numerical_policy);
            expert.up = std::make_unique<CPUNativeQuantisedGemmKernel>(
                expert.up_weights.get(), 0, -1, numerical_policy);
            expert.down = std::make_unique<CPUNativeQuantisedGemmKernel>(
                expert.down_weights.get(), 0, -1, numerical_policy);
            if (!expert.gate->isValid() || !expert.up->isValid() ||
                !expert.down->isValid())
            {
                throw std::runtime_error(
                    std::string("NativeQuantised preparation failed for ") +
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
     * @param gate_up_schedule Explicit gate/up candidate, or original Auto.
     * @param down_schedule Explicit down candidate, or original Auto.
     * @param gate_up_ownership Independent ordinary-prefill gate/up ownership diagnostic.
     * @param down_ownership Independent ordinary-prefill down ownership diagnostic.
     * @param purpose Typed grouped scheduling intent; serial oracles remain Decode.
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
        VerifierRowsPolicy gate_up_schedule,
        VerifierRowsPolicy down_schedule,
        FusedOutputJobOwnership gate_up_ownership,
        FusedOutputJobOwnership down_ownership,
        ProjectionRowsPurpose purpose,
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
        const AlignedVector<float> hidden =
            makeHiddenRows(local_route_rows, d_model);
        AlignedVector<Q8_1Block> router_q8(
            static_cast<size_t>(local_route_rows) * hidden_blocks);
        quantize_activations_to_q8_1(
            hidden.data(),
            router_q8.data(),
            local_route_rows,
            d_model,
            hidden_blocks,
            numerical_policy);

        AlignedVector<float> grouped_gate(
            static_cast<size_t>(local_route_rows) * intermediate);
        AlignedVector<float> grouped_up(grouped_gate.size());
        AlignedVector<Q8_1Block> grouped_activation(
            static_cast<size_t>(local_route_rows) * activation_blocks);
        AlignedVector<float> grouped_down(
            static_cast<size_t>(local_route_rows) * d_model);

        AlignedVector<float> serial_gate(grouped_gate.size());
        AlignedVector<float> serial_up(grouped_up.size());
        AlignedVector<float> serial_activated(grouped_gate.size());
        AlignedVector<Q8_1Block> serial_activation(grouped_activation.size());
        AlignedVector<float> serial_down(grouped_down.size());

        using Descriptor =
            CPUNativeQuantisedGemmKernel::BatchedPrequantizedProjectionDesc;
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
                .verifier_schedule = gate_up_schedule,
                .purpose = purpose,
            });
            gate_up_descriptors.push_back({
                .kernel = expert.up.get(),
                .input_q8 = expert_input,
                .output = grouped_up.data() + row * intermediate,
                .bias = nullptr,
                .rows = expert.rows,
                .n = intermediate,
                .ldc = intermediate,
                .verifier_schedule = gate_up_schedule,
                .purpose = purpose,
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
                .verifier_schedule = down_schedule,
                .purpose = purpose,
            });
        }

        // The production phase roots are64B aligned. Keep exact original
        // tight strides: Q8 block rows need not themselves be cache-line aligned.
        for(const auto& descriptor:gate_up_descriptors)
            if(!alignedFFNFloatRows(descriptor.output,descriptor.ldc))
                throw std::runtime_error("FFN gate/up descriptor lost its actual aligned root/row stride");
        for(const auto& descriptor:down_descriptors)
            if(!alignedFFNFloatRows(descriptor.output,descriptor.ldc))
                throw std::runtime_error("FFN down descriptor lost its actual aligned root/row stride");
        const auto publish_layout=[&]() {
            const auto emit=[&]<class T>(const char* role,const AlignedVector<T>& values,size_t row_elements) {
                const auto base=reinterpret_cast<std::uintptr_t>(values.data());
                if(base%AlignedVector<T>::ALIGNMENT)
                    throw std::runtime_error("FFN phase root lost its64B storage contract");
                std::cout<<"CPU_FFN_LAYOUT,revision=aligned_phase_roots_v1,format="<<format.label
                    <<",down_format="<<down_format.label
                    <<",numerical_policy="<<(numerical_policy==CPUProjectionNumericalPolicy::GPUAlignedExpert?"GPUAlignedExpert":"BackendNative")
                    <<",purpose="<<projectionRowsPurposeName(purpose)<<",buffer_role="<<role
                    <<",base_mod64="<<base%64u<<",live_elements="<<values.size()
                    <<",live_bytes="<<values.size()*sizeof(T)<<",row_elements="<<row_elements
                    <<",row_stride_bytes="<<row_elements*sizeof(T)<<",expert_start_mod64=";
                for(size_t expert=0;expert<experts.size();++expert) {
                    if(expert)std::cout<<':';
                    std::cout<<reinterpret_cast<std::uintptr_t>(values.data()+
                        static_cast<size_t>(experts[expert].first_row)*row_elements)%64u;
                }
                std::cout<<",outside_timing=1,outside_profile=1\n";
            };
            emit("hidden_fp32",hidden,static_cast<size_t>(d_model));
            emit("router_q8",router_q8,static_cast<size_t>(hidden_blocks));
            emit("gate_fp32",grouped_gate,static_cast<size_t>(intermediate));
            emit("up_fp32",grouped_up,static_cast<size_t>(intermediate));
            emit("activation_q8",grouped_activation,static_cast<size_t>(activation_blocks));
            emit("down_fp32",grouped_down,static_cast<size_t>(d_model));
            emit("serial_oracle_gate_fp32",serial_gate,static_cast<size_t>(intermediate));
            emit("serial_oracle_up_fp32",serial_up,static_cast<size_t>(intermediate));
            emit("serial_oracle_activated_fp32",serial_activated,static_cast<size_t>(intermediate));
            emit("serial_oracle_q8",serial_activation,static_cast<size_t>(activation_blocks));
            emit("serial_oracle_down_fp32",serial_down,static_cast<size_t>(d_model));
        };

        using PartialStorage = llaminar2::test::NativeQuantisedTestPartialStorage;
        PartialStorage grouped_partials(std::max(
            PartialStorage::bundleFloats(gate_up_descriptors.data(), gate_up_descriptors.size(), 1),
            PartialStorage::bundleFloats(down_descriptors.data(), down_descriptors.size(), 1)));
        auto grouped_gate_up = [&]()
        {
            if (!CPUNativeQuantisedGemmKernel::
                    multiply_batched_preq_decode_equivalent(
                        gate_up_descriptors.data(),
                        static_cast<int>(gate_up_descriptors.size()),
                        d_model, grouped_partials.span(), gate_up_ownership))
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
            if (!CPUNativeQuantisedGemmKernel::
                    multiply_batched_preq_decode_equivalent(
                        down_descriptors.data(),
                        static_cast<int>(down_descriptors.size()),
                        intermediate, grouped_partials.span(), down_ownership))
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
            if (!CPUNativeQuantisedGemmKernel::
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
                        static_cast<int>(down_descriptors.size()), grouped_partials.span(),
                        gate_up_ownership, down_ownership))
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
                    cpu::native_quantised::nativeQuantisedProjectionPartialFloats(kernel->packedWeights(), 1));
        test::NativeQuantisedTestPartialStorage partial_storage(partial_floats);
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

                    gemv_native_quantised_preq(
                        expert.gate->packedWeights(), hidden_row, gate_row, partial_storage.span());
                    gemv_native_quantised_preq(
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
                    gemv_native_quantised_preq(
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

        const auto exact_profile_request = test::native_quantised_dispatch::profilerRequestId();
        if (!exact_profile_request.empty())
        {
            if (::testing::Test::HasFailure())
                throw std::runtime_error("FFN isolated profiling rejected a failed preparation byte oracle");
            // Match CPUExecutionExactBatch's retained-team protocol. This call
            // must originate outside an active team: perf control gathers the
            // real persistent TIDs and concurrently arms/disarms those workers.
            if (omp_in_parallel()) throw std::invalid_argument("FFN isolated profiling must enter from outside an active OpenMP team");
            const auto selected_phase = [&]() {
                if (profile_phase == "gate_up") grouped_gate_up();
                else grouped_down_projection();
            };
            const auto launch = [&]() {
#pragma omp parallel
                { selected_phase(); }
            };
            for (int warmup = 0; warmup < 2; ++warmup) launch();
            constexpr const char *route_family="kernel.cpu_native_quantised_fused_verifier_rows_projection_launch";
            constexpr const char *ownership_family="kernel.cpu_native_quantised_fused_job_ownership";
            const auto before_routes=PerfStatsCollector::snapshot({route_family});
            const auto before_ownership=PerfStatsCollector::snapshot({ownership_family});
            launch(); // Fresh actual proof is a separate, unprofiled call.
            const auto fresh_count=[](const auto &records,const auto &before,auto emit) {
                uint64_t count=0;
                for(const auto &record:records) {
                    const auto prior=std::find_if(before.begin(),before.end(),[&](const auto &entry){return entry.tags==record.tags;});
                    const uint64_t old=prior==before.end()?0:prior->count;
                    if(record.count<old)throw std::runtime_error("FFN profile proof epoch changed");
                    const auto fresh=record.count-old;
                    if(fresh){count+=fresh;emit(record.tags,fresh);}
                }
                return count;
            };
            const auto emit=[&](const char *kind,const auto &tags,uint64_t count) {
                std::cout<<"CPU_FFN_PROFILE_PROOF,request="<<exact_profile_request<<",phase="<<profile_phase
                    <<",kind="<<kind<<",fresh_count="<<count;
                for(const auto &[name,value]:tags)std::cout<<','<<name<<'='<<value;
                std::cout<<",outside_profile=1\n";
            };
            const auto expected_descriptors=static_cast<uint64_t>(active_experts)*(profile_phase=="gate_up"?2u:1u);
            const auto routes=fresh_count(PerfStatsCollector::snapshot({route_family}),before_routes,
                [&](const auto &tags,uint64_t count){emit("route",tags,count);});
            const auto ownership=fresh_count(PerfStatsCollector::snapshot({ownership_family}),before_ownership,
                [&](const auto &tags,uint64_t count){emit("ownership",tags,count);});
            const auto requested_ownership=profile_phase=="gate_up"?gate_up_ownership:down_ownership;
            if(routes!=expected_descriptors || ownership!=uint64_t(requested_ownership==FusedOutputJobOwnership::Contiguous))
                throw std::runtime_error("FFN isolated profiling requires fresh actual selected-phase route/ownership proof");
            const auto output=test::native_quantised_dispatch::profilerEnvironment(
                test::native_quantised_dispatch::kPerfStatsPathEnvironment);
            test::native_quantised_dispatch::LinuxPerfControl control(exact_profile_request,output);
            test::native_quantised_dispatch::profileExactOpenMPRegion(control,exact_profile_request,output,launch);
            // Counter read/publication and these byte oracles occur AFTER disarm.
            expectByteEqual(grouped_gate,serial_gate,std::string(format.label)+" isolated profile gate");
            expectByteEqual(grouped_up,serial_up,std::string(format.label)+" isolated profile up");
            expectByteEqual(grouped_activation,serial_activation,std::string(format.label)+" isolated profile Q8");
            expectByteEqual(grouped_down,serial_down,std::string(format.label)+" isolated profile down");
            if (::testing::Test::HasFailure())
                throw std::runtime_error("FFN isolated profile failed its post-launch byte oracle");
            std::cout<<"CPU_FFN_PROFILE,request="<<exact_profile_request<<",output="<<output
                <<",format="<<format.label<<",down_format="<<down_format.label<<",phase="<<profile_phase
                <<",numerical_policy="<<(numerical_policy==CPUProjectionNumericalPolicy::GPUAlignedExpert?"GPUAlignedExpert":"BackendNative")
                <<",purpose="<<projectionRowsPurposeName(purpose)<<",runtime_isa="<<isaLevelName(activeISALevel())
                <<",workers="<<omp_get_max_threads()<<",team_lifetime=retained,profiled_calls=1,preconditioning_calls=2,proof_calls=1"
                <<",d_model="<<d_model<<",intermediate="<<intermediate<<",active_experts="<<active_experts<<",route_rows="<<local_route_rows
                <<",requested_gate_up_policy="<<cpu::native_quantised::verifierRowsPolicyName(gate_up_schedule)
                <<",requested_down_policy="<<cpu::native_quantised::verifierRowsPolicyName(down_schedule)
                <<",requested_gate_up_ownership="<<fusedOutputJobOwnershipName(gate_up_ownership)
                <<",requested_down_ownership="<<fusedOutputJobOwnershipName(down_ownership)<<",bytes_pass=1\n";
            publish_layout();
            return {}; // Never enter legacy repeat loops or canonical timing clocks.
        }

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
                << " purpose=" << projectionRowsPurposeName(purpose)
                << " gate_up_policy_n="
                << cpuNativeQuantisedSerialEquivalentPolicyN(intermediate)
                << " down_policy_n="
                << cpuNativeQuantisedSerialEquivalentPolicyN(d_model)
                << " gate_up_policy=" << cpu::native_quantised::verifierRowsPolicyName(gate_up_schedule)
                << " down_policy=" << cpu::native_quantised::verifierRowsPolicyName(down_schedule);
            if (gate_up_ownership != FusedOutputJobOwnership::Current || down_ownership != FusedOutputJobOwnership::Current)
                std::cerr << " gate_up_ownership=" << fusedOutputJobOwnershipName(gate_up_ownership)
                    << " down_ownership=" << fusedOutputJobOwnershipName(down_ownership);
            std::cerr << " iterations=" << profile_iterations
                << " checksum=" << checksum << '\n';
            // Profiling and paired timing are disjoint executions. Returning
            // here also prevents later phases from polluting the hot samples.
            publish_layout();
            return {};
        }

        PipelineTiming timing;
        std::array<std::vector<double>,6> raw_samples;
        timing.gate_up_us =
            medianMicros(grouped_gate_up, warmup, samples, batch_iterations,&raw_samples[0]);
        timing.swiglu_q8_us =
            medianMicros(grouped_swiglu_q8, warmup, samples, batch_iterations,&raw_samples[1]);
        timing.down_us =
            medianMicros(
                grouped_down_projection, warmup, samples, batch_iterations,&raw_samples[2]);
        timing.complete_us =
            medianMicros(grouped_complete, warmup, samples, batch_iterations,&raw_samples[3]);
        timing.persistent_complete_us =
            medianMicros(
                grouped_complete_persistent, warmup, samples, batch_iterations,&raw_samples[4]);
        timing.serial_us =
            medianMicros(
                serial_complete, /*warmup=*/1, samples, batch_iterations,&raw_samples[5]);
        // Snapshot only outside clocks. One fresh complete call then proves
        // actual selected kinds under the REAL existing FFN team entry, not
        // a metadata-only standalone guess of an installed exact key.
        constexpr const char* route_counter="kernel.cpu_native_quantised_fused_verifier_rows_projection_launch";
        std::map<PerfStatsCollector::Tags,uint64_t> prior;
        for(const auto& record:PerfStatsCollector::snapshot({route_counter}))prior[record.tags]=record.count;
        constexpr const char *ownership_counter="kernel.cpu_native_quantised_fused_job_ownership";
        std::map<PerfStatsCollector::Tags,uint64_t> prior_ownership;
        const bool ownership_diagnostic = gate_up_ownership != FusedOutputJobOwnership::Current ||
            down_ownership != FusedOutputJobOwnership::Current;
        if (ownership_diagnostic)
            for (const auto &record : PerfStatsCollector::snapshot({ownership_counter}))
                prior_ownership[record.tags] = record.count;
        grouped_complete_persistent();
        if (ownership_diagnostic) {
            uint64_t fresh_ownership = 0;
            for (const auto &record : PerfStatsCollector::snapshot({ownership_counter})) {
                const auto found = prior_ownership.find(record.tags);
                const uint64_t before = found == prior_ownership.end() ? 0 : found->second;
                if (record.count < before) throw std::runtime_error("FFN ownership epoch changed during fresh proof");
                const auto count = record.count - before;
                if (!count) continue;
                fresh_ownership += count;
                std::cout << "CPU_FFN_OWNERSHIP,format=" << format.label << ",down_format=" << down_format.label
                    << ",fresh_bundles=" << count;
                for (const auto &[name,value] : record.tags) std::cout << ',' << name << '=' << value;
                std::cout << ",outside_timing=1\n";
            }
            const auto expected = uint64_t(gate_up_ownership == FusedOutputJobOwnership::Contiguous) +
                uint64_t(down_ownership == FusedOutputJobOwnership::Contiguous);
            if (fresh_ownership != expected)
                throw std::runtime_error("Explicit FFN ownership requires its fresh outside-timing ownership PerfStats proof");
        }
        uint64_t fresh_descriptors=0;bool saw_ordered_three=false;
        for(const auto& record:PerfStatsCollector::snapshot({route_counter})) {
            const auto found=prior.find(record.tags);
            const auto before=found==prior.end()?uint64_t{0}:found->second;
            if(record.count<before)throw std::runtime_error("FFN route epoch changed during fresh authentication");
            const auto count=record.count-before;
            if(!count)continue;
            fresh_descriptors+=count;
            saw_ordered_three=saw_ordered_three || record.tags.at("effective_verifier_policy")=="OrderedThreeRows";
            std::cout<<"CPU_FFN_RESOLVED,format="<<format.label<<",down_format="<<down_format.label
                <<",numerical_policy="<<(numerical_policy==CPUProjectionNumericalPolicy::GPUAlignedExpert?"GPUAlignedExpert":"BackendNative")
                <<",fresh_descriptors="<<count;
            for(const auto name:{"m","n","policy_n","k","codebook","purpose","isa","k_tiles",
                 "requested_verifier_policy","effective_verifier_policy","physical_row_tile","route","task_grid"})
                std::cout<<','<<name<<'='<<record.tags.at(name);
            std::cout<<",outside_timing=1\n";
        }
        if(gate_up_schedule==VerifierRowsPolicy::OrderedThreeRows || down_schedule==VerifierRowsPolicy::OrderedThreeRows) {
            if(!saw_ordered_three || fresh_descriptors!=3u*static_cast<uint64_t>(active_experts))
                throw std::runtime_error("Explicit OrderedThreeRows FFN diagnostic requires the exact projection PerfStats filter for fresh outside-timing route proof");
        }
        expectByteEqual(grouped_gate,serial_gate,std::string(format.label)+" post-timing gate");
        expectByteEqual(grouped_up,serial_up,std::string(format.label)+" post-timing up");
        expectByteEqual(grouped_activation,serial_activation,std::string(format.label)+" post-timing activation Q8");
        expectByteEqual(grouped_down,serial_down,std::string(format.label)+" post-timing down");
        // Publish only after ALL six phase clocks and fresh authentication.
        // These are chronological per-bundle microseconds, divided by repeat
        // count exactly once inside medianMicros; profiler mode never reaches here.
        constexpr std::array<const char*,6> phases{
            "gate_up","swiglu_q8","down","complete","persistent_complete","serial_oracle"};
        const auto original_precision=std::cout.precision();
        std::cout<<std::setprecision(std::numeric_limits<double>::max_digits10);
        for(size_t phase=0;phase<raw_samples.size();++phase)
            for(size_t sample=0;sample<raw_samples[phase].size();++sample) {
                std::cout<<"CPU_FFN_SAMPLE,format="<<format.label<<",down_format="<<down_format.label
                    <<",numerical_policy="<<(numerical_policy==CPUProjectionNumericalPolicy::GPUAlignedExpert?"GPUAlignedExpert":"BackendNative")
                    <<",purpose="<<projectionRowsPurposeName(phase==5?ProjectionRowsPurpose::Decode:purpose)
                    <<",grouped_purpose="<<projectionRowsPurposeName(purpose)
                    <<",arithmetic_scope="<<(phase==5?"independent_serial_decode_oracle":"production_grouped")
                    <<",runtime_isa="<<isaLevelName(activeISALevel())<<",threads="<<omp_get_max_threads()
                    <<",candidate=gate_up:"<<cpu::native_quantised::verifierRowsPolicyName(gate_up_schedule)
                    <<"|down:"<<cpu::native_quantised::verifierRowsPolicyName(down_schedule)
                    <<",gate_up_policy_n="<<cpuNativeQuantisedSerialEquivalentPolicyN(intermediate)
                    <<",down_policy_n="<<cpuNativeQuantisedSerialEquivalentPolicyN(d_model)
                    <<",d_model="<<d_model<<",intermediate="<<intermediate
                    <<",active_experts="<<active_experts<<",route_rows="<<local_route_rows
                    <<",ffn_output_layout=aligned_phase_roots_v1"
                    <<",phase="<<phases[phase]<<",warmups="<<(phase==5?1:warmup)
                    <<",repeats="<<batch_iterations<<",sample="<<sample
                    <<",us="<<raw_samples[phase][sample];
                if (ownership_diagnostic)
                    std::cout << ",requested_gate_up_ownership=" << fusedOutputJobOwnershipName(gate_up_ownership)
                        << ",requested_down_ownership=" << fusedOutputJobOwnershipName(down_ownership)
                        << ",ownership_scope=" << (phase == 5 ? "serial_oracle_unchanged" :
                            phase == 1 ? "activation_unchanged" : "production_grouped");
                std::cout << ",outside_timing=1\n";
            }
        std::cout.precision(original_precision);
        publish_layout();
        return timing;
    }
} // namespace

/** @brief Validate authored purpose tokens without preparing weights or timing. */
TEST(Perf_CPUMoEExpertVerifierRows, ExplicitPurposeSelectionContract)
{
    EXPECT_EQ(parseProjectionPurpose("decode"), ProjectionRowsPurpose::Decode);
    EXPECT_EQ(parseProjectionPurpose("mtp"), ProjectionRowsPurpose::MTPVerifier);
    EXPECT_EQ(parseProjectionPurpose("prefill"), ProjectionRowsPurpose::Prefill);
    EXPECT_STREQ(projectionRowsPurposeName(parseProjectionPurpose("mtp")),
                 "mtp_verifier");
    EXPECT_THROW((void)parseProjectionPurpose("unknown"), std::invalid_argument);
    EXPECT_THROW((void)parseProjectionPurpose(""), std::invalid_argument);
    EXPECT_THROW((void)parseProjectionPurpose("Decode"), std::invalid_argument);
}

/** @test Authored policy controls remain explicit; no weights, GPU or timing work. */
TEST(Perf_CPUMoEExpertVerifierRows, ExplicitRoleAndNumericalPolicyContract)
{
    EXPECT_EQ(parseVerifierPolicy("Auto"),VerifierRowsPolicy::Auto);
    EXPECT_EQ(parseVerifierPolicy("Pairwise"),VerifierRowsPolicy::Pairwise);
    EXPECT_EQ(parseVerifierPolicy("WideRows"),VerifierRowsPolicy::WideRows);
    EXPECT_EQ(parseVerifierPolicy("OrderedThreeRows"),VerifierRowsPolicy::OrderedThreeRows);
    EXPECT_THROW((void)parseVerifierPolicy("unknown"),std::invalid_argument);
    EXPECT_THROW((void)parseVerifierPolicy(""),std::invalid_argument);
    EXPECT_EQ(parseJobOwnership("Current"),FusedOutputJobOwnership::Current);
    EXPECT_EQ(parseJobOwnership("Contiguous"),FusedOutputJobOwnership::Contiguous);
    EXPECT_THROW((void)parseJobOwnership(""),std::invalid_argument);
    EXPECT_THROW((void)parseJobOwnership("Auto"),std::invalid_argument);
    EXPECT_THROW((void)parseJobOwnership("contiguous"),std::invalid_argument);
    // Pure validation: no weights, OpenMP team, perf_event_open or filesystem access.
    for(const auto policy:{ProfilePolicy::Native,ProfilePolicy::Aligned}) {
        for(const auto *phase:{"gate_up","down"})
            EXPECT_NO_THROW(requireIsolatedFFNProfileOptions("request","output","",1,phase,1,false,policy));
        EXPECT_NO_THROW(requireIsolatedFFNProfileOptions("","","",2000,"persistent",3,true,policy));
    }
    const auto invalid_profile=[](std::string_view request,std::string_view output,std::string_view batch,
        int calls,std::string_view phase,size_t formats,bool all_formats,ProfilePolicy policy) {
        EXPECT_THROW(requireIsolatedFFNProfileOptions(request,output,batch,calls,phase,formats,all_formats,policy),std::invalid_argument);
    };
    invalid_profile("request","","",1,"down",1,false,ProfilePolicy::Aligned);
    invalid_profile("","output","",1,"down",1,false,ProfilePolicy::Aligned);
    invalid_profile("request","output","batch",1,"down",1,false,ProfilePolicy::Aligned);
    invalid_profile("","","batch",0,"down",1,false,ProfilePolicy::Aligned);
    for(int calls:{0,2,2000})invalid_profile("request","output","",calls,"down",1,false,ProfilePolicy::Aligned);
    for(const auto *phase:{"","swiglu_q8","complete","persistent"})
        invalid_profile("request","output","",1,phase,1,false,ProfilePolicy::Aligned);
    for(size_t formats:{size_t{0},size_t{2}})invalid_profile("request","output","",1,"down",formats,false,ProfilePolicy::Aligned);
    invalid_profile("request","output","",1,"down",1,true,ProfilePolicy::Aligned);
    invalid_profile("request","output","",1,"down",1,false,static_cast<ProfilePolicy>(99));
    EXPECT_EQ(parseMeasurementPolicy("both"),MeasurementPolicy::Both);
    EXPECT_EQ(parseMeasurementPolicy("native"),MeasurementPolicy::Native);
    EXPECT_EQ(parseMeasurementPolicy("aligned"),MeasurementPolicy::Aligned);
    EXPECT_THROW((void)parseMeasurementPolicy(""),std::invalid_argument);
    EXPECT_THROW((void)parseMeasurementPolicy("unknown"),std::invalid_argument);
}

/** @test Aligned phase roots retain tight FP32/Q8 extents and reject shifted output slices. */
TEST(Perf_CPUMoEExpertVerifierRows, OutputLayoutContract)
{
    AlignedVector<float> fp32(3u*320u);
    AlignedVector<Q8_1Block> q8(3u*10u);
    EXPECT_TRUE(alignedFFNFloatRows(fp32.data(),320));
    EXPECT_TRUE(alignedFFNFloatRows(fp32.data()+320,320));
    EXPECT_FALSE(alignedFFNFloatRows(fp32.data()+1,320));
    EXPECT_FALSE(alignedFFNFloatRows(fp32.data(),321));
    EXPECT_FALSE(alignedFFNFloatRows(nullptr,320));
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(q8.data())%64u,0u);
    EXPECT_EQ(q8.size(),30u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(q8.data()+10)%64u,(10u*sizeof(Q8_1Block))%64u);
}

/** @test CPU phase warmup/sample calls emit zero records; fresh proof stays outside clocks. */
TEST(Perf_CPUMoEExpertVerifierRows, TimingCollectionScopeContract)
{
    struct Restore {
        std::optional<std::string> filter,summary;
        Restore(){
            if(const char* value=std::getenv("LLAMINAR_PERF_STATS_FILTER"))filter=value;
            if(const char* value=std::getenv("LLAMINAR_PERF_STATS_SUMMARY"))summary=value;
            if(setenv("LLAMINAR_PERF_STATS_FILTER","kernel.ffn_timing_collection_contract",1) ||
               setenv("LLAMINAR_PERF_STATS_SUMMARY","1",1))throw std::runtime_error("Cannot bind timing contract");
            PerfStatsCollector::reloadConfigurationFromEnvironment();
        }
        ~Restore(){
            if(filter)setenv("LLAMINAR_PERF_STATS_FILTER",filter->c_str(),1);else unsetenv("LLAMINAR_PERF_STATS_FILTER");
            if(summary)setenv("LLAMINAR_PERF_STATS_SUMMARY",summary->c_str(),1);else unsetenv("LLAMINAR_PERF_STATS_SUMMARY");
            PerfStatsCollector::reloadConfigurationFromEnvironment();
        }
    } restore;
    const auto observations=[](){uint64_t count=0;for(const auto& record:PerfStatsCollector::snapshot(
        {"kernel.ffn_timing_collection_contract"}))count+=record.count;return count;};
    const auto call=[](){PerfStatsCollector::addCounter("kernel","ffn_timing_collection_contract",1.0,"gemm","cpu");};
    const auto before=observations();
    std::vector<double> raw_samples;
    const auto median=medianMicros(call,2,3,2,&raw_samples);
    ASSERT_EQ(raw_samples.size(),3u);
    EXPECT_TRUE(std::all_of(raw_samples.begin(),raw_samples.end(),
        [](double value){return std::isfinite(value) && value>=0.0;}));
    auto sorted=raw_samples;
    std::sort(sorted.begin(),sorted.end());
    EXPECT_EQ(median,sorted[sorted.size()/2u]);
    EXPECT_EQ(observations(),before);
    call();
    EXPECT_EQ(observations(),before+1u);
}

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
    const VerifierRowsPolicy verifier_schedule = selectedVerifierPolicy("LLAMINAR_CPU_MOE_EXPERT_VERIFIER_POLICY");
    const VerifierRowsPolicy gate_up_schedule=selectedVerifierPolicy("LLAMINAR_CPU_MOE_EXPERT_GATE_UP_POLICY",verifier_schedule);
    const VerifierRowsPolicy down_schedule=selectedVerifierPolicy("LLAMINAR_CPU_MOE_EXPERT_DOWN_POLICY",verifier_schedule);
    const auto gate_up_ownership=selectedJobOwnership("LLAMINAR_CPU_MOE_EXPERT_GATE_UP_OWNERSHIP");
    const auto down_ownership=selectedJobOwnership("LLAMINAR_CPU_MOE_EXPERT_DOWN_OWNERSHIP");
    const bool ownership_diagnostic=gate_up_ownership!=FusedOutputJobOwnership::Current ||
        down_ownership!=FusedOutputJobOwnership::Current;
    const auto measurement_policy=selectedMeasurementPolicy();
    const bool include_native=measurement_policy!=MeasurementPolicy::Aligned;
    const bool include_aligned=measurement_policy!=MeasurementPolicy::Native;
    const ProjectionRowsPurpose purpose = selectedProjectionPurpose();
    const bool ordered_three=gate_up_schedule==VerifierRowsPolicy::OrderedThreeRows ||
        down_schedule==VerifierRowsPolicy::OrderedThreeRows;
    if(ordered_three || ownership_diagnostic) {
        ASSERT_EQ(purpose,ProjectionRowsPurpose::Prefill);
        if(profile_iterations>0)ASSERT_EQ(profile_policy,ProfilePolicy::Aligned);
        else ASSERT_EQ(measurement_policy,MeasurementPolicy::Aligned)
            << "Explicit Prefill diagnostics require aligned-only clocks, never a BackendNative substitution";
    }
    // Batched descriptors have no original-output-slice binding. Query the
    // same serial-width authority used by the production fused planner; do
    // not substitute an imagined TP parent width for these physical matrices.
    const int gate_up_policy_n =
        cpuNativeQuantisedSerialEquivalentPolicyN(intermediate);
    const int down_policy_n =
        cpuNativeQuantisedSerialEquivalentPolicyN(d_model);
    ASSERT_LE(active_experts, 256);
    ASSERT_GE(local_route_rows, active_experts);
    ASSERT_EQ(d_model % static_cast<int>(Q8_1Block::BLOCK_SIZE), 0)
        << "d_model must contain complete Q8_1 blocks";
    ASSERT_EQ(intermediate % static_cast<int>(Q8_1Block::BLOCK_SIZE), 0)
        << "intermediate must contain complete Q8_1 blocks";

    const std::set<std::string> requested = selectedFormats();
    const bool all_formats = requested.count("all") != 0u;
    const auto exact_profile_request=test::native_quantised_dispatch::profilerRequestId();
    const auto exact_profile_output=test::native_quantised_dispatch::profilerEnvironment(
        test::native_quantised_dispatch::kPerfStatsPathEnvironment);
    requireIsolatedFFNProfileOptions(exact_profile_request,exact_profile_output,
        test::native_quantised_dispatch::profilerEnvironment(test::native_quantised_dispatch::kProfilerBatchPathEnvironment),
        profile_iterations,profile_phase,requested.size(),all_formats,profile_policy);
    if (!exact_profile_request.empty() && (std::ifstream(exact_profile_output).good() ||
        std::ifstream(exact_profile_output+".partial").good()))
        throw std::invalid_argument("FFN isolated profiler output already exists");
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
               "aligned_over_native_persistent,purpose,gate_up_policy_n,down_policy_n,requested_gate_up_policy,requested_down_policy,measured_numerics";
        if (ownership_diagnostic) csv << ",requested_gate_up_ownership,requested_down_ownership";
        csv << ",ffn_output_layout\n";
    }
    int executed = 0;
    std::cout
        << "\nCPU MoE expert numerical-policy speedometer"
        << " (threads=" << omp_get_max_threads()
        << ", d_model=" << d_model
        << ", intermediate=" << intermediate
        << ", active_experts=" << active_experts
        << ", local_route_rows=" << local_route_rows
        << ", purpose=" << projectionRowsPurposeName(purpose)
        << ", gate_up_policy_n=" << gate_up_policy_n
        << ", down_policy_n=" << down_policy_n
        << ", verifier_policy="
        << cpu::native_quantised::verifierRowsPolicyName(verifier_schedule)
        << ", gate_up_policy=" << cpu::native_quantised::verifierRowsPolicyName(gate_up_schedule)
        << ", down_policy=" << cpu::native_quantised::verifierRowsPolicyName(down_schedule)
        << ", measured_numerics=" << (include_native?(include_aligned?"both":"native"):"aligned")
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
                gate_up_schedule, down_schedule, gate_up_ownership, down_ownership, purpose, warmup, samples, batch_iterations,
                profile_iterations, profile_phase);
            ++executed;
            continue;
        }

        const double missing=std::numeric_limits<double>::quiet_NaN();
        PipelineTiming native{missing,missing,missing,missing,missing,missing};
        PipelineTiming aligned=native;
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
                gate_up_schedule, down_schedule, gate_up_ownership, down_ownership,
                purpose,
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
                gate_up_schedule, down_schedule, gate_up_ownership, down_ownership,
                purpose,
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
            if(include_aligned)measure_aligned();
            if(include_native)measure_native();
        }
        else
        {
            if(include_native)measure_native();
            if(include_aligned)measure_aligned();
        }
        const double native_speedup = !include_native?missing:native.persistent_complete_us > 0.0
            ? native.serial_us / native.persistent_complete_us
            : 0.0;
        const double aligned_speedup = !include_aligned?missing:aligned.persistent_complete_us > 0.0
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
            << aligned_speedup
            << " purpose=" << projectionRowsPurposeName(purpose)
            << " gate_up_policy_n=" << gate_up_policy_n
            << " down_policy_n=" << down_policy_n << '\n';
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
                << aligned.persistent_complete_us << ',' << slowdown << ','
                << projectionRowsPurposeName(purpose) << ','
                << gate_up_policy_n << ',' << down_policy_n << ','
                << cpu::native_quantised::verifierRowsPolicyName(gate_up_schedule) << ','
                << cpu::native_quantised::verifierRowsPolicyName(down_schedule) << ','
                << (include_native?(include_aligned?"both":"native"):"aligned");
            if (ownership_diagnostic) csv << ',' << fusedOutputJobOwnershipName(gate_up_ownership)
                << ',' << fusedOutputJobOwnershipName(down_ownership);
            csv << ",aligned_phase_roots_v1" << '\n';
        }
        if(include_native)EXPECT_GT(native_speedup, 1.0)
            << format.label
            << " backend-native grouped execution must beat serial row replay";
        if(include_aligned)EXPECT_GT(aligned_speedup, 1.0)
            << format.label
            << " GPU-aligned grouped execution must beat serial row replay";
        ++executed;
    }
    ASSERT_GT(executed, 0)
        << "No canonical CPU MoE expert formats matched the selection";
}
