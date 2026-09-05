/**
 * @file Perf__CPUFloatingMoEExpert.cpp
 * @brief Exact-tree CPU floating ExpertOverlay microbenchmark.
 *
 * Qwen3.5-122B contains a routed-expert layer whose gate, up, and down
 * projections are stored as BF16 even in the Q8_K_XL artifact. ExpertOverlay
 * may move that expert between CPU, CUDA, and ROCm, so CPU execution must use
 * the same 256-lane reduction tree as both GPU backends. This benchmark pairs
 * that production numerical contract with the otherwise-equivalent CPU-native
 * projection at the real `d_model=3072`, `intermediate=1024` geometry for
 * FP16, BF16, and FP32 weights and decode/MTP row counts 1, 2, 3, and 15.
 * Each timed stream rotates through several experts, making the reuse distance
 * larger than the socket LLC instead of reporting an unrealistically cache-hot
 * single-expert result.
 *
 * Set `LLAMINAR_ISA_LEVEL=scalar|avx2|avx512` before process startup to select
 * one implementation through the canonical runtime ISA authority. Optional
 * `LLAMINAR_CPU_FLOATING_MOE_CSV` writes the same measurements as CSV for
 * paired tuning records. No model loading or device work occurs here.
 */

#include <gtest/gtest.h>

#include "kernels/cpu/gemm/FloatingPointGemmKernel.h"
#include "utils/CPUFeatures.h"
#include "utils/TestTensorFactory.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

using namespace llaminar2;

namespace
{
    /** Production width of Qwen3.5-122B routed hidden rows. */
    constexpr int kDefaultDModel = 3072;

    /** Production width of Qwen3.5-122B routed expert intermediates. */
    constexpr int kDefaultIntermediate = 1024;

    /** One floating storage format exercised by the common benchmark. */
    struct FormatCase
    {
        TensorType type; ///< Native expert weight type.
        const char *label; ///< Stable report label.
        std::size_t element_bytes; ///< Bytes fetched for one weight.
    };

    /** Timings and derived throughput for one `(format,M)` cell. */
    struct Timing
    {
        double gate_up_us = 0.0; ///< Two input projections.
        double down_us = 0.0; ///< Exact SwiGLU plus down projection.
        double complete_us = 0.0; ///< Complete three-projection transaction.
        double gflops = 0.0; ///< Mathematical projection throughput.
        double weight_gbps = 0.0; ///< Effective compulsory weight bandwidth.
        float checksum = 0.0f; ///< Observable output witness.
    };

    /** Native gate, up, and down weights for one logical expert. */
    using ExpertWeightSet = std::array<TensorBase *, 3>;

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

    /** @brief Parse a comma-separated positive integer selection. */
    std::vector<int> envPositiveIntList(
        const char *name,
        std::initializer_list<int> fallback)
    {
        const char *raw = std::getenv(name);
        if (!raw || !*raw)
            return std::vector<int>(fallback);
        std::vector<int> values;
        std::stringstream stream(raw);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            char *end = nullptr;
            const long parsed = std::strtol(token.c_str(), &end, 10);
            if (end == token.c_str() || *end != '\0' || parsed <= 0 ||
                parsed > std::numeric_limits<int>::max())
            {
                throw std::invalid_argument(
                    std::string(name) +
                    " must contain comma-separated positive integers");
            }
            values.push_back(static_cast<int>(parsed));
        }
        if (values.empty())
            throw std::invalid_argument(std::string(name) + " is empty");
        return values;
    }

    /** @brief Parse a comma-separated floating-format selection. */
    std::set<std::string> selectedFormats()
    {
        const char *raw = std::getenv("LLAMINAR_CPU_FLOATING_MOE_FORMATS");
        if (!raw || !*raw)
            return {"fp16", "bf16", "fp32"};
        std::set<std::string> values;
        std::stringstream stream(raw);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            if (!token.empty())
                values.insert(token);
        }
        if (values.empty())
        {
            throw std::invalid_argument(
                "LLAMINAR_CPU_FLOATING_MOE_FORMATS is empty");
        }
        return values;
    }

    /** @brief Construct deterministic floating weights in their native format. */
    std::unique_ptr<TensorBase> makeWeight(
        TensorType type,
        const std::vector<std::size_t> &shape,
        std::uint32_t seed)
    {
        switch (type)
        {
        case TensorType::FP16:
            return test::TestTensorFactory::createFP16Random(
                shape, -0.125f, 0.125f, seed);
        case TensorType::BF16:
            return test::TestTensorFactory::createBF16Random(
                shape, -0.125f, 0.125f, seed);
        case TensorType::FP32:
            return test::TestTensorFactory::createFP32Random(
                shape, -0.125f, 0.125f, seed);
        default:
            throw std::invalid_argument(
                "CPU floating MoE benchmark received a non-floating format");
        }
    }

    /** @brief Return median latency after untimed warmup. */
    template <typename Function>
    double medianMicros(Function &&function, int warmup, int samples)
    {
        for (int iteration = 0; iteration < warmup; ++iteration)
            function();
        std::vector<double> measurements;
        measurements.reserve(static_cast<std::size_t>(samples));
        for (int sample = 0; sample < samples; ++sample)
        {
            const auto begin = std::chrono::steady_clock::now();
            function();
            const auto end = std::chrono::steady_clock::now();
            measurements.push_back(
                std::chrono::duration<double, std::micro>(end - begin)
                    .count());
        }
        std::sort(measurements.begin(), measurements.end());
        return measurements[measurements.size() / 2u];
    }

    /**
     * @brief Time one complete floating expert at one runtime row count.
     *
     * M=1 uses the ordinary production projection bundle. M>1 uses the
     * grouped-verifier hooks selected by MoEExpertComputeStage. The caller
     * chooses either the production GPU-aligned expert contract or the
     * backend-native comparison contract; everything else remains identical.
     */
    Timing benchmarkCell(
        const FormatCase &format,
        const std::vector<ExpertWeightSet> &weight_sets,
        CPUProjectionNumericalPolicy numerical_policy,
        int m,
        int d_model,
        int intermediate,
        int warmup,
        int samples)
    {
        using Kernel = gemm::FloatingPointGemmKernel;
        struct PreparedExpert
        {
            std::unique_ptr<Kernel> gate; ///< Prepared gate projection.
            std::unique_ptr<Kernel> up; ///< Prepared up projection.
            std::unique_ptr<Kernel> down; ///< Prepared down projection.
            std::vector<ITensorGemm::TensorProjectionDesc> gate_up; ///< Fused input bundle.
        };

        if (weight_sets.empty())
            throw std::invalid_argument("Floating MoE benchmark needs an expert");

        auto input = test::TestTensorFactory::createFP32Random(
            {static_cast<std::size_t>(m),
             static_cast<std::size_t>(d_model)},
            -0.75f,
            0.75f,
            static_cast<std::uint32_t>(47000 + m));
        if (!input)
            throw std::runtime_error("Could not create floating MoE input");
        FP32Tensor gate_output(
            {static_cast<std::size_t>(m),
             static_cast<std::size_t>(intermediate)});
        FP32Tensor up_output(
            {static_cast<std::size_t>(m),
             static_cast<std::size_t>(intermediate)});
        FP32Tensor output(
            {static_cast<std::size_t>(m),
             static_cast<std::size_t>(d_model)});
        std::vector<PreparedExpert> experts;
        experts.reserve(weight_sets.size());
        for (const ExpertWeightSet &weights : weight_sets)
        {
            PreparedExpert expert{
                .gate = std::make_unique<Kernel>(
                    weights[0], numerical_policy),
                .up = std::make_unique<Kernel>(
                    weights[1], numerical_policy),
                .down = std::make_unique<Kernel>(
                    weights[2], numerical_policy),
            };
            expert.gate_up = {
                {expert.gate.get(), &gate_output, intermediate, nullptr, "gate"},
                {expert.up.get(), &up_output, intermediate, nullptr, "up"},
            };
            experts.push_back(std::move(expert));
        }

        const auto run_gate_up = [&](PreparedExpert &expert)
        {
            const bool ok = m == 1
                                ? expert.gate->multiply_fused_tensor(
                                      input.get(), expert.gate_up, m, d_model)
                                : expert.gate->
                                      multiply_fused_verifier_rows_decode_equivalent(
                                          input.get(),
                                          expert.gate_up,
                                          m,
                                          d_model);
            if (!ok)
                throw std::runtime_error("Floating gate/up projection failed");
        };
        const auto run_down = [&](PreparedExpert &expert)
        {
            const bool ok = m == 1
                                ? expert.down->multiply_tensor_with_fused_swiglu(
                                      &gate_output,
                                      &up_output,
                                      &output,
                                      m,
                                      d_model,
                                      intermediate)
                                : expert.down->
                                      multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                                          &gate_output,
                                          &up_output,
                                          &output,
                                          m,
                                          d_model,
                                          intermediate);
            if (!ok)
                throw std::runtime_error("Floating SwiGLU/down failed");
        };
        std::size_t gate_up_cursor = 0u;
        std::size_t down_cursor = 0u;
        std::size_t complete_cursor = 0u;
        const auto run_gate_up_rotating = [&]()
        {
            PreparedExpert &expert =
                experts[gate_up_cursor++ % experts.size()];
            run_gate_up(expert);
        };
        const auto run_down_rotating = [&]()
        {
            PreparedExpert &expert = experts[down_cursor++ % experts.size()];
            run_down(expert);
        };
        const auto run_complete_rotating = [&]()
        {
            PreparedExpert &expert =
                experts[complete_cursor++ % experts.size()];
            run_gate_up(expert);
            run_down(expert);
        };

        run_complete_rotating();
        Timing timing;
        timing.checksum = output.data()[0] + output.data()[output.numel() - 1u];
        timing.gate_up_us = medianMicros(
            run_gate_up_rotating, warmup, samples);
        timing.down_us = medianMicros(
            run_down_rotating, warmup, samples);
        timing.complete_us = medianMicros(
            run_complete_rotating, warmup, samples);
        const double operations =
            6.0 * static_cast<double>(m) * d_model * intermediate;
        timing.gflops = operations / (timing.complete_us * 1.0e3);
        const double weight_bytes =
            3.0 * static_cast<double>(d_model) * intermediate *
            static_cast<double>(format.element_bytes);
        timing.weight_gbps = weight_bytes / (timing.complete_us * 1.0e3);
        return timing;
    }
} // namespace

/**
 * @brief Compare GPU-aligned and backend-native floating expert arithmetic.
 */
TEST(Perf_CPUFloatingMoEExpert, ProductionGeometryAllFormats)
{
    constexpr std::array<FormatCase, 3> formats{{
        {TensorType::FP16, "fp16", sizeof(std::uint16_t)},
        {TensorType::BF16, "bf16", sizeof(std::uint16_t)},
        {TensorType::FP32, "fp32", sizeof(float)},
    }};
    const int d_model = envPositiveInt(
        "LLAMINAR_CPU_FLOATING_MOE_D_MODEL", kDefaultDModel);
    const int intermediate = envPositiveInt(
        "LLAMINAR_CPU_FLOATING_MOE_INTERMEDIATE", kDefaultIntermediate);
    const int warmup = envPositiveInt(
        "LLAMINAR_CPU_FLOATING_MOE_WARMUP", 2);
    const int samples = envPositiveInt(
        "LLAMINAR_CPU_FLOATING_MOE_SAMPLES", 7);
    const int active_experts = envPositiveInt(
        "LLAMINAR_CPU_FLOATING_MOE_ACTIVE_EXPERTS", 8);
    const std::vector<int> row_counts = envPositiveIntList(
        "LLAMINAR_CPU_FLOATING_MOE_ROWS", {1, 2, 3, 15});
    const std::set<std::string> requested = selectedFormats();

    std::ofstream csv;
    if (const char *path =
            std::getenv("LLAMINAR_CPU_FLOATING_MOE_CSV");
        path && *path)
    {
        csv.open(path, std::ios::trunc);
        ASSERT_TRUE(csv.is_open()) << "Could not open CSV output " << path;
        csv << "format,isa,m,d_model,intermediate,active_experts,threads,"
               "native_gate_up_us,native_down_us,native_complete_us,"
               "aligned_gate_up_us,aligned_down_us,aligned_complete_us,"
               "aligned_over_native,native_gflops,aligned_gflops,"
               "native_weight_gbps,aligned_weight_gbps,native_checksum,"
               "aligned_checksum\n";
    }

    std::cout
        << "\nCPU floating-expert numerical-policy speedometer"
        << " (isa=" << isaLevelName(activeISALevel())
        << ", threads=" << omp_get_max_threads()
        << ", d_model=" << d_model
        << ", intermediate=" << intermediate
        << ", active_experts=" << active_experts << ")\n"
        << "format m native_us aligned_us aligned/native native_gflops "
           "aligned_gflops native_weight_gbps aligned_weight_gbps\n";

    int cells = 0;
    for (const FormatCase &format : formats)
    {
        if (requested.count(format.label) == 0u &&
            requested.count("all") == 0u)
            continue;
        std::vector<std::array<std::unique_ptr<TensorBase>, 3>> owned_weights;
        std::vector<ExpertWeightSet> weight_sets;
        owned_weights.reserve(static_cast<std::size_t>(active_experts));
        weight_sets.reserve(static_cast<std::size_t>(active_experts));
        for (int expert = 0; expert < active_experts; ++expert)
        {
            const std::uint32_t seed =
                46000u + static_cast<std::uint32_t>(expert) * 3u;
            std::array<std::unique_ptr<TensorBase>, 3> weights{
                makeWeight(
                    format.type,
                    {static_cast<std::size_t>(intermediate),
                     static_cast<std::size_t>(d_model)},
                    seed + 1u),
                makeWeight(
                    format.type,
                    {static_cast<std::size_t>(intermediate),
                     static_cast<std::size_t>(d_model)},
                    seed + 2u),
                makeWeight(
                    format.type,
                    {static_cast<std::size_t>(d_model),
                     static_cast<std::size_t>(intermediate)},
                    seed + 3u),
            };
            ASSERT_NE(weights[0], nullptr);
            ASSERT_NE(weights[1], nullptr);
            ASSERT_NE(weights[2], nullptr);
            owned_weights.push_back(std::move(weights));
            const auto &owned = owned_weights.back();
            weight_sets.push_back(
                {owned[0].get(), owned[1].get(), owned[2].get()});
        }

        for (const int m : row_counts)
        {
            Timing native;
            Timing aligned;
            /*
             * Alternate the first policy between adjacent cells so one policy
             * cannot systematically inherit warmer source-weight cache lines.
             * Each measurement also rotates through the complete expert set.
             */
            const bool aligned_first = (cells & 1) != 0;
            auto measure_native = [&]
            {
                native = benchmarkCell(
                    format,
                    weight_sets,
                    CPUProjectionNumericalPolicy::BackendNative,
                    m,
                    d_model,
                    intermediate,
                    warmup,
                    samples);
            };
            auto measure_aligned = [&]
            {
                aligned = benchmarkCell(
                    format,
                    weight_sets,
                    CPUProjectionNumericalPolicy::GPUAlignedExpert,
                    m,
                    d_model,
                    intermediate,
                    warmup,
                    samples);
            };
            if (aligned_first)
            {
                measure_aligned();
                measure_native();
            }
            else
            {
                measure_native();
                measure_aligned();
            }
            const double slowdown =
                aligned.complete_us / native.complete_us;
            std::cout
                << format.label << ' ' << m << ' '
                << std::fixed << std::setprecision(3)
                << native.complete_us << ' '
                << aligned.complete_us << ' '
                << slowdown << ' '
                << native.gflops << ' '
                << aligned.gflops << ' '
                << native.weight_gbps << ' '
                << aligned.weight_gbps << '\n';
            if (csv)
            {
                csv << format.label << ','
                    << isaLevelName(activeISALevel()) << ','
                    << m << ',' << d_model << ',' << intermediate << ','
                    << active_experts << ','
                    << omp_get_max_threads() << ','
                    << native.gate_up_us << ',' << native.down_us << ','
                    << native.complete_us << ','
                    << aligned.gate_up_us << ',' << aligned.down_us << ','
                    << aligned.complete_us << ',' << slowdown << ','
                    << native.gflops << ',' << aligned.gflops << ','
                    << native.weight_gbps << ',' << aligned.weight_gbps << ','
                    << native.checksum << ',' << aligned.checksum << '\n';
            }
            EXPECT_GT(native.complete_us, 0.0);
            EXPECT_GT(aligned.complete_us, 0.0);
            EXPECT_TRUE(std::isfinite(native.checksum));
            EXPECT_TRUE(std::isfinite(aligned.checksum));
            ++cells;
        }
    }
    ASSERT_GT(cells, 0) << "No floating format matched the selection";
}
