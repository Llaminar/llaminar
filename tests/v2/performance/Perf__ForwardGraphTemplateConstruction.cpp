/**
 * @file Perf__ForwardGraphTemplateConstruction.cpp
 * @brief Actual CPU Qwen template preparation versus N real specializations.
 *
 * Tiny source tensors and activation storage are created outside each timed
 * interval. This reports unprofiled host construction, not graph execution or
 * native capture performance. Counts are deterministic economy observations;
 * wall time is machine-specific and is never used to relax a correctness gate.
 */
#include "utils/ForwardGraphTemplateFixture.h"
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <gtest/gtest.h>
#include "utils/Logger.h"
#include <cstdlib>
#include <iostream>
#include <vector>

int main()
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    using Clock = std::chrono::steady_clock;
    // Preserve complete canonical CTest output while stdout stays one JSON document.
    std::cerr << "CTEST_FULL_OUTPUT\n";
    Logger::getInstance().setLogLevel(LogLevel::ERROR);
    ForwardTemplateConstructionFixture fixture;
    std::cout << "{\"benchmark\":\"actual_qwen_forward_construction\",\"backend\":\"CPU\","
        "\"compiler\":\"" << __VERSION__ << "\",\"omp_max_threads\":" << omp_get_max_threads()
        << ",\"host_overlap\":\"runner_must_record_competing_workloads\","
        "\"timing_scope\":\"unprofiled_construction_excludes_tensor_preparation_and_destruction\","
        "\"dense_geometry\":{\"layers\":1,\"hidden\":64,\"rows\":[1,4]},\"samples\":[";
    bool first = true;
    std::vector<std::pair<unsigned, std::int64_t>> dense_medians, routed_medians;
    for (const unsigned family : {1u, 2u, 4u, 8u, 16u, 32u, 64u})
    {
        std::vector<std::int64_t> times;
        for (unsigned sample = 0; sample < 5; ++sample)
        {
        auto builder = fixture.builder();
        std::vector<ComputeGraph> graphs; graphs.reserve(family);
        const auto begin = Clock::now();
        builder->prepareForwardGraphTemplate(DeviceId::cpu());
        const auto prepared = Clock::now();
        std::size_t stages = 0;
        for (unsigned member = 0; member < family; ++member)
        {
            const int rows = member % 2 ? 4 : 1;
            builder->setComputeAllPositionLogits(rows > 1);
            ForwardOutput output;
            graphs.push_back(builder->buildForwardGraph(fixture.input(rows), output));
            stages += graphs.back().size();
        }
        const auto done = Clock::now();
        const auto stats = builder->forwardGraphTemplateStats();
        if (stats.preparations != 1 || stats.specialization_attempts != family || stages == 0)
            return EXIT_FAILURE;
        if (!first) std::cout << ','; first = false;
        times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(done - prepared).count());
        std::cout << "{\"N\":" << family << ",\"sample\":" << sample << ",\"prepare_ns\":"
            << std::chrono::duration_cast<std::chrono::nanoseconds>(prepared - begin).count()
            << ",\"specialize_ns\":" << std::chrono::duration_cast<std::chrono::nanoseconds>(done - prepared).count()
            << ",\"preparations\":" << stats.preparations
            << ",\"specializations\":" << stats.specialization_attempts
            << ",\"stages\":" << stages << '}';
        }
        std::sort(times.begin(), times.end()); dense_medians.emplace_back(family, times[times.size()/2]);
    }
    std::cout << "],\"routed_geometry\":{\"layers\":2,\"hidden\":4,\"experts\":2,\"top_k\":1,\"rows\":[1,2]},\"routed_ffn_family_samples\":[";
    RoutedForwardTemplateFixture routed;
    first = true;
    for (const unsigned family : {1u, 2u, 4u, 8u, 16u, 32u, 64u})
    {
        std::vector<std::int64_t> times;
        for (unsigned sample = 0; sample < 5; ++sample)
        {
        auto builder = routed.builder();
        std::vector<ComputeGraph> graphs; graphs.reserve(family * routed.config.n_layers);
        const auto initial_calls = MoEExpertOwnerMap::initialBuildAttempts();
        const auto begin = Clock::now();
        builder->prepareForwardGraphTemplate(DeviceId::cpu());
        const auto prepared = Clock::now();
        std::size_t stages = 0;
        for (unsigned member = 0; member < family; ++member)
        for (int layer = 0; layer < routed.config.n_layers; ++layer)
        {
            graphs.push_back(routed.specialize(*builder, layer, member % 2 ? 2 : 1));
            if (!graphs.back().getNode("layer" + std::to_string(layer) + "_moe_expert_ffn_overlay_fast"))
                return EXIT_FAILURE;
            stages += graphs.back().size();
        }
        const auto done = Clock::now();
        const auto calls = MoEExpertOwnerMap::initialBuildAttempts() - initial_calls;
        const auto stats = builder->forwardGraphTemplateStats();
        if (stats.preparations != 1 || calls != 1 ||
            stats.routed_layer_specialization_attempts != family * routed.config.n_layers)
            return EXIT_FAILURE;
        if (!first) std::cout << ','; first = false;
        times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(done - prepared).count());
        std::cout << "{\"N\":" << family << ",\"sample\":" << sample << ",\"layers\":" << routed.config.n_layers
            << ",\"prepare_ns\":" << std::chrono::duration_cast<std::chrono::nanoseconds>(prepared - begin).count()
            << ",\"specialize_ns\":" << std::chrono::duration_cast<std::chrono::nanoseconds>(done - prepared).count()
            << ",\"canonical_map_calls\":" << calls
            << ",\"routed_layer_specializations\":" << stats.routed_layer_specialization_attempts
            << ",\"stages\":" << stages << '}';
        }
        std::sort(times.begin(), times.end()); routed_medians.emplace_back(family, times[times.size()/2]);
    }
    std::cout << "],\"median_specialize_ns\":{\"dense\":[";
    first = true;
    for (auto [family, median] : dense_medians)
    { if (!first) std::cout << ','; first = false; std::cout << "{\"N\":" << family << ",\"ns\":" << median << '}'; }
    std::cout << "],\"routed_ffn_family\":[";
    first = true;
    for (auto [family, median] : routed_medians)
    { if (!first) std::cout << ','; first = false; std::cout << "{\"N\":" << family << ",\"ns\":" << median << '}'; }
    std::cout << "]}}\n";
    return ::testing::UnitTest::GetInstance()->ad_hoc_test_result().Failed() ? EXIT_FAILURE : EXIT_SUCCESS;
}
