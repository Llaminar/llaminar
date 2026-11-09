/**
 * @file Perf__CPUFusedVerifier.cpp
 * @brief Isolate CPU fused-projection scheduling from standalone kernel cost.
 *
 * Both paths use independent prepared weights and the same selected policy;
 * Auto is the default, with explicit row-reuse candidates for diagnosis only.
 * Serial M=1 rows are a diagnostic oracle only. Timing alternates path order
 * and publishes raw samples after complete byte checks. Timing is a standalone
 * economy diagnostic, not a policy installer. Focused model-free preflight
 * aliases validate output layout and byte/count contracts without speed gates.
 * Counter-bundle timing uses contiguous 64-byte aligned expert-major output
 * storage with logical-N row strides. Separate N+7 guarded replays before and
 * after timing retain tail-write checks without imposing padded timing rows.
 * The native-Q6 serial-tile experiment similarly compares existing physical
 * register geometries without modifying the production selector.
 * Exact-batch warmup/timing suppresses recording while final fresh route
 * proofs remain fully observed; focused tests cover both full/adaptive clocks.
 */

#include "../../../../utils/NativeQuantisedTestPartialStorage.h"
#include "../../../../utils/CPUPerfStatsTimingCollectionScope.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <tuple>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <sched.h>
#include <fstream>
#include <exception>
#include <string>
#include <vector>
#include <omp.h>

#include "kernels/cpu/gemm/CPUNativeQuantisedGemmKernel.h"
#include "kernels/cpu/gemm/FloatingPointGemmKernel.h"
#include "utils/QuantizedVerifierFormats.h"
#include "utils/NativeQuantisedTrainerEvidence.h"
#include "../../native_quantised_dispatch/NativeQuantisedProfilerControl.h"
#include <nlohmann/json.hpp>
#include <sstream>
#include <filesystem>
#include "app/MPIBootstrapPhase.h"
#include "tensors/AlignedVector.h"

namespace
{
    using namespace llaminar2;
    using namespace llaminar2::cpu::native_quantised;
    using namespace llaminar2::test;

    /** Read a positive diagnostic geometry without changing production policy. */
    int dimension(const char *name, int default_value)
    {
        const char *text = std::getenv(name);
        if (!text)
            return default_value;
        size_t consumed = 0;
        const int value = std::stoi(text, &consumed);
        if (value <= 0 || consumed != std::strlen(text))
            throw std::invalid_argument(std::string("invalid diagnostic dimension: ") + name);
        return value;
    }

    /** Select a row-reuse experiment using production policy names, never an installed override. */
    VerifierRowsPolicy diagnosticPolicy()
    {
        const char *requested = std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_POLICY");
        if (!requested)
            return VerifierRowsPolicy::Auto;
        for (const auto policy : {VerifierRowsPolicy::Auto,
                 VerifierRowsPolicy::Pairwise, VerifierRowsPolicy::WideRows})
            if (std::strcmp(requested, verifierRowsPolicyName(policy)) == 0)
                return policy;
        throw std::invalid_argument("unsupported fused verifier diagnostic policy");
    }

    /** One format from the shared complete source-format registry. */
    class CPUFusedVerifier : public ::testing::TestWithParam<QuantizedVerifierFormatCase> {};

    /** Compare a real two-projection bundle with the same two standalone calls. */
    TEST_P(CPUFusedVerifier, BundleVersusStandalone)
    {
        const auto &format_a = GetParam();
        const auto *format_b = &format_a;
        if (const char *requested = std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_FORMAT_B"))
        {
            const auto &formats = quantizedVerifierFormats();
            const auto found = std::find_if(formats.begin(), formats.end(),
                [&](const auto &format) { return std::strcmp(format.label, requested) == 0; });
            ASSERT_NE(found, formats.end());
            format_b = &*found;
        }
        const int m = dimension("LLAMINAR_CPU_FUSED_VERIFIER_M", 16);
        const int n_a = dimension("LLAMINAR_CPU_FUSED_VERIFIER_N_A", 17408);
        const int n_b = dimension("LLAMINAR_CPU_FUSED_VERIFIER_N_B", 17408);
        const int k = dimension("LLAMINAR_CPU_FUSED_VERIFIER_K", 5120);
        const auto policy = diagnosticPolicy();
        ASSERT_GT(m, 1);
        ASSERT_EQ(k % 256, 0);

        auto source_a = format_a.create({static_cast<size_t>(n_a), static_cast<size_t>(k)}, 4242);
        auto source_b = format_b->create({static_cast<size_t>(n_b), static_cast<size_t>(k)}, 4243);
        CPUNativeQuantisedGemmKernel kernel_a(source_a.get()), kernel_b(source_b.get());
        ASSERT_TRUE(kernel_a.isValid());
        ASSERT_TRUE(kernel_b.isValid());
        const std::array<const CPUNativeQuantisedPackedWeights *, 2> weights = {
            &kernel_a.packedWeights(), &kernel_b.packedWeights()};
        const std::array<int, 2> columns = {n_a, n_b};
        std::array<NativeQuantisedTestPartialStorage, 2> partial_storage = {
            NativeQuantisedTestPartialStorage(*weights[0], m), NativeQuantisedTestPartialStorage(*weights[1], m)};
        std::mt19937 random(4242);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        std::vector<float> input(static_cast<size_t>(m) * k);
        for (auto &value : input)
            value = distribution(random);
        std::vector<Q8_1Block> quantized(static_cast<size_t>(m) * k / 32);
        quantize_activations_to_q8_1(input.data(), quantized.data(), m, k, k / 32);
        std::array<std::vector<float>, 2> fused, separate, serial;
        std::array<FusedVerifierRowsDesc, 2> descriptors;
        for (int projection = 0; projection < 2; ++projection)
        {
            const size_t size = static_cast<size_t>(m) * columns[projection];
            fused[projection].resize(size);
            separate[projection].resize(size);
            serial[projection].resize(size);
            descriptors[projection] = {.packed = weights[projection],
                .output = fused[projection].data(), .bias = nullptr,
                .N = columns[projection], .ldc = columns[projection],
                .verifier_schedule = policy};
            for (int row = 0; row < m; ++row)
                gemv_native_quantised_preq(*weights[projection], quantized.data() + row * (k / 32),
                    serial[projection].data() + row * columns[projection], partial_storage[projection].span());
        }
        llaminar2::test::NativeQuantisedTestPartialStorage fused_storage(descriptors.data(), 2, m);
        const auto run = [&](bool bundle)
        {
            if (bundle)
            {
                if (!gemm_native_quantised_fused_verifier_rows_preq(quantized.data(),
                        descriptors.data(), fused_storage.span(), 2, m, k / 32))
                    throw std::runtime_error("production fused verifier rejected its descriptors");
            }
            else
                for (int projection = 0; projection < 2; ++projection)
                    gemm_native_quantised_preq_decode_equivalent_rows(*weights[projection],
                        quantized.data(), separate[projection].data(), partial_storage[projection].span(), m, columns[projection],
                        ISAPath::AUTO, policy);
        };
        run(false);
        run(true);
        // Optional diagnostic counters describe the executed routes, not a
        // second reconstruction of the production dispatch decision.
        for (const auto &record : PerfStatsCollector::snapshot({"kernel"}))
        {
            if (record.name.find("launch") == std::string::npos)
                continue;
            std::cout << "route," << record.name;
            for (const auto &[key, value] : record.tags)
                std::cout << ',' << key << '=' << value;
            std::cout << '\n';
        }
        for (int projection = 0; projection < 2; ++projection)
        {
            const size_t bytes = serial[projection].size() * sizeof(float);
            ASSERT_EQ(std::memcmp(serial[projection].data(), fused[projection].data(), bytes), 0);
            ASSERT_EQ(std::memcmp(serial[projection].data(), separate[projection].data(), bytes), 0);
        }

        // A profiler may isolate one path after setup. Paired timing remains
        // the default and is never replaced by profiler-contaminated evidence.
        const char *profile_path = std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_PROFILE_PATH");
        if (profile_path)
        {
            const std::string path(profile_path);
            ASSERT_TRUE(path == "bundle" || path == "separate");
            const int iterations = dimension("LLAMINAR_CPU_FUSED_VERIFIER_ITERS", 1000);
            for (int iteration = 0; iteration < iterations; ++iteration)
                run(path == "bundle");
            return;
        }
        std::cout << "format_a,format_b,m,n_a,n_b,k,threads,policy,sample,first,bundle_us,separate_us\n";
        // Several calls per sample amortize cold thread-team wakeup variance.
        constexpr int repetitions = 16;
        for (int sample = -2; sample < 12; ++sample)
        {
            std::array<double, 2> times{};
            for (int order = 0; order < 2; ++order)
            {
                const bool bundle = ((sample + 2 + order) % 2) == 0;
                const auto start = std::chrono::steady_clock::now();
                for (int repeat = 0; repeat < repetitions; ++repeat)
                    run(bundle);
                times[bundle ? 0 : 1] = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - start).count() / repetitions;
            }
            if (sample >= 0)
                std::cout << format_a.label << ',' << format_b->label << ',' << m << ','
                          << n_a << ',' << n_b << ',' << k << ',' << omp_get_max_threads() << ','
                          << verifierRowsPolicyName(policy) << ','
                          << sample << ',' << (sample % 2 ? "separate" : "bundle") << ','
                          << times[0] << ',' << times[1] << '\n';
        }
    }

    /**
     * @test Replay one actual model slice and live bundle with optional enabled counter evidence.
     *
     * Every projection retains independent prepared weights, output and serial
     * byte oracle. Geometry, policy, ISA and worker count stay production-owned;
     * no diagnostic K partition or simplified arithmetic is selected. Buffers
     * and exact planned partials are allocated before warmup/timing. The same
     * executable accepts enabled/disabled collection in fresh processes.
     */
    /**
     * @brief Authenticate actual worker masks through the existing topology authority.
     * @param actual_workers Receives the observed native team width.
     * @param allowed_physical_cores Receives the union of admitted physical cores.
     * @param disjoint_single_core_masks True only for distinct singleton core masks.
     *
     * Instantaneous CPUs remain observations, never pinned-placement proof.
     * This shared fixture helper performs no model or accelerator work.
     */
    void captureCPUFusedPlacement(int &actual_workers, std::set<int> &allowed_physical_cores,
        bool &disjoint_single_core_masks)
    {
        // A requested team size is not a physical-core placement proof. Sample
        // every actual worker before allocation or timing, retain its native
        // mask, and reject an inherited domain too small for physical workers.
        // Current CPUs are transient observations; shared/unbound masks do not
        // certify disjoint placement, even when this sample happens to differ.
        const int requested_workers = omp_get_max_threads();
        std::vector<cpu_set_t> worker_masks(static_cast<size_t>(requested_workers));
        std::vector<int> worker_cpus(static_cast<size_t>(requested_workers), -1);
        std::vector<int> affinity_errors(static_cast<size_t>(requested_workers), 0);
        actual_workers = 0;
#pragma omp parallel
        {
            const int worker = omp_get_thread_num();
#pragma omp single
            actual_workers = omp_get_num_threads();
            if (worker < requested_workers)
            {
                CPU_ZERO(&worker_masks[worker]);
                affinity_errors[worker] = sched_getaffinity(0, sizeof(cpu_set_t), &worker_masks[worker]);
                worker_cpus[worker] = sched_getcpu();
            }
        }
        ASSERT_EQ(actual_workers, requested_workers);
        std::set<int> sampled_physical_cores;
        allowed_physical_cores.clear();
        std::set<int> singleton_mask_cores;
        disjoint_single_core_masks = true;
        for (int worker = 0; worker < actual_workers; ++worker)
        {
            ASSERT_EQ(affinity_errors[worker], 0) << "Cannot observe actual worker affinity";
            const int cpu = worker_cpus[worker];
            ASSERT_GE(cpu, 0);
            ASSERT_TRUE(CPU_ISSET(cpu, &worker_masks[worker]));
            // The existing startup topology authority resolves physical cores.
            // Missing sysfs evidence cannot authenticate a physical-core result.
            ASSERT_TRUE(std::ifstream("/sys/devices/system/cpu/cpu" + std::to_string(cpu)
                + "/topology/thread_siblings_list").good());
            const int core = MPIBootstrapPhase::physicalRepresentativeForCpu(cpu);
            sampled_physical_cores.insert(core);
            std::set<int> worker_physical_cores;
            std::cout << "placement,worker=" << worker << ",cpu=" << cpu
                      << ",physical_core=" << core << ",allowed=";
            bool first = true;
            for (int allowed = 0; allowed < CPU_SETSIZE; ++allowed)
                if (CPU_ISSET(allowed, &worker_masks[worker]))
                {
                    ASSERT_TRUE(std::ifstream("/sys/devices/system/cpu/cpu" + std::to_string(allowed)
                        + "/topology/thread_siblings_list").good());
                    const int allowed_core = MPIBootstrapPhase::physicalRepresentativeForCpu(allowed);
                    allowed_physical_cores.insert(allowed_core);
                    worker_physical_cores.insert(allowed_core);
                    if (!first) std::cout << ';';
                    std::cout << allowed;
                    first = false;
                }
            std::cout << '\n';
            if (worker_physical_cores.size() != 1)
                disjoint_single_core_masks = false;
            else if (!singleton_mask_cores.insert(*worker_physical_cores.begin()).second)
                disjoint_single_core_masks = false;
        }
        ASSERT_GE(allowed_physical_cores.size(), static_cast<size_t>(actual_workers))
            << "Received CPU domain cannot admit the requested physical workers";
        std::cout << "placement_summary,workers=" << actual_workers
                  << ",admitted_physical_cores=" << allowed_physical_cores.size()
                  << ",mask_contract=" << (disjoint_single_core_masks
                      ? "disjoint_single_core" : "shared_or_multi_core")
                  << ",instantaneous_distinct_physical_cores=" << sampled_physical_cores.size() << '\n';
    }

#include "CPUExecutionExactBatch.inc"

    /** @brief Authored original-owner clock lifecycle exercised by the tests below. */
    enum class CPUExactCollectionFixtureMode { Full, FullStandalone, Adaptive };

    /** @brief Execute real batch bytes/counters under one explicit clock protocol.
     * @param mode Original full or adaptive retained-team batch entrypoint.
     */
    void proveCPUExactBatchCollection(CPUExactCollectionFixtureMode mode)
    {
        std::string pattern = "/tmp/llaminar-execution-exact-batch-XXXXXX";
        char *created = mkdtemp(pattern.data());
        ASSERT_NE(created, nullptr);
        struct OwnedDirectory
        {
            std::filesystem::path path;
            /** @brief Retire only this test's uniquely owned temporary results. */
            ~OwnedDirectory() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
        } directory{created};
        struct CollectionScope
        {
            std::optional<std::string> prior_json, prior_filter;
            /** @brief Bind the real route recorder, retaining caller collection ownership. */
            CollectionScope()
            {
                if (const char *text = std::getenv("LLAMINAR_PERF_STATS_JSON")) prior_json = text;
                if (const char *text = std::getenv("LLAMINAR_PERF_STATS_FILTER")) prior_filter = text;
                setenv("LLAMINAR_PERF_STATS_JSON", "1", 1);
                setenv("LLAMINAR_PERF_STATS_FILTER", "kernel.cpu_native_quantised_fused_verifier_rows_projection_launch", 1);
                PerfStatsCollector::reset();
            }
            /** @brief Restore the original collection contract without exporting test state. */
            ~CollectionScope()
            {
                if (prior_json) setenv("LLAMINAR_PERF_STATS_JSON", prior_json->c_str(), 1);
                else unsetenv("LLAMINAR_PERF_STATS_JSON");
                if (prior_filter) setenv("LLAMINAR_PERF_STATS_FILTER", prior_filter->c_str(), 1);
                else unsetenv("LLAMINAR_PERF_STATS_FILTER");
                PerfStatsCollector::reset();
            }
        } collection;
        struct TimingSettingsScope
        {
            std::map<std::string, std::optional<std::string>> prior;
            /** @brief Supply small real geometry and canonical adaptive sample floors. */
            explicit TimingSettingsScope(CPUExactCollectionFixtureMode mode)
            {
                // N80 keeps an N64 tail while ldc=N has a 64-byte-aligned
                // row stride. N65 would violate the exact class's layout ABI.
                const std::map<std::string, std::string> fields{
                    {"LLAMINAR_CPU_FUSED_VERIFIER_NUMERICAL_POLICY", "GPUAlignedExpert"},
                    {"LLAMINAR_CPU_FUSED_VERIFIER_N_A", "80"},
                    {"LLAMINAR_CPU_FUSED_VERIFIER_K", "512"},
                    {"LLAMINAR_CPU_FUSED_VERIFIER_POLICY_N", "80"},
                    {"LLAMINAR_CPU_FUSED_VERIFIER_TEAM_LIFETIME", mode == CPUExactCollectionFixtureMode::FullStandalone ? "standalone" : "retained"},
                    {"LLAMINAR_CPU_FUSED_VERIFIER_OUTPUT_ROLE", "projection"},
                    {"LLAMINAR_CPU_FUSED_VERIFIER_WARMUP", "5"},
                    {"LLAMINAR_CPU_FUSED_VERIFIER_SAMPLES", "30"},
                    {"LLAMINAR_CPU_FUSED_VERIFIER_REPEATS", "2"},
                    {"LLAMINAR_NATIVE_QUANTISED_TOURNAMENT", mode == CPUExactCollectionFixtureMode::Adaptive ? "adaptive_v1" : ""}};
                for (const auto &[name, value] : fields)
                {
                    const char *old = std::getenv(name.c_str());
                    prior.emplace(name, old ? std::optional<std::string>(old) : std::nullopt);
                    setenv(name.c_str(), value.c_str(), 1);
                }
            }
            /** @brief Withdraw only this authored fixture's temporary settings. */
            ~TimingSettingsScope()
            {
                for (const auto &[name, value] : prior)
                    if (value) setenv(name.c_str(), value->c_str(), 1);
                    else unsetenv(name.c_str());
            }
        } timing_settings(mode);
        const auto &formats = quantizedVerifierFormats();
        const auto format = std::find_if(formats.begin(), formats.end(),
            [](const auto &entry) { return std::string(entry.label) == "Q4_1"; });
        ASSERT_NE(format, formats.end());
#if LLAMINAR_COMPILED_WITH_AVX512
        const bool wide = activeISALevel() >= ISALevel::AVX512;
#else
        const bool wide = false;
#endif
        const bool adaptive = mode == CPUExactCollectionFixtureMode::Adaptive;
        nlohmann::json manifest{{"schema", adaptive ? "cpu_execution_exact_batch_v2" : "cpu_execution_exact_batch_v1"},
            {"shape", "fixture-only-mixed-target-replay"}, {"requests", nlohmann::json::array()}};
        std::vector<std::vector<int>> rows{{8, 3}, {3, 1}, {8, 1}};
        std::vector<std::string> purposes{"prefill", "mtp_verifier", "decode"};
        std::vector<std::string> candidates{wide ? "WideRows" : "Pairwise", "Pairwise", "Pairwise"};
        if (adaptive)
        {
            rows = wide ? std::vector<std::vector<int>>{{3, 1}, {3, 1}}
                        : std::vector<std::vector<int>>{{3, 1}};
            purposes.assign(rows.size(), "mtp_verifier");
            candidates = wide ? std::vector<std::string>{"Pairwise", "WideRows"}
                              : std::vector<std::string>{"Pairwise"};
        }
        for (size_t index = 0; index < rows.size(); ++index)
            manifest["requests"].push_back({{"id", std::to_string(index)}, {"m", rows[index][0]},
                {"purpose", purposes[index]}, {"expert_rows", rows[index]}, {"target_descriptor", 0},
                {"candidate", candidates[index]}, {"holdout", true},
                {"evidence_path", (directory.path / (std::to_string(index) + ".json")).string()}});
        const auto request_path = directory.path / "requests.json";
        { std::ofstream output(request_path); ASSERT_TRUE(output.is_open()); output << manifest.dump(2); }
        runCPUExecutionExactBatch(*format, request_path.c_str());
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        for (size_t index = 0; index < rows.size(); ++index)
        {
            std::ifstream input(directory.path / (std::to_string(index) + ".json")); ASSERT_TRUE(input.is_open());
            const auto result = nlohmann::json::parse(input);
            EXPECT_TRUE(result.at("exact_key_verified").get<bool>());
            EXPECT_TRUE(result.at("correctness").at("bitwise_equal").get<bool>());
            EXPECT_TRUE(result.at("counters").at("enabled").get<bool>());
            EXPECT_EQ(result.at("counters").at("authenticated_projection_observations").get<uint64_t>(),
                result.at("descriptor_keys").size());
            // Original instrumented code records warmup/sample calls here and
            // genuinely fails this behavioral count before checking new markers.
            EXPECT_EQ(result.at("counters").at("projection_observations").get<uint64_t>(),
                adaptive ? 0u : 2u * result.at("descriptor_keys").size());
            EXPECT_EQ(result.at("counters").at("expected_projection_observations").get<uint64_t>(),
                adaptive ? 0u : 2u * result.at("descriptor_keys").size());
            EXPECT_TRUE(result.at("counters").value("timed_collection_disabled", false));
            EXPECT_EQ(result.at("counters").value("timed_projection_observations", uint64_t{1}), 0u);
            EXPECT_TRUE(PerfStatsCollector::isDomainEnabled("kernel"));
            EXPECT_EQ(result.at("purpose").get<std::string>(), purposes[index]);
            const auto target = result.at("exact_key");
            const auto keys = result.at("descriptor_keys");
            for (size_t member = 0; member < keys.size(); ++member)
                if (keys[member] != target)
                    EXPECT_EQ(result.at("descriptor_policies")[member], result.at("baseline_descriptor_policies")[member]);
        }
    }

    /** @test One prepared owner preserves target-only policies and dirty mixed-M replay. */
    TEST(CPUExecutionExactBatch, MixedTargetPoliciesAndLargeSmallReuse)
    {
        proveCPUExactBatchCollection(CPUExactCollectionFixtureMode::Full);
    }

    /** @test Real full/adaptive clocks record zero work; final fresh proofs remain enabled. */
    TEST(CPUExecutionExactBatch, TimingCollectionIsSuppressedAndRestored)
    {
        proveCPUExactBatchCollection(CPUExactCollectionFixtureMode::Full);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        proveCPUExactBatchCollection(CPUExactCollectionFixtureMode::FullStandalone);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        proveCPUExactBatchCollection(CPUExactCollectionFixtureMode::Adaptive);
    }

    TEST_P(CPUFusedVerifier, ProjectionCounterBundleReplay)
    {
        const auto &format = GetParam();
        if (const char *batch = std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_EXACT_REQUESTS_PATH"))
        {
            runCPUExecutionExactBatch(format, batch);
            return;
        }
        const int m = dimension("LLAMINAR_CPU_FUSED_VERIFIER_M", 3);
        const int n = dimension("LLAMINAR_CPU_FUSED_VERIFIER_N_A", 320);
        const int k = dimension("LLAMINAR_CPU_FUSED_VERIFIER_K", 2560);
        const int projections = dimension("LLAMINAR_CPU_FUSED_VERIFIER_PROJECTIONS", 2);
        const int n_step = dimension("LLAMINAR_CPU_FUSED_VERIFIER_N_STEP", 0);
        const int warmup = dimension("LLAMINAR_CPU_FUSED_VERIFIER_WARMUP", 4);
        const int samples = dimension("LLAMINAR_CPU_FUSED_VERIFIER_SAMPLES", 12);
        const int repeats = dimension("LLAMINAR_CPU_FUSED_VERIFIER_REPEATS", 8);
        const auto policy = diagnosticPolicy();
        const std::string exact_profile_request = native_quantised_dispatch::profilerRequestId();
        std::optional<native_quantised_dispatch::LinuxPerfControl> exact_perf_control;
        if (!exact_profile_request.empty()) exact_perf_control.emplace(exact_profile_request);
        const std::string purpose_name = std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_PURPOSE")
            ? std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_PURPOSE") : "decode";
        ASSERT_TRUE(purpose_name == "decode" || purpose_name == "mtp_verifier" || purpose_name == "prefill");
        const auto purpose = purpose_name == "prefill" ? ProjectionRowsPurpose::Prefill
            : purpose_name == "mtp_verifier" ? ProjectionRowsPurpose::MTPVerifier : ProjectionRowsPurpose::Decode;
        const std::string numerical_name = std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_NUMERICAL_POLICY")
            ? std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_NUMERICAL_POLICY") : "BackendNative";
        ASSERT_TRUE(numerical_name == "BackendNative" || numerical_name == "GPUAlignedExpert");
        const auto numerical_policy = numerical_name == "GPUAlignedExpert"
            ? CPUProjectionNumericalPolicy::GPUAlignedExpert : CPUProjectionNumericalPolicy::BackendNative;
        const int requested_policy_n = dimension("LLAMINAR_CPU_FUSED_VERIFIER_POLICY_N", 0);
        const int expected_k_tiles = dimension("LLAMINAR_CPU_FUSED_VERIFIER_EXPECTED_K_TILES", 0);
        std::optional<int> expected_codebook;
        if (const char *text = std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_EXPECTED_CODEBOOK"))
        {
            size_t consumed = 0;
            expected_codebook = std::stoi(text, &consumed);
            ASSERT_EQ(consumed, std::strlen(text));
            ASSERT_GE(*expected_codebook, 0);
            ASSERT_LE(*expected_codebook, std::numeric_limits<uint8_t>::max());
        }
        if (numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
        {
            ASSERT_GT(requested_policy_n, 0) << "Expert replay requires its observed original policy N";
            ASSERT_GT(expected_k_tiles, 0) << "Expert replay requires its observed production K partitions";
            ASSERT_TRUE(expected_codebook.has_value()) << "Expert replay requires its observed native codebook";
        }
        enum class TeamLifetime { Standalone, Retained };
        const std::string team_name = std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_TEAM_LIFETIME")
            ? std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_TEAM_LIFETIME") : "standalone";
        ASSERT_TRUE(team_name == "standalone" || team_name == "retained");
        const auto team_lifetime = team_name == "retained" ? TeamLifetime::Retained : TeamLifetime::Standalone;
        enum class OutputRole { Projection, GateUp, Down };
        const std::string output_role_name = std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_OUTPUT_ROLE")
            ? std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_OUTPUT_ROLE") : "projection";
        ASSERT_TRUE(output_role_name == "projection" || output_role_name == "gate_up" || output_role_name == "down");
        const auto output_role = output_role_name == "gate_up" ? OutputRole::GateUp
            : output_role_name == "down" ? OutputRole::Down : OutputRole::Projection;
        const size_t output_root_count = output_role == OutputRole::GateUp ? 2u : 1u;
        ASSERT_TRUE(output_role != OutputRole::GateUp || projections % 2 == 0)
            << "A gate/up bundle must contain complete interleaved expert pairs";
        ASSERT_TRUE(output_role == OutputRole::Projection || n_step == 0)
            << "Expert role replay requires one model width; varying-N controls use projection layout";
        const size_t expert_count = output_role == OutputRole::GateUp
            ? static_cast<size_t>(projections / 2) : static_cast<size_t>(projections);
        std::vector<int> expert_rows(expert_count, m);
        if (const char *text = std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_EXPERT_ROWS"))
        {
            std::istringstream stream(text);
            std::string item;
            size_t expert = 0;
            while (std::getline(stream, item, ','))
            {
                ASSERT_LT(expert, expert_count);
                size_t consumed = 0;
                const int rows = std::stoi(item, &consumed);
                ASSERT_EQ(consumed, item.size()); ASSERT_GT(rows, 0);
                expert_rows[expert++] = rows;
            }
            ASSERT_EQ(expert, expert_count);
        }
        std::vector<int> member_rows(static_cast<size_t>(projections));
        std::vector<size_t> activation_offsets(static_cast<size_t>(projections));
        std::vector<size_t> expert_offsets(expert_count);
        size_t activation_rows = 0;
        int maximum_rows = m;
        for (size_t expert = 0; expert < expert_count; ++expert)
        {
            expert_offsets[expert] = activation_rows;
            ASSERT_LE(static_cast<size_t>(expert_rows[expert]), std::numeric_limits<size_t>::max() - activation_rows);
            activation_rows += static_cast<size_t>(expert_rows[expert]);
            maximum_rows = std::max(maximum_rows, expert_rows[expert]);
        }
        for (int member = 0; member < projections; ++member)
        {
            const size_t expert = output_role == OutputRole::GateUp ? static_cast<size_t>(member / 2)
                : static_cast<size_t>(member);
            member_rows[member] = expert_rows[expert];
            activation_offsets[member] = expert_offsets[expert];
        }
        const bool exact_target_mode = std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_EXACT_TARGET_DESCRIPTOR") != nullptr;
        int target_descriptor = 0;
        if (const char *text = std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_EXACT_TARGET_DESCRIPTOR"))
        {
            size_t consumed = 0;
            target_descriptor = std::stoi(text, &consumed);
            ASSERT_EQ(consumed, std::strlen(text));
        }
        ASSERT_LT(target_descriptor, projections); ASSERT_GE(target_descriptor, 0);
        if (exact_target_mode)
        {
            ASSERT_EQ(member_rows[target_descriptor], m);
            ASSERT_GT(m, 1) << "M1 belongs to the existing decode surface, not a grouped row overlay";
            ASSERT_EQ(n_step, 0);
            ASSERT_EQ(numerical_policy, CPUProjectionNumericalPolicy::GPUAlignedExpert);
        }
        constexpr float guard = -12345.0f;
        ASSERT_LE(projections, NativeQuantisedFusedRowsPlan::capacity);
        ASSERT_LE(static_cast<int64_t>(n) + static_cast<int64_t>(projections - 1) * n_step,
            static_cast<int64_t>(std::numeric_limits<int>::max()) - 7);
        ASSERT_NO_THROW(requireQuantizedVerifierSourceShape(format, n, k));
        ASSERT_TRUE(m > 1 || policy == VerifierRowsPolicy::Auto);
        int actual_workers = 0;
        std::set<int> allowed_physical_cores;
        bool disjoint_single_core_masks = false;
        captureCPUFusedPlacement(actual_workers, allowed_physical_cores, disjoint_single_core_masks);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        std::vector<std::unique_ptr<TensorBase>> sources;
        std::vector<std::unique_ptr<CPUNativeQuantisedGemmKernel>> kernels;
        // Production stores expert-major outputs in aligned backing with
        // logical-N row strides. Padded buffers belong only to guard replays.
        std::vector<std::vector<float>> guarded_outputs(static_cast<size_t>(projections));
        std::vector<std::vector<float>> references(static_cast<size_t>(projections));
        size_t live_output_floats = 0;
        std::array<size_t, 2> root_floats{};
        std::vector<size_t> projection_offsets(static_cast<size_t>(projections));
        for (int p = 0; p < projections; ++p)
        {
            const size_t columns = static_cast<size_t>(n + p * n_step);
            ASSERT_LE(static_cast<size_t>(member_rows[p]), std::numeric_limits<size_t>::max() / columns);
            const size_t expert_floats = static_cast<size_t>(member_rows[p]) * columns;
            ASSERT_LE(expert_floats, std::numeric_limits<size_t>::max() - live_output_floats);
            live_output_floats += expert_floats;
            const size_t root = static_cast<size_t>(p) % output_root_count;
            projection_offsets[p] = root_floats[root];
            root_floats[root] += expert_floats;
        }
        std::vector<AlignedVector<float>> timed_outputs;
        timed_outputs.reserve(output_root_count);
        for (size_t root = 0; root < output_root_count; ++root)
        {
            timed_outputs.emplace_back(root_floats[root], guard);
            ASSERT_EQ(reinterpret_cast<std::uintptr_t>(timed_outputs.back().data())
                % AlignedVector<float>::ALIGNMENT, 0u);
        }
        std::vector<FusedVerifierRowsDesc> descriptors;
        std::vector<FusedVerifierRowsDesc> guarded_descriptors;
        uint64_t weight_bytes = 0;
        sources.reserve(projections); kernels.reserve(projections); descriptors.reserve(projections);
        guarded_descriptors.reserve(projections);
        ASSERT_LE(activation_rows, std::numeric_limits<size_t>::max() / static_cast<size_t>(k));
        std::vector<float> input(activation_rows * static_cast<size_t>(k));
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = float(int((i * 37 + 13) % 257) - 128) * 0.003125f;
        ASSERT_LE(activation_rows, static_cast<size_t>(std::numeric_limits<int>::max()));
        AlignedVector<Q8_1Block> quantized(activation_rows * static_cast<size_t>(k / 32));
        quantize_activations_to_q8_1(input.data(), quantized.data(), static_cast<int>(activation_rows),
            k, k / 32, numerical_policy);
        for (int p = 0; p < projections; ++p)
        {
            const int columns = n + p * n_step;
            const int stride = columns + 7;
            sources.push_back(format.create({static_cast<size_t>(columns), static_cast<size_t>(k)}, 4242 + p));
            ASSERT_NE(sources.back(), nullptr);
            kernels.push_back(std::make_unique<CPUNativeQuantisedGemmKernel>(
                sources.back().get(), 0, -1, numerical_policy));
            ASSERT_TRUE(kernels.back()->isValid());
            const auto &packed = kernels.back()->packedWeights();
            ASSERT_EQ(packed.numerical_policy, numerical_policy);
            weight_bytes += packed.native_interleaved.size() + packed.payload.size() + packed.int8_flat.size();
            ASSERT_EQ(packed.is_superblock, format.source_is_superblock);
            if (expected_codebook) ASSERT_EQ(static_cast<int>(packed.codebook_id), *expected_codebook);
            const int policy_n = requested_policy_n > 0 ? requested_policy_n : columns;
            auto original_scope = kernels.back()->beginOutputPartitionEquivalenceScope(columns, policy_n);
            ASSERT_LE(static_cast<size_t>(maximum_rows), std::numeric_limits<size_t>::max() / static_cast<size_t>(stride));
            guarded_outputs[p].assign(static_cast<size_t>(maximum_rows) * stride, guard);
            references[p].assign(static_cast<size_t>(maximum_rows) * stride, guard);
            NativeQuantisedTestPartialStorage serial(packed, 1);
            if (exact_profile_request.empty())
                for (int row = 0; row < member_rows[p]; ++row)
                    gemv_native_quantised_preq(packed, quantized.data() + (activation_offsets[p] + static_cast<size_t>(row)) * k / 32,
                        references[p].data() + static_cast<size_t>(row) * stride, serial.span());
            descriptors.push_back({.packed = &packed,
                .output = timed_outputs[static_cast<size_t>(p) % output_root_count].data() + projection_offsets[p],
                .N = columns, .ldc = columns, .rows = member_rows[p],
                .input = quantized.data() + activation_offsets[p] * k / 32,
                .verifier_schedule = member_rows[p] == 1 || exact_target_mode ? VerifierRowsPolicy::Auto : policy,
                .purpose = purpose, .policy_n = policy_n});
            guarded_descriptors.push_back(descriptors.back());
            guarded_descriptors.back().output = guarded_outputs[p].data();
            guarded_descriptors.back().ldc = stride;
        }
        bool all_experts_aligned_64 = true;
        bool all_rows_aligned_64 = true;
        bool cacheline_multiple_columns = true;
        std::array<size_t, 2> expected_offsets{};
        for (size_t p = 0; p < descriptors.size(); ++p)
        {
            const auto &descriptor = descriptors[p];
            ASSERT_EQ(descriptor.ldc, descriptor.N);
            const size_t root = p % output_root_count;
            ASSERT_EQ(descriptor.output, timed_outputs[root].data() + expected_offsets[root]);
            const size_t expert_floats = static_cast<size_t>(descriptor.rows) * descriptor.N;
            const auto address = reinterpret_cast<std::uintptr_t>(descriptor.output);
            all_experts_aligned_64 &= address % AlignedVector<float>::ALIGNMENT == 0u;
            cacheline_multiple_columns &= (static_cast<size_t>(descriptor.N) * sizeof(float))
                % AlignedVector<float>::ALIGNMENT == 0u;
            for (int row = 0; row < descriptor.rows; ++row)
                all_rows_aligned_64 &= reinterpret_cast<std::uintptr_t>(descriptor.output
                    + static_cast<size_t>(row) * descriptor.ldc) % AlignedVector<float>::ALIGNMENT == 0u;
            std::cout << "output_projection,index=" << p << ",root=" << root
                      << ",offset_floats=" << expected_offsets[root]
                      << ",expert_floats=" << expert_floats << ",ldc=" << descriptor.ldc
                      << ",base_mod_64=" << address % AlignedVector<float>::ALIGNMENT
                      << ",row_stride_bytes=" << static_cast<size_t>(descriptor.ldc) * sizeof(float)
                      << ",guard_ldc=" << guarded_descriptors[p].ldc << '\n';
            expected_offsets[root] += expert_floats;
        }
        for (size_t root = 0; root < output_root_count; ++root)
            ASSERT_EQ(expected_offsets[root], timed_outputs[root].size());
        // The observed gate/down columns (320/2560) are cache-line multiples.
        // Generic odd-N controls retain logical strides and publish their real
        // alignment instead of padding or claiming production alignment.
        if (cacheline_multiple_columns)
        {
            ASSERT_TRUE(all_experts_aligned_64);
            ASSERT_TRUE(all_rows_aligned_64);
        }
        std::cout << "output_layout,revision=aligned_expert_major_v1,storage=AlignedVector"
                  << ",base_alignment_bytes=" << AlignedVector<float>::ALIGNMENT
                  << ",ldc=logical_n,expert_order=contiguous_within_root,tail_padding_floats=0"
                  << ",output_role=" << output_role_name << ",root_count=" << output_root_count
                  << ",guard_layout=separate_n_plus_7,live_floats=" << live_output_floats
                  << ",all_experts_aligned_64=" << all_experts_aligned_64
                  << ",all_rows_aligned_64=" << all_rows_aligned_64 << '\n';
        const auto plan = planNativeQuantisedFusedRows(descriptors, maximum_rows, omp_get_max_threads());
        const auto guarded_plan = planNativeQuantisedFusedRows(guarded_descriptors, maximum_rows, omp_get_max_threads());
        ASSERT_EQ(guarded_plan.sharedPartialFloats(), plan.sharedPartialFloats());
        for (size_t p = 0; p < descriptors.size(); ++p)
        {
            EXPECT_EQ(plan.projections[p].policy_n, descriptors[p].policy_n);
            if (expected_k_tiles > 0) ASSERT_EQ(plan.projections[p].k_tiles, expected_k_tiles);
        }
        std::cout << "preparation,numerical_policy=" << numerical_name << ",source_format=" << format.label
                  << ",source_superblock=" << format.source_is_superblock
                  << ",prepared_codebook=" << static_cast<int>(kernels[0]->packedWeights().codebook_id)
                  << ",prepared_encoding=" << static_cast<int>(kernels[0]->packedWeights().encoding)
                  << ",policy_n=" << plan.projections[0].policy_n << ",team_lifetime=" << team_name
                  << ",output_layout=aligned_expert_major_v1,output_ldc=logical_n"
                  << ",output_role=" << output_role_name << ",output_roots=" << output_root_count << '\n';
        NativeQuantisedTestPartialStorage partials(plan.sharedPartialFloats());
        const auto run = [&]() {
            if (!gemm_native_quantised_fused_verifier_rows_preq(quantized.data(), descriptors.data(),
                    partials.span(), projections, maximum_rows, k / 32))
                throw std::runtime_error("production counter replay rejected a live bundle");
        };
        const auto run_guarded = [&]() {
            if (!gemm_native_quantised_fused_verifier_rows_preq(quantized.data(), guarded_descriptors.data(),
                    partials.span(), projections, maximum_rows, k / 32))
                throw std::runtime_error("production guard replay rejected a live bundle");
        };
        const auto check_guarded = [&]() {
            for (int p = 0; p < projections; ++p)
                if (std::memcmp(guarded_outputs[p].data(), references[p].data(),
                        references[p].size() * sizeof(float)) != 0)
                    return false;
            return true;
        };
        const auto check_timed = [&]() {
            for (int p = 0; p < projections; ++p)
                for (int row = 0; row < member_rows[p]; ++row)
                    if (std::memcmp(descriptors[p].output + static_cast<size_t>(row) * descriptors[p].ldc,
                            references[p].data() + static_cast<size_t>(row) * guarded_descriptors[p].ldc,
                            static_cast<size_t>(descriptors[p].N) * sizeof(float)) != 0)
                        return false;
            return true;
        };
        NativeQuantisedFusedRowsPlan actual_plan;
        std::vector<CPUExecutionExactKey> descriptor_keys(static_cast<size_t>(projections));
        std::vector<uint32_t> baseline_descriptor_policies(static_cast<size_t>(projections));
        std::vector<uint32_t> descriptor_policies(static_cast<size_t>(projections));
        CPUExecutionExactKey target_key;
        bool exact_key_verified = false;
        const auto policy_id = [](const FusedVerifierRowsPlan &projection)
        {
            if (projection.rows == 1) return static_cast<uint32_t>(projection.effective_decode_schedule);
            if (projection.effective_verifier_schedule == VerifierRowsPolicy::Pairwise) return uint32_t{0};
            if (projection.effective_verifier_schedule == VerifierRowsPolicy::WideRows) return uint32_t{1};
            // Foreign legacy full-K families remain observable but cannot be
            // emitted as exact Pairwise/WideRows observations in this class.
            return uint32_t{0x10000} + static_cast<uint32_t>(projection.effective_verifier_schedule);
        };
        // Called once before warmup in the actual team entry. No key/vector
        // allocation, rebinding or policy mutation occurs inside timed samples.
        const auto capture_execution_context = [&]()
        {
            const int workers = nativeQuantisedInvocationWorkerCount();
            const auto baseline = planNativeQuantisedFusedRows(descriptors, maximum_rows, workers);
            for (int member = 0; member < projections; ++member)
            {
                descriptor_keys[member] = cpuExecutionExactKey(descriptors[member], baseline.projections[member],
                    workers, baseline.use_avx512, baseline.execution_exact_class);
                baseline_descriptor_policies[member] = policy_id(baseline.projections[member]);
            }
            target_key = descriptor_keys[target_descriptor];
            if (exact_target_mode && policy != VerifierRowsPolicy::Auto)
                for (int member = 0; member < projections; ++member)
                    if (descriptor_keys[member] == target_key)
                        descriptors[member].verifier_schedule = policy;
            for (int member = 0; member < projections; ++member)
                guarded_descriptors[member].verifier_schedule = descriptors[member].verifier_schedule;
            actual_plan = planNativeQuantisedFusedRows(descriptors, maximum_rows, workers);
            exact_key_verified = actual_plan.execution_exact_class == CPUExecutionExactBundleClass::FilledKPartOutputTileLocal &&
                actual_plan.partial_route == NativeQuantisedFusedPartialRoute::OutputTileLocal;
            for (int member = 0; member < projections; ++member)
            {
                const auto key = cpuExecutionExactKey(descriptors[member], actual_plan.projections[member], workers,
                    actual_plan.use_avx512, actual_plan.execution_exact_class);
                exact_key_verified = exact_key_verified && key == descriptor_keys[member];
                descriptor_policies[member] = policy_id(actual_plan.projections[member]);
                if (exact_target_mode && descriptor_keys[member] != target_key &&
                    descriptor_policies[member] != baseline_descriptor_policies[member])
                    throw std::runtime_error("Exact target candidate changed an unrelated descriptor policy");
            }
        };
        if (team_lifetime == TeamLifetime::Standalone) capture_execution_context();
        if (!exact_profile_request.empty())
        {
            ASSERT_TRUE(exact_target_mode);
            ASSERT_EQ(std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_PROFILE_PATH"), nullptr);
            const auto profile_launch = [&]()
            {
                if (team_lifetime == TeamLifetime::Retained)
                {
#pragma omp parallel
                    { run(); }
                }
                else run();
            };
            if (team_lifetime == TeamLifetime::Retained)
            {
#pragma omp parallel
                {
#pragma omp master
                    capture_execution_context();
#pragma omp barrier
                    for (int iteration = 0; iteration < 2; ++iteration) run();
                }
            }
            else for (int iteration = 0; iteration < 2; ++iteration) run();
            ASSERT_TRUE(exact_key_verified);
            PerfStatsCollector::reset();
            native_quantised_dispatch::profileExactOpenMPRegion(*exact_perf_control, exact_profile_request,
                native_quantised_dispatch::profilerEnvironment(native_quantised_dispatch::kPerfStatsPathEnvironment),
                profile_launch);
            nlohmann::json keys = nlohmann::json::array();
            for (const auto &key : descriptor_keys) keys.push_back(key.scalars());
            nlohmann::json profile{{"schema", "cpu_execution_exact_profile_v1"},
                {"request_id", exact_profile_request}, {"exact_key", target_key.scalars()},
                {"descriptor_keys", keys}, {"descriptor_policies", descriptor_policies},
                {"baseline_descriptor_policies", baseline_descriptor_policies},
                {"target_descriptor", target_descriptor}, {"candidate", verifierRowsPolicyName(policy)},
                {"exact_key_verified", exact_key_verified}, {"profiled_calls", 1},
                {"preconditioning_calls", 2}, {"team_lifetime", team_name},
                {"source_format", format.label}, {"weight_bytes", weight_bytes},
                {"purpose", purpose_name}, {"output_layout", "aligned_expert_major_v1"},
                {"activation_layout", "aligned_tightly_packed_expert_rows_v1"},
                {"member_rows", member_rows}, {"activation_row_offsets", activation_offsets}};
            if (const char *path = std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_EXACT_EVIDENCE_PATH"))
            {
                std::ofstream output(path); ASSERT_TRUE(output.is_open());
                output << profile.dump(2) << '\n'; ASSERT_TRUE(output.good());
            }
            std::cout << "execution_exact_profile," << profile.dump() << '\n';
            ASSERT_TRUE(PerfStatsCollector::flushFromEnv());
            return;
        }
        PerfStatsCollector::reset();
        run_guarded(); ASSERT_TRUE(check_guarded());
        if (team_lifetime == TeamLifetime::Standalone)
            for (int iteration = 0; iteration < warmup; ++iteration) run();
        std::cout << "format,purpose,m,n,n_step,k,projections,threads,k_tiles,partial_route,sample,bundle_us\n";
        const char *profile_path = std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_PROFILE_PATH");
        if (profile_path)
        {
            ASSERT_STREQ(profile_path, "bundle");
            const int iterations = dimension("LLAMINAR_CPU_FUSED_VERIFIER_ITERS", 1000);
            if (team_lifetime == TeamLifetime::Retained)
            {
#pragma omp parallel
                {
#pragma omp master
                    capture_execution_context();
#pragma omp barrier
                    for (int iteration = 0; iteration < warmup; ++iteration) run();
                    for (int iteration = 0; iteration < iterations; ++iteration) run();
                }
            }
            else
                for (int iteration = 0; iteration < iterations; ++iteration) run();
            ASSERT_TRUE(check_timed());
            for (auto &output : guarded_outputs) std::fill(output.begin(), output.end(), guard);
            run_guarded(); ASSERT_TRUE(check_guarded());
            std::cout << "output_layout_verified,guard_replays=2,timed_serial_bytes=1,guard_bytes=1\n";
            ASSERT_TRUE(PerfStatsCollector::flushFromEnv());
            return;
        }
        // Allocate sample storage before either timed regime. The retained
        // team executes the actual nested-safe production entry on every
        // worker; only master reads clocks at amortized sample boundaries.
        std::vector<double> elapsed_samples(static_cast<size_t>(samples));
        if (team_lifetime == TeamLifetime::Retained)
        {
            std::chrono::steady_clock::time_point sample_start;
#pragma omp parallel
            {
#pragma omp master
                capture_execution_context();
#pragma omp barrier
                for (int iteration = 0; iteration < warmup; ++iteration) run();
                for (int sample = 0; sample < samples; ++sample)
                {
#pragma omp barrier
#pragma omp master
                    sample_start = std::chrono::steady_clock::now();
#pragma omp barrier
                    for (int repeat = 0; repeat < repeats; ++repeat) run();
#pragma omp barrier
#pragma omp master
                    elapsed_samples[static_cast<size_t>(sample)] = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - sample_start).count() / repeats;
                }
            }
        }
        else
            for (int sample = 0; sample < samples; ++sample)
            {
                const auto sample_start = std::chrono::steady_clock::now();
                for (int repeat = 0; repeat < repeats; ++repeat) run();
                elapsed_samples[static_cast<size_t>(sample)] = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - sample_start).count() / repeats;
            }
        for (int sample = 0; sample < samples; ++sample)
            std::cout << format.label << ',' << purpose_name << ',' << m << ',' << n << ',' << n_step << ',' << k << ',' << projections
                      << ',' << actual_workers << ',' << actual_plan.projections[target_descriptor].k_tiles
                      << ',' << static_cast<int>(actual_plan.partial_route) << ',' << sample << ','
                      << elapsed_samples[static_cast<size_t>(sample)] << '\n';
        ASSERT_TRUE(check_timed());
        for (auto &output : guarded_outputs) std::fill(output.begin(), output.end(), guard);
        run_guarded(); ASSERT_TRUE(check_guarded());
        std::cout << "output_layout_verified,guard_replays=2,timed_serial_bytes=1,guard_bytes=1\n";
        const auto records = PerfStatsCollector::snapshot({
            "kernel.cpu_native_quantised_fused_verifier_rows_projection_launch"});
        uint64_t observations = 0;
        double value = 0.0;
        for (const auto &record : records) { observations += record.count; value += record.value; }
        if (PerfStatsCollector::isDomainEnabled("kernel"))
        {
            const auto calls = 2u + static_cast<uint64_t>(warmup)
                + static_cast<uint64_t>(samples) * repeats;
            ASSERT_LE(calls, std::numeric_limits<uint64_t>::max() / projections);
            const auto expected = calls * projections;
            EXPECT_EQ(observations, expected);
            EXPECT_DOUBLE_EQ(value, static_cast<double>(expected));
        }
        else EXPECT_TRUE(records.empty());
        std::cout << "projection_counter_records," << records.size() << ",observations," << observations
                  << ",value," << value << '\n';
        if (const char *evidence_path = std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_EXACT_EVIDENCE_PATH"))
        {
            ASSERT_TRUE(exact_target_mode);
            ASSERT_TRUE(exact_key_verified);
            ASSERT_EQ(target_key.rows, static_cast<uint32_t>(m));
            ASSERT_GT(weight_bytes, 0u);
            PerfStatsCollector::reset();
            if (team_lifetime == TeamLifetime::Retained)
            {
#pragma omp parallel
                { run(); }
            }
            else run();
            ASSERT_TRUE(check_timed());
            const auto authenticated_records = PerfStatsCollector::snapshot({
                "kernel.cpu_native_quantised_fused_verifier_rows_projection_launch"});
            nlohmann::json route_records = nlohmann::json::array();
            uint64_t authenticated_observations = 0;
            for (const auto &record : authenticated_records)
            {
                authenticated_observations += record.count;
                route_records.push_back({{"tags", record.tags}, {"count", record.count}, {"value", record.value}});
            }
            if (PerfStatsCollector::isDomainEnabled("kernel"))
                ASSERT_EQ(authenticated_observations, static_cast<uint64_t>(projections));
            std::vector<float> live_actual, live_serial;
            live_actual.reserve(live_output_floats); live_serial.reserve(live_output_floats);
            nlohmann::json keys = nlohmann::json::array();
            for (int member = 0; member < projections; ++member)
            {
                keys.push_back(descriptor_keys[member].scalars());
                for (int row = 0; row < member_rows[member]; ++row)
                {
                    const float *actual = descriptors[member].output + static_cast<size_t>(row) * descriptors[member].ldc;
                    const float *serial = references[member].data() + static_cast<size_t>(row) * guarded_descriptors[member].ldc;
                    live_actual.insert(live_actual.end(), actual, actual + descriptors[member].N);
                    live_serial.insert(live_serial.end(), serial, serial + descriptors[member].N);
                }
            }
            const auto bytes = trainer::compareFP32(live_actual, live_serial, static_cast<size_t>(n));
            ASSERT_TRUE(bytes.bitwiseEqual());
            auto sorted = elapsed_samples; std::sort(sorted.begin(), sorted.end());
            const auto timing = trainer::summarizeSortedTimingSamples(sorted);
            uint64_t execution_exact_hits = 0;
            for (const auto &record : PerfStatsCollector::snapshot({"kernel.cpu_native_quantised_execution_exact_policy_hits"}))
                execution_exact_hits += record.count;
            if (const char *expected = std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_EXPECT_EXACT_HIT"))
            {
                ASSERT_TRUE(std::string(expected) == "0" || std::string(expected) == "1");
                EXPECT_EQ(actual_plan.projections[target_descriptor].execution_exact_hit, std::string(expected) == "1");
            }
            nlohmann::json evidence{
                {"schema", "cpu_execution_exact_native_v1"},
                {"shape", std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_EXACT_SHAPE")
                    ? std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_EXACT_SHAPE") : ""},
                {"weight_bytes", weight_bytes}, {"exact_key", target_key.scalars()},
                {"descriptor_keys", keys}, {"target_descriptor", target_descriptor},
                {"descriptor_policies", descriptor_policies}, {"baseline_descriptor_policies", baseline_descriptor_policies},
                {"candidate", verifierRowsPolicyName(policy)}, {"exact_key_verified", exact_key_verified},
                {"source_format", format.label}, {"source_superblock", format.source_is_superblock},
                {"prepared_codebook", kernels[target_descriptor]->packedWeights().codebook_id},
                {"prepared_encoding", static_cast<int>(kernels[target_descriptor]->packedWeights().encoding)},
                {"numerical_policy", numerical_name}, {"purpose", purpose_name},
                {"partial_route", "OutputTileLocal"}, {"partial_route_id", static_cast<int>(actual_plan.partial_route)},
                {"bundle_class", static_cast<uint32_t>(actual_plan.execution_exact_class)},
                {"team_lifetime", team_name}, {"output_layout", "aligned_expert_major_v1"},
                {"activation_layout", "aligned_tightly_packed_expert_rows_v1"},
                {"expert_rows", expert_rows}, {"member_rows", member_rows}, {"activation_row_offsets", activation_offsets},
                {"output_role", output_role_name}, {"raw_samples_us", elapsed_samples}, {"sorted_samples_us", sorted},
                {"timing", {{"sample_count", samples}, {"repeat_count", repeats}, {"warmup_count", warmup},
                    {"min_us", timing.min}, {"median_us", timing.median}, {"p95_us", timing.p95},
                    {"mad_us", timing.mad}, {"cv", timing.cv}, {"sorted_native_digest", timing.digest}}},
                {"correctness", {{"bitwise_equal", bytes.bitwiseEqual()}, {"mismatch_count", bytes.mismatch_count},
                    {"nonfinite_count", bytes.nonfinite_count}, {"actual_digest", bytes.actual_digest},
                    {"expected_digest", bytes.expected_digest}, {"max_abs", bytes.max_abs},
                    {"relative_l2", bytes.relative_l2}, {"cosine", bytes.cosine}, {"symmetric_kld", bytes.symmetric_kld},
                    {"guard_replays", 2}, {"repeated_replay", samples * repeats > 1}}},
                {"counters", {{"enabled", PerfStatsCollector::isDomainEnabled("kernel")},
                    {"projection_observations", observations}, {"expected_projection_observations",
                        (2u + static_cast<uint64_t>(warmup) + static_cast<uint64_t>(samples) * repeats) * projections},
                    {"execution_exact_hits", execution_exact_hits},
                    {"authenticated_projection_observations", authenticated_observations},
                    {"route_records", route_records}, {"authentication_calls", 1},
                    {"authentication_outside_timing", true}}},
                {"placement", {{"workers", actual_workers}, {"admitted_physical_cores", allowed_physical_cores.size()},
                    {"disjoint_single_core_masks", disjoint_single_core_masks}}}};
            std::ofstream output(evidence_path, std::ios::out | std::ios::trunc);
            ASSERT_TRUE(output.is_open()); output << evidence.dump(2) << '\n';
            ASSERT_TRUE(output.good());
        }
        ASSERT_TRUE(PerfStatsCollector::flushFromEnv());
    }

    /**
     * @test Many narrow members (routed-expert gate/up) pooled into one bundle.
     *
     * Qwen3.6-35B-A3B decode fuses 16 routed gate/up projections of N=512,
     * K=2048 reusing one Q8 row. The bundle must equal serial rows byte for
     * byte and is timed against the same members executed one by one.
     */
    TEST_P(CPUFusedVerifier, ManyNarrowMembersBundleVersusStandalone)
    {
        const auto &format = GetParam();
        const int members = dimension("LLAMINAR_CPU_FUSED_MEMBERS", 16);
        const int m = dimension("LLAMINAR_CPU_FUSED_MEMBERS_M", 1);
        const int n = dimension("LLAMINAR_CPU_FUSED_MEMBERS_N", 512);
        const int k = dimension("LLAMINAR_CPU_FUSED_MEMBERS_K", 2048);
        ASSERT_GE(members, 1);
        ASSERT_LE(members, int(NativeQuantisedFusedRowsPlan::capacity));
        ASSERT_EQ(k % 256, 0);
        std::vector<std::unique_ptr<TensorBase>> sources;
        std::vector<std::unique_ptr<CPUNativeQuantisedGemmKernel>> kernels;
        std::vector<NativeQuantisedTestPartialStorage> partial;
        for (int member = 0; member < members; ++member)
        {
            sources.push_back(format.create({size_t(n), size_t(k)}, 9000 + member));
            kernels.push_back(std::make_unique<CPUNativeQuantisedGemmKernel>(sources.back().get()));
            ASSERT_TRUE(kernels.back()->isValid());
            partial.emplace_back(kernels.back()->packedWeights(), m);
        }
        std::mt19937 random(77);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        std::vector<float> input(size_t(m) * k);
        for (auto &value : input) value = distribution(random);
        std::vector<Q8_1Block> quantized(size_t(m) * k / 32);
        quantize_activations_to_q8_1(input.data(), quantized.data(), m, k, k / 32);
        std::vector<std::vector<float>> fused(members), separate(members), serial(members);
        std::vector<FusedVerifierRowsDesc> descriptors(members);
        for (int member = 0; member < members; ++member)
        {
            const auto &weights = kernels[member]->packedWeights();
            fused[member].resize(size_t(m) * n);
            separate[member].resize(size_t(m) * n);
            serial[member].resize(size_t(m) * n);
            descriptors[member] = {.packed = &weights, .output = fused[member].data(), .bias = nullptr,
                .N = n, .ldc = n};
            for (int row = 0; row < m; ++row)
                gemv_native_quantised_preq(weights, quantized.data() + row * (k / 32),
                    serial[member].data() + row * n, partial[member].span());
        }
        NativeQuantisedTestPartialStorage fused_storage(descriptors.data(), members, m);
        const auto run = [&](bool bundle)
        {
            if (bundle)
            {
                if (!gemm_native_quantised_fused_verifier_rows_preq(quantized.data(), descriptors.data(),
                        fused_storage.span(), members, m, k / 32))
                    throw std::runtime_error("production fused bundle rejected its descriptors");
            }
            else
                for (int member = 0; member < members; ++member)
                    gemm_native_quantised_preq_decode_equivalent_rows(kernels[member]->packedWeights(),
                        quantized.data(), separate[member].data(), partial[member].span(), m, n);
        };
        run(false);
        run(true);
        for (int member = 0; member < members; ++member)
        {
            ASSERT_EQ(std::memcmp(serial[member].data(), fused[member].data(), serial[member].size() * sizeof(float)), 0);
            ASSERT_EQ(std::memcmp(serial[member].data(), separate[member].data(), serial[member].size() * sizeof(float)), 0);
        }
        std::cout << "format,members,m,n,k,threads,sample,bundle_us,separate_us\n";
        constexpr int repetitions = 32;
        for (int sample = -2; sample < 12; ++sample)
        {
            std::array<double, 2> times{};
            for (int order = 0; order < 2; ++order)
            {
                const bool bundle = ((sample + 2 + order) % 2) == 0;
                const auto start = std::chrono::steady_clock::now();
                for (int repeat = 0; repeat < repetitions; ++repeat) run(bundle);
                times[bundle ? 0 : 1] = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - start).count() / repetitions;
            }
            if (sample >= 0)
                std::cout << format.label << ',' << members << ',' << m << ',' << n << ',' << k << ','
                          << omp_get_max_threads() << ',' << sample << ',' << times[0] << ',' << times[1] << '\n';
        }
    }

    INSTANTIATE_TEST_SUITE_P(AllFormats, CPUFusedVerifier,
        ::testing::ValuesIn(quantizedVerifierFormats()),
        [](const ::testing::TestParamInfo<QuantizedVerifierFormatCase> &info)
        { return std::string(info.param.label); });

    /**
     * @test Compare existing native-Q6 serial register tiles without changing dispatch.
     *
     * Each candidate owns the same complete 64-column chunk and visits every
     * K block in order. Only the number of column vectors kept in registers
     * together differs. The current four-vector primitive is the byte oracle;
     * this diagnostic does not install a generated policy or change packing.
     */
    TEST(CPUQ6SerialTile, RegisterGeometry)
    {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        const int n = dimension("LLAMINAR_CPU_FUSED_VERIFIER_N_A", 2048);
        const int k = dimension("LLAMINAR_CPU_FUSED_VERIFIER_K", 2048);
        ASSERT_EQ(k % 256, 0);
        auto source = TestTensorFactory::createQ6_KRandom(
            {static_cast<size_t>(n), static_cast<size_t>(k)}, 4251);
        CPUNativeQuantisedGemmKernel kernel(source.get());
        ASSERT_TRUE(kernel.isValid());
        const auto &packed = kernel.packedWeights();
        auto input = TestTensorFactory::createFP32Random(
            {1, static_cast<size_t>(k)}, -1.0f, 1.0f, 4252);
        std::vector<Q8_1Block> quantized(packed.blocks_per_row);
        quantize_activations_to_q8_1(input->data(), quantized.data(),
            1, k, packed.blocks_per_row);
        const std::array<int, 3> spans = {1, 2, 4};
        std::array<std::vector<float>, 3> outputs;
        for (auto &output : outputs)
            output.resize(packed.N_padded);

        // Persistent output storage and the production worksharing scope keep
        // allocation, packing and team setup identical across the candidates.
        const auto run = [&](int candidate)
        {
            auto compute = [&]()
            {
#pragma omp for schedule(static)
                for (int chunk = 0; chunk < packed.N_padded / 64; ++chunk)
                {
                    const std::array<const Q8_1Block *, 1> activation = {quantized.data()};
                    const std::array<float *, 1> output = {outputs[candidate].data() + chunk * 64};
                    switch (spans[candidate])
                    {
                    case 1:
                        q6k_detail::rowsChunkAVX512<1, 1>(packed, activation,
                            output, chunk, 0, packed.blocks_per_row, false);
                        break;
                    case 2:
                        q6k_detail::rowsChunkAVX512<1, 2>(packed, activation,
                            output, chunk, 0, packed.blocks_per_row, false);
                        break;
                    case 4:
                        q6k_detail::rowsChunkAVX512<1, 4>(packed, activation,
                            output, chunk, 0, packed.blocks_per_row, false);
                        break;
                    }
                }
            };
            OMP_WORKSHARE_REGION(compute);
        };
        for (int candidate = 0; candidate < 3; ++candidate)
            run(candidate);
        for (int candidate = 0; candidate < 2; ++candidate)
            ASSERT_EQ(std::memcmp(outputs[candidate].data(), outputs[2].data(),
                static_cast<size_t>(packed.N_padded) * sizeof(float)), 0);

        if (std::getenv("LLAMINAR_CPU_Q6_SERIAL_PROFILE_ZSPAN"))
        {
            const int span = dimension("LLAMINAR_CPU_Q6_SERIAL_PROFILE_ZSPAN", 4);
            const auto found = std::find(spans.begin(), spans.end(), span);
            ASSERT_NE(found, spans.end());
            const int candidate = static_cast<int>(found - spans.begin());
            const int iterations = dimension("LLAMINAR_CPU_FUSED_VERIFIER_ITERS", 1000);
            for (int iteration = 0; iteration < iterations; ++iteration)
                run(candidate);
            return;
        }

        // Give every tile an equal sustained warmup; tiny projections otherwise
        // finish their first samples while the CPU is still changing frequency.
        for (int candidate = 0; candidate < 3; ++candidate)
        {
            const auto until = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(250);
            do { run(candidate); } while (std::chrono::steady_clock::now() < until);
        }
        std::cout << "format,m,n,k,threads,sample,zspan,microseconds\n";
        for (int sample = 0; sample < 12; ++sample)
            for (int order = 0; order < 3; ++order)
            {
                const int candidate = (sample + order) % 3;
                constexpr int repetitions = 16;
                const auto start = std::chrono::steady_clock::now();
                for (int repeat = 0; repeat < repetitions; ++repeat)
                    run(candidate);
                const double elapsed = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - start).count() / repetitions;
                std::cout << "Q6_K,1," << n << ',' << k << ',' << omp_get_max_threads()
                          << ',' << sample << ',' << spans[candidate] << ',' << elapsed << '\n';
            }
#else
        GTEST_SKIP() << "The native AVX512 register-tile diagnostic requires an AVX512 build";
#endif
    }

    /** Narrow real-model projections retain FP32 activations for every weight dtype. */
    class CPUFloatingVerifier : public ::testing::TestWithParam<TensorType> {};

    /** Time the production narrow grouped entry point after exact M=1 validation. */
    TEST_P(CPUFloatingVerifier, NarrowGroupedRows)
    {
        const int m = dimension("LLAMINAR_CPU_FUSED_VERIFIER_M", 16);
        const int n = dimension("LLAMINAR_CPU_FUSED_VERIFIER_N_A", 48);
        const int k = dimension("LLAMINAR_CPU_FUSED_VERIFIER_K", 5120);
        std::mt19937 random(18448);
        std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
        std::vector<float> input(static_cast<size_t>(m) * k);
        std::vector<float> weights(static_cast<size_t>(n) * k);
        for (float &value : input)
            value = distribution(random);
        for (float &value : weights)
            value = distribution(random);
        std::vector<uint16_t> stored(weights.size());
        const TensorType dtype = GetParam();
        const char *label = dtype == TensorType::FP32 ? "FP32"
            : dtype == TensorType::FP16 ? "FP16" : "BF16";
        for (size_t i = 0; i < weights.size(); ++i)
            stored[i] = dtype == TensorType::BF16
                ? simd::fp32_to_bf16(weights[i]) : fp32_to_fp16(weights[i]);
        std::vector<float> grouped(static_cast<size_t>(m) * n);
        std::vector<float> serial(grouped.size());
        const auto run = [&](const float *a, float *c, int rows)
        {
            switch (dtype)
            {
            case TensorType::FP32:
                return gemm::run_fp32_skinny_matmul(a, weights.data(), c, rows, n, k, true);
            case TensorType::FP16:
                return gemm::run_fp32xfp16_skinny_matmul(a, stored.data(), c, rows, n, k, true);
            case TensorType::BF16:
                return gemm::run_fp32xbf16_skinny_matmul(a, stored.data(), c, rows, n, k, true);
            default:
                throw std::invalid_argument("unsupported floating verifier benchmark dtype");
            }
        };
        for (int row = 0; row < m; ++row)
            ASSERT_TRUE(run(input.data() + row * k, serial.data() + row * n, 1));
        ASSERT_TRUE(run(input.data(), grouped.data(), m));
        ASSERT_EQ(std::memcmp(grouped.data(), serial.data(), serial.size() * sizeof(float)), 0);
        std::cout << "floating_weight,m,n,k,threads,sample,grouped_us\n";
        constexpr int repetitions = 64;
        for (int sample = -2; sample < 12; ++sample)
        {
            const auto start = std::chrono::steady_clock::now();
            bool ok = true;
            for (int repeat = 0; repeat < repetitions; ++repeat)
                ok = run(input.data(), grouped.data(), m) && ok;
            const double elapsed = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - start).count() / repetitions;
            ASSERT_TRUE(ok);
            if (sample >= 0)
                std::cout << label << ',' << m << ',' << n << ',' << k << ','
                          << omp_get_max_threads() << ',' << sample << ',' << elapsed << '\n';
        }
    }

    INSTANTIATE_TEST_SUITE_P(FloatingFormats, CPUFloatingVerifier,
        ::testing::Values(TensorType::FP32, TensorType::FP16, TensorType::BF16),
        [](const ::testing::TestParamInfo<TensorType> &info)
        { return info.param == TensorType::FP32 ? "FP32"
            : info.param == TensorType::FP16 ? "FP16" : "BF16"; });
}
