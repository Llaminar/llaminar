/**
 * @file Test__W4A4CPUModelGraph.cpp
 * @brief Complete CPU Qwen35 graphs retain native FFNs across source/request retirement.
 *
 * The real parser, planner, ModelContext, prepared store, graph builder, executor,
 * prefix cache and sampler consume a tiny generated all-attention GGUF. Every
 * dense FFN uses W4A4/H, including the target's W4A4 gate/up plus W4A4H down;
 * attention, embeddings and the head remain FP32.
 * Native producer counters authenticate execution without replacing any graph
 * stage or enabling diagnostic kernel determinism. Both greedy and stochastic
 * requests retain exact logits/tokens across reset and full prefix restoration.
 *
 * These are synthetic graph/lifecycle contracts, not recorded numerical replay,
 * real-model HF parity, GPU capture evidence or MTP basis qualification. The
 * independent immutable upstream recordings remain the arithmetic oracle.
 */
#include "../../../utils/PlanningGGUFFixture.h"
#include "../../../utils/ServingGGUFFixturePayload.h"
#include "backends/BackendManager.h"
#include "config/OrchestrationConfigParser.h"
#include "config/OrchestrationStartupPolicy.h"
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
#include "execution/runner/OrchestrationRunner.h"
#include "interfaces/IModelContext.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

#include <gtest/gtest.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        /** @brief Restore process-wide diagnostics, bucket policy and actual OpenMP teams. */
        class CPUModelPolicy final
        {
        public:
            /** @brief Install passive counters and ordinary optimized dispatch for this fixture. */
            CPUModelPolicy() : threads_(omp_get_max_threads()), dynamic_(omp_get_dynamic())
            {
                for (const char *key : {"LLAMINAR_PERF_STATS_JSON", "LLAMINAR_PERF_STATS_FILTER",
                    "LLAMINAR_PERF_STATS_CPU_STAGE_TIMING", "LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES",
                    "LLAMINAR_DETERMINISTIC"})
                {
                    const char *value = std::getenv(key);
                    previous_.emplace_back(key, value ? std::optional<std::string>{value} : std::nullopt);
                }
                try
                {
                    if (setenv("LLAMINAR_PERF_STATS_JSON", "1", 1) != 0 ||
                        setenv("LLAMINAR_PERF_STATS_FILTER", "kernel.cpu_w4a4_", 1) != 0 ||
                        setenv("LLAMINAR_PERF_STATS_CPU_STAGE_TIMING", "0", 1) != 0 ||
                        unsetenv("LLAMINAR_DETERMINISTIC") != 0)
                        throw std::runtime_error("Cannot install native CPU graph test policy");
                    omp_set_dynamic(0);
                    omp_set_num_threads(1);
                    mutableDebugEnv().reload();
                    PerfStatsCollector::reloadConfigurationFromEnvironment();
                    PerfStatsCollector::reset();
                }
                catch (...)
                {
                    restore();
                    throw;
                }
            }
            /** @brief Restore only after all inference/cache producers have retired. */
            ~CPUModelPolicy() { restore(); }
            CPUModelPolicy(const CPUModelPolicy &) = delete;
            CPUModelPolicy &operator=(const CPUModelPolicy &) = delete;
        private:
            /** @brief Retire this evidence epoch and restore the exact caller environment. */
            void restore()
            {
                PerfStatsCollector::reset();
                for (const auto &[key, value] : previous_)
                    if (value) setenv(key.c_str(), value->c_str(), 1);
                    else unsetenv(key.c_str());
                omp_set_num_threads(threads_);
                omp_set_dynamic(dynamic_);
                mutableDebugEnv().reload();
                PerfStatsCollector::reloadConfigurationFromEnvironment();
            }
            int threads_, dynamic_;
            std::vector<std::pair<std::string, std::optional<std::string>>> previous_;
        };

        /** @brief Unique disk-prefix root; never touches the caller's durable cache. */
        class PrefixDirectory final
        {
        public:
            /** @brief Create the only directory this fixture is permitted to remove. */
            PrefixDirectory()
            {
                auto pattern = (std::filesystem::temp_directory_path() / "llaminar-w4a4-prefix-XXXXXX").string();
                const char *created = ::mkdtemp(pattern.data());
                if (!created) throw std::runtime_error("Cannot create native graph prefix directory");
                path_ = created;
            }
            /** @brief Remove the private directory after its cache/runner have retired. */
            ~PrefixDirectory()
            {
                std::error_code ignored;
                std::filesystem::remove_all(path_, ignored);
            }
            PrefixDirectory(const PrefixDirectory &) = delete;
            PrefixDirectory &operator=(const PrefixDirectory &) = delete;
            /** @return Original private cache root, stable through runner retirement. */
            const std::string &path() const noexcept { return path_; }
        private:
            std::string path_;
        };

        /** @return Actual native projection submissions, never inferred from token counts. */
        double nativeCalls()
        {
            double calls = 0;
            for (const auto &record : PerfStatsCollector::snapshot())
                if (record.domain == "kernel" && record.device == "cpu" &&
                    (record.name == "cpu_w4a4_projection_calls" ||
                     record.name == "cpu_w4a4_fused_projection_calls" ||
                     record.name == "cpu_w4a4_swiglu_projection_calls"))
                    calls += record.value;
            return calls;
        }

        /**
         * @brief Observe existing CPU terminal logits without requesting stage snapshots.
         * @param runner Initialized CPU runner whose public forward has returned.
         * @return Exact finite nondegenerate logits as IEEE-754 words.
         * @throws std::runtime_error If this fixture did not produce its complete CPU head.
         */
        std::vector<std::uint32_t> logits(const OrchestrationRunner &runner)
        {
            const auto *values = runner.lastLogits();
            if (runner.primaryDeviceId() != DeviceId::cpu() || runner.vocabSize() != 320 || !values)
                throw std::runtime_error("Native CPU graph has no complete terminal logits");
            std::vector<std::uint32_t> result(320);
            for (size_t i = 0; i < result.size(); ++i)
            {
                if (!std::isfinite(values[i]))
                    throw std::runtime_error("Native CPU graph produced nonfinite terminal logits");
                result[i] = std::bit_cast<std::uint32_t>(values[i]);
            }
            if (std::all_of(result.begin(), result.end(), [&](auto value) { return value == result.front(); }))
                throw std::runtime_error("Native CPU graph fixture produced a degenerate terminal head");
            return result;
        }

        /** @brief Explicit ordinary request cell; MTP remains disabled rather than bypassed. */
        struct GraphCase
        {
            GGUFTensorType gate_up_format;
            GGUFTensorType down_format;
            bool stochastic;
            const char *name;
        };
        /** @brief Real graph execution, separately selected by source format and sampler. */
        class W4A4CPUModelGraphLifecycle : public ::testing::TestWithParam<GraphCase> {};

        TEST_P(W4A4CPUModelGraphLifecycle, RetainsNativeFFNsThroughPrefixReuseAndRequestReset)
        {
            CPUModelPolicy policy;
            PrefixDirectory prefix;
            if (!hasCPUBackend()) initCPUBackend(-1);
            ASSERT_NE(getCPUBackend(), nullptr);
            const auto cell = GetParam();
            PlanningGGUFFixture fixture(false, false, cell.gate_up_format, 512, std::nullopt, 8,
                PlanningGGUFFixture::Geometry::FullAttentionGraph, 2, 8, cell.down_format);
            populateServingGGUFFixture(fixture);
            std::vector<std::string> arguments{"native-cpu-graph", "-m", fixture.path(),
                "--device", "cpu:0", "--mpi-procs", "1", "--threads", "1",
                "--context-length", "64", "--no-mtp", "--activation-precision", "fp32",
                "--kv-cache-precision", "fp16"};
            std::vector<char *> argv;
            for (auto &argument : arguments) argv.push_back(argument.data());
            auto config = OrchestrationConfigParser{}.parseArgs(static_cast<int>(argv.size()), argv.data());
            config.prefill_max_bucket_size = 8;
            // Real enabled tiers, bounded to this tiny graph. No fabricated RAM
            // capacity or blanket prefix-cache disablement admits the fixture.
            constexpr size_t prefix_bytes = 8u * 1024u * 1024u;
            config.prefix_cache.block_size = 4;
            config.prefix_cache.ram_budget_bytes = prefix_bytes;
            config.prefix_cache.device_budget_bytes = prefix_bytes;
            config.prefix_cache.disk_budget_bytes = prefix_bytes;
            config.prefix_cache.disk_dir = prefix.path();
            ASSERT_TRUE(config.prefix_cache.enabled);
            ASSERT_FALSE(config.deterministic);
            publishOrchestrationStartupPolicy(config);
            ASSERT_FALSE(PerfStatsCollector::cpuStageTimingEnabled());

            std::shared_ptr<PhysicalMemoryAuthority> memory;
            {
                OrchestrationRunner runner(config, createExecutionPlanBuilder());
                ASSERT_TRUE(runner.initialize()) << runner.lastError();
                ASSERT_EQ(runner.primaryDeviceId(), DeviceId::cpu());
                ASSERT_EQ(runner.architecture(), "qwen35");
                ASSERT_FALSE(runner.executionPlan().runtime.mtp.enabled);
                const auto *context = runner.modelContextForDiagnostics();
                ASSERT_NE(context, nullptr);
                const auto *directory = context->ggufDirectory();
                ASSERT_NE(directory, nullptr);
                for (int layer = 0; layer < 2; ++layer)
                    for (const auto *role : {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"})
                    {
                        const auto *tensor = directory->findTensor("blk." + std::to_string(layer) + "." + role);
                        ASSERT_NE(tensor, nullptr);
                        EXPECT_EQ(tensor->type, std::string_view(role) == "ffn_down.weight"
                            ? cell.down_format : cell.gate_up_format);
                    }
                // Retain only the canonical ledger, not an extra model owner
                // that could conceal leaked prepared storage during shutdown.
                auto contract = runner.modelContextReuseContract();
                ASSERT_TRUE(contract.has_value());
                memory = contract->physical_memory_authority;
                ASSERT_NE(memory, nullptr);
                contract.reset();
                constexpr auto owner = PhysicalMemoryOwner::PrimaryModelWeights;
                constexpr auto fresh = PhysicalMemoryMaterializationKind::NewAllocation;
                constexpr auto adopted = PhysicalMemoryMaterializationKind::AdoptedResident;
                const auto new_weight_bytes = memory->claimedBytes(DeviceId::cpu(), owner, fresh);
                const auto adopted_weight_bytes = memory->claimedBytes(DeviceId::cpu(), owner, adopted);
                ASSERT_GT(new_weight_bytes + adopted_weight_bytes, 0u);
                ASSERT_TRUE(std::filesystem::remove(fixture.path()));

                SamplingParams sampling;
                sampling.temperature = cell.stochastic ? 0.8F : 0.0F;
                sampling.seed = 123;
                runner.setSamplingParams(sampling);
                runner.setStopTokens({});
                const std::array<std::vector<int32_t>, 4> prompts{{
                    {11, 12, 13, 14, 15, 16, 17, 18}, {31}, {41, 42, 43},
                    {11, 12, 13, 14, 51, 52, 53, 54}}};
                struct Observation
                {
                    std::vector<int32_t> tokens;
                    std::vector<std::vector<std::uint32_t>> terminal_words;
                };
                std::array<Observation, 4> first;
                for (int round = 0; round < 2; ++round)
                {
                    for (size_t request = 0; request < prompts.size(); ++request)
                    {
                        SCOPED_TRACE(round);
                        SCOPED_TRACE(request);
                        runner.clearCache();
                        const auto before_prefill = nativeCalls();
                        ASSERT_TRUE(runner.prefill(prompts[request])) << runner.lastError();
                        const auto summary = runner.requestRuntimeSummary();
                        EXPECT_FALSE(summary.mtp_request.enabled);
                        EXPECT_FALSE(summary.mtp_request.bypassed);
                        EXPECT_TRUE(summary.prefix_request.enabled);
                        EXPECT_FALSE(summary.prefix_request.bypassed);
                        if (round == 0 && request == 0)
                        {
                            EXPECT_FALSE(summary.prefix_request.hit);
                            EXPECT_GT(nativeCalls(), before_prefill);
                        }
                        if (round == 0 && request == 3)
                        {
                            EXPECT_TRUE(summary.prefix_request.partial_hit);
                            EXPECT_EQ(summary.prefix_request.matched_tokens, 4);
                            EXPECT_GT(nativeCalls(), before_prefill);
                        }
                        if (round == 1 && (request == 0 || request == 3))
                        {
                            EXPECT_TRUE(summary.prefix_request.hit);
                            EXPECT_EQ(summary.prefix_request.matched_tokens, 8);
                            EXPECT_TRUE(summary.prefix_request.terminal_logits_restored);
                        }
                        Observation observed;
                        observed.terminal_words.push_back(logits(runner));
                        for (int step = 0; step < 3; ++step)
                        {
                            SCOPED_TRACE(step);
                            const auto before_decode = nativeCalls();
                            runner.setDecodeStepTokenBudget(1);
                            const auto result = runner.decodeStep();
                            ASSERT_TRUE(result.success()) << result.error;
                            ASSERT_EQ(result.tokens.size(), 1u);
                            ASSERT_FALSE(result.is_complete);
                            EXPECT_GE(result.tokens.front(), 0);
                            EXPECT_LT(result.tokens.front(), 320);
                            // The first sample consumes existing prefill logits;
                            // subsequent samples require a real history-bearing
                            // model row, including native FFNs on both layers.
                            if (step > 0) EXPECT_GT(nativeCalls(), before_decode);
                            ASSERT_TRUE(runner.maybeApplyMoERebalance(result.tokens.size())) << runner.lastError();
                            observed.tokens.push_back(result.tokens.front());
                            observed.terminal_words.push_back(logits(runner));
                        }
                        if (round == 0) first[request] = std::move(observed);
                        else
                        {
                            EXPECT_EQ(observed.tokens, first[request].tokens);
                            EXPECT_EQ(observed.terminal_words, first[request].terminal_words);
                        }
                        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), owner, fresh), new_weight_bytes);
                        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), owner, adopted), adopted_weight_bytes);
                    }
                    runner.clearCache();
                    const auto before_empty = nativeCalls();
                    EXPECT_FALSE(runner.prefill({}));
                    EXPECT_EQ(nativeCalls(), before_empty);
                }
                runner.shutdown();
            }
            ASSERT_NE(memory, nullptr);
            for (const auto kind : {PhysicalMemoryMaterializationKind::NewAllocation,
                                   PhysicalMemoryMaterializationKind::AdoptedResident})
                EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::PrimaryModelWeights, kind), 0u);
        }

        INSTANTIATE_TEST_SUITE_P(Formats, W4A4CPUModelGraphLifecycle, ::testing::Values(
            GraphCase{GGUFTensorType::W4A4, GGUFTensorType::W4A4, false, "W4A4_Greedy"},
            GraphCase{GGUFTensorType::W4A4, GGUFTensorType::W4A4, true, "W4A4_Stochastic"},
            GraphCase{GGUFTensorType::W4A4H, GGUFTensorType::W4A4H, false, "W4A4H_Greedy"},
            GraphCase{GGUFTensorType::W4A4H, GGUFTensorType::W4A4H, true, "W4A4H_Stochastic"},
            GraphCase{GGUFTensorType::W4A4, GGUFTensorType::W4A4H, false, "TargetMixed_Greedy"},
            GraphCase{GGUFTensorType::W4A4, GGUFTensorType::W4A4H, true, "TargetMixed_Stochastic"}),
            [](const auto &info) { return info.param.name; });
    }
}
