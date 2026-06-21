#include <gtest/gtest.h>
#include <mpi.h>

#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"
#include "execution/runner/MTPVerifierForwardExecutor.h"
#include "integration/parity/qwen36/Qwen36DenseParityTestBase.h"
#include "utils/PerfStatsCollector.h"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

/**
 * @file Perf__DenseVerifierRows.cpp
 * @brief Qwen3.6 dense MTP verifier-row correctness and replay speed proof.
 *
 * Phase 9.8 needs two different facts before a grouped verifier can be called
 * economical:
 *
 * 1. Grouped all-position verifier rows must be numerically equivalent to
 *    serial decode rows.  This harness checks cosine, relative L2, symmetric
 *    KL, and sampled-token equality through the same helpers as the PyTorch
 *    parity suite.
 * 2. Once graph capture and row-indexed verifier modes are warm, one grouped
 *    verifier replay must be faster than the decode-equivalent serial replay
 *    it replaces.  Setup and full-logit diagnostic copies stay outside the
 *    timing loop so the measurement represents the hot inference path.
 */

using namespace llaminar2;
using namespace llaminar2::test::parity::qwen36;

namespace
{
    struct DenseVerifierEconomyRow
    {
        std::string backend;
        int rows = 0;
        double grouped_ms = 0.0;
        double serial_ms = 0.0;
        double grouped_restore_ms = 0.0;
        double serial_restore_ms = 0.0;
        double grouped_setup_ms = 0.0;
        double serial_setup_ms = 0.0;
        double grouped_forward_ms = 0.0;
        double serial_forward_ms = 0.0;
        double grouped_sample_ms = 0.0;
        double serial_sample_ms = 0.0;
        DenseVerifierLogitMetrics worst_metrics;
        int32_t grouped_ready_token = -1;
        int32_t serial_ready_token = -1;
    };

    /**
     * @brief Replay-local timing buckets for one verifier execution.
     *
     * The headline replay time includes restoring the shared baseline state.
     * That is the correct end-to-end acceptance number, but it can hide whether
     * the grouped verifier work itself is economical.  These buckets let the
     * perf test print both views without changing production instrumentation.
     */
    struct DenseVerifierReplayTiming
    {
        double restore_ms = 0.0;
        double setup_ms = 0.0;
        double forward_ms = 0.0;
        double sample_ms = 0.0;
        double logits_copy_ms = 0.0;

        void add(const DenseVerifierReplayTiming &other)
        {
            restore_ms += other.restore_ms;
            setup_ms += other.setup_ms;
            forward_ms += other.forward_ms;
            sample_ms += other.sample_ms;
            logits_copy_ms += other.logits_copy_ms;
        }

        void divide(double divisor)
        {
            if (divisor <= 0.0)
            {
                return;
            }
            restore_ms /= divisor;
            setup_ms /= divisor;
            forward_ms /= divisor;
            sample_ms /= divisor;
            logits_copy_ms /= divisor;
        }
    };

    struct DenseVerifierEconomyFixture
    {
        std::shared_ptr<ModelContext> model_ctx;
        std::unique_ptr<IInferenceRunner> runner;
        PrefixStateSnapshot verifier_base;
        std::vector<int32_t> verifier_tokens;
        std::vector<int32_t> ready_tokens_by_count;
        int vocab_size = 0;
    };

    int envInt(const char *name, int fallback)
    {
        const char *raw = std::getenv(name);
        if (!raw || !*raw)
        {
            return fallback;
        }
        char *end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end == raw || parsed <= 0)
        {
            return fallback;
        }
        return static_cast<int>(parsed);
    }

    bool envBool(const char *name)
    {
        const char *raw = std::getenv(name);
        if (!raw || !*raw)
        {
            return false;
        }
        std::string value(raw);
        std::transform(
            value.begin(),
            value.end(),
            value.begin(),
            [](unsigned char c)
            {
                return static_cast<char>(std::tolower(c));
            });
        return value == "1" || value == "true" ||
               value == "yes" || value == "on";
    }

    PerfStatsCollector::Tags denseVerifierPerfTags(
        const std::string &backend,
        const char *path,
        int rows)
    {
        return {
            {"backend", backend},
            {"path", path},
            {"rows", std::to_string(rows)},
        };
    }

    template <typename Fn>
    bool measureReplayBucket(double *bucket_ms, Fn &&fn)
    {
        const auto start = std::chrono::steady_clock::now();
        const bool ok = fn();
        const auto end = std::chrono::steady_clock::now();
        if (bucket_ms)
        {
            *bucket_ms += std::chrono::duration<double, std::milli>(
                              end - start)
                              .count();
        }
        return ok;
    }

    DensePrefixRestoreParityCase denseSingleDeviceCase(
        const std::string &backend_name,
        GlobalDeviceAddress device)
    {
        auto test_case = qwen36DensePrefixParityCase(
            "Qwen3.6 dense " + backend_name + " verifier economy",
            DensePrefixParityTopology::SingleDevice);
        test_case.devices = {device};
        test_case.required_cuda_devices =
            device.device_type == DeviceType::CUDA ? 1 : 0;
        test_case.required_rocm_devices =
            device.device_type == DeviceType::ROCm ? 1 : 0;
        return test_case;
    }

    int32_t sampleCurrentToken(IInferenceRunner &runner, const char *label)
    {
        const int32_t sampled = runner.sampleGreedyOnDevice();
        if (sampled >= 0)
        {
            return sampled;
        }
        const float *logits = runner.logits();
        ADD_FAILURE() << "sampleGreedyOnDevice failed after " << label
                      << "; falling back to host-visible logits";
        return denseArgmaxToken(logits, runner.vocab_size());
    }

    /**
     * @brief Build the common prompt/setup state used by both replay modes.
     *
     * The first two decode tokens establish the same shifted-MTP condition
     * window used by the dense grouped parity tests.  The verifier tokens are
     * then taken from the serial oracle so every row should accept and expose a
     * deterministic ready token.
     */
    DenseVerifierEconomyFixture prepareDenseVerifierEconomyFixture(
        const DensePrefixRestoreParityCase &test_case,
        int max_verifier_rows)
    {
        DenseVerifierEconomyFixture fixture;

        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(
            test_case,
            &model_path,
            &prompt_tokens,
            &expected_tokens);
        EXPECT_GE(expected_tokens.size(), 2u)
            << "dense verifier economy fixture needs two setup tokens";
        if (expected_tokens.size() < 2u)
        {
            return fixture;
        }

        const DeviceId device = test_case.devices.empty()
                                    ? DeviceId::cpu()
                                    : test_case.devices.front().toLocalDeviceId();

        DeviceManager::instance().initialize(-1);
        fixture.model_ctx = createQwen36ParityModelContext(model_path, device);
        EXPECT_NE(fixture.model_ctx, nullptr);
        if (!fixture.model_ctx)
        {
            return fixture;
        }

        InferenceRunnerConfig config;
        config.max_seq_len = test_case.max_seq_len;
        config.batch_size = 1;
        config.force_graph = true;
        config.activation_precision = ActivationPrecision::FP32;
        config.kv_cache_precision =
            parseKVCachePrecision(test_case.kv_cache_precision);
        config.use_mapped_memory = false;
        config.mtp.enabled = true;
        config.mtp.draft_tokens = max_verifier_rows;

        fixture.runner =
            createInferenceRunner(fixture.model_ctx, nullptr, device, config);
        EXPECT_NE(fixture.runner, nullptr);
        if (!fixture.runner)
        {
            return fixture;
        }

        IInferenceRunner &runner = *fixture.runner;
        runner.setSuppressTimeline(
            !envBool("LLAMINAR_DENSE_VERIFIER_ECONOMY_STAGE_BREAKDOWN"));
        EXPECT_GT(runner.vocab_size(), 0);
        fixture.vocab_size = runner.vocab_size();

        if (!runner.forward(
                prompt_tokens.data(),
                static_cast<int>(prompt_tokens.size())))
        {
            ADD_FAILURE() << "prefill forward failed";
            return fixture;
        }
        EXPECT_EQ(sampleCurrentToken(runner, "prefill"), expected_tokens[0]);

        int32_t token_after_setup = -1;
        for (int i = 0; i < 2; ++i)
        {
            const int32_t token = expected_tokens[static_cast<size_t>(i)];
            if (!runner.forward(&token, 1))
            {
                ADD_FAILURE()
                    << "serial setup forward failed at token index " << i;
                return fixture;
            }
            token_after_setup = sampleCurrentToken(runner, "serial setup");
            if (token_after_setup < 0)
            {
                ADD_FAILURE()
                    << "serial setup sampled an invalid token at index " << i;
                return fixture;
            }
        }

        fixture.verifier_base = runner.captureLivePrefixState();
        EXPECT_TRUE(fixture.verifier_base.valid);
        if (!fixture.verifier_base.valid)
        {
            return fixture;
        }

        fixture.ready_tokens_by_count.assign(
            static_cast<size_t>(max_verifier_rows + 1),
            -1);
        fixture.verifier_tokens.reserve(static_cast<size_t>(max_verifier_rows));
        int32_t next_verifier_token = token_after_setup;
        for (int i = 0; i < max_verifier_rows; ++i)
        {
            fixture.verifier_tokens.push_back(next_verifier_token);
            if (!runner.forward(&next_verifier_token, 1))
            {
                ADD_FAILURE()
                    << "serial verifier-token extension failed at row " << i;
                return fixture;
            }
            next_verifier_token =
                sampleCurrentToken(runner, "serial verifier-token extension");
            if (next_verifier_token < 0)
            {
                ADD_FAILURE()
                    << "serial verifier-token extension sampled an invalid "
                    << "token at row " << i;
                return fixture;
            }
            fixture.ready_tokens_by_count[static_cast<size_t>(i + 1)] =
                next_verifier_token;
        }

        if (!runner.restoreLivePrefixState(fixture.verifier_base))
        {
            ADD_FAILURE()
                << "failed to restore verifier base after fixture construction";
            return fixture;
        }
        return fixture;
    }

    std::vector<int32_t> verifierTokenPrefix(
        const DenseVerifierEconomyFixture &fixture,
        int verifier_row_count)
    {
        const size_t count = static_cast<size_t>(verifier_row_count);
        return std::vector<int32_t>(
            fixture.verifier_tokens.begin(),
            fixture.verifier_tokens.begin() +
                static_cast<std::ptrdiff_t>(count));
    }

    MTPSpecDecodeVerifierInputPlan buildSingleRequestPlan(
        const std::vector<int32_t> &verifier_tokens)
    {
        MTPSpecDecodeMetadataShape shape;
        shape.max_requests = 1;
        shape.max_draft_tokens = static_cast<int>(verifier_tokens.size());

        MTPSpecDecodeVerifierDraftRequest verifier_request;
        verifier_request.request_id = 0;
        verifier_request.draft_tokens.assign(
            verifier_tokens.begin(),
            verifier_tokens.end());
        return buildMTPSpecDecodeVerifierInputPlan(shape, {verifier_request});
    }

    /**
     * @brief RAII owner for row-indexed all-position verifier modes.
     *
     * The grouped verifier modes alter graph materialization for the whole
     * runner.  Keeping their lifetime explicit makes the speed harness behave
     * like production code and prevents the next serial measurement from
     * accidentally inheriting verifier-only graph flags.
     */
    class ScopedGroupedVerifierMode
    {
    public:
        ScopedGroupedVerifierMode(
            IInferenceRunner &runner,
            const MTPSpecDecodeVerifierInputPlan &plan)
            : runner_(runner)
        {
            if (!plan.ok)
            {
                error_ = plan.error;
                return;
            }
            if (!runner_.setComputeRowIndexedAllPositionLogits(
                    true,
                    plan.compact_logit_row_count))
            {
                error_ = "failed to enable row-indexed all-position logits";
                return;
            }
            row_indexed_enabled_ = true;

            if (!runner_.setMTPSpecVerifierInputPlan(plan))
            {
                error_ = "failed to install MTP verifier row plan";
                return;
            }
            plan_installed_ = true;

            if (!runner_.setComputeAllPositionLogits(true))
            {
                error_ = "failed to enable all-position logits";
                return;
            }
            all_position_enabled_ = true;
            ok_ = true;
        }

        ~ScopedGroupedVerifierMode()
        {
            runner_.clearMTPSpecVerifierInputPlan();
            if (all_position_enabled_)
            {
                (void)runner_.setComputeAllPositionLogits(false);
            }
            if (row_indexed_enabled_)
            {
                (void)runner_.setComputeRowIndexedAllPositionLogits(false, 0);
            }
        }

        ScopedGroupedVerifierMode(const ScopedGroupedVerifierMode &) = delete;
        ScopedGroupedVerifierMode &operator=(const ScopedGroupedVerifierMode &) = delete;

        bool ok() const { return ok_; }
        const std::string &error() const { return error_; }

    private:
        IInferenceRunner &runner_;
        bool ok_ = false;
        bool row_indexed_enabled_ = false;
        bool plan_installed_ = false;
        bool all_position_enabled_ = false;
        std::string error_;
    };

    bool runGroupedVerifierReplay(
        const std::string &backend,
        DenseVerifierEconomyFixture &fixture,
        int verifier_row_count,
        std::vector<int32_t> *sampled_rows,
        std::vector<float> *logits_copy,
        DenseVerifierReplayTiming *timing = nullptr)
    {
        IInferenceRunner &runner = *fixture.runner;
        const std::vector<int32_t> verifier_tokens =
            verifierTokenPrefix(fixture, verifier_row_count);
        const auto tags =
            denseVerifierPerfTags(backend, "grouped", verifier_row_count);
        if (!measureReplayBucket(
                timing ? &timing->restore_ms : nullptr,
                [&]()
                {
                    PerfStatsCollector::ScopedTimer timer(
                        "dense_verifier",
                        "restore_live_prefix_state",
                        "economy",
                        backend,
                        tags);
                    return runner.restoreLivePrefixState(fixture.verifier_base);
                }))
        {
            ADD_FAILURE() << "grouped replay failed to restore verifier base";
            return false;
        }

        MTPSpecDecodeVerifierInputPlan row_plan;
        if (!measureReplayBucket(
                timing ? &timing->setup_ms : nullptr,
                [&]()
                {
                    PerfStatsCollector::ScopedTimer timer(
                        "dense_verifier",
                        "build_verifier_input_plan",
                        "economy",
                        backend,
                        tags);
                    row_plan = buildSingleRequestPlan(verifier_tokens);
                    return row_plan.ok;
                }))
        {
            ADD_FAILURE() << "could not build grouped verifier row plan: "
                          << row_plan.error;
            return false;
        }

        std::unique_ptr<ScopedGroupedVerifierMode> mode;
        if (!measureReplayBucket(
                timing ? &timing->setup_ms : nullptr,
                [&]()
                {
                    PerfStatsCollector::ScopedTimer timer(
                        "dense_verifier",
                        "enter_grouped_mode",
                        "economy",
                        backend,
                        tags);
                    mode = std::make_unique<ScopedGroupedVerifierMode>(runner, row_plan);
                    return mode && mode->ok();
                }))
        {
            ADD_FAILURE() << "could not enter grouped verifier mode: "
                          << (mode ? mode->error() : "mode allocation failed");
            return false;
        }

        if (!measureReplayBucket(
                timing ? &timing->forward_ms : nullptr,
                [&]()
                {
                    PerfStatsCollector::ScopedTimer forward_timer(
                        "dense_verifier",
                        "grouped_forward",
                        "economy",
                        backend,
                        tags);
                    return runner.forward(
                        verifier_tokens.data(),
                        static_cast<int>(verifier_tokens.size()));
                }))
        {
            ADD_FAILURE() << "dense grouped all-position verifier forward failed";
            return false;
        }

        if (!measureReplayBucket(
                timing ? &timing->sample_ms : nullptr,
                [&]()
                {
                    PerfStatsCollector::ScopedTimer sample_timer(
                        "dense_verifier",
                        "grouped_sample_rows",
                        "economy",
                        backend,
                        tags);
                    if (sampled_rows)
                    {
                        sampled_rows->assign(verifier_tokens.size(), -1);
                        return runner.sampleGreedyFromAllPositionLogitsOnDeviceRows(
                            0,
                            static_cast<int>(sampled_rows->size()),
                            sampled_rows->data());
                    }

                    std::vector<int32_t> ignored(verifier_tokens.size(), -1);
                    return runner.sampleGreedyFromAllPositionLogitsOnDeviceRows(
                        0,
                        static_cast<int>(ignored.size()),
                        ignored.data());
                }))
        {
            ADD_FAILURE()
                << "grouped verifier could not sample compact rows";
            return false;
        }

        if (logits_copy)
        {
            if (!measureReplayBucket(
                    timing ? &timing->logits_copy_ms : nullptr,
                    [&]()
                    {
                        PerfStatsCollector::ScopedTimer copy_timer(
                            "dense_verifier",
                            "grouped_logits_copy",
                            "economy",
                            backend,
                            tags);
                        const float *logits = runner.getAllPositionLogits();
                        if (!logits)
                        {
                            return false;
                        }
                        logits_copy->assign(
                            logits,
                            logits + static_cast<size_t>(verifier_tokens.size()) *
                                         static_cast<size_t>(fixture.vocab_size));
                        return true;
                    }))
            {
                ADD_FAILURE() << "grouped verifier did not expose row logits";
                return false;
            }
        }
        return true;
    }

    bool runSerialVerifierReplay(
        const std::string &backend,
        DenseVerifierEconomyFixture &fixture,
        int verifier_row_count,
        std::vector<int32_t> *sampled_rows,
        std::vector<std::vector<float>> *logits_by_row,
        DenseVerifierReplayTiming *timing = nullptr)
    {
        IInferenceRunner &runner = *fixture.runner;
        const std::vector<int32_t> verifier_tokens =
            verifierTokenPrefix(fixture, verifier_row_count);
        const auto tags =
            denseVerifierPerfTags(backend, "serial", verifier_row_count);
        if (!measureReplayBucket(
                timing ? &timing->restore_ms : nullptr,
                [&]()
                {
                    PerfStatsCollector::ScopedTimer timer(
                        "dense_verifier",
                        "restore_live_prefix_state",
                        "economy",
                        backend,
                        tags);
                    return runner.restoreLivePrefixState(fixture.verifier_base);
                }))
        {
            ADD_FAILURE() << "serial replay failed to restore verifier base";
            return false;
        }

        if (sampled_rows)
        {
            sampled_rows->clear();
            sampled_rows->reserve(verifier_tokens.size());
        }
        if (logits_by_row)
        {
            logits_by_row->clear();
            logits_by_row->reserve(verifier_tokens.size());
        }

        for (size_t row = 0; row < verifier_tokens.size(); ++row)
        {
            const int32_t token = verifier_tokens[row];
            if (!measureReplayBucket(
                    timing ? &timing->forward_ms : nullptr,
                    [&]()
                    {
                        PerfStatsCollector::ScopedTimer timer(
                            "dense_verifier",
                            "serial_forward_row",
                            "economy",
                            backend,
                            tags);
                        return runner.forward(&token, 1);
                    }))
            {
                ADD_FAILURE() << "serial verifier replay failed at row " << row;
                return false;
            }
            int32_t sampled = -1;
            if (!measureReplayBucket(
                    timing ? &timing->sample_ms : nullptr,
                    [&]()
                    {
                        PerfStatsCollector::ScopedTimer timer(
                            "dense_verifier",
                            "serial_sample_row",
                            "economy",
                            backend,
                            tags);
                        sampled = sampleCurrentToken(runner, "serial verifier replay");
                        return sampled >= 0;
                    }))
            {
                ADD_FAILURE()
                    << "serial verifier replay sampled an invalid token at row "
                    << row;
                return false;
            }
            if (sampled_rows)
            {
                sampled_rows->push_back(sampled);
            }
            if (logits_by_row)
            {
                if (!measureReplayBucket(
                        timing ? &timing->logits_copy_ms : nullptr,
                        [&]()
                        {
                            PerfStatsCollector::ScopedTimer timer(
                                "dense_verifier",
                                "serial_logits_copy",
                                "economy",
                                backend,
                                tags);
                            const float *logits = runner.logits();
                            if (!logits)
                            {
                                return false;
                            }
                            logits_by_row->emplace_back(
                                logits,
                                logits + static_cast<size_t>(fixture.vocab_size));
                            return true;
                        }))
                {
                    ADD_FAILURE()
                        << "serial verifier row " << row
                        << " did not expose logits";
                    return false;
                }
            }
        }
        return true;
    }

    struct AverageReplayTiming
    {
        double total_ms = 0.0;
        DenseVerifierReplayTiming component_ms;
    };

    AverageReplayTiming averageReplayTimedMs(
        int warmups,
        int iterations,
        const std::function<bool(DenseVerifierReplayTiming *)> &body)
    {
        for (int i = 0; i < warmups; ++i)
        {
            if (!body(nullptr))
            {
                ADD_FAILURE() << "warmup iteration " << i << " failed";
                return {std::numeric_limits<double>::infinity(), {}};
            }
        }

        AverageReplayTiming result;
        double total_ms = 0.0;
        DenseVerifierReplayTiming component_total;
        for (int i = 0; i < iterations; ++i)
        {
            DenseVerifierReplayTiming iteration_timing;
            const auto start = std::chrono::steady_clock::now();
            if (!body(&iteration_timing))
            {
                ADD_FAILURE() << "timed iteration " << i << " failed";
                return {std::numeric_limits<double>::infinity(), {}};
            }
            const auto end = std::chrono::steady_clock::now();
            total_ms += std::chrono::duration<double, std::milli>(
                            end - start)
                            .count();
            component_total.add(iteration_timing);
        }
        component_total.divide(static_cast<double>(iterations));
        result.total_ms = total_ms / static_cast<double>(iterations);
        result.component_ms = component_total;
        return result;
    }

    struct StageBreakdownRow
    {
        std::string stage;
        std::string stage_type;
        uint64_t count = 0;
        double total_ms = 0.0;
    };

    /**
     * @brief Print opt-in per-stage CPU attribution for one verifier replay path.
     *
     * `stage_cpu_detail` records are aggregated globally, which is useful for
     * whole-run summaries but too blurry when comparing grouped and serial
     * verifier replay. This helper isolates a single replay by resetting the
     * perf collector around that body, then emits path- and M-tagged CSV-style
     * lines. It is deliberately test-only instrumentation: production code keeps
     * using the unified `PerfStatsCollector` without verifier-specific state.
     */
    void printStageBreakdownForReplay(
        const std::string &backend,
        const char *path,
        int verifier_row_count,
        const std::function<bool()> &body)
    {
        if (!PerfStatsCollector::isEnabled())
        {
            return;
        }

        PerfStatsCollector::reset();
        if (!body())
        {
            ADD_FAILURE() << "stage breakdown replay failed for path="
                          << path << " M=" << verifier_row_count;
            PerfStatsCollector::reset();
            return;
        }

        std::vector<StageBreakdownRow> rows;
        std::map<std::string, StageBreakdownRow> by_type;
        double total_ms = 0.0;
        for (const auto &record :
             PerfStatsCollector::snapshot({"stage_cpu_detail"}))
        {
            if (record.kind != PerfStatRecord::Kind::Timer ||
                record.domain != "stage_cpu_detail")
            {
                continue;
            }
            StageBreakdownRow row;
            row.stage = record.name;
            const auto type_it = record.tags.find("stage_type");
            row.stage_type =
                type_it == record.tags.end() ? "" : type_it->second;
            row.count = record.count;
            row.total_ms = static_cast<double>(record.total_ns) / 1.0e6;
            total_ms += row.total_ms;
            auto &type_row = by_type[row.stage_type];
            type_row.stage = row.stage_type;
            type_row.stage_type = row.stage_type;
            type_row.count += row.count;
            type_row.total_ms += row.total_ms;
            rows.push_back(std::move(row));
        }

        std::sort(
            rows.begin(),
            rows.end(),
            [](const StageBreakdownRow &a, const StageBreakdownRow &b)
            {
                return a.total_ms > b.total_ms;
            });

        const int top_n =
            envInt("LLAMINAR_DENSE_VERIFIER_ECONOMY_STAGE_TOP", 16);
        std::cout << std::fixed << std::setprecision(4)
                  << "dense_verifier_stage_total,backend=" << backend
                  << ",path=" << path
                  << ",m=" << verifier_row_count
                  << ",stages=" << rows.size()
                  << ",total_ms=" << total_ms
                  << std::endl;

        const int limit = std::min<int>(
            static_cast<int>(rows.size()),
            std::max(0, top_n));
        for (int i = 0; i < limit; ++i)
        {
            const auto &row = rows[static_cast<size_t>(i)];
            std::cout << std::fixed << std::setprecision(4)
                      << "dense_verifier_stage_detail,backend=" << backend
                      << ",path=" << path
                      << ",m=" << verifier_row_count
                      << ",rank=" << i
                      << ",stage=" << row.stage
                      << ",stage_type=" << row.stage_type
                      << ",count=" << row.count
                      << ",total_ms=" << row.total_ms
                      << ",avg_us="
                      << (row.count > 0
                              ? row.total_ms * 1000.0 /
                                    static_cast<double>(row.count)
                              : 0.0)
                      << std::endl;
        }

        std::vector<StageBreakdownRow> type_rows;
        type_rows.reserve(by_type.size());
        for (auto &entry : by_type)
        {
            type_rows.push_back(std::move(entry.second));
        }
        std::sort(
            type_rows.begin(),
            type_rows.end(),
            [](const StageBreakdownRow &a, const StageBreakdownRow &b)
            {
                return a.total_ms > b.total_ms;
            });
        const int type_limit = std::min<int>(
            static_cast<int>(type_rows.size()),
            std::max(0, top_n));
        for (int i = 0; i < type_limit; ++i)
        {
            const auto &row = type_rows[static_cast<size_t>(i)];
            std::cout << std::fixed << std::setprecision(4)
                      << "dense_verifier_stage_type,backend=" << backend
                      << ",path=" << path
                      << ",m=" << verifier_row_count
                      << ",rank=" << i
                      << ",stage_type=" << row.stage_type
                      << ",count=" << row.count
                      << ",total_ms=" << row.total_ms
                      << ",avg_us="
                      << (row.count > 0
                              ? row.total_ms * 1000.0 /
                                    static_cast<double>(row.count)
                              : 0.0)
                      << std::endl;
        }

        std::vector<StageBreakdownRow> gpu_rows;
        std::map<std::string, StageBreakdownRow> gpu_by_type;
        double gpu_total_ms = 0.0;
        for (const auto &record :
             PerfStatsCollector::snapshot({"stage_gpu", "mtp_stage_gpu"}))
        {
            if (record.kind != PerfStatRecord::Kind::Timer ||
                (record.domain != "stage_gpu" &&
                 record.domain != "mtp_stage_gpu"))
            {
                continue;
            }
            if (record.name == "total" ||
                record.name.rfind("type.", 0) == 0)
            {
                continue;
            }

            StageBreakdownRow row;
            row.stage = record.name;
            const auto type_it = record.tags.find("type");
            row.stage_type =
                type_it == record.tags.end() ? "" : type_it->second;
            row.count = record.count;
            row.total_ms = static_cast<double>(record.total_ns) / 1.0e6;
            gpu_total_ms += row.total_ms;
            auto &type_row = gpu_by_type[row.stage_type];
            type_row.stage = row.stage_type;
            type_row.stage_type = row.stage_type;
            type_row.count += row.count;
            type_row.total_ms += row.total_ms;
            gpu_rows.push_back(std::move(row));
        }
        std::sort(
            gpu_rows.begin(),
            gpu_rows.end(),
            [](const StageBreakdownRow &a, const StageBreakdownRow &b)
            {
                return a.total_ms > b.total_ms;
            });

        std::cout << std::fixed << std::setprecision(4)
                  << "dense_verifier_gpu_stage_total,backend=" << backend
                  << ",path=" << path
                  << ",m=" << verifier_row_count
                  << ",stages=" << gpu_rows.size()
                  << ",total_ms=" << gpu_total_ms
                  << std::endl;
        const int gpu_limit = std::min<int>(
            static_cast<int>(gpu_rows.size()),
            std::max(0, top_n));
        for (int i = 0; i < gpu_limit; ++i)
        {
            const auto &row = gpu_rows[static_cast<size_t>(i)];
            std::cout << std::fixed << std::setprecision(4)
                      << "dense_verifier_gpu_stage_detail,backend=" << backend
                      << ",path=" << path
                      << ",m=" << verifier_row_count
                      << ",rank=" << i
                      << ",stage=" << row.stage
                      << ",stage_type=" << row.stage_type
                      << ",count=" << row.count
                      << ",total_ms=" << row.total_ms
                      << ",avg_us="
                      << (row.count > 0
                              ? row.total_ms * 1000.0 /
                                    static_cast<double>(row.count)
                              : 0.0)
                      << std::endl;
        }

        std::vector<StageBreakdownRow> gpu_type_rows;
        gpu_type_rows.reserve(gpu_by_type.size());
        for (auto &entry : gpu_by_type)
        {
            gpu_type_rows.push_back(std::move(entry.second));
        }
        std::sort(
            gpu_type_rows.begin(),
            gpu_type_rows.end(),
            [](const StageBreakdownRow &a, const StageBreakdownRow &b)
            {
                return a.total_ms > b.total_ms;
            });
        const int gpu_type_limit = std::min<int>(
            static_cast<int>(gpu_type_rows.size()),
            std::max(0, top_n));
        for (int i = 0; i < gpu_type_limit; ++i)
        {
            const auto &row = gpu_type_rows[static_cast<size_t>(i)];
            std::cout << std::fixed << std::setprecision(4)
                      << "dense_verifier_gpu_stage_type,backend=" << backend
                      << ",path=" << path
                      << ",m=" << verifier_row_count
                      << ",rank=" << i
                      << ",stage_type=" << row.stage_type
                      << ",count=" << row.count
                      << ",total_ms=" << row.total_ms
                      << ",avg_us="
                      << (row.count > 0
                              ? row.total_ms * 1000.0 /
                                    static_cast<double>(row.count)
                              : 0.0)
                      << std::endl;
        }

        // Graph-captured verifier runs may not emit per-stage CPU detail, but
        // the harness always emits these replay-scoped timers.  Keep this
        // companion breakdown so a zero-stage report still shows whether the
        // measured time is in graph replay, sampling, restore, or host setup.
        std::vector<StageBreakdownRow> replay_rows;
        double replay_total_ms = 0.0;
        for (const auto &record :
             PerfStatsCollector::snapshot({"dense_verifier"}))
        {
            if (record.kind != PerfStatRecord::Kind::Timer ||
                record.domain != "dense_verifier")
            {
                continue;
            }

            StageBreakdownRow row;
            row.stage = record.name;
            row.stage_type = record.phase;
            row.count = record.count;
            row.total_ms = static_cast<double>(record.total_ns) / 1.0e6;
            replay_total_ms += row.total_ms;
            replay_rows.push_back(std::move(row));
        }
        std::sort(
            replay_rows.begin(),
            replay_rows.end(),
            [](const StageBreakdownRow &a, const StageBreakdownRow &b)
            {
                return a.total_ms > b.total_ms;
            });

        std::cout << std::fixed << std::setprecision(4)
                  << "dense_verifier_replay_total,backend=" << backend
                  << ",path=" << path
                  << ",m=" << verifier_row_count
                  << ",timers=" << replay_rows.size()
                  << ",total_ms=" << replay_total_ms
                  << std::endl;
        const int replay_limit = std::min<int>(
            static_cast<int>(replay_rows.size()),
            std::max(0, top_n));
        for (int i = 0; i < replay_limit; ++i)
        {
            const auto &row = replay_rows[static_cast<size_t>(i)];
            std::cout << std::fixed << std::setprecision(4)
                      << "dense_verifier_replay_detail,backend=" << backend
                      << ",path=" << path
                      << ",m=" << verifier_row_count
                      << ",rank=" << i
                      << ",timer=" << row.stage
                      << ",phase=" << row.stage_type
                      << ",count=" << row.count
                      << ",total_ms=" << row.total_ms
                      << ",avg_us="
                      << (row.count > 0
                              ? row.total_ms * 1000.0 /
                                    static_cast<double>(row.count)
                              : 0.0)
                      << std::endl;
        }
        PerfStatsCollector::reset();
    }

    struct CrossRowMatch
    {
        size_t serial_row = 0;
        DenseVerifierLogitMetrics metrics;
    };

    /**
     * @brief Find the serial row whose distribution is closest to one grouped row.
     *
     * A row-indexing bug usually appears as "grouped row i matches serial row
     * j".  A real state/coherence bug does not match any serial row well.  The
     * verifier economy test only prints this on failure so normal benchmark
     * output stays compact.
     */
    CrossRowMatch bestSerialRowForGroupedLogits(
        const float *grouped_row,
        const std::vector<std::vector<float>> &serial_logits_by_row,
        int vocab_size)
    {
        CrossRowMatch best;
        best.metrics.cosine = -1.0;
        best.metrics.rel_l2 = std::numeric_limits<double>::infinity();
        best.metrics.symmetric_kl = std::numeric_limits<double>::infinity();

        for (size_t candidate = 0; candidate < serial_logits_by_row.size(); ++candidate)
        {
            const DenseVerifierLogitMetrics metrics =
                computeDenseVerifierLogitMetrics(
                    grouped_row,
                    serial_logits_by_row[candidate].data(),
                    vocab_size);
            const bool better =
                metrics.cosine > best.metrics.cosine ||
                (metrics.cosine == best.metrics.cosine &&
                 metrics.rel_l2 < best.metrics.rel_l2);
            if (better)
            {
                best.serial_row = candidate;
                best.metrics = metrics;
            }
        }
        return best;
    }

    DenseVerifierEconomyRow runDenseVerifierEconomyCase(
        const std::string &backend,
        DenseVerifierEconomyFixture &fixture,
        int verifier_row_count)
    {
        DenseVerifierEconomyRow row;
        row.backend = backend;
        row.rows = verifier_row_count;
        const int32_t expected_ready_token =
            verifier_row_count >= 0 &&
                    static_cast<size_t>(verifier_row_count) <
                        fixture.ready_tokens_by_count.size()
                ? fixture.ready_tokens_by_count[static_cast<size_t>(verifier_row_count)]
                : -1;

        if (!fixture.runner || !fixture.verifier_base.valid ||
            fixture.verifier_tokens.size() <
                static_cast<size_t>(verifier_row_count) ||
            expected_ready_token < 0)
        {
            ADD_FAILURE()
                << backend << " dense verifier economy fixture construction "
                << "failed for M=" << verifier_row_count;
            return row;
        }

        std::vector<int32_t> grouped_rows;
        std::vector<float> grouped_logits;
        if (!runGroupedVerifierReplay(
                backend,
                fixture,
                verifier_row_count,
                &grouped_rows,
                &grouped_logits) ||
            grouped_rows.size() != static_cast<size_t>(verifier_row_count) ||
            grouped_rows.empty() ||
            grouped_rows.back() != expected_ready_token)
        {
            ADD_FAILURE()
                << backend << " grouped verifier diagnostic replay failed for M="
                << verifier_row_count;
            return row;
        }

        std::vector<int32_t> serial_rows;
        std::vector<std::vector<float>> serial_logits_by_row;
        if (!runSerialVerifierReplay(
                backend,
                fixture,
                verifier_row_count,
                &serial_rows,
                &serial_logits_by_row) ||
            serial_rows.size() != static_cast<size_t>(verifier_row_count) ||
            serial_logits_by_row.size() !=
                static_cast<size_t>(verifier_row_count) ||
            serial_rows.empty() ||
            serial_rows.back() != expected_ready_token)
        {
            ADD_FAILURE()
                << backend << " serial verifier diagnostic replay failed for M="
                << verifier_row_count;
            return row;
        }

        row.grouped_ready_token = grouped_rows.back();
        row.serial_ready_token = serial_rows.back();

        for (size_t i = 0; i < static_cast<size_t>(verifier_row_count); ++i)
        {
            const float *grouped_row =
                grouped_logits.data() + i * static_cast<size_t>(fixture.vocab_size);
            const DenseVerifierLogitMetrics metrics =
                computeDenseVerifierLogitMetrics(
                    grouped_row,
                    serial_logits_by_row[i].data(),
                    fixture.vocab_size);
            if (metrics.rel_l2 > row.worst_metrics.rel_l2 ||
                metrics.symmetric_kl > row.worst_metrics.symmetric_kl ||
                metrics.cosine < row.worst_metrics.cosine)
            {
                row.worst_metrics = metrics;
            }

            const ::testing::AssertionResult equivalent =
                denseVerifierLogitsNumericallyEquivalent(
                grouped_row,
                serial_logits_by_row[i].data(),
                fixture.vocab_size,
                backend + " dense grouped verifier row " +
                    std::to_string(i) + " economy proof");
            if (!equivalent)
            {
                const CrossRowMatch best_match =
                    bestSerialRowForGroupedLogits(
                        grouped_row,
                        serial_logits_by_row,
                        fixture.vocab_size);
                std::cout << backend << " dense verifier row mapping diagnostic"
                          << ", grouped_row=" << i
                          << ", serial_row=" << best_match.serial_row
                          << ", cosine=" << best_match.metrics.cosine
                          << ", rel_l2=" << best_match.metrics.rel_l2
                          << ", symmetric_kl=" << best_match.metrics.symmetric_kl
                          << ", max_abs_diff=" << best_match.metrics.max_abs_diff
                          << ", max_abs_index=" << best_match.metrics.max_abs_index
                          << std::endl;
            }
            EXPECT_TRUE(equivalent);
            EXPECT_EQ(grouped_rows[i], serial_rows[i])
                << backend << " grouped verifier row " << i
                << " sampled a different token than serial replay";
        }

        const int warmups =
            envInt("LLAMINAR_DENSE_VERIFIER_ECONOMY_WARMUPS", 1);
        const int iterations =
            envInt("LLAMINAR_DENSE_VERIFIER_ECONOMY_ITERS", 1);
        if (envBool("LLAMINAR_DENSE_VERIFIER_ECONOMY_STAGE_BREAKDOWN"))
        {
            printStageBreakdownForReplay(
                backend,
                "grouped",
                verifier_row_count,
                [&]()
                {
                    return runGroupedVerifierReplay(
                        backend,
                        fixture,
                        verifier_row_count,
                        nullptr,
                        nullptr);
                });
            printStageBreakdownForReplay(
                backend,
                "serial",
                verifier_row_count,
                [&]()
                {
                    return runSerialVerifierReplay(
                        backend,
                        fixture,
                        verifier_row_count,
                        nullptr,
                        nullptr);
                });
        }

        const AverageReplayTiming grouped_timing = averageReplayTimedMs(
            warmups,
            iterations,
            [&](DenseVerifierReplayTiming *timing)
            {
                return runGroupedVerifierReplay(
                    backend,
                    fixture,
                    verifier_row_count,
                    nullptr,
                    nullptr,
                    timing);
            });
        const AverageReplayTiming serial_timing = averageReplayTimedMs(
            warmups,
            iterations,
            [&](DenseVerifierReplayTiming *timing)
            {
                return runSerialVerifierReplay(
                    backend,
                    fixture,
                    verifier_row_count,
                    nullptr,
                    nullptr,
                    timing);
            });
        row.grouped_ms = grouped_timing.total_ms;
        row.serial_ms = serial_timing.total_ms;
        row.grouped_restore_ms = grouped_timing.component_ms.restore_ms;
        row.serial_restore_ms = serial_timing.component_ms.restore_ms;
        row.grouped_setup_ms = grouped_timing.component_ms.setup_ms;
        row.serial_setup_ms = serial_timing.component_ms.setup_ms;
        row.grouped_forward_ms = grouped_timing.component_ms.forward_ms;
        row.serial_forward_ms = serial_timing.component_ms.forward_ms;
        row.grouped_sample_ms = grouped_timing.component_ms.sample_ms;
        row.serial_sample_ms = serial_timing.component_ms.sample_ms;

        std::cout << std::fixed << std::setprecision(4)
                  << "dense_verifier_economy,backend=" << backend
                  << ",m=" << row.rows
                  << ",grouped_ms=" << row.grouped_ms
                  << ",serial_ms=" << row.serial_ms
                  << ",speedup="
                  << (row.serial_ms > 0.0 ? row.serial_ms / row.grouped_ms : 0.0)
                  << ",grouped_restore_ms=" << row.grouped_restore_ms
                  << ",serial_restore_ms=" << row.serial_restore_ms
                  << ",grouped_setup_ms=" << row.grouped_setup_ms
                  << ",serial_setup_ms=" << row.serial_setup_ms
                  << ",grouped_forward_ms=" << row.grouped_forward_ms
                  << ",serial_forward_ms=" << row.serial_forward_ms
                  << ",forward_speedup="
                  << (row.grouped_forward_ms > 0.0
                          ? row.serial_forward_ms / row.grouped_forward_ms
                          : 0.0)
                  << ",grouped_sample_ms=" << row.grouped_sample_ms
                  << ",serial_sample_ms=" << row.serial_sample_ms
                  << ",cosine=" << row.worst_metrics.cosine
                  << ",rel_l2=" << row.worst_metrics.rel_l2
                  << ",symmetric_kl=" << row.worst_metrics.symmetric_kl
                  << std::endl;

        EXPECT_LT(row.grouped_ms, row.serial_ms)
            << backend << " dense grouped verifier replay must be faster than "
            << "the serial decode-equivalent fallback for M="
            << verifier_row_count;
        return row;
    }

    void runDenseVerifierEconomyBackend(
        const std::string &backend,
        const DensePrefixRestoreParityCase &test_case)
    {
        ScopedDenseParityProductionMode production_mode(
            shouldForceDenseParityProductionMode(test_case));

        DenseVerifierEconomyFixture fixture =
            prepareDenseVerifierEconomyFixture(
                test_case,
                /*max_verifier_rows=*/4);
        ASSERT_NE(fixture.runner, nullptr);
        ASSERT_TRUE(fixture.verifier_base.valid);
        ASSERT_GE(fixture.verifier_tokens.size(), 4u);
        ASSERT_GE(fixture.ready_tokens_by_count.size(), 5u);

        std::vector<DenseVerifierEconomyRow> rows;
        const int requested_m =
            envInt("LLAMINAR_DENSE_VERIFIER_ECONOMY_M", 0);
        const std::vector<int> requested_rows =
            requested_m >= 2 && requested_m <= 4
                ? std::vector<int>{requested_m}
                : std::vector<int>{2, 3, 4};
        for (int m : requested_rows)
        {
            rows.push_back(runDenseVerifierEconomyCase(backend, fixture, m));
        }

        const double min_speedup =
            std::accumulate(
                rows.begin(),
                rows.end(),
                std::numeric_limits<double>::infinity(),
                [](double best, const DenseVerifierEconomyRow &row)
                {
                    const double speedup =
                        row.grouped_ms > 0.0 ? row.serial_ms / row.grouped_ms : 0.0;
                    return std::min(best, speedup);
                });
        EXPECT_GT(min_speedup, 1.0)
            << backend << " dense verifier economy failed to produce a speedup "
            << "for every production row count";
    }
} // namespace

TEST(Perf__DenseVerifierRows, CPU_M234_GroupedReplayFasterThanSerial)
{
    runDenseVerifierEconomyBackend(
        "CPU",
        denseSingleDeviceCase("CPU", GlobalDeviceAddress::cpu()));
}

TEST(Perf__DenseVerifierRows, CUDA_M234_GroupedReplayFasterThanSerial)
{
    runDenseVerifierEconomyBackend(
        "CUDA",
        denseSingleDeviceCase("CUDA", GlobalDeviceAddress::cuda(0)));
}

TEST(Perf__DenseVerifierRows, ROCm_M234_GroupedReplayFasterThanSerial)
{
    runDenseVerifierEconomyBackend(
        "ROCm",
        denseSingleDeviceCase("ROCm", GlobalDeviceAddress::rocm(0)));
}

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();

    (void)PerfStatsCollector::flushFromEnv();

    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
