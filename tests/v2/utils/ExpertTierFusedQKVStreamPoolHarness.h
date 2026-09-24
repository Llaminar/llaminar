/**
 * @file ExpertTierFusedQKVStreamPoolHarness.h
 * @brief Cross-backend adversarial FusedQKV/migration overlap certificate.
 *
 * FusedQKV is a stronger interference witness than a single GEMV because the
 * production quantized kernel forks Q/K/V onto a persistent stream pool and
 * joins three completion events back to the graph stream. This harness runs
 * both decode and prefill transactions repeatedly while an ExpertOverlay
 * demotion advances on a fourth, independently owned auxiliary stream.
 */

#pragma once

#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "execution/compute_stages/stages/FusedQKVGEMMStage.h"
#include "execution/local_execution/coherence/GpuCoherence.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/moe/ExpertTierWeightTransferLane.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

#include "PreparedWeightTestHarness.h"
#include "GpuPreparedGemmHarness.h"
#include "TestTensorFactory.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace llaminar2::test
{
    /** @brief Enable structured evidence and restore the process setting. */
    class ScopedFusedQKVOverlapPerfStats final
    {
    public:
        /** @brief Enable inexpensive PerfStats counters for this certificate. */
        ScopedFusedQKVOverlapPerfStats()
        {
            if (const char *value =
                    std::getenv("LLAMINAR_PERF_STATS_SUMMARY"))
            {
                had_value_ = true;
                value_ = value;
            }
            setenv("LLAMINAR_PERF_STATS_SUMMARY", "1", 1);
            mutableDebugEnv().reload();
            PerfStatsCollector::reset();
        }

        /** @brief Restore PerfStats configuration without exporting an artifact. */
        ~ScopedFusedQKVOverlapPerfStats()
        {
            if (had_value_)
                setenv(
                    "LLAMINAR_PERF_STATS_SUMMARY", value_.c_str(), 1);
            else
                unsetenv("LLAMINAR_PERF_STATS_SUMMARY");
            mutableDebugEnv().reload();
            PerfStatsCollector::reset();
        }

        ScopedFusedQKVOverlapPerfStats(
            const ScopedFusedQKVOverlapPerfStats &) = delete;
        ScopedFusedQKVOverlapPerfStats &operator=(
            const ScopedFusedQKVOverlapPerfStats &) = delete;

    private:
        bool had_value_ = false;
        std::string value_;
    };

    /**
     * @brief Restore mutable backend dispatch switches after an adversarial run.
     *
     * Tests change the typed snapshot directly rather than repeatedly mutating
     * process environment between launches. The production branch reads these
     * exact fields on every fused transaction.
     */
    class ScopedFusedQKVConcurrencySettings final
    {
    public:
        /** @brief Capture all CUDA and ROCm projection-pool switches. */
        ScopedFusedQKVConcurrencySettings()
            : cuda_prefill_(mutableDebugEnv().gemm.cuda_concurrent_prefill),
              cuda_decode_(mutableDebugEnv().gemm.cuda_concurrent_decode),
              rocm_prefill_(mutableDebugEnv().rocm.concurrent_prefill),
              rocm_decode_(mutableDebugEnv().rocm.concurrent_decode)
        {
        }

        /** @brief Restore every switch even when a GTest assertion returns. */
        ~ScopedFusedQKVConcurrencySettings()
        {
            mutableDebugEnv().gemm.cuda_concurrent_prefill = cuda_prefill_;
            mutableDebugEnv().gemm.cuda_concurrent_decode = cuda_decode_;
            mutableDebugEnv().rocm.concurrent_prefill = rocm_prefill_;
            mutableDebugEnv().rocm.concurrent_decode = rocm_decode_;
        }

        ScopedFusedQKVConcurrencySettings(
            const ScopedFusedQKVConcurrencySettings &) = delete;
        ScopedFusedQKVConcurrencySettings &operator=(
            const ScopedFusedQKVConcurrencySettings &) = delete;

        /**
         * @brief Select serial or pooled dispatch for one exact backend/mode.
         * @param device CUDA or ROCm endpoint under test.
         * @param decode True for M=1 decode; false for prompt prefill.
         * @param enabled Whether the production side-stream pool must run.
         */
        void set(DeviceId device, bool decode, bool enabled)
        {
            if (device.is_cuda())
            {
                if (decode)
                    mutableDebugEnv().gemm.cuda_concurrent_decode = enabled;
                else
                    mutableDebugEnv().gemm.cuda_concurrent_prefill = enabled;
            }
            else if (device.is_rocm())
            {
                if (decode)
                    mutableDebugEnv().rocm.concurrent_decode = enabled;
                else
                    mutableDebugEnv().rocm.concurrent_prefill = enabled;
            }
        }

    private:
        bool cuda_prefill_ = false;
        bool cuda_decode_ = false;
        bool rocm_prefill_ = false;
        bool rocm_decode_ = false;
    };

    /** @brief Host observation of the three projection outputs. */
    struct FusedQKVOutputSnapshot
    {
        std::vector<float> q;
        std::vector<float> k;
        std::vector<float> v;
    };

    /**
     * @brief Copy authoritative Q/K/V results at a test-only observation edge.
     * @param q Query output.
     * @param k Key output.
     * @param v Value output.
     * @return Complete host snapshot preserving FP32 bytes.
     */
    inline FusedQKVOutputSnapshot snapshotFusedQKVOutputs(
        FP32Tensor &q,
        FP32Tensor &k,
        FP32Tensor &v)
    {
        const float *q_data = q.data();
        const float *k_data = k.data();
        const float *v_data = v.data();
        return {
            std::vector<float>(q_data, q_data + q.numel()),
            std::vector<float>(k_data, k_data + k.numel()),
            std::vector<float>(v_data, v_data + v.numel()),
        };
    }

    /**
     * @brief Return summed stream-pool calls for one backend and phase.
     * @param device Backend endpoint selecting the stable counter name.
     * @param mode Exact `decode` or `prefill` tag.
     */
    inline double fusedQKVStreamPoolCalls(
        DeviceId device,
        const std::string &mode)
    {
        const std::string name = device.is_cuda()
                                     ? "cuda_fused_projection_stream_pool_calls"
                                     : "rocm_fused_projection_stream_pool_calls";
        double total = 0.0;
        for (const auto &record : PerfStatsCollector::snapshot({"kernel"}))
        {
            const auto mode_it = record.tags.find("mode");
            if (record.kind == PerfStatRecord::Kind::Counter &&
                record.name == name && mode_it != record.tags.end() &&
                mode_it->second == mode)
            {
                total += record.value;
            }
        }
        return total;
    }

    /**
     * @brief Exercise decode and prefill stream pools during tier migration.
     *
     * Each phase first records a sequential GPU baseline, then submits several
     * back-to-back FusedQKV stage executions through the real persistent pool.
     * The root completion event must become ready while the host-pumped weight
     * stream still has chunks outstanding. Final Q/K/V bytes and streamed CPU
     * execution bytes must be exact.
     *
     * @param device Exact CUDA or ROCm endpoint.
     * @param root_stream Explicit non-default graph/inference stream.
     * @param backend Backend owning events for @p device.
     * @param migration_layout Long, one-unit GPU-to-CPU transfer layout.
     * @param migration_source Immutable packed source arrays on @p device.
     * @param expected_cpu_bytes Byte-exact CPU-format conversion oracle.
     * @param lane_prefix Stable backend-specific auxiliary-lane prefix.
     */
    inline void runFusedQKVStreamPoolMigrationStress(
        DeviceId device,
        void *root_stream,
        IBackend &backend,
        const ExpertTierWeightDeviceLayout &migration_layout,
        const ExpertTierGpuConstProjectionView &migration_source,
        std::span<const std::uint8_t> expected_cpu_bytes,
        const std::string &lane_prefix)
    {
        ASSERT_TRUE(device.is_cuda() || device.is_rocm());
        ASSERT_NE(root_stream, nullptr);
        ASSERT_TRUE(migration_layout.valid());
        ASSERT_EQ(
            migration_layout.direction,
            ExpertTierWeightConversionDirection::GpuToCpu);
        ASSERT_TRUE(migration_source.validFor(migration_layout));
        ASSERT_EQ(
            expected_cpu_bytes.size(),
            migration_layout.chunkBytes(migration_layout.unit_count));
        ASSERT_GE(migration_layout.unit_count, 64u)
            << "The overlap witness needs enough host-pumped chunks to remain pending";

        ScopedFusedQKVOverlapPerfStats perf_stats;
        ScopedFusedQKVConcurrencySettings concurrency_settings;

        struct Phase
        {
            int m;
            bool decode;
            int repetitions;
            int migration_prepolls;
            const char *name;
        };
        constexpr Phase phases[] = {
            {1, true, 12, 0, "decode"},
            {32, false, 8, 3, "prefill"},
        };

        constexpr int k = 256;
        constexpr int n_q = 192;
        constexpr int n_k = 128;
        constexpr int n_v = 128;

        for (std::size_t phase_index = 0;
             phase_index < std::size(phases);
             ++phase_index)
        {
            const Phase phase = phases[phase_index];
            SCOPED_TRACE(phase.name);

            auto wq = TestTensorFactory::createQ4_0Random(
                {n_q, k}, 95101u + static_cast<std::uint32_t>(phase_index * 10));
            auto wk = TestTensorFactory::createQ4_0Random(
                {n_k, k}, 95102u + static_cast<std::uint32_t>(phase_index * 10));
            auto wv = TestTensorFactory::createQ4_0Random(
                {n_v, k}, 95103u + static_cast<std::uint32_t>(phase_index * 10));
            ASSERT_NE(wq, nullptr);
            ASSERT_NE(wk, nullptr);
            ASSERT_NE(wv, nullptr);

            auto prepared = makeGpuPreparedQKVFixture(
                wq.get(),
                wk.get(),
                wv.get(),
                device,
                lane_prefix + ":prepared:" + phase.name,
                ModelContextId{95100u + phase_index});
            ASSERT_TRUE(prepared.store->contains(prepared.q_ref));
            ASSERT_TRUE(prepared.store->contains(prepared.k_ref));
            ASSERT_TRUE(prepared.store->contains(prepared.v_ref));

            FP32Tensor input({
                static_cast<std::size_t>(phase.m),
                static_cast<std::size_t>(k)});
            for (std::size_t index = 0; index < input.numel(); ++index)
            {
                const int centered = static_cast<int>(
                    (index * 29u + phase_index * 17u) % 251u) - 125;
                input.mutable_data()[index] =
                    static_cast<float>(centered) / 128.0f;
            }

            FP32Tensor output_q({
                static_cast<std::size_t>(phase.m),
                static_cast<std::size_t>(n_q)});
            FP32Tensor output_k({
                static_cast<std::size_t>(phase.m),
                static_cast<std::size_t>(n_k)});
            FP32Tensor output_v({
                static_cast<std::size_t>(phase.m),
                static_cast<std::size_t>(n_v)});

            FusedQKVGEMMStage stage({
                .device_id = device,
                .input = &input,
                .m = phase.m,
                .k = k,
                .wq = wq.get(),
                .output_q = &output_q,
                .n_q = n_q,
                .wk = wk.get(),
                .output_k = &output_k,
                .n_k = n_k,
                .wv = wv.get(),
                .output_v = &output_v,
                .n_v = n_v,
                .prepared_ref_q = prepared.q_ref,
                .prepared_ref_k = prepared.k_ref,
                .prepared_ref_v = prepared.v_ref,
                .prepared_store = prepared.store.get(),
            });
            auto context = IDeviceContext::create(device);
            ASSERT_NE(context, nullptr);
            const auto requirements =
                stage.getWorkspaceRequirements(phase.m, n_q, k);
            DeviceWorkspaceManager workspace(
                device,
                requirements.total_bytes_with_alignment() + 4096u);
            ASSERT_TRUE(workspace.allocate(requirements));
            stage.bindWorkspace(&workspace);
            ASSERT_TRUE(stage.prepareGraphLaunch(context.get(), root_stream));

            ExpertTierWeightTransferLane lane({
                .device = device,
                .staging = TransferEngine::instance()
                               .allocatePersistentTransferStagingSlices(
                                   migration_layout.chunkBytes(
                                       migration_layout.maximum_units_per_chunk),
                                   1u,
                                   device)
                               .front(),
                .execution = TransferEngine::instance()
                                 .allocatePersistentTransferExecutionLanes(
                                     1u,
                                     device,
                                     lane_prefix + ":" + phase.name)
                                 .front(),
                .progress = BackgroundTransferProgressBinding::nativeStream(),
                .lane_name = lane_prefix + ":" + phase.name,
                .perf_device = device.to_string(),
            });
            std::string error;
            ASSERT_TRUE(lane.materialize(&error)) << error;

            const auto execute_stage = [&]()
            {
                stage.setGPUStream(root_stream);
                return with_gpu_coherence(
                    device,
                    {&input},
                    {&output_q, &output_k, &output_v},
                    root_stream,
                    [&]() { return stage.execute(context.get()); });
            };

            // First establish the exact same GPU math without side-stream fan-out.
            concurrency_settings.set(device, phase.decode, false);
            ASSERT_TRUE(execute_stage());
            const FusedQKVOutputSnapshot baseline =
                snapshotFusedQKVOutputs(output_q, output_k, output_v);
            ASSERT_TRUE(std::all_of(
                baseline.q.begin(), baseline.q.end(),
                [](float value) { return std::isfinite(value); }));
            EXPECT_TRUE(std::any_of(
                baseline.q.begin(), baseline.q.end(),
                [](float value) { return value != 0.0f; }));

            void *source_ready = backend.createEvent(device.gpu_ordinal());
            void *stage_done = backend.createEvent(device.gpu_ordinal());
            ASSERT_NE(source_ready, nullptr);
            ASSERT_NE(stage_done, nullptr);
            ASSERT_TRUE(backend.recordEvent(
                source_ready, device.gpu_ordinal(), root_stream));

            std::vector<std::uint8_t> observed_cpu(
                expected_cpu_bytes.size());
            ASSERT_TRUE(lane.startGpuToCpu(
                migration_layout,
                migration_source,
                observed_cpu,
                ExpertTierSourceReadiness::producerEvent(source_ready),
                &error)) << error;

            // Perturb when the migration worker gets CPU time before fan-out.
            for (int poll = 0; poll < phase.migration_prepolls; ++poll)
            {
                const auto progress = lane.poll(&error);
                ASSERT_NE(
                    progress,
                    ExpertTierWeightTransferProgress::Failed) << error;
            }

            concurrency_settings.set(device, phase.decode, true);
            PerfStatsCollector::reset();
            bool all_launches_succeeded = true;
            for (int repetition = 0;
                 repetition < phase.repetitions;
                 ++repetition)
            {
                all_launches_succeeded =
                    execute_stage() && all_launches_succeeded;
            }
            const bool stage_event_recorded = backend.recordEvent(
                stage_done, device.gpu_ordinal(), root_stream);

            bool stage_ready = false;
            bool stage_ready_while_migration_pending = false;
            auto migration_progress = lane.progress();
            auto &gpu_context =
                GPUDeviceContextPool::instance().getContext(device);
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(15);
            while (std::chrono::steady_clock::now() < deadline &&
                   (migration_progress ==
                        ExpertTierWeightTransferProgress::Pending ||
                    !stage_ready))
            {
                if (stage_event_recorded)
                {
                    const bool query_ok = gpu_context.queryEventChecked(
                        stage_done, stage_ready);
                    EXPECT_TRUE(query_ok);
                }
                if (stage_ready &&
                    lane.progress() ==
                        ExpertTierWeightTransferProgress::Pending)
                {
                    stage_ready_while_migration_pending = true;
                }
                if (migration_progress ==
                    ExpertTierWeightTransferProgress::Pending)
                {
                    migration_progress = lane.poll(&error);
                }
                if (migration_progress ==
                    ExpertTierWeightTransferProgress::Failed)
                {
                    break;
                }
                std::this_thread::yield();
            }

            /*
             * Report only after the lane is terminal. This ordering keeps a
             * failed assertion from destroying pinned/device staging while an
             * asynchronous copy still owns it.
             */
            EXPECT_TRUE(all_launches_succeeded);
            EXPECT_TRUE(stage_event_recorded);
            EXPECT_TRUE(stage_ready);
            EXPECT_TRUE(stage_ready_while_migration_pending)
                << "FusedQKV root completion did not overlap background migration";
            ASSERT_EQ(
                migration_progress,
                ExpertTierWeightTransferProgress::Ready) << error;
            ASSERT_EQ(observed_cpu.size(), expected_cpu_bytes.size());
            EXPECT_TRUE(std::equal(
                observed_cpu.begin(),
                observed_cpu.end(),
                expected_cpu_bytes.begin()));

            const FusedQKVOutputSnapshot concurrent =
                snapshotFusedQKVOutputs(output_q, output_k, output_v);
            EXPECT_EQ(concurrent.q, baseline.q)
                << "Q bytes changed under migration and stream-pool reuse";
            EXPECT_EQ(concurrent.k, baseline.k)
                << "K bytes changed under migration and stream-pool reuse";
            EXPECT_EQ(concurrent.v, baseline.v)
                << "V bytes changed under migration and stream-pool reuse";
            EXPECT_EQ(
                fusedQKVStreamPoolCalls(device, phase.name),
                static_cast<double>(phase.repetitions))
                << "PerfStats must prove every launch used the internal pool";

            const auto lane_stats = lane.stats();
            EXPECT_EQ(lane_stats.transfers_completed, 1u);
            EXPECT_EQ(
                lane_stats.chunks_submitted,
                migration_layout.unit_count);
            EXPECT_EQ(lane_stats.inference_stream_waits, 0u);
            EXPECT_EQ(lane_stats.blocking_synchronizations, 0u);

            backend.destroyEvent(stage_done, device.gpu_ordinal());
            backend.destroyEvent(source_ready, device.gpu_ordinal());
            stage.unbindWorkspace();
        }
    }
} // namespace llaminar2::test
