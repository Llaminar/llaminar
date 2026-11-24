/**
 * @file FixedKPrefillChunkHarness.h
 * @brief Backend-neutral chunk-invariance replay for explicit-operand GDN prefill projections.
 *
 * Issue #74: with prefix caching on, a cold prompt is prefilled in different
 * chunks (for example 512+256+10 rows instead of 512+266) because recurrent
 * checkpoints split it. Each chunk runs a retained prefill graph whose physical
 * row count is the chunk's bucket. Explicit half-RNE GDN projections must give
 * every prompt row identical bytes regardless of the chunk that carries it,
 * its bucket and its row offset within the chunk.
 *
 * This harness builds the real GDNProjectionStage for each bucket, captures one
 * native graph per bucket over one shared admitted workspace, and replays a
 * prompt under several chunk schedules. Between replays it publishes only the
 * chunk's live rows into the bucket input and poisons the padded rows, then
 * compares every live output row of every explicit-operand projection across
 * schedules byte for byte. Capture-time route counters prove that ordinary
 * half prefill used the fixed-K schedule family, never vendor BLAS.
 */
#pragma once

#include <gtest/gtest.h>
#include "GpuPreparedGemmHarness.h"
#include "loaders/PreparedHalfWeights.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "QuantizedVerifierFormats.h"
#include "ScopedGPUStream.h"
#include "backends/IGPUGraphCapture.h"
#include "execution/compute_stages/stages/GDNProjectionStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "transfer/TransferEngine.h"
#include "utils/PerfStatsCollector.h"
#include "utils/PrefillGraphBucketDefaults.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <span>
#include <vector>

namespace llaminar2::test
{
    /** @brief Backend factory binding a graph to its exact test stream. */
    using FixedKGraphFactory = std::function<std::unique_ptr<IGPUGraphCapture>(void *)>;

    /** @brief One way to split a prompt into consecutive prefill chunks (live rows). */
    struct FixedKChunkSchedule
    {
        const char *label;       ///< Diagnostic name.
        std::vector<int> chunks; ///< Live rows per chunk; sums to the prompt length.
    };

    /**
     * @brief Smallest supported prefill bucket covering @p rows.
     * @param rows Positive live rows. @return Physical graph rows used by production.
     */
    inline int fixedKBucketFor(int rows)
    {
        for (int bucket : kSupportedPrefillGraphBucketSizes)
            if (bucket >= rows) return bucket;
        throw std::invalid_argument("Chunk exceeds the supported prefill bucket inventory");
    }

    /**
     * @brief Replay one prompt under several chunk schedules and require identical rows.
     * @param device Physical participant owning every tensor, stream and graph.
     * @param context Backend stage context bound to @p device.
     * @param graph_factory Native CUDA/HIP graph constructor.
     * @param worker Actual worker owning retained captures.
     * @param wide_format Native format for QKV/Z when @p full_half is false.
     * @param k Shared activation width.
     * @param columns QKV, Z, alpha and beta output widths.
     * @param full_half True: all four projections use half-RNE operands (explicit
     *        full-FP16 mode). False: only alpha/beta (default FP16MoEAlphaBeta).
     * @param schedules Chunkings of one prompt; every pair must agree.
     * @param route_counter PerfStats name the fixed-K route must publish.
     */
    inline void runFixedKChunkReplay(
        DeviceId device, IDeviceContext &context, const FixedKGraphFactory &graph_factory,
        IWorkerGPUContext *worker, const QuantizedVerifierFormatCase &wide_format,
        int k, std::array<int, 4> columns, bool full_half,
        std::span<const FixedKChunkSchedule> schedules, const std::string &route_counter)
    {
        ASSERT_FALSE(schedules.empty());
        int prompt = 0;
        for (int rows : schedules.front().chunks) prompt += rows;
        for (const auto &schedule : schedules)
        {
            int total = 0;
            for (int rows : schedule.chunks) total += rows;
            ASSERT_EQ(total, prompt) << schedule.label;
        }
        SCOPED_TRACE(::testing::Message() << "K=" << k << " N=" << columns[0] << "+" << columns[1]
            << "+" << columns[2] << "+" << columns[3] << " full_half=" << full_half
            << " device=" << device.toString());

        auto *backend = getBackendFor(device);
        ASSERT_NE(backend, nullptr);
        ScopedGPUStream stream_owner(device);
        void *stream = stream_owner.get();

        // Prepared engines: alpha/beta (and QKV/Z in full-half mode) derive
        // half-RNE weights from original FP32 sources exactly like setup.
        const auto half = FloatingActivationOperandPrecision::FP16RNE;
        std::array<std::shared_ptr<TensorBase>, 4> weights;
        for (size_t i = 0; i < 4; ++i)
        {
            const uint32_t seed = 74001u + static_cast<uint32_t>(i);
            if (i < 2 && !full_half)
                weights[i] = wide_format.create({size_t(columns[i]), size_t(k)}, seed);
            else
                weights[i] = materializePreparedHalfWeights(*TestTensorFactory::createFP32Random(
                    {size_t(columns[i]), size_t(k)}, -0.125f, 0.125f, seed));
        }
        const std::array<const char *, 4> names{"qkv", "z", "alpha", "beta"};
        std::array<GpuPreparedGemm, 4> prepared;
        for (size_t i = 0; i < 4; ++i)
        {
            const std::string name = std::string("test.fixed_k.") + names[i];
            const bool explicit_half = full_half || i >= 2;
            prepared[i] = explicit_half
                ? makeGpuPreparedFloatingPointGemm(weights[i].get(), device, name, ModelContextId{7400}, half)
                : makeGpuPreparedGemm(weights[i].get(), device, name);
            ASSERT_NE(prepared[i].kernel, nullptr);
        }

        // Host prompt: FP32 rows the production cast rounds to half on device.
        std::vector<float> prompt_rows(size_t(prompt) * k);
        std::mt19937 rng(7474);
        std::uniform_real_distribution<float> values(-0.75f, 0.75f);
        for (auto &value : prompt_rows) value = values(rng);

        // One stage, input and output set per distinct bucket, captured once.
        std::vector<int> buckets;
        for (const auto &schedule : schedules)
            for (int rows : schedule.chunks) buckets.push_back(fixedKBucketFor(rows));
        std::sort(buckets.begin(), buckets.end());
        buckets.erase(std::unique(buckets.begin(), buckets.end()), buckets.end());

        struct BucketGraph
        {
            int rows = 0;
            std::unique_ptr<FP32Tensor> input;
            std::array<std::unique_ptr<FP32Tensor>, 4> outputs;
            std::unique_ptr<GDNProjectionStage> stage;
            std::unique_ptr<IGPUGraphCapture> graph;
        };
        std::map<int, BucketGraph> graphs;
        WorkspaceRequirements requirements;
        for (int bucket : buckets)
        {
            auto &entry = graphs[bucket];
            entry.rows = bucket;
            entry.input = TestTensorFactory::createFP32Zeros({size_t(bucket), size_t(k)});
            ASSERT_TRUE(entry.input->ensureOnDevice(device, stream));
            for (size_t i = 0; i < 4; ++i)
            {
                entry.outputs[i] = TestTensorFactory::createFP32Zeros({size_t(bucket), size_t(columns[i])});
                ASSERT_TRUE(entry.outputs[i]->allocateOnDevice(device, stream));
            }
            GDNProjectionStage::Params params;
            params.device_id = device;
            params.input = entry.input.get();
            params.m = bucket;
            params.k = k;
            params.w_qkv = weights[0].get(); params.output_qkv = entry.outputs[0].get();
            params.n_qkv = columns[0]; params.gemm_qkv = prepared[0].kernel;
            params.w_z = weights[1].get(); params.output_z = entry.outputs[1].get();
            params.n_z = columns[1]; params.gemm_z = prepared[1].kernel;
            params.w_a = weights[2].get(); params.output_a = entry.outputs[2].get();
            params.n_a = columns[2]; params.gemm_a = prepared[2].kernel;
            params.w_b = weights[3].get(); params.output_b = entry.outputs[3].get();
            params.n_b = columns[3]; params.gemm_b = prepared[3].kernel;
            entry.stage = std::make_unique<GDNProjectionStage>(params);
            requirements.merge(entry.stage->getWorkspaceRequirements(bucket, 0, k));
        }
        const auto workspace_bytes = requirements.total_bytes_with_alignment();
        PhysicalMemoryPlanBuilder memory_plan;
        memory_plan.add({.world_rank = 0, .device = device, .total_bytes = workspace_bytes,
            .admission_available_bytes = workspace_bytes}, PhysicalMemoryOwner::ExecutionWorkspace, workspace_bytes);
        auto memory = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(memory_plan.build()), 0);
        DeviceWorkspaceManager workspace(device, workspace_bytes, memory);
        ASSERT_TRUE(workspace.allocate(requirements));
        struct Retirement
        {
            IBackend *backend; DeviceId device; void *stream; std::map<int, BucketGraph> &graphs;
            /** @brief Retire retained work before unbinding its persistent scratch. */
            ~Retirement()
            {
                (void)backend->synchronizeStream(stream, device.ordinal);
                for (auto &[bucket, entry] : graphs)
                {
                    entry.graph.reset();
                    if (entry.stage) entry.stage->unbindWorkspace();
                }
            }
        } retirement{backend, device, stream, graphs};

        PerfStatsCollector::reset();
        for (auto &[bucket, entry] : graphs)
        {
            SCOPED_TRACE(::testing::Message() << "capture bucket=" << bucket);
            entry.stage->bindWorkspace(&workspace);
            ASSERT_TRUE(entry.stage->prepareGraphLaunch(&context, stream));
            TransferEngine::requireDeviceInput(entry.input.get(), device, stream);
            GraphCaptureDependencyLedger::StagePlan plan{
                .stage_identity = entry.stage.get(),
                .stage_name = "fixed_k_chunk_gdn_projection",
                .external_inputs = {entry.input.get()},
                .internal_inputs = {},
                .outputs = {entry.outputs[0].get(), entry.outputs[1].get(),
                            entry.outputs[2].get(), entry.outputs[3].get()},
            };
            GraphCaptureDependencyLedger ledger(device, stream, {std::move(plan)},
                "fixed-K chunk-invariance regression");
            entry.graph = graph_factory(stream);
            ASSERT_NE(entry.graph, nullptr);
            auto transaction = worker
                ? std::make_unique<ScopedBackendGraphCapture>(*worker, *entry.graph,
                    "fixed-K chunk-invariance regression", &ledger)
                : std::make_unique<ScopedBackendGraphCapture>(*entry.graph,
                    "fixed-K chunk-invariance regression", &ledger);
            ASSERT_TRUE(transaction->begin());
            bool success = false;
            {
                ScopedGraphCaptureStage scope(entry.stage.get());
                success = entry.stage->execute(&context);
                if (success) scope.complete();
            }
            transaction->finish();
            ASSERT_TRUE(success);
            ASSERT_TRUE(entry.graph->instantiate());
        }

        // Every captured explicit-half group used the fixed-K route at its bucket.
        double fixed_launches = 0.0;
        for (const auto &record : PerfStatsCollector::snapshot({route_counter}))
        {
            fixed_launches += record.value;
            EXPECT_EQ(record.tags.at("activation"), "fp16_rne");
            EXPECT_EQ(record.tags.at("accumulation"), "fp32");
        }
        EXPECT_GE(fixed_launches, static_cast<double>(graphs.size())) << route_counter;
        for (const char *vendor : {"gemm.cuda_floating_operand_projection_calls", "gemm.rocm_fp16_operand_compute32_calls"})
            for (const auto &record : PerfStatsCollector::snapshot({vendor}))
                EXPECT_NE(record.tags.count("activation") ? record.tags.at("activation") : "", "fp16_rne")
                    << "Ordinary half prefill reached vendor BLAS: " << vendor;
        PerfStatsCollector::reset();

        // Replay every schedule and gather live rows for each explicit projection.
        std::vector<std::array<std::vector<float>, 4>> observed(schedules.size());
        for (size_t s = 0; s < schedules.size(); ++s)
        {
            for (size_t i = 0; i < 4; ++i) observed[s][i].assign(size_t(prompt) * columns[i], 0.0f);
            int cursor = 0;
            for (int live : schedules[s].chunks)
            {
                SCOPED_TRACE(::testing::Message() << schedules[s].label << " rows=" << cursor << "+" << live);
                auto &entry = graphs.at(fixedKBucketFor(live));
                // Poison padded rows with finite values distinct from any live row.
                std::vector<float> chunk(size_t(entry.rows) * k, 1.0e4f + static_cast<float>(s));
                std::memcpy(chunk.data(), prompt_rows.data() + size_t(cursor) * k, size_t(live) * k * sizeof(float));
                ASSERT_TRUE(backend->hostToDeviceOnStream(entry.input->gpu_data_ptr(), chunk.data(),
                    chunk.size() * sizeof(float), device.ordinal, stream));
                for (auto &output : entry.outputs)
                    ASSERT_TRUE(backend->memset(output->gpu_data_ptr(), 0xff,
                        output->numel() * sizeof(float), device.ordinal, stream));
                ASSERT_TRUE(entry.graph->launch());
                for (size_t i = 0; i < 4; ++i)
                    ASSERT_TRUE(backend->deviceToHostOnStream(
                        observed[s][i].data() + size_t(cursor) * columns[i], entry.outputs[i]->gpu_data_ptr(),
                        size_t(live) * columns[i] * sizeof(float), device.ordinal, stream));
                ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
                cursor += live;
            }
        }

        for (size_t i = 0; i < 4; ++i)
        {
            if (!full_half && i < 2) continue;  // native codebooks are outside this contract
            for (size_t s = 1; s < schedules.size(); ++s)
                for (int row = 0; row < prompt; ++row)
                {
                    const float *expected = observed[0][i].data() + size_t(row) * columns[i];
                    const float *actual = observed[s][i].data() + size_t(row) * columns[i];
                    ASSERT_EQ(std::memcmp(expected, actual, size_t(columns[i]) * sizeof(float)), 0)
                        << names[i] << " prompt row " << row << " differs between "
                        << schedules[0].label << " and " << schedules[s].label;
                }
        }
    }
}
