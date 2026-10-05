/**
 * @file GDNProjectionCaptureHarness.h
 * @brief Backend-neutral retained GDN projection grouping regression.
 *
 * Prepared projections own their weight decoders independently. A stage must
 * share activation quantization between different native codebooks without
 * merging incompatible floating formats or sharing concurrently written
 * scratch. Compare each retained graph with independently submitted projection
 * oracles, then alternate large and small graphs over dirty output capacity.
 */
#pragma once

#include <gtest/gtest.h>
#include "GpuPreparedGemmHarness.h"
#include "QuantizedVerifierFormats.h"
#include "ScopedGPUStream.h"
#include "VerifierRowTestInventory.h"
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
#include <bit>
#include <cmath>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace llaminar2::test
{
    /** @brief Distinguish ordinary prompt math from serial-equivalent verification. */
    enum class GDNProjectionRowsPurpose { Ordinary, Verifier };

    /** @brief One immutable retained capture geometry and arithmetic contract. */
    struct GDNProjectionCaptureRows
    {
        int rows; ///< Live rows written by this graph.
        GDNProjectionRowsPurpose purpose; ///< Public projection entrypoint.
    };

    /** @brief Backend factory binding a graph to its exact test stream. */
    using GDNProjectionGraphFactory =
        std::function<std::unique_ptr<IGPUGraphCapture>(void *)>;

    /**
     * @brief Prove mixed native grouping through the real stage and native graphs.
     * @param device Physical participant for every tensor, stream and graph.
     * @param context Backend stage context bound to @p device.
     * @param graph_factory Native CUDA/HIP graph constructor.
     * @param qkv_format Source format owned by the first prepared projection.
     * @param z_format Different source/decoder format owned by the second.
     * @param k Shared activation width.
     * @param columns QKV, Z, alpha and beta output widths.
     * @param cases Complete capture family retained over one workspace.
     */
    inline void runGDNProjectionCaptureCase(
        DeviceId device, IDeviceContext &context,
        const GDNProjectionGraphFactory &graph_factory,
        const QuantizedVerifierFormatCase &qkv_format,
        const QuantizedVerifierFormatCase &z_format,
        int k, std::array<int, 4> columns,
        std::span<const GDNProjectionCaptureRows> cases)
    {
        ASSERT_FALSE(cases.empty());
        SCOPED_TRACE(::testing::Message() << qkv_format.label << "+"
            << z_format.label << " K=" << k << " N=" << columns[0]
            << "+" << columns[1] << " device=" << device.toString());
        const int capacity = std::max_element(cases.begin(), cases.end(),
            [](const auto &a, const auto &b) { return a.rows < b.rows; })->rows;
        auto *backend = getBackendFor(device);
        ASSERT_NE(backend, nullptr);
        ScopedGPUStream stream_owner(device);
        void *stream = stream_owner.get();

        std::array<std::unique_ptr<TensorBase>, 4> weights;
        weights[0] = qkv_format.create({size_t(columns[0]), size_t(k)}, 98901);
        weights[1] = z_format.create({size_t(columns[1]), size_t(k)}, 98902);
        weights[2] = TestTensorFactory::createFP32Random(
            {size_t(columns[2]), size_t(k)}, -0.125f, 0.125f, 98903);
        weights[3] = TestTensorFactory::createFP32Random(
            {size_t(columns[3]), size_t(k)}, -0.125f, 0.125f, 98904);
        const std::array<const char *, 4> names{"qkv", "z", "alpha", "beta"};
        std::array<GpuPreparedGemm, 4> prepared;
        std::array<std::unique_ptr<FP32Tensor>, 4> outputs;
        for (size_t i = 0; i < prepared.size(); ++i)
        {
            const std::string name = std::string("test.gdn.mixed.") + names[i];
            prepared[i] = i < 2
                ? makeGpuPreparedGemm(weights[i].get(), device, name)
                : makeGpuPreparedFloatingPointGemm(weights[i].get(), device, name);
            ASSERT_NE(prepared[i].kernel, nullptr);
            outputs[i] = TestTensorFactory::createFP32Zeros(
                {size_t(capacity), size_t(columns[i])});
            ASSERT_TRUE(outputs[i]->allocateOnDevice(device, stream));
        }
        auto input = TestTensorFactory::createFP32Random(
            {size_t(capacity), size_t(k)}, -0.75f, 0.75f, 98905);
        ASSERT_TRUE(input->ensureOnDevice(device, stream));

        std::vector<std::unique_ptr<GDNProjectionStage>> stages;
        WorkspaceRequirements requirements;
        for (const auto row_case : cases)
        {
            GDNProjectionStage::Params params;
            params.device_id = device;
            params.input = input.get();
            params.m = row_case.rows;
            params.k = k;
            params.w_qkv = weights[0].get(); params.output_qkv = outputs[0].get();
            params.n_qkv = columns[0]; params.gemm_qkv = prepared[0].kernel;
            params.w_z = weights[1].get(); params.output_z = outputs[1].get();
            params.n_z = columns[1]; params.gemm_z = prepared[1].kernel;
            params.w_a = weights[2].get(); params.output_a = outputs[2].get();
            params.n_a = columns[2]; params.gemm_a = prepared[2].kernel;
            params.w_b = weights[3].get(); params.output_b = outputs[3].get();
            params.n_b = columns[3]; params.gemm_b = prepared[3].kernel;
            params.force_decode_equivalent_verifier_prefill =
                row_case.purpose == GDNProjectionRowsPurpose::Verifier;
            stages.push_back(std::make_unique<GDNProjectionStage>(params));
            requirements.merge(stages.back()->getWorkspaceRequirements(row_case.rows, 0, k));
        }
        DeviceWorkspaceManager workspace(device,
            requirements.total_bytes_with_alignment() + 4096u);
        ASSERT_TRUE(workspace.allocate(requirements));
        std::vector<std::unique_ptr<IGPUGraphCapture>> graphs;
        std::vector<std::array<std::vector<float>, 4>> expected(cases.size());
        struct Retirement
        {
            IBackend *backend;
            DeviceId device;
            void *stream;
            std::vector<std::unique_ptr<IGPUGraphCapture>> &graphs;
            std::vector<std::unique_ptr<GDNProjectionStage>> &stages;
            /** @brief Retire retained work before unbinding its persistent scratch. */
            ~Retirement()
            {
                (void)backend->synchronizeStream(stream, device.ordinal);
                graphs.clear();
                for (auto &stage : stages)
                    stage->unbindWorkspace();
                stages.clear();
            }
        } retirement{backend, device, stream, graphs, stages};

        for (size_t index = 0; index < cases.size(); ++index)
        {
            const auto row_case = cases[index];
            SCOPED_TRACE(::testing::Message() << "rows=" << row_case.rows
                << " verifier=" << (row_case.purpose == GDNProjectionRowsPurpose::Verifier));
            auto &stage = *stages[index];
            stage.bindWorkspace(&workspace);
            ASSERT_TRUE(stage.prepareGraphLaunch(&context, stream));
            // Independent projection submission is only an untimed test oracle.
            // Production execution below always captures the complete GDN stage.
            for (size_t i = 0; i < prepared.size(); ++i)
            {
                auto *kernel = prepared[i].kernel;
                const std::vector<ITensorGemm::TensorProjectionDesc> projection{
                    {kernel, outputs[i].get(), columns[i], nullptr, names[i]}};
                const bool success = row_case.purpose == GDNProjectionRowsPurpose::Verifier
                    ? kernel->multiply_fused_verifier_rows_decode_equivalent(
                        input.get(), projection, row_case.rows, k, nullptr, &workspace)
                    : kernel->multiply_fused_tensor(
                        input.get(), projection, row_case.rows, k, nullptr, &workspace);
                ASSERT_TRUE(success);
                expected[index][i].resize(size_t(row_case.rows) * columns[i]);
                ASSERT_TRUE(backend->deviceToHostOnStream(expected[index][i].data(),
                    outputs[i]->gpu_data_ptr(), expected[index][i].size() * sizeof(float),
                    device.ordinal, stream));
            }
            ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
            PerfStatsCollector::reset();
            TransferEngine::requireDeviceInput(input.get(), device, stream);
            GraphCaptureDependencyLedger::StagePlan plan{
                .stage_identity = &stage,
                .stage_name = "mixed_native_gdn_projection",
                .external_inputs = {input.get()},
                .internal_inputs = {},
                .outputs = {outputs[0].get(), outputs[1].get(), outputs[2].get(), outputs[3].get()},
            };
            GraphCaptureDependencyLedger ledger(device, stream,
                {std::move(plan)}, "mixed native GDN projection regression");
            auto graph = graph_factory(stream);
            ASSERT_NE(graph, nullptr);
            ScopedBackendGraphCapture transaction(*graph,
                "mixed native GDN projection regression", &ledger);
            ASSERT_TRUE(transaction.begin());
            bool success = false;
            {
                ScopedGraphCaptureStage scope(&stage);
                success = stage.execute(&context);
                if (success) scope.complete();
            }
            transaction.finish();
            ASSERT_TRUE(success);
            ASSERT_TRUE(graph->instantiate());
            bool found_pair = false;
            for (const auto &record : PerfStatsCollector::snapshot({"kernel.gdn_projection_route"}))
            {
                if (record.tags.at("names") == "qkv+z")
                {
                    EXPECT_EQ(record.tags.at("projections"), "2");
                    EXPECT_EQ(record.tags.at("codebooks"),
                        std::to_string(qkv_format.device_execution_codebook_id) + "+" +
                        std::to_string(z_format.device_execution_codebook_id));
                    found_pair = true;
                }
            }
            EXPECT_TRUE(found_pair) << "Native codebooks must retain one shared-quantization subgroup";
            if (row_case.purpose == GDNProjectionRowsPurpose::Ordinary && row_case.rows > 16)
            {
                const std::string counter = device.is_cuda()
                    ? "cuda_fused_projection_stream_pool_calls"
                    : "rocm_fused_projection_stream_pool_calls";
                bool found_concurrent = false;
                for (const auto &record : PerfStatsCollector::snapshot({"kernel." + counter}))
                    found_concurrent |= record.name == counter &&
                        record.tags.at("mode") == "prefill" &&
                        record.tags.at("projections") == "2" &&
                        record.tags.at("streams") == "2";
                EXPECT_TRUE(found_concurrent) << "Retained graph must contain the persistent two-stream fork";
            }
            graphs.push_back(std::move(graph));
        }

        // Reverse the capture order on alternate rounds; the same largest arena
        // is reused after both larger and smaller graphs, without recapturing.
        for (int round = 0; round < 4; ++round)
        {
            for (size_t step = 0; step < cases.size(); ++step)
            {
                const size_t index = (round & 1) ? step : cases.size() - 1 - step;
                SCOPED_TRACE(::testing::Message() << "round=" << round
                    << " rows=" << cases[index].rows);
                for (auto &output : outputs)
                    ASSERT_TRUE(backend->memset(output->gpu_data_ptr(), 0xff,
                        output->numel() * sizeof(float), device.ordinal, stream));
                ASSERT_TRUE(graphs[index]->launch());
                std::array<std::vector<float>, 4> actual;
                for (size_t i = 0; i < outputs.size(); ++i)
                {
                    actual[i].resize(outputs[i]->numel());
                    ASSERT_TRUE(backend->deviceToHostOnStream(actual[i].data(),
                        outputs[i]->gpu_data_ptr(), actual[i].size() * sizeof(float),
                        device.ordinal, stream));
                }
                ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
                for (size_t i = 0; i < outputs.size(); ++i)
                {
                    const auto live = expected[index][i].size();
                    ASSERT_EQ(std::memcmp(actual[i].data(), expected[index][i].data(),
                        live * sizeof(float)), 0) << names[i];
                    for (size_t row = live; row < actual[i].size(); ++row)
                        ASSERT_EQ(std::bit_cast<uint32_t>(actual[i][row]), 0xffffffffu)
                            << names[i] << " wrote inactive output element " << row;
                }
            }
        }
        PerfStatsCollector::reset();
    }
}
