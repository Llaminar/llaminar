/**
 * @file Test__CPUPipelinePrefillGeometry.cpp
 * @brief Prove CPU pipeline execution never mistakes GPU transport padding for tokens.
 *
 * The real DeviceGraphOrchestrator, arena, forward cache and chunk scheduler
 * execute a tiny row probe. No model or accelerator is needed: the defect is at
 * request admission, before model mathematics. Head and tail participants must
 * agree on the logical row frontier across first use, reuse and request reset.
 */
#include <gtest/gtest.h>
#include "backends/BackendManager.h"
#include "backends/ComputeBackend.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "models/qwen/QwenStandardGraph.h"
#include "mocks/MockComputeStage.h"
#include "tensors/Tensors.h"

#include <algorithm>
#include <memory>
#include <vector>

namespace llaminar2::test
{
    /** @brief Test graph that exposes admitted rows without loading model weights. */
    class CPUPipelineRowProbe final : public QwenStandardGraph
    {
    public:
        /** @param config Small CPU arena and pipeline geometry. */
        explicit CPUPipelineRowProbe(const GraphConfig &config)
            : QwenStandardGraph(config, nullptr) {}

        /** @return True: this protocol probe has no external weights. */
        bool isInitialized() const override { return true; }

        /**
         * @brief Bind one probe to the actual cached invocation's row count.
         * @param input Production admission, including stable token storage.
         * @param output Publish the same arena-owned hidden/logit interfaces.
         * @param first_layer Unused model layer bound for this protocol probe.
         * @param last_layer Unused model layer bound for this protocol probe.
         * @param has_embedding Whether rows originate from admitted token IDs.
         * @param has_lm_head Whether this participant publishes terminal logits.
         * @return Executable CPU graph; no model math is substituted in parity.
         */
        ComputeGraph buildPartialForwardGraph(
            const ForwardInput &input, ForwardOutput &output,
            int first_layer, int last_layer, bool has_embedding,
            bool has_lm_head) override
        {
            (void)first_layer;
            (void)last_layer;
            (void)has_lm_head;
            auto *hidden = dynamic_cast<FP32Tensor *>(buffers_.current_hidden);
            auto *logits = dynamic_cast<FP32Tensor *>(buffers_.logits);
            if (!hidden || !logits)
                throw std::logic_error("CPU row probe requires FP32 arena outputs");
            const int rows = input.seq_len;
            const int *tokens = input.token_ids;
            auto stage = std::make_unique<testing::MockComputeStage>();
            stage->setOnExecute([=, this](IDeviceContext *)
            {
                executed_rows.push_back(rows);
                float *values = hidden->mutable_data();
                if (has_embedding)
                {
                    for (int row = 0; row < rows; ++row)
                        std::fill_n(values + row * config_.d_model,
                                    config_.d_model, static_cast<float>(tokens[row]));
                }
                // Terminal selection deliberately follows the admitted extent.
                // A padded invocation therefore exposes the bug directly.
                logits->mutable_data()[0] = values[(rows - 1) * config_.d_model];
            });
            ComputeGraph graph;
            graph.addNode("cpu_pipeline_row_probe", std::move(stage), DeviceId::cpu());
            output.hidden = hidden;
            output.logits = logits;
            return graph;
        }

        std::vector<int> executed_rows; ///< Actual executor work, not metadata estimates.
    };

    /** @brief Cover producer and consumer CPU stages with inexact transport buckets. */
    TEST(CPUPipelinePrefillGeometry, ExecutesOnlyRealRowsAcrossReuseAndReset)
    {
        initCPUBackend(-1);
        DeviceManager::instance().initialize(-1);
        GraphConfig config;
        config.d_model = 32;
        config.d_ff = config.d_ff_local = 64;
        config.n_heads = config.local_n_heads = 1;
        config.n_kv_heads = config.local_n_kv_heads = 1;
        config.head_dim = 32;
        config.n_layers = config.total_n_layers = 1;
        config.vocab_size = config.vocab_local = 32;
        config.max_seq_len = 256;
        config.default_device = DeviceId::cpu();

        for (const bool head : {true, false})
        {
            SCOPED_TRACE(head ? "CPU producer" : "CPU consumer");
            auto graph = std::make_shared<CPUPipelineRowProbe>(config);
            DeviceGraphOrchestrator runner(graph, nullptr);
            runner.setPPStageConfig({.first_layer = 0, .last_layer = 1,
                                     .has_embedding = head, .has_lm_head = !head});
            ASSERT_TRUE(runner.initializeInferenceStateFromArena(1, 256, DeviceId::cpu()));
            FP32Tensor external({256u, 32u});
            for (const int rows : {17, 17, 33, 139})
            {
                SCOPED_TRACE(rows);
                runner.clear_cache();
                const int bucket = rows <= 64 ? 64 : 256;
                std::vector<int> tokens(static_cast<size_t>(rows));
                for (int row = 0; row < rows; ++row)
                {
                    tokens[row] = row % 31 + 1;
                    std::fill_n(external.mutable_data() + row * 32, 32,
                                static_cast<float>(tokens[row]));
                }
                if (!head)
                    runner.setHiddenState(&external);
                PrefillChunkSchedulerPolicy policy;
                policy.bucket_sizes = {bucket};
                policy.fixed_chunk_real_tokens = bucket;
                policy.real_token_count = rows;
                ASSERT_TRUE(runner.forwardPrefillChunkSchedule(
                    tokens.data(), rows, policy, 0, true));
                ASSERT_FALSE(graph->executed_rows.empty());
                EXPECT_EQ(graph->executed_rows.back(), rows);
                EXPECT_EQ(runner.get_position(), rows);
                if (!head)
                {
                    ASSERT_NE(runner.logits(), nullptr);
                    EXPECT_FLOAT_EQ(runner.logits()[0], static_cast<float>(tokens.back()));
                }
                else
                {
                    auto *hidden = dynamic_cast<const FP32Tensor *>(runner.getHiddenState());
                    ASSERT_NE(hidden, nullptr);
                    EXPECT_FLOAT_EQ(hidden->data()[(rows - 1) * 32], static_cast<float>(tokens.back()));
                    for (int row = rows; row < bucket; ++row)
                        EXPECT_FLOAT_EQ(hidden->data()[row * 32], 0.0f);
                }
            }
        }
    }
}
