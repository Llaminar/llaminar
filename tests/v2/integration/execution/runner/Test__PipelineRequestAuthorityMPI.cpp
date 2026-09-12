/**
 * @file Test__PipelineRequestAuthorityMPI.cpp
 * @brief Model-free proof that pipeline serving and sampling share the tail owner.
 *
 * Real MPI commands, GlobalOrchestrator transfers and OrchestrationRunner
 * sampling run around tiny deterministic CPU stages. Only the terminal stage
 * exposes logits; copying them to the command rank cannot hide a misplaced
 * authority. Reversing stage owners prevents rank zero/last-rank heuristics.
 */
#include "execution/global/GlobalOrchestrator.h"
#include "execution/runner/OrchestrationRunner.h"
#include "utils/MPIContext.h"
#include "../../../utils/TestTensorFactory.h"

#include <gtest/gtest.h>
#include <mpi.h>
#include <array>

namespace llaminar2::test
{
    namespace
    {
        /** Deterministic stage with explicit hidden input and terminal-only logits. */
        class PipelineStage final : public IInferenceRunner
        {
        public:
            /** @param terminal Whether this stage owns the vocabulary head. */
            explicit PipelineStage(bool terminal) : terminal_(terminal) {}
            /** Execute one stage; the real global runner transfers the hidden row. */
            bool forward(const int *tokens, int count) override
            {
                if (terminal_ && (!input_ || tokens))
                    return false;
                if (!terminal_ && !tokens)
                    return false;
                hidden_ = TestTensorFactory::createFP32Ones({static_cast<size_t>(count), 4});
                position_ += count;
                return true;
            }
            /** Nonterminal ranks deliberately have no host logits. */
            const float *logits() const override { return terminal_ ? logits_.data() : nullptr; }
            /** Complete tiny vocabulary, identical to the outer plan. */
            int vocab_size() const override { return 4; }
            /** Reset request data without changing stage ownership. */
            void clear_cache() override { position_ = 0; input_ = nullptr; }
            /** Current logical output position for position-keyed sampling. */
            int get_position() const override { return position_; }
            /** Production orchestration executes the stages as a CPU graph. */
            ExecutionPath executionPath() const override { return ExecutionPath::GRAPH; }
            /** No model or tokenizer file is needed by this protocol regression. */
            const char *architecture() const override { return "pipeline_protocol"; }
            /** The terminal CPU stage has an exact greedy candidate. */
            int sampleGreedyOnDevice() override { return terminal_ ? 2 : -1; }
            /** Expose the local stage output to the real activation transfer. */
            TensorBase *getHiddenState() override { return hidden_.get(); }
            /** Read-only view of the same stage output. */
            const TensorBase *getHiddenState() const override { return hidden_.get(); }
            /** Borrow the transfer-owned input only until the next stage execution. */
            void setHiddenState(TensorBase *input) override { input_ = input; }
            /** The fixture has no EOS token, so every requested step executes. */
            bool configureMTPRequestStopTokens(const std::vector<int32_t> &) override { return true; }
        private:
            bool terminal_;
            int position_ = 0;
            TensorBase *input_ = nullptr;
            std::unique_ptr<FP32Tensor> hidden_;
            std::array<float, 4> logits_{-2.0f, -1.0f, 3.0f, 0.5f};
        };

        /** Exercise the same command loop with either physical rank owning the tail. */
        void provePipeline(int tail, float temperature)
        {
            int rank = -1, size = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);
            ASSERT_EQ(size, 2);
            auto mpi = std::make_shared<MPIContext>(rank, size, MPI_COMM_WORLD);
            std::vector<GlobalPPStageSpec> stages;
            for (int stage = 0; stage != 2; ++stage)
            {
                GlobalPPStageSpec spec;
                spec.stage_id = stage;
                spec.first_layer = spec.last_layer = stage;
                spec.has_embedding = stage == 0;
                spec.has_lm_head = stage == 1;
                spec.owning_rank = stage == 1 ? tail : 1 - tail;
                spec.devices = {GlobalDeviceAddress::cpu(spec.owning_rank)};
                stages.push_back(std::move(spec));
            }
            GlobalOrchestrator::Config global;
            global.topology = GlobalPPTopology::build(std::move(stages), 2, 2);
            global.rank = rank;
            global.world_size = size;
            global.mpi_ctx = mpi.get();
            global.vocab_size = 4;
            global.d_model = 4;
            global.rank_runner = std::make_unique<PipelineStage>(rank == tail);

            RankExecutionPlan plan;
            plan.rank = rank;
            plan.hostname = "localhost";
            plan.primary_device = GlobalDeviceAddress::cpu(rank);
            plan.has_lm_head = rank == tail;
            plan.has_embedding = rank != tail;
            OrchestrationConfig config;
            config.prefix_cache.enabled = false;
            config.prefix_cache.storage_mode = PrefixCacheStorageMode::Disabled;
            config.mtp.graph_capacity_draft_tokens = 0;
            plan.runtime.prefix_cache = config.prefix_cache;
            plan.runtime.mtp = config.mtp;
            OrchestrationRunner runner(config, plan,
                std::make_unique<GlobalOrchestrator>(std::move(global)), mpi);

            // All ranks check before entering the worker loop, so an unfixed
            // owner fails quickly rather than stranding a follower in Bcast.
            ASSERT_EQ(runner.coordinatedRootRank(), tail);
            runner.setMPICoordinatedMode(true);
            if (rank != tail)
            {
                runner.runMPIWorkerLoop();
                return;
            }
            SamplingParams params;
            params.temperature = temperature;
            params.seed = 4242;
            params.top_k = 4;
            params.top_p = 0.9f;
            runner.setSamplingParams(params);
            // Re-admission exercises both the initial prefill sample and
            // history-bearing decode. Shutdown always releases the worker.
            for (int request = 0; request != 2; ++request)
            {
                runner.clearCache();
                const auto prefill = runner.prefill({1, 3, 1});
                EXPECT_TRUE(prefill) << runner.lastError();
                if (!prefill)
                    break;
                for (int token = 0; token != 8; ++token)
                {
                    const auto result = runner.decodeStep();
                    EXPECT_TRUE(result.success()) << result.error;
                    EXPECT_EQ(result.tokens.size(), 1u);
                    if (!result.success())
                        break;
                    if (temperature == 0.0f)
                        EXPECT_EQ(result.tokens.front(), 2);
                }
            }
            runner.shutdownMPIWorkers();
        }
    }

    TEST(PipelineRequestAuthorityMPI, RankOneTailStochastic) { provePipeline(1, 0.7f); }
    TEST(PipelineRequestAuthorityMPI, RankZeroTailStochastic) { provePipeline(0, 0.7f); }
    TEST(PipelineRequestAuthorityMPI, RankOneTailGreedy) { provePipeline(1, 0.0f); }
    TEST(PipelineRequestAuthorityMPI, RankZeroTailGreedy) { provePipeline(0, 0.0f); }
}
