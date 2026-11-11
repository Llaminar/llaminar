/**
 * @file Test__PipelineRequestAuthorityMPI.cpp
 * @brief Model-free proof that pipeline serving and sampling share the tail owner.
 *
 * Real MPI commands, GlobalOrchestrator transfers and OrchestrationRunner
 * sampling run around tiny deterministic CPU stages. Only the terminal stage
 * exposes logits; copying them to the command rank cannot hide a misplaced
 * authority. Reversing stage owners prevents rank zero/last-rank heuristics.
 * Each model-free stage explicitly admits the host response law and records
 * balanced begin/end forwarding; no sparse-query numerical capability is mocked.
 */
#include "execution/global/GlobalOrchestrator.h"
#include "../../../utils/HostedPipelineTestAdmission.h"
#include "execution/global_pp/GlobalPPRankPlanBuilder.h"
#include "execution/runner/OrchestrationRunner.h"
#include "utils/MPIContext.h"
#include "../../../utils/TestTensorFactory.h"

#include <gtest/gtest.h>
#include <mpi.h>
#include <array>
#include <optional>
#include <vector>

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
            /**
             * @brief Admit the real immutable response policy in this no-query-consumer witness.
             * @param request Existing response budget/depth and optional prefix provenance.
             * @return True only for a valid, idle admission of a closed query program.
             * No QSA/indexer graph exists here; the production registry must
             * still forward admission to every participant instead of dropping it.
             */
            bool admitSparseQueryExecutionProgram(const DeviceGenerationAdmissionRequest &request) override
            {
                if (active_admission_ || !request.valid()) return false;
                (void)resolveSparseQueryExecutionProgram(request);
                active_admission_.emplace(request);
                ++admission_begins_;
                return true;
            }
            /** @brief End only an accepted response law; repeated/rollback cleanup is idempotent. */
            void endSparseQueryExecutionProgram() noexcept override
            {
                if (!active_admission_) return;
                active_admission_.reset();
                ++admission_ends_;
            }
            /** @return Actual successful local admission count, retained across request resets. */
            int admissionBegins() const noexcept { return admission_begins_; }
            /** @return Actual local response cleanup count, retained across request resets. */
            int admissionEnds() const noexcept { return admission_ends_; }
            /** @return Whether the fixture still owns an unclosed response law. */
            bool hasActiveAdmission() const noexcept { return active_admission_.has_value(); }
            /** Execute one stage; the real global runner transfers the hidden row. */
            bool forward(const int *tokens, int count) override
            {
                if (terminal_ && (!input_ || tokens))
                    return false;
                if (!terminal_ && !tokens)
                    return false;
                return executeRows(count);
            }
            /**
             * @brief Admit original prefill token metadata alongside the transferred hidden rows.
             * @param tokens Exact immutable host row forwarded to every prefill participant.
             * @param count Positive number of live prompt rows.
             * @return True after the stage executes with its original activation input.
             *
             * Prefill carries token ingress even on a nonembedding stage. That
             * metadata does not replace the preceding stage's hidden values;
             * ordinary CPU decode retains the null-token guard in forward().
             */
            bool forwardPrefill(const int *tokens, int count) override
            {
                if (!tokens || count <= 0 || (terminal_ &&
                        (!input_ || input_->numel() < static_cast<size_t>(count) * 4)))
                    return false;
                const auto *hidden = terminal_ ? static_cast<const float *>(input_->data()) : nullptr;
                if (terminal_ && !hidden) return false;
                if (!executeRows(count)) return false;
                last_prefill_tokens_.assign(tokens, tokens + count);
                last_prefill_hidden_.clear();
                if (hidden) last_prefill_hidden_.assign(hidden, hidden + static_cast<size_t>(count) * 4);
                ++prefill_count_;
                return true;
            }
            /** @return Successful explicit prefill count, retained across request resets. */
            int prefillCount() const noexcept { return prefill_count_; }
            /** @return Exact latest accepted token ingress; nonembedding stages keep the same row. */
            const std::vector<int> &prefillTokens() const noexcept { return last_prefill_tokens_; }
            /** @return Actual live transferred CPU values observed by the latest terminal prefill. */
            const std::vector<float> &prefillHiddenValues() const noexcept { return last_prefill_hidden_; }
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
            /**
             * @brief Execute the fixture's deterministic row arithmetic for either phase.
             * @param count Positive live row count, independent of retained input capacity.
             * @return True after output publication and logical position advancement.
             */
            bool executeRows(int count)
            {
                if (count <= 0) return false;
                hidden_ = TestTensorFactory::createFP32Ones({static_cast<size_t>(count), 4});
                position_ += count;
                return true;
            }
            bool terminal_;
            int prefill_count_ = 0;
            std::vector<int> last_prefill_tokens_;
            std::vector<float> last_prefill_hidden_;
            std::optional<DeviceGenerationAdmissionRequest> active_admission_;
            int admission_begins_ = 0;
            int admission_ends_ = 0;
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
            const auto local_plan = GlobalPPRankPlanBuilder::build(global.topology, rank);
            ASSERT_EQ(local_plan.executeStages().size(), 1u);
            StageRunnerEntry entry;
            entry.action = *local_plan.executeStages().front();
            entry.stage_id = entry.action.stage_id;
            entry.domain_name = entry.action.domain_name;
            auto local_stage = std::make_unique<PipelineStage>(rank == tail);
            const PipelineStage *local_observer = local_stage.get();
            entry.runner = std::move(local_stage);
            global.stage_runners.push_back(std::move(entry));

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
                std::make_unique<GlobalOrchestrator>(admitHostedPipelineFixture(std::move(global), 3)), mpi);

            // All ranks check before entering the worker loop, so an unfixed
            // owner fails quickly rather than stranding a follower in Bcast.
            ASSERT_EQ(runner.coordinatedRootRank(), tail);
            runner.setMPICoordinatedMode(true);
            if (rank != tail)
            {
                runner.runMPIWorkerLoop();
                EXPECT_EQ(local_observer->prefillCount(), 2);
                EXPECT_EQ(local_observer->prefillTokens(), (std::vector<int>{1, 3, 1}));
                EXPECT_TRUE(local_observer->prefillHiddenValues().empty());
                EXPECT_EQ(local_observer->admissionBegins(), 16);
                EXPECT_EQ(local_observer->admissionEnds(), 16);
                EXPECT_FALSE(local_observer->hasActiveAdmission());
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
                EXPECT_EQ(local_observer->prefillCount(), request + 1);
                EXPECT_EQ(local_observer->prefillTokens(), (std::vector<int>{1, 3, 1}));
                EXPECT_EQ(local_observer->prefillHiddenValues(), std::vector<float>(12, 1.0f));
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
            EXPECT_EQ(local_observer->admissionBegins(), 16);
            EXPECT_EQ(local_observer->admissionEnds(), 16);
            EXPECT_FALSE(local_observer->hasActiveAdmission());
        }
    }

    /** @brief The explicit no-consumer participant rejects invalid/replacement laws and balances cleanup. */
    TEST(PipelineRequestAuthorityMPI, SparseQueryAdmissionIsExplicitAndBalanced)
    {
        PipelineStage stage(false);
        EXPECT_FALSE(stage.admitSparseQueryExecutionProgram({}));
        EXPECT_EQ(stage.admissionBegins(), 0);
        EXPECT_EQ(stage.admissionEnds(), 0);
        DeviceGenerationAdmissionRequest request{
            .request_count = 1, .max_new_tokens = 8,
            .depth_policy = sampling_math::DeviceGenerationPolicy::ordinary()};
        request.initial_target_counts.emplace(std::vector<int>{3});
        ASSERT_TRUE(stage.admitSparseQueryExecutionProgram(request));
        EXPECT_TRUE(stage.hasActiveAdmission());
        EXPECT_FALSE(stage.admitSparseQueryExecutionProgram(request));
        EXPECT_EQ(stage.admissionBegins(), 1);
        stage.endSparseQueryExecutionProgram();
        stage.endSparseQueryExecutionProgram();
        EXPECT_FALSE(stage.hasActiveAdmission());
        EXPECT_EQ(stage.admissionEnds(), 1);
        ASSERT_TRUE(stage.admitSparseQueryExecutionProgram(request));
        stage.endSparseQueryExecutionProgram();
        EXPECT_EQ(stage.admissionBegins(), 2);
        EXPECT_EQ(stage.admissionEnds(), 2);
    }

    /**
     * @brief Prefill accepts token metadata and hidden input while CPU decode retains its own ingress law.
     *
     * The original failure called the inherited prefill-to-forward adapter,
     * which applied the terminal CPU decode token guard to valid prefill ingress.
     * Exercise that exact phase boundary without entering the MPI command loop.
     * The containing integration test is explicitly registered in production preflight.
     */
    TEST(PipelineRequestAuthorityMPI, ExplicitPrefillSeparatesTokenIngressFromHiddenInput)
    {
        const std::array<int, 3> tokens{1, 3, 1};
        PipelineStage terminal(true);
        EXPECT_FALSE(terminal.forwardPrefill(tokens.data(), 3));
        auto input = TestTensorFactory::createFP32({3, 4});
        TestTensorFactory::fillValue(input.get(), 7.25f);
        terminal.setHiddenState(input.get());
        EXPECT_FALSE(terminal.forwardPrefill(nullptr, 3));
        EXPECT_FALSE(terminal.forwardPrefill(tokens.data(), 0));
        EXPECT_FALSE(terminal.forwardPrefill(tokens.data(), 4));
        EXPECT_EQ(terminal.prefillCount(), 0);
        EXPECT_EQ(terminal.get_position(), 0);
        EXPECT_FALSE(terminal.forward(tokens.data(), 3));
        ASSERT_TRUE(terminal.forwardPrefill(tokens.data(), 3));
        EXPECT_EQ(terminal.prefillCount(), 1);
        EXPECT_EQ(terminal.prefillTokens(), (std::vector<int>{1, 3, 1}));
        EXPECT_EQ(terminal.prefillHiddenValues(), std::vector<float>(12, 7.25f));
        EXPECT_EQ(terminal.get_position(), 3);
        EXPECT_FALSE(terminal.forward(tokens.data(), 1));
        EXPECT_TRUE(terminal.forward(nullptr, 1));
        EXPECT_EQ(terminal.get_position(), 4);
        terminal.clear_cache();
        EXPECT_FALSE(terminal.forwardPrefill(tokens.data(), 3));
        EXPECT_FALSE(terminal.forward(nullptr, 1));
        TestTensorFactory::fillValue(input.get(), -3.0f);
        const std::array<int, 3> replay_tokens{3, 1, 2};
        terminal.setHiddenState(input.get());
        ASSERT_TRUE(terminal.forwardPrefill(replay_tokens.data(), 3));
        EXPECT_EQ(terminal.prefillCount(), 2);
        EXPECT_EQ(terminal.prefillTokens(), (std::vector<int>{3, 1, 2}));
        EXPECT_EQ(terminal.prefillHiddenValues(), std::vector<float>(12, -3.0f));
        EXPECT_EQ(terminal.get_position(), 3);

        PipelineStage head(false);
        EXPECT_FALSE(head.forwardPrefill(nullptr, 3));
        ASSERT_TRUE(head.forwardPrefill(tokens.data(), 3));
        EXPECT_EQ(head.prefillTokens(), (std::vector<int>{1, 3, 1}));
        EXPECT_TRUE(head.prefillHiddenValues().empty());
        EXPECT_FALSE(head.forward(nullptr, 1));
        EXPECT_TRUE(head.forward(tokens.data(), 1));
    }

    TEST(PipelineRequestAuthorityMPI, RankOneTailStochastic) { provePipeline(1, 0.7f); }
    TEST(PipelineRequestAuthorityMPI, RankZeroTailStochastic) { provePipeline(0, 0.7f); }
    TEST(PipelineRequestAuthorityMPI, RankOneTailGreedy) { provePipeline(1, 0.0f); }
    TEST(PipelineRequestAuthorityMPI, RankZeroTailGreedy) { provePipeline(0, 0.0f); }
}
