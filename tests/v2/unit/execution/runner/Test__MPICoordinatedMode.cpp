/**
 * @file Test__MPICoordinatedMode.cpp
 * @brief Unit tests for MPI coordinated mode in OrchestrationRunner
 *
 * Tests the MPI worker loop protocol where rank 0 broadcasts commands
 * to non-root ranks, enabling them to participate in inference collectives
 * (AllreduceStage) in lockstep.
 *
 * Uses a RecordingMPIContext that records all broadcast calls (data + types)
 * so we can verify the protocol without real MPI.
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "execution/runner/OrchestrationRunner.h"
#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include "config/OrchestrationConfig.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "backends/GlobalDeviceAddress.h"
#include "interfaces/IMPIContext.h"

using namespace llaminar2;
using namespace testing;

namespace
{

    // =========================================================================
    // RecordingMPIContext — Records broadcast calls for test verification
    // =========================================================================

    /**
     * @brief MPI context mock that records all broadcast data for verification.
     *
     * Unlike the generic MockMPIContext which just counts calls, this mock
     * captures the actual data passed to broadcast_int32() and broadcast()
     * so tests can verify the exact protocol bytes.
     */
    class RecordingMPIContext : public IMPIContext
    {
    public:
        struct BroadcastRecord
        {
            enum class Type
            {
                INT32,
                FLOAT
            };
            Type type;
            std::vector<int32_t> int_data;
            std::vector<float> float_data;
            int root;
        };

        RecordingMPIContext(int rank, int world_size)
            : rank_(rank), world_size_(world_size)
        {
        }

        // Identity
        int rank() const override { return rank_; }
        int world_size() const override { return world_size_; }
        bool is_root() const override { return rank_ == 0; }
        MPI_Comm communicator() const override { return MPI_COMM_NULL; }

        // Broadcast — records data
        void broadcast(float *data, size_t count, int root) const override
        {
            BroadcastRecord rec;
            rec.type = BroadcastRecord::Type::FLOAT;
            rec.float_data.assign(data, data + count);
            rec.root = root;
            broadcasts_.push_back(std::move(rec));
        }

        void broadcast_int32(int32_t *data, size_t count, int root) const override
        {
            BroadcastRecord rec;
            rec.type = BroadcastRecord::Type::INT32;
            rec.int_data.assign(data, data + count);
            rec.root = root;
            broadcasts_.push_back(std::move(rec));
        }

        // Test inspection
        const std::vector<BroadcastRecord> &broadcasts() const { return broadcasts_; }
        size_t broadcastCount() const { return broadcasts_.size(); }
        size_t barrierCount() const { return barrier_count_; }
        void clearRecords() { broadcasts_.clear(); }

        // =====================================================================
        // Remaining IMPIContext stubs (no-ops)
        // =====================================================================

        void barrier() const override { ++barrier_count_; }
        void allreduce_sum(const float *send, float *recv, size_t count) const override
        {
            std::memcpy(recv, send, count * sizeof(float));
        }
        void allreduce_sum_inplace(float *, size_t) const override {}
        void allreduce_q8_1_inplace(Q8_1Block *, size_t) const override {}
        void allreduce_q16_1_inplace(Q16_1Block *, size_t) const override {}
        void allreduce_fp16_inplace(uint16_t *, size_t) const override {}
        void allreduce_bf16_inplace(uint16_t *, size_t) const override {}
        void allgather(const float *send, float *recv, size_t count) const override
        {
            for (int r = 0; r < world_size_; ++r)
                std::memcpy(recv + r * count, send, count * sizeof(float));
        }
        void allgather_bytes(const void *send, void *recv, size_t byte_count) const override
        {
            auto *dst = static_cast<char *>(recv);
            for (int r = 0; r < world_size_; ++r)
                std::memcpy(dst + r * byte_count, send, byte_count);
        }
        void allgatherv_bytes(const void *send, int send_count,
                              void *recv, const int *recv_counts,
                              const int *displs) const override
        {
            auto *dst = static_cast<char *>(recv);
            for (int r = 0; r < world_size_; ++r)
                std::memcpy(dst + displs[r], send, std::min(send_count, recv_counts[r]));
        }
        std::pair<size_t, size_t> get_local_slice(size_t total) const override
        {
            size_t per_rank = total / world_size_;
            return {rank_ * per_rank, per_rank};
        }
        std::pair<size_t, size_t> distribute_rows(size_t total) const override
        {
            return get_local_slice(total);
        }
        void send(const void *, size_t, MPI_Datatype, int, int) const override {}
        void recv(void *, size_t, MPI_Datatype, int, int, MPI_Status *) const override {}
        MPI_Request isend(const void *, size_t, MPI_Datatype, int, int) const override { return MPI_REQUEST_NULL; }
        MPI_Request irecv(void *, size_t, MPI_Datatype, int, int) const override { return MPI_REQUEST_NULL; }
        void wait(MPI_Request *r, MPI_Status *) const override
        {
            if (r)
                *r = MPI_REQUEST_NULL;
        }
        void waitAll(std::vector<MPI_Request> &reqs) const override
        {
            for (auto &r : reqs)
                r = MPI_REQUEST_NULL;
        }
        void probe(int, int, MPI_Status *s) const override
        {
            if (s)
                s->MPI_ERROR = MPI_SUCCESS;
        }
        bool iprobe(int, int, MPI_Status *) const override { return false; }
        void sendFloat(const float *, size_t, int, int) const override {}
        void recvFloat(float *, size_t, int, int, MPI_Status *) const override {}
        void sendBytes(const void *, size_t, int, int) const override {}
        void recvBytes(void *, size_t, int, int, MPI_Status *) const override {}
        int getCount(const MPI_Status &, MPI_Datatype) const override { return 0; }

    private:
        int rank_;
        int world_size_;
        mutable std::vector<BroadcastRecord> broadcasts_;
        mutable size_t barrier_count_{0};
    };

    // =========================================================================
    // MockInferenceRunner — Minimal mock for coordination tests
    // =========================================================================

    class MockInferenceRunner : public IInferenceRunner
    {
    public:
        static constexpr int VOCAB_SIZE = 10;

        MockInferenceRunner()
        {
            logits_.assign(VOCAB_SIZE, -10.0f);
            logits_[3] = 10.0f; // Token 3 is always argmax
        }

        bool forward(const int *tokens, int seq_len) override
        {
            forward_call_count_++;
            last_forward_tokens_.assign(tokens, tokens + seq_len);
            if (throw_on_forward_)
                throw std::runtime_error("Simulated forward failure");
            if (fail_after_n_forwards_ > 0 && forward_call_count_ > fail_after_n_forwards_)
                return false;
            return forward_success_;
        }

        const float *logits() const override { return logits_.data(); }
        int vocab_size() const override { return VOCAB_SIZE; }
        void clear_cache() override { clear_cache_count_++; }
        int get_position() const override { return 0; }
        ExecutionPath executionPath() const override { return ExecutionPath::GRAPH; }
        const char *architecture() const override { return "mock"; }
        int sampleGreedyOnDevice() override
        {
            sample_greedy_on_device_count_++;
            return sample_greedy_on_device_token_;
        }
        bool requiresMPICoordinatedDecodeSampling(const SamplingParams &) const override
        {
            return requires_mpi_coordinated_decode_sampling_;
        }

        void setSkipLogitsGatherDecode(bool skip) override
        {
            skip_logits_gather_ = skip;
            skip_logits_calls_++;
        }

        // Test inspection
        int forwardCallCount() const { return forward_call_count_; }
        int clearCacheCount() const { return clear_cache_count_; }
        int sampleGreedyOnDeviceCount() const { return sample_greedy_on_device_count_; }
        int skipLogitsCalls() const { return skip_logits_calls_; }
        bool skipLogitsGather() const { return skip_logits_gather_; }
        const std::vector<int> &lastForwardTokens() const { return last_forward_tokens_; }

        // Failure injection
        void setForwardSuccess(bool success) { forward_success_ = success; }
        void setThrowOnForward(bool throw_on) { throw_on_forward_ = throw_on; }
        /// Make forward() return false after the Nth successful call
        void setFailAfterNForwards(int n) { fail_after_n_forwards_ = n; }
        void setSampleGreedyOnDeviceToken(int token) { sample_greedy_on_device_token_ = token; }
        void setRequiresMPICoordinatedDecodeSampling(bool required)
        {
            requires_mpi_coordinated_decode_sampling_ = required;
        }

    private:
        std::vector<float> logits_;
        int forward_call_count_{0};
        int clear_cache_count_{0};
        int sample_greedy_on_device_count_{0};
        int skip_logits_calls_{0};
        int sample_greedy_on_device_token_{-1};
        bool skip_logits_gather_{false};
        bool forward_success_{true};
        bool throw_on_forward_{false};
        bool requires_mpi_coordinated_decode_sampling_{false};
        int fail_after_n_forwards_{0}; // 0 = disabled
        std::vector<int> last_forward_tokens_;
    };

    // =========================================================================
    // Test Fixture
    // =========================================================================

    class Test__MPICoordinatedMode : public ::testing::Test
    {
    protected:
        /**
         * @brief Create a minimal execution plan for a given rank
         */
        static RankExecutionPlan createPlan(int rank = 0)
        {
            RankExecutionPlan plan;
            plan.rank = rank;
            plan.hostname = "localhost";
            plan.numa_node = 0;
            plan.pp_stage_id = 0;
            plan.first_layer = 0;
            plan.last_layer = 23;
            plan.has_embedding = true;
            plan.has_lm_head = true;
            plan.primary_device = GlobalDeviceAddress::cpu();
            return plan;
        }

        /**
         * @brief Create an OrchestrationRunner with MPI context injected
         *
         * @param mpi_rank Rank to use for the MPI mock
         * @param mpi_world_size World size for the MPI mock
         * @return Tuple of (runner, mock_runner_ptr, recording_mpi_ptr)
         */
        struct RunnerBundle
        {
            std::unique_ptr<OrchestrationRunner> runner;
            MockInferenceRunner *mock_runner;
            RecordingMPIContext *mpi;
        };

        RunnerBundle createRunner(int mpi_rank, int mpi_world_size)
        {
            auto mock = std::make_unique<MockInferenceRunner>();
            auto *mock_ptr = mock.get();

            auto mpi = std::make_shared<RecordingMPIContext>(mpi_rank, mpi_world_size);
            auto *mpi_ptr = mpi.get();

            OrchestrationConfig config;
            config.device_for_this_rank = GlobalDeviceAddress::cpu();

            auto plan = createPlan(mpi_rank);

            auto runner = std::make_unique<OrchestrationRunner>(
                std::move(config), plan, std::move(mock), mpi);

            // Set greedy sampling
            SamplingParams greedy;
            greedy.temperature = 0.0f;
            runner->setSamplingParams(greedy);

            // Clear recording from the setSamplingParams call above
            mpi_ptr->clearRecords();

            return {std::move(runner), mock_ptr, mpi_ptr};
        }
    };

    // =========================================================================
    // broadcastCommand() Tests
    // =========================================================================

    TEST_F(Test__MPICoordinatedMode, NoBroadcastWhenCoordinatedModeOff)
    {
        auto [runner, mock, mpi] = createRunner(0, 2);
        // Coordinated mode is OFF by default

        runner->prefill({1, 2, 3});

        // No broadcasts should have occurred
        EXPECT_EQ(mpi->broadcastCount(), 0u);
    }

    TEST_F(Test__MPICoordinatedMode, NoBroadcastForSingleRank)
    {
        auto [runner, mock, mpi] = createRunner(0, 1);
        runner->setMPICoordinatedMode(true);

        runner->prefill({1, 2, 3});

        // Single rank: no broadcasts needed
        EXPECT_EQ(mpi->broadcastCount(), 0u);
    }

    // =========================================================================
    // prefill() Coordination Tests
    // =========================================================================

    TEST_F(Test__MPICoordinatedMode, PrefillBroadcastsCommandAndTokens)
    {
        auto [runner, mock, mpi] = createRunner(0, 2);
        runner->setMPICoordinatedMode(true);

        std::vector<int32_t> tokens = {10, 20, 30, 40, 50};
        runner->prefill(tokens);

        // Expected broadcasts:
        // 1. Command tag (PREFILL = 3)
        // 2. Token count (5)
        // 3. Token data (10, 20, 30, 40, 50)
        ASSERT_EQ(mpi->broadcastCount(), 3u);

        // Command tag
        const auto &cmd = mpi->broadcasts()[0];
        EXPECT_EQ(cmd.type, RecordingMPIContext::BroadcastRecord::Type::INT32);
        ASSERT_EQ(cmd.int_data.size(), 1u);
        EXPECT_EQ(cmd.int_data[0],
                  static_cast<int32_t>(OrchestrationRunner::MPICommand::PREFILL));

        // Token count
        const auto &count = mpi->broadcasts()[1];
        EXPECT_EQ(count.type, RecordingMPIContext::BroadcastRecord::Type::INT32);
        ASSERT_EQ(count.int_data.size(), 1u);
        EXPECT_EQ(count.int_data[0], 5);

        // Token data
        const auto &data = mpi->broadcasts()[2];
        EXPECT_EQ(data.type, RecordingMPIContext::BroadcastRecord::Type::INT32);
        EXPECT_THAT(data.int_data, ElementsAre(10, 20, 30, 40, 50));
    }

    TEST_F(Test__MPICoordinatedMode, PrefillStillCallsForward)
    {
        auto [runner, mock, mpi] = createRunner(0, 2);
        runner->setMPICoordinatedMode(true);

        runner->prefill({1, 2, 3});

        // Verify forward was called with the tokens
        EXPECT_EQ(mock->forwardCallCount(), 1);
        EXPECT_THAT(mock->lastForwardTokens(), ElementsAre(1, 2, 3));
    }

    // =========================================================================
    // decodeStep() Coordination Tests
    // =========================================================================

    TEST_F(Test__MPICoordinatedMode, DecodeStepBroadcastsCommand)
    {
        auto [runner, mock, mpi] = createRunner(0, 2);
        runner->setMPICoordinatedMode(true);

        // Prefill first (required before decode)
        runner->prefill({1, 2, 3});
        mpi->clearRecords(); // Clear prefill broadcasts

        runner->decodeStep();

        // First decodeStep after prefill uses prefill logits (no forward call)
        // but still broadcasts DECODE_STEP
        ASSERT_GE(mpi->broadcastCount(), 1u);

        const auto &cmd = mpi->broadcasts()[0];
        EXPECT_EQ(cmd.type, RecordingMPIContext::BroadcastRecord::Type::INT32);
        ASSERT_EQ(cmd.int_data.size(), 1u);
        EXPECT_EQ(cmd.int_data[0],
                  static_cast<int32_t>(OrchestrationRunner::MPICommand::DECODE_STEP));
    }

    TEST_F(Test__MPICoordinatedMode, DecodeStepPublishesRootTokenToWorkers)
    {
        auto [runner, mock, mpi] = createRunner(0, 2);
        runner->setMPICoordinatedMode(true);

        // Prefill makes token 3 available from the mock logits.  Coordinated
        // server mode must publish that root-owned sample before rank 0 returns
        // to the command loop, otherwise worker ranks can drift before the next
        // decode/forced-token command.
        ASSERT_TRUE(runner->prefill({1, 2, 3}));
        mpi->clearRecords();

        GenerationResult result = runner->decodeStep();

        ASSERT_TRUE(result.success()) << result.error;
        ASSERT_THAT(result.tokens, ElementsAre(3));
        ASSERT_EQ(mpi->broadcastCount(), 3u);
        EXPECT_EQ(mpi->broadcasts()[0].int_data[0],
                  static_cast<int32_t>(OrchestrationRunner::MPICommand::DECODE_STEP));
        EXPECT_THAT(mpi->broadcasts()[1].int_data, ElementsAre(0));
        EXPECT_THAT(mpi->broadcasts()[2].int_data, ElementsAre(3));
    }

    TEST_F(Test__MPICoordinatedMode, DecodeStepSkipsDeviceSamplerWhenCoordinatedSamplingDisabled)
    {
        auto [runner, mock, mpi] = createRunner(0, 2);
        runner->setMPICoordinatedMode(true);
        mock->setSampleGreedyOnDeviceToken(7);
        mock->setRequiresMPICoordinatedDecodeSampling(false);

        ASSERT_TRUE(runner->prefill({1, 2, 3}));
        mpi->clearRecords();

        GenerationResult result = runner->decodeStep();

        ASSERT_TRUE(result.success()) << result.error;
        EXPECT_THAT(result.tokens, ElementsAre(3));
        EXPECT_EQ(mock->sampleGreedyOnDeviceCount(), 0);
        ASSERT_EQ(mpi->broadcastCount(), 3u);
        EXPECT_EQ(mpi->broadcasts()[0].int_data[0],
                  static_cast<int32_t>(OrchestrationRunner::MPICommand::DECODE_STEP));
        EXPECT_THAT(mpi->broadcasts()[1].int_data, ElementsAre(0));
        EXPECT_THAT(mpi->broadcasts()[2].int_data, ElementsAre(3));
    }

    TEST_F(Test__MPICoordinatedMode, DecodeStepUsesDeviceSamplerWhenCoordinatedSamplingEnabled)
    {
        auto [runner, mock, mpi] = createRunner(0, 2);
        runner->setMPICoordinatedMode(true);
        mock->setSampleGreedyOnDeviceToken(7);
        mock->setRequiresMPICoordinatedDecodeSampling(true);

        ASSERT_TRUE(runner->prefill({1, 2, 3}));
        mpi->clearRecords();

        GenerationResult result = runner->decodeStep();

        ASSERT_TRUE(result.success()) << result.error;
        EXPECT_THAT(result.tokens, ElementsAre(7));
        EXPECT_EQ(mock->sampleGreedyOnDeviceCount(), 1);
        ASSERT_EQ(mpi->broadcastCount(), 3u);
        EXPECT_EQ(mpi->broadcasts()[0].int_data[0],
                  static_cast<int32_t>(OrchestrationRunner::MPICommand::DECODE_STEP));
        EXPECT_THAT(mpi->broadcasts()[1].int_data, ElementsAre(0));
        EXPECT_THAT(mpi->broadcasts()[2].int_data, ElementsAre(7));
    }

    TEST_F(Test__MPICoordinatedMode, ForceDecodeTokenBroadcastsCommandTokenAndFence)
    {
        auto [runner, mock, mpi] = createRunner(0, 2);
        runner->setMPICoordinatedMode(true);

        ASSERT_TRUE(runner->prefill({1, 2, 3}));
        mpi->clearRecords();

        GenerationResult result = runner->forceDecodeToken(90);

        ASSERT_TRUE(result.success()) << result.error;
        ASSERT_THAT(result.tokens, ElementsAre(90));
        ASSERT_EQ(mpi->broadcastCount(), 2u);
        EXPECT_EQ(mpi->broadcasts()[0].int_data[0],
                  static_cast<int32_t>(OrchestrationRunner::MPICommand::FORCE_DECODE_TOKEN));
        EXPECT_THAT(mpi->broadcasts()[1].int_data, ElementsAre(90));
        EXPECT_EQ(mpi->barrierCount(), 1u);
    }

    // =========================================================================
    // clearCache() Coordination Tests
    // =========================================================================

    TEST_F(Test__MPICoordinatedMode, ClearCacheBroadcastsCommand)
    {
        auto [runner, mock, mpi] = createRunner(0, 2);
        runner->setMPICoordinatedMode(true);

        runner->clearCache();

        ASSERT_EQ(mpi->broadcastCount(), 1u);

        const auto &cmd = mpi->broadcasts()[0];
        EXPECT_EQ(cmd.type, RecordingMPIContext::BroadcastRecord::Type::INT32);
        ASSERT_EQ(cmd.int_data.size(), 1u);
        EXPECT_EQ(cmd.int_data[0],
                  static_cast<int32_t>(OrchestrationRunner::MPICommand::CLEAR_CACHE));
    }

    TEST_F(Test__MPICoordinatedMode, ClearCacheAlsoCallsRunnerClearCache)
    {
        auto [runner, mock, mpi] = createRunner(0, 2);
        runner->setMPICoordinatedMode(true);

        runner->clearCache();

        EXPECT_EQ(mock->clearCacheCount(), 1);
    }

    // =========================================================================
    // setSamplingParams() Coordination Tests
    // =========================================================================

    TEST_F(Test__MPICoordinatedMode, SetSamplingParamsBroadcastsCommandAndParams)
    {
        auto [runner, mock, mpi] = createRunner(0, 2);
        runner->setMPICoordinatedMode(true);

        SamplingParams params;
        params.temperature = 0.7f;
        params.top_p = 0.9f;
        params.top_k = 40;
        params.seed = 42;

        runner->setSamplingParams(params);

        // Expected: command tag + 4 float params
        ASSERT_EQ(mpi->broadcastCount(), 2u);

        // Command tag
        const auto &cmd = mpi->broadcasts()[0];
        EXPECT_EQ(cmd.int_data[0],
                  static_cast<int32_t>(OrchestrationRunner::MPICommand::SET_SAMPLING));

        // Params buffer (4 floats: temperature, top_p, top_k, seed)
        const auto &data = mpi->broadcasts()[1];
        EXPECT_EQ(data.type, RecordingMPIContext::BroadcastRecord::Type::FLOAT);
        ASSERT_EQ(data.float_data.size(), 4u);
        EXPECT_FLOAT_EQ(data.float_data[0], 0.7f);
        EXPECT_FLOAT_EQ(data.float_data[1], 0.9f);
        EXPECT_FLOAT_EQ(data.float_data[2], 40.0f);
        EXPECT_FLOAT_EQ(data.float_data[3], 42.0f);
    }

    // =========================================================================
    // setSkipLogitsGatherDecode() Coordination Tests
    // =========================================================================

    TEST_F(Test__MPICoordinatedMode, SkipLogitsDecodeBroadcastsCommandAndValue)
    {
        auto [runner, mock, mpi] = createRunner(0, 2);
        runner->setMPICoordinatedMode(true);

        runner->setSkipLogitsGatherDecode(true);

        // Expected: command tag + int32 value (1)
        ASSERT_EQ(mpi->broadcastCount(), 2u);

        // Command tag
        const auto &cmd = mpi->broadcasts()[0];
        EXPECT_EQ(cmd.int_data[0],
                  static_cast<int32_t>(OrchestrationRunner::MPICommand::SKIP_LOGITS_DECODE));

        // Value
        const auto &val = mpi->broadcasts()[1];
        EXPECT_EQ(val.type, RecordingMPIContext::BroadcastRecord::Type::INT32);
        ASSERT_EQ(val.int_data.size(), 1u);
        EXPECT_EQ(val.int_data[0], 1);
    }

    TEST_F(Test__MPICoordinatedMode, SkipLogitsDecodeBroadcastsFalseValue)
    {
        auto [runner, mock, mpi] = createRunner(0, 2);
        runner->setMPICoordinatedMode(true);

        runner->setSkipLogitsGatherDecode(false);

        ASSERT_EQ(mpi->broadcastCount(), 2u);

        const auto &val = mpi->broadcasts()[1];
        EXPECT_EQ(val.int_data[0], 0);
    }

    // =========================================================================
    // shutdownMPIWorkers() Tests
    // =========================================================================

    TEST_F(Test__MPICoordinatedMode, ShutdownMPIWorkersBroadcastsShutdown)
    {
        auto [runner, mock, mpi] = createRunner(0, 2);
        runner->setMPICoordinatedMode(true);

        runner->shutdownMPIWorkers();

        ASSERT_EQ(mpi->broadcastCount(), 1u);

        const auto &cmd = mpi->broadcasts()[0];
        EXPECT_EQ(cmd.type, RecordingMPIContext::BroadcastRecord::Type::INT32);
        ASSERT_EQ(cmd.int_data.size(), 1u);
        EXPECT_EQ(cmd.int_data[0],
                  static_cast<int32_t>(OrchestrationRunner::MPICommand::SHUTDOWN));
    }

    TEST_F(Test__MPICoordinatedMode, ShutdownNoopForSingleRank)
    {
        auto [runner, mock, mpi] = createRunner(0, 1);
        runner->setMPICoordinatedMode(true);

        runner->shutdownMPIWorkers();

        EXPECT_EQ(mpi->broadcastCount(), 0u);
    }

    // =========================================================================
    // runMPIWorkerLoop() Tests
    // =========================================================================

    TEST_F(Test__MPICoordinatedMode, WorkerLoopReturnsImmediatelyForRank0)
    {
        auto [runner, mock, mpi] = createRunner(0, 2);
        runner->setMPICoordinatedMode(true);

        // Should return immediately without hanging
        runner->runMPIWorkerLoop();

        // No broadcasts from the worker side
        EXPECT_EQ(mpi->broadcastCount(), 0u);
    }

    TEST_F(Test__MPICoordinatedMode, WorkerLoopReturnsForNullMPIContext)
    {
        // Create runner without MPI context (nullptr)
        auto mock = std::make_unique<MockInferenceRunner>();
        OrchestrationConfig config;
        config.device_for_this_rank = GlobalDeviceAddress::cpu();
        auto plan = createPlan(0);

        auto runner = std::make_unique<OrchestrationRunner>(
            std::move(config), plan, std::move(mock));

        // Should not hang or crash
        runner->runMPIWorkerLoop();
    }

    // =========================================================================
    // Coordinated Mode Toggle Tests
    // =========================================================================

    TEST_F(Test__MPICoordinatedMode, CoordinatedModeCanBeToggled)
    {
        auto [runner, mock, mpi] = createRunner(0, 2);

        // Off by default
        runner->clearCache();
        EXPECT_EQ(mpi->broadcastCount(), 0u);

        // Turn on
        runner->setMPICoordinatedMode(true);
        runner->clearCache();
        EXPECT_EQ(mpi->broadcastCount(), 1u);
        mpi->clearRecords();

        // Turn off
        runner->setMPICoordinatedMode(false);
        runner->clearCache();
        EXPECT_EQ(mpi->broadcastCount(), 0u);
    }

    // =========================================================================
    // Full Sequence Tests
    // =========================================================================

    TEST_F(Test__MPICoordinatedMode, TypicalServerSequenceBroadcastsCorrectly)
    {
        auto [runner, mock, mpi] = createRunner(0, 2);
        runner->setMPICoordinatedMode(true);

        // Simulate a typical server request sequence:
        // 1. Set sampling params
        // 2. Clear cache
        // 3. Prefill
        // 4. Decode step
        // 5. Decode step

        SamplingParams params;
        params.temperature = 0.8f;
        runner->setSamplingParams(params);

        runner->clearCache();
        runner->prefill({100, 200, 300});
        runner->decodeStep();

        mpi->clearRecords();
        runner->decodeStep();

        // Second decode step should broadcast DECODE_STEP command
        ASSERT_GE(mpi->broadcastCount(), 1u);
        EXPECT_EQ(mpi->broadcasts()[0].int_data[0],
                  static_cast<int32_t>(OrchestrationRunner::MPICommand::DECODE_STEP));
    }

    TEST_F(Test__MPICoordinatedMode, AllBroadcastsUseRoot0)
    {
        auto [runner, mock, mpi] = createRunner(0, 2);
        runner->setMPICoordinatedMode(true);

        runner->clearCache();
        runner->prefill({1, 2, 3});
        runner->setSamplingParams({});

        // Every broadcast should have root=0
        for (size_t i = 0; i < mpi->broadcastCount(); ++i)
        {
            EXPECT_EQ(mpi->broadcasts()[i].root, 0)
                << "Broadcast at index " << i << " has wrong root";
        }
    }

    // =========================================================================
    // Non-Root Rank Does Not Broadcast
    // =========================================================================

    TEST_F(Test__MPICoordinatedMode, NonRootRankDoesNotBroadcastOnPrefill)
    {
        // Create a runner as rank 1 (non-root)
        auto [runner, mock, mpi] = createRunner(1, 2);
        runner->setMPICoordinatedMode(true);

        // Non-root calling prefill should NOT broadcast (only rank 0 broadcasts)
        runner->prefill({1, 2, 3});

        EXPECT_EQ(mpi->broadcastCount(), 0u);
    }

    TEST_F(Test__MPICoordinatedMode, NonRootRankDoesNotBroadcastOnClearCache)
    {
        auto [runner, mock, mpi] = createRunner(1, 2);
        runner->setMPICoordinatedMode(true);

        runner->clearCache();

        EXPECT_EQ(mpi->broadcastCount(), 0u);
    }

    // =========================================================================
    // ScriptedMPIContext — Feeds pre-recorded data for worker loop tests
    // =========================================================================

    /**
     * @brief MPI context that replays a scripted sequence of broadcasts.
     *
     * Used to test runMPIWorkerLoop() without real MPI. Each call to
     * broadcast_int32() or broadcast() pops the next scripted response
     * and writes it into the caller's buffer (simulating rank 0 data).
     *
     * Also records all calls for verification of outgoing broadcasts.
     */
    class ScriptedMPIContext : public IMPIContext
    {
    public:
        struct ScriptEntry
        {
            enum class Type
            {
                INT32,
                FLOAT
            };
            Type type;
            std::vector<int32_t> int_data;
            std::vector<float> float_data;
        };

        ScriptedMPIContext(int rank, int world_size)
            : rank_(rank), world_size_(world_size)
        {
        }

        /// Add a scripted int32 broadcast response
        void scriptInt32(std::initializer_list<int32_t> data)
        {
            ScriptEntry e;
            e.type = ScriptEntry::Type::INT32;
            e.int_data = data;
            script_.push_back(std::move(e));
        }

        /// Add a scripted float broadcast response
        void scriptFloat(std::initializer_list<float> data)
        {
            ScriptEntry e;
            e.type = ScriptEntry::Type::FLOAT;
            e.float_data = data;
            script_.push_back(std::move(e));
        }

        // Identity
        int rank() const override { return rank_; }
        int world_size() const override { return world_size_; }
        bool is_root() const override { return rank_ == 0; }
        MPI_Comm communicator() const override { return MPI_COMM_NULL; }

        // Broadcast — replays scripted data
        void broadcast(float *data, size_t count, int root) const override
        {
            (void)root;
            if (script_pos_ < script_.size())
            {
                const auto &entry = script_[script_pos_++];
                size_t to_copy = std::min(count, entry.float_data.size());
                std::memcpy(data, entry.float_data.data(), to_copy * sizeof(float));
            }
            broadcast_count_++;
        }

        void broadcast_int32(int32_t *data, size_t count, int root) const override
        {
            (void)root;
            if (script_pos_ < script_.size())
            {
                const auto &entry = script_[script_pos_++];
                size_t to_copy = std::min(count, entry.int_data.size());
                std::memcpy(data, entry.int_data.data(), to_copy * sizeof(int32_t));
            }
            broadcast_count_++;
        }

        size_t broadcastCount() const { return broadcast_count_; }
        size_t scriptPosition() const { return script_pos_; }
        size_t scriptSize() const { return script_.size(); }
        size_t barrierCount() const { return barrier_count_; }

        // Remaining stubs (same as RecordingMPIContext)
        void barrier() const override { ++barrier_count_; }
        void allreduce_sum(const float *send, float *recv, size_t count) const override
        {
            std::memcpy(recv, send, count * sizeof(float));
        }
        void allreduce_sum_inplace(float *, size_t) const override {}
        void allreduce_q8_1_inplace(Q8_1Block *, size_t) const override {}
        void allreduce_q16_1_inplace(Q16_1Block *, size_t) const override {}
        void allreduce_fp16_inplace(uint16_t *, size_t) const override {}
        void allreduce_bf16_inplace(uint16_t *, size_t) const override {}
        void allgather(const float *send, float *recv, size_t count) const override
        {
            for (int r = 0; r < world_size_; ++r)
                std::memcpy(recv + r * count, send, count * sizeof(float));
        }
        void allgather_bytes(const void *send, void *recv, size_t byte_count) const override
        {
            auto *dst = static_cast<char *>(recv);
            for (int r = 0; r < world_size_; ++r)
                std::memcpy(dst + r * byte_count, send, byte_count);
        }
        void allgatherv_bytes(const void *send, int send_count,
                              void *recv, const int *recv_counts,
                              const int *displs) const override
        {
            auto *dst = static_cast<char *>(recv);
            for (int r = 0; r < world_size_; ++r)
                std::memcpy(dst + displs[r], send, std::min(send_count, recv_counts[r]));
        }
        std::pair<size_t, size_t> get_local_slice(size_t total) const override
        {
            size_t per_rank = total / world_size_;
            return {rank_ * per_rank, per_rank};
        }
        std::pair<size_t, size_t> distribute_rows(size_t total) const override
        {
            return get_local_slice(total);
        }
        void send(const void *, size_t, MPI_Datatype, int, int) const override {}
        void recv(void *, size_t, MPI_Datatype, int, int, MPI_Status *) const override {}
        MPI_Request isend(const void *, size_t, MPI_Datatype, int, int) const override { return MPI_REQUEST_NULL; }
        MPI_Request irecv(void *, size_t, MPI_Datatype, int, int) const override { return MPI_REQUEST_NULL; }
        void wait(MPI_Request *r, MPI_Status *) const override
        {
            if (r)
                *r = MPI_REQUEST_NULL;
        }
        void waitAll(std::vector<MPI_Request> &reqs) const override
        {
            for (auto &r : reqs)
                r = MPI_REQUEST_NULL;
        }
        void probe(int, int, MPI_Status *s) const override
        {
            if (s)
                s->MPI_ERROR = MPI_SUCCESS;
        }
        bool iprobe(int, int, MPI_Status *) const override { return false; }
        void sendFloat(const float *, size_t, int, int) const override {}
        void recvFloat(float *, size_t, int, int, MPI_Status *) const override {}
        void sendBytes(const void *, size_t, int, int) const override {}
        void recvBytes(void *, size_t, int, int, MPI_Status *) const override {}
        int getCount(const MPI_Status &, MPI_Datatype) const override { return 0; }

    private:
        int rank_;
        int world_size_;
        mutable std::vector<ScriptEntry> script_;
        mutable size_t script_pos_{0};
        mutable size_t broadcast_count_{0};
        mutable size_t barrier_count_{0};
    };

    // =========================================================================
    // Helper: Create a worker runner with scripted MPI
    // =========================================================================

    struct WorkerBundle
    {
        std::unique_ptr<OrchestrationRunner> runner;
        MockInferenceRunner *mock_runner;
        ScriptedMPIContext *mpi;
    };

    static WorkerBundle createWorkerRunner(
        std::shared_ptr<ScriptedMPIContext> scripted_mpi)
    {
        auto mock = std::make_unique<MockInferenceRunner>();
        auto *mock_ptr = mock.get();
        auto *mpi_ptr = scripted_mpi.get();

        OrchestrationConfig config;
        config.device_for_this_rank = GlobalDeviceAddress::cpu();

        RankExecutionPlan plan;
        plan.rank = scripted_mpi->rank();
        plan.hostname = "localhost";
        plan.numa_node = 0;
        plan.pp_stage_id = 0;
        plan.first_layer = 0;
        plan.last_layer = 23;
        plan.has_embedding = true;
        plan.has_lm_head = true;
        plan.primary_device = GlobalDeviceAddress::cpu();

        auto runner = std::make_unique<OrchestrationRunner>(
            std::move(config), plan, std::move(mock), scripted_mpi);
        /*
         * Server/interactive modes enable coordinated mode before calling the
         * worker loop.  Keep this helper aligned with that production contract
         * so worker tests exercise decode token publication and forced-token
         * fences instead of the standalone rank path.
         */
        runner->setMPICoordinatedMode(true);

        return {std::move(runner), mock_ptr, mpi_ptr};
    }

    // =========================================================================
    // Worker Loop Dispatch Tests
    // =========================================================================

    TEST_F(Test__MPICoordinatedMode, WorkerLoopDispatchesClearCache)
    {
        auto scripted = std::make_shared<ScriptedMPIContext>(1, 2);
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::CLEAR_CACHE)});
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::SHUTDOWN)});

        auto [runner, mock, mpi] = createWorkerRunner(scripted);
        runner->runMPIWorkerLoop();

        EXPECT_EQ(mock->clearCacheCount(), 1);
    }

    TEST_F(Test__MPICoordinatedMode, WorkerLoopDispatchesPrefill)
    {
        auto scripted = std::make_shared<ScriptedMPIContext>(1, 2);
        // PREFILL command
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::PREFILL)});
        // Token count
        scripted->scriptInt32({3});
        // Token data
        scripted->scriptInt32({100, 200, 300});
        // SHUTDOWN
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::SHUTDOWN)});

        auto [runner, mock, mpi] = createWorkerRunner(scripted);
        runner->runMPIWorkerLoop();

        EXPECT_EQ(mock->forwardCallCount(), 1);
        EXPECT_THAT(mock->lastForwardTokens(), ElementsAre(100, 200, 300));
    }

    TEST_F(Test__MPICoordinatedMode, WorkerLoopDispatchesDecodeStep)
    {
        auto scripted = std::make_shared<ScriptedMPIContext>(1, 2);
        // Prefill first so decode has state
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::PREFILL)});
        scripted->scriptInt32({1});
        scripted->scriptInt32({42});
        // First decode: consumes prefill logits (no forward)
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::DECODE_STEP)});
        scripted->scriptInt32({0}); // decode token budget
        scripted->scriptInt32({7}); // root-published token for next decode
        // Second decode: must call forward
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::DECODE_STEP)});
        scripted->scriptInt32({0}); // decode token budget
        scripted->scriptInt32({8}); // root-published token before shutdown
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::SHUTDOWN)});

        auto [runner, mock, mpi] = createWorkerRunner(scripted);
        runner->runMPIWorkerLoop();

        // Prefill = 1 forward, first decode skips (prefill logits),
        // second decode = 1 forward. Total: 2.
        EXPECT_EQ(mock->forwardCallCount(), 2);
    }

    TEST_F(Test__MPICoordinatedMode, WorkerLoopUsesRootPublishedTokenForNextDecode)
    {
        auto scripted = std::make_shared<ScriptedMPIContext>(1, 2);
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::PREFILL)});
        scripted->scriptInt32({1});
        scripted->scriptInt32({42});
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::DECODE_STEP)});
        scripted->scriptInt32({0}); // decode token budget
        scripted->scriptInt32({777}); // authoritative root sample
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::DECODE_STEP)});
        scripted->scriptInt32({0}); // decode token budget
        scripted->scriptInt32({778});
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::SHUTDOWN)});

        auto [runner, mock, mpi] = createWorkerRunner(scripted);
        runner->runMPIWorkerLoop();

        EXPECT_EQ(mock->forwardCallCount(), 2);
        EXPECT_THAT(mock->lastForwardTokens(), ElementsAre(777));
        EXPECT_EQ(mpi->scriptPosition(), mpi->scriptSize());
    }

    TEST_F(Test__MPICoordinatedMode, WorkerLoopDispatchesSetSampling)
    {
        auto scripted = std::make_shared<ScriptedMPIContext>(1, 2);
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::SET_SAMPLING)});
        scripted->scriptFloat({0.7f, 0.9f, 40.0f, 42.0f}); // temp, top_p, top_k, seed
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::SHUTDOWN)});

        auto [runner, mock, mpi] = createWorkerRunner(scripted);
        runner->runMPIWorkerLoop();

        // Verify sampling params were actually applied to the runner
        auto params = runner->getSamplingParams();
        EXPECT_FLOAT_EQ(params.temperature, 0.7f);
        EXPECT_FLOAT_EQ(params.top_p, 0.9f);
        EXPECT_EQ(params.top_k, 40);
        EXPECT_EQ(params.seed, 42u);
    }

    TEST_F(Test__MPICoordinatedMode, WorkerLoopDispatchesSkipLogitsDecode)
    {
        auto scripted = std::make_shared<ScriptedMPIContext>(1, 2);
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::SKIP_LOGITS_DECODE)});
        scripted->scriptInt32({1}); // skip = true
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::SHUTDOWN)});

        auto [runner, mock, mpi] = createWorkerRunner(scripted);
        runner->runMPIWorkerLoop();

        EXPECT_TRUE(mock->skipLogitsGather());
    }

    TEST_F(Test__MPICoordinatedMode, WorkerLoopHandlesMultipleCommands)
    {
        auto scripted = std::make_shared<ScriptedMPIContext>(1, 2);
        // Typical server sequence: clear → set sampling → prefill → decode → decode → shutdown
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::CLEAR_CACHE)});
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::SET_SAMPLING)});
        scripted->scriptFloat({0.0f, 1.0f, 0.0f, 0.0f}); // greedy params
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::PREFILL)});
        scripted->scriptInt32({2}); // token count
        scripted->scriptInt32({10, 20}); // tokens
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::DECODE_STEP)});
        scripted->scriptInt32({0}); // decode token budget
        scripted->scriptInt32({30}); // root-published token
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::DECODE_STEP)});
        scripted->scriptInt32({0}); // decode token budget
        scripted->scriptInt32({31}); // root-published token
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::SHUTDOWN)});

        auto [runner, mock, mpi] = createWorkerRunner(scripted);
        runner->runMPIWorkerLoop();

        EXPECT_EQ(mock->clearCacheCount(), 1);
        // Prefill calls forward once. First decode uses cached prefill logits.
        // Second decode calls forward again. Total: 2 forward calls.
        EXPECT_EQ(mock->forwardCallCount(), 2);
        EXPECT_EQ(mpi->scriptPosition(), mpi->scriptSize()); // All script entries consumed
    }

    TEST_F(Test__MPICoordinatedMode, WorkerLoopHandlesUnknownCommand)
    {
        auto scripted = std::make_shared<ScriptedMPIContext>(1, 2);
        scripted->scriptInt32({999}); // Unknown command
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::SHUTDOWN)});

        auto [runner, mock, mpi] = createWorkerRunner(scripted);

        // Should not crash — unknown command is logged and skipped
        EXPECT_NO_THROW(runner->runMPIWorkerLoop());
        EXPECT_EQ(mpi->scriptPosition(), mpi->scriptSize());
    }

    // =========================================================================
    // Error / Exception Handling — Rank Desync Prevention
    // =========================================================================

    TEST_F(Test__MPICoordinatedMode, PrefillFailureStillCompletesProtocol)
    {
        // Scenario: Rank 0 broadcasts PREFILL + tokens, then forward() fails.
        // The broadcast has already been sent, so worker ranks received the
        // tokens and called their own prefill(). The protocol remains in sync
        // because the broadcast was completed before the forward error.
        auto [runner, mock, mpi] = createRunner(0, 2);
        runner->setMPICoordinatedMode(true);

        // Make forward fail
        mock->setForwardSuccess(false);

        bool success = runner->prefill({7, 8, 9});

        // Prefill should fail
        EXPECT_FALSE(success);

        // But the protocol broadcasts should have completed (3 broadcasts:
        // command + token count + tokens)
        ASSERT_EQ(mpi->broadcastCount(), 3u);

        // Verify token data was broadcast correctly despite forward failure
        // (workers need the tokens to stay in sync)
        const auto &data = mpi->broadcasts()[2];
        EXPECT_THAT(data.int_data, ElementsAre(7, 8, 9));
    }

    TEST_F(Test__MPICoordinatedMode, PrefillExceptionStillCompletesProtocol)
    {
        // Scenario: forward() throws an exception. The protocol broadcasts
        // were already sent, so worker ranks are not left hanging.
        auto [runner, mock, mpi] = createRunner(0, 2);
        runner->setMPICoordinatedMode(true);

        mock->setThrowOnForward(true);

        bool success = runner->prefill({1, 2, 3});

        // Prefill catches the exception and returns false
        EXPECT_FALSE(success);

        // Protocol broadcasts completed before the exception
        EXPECT_EQ(mpi->broadcastCount(), 3u);
    }

    TEST_F(Test__MPICoordinatedMode, WorkerLoopContinuesAfterPrefillFailure)
    {
        // Scenario: Worker receives PREFILL, its forward() fails.
        // The worker loop should NOT exit — it should continue to the next
        // command. Otherwise rank 0 sends the next command and the worker
        // misses it → deadlock.
        auto scripted = std::make_shared<ScriptedMPIContext>(1, 2);
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::PREFILL)});
        scripted->scriptInt32({2}); // token count
        scripted->scriptInt32({10, 20}); // tokens
        // After failed prefill, worker continues and receives next command
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::CLEAR_CACHE)});
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::SHUTDOWN)});

        auto [runner, mock, mpi] = createWorkerRunner(scripted);
        mock->setForwardSuccess(false);

        // Worker loop should complete without hanging
        EXPECT_NO_THROW(runner->runMPIWorkerLoop());

        // Both prefill (failed) and clear_cache were processed
        EXPECT_EQ(mock->forwardCallCount(), 1); // prefill called forward
        EXPECT_EQ(mock->clearCacheCount(), 1);  // continued to clear_cache
        EXPECT_EQ(mpi->scriptPosition(), mpi->scriptSize()); // All consumed
    }

    TEST_F(Test__MPICoordinatedMode, WorkerLoopContinuesAfterPrefillException)
    {
        // Scenario: Worker's forward() throws during PREFILL.
        // OrchestrationRunner::prefill() catches std::exception internally,
        // so the worker loop should survive and process the next command.
        auto scripted = std::make_shared<ScriptedMPIContext>(1, 2);
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::PREFILL)});
        scripted->scriptInt32({2}); // token count
        scripted->scriptInt32({10, 20}); // tokens
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::CLEAR_CACHE)});
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::SHUTDOWN)});

        auto [runner, mock, mpi] = createWorkerRunner(scripted);
        mock->setThrowOnForward(true);

        // Worker loop should complete — prefill() catches the exception
        EXPECT_NO_THROW(runner->runMPIWorkerLoop());

        // Worker continued past the failed prefill
        EXPECT_EQ(mock->clearCacheCount(), 1);
        EXPECT_EQ(mpi->scriptPosition(), mpi->scriptSize());
    }

    TEST_F(Test__MPICoordinatedMode, WorkerLoopContinuesAfterDecodeStepFailure)
    {
        // Scenario: Worker's forward() fails during DECODE_STEP.
        // decodeStep() returns an error result but does not throw.
        // The worker loop should continue to the next command.
        auto scripted = std::make_shared<ScriptedMPIContext>(1, 2);
        // Prefill to set up state (1st forward call — succeeds)
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::PREFILL)});
        scripted->scriptInt32({2});
        scripted->scriptInt32({10, 20});
        // First decode uses prefill logits (no forward call)
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::DECODE_STEP)});
        scripted->scriptInt32({0}); // decode token budget
        scripted->scriptInt32({21}); // root-published token after first decode
        // Second decode calls forward — the 2nd forward call — which we fail
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::DECODE_STEP)});
        scripted->scriptInt32({0}); // decode token budget
        // Should still continue past the failed decode
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::CLEAR_CACHE)});
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::SHUTDOWN)});

        auto [runner, mock, mpi] = createWorkerRunner(scripted);
        // Let prefill's forward succeed, then fail on the 2nd forward (decode)
        mock->setFailAfterNForwards(1);

        runner->runMPIWorkerLoop();

        // 2 forward calls total: prefill (success) + decode (failure)
        EXPECT_EQ(mock->forwardCallCount(), 2);
        // Loop continued past decode failure to process clear_cache
        EXPECT_EQ(mock->clearCacheCount(), 1);
        // All script entries consumed (no hang)
        EXPECT_EQ(mpi->scriptPosition(), mpi->scriptSize());
    }

    TEST_F(Test__MPICoordinatedMode, EmptyPrefillDoesNotBroadcast)
    {
        // Scenario: Rank 0 calls prefill({}) — empty tokens.
        // This should NOT broadcast anything, because the early-exit
        // check for empty tokens happens before the broadcast code.
        // If it DID broadcast, workers would try to prefill with 0 tokens
        // and the protocol would desync.
        auto [runner, mock, mpi] = createRunner(0, 2);
        runner->setMPICoordinatedMode(true);

        bool success = runner->prefill({});

        EXPECT_FALSE(success);
        // No broadcasts should have been sent for empty tokens
        EXPECT_EQ(mpi->broadcastCount(), 0u);
    }

    TEST_F(Test__MPICoordinatedMode, MultipleRequestsCycleCorrectly)
    {
        // Simulate two complete server request cycles through the worker loop.
        // This verifies the protocol stays in sync across request boundaries.
        auto scripted = std::make_shared<ScriptedMPIContext>(1, 2);

        // Request 1: clear → prefill → decode → decode
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::CLEAR_CACHE)});
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::PREFILL)});
        scripted->scriptInt32({3}); // token count
        scripted->scriptInt32({1, 2, 3}); // tokens
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::DECODE_STEP)});
        scripted->scriptInt32({0}); // decode token budget
        scripted->scriptInt32({13}); // root-published token
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::DECODE_STEP)});
        scripted->scriptInt32({0}); // decode token budget
        scripted->scriptInt32({14}); // root-published token

        // Request 2: clear → prefill → decode
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::CLEAR_CACHE)});
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::PREFILL)});
        scripted->scriptInt32({2}); // token count
        scripted->scriptInt32({4, 5}); // tokens
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::DECODE_STEP)});
        scripted->scriptInt32({0}); // decode token budget
        scripted->scriptInt32({15}); // root-published token

        // Shutdown
        scripted->scriptInt32({static_cast<int32_t>(OrchestrationRunner::MPICommand::SHUTDOWN)});

        auto [runner, mock, mpi] = createWorkerRunner(scripted);
        runner->runMPIWorkerLoop();

        // Request 1: prefill (1 forward) + 1st decode (uses cached logits, 0 forward) + 2nd decode (1 forward) = 2
        // Request 2: prefill (1 forward) + 1st decode (uses cached logits, 0 forward) = 1
        // Total: 3 forward calls
        EXPECT_EQ(mock->forwardCallCount(), 3);
        // Both clear_caches executed
        EXPECT_EQ(mock->clearCacheCount(), 2);
        // All script entries consumed
        EXPECT_EQ(mpi->scriptPosition(), mpi->scriptSize());
    }

} // namespace
