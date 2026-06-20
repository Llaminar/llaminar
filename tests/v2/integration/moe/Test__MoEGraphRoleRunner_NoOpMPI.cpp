#include "execution/moe/MoEGraphRoleRunner.h"
#include "utils/MPIContext.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include <memory>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        class CountingNoOpParticipantRunner final : public IInferenceRunner
        {
        public:
            bool forward(const int *, int seq_len) override
            {
                ++forward_calls_;
                position_ += seq_len;
                return true;
            }

            const float *logits() const override { return nullptr; }
            int vocab_size() const override { return 0; }
            void clear_cache() override { position_ = 0; }
            int get_position() const override { return position_; }
            ExecutionPath executionPath() const override { return ExecutionPath::GRAPH; }
            const char *architecture() const override { return "qwen35moe"; }

            int forwardCallCount() const { return forward_calls_; }

        private:
            int position_ = 0;
            int forward_calls_ = 0;
        };

    } // namespace

    class Test__MoEGraphRoleRunner_NoOpMPI : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

            if (world_size_ < 2)
            {
                GTEST_SKIP() << "Test requires at least 2 MPI ranks (got " << world_size_ << ")";
            }

            mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);
        }

        int rank_ = -1;
        int world_size_ = 0;
        std::shared_ptr<IMPIContext> mpi_ctx_;
    };

    TEST_F(Test__MoEGraphRoleRunner_NoOpMPI, ParticipantsCompleteNoOpStepWithMatchingScalarKeys)
    {
        std::vector<MoEGraphParticipantSpec> specs;
        specs.reserve(static_cast<size_t>(world_size_) + 1u);

        for (int rank = 0; rank < world_size_; ++rank)
        {
            MoEGraphParticipantSpec spec;
            spec.world_rank = rank;
            spec.participant_id = 0;
            spec.device = DeviceId::cpu();
            spec.continuation_root = (rank == 0);
            if (rank == 0)
                spec.role = MoEGraphParticipantRole::Continuation;
            else if (rank == 1)
                spec.role = MoEGraphParticipantRole::CpuExpert;
            else
                spec.role = MoEGraphParticipantRole::AcceleratorExpert;
            specs.push_back(spec);
        }

        // Include a relay participant to lock in skeleton role assignment coverage.
        specs.push_back(MoEGraphParticipantSpec{
            .world_rank = 1,
            .participant_id = 1,
            .device = DeviceId::cpu(),
            .role = MoEGraphParticipantRole::Relay,
            .continuation_root = false,
        });

        MoEGraphRoleRunner::Config config;
        config.participant_specs = specs;
        config.mpi_ctx = mpi_ctx_;
        config.continuation_root = (rank_ == 0);
        config.architecture = "qwen35moe";

        auto local_runner = std::make_unique<CountingNoOpParticipantRunner>();
        CountingNoOpParticipantRunner *local_runner_ptr = local_runner.get();

        if (rank_ == 0)
        {
            config.local_participant_runners.push_back(std::move(local_runner));
        }

        MoEGraphRoleRunner runner(std::move(config));

        const int tokens[] = {11, 22, 33};
        ASSERT_TRUE(runner.forward(tokens, 3));

        const auto &header = runner.lastStepHeader();
        EXPECT_EQ(header.generation_id, 1u);
        EXPECT_EQ(header.step_id, 0u);
        EXPECT_EQ(header.seq_len, 3);
        EXPECT_FALSE(header.decode);

        const auto &statuses = runner.lastStatuses();
        ASSERT_EQ(statuses.size(), specs.size());
        for (const auto &status : statuses)
        {
            EXPECT_TRUE(status.ok);
            EXPECT_EQ(status.generation_id, header.generation_id);
            EXPECT_EQ(status.step_id, header.step_id);
            EXPECT_EQ(status.seq_len, header.seq_len);
        }

        if (rank_ == 0)
        {
            ASSERT_NE(local_runner_ptr, nullptr);
            EXPECT_EQ(local_runner_ptr->forwardCallCount(), 1);
        }
    }

} // namespace llaminar2::test
