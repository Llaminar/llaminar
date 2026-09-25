/**
 * @file Test__CPUMainSamplingPublication.cpp
 * @brief Real CPU forward/terminal-allgather proof of sampler output ownership.
 *
 * Two MPI processes execute the tiny production graph. Candidate coordination
 * is scripted separately so unwanted sampler collectives fail without hanging.
 */
#include "utils/CPUForwardSamplingTestFixture.h"
#include "config/TensorParallelConfig.h"
#include "utils/MPIContext.h"
#include <algorithm>
#include <array>
#include <mpi.h>
using namespace llaminar2;
using namespace llaminar2::test;

/**
 * @test CPU samplers consume the completed graph's logits, not a dormant shard.
 *
 * A sharded projection can publish either a gathered full row or a local row.
 * Exercise both through the real CPU forward engine with a scripted candidate
 * peer, so an incorrect extra collective fails immediately instead of hanging
 * an MPI worker at its post-sampling barrier. Penalties must mutate the same
 * published row that greedy sampling consumes.
 */
TEST(CPUMainSamplingPublication, PublishedLogitsOwnership)
{
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    ASSERT_EQ(world_size, 2);
    initCPUBackend(-1);
    DeviceManager::instance().initialize(-1, false);
    for (const bool graph_gathers : {true, false})
    {
        SCOPED_TRACE(::testing::Message() << "graph_gathers=" << graph_gathers);
        TinyQwenForwardFixture fixture(DeviceId::cpu(), KVCachePrecision::FP32);
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::shared_ptr<IMPIContext> mpi = graph_gathers
            ? std::static_pointer_cast<IMPIContext>(
                  std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD))
            : std::static_pointer_cast<IMPIContext>(fixture.mpi);
        fixture.config.mtp.enabled = false;
        fixture.config.mtp.terminal_head_policy = MTPTerminalHeadPolicy::VocabularySharded;
        fixture.config.lm_head_column_parallel = true;
        fixture.config.vocab_local = fixture.config.vocab_size / 2;
        fixture.config.tp_config = std::make_shared<TensorParallelConfig>(
            TensorParallelConfig::equalSplit(2, fixture.config.n_heads,
                fixture.config.n_kv_heads, fixture.config.d_ff,
                fixture.config.vocab_size,
                std::vector<DeviceId>{DeviceId::cpu(), DeviceId::cpu()}));
        fixture.config.local_rank = 0;
        fixture.config.tp_device_idx = 0;
        fixture.lm_head = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(fixture.config.vocab_local),
             static_cast<size_t>(fixture.config.d_model)}, -0.02f, 0.02f, 102);

        // Prepared handles must outlive the runner that borrows them. Declare
        // their owner first so reverse-order destruction retires the runner.
        PreparedWeightStore prepared;
        auto builder = std::make_shared<QwenStandardGraph>(fixture.config, mpi);
        DeviceGraphOrchestrator runner(builder, mpi);
        auto peer = std::make_shared<ScriptedGlobalTPContext>();
        runner.setGlobalTPContext(peer);
        ASSERT_TRUE(runner.initializeInferenceStateFromArena(
            1, fixture.config.max_seq_len, DeviceId::cpu()));
        runner.setWeights(fixture.modelWeights());
        prepareDenseForwardWeights(runner, *builder, prepared, DeviceId::cpu());

        SamplingParams greedy;
        greedy.temperature = 0.0f;
        SamplingParams penalized = greedy;
        penalized.presence_penalty = 1.0f;
        SamplingParams stochastic;
        stochastic.temperature = 0.7f;
        const std::array<int, 3> prompt{1, 3, 5};
        for (int phase = 0; phase != 2; ++phase)
        {
            SCOPED_TRACE(::testing::Message() << "phase=" << phase);
            ASSERT_NE(runner.forward(prompt.data(), phase == 0 ? 3 : 1, 1), nullptr);
            auto *published = graph_gathers ? runner.inferenceState().logits.get()
                                           : runner.inferenceState().logits_local.get();
            ASSERT_NE(published, nullptr);
            const int columns = graph_gathers ? fixture.config.vocab_size
                                             : fixture.config.vocab_local;
            float *values = published->mutable_data();
            ASSERT_NE(values, nullptr);
            const int winner = static_cast<int>(
                std::max_element(values, values + columns) - values);
            const float original_winner = values[winner];
            const GreedyCandidateRecord remote{10000.0f, fixture.config.vocab_local + 5, 1, 0};
            peer->setRemoteRecordBytes(&remote, sizeof(remote));
            if (graph_gathers)
            {
                // The retired shard remains allocated but is not the output.
                // Poison it to distinguish ownership from coincidental equality.
                std::fill_n(runner.inferenceState().logits_local->mutable_data(),
                            fixture.config.vocab_local, 20000.0f);
            }
            EXPECT_EQ(runner.requiresMPICoordinatedDecodeSampling(greedy), !graph_gathers);
            EXPECT_EQ(runner.requiresMPICoordinatedDecodeSampling(penalized), !graph_gathers);
            EXPECT_FALSE(runner.requiresMPICoordinatedDecodeSampling(stochastic));
            const int calls_before = peer->allgatherBytesCalls();
            EXPECT_EQ(runner.sampleGreedyOnDevice(), graph_gathers ? winner : remote.token);
            EXPECT_EQ(peer->allgatherBytesCalls() - calls_before, graph_gathers ? 0 : 1);

            const std::vector<LogitPenalty> penalties{{.token_id = winner, .penalty = 10.0f}};
            ASSERT_TRUE(runner.applyPenaltiesOnDevice(penalties, fixture.config.vocab_size));
            EXPECT_FLOAT_EQ(values[winner], original_winner - 10.0f);

            // Exercise the real host sampler after forward publication with
            // tied maxima across SIMD lanes and at the final vocabulary slot.
            // A shard still coordinates its global candidate, whereas a full
            // publication must never introduce a second collective.
            const GreedyCandidateRecord weaker_peer{-200.0f, columns + 5, 1, 0};
            peer->setRemoteRecordBytes(&weaker_peer, sizeof(weaker_peer));
            for (int first : {0, 7, 8, 15, 16, columns - 1})
            {
                std::fill_n(values, columns, -100.0f);
                values[first] = 1.0f;
                values[columns - 1] = 1.0f;
                const int previous_calls = peer->allgatherBytesCalls();
                EXPECT_EQ(runner.sampleGreedyOnDevice(), first);
                EXPECT_EQ(peer->allgatherBytesCalls() - previous_calls,
                          graph_gathers ? 0 : 1);
            }
        }
        runner.clear_cache();
        EXPECT_THROW(runner.sampleGreedyOnDevice(), std::logic_error)
            << "A reset cannot sample a row from the retired request.";
    }
}
