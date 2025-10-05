// Multi-rank incremental generation test.
// Verifies that prefill + incremental decode path works under MPI (2 ranks) and
// produces consistent KV cache usage plus identical sampled tokens given a fixed
// deterministic weight initialization and sampling temperature=0 (argmax sampling).

#include "gtest/gtest.h"
#include "qwen_pipeline.h"
#include "chat/response_generator.h"
#include "model_loader.h"
#include "utils/debug_env.h"
#include <mpi.h>

using namespace llaminar;

namespace
{

    // Global MPI environment to avoid multiple init/finalize cycles across tests.
    struct MPIGlobalEnvironment : public ::testing::Environment
    {
        void SetUp() override
        {
            int inited = 0;
            MPI_Initialized(&inited);
            if (!inited)
            {
                int argc = 0;
                char **argv = nullptr;
                MPI_Init(&argc, &argv);
            }
            setenv("LLAMINAR_TEST_MPI_NO_FINALIZE", "1", 1);
        }
        void TearDown() override
        {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized)
                MPI_Finalize();
        }
    };
    static ::testing::Environment *const mpi_env = ::testing::AddGlobalTestEnvironment(new MPIGlobalEnvironment());

    struct MiniDeterministicWeights
    {
        static QwenPipeline::ModelWeights build(QwenPipeline &pipe, const TransformerLayerConfig &cfg)
        {
            QwenPipeline::ModelWeights W;
            auto make_matrix = [&](int rows, int cols)
            {
                auto t = pipe.allocateTestLocalTensor({rows, cols});
                for (int r = 0; r < rows; ++r)
                    for (int c = 0; c < cols; ++c)
                    {
                        // Deterministic small values independent of rank to ensure identical logits
                        t->data()[r * cols + c] = 0.001f * (r + 1) + 0.0001f * (c + 1);
                    }
                return t;
            };
            auto make_vec = [&](int dim)
            { auto t = pipe.allocateTestLocalTensor({dim}); for(int i=0;i<dim;++i) t->data()[i] = 1.0f; return t; };
            W.token_embedding = make_matrix(cfg.vocab_size, cfg.d_model);
            for (int l = 0; l < cfg.n_layers; ++l)
            {
                W.attn_norm_weight.push_back(make_vec(cfg.d_model));
                W.wq.push_back(make_matrix(cfg.d_model, cfg.n_head * cfg.head_dim));
                W.wk.push_back(make_matrix(cfg.d_model, cfg.n_head_kv * cfg.head_dim));
                W.wv.push_back(make_matrix(cfg.d_model, cfg.n_head_kv * cfg.head_dim));
                W.wo.push_back(make_matrix(cfg.n_head * cfg.head_dim, cfg.d_model));
                W.ffn_norm_weight.push_back(make_vec(cfg.d_model));
                W.w_gate.push_back(make_matrix(cfg.d_model, cfg.d_ff));
                W.w_up.push_back(make_matrix(cfg.d_model, cfg.d_ff));
                W.w_down.push_back(make_matrix(cfg.d_ff, cfg.d_model));
            }
            W.output_norm_weight = make_vec(cfg.d_model);
            W.lm_head = make_matrix(cfg.d_model, cfg.vocab_size);
            return W;
        }
    };
} // namespace

TEST(IncrementalGenerationMultiRankTest, PrefillDecodeIdenticalAcrossRanks)
{
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    ASSERT_GE(world, 2) << "Run with at least 2 ranks";
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Tiny config stressing distribution logic (heads=1 keeps shape simple)
    // IMPORTANT: Use n_head divisible by world size to avoid ranks with zero local heads
    // (which would yield empty local projection tensors and trigger null pointer checks
    // inside MPIAttentionKernel::distributeInputs). With world=2, choose n_head=2.
    TransformerLayerConfig cfg;
    cfg.n_layers = 1;
    cfg.n_head = 2;
    cfg.n_head_kv = 2;
    cfg.head_dim = 4;
    cfg.d_model = cfg.n_head * cfg.head_dim;
    cfg.d_ff = 8;
    cfg.vocab_size = 32;
    cfg.max_seq_len = 64;
    cfg.eps = 1e-5f;
    auto pipe_unique = createQwenPipeline(ModelConfig(cfg, "qwen"));
    ASSERT_NE(pipe_unique, nullptr);
    auto weights = MiniDeterministicWeights::build(*pipe_unique, cfg);
    auto shared_pipe = std::shared_ptr<AbstractPipeline>(pipe_unique.release());

    // Prefill tokens broadcast from rank 0
    std::vector<int> prompt_local;
    if (rank == 0)
        prompt_local = {5, 6, 7, 8};
    int count = (int)prompt_local.size();
    MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0)
        prompt_local.resize(count);
    MPI_Bcast(prompt_local.data(), count, MPI_INT, 0, MPI_COMM_WORLD);

    StageContext ctx;
    QwenModelWeights wrapper;
    wrapper.inner = weights;
    ASSERT_TRUE(shared_pipe->prefill(prompt_local, wrapper, ctx));

    // Capture logits after prefill
    std::shared_ptr<TensorBase> logits_tensor;
    ASSERT_TRUE(shared_pipe->logits(logits_tensor));
    ASSERT_TRUE(logits_tensor);
    const auto &shape = logits_tensor->shape();
    ASSERT_EQ(shape.size(), 2u);
    int rows = shape[0];
    int cols = shape[1];
    ASSERT_EQ(rows, (int)prompt_local.size());
    std::vector<float> last_row(cols);
    std::memcpy(last_row.data(), logits_tensor->data() + (rows - 1) * cols, sizeof(float) * cols);

    // All ranks reduce the last row to rank 0 then broadcast a checksum for equality verification
    std::vector<float> sum_row(cols, 0.f);
    MPI_Reduce(last_row.data(), sum_row.data(), cols, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    // Rank 0 computes reference and broadcasts
    std::vector<float> ref_row(cols);
    if (rank == 0)
    {
        // Compute average (should equal each row if identical across ranks)
        for (int i = 0; i < cols; ++i)
            ref_row[i] = sum_row[i] / world;
    }
    MPI_Bcast(ref_row.data(), cols, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // Compare locally
    for (int i = 0; i < cols; ++i)
    {
        ASSERT_NEAR(last_row[i], ref_row[i], 1e-6) << "Mismatch at col " << i;
    }

    // Perform a couple of decode steps deterministically: choose argmax.
    int steps = 3;
    auto *dist_pipeline = dynamic_cast<QwenPipeline *>(shared_pipe.get());
    ASSERT_NE(dist_pipeline, nullptr);
    for (int s = 0; s < steps; ++s)
    {
        // Argmax of last_row
        int best = 0;
        float bestv = last_row[0];
        for (int i = 1; i < cols; ++i)
            if (last_row[i] > bestv)
            {
                bestv = last_row[i];
                best = i;
            }
        ASSERT_TRUE(shared_pipe->decode(best, wrapper, ctx));
        ASSERT_TRUE(shared_pipe->logits(logits_tensor));
        rows = logits_tensor->shape()[0];
        cols = logits_tensor->shape()[1];
        std::memcpy(last_row.data(), logits_tensor->data() + (rows - 1) * cols, sizeof(float) * cols);
        // Reduce + broadcast + compare again
        std::vector<float> step_sum(cols, 0.f);
        MPI_Reduce(last_row.data(), step_sum.data(), cols, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
            for (int i = 0; i < cols; ++i)
                ref_row[i] = step_sum[i] / world;
        MPI_Bcast(ref_row.data(), cols, MPI_FLOAT, 0, MPI_COMM_WORLD);
        for (int i = 0; i < cols; ++i)
            ASSERT_NEAR(last_row[i], ref_row[i], 1e-6) << "Decode step=" << s << " col=" << i;
        // Per-step KV cache usage validation
        EXPECT_EQ(dist_pipeline->getKVCacheUsed(), (int)prompt_local.size() + (s + 1));
    }

    // Final KV cache usage should reflect prompt + all decode steps
    EXPECT_EQ(dist_pipeline->getKVCacheUsed(), (int)prompt_local.size() + steps);
}

// EOS early stop variant: we stop decoding when the sampled token equals an EOS id.
TEST(IncrementalGenerationMultiRankTest, PrefillDecodeStopsOnEOS)
{
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    ASSERT_GE(world, 2) << "Run with at least 2 ranks";
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    TransformerLayerConfig cfg;
    cfg.n_layers = 1;
    cfg.n_head = 2;
    cfg.n_head_kv = 2;
    cfg.head_dim = 4;
    cfg.d_model = cfg.n_head * cfg.head_dim;
    cfg.d_ff = 8;
    cfg.vocab_size = 32;
    cfg.max_seq_len = 64;
    cfg.eps = 1e-5f;
    auto pipe_unique = createQwenPipeline(ModelConfig(cfg, "qwen"));
    ASSERT_NE(pipe_unique, nullptr);
    auto weights = MiniDeterministicWeights::build(*pipe_unique, cfg);
    auto shared_pipe = std::shared_ptr<AbstractPipeline>(pipe_unique.release());

    std::vector<int> prompt_local;
    if (rank == 0)
        prompt_local = {1, 2, 3};
    int count = (int)prompt_local.size();
    MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0)
        prompt_local.resize(count);
    MPI_Bcast(prompt_local.data(), count, MPI_INT, 0, MPI_COMM_WORLD);

    StageContext ctx;
    QwenModelWeights wrapper;
    wrapper.inner = weights;
    ASSERT_TRUE(shared_pipe->prefill(prompt_local, wrapper, ctx));

    std::shared_ptr<TensorBase> logits_tensor;
    ASSERT_TRUE(shared_pipe->logits(logits_tensor));
    ASSERT_TRUE(logits_tensor);
    const auto &shape = logits_tensor->shape();
    ASSERT_EQ(shape.size(), 2u);
    int rows = shape[0];
    int cols = shape[1];
    ASSERT_EQ(rows, (int)prompt_local.size());
    std::vector<float> last_row(cols);
    std::memcpy(last_row.data(), logits_tensor->data() + (rows - 1) * cols, sizeof(float) * cols);

    const int eos_id = cols - 1; // deterministic largest logit expected at final column
    auto *dist_pipeline = dynamic_cast<QwenPipeline *>(shared_pipe.get());
    ASSERT_NE(dist_pipeline, nullptr);

    int max_steps = 8;
    int decoded = 0;
    bool hit_eos = false;
    for (int s = 0; s < max_steps; ++s)
    {
        int best = 0;
        float bestv = last_row[0];
        for (int i = 1; i < cols; ++i)
            if (last_row[i] > bestv)
            {
                bestv = last_row[i];
                best = i;
            }
        ASSERT_TRUE(shared_pipe->decode(best, wrapper, ctx));
        ++decoded;
        EXPECT_EQ(dist_pipeline->getKVCacheUsed(), (int)prompt_local.size() + decoded);
        if (best == eos_id)
        {
            hit_eos = true;
            break;
        }
        ASSERT_TRUE(shared_pipe->logits(logits_tensor));
        rows = logits_tensor->shape()[0];
        cols = logits_tensor->shape()[1];
        std::memcpy(last_row.data(), logits_tensor->data() + (rows - 1) * cols, sizeof(float) * cols);
    }

    ASSERT_TRUE(hit_eos) << "Did not encounter EOS token within max_steps";
    EXPECT_LE(decoded, max_steps);
    EXPECT_EQ(dist_pipeline->getKVCacheUsed(), (int)prompt_local.size() + decoded);
}
