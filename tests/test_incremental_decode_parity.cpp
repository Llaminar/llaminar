#include "gtest/gtest.h"
#include "distributed_transformer_pipeline.h"
#include "model_loader.h"
#include "utils/debug_env.h"
#include <mpi.h>

using namespace llaminar;

namespace
{
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
                int provided = 0;
                MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
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

    DistributedTransformerPipeline::ModelWeights makeTinyWeights(DistributedTransformerPipeline &pipe, const TransformerLayerConfig &cfg)
    {
        auto make_matrix = [&](int rows, int cols)
        { auto t = pipe.allocateTestLocalTensor({rows,cols}); for(int r=0;r<rows;++r) for(int c=0;c<cols;++c) t->data()[r*cols+c] = 0.001f*(r+1)+0.0001f*(c+3); return t; };
        auto make_vec = [&](int n)
        { auto t = pipe.allocateTestLocalTensor({n}); for(int i=0;i<n;++i) t->data()[i]=1.0f; return t; };
        DistributedTransformerPipeline::ModelWeights W;
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

    float relL2(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size() || a.empty())
            return 1.0f;
        long double num = 0, den = 0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            long double d = (long double)a[i] - b[i];
            num += d * d;
            den += (long double)b[i] * b[i];
        }
        return (float)(std::sqrt((double)num) / (std::sqrt((double)den) + 1e-30));
    }

} // namespace

TEST(IncrementalDecodeParity, ReplayVsIncrementalSingleRank)
{
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world > 1)
    {
        GTEST_SKIP() << "Single-rank parity test requires world size ==1 (launched under mpirun).";
    }
    TransformerLayerConfig cfg;
    cfg.n_layers = 2;
    cfg.n_head = 1;
    cfg.n_head_kv = 1;
    cfg.head_dim = 8;
    cfg.d_model = cfg.n_head * cfg.head_dim;
    cfg.d_ff = 16;
    cfg.vocab_size = 48;
    cfg.max_seq_len = 128;
    cfg.eps = 1e-5f;
    auto pipeA = createDistributedTransformerPipeline(cfg);
    ASSERT_TRUE(pipeA);
    auto weights = makeTinyWeights(*pipeA, cfg);
    // Prefill sequence
    std::vector<int> prompt = {3, 7, 11, 5};
    auto out_full = TensorFactory::create_simple({(int)prompt.size(), cfg.vocab_size});
    ASSERT_TRUE(pipeA->execute(prompt, weights, out_full));
    // Capture full logits after prefill
    std::vector<float> base_logits((size_t)cfg.vocab_size);
    std::memcpy(base_logits.data(), out_full->data() + ((prompt.size() - 1) * cfg.vocab_size), sizeof(float) * cfg.vocab_size);

    // Now incremental pipeline: prefill then decode tokens one by one comparing to replay extension
    auto pipeInc = createDistributedTransformerPipeline(cfg);
    ASSERT_TRUE(pipeInc);
    pipeInc->enableKVCache(true);
    // Prefill stage for incremental pipeline
    auto prefill_logits = TensorFactory::create_simple({(int)prompt.size(), cfg.vocab_size});
    ASSERT_TRUE(pipeInc->execute(prompt, weights, prefill_logits));
    // Sanity: last row parity with full pipeline prefill
    std::vector<float> inc_prefill_last(cfg.vocab_size);
    std::memcpy(inc_prefill_last.data(), prefill_logits->data() + ((prompt.size() - 1) * cfg.vocab_size), sizeof(float) * cfg.vocab_size);
    EXPECT_LT(relL2(base_logits, inc_prefill_last), 1e-5f);

    // Generate next tokens deterministically (choose token id = (i*7)%vocab)
    std::vector<int> generated;
    int steps = 5;
    for (int i = 0; i < steps; ++i)
    {
        int next = (i * 7) % cfg.vocab_size; // incremental
        // Replay path: build combined sequence and run full execute
        auto replay_pipe = createDistributedTransformerPipeline(cfg);
        auto replay_logits = TensorFactory::create_simple({(int)prompt.size() + (int)generated.size() + 1, cfg.vocab_size});
        auto replay_weights = weights; // reuse same weights
        std::vector<int> replay_seq = prompt;
        replay_seq.insert(replay_seq.end(), generated.begin(), generated.end());
        replay_seq.push_back(next);
        ASSERT_TRUE(replay_pipe->execute(replay_seq, replay_weights, replay_logits));
        std::vector<float> replay_last(cfg.vocab_size);
        std::memcpy(replay_last.data(), replay_logits->data() + ((replay_seq.size() - 1) * cfg.vocab_size), sizeof(float) * cfg.vocab_size);
        // Incremental path: call internal decodeToken (simulate pipeline decode use)
        std::shared_ptr<TensorBase> inc_logits_row;
        bool used = pipeInc->incrementalDecodeToken(next, weights, inc_logits_row);
        ASSERT_TRUE(used);
        ASSERT_TRUE(inc_logits_row);
        ASSERT_EQ(inc_logits_row->shape()[0], 1);
        ASSERT_EQ(inc_logits_row->shape()[1], cfg.vocab_size);
        std::vector<float> inc_last(cfg.vocab_size);
        std::memcpy(inc_last.data(), inc_logits_row->data(), sizeof(float) * cfg.vocab_size);
        float diff = relL2(replay_last, inc_last);
        EXPECT_LT(diff, 1e-4f) << "step=" << i;
        // Optional per-layer token diff instrumentation: if enabled via env, dump earliest layer mismatch summary
        const char *lt_env = std::getenv("LLAMINAR_PIPELINE_LAYER_TOKEN_DIFF");
        if (lt_env && std::string(lt_env) == "1")
        {
            const auto &rows = DistributedTransformerPipeline::getLastLayerTokenRows();
            // rows vector accumulates across calls; last 2*n_layers entries should be the current iteration's replay + incremental
            // Heuristic: collect last incremental single-row captures (incremental flag true) and compare to a fresh replay executed above
            // For deeper diagnostics a dedicated harness can refine this, but we at least surface counts here.
            int inc_count = 0;
            for (auto &r : rows)
                if (r.incremental)
                    ++inc_count;
            (void)inc_count; // silence unused if assertions compiled out
        }
        generated.push_back(next);
    }
}

// Multi-rank parity skeleton: run only if world size >=2 (uses MPI but still single-process test binary launched via mpirun)
TEST(IncrementalDecodeParity, ReplayVsIncrementalMultiRank)
{
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world < 2)
    {
        GTEST_SKIP() << "Requires mpirun -np >=2";
    }
    TransformerLayerConfig cfg;
    cfg.n_layers = 2;
    cfg.n_head = 2;
    cfg.n_head_kv = 1;
    cfg.head_dim = 8;
    cfg.d_model = cfg.n_head * cfg.head_dim;
    cfg.d_ff = 32;
    cfg.vocab_size = 64;
    cfg.max_seq_len = 128;
    cfg.eps = 1e-5f;
    auto pipe = createDistributedTransformerPipeline(cfg);
    ASSERT_TRUE(pipe);
    pipe->enableKVCache(true);
    auto weights = makeTinyWeights(*pipe, cfg);
    std::vector<int> prompt = {1, 2, 3, 4};
    auto prefill_logits = TensorFactory::create_simple({(int)prompt.size(), cfg.vocab_size});
    ASSERT_TRUE(pipe->execute(prompt, weights, prefill_logits));
    // Perform incremental decode and replay check for 3 steps
    std::vector<int> generated;
    for (int step = 0; step < 3; ++step)
    {
        int next = (step * 5) % cfg.vocab_size; // Replay
        auto replay_pipe = createDistributedTransformerPipeline(cfg);
        auto replay_logits = TensorFactory::create_simple({(int)prompt.size() + (int)generated.size() + 1, cfg.vocab_size});
        ASSERT_TRUE(replay_pipe->execute([&]()
                                         { std::vector<int> seq=prompt; seq.insert(seq.end(), generated.begin(), generated.end()); seq.push_back(next); return seq; }(), weights, replay_logits));
        std::vector<float> replay_last(cfg.vocab_size);
        std::memcpy(replay_last.data(), replay_logits->data() + (((int)prompt.size() + (int)generated.size()) * cfg.vocab_size), sizeof(float) * cfg.vocab_size);
        std::shared_ptr<TensorBase> inc_logits_row;
        ASSERT_TRUE(pipe->incrementalDecodeToken(next, weights, inc_logits_row));
        std::vector<float> inc_last(cfg.vocab_size);
        std::memcpy(inc_last.data(), inc_logits_row->data(), sizeof(float) * cfg.vocab_size);
        float diff = relL2(replay_last, inc_last);
        EXPECT_LT(diff, 1e-4f) << "multi-rank step=" << step;
        generated.push_back(next);
    }
}
