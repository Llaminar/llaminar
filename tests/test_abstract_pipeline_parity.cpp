// Parity test: legacy MPITransformerPipeline vs AbstractPipeline QwenPipelineAdapter
// Ensures logits parity for: (1) full prefill legacy execute, (2) adapter prefill decode replay,
// and (3) incremental decode path using KV cache (when enabled) for subsequent tokens.
//
// Strategy:
// 1. Build identical random weights (small config) and prompt tokens.
// 2. Run legacy pipeline execute(prompt) capturing final logits.
// 3. Run adapter prefill(prompt) then decode one additional token using:
//    a) replay path (env forces disable incremental decode)
//    b) incremental path (default) capturing logits for that extra token.
// 4. Compare:
//    - Adapter prefill final prefill-token logits vs legacy final logits (prefill parity)
//    - Incremental decode logits for new token vs replay decode logits for same new token.
// Tolerances are tight (1e-5) since computations are identical sequences of operations.

#include "gtest/gtest.h"
#include "mpi_transformer_pipeline.h"
#include "qwen_pipeline_adapter.h"
#include "abstract_pipeline.h"
#include "tensors/tensor_factory.h"
#include "logger.h"
#include "test_mpi_utils.h"
#include <random>
#include <numeric>

using namespace llaminar;

namespace
{

    struct ParityConfig
    {
        TransformerLayerConfig cfg;
        ParityConfig()
        {
            cfg.n_head = 4;
            cfg.n_head_kv = 4;
            cfg.head_dim = 32;
            cfg.d_model = 128; // 4 * 32
            cfg.d_ff = 512;    // 4 * d_model
            cfg.vocab_size = 256;
            cfg.max_seq_len = 64;
            cfg.n_layers = 2;
            cfg.eps = 1e-6f;
        }
    };

    struct RandomWeightBuilder
    {
        explicit RandomWeightBuilder(const TransformerLayerConfig &c) : cfg(c)
        {
            gen.seed(123);
        }
        MPITransformerPipeline::ModelWeights build()
        {
            MPITransformerPipeline::ModelWeights w;
            auto randTensor = [&](const std::vector<int> &shape, float a = -0.01f, float b = 0.01f)
            {
                auto t = TensorFactory::create_simple(shape);
                std::uniform_real_distribution<float> dist(a, b);
                size_t total = 1;
                for (int d : shape)
                    total *= d;
                float *dst = const_cast<float *>(t->data());
                for (size_t i = 0; i < total; ++i)
                    dst[i] = dist(gen);
                return t;
            };
            w.token_embedding = randTensor({cfg.vocab_size, cfg.d_model});
            w.output_norm_weight = randTensor({cfg.d_model}, 0.8f, 1.2f);
            w.lm_head = randTensor({cfg.d_model, cfg.vocab_size});
            for (int i = 0; i < cfg.n_layers; ++i)
            {
                w.attn_norm_weight.push_back(randTensor({cfg.d_model}, 0.8f, 1.2f));
                w.wq.push_back(randTensor({cfg.d_model, cfg.n_head * cfg.head_dim}));
                w.wk.push_back(randTensor({cfg.d_model, cfg.n_head_kv * cfg.head_dim}));
                w.wv.push_back(randTensor({cfg.d_model, cfg.n_head_kv * cfg.head_dim}));
                w.wo.push_back(randTensor({cfg.n_head * cfg.head_dim, cfg.d_model}));
                w.ffn_norm_weight.push_back(randTensor({cfg.d_model}, 0.8f, 1.2f));
                w.w_gate.push_back(randTensor({cfg.d_model, cfg.d_ff}));
                w.w_up.push_back(randTensor({cfg.d_model, cfg.d_ff}));
                w.w_down.push_back(randTensor({cfg.d_ff, cfg.d_model}));
            }
            return w;
        }
        TransformerLayerConfig cfg;
        std::mt19937 gen;
    };

    std::vector<int> makeTokens(int n, int vocab, uint32_t seed = 999)
    {
        std::mt19937 g(seed);
        std::uniform_int_distribution<int> dist(0, vocab - 1);
        std::vector<int> t;
        t.reserve(n);
        for (int i = 0; i < n; ++i)
            t.push_back(dist(g));
        return t;
    }

    // Helper to wrap weights for adapter
    struct WrappedQwenWeights : public QwenModelWeights
    {
        explicit WrappedQwenWeights(const MPITransformerPipeline::ModelWeights &mw) { inner = mw; }
    };

} // namespace

// MPI main
LLAMINAR_DEFINE_GTEST_MPI_MAIN();

TEST(AbstractPipelineParity, PrefillAndIncrementalDecodeParity)
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    ASSERT_TRUE(initialized);
    ParityConfig pc;
    RandomWeightBuilder builder(pc.cfg);
    auto weights = builder.build();
    auto legacy = createMPITransformerPipeline(pc.cfg);
    // Create adapter
    registerQwenPipeline();
    auto adapter = PipelineFactory::instance().create("qwen", pc.cfg);
    ASSERT_TRUE(adapter);
    // Prepare tokens
    const int prefill_len = 5;
    const int extra_tokens = 2; // decode two tokens
    auto prefill_tokens = makeTokens(prefill_len, pc.cfg.vocab_size);
    auto decode_tokens = makeTokens(extra_tokens, pc.cfg.vocab_size, 12345); // independent stream
    // Legacy full prefill
    std::shared_ptr<TensorBase> legacy_logits = TensorFactory::create_simple({prefill_len, pc.cfg.vocab_size});
    ASSERT_TRUE(legacy->execute(prefill_tokens, weights, legacy_logits));
    // Adapter prefill
    WrappedQwenWeights wrapped(weights);
    StageContext ctx;
    ASSERT_TRUE(adapter->prefill(prefill_tokens, wrapped, ctx));
    std::shared_ptr<TensorBase> adapter_prefill_logits;
    ASSERT_TRUE(adapter->logits(adapter_prefill_logits));
    ASSERT_EQ(adapter_prefill_logits->shape()[0], prefill_len);
    // Compare last-token logits of prefill legacy vs adapter
    const float *legacy_last = legacy_logits->data() + (prefill_len - 1) * pc.cfg.vocab_size;
    const float *adapter_last = adapter_prefill_logits->data() + (prefill_len - 1) * pc.cfg.vocab_size;
    for (int i = 0; i < pc.cfg.vocab_size; ++i)
    {
        ASSERT_NEAR(legacy_last[i], adapter_last[i], 1e-5f) << "Prefill parity mismatch at logit index " << i;
    }

    // Replay decode reference path (disable incremental via env flag manually by calling execute with concatenated sequence)
    // Build cumulative token stream as we go to produce reference logits for each new token.
    std::vector<int> running = prefill_tokens;
    std::vector<std::vector<float>> replay_new_logits;
    replay_new_logits.reserve(extra_tokens);
    for (int t = 0; t < extra_tokens; ++t)
    {
        running.push_back(decode_tokens[t]);
        std::shared_ptr<TensorBase> out = TensorFactory::create_simple({(int)running.size(), pc.cfg.vocab_size});
        ASSERT_TRUE(legacy->execute(running, weights, out));
        // Extract last row as reference
        std::vector<float> ref(pc.cfg.vocab_size);
        const float *row = out->data() + (running.size() - 1) * pc.cfg.vocab_size;
        std::copy(row, row + pc.cfg.vocab_size, ref.begin());
        replay_new_logits.push_back(std::move(ref));
    }

    // Incremental decode via adapter (uses KV cache incremental path where available)
    for (int t = 0; t < extra_tokens; ++t)
    {
        ASSERT_TRUE(adapter->decode(decode_tokens[t], wrapped, ctx)) << "Adapter decode failed at step " << t;
        std::shared_ptr<TensorBase> inc_logits;
        ASSERT_TRUE(adapter->logits(inc_logits));
        ASSERT_EQ(inc_logits->shape()[0], (int)(prefill_len + t + 1));
        const float *row_inc = inc_logits->data() + (inc_logits->shape()[0] - 1) * pc.cfg.vocab_size;
        const auto &ref_vec = replay_new_logits[t];
        for (int i = 0; i < pc.cfg.vocab_size; ++i)
        {
            ASSERT_NEAR(row_inc[i], ref_vec[i], 1e-5f) << "Incremental parity mismatch at token step=" << t << " logit=" << i;
        }
    }
}
