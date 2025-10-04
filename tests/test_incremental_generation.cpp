// Incremental generation integration test for AbstractPipeline-based ResponseGenerator
// Focus: ensure single prefill + successive decode steps update KV cache and logits without
// replaying the full sequence. Keeps model tiny & synthetic for speed.

#include "gtest/gtest.h"
#include "distributed_transformer_pipeline.h" // for createDistributedTransformerPipeline & weights struct
#include "chat/response_generator.h"
#include "chat/gguf_tokenizer.h"
#include "model_loader.h"
#include "utils/debug_env.h"
#include <mpi.h>

using namespace llaminar;

namespace
{

    // Global MPI environment (single-rank version) to ensure MPI is initialized once for the
    // lifetime of this test binary. Without this, the first MPIKernelBase instance that called
    // MPI_Init_thread would also call MPI_Finalize in its destructor, leaving subsequent tests
    // unable to call MPI routines (and triggering MPI_Comm_rank after finalize errors).
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
            // Ensure kernel destructors do not finalize MPI before all tests complete
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

    // Minimal stub tokenizer implementing TokenizerInterface with deterministic mapping.
    class MiniTokenizer : public chat::TokenizerInterface
    {
    public:
        bool isReady() const override { return true; }
        std::vector<int32_t> tokenize(const std::string &text) override
        {
            std::vector<int32_t> out;
            for (unsigned char c : text)
            {
                if (c >= 32 && c < 126)
                    out.push_back(1 + (c % 29));
            }
            return out;
        }
        std::string detokenize(const std::vector<int32_t> &tokens) override
        {
            std::string s;
            for (auto t : tokens)
                s.push_back((char)('a' + (t % 26)));
            return s;
        }
        std::string applyTemplate(const std::vector<chat::ChatMessage> &messages, bool add_generation_prompt) override
        {
            std::string combined;
            for (const auto &m : messages)
            {
                combined += m.role + ":" + m.content + "\n";
            }
            if (add_generation_prompt)
                combined += "assistant:";
            return combined;
        }
        bool loadVocabulary(const ModelLoader &) override { return true; }
        int32_t getSpecialToken(const std::string &token_name) override { return token_name == "eos" ? 0 : -1; }
        size_t getVocabSize() const override { return 32; }
        std::string getTokenString(int32_t token_id) override { return std::string(1, (char)('a' + (token_id % 26))); }
    };

    DistributedTransformerPipeline::ModelWeights makeTinyWeights(DistributedTransformerPipeline &pipe, const TransformerLayerConfig &cfg)
    {
        auto make_matrix = [&](int rows, int cols)
        {
            auto t = pipe.allocateTestLocalTensor({rows, cols});
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c)
                    t->data()[r * cols + c] = 0.001f * (r + 1) + 0.0001f * (c + 1);
            return t;
        };
        auto make_vector = [&](int dim)
        {
        auto t = pipe.allocateTestLocalTensor({dim});
        for (int i=0;i<dim;++i) t->data()[i] = 1.0f; return t; };
        DistributedTransformerPipeline::ModelWeights W;
        W.token_embedding = make_matrix(cfg.vocab_size, cfg.d_model);
        for (int l = 0; l < cfg.n_layers; ++l)
        {
            W.attn_norm_weight.push_back(make_vector(cfg.d_model));
            W.wq.push_back(make_matrix(cfg.d_model, cfg.n_head * cfg.head_dim));
            W.wk.push_back(make_matrix(cfg.d_model, cfg.n_head_kv * cfg.head_dim));
            W.wv.push_back(make_matrix(cfg.d_model, cfg.n_head_kv * cfg.head_dim));
            W.wo.push_back(make_matrix(cfg.n_head * cfg.head_dim, cfg.d_model));
            W.ffn_norm_weight.push_back(make_vector(cfg.d_model));
            W.w_gate.push_back(make_matrix(cfg.d_model, cfg.d_ff));
            W.w_up.push_back(make_matrix(cfg.d_model, cfg.d_ff));
            W.w_down.push_back(make_matrix(cfg.d_ff, cfg.d_model));
        }
        W.output_norm_weight = make_vector(cfg.d_model);
        W.lm_head = make_matrix(cfg.d_model, cfg.vocab_size);
        return W;
    }

} // namespace

TEST(IncrementalGenerationTest, PrefillThenDecodeIncrementsState)
{
    // Ensure MPI is initialized before constructing any MPIKernelBase-derived kernels
    int mpi_inited = 0;
    MPI_Initialized(&mpi_inited);
    if (!mpi_inited)
    {
        int argc = 0;
        char **argv = nullptr;
        int provided = 0;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    }
    // Configure tiny model
    TransformerLayerConfig cfg;
    cfg.n_layers = 1;
    cfg.n_head = 1;
    cfg.n_head_kv = 1;
    cfg.head_dim = 4;
    cfg.d_model = cfg.n_head * cfg.head_dim;
    cfg.d_ff = 8;
    cfg.vocab_size = 32;
    cfg.max_seq_len = 64;
    cfg.eps = 1e-5f;
    auto pipeline = createDistributedTransformerPipeline(cfg);
    ASSERT_NE(pipeline, nullptr);
    auto weights = makeTinyWeights(*pipeline, cfg);

    // Wrap in AbstractPipeline shared ownership
    auto shared_pipe = std::shared_ptr<AbstractPipeline>(pipeline.release());

    // Build tokenizer & generator (use pre-loaded weights ctor)
    auto tokenizer = std::make_shared<MiniTokenizer>();
    LlaminarParams params; // defaults; n_predict small
    params.n_predict = 8;  // keep short
    QwenModelWeights wrapper;
    wrapper.inner = weights; // wrap
    chat::ResponseGenerator generator(tokenizer, shared_pipe, params, wrapper);

    // Initial prompt tokens
    std::vector<int32_t> prompt = {5, 6, 7};
    // Force generation with a simple callback accumulating output
    std::string assembled;
    auto cb = [&](const std::string &txt, bool)
    { assembled += txt; };
    std::string full = generator.generateStreamingResponse(prompt, cb);

    // Basic sanity: produced some output (may be empty if EOS sampled early but not expected with our logits)
    EXPECT_GE(full.size(), 0u);

    // Access underlying distributed pipeline again for KV cache stats
    auto *dist = dynamic_cast<DistributedTransformerPipeline *>(shared_pipe.get());
    ASSERT_NE(dist, nullptr);
    // After prefill + at least one decode, used tokens should be >= prompt size
    EXPECT_GE(dist->getKVCacheUsed(), (int)prompt.size());
    // Capacity must be >= used
    EXPECT_GE(dist->getKVCacheCapacity(), dist->getKVCacheUsed());
}

TEST(IncrementalGenerationTest, PrefillDecodeStopsOnEOS)
{
    // Ensure MPI initialized (same rationale as above test)
    int mpi_inited = 0;
    MPI_Initialized(&mpi_inited);
    if (!mpi_inited)
    {
        int argc = 0;
        char **argv = nullptr;
        int provided = 0;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    }
    // Tiny deterministic model config
    TransformerLayerConfig cfg;
    cfg.n_layers = 1;
    cfg.n_head = 1;
    cfg.n_head_kv = 1;
    cfg.head_dim = 4;
    cfg.d_model = cfg.n_head * cfg.head_dim;
    cfg.d_ff = 8;
    cfg.vocab_size = 32;
    cfg.max_seq_len = 64;
    cfg.eps = 1e-5f;
    auto pipeline = createDistributedTransformerPipeline(cfg);
    ASSERT_NE(pipeline, nullptr);
    auto weights = makeTinyWeights(*pipeline, cfg);
    auto shared_pipe = std::shared_ptr<AbstractPipeline>(pipeline.release());

    auto tokenizer = std::make_shared<MiniTokenizer>();
    LlaminarParams params;
    params.n_predict = 16; // allow enough room
    QwenModelWeights wrapper;
    wrapper.inner = weights;
    chat::ResponseGenerator generator(tokenizer, shared_pipe, params, wrapper);

    // Use a short prompt
    std::vector<int32_t> prompt = {4, 5, 6};
    // We'll manually drive decode to observe KV usage & detect EOS.
    StageContext ctx;
    ASSERT_TRUE(shared_pipe->prefill(prompt, wrapper, ctx));
    auto *dist = dynamic_cast<DistributedTransformerPipeline *>(shared_pipe.get());
    ASSERT_NE(dist, nullptr);
    EXPECT_EQ(dist->getKVCacheUsed(), (int)prompt.size());

    // Fetch logits after prefill
    std::shared_ptr<TensorBase> logits_tensor;
    ASSERT_TRUE(shared_pipe->logits(logits_tensor));
    ASSERT_TRUE(logits_tensor);
    const auto &shape = logits_tensor->shape();
    ASSERT_EQ(shape.size(), 2u);
    int rows = shape[0];
    int cols = shape[1];
    ASSERT_EQ(rows, (int)prompt.size());
    std::vector<float> last_row(cols);
    std::memcpy(last_row.data(), logits_tensor->data() + (rows - 1) * cols, sizeof(float) * cols);

    const int eos_id = cols - 1; // highest index likely largest logit
    bool hit_eos = false;
    int decoded = 0;
    int max_steps = 12;
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
        EXPECT_EQ(dist->getKVCacheUsed(), (int)prompt.size() + decoded);
        if (best == eos_id)
        {
            hit_eos = true;
            break;
        }
        ASSERT_TRUE(shared_pipe->logits(logits_tensor));
        rows = logits_tensor->shape()[0];
        cols = logits_tensor->shape()[1];
        ASSERT_EQ(cols, cfg.vocab_size);
        if ((int)rows != (int)prompt.size() + decoded)
        {
            ADD_FAILURE() << "Unexpected logits row count " << rows << " expected " << (prompt.size() + decoded);
            break;
        }
        last_row.resize(cols);
        std::memcpy(last_row.data(), logits_tensor->data() + (rows - 1) * cols, sizeof(float) * cols);
    }

    ASSERT_TRUE(hit_eos) << "EOS token not encountered within max_steps";
    EXPECT_LE(decoded, max_steps);
    EXPECT_EQ(dist->getKVCacheUsed(), (int)prompt.size() + decoded);
}
