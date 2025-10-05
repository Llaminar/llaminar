#include "qwen_pipeline.h" // QwenPipeline
#include "gtest/gtest.h"
#include <cstdlib>

using namespace llaminar;

// Simple fixture builds a tiny config suitable for rapid KV cache growth testing.
class KVCacheGrowthTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Ensure environment picks up dynamic init
        setenv("LLAMINAR_KV_DYNAMIC_INIT", "1", 1);
        setenv("LLAMINAR_KV_GROWTH_FACTOR", "2", 1);
        debugEnvRefresh(); // refresh snapshot after setting env

        TransformerLayerConfig cfg;
        cfg.n_layers = 2;
        cfg.n_head = 2;
        cfg.n_head_kv = 2;
        cfg.head_dim = 4; // d_model = n_head * head_dim = 8
        cfg.d_model = cfg.n_head * cfg.head_dim;
        cfg.d_ff = 16;
        cfg.vocab_size = 32;
        cfg.max_seq_len = 64;
        cfg.eps = 1e-5f;
        pipeline_ = createQwenPipeline(ModelConfig(cfg, "qwen"));

        // Minimal weight tensors sized per config
        auto make_matrix = [&](int rows, int cols)
        {
            auto t = pipeline_->allocateTestLocalTensor({rows, cols});
            // Fill deterministic small values
            for (int r = 0; r < rows; ++r)
            {
                for (int c = 0; c < cols; ++c)
                {
                    t->data()[r * cols + c] = (float)((r + 1) * 0.001 + (c + 1) * 0.0001);
                }
            }
            return t;
        };
        auto make_vector = [&](int dim)
        { auto t = pipeline_->allocateTestLocalTensor({dim}); for(int i=0;i<dim;++i) t->data()[i] = 1.0f; return t; };

        weights_.token_embedding = make_matrix(cfg.vocab_size, cfg.d_model);
        for (int l = 0; l < cfg.n_layers; ++l)
        {
            weights_.attn_norm_weight.push_back(make_vector(cfg.d_model));
            weights_.wq.push_back(make_matrix(cfg.d_model, cfg.n_head * cfg.head_dim));
            weights_.wk.push_back(make_matrix(cfg.d_model, cfg.n_head_kv * cfg.head_dim));
            weights_.wv.push_back(make_matrix(cfg.d_model, cfg.n_head_kv * cfg.head_dim));
            weights_.wo.push_back(make_matrix(cfg.n_head * cfg.head_dim, cfg.d_model));
            weights_.ffn_norm_weight.push_back(make_vector(cfg.d_model));
            weights_.w_gate.push_back(make_matrix(cfg.d_model, cfg.d_ff));
            weights_.w_up.push_back(make_matrix(cfg.d_model, cfg.d_ff));
            weights_.w_down.push_back(make_matrix(cfg.d_ff, cfg.d_model));
        }
        weights_.output_norm_weight = make_vector(cfg.d_model);
        weights_.lm_head = make_matrix(cfg.d_model, cfg.vocab_size);
    }

    std::unique_ptr<QwenPipeline> pipeline_;
    QwenPipeline::ModelWeights weights_;
};

TEST_F(KVCacheGrowthTest, DynamicInitializationAndGrowth)
{
    ASSERT_NE(pipeline_, nullptr);
    // Prefill with small sequence
    std::vector<int> tokens = {1, 2, 3, 4};
    std::shared_ptr<TensorBase> logits;
    ASSERT_TRUE(pipeline_->execute(tokens, weights_, logits));
    int cap_after_prefill = pipeline_->getKVCacheCapacity();
    int used_after_prefill = pipeline_->getKVCacheUsed();
    EXPECT_GE(cap_after_prefill, (int)tokens.size());
    EXPECT_EQ(used_after_prefill, (int)tokens.size());

    // Record original capacity to test growth trigger
    int original_capacity = cap_after_prefill;

    // Incrementally decode tokens to exceed original capacity (force at least one growth)
    int target_total = original_capacity + 3; // ensure we need additional space
    int vocab = pipeline_->getConfig().vocab_size;
    while (pipeline_->getKVCacheUsed() < target_total)
    {
        int next_tok = (pipeline_->getKVCacheUsed() % vocab);
        std::shared_ptr<TensorBase> dec_logits;
        bool inc = pipeline_->incrementalDecodeToken(next_tok, weights_, dec_logits);
        if (!inc)
        {
            // Fallback path not expected in dynamic mode growth scenario
            FAIL() << "Incremental decode unexpectedly failed / fell back";
        }
    }

    EXPECT_GT(pipeline_->getKVCacheCapacity(), original_capacity) << "Capacity should have grown";
    EXPECT_EQ(pipeline_->getKVCacheUsed(), target_total);
    EXPECT_GE(pipeline_->getKVCacheGrowthEvents(), 1);
}
