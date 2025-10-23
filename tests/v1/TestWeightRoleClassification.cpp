// Basic tests for weight role classification (Phase 0)
#include <gtest/gtest.h>
#include "weights/WeightRoles.h"

using namespace llaminar;

TEST(WeightRoleClassification, AttentionQKV)
{
    EXPECT_EQ(classifyWeightRole("layers.0.attention.wq.weight"), WeightRole::W_Q);
    EXPECT_EQ(classifyWeightRole("layers.1.attention.wk.weight"), WeightRole::W_K);
    EXPECT_EQ(classifyWeightRole("layers.2.attention.wv.weight"), WeightRole::W_V);
    EXPECT_EQ(classifyWeightRole("layers.3.attention.wo.weight"), WeightRole::W_O);
}

TEST(WeightRoleClassification, MlpW123)
{
    EXPECT_EQ(classifyWeightRole("layers.5.feed_forward.w1"), WeightRole::W1);
    EXPECT_EQ(classifyWeightRole("layers.5.feed_forward.w2"), WeightRole::W2);
    EXPECT_EQ(classifyWeightRole("layers.5.feed_forward.w3"), WeightRole::W3);
}

TEST(WeightRoleClassification, Embedding)
{
    EXPECT_EQ(classifyWeightRole("tok_embeddings.weight"), WeightRole::Embedding);
    EXPECT_EQ(classifyWeightRole("model.embed_tokens.weight"), WeightRole::Embedding);
}

TEST(WeightRoleClassification, UnknownFallback)
{
    EXPECT_EQ(classifyWeightRole("layers.7.attention.bias"), WeightRole::Unknown);
    EXPECT_EQ(classifyWeightRole("some_random_tensor"), WeightRole::Unknown);
}
