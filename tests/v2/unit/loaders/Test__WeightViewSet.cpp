/**
 * @file Test__WeightViewSet.cpp
 * @brief Unit tests for WeightViewSet — lightweight read-only weight accessor
 *
 * Tests construction, lookup, iteration, and edge cases without any model loading.
 */

#include <gtest/gtest.h>
#include <memory>

#include "loaders/WeightViewSet.h"
#include "tensors/TensorClasses.h"

using namespace llaminar2;

class Test__WeightViewSet : public ::testing::Test
{
protected:
    static std::shared_ptr<TensorBase> makeDummyTensor()
    {
        return std::make_shared<FP32Tensor>(std::vector<size_t>{4, 4});
    }

    WeightViewSet makePopulatedSet()
    {
        WeightViewSet set(0, 12, true, false);

        set.addView({.name = "token_embd.weight",
                     .tensor = makeDummyTensor(),
                     .layer_index = -1,
                     .is_gemm_weight = false});

        set.addView({.name = "blk.0.attn_q.weight",
                     .tensor = makeDummyTensor(),
                     .layer_index = 0,
                     .is_gemm_weight = true});

        set.addView({.name = "blk.5.ffn_down.weight",
                     .tensor = makeDummyTensor(),
                     .layer_index = 5,
                     .is_gemm_weight = true});

        return set;
    }
};

TEST_F(Test__WeightViewSet, ConstructEmpty)
{
    WeightViewSet set(0, 24, true, true);

    EXPECT_EQ(set.size(), 0u);
    EXPECT_TRUE(set.empty());
    EXPECT_EQ(set.firstLayer(), 0);
    EXPECT_EQ(set.lastLayer(), 24);
    EXPECT_TRUE(set.hasEmbedding());
    EXPECT_TRUE(set.hasLMHead());
}

TEST_F(Test__WeightViewSet, LayerRangeMetadata)
{
    WeightViewSet set(4, 16, false, true);

    EXPECT_EQ(set.firstLayer(), 4);
    EXPECT_EQ(set.lastLayer(), 16);
    EXPECT_FALSE(set.hasEmbedding());
    EXPECT_TRUE(set.hasLMHead());
}

TEST_F(Test__WeightViewSet, AddAndFind)
{
    auto set = makePopulatedSet();

    EXPECT_EQ(set.size(), 3u);
    EXPECT_FALSE(set.empty());

    // Find by name
    auto *view = set.find("blk.0.attn_q.weight");
    ASSERT_NE(view, nullptr);
    EXPECT_EQ(view->name, "blk.0.attn_q.weight");
    EXPECT_EQ(view->layer_index, 0);
    EXPECT_TRUE(view->is_gemm_weight);
    EXPECT_NE(view->tensor, nullptr);

    // Find embedding
    auto *emb = set.find("token_embd.weight");
    ASSERT_NE(emb, nullptr);
    EXPECT_EQ(emb->layer_index, -1);
    EXPECT_FALSE(emb->is_gemm_weight);

    // Not found
    EXPECT_EQ(set.find("nonexistent"), nullptr);
}

TEST_F(Test__WeightViewSet, GetTensorConvenience)
{
    auto set = makePopulatedSet();

    auto tensor = set.getTensor("blk.5.ffn_down.weight");
    ASSERT_NE(tensor, nullptr);

    auto missing = set.getTensor("missing");
    EXPECT_EQ(missing, nullptr);
}

TEST_F(Test__WeightViewSet, DuplicateNameThrows)
{
    WeightViewSet set(0, 12, true, false);

    set.addView({.name = "blk.0.attn_q.weight",
                 .tensor = makeDummyTensor(),
                 .layer_index = 0,
                 .is_gemm_weight = true});

    EXPECT_THROW(
        set.addView({.name = "blk.0.attn_q.weight",
                     .tensor = makeDummyTensor(),
                     .layer_index = 0,
                     .is_gemm_weight = true}),
        std::runtime_error);
}

TEST_F(Test__WeightViewSet, RangeBasedFor)
{
    auto set = makePopulatedSet();

    std::vector<std::string> names;
    for (const auto &view : set)
    {
        names.push_back(view.name);
    }

    EXPECT_EQ(names.size(), 3u);
    EXPECT_EQ(names[0], "token_embd.weight");
    EXPECT_EQ(names[1], "blk.0.attn_q.weight");
    EXPECT_EQ(names[2], "blk.5.ffn_down.weight");
}

TEST_F(Test__WeightViewSet, SharedOwnership)
{
    auto tensor = makeDummyTensor();
    EXPECT_EQ(tensor.use_count(), 1);

    {
        WeightViewSet set(0, 1, false, false);
        set.addView({.name = "shared_test",
                     .tensor = tensor,
                     .layer_index = 0,
                     .is_gemm_weight = false});

        // WeightViewSet holds a shared_ptr copy
        EXPECT_EQ(tensor.use_count(), 2);
    }

    // After set destruction, only original ref remains
    EXPECT_EQ(tensor.use_count(), 1);
}
