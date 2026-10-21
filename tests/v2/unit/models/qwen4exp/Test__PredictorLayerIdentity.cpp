/**
 * @file Test__PredictorLayerIdentity.cpp
 * @brief Device-free source48/cache0, explicit reader and impossible-tag regressions.
 */
#include "models/qwen/LayerGraphIdentity.h"
#include <gtest/gtest.h>
#include <type_traits>
using namespace llaminar2;
namespace
{
/** @return Exact authenticated policy shape of the published separate Dense0 head. */
Qwen4ExpPredictorMetadata dense()
{ return {48, Qwen4ExpPredictorAttention::Dense, Qwen4ExpPredictorHiddenNorm::PerBranch,
    Qwen4ExpPredictorSource::SharedGGUF}; }
}
TEST(PredictorLayerIdentity, MainHasNoPredictorDepthAndPreservesGlobalOrPPIdentity)
{
    static_assert(!std::is_default_constructible_v<LayerGraphSourceIdentity>);
    static_assert(!std::is_aggregate_v<LayerGraphSourceIdentity>);
    const auto global = LayerGraphSourceIdentity::main(24, 48, 16, GraphCacheLayerIndex::ModelGlobal);
    const auto local = LayerGraphSourceIdentity::main(24, 48, 16, GraphCacheLayerIndex::CacheLocal);
    EXPECT_EQ(global.sourceLayer(), 24);
    EXPECT_EQ(global.cacheLayer(), 24);
    EXPECT_EQ(local.cacheLayer(), 8);
    EXPECT_FALSE(global.predictorDepth());
    EXPECT_EQ(global.workload(), GraphLayerWorkload::MainDecoder);
    EXPECT_EQ(global.prefix(), "layer24_");
    EXPECT_THROW((void)LayerGraphSourceIdentity::main(48, 48, 0, GraphCacheLayerIndex::ModelGlobal), std::invalid_argument);
    EXPECT_THROW((void)LayerGraphSourceIdentity::main(15, 48, 16, GraphCacheLayerIndex::ModelGlobal), std::invalid_argument);
    EXPECT_THROW((void)LayerGraphSourceIdentity::main(0, 48, 0, static_cast<GraphCacheLayerIndex>(99)), std::invalid_argument);
}
TEST(PredictorLayerIdentity, SharedDenseSource48UsesOnlyItsOwnCache0AndDepth0)
{
    const auto source = LayerGraphSourceIdentity::qwen4ExpSharedPredictor(48, dense(), 48, 0, 0);
    const auto identity = FullAttentionLayerIdentity::qwen4ExpSharedPredictor(source);
    EXPECT_EQ(source.sourceLayer(), 48);
    EXPECT_EQ(source.cacheLayer(), 0);
    EXPECT_EQ(source.cacheIndex(), GraphCacheLayerIndex::CacheLocal);
    EXPECT_EQ(source.predictorDepth(), 0);
    EXPECT_EQ(source.prefix(), "MTP0_");
    EXPECT_EQ(identity.consumer(), GraphAttentionConsumer::Dense);
    EXPECT_THROW(identity.requireSelectedConsumer(), std::invalid_argument);
    EXPECT_THROW(source.requireFrozenSource(0), std::invalid_argument);
    EXPECT_NO_THROW(source.requireFrozenSource(48));
    EXPECT_THROW((void)FullAttentionLayerIdentity::main(source, GraphAttentionConsumer::Selected), std::invalid_argument);
    EXPECT_THROW((void)FullAttentionLayerIdentity::inlinePredictor(source), std::invalid_argument);
}
TEST(PredictorLayerIdentity, RejectsEveryUnpairedOrContradictoryPredictorTag)
{
    for (int field = 0; field < 7; ++field) {
        auto metadata = dense();
        int source = 48, depth = 0, cache = 0;
        if (field == 0) metadata.source_layer = 49;
        if (field == 1) metadata.attention = Qwen4ExpPredictorAttention::Sparse;
        if (field == 2) metadata.hidden_norm = static_cast<Qwen4ExpPredictorHiddenNorm>(99);
        if (field == 3) metadata.source = Qwen4ExpPredictorSource::Inline;
        if (field == 4) source = 0;
        if (field == 5) depth = 1;
        if (field == 6) cache = 48;
        EXPECT_THROW((void)LayerGraphSourceIdentity::qwen4ExpSharedPredictor(48, metadata, source, depth, cache), std::invalid_argument) << field;
    }
}
TEST(PredictorLayerIdentity, OrdinaryDenseSelectedAndInlineRolesRemainExplicit)
{
    const auto main = LayerGraphSourceIdentity::main(3, 48, 0, GraphCacheLayerIndex::ModelGlobal);
    const auto selected = FullAttentionLayerIdentity::main(main, GraphAttentionConsumer::Selected);
    EXPECT_NO_THROW(selected.requireSelectedConsumer());
    const auto plain = FullAttentionLayerIdentity::main(main, GraphAttentionConsumer::Dense);
    EXPECT_THROW(plain.requireSelectedConsumer(), std::invalid_argument);
    const auto inline_source = LayerGraphSourceIdentity::inlinePredictor(51, 3);
    EXPECT_EQ(inline_source.sourceLayer(), 51);
    EXPECT_EQ(inline_source.cacheLayer(), 0);
    EXPECT_EQ(inline_source.predictorDepth(), 3);
    EXPECT_EQ(inline_source.prefix(), "MTP3_");
    EXPECT_EQ(FullAttentionLayerIdentity::inlinePredictor(inline_source).consumer(), GraphAttentionConsumer::Dense);
    EXPECT_THROW((void)FullAttentionLayerIdentity::qwen4ExpSharedPredictor(inline_source), std::invalid_argument);
}
