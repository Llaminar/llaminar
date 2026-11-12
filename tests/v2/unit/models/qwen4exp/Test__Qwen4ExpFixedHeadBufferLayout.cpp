/**
 * @file Test__Qwen4ExpFixedHeadBufferLayout.cpp
 * @brief Exact fixed-head packet/header arena geometry without device startup.
 *
 * Allocation word rounding is applied to complete banks only. Native transport
 * sends liveRows*(16/P)*originalRowBytes; source-range length never controls
 * message width. Runtime byte evidence is a separate native component gate.
 */
#include "Qwen4ExpMetadataFixture.h"
#include "models/qwen4exp/Qwen4ExpBufferLayout.h"
#include "execution/qwen4exp/Qwen4ExpPLEGatherSnapshot.h"
#include "snapshots/SnapshotCapture.h"
#include "execution/local_execution/graph/GraphResolver.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
using namespace llaminar2;
using namespace llaminar2::test;
namespace {
/** @return One complete PLE-owning main interval with exact original source row bytes. */
Qwen4ExpBufferDimensions dimensions(int rows,int degree,DeviceId device)
{
    return {.main_layers=48,.first_layer=0,.layer_count=48,.compute_width=2560,.context_tokens=4096,
        .request_capacity=1,.flattened_rows=rows,.local_query_heads=24,.device=device,
        .ngram_lookup=ImmutableRowLookupExecution::FixedHeadShardGPU,.ngram_fixed_head_participants=degree};
}
/** @return Only allocation rounds the complete bank to canonical INT32 storage. */
std::size_t words(std::size_t bytes){return (bytes+3)/4*4;}
}
/** @test P2/4/8 M1/grouped/prefill banks are exactly source-derived, with no hosted protocol bank. */
TEST(Qwen4ExpFixedHeadBufferLayout, NativePacketsAndHeadersHaveOneExactDeclarationBOM)
{
    const auto g=Qwen4ExpMemoryGeometry::fromConfig(Qwen4ExpConfig(qwen4MetadataFixture(false)));
    for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})for(const int p:{2,4,8})
        for(const int m:{1,3,15,512,1024,2048})
        {
            const auto d=dimensions(m,p,device);GraphResolverConfig resolver;
            resolver.seq_len=m;resolver.d_model=2560;resolver.local_n_heads=24;
            const auto expected=8u+8u*static_cast<std::size_t>(p)+4u+
                words(static_cast<std::size_t>(m)*(16/p)*g.ple_table_row_bytes)+
                words(static_cast<std::size_t>(m)*16*g.ple_table_row_bytes);
            EXPECT_EQ(qwen4ExpLookupArenaBytes(resolver,g,d),expected);
            applyQwen4ExpBufferGeometry(resolver,g,d);
            EXPECT_EQ(resolver.custom_formulas.at("qfn_fixed_local_words")*4,
                words(static_cast<std::size_t>(m)*(16/p)*g.ple_table_row_bytes));
            GraphSchema schema;applyQwen4ExpBufferLayout(schema,g,d);
            const auto has=[&](const char *name){return std::ranges::any_of(schema.layer_buffers,
                [&](const auto &spec){return spec.name==name;});};
            EXPECT_TRUE(has("qfn_ngram_shard_local_native"));EXPECT_TRUE(has("qfn_ngram_shard_gathered_native"));
            EXPECT_FALSE(has("qfn_ngram_native_rows"));EXPECT_FALSE(has("qfn_ngram_lookup_state"));
            auto nonowner=d;nonowner.first_layer=2;nonowner.layer_count=2;
            EXPECT_EQ(qwen4ExpLookupArenaBytes(resolver,g,nonowner),0u);
        }
}
/** @test Inconsistent mode/P/backend and overflowing native packet geometry fail before allocation. */
TEST(Qwen4ExpFixedHeadBufferLayout, ForeignCohortAndPacketOverflowRejectBeforeAllocation)
{
    auto g=Qwen4ExpMemoryGeometry::fromConfig(Qwen4ExpConfig(qwen4MetadataFixture(false)));
    GraphResolverConfig resolver;resolver.seq_len=15;resolver.d_model=2560;resolver.local_n_heads=24;
    for(const int p:{0,1,3,16}) {
        EXPECT_THROW(qwen4ExpLookupArenaBytes(resolver,g,dimensions(15,p,DeviceId::cuda(0))),std::invalid_argument);
    }
    EXPECT_THROW(qwen4ExpLookupArenaBytes(resolver,g,dimensions(15,2,DeviceId::cpu())),std::invalid_argument);
    auto foreign=dimensions(15,2,DeviceId::cuda(0));foreign.ngram_lookup=ImmutableRowLookupExecution::ResidentGPU;
    EXPECT_THROW(qwen4ExpLookupArenaBytes(resolver,g,foreign),std::invalid_argument);
    g.ple_table_row_bytes=std::numeric_limits<std::size_t>::max();
    EXPECT_THROW(qwen4ExpLookupArenaBytes(resolver,g,dimensions(15,2,DeviceId::cuda(0))),std::overflow_error);
}

/** @test The production pipeline descriptor exposes only original decoded FP32 rows across reset/replay geometries. */
TEST(Qwen4ExpFixedHeadBufferLayout, PipelineGatherSnapshotRetainsOriginalOutputAndExactSemanticKey)
{
    constexpr std::size_t columns=16*3;
    FP32Tensor output(std::vector<std::size_t>{9*columns+1});
    auto *values=output.mutable_data();
    for(std::size_t index=0;index<output.numel();++index)values[index]=static_cast<float>(index);
    for(const std::size_t rows:{1u,3u,9u,1u}) {
        const Qwen4ExpPLEGatherSnapshot descriptor(output,rows,columns,sizeof(float));
        for(int replay=0;replay<3;++replay) {
            const auto dump=descriptor.describe();
            ASSERT_TRUE(dump.inputs.empty());ASSERT_EQ(dump.outputs.size(),1u);
            const auto &view=dump.outputs.front();
            EXPECT_STREQ(view.name,"gathered_rows");EXPECT_STREQ(view.dtype,"FP32");
            EXPECT_EQ(view.tensor,&output);EXPECT_EQ(view.tensor_byte_offset,sizeof(float));
            EXPECT_EQ(view.data,values+1);EXPECT_EQ(view.rows,rows);EXPECT_EQ(view.cols,columns);
            EXPECT_EQ(view.byte_size,rows*columns*sizeof(float));
            const auto captured=SnapshotCapture::extractFp32FromOutput(view);
            ASSERT_EQ(captured.size(),rows*columns);
            for(std::size_t index=0;index<captured.size();++index)EXPECT_EQ(captured[index],values[index+1]);
        }
    }
    // Existing semantic stage names retain all PLE checkpoints; the descriptor
    // changes no gate/convolution graph or snapshot key policy.
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer1_ple_gather"),"layer1_PLE_GATHER");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer1_ple_gate"),"layer1_PLE_GATE");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer1_ple_convolution"),"layer1_PLE_CONVOLUTION");
    EXPECT_THROW((Qwen4ExpPLEGatherSnapshot(output,0,columns)),std::invalid_argument);
    EXPECT_THROW((Qwen4ExpPLEGatherSnapshot(output,10,columns)),std::invalid_argument);
    EXPECT_THROW((Qwen4ExpPLEGatherSnapshot(output,1,columns,1)),std::invalid_argument);
    EXPECT_THROW((Qwen4ExpPLEGatherSnapshot(output,1,std::numeric_limits<std::size_t>::max())),std::invalid_argument);
}
