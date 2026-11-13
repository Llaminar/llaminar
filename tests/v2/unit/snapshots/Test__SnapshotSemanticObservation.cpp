/**
 * @file Test__SnapshotSemanticObservation.cpp
 * @brief Exact diagnostic reconstruction, empty-state presence and chunk-axis regressions.
 *
 * Tests exercise the production snapshot publisher with acquired native metadata.
 * They require neither a GPU nor a model and never initialize output padding in
 * inference buffers. Identity tests prove padded cosine cannot certify wrong IDs.
 */
#include "snapshots/SnapshotCapture.h"
#include "snapshots/SnapshotInfo.h"
#include "snapshots/SnapshotSelectedIds.h"
#include "snapshots/SnapshotEmptyObservation.h"
#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/GpuTensorView.h"
#include <array>
#include <vector>
using namespace llaminar2;

/** @test A counted selection creates sentinels only in its private host diagnostic payload. */
TEST(SnapshotSemanticObservation, ReconstructsOnlyValidatedSelectedPrefixes)
{
    std::array<std::int32_t,8> packed{0,3,7,2,99,99,99,99};
    const auto original=packed;
    const SnapshotObservedHeader header{0,2,16};
    const std::array<std::uint64_t,3> offsets{0,3,4};
    StageDumpInfo dump;
    dump.outputs.push_back({.name="selected",.data=packed.data(),.rows=4,.cols=4,.dtype="INT32",
        .element_size=4,.byte_size=64,.semantic_extent=SnapshotCountedSelectionRows{},
        .observed_header=&header,.observed_row_offsets=offsets});
    SnapshotCapture capture;capture.captureStage("layer1_qsa_indexer",dump);
    const auto result=capture.getShared("layer1_QSA_SELECTED_INDICES");
    ASSERT_TRUE(result);EXPECT_EQ(result->rows,2u);EXPECT_EQ(result->cols,4u);
    EXPECT_EQ(result->data,(std::vector<float>{0,3,7,-1,2,-1,-1,-1}));
    EXPECT_EQ(packed,original);
    const std::array<std::uint64_t,3> invalid{0,5,4};
    dump.outputs[0].observed_row_offsets=invalid;
    EXPECT_THROW(capture.captureStage("layer1_qsa_indexer",dump),std::logic_error);
    EXPECT_EQ(capture.getShared("layer1_QSA_SELECTED_INDICES"),result);
    dump.outputs[0].observed_row_offsets=offsets;packed[2]=-1;
    EXPECT_THROW(capture.captureStage("layer1_qsa_indexer",dump),std::logic_error);
    EXPECT_EQ(capture.getShared("layer1_QSA_SELECTED_INDICES"),result);
}

/** @test An explicitly observed empty state remains distinguishable from an absent checkpoint. */
TEST(SnapshotSemanticObservation, PublishesObservedEmptyState)
{
    const std::array<float,8> payload{1,2,3,4,5,6,7,8};
    const SnapshotObservedHeader header{2,1,0};
    StageDumpInfo dump;
    dump.outputs.push_back({.name="pooled_keys",.data=payload.data(),.rows=4,.cols=2,.dtype="FP32",
        .element_size=4,.byte_size=32,.semantic_extent=SnapshotStatePrefixRows{{},16,4},.observed_header=&header});
    SnapshotCapture capture;capture.captureStage("layer1_qsa_indexer",dump);
    const auto result=capture.getShared("layer1_QSA_INDEXER_POOLED_KEYS");
    ASSERT_TRUE(result);EXPECT_EQ(result->rows,0u);EXPECT_EQ(result->cols,2u);EXPECT_TRUE(result->data.empty());
    SnapshotInfo present;present.rows=0;present.cols=2;present.lifetime_owner=result;
    EXPECT_TRUE(present);EXPECT_FALSE(SnapshotInfo{});
    EXPECT_TRUE(matchesFloatingEmptyObservation(present,2));
    EXPECT_FALSE(matchesFloatingEmptyObservation(present,128));
    EXPECT_FALSE(matchesFloatingEmptyObservation(SnapshotInfo{},2));
    present.size=1;EXPECT_FALSE(matchesFloatingEmptyObservation(present,2));
}

/** @test State rows numerically matching a token chunk never become concatenated token rows. */
TEST(SnapshotSemanticObservation, ChunkJoinKeepsLatestCompleteStatePrefix)
{
    const std::array<float,8> first{1,2,3,4,5,6,7,8},last{11,12,13,14,15,16,17,18};
    const SnapshotObservedHeader h0{6,2,16},h1{8,4,24};
    SnapshotCapture capture;
    for(const auto &[context,payload,header]:std::array{
        std::tuple{"chunk0",first.data(),&h0},std::tuple{"chunk1",last.data(),&h1}})
    {
        StageDumpInfo dump;dump.outputs.push_back({.name="pooled_keys",.data=payload,.rows=4,.cols=2,.dtype="FP32",
            .element_size=4,.byte_size=32,.semantic_extent=SnapshotStatePrefixRows{{},16,4},.observed_header=header});
        capture.captureStage(std::string(context)+"::layer1_qsa_indexer",dump);
    }
    const auto joined=capture.aggregateSequentialChunkSnapshots({{"chunk0",2},{"chunk1",4}});
    ASSERT_TRUE(joined)<<joined.error;
    const auto result=capture.getShared("layer1_QSA_INDEXER_POOLED_KEYS");
    ASSERT_TRUE(result);EXPECT_EQ(result->rows,3u);
    EXPECT_EQ(result->data,(std::vector<float>{11,12,13,14,15,16}));
    EXPECT_EQ(joined.aggregated_sequence_keys,0u);EXPECT_EQ(joined.terminal_or_nonsequence_keys,1u);
}

/** @test One wrong identity cannot hide among thousands of canonical padding cells. */
TEST(SnapshotSelectedIds, OneWrongIdentityCannotHideAmongPadding)
{
    std::vector<float> a(2051,-1),b=a;a[0]=b[0]=0;a[1]=1;b[1]=2;
    const auto result=compareSnapshotSelectedIds(a,b,2051);
    EXPECT_EQ(result.identityMismatches,1u);EXPECT_EQ(result.countMismatches,0u);EXPECT_FALSE(result.equal());
}
/** @test Exact count equality and canonical sorted integral prefixes are independent requirements. */
TEST(SnapshotSelectedIds, ExactCountsAndCanonicalSentinels)
{
    std::vector<float>a{0,1,-1,-1},b{0,-1,-1,-1};
    EXPECT_EQ(compareSnapshotSelectedIds(a,b,4).countMismatches,1u);
    a={0,-1,1,-1};EXPECT_EQ(compareSnapshotSelectedIds(a,b,4).malformedRows,1u);
    a={0,.5,-1,-1};EXPECT_EQ(compareSnapshotSelectedIds(a,b,4).malformedRows,1u);
    EXPECT_TRUE(compareSnapshotSelectedIds({}, {},4).equal());
}

/** @test Real CPU tensors use host coherence, not an optional GPU allocation, as authority. */
TEST(SnapshotSemanticObservation, CPUOwnedTensorsObserveNativeCountsWithoutGPUHeader)
{
    constexpr int capacity=4,columns=4;
    INT32Tensor controls({4}),counts({capacity}),selected({capacity,columns});
    FP32Tensor pools({capacity,128}),query({capacity,512});
    auto *control=static_cast<std::int32_t *>(controls.raw_mutable_data());
    auto *authored=static_cast<std::int32_t *>(counts.raw_mutable_data());
    auto *ids=static_cast<std::int32_t *>(selected.raw_mutable_data());
    std::fill_n(pools.mutable_data(),pools.numel(),3.5f);
    std::fill_n(query.mutable_data(),query.numel(),1.5f);
    EXPECT_FALSE(controls.current_device());EXPECT_TRUE(controls.isHostValid());
    EXPECT_FALSE(pools.current_device());EXPECT_TRUE(pools.isHostValid());
    for(const int live:{4,2,0,3})
    {
        control[0]=-99;control[1]=0;control[2]=live;control[3]=-99;
        std::fill_n(ids,capacity*columns,-99);
        for(int row=0;row<capacity;++row){authored[row]=row<live?row+1:-99;
            if(row<live)for(int column=0;column<authored[row];++column)ids[row*columns+column]=column;}
        const SnapshotInt32Binding liveBinding{&controls,8};
        StageDumpInfo dump;
        dump.outputs.push_back({.name="selected",.data=selected.raw_data(),.rows=capacity,.cols=columns,.dtype="INT32",
            .element_size=4,.byte_size=selected.size_bytes(),.tensor=&selected,
            .semantic_extent=SnapshotCountedSelectionRows{liveBinding,{&counts,0}}});
        dump.outputs.push_back({.name="pooled_keys",.data=pools.raw_data(),.rows=capacity,.cols=128,.dtype="FP32",
            .element_size=4,.byte_size=pools.size_bytes(),.tensor=&pools,
            .semantic_extent=SnapshotStatePrefixRows{{&controls,4},16,4}});
        dump.outputs.push_back({.name="prepared_query",.data=query.raw_data(),.rows=capacity,.cols=512,.dtype="FP32",
            .element_size=4,.byte_size=query.size_bytes(),.tensor=&query,.semantic_extent=SnapshotTokenRows{liveBinding}});
        SnapshotCapture capture;capture.captureStage("layer3_qsa_indexer",dump);
        const auto selection=capture.getShared("layer3_QSA_SELECTED_INDICES"),pool=capture.getShared("layer3_QSA_INDEXER_POOLED_KEYS"),
            prepared=capture.getShared("layer3_QSA_INDEXER_QUERY");
        ASSERT_TRUE(selection);ASSERT_TRUE(pool);ASSERT_TRUE(prepared);
        EXPECT_EQ(selection->rows,size_t(live));EXPECT_EQ(pool->rows,size_t(live/4));EXPECT_EQ(prepared->rows,size_t(live));
        for(int row=0;row<live;++row)for(int column=0;column<columns;++column)
            EXPECT_EQ(selection->data[size_t(row)*columns+column],column<authored[row]?float(column):-1.f);
        if(live==0){EXPECT_TRUE(selection->data.empty());EXPECT_TRUE(pool->data.empty());EXPECT_TRUE(prepared->data.empty());}
        EXPECT_EQ(ids[(capacity-1)*columns+columns-1],live==capacity?float(columns-1):-99);
    }
}
/** @test Device-only payloads and count authorities cannot be read through a fabricated host view. */
TEST(SnapshotSemanticObservation, CPUObserverRejectsDeviceOnlyAuthorityWithoutRepair)
{
    std::array<float,8> payload{};
    GpuTensorView devicePayload(reinterpret_cast<void *>(0x10000),2,4,TensorType::FP32,DeviceId::cuda(0));
    StageDumpInfo::OutputBuffer output{.name="prepared_query",.data=payload.data(),.rows=2,.cols=4,.dtype="FP32",
        .element_size=4,.byte_size=32,.tensor=&devicePayload,.semantic_extent=SnapshotTokenRows{}};
    EXPECT_THROW((SnapshotSemanticObservation(output)),std::logic_error);
    GpuTensorView deviceCount(reinterpret_cast<void *>(0x20000),1,1,TensorType::INT32,DeviceId::cuda(0));
    output.tensor=nullptr;output.semantic_extent=SnapshotTokenRows{{&deviceCount,0}};
    EXPECT_THROW((SnapshotSemanticObservation(output)),std::logic_error);
}
