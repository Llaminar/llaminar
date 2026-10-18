/**
 * @file Test__FusedNormSnapshotPublication.cpp
 * @brief Authenticate canonical residuals at their actual fused graph producers.
 *
 * Output-specific filters must select a residual even when its normalization
 * result is outside the requested layer cohort. Names, row ownership, immutable
 * lifetime and transaction namespaces survive singleton and reordered views;
 * no tensor is reconstructed with a host residual add.
 */
#include "snapshots/SnapshotCapture.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <vector>

using namespace llaminar2;
namespace
{
    /** @brief Make one concrete named host-observed output without hidden arithmetic.
     * @param name Producer's actual output name.
     * @param values Immutable observed FP32 values, one two-feature token row.
     * @return Descriptor suitable for single-output capture/filtering. */
    StageDumpInfo oneOutput(const char *name, const std::vector<float> &values)
    {
        StageDumpInfo dump;
        dump.addOutput(name, values.data(), values.size()/2, 2);
        return dump;
    }

    /** @brief Test the same exact-key selection used by descriptor-filtered capture.
     * @param stage Actual producer name, possibly across the layer-cohort boundary.
     * @param dump Actual output view; it may contain a single selected slot.
     * @param requested Canonical semantic key requested by parity inventory.
     * @return Whether this producer/output really owns the requested key. */
    bool selects(const std::string &stage, const StageDumpInfo &dump, const std::string &requested)
    {
        const auto keys=SnapshotCapture::possibleKeysForStage(stage,dump);
        return std::find(keys.begin(),keys.end(),requested)!=keys.end();
    }
}

/** @test Singleton residual/norm views and reordered outputs retain their semantics. */
TEST(FusedNormSnapshotPublication, NamedOutputsSurviveSingletonAndReorderedFilters)
{
    const std::vector<float> residual{-0.f,2.f,3.f,4.f},normalized{10.f,20.f,30.f,40.f};
    for (const auto &[stage,key] : std::array<std::pair<std::string,std::string>,3>{{
            {"layer0_ffn_norm","layer0_ATTENTION_RESIDUAL"},
            {"layer1_attn_norm","layer0_FFN_RESIDUAL"},
            {"layer8_attn_norm","layer7_FFN_RESIDUAL"}}})
    {
        const auto residual_dump=oneOutput("residual_out",residual);
        const auto norm_dump=oneOutput("norm_output",normalized);
        const auto norm_key=SnapshotCapture::convertStageNameToSnapshotKey(stage);
        EXPECT_TRUE(selects(stage,residual_dump,key));
        EXPECT_FALSE(selects(stage,norm_dump,key));
        EXPECT_TRUE(selects(stage,norm_dump,norm_key));
        EXPECT_FALSE(selects(stage,residual_dump,norm_key));
        for (const int variant : {0,1,2})
        {
            auto dump=variant==0 ? residual_dump : norm_dump;
            if (variant==2) dump.outputs.push_back(residual_dump.outputs.front());
            SnapshotCapture capture;
            capture.captureStage(stage,dump);
            const auto actual=capture.getShared(key);
            if (variant==1) EXPECT_FALSE(actual);
            else { ASSERT_TRUE(actual); EXPECT_EQ(actual->data,residual);
                EXPECT_EQ(std::memcmp(actual->data.data(),residual.data(),residual.size()*sizeof(float)),0); }
            const auto norm=capture.getShared(norm_key);
            if (variant==0) EXPECT_FALSE(norm);
            else { ASSERT_TRUE(norm); EXPECT_EQ(norm->data,normalized); }
        }
    }
}

/** @test Canonical and diagnostic aliases share typed immutable payloads and contexts. */
TEST(FusedNormSnapshotPublication, AliasesPreserveOwnershipAndTransactionLifetime)
{
    const std::vector<float> values{1.f,2.f,3.f,4.f,5.f,6.f};
    auto dump=oneOutput("residual_out",values);
    dump.outputs.front().row_layout=SnapshotOwnedRows{{{1,2}}};
    SnapshotCapture capture;
    capture.captureStage("prefill_chunk_3::layer8_attn_norm",dump);
    const auto legacy=capture.getShared("PREFILL_CHUNK_3_layer8_ATTENTION_NORM_RESIDUAL_OUT");
    const auto canonical=capture.getShared("PREFILL_CHUNK_3_layer7_FFN_RESIDUAL");
    ASSERT_TRUE(legacy);ASSERT_TRUE(canonical);
    EXPECT_EQ(legacy.get(),canonical.get());
    EXPECT_EQ(canonical->publication,SnapshotPublication::RowPartition);
    EXPECT_EQ(canonical->row_ownership,(SnapshotOwnedRows{{{1,2}}}));
    EXPECT_EQ(canonical->rows,3u);EXPECT_EQ(canonical->cols,2u);
    EXPECT_TRUE(selects("prefill_chunk_3::layer8_attn_norm",dump,
                        "PREFILL_CHUNK_3_layer7_FFN_RESIDUAL"));
    capture.clear();
    EXPECT_EQ(canonical->data,values);
    EXPECT_FALSE(capture.getShared("layer7_FFN_RESIDUAL"));
}

/** @test Only exact main-layer indices may name the previous layer's residual. */
TEST(FusedNormSnapshotPublication, MainLayerParsingCannotCrossMTPDepthsOrInvalidPrefixes)
{
    const std::vector<float> values{1.f,2.f};
    const auto dump=oneOutput("residual_out",values);
    for (const std::string stage : {"layer0_attn_norm","MTP0_attn_norm","MTP2_attn_norm",
            "MTP15_ffn_norm","layer-1_attn_norm","layer-0_ffn_norm","layer01_attn_norm",
            "layer2147483648_attn_norm","remote_layer8_attn_norm","layerx_ffn_norm"})
    {
        const auto keys=SnapshotCapture::possibleKeysForStage(stage,dump);
        ASSERT_EQ(keys.size(),1u) << stage;
        EXPECT_TRUE(keys.front().ends_with("_RESIDUAL_OUT"));
        SnapshotCapture capture;capture.captureStage(stage,dump);
        EXPECT_EQ(capture.keys(),keys);
    }
    SnapshotCapture capture;
    capture.captureStage("mtp_draft_depth2::layer8_attn_norm",dump);
    EXPECT_TRUE(capture.getShared("MTP_DRAFT_DEPTH2_layer7_FFN_RESIDUAL"));
    EXPECT_FALSE(capture.getShared("layer7_FFN_RESIDUAL"));
    EXPECT_FALSE(capture.getShared("MTP1_FFN_RESIDUAL"));
    // A plain RMSNorm has only its declared output and no residual publication.
    const auto ordinary=oneOutput("output",values);
    EXPECT_EQ(SnapshotCapture::possibleKeysForStage("layer0_attn_norm",ordinary),
              (std::vector<std::string>{"layer0_ATTENTION_NORM"}));
}

/** @test All eighty canonical residual checkpoints have a real selected producer. */
TEST(FusedNormSnapshotPublication, FortyLayerInventorySelectsFusedAndTerminalProducers)
{
    SnapshotCapture capture;
    const std::vector<float> values{-0.f,2.f};
    const auto residual=oneOutput("residual_out",values);
    for (int layer=0;layer<40;++layer)
    {
        const auto prefix="layer"+std::to_string(layer);
        ASSERT_TRUE(selects(prefix+"_ffn_norm",residual,prefix+"_ATTENTION_RESIDUAL"));
        capture.captureStage(prefix+"_ffn_norm",residual);
        if (layer>0)
        {
            const auto previous="layer"+std::to_string(layer-1)+"_FFN_RESIDUAL";
            ASSERT_TRUE(selects(prefix+"_attn_norm",residual,previous));
            capture.captureStage(prefix+"_attn_norm",residual);
        }
    }
    const auto terminal=oneOutput("output",values);
    EXPECT_TRUE(selects("layer39_ffn_residual",terminal,"layer39_FFN_RESIDUAL"));
    capture.captureStage("layer39_ffn_residual",terminal);
    for (int layer=0;layer<40;++layer)
    for (const std::string suffix : {"_ATTENTION_RESIDUAL","_FFN_RESIDUAL"})
    {
        const auto actual=capture.getShared("layer"+std::to_string(layer)+suffix);
        ASSERT_TRUE(actual);EXPECT_EQ(actual->data,values);
        EXPECT_EQ(std::memcmp(actual->data.data(),values.data(),values.size()*sizeof(float)),0);
    }
}
