/**
 * @file Test__Qwen4ExpSidecarManifest.cpp
 * @brief Original-source policy, role, tokenizer and native-format pairing regressions.
 *
 * Synthetic metadata exercises the actual32-role sidecar without model loads,
 * accelerator initialization, payload hashes or numerical requantization.
 */
#include "Qwen4ExpSidecarFixture.h"
#include "models/qwen4exp/Qwen4ExpSidecarManifest.h"
#include <gtest/gtest.h>

using namespace llaminar2;
using namespace llaminar2::test;

/** @test Dense means dense; native operands and borrowed target coordinates remain unchanged. */
TEST(Qwen4ExpSidecarManifest, DenseArtifactRetainsNativeSourcesAndSharedTargetIdentities)
{
    Qwen4ExpSidecarFixture fixture;
    fixture.main.findTensor("token_embd.weight")->offset=65536;
    const auto metadata_before=fixture.predictor.metadata.at("qwen4exp.attention.compress_ratios").data;
    const Qwen4ExpSidecarManifest manifest(fixture.main,fixture.predictor);
    EXPECT_EQ(manifest.attention(),Qwen4ExpPredictorAttention::Dense);
    EXPECT_EQ(manifest.hiddenNorm(),Qwen4ExpPredictorHiddenNorm::PerBranch);
    EXPECT_FALSE(manifest.sparseSelector());
    EXPECT_EQ(manifest.sourceLayer(),48u);
    EXPECT_EQ(manifest.activations().trunk,10240u);
    EXPECT_EQ(manifest.sources().size(),34u);
    EXPECT_EQ(manifest.consumedNames().size(),30u);
    const auto &embedding=manifest.tensor("token_embd.weight");
    EXPECT_EQ(embedding.source,Qwen4ExpTensorSource::Main);
    EXPECT_EQ(embedding.tensor.offset,65536u);
    EXPECT_EQ(embedding.tensor.type,GGUFTensorType::F16);
    EXPECT_EQ(manifest.tensor("output.weight").source,Qwen4ExpTensorSource::Main);
    EXPECT_EQ(manifest.tensor("output.weight").tensor.type,GGUFTensorType::Q4_K);
    const auto &indexer=manifest.tensor("blk.48.indexer.q_proj.weight");
    EXPECT_EQ(indexer.source,Qwen4ExpTensorSource::Predictor);
    EXPECT_EQ(indexer.tensor.type,GGUFTensorType::BF16);
    EXPECT_FALSE(indexer.consumed);
    EXPECT_EQ(manifest.tensor("blk.48.ffn_gate_exps.weight").tensor.type,GGUFTensorType::Q8_0);
    EXPECT_EQ(fixture.predictor.metadata.at("qwen4exp.attention.compress_ratios").data,metadata_before);
    EXPECT_EQ(fixture.main.metadata.at("qwen4exp.block_count").asUInt64Checked(),48u);
}

/** @test A separately declared sparse head remains a distinct typed source policy. */
TEST(Qwen4ExpSidecarManifest, SparsePolicyRequiresItsOwnExplicitCompressionEntry)
{
    Qwen4ExpSidecarFixture fixture;
    auto ratios=fixture.predictor.metadata.at("qwen4exp.attention.compress_ratios").asUInt64Array();
    ratios.back()=4;
    fixture.predictor.metadata["qwen4exp.attention.compress_ratios"]=qwen4Array(GGUFValueType::UINT64,ratios);
    const Qwen4ExpSidecarManifest manifest(fixture.main,fixture.predictor);
    EXPECT_EQ(manifest.attention(),Qwen4ExpPredictorAttention::Sparse);
    ASSERT_TRUE(manifest.sparseSelector());
    EXPECT_EQ(manifest.sparseSelector()->compression_ratio,4u);
    EXPECT_EQ(manifest.consumedNames().size(),34u);
    EXPECT_TRUE(manifest.tensor("blk.48.indexer.q_proj.weight").consumed);
}

/** @test A sidecar's unused chat template cannot replace the target's prompt-rendering authority. */
TEST(Qwen4ExpSidecarManifest, TargetOwnsChatRenderingWhileTokenizationRemainsExact)
{
    Qwen4ExpSidecarFixture fixture;
    fixture.main.metadata["tokenizer.chat_template"]={.type=GGUFValueType::STRING,.data={'m','a','i','n'}};
    fixture.predictor.metadata["tokenizer.chat_template"]={.type=GGUFValueType::STRING,.data={'d','r','a','f','t'}};
    const Qwen4ExpSidecarManifest manifest(fixture.main,fixture.predictor);
    EXPECT_EQ(manifest.tensor("token_embd.weight").source,Qwen4ExpTensorSource::Main);
    EXPECT_EQ(fixture.main.metadata.at("tokenizer.chat_template").data,std::vector<uint8_t>({'m','a','i','n'}));
    EXPECT_EQ(fixture.predictor.metadata.at("tokenizer.chat_template").data,std::vector<uint8_t>({'d','r','a','f','t'}));
    fixture.predictor.metadata.at("tokenizer.ggml.tokens").string_array_value[0]="foreign";
    EXPECT_THROW((void)Qwen4ExpSidecarManifest(fixture.main,fixture.predictor),std::invalid_argument);
}

/** @test No absent role, shape/codebook mismatch or extra tensor can hide behind a shared flag. */
TEST(Qwen4ExpSidecarManifest, RejectsMalformedRoleGeometryAndNativeExtents)
{
    Qwen4ExpSidecarFixture fixture;
    auto *norm=fixture.predictor.findTensor("blk.48.nextn.hnorm.weight");
    norm->type=GGUFTensorType::F16;
    EXPECT_THROW((void)Qwen4ExpSidecarManifest(fixture.main,fixture.predictor),std::invalid_argument);
    norm->type=GGUFTensorType::F32;
    auto *experts=fixture.predictor.findTensor("blk.48.ffn_gate_exps.weight");
    std::swap(experts->dimensions[0],experts->dimensions[2]);
    EXPECT_THROW((void)Qwen4ExpSidecarManifest(fixture.main,fixture.predictor),std::invalid_argument);
    std::swap(experts->dimensions[0],experts->dimensions[2]);
    --experts->size_bytes;
    EXPECT_THROW((void)Qwen4ExpSidecarManifest(fixture.main,fixture.predictor),std::invalid_argument);
    ++experts->size_bytes;
    fixture.predictor.tensors.push_back(fixture.predictor.tensors.front());
    EXPECT_THROW((void)Qwen4ExpSidecarManifest(fixture.main,fixture.predictor),std::invalid_argument);
    fixture.predictor.tensors.pop_back();
    fixture.predictor.tensors.pop_back();
    EXPECT_THROW((void)Qwen4ExpSidecarManifest(fixture.main,fixture.predictor),std::invalid_argument);
}

/** @test Every shared tokenizer ID and exact metadata scalar must name the same target. */
TEST(Qwen4ExpSidecarManifest, RejectsForeignTokenizerAndAmbiguousOrUnsupportedPolicies)
{
    Qwen4ExpSidecarFixture fixture;
    auto &tokens=fixture.predictor.metadata.at("tokenizer.ggml.tokens").string_array_value;
    tokens.back()="foreign";
    EXPECT_THROW((void)Qwen4ExpSidecarManifest(fixture.main,fixture.predictor),std::invalid_argument);
    tokens.back().clear();
    auto &shared=fixture.predictor.metadata.at("qwen4exp.nextn_shared_target_tensors");
    shared.type=GGUFValueType::UINT8;
    EXPECT_THROW((void)Qwen4ExpSidecarManifest(fixture.main,fixture.predictor),std::invalid_argument);
    shared.type=GGUFValueType::BOOL;
    auto ratios=fixture.predictor.metadata.at("qwen4exp.attention.compress_ratios").asUInt64Array();
    ratios.back()=2;
    fixture.predictor.metadata["qwen4exp.attention.compress_ratios"]=qwen4Array(GGUFValueType::UINT64,ratios);
    EXPECT_THROW((void)Qwen4ExpSidecarManifest(fixture.main,fixture.predictor),std::invalid_argument);
    ratios.back()=0;ratios.front()=4;
    fixture.predictor.metadata["qwen4exp.attention.compress_ratios"]=qwen4Array(GGUFValueType::UINT64,ratios);
    EXPECT_THROW((void)Qwen4ExpSidecarManifest(fixture.main,fixture.predictor),std::invalid_argument);
}
