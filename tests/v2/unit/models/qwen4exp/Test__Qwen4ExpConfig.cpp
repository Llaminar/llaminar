/**
 * @file Test__Qwen4ExpConfig.cpp
 * @brief Device-free regressions for exact qwen4exp metadata and native row geometry.
 *
 * These tests authenticate configuration ownership and invalid metadata rejection;
 * they do not claim model arithmetic or backend generation certification.
 */
#include "Qwen4ExpMetadataFixture.h"
#include <gtest/gtest.h>
#include <limits>

namespace llaminar2::test
{
    TEST(Qwen4ExpConfig, DistinguishesComputeTrunkHeadAndPredictorGeometry)
    {
        const Qwen4ExpConfig config(qwen4MetadataFixture());
        EXPECT_EQ(config.activations().compute, 2560u);
        EXPECT_EQ(config.activations().trunk, 10240u);
        EXPECT_EQ(config.activations().terminal_head, 2560u);
        EXPECT_EQ(config.hyperConnections().gate_rank, 320u);
        EXPECT_EQ(config.mainLayers(), 48u);
        EXPECT_EQ(config.predictorLayers(), 1u);
        EXPECT_EQ(config.mixers().size(), 49u);
        EXPECT_EQ(config.mixers()[0], Qwen4ExpMixer::GatedDeltaNet);
        EXPECT_EQ(config.mixers()[3], Qwen4ExpMixer::SparseAttention);
        EXPECT_EQ(config.mixers()[48], Qwen4ExpMixer::SparseAttention);
        EXPECT_EQ(config.experts().count, 512u);
        EXPECT_EQ(config.experts().selected, 10u);
        EXPECT_EQ(config.deltaNet().value_width, 6144u);
    }

    /** @test Retained GDN requires original Q8 QKV/Z and prepared half controls. */
    TEST(Qwen4ExpConfig, SelectsRetainedGDNPerOriginalSourcesAndArithmetic)
    {
        using Format = GGUFTensorType;
        using Arithmetic = GDNProjectionArithmetic;
        auto model = qwen4MetadataFixture();
        const auto add = [&](int layer, const char *suffix, Format format) {
            model.tensors.push_back({.name = "blk." + std::to_string(layer) + "." + suffix,
                .dimensions = {128, 2560}, .type = format});
        };
        const auto source = [&](int layer, Format qkv, Format gate, Format alpha, Format beta) {
            add(layer, "attn_qkv.weight", qkv);
            add(layer, "attn_gate.weight", gate);
            add(layer, "ssm_alpha.weight", alpha);
            add(layer, "ssm_beta.weight", beta);
        };
        source(0, Format::Q8_0, Format::Q8_0, Format::F16, Format::F32);
        source(1, Format::Q8_0, Format::IQ4_NL, Format::F16, Format::F16);
        source(2, Format::Q6_K, Format::Q8_0, Format::F16, Format::F16);
        source(3, Format::Q8_0, Format::Q8_0, Format::F16, Format::F16);
        source(4, Format::Q8_0, Format::Q8_0, Format::Q8_0, Format::F16);
        source(5, Format::Q8_0, Format::Q8_0, Format::F16, Format::Q8_0);
        add(6, "attn_qkv.weight", Format::Q8_0);
        add(6, "attn_gate.weight", Format::Q8_0);
        add(6, "ssm_alpha.weight", Format::F16);
        source(8, Format::Q8_0, Format::Q8_0, Format::BF16, Format::BF16);
        const Qwen4ExpConfig config(model);
        for (const auto policy : {Arithmetic::FP16MoEAlphaBeta, Arithmetic::FP16AlphaBeta})
        {
            EXPECT_TRUE(config.supportsRetainedGDN(0, policy));
            EXPECT_TRUE(config.supportsRetainedGDN(8, policy));
            for (const int layer : {1, 2, 3, 4, 5, 6})
                EXPECT_FALSE(config.supportsRetainedGDN(layer, policy)) << layer;
        }
        for (const auto policy : {Arithmetic::Native, Arithmetic::FP32, Arithmetic::FP16})
            EXPECT_FALSE(config.supportsRetainedGDN(0, policy));
    }

    /** @test Retained HC/FFN selects only complete original measured source tuples in main blocks. */
    TEST(Qwen4ExpConfig, SelectsRetainedM1HCFFNPerOriginalSource)
    {
        using Format = GGUFTensorType;
        auto model = qwen4MetadataFixture();
        const auto add = [&](int layer, const char *role, Format format) {
            model.tensors.push_back({.name = "blk." + std::to_string(layer) + "." + role,
                .dimensions = {128, 2560}, .type = format});
        };
        const auto complete_layer = [&](int layer, Format gate, Format up, Format down,
                                         Format shared_down = Format::Q8_0, Format hc_up = Format::Q8_0) {
            add(layer, "hc_ffn_down.weight", Format::Q8_0);
            add(layer, "hc_ffn_up.weight", hc_up);
            add(layer, "ffn_gate_shexp.weight", Format::Q8_0);
            add(layer, "ffn_up_shexp.weight", Format::Q8_0);
            add(layer, "ffn_down_shexp.weight", shared_down);
            add(layer, "ffn_gate_exps.weight", gate);
            add(layer, "ffn_up_exps.weight", up);
            add(layer, "ffn_down_exps.weight", down);
        };
        complete_layer(0, Format::Q4_K, Format::Q4_K, Format::Q5_1);
        complete_layer(1, Format::Q4_K, Format::Q4_K, Format::Q8_0);
        complete_layer(2, Format::Q5_K, Format::Q5_K, Format::Q8_0);
        complete_layer(3, Format::IQ2_S, Format::IQ3_S, Format::IQ4_NL, Format::Q6_K);
        complete_layer(4, Format::Q4_K, Format::Q5_K, Format::Q8_0);
        complete_layer(5, Format::Q5_K, Format::Q5_K, Format::Q5_1);
        complete_layer(6, Format::Q4_K, Format::Q4_K, Format::Q5_1, Format::Q6_K);
        complete_layer(7, Format::Q4_K, Format::Q4_K, Format::Q5_1, Format::Q8_0, Format::Q6_K);
        complete_layer(48, Format::Q4_K, Format::Q4_K, Format::Q5_1);
        const Qwen4ExpConfig config(model);
        for (int layer = 0; layer < 9; ++layer)
        {
            SCOPED_TRACE(layer);
            EXPECT_EQ(config.hasRetainedM1HCFFNSources(layer), layer < 3);
        }
        EXPECT_FALSE(config.hasRetainedM1HCFFNSources(48));
    }

    TEST(Qwen4ExpConfig, PreservesSparseTailAndDilatedNgramHistory)
    {
        const Qwen4ExpConfig config(qwen4MetadataFixture());
        EXPECT_EQ(config.sparseAttention().selected_complete_blocks, 512u);
        EXPECT_EQ(config.sparseAttention().maximum_selected_tokens, 2051u);
        EXPECT_EQ(config.sparseAttention().indexer_key_heads, 1u);
        EXPECT_EQ(config.ngrams().decoder_layer, 1u);
        EXPECT_EQ(config.ngrams().heads, 16u);
        EXPECT_EQ(config.ngrams().convolution_dilation, 3u);
        EXPECT_EQ(config.ngrams().convolution_history, 9u);
        EXPECT_EQ(config.ngrams().hash_reset_token, 248044u);
        EXPECT_EQ(Qwen4ExpConfig::generationStopTokens[1], 248046u);
        EXPECT_EQ(config.ngrams().multipliers[2], (uint64_t{1} << 44) + 7);
        EXPECT_EQ(config.ngrams().table.type, GGUFTensorType::IQ4_NL);
        EXPECT_EQ(config.ngrams().row_bytes, 90u);
        EXPECT_EQ(config.ngrams().table.size_bytes, 28800138240ULL);
        EXPECT_EQ(config.ngrams().table.split_idx, 1u);
    }

    TEST(Qwen4ExpConfig, RecognizesIntentionallyOmittedPredictor)
    {
        const Qwen4ExpConfig config(qwen4MetadataFixture(false));
        EXPECT_EQ(config.predictorLayers(), 0u);
        EXPECT_EQ(config.mixers().size(), 48u);
    }

    TEST(Qwen4ExpConfig, RequiredMetadataNeverUsesGeometryDefaults)
    {
        const auto complete = qwen4MetadataFixture();
        for (const auto &[key, value] : complete.metadata)
        {
            if (key == "qwen4exp.nextn_predict_layers" || key == "qwen4exp.attention.recurrent_layers")
                continue;
            SCOPED_TRACE(key);
            auto missing = complete;
            missing.metadata.erase(key);
            EXPECT_THROW((void)Qwen4ExpConfig(missing), std::invalid_argument);
        }
    }

    TEST(Qwen4ExpConfig, RejectsOtherArchitectureAndInvalidTrunkSparseOrPleMetadata)
    {
        const std::vector<std::pair<std::string, uint32_t>> changes{
            {"embedding_length", 10240}, {"hyper_connection.count", 1},
            {"attention.indexer.top_k", 512}, {"ple.eos_token_id", 248046},
            {"expert_count", 256}, {"block_count", 48}};
        for (const auto &[suffix, value] : changes)
        {
            SCOPED_TRACE(suffix);
            auto model = qwen4MetadataFixture();
            model.metadata["qwen4exp." + suffix] = qwen4Scalar(GGUFValueType::UINT32, value);
            EXPECT_THROW((void)Qwen4ExpConfig(model), std::invalid_argument);
        }
        auto other = qwen4MetadataFixture();
        other.architecture = "qwen35moe";
        EXPECT_THROW((void)Qwen4ExpConfig(other), std::invalid_argument);
        other = qwen4MetadataFixture();
        other.metadata["qwen4exp.ple.layers"] = qwen4Array<uint32_t>(GGUFValueType::UINT32, {2});
        EXPECT_THROW((void)Qwen4ExpConfig(other), std::invalid_argument);
    }

    TEST(Qwen4ExpConfig, RejectsSparseOwnershipAndProjectionShapeDrift)
    {
        auto model = qwen4MetadataFixture();
        model.metadata["qwen4exp.attention.recurrent_layers"].data[3] = 1;
        EXPECT_THROW((void)Qwen4ExpConfig(model), std::invalid_argument);
        model = qwen4MetadataFixture();
        model.findTensor("blk.3.indexer.k_proj.weight")->dimensions[0] = 256;
        EXPECT_THROW((void)Qwen4ExpConfig(model), std::invalid_argument);
        model = qwen4MetadataFixture();
        model.metadata["qwen4exp.attention.compress_ratios"].array_length = 48;
        EXPECT_THROW((void)Qwen4ExpConfig(model), std::invalid_argument);
    }

    TEST(Qwen4ExpConfig, RejectsLossyHashArraysAndInvalidHeadRanges)
    {
        auto model = qwen4MetadataFixture();
        model.metadata["qwen4exp.ple.layer_multipliers"] = qwen4Array<float>(GGUFValueType::FLOAT32, {1, 2, 3});
        EXPECT_THROW((void)Qwen4ExpConfig(model), std::invalid_argument);
        model = qwen4MetadataFixture();
        auto offsets = model.metadata["qwen4exp.ple.head_offsets"].asUInt64Array();
        offsets[1] = 0;
        model.metadata["qwen4exp.ple.head_offsets"] = qwen4Array(GGUFValueType::UINT64, offsets);
        EXPECT_THROW((void)Qwen4ExpConfig(model), std::invalid_argument);
        offsets.back() = std::numeric_limits<uint64_t>::max();
        model.metadata["qwen4exp.ple.head_offsets"] = qwen4Array(GGUFValueType::UINT64, offsets);
        EXPECT_THROW((void)Qwen4ExpConfig(model), std::invalid_argument);
    }

    TEST(Qwen4ExpConfig, UsesActualTableFormatAndRejectsIncorrectRowExtents)
    {
        for (auto format : {GGUFTensorType::F32, GGUFTensorType::F16, GGUFTensorType::BF16,
                            GGUFTensorType::Q4_0, GGUFTensorType::IQ4_NL})
        {
            auto model = qwen4MetadataFixture();
            auto *table = model.findTensor("per_layer_token_embd.weight");
            table->type = format;
            const uint64_t block = table->isQuantized() ? table->getBlockSize() : 1;
            const uint64_t row_bytes = 160 / block * table->getTypeSize();
            table->size_bytes = table->dimensions[0] * row_bytes;
            const Qwen4ExpConfig config(model);
            EXPECT_EQ(config.ngrams().table.type, format);
            EXPECT_EQ(config.ngrams().row_bytes, row_bytes);
        }
        auto model = qwen4MetadataFixture();
        model.findTensor("per_layer_token_embd.weight")->type = GGUFTensorType::Q4_K;
        EXPECT_THROW((void)Qwen4ExpConfig(model), std::invalid_argument);
        model = qwen4MetadataFixture();
        model.findTensor("per_layer_token_embd.weight")->size_bytes -= 1;
        EXPECT_THROW((void)Qwen4ExpConfig(model), std::invalid_argument);
        model = qwen4MetadataFixture();
        model.findTensor("per_layer_token_embd.weight")->dimensions = {160, 320001536};
        EXPECT_THROW((void)Qwen4ExpConfig(model), std::invalid_argument);
    }
}
