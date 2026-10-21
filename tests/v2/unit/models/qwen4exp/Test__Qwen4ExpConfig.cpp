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
