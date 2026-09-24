/**
 * @file Test__PlanningModelMetadata.cpp
 * @brief Device-free descriptor identity, geometry and publication contracts.
 *
 * Synthetic metadata keeps these tests independent of model files and MPI.
 * Real multi-rank root-only reading and failure consensus are covered in the
 * existing initialization-lifecycle Integration gate, not simulated here.
 */
#include "planning/PlanningModelMetadata.h"
#include "execution/mpi_orchestration/IExecutionPlanBuilder.h"
#include "../../utils/PlanningGGUFFixture.h"
#include <gtest/gtest.h>

using namespace llaminar2;

namespace
{
    /** @return Valid dense/MoE geometry with an appended learned-MTP block. */
    ModelMemoryProfile metadataProfile()
    {
        ModelMemoryProfile profile;
        profile.architecture = "qwen35moe";
        profile.n_layers = 49;
        profile.d_model = 2048;
        profile.d_ff = 6144;
        profile.n_heads = 16;
        profile.n_kv_heads = 4;
        profile.head_dim = 128;
        profile.vocab_size = 256000;
        profile.max_seq_len = 32768;
        profile.expert_count = 256;
        profile.expert_used_count = 8;
        profile.expert_feed_forward_length = 512;
        profile.mtp_layer_count = 1;
        profile.total_native_bytes = 987654321;
        profile.tensors = {{.name = "blk.0.ffn_gate_exps.weight", .native_bytes = 1000,
            .quant_type = "IQ3_S", .elements = 4096, .K = 256, .layer_index = 0}};
        return profile;
    }
}

TEST(PlanningModelMetadata, ExecutionViewRetainsGeometryWithoutCountingMTPAsMainLayers)
{
    const PlanningModelMetadata metadata(metadataProfile(), 48);
    const auto model = metadata.executionModelConfig();
    EXPECT_EQ(model.n_layers, 48);
    EXPECT_EQ(metadata.memoryProfile().n_layers, 49);
    EXPECT_EQ(model.name, "qwen35moe");
    EXPECT_EQ(model.n_heads, 16);
    EXPECT_EQ(model.n_kv_heads, 4);
    EXPECT_EQ(model.hidden_size, 2048);
    EXPECT_EQ(model.intermediate_size, 6144);
    EXPECT_EQ(model.vocab_size, 256000);
    EXPECT_EQ(model.head_dim, 128);
    EXPECT_EQ(model.estimated_weight_bytes, 987654321u);
}

TEST(PlanningModelMetadata, DescriptorRoundTripPreservesExactProfileBytesAndMainBoundary)
{
    for (const auto format : {"F32", "F16", "BF16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
            "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ1_S", "IQ1_M", "IQ2_XXS",
            "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_NL", "IQ4_XS"})
    {
        auto profile = metadataProfile();
        profile.tensors.front().quant_type = format;
        const PlanningModelMetadata metadata(profile, 48);
        const auto bytes = metadata.serialize();
        const auto restored = PlanningModelMetadata::deserialize(bytes);
        EXPECT_EQ(restored.mainLayerCount(), 48);
        EXPECT_EQ(restored.memoryProfile().serialize(), profile.serialize());
        EXPECT_EQ(restored.serialize(), bytes);
    }
}

TEST(PlanningModelMetadata, TruncationOrStaleDescriptorCannotBecomeAValidModel)
{
    const auto bytes = PlanningModelMetadata(metadataProfile(), 48).serialize();
    for (size_t length = 0; length < bytes.size(); ++length)
    {
        SCOPED_TRACE(length);
        EXPECT_THROW((void)PlanningModelMetadata::deserialize(std::span(bytes).first(length)), std::exception);
    }
    auto stale = bytes;
    ++stale[3];
    EXPECT_THROW((void)PlanningModelMetadata::deserialize(stale), std::runtime_error);
    auto overflow = bytes;
    overflow[7] = 255;
    EXPECT_THROW((void)PlanningModelMetadata::deserialize(overflow), std::runtime_error);
    auto trailing = bytes;
    trailing.push_back(0);
    EXPECT_THROW((void)PlanningModelMetadata::deserialize(trailing), std::runtime_error);
}

TEST(PlanningModelMetadata, InvalidGeometryFailsAtDescriptorConstruction)
{
    for (const int main_layers : {-1, 0, 50})
        EXPECT_THROW((void)PlanningModelMetadata(metadataProfile(), main_layers), std::invalid_argument);
    for (auto member : {&ModelMemoryProfile::n_heads, &ModelMemoryProfile::n_kv_heads, &ModelMemoryProfile::d_model})
    {
        auto profile = metadataProfile();
        profile.*member = 0;
        EXPECT_THROW((void)PlanningModelMetadata(profile, 48), std::invalid_argument);
    }
}

TEST(PlanningModelMetadata, ProcessLocalPublicationCallsTheReaderExactlyOnce)
{
    int calls = 0;
    const auto metadata = exchangePlanningModelMetadata(nullptr, [&] {
        ++calls;
        return PlanningModelMetadata(metadataProfile(), 48);
    });
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(metadata.mainLayerCount(), 48);
    EXPECT_THROW((void)exchangePlanningModelMetadata(nullptr, []() -> PlanningModelMetadata {
        throw std::runtime_error("injected metadata failure");
    }), std::runtime_error);
}

TEST(PlanningModelMetadata, MissingFileIsAHardReadFailure)
{
    EXPECT_THROW((void)readPlanningModelMetadata(""), std::invalid_argument);
    EXPECT_THROW((void)readPlanningModelMetadata("/llaminar-metadata-test/nonexistent.gguf"), std::runtime_error);
}

TEST(PlanningModelMetadata, RetainedSourceDoesNotReopenOrRelocateItsLoader)
{
    test::PlanningGGUFFixture fixture(true, true);
    PlanningModelSource source(fixture.path());
    const auto *loader = &source.loader();
    EXPECT_FALSE(loader->usesMmap());
    EXPECT_FALSE(loader->isMmapActive());
    const auto bytes = source.metadata().serialize();
    EXPECT_EQ(source.metadata().mainLayerCount(), 2);
    EXPECT_EQ(source.metadata().memoryProfile().n_layers, 3);
    // Removing this uniquely owned fixture after metadata parsing proves that
    // repeated planning uses the retained directory, not a hidden second read.
    ASSERT_TRUE(std::filesystem::remove(fixture.path()));
    PlanningModelSource moved(std::move(source));
    EXPECT_EQ(&moved.loader(), loader);
    EXPECT_EQ(moved.metadata().serialize(), bytes);
    EXPECT_THROW((void)source.loader(), std::logic_error);
    EXPECT_THROW((void)source.metadata(), std::logic_error);
    EXPECT_THROW((void)source.path(), std::logic_error);
    test::PlanningGGUFFixture other;
    PlanningModelSource assigned(other.path());
    assigned = std::move(moved);
    EXPECT_EQ(&assigned.loader(), loader);
    EXPECT_EQ(assigned.metadata().serialize(), bytes);
    EXPECT_THROW((void)moved.loader(), std::logic_error);
}

TEST(PlanningModelMetadata, UploadExtentUsesTheWholeDirectoryAndRejectsEmptyPayloads)
{
    GGUFModel model;
    EXPECT_THROW((void)maximumGGUFTensorPayloadBytes(model), std::invalid_argument);
    model.tensors = {{.name = "first", .size_bytes = 123}, {.name = "largest", .size_bytes = 456},
        {.name = "last", .size_bytes = 10}};
    EXPECT_EQ(maximumGGUFTensorPayloadBytes(model), 456u);
}
