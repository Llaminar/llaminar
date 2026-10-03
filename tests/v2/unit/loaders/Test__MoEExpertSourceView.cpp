/**
 * @file Test__MoEExpertSourceView.cpp
 * @brief Raw-source slicing proofs across every expert format, without devices.
 *
 * Preparation must use the requested logical expert and original down-output
 * interval even when a frozen parent has already compacted its expert axis.
 * These tests authenticate the view's raw byte address, extent and source
 * identity rather than accepting numerically similar re-quantized weights.
 */
#include "loaders/MoEExpertSourceView.h"
#include "utils/QuantizedVerifierFormats.h"

#include <gtest/gtest.h>
#include <array>
#include <cstring>

namespace llaminar2::test
{
TEST(MoEExpertSourceView, LogicalExpertsResolveUnpackedAndCompactedParents)
{
    const std::array<size_t, 3> full{256, 512, 8};
    const MoEExpertSourceView contiguous(full, WeightSliceSpec{.expert_start = 3, .expert_count = 2});
    EXPECT_EQ(contiguous.experts(), (std::vector<int>{3, 4}));
    EXPECT_EQ(contiguous.matrix(4, WeightRole::MoEExpertDown).element_offset, 4u * 512 * 256);
    EXPECT_THROW(contiguous.matrix(0, WeightRole::MoEExpertDown), std::invalid_argument);
    const MoEExpertSourceView explicit_full(full, WeightSliceSpec{.expert_ids = {1, 7}});
    EXPECT_EQ(explicit_full.matrix(7, WeightRole::MoEExpertDown).element_offset, 7u * 512 * 256);
    const MoEExpertSourceView packed(std::array<size_t, 3>{256, 512, 2},
        WeightSliceSpec{.expert_count = 2, .expert_ids = {1, 7}, .inner_is_presliced = true});
    const auto selected = packed.matrix(7, WeightRole::MoEExpertDown);
    EXPECT_EQ(selected.element_offset, 512u * 256);
    EXPECT_EQ(selected.source_identity.expert_ids, (std::vector<int>{7}));
    EXPECT_EQ(selected.source_identity.expert_start, 7u);
    EXPECT_EQ(selected.source_identity.expert_count, 1u);
    EXPECT_TRUE(selected.source_identity.inner_is_presliced);
}

TEST(MoEExpertSourceView, RejectsMissingMetadataAndWrongPartitions)
{
    const auto owner = MoEExpertProjectionOwnership::gateUpOwnedDownColumns({8, 512, 256}, 1, 4);
    // WeightManager's fullSliceSpec retains GGUF [K,N,E] axes. Use its
    // non-square identity here; an empty or square fixture hides axis swaps.
    const MoEExpertSourceView full(std::array<size_t, 3>{256, 512, 8},
        WeightSliceSpec{.source_rows = 256, .source_cols = 512,
            .row_count = 256, .col_count = 512, .expert_count = 8});
    const auto complete = full.matrix(7, WeightRole::MoEExpertDown);
    EXPECT_EQ(complete.source_identity.source_rows, 512u);
    EXPECT_EQ(complete.source_identity.source_cols, 256u);
    const auto view = full.matrix(7, WeightRole::MoEExpertDown, &owner);
    EXPECT_EQ(view.rows, 128u);
    EXPECT_EQ(view.columns, 256u);
    EXPECT_EQ(view.element_offset, (7u * 512 + 128) * 256);
    EXPECT_EQ(view.source_identity.source_rows, 512u);
    EXPECT_EQ(view.source_identity.row_start, 128u);
    EXPECT_EQ(view.source_identity.row_count, 128u);

    const WeightSliceSpec identity{.source_rows = 256, .source_cols = 512,
        .row_count = 256, .col_start = 128, .col_count = 128,
        .expert_count = 2, .expert_ids = {1, 7}, .inner_is_presliced = true};
    const MoEExpertSourceView sliced(std::array<size_t, 3>{256, 128, 2}, identity);
    EXPECT_EQ(sliced.matrix(7, WeightRole::MoEExpertDown, &owner).element_offset, 128u * 256);
    const auto neighbour = MoEExpertProjectionOwnership::gateUpOwnedDownColumns({8, 512, 256}, 2, 4);
    EXPECT_THROW(sliced.matrix(7, WeightRole::MoEExpertDown, &neighbour), std::invalid_argument);
    auto missing_identity = identity;
    missing_identity.source_cols = 0;
    const MoEExpertSourceView ambiguous(std::array<size_t, 3>{256, 128, 2}, missing_identity);
    EXPECT_THROW(ambiguous.matrix(7, WeightRole::MoEExpertDown, &owner), std::invalid_argument);
    const MoEExpertSourceView k_shard(std::array<size_t, 3>{128, 512, 8}, {});
    EXPECT_THROW(k_shard.matrix(7, WeightRole::MoEExpertDown, &owner), std::invalid_argument);
    const MoEExpertSourceView wrong_inventory(std::array<size_t, 3>{256, 512, 9}, {});
    EXPECT_THROW(wrong_inventory.matrix(0, WeightRole::MoEExpertDown, &owner), std::invalid_argument);
    const MoEExpertSourceView wrong_selected_inventory(std::array<size_t, 3>{256, 512, 2},
        WeightSliceSpec{.expert_ids = {0, 8}, .inner_is_presliced = true});
    EXPECT_THROW(wrong_selected_inventory.matrix(0, WeightRole::MoEExpertDown, &owner), std::invalid_argument);
    EXPECT_THROW(full.matrix(0, WeightRole::Other), std::invalid_argument);
    EXPECT_THROW((MoEExpertSourceView(std::array<size_t, 3>{256, 512, 8},
        WeightSliceSpec{.expert_start = 7, .expert_count = 2})), std::invalid_argument);
    EXPECT_THROW((MoEExpertSourceView(std::array<size_t, 3>{256, 512, 2},
        WeightSliceSpec{.expert_ids = {7, 1}, .inner_is_presliced = true})), std::invalid_argument);
    EXPECT_THROW((MoEExpertSourceView(std::array<size_t, 3>{256, 512, 2},
        WeightSliceSpec{.expert_ids = {1, 1}, .inner_is_presliced = true})), std::invalid_argument);
    EXPECT_THROW((MoEExpertSourceView(std::array<size_t, 3>{256, 512, 2},
        WeightSliceSpec{.expert_count = 1, .expert_ids = {1, 7}, .inner_is_presliced = true})), std::invalid_argument);
    EXPECT_THROW((MoEExpertSourceView(std::array<size_t, 3>{256, 512, 2},
        WeightSliceSpec{.expert_count = 1, .inner_is_presliced = true})), std::invalid_argument);
    EXPECT_THROW((MoEExpertSourceView(std::array<size_t, 3>{std::numeric_limits<size_t>::max(), 512, 2}, {})),
        std::overflow_error);
}

TEST(MoEExpertSourceView, AllFormatsUseExactOriginalRowsWithoutRepacking)
{
    std::vector<QuantizedVerifierWeightCreator> creators;
    std::vector<std::string> names;
    for (const auto &format : quantizedVerifierFormats())
    {
        creators.push_back(format.create);
        names.emplace_back(format.label);
    }
    creators.push_back([](const auto &shape, uint32_t seed) { return TestTensorFactory::createFP32Random(shape, -1.f, 1.f, seed); });
    creators.push_back([](const auto &shape, uint32_t seed) { return TestTensorFactory::createFP16Random(shape, -1.f, 1.f, seed); });
    creators.push_back([](const auto &shape, uint32_t seed) { return TestTensorFactory::createBF16Random(shape, -1.f, 1.f, seed); });
    names.insert(names.end(), {"FP32", "FP16", "BF16"});
    for (size_t format = 0; format < creators.size(); ++format)
    {
        SCOPED_TRACE(names[format]);
        // The expert axis is compacted to two physical slots for logical IDs
        // 1 and 7. Build deterministic source bytes in GGUF's contiguous order.
        std::shared_ptr<TensorBase> source(creators[format]({2 * 512, 256}, 77));
        const size_t bytes_per_row = source->size_bytes() / (2 * 512);
        const MoEExpertSourceView selection(std::array<size_t, 3>{256, 512, 2},
            WeightSliceSpec{.source_rows = 256, .source_cols = 512,
                .row_count = 256, .col_count = 512,
                .expert_count = 2, .expert_ids = {1, 7}, .inner_is_presliced = true});
        for (const int participants : {1, 2, 4, 8})
            for (int participant = 0; participant < participants; ++participant)
            {
                const auto owner = MoEExpertProjectionOwnership::gateUpOwnedDownColumns({8, 512, 256}, participant, participants);
                for (const int expert : {1, 7})
                {
                    const auto matrix = selection.matrix(expert, WeightRole::MoEExpertDown, &owner);
                    const auto view = source->create_view({matrix.rows, matrix.columns}, matrix.element_offset);
                    ASSERT_NE(view, nullptr);
                    EXPECT_EQ(view->native_type(), source->native_type());
                    const size_t expected_row = (expert == 7 ? 512 : 0) + participant * (512 / participants);
                    const auto *expected = static_cast<const uint8_t *>(source->raw_data()) + expected_row * bytes_per_row;
                    EXPECT_EQ(view->raw_data(), expected);
                    EXPECT_EQ(view->size_bytes(), matrix.rows * bytes_per_row);
                    EXPECT_EQ(std::memcmp(view->raw_data(), expected, view->size_bytes()), 0);
                    EXPECT_EQ(matrix.source_identity.source_cols, 256u);
                    EXPECT_EQ(matrix.source_identity.source_rows, 512u);
                }
            }
    }
}
} // namespace llaminar2::test
