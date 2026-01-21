/**
 * @file Test__PipelineParallelConfig.cpp
 * @brief Unit tests for PipelineParallelConfig infrastructure
 *
 * Tests the configuration structures that enable pipeline parallelism
 * for distributing transformer layers across MPI ranks.
 */

#include <gtest/gtest.h>
#include "config/PipelineParallelConfig.h"

namespace llaminar2
{
    namespace test
    {

        class Test__PipelineParallelConfig : public ::testing::Test
        {
        protected:
            // Common test parameters
            static constexpr int QWEN_7B_LAYERS = 28;  // Qwen2.5-7B
            static constexpr int QWEN_05B_LAYERS = 24; // Qwen2.5-0.5B
            static constexpr int LLAMA_32_LAYERS = 32; // Llama-3 8B
        };

        // =============================================================================
        // Equal Split Tests
        // =============================================================================

        TEST_F(Test__PipelineParallelConfig, EqualSplit_TwoRanks)
        {
            // 28 layers / 2 ranks = 14 + 14
            auto config = PipelineParallelConfig::equalSplit(2, QWEN_7B_LAYERS);

            EXPECT_EQ(config.numStages(), 2);
            EXPECT_EQ(config.totalLayers(), QWEN_7B_LAYERS);
            EXPECT_TRUE(config.validate());

            const auto &rank0 = config.forRank(0);
            const auto &rank1 = config.forRank(1);

            EXPECT_EQ(rank0.first_layer, 0);
            EXPECT_EQ(rank0.last_layer, 13);
            EXPECT_EQ(rank0.count(), 14);

            EXPECT_EQ(rank1.first_layer, 14);
            EXPECT_EQ(rank1.last_layer, 27);
            EXPECT_EQ(rank1.count(), 14);
        }

        TEST_F(Test__PipelineParallelConfig, EqualSplit_ThreeRanks)
        {
            // 28 layers / 3 ranks = 9 base + 1 remainder
            // Earlier ranks get the extra layers: 10 + 9 + 9
            auto config = PipelineParallelConfig::equalSplit(3, QWEN_7B_LAYERS);

            EXPECT_EQ(config.numStages(), 3);
            EXPECT_EQ(config.totalLayers(), QWEN_7B_LAYERS);
            EXPECT_TRUE(config.validate());

            const auto &rank0 = config.forRank(0);
            const auto &rank1 = config.forRank(1);
            const auto &rank2 = config.forRank(2);

            // Verify total adds up
            EXPECT_EQ(rank0.count() + rank1.count() + rank2.count(), QWEN_7B_LAYERS);

            // Earlier ranks get more: 10 + 9 + 9
            EXPECT_EQ(rank0.count(), 10);
            EXPECT_EQ(rank1.count(), 9);
            EXPECT_EQ(rank2.count(), 9);

            // Verify contiguity
            EXPECT_EQ(rank0.first_layer, 0);
            EXPECT_EQ(rank1.first_layer, rank0.last_layer + 1);
            EXPECT_EQ(rank2.first_layer, rank1.last_layer + 1);
            EXPECT_EQ(rank2.last_layer, QWEN_7B_LAYERS - 1);
        }

        TEST_F(Test__PipelineParallelConfig, EqualSplit_OddLayers)
        {
            // 29 layers / 2 ranks = 15 + 14
            constexpr int ODD_LAYERS = 29;
            auto config = PipelineParallelConfig::equalSplit(2, ODD_LAYERS);

            EXPECT_EQ(config.numStages(), 2);
            EXPECT_EQ(config.totalLayers(), ODD_LAYERS);
            EXPECT_TRUE(config.validate());

            const auto &rank0 = config.forRank(0);
            const auto &rank1 = config.forRank(1);

            EXPECT_EQ(rank0.count(), 15); // Earlier rank gets the extra
            EXPECT_EQ(rank1.count(), 14);

            EXPECT_EQ(rank0.first_layer, 0);
            EXPECT_EQ(rank0.last_layer, 14);
            EXPECT_EQ(rank1.first_layer, 15);
            EXPECT_EQ(rank1.last_layer, 28);
        }

        TEST_F(Test__PipelineParallelConfig, EqualSplit_FourRanks)
        {
            // 32 layers / 4 ranks = 8 + 8 + 8 + 8 (divides evenly)
            auto config = PipelineParallelConfig::equalSplit(4, LLAMA_32_LAYERS);

            EXPECT_EQ(config.numStages(), 4);
            EXPECT_EQ(config.totalLayers(), LLAMA_32_LAYERS);
            EXPECT_TRUE(config.validate());

            for (int rank = 0; rank < 4; ++rank)
            {
                const auto &range = config.forRank(rank);
                EXPECT_EQ(range.count(), 8);
                EXPECT_EQ(range.first_layer, rank * 8);
                EXPECT_EQ(range.last_layer, rank * 8 + 7);
            }
        }

        // =============================================================================
        // Custom Split Tests
        // =============================================================================

        TEST_F(Test__PipelineParallelConfig, CustomSplit_Unequal)
        {
            // rank 0: layers 0-9 (10 layers)
            // rank 1: layers 10-27 (18 layers)
            auto config = PipelineParallelConfig::customSplit({{0, 9}, {10, 27}});

            EXPECT_EQ(config.numStages(), 2);
            EXPECT_EQ(config.totalLayers(), QWEN_7B_LAYERS);
            EXPECT_TRUE(config.validate());

            const auto &rank0 = config.forRank(0);
            const auto &rank1 = config.forRank(1);

            EXPECT_EQ(rank0.first_layer, 0);
            EXPECT_EQ(rank0.last_layer, 9);
            EXPECT_EQ(rank0.count(), 10);

            EXPECT_EQ(rank1.first_layer, 10);
            EXPECT_EQ(rank1.last_layer, 27);
            EXPECT_EQ(rank1.count(), 18);
        }

        TEST_F(Test__PipelineParallelConfig, CustomSplit_ThreeWay)
        {
            // rank 0: layers 0-7 (8 layers)
            // rank 1: layers 8-19 (12 layers)
            // rank 2: layers 20-27 (8 layers)
            auto config = PipelineParallelConfig::customSplit({{0, 7}, {8, 19}, {20, 27}});

            EXPECT_EQ(config.numStages(), 3);
            EXPECT_EQ(config.totalLayers(), QWEN_7B_LAYERS);
            EXPECT_TRUE(config.validate());

            EXPECT_EQ(config.forRank(0).count(), 8);
            EXPECT_EQ(config.forRank(1).count(), 12);
            EXPECT_EQ(config.forRank(2).count(), 8);
        }

        // =============================================================================
        // Single Rank Tests
        // =============================================================================

        TEST_F(Test__PipelineParallelConfig, SingleRank)
        {
            auto config = PipelineParallelConfig::singleRank(QWEN_7B_LAYERS);

            EXPECT_EQ(config.numStages(), 1);
            EXPECT_EQ(config.totalLayers(), QWEN_7B_LAYERS);
            EXPECT_TRUE(config.validate());

            const auto &rank0 = config.forRank(0);
            EXPECT_EQ(rank0.first_layer, 0);
            EXPECT_EQ(rank0.last_layer, 27);
            EXPECT_EQ(rank0.count(), QWEN_7B_LAYERS);
            EXPECT_EQ(rank0.owning_rank, 0);
        }

        // =============================================================================
        // ForRank Tests
        // =============================================================================

        TEST_F(Test__PipelineParallelConfig, ForRank_ValidRank)
        {
            auto config = PipelineParallelConfig::equalSplit(3, QWEN_7B_LAYERS);

            // All ranks should be accessible
            EXPECT_NO_THROW(config.forRank(0));
            EXPECT_NO_THROW(config.forRank(1));
            EXPECT_NO_THROW(config.forRank(2));

            // Check owning_rank matches
            EXPECT_EQ(config.forRank(0).owning_rank, 0);
            EXPECT_EQ(config.forRank(1).owning_rank, 1);
            EXPECT_EQ(config.forRank(2).owning_rank, 2);
        }

        TEST_F(Test__PipelineParallelConfig, ForRank_InvalidRank)
        {
            auto config = PipelineParallelConfig::equalSplit(2, QWEN_7B_LAYERS);

            EXPECT_THROW(config.forRank(-1), std::out_of_range);
            EXPECT_THROW(config.forRank(2), std::out_of_range);
            EXPECT_THROW(config.forRank(100), std::out_of_range);
        }

        // =============================================================================
        // RankForLayer Tests
        // =============================================================================

        TEST_F(Test__PipelineParallelConfig, RankForLayer_AllLayers)
        {
            auto config = PipelineParallelConfig::equalSplit(2, QWEN_7B_LAYERS);

            // Layers 0-13 should be owned by rank 0
            for (int layer = 0; layer < 14; ++layer)
            {
                EXPECT_EQ(config.rankForLayer(layer), 0)
                    << "Layer " << layer << " should be owned by rank 0";
            }

            // Layers 14-27 should be owned by rank 1
            for (int layer = 14; layer < 28; ++layer)
            {
                EXPECT_EQ(config.rankForLayer(layer), 1)
                    << "Layer " << layer << " should be owned by rank 1";
            }
        }

        TEST_F(Test__PipelineParallelConfig, RankForLayer_ThreeRanks)
        {
            auto config = PipelineParallelConfig::equalSplit(3, QWEN_7B_LAYERS);

            // With 28 layers / 3 ranks: 10 + 9 + 9
            // Rank 0: 0-9, Rank 1: 10-18, Rank 2: 19-27
            EXPECT_EQ(config.rankForLayer(0), 0);
            EXPECT_EQ(config.rankForLayer(9), 0);
            EXPECT_EQ(config.rankForLayer(10), 1);
            EXPECT_EQ(config.rankForLayer(18), 1);
            EXPECT_EQ(config.rankForLayer(19), 2);
            EXPECT_EQ(config.rankForLayer(27), 2);
        }

        TEST_F(Test__PipelineParallelConfig, RankForLayer_InvalidLayer)
        {
            auto config = PipelineParallelConfig::equalSplit(2, QWEN_7B_LAYERS);

            EXPECT_THROW(config.rankForLayer(-1), std::out_of_range);
            EXPECT_THROW(config.rankForLayer(28), std::out_of_range);
            EXPECT_THROW(config.rankForLayer(100), std::out_of_range);
        }

        // =============================================================================
        // OwnsLayer Tests
        // =============================================================================

        TEST_F(Test__PipelineParallelConfig, OwnsLayer_Boundary)
        {
            auto config = PipelineParallelConfig::equalSplit(2, QWEN_7B_LAYERS);

            // Layer 13 (last layer of rank 0)
            EXPECT_TRUE(config.ownsLayer(0, 13));
            EXPECT_FALSE(config.ownsLayer(1, 13));

            // Layer 14 (first layer of rank 1)
            EXPECT_FALSE(config.ownsLayer(0, 14));
            EXPECT_TRUE(config.ownsLayer(1, 14));

            // Layer 0 (first layer, rank 0)
            EXPECT_TRUE(config.ownsLayer(0, 0));
            EXPECT_FALSE(config.ownsLayer(1, 0));

            // Layer 27 (last layer, rank 1)
            EXPECT_FALSE(config.ownsLayer(0, 27));
            EXPECT_TRUE(config.ownsLayer(1, 27));
        }

        TEST_F(Test__PipelineParallelConfig, OwnsLayer_InvalidRank)
        {
            auto config = PipelineParallelConfig::equalSplit(2, QWEN_7B_LAYERS);

            // Invalid ranks should return false, not throw
            EXPECT_FALSE(config.ownsLayer(-1, 0));
            EXPECT_FALSE(config.ownsLayer(5, 0));
        }

        // =============================================================================
        // Pipeline Topology Tests
        // =============================================================================

        TEST_F(Test__PipelineParallelConfig, Topology_FirstLastStage)
        {
            auto config = PipelineParallelConfig::equalSplit(3, QWEN_7B_LAYERS);

            // First stage
            EXPECT_TRUE(config.isFirstStage(0));
            EXPECT_FALSE(config.isLastStage(0));
            EXPECT_EQ(config.prevRank(0), -1);
            EXPECT_EQ(config.nextRank(0), 1);

            // Middle stage
            EXPECT_FALSE(config.isFirstStage(1));
            EXPECT_FALSE(config.isLastStage(1));
            EXPECT_EQ(config.prevRank(1), 0);
            EXPECT_EQ(config.nextRank(1), 2);

            // Last stage
            EXPECT_FALSE(config.isFirstStage(2));
            EXPECT_TRUE(config.isLastStage(2));
            EXPECT_EQ(config.prevRank(2), 1);
            EXPECT_EQ(config.nextRank(2), -1);
        }

        TEST_F(Test__PipelineParallelConfig, Topology_TwoStages)
        {
            auto config = PipelineParallelConfig::equalSplit(2, QWEN_7B_LAYERS);

            // Rank 0: first stage
            EXPECT_TRUE(config.isFirstStage(0));
            EXPECT_FALSE(config.isLastStage(0));
            EXPECT_EQ(config.prevRank(0), -1);
            EXPECT_EQ(config.nextRank(0), 1);

            // Rank 1: last stage
            EXPECT_FALSE(config.isFirstStage(1));
            EXPECT_TRUE(config.isLastStage(1));
            EXPECT_EQ(config.prevRank(1), 0);
            EXPECT_EQ(config.nextRank(1), -1);
        }

        TEST_F(Test__PipelineParallelConfig, Topology_SingleStage)
        {
            auto config = PipelineParallelConfig::singleRank(QWEN_7B_LAYERS);

            // Single stage is both first and last
            EXPECT_TRUE(config.isFirstStage(0));
            EXPECT_TRUE(config.isLastStage(0));
            EXPECT_EQ(config.prevRank(0), -1);
            EXPECT_EQ(config.nextRank(0), -1);
        }

        TEST_F(Test__PipelineParallelConfig, Topology_InvalidRank)
        {
            auto config = PipelineParallelConfig::equalSplit(2, QWEN_7B_LAYERS);

            // Invalid ranks should return -1
            EXPECT_EQ(config.prevRank(-1), -1);
            EXPECT_EQ(config.nextRank(-1), -1);
            EXPECT_EQ(config.prevRank(5), -1);
            EXPECT_EQ(config.nextRank(5), -1);
        }

        // =============================================================================
        // Validation Error Tests
        // =============================================================================

        TEST_F(Test__PipelineParallelConfig, Validate_Gaps)
        {
            // Gap between layer 9 and layer 11
            EXPECT_THROW(
                PipelineParallelConfig::customSplit({{0, 9}, {11, 27}}),
                std::invalid_argument);
        }

        TEST_F(Test__PipelineParallelConfig, Validate_Overlaps)
        {
            // Overlap: rank 0 ends at 10, rank 1 starts at 10
            EXPECT_THROW(
                PipelineParallelConfig::customSplit({{0, 10}, {10, 27}}),
                std::invalid_argument);
        }

        TEST_F(Test__PipelineParallelConfig, Validate_Empty)
        {
            // Empty configuration should throw
            EXPECT_THROW(
                PipelineParallelConfig::customSplit({}),
                std::invalid_argument);
        }

        TEST_F(Test__PipelineParallelConfig, Validate_DoesNotStartAtZero)
        {
            // Must start at layer 0
            EXPECT_THROW(
                PipelineParallelConfig::customSplit({{1, 27}}),
                std::invalid_argument);
        }

        TEST_F(Test__PipelineParallelConfig, Validate_NegativeLayer)
        {
            // Negative layer indices
            EXPECT_THROW(
                PipelineParallelConfig::customSplit({{-1, 13}, {14, 27}}),
                std::invalid_argument);
        }

        TEST_F(Test__PipelineParallelConfig, Validate_InvalidRange)
        {
            // last_layer < first_layer
            EXPECT_THROW(
                PipelineParallelConfig::customSplit({{0, 13}, {20, 14}}),
                std::invalid_argument);
        }

        // =============================================================================
        // Factory Method Error Tests
        // =============================================================================

        TEST_F(Test__PipelineParallelConfig, EqualSplit_InvalidArgs)
        {
            // Zero or negative ranks
            EXPECT_THROW(
                PipelineParallelConfig::equalSplit(0, 28),
                std::invalid_argument);
            EXPECT_THROW(
                PipelineParallelConfig::equalSplit(-1, 28),
                std::invalid_argument);

            // Zero or negative layers
            EXPECT_THROW(
                PipelineParallelConfig::equalSplit(2, 0),
                std::invalid_argument);
            EXPECT_THROW(
                PipelineParallelConfig::equalSplit(2, -1),
                std::invalid_argument);

            // More ranks than layers
            EXPECT_THROW(
                PipelineParallelConfig::equalSplit(10, 5),
                std::invalid_argument);
        }

        TEST_F(Test__PipelineParallelConfig, SingleRank_InvalidArgs)
        {
            EXPECT_THROW(
                PipelineParallelConfig::singleRank(0),
                std::invalid_argument);
            EXPECT_THROW(
                PipelineParallelConfig::singleRank(-1),
                std::invalid_argument);
        }

        // =============================================================================
        // LayerRange Tests
        // =============================================================================

        TEST_F(Test__PipelineParallelConfig, LayerRange_Contains)
        {
            LayerRange range{5, 10, 0}; // Layers 5-10 inclusive

            EXPECT_FALSE(range.contains(4));
            EXPECT_TRUE(range.contains(5));
            EXPECT_TRUE(range.contains(7));
            EXPECT_TRUE(range.contains(10));
            EXPECT_FALSE(range.contains(11));
        }

        TEST_F(Test__PipelineParallelConfig, LayerRange_Count)
        {
            LayerRange range1{0, 13, 0};
            EXPECT_EQ(range1.count(), 14);

            LayerRange range2{10, 27, 1};
            EXPECT_EQ(range2.count(), 18);

            LayerRange range3{5, 5, 0}; // Single layer
            EXPECT_EQ(range3.count(), 1);
        }

        TEST_F(Test__PipelineParallelConfig, LayerRange_ToString)
        {
            LayerRange range{0, 13, 0};
            std::string str = range.toString();

            EXPECT_NE(str.find("Rank 0"), std::string::npos);
            EXPECT_NE(str.find("0"), std::string::npos);
            EXPECT_NE(str.find("13"), std::string::npos);
            EXPECT_NE(str.find("14 layers"), std::string::npos);
        }

        // =============================================================================
        // Validation Error Message Tests
        // =============================================================================

        TEST_F(Test__PipelineParallelConfig, ValidationError_ReturnsMessage)
        {
            // Create valid config
            auto valid = PipelineParallelConfig::equalSplit(2, 28);
            EXPECT_TRUE(valid.validationError().empty());

            // Can't directly test invalid configs since constructor throws,
            // but we can verify the error message logic works
        }

        // =============================================================================
        // Edge Case Tests
        // =============================================================================

        TEST_F(Test__PipelineParallelConfig, EdgeCase_OneLayerPerRank)
        {
            // 4 layers, 4 ranks = 1 layer each
            auto config = PipelineParallelConfig::equalSplit(4, 4);

            EXPECT_EQ(config.numStages(), 4);
            EXPECT_EQ(config.totalLayers(), 4);

            for (int rank = 0; rank < 4; ++rank)
            {
                const auto &range = config.forRank(rank);
                EXPECT_EQ(range.first_layer, rank);
                EXPECT_EQ(range.last_layer, rank);
                EXPECT_EQ(range.count(), 1);
            }
        }

        TEST_F(Test__PipelineParallelConfig, EdgeCase_SingleLayer)
        {
            auto config = PipelineParallelConfig::singleRank(1);

            EXPECT_EQ(config.numStages(), 1);
            EXPECT_EQ(config.totalLayers(), 1);

            const auto &range = config.forRank(0);
            EXPECT_EQ(range.first_layer, 0);
            EXPECT_EQ(range.last_layer, 0);
            EXPECT_EQ(range.count(), 1);
        }

    } // namespace test
} // namespace llaminar2
