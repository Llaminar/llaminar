/**
 * @file test_layer_by_layer_parity.cpp
 * @brief Layer-by-layer parity test between Llaminar and PyTorch reference
 *
 * This test enables deep debugging of divergence by comparing intermediate
 * layer outputs from Llaminar against PyTorch reference implementation.
 *
 * Usage:
 *   1. Run this test to capture Llaminar layer outputs
 *   2. Run python/reference/capture_pytorch_layers.py to capture PyTorch outputs
 *   3. Use the comparison functionality to identify first diverging layer
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <memory>
#include <vector>
#include <fstream>

#include "parity_test_framework.h"
#include "logger.h"

using namespace llaminar;
using namespace llaminar::parity;

class LayerByLayerParityTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        rank_ = rank;

        // Enable snapshot capture
        LlaminarSnapshotHook::set_enabled(true);

        // Clear any existing snapshots
        SnapshotRegistry::instance().clear();
    }

    void TearDown() override
    {
        LlaminarSnapshotHook::set_enabled(false);
    }

    int rank_;
};

/**
 * @brief Capture Llaminar layer outputs for comparison with PyTorch
 *
 * This test demonstrates the framework structure but is DISABLED because
 * Llaminar pipeline instrumentation is not yet implemented.
 *
 * To enable, add snapshot capture calls to the transformer layer implementations.
 */
TEST_F(LayerByLayerParityTest, DISABLED_CaptureLlaminarLayers)
{
    // Only rank 0 runs the test
    if (rank_ != 0)
    {
        GTEST_SKIP() << "Skipping on non-zero rank";
    }

    // This test is disabled until pipeline instrumentation is added
    // See docs/LAYER_BY_LAYER_PARITY_GUIDE.md for implementation details

    std::cout << "\n";
    std::cout << "This test is currently DISABLED.\n";
    std::cout << "To enable:\n";
    std::cout << "1. Add LlaminarSnapshotHook::capture() calls to transformer layers\n";
    std::cout << "2. Set LLAMINAR_PARITY_CAPTURE=1 environment variable\n";
    std::cout << "3. Re-enable this test by removing DISABLED_ prefix\n";
    std::cout << "\n";
    std::cout << "For now, use the Python-side comparison which is fully functional:\n";
    std::cout << "  python3 python/reference/capture_pytorch_layers.py \\\n";
    std::cout << "    -m models/qwen2.5-0.5b-instruct-q4_0.gguf \\\n";
    std::cout << "    --tokens 1639 266 285 17 10 17 30\n";
    std::cout << "\n";
}

/**
 * @brief Compare Llaminar and PyTorch layer outputs (when both available)
 *
 * This test loads PyTorch snapshots and compares them against Llaminar
 * snapshots to identify divergence points.
 *
 * NOTE: This test is currently disabled as it requires integration with
 * the Python-side snapshot capture. Use the Python script directly for
 * comparison until NPZ loading is fully implemented.
 */
TEST_F(LayerByLayerParityTest, DISABLED_CompareLlaminarVsPytorch)
{
    if (rank_ != 0)
    {
        GTEST_SKIP() << "Skipping on non-zero rank";
    }

    // This would load PyTorch snapshots from NPZ
    const char *pytorch_npz = "pytorch_layer_captures.npz";

    std::ifstream npz_file(pytorch_npz);
    if (!npz_file.good())
    {
        GTEST_SKIP() << "PyTorch captures not found: " << pytorch_npz;
    }

    // Load PyTorch snapshots
    size_t loaded = PytorchSnapshotLoader::load_from_npz(pytorch_npz, "pytorch");
    ASSERT_GT(loaded, 0) << "Failed to load PyTorch snapshots";

    // Compare all layers
    ComparisonTolerance tolerance;
    tolerance.max_abs = 0.1f; // Allow small numerical differences
    tolerance.rel_l2 = 0.01f; // 1% relative error

    auto results = LayerByLayerComparator::compare_all(
        "llaminar", "pytorch", tolerance);

    // Print report
    LayerByLayerComparator::print_report(results, /*verbose=*/true);

    // Find first divergence
    auto first_divergence = LayerByLayerComparator::find_first_divergence(
        "llaminar", "pytorch", tolerance);

    if (!first_divergence.empty())
    {
        std::cout << "\n🔍 FIRST DIVERGING LAYER: " << first_divergence << "\n";

        // This is where we would drill down into the specific layer
        // to understand what went wrong
        FAIL() << "Divergence detected at layer: " << first_divergence;
    }
    else
    {
        std::cout << "\n✓ All layers match within tolerance!\n";
    }
}

/**
 * @brief Test PyTorch snapshot key parsing
 */
TEST_F(LayerByLayerParityTest, ParsePytorchKeys)
{
    int layer_index;
    std::string stage_name;

    // Test layer keys
    EXPECT_TRUE(PytorchSnapshotLoader::parse_key("layer_0_attn_out", layer_index, stage_name));
    EXPECT_EQ(layer_index, 0);
    EXPECT_EQ(stage_name, "attn_out");

    EXPECT_TRUE(PytorchSnapshotLoader::parse_key("layer_15_ffn_out", layer_index, stage_name));
    EXPECT_EQ(layer_index, 15);
    EXPECT_EQ(stage_name, "ffn_out");

    // Test special keys
    EXPECT_TRUE(PytorchSnapshotLoader::parse_key("embeddings", layer_index, stage_name));
    EXPECT_EQ(layer_index, -1);
    EXPECT_EQ(stage_name, "embeddings");

    EXPECT_TRUE(PytorchSnapshotLoader::parse_key("final_norm_out", layer_index, stage_name));
    EXPECT_EQ(layer_index, -1);
    EXPECT_EQ(stage_name, "final_norm_out");

    // Test invalid keys
    EXPECT_FALSE(PytorchSnapshotLoader::parse_key("invalid_key", layer_index, stage_name));
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);

    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
