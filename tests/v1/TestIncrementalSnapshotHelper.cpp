/**
 * @file TestIncrementalSnapshotHelper.cpp
 * @brief Quick test to verify IncrementalSnapshotHelper and .npy writer work correctly
 * @author David Sanftenberg
 */

#include "ParityTestFramework.h"
#include "NpzLoader.h"
#include "Logger.h"
#include "PipelineStages.h"
#include <gtest/gtest.h>
#include <vector>
#include <cstdio>

using namespace llaminar::parity;

TEST(IncrementalSnapshotHelper, BasicFunctionality)
{
    // Create helper
    IncrementalSnapshotHelper helper("/tmp/test_incremental_snapshots");

    // Clear any previous test data
    system("rm -rf /tmp/test_incremental_snapshots");

    // Enable snapshot capture
    LlaminarSnapshotHook::set_enabled(true);

    // Simulate capturing snapshots for 3 tokens
    for (int token_idx = 0; token_idx < 3; ++token_idx)
    {
        helper.beforeToken(token_idx);

        // Simulate capturing some snapshots
        // Token 0: EMBEDDING
        std::vector<float> embedding_data(896, static_cast<float>(token_idx) * 1.0f);
        TensorSnapshot embedding_snap;
        embedding_snap.metadata.stage_name = "EMBEDDING";
        embedding_snap.metadata.stage = llaminar::PipelineStage::EMBEDDING;
        embedding_snap.metadata.layer_index = -1;
        embedding_snap.metadata.seq_len = 1;
        embedding_snap.metadata.feature_dim = 896;
        embedding_snap.metadata.source = "llaminar";
        embedding_snap.data = embedding_data;
        SnapshotRegistry::instance().register_snapshot("llaminar_EMBEDDING", embedding_snap);

        // Layer 0 ATTENTION_OUTPUT
        std::vector<float> attn_data(896, static_cast<float>(token_idx) * 2.0f);
        TensorSnapshot attn_snap;
        attn_snap.metadata.stage_name = "ATTENTION_OUTPUT";
        attn_snap.metadata.stage = llaminar::PipelineStage::ATTENTION_OUTPUT;
        attn_snap.metadata.layer_index = 0;
        attn_snap.metadata.seq_len = 1;
        attn_snap.metadata.feature_dim = 896;
        attn_snap.metadata.source = "llaminar";
        attn_snap.data = attn_data;
        SnapshotRegistry::instance().register_snapshot("llaminar_layer_0_ATTENTION_OUTPUT", attn_snap);

        // Save snapshots for this token
        ASSERT_TRUE(helper.afterToken(token_idx)) << "Failed to save token_" << token_idx;

        // Verify directory was created
        std::string token_dir = helper.getTokenDir(token_idx);
        std::ifstream test_file(token_dir + "/EMBEDDING.npy");
        ASSERT_TRUE(test_file.good()) << "EMBEDDING.npy not created for token_" << token_idx;
        test_file.close();

        std::ifstream test_file2(token_dir + "/ATTENTION_OUTPUT_layer0.npy");
        ASSERT_TRUE(test_file2.good()) << "ATTENTION_OUTPUT_layer0.npy not created for token_" << token_idx;
        test_file2.close();
    }

    // Verify we can read back the .npy files
    for (int token_idx = 0; token_idx < 3; ++token_idx)
    {
        std::string token_dir = helper.getTokenDir(token_idx);

        // Load and verify EMBEDDING
        NpyArray embedding_loaded;
        ASSERT_TRUE(NpzLoader::load_npy(token_dir + "/EMBEDDING.npy", embedding_loaded))
            << "Failed to load EMBEDDING.npy for token_" << token_idx;

        EXPECT_EQ(embedding_loaded.shape.size(), 2);
        EXPECT_EQ(embedding_loaded.shape[0], 1);   // seq_len
        EXPECT_EQ(embedding_loaded.shape[1], 896); // feature_dim
        EXPECT_EQ(embedding_loaded.data.size(), 896);

        // Check values
        float expected_val = static_cast<float>(token_idx) * 1.0f;
        for (size_t i = 0; i < 10; ++i)
        { // Check first 10 values
            EXPECT_FLOAT_EQ(embedding_loaded.data[i], expected_val)
                << "Value mismatch at index " << i << " for token_" << token_idx;
        }

        // Load and verify ATTENTION_OUTPUT
        NpyArray attn_loaded;
        ASSERT_TRUE(NpzLoader::load_npy(token_dir + "/ATTENTION_OUTPUT_layer0.npy", attn_loaded))
            << "Failed to load ATTENTION_OUTPUT_layer0.npy for token_" << token_idx;

        EXPECT_EQ(attn_loaded.shape.size(), 2);
        EXPECT_EQ(attn_loaded.shape[0], 1);
        EXPECT_EQ(attn_loaded.shape[1], 896);

        // Check values
        float expected_attn_val = static_cast<float>(token_idx) * 2.0f;
        for (size_t i = 0; i < 10; ++i)
        {
            EXPECT_FLOAT_EQ(attn_loaded.data[i], expected_attn_val)
                << "Attention value mismatch at index " << i << " for token_" << token_idx;
        }
    }

    // Cleanup
    system("rm -rf /tmp/test_incremental_snapshots");
}

TEST(NpyWriter, BasicWriteRead)
{
    // Test basic .npy write/read round-trip
    std::string test_file = "/tmp/test_write.npy";

    // Create test data
    std::vector<float> original_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<size_t> shape = {2, 3}; // 2x3 matrix

    // Write
    ASSERT_TRUE(NpzLoader::write_npy(test_file, original_data, shape))
        << "Failed to write .npy file";

    // Read back
    NpyArray loaded;
    ASSERT_TRUE(NpzLoader::load_npy(test_file, loaded))
        << "Failed to load .npy file";

    // Verify shape
    ASSERT_EQ(loaded.shape.size(), 2);
    EXPECT_EQ(loaded.shape[0], 2);
    EXPECT_EQ(loaded.shape[1], 3);

    // Verify data
    ASSERT_EQ(loaded.data.size(), 6);
    for (size_t i = 0; i < 6; ++i)
    {
        EXPECT_FLOAT_EQ(loaded.data[i], original_data[i])
            << "Data mismatch at index " << i;
    }

    // Cleanup
    std::remove(test_file.c_str());
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
