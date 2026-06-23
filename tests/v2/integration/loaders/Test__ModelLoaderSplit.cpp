/**
 * @file Test__ModelLoaderSplit.cpp
 * @brief Integration tests for multi-part (split) GGUF loading
 *
 * Tests ModelLoader functionality with split GGUF files:
 *  - Split file detection and loading
 *  - Tensor discovery across multiple splits
 *  - Tensor loading from non-primary split files
 *  - Dimension correctness for split tensors (2D swap)
 *  - tensorCount() returns correct total across all splits
 *  - Data correctness: split tensor data matches single-file data
 *  - mmap with split files
 *  - Error handling (missing split file)
 *
 * Requires: models/qwen2.5-0.5b-instruct-q8_0.gguf (single file)
 *           models/qwen2.5-0.5b-instruct-q8_0-split-00001-of-00002.gguf (split part 1)
 *           models/qwen2.5-0.5b-instruct-q8_0-split-00002-of-00002.gguf (split part 2)
 */

#include <gtest/gtest.h>
#include "loaders/ModelLoader.h"
#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"
#include "utils/MPIContext.h"
#include "backends/DeviceId.h"
#include <fstream>
#include <set>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

using namespace llaminar2;

// =============================================================================
// TEST FIXTURE
// =============================================================================

class Test__ModelLoaderSplit : public ::testing::Test
{
protected:
    static constexpr const char *SINGLE_MODEL_PATH =
        "models/qwen2.5-0.5b-instruct-q8_0.gguf";
    static constexpr const char *SPLIT_MODEL_PATH =
        "models/qwen2.5-0.5b-instruct-q8_0-split-00001-of-00002.gguf";
    static constexpr const char *SPLIT_PART2_PATH =
        "models/qwen2.5-0.5b-instruct-q8_0-split-00002-of-00002.gguf";

    void SetUp() override
    {
        // Check if split model files are available
        std::ifstream test1(SPLIT_MODEL_PATH);
        std::ifstream test2(SPLIT_PART2_PATH);
        if (!test1.good() || !test2.good())
        {
            GTEST_SKIP() << "Split model files not available";
        }
    }

    bool singleModelAvailable() const
    {
        std::ifstream test(SINGLE_MODEL_PATH);
        return test.good();
    }
};

// =============================================================================
// SPLIT FILE DETECTION AND METADATA
// =============================================================================

/**
 * @brief Loading a split GGUF file should detect split metadata
 */
TEST_F(Test__ModelLoaderSplit, DetectsSplitMetadata)
{
    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(SPLIT_MODEL_PATH));

    const auto &model = loader.getModel();

    // Should detect that this is a 2-part split
    EXPECT_EQ(model.split_count, 2);
    EXPECT_EQ(model.split_no, 0); // First part is index 0
}

/**
 * @brief Split model should have correct architecture metadata
 */
TEST_F(Test__ModelLoaderSplit, ArchitectureMetadataCorrect)
{
    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(SPLIT_MODEL_PATH));

    const auto &model = loader.getModel();
    EXPECT_EQ(model.architecture, "qwen2");
    EXPECT_EQ(model.block_count, 24);
    EXPECT_EQ(model.embedding_length, 896);
    EXPECT_EQ(model.head_count, 14);
    EXPECT_EQ(model.head_count_kv, 2);
}

// =============================================================================
// TENSOR DISCOVERY ACROSS SPLITS
// =============================================================================

/**
 * @brief tensorCount() should return total tensor count across ALL splits
 */
TEST_F(Test__ModelLoaderSplit, TensorCountReturnsTotal)
{
    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(SPLIT_MODEL_PATH));

    // The Qwen2.5 0.5B Q8_0 model has 291 tensors total:
    // 12 per layer × 24 layers + 3 global (token_embd, output_norm, output) = 291
    EXPECT_EQ(loader.tensorCount(), 291);
}

/**
 * @brief tensorNames() should return all tensor names from all splits
 */
TEST_F(Test__ModelLoaderSplit, TensorNamesSpanAllSplits)
{
    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(SPLIT_MODEL_PATH));

    auto names = loader.tensorNames();
    EXPECT_EQ(names.size(), 291);

    // Check that we have tensors from layer 0 (should be in part 1)
    EXPECT_NE(std::find(names.begin(), names.end(), "blk.0.attn_q.weight"), names.end());
    EXPECT_NE(std::find(names.begin(), names.end(), "blk.0.ffn_norm.weight"), names.end());

    // Check that we have tensors from layer 23 (should be in part 2)
    EXPECT_NE(std::find(names.begin(), names.end(), "blk.23.attn_q.weight"), names.end());
    EXPECT_NE(std::find(names.begin(), names.end(), "blk.23.ffn_norm.weight"), names.end());

    // Check global tensors
    EXPECT_NE(std::find(names.begin(), names.end(), "token_embd.weight"), names.end());
    EXPECT_NE(std::find(names.begin(), names.end(), "output_norm.weight"), names.end());
    EXPECT_NE(std::find(names.begin(), names.end(), "output.weight"), names.end());
}

/**
 * @brief All tensor names should be unique (no duplicates across splits)
 */
TEST_F(Test__ModelLoaderSplit, NoDuplicateTensorNames)
{
    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(SPLIT_MODEL_PATH));

    auto names = loader.tensorNames();
    std::set<std::string> unique_names(names.begin(), names.end());
    EXPECT_EQ(unique_names.size(), names.size()) << "Found duplicate tensor names across splits";
}

/**
 * @brief Split index should be correctly assigned for tensors in each split
 */
TEST_F(Test__ModelLoaderSplit, SplitIndexAssignment)
{
    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(SPLIT_MODEL_PATH));

    const auto &model = loader.getModel();

    int part0_count = 0;
    int part1_count = 0;

    for (const auto &tensor : model.tensors)
    {
        if (tensor.split_idx == 0)
            part0_count++;
        else if (tensor.split_idx == 1)
            part1_count++;
        else
            FAIL() << "Unexpected split_idx: " << tensor.split_idx
                   << " for tensor: " << tensor.name;
    }

    // Part 1 (split_idx=0) should have ~146 tensors
    EXPECT_GT(part0_count, 100) << "Part 1 has too few tensors";
    // Part 2 (split_idx=1) should have ~145 tensors
    EXPECT_GT(part1_count, 100) << "Part 2 has too few tensors";
    // Total should be 291
    EXPECT_EQ(part0_count + part1_count, 291);
}

// =============================================================================
// TENSOR LOADING FROM SPLIT FILES
// =============================================================================

/**
 * @brief Should be able to load a tensor from the primary split (part 1)
 */
TEST_F(Test__ModelLoaderSplit, LoadTensorFromPart1)
{
    MPIContext mpi_ctx(0, 1);
    TensorFactory factory(mpi_ctx);
    ModelLoader loader(&factory);
    ASSERT_TRUE(loader.loadModel(SPLIT_MODEL_PATH));

    // token_embd.weight should be in part 1
    auto tensor = loader.loadTensor("token_embd.weight");
    ASSERT_NE(tensor, nullptr);
    EXPECT_GT(tensor->size_bytes(), 0);
}

/**
 * @brief Should be able to load a tensor from the secondary split (part 2)
 */
TEST_F(Test__ModelLoaderSplit, LoadTensorFromPart2)
{
    MPIContext mpi_ctx(0, 1);
    TensorFactory factory(mpi_ctx);
    ModelLoader loader(&factory);
    ASSERT_TRUE(loader.loadModel(SPLIT_MODEL_PATH));

    // blk.23 tensors should be in part 2
    auto tensor = loader.loadTensor("blk.23.attn_q.weight");
    ASSERT_NE(tensor, nullptr);
    EXPECT_GT(tensor->size_bytes(), 0);
}

/**
 * @brief Load all tensors from all splits (verify no failures)
 */
TEST_F(Test__ModelLoaderSplit, LoadAllTensorsAcrossSplits)
{
    MPIContext mpi_ctx(0, 1);
    TensorFactory factory(mpi_ctx);
    ModelLoader loader(&factory);
    ASSERT_TRUE(loader.loadModel(SPLIT_MODEL_PATH));

    auto names = loader.tensorNames();
    int loaded = 0;
    int failed = 0;

    for (const auto &name : names)
    {
        auto tensor = loader.loadTensor(name);
        if (tensor != nullptr)
            loaded++;
        else
            failed++;
    }

    EXPECT_EQ(loaded, 291) << "Failed to load " << failed << " tensors";
    EXPECT_EQ(failed, 0);
}

// =============================================================================
// DIMENSION CORRECTNESS
// =============================================================================

/**
 * @brief 2D tensors from split files must have correct (swapped) dimensions
 *
 * GGUF stores dimensions in reversed order for 2D tensors. The dimension swap
 * must be applied consistently in both parseTensorInfo() (main file) AND
 * loadSplitFiles() (additional splits). This test verifies that fix.
 */
TEST_F(Test__ModelLoaderSplit, DimensionSwapAppliedToSplitTensors)
{
    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(SPLIT_MODEL_PATH));

    const auto &model = loader.getModel();

    // For Qwen2.5 0.5B:
    //   attn_q.weight should be [896, 896] (embedding_length × embedding_length)
    //   ffn_gate.weight should be [4864, 896] (intermediate_size × embedding_length)
    //   ffn_down.weight should be [896, 4864] (embedding_length × intermediate_size)

    // Check a tensor known to be in part 2 (split_idx=1)
    for (const auto &tensor : model.tensors)
    {
        if (tensor.name == "blk.23.attn_q.weight")
        {
            ASSERT_EQ(tensor.split_idx, 1) << "Expected blk.23 to be in split 1";
            ASSERT_EQ(tensor.dimensions.size(), 2);
            // After swap: [rows, cols] = [896, 896]
            EXPECT_EQ(tensor.dimensions[0], 896) << "Dimension[0] wrong for blk.23.attn_q.weight";
            EXPECT_EQ(tensor.dimensions[1], 896) << "Dimension[1] wrong for blk.23.attn_q.weight";
            break;
        }
    }

    // Also check an FFN weight from part 2
    for (const auto &tensor : model.tensors)
    {
        if (tensor.name == "blk.23.ffn_gate.weight")
        {
            ASSERT_EQ(tensor.split_idx, 1);
            ASSERT_EQ(tensor.dimensions.size(), 2);
            // ffn_gate: [intermediate_size, embedding_length] = [4864, 896]
            EXPECT_EQ(tensor.dimensions[0], 4864) << "Dimension[0] wrong for blk.23.ffn_gate.weight";
            EXPECT_EQ(tensor.dimensions[1], 896) << "Dimension[1] wrong for blk.23.ffn_gate.weight";
            break;
        }
    }
}

/**
 * @brief Dimensions of split tensors should match dimensions of corresponding
 *        tensors from part 1 (same layer structure, same weight shapes)
 */
TEST_F(Test__ModelLoaderSplit, SplitDimensionsMatchPart1Pattern)
{
    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(SPLIT_MODEL_PATH));

    const auto &model = loader.getModel();

    // Find dimensions for blk.0 (part 1) and blk.23 (part 2)
    auto find_tensor = [&](const std::string &name) -> const GGUFTensorInfo *
    {
        for (const auto &t : model.tensors)
        {
            if (t.name == name)
                return &t;
        }
        return nullptr;
    };

    // Compare matching weight types across parts
    std::vector<std::string> suffixes = {
        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight",
        ".attn_output.weight", ".ffn_gate.weight", ".ffn_up.weight",
        ".ffn_down.weight", ".attn_norm.weight", ".ffn_norm.weight"};

    for (const auto &suffix : suffixes)
    {
        auto *t0 = find_tensor("blk.0" + suffix);
        auto *t23 = find_tensor("blk.23" + suffix);

        ASSERT_NE(t0, nullptr) << "Missing blk.0" << suffix;
        ASSERT_NE(t23, nullptr) << "Missing blk.23" << suffix;

        EXPECT_EQ(t0->dimensions, t23->dimensions)
            << "Dimension mismatch for " << suffix
            << ": blk.0 has [" << t0->dimensions[0]
            << (t0->dimensions.size() > 1 ? "," + std::to_string(t0->dimensions[1]) : "")
            << "] but blk.23 has [" << t23->dimensions[0]
            << (t23->dimensions.size() > 1 ? "," + std::to_string(t23->dimensions[1]) : "")
            << "]";
    }
}

// =============================================================================
// DATA CORRECTNESS: SPLIT vs SINGLE FILE
// =============================================================================

/**
 * @brief Tensor data loaded from split files should exactly match
 *        data loaded from the single (non-split) file
 */
TEST_F(Test__ModelLoaderSplit, SplitDataMatchesSingleFile)
{
    if (!singleModelAvailable())
        GTEST_SKIP() << "Single-file model not available for comparison";

    MPIContext mpi_ctx(0, 1);
    TensorFactory factory_single(mpi_ctx);
    TensorFactory factory_split(mpi_ctx);

    ModelLoader single_loader(&factory_single);
    ModelLoader split_loader(&factory_split);

    ASSERT_TRUE(single_loader.loadModel(SINGLE_MODEL_PATH));
    ASSERT_TRUE(split_loader.loadModel(SPLIT_MODEL_PATH));

    // Compare specific tensors from different parts
    std::vector<std::string> test_tensors = {
        "token_embd.weight",      // Global tensor (part 1)
        "blk.0.attn_q.weight",    // Layer 0 (part 1)
        "blk.0.ffn_norm.weight",  // Layer 0 norm (part 1)
        "blk.23.attn_q.weight",   // Layer 23 (part 2)
        "blk.23.ffn_norm.weight", // Layer 23 norm (part 2)
        "output_norm.weight",     // Global output norm
        "output.weight",          // LM head
    };

    for (const auto &name : test_tensors)
    {
        auto single_tensor = single_loader.loadTensor(name);
        auto split_tensor = split_loader.loadTensor(name);

        ASSERT_NE(single_tensor, nullptr) << "Single-file failed to load: " << name;
        ASSERT_NE(split_tensor, nullptr) << "Split-file failed to load: " << name;

        // Sizes must match
        ASSERT_EQ(single_tensor->size_bytes(), split_tensor->size_bytes())
            << "Size mismatch for " << name
            << ": single=" << single_tensor->size_bytes()
            << " split=" << split_tensor->size_bytes();

        // Raw data must be identical
        EXPECT_EQ(std::memcmp(single_tensor->raw_data(), split_tensor->raw_data(),
                              single_tensor->size_bytes()),
                  0)
            << "Data mismatch for " << name;
    }
}

/**
 * @brief All 291 tensors should have identical data between split and single file
 */
TEST_F(Test__ModelLoaderSplit, AllTensorsMatchSingleFile)
{
    if (!singleModelAvailable())
        GTEST_SKIP() << "Single-file model not available for comparison";

    MPIContext mpi_ctx(0, 1);
    TensorFactory factory_single(mpi_ctx);
    TensorFactory factory_split(mpi_ctx);

    ModelLoader single_loader(&factory_single);
    ModelLoader split_loader(&factory_split);

    ASSERT_TRUE(single_loader.loadModel(SINGLE_MODEL_PATH));
    ASSERT_TRUE(split_loader.loadModel(SPLIT_MODEL_PATH));

    auto names = split_loader.tensorNames();
    ASSERT_EQ(names.size(), 291);

    int mismatches = 0;
    for (const auto &name : names)
    {
        auto single_tensor = single_loader.loadTensor(name);
        auto split_tensor = split_loader.loadTensor(name);

        ASSERT_NE(single_tensor, nullptr) << "Single-file missing: " << name;
        ASSERT_NE(split_tensor, nullptr) << "Split-file missing: " << name;

        if (single_tensor->size_bytes() != split_tensor->size_bytes() ||
            std::memcmp(single_tensor->raw_data(), split_tensor->raw_data(),
                        single_tensor->size_bytes()) != 0)
        {
            mismatches++;
            ADD_FAILURE() << "Data mismatch for " << name;
        }
    }

    EXPECT_EQ(mismatches, 0) << mismatches << " of 291 tensors had data mismatches";
}

// =============================================================================
// MMAP WITH SPLIT FILES
// =============================================================================

/**
 * @brief Split file loading should work with mmap enabled (default)
 */
TEST_F(Test__ModelLoaderSplit, MmapSplitLoading)
{
    MPIContext mpi_ctx(0, 1);
    TensorFactory factory(mpi_ctx);
    ModelLoader loader(&factory);

    // Default is mmap enabled
    ASSERT_TRUE(loader.loadModel(SPLIT_MODEL_PATH));

    // Load tensors from both parts to verify mmap works for splits
    auto t_part1 = loader.loadTensor("blk.0.attn_q.weight");
    auto t_part2 = loader.loadTensor("blk.23.attn_q.weight");

    ASSERT_NE(t_part1, nullptr);
    ASSERT_NE(t_part2, nullptr);
    EXPECT_GT(t_part1->size_bytes(), 0);
    EXPECT_GT(t_part2->size_bytes(), 0);
}

/**
 * @brief Split file loading should work with mmap disabled
 */
TEST_F(Test__ModelLoaderSplit, NoMmapSplitLoading)
{
    MPIContext mpi_ctx(0, 1);
    TensorFactory factory(mpi_ctx);
    ModelLoader loader(&factory);
    loader.setUseMmap(false);

    ASSERT_TRUE(loader.loadModel(SPLIT_MODEL_PATH));

    // Load tensors from both parts
    auto t_part1 = loader.loadTensor("blk.0.attn_q.weight");
    auto t_part2 = loader.loadTensor("blk.23.attn_q.weight");

    ASSERT_NE(t_part1, nullptr);
    ASSERT_NE(t_part2, nullptr);
}

/**
 * @brief With mmap, tensor data from splits should still match single file
 */
TEST_F(Test__ModelLoaderSplit, MmapDataMatchesSingleFile)
{
    if (!singleModelAvailable())
        GTEST_SKIP() << "Single-file model not available for comparison";

    MPIContext mpi_ctx(0, 1);
    TensorFactory factory_single(mpi_ctx);
    TensorFactory factory_split(mpi_ctx);

    // Both with mmap
    ModelLoader single_loader(&factory_single);
    ModelLoader split_loader(&factory_split);

    ASSERT_TRUE(single_loader.loadModel(SINGLE_MODEL_PATH));
    ASSERT_TRUE(split_loader.loadModel(SPLIT_MODEL_PATH));

    // Compare a part-2 tensor via mmap
    auto single_tensor = single_loader.loadTensor("blk.23.ffn_down.weight");
    auto split_tensor = split_loader.loadTensor("blk.23.ffn_down.weight");

    ASSERT_NE(single_tensor, nullptr);
    ASSERT_NE(split_tensor, nullptr);
    ASSERT_EQ(single_tensor->size_bytes(), split_tensor->size_bytes());

    EXPECT_EQ(std::memcmp(single_tensor->raw_data(), split_tensor->raw_data(),
                          single_tensor->size_bytes()),
              0)
        << "mmap data mismatch for blk.23.ffn_down.weight";
}

// =============================================================================
// ERROR HANDLING
// =============================================================================

/**
 * @brief Loading a split file when the other part is missing should fail gracefully
 */
TEST_F(Test__ModelLoaderSplit, MissingSplitPartFails)
{
    // Create a temporary copy of part 1, but ensure part 2 doesn't exist
    // at the generated path. We do this by copying part 1 to a temp location.
    std::string temp_dir = "/tmp/llaminar_test_split/";
    int ret = system(("rm -rf " + temp_dir + " && mkdir -p " + temp_dir).c_str());
    ASSERT_EQ(ret, 0);

    // Copy part 1 only
    std::string temp_path = temp_dir + "model-00001-of-00002.gguf";
    ret = system(("cp " + std::string(SPLIT_MODEL_PATH) + " " + temp_path).c_str());
    ASSERT_EQ(ret, 0);

    // Try to load — should fail because part 2 is missing
    ModelLoader loader;
    EXPECT_THROW(loader.loadModel(temp_path), std::runtime_error)
        << "Loading should fail when a split part is missing";

    // Cleanup
    ret = system(("rm -rf " + temp_dir).c_str());
    EXPECT_EQ(ret, 0);
}

/**
 * @brief split_paths and split_data_offsets should be correctly populated
 */
TEST_F(Test__ModelLoaderSplit, SplitPathsAndOffsetsPopulated)
{
    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(SPLIT_MODEL_PATH));

    const auto &model = loader.getModel();

    // Should have 2 entries
    ASSERT_EQ(model.split_paths.size(), 2);
    ASSERT_EQ(model.split_data_offsets.size(), 2);

    // Paths should exist
    std::ifstream f1(model.split_paths[0]);
    EXPECT_TRUE(f1.good()) << "Part 1 path invalid: " << model.split_paths[0];

    std::ifstream f2(model.split_paths[1]);
    EXPECT_TRUE(f2.good()) << "Part 2 path invalid: " << model.split_paths[1];

    // Data offsets should be non-zero (aligned to at least 32 bytes)
    EXPECT_GT(model.split_data_offsets[0], 0);
    EXPECT_GT(model.split_data_offsets[1], 0);

    // Data offsets should be 32-byte aligned
    EXPECT_EQ(model.split_data_offsets[0] % 32, 0);
    EXPECT_EQ(model.split_data_offsets[1] % 32, 0);
}

// =============================================================================
// CONSISTENCY CHECKS
// =============================================================================

/**
 * @brief Every layer (0-23) should have exactly 12 tensors
 */
TEST_F(Test__ModelLoaderSplit, EveryLayerHas12Tensors)
{
    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(SPLIT_MODEL_PATH));

    const auto &model = loader.getModel();

    for (int layer = 0; layer < 24; layer++)
    {
        std::string prefix = "blk." + std::to_string(layer) + ".";
        int count = 0;
        for (const auto &t : model.tensors)
        {
            if (t.name.rfind(prefix, 0) == 0)
                count++;
        }
        EXPECT_EQ(count, 12) << "Layer " << layer << " has " << count << " tensors (expected 12)";
    }
}

/**
 * @brief Tensor types should be consistent (Q8_0 for weights, F32 for norms/biases)
 */
TEST_F(Test__ModelLoaderSplit, TensorTypesConsistent)
{
    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(SPLIT_MODEL_PATH));

    const auto &model = loader.getModel();

    for (const auto &t : model.tensors)
    {
        // All weight tensors (2D) should be Q8_0 in this model
        if (t.name.find(".weight") != std::string::npos && t.dimensions.size() == 2)
        {
            EXPECT_EQ(static_cast<int>(t.type), 8) // Q8_0 = 8
                << "Expected Q8_0 for " << t.name
                << " but got type " << static_cast<int>(t.type);
        }

        // Bias tensors (1D) should be F32
        if (t.name.find(".bias") != std::string::npos)
        {
            EXPECT_EQ(static_cast<int>(t.type), 0) // F32 = 0
                << "Expected F32 for " << t.name
                << " but got type " << static_cast<int>(t.type);
        }

        // Norm weight tensors (1D) should be F32
        if (t.name.find("_norm.weight") != std::string::npos)
        {
            EXPECT_EQ(static_cast<int>(t.type), 0) // F32 = 0
                << "Expected F32 for " << t.name
                << " but got type " << static_cast<int>(t.type);
        }
    }
}
