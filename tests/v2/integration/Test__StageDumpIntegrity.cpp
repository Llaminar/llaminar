/**
 * @file Test__StageDumpIntegrity.cpp
 * @brief Integration tests for stage dump infrastructure integrity
 *
 * Verifies that:
 * 1. Layer filtering works correctly (only requested layers are dumped)
 * 2. Dump file sizes match metadata byte_size
 * 3. Block counts in metadata are correct for quantized formats
 * 4. Metadata is trustworthy and intuitive
 * 5. Q16_1 variable block sizes (32, 64, 128) are correctly handled
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <cstdlib>
#include <memory>

#include "execution/InferenceRunnerFactory.h"
#include "execution/IInferenceRunner.h"
#include "execution/StageDumper.h"
#include "execution/compute_stages/IComputeStage.h"
#include "loaders/ModelContext.h"
#include "backends/ComputeBackend.h"
#include "backends/DeviceId.h"
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "utils/MPIContext.h"
#include "utils/DebugEnv.h"
#include "utils/Logger.h"

namespace fs = std::filesystem;

namespace llaminar2::test
{

    /**
     * @brief Parse a metadata file into key-value pairs
     */
    std::map<std::string, std::string> parseMetadataFile(const fs::path &path)
    {
        std::map<std::string, std::string> result;
        std::ifstream file(path);
        if (!file.is_open())
            return result;

        std::string line;
        while (std::getline(file, line))
        {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#')
                continue;

            size_t eq_pos = line.find('=');
            if (eq_pos != std::string::npos)
            {
                std::string key = line.substr(0, eq_pos);
                std::string value = line.substr(eq_pos + 1);
                result[key] = value;
            }
        }
        return result;
    }

    /**
     * @brief Extract layer index from dump directory name
     *
     * Directory names look like: stage_0000_FUSED_ATTENTION_WO_layer5_fused_attn_wo_rank0
     * Returns -1 if no layer found.
     */
    int extractLayerFromDirName(const std::string &dir_name)
    {
        // Look for "layer" followed by digits
        size_t layer_pos = dir_name.find("layer");
        if (layer_pos == std::string::npos)
            return -1;

        size_t start = layer_pos + 5; // Skip "layer"
        size_t end = start;
        while (end < dir_name.length() && std::isdigit(dir_name[end]))
        {
            ++end;
        }

        if (end > start)
        {
            return std::stoi(dir_name.substr(start, end - start));
        }
        return -1;
    }

    /**
     * @brief Get file size in bytes
     */
    size_t getFileSize(const fs::path &path)
    {
        if (!fs::exists(path))
            return 0;
        return fs::file_size(path);
    }

    class Test__StageDumpIntegrity : public ::testing::Test
    {
    protected:
        static constexpr const char *MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
        static constexpr const char *DUMP_DIR = "/tmp/llaminar_stage_dump_test";

        void SetUp() override
        {
            // Initialize DeviceManager (required for device backends)
            DeviceManager::instance().initialize(-1); // -1 = no NUMA filtering

            // Clean up any existing dumps
            if (fs::exists(DUMP_DIR))
            {
                fs::remove_all(DUMP_DIR);
            }
            fs::create_directories(DUMP_DIR);
        }

        void TearDown() override
        {
            // Optionally clean up after test
            // fs::remove_all(DUMP_DIR);
        }

        /**
         * @brief Run a minimal inference with stage dumping enabled
         */
        void runInferenceWithDumps(const std::set<int> &layers_to_dump,
                                   const std::string &stage_type = "GEMM")
        {
            // Build layer list string
            std::string layers_str;
            for (int layer : layers_to_dump)
            {
                if (!layers_str.empty())
                    layers_str += ",";
                layers_str += std::to_string(layer);
            }

            // Set environment variables for dumping
            setenv("LLAMINAR_STAGE_DUMP_ENABLED", "1", 1);
            setenv("LLAMINAR_STAGE_DUMP_DIR", DUMP_DIR, 1);
            setenv("LLAMINAR_STAGE_DUMP_TYPES", stage_type.c_str(), 1);
            setenv("LLAMINAR_STAGE_DUMP_LAYERS", layers_str.c_str(), 1);

            // Reload debug env to pick up new settings
            mutableDebugEnv().stage_dump.reload();

            // Create MPI context (single rank)
            auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);

            // Load model
            auto model_ctx = ModelContext::create(MODEL_PATH, mpi_ctx);
            ASSERT_NE(model_ctx, nullptr) << "Failed to load model";

            // Create inference runner
            auto runner = createInferenceRunner(model_ctx, mpi_ctx,
                                                DeviceId::cpu());
            ASSERT_NE(runner, nullptr) << "Failed to create inference runner";

            // Run minimal prefill (just a few tokens)
            std::vector<int32_t> tokens = {151644, 872, 198}; // <|im_start|>user\n
            runner->forward(tokens.data(), tokens.size());

            // Disable dumping after test
            setenv("LLAMINAR_STAGE_DUMP_ENABLED", "0", 1);
            mutableDebugEnv().stage_dump.reload();
        }

        /**
         * @brief Get all dump directories created
         */
        std::vector<fs::path> getDumpDirectories()
        {
            std::vector<fs::path> result;
            if (!fs::exists(DUMP_DIR))
                return result;

            for (const auto &entry : fs::directory_iterator(DUMP_DIR))
            {
                if (entry.is_directory() && entry.path().filename().string().find("stage_") == 0)
                {
                    result.push_back(entry.path());
                }
            }

            // Sort by name for deterministic order
            std::sort(result.begin(), result.end());
            return result;
        }

        /**
         * @brief Verify a single tensor dump's metadata matches its file
         */
        struct TensorVerifyResult
        {
            bool exists = false;
            bool size_matches = false;
            bool block_count_valid = false;
            size_t metadata_byte_size = 0;
            size_t actual_file_size = 0;
            size_t metadata_block_count = 0;
            size_t computed_block_count = 0;
            std::string dtype;
        };

        TensorVerifyResult verifyTensorDump(const fs::path &dump_dir,
                                            const std::string &subdir,
                                            const std::string &tensor_name)
        {
            TensorVerifyResult result;

            // Find metadata file
            fs::path meta_path = dump_dir / subdir / (tensor_name + "_meta.txt");
            if (!fs::exists(meta_path))
            {
                return result;
            }
            result.exists = true;

            // Parse metadata
            auto meta = parseMetadataFile(meta_path);
            result.dtype = meta["dtype"];

            // Handle missing byte_size (metadata-only entries like weights)
            if (meta.count("byte_size") && !meta["byte_size"].empty())
            {
                result.metadata_byte_size = std::stoull(meta["byte_size"]);
            }
            else
            {
                result.metadata_byte_size = 0;
            }

            if (meta.count("block_count") && !meta["block_count"].empty())
            {
                result.metadata_block_count = std::stoull(meta["block_count"]);
            }

            // Find the binary file (could be .bin or _<dtype>.bin)
            fs::path bin_path;
            std::string dtype_lower = result.dtype;
            std::transform(dtype_lower.begin(), dtype_lower.end(), dtype_lower.begin(), ::tolower);

            // Try dtype-specific extension first
            bin_path = dump_dir / subdir / (tensor_name + "_" + dtype_lower + ".bin");
            if (!fs::exists(bin_path))
            {
                // Try plain .bin
                bin_path = dump_dir / subdir / (tensor_name + ".bin");
            }

            if (fs::exists(bin_path))
            {
                result.actual_file_size = getFileSize(bin_path);

                // Size matches if metadata byte_size equals actual file size
                result.size_matches = (result.metadata_byte_size == result.actual_file_size);

                // Compute expected block count from file size
                // Handle both old dtype (Q16_1) and new dtypes (Q16_1_32, Q16_1_64, Q16_1_128)
                if (result.dtype == "Q8_1")
                {
                    result.computed_block_count = result.actual_file_size / 36; // 36 bytes per Q8_1 block
                }
                else if (result.dtype == "Q16_1" || result.dtype == "Q16_1_32")
                {
                    result.computed_block_count = result.actual_file_size / 72; // 72 bytes per Q16_1 block (32 elements)
                }
                else if (result.dtype == "Q16_1_64")
                {
                    result.computed_block_count = result.actual_file_size / 136; // 136 bytes per Q16_1_64 block
                }
                else if (result.dtype == "Q16_1_128")
                {
                    result.computed_block_count = result.actual_file_size / 264; // 264 bytes per Q16_1_128 block
                }
                else if (result.dtype == "Q8_0")
                {
                    result.computed_block_count = result.actual_file_size / 34; // 34 bytes per Q8_0 block
                }
                else if (result.dtype == "FP32")
                {
                    result.computed_block_count = 0; // Not block-based
                }

                if (result.metadata_block_count > 0)
                {
                    result.block_count_valid = (result.metadata_block_count == result.computed_block_count);
                }
                else
                {
                    result.block_count_valid = true; // Non-block format, no validation needed
                }
            }
            else
            {
                // No binary file found - this is valid if metadata indicates no data (byte_size=0)
                // e.g., weight metadata without actual weight dump
                if (result.metadata_byte_size == 0)
                {
                    result.size_matches = true; // Metadata-only entry
                    result.block_count_valid = true;
                }
                else
                {
                    // Metadata indicates data should exist, but file is missing - error
                    result.size_matches = false;
                    result.block_count_valid = false;
                }
            }

            return result;
        }
    };

    /**
     * @brief Test that layer filtering only dumps requested layers
     */
    TEST_F(Test__StageDumpIntegrity, LayerFilteringWorksCorrectly)
    {
        // Request only layers 0 and 5
        std::set<int> requested_layers = {0, 5};
        // Use GEMM stage type which definitely exists in the pipeline
        runInferenceWithDumps(requested_layers, "GEMM");

        // Get all dump directories
        auto dumps = getDumpDirectories();
        ASSERT_FALSE(dumps.empty()) << "No dumps created";

        // Extract layers from directory names
        std::set<int> dumped_layers;
        for (const auto &dump_dir : dumps)
        {
            int layer = extractLayerFromDirName(dump_dir.filename().string());
            if (layer >= 0)
            {
                dumped_layers.insert(layer);
            }
        }

        // Verify only requested layers were dumped
        LOG_INFO("[StageDumpIntegrity] Requested layers: {0, 5}");
        LOG_INFO("[StageDumpIntegrity] Dumped layers: " << [&]()
                 {
            std::ostringstream oss;
            oss << "{";
            bool first = true;
            for (int l : dumped_layers) {
                if (!first) oss << ", ";
                oss << l;
                first = false;
            }
            oss << "}";
            return oss.str(); }());
        LOG_INFO("[StageDumpIntegrity] Total dumps: " << dumps.size());

        // Verify only requested layers were dumped (multiple stages per layer is OK)
        EXPECT_EQ(dumped_layers, requested_layers)
            << "Layer filtering did not work correctly. "
            << "Expected only layers 0 and 5 to be dumped.";

        // Verify we got at least one dump per requested layer
        EXPECT_GE(dumps.size(), requested_layers.size())
            << "Expected at least " << requested_layers.size() << " dumps, got " << dumps.size();
    }

    /**
     * @brief Test that metadata byte_size matches actual file size
     */
    TEST_F(Test__StageDumpIntegrity, MetadataByteSizeMatchesFileSize)
    {
        // Dump layer 0 only
        std::set<int> requested_layers = {0};
        runInferenceWithDumps(requested_layers, "GEMM");

        auto dumps = getDumpDirectories();
        ASSERT_FALSE(dumps.empty()) << "No dumps created";

        // Pick the first dump directory
        const auto &dump_dir = dumps[0];
        LOG_INFO("[StageDumpIntegrity] Verifying dump: " << dump_dir.filename().string());

        // Enumerate all metadata files in inputs/ and outputs/
        int verified_count = 0;
        int mismatch_count = 0;

        for (const auto &subdir : {"inputs", "outputs", "weights"})
        {
            fs::path subdir_path = dump_dir / subdir;
            if (!fs::exists(subdir_path))
                continue;

            for (const auto &entry : fs::directory_iterator(subdir_path))
            {
                std::string filename = entry.path().filename().string();
                if (filename.find("_meta.txt") == std::string::npos)
                    continue;

                // Extract tensor name (remove _meta.txt suffix)
                std::string tensor_name = filename.substr(0, filename.length() - 9);

                auto result = verifyTensorDump(dump_dir, subdir, tensor_name);
                if (result.exists)
                {
                    verified_count++;
                    LOG_INFO("[StageDumpIntegrity] " << subdir << "/" << tensor_name
                                                     << ": dtype=" << result.dtype
                                                     << " meta_bytes=" << result.metadata_byte_size
                                                     << " actual_bytes=" << result.actual_file_size
                                                     << " match=" << (result.size_matches ? "YES" : "NO"));

                    if (!result.size_matches)
                    {
                        mismatch_count++;
                        ADD_FAILURE() << "Tensor " << subdir << "/" << tensor_name << " size mismatch: "
                                      << "metadata says " << result.metadata_byte_size << " bytes, "
                                      << "file is " << result.actual_file_size << " bytes";
                    }
                }
            }
        }

        EXPECT_GT(verified_count, 0) << "No tensors were verified";
        EXPECT_EQ(mismatch_count, 0) << "Some tensors had size mismatches";
    }

    /**
     * @brief Test that block counts in metadata are mathematically correct
     */
    TEST_F(Test__StageDumpIntegrity, BlockCountsAreCorrect)
    {
        // Dump layer 0 only
        std::set<int> requested_layers = {0};
        runInferenceWithDumps(requested_layers, "GEMM");

        auto dumps = getDumpDirectories();
        ASSERT_FALSE(dumps.empty()) << "No dumps created";

        const auto &dump_dir = dumps[0];

        // Enumerate all metadata files and check block counts for quantized tensors
        int verified_count = 0;
        int mismatch_count = 0;

        for (const auto &subdir : {"inputs", "outputs", "weights"})
        {
            fs::path subdir_path = dump_dir / subdir;
            if (!fs::exists(subdir_path))
                continue;

            for (const auto &entry : fs::directory_iterator(subdir_path))
            {
                std::string filename = entry.path().filename().string();
                if (filename.find("_meta.txt") == std::string::npos)
                    continue;

                std::string tensor_name = filename.substr(0, filename.length() - 9);
                auto result = verifyTensorDump(dump_dir, subdir, tensor_name);

                if (result.exists && result.metadata_block_count > 0)
                {
                    verified_count++;
                    LOG_INFO("[StageDumpIntegrity] " << subdir << "/" << tensor_name
                                                     << ": meta_blocks=" << result.metadata_block_count
                                                     << " computed_blocks=" << result.computed_block_count
                                                     << " match=" << (result.block_count_valid ? "YES" : "NO"));

                    if (!result.block_count_valid)
                    {
                        mismatch_count++;
                        ADD_FAILURE() << "Tensor " << subdir << "/" << tensor_name << " block count mismatch: "
                                      << "metadata says " << result.metadata_block_count << " blocks, "
                                      << "computed from file size: " << result.computed_block_count << " blocks";
                    }
                }
            }
        }

        // Note: FP32 activation mode may not have quantized tensors in GEMM stages
        // This is OK - we just verify that if block counts exist, they're correct
        if (verified_count == 0)
        {
            LOG_INFO("[StageDumpIntegrity] No quantized tensors found in dumps (FP32 mode?)");
        }
        else
        {
            EXPECT_EQ(mismatch_count, 0) << "Some tensors had block count mismatches";
        }
    }

    /**
     * @brief Test that metadata rows×cols produces correct block count
     */
    TEST_F(Test__StageDumpIntegrity, MetadataDimensionsProduceCorrectBlockCount)
    {
        // Dump layer 0 only
        std::set<int> requested_layers = {0};
        runInferenceWithDumps(requested_layers, "GEMM");

        auto dumps = getDumpDirectories();
        ASSERT_FALSE(dumps.empty()) << "No dumps created";

        const auto &dump_dir = dumps[0];

        // For each quantized tensor, verify:
        // block_count == rows × ceil(cols/32)
        constexpr size_t BLOCK_SIZE = 32;

        int verified_count = 0;
        int mismatch_count = 0;

        for (const auto &subdir : {"inputs", "outputs", "weights"})
        {
            fs::path subdir_path = dump_dir / subdir;
            if (!fs::exists(subdir_path))
                continue;

            for (const auto &entry : fs::directory_iterator(subdir_path))
            {
                std::string filename = entry.path().filename().string();
                if (filename.find("_meta.txt") == std::string::npos)
                    continue;

                std::string tensor_name = filename.substr(0, filename.length() - 9);
                fs::path meta_path = dump_dir / subdir / (tensor_name + "_meta.txt");
                auto meta = parseMetadataFile(meta_path);

                // Skip if not a block-based format
                if (!meta.count("block_count") || std::stoull(meta["block_count"]) == 0)
                    continue;

                size_t rows = std::stoull(meta["rows"]);
                size_t cols = std::stoull(meta["cols"]);
                size_t block_count = std::stoull(meta["block_count"]);
                size_t blocks_per_row = meta.count("blocks_per_row") ? std::stoull(meta["blocks_per_row"]) : 0;

                size_t expected_blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
                size_t expected_block_count = rows * expected_blocks_per_row;

                verified_count++;
                LOG_INFO("[StageDumpIntegrity] " << subdir << "/" << tensor_name
                                                 << ": [" << rows << ", " << cols << "] -> "
                                                 << "expected " << expected_block_count << " blocks, "
                                                 << "metadata says " << block_count << " blocks");

                if (block_count != expected_block_count)
                {
                    mismatch_count++;
                    ADD_FAILURE() << "Tensor " << subdir << "/" << tensor_name << " block count formula mismatch: "
                                  << "rows=" << rows << " cols=" << cols << " should give "
                                  << expected_block_count << " blocks, got " << block_count;
                }

                if (blocks_per_row > 0 && blocks_per_row != expected_blocks_per_row)
                {
                    mismatch_count++;
                    ADD_FAILURE() << "Tensor " << subdir << "/" << tensor_name << " blocks_per_row mismatch: "
                                  << "cols=" << cols << " should give "
                                  << expected_blocks_per_row << " blocks/row, got " << blocks_per_row;
                }
            }
        }

        // Note: FP32 activation mode may not have quantized tensors
        if (verified_count == 0)
        {
            LOG_INFO("[StageDumpIntegrity] No quantized tensors found in dumps (FP32 mode?)");
        }
        else
        {
            EXPECT_EQ(mismatch_count, 0) << "Some tensors had dimension/block count mismatches";
        }
    }

    /**
     * @brief Test that multiple layers filter correctly
     */
    TEST_F(Test__StageDumpIntegrity, MultipleLayerFilteringWorks)
    {
        // Request layers 0, 3, 7, 15
        std::set<int> requested_layers = {0, 3, 7, 15};
        runInferenceWithDumps(requested_layers, "GEMM");

        auto dumps = getDumpDirectories();
        ASSERT_FALSE(dumps.empty()) << "No dumps created";

        // Extract layers
        std::set<int> dumped_layers;
        for (const auto &dump_dir : dumps)
        {
            int layer = extractLayerFromDirName(dump_dir.filename().string());
            if (layer >= 0)
            {
                dumped_layers.insert(layer);
            }
        }

        LOG_INFO("[StageDumpIntegrity] Multi-layer test: requested={0,3,7,15} dumped="
                 << [&]()
                 {
                     std::ostringstream oss;
                     oss << "{";
                     bool first = true;
                     for (int l : dumped_layers)
                     {
                         if (!first)
                             oss << ", ";
                         oss << l;
                         first = false;
                     }
                     oss << "}";
                     return oss.str(); }());

        // Layer filtering should work correctly - only requested layers dumped
        EXPECT_EQ(dumped_layers, requested_layers)
            << "Multi-layer filtering failed";
    }

    /**
     * @brief Test that "all" layers filter dumps all layers
     */
    TEST_F(Test__StageDumpIntegrity, AllLayersFilterDumpsAllLayers)
    {
        // Set up for "all" layers
        setenv("LLAMINAR_STAGE_DUMP_ENABLED", "1", 1);
        setenv("LLAMINAR_STAGE_DUMP_DIR", DUMP_DIR, 1);
        setenv("LLAMINAR_STAGE_DUMP_TYPES", "GEMM", 1);
        setenv("LLAMINAR_STAGE_DUMP_LAYERS", "all", 1);
        mutableDebugEnv().stage_dump.reload();

        // Create MPI context and load model
        auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
        auto model_ctx = ModelContext::create(MODEL_PATH, mpi_ctx);
        ASSERT_NE(model_ctx, nullptr);

        auto runner = createInferenceRunner(model_ctx, mpi_ctx,
                                            DeviceId::cpu());
        ASSERT_NE(runner, nullptr);

        std::vector<int32_t> tokens = {151644, 872, 198};
        runner->forward(tokens.data(), tokens.size());

        setenv("LLAMINAR_STAGE_DUMP_ENABLED", "0", 1);
        mutableDebugEnv().stage_dump.reload();

        // Verify all 24 layers were dumped
        auto dumps = getDumpDirectories();
        std::set<int> dumped_layers;
        for (const auto &dump_dir : dumps)
        {
            int layer = extractLayerFromDirName(dump_dir.filename().string());
            if (layer >= 0)
            {
                dumped_layers.insert(layer);
            }
        }

        LOG_INFO("[StageDumpIntegrity] All-layers test: dumped " << dumped_layers.size() << " layers");

        // Qwen2.5-0.5B has 24 layers
        EXPECT_EQ(dumped_layers.size(), 24)
            << "Expected all 24 layers to be dumped with 'all' filter";

        // Verify it's actually 0-23
        for (int i = 0; i < 24; ++i)
        {
            EXPECT_TRUE(dumped_layers.count(i) > 0)
                << "Missing layer " << i << " from 'all' dump";
        }
    }

    // =========================================================================
    // Q16_1 Variable Block Size Tests
    // =========================================================================

    /**
     * @brief Helper class for Q16_1 block size dump verification
     *
     * Creates Q16_1 tensors with different block sizes, simulates dumping,
     * and verifies the metadata correctly reflects block size and byte counts.
     */
    class Test__Q16_1BlockSizeDump : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            DeviceManager::instance().initialize(-1);
        }
    };

    /**
     * @brief Verify computeByteSizeForDtype handles Q16_1 variant dtypes correctly
     */
    TEST_F(Test__Q16_1BlockSizeDump, ComputeByteSizeForDtypeHandlesQ16_1Variants)
    {
        // Test data: 4 rows x 64 cols
        constexpr size_t ROWS = 4;
        constexpr size_t COLS = 64;

        // Q16_1 / Q16_1_32: 32 elements/block, 72 bytes/block
        // 64 cols = 2 blocks per row, 4 rows = 8 blocks = 576 bytes
        {
            size_t byte_size = computeByteSizeForDtype("Q16_1", ROWS, COLS);
            size_t expected = ROWS * 2 * 72; // 4 rows * 2 blocks/row * 72 bytes/block
            EXPECT_EQ(byte_size, expected) << "Q16_1 (default) byte size mismatch";
        }
        {
            size_t byte_size = computeByteSizeForDtype("Q16_1_32", ROWS, COLS);
            size_t expected = ROWS * 2 * 72;
            EXPECT_EQ(byte_size, expected) << "Q16_1_32 byte size mismatch";
        }

        // Q16_1_64: 64 elements/block, 136 bytes/block
        // 64 cols = 1 block per row, 4 rows = 4 blocks = 544 bytes
        {
            size_t byte_size = computeByteSizeForDtype("Q16_1_64", ROWS, COLS);
            size_t expected = ROWS * 1 * 136; // 4 rows * 1 block/row * 136 bytes/block
            EXPECT_EQ(byte_size, expected) << "Q16_1_64 byte size mismatch";
        }

        // Q16_1_128: 128 elements/block, 264 bytes/block
        // 64 cols = 1 block per row (ceil), 4 rows = 4 blocks = 1056 bytes
        {
            size_t byte_size = computeByteSizeForDtype("Q16_1_128", ROWS, COLS);
            size_t expected = ROWS * 1 * 264; // 4 rows * 1 block/row * 264 bytes/block
            EXPECT_EQ(byte_size, expected) << "Q16_1_128 byte size mismatch";
        }

        // Test with cols = 128 for complete block alignment with Q16_1_128
        {
            size_t byte_size = computeByteSizeForDtype("Q16_1_128", ROWS, 128);
            size_t expected = ROWS * 1 * 264;
            EXPECT_EQ(byte_size, expected) << "Q16_1_128 (aligned) byte size mismatch";
        }
    }

    /**
     * @brief Verify Q16_1Tensor.dtype_name_with_block_size() returns correct strings
     */
    TEST_F(Test__Q16_1BlockSizeDump, DtypeNameWithBlockSizeReturnsCorrectStrings)
    {
        // Create Q16_1 tensors with different block sizes
        std::vector<size_t> shape = {4, 64}; // 4 rows x 64 cols

        // Block size 32 (default)
        {
            auto tensor = std::make_shared<Q16_1Tensor>(shape, Q16BlockSize::BLOCK_32);
            EXPECT_STREQ(tensor->dtype_name_with_block_size(), "Q16_1_32");
            EXPECT_EQ(tensor->q16_block_size(), Q16BlockSize::BLOCK_32);
        }

        // Block size 64
        {
            auto tensor = std::make_shared<Q16_1Tensor>(shape, Q16BlockSize::BLOCK_64);
            EXPECT_STREQ(tensor->dtype_name_with_block_size(), "Q16_1_64");
            EXPECT_EQ(tensor->q16_block_size(), Q16BlockSize::BLOCK_64);
        }

        // Block size 128
        {
            auto tensor = std::make_shared<Q16_1Tensor>(shape, Q16BlockSize::BLOCK_128);
            EXPECT_STREQ(tensor->dtype_name_with_block_size(), "Q16_1_128");
            EXPECT_EQ(tensor->q16_block_size(), Q16BlockSize::BLOCK_128);
        }
    }

    /**
     * @brief Verify StageDumpInfo correctly uses block-size-aware dtype for Q16_1 tensors
     */
    TEST_F(Test__Q16_1BlockSizeDump, StageDumpInfoUsesBlockSizeAwareDtype)
    {
        constexpr size_t ROWS = 4;
        constexpr size_t COLS = 64;

        // Test Q16_1_64 tensor
        auto tensor_64 = std::make_shared<Q16_1Tensor>(
            std::vector<size_t>{ROWS, COLS}, Q16BlockSize::BLOCK_64);

        // Fill with some test data to make it valid
        float *fp32_data = new float[ROWS * COLS];
        for (size_t i = 0; i < ROWS * COLS; ++i)
        {
            fp32_data[i] = static_cast<float>(i) * 0.01f;
        }
        tensor_64->copyFrom_fp32(fp32_data);
        delete[] fp32_data;

        StageDumpInfo info;
        info.addInput("test_tensor", tensor_64.get(), ROWS, COLS);

        ASSERT_EQ(info.inputs.size(), 1);
        const auto &input = info.inputs[0];

        // Verify dtype includes block size
        EXPECT_STREQ(input.dtype, "Q16_1_64")
            << "Expected dtype 'Q16_1_64' for Q16_1Tensor with BLOCK_64";

        // Verify byte size is computed correctly for Q16_1_64
        // 64 cols = 1 block per row, 4 rows = 4 blocks, 136 bytes/block = 544 bytes
        size_t expected_bytes = ROWS * 1 * 136;
        EXPECT_EQ(input.byte_size, expected_bytes)
            << "Expected byte_size=" << expected_bytes << " for Q16_1_64";
    }

    /**
     * @brief Verify TensorDumpMeta.computeBlockInfo() handles Q16_1 variants
     */
    TEST_F(Test__Q16_1BlockSizeDump, TensorDumpMetaComputeBlockInfoHandlesVariants)
    {
        constexpr size_t ROWS = 4;
        constexpr size_t COLS = 64;

        // Test Q16_1_32 (2 blocks per row)
        {
            TensorDumpMeta meta;
            meta.rows = ROWS;
            meta.cols = COLS;
            meta.dtype = "Q16_1_32";
            meta.computeBlockInfo();

            EXPECT_EQ(meta.block_element_size, 32);
            EXPECT_EQ(meta.blocks_per_row, 2);     // 64 / 32 = 2
            EXPECT_EQ(meta.block_count, ROWS * 2); // 4 * 2 = 8
        }

        // Test Q16_1_64 (1 block per row)
        {
            TensorDumpMeta meta;
            meta.rows = ROWS;
            meta.cols = COLS;
            meta.dtype = "Q16_1_64";
            meta.computeBlockInfo();

            EXPECT_EQ(meta.block_element_size, 64);
            EXPECT_EQ(meta.blocks_per_row, 1);     // 64 / 64 = 1
            EXPECT_EQ(meta.block_count, ROWS * 1); // 4 * 1 = 4
        }

        // Test Q16_1_128 (1 block per row, even though only 64 elements)
        {
            TensorDumpMeta meta;
            meta.rows = ROWS;
            meta.cols = COLS;
            meta.dtype = "Q16_1_128";
            meta.computeBlockInfo();

            EXPECT_EQ(meta.block_element_size, 128);
            EXPECT_EQ(meta.blocks_per_row, 1);     // ceil(64 / 128) = 1
            EXPECT_EQ(meta.block_count, ROWS * 1); // 4 * 1 = 4
        }
    }

    /**
     * @brief End-to-end test: Create Q16_1 tensors with different block sizes,
     * simulate stage dump, verify metadata is correct
     */
    TEST_F(Test__Q16_1BlockSizeDump, EndToEndDumpMetadataIntegrity)
    {
        // This test verifies the full pipeline from tensor creation to dump metadata
        // without actually writing to disk

        constexpr size_t ROWS = 14; // n_heads (Qwen2.5-0.5B)
        constexpr size_t COLS = 64; // head_dim (Qwen2.5-0.5B)

        // Create tensors with different block sizes
        auto tensor_32 = std::make_shared<Q16_1Tensor>(
            std::vector<size_t>{ROWS, COLS}, Q16BlockSize::BLOCK_32);
        auto tensor_64 = std::make_shared<Q16_1Tensor>(
            std::vector<size_t>{ROWS, COLS}, Q16BlockSize::BLOCK_64);
        auto tensor_128 = std::make_shared<Q16_1Tensor>(
            std::vector<size_t>{ROWS, COLS}, Q16BlockSize::BLOCK_128);

        // Fill with test data
        std::vector<float> test_data(ROWS * COLS);
        for (size_t i = 0; i < test_data.size(); ++i)
        {
            test_data[i] = static_cast<float>(i) * 0.001f - 0.5f;
        }
        tensor_32->copyFrom_fp32(test_data.data());
        tensor_64->copyFrom_fp32(test_data.data());
        tensor_128->copyFrom_fp32(test_data.data());

        // Create StageDumpInfo with all three tensors
        StageDumpInfo info;
        info.addInput("Q_32", tensor_32.get(), ROWS, COLS);
        info.addInput("Q_64", tensor_64.get(), ROWS, COLS);
        info.addInput("Q_128", tensor_128.get(), ROWS, COLS);

        ASSERT_EQ(info.inputs.size(), 3);

        // Verify Q_32 (BLOCK_32: 2 blocks per row, 72 bytes per block)
        {
            const auto &input = info.inputs[0];
            EXPECT_STREQ(input.dtype, "Q16_1_32");
            size_t expected_blocks = ROWS * 2;            // 14 * 2 = 28 blocks
            size_t expected_bytes = expected_blocks * 72; // 28 * 72 = 2016 bytes
            EXPECT_EQ(input.byte_size, expected_bytes)
                << "Q_32 byte_size mismatch";
        }

        // Verify Q_64 (BLOCK_64: 1 block per row, 136 bytes per block)
        {
            const auto &input = info.inputs[1];
            EXPECT_STREQ(input.dtype, "Q16_1_64");
            size_t expected_blocks = ROWS * 1;             // 14 * 1 = 14 blocks
            size_t expected_bytes = expected_blocks * 136; // 14 * 136 = 1904 bytes
            EXPECT_EQ(input.byte_size, expected_bytes)
                << "Q_64 byte_size mismatch";
        }

        // Verify Q_128 (BLOCK_128: 1 block per row, 264 bytes per block)
        {
            const auto &input = info.inputs[2];
            EXPECT_STREQ(input.dtype, "Q16_1_128");
            size_t expected_blocks = ROWS * 1;             // 14 * 1 = 14 blocks
            size_t expected_bytes = expected_blocks * 264; // 14 * 264 = 3696 bytes
            EXPECT_EQ(input.byte_size, expected_bytes)
                << "Q_128 byte_size mismatch";
        }

        // Verify actual tensor sizes match expected
        EXPECT_EQ(tensor_32->size_bytes(), ROWS * 2 * 72);
        EXPECT_EQ(tensor_64->size_bytes(), ROWS * 1 * 136);
        EXPECT_EQ(tensor_128->size_bytes(), ROWS * 1 * 264);
    }

} // namespace llaminar2::test
