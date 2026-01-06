/**
 * @file TensorDumpLoader.h
 * @brief Utility functions for loading native-format tensor dumps from StageDumper
 *
 * This header provides functions to load tensor data captured by the StageDumper
 * in their native format (Q8_1, Q16_1, FP32) rather than dequantized FP32.
 *
 * @see StageDumper.h for the dump format specification
 * @see tests/v2/integration/_data/ for persistent test data
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <stdexcept>

#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h" // For fp16_to_fp32()

namespace llaminar2::test
{
    namespace fs = std::filesystem;

    /**
     * @brief Metadata parsed from *_meta.txt files
     */
    struct TensorDumpMeta
    {
        std::string name;
        int rows = 0;
        int cols = 0;
        std::string dtype; // "FP32", "Q8_1", "Q16_1", etc.
        size_t element_count = 0;
        size_t byte_size = 0;
        float sample_min = 0.0f;
        float sample_max = 0.0f;
        float sample_mean = 0.0f;
    };

    /**
     * @brief Stage dump metadata parsed from metadata.txt
     */
    struct StageDumpMeta
    {
        int dump_id = 0;
        std::string type;
        int layer_idx = -1;
        int iteration = -1;
        int rank = 0;
        float execution_time_ms = 0.0f;
        int inputs_dumped = 0;
        int outputs_dumped = 0;
    };

    /**
     * @brief Parse tensor metadata from a *_meta.txt file
     */
    inline TensorDumpMeta parseTensorMeta(const std::string &meta_path)
    {
        TensorDumpMeta meta;
        std::ifstream file(meta_path);
        if (!file.is_open())
        {
            throw std::runtime_error("Cannot open metadata file: " + meta_path);
        }

        std::string line;
        while (std::getline(file, line))
        {
            size_t eq_pos = line.find('=');
            if (eq_pos == std::string::npos)
                continue;

            std::string key = line.substr(0, eq_pos);
            std::string value = line.substr(eq_pos + 1);

            if (key == "name")
                meta.name = value;
            else if (key == "rows")
                meta.rows = std::stoi(value);
            else if (key == "cols")
                meta.cols = std::stoi(value);
            else if (key == "dtype")
                meta.dtype = value;
            else if (key == "element_count")
                meta.element_count = std::stoull(value);
            else if (key == "byte_size")
                meta.byte_size = std::stoull(value);
            else if (key == "sample_min")
                meta.sample_min = std::stof(value);
            else if (key == "sample_max")
                meta.sample_max = std::stof(value);
            else if (key == "sample_mean")
                meta.sample_mean = std::stof(value);
        }
        return meta;
    }

    /**
     * @brief Parse stage dump metadata from metadata.txt
     */
    inline StageDumpMeta parseStageMeta(const std::string &dump_dir)
    {
        StageDumpMeta meta;
        std::ifstream file(dump_dir + "/metadata.txt");
        if (!file.is_open())
        {
            throw std::runtime_error("Cannot open metadata file: " + dump_dir + "/metadata.txt");
        }

        std::string line;
        while (std::getline(file, line))
        {
            if (line.empty() || line[0] == '#')
                continue;

            size_t eq_pos = line.find('=');
            if (eq_pos == std::string::npos)
                continue;

            std::string key = line.substr(0, eq_pos);
            std::string value = line.substr(eq_pos + 1);

            if (key == "dump_id")
                meta.dump_id = std::stoi(value);
            else if (key == "type")
                meta.type = value;
            else if (key == "layer_idx")
                meta.layer_idx = std::stoi(value);
            else if (key == "iteration")
                meta.iteration = std::stoi(value);
            else if (key == "rank")
                meta.rank = std::stoi(value);
            else if (key == "execution_time_ms")
                meta.execution_time_ms = std::stof(value);
            else if (key == "inputs_dumped")
                meta.inputs_dumped = std::stoi(value);
            else if (key == "outputs_dumped")
                meta.outputs_dumped = std::stoi(value);
        }
        return meta;
    }

    /**
     * @brief Load raw binary data from a file
     */
    inline std::vector<uint8_t> loadRawBytes(const std::string &path, size_t expected_size = 0)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open())
        {
            throw std::runtime_error("Cannot open file: " + path);
        }

        size_t file_size = file.tellg();
        file.seekg(0);

        if (expected_size > 0 && file_size != expected_size)
        {
            throw std::runtime_error("File " + path + " has " + std::to_string(file_size) +
                                     " bytes, expected " + std::to_string(expected_size));
        }

        std::vector<uint8_t> data(file_size);
        file.read(reinterpret_cast<char *>(data.data()), file_size);
        return data;
    }

    /**
     * @brief Load FP32 tensor data from a .bin file
     */
    inline std::vector<float> loadFP32Tensor(const std::string &path, size_t expected_elements = 0)
    {
        auto bytes = loadRawBytes(path);
        if (bytes.size() % sizeof(float) != 0)
        {
            throw std::runtime_error("File size not divisible by sizeof(float): " + path);
        }

        size_t num_elements = bytes.size() / sizeof(float);
        if (expected_elements > 0 && num_elements != expected_elements)
        {
            throw std::runtime_error("File " + path + " has " + std::to_string(num_elements) +
                                     " elements, expected " + std::to_string(expected_elements));
        }

        std::vector<float> data(num_elements);
        std::memcpy(data.data(), bytes.data(), bytes.size());
        return data;
    }

    /**
     * @brief Load Q8_1 blocks from a native format dump
     */
    inline std::vector<Q8_1Block> loadQ8_1Blocks(const std::string &path, size_t expected_blocks = 0)
    {
        auto bytes = loadRawBytes(path);
        if (bytes.size() % sizeof(Q8_1Block) != 0)
        {
            throw std::runtime_error("File size not divisible by sizeof(Q8_1Block): " + path);
        }

        size_t num_blocks = bytes.size() / sizeof(Q8_1Block);
        if (expected_blocks > 0 && num_blocks != expected_blocks)
        {
            throw std::runtime_error("File " + path + " has " + std::to_string(num_blocks) +
                                     " blocks, expected " + std::to_string(expected_blocks));
        }

        std::vector<Q8_1Block> blocks(num_blocks);
        std::memcpy(blocks.data(), bytes.data(), bytes.size());
        return blocks;
    }

    /**
     * @brief Load Q16_1 blocks from a native format dump
     */
    inline std::vector<Q16_1Block> loadQ16_1Blocks(const std::string &path, size_t expected_blocks = 0)
    {
        auto bytes = loadRawBytes(path);
        if (bytes.size() % sizeof(Q16_1Block) != 0)
        {
            throw std::runtime_error("File size not divisible by sizeof(Q16_1Block): " + path);
        }

        size_t num_blocks = bytes.size() / sizeof(Q16_1Block);
        if (expected_blocks > 0 && num_blocks != expected_blocks)
        {
            throw std::runtime_error("File " + path + " has " + std::to_string(num_blocks) +
                                     " blocks, expected " + std::to_string(expected_blocks));
        }

        std::vector<Q16_1Block> blocks(num_blocks);
        std::memcpy(blocks.data(), bytes.data(), bytes.size());
        return blocks;
    }

    /**
     * @brief Load Q16_1Block_64 (64-element) blocks from a native format dump
     */
    inline std::vector<Q16_1Block_64> loadQ16_1_64Blocks(const std::string &path, size_t expected_blocks = 0)
    {
        auto bytes = loadRawBytes(path);
        if (bytes.size() % sizeof(Q16_1Block_64) != 0)
        {
            throw std::runtime_error("File size not divisible by sizeof(Q16_1Block_64): " + path);
        }

        size_t num_blocks = bytes.size() / sizeof(Q16_1Block_64);
        if (expected_blocks > 0 && num_blocks != expected_blocks)
        {
            throw std::runtime_error("File " + path + " has " + std::to_string(num_blocks) +
                                     " Q16_1Block_64 blocks, expected " + std::to_string(expected_blocks));
        }

        std::vector<Q16_1Block_64> blocks(num_blocks);
        std::memcpy(blocks.data(), bytes.data(), bytes.size());
        return blocks;
    }

    /**
     * @brief Dequantize Q16_1 blocks to FP32
     * @param blocks Q16_1 block array
     * @param rows Number of rows
     * @param cols Number of columns
     * @return FP32 tensor data [rows, cols]
     */
    inline std::vector<float> dequantQ16_1ToFP32(
        const std::vector<Q16_1Block> &blocks,
        int rows,
        int cols)
    {
        constexpr int BLOCK_SIZE = 32;
        const size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

        std::vector<float> output(rows * cols);

        for (int row = 0; row < rows; ++row)
        {
            for (size_t block_idx = 0; block_idx < blocks_per_row; ++block_idx)
            {
                const Q16_1Block &block = blocks[row * blocks_per_row + block_idx];
                float d = block.d;

                for (int k = 0; k < BLOCK_SIZE; ++k)
                {
                    int col = block_idx * BLOCK_SIZE + k;
                    if (col < cols)
                    {
                        output[row * cols + col] = static_cast<float>(block.qs[k]) * d;
                    }
                }
            }
        }
        return output;
    }

    /**
     * @brief Dequantize Q16_1Block_64 (64-element) blocks to FP32
     * @param blocks Q16_1Block_64 block array
     * @param rows Number of rows
     * @param cols Number of columns
     * @return FP32 tensor data [rows, cols]
     */
    inline std::vector<float> dequantQ16_1_64ToFP32(
        const std::vector<Q16_1Block_64> &blocks,
        int rows,
        int cols)
    {
        constexpr int BLOCK_SIZE = 64;
        const size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

        std::vector<float> output(rows * cols);

        for (int row = 0; row < rows; ++row)
        {
            for (size_t block_idx = 0; block_idx < blocks_per_row; ++block_idx)
            {
                const Q16_1Block_64 &block = blocks[row * blocks_per_row + block_idx];
                float d = block.d;

                for (int k = 0; k < BLOCK_SIZE; ++k)
                {
                    int col = block_idx * BLOCK_SIZE + k;
                    if (col < cols)
                    {
                        output[row * cols + col] = static_cast<float>(block.qs[k]) * d;
                    }
                }
            }
        }
        return output;
    }

    /**
     * @brief Dequantize Q8_1 blocks to FP32
     * @param blocks Q8_1 block array
     * @param rows Number of rows
     * @param cols Number of columns
     * @return FP32 tensor data [rows, cols]
     */
    inline std::vector<float> dequantQ8_1ToFP32(
        const std::vector<Q8_1Block> &blocks,
        int rows,
        int cols)
    {
        constexpr int BLOCK_SIZE = 32;
        const size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

        std::vector<float> output(rows * cols);

        for (int row = 0; row < rows; ++row)
        {
            for (size_t block_idx = 0; block_idx < blocks_per_row; ++block_idx)
            {
                const Q8_1Block &block = blocks[row * blocks_per_row + block_idx];
                float d = fp16_to_fp32(block.d);

                for (int k = 0; k < BLOCK_SIZE; ++k)
                {
                    int col = block_idx * BLOCK_SIZE + k;
                    if (col < cols)
                    {
                        output[row * cols + col] = static_cast<float>(block.qs[k]) * d;
                    }
                }
            }
        }
        return output;
    }

    /**
     * @brief Find file in dump directory, handling dtype suffixes
     *
     * The new native format may store files as:
     * - Q.bin (FP32)
     * - Q_q16_1.bin (Q16_1 native)
     * - V_q8_1.bin (Q8_1 native)
     */
    inline std::string findTensorFile(const std::string &dir, const std::string &name)
    {
        // Try exact match first
        std::string exact_path = dir + "/" + name + ".bin";
        if (fs::exists(exact_path))
        {
            return exact_path;
        }

        // Try with common dtype suffixes (including variable Q16_1 block sizes)
        for (const auto &suffix : {"_q16_1_32", "_q16_1_64", "_q16_1_128", "_q16_1", "_q8_1", "_q8_0",
                                   "_fp32", "_bf16", "_fp16"})
        {
            std::string suffixed_path = dir + "/" + name + suffix + ".bin";
            if (fs::exists(suffixed_path))
            {
                return suffixed_path;
            }
        }

        throw std::runtime_error("Cannot find tensor file for: " + name + " in " + dir);
    }

    /**
     * @brief Load tensor from dump directory with automatic format detection
     * @param dump_dir The stage dump directory
     * @param tensor_name Name of the tensor (e.g., "Q", "K", "V")
     * @param subdir Subdirectory ("inputs", "outputs", "weights")
     * @return Tuple of (dequantized FP32 data, metadata)
     */
    inline std::pair<std::vector<float>, TensorDumpMeta> loadTensorAsFP32(
        const std::string &dump_dir,
        const std::string &tensor_name,
        const std::string &subdir = "inputs")
    {
        std::string base_dir = dump_dir + "/" + subdir;

        // Find the tensor file
        std::string bin_path = findTensorFile(base_dir, tensor_name);

        // Find matching metadata file
        std::string meta_path = base_dir + "/" + tensor_name + "_meta.txt";
        if (!fs::exists(meta_path))
        {
            throw std::runtime_error("Cannot find metadata file: " + meta_path);
        }

        TensorDumpMeta meta = parseTensorMeta(meta_path);

        // Load and dequantize based on dtype
        // Handle Q16_1 variants: Q16_1, Q16_1_32, Q16_1_64, Q16_1_128
        std::vector<float> data;
        if (meta.dtype == "FP32")
        {
            data = loadFP32Tensor(bin_path);
        }
        else if (meta.dtype == "Q16_1" || meta.dtype == "Q16_1_32" ||
                 meta.dtype == "Q16_1_64" || meta.dtype == "Q16_1_128")
        {
            // Note: For Q16_1_64 and Q16_1_128, we'd need separate block loaders
            // For now, this only works correctly for Q16_1/Q16_1_32 (32-element blocks)
            auto blocks = loadQ16_1Blocks(bin_path);
            data = dequantQ16_1ToFP32(blocks, meta.rows, meta.cols);
        }
        else if (meta.dtype == "Q8_1")
        {
            auto blocks = loadQ8_1Blocks(bin_path);
            data = dequantQ8_1ToFP32(blocks, meta.rows, meta.cols);
        }
        else
        {
            throw std::runtime_error("Unsupported tensor dtype: " + meta.dtype);
        }

        return {data, meta};
    }

    /**
     * @brief Load Q16_1 tensor from dump directory, returning native blocks
     *
     * Note: This function assumes 32-element blocks (Q16_1Block).
     * For Q16_1_64 or Q16_1_128 tensors, the block count will be interpreted
     * differently. Use loadTensorAsQ16_1_64() or loadTensorAsQ16_1_128() for
     * variable block size tensors.
     */
    inline std::pair<std::vector<Q16_1Block>, TensorDumpMeta> loadTensorAsQ16_1(
        const std::string &dump_dir,
        const std::string &tensor_name,
        const std::string &subdir = "inputs")
    {
        std::string base_dir = dump_dir + "/" + subdir;
        std::string bin_path = findTensorFile(base_dir, tensor_name);
        std::string meta_path = base_dir + "/" + tensor_name + "_meta.txt";

        TensorDumpMeta meta = parseTensorMeta(meta_path);
        // Accept both old "Q16_1" and new "Q16_1_32" dtype strings
        if (meta.dtype != "Q16_1" && meta.dtype != "Q16_1_32")
        {
            throw std::runtime_error("Expected Q16_1 or Q16_1_32 tensor, got: " + meta.dtype);
        }

        auto blocks = loadQ16_1Blocks(bin_path);
        return {blocks, meta};
    }

    /**
     * @brief Load Q16_1_64 tensor from dump directory, returning native 64-element blocks
     */
    inline std::pair<std::vector<Q16_1Block_64>, TensorDumpMeta> loadTensorAsQ16_1_64(
        const std::string &dump_dir,
        const std::string &tensor_name,
        const std::string &subdir = "inputs")
    {
        std::string base_dir = dump_dir + "/" + subdir;
        std::string bin_path = findTensorFile(base_dir, tensor_name);
        std::string meta_path = base_dir + "/" + tensor_name + "_meta.txt";

        TensorDumpMeta meta = parseTensorMeta(meta_path);
        if (meta.dtype != "Q16_1_64")
        {
            throw std::runtime_error("Expected Q16_1_64 tensor, got: " + meta.dtype);
        }

        auto blocks = loadQ16_1_64Blocks(bin_path);
        return {blocks, meta};
    }

    /**
     * @brief Load Q8_1 tensor from dump directory, returning native blocks
     */
    inline std::pair<std::vector<Q8_1Block>, TensorDumpMeta> loadTensorAsQ8_1(
        const std::string &dump_dir,
        const std::string &tensor_name,
        const std::string &subdir = "inputs")
    {
        std::string base_dir = dump_dir + "/" + subdir;
        std::string bin_path = findTensorFile(base_dir, tensor_name);
        std::string meta_path = base_dir + "/" + tensor_name + "_meta.txt";

        TensorDumpMeta meta = parseTensorMeta(meta_path);
        if (meta.dtype != "Q8_1")
        {
            throw std::runtime_error("Expected Q8_1 tensor, got: " + meta.dtype);
        }

        auto blocks = loadQ8_1Blocks(bin_path);
        return {blocks, meta};
    }

    /**
     * @brief Get test data directory path
     * @return Absolute path to tests/v2/integration/_data/
     */
    inline std::string getTestDataDir()
    {
        // Try relative path from build directory
        std::vector<std::string> candidates = {
            "../../tests/v2/integration/_data",
            "../tests/v2/integration/_data",
            "tests/v2/integration/_data",
            "/workspaces/llaminar/tests/v2/integration/_data"};

        for (const auto &path : candidates)
        {
            if (fs::exists(path))
            {
                return fs::canonical(path).string();
            }
        }

        throw std::runtime_error("Cannot find test data directory");
    }

} // namespace llaminar2::test
