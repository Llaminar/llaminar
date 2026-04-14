/**
 * @file Test__ParallelismTreeParser.cpp
 * @brief Unit tests for ParallelismTreeParser
 *
 * Tests YAML and CLI parsing for ParallelismTree construction, including:
 * - Simple and complex YAML topologies
 * - CLI inline format parsing
 * - Error handling for malformed input
 * - Round-trip YAML serialization
 * - File-based parsing
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include "config/ParallelismTreeParser.h"
#include <fstream>
#include <filesystem>

using namespace llaminar2;

// =============================================================================
// Test Constants
// =============================================================================

constexpr int DEFAULT_TOTAL_LAYERS = 24;
constexpr int DEFAULT_WORLD_SIZE = 4;

// =============================================================================
// YAML Parsing Tests
// =============================================================================

TEST(Test__ParallelismTreeParser, ParseYAML_SimpleTPOnly)
{
    const char *yaml = R"(
topology:
  type: tp
  name: gpu_tp
  rank: 0
  devices: [cuda:0, cuda:1]
)";

    auto result = ParallelismTreeParser::parseYAML(yaml, DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);
    ASSERT_TRUE(result.success()) << result.errorString();
    ASSERT_TRUE(result.tree.has_value());

    const auto &tree = *result.tree;
    EXPECT_EQ(tree.root.type, ParallelismNodeType::TENSOR_PARALLEL);
    EXPECT_EQ(tree.root.name, "gpu_tp");
    EXPECT_EQ(tree.root.children.size(), 2);

    // All children should be DEVICE nodes
    for (const auto &child : tree.root.children)
    {
        EXPECT_EQ(child.type, ParallelismNodeType::DEVICE);
        EXPECT_EQ(child.owning_rank, 0);
    }

    // Layer assignment
    EXPECT_EQ(tree.root.first_layer, 0);
    EXPECT_EQ(tree.root.last_layer, DEFAULT_TOTAL_LAYERS - 1);
}

TEST(Test__ParallelismTreeParser, ParseYAML_SimplePPTP)
{
    const char *yaml = R"(
topology:
  type: pp
  name: global
  children:
    - type: tp
      name: socket0_tp
      rank: 0
      devices: [cuda:0, cuda:1]
    - type: tp
      name: socket1_tp
      rank: 1
      devices: [cuda:0, cuda:1]
)";

    auto result = ParallelismTreeParser::parseYAML(yaml, DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);
    ASSERT_TRUE(result.success()) << result.errorString();
    ASSERT_TRUE(result.tree.has_value());

    const auto &tree = *result.tree;
    EXPECT_EQ(tree.root.type, ParallelismNodeType::PIPELINE_PARALLEL);
    EXPECT_EQ(tree.root.name, "global");
    EXPECT_EQ(tree.root.children.size(), 2);

    // First TP node
    const auto &tp0 = tree.root.children[0];
    EXPECT_EQ(tp0.type, ParallelismNodeType::TENSOR_PARALLEL);
    EXPECT_EQ(tp0.name, "socket0_tp");
    EXPECT_EQ(tp0.owning_rank, 0);
    EXPECT_EQ(tp0.children.size(), 2);

    // Second TP node
    const auto &tp1 = tree.root.children[1];
    EXPECT_EQ(tp1.type, ParallelismNodeType::TENSOR_PARALLEL);
    EXPECT_EQ(tp1.name, "socket1_tp");
    EXPECT_EQ(tp1.owning_rank, 1);
    EXPECT_EQ(tp1.children.size(), 2);

    // Layer distribution (equal split)
    EXPECT_EQ(tp0.first_layer, 0);
    EXPECT_EQ(tp0.last_layer, 11); // 0-11 (12 layers)
    EXPECT_EQ(tp1.first_layer, 12);
    EXPECT_EQ(tp1.last_layer, 23); // 12-23 (12 layers)
}

TEST(Test__ParallelismTreeParser, ParseYAML_NestedPPTPPP)
{
    const char *yaml = R"(
topology:
  type: pp
  name: global
  children:
    - type: pp
      name: host0
      children:
        - type: tp
          name: socket0_tp
          rank: 0
          devices: [cuda:0, cuda:1]
        - type: tp
          name: socket1_tp
          rank: 1
          devices: [cuda:0, cuda:1]
    - type: pp
      name: host1
      children:
        - type: tp
          name: socket2_tp
          rank: 2
          devices: [cuda:0, cuda:1]
        - type: tp
          name: socket3_tp
          rank: 3
          devices: [cuda:0, cuda:1]
)";

    auto result = ParallelismTreeParser::parseYAML(yaml, DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);
    ASSERT_TRUE(result.success()) << result.errorString();
    ASSERT_TRUE(result.tree.has_value());

    const auto &tree = *result.tree;
    EXPECT_EQ(tree.root.type, ParallelismNodeType::PIPELINE_PARALLEL);
    EXPECT_EQ(tree.root.children.size(), 2);

    // Host0 (nested PP)
    const auto &host0 = tree.root.children[0];
    EXPECT_EQ(host0.type, ParallelismNodeType::PIPELINE_PARALLEL);
    EXPECT_EQ(host0.name, "host0");
    EXPECT_EQ(host0.children.size(), 2);

    // Verify 4-way layer distribution
    auto all_ranks = tree.root.leafRanks();
    EXPECT_EQ(all_ranks.size(), 4);
}

TEST(Test__ParallelismTreeParser, ParseYAML_WithBackends)
{
    const char *yaml = R"(
topology:
  type: pp
  name: global
  children:
    - type: tp
      name: nccl_tp
      rank: 0
      backend: nccl
      devices: [cuda:0, cuda:1]
    - type: tp
      name: rccl_tp
      rank: 1
      backend: rccl
      devices: [rocm:0, rocm:1]
)";

    auto result = ParallelismTreeParser::parseYAML(yaml, DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);
    ASSERT_TRUE(result.success()) << result.errorString();
    ASSERT_TRUE(result.tree.has_value());

    const auto &tree = *result.tree;
    EXPECT_EQ(tree.root.children[0].backend, CollectiveBackendType::NCCL);
    EXPECT_EQ(tree.root.children[1].backend, CollectiveBackendType::RCCL);
}

TEST(Test__ParallelismTreeParser, ParseYAML_WithWeights)
{
    const char *yaml = R"(
topology:
  type: pp
  name: global
  children:
    - type: tp
      name: fast_tp
      rank: 0
      devices: [cuda:0, cuda:1]
      weights: [0.6, 0.4]
    - type: tp
      name: slow_tp
      rank: 1
      devices: [rocm:0, rocm:1]
)";

    auto result = ParallelismTreeParser::parseYAML(yaml, DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);
    ASSERT_TRUE(result.success()) << result.errorString();
    ASSERT_TRUE(result.tree.has_value());

    const auto &tree = *result.tree;
    const auto &tp0 = tree.root.children[0];
    ASSERT_EQ(tp0.tp_weights.size(), 2);
    EXPECT_FLOAT_EQ(tp0.tp_weights[0], 0.6f);
    EXPECT_FLOAT_EQ(tp0.tp_weights[1], 0.4f);
}

TEST(Test__ParallelismTreeParser, ParseYAML_DeviceLeaf)
{
    const char *yaml = R"(
topology:
  type: device
  name: single_gpu
  rank: 0
  device: cuda:0
)";

    auto result = ParallelismTreeParser::parseYAML(yaml, DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);
    ASSERT_TRUE(result.success()) << result.errorString();
    ASSERT_TRUE(result.tree.has_value());

    const auto &tree = *result.tree;
    EXPECT_EQ(tree.root.type, ParallelismNodeType::DEVICE);
    EXPECT_EQ(tree.root.name, "single_gpu");
    EXPECT_EQ(tree.root.device.device_type, DeviceType::CUDA);
    EXPECT_EQ(tree.root.device.device_ordinal, 0);
}

TEST(Test__ParallelismTreeParser, ParseYAML_MixedVendorTP)
{
    const char *yaml = R"(
topology:
  type: tp
  name: mixed_gpu
  rank: 0
  backend: heterogeneous
  devices: [cuda:0, rocm:0]
)";

    auto result = ParallelismTreeParser::parseYAML(yaml, DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);
    ASSERT_TRUE(result.success()) << result.errorString();
    ASSERT_TRUE(result.tree.has_value());

    const auto &tree = *result.tree;
    EXPECT_EQ(tree.root.backend, CollectiveBackendType::HETEROGENEOUS);
    EXPECT_TRUE(tree.root.isMixedVendor());

    auto device_types = tree.root.leafDeviceTypes();
    EXPECT_EQ(device_types.count(DeviceType::CUDA), 1);
    EXPECT_EQ(device_types.count(DeviceType::ROCm), 1);
}

TEST(Test__ParallelismTreeParser, ParseYAML_ErrorMissingType)
{
    const char *yaml = R"(
topology:
  name: no_type
  devices: [cuda:0, cuda:1]
)";

    auto result = ParallelismTreeParser::parseYAML(yaml, DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);
    EXPECT_FALSE(result.success());
    EXPECT_FALSE(result.errors.empty());
    EXPECT_NE(result.errorString().find("type"), std::string::npos);
}

TEST(Test__ParallelismTreeParser, ParseYAML_ErrorInvalidType)
{
    const char *yaml = R"(
topology:
  type: unknown_type
  name: bad
)";

    auto result = ParallelismTreeParser::parseYAML(yaml, DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);
    EXPECT_FALSE(result.success());
    EXPECT_FALSE(result.errors.empty());
    EXPECT_NE(result.errorString().find("Invalid type"), std::string::npos);
}

TEST(Test__ParallelismTreeParser, ParseYAML_ErrorMissingDevices)
{
    const char *yaml = R"(
topology:
  type: tp
  name: empty_tp
  rank: 0
)";

    auto result = ParallelismTreeParser::parseYAML(yaml, DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);
    EXPECT_FALSE(result.success());
    EXPECT_FALSE(result.errors.empty());
}

TEST(Test__ParallelismTreeParser, ParseYAML_ErrorInvalidDevice)
{
    const char *yaml = R"(
topology:
  type: tp
  name: bad_tp
  rank: 0
  devices: [cuda:0, invalid:device, cuda:1]
)";

    auto result = ParallelismTreeParser::parseYAML(yaml, DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);
    EXPECT_FALSE(result.success());
    EXPECT_FALSE(result.errors.empty());
    EXPECT_NE(result.errorString().find("Invalid device"), std::string::npos);
}

// =============================================================================
// CLI Parsing Tests
// =============================================================================

TEST(Test__ParallelismTreeParser, ParseCLI_SimpleTP)
{
    auto result = ParallelismTreeParser::parseCLI(
        "TP(tp0, cuda:0, cuda:1)",
        DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);

    ASSERT_TRUE(result.success()) << result.errorString();
    ASSERT_TRUE(result.tree.has_value());

    const auto &tree = *result.tree;
    EXPECT_EQ(tree.root.type, ParallelismNodeType::TENSOR_PARALLEL);
    EXPECT_EQ(tree.root.name, "tp0");
    EXPECT_EQ(tree.root.children.size(), 2);
}

TEST(Test__ParallelismTreeParser, ParseCLI_SimplePP)
{
    auto result = ParallelismTreeParser::parseCLI(
        "PP(global, TP(s0, cuda:0, cuda:1), TP(s1, rocm:0, rocm:1))",
        DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);

    ASSERT_TRUE(result.success()) << result.errorString();
    ASSERT_TRUE(result.tree.has_value());

    const auto &tree = *result.tree;
    EXPECT_EQ(tree.root.type, ParallelismNodeType::PIPELINE_PARALLEL);
    EXPECT_EQ(tree.root.name, "global");
    EXPECT_EQ(tree.root.children.size(), 2);

    EXPECT_EQ(tree.root.children[0].name, "s0");
    EXPECT_EQ(tree.root.children[1].name, "s1");
}

TEST(Test__ParallelismTreeParser, ParseCLI_NestedPP)
{
    auto result = ParallelismTreeParser::parseCLI(
        "PP(global, PP(h0, TP(s0, cuda:0, cuda:1), TP(s1, cuda:2, cuda:3)), PP(h1, TP(s2, rocm:0, rocm:1)))",
        DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);

    ASSERT_TRUE(result.success()) << result.errorString();
    ASSERT_TRUE(result.tree.has_value());

    const auto &tree = *result.tree;
    EXPECT_EQ(tree.root.type, ParallelismNodeType::PIPELINE_PARALLEL);
    EXPECT_EQ(tree.root.children.size(), 2);
    EXPECT_EQ(tree.root.children[0].type, ParallelismNodeType::PIPELINE_PARALLEL);
    EXPECT_EQ(tree.root.children[0].children.size(), 2);
}

TEST(Test__ParallelismTreeParser, ParseCLI_WithOptions)
{
    auto result = ParallelismTreeParser::parseCLI(
        "TP(tp0, rank=2, backend=nccl, cuda:0, cuda:1)",
        DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);

    ASSERT_TRUE(result.success()) << result.errorString();
    ASSERT_TRUE(result.tree.has_value());

    const auto &tree = *result.tree;
    EXPECT_EQ(tree.root.owning_rank, 2);
    EXPECT_EQ(tree.root.backend, CollectiveBackendType::NCCL);
}

TEST(Test__ParallelismTreeParser, ParseCLI_ErrorUnmatchedParen)
{
    auto result = ParallelismTreeParser::parseCLI(
        "PP(global, TP(s0, cuda:0, cuda:1)",
        DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);

    EXPECT_FALSE(result.success());
    EXPECT_NE(result.errorString().find("Unmatched"), std::string::npos);
}

TEST(Test__ParallelismTreeParser, ParseCLI_ErrorEmptyTP)
{
    auto result = ParallelismTreeParser::parseCLI(
        "TP(empty)",
        DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);

    EXPECT_FALSE(result.success());
    // Should report TP needs at least 2 children
}

// =============================================================================
// Round-Trip Tests
// =============================================================================

TEST(Test__ParallelismTreeParser, ToYAMLRoundTrip_Simple)
{
    const char *original_yaml = R"(
topology:
  type: tp
  name: gpu_tp
  rank: 0
  devices: [cuda:0, cuda:1]
)";

    // Parse original
    auto result1 = ParallelismTreeParser::parseYAML(original_yaml, DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);
    ASSERT_TRUE(result1.success()) << result1.errorString();

    // Serialize to YAML
    std::string serialized = ParallelismTreeParser::toYAML(*result1.tree);

    // Parse serialized
    auto result2 = ParallelismTreeParser::parseYAML(serialized, DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);
    ASSERT_TRUE(result2.success()) << result2.errorString();

    // Compare trees (excluding layer assignments which may differ due to formatting)
    EXPECT_EQ(result1.tree->root.type, result2.tree->root.type);
    EXPECT_EQ(result1.tree->root.name, result2.tree->root.name);
    EXPECT_EQ(result1.tree->root.children.size(), result2.tree->root.children.size());
}

TEST(Test__ParallelismTreeParser, ToYAMLRoundTrip_Complex)
{
    const char *original_yaml = R"(
topology:
  type: pp
  name: global
  children:
    - type: pp
      name: host0
      children:
        - type: tp
          name: s0
          rank: 0
          backend: nccl
          devices: [cuda:0, cuda:1]
        - type: tp
          name: s1
          rank: 1
          devices: [cuda:0, cuda:1]
    - type: tp
      name: s2
      rank: 2
      backend: rccl
      devices: [rocm:0, rocm:1]
)";

    // Parse original
    auto result1 = ParallelismTreeParser::parseYAML(original_yaml, DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);
    ASSERT_TRUE(result1.success()) << result1.errorString();

    // Serialize to YAML
    std::string serialized = ParallelismTreeParser::toYAML(*result1.tree);

    // Parse serialized
    auto result2 = ParallelismTreeParser::parseYAML(serialized, DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);
    ASSERT_TRUE(result2.success()) << "Serialized YAML:\n"
                                   << serialized << "\nError: " << result2.errorString();

    // Compare structure
    EXPECT_EQ(result1.tree->root.type, result2.tree->root.type);
    EXPECT_EQ(result1.tree->root.children.size(), result2.tree->root.children.size());
    EXPECT_EQ(result1.tree->root.leafCount(), result2.tree->root.leafCount());
}

// =============================================================================
// File Parsing Tests
// =============================================================================

TEST(Test__ParallelismTreeParser, ParseFile_Success)
{
    // Create a temporary file
    std::string temp_path = std::filesystem::temp_directory_path() / "test_topology.yaml";

    {
        std::ofstream file(temp_path);
        file << R"(
topology:
  type: tp
  name: file_tp
  rank: 0
  devices: [cuda:0, cuda:1]
)";
    }

    auto result = ParallelismTreeParser::parseFile(temp_path, DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);
    ASSERT_TRUE(result.success()) << result.errorString();
    EXPECT_EQ(result.tree->root.name, "file_tp");

    // Cleanup
    std::filesystem::remove(temp_path);
}

TEST(Test__ParallelismTreeParser, ParseFile_FileNotFound)
{
    auto result = ParallelismTreeParser::parseFile(
        "/nonexistent/path/to/topology.yaml",
        DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);

    EXPECT_FALSE(result.success());
    EXPECT_NE(result.errorString().find("Failed to open"), std::string::npos);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(Test__ParallelismTreeParser, ParseYAML_EmptyContent)
{
    auto result = ParallelismTreeParser::parseYAML("", DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);
    EXPECT_FALSE(result.success());
}

TEST(Test__ParallelismTreeParser, ParseCLI_EmptyContent)
{
    auto result = ParallelismTreeParser::parseCLI("", DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);
    EXPECT_FALSE(result.success());
}

TEST(Test__ParallelismTreeParser, ParseYAML_MissingTopologyKey)
{
    const char *yaml = R"(
other_key:
  type: tp
  name: no_topology_key
)";

    auto result = ParallelismTreeParser::parseYAML(yaml, DEFAULT_TOTAL_LAYERS, DEFAULT_WORLD_SIZE);
    EXPECT_FALSE(result.success());
    EXPECT_NE(result.errorString().find("topology"), std::string::npos);
}
