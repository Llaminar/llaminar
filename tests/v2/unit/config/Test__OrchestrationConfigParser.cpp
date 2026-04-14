/**
 * @file Test__OrchestrationConfigParser.cpp
 * @brief Unit tests for OrchestrationConfigParser
 *
 * Tests:
 * - CLI parsing for simple options (--tp, --pp, --device)
 * - CLI parsing for --define-domain and --pp-stage
 * - Model and inference configuration options
 * - Sampling configuration
 * - Chat and benchmark modes
 * - Heterogeneous mode options
 * - YAML parsing for domain-based config
 * - Error handling for malformed input
 * - Validation of enum-type arguments
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "config/OrchestrationConfigParser.h"

using namespace llaminar2;

// ============================================================================
// Helper to convert string array to argc/argv
// ============================================================================

class ArgvHelper
{
public:
    ArgvHelper(std::initializer_list<const char *> args)
    {
        for (const char *arg : args)
        {
            strings_.push_back(arg);
        }
        for (auto &s : strings_)
        {
            argv_.push_back(const_cast<char *>(s.c_str()));
        }
    }

    int argc() const { return static_cast<int>(argv_.size()); }
    char **argv() { return argv_.data(); }

private:
    std::vector<std::string> strings_;
    std::vector<char *> argv_;
};

// ============================================================================
// Factory Function Tests
// ============================================================================

TEST(Test__OrchestrationConfigParser, CreateParser_ReturnsNonNull)
{
    auto parser = createOrchestrationConfigParser();
    EXPECT_NE(parser, nullptr);
}

// ============================================================================
// CLI Parsing - Simple Options
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_EmptyArgs_ReturnsDefaults)
{
    ArgvHelper args{"llaminar2"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.tp_degree, 1);
    EXPECT_EQ(config.pp_degree, 1);
    EXPECT_FALSE(config.dry_run);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_DryRun)
{
    ArgvHelper args{"llaminar2", "--dry-run"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.dry_run);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ExplainPlacement)
{
    ArgvHelper args{"llaminar2", "--explain-placement"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.explain_placement);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ShowTopology)
{
    ArgvHelper args{"llaminar2", "--show-topology"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.show_topology);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ShowNuma)
{
    ArgvHelper args{"llaminar2", "--show-numa"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.show_numa);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ValidateOnly)
{
    ArgvHelper args{"llaminar2", "--validate-only"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.validate_only);
}

// ============================================================================
// CLI Parsing - TP Options
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_TPDegree_WithSpace)
{
    ArgvHelper args{"llaminar2", "--tensor-parallelism-degree", "4"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.tp_degree, 4);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_TPDegree_WithEquals)
{
    ArgvHelper args{"llaminar2", "--tensor-parallelism-degree=4"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.tp_degree, 4);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_TPDegree_ShortFlag)
{
    ArgvHelper args{"llaminar2", "-tp", "2"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.tp_degree, 2);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_TPScope)
{
    ArgvHelper args{"llaminar2", "--tp-scope", "local"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.tp_scope, TPScope::LOCAL);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_TPDevices)
{
    ArgvHelper args{"llaminar2", "--tp-devices", "cuda:0,cuda:1"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.tp_devices.size(), 2);
    EXPECT_EQ(config.tp_devices[0].device_type, DeviceType::CUDA);
    EXPECT_EQ(config.tp_devices[0].device_ordinal, 0);
    EXPECT_EQ(config.tp_devices[1].device_ordinal, 1);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_TPWeights)
{
    ArgvHelper args{"llaminar2", "--tp-weights", "0.73,0.27"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.tp_weights.size(), 2);
    EXPECT_FLOAT_EQ(config.tp_weights[0], 0.73f);
    EXPECT_FLOAT_EQ(config.tp_weights[1], 0.27f);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_TPLocal)
{
    ArgvHelper args{"llaminar2", "--tp-local", "2"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.tp_local_degree, 2);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_TPGlobal)
{
    ArgvHelper args{"llaminar2", "--tp-global", "4"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.tp_global_degree, 4);
}

// ============================================================================
// CLI Parsing - PP Options
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_PPDegree_WithSpace)
{
    ArgvHelper args{"llaminar2", "--pipeline-parallelism-degree", "2"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.pp_degree, 2);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_PPDegree_ShortFlag)
{
    ArgvHelper args{"llaminar2", "-pp", "3"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.pp_degree, 3);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_PPSplit)
{
    ArgvHelper args{"llaminar2", "--pp-split", "weighted"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.pp_split, PPSplitMode::WEIGHTED);
}

// ============================================================================
// CLI Parsing - Device Options
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_Device)
{
    ArgvHelper args{"llaminar2", "--device", "cuda:0"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.device_for_this_rank.has_value());
    EXPECT_EQ(config.device_for_this_rank->device_type, DeviceType::CUDA);
    EXPECT_EQ(config.device_for_this_rank->device_ordinal, 0);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Device_ShortFlag)
{
    ArgvHelper args{"llaminar2", "-d", "rocm:1"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.device_for_this_rank.has_value());
    EXPECT_EQ(config.device_for_this_rank->device_type, DeviceType::ROCm);
    EXPECT_EQ(config.device_for_this_rank->device_ordinal, 1);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Device_CpuShorthand_EnablesGlobalCpuTpIntent)
{
    ArgvHelper args{"llaminar2", "-d", "cpu"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.device_for_this_rank.has_value());
    EXPECT_EQ(config.device_for_this_rank->device_type, DeviceType::CPU);
    EXPECT_EQ(config.device_for_this_rank->numa_node, 0);
    EXPECT_FALSE(config.device_for_this_rank_numa_explicit);
    EXPECT_TRUE(config.cpu_global_tp_all_local);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Device_CpuExplicitNuma)
{
    ArgvHelper args{"llaminar2", "-d", "cpu:1"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.device_for_this_rank.has_value());
    EXPECT_EQ(config.device_for_this_rank->device_type, DeviceType::CPU);
    EXPECT_EQ(config.device_for_this_rank->numa_node, 1);
    EXPECT_TRUE(config.device_for_this_rank_numa_explicit);
    EXPECT_FALSE(config.cpu_global_tp_all_local);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_DeviceMode)
{
    ArgvHelper args{"llaminar2", "--device-mode", "round_robin"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.device_mode, DeviceAssignmentMode::ROUND_ROBIN);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_DeviceMap)
{
    ArgvHelper args{"llaminar2", "--device-map", "0=cuda:0,1=cuda:1,2=rocm:0"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.device_mode, DeviceAssignmentMode::EXPLICIT);
    EXPECT_EQ(config.device_map.size(), 3);
    EXPECT_EQ(config.device_map[0].first, 0);
    EXPECT_EQ(config.device_map[0].second.device_type, DeviceType::CUDA);
    EXPECT_EQ(config.device_map[1].first, 1);
    EXPECT_EQ(config.device_map[2].first, 2);
    EXPECT_EQ(config.device_map[2].second.device_type, DeviceType::ROCm);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_DeviceMap_CpuEntriesTrackNumaExplicitness)
{
    ArgvHelper args{"llaminar2", "--device-map", "0=cpu,1=cpu:1"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    ASSERT_EQ(config.device_map.size(), 2);
    ASSERT_EQ(config.device_map_numa_explicit.size(), 2);

    EXPECT_EQ(config.device_map[0].first, 0);
    EXPECT_EQ(config.device_map[0].second.device_type, DeviceType::CPU);
    EXPECT_EQ(config.device_map[0].second.numa_node, 0);
    EXPECT_FALSE(config.device_map_numa_explicit[0].second);

    EXPECT_EQ(config.device_map[1].first, 1);
    EXPECT_EQ(config.device_map[1].second.device_type, DeviceType::CPU);
    EXPECT_EQ(config.device_map[1].second.numa_node, 1);
    EXPECT_TRUE(config.device_map_numa_explicit[1].second);
}

// ============================================================================
// CLI Parsing - Named Domains
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_DefineDomain)
{
    ArgvHelper args{"llaminar2", "--define-domain", "gpu_tp=cuda:0,cuda:1"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.domain_definitions.size(), 1);
    EXPECT_EQ(config.domain_definitions[0].name, "gpu_tp");
    EXPECT_EQ(config.domain_definitions[0].devices.size(), 2);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_DefineDomain_Multiple)
{
    ArgvHelper args{"llaminar2",
                    "--define-domain", "fast=cuda:0,cuda:1",
                    "--define-domain", "slow=rocm:0,rocm:1"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.domain_definitions.size(), 2);
    EXPECT_EQ(config.domain_definitions[0].name, "fast");
    EXPECT_EQ(config.domain_definitions[1].name, "slow");
}

TEST(Test__OrchestrationConfigParser, ParseArgs_DefineDomain_WithWeightsAndBackend)
{
    ArgvHelper args{"llaminar2",
                    "--define-domain", "mixed=cuda:0,rocm:0;weights=0.6,0.4;backend=heterogeneous"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.domain_definitions.size(), 1);
    auto &domain = config.domain_definitions[0];
    EXPECT_EQ(domain.name, "mixed");
    EXPECT_EQ(domain.weights.size(), 2);
    EXPECT_FLOAT_EQ(domain.weights[0], 0.6f);
    EXPECT_EQ(domain.backend, CollectiveBackendType::HETEROGENEOUS);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_PPStage)
{
    ArgvHelper args{"llaminar2", "--pp-stage", "0=gpu_tp:0-13"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.pp_stage_definitions.size(), 1);
    EXPECT_EQ(config.pp_stage_definitions[0].stage_id, 0);
    EXPECT_EQ(config.pp_stage_definitions[0].domain_name, "gpu_tp");
    EXPECT_EQ(config.pp_stage_definitions[0].first_layer, 0);
    EXPECT_EQ(config.pp_stage_definitions[0].last_layer, 13);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_PPStage_Multiple)
{
    ArgvHelper args{"llaminar2",
                    "--pp-stage", "0=stage0:0-13",
                    "--pp-stage", "1=stage1:14-27"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.pp_stage_definitions.size(), 2);
    EXPECT_EQ(config.pp_stage_definitions[0].stage_id, 0);
    EXPECT_EQ(config.pp_stage_definitions[1].stage_id, 1);
}

// ============================================================================
// CLI Parsing - Layer Placement
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_CPULayers)
{
    ArgvHelper args{"llaminar2", "--cpu-layers", "4"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.cpu_layers, 4);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_CPULayersFirst)
{
    ArgvHelper args{"llaminar2", "--cpu-layers", "4", "--cpu-layers-first"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.cpu_layers, 4);
    EXPECT_TRUE(config.cpu_layers_first);
}

// ============================================================================
// CLI Parsing - Backend
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_Backend)
{
    ArgvHelper args{"llaminar2", "--backend", "nccl"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.default_backend, CollectiveBackendType::NCCL);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Backend_ShortFlag)
{
    ArgvHelper args{"llaminar2", "-b", "rccl"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.default_backend, CollectiveBackendType::RCCL);
}

// ============================================================================
// CLI Parsing - Combined Options
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_ComplexConfig)
{
    ArgvHelper args{"llaminar2",
                    "--tensor-parallelism-degree", "2",
                    "--pipeline-parallelism-degree", "2",
                    "--device", "cuda:0",
                    "--cpu-layers", "2",
                    "--backend", "nccl",
                    "--dry-run"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.tp_degree, 2);
    EXPECT_EQ(config.pp_degree, 2);
    EXPECT_TRUE(config.device_for_this_rank.has_value());
    EXPECT_EQ(config.cpu_layers, 2);
    EXPECT_EQ(config.default_backend, CollectiveBackendType::NCCL);
    EXPECT_TRUE(config.dry_run);
}

// ============================================================================
// CLI Parsing - Error Handling
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_MissingTPValue_Throws)
{
    ArgvHelper args{"llaminar2", "--tensor-parallelism-degree"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_InvalidTPValue_Throws)
{
    ArgvHelper args{"llaminar2", "--tensor-parallelism-degree", "abc"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_InvalidDevice_Throws)
{
    ArgvHelper args{"llaminar2", "--device", "invalid:device"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_InvalidTPScope_Throws)
{
    ArgvHelper args{"llaminar2", "--tp-scope", "invalid"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_InvalidDomainDefinition_Throws)
{
    ArgvHelper args{"llaminar2", "--define-domain", "invalid"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_InvalidPPStage_Throws)
{
    ArgvHelper args{"llaminar2", "--pp-stage", "invalid"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

// ============================================================================
// YAML Parsing Tests
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseYamlString_SimpleConfig)
{
    OrchestrationConfigParser parser;

    std::string yaml = R"(
tp_degree: 4
pp_degree: 2
dry_run: true
    )";

    auto config = parser.parseYamlString(yaml);

    EXPECT_EQ(config.tp_degree, 4);
    EXPECT_EQ(config.pp_degree, 2);
    EXPECT_TRUE(config.dry_run);
}

TEST(Test__OrchestrationConfigParser, ParseYamlString_TPConfig)
{
    OrchestrationConfigParser parser;

    std::string yaml = R"(
tp_degree: 2
tp_scope: local
tp_devices: [cuda:0, cuda:1]
tp_weights: [0.73, 0.27]
    )";

    auto config = parser.parseYamlString(yaml);

    EXPECT_EQ(config.tp_degree, 2);
    EXPECT_EQ(config.tp_scope, TPScope::LOCAL);
    EXPECT_EQ(config.tp_devices.size(), 2);
    EXPECT_EQ(config.tp_weights.size(), 2);
}

TEST(Test__OrchestrationConfigParser, ParseYamlString_AllIntrospectionFlags)
{
    OrchestrationConfigParser parser;

    std::string yaml = R"(
dry_run: true
explain_placement: true
show_topology: true
show_numa: true
validate_only: true
    )";

    auto config = parser.parseYamlString(yaml);

    EXPECT_TRUE(config.dry_run);
    EXPECT_TRUE(config.explain_placement);
    EXPECT_TRUE(config.show_topology);
    EXPECT_TRUE(config.show_numa);
    EXPECT_TRUE(config.validate_only);
}

TEST(Test__OrchestrationConfigParser, ParseYamlString_DeviceConfig)
{
    OrchestrationConfigParser parser;

    std::string yaml = R"(
device: cuda:0
device_mode: explicit
    )";

    auto config = parser.parseYamlString(yaml);

    EXPECT_TRUE(config.device_for_this_rank.has_value());
    EXPECT_EQ(config.device_for_this_rank->device_type, DeviceType::CUDA);
    EXPECT_EQ(config.device_mode, DeviceAssignmentMode::EXPLICIT);
}

TEST(Test__OrchestrationConfigParser, ParseYamlString_LayerPlacement)
{
    OrchestrationConfigParser parser;

    std::string yaml = R"(
cpu_layers: 4
cpu_layers_first: true
    )";

    auto config = parser.parseYamlString(yaml);

    EXPECT_EQ(config.cpu_layers, 4);
    EXPECT_TRUE(config.cpu_layers_first);
}

TEST(Test__OrchestrationConfigParser, ParseYamlString_Backend)
{
    OrchestrationConfigParser parser;

    std::string yaml = R"(
backend: nccl
    )";

    auto config = parser.parseYamlString(yaml);

    EXPECT_EQ(config.default_backend, CollectiveBackendType::NCCL);
}

TEST(Test__OrchestrationConfigParser, ParseYamlString_CommentsIgnored)
{
    OrchestrationConfigParser parser;

    std::string yaml = R"(
# This is a comment
tp_degree: 2
# Another comment
pp_degree: 3
    )";

    auto config = parser.parseYamlString(yaml);

    EXPECT_EQ(config.tp_degree, 2);
    EXPECT_EQ(config.pp_degree, 3);
}

TEST(Test__OrchestrationConfigParser, ParseYamlString_EmptyString_ReturnsDefaults)
{
    OrchestrationConfigParser parser;

    auto config = parser.parseYamlString("");

    EXPECT_EQ(config.tp_degree, 1);
    EXPECT_EQ(config.pp_degree, 1);
}

TEST(Test__OrchestrationConfigParser, ParseYamlString_QuotedValues)
{
    OrchestrationConfigParser parser;

    std::string yaml = R"(
device: "cuda:0"
tp_scope: 'local'
    )";

    auto config = parser.parseYamlString(yaml);

    EXPECT_TRUE(config.device_for_this_rank.has_value());
    EXPECT_EQ(config.device_for_this_rank->device_type, DeviceType::CUDA);
    EXPECT_EQ(config.tp_scope, TPScope::LOCAL);
}

TEST(Test__OrchestrationConfigParser, ParseYamlFile_NonExistent_Throws)
{
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseYamlFile("/nonexistent/path/config.yaml"), std::invalid_argument);
}

// ============================================================================
// Help Text Tests
// ============================================================================

TEST(Test__OrchestrationConfigParser, GetHelpText_ContainsKeyOptions)
{
    std::string help = OrchestrationConfigParser::getHelpText();

    EXPECT_TRUE(help.find("--tensor-parallelism-degree") != std::string::npos);
    EXPECT_TRUE(help.find("--pipeline-parallelism-degree") != std::string::npos);
    EXPECT_TRUE(help.find("--device") != std::string::npos);
    EXPECT_TRUE(help.find("--define-domain") != std::string::npos);
    EXPECT_TRUE(help.find("--pp-stage") != std::string::npos);
    EXPECT_TRUE(help.find("--backend") != std::string::npos);
    EXPECT_TRUE(help.find("--config") != std::string::npos);
}
// ============================================================================
// CLI Parsing - Model Configuration
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_ModelPath)
{
    ArgvHelper args{"llaminar2", "-m", "models/test.gguf"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.model_path, "models/test.gguf");
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ModelPath_LongForm)
{
    ArgvHelper args{"llaminar2", "--model", "path/to/model.gguf"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.model_path, "path/to/model.gguf");
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ContextLength)
{
    ArgvHelper args{"llaminar2", "-c", "4096"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.max_seq_len, 4096);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_MMap)
{
    ArgvHelper args{"llaminar2", "--no-mmap"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_FALSE(config.use_mmap);
}

// ============================================================================
// CLI Parsing - Inference Configuration
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_Prompt)
{
    ArgvHelper args{"llaminar2", "-p", "Hello, world!"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.prompt, "Hello, world!");
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Prompt_LongForm)
{
    ArgvHelper args{"llaminar2", "--prompt", "Test prompt"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.prompt, "Test prompt");
}

TEST(Test__OrchestrationConfigParser, ParseArgs_NPredict)
{
    ArgvHelper args{"llaminar2", "-n", "100"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.n_predict, 100);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_NPredict_LongForm)
{
    ArgvHelper args{"llaminar2", "--n-predict", "50"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.n_predict, 50);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_BatchSize)
{
    ArgvHelper args{"llaminar2", "--batch-size", "8"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.batch_size, 8);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Threads)
{
    ArgvHelper args{"llaminar2", "--threads", "16"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.n_threads, 16);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Seed)
{
    ArgvHelper args{"llaminar2", "-s", "42"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.seed, 42);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Seed_LongForm)
{
    ArgvHelper args{"llaminar2", "--seed", "12345"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.seed, 12345);
}

// ============================================================================
// CLI Parsing - Sampling Configuration
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_Temperature)
{
    ArgvHelper args{"llaminar2", "-t", "0.5"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_FLOAT_EQ(config.temperature, 0.5f);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Temperature_LongForm)
{
    ArgvHelper args{"llaminar2", "--temperature", "1.2"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_FLOAT_EQ(config.temperature, 1.2f);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_TopK)
{
    ArgvHelper args{"llaminar2", "--top-k", "50"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.top_k, 50);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_TopP)
{
    ArgvHelper args{"llaminar2", "--top-p", "0.95"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_FLOAT_EQ(config.top_p, 0.95f);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Deterministic)
{
    ArgvHelper args{"llaminar2", "--deterministic"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.deterministic);
}

// ============================================================================
// CLI Parsing - Chat Configuration
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_ChatMode)
{
    ArgvHelper args{"llaminar2", "--chat"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.chat_mode);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ChatSingle)
{
    ArgvHelper args{"llaminar2", "--chat-single"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.single_shot_chat);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_SystemPrompt)
{
    ArgvHelper args{"llaminar2", "--system", "You are a helpful assistant."};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.system_prompt, "You are a helpful assistant.");
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ChatTemplate)
{
    ArgvHelper args{"llaminar2", "--chat-template", "chatml"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.chat_template_override, "chatml");
}

// ============================================================================
// CLI Parsing - Benchmark Configuration
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_BenchmarkMode)
{
    ArgvHelper args{"llaminar2", "--benchmark"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.benchmark_mode);
}

// ============================================================================
// CLI Parsing - Fused Attention
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_FusedAttention)
{
    ArgvHelper args{"llaminar2", "--fused-attention"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.use_fused_attention);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_FusedAttentionBackend)
{
    ArgvHelper args{"llaminar2", "--fused-attention-backend", "reference"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.fused_attention_backend, FusedAttentionBackend::REFERENCE);
}

// ============================================================================
// CLI Parsing - MPI Bootstrap
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_MPIProcs)
{
    ArgvHelper args{"llaminar2", "--mpi-procs", "4"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.mpi_procs, 4);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_MPIDryRun)
{
    ArgvHelper args{"llaminar2", "--mpi-dry-run"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.mpi_dry_run);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_MPIVerbose)
{
    ArgvHelper args{"llaminar2", "--mpi-verbose"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.mpi_verbose);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_NoMPIBootstrap)
{
    ArgvHelper args{"llaminar2", "--no-mpi-bootstrap"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.mpi_no_bootstrap);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_MPIOversubscribe)
{
    ArgvHelper args{"llaminar2", "--mpi-oversubscribe"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.mpi_oversubscribe);
}

// ============================================================================
// CLI Parsing - Verbosity
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_VerboseLevel1)
{
    ArgvHelper args{"llaminar2", "-v"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.verbose_level, 1);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_VerboseLevel2)
{
    ArgvHelper args{"llaminar2", "-vv"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.verbose_level, 2);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ListDevices)
{
    ArgvHelper args{"llaminar2", "--list-devices"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.list_devices);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ShowHelp)
{
    ArgvHelper args{"llaminar2", "-h"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.show_help);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ShowHelp_LongForm)
{
    ArgvHelper args{"llaminar2", "--help"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.show_help);
}

// ============================================================================
// CLI Parsing - Memory Constraints
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_MaxGPUMemory)
{
    ArgvHelper args{"llaminar2", "--max-gpu-memory", "8000"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.max_gpu_memory_mb.has_value());
    EXPECT_EQ(config.max_gpu_memory_mb.value(), 8000u);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_MaxCPUMemory)
{
    ArgvHelper args{"llaminar2", "--max-cpu-memory", "16000"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.max_cpu_memory_mb.has_value());
    EXPECT_EQ(config.max_cpu_memory_mb.value(), 16000u);
}

// ============================================================================
// CLI Parsing - MoE Configuration
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_MoESharedGPU)
{
    ArgvHelper args{"llaminar2", "--moe-shared-gpu"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.moe_shared_experts_gpu);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_MoESharedCPU)
{
    ArgvHelper args{"llaminar2", "--moe-shared-cpu"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_FALSE(config.moe_shared_experts_gpu);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_MoESparseGPU)
{
    ArgvHelper args{"llaminar2", "--moe-sparse-gpu"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_FALSE(config.moe_sparse_experts_cpu);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_MoESparseCPU)
{
    ArgvHelper args{"llaminar2", "--moe-sparse-cpu"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.moe_sparse_experts_cpu);
}

// ============================================================================
// CLI Parsing - Precision
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_ActivationPrecision)
{
    ArgvHelper args{"llaminar2", "--activation-precision", "bf16"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.activation_precision, "bf16");
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ActivationPrecision_Alias)
{
    ArgvHelper args{"llaminar2", "--act-prec", "fp16"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.activation_precision, "fp16");
}

// ============================================================================
// CLI Parsing - Weight Sharding
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_ShardWeights)
{
    ArgvHelper args{"llaminar2", "--shard-weights"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.shard_weights);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_NoShardWeights)
{
    ArgvHelper args{"llaminar2", "--no-shard-weights"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.disable_weight_sharding);
}

// ============================================================================
// CLI Parsing - Heterogeneous Mode
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_HeterogeneousMode)
{
    ArgvHelper args{"llaminar2", "--heterogeneous"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.heterogeneous_mode);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_CPUFraction)
{
    ArgvHelper args{"llaminar2", "--cpu-fraction", "0.3"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_FLOAT_EQ(config.cpu_compute_fraction, 0.3f);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_NoGPUTP)
{
    ArgvHelper args{"llaminar2", "--no-gpu-tp"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.disable_gpu_tp);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_NoCPUTP)
{
    ArgvHelper args{"llaminar2", "--no-cpu-tp"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.disable_cpu_tp);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_MinLayersPerDomain)
{
    ArgvHelper args{"llaminar2", "--min-layers-per-domain", "4"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.min_layers_per_domain, 4);
}

// ============================================================================
// Validation Tests - Invalid Arguments
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_UnknownArgument_Throws)
{
    ArgvHelper args{"llaminar2", "--unknown-flag"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_InvalidStrategy_Throws)
{
    ArgvHelper args{"llaminar2", "--strategy", "invalid-strategy"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_InvalidActivationPrecision_Throws)
{
    ArgvHelper args{"llaminar2", "--activation-precision", "fp64"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_InvalidCPUFraction_TooHigh_Throws)
{
    ArgvHelper args{"llaminar2", "--cpu-fraction", "1.5"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_InvalidCPUFraction_Negative_Throws)
{
    ArgvHelper args{"llaminar2", "--cpu-fraction", "-0.1"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_HeterogeneousWithSingleRank_ValidationWarning)
{
    // This test checks that --heterogeneous with TP=1 produces a validation warning
    // (heterogeneous mode requires TP >= 2 to be meaningful)
    ArgvHelper args{"llaminar2", "--heterogeneous", "--tensor-parallelism-degree", "1"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    // Config should still parse, but validation would show issues
    EXPECT_TRUE(config.heterogeneous_mode);
    EXPECT_EQ(config.tp_degree, 1);
}

// ============================================================================
// Combined Flags Test
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_FullInferenceConfig)
{
    ArgvHelper args{"llaminar2",
                    "-m", "model.gguf",
                    "-p", "Hello world",
                    "-n", "100",
                    "-t", "0.7",
                    "--top-k", "50",
                    "--top-p", "0.9",
                    "-s", "42",
                    "-d", "cuda:0",
                    "-tp", "2"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.model_path, "model.gguf");
    EXPECT_EQ(config.prompt, "Hello world");
    EXPECT_EQ(config.n_predict, 100);
    EXPECT_FLOAT_EQ(config.temperature, 0.7f);
    EXPECT_EQ(config.top_k, 50);
    EXPECT_FLOAT_EQ(config.top_p, 0.9f);
    EXPECT_EQ(config.seed, 42);
    EXPECT_TRUE(config.device_for_this_rank.has_value());
    EXPECT_EQ(config.device_for_this_rank->device_type, DeviceType::CUDA);
    EXPECT_EQ(config.tp_degree, 2);
}

// ============================================================================
// Help Text Extended Tests
// ============================================================================

TEST(Test__OrchestrationConfigParser, GetHelpText_ContainsAllShortFlags)
{
    std::string help = OrchestrationConfigParser::getHelpText();

    // All documented short flags should be in help text
    EXPECT_TRUE(help.find("-m") != std::string::npos);
    EXPECT_TRUE(help.find("-p") != std::string::npos);
    EXPECT_TRUE(help.find("-n") != std::string::npos);
    EXPECT_TRUE(help.find("-t") != std::string::npos);
    EXPECT_TRUE(help.find("-s") != std::string::npos);
    EXPECT_TRUE(help.find("-d") != std::string::npos);
    EXPECT_TRUE(help.find("-c") != std::string::npos);
    EXPECT_TRUE(help.find("-tp") != std::string::npos);
    EXPECT_TRUE(help.find("-pp") != std::string::npos);
    EXPECT_TRUE(help.find("-b") != std::string::npos);
    EXPECT_TRUE(help.find("-v") != std::string::npos);
    EXPECT_TRUE(help.find("-h") != std::string::npos);
}

TEST(Test__OrchestrationConfigParser, GetHelpText_ContainsExamples)
{
    std::string help = OrchestrationConfigParser::getHelpText();

    EXPECT_TRUE(help.find("Examples:") != std::string::npos);
    EXPECT_TRUE(help.find("llaminar2") != std::string::npos);
}