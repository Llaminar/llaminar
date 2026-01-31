/**
 * @file Test__OrchestrationConfigParser.cpp
 * @brief Unit tests for OrchestrationConfigParser
 *
 * Tests:
 * - CLI parsing for simple options (--tp, --pp, --device)
 * - CLI parsing for --define-domain and --pp-stage
 * - YAML parsing for domain-based config
 * - Error handling for malformed input
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
    ArgvHelper args{"llaminar2", "--tp", "4"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.tp_degree, 4);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_TPDegree_WithEquals)
{
    ArgvHelper args{"llaminar2", "--tp=4"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.tp_degree, 4);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_TPDegree_ShortFlag)
{
    ArgvHelper args{"llaminar2", "-t", "2"};
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
    ArgvHelper args{"llaminar2", "--pp", "2"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.pp_degree, 2);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_PPDegree_ShortFlag)
{
    ArgvHelper args{"llaminar2", "-p", "3"};
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
                    "--define-domain", "mixed=cuda:0,rocm:0;weights=0.6,0.4;backend=pciebar"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.domain_definitions.size(), 1);
    auto &domain = config.domain_definitions[0];
    EXPECT_EQ(domain.name, "mixed");
    EXPECT_EQ(domain.weights.size(), 2);
    EXPECT_FLOAT_EQ(domain.weights[0], 0.6f);
    EXPECT_EQ(domain.backend, CollectiveBackendType::PCIE_BAR);
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
                    "--tp", "2",
                    "--pp", "2",
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
    ArgvHelper args{"llaminar2", "--tp"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_InvalidTPValue_Throws)
{
    ArgvHelper args{"llaminar2", "--tp", "abc"};
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

    EXPECT_TRUE(help.find("--tp") != std::string::npos);
    EXPECT_TRUE(help.find("--pp") != std::string::npos);
    EXPECT_TRUE(help.find("--device") != std::string::npos);
    EXPECT_TRUE(help.find("--define-domain") != std::string::npos);
    EXPECT_TRUE(help.find("--pp-stage") != std::string::npos);
    EXPECT_TRUE(help.find("--backend") != std::string::npos);
    EXPECT_TRUE(help.find("--config") != std::string::npos);
}
