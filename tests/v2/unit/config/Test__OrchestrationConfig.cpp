/**
 * @file Test__OrchestrationConfig.cpp
 * @brief Unit tests for OrchestrationConfig and related structures
 *
 * Tests:
 * - DomainDefinition parsing and validation
 * - PPStageDefinition parsing and validation
 * - OrchestrationConfig validation
 * - usesNamedDomains() detection
 * - Enum conversion functions
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "config/OrchestrationConfig.h"

using namespace llaminar2;

// ============================================================================
// Enum Conversion Tests
// ============================================================================

TEST(Test__OrchestrationConfig, TPScopeToString)
{
    EXPECT_STREQ(tpScopeToString(TPScope::AUTO), "auto");
    EXPECT_STREQ(tpScopeToString(TPScope::LOCAL), "local");
    EXPECT_STREQ(tpScopeToString(TPScope::GLOBAL), "global");
    EXPECT_STREQ(tpScopeToString(TPScope::HYBRID), "hybrid");
}

TEST(Test__OrchestrationConfig, ParseTpScope)
{
    EXPECT_EQ(parseTpScope("auto"), TPScope::AUTO);
    EXPECT_EQ(parseTpScope("AUTO"), TPScope::AUTO);
    EXPECT_EQ(parseTpScope("local"), TPScope::LOCAL);
    EXPECT_EQ(parseTpScope("global"), TPScope::GLOBAL);
    EXPECT_EQ(parseTpScope("hybrid"), TPScope::HYBRID);
    EXPECT_FALSE(parseTpScope("invalid").has_value());
}

TEST(Test__OrchestrationConfig, DeviceAssignmentModeToString)
{
    EXPECT_STREQ(deviceAssignmentModeToString(DeviceAssignmentMode::AUTO), "auto");
    EXPECT_STREQ(deviceAssignmentModeToString(DeviceAssignmentMode::LOCAL_GPU), "local_gpu");
    EXPECT_STREQ(deviceAssignmentModeToString(DeviceAssignmentMode::ROUND_ROBIN), "round_robin");
    EXPECT_STREQ(deviceAssignmentModeToString(DeviceAssignmentMode::EXPLICIT), "explicit");
}

TEST(Test__OrchestrationConfig, ParseDeviceAssignmentMode)
{
    EXPECT_EQ(parseDeviceAssignmentMode("auto"), DeviceAssignmentMode::AUTO);
    EXPECT_EQ(parseDeviceAssignmentMode("local_gpu"), DeviceAssignmentMode::LOCAL_GPU);
    EXPECT_EQ(parseDeviceAssignmentMode("local-gpu"), DeviceAssignmentMode::LOCAL_GPU);
    EXPECT_EQ(parseDeviceAssignmentMode("round_robin"), DeviceAssignmentMode::ROUND_ROBIN);
    EXPECT_EQ(parseDeviceAssignmentMode("round-robin"), DeviceAssignmentMode::ROUND_ROBIN);
    EXPECT_EQ(parseDeviceAssignmentMode("explicit"), DeviceAssignmentMode::EXPLICIT);
    EXPECT_FALSE(parseDeviceAssignmentMode("invalid").has_value());
}

TEST(Test__OrchestrationConfig, PPSplitModeToString)
{
    EXPECT_STREQ(ppSplitModeToString(PPSplitMode::EQUAL), "equal");
    EXPECT_STREQ(ppSplitModeToString(PPSplitMode::WEIGHTED), "weighted");
    EXPECT_STREQ(ppSplitModeToString(PPSplitMode::MANUAL), "manual");
}

TEST(Test__OrchestrationConfig, ParsePpSplitMode)
{
    EXPECT_EQ(parsePpSplitMode("equal"), PPSplitMode::EQUAL);
    EXPECT_EQ(parsePpSplitMode("EQUAL"), PPSplitMode::EQUAL);
    EXPECT_EQ(parsePpSplitMode("weighted"), PPSplitMode::WEIGHTED);
    EXPECT_EQ(parsePpSplitMode("manual"), PPSplitMode::MANUAL);
    EXPECT_FALSE(parsePpSplitMode("invalid").has_value());
}

TEST(Test__OrchestrationConfig, CollectiveBackendTypeToString)
{
    EXPECT_STREQ(collectiveBackendTypeToString(CollectiveBackendType::AUTO), "auto");
    EXPECT_STREQ(collectiveBackendTypeToString(CollectiveBackendType::NCCL), "nccl");
    EXPECT_STREQ(collectiveBackendTypeToString(CollectiveBackendType::RCCL), "rccl");
    EXPECT_STREQ(collectiveBackendTypeToString(CollectiveBackendType::UPI), "upi");
    EXPECT_STREQ(collectiveBackendTypeToString(CollectiveBackendType::MPI), "mpi");
    EXPECT_STREQ(collectiveBackendTypeToString(CollectiveBackendType::HOST), "host");
}

TEST(Test__OrchestrationConfig, ParseCollectiveBackendType)
{
    EXPECT_EQ(parseCollectiveBackendType("auto"), CollectiveBackendType::AUTO);
    EXPECT_EQ(parseCollectiveBackendType("nccl"), CollectiveBackendType::NCCL);
    EXPECT_EQ(parseCollectiveBackendType("NCCL"), CollectiveBackendType::NCCL);
    EXPECT_EQ(parseCollectiveBackendType("rccl"), CollectiveBackendType::RCCL);
    EXPECT_EQ(parseCollectiveBackendType("upi"), CollectiveBackendType::UPI);
    EXPECT_EQ(parseCollectiveBackendType("mpi"), CollectiveBackendType::MPI);
    EXPECT_EQ(parseCollectiveBackendType("host"), CollectiveBackendType::HOST);
    EXPECT_FALSE(parseCollectiveBackendType("invalid").has_value());
}

// ============================================================================
// DomainDefinition Tests
// ============================================================================

TEST(Test__DomainDefinition, Parse_SimpleDevices)
{
    auto def = DomainDefinition::parse("gpu_tp=cuda:0,cuda:1");

    EXPECT_EQ(def.name, "gpu_tp");
    EXPECT_EQ(def.devices.size(), 2);
    EXPECT_EQ(def.devices[0].device_type, DeviceType::CUDA);
    EXPECT_EQ(def.devices[0].device_ordinal, 0);
    EXPECT_EQ(def.devices[1].device_type, DeviceType::CUDA);
    EXPECT_EQ(def.devices[1].device_ordinal, 1);
    EXPECT_FALSE(def.hasWeights());
    EXPECT_EQ(def.backend, CollectiveBackendType::AUTO);
}

TEST(Test__DomainDefinition, Parse_WithWeights)
{
    auto def = DomainDefinition::parse("mixed=cuda:0,rocm:0;weights=0.73,0.27");

    EXPECT_EQ(def.name, "mixed");
    EXPECT_EQ(def.devices.size(), 2);
    EXPECT_TRUE(def.hasWeights());
    EXPECT_EQ(def.weights.size(), 2);
    EXPECT_FLOAT_EQ(def.weights[0], 0.73f);
    EXPECT_FLOAT_EQ(def.weights[1], 0.27f);
}

TEST(Test__DomainDefinition, Parse_WithBackend)
{
    auto def = DomainDefinition::parse("fast=cuda:0,cuda:1;backend=nccl");

    EXPECT_EQ(def.name, "fast");
    EXPECT_EQ(def.backend, CollectiveBackendType::NCCL);
}

TEST(Test__DomainDefinition, Parse_WithWeightsAndBackend)
{
    auto def = DomainDefinition::parse("hybrid=cuda:0,rocm:0;weights=0.6,0.4;backend=heterogeneous");

    EXPECT_EQ(def.name, "hybrid");
    EXPECT_EQ(def.devices.size(), 2);
    EXPECT_EQ(def.weights.size(), 2);
    EXPECT_FLOAT_EQ(def.weights[0], 0.6f);
    EXPECT_FLOAT_EQ(def.weights[1], 0.4f);
    EXPECT_EQ(def.backend, CollectiveBackendType::HETEROGENEOUS);
}

TEST(Test__DomainDefinition, Parse_SingleDevice)
{
    auto def = DomainDefinition::parse("single=cuda:0");

    EXPECT_EQ(def.name, "single");
    EXPECT_EQ(def.devices.size(), 1);
}

TEST(Test__DomainDefinition, TryParse_EmptyString_ReturnsNullopt)
{
    auto result = DomainDefinition::tryParse("");
    EXPECT_FALSE(result.has_value());
}

TEST(Test__DomainDefinition, TryParse_NoEquals_ReturnsNullopt)
{
    auto result = DomainDefinition::tryParse("nodequals");
    EXPECT_FALSE(result.has_value());
}

TEST(Test__DomainDefinition, TryParse_EmptyName_ReturnsNullopt)
{
    auto result = DomainDefinition::tryParse("=cuda:0");
    EXPECT_FALSE(result.has_value());
}

TEST(Test__DomainDefinition, TryParse_InvalidDevice_ReturnsNullopt)
{
    auto result = DomainDefinition::tryParse("domain=invalid:device");
    EXPECT_FALSE(result.has_value());
}

TEST(Test__DomainDefinition, Parse_InvalidFormat_Throws)
{
    EXPECT_THROW(DomainDefinition::parse("invalid"), std::invalid_argument);
}

TEST(Test__DomainDefinition, Validate_ValidDomain_ReturnsEmpty)
{
    auto def = DomainDefinition::parse("gpu_tp=cuda:0,cuda:1");
    auto errors = def.validate();
    EXPECT_TRUE(errors.empty());
}

TEST(Test__DomainDefinition, Validate_EmptyName_ReturnsError)
{
    DomainDefinition def;
    def.name = "";
    def.devices.push_back(GlobalDeviceAddress::cuda(0));

    auto errors = def.validate();
    EXPECT_FALSE(errors.empty());
    EXPECT_TRUE(errors[0].find("empty") != std::string::npos);
}

TEST(Test__DomainDefinition, Validate_NoDevices_ReturnsError)
{
    DomainDefinition def;
    def.name = "test";

    auto errors = def.validate();
    EXPECT_FALSE(errors.empty());
    EXPECT_TRUE(errors[0].find("no devices") != std::string::npos);
}

TEST(Test__DomainDefinition, Validate_WeightCountMismatch_ReturnsError)
{
    auto def = DomainDefinition::parse("test=cuda:0,cuda:1;weights=0.5");

    auto errors = def.validate();
    EXPECT_FALSE(errors.empty());
    EXPECT_TRUE(errors[0].find("2 devices") != std::string::npos);
}

TEST(Test__DomainDefinition, Validate_WeightsSumNot1_ReturnsError)
{
    DomainDefinition def;
    def.name = "test";
    def.devices.push_back(GlobalDeviceAddress::cuda(0));
    def.devices.push_back(GlobalDeviceAddress::cuda(1));
    def.weights = {0.3f, 0.3f}; // Sum = 0.6, not 1.0

    auto errors = def.validate();
    EXPECT_FALSE(errors.empty());
    EXPECT_TRUE(errors[0].find("sum") != std::string::npos);
}

TEST(Test__DomainDefinition, Validate_NegativeWeight_ReturnsError)
{
    DomainDefinition def;
    def.name = "test";
    def.devices.push_back(GlobalDeviceAddress::cuda(0));
    def.devices.push_back(GlobalDeviceAddress::cuda(1));
    def.weights = {1.5f, -0.5f}; // Invalid weights

    auto errors = def.validate();
    EXPECT_FALSE(errors.empty());
}

TEST(Test__DomainDefinition, ToString)
{
    auto def = DomainDefinition::parse("gpu_tp=cuda:0,cuda:1");
    std::string str = def.toString();

    EXPECT_TRUE(str.find("gpu_tp") != std::string::npos);
    EXPECT_TRUE(str.find("cuda:0") != std::string::npos);
}

// ============================================================================
// PPStageDefinition Tests
// ============================================================================

TEST(Test__PPStageDefinition, Parse_ValidFormat)
{
    auto def = PPStageDefinition::parse("0=gpu_tp:0-13");

    EXPECT_EQ(def.stage_id, 0);
    EXPECT_EQ(def.domain_name, "gpu_tp");
    EXPECT_EQ(def.first_layer, 0);
    EXPECT_EQ(def.last_layer, 13);
    EXPECT_EQ(def.layerCount(), 14);
}

TEST(Test__PPStageDefinition, Parse_Stage1)
{
    auto def = PPStageDefinition::parse("1=cpu_tp:14-27");

    EXPECT_EQ(def.stage_id, 1);
    EXPECT_EQ(def.domain_name, "cpu_tp");
    EXPECT_EQ(def.first_layer, 14);
    EXPECT_EQ(def.last_layer, 27);
}

TEST(Test__PPStageDefinition, Parse_SingleLayer)
{
    auto def = PPStageDefinition::parse("2=domain:5-5");

    EXPECT_EQ(def.stage_id, 2);
    EXPECT_EQ(def.first_layer, 5);
    EXPECT_EQ(def.last_layer, 5);
    EXPECT_EQ(def.layerCount(), 1);
}

TEST(Test__PPStageDefinition, TryParse_EmptyString_ReturnsNullopt)
{
    auto result = PPStageDefinition::tryParse("");
    EXPECT_FALSE(result.has_value());
}

TEST(Test__PPStageDefinition, TryParse_NoEquals_ReturnsNullopt)
{
    auto result = PPStageDefinition::tryParse("0domain:0-10");
    EXPECT_FALSE(result.has_value());
}

TEST(Test__PPStageDefinition, TryParse_NoColon_ReturnsNullopt)
{
    auto result = PPStageDefinition::tryParse("0=domain0-10");
    EXPECT_FALSE(result.has_value());
}

TEST(Test__PPStageDefinition, TryParse_NoDash_ReturnsNullopt)
{
    auto result = PPStageDefinition::tryParse("0=domain:010");
    EXPECT_FALSE(result.has_value());
}

TEST(Test__PPStageDefinition, TryParse_InvalidStageId_ReturnsNullopt)
{
    auto result = PPStageDefinition::tryParse("abc=domain:0-10");
    EXPECT_FALSE(result.has_value());
}

TEST(Test__PPStageDefinition, TryParse_InvalidLayerRange_ReturnsNullopt)
{
    auto result = PPStageDefinition::tryParse("0=domain:abc-10");
    EXPECT_FALSE(result.has_value());
}

TEST(Test__PPStageDefinition, Parse_InvalidFormat_Throws)
{
    EXPECT_THROW(PPStageDefinition::parse("invalid"), std::invalid_argument);
}

TEST(Test__PPStageDefinition, Validate_ValidDefinition_ReturnsEmpty)
{
    auto def = PPStageDefinition::parse("0=gpu_tp:0-13");
    auto errors = def.validate();
    EXPECT_TRUE(errors.empty());
}

TEST(Test__PPStageDefinition, Validate_NegativeStageId_ReturnsError)
{
    PPStageDefinition def;
    def.stage_id = -1;
    def.domain_name = "test";
    def.first_layer = 0;
    def.last_layer = 10;

    auto errors = def.validate();
    EXPECT_FALSE(errors.empty());
    EXPECT_TRUE(errors[0].find("invalid ID") != std::string::npos);
}

TEST(Test__PPStageDefinition, Validate_EmptyDomain_ReturnsError)
{
    PPStageDefinition def;
    def.stage_id = 0;
    def.domain_name = "";
    def.first_layer = 0;
    def.last_layer = 10;

    auto errors = def.validate();
    EXPECT_FALSE(errors.empty());
    EXPECT_TRUE(errors[0].find("empty domain") != std::string::npos);
}

TEST(Test__PPStageDefinition, Validate_NegativeFirstLayer_ReturnsError)
{
    PPStageDefinition def;
    def.stage_id = 0;
    def.domain_name = "test";
    def.first_layer = -1;
    def.last_layer = 10;

    auto errors = def.validate();
    EXPECT_FALSE(errors.empty());
}

TEST(Test__PPStageDefinition, Validate_LastLayerLessThanFirst_ReturnsError)
{
    PPStageDefinition def;
    def.stage_id = 0;
    def.domain_name = "test";
    def.first_layer = 10;
    def.last_layer = 5;

    auto errors = def.validate();
    EXPECT_FALSE(errors.empty());
    EXPECT_TRUE(errors[0].find("last_layer") != std::string::npos);
}

TEST(Test__PPStageDefinition, ToString)
{
    auto def = PPStageDefinition::parse("0=gpu_tp:0-13");
    std::string str = def.toString();

    EXPECT_TRUE(str.find("0") != std::string::npos);
    EXPECT_TRUE(str.find("gpu_tp") != std::string::npos);
    EXPECT_TRUE(str.find("0-13") != std::string::npos || str.find("[0-13]") != std::string::npos);
}

// ============================================================================
// OrchestrationConfig Tests
// ============================================================================

TEST(Test__OrchestrationConfig, Defaults)
{
    auto config = OrchestrationConfig::defaults();

    EXPECT_FALSE(config.dry_run);
    EXPECT_FALSE(config.explain_placement);
    EXPECT_EQ(config.device_mode, DeviceAssignmentMode::AUTO);
    EXPECT_EQ(config.tp_degree, 1);
    EXPECT_EQ(config.tp_scope, TPScope::AUTO);
    EXPECT_EQ(config.pp_degree, 1);
    EXPECT_EQ(config.pp_split, PPSplitMode::EQUAL);
    EXPECT_EQ(config.cpu_layers, 0);
    EXPECT_EQ(config.default_backend, CollectiveBackendType::AUTO);
}

TEST(Test__OrchestrationConfig, UsesNamedDomains_Empty_ReturnsFalse)
{
    OrchestrationConfig config;
    EXPECT_FALSE(config.usesNamedDomains());
}

TEST(Test__OrchestrationConfig, UsesNamedDomains_WithDomains_ReturnsTrue)
{
    OrchestrationConfig config;
    config.domain_definitions.push_back(DomainDefinition::parse("test=cuda:0"));

    EXPECT_TRUE(config.usesNamedDomains());
}

TEST(Test__OrchestrationConfig, UsesNamedDomains_WithPPStages_ReturnsTrue)
{
    OrchestrationConfig config;
    config.pp_stage_definitions.push_back(PPStageDefinition::parse("0=test:0-10"));

    EXPECT_TRUE(config.usesNamedDomains());
}

TEST(Test__OrchestrationConfig, Validate_DefaultConfig_ReturnsEmpty)
{
    auto config = OrchestrationConfig::defaults();
    auto errors = config.validate();
    EXPECT_TRUE(errors.empty());
}

TEST(Test__OrchestrationConfig, Validate_InvalidTPDegree_ReturnsError)
{
    OrchestrationConfig config;
    config.tp_degree = 0;

    auto errors = config.validate();
    EXPECT_FALSE(errors.empty());
    EXPECT_TRUE(errors[0].find("TP degree") != std::string::npos);
}

TEST(Test__OrchestrationConfig, Validate_InvalidPPDegree_ReturnsError)
{
    OrchestrationConfig config;
    config.pp_degree = -1;

    auto errors = config.validate();
    EXPECT_FALSE(errors.empty());
    EXPECT_TRUE(errors[0].find("PP degree") != std::string::npos);
}

TEST(Test__OrchestrationConfig, Validate_DuplicateDomainName_ReturnsError)
{
    OrchestrationConfig config;
    config.domain_definitions.push_back(DomainDefinition::parse("test=cuda:0"));
    config.domain_definitions.push_back(DomainDefinition::parse("test=cuda:1"));

    auto errors = config.validate();
    EXPECT_FALSE(errors.empty());
    EXPECT_TRUE(errors[0].find("Duplicate") != std::string::npos);
}

TEST(Test__OrchestrationConfig, Validate_DuplicatePPStageId_ReturnsError)
{
    OrchestrationConfig config;
    config.domain_definitions.push_back(DomainDefinition::parse("test1=cuda:0"));
    config.domain_definitions.push_back(DomainDefinition::parse("test2=cuda:1"));
    config.pp_stage_definitions.push_back(PPStageDefinition::parse("0=test1:0-5"));
    config.pp_stage_definitions.push_back(PPStageDefinition::parse("0=test2:6-10"));

    auto errors = config.validate();
    EXPECT_FALSE(errors.empty());
    // Find the duplicate error anywhere in the list (device selection rules may also fire)
    bool found_duplicate = false;
    for (const auto &e : errors)
    {
        if (e.find("Duplicate") != std::string::npos)
        {
            found_duplicate = true;
            break;
        }
    }
    EXPECT_TRUE(found_duplicate) << "Expected 'Duplicate' error for duplicate PP stage ID";
}

TEST(Test__OrchestrationConfig, Validate_PPStageReferencesUndefinedDomain_ReturnsError)
{
    OrchestrationConfig config;
    config.domain_definitions.push_back(DomainDefinition::parse("gpu_tp=cuda:0"));
    config.pp_stage_definitions.push_back(PPStageDefinition::parse("0=undefined:0-10"));

    auto errors = config.validate();
    EXPECT_FALSE(errors.empty());
    EXPECT_TRUE(errors[0].find("undefined") != std::string::npos);
}

TEST(Test__OrchestrationConfig, Validate_TPDevicesWeightsMismatch_ReturnsError)
{
    OrchestrationConfig config;
    config.tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    config.tp_weights = {0.5f}; // Only one weight for two devices

    auto errors = config.validate();
    EXPECT_FALSE(errors.empty());
    EXPECT_TRUE(errors[0].find("mismatch") != std::string::npos);
}

TEST(Test__OrchestrationConfig, Validate_TPWeightsSumNot1_ReturnsError)
{
    OrchestrationConfig config;
    config.tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    config.tp_weights = {0.3f, 0.3f}; // Sum = 0.6

    auto errors = config.validate();
    EXPECT_FALSE(errors.empty());
    EXPECT_TRUE(errors[0].find("sum") != std::string::npos);
}

TEST(Test__OrchestrationConfig, Validate_NegativeCPULayers_ReturnsError)
{
    OrchestrationConfig config;
    config.cpu_layers = -1;

    auto errors = config.validate();
    EXPECT_FALSE(errors.empty());
}

TEST(Test__OrchestrationConfig, Validate_DuplicateDeviceMapRank_ReturnsError)
{
    OrchestrationConfig config;
    config.device_map = {
        {0, GlobalDeviceAddress::cuda(0)},
        {0, GlobalDeviceAddress::cuda(1)} // Duplicate rank 0
    };

    auto errors = config.validate();
    EXPECT_FALSE(errors.empty());
    EXPECT_TRUE(errors[0].find("duplicate") != std::string::npos);
}

TEST(Test__OrchestrationConfig, Validate_NegativeDeviceMapRank_ReturnsError)
{
    OrchestrationConfig config;
    config.device_map = {
        {-1, GlobalDeviceAddress::cuda(0)}};

    auto errors = config.validate();
    EXPECT_FALSE(errors.empty());
    EXPECT_TRUE(errors[0].find("negative") != std::string::npos);
}

TEST(Test__OrchestrationConfig, ToString_ContainsRelevantInfo)
{
    OrchestrationConfig config;
    config.tp_degree = 2;
    config.pp_degree = 2;
    config.cpu_layers = 4;

    std::string str = config.toString();

    EXPECT_TRUE(str.find("tp_degree: 2") != std::string::npos);
    EXPECT_TRUE(str.find("pp_degree: 2") != std::string::npos);
    EXPECT_TRUE(str.find("cpu_layers: 4") != std::string::npos);
}
