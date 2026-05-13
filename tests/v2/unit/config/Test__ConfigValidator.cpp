/**
 * @file Test__ConfigValidator.cpp
 * @brief Unit tests for the declarative ConfigValidator framework
 *
 * Tests:
 * - DeviceSelectionMode detection
 * - Cross-mode mutual exclusion rules
 * - Co-requirement rules
 * - Intra-mode consistency rules
 * - Rule infrastructure (findRule, ruleCount, error formatting)
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include "config/ConfigValidator.h"
#include "config/OrchestrationConfig.h"
#include "backends/GlobalDeviceAddress.h"

using namespace llaminar2;

// ============================================================================
// Helper: check that a specific rule fires for a given config
// ============================================================================

namespace
{
    /// Create a default config for testing (everything at defaults = no conflicts)
    OrchestrationConfig makeClean()
    {
        return OrchestrationConfig::defaults();
    }

    /// Check that a specific rule ID fires (produces an error)
    bool ruleFiresFor(const ConfigValidator &v, const std::string &rule_id, const OrchestrationConfig &cfg)
    {
        auto errors = v.validate(cfg);
        for (const auto &e : errors)
        {
            if (e.rule_id == rule_id)
                return true;
        }
        return false;
    }

    /// Check that NO rules fire for a given config
    bool noErrors(const ConfigValidator &v, const OrchestrationConfig &cfg)
    {
        return v.validate(cfg).empty();
    }

} // anonymous namespace

// ============================================================================
// DeviceSelectionMode Detection Tests
// ============================================================================

TEST(Test__ConfigValidator, DetectMode_Unspecified)
{
    auto cfg = makeClean();
    EXPECT_EQ(detectDeviceSelectionMode(cfg), DeviceSelectionMode::UNSPECIFIED);
}

TEST(Test__ConfigValidator, DetectMode_SingleDevice)
{
    auto cfg = makeClean();
    cfg.device_for_this_rank = GlobalDeviceAddress::cuda(0);
    EXPECT_EQ(detectDeviceSelectionMode(cfg), DeviceSelectionMode::SINGLE_DEVICE);
}

TEST(Test__ConfigValidator, DetectMode_DeviceMap)
{
    auto cfg = makeClean();
    cfg.device_map = {{0, GlobalDeviceAddress::cuda(0)}, {1, GlobalDeviceAddress::cuda(1)}};
    EXPECT_EQ(detectDeviceSelectionMode(cfg), DeviceSelectionMode::DEVICE_MAP);
}

TEST(Test__ConfigValidator, DetectMode_SimpleTP)
{
    auto cfg = makeClean();
    cfg.tp_degree = 2;
    // No tp_devices → auto-pick
    EXPECT_EQ(detectDeviceSelectionMode(cfg), DeviceSelectionMode::SIMPLE_TP);
}

TEST(Test__ConfigValidator, DetectMode_ExplicitTP)
{
    auto cfg = makeClean();
    cfg.tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    EXPECT_EQ(detectDeviceSelectionMode(cfg), DeviceSelectionMode::EXPLICIT_TP);
}

TEST(Test__ConfigValidator, DetectMode_NamedDomains)
{
    auto cfg = makeClean();
    cfg.domain_definitions.push_back(DomainDefinition::parse("gpu_tp=cuda:0,cuda:1"));
    EXPECT_EQ(detectDeviceSelectionMode(cfg), DeviceSelectionMode::NAMED_DOMAINS);
}

TEST(Test__ConfigValidator, DetectMode_TopologyTree)
{
    auto cfg = makeClean();
    cfg.topology_string = "PP(TP(cuda:0,cuda:1), TP(cuda:2,cuda:3))";
    EXPECT_EQ(detectDeviceSelectionMode(cfg), DeviceSelectionMode::TOPOLOGY_TREE);
}

TEST(Test__ConfigValidator, DetectMode_TopologyTreePriority)
{
    // When topology + other modes are set, topology wins (detection only, validator flags the conflict)
    auto cfg = makeClean();
    cfg.topology_string = "PP(TP(cuda:0))";
    cfg.device_for_this_rank = GlobalDeviceAddress::cuda(0);
    cfg.tp_devices = {GlobalDeviceAddress::cuda(0)};
    EXPECT_EQ(detectDeviceSelectionMode(cfg), DeviceSelectionMode::TOPOLOGY_TREE);
}

TEST(Test__ConfigValidator, DetectMode_ToString)
{
    EXPECT_STREQ(deviceSelectionModeToString(DeviceSelectionMode::UNSPECIFIED), "unspecified (auto-detect)");
    EXPECT_STREQ(deviceSelectionModeToString(DeviceSelectionMode::SINGLE_DEVICE), "single-device (-d)");
    EXPECT_STREQ(deviceSelectionModeToString(DeviceSelectionMode::EXPLICIT_TP), "explicit-tp (--tp-devices)");
    EXPECT_STREQ(deviceSelectionModeToString(DeviceSelectionMode::NAMED_DOMAINS), "named-domains (--define-domain)");
    EXPECT_STREQ(deviceSelectionModeToString(DeviceSelectionMode::TOPOLOGY_TREE), "topology-tree (--topology)");
}

// ============================================================================
// Infrastructure Tests
// ============================================================================

TEST(Test__ConfigValidator, DefaultConfig_NoErrors)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    EXPECT_TRUE(noErrors(v, cfg));
}

TEST(Test__ConfigValidator, StandardValidator_HasRules)
{
    auto v = ConfigValidator::createStandard();
    EXPECT_GT(v.ruleCount(), 10u);
}

TEST(Test__ConfigValidator, FindRule_Exists)
{
    auto v = ConfigValidator::createStandard();
    EXPECT_NE(v.findRule("device-tp-devices-mutex"), nullptr);
    EXPECT_NE(v.findRule("tp-weights-requires-tp-devices"), nullptr);
    EXPECT_NE(v.findRule("tp-devices-no-duplicates"), nullptr);
}

TEST(Test__ConfigValidator, FindRule_NotExists)
{
    auto v = ConfigValidator::createStandard();
    EXPECT_EQ(v.findRule("nonexistent-rule"), nullptr);
}

TEST(Test__ConfigValidator, ErrorFormatting_ToString)
{
    ConfigValidationError err{
        .rule_id = "test-rule",
        .message = "Something went wrong",
        .fix_hint = "Try something else",
    };

    std::string s = err.toString();
    EXPECT_TRUE(s.find("test-rule") != std::string::npos);
    EXPECT_TRUE(s.find("Something went wrong") != std::string::npos);
    EXPECT_TRUE(s.find("Try something else") != std::string::npos);
}

TEST(Test__ConfigValidator, ErrorFormatting_ToDetailedString)
{
    ConfigValidationError err{
        .rule_id = "test-rule",
        .message = "Bad config",
        .fix_hint = "Fix it",
    };

    std::string s = err.toDetailedString();
    EXPECT_TRUE(s.find("Rule:") != std::string::npos);
    EXPECT_TRUE(s.find("Error:") != std::string::npos);
    EXPECT_TRUE(s.find("Fix:") != std::string::npos);
}

TEST(Test__ConfigValidator, ValidateToStrings)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.device_for_this_rank = GlobalDeviceAddress::cuda(0);
    cfg.tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};

    auto errors = v.validateToStrings(cfg);
    EXPECT_FALSE(errors.empty());
    // Each string should contain the rule ID
    EXPECT_TRUE(errors[0].find("device-tp-devices-mutex") != std::string::npos);
}

// ============================================================================
// Cross-Mode Mutual Exclusion Tests
// ============================================================================

TEST(Test__ConfigValidator, MutualExclusion_Device_TPDevices)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.device_for_this_rank = GlobalDeviceAddress::cuda(0);
    cfg.tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};

    EXPECT_TRUE(ruleFiresFor(v, "device-tp-devices-mutex", cfg));
}

TEST(Test__ConfigValidator, MutualExclusion_Device_NamedDomains)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.device_for_this_rank = GlobalDeviceAddress::cuda(0);
    cfg.domain_definitions.push_back(DomainDefinition::parse("gpu=cuda:0,cuda:1"));

    EXPECT_TRUE(ruleFiresFor(v, "device-named-domains-mutex", cfg));
}

TEST(Test__ConfigValidator, MutualExclusion_Device_Topology)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.device_for_this_rank = GlobalDeviceAddress::cuda(0);
    cfg.topology_string = "PP(TP(cuda:0))";

    EXPECT_TRUE(ruleFiresFor(v, "device-topology-mutex", cfg));
}

TEST(Test__ConfigValidator, MutualExclusion_TPDevices_NamedDomains)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    cfg.domain_definitions.push_back(DomainDefinition::parse("gpu=cuda:0,cuda:1"));

    EXPECT_TRUE(ruleFiresFor(v, "tp-devices-named-domains-mutex", cfg));
}

TEST(Test__ConfigValidator, MutualExclusion_TPDevices_Topology)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    cfg.topology_string = "PP(TP(cuda:0))";

    EXPECT_TRUE(ruleFiresFor(v, "tp-devices-topology-mutex", cfg));
}

TEST(Test__ConfigValidator, MutualExclusion_NamedDomains_Topology)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.domain_definitions.push_back(DomainDefinition::parse("gpu=cuda:0,cuda:1"));
    cfg.topology_string = "PP(TP(cuda:0))";

    EXPECT_TRUE(ruleFiresFor(v, "named-domains-topology-mutex", cfg));
}

TEST(Test__ConfigValidator, MutualExclusion_DeviceMap_TPDevices)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.device_map = {{0, GlobalDeviceAddress::cuda(0)}};
    cfg.tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};

    EXPECT_TRUE(ruleFiresFor(v, "device-map-tp-devices-mutex", cfg));
}

TEST(Test__ConfigValidator, MutualExclusion_DeviceMap_NamedDomains)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.device_map = {{0, GlobalDeviceAddress::cuda(0)}};
    cfg.domain_definitions.push_back(DomainDefinition::parse("gpu=cuda:0,cuda:1"));

    EXPECT_TRUE(ruleFiresFor(v, "device-map-named-domains-mutex", cfg));
}

TEST(Test__ConfigValidator, MutualExclusion_Device_SimpleTP)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.device_for_this_rank = GlobalDeviceAddress::cuda(0);
    cfg.tp_degree = 2;

    EXPECT_TRUE(ruleFiresFor(v, "device-simple-tp-conflict", cfg));
}

TEST(Test__ConfigValidator, MutualExclusion_Device_DeviceMap)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.device_for_this_rank = GlobalDeviceAddress::cuda(0);
    cfg.device_map = {{0, GlobalDeviceAddress::cuda(0)}, {1, GlobalDeviceAddress::cuda(1)}};

    EXPECT_TRUE(ruleFiresFor(v, "device-map-device-mutex", cfg));
}

// ============================================================================
// Non-Conflicting Combinations (should NOT fire)
// ============================================================================

TEST(Test__ConfigValidator, NoConflict_SingleDevice_Only)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.device_for_this_rank = GlobalDeviceAddress::cuda(0);

    EXPECT_TRUE(noErrors(v, cfg));
}

TEST(Test__ConfigValidator, NoConflict_TPDevices_Only)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};

    EXPECT_TRUE(noErrors(v, cfg));
}

TEST(Test__ConfigValidator, NoConflict_SimpleTP_Only)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.tp_degree = 4;

    EXPECT_TRUE(noErrors(v, cfg));
}

TEST(Test__ConfigValidator, NoConflict_NamedDomains_WithPPStage)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.domain_definitions.push_back(DomainDefinition::parse("gpu1=cuda:0,cuda:1"));
    cfg.domain_definitions.push_back(DomainDefinition::parse("gpu2=cuda:2,cuda:3"));
    cfg.pp_stage_definitions.push_back(PPStageDefinition::parse("0=gpu1:0-13"));
    cfg.pp_stage_definitions.push_back(PPStageDefinition::parse("1=gpu2:14-27"));

    EXPECT_TRUE(noErrors(v, cfg));
}

TEST(Test__ConfigValidator, NoConflict_DeviceMap_Only)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.device_map = {{0, GlobalDeviceAddress::cuda(0)}, {1, GlobalDeviceAddress::cuda(1)}};

    EXPECT_TRUE(noErrors(v, cfg));
}

TEST(Test__ConfigValidator, NoConflict_TPDevices_WithWeights)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    cfg.tp_weights = {0.6f, 0.4f};

    EXPECT_TRUE(noErrors(v, cfg));
}

TEST(Test__ConfigValidator, NoConflict_TPDevices_WithMatchingDegree)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    cfg.tp_degree = 2; // Matches tp_devices count

    EXPECT_TRUE(noErrors(v, cfg));
}

// ============================================================================
// Co-Requirement Tests
// ============================================================================

TEST(Test__ConfigValidator, CoReq_TPWeights_RequiresTPDevices)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.tp_weights = {0.5f, 0.5f};
    // No tp_devices

    EXPECT_TRUE(ruleFiresFor(v, "tp-weights-requires-tp-devices", cfg));
}

TEST(Test__ConfigValidator, CoReq_TPWeights_WithTPDevices_OK)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    cfg.tp_weights = {0.5f, 0.5f};

    EXPECT_FALSE(ruleFiresFor(v, "tp-weights-requires-tp-devices", cfg));
}

TEST(Test__ConfigValidator, CoReq_PPStage_RequiresDomain)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.pp_stage_definitions.push_back(PPStageDefinition::parse("0=orphan:0-13"));
    // No domain_definitions

    EXPECT_TRUE(ruleFiresFor(v, "pp-stage-requires-domain", cfg));
}

TEST(Test__ConfigValidator, CoReq_PPStage_WithDomain_OK)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.domain_definitions.push_back(DomainDefinition::parse("gpu=cuda:0,cuda:1"));
    cfg.pp_stage_definitions.push_back(PPStageDefinition::parse("0=gpu:0-13"));

    EXPECT_FALSE(ruleFiresFor(v, "pp-stage-requires-domain", cfg));
}

TEST(Test__ConfigValidator, CoReq_DeviceModeExplicit_RequiresMap)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.device_mode = DeviceAssignmentMode::EXPLICIT;
    // No device_map

    EXPECT_TRUE(ruleFiresFor(v, "device-mode-explicit-requires-map", cfg));
}

TEST(Test__ConfigValidator, CoReq_DeviceModeExplicit_WithMap_OK)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.device_mode = DeviceAssignmentMode::EXPLICIT;
    cfg.device_map = {{0, GlobalDeviceAddress::cuda(0)}};

    EXPECT_FALSE(ruleFiresFor(v, "device-mode-explicit-requires-map", cfg));
}

// ============================================================================
// Intra-Mode Consistency Tests
// ============================================================================

TEST(Test__ConfigValidator, Consistency_TPDevices_DegreeMismatch)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    cfg.tp_degree = 4; // 4 != 2 devices

    EXPECT_TRUE(ruleFiresFor(v, "tp-devices-degree-mismatch", cfg));
}

TEST(Test__ConfigValidator, Consistency_TPDevices_DegreeMatches_OK)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    cfg.tp_degree = 2;

    EXPECT_FALSE(ruleFiresFor(v, "tp-devices-degree-mismatch", cfg));
}

TEST(Test__ConfigValidator, Consistency_TPDevices_DegreeDefault_OK)
{
    // tp_degree=1 (default), tp_devices set — rule doesn't apply (degree wasn't explicitly set)
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    // tp_degree = 1 (default)

    EXPECT_FALSE(ruleFiresFor(v, "tp-devices-degree-mismatch", cfg));
}

TEST(Test__ConfigValidator, Consistency_TPDevices_NoDuplicates)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(0)}; // Duplicate!

    EXPECT_TRUE(ruleFiresFor(v, "tp-devices-no-duplicates", cfg));
}

TEST(Test__ConfigValidator, Consistency_TPDevices_DistinctDevices_OK)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};

    EXPECT_FALSE(ruleFiresFor(v, "tp-devices-no-duplicates", cfg));
}

TEST(Test__ConfigValidator, Consistency_TPScopeGlobal_WithTPDevices)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.tp_scope = TPScope::GLOBAL;
    cfg.tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};

    EXPECT_TRUE(ruleFiresFor(v, "tp-scope-global-tp-devices-conflict", cfg));
}

TEST(Test__ConfigValidator, Consistency_TPScopeLocal_WithTPDevices_OK)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.tp_scope = TPScope::LOCAL;
    cfg.tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};

    EXPECT_FALSE(ruleFiresFor(v, "tp-scope-global-tp-devices-conflict", cfg));
}

TEST(Test__ConfigValidator, Consistency_SimpleTP_WithNamedDomains)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.tp_degree = 2;
    cfg.domain_definitions.push_back(DomainDefinition::parse("gpu=cuda:0,cuda:1"));

    EXPECT_TRUE(ruleFiresFor(v, "simple-tp-with-named-domains", cfg));
}

TEST(Test__ConfigValidator, Consistency_SimpleTP_WithTopology)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();
    cfg.tp_degree = 2;
    cfg.topology_string = "PP(TP(cuda:0,cuda:1))";

    EXPECT_TRUE(ruleFiresFor(v, "simple-tp-with-topology", cfg));
}

// ============================================================================
// Multiple Errors Test (validator collects ALL violations, not just first)
// ============================================================================

TEST(Test__ConfigValidator, MultipleErrors_CollectsAll)
{
    auto v = ConfigValidator::createStandard();
    auto cfg = makeClean();

    // Set up a maximally conflicting config
    cfg.device_for_this_rank = GlobalDeviceAddress::cuda(0);
    cfg.tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    cfg.domain_definitions.push_back(DomainDefinition::parse("gpu=cuda:0,cuda:1"));

    auto errors = v.validate(cfg);
    // Should fire at least 3 rules:
    // device-tp-devices-mutex, device-named-domains-mutex, tp-devices-named-domains-mutex
    EXPECT_GE(errors.size(), 3u);

    // Verify each expected rule is present
    bool has_device_tp = false, has_device_domains = false, has_tp_domains = false;
    for (const auto &e : errors)
    {
        if (e.rule_id == "device-tp-devices-mutex")
            has_device_tp = true;
        if (e.rule_id == "device-named-domains-mutex")
            has_device_domains = true;
        if (e.rule_id == "tp-devices-named-domains-mutex")
            has_tp_domains = true;
    }
    EXPECT_TRUE(has_device_tp);
    EXPECT_TRUE(has_device_domains);
    EXPECT_TRUE(has_tp_domains);
}

// ============================================================================
// Custom Rule Test (extensibility)
// ============================================================================

TEST(Test__ConfigValidator, CustomRule_CanBeAdded)
{
    ConfigValidator v;

    v.addRule({
        .id = "custom-test-rule",
        .description = "Test rule",
        .fix_hint = "Fix it",
        .applies = [](const OrchestrationConfig &)
        { return true; },
        .check = [](const OrchestrationConfig &c) -> std::optional<std::string>
        {
            if (c.tp_degree == 42)
            {
                return "TP degree 42 is not allowed";
            }
            return std::nullopt;
        },
    });

    auto cfg = makeClean();
    EXPECT_TRUE(noErrors(v, cfg));

    cfg.tp_degree = 42;
    auto errors = v.validate(cfg);
    ASSERT_EQ(errors.size(), 1u);
    EXPECT_EQ(errors[0].rule_id, "custom-test-rule");
    EXPECT_TRUE(errors[0].message.find("42") != std::string::npos);
}

// ============================================================================
// Integration: OrchestrationConfig::validate() delegates to ConfigValidator
// ============================================================================

TEST(Test__ConfigValidator, Integration_OrchestrationConfig_Validate_FiresDeviceRules)
{
    OrchestrationConfig cfg;
    cfg.device_for_this_rank = GlobalDeviceAddress::cuda(0);
    cfg.tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};

    auto errors = cfg.validate();
    EXPECT_FALSE(errors.empty());

    // The error should come from ConfigValidator and contain the rule ID
    bool found = false;
    for (const auto &e : errors)
    {
        if (e.find("device-tp-devices-mutex") != std::string::npos)
        {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "Expected device-tp-devices-mutex error from OrchestrationConfig::validate()";
}

TEST(Test__ConfigValidator, Integration_OrchestrationConfig_Validate_StillChecksExistingRules)
{
    // Ensure existing validation (TP degree, PP degree, etc.) still works alongside new rules
    OrchestrationConfig cfg;
    cfg.tp_degree = 0; // Invalid

    auto errors = cfg.validate();
    EXPECT_FALSE(errors.empty());

    bool found_tp = false;
    for (const auto &e : errors)
    {
        if (e.find("TP degree") != std::string::npos)
        {
            found_tp = true;
            break;
        }
    }
    EXPECT_TRUE(found_tp) << "Existing TP degree validation should still fire";
}

TEST(Test__ConfigValidator, Integration_OverlayRootPlacementRejectsLegacySingleDevice)
{
    OrchestrationConfig cfg;
    cfg.device_for_this_rank = GlobalDeviceAddress::cuda(0);
    cfg.moe_expert_parallel_plan = std::make_shared<MoEExpertParallelPlan>();
    cfg.moe_expert_parallel_plan->enabled = true;
    cfg.moe_expert_parallel_plan->execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
    cfg.moe_expert_parallel_plan->continuation_domain = "cuda_fast";
    cfg.moe_expert_parallel_plan->shared_expert_domain = "cuda_fast";
    cfg.moe_expert_parallel_plan->domains = {
        ExpertComputeDomain{
            .name = "cuda_fast",
            .kind = ExpertDomainKind::SingleDevice,
            .backend = CollectiveBackendType::AUTO,
            .participants = {GlobalDeviceAddress::cuda(0)},
            .owner_rank = 0,
            .compute_kind = ExpertDomainComputeKind::ReplicatedExperts,
        },
        ExpertComputeDomain{
            .name = "cpu_cold",
            .kind = ExpertDomainKind::NodeLocalTP,
            .backend = CollectiveBackendType::UPI,
            .participants = {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)},
            .world_ranks = {0, 1},
            .compute_kind = ExpertDomainComputeKind::TensorParallelExperts,
        },
    };
    cfg.moe_expert_parallel_plan->routed_tiers = {
        ExpertRoutedTier{.name = "fast", .domain = "cuda_fast", .priority = 0},
        ExpertRoutedTier{.name = "cold", .domain = "cpu_cold", .priority = 1, .fallback = true},
    };

    const auto errors = cfg.validate();

    bool found_overlay_conflict = false;
    for (const auto &error : errors)
    {
        if (error.find("Conflicting options: --device/-d cuda:0 and --moe-expert-overlay-continuation cuda_fast") != std::string::npos)
        {
            found_overlay_conflict = true;
            break;
        }
    }
    EXPECT_TRUE(found_overlay_conflict) << "Expected exact overlay root placement conflict; first error: "
                                        << (errors.empty() ? "<none>" : errors.front());
}
