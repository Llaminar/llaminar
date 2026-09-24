/**
 * @file Test__OrchestrationPlanningPolicy.cpp
 * @brief Device-free automatic/apply intent and hard-filter invariants.
 *
 * Exercise identical CLI/YAML requests without model/device initialization.
 * Candidate filters must inspect every compute participant, while host memory
 * and the process control plane are deliberately outside compute selection.
 * These tests certify policy semantics, not automatic runtime implementation.
 */
#include "config/OrchestrationPlanningPolicy.h"
#include "config/OrchestrationConfigParser.h"
#include "config/ConfigValidator.h"
#include "config/OrchestrationConfigDocument.h"
#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <limits>

using namespace llaminar2;

namespace
{
    /** @return Shared-parser configuration with argv storage owned for the call. */
    OrchestrationConfig parsePlanning(std::vector<std::string> args)
    {
        args.insert(args.begin(), "llaminar2");
        std::vector<char *> argv;
        for (auto &arg : args) argv.push_back(arg.data());
        return OrchestrationConfigParser{}.parseArgs(static_cast<int>(argv.size()), argv.data());
    }
}

TEST(OrchestrationPlanningPolicy, DefaultAndExplicitAutoHaveIdenticalUnrestrictedIntent)
{
    for (const auto &args : {std::vector<std::string>{}, std::vector<std::string>{"--auto"},
                            std::vector<std::string>{"--planning-mode", "auto"}})
    {
        const auto resolved = resolveOrchestrationIntent(parsePlanning(args));
        ASSERT_TRUE(std::holds_alternative<AutomaticOrchestrationRequest>(resolved));
        const auto &automatic = std::get<AutomaticOrchestrationRequest>(resolved);
        for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
            EXPECT_TRUE(automatic.allows(backend));
        for (const auto strategy : {OrchestrationStrategy::SingleDevice, OrchestrationStrategy::TensorParallel,
                                   OrchestrationStrategy::PipelineParallel, OrchestrationStrategy::ExpertOverlay})
            EXPECT_TRUE(automatic.allows(strategy));
        EXPECT_FALSE(automatic.allows(DeviceType::Metal));
        EXPECT_FALSE(automatic.allows(DeviceType::Vulkan));
    }
}

TEST(OrchestrationPlanningPolicy, SharedWorkloadHintIsAcceptedBeforeDeviceAdmission)
{
    const auto cli = parsePlanning({"--plan-workload", "512,384", "--context-length", "1024"});
    const auto yaml = OrchestrationConfigParser{}.parseYamlString("planning:\n  workload: 512,384\n");
    EXPECT_EQ(cli.automatic_planning.workload, OrchestrationPlanningWorkload(512, 384));
    EXPECT_EQ(yaml.automatic_planning.workload, cli.automatic_planning.workload);
    EXPECT_TRUE(cli.automatic_planning.specified());
    EXPECT_NO_THROW((void)resolveOrchestrationIntent(cli));
    EXPECT_FALSE(parsePlanning({}).automatic_planning.workload);
    auto applied = cli;
    applied.device_for_this_rank = GlobalDeviceAddress::cpu(0, "host");
    EXPECT_THROW((void)resolveOrchestrationIntent(applied), std::invalid_argument);
    auto too_small = cli;
    too_small.max_seq_len = 895;
    EXPECT_THROW((void)resolveOrchestrationIntent(too_small), std::invalid_argument);
}

TEST(OrchestrationPlanningPolicy, WorkloadParsingRejectsPartialAndCoercedCounts)
{
    for (const auto value : {"", "512", "512,", ",384", "512,384,1", "0,1", "1,0", "-1,2",
                            "1,-2", "+1,2", "1.0,2", "1,2junk", "2147483648,1", "1,2147483648"})
    {
        SCOPED_TRACE(value);
        EXPECT_THROW((void)parsePlanning({"--plan-workload", value}), std::invalid_argument);
        EXPECT_THROW((void)parseOrchestrationPlanningWorkload(value), std::invalid_argument);
    }
    EXPECT_EQ(parseOrchestrationPlanningWorkload(" 512 , 512 "), OrchestrationPlanningWorkload(512, 512));
}

TEST(OrchestrationPlanningPolicy, DefaultWorkloadKeepsBothPhasesAndChecksCapacityWithoutOverflow)
{
    for (const int context : {2, 3, 128, 511, 512, 8192, std::numeric_limits<int>::max()})
    {
        const auto work = OrchestrationPlanningWorkload::defaultsForContext(context);
        EXPECT_GT(work.promptTokens(), 0);
        EXPECT_GT(work.generationTokens(), 0);
        EXPECT_NO_THROW(work.requireFitsContext(context));
    }
    EXPECT_EQ(OrchestrationPlanningWorkload::defaultsForContext(128), OrchestrationPlanningWorkload(64, 64));
    EXPECT_EQ(OrchestrationPlanningWorkload::defaultsForContext(8192), OrchestrationPlanningWorkload(256, 256));
    for (const int context : {std::numeric_limits<int>::min(), -1, 0, 1})
    {
        EXPECT_THROW((void)OrchestrationPlanningWorkload::defaultsForContext(context), std::invalid_argument);
        EXPECT_THROW(OrchestrationPlanningWorkload(1, 1).requireFitsContext(context), std::invalid_argument);
    }
    EXPECT_THROW(OrchestrationPlanningWorkload(std::numeric_limits<int>::max(), 1)
        .requireFitsContext(std::numeric_limits<int>::max()), std::invalid_argument);
}

TEST(OrchestrationPlanningPolicy, EveryExplicitTopologySelectsApply)
{
    for (const auto &args : {
        std::vector<std::string>{"-d", "cpu"}, {"-d", "cuda:0"}, {"-d", "rocm:0"},
        {"--tp-devices", "cuda:0,cuda:1"}, {"-tp", "2"}, {"-pp", "2"},
        {"--define-domain", "workers=rocm:0,rocm:1"},
        {"--expert-tier", "compute=cuda:0;priority=0", "--expert-tier", "storage=cpu:0;priority=10"}})
    {
        const auto config = parsePlanning(args);
        EXPECT_TRUE(std::holds_alternative<ApplyOrchestrationRequest>(resolveOrchestrationIntent(config)));
        auto automatic = config;
        automatic.planning_mode = OrchestrationPlanningMode::Automatic;
        EXPECT_THROW((void)resolveOrchestrationIntent(automatic), std::invalid_argument);
        const auto errors = ConfigValidator::createStandard().validate(automatic);
        EXPECT_TRUE(std::any_of(errors.begin(), errors.end(), [](const auto &error) {
            return error.rule_id == "orchestration-planning-intent";
        }));
    }
}

TEST(OrchestrationPlanningPolicy, HostParticipationIsTypedExplicitAndCannotModifyApply)
{
    EXPECT_FALSE(parsePlanning({}).automatic_planning.host_participation);
    for (const auto value : {"all", "best-subset"})
    {
        const auto config = parsePlanning({"--auto-hosts", value});
        EXPECT_EQ(config.automatic_planning.host_participation, parseAutomaticHostParticipation(value));
        EXPECT_TRUE(config.automatic_planning.specified());
        EXPECT_TRUE(std::holds_alternative<AutomaticOrchestrationRequest>(resolveOrchestrationIntent(config)));
        auto applied = config;
        applied.device_for_this_rank = GlobalDeviceAddress::cpu(0, "host");
        EXPECT_THROW((void)resolveOrchestrationIntent(applied), std::invalid_argument);
        const auto yaml = OrchestrationConfigParser{}.parseYamlString(
            std::string("planning:\n  hosts: ") + value + "\n");
        EXPECT_EQ(yaml.automatic_planning.host_participation, config.automatic_planning.host_participation);
    }
    for (const auto value : {"", "any", "all-ranks", "ALL"})
        EXPECT_THROW((void)parsePlanning({"--auto-hosts", value}), std::invalid_argument);
    AutomaticOrchestrationOptions invalid;
    invalid.host_participation = static_cast<AutomaticHostParticipation>(255);
    EXPECT_THROW((void)AutomaticOrchestrationRequest{invalid}, std::invalid_argument);
}

TEST(OrchestrationPlanningPolicy, SavedTopologyCannotBeSilentlyReoptimized)
{
    auto config = parsePlanning({"-d", "cuda:1"});
    config.config_file_path = "saved-plan.yaml";
    EXPECT_TRUE(std::holds_alternative<ApplyOrchestrationRequest>(resolveOrchestrationIntent(config)));
    config.automatic_planning.prefer_backend = DeviceType::ROCm;
    EXPECT_THROW((void)resolveOrchestrationIntent(config), std::invalid_argument);
    config.automatic_planning = {};
    config.device_for_this_rank.reset();
    EXPECT_THROW((void)resolveOrchestrationIntent(config), std::invalid_argument);
    config.config_file_path.clear();
    config.planning_mode = OrchestrationPlanningMode::Apply;
    EXPECT_THROW((void)resolveOrchestrationIntent(config), std::invalid_argument);
}

TEST(OrchestrationPlanningPolicy, FiltersInspectAllComputeParticipantsNotHostControl)
{
    const auto resolved = resolveOrchestrationIntent(parsePlanning({
        "--only-backends", "cuda,rocm", "--only-strategies", "tp,expert-overlay"}));
    const auto &request = std::get<AutomaticOrchestrationRequest>(resolved);
    const std::array gpu_only{DeviceType::CUDA, DeviceType::ROCm, DeviceType::CUDA};
    const std::array with_cpu{DeviceType::CUDA, DeviceType::ROCm, DeviceType::CPU};
    EXPECT_TRUE(request.allows(OrchestrationStrategy::ExpertOverlay, gpu_only));
    EXPECT_FALSE(request.allows(OrchestrationStrategy::ExpertOverlay, with_cpu));
    EXPECT_FALSE(request.allows(OrchestrationStrategy::PipelineParallel, gpu_only));
    EXPECT_FALSE(request.allows(OrchestrationStrategy::ExpertOverlay, {}));
}

TEST(OrchestrationPlanningPolicy, OnlyTensorAndPipelineDoesNotIncludeSingleDevice)
{
    const auto resolved = resolveOrchestrationIntent(parsePlanning({"--only-strategies=tp,pp"}));
    const auto &request = std::get<AutomaticOrchestrationRequest>(resolved);
    EXPECT_TRUE(request.allows(OrchestrationStrategy::TensorParallel));
    EXPECT_TRUE(request.allows(OrchestrationStrategy::PipelineParallel));
    EXPECT_FALSE(request.allows(OrchestrationStrategy::SingleDevice));
    EXPECT_FALSE(request.allows(OrchestrationStrategy::ExpertOverlay));
}

TEST(OrchestrationPlanningPolicy, BackendRestrictionsAreSymmetric)
{
    for (const auto backend : {"cpu", "cuda", "rocm"})
    {
        const auto resolved = resolveOrchestrationIntent(parsePlanning({"--only-backends", backend}));
        const auto &request = std::get<AutomaticOrchestrationRequest>(resolved);
        for (const auto candidate : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
            EXPECT_EQ(request.allows(candidate), candidate == parseOrchestrationComputeBackend(backend));
    }
}

TEST(OrchestrationPlanningPolicy, PreferencesCannotOverrideHardRestrictions)
{
    for (const auto &args : {std::vector<std::string>{"--only-backends", "cuda", "--prefer-backend", "rocm"},
                            {"--only-strategies", "tp,pp", "--prefer-strategy", "single"}})
        EXPECT_THROW((void)resolveOrchestrationIntent(parsePlanning(args)), std::invalid_argument);
    const auto resolved = resolveOrchestrationIntent(parsePlanning({"--only-backends", "cuda,rocm",
        "--prefer-backend", "rocm", "--only-strategies", "tp,pp", "--prefer-strategy", "tp"}));
    const auto &options = std::get<AutomaticOrchestrationRequest>(resolved).options();
    EXPECT_EQ(options.prefer_backend, DeviceType::ROCm);
    EXPECT_EQ(options.prefer_strategy, OrchestrationStrategy::TensorParallel);
}

TEST(OrchestrationPlanningPolicy, MalformedFiltersCannotBroadenSelection)
{
    for (const auto input : {"", ",cuda", "cuda,", "cuda,,rocm", "cuda,cuda", "auto", "metal", "gpu", "CUDA"})
        EXPECT_THROW((void)parsePlanning({"--only-backends", input}), std::invalid_argument);
    for (const auto input : {"", "tp,", "tp,,pp", "tp,tp", "hybrid", "single-gpu", "cpu-only", "auto"})
        EXPECT_THROW((void)parsePlanning({"--only-strategies", input}), std::invalid_argument);
    EXPECT_EQ(parseOrchestrationBackendList(" cuda , rocm "), (std::vector{DeviceType::CUDA, DeviceType::ROCm}));
}

TEST(OrchestrationPlanningPolicy, ProgrammaticCallersCannotConstructInvalidSealedRequests)
{
    AutomaticOrchestrationOptions options;
    options.only_backends = std::vector<DeviceType>{};
    EXPECT_THROW((void)AutomaticOrchestrationRequest{options}, std::invalid_argument);
    options.only_backends = std::vector{DeviceType::Metal};
    EXPECT_THROW((void)AutomaticOrchestrationRequest{options}, std::invalid_argument);
    options.only_backends = std::vector{DeviceType::CPU, DeviceType::CPU};
    EXPECT_THROW((void)AutomaticOrchestrationRequest{options}, std::invalid_argument);
    options = {};
    options.only_strategies = std::vector{static_cast<OrchestrationStrategy>(200)};
    EXPECT_THROW((void)AutomaticOrchestrationRequest{options}, std::invalid_argument);
    auto config = parsePlanning({});
    config.planning_mode = static_cast<OrchestrationPlanningMode>(200);
    EXPECT_THROW((void)resolveOrchestrationIntent(config), std::invalid_argument);
}

TEST(OrchestrationPlanningPolicy, YamlAndCliShareTypedSemanticsAndSectionBoundaries)
{
    const auto yaml = OrchestrationConfigParser{}.parseYamlString(
        "planning:\n  mode: auto\n  only_backends: cuda,rocm\n  only_strategies: tp,pp\n"
        "  prefer_backend: rocm\n  prefer_strategy: pp\nmodel_path: model.gguf\n");
    const auto cli = parsePlanning({"--auto", "--only-backends", "cuda,rocm", "--only-strategies", "tp,pp",
        "--prefer-backend", "rocm", "--prefer-strategy", "pp", "-m", "model.gguf"});
    EXPECT_EQ(yaml.model_path, cli.model_path);
    const auto resolved = resolveOrchestrationIntent(yaml);
    EXPECT_TRUE(std::holds_alternative<AutomaticOrchestrationRequest>(resolved));
    const auto &expected = std::get<AutomaticOrchestrationRequest>(resolved).options();
    EXPECT_EQ(expected.only_backends, cli.automatic_planning.only_backends);
    EXPECT_EQ(expected.only_strategies, cli.automatic_planning.only_strategies);
    EXPECT_EQ(expected.prefer_backend, cli.automatic_planning.prefer_backend);
    EXPECT_EQ(expected.prefer_strategy, cli.automatic_planning.prefer_strategy);
    EXPECT_THROW(OrchestrationConfigParser{}.parseYamlString("planning:\n  only_backend: cuda\n"), std::invalid_argument);
    EXPECT_THROW(OrchestrationConfigParser{}.parseYamlString("planning:\n  mode: automatic\n"), std::invalid_argument);
    EXPECT_THROW(OrchestrationConfigParser{}.parseYamlString("planning:\n  only_backends:\n"), std::invalid_argument);
    EXPECT_THROW(OrchestrationConfigParser{}.parseYamlString("planning:\n  - cuda\n"), std::invalid_argument);
}

TEST(OrchestrationPlanningPolicy, HelpComesFromTheSameSharedSpecification)
{
    const auto help = OrchestrationConfigParser::getHelpText();
    for (const auto option : {"--auto", "--planning-mode", "--only-backends", "--only-strategies",
                              "--prefer-backend", "--prefer-strategy", "--plan-workload", "--auto-device-counts"})
        EXPECT_NE(help.find(option), std::string::npos);
}

TEST(OrchestrationPlanningPolicy, DeviceCountsSurviveCliYamlAndMPIDocument)
{
    const auto cli = parsePlanning({"--auto-device-counts", "cuda=2,rocm=4,cpu=2"});
    const auto yaml = OrchestrationConfigParser{}.parseYamlString(
        "planning:\n  device_counts: cuda=2,rocm=4,cpu=2\n");
    ASSERT_EQ(cli.automatic_planning.device_counts, yaml.automatic_planning.device_counts);
    const auto copy = deserializeOrchestrationConfig(serializeOrchestrationConfig(cli));
    EXPECT_EQ(copy.automatic_planning.device_counts, cli.automatic_planning.device_counts);
    const auto resolved = resolveOrchestrationIntent(copy);
    ASSERT_TRUE(std::holds_alternative<AutomaticOrchestrationRequest>(resolved));
    const auto &policy = std::get<AutomaticOrchestrationRequest>(resolved);
    const std::vector backends{DeviceType::CUDA, DeviceType::CUDA, DeviceType::ROCm,
        DeviceType::ROCm, DeviceType::ROCm, DeviceType::ROCm, DeviceType::CPU, DeviceType::CPU};
    EXPECT_TRUE(policy.allows(OrchestrationStrategy::ExpertOverlay, backends));
    EXPECT_FALSE(policy.allows(OrchestrationStrategy::ExpertOverlay, std::span(backends).first(7)));
    auto excess = backends;
    excess.push_back(DeviceType::CUDA);
    EXPECT_FALSE(policy.allows(OrchestrationStrategy::ExpertOverlay, excess));
    auto applied = cli;
    applied.device_for_this_rank = GlobalDeviceAddress::cuda(0);
    EXPECT_THROW((void)resolveOrchestrationIntent(applied), std::invalid_argument);
}

TEST(OrchestrationPlanningPolicy, DeviceCountsRejectInvalidAndConflictingIntent)
{
    for (const auto value : {"", "cuda", "cuda=", "=2", "cuda=0", "cpu=-1", "rocm=+2",
                            "cuda=2.0", "cuda=2junk", "cuda=2147483648", "metal=1", "cuda=1,",
                            "cuda=1,cuda=1", "cuda=1,cuda=2", "cuda=1=2"})
    {
        SCOPED_TRACE(value);
        EXPECT_THROW((void)parseAutomaticDeviceCounts(value), std::invalid_argument);
    }
    auto config = parsePlanning({"--only-backends", "rocm", "--auto-device-counts", "cuda=2"});
    EXPECT_THROW((void)resolveOrchestrationIntent(config), std::invalid_argument);
    AutomaticOrchestrationOptions options;
    options.device_counts = std::vector<AutomaticBackendDeviceCount>{};
    EXPECT_THROW((void)AutomaticOrchestrationRequest(options), std::invalid_argument);
    options.device_counts = {{DeviceType::CPU, 0}};
    EXPECT_THROW((void)AutomaticOrchestrationRequest(options), std::invalid_argument);
    EXPECT_EQ(parseAutomaticDeviceCounts(" cuda = 2 , cpu = 1 "),
        (std::vector<AutomaticBackendDeviceCount>{{DeviceType::CUDA, 2}, {DeviceType::CPU, 1}}));
}
