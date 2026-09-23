/**
 * @file Test__OrchestrationConfigDocument.cpp
 * @brief Device-free round-trip and hostile-input proofs for plan configuration.
 *
 * Exercise the same value codec used by public --config, not a test-owned YAML
 * projection. No model, MPI session, accelerator, or memory admission is needed.
 * Documents retain policy intent and exact topology; they are not certificates.
 * Saved selection preserves the discovery order without redefining physical
 * CPU/GPU identities or allocating an execution communicator in the parser.
 */
#include "config/OrchestrationConfigDocument.h"
#include "config/OrchestrationConfigParser.h"
#include "utils/DebugEnv.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <unistd.h>

using namespace llaminar2;

namespace
{
    using Json = nlohmann::json;

    /** @brief Restore the inherited process policy after parser publication tests. */
    class StartupEnvironment final
    {
    public:
        /** @brief Save both environment-backed production startup inputs. */
        StartupEnvironment()
        {
            for (const auto *key : {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "LLAMINAR_DETERMINISTIC"})
                saved.emplace_back(key, std::getenv(key) ? std::optional<std::string>{std::getenv(key)} : std::nullopt);
        }
        /** @brief Restore both environ and the already-read kernel-policy snapshot. */
        ~StartupEnvironment()
        {
            for (const auto &[key, value] : saved)
                if (value) setenv(key.c_str(), value->c_str(), 1);
                else unsetenv(key.c_str());
            mutableDebugEnv().reload();
        }
        StartupEnvironment(const StartupEnvironment &) = delete;
        StartupEnvironment &operator=(const StartupEnvironment &) = delete;
    private:
        std::vector<std::pair<std::string, std::optional<std::string>>> saved;
    };

    /** @return A rich multi-host input with deliberately non-default policies. */
    OrchestrationConfig example(DeviceType backend)
    {
        OrchestrationConfig config;
        config.model_path = "models/a \"quoted\" model.gguf";
        config.hostfile = "cluster hosts.txt";
        config.planning_mode = OrchestrationPlanningMode::Apply;
        config.mpi_procs = 3;
        config.execution_rank_selection = ExecutionRankSelection({4, 2, 1});
        config.max_seq_len = 32768;
        config.batch_size = 4;
        config.prompt = "line one\nline two\twith unicode: \u00e9";
        config.prompt_was_explicitly_provided = true;
        config.seed = 91;
        config.temperature = 0.61f;
        config.top_p = 0.78f;
        config.n_predict = 384;
        config.kv_cache_precision = "q8_1";
        config.tp_allreduce_precision_override = "bf16";
        config.max_gpu_memory_mb = 12288;
        config.max_cpu_memory_mb = 32768;
        config.prefix_cache.ram_budget_bytes = (1ull << 33) + 64;
        config.prefix_cache.disk_budget_bytes = (1ull << 37) + 128;
        config.prefix_cache.disk_dir = "prefix cache/quoted \"tier\"";
        config.mtp.enabled = true;
        config.mtp.draft_tokens = 15;
        config.mtp.graph_capacity_draft_tokens = 15;
        config.mtp.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
        config.mtp.depth_policy.min_depth = 0;
        config.mtp.depth_policy.max_depth = 15;
        config.mtp.verify_mode = MTPVerifyMode::SpeculativeSampling;
        config.moe_rebalance.migration_transfer_slots = 8;
        config.moe_rebalance.migration_execution_streams = 2;
        config.moe_rebalance.migration_cycles_per_wave = 3;
        config.moe_rebalance.migration_payoff_horizon_tokens = (1ull << 34) + 1;
        config.moe_routed_prefill.llep_alpha_numerator = 5;
        config.moe_routed_prefill.llep_alpha_denominator = 3;

        auto &plan = *(config.moe_routed_expert_plan = std::make_shared<MoERoutedExpertPlacementPlan>());
        plan.enabled = true;
        plan.continuation_domain = "continuation";
        plan.base_model_domain = "continuation";
        plan.shared_expert_domain = "continuation";
        plan.continuation_domain_spec.domain = "continuation";
        plan.continuation_domain_spec.setDensePolicy(DenseParallelPolicy::Replicated);
        plan.continuation_dense_policy_intent = MoEContinuationDensePolicyIntent::Explicit;
        plan.residency_policy = RoutedExpertResidencyPolicy::RoutedTierRebalanced;
        plan.authority_execution = MoEOverlayAuthorityExecutionKind::HostResident;
        plan.replica_cache_capacity.emplace(8, 3);
        RoutedExpertDomain continuation;
        continuation.name = "continuation";
        continuation.scope = ExecutionDomainScope::SINGLE;
        continuation.participants = {{"gpu-host", 1, backend, 0}};
        continuation.world_ranks = {0};
        continuation.owner_rank = 0;
        RoutedExpertDomain remote;
        remote.name = "remote";
        remote.scope = ExecutionDomainScope::GLOBAL;
        remote.backend = CollectiveBackendType::MPI;
        remote.participants = {GlobalDeviceAddress::cpu(0, "cpu-host-a"),
                               GlobalDeviceAddress::cpu(1, "cpu-host-b")};
        remote.world_ranks = {1, 2};
        remote.weights = {0.25f, 0.75f};
        plan.domains = {continuation, remote};
        plan.routed_tiers = {{.name = "preferred", .domain = "continuation", .priority = -5,
                              .memory_budget_bytes = 8ull << 30, .resolved_live_experts_per_layer = {2}},
                             {.name = "remaining", .domain = "remote", .priority = 70,
                              .fallback = true, .resolved_live_experts_per_layer = {2}}};
        plan.placements = {{.layer = 0, .routed_expert_tier = {0, 1, 1, 0}}};
        for (const auto &domain : plan.domains)
            config.domain_definitions.push_back(DomainDefinition::fromExecutionDomainDefinition(domain.toExecutionDomainDefinition()));
        return config;
    }

    /** @brief Own the one temporary file exercised by the public --config route. */
    class ConfigFile final
    {
    public:
        /** @brief Materialize a uniquely owned small document fixture. */
        explicit ConfigFile(const std::string &contents)
        {
            char pattern[] = "/tmp/llaminar-config-document-XXXXXX";
            const int fd = mkstemp(pattern);
            if (fd < 0) throw std::runtime_error("Cannot create configuration fixture");
            close(fd);
            path = pattern;
            std::ofstream output(path);
            output << contents;
            if (!output) throw std::runtime_error("Cannot write configuration fixture");
        }
        /** @brief Remove only the uniquely created fixture. */
        ~ConfigFile() { std::remove(path.c_str()); }
        ConfigFile(const ConfigFile &) = delete;
        ConfigFile &operator=(const ConfigFile &) = delete;
        std::string path;
    };
}

TEST(OrchestrationConfigDocument, DefaultConfigurationRoundTripsExactly)
{
    const auto document = serializeOrchestrationConfig(OrchestrationConfig{});
    EXPECT_EQ(serializeOrchestrationConfig(deserializeOrchestrationConfig(document)), document);
    const auto parsed = OrchestrationConfigParser{}.parseYamlString("\n\t" + document);
    EXPECT_EQ(serializeOrchestrationConfig(parsed), document);
}

TEST(OrchestrationConfigDocument, AllBackendsMovementOwnerAndOptionalMtpPoliciesRoundTrip)
{
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
    for (const auto movement : {MoERebalanceRuntimeMode::Off, MoERebalanceRuntimeMode::Observe, MoERebalanceRuntimeMode::Dynamic})
    for (const auto owner : {RoutedExpertOwnerOrder::Ordinal, RoutedExpertOwnerOrder::Random})
    for (const auto rate : {std::optional<double>{}, std::optional<double>{0.3}})
    {
        auto config = example(backend);
        config.moe_rebalance.mode = movement;
        config.moe_routed_expert_plan->owner_order = owner;
        config.mtp.depth_policy.demote_zero_accept_rate = rate;
        const auto text = serializeOrchestrationConfig(config);
        auto decoded = deserializeOrchestrationConfig(text);
        EXPECT_EQ(serializeOrchestrationConfig(decoded), text);
        EXPECT_EQ(decoded.mtp.depth_policy.demote_zero_accept_rate, rate);
        EXPECT_NE(decoded.moe_routed_expert_plan.get(), config.moe_routed_expert_plan.get());
        decoded.moe_routed_expert_plan->routed_tiers[0].priority = -9;
        EXPECT_EQ(config.moe_routed_expert_plan->routed_tiers[0].priority, -5);
    }
}

TEST(OrchestrationConfigDocument, PublicConfigRoutePreservesApplyAndAllowsExplicitRequestOverrides)
{
    auto config = example(DeviceType::ROCm);
    // A production compiler saves the normalized domain inventory. Raw
    // declarative input can legitimately gain its derived dense-role views.
    ASSERT_TRUE(normalizeMoERoutedExpertPlacementDomains(config).empty());
    const ConfigFile file(serializeOrchestrationConfig(config));
    std::vector<std::string> arguments{"llaminar2", "--config=" + file.path, "--port", "9812"};
    std::vector<char *> argv;
    for (auto &argument : arguments) argv.push_back(argument.data());
    const auto parsed = OrchestrationConfigParser{}.parseArgs(argv.size(), argv.data());
    config.config_file_path = file.path;
    config.serve_port = 9812;
    const auto differences = Json::diff(Json::parse(serializeOrchestrationConfig(config)),
                                        Json::parse(serializeOrchestrationConfig(parsed)));
    EXPECT_TRUE(differences.empty()) << differences.dump(2);
    EXPECT_TRUE(std::holds_alternative<ApplyOrchestrationRequest>(resolveOrchestrationIntent(parsed)));
}

TEST(OrchestrationConfigDocument, AutomaticConstraintsAndUnresolvedNumaRemainUnresolved)
{
    OrchestrationConfig config;
    config.automatic_planning.only_backends = std::vector{DeviceType::ROCm, DeviceType::CPU};
    config.automatic_planning.only_strategies = std::vector{OrchestrationStrategy::ExpertOverlay};
    config.automatic_planning.prefer_backend = DeviceType::ROCm;
    config.automatic_planning.prefer_strategy = OrchestrationStrategy::ExpertOverlay;
    config.automatic_planning.workload = OrchestrationPlanningWorkload(512, 384);
    config.tp_devices = {GlobalDeviceAddress::cuda(0)};
    config.device_map = {{2, GlobalDeviceAddress::cpu(1, "other-host")}};
    config.device_map_numa_explicit = {{2, true}};
    config.pp_stage_definitions = {{0, "stage-zero", 0, 11}, {1, "stage-one", 12, 39}};
    const auto text = serializeOrchestrationConfig(config);
    const auto parsed = deserializeOrchestrationConfig(text);
    EXPECT_EQ(serializeOrchestrationConfig(parsed), text);
    EXPECT_EQ(parsed.automatic_planning.workload, config.automatic_planning.workload);
    EXPECT_EQ(parsed.tp_devices[0].numa_node, NUMA_NODE_UNKNOWN);
    EXPECT_FALSE(parsed.moe_rebalance.migration_execution_streams.has_value());
    EXPECT_FALSE(parsed.moe_rebalance.migration_cycles_per_wave.has_value());
}

TEST(OrchestrationConfigDocument, WorkloadRoundTripRejectsPartialObjectsAndNumericCoercion)
{
    OrchestrationConfig config;
    config.automatic_planning.workload = OrchestrationPlanningWorkload(512, 384);
    const auto encoded = serializeOrchestrationConfig(config);
    const auto valid = Json::parse(encoded);
    EXPECT_EQ(deserializeOrchestrationConfig(encoded).automatic_planning.workload,
        config.automatic_planning.workload);
    for (const Json &bad : std::vector<Json>{
        Json::object(), Json{{"prompt_tokens", 512}}, Json{{"generation_tokens", 384}},
        Json{{"prompt_tokens", 512}, {"generation_tokens", 384}, {"typo", 1}},
        Json{{"prompt_tokens", 0}, {"generation_tokens", 384}},
        Json{{"prompt_tokens", 512}, {"generation_tokens", -1}},
        Json{{"prompt_tokens", 512.0}, {"generation_tokens", 384}},
        Json{{"prompt_tokens", "512"}, {"generation_tokens", 384}},
        Json{{"prompt_tokens", 512}, {"generation_tokens", 2147483648LL}}, Json::array({512, 384})})
    {
        auto malformed = valid;
        malformed["configuration"]["automatic_planning"]["workload"] = bad;
        EXPECT_THROW((void)deserializeOrchestrationConfig(malformed.dump()), std::invalid_argument) << bad;
    }
}

TEST(OrchestrationConfigDocument, EveryMissingOrExtraObjectFieldIsRejected)
{
    const Json document = Json::parse(serializeOrchestrationConfig(example(DeviceType::CUDA)));
    // Sweep the entire nested document, including every field in MTP, prefix,
    // domains and movement. No read-side default may conceal an omitted field.
    const auto visit = [&](const auto &self, const Json &node, const Json::json_pointer &path) -> void
    {
        if (node.is_object())
        {
            auto extra = document;
            extra[path]["typo"] = true;
            EXPECT_THROW((void)deserializeOrchestrationConfig(extra.dump()), std::invalid_argument) << path;
            for (const auto &[key, value] : node.items())
            {
                auto missing = document;
                missing[path].erase(key);
                EXPECT_THROW((void)deserializeOrchestrationConfig(missing.dump()), std::invalid_argument) << path << "/" << key;
                self(self, value, path / key);
            }
        }
        else if (node.is_array())
            for (size_t i = 0; i < node.size(); ++i) self(self, node[i], path / std::to_string(i));
    };
    visit(visit, document, Json::json_pointer{});
}

TEST(OrchestrationConfigDocument, BadTypesEnumsOverflowAndDuplicateKeysFailAtInputBoundary)
{
    const Json base = Json::parse(serializeOrchestrationConfig(example(DeviceType::ROCm)));
    for (const auto &[path, value] : std::vector<std::pair<std::string, Json>>{
             {"/schema_version", 1}, {"/schema_version", 2}, {"/schema_version", 3}, {"/schema_version", 4}, {"/schema_version", 6}, {"/schema_version", 5.0}, {"/kind", "memory-report"},
             {"/configuration/mpi_procs", "3"}, {"/configuration/mpi_procs", 1ull << 32},
             {"/configuration/mtp/enabled", 1}, {"/configuration/mtp/draft_tokens", 1.5},
             {"/configuration/mtp/verify_mode", "invented"}, {"/configuration/mtp/verify_mode", 0},
             {"/configuration/max_cpu_memory_mb", -1}, {"/configuration/prefix_cache/ram_budget_bytes", -1},
             {"/configuration/temperature", 1e300}, {"/configuration/hostfile", nullptr},
             {"/configuration/moe_rebalance/migration_transfer_slots", 1ull << 32},
             {"/configuration/moe_routed_expert_plan/replica_cache_capacity/admitted", 99},
             {"/configuration/topology_tree", Json::object()}, {"/configuration/topology_string", "TP(x)"}})
    {
        auto changed = base;
        changed[Json::json_pointer(path)] = value;
        EXPECT_THROW((void)deserializeOrchestrationConfig(changed.dump()), std::invalid_argument) << path << "=" << value;
    }
    const auto duplicated = [](std::string text, const std::string &field)
    {
        const auto start = text.find('"' + field + '"');
        text.insert(start, '"' + field + "\": null, ");
        return text;
    };
    for (const auto *key : {"schema_version", "migration_execution_streams", "hostfile", "demote_zero_accept_rate"})
        EXPECT_THROW((void)deserializeOrchestrationConfig(duplicated(base.dump(), key)), std::invalid_argument);
    EXPECT_THROW(OrchestrationConfigParser{}.parseYamlString("{broken json"), std::invalid_argument);
}

TEST(OrchestrationConfigDocument, SavedExecutionRanksRetainOrderAndRejectInvalidMembership)
{
    auto config = example(DeviceType::CUDA);
    const Json document = Json::parse(serializeOrchestrationConfig(config));
    const auto loaded = OrchestrationConfigParser{}.parseYamlString(document.dump());
    ASSERT_TRUE(loaded.execution_rank_selection);
    EXPECT_EQ(loaded.execution_rank_selection->discoveryRanks(), (std::vector<int>{4, 2, 1}));
    EXPECT_EQ(loaded.execution_rank_selection->executionRank(4), 0);
    EXPECT_FALSE(loaded.execution_rank_selection->executionRank(0));
    for (const Json &bad : {Json::array(), Json::array({0, 0}), Json::array({-1}),
                           Json::array({1.0}), Json::array({true}), Json::array({1ull << 32}),
                           Json::object(), Json("1,0")})
    {
        auto malformed = document;
        malformed["configuration"]["execution_rank_selection"] = bad;
        EXPECT_THROW((void)deserializeOrchestrationConfig(malformed.dump()), std::invalid_argument) << bad;
    }
    // A selected communicator alone does not supply a device/domain layout.
    // It must not silently turn an unfinished saved plan into an automatic run.
    OrchestrationConfig missing;
    missing.execution_rank_selection = ExecutionRankSelection({1});
    EXPECT_THROW((void)resolveOrchestrationIntent(missing), std::invalid_argument);
    missing.planning_mode = OrchestrationPlanningMode::Apply;
    EXPECT_THROW((void)resolveOrchestrationIntent(missing), std::invalid_argument);
}

TEST(OrchestrationConfigDocument, WriterRejectsInvalidStatesRatherThanSerializingDefaults)
{
    OrchestrationConfig config;
    config.mtp.verify_mode = static_cast<MTPVerifyMode>(127);
    EXPECT_THROW((void)serializeOrchestrationConfig(config), std::invalid_argument);
    config.mtp.verify_mode = MTPVerifyMode::Greedy;
    config.temperature = std::numeric_limits<float>::infinity();
    EXPECT_THROW((void)serializeOrchestrationConfig(config), std::invalid_argument);
    config.temperature = 0.8f;
    config.mtp.depth_policy.demote_zero_accept_rate = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void)serializeOrchestrationConfig(config), std::invalid_argument);
    config.mtp.depth_policy.demote_zero_accept_rate.reset();
    config.topology_tree.emplace();
    EXPECT_THROW((void)serializeOrchestrationConfig(config), std::invalid_argument);
}

TEST(OrchestrationConfigDocument, DocumentCannotBypassProductionActivationAdmission)
{
    auto document = Json::parse(serializeOrchestrationConfig(OrchestrationConfig{}));
    for (const auto *precision : {"fp16", "bf16", "q8_1", "nonsense"})
    {
        document["configuration"]["activation_precision"] = precision;
        EXPECT_THROW(OrchestrationConfigParser{}.parseYamlString(document.dump()), std::invalid_argument);
    }
}

TEST(OrchestrationConfigDocument, SavedPrefillAndDeterministicPoliciesReachTheSameStartupAuthority)
{
    StartupEnvironment restore;
    OrchestrationConfigParser parser;
    std::vector<std::string> args{"llaminar2", "--prefill-max-bucket-size", "1024", "--deterministic"};
    std::vector<char *> argv;
    for (auto &argument : args) argv.push_back(argument.data());
    const auto configured = parser.parseArgs(argv.size(), argv.data());
    ASSERT_EQ(configured.prefill_max_bucket_size, 1024);
    const ConfigFile file(serializeOrchestrationConfig(configured));

    ASSERT_EQ(setenv("LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "64,256", 1), 0);
    ASSERT_EQ(setenv("LLAMINAR_DETERMINISTIC", "0", 1), 0);
    mutableDebugEnv().reload();
    ASSERT_FALSE(debugEnv().gemm.deterministic);
    const auto loaded = parser.parseYamlFile(file.path);
    EXPECT_EQ(serializeOrchestrationConfig(loaded), serializeOrchestrationConfig(configured));
    EXPECT_TRUE(debugEnv().gemm.deterministic);
    EXPECT_EQ(debugEnv().execution.prefill_graph_bucket_sizes, prefillGraphBucketSizes(1024));

    args = {"llaminar2", "--prefill-max-bucket-size", "256", "--config=" + file.path};
    argv.clear();
    for (auto &argument : args) argv.push_back(argument.data());
    const auto overridden = parser.parseArgs(argv.size(), argv.data());
    EXPECT_EQ(overridden.prefill_max_bucket_size, 256);
    EXPECT_EQ(debugEnv().execution.prefill_graph_bucket_sizes, prefillGraphBucketSizes(256));

    const auto yaml = parser.parseYamlString("prefill_max_bucket_size: 768\ndeterministic: true\n");
    EXPECT_EQ(yaml.prefill_max_bucket_size, 768);
    EXPECT_EQ(yaml.temperature, 0.0f);
    EXPECT_EQ(debugEnv().execution.prefill_graph_bucket_sizes, prefillGraphBucketSizes(768));
}

TEST(OrchestrationConfigDocument, InvalidPrefillDocumentCannotPublishKernelPolicy)
{
    StartupEnvironment restore;
    ASSERT_EQ(setenv("LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "64,256", 1), 0);
    ASSERT_EQ(setenv("LLAMINAR_DETERMINISTIC", "0", 1), 0);
    mutableDebugEnv().reload();
    OrchestrationConfig config;
    config.deterministic = true;
    for (const int cap : {0, -1})
    {
        config.prefill_max_bucket_size = cap;
        EXPECT_THROW(OrchestrationConfigParser{}.parseYamlString(serializeOrchestrationConfig(config)), std::invalid_argument);
        EXPECT_FALSE(debugEnv().gemm.deterministic);
        EXPECT_STREQ(std::getenv("LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES"), "64,256");
    }
}
