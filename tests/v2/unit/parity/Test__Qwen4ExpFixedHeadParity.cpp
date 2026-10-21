/**
 * @file Test__Qwen4ExpFixedHeadParity.cpp
 * @brief Device-free proof of exact fixed-head model test requests and exports.
 *
 * Real hardware/model qualification uses the ordinary production parity body.
 * This inventory test cannot manufacture admission or numerical evidence. It
 * checks that a fixed GPU expert budget, ordered collective membership, source
 * identity and complete checkpoint obligations survive public CLI projection.
 */
#include "integration/parity/qwen4exp/Qwen4ExpFixedHeadParityCases.h"
#include "integration/parity/qwen4exp/Qwen4ExpFixedHeadLookupEvidence.h"
#include "integration/parity/ModelParityRuntimeExport.h"
#include "config/OrchestrationConfigParser.h"
#include <gtest/gtest.h>
#include <map>
#include <optional>
#include <set>
#include <tuple>

namespace llaminar2::test::parity
{
namespace
{
/** @return Public server configuration parsed from the exact case's exported argv. */
OrchestrationConfig parseFixedHeadCase(const ModelParityCase &cell)
{
    auto args = modelParityServerArguments(cell);
    args.insert(args.begin(), "llaminar2");
    std::vector<char *> argv;
    for (auto &arg : args) argv.push_back(arg.data());
    return OrchestrationConfigParser{}.parseArgs(static_cast<int>(argv.size()), argv.data());
}

/** @return Independent TP2 example: live3 then empty, eight owned90B heads per peer. */
std::vector<PerfStatRecord> fixedHeadEvidenceExample()
{
    std::vector<PerfStatRecord> records;
    // Calls include the empty transaction; native receipts do not include its
    // admitted but inactive capacity. Values are explicit oracle constants.
    constexpr std::array<double, 6> values{2, 24, 2160, 0, 16, 2160};
    for (const auto device : {DeviceId::cuda(0), DeviceId::cuda(1)})
        for (std::size_t i = 0; i < values.size(); ++i)
            records.push_back({.kind = PerfStatRecord::Kind::Counter,
                .domain = "immutable_lookup", .name = std::string(qwen4exp::kFixedHeadLookupCounters[i]),
                .phase = "generation_terminal", .device = device.toString(),
                .tags = {{"storage", "vram"}, {"transport", "native_fixed_head"}},
                .count = 1, .value = values[i]});
    return records;
}
}

/** @test Every added case preserves the same strict mathematical and lifecycle contract. */
TEST(Qwen4ExpFixedHeadParity, CanonicalCasesRetainEveryCheckpointAndOriginalFamilies)
{
    const auto &original = qwen4exp::qwen4ExpProductionCases();
    const auto &adaptive = qwen4exp::qwen4ExpHostAdaptiveProductionCases();
    const auto &fixed = qwen4exp::qwen4ExpFixedHeadProductionCases();
    ASSERT_EQ(original.size(), 20u);
    ASSERT_EQ(adaptive.size(), 20u);
    ASSERT_EQ(fixed.size(), 64u);
    std::set<std::string> names;
    for (const auto &family : {&original, &adaptive})
        for (const auto &cell : *family) ASSERT_TRUE(names.insert(cell.testName()).second);
    std::map<std::string, std::size_t> counts;
    const auto &reference = original.front();
    std::set<std::pair<DeviceType, std::size_t>> geometry;
    for (const auto &cell : fixed)
    {
        ASSERT_TRUE(names.insert(cell.testName()).second);
        ++counts[cell.topology.test_id];
        ASSERT_TRUE(cell.topology.expert_overlay_plan);
        const auto &domain = cell.topology.expert_overlay_plan->domains.front();
        geometry.emplace(domain.participants.front().device_type, domain.participants.size());
        EXPECT_EQ(cell.model.model_path, reference.model.model_path);
        EXPECT_EQ(cell.model.reference_directory, reference.model.reference_directory);
        EXPECT_EQ(cell.model.token_ids, reference.model.token_ids);
        EXPECT_EQ(cell.model.transformer_layers, 48);
        EXPECT_EQ(cell.model.decode_steps, 2);
        EXPECT_EQ(cell.model.max_seq_len, reference.model.max_seq_len);
        EXPECT_EQ(cell.activation_precision, reference.activation_precision);
        EXPECT_EQ(cell.kv_cache_precision, reference.kv_cache_precision);
        EXPECT_EQ(cell.mtp, ModelParityMTP::Off);
        EXPECT_EQ(cell.retained_mtp_draft_capacity, 0);
        EXPECT_EQ(cell.thresholds.early_layers_count, 48);
        EXPECT_EQ(cell.thresholds.min_early_layers_passed, 48);
        EXPECT_EQ(cell.graph_snapshot_policy.required_prefill_snapshot_keys,
                  reference.graph_snapshot_policy.required_prefill_snapshot_keys);
        EXPECT_EQ(cell.graph_snapshot_policy.required_decode_snapshot_keys,
                  reference.graph_snapshot_policy.required_decode_snapshot_keys);
        EXPECT_TRUE(cell.graph_snapshot_policy.require_graph_execution_on_gpu);
        EXPECT_TRUE(cell.graph_snapshot_policy.require_snapshot_publication);
        EXPECT_EQ(cell.prefix_restore_geometry, reference.prefix_restore_geometry);
        EXPECT_EQ(cell.collective_evidence_source, reference.collective_evidence_source);
        EXPECT_FALSE(cell.e2e_certification);
    }
    EXPECT_EQ(counts.size(), 8u);
    for (const auto &[identity, count] : counts) EXPECT_EQ(count, 8u) << identity;
    EXPECT_EQ(geometry, (std::set<std::pair<DeviceType, std::size_t>>{
        {DeviceType::CUDA, 2}, {DeviceType::CUDA, 4}, {DeviceType::CUDA, 8},
        {DeviceType::ROCm, 2}, {DeviceType::ROCm, 4}, {DeviceType::ROCm, 8}}));
}

/** @test User-selected budgets and communicator order survive export without a second flat topology. */
TEST(Qwen4ExpFixedHeadParity, PublicCliPreservesOrderedDomainsBudgetsAndStorage)
{
    for (const auto &cell : qwen4exp::qwen4ExpFixedHeadProductionCases())
    {
        const auto native = cell.makeOrchestrationConfig(cell.model.model_path, 0);
        const auto exported = parseFixedHeadCase(cell);
        ASSERT_TRUE(cell.ngram_table_policy);
        const auto &policy = *cell.ngram_table_policy;
        EXPECT_EQ(native.ngram_table_policy, policy);
        EXPECT_EQ(exported.ngram_table_policy, policy);
        EXPECT_EQ(policy.storage, policy.isFixedHeadSharded()
            ? ImmutableRowStoragePreference::VRAM : ImmutableRowStoragePreference::RAM);
        const auto &authored = cell.topology.expert_overlay_plan->domains;
        // Native projection owns only the typed overlay request. CLI parsing
        // additionally publishes its normalized legacy named-domain inventory;
        // it must preserve the original communicator order, not form a flat TP map.
        EXPECT_TRUE(native.domain_definitions.empty());
        ASSERT_EQ(exported.domain_definitions.size(), authored.size());
        for (std::size_t index = 0; index < authored.size(); ++index)
        {
            const auto &raw = exported.domain_definitions[index];
            const auto &expected = authored[index];
            EXPECT_EQ(raw.name, expected.name);
            EXPECT_EQ(raw.devices, expected.participants);
            EXPECT_EQ(raw.weights, expected.weights);
            EXPECT_EQ(raw.backend, expected.backend);
            // TPScope has no SINGLE value. This existing wrapper projects the
            // CPU singleton to AUTO; the effective typed inventory below must
            // still retain SINGLE from the authoritative overlay declaration.
            EXPECT_EQ(expected.scope, index == 0 ? ExecutionDomainScope::RANK_LOCAL : ExecutionDomainScope::SINGLE);
            EXPECT_EQ(raw.scope, index == 0 ? TPScope::RANK_LOCAL : TPScope::AUTO);
            EXPECT_EQ(raw.owner_rank, std::optional<int>(expected.owner_rank));
            EXPECT_EQ(raw.explicit_ranks, expected.world_ranks);
            EXPECT_EQ(raw.routed_compute_policy, expected.routed_compute_policy);
            EXPECT_EQ(raw.routed_phase_policy, expected.routed_phase_policy);
            EXPECT_EQ(raw.routed_decode_assignment_policy, expected.routed_decode_assignment_policy);
            EXPECT_EQ(raw.routed_prefill_assignment_policy, expected.routed_prefill_assignment_policy);
        }
        for (const auto *config : {&native, &exported})
        {
            ASSERT_TRUE(config->moe_routed_expert_plan);
            EXPECT_TRUE(config->pp_stage_definitions.empty());
            EXPECT_TRUE(config->tp_devices.empty());
            const auto &plan = *config->moe_routed_expert_plan;
            ASSERT_EQ(plan.domains.size(), 2u);
            const auto effective = config->executionDomainDefinitions();
            ASSERT_EQ(effective.size(), authored.size());
            for (std::size_t index = 0; index < authored.size(); ++index)
            {
                const auto &actual = effective[index];
                const auto &expected = authored[index];
                EXPECT_EQ(actual.name, expected.name);
                EXPECT_EQ(actual.participants, expected.participants);
                EXPECT_EQ(actual.weights, expected.weights);
                EXPECT_EQ(actual.backend, expected.backend);
                EXPECT_EQ(actual.scope, expected.scope);
                EXPECT_EQ(actual.owner_rank, std::optional<int>(expected.owner_rank));
                EXPECT_EQ(actual.ranks, expected.world_ranks);
                EXPECT_EQ(actual.routed_compute_policy, expected.routed_compute_policy);
                EXPECT_EQ(actual.routed_phase_policy, expected.routed_phase_policy);
                EXPECT_EQ(actual.routed_decode_assignment_policy, expected.routed_decode_assignment_policy);
                EXPECT_EQ(actual.routed_prefill_assignment_policy, expected.routed_prefill_assignment_policy);
            }
            const auto &domain = plan.domains.front();
            EXPECT_EQ(domain.scope, ExecutionDomainScope::RANK_LOCAL);
            EXPECT_EQ(domain.owner_rank, 0);
            EXPECT_EQ(domain.participants, cell.topology.expert_overlay_plan->domains.front().participants);
            EXPECT_EQ(domain.weights, cell.topology.expert_overlay_plan->domains.front().weights);
            EXPECT_EQ(plan.continuation_domain, domain.name);
            EXPECT_EQ(plan.base_model_domain, domain.name);
            EXPECT_EQ(plan.shared_expert_domain, domain.name);
            EXPECT_EQ(plan.continuation_domain_spec.logical_root_participant, 0);
            EXPECT_EQ(plan.continuation_domain_spec.effectiveDensePolicy(), DenseParallelPolicy::TensorParallel);
            EXPECT_EQ(plan.domains.back().participants, (std::vector{GlobalDeviceAddress::cpu(0)}));
            ASSERT_EQ(plan.routed_tiers.size(), 2u);
            EXPECT_EQ(plan.routed_tiers.front().domain, domain.name);
            EXPECT_EQ(plan.routed_tiers.front().max_experts_per_layer, 8);
            EXPECT_EQ(plan.routed_tiers.front().memory_budget_bytes, 0u);
            EXPECT_EQ(plan.routed_tiers.back().max_experts_per_layer, 0);
        }
    }
}

/** @test The only differences in a paired table-policy case are the requested source placement and name. */
TEST(Qwen4ExpFixedHeadParity, RamControlAndShardsHaveMatchedHigherPriorityRequests)
{
    std::map<std::string, std::vector<const ModelParityCase *>> pairs;
    for (const auto &cell : qwen4exp::qwen4ExpFixedHeadProductionCases())
    {
        auto control = cell;
        control.ngram_table_policy = ImmutableRowTablePolicy{.storage = ImmutableRowStoragePreference::RAM};
        pairs[control.testName()].push_back(&cell);
    }
    ASSERT_EQ(pairs.size(), 32u);
    for (const auto &[name, cells] : pairs)
    {
        ASSERT_EQ(cells.size(), 2u) << name;
        ASSERT_TRUE(cells[0]->ngram_table_policy);
        ASSERT_TRUE(cells[1]->ngram_table_policy);
        EXPECT_NE(cells[0]->ngram_table_policy->isFixedHeadSharded(), cells[1]->ngram_table_policy->isFixedHeadSharded());
        auto a = *cells[0];
        auto b = *cells[1];
        a.ngram_table_policy = b.ngram_table_policy = ImmutableRowTablePolicy{.storage = ImmutableRowStoragePreference::RAM};
        EXPECT_EQ(modelParityServerArguments(a), modelParityServerArguments(b));
    }
}

/** @test Invalid membership is rejected before source loading, while reversed order remains meaningful. */
TEST(Qwen4ExpFixedHeadParity, RejectsUnsupportedMembershipWithoutSortingValidOrder)
{
    const std::array two{0, 1};
    const std::array reversed{1, 0};
    const std::array one{0};
    const std::array three{0, 1, 2};
    const std::array duplicate{0, 0};
    const std::array negative{0, -1};
    EXPECT_THROW(qwen4exp::qwen4ExpFixedHeadParityTopology(DeviceType::CPU, two), std::invalid_argument);
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        EXPECT_THROW(qwen4exp::qwen4ExpFixedHeadParityTopology(backend, one), std::invalid_argument);
        EXPECT_THROW(qwen4exp::qwen4ExpFixedHeadParityTopology(backend, three), std::invalid_argument);
        EXPECT_THROW(qwen4exp::qwen4ExpFixedHeadParityTopology(backend, duplicate), std::invalid_argument);
        EXPECT_THROW(qwen4exp::qwen4ExpFixedHeadParityTopology(backend, negative), std::invalid_argument);
        const auto ordinary = qwen4exp::qwen4ExpFixedHeadParityTopology(backend, two);
        const auto reverse = qwen4exp::qwen4ExpFixedHeadParityTopology(backend, reversed);
        EXPECT_NE(ordinary.test_id, reverse.test_id);
        const auto &a = ordinary.expert_overlay_plan->domains.front().participants;
        const auto &b = reverse.expert_overlay_plan->domains.front().participants;
        ASSERT_EQ(a.size(), 2u);
        ASSERT_EQ(b.size(), 2u);
        EXPECT_EQ(a.front(), b.back());
        EXPECT_EQ(a.back(), b.front());
    }
}

/** @test Actual partial and empty useful-byte receipts pass, independent of allocation capacity. */
TEST(Qwen4ExpFixedHeadParity, TerminalEvidenceAcceptsCompleteLiveContributions)
{
    const std::array devices{DeviceId::cuda(0), DeviceId::cuda(1)};
    const auto records = fixedHeadEvidenceExample();
    EXPECT_NO_THROW(qwen4exp::requireQwen4ExpFixedHeadLookupEvidence(records, devices, 90));
    const std::array reverse{devices[1], devices[0]};
    EXPECT_NO_THROW(qwen4exp::requireQwen4ExpFixedHeadLookupEvidence(records, reverse, 90));
}

/** @test Missing or disabled counters, including a missing zero error count, cannot prove execution. */
TEST(Qwen4ExpFixedHeadParity, TerminalEvidenceRejectsMissingParticipantOrCounter)
{
    const std::array devices{DeviceId::cuda(0), DeviceId::cuda(1)};
    const auto good = fixedHeadEvidenceExample();
    EXPECT_THROW(qwen4exp::requireQwen4ExpFixedHeadLookupEvidence({}, devices, 90), std::invalid_argument);
    for (std::size_t missing = 0; missing < good.size(); ++missing)
    {
        auto bad = good;
        bad.erase(bad.begin() + missing);
        EXPECT_THROW(qwen4exp::requireQwen4ExpFixedHeadLookupEvidence(bad, devices, 90), std::invalid_argument);
    }
    auto peer = good;
    peer.resize(6);
    EXPECT_THROW(qwen4exp::requireQwen4ExpFixedHeadLookupEvidence(peer, devices, 90), std::invalid_argument);
}

/** @test Capacity-sized traffic, lost empty headers and owner disagreement fail the strict model gate. */
TEST(Qwen4ExpFixedHeadParity, TerminalEvidenceRejectsPaddingFailuresAndUnequalWindows)
{
    const std::array devices{DeviceId::cuda(0), DeviceId::cuda(1)};
    const auto good = fixedHeadEvidenceExample();
    for (const auto [index, value] : std::array<std::pair<std::size_t, double>, 5>{
             {{5, 2880}, {4, 8}, {3, 1}, {2, 2880}, {1, 32}}})
    {
        auto bad = good;
        bad[index].value = value;
        EXPECT_THROW(qwen4exp::requireQwen4ExpFixedHeadLookupEvidence(bad, devices, 90), std::invalid_argument);
    }
    auto double_window = good;
    for (std::size_t i = 0; i < 6; ++i) double_window[i].value *= 2;
    EXPECT_THROW(qwen4exp::requireQwen4ExpFixedHeadLookupEvidence(double_window, devices, 90), std::invalid_argument);
    // Internally consistent byte counters still cannot describe25 owned IDs:
    // each TP2 token contributes eight complete heads on every participant.
    // Removing the token/head divisibility guard makes this negative pass.
    auto fractional_token = good;
    for (const std::size_t peer : {0u, 6u})
    {
        fractional_token[peer + 1].value = 25;
        fractional_token[peer + 2].value = 2250;
        fractional_token[peer + 5].value = 2250;
    }
    EXPECT_THROW(qwen4exp::requireQwen4ExpFixedHeadLookupEvidence(fractional_token, devices, 90), std::invalid_argument);
}

/** @test Foreign/hosted/setup evidence and noninteger observations cannot masquerade as native terminal work. */
TEST(Qwen4ExpFixedHeadParity, TerminalEvidenceRejectsWrongProvenanceAndMalformedValues)
{
    const std::array devices{DeviceId::cuda(0), DeviceId::cuda(1)};
    const auto good = fixedHeadEvidenceExample();
    auto wrong_device = good;
    wrong_device.front().device = DeviceId::rocm(0).toString();
    EXPECT_THROW(qwen4exp::requireQwen4ExpFixedHeadLookupEvidence(wrong_device, devices, 90), std::invalid_argument);
    auto setup = good;
    setup.front().phase = "graph_preparation";
    EXPECT_THROW(qwen4exp::requireQwen4ExpFixedHeadLookupEvidence(setup, devices, 90), std::invalid_argument);
    auto hosted = good;
    hosted.front().tags["transport"] = "mapped_channel";
    EXPECT_THROW(qwen4exp::requireQwen4ExpFixedHeadLookupEvidence(hosted, devices, 90), std::invalid_argument);
    for (const double value : {-1.0, 0.5, std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN(), 9007199254740992.0})
    {
        auto bad = good;
        bad.front().value = value;
        EXPECT_THROW(qwen4exp::requireQwen4ExpFixedHeadLookupEvidence(bad, devices, 90), std::invalid_argument);
    }
    const std::array duplicate{devices[0], devices[0]};
    EXPECT_THROW(qwen4exp::requireQwen4ExpFixedHeadLookupEvidence(good, duplicate, 90), std::invalid_argument);
    EXPECT_THROW(qwen4exp::requireQwen4ExpFixedHeadLookupEvidence(good, devices, 0), std::invalid_argument);
}
/** @test Explicit live3/empty receipts preserve every supported native cohort, with no backend discovery. */
TEST(Qwen4ExpFixedHeadParity, TerminalEvidenceCoversEveryNativeBackendAndDegree)
{
    /**
     * @brief Author independent metadata records without invoking the production validator or a backend.
     * @param backend Explicit CUDA or ROCm device identity. @param degree Number of independent owners.
     * @param ids Explicit expected live-ID total. @param native Explicit expected source/payload90B extent.
     * @return Complete owners and independently authored six-counter records.
     */
    const auto authored = [](DeviceType backend, int degree, double ids, double native)
    {
        std::vector<DeviceId> owners;
        std::vector<PerfStatRecord> records;
        const std::array<double, 6> values{2, ids, native, 0, 16, native};
        for (int participant = 0; participant < degree; ++participant)
        {
            const DeviceId device{backend, participant};
            owners.push_back(device);
            for (std::size_t field = 0; field < values.size(); ++field)
                records.push_back({.kind = PerfStatRecord::Kind::Counter,
                    .domain = "immutable_lookup", .name = std::string(qwen4exp::kFixedHeadLookupCounters[field]),
                    .phase = "generation_terminal", .device = device.toString(),
                    .tags = {{"storage", "vram"}, {"transport", "native_fixed_head"}},
                    .count = 1, .value = values[field]});
        }
        return std::pair{std::move(owners), std::move(records)};
    };
    struct Expected { int degree; double ids; double native; };
    // Independent original-query oracle: live3 then empty,16 heads and90B/head.
    // No expected value is derived from the helper whose equations are tested.
    constexpr std::array expected{Expected{2, 24, 2160}, Expected{4, 12, 1080}, Expected{8, 6, 540}};
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        for (const auto &entry : expected)
        {
            SCOPED_TRACE(static_cast<int>(backend));
            SCOPED_TRACE(entry.degree);
            auto [owners, records] = authored(backend, entry.degree, entry.ids, entry.native);
            EXPECT_NO_THROW(qwen4exp::requireQwen4ExpFixedHeadLookupEvidence(records, owners, 90));
            // Mutate EVERY owner consistently so byte equations and equality
            // still hold. Only a fractional token/head extent invalidates it.
            for (int participant = 0; participant < entry.degree; ++participant)
            {
                const auto at = static_cast<std::size_t>(participant) * 6;
                records[at + 1].value = entry.ids + 1;
                records[at + 2].value = entry.native + 90;
                records[at + 5].value = entry.native + 90;
            }
            EXPECT_THROW(qwen4exp::requireQwen4ExpFixedHeadLookupEvidence(records, owners, 90), std::invalid_argument);
        }
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        // All other equations are internally valid: a permissive degree guard
        // would accept these totals (including15 divisible by floor(16/3)).
        for (const auto &entry : std::array{Expected{1, 48, 4320}, Expected{3, 15, 1350}})
        {
            auto [owners, records] = authored(backend, entry.degree, entry.ids, entry.native);
            EXPECT_THROW(qwen4exp::requireQwen4ExpFixedHeadLookupEvidence(records, owners, 90), std::invalid_argument);
        }
    }
    auto [mixedOwners, mixedRecords] = authored(DeviceType::CUDA, 2, 24, 2160);
    mixedOwners[1] = DeviceId::rocm(1);
    for (std::size_t field = 6; field < 12; ++field) mixedRecords[field].device = mixedOwners[1].toString();
    // Record owners match the declared mixed list; foreign-record rejection
    // cannot accidentally stand in for the homogeneous-cohort requirement.
    EXPECT_THROW(qwen4exp::requireQwen4ExpFixedHeadLookupEvidence(mixedRecords, mixedOwners, 90), std::invalid_argument);
}
} // namespace llaminar2::test::parity
