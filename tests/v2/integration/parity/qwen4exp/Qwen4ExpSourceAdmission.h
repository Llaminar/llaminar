/**
 * @file Qwen4ExpSourceAdmission.h
 * @brief Gold-free original-source QFN serving admission through the real runner.
 *
 * This test-only observer consumes canonical ModelParityCase declarations and
 * the production startup, prepared-weight, lookup-owner and PMA interfaces. It
 * never resolves a reference directory, enables diagnostic snapshots, executes
 * inference or reads/hashes tensor payloads. Whole, fixed-head and adaptive
 * layouts retain distinct owner contracts; planned capacity is not live bytes.
 * The runner and model certificate remain alive throughout every observation.
 * Active policies authenticate real retained sidecar/forward setup; they do not
 * claim generation-controller execution or numerical sidecar certification.
 */
#pragma once
#include "Qwen4ExpProductionParityCases.h"
#include "Qwen4ExpSourceAdmissionCases.h"
#include "Qwen4ExpSourceAdmissionEvidence.h"
#include "Qwen4ExpSourceAdmissionObservation.h"
#include "execution/runner/ModelContextRetirement.h"
#include "backends/BackendManager.h"
#include "backends/ComputeBackend.h"
#include "backends/HardwareInventory.h"
#include "config/OrchestrationStartupPolicy.h"
#include "execution/qwen4exp/ImmutableLookupExecutionBoundary.h"
#include "loaders/AdaptiveImmutableLookupGeometry.h"
#include "kernels/qwen4exp/PooledImmutableSourcePreparedStorage.h"
#include "loaders/ModelContext.h"
#include "loaders/PreparedWeightStore.h"
#include "models/qwen4exp/Qwen4ExpSidecarManifest.h"
#include "execution/mtp/MTPWeightManifest.h"
#include "utils/PerfStatsCollector.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "utils/TestOrchestrationHelper.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <set>

namespace llaminar2::test::parity::qwen4exp
{
/**
 * @brief Compare a live retained owner's bytes with the one PMA attestation.
 * @param memory Existing model admission/materialization authority.
 * @param allocator Exact physical owner, including its rank and device.
 * @param owner Typed PMA owner category, never an anonymous reserve.
 * @param live Bytes issued by the retained native owner, not a BOM estimate.
 */
inline void expectQwen4ExpSourceLease(const PhysicalMemoryAuthority &memory,
    PhysicalMemoryAllocatorIdentity allocator, PhysicalMemoryOwner owner, std::size_t live)
{
    const auto attestation = memory.rankAttestation();
    const auto line = std::find_if(attestation.begin(), attestation.end(), [&](const auto &row) {
        return row.identity == allocator && row.owner == owner;
    });
    ASSERT_NE(line, attestation.end());
    ASSERT_LE(line->materialized_new_bytes,
              std::numeric_limits<std::size_t>::max() - line->adopted_resident_bytes);
    EXPECT_EQ(line->materialized_new_bytes + line->adopted_resident_bytes, live);
}

/**
 * @brief Require every published domain field in the same canonical representation.
 * @param actual Observed publication or typed effective domain.
 * @param expected Corresponding authored/resolved production definition.
 */
inline void expectQwen4ExpSourceDomain(const ExecutionDomainDefinition &actual, const ExecutionDomainDefinition &expected)
{
    EXPECT_EQ(actual.name, expected.name);
    EXPECT_EQ(actual.participants, expected.participants);
    EXPECT_EQ(actual.weights, expected.weights);
    EXPECT_EQ(actual.backend, expected.backend);
    EXPECT_EQ(actual.scope, expected.scope);
    EXPECT_EQ(actual.owner_rank, expected.owner_rank);
    EXPECT_EQ(actual.ranks, expected.ranks);
    EXPECT_EQ(actual.routed_compute_policy, expected.routed_compute_policy);
    EXPECT_EQ(actual.routed_phase_policy, expected.routed_phase_policy);
    EXPECT_EQ(actual.routed_decode_assignment_policy, expected.routed_decode_assignment_policy);
    EXPECT_EQ(actual.routed_prefill_assignment_policy, expected.routed_prefill_assignment_policy);
}

/**
 * @brief Authenticate the published named inventory through canonical overlay installation.
 * @param authored Frozen original request, including typed domain authorities without duplicate names.
 * @param resolved Runner configuration after inventory binding and domain publication.
 *
 * Raw domain_definitions is a publication surface, not the complete authored
 * topology. The canonical union includes routed/dense domains. Reinstalling a
 * copy of the actual bound plan validates original intent, exact selectors and
 * rank ownership; complete field comparisons then reject missing/foreign aliases.
 */
inline void expectQwen4ExpSourceDomains(OrchestrationConfig authored, const OrchestrationConfig &resolved)
{
    const auto requested = authored.executionDomainDefinitions();
    ASSERT_EQ(resolved.domain_definitions.size(), requested.size());
    if (authored.moe_routed_expert_plan)
    {
        ASSERT_NE(resolved.moe_routed_expert_plan, nullptr);
        const auto &intent = *authored.moe_routed_expert_plan;
        const auto &actual = *resolved.moe_routed_expert_plan;
        EXPECT_EQ(actual.continuation_domain, intent.continuation_domain);
        EXPECT_EQ(actual.base_model_domain, intent.base_model_domain);
        EXPECT_EQ(actual.shared_expert_domain, intent.shared_expert_domain);
        EXPECT_EQ(actual.continuation_domain_spec.effectiveDensePolicy(),
                  intent.continuation_domain_spec.effectiveDensePolicy());
        EXPECT_EQ(actual.continuation_domain_spec.hidden_layout, intent.continuation_domain_spec.hidden_layout);
        EXPECT_EQ(actual.owner_order, intent.owner_order);
        EXPECT_EQ(actual.residency_policy, intent.residency_policy);
        ASSERT_EQ(actual.routed_tiers.size(), intent.routed_tiers.size());
        for (std::size_t i = 0; i < intent.routed_tiers.size(); ++i)
        {
            EXPECT_EQ(actual.routed_tiers[i].name, intent.routed_tiers[i].name);
            EXPECT_EQ(actual.routed_tiers[i].domain, intent.routed_tiers[i].domain);
            EXPECT_EQ(actual.routed_tiers[i].priority, intent.routed_tiers[i].priority);
            EXPECT_EQ(actual.routed_tiers[i].max_experts_per_layer, intent.routed_tiers[i].max_experts_per_layer);
            EXPECT_EQ(actual.routed_tiers[i].memory_budget_bytes, intent.routed_tiers[i].memory_budget_bytes);
        }
        const auto errors = installResolvedMoEExpertOverlayPlan(authored,
            std::make_shared<MoERoutedExpertPlacementPlan>(actual),
            MoEExpertOverlayPlanInstallOrigin::UserDeclaredDomains);
        ASSERT_TRUE(errors.empty()) << (errors.empty() ? "" : errors.front());
    }
    const auto expected = authored.executionDomainDefinitions();
    ASSERT_EQ(expected.size(), requested.size());
    const auto effective = resolved.executionDomainDefinitions();
    ASSERT_EQ(effective.size(), requested.size());
    std::set<std::string> observed;
    for (const auto &published : resolved.domain_definitions)
    {
        const auto actual = published.toExecutionDomainDefinition();
        ASSERT_TRUE(observed.insert(actual.name).second) << actual.name;
        const auto found = std::find_if(expected.begin(), expected.end(), [&](const auto &domain) {
            return domain.name == actual.name;
        });
        ASSERT_NE(found, expected.end()) << actual.name;
        // The legacy TPScope wrapper cannot represent SINGLE and intentionally
        // publishes AUTO. Authenticate that exact projection independently from
        // the effective typed authority; never relax SINGLE to arbitrary scope.
        expectQwen4ExpSourceDomain(actual, DomainDefinition::fromExecutionDomainDefinition(*found)
            .toExecutionDomainDefinition());
        const auto typed = std::find_if(effective.begin(), effective.end(), [&](const auto &domain) {
            return domain.name == actual.name;
        });
        ASSERT_NE(typed, effective.end()) << actual.name;
        expectQwen4ExpSourceDomain(*typed, *found);
    }
}


/**
 * @brief Authenticate a GPU-adaptive source, cache roots and their actual leases.
 * @param cell Exact canonical request, including zero/positive cache ceilings.
 * @param runner Successfully initialized real production runner.
 *
 * This observes existing source metadata and retained native owners only. It
 * does not perform a lookup or download a cache tag, row or mutable model state.
 */
inline void expectQwen4ExpSourceGPUAdaptiveOwners(const ModelParityCase &cell, IOrchestrationRunner &runner)
{
    ASSERT_TRUE(cell.ngram_table_policy);ASSERT_TRUE(cell.ngram_table_policy->isGPUAdaptive());
    const auto retained=runner.modelContextReuseContract();ASSERT_TRUE(retained);
    const auto *context=retained->context.get();ASSERT_NE(context,nullptr);
    EXPECT_EQ(runner.modelContextForDiagnostics(),context);
    const auto placement=context->resolvedImmutableRowTablePlacement();ASSERT_TRUE(placement);
    ASSERT_TRUE(placement->isGPUAdaptive());placement->requireRequestedPolicy(*cell.ngram_table_policy);
    const auto &layout=placement->gpuAdaptiveLayout();const auto &request=placement->gpuAdaptiveRequest();
    ASSERT_EQ(layout.consumers().size(),1u);ASSERT_EQ(layout.sourceServices().size(),1u);
    ASSERT_EQ(layout.owners().size(),2u);ASSERT_EQ(placement->allocators().size(),2u);
    const auto gpu=layout.consumers().front(),host=layout.sourceServices().front().allocator;
    ASSERT_TRUE(gpu.device.is_gpu());ASSERT_TRUE(host.device.is_cpu());EXPECT_EQ(gpu.world_rank,host.world_rank);
    EXPECT_EQ(placement->lookupForConsumer(gpu),ImmutableRowLookupExecution::AdaptiveGPUCache);
    const auto boundary=context->adaptiveImmutableLookupBoundary(gpu.device);ASSERT_TRUE(boundary);
    ASSERT_TRUE(boundary->isPooled());EXPECT_EQ(boundary->device(),gpu.device);
    EXPECT_EQ(context->adaptiveImmutableLookupBoundary(gpu.device).get(),boundary.get());
    const auto provider=context->hostImmutableRowProvider();ASSERT_TRUE(provider);
    EXPECT_EQ(provider->mode(),HostImmutableRowMode::AdaptiveRAMSSD);
    EXPECT_EQ(provider->subordinateRequester(),std::optional{gpu});
    EXPECT_EQ(provider->maximumLiveRows(),placement->gpuAdaptiveMaximumLiveRows());
    // Authenticate source coordinates against the public original GGUF
    // directory and ordered artifact set. Do not widen ModelContext's
    // private materialization extractor merely for fixture convenience.
    ASSERT_NO_THROW(requireQwen4ExpSourceTableIdentity(
        context->model(), context->artifactPaths(), provider->source()));
    EXPECT_EQ(provider->source().rowBytes(),placement->extent().rowBytes());
    EXPECT_EQ(provider->source().rows,placement->extent().rows());
    EXPECT_EQ(provider->source().row_elements,placement->gpuAdaptiveRowElements());
    ASSERT_NO_THROW(provider->authenticateSource());
    EXPECT_EQ(context->residentImmutableRowTable(gpu.device),nullptr);
    EXPECT_EQ(context->residentImmutableRowTableBytes(gpu.device),0u);
    const auto geometry=AdaptiveImmutableLookupGeometry::fromLayout(layout,provider->source(),request);
    const auto gpu_slots=geometry.physical().owners().front().slots;
    const auto ram_slots=geometry.host().slots();
    EXPECT_LE(gpu_slots,request.maximum_gpu_cache_rows);EXPECT_LE(ram_slots,request.host.maximum_cache_rows);
    if(request.maximum_gpu_cache_rows && request.gpu_cache_physical_byte_limit)
    {
        EXPECT_GT(gpu_slots,0u) << "Positive GPU-cache cell cannot certify a zero optional grant";
    }
    else { EXPECT_EQ(gpu_slots,0u); }
    if(request.host.maximum_cache_rows && request.host.cache_physical_byte_limit)
    {
        EXPECT_GT(ram_slots,0u) << "Positive RAM-cache cell cannot certify a zero optional grant";
    }
    else { EXPECT_EQ(ram_slots,0u); }
    const auto storage=boundary->pooledStorage();
    const std::array roots{storage->owner()->roots(),storage->coordinator()->roots(),storage->roots()};
    EXPECT_EQ(bool(roots[0].cache),gpu_slots!=0);EXPECT_EQ(bool(roots[1].cache),gpu_slots!=0);
    for(const auto& group:roots)for(const auto& root:{group.cache,group.frame,group.native})if(root){
        EXPECT_TRUE(root->is_raw_data_released());EXPECT_NE(root->gpu_data_ptr(),nullptr);
    }
    const auto gpu_owner=storage->retainedOwnerObservation();
    const auto source_owner=boundary->retainedSourceOwnerObservation();
    const auto provider_owner=provider->retainedOwnerObservation();
    ASSERT_TRUE(retained->physical_memory_authority);
    const auto &memory=retained->physical_memory_authority;
    ASSERT_NO_THROW(storage->requireFor(geometry.physical(),geometry.coordinator(),provider->source(),memory));
    EXPECT_EQ(gpu_owner.authority().get(),memory.get());EXPECT_EQ(source_owner.authority().get(),memory.get());
    EXPECT_EQ(provider_owner.authority().get(),memory.get());
    EXPECT_EQ(gpu_owner.allocator(),gpu);EXPECT_EQ(source_owner.allocator(),host);
    EXPECT_EQ(source_owner.hostPlacement(),placement->hostPlacement());
    EXPECT_EQ(source_owner.tableStorageBytes(),provider_owner.tableStorageBytes());
    EXPECT_GE(source_owner.lookupStagingBytes(),provider_owner.lookupStagingBytes());
    EXPECT_EQ(source_owner.lookupStagingBytes()-provider_owner.lookupStagingBytes(),
        ::llaminar2::qwen4exp::ImmutableLookupService::hostBytes(geometry.channelLayout()));
    EXPECT_EQ(gpu_owner.tableStorageBytes(),geometry.gpuCacheBytes());
    EXPECT_EQ(gpu_owner.lookupStagingBytes(),geometry.mandatoryDeviceBytes());
    EXPECT_LE(geometry.gpuSlotDependentBytes(),request.gpu_cache_physical_byte_limit);
    EXPECT_LE(geometry.host().storageBytes(),request.host.cache_physical_byte_limit);
    ASSERT_NO_FATAL_FAILURE(expectQwen4ExpSourceLease(*memory, gpu,
        PhysicalMemoryOwner::ImmutableTableStorage, gpu_owner.tableStorageBytes()));
    ASSERT_NO_FATAL_FAILURE(expectQwen4ExpSourceLease(*memory, gpu,
        PhysicalMemoryOwner::ImmutableLookupStaging, gpu_owner.lookupStagingBytes()));
    ASSERT_NO_FATAL_FAILURE(expectQwen4ExpSourceLease(*memory, host,
        PhysicalMemoryOwner::ImmutableTableStorage, source_owner.tableStorageBytes()));
    ASSERT_NO_FATAL_FAILURE(expectQwen4ExpSourceLease(*memory, host,
        PhysicalMemoryOwner::ImmutableLookupStaging, source_owner.lookupStagingBytes()));
}


/**
 * @brief Authenticate the selected lookup layout and actual retained owners.
 * @param cell Canonical table policy, without inferred tier substitutions.
 * @param runner Successfully initialized production runner and model authority.
 */
inline void expectQwen4ExpSourceLookup(const ModelParityCase &cell, IOrchestrationRunner &runner)
{
    ASSERT_TRUE(cell.ngram_table_policy);
    const auto reuse = runner.modelContextReuseContract();
    ASSERT_TRUE(reuse);
    const auto context = reuse->context;
    ASSERT_NE(context, nullptr);
    const auto memory = reuse->physical_memory_authority;
    ASSERT_NE(memory, nullptr);
    const auto placement = context->resolvedImmutableRowTablePlacement();
    ASSERT_NE(placement, nullptr);
    ASSERT_NE(runner.executionPlan().runtime.ngram_table_placement, nullptr);
    EXPECT_EQ(*runner.executionPlan().runtime.ngram_table_placement, *placement);
    placement->requireRequestedPolicy(*cell.ngram_table_policy);
    EXPECT_EQ(placement->extent().rows(), 320001536ull);
    EXPECT_EQ(placement->extent().rowBytes(), 90u);
    EXPECT_EQ(placement->extent().nativeBytes(), 28800138240ull);
    if (placement->isGPUAdaptive())
    {
        ASSERT_NO_FATAL_FAILURE(expectQwen4ExpSourceGPUAdaptiveOwners(cell, runner));
        return;
    }
    if (placement->isFixedHeadShards() ||
        (!placement->isHostAdaptive() && placement->storage() == ImmutableRowStorage::VRAM))
    {
        EXPECT_EQ(context->hostImmutableRowProvider(), nullptr);
        std::size_t total = 0;
        for (const auto &allocator : placement->allocators())
        {
            const auto table = context->residentImmutableRowTable(allocator.device);
            ASSERT_NE(table, nullptr);
            EXPECT_EQ(table->device(), allocator.device);
            ASSERT_NO_THROW(requireQwen4ExpSourceTableIdentity(
                context->model(), context->artifactPaths(), table->source()));
            ASSERT_NO_THROW(table->authenticateSource());
            const auto range = placement->isFixedHeadShards()
                ? placement->fixedHeadSourceRangeFor(allocator)
                : ImmutableRowTableSourceRange::whole(placement->extent());
            EXPECT_EQ(table->sourceRange(), range);
            EXPECT_EQ(table->nativeBytes(), range.nativeBytes());
            EXPECT_EQ(table->source().rowBytes(), placement->extent().rowBytes());
            EXPECT_EQ(context->residentImmutableRowTableBytes(allocator.device), table->nativeBytes());
            ASSERT_LE(table->nativeBytes(), std::numeric_limits<std::size_t>::max() - total);
            total += table->nativeBytes();
            ASSERT_NO_FATAL_FAILURE(expectQwen4ExpSourceLease(*memory, allocator,
                PhysicalMemoryOwner::ImmutableTableStorage, table->nativeBytes()));
        }
        if (placement->isFixedHeadShards())
        {
            EXPECT_TRUE(cell.ngram_table_policy->isFixedHeadSharded());
            EXPECT_EQ(total, placement->extent().nativeBytes());
            EXPECT_EQ(placement->allocators().size(), placement->consumers().size());
        }
        return;
    }
    const auto provider = context->hostImmutableRowProvider();
    ASSERT_NE(provider, nullptr);
    ASSERT_EQ(placement->allocators().size(), 1u);
    const auto host = placement->allocators().front();
    EXPECT_TRUE(host.device.is_cpu());
    EXPECT_EQ(provider->source().rowBytes(), placement->extent().rowBytes());
    EXPECT_EQ(provider->source().tensorBytes(), placement->extent().nativeBytes());
    ASSERT_NO_THROW(provider->authenticateSource());
    ASSERT_NO_THROW(requireQwen4ExpSourceTableIdentity(
        context->model(), context->artifactPaths(), provider->source()));
    const auto observed = provider->retainedOwnerObservation();
    EXPECT_EQ(observed.authority().get(), memory.get());
    EXPECT_EQ(observed.allocator(), host);
    EXPECT_EQ(observed.hostPlacement(), placement->hostPlacement());
    if (placement->isHostAdaptive())
    {
        const auto &policy = *cell.ngram_table_policy;
        EXPECT_TRUE(policy.isHostAdaptive());
        EXPECT_EQ(provider->mode(), HostImmutableRowMode::AdaptiveRAMSSD);
        EXPECT_EQ(provider->maximumLiveRows(), placement->hostAdaptiveMaximumLiveRows());
        ASSERT_EQ(placement->hostAdaptiveLayout().owners().size(), 1u);
        const auto slots = placement->hostAdaptiveLayout().owners().front().cache_rows;
        EXPECT_LE(slots, policy.host_adaptive.maximum_cache_rows);
        if (policy.host_adaptive.maximum_cache_rows && policy.host_adaptive.cache_physical_byte_limit)
            EXPECT_GT(slots, 0u) << "Positive-cache admission cannot certify an empty optional grant";
        else EXPECT_EQ(slots, 0u);
        EXPECT_EQ(observed.lookupStagingBytes(), CpuAdaptiveImmutableRowCacheGeometry::fromLayout(
            placement->hostAdaptiveLayout(), host, placement->hostAdaptiveMaximumLiveRows(),
            placement->hostAdaptivePolicy()).executionBytes());
    }
    else
    {
        const auto requested = cell.ngram_table_policy->storage;
        ASSERT_TRUE(requested == ImmutableRowStoragePreference::RAM ||
                    requested == ImmutableRowStoragePreference::MemoryMapped);
        EXPECT_EQ(placement->storage(), requested == ImmutableRowStoragePreference::RAM
            ? ImmutableRowStorage::RAM : ImmutableRowStorage::MemoryMapped);
        EXPECT_EQ(provider->wholeStorage(), placement->storage());
        EXPECT_EQ(observed.lookupStagingBytes(), 0u);
    }
    ASSERT_NO_FATAL_FAILURE(expectQwen4ExpSourceLease(*memory, host,
        PhysicalMemoryOwner::ImmutableTableStorage, observed.tableStorageBytes()));
    // Provider geometry above excludes the mapped lookup channels retained by
    // host-backed GPU consumers. Authenticate the entire canonical PMA owner as
    // live, while retaining exact provider equality for CPU-only placements.
    // Whole providers have no own staging but may still serve those GPU channels.
    const auto attestation = memory->rankAttestation();
    ASSERT_NO_THROW(requireQwen4ExpHostLookupStaging(attestation, host,
        observed.lookupStagingBytes(), placement->consumers()));
    for (const auto &consumer : placement->consumers())
        EXPECT_EQ(context->residentImmutableRowTableBytes(consumer.device), 0u);
}

/**
 * @brief Own one real runner and its optional final prepared-model contract.
 *
 * Observation aliases must be scoped inside the caller's verification callback.
 * Normal and assertion-failure exits then close the actual runner, consume the
 * sole Reusable contract through canonical TransferEngine retirement, and only
 * afterwards unlink disposable prefix files. No custom allocation ledger,
 * device reset, reference data or production lifecycle is introduced.
 */
class Qwen4ExpSourceAdmissionLifetime final
{
public:
    /** @brief Construct the normal runner under one owned cache namespace. @param config Public request. @param cache Owned prefix child. */
    Qwen4ExpSourceAdmissionLifetime(const OrchestrationConfig &config, std::filesystem::path cache)
        : cache_(std::move(cache)), runner_(test::TestOrchestrationHelper::create(config))
    {
        if (!runner_) throw std::runtime_error("QFN source admission factory returned no runner");
    }
    /** @brief Retire remaining owners after a failed assertion without hiding the failure. */
    ~Qwen4ExpSourceAdmissionLifetime()
    {
        try { finish(); }
        catch (const std::exception &error) { ADD_FAILURE() << "Admission final retirement failed: " << error.what(); }
    }
    Qwen4ExpSourceAdmissionLifetime(const Qwen4ExpSourceAdmissionLifetime &) = delete;
    Qwen4ExpSourceAdmissionLifetime &operator=(const Qwen4ExpSourceAdmissionLifetime &) = delete;
    /** @return This fixture's actual runner before its explicit final retirement. */
    IOrchestrationRunner &runner() { return *runner_; }
    /** @return The sole retained model contract; temporary observation aliases must expire before finish(). */
    const std::optional<ModelContextReuseContract> &reuseContract()
    {
        if (!reuse_) reuse_ = runner().modelContextReuseContract();
        return reuse_;
    }
    /** @brief Close runner, final model and owned disk in that order. @throws std::exception On any unproven retirement. */
    void finish()
    {
        if (finished_) return;
        if (runner_)
        {
            runner_->shutdown();
            if (runner_->isInitialized()) throw std::runtime_error("QFN admission runner remains initialized after shutdown");
            runner_.reset();
        }
        if (reuse_ && reuse_->context)
        {
            const auto receipt = retireExclusiveModelContextReuseContract(*reuse_);
            retired_devices_ = receipt.retiredDeviceCount();
        }
        cache_.cleanup();
        finished_ = true;
    }
    /** @return Actual canonical GPU retirement receipts, zero for a host-only admission. */
    std::size_t retiredDeviceCount() const noexcept { return retired_devices_; }
private:
    // Reverse member destruction is also safe if construction or finish fails:
    // runner first, retained model second, disposable directory last.
    Qwen4ExpAdmissionPrefixDirectory cache_;
    std::optional<ModelContextReuseContract> reuse_;
    std::unique_ptr<IOrchestrationRunner> runner_;
    std::size_t retired_devices_ = 0;
    bool finished_ = false;
};

/** @brief Begin immutable setup evidence before initialization.
 * @param cell Existing source policy; Off requires no counter collection.
 * @return Baseline preserving prior cells without global observation reset.
 */
inline std::vector<PerfStatRecord> beginQwen4ExpSourceMTPEvidence(const ModelParityCase &cell)
{
    if (cell.mtp == ModelParityMTP::Off)
        return {};
    if (!PerfStatsCollector::isDomainEnabled("forward_graph"))
        throw std::logic_error("Active source admission requires forward_graph setup-family collection");
    return PerfStatsCollector::snapshot();
}
/** @brief Observe shared source/MTP materialization, including special table cases.
 * @param cell Existing typed source identity.
 * @param authored Original projected policy before planning.
 * @param runner_value Initialized real source runner.
 * @param reuse_value Original retained model/PMA certificate.
 * @param output Owned source evidence directory.
 * @param setup_before Snapshot from before this runner initialized.
 *
 * Authenticates sources, sidecar engines and plan-owned forward setup only;
 * no inference or generation-controller execution occurs here.
 */
inline void expectQwen4ExpSourceMTPMaterialization(const ModelParityCase &cell,
                                                   const OrchestrationConfig &authored,
                                                   IOrchestrationRunner &runner_value,
                                                   const ModelContextReuseContract &reuse_value,
                                                   const std::filesystem::path &output,
                                                   std::span<const PerfStatRecord> setup_before)
{
    auto *runner = &runner_value;
    const auto *reuse = &reuse_value;
    const auto &config = authored;
    const bool active_mtp = cell.mtp != ModelParityMTP::Off;
    ASSERT_NE(reuse->context, nullptr);
    ASSERT_NE(reuse->physical_memory_authority, nullptr);
    const auto model = reuse->context;
    EXPECT_EQ(model->path(), config.model_path);
    EXPECT_EQ(runner->config().shared_predictor, config.shared_predictor);
    if (!active_mtp)
        EXPECT_FALSE(runner->config().mtp.enabled);
    else
    {
        ASSERT_TRUE(config.shared_predictor);
        EXPECT_TRUE(runner->config().mtp.enabled);
        const auto &runtime = runner->executionPlan().runtime.mtp;
        EXPECT_TRUE(runtime.enabled);
        EXPECT_EQ(runtime.verify_mode, authored.mtp.verify_mode);
        EXPECT_FALSE(authored.deterministic);
        EXPECT_FALSE(runner->config().deterministic);
        // The source-only envelope may be explicitly depth3 while the
        // original numerical model declaration remains unchanged.
        EXPECT_EQ(runtime.draft_tokens, authored.mtp.draft_tokens);
        EXPECT_EQ(runtime.graph_capacity_draft_tokens, cell.retained_mtp_draft_capacity);
        EXPECT_EQ(runtime.depth_policy.mode,
                  cell.usesDynamicMTPDepth() ? MTPDepthPolicyMode::Dynamic : MTPDepthPolicyMode::Fixed);
        EXPECT_TRUE(retainsMTPGraphCapacity(runtime));
        if (cell.usesDynamicMTPDepth())
        {
            // Zero retains production automatic initial-depth intent. Setup
            // evidence does not claim any particular executed controller choice.
            EXPECT_EQ(authored.mtp.depth_policy.initial_depth, 0);
            EXPECT_EQ(runtime.depth_policy.min_depth, 1);
            EXPECT_EQ(runtime.depth_policy.max_depth, cell.retained_mtp_draft_capacity);
        }
    }
    const auto &loader = model->concreteLoader();
    const auto *main = loader.ggufDirectory();
    ASSERT_NE(main, nullptr);
    EXPECT_EQ(main->block_count, 48u);
    ASSERT_EQ(main->split_paths.size(), 4u);
    const auto paths = model->artifactPaths();
    ASSERT_EQ(paths.size(), config.shared_predictor ? 5u : 4u);
    EXPECT_TRUE(std::equal(main->split_paths.begin(), main->split_paths.end(), paths.begin()));
    EXPECT_EQ(paths.front(), config.model_path);
    for (const auto &path : paths)
        EXPECT_TRUE(std::filesystem::is_regular_file(path)) << path;
    if (config.shared_predictor)
    {
        EXPECT_EQ(paths.back(), config.shared_predictor->path());
        EXPECT_EQ(loader.predictorMaterialization(), active_mtp ? SharedPredictorMaterialization::Active
                                                                : SharedPredictorMaterialization::Inactive);
    }
    // The public materialization selection must retain every original main
    // tensor, including all expert banks and the full native PLE. initialize()
    // above constructs the actual prepared operands/engines and serving graphs;
    // source metadata is not used to claim an unmaterialized engine exists.
    const auto selected = loader.materializationNames();
    std::set<std::string> expected_selected;
    for (const auto &tensor : main->tensors)
        ASSERT_TRUE(expected_selected.insert(tensor.name).second);
    if (config.shared_predictor)
    {
        const auto *predictor = loader.ggufPredictorDirectory();
        const auto *manifest = loader.predictorManifest();
        ASSERT_NE(predictor, nullptr);
        ASSERT_NE(manifest, nullptr);
        EXPECT_EQ(predictor->block_count, 49u);
        EXPECT_EQ(manifest->sourceLayer(), 48u);
        EXPECT_EQ(manifest->attention(), Qwen4ExpPredictorAttention::Dense);
        EXPECT_EQ(manifest->hiddenNorm(), Qwen4ExpPredictorHiddenNorm::PerBranch);
        EXPECT_FALSE(manifest->sparseSelector());
        EXPECT_EQ(manifest->predictorMetadata().cacheOrigin(),
                  MTPShiftedCacheOrigin::VisibleZeroHiddenTokenZero);
        for (const auto &source : manifest->sources())
        {
            if (source.source == Qwen4ExpTensorSource::Predictor)
            {
                EXPECT_EQ(main->findTensor(source.tensor.name), nullptr);
                if (active_mtp && source.consumed)
                    ASSERT_TRUE(expected_selected.insert(source.tensor.name).second);
            }
            else
            {
                const auto *original = main->findTensor(source.tensor.name);
                ASSERT_NE(original, nullptr);
                EXPECT_EQ(predictor->findTensor(source.tensor.name), nullptr);
                EXPECT_EQ(source.tensor.dimensions, original->dimensions);
                EXPECT_EQ(source.tensor.offset, original->offset);
                EXPECT_EQ(source.tensor.split_idx, original->split_idx);
                EXPECT_EQ(source.tensor.type, original->type);
                EXPECT_EQ(source.tensor.size_bytes, original->size_bytes);
            }
        }
        if (active_mtp)
        {
            const auto learned = requireMTPWeightManifest(loader, "qwen4exp", 48);
            ASSERT_TRUE(learned.available);
            ASSERT_EQ(learned.depths.size(), 1u);
            EXPECT_EQ(learned.depths.front().depth_index, 0);
            EXPECT_EQ(learned.depths.front().source_layer_index, 48);
            EXPECT_EQ(learned.flash_next, std::optional(manifest->predictorMetadata()));
        }
    }
    else
        EXPECT_FALSE(active_mtp);
    ASSERT_EQ(selected.size(), expected_selected.size());
    std::set<std::string> unique_selected(selected.begin(), selected.end());
    EXPECT_EQ(unique_selected.size(), selected.size());
    EXPECT_EQ(unique_selected, expected_selected);
    const auto store = model->concreteWeightManager()->preparedWeightStoreIfInitialized();
    ASSERT_NE(store, nullptr);
    EXPECT_EQ(store->physicalMemoryAuthority(), reuse->physical_memory_authority);
    size_t prepared_predictor_records = 0, setup_participants = 0, planned_serving_participants = 0,
           deferred_host_participants = 0;
    if (config.shared_predictor)
    {
        prepared_predictor_records = qwen4ExpSourcePreparedPredictorExpertRecords(*reuse);
        if (active_mtp)
            EXPECT_GT(prepared_predictor_records, 0u);
        else
            EXPECT_EQ(prepared_predictor_records, 0u);
    }
    if (active_mtp)
    {
        const Qwen4ExpConfig source_model(*main);
        ASSERT_LE(source_model.mainLayers(), source_model.mixers().size());
        const auto programs = std::any_of(source_model.mixers().begin(),
            source_model.mixers().begin() + source_model.mainLayers(),
            [](auto mixer) { return mixer == Qwen4ExpMixer::SparseAttention; })
            ? Qwen4ExpMTPSourceProgramFamily::CompleteSparse
            : Qwen4ExpMTPSourceProgramFamily::RankingOnly;
        const auto &resolved_overlay = runner->config().moe_routed_expert_plan;
        const auto host_setup = reuse->pipeline_prepared_plan ||
            (resolved_overlay && resolved_overlay->usesExpertOverlayAuthority())
            ? Qwen4ExpMTPHostSetup::OverlayMaterialized : Qwen4ExpMTPHostSetup::OrdinaryDeferred;
        const auto expected = qwen4ExpMTPSourceServingParticipants(runner->executionPlan(), programs, host_setup);
        planned_serving_participants = expected.size();
        deferred_host_participants = std::count_if(expected.begin(), expected.end(), [](const auto &leaf)
                                                   { return !leaf.completion_required; });
        ASSERT_NO_THROW(setup_participants = requireQwen4ExpMTPSourceSetupFamilies(
                            setup_before, PerfStatsCollector::snapshot(), expected));
    }
    ASSERT_NO_THROW(
        requireQwen4ExpCPUStorageMaterialization(reuse->physical_memory_authority->rankAttestation()));
    std::ofstream mtp_setup(output / "mtp-setup.csv");
    ASSERT_TRUE(mtp_setup.is_open());
    mtp_setup
        << "cell,active,retained_depth,automatic_initial_depth,prepared_predictor_records,planned_serving_"
           "participants,setup_participants,deferred_host_participants,controller_execution_proved\n"
        << cell.testName() << ',' << active_mtp << ',' << runner->config().mtp.graph_capacity_draft_tokens
        << ',' << (active_mtp && cell.usesDynamicMTPDepth() && authored.mtp.depth_policy.initial_depth == 0)
        << ',' << prepared_predictor_records << ',' << planned_serving_participants << ','
        << setup_participants << ',' << deferred_host_participants << ",0\n";
    mtp_setup.flush();
    ASSERT_TRUE(mtp_setup.good());
}

/**
 * @brief Share canonical setup and final-owner cleanup across admission outcomes.
 * @param cell Original typed source/topology request, never a rewritten fallback.
 * @param observe Void observer selecting successful admission or precise rejection.
 * @param gpu_memory_cap_mb Optional explicitly authored public binary-MiB ceiling.
 *
 * Observer-local aliases expire before finish(). Fatal assertions and exceptions
 * retain their failed outcome while explicit final retirement and disposable-cache
 * removal publish independent evidence. No receipt is emitted if retirement fails.
 */
template<class Observe>
void withQwen4ExpSourceAdmission(const ModelParityCase &cell, Observe &&observe,
    std::optional<std::size_t> gpu_memory_cap_mb = std::nullopt)
{
    int rank = -1, ranks = 0;
    ASSERT_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &rank), MPI_SUCCESS);
    ASSERT_EQ(MPI_Comm_size(MPI_COMM_WORLD, &ranks), MPI_SUCCESS);
    ASSERT_EQ(ranks, cell.topology.mpi_ranks);
    auto config = qwen4ExpSourceAdmissionCapacityConfig(cell, cell.model.model_path, rank, gpu_memory_cap_mb);
    publishOrchestrationStartupPolicy(config);
    initCPUBackend(-1);
    // Establish the normal startup inventory once before materializing model
    // storage. Subsequent cases consume that frozen observation, never a live
    // post-admission rediscovery or CPUBackend's rank-local deviceCount()==1.
    auto &devices = DeviceManager::instance();
    if (!devices.hardware()) devices.initialize(-1, false);
    const auto *hardware = devices.hardware();
    ASSERT_NE(hardware, nullptr);
    ASSERT_NO_THROW(requireQwen4ExpSourceCPUInventory(cell, rank, *hardware));
    for (const auto &participant : cell.topology.participants)
    {
        if (participant.world_rank && *participant.world_rank != rank) continue;
        const auto type = participant.address.device_type;
        if (type == DeviceType::CPU) continue;
        auto *backend = type == DeviceType::CUDA ? getCUDABackend() : getROCmBackend();
        if (!backend || participant.address.device_ordinal >= backend->deviceCount())
            GTEST_SKIP() << "Missing declared admission backend " << participant.address.toShortString();
    }
    ASSERT_TRUE(std::filesystem::is_regular_file(config.model_path)) << config.model_path;
    if (config.shared_predictor)
        ASSERT_TRUE(std::filesystem::is_regular_file(config.shared_predictor->path())) << config.shared_predictor->path();
    const char *artifact_root = std::getenv("LLAMINAR_PRODUCTION_PARITY_ARTIFACT_ROOT");
    const auto base = artifact_root ? std::filesystem::path(artifact_root)
        : std::filesystem::current_path() / "parity-results" / "qwen-flash-next";
    ASSERT_TRUE(base.is_absolute());
    const auto output = base / "source-admission" / cell.testName() / ("rank-" + std::to_string(rank));
    ASSERT_NO_THROW(prepareQwen4ExpAdmissionEvidenceDirectory(output));
    config.prefix_cache.disk_dir = (output / "prefix-cache").string();
    auto authored = config;
    if (config.moe_routed_expert_plan)
        authored.moe_routed_expert_plan = std::make_shared<MoERoutedExpertPlacementPlan>(*config.moe_routed_expert_plan);
    Qwen4ExpSourceAdmissionLifetime lifetime(config, output / "prefix-cache");
    observeQwen4ExpAdmissionThenRetire(
        [&] { observe(rank, config, authored, lifetime, output); },
        [&] {
            ASSERT_NO_THROW(lifetime.finish());
            std::ofstream cleanup(output / "retirement.csv");
            ASSERT_TRUE(cleanup.is_open());
            cleanup << "cell,runner_retired,model_retired,prefix_removed,retired_gpu_generations\n"
                << cell.testName() << ",1,1,1," << lifetime.retiredDeviceCount() << '\n';
            cleanup.flush();
            ASSERT_TRUE(cleanup.good());
        });
}

/**
 * @brief Materialize one original-source serving configuration without gold or inference.
 * @param cell Complete existing typed source/topology/precision/storage declaration.
 *
 * Successful setup authenticates exact original sources, actual execution plan,
 * retained native owners and snapshot-zero PMA before final exclusive retirement.
 */
inline void admitQwen4ExpSource(const ModelParityCase &cell)
{
    withQwen4ExpSourceAdmission(cell, [&](int rank, const OrchestrationConfig &config,
        const OrchestrationConfig &authored, Qwen4ExpSourceAdmissionLifetime &lifetime,
        const std::filesystem::path &output) {
    auto *runner = &lifetime.runner();
    runner->disableSnapshotCapture();
    const auto setup_before=beginQwen4ExpSourceMTPEvidence(cell);
    ASSERT_TRUE(runner->initialize()) << runner->lastError();
    ASSERT_TRUE(runner->isInitialized());
    EXPECT_TRUE(runner->getSnapshotKeys().empty());
    EXPECT_EQ(runner->executionPlan().layerCount(), cell.model.transformer_layers);
    EXPECT_EQ(cell.model.transformer_layers, 48);
    EXPECT_EQ(runner->config().model_path, config.model_path);
    EXPECT_EQ(runner->config().shared_predictor, config.shared_predictor);
    EXPECT_EQ(runner->config().max_seq_len, config.max_seq_len);
    EXPECT_EQ(runner->config().batch_size, 1);
    EXPECT_EQ(runner->config().activation_precision, config.activation_precision);
    EXPECT_EQ(runner->config().kv_cache_precision, config.kv_cache_precision);
    EXPECT_EQ(runner->config().ngram_table_policy, config.ngram_table_policy);
    ASSERT_NO_FATAL_FAILURE(expectQwen4ExpSourceDomains(
        canonicalQwen4ExpSourceAdmissionConfig(authored, rank), runner->config()));
    ASSERT_NO_THROW(requireQwen4ExpSourceMaintenancePolicy(authored.moe_rebalance, runner->config().moe_rebalance));
    ASSERT_NO_THROW(requireQwen4ExpSourceMaintenancePolicy(authored.moe_rebalance, runner->executionPlan().runtime.moe_rebalance));
    ASSERT_NO_THROW(requireQwen4ExpSourceExecutionPlan(cell, rank, authored,
        runner->executionPlan(), *DeviceManager::instance().hardware()));
    ASSERT_EQ(runner->config().pp_stage_definitions.size(), authored.pp_stage_definitions.size());
    for (std::size_t i = 0; i < authored.pp_stage_definitions.size(); ++i)
    {
        const auto &actual = runner->config().pp_stage_definitions[i];
        const auto &expected = authored.pp_stage_definitions[i];
        EXPECT_EQ(actual.stage_id, expected.stage_id);
        EXPECT_EQ(actual.domain_name, expected.domain_name);
        EXPECT_EQ(actual.first_layer, expected.first_layer);
        EXPECT_EQ(actual.last_layer, expected.last_layer);
    }
    const auto &reuse = lifetime.reuseContract();
    ASSERT_TRUE(reuse);
    ASSERT_NE(reuse->context, nullptr);
    ASSERT_NE(reuse->physical_memory_authority, nullptr);
    ASSERT_NE(reuse->physical_memory_authority->admission(), nullptr);
    const auto model = reuse->context;
    EXPECT_EQ(runner->modelContextForDiagnostics(), model.get());
    EXPECT_EQ(model->path(), config.model_path);
    EXPECT_EQ(model->totalBlockCount(), 48);
    const auto &loader=model->concreteLoader();
    const auto *main=loader.ggufDirectory();ASSERT_NE(main,nullptr);
    for (int layer = 0; layer < 48; ++layer)
        for (const auto *role : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"})
        {
            const auto name = "blk." + std::to_string(layer) + "." + role;
            const auto *tensor = main->findTensor(name);
            ASSERT_NE(tensor, nullptr) << name;
            ASSERT_EQ(tensor->dimensions.size(), 3u) << name;
            EXPECT_EQ(tensor->dimensions[2], 512u) << name;
            EXPECT_GT(tensor->size_bytes, 0u) << name;
        }
    const auto store = model->concreteWeightManager()->preparedWeightStoreIfInitialized();
    ASSERT_NE(store, nullptr);
    EXPECT_GT(store->size(), 0u);
    EXPECT_EQ(store->physicalMemoryAuthority(), reuse->physical_memory_authority);
    ASSERT_NO_FATAL_FAILURE(expectQwen4ExpSourceMTPMaterialization(
        cell,authored,*runner,*reuse,output,setup_before));
    if (config.moe_routed_expert_plan)
    {
        ASSERT_NE(reuse->prepared_routed_weight_plan, nullptr);
        ASSERT_NE(reuse->expert_overlay_memory_admission, nullptr);
        EXPECT_FALSE(reuse->routed_weight_authority_identity.empty());
    }
    for (const auto &resource : reuse->physical_memory_authority->admission()->plan().resources())
        EXPECT_EQ(resource.bytes(PhysicalMemoryOwner::GraphSnapshotArena), 0u);
    ASSERT_NO_FATAL_FAILURE(expectQwen4ExpSourceLookup(cell, *runner));
    std::ofstream memory(output / "admission_memory.txt");
    ASSERT_TRUE(memory.is_open());
    memory << reuse->physical_memory_authority->rankAttestationSummary(true);
    ASSERT_NO_THROW(requireQwen4ExpCPUStorageMaterialization(
        reuse->physical_memory_authority->rankAttestation()));
    memory.flush();
    ASSERT_TRUE(memory.good());
    std::ofstream csv(output / "admission.csv");
    ASSERT_TRUE(csv.is_open());
    const auto placement = model->resolvedImmutableRowTablePlacement();
    ASSERT_NE(placement, nullptr);
    csv << "cell,layers,initialized,snapshots_enabled,numerical_inference_executed,native_table_bytes\n"
        << cell.testName() << ",48,1,0,0," << placement->extent().nativeBytes() << '\n';
    csv.flush();
    ASSERT_TRUE(csv.good());
    });
}

/**
 * @brief Prove explicit physical capacity rejection without claiming unsupported execution.
 * @param cell Exact larger request retaining the same head/expert/MTP declarations.
 * @param gpu_memory_cap_mb Authored public ceiling; applied before any capacity planning.
 */
inline void admitQwen4ExpSourceCapacityRejected(const ModelParityCase &cell, std::size_t gpu_memory_cap_mb)
{
    withQwen4ExpSourceAdmission(cell, [&](int, const OrchestrationConfig &config,
        const OrchestrationConfig &, Qwen4ExpSourceAdmissionLifetime &lifetime,
        const std::filesystem::path &output) {
        auto &runner = lifetime.runner();
        runner.disableSnapshotCapture();
        ASSERT_EQ(config.max_gpu_memory_mb, gpu_memory_cap_mb);
        ASSERT_FALSE(runner.initialize());
        ASSERT_FALSE(runner.isInitialized());
        const auto &error = runner.lastError();
        ASSERT_NE(error.find("Explicit fixed head native ranges exceed complete frozen PMA capacity"), std::string::npos) << error;
        ASSERT_NE(error.find("immutable_table_storage="), std::string::npos) << error;
        ASSERT_NE(error.find("deficit_bytes="), std::string::npos) << error;
        ASSERT_FALSE(runner.modelContextReuseContract().has_value());
        ASSERT_TRUE(runner.getSnapshotKeys().empty());
        std::ofstream diagnostic(output / "capacity-diagnostic.txt");
        ASSERT_TRUE(diagnostic.is_open());
        diagnostic << error << '\n';
        diagnostic.flush();
        ASSERT_TRUE(diagnostic.good());
        std::ofstream receipt(output / "capacity-rejection.csv");
        ASSERT_TRUE(receipt.is_open());
        receipt << "cell,initialized,capacity_rejected,gpu_memory_cap_mb,numerical_inference_executed\n"
            << cell.testName() << ",0,1," << gpu_memory_cap_mb << ",0\n";
        receipt.flush();
        ASSERT_TRUE(receipt.good());
    }, gpu_memory_cap_mb);
}

/**
 * @brief Prove precise rejection of the explicitly unsupported cross-tier expert slicing contract.
 * @param cell Unchanged source/topology request with GateUpOwnedDownColumns across tiers.
 *
 * This never substitutes apportioned experts, changes storage, or emits an
 * admission success row. Failed initialization must publish no reusable model
 * certificate or snapshot materialization; the same owner drains partial setup
 * and removes only its disposable prefix child before returning.
 */
inline void admitQwen4ExpSourceRejected(const ModelParityCase &cell)
{
    withQwen4ExpSourceAdmission(cell, [&](int, const OrchestrationConfig &,
        const OrchestrationConfig &, Qwen4ExpSourceAdmissionLifetime &lifetime,
        const std::filesystem::path &output) {
        auto &runner = lifetime.runner();
        runner.disableSnapshotCapture();
        ASSERT_FALSE(runner.initialize());
        ASSERT_FALSE(runner.isInitialized());
        ASSERT_NE(runner.lastError().find("Gate/up-owned down columns require a single native GPU domain; cross-tier slice publication is not implemented"), std::string::npos)
            << runner.lastError();
        ASSERT_FALSE(runner.modelContextReuseContract().has_value());
        ASSERT_TRUE(runner.getSnapshotKeys().empty());
        std::ofstream rejected(output / "rejection.csv");
        ASSERT_TRUE(rejected.is_open());
        rejected << "cell,initialized,expected_rejection,numerical_inference_executed\n"
            << cell.testName() << ",0,1,0\n";
        rejected.flush();
        ASSERT_TRUE(rejected.good());
    });
}
} // namespace llaminar2::test::parity::qwen4exp
