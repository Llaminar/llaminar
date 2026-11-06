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
 * Expert-owner source proofs retain only weak native-parent witnesses across
 * original-contract reuse; attachment evidence explicitly claims no execution.
 * Active policies authenticate real retained sidecar/forward setup; they do not
 * claim generation-controller execution or numerical sidecar certification.
 */
#pragma once
#include <functional>
#include "Qwen4ExpProductionParityCases.h"
#include "integration/parity/ParityArtifactDirectory.h"
#include "Qwen4ExpSourceAdmissionCases.h"
#include "Qwen4ExpCPUJointSocketEvidence.h"
#include "utils/MPIContext.h"
#include "execution/runner/IOrchestrationRunnerFactory.h"
#include "integration/parity/ParityPredictorSource.h"
#include "Qwen4ExpSourceAdmissionEvidence.h"
#include "Qwen4ExpSourceAdmissionObservation.h"
#include "execution/runner/ModelContextRetirement.h"
#include "backends/BackendManager.h"
#include "backends/ComputeBackend.h"
#include "backends/HardwareInventory.h"
#include "backends/NativeContextPreparation.h"
#include "backends/GPUDeviceContextPool.h"
#include "config/OrchestrationStartupPolicy.h"
#include "execution/qwen4exp/ImmutableLookupExecutionBoundary.h"
#include "execution/qwen4exp/PooledImmutableRetainedParticipants.h"
#include "execution/qwen4exp/PooledImmutableParticipantExecutionPolicy.h"
#include "loaders/AdaptiveImmutableLookupGeometry.h"
#include "kernels/qwen4exp/PooledImmutableSourcePreparedStorage.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotPreparedStorage.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotTransportProtocol.h"
#include "loaders/ModelContext.h"
#include "loaders/PreparedWeightStore.h"
#include "loaders/ExpertGemmRegistry.h"
#include "execution/moe/CPUExpertShardSourceSelection.h"
#include "models/qwen4exp/Qwen4ExpSidecarManifest.h"
#include "execution/mtp/MTPWeightManifest.h"
#include "utils/PerfStatsCollector.h"
#include "loaders/ImmutableLookupSetupObservation.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "utils/TestOrchestrationHelper.h"
#include "utils/FinalPhysicalMemoryCostEvidence.h"
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
 * @brief Sum only actual disjoint local native leases beside the original pool/source owner.
 * @param context Same retained model whose resolved placement declares every local consumer.
 * @param allocator Original GPU or CPU0 readiness allocator.
 * @param category Exact physical category; no capacity or hypothetical credit is used.
 * @return Actual local-only bytes, zero in unchanged shared-only modes.
 */
inline std::size_t qwen4ExpSourceLocalHotLeaseBytes(const ModelContext& context,
    PhysicalMemoryAllocatorIdentity allocator, PhysicalMemoryOwner category)
{
    const auto placement=context.resolvedImmutableRowTablePlacement();
    if(!placement || !placement->isGPUAdaptive() || !placement->gpuAdaptiveRequest().local_hot_cache)return 0;
    std::size_t bytes=0;
    for(const auto& geometry:placement->gpuAdaptiveLocalHotCaches()) {
        const auto local=context.adaptiveImmutableLookupLocalHot(geometry.allocator().device);
        if(!local)throw std::logic_error("Combined witness omitted an actual local native owner");
        const auto observation=allocator.device.is_cpu()?local->retainedReadyObservation():local->retainedOwnerObservation();
        if(observation.allocator()!=allocator)continue;
        const auto add=category==PhysicalMemoryOwner::ImmutableTableStorage?observation.tableStorageBytes():
            category==PhysicalMemoryOwner::ImmutableLookupStaging?observation.lookupStagingBytes():0;
        if(add>std::numeric_limits<std::size_t>::max()-bytes)throw std::overflow_error("Local witness bytes overflow");
        bytes+=add;
    }
    return bytes;
}

/**
 * @brief Prove complete original local roots, compact source identity and separately retained CPU ready pages.
 * @param original Actual retained model/PMA; this observer cannot materialize roots or grant native work.
 */
inline void expectQwen4ExpSourceLocalHotOwners(const ModelContextReuseContract& original)
{
    const auto& context=original.context;ASSERT_TRUE(context);
    const auto placement=context->resolvedImmutableRowTablePlacement();ASSERT_TRUE(placement);
    if(!placement->gpuAdaptiveRequest().local_hot_cache) {
        EXPECT_TRUE(placement->gpuAdaptiveLocalHotCaches().empty());
        return;
    }
    const auto& memory=original.physical_memory_authority;ASSERT_TRUE(memory);
    const auto& request=placement->gpuAdaptiveRequest();
    ASSERT_EQ(request.placement.mode,AdaptiveGPUCachePlacementMode::DeclaredRankLocal);
    ASSERT_TRUE(request.placement.coordinator);
    const auto coordinator=request.placement.coordinator->device();
    const auto participants=context->adaptiveImmutableLookupParticipants(coordinator);ASSERT_TRUE(participants);
    ASSERT_NO_THROW(participants->requireCurrent());
    const auto transport=participants->transport();ASSERT_TRUE(transport);
    ASSERT_TRUE(transport->localHotGeometry());ASSERT_NO_THROW(transport->requireCurrent());
    const auto source=participants->boundary();ASSERT_TRUE(source);
    ASSERT_EQ(participants->localHotParticipants().size(),placement->gpuAdaptiveLocalHotCaches().size());
    ASSERT_EQ(placement->gpuAdaptiveLocalHotCaches().size(),placement->gpuAdaptiveLayout().consumers().size());
    const auto setup=context->immutableLookupSetupObservation();ASSERT_TRUE(setup);
    std::size_t local_rows=0,ready_rows=0;
    for(const auto& row:setup->owners()) {
        local_rows+=row.role==ImmutableLookupSetupObservation::Role::LocalGPUCache;
        ready_rows+=row.role==ImmutableLookupSetupObservation::Role::LocalReady;
    }
    EXPECT_EQ(local_rows,placement->gpuAdaptiveLocalHotCaches().size());EXPECT_EQ(ready_rows,local_rows);
    for(const auto& geometry:placement->gpuAdaptiveLocalHotCaches()) {
        const auto gpu=geometry.allocator();
        const auto local=context->adaptiveImmutableLookupLocalHot(gpu.device);ASSERT_TRUE(local);
        ASSERT_NO_THROW(local->requireFor(geometry,source->pooledStorage(),memory));
        ASSERT_EQ(local->originalSource(),source->pooledStorage());
        EXPECT_FALSE(local->originalSource().owner_before(source->pooledStorage()));
        EXPECT_FALSE(source->pooledStorage().owner_before(local->originalSource()));
        const auto retained=std::find(participants->localHotParticipants().begin(),participants->localHotParticipants().end(),local);
        ASSERT_NE(retained,participants->localHotParticipants().end());
        EXPECT_FALSE(retained->owner_before(local));EXPECT_FALSE(local.owner_before(*retained));
        EXPECT_NE(local->captureIdentity(),0u);EXPECT_NE(local->consumerStorageIdentity(),0u);
        const auto roots=local->roots();
        EXPECT_EQ(bool(roots.cache),geometry.slots()!=0);ASSERT_TRUE(roots.frame);ASSERT_TRUE(roots.native);
        for(const auto& root:{roots.cache,roots.frame,roots.native})if(root) {
            EXPECT_TRUE(root->is_raw_data_released());EXPECT_NE(root->gpu_data_ptr(),nullptr);
        }
        const auto native=local->retainedOwnerObservation();const auto ready=local->retainedReadyObservation();
        EXPECT_EQ(native.authority(),memory);EXPECT_EQ(ready.authority(),memory);
        EXPECT_EQ(native.allocator(),gpu);EXPECT_EQ(native.tableStorageBytes(),geometry.physicalBytes());
        EXPECT_EQ(native.lookupStagingBytes(),geometry.frameOffsets().bytes+geometry.scratchOffsets().bytes);
        EXPECT_EQ(ready.allocator(),(PhysicalMemoryAllocatorIdentity{gpu.world_rank,DeviceId::cpu()}));
        EXPECT_EQ(ready.tableStorageBytes(),0u);
        EXPECT_EQ(ready.lookupStagingBytes(),TransferEngine::mappedHostRegionAllocationBytes(sizeof(std::uint64_t)));
        ASSERT_TRUE(local->readyRegion());
        const auto recorded=std::find_if(setup->owners().begin(),setup->owners().end(),[&](const auto& row) {
            return row.role==ImmutableLookupSetupObservation::Role::LocalGPUCache && row.allocator==gpu;
        });
        ASSERT_NE(recorded,setup->owners().end());
        EXPECT_EQ(recorded->admitted_slots,geometry.slots());
        EXPECT_EQ(recorded->payload_bytes,geometry.rowBytes()*geometry.slots());
        EXPECT_EQ(recorded->metadata_bytes,geometry.physicalBytes()-recorded->payload_bytes);
        EXPECT_EQ(recorded->table_storage_bytes,native.tableStorageBytes());
        EXPECT_EQ(recorded->lookup_staging_bytes,native.lookupStagingBytes());
        const auto request_span=transport->localHotOriginalRequest(gpu.device);
        const auto admission=transport->localHotAdmission(gpu.device);
        EXPECT_EQ(request_span.bytes(),sizeof(::llaminar2::qwen4exp::local_hot_transport::RequestHeader));
        EXPECT_EQ(admission.bytes(),sizeof(::llaminar2::qwen4exp::local_hot_transport::SourceAdmissionHeader));
        EXPECT_NO_THROW(request_span.require(gpu.device));
        EXPECT_NO_THROW(admission.require(gpu.device));
        EXPECT_NO_THROW(request_span.requireDisjoint(admission));
    }
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
 * @brief Authenticate actual cold expert-parent attachments through the existing original participant authority.
 * @param cell Unchanged canonical case.
 * @param original SAME retained model/source/PMA contract used by ordinary source proof.
 * @param output Owned rank evidence file for this original or replacement lifetime.
 * @param observations Weak original native-parent witnesses returned for the existing reuse frontier.
 * @throws std::exception For stale original participant/worker/parent authority.
 * @note Parent attachment proves setup only, never execution, useful traffic or cache movement.
 */
inline void expectQwen4ExpSourceExpertLookupParents(const ModelParityCase& cell,
    const ModelContextReuseContract& original,const std::filesystem::path& output,
    std::vector<::llaminar2::qwen4exp::PooledImmutableRetainedParticipants::ExpertOwnerParentObservation>& observations)
{
    const auto placement=original.context->resolvedImmutableRowTablePlacement();
    const auto expected=qwen4ExpSourceExpertLookupOwners(cell,placement.get());
    if(expected.empty())return;
    ASSERT_TRUE(placement->gpuAdaptiveRequest().placement.coordinator);
    const auto coordinator=placement->gpuAdaptiveRequest().placement.coordinator->device();
    const auto participants=original.context->adaptiveImmutableLookupParticipants(coordinator);
    ASSERT_TRUE(participants);ASSERT_NO_THROW(participants->requireCurrent());
    observations=participants->expertOwnerParentObservations();
    ASSERT_EQ(observations.size(),expected.size());
    std::vector<PhysicalMemoryAllocatorIdentity> seen;
    std::ofstream proof(output);ASSERT_TRUE(proof.is_open());
    proof<<"world_rank,device,owner_coordinate,worker_generation,attached_parents,source_variants,execution_proven\n";
    for(const auto& actual:observations){
        EXPECT_EQ(std::find(seen.begin(),seen.end(),actual.allocator),seen.end());
        seen.push_back(actual.allocator);
        ASSERT_NE(std::find(expected.begin(),expected.end(),actual.allocator),expected.end());
        EXPECT_EQ(actual.allocator.world_rank,original.physical_memory_authority->worldRank());
        const auto& declared=placement->gpuAdaptiveRequest().placement.owners;
        ASSERT_LT(actual.owner_coordinate,declared.size());
        EXPECT_EQ(actual.allocator,(PhysicalMemoryAllocatorIdentity{
            declared[actual.owner_coordinate].participant.world_rank,
            declared[actual.owner_coordinate].participant.device()}));
        EXPECT_EQ(actual.memory_identity,original.physical_memory_authority.get());
        const auto permit=participants->originalParticipant(actual.allocator.device);
        EXPECT_EQ(actual.worker_generation,permit.generation());
        const auto worker=actual.worker.lock();ASSERT_TRUE(worker);EXPECT_EQ(worker,permit.worker());
        EXPECT_EQ(permit.memory(),original.physical_memory_authority);
        EXPECT_FALSE(permit.memory().owner_before(original.physical_memory_authority));
        EXPECT_FALSE(original.physical_memory_authority.owner_before(permit.memory()));
        ASSERT_FALSE(actual.parent_lifetimes.empty());EXPECT_GT(actual.source_variant_count,0u);
        for(const auto& parent:actual.parent_lifetimes)EXPECT_FALSE(parent.expired());
        proof<<actual.allocator.world_rank<<','<<actual.allocator.device.toString()<<','<<actual.owner_coordinate<<','
            <<actual.worker_generation<<','<<actual.parent_lifetimes.size()<<','<<actual.source_variant_count<<",0\n";
    }
    proof.flush();ASSERT_TRUE(proof.good());
}

/**
 * @brief Authenticate a GPU-adaptive source, cache roots and their actual leases.
 * @param cell Exact canonical request, including zero/positive cache ceilings.
 * @param runner Successfully initialized real production runner.
 * @param original Original export retained by the caller until final retirement.
 * @param gpu Exact original consumer key; independent of TP participant ordinal.
 *
 * This is a borrowed observation and never exports another lifecycle owner.
 * It observes existing source metadata and retained native owners only. It
 * does not perform a lookup or download a cache tag, row or mutable model state.
 */
inline void expectQwen4ExpSourceGPUAdaptiveConsumer(const ModelParityCase &cell, IOrchestrationRunner &runner,
    const ModelContextReuseContract &original, PhysicalMemoryAllocatorIdentity gpu)
{
    ASSERT_TRUE(cell.ngram_table_policy);ASSERT_TRUE(cell.ngram_table_policy->isGPUAdaptive());
    const auto *retained=&original;
    const auto expect_original_lease=[&](const PhysicalMemoryAuthority& memory,
        PhysicalMemoryAllocatorIdentity allocator,PhysicalMemoryOwner owner,std::size_t bytes) {
        const auto local=qwen4ExpSourceLocalHotLeaseBytes(*original.context,allocator,owner);
        ASSERT_LE(local,std::numeric_limits<std::size_t>::max()-bytes);
        const auto resolved=original.context->resolvedImmutableRowTablePlacement();
        if(allocator.device.is_cpu() && owner==PhysicalMemoryOwner::ImmutableLookupStaging &&
            resolved && resolved->isGPUAdaptive() && resolved->gpuAdaptiveRequest().local_hot_cache) {
            // Source/LocalReady observations above authenticate their own exact
            // retained roots. The runner's separate terminal lease shares this
            // category, so require the complete original PMA category instead
            // of mistaking the retained subset for all live physical bytes.
            ASSERT_NO_THROW(test::requireCompleteFinalPhysicalMemoryOwner(memory,allocator,owner));
        } else {
            ASSERT_NO_FATAL_FAILURE(expectQwen4ExpSourceLease(memory,allocator,owner,bytes+local));
        }
    };
    const auto *context=retained->context.get();ASSERT_NE(context,nullptr);
    EXPECT_EQ(runner.modelContextForDiagnostics(),context);
    const auto placement=context->resolvedImmutableRowTablePlacement();ASSERT_TRUE(placement);
    ASSERT_TRUE(placement->isGPUAdaptive());placement->requireRequestedPolicy(*cell.ngram_table_policy);
    const auto &layout=placement->gpuAdaptiveLayout();const auto &request=placement->gpuAdaptiveRequest();
    ASSERT_EQ(layout.sourceServices().size(),1u);
    ASSERT_NE(std::find(layout.consumers().begin(),layout.consumers().end(),gpu),layout.consumers().end());
    const bool declared=request.placement.mode==AdaptiveGPUCachePlacementMode::DeclaredRankLocal;
    const bool grouped=layout.adaptiveGPUDomain()==ImmutableRowTableAdaptiveGPUDomain::ConsumerLocal;
    const auto expected_owner_count=declared?request.placement.owners.size():1u;
    const auto inventory_owner_count=declared?request.placement.owners.size():layout.consumers().size();
    ASSERT_EQ(layout.owners().size(),inventory_owner_count+1u);
    ASSERT_EQ(placement->allocators().size(),inventory_owner_count+1u);
    const auto host=layout.sourceServices().front().allocator;
    ASSERT_TRUE(gpu.device.is_gpu());ASSERT_TRUE(host.device.is_cpu());EXPECT_EQ(gpu.world_rank,host.world_rank);
    EXPECT_EQ(placement->lookupForConsumer(gpu),ImmutableRowLookupExecution::AdaptiveGPUCache);
    const bool shared = declared && layout.consumers().size() > 1;
    if (shared) {
        ASSERT_TRUE(request.placement.coordinator);
        const PhysicalMemoryAllocatorIdentity coordinator{request.placement.coordinator->world_rank,
            request.placement.coordinator->device()};
        if (gpu != coordinator) {
            // A replicated query reader owns its result/ACK endpoints, never a
            // duplicate source service, table, pin authority or provider grant.
            EXPECT_FALSE(context->adaptiveImmutableLookupBoundary(gpu.device));
            const auto participants = context->adaptiveImmutableLookupParticipants(gpu.device);
            ASSERT_TRUE(participants);
            const auto sourceParticipants = context->adaptiveImmutableLookupParticipants(coordinator.device);
            ASSERT_EQ(participants, sourceParticipants);
            EXPECT_FALSE(participants.owner_before(sourceParticipants));
            EXPECT_FALSE(sourceParticipants.owner_before(participants));
            const auto transport = participants->transport();
            ASSERT_TRUE(transport);
            ASSERT_NO_THROW(transport->requireCurrent());
            ASSERT_TRUE(transport->readerGeometry());
            const auto& readers = *transport->readerGeometry();
            EXPECT_EQ(readers.coordinator(), coordinator);
            EXPECT_EQ(std::vector<PhysicalMemoryAllocatorIdentity>(readers.consumers().begin(),readers.consumers().end()),
                std::vector<PhysicalMemoryAllocatorIdentity>(layout.consumers().begin(),layout.consumers().end()));
            const auto sourceBoundary = participants->boundary();
            ASSERT_TRUE(sourceBoundary);
            EXPECT_EQ(sourceBoundary->device(), coordinator.device);
            EXPECT_EQ(context->adaptiveImmutableLookupBoundary(coordinator.device), sourceBoundary);
            ASSERT_EQ(sourceBoundary->pooledStorage()->geometry().parameters().consumers, layout.consumers().size());
            const auto& workers = sourceBoundary->pooledStorage()->retainedPreparedSourceWorkers();
            const auto originalSource = context->nativeContextPreparation();
            ASSERT_NO_THROW((void)requireQwen4ExpConsumedPreparedSource(originalSource,workers));
            const auto& memory = original.physical_memory_authority;
            ASSERT_TRUE(memory);
            EXPECT_EQ(originalSource->memory(),memory);
            EXPECT_FALSE(originalSource->memory().owner_before(memory));
            EXPECT_FALSE(memory.owner_before(originalSource->memory()));
            const auto reader = std::find_if(transport->readers().begin(),transport->readers().end(),[&](const auto& actual) {
                return actual->originalReader().device() == gpu.device;
            });
            ASSERT_NE(reader,transport->readers().end());
            ASSERT_NO_THROW((*reader)->requireCurrent());
            for (const auto* permit : {&(*reader)->originalSource(), &(*reader)->originalReader()}) {
                EXPECT_EQ(permit->memory(), memory);
                EXPECT_FALSE(permit->memory().owner_before(memory));
                EXPECT_FALSE(memory.owner_before(permit->memory()));
                const auto existing = GPUDeviceContextPool::instance().retainExistingPreparedContext(permit->device(),memory);
                EXPECT_EQ(permit->generation(),existing.generation());
                EXPECT_NE(permit->generation(),0u);
                EXPECT_NO_THROW(permit->requireWorker(existing.worker()));
            }
            EXPECT_EQ((*reader)->originalSource().device(),coordinator.device);
            if (readers.nativeCollective()) {
                ASSERT_TRUE((*reader)->nativeCollective());
                EXPECT_FALSE((*reader)->result());
                EXPECT_FALSE((*reader)->acknowledgement());
                const auto& result = (*reader)->nativeResult();
                const auto& ack = (*reader)->nativeAcknowledgement();
                ASSERT_TRUE(result.nativeStorage());
                EXPECT_EQ(result.nativeStorage(), ack.nativeStorage());
                EXPECT_EQ(result.endpointDevice(CapturedTransferEndpoint::Producer), coordinator.device);
                EXPECT_EQ(result.endpointDevice(CapturedTransferEndpoint::Consumer), gpu.device);
                EXPECT_EQ(ack.endpointDevice(CapturedTransferEndpoint::Producer), gpu.device);
                EXPECT_EQ(ack.endpointDevice(CapturedTransferEndpoint::Consumer), coordinator.device);
                if(request.local_hot_cache) {
                    ASSERT_TRUE(transport->localHotGeometry());
                    EXPECT_EQ(result.maximumBytes(),std::max(transport->localHotGeometry()->resultCapacityBytes(),
                        sizeof(::llaminar2::qwen4exp::local_hot_transport::ResultHeader)));
                    EXPECT_EQ(ack.maximumBytes(),std::max(transport->localHotGeometry()->keyCapacityBytes(),
                        sizeof(::llaminar2::qwen4exp::local_hot_transport::RequestHeader)));
                } else {
                    EXPECT_FALSE(transport->localHotGeometry());
                    EXPECT_EQ(result.maximumBytes(), std::max(readers.resultMaximum(), sizeof(::llaminar2::qwen4exp::replicated_reader::Header)));
                    EXPECT_EQ(ack.maximumBytes(), sizeof(::llaminar2::qwen4exp::replicated_reader::Header));
                }
                EXPECT_TRUE(result.nativeStorage()->authority(gpu.device).valid());
                EXPECT_EQ(result.nativeStorage()->cohort()->modelOwner(), context);
            } else {
                EXPECT_EQ((*reader)->result()->endpointDevice(CapturedTransferEndpoint::Producer),coordinator.device);
                EXPECT_EQ((*reader)->result()->endpointDevice(CapturedTransferEndpoint::Consumer),gpu.device);
                EXPECT_EQ((*reader)->acknowledgement()->endpointDevice(CapturedTransferEndpoint::Producer),gpu.device);
                EXPECT_EQ((*reader)->acknowledgement()->endpointDevice(CapturedTransferEndpoint::Consumer),coordinator.device);
                if(request.local_hot_cache) {
                    ASSERT_TRUE(transport->localHotGeometry());
                    EXPECT_EQ((*reader)->result()->capacityBytes(),std::max(transport->localHotGeometry()->resultCapacityBytes(),
                        sizeof(::llaminar2::qwen4exp::local_hot_transport::ResultHeader)));
                    EXPECT_EQ((*reader)->acknowledgement()->capacityBytes(),std::max(transport->localHotGeometry()->keyCapacityBytes(),
                        sizeof(::llaminar2::qwen4exp::local_hot_transport::RequestHeader)));
                } else {
                EXPECT_EQ((*reader)->result()->capacityBytes(),std::max(readers.resultMaximum(),sizeof(::llaminar2::qwen4exp::replicated_reader::Header)));
                EXPECT_EQ((*reader)->result()->payloadStorage(), readers.payloadStorage(gpu));
                EXPECT_EQ((*reader)->acknowledgement()->payloadStorage(), readers.payloadStorage(gpu));
                EXPECT_EQ((*reader)->acknowledgement()->capacityBytes(),sizeof(::llaminar2::qwen4exp::replicated_reader::Header));
                }
            }
            if(request.local_hot_cache) {
                ASSERT_TRUE(transport->localHotGeometry());
                const auto control=transport->localHotControl(gpu.device,(*reader)->coordinate());
                EXPECT_EQ(control.bytes(),sizeof(::llaminar2::qwen4exp::local_hot_transport::Control));
                EXPECT_NO_THROW(control.require(gpu.device,alignof(::llaminar2::qwen4exp::local_hot_transport::Control)));
                EXPECT_THROW((void)(*reader)->control(CapturedTransferEndpoint::Consumer),std::logic_error);
            } else {
            const auto& control = (*reader)->control(CapturedTransferEndpoint::Consumer);
            ASSERT_TRUE(control);
            EXPECT_TRUE(control->isBound());
            EXPECT_EQ(control->device(),gpu.device);
            EXPECT_EQ(control->sizeBytes(),sizeof(::llaminar2::qwen4exp::replicated_reader::Control));
            EXPECT_EQ(reinterpret_cast<std::uintptr_t>(control->deviceData()) % alignof(::llaminar2::qwen4exp::replicated_reader::Control),0u);
            }
            const auto provider = context->hostImmutableRowProvider();
            ASSERT_TRUE(provider);
            EXPECT_EQ(provider->subordinateRequester(),std::optional{coordinator});
            EXPECT_EQ(provider->subordinateRequesters().size(),1u);
            EXPECT_EQ(context->residentImmutableRowTable(gpu.device),nullptr);
            EXPECT_EQ(context->residentImmutableRowTableBytes(gpu.device),0u);
            const auto geometry = AdaptiveImmutableLookupGeometry::fromLayoutWithLocalHot(layout,provider->source(),request,placement->gpuAdaptiveLocalHotCaches());
            const auto physicalOwner = std::find_if(sourceBoundary->pooledStorage()->originalOwners().begin(),
                sourceBoundary->pooledStorage()->originalOwners().end(), [&](const auto& owner) {
                    return owner->plan().allocator() == gpu;
                });
            if (physicalOwner == sourceBoundary->pooledStorage()->originalOwners().end()) {
                EXPECT_EQ(geometry.gpuCacheBytes(gpu), 0u);
                EXPECT_EQ(geometry.mandatoryDeviceBytes(gpu), 0u);
                for (const auto& row : memory->rankAttestation()) {
                    if (row.identity != gpu || (row.owner != PhysicalMemoryOwner::ImmutableTableStorage &&
                        row.owner != PhysicalMemoryOwner::ImmutableLookupStaging)) continue;
                    EXPECT_EQ(row.materialized_new_bytes + row.adopted_resident_bytes,
                        qwen4ExpSourceLocalHotLeaseBytes(*context,gpu,row.owner));
                }
            } else {
                // A continuation reader may own real evictable row slots, but
                // never another CPU table/channel/source boundary. Authenticate
                // those SAME roots once through the original source owner view.
                const auto actual = (*physicalOwner)->retainedOwnerObservation();
                EXPECT_EQ(actual.authority(), memory);
                EXPECT_FALSE(actual.authority().owner_before(memory));
                EXPECT_FALSE(memory.owner_before(actual.authority()));
                EXPECT_EQ(actual.allocator(), gpu);
                EXPECT_EQ(actual.tableStorageBytes(), geometry.gpuCacheBytes(gpu));
                EXPECT_EQ(actual.lookupStagingBytes(), geometry.mandatoryDeviceBytes(gpu));
                ASSERT_NO_FATAL_FAILURE(expect_original_lease(*memory, gpu,
                    PhysicalMemoryOwner::ImmutableTableStorage, actual.tableStorageBytes()));
                ASSERT_NO_FATAL_FAILURE(expect_original_lease(*memory, gpu,
                    PhysicalMemoryOwner::ImmutableLookupStaging, actual.lookupStagingBytes()));
                EXPECT_TRUE(participants->hasEmbeddedParticipant(gpu.device));
            }
            // The coordinator iteration below authenticates shared source and
            // provider statistics exactly once. Reader observations do not add
            // those bytes or channel counts to another consumer's totals.
            return;
        }
    }
    const auto boundary=context->adaptiveImmutableLookupBoundary(gpu.device);ASSERT_TRUE(boundary);
    ASSERT_TRUE(boundary->isPooled());EXPECT_EQ(boundary->device(),gpu.device);
    EXPECT_EQ(context->adaptiveImmutableLookupBoundary(gpu.device).get(),boundary.get());
    const auto provider=context->hostImmutableRowProvider();ASSERT_TRUE(provider);
    EXPECT_EQ(provider->mode(),request.sourceStorage()==ImmutableRowStorage::RAM
        ? HostImmutableRowMode::AdaptiveFullRAM : HostImmutableRowMode::AdaptiveRAMSSD);
    if (grouped) {
        EXPECT_FALSE(provider->subordinateRequester());
        const auto requesters=provider->subordinateRequesters();
        EXPECT_EQ(std::vector<PhysicalMemoryAllocatorIdentity>(requesters.begin(),requesters.end()),
            std::vector<PhysicalMemoryAllocatorIdentity>(layout.consumers().begin(),layout.consumers().end()));
    } else EXPECT_EQ(provider->subordinateRequester(),std::optional{gpu});
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
    const auto geometry=grouped ? AdaptiveImmutableLookupGeometry::forRequester(layout,provider->source(),request,gpu)
        : AdaptiveImmutableLookupGeometry::fromLayoutWithLocalHot(layout,provider->source(),request,placement->gpuAdaptiveLocalHotCaches());
    const auto ram_slots=geometry.host().slots();EXPECT_LE(ram_slots,request.host.maximum_cache_rows);
    const auto setup=context->immutableLookupSetupObservation();
    ASSERT_NE(setup,nullptr) << "Ready source requires original immutable admitted-capacity evidence";
    EXPECT_EQ(setup->mode(),provider->mode());EXPECT_EQ(setup->source(),provider->source());
    std::size_t setupHostOwners=0;
    for(const auto& row:setup->owners()) {
        EXPECT_EQ(row.table_storage_bytes,row.payload_bytes+row.metadata_bytes+row.source_storage_bytes);
        if(row.role==ImmutableLookupSetupObservation::Role::SharedHostSource) {
            ++setupHostOwners;
            EXPECT_EQ(row.allocator,layout.sourceServices().front().allocator);
            EXPECT_EQ(row.admitted_slots,ram_slots);
            const auto actual=boundary->retainedSourceOwnerObservation();
            EXPECT_EQ(row.table_storage_bytes,actual.tableStorageBytes());
            EXPECT_EQ(row.lookup_staging_bytes,actual.lookupStagingBytes());
        }
    }
    EXPECT_EQ(setupHostOwners,1u); // Per-consumer borrows never multiply the shared source/channel block.

    ASSERT_EQ(geometry.physical().owners().size(),expected_owner_count);
    for(std::size_t coordinate=0;coordinate<expected_owner_count;++coordinate)
    {
        const auto &owner=geometry.physical().owners()[coordinate];
        const auto maximum=declared?request.placement.owners[coordinate].maximum_cache_rows:request.maximum_gpu_cache_rows;
        const auto ceiling=declared?request.placement.owners[coordinate].cache_physical_byte_limit:request.gpu_cache_physical_byte_limit;
        const auto endpoint=declared?PhysicalMemoryAllocatorIdentity{request.placement.owners[coordinate].participant.world_rank,
            request.placement.owners[coordinate].participant.device()}:gpu;
        EXPECT_EQ(owner.allocator,endpoint);EXPECT_LE(owner.slots,maximum);
        if(maximum && ceiling)EXPECT_GT(owner.slots,0u)<<"Positive owner cell cannot certify a zero optional grant";
        else EXPECT_EQ(owner.slots,0u);
        EXPECT_LE(geometry.gpuSlotDependentBytes(endpoint),ceiling);
        const auto recorded=std::find_if(setup->owners().begin(),setup->owners().end(),[&](const auto& row){
            return row.role==ImmutableLookupSetupObservation::Role::GPUCache && row.allocator==endpoint;});
        ASSERT_NE(recorded,setup->owners().end());
        EXPECT_EQ(recorded->admitted_slots,owner.slots);
        EXPECT_EQ(recorded->table_storage_bytes,owner.cache.bytes);
        EXPECT_EQ(recorded->metadata_bytes,owner.cache.payload);
        EXPECT_EQ(recorded->payload_bytes,owner.cache.bytes-owner.cache.payload);

    }
    if(request.host.maximum_cache_rows && request.host.cache_physical_byte_limit)
    {
        EXPECT_GT(ram_slots,0u) << "Positive RAM-cache cell cannot certify a zero optional grant";
    }
    else { EXPECT_EQ(ram_slots,0u); }
    const auto storage=boundary->pooledStorage();
    ASSERT_TRUE(storage);
    // Observations must not create the missing worker or retrofit source
    // ownership. A merely bound source is insufficient: the loader retains
    // original use permits only after actual native owner/coordinator setup.
    const auto native_source=context->nativeContextPreparation();
    const auto& consumed=storage->retainedPreparedSourceWorkers();
    ASSERT_NO_THROW((void)requireQwen4ExpConsumedPreparedSource(native_source,consumed));
    EXPECT_EQ(consumed->source(),native_source);
    EXPECT_FALSE(consumed->source().owner_before(native_source));
    EXPECT_FALSE(native_source.owner_before(consumed->source()));
    ASSERT_TRUE(retained->physical_memory_authority);
    const auto& original_memory=retained->physical_memory_authority;
    EXPECT_EQ(native_source->memory(),original_memory);
    EXPECT_FALSE(native_source->memory().owner_before(original_memory));
    EXPECT_FALSE(original_memory.owner_before(native_source->memory()));
    EXPECT_EQ(consumed->rootIdentity(),storage->captureIdentity());
    EXPECT_NE(consumed->rootIdentity(),0u);
    const auto original_attestation=original_memory->rankAttestation();
    // Exact mapped stack/object pages remain source preparation materialization
    // on the original host-root owner. Opaque SDK bounds are separate canonical
    // commitments, never invented physical backing or an equality to RSS.
    const auto exact_stack=std::find_if(original_attestation.begin(),original_attestation.end(),[&](const auto& row){
        return row.identity==host && row.owner==PhysicalMemoryOwner::LocalCollective && row.preparation_materialized_bytes>0;
    });
    ASSERT_NE(exact_stack,original_attestation.end()) << "Original exact source-owned worker stack/object root was not materialized";
    EXPECT_TRUE(exact_stack->committed());
    EXPECT_LE(exact_stack->preparation_materialized_bytes,exact_stack->materialized_new_bytes);
    EXPECT_TRUE(std::any_of(original_attestation.begin(),original_attestation.end(),[&](const auto& row){
        return row.identity.world_rank==original_memory->worldRank() && row.owner==PhysicalMemoryOwner::NativeExecutionContext &&
            row.opaque_runtime_committed_bytes>0 && row.committed();
    })) << "Original source worker has no named qualified opaque family commitment";
    const auto require_original_worker=[&](const PreparedGPUContextUsePermit& actual,DeviceId expected){
        ASSERT_EQ(actual.device(),expected);
        ASSERT_EQ(actual.memory(),original_memory);
        EXPECT_FALSE(actual.memory().owner_before(original_memory));
        EXPECT_FALSE(original_memory.owner_before(actual.memory()));
        // This public existing-only borrow cannot discover, create or admit a
        // worker. It authenticates the original pool/source/stack-object root;
        // provenance rejects a replacement pointer or shared ownership block.
        const auto existing=GPUDeviceContextPool::instance().retainExistingPreparedContext(expected,original_memory);
        EXPECT_EQ(actual.generation(),existing.generation());
        EXPECT_NE(actual.generation(),0u);
        EXPECT_EQ(actual.provenance().device(),expected);
        EXPECT_NO_THROW(actual.requireWorker(existing.worker()));
        EXPECT_NO_THROW(existing.requireWorker(actual.worker()));
        EXPECT_EQ(actual.worker()->defaultStream(),existing.worker()->defaultStream());
        EXPECT_GT(reinterpret_cast<std::uintptr_t>(actual.worker()->defaultStream()),std::uintptr_t{2});
        const auto native_context=std::find_if(original_attestation.begin(),original_attestation.end(),[&](const auto& row){
            return row.identity.world_rank==original_memory->worldRank() && row.identity.device==expected &&
                row.owner==PhysicalMemoryOwner::NativeExecutionContext;
        });
        ASSERT_NE(native_context,original_attestation.end());
        EXPECT_TRUE(native_context->committed());
    };
    ASSERT_NO_FATAL_FAILURE(require_original_worker(consumed->coordinator(),gpu.device));
    ASSERT_EQ(consumed->owners().size(),geometry.physical().owners().size());
    for(std::size_t coordinate=0;coordinate<consumed->owners().size();++coordinate){
        const auto& actual_owner=geometry.physical().owners()[coordinate].allocator;
        EXPECT_EQ(actual_owner.world_rank,original_memory->worldRank());
        ASSERT_NO_FATAL_FAILURE(require_original_worker(consumed->owners()[coordinate],actual_owner.device));
    }
    ASSERT_EQ(storage->originalOwners().size(),expected_owner_count);
    std::vector<::llaminar2::qwen4exp::PooledImmutableNativeRoots> roots;
    for(std::size_t coordinate=0;coordinate<expected_owner_count;++coordinate)
    {
        const auto& owner=storage->originalOwners()[coordinate];ASSERT_TRUE(owner);
        roots.push_back(owner->roots());EXPECT_EQ(bool(roots.back().cache),geometry.physical().owners()[coordinate].slots!=0);
    }
    roots.push_back(storage->coordinator()->roots());roots.push_back(storage->roots());
    for(const auto& group:roots)for(const auto& root:{group.cache,group.frame,group.native})if(root){
        EXPECT_TRUE(root->is_raw_data_released());EXPECT_NE(root->gpu_data_ptr(),nullptr);
    }
    const auto gpu_owner=storage->retainedOwnerObservation();
    const auto source_owner=boundary->retainedSourceOwnerObservation();
    const auto provider_owner=provider->retainedOwnerObservation();
    ASSERT_TRUE(retained->physical_memory_authority);
    const auto &memory=retained->physical_memory_authority;
    ASSERT_NO_THROW(storage->requireFor(geometry.physical(),geometry.coordinator(),provider->source(),geometry.computeReaderCount(),memory));
    EXPECT_EQ(gpu_owner.authority().get(),memory.get());EXPECT_EQ(source_owner.authority().get(),memory.get());
    EXPECT_EQ(provider_owner.authority().get(),memory.get());
    EXPECT_EQ(gpu_owner.allocator(),gpu);EXPECT_EQ(source_owner.allocator(),host);
    EXPECT_EQ(source_owner.hostPlacement(),placement->hostPlacement());
    EXPECT_EQ(source_owner.tableStorageBytes(),provider_owner.tableStorageBytes());
    EXPECT_GE(source_owner.lookupStagingBytes(),provider_owner.lookupStagingBytes());
    std::vector<DeviceId> channelDevices;
    if (shared) channelDevices.push_back(gpu.device);
    else for(const auto consumer:layout.consumers())channelDevices.push_back(consumer.device);
    EXPECT_EQ(source_owner.lookupStagingBytes()-provider_owner.lookupStagingBytes(),
        grouped ? ::llaminar2::qwen4exp::ImmutableLookupServiceGeometry::checked(geometry.channelLayout(),channelDevices).hostBytes()
                : ::llaminar2::qwen4exp::ImmutableLookupService::hostBytes(geometry.channelLayout()));
    EXPECT_EQ(gpu_owner.tableStorageBytes(),geometry.gpuCacheBytes(gpu));
    EXPECT_EQ(gpu_owner.lookupStagingBytes(),geometry.mandatoryDeviceBytes(gpu));
    for(std::size_t coordinate=0;coordinate<expected_owner_count;++coordinate)
    {
        const auto endpoint=geometry.physical().owners()[coordinate].allocator;
        if(endpoint==gpu)continue; // The same original source observation already includes local owner+coordinator roots.
        const auto actual=storage->originalOwners()[coordinate]->retainedOwnerObservation();
        EXPECT_EQ(actual.authority(),original_memory);EXPECT_EQ(actual.allocator(),endpoint);
        EXPECT_EQ(actual.tableStorageBytes(),geometry.gpuCacheBytes(endpoint));
        EXPECT_EQ(actual.lookupStagingBytes(),geometry.mandatoryDeviceBytes(endpoint));
        ASSERT_NO_FATAL_FAILURE(expect_original_lease(*original_memory,endpoint,
            PhysicalMemoryOwner::ImmutableTableStorage,actual.tableStorageBytes()));
        ASSERT_NO_FATAL_FAILURE(expect_original_lease(*original_memory,endpoint,
            PhysicalMemoryOwner::ImmutableLookupStaging,actual.lookupStagingBytes()));
    }
    EXPECT_LE(geometry.host().storageBytes(),request.host.cache_physical_byte_limit);
    ASSERT_NO_FATAL_FAILURE(expect_original_lease(*memory, gpu,
        PhysicalMemoryOwner::ImmutableTableStorage, gpu_owner.tableStorageBytes()));
    ASSERT_NO_FATAL_FAILURE(expect_original_lease(*memory, gpu,
        PhysicalMemoryOwner::ImmutableLookupStaging, gpu_owner.lookupStagingBytes()));
    ASSERT_NO_FATAL_FAILURE(expect_original_lease(*memory, host,
        PhysicalMemoryOwner::ImmutableTableStorage, source_owner.tableStorageBytes()));
    ASSERT_NO_FATAL_FAILURE(expect_original_lease(*memory, host,
        PhysicalMemoryOwner::ImmutableLookupStaging, source_owner.lookupStagingBytes()));
}


/** @brief Authenticate every actual consumer and the same original shared host owner, without acquiring new resources. */
inline void expectQwen4ExpSourceGPUAdaptiveOwners(const ModelParityCase &cell, IOrchestrationRunner &runner,
    const ModelContextReuseContract &original)
{
    ASSERT_TRUE(original.context);
    const auto placement=original.context->resolvedImmutableRowTablePlacement();
    ASSERT_TRUE(placement);
    for(const auto consumer:placement->gpuAdaptiveLayout().consumers())
        ASSERT_NO_FATAL_FAILURE(expectQwen4ExpSourceGPUAdaptiveConsumer(cell,runner,original,consumer));
    ASSERT_NO_FATAL_FAILURE(expectQwen4ExpSourceLocalHotOwners(original));
}

/**
 * @brief Authenticate the selected lookup layout and actual retained owners.
 * @param cell Canonical table policy, without inferred tier substitutions.
 * @param runner Successfully initialized production runner and model authority.
 * @param original Same final export retained by Qwen4ExpSourceAdmissionLifetime.
 */
inline void expectQwen4ExpSourceLookup(const ModelParityCase &cell, IOrchestrationRunner &runner,
                                     const ModelContextReuseContract &original)
{
    ASSERT_TRUE(cell.ngram_table_policy);
    const auto *reuse = &original;
    const auto context = reuse->context;
    ASSERT_NE(context, nullptr);
    const auto memory = reuse->physical_memory_authority;
    ASSERT_NE(memory, nullptr);
    const auto placement = context->resolvedImmutableRowTablePlacement();
    if(cell.topology.expert_overlay_plan && MoEExpertOwnerMap::usesCPUJointFragments(*cell.topology.expert_overlay_plan))
    {
        int rank=-1;ASSERT_EQ(MPI_Comm_rank(MPI_COMM_WORLD,&rank),MPI_SUCCESS);
        const bool continuation=qwen4ExpCPUJointSourceHasContinuation(cell,rank);
        if(placement) for(const auto& consumer:placement->consumers())
            if(consumer.world_rank==rank) {
                EXPECT_TRUE(continuation) << "Expert-only CPU follower cannot become a PLE consumer";
                EXPECT_TRUE(consumer.device.is_gpu()) << "CPU fragments are not continuation PLE consumers";
            }
        if(!continuation) {
            EXPECT_EQ(context->hostImmutableRowProvider(),nullptr);
            if(placement) for(const auto& allocator:placement->allocators())EXPECT_NE(allocator.world_rank,rank);
            for(const auto& row:memory->rankAttestation())
                if(row.identity.world_rank==rank && (row.owner==PhysicalMemoryOwner::ImmutableTableStorage ||
                    row.owner==PhysicalMemoryOwner::ImmutableLookupStaging))
                    EXPECT_EQ(row.materialized_new_bytes+row.adopted_resident_bytes,0u);
            return; // No source/provider or consumer authority is fabricated for the expert follower.
        }
    }
    ASSERT_NE(placement, nullptr);
    ASSERT_NE(runner.executionPlan().runtime.ngram_table_placement, nullptr);
    EXPECT_EQ(*runner.executionPlan().runtime.ngram_table_placement, *placement);
    placement->requireRequestedPolicy(*cell.ngram_table_policy);
    EXPECT_EQ(placement->extent().rows(), 320001536ull);
    EXPECT_EQ(placement->extent().rowBytes(), 90u);
    EXPECT_EQ(placement->extent().nativeBytes(), 28800138240ull);
    if (placement->isGPUAdaptive())
    {
        ASSERT_NO_FATAL_FAILURE(expectQwen4ExpSourceGPUAdaptiveOwners(cell, runner, original));
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
    /** @brief Construct the normal runner under one owned cache namespace. @param config Public request. @param cache Owned prefix child.
     * @param original_context Same original observed MPI owner for CPU-joint setup; absent preserves ordinary construction. */
    Qwen4ExpSourceAdmissionLifetime(const OrchestrationConfig &config, std::filesystem::path cache,
        std::shared_ptr<IMPIContext> original_context = {})
        : cache_(std::move(cache)), runner_(original_context
            ? createOrchestrationRunnerFactory(std::move(original_context))->createFromOrchestrationConfig(config)
            : test::TestOrchestrationHelper::create(config))
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
    /**
     * @brief Recreate the next actual runner from the SAME original reuse contract.
     * Old graph/parent borrowers retire before this returns; immutable source,
     * prepared weights, physical caches and PMA remain original model owners.
     * @param config Identical public policy, revalidated by the normal factory.
     */
    void recreateFromOriginalContract(const OrchestrationConfig& config)
    {
        if (finished_ || !runner_) throw std::logic_error("Original source lifetime is not reusable");
        if (!reuse_) reuse_ = runner().modelContextReuseContract();
        if (!reuse_ || !reuse_->context) throw std::logic_error("Original source has no reusable model contract");
        runner_->shutdown();
        if (runner_->isInitialized()) throw std::runtime_error("Original runner native retirement remains unproven");
        runner_.reset();
        runner_ = test::TestOrchestrationHelper::create(config, *reuse_);
        if (!runner_) throw std::runtime_error("Original source reuse factory returned no second runner");
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
            if(reuse_->reuse_authority && reuse_->reuse_authority->state()==ModelContextReuseAuthority::State::Invalid)
                retired_devices_=discardExclusiveInvalidModelContextReuseContract(*reuse_).retiredDeviceCount();
            else
                retired_devices_=retireExclusiveModelContextReuseContract(*reuse_).retiredDeviceCount();
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
 * @param rank Original MPI rank used to authenticate continuation versus expert-only source role.
 * @param authored Original projected policy before planning.
 * @param rank Original MPI rank used to authenticate continuation or expert-only ownership.
 * @param runner_value Initialized real source runner.
 * @param reuse_value Original retained model/PMA certificate.
 * @param output Owned source evidence directory.
 * @param setup_before Snapshot from before this runner initialized.
 *
 * Authenticates sources, sidecar engines and plan-owned forward setup only;
 * Disabled execution may retain predictor weights through positive graph capacity.
 * Only enabled execution requires speculative serving setup-family evidence;
 * no inference or generation-controller execution occurs here.
 */
inline void expectQwen4ExpSourceMTPMaterialization(const ModelParityCase &cell, int rank,
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
    const auto predictor_materialization = parityPredictorMaterialization(authored.mtp);
    const bool retained_predictor = predictor_materialization == SharedPredictorMaterialization::Active;
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
        EXPECT_EQ(loader.predictorMaterialization(), predictor_materialization);
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
                if (retained_predictor && source.consumed)
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
        if (retained_predictor)
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
        if (retained_predictor)
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
        const auto expected = qwen4ExpMTPSourceServingParticipants(cell, rank, authored, runner->executionPlan(),
            *DeviceManager::instance().hardware(), programs, host_setup);
        planned_serving_participants = std::count_if(expected.begin(), expected.end(), [](const auto &leaf)
            { return leaf.role == Qwen4ExpMTPSourceParticipantRole::Continuation; });
        deferred_host_participants = std::count_if(expected.begin(), expected.end(), [](const auto &leaf)
                                                   { return leaf.role == Qwen4ExpMTPSourceParticipantRole::Continuation && !leaf.completion_required; });
        ASSERT_NO_THROW(setup_participants = requireQwen4ExpMTPSourceSetupFamilies(
                            setup_before, PerfStatsCollector::snapshot(), expected));
    }
    const auto placement = model->resolvedImmutableRowTablePlacement();
    ASSERT_NE(placement, nullptr) << "QFN CPU materialization lost its original immutable lookup placement";
    ASSERT_NO_THROW(
        requireQwen4ExpCPUStorageMaterialization(reuse->physical_memory_authority->rankAttestation(),
            reuse->prepared_routed_weight_plan, placement->allocators()));
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
    const auto output = base / "source-admission" / parity::artifactIdentityPath(cell.testName()) / ("rank-" + std::to_string(rank));
    ASSERT_NO_THROW(prepareQwen4ExpAdmissionEvidenceDirectory(output));
    config.prefix_cache.disk_dir = (output / "prefix-cache").string();
    auto authored = config;
    if (config.moe_routed_expert_plan)
        authored.moe_routed_expert_plan = std::make_shared<MoERoutedExpertPlacementPlan>(*config.moe_routed_expert_plan);
    std::shared_ptr<IMPIContext> original_cpu_joint_context;
    if(!qwen4ExpCPUJointSocketMembers(cell).empty()) {
        // Keep the SAME factory/context alive through the Runner: its one
        // published rank observation is consumed here and later by planning.
        original_cpu_joint_context=MPIContextFactory::global();
        ASSERT_TRUE(original_cpu_joint_context);
        const auto inventory=original_cpu_joint_context->clusterInventory();ASSERT_TRUE(inventory);
        std::vector<test::CPUExpertSocketEndpoint> endpoints;
        ASSERT_NO_THROW(endpoints=requireOriginalCPUJointSocketRanks(cell,*inventory));
        const auto endpoint=std::find_if(endpoints.begin(),endpoints.end(),[&](const auto& item){return item.rank==rank;});
        ASSERT_NE(endpoint,endpoints.end());
        const auto original_rank=std::find_if(inventory->ranks.begin(),inventory->ranks.end(),
            [&](const auto& item){return item.rank==rank;});
        ASSERT_NE(original_rank,inventory->ranks.end());ASSERT_GT(original_rank->cpu_worker_threads,0);
        std::ofstream socket_evidence(output / "cpu-joint-socket-membership.csv");ASSERT_TRUE(socket_evidence.is_open());
        socket_evidence << "cell,rank,original_node_id,numa_node,socket_id,worker_threads\n"
            << cell.testName() << ',' << rank << ',' << original_rank->node_id << ',' << endpoint->numa_node
            << ',' << endpoint->socket << ',' << original_rank->cpu_worker_threads << '\n';
        socket_evidence.flush();ASSERT_TRUE(socket_evidence.good());
    }
    Qwen4ExpSourceAdmissionLifetime lifetime(config, output / "prefix-cache",original_cpu_joint_context);
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
 * @brief Authenticate each local CPU fragment from the original frozen plan and source-bound registry.
 * @param rank Actual original world rank.
 * @param original Retained model/PMA/placement export from successful admission.
 * @note No owner map is rebuilt and no source bytes are loaded or hashed.
 */
inline void expectQwen4ExpCPUJointSourceFragments(int rank,const ModelContextReuseContract& original)
{
    const auto plan=original.prepared_routed_weight_plan;
    if(!plan || !MoEExpertOwnerMap::usesCPUJointFragments(*plan))return;
    ASSERT_TRUE(original.context);ASSERT_TRUE(original.physical_memory_authority);
    const auto manager=original.context->concreteWeightManager();ASSERT_TRUE(manager);
    const auto store=manager->preparedWeightStoreIfInitialized();ASSERT_TRUE(store);
    EXPECT_EQ(store->physicalMemoryAuthority(),original.physical_memory_authority);
    auto& registry=manager->expertGemmRegistry();
    const auto participants=MoEExpertOwnerMap::declaredParticipants(*plan,{.reject_tensor_sharded_domains=false});
    const auto expected_observations=qwen4ExpCPUJointSourceExpectedLayerObservations(*plan,participants,rank);
    size_t observed=0;
    for(const auto& participant:participants) {
        if(participant.world_rank!=rank || !qwen4ExpOriginalCPUJointParticipant(*plan,participant))continue;
        for(const auto& layer:plan->placements) {
            const auto lease=registry.acquireCPUJointLayer(store->modelId(),participant,layer.layer);
            ASSERT_TRUE(lease);ASSERT_TRUE(lease->cpu_joint_source);const auto& source=*lease->cpu_joint_source;
            EXPECT_EQ(source.participant(),participant);EXPECT_EQ(source.layer(),layer.layer);
            ASSERT_TRUE(source.cohort());EXPECT_EQ(source.cohort()->source().layer_idx,layer.layer);
            EXPECT_EQ(source.ownership().geometry().experts,512);
            EXPECT_EQ(source.participantIndex(),size_t(participant.domain_participant_index));
            std::vector<int> expected;
            for(size_t expert=0;expert<layer.routed_expert_tier.size();++expert)
                if(layer.routed_expert_tier[expert]==participant.tier_idx)expected.push_back(static_cast<int>(expert));
            if(plan->routed_tiers.at(participant.tier_idx).max_experts_per_layer>0)
                EXPECT_EQ(expected.size(),size_t(plan->routed_tiers.at(participant.tier_idx).max_experts_per_layer));
            EXPECT_EQ(std::vector<int>(source.expertIds().begin(),source.expertIds().end()),expected);
            ASSERT_EQ(lease->experts.size(),layer.routed_expert_tier.size());
            EXPECT_EQ(lease->sources.size(),expected.size()*3u);
            for(size_t expert=0;expert<lease->experts.size();++expert) {
                const auto& engine=lease->experts[expert];const bool selected=source.contains(static_cast<int>(expert));
                EXPECT_EQ(bool(engine.gate),selected);EXPECT_EQ(bool(engine.up),selected);EXPECT_EQ(bool(engine.down),selected);
            }
            for(const auto& binding:lease->sources) {
                EXPECT_EQ(binding.identity.model_id,store->modelId());ASSERT_TRUE(binding.cpu_joint_source);
                EXPECT_EQ(*binding.cpu_joint_source,source);ASSERT_TRUE(binding.projection_ownership);
                EXPECT_EQ(*binding.projection_ownership,source.ownership());
            }
            ++observed;
        }
    }
    // A genuine other participant can have no CPU-joint member. Its expected
    // source observations are zero, not a fabricated local fragment lease.
    EXPECT_EQ(observed,expected_observations);
}

/** @brief Existing initialization/recreation observation points; neither proves inference. */
enum class Qwen4ExpSourceInspectionPhase { Original, RetainedReplacement };
/** @brief Optional immutable view; callback-local aliases must expire before original retirement. */
using Qwen4ExpSourceInspection = std::function<void(const IOrchestrationRunner&,
    Qwen4ExpSourceInspectionPhase, const std::filesystem::path&)>;

/**
 * @brief Materialize one original-source serving configuration without gold or inference.
 * @param cell Complete existing typed source/topology/precision/storage declaration.
 *
 * Successful setup authenticates exact original sources, actual execution plan,
 * retained native owners and snapshot-zero PMA before final exclusive retirement.
 */
/** @brief Materialize the original full source and optionally require its existing cold recreation proof.
 * @param cell Original real-model source request; no test creates a model authority.
 * @param recreation Require original role eligibility and execute the same cold lifecycle assertions.
 * @param inspect Optional read-only observation after the original mandatory phase checks; failures do not skip recreation/retirement.
 */
inline void admitQwen4ExpSource(const ModelParityCase &cell,
    Qwen4ExpSourceRecreationProof recreation = Qwen4ExpSourceRecreationProof::FromOriginalRoles,
    const Qwen4ExpSourceInspection &inspect = {})
{
    ASSERT_TRUE(recreation==Qwen4ExpSourceRecreationProof::FromOriginalRoles ||
        recreation==Qwen4ExpSourceRecreationProof::Required);
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
        cell,rank,authored,*runner,*reuse,output,setup_before));
    if (config.moe_routed_expert_plan)
    {
        ASSERT_NE(reuse->prepared_routed_weight_plan, nullptr);
        ASSERT_NE(reuse->expert_overlay_memory_admission, nullptr);
        EXPECT_FALSE(reuse->routed_weight_authority_identity.empty());
    }
    for (const auto &resource : reuse->physical_memory_authority->admission()->plan().resources())
        EXPECT_EQ(resource.bytes(PhysicalMemoryOwner::GraphSnapshotArena), 0u);
    ASSERT_NO_FATAL_FAILURE(expectQwen4ExpCPUJointSourceFragments(rank,*reuse));
    ASSERT_NO_FATAL_FAILURE(expectQwen4ExpSourceLookup(cell, *runner, *reuse));
    std::vector<::llaminar2::qwen4exp::PooledImmutableRetainedParticipants::ExpertOwnerParentObservation> old_expert_parents;
    ASSERT_NO_FATAL_FAILURE(expectQwen4ExpSourceExpertLookupParents(cell,*reuse,output/"ngram-expert-owner-attachments.csv",old_expert_parents));
    if(const auto setup=model->immutableLookupSetupObservation()) {
        // Serialize the SAME passive setup value used by the assertions and
        // benchmark family, once per rank, independent of consumer borrows.
        std::ofstream grants(output / "ngram-admitted-setup.csv");ASSERT_TRUE(grants.is_open());
        grants << "world_rank,device,role,admitted_slots,payload_bytes,metadata_bytes,source_storage_bytes,table_storage_bytes,lookup_staging_bytes,source_file_offset,source_rows,source_row_elements\n";
        for(const auto& row:setup->owners())
            grants << row.allocator.world_rank << ',' << row.allocator.device.toString() << ',' << static_cast<unsigned>(row.role)
                << ',' << row.admitted_slots << ',' << row.payload_bytes << ',' << row.metadata_bytes << ',' << row.source_storage_bytes
                << ',' << row.table_storage_bytes << ',' << row.lookup_staging_bytes << ',' << setup->source().file_offset
                << ',' << setup->source().rows << ',' << setup->source().row_elements << '\n';
        grants.flush();ASSERT_TRUE(grants.good());
    }

    const auto placement = model->resolvedImmutableRowTablePlacement();
    ASSERT_NE(placement, nullptr) << "QFN CPU materialization lost its original immutable lookup placement";
    std::ofstream memory(output / "admission_memory.txt");
    ASSERT_TRUE(memory.is_open());
    memory << reuse->physical_memory_authority->rankAttestationSummary(true);
    ASSERT_NO_THROW(requireQwen4ExpCPUStorageMaterialization(
        reuse->physical_memory_authority->rankAttestation(), reuse->prepared_routed_weight_plan, placement->allocators()));
    memory.flush();
    ASSERT_TRUE(memory.good());
    // The original rank owns each allocator exactly once. Keep resource totals
    // separate from per-owner rows; included preparation/SDK detail is not an
    // additional physical allocation or a remote materialization observation.
    std::ofstream resource_cost(output / "admitted-resources.csv");
    std::ofstream owner_cost(output / "admitted-owners.csv");
    std::string source_identity;
    for (const auto& path : model->artifactPaths()) {
        if (!source_identity.empty()) source_identity += '\n';
        source_identity += path;
    }
    ASSERT_NO_THROW(::llaminar2::test::writeFinalPhysicalMemoryCostCSV(
        resource_cost, owner_cost, *reuse->physical_memory_authority,
        {cell.topology.test_id, source_identity, cell.testName()}));
    resource_cost.flush(); owner_cost.flush();
    ASSERT_TRUE(resource_cost.good()); ASSERT_TRUE(owner_cost.good());
    std::ofstream csv(output / "admission.csv");
    ASSERT_TRUE(csv.is_open());
    const bool joint_follower=cell.topology.expert_overlay_plan &&
        MoEExpertOwnerMap::usesCPUJointFragments(*cell.topology.expert_overlay_plan) &&
        !qwen4ExpCPUJointSourceHasContinuation(cell,rank);
    ASSERT_TRUE(joint_follower || placement);
    csv << "cell,layers,initialized,snapshots_enabled,numerical_inference_executed,native_table_bytes\n"
        << cell.testName() << ",48,1,0,0," << (joint_follower?0u:placement->extent().nativeBytes()) << '\n';
    csv.flush();
    ASSERT_TRUE(csv.good());
    // Inspection is additional evidence. A failed callback cannot bypass the
    // original recreation, source-owner verification or final retirement path.
    if (inspect) EXPECT_NO_THROW(inspect(*runner, Qwen4ExpSourceInspectionPhase::Original, output));
    const bool original_recreation = qwen4ExpSourceReusesContinuationOwners(placement.get()) || !old_expert_parents.empty();
    if (recreation == Qwen4ExpSourceRecreationProof::Required)
        ASSERT_TRUE(original_recreation) << "This source candidate requires original cold recreation/retirement";
    if (original_recreation) {
        // Keep only weak execution witnesses: a strong alias here would
        // itself prevent the runner-bound participant authority from dying.
        using Participants = ::llaminar2::qwen4exp::PooledImmutableRetainedParticipants;
        std::vector<std::weak_ptr<Participants>> old;
        std::vector<bool> participant_required;
        std::vector<std::shared_ptr<::llaminar2::qwen4exp::ImmutableLookupExecutionBoundary>> boundaries;
        for (const auto consumer : placement->consumers()) {
            const auto participants=model->adaptiveImmutableLookupParticipants(consumer.device);
            const auto boundary=model->adaptiveImmutableLookupBoundary(consumer.device);
            bool required=true; // A query-only reader borrows its original source's retained program.
            if(boundary) {
                ASSERT_TRUE(boundary->isPooled());
                const auto storage=boundary->pooledStorage();ASSERT_TRUE(storage);
                required=bool(model->adaptiveImmutableLookupLocalHot(consumer.device)) ||
                    ::llaminar2::qwen4exp::PooledImmutableParticipantExecutionPolicy::requiresRetainedParticipants(storage->geometry());
            }
            // ConsumerLocal's original singleton source needs no remote/reader
            // program. Preserve that absence rather than fabricate an owner;
            // local-hot, remote-owner and query-only reader roles still require it.
            ASSERT_EQ(bool(participants),required);
            if(participants)ASSERT_NO_THROW(participants->requireCurrent());
            old.emplace_back(participants);
            participant_required.push_back(required);
            boundaries.push_back(boundary);
        }
        const auto source = model->nativeContextPreparation();
        const auto setup = model->immutableLookupSetupObservation();
        lifetime.recreateFromOriginalContract(config);
        for (const auto& witness : old) EXPECT_TRUE(witness.expired())
            << "Old runner participants retained its borrowed Rank context";
        for(const auto& owner:old_expert_parents)
            for(const auto& parent:owner.parent_lifetimes)EXPECT_TRUE(parent.expired())
                << "Original expert-owner native parent survived runner retirement";
        runner = &lifetime.runner();
        runner->disableSnapshotCapture();
        ASSERT_TRUE(runner->initialize()) << runner->lastError();
        EXPECT_EQ(runner->modelContextForDiagnostics(), model.get());
        EXPECT_EQ(model->nativeContextPreparation(), source);
        EXPECT_EQ(model->concreteWeightManager()->preparedWeightStoreIfInitialized(), store);
        EXPECT_EQ(model->immutableLookupSetupObservation(), setup);
        const auto& second = lifetime.reuseContract();
        ASSERT_TRUE(second);EXPECT_EQ(second->physical_memory_authority, reuse->physical_memory_authority);
        std::size_t coordinate = 0;
        for (const auto consumer : placement->consumers()) {
            const auto current = model->adaptiveImmutableLookupParticipants(consumer.device);
            ASSERT_EQ(bool(current),participant_required[coordinate]);
            if(current) {
                ASSERT_NO_THROW(current->requireCurrent());
                EXPECT_TRUE(old[coordinate].owner_before(current) || current.owner_before(old[coordinate]));
            }
            EXPECT_EQ(model->adaptiveImmutableLookupBoundary(consumer.device), boundaries[coordinate]);
            ++coordinate;
        }
        ASSERT_NO_FATAL_FAILURE(expectQwen4ExpSourceLookup(cell, *runner, *second));
        std::vector<::llaminar2::qwen4exp::PooledImmutableRetainedParticipants::ExpertOwnerParentObservation> replacement;
        ASSERT_NO_FATAL_FAILURE(expectQwen4ExpSourceExpertLookupParents(cell,*second,output/"ngram-expert-owner-attachments-replacement.csv",replacement));
        ASSERT_EQ(replacement.size(),old_expert_parents.size());
        for(const auto& actual:replacement){
            const auto before=std::find_if(old_expert_parents.begin(),old_expert_parents.end(),
                [&](const auto& owner){return owner.allocator==actual.allocator;});
            ASSERT_NE(before,old_expert_parents.end());
            EXPECT_EQ(actual.worker_generation,before->worker_generation);
            EXPECT_EQ(actual.memory_identity,before->memory_identity);
            EXPECT_EQ(actual.owner_coordinate,before->owner_coordinate);
            EXPECT_EQ(actual.source_variant_count,before->source_variant_count);
            EXPECT_EQ(actual.parent_lifetimes.size(),before->parent_lifetimes.size());
            for(const auto& parent:actual.parent_lifetimes)
                for(const auto& previous:before->parent_lifetimes)
                    EXPECT_TRUE(parent.owner_before(previous) || previous.owner_before(parent))
                        << "Replacement expert parent reused an old attachment control block";
            EXPECT_FALSE(actual.worker.owner_before(before->worker));
            EXPECT_FALSE(before->worker.owner_before(actual.worker));
        }
        if (inspect) EXPECT_NO_THROW(inspect(*runner, Qwen4ExpSourceInspectionPhase::RetainedReplacement, output));
        if(!::testing::Test::HasFailure())
            ::testing::Test::RecordProperty("original_shared_source_cold_recreation","completed");
    }
    });
}

/**
 * @brief Prove explicit physical capacity rejection without claiming unsupported execution.
 * @param cell Exact larger request retaining the same head/expert/MTP declarations.
 * @param gpu_memory_cap_mb Authored public ceiling; applied before any capacity planning.
 * @param boundary Exact production physical-admission boundary required by this control.
 */
inline void admitQwen4ExpSourceCapacityRejected(const ModelParityCase &cell, std::size_t gpu_memory_cap_mb,
    Qwen4ExpSourceCapacityBoundary boundary)
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
        ASSERT_TRUE(qwen4ExpSourceCapacityDiagnosticMatches(boundary, error)) << error;
        ASSERT_FALSE(runner.modelContextReuseContract().has_value());
        ASSERT_TRUE(runner.getSnapshotKeys().empty());
        std::ofstream diagnostic(output / "capacity-diagnostic.txt");
        ASSERT_TRUE(diagnostic.is_open());
        diagnostic << error << '\n';
        diagnostic.flush();
        ASSERT_TRUE(diagnostic.good());
        std::ofstream receipt(output / "capacity-rejection.csv");
        ASSERT_TRUE(receipt.is_open());
        receipt << "cell,initialized,capacity_rejected,gpu_memory_cap_mb,numerical_inference_executed,capacity_boundary\n"
            << cell.testName() << ",0,1," << gpu_memory_cap_mb << ",0,"
            << qwen4ExpSourceCapacityBoundaryName(boundary) << '\n';
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
