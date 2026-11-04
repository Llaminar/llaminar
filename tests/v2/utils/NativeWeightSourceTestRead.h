/**
 * @file NativeWeightSourceTestRead.h
 * @brief Caller-owned admitted input arena for actual common-reader contract tests.
 *
 * This fixture contains no file reader, codec or batching implementation. The
 * production source projector/iterator, resolver and ModelLoader perform those
 * operations. It declares one real PMA root and retires it before unlocking
 * the original source owner; the callback cannot transfer arena ownership.
 */
#pragma once
#include "loaders/NativeWeightLoadStaging.h"
#include "loaders/WeightPlan.h"
#include "CPUJointSourceTestPlan.h"
#include "memory/NUMAAllocator.h"
#include "execution/moe/CPUExpertShardSourceSelection.h"
#include "tensors/NativeWeightSourceRange.h"
#include <algorithm>
namespace llaminar2::test {
/**
 * @return One discovered local allocation node retained for this component fixture.
 * @note A later scheduler migration must not silently change the authored
 * source identity. This is not an affinity or remote NUMA readiness claim.
 */
inline int localNativeSourceTestNUMANode() {
    static const int node=NUMAAllocator::instance().getCurrentNUMANode();
    return node;
}
/**
 * @return Original topology with one local actor and a distinct metadata-only peer.
 * @note The peer grants no allocation or socket readiness. Genuine two-process
 * execution has its separate native gate.
 */
inline MoERoutedExpertPlacementPlan localCPUJointSourceTestPlan() {
    auto plan=cpuJointSourceTestPlan();
    const int node=localNativeSourceTestNUMANode();
    for(auto& domain:plan.domains)if(domain.name=="cpu_joint")
        domain.participants={GlobalDeviceAddress::cpu(node),GlobalDeviceAddress::cpu(node+1)};
    return plan;
}
/** @return Same normalized original source requirement authored by the closed cohort selection. */
inline WeightRequirement nativeJointRequirementForTest(
    const std::shared_ptr<const CPUExpertShardSourceSelection>& selected,WeightRole role,const std::string& name)
{
    if(!selected)throw std::invalid_argument("Source fixture requires its original cohort selection");
    WeightRequirement request;request.canonical_name=name;request.role=role;request.layer=selected->layer();
    request.derivation=WeightDerivationKind::ExpertSlice;request.bypass_tensor_parallel=true;
    request.target_device=selected->participant().device;
    request.expected_prepared_kind=PreparedWeightKind::MoeExpertSlab;
    request.tp_rank_or_device_index=request.overlay_participant_index=int(selected->participantIndex());
    request.overlay_participant_world_rank=selected->participant().world_rank;
    request.overlay_domain=selected->participant().domain_name;
    request.cpu_joint_source=selected;request.projection_ownership=selected->ownership();
    WeightPlan plan;plan.add(std::move(request));return plan.requirements().front();
}
/** @return Exact native-only input quote for a reader test; no final consumer scratch is invented. */
inline NativeWeightLoadStagingRequest nativeReadRequestForTest(
    const NativeWeightSourceGeometry& source,const NativeWeightSourceWindow& range)
{
    if(!source.native_group_bytes || range.nativeBytes()%source.native_group_bytes)
        throw std::invalid_argument("Native read fixture requires original complete encoded units");
    return {source.native_group_bytes,0,0,range.nativeBytes()/source.native_group_bytes};
}
/**
 * @brief Visit real reader windows with one caller-declared arena, preserving untouched capacity.
 * @param request Actual consumer quote (or native-only input quote for byte-oracle reads).
 * @param consume Synchronous observer/writer of useful bytes and separate actual scratch.
 * @return Number of actual original reader invocations, not an estimated physical byte count.
 */
template<class Loader,class Consumer>
size_t visitNativeSourceForTest(Loader& loader,const std::string& name,const NativeWeightSourceWindow& range,
    NativeWeightLoadStagingRequest request,CPUStoragePlacement placement,
    const std::shared_ptr<PhysicalMemoryAuthority>& memory,DeviceId allocator,std::optional<size_t> cap,Consumer&& consume)
{
    const auto source=loader.nativeWeightSourceGeometry(name);
    const auto geometry=resolveNativeWeightLoadStaging(request,placement,cap);
    auto owner=memory->acquireCPUWeightSourceTransaction(allocator);
    const auto capacity=geometry.scratch_bytes?geometry.scratch_offset+geometry.scratch_bytes:geometry.native_bytes;
    auto arena=CPUStorageRoot<uint8_t>(capacity,PhysicalMemoryOwner::WeightLoadStaging,allocator,placement,
        CPUStorageInitialization::Uninitialized).allocate(memory);
    size_t calls=0;
    forEachNativeWeightSourceWindow(source,range,geometry.native_bytes,[&](const auto& window) {
        std::fill(arena.begin(),arena.end(),uint8_t{0xcd});
        auto* bytes=reinterpret_cast<std::byte*>(arena.data());
        loader.readNativeWeightSourceWindow(name,window,{bytes,window.nativeBytes()});
        if(!std::all_of(arena.begin()+window.nativeBytes(),arena.end(),[](uint8_t byte){return byte==0xcd;}))
            throw std::logic_error("Original native reader overwrote inactive source/scratch capacity");
        consume(window,std::span<const std::byte>(bytes,window.nativeBytes()),
            std::span<std::byte>(bytes+geometry.scratch_offset,geometry.scratch_bytes));
        ++calls;
    });
    return calls;
}
} // namespace llaminar2::test
