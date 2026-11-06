/** @file Test__PooledOriginalSourceSetup.cpp
 * @brief Native original-source/PMA setup for local C0/positive pooled owner, coordinator and boundary.
 *
 * This gate runs the actual source factories and native initialization kernels.
 * It deliberately measures no graph or timing family and claims no numerical
 * inference. Existing captured-provider bodies and their lookup byte oracles
 * remain a separate qualification requiring their own graph/auxiliary bounds.
 */
#include "fixtures/execution/RetainedHostServiceFixture.h"
#include "fixtures/backends/PreparedNativeLookupFixture.h"
#include "kernels/qwen4exp/PooledImmutableSourcePreparedStorage.h"
#include "kernels/qwen4exp/PooledImmutableSourceKernels.h"
#include "kernels/qwen4exp/PooledImmutableNativeKernels.h"
#include "execution/qwen4exp/ImmutableLookupExecutionBoundary.h"
#include "backends/NativeOrderingEventGroup.h"
#include "execution/qwen4exp/PooledImmutableDirectedTransport.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotKernels.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotPreparedStorage.h"
#include "fixtures/qwen4exp/PooledOriginalSourceSetupPrograms.h"
#include "fixtures/qwen4exp/PooledImmutableFactoryReplayAccess.h"
#include "backends/ExplicitGPUStream.h"
#include "loaders/CpuAdaptiveImmutableRowCache.h"
#include "loaders/HostImmutableRowProvider.h"
#include "utils/PooledNgramReplayTrace.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <array>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <map>
#include <limits>
#include <iostream>
#include <optional>
#include <unistd.h>
namespace llaminar2::test {namespace {
namespace q=qwen4exp;namespace n=q::pooled_native;namespace p=n::p;
#include "PooledImmutableProviderFile.inc"
/** @brief Read final setup bytes only at this explicit diagnostic boundary. */
template<class T>T observe(IBackend& backend,DeviceId device,void* stream,const std::shared_ptr<TensorBase>& root,std::size_t offset){
    T value{};requireProvider(root && offset+sizeof(T)<=root->size_bytes(),"Setup oracle exceeds original root");
    requireProvider(backend.deviceToHostOnStream(&value,static_cast<const std::byte*>(root->gpu_data_ptr())+offset,sizeof(T),device.ordinal,stream),"Setup byte observation failed");
    requireProvider(backend.synchronizeStream(stream,device.ordinal),"Setup byte observation did not retire");return value;
}
/** @brief Reuse actual prepared source roots twice under one admitted authority and native population. */
void originalSetup(DeviceId device,std::uint32_t slots,bool compactDeclaration=false){
    if(!hasCPUBackend())initCPUBackend(-1);
    const std::array devices{device};auto session=PreparedNativeLookupFixture::prepare(devices);
    const auto memory=session->memory();const auto gpu=session->resource(device),host=session->resource(DeviceId::cpu());
    ProviderFile file;const auto source=file.descriptor();const PhysicalMemoryAllocatorIdentity gpuId{0,device},hostId{0,DeviceId::cpu()};
    const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped};
    const auto layout=ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(source.format,source.rows,source.rowBytes()),
        ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,
        {{{gpuId,ImmutableRowStorage::VRAM},0,slots,{}},{{hostId,ImmutableRowStorage::RAM},0,slots?2u:0u,{}}},{gpuId},{},
        {{hostId,ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped,4096,2)}},{},slots?std::optional(gpuId):std::nullopt,16);
    const auto physical=q::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,std::array{gpuId},gpuId);
    const auto policy=q::PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},32);
    const auto cpu=CpuAdaptiveImmutableRowCacheGeometry::forSubordinateGPU(layout,hostId,gpuId);
    const auto channel=q::ImmutableLookupLayout::createSubordinate(16,source.rowBytes());
    const auto localGeometry=q::PooledImmutableLocalHotGeometry::declare(source,gpuId,32,slots,{65536,1,2});
    const auto compactGeometry=q::PooledImmutableLocalHotTransportGeometry::singleton(gpuId,32,
        ImmutableRowTableExtent::fromNative(source.format,source.rows,source.rowBytes()));
    PhysicalMemoryPlanBuilder payload;
    for(const auto& bom:{cpu.bom(host),physical.cacheBOM(0,gpu),physical.mandatoryBOM(0,gpu),policy.policyBOM(gpu),policy.mandatoryBOM(gpu),
        q::PooledImmutableOwnerNativePlan::mandatoryBOM(physical,policy,0,gpu),q::PooledImmutableCoordinatorNativePlan::mandatoryBOM(physical,policy,gpu),
        q::PooledImmutableSourceGeometry::mandatoryBOM(physical,policy,source,gpu)})for(const auto& row:bom.resources())payload.add(row);
    payload.add(host,PhysicalMemoryOwner::ImmutableLookupStaging,q::ImmutableLookupService::hostBytes(channel));
    constexpr unsigned Replays=2;
    const auto io=RetainedHostServiceProgram::original(RetainedHostServiceRole::ImmutableLookupIO);const auto ioBOM=io.bom(host);
    for(const auto& row:ioBOM.resources())payload.add(row.resource(),PhysicalMemoryOwner::RetainedHostServiceStorage,row.bytes(PhysicalMemoryOwner::RetainedHostServiceStorage));
    for(unsigned i=0;i<Replays;++i)for(const auto& row:ioBOM.resources())payload.add(row.resource(),PhysicalMemoryOwner::RetainedHostServiceRuntime,row.bytes(PhysicalMemoryOwner::RetainedHostServiceRuntime));
    const auto programs=originalPooledSourceSetupOrderingPrograms(physical,policy,source);
    // Explicit payload owners are serialized; each potentially retained native
    // group is independently declared before admission, never repriced later.
    for(unsigned replay=0;replay<Replays;++replay)for(const auto program:programs){
        const auto contribution=NativeOrderingEventGroup::bom(program,host,gpu);
        for(const auto& row:contribution.resources())payload.add(row);
    }
    if(compactDeclaration)for(unsigned replay=0;replay<Replays;++replay)
        for(const auto& bom:{localGeometry.mandatoryBOM(gpu,host),localGeometry.slotDependentBOM(gpu,host),
            compactGeometry.completeBOM(host,std::array{gpu})})for(const auto& row:bom.resources())payload.add(row);
    session->admit(payload.build());EXPECT_EQ(session->memory(),memory);
    const auto& permit=session->permit(device);auto& worker=*permit.worker();auto* backend=getBackendFor(device);
    const auto stream=ExplicitGPUStream(worker.defaultStream()).get();
    for(unsigned replay=0;replay<Replays;++replay){
        const RetainedHostServiceInstance ioKey{0,device,RetainedHostServiceRole::ImmutableLookupIO,replay};RetainedHostServiceDeclarations::Builder declare(0);declare.add(ioKey,io);
        const auto declared=std::move(declare).freeze();auto ioBinding=declared->bind(memory,std::span{&ioKey,std::size_t{1}});
        std::shared_ptr<q::ImmutableLookupExecutionBoundary> compactBoundary;
        std::shared_ptr<q::PooledImmutableLocalHotPreparedStorage> compactLocal;
        worker.submitAndWait([&]{
        auto provider=HostImmutableRowProvider::fromSubordinateAdaptiveCache(CpuAdaptiveImmutableRowCache::create(source,cpu,memory));
        auto service=q::ImmutableLookupService::create(provider,channel,std::array{device},memory,DeviceId::cpu(),ioBinding->take(ioKey));
        const auto binding=service->bindSubordinate(device);
        auto ownerPlan=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,policy,0,binding.deviceView().base.table,binding.storageGeneration());
        auto coordinatorPlan=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,policy,binding.deviceView().base.table,binding.storageGeneration());
        auto sourcePlan=q::PooledImmutableSourceGeometry::fromPlans(ownerPlan,coordinatorPlan,source,1);
        auto owner=q::PooledImmutableOwnerPreparedStorage::create(ownerPlan,memory,stream);
        auto coordinator=q::PooledImmutableCoordinatorPreparedStorage::create(coordinatorPlan,memory,stream);
        std::shared_ptr<const q::IPooledImmutableNativeKernels> code;
#ifdef HAVE_CUDA
        if(device.is_cuda())code=q::createCUDAPooledImmutableNativeKernels();
#endif
#ifdef HAVE_ROCM
        if(device.is_rocm())code=q::createROCmPooledImmutableNativeKernels();
#endif
        requireProvider(bool(code),"Original native setup kernel bundle unavailable");code->initialize(*owner,stream);code->initialize(*coordinator,stream);
        auto boundary=q::ImmutableLookupExecutionBoundary::createPooled(service,sourcePlan,owner,coordinator,memory,stream);
        if(compactDeclaration){
            // Reuse only the closed original-worker binder, with this REAL
            // admitted preparation and its actual native initialized roots.
            // No recorded creator, event claim or replay cohort is consumed.
            PooledImmutableFactoryReplayAccess::bindOriginalPreparedWorkers(*boundary->pooledStorage(),session->source());
            ASSERT_TRUE(boundary->pooledStorage()->retainedPreparedSourceWorkers().has_value());
            ASSERT_EQ(boundary->pooledStorage()->originalOwners().size(),1u);
            const auto& prepared=*boundary->pooledStorage()->retainedPreparedSourceWorkers();
            EXPECT_EQ(prepared.source(),session->source());EXPECT_EQ(prepared.source()->memory(),memory);
            EXPECT_EQ(prepared.coordinator().worker(),permit.worker());EXPECT_EQ(prepared.coordinator().generation(),permit.generation());
            ASSERT_EQ(prepared.owners().size(),1u);EXPECT_EQ(prepared.owners()[0].worker(),permit.worker());
            EXPECT_EQ(prepared.owners()[0].generation(),permit.generation());
            EXPECT_EQ(prepared.rootIdentity(),boundary->pooledStorage()->captureIdentity());
            compactBoundary=boundary;
            compactLocal=q::PooledImmutableLocalHotPreparedStorage::create(localGeometry,boundary->pooledStorage(),memory,stream);
            std::unique_ptr<q::IPooledImmutableLocalHotKernels> localCode;
#ifdef HAVE_CUDA
            if(device.is_cuda())localCode=q::createCUDAPooledImmutableLocalHotKernels();
#endif
#ifdef HAVE_ROCM
            if(device.is_rocm())localCode=q::createROCmPooledImmutableLocalHotKernels();
#endif
            requireProvider(bool(localCode),"Original local setup kernel bundle unavailable");localCode->initialize(*compactLocal,stream);
        }
        owner->requireAuthority(memory);coordinator->requireAuthority(memory);boundary->pooledStorage()->requireFor(physical,policy,source,1,memory);
        {const auto actual=GPUDeviceContextPool::instance().retainExistingPreparedContext(device,memory);actual.requireWorker(permit.worker());EXPECT_EQ(actual.generation(),permit.generation());}
        owner->acquireSetup(stream);coordinator->acquireSetup(stream);boundary->pooledStorage()->acquireSetup(stream);
        const auto progress=observe<n::OwnerProgress>(*backend,device,stream,owner->roots().native,ownerPlan.parameters().native.progress);
        EXPECT_EQ(progress.setup_identity,ownerPlan.parameters().identity);EXPECT_EQ(progress.initialized,2u);
        const auto ownerFrame=observe<p::OwnerFrame>(*backend,device,stream,owner->roots().frame,ownerPlan.parameters().frame.control);
        EXPECT_EQ(ownerFrame.phase,p::OwnerPhase::Idle);EXPECT_EQ(ownerFrame.status,q::adaptive_lookup::Status::Success);EXPECT_EQ(ownerFrame.held_pins,0u);EXPECT_EQ(ownerFrame.reservations,0u);
        const auto coordinatorFrame=observe<p::LookupFrame>(*backend,device,stream,coordinator->roots().frame,coordinatorPlan.parameters().frame.control);
        EXPECT_EQ(coordinatorFrame.phase,p::LookupPhase::Idle);EXPECT_EQ(coordinatorFrame.status,q::adaptive_lookup::Status::Success);
        const auto io=observe<q::ImmutableLookupDeviceState>(*backend,device,stream,boundary->pooledStorage()->roots().native,sourcePlan.parameters().offsets.protocol);
        const q::ImmutableLookupDeviceState empty{};EXPECT_EQ(std::memcmp(&io,&empty,sizeof(io)),0);
        EXPECT_NE(boundary->captureIdentity(),0u);requireProvider(backend->synchronizeStream(stream,device.ordinal),"Original local setup did not retire");
    });
        if(compactDeclaration){
            // Actual cold transport setup runs off its original worker; no replay
            // cohort or fabricated source permit stands in for native authority.
            auto transport=q::PooledImmutableDirectedTransport::create(compactBoundary,{}, {}, {},compactGeometry);
            auto query=transport->localHotPoolQuery(compactLocal);
            const auto& actual=compactBoundary->pooledStorage()->coordinator()->plan();
            const auto declaration=query.declarationIdentity();
            EXPECT_NO_THROW(query.requireDeclaration(actual));
            EXPECT_THROW(query.require(actual),std::logic_error);
            EXPECT_THROW(query.captureIdentity(),std::logic_error);
            const auto foreignPolicy=q::PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},48);
            const auto foreign=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,foreignPolicy,
                actual.parameters().table,actual.parameters().source_storage_generation);
            EXPECT_THROW(query.requireDeclaration(foreign),std::logic_error);
            worker.submitAndWait([&]{
                query.seal();EXPECT_NO_THROW(query.require(actual));EXPECT_NE(query.captureIdentity(),0u);
                EXPECT_EQ(query.declarationIdentity(),declaration);
                query.seal();EXPECT_NO_THROW(query.require(actual));
                EXPECT_THROW(query.require(foreign),std::logic_error);
                requireProvider(backend->synchronizeStream(stream,device.ordinal),"Compact setup did not retire");
            });
            transport.reset();compactLocal.reset();compactBoundary.reset();
        }
    }
    // All root/service/canonical publication handles are gone before resetting
    // this exact original source population. Native opaque debts may remain.
    session->retire();std::size_t exact=0;for(const auto& row:memory->rankAttestation())exact+=row.preparation_materialized_bytes;EXPECT_EQ(exact,0u);
    std::cout<<"POOLED_ORIGINAL_SETUP "<<nlohmann::json{{"device",device.toString()},{"gpu_slots",slots},{"replays",Replays},{"original_source_consumed",true},
        {"same_pma",true},{"native_initialization_bytes_checked",true},{"graph_created",false},{"timing_events_created",false},{"cpu_service_worker_qualified",false},{"whole_process_byte_completeness",false},{"numerical_inference_executed",false}}.dump()<<'\n';
}
TEST(PooledOriginalSourceSetup,CUDACompactDeclarationBeforeSeal){originalSetup(DeviceId::cuda(0),0,true);}
TEST(PooledOriginalSourceSetup,ROCmCompactDeclarationBeforeSeal){originalSetup(DeviceId::rocm(0),0,true);}
TEST(PooledOriginalSourceSetup,CUDAC0){originalSetup(DeviceId::cuda(0),0);}
TEST(PooledOriginalSourceSetup,CUDAPositive){originalSetup(DeviceId::cuda(0),4);}
TEST(PooledOriginalSourceSetup,ROCmC0){originalSetup(DeviceId::rocm(0),0);}
TEST(PooledOriginalSourceSetup,ROCmPositive){originalSetup(DeviceId::rocm(0),4);}
}}
