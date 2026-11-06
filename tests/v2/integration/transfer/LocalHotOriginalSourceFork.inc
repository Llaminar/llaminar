/**
 * @file LocalHotOriginalSourceFork.inc
 * @brief Real compact/provider warmup and actual L1 model-read progress under held auxiliary service.
 *
 * The original singleton source, CPU I/O worker and PMA roots are real. The
 * shared production builder emits every Probe, compact/coalescer/provider,
 * native result, LocalRead and retirement stage. Only diagnostic mapped gates
 * and completion observations are authored here. This is an actual native
 * component, not a Rank/cohort/model-parent or throughput qualification.
 */
#include "execution/qwen4exp/PooledImmutableLocalHotSourceGraph.h"
#include "execution/local_execution/graph/GraphCaptureStageActivity.h"
#include "execution/compute_stages/stages/qwen4exp/PooledImmutableLocalHotPhaseStage.h"
#include "fixtures/execution/RetainedHostServiceFixture.h"
#include "fixtures/qwen4exp/PooledOriginalSourceSetupPrograms.h"
#include "fixtures/qwen4exp/PooledImmutableFactoryReplayAccess.h"
#include "kernels/qwen4exp/PooledImmutableSourceKernels.h"
#include "kernels/qwen4exp/ImmutableLookupKernels.h"
#include "loaders/CpuAdaptiveImmutableRowCache.h"
#include "loaders/HostImmutableRowProvider.h"
#include "memory/CPUTensorStorageDeclaration.h"
#include "utils/PooledNgramReplayTrace.h"
#include <filesystem>
#include <fcntl.h>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <unistd.h>

namespace llaminar2::test { namespace original_l1_fork {
namespace q=qwen4exp;namespace h=q::local_hot;namespace t=q::local_hot_transport;namespace n=q::pooled_native;
#include "../kernels/qwen4exp/PooledImmutableProviderFile.inc"

/**
 * @brief Replay the selected original source and captured local query on a real backend.
 * @tparam Maximum Original admitted bank/frame extent in IDs, unchanged across replays.
 * @tparam QueryTokens Exact immutable captured token-row ceiling, independent from bank capacity.
 * @param device Actual physical participant; absence or native failure is fatal.
 * @throws std::exception For invalid original source, admission, capture or retirement.
 * @note Diagnostic terminal copies observe outputs only after actual parent retirement.
 */
template<std::uint32_t Maximum=32,std::uint32_t QueryTokens=Maximum/16>
void actualSourceFork(DeviceId device)
{
    if(!hasCPUBackend())initCPUBackend(-1);
    const std::array devices{device};auto session=PreparedNativeLookupFixture::prepare(devices);
    const auto memory=session->memory();const auto gpu=session->resource(device),host=session->resource(DeviceId::cpu());
    const PhysicalMemoryAllocatorIdentity gpuId{0,device},hostId{0,DeviceId::cpu()};
    const std::array<std::uint64_t,4> authored{7,8,9,10};ProviderFile file(32,authored);const auto source=file.descriptor();
    static_assert(QueryTokens>0 && Maximum%16==0 && QueryTokens<=Maximum/16);
    constexpr std::uint32_t Slots=4,QueryOriginal=QueryTokens*16;
    const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::RAM};
    const auto layout=ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(source.format,source.rows,source.rowBytes()),
        ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,
        {{{gpuId,ImmutableRowStorage::VRAM},0,0,{}},{{hostId,ImmutableRowStorage::RAM},0,0,{}}},{gpuId},{},
        {{hostId,ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::RAM)}},{},{},Maximum);
    const auto physical=q::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,std::array{gpuId},gpuId);
    const auto policy=q::PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},Maximum);
    const auto cpu=CpuAdaptiveImmutableRowCacheGeometry::forSubordinateGPU(layout,hostId,gpuId);
    const auto channel=q::ImmutableLookupLayout::createSubordinate(Maximum,source.rowBytes());
    const auto localGeometry=q::PooledImmutableLocalHotGeometry::declare(source,gpuId,Maximum,Slots,{65536,1,2});
    const auto compact=q::PooledImmutableLocalHotTransportGeometry::singleton(gpuId,Maximum,
        ImmutableRowTableExtent::fromNative(source.format,source.rows,source.rowBytes()));
    const auto io=RetainedHostServiceProgram::original(RetainedHostServiceRole::ImmutableLookupIO);
    const std::array declarations{
        CPUTensorStorageDeclaration(1,Maximum*2,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
        CPUTensorStorageDeclaration(1,1,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
        CPUTensorStorageDeclaration(1,sampling_math::kDeviceGenerationControlCount,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
        CPUTensorStorageDeclaration(Maximum,static_cast<int>(file.columns),BufferTensorType::FP32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu())};
    PhysicalMemoryPlanBuilder bom;
    for(const auto& part:{cpu.bom(host),physical.cacheBOM(0,gpu),physical.mandatoryBOM(0,gpu),policy.policyBOM(gpu),policy.mandatoryBOM(gpu),
        q::PooledImmutableOwnerNativePlan::mandatoryBOM(physical,policy,0,gpu),q::PooledImmutableCoordinatorNativePlan::mandatoryBOM(physical,policy,gpu),
        q::PooledImmutableSourceGeometry::mandatoryBOM(physical,policy,source,gpu),localGeometry.mandatoryBOM(gpu,host),localGeometry.slotDependentBOM(gpu,host),
        compact.completeBOM(host,std::array{gpu}),io.bom(host),RetainedParticipantGraphExecutable::generalBOM(host,gpu,1)})
        for(const auto& row:part.resources())bom.add(row);
    bom.add(host,PhysicalMemoryOwner::ImmutableLookupStaging,q::ImmutableLookupService::hostBytes(channel));
    bom.add(gpu,PhysicalMemoryOwner::ActivationArena,Maximum*8+4+sampling_math::kDeviceGenerationControlCount*4+Maximum*file.columns*4);
    for(const auto& declaration:declarations)bom.add(host,PhysicalMemoryOwner::ActivationArena,declaration.physicalBytes());
    bom.add(host,PhysicalMemoryOwner::ImmutableLookupStaging,3*TransferEngine::mappedHostRegionAllocationBytes(sizeof(std::uint64_t)));
    for(const auto program:originalPooledSourceSetupOrderingPrograms(physical,policy,source)){
        const auto part=NativeOrderingEventGroup::bom(program,host,gpu);for(const auto& row:part.resources())bom.add(row);
    }
    session->admit(bom.build());
    const auto& original=session->permit(device);auto worker=original.worker();auto* backend=getBackendFor(device);
    const auto stream=ExplicitGPUStream(worker->defaultStream()).get();
    const RetainedHostServiceInstance ioKey{0,device,RetainedHostServiceRole::ImmutableLookupIO,0};
    RetainedHostServiceDeclarations::Builder declaration(0);declaration.add(ioKey,io);const auto declared=std::move(declaration).freeze();
    auto ioBinding=declared->bind(memory,std::span{&ioKey,std::size_t{1}});
    {
        std::shared_ptr<HostImmutableRowProvider> provider;std::shared_ptr<q::ImmutableLookupService> service;
        std::shared_ptr<q::ImmutableLookupExecutionBoundary> boundary;std::shared_ptr<q::PooledImmutableLocalHotPreparedStorage> local;
        std::shared_ptr<const q::PooledImmutableDirectedTransport> transport;
        std::shared_ptr<const q::IPooledImmutableNativeKernels> poolCode;
        std::shared_ptr<const q::IPooledImmutableLocalHotKernels> localCode;
        std::array<std::shared_ptr<TensorBase>,4> roots;BufferArena arena;
        std::optional<DeviceGenerationControlBinding> control;
        auto gate=q::PooledImmutableLocalHotReadyRoot::create(gpuId,memory);
        auto probeDone=q::PooledImmutableLocalHotReadyRoot::create(gpuId,memory);
        auto readDone=q::PooledImmutableLocalHotReadyRoot::create(gpuId,memory);
        worker->submitAndWait([&]{
            provider=HostImmutableRowProvider::fromSubordinateAdaptiveCache(CpuAdaptiveImmutableRowCache::create(source,cpu,memory));
            service=q::ImmutableLookupService::create(provider,channel,std::array{device},memory,DeviceId::cpu(),ioBinding->take(ioKey));
            const auto binding=service->bindSubordinate(device);
            const auto ownPlan=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,policy,0,binding.deviceView().base.table,binding.storageGeneration());
            const auto coordPlan=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,policy,binding.deviceView().base.table,binding.storageGeneration());
            auto owner=q::PooledImmutableOwnerPreparedStorage::create(ownPlan,memory,stream);
            auto coordinator=q::PooledImmutableCoordinatorPreparedStorage::create(coordPlan,memory,stream);
#ifdef HAVE_CUDA
            if(device.is_cuda()){poolCode=q::createCUDAPooledImmutableNativeKernels();localCode=q::createCUDAPooledImmutableLocalHotKernels();}
#endif
#ifdef HAVE_ROCM
            if(device.is_rocm()){poolCode=q::createROCmPooledImmutableNativeKernels();localCode=q::createROCmPooledImmutableLocalHotKernels();}
#endif
            requireProvider(poolCode && localCode,"Real local/source native bundles missing");
            poolCode->initialize(*owner,stream);poolCode->initialize(*coordinator,stream);
            const auto sourcePlan=q::PooledImmutableSourceGeometry::fromPlans(ownPlan,coordPlan,source,1);
            boundary=q::ImmutableLookupExecutionBoundary::createPooled(service,sourcePlan,owner,coordinator,memory,stream);
            PooledImmutableFactoryReplayAccess::bindOriginalPreparedWorkers(*boundary->pooledStorage(),session->source());
            const auto& witness=*boundary->pooledStorage()->retainedPreparedSourceWorkers();
            requireProvider(witness.source()==session->source() && witness.source()->memory()==memory &&
                witness.rootIdentity()==boundary->pooledStorage()->captureIdentity() && witness.coordinator().worker()==worker &&
                witness.coordinator().generation()==original.generation() && witness.owners().size()==1 &&
                witness.owners()[0].worker()==worker && witness.owners()[0].generation()==original.generation(),
                "Actual original source preparation identity changed");
            local=q::PooledImmutableLocalHotPreparedStorage::create(localGeometry,boundary->pooledStorage(),memory,stream);localCode->initialize(*local,stream);
            const std::array roles{BufferId::QFN_NGRAM_ROW_IDS,BufferId::PREFILL_CHUNK_GEOMETRY,BufferId::MTP_GENERATION_CONTROL,BufferId::QFN_PLE_EMBEDDING};
            for(std::size_t i=0;i<roots.size();++i){
                roots[i]=std::shared_ptr<TensorBase>(declarations[i].materialize(memory));std::memset(roots[i]->raw_mutable_data(),0,roots[i]->size_bytes());
                TransferEngine::prepareDeviceInput(roots[i].get(),device,stream);TransferEngine::requireDeviceInput(roots[i].get(),device,stream);
                requireProvider(arena.registerExternalBuffer(roles[i],roots[i]),"Real local/source arena binding failed");
            }
            control.emplace(DeviceGenerationControlBinding::admit(arena,device,0));
        });
        transport=q::PooledImmutableDirectedTransport::create(boundary,{},{},{},compact);
        ComputeGraph graph;DeviceGraphExecutor executor;executor.setArena(&arena);
        std::unique_ptr<IDeviceContext> context;
        worker->submitAndWait([&]{context=IDeviceContext::create(device,1);});
        const auto body=appendPooledImmutableLocalHotSourceBody(graph,arena,transport,std::array{local},poolCode,local,localCode,
            DeviceRowRange::deviceCounted(QueryTokens,static_cast<const int*>(roots[1]->gpu_data_ptr())),
            {roots[0],BufferId::QFN_NGRAM_ROW_IDS,0,QueryOriginal*8},q::PooledNativeTensorSpan{roots[1],BufferId::PREFILL_CHUNK_GEOMETRY,0,4},*control,
            {roots[3],BufferId::QFN_PLE_EMBEDDING,0,QueryOriginal*file.columns*4},"original_local","actual_local_read");
        const PooledImmutableLocalHotBeginStage* begin=nullptr;
        for(const auto& name:graph.getExecutionOrder()){
            const auto* actual=dynamic_cast<const PooledImmutableLocalHotBeginStage*>(graph.getNode(name)->stage.get());
            if(actual && actual->source()==boundary->pooledStorage()){
                ASSERT_EQ(begin,nullptr);begin=actual;
            }
        }
        ASSERT_NE(begin,nullptr);
        EXPECT_EQ(begin->query().queryBound().canonicalOriginal(),QueryOriginal);
        EXPECT_EQ(begin->query().queryBound().maximumOriginal(),QueryOriginal);
        EXPECT_TRUE(n::coordinatorUsesSmallOriginal(DeviceRowRange::fullyActive(QueryOriginal)));
        if constexpr(Maximum>QueryOriginal)EXPECT_FALSE(n::coordinatorUsesSmallOriginal(DeviceRowRange::fullyActive(Maximum)));
        EXPECT_FALSE(graph.retainedParticipantSourcePlan());EXPECT_FALSE(graph.retainedParticipantAuxiliaryRange());
        requirePooledSourceRetirementCompletion(graph,body.pool_retirement);
        requirePooledLocalHotAuxiliary(graph,body.local.auxiliary,transport,local);
        auto parent=RetainedParticipantGraphExecutableFixtureAccess::original(original,stream);
        RetainedParticipantGraphExecutableFixtureAccess::NativeForkDefinitions definitions;
        bool terminal=false;
        const auto join=[&]{worker->submitAndWait([&]{requireProvider(backend->synchronizeStream(stream,device.ordinal),"Real local/source diagnostic join failed");});};
        struct Close {
            std::function<void()> finish;bool& terminal;
            ~Close(){if(!terminal)try{finish();}catch(...){std::terminate();}}
        } close{[&]{hostWord(gate).store(1,std::memory_order_release);join();if(boundary->hasActiveInterval()){
            boundary->recordTerminalClose(stream);join();boundary->awaitTerminal();}parent.reset();definitions={};},terminal};
        worker->submitAndWait([&]{
            for(const auto& name:graph.getExecutionOrder())requireProvider(graph.getNode(name)->stage->prepareGraphLaunch(context.get(),stream),"Real combined cold stage binding failed");
            requireProvider(executor.prepareGraphStorageForCapture(graph,context.get(),stream,"actual combined original source"),"Real source graph storage binding failed");
            const std::vector<std::string> probeNames{body.local.probe};
            const std::vector<std::string> readNames{body.local.ready_wait,body.local.read};
            const auto& auxNames=body.local.auxiliary.nodes;
            const std::array imported{BufferId::MTP_GENERATION_CONTROL};
            const auto record=[&](IGPUGraphCapture& native,const std::vector<std::string>& names,bool auxiliary,const std::shared_ptr<q::PooledImmutableLocalHotReadyRoot>& marker){
                auto ledger=executor.planGraphCaptureDependencies(graph,names,device,stream,"actual local/source child",
                    names==probeNames?std::span<const BufferId>{}:std::span<const BufferId>{imported});
                requireProvider(bool(ledger),"Actual local/source dependency plan missing");
                std::vector<IComputeStage*> stages;for(const auto& name:names)stages.push_back(graph.getNode(name)->stage.get());
                ScopedGraphCaptureStageActivity activity(stages,context.get(),stream,"actual local/source child");
                requireProvider(activity.begin(),"Actual local/source activity begin failed");
                ScopedBackendGraphCapture capture(*worker,native,"actual local/source child",ledger.get());requireProvider(capture.begin(),"Actual local/source capture begin failed");
                if(auxiliary)TransferEngine::instance().enqueueMappedTimelineWait64(*gate->mappedRegion(),0,1,device,stream);
                for(const auto& name:names){auto* stage=graph.getNode(name)->stage.get();ScopedGraphCaptureStage scope(stage,stream);
                    requireProvider(stage->execute(context.get()),"Actual local/source stage recording failed");scope.complete();}
                if(marker)TransferEngine::instance().enqueueMappedTimelinePublish64(*marker->mappedRegion(),0,1,device,stream);
                capture.finish();requireProvider(activity.finish(GraphCaptureActivityTransition::Completed),"Actual local/source activity finish failed");
            };
            definitions=RetainedParticipantGraphExecutableFixtureAccess::composeNativeForkFixture(parent,true,false,
                [&](auto& probe,auto& read,auto& auxiliary){record(probe,probeNames,false,probeDone);record(read,readNames,false,readDone);record(auxiliary,auxNames,true,{});});
            auto& native=RetainedParticipantGraphExecutableFixtureAccess::capture(parent);
            requireProvider(native.prepareRuntimeContextStorage(memory)&&native.instantiate(),"Actual combined parent instantiation failed");
        });
        join();const auto localIdentity=local->captureIdentity();
        std::uint64_t ordinal=0,totalRead=0,totalAllHit=0,totalHits=0,totalPromotions=0;
        const auto waitWord=[&](const auto& word){const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(30);
            while(hostWord(word).load(std::memory_order_acquire)!=1){requireProvider(std::chrono::steady_clock::now()<end,"Actual local/source marker deadline");std::this_thread::yield();}};
        const auto observe=[&]<class T>(const std::shared_ptr<TensorBase>& root,std::size_t offset){T value{};worker->submitAndWait([&]{
            requireProvider(backend->deviceToHostOnStream(&value,static_cast<const std::byte*>(root->gpu_data_ptr())+offset,sizeof(T),device.ordinal,stream),"Actual source terminal metadata copy failed");});join();return value;};
        for(std::uint64_t request=1;request<=2;++request){
            const auto beforeService=service->stats();
            boundary->beginAdmission(request,stream);
            worker->submitAndWait([&]{transport->enqueueLocalHotAdmission(local,localCode,*control,stream);});join();
            std::uint64_t expectedRequests=0,expectedRows=0;
            /**
             * @brief Check exact recorded native work, including original-ID and unique-row units.
             * @param input Authored original ID order for this retained parent replay.
             * @param heldAllHit Whether LocalRead must finish before the original source gate opens.
             * @param expectedMisses Unique native rows fetched from the genuine source service.
             * @param expectedHits Original ID occurrences already pinned in the local cache.
             * @param expectedPromotions Unique native rows made Ready after actual demand consumption.
             */
            const auto replay=[&](std::span<const std::uint64_t> input,bool heldAllHit,
                std::uint64_t expectedMisses,std::uint64_t expectedHits,std::uint64_t expectedPromotions){
                requireProvider(input.size()%16==0 && input.size()<=QueryOriginal,"Actual full query extent invalid");file.requireAuthored(input);
                std::array<std::uint64_t,Maximum> ids{};std::copy(input.begin(),input.end(),ids.begin());const int tokens=static_cast<int>(input.size()/16);
                std::vector<float> poison(Maximum*file.columns,-12345.f);
                hostWord(probeDone).store(0,std::memory_order_release);hostWord(readDone).store(0,std::memory_order_release);hostWord(gate).store(heldAllHit || expectedMisses?0:1,std::memory_order_release);
                worker->submitAndWait([&]{
                    requireProvider(backend->hostToDeviceOnStream(roots[0]->gpu_data_ptr(),ids.data(),sizeof(ids),device.ordinal,stream),"Actual source hashes upload failed");
                    requireProvider(backend->hostToDeviceOnStream(roots[1]->gpu_data_ptr(),&tokens,4,device.ordinal,stream),"Actual source count upload failed");
                    requireProvider(backend->hostToDeviceOnStream(roots[3]->gpu_data_ptr(),poison.data(),poison.size()*4,device.ordinal,stream),"Actual source output poison failed");
                    for(const auto& root:{roots[0],roots[1],roots[3]})TransferEngine::publishDeviceWrite(root.get(),device,stream);
                    requireProvider(RetainedParticipantGraphExecutableFixtureAccess::submit(parent,++ordinal),"Actual local/source parent submit failed");
                });
                waitWord(probeDone);
                if(heldAllHit){waitWord(readDone);EXPECT_FALSE(RetainedParticipantGraphExecutableFixtureAccess::observeTerminal(parent));}
                else if(expectedMisses)EXPECT_EQ(hostWord(readDone).load(std::memory_order_acquire),0u);
                hostWord(gate).store(1,std::memory_order_release);
                const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(30);
                while(!RetainedParticipantGraphExecutableFixtureAccess::observeTerminal(parent)){
                    requireProvider(std::chrono::steady_clock::now()<end,"Actual combined parent retirement deadline");std::this_thread::yield();}
                EXPECT_EQ(hostWord(readDone).load(std::memory_order_acquire),1u);
                std::vector<std::byte> nativeRows;for(const auto id:input){const auto row=file.original(id);nativeRows.insert(nativeRows.end(),row.begin(),row.end());}
                std::vector<float> expected(input.size()*file.columns),actual(poison.size());
                q::decodeCPUImmutableLookupRows(file.type,static_cast<int>(file.columns),file.row_bytes,nativeRows.data(),input.size(),expected.data());
                worker->submitAndWait([&]{requireProvider(backend->deviceToHostOnStream(actual.data(),roots[3]->gpu_data_ptr(),actual.size()*4,device.ordinal,stream),"Actual LocalRead terminal copy failed");});join();
                if(!expected.empty())EXPECT_EQ(std::memcmp(actual.data(),expected.data(),expected.size()*4),0);
                for(std::size_t i=expected.size();i<actual.size();++i)EXPECT_EQ(actual[i],-12345.f);
                const auto& offsets=local->geometry().frameOffsets();
                const auto frame=observe.template operator()<h::FrameIdentity>(local->roots().frame,offsets.identity);
                const auto result=observe.template operator()<t::ResultHeader>(local->roots().frame,offsets.result_header);
                const auto ioState=observe.template operator()<q::ImmutableLookupDeviceState>(boundary->pooledStorage()->roots().native,
                    boundary->pooledStorage()->geometry().parameters().offsets.protocol);
                EXPECT_EQ(frame.disposition,h::Disposition::Idle);EXPECT_EQ(frame.terminal_error,q::adaptive_lookup::Status::Success);
                EXPECT_EQ(result.status,q::adaptive_lookup::Status::Success);EXPECT_TRUE(t::same(result.query,frame.query));
                EXPECT_EQ(result.miss_rows,expectedMisses);EXPECT_EQ(result.native_bytes,expectedMisses*file.row_bytes);
                const auto counters=observe.template operator()<h::Counters>(local->roots().frame,offsets.counters);
                totalRead+=input.size();if(!expectedMisses)totalAllHit+=input.size();
                totalHits+=expectedHits;totalPromotions+=expectedPromotions;
                EXPECT_EQ(counters.read_original_ids,totalRead);
                EXPECT_EQ(counters.local_hit_original_ids,totalHits);
                EXPECT_EQ(counters.all_hit_wait_avoided_original_ids,totalAllHit);
                EXPECT_EQ(counters.consumed_original_ids,totalRead);
                EXPECT_EQ(counters.promotions,totalPromotions);
                // Only two distinct consumed rows enter four admitted slots.
                // This trace must neither evict nor claim speculative work.
                EXPECT_EQ(counters.evictions,0u);
                EXPECT_EQ(counters.future_on_time_native_rows,0u);
                EXPECT_EQ(counters.future_late_native_rows,0u);
                EXPECT_EQ(counters.future_useful_native_rows,0u);
                EXPECT_EQ(counters.future_wasted_native_rows,0u);
                EXPECT_EQ(ioState.idCount,expectedMisses);EXPECT_EQ(ioState.nativeBytes,expectedMisses*file.row_bytes);
                EXPECT_EQ(ioState.issuedSequence,ioState.consumedSequence);EXPECT_EQ(local->captureIdentity(),localIdentity);
                expectedRequests+=expectedMisses!=0;expectedRows+=expectedMisses;
            };
            std::vector<std::uint64_t> warm(16,7);
            replay(warm,false,request==1?1:0,request==1?0:16,request==1?1:0);
            for(int repeat=0;repeat<20;++repeat)replay(warm,true,0,16,0);
            std::vector<std::uint64_t> mixed(16,7);mixed.back()=8;
            replay(mixed,false,request==1?1:0,request==1?15:16,request==1?1:0);
            replay(mixed,true,0,16,0);
            {
                // Both source-storage shapes exercise the same fused small query: replay
                // its full live tile, zero progress, then small work without
                // changing graph, root pointers, source or allocation geometry.
                std::vector<std::uint64_t> full(QueryOriginal,7);
                replay(full,true,0,QueryOriginal,0);
                replay(std::span<const std::uint64_t>{},true,0,0,0);
                replay(warm,true,0,16,0);
            }
            worker->submitAndWait([&]{boundary->recordTerminalClose(stream);});join();boundary->awaitTerminal();
            const auto stats=service->stats();EXPECT_GE(stats.requests,beforeService.requests);EXPECT_GE(stats.ids,beforeService.ids);EXPECT_GE(stats.nativeBytes,beforeService.nativeBytes);
            EXPECT_EQ(stats.requests-beforeService.requests,expectedRequests);EXPECT_EQ(stats.ids-beforeService.ids,expectedRows);
            EXPECT_EQ(stats.nativeBytes-beforeService.nativeBytes,expectedRows*file.row_bytes);
        }
        parent.reset();definitions={};graph.clear();context.reset();transport.reset();local.reset();boundary.reset();service.reset();provider.reset();roots={};
        join();terminal=true;
        std::cout<<"LOCAL_HOT_ORIGINAL_SOURCE_FORK device="<<device.toString()<<" real_compact_provider=1 held_all_hit=40 requests=2 native_children_executable=0 rank_model_qualified=0\n";
    }
    session->requirePayloadRetired();worker.reset();session->retire();session->requireExactRetired();
}
}
#ifdef HAVE_CUDA
TEST(LocalHotNativeParentFork,CUDA_OriginalSourceAllHitAndMixed){original_l1_fork::actualSourceFork(DeviceId::cuda(0));}
TEST(LocalHotNativeParentFork,CUDA_OriginalSourceTightQueryBound){original_l1_fork::actualSourceFork<1024,2>(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
TEST(LocalHotNativeParentFork,ROCm_OriginalSourceAllHitAndMixed){original_l1_fork::actualSourceFork(DeviceId::rocm(0));}
TEST(LocalHotNativeParentFork,ROCm_OriginalSourceTightQueryBound){original_l1_fork::actualSourceFork<1024,2>(DeviceId::rocm(0));}
#endif
} // namespace llaminar2::test
