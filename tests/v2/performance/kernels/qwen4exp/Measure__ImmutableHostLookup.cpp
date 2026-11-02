/**
 * @file Measure__ImmutableHostLookup.cpp
 * @brief Matched ordinary GPU-continuation RAM/mmap/adaptive lookup measurement.
 *
 * This diagnostic binds the same real provider, service, boundary and captured
 * stage as proveLookup. Lookup and real terminal close have distinct retained
 * graphs. The actual worker publishes its own tier observations after close.
 * Original prepared source admission owns both retained graph slots and the IO
 * service on one PMA. Exact native owners die before original physical reset.
 * Original input publication, file-local cache advice, CPU oracle and terminal
 * readback are outside timing. No diagnostic observation controls native work.
 */
#include "utils/PooledNgramReplayTrace.h"
#include "fixtures/backends/PreparedNativeLookupFixture.h"
#include "fixtures/backends/RetainedParticipantGraphExecutableFixtureAccess.h"
#include "backends/NativeOrderingEventGroup.h"
#include "execution/local_execution/graph/RetainedHostServiceDeclarations.h"
#include "utils/PooledNgramReplaySchedule.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include "backends/BackendManager.h"
#include "backends/ComputeBackend.h"
#include "backends/HardwareInventory.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "backends/GPUGraphMemoryContract.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/compute_stages/stages/qwen4exp/ImmutableLookupStage.h"
#include "execution/qwen4exp/ImmutableLookupExecutionBoundary.h"
#include "memory/CPUTensorStorageDeclaration.h"
#include "kernels/common/SamplingMath.h"
#include "kernels/qwen4exp/ImmutableLookupDecode.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "transfer/TransferEngine.h"
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <span>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/statfs.h>
#include <sys/resource.h>
#include <unistd.h>

using namespace llaminar2;
using namespace llaminar2::qwen4exp;
namespace {
constexpr size_t Width=160,RowBytes=90;
namespace trace=llaminar2::test::ngram_replay;
/** @brief Fail the explicitly selected measurement before substituting a mode. */
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
/** @brief Immutable one-geometry selection; no accelerator API is called by parsing. */
struct Options {
    DeviceId device=DeviceId::invalid();std::string storage,scenario="distinct_a";
    size_t tableRows=1048576,capacity=512,tokens=1,ramSlots=64,samples=7,warmup=2,windowBytes=65536,windowCount=2;
    bool cold=false;
};
/** @brief Parse a complete bounded decimal value, rejecting overflow and trailing bytes. */
size_t number(std::string_view text){size_t n=0;auto p=std::from_chars(text.data(),text.data()+text.size(),n);require(p.ec==std::errc{}&&p.ptr==text.data()+text.size(),"Invalid numeric option");return n;}
/** @brief Read explicit endpoint/storage/trace values before backend initialization. */
Options parse(int argc,char** argv){
    Options o;std::set<std::string> seen;
    for(int i=1;i<argc;i+=2){require(i+1<argc,"Missing option value");std::string key=argv[i],value=argv[i+1];require(seen.insert(key).second,"Duplicate option");
        if(key=="--backend"){require(value=="cuda"||value=="rocm"||value=="cpu","Explicit cpu/cuda/rocm required");o.device=value=="cuda"?DeviceId::cuda(0):value=="rocm"?DeviceId::rocm(0):DeviceId::cpu();}
        else if(key=="--storage")o.storage=value;else if(key=="--scenario")o.scenario=value;
        else if(key=="--cold-attempt"){require(value=="0"||value=="1","Cold attempt must be0/1");o.cold=value=="1";}
        else if(key=="--table-rows")o.tableRows=number(value);else if(key=="--capacity-tokens")o.capacity=number(value);
        else if(key=="--tokens")o.tokens=number(value);else if(key=="--ram-slots")o.ramSlots=number(value);
        else if(key=="--samples")o.samples=number(value);else if(key=="--warmup")o.warmup=number(value);
        else if(key=="--window-bytes")o.windowBytes=number(value);else if(key=="--window-count")o.windowCount=number(value);
        else throw std::invalid_argument("Unknown host lookup measurement option");}
    require(o.device.is_valid()&&(o.storage=="ram"||o.storage=="mmap"||o.storage=="adaptive"),"Explicit backend and ram/mmap/adaptive storage required");
    require(o.capacity&&o.capacity<=512&&o.tokens<=o.capacity&&o.tableRows>=16*(o.capacity+1)*4&&o.tableRows%16==0&&o.tableRows<=static_cast<size_t>(std::numeric_limits<off_t>::max())/RowBytes,"Invalid original token/head geometry");
    require(o.samples&&o.samples<=100000&&(o.warmup==0||o.warmup==trace::PreludeFrames)&&o.ramSlots<=o.tableRows,"Invalid sample/cache geometry or unmatched prelude count");
    require(!o.cold||o.storage!="ram","WholeRAM cannot claim a cold source lookup");return o;
}

/** @brief Private durable source; mutation ends before its original immutable owner exists. */
class NativeFile {
public:
    /** @brief Exact live-page coverage plus whole-owned-file cache residency, never a physical memory claim. */
    struct Residency {size_t first,second,file_resident,file_pages;};
    /** @brief Author the shared original IQ4_NL fixture pattern once, never requantize. */
    explicit NativeFile(const Options& o):rows(o.tableRows){
        // The canonical process guardian owns TMPDIR. Keep the source under
        // that existing owner even when native runtime exit bypasses destructors.
        // Durable filesystem selection belongs to launcher --session-base.
        std::string pattern=(std::filesystem::temp_directory_path()/"qfn-host-perf-XXXXXX").string();std::vector<char> name(pattern.begin(),pattern.end());name.push_back(0);
        fd=::mkstemp(name.data());require(fd>=0,"Fixture file creation failed");path=name.data();
        try{std::vector<std::byte> chunk(4096*RowBytes);for(size_t first=0;first<rows;first+=4096){const auto count=std::min(size_t{4096},rows-first);
                for(size_t row=0;row<count;++row)trace::fillRow(first+row,{chunk.data()+row*RowBytes,RowBytes});
                require(::write(fd,chunk.data(),count*RowBytes)==static_cast<ssize_t>(count*RowBytes),"Fixture native source write failed");}
            require(::fsync(fd)==0,"Fixture source fsync failed");struct statfs info{};require(::fstatfs(fd,&info)==0,"Fixture filesystem observation failed");filesystemType=info.f_type;
        }catch(...){::close(fd);::unlink(path.c_str());throw;}}
    /** @brief All actual table/provider/worker aliases die before this source unlinks. */
    ~NativeFile(){::close(fd);::unlink(path.c_str());}
    /** @return Exact immutable native source descriptor. */
    ImmutableRowTableSource source()const{return {.path=path,.tensor_name="per_layer_token_embd.weight",.rows=rows,.row_elements=Width,.format=GGUFTensorType::IQ4_NL};}
    /** @brief Request eviction of this fixture file only; no host-wide cache control exists. */
    int adviseCold()const{return ::posix_fadvise(fd,0,0,POSIX_FADV_DONTNEED);}
    /** @return Number of currently resident required pages and number of distinct required pages. */
    Residency liveResidency(std::span<const std::uint64_t> ids)const {
        const auto page=static_cast<size_t>(::sysconf(_SC_PAGESIZE)),bytes=rows*RowBytes,pages=(bytes+page-1)/page;
        void* map=::mmap(nullptr,bytes,PROT_NONE,MAP_SHARED,fd,0);require(map!=MAP_FAILED,"Fixture mincore mapping failed");
        std::vector<unsigned char> state(pages);const auto status=::mincore(map,bytes,state.data());const auto unmap=::munmap(map,bytes);require(status==0&&unmap==0,"Fixture mincore observation failed");
        std::set<size_t> required;for(auto row:ids){require(row<rows,"Fixture trace exceeds source");required.insert(row*RowBytes/page);required.insert((row*RowBytes+RowBytes-1)/page);}
        size_t resident=0,total=0;for(auto p:required)resident+=state[p]&1u;for(auto p:state)total+=p&1u;return {resident,required.size(),total,pages};
    }
    int fd=-1;std::string path;size_t rows;long filesystemType=0;
};
/** @return Process block-read bytes; unknown remains absent, never a fabricated zero. */
std::optional<std::uint64_t> readBytes(){std::ifstream input("/proc/self/io");std::string key;std::uint64_t value;while(input>>key>>value)if(key=="read_bytes:")return value;return {};}

/** @brief Persistent actor retains actual graph/stage/provider/worker roots through final native join. */
class Actor {
public:
    /** @brief Admit exact typed storage then bind the existing production stage and worker. */
    Actor(const Options& options, NativeFile& sourceFile, test::PreparedNativeLookupFixture& session)
        : o(options), file(sourceFile), original(session.permit(options.device)),
          worker(*original.worker()), backend(getBackendFor(o.device)),
          stream(ExplicitGPUStream(worker.defaultStream()).get()) {
        require(backend,"Requested native backend unavailable");
        require(!worker.ownsCurrentThread(), "Actor admission and retained graph ownership require the original control thread");
        // Failed construction must also retire original graph owners off-worker
        // before native payload destruction and the session's physical reset.
        struct ConstructionRetirement {
            Actor& actor;
            bool completed = false;
            ~ConstructionRetirement() noexcept {
                if (!completed) actor.retireNativeResources();
            }
        } construction{*this};
        const auto maxIds=o.capacity*16;
        const std::array declarations{
            CPUTensorStorageDeclaration(1,16,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
            CPUTensorStorageDeclaration(1,1,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
            CPUTensorStorageDeclaration(1,maxIds*2,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
            CPUTensorStorageDeclaration(1,maxIds*RowBytes/4,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
            CPUTensorStorageDeclaration(o.capacity,16*Width,BufferTensorType::FP32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
            CPUTensorStorageDeclaration(1,sampling_math::kDeviceGenerationControlCount,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu())};
        const auto gr = session.resource(o.device);
        const auto hr = session.resource(DeviceId::cpu());
        const RetainedHostServiceInstance ioInstance{0, o.device, RetainedHostServiceRole::ImmutableLookupIO, 0};
        RetainedHostServiceDeclarations::Builder services(0);
        services.addOriginal(ioInstance);
        const auto serviceDeclarations = std::move(services).freeze();
        PhysicalMemoryPlanBuilder bom;const auto descriptor=file.source();const auto policy=ImmutableRowTableStoragePolicy::forStorage(o.storage=="ram"?ImmutableRowStorage::RAM:ImmutableRowStorage::MemoryMapped,o.windowBytes,o.windowCount);
        std::optional<CpuAdaptiveImmutableRowCacheGeometry> adaptive;
        if(o.storage=="adaptive") {const PhysicalMemoryAllocatorIdentity host{0,DeviceId::cpu()};const std::array tiers{ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped};
            const auto layout=ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(descriptor.format,descriptor.rows,descriptor.rowBytes()),ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,
                {{{host,ImmutableRowStorage::RAM},0,o.ramSlots,{}}},{{0,o.device}},{},{{host,policy}},{},std::nullopt,maxIds);
            adaptive=CpuAdaptiveImmutableRowCacheGeometry::fromLayout(layout,host,maxIds);const auto cacheBom=adaptive->bom(hr);for(const auto& resource:cacheBom.resources())bom.add(resource);
        }else bom.add(hr,PhysicalMemoryOwner::ImmutableTableStorage,ImmutableRowTable::physicalBytes(descriptor,policy));
        const auto channel=ImmutableLookupLayout::create(maxIds,RowBytes);bom.add(hr,PhysicalMemoryOwner::ImmutableLookupStaging,ImmutableLookupService::hostBytes(channel));
        const size_t activation=64+4+maxIds*8+maxIds*RowBytes+maxIds*Width*4+sampling_math::kDeviceGenerationControlCount*4;
        bom.add(gr,PhysicalMemoryOwner::ActivationArena,activation);for(const auto& d:declarations)bom.add(hr,PhysicalMemoryOwner::ActivationArena,d.physicalBytes());
        for (const auto& contribution : {
                serviceDeclarations->bom(ioInstance, hr),
                NativeOrderingEventGroup::bom(NativeOrderingEventProgram::lookupSubmissionRetirement(), hr, gr),
                RetainedParticipantGraphExecutable::generalBOM(hr, gr, 2)})
            for (const auto& resource : contribution.resources()) bom.add(resource);
        session.admit(bom.build());
        memory = session.memory();
        auto serviceBinding = serviceDeclarations->bind(memory, std::array{ioInstance});
        auto ioToken = serviceBinding->take(ioInstance);
        deviceClaim = memory->claimNewAllocation(o.device, PhysicalMemoryOwner::ActivationArena, activation);
        worker.submitAndWait([&] {
            for(size_t i=0;i<roots.size();++i){roots[i]=std::shared_ptr<TensorBase>(declarations[i].materialize(memory));TransferEngine::prepareDeviceInput(roots[i].get(),o.device,stream);TransferEngine::requireDeviceInput(roots[i].get(),o.device,stream);}
        });
        provider=adaptive?HostImmutableRowProvider::fromAdaptiveCache(CpuAdaptiveImmutableRowCache::create(descriptor,*adaptive,memory)):
            HostImmutableRowProvider::fromWholeTable(std::make_unique<ImmutableRowTable>(descriptor,policy,DeviceId::cpu(),memory,ImmutableRowTableHostPlacement::taskPolicy()));
        worker.submitAndWait([&] {
        boundary=ImmutableLookupExecutionBoundary::create(provider,o.device,roots[0],o.capacity,memory,std::move(ioToken));
        auto* controller=static_cast<int*>(roots[5]->gpu_data_ptr());boundary->bindGenerationControl({controller+sampling_math::kDeviceGenerationControlOk,controller+sampling_math::kDeviceGenerationControlErrorCode});
        stage=std::make_unique<ImmutableLookupStage>(ImmutableLookupStage::Params{.device_id=o.device,.rows={static_cast<int>(o.capacity),Qwen4ExpIntBinding(*roots[1],BufferId::PREFILL_CHUNK_GEOMETRY)},
            .ids=Qwen4ExpIntBinding(*roots[2],BufferId::QFN_NGRAM_ROW_IDS),.output=Qwen4ExpFloatBinding(*roots[4],BufferId::QFN_PLE_EMBEDDING),.format=TensorType::IQ4_NL,.rowElements=Width,.rowBytes=RowBytes,
            .storage=ImmutableLookupStage::GPUStorage{boundary,Qwen4ExpIntBinding(*roots[0],BufferId::QFN_NGRAM_LOOKUP_STATE),Qwen4ExpIntBinding(*roots[3],BufferId::QFN_NGRAM_NATIVE_ROWS)}});
        context=IDeviceContext::create(o.device,1);require(stage->prepareGraphLaunch(context.get(),stream),"Ordinary stage cold storage preparation failed");
        });
        graph = test::RetainedParticipantGraphExecutableFixtureAccess::original(original, stream);
        terminal = test::RetainedParticipantGraphExecutableFixtureAccess::original(original, stream);
        worker.submitAndWait([&] {
        auto& lookupCapture = test::RetainedParticipantGraphExecutableFixtureAccess::capture(graph);
        auto& terminalCapture = test::RetainedParticipantGraphExecutableFixtureAccess::capture(terminal);
        {ScopedBackendGraphCapture capture(worker,lookupCapture,"ordinary native provider measurement");require(capture.begin(),"Ordinary lookup capture begin failed");require(stage->execute(context.get()),"Ordinary lookup capture failed");capture.finish();}
        require(lookupCapture.prepareRuntimeContextStorage(memory) && lookupCapture.instantiate(),"Ordinary lookup graph instantiation failed");
        {ScopedBackendGraphCapture capture(worker,terminalCapture,"ordinary original terminal close");require(capture.begin(),"Ordinary close capture begin failed");boundary->recordTerminalClose(stream);capture.finish();}
        require(terminalCapture.prepareRuntimeContextStorage(memory) && terminalCapture.instantiate(),"Ordinary terminal graph instantiation failed");
        start=backend->createTimingEvent(o.device.ordinal);stop=backend->createTimingEvent(o.device.ordinal);require(start&&stop,"Ordinary timing events unavailable");
        });
        construction.completed=true;
    }
    /** @brief Join actual submissions before dropping capture pointers or the original worker/provider. */
    ~Actor() noexcept {
        if (generation && !closed) {
            try { close(); } catch (...) { std::terminate(); }
        }
        retireNativeResources();
    }
    /** @brief Execute one checked original trace with observations entirely outside its native interval. */
    void sample(size_t index,bool report);
    /** @brief Close the actual service once after its unchanged trajectory; only this join permits host tier statistics. */
    void close();
private:
    /** @brief Retire exact original handles off-worker, then payloads on their original worker. */
    void retireNativeResources() noexcept {
        try {
            worker.submitAndWait([&] {
                if (!backend->synchronizeStream(stream,o.device.ordinal)) std::terminate();
            });
            for (const auto& owner : {terminal, graph})
                if (owner && !test::RetainedParticipantGraphExecutableFixtureAccess::observeTerminal(owner)) std::terminate();
            terminal.reset();
            graph.reset();
            worker.submitAndWait([&] {
                if (start) backend->destroyEvent(start,o.device.ordinal);
                if (stop) backend->destroyEvent(stop,o.device.ordinal);
                start = nullptr;
                stop = nullptr;
                stage.reset();
                context.reset();
                boundary.reset();
                provider.reset();
                for (auto& root : roots) root.reset();
                deviceClaim = {};
            });
        } catch (...) { std::terminate(); }
    }
    Options o;NativeFile& file;PreparedGPUContextUsePermit original;IWorkerGPUContext& worker;IBackend* backend;void* stream;void* start=nullptr;void* stop=nullptr;std::uint64_t generation=0,expectedIds=0;
    std::shared_ptr<PhysicalMemoryAuthority> memory;PhysicalMemoryAllocationLease deviceClaim;
    std::array<std::shared_ptr<TensorBase>,6> roots;std::shared_ptr<HostImmutableRowProvider> provider;std::shared_ptr<ImmutableLookupExecutionBoundary> boundary;
    std::shared_ptr<IDeviceContext> context;std::unique_ptr<ImmutableLookupStage> stage;std::shared_ptr<RetainedParticipantGraphExecutable> graph,terminal;
    std::uint64_t lookupOrdinal = 0;
    bool closed = false;
};

/** @return Fixed original ID program shared with pooled timing; no native state selects IDs. */
trace::Scenario scenario(std::string_view name) {
    if(name=="distinct_a"||name=="warm")return trace::Scenario::WarmHits;
    if(name=="alternating")return trace::Scenario::AlternatingSets;
    if(name=="mixed")return trace::Scenario::Mixed;
    if(name=="churn")return trace::Scenario::Churn;
    if(name=="duplicate_heavy")return trace::Scenario::DuplicateControl;
    if(name=="empty")return trace::Scenario::Empty;
    throw std::invalid_argument("Unknown original ngram trace scenario");
}
/** @brief Share the exact prelude/sample banks with both actual pooled actors. */
trace::Pattern pattern(std::string_view name,size_t index,size_t preludes=0) {
    return trace::scheduledInvocation(scenario(name),static_cast<std::uint32_t>(preludes),index).pattern;
}

void Actor::sample(size_t index,bool report) {
    const auto original=trace::ids({o.tableRows,static_cast<std::uint32_t>(o.capacity),static_cast<std::uint32_t>(o.tokens)},pattern(o.scenario,index,o.warmup));
    const auto maxIds=o.capacity*16;const auto liveTokens=original.size()/16;
    std::memset(roots[2]->raw_mutable_data(),0xff,roots[2]->size_bytes());if(!original.empty())std::memcpy(roots[2]->raw_mutable_data(),original.data(),original.size()*sizeof(std::uint64_t));
    *static_cast<int*>(roots[1]->raw_mutable_data())=static_cast<int>(liveTokens);
    std::fill_n(static_cast<float*>(roots[4]->raw_mutable_data()),maxIds*Width,-12345.0f);
    std::memset(roots[5]->raw_mutable_data(),0,roots[5]->size_bytes());static_cast<int*>(roots[5]->raw_mutable_data())[sampling_math::kDeviceGenerationControlOk]=1;
    for(const auto i:{1u,2u,4u,5u}){TransferEngine::publishHostWrite(roots[i].get());TransferEngine::prepareDeviceInput(roots[i].get(),o.device,stream);TransferEngine::requireDeviceInput(roots[i].get(),o.device,stream);}
    require(backend->synchronizeStream(stream,o.device.ordinal),"Diagnostic inputs did not retire");
    const auto beforeAdvice=file.liveResidency(original);const int advice=o.cold?file.adviseCold():0;
    const auto before=file.liveResidency(original);const auto ioBefore=readBytes();rusage usageBefore{};const bool usageBeforeKnown=::getrusage(RUSAGE_SELF,&usageBefore)==0;
    if(!generation){boundary->beginAdmission(++generation,stream);require(backend->synchronizeStream(stream,o.device.ordinal),"Original admission publication did not retire");}
    boundary->requireArmedForLaunch();
    const auto wallStart=std::chrono::steady_clock::now();
    require(backend->recordEvent(start,o.device.ordinal,stream),"Ordinary timing start failed");
    require(test::RetainedParticipantGraphExecutableFixtureAccess::submit(graph, ++lookupOrdinal),"Ordinary captured lookup launch failed");
    require(backend->recordEvent(stop,o.device.ordinal,stream)&&backend->waitForEvent(stop,o.device.ordinal),"Ordinary captured lookup did not complete");
    const auto wallStop=std::chrono::steady_clock::now();
    float elapsed=0;require(backend->eventElapsedTimeMs(start,stop,o.device.ordinal,&elapsed),"Ordinary native time unavailable");
    const auto ioAfter=readBytes();rusage usageAfter{};const bool usageAfterKnown=::getrusage(RUSAGE_SELF,&usageAfter)==0;const auto after=file.liveResidency(original);
    std::vector<std::byte> native(original.size()*RowBytes);for(size_t i=0;i<original.size();++i)trace::fillRow(original[i],{native.data()+i*RowBytes,RowBytes});
    std::vector<float> expected(original.size()*Width),actual(maxIds*Width);
    if(!original.empty())decodeCPUImmutableLookupRows(TensorType::IQ4_NL,Width,RowBytes,native.data(),original.size(),expected.data());
    require(backend->deviceToHostOnStream(actual.data(),roots[4]->gpu_data_ptr(),actual.size()*sizeof(float),o.device.ordinal,stream)&&backend->synchronizeStream(stream,o.device.ordinal),"Ordinary terminal output observation failed");
    require(expected.empty()||std::memcmp(actual.data(),expected.data(),expected.size()*sizeof(float))==0,"Ordinary lookup byte oracle mismatch");
    require(std::all_of(actual.begin()+expected.size(),actual.end(),[](float x){return x==-12345.0f;}),"Ordinary inactive output tail changed");
    std::array<int,sampling_math::kDeviceGenerationControlCount> controller{};require(backend->deviceToHostOnStream(controller.data(),roots[5]->gpu_data_ptr(),controller.size()*sizeof(int),o.device.ordinal,stream)&&backend->synchronizeStream(stream,o.device.ordinal),"Ordinary terminal health observation failed");
    require(controller[sampling_math::kDeviceGenerationControlOk]==1&&controller[sampling_math::kDeviceGenerationControlErrorCode]==0,"Ordinary native lookup rejected its original input");expectedIds+=original.size();
    const bool ioKnown=ioBefore&&ioAfter&&*ioAfter>=*ioBefore;const auto blockBytes=ioKnown?*ioAfter-*ioBefore:0;
    const bool tmpfs=file.filesystemType==0x01021994;
    const char* regime=o.storage=="ram"?"whole_ram":o.cold?(advice==0&&before.first==0&&before.second&&ioKnown&&blockBytes&&!tmpfs?"os_cold_confirmed_block_io":"file_cold_attempt_unconfirmed"):before.first==before.second?"os_hot_live_pages":"mixed_os_residency";
    if(!report)return;
    std::cout<<"NGRAM_PERF_SAMPLE {\"schema\":1,\"continuation\":\""<<o.device.toString()<<"\",\"owners\":[\"cpu\"],\"kind\":\"ordinary_"<<o.storage
        <<"\",\"format\":\"IQ4_NL\",\"width\":160,\"row_bytes\":90,\"tokens\":"<<liveTokens<<",\"capacity_tokens\":"<<o.capacity<<",\"live_ids\":"<<original.size()
        <<",\"unique_ids\":"<<trace::uniqueCount(original)<<",\"scenario\":\""<<o.scenario<<"\",\"phase\":\""<<trace::scheduledInvocation(scenario(o.scenario),o.warmup,index).phase<<"\",\"schedule\":\""<<trace::Schedule<<"\",\"gpu_slots\":0,\"ram_slots\":"<<(o.storage=="adaptive"?std::to_string(o.ramSlots):"null")
        <<",\"sample\":"<<index-o.warmup<<",\"trajectory_index\":"<<index<<",\"locality\":\"scattered\",\"event_us\":"<<std::setprecision(10)<<elapsed*1000.0<<",\"wall_us\":"<<std::chrono::duration<double,std::micro>(wallStop-wallStart).count()
        <<",\"window_bytes\":"<<o.windowBytes<<",\"window_count\":"<<o.windowCount<<",\"live_working_page_bytes\":"<<before.second*static_cast<size_t>(::sysconf(_SC_PAGESIZE))
        <<",\"scope\":\"open_interval_complete_lookup_excluding_admission_terminal_and_observations\",\"trajectory_id\":1,\"table_rows\":"<<o.tableRows<<",\"source_pattern\":\""<<trace::NativePattern
        <<"\",\"cache_regime\":\""<<regime<<"\",\"actual_tier_stats\":null,\"promotion\":null,\"eviction\":null"
        <<",\"physical_io\":{\"scope\":\"process_not_per_file_attribution\",\"read_bytes\":"<<(ioKnown?std::to_string(blockBytes):"null")<<",\"major_faults\":"<<(usageBeforeKnown&&usageAfterKnown&&usageAfter.ru_majflt>=usageBefore.ru_majflt?std::to_string(usageAfter.ru_majflt-usageBefore.ru_majflt):"null")
        <<",\"read_amplification\":null,\"read_bytes_per_requested_native_byte\":"<<(ioKnown&&!original.empty()?std::to_string(static_cast<double>(blockBytes)/(original.size()*RowBytes)):"null")
        <<",\"file_advice_result\":"<<(o.cold?std::to_string(advice):"null")<<",\"live_pages\":"<<before.second<<",\"file_pages\":"<<before.file_pages<<",\"file_resident_before_advice\":"<<beforeAdvice.file_resident<<",\"file_resident_before\":"<<before.file_resident<<",\"file_resident_after\":"<<after.file_resident<<",\"resident_before_advice\":"<<beforeAdvice.first
        <<",\"resident_before\":"<<before.first<<",\"resident_after\":"<<after.first<<",\"filesystem_type\":"<<file.filesystemType<<",\"tmpfs\":"<<(tmpfs?"true":"false")<<",\"physical_device_identity\":null},\"byte_validation\":\"exact_live_and_inactive_tail\",\"profile\":\"component_only\"}\n";
}

void Actor::close() {
    // The original await contract consumes an already-published GPU close.
    // Retire this exact terminal graph before the host can validate/join its
    // service ticket; awaiting earlier truthfully rejects a missing close.
    require(generation>0,"Cannot close an unadmitted original service");
    std::chrono::steady_clock::time_point begin;
    worker.submitAndWait([&] {
        begin = std::chrono::steady_clock::now();
        require(test::RetainedParticipantGraphExecutableFixtureAccess::submit(terminal, 1),"Original captured close failed");
        require(backend->synchronizeStream(stream,o.device.ordinal),"Original terminal graph did not retire");
    });
    boundary->awaitTerminal();
    const auto end = std::chrono::steady_clock::now();
    require(test::RetainedParticipantGraphExecutableFixtureAccess::observeTerminal(graph) &&
        test::RetainedParticipantGraphExecutableFixtureAccess::observeTerminal(terminal),
        "Original whole-trajectory graph readers did not retire");
    closed = true;
    const auto observation=boundary->terminalObservation();const auto& c=observation.counters;require(observation.generation==generation&&!c.failures&&c.ids==expectedIds&&c.nativeBytes==expectedIds*RowBytes,"Original service terminal observation failed");
    std::cout<<"NGRAM_PERF_TRAJECTORY {\"schema\":1,\"trajectory_id\":1,\"continuation\":\""<<o.device.toString()<<"\",\"kind\":\"ordinary_"<<o.storage<<"\",\"scope\":\"warmup_and_samples_one_original_service_interval\",\"warmup\":"<<o.warmup<<",\"samples\":"<<o.samples
        <<",\"terminal_close_us\":"<<std::chrono::duration<double,std::micro>(end-begin).count()<<",\"actual_tier_stats\":{\"requests\":"<<c.requests<<",\"ids\":"<<c.ids<<",\"id_bytes\":"<<c.idBytes<<",\"native_bytes\":"<<c.nativeBytes
        <<",\"adaptive_batches\":"<<c.adaptiveBatches<<",\"ram_rows\":"<<(o.storage=="ram"?c.ids:c.ramRows)<<",\"source_rows\":"<<(o.storage=="mmap"?c.ids:c.sourceRows)<<",\"duplicate_rows\":"<<c.duplicateRows
        <<",\"window_hits\":"<<c.windowHits<<",\"window_misses\":"<<c.windowMisses<<",\"window_evictions\":"<<c.windowEvictions<<",\"cache_admissions\":"<<c.cacheAdmissions<<",\"cache_evictions\":"<<c.cacheEvictions<<",\"admission_rejections\":"<<c.cacheAdmissionRejections
        <<"},\"promotion\":"<<c.cacheAdmissions<<",\"eviction\":"<<c.cacheEvictions<<",\"worker_retirement\":\"proved_by_original_awaitTerminal\"}\n";
}

/**
 * @brief Measure the exact CPU provider portion without an accelerator or I/O service.
 * @param o Explicit CPU source geometry and immutable sample trajectory.
 * @details This lower-level attribution control reports native-row copy time,
 * not completed GPU lookup latency. It shares the actual source/provider,
 * admission geometry and byte oracle with the GPU continuation measurement.
 */
void measureHostProvider(const Options& o) {
    for(const char* key:{"LLAMINAR_FORCE_CPU_ONLY_STARTUP","LLAMINAR_SKIP_CUDA_STARTUP","LLAMINAR_SKIP_ROCM_STARTUP"})
        require(std::getenv(key)&&std::string_view(std::getenv(key))=="1","CPU source control requires explicitly disabled accelerator startup");
    NativeFile file(o);
    auto& manager=DeviceManager::instance();manager.initialize(-1,false);
    require(manager.hardware(),"CPU source inventory unavailable");
    const auto cpu=manager.hardware()->cpuDevice();const PhysicalMemoryResource hr{0,DeviceId::cpu(),cpu.total_memory_bytes,cpu.free_memory_bytes};
    const PhysicalMemoryAllocatorIdentity host{0,DeviceId::cpu()};
    const auto descriptor=file.source();const auto policy=ImmutableRowTableStoragePolicy::forStorage(o.storage=="ram"?ImmutableRowStorage::RAM:ImmutableRowStorage::MemoryMapped,o.windowBytes,o.windowCount);
    std::optional<CpuAdaptiveImmutableRowCacheGeometry> adaptive;PhysicalMemoryPlanBuilder bom;
    if(o.storage=="adaptive") {
        const std::array tiers{ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped};
        const auto layout=ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(descriptor.format,descriptor.rows,descriptor.rowBytes()),ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,
            {{{host,ImmutableRowStorage::RAM},0,o.ramSlots,{}}},{host},{},{{host,policy}},{},std::nullopt,o.capacity*16);
        adaptive=CpuAdaptiveImmutableRowCacheGeometry::fromLayout(layout,host,o.capacity*16);
        const auto cacheBom=adaptive->bom(hr);for(const auto& resource:cacheBom.resources())bom.add(resource);
    }else bom.add(hr,PhysicalMemoryOwner::ImmutableTableStorage,ImmutableRowTable::physicalBytes(descriptor,policy));
    const auto memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(bom.build()),0);
    const auto provider=adaptive?HostImmutableRowProvider::fromAdaptiveCache(CpuAdaptiveImmutableRowCache::create(descriptor,*adaptive,memory)):
        HostImmutableRowProvider::fromWholeTable(std::make_unique<ImmutableRowTable>(descriptor,policy,DeviceId::cpu(),memory,ImmutableRowTableHostPlacement::taskPolicy()));
    std::vector<std::int64_t> ids(o.capacity*16);std::vector<std::byte> actual(o.capacity*16*RowBytes),expected(actual.size());
    for(size_t index=0;index<o.warmup+o.samples;++index) {
        const auto original=trace::ids({o.tableRows,static_cast<std::uint32_t>(o.capacity),static_cast<std::uint32_t>(o.tokens)},pattern(o.scenario,index,o.warmup));
        std::copy(original.begin(),original.end(),ids.begin());
        for(size_t row=0;row<original.size();++row)trace::fillRow(original[row],{expected.data()+row*RowBytes,RowBytes});
        std::fill(actual.begin(),actual.end(),std::byte{0xa5});
        const int advice=o.cold?file.adviseCold():0;const auto before=file.liveResidency(original);
        const auto ioBefore=readBytes();const auto statsBefore=provider->sourceStats();rusage usageBefore{};
        require(::getrusage(RUSAGE_SELF,&usageBefore)==0,"CPU source usage observation failed");
        const auto start=std::chrono::steady_clock::now();
        (void)provider->gather({ids.data(),original.size()},{actual.data(),original.size()*RowBytes});
        const auto stop=std::chrono::steady_clock::now();
        rusage usageAfter{};require(::getrusage(RUSAGE_SELF,&usageAfter)==0,"CPU source usage observation failed");
        const auto statsAfter=provider->sourceStats();const auto ioAfter=readBytes();const auto after=file.liveResidency(original);
        require(std::equal(actual.begin(),actual.begin()+original.size()*RowBytes,expected.begin()),"CPU source native byte oracle mismatch");
        require(std::all_of(actual.begin()+original.size()*RowBytes,actual.end(),[](std::byte x){return x==std::byte{0xa5};}),"CPU source inactive tail changed");
        if(index<o.warmup)continue;
        const bool ioKnown=ioBefore&&ioAfter&&*ioAfter>=*ioBefore;
        std::cout<<"NGRAM_PERF_SAMPLE {\"schema\":1,\"kind\":\"cpu_provider_"<<o.storage<<"\",\"scope\":\"actual_host_provider_native_copy_without_gpu_or_service\",\"tokens\":"<<original.size()/16
            <<",\"capacity_tokens\":"<<o.capacity<<",\"sample\":"<<index-o.warmup<<",\"trajectory_index\":"<<index<<",\"wall_us\":"<<std::setprecision(10)<<std::chrono::duration<double,std::micro>(stop-start).count()
            <<",\"source_pattern\":\""<<trace::NativePattern<<"\",\"schedule\":\""<<trace::Schedule<<"\",\"table_rows\":"<<o.tableRows<<",\"locality\":\"scattered\",\"scenario\":\""<<o.scenario<<"\",\"live_ids\":"<<original.size()
            <<",\"native_bytes\":"<<original.size()*RowBytes<<",\"window_bytes\":"<<o.windowBytes<<",\"window_count\":"<<o.windowCount<<",\"window_hits\":"<<statsAfter.window_hits-statsBefore.window_hits
            <<",\"window_misses\":"<<statsAfter.window_misses-statsBefore.window_misses<<",\"window_evictions\":"<<statsAfter.window_evictions-statsBefore.window_evictions
            <<",\"minor_faults\":"<<usageAfter.ru_minflt-usageBefore.ru_minflt<<",\"major_faults\":"<<usageAfter.ru_majflt-usageBefore.ru_majflt
            <<",\"read_bytes\":"<<(ioKnown?std::to_string(*ioAfter-*ioBefore):"null")<<",\"file_advice_result\":"<<(o.cold?std::to_string(advice):"null")
            <<",\"live_pages\":"<<before.second<<",\"resident_before\":"<<before.first<<",\"resident_after\":"<<after.first<<",\"byte_validation\":\"exact_live_and_inactive_tail\"}\n";
    }
}

/** @brief Replay real fixture-file/trace/residency declarations without initializing any GPU backend. */
void selfTest() {
    for(const char* key:{"LLAMINAR_FORCE_CPU_ONLY_STARTUP","LLAMINAR_SKIP_CUDA_STARTUP","LLAMINAR_SKIP_ROCM_STARTUP"})require(std::getenv(key)&&std::string_view(std::getenv(key))=="1","CPU proof requires explicit disabled accelerator startup");
    Options o;o.tableRows=4096;o.capacity=8;o.tokens=1;NativeFile file(o);const auto ids=trace::ids({o.tableRows,8,1},trace::Pattern::DistinctA);
    require(ids.size()==16&&trace::uniqueCount(ids)==16,"Original sixteen-head trace lost distinct rows");
    require(std::filesystem::equivalent(std::filesystem::path(file.path).parent_path(),std::filesystem::temp_directory_path()),"fixture_escaped_guardian_TMPDIR");
    for(auto row:ids){std::array<std::byte,RowBytes> expected{},actual{};trace::fillRow(row,expected);
        require(::pread(file.fd,actual.data(),actual.size(),static_cast<off_t>(row*RowBytes))==static_cast<ssize_t>(actual.size()),"Original source read failed");require(actual==expected,"Shared native source pattern changed");}
    const auto before=file.liveResidency(ids);require(before.first<=before.second&&before.file_resident<=before.file_pages&&before.second>0,"Invalid exact owned-file residency projection");
    require(file.adviseCold()==0,"Own-file cold advice rejected");const auto after=file.liveResidency(ids);require(after.first<=after.second&&after.file_resident<=after.file_pages,"Own-file cold projection out of bounds");
    require(pattern("churn",0)==trace::Pattern::DistinctA&&pattern("churn",1)==trace::Pattern::DistinctB&&pattern("churn",2)==trace::Pattern::DistinctA,"Original A-B-A order changed");
    std::array<std::byte,RowBytes> first{},head{},high{};trace::fillRow(1,first);trace::fillRow(65537,head);trace::fillRow(0x100000001ULL,high);
    require(first!=head&&first!=high&&head!=high,"shared_native_row_identity_collision");
    bool rejected=false;try{(void)pattern("invented",0);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"Unknown trace accepted");
    std::cout<<"NGRAM_HOST_PERF_CPU_INTERFACE_PASS rows="<<o.tableRows<<" native_pattern="<<trace::NativePattern<<" live_ids=16 before_pages="<<before.second<<" file_pages="<<before.file_pages<<" cold_advice_only_not_disk_claim=1\n";
}
} // namespace

/** @brief Run one explicitly selected component; exceptions fail without another tier/path. */
int main(int argc,char** argv) {
    try{if(argc==2&&std::string_view(argv[1])=="--self-test"){selfTest();return 0;}
        const auto o=parse(argc,argv);(void)pattern(o.scenario,0);if(o.device.is_cpu()){measureHostProvider(o);return 0;}GPUDeviceProcessLifetime process;NativeFile file(o);
        auto& manager=DeviceManager::instance();manager.initialize(-1,false);require(manager.hardware(),"Original startup inventory unavailable");
        const auto kind=o.device.is_cuda()?ComputeBackendType::GPU_CUDA:ComputeBackendType::GPU_ROCM;
        const auto it=std::find_if(manager.devices().begin(),manager.devices().end(),[&](const auto& d){return d.type==kind&&d.device_id==o.device.ordinal;});require(it!=manager.devices().end(),"Explicit native endpoint unavailable");
        auto session = test::PreparedNativeLookupFixture::prepare(std::array{o.device});
        auto& worker = *session->permit(o.device).worker();
        {
            Actor actor(o, file, *session);
            for(size_t i=0;i<o.warmup;++i) worker.submitAndWait([&]{actor.sample(i,false);});
            for(size_t i=0;i<o.samples;++i) worker.submitAndWait([&]{actor.sample(o.warmup+i,true);});
            actor.close();
        }
        session->retire();
        return 0;
    }catch(const std::exception& e){std::cerr<<"FATAL "<<e.what()<<'\n';return 1;}
}
