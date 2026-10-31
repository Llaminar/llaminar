/**
 * @file Measure__ImmutableAdaptiveLookup.cpp
 * @brief One explicit captured protocol2 adaptive component per unprofiled timing process.
 *
 * Reuses the actual admitted component/source/service fixture. Native events
 * bracket complete retained gather plus conditional-close submissions; source
 * preparation, control publication, warmup, CPU decode and terminal readbacks
 * are outside timing. This is sized component evidence, never automatic tier
 * selection, a serving admission certificate or GPU-only decode bandwidth.
 */
#include "backends/GPUDeviceProcessLifetime.h"
#include "fixtures/qwen4exp/AdaptiveLookupNativeComponent.h"
#include "kernels/qwen4exp/ImmutableLookupKernels.h"
#include <charconv>
#include <chrono>
#include "backends/ComputeBackend.h"
#include <iomanip>
#include <iostream>
#include <set>
#include <string_view>
#include <sys/resource.h>
#include <sys/vfs.h>

using namespace llaminar2;
using namespace llaminar2::test::adaptive_native;
namespace {
/** @brief Explicit one-process geometry; no implicit policy or alternate execution path. */
struct Options{DeviceId device=DeviceId::invalid();int rows=0,live=-1,width=0,iterations=100,batches=7,warmup=10;std::uint32_t slots=0,hostSlots=0;std::size_t tableRows=0,workingSet=0;bool originalIQ4=false;};
/** @return Complete bounded decimal, before backend/materialization.
 * @param s Exact unsigned decimal with no suffix/trailing characters. */
std::size_t number(std::string_view s){std::size_t n=0;const auto r=std::from_chars(s.data(),s.data()+s.size(),n);require(r.ec==std::errc{}&&r.ptr==s.data()+s.size(),"Invalid adaptive component decimal");return n;}
/** @return Validated explicit workload; parser performs no I/O/backend activity.
 * @param argc Argument count. @param argv Exact alternating harness option/value arguments. */
Options options(int argc,char **argv){
    Options o;std::set<std::string_view> seen;
    for(int i=1;i<argc;i+=2){require(i+1<argc,"Missing adaptive option value");const std::string_view key=argv[i],value=argv[i+1];require(seen.insert(key).second,"Duplicate adaptive option");
        if(key=="--backend"){require(value=="cuda"||value=="rocm","Adaptive backend must be cuda or rocm");o.device=value=="cuda"?DeviceId::cuda(0):DeviceId::rocm(0);continue;}
        if(key=="--format"){require(value=="IQ4_NL","Adaptive timing supports explicit original IQ4_NL only");o.originalIQ4=true;continue;}
        const auto n=number(value);require(n<=static_cast<std::size_t>(std::numeric_limits<int>::max()),"Adaptive scalar exceeds INT32");
        if(key=="--width")o.width=static_cast<int>(n);else if(key=="--rows")o.rows=static_cast<int>(n);else if(key=="--live")o.live=static_cast<int>(n);else if(key=="--slots")o.slots=static_cast<std::uint32_t>(n);else if(key=="--host-slots")o.hostSlots=static_cast<std::uint32_t>(n);else if(key=="--table-rows")o.tableRows=n;else if(key=="--working-set")o.workingSet=n;else if(key=="--iterations")o.iterations=static_cast<int>(n);else if(key=="--batches")o.batches=static_cast<int>(n);else if(key=="--warmup")o.warmup=static_cast<int>(n);else throw std::invalid_argument("Unknown adaptive component option");
    }
    require(o.originalIQ4&&o.width==160&&o.device.is_gpu()&&o.rows>0&&o.rows<=2048&&o.live>=0&&o.live<=o.rows,"Exact native endpoint and original bounded live prefix required");
    require(o.tableRows>0&&o.workingSet>0&&o.workingSet<=o.tableRows&&o.workingSet<=std::max<std::size_t>(1,static_cast<std::size_t>(o.live)*16),"Working set must be observed in one original live frame");
    require(o.slots<=o.tableRows&&o.hostSlots<=o.tableRows&&product(o.tableRows,90)<=64u*1024*1024,"Explicit bounded synthetic source/cache extents required");
    require(o.iterations>0&&o.batches>0&&o.warmup>0,"Positive adaptive timing counts required");return o;
}
/** @brief Timing-capable native events retire after their last terminal wait. */
struct Events{Component &c;void *start,*stop;explicit Events(Component &c):c(c),start(c.backend->createTimingEvent(c.device.ordinal)),stop(c.backend->createTimingEvent(c.device.ordinal)){if(!start||!stop){if(start)c.backend->destroyEvent(start,c.device.ordinal);if(stop)c.backend->destroyEvent(stop,c.device.ordinal);throw std::runtime_error("Adaptive timing events unavailable");}}~Events(){if(!c.backend->synchronizeStream(c.stream,c.device.ordinal))std::terminate();c.backend->destroyEvent(stop,c.device.ordinal);c.backend->destroyEvent(start,c.device.ordinal);}};
/** @brief Independent production CPU decode and full poisoned-suffix comparison outside timing.
 * @param c Completed actual component. @param ids Exact original live source coordinates. */
void validate(Component &c,std::span<const std::int64_t> ids){
    auto actual=c.read(c.output(),c.maximumIds*160);std::vector<float> expected(actual.size(),-12345.f);std::vector<std::byte> rows(ids.size()*90);
    for(std::size_t i=0;i<ids.size();++i)std::memcpy(rows.data()+i*90,c.source.native.data()+static_cast<std::size_t>(ids[i])*90,90);
    decodeCPUImmutableLookupRows(TensorType::IQ4_NL,160,90,rows.data(),ids.size(),expected.data());require(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float))==0,"Adaptive component native output/suffix mismatch");require(c.read(c.control->error(),1).front()==0,"Adaptive component canonical error");
    require(c.protocolState().issuedSequence==c.protocolState().consumedSequence,"Adaptive component source interval unretired");require(c.frameState().held_pins==0,"Adaptive component native reader pin leaked");require(c.kernels->captureIdentity(*c.storage)==c.identity,"Adaptive component capture identity drift");
}
/** @brief Prepare exactly one original-source geometry, then report complete submissions.
 * @param o Complete explicitly selected immutable benchmark variant. */
void measure(const Options &o){
    auto &manager=DeviceManager::instance();manager.initialize(-1,false);const auto wanted=o.device.is_cuda()?ComputeBackendType::GPU_CUDA:ComputeBackendType::GPU_ROCM;
    std::optional<PhysicalMemoryResource> gpuResource,hostResource;
    for(const auto &d:manager.devices()){if(d.type==wanted&&d.device_id==o.device.ordinal)gpuResource=PhysicalMemoryResource{0,o.device,d.total_memory_bytes,d.free_memory_bytes};if(d.type==ComputeBackendType::CPU&&d.device_id==0)hostResource=PhysicalMemoryResource{0,DeviceId::cpu(),d.total_memory_bytes,d.free_memory_bytes};}
    require(gpuResource&&hostResource,"Adaptive benchmark requires frozen actual endpoint/CPU observations");
    auto *backend=getBackendFor(o.device);require(backend&&backend->deviceCount()>o.device.ordinal,"Adaptive selected native backend unavailable");if(!hasCPUBackend())initCPUBackend(-1);
    auto &worker=GPUDeviceContextPool::instance().getContext(o.device);worker.submitAndWait([&]{
        const auto native=iq4Source(o.tableRows,160);Source source(native,o.tableRows,160,GGUFTensorType::IQ4_NL);Component c(source,o.device,worker,o.rows,o.slots,AdaptiveFixtureGraphFamily::LookupAndTerminal,o.hostSlots,std::array{*gpuResource,*hostResource});
        c.prepareBundle();c.initializeMetadata();c.admit();std::vector<std::int64_t> ids(static_cast<std::size_t>(o.live)*16);for(std::size_t i=0;i<ids.size();++i)ids[i]=static_cast<std::int64_t>(i%o.workingSet);
        c.workload(o.live,ids);c.capture();Events events(c);std::vector<GPUGraphKernelNodeInfo> nodes;std::string detail;require(c.graph->inspectKernelNodes(nodes,&detail),detail.c_str());require(!nodes.empty(),"Adaptive native node inventory missing");
        for(const auto &n:nodes){require(n.valid()&&n.local_memory_bytes_per_thread==0,"Adaptive kernel resource record/spill limit invalid");std::cout<<"ADAPTIVE_NODE name="<<std::quoted(n.name)<<" grid="<<n.grid_x<<'x'<<n.grid_y<<'x'<<n.grid_z<<" block="<<n.block_x<<'x'<<n.block_y<<'x'<<n.block_z<<" registers="<<n.registers_per_thread<<" local_bytes="<<n.local_memory_bytes_per_thread<<" shared_bytes="<<n.static_shared_memory_bytes+n.dynamic_shared_memory_bytes<<" active_blocks_per_sm="<<n.max_active_blocks_per_sm<<'\n';}
        struct statfs fs{};require(::fstatfs(source.fd,&fs)==0,"Adaptive source filesystem observation failed");struct rusage before{},after{};require(::getrusage(RUSAGE_SELF,&before)==0,"Adaptive process page-fault observation failed");
        const auto first0=std::chrono::steady_clock::now();c.launch();c.join();const auto first1=std::chrono::steady_clock::now();validate(c,ids);
        const auto first=c.frameState();for(int i=0;i<o.warmup;++i)c.launch();c.join();validate(c,ids);
        const auto warmFrame=c.frameState();require(c.close(),"Adaptive warm interval terminal failed");const auto countersBefore=c.service->stats();const auto sequenceBefore=c.protocolState().issuedSequence;
        c.admit();c.workload(o.live,ids);
        std::cout<<"ADAPTIVE_COMPONENT scope=explicit_sized_component protocol=2 format=IQ4_NL width=160 row_bytes=90 rows="<<o.rows<<" live="<<o.live<<" gpu_slots="<<o.slots<<" host_slots="<<o.hostSlots<<" source_rows="<<o.tableRows<<" source_native_bytes="<<native.size()<<" working_set_rows="<<(ids.empty()?0:o.workingSet)<<" live_ids="<<ids.size()<<" live_output_bytes="<<ids.size()*160*4<<" gpu_cache_bytes="<<c.geometry->cache().bytes<<" mandatory_gpu_bytes="<<c.geometry->mandatoryDeviceBytes()<<" pma_host_table_bytes="<<c.memory->plannedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableTableStorage)<<" pma_host_source_channel_bytes="<<c.memory->plannedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableLookupStaging)<<" pma_fixture_host_backing_bytes="<<c.memory->plannedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ActivationArena)<<" pma_original_gpu_activation_bytes="<<c.memory->plannedBytes(c.device,PhysicalMemoryOwner::ActivationArena)<<" filesystem_type="<<fs.f_type<<" os_page_cache=uncontrolled_no_global_drop application_hits_are_not_os_hits=1 initial_source_misses="<<first.source_rows<<" last_warm_source_rows="<<warmFrame.source_rows<<" first_complete_wall_us="<<std::chrono::duration<double,std::micro>(first1-first0).count()<<" setup_and_warmup_timed=0\n";
        std::vector<float> samples;
        for(int batch=0;batch<o.batches;++batch){require(c.backend->recordEvent(events.start,c.device.ordinal,c.stream),"Adaptive start event failed");for(int i=0;i<o.iterations;++i)c.launch();require(c.backend->recordEvent(events.stop,c.device.ordinal,c.stream),"Adaptive stop event failed");require(c.backend->waitForEvent(events.stop,c.device.ordinal),"Adaptive stop wait failed");float ms=0;require(c.backend->eventElapsedTimeMs(events.start,events.stop,c.device.ordinal,&ms),"Adaptive elapsed event unavailable");validate(c,ids);samples.push_back(ms*1000/o.iterations);std::cout<<"ADAPTIVE_TIMING batch="<<batch<<" iterations="<<o.iterations<<" complete_captured_transaction_us="<<std::setprecision(9)<<samples.back()<<" no_timed_host_payload_observation=1\n";}
        const auto sequenceAfter=c.protocolState().issuedSequence;const auto timed=static_cast<std::uint64_t>(o.batches)*o.iterations;require(sequenceAfter>=sequenceBefore&&sequenceAfter-sequenceBefore<=timed,"Adaptive timed source sequence drift");
        require(::getrusage(RUSAGE_SELF,&after)==0,"Adaptive terminal page-fault observation failed");require(c.close(),"Adaptive real source terminal failed");const auto stats=c.service->stats();
        std::uint64_t gpuAdmissions=0,gpuEvictions=0,gpuReady=0;
        if(c.cache){const auto owner=c.read(reinterpret_cast<const policy::OwnerControl*>(static_cast<const std::byte*>(c.cache->gpu_data_ptr())+c.geometry->cache().owner),1).front();require(owner.flags==policy::OwnerInitialized,"Adaptive optional owner poisoned");gpuAdmissions=owner.next_slot_generation;gpuReady=owner.filled;require(gpuAdmissions>=gpuReady,"Adaptive native slot lifetime counter invalid");gpuEvictions=gpuAdmissions-gpuReady;}
        require(c.memory->claimedBytes(c.device,PhysicalMemoryOwner::ImmutableTableStorage,PhysicalMemoryMaterializationKind::NewAllocation)==c.geometry->cache().bytes,"Adaptive canonical cache lease changed");
        const std::uint64_t submitted=1+o.warmup+static_cast<std::uint64_t>(o.batches)*o.iterations;
        const auto actual=RetainedStorageTerminalObservation::between(c.generation,HostImmutableRowMode::AdaptiveRAMSSD,countersBefore,stats).counters;
        require(actual.requests==sequenceAfter-sequenceBefore&&actual.requests<=timed,"Adaptive source request receipts disagree with actual sequence delta");
        require(actual.ids<=timed*o.workingSet&&actual.nativeBytes==actual.ids*90&&actual.idBytes==actual.ids*8,"Adaptive actual live source byte receipts disagree");
        require(actual.failures==0&&actual.terminalBytes==8,"Adaptive actual timed terminal receipt invalid");
        if(!warmFrame.source_rows)require(actual.requests==0&&actual.ids==0&&actual.nativeBytes==0,"Proven all-hit unchanged input issued an unexpected source request");
        std::cout<<"ADAPTIVE_SOURCE submitted="<<submitted<<" timed_submissions="<<timed<<" requests="<<actual.requests<<" compact_ids="<<actual.ids<<" native_bytes="<<actual.nativeBytes<<" id_bytes="<<actual.idBytes<<" protocol_metadata_bytes="<<actual.metadataBytes<<" ack_bytes="<<actual.acknowledgementBytes<<" terminal_bytes="<<actual.terminalBytes<<" cpu_cache_hits="<<actual.ramRows<<" cpu_source_rows="<<actual.sourceRows<<" cpu_promotions="<<actual.cacheAdmissions<<" cpu_evictions="<<actual.cacheEvictions<<" gpu_lifetime_committed_admissions="<<gpuAdmissions<<" gpu_lifetime_evictions="<<gpuEvictions<<" gpu_ready_rows="<<gpuReady<<" gpu_counter_basis=successful_native_owner_generations_no_cancellations minor_fault_delta="<<after.ru_minflt-before.ru_minflt<<" major_fault_delta="<<after.ru_majflt-before.ru_majflt<<" page_fault_scope=process_includes_validation prior_all_hit_proven="<<(warmFrame.source_rows==0)<<" native_sequence_delta="<<sequenceAfter-sequenceBefore<<'\n';
        std::sort(samples.begin(),samples.end());std::cout<<"ADAPTIVE_MEDIAN complete_transaction_us="<<samples[samples.size()/2]<<" validation=byte_exact samples="<<samples.size()<<" admission=component_only serving=unqualified source_hash=none profiler_attachment=none\n";
    });
}
} // namespace
/** @brief Run one explicit component variant; errors never select another storage path.
 * @param argc Argument count. @param argv Explicit harness selections.
 * @return Zero only after byte validation, actual source receipts and complete timing succeed. */
int main(int argc,char **argv){
    llaminar2::GPUDeviceProcessLifetime gpu_process_lifetime;
try{measure(options(argc,argv));return 0;}catch(const std::exception &e){std::cerr<<"ADAPTIVE_COMPONENT_ERROR "<<e.what()<<'\n';return 1;}}
