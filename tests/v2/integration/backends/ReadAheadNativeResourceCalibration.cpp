/**
 * @file ReadAheadNativeResourceCalibration.cpp
 * @brief Explicit diagnostic one-stream/two-event native backing observations, never production admission.
 *
 * A fresh process authors a single fixed resource family through the original
 * worker APIs. Parent terminal/payload are separately observed baseline owners.
 * All checkpoints retain peaks/lazy use/capture/replay/teardown evidence. Raw
 * deltas are observations, not certified limits or a second admission ledger.
 * The production certificate catalog is never changed or bypassed by a fake
 * certificate. Only this named diagnostic directly calls unqualified creators.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include "backends/GPUGraphMemoryContract.h"
#include "ReadAheadCalibrationVersions.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IGPUGraphCapture.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "transfer/TransferEngine.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <malloc.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>
#include <unistd.h>
#ifdef HAVE_CUDA
#include "backends/cuda/CUDAHardwareMetadata.h"

#endif
#ifdef HAVE_ROCM
#include "backends/rocm/ROCmHardwareMetadata.h"

#endif
#ifndef READ_AHEAD_RESOURCE_SOURCE_SHA256
#error "Resource calibration needs the exact resource source identity"
#endif
#ifndef READ_AHEAD_CALIBRATION_SOURCE_SHA256
#error "Resource calibration needs its exact diagnostic source identity"
#endif
namespace llaminar2::test {
namespace {
using Json=nlohmann::ordered_json;
/** @brief Fail an explicit diagnostic instead of skipping or changing policy. */
void require(bool value,const char* reason){if(!value)throw std::runtime_error(reason);}
/** @brief Parse bounded positive CLI integers without trailing characters or overflow. */
std::size_t number(const std::string& text){std::size_t used=0;const auto value=std::stoull(text,&used);require(used==text.size() && value && value<=std::numeric_limits<std::size_t>::max(),"Invalid positive calibration integer");return static_cast<std::size_t>(value);}
/** @brief One source-defined diagnostic population; no model/cache capacity decisions. */
struct Options {DeviceId device=DeviceId::invalid();std::size_t bytes=4096,replays=20,lifetimes=1,parentLifetimes=1,failCreator=0;bool handshake=false;};
/** @brief Admit exactly one explicit native endpoint and reject unknown/malformed options before runtime use. */
Options parse(int argc,char**argv){Options result;for(int i=1;i<argc;++i){const std::string key=argv[i];if(key=="--handshake"){result.handshake=true;continue;}require(i+1<argc,"Missing calibration argument");const std::string value=argv[++i];if(key=="--device"){const auto colon=value.find(':');require(colon!=std::string::npos,"Device must be cuda:N or rocm:N");const auto ordinalText=value.substr(colon+1);std::size_t used=0;const auto ordinal=std::stoi(ordinalText,&used);require(used==ordinalText.size() && ordinal>=0,"Invalid device ordinal");if(value.substr(0,colon)=="cuda")result.device=DeviceId::cuda(ordinal);else if(value.substr(0,colon)=="rocm")result.device=DeviceId::rocm(ordinal);else throw std::invalid_argument("Unknown native backend");}else if(key=="--bytes")result.bytes=number(value);else if(key=="--replays")result.replays=number(value);else if(key=="--lifetimes")result.lifetimes=number(value);else if(key=="--parent-terminal-lifetimes")result.parentLifetimes=number(value);else if(key=="--fail-creator")result.failCreator=number(value);else throw std::invalid_argument("Unknown calibration option: "+key);}
    require(result.device.is_gpu() && result.bytes<=64u*1024u*1024u && result.replays<=1024 && result.lifetimes<=128 && result.parentLifetimes<=128 && result.failCreator<=3,"Calibration geometry outside explicit diagnostic range");return result;}
/** @return Canonical hexadecimal physical UUID, without inventing a PCI alias. */
template<class UUID>std::string uuid(const UUID& value){static constexpr char hex[]="0123456789abcdef";std::string result;for(auto byte:value){result+=hex[(byte>>4)&15];result+=hex[byte&15];}return result;}
/** @brief Project already admitted physical metadata and actual selected runtime versions. */
Json closure(DeviceId device){Json value={{"device",device.toString()},{"resource_source_sha256",READ_AHEAD_RESOURCE_SOURCE_SHA256},{"calibration_source_sha256",READ_AHEAD_CALIBRATION_SOURCE_SHA256}};int runtime=0,driver=0;
#ifdef HAVE_CUDA
    if(device.is_cuda()){const auto& hardware=cudaHardwareMetadata().at(device.ordinal);const auto versions=readAheadCalibrationCUDAVersions();runtime=versions.first;driver=versions.second;value["physical_uuid"]=uuid(hardware.uuid);value["pci"]=hardware.pci_address;value["architecture"]="sm_"+std::to_string(hardware.sm_major)+std::to_string(hardware.sm_minor);}
#endif
#ifdef HAVE_ROCM
    if(device.is_rocm()){const auto& hardware=rocmHardwareMetadata().at(device.ordinal);const auto versions=readAheadCalibrationROCmVersions();runtime=versions.first;driver=versions.second;value["physical_uuid"]=uuid(hardware.uuid);value["pci"]=hardware.pci_address;value["architecture"]=hardware.isa_target;}
#endif
    require(value.contains("physical_uuid") && runtime>0 && driver>0,"Incomplete native closure");value["runtime_version"]=runtime;value["driver_api_version"]=driver;value["compiler"]=__VERSION__;value["cplusplus"]=__cplusplus;return value;
}
/** @brief Flush a process-correlated receipt checkpoint; parent handshake samples outside native work. */
class Receipt final {
public:
    Receipt(Options options,Json identity,std::weak_ptr<IWorkerGPUContext> worker={}):options_(options),identity_(std::move(identity)),worker_(std::move(worker)){sequence_=identity_.contains("physical_uuid")?1:0;}
    /** @brief Retain host/device observations, explicitly excluding post-reset native queries. */
    void checkpoint(const std::string& stage,IBackend* backend=nullptr,Json extra=Json::object()){
        const auto heap=::mallinfo2();rusage usage{};require(::getrusage(RUSAGE_SELF,&usage)==0,"getrusage failed");Json record={{"schema","llaminar.read_ahead_native_resource_calibration.v1"},{"pid",::getpid()},{"sequence",sequence_++},{"monotonic_ns",std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()},{"stage",stage},{"closure",identity_},{"geometry",{{"side_streams",1},{"ordering_events",2},{"borrowed_parent_terminal_events",1},{"concurrent_parents",1},{"payload_bytes",options_.bytes},{"replays",options_.replays},{"lifetimes",options_.lifetimes},{"parent_terminal_lifetimes",options_.parentLifetimes},{"read_ahead_owner","native_read_ahead_family"},{"parent_terminal_owner","native_read_ahead_parent_terminal"}}},{"host",{{"maxrss_kib",usage.ru_maxrss},{"mallinfo_arena",heap.arena},{"mallinfo_hblkhd",heap.hblkhd},{"mallinfo_uordblks",heap.uordblks}}},{"qualified",false},{"certified_limit",nullptr},{"production_admission",false},{"extra",std::move(extra)}};
        if(backend){auto worker=worker_.lock();require(bool(worker),"Device checkpoint lost its original worker");worker->submitAndWait([&]{const auto accounting=backend->deviceAllocationAccounting(options_.device.ordinal);record["device"]={{"driver_free_bytes",backend->deviceMemoryFree(options_.device.ordinal)},{"driver_total_bytes",backend->deviceMemoryTotal(options_.device.ordinal)},{"runtime_generation",backend->deviceRuntimeGeneration(options_.device.ordinal)},{"canonical_accounting_supported",accounting.supported},{"canonical_active_allocations",accounting.active_allocations},{"canonical_active_bytes",accounting.active_bytes}};require(accounting.supported && record["device"]["driver_free_bytes"].get<std::size_t>()>0,"Incomplete diagnostic device observation");});}
        std::cout<<record.dump()<<'\n'<<std::flush;if(options_.handshake){std::string ack;require(bool(std::getline(std::cin,ack)) && ack=="continue","Parent observation handshake failed");}
    }
private:Options options_;Json identity_;std::weak_ptr<IWorkerGPUContext> worker_;std::uint64_t sequence_=0;
};
/** @brief Exact diagnostic owner prefix; created handles retire in reverse after actual terminal completion. */
struct Handles {
    std::shared_ptr<IWorkerGPUContext> worker;void* side=nullptr;void* fork=nullptr;void* join=nullptr;bool active=false;
    /** @brief Cleanup failure/outstanding readers are fatal, never evidence of reclaimed backing. */
    ~Handles(){if(active)std::terminate();try{worker->submitAndWait([&]{if(join)worker->destroyEvent(join);if(fork)worker->destroyEvent(fork);if(side)worker->destroyStream(side);});}catch(...){std::terminate();}}
};
/** @brief Run an explicit raw diagnostic; it cannot mint production resource admission. */
int run(const Options& options){
    // This process-start record is deliberately before backend lookup/context
    // admission. The external parent observes module loading before main too.
    Receipt early(options,{{"device",options.device.toString()},{"resource_source_sha256",READ_AHEAD_RESOURCE_SOURCE_SHA256},{"calibration_source_sha256",READ_AHEAD_CALIBRATION_SOURCE_SHA256}});early.checkpoint("before_backend_lookup");
    auto* backend=getBackendFor(options.device);require(backend && backend->deviceCount()>options.device.ordinal,"Requested calibration native endpoint unavailable");
    auto worker=GPUDeviceContextPool::instance().retainContext(options.device);require(bool(worker),"Original diagnostic context unavailable");Json identity;worker->submitAndWait([&]{identity=closure(options.device);});Receipt receipt(options,identity,worker);receipt.checkpoint("original_context_ready",backend);
    std::size_t total=0,available=0;worker->submitAndWait([&]{total=backend->deviceMemoryTotal(options.device.ordinal);available=backend->deviceMemoryFree(options.device.ordinal);});const auto bytes=2*options.bytes;const auto graphBytes=GPUGraphMemoryContract::reservationBytesPerExecutable(options.device,GPUGraphExecutableClass::General);PhysicalMemoryPlanBuilder builder;builder.add({0,options.device,total,available},PhysicalMemoryOwner::ExecutionWorkspace,bytes);builder.add({0,options.device,total,available},PhysicalMemoryOwner::NativeGraphExecutable,graphBytes);auto memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()),0);auto payloadClaim=memory->claimNewAllocation(options.device,PhysicalMemoryOwner::ExecutionWorkspace,bytes);
    // Native graph storage uses the existing graph contract, independently of
    // the unqualified stream/event populations. Keep its original commitment
    // through actual context reset because backend graph pools may retain it.
    auto graphReservation=memory->reserveNewAllocations(options.device,PhysicalMemoryOwner::NativeGraphExecutable,graphBytes);
    void* payload=nullptr;void* terminal=nullptr;void* primary=nullptr;
    std::exception_ptr failure;
    bool injected=false;
    try {
    worker->submitAndWait([&]{primary=worker->defaultStream();require(reinterpret_cast<std::uintptr_t>(primary)>2,"Default sentinel is not a diagnostic primary");payload=backend->allocate(bytes,options.device.ordinal);require(payload,"Diagnostic payload allocation failed");});
    receipt.checkpoint("explicit_parent_payload_allocated",backend,{{"explicit_payload_bytes",bytes},{"opaque_family_created",false}});
    for(std::size_t parentLifetime=0;parentLifetime<options.parentLifetimes;++parentLifetime){
        receipt.checkpoint("before_parent_terminal",backend,{{"parent_lifetime",parentLifetime}});
        worker->submitAndWait([&]{const auto created=worker->createEvent(GPUEventPurpose::Ordering);require(reinterpret_cast<std::uintptr_t>(created)>2,"Diagnostic parent terminal failed");terminal=created;});
        receipt.checkpoint("after_parent_terminal_created",backend,{{"parent_lifetime",parentLifetime},{"terminal_handle",reinterpret_cast<std::uintptr_t>(terminal)}});
        worker->submitAndWait([&]{require(worker->recordEventChecked(terminal,primary) && worker->synchronizeEventChecked(terminal),"Initial empty parent terminal did not retire");});
        receipt.checkpoint("first_parent_terminal_record_complete",backend,{{"parent_lifetime",parentLifetime}});
        if(parentLifetime+1<options.parentLifetimes){worker->submitAndWait([&]{worker->destroyEvent(terminal);terminal=nullptr;});receipt.checkpoint("after_parent_terminal_destroyed",backend,{{"parent_lifetime",parentLifetime}});}
    }
    worker->submitAndWait([&]{require(backend->memset(payload,0,bytes,options.device.ordinal,primary),"Diagnostic payload setup failed");require(worker->recordEventChecked(terminal,primary) && worker->synchronizeEventChecked(terminal),"Baseline parent did not retire");});receipt.checkpoint("parent_payload_and_terminal_baseline",backend,{{"explicit_payload_bytes",bytes},{"opaque_family_created",false}});
    for(std::size_t lifetime=0;lifetime<options.lifetimes;++lifetime){Handles handles{worker};std::unique_ptr<IGPUGraphCapture> graph;std::vector<std::byte> observed(options.bytes);unsigned creators=0;
        const auto inject=[&]{if(++creators==options.failCreator){injected=true;throw std::runtime_error("Intentional diagnostic creator failure");}};
        try{receipt.checkpoint("before_stream",backend,{{"lifetime",lifetime}});worker->submitAndWait([&]{inject();const auto created=worker->createStream();require(reinterpret_cast<std::uintptr_t>(created)>2 && created!=primary,"Diagnostic side stream failed/aliased");handles.side=created;});receipt.checkpoint("after_stream",backend,{{"lifetime",lifetime},{"handle",reinterpret_cast<std::uintptr_t>(handles.side)}});
            worker->submitAndWait([&]{inject();const auto created=worker->createEvent(GPUEventPurpose::Ordering);require(reinterpret_cast<std::uintptr_t>(created)>2,"Diagnostic fork event failed");handles.fork=created;});receipt.checkpoint("after_fork",backend,{{"lifetime",lifetime},{"handle",reinterpret_cast<std::uintptr_t>(handles.fork)}});
            worker->submitAndWait([&]{inject();const auto created=worker->createEvent(GPUEventPurpose::Ordering);require(reinterpret_cast<std::uintptr_t>(created)>2 && created!=handles.fork,"Diagnostic join event failed/aliased");handles.join=created;});receipt.checkpoint("after_join",backend,{{"lifetime",lifetime},{"handle",reinterpret_cast<std::uintptr_t>(handles.join)}});
            const auto body=[&]{require(worker->recordEventChecked(handles.fork,primary) && worker->waitEventChecked(handles.fork,handles.side),"Diagnostic fork failed");require(backend->memset(payload,0x2a,options.bytes,options.device.ordinal,handles.side),"Diagnostic actual side work failed");require(worker->recordEventChecked(handles.join,handles.side) && worker->waitEventChecked(handles.join,primary),"Diagnostic join failed");require(backend->deviceCopyAsync(static_cast<std::byte*>(payload)+options.bytes,payload,options.bytes,options.device.ordinal,primary),"Diagnostic actual primary consumer failed");};
            const auto complete=[&]{require(worker->recordEventChecked(terminal,primary) && worker->synchronizeEventChecked(terminal),"Diagnostic parent terminal did not complete");handles.active=false;require(backend->deviceToHostOnStream(observed.data(),static_cast<std::byte*>(payload)+options.bytes,options.bytes,options.device.ordinal,primary),"Diagnostic native byte observation failed");require(worker->recordEventChecked(terminal,primary) && worker->synchronizeEventChecked(terminal),"Diagnostic byte observation did not retire");require(std::all_of(observed.begin(),observed.end(),[](auto value){return value==std::byte{0x2a};}),"Diagnostic native byte oracle mismatch");};
            // Poison both producer and consumer storage before each transaction.
            // Otherwise the eager warmup's constant result could falsely certify
            // a later graph that omitted its side write or primary read.
            const auto poison=[&]{require(backend->memset(payload,0xa5,bytes,options.device.ordinal,primary),"Diagnostic replay poison failed");};
            worker->submitAndWait([&]{handles.active=true;poison();body();complete();});receipt.checkpoint("lazy_first_use_complete",backend,{{"lifetime",lifetime},{"byte_oracle",true}});
            worker->submitAndWait([&]{graph=worker->createGraphCapture(primary);require(bool(graph),"Diagnostic original graph unavailable");ScopedBackendGraphCapture capture(*worker,*graph,"explicit read-ahead native resource calibration");require(capture.begin(),"Diagnostic capture begin failed");body();capture.finish();});receipt.checkpoint("capture_definition_complete",backend,{{"lifetime",lifetime},{"nodes",graph->nodeCount()}});
            worker->submitAndWait([&]{std::string detail;require(graph->validateExecutableMemoryClass(GPUGraphExecutableClass::General,&detail),detail.c_str());require(graph->instantiate(),"Diagnostic graph instantiation failed");});receipt.checkpoint("capture_instantiated",backend,{{"lifetime",lifetime},{"nodes",graph->nodeCount()}});
            for(std::size_t replay=0;replay<options.replays;++replay){worker->submitAndWait([&]{handles.active=true;poison();require(graph->launch(),"Diagnostic retained graph launch failed");complete();});receipt.checkpoint(replay==0?"first_parent_replay_complete":"parent_replay_complete",backend,{{"lifetime",lifetime},{"replay",replay},{"byte_oracle",true}});}
            worker->submitAndWait([&]{graph.reset();});receipt.checkpoint("capture_destroyed",backend,{{"lifetime",lifetime}});
        }catch(const std::exception& error){
            if(injected && !handles.active)receipt.checkpoint("injected_creator_failure",backend,{{"creator",creators},{"reason",error.what()}});
            else {
                failure=std::current_exception();
                receipt.checkpoint("diagnostic_failure_cleanup_begin",nullptr,{{"reason",error.what()},{"lifetime",lifetime}});
                // This explicit diagnostic failure boundary may drain; production
                // capture never gains a blocking or coherence-relaxing path.
                worker->submitAndWait([&]{
                    if((handles.side && !worker->synchronizeStreamChecked(handles.side)) || !worker->synchronizeStreamChecked(primary))std::terminate();
                    require(worker->recordEventChecked(terminal,primary) && worker->synchronizeEventChecked(terminal),"Diagnostic failure terminal drain failed");
                    handles.active=false;
                    graph.reset();
                });
                receipt.checkpoint("diagnostic_failure_terminal_drained",backend,{{"lifetime",lifetime}});
            }
        }
        const bool ownedJoin=handles.join!=nullptr;worker->submitAndWait([&]{if(handles.join){worker->destroyEvent(handles.join);handles.join=nullptr;}});receipt.checkpoint("after_destroy_join",backend,{{"lifetime",lifetime},{"owned",ownedJoin}});
        const bool ownedFork=handles.fork!=nullptr;worker->submitAndWait([&]{if(handles.fork){worker->destroyEvent(handles.fork);handles.fork=nullptr;}});receipt.checkpoint("after_destroy_fork",backend,{{"lifetime",lifetime},{"owned",ownedFork}});
        const bool ownedSide=handles.side!=nullptr;worker->submitAndWait([&]{if(handles.side){worker->destroyStream(handles.side);handles.side=nullptr;}});receipt.checkpoint("after_destroy_stream",backend,{{"lifetime",lifetime},{"owned",ownedSide}});
        if(injected || failure)break;
    }
    }catch(const std::exception& error){
        failure=std::current_exception();
        receipt.checkpoint("diagnostic_baseline_failure_cleanup_begin",nullptr,{{"reason",error.what()}});
        worker->submitAndWait([&]{if(primary && !worker->synchronizeStreamChecked(primary))std::terminate();});
    }
    auto ticket=TransferEngine::instance().beginExclusiveModelRetirement({.device=options.device,.reusable_workspace_bytes=bytes});worker->submitAndWait([&]{if(terminal){worker->destroyEvent(terminal);terminal=nullptr;}});receipt.checkpoint("after_parent_terminal_destroyed",backend,{{"parent_lifetime",options.parentLifetimes-1}});worker->submitAndWait([&]{if(payload){backend->free(payload,options.device.ordinal);payload=nullptr;}});payloadClaim={};receipt.checkpoint("explicit_baseline_owners_destroyed",backend);worker.reset();
    const auto retired=TransferEngine::instance().completeExclusiveModelRetirement(std::move(ticket));require(retired.runtime_reset_invoked && retired.retired_context_generation && retired.retired_runtime_generation && retired.successor_runtime_generation && retired.runtime_post_reset_state==DeviceRuntimePostResetState::Quiescent,"Diagnostic native context retirement proof incomplete");
    graphReservation={}; // Original graph-pool obligation ends only at proven reset.
    // Do NOT ask the native runtime for memory after reset: that can create a
    // successor context. The external parent owns post-reset observations.
    receipt.checkpoint("exclusive_context_retirement_complete",nullptr,{{"reset_invoked",retired.runtime_reset_invoked},{"retired_context_generation",retired.retired_context_generation},{"retired_runtime_generation",retired.retired_runtime_generation},{"successor_runtime_generation",retired.successor_runtime_generation},{"runtime_post_reset_state",to_string(retired.runtime_post_reset_state)},{"injected_failure",injected},{"diagnostic_failed",bool(failure)},{"certified_limit",nullptr}});if(failure)std::rethrow_exception(failure);return 0;
}
} // namespace
} // namespace llaminar2::test
/** @brief Diagnostic entrypoint; every failure is an explicit nonzero receipt/process outcome. */
int main(int argc,char**argv)
{
    // The canonical lexical owner admits native contexts and retires them
    // before process-exit callbacks, on both success and exception returns.
    llaminar2::GPUDeviceProcessLifetime process_lifetime;
    try{return llaminar2::test::run(llaminar2::test::parse(argc,argv));}
    catch(const std::exception& error){std::cerr<<"READ_AHEAD_CALIBRATION_ERROR: "<<error.what()<<'\n';return 1;}
}
