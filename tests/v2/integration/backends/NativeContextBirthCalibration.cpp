/** @file NativeContextBirthCalibration.cpp
 * @brief Raw diagnostic of original selected worker birth; never a certified admission.
 *
 * Exact mapped stack/object bytes use their original Preparing PMA. The opaque
 * stream/BLAS/Lt/32-event family is deliberately unqualified at this named test
 * boundary. Parent external observation starts before the first child SDK API.
 * Startup discovery, exact roots, original construction, use, destroy and actual
 * runtime reset are separate checkpoints. No fake certificate/UsePermit exists.
 */
#include "backends/NativeContextPreparation.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include "backends/IWorkerGPUContext.h"
#include "transfer/TransferEngine.h"
#include "ReadAheadCalibrationVersions.h"
#include "NativeContextBirthDiagnosticHooks.h"
#include "backends/NativeOrderingEventProgram.h"
#include "backends/NativeAuxiliaryResourceCatalog.h"
#include <nlohmann/json.hpp>
#include <array>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <malloc.h>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#ifdef HAVE_CUDA
#include "backends/cuda/CUDAHardwareMetadata.h"
#endif
#ifdef HAVE_ROCM
#include "backends/rocm/ROCmHardwareMetadata.h"
#endif
#ifndef NATIVE_CONTEXT_BIRTH_SOURCE_SHA256
#error "Birth diagnostic requires exact original constructor/source identity"
#endif
#ifndef NATIVE_CONTEXT_BIRTH_DIAGNOSTIC_SHA256
#error "Birth diagnostic requires exact diagnostic source identity"
#endif
namespace llaminar2::test {
/** @brief The sole raw diagnostic bridge to existing private exact-root constructors.
 * It creates no opaque contract, production certificate, pool entry or permit.
 */
struct NativeContextBirthDiagnosticAccess {
    /** @brief Create original Preparing PMA before any exact root mapping. */
    static std::shared_ptr<NativeStoragePreparationPmaBinding> source(std::span<const PhysicalMemoryResource> observations){
        return std::shared_ptr<NativeStoragePreparationPmaBinding>(new NativeStoragePreparationPmaBinding(observations,0));
    }
    /** @brief Map the actual compiled object and original default stack exactly once. */
    static std::shared_ptr<PreparedGPUWorkerStack> root(const std::shared_ptr<NativeStoragePreparationPmaBinding>& source,DeviceId device){
        const auto object=PreparedGPUContextFactory::objectGeometry(device);
        return PreparedGPUWorkerStack::create(source,GPUWorkerStackProgram::originalDefaults().withContextObject(object.bytes(),object.alignment()));
    }
    /** @brief Invoke the original placement constructor and original destructor/join owner. */
    static std::shared_ptr<IWorkerGPUContext> worker(DeviceId device,std::shared_ptr<PreparedGPUWorkerStack> root){
#ifdef HAVE_CUDA
        if(device.is_cuda())return PreparedGPUContextFactory::cuda(device.ordinal,std::move(root));
#endif
#ifdef HAVE_ROCM
        if(device.is_rocm())return PreparedGPUContextFactory::rocm(device.ordinal,std::move(root));
#endif
        throw std::invalid_argument("Requested raw worker backend was not compiled");
    }
};
namespace {
using Json=nlohmann::ordered_json;
using namespace birth_diagnostic;
/** @brief Diagnostic failure remains explicit and preserves parent partial records. */
void require(bool value,const char* reason){if(!value)throw std::runtime_error(reason);}
/** @brief Strict positive bounded integer; zero is admitted only for failure-disabled geometry. */
std::size_t number(const std::string& text,bool zero=false){std::size_t used=0;const auto value=std::stoull(text,&used);require(used==text.size() && (zero || value) && value<=std::numeric_limits<std::size_t>::max(),"Invalid calibration integer");return value;}
/** @brief Explicit diagnostic reset geometry; the ordinary warm program is unchanged. */
enum class RuntimeResetPolicy {FinalLifetime,EachLifetime};
struct Options {DeviceId device=DeviceId::invalid();std::size_t lifetimes=1,replays=20,fail=0;std::string lookupProgram;std::optional<std::size_t> lookupPrefix;bool handshake=false;RuntimeResetPolicy resetPolicy=RuntimeResetPolicy::FinalLifetime;};
/** @return Closed actual existing root/boundary event census; no arbitrary count is accepted. */
std::vector<NativeOrderingEventProgram> lookupPrograms(const std::string& name){
    using P=NativeOrderingEventProgram;
    if(name.empty())return {};
    if(name=="owner-c0" || name=="remote-c0")return {P::pooledOwner(false)};
    if(name=="owner-positive" || name=="remote-positive")return {P::pooledOwner(true)};
    if(name=="coordinator-c0")return {P::pooledCoordinator(false)};
    if(name=="coordinator-positive")return {P::pooledCoordinator(true)};
    if(name=="source")return {P::pooledSource()};
    if(name=="boundary")return {P::lookupSubmissionRetirement()};
    if(name=="local-c0" || name=="local-mixed" || name=="local-positive")return {
        P::pooledOwner(name!="local-c0"),P::pooledCoordinator(name=="local-positive"),P::pooledSource(),P::lookupSubmissionRetirement()};
    throw std::invalid_argument("Unknown closed lookup event diagnostic program");
}
/** @return Exact simultaneous auxiliary population following the original 32 baseline events. */
std::size_t lookupCount(const std::string& name){std::size_t count=0;for(const auto& p:lookupPrograms(name))count+=p.orderingEvents();return count;}
/** @brief Parse geometry before backend lookup or any vendor API. */
Options parse(int argc,char** argv){Options result;for(int i=1;i<argc;++i){const std::string key=argv[i];if(key=="--handshake"){result.handshake=true;continue;}if(key=="--reset-between-lifetimes"){result.resetPolicy=RuntimeResetPolicy::EachLifetime;continue;}require(i+1<argc,"Missing diagnostic argument");const std::string value=argv[++i];if(key=="--device"){
    const auto colon=value.find(':');require(colon!=std::string::npos,"Device must be cuda:N or rocm:N");const auto ordinal=number(value.substr(colon+1),true);require(ordinal<=std::numeric_limits<int>::max(),"Device ordinal overflow");
    if(value.substr(0,colon)=="cuda")result.device=DeviceId::cuda(ordinal);else if(value.substr(0,colon)=="rocm")result.device=DeviceId::rocm(ordinal);else throw std::invalid_argument("Unknown birth backend");
}else if(key=="--lifetimes")result.lifetimes=number(value);else if(key=="--replays")result.replays=number(value);else if(key=="--lookup-event-program")result.lookupProgram=value;else if(key=="--lookup-event-prefix")result.lookupPrefix=number(value,true);else if(key=="--fail-creator")result.fail=number(value);else throw std::invalid_argument("Unknown diagnostic option: "+key);}
require(result.device.is_gpu() && result.lifetimes<=128 && result.replays<=1024 && result.fail<=35 && (!result.fail || result.lifetimes==1) && !(result.fail && !result.lookupProgram.empty()) && (!result.lookupPrefix || !result.lookupProgram.empty()),"Invalid birth diagnostic geometry");require(!result.lookupPrefix || *result.lookupPrefix<=lookupCount(result.lookupProgram),"Lookup prefix exceeds its closed program");lookupPrograms(result.lookupProgram);return result;}
/** @brief Exact immutable physical identity, not a guessed ordinal alias. */
template<class UUID>std::string uuid(const UUID& value){static constexpr char hex[]="0123456789abcdef";std::string result;for(auto byte:value){result+=hex[(byte>>4)&15];result+=hex[byte&15];}return result;}
/** @brief Discovery/version APIs are explicitly included in startup, before selected worker birth. */
Json startupIdentity(DeviceId device){Json value;std::pair<int,int> versions{};
#ifdef HAVE_CUDA
if(device.is_cuda()){const auto& all=cudaHardwareMetadata();require(device.ordinal<all.size(),"Selected CUDA ordinal unavailable");const auto& hardware=all.at(device.ordinal);value={{"physical_uuid",uuid(hardware.uuid)},{"pci",hardware.pci_address},{"architecture","sm_"+std::to_string(hardware.sm_major)+std::to_string(hardware.sm_minor)},{"startup_visible_devices",all.size()}};versions=readAheadCalibrationCUDAVersions();}
#endif
#ifdef HAVE_ROCM
if(device.is_rocm()){const auto& all=rocmHardwareMetadata();require(device.ordinal<all.size(),"Selected ROCm ordinal unavailable");const auto& hardware=all.at(device.ordinal);value={{"physical_uuid",uuid(hardware.uuid)},{"pci",hardware.pci_address},{"architecture",hardware.isa_target},{"startup_visible_devices",all.size()}};versions=readAheadCalibrationROCmVersions();}
#endif
require(value.contains("physical_uuid") && versions.first>0 && versions.second>0,"Missing selected birth closure");value["runtime_version"]=versions.first;value["driver_api_version"]=versions.second;return value;}
/** @brief Project complete original pthread creation defaults without a GPU API.
 * The original prepared thread deep-copies these same defaults; exact stack and
 * object pages remain separately owned roots rather than opaque allowances. */
Json pthreadDefaults(){
    pthread_attr_t attributes;require(::pthread_getattr_default_np(&attributes)==0,"Original pthread defaults unavailable");
    std::size_t stack=0,guard=0;int detach=0,inherit=0,policy=0,scope=0;sched_param parameters{};cpu_set_t affinity,effective;CPU_ZERO(&affinity);CPU_ZERO(&effective);
    const bool complete=::pthread_attr_getstacksize(&attributes,&stack)==0 && ::pthread_attr_getguardsize(&attributes,&guard)==0 &&
        ::pthread_attr_getdetachstate(&attributes,&detach)==0 && ::pthread_attr_getinheritsched(&attributes,&inherit)==0 &&
        ::pthread_attr_getschedpolicy(&attributes,&policy)==0 && ::pthread_attr_getschedparam(&attributes,&parameters)==0 &&
        ::pthread_attr_getscope(&attributes,&scope)==0 && ::pthread_attr_getaffinity_np(&attributes,sizeof(affinity),&affinity)==0 &&
        ::sched_getaffinity(0,sizeof(effective),&effective)==0;
    require(::pthread_attr_destroy(&attributes)==0,"Original pthread attribute cleanup failed");require(complete,"Incomplete original pthread policy snapshot");
    Json cpus=Json::array(),allowed=Json::array();for(int cpu=0;cpu<CPU_SETSIZE;++cpu){if(CPU_ISSET(cpu,&affinity))cpus.push_back(cpu);if(CPU_ISSET(cpu,&effective))allowed.push_back(cpu);}
    return {{"stack_bytes",stack},{"guard_bytes",guard},{"detach_state",detach},{"inherit_sched",inherit},{"sched_policy",policy},{"sched_priority",parameters.sched_priority},{"scope",scope},{"default_affinity",cpus},{"effective_creator_affinity",allowed},{"cpu_set_bytes",sizeof(affinity)}};
}
/** @brief Handshake receipts contain host observations only; parent owns GPU sampling. */
class Receipt final {
public:
    explicit Receipt(Options options):options_(options){identity_={{"device",options.device.toString()},{"resource_source_sha256",NATIVE_CONTEXT_BIRTH_SOURCE_SHA256},{"calibration_source_sha256",NATIVE_CONTEXT_BIRTH_DIAGNOSTIC_SHA256},{"compiler",__VERSION__}};for(const auto* name:{"CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES","GPU_DEVICE_ORDINAL"}){const auto* value=std::getenv(name);identity_[name]=value?Json(value):Json(nullptr);}}
    /** @brief Freeze the original startup metadata before entering selected construction. */
    void identity(Json value){for(auto it=value.begin();it!=value.end();++it)identity_[it.key()]=it.value();}
    /** @brief Flush one source-correlated checkpoint before acknowledging further child APIs. */
    void checkpoint(const std::string& stage,Json extra=Json::object()){
        std::lock_guard lock(mutex_);const auto heap=::mallinfo2();rusage usage{};require(::getrusage(RUSAGE_SELF,&usage)==0,"getrusage failed");
        Json value={{"schema","llaminar.native_context_birth_calibration.v1"},{"pid",::getpid()},{"sequence",sequence_++},{"monotonic_ns",std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()},{"stage",stage},{"closure",identity_},{"geometry",{{"primary_streams",1},{"primary_blas_handles",1},{"secondary_blas_handles",1},{"ordering_events",32},{"lookup_event_program",options_.lookupProgram},{"lookup_event_count",lookupCount(options_.lookupProgram)},{"lookup_event_prefix",options_.lookupPrefix?Json(*options_.lookupPrefix):Json(nullptr)},{"selected_workers",1},{"lifetimes",options_.lifetimes},{"reset_between_lifetimes",options_.resetPolicy==RuntimeResetPolicy::EachLifetime},{"replays",options_.replays},{"fail_creator",options_.fail},{"pool_admission",false},{"opaque_family_excludes_exact_stack_object",true}}},{"host",{{"maxrss_kib",usage.ru_maxrss},{"mallinfo_arena",heap.arena},{"mallinfo_hblkhd",heap.hblkhd},{"mallinfo_uordblks",heap.uordblks}}},{"qualified",false},{"certified_limit",nullptr},{"production_admission",false},{"extra",std::move(extra)}};
        std::cout<<value.dump()<<'\n'<<std::flush;if(options_.handshake){std::string ack;require(bool(std::getline(std::cin,ack)) && ack=="continue","Parent observation handshake failed");}
    }
private:Options options_;Json identity_;std::size_t sequence_=0;std::mutex mutex_;
};
/** @brief Stable typed creator roles in the diagnostic receipt. */
const char* role(Creator creator){switch(creator){case Creator::PrimaryStream:return "primary_stream";case Creator::PrimaryBLAS:return "primary_blas";case Creator::SecondaryBLAS:return "secondary_blas";case Creator::OrderingEvent:return "ordering_event";}throw std::invalid_argument("Unknown diagnostic creator");}
/** @brief Prove interception reached actual project creators, without inferring missing observations. */
void validatePrefix(const std::vector<Observation>& observed,std::size_t failure){
    require(observed.size()==(failure?failure:35),"Original worker creator population/interposition incomplete");
    for(std::size_t i=0;i<observed.size();++i){const auto expected=i==0?Creator::PrimaryStream:i==1?Creator::PrimaryBLAS:i==2?Creator::SecondaryBLAS:Creator::OrderingEvent;
        require(observed[i].creator==expected && observed[i].index==i+1 && !observed[i].caller.empty(),"Original worker creator order/source changed");
        const bool injected=failure && i+1==failure;require(observed[i].injected==injected && observed[i].success!=injected && (injected?observed[i].native_status!=0:observed[i].native_status==0) && (injected || observed[i].handle),"Original worker creator prefix did not execute");}
}
/** @brief Raw/direct family follows original destruction and TransferEngine physical reset. */
int run(const Options& options){
    Receipt receipt(options);if(!options.lookupProgram.empty())receipt.identity({{"lookup_program_family",NativeAuxiliaryResourceCatalog::resourceSourceIdentity()},{"lookup_receipt_role","raw_existing_ordering_program_no_certificate"}});const auto originalDefaults=pthreadDefaults();receipt.identity({{"pthread_defaults",originalDefaults}});receipt.checkpoint("before_backend_lookup");
    const auto pages=::sysconf(_SC_PHYS_PAGES),available=::sysconf(_SC_AVPHYS_PAGES),page=::sysconf(_SC_PAGESIZE);require(pages>0 && available>0 && page>0,"Original host capacity observation unavailable");
    const std::array resources{PhysicalMemoryResource{0,DeviceId::cpu(),static_cast<std::size_t>(pages)*page,static_cast<std::size_t>(available)*page}};
    auto source=NativeContextBirthDiagnosticAccess::source(resources);
    auto* backend=getBackendFor(options.device);require(backend,"Selected original backend unavailable");receipt.identity(startupIdentity(options.device));receipt.checkpoint("startup_metadata_complete");
    bool existingPoolWorker=false;
    try{(void)GPUDeviceContextPool::instance().retainExistingContextProvenance(options.device);existingPoolWorker=true;}
    catch(const std::logic_error&){} // Existing-only observation must find no selected pool entry.
    require(!existingPoolWorker,"Startup unexpectedly created a selected pool worker");
    std::exception_ptr unexpected;
    std::optional<ExclusiveModelRetirementTicket> ticket;
    for(std::size_t lifetime=0;lifetime<options.lifetimes;++lifetime){
        require(pthreadDefaults()==originalDefaults,"Original pthread default policy changed during diagnostic");
        auto root=NativeContextBirthDiagnosticAccess::root(source,options.device);const auto& program=root->program();
        require(program.stackBytes()==originalDefaults["stack_bytes"].get<std::size_t>() && program.guardBytes()==originalDefaults["guard_bytes"].get<std::size_t>(),"Prepared root differs from observed original pthread defaults");
        receipt.checkpoint("exact_stack_object_root_ready",{{"lifetime",lifetime},{"stack_bytes",program.stackBytes()},{"guard_bytes",program.guardBytes()},{"object_bytes",program.contextObjectBytes()},{"object_alignment",program.contextObjectAlignment()},{"exact_pma_bytes",program.allocationBytes()}});
        std::shared_ptr<IWorkerGPUContext> worker;
        begin(options.fail,[&](const Observation& observed,bool beforeCall){receipt.checkpoint(observed.index>35?(observed.destroyed?"after_lookup_event_destroy":beforeCall?"before_lookup_event_creator":"after_lookup_event_creator"):(observed.destroyed?"after_original_destroy":beforeCall?"before_original_creator":"after_original_creator"),{{"lifetime",lifetime},{"creator",observed.index},{"role",role(observed.creator)},{"caller",observed.caller},{"injected",observed.injected},{"success",observed.success},{"handle",reinterpret_cast<std::uintptr_t>(observed.handle)},{"destroy_success",observed.destroy_success},{"native_status",observed.native_status},{"destroy_status",observed.destroy_status}});});
        try{worker=NativeContextBirthDiagnosticAccess::worker(options.device,root);}
        catch(const std::exception& error){unexpected=std::current_exception();receipt.checkpoint("original_constructor_failed",{{"lifetime",lifetime},{"reason",error.what()}});}
        catch(...){unexpected=std::current_exception();receipt.checkpoint("original_constructor_failed",{{"lifetime",lifetime},{"reason","non-standard original constructor exception"}});}
        const auto observed=snapshot();
        try{
            validatePrefix(observed,options.fail);
            if(options.fail){require(!worker && bool(unexpected),"Injected constructor failure unexpectedly published a worker");unexpected={};receipt.checkpoint("worker_birth_failed_cleanup_complete",{{"lifetime",lifetime},{"creator",options.fail}});}
            else{
                require(worker && !unexpected,"Actual original worker birth failed");require(reinterpret_cast<std::uintptr_t>(worker->defaultStream())>2,"Original worker has no exact primary stream");receipt.checkpoint("selected_worker_ready",{{"lifetime",lifetime},{"pool_generation",0},{"raw_direct_owner",true}});
                for(std::size_t replay=0;replay<options.replays;++replay){worker->submitAndWait([&]{for(const auto& event:observed)if(event.creator==Creator::OrderingEvent){require(worker->recordEventChecked(event.handle,worker->defaultStream()),"Original ordering event record failed");require(worker->waitEventChecked(event.handle,worker->defaultStream()),"Original ordering event wait failed");}require(worker->synchronizeStreamChecked(worker->defaultStream()),"Diagnostic event family did not retire");});receipt.checkpoint(replay?"native_reuse_complete":"native_first_use_complete",{{"lifetime",lifetime},{"replay",replay},{"ordering_events_used",32}});}
                if(!options.lookupProgram.empty()){
                    const auto total=lookupCount(options.lookupProgram),prefix=options.lookupPrefix.value_or(total);
                    std::vector<void*> handles;handles.reserve(prefix);
                    receipt.checkpoint("lookup_program_baseline",{{"lifetime",lifetime},{"original_ordering_events",32},{"program",options.lookupProgram},{"simultaneous_events",total},{"requested_prefix",prefix}});
                    try{
                        for(std::size_t index=0;index<prefix;++index){worker->submitAndWait([&]{auto* event=worker->createEvent(GPUEventPurpose::Ordering);require(reinterpret_cast<std::uintptr_t>(event)>2,"Lookup diagnostic event creator failed");handles.push_back(event);});
                            receipt.checkpoint("lookup_event_created",{{"lifetime",lifetime},{"event",index},{"live_events",handles.size()}});}
                        receipt.checkpoint("lookup_program_prefix_ready",{{"lifetime",lifetime},{"live_events",handles.size()},{"partial_prefix",prefix<total}});
                        for(std::size_t replay=0;replay<options.replays;++replay){worker->submitAndWait([&]{for(auto* event:handles){require(worker->recordEventChecked(event,worker->defaultStream()),"Lookup event record failed");require(worker->waitEventChecked(event,worker->defaultStream()),"Lookup event wait failed");}require(worker->synchronizeStreamChecked(worker->defaultStream()),"Lookup diagnostic event program did not retire");});
                            receipt.checkpoint("lookup_program_use_complete",{{"lifetime",lifetime},{"replay",replay},{"ordering_events_used",handles.size()},{"rerecord",replay!=0}});}
                    }catch(...){worker->submitAndWait([&]{if(!worker->synchronizeStreamChecked(worker->defaultStream()))std::terminate();for(auto it=handles.rbegin();it!=handles.rend();++it)worker->destroyEvent(*it);handles.clear();});throw;}
                    worker->submitAndWait([&]{for(auto it=handles.rbegin();it!=handles.rend();++it)worker->destroyEvent(*it);handles.clear();});
                    receipt.checkpoint("lookup_program_handles_destroyed",{{"lifetime",lifetime},{"destroyed_events",prefix},{"physical_release_claimed",false}});
                }
            }
        }catch(const std::exception& error){
            receipt.checkpoint("diagnostic_validation_failed",{{"lifetime",lifetime},{"reason",error.what()},{"primary_constructor_failure_retained",bool(unexpected)}});
            if(!unexpected)unexpected=std::current_exception();
        }catch(...){
            receipt.checkpoint("diagnostic_validation_failed",{{"lifetime",lifetime},{"reason","non-standard diagnostic validation exception"},{"primary_constructor_failure_retained",bool(unexpected)}});
            if(!unexpected)unexpected=std::current_exception();
        }
        if(options.resetPolicy==RuntimeResetPolicy::EachLifetime || lifetime+1==options.lifetimes || unexpected)ticket.emplace(TransferEngine::instance().beginExclusiveNativeContextRetirement(options.device));
        worker.reset();const auto destroyed=finish();std::size_t createdCount=0,destroyedCount=0;
        for(const auto& value:destroyed)if(value.success){++createdCount;if(value.destroyed && value.destroy_success)++destroyedCount;}
        if(createdCount!=destroyedCount && !unexpected)unexpected=std::make_exception_ptr(std::runtime_error("Original worker rollback/destruction omitted observed native owners"));
        if(unexpected && !ticket)ticket.emplace(TransferEngine::instance().beginExclusiveNativeContextRetirement(options.device));
        receipt.checkpoint("original_worker_destroyed",{{"lifetime",lifetime},{"failed",bool(unexpected)},{"created_handles",createdCount},{"destroyed_handles",destroyedCount},{"rollback_complete",createdCount==destroyedCount}});
        root.reset();receipt.checkpoint("exact_stack_object_root_released",{{"lifetime",lifetime}});
        if(ticket){
            const auto retired=TransferEngine::instance().completeExclusiveModelRetirement(std::move(*ticket));
            ticket.reset();
            require(retired.runtime_reset_invoked && retired.retired_context_generation==0 && retired.retired_runtime_generation && retired.successor_runtime_generation && retired.runtime_post_reset_state==DeviceRuntimePostResetState::Quiescent && retired.canonical_allocations_before_runtime_reset==0 && retired.canonical_allocation_bytes_before_runtime_reset==0,"Raw direct worker physical reset receipt incomplete");
            // Retain this one original Preparing source across all requested births.
            // No SDK/device query follows reset: the parent observes the quiescent
            // boundary before acknowledging the next explicitly requested birth.
            receipt.checkpoint("exclusive_context_retirement_complete",{{"lifetime",lifetime},{"reset_invoked",retired.runtime_reset_invoked},{"retired_context_generation",retired.retired_context_generation},{"raw_direct_owner",true},{"retired_runtime_generation",retired.retired_runtime_generation},{"successor_runtime_generation",retired.successor_runtime_generation},{"runtime_post_reset_state",to_string(retired.runtime_post_reset_state)},{"canonical_allocations_before_runtime_reset",retired.canonical_allocations_before_runtime_reset},{"canonical_bytes_before_runtime_reset",retired.canonical_allocation_bytes_before_runtime_reset},{"injected_failure",bool(options.fail)},{"diagnostic_failed",bool(unexpected)}});
        }
        if(unexpected)break;
    }
    // CPU-only source cancellation neither probes nor initializes a successor.
    source->cancel();source.reset();
    if(unexpected)std::rethrow_exception(unexpected);return 0;
}
} // namespace
} // namespace llaminar2::test
/** @brief Explicit diagnostic entrypoint owns original native process lifetime. */
int main(int argc,char** argv){llaminar2::GPUDeviceProcessLifetime lifetime;try{return llaminar2::test::run(llaminar2::test::parse(argc,argv));}catch(const std::exception& error){std::cerr<<"NATIVE_CONTEXT_BIRTH_CALIBRATION_ERROR: "<<error.what()<<'\n';return 1;}}
