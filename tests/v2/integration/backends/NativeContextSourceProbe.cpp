/**
 * @file NativeContextSourceProbe.cpp
 * @brief Uninstrumented original production key and same-PMA source-lifetime integration probe.
 *
 * KeyOnly never opens a source or invokes a catalog. SourceLifecycle uses only
 * original public creation/admission APIs and installed reviewed limits. No
 * SDK interposer, numeric override, raw constructor or registration is linked.
 * Parent observation owns process-exit physical-return evidence.
 */
#include "NativeContextSourceProbeSupport.h"
#include "NativeContextSourceProbePma.h"
#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include "transfer/TransferEngine.h"
#include <array>
#include <chrono>
#include <iostream>
#include <malloc.h>
#include <mutex>
#include <optional>
#include <set>
#include <sys/resource.h>
#include <unistd.h>
namespace llaminar2::test {
namespace {
using Json=SourceProbeJson;
/** @brief Separate explicit roles prevent metadata projection from claiming source qualification. */
enum class Role {KeyOnly,SourceLifecycle};
struct Options {DeviceId device=DeviceId::invalid();std::optional<Role> role;bool handshake=false;std::size_t replays=2;};
/** @brief Reject ambiguous mode/device input before any native metadata startup. */
Options parse(int argc,char** argv){Options value;for(int i=1;i<argc;++i){const std::string key=argv[i];
    if(key=="--handshake"){value.handshake=true;continue;}
    if(key=="--key-only" || key=="--source-lifecycle"){if(value.role)throw std::invalid_argument("Choose exactly one original source probe role");value.role=key=="--key-only"?Role::KeyOnly:Role::SourceLifecycle;continue;}
    if(i+1>=argc)throw std::invalid_argument("Missing original source probe argument");const std::string argument=argv[++i];
    if(key=="--device"){const auto colon=argument.find(':');if(colon==std::string::npos)throw std::invalid_argument("Native source device requires cuda:N or rocm:N");
        std::size_t used=0;const int ordinal=std::stoi(argument.substr(colon+1),&used);if(ordinal<0 || used!=argument.size()-colon-1)throw std::invalid_argument("Invalid native source ordinal");
        if(argument.substr(0,colon)=="cuda")value.device=DeviceId::cuda(ordinal);else if(argument.substr(0,colon)=="rocm")value.device=DeviceId::rocm(ordinal);else throw std::invalid_argument("Unknown original source backend");
    }else if(key=="--replays"){std::size_t used=0;value.replays=std::stoull(argument,&used);if(used!=argument.size() || !value.replays || value.replays>1024)throw std::invalid_argument("Invalid event-family replay count");}
    else throw std::invalid_argument("Unknown original source probe option");}
    if(!value.device.is_gpu() || !value.role)throw std::invalid_argument("Original source probe requires selected device and explicit role");return value;
}
/** @brief Fail at the original operation boundary rather than falling through to another role. */
void require(bool value,const char* reason){if(!value)throw std::runtime_error(reason);}
/** @brief Source-correlated host receipt; no GPU query occurs when emitting a checkpoint. */
class Receipt final {
public:
    explicit Receipt(Options value):options_(value){closure_={{"device",value.device.toString()},{"projection_role","uninstrumented_production_birth_key"},
        {"resource_source_sha256",NATIVE_CONTEXT_BIRTH_SOURCE_SHA256},{"diagnostic_source_sha256",NATIVE_CONTEXT_SOURCE_PROBE_SHA256},{"compiler",__VERSION__}};}
    /** @brief Add immutable actual metadata before any source worker is born. */
    void metadata(Json value){for(auto i=value.begin();i!=value.end();++i)closure_[i.key()]=i.value();}
    /** @brief Publish exact observed host/owner facts and wait only at this diagnostic boundary. */
    void checkpoint(const char* stage,Json extra=Json::object()){
        const auto heap=::mallinfo2();rusage usage{};require(::getrusage(RUSAGE_SELF,&usage)==0,"Original source host observation failed");
        Json value={{"schema","llaminar.native_context_source_probe.v1"},{"pid",::getpid()},{"sequence",sequence_++},{"stage",stage},{"closure",closure_},
            {"geometry",{{"role",*options_.role==Role::KeyOnly?"key_only":"source_lifecycle"},{"selected_workers",1},{"replays",*options_.role==Role::KeyOnly?0:options_.replays}}},
            {"production_admission",*options_.role==Role::SourceLifecycle && admitted_},{"numerical_inference_executed",false},
            {"host",{{"maxrss_kib",usage.ru_maxrss},{"mallinfo_arena",heap.arena},{"mallinfo_hblkhd",heap.hblkhd},{"mallinfo_uordblks",heap.uordblks}}},{"extra",std::move(extra)}};
        std::cout<<value.dump()<<'\n'<<std::flush;if(options_.handshake){std::string ack;require(bool(std::getline(std::cin,ack)) && ack=="continue","Original source parent acknowledgment failed");}
    }
    /** @brief Mark only a successful actual same-state admission. */
    void admitted(){admitted_=true;}
private:Options options_;Json closure_;std::size_t sequence_=0;bool admitted_=false;
};
/** @return Actual original PMA state; all numbers are ledger observations. */
Json pma(const std::shared_ptr<PhysicalMemoryAuthority>& memory,DeviceId device){return nativeSourcePmaProjection(memory,device);}
/** @brief Verify both SDK providers actually selected by the uninstrumented production key. */
std::vector<NativeSourceSDKBinding> bindings(DeviceId device){
#ifdef HAVE_CUDA
    if(device.is_cuda())return nativeSourceCUDABindings();
#endif
#ifdef HAVE_ROCM
    if(device.is_rocm())return nativeSourceROCmBindings();
#endif
    throw std::invalid_argument("Selected source backend was not compiled");
}
/** @brief Obtain already-owned original32events only inside the retained worker submission. */
std::vector<void*> originalEvents(const PreparedGPUContextUsePermit& permit){
#ifdef HAVE_CUDA
    if(permit.device().is_cuda())return NativeContextSourceLifecycleDiagnosticAccess::cudaEvents(permit);
#endif
#ifdef HAVE_ROCM
    if(permit.device().is_rocm())return NativeContextSourceLifecycleDiagnosticAccess::rocmEvents(permit);
#endif
    throw std::invalid_argument("Selected source event backend was not compiled");
}
/** @brief Execute one explicit metadata-only or actual original source lifecycle. */
int run(const Options& options){
    Receipt receipt(options);receipt.checkpoint("before_backend_lookup");auto* backend=getBackendFor(options.device);require(backend,"Selected original backend unavailable");
    const auto key=PreparedGPUContextFactory::birthKey(options.device);const auto providers=bindings(options.device);
    require(providers.size()==2,"Original source requires exact native BLAS and Lt provider facts");
    const auto combined=sha256BytesHex(providers[0].sha256+":"+providers[1].sha256);require(combined && *combined==key.backend_sha256,"Public production BirthKey differs from exact configured native SDK symbols");
    Json sdk=Json::array();for(const auto& value:providers)sdk.push_back({{"symbol",value.symbol},{"path",value.path},{"sha256",value.sha256},{"expected_path",value.expected_path},{"matches_configured_provider",true},{"provider_is_executable",false}});
    bool existingWorker=false;
    try{(void)GPUDeviceContextPool::instance().retainExistingContextProvenance(options.device);existingWorker=true;}catch(const std::logic_error&){}
    require(!existingWorker,"Source metadata startup unexpectedly created an original pool worker");
    receipt.metadata({{"birth_key",nativeSourceBirthKeyJson(key)},{"native_blas_providers",sdk},{"creator_symbols_interposed",false},{"existing_pool_worker",false}});
    receipt.checkpoint("startup_metadata_complete");
    if(*options.role==Role::KeyOnly){receipt.checkpoint("production_key_complete",{{"source_created",false},{"catalog_invoked",false},{"pool_worker_created",false}});return 0;}
    const auto pages=::sysconf(_SC_PHYS_PAGES),available=::sysconf(_SC_AVPHYS_PAGES),page=::sysconf(_SC_PAGESIZE);require(pages>0 && available>0 && page>0,"Original host observation unavailable");
    // Setup-only capacity observation precedes selected worker construction.
    const auto total=backend->deviceMemoryTotal(options.device.ordinal),free=backend->deviceMemoryFree(options.device.ordinal);
    require(total && free && free<=total,"Original preworker GPU capacity observation is incomplete");
    const std::array observations{PhysicalMemoryResource{0,DeviceId::cpu(),static_cast<std::size_t>(pages)*page,static_cast<std::size_t>(available)*page},
        PhysicalMemoryResource{0,options.device,total,free}};
    receipt.checkpoint("preworker_observations_ready",{{"device_total_bytes",total},{"device_free_bytes",free},{"cpu_total_bytes",observations.front().total_bytes},{"cpu_free_bytes",observations.front().admission_available_bytes}});
    std::shared_ptr<NativeContextPreparation> source;std::optional<PreparedGPUContextUsePermit> permit;bool retired=false;
    try{
        const std::array selected{options.device};source=NativeContextPreparation::create(observations,0,selected);const auto memory=source->memory();
        const auto contract=NativeContextOpaqueCatalog::require(key); // Read SAME closed source data, never caller limits.
        require(memory->phase()==PhysicalMemoryAuthorityPhase::Preparing,"Original source skipped Preparing authority");
        const auto prepared=nativeSourcePmaProjection(memory,options.device);
        require(prepared["host_opaque_bytes"].get<std::size_t>()==contract.hostUpperBytes() && prepared["device_opaque_bytes"].get<std::size_t>()==contract.deviceUpperBytes(),"Original source changed its closed family commitment");
        receipt.checkpoint("source_prepared",{{"pma",pma(memory,options.device)},{"contract_host_upper",contract.hostUpperBytes()},{"contract_device_upper",contract.deviceUpperBytes()},
            {"physical_lifetime",contract.lifetime()==NativePhysicalLifetime::ProcessRuntime?"ProcessRuntime":"ContextGeneration"}});
        permit.emplace(source->acquire(options.device));const auto generation=permit->generation();
        {const auto original=GPUDeviceContextPool::instance().retainExistingPreparedContext(options.device,memory);original.requireWorker(permit->worker());
            require(original.generation()==generation && permit->memory()==memory,"Original source worker/controlblock/generation changed");
            const auto originalKey=original.provenance().originalBirthKey();
            const auto repeatedKey=permit->provenance().originalBirthKey();
            require(originalKey && repeatedKey && *originalKey==key && originalKey.get()==repeatedKey.get() &&
                !originalKey.owner_before(repeatedKey) && !repeatedKey.owner_before(originalKey),
                "Prepared worker did not retain its exact original immutable birth key");}
        receipt.checkpoint("original_worker_ready",{{"generation",generation},{"exact_original_worker",true},{"original_key_alias_verified",true},{"pma",pma(memory,options.device)}});
        source->freezeForPlanning();auto certificate=std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(source->mandatoryBOM());source->admitFinal(certificate);
        require(source->memory()==memory && memory->admission()==certificate,"Source final admission replaced original PMA or certificate");memory->requireRankCommitted();receipt.admitted();
        receipt.checkpoint("same_pma_admitted",{{"same_authority_controlblock",true},{"same_certificate",true},{"pma",pma(memory,options.device)}});
        for(std::size_t replay=0;replay<options.replays;++replay){permit->worker()->submitAndWait([&]{
            auto events=originalEvents(*permit);std::set<void*> unique(events.begin(),events.end());require(unique.size()==32,"Original event family contains aliased handles");
            auto* stream=permit->worker()->defaultStream();require(reinterpret_cast<std::uintptr_t>(stream)>2,"Original source primary stream is null/default");
            for(auto* event:events){require(permit->worker()->recordEventChecked(event,stream),"Original event record failed");require(permit->worker()->waitEventChecked(event,stream),"Original event wait failed");}
            require(permit->worker()->synchronizeStreamChecked(stream),"Explicit integration event-family boundary failed");
        });receipt.checkpoint("original_event_family_used",{{"replay",replay},{"ordering_events",32},{"same_retained_worker",true},{"exact_primary_stream",true}});}
        permit.reset();receipt.checkpoint("original_permit_released");
        const auto result=TransferEngine::instance().completeExclusiveModelRetirement(TransferEngine::instance().beginExclusiveNativeContextRetirement(options.device));retired=true;
        require(result.runtime_reset_invoked && result.retired_context_generation==generation && result.runtime_post_reset_state==DeviceRuntimePostResetState::Quiescent && result.canonical_allocations_before_runtime_reset==0 && result.canonical_allocation_bytes_before_runtime_reset==0,"Original source native-only retirement receipt incomplete");
        const auto remaining=pma(memory,options.device);require(remaining["exact_preparation_bytes"]==0,"Original mapped stack/object did not retire");
        const auto expected=contract.lifetime()==NativePhysicalLifetime::ProcessRuntime?contract.hostUpperBytes()+contract.deviceUpperBytes():0;
        require(remaining["opaque_committed_bytes"]==expected,"Original opaque obligation disagrees with its closed physical lifetime");
        receipt.checkpoint("original_source_retired",{{"generation",generation},{"reset_invoked",true},{"runtime_post_reset_state","quiescent"},{"retired_context_generation",result.retired_context_generation},
            {"retired_runtime_generation",result.retired_runtime_generation},{"successor_runtime_generation",result.successor_runtime_generation},{"canonical_allocations_before_runtime_reset",result.canonical_allocations_before_runtime_reset},
            {"canonical_bytes_before_runtime_reset",result.canonical_allocation_bytes_before_runtime_reset},{"physical_lifetime",contract.lifetime()==NativePhysicalLifetime::ProcessRuntime?"ProcessRuntime":"ContextGeneration"},{"pma",remaining}});
    }catch(...){
        const auto primary=std::current_exception();std::string reason="non-standard original source failure";
        try{std::rethrow_exception(primary);}catch(const std::exception& error){reason=error.what();}catch(...){}
        receipt.checkpoint(source?"source_lifecycle_failed":"source_admission_rejected",{{"reason",reason},{"source_created",bool(source)},{"original_worker_ready",bool(permit)}});
        permit.reset();if(source && !retired){
            if(source->memory()->phase()==PhysicalMemoryAuthorityPhase::Preparing)source->retireUnadmitted();
            else {const auto memory=source->memory();{const auto exact=GPUDeviceContextPool::instance().retainExistingPreparedContext(options.device,memory);require(exact.memory()==memory,"Failed source cleanup cannot adopt foreign worker");}
                (void)TransferEngine::instance().completeExclusiveModelRetirement(TransferEngine::instance().beginExclusiveNativeContextRetirement(options.device));}}
        std::rethrow_exception(primary);}
    return 0;
}
}
}
/** @brief Explicit native test entry owns original process lifetime through all source cleanup. */
int main(int argc,char** argv){llaminar2::GPUDeviceProcessLifetime lifetime;try{return llaminar2::test::run(llaminar2::test::parse(argc,argv));}
    catch(const std::exception& error){std::cerr<<"NATIVE_CONTEXT_SOURCE_PROBE_ERROR: "<<error.what()<<'\n';return 1;}}
