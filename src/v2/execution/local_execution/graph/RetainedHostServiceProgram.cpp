/**
 * @file RetainedHostServiceProgram.cpp
 * @brief Admit exact CPU thread roots independently of process-retained libc/TLS backing.
 *
 * Attribute snapshots preserve glibc process defaults (including affinity and
 * signal-mask overrides). Only the supplied stack address/guard binding changes.
 * Source/TLS snapshots are cold metadata, never an inference-time observation.
 */
#include "RetainedHostServiceProgram.h"
#include "RetainedHostServiceDeclarations.h"
#include "RetainedParentTicketServiceWorker.h"
#include "RetainedHostServiceQualifiedLimits.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include "utils/Sha256.h"
#include "utils/DebugEnv.h"
#include "utils/ScopedOpenMPPartitionAffinity.h"
#include <sys/mman.h>
#include <unistd.h>
#include <signal.h>
#include <sched.h>
#include <link.h>
#include <dlfcn.h>
#include <filesystem>
#include <limits>
#include <mutex>
#include <sstream>
#include <algorithm>
#include <exception>
#include <stdexcept>
#include <system_error>
#include <vector>
#include <array>
#include <sys/stat.h>
#include <cstdlib>
#include <map>
#include <chrono>
#include <atomic>
#include <gnu/libc-version.h>
// GNU TLS resolver is an original runtime dependency; never search past an interposed selection.
extern "C" void* __tls_get_addr(void*);
namespace llaminar2 {
namespace {
/** @return Checked page rounding without introducing an anonymous reserve. */
std::size_t pages(std::size_t bytes,std::size_t page){
    if(!bytes || !page || bytes>std::numeric_limits<std::size_t>::max()-(page-1))throw std::overflow_error("CPU service page extent overflow");
    return ((bytes+page-1)/page)*page;
}
/** @return Mandatory identity; unavailable file evidence is a rejection. */
std::string digest(const std::string& value){auto d=sha256BytesHex(value);if(!d)throw std::runtime_error("CPU service identity hash failed");return *d;}
/** @return Stable supported creator/role family; exact live declaration stays separate. */
std::string familyIdentity(RetainedHostServiceRole role){
    switch(role){case RetainedHostServiceRole::ImmutableLookupIO:case RetainedHostServiceRole::ExecutorParentTicket:break;default:throw std::invalid_argument("Unknown CPU resource role");}
    return digest("mapped-pthread-linux-glibc2-merged-64-v1:"+std::to_string(static_cast<unsigned>(role)));
}
/** @brief Immutable loaded-file facts; comparison never hashes or dereferences an unloaded image. */
struct NativeFileFact {
    std::string path;std::uint64_t device,inode,bytes;std::int64_t modifiedSeconds,modifiedNanos,changedSeconds,changedNanos;
    bool operator==(const NativeFileFact&)const=default;
};
/** @return Original immutable provider identity, with no native resource creator. */
NativeFileFact fileFact(const std::string& path){
    const auto canonical=std::filesystem::canonical(path).string();struct stat value{};
    if(::stat(canonical.c_str(),&value))throw std::system_error(errno,std::generic_category(),"CPU service original ELF file facts");
    return {canonical,static_cast<std::uint64_t>(value.st_dev),static_cast<std::uint64_t>(value.st_ino),static_cast<std::uint64_t>(value.st_size),value.st_mtim.tv_sec,value.st_mtim.tv_nsec,value.st_ctim.tv_sec,value.st_ctim.tv_nsec};
}
/** @return Provider selected by the actual creator symbol; no resolver substitution is permitted. */
NativeFileFact nativeProviderFact(const void* selected,const char* role){
    Dl_info info{};if(!selected||!dladdr(selected,&info)||!info.dli_fname)throw std::runtime_error(std::string("CPU service has no original ")+role+" provider");return fileFact(info.dli_fname);
}
NativeFileFact pthreadProviderFact(){return nativeProviderFact(reinterpret_cast<const void*>(&::pthread_create),"pthread");}
NativeFileFact cppProviderFact(){return nativeProviderFact(reinterpret_cast<const void*>(&std::terminate),"C++ runtime");}
NativeFileFact openMPProviderFact(){return nativeProviderFact(reinterpret_cast<const void*>(&::omp_get_proc_bind),"OpenMP runtime");}
/** @brief Loaded TLS layout uses only copied ELF file geometry, never relocated initializer pointers. */
struct TLSImageFact {NativeFileFact file;std::size_t fileOffset,fileBytes,memoryBytes,alignment,moduleId;bool ownSource;bool operator==(const TLSImageFact&)const=default;};
/** @return One loader-owned snapshot, with all file operations outside its callback. */
struct LoaderFacts {std::vector<TLSImageFact> tls;std::vector<NativeFileFact> modules;std::uint64_t adds=0,subs=0;bool operator==(const LoaderFacts&)const=default;};
LoaderFacts loadedFacts(){
    struct Image{std::string path;std::size_t offset=0,fileBytes=0,memoryBytes=0,alignment=0,moduleId=0;bool ownSource=false,hasTLS=false;};
    struct Snapshot{std::vector<Image> images;std::exception_ptr failure;std::uint64_t adds=0,subs=0;bool counters=false;};Snapshot snapshot;
    const auto copy=[](dl_phdr_info* info,std::size_t infoBytes,void* opaque)->int{
        auto& s=*static_cast<Snapshot*>(opaque);try{
            if(infoBytes<offsetof(dl_phdr_info,dlpi_tls_modid)+sizeof(info->dlpi_tls_modid))
                throw std::runtime_error("CPU service loader lacks authenticated lifecycle/TLS facts");
            if(s.counters&&(s.adds!=info->dlpi_adds||s.subs!=info->dlpi_subs))throw std::logic_error("CPU service loader snapshot changed within traversal");
            s.counters=true;s.adds=info->dlpi_adds;s.subs=info->dlpi_subs;
            bool own=false;const auto address=reinterpret_cast<std::uintptr_t>(&RetainedHostServiceProgram::originalEvidence);
            for(std::size_t j=0;j<info->dlpi_phnum;++j){const auto& load=info->dlpi_phdr[j];if(load.p_type==PT_LOAD&&address>=info->dlpi_addr+load.p_vaddr&&address-(info->dlpi_addr+load.p_vaddr)<load.p_memsz)own=true;}
            Image image;image.path=info->dlpi_name&&*info->dlpi_name?info->dlpi_name:"/proc/self/exe";image.ownSource=own;
            for(std::size_t i=0;i<info->dlpi_phnum;++i)if(info->dlpi_phdr[i].p_type==PT_TLS){const auto& ph=info->dlpi_phdr[i];
                image.offset=ph.p_offset;image.fileBytes=ph.p_filesz;image.memoryBytes=ph.p_memsz;image.alignment=ph.p_align;image.moduleId=info->dlpi_tls_modid;image.hasTLS=true;}
            s.images.push_back(std::move(image));return 0;
        }catch(...){s.failure=std::current_exception();return 1;}};
    dl_iterate_phdr(copy,&snapshot);if(snapshot.failure)std::rethrow_exception(snapshot.failure);
    LoaderFacts result;result.adds=snapshot.adds;result.subs=snapshot.subs;for(const auto& image:snapshot.images){
        const auto file=image.path.starts_with("linux-vdso")?NativeFileFact{image.path,0,0,0,0,0,0,0}:fileFact(image.path);
        result.modules.push_back(file);if(image.hasTLS)result.tls.push_back({file,image.offset,image.fileBytes,image.memoryBytes,image.alignment,image.moduleId,image.ownSource});
    }
    std::sort(result.tls.begin(),result.tls.end(),[](const auto& a,const auto& b){return std::tie(a.file.path,a.fileOffset)<std::tie(b.file.path,b.fileOffset);});
    std::sort(result.modules.begin(),result.modules.end(),[](const auto& a,const auto& b){return a.path<b.path;});return result;
}
/** @return Authenticated immutable bytes, rejecting replacement during the one cold read. */
std::atomic<std::uint64_t> native_file_hash_attempts{0};
std::string originalFileHash(const NativeFileFact& fact){native_file_hash_attempts.fetch_add(1,std::memory_order_relaxed);const auto hash=sha256FileHex(fact.path);if(!hash||fileFact(fact.path)!=fact)throw std::runtime_error("CPU service original ELF changed while evidence was read");return *hash;}
/** @return Exact live loader/default policy inputs, separate from portable resource policy. */
std::vector<std::pair<std::string,std::string>> runtimePolicy(){
    std::vector<std::pair<std::string,std::string>> result;
    for(const char* name:{"GLIBC_TUNABLES","MALLOC_ARENA_MAX","MALLOC_ARENA_TEST","MALLOC_MMAP_THRESHOLD_","MALLOC_TRIM_THRESHOLD_","MALLOC_MMAP_MAX_","LD_PRELOAD","LD_LIBRARY_PATH","LD_AUDIT"})
    {
        // Keep the original closed-key order and absent/empty equivalence;
        // DebugEnv is the sole engine authority for reading process inputs.
        const char* value=DebugEnv::envValue(name);
        result.emplace_back(name,value?value:"");
    }
    return result;
}
}
/** @brief Immutable actual loader provider; its code owns DTV/TLS allocation even without PT_TLS itself. */
class RetainedHostServiceProgram::TLSProviderEvidence final {
public:
    TLSProviderEvidence(NativeFileFact file,std::string digest):file_(std::move(file)),digest_(std::move(digest)){}
    /** @brief Revalidate the selected provider without rehashing its immutable file. */
    void requireCurrent(const void* selected)const{
        if(nativeProviderFact(selected,"loader/TLS resolver")!=file_)throw std::logic_error("CPU service original loader/TLS resolver provider changed after declaration");
    }
    const std::string& digest()const noexcept{return digest_;}
private:
    NativeFileFact file_;
    std::string digest_;
};
std::shared_ptr<const RetainedHostServiceProgram::TLSProviderEvidence> RetainedHostServiceProgram::originalTLSProviderEvidence(){
    const auto fact=nativeProviderFact(reinterpret_cast<const void*>(&::__tls_get_addr),"loader/TLS resolver");
    return std::make_shared<const TLSProviderEvidence>(fact,originalFileHash(fact));
}
void RetainedHostServiceProgram::requireTLSProviderEvidence(const std::shared_ptr<const TLSProviderEvidence>& evidence,const void* selected){
    if(!evidence)throw std::logic_error("CPU service loader/TLS provider evidence is missing");evidence->requireCurrent(selected);
}
/** @brief One immutable declaration-session snapshot; liveness comparisons never rehash providers. */
class RetainedHostServiceProgram::Evidence final {
    friend class RetainedHostServiceProgram;
    LoaderFacts loader;
    NativeFileFact provider{},cppProvider{},openMPProvider{};
    std::vector<std::pair<std::string,std::string>> policy;
    std::string resourcePolicy,liveLoader;
    std::size_t tlsModules=0,tlsBytes=0;
    std::shared_ptr<const TLSProviderEvidence> tlsResolver;
    std::uint64_t hashBytes=0,hashFiles=0,elapsedNs=0;
    bool recorded=false;
    /** @brief Reject module/file/default-environment mutation before any admitted reservation/creator. */
    void requireCurrent()const{
        if(recorded)return;
        if(!tlsResolver)throw std::logic_error("CPU service loader/TLS resolver evidence is missing");
        tlsResolver->requireCurrent(reinterpret_cast<const void*>(&::__tls_get_addr));
        if(loadedFacts()!=loader||pthreadProviderFact()!=provider||cppProviderFact()!=cppProvider||openMPProviderFact()!=openMPProvider||runtimePolicy()!=policy)
            throw std::logic_error("CPU service original loaded TLS/provider/runtime policy changed after declaration");
    }
};
std::shared_ptr<const RetainedHostServiceProgram::Evidence> RetainedHostServiceProgram::originalEvidence(){
    const auto start=std::chrono::steady_clock::now();
    auto evidence=std::shared_ptr<Evidence>(new Evidence);evidence->loader=loadedFacts();evidence->provider=pthreadProviderFact();evidence->cppProvider=cppProviderFact();evidence->openMPProvider=openMPProviderFact();evidence->policy=runtimePolicy();
    const auto resolver=nativeProviderFact(reinterpret_cast<const void*>(&::__tls_get_addr),"loader/TLS resolver");
    evidence->tlsResolver=std::make_shared<const TLSProviderEvidence>(resolver,std::string{});
    // Passive modules affect only the reviewed eligibility and this session's
    // cheap liveness facts. Their incidental code/payload bytes are not a new
    // portable creator class or another full-library hashing campaign.
    evidence->tlsModules=evidence->loader.tls.size();
    for(const auto& image:evidence->loader.tls){
        if(image.memoryBytes>std::numeric_limits<std::size_t>::max()-evidence->tlsBytes)throw std::overflow_error("CPU service TLS eligibility extent overflow");
        evidence->tlsBytes+=image.memoryBytes;
    }
    std::ostringstream policy;for(const auto& [name,value]:evidence->policy)if(name!="LD_LIBRARY_PATH")policy<<name.size()<<':'<<name<<value.size()<<':'<<value;
    evidence->resourcePolicy=digest(policy.str());
    std::ostringstream live;live<<evidence->loader.adds<<':'<<evidence->loader.subs<<':';
    for(const auto& image:evidence->loader.tls)live<<image.file.path.size()<<':'<<image.file.path<<':'<<image.moduleId<<':';
    evidence->liveLoader=digest(live.str());evidence->requireCurrent();
    evidence->elapsedNs=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();return evidence;
}

RetainedHostServiceProgram::EvidenceObservation RetainedHostServiceProgram::evidenceObservation(const std::shared_ptr<const Evidence>& evidence){
    if(!evidence)throw std::invalid_argument("CPU service evidence observation requires its original session");
    return {evidence->hashFiles,evidence->hashBytes,evidence->elapsedNs};
}
std::uint64_t RetainedHostServiceProgram::nativeEvidenceHashAttempts()noexcept{return native_file_hash_attempts.load(std::memory_order_relaxed);}

/** @brief Deep original defaults retained until all declarations/instances retire. */
struct RetainedHostServiceProgram::State {
    pthread_attr_t attributes{};
    mutable std::mutex mutex;
    RetainedHostServiceRole role;
    RetainedHostServiceResourceDeclaration declaration;
    std::string identity,resourceIdentity;
    bool qualified=false;
    MetadataAuthority origin=MetadataAuthority::Installed;
    std::shared_ptr<const Evidence> evidence;
    bool hasAttributes=false;
    explicit State(RetainedHostServiceRole r,MetadataAuthority from):role(r),origin(from){
        if(from!=MetadataAuthority::InstalledWire){
            const auto result=::pthread_getattr_default_np(&attributes);if(result)throw std::system_error(result,std::generic_category(),"CPU service original pthread defaults");
            hasAttributes=true;
        }
    }
    ~State(){if(hasAttributes&&::pthread_attr_destroy(&attributes))std::terminate();}
};
void RetainedHostServiceResourceDeclaration::requireValid()const{
    if(!pageBytes||(pageBytes&(pageBytes-1))||stackBytes<PTHREAD_STACK_MIN||stackBytes%pageBytes||guardBytes%pageBytes||
       !objectBytes||!objectAlignment||(objectAlignment&(objectAlignment-1))||objectAlignment>pageBytes)
        throw std::invalid_argument("CPU service declaration has invalid exact mapping geometry");
    const auto objectPages=pages(objectBytes,pageBytes);
    if(stackBytes>std::numeric_limits<std::size_t>::max()-objectPages||stackBytes+objectPages>std::numeric_limits<std::size_t>::max()-guardBytes)
        throw std::overflow_error("CPU exact mapping extent overflow");
    if(opaqueBytes!=detail::plannedRetainedHostServiceOpaqueBytes(tlsBytes))throw std::invalid_argument("CPU wire budget differs from declared TLS policy");
}
RetainedHostServiceProgram::RetainedHostServiceProgram(std::shared_ptr<const State> s):state_(std::move(s)){}
RetainedHostServiceProgram RetainedHostServiceProgram::original(RetainedHostServiceRole role){return original(role,originalEvidence());}
RetainedHostServiceProgram RetainedHostServiceProgram::original(RetainedHostServiceRole role,std::shared_ptr<const Evidence> evidence){
    if(!evidence)throw std::invalid_argument("CPU service original session evidence is missing");return snapshot(role,MetadataAuthority::Installed,std::move(evidence));
}
RetainedHostServiceProgram RetainedHostServiceProgram::snapshot(RetainedHostServiceRole role,MetadataAuthority origin,std::shared_ptr<const Evidence> evidence){
    switch(role){case RetainedHostServiceRole::ImmutableLookupIO:case RetainedHostServiceRole::ExecutorParentTicket:break;default:throw std::invalid_argument("Unknown retained CPU service role");}
    if(evidence)evidence->requireCurrent();
    detail::requireRetainedHostServiceABI(::gnu_get_libc_version(),sizeof(void*));
    auto s=std::make_shared<State>(role,origin);s->evidence=std::move(evidence);const auto page=::sysconf(_SC_PAGESIZE);if(page<=0)throw std::runtime_error("CPU service page geometry unavailable");s->declaration.pageBytes=page;
    int detach=0,inherit=0,policy=0;struct sched_param parameter{};cpu_set_t affinity{};sigset_t signalMask{};
    if(pthread_attr_getstacksize(&s->attributes,&s->declaration.stackBytes)||pthread_attr_getguardsize(&s->attributes,&s->declaration.guardBytes)||
       pthread_attr_getdetachstate(&s->attributes,&detach)||pthread_attr_getinheritsched(&s->attributes,&inherit)||
       pthread_attr_getschedpolicy(&s->attributes,&policy)||pthread_attr_getschedparam(&s->attributes,&parameter)||
       pthread_attr_getaffinity_np(&s->attributes,sizeof(affinity),&affinity))throw std::runtime_error("CPU service original pthread policy incomplete");
    const auto signals=pthread_attr_getsigmask_np(&s->attributes,&signalMask);
    if(signals!=0 && signals!=PTHREAD_ATTR_NO_SIGMASK_NP)throw std::runtime_error("CPU service signal policy unavailable");
    if(detach!=PTHREAD_CREATE_JOINABLE||s->declaration.stackBytes<PTHREAD_STACK_MIN||s->declaration.stackBytes%s->declaration.pageBytes||s->declaration.guardBytes%s->declaration.pageBytes)throw std::invalid_argument("CPU service requires exact joinable original stack policy");
    s->declaration.objectBytes=sizeof(RetainedParentTicketServiceWorker);s->declaration.objectAlignment=alignof(RetainedParentTicketServiceWorker);
    s->declaration.tlsBytes=s->evidence->tlsBytes;s->declaration.opaqueBytes=detail::plannedRetainedHostServiceOpaqueBytes(s->declaration.tlsBytes);
    if(s->declaration.stackBytes>std::numeric_limits<std::size_t>::max()-pages(s->declaration.objectBytes,s->declaration.pageBytes) || s->declaration.stackBytes+pages(s->declaration.objectBytes,s->declaration.pageBytes)>std::numeric_limits<std::size_t>::max()-s->declaration.guardBytes)throw std::overflow_error("CPU service mapped extent overflow");
    s->resourceIdentity=familyIdentity(role);
    std::ostringstream identity;identity<<s->resourceIdentity<<':'<<s->evidence->resourcePolicy<<':'<<s->evidence->liveLoader<<':'
        <<s->declaration.stackBytes<<':'<<s->declaration.guardBytes<<':'<<s->declaration.pageBytes<<':'
        <<s->declaration.objectBytes<<':'<<s->declaration.objectAlignment<<':'<<s->declaration.tlsBytes<<':'<<s->declaration.opaqueBytes<<':'
        <<inherit<<':'<<policy<<':'<<parameter.sched_priority<<':'<<signals<<':';
    for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&affinity))identity<<cpu<<',';
    if(signals==0)for(int signal=1;signal<NSIG;++signal)identity<<sigismember(&signalMask,signal);
    s->identity=digest(identity.str());s->declaration.requireValid();
    for(const auto& reviewed:detail::retainedHostServiceQualifiedClasses)if(reviewed.role==role){
        if(reviewed.creatorVersion!=detail::retainedHostServiceCreatorVersion||reviewed.evidence.empty())throw std::logic_error("Unsupported CPU creator ABI evidence");
        s->qualified=true;
    }
    if(origin==MetadataAuthority::UnqualifiedDiagnostic)s->declaration.opaqueBytes=0;
    return RetainedHostServiceProgram(std::move(s));
}
RetainedHostServiceProgram RetainedHostServiceProgram::fromInstalledWire(RetainedHostServiceRole role,const std::string& resource,const std::string& live,const RetainedHostServiceResourceDeclaration& declaration){
    declaration.requireValid();
    if(resource!=familyIdentity(role)||live.size()!=64)throw std::invalid_argument("CPU wire declaration has an unsupported resource family");
    auto s=std::make_shared<State>(role,MetadataAuthority::InstalledWire);s->declaration=declaration;
    s->identity=live;s->resourceIdentity=resource;s->qualified=true;
    return RetainedHostServiceProgram(std::move(s));
}
RetainedHostServiceProgram RetainedHostServiceProgram::unqualifiedDiagnostic(RetainedHostServiceRole role){return snapshot(role,MetadataAuthority::UnqualifiedDiagnostic,originalEvidence());}
RetainedHostServiceProgram RetainedHostServiceProgram::recorded(RetainedHostServiceRole role,std::size_t opaque){
    auto evidence=std::shared_ptr<Evidence>(new Evidence);evidence->recorded=true;evidence->resourcePolicy="recorded-policy";evidence->liveLoader="recorded-loader";
    auto p=snapshot(role,MetadataAuthority::Recorded,std::move(evidence));auto state=std::const_pointer_cast<State>(p.state_);state->declaration.opaqueBytes=opaque;state->qualified=true;return p;
}
RetainedHostServiceProgram RetainedHostServiceProgram::recordedDiagnostic(RetainedHostServiceRole role){
    auto p=recorded(role,0);auto state=std::const_pointer_cast<State>(p.state_);state->origin=MetadataAuthority::UnqualifiedDiagnostic;return p;
}
RetainedHostServiceProgram RetainedHostServiceProgram::recordedWithEvidence(RetainedHostServiceRole role,std::shared_ptr<const Evidence> evidence,std::size_t opaque){
    auto p=snapshot(role,MetadataAuthority::Recorded,std::move(evidence));auto state=std::const_pointer_cast<State>(p.state_);state->declaration.opaqueBytes=opaque;state->qualified=true;return p;
}
void RetainedHostServiceProgram::requireCurrent()const{
    if(state_->origin==MetadataAuthority::InstalledWire)throw std::logic_error("CPU service wire projection cannot authorize local native creation");
    if(snapshot(role(),state_->origin,state_->evidence).identity()!=identity())throw std::logic_error("CPU service original pthread defaults/runtime changed after declaration");
}
PhysicalMemoryPlan RetainedHostServiceProgram::bom(PhysicalMemoryResource cpu)const{
    if(!cpu.valid()||cpu.device!=DeviceId::cpu())throw std::invalid_argument("CPU service BOM requires original physical CPU0");
    if(!state_->qualified&&state_->origin!=MetadataAuthority::UnqualifiedDiagnostic)throw std::runtime_error("Retained CPU service libc/TLS program is unqualified: no installed opaque upper-limit certificate");
    PhysicalMemoryPlanBuilder b;b.add(cpu,PhysicalMemoryOwner::RetainedHostServiceStorage,allocationBytes()).add(cpu,PhysicalMemoryOwner::RetainedHostServiceRuntime,state_->declaration.opaqueBytes);return b.build();
}
std::size_t RetainedHostServiceProgram::allocationBytes()const noexcept{return state_->declaration.stackBytes+pages(state_->declaration.objectBytes,state_->declaration.pageBytes);}
std::size_t RetainedHostServiceProgram::stackBytes()const noexcept{return state_->declaration.stackBytes;}
std::size_t RetainedHostServiceProgram::guardBytes()const noexcept{return state_->declaration.guardBytes;}
std::size_t RetainedHostServiceProgram::pageBytes()const noexcept{return state_->declaration.pageBytes;}
RetainedHostServiceResourceDeclaration RetainedHostServiceProgram::resourceDeclaration()const noexcept{return state_->declaration;}
RetainedHostServiceRole RetainedHostServiceProgram::role()const noexcept{return state_->role;}
RetainedHostServiceCreatorPolicy RetainedHostServiceProgram::creatorPolicy()const noexcept{return role()==RetainedHostServiceRole::ImmutableLookupIO?RetainedHostServiceCreatorPolicy::OriginalOpenMPPartition:RetainedHostServiceCreatorPolicy::OriginalCaller;}
const std::string& RetainedHostServiceProgram::identity()const noexcept{return state_->identity;}
const std::string& RetainedHostServiceProgram::resourceIdentity()const noexcept{return state_->resourceIdentity;}
RetainedHostServiceEvidence RetainedHostServiceProgram::evidenceKind()const noexcept{
    switch(state_->origin){case MetadataAuthority::Installed:return RetainedHostServiceEvidence::Installed;case MetadataAuthority::Recorded:return RetainedHostServiceEvidence::Recorded;case MetadataAuthority::UnqualifiedDiagnostic:return RetainedHostServiceEvidence::UnqualifiedDiagnostic;case MetadataAuthority::InstalledWire:return RetainedHostServiceEvidence::InstalledWireProjection;}
    std::terminate();
}
int RetainedHostServiceProgram::createThread(pthread_t* thread,void* stack,void*(*entry)(void*),void* argument,void(*before)(void*),void* context)const{
    if(!state_->hasAttributes)throw std::logic_error("CPU service wire projection cannot create a native thread");
    requireCurrent(); // Final no-rehash gate precedes opaque arm and the actual pthread_create.
    std::lock_guard lock(state_->mutex);
    // The original deep defaults retain scheduling/affinity/signal precedence;
    // only backing and guard management become exact admitted ownership.
    auto* attributes=&const_cast<State*>(state_.get())->attributes;
    auto result=pthread_attr_setstack(attributes,stack,state_->declaration.stackBytes);if(!result)result=pthread_attr_setguardsize(attributes,0);
    if(result)return result;
    before(context); // Last potentially throwing operation; publishes original opaque token before API attempt.
    return pthread_create(thread,attributes,entry,argument);
}
RetainedHostServiceProcessObligation::RetainedHostServiceProcessObligation(std::shared_ptr<PhysicalMemoryAuthority> memory,Original token,std::string identity)
    :memory_(std::move(memory)),original_(std::move(token)),identity_(std::move(identity)){}
/** @brief Exact mapping root; destruction orders thread/object before munmap and PMA release. */
struct RetainedHostServiceInstanceState {
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    RetainedHostServiceProgram program;
    PhysicalMemoryHostServiceInstanceLease instance;
    PhysicalMemoryAllocationLease exact;
    std::shared_ptr<RetainedHostServiceProcessObligation> opaque;
    void* mapping=MAP_FAILED;
    RetainedParentTicketServiceWorker* worker=nullptr;
    RetainedHostServiceInstanceState(std::shared_ptr<PhysicalMemoryAuthority> m,RetainedHostServiceProgram p,PhysicalMemoryHostServiceInstanceLease i):memory(std::move(m)),program(std::move(p)),instance(std::move(i)){}
    ~RetainedHostServiceInstanceState()noexcept{
        if(worker){worker->~RetainedParentTicketServiceWorker();worker=nullptr;}
        if(mapping!=MAP_FAILED && munmap(mapping,program.allocationBytes()+program.guardBytes()))std::terminate();
        if(opaque)GPUDeviceProcessLifetime::completeHostServiceUse(opaque);
    }
};
PreparedRetainedHostService::PreparedRetainedHostService(std::unique_ptr<RetainedHostServiceInstanceState> s):state_(std::move(s)){}
PreparedRetainedHostService::~PreparedRetainedHostService()noexcept=default;
RetainedHostServiceUseToken::RetainedHostServiceUseToken(std::unique_ptr<RetainedHostServiceInstanceState> state):state_(std::move(state)){}
RetainedHostServiceUseToken::RetainedHostServiceUseToken(RetainedHostServiceUseToken&&)noexcept=default;
RetainedHostServiceUseToken& RetainedHostServiceUseToken::operator=(RetainedHostServiceUseToken&&)noexcept=default;
RetainedHostServiceUseToken::~RetainedHostServiceUseToken()=default;
const RetainedHostServiceProgram& RetainedHostServiceUseToken::program()const{if(!state_)throw std::logic_error("CPU service instance token was already consumed");return state_->program;}
const std::shared_ptr<PhysicalMemoryAuthority>& RetainedHostServiceUseToken::memory()const{(void)program();return state_->memory;}
RetainedHostServiceUseToken RetainedHostServiceUseToken::claim(std::shared_ptr<PhysicalMemoryAuthority> memory,RetainedHostServiceProgram program,RetainedHostServiceInstance key){
    if(!memory||memory->phase()!=PhysicalMemoryAuthorityPhase::Admitted)throw std::logic_error("CPU service instance requires original final Admitted PMA");
    if(!program.state_->qualified&&program.state_->origin!=RetainedHostServiceProgram::MetadataAuthority::UnqualifiedDiagnostic)throw std::runtime_error("Retained CPU service libc/TLS program is unqualified before native creation");
    if(program.state_->origin==RetainedHostServiceProgram::MetadataAuthority::InstalledWire)throw std::logic_error("CPU service wire projection cannot mint a local materialization token");
    if(key.role!=program.role())throw std::logic_error("CPU service original instance changed its declared role");
    GPUDeviceProcessLifetime::requireOriginalHostServiceAuthority(memory);
    program.requireCurrent();
    const auto policy=program.evidenceKind()==RetainedHostServiceEvidence::UnqualifiedDiagnostic?
        PhysicalMemoryHostServiceInstancePolicy::DiagnosticSequential:PhysicalMemoryHostServiceInstancePolicy::OneShot;
    auto instance=memory->claimHostServiceInstance(memory,key,program.identity(),policy);
    auto s=std::make_unique<RetainedHostServiceInstanceState>(std::move(memory),std::move(program),std::move(instance));
    s->exact=s->memory->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::RetainedHostServiceStorage,s->program.allocationBytes());
    RetainedHostServiceProcessObligation::Original token=RetainedHostServiceProcessObligation::UnqualifiedDiagnostic{};
    if(s->program.state_->origin!=RetainedHostServiceProgram::MetadataAuthority::UnqualifiedDiagnostic)
        token=s->memory->reserveNewAllocations(DeviceId::cpu(),PhysicalMemoryOwner::RetainedHostServiceRuntime,s->program.state_->declaration.opaqueBytes);
    s->opaque=std::shared_ptr<RetainedHostServiceProcessObligation>(new RetainedHostServiceProcessObligation(s->memory,std::move(token),s->program.identity()));
    s->instance.publish();
    return RetainedHostServiceUseToken(std::move(s));
}
std::unique_ptr<PreparedRetainedHostService> PreparedRetainedHostService::materialize(RetainedHostServiceUseToken token){
    (void)token.program();
    auto s=std::move(token.state_);
    s->program.requireCurrent(); // A delayed token cannot hide new TLS/default policy before mapping.
    std::optional<ScopedOpenMPPartitionAffinity> originalIOScope;
    if(s->program.creatorPolicy()==RetainedHostServiceCreatorPolicy::OriginalOpenMPPartition)originalIOScope.emplace();
    s->mapping=mmap(nullptr,s->program.allocationBytes()+s->program.guardBytes(),PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK,-1,0);
    if(s->mapping==MAP_FAILED)throw std::system_error(errno,std::generic_category(),"CPU service exact mmap");
    auto* stack=static_cast<std::byte*>(s->mapping)+s->program.guardBytes();
    if(mprotect(stack,s->program.allocationBytes(),PROT_READ|PROT_WRITE))throw std::system_error(errno,std::generic_category(),"CPU service exact mprotect");
    for(std::size_t at=0;at<s->program.allocationBytes();at+=s->program.pageBytes())reinterpret_cast<volatile std::byte*>(stack)[at]=std::byte{};
    s->worker=new(stack+s->program.stackBytes())RetainedParentTicketServiceWorker(pthread_t{});
    const auto retain=[](void* value){auto& state=*static_cast<RetainedHostServiceInstanceState*>(value);GPUDeviceProcessLifetime::retainHostServiceRoot(state.opaque);};
    auto resultOwner=std::unique_ptr<PreparedRetainedHostService>(new PreparedRetainedHostService(std::move(s)));
    auto& owned=*resultOwner->state_;
    const auto result=owned.program.createThread(&owned.worker->worker_,stack,&RetainedParentTicketServiceWorker::invoke,owned.worker,retain,&owned);
    if(result)throw std::system_error(result,std::generic_category(),"CPU service original pthread_create");
    owned.worker->running_=true;
    return resultOwner;
}
RetainedParentTicketServiceWorker& PreparedRetainedHostService::worker()const noexcept{return *state_->worker;}
const RetainedHostServiceProgram& PreparedRetainedHostService::program()const noexcept{return state_->program;}
const std::shared_ptr<PhysicalMemoryAuthority>& PreparedRetainedHostService::memory()const noexcept{return state_->memory;}
}
