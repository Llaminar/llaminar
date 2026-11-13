/**
 * @file NativeContextPreparation.cpp
 * @brief Original source session consumption by selected GPU worker birth and admission.
 */
#include "NativeContextPreparation.h"
#include "GPUDeviceContextPool.h"
#include <algorithm>
#include <stdexcept>
#include "GPUDeviceProcessLifetime.h"
#include "utils/Sha256.h"
#include <dlfcn.h>
#include <link.h>
#include <map>
#include <set>
#include <cstring>
#include <exception>
#include <filesystem>
#include "transfer/TransferEngine.h"
namespace llaminar2 {
PreparedGPUContextObjectGeometry PreparedGPUContextFactory::objectGeometry(DeviceId device){
#ifdef HAVE_CUDA
    if(device.is_cuda())return cudaObjectGeometry();
#endif
#ifdef HAVE_ROCM
    if(device.is_rocm())return rocmObjectGeometry();
#endif
    throw std::invalid_argument("Prepared context object requires its compiled original native backend");
}
std::string PreparedGPUContextFactory::digestIdentity(const std::string& value){
    const auto result=sha256BytesHex(value);if(!result)throw std::runtime_error("Native source software identity could not be hashed");return *result;
}
std::string PreparedGPUContextFactory::libraryIdentity(const void* symbol){
    Dl_info info{};if(!symbol || !dladdr(symbol,&info) || !info.dli_fname)throw std::runtime_error("Native source symbol has no original loaded library identity");
    const auto result=sha256FileHex(std::filesystem::canonical(info.dli_fname));
    if(!result)throw std::runtime_error("Native source original loaded library could not be authenticated");return *result;
}
std::string PreparedGPUContextFactory::libraryClosureIdentity(const void* symbol){
    if(!symbol)throw std::runtime_error("Native runtime has no original loaded ELF symbol");
    struct Image {std::string path,soname;std::vector<std::string> needed;bool contains=false;};
    struct Snapshot {const void* symbol;std::vector<Image> images;std::exception_ptr failure;};
    Snapshot snapshot{symbol,{},{}};
    // glibc owns its loader write lock for this complete traversal. Copy every
    // needed dynamic string and provider identity while that authority pins the
    // images; no link_map or mapped ELF pointer escapes the callback.
    const auto capture=[](dl_phdr_info* info,std::size_t,void* user)->int{
        auto& out=*static_cast<Snapshot*>(user);
        try{
            Image image;image.path=info->dlpi_name?info->dlpi_name:"";
            const ElfW(Dyn)* dynamic=nullptr;std::size_t dynamicCount=0;
            const auto address=reinterpret_cast<std::uintptr_t>(out.symbol);
            for(std::size_t i=0;i<info->dlpi_phnum;++i){const auto& ph=info->dlpi_phdr[i];
                if(ph.p_type==PT_LOAD && address>=info->dlpi_addr+ph.p_vaddr &&
                   address-(info->dlpi_addr+ph.p_vaddr)<ph.p_memsz)image.contains=true;
                if(ph.p_type==PT_DYNAMIC){dynamic=reinterpret_cast<const ElfW(Dyn)*>(info->dlpi_addr+ph.p_vaddr);dynamicCount=ph.p_memsz/sizeof(ElfW(Dyn));}}
            if(!dynamic)return 0;
            const char* table=nullptr;std::size_t tableBytes=0;
            for(std::size_t i=0;i<dynamicCount && dynamic[i].d_tag!=DT_NULL;++i){const auto& d=dynamic[i];
                if(d.d_tag==DT_STRTAB){auto pointer=d.d_un.d_ptr;if(pointer<info->dlpi_addr)pointer+=info->dlpi_addr;table=reinterpret_cast<const char*>(pointer);}
                if(d.d_tag==DT_STRSZ)tableBytes=d.d_un.d_val;}
            if(!table || !tableBytes)throw std::runtime_error("Loaded ELF snapshot lacks its bounded dynamic strings");
            const auto string=[&](std::size_t offset){
                if(offset>=tableBytes)throw std::runtime_error("Loaded ELF dynamic string offset is invalid");
                const auto* end=static_cast<const char*>(std::memchr(table+offset,0,tableBytes-offset));
                if(!end)throw std::runtime_error("Loaded ELF dynamic string is unterminated");return std::string(table+offset,end);};
            for(std::size_t i=0;i<dynamicCount && dynamic[i].d_tag!=DT_NULL;++i){const auto& d=dynamic[i];
                if(d.d_tag==DT_SONAME)image.soname=string(d.d_un.d_val);
                if(d.d_tag==DT_NEEDED)image.needed.push_back(string(d.d_un.d_val));}
            out.images.push_back(std::move(image));return 0;
        }catch(...){out.failure=std::current_exception();return 1;}
    };
    (void)dl_iterate_phdr(capture,&snapshot);
    if(snapshot.failure)std::rethrow_exception(snapshot.failure);
    std::optional<std::size_t> root;std::multimap<std::string,std::size_t> providers;
    for(std::size_t i=0;i<snapshot.images.size();++i){const auto& image=snapshot.images[i];
        if(image.contains){if(root)throw std::runtime_error("Native symbol has ambiguous loaded image ranges");root=i;}
        if(!image.soname.empty())providers.emplace(image.soname,i);
        else if(!image.path.empty())providers.emplace(std::filesystem::path(image.path).filename().string(),i);
    }
    if(!root)throw std::runtime_error("Native runtime symbol is outside the original loaded ELF snapshot");
    std::set<std::size_t> visited;std::multimap<std::string,std::string> evidence;
    const auto visit=[&](auto&& self,std::size_t index)->void{
        if(!visited.insert(index).second)return;const auto& image=snapshot.images[index];
        if(image.path.empty())throw std::runtime_error("Native runtime closure unexpectedly reaches the main executable");
        const auto path=std::filesystem::canonical(image.path);const auto hash=sha256FileHex(path);
        if(!hash)throw std::runtime_error("Native runtime closure provider could not be authenticated");
        evidence.emplace(path.filename().string(),*hash);
        for(const auto& needed:image.needed){const auto [first,last]=providers.equal_range(needed);
            if(first==last)throw std::runtime_error("Native runtime DT_NEEDED has no exact snapshotted SONAME provider");
            if(std::next(first)!=last)throw std::runtime_error("Native runtime DT_NEEDED has ambiguous snapshotted providers");
            self(self,first->second);}
    };
    visit(visit,*root);std::string value="native-loaded-closure-v1:";
    for(const auto& [name,hash]:evidence)value+=std::to_string(name.size())+":"+name+":"+hash+":";
    return digestIdentity(value);
}
NativeContextBirthKey PreparedGPUContextFactory::commonBirthKey(DeviceId device){
    NativeContextBirthKey key;key.device=device;key.population=NativeContextBirthPopulation::AdditionalContext;
    const auto program=GPUWorkerStackProgram::originalDefaults();const auto object=objectGeometry(device);
    key.worker_stack_bytes=program.stackBytes();key.worker_guard_bytes=program.guardBytes();key.pthread_defaults_identity=program.identity();
    key.context_object_bytes=object.bytes();key.context_object_alignment=object.alignment();
    key.primary_streams=key.primary_blas_handles=key.secondary_blas_handles=1;key.dependency_events=32;
    return key;
}
NativeContextBirthKey PreparedGPUContextFactory::birthKey(DeviceId device){
#ifdef HAVE_CUDA
    if(device.is_cuda())return cudaBirthKey(device.gpu_ordinal());
#endif
#ifdef HAVE_ROCM
    if(device.is_rocm())return rocmBirthKey(device.gpu_ordinal());
#endif
    throw std::invalid_argument("Native source worker requires its compiled original backend");
}
PreparedGPUContextBirth::PreparedGPUContextBirth(std::shared_ptr<NativeStoragePreparationPmaBinding> source,
    const NativeContextOpaqueContract& contract,GPUWorkerStackProgram program):source_(std::move(source)),contract_(contract){
    if(program.stackBytes()!=contract.key().worker_stack_bytes || program.guardBytes()!=contract.key().worker_guard_bytes || program.identity()!=contract.key().pthread_defaults_identity)
        throw std::invalid_argument("Original source worker stack geometry differs from certified constructor");
    opaque_=source_->reserveContextOpaque(contract_);
    stack_=PreparedGPUWorkerStack::create(source_,program.withContextObject(contract.key().context_object_bytes,contract.key().context_object_alignment));
}
void PreparedGPUContextBirth::begin(){
    if(contract_.lifetime()==NativePhysicalLifetime::ProcessRuntime)
        GPUDeviceProcessLifetime::retainPreparingOpaqueRoot(source_,opaque_,contract_);
    else opaque_->beginNative();
}
void PreparedGPUContextBirth::created(){opaque_->publishCreated();}
void PreparedGPUContextBirth::physicallyRetired(){
    stack_.reset();
    // Original worker handles and stack retire at validated reset. A reviewed
    // process-runtime contract retains the SAME opaque commitment/source in
    // the original process owner; reset is not physical return of those bytes.
    if(contract_.lifetime()==NativePhysicalLifetime::ContextGeneration)opaque_->retirePhysically();
}
NativeContextPreparation::NativeContextPreparation(std::span<const PhysicalMemoryResource> observations,int rank)
    :source_(new NativeStoragePreparationPmaBinding(observations,rank)){
    // The immutable registry is topology-wide, but process ownership and
    // ordinal namespaces are strictly local. A remote GPU0 is not this
    // process's GPU0 and must never create or borrow its native context.
    for(const auto& resource:observations)if(resource.world_rank==rank && resource.device.is_gpu())
        GPUDeviceProcessLifetime::requireOriginalProcessAuthority(resource.device,source_->memory());
}
std::shared_ptr<NativeContextPreparation> NativeContextPreparation::create(std::span<const PhysicalMemoryResource> observations,
    int rank,std::span<const DeviceId> devices){
    if(devices.empty())throw std::invalid_argument("Native source creation needs its selected participant population");
    const auto startup=GPUDeviceProcessLifetime::acquireNativeContext();
    std::vector<NativeContextBirthKey> keys;keys.reserve(devices.size());
    for(const auto device:devices)keys.push_back(PreparedGPUContextFactory::birthKey(device));
    return createQualified(observations,rank,keys);
}
std::shared_ptr<NativeContextPreparation> NativeContextPreparation::createQualified(std::span<const PhysicalMemoryResource> observations,
    int rank,std::span<const NativeContextBirthKey> keys){
    if(keys.empty())throw std::invalid_argument("Native source preparation needs an explicit selected worker population");
    // Validate the COMPLETE selected population before either exact mappings or native calls.
    std::vector<NativeContextOpaqueContract> contracts;contracts.reserve(keys.size());
    for(const auto& key:keys){
        if(std::any_of(contracts.begin(),contracts.end(),[&](const auto& value){return value.key().device==key.device;}))
            throw std::invalid_argument("Native source preparation repeats a physical worker");
        key.requireComplete();
        const auto object=PreparedGPUContextFactory::objectGeometry(key.device);
        if(key.context_object_bytes!=object.bytes() || key.context_object_alignment!=object.alignment())
            throw std::invalid_argument("Selected source certificate has foreign compiled context object geometry");
        contracts.push_back(NativeContextOpaqueCatalog::require(key));
    }
    auto result=std::shared_ptr<NativeContextPreparation>(new NativeContextPreparation(observations,rank));
    for(const auto& contract:contracts)result->births_.push_back(std::shared_ptr<PreparedGPUContextBirth>(
        new PreparedGPUContextBirth(result->source_,contract,GPUWorkerStackProgram::originalDefaults())));
    return result;
}
void NativeContextPreparation::freezeForPlanning(){
    if(planning_snapshot_ || admitted_binding_)throw std::logic_error("Original native source planning closure is one-shot");
    planning_snapshot_.emplace(closeForPlanning());
}
void NativeContextPreparation::admitFinal(std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate> certificate){
    if(admitted_binding_){
        if(memory()->admission()!=certificate)throw std::logic_error("Native source cannot replace its admitted canonical certificate");
        return;
    }
    if(!planning_snapshot_)throw std::logic_error("Native source was not frozen before final admission");
    admitted_binding_=admit(std::move(*planning_snapshot_),std::move(certificate));planning_snapshot_.reset();
    admitted_binding_->sealForExecution();
}
void NativeContextPreparation::retireUnadmitted(){
    if(memory()->phase()!=PhysicalMemoryAuthorityPhase::Preparing)
        throw std::logic_error("Admitted native source requires complete model retirement, not birth-only cleanup");
    retireReleasedContexts();
}
std::shared_ptr<RetainedParticipantGraphExecutable> PreparedGPUContextUsePermit::borrowPreparedCalibration(void* stream)const{
    return GPUDeviceContextPool::instance().borrowPreparedCalibration(*this,stream);
}
void NativeContextPreparation::retirePreparedCalibrations(){
    for(const auto& birth:births_)if(GPUDeviceContextPool::instance().ownsPreparedBirth(birth))
        GPUDeviceContextPool::instance().retirePreparedCalibration(birth);
}
std::size_t NativeContextPreparation::retireReleasedContexts(){return retireReleasedContexts(TransferEngine::instance());}
std::size_t NativeContextPreparation::retireReleasedContexts(const TransferEngine& transfers){
    if(memory()->phase()!=PhysicalMemoryAuthorityPhase::Preparing && memory()->phase()!=PhysicalMemoryAuthorityPhase::Admitted)
        throw std::logic_error("Original released source is not in a live physical authority phase");
    retirePreparedCalibrations();
    std::vector<ExclusiveModelRetirementTicket> tickets;
    tickets.reserve(births_.size());
    for(const auto& birth:births_)if(GPUDeviceContextPool::instance().ownsPreparedBirth(birth))
        tickets.push_back(transfers.beginExclusiveNativeContextRetirement(birth->device()));
    if(tickets.empty())return 0;
    return transfers.completeExclusiveModelRetirements(std::move(tickets)).size();
}
NativePreparationPlanningSnapshot NativeContextPreparation::closeForPlanning(){
    if(mandatory_bom_)throw std::logic_error("Original source preparation was already closed for planning");
    auto snapshot=source_->closeForPlanning();
    mandatory_bom_=std::make_shared<const PhysicalMemoryPlan>(snapshot.retainedPreparationBOM());
    return snapshot;
}
const PhysicalMemoryPlan& NativeContextPreparation::mandatoryBOM()const{
    if(!mandatory_bom_)throw std::logic_error("Native source mandatory declaration is not closed before planning");
    return *mandatory_bom_;
}
void NativeContextPreparation::requireSelectedDevice(DeviceId device)const{
    if(std::none_of(births_.begin(),births_.end(),[&](const auto& value){return value->device()==device;}))
        throw std::invalid_argument("Original native source did not declare this participant; late population extension forbidden");
}
const PhysicalMemoryResource& NativeContextPreparation::originalResource(DeviceId device)const{
    if(device.is_gpu())requireSelectedDevice(device);
    const auto* resource=mandatoryBOM().find({memory()->worldRank(),device});
    if(!resource)throw std::invalid_argument("Source did not declare this original physical allocator");return resource->resource();
}
PreparedGPUContextUsePermit NativeContextPreparation::acquire(DeviceId device){
    const auto found=std::find_if(births_.begin(),births_.end(),[&](const auto& value){return value->device()==device;});
    if(found==births_.end())throw std::invalid_argument("Source did not prepare the requested original worker");
    return GPUDeviceContextPool::instance().retainPreparedContext(*found);
}
std::shared_ptr<NativeStoragePmaBinding> NativeContextPreparation::admit(NativePreparationPlanningSnapshot&& snapshot,
    std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate> certificate){
    std::vector<NativeStorageCreatorDeclaration> creators;creators.reserve(births_.size());
    for(const auto& birth:births_){
        if(!birth->stack_)throw std::logic_error("Source native root retired before final admission");
        creators.push_back(birth->stack_->creatorDeclaration());
    }
    auto declarations=NativeMandatoryStorageDeclarations::fromCreators(snapshot,creators);
    return source_->admit(std::move(snapshot),std::move(declarations),std::move(certificate));
}
}
