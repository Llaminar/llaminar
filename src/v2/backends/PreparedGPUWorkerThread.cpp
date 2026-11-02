/**
 * @file PreparedGPUWorkerThread.cpp
 * @brief Source-owned worker stack/object allocation and join on the original Preparing PMA.
 *
 * Stack pages are preclaimed before mmap and prefaulted before the worker can
 * touch them. Guard pages have explicit virtual extent and no physical-byte
 * charge. A failed join is fatal before any mapping or source token can retire.
 */
#include "PreparedGPUWorkerThread.h"
#include "collective/NativeStoragePreparationPmaBinding.h"
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <sys/mman.h>
#include <unistd.h>
#include "utils/Sha256.h"
#include <sstream>
#include <signal.h>
#include <sched.h>
namespace llaminar2 {
/** @brief Deep original pthread attribute snapshot; libc owns any default-affinity representation. */
struct GPUWorkerStackProgram::Attributes {
    pthread_attr_t value;
    std::mutex mutex;
    Attributes(){const auto result=::pthread_getattr_default_np(&value);
        if(result)throw std::system_error(result,std::generic_category(),"GPU worker original pthread defaults");}
    ~Attributes(){if(::pthread_attr_destroy(&value)!=0)std::terminate();}
};
GPUWorkerStackProgram GPUWorkerStackProgram::originalDefaults(){
    const auto page=::sysconf(_SC_PAGESIZE);
    if(page<=0)throw std::runtime_error("GPU worker stack has no original OS page geometry");
    auto attributes=std::make_shared<Attributes>();
    std::size_t stack=0,guard=0;int detached=0;
    const auto stackResult=::pthread_attr_getstacksize(&attributes->value,&stack);
    const auto guardResult=::pthread_attr_getguardsize(&attributes->value,&guard);
    const auto detachResult=::pthread_attr_getdetachstate(&attributes->value,&detached);
    if(stackResult || guardResult || detachResult || detached!=PTHREAD_CREATE_JOINABLE)
        throw std::runtime_error("GPU worker original attributes require a complete joinable thread");
    const auto granule=static_cast<std::size_t>(page);
    if(stack<static_cast<std::size_t>(PTHREAD_STACK_MIN) || stack%granule || guard%granule ||
       stack>std::numeric_limits<std::size_t>::max()-guard)
        throw std::invalid_argument("GPU worker original stack/guard geometry is not representable");
    return {stack,guard,granule,std::move(attributes)};
}
std::string GPUWorkerStackProgram::identity()const{
    std::lock_guard lock(attributes_->mutex);int inherit=0,policy=0,detach=0;struct sched_param parameter{};cpu_set_t affinity{};sigset_t mask{};
    if(pthread_attr_getinheritsched(&attributes_->value,&inherit) || pthread_attr_getschedpolicy(&attributes_->value,&policy) ||
       pthread_attr_getschedparam(&attributes_->value,&parameter) || pthread_attr_getdetachstate(&attributes_->value,&detach) ||
       pthread_attr_getaffinity_np(&attributes_->value,sizeof(affinity),&affinity))
        throw std::runtime_error("Original source pthread default identity is incomplete");
    const auto signalResult=pthread_attr_getsigmask_np(&attributes_->value,&mask);
    if(signalResult!=0 && signalResult!=PTHREAD_ATTR_NO_SIGMASK_NP)throw std::runtime_error("Original source pthread signal-mask identity is incomplete");
    std::ostringstream value;value<<"native-worker-defaults-v1:"<<stack_<<':'<<guard_<<':'<<page_<<':'<<inherit<<':'<<policy<<':'<<parameter.sched_priority<<':'<<detach<<':'<<signalResult<<':';
    for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&affinity))value<<cpu<<',';
    if(signalResult==0)for(int signal=1;signal<NSIG;++signal)value<<sigismember(&mask,signal);
    const auto digest=sha256BytesHex(value.str());if(!digest)throw std::runtime_error("Cannot hash original source pthread policy");return *digest;
}
GPUWorkerStackProgram GPUWorkerStackProgram::withContextObject(std::size_t bytes,std::size_t alignment) const {
    if(!bytes || !alignment || (alignment&(alignment-1)) || alignment>page_ || bytes>std::numeric_limits<std::size_t>::max()-(page_-1))
        throw std::invalid_argument("Original context object has invalid compiled size/alignment");
    const auto pages=((bytes+page_-1)/page_)*page_;
    if(stack_>std::numeric_limits<std::size_t>::max()-pages || stack_+pages>std::numeric_limits<std::size_t>::max()-guard_)
        throw std::overflow_error("Original stack/context-object physical extent overflows");
    auto result=*this;result.object_=bytes;result.object_alignment_=alignment;result.object_pages_=pages;return result;
}
/** @brief One actual allocation root; its callback token is the sole physical admission owner. */
struct PreparedGPUWorkerStack::State {
    std::shared_ptr<NativeStoragePreparationPmaBinding> owner;
    const GPUWorkerStackProgram program;
    const llaminarNativeStorageAuthorityV1 callbacks;
    llaminarNativeRootRequestV1 declaration{};
    void* token=nullptr;
    void* mapping=MAP_FAILED;
    std::atomic<bool> active=false;
    State(std::shared_ptr<NativeStoragePreparationPmaBinding> original,GPUWorkerStackProgram geometry)
        :owner(std::move(original)),program(geometry),callbacks(owner->preparationAuthority()){}
    /** @brief Free exact mapped bytes before returning the original canonical preclaim. */
    ~State() noexcept {
        if(active.load(std::memory_order_acquire))std::terminate();
        if(mapping!=MAP_FAILED && ::munmap(mapping,program.allocationBytes()+program.guardBytes())!=0)std::terminate();
        if(token && callbacks.release(&token)!=llaminarNativeStorageSuccessV1)std::terminate();
    }
};
PreparedGPUWorkerStack::PreparedGPUWorkerStack(std::unique_ptr<State> state):state_(std::move(state)){}
PreparedGPUWorkerStack::~PreparedGPUWorkerStack() noexcept=default;
const GPUWorkerStackProgram& PreparedGPUWorkerStack::program() const noexcept { return state_->program; }
std::shared_ptr<PreparedGPUWorkerStack> PreparedGPUWorkerStack::create(
    std::shared_ptr<NativeStoragePreparationPmaBinding> original,GPUWorkerStackProgram program){
    if(!original || original->memory()->phase()!=PhysicalMemoryAuthorityPhase::Preparing)
        throw std::invalid_argument("GPU worker stack requires its original open Preparing PMA");
    auto state=std::make_unique<State>(std::move(original),program);
    const auto identity=static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(state.get()));
    state->declaration={.abi_version=1,.record_bytes=sizeof(llaminarNativeRootRequestV1),
        .owner_identity=identity,.shared_anchor_identity=identity,.requested_bytes=program.allocationBytes(),
        .physical_bytes=program.allocationBytes(),.allocation_granularity=program.pageBytes(),
        .allocator_flags=static_cast<std::uint64_t>(program.guardBytes()),.setup_epoch=state->callbacks.setup_epoch,
        .resource_kind=llaminarNativeHostCPU0V1,.resource_ordinal=0,.host_numa=-1,
        .owner_family=llaminarNativeGPUWorkerAllocationV1,.lifetime=llaminarNativeContextGenerationLifetimeV1,
        .allocator_kind=llaminarNativeHostAnonymousPagesV1,.phase=llaminarNativeRetainedSetupV1,.reserved_zero=0};
    const auto status=state->callbacks.claim(state->callbacks.context,&state->declaration,&state->token);
    if(status==llaminarNativeStorageCapacityV1)throw PhysicalMemoryCapacityExhausted("Original GPU worker stack exceeds Preparing PMA");
    if(status!=llaminarNativeStorageSuccessV1)throw std::logic_error("Original GPU worker stack preclaim rejected");
    const auto bytes=program.allocationBytes()+program.guardBytes();
    state->mapping=::mmap(nullptr,bytes,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK,-1,0);
    if(state->mapping==MAP_FAILED)throw std::system_error(errno,std::generic_category(),"GPU worker stack mmap");
    auto* stack=static_cast<std::byte*>(state->mapping)+program.guardBytes();
    if(::mprotect(stack,program.allocationBytes(),PROT_READ|PROT_WRITE)!=0)
        throw std::system_error(errno,std::generic_category(),"GPU worker stack protection");
    // Materialize every promised stack/object page, not just an untouched virtual range.
    for(std::size_t at=0;at<program.allocationBytes();at+=program.pageBytes())
        reinterpret_cast<volatile std::byte*>(stack)[at]=std::byte{0};
    if(state->callbacks.materialize(state->token)!=llaminarNativeStorageSuccessV1)
        throw std::logic_error("Original GPU worker stack physical publication rejected");
    return std::shared_ptr<PreparedGPUWorkerStack>(new PreparedGPUWorkerStack(std::move(state)));
}
void* PreparedGPUWorkerStack::contextObjectStorage(std::size_t bytes,std::size_t alignment) const {
    if(bytes!=state_->program.contextObjectBytes() || alignment!=state_->program.contextObjectAlignment())
        throw std::invalid_argument("Actual compiled context object differs from certified source extent");
    return static_cast<std::byte*>(state_->mapping)+state_->program.guardBytes()+state_->program.stackBytes();
}
void PreparedGPUWorkerStack::requireContextObjectPointer(const void* pointer) const {
    const auto start=reinterpret_cast<std::uintptr_t>(contextObjectStorage(state_->program.contextObjectBytes(),state_->program.contextObjectAlignment()));
    const auto value=reinterpret_cast<std::uintptr_t>(pointer);
    if(!state_->program.contextObjectBytes() || value<start || value-start>=state_->program.contextObjectBytes())
        throw std::logic_error("Original prepared worker escaped its source-owned context object extent");
}
NativeStorageCreatorDeclaration PreparedGPUWorkerStack::creatorDeclaration() const {
    auto descriptor=state_->declaration;descriptor.owner_identity=0;descriptor.setup_epoch=0;
    return {NativeStorageCreatorUse::PreparationOwner,descriptor};
}
PreparedGPUWorkerThread& PreparedGPUWorkerThread::operator=(std::thread&& thread){
    if(joinable() || stack_)std::terminate();
    ordinary_=std::move(thread);return *this;
}
PreparedGPUWorkerThread::~PreparedGPUWorkerThread() noexcept { if(joinable())std::terminate(); }
bool PreparedGPUWorkerThread::joinable() const noexcept { return ordinary_.joinable() || running_; }
void* PreparedGPUWorkerThread::invoke(void* opaque){
    auto& owner=*static_cast<PreparedGPUWorkerThread*>(opaque);
    try{owner.entry_(owner.argument_);}
    catch(...){std::terminate();} // Match the original std::thread uncaught-entry contract.
    return nullptr;
}
void PreparedGPUWorkerThread::start(std::shared_ptr<PreparedGPUWorkerStack> stack,void(*entry)(void*),void* argument){
    if(joinable() || stack_ || !stack || !entry)throw std::invalid_argument("GPU worker needs one unused original prepared stack and entry");
    // Preserve the complete original pthread default attributes, not just the
    // standard scheduling fields. This also retains GNU default affinity and
    // signal-mask state that an attr_init reconstruction would silently drop.
    const auto attributes=stack->program().attributes_;
    std::unique_lock attributeLock(attributes->mutex);
    bool expected=false;
    if(!stack->state_->active.compare_exchange_strong(expected,true,std::memory_order_acq_rel))
        throw std::logic_error("Original GPU worker stack already has a running borrower");
    auto* address=static_cast<std::byte*>(stack->state_->mapping)+stack->program().guardBytes();
    auto result=::pthread_attr_setstack(&attributes->value,address,stack->program().stackBytes());
    if(!result)result=::pthread_attr_setguardsize(&attributes->value,0);
    if(result){stack->state_->active.store(false,std::memory_order_release);
        throw std::system_error(result,std::generic_category(),"GPU worker original stack binding");}
    stack_=std::move(stack);entry_=entry;argument_=argument;
    result=create_(&prepared_,&attributes->value,invoke,this);
    if(result){stack_->state_->active.store(false,std::memory_order_release);stack_.reset();entry_=nullptr;argument_=nullptr;
        throw std::system_error(result,std::generic_category(),"GPU worker pthread creation");}
    running_=true;
}
void PreparedGPUWorkerThread::join(){
    if(ordinary_.joinable()){ordinary_.join();return;}
    if(!running_)throw std::logic_error("Original GPU worker has no created thread to join");
    if(join_(prepared_,nullptr)!=0)std::terminate();
    running_=false;stack_->state_->active.store(false,std::memory_order_release);stack_.reset();entry_=nullptr;argument_=nullptr;
}
}
