/**
 * @file Test__PreparedGPUWorkerThread.cpp
 * @brief Replay the actual original-PMA stack mapping and pthread lifetime implementation.
 *
 * No native accelerator or model is created. Exact source stack pages and
 * guards, preclaim/materialization, creation failure and fatal join behavior
 * are exercised through the production owner and original central ledger.
 */
#include "backends/PreparedGPUWorkerThread.h"
#include "fixtures/planning/NativeStoragePreparingReplayAccess.h"
#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <cerrno>
#include <signal.h>
#include <sched.h>
#include <system_error>
#include <unistd.h>
namespace llaminar2::test {
/** @brief Inject only pthread failure outcomes; production calls remain fixed original APIs. */
struct PreparedGPUWorkerThreadReplayAccess {
    static int failCreate(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*) { return EAGAIN; }
    static int failJoin(pthread_t,void**) { return EDEADLK; }
    static void creationFailure(PreparedGPUWorkerThread& worker){ worker.create_=failCreate; }
    static void normalCreation(PreparedGPUWorkerThread& worker){ worker.create_=pthread_create; }
    static void joinFailure(PreparedGPUWorkerThread& worker){ worker.join_=failJoin; }
    inline static std::atomic<int> inheritance=-1;
    inline static std::atomic<bool> affinityMatched=false;
    inline static cpu_set_t expectedAffinity{};
    static int observeCreate(pthread_t* thread,const pthread_attr_t* attr,void*(*entry)(void*),void* argument){
        int policy=-1;cpu_set_t affinity{};
        if(pthread_attr_getinheritsched(attr,&policy) || pthread_attr_getaffinity_np(attr,sizeof(affinity),&affinity))std::terminate();
        inheritance.store(policy);affinityMatched.store(CPU_EQUAL(&affinity,&expectedAffinity));
        return pthread_create(thread,attr,entry,argument);
    }
    static void observe(PreparedGPUWorkerThread& worker){worker.create_=observeCreate;}
};
namespace {
/** @return Sole original Preparing authority; fixed test capacity is not an opaque resource certificate. */
std::shared_ptr<NativeStoragePreparationPmaBinding> preparation(){
    const std::array observed{PhysicalMemoryResource{0,DeviceId::cpu(),128u<<20,128u<<20}};
    return std::shared_ptr<NativeStoragePreparationPmaBinding>(NativeStoragePreparationReplayAccess::begin(observed));
}
/** @return Actual canonical exact stack allocation totals, never a duplicated live balance. */
PhysicalMemoryOwnerAttestation actual(const std::shared_ptr<PhysicalMemoryAuthority>& memory){
    for(const auto& line:memory->rankAttestation())if(line.owner==PhysicalMemoryOwner::LocalCollective)return line;
    throw std::logic_error("Missing original source owner");
}
/** @brief Inspect original native pthread facts while running on its actual source-owned stack. */
struct Entry {
    void* stack=nullptr;std::size_t bytes=0;std::atomic<unsigned> calls=0;cpu_set_t affinity{};
    static void invoke(void* argument){auto& entry=*static_cast<Entry*>(argument);pthread_attr_t attributes;
        if(pthread_getattr_np(pthread_self(),&attributes) || pthread_attr_getstack(&attributes,&entry.stack,&entry.bytes) ||
           pthread_attr_destroy(&attributes) || pthread_getaffinity_np(pthread_self(),sizeof(entry.affinity),&entry.affinity))std::terminate();
        entry.calls.fetch_add(1,std::memory_order_release);
    }
};
}
/** @brief Restore process-global pthread defaults even when a focused assertion throws/returns. */
class DefaultAttributeControl final {
public:
    DefaultAttributeControl(){
        auto result=pthread_getattr_default_np(&original_);
        if(result)throw std::system_error(result,std::generic_category());
        result=pthread_getattr_default_np(&changed_);
        if(result){pthread_attr_destroy(&original_);throw std::system_error(result,std::generic_category());}
    }
    ~DefaultAttributeControl(){
        if(pthread_setattr_default_np(&original_) || pthread_attr_destroy(&changed_) || pthread_attr_destroy(&original_))std::terminate();
    }
    void install(const cpu_set_t& affinity){
        sched_param parameter{};
        if(pthread_attr_setinheritsched(&changed_,PTHREAD_EXPLICIT_SCHED) ||
           pthread_attr_setschedpolicy(&changed_,SCHED_OTHER) || pthread_attr_setschedparam(&changed_,&parameter) ||
           pthread_attr_setaffinity_np(&changed_,sizeof(affinity),&affinity) || pthread_setattr_default_np(&changed_))
            throw std::runtime_error("Cannot install supported test pthread defaults");
    }
private:
    pthread_attr_t original_,changed_;
};
TEST(PreparedGPUWorkerThread, OriginalDefaultSchedulingAndAffinityArePreserved){
    cpu_set_t admitted{},selected{};ASSERT_EQ(sched_getaffinity(0,sizeof(admitted),&admitted),0);CPU_ZERO(&selected);
    int cpu=0;while(cpu<CPU_SETSIZE && !CPU_ISSET(cpu,&admitted))++cpu;ASSERT_LT(cpu,CPU_SETSIZE);CPU_SET(cpu,&selected);
    DefaultAttributeControl defaults;defaults.install(selected);
    auto source=preparation();auto stack=PreparedGPUWorkerStack::create(source,GPUWorkerStackProgram::originalDefaults());
    PreparedGPUWorkerThreadReplayAccess::expectedAffinity=selected;
    PreparedGPUWorkerThread worker;PreparedGPUWorkerThreadReplayAccess::observe(worker);Entry entry;
    worker.start(stack,Entry::invoke,&entry);worker.join();
    EXPECT_EQ(PreparedGPUWorkerThreadReplayAccess::inheritance.load(),PTHREAD_EXPLICIT_SCHED);
    EXPECT_TRUE(PreparedGPUWorkerThreadReplayAccess::affinityMatched.load());EXPECT_TRUE(CPU_EQUAL(&entry.affinity,&selected));
}
TEST(PreparedGPUWorkerThread, EscapingEntryExceptionTerminatesLikeStdThread){
    EXPECT_DEATH({auto source=preparation();auto stack=PreparedGPUWorkerStack::create(source,GPUWorkerStackProgram::originalDefaults());
        PreparedGPUWorkerThread worker;worker.start(stack,[](void*){throw std::runtime_error("original worker entry escaped");},nullptr);
        worker.join();},"terminate");
}

TEST(PreparedGPUWorkerThread, ExactStackGuardAndSamePMAAdmissionReplay){
    auto source=preparation();auto memory=source->memory();const auto program=GPUWorkerStackProgram::originalDefaults();
    auto stack=PreparedGPUWorkerStack::create(source,program);
    EXPECT_EQ(actual(memory).preparation_committed_bytes,program.stackBytes());
    EXPECT_EQ(actual(memory).preparation_materialized_bytes,program.stackBytes());
    const std::array creators{stack->creatorDeclaration()};auto snapshot=source->closeForPlanning();
    auto declarations=NativeMandatoryStorageDeclarations::fromCreators(snapshot,creators);
    auto certificate=std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(declarations.mandatoryBOM());
    auto final=source->admit(std::move(snapshot),std::move(declarations),certificate);
    EXPECT_EQ(source->memory().get(),memory.get());EXPECT_EQ(memory->phase(),PhysicalMemoryAuthorityPhase::Admitted);
    for(unsigned replay=0;replay<3;++replay){Entry entry;PreparedGPUWorkerThread worker;
        worker.start(stack,Entry::invoke,&entry);EXPECT_TRUE(worker.joinable());worker.join();EXPECT_FALSE(worker.joinable());
        EXPECT_EQ(entry.calls.load(std::memory_order_acquire),1u);EXPECT_EQ(entry.bytes,program.stackBytes());
        if(program.guardBytes())EXPECT_EXIT({*static_cast<volatile unsigned char*>(static_cast<void*>(static_cast<char*>(entry.stack)-1))=1;_exit(0);},
            ::testing::KilledBySignal(SIGSEGV),"");
    }
    final->sealForExecution();EXPECT_FALSE(final->retirementGate().retired());
    stack.reset();EXPECT_EQ(actual(memory).preparation_committed_bytes,0u);EXPECT_TRUE(final->retirementGate().retired());
}
TEST(PreparedGPUWorkerThread, FailedCreationKeepsOriginalClaimAndNoRunningAlias){
    auto source=preparation();auto memory=source->memory();auto stack=PreparedGPUWorkerStack::create(source,GPUWorkerStackProgram::originalDefaults());
    const auto bytes=actual(memory).preparation_materialized_bytes;PreparedGPUWorkerThread worker;Entry entry;
    PreparedGPUWorkerThreadReplayAccess::creationFailure(worker);
    EXPECT_THROW(worker.start(stack,Entry::invoke,&entry),std::system_error);EXPECT_FALSE(worker.joinable());
    EXPECT_EQ(entry.calls.load(),0u);EXPECT_EQ(actual(memory).preparation_materialized_bytes,bytes);
    PreparedGPUWorkerThreadReplayAccess::normalCreation(worker);worker.start(stack,Entry::invoke,&entry);worker.join();
    EXPECT_EQ(entry.calls.load(),1u);stack.reset();EXPECT_EQ(actual(memory).preparation_materialized_bytes,0u);
}
TEST(PreparedGPUWorkerThread, FailedJoinIsFatalBeforeRootCleanup){
    EXPECT_DEATH({auto source=preparation();auto stack=PreparedGPUWorkerStack::create(source,GPUWorkerStackProgram::originalDefaults());
        PreparedGPUWorkerThread worker;Entry entry;worker.start(stack,Entry::invoke,&entry);
        PreparedGPUWorkerThreadReplayAccess::joinFailure(worker);worker.join();},"");
}
}
