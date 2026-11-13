/**
 * @file rccl_thread_lifetime_caller_probe.cpp
 * @brief Actual native join→group abort→EnsureReady→CommAbort lifetime replay.
 * The complete caller bodies remain unchanged. Only pthread result, trace,
 * post-gate allocation and actual reclaim execution are explicit CPU seams.
 * A failed join must be fatal before an ignored return permits reclaim.
 */
#include "group.h"
#include "debug.h"
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
int ncclDebugLevel=0;unsigned long ncclDebugMask=0;thread_local int ncclDebugNoWarn=0;
void ncclDebugLog(ncclDebugLogLevel,unsigned long,const char*,int,const char*,...){}
__thread int ncclGroupDepth=0;
__thread ncclResult_t ncclGroupError=ncclSuccess;
namespace latency_profiler { CollTrace::~CollTrace(){std::abort();} }
namespace rccl {
/** @brief Explicit inactive trace boundary; never performs native setup work. */
struct ReplayRecorder {
 static ReplayRecorder& instance(){static ReplayRecorder recorder;return recorder;}
 template<class... T>ncclResult_t record(T&&...){return ncclSuccess;}
};
}
using namespace rccl;
struct Facts {bool denyJoin=false;int joins=0,allocations=0,reclaims=0,destructors=0;} facts;
int fatalPipe=-1;ncclGroupJob* fatalGroup=nullptr;
extern "C" [[noreturn]] void __real_abort();
extern "C" [[noreturn]] void __wrap_abort()
{
 if(fatalPipe>=0){
  bool retained=fatalGroup&&facts.joins==1&&facts.allocations==0&&facts.reclaims==0&&facts.destructors==0&&fatalGroup->groupRefCount==2;
#if LLAMINAR_HAS_THREAD_RECEIPT
  retained=retained&&ncclLlaminarLoadThreadOwnership(&fatalGroup->base)==ncclNativeThreadOwnership::JoinFailed;
#endif
  const char result=retained?'Y':'N';if(write(fatalPipe,&result,1)!=1)_exit(3);
 }
 __real_abort();
}
int replayJoin(pthread_t,void**){++facts.joins;return facts.denyJoin?EINVAL:0;}
#define pthread_join replayJoin
#include "actual-destructor.inc"
#include "actual-complete.inc"
#include "actual-group-abort.inc"
#undef pthread_join
ncclResult_t ncclCommGetAsyncError(ncclComm_t,ncclResult_t* result){*result=ncclSystemError;return ncclSuccess;}
#include "actual-ready.inc"
#include "actual-set-abort.inc"
#include "actual-finalize-job.inc"
ncclResult_t ncclGroupStartInternal(){++ncclGroupDepth;return ncclSuccess;}
ncclResult_t ncclGroupEndInternal(ncclSimInfo_t*){--ncclGroupDepth;return ncclSuccess;}
static ncclResult_t commReclaim(ncclAsyncJob*){++facts.reclaims;return ncclSuccess;}
/** @brief Observe actual post-gate publication without freeing a live native comm. */
ncclResult_t ncclAsyncLaunch(ncclAsyncJob* job,ncclResult_t(*func)(ncclAsyncJob*),void(*)(ncclAsyncJob*),void(*destroy)(void*),ncclComm_t)
{const auto status=func(job);if(destroy)destroy(job);return status;}
template<class T>ncclResult_t replayCalloc(T** output,size_t count)
{++facts.allocations;*output=static_cast<T*>(std::calloc(count,sizeof(T)));return *output?ncclSuccess:ncclSystemError;}
#define Recorder ReplayRecorder
#define ncclCalloc replayCalloc
#undef NVTX3_RANGE
#define NVTX3_RANGE(...)
#undef NVTX3_RANGE_ADD_PAYLOAD
#define NVTX3_RANGE_ADD_PAYLOAD(...)
#include "actual-comm-abort.inc"
#undef ncclCalloc
#undef Recorder
void observeDestructor(void*){++facts.destructors;}
struct Fixture {
 ncclComm comm{};ncclGroupJob group{};uint32_t host=0,device=0;
 Fixture(){comm.abortFlag=&host;comm.abortFlagDev=&device;comm.groupJob=&group;comm.nRanks=2;
  group.nonBlockingInit=true;group.groupRefCount=2;group.base.result=ncclSystemError;group.base.destructor=observeDestructor;
#if LLAMINAR_HAS_THREAD_RECEIPT
  ncclLlaminarStoreThreadOwnership(&group.base,ncclNativeThreadOwnership::Created);
#endif
 }
};
int main()
{
 const rlimit noCore{0,0};if(setrlimit(RLIMIT_CORE,&noCore))return 2;
 int violations=0;
 for(int replay=0;replay<20;++replay){facts={};Fixture fixture;const auto status=ncclCommAbort_impl(&fixture.comm);
  if(status!=ncclSuccess||facts.joins!=1||facts.allocations!=1||facts.reclaims!=1||facts.destructors!=1||fixture.group.groupRefCount!=1)++violations;
 }
 for(int replay=0;replay<20;++replay){int report[2];if(pipe(report))return 2;const auto child=fork();if(child<0)return 2;
  if(!child){close(report[0]);facts={};facts.denyJoin=true;Fixture fixture;fatalGroup=&fixture.group;fatalPipe=report[1];
   (void)ncclCommAbort_impl(&fixture.comm);
   const char result=facts.reclaims?'N':'R';if(write(report[1],&result,1)!=1)_exit(3);_exit(0);
  }
  close(report[1]);char result=0;const auto bytes=read(report[0],&result,1);close(report[0]);int status=0;
  if(waitpid(child,&status,0)!=child||bytes!=1||result!='Y'||!WIFSIGNALED(status)||WTERMSIG(status)!=SIGABRT)++violations;
 }
 std::cout << "{\"schema\":1,\"suite\":\"ignored-join-caller\",\"cases\":40,\"ordinary_error_cases\":20,\"join_failure_cases\":20,\"violations\":" << violations << "}\n";
 return violations?1:0;
}
