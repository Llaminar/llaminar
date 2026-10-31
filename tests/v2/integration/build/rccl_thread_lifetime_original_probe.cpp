/**
 * @file rccl_thread_lifetime_original_probe.cpp
 * @brief Literal native RCCL partial pthread creation and destructor ordering replay.
 * Native async launch, worker and cleanup bodies are included unchanged. Only
 * pthread operations are deterministic device-free seams; no thread race or
 * freed-memory dereference is required to expose the missing join obligation.
 */
#include "group.h"
#include "debug.h"
#include <array>
#include <cerrno>
#include <iostream>
#include <stdexcept>
int ncclDebugLevel=0;unsigned long ncclDebugMask=0;thread_local int ncclDebugNoWarn=0;
void ncclDebugLog(ncclDebugLogLevel,unsigned long,const char*,int,const char*,...){}
using Queue=ncclIntruQueue<ncclAsyncJob,&ncclAsyncJob::next>;
struct Launch {pthread_t thread{};void*(*entry)(void*)=nullptr;void* argument=nullptr;bool entered=false,joined=false;};
struct Facts {bool eager=false;int creates=0,joins=0,executions=0,destructors=0,unsafe=0;std::array<Launch,2> launches{};} facts;
int replayCreate(pthread_t* output,const pthread_attr_t*,void*(*entry)(void*),void* argument)
{
 const int index=facts.creates++;
 if(index==1)return EAGAIN;
 *output=100+index;facts.launches[index]={*output,entry,argument,false,false};
 if(facts.eager){facts.launches[index].entered=true;entry(argument);}
 return 0;
}
int replayJoin(pthread_t thread,void**)
{
 for(auto& launch:facts.launches)if(launch.thread==thread&&launch.entry){
  if(launch.joined)std::abort();
  if(!launch.entered){launch.entered=true;launch.entry(launch.argument);}
  launch.joined=true;++facts.joins;return 0;
 }
 return EINVAL;
}
#define pthread_create replayCreate
#define pthread_join replayJoin
#include "actual-worker.inc"
#include "actual-async-launch.inc"
#undef pthread_create
#undef pthread_join
ncclResult_t ncclCommSetAsyncError(ncclComm_t,ncclResult_t){std::abort();}
void cleanup(Queue* asyncJobsPtr,ncclResult_t error)
{
#include "actual-cleanup.inc"
}
ncclResult_t execute(ncclAsyncJob*){++facts.executions;return ncclSuccess;}
void destroy(void* source)
{
 ++facts.destructors;
 for(const auto& launch:facts.launches)
  if(launch.argument==source&&!launch.joined)++facts.unsafe;
}
int main()
{
 int violations=0;
 for(int mode=0;mode<2;++mode)for(int replay=0;replay<20;++replay){
  facts={};facts.eager=mode;std::array<ncclAsyncJob,2> jobs{};Queue queue;ncclIntruQueueConstruct(&queue);
  for(auto& job:jobs){job.func=execute;job.destructor=destroy;job.destroyFlag=1;job.state=ncclGroupJobRunning;ncclIntruQueueEnqueue(&queue,&job);}
  bool abort=false;const auto status=asyncJobLaunch(&queue,&abort);
  if(status!=ncclSystemError)throw std::runtime_error("Expected the actual later create failure");
  cleanup(&queue,status);
  if(facts.unsafe||facts.joins!=1||facts.executions!=1||facts.destructors!=2)++violations;
  // After reporting, complete the deliberately deferred CPU seam so the fixture
  // itself owns no outstanding execution even when the literal source is red.
  for(auto& launch:facts.launches)if(launch.entry&&!launch.joined)replayJoin(launch.thread,nullptr);
 }
 std::cout << "{\"schema\":1,\"suite\":\"created-thread\",\"cases\":40,\"real_thread_cases\":0,\"join_failure_cases\":0,\"violations\":" << violations << "}\n";
 return violations?1:0;
}
