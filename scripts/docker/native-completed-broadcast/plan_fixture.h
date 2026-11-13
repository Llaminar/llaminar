/** @file plan_fixture.h
 * @brief CPU-only queue/allocation doubles for literal native planner replay.
 * No GPU/context/PMA/model is created; native source owns all decisions.
 */
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#define MAXCHANNELS 64
#define NCCL_MAX_NTHREADS 640
#define NCCL_MIN_NTHREADS 128
#define WARN(...) ++warnings
int warnings=0;
enum ncclResult_t {ncclSuccess,ncclInvalidUsage};
enum ncclDevWorkType {ncclDevWorkTypeColl,ncclDevWorkTypeCollReg,ncclDevWorkTypeP2p};
enum ncclDevWorkStorageType {ncclDevWorkStorageTypeArgs,ncclDevWorkStorageTypeFifo};
enum class ncclLlaminarPayloadKind {Rows,CompletedBytes,CompletedSourceSlice};
struct ncclDevWorkColl {struct{ncclLlaminarPayloadKind kind;bool present=false;bool active()const{return present;}}llaminarPayload;};
struct alignas(16) ncclWorkList {ncclWorkList*next;ncclDevWorkType workType;int size;};
struct Work{ncclWorkList node;ncclDevWorkColl actual;};
struct ncclDevWorkBatch {unsigned funcId;int nextJump;};
struct ncclWorkBatchList {ncclWorkBatchList*next;ncclDevWorkBatch batch;};
struct ncclProxyOp{ncclProxyOp*next;uint64_t opCount;};
template<class T>struct Queue{T*head=nullptr;};
template<class T>T*ncclIntruQueueHead(Queue<T>*q){return q->head;}
template<class T>bool ncclIntruQueueEmpty(Queue<T>*q){return !q->head;}
template<class T>T*ncclIntruQueueDequeue(Queue<T>*q){T*x=q->head;q->head=x->next;return x;}
template<class T>void ncclIntruQueueEnqueue(Queue<T>*q,T*x){x->next=q->head;q->head=x;}
struct ncclKernelPlanner {struct WipPlan {struct Channel{Queue<ncclWorkBatchList>workBatchQueue;Queue<ncclProxyOp>proxyOpQueue;}channels[MAXCHANNELS];}wipPlan;};
struct ncclComm{ncclKernelPlanner planner;size_t workArgsBytes;int memScoped;void*devComm;int nNodes;};
struct ncclDevKernelArgs{void*comm;uint64_t channelMask;ncclDevWorkStorageType workStorageType;};
struct ncclKernelPlan{bool hasProxyOps=false;bool isSymColl=false;size_t workBytes=0;int nWorkBatches=1;int threadPerBlock=128;ncclDevWorkStorageType workStorageType=ncclDevWorkStorageTypeFifo;size_t kernelArgsSize;ncclDevKernelArgs*kernelArgs;uint64_t channelMask=1;void*kernelFn=reinterpret_cast<void*>(7);bool kernelSpecialized=false;Queue<ncclWorkList>workQueue;Queue<ncclProxyOp>proxyOpQueue;};
alignas(16) unsigned char memory[4096];
void*ncclMemoryStackAlloc(int*,size_t bytes,int){assert(bytes<=sizeof(memory));return memory;}
size_t alignUp(size_t n,int a){return (n+a-1)&~size_t(a-1);}
int popFirstOneBit(uint64_t*p){int n=__builtin_ctzll(*p);*p&=*p-1;return n;}
void*ncclDevKernelList[4096];
