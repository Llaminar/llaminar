/** @file PooledNativeProgramCUDAInterpose.cpp
 * @brief Explicit actual-program diagnostic API observations; original SDK bodies remain authoritative.
 * No production target links these symbols. Cold read-only graph inventory queries
 * are explicit diagnostic overhead; native creation and execution remain unchanged.
 */
#include "PooledNativeProgramObservation.h"
#include "PooledNativeGraphInventory.h"
#include <cuda_runtime_api.h>
#include <atomic>
#include <dlfcn.h>
#include <cstdlib>
namespace {
/** @brief Cache the exact next SDK definition before taking the measured host snapshot. */
template<class F>F original(const char* name){auto* symbol=::dlsym(RTLD_NEXT,name);if(!symbol)std::abort();return reinterpret_cast<F>(symbol);}
}
#define LLAMINAR_PROGRAM_SYMBOL_I(value) #value
#define LLAMINAR_PROGRAM_SYMBOL(value) LLAMINAR_PROGRAM_SYMBOL_I(value)
using namespace llaminar2::test::pooled_program_observation;
namespace {
/** @brief Exact selected SDK's read-only flat-lookup inventory, used only before diagnostic instantiation. */
struct CUDALookupInventory {
    using Graph=cudaGraph_t;using Node=cudaGraphNode_t;using Kind=cudaGraphNodeType;
    static constexpr auto success=cudaSuccess;
    static constexpr const char* backend="CUDA";
    static constexpr const char* operation="GraphInstantiateWithFlags";
    /** @brief Enumerate the original definition without loading another runtime or launching work. */
    static auto nodes(Graph graph,Node* values,std::size_t* count){
        static const auto invoke=original<decltype(&cudaGraphGetNodes)>(LLAMINAR_PROGRAM_SYMBOL(cudaGraphGetNodes));
        return invoke(graph,values,count);
    }
    /** @brief Require all actual default dependency edges; lossy CUDA queries fail the diagnostic. */
    static auto edges(Graph graph,Node* from,Node* to,std::size_t* count){
        static const auto invoke=original<decltype(&cudaGraphGetEdges)>(LLAMINAR_PROGRAM_SYMBOL(cudaGraphGetEdges));
        return invoke(graph,from,to,nullptr,count);
    }
    /** @brief Read the actual enum kind instead of inferring it from an observed kernel subset. */
    static auto kind(Node node,Kind* value){
        static const auto invoke=original<decltype(&cudaGraphNodeGetType)>(LLAMINAR_PROGRAM_SYMBOL(cudaGraphNodeGetType));
        return invoke(node,value);
    }
    /** @brief Closed flat lookup scope; child/control/host/allocation programs need separate evidence. */
    static bool flatLookupKind(Kind value)noexcept{
        switch(value){
        case cudaGraphNodeTypeKernel:case cudaGraphNodeTypeMemcpy:case cudaGraphNodeTypeMemset:
        case cudaGraphNodeTypeEmpty:case cudaGraphNodeTypeWaitEvent:case cudaGraphNodeTypeEventRecord:return true;
        default:return false;
        }
    }
};
}

/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" cudaError_t CUDARTAPI cudaStreamBeginCapture(cudaStream_t stream,cudaStreamCaptureMode mode){
    static const auto invoke=original<decltype(&cudaStreamBeginCapture)>(LLAMINAR_PROGRAM_SYMBOL(cudaStreamBeginCapture));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("CUDA","StreamBeginCapture",ordinal);
    const auto* caller=__builtin_return_address(0);record("CUDA","StreamBeginCapture","before",ordinal,stream,caller,0,sample(),stream,static_cast<std::uint64_t>(mode));
    const auto result=invoke(stream,mode);
    record("CUDA","StreamBeginCapture","after",ordinal,stream,caller,static_cast<int>(result),sample(),stream,static_cast<std::uint64_t>(mode));return result;
}
/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" cudaError_t CUDARTAPI cudaStreamEndCapture(cudaStream_t stream,cudaGraph_t* graph){
    static const auto invoke=original<decltype(&cudaStreamEndCapture)>(LLAMINAR_PROGRAM_SYMBOL(cudaStreamEndCapture));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("CUDA","StreamEndCapture",ordinal);
    const auto* caller=__builtin_return_address(0);record("CUDA","StreamEndCapture","before",ordinal,stream,caller,0,sample(),stream,0);
    const auto result=invoke(stream,graph);
    record("CUDA","StreamEndCapture","after",ordinal,result==cudaSuccess?*graph:nullptr,caller,static_cast<int>(result),sample(),stream,0);return result;
}
/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t exec,cudaStream_t stream){
    static const auto invoke=original<decltype(&cudaGraphLaunch)>(LLAMINAR_PROGRAM_SYMBOL(cudaGraphLaunch));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("CUDA","GraphLaunch",ordinal);
    const auto* caller=__builtin_return_address(0);record("CUDA","GraphLaunch","before",ordinal,exec,caller,0,sample(),stream,0);
    const auto result=invoke(exec,stream);
    record("CUDA","GraphLaunch","after",ordinal,exec,caller,static_cast<int>(result),sample(),stream,0);return result;
}
/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event,cudaStream_t stream){
    static const auto invoke=original<decltype(&cudaEventRecord)>(LLAMINAR_PROGRAM_SYMBOL(cudaEventRecord));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("CUDA","EventRecord",ordinal);
    const auto* caller=__builtin_return_address(0);record("CUDA","EventRecord","before",ordinal,event,caller,0,sample(),stream,0);
    const auto result=invoke(event,stream);
    record("CUDA","EventRecord","after",ordinal,event,caller,static_cast<int>(result),sample(),stream,0);return result;
}
/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream,cudaEvent_t event,unsigned flags){
    static const auto invoke=original<decltype(&cudaStreamWaitEvent)>(LLAMINAR_PROGRAM_SYMBOL(cudaStreamWaitEvent));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("CUDA","StreamWaitEvent",ordinal);
    const auto* caller=__builtin_return_address(0);record("CUDA","StreamWaitEvent","before",ordinal,event,caller,0,sample(),stream,static_cast<std::uint64_t>(flags));
    const auto result=invoke(stream,event,flags);
    record("CUDA","StreamWaitEvent","after",ordinal,event,caller,static_cast<int>(result),sample(),stream,static_cast<std::uint64_t>(flags));return result;
}
/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" cudaError_t CUDARTAPI cudaGraphExecDestroy(cudaGraphExec_t exec){
    static const auto invoke=original<decltype(&cudaGraphExecDestroy)>(LLAMINAR_PROGRAM_SYMBOL(cudaGraphExecDestroy));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("CUDA","GraphExecDestroy",ordinal);
    const auto* caller=__builtin_return_address(0);record("CUDA","GraphExecDestroy","before",ordinal,exec,caller,0,sample(),nullptr,0);
    const auto result=invoke(exec);
    record("CUDA","GraphExecDestroy","after",ordinal,exec,caller,static_cast<int>(result),sample(),nullptr,0);return result;
}
/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" cudaError_t CUDARTAPI cudaGraphDestroy(cudaGraph_t graph){
    static const auto invoke=original<decltype(&cudaGraphDestroy)>(LLAMINAR_PROGRAM_SYMBOL(cudaGraphDestroy));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("CUDA","GraphDestroy",ordinal);
    const auto* caller=__builtin_return_address(0);record("CUDA","GraphDestroy","before",ordinal,graph,caller,0,sample(),nullptr,0);
    const auto result=invoke(graph);
    record("CUDA","GraphDestroy","after",ordinal,graph,caller,static_cast<int>(result),sample(),nullptr,0);return result;
}
/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" cudaError_t CUDARTAPI cudaDeviceReset(){
    static const auto invoke=original<decltype(&cudaDeviceReset)>(LLAMINAR_PROGRAM_SYMBOL(cudaDeviceReset));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("CUDA","DeviceReset",ordinal);
    const auto* caller=__builtin_return_address(0);record("CUDA","DeviceReset","before",ordinal,nullptr,caller,0,sample(),nullptr,0);
    const auto result=invoke();
    record("CUDA","DeviceReset","after",ordinal,nullptr,caller,static_cast<int>(result),sample(),nullptr,0);return result;
}
/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t* stream,unsigned flags){
    static const auto invoke=original<decltype(&cudaStreamCreateWithFlags)>(LLAMINAR_PROGRAM_SYMBOL(cudaStreamCreateWithFlags));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("CUDA","StreamCreateWithFlags",ordinal);
    const auto* caller=__builtin_return_address(0);record("CUDA","StreamCreateWithFlags","before",ordinal,nullptr,caller,0,sample(),nullptr,static_cast<std::uint64_t>(flags));
    const auto result=invoke(stream,flags);
    record("CUDA","StreamCreateWithFlags","after",ordinal,result==cudaSuccess?*stream:nullptr,caller,static_cast<int>(result),sample(),nullptr,static_cast<std::uint64_t>(flags));return result;
}
/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t* event,unsigned flags){
    static const auto invoke=original<decltype(&cudaEventCreateWithFlags)>(LLAMINAR_PROGRAM_SYMBOL(cudaEventCreateWithFlags));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("CUDA","EventCreateWithFlags",ordinal);
    const auto* caller=__builtin_return_address(0);record("CUDA","EventCreateWithFlags","before",ordinal,nullptr,caller,0,sample(),nullptr,static_cast<std::uint64_t>(flags));
    const auto result=invoke(event,flags);
    record("CUDA","EventCreateWithFlags","after",ordinal,result==cudaSuccess?*event:nullptr,caller,static_cast<int>(result),sample(),nullptr,static_cast<std::uint64_t>(flags));return result;
}
/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event){
    static const auto invoke=original<decltype(&cudaEventDestroy)>(LLAMINAR_PROGRAM_SYMBOL(cudaEventDestroy));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("CUDA","EventDestroy",ordinal);
    const auto* caller=__builtin_return_address(0);record("CUDA","EventDestroy","before",ordinal,event,caller,0,sample(),nullptr,0);
    const auto result=invoke(event);
    record("CUDA","EventDestroy","after",ordinal,event,caller,static_cast<int>(result),sample(),nullptr,0);return result;
}
/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream){
    static const auto invoke=original<decltype(&cudaStreamDestroy)>(LLAMINAR_PROGRAM_SYMBOL(cudaStreamDestroy));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("CUDA","StreamDestroy",ordinal);
    const auto* caller=__builtin_return_address(0);record("CUDA","StreamDestroy","before",ordinal,stream,caller,0,sample(),stream,0);
    const auto result=invoke(stream);
    record("CUDA","StreamDestroy","after",ordinal,stream,caller,static_cast<int>(result),sample(),stream,0);return result;
}
/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" cudaError_t CUDARTAPI cudaGraphInstantiateWithFlags(cudaGraphExec_t* exec,cudaGraph_t graph,unsigned long long flags){
    static const auto invoke=original<decltype(&cudaGraphInstantiateWithFlags)>(LLAMINAR_PROGRAM_SYMBOL(cudaGraphInstantiateWithFlags));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("CUDA","GraphInstantiateWithFlags",ordinal);
    const auto* caller=__builtin_return_address(0);record("CUDA","GraphInstantiateWithFlags","before",ordinal,graph,caller,0,sample(),nullptr,static_cast<std::uint64_t>(flags));
    inspectFlatLookupGraph<CUDALookupInventory>(graph,ordinal);
    const auto result=rejectInstantiation(ordinal,"CUDA")?cudaErrorMemoryAllocation:invoke(exec,graph,flags);
    record("CUDA","GraphInstantiateWithFlags","after",ordinal,result==cudaSuccess?*exec:nullptr,caller,static_cast<int>(result),sample(),nullptr,static_cast<std::uint64_t>(flags));return result;
}

/** @brief Observe only the original SDK's existing captured-graph enumeration; never request another native inspection. */
extern "C" cudaError_t CUDARTAPI cudaGraphGetNodes(cudaGraph_t graph,cudaGraphNode_t* nodes,std::size_t* count){
    static const auto invoke=original<decltype(&cudaGraphGetNodes)>(LLAMINAR_PROGRAM_SYMBOL(cudaGraphGetNodes));
    const auto capacity=nodes && count?*count:0;const auto result=invoke(graph,nodes,count);
    if(result==cudaSuccess && nodes && count){
        if(*count>capacity)std::abort();
        for(std::size_t i=0;i<*count;++i)topology("CUDA","member",graph,nodes[i],-1,0);
    }else if(result!=cudaSuccess)topology("CUDA","enumeration_failed",graph,nullptr,-1,static_cast<int>(result));
    return result;
}
/** @brief Retain actual native enum kinds as metadata, bound to the selected SDK; no inferred generic host certificate. */
extern "C" cudaError_t CUDARTAPI cudaGraphNodeGetType(cudaGraphNode_t node,cudaGraphNodeType* kind){
    static const auto invoke=original<decltype(&cudaGraphNodeGetType)>(LLAMINAR_PROGRAM_SYMBOL(cudaGraphNodeGetType));
    const auto result=invoke(node,kind);topology("CUDA","kind",nullptr,node,result==cudaSuccess && kind?static_cast<int>(*kind):-1,static_cast<int>(result));return result;
}

/** @brief Observe the exact original EventCreate API, including real SDK forwarding ancestry. */
extern "C" cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t* event){
    static const auto invoke=original<decltype(&cudaEventCreate)>(LLAMINAR_PROGRAM_SYMBOL(cudaEventCreate));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("CUDA","EventCreate",ordinal);
    const auto* caller=__builtin_return_address(0);record("CUDA","EventCreate","before",ordinal,nullptr,caller,0,sample(),nullptr,0);
    const auto result=invoke(event);
    record("CUDA","EventCreate","after",ordinal,result==cudaSuccess?*event:nullptr,caller,static_cast<int>(result),sample(),nullptr,0);return result;
}

/** @brief Observe the exact original GraphCreate API, including real SDK forwarding ancestry. */
extern "C" cudaError_t CUDARTAPI cudaGraphCreate(cudaGraph_t* graph,unsigned flags){
    static const auto invoke=original<decltype(&cudaGraphCreate)>(LLAMINAR_PROGRAM_SYMBOL(cudaGraphCreate));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("CUDA","GraphCreate",ordinal);
    const auto* caller=__builtin_return_address(0);record("CUDA","GraphCreate","before",ordinal,nullptr,caller,0,sample(),nullptr,static_cast<std::uint64_t>(flags));
    const auto result=invoke(graph,flags);
    record("CUDA","GraphCreate","after",ordinal,result==cudaSuccess?*graph:nullptr,caller,static_cast<int>(result),sample(),nullptr,static_cast<std::uint64_t>(flags));return result;
}

/** @brief Observe the exact original GraphClone API, including real SDK forwarding ancestry. */
extern "C" cudaError_t CUDARTAPI cudaGraphClone(cudaGraph_t* clone,cudaGraph_t graph){
    static const auto invoke=original<decltype(&cudaGraphClone)>(LLAMINAR_PROGRAM_SYMBOL(cudaGraphClone));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("CUDA","GraphClone",ordinal);
    const auto* caller=__builtin_return_address(0);record("CUDA","GraphClone","before",ordinal,graph,caller,0,sample(),nullptr,0);
    const auto result=invoke(clone,graph);
    record("CUDA","GraphClone","after",ordinal,result==cudaSuccess?*clone:nullptr,caller,static_cast<int>(result),sample(),nullptr,0);return result;
}
