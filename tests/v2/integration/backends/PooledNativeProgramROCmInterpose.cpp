/** @file PooledNativeProgramROCmInterpose.cpp
 * @brief Explicit actual-program diagnostic API observations; original SDK bodies remain authoritative.
 * No production target links these symbols. Cold read-only graph inventory queries
 * are explicit diagnostic overhead; native creation and execution remain unchanged.
 */
#include "PooledNativeProgramObservation.h"
#include "PooledNativeGraphInventory.h"
#include <hip/hip_runtime_api.h>
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
struct ROCmLookupInventory {
    using Graph=hipGraph_t;using Node=hipGraphNode_t;using Kind=hipGraphNodeType;
    static constexpr auto success=hipSuccess;
    static constexpr const char* backend="ROCm";
    static constexpr const char* operation="GraphInstantiate";
    /** @brief Enumerate the original definition without loading another runtime or launching work. */
    static auto nodes(Graph graph,Node* values,std::size_t* count){
        static const auto invoke=original<decltype(&hipGraphGetNodes)>(LLAMINAR_PROGRAM_SYMBOL(hipGraphGetNodes));
        return invoke(graph,values,count);
    }
    /** @brief Require all actual default dependency edges; lossy CUDA queries fail the diagnostic. */
    static auto edges(Graph graph,Node* from,Node* to,std::size_t* count){
        static const auto invoke=original<decltype(&hipGraphGetEdges)>(LLAMINAR_PROGRAM_SYMBOL(hipGraphGetEdges));
        return invoke(graph,from,to,count);
    }
    /** @brief Read the actual enum kind instead of inferring it from an observed kernel subset. */
    static auto kind(Node node,Kind* value){
        static const auto invoke=original<decltype(&hipGraphNodeGetType)>(LLAMINAR_PROGRAM_SYMBOL(hipGraphNodeGetType));
        return invoke(node,value);
    }
    /** @brief Closed flat lookup scope; child/control/host/allocation programs need separate evidence. */
    static bool flatLookupKind(Kind value)noexcept{
        switch(value){
        case hipGraphNodeTypeKernel:case hipGraphNodeTypeMemcpy:case hipGraphNodeTypeMemset:
        case hipGraphNodeTypeEmpty:case hipGraphNodeTypeWaitEvent:case hipGraphNodeTypeEventRecord:return true;
        default:return false;
        }
    }
};
}

/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" hipError_t hipStreamBeginCapture(hipStream_t stream,hipStreamCaptureMode mode){
    static const auto invoke=original<decltype(&hipStreamBeginCapture)>(LLAMINAR_PROGRAM_SYMBOL(hipStreamBeginCapture));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("ROCm","StreamBeginCapture",ordinal);
    const auto* caller=__builtin_return_address(0);record("ROCm","StreamBeginCapture","before",ordinal,stream,caller,0,sample(),stream,static_cast<std::uint64_t>(mode));
    const auto result=invoke(stream,mode);
    record("ROCm","StreamBeginCapture","after",ordinal,stream,caller,static_cast<int>(result),sample(),stream,static_cast<std::uint64_t>(mode));return result;
}
/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" hipError_t hipStreamEndCapture(hipStream_t stream,hipGraph_t* graph){
    static const auto invoke=original<decltype(&hipStreamEndCapture)>(LLAMINAR_PROGRAM_SYMBOL(hipStreamEndCapture));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("ROCm","StreamEndCapture",ordinal);
    const auto* caller=__builtin_return_address(0);record("ROCm","StreamEndCapture","before",ordinal,stream,caller,0,sample(),stream,0);
    const auto result=invoke(stream,graph);
    record("ROCm","StreamEndCapture","after",ordinal,result==hipSuccess?*graph:nullptr,caller,static_cast<int>(result),sample(),stream,0);return result;
}
/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" hipError_t hipGraphLaunch(hipGraphExec_t exec,hipStream_t stream){
    static const auto invoke=original<decltype(&hipGraphLaunch)>(LLAMINAR_PROGRAM_SYMBOL(hipGraphLaunch));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("ROCm","GraphLaunch",ordinal);
    const auto* caller=__builtin_return_address(0);record("ROCm","GraphLaunch","before",ordinal,exec,caller,0,sample(),stream,0);
    const auto result=invoke(exec,stream);
    record("ROCm","GraphLaunch","after",ordinal,exec,caller,static_cast<int>(result),sample(),stream,0);return result;
}
/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" hipError_t hipEventRecord(hipEvent_t event,hipStream_t stream){
    static const auto invoke=original<decltype(&hipEventRecord)>(LLAMINAR_PROGRAM_SYMBOL(hipEventRecord));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("ROCm","EventRecord",ordinal);
    const auto* caller=__builtin_return_address(0);record("ROCm","EventRecord","before",ordinal,event,caller,0,sample(),stream,0);
    const auto result=invoke(event,stream);
    record("ROCm","EventRecord","after",ordinal,event,caller,static_cast<int>(result),sample(),stream,0);return result;
}
/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" hipError_t hipStreamWaitEvent(hipStream_t stream,hipEvent_t event,unsigned flags){
    static const auto invoke=original<decltype(&hipStreamWaitEvent)>(LLAMINAR_PROGRAM_SYMBOL(hipStreamWaitEvent));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("ROCm","StreamWaitEvent",ordinal);
    const auto* caller=__builtin_return_address(0);record("ROCm","StreamWaitEvent","before",ordinal,event,caller,0,sample(),stream,static_cast<std::uint64_t>(flags));
    const auto result=invoke(stream,event,flags);
    record("ROCm","StreamWaitEvent","after",ordinal,event,caller,static_cast<int>(result),sample(),stream,static_cast<std::uint64_t>(flags));return result;
}
/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" hipError_t hipGraphExecDestroy(hipGraphExec_t exec){
    static const auto invoke=original<decltype(&hipGraphExecDestroy)>(LLAMINAR_PROGRAM_SYMBOL(hipGraphExecDestroy));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("ROCm","GraphExecDestroy",ordinal);
    const auto* caller=__builtin_return_address(0);record("ROCm","GraphExecDestroy","before",ordinal,exec,caller,0,sample(),nullptr,0);
    const auto result=invoke(exec);
    record("ROCm","GraphExecDestroy","after",ordinal,exec,caller,static_cast<int>(result),sample(),nullptr,0);return result;
}
/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" hipError_t hipGraphDestroy(hipGraph_t graph){
    static const auto invoke=original<decltype(&hipGraphDestroy)>(LLAMINAR_PROGRAM_SYMBOL(hipGraphDestroy));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("ROCm","GraphDestroy",ordinal);
    const auto* caller=__builtin_return_address(0);record("ROCm","GraphDestroy","before",ordinal,graph,caller,0,sample(),nullptr,0);
    const auto result=invoke(graph);
    record("ROCm","GraphDestroy","after",ordinal,graph,caller,static_cast<int>(result),sample(),nullptr,0);return result;
}
/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" hipError_t hipDeviceReset(){
    static const auto invoke=original<decltype(&hipDeviceReset)>(LLAMINAR_PROGRAM_SYMBOL(hipDeviceReset));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("ROCm","DeviceReset",ordinal);
    const auto* caller=__builtin_return_address(0);record("ROCm","DeviceReset","before",ordinal,nullptr,caller,0,sample(),nullptr,0);
    const auto result=invoke();
    record("ROCm","DeviceReset","after",ordinal,nullptr,caller,static_cast<int>(result),sample(),nullptr,0);return result;
}
/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" hipError_t hipStreamCreateWithFlags(hipStream_t* stream,unsigned flags){
    static const auto invoke=original<decltype(&hipStreamCreateWithFlags)>(LLAMINAR_PROGRAM_SYMBOL(hipStreamCreateWithFlags));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("ROCm","StreamCreateWithFlags",ordinal);
    const auto* caller=__builtin_return_address(0);record("ROCm","StreamCreateWithFlags","before",ordinal,nullptr,caller,0,sample(),nullptr,static_cast<std::uint64_t>(flags));
    const auto result=invoke(stream,flags);
    record("ROCm","StreamCreateWithFlags","after",ordinal,result==hipSuccess?*stream:nullptr,caller,static_cast<int>(result),sample(),nullptr,static_cast<std::uint64_t>(flags));return result;
}
/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" hipError_t hipEventCreateWithFlags(hipEvent_t* event,unsigned flags){
    static const auto invoke=original<decltype(&hipEventCreateWithFlags)>(LLAMINAR_PROGRAM_SYMBOL(hipEventCreateWithFlags));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("ROCm","EventCreateWithFlags",ordinal);
    const auto* caller=__builtin_return_address(0);record("ROCm","EventCreateWithFlags","before",ordinal,nullptr,caller,0,sample(),nullptr,static_cast<std::uint64_t>(flags));
    const auto result=invoke(event,flags);
    record("ROCm","EventCreateWithFlags","after",ordinal,result==hipSuccess?*event:nullptr,caller,static_cast<int>(result),sample(),nullptr,static_cast<std::uint64_t>(flags));return result;
}
/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" hipError_t hipEventDestroy(hipEvent_t event){
    static const auto invoke=original<decltype(&hipEventDestroy)>(LLAMINAR_PROGRAM_SYMBOL(hipEventDestroy));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("ROCm","EventDestroy",ordinal);
    const auto* caller=__builtin_return_address(0);record("ROCm","EventDestroy","before",ordinal,event,caller,0,sample(),nullptr,0);
    const auto result=invoke(event);
    record("ROCm","EventDestroy","after",ordinal,event,caller,static_cast<int>(result),sample(),nullptr,0);return result;
}
/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" hipError_t hipStreamDestroy(hipStream_t stream){
    static const auto invoke=original<decltype(&hipStreamDestroy)>(LLAMINAR_PROGRAM_SYMBOL(hipStreamDestroy));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("ROCm","StreamDestroy",ordinal);
    const auto* caller=__builtin_return_address(0);record("ROCm","StreamDestroy","before",ordinal,stream,caller,0,sample(),stream,0);
    const auto result=invoke(stream);
    record("ROCm","StreamDestroy","after",ordinal,stream,caller,static_cast<int>(result),sample(),stream,0);return result;
}
/** @brief Preserve the original SDK arguments/result while labeling both host-only observation boundaries. */
extern "C" hipError_t hipGraphInstantiate(hipGraphExec_t* exec,hipGraph_t graph,hipGraphNode_t* errorNode,char* log,std::size_t logBytes){
    static const auto invoke=original<decltype(&hipGraphInstantiate)>(LLAMINAR_PROGRAM_SYMBOL(hipGraphInstantiate));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("ROCm","GraphInstantiate",ordinal);
    const auto* caller=__builtin_return_address(0);record("ROCm","GraphInstantiate","before",ordinal,graph,caller,0,sample(),nullptr,0);
    inspectFlatLookupGraph<ROCmLookupInventory>(graph,ordinal);
    const auto result=rejectInstantiation(ordinal,"ROCm")?hipErrorOutOfMemory:invoke(exec,graph,errorNode,log,logBytes);
    record("ROCm","GraphInstantiate","after",ordinal,result==hipSuccess?*exec:nullptr,caller,static_cast<int>(result),sample(),nullptr,0);return result;
}

/** @brief Observe only the original SDK's existing captured-graph enumeration; never request another native inspection. */
extern "C" hipError_t hipGraphGetNodes(hipGraph_t graph,hipGraphNode_t* nodes,std::size_t* count){
    static const auto invoke=original<decltype(&hipGraphGetNodes)>(LLAMINAR_PROGRAM_SYMBOL(hipGraphGetNodes));
    const auto capacity=nodes && count?*count:0;const auto result=invoke(graph,nodes,count);
    if(result==hipSuccess && nodes && count){
        if(*count>capacity)std::abort();
        for(std::size_t i=0;i<*count;++i)topology("ROCm","member",graph,nodes[i],-1,0);
    }else if(result!=hipSuccess)topology("ROCm","enumeration_failed",graph,nullptr,-1,static_cast<int>(result));
    return result;
}
/** @brief Retain actual native enum kinds as metadata, bound to the selected SDK; no inferred generic host certificate. */
extern "C" hipError_t hipGraphNodeGetType(hipGraphNode_t node,hipGraphNodeType* kind){
    static const auto invoke=original<decltype(&hipGraphNodeGetType)>(LLAMINAR_PROGRAM_SYMBOL(hipGraphNodeGetType));
    const auto result=invoke(node,kind);topology("ROCm","kind",nullptr,node,result==hipSuccess && kind?static_cast<int>(*kind):-1,static_cast<int>(result));return result;
}

/** @brief Observe the exact original EventCreate API, including real SDK forwarding ancestry. */
extern "C" hipError_t hipEventCreate(hipEvent_t* event){
    static const auto invoke=original<decltype(&hipEventCreate)>(LLAMINAR_PROGRAM_SYMBOL(hipEventCreate));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("ROCm","EventCreate",ordinal);
    const auto* caller=__builtin_return_address(0);record("ROCm","EventCreate","before",ordinal,nullptr,caller,0,sample(),nullptr,0);
    const auto result=invoke(event);
    record("ROCm","EventCreate","after",ordinal,result==hipSuccess?*event:nullptr,caller,static_cast<int>(result),sample(),nullptr,0);return result;
}

/** @brief Observe the exact original GraphCreate API, including real SDK forwarding ancestry. */
extern "C" hipError_t hipGraphCreate(hipGraph_t* graph,unsigned flags){
    static const auto invoke=original<decltype(&hipGraphCreate)>(LLAMINAR_PROGRAM_SYMBOL(hipGraphCreate));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("ROCm","GraphCreate",ordinal);
    const auto* caller=__builtin_return_address(0);record("ROCm","GraphCreate","before",ordinal,nullptr,caller,0,sample(),nullptr,static_cast<std::uint64_t>(flags));
    const auto result=invoke(graph,flags);
    record("ROCm","GraphCreate","after",ordinal,result==hipSuccess?*graph:nullptr,caller,static_cast<int>(result),sample(),nullptr,static_cast<std::uint64_t>(flags));return result;
}

/** @brief Observe the exact original GraphClone API, including real SDK forwarding ancestry. */
extern "C" hipError_t hipGraphClone(hipGraph_t* clone,hipGraph_t graph){
    static const auto invoke=original<decltype(&hipGraphClone)>(LLAMINAR_PROGRAM_SYMBOL(hipGraphClone));
    static std::atomic<std::uint64_t> next{0};const auto ordinal=next.fetch_add(1,std::memory_order_relaxed)+1;
    CallScope observation("ROCm","GraphClone",ordinal);
    const auto* caller=__builtin_return_address(0);record("ROCm","GraphClone","before",ordinal,graph,caller,0,sample(),nullptr,0);
    const auto result=invoke(clone,graph);
    record("ROCm","GraphClone","after",ordinal,result==hipSuccess?*clone:nullptr,caller,static_cast<int>(result),sample(),nullptr,0);return result;
}
