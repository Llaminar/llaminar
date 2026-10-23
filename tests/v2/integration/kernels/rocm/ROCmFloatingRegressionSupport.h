/**
 * @file ROCmFloatingRegressionSupport.h
 * @brief Shared existing captured floating regression storage/stream/event owners.
 *
 * These are the original reduction fixture owners, extracted without changing
 * behavior so immediate-original pipeline tests share their retirement contract.
 */
#pragma once
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <vector>
namespace qfn_reduction_regression {
/** @brief Turn every native infrastructure failure into a visible test failure. */
inline void checked(hipError_t status) {
    if(status!=hipSuccess)throw std::runtime_error(hipGetErrorString(status));
}
/** @brief Setup-owned device allocation; destroyed after graph and terminal observation. */
template<class T> struct Allocation {
    T* p=nullptr;
    explicit Allocation(size_t count){checked(hipMalloc(reinterpret_cast<void**>(&p),count*sizeof(T)));}
    ~Allocation(){if(p)(void)hipFree(p);}
    Allocation(const Allocation&)=delete;
    Allocation& operator=(const Allocation&)=delete;
};
/** @brief Retain one captured executable and its exact explicit stream. */
struct Captured {
    hipStream_t stream=nullptr;
    hipGraph_t graph=nullptr;
    hipGraphExec_t exec=nullptr;
    Captured(){checked(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));}
    ~Captured(){if(stream)(void)hipStreamSynchronize(stream);if(exec)(void)hipGraphExecDestroy(exec);
        if(graph)(void)hipGraphDestroy(graph);if(stream)(void)hipStreamDestroy(stream);}
    template<class F> void capture(F launch,int copies=1){
        checked(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal));
        for(int i=0;i<copies;++i)launch(stream);
        checked(hipStreamEndCapture(stream,&graph));
        checked(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    }
};
/** @brief Native event pair isolates a retained batch of kernels without launch-loop timing. */
struct Events {
    hipEvent_t first=nullptr,last=nullptr;
    Events(){checked(hipEventCreate(&first));checked(hipEventCreate(&last));}
    ~Events(){if(last)(void)hipEventDestroy(last);if(first)(void)hipEventDestroy(first);}
    float sample(Captured& graph,int copies){
        checked(hipEventRecord(first,graph.stream));checked(hipGraphLaunch(graph.exec,graph.stream));
        checked(hipEventRecord(last,graph.stream));checked(hipEventSynchronize(last));
        float milliseconds=0;checked(hipEventElapsedTime(&milliseconds,first,last));
        return milliseconds*1000.f/copies;
    }
};
/** @brief Copy setup/test observation bytes only outside captured execution. */
template<class T> void upload(T* target,const std::vector<T>& source,hipStream_t stream){
    checked(hipMemcpyAsync(target,source.data(),source.size()*sizeof(T),hipMemcpyHostToDevice,stream));
}
}
