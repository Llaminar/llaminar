/**
 * @file FloatingRequestProjectionTest.h
 * @brief Captured production floating projections preserve request-local holes.
 *
 * The independent old reduction computes the original full matrix once. Native
 * count words then change across retained replays; inactive inputs are NaNs and
 * dirty outputs must retain their previous bytes. Storage is setup-owned.
 */
#pragma once
#include <gtest/gtest.h>
#include "kernels/common/FloatingPointVerifierLaunch.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#define RT(name) hip##name
#define FP32_LAUNCH rocmFp32_small_n_batched_projection
#define FP16_LAUNCH rocmFp32x16_batched_projection
#define REQUEST_SUITE ROCmFloatingRequestProjection
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define RT(name) cuda##name
#define FP32_LAUNCH cudaFp32_tiny_batched_projection
#define FP16_LAUNCH cudaFp32x16_tiny_batched_projection
#define REQUEST_SUITE CUDAFloatingRequestProjection
#endif
namespace {
/** @brief Report native failures at the exact fixture boundary. */
void requestChecked(RT(Error_t) status)
{
    if (status != RT(Success)) throw std::runtime_error(RT(GetErrorString)(status));
}
/** @brief Keep admitted storage alive until the captured graph is destroyed. */
template<class T> class RequestStorage {
public:
    /** @brief Allocate only before graph capture. */
    explicit RequestStorage(size_t n) { requestChecked(RT(Malloc)(&data,n*sizeof(T))); }
    /** @brief Release only after graph retirement. */
    ~RequestStorage() { (void)RT(Free)(data); }
    RequestStorage(const RequestStorage&)=delete;
    RequestStorage& operator=(const RequestStorage&)=delete;
    T* data=nullptr;
};
/** @brief Decode original native source bytes independently of production readers. */
__device__ float requestWeight(const void* b, size_t i, int type)
{
    if (type==0) return static_cast<const float*>(b)[i];
    uint32_t bits=static_cast<const uint16_t*>(b)[i];
    if (type==2) return __uint_as_float(bits<<16);
    const uint32_t sign=(bits&0x8000u)<<16;
    int exponent=(bits>>10)&31; uint32_t mantissa=bits&1023;
    if (exponent==31) return __uint_as_float(sign|0x7f800000u|(mantissa<<13));
    if (exponent==0) {
        if (!mantissa) return __uint_as_float(sign);
        exponent=1; while (!(mantissa&1024)) { mantissa<<=1; --exponent; }
        mantissa&=1023;
    }
    return __uint_as_float(sign|(uint32_t(exponent+112)<<23)|(mantissa<<13));
}
/** @brief Independent serial-row arithmetic, including the fixed 256-lane tree. */
__global__ void requestLegacy(const float* a,const void* b,float* c,int n,int k,int type)
{
    float sum=0;
    for (int i=threadIdx.x;i<k;i+=256)
        sum+=a[size_t(blockIdx.y)*k+i]*requestWeight(b,size_t(blockIdx.x)*k+i,type);
    __shared__ float parts[256]; parts[threadIdx.x]=sum; __syncthreads();
    for (int width=128;width;width>>=1) {
        if (threadIdx.x<width) parts[threadIdx.x]+=parts[threadIdx.x+width];
        __syncthreads();
    }
    if (!threadIdx.x) c[size_t(blockIdx.y)*n+blockIdx.x]=parts[0];
}
/** @brief Capture once and vary only canonical execution-owned count words. */
void verifyRequestProjection(int type)
{
    constexpr int requests=3,stride=173,m=requests*stride,n=33,k=257;
    // Boundaries 173/346 cut through both four- and eight-row shared CUDA tiles.
    RequestStorage<float> a(size_t(m)*k),b(size_t(n)*k),out(size_t(m)*n),gold(size_t(m)*n);
    RequestStorage<int32_t> counts(requests);
    RequestStorage<const float*> aa(1),bb(1); RequestStorage<float*> cc(1);
    RT(Stream_t) stream=nullptr; requestChecked(RT(StreamCreateWithFlags)(&stream,RT(StreamNonBlocking)));
    std::vector<float> input(size_t(m)*k),weights(size_t(n)*k),expected(size_t(m)*n),actual(expected.size());
    std::vector<uint16_t> native(weights.size()); uint32_t rng=1977;
    auto next=[&](){rng=rng*1664525u+1013904223u;return float(int(rng>>12)-524288)/524288.0f;};
    for (auto& v:input) v=next(); for (auto& v:weights) v=next();
    for (size_t i=0;i<native.size();++i) {
        if (type==1) native[i]=__half_as_ushort(__float2half_rn(weights[i]));
        else { uint32_t u; std::memcpy(&u,&weights[i],4); native[i]=uint16_t((u+0x7fffu+((u>>16)&1))>>16); }
    }
    requestChecked(RT(MemcpyAsync)(a.data,input.data(),input.size()*4,RT(MemcpyHostToDevice),stream));
    requestChecked(RT(MemcpyAsync)(b.data,type==0?static_cast<const void*>(weights.data()):native.data(),weights.size()*(type==0?4:2),RT(MemcpyHostToDevice),stream));
    const float* ap=a.data; const float* bp=b.data; float* cp=out.data;
    requestChecked(RT(MemcpyAsync)(aa.data,&ap,sizeof(ap),RT(MemcpyHostToDevice),stream));
    requestChecked(RT(MemcpyAsync)(bb.data,&bp,sizeof(bp),RT(MemcpyHostToDevice),stream));
    requestChecked(RT(MemcpyAsync)(cc.data,&cp,sizeof(cp),RT(MemcpyHostToDevice),stream));
    requestLegacy<<<dim3(n,m),256,0,stream>>>(a.data,b.data,gold.data,n,k,type);
    requestChecked(RT(GetLastError)());
    requestChecked(RT(MemcpyAsync)(expected.data(),gold.data,expected.size()*4,RT(MemcpyDeviceToHost),stream));
    requestChecked(RT(StreamSynchronize)(stream));
    const auto rows=llaminar2::DeviceRequestRowRanges::deviceCounted(requests,stride,counts.data);
    RT(Graph_t) graph=nullptr; RT(GraphExec_t) exec=nullptr;
    requestChecked(RT(StreamBeginCapture)(stream,RT(StreamCaptureModeThreadLocal)));
    const bool launched=type==0 ? FP32_LAUNCH(aa.data,bb.data,cc.data,m,n,k,1,0,stream,nullptr,&rows)
        : FP16_LAUNCH(aa.data,bb.data,cc.data,m,n,k,1,type-1,0,stream,nullptr,&rows);
    requestChecked(RT(StreamEndCapture)(stream,&graph)); ASSERT_TRUE(launched);
    requestChecked(RT(GraphInstantiate)(&exec,graph,nullptr,nullptr,0));
#if !defined(__HIPCC__)
    size_t nodes=0; requestChecked(cudaGraphGetNodes(graph,nullptr,&nodes)); ASSERT_EQ(nodes,1u);
    cudaGraphNode_t node=nullptr; requestChecked(cudaGraphGetNodes(graph,&node,&nodes));
    cudaKernelNodeParams params{}; requestChecked(cudaGraphKernelNodeGetParams(node,&params));
    EXPECT_EQ(params.gridDim.y,unsigned((m+(type==1?4:8)-1)/(type==1?4:8)));
#endif
    std::vector<float> dirty(expected.size(),-987.25f);
    requestChecked(RT(MemcpyAsync)(out.data,dirty.data(),dirty.size()*4,RT(MemcpyHostToDevice),stream));
    for (const auto lengths: {std::array{173,173,173},std::array{0,1,172},std::array{1,172,0},
                              std::array{172,0,1},std::array{0,0,0},std::array{173,0,173},std::array{0,1,0}}) {
        auto poisoned=input;
        for (int r=0;r<m;++r) if (r%stride>=lengths[r/stride])
            std::fill_n(poisoned.data()+size_t(r)*k,k,std::numeric_limits<float>::quiet_NaN());
        requestChecked(RT(MemcpyAsync)(a.data,poisoned.data(),poisoned.size()*4,RT(MemcpyHostToDevice),stream));
        requestChecked(RT(MemcpyAsync)(counts.data,lengths.data(),sizeof(lengths),RT(MemcpyHostToDevice),stream));
        requestChecked(RT(GraphLaunch)(exec,stream));
        requestChecked(RT(MemcpyAsync)(actual.data(),out.data,actual.size()*4,RT(MemcpyDeviceToHost),stream));
        requestChecked(RT(StreamSynchronize)(stream));
        for (int r=0;r<m;++r) if (r%stride<lengths[r/stride])
            std::copy_n(expected.data()+size_t(r)*n,n,dirty.data()+size_t(r)*n);
        EXPECT_EQ(std::memcmp(actual.data(),dirty.data(),dirty.size()*4),0) << "type="<<type<<" lengths="<<lengths[0]<<','<<lengths[1]<<','<<lengths[2];
    }
    requestChecked(RT(GraphExecDestroy)(exec)); requestChecked(RT(GraphDestroy)(graph));
    requestChecked(RT(StreamDestroy)(stream));
}
}
/** @brief Exercise original native symbols, all source formats, and dirty partial/empty replay. */
TEST(REQUEST_SUITE,CapturedRequestHolesCrossTilesAndPreserveDirtyRows)
{
    requestChecked(RT(SetDevice)(0));
    for (int type=0;type<3;++type) { SCOPED_TRACE(type); verifyRequestProjection(type); }
}
#undef RT
#undef FP32_LAUNCH
#undef FP16_LAUNCH
#undef REQUEST_SUITE
