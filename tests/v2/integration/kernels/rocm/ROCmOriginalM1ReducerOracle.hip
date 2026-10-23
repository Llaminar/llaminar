/**
 * @file ROCmOriginalM1ReducerOracle.hip
 * @brief Test-only frozen ordinary M1 RN reducer and adversarial partial witness.
 *
 * The oracle retains the original pre-grouping reducer expression verbatim.
 * It is linked only into the existing explicit ROCm integration executable;
 * production submission never calls this kernel or the synthetic witness.
 */
#include <hip/hip_runtime.h>
#include "kernels/rocm/gemm/ROCmNativeVNNIDecodeEquivalentMath.h"
#include <cstdint>

namespace
{
/** @brief Frozen original column ownership and ascending-KB RN fold. */
template <int TILE_N,int CPT>
__global__ void originalNativeM1Reducer(const float *partial,float *output,int N,int kblocks)
{
    const int n_base=blockIdx.x*TILE_N+threadIdx.x*CPT;
    #pragma unroll
    for(int c=0;c<CPT;++c)
    {
        const int n=n_base+c;
        if(n>=N)return;
        float sum=0.0f;
        #pragma unroll 16
        for(int kb=0;kb<kblocks;++kb)
            sum=rocm_native_vnni_accumulate_rn(sum,partial[static_cast<std::size_t>(kb)*N+n]);
        output[n]=sum;
    }
}
/** @brief Publish exact synthetic NaN/zero/infinity/cancellation operands after real producer submission. */
__global__ void injectRNPartials(float *partials,int n,int kb)
{
    const auto i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=static_cast<unsigned>(n*kb))return;
    std::uint32_t bits=0;
    switch((i/n+i%n)%8)
    {
    case 1:bits=0x80000000u;break;
    case 2:bits=0x7f800000u;break;
    case 3:bits=0xff800000u;break;
    case 4:bits=0x7fc01234u;break;
    case 5:bits=0x7fc05678u;break;
    case 6:bits=0x3f800000u;break;
    case 7:bits=0xbf800000u;break;
    }
    partials[i]=__uint_as_float(bits);
}
}
/** @brief Launch the independent original reducer at the original issued geometry. */
extern "C" bool testOriginalNativeM1Reducer(const float *partials,float *output,int n,int kb,
    int tile,int cpt,void *stream)
{
    if(!partials || !output || n<=0 || kb<=1 || reinterpret_cast<std::uintptr_t>(stream)<=2)return false;
#define ORIGINAL_REDUCER(TN,CPT) \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(originalNativeM1Reducer<TN,CPT>),dim3(1+(n-1)/TN),dim3(TN/CPT),0,static_cast<hipStream_t>(stream),partials,output,n,kb)
    switch(tile*10+cpt)
    {
    case 641: ORIGINAL_REDUCER(64,1);break;
    case 1281: ORIGINAL_REDUCER(128,1);break;
    case 1282: ORIGINAL_REDUCER(128,2);break;
    case 2561: ORIGINAL_REDUCER(256,1);break;
    case 2562: ORIGINAL_REDUCER(256,2);break;
    default:return false;
    }
#undef ORIGINAL_REDUCER
    return hipPeekAtLastError()==hipSuccess;
}
/** @brief Enqueue only a declared test witness, never an inference arithmetic route. */
extern "C" bool testInjectNativeM1RNPartials(float *partials,int n,int kb,void *stream)
{
    if(!partials || n<=0 || kb<=1 || reinterpret_cast<std::uintptr_t>(stream)<=2)return false;
    hipLaunchKernelGGL(injectRNPartials,dim3((n*kb+255)/256),dim3(256),0,static_cast<hipStream_t>(stream),partials,n,kb);
    return hipPeekAtLastError()==hipSuccess;
}
