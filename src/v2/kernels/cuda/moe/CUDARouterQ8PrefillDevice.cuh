/**
 * @file CUDARouterQ8PrefillDevice.cuh
 * @brief Ampere INT8 tiles with the serial router's exact block32 reduction tree.
 *
 * Each MMA computes one independently scaled block with exact INT32 products.
 * Register-resident binary folds reproduce the serial FP32 arithmetic. Sparse
 * grids may assign disjoint contiguous subtrees to K warps; their shared-memory
 * join uses the identical parent order. There are no global partials or atomics.
 */
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "kernels/common/DeviceRowPartition.h"
namespace llaminar2::cuda::router_q8_prefill_detail {
/** @brief Add fragments in the prescribed order.
 * @param a Left subtree. @param b Right subtree. @return Rounded FP32 sums. */
__device__ __forceinline__ float4 add(float4 a,float4 b) {
    return make_float4(__fadd_rn(a.x,b.x),__fadd_rn(a.y,b.y),__fadd_rn(a.z,b.z),__fadd_rn(a.w,b.w));
}
/** @brief One exact signed-byte block with INT32 accumulators.
 * @param a0 First A word. @param a1 Second A word. @param a2 Third A word. @param a3 Fourth A word.
 * @param b0 First B word. @param b1 Second B word. @return Four exact block32 dot products. */
__device__ __forceinline__ int4 mma(unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1) {
    int4 d=make_int4(0,0,0,0);
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+r"(d.x),"+r"(d.y),"+r"(d.z),"+r"(d.w)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
#else
    asm volatile("trap;");
#endif
    return d;
}
/**
 * @brief Compute independently owned output tiles with the fixed 128-lane K tree.
 * @tparam WarpRows Number of 16-row warp tiles per CTA.
 * @tparam WarpColumns Number of 8-expert warp tiles per CTA.
 * @tparam LogLeaves Log2 of the nonempty K-partition extent, from 0 through 7.
 * @tparam FixedWidth Constant hidden width, or zero for runtime geometry.
 * @tparam LogKWarps Log2 of cooperating K warps sharing one output tile.
 * @tparam Partitioned Resolve a compact participant-local live row interval.
 * @param hidden_codes Prepared signed INT8 input rows in the gate's basis.
 * @param hidden_scales Their independent block32 scales.
 * @param gate_codes Immutable expert-major INT8 gate rows.
 * @param gate_scales Their independent block32 scales.
 * @param output Compact FP32 logits, retaining the source expert order.
 * @param capacity Retained full-row capacity. @param runtime_width Positive multiple of 32.
 * @param runtime_experts Positive output width. @param live_rows Optional device live extent.
 * @param partition Declared owner, used only by Partitioned launches.
 */
template<int WarpRows,int WarpColumns,int LogLeaves,bool Partitioned=false,int FixedWidth=0,int LogKWarps=0>
__global__ void kernel(const int8_t*__restrict__ hidden_codes,const float*__restrict__ hidden_scales,
    const int8_t*__restrict__ gate_codes,const float*__restrict__ gate_scales,float*__restrict__ output,
    int capacity,int runtime_width,int runtime_experts,const int*live_rows=nullptr,llaminar2::DeviceRowPartition partition={}) {
    static_assert(LogLeaves>=0&&LogLeaves<=7);
    static_assert(LogKWarps>=0&&LogKWarps<=4&&LogKWarps<=LogLeaves);
    static_assert(LogKWarps==0||(WarpRows==1&&WarpColumns==1));
    constexpr int LocalLogLeaves=LogLeaves-LogKWarps;
    const int width=FixedWidth?FixedWidth:runtime_width;
    const int experts=runtime_experts;
    int rows=live_rows?*live_rows:capacity;
    if(rows<0||rows>capacity){__trap();return;}
    if constexpr(Partitioned) {
        const auto local=partition.resolve(capacity,live_rows);
        hidden_codes+=size_t(local.first)*width;hidden_scales+=size_t(local.first)*(width/32);rows=local.count;
    }
    const int warp=(threadIdx.x/32)>>LogKWarps,kgroup=(threadIdx.x/32)&((1<<LogKWarps)-1);
    const int lane=threadIdx.x%32,group=lane/4,t=lane%4;
    const int rbase=blockIdx.y*(16*WarpRows)+(warp/WarpColumns)*16;
    const int nbase=blockIdx.x*(8*WarpColumns)+(warp%WarpColumns)*8;
    if(rbase>=rows||nbase>=experts)return;
    const int r0=rbase+group,r1=r0+8,n0=nbase+t*2,n1=n0+1,nb=nbase+group;
    const int blocks=width/32;
    float4 pending0,pending1,pending2,pending3,pending4,pending5,pending6;
    float4 total=make_float4(0,0,0,0);
#pragma unroll 1
    for(int ordinal=kgroup*(1<<LocalLogLeaves);ordinal<(kgroup+1)*(1<<LocalLogLeaves);++ordinal) {
        // Serial shuffles first reduce lanes within each warp, then warp sums.
        // The warp order is 0,2,1,3 for128 leaves and 0,1 for64 leaves.
        unsigned low=0;
        if constexpr(LogLeaves>0)low=__brev(unsigned(ordinal&31))>>(32-(LogLeaves<5?LogLeaves:5));
        unsigned high=0;
        if constexpr(LogLeaves>5)high=__brev(unsigned(ordinal>>5))>>(32-(LogLeaves-5));
        const int leaf=int(high*32+low);
        float4 value=make_float4(0,0,0,0);
        for(int block=leaf;block<blocks;block+=128) {
            const auto* a0=reinterpret_cast<const unsigned*>(hidden_codes+size_t(r0)*width+block*32);
            const auto* a1=reinterpret_cast<const unsigned*>(hidden_codes+size_t(r1)*width+block*32);
            const auto* b=reinterpret_cast<const unsigned*>(gate_codes+size_t(nb)*width+block*32);
            const unsigned av0=r0<rows?a0[t]:0,av1=r1<rows?a1[t]:0;
            const unsigned av2=r0<rows?a0[t+4]:0,av3=r1<rows?a1[t+4]:0;
            const unsigned bv0=nb<experts?b[t]:0,bv1=nb<experts?b[t+4]:0;
            const int4 d=mma(av0,av1,av2,av3,bv0,bv1);
            const float ha0=r0<rows?hidden_scales[size_t(r0)*blocks+block]:0;
            const float ha1=r1<rows?hidden_scales[size_t(r1)*blocks+block]:0;
            const float wb0=n0<experts?gate_scales[size_t(n0)*blocks+block]:0;
            const float wb1=n1<experts?gate_scales[size_t(n1)*blocks+block]:0;
            value.x=__fmaf_rn(__fmul_rn(float(d.x),ha0),wb0,value.x);
            value.y=__fmaf_rn(__fmul_rn(float(d.y),ha0),wb1,value.y);
            value.z=__fmaf_rn(__fmul_rn(float(d.z),ha1),wb0,value.z);
            value.w=__fmaf_rn(__fmul_rn(float(d.w),ha1),wb1,value.w);
        }
        if constexpr(LocalLogLeaves>0) {
        if((ordinal & 1)==0) pending0=value;
        else {value=add(pending0,value);
            if constexpr(LocalLogLeaves>1) {
                if((ordinal & 2)==0) pending1=value;
                else {value=add(pending1,value);
                if constexpr(LocalLogLeaves>2) {
                    if((ordinal & 4)==0) pending2=value;
                    else {value=add(pending2,value);
                    if constexpr(LocalLogLeaves>3) {
                        if((ordinal & 8)==0) pending3=value;
                        else {value=add(pending3,value);
                        if constexpr(LocalLogLeaves>4) {
                            if((ordinal & 16)==0) pending4=value;
                            else {value=add(pending4,value);
                            if constexpr(LocalLogLeaves>5) {
                                if((ordinal & 32)==0) pending5=value;
                                else {value=add(pending5,value);
                                if constexpr(LocalLogLeaves>6) {
                                    if((ordinal & 64)==0) pending6=value;
                                    else {value=add(pending6,value);
                                    }
                                }
                                }
                            }
                            }
                        }
                        }
                    }
                    }
                }
                }
            }
        }
        }
        total=value;
    }
    // Each K warp owns a contiguous subtree of the bit-reversed leaf order.
    // All participating warps cover the same output tile, so early row/expert
    // exits above are CTA-uniform. Publish every subtree before its sole reader.
    if constexpr(LogKWarps>0) {
        __shared__ float4 partials[(1<<LogKWarps)*32];
        partials[kgroup*32+lane]=total;
        __syncthreads();
        if(kgroup!=0)return;
        total=add(partials[lane],partials[32+lane]);
        if constexpr(LogKWarps>1) {
            total=add(total,add(partials[64+lane],partials[96+lane]));
            if constexpr(LogKWarps>2) {
                const auto right=add(add(partials[128+lane],partials[160+lane]),
                                     add(partials[192+lane],partials[224+lane]));
                total=add(total,right);
                if constexpr(LogKWarps>3) {
                    const auto lower=add(add(partials[256+lane],partials[288+lane]),
                                         add(partials[320+lane],partials[352+lane]));
                    const auto upper=add(add(partials[384+lane],partials[416+lane]),
                                         add(partials[448+lane],partials[480+lane]));
                    total=add(total,add(lower,upper));
                }
            }
        }
    }
    if(r0<rows&&n0<experts)output[size_t(r0)*experts+n0]=total.x;
    if(r0<rows&&n1<experts)output[size_t(r0)*experts+n1]=total.y;
    if(r1<rows&&n0<experts)output[size_t(r1)*experts+n0]=total.z;
    if(r1<rows&&n1<experts)output[size_t(r1)*experts+n1]=total.w;
}
} // namespace llaminar2::cuda::router_q8_prefill_detail
