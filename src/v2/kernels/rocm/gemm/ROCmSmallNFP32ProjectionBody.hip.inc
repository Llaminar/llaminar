/** @file ROCmSmallNFP32ProjectionBody.hip.inc
 * @brief Original indexed256-lane FP32 row projection shared by standalone and resident HC injection.
 * Load-pair prefetch changes no ordered lane FMA/reduction edge or output ownership.
 * Larger physical CTAs name the original256 members while all lanes join.
 */
#pragma once
namespace llaminar2::rocm::floating_projection_native {
/** @brief Original shared128/64 edges with explicit logical membership.
 * @param value Original lane chain. @param scratch Exact256-float caller LDS.
 * @param lane Original logical lane. @param member Original arithmetic membership.
 * @return Original lane-zero result; other return values are not consumed.
 * @pre All physical lanes participate in every join, including nonmembers.
 */
__device__ __forceinline__ float reduce(float value,float* scratch,unsigned lane,bool member) {
        if(member) scratch[lane] = value;
        __syncthreads();
        for (unsigned stride = 128; stride >= 64; stride >>= 1)
        {
            if (member && lane < stride) scratch[lane] += scratch[lane + stride];
            __syncthreads();
        }
        if (member && lane < 64)
        {
            value = scratch[lane];
#pragma unroll
            for (unsigned stride = 32; stride; stride >>= 1)
            {
                const float partner = __shfl_down(value, stride, 64);
                if (lane < stride) value += partner;
            }
        }
        return member ? value : 0.0f;

}
/** @brief Existing standalone256-member reducer overload.
 * @param value Original lane chain. @param scratch Exact256-float LDS.
 * @return Original lane-zero result.
 */
__device__ __forceinline__ float reduce(float value,float* scratch) {
 return reduce(value,scratch,threadIdx.x,true);
}
/** @brief Original load-pair scheduling with explicit logical256 membership.
 * @tparam LoadPairs Original cold load-pair policy1/4/8.
 * @param input Original physical row. @param weight Original contiguous weight row.
 * @param k Original reduction width. @param scratch Exact256-float caller LDS.
 * @param lane Original logical lane. @param member Original arithmetic membership.
 * @return Original reduced result, consumed only by original lane zero.
 * @pre All physical lanes call, including nonmembers; no caller may wrap this
 * helper in a physical256-thread predicate inside a larger CTA.
 */
template<int LoadPairs>__device__ __forceinline__ float row(const float* input,
 const float* weight,int k,float* scratch,unsigned lane,bool member) {
 static_assert(LoadPairs==1||LoadPairs==4||LoadPairs==8);
 float sum=0.0f;
 if(member) {
  if constexpr(LoadPairs==1) {
   for(int index=static_cast<int>(lane);index<k;index+=256)sum+=input[index]*weight[index];
  } else {
   int index=static_cast<int>(lane);constexpr int Group=LoadPairs*256;const int end=k-k%Group;
   for(;index<end;index+=Group) {
    float inputs[LoadPairs],weights[LoadPairs];
#pragma unroll
    for(int pair=0;pair<LoadPairs;++pair){inputs[pair]=input[index+pair*256];weights[pair]=weight[index+pair*256];}
#pragma unroll
    for(int pair=0;pair<LoadPairs;++pair)sum+=inputs[pair]*weights[pair];
   }
   for(;index<k;index+=256)sum+=input[index]*weight[index];
  }
 }
 return reduce(sum,scratch,lane,member);
}
/** @brief Existing standalone row projection with exactly256 original members.
 * @tparam LoadPairs Original immutable load policy.
 * @param input Original physical row. @param weight Original weight row.
 * @param k Original reduction width. @param scratch Exact256-float caller LDS.
 * @return Original lane-zero result.
 */
template<int LoadPairs>__device__ __forceinline__ float row(const float* input,
 const float* weight,int k,float* scratch) {
 return row<LoadPairs>(input,weight,k,scratch,threadIdx.x,true);
}
}
