/**
 * @file GPUBranchRMSNormBody.cuh
 * @brief Original HC write, 256-lane norm and ordered read arithmetic.
 *
 * Original standalone and cooperative phases share every arithmetic edge. Exactly
 *256 logical threads enter the reduction; larger physical CTAs explicitly name
 * membership while every physical lane reaches each join. Operand order is fixed.
 */
#pragma once
#include <cstdint>
#if defined(__CUDACC__)
#include "kernels/common/OriginalSingleQ8Block32.cuh"
#endif
namespace llaminar2::qwen4exp::branch_norm_native {
/** @brief Assembly of the first two fixed tree levels. */
enum class SquareSumPublication { SharedEdge128, WaveOwnsBothRoots };

/**
 * @brief Original fixed256-lane sum inside an explicitly named physical CTA.
 * @param partial Original logical-lane FMA sum. @param sums Exact256-float LDS.
 * @param lane Original logical lane0..255 for members; unused otherwise.
 * @param member Whether this physical lane owns an original arithmetic lane.
 * @return Original total for members, zero for nonmembers.
 * @pre All physical lanes call this helper; every original lane has one member.
 * @details Membership gates memory/arithmetic only, never a CTA join. Standalone
 * CUDA uses physical lanes 0 through 31 as the final reduction warp. All
 * callers retain the original logical256 tree and operand order.
 */
template<SquareSumPublication Publication=SquareSumPublication::SharedEdge128>
__device__ __forceinline__ float squareSum(float partial,float* sums,unsigned lane,bool member) {
    if (member) sums[lane] = partial;
    __syncthreads();
#if defined(__HIPCC__)
    if constexpr(Publication==SquareSumPublication::SharedEdge128) {
        // Publish edge128 across waves before wave0 reads both sides of edge64.
        if (member && lane < 128) sums[lane] = __fadd_rn(sums[lane], sums[lane + 128]);
        __syncthreads();
    }
    if (member && lane < 64)
    {
        // Wave0 can read all four published leaves and perform the same
        // edge128 left and right operations locally before combining at edge64.
        // No other wave reads one of these 256 words after the first join.
        float sum;
        if constexpr(Publication==SquareSumPublication::WaveOwnsBothRoots) {
            const float left=__fadd_rn(sums[lane],sums[lane+128]);
            const float right=__fadd_rn(sums[lane+64],sums[lane+192]);
            sum=__fadd_rn(left,right);
        } else {
            sum=__fadd_rn(sums[lane],sums[lane+64]);
        }
#pragma unroll
        for (unsigned stride = 32; stride > 0; stride /= 2)
        {
            const float partner = __shfl_down(sum, stride, 64);
            if (lane < stride) sum = __fadd_rn(sum, partner);
        }
        if (lane == 0) sums[0] = sum;
    }
    // Unlike a lane0-only projection reducer, normalization has consumers in
    // all four waves. Publish lane0's result before any output lane reads it.
    __syncthreads();
#else
    // Warp zero acquires all eight aligned stripes after the producer join.
    // Combine the same left/right operands at strides 128, 64 and 32.
    if (member && lane < 32)
    {
        const float low = __fadd_rn(sums[lane], sums[lane + 128]);
        const float lowPeer = __fadd_rn(sums[lane + 64], sums[lane + 192]);
        const float high = __fadd_rn(sums[lane + 32], sums[lane + 160]);
        const float highPeer = __fadd_rn(sums[lane + 96], sums[lane + 224]);
        float sum = __fadd_rn(__fadd_rn(low, lowPeer), __fadd_rn(high, highPeer));
#pragma unroll
        for (unsigned stride = 16; stride > 0; stride /= 2)
        {
            const float partner = __shfl_down_sync(0xffffffffu, sum, stride);
            if (lane < stride) sum = __fadd_rn(sum, partner);
        }
        if (lane == 0) sums[0] = sum;
    }
    __syncthreads();
#endif
    return member ? sums[0] : 0.0f;

}
/** @brief Original standalone256-lane reduction overload.
 * @param partial Original lane FMA sum. @param sums Exact256-float LDS.
 * @return Original total broadcast to all original output lanes.
 */
__device__ __forceinline__ float squareSum(float partial,float* sums) {
 return squareSum(partial,sums,threadIdx.x,true);
}
/** @brief Original residual-write gate, shared by standalone and fused writers.
 * @param logit Original branch injection value. @return2*sigmoid(logit/4).
 */
__device__ __forceinline__ float residualGate(float logit) {
 return 2.0f*(1.0f/(1.0f+expf(-(logit*0.25f))));
}
/** @brief Original two rounded residual-write operations.
 * @param residual Original retained branch. @param gate Original scaled sigmoid.
 * @param block Original block value. @return RN(residual+RN(gate*block)).
 */
__device__ __forceinline__ float residualValue(float residual,float gate,float block) {
 return __fadd_rn(residual,__fmul_rn(gate,block));
}
/**
 * @brief Append one rounded HC read term in checkpoint branch order.
 * @param sum FP32 accumulator after the preceding branches.
 * @param logit Original completed Up value for this branch and channel.
 * @param normalized Original normalized branch value at this channel.
 * @return One original sigmoid-gated FP32 FMA.
 */
__device__ __forceinline__ float appendReadBranch(float sum, float logit, float normalized)
{
    const float gate = 1.0f / (1.0f + expf(-logit));
    return fmaf(gate, normalized, sum);
}
#if defined(__CUDACC__)
/** @brief Publish the same original CUDA norm value as FP32 and Q8. */
struct CUDAQuantizedPublication
{
    float* normalized;
    std::int8_t* codes;
    float* scales;

    /**
     * @brief Send the rounded norm value to its prepared destinations.
     * @param index Original branch-major output channel.
     * @param value Exact original normalized FP32 value.
     * @pre The owning warp participates with one full K32 block.
     */
    __device__ __forceinline__ void publish(std::uint64_t index, float value) const
    {
        const auto block = index / 32;
        normalized[index] = value;
        original_single_q8::publish(value, true, codes + block * 32, scales + block, nullptr, threadIdx.x % 32);
    }
};
#endif
/** @brief An existing branch's immutable values; publication is already owned. */
struct RetainedBranchInput {
 const float* values;
 /** @brief Consume an original channel. @param channel Logical channel. @return Original bits. */
 __device__ __forceinline__ float publish(unsigned channel) const {return values[channel];}
 /** @brief Reload the same original channel. @param channel Logical channel. @return Original bits. */
 __device__ __forceinline__ float retained(unsigned channel) const {return values[channel];}
};
/** @brief Original rounded write values, retained through the norm's same-lane reload. */
struct ResidualBranchInput {
 const float *residual,*block;float* output;float gate;
 /** @brief Publish one original rounded write before its square FMA.
  * @param channel Original logical channel. @return Exact public FP32 write.
  */
 __device__ __forceinline__ float publish(unsigned channel) const {
  const float value=residualValue(residual[channel],gate,block[channel]);
  output[channel]=value;return value;
 }
 /** @brief Reload only this lane's original public write after the reduction.
  * @param channel Original logical channel. @return Retained rounded write.
  */
 __device__ __forceinline__ float retained(unsigned channel) const {return output[channel];}
};
/**
 * @brief Publish one original normalized value to its existing global output.
 * @param output Original retained normalized output, with its original task base.
 * @param index Original task-major channel coordinate.
 * @param value Already-rounded original normalization/gain result.
 */
__device__ __forceinline__ void publishNormalizedValue(float* output,
 std::uint64_t index,float value) {
 output[index]=value;
}
/**
 * @brief Publish the identical value through an explicit fused consumer policy.
 * @tparam Publication Small by-value policy with publish(index,value).
 * @param publication Borrowed original public output and phase-local consumer.
 * @param index Original task-major channel coordinate.
 * @param value Already-rounded original normalization/gain result.
 * @details The policy changes storage only. It may not recompute normalization
 * or change the original public value. Any cross-lane consumer requires the
 * enclosing phase's explicit publication barrier after this body returns.
 */
template<class Publication>
__device__ __forceinline__ void publishNormalizedValue(const Publication& publication,
 std::uint64_t index,float value) {
 publication.publish(index,value);
}
/**
 * @brief Original RMS inverse, shared by logical and explicit physical output owners.
 * @param sum Completed original256-lane square tree. @param hidden Original branch width.
 * @param epsilon Original RMS epsilon. @return The original rounded inverse expression.
 * @details This helper does not publish or alter the reduction's membership contract.
 */
__device__ __forceinline__ float normalizationInverse(float sum,int hidden,float epsilon) {
 return 1.0f/sqrtf(sum/static_cast<float>(hidden)+epsilon);
}
/** @brief Original compiler-native products; unchanged arithmetic for every default scalar caller. */
struct CompilerNormalizedProducts {
 /** @brief Original inner RN multiply. @param inverse Original inverse. @param post Retained POST. @return RN(post*inverse). */
 __device__ __forceinline__ float inversePost(float inverse,float post) const {return __fmul_rn(post,inverse);}
 /** @brief Original outer RN multiply. @param value Inner result. @param gain Original gain. @return RN(value*gain). */
 __device__ __forceinline__ float appendGain(float value,float gain) const {return __fmul_rn(value,gain);}
};
/**
 * @brief Execute the same two original products through an explicit instruction-binding policy.
 * @tparam Products Default compiler products or a caller's exact native operand binding.
 * @param value Original retained POST. @param inverse Original inverse. @param gain Original gain.
 * @param products Two original RN products only; no reassociation or correction pass.
 * @return The same inner-then-outer normalized element.
 */
template<class Products=CompilerNormalizedProducts>
__device__ __forceinline__ float normalizedElement(float value,float inverse,float gain,Products products={}) {
 return products.appendGain(products.inversePost(inverse,value),gain);
}
/** @brief Existing scalar logical256 output ownership; the default for every original caller. */
struct Logical256NormalizedOutput {
 /**
  * @brief Publish the unchanged original per-lane channel sequence.
  * @tparam Input Original retained input view. @tparam Output Original public/publication view.
  * @param input Stable original POST. @param branch_gain Original branch gain row.
  * @param output Original normalized publication. @param hidden Original branch width.
  * @param epsilon Original epsilon. @param base Original task-major channel base.
  * @param logical_sum Original squareSum return for this arithmetic member.
  * @param completed_total Completed original reduction plane; unused by this policy.
  * @param lane Original logical lane. @param member Original arithmetic membership.
  */
 template<class Input,class Output>
 __device__ __forceinline__ void publish(const Input& input,const float* branch_gain,Output output,
  int hidden,float epsilon,std::uint64_t base,float logical_sum,const float* completed_total,
  unsigned lane,bool member) const {
  (void)completed_total;
  if(member) {
   const float inverse=normalizationInverse(logical_sum,hidden,epsilon);
   for(unsigned channel=lane;channel<static_cast<unsigned>(hidden);channel+=256)
    publishNormalizedValue(output,base+channel,
     normalizedElement(input.retained(channel),inverse,branch_gain[channel]));
  }
 }
};
/**
 * @brief Original norm arithmetic over an explicit branch-local gain view.
 * @tparam Input Borrowed channel source, stable through both complete loops.
 * @tparam Output Original float output or explicit fused publication policy.
 * @tparam OutputWork Explicit output ownership; the original scalar logical256 policy by default.
 * @param input Original branch-local input/publication policy.
 * @param branch_gain Original gain row for this exact branch.
 * @param output Retained normalized rows.
 * @param hidden Original per-branch width. @param epsilon Original RMS epsilon.
 * @param task Original row*4+branch. @param sums Exact256-float LDS.
 * @param lane Original logical lane. @param member Original arithmetic membership.
 * @param output_work Typed publication policy; every physical lane calls it after the tree.
 * @pre Every physical lane participates, even if member is false.
 */
template<class Input,class Output,class OutputWork=Logical256NormalizedOutput,
 SquareSumPublication Publication=SquareSumPublication::SharedEdge128>
__device__ __forceinline__ void taskFromBranchInput(const Input& input,const float* branch_gain,Output output,
 int hidden,float epsilon,std::uint64_t task,float* sums,unsigned lane,bool member,OutputWork output_work={}) {
 const auto base=task*hidden;
 float partial=0.0f;
 if(member)for(unsigned channel=lane;channel<static_cast<unsigned>(hidden);channel+=256) {
  const float value=input.publish(channel);partial=fmaf(value,value,partial);
 }
 const float sum=squareSum<Publication>(partial,sums,lane,member);
 // The tree's nonmember return remains zero. An explicit wider policy may
 // consume completed sums[0] only after squareSum's existing final CTA edge.
 output_work.publish(input,branch_gain,output,hidden,epsilon,base,sum,sums,lane,member);
}
/**
 * @brief Bind the original task's gain row to the shared arithmetic body.
 * @tparam Input Original channel source/publication policy.
 * @tparam Output Original public or typed normalized publication.
 * @param input Original branch input, copied once before its complete loops.
 * @param gain Original four branch gain rows.
 * @param output Retained normalized output/publication.
 * @param hidden Original branch width.
 * @param epsilon Original RMS epsilon.
 * @param task Original task-major branch coordinate.
 * @param sums Exact 256-float reduction plane.
 * @param lane Original logical lane.
 * @param member Whether this physical lane owns original arithmetic.
 * @pre Every physical lane calls and reaches all original reduction joins.
 * @details Selecting a branch-local pointer changes no operand index or FP
 * node. Standalone and staged inputs therefore share one complete numerical
 * implementation rather than maintaining two normalization algorithms.
 */
template<class Input,class Output>
__device__ __forceinline__ void taskFromInput(Input input,const float* gain,Output output,
 int hidden,float epsilon,std::uint64_t task,float* sums,unsigned lane,bool member) {
 taskFromBranchInput(input,gain+(task%4)*hidden,output,hidden,epsilon,task,sums,lane,member);
}
/** @brief Original norm with explicit logical256 membership in a physical CTA.
 * @param input Original branch-major rows. @param gain Original gain rows.
 * @param output Retained FP32 result. @param hidden Original per-branch width.
 * @param epsilon Original RMS epsilon. @param task Original row*4+branch.
 * @param sums Exact256-float LDS. @param lane Original logical lane.
 * @param member Whether this physical lane owns original arithmetic.
 */
__device__ __forceinline__ void task(const float* input,const float* gain,float* output,
 int hidden,float epsilon,std::uint64_t task,float* sums,unsigned lane,bool member) {
 taskFromInput(RetainedBranchInput{input+task*hidden},gain,output,hidden,epsilon,task,sums,lane,member);
}
/** @brief Existing standalone norm, retaining exactly256 original lanes.
 * @param input Original rows. @param gain Original gain. @param output Public result.
 * @param hidden Original width. @param epsilon Original RMS epsilon.
 * @param task Original row*4+branch. @param sums Exact256-float LDS.
 */
__device__ __forceinline__ void task(const float* input,const float* gain,float* output,
 int hidden,float epsilon,std::uint64_t task,float* sums) {
 branch_norm_native::task(input,gain,output,hidden,epsilon,task,sums,threadIdx.x,true);
}
/**
 * @brief Original rounded residual write followed by that same norm body.
 * @tparam Output Original float output or explicit fused publication policy.
 * @param residual Retained four branches. @param block Original H-wide block rows.
 * @param gateLogits Original four injection logits per row.
 * @param written Original public rounded write, optionally exactly residual.
 * @param gain Original branch gains. @param normalized Distinct norm output.
 * @param hidden Original width. @param epsilon Original RMS epsilon.
 * @param task Original row*4+branch. @param sums Exact256-float LDS.
 * @param lane Original logical lane. @param member Original arithmetic membership.
 * @pre All physical lanes call; written is the following norm's original input.
 */
template<class Output>
__device__ __forceinline__ void writeTask(const float* residual,const float* block,
 const float* gateLogits,float* written,const float* gain,Output normalized,int hidden,
 float epsilon,std::uint64_t task,float* sums,unsigned lane,bool member) {
 const auto base=task*hidden,blockBase=(task/4)*hidden;
 const float gate=member?residualGate(gateLogits[task]):0.0f;
 taskFromInput(ResidualBranchInput{residual+base,block+blockBase,written+base,gate},
  gain,normalized,hidden,epsilon,task,sums,lane,member);
}
}
