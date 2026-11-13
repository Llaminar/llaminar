/**
 * @file GPUPooledImmutableExactIdentity.cuh
 * @brief Cooperative exact integer validation over original admitted owner scratch.
 *
 * All CTA lanes participate. Only immutable key scratch is permuted; this helper
 * never changes cache slots, grants, phases or physical ownership. A caller must
 * finish every key domain before its original ordered mutation writer starts.
 */
#pragma once
namespace llaminar2::qwen4exp::QFN_NATIVE_NAMESPACE {
/** @brief Fixed CTA-local sort workspace, independent of admitted key capacity. */
struct ExactIdentityCTA final {
    unsigned histogram[Bins],cursor[Bins],warp_counts[8*Bins],invalid;
};
/**
 * @brief Sort one exact live uint64 key bank using the existing stable CTA ranks.
 * @param input Original admitted first key bank, at least count elements.
 * @param output Original disjoint admitted second key bank.
 * @param count Authenticated live prefix; inactive capacity is never read.
 * @param bytes Original significant key width (four or eight).
 * @param shared Fixed CTA-local workspace shared by every lane.
 * @return Sorted original first bank (the even pass count preserves its identity).
 * @note Every lane must call with identical bounded inputs, including count zero.
 */
__device__ __forceinline__ std::uint64_t* exactIdentitySort(
    std::uint64_t* input,std::uint64_t* output,unsigned count,unsigned bytes,ExactIdentityCTA& shared)
{
    for(unsigned pass=0;pass<bytes*2;++pass){
        if(threadIdx.x<Bins)shared.histogram[threadIdx.x]=0;
        __syncthreads();
        for(std::uint64_t i=threadIdx.x;i<count;i+=Threads)
            atomicAdd(shared.histogram+((input[i]>>(4*pass))&15),1U);
        __syncthreads();
        if(threadIdx.x<Bins){
            unsigned base=0;
            for(unsigned prior=0;prior<threadIdx.x;++prior)base+=shared.histogram[prior];
            shared.cursor[threadIdx.x]=base;
        }
        __syncthreads();
        for(unsigned first=0;first<count;){
            const auto left=count-first;
            const auto i=first+threadIdx.x;
            const bool valid=threadIdx.x<left;
            const auto value=valid?input[i]:std::uint64_t{};
            const auto bin=static_cast<unsigned>((value>>(4*pass))&15);
            const auto rank=bucketRank(valid,bin,shared.warp_counts);
            if(valid)output[shared.cursor[bin]+rank]=value;
            __syncthreads();
            if(threadIdx.x<Bins){
                unsigned size=0;
                for(unsigned warp=0;warp<Threads/warpWidth();++warp)
                    size+=shared.warp_counts[warp*Bins+threadIdx.x];
                shared.cursor[threadIdx.x]+=size;
            }
            __syncthreads();
            if(left<=Threads)break;
            first+=Threads;
        }
        auto* swap=input;input=output;output=swap;
    }
    return input;
}
/**
 * @brief Reject exact duplicates without changing original physical state.
 * @param first Original admitted live keys; larger sets are sorted in place.
 * @param second Disjoint original admitted key scratch.
 * @param count Authenticated live count within both banks.
 * @param bytes Original significant key width (four or eight).
 * @param policy Original unassigned-slot sentinel semantics.
 * @param shared Fixed CTA-local state, reusable only after all lanes return.
 * @return Uniform true exactly when the scalar exact-identity contract accepts.
 * @note No hash, approximate match, success token or cross-kernel state is minted.
 */
__device__ __forceinline__ bool cooperativeExactIdentities(
    std::uint64_t* first,std::uint64_t* second,unsigned count,unsigned bytes,
    p::DuplicateIdentityPolicy policy,ExactIdentityCTA& shared)
{
    if(threadIdx.x==0)shared.invalid=(bytes!=4 && bytes!=8);
    __syncthreads();
    if(shared.invalid)return false;
    if(count<=p::SmallIdentityLimit){
        for(std::uint64_t i=threadIdx.x;i<count;i+=Threads){
            const auto key=first[i];
            if(policy==p::DuplicateIdentityPolicy::IgnoreUnassignedSlot && key==adaptive_lookup::None)continue;
            for(unsigned prior=0;prior<i;++prior)
                if(first[prior]==key)atomicOr(&shared.invalid,1U);
        }
    }else{
        auto* sorted=exactIdentitySort(first,second,count,bytes,shared);
        for(std::uint64_t i=std::uint64_t{threadIdx.x}+1;i<count;i+=Threads)
            if(sorted[i]==sorted[i-1] &&
               !(policy==p::DuplicateIdentityPolicy::IgnoreUnassignedSlot && sorted[i]==adaptive_lookup::None))
                atomicOr(&shared.invalid,1U);
    }
    __syncthreads();
    return shared.invalid==0;
}
} // namespace llaminar2::qwen4exp::QFN_NATIVE_NAMESPACE
