/**
 * @file GPUImmutableStableRadix.cuh
 * @brief Shared original cooperative stable integer radix/scan primitives.
 *
 * Extracted without arithmetic/order changes from the original pooled
 * coalescer. Native local cache and pooled union therefore share the same
 * vote ranks, chunk order and even-pass input/output bank convention.
 */
#pragma once
namespace llaminar2::qwen4exp::immutable_stable_radix {
inline constexpr unsigned Threads=256,Bins=16;
/** @return Exact native warp vote, used only for deterministic integer permutation ranks. */
__device__ __forceinline__ unsigned long long vote(bool predicate)
{
#if defined(__CUDACC__)
    return __ballot_sync(0xffffffffU,predicate);
#else
    return __ballot(predicate);
#endif
}
/** @return Native warp width is fixed by the compiled backend/ISA launch policy. */
__device__ __forceinline__ unsigned warpWidth()
{
#if defined(__CUDACC__)
    return 32;
#else
    return 64;
#endif
}
/** @return Population count of one exact integer vote. */
__device__ __forceinline__ unsigned population(unsigned long long bits){return __popcll(bits);}
/**
 * @brief Stable intra-CTA bucket ranks with no atomics or partition-dependent arithmetic.
 * @param valid Actual live item in this chunk; inactive lanes still join votes/barriers.
 * @param bucket Four-bit stable radix key.
 * @param warp_counts Exactly one count per native warp and bucket in CTA-shared memory.
 * @return Number of earlier live items with this exact bucket in original chunk order.
 */
__device__ __forceinline__ unsigned bucketRank(bool valid,unsigned bucket,unsigned* warp_counts)
{
    const unsigned width=warpWidth(),lane=threadIdx.x%width,warp=threadIdx.x/width;
    const unsigned long long earlier=lane?~0ULL>>(64-lane):0;
    unsigned rank=0;
#pragma unroll
    for(unsigned bin=0;bin<Bins;++bin){
        const auto mask=vote(valid && bucket==bin);
        if(lane==0)warp_counts[warp*Bins+bin]=population(mask);
        if(bucket==bin)rank=population(mask&earlier);
    }
    __syncthreads();
    for(unsigned prior=0;prior<warp;++prior)rank+=warp_counts[prior*Bins+bucket];
    return rank;
}
/** @return Original complete native key for row or first-occurrence sorting. */
template<bool Positions,class Record> __device__ __forceinline__ std::uint64_t key(const Record& value)
{if constexpr(Positions)return value.position;else return value.row;}
/** @brief Cooperative stable four-bit radix sort in one small CTA, using only actual count. */
template<bool Positions,class Record>
__device__ __forceinline__ Record* radixSmall(Record* input,Record* output,unsigned count,
    unsigned* histogram,unsigned* cursor,unsigned* warp_counts,unsigned passes)
{
    for(unsigned pass=0;pass<passes;++pass){
        if(threadIdx.x<Bins)histogram[threadIdx.x]=0;
        __syncthreads();
        for(auto i=threadIdx.x;i<count;i+=Threads)atomicAdd(histogram+((key<Positions>(input[i])>>(4*pass))&15),1U);
        __syncthreads();
        if(threadIdx.x<Bins){
            unsigned base=0;
            for(unsigned bin=0;bin<threadIdx.x;++bin)base+=histogram[bin];
            cursor[threadIdx.x]=base;
        }
        __syncthreads();
        for(unsigned first=0;first<count;first+=Threads){
            const auto i=first+threadIdx.x;const bool valid=i<count;
            const auto value=valid?input[i]:Record{};
            const auto bin=static_cast<unsigned>((key<Positions>(value)>>(4*pass))&15);
            const auto rank=bucketRank(valid,bin,warp_counts);
            if(valid)output[cursor[bin]+rank]=value;
            __syncthreads();
            if(threadIdx.x<Bins){
                unsigned size=0;
                for(unsigned warp=0;warp<Threads/warpWidth();++warp)size+=warp_counts[warp*Bins+threadIdx.x];
                cursor[threadIdx.x]+=size;
            }
            __syncthreads();
        }
        auto* swap=input;input=output;output=swap;
    }
    return input;
}
/** @brief Deterministic live-prefix inclusive scan for one native CTA chunk. */
__device__ __forceinline__ unsigned scan(unsigned value,unsigned* shared)
{
    shared[threadIdx.x]=value;__syncthreads();
#pragma unroll
    for(unsigned delta=1;delta<Threads;delta*=2){
        const auto prior=threadIdx.x>=delta?shared[threadIdx.x-delta]:0;
        __syncthreads();shared[threadIdx.x]+=prior;__syncthreads();
    }
    return shared[threadIdx.x];
}
} // namespace llaminar2::qwen4exp::immutable_stable_radix
