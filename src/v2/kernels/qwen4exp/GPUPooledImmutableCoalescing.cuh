/**
 * @file GPUPooledImmutableCoalescing.cuh
 * @brief Stable cooperative GPU coalescing over actual original counts.
 *
 * Small captured shapes use one cooperative CTA and shared temporary records.
 * Larger shapes use disjoint partition histograms, deterministic global offsets
 * and parallel scatter/scan producers. Neither path loops through original keys
 * on one lane or reads inactive rows. Both retain one canonical policy writer.
 * All mode/geometry choices depend only on immutable captured row capacity.
 */
#pragma once
namespace llaminar2::qwen4exp::QFN_NATIVE_NAMESPACE
{
namespace n=pooled_native;
namespace p=adaptive_lookup::pooled;
using Occurrence=p::CoordinatorOccurrence;
constexpr unsigned Threads=256,SmallRows=512,Bins=16;
/** @brief Publish an order-independent structural failure from independent disjoint producers. */
__device__ __forceinline__ void failure(const n::CoordinatorArgs& a)
{
    auto v=n::coordinatorView(a);
    atomicCAS(reinterpret_cast<unsigned*>(&v.frame->status),static_cast<unsigned>(adaptive_lookup::Status::Success),
        static_cast<unsigned>(adaptive_lookup::Status::Invariant));
    if(v.policy)atomicOr(&v.policy->flags,adaptive_lookup::OwnerFailed);
}
/** @return Whether the actual in-progress coalescer still owns this frame. */
__device__ __forceinline__ bool coalescing(const n::CoordinatorArgs& a)
{
    const auto& f=*n::coordinatorView(a).frame;
    return f.status==adaptive_lookup::Status::Success && f.phase==p::LookupPhase::Coalescing;
}
/** @return Exact native warp vote, used only for deterministic integer permutation ranks. */
__device__ __forceinline__ unsigned long long vote(bool predicate)
{
#if defined(QFN_POOLED_NATIVE_CUDA)
    return __ballot_sync(0xffffffffU,predicate);
#else
    return __ballot(predicate);
#endif
}
/** @return Native warp width is fixed by the compiled backend/ISA launch policy. */
__device__ __forceinline__ unsigned warpWidth()
{
#if defined(QFN_POOLED_NATIVE_CUDA)
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
template<bool Positions> __device__ __forceinline__ std::uint64_t key(const Occurrence& value)
{if constexpr(Positions)return value.position;else return value.row;}
/** @brief Cooperative stable four-bit radix sort in one small CTA, using only actual count. */
template<bool Positions>
__device__ __forceinline__ Occurrence* radixSmall(Occurrence* input,Occurrence* output,unsigned count,
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
            const auto value=valid?input[i]:Occurrence{};
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
/** @return One actual typed original producer value, with unsigned UInt32 zero extension. */
template<n::OriginalIdWidth Width> __device__ __forceinline__ std::uint64_t original(const n::CoordinatorBeginArgs& a,std::uint64_t i)
{
    using Row=std::conditional_t<Width==n::OriginalIdWidth::UInt32,std::uint32_t,std::uint64_t>;
    return static_cast<const Row*>(a.query.ids)[i];
}
/** @brief Open only original identity/bounds; every data producer follows on the same exact stream. */
template<n::OriginalIdWidth Width> __global__ void openOriginal(n::CoordinatorBeginArgs a)
{
    auto v=n::coordinatorView(a);auto op=p::CoordinatorOperations(v);
    const auto count=a.query.original.rows.activeRows();
    if(!a.query.admission || !p::CoordinatorCoalescingAccess::open(op,*a.query.admission,static_cast<const std::conditional_t<Width==n::OriginalIdWidth::UInt32,std::uint32_t,std::uint64_t>*>(a.query.ids),static_cast<unsigned>(count)))
        failure(a);
}
/** @brief One cooperative small-shape Begin bundle, including actual commit and immutable header publication. */
template<n::OriginalIdWidth Width> __global__ __launch_bounds__(Threads,2) void coalesceSmall(n::CoordinatorBeginArgs a)
{
    __shared__ Occurrence one[SmallRows],two[SmallRows];
    __shared__ unsigned histogram[Bins],cursor[Bins],warp_counts[8*Bins],prefix[Threads],carry,unique;
    auto v=n::coordinatorView(a);
    if(!coalescing(a))return;
    const unsigned count=v.frame->original_ids;
    if(!count){if(threadIdx.x==0)v.frame->unique_rows=0;return;}
    for(auto i=threadIdx.x;i<count;i+=Threads){
        const auto row=original<Width>(a,i);if(row>=a.plan.policy.sourceRows())failure(a);
        one[i]={row,i,0};
    }
    __syncthreads();
    if(!coalescing(a))return;
    radixSmall<false>(one,two,count,histogram,cursor,warp_counts,n::radixPasses(a.plan.policy.sourceRows()));
    if(threadIdx.x==0)carry=0;__syncthreads();
    for(unsigned first=0;first<count;first+=Threads){
        const auto i=first+threadIdx.x;
        const unsigned flag=i<count && (!i || one[i].row!=one[i-1].row);
        const auto inclusive=scan(flag,prefix);
        if(i<count)one[i].unique=carry+inclusive-1;
        __syncthreads();
        if(threadIdx.x==0)carry+=prefix[(count-first<Threads?count-first:Threads)-1];
        __syncthreads();
    }
    if(threadIdx.x==0){unique=carry;v.frame->unique_rows=carry;if(carry>a.plan.policy.maximumUnique())failure(a);}
    __syncthreads();
    if(!coalescing(a))return;
    for(auto i=threadIdx.x;i<count;i+=Threads){
        const auto value=one[i];
        v.original_to_unique[value.position]=value.unique;
        if(!i || value.row!=one[i-1].row)two[value.unique]={value.row,value.position,value.unique};
    }
    __syncthreads();
    radixSmall<true>(two,one,unique,histogram,cursor,warp_counts,n::radixPasses(a.query.original.rows.physicalRows()));
    for(auto i=threadIdx.x;i<unique;i+=Threads){
        const auto value=two[i];one[value.unique].unique=i;
        auto& out=v.unique[i];out={};out.row=value.row;out.first_position=value.position;
        const auto result=p::lookupDirectory(v.entries,v.nodes,a.plan.policy.entries(),
            v.policy?v.policy->directory_root:adaptive_lookup::None,value.row);
        if(result.error)failure(a);
        out.entry=result.entry;out.phase=result.entry==adaptive_lookup::None?p::UniquePhase::Missing:p::UniquePhase::NeedsPin;
    }
    __syncthreads();
    for(auto i=threadIdx.x;i<count;i+=Threads)v.original_to_unique[i]=one[v.original_to_unique[i]].unique;
    __syncthreads();
}
/** @brief Parallel actual-key validation and seed; inactive capacity is never read or written. */
template<n::OriginalIdWidth Width> __global__ __launch_bounds__(Threads,2) void seedKeys(n::CoordinatorBeginArgs a)
{
    if(!coalescing(a))return;auto v=n::coordinatorView(a);
    for(auto i=std::uint64_t{blockIdx.x}*Threads+threadIdx.x;i<v.frame->original_ids;i+=std::uint64_t{gridDim.x}*Threads){
        const auto row=original<Width>(a,i);if(row>=a.plan.policy.sourceRows())failure(a);
        v.sort_a[i]={row,static_cast<unsigned>(i),0};
    }
}
/** @brief Parallel per-partition histograms use only live chunks and exact four-bit keys. */
template<bool Positions> __global__ __launch_bounds__(Threads,2) void radixCount(n::CoordinatorBeginArgs a,unsigned pass,bool from_a)
{
    if(!coalescing(a))return;auto v=n::coordinatorView(a);
    const auto count=Positions?v.frame->unique_rows:v.frame->original_ids;
    if(std::uint64_t{blockIdx.x}*Threads>=count)return;
    __shared__ unsigned warp_counts[8*Bins];
    const auto* input=from_a?v.sort_a:v.sort_b;const auto i=std::uint64_t{blockIdx.x}*Threads+threadIdx.x;
    const auto bin=i<count?static_cast<unsigned>((key<Positions>(input[i])>>(4*pass))&15):0;
    bucketRank(i<count,bin,warp_counts);
    if(threadIdx.x<Bins){
        unsigned total=0;for(unsigned warp=0;warp<Threads/warpWidth();++warp)total+=warp_counts[warp*Bins+threadIdx.x];
        n::at<unsigned>(a.native,a.plan.native.radix_offsets)[blockIdx.x*Bins+threadIdx.x]=total;
    }
}
/** @brief Deterministic global bucket/partition offsets; work is O(live partitions), not O(original rows) on one lane. */
template<bool Positions> __global__ __launch_bounds__(Threads,2) void radixOffsets(n::CoordinatorBeginArgs a)
{
    if(!coalescing(a))return;auto v=n::coordinatorView(a);
    const auto count=Positions?v.frame->unique_rows:v.frame->original_ids,blocks=(count+Threads-1)/Threads;
    __shared__ unsigned totals[Bins];
    auto* offsets=n::at<unsigned>(a.native,a.plan.native.radix_offsets);
    if(threadIdx.x<Bins){
        unsigned total=0;for(unsigned block=0;block<blocks;++block)total+=offsets[block*Bins+threadIdx.x];
        totals[threadIdx.x]=total;
    }
    __syncthreads();
    if(threadIdx.x<Bins){
        unsigned cursor=0;for(unsigned bin=0;bin<threadIdx.x;++bin)cursor+=totals[bin];
        for(unsigned block=0;block<blocks;++block){const auto size=offsets[block*Bins+threadIdx.x];offsets[block*Bins+threadIdx.x]=cursor;cursor+=size;}
    }
}
/** @brief Disjoint stable parallel scatter from exact partition and earlier-lane ranks. */
template<bool Positions> __global__ __launch_bounds__(Threads,2) void radixScatter(n::CoordinatorBeginArgs a,unsigned pass,bool from_a)
{
    if(!coalescing(a))return;auto v=n::coordinatorView(a);
    const auto count=Positions?v.frame->unique_rows:v.frame->original_ids;
    if(std::uint64_t{blockIdx.x}*Threads>=count)return;
    __shared__ unsigned warp_counts[8*Bins];
    const auto* input=from_a?v.sort_a:v.sort_b;auto* output=from_a?v.sort_b:v.sort_a;
    const auto i=std::uint64_t{blockIdx.x}*Threads+threadIdx.x;const bool valid=i<count;
    const auto value=valid?input[i]:Occurrence{};
    const auto bin=static_cast<unsigned>((key<Positions>(value)>>(4*pass))&15);
    const auto rank=bucketRank(valid,bin,warp_counts);
    if(valid)output[n::at<unsigned>(a.native,a.plan.native.radix_offsets)[blockIdx.x*Bins+bin]+rank]=value;
}
/**
 * @brief Stable disjoint run producers for the original live permutation, after coalescing scratch retires.
 *
 * Each CTA owns exactly one 256-position partition. Parallel boundary detection
 * and integer scan compact only equal-row consecutive observations; partition
 * boundaries are never merged or reordered. No frequency/policy value is read
 * or written here. All authored run records precede the sole policy writer.
 */
__global__ __launch_bounds__(Threads,2) void observationRuns(n::CoordinatorArgs a)
{
    static_assert(Threads==p::OriginalObservationRuns::PartitionRows);
    auto v=n::coordinatorView(a);
    if(v.frame->status!=adaptive_lookup::Status::Success || v.frame->phase!=p::LookupPhase::Availability)return;
    const auto first=std::uint64_t{blockIdx.x}*Threads;
    if(first>=v.frame->original_ids)return;
    const auto position=first+threadIdx.x;
    const auto live=v.frame->original_ids-first<Threads?v.frame->original_ids-first:Threads;
    const bool valid=threadIdx.x<live;
    const auto unique=valid?v.original_to_unique[position]:adaptive_lookup::None;
    if(valid && unique>=v.frame->unique_rows)failure(a);
    __shared__ unsigned prefix[Threads],starts[Threads],count;
    const bool start=valid && (!threadIdx.x || unique!=v.original_to_unique[position-1]);
    const auto rank=scan(start?1u:0u,prefix);
    if(start)starts[rank-1]=static_cast<unsigned>(position);
    if(threadIdx.x==0)count=prefix[live-1];
    __syncthreads();
    if(v.frame->status!=adaptive_lookup::Status::Success)return;
    if(start){
        const auto end=rank<count?starts[rank]:static_cast<unsigned>(first+live);
        v.sort_a[first+rank-1]={v.unique[unique].row,static_cast<unsigned>(position),end-static_cast<unsigned>(position)};
    }
    if(threadIdx.x==0)v.sort_b[blockIdx.x].unique=count;
}
/** @brief Mark actual sorted row groups independently. */
__global__ __launch_bounds__(Threads,2) void markGroups(n::CoordinatorBeginArgs a)
{
    if(!coalescing(a))return;auto v=n::coordinatorView(a);
    for(auto i=std::uint64_t{blockIdx.x}*Threads+threadIdx.x;i<v.frame->original_ids;i+=std::uint64_t{gridDim.x}*Threads)
        v.sort_a[i].unique=!i || v.sort_a[i].row!=v.sort_a[i-1].row;
}
/** @brief Cooperative integer group scan; only O(live chunks) uniform control is serial. */
__global__ __launch_bounds__(Threads,2) void scanGroups(n::CoordinatorBeginArgs a)
{
    if(!coalescing(a))return;auto v=n::coordinatorView(a);
    __shared__ unsigned prefix[Threads],carry;
    if(threadIdx.x==0)carry=0;__syncthreads();
    const auto count=v.frame->original_ids;
    for(unsigned first=0;first<count;first+=Threads){
        const auto i=first+threadIdx.x;
        const auto inclusive=scan(i<count?v.sort_a[i].unique:0,prefix);
        if(i<count)v.sort_a[i].unique=carry+inclusive-1;
        __syncthreads();
        if(threadIdx.x==0)carry+=prefix[(count-first<Threads?count-first:Threads)-1];
        __syncthreads();
    }
    if(threadIdx.x==0){v.frame->unique_rows=carry;if(carry>a.plan.policy.maximumUnique())failure(a);}
}
/** @brief Materialize each distinct group and original mapping with one deterministic writer. */
__global__ __launch_bounds__(Threads,2) void materializeGroups(n::CoordinatorBeginArgs a)
{
    if(!coalescing(a))return;auto v=n::coordinatorView(a);
    for(auto i=std::uint64_t{blockIdx.x}*Threads+threadIdx.x;i<v.frame->original_ids;i+=std::uint64_t{gridDim.x}*Threads){
        const auto value=v.sort_a[i];v.original_to_unique[value.position]=value.unique;
        if(!i || value.row!=v.sort_a[i-1].row)v.sort_b[value.unique]={value.row,value.position,value.unique};
    }
}
/** @brief Independent read-only directory probes and stable unique initialization. */
__global__ __launch_bounds__(Threads,2) void materializeUnique(n::CoordinatorBeginArgs a)
{
    if(!coalescing(a))return;auto v=n::coordinatorView(a);
    for(auto i=std::uint64_t{blockIdx.x}*Threads+threadIdx.x;i<v.frame->unique_rows;i+=std::uint64_t{gridDim.x}*Threads){
        const auto value=v.sort_b[i];v.sort_a[value.unique].unique=static_cast<unsigned>(i);
        auto& out=v.unique[i];out={};out.row=value.row;out.first_position=value.position;
        const auto result=p::lookupDirectory(v.entries,v.nodes,a.plan.policy.entries(),
            v.policy?v.policy->directory_root:adaptive_lookup::None,value.row);
        if(result.error)failure(a);
        out.entry=result.entry;out.phase=result.entry==adaptive_lookup::None?p::UniquePhase::Missing:p::UniquePhase::NeedsPin;
    }
}
/** @brief Finish every original permutation position after the complete remap producer. */
__global__ __launch_bounds__(Threads,2) void remapOriginal(n::CoordinatorBeginArgs a)
{
    if(!coalescing(a))return;auto v=n::coordinatorView(a);
    for(auto i=std::uint64_t{blockIdx.x}*Threads+threadIdx.x;i<v.frame->original_ids;i+=std::uint64_t{gridDim.x}*Threads)
        v.original_to_unique[i]=v.sort_a[v.original_to_unique[i]].unique;
}
/** @brief Commit only after the complete cooperative producer DAG, then publish the compact immutable header. */
__global__ void finishOriginal(n::CoordinatorBeginArgs a)
{auto op=p::CoordinatorOperations(n::coordinatorView(a));p::CoordinatorCoalescingAccess::finish(op);n::invocation(a);}
}
