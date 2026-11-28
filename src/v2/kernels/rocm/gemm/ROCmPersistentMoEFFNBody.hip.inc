/**
 * @file ROCmPersistentMoEFFNBody.hip.inc
 * @brief Participant-local persistent FFN with CTA-owned original split-K folds.
 * @details Included after the original native decode and runtime helpers. Every
 * lane owns one original (column,partition) chain; LDS changes publication, never
 * arithmetic. The physical cooperative grid is admitted from this exact entry.
 * Routed/shared phases share one quantized activation publication and reuse the
 * original named scratch, with grid barriers closing all inter-CTA lifetimes.
 */

/** @brief Closed diagnostic physical decoder scope, including native CPU promotions. */
static constexpr std::uint32_t kPersistentMoEFFNDecoders=(1u<<5)|(1u<<7)|(1u<<19);

/** @brief One CTA's reusable partition plane; no global partial publication exists. */
struct PersistentMoEFFNShared
{
    float gate[256]; ///< Original partition-major gate/down results.
    float up[256];   ///< Original partition-major up results, dead during down.
};

/**
 * @brief Evaluate the original native K32 chain using an authoritative integer sum.
 * @tparam FMT Cold admitted physical decoder; original source policy selects partitions separately.
 * @param descriptor Original immutable descriptor. @param n Original output column.
 * @param codes Original canonical Q8 codes. @param scales Original half-rounded/widened scales.
 * @param sums Exact sum of each original32-code block, computed once at publication.
 * @param N Original output width. @param K Original input width.
 * @param part Original arithmetic partition. @param partitions Original partition count.
 * @return Identical original ascending-K32 FP32 partial, including empty +0.
 */
template<NativeQuantisedFormat FMT>
__device__ __forceinline__ float persistentMoEFFNPartial(const DeviceNativeQuantisedMatrixDesc* descriptor,
    int n,const int8_t* codes,const float* scales,const int32_t* sums,int N,int K,int part,int partitions)
{
    if(!native_quantised_desc_shape_ok<FMT>(*descriptor,N,K))return 0.f;
    const int blocks=K/32,width=(blocks+partitions-1)/partitions;
    const int first=part*width,last=min(blocks,first+width);
    const auto* payload=descriptor->payload;
    // Per-block or superblock-coded metadata of this immutable expert matrix.
    const auto meta=::llaminar2::native_quantised_scale::metadataPlanes(*descriptor);
    float accumulator=0.f;
    #pragma unroll 1
    for(int block=first;block<last;++block)
    {
        const auto* activation=reinterpret_cast<const int32_t*>(codes+size_t(block)*32);
        // This is the same source decoder and complete rounded-block commit
        // used by the original producer. Integer summation has no overflow.
        moePairedK32CommitPlanes<FMT>(payload,meta,::llaminar2::rocm_native_metadata::blockAddress(meta,uint32_t(block),uint32_t(n),uint32_t(N)),
            activation,scales[block],sums[block],accumulator);
    }
    return accumulator;
}

/** @brief Dispatch only the explicit immutable physical envelope; never widen a missing representation. */
__device__ __forceinline__ float persistentMoEFFNPartial(const DeviceNativeQuantisedMatrixDesc* descriptor,
    int n,const int8_t* codes,const float* scales,const int32_t* sums,int N,int K,int part,int partitions)
{
    switch(descriptor->codebook_id)
    {
    case 5:return persistentMoEFFNPartial<static_cast<NativeQuantisedFormat>(5)>(descriptor,n,codes,scales,sums,N,K,part,partitions);
    case 7:return persistentMoEFFNPartial<static_cast<NativeQuantisedFormat>(7)>(descriptor,n,codes,scales,sums,N,K,part,partitions);
    case 19:return persistentMoEFFNPartial<static_cast<NativeQuantisedFormat>(19)>(descriptor,n,codes,scales,sums,N,K,part,partitions);
    default:__builtin_trap();return 0.f;
    }
}

/**
 * @brief Fold a CTA-owned plane in the original ascending-partition RN order.
 * @tparam Plane Original pointee qualification, preserving explicit LDS address space.
 */
template<class Plane>
__device__ __forceinline__ float persistentMoEFFNFold(const Plane* plane,unsigned column,unsigned columns,int partitions)
{
    if(partitions==1)return plane[column];
    float value=0.f;
    #pragma unroll 1
    for(int part=0;part<partitions;++part)
        value=rocm_native_quantised_accumulate_rn(value,plane[size_t(part)*columns+column]);
    return value;
}

#include "ROCmPersistentMoEFFNQuantize.h"

/**
 * @brief Complete composable post-router phase using one physical256-thread grid.
 * @tparam TaskIndex Cold-selected exact scheduler width; all tensor coordinates and offsets stay size_t.
 * @param a Existing source-authenticated prepared owner; descriptor pointers remain immutable.
 * @param shared CTA-local partition storage, reusable by preceding/following typed phases only after a barrier.
 * @param grid Exact native cooperative grid; every physical CTA participates in every grid barrier.
 * @details Scratch byte intervals are derived solely from original admitted shapes:
 * V=topK*routedF+sharedF; gate_partials[0,V) holds activated FP32, up_partials
 * holds V code bytes then V/32 scales and V/32 INT32 sums. down_partials holds
 * H/32 hidden sums. Existing banks havePmax*max(topK*routedF,sharedF) floats and
 * Pmax*topK*H floats respectively, so these disjoint live prefixes always fit.
 */
template<class TaskIndex=std::uint64_t>
__device__ __forceinline__ void persistentMoEFFNPhase(const llaminar2::rocm::ResidentMoEFFNArgs& a,
    PersistentMoEFFNShared& shared,cooperative_groups::grid_group grid)
{
    static_assert(std::is_same_v<TaskIndex,std::uint32_t>||std::is_same_v<TaskIndex,std::uint64_t>);
    using llaminar2::rocm::residentMoEFFNNextTask;
    const TaskIndex task_stride=static_cast<TaskIndex>(gridDim.x);
    const unsigned lane=threadIdx.x;
    const auto* runtime=reinterpret_cast<const DeviceMoELayerRuntimeView*>(a.runtime);
    const int H=a.hidden_columns;
    const bool live=a.rows.activeRows()!=0;
    if(live&&(!runtime||runtime->expert_count!=unsigned(a.experts)||runtime->top_k!=unsigned(a.top_k)||
        !runtime->intermediate_shard.sameIdentity(a.original)||
        moe_runtime_execution_bank(runtime)==kInvalidMoERuntimeExecutionBank))__builtin_trap();
    const size_t routed_values=size_t(a.top_k)*a.routed_columns;
    const size_t values=routed_values+a.shared_columns,blocks=values/32;
    auto* activated=a.gate_partials;
    auto* codes=reinterpret_cast<int8_t*>(a.up_partials);
    auto* scales=reinterpret_cast<float*>(codes+values);
    auto* sums=reinterpret_cast<int32_t*>(scales+blocks);
    auto* hidden_sums=reinterpret_cast<int32_t*>(a.down_partials);
    for(size_t n=size_t(blockIdx.x)*256+lane;n<size_t(H);n+=size_t(gridDim.x)*256)
        persistentMoEFFNQuantize(a.hidden[n],n,a.hidden_q8,a.hidden_scales,hidden_sums);
    grid.sync();

    // One CTA owns ALL original arithmetic partitions for each column. The
    // lane layout is partition-major, preserving adjacent-column weight loads.
    const unsigned routed_columns=256/a.routed_gate_policy.maximum;
    const unsigned shared_columns=256/a.shared_gate_policy.maximum;
    const TaskIndex routed_tiles=(TaskIndex(a.routed_columns)+routed_columns-1)/routed_columns;
    const TaskIndex shared_tiles=(TaskIndex(a.shared_columns)+shared_columns-1)/shared_columns;
    const TaskIndex routed_tasks=live?routed_tiles*TaskIndex(a.top_k):0;
    const TaskIndex gate_tasks=routed_tasks+shared_tiles;
    for(TaskIndex task=TaskIndex(blockIdx.x);task<gate_tasks;task=residentMoEFFNNextTask(task,gate_tasks,task_stride))
    {
        const bool is_shared=task>=routed_tasks;
        const auto policy=is_shared?a.shared_gate_policy:a.routed_gate_policy;
        const unsigned columns=is_shared?shared_columns:routed_columns;
        const unsigned column=lane%columns,part=lane/columns;
        const TaskIndex local=is_shared?task-routed_tasks:task;
        const int route=is_shared?0:int(local/routed_tiles);
        const TaskIndex tile=is_shared?local:local%routed_tiles;
        const int F=is_shared?a.shared_columns:a.routed_columns;
        const size_t n=size_t(tile)*columns+column;
        const DeviceNativeQuantisedMatrixDesc *gate=nullptr,*up=nullptr;
        const bool ready=is_shared?(gate=a.shared_gate,up=a.shared_up,true):
            moe_native_quantised_load_gate_up_descriptors<true>(nullptr,nullptr,runtime,
                runtime->topk_expert_ids[route],a.experts,gate,up);
        const auto mask=is_shared?a.execution.shared_gate:a.execution.routed_gate;
        const int gate_parts=ready?residentMoEPartitions(gate,policy,mask):1;
        const int up_parts=ready?residentMoEPartitions(up,policy,mask):1;
        if(part<policy.maximum)
        {
            // Only one accumulator/decoder lifetime exists at a time.
            shared.gate[lane]=ready&&n<size_t(F)&&part<unsigned(gate_parts)
                ?persistentMoEFFNPartial(gate,int(n),a.hidden_q8,a.hidden_scales,hidden_sums,F,H,part,gate_parts):0.f;
            shared.up[lane]=ready&&n<size_t(F)&&part<unsigned(up_parts)
                ?persistentMoEFFNPartial(up,int(n),a.hidden_q8,a.hidden_scales,hidden_sums,F,H,part,up_parts):0.f;
        }
        __syncthreads();
        if(part==0&&n<size_t(F))
        {
            const float g=persistentMoEFFNFold(shared.gate,column,columns,gate_parts);
            const float u=persistentMoEFFNFold(shared.up,column,columns,up_parts);
            if(is_shared){a.shared_gate_output[n]=g;a.shared_up_output[n]=u;}
            activated[(is_shared?routed_values:size_t(route)*F)+n]=
                ready?llaminar2::rocm::decode_equivalent::swigluValue(g,u):0.f;
        }
        __syncthreads(); // Retire both LDS planes before the next CTA-owned tile.
    }
    if(!live)
        for(size_t n=size_t(blockIdx.x)*256+lane;n<routed_values;n+=size_t(gridDim.x)*256)activated[n]=0.f;
    grid.sync();
    for(size_t n=size_t(blockIdx.x)*256+lane;n<values;n+=size_t(gridDim.x)*256)
        persistentMoEFFNQuantize(activated[n],n,codes,scales,sums);
    grid.sync();

    const unsigned routed_down_columns=256/a.routed_down_policy.maximum;
    const unsigned shared_down_columns=256/a.shared_down_policy.maximum;
    const TaskIndex routed_down_tiles=(TaskIndex(H)+routed_down_columns-1)/routed_down_columns;
    const TaskIndex shared_down_tiles=(TaskIndex(H)+shared_down_columns-1)/shared_down_columns;
    const TaskIndex routed_down_tasks=live?routed_down_tiles*TaskIndex(a.top_k):0;
    const TaskIndex down_tasks=routed_down_tasks+shared_down_tiles;
    for(TaskIndex task=TaskIndex(blockIdx.x);task<down_tasks;task=residentMoEFFNNextTask(task,down_tasks,task_stride))
    {
        const bool is_shared=task>=routed_down_tasks;
        const auto policy=is_shared?a.shared_down_policy:a.routed_down_policy;
        const unsigned columns=is_shared?shared_down_columns:routed_down_columns;
        const unsigned column=lane%columns,part=lane/columns;
        const TaskIndex local=is_shared?task-routed_down_tasks:task;
        const int route=is_shared?0:int(local/routed_down_tiles);
        const TaskIndex tile=is_shared?local:local%routed_down_tiles;
        const size_t n=size_t(tile)*columns+column;
        const int F=is_shared?a.shared_columns:a.routed_columns;
        const size_t offset=is_shared?routed_values:size_t(route)*F;
        const DeviceNativeQuantisedMatrixDesc* down=nullptr;
        const bool ready=is_shared?(down=a.shared_down,true):moe_native_quantised_load_down_descriptor<true>(
            nullptr,runtime,runtime->topk_expert_ids[route],a.experts,down);
        const auto mask=is_shared?a.execution.shared_down:a.execution.routed_down;
        const int parts=ready?residentMoEPartitions(down,policy,mask):1;
        if(part<policy.maximum)
            shared.gate[lane]=ready&&n<size_t(H)&&part<unsigned(parts)
                ?persistentMoEFFNPartial(down,int(n),codes+offset,scales+offset/32,sums+offset/32,H,F,part,parts):0.f;
        __syncthreads();
        if(part==0&&n<size_t(H))
        {
            const float partial=persistentMoEFFNFold(shared.gate,column,columns,parts);
            const float value=ready?rocm_native_quantised_weight_route_rn(is_shared?1.f:runtime->topk_weights[route],partial):0.f;
            if(is_shared)a.shared_output[n]=rocm_native_quantised_accumulate_rn(0.f,value);
            else a.route_contributions[size_t(route)*H+n]=value;
        }
        __syncthreads();
    }
    if(!live)
        for(size_t n=size_t(blockIdx.x)*256+lane;n<size_t(a.top_k)*H;n+=size_t(gridDim.x)*256)a.route_contributions[n]=0.f;
    grid.sync();
    for(size_t n=size_t(blockIdx.x)*256+lane;n<size_t(H);n+=size_t(gridDim.x)*256)
        a.routed_output[n]=llaminar2::device_ordered_route_fold::sum(a.route_contributions+n,size_t(H),a.top_k);
    grid.sync();
    if(blockIdx.x==0)llaminar2::rocm::moe_shared_gate::gateAdd<false>(a.hidden,a.shared_input_gate,
        a.shared_output,a.routed_output,a.combined_output,1,H,0,live?1:0);
}

/** @brief Exact single-region native entry, retaining the existing prepared-owner/capture boundary. */
template<class TaskIndex>
__global__ __launch_bounds__(256,1) void rocmPersistentMoEFFNKernel(const llaminar2::rocm::ResidentMoEFFNArgs a)
{
    const auto grid=cooperative_groups::this_grid();
    if(!grid.is_valid())__builtin_trap();
    __shared__ PersistentMoEFFNShared shared;
    persistentMoEFFNPhase<TaskIndex>(a,shared,grid);
}
