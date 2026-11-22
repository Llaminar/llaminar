/**
 * @file ROCmOrdinaryMoEFFNBody.hip.inc
 * @brief Two ordinary CTA-owned FFN phases with no cross-CTA synchronization.
 * @details Phase A owns complete original gate/up partition folds and publishes
 * activated FP32. Phase B locally quantizes all routed/shared rows, owns every
 * down partition and route for its output tile, and publishes all six original
 * public planes. Only the captured A-to-B kernel edge orders different CTAs.
 */

/** @brief Emitted decoder inventory, proved against the existing canonical registry. */
#define LLAMINAR_ORDINARY_FFN_DECODERS(Apply) \
    Apply(NATIVE_QUANTISED_Q4_0) Apply(NATIVE_QUANTISED_IQ4_NL) \
    Apply(NATIVE_QUANTISED_IQ4_XS_SOURCE) Apply(NATIVE_QUANTISED_MXFP4) Apply(NATIVE_QUANTISED_Q4_1) \
    Apply(NATIVE_QUANTISED_Q5_0) Apply(NATIVE_QUANTISED_Q5_1) \
    Apply(NATIVE_QUANTISED_Q6_K) Apply(NATIVE_QUANTISED_Q3_K) Apply(NATIVE_QUANTISED_Q2_K) \
    Apply(NATIVE_QUANTISED_IQ3_S) Apply(NATIVE_QUANTISED_IQ3_XXS) \
    Apply(NATIVE_QUANTISED_IQ2_S) Apply(NATIVE_QUANTISED_IQ2_XS) Apply(NATIVE_QUANTISED_IQ2_XXS) \
    Apply(NATIVE_QUANTISED_IQ1_S) Apply(NATIVE_QUANTISED_IQ1_M) \
    Apply(NATIVE_QUANTISED_Q8_0) Apply(NATIVE_QUANTISED_EXPANDED_INT8_MIN)
#define LLAMINAR_ORDINARY_FFN_BIT(Format) | (1u<<unsigned(Format))
static_assert((0u LLAMINAR_ORDINARY_FFN_DECODERS(LLAMINAR_ORDINARY_FFN_BIT))==
    llaminar2::rocm::ResidentMoEFFNDecoderProgram::all_decoders,
    "Ordinary FFN must emit the complete canonical native decoder inventory");
#undef LLAMINAR_ORDINARY_FFN_BIT

/**
 * @brief Original per-format dot with the same canonical contribution helper and cached exact integer sum.
 * @param descriptor Original acquired descriptor. @param n Original independent column.
 * @param codes CTA-local canonical Q8 row. @param scales CTA-local original scales.
 * @param sums Exact sums of canonical codes. @param N Original output width. @param K Original input width.
 * @param part Original arithmetic partition. @param count Original descriptor-local partition count.
 * @return Original ascending-K32 rounded partial; excluded/unknown formats trap.
 */
template<std::uint32_t DecoderMask>
__device__ __forceinline__ float ordinaryMoEFFNPartial(const DeviceNativeQuantisedMatrixDesc* descriptor,
    int n,const int8_t* codes,const float* scales,const int32_t* sums,int N,int K,int part,int count)
{
    switch(descriptor->codebook_id)
    {
#define LLAMINAR_ORDINARY_FFN_CASE(Format) \
    case Format:if constexpr((DecoderMask&(1u<<unsigned(Format)))!=0) \
        return persistentMoEFFNPartial<Format>(descriptor,n,codes,scales,sums,N,K,part,count); \
        break;
        LLAMINAR_ORDINARY_FFN_DECODERS(LLAMINAR_ORDINARY_FFN_CASE)
#undef LLAMINAR_ORDINARY_FFN_CASE
    default:break;
    }
    __builtin_trap();return 0.f; // Excluded descriptors never widen this immutable program.
}
#undef LLAMINAR_ORDINARY_FFN_DECODERS

/**
 * @brief Retain original runtime geometry after the exact router producer's ordered publication.
 * @details The prepared owner requires the original router to authenticate its
 * acquired epoch and publish these exact expert-indexed descriptor arrays.
 * Rewalking the epoch here would duplicate the producer's validation. These
 * checks still reject foreign runtime/source geometry before any selected row.
 */
__device__ __forceinline__ const DeviceMoELayerRuntimeView* ordinaryMoEFFNRuntime(
    const llaminar2::rocm::ResidentMoEFFNArgs& a,bool live)
{
    const auto* runtime=reinterpret_cast<const DeviceMoELayerRuntimeView*>(a.runtime);
    if(live&&(!runtime||runtime->expert_count!=unsigned(a.experts)||runtime->top_k!=unsigned(a.top_k)||
        !runtime->intermediate_shard.sameIdentity(a.original)))__builtin_trap();
    return runtime;
}

/**
 * @brief CTA-local immutable original descriptor selection and policy result.
 * @details One leader consumes the exact original router's already-validated
 * descriptor publication and retains the original source-policy checks. A CTA
 * barrier publishes the result to the original column/partition workers;
 * no descriptor payload is copied or moved to another ownership authority.
 */
struct OrdinaryMoEFFNRouteBinding
{
    const DeviceNativeQuantisedMatrixDesc* descriptor;
    int partitions,ready;
    float weight;
};
/** @brief CTA-local gate/up source selection with independent original partition counts. */
struct OrdinaryMoEFFNGateBinding
{
    const DeviceNativeQuantisedMatrixDesc *gate,*up;
    int gate_parts,up_parts,ready;
};

/**
 * @brief One ordinary CTA owns an activated column tile and every original gate/up K partition.
 * @param a Same prepared source/table/arena owner as the staged oracle.
 * @details Local Q8 preparation deliberately duplicates the hot full-H input
 * across useful CTAs. Weight elements and original K32 commits are not duplicated.
 * The first shared tile alone publishes the original diagnostic hidden Q8 plane;
 * no other CTA consumes that global copy. Static LDS contains two256-float planes.
 */
template<std::uint32_t DecoderMask>
__global__ __launch_bounds__(256,1) void rocmOrdinaryMoEFFNGateUpKernel(const llaminar2::rocm::ResidentMoEFFNArgs a)
{
    const bool live=a.rows.activeRows()!=0;
    const auto* runtime=ordinaryMoEFFNRuntime(a,live);
    const unsigned routed_columns=256/a.routed_gate_policy.maximum;
    const unsigned shared_columns=256/a.shared_gate_policy.maximum;
    const unsigned routed_tiles=(unsigned(a.routed_columns)+routed_columns-1)/routed_columns;
    const unsigned routed_tasks=routed_tiles*unsigned(a.top_k);
    const unsigned task=blockIdx.x;
    const bool is_shared=task>=routed_tasks;
    const unsigned columns=is_shared?shared_columns:routed_columns;
    const unsigned column=threadIdx.x%columns,part=threadIdx.x/columns;
    const unsigned local=is_shared?task-routed_tasks:task;
    const int route=is_shared?0:int(local/routed_tiles);
    const unsigned tile=is_shared?local:local%routed_tiles;
    const int F=is_shared?a.shared_columns:a.routed_columns;
    const size_t n=size_t(tile)*columns+column;
    const size_t routed_values=size_t(a.top_k)*a.routed_columns;
    const size_t offset=is_shared?routed_values:size_t(route)*F;
    __shared__ OrdinaryMoEFFNGateBinding binding;
    if(threadIdx.x==0)
    {
        const DeviceNativeQuantisedMatrixDesc *gate=nullptr,*up=nullptr;
        const bool selected=is_shared?(gate=a.shared_gate,up=a.shared_up,true):live&&
            moe_native_quantised_load_gate_up_descriptors<false>(a.gates,a.ups,nullptr,
                runtime->topk_expert_ids[route],a.experts,gate,up);
        // The router writes all56 bytes as zero for a selected but unready
        // descriptor. Remote/empty IDs fail before this expert-indexed read.
        const bool ready=selected&&gate->payload&&gate->scales&&up->payload&&up->scales;
        const auto policy=is_shared?a.shared_gate_policy:a.routed_gate_policy;
        const auto execution=is_shared?a.execution.shared_gate:a.execution.routed_gate;
        binding.gate=gate;binding.up=up;binding.ready=ready;
        binding.gate_parts=ready?residentMoEPartitions(gate,policy,execution):1;
        binding.up_parts=ready?residentMoEPartitions(up,policy,execution):1;
    }
    __syncthreads();
    if(!binding.ready)
    {
        if(part==0&&n<size_t(F))a.gate_partials[offset+n]=0.f;
        return; // Published readiness is CTA-uniform; no live peer skips a barrier.
    }
    const auto policy=is_shared?a.shared_gate_policy:a.routed_gate_policy;
    extern __shared__ __align__(16) unsigned char ordinary_gate_storage[];
    auto* codes=reinterpret_cast<int8_t*>(ordinary_gate_storage);
    auto* scales=reinterpret_cast<float*>(codes+a.hidden_columns);
    auto* sums=reinterpret_cast<int32_t*>(scales+a.hidden_columns/32);
    __shared__ PersistentMoEFFNShared partials;
    for(size_t k=threadIdx.x;k<size_t(a.hidden_columns);k+=256)
        persistentMoEFFNQuantize(a.hidden[k],k,codes,scales,sums);
    __syncthreads();
    if(is_shared&&tile==0)
    {
        for(size_t k=threadIdx.x;k<size_t(a.hidden_columns);k+=256)a.hidden_q8[k]=codes[k];
        for(size_t block=threadIdx.x;block<size_t(a.hidden_columns/32);block+=256)
        {
            a.hidden_scales[block]=scales[block];
            reinterpret_cast<int32_t*>(a.down_partials)[block]=sums[block];
        }
    }
    // Reuse one emitted decoder callsite while retiring each role's partial
    // immediately. No decoded operand/FP32 accumulator spans the other role.
    #pragma unroll 1
    for(int projection=0;projection<2;++projection)
    {
        const auto* descriptor=projection?binding.up:binding.gate;
        const int count=projection?binding.up_parts:binding.gate_parts;
        if(part<policy.maximum)
            (projection?partials.up:partials.gate)[threadIdx.x]=n<size_t(F)&&part<unsigned(count)
                ?ordinaryMoEFFNPartial<DecoderMask>(descriptor,int(n),codes,scales,sums,F,a.hidden_columns,int(part),count):0.f;
    }
    __syncthreads();
    if(part==0&&n<size_t(F))
    {
        const float g=persistentMoEFFNFold(partials.gate,column,columns,binding.gate_parts);
        const float u=persistentMoEFFNFold(partials.up,column,columns,binding.up_parts);
        if(is_shared){a.shared_gate_output[n]=g;a.shared_up_output[n]=u;}
        a.gate_partials[offset+n]=llaminar2::rocm::decode_equivalent::swigluValue(g,u);
    }
}

/**
 * @brief One ordinary CTA owns all routes, original down folds and final outputs for32 H columns.
 * @param a Same sealed original source and named scratch owner as phase A.
 * @details Each CTA reproduces the original256-lane shared gate dot locally.
 * It then quantizes the complete activated row set into private LDS, as the fork
 * does per output block. Route batches fill otherwise idle partition workers;
 * each route still owns its original parts and exact ordered RN fold/weight.
 * Weighted rows live in LDS until the unchanged canonical slot fold consumes
 * them. Only one CTA publishes diagnostic global Q8 bytes; peers never read them.
 */
template<std::uint32_t DecoderMask>
__global__ __launch_bounds__(256,1) void rocmOrdinaryMoEFFNDownKernel(const llaminar2::rocm::ResidentMoEFFNArgs a)
{
    const bool live=a.rows.activeRows()!=0;
    const auto* runtime=ordinaryMoEFFNRuntime(a,live);
    const unsigned column=threadIdx.x%32,group=threadIdx.x/32;
    // Cold admission bounds the useful grid to ceil(H/32), with positive
    // signed32 H. Keep this logical column unsigned; pointer offsets still
    // widen through their original size_t route/row strides below.
    const unsigned n=blockIdx.x*32+column;
    __shared__ float gate_scratch[16];
    __shared__ OrdinaryMoEFFNRouteBinding bindings[8];
    if(live)(void)llaminar2::rocm::moe_shared_gate::gateValue256(
        a.hidden,a.shared_input_gate,a.hidden_columns,gate_scratch);
    const size_t routed_values=size_t(a.top_k)*a.routed_columns;
    const size_t values=routed_values+a.shared_columns,blocks=values/32;
    extern __shared__ __align__(16) unsigned char ordinary_down_storage[];
    auto* codes=reinterpret_cast<int8_t*>(ordinary_down_storage);
    auto* scales=reinterpret_cast<float*>(codes+values);
    auto* sums=reinterpret_cast<int32_t*>(scales+blocks);
    auto* partials=reinterpret_cast<float*>(sums+blocks);
    const unsigned workers=min(8u,a.routed_down_policy.maximum);
    const unsigned batch_capacity=8/workers;
    const unsigned partial_rows=max(batch_capacity*a.routed_down_policy.maximum,a.shared_down_policy.maximum);
    auto* weighted=partials+size_t(partial_rows)*32;
    for(size_t index=threadIdx.x;index<values;index+=256)
        persistentMoEFFNQuantize(a.gate_partials[index],index,codes,scales,sums);
    __syncthreads();
    if(blockIdx.x==0)
    {
        auto* public_codes=reinterpret_cast<int8_t*>(a.up_partials);
        auto* public_scales=reinterpret_cast<float*>(public_codes+values);
        auto* public_sums=reinterpret_cast<int32_t*>(public_scales+blocks);
        for(size_t index=threadIdx.x;index<values;index+=256)public_codes[index]=codes[index];
        for(size_t block=threadIdx.x;block<blocks;block+=256)
        {public_scales[block]=scales[block];public_sums[block]=sums[block];}
    }
    // The shared row is the terminal logical route of this same phase loop.
    // One expanded decoder callsite serves both roles, retiring its descriptor
    // and partition state before the next role. Routed batches never absorb
    // the shared row; their original route order and weights remain explicit.
    int first_route=0;
    #pragma unroll 1
    while(first_route<=a.top_k)
    {
        const bool is_shared=first_route==a.top_k;
        const unsigned maximum=is_shared?a.shared_down_policy.maximum:a.routed_down_policy.maximum;
        const unsigned role_workers=is_shared?8u:workers;
        const unsigned batch_route=group/role_workers,part_worker=group%role_workers;
        const int route=first_route+int(batch_route);
        const bool active=is_shared?batch_route==0:batch_route<batch_capacity&&route<a.top_k;
        if(column==0&&part_worker==0&&active)
        {
            const DeviceNativeQuantisedMatrixDesc* down=nullptr;
            const bool selected=is_shared?(down=a.shared_down,true):live&&
                moe_native_quantised_load_down_descriptor<false>(a.downs,nullptr,
                    runtime->topk_expert_ids[route],a.experts,down);
            const bool ready=selected&&down->payload&&down->scales;
            // The complete source-policy tuple is consumed only by this
            // leader's cold-immutable descriptor check. Keep no copied tuple
            // live across the CTA publication or expanded decoder body.
            const auto& policy=is_shared?a.shared_down_policy:a.routed_down_policy;
            const auto execution=is_shared?a.execution.shared_down:a.execution.routed_down;
            bindings[batch_route].descriptor=down;
            bindings[batch_route].partitions=ready?residentMoEPartitions(down,policy,execution):1;
            bindings[batch_route].ready=ready;
            bindings[batch_route].weight=ready?(is_shared?1.f:runtime->topk_weights[route]):0.f;
        }
        __syncthreads(); // Publish the original router-selected source/policy result to this CTA only.
        const auto* down=active?bindings[batch_route].descriptor:nullptr;
        const bool ready=active&&bindings[batch_route].ready;
        const int count=active?bindings[batch_route].partitions:1;
        const int reduction=is_shared?a.shared_columns:a.routed_columns;
        const size_t input_offset=is_shared?routed_values:size_t(route)*a.routed_columns;
        if(active)
            for(unsigned part=part_worker;part<maximum;part+=role_workers)
                partials[(size_t(batch_route)*maximum+part)*32+column]=
                    ready&&n<unsigned(a.hidden_columns)&&part<unsigned(count)
                        ?ordinaryMoEFFNPartial<DecoderMask>(down,int(n),codes+input_offset,
                            scales+input_offset/32,sums+input_offset/32,
                            a.hidden_columns,reduction,int(part),count):0.f;
        __syncthreads();
        if(active&&part_worker==0&&n<unsigned(a.hidden_columns))
        {
            const float partial=persistentMoEFFNFold(partials+size_t(batch_route)*maximum*32,
                column,32,count);
            if(is_shared)
                // The terminal role retires every partial before its first
                // row becomes the original shared RNmul(1), RNadd(+0) value.
                partials[column]=rocm_native_quantised_accumulate_rn(0.f,
                    rocm_native_quantised_weight_route_rn(1.f,partial));
            else
            {
                const float value=ready?rocm_native_quantised_weight_route_rn(bindings[batch_route].weight,partial):0.f;
                a.route_contributions[size_t(route)*a.hidden_columns+n]=value;
                weighted[size_t(route)*32+column]=value;
            }
        }
        __syncthreads(); // Retire all batch partial readers before LDS reuse.
        if(is_shared)break;
        first_route+=min(int(batch_capacity),a.top_k-first_route);
    }
    if(group==0&&n<unsigned(a.hidden_columns))
    {
        // Preserve the original shared RNmul(1,partial), then RNadd(+0,value).
        const float raw_shared=partials[column];
        const float routed=llaminar2::device_ordered_route_fold::sum(weighted+column,size_t(32),a.top_k);
        a.routed_output[n]=routed;
        // The original dot published its exact scalar before the decoder
        // phases. Read that same CTA-owned slot only now; volatile prevents
        // hoisting this observation into a long-lived register across them.
        const float gate=live?static_cast<volatile float*>(gate_scratch)[0]:0.f;
        const float gated=live?llaminar2::device_fp32_contract::multiply(gate,raw_shared):0.f;
        a.shared_output[n]=gated;
        a.combined_output[n]=live?llaminar2::device_fp32_contract::add(routed,gated):0.f;
    }
}
