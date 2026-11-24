/**
 * @file ROCmNativeLayoutRouterFFNBody.hip.inc
 * @brief Canonical K32-major translation of the fork T1 POST expert phase layout.
 *
 * Gate/up and down use wave-owned original-P tasks with adjacent native
 * columns. Existing64-column CTA down tiles retain the final ordered route fold. Original
 * arithmetic, source role policy and public planes remain unchanged. One
 * occupancy-admitted ordinary entry and composed HC entry call the same inline
 * suffix phase over caller-owned CTA-local storage and admitted grid joins.
 * Phase-local joins publish/retire LDS only; explicit grid joins still own
 * global logits/activation/descriptor publication and cross-CTA retirement.
 * The phase layout is adapted from Castagna Veloce's hc-persist.cu at
 * 3483d462715d61f21ed5877836be35a724446359; see ROCmCastagnaT1.LICENSE.
 */
#include "ROCmNativeLayoutRouterFFN.h"
#include "ROCmNativeLayoutDownGeometry.h"
#include "ROCmNativeLayoutGateUpGeometry.h"
#include "../moe/ROCmOriginalRouterPhase.hip.inc"
#include "ROCmNativeLayoutPreload.hip.inc"
#include "../common/ROCmKernargField.h"
#include "../common/ROCmWorkgroupJoin.h"
namespace llaminar2::rocm::native_layout_ffn {
#if defined(__AMDGCN_WAVEFRONT_SIZE)
static_assert(__AMDGCN_WAVEFRONT_SIZE==rocmOrderedWaveWidth,
    "Native-layout cold geometry must match the compiled wave contract");
#endif
/** @brief Load the original nested FFN field without making its enclosing record addressable. */
#define LLAMINAR_PHASE_FFN_FIELD(Field) \
    phaseKernargField<NativeLayoutRouterFFNArgs,decltype(ResidentMoEFFNArgs::Field), \
        offsetof(NativeLayoutRouterFFNArgs,ffn)+offsetof(ResidentMoEFFNArgs,Field)>()
/**
 * @brief Load the five original policy fields only when their role starts.
 * @tparam Offset Exact policy position in the actual first kernel argument.
 * @return Original captured partition table and maximum, without an aggregate load.
 */
template<std::size_t Offset>
__device__ __forceinline__ ResidentMoEPartitionPolicy phasePolicy()
{
    using A=NativeLayoutRouterFFNArgs;using P=ResidentMoEPartitionPolicy;
    return {phaseKernargField<A,decltype(P::p0),Offset+offsetof(P,p0)>(),
        phaseKernargField<A,decltype(P::p1),Offset+offsetof(P,p1)>(),
        phaseKernargField<A,decltype(P::p2),Offset+offsetof(P,p2)>(),
        phaseKernargField<A,decltype(P::p3),Offset+offsetof(P,p3)>(),
        phaseKernargField<A,decltype(P::maximum),Offset+offsetof(P,maximum)>()};
}
#define LLAMINAR_PHASE_POLICY(Field) \
    phasePolicy<offsetof(NativeLayoutRouterFFNArgs,ffn)+offsetof(ResidentMoEFFNArgs,Field)>()
/** @brief Load only a role's captured maximum for common scratch strides. */
#define LLAMINAR_PHASE_MAXIMUM(Field) \
    phaseKernargField<NativeLayoutRouterFFNArgs,decltype(ResidentMoEPartitionPolicy::maximum), \
        offsetof(NativeLayoutRouterFFNArgs,ffn)+offsetof(ResidentMoEFFNArgs,Field)+ \
        offsetof(ResidentMoEPartitionPolicy,maximum)>()
/** @brief Read one role's exact execution mask from the original captured bytes. */
#define LLAMINAR_PHASE_MASK(Field) \
    phaseKernargField<NativeLayoutRouterFFNArgs,decltype(ResidentMoEFFNExecutionMasks::Field), \
        offsetof(NativeLayoutRouterFFNArgs,ffn)+offsetof(ResidentMoEFFNArgs,execution)+ \
        offsetof(ResidentMoEFFNExecutionMasks,Field)>()
/** @brief Original shared-gate tree mapped to256logical lanes of a1024thread CTA. */
__device__ __forceinline__ float sharedGate(const float* x,const float* weights,unsigned H,float* scratch)
{
    float sum=0.f;
    if(threadIdx.x<256)
    {
        const auto* a=reinterpret_cast<const float4*>(x);const auto* b=reinterpret_cast<const float4*>(weights);
        for(unsigned i=threadIdx.x;i<H/4;i+=256)
        {const float4 av=a[i],bv=b[i];sum+=av.x*bv.x+av.y*bv.y+av.z*bv.z+av.w*bv.w;}
        sum=moe_wave_contract::rocm_moe_wave_reduce_sum_f32(sum);
        if(threadIdx.x%warpSize==0)scratch[threadIdx.x/warpSize]=sum;
    }
    workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
    if(threadIdx.x<warpSize)
    {
        float v=threadIdx.x<(256+warpSize-1)/warpSize?scratch[threadIdx.x]:0.f;
        v=moe_wave_contract::rocm_moe_wave_reduce_sum_f32(v);
        if(threadIdx.x==0)scratch[0]=1.f/(1.f+expf(-v));
    }
    workgroupJoin<WorkgroupJoinMemory::LDSOnly>();return scratch[0];
}
/** @brief Exact original hidden/activation Q8 publication, using original32lane groups. */
__device__ __forceinline__ void quantize(const float* x,unsigned count,int8_t* codes,float* scales,int32_t* sums)
{
    for(unsigned i=threadIdx.x;i<count;i+=1024)persistentMoEFFNQuantize<moe_wave_contract::QuantizerXorPolicy::TargetNative>(x[i],i,codes,scales,sums);
    workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
}
/**
 * @brief Publish one role's metadata using the same source-policy readiness authority.
 * @param descriptor Original prepared source. @param policy Exact original partition policy.
 * @param mask Original physical format envelope. @param selected Original route selection.
 * @param columns This role's original complete N in elements.
 * @param reduction This role's original complete K in elements.
 * @return Only immutable borrowed planes and the original partition width.
 */
__device__ __forceinline__ PhaseProjection phaseProjection(const DeviceNativeQuantisedMatrixDesc* descriptor,
    ResidentMoEPartitionPolicy policy,uint32_t mask,bool selected,unsigned columns,unsigned reduction)
{
    const bool ready=selected&&descriptor&&descriptor->payload&&descriptor->scales;
    const unsigned parts=ready?unsigned(residentMoEPartitions(descriptor,policy,mask)):1u;
    PhaseProjection result{};
    result.parts=parts;result.ready=ready;
    result.partition_blocks=(reduction/32+parts-1)/parts;
    if(ready)
    {
        result.payload=descriptor->payload;result.scales=static_cast<const uint16_t*>(descriptor->scales);
        result.mins=static_cast<const uint16_t*>(descriptor->mins);result.emins=static_cast<const uint32_t*>(descriptor->emins);
        result.codebook=descriptor->codebook_id;
        result.metadata_layout=descriptor->metadata_layout;
        // Keep policy failure fatal before this independent numeric-zero decision.
        result.numeric_validity=PhaseProjectionValidation::classify(*descriptor,columns,reduction);
    }
    return result;
}
/**
 * @brief Consume a producer-issued projection without repeating immutable validation.
 * @param projection SAME prepared/acquired source view. @param policy Cold sealed role tree.
 * @param mask Captured physical envelope. @param reduction Captured complete reduction width.
 * @return Original phase metadata and arithmetic partitions, with explicit remote/unready state.
 * @details Physical dispatch remains checked against the live descriptor. This
 * overload accepts no raw pointer, shape guess or caller-supplied validity flag.
 */
__device__ __forceinline__ PhaseProjection phaseProjection(
    const PublishedNativeProjection& projection,ResidentMoEPartitionPolicy policy,
    uint32_t mask,unsigned reduction)
{
    PhaseProjection result{};
    result.parts=1;result.partition_blocks=reduction/32;
    if(!projection)return result;
    const auto& descriptor=projection.descriptor();
    if(!residentMoEFFNDecoderAllows(mask,descriptor.codebook_id))__builtin_trap();
    result.parts=policy.maximum;
    result.partition_blocks=(reduction/32+result.parts-1)/result.parts;
    result.payload=descriptor.payload;result.scales=static_cast<const uint16_t*>(descriptor.scales);
    result.mins=static_cast<const uint16_t*>(descriptor.mins);result.emins=static_cast<const uint32_t*>(descriptor.emins);
    result.codebook=descriptor.codebook_id;result.ready=true;
    result.metadata_layout=descriptor.metadata_layout;
    result.numeric_validity=PhaseProjectionValidation::State::Valid;
    return result;
}
/** @brief Dispatch one original role after phase-local metadata publication.
 * @tparam Issue Internal activation issue contract, defaulting to down words-only preload.
 * @tparam Mask Admitted immutable physical decoder envelope.
 * @tparam Reader Default original generic loads or an explicitly proven weight-only reader.
 * @tparam SplitRepresentation Select the role metadata representation once per
 *         partition (two loops) instead of a uniform per-block branch.
 * @tparam Binding Existing native LDS or generic metadata reference; never copied.
 * @tparam Grid Original device table or exact published lookup view.
 * @param grid Immutable lookup bytes retained for the original role consumer. */
template<uint32_t Mask,ActivationIssue Issue=ActivationIssue::DownWordsPreload,
    class Reader=NativeLayoutWeightReader,bool SplitRepresentation=false,class Binding,class Grid=NativeQuantisedGridLookup>
__device__ __forceinline__ float phasePartial(const Binding& p,unsigned n,const int8_t* code,const float* scale,
    const int32_t* sums,unsigned N,unsigned K,unsigned part,Grid grid={})
{
    // The enclosing gate team or down batch owns one immutable record per
    // physical wave. Preserve that authority at the dispatcher as well as
    // within each decoder; column and partition eligibility remain lane-local.
    if(!waveMetadataValue(p.ready)||n>=N||part>=waveMetadataValue(p.parts))return 0.f;
#define LAYOUT_CASE(F) case F:if constexpr(Mask&(1u<<unsigned(F)))return phaseRolePartial<F,Issue,Reader,SplitRepresentation>(p,n,code,scale,sums,N,K,part,grid);else break;
    switch(waveMetadataValue(p.codebook))
    {
    LAYOUT_CASE(NATIVE_QUANTISED_Q4_0) LAYOUT_CASE(NATIVE_QUANTISED_IQ4_NL)
    LAYOUT_CASE(NATIVE_QUANTISED_IQ4_XS_SOURCE) LAYOUT_CASE(NATIVE_QUANTISED_MXFP4) LAYOUT_CASE(NATIVE_QUANTISED_Q4_1)
    LAYOUT_CASE(NATIVE_QUANTISED_Q5_0) LAYOUT_CASE(NATIVE_QUANTISED_Q5_1)
    LAYOUT_CASE(NATIVE_QUANTISED_Q6_K) LAYOUT_CASE(NATIVE_QUANTISED_Q3_K) LAYOUT_CASE(NATIVE_QUANTISED_Q2_K)
    LAYOUT_CASE(NATIVE_QUANTISED_IQ3_S) LAYOUT_CASE(NATIVE_QUANTISED_IQ3_XXS)
    LAYOUT_CASE(NATIVE_QUANTISED_IQ2_S) LAYOUT_CASE(NATIVE_QUANTISED_IQ2_XS) LAYOUT_CASE(NATIVE_QUANTISED_IQ2_XXS)
    LAYOUT_CASE(NATIVE_QUANTISED_IQ1_S) LAYOUT_CASE(NATIVE_QUANTISED_IQ1_M)
    LAYOUT_CASE(NATIVE_QUANTISED_Q8_0)
    default:break;
    }
#undef LAYOUT_CASE
    __builtin_trap();return 0.f;
}
/** @brief Preload Q4 gate/up together; every other format keeps independent original role traversal.
 * @tparam Mask Admitted immutable physical decoder envelope.
 * @tparam Binding Existing pair of native LDS metadata references; never copied. */
template<uint32_t Mask,class Binding>
__device__ __forceinline__ void gateUp(const Binding& gate,const Binding& up,unsigned n,const int8_t* code,const float* scale,
    const int32_t* sums,unsigned N,unsigned K,unsigned part,float& g,float& u)
{
    if constexpr(Mask&(1u<<unsigned(NATIVE_QUANTISED_Q4_1)))
        if(waveMetadataValue(gate.ready)&&waveMetadataValue(up.ready)&&n<N&&part<waveMetadataValue(gate.parts)&&
           waveMetadataValue(gate.parts)==waveMetadataValue(up.parts)&&
           waveMetadataValue(gate.codebook)==NATIVE_QUANTISED_Q4_1&&waveMetadataValue(up.codebook)==NATIVE_QUANTISED_Q4_1)
        {
            pairedRolePartial<NATIVE_QUANTISED_Q4_1>(gate,up,n,code,scale,sums,N,K,part,g,u);
            return;
        }
    g=phasePartial<Mask,ActivationIssue::GateUpBlockPreload>(gate,n,code,scale,sums,N,K,part);
    u=phasePartial<Mask,ActivationIssue::GateUpBlockPreload>(up,n,code,scale,sums,N,K,part);
}
/**
 * @brief Transfer exact FP32 bits from a proven absolute lane of this ordered wave.
 * @param value This lane's original partial. @param source Absolute physical lane0..63.
 * @return The selected lane's identical bits, using the selected SDK's own native primitive.
 * @pre All64 lanes participate and reconverge. Source is either column for P1
 * or the sourceLane result of the validated original-P geometry, never a user index.
 * @details HIP __shfl(value,source,64) normalizes index as
 * (source&63)+(self&~63). Both source and hardware self lie in0..63 here, so
 * this index is exactly source. Keep the SDK's compile-time builtin choice and
 * fatal invocability guard; remove only redundant group-base/self-lane arithmetic.
 * This is the same hardware bit transfer, not another reduction or quantizer.
 */
__device__ __forceinline__ float absoluteWaveValue(float value,unsigned source)
{
    static_assert(rocmOrderedWaveWidth==64,"Absolute partial transfer requires the ordered wave64 program");
    if(__builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_bpermute))
    {
        const auto bits=__builtin_bit_cast(int32_t,value);
#if __has_builtin(__builtin_amdgcn_wave_shuffle)
        const auto moved=__builtin_amdgcn_wave_shuffle(bits,static_cast<int>(source));
#else
        const auto moved=__builtin_amdgcn_ds_bpermute(static_cast<int>(source<<2),bits);
#endif
        return __builtin_bit_cast(float,moved);
    }
    __builtin_trap();
}
/**
 * @brief Materialize the exact physical wave64 lane at its consuming phase.
 * @return This lane's unchanged index0..63, independent of other phases' coordinates.
 * @pre Every active64-lane wave enters convergently; the compiled wave64 contract
 * already binds all canonical indexed folds and native column ownership.
 * @details These are the same two integer MBCNT operations used by HIP's lane
 * query. Explicit volatile instructions prevent one selector index from being
 * kept through all later decoders merely to serve their identical lane queries.
 * No input register, FP operation, memory clobber, fence or join is introduced.
 * The early-clobber output also excludes accidental operand aliasing if the
 * helper is later inlined around other explicit instruction operands.
 */
__device__ __forceinline__ unsigned phasePhysicalLane64()
{
    static_assert(rocmOrderedWaveWidth==64);
#if defined(__HIP_DEVICE_COMPILE__)
    unsigned lane;
    asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0\n\t"
                 "v_mbcnt_hi_u32_b32 %0, -1, %0" : "=&v"(lane));
    return lane;
#else
    // Host compilation only parses this device helper; it is never executed.
    return unsigned(threadIdx.x)&63u;
#endif
}
/**
 * @brief Fold one wave-owned canonical source partial in the original P order.
 * @param partial This lane's original independent partition result, already complete.
 * @param parts Original descriptor-local P, bounded by the canonical source contract.
 * @param geometry Original-P-derived wave mapping. @param column Output within the wave.
 * @return P1's bit-preserving direct partial, or the original +0/increasing-P RN chain.
 * @pre All64 lanes reconverge after decoding and execute every shuffle. Parts
 * and geometry are wave-uniform; preparation and residentMoEPartitions prove
 * 1<=parts<=geometry.workers from the same immutable original source policy.
 * No inactive column or partition lane returns early from this collective.
 */
__device__ __forceinline__ float foldWavePartial(float partial,unsigned parts,
    NativeLayoutDownWaveGeometry geometry,unsigned column)
{
    if(parts==1)return absoluteWaveValue(partial,column);
    float value=0.f;
    for(unsigned part=0;part<parts;++part)
        value=rocm_native_quantised_accumulate_rn(value,
            absoluteWaveValue(partial,geometry.sourceLane(part,column)));
    return value;
}
/**
 * @brief Move original bits left within a16-lane row, without a floating-point operation.
 * @tparam Shift Constant row distance in4,8,12, always inside the original row.
 * @param value Original bank-gathered partial bits.
 * @return Row-local source bits; out-of-row destinations receive zero and are never published.
 * @pre All64 lanes participate. Only the first Columns lanes are numerical consumers.
 * @details The register boundary forbids folding this move into a source-swapped
 * DPP add: the following canonical RN call keeps accumulator first, term second,
 * including its existing NaN behavior. No memory order or grid/CTA join is added.
 */
template<unsigned Shift>
__device__ __forceinline__ float orderedRowValue(float value)
{
    static_assert(Shift==4||Shift==8||Shift==12);
    if(__builtin_amdgcn_is_invocable(__builtin_amdgcn_update_dpp))
    {
        const int32_t bits=__builtin_bit_cast(int32_t,value);
        int32_t moved=__builtin_amdgcn_update_dpp(0,bits,0x100+Shift,0xf,0xf,true);
#if defined(__HIP_DEVICE_COMPILE__)
        asm volatile("" : "+v"(moved));
#endif
        return __builtin_bit_cast(float,moved);
    }
    __builtin_trap();
}
/**
 * @brief Append one original16-lane row bank in exact increasing-P order.
 * @tparam Columns Original adjacent-column extent,4 or8.
 * @tparam Bank Original16-lane row bank0..3.
 * @param partial Unchanged original lane-owned P partial.
 * @param parts This role's actual original P, never the padded worker bound.
 * @param lane Original physical lane0..63, already phase-local in the caller.
 * @param accumulator Original +0/RN prefix, updated only for p<parts.
 * @pre All64 lanes enter every live bank together. Only worker0's result is consumed.
 */
template<unsigned Columns,unsigned Bank>
__device__ __forceinline__ void appendOrderedBank(float partial,unsigned parts,unsigned lane,float& accumulator)
{
    static_assert((Columns==4||Columns==8)&&Bank<4);
    constexpr unsigned first=Bank*(16/Columns);
    if(parts>first)
    {
        // No bank value survives its own ordered appends. Every input shuffle
        // reads the ORIGINAL partial, never another bank's accumulated result.
        float bank=partial;
        if constexpr(Bank!=0)
        {
            // Derive this bank's source only after the preceding original RN
            // prefix. Otherwise the optimizer can retain all bank coordinates
            // through the entire weight/decode loop, although no fold uses them
            // there. This register-only identity changes no bits or ordering.
            unsigned current_lane=lane;
#if defined(__HIP_DEVICE_COMPILE__)
            asm volatile("" : "+v"(current_lane) : "v"(accumulator));
#endif
            bank=absoluteWaveValue(partial,(current_lane+16*Bank)&63u);
        }
        accumulator=rocm_native_quantised_accumulate_rn(accumulator,bank);
        if(parts>first+1)
            accumulator=rocm_native_quantised_accumulate_rn(accumulator,orderedRowValue<Columns>(bank));
        if constexpr(Columns==4)
        {
            if(parts>first+2)
                accumulator=rocm_native_quantised_accumulate_rn(accumulator,orderedRowValue<8>(bank));
            if(parts>first+3)
                accumulator=rocm_native_quantised_accumulate_rn(accumulator,orderedRowValue<12>(bank));
        }
    }
}
/**
 * @brief Preserve the exact original fold for GU/down output owners, using row-local bit moves where useful.
 * @param partial Original completed lane-owned P partial. @param parts Actual original role P.
 * @param geometry Existing original-P worker/column geometry. @param column Output column inside the wave.
 * @param lane Original phase-local physical lane0..63, not a new hardware-lane query.
 * @return Exact canonical result ONLY for worker0 lanes; other lane results are deliberately unspecified.
 * @pre All64 lanes reconverge and enter. Caller consumes the result exclusively
 * under geometry.laneWorker(lane)==0. Sources remain active through every bank transfer.
 * @details P1 keeps the original direct-bit transfer. Columns4/8 gather each later
 * row bank once, then append each ORIGINAL p via a fixed row move and the SAME
 * accumulator-first RN operation. This is a linear ordered fold, never the
 * fork's reassociated four-stage tree. No p>=parts add is executed. Columns16/32/64
 * retain the existing exact gather. No final broadcast restores unused results.
 */
__device__ __forceinline__ float orderedWorkerZeroFold(float partial,unsigned parts,
    NativeLayoutDownWaveGeometry geometry,unsigned column,unsigned lane)
{
    if(parts==1)return absoluteWaveValue(partial,column);
    float value=0.f;
    if(geometry.columns==4)
    {
        appendOrderedBank<4,0>(partial,parts,lane,value);
        appendOrderedBank<4,1>(partial,parts,lane,value);
        appendOrderedBank<4,2>(partial,parts,lane,value);
        appendOrderedBank<4,3>(partial,parts,lane,value);
        return value;
    }
    if(geometry.columns==8)
    {
        appendOrderedBank<8,0>(partial,parts,lane,value);
        appendOrderedBank<8,1>(partial,parts,lane,value);
        appendOrderedBank<8,2>(partial,parts,lane,value);
        appendOrderedBank<8,3>(partial,parts,lane,value);
        return value;
    }
    return foldWavePartial(partial,parts,geometry,column);
}
/**
 * @brief Publish each needed expert's role pair once, then execute independent gate/up waves.
 * @tparam Shared Original shared role computes even for a live0 row.
 * @tparam Mask Original complete physical decoder envelope.
 * @param width Original role N. @param hidden Original complete K.
 * @param routes Original selected slot count. @param experts Original descriptor extent.
 * @param live Original device-owned row decision.
 * @param maximum Existing common captured gate-policy maximum, covering each role's own resolved P.
 * @param gates Original shared descriptor or routed source table. @param ups Original matching up descriptor/table.
 * @param route_ids Same CTA's original canonical selected IDs.
 * @param bank_storage Existing issued acquired-bank token, unchanged across this phase.
 * @param codes Original CTA-local hiddenQ8. @param scales Original block scales. @param sums Original signed-code sums.
 * @param activated Original activated prefix. @param output_base Original phase offset in that prefix.
 * @param gate_output Original public shared gate plane. @param up_output Original public shared up plane.
 * @param bindings Existing dynamic P-plane storage, retyped as at most2*topK original role records.
 * @pre Every1024-thread CTA enters; each nonempty CTA reaches the sole metadata join.
 * The dynamic hiddenQ8/scale/sum prefix is a multiple of8, meeting PhaseProjection alignment.
 * @details The same captured gate policy/mask classifies BOTH original roles.
 * Neither source P nor paired-Q4 readiness/shape behavior is changed. Each wave
 * finishes its original K32 chains and then folds each role's own P in ascending
 * order, including the bit-preserving P1 case. Metadata stays immutable through
 * all wave items; the immediately following unconditional grid join retires it.
 * No partial plane, per-output-tile metadata publication, or task CTA join remains.
 */
template<bool Shared,uint32_t Mask>
__device__ __forceinline__ void gateUpWavePhase(unsigned width,unsigned hidden,unsigned routes,unsigned experts,bool live,
    unsigned maximum,
    const DeviceNativeQuantisedMatrixDesc* gates,const DeviceNativeQuantisedMatrixDesc* ups,const int* route_ids,
    const router_native::RuntimeDecodeExecutionBankStorage<DeviceMoELayerRuntimeView>* bank_storage,
    const int8_t* codes,const float* scales,const int32_t* sums,float* activated,size_t output_base,
    float* gate_output,float* up_output,PhaseProjection* bindings)
{
    using Tasks=NativeLayoutGateUpGeometry;
    const auto resolved=NativeLayoutDownWaveGeometry::resolve(maximum);
    const NativeLayoutDownWaveGeometry geometry{
        waveMetadataValue(resolved.workers),waveMetadataValue(resolved.columns)};
    const unsigned phase_routes=Shared?1u:routes;
    const auto tiles=Tasks::tiles(width,geometry.columns);
    const auto tasks=Tasks::tasks(width,phase_routes,geometry.columns);
    // An empty CTA never reads or publishes a role record. All of its lanes
    // return together and still enter the original unconditional grid join.
    NativeLayoutDownRowRange shared_range{};
    if constexpr(Shared)
    {
        shared_range=Tasks::sharedRange(width,geometry.columns,blockIdx.x,gridDim.x);
        if(shared_range.begin==shared_range.end)return;
    }
    else if(Tasks::firstTask(blockIdx.x,0)>=tasks)return;
    const unsigned leader=threadIdx.x;
    // Every nonempty shared CTA owns route0 after balancing; routed records
    // retain the original expert-major grid-stride ownership predicate.
    if(leader<phase_routes&&(Shared||Tasks::ctaOwnsRoute(leader,tiles,blockIdx.x,gridDim.x)))
    {
        ResidentMoEPartitionPolicy policy;uint32_t execution;
        if constexpr(Shared)
        {
            policy=LLAMINAR_PHASE_POLICY(shared_gate_policy);
            execution=LLAMINAR_PHASE_MASK(shared_gate);
        }
        else
        {
            policy=LLAMINAR_PHASE_POLICY(routed_gate_policy);
            execution=LLAMINAR_PHASE_MASK(routed_gate);
        }
        const DeviceNativeQuantisedMatrixDesc *g=nullptr,*u=nullptr;
        bool selected=false;
        if constexpr(Shared){g=gates;u=ups;selected=true;}
        else if(live)
        {
            const int expert=route_ids[leader];
            if(expert>=0&&expert<int(experts))
            {
                const auto acquired=bank_storage->borrow();
                const auto* runtime=acquired.runtime();
                const auto& bank=acquired.bank();
                const auto row=router_native::runtime_intermediate_shard_row(runtime,bank,expert);
                selected=row.disposition==MoEIntermediateShardRowDisposition::Local&&
                    bank.experts[expert].weight_format==DeviceMoEWeightFormat::NativeQuantised;
                if(selected){g=&bank.experts[expert].gate;u=&bank.experts[expert].up;}
            }
        }
        auto* publication=currentFormatAddress((CurrentFormatLDSValue<PhaseProjection>*)bindings);
        #pragma unroll
        for(unsigned role=0;role<2;++role)
        {
            const auto record=phaseProjection(role==0?g:u,policy,execution,selected,width,hidden);
            auto& value=publication[leader*2+role];
            value.payload=record.payload;value.scales=record.scales;
            value.mins=record.mins;value.emins=record.emins;
            value.parts=record.parts;value.partition_blocks=record.partition_blocks;
            value.codebook=record.codebook;value.ready=record.ready;
            value.numeric_validity=record.numeric_validity;value.metadata_layout=record.metadata_layout;
        }
    }
    workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
    const unsigned wave=waveMetadataValue(unsigned(threadIdx.x)/rocmOrderedWaveWidth);
    const unsigned lane=phasePhysicalLane64();
    const unsigned column=geometry.laneColumn(lane),worker=geometry.laneWorker(lane);
    // Shared items retain their exact source tile index but are distributed
    // cardinally over resident CTAs instead of packing the first16 into CTA0.
    // Each wave remains complete, including source tails and live0 shared work.
    const auto first=Shared?Tasks::Index(shared_range.begin)+wave:Tasks::firstTask(blockIdx.x,wave);
    const auto end=Shared?Tasks::Index(shared_range.end):tasks;
    const auto stride=Shared?Tasks::Index(Tasks::waves):Tasks::stride(gridDim.x);
    for(auto task=first;task<end;task+=stride)
    {
        const auto item=Tasks::coordinate(task,tiles,geometry.columns);
        const unsigned n=item.first_column+column;
        const auto* role_bindings=currentFormatAddress((CurrentFormatLDSValue<const PhaseProjection>*)bindings);
        const auto& gate=role_bindings[item.route*2];
        const auto& up=role_bindings[item.route*2+1];
        float g=0.f,u=0.f;
        gateUp<Mask>(gate,up,n,codes,scales,sums,width,hidden,worker,g,u);
        // Every64-lane wave reconverges before either indexed transfer. Gate and
        // up may resolve different P; retain their independent original folds.
        g=orderedWorkerZeroFold(g,waveMetadataValue(gate.parts),geometry,column,lane);
        u=orderedWorkerZeroFold(u,waveMetadataValue(up.parts),geometry,column,lane);
        if(worker==0&&n<width)
        {
            if constexpr(Shared){gate_output[n]=g;up_output[n]=u;}
            activated[output_base+(Shared?0:size_t(item.route)*width)+n]=
                waveMetadataValue(gate.ready)&&waveMetadataValue(up.ready)?decode_equivalent::swigluValue(g,u):0.f;
        }
    }
}
/** @brief Explicit issuer layout; existing EB4 consumers retain padded records. */
enum class DownMetadataPublication : uint8_t { PacketLeaders, LiveWaveLeaders };
/**
 * @brief Snapshot selected down descriptors once, before any independent wave task.
 * @param hidden Original output N.
 * @param routed_width Original routed K. @param shared_width Original shared K.
 * @param routes Original topK. @param experts Original descriptor extent. @param live Original row decision.
 * @param descriptors Original acquired-epoch selected down table. @param shared Original shared descriptor.
 * @param ids Original device-selected IDs. @param weights Original route weights.
 * @param bindings Existing64-record phase storage, immutable until down finishes.
 * @param route_weights Existing scalar weight storage.
 * @pre All1024 lanes enter after selection/Q8 consumers retire. No helper owns a new byte.
 * @tparam Schedule Existing padded EB4 publication or full-entry live-wave issuers.
 */
/** @brief Default checked standalone descriptor source; no alternate storage. */
struct CheckedDownProjectionSource {};
template<DownMetadataPublication Schedule=DownMetadataPublication::PacketLeaders,
    class Source=CheckedDownProjectionSource>
__device__ __forceinline__ void publishDownPhase(unsigned hidden,unsigned routed_width,unsigned shared_width,
    unsigned routes,unsigned experts,bool live,const DeviceNativeQuantisedMatrixDesc* descriptors,
    const DeviceNativeQuantisedMatrixDesc* shared,const int* ids,const float* weights,
    PhaseProjection* bindings,float* route_weights,Source source={})
{
    constexpr unsigned shared_slot=NativeLayoutDownGeometry::maximum_capacity-1;
    static_assert(Schedule==DownMetadataPublication::PacketLeaders||Schedule==DownMetadataPublication::LiveWaveLeaders);
    constexpr bool wave_leaders=Schedule==DownMetadataPublication::LiveWaveLeaders;
    const unsigned packets=(routes+3u)/4u;
    // Full-entry consumers read only actual routes and the fixed shared slot.
    // The default EB4 schedule continues authoring its final packet padding.
    const unsigned first=wave_leaders?waveMetadataValue(unsigned(threadIdx.x)/rocmOrderedWaveWidth):unsigned(threadIdx.x);
    const unsigned count=wave_leaders?routes+1u:packets+1u;
    const unsigned stride=wave_leaders?NativeLayoutDownGeometry::waves:1024u;
    static_assert(kDeviceMoEMaxTopK<shared_slot);
    if(!wave_leaders||threadIdx.x%rocmOrderedWaveWidth==0)
    for(unsigned item=first;item<count;item+=stride)
    {
        const bool is_shared=item==(wave_leaders?routes:packets);
        ResidentMoEPartitionPolicy policy;uint32_t execution;unsigned reduction;
        if(is_shared)
        {
            reduction=shared_width;policy=LLAMINAR_PHASE_POLICY(shared_down_policy);
            execution=LLAMINAR_PHASE_MASK(shared_down);
        }
        else
        {
            reduction=routed_width;policy=LLAMINAR_PHASE_POLICY(routed_down_policy);
            execution=LLAMINAR_PHASE_MASK(routed_down);
        }
        // Each issuer borrows the same original role policy. EB4 retains its
        // four-record loop; live wave issuers own one record per counted visit.
        for(unsigned e=0;e<(is_shared||wave_leaders?1u:4u);++e)
        {
            const unsigned route=wave_leaders?item:item*4+e,index=is_shared?shared_slot:route;
            const DeviceNativeQuantisedMatrixDesc* descriptor=nullptr;bool selected=false;
            PhaseProjection record{};
            if constexpr(std::is_same_v<Source,CheckedDownProjectionSource>) {
                if(is_shared){descriptor=shared;selected=true;}
                else selected=live&&route<routes&&moe_native_quantised_load_down_descriptor<false>(
                    descriptors,nullptr,ids[route],experts,descriptor);
                record=phaseProjection(descriptor,policy,execution,selected,hidden,reduction);
            } else {
                // The typed full source issues views only from the SAME acquired
                // row or retained shared owner. Padding never reads ids[route].
                const int expert=!is_shared&&live&&route<routes?ids[route]:-1;
                const auto projection=source.projection(is_shared,expert);
                selected=bool(projection);
                record=phaseProjection(projection,policy,execution,reduction);
            }
            auto* publication=currentFormatAddress((CurrentFormatLDSValue<PhaseProjection>*)bindings);
            auto& value=publication[index];
            value.payload=record.payload;value.scales=record.scales;value.mins=record.mins;value.emins=record.emins;
            value.parts=record.parts;value.partition_blocks=record.partition_blocks;
            value.codebook=record.codebook;value.ready=record.ready;
            value.numeric_validity=record.numeric_validity;value.metadata_layout=record.metadata_layout;
            if(!is_shared)
            {
                auto* publication_weights=currentFormatAddress((CurrentFormatLDSValue<float>*)route_weights);
                publication_weights[index]=selected?weights[route]:0.f;
            }
        }
    }
    workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
}
/**
 * @brief Publish one matching route only after its original complete P fold.
 * @tparam Binding Immutable original phase metadata, retaining its address space.
 * @param partial Exact independent original p result. @param binding Original matching ready record.
 * @param route Original selected slot. @param n Original output column. @param hidden Original H.
 * @param local Column inside the existing64-column tile. @param outputs Actual tile extent.
 * @param worker Original P worker. @param column Output inside this wave.
 * @param geometry Original-P lane mapping. @param route_weights Original immutable route weights.
 * @param weighted Existing route-major64-stride result plane.
 * @pre Every64-lane wave enters under the same ready/codebook predicate and
 * reconverges after partial calculation; source tail/P validity never gates a shuffle.
 */
template<class Binding>
__device__ __forceinline__ void finishRoutedPartial(float partial,const Binding& binding,unsigned route,
    unsigned n,unsigned hidden,unsigned local,unsigned outputs,unsigned worker,unsigned column,
    NativeLayoutDownWaveGeometry geometry,const float* route_weights,float* weighted)
{
    const float value=foldWavePartial(partial,waveMetadataValue(binding.parts),geometry,column);
    if(worker==0&&local<outputs)
    {
        const auto* weights=currentFormatAddress((CurrentFormatLDSValue<const float>*)route_weights);
        auto* result=currentFormatAddress((CurrentFormatLDSValue<float>*)weighted);
        auto* contributions=LLAMINAR_PHASE_FFN_FIELD(route_contributions);
        const float output=rocm_native_quantised_weight_route_rn(weights[route],value);
        contributions[size_t(route)*hidden+n]=output;
        result[size_t(route)*NativeLayoutDownGeometry::columns+local]=output;
    }
}
/**
 * @brief Compute, fold and publish one present format before another format starts.
 * @tparam FMT Original physical codebook, selected only from the admitted Mask.
 * @param bindings Full existing immutable metadata table. @param first_route Original EB4 packet start.
 * @param routes Original topK. @param n Original output coordinate. @param hidden Original H.
 * @param reduction Original routed K. @param local Owned tile column. @param outputs Live tile count.
 * @param worker Original partition. @param column Wave-owned output. @param geometry Original-P mapping.
 * @param code Exact packet-local activation prefix. @param scale Original scales. @param sums Original sums.
 * @param route_weights Original route weights. @param weighted Existing completed route values.
 * @pre The present-format bit was built from these same immutable records.
 * All64 lanes execute each matching record's fold; only the final store is lane-masked.
 * @details Non-eight-bit formats retain canonical EB4 payload-first loads. Q8
 * and physical23 retire one expert's canonical partial/fold/store before reading
 * the next, exactly as the fork's sequential arm. No cross-format result survives.
 */
template<NativeQuantisedFormat FMT>
__device__ __forceinline__ void routedFormat(const PhaseProjection* bindings,unsigned first_route,unsigned routes,
    unsigned n,unsigned hidden,unsigned reduction,unsigned local,unsigned outputs,unsigned worker,unsigned column,
    NativeLayoutDownWaveGeometry geometry,const int8_t* code,const float* scale,const int32_t* sums,
    const float* route_weights,float* weighted)
{
    if constexpr(NATIVE_QUANTISEDTraits<FMT>::is_8bit)
    {
        #pragma unroll 1
        for(unsigned e=0;e<4&&first_route+e<routes;++e)
        {
            const auto* current=currentFormatAddress((CurrentFormatLDSValue<const PhaseProjection>*)bindings);
            const auto& binding=current[first_route+e];
            if(waveMetadataValue(binding.ready)&&waveMetadataValue(binding.codebook)==FMT)
            {
                float partial=0.f;
                if(n<hidden&&worker<waveMetadataValue(binding.parts))
                    partial=phaseRolePartial<FMT>(binding,n,code+size_t(e)*reduction,
                        scale+size_t(e)*(reduction/32),sums+size_t(e)*(reduction/32),hidden,reduction,worker);
                finishRoutedPartial(partial,binding,first_route+e,n,hidden,local,outputs,worker,column,
                    geometry,route_weights,weighted);
            }
        }
    }
    else
    {
        const auto value=expertBatchPartial<FMT>(bindings+first_route,n,code,scale,sums,hidden,reduction,worker);
        #pragma unroll
        for(unsigned e=0;e<4;++e)
        {
            const unsigned route=first_route+e;
            const auto* current=currentFormatAddress((CurrentFormatLDSValue<const PhaseProjection>*)bindings);
            const auto& binding=current[route];
            if(route<routes&&waveMetadataValue(binding.ready)&&waveMetadataValue(binding.codebook)==FMT)
            {
                const float partial=e==0?value.x:e==1?value.y:e==2?value.z:value.w;
                finishRoutedPartial(partial,binding,route,n,hidden,local,outputs,worker,column,
                    geometry,route_weights,weighted);
            }
        }
    }
}
/**
 * @brief Select present immutable packet formats and publish unready real slots exactly once.
 * @tparam Mask Original admitted physical envelope; no new format policy is inferred.
 * @param bindings Existing selected metadata. @param first_route Original packet start.
 * @param routes Original topK. @param n Original output column. @param hidden Original H.
 * @param reduction Original routed K. @param local Owned tile coordinate. @param outputs Actual extent.
 * @param worker Original P lane. @param column Wave-owned output. @param geometry Exact lane mapping.
 * @param code Original packet Q8 prefix. @param scale Original scales. @param sums Original sums.
 * @param route_weights Original route weights. @param weighted Existing result plane.
 * @pre Packet, readiness and codebooks are wave-uniform and immutable. Every
 * matching source has exactly one codebook. PhaseProjection authenticates
 * readiness, role mask and source P; the original cold prepared owner owns full
 * shape authentication. All64 lanes enter each selected format.
 * @details At most four format bits exist. Independent formats may retire in
 * bit order because the later canonical route-order fold is unchanged. No CTA
 * barrier or new record is introduced, and no previous format's partial is carried.
 */
template<uint32_t Mask>
__device__ __forceinline__ void routedPacket(const PhaseProjection* bindings,unsigned first_route,unsigned routes,
    unsigned n,unsigned hidden,unsigned reduction,unsigned local,unsigned outputs,unsigned worker,unsigned column,
    NativeLayoutDownWaveGeometry geometry,const int8_t* code,const float* scale,const int32_t* sums,
    const float* route_weights,float* weighted)
{
    uint32_t present=0;
    for(unsigned e=0;e<4&&first_route+e<routes;++e)
    {
        const unsigned route=first_route+e;
        const auto* current=currentFormatAddress((CurrentFormatLDSValue<const PhaseProjection>*)bindings);
        const bool ready=waveMetadataValue(current[route].ready);
        if(ready)
        {
            const unsigned format=waveMetadataValue(current[route].codebook);
            if(format>=32)__builtin_trap();
            present|=uint32_t(1)<<format;
        }
        else if(worker==0&&local<outputs)
        {
            // Only unready real slots take this path. Present-format stores
            // require ready, so no route/column can be initialized twice.
            auto* result=currentFormatAddress((CurrentFormatLDSValue<float>*)weighted);
            auto* contributions=LLAMINAR_PHASE_FFN_FIELD(route_contributions);
            contributions[size_t(route)*hidden+n]=0.f;
            result[size_t(route)*NativeLayoutDownGeometry::columns+local]=0.f;
        }
    }
    present=waveMetadataValue(present);
    while(present)
    {
        const unsigned format=unsigned(__builtin_ctz(present));
#define ROUTED_FORMAT(F) case F:if constexpr(Mask&(1u<<unsigned(F))) \
        routedFormat<F>(bindings,first_route,routes,n,hidden,reduction,local,outputs,worker,column,geometry, \
            code,scale,sums,route_weights,weighted);else __builtin_trap();break;
        switch(format)
        {
        ROUTED_FORMAT(NATIVE_QUANTISED_Q4_0) ROUTED_FORMAT(NATIVE_QUANTISED_IQ4_NL) ROUTED_FORMAT(NATIVE_QUANTISED_IQ4_XS_SOURCE)
        ROUTED_FORMAT(NATIVE_QUANTISED_MXFP4)
        ROUTED_FORMAT(NATIVE_QUANTISED_Q4_1) ROUTED_FORMAT(NATIVE_QUANTISED_Q5_0) ROUTED_FORMAT(NATIVE_QUANTISED_Q5_1)
        ROUTED_FORMAT(NATIVE_QUANTISED_Q6_K) ROUTED_FORMAT(NATIVE_QUANTISED_Q3_K) ROUTED_FORMAT(NATIVE_QUANTISED_Q2_K)
        ROUTED_FORMAT(NATIVE_QUANTISED_IQ3_S) ROUTED_FORMAT(NATIVE_QUANTISED_IQ3_XXS) ROUTED_FORMAT(NATIVE_QUANTISED_IQ2_S)
        ROUTED_FORMAT(NATIVE_QUANTISED_IQ2_XS) ROUTED_FORMAT(NATIVE_QUANTISED_IQ2_XXS) ROUTED_FORMAT(NATIVE_QUANTISED_IQ1_S)
        ROUTED_FORMAT(NATIVE_QUANTISED_IQ1_M) ROUTED_FORMAT(NATIVE_QUANTISED_Q8_0)
        default:__builtin_trap();
        }
#undef ROUTED_FORMAT
        present&=present-1u;
    }
}
/**
 * @brief Execute independent (EB4 packet,output tile) and shared-output wave items.
 * @tparam Mask Exact admitted physical envelope, including mixed packet codebooks.
 * @param begin First original output column in this CTA tile. @param outputs Actual owned count<=64.
 * @param hidden Original full output extent H; also the inactive-column sentinel.
 * @param routed_width Original routed K. @param shared_width Original shared K.
 * @param routes Original topK. @param routed_maximum Original routed source-policy bound.
 * @param shared_maximum Original shared source-policy bound. @param code Original local Q8 concatenation.
 * @param scale Original local scales. @param sums Original local code sums.
 * @param bindings Existing immutable original metadata. @param route_weights Existing original scalar weights.
 * @param shared_output Existing64-value shared-result plane. @param weighted Existing route-major64-stride plane.
 * @pre Every physical wave participates with all64 lanes; only per-wave loop
 * bounds differ. Metadata is immutable and no CTA barrier occurs inside tasks.
 * @details Mirrors fork output-wave/EB4 items while retaining original partitions
 * and route-order publication. The canonical source contract bounds P by16,
 * so every original partition has one worker. Complete decoding before forming
 * any fold accumulator; no hypothetical future bank keeps another float4 alive
 * across EB4 payloads. Generic1..64 geometry proof does not select native policy.
 */
template<uint32_t Mask>
__device__ __forceinline__ void downWaveTasks(unsigned begin,unsigned outputs,unsigned hidden,
    unsigned routed_width,unsigned shared_width,unsigned routes,unsigned routed_maximum,unsigned shared_maximum,
    const int8_t* code,const float* scale,const int32_t* sums,const PhaseProjection* bindings,
    const float* route_weights,float* shared_output,float* weighted)
{
    const auto routed_resolved=NativeLayoutDownWaveGeometry::resolve(routed_maximum);
    // These fields derive only from the immutable original policy maximum.
    // Preserve their wave authority before deriving task/packet addresses.
    const NativeLayoutDownWaveGeometry routed_geometry{
        waveMetadataValue(routed_resolved.workers),waveMetadataValue(routed_resolved.columns)};
    const unsigned routed_tiles=routed_geometry.tiles(outputs);
    const unsigned routed_tasks=routed_geometry.routedTasks(outputs,routes);
    // Like the fork's wu_, every lane owns the same wave task. Column and
    // worker/P arithmetic below remain lane-private and are never broadcast.
    const unsigned wave=waveMetadataValue(unsigned(threadIdx.x)/64);
    const unsigned routed_lane=phasePhysicalLane64();
    for(unsigned task=wave;task<routed_tasks;task+=NativeLayoutDownGeometry::waves)
    {
        const auto coordinates=routed_geometry.routedTask(task,routed_tiles);
        const unsigned first_route=coordinates.first_route;
        const unsigned column=routed_geometry.laneColumn(routed_lane),worker=routed_geometry.laneWorker(routed_lane);
        const unsigned local=coordinates.first_column+column,n=local<outputs?begin+local:hidden;
        const size_t offset=size_t(first_route)*routed_width;
        routedPacket<Mask>(bindings,first_route,routes,n,hidden,routed_width,local,outputs,worker,column,
            routed_geometry,code+offset,scale+offset/32,sums+offset/32,route_weights,weighted);
    }
    // The fork finishes routed experts before its shared expert for each owned
    // output item. Retire routed decoder state before resolving this independent
    // shared role. No CTA join is required: route/shared writers own disjoint
    // planes, and the original final join still precedes their joint consumer.
    const auto shared_resolved=NativeLayoutDownWaveGeometry::resolve(shared_maximum);
    const NativeLayoutDownWaveGeometry shared_geometry{
        waveMetadataValue(shared_resolved.workers),waveMetadataValue(shared_resolved.columns)};
    const unsigned shared_tiles=shared_geometry.tiles(outputs);
    // Routed decoder state has retired. This independent shared-role query
    // must not keep one physical-lane value live across the entire routed phase.
    const unsigned shared_lane=phasePhysicalLane64();
    for(unsigned shared_task=NativeLayoutDownGeometry::firstSharedTask(wave,routed_tasks);
        shared_task<shared_tiles;shared_task+=NativeLayoutDownGeometry::waves)
    {
        constexpr unsigned shared_slot=NativeLayoutDownGeometry::maximum_capacity-1;
        const unsigned column=shared_geometry.laneColumn(shared_lane),worker=shared_geometry.laneWorker(shared_lane);
        const unsigned local=shared_geometry.sharedFirstColumn(shared_task,0)+column,n=local<outputs?begin+local:hidden;
        const size_t offset=size_t(routes)*routed_width;
        const auto partial=phasePartial<Mask>(bindings[shared_slot],n,code+offset,
            scale+offset/32,sums+offset/32,hidden,shared_width,worker);
        const auto* fold_bindings=currentFormatAddress((CurrentFormatLDSValue<const PhaseProjection>*)bindings);
        const float value=foldWavePartial(partial,waveMetadataValue(fold_bindings[shared_slot].parts),shared_geometry,column);
        if(worker==0&&local<outputs)
        {
            auto* result=currentFormatAddress((CurrentFormatLDSValue<float>*)shared_output);
            result[local]=rocm_native_quantised_accumulate_rn(0.f,rocm_native_quantised_weight_route_rn(1.f,value));
        }
    }
    // Unique packet/column writers have finished. Preserve the canonical final
    // CTA-owned route fold, rather than introducing a parallel route reduction.
    workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
}

}
#include "ROCmResidentGridBarrier.cuh"
namespace llaminar2::rocm {
/**
 * @brief Caller-owned static LDS for one complete canonical router/FFN suffix.
 * @details A full entry may reuse retired HC phase storage for this object only
 * after its mixed-hidden publication grid edge. The selector stays live through
 * suffix join 2 and CTA0 probability publication; the existing activation
 * quantizer CTA join retires those reads before the union becomes down metadata.
 * Hidden/activated Q8 and gate/up records occupy the separately admitted dynamic
 * LDS region. No constructor, clear, new allocation or host mirror is required.
 */
struct NativeLayoutFFNPhaseScratch {
    /** @brief Phase-exclusive selector storage followed by original down records. */
    union PhaseStorage {
        router_native::RouterSelectionShared selection; ///< Original probability/ID/weight authority through join 2.
        native_layout_ffn::PhaseProjection down[NativeLayoutDownGeometry::maximum_capacity]; ///< Original down metadata after selector retirement.
    };
    /** @brief Original shared-gate reduction and immutable acquired runtime-bank token. */
    struct GateScratchState {
        float reduction[4]; ///< Original shared-gate tree and retained shared-gate scalar.
        router_native::RuntimeDecodeExecutionBankStorage<DeviceMoELayerRuntimeView> bank; ///< Existing typed launch-epoch borrow.
        float spare[8]; ///< Original reserved bytes; no new consumer or storage contract.
    };
    PhaseStorage phase; ///< Selector/down region, with the original phase lifetime.
    float route_weights[NativeLayoutDownGeometry::maximum_capacity]; ///< Original down route weights, published with down metadata.
    GateScratchState gate_state; ///< Original gate reduction and runtime-bank publication.
    float router_totals[16]; ///< Original eight logical 128 projection teams' two summaries.
};
static_assert(sizeof(NativeLayoutFFNPhaseScratch::PhaseStorage)==
    sizeof(native_layout_ffn::PhaseProjection)*NativeLayoutDownGeometry::maximum_capacity);
static_assert(alignof(native_layout_ffn::PhaseProjection)==8);
static_assert(2*kDeviceMoEMaxTopK<=NativeLayoutDownGeometry::maximum_capacity);
static_assert(rocmOrderedWaveWidth==64 && sizeof(NativeLayoutFFNPhaseScratch::GateScratchState)==16*sizeof(float));
static_assert(sizeof(NativeLayoutFFNPhaseScratch)==sizeof(NativeLayoutFFNPhaseScratch::PhaseStorage)+
    sizeof(float)*(NativeLayoutDownGeometry::maximum_capacity+16+16));
/**
 * @brief Execute the complete canonical router/shared/routed FFN suffix in an admitted full entry.
 * @tparam Mask Original complete physical decoder family, unchanged by composition.
 * @param args Original immutable captured suffix arguments; all source policies and publications are retained.
 * @param grid Caller-owned suffix barrier starting at phase 0, with its distinct two admitted join slots.
 * @param scratch Caller-owned CTA-local static LDS; HC readers have already retired before this borrow.
 * @param storage Caller-owned 16-byte-aligned dynamic LDS, sized by the existing suffix admission calculation.
 * @pre All 1024 threads in every admitted CTA enter after the original mixed-hidden
 * producer has published globally. Actual full-entry occupancy, exact stream,
 * source/table/workspace generations and every embedded pointer are cold-sealed.
 * NativeLayoutRouterFFNArgs MUST remain the actual kernel's first by-value argument:
 * phaseKernargField retains its existing field offsets and reads that authority.
 * @post Both existing suffix grid joins have completed before any no-output CTA
 * returns. Public down outputs retain their original terminal kernel-completion
 * publication; callers adding a later global consumer must supply its ordering.
 * @details This is the original complete body, including canonical logical router
 * teams, Q8 quantizers, per-role P folds, runtime-bank publication and ordered
 * selected-slot fold. It adds no join, numerical operation or alternate policy.
 */
template<uint32_t Mask>
__device__ __forceinline__ void nativeLayoutRouterFFNPhase(const NativeLayoutRouterFFNArgs args,
    ResidentGridBarrier& grid,NativeLayoutFFNPhaseScratch& scratch,unsigned char* storage)
{
    using namespace native_layout_ffn;
    // Keep the argument record immutable; helpers still borrow only scalar fields.
    const auto& a=args.ffn;
    const bool live=a.rows.activeRows()!=0;
    const unsigned tid=threadIdx.x,column=tid%32;
    const unsigned H=a.hidden_columns,F=a.routed_columns,S=a.shared_columns,T=a.top_k;
    const auto* runtime=reinterpret_cast<const DeviceMoELayerRuntimeView*>(a.runtime);
    if(live&&(!runtime||runtime->expert_count!=unsigned(a.experts)||runtime->top_k!=unsigned(a.top_k)||
        !runtime->intermediate_shard.sameIdentity(a.original)))__builtin_trap();
    auto* code=reinterpret_cast<int8_t*>(storage);
    auto* scale=reinterpret_cast<float*>(code+H);
    auto* sum=reinterpret_cast<int32_t*>(scale+H/32);
    auto* planes=reinterpret_cast<float*>(sum+H/32);
    // The old dynamic gate/up P-plane region now holds immutable role records;
    // the original hidden prefix is8-byte aligned and remains live separately.
    // Static selector probabilities/IDs/weights retain their original lifetime
    // through join2 and CTA0 publication. Only the quantizer's existing CTA join
    // retires selection before this same static union becomes down metadata.
    auto& phase_storage=scratch.phase;
    auto* gate_binding=reinterpret_cast<PhaseProjection*>(planes);
    auto* route_weights=scratch.route_weights;
    auto& gate_state=scratch.gate_state;
    auto* router_totals=scratch.router_totals;
    auto* gate_scratch=gate_state.reduction;
    // Each CTA validates its immutable acquired epoch once, including live0.
    // The existing quantizer LDS join precedes every peer token read.
    if(tid==0)gate_state.bank.publish(router_native::AcquiredRuntimeDecodeBank<DeviceMoELayerRuntimeView>::acquire(
        reinterpret_cast<DeviceMoELayerRuntimeView*>(a.runtime)));
    // Original router projection remains eight logical128teams, not a new dot tree.
    for(unsigned first=blockIdx.x*8;first<unsigned(a.experts);first+=gridDim.x*8)
    {
        router_native::projectionExpert128<WorkgroupJoinMemory::LDSOnly>(a.hidden,static_cast<const __half*>(args.router_weights),
            args.router_logits,H,a.experts,first+tid/128,tid%128,router_totals+(tid/128)*2);
        workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
    }
    if(blockIdx.x==0)
    {
        const float shared_gate=live?sharedGate(a.hidden,a.shared_input_gate,H,gate_scratch):0.f;
        if(tid==0)LLAMINAR_PHASE_FFN_FIELD(down_partials)[0]=shared_gate;
    }
    quantize(a.hidden,H,code,scale,sum);
    const unsigned max_parts=max(LLAMINAR_PHASE_MAXIMUM(routed_gate_policy),LLAMINAR_PHASE_MAXIMUM(shared_gate_policy));
    // Match the source's phase boundaries explicitly. Each helper receives
    // only this role's scalar/pointer values; no enclosing args record escapes.
    gateUpWavePhase<true,Mask>(S,H,T,a.experts,live,max_parts,
        LLAMINAR_PHASE_FFN_FIELD(shared_gate),LLAMINAR_PHASE_FFN_FIELD(shared_up),nullptr,nullptr,code,scale,sum,LLAMINAR_PHASE_FFN_FIELD(gate_partials),size_t(T)*F,
        LLAMINAR_PHASE_FFN_FIELD(shared_gate_output),LLAMINAR_PHASE_FFN_FIELD(shared_up_output),gate_binding);
    grid.sync();
    // Only one global scalar load per CTA; the canonical selector's first
    // existing CTA barrier publishes this LDS value to every eventual consumer.
    if(tid==0)gate_scratch[0]=LLAMINAR_PHASE_FFN_FIELD(down_partials)[0];
    auto& selection=phase_storage.selection;
    router_native::selectProbabilities<false,256,false,WorkgroupJoinMemory::LDSOnly>(args.router_logits,nullptr,a.experts,T,
        args.normalize_router_weights,selection);
    if(blockIdx.x==0&&tid<T)
        router_native::runtime_publish_independent_decode_dispatch_from_bank<true,RoutedExpertRowExecutionPolicy::IntermediateShardLocal,true>(
            gate_state.bank.borrow(),selection.selected_ids,selection.selected_weights,
            a.experts,T,args.router_indices,args.router_weights_output,args.write_router_outputs,
            a.rows.countOwner(),args.router_publication);
    gateUpWavePhase<false,Mask>(F,H,T,a.experts,live,max_parts,
        a.gates,a.ups,selection.selected_ids,&gate_state.bank,code,scale,sum,
        LLAMINAR_PHASE_FFN_FIELD(gate_partials),0,nullptr,nullptr,gate_binding);
    grid.sync();
    // Every CTA has finished reading raw logits. Only now may the original
    // public probability array be overwritten; down never consumes it.
    if(blockIdx.x==0)
        for(unsigned expert=tid;expert<unsigned(a.experts);expert+=1024)
            args.router_logits[expert]=selection.expert_probs[expert];
    const size_t values=size_t(T)*F+S;
    code=reinterpret_cast<int8_t*>(storage);scale=reinterpret_cast<float*>(code+values);sum=reinterpret_cast<int32_t*>(scale+values/32);
    planes=reinterpret_cast<float*>(sum+values/32);
    quantize(LLAMINAR_PHASE_FFN_FIELD(gate_partials),values,code,scale,sum);
    // Existing quantizer CTA join also retires CTA0's selection-output reads.
    auto* phase_binding=phase_storage.down;
    const auto routed_down_maximum=LLAMINAR_PHASE_MAXIMUM(routed_down_policy);
    const auto shared_down_maximum=LLAMINAR_PHASE_MAXIMUM(shared_down_policy);
    constexpr unsigned down_columns=NativeLayoutDownGeometry::columns;
    // Only exact completed route/shared values occupy these existing planes.
    // Original partition partials remain registers and never enter LDS.
    float* weighted=planes+down_columns;
    const auto owned=NativeLayoutDownGeometry::rows(H,blockIdx.x,gridDim.x);
    // Both global joins have completed. A CTA with no output owns no down
    // descriptor read or publication; all its lanes retire together here.
    if(owned.begin==owned.end)return;
    publishDownPhase(H,F,S,T,a.experts,live,args.router_publication.down,
        LLAMINAR_PHASE_FFN_FIELD(shared_down),runtime->topk_expert_ids,runtime->topk_weights,
        phase_binding,route_weights);
    for(unsigned first_column=owned.begin;first_column<owned.end;first_column+=down_columns)
    {
        const unsigned outputs=min(down_columns,owned.end-first_column);
        downWaveTasks<Mask>(first_column,outputs,H,F,S,T,routed_down_maximum,shared_down_maximum,
            code,scale,sum,phase_binding,route_weights,planes,weighted);
        // Keep publication-only coordinates at the actual consumer. This is
        // also the pending R38 final-thread predicate's exact late-lane form.
        unsigned publication_thread=threadIdx.x;
        asm volatile("" : "+v"(publication_thread));
        const unsigned down_column=publication_thread;
        const unsigned n=publication_thread<outputs?first_column+publication_thread:H;
        if(publication_thread<down_columns&&n<H)
        {
            // Output pointers are publication-only; do not retain them while
            // either original projection/partition phase is computing.
            auto* routed_output=LLAMINAR_PHASE_FFN_FIELD(routed_output);
            auto* shared_output=LLAMINAR_PHASE_FFN_FIELD(shared_output);
            auto* combined_output=LLAMINAR_PHASE_FFN_FIELD(combined_output);
            // Re-form the proven LDS base at its final consumer. Add the
            // column only after this boundary, so the derived fold pointer
            // cannot remain live through either down projection. The explicit
            // local-to-generic cast preserves the canonical fold's public API;
            // it does not change the pointed-to storage or arithmetic window.
            const auto* final_weighted=currentFormatAddress(
                (CurrentFormatLDSValue<const float>*)weighted);
            const float routed=llaminar2::device_ordered_route_fold::sum(
                (const float*)(final_weighted+down_column),size_t(down_columns),T);
            const float shared=live?llaminar2::device_fp32_contract::multiply(gate_scratch[0],planes[down_column]):0.f;
            routed_output[n]=routed;shared_output[n]=shared;
            combined_output[n]=live?llaminar2::device_fp32_contract::add(routed,shared):0.f;
        }
        workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
    }
}
/**
 * @brief Preserve the existing standalone entry while borrowing its canonical suffix phase.
 * @tparam Mask Original admitted physical decoder family.
 * @param args Original first by-value captured argument, preserving all kernarg field offsets.
 * @details This entry owns exactly the suffix's former static and dynamic CTA
 * storage and two grid slots. Full HC entries call the same phase after their
 * separately ordered prefix; no second router/FFN implementation is retained.
 */
template<uint32_t Mask>
__global__ __launch_bounds__(1024,1) void nativeLayoutRouterFFNKernel(const NativeLayoutRouterFFNArgs args)
{
    extern __shared__ __align__(16) unsigned char storage[];
    __shared__ NativeLayoutFFNPhaseScratch scratch;
    ResidentGridBarrier grid{args.barrier_state};
    nativeLayoutRouterFFNPhase<Mask>(args,grid,scratch,storage);
}
}

#undef LLAMINAR_PHASE_FFN_FIELD
#undef LLAMINAR_PHASE_POLICY
#undef LLAMINAR_PHASE_MAXIMUM
#undef LLAMINAR_PHASE_MASK
