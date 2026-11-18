/**
 * @file ROCmGroupedFFNGenericProgram.hip.inc
 * @brief Two stream-ordered native grouped projections over original acquired bank roles.
 *
 * Each output tile consumes its original row and route. The gate and up
 * readers can share an activation load while retaining independent FP32 sums.
 * The launch owner supplies exact sealed source and physical format envelopes.
 */
#include "ROCmGroupedFFNGenericNumerics.hip.inc"

namespace {
/** @brief One immutable shared descriptor and its separate source and decoder policies. */
struct SealedSharedRole {
    DeviceNativeQuantisedMatrixDesc descriptor;
    NativeQuantisedExecutionFormatEnvelope formats;
    int partitions;
};

/** @brief Typed ready original local native expert with exact physical publication identity. */
__device__ __forceinline__ const native_runtime_abi::DeviceMoEExpertDescriptorView&
groupedExpert(const AcquiredBank& bank,int expert)
{
    const auto& selected=bank.bank().experts[expert];
    if(selected.weight_format!=DeviceMoEWeightFormat::NativeQuantised ||
        selected.projection_set!=DeviceMoEProjectionSet::IntermediateShard)
        __builtin_trap();
    return selected;
}

/**
 * @brief Share the true row's activation loads across an immutable physical gate/up pair.
 * @tparam Format Original physical decoder sealed by the enclosing capture.
 * @tparam Routed Whether to validate a descriptor from the live acquired bank.
 * @param gate Original prepared gate operand.
 * @param up Original prepared up operand.
 * @param row Actual Q8 activation and optional producer-owned K32 sums.
 * @param parts One original ascending serial split-K schedule, proved equal for this pair at capture.
 * @param span Participant-owned output columns.
 * @param column This lane's output column.
 * @param lane Physical wave lane determining the original partition subset.
 * @param originalSourceMask Sealed permitted source arithmetic IDs for a live routed pair.
 * @return Independent, ordered gate and up projection values.
 */
template<NativeQuantisedFormatPrefill Format,bool Routed>
__device__ __forceinline__ Projections sealedPair(const DeviceNativeQuantisedMatrixDesc& gate,
    const DeviceNativeQuantisedMatrixDesc& up,const generic::ProjectionRow& row,
    int parts,int span,int column,int lane,std::uint32_t originalSourceMask=0)
{
    if constexpr(Routed) {
        if(!prefill_native_quantised_desc_shape_ok<Format>(gate,span,hidden) ||
           !prefill_native_quantised_desc_shape_ok<Format>(up,span,hidden) ||
           !(originalSourceMask & nativeQuantisedCodebookMaskBit(
               canonicalDeviceVnniCodebookId(gate.arithmeticPolicyCodebookId()))) ||
           !(originalSourceMask & nativeQuantisedCodebookMaskBit(
               canonicalDeviceVnniCodebookId(up.arithmeticPolicyCodebookId()))))
            __builtin_trap();
    }
    if constexpr(PrefillNATIVE_QUANTISEDTraits<Format>::is_asymmetric &&
                 !PrefillNATIVE_QUANTISEDTraits<Format>::is_dual_scale) {
        if(row.sum32)
            return pairedProjections<Format,0,true>(
                gate,up,row.codes,row.scales,column,parts,lane,row.sum32);
    }
    return pairedProjections<Format,0>(gate,up,row.codes,row.scales,column,parts,lane);
}

template<unsigned Groups,int RoutedCodebook=-1>
__global__ __launch_bounds__(threads)
void groupedFFNFormatProgramGate(const int32_t* liveOwner,const int8_t* hiddenCodes,
    const float* hiddenScales,const float* hiddenFP32,Scratch scratch,Output output,DirectBankInput direct,
    int routedColumns,int sharedColumns,NativeQuantisedExecutionFormatEnvelope routedFormats,
    int gateParts,int upParts,SealedSharedRole sharedGate,SealedSharedRole sharedUp)
{
    const int tid=threadIdx.x;
    const int live=__builtin_amdgcn_readfirstlane(*liveOwner);
    if(live<0 || live>capacity)__builtin_trap();
    const auto bank=acquireBank<false>(direct);
    if(blockIdx.x==0)publishRoutes(direct,bank,live);
    if(live==0)return;
    const int participant=__builtin_amdgcn_readfirstlane(static_cast<int>(bank.runtime()->participant_id));

    __shared__ float gate_sums[16];
    if(blockIdx.x<live) {
        const float gate=moe_shared_gate::gateValue256(hiddenFP32+blockIdx.x*hidden,
            output.sharedInputGate,hidden,gate_sums);
        if(tid==0)scratch.gate_values[blockIdx.x]=gate;
    }
    constexpr int owners=threads/tileThreads;
    __shared__ float published[owners][32];
    // Ownership and the original route slots are identical across a whole
    // physical wave, even though columns and partition lanes are different.
    const int owner=__builtin_amdgcn_readfirstlane(tid/tileThreads);
    const int lane=tid%64;
    const int tileLane=tid%tileThreads;
    const int routedBlocks=routedColumns/32;
    const int sharedBlocksNow=sharedColumns/32;
    const int routedTiles=live*topK*routedBlocks;
    const int tiles=routedTiles+live*sharedBlocksNow;
    for(int start=blockIdx.x*owners;start<tiles;start+=gridDim.x*owners) {
        const int tile=start+owner;
        bool active=tile<tiles;
        const bool routed=tile<routedTiles;
        const int span=routed?routedColumns:sharedColumns;
        const int blocks=routed?routedBlocks:sharedBlocksNow;
        const int local=routed?tile:tile-routedTiles;
        const int slot=local/blocks;
        const int row=routed?slot/topK:slot;
        const int block=local%blocks;
        const int column=block*32+lane%16+__builtin_amdgcn_readfirstlane(tileLane/64)*16;
        int id=-1;
        if(active && routed) {
            const auto route=selectRoute(direct,bank,slot,live);
            id=__builtin_amdgcn_readfirstlane(route.expert);
            active=id>=0 && __builtin_amdgcn_readfirstlane(route.participant)==participant;
        }
        if(active) {
            const auto* expert=routed?&groupedExpert(bank,id):nullptr;
            const generic::ProjectionUsers<1> users{{{hiddenCodes+row*hidden,hiddenScales+row*(hidden/32),
                direct.affineSums?direct.affineSums->hiddenRows+row*(hidden/32):nullptr}},1};
            if constexpr(RoutedCodebook>=0) {
                const auto values=routed
                    ?sealedPair<static_cast<NativeQuantisedFormatPrefill>(RoutedCodebook),true>(
                        expert->gate,expert->up,users.rows[0],gateParts,span,column,lane,
                        routedFormats.policy_codebook_mask)
                    :sealedPair<NATIVE_QUANTISED_PF_Q8_0,false>(
                        sharedGate.descriptor,sharedUp.descriptor,users.rows[0],sharedGate.partitions,
                        span,column,lane);
                if(lane<16) {
                    if(!routed) {
                        output.sharedGateScratch[row*span+column]=values.gate;
                        output.sharedUpScratch[row*span+column]=values.up;
                    }
                    published[owner][column%32]=decode_equivalent::swigluValue(values.gate,values.up);
                }
            } else {
                float gateValue=0.0f;
#pragma unroll 1
                for(int role=0;role<2;++role) {
                    const bool isGate=role==0;
                    // Select scalar fields from kernargs. Selecting a by-value role
                    // aggregate here forces two full per-thread private shadow copies.
#define LLAMINAR_FRONTIER_ROLE_FIELD(FIELD) \
        (expert?(isGate?expert->gate.FIELD:expert->up.FIELD) \
               :(isGate?sharedGate.descriptor.FIELD:sharedUp.descriptor.FIELD))
                    const DeviceNativeQuantisedMatrixDesc matrix{
                        .payload=LLAMINAR_FRONTIER_ROLE_FIELD(payload),
                        .scales=LLAMINAR_FRONTIER_ROLE_FIELD(scales),
                        .mins=LLAMINAR_FRONTIER_ROLE_FIELD(mins),
                        .emins=LLAMINAR_FRONTIER_ROLE_FIELD(emins),
                        .n=static_cast<decltype(DeviceNativeQuantisedMatrixDesc::n)>(
                            __builtin_amdgcn_readfirstlane(LLAMINAR_FRONTIER_ROLE_FIELD(n))),
                        .k=static_cast<decltype(DeviceNativeQuantisedMatrixDesc::k)>(
                            __builtin_amdgcn_readfirstlane(LLAMINAR_FRONTIER_ROLE_FIELD(k))),
                        .blocks_per_row=static_cast<decltype(DeviceNativeQuantisedMatrixDesc::blocks_per_row)>(
                            __builtin_amdgcn_readfirstlane(LLAMINAR_FRONTIER_ROLE_FIELD(blocks_per_row))),
                        .codebook_id=static_cast<decltype(DeviceNativeQuantisedMatrixDesc::codebook_id)>(
                            __builtin_amdgcn_readfirstlane(LLAMINAR_FRONTIER_ROLE_FIELD(codebook_id))),
                        .source_codebook_id=static_cast<decltype(DeviceNativeQuantisedMatrixDesc::source_codebook_id)>(
                            __builtin_amdgcn_readfirstlane(LLAMINAR_FRONTIER_ROLE_FIELD(source_codebook_id))),
                        .source_identity_present=static_cast<decltype(DeviceNativeQuantisedMatrixDesc::source_identity_present)>(
                            __builtin_amdgcn_readfirstlane(LLAMINAR_FRONTIER_ROLE_FIELD(source_identity_present))),
                    };
#undef LLAMINAR_FRONTIER_ROLE_FIELD
                    const NativeQuantisedExecutionFormatEnvelope formats{
                        expert?routedFormats.execution_codebook_mask:(isGate?sharedGate.formats.execution_codebook_mask:sharedUp.formats.execution_codebook_mask),
                        expert?routedFormats.policy_codebook_mask:(isGate?sharedGate.formats.policy_codebook_mask:sharedUp.formats.policy_codebook_mask),
                    };
                    const int parts=expert?(isGate?gateParts:upParts):
                        (isGate?sharedGate.partitions:sharedUp.partitions);
                    const auto values=generic::orderedProjection<1,4,Groups>(matrix,column,span,hidden,users,parts,lane,formats);
                    if(isGate)gateValue=values.value[0];
                    else if(lane<16) {
                        if(!routed) {
                            output.sharedGateScratch[row*span+column]=gateValue;
                            output.sharedUpScratch[row*span+column]=values.value[0];
                        }
                        published[owner][column%32]=decode_equivalent::swigluValue(gateValue,values.value[0]);
                    }
                }
            }
        }
        __syncthreads();
        const int localColumn=tileLane%32;
        if(active && tileLane<32) {
            const float value=published[owner][localColumn];
            float maximum=fabsf(value);
#pragma unroll
            for(int mask=16;mask>0;mask>>=1)maximum=fmaxf(maximum,__shfl_xor(maximum,mask,32));
            const float scale=decode_equivalent::q8Scale(maximum);
            auto* q=routed?scratch.routed_codes:scratch.shared_codes;
            auto* s=routed?scratch.routed_scales:scratch.shared_scales;
            if(localColumn==0)s[slot*blocks+block]=scale;
            const auto code=static_cast<int8_t>(decode_equivalent::quantizeQ8FromScale(value,scale));
            q[slot*span+block*32+localColumn]=code;
            if(routed && direct.affineSums) {
                int sum=code;
#pragma unroll
                for(int mask=16;mask>0;mask>>=1)sum+=__shfl_xor(sum,mask,32);
                if(localColumn==0)direct.affineSums->routedRows[slot*blocks+block]=static_cast<int16_t>(sum);
            }
        }
        __syncthreads();
    }
}

/** @brief Compute local contributions once per original expert and reduce in original slot order. */
template<unsigned Groups,bool OneBlockPerPartition,int RoutedPhysical=-1>
__global__ __launch_bounds__(threads)
void groupedFFNFormatProgramDown(const int32_t* liveOwner,Scratch scratch,Output output,DirectBankInput direct,
    int routedColumns,int sharedColumns,NativeQuantisedExecutionFormatEnvelope routedFormats,
    int routedParts,SealedSharedRole sharedDown)
{
    constexpr int columns=64;
    const int lane=threadIdx.x%columns;
    const int wave=__builtin_amdgcn_readfirstlane(threadIdx.x/columns);
    const int column=blockIdx.x*columns+lane;
    const int live=__builtin_amdgcn_readfirstlane(*liveOwner);
    if(live<0 || live>capacity)__builtin_trap();
    const auto bank=acquireBank<true>(direct);
    const int row=blockIdx.y;
    __shared__ float complete[topK+1][columns];
    for(int route=wave;route<=topK;route+=threads/columns) {
        float dot=0.0f;
        const bool routed=route<topK;
        const int slot=routed?row*topK+route:row;
        if(row<live && (!routed || __builtin_amdgcn_readfirstlane(bank.runtime()->route_participant_ids[slot])==static_cast<int>(bank.runtime()->participant_id))) {
            const int span=routed?routedColumns:sharedColumns;
            const int blocks=span/32;
            const int id=routed?__builtin_amdgcn_readfirstlane(bank.runtime()->route_expert_ids[slot]):-1;
            if(routed && (id<0 || id>=experts))__builtin_trap();
            const auto matrix=routed?groupedExpert(bank,id).down:sharedDown.descriptor;
            const auto formats=routed?routedFormats:sharedDown.formats;
            const auto parts=routed?routedParts:sharedDown.partitions;
            const auto* codes=routed?scratch.routed_codes:scratch.shared_codes;
            const auto* scales=routed?scratch.routed_scales:scratch.shared_scales;
            const generic::ProjectionUsers<1> user{{{codes+slot*span,scales+slot*blocks,
                routed && direct.affineSums?direct.affineSums->routedRows+slot*blocks:nullptr}},1};
            float value;
            if constexpr(RoutedPhysical>=0) {
                constexpr auto physical=static_cast<NativeQuantisedFormatPrefill>(RoutedPhysical);
                if(matrix.codebook_id!=(routed?RoutedPhysical:int(NATIVE_QUANTISED_PF_Q8_0)) ||
                   !(formats.policy_codebook_mask & nativeQuantisedCodebookMaskBit(canonicalDeviceVnniCodebookId(matrix.arithmeticPolicyCodebookId()))) ||
                   matrix.n!=hidden || parts<1 || parts>blocks || (OneBlockPerPartition && parts!=blocks))__builtin_trap();
                value=routed?generic::orderedProjectionFormat<physical,1,1,OneBlockPerPartition>(matrix,column,span,user,parts,lane).value[0]:
                    generic::orderedProjectionFormat<NATIVE_QUANTISED_PF_Q8_0,1,1,OneBlockPerPartition>(matrix,column,span,user,parts,lane).value[0];
            } else {
                value=generic::orderedProjection<1,1,Groups,OneBlockPerPartition>(matrix,column,hidden,span,user,parts,lane,formats).value[0];
            }
            dot=routed?rocm_native_quantised_weight_route_rn(bank.runtime()->route_weights[slot],value):
                rocm_native_quantised_accumulate_rn(0.0f,rocm_native_quantised_weight_route_rn(1.0f,value));
        }
        complete[route][lane]=dot;
        if(routed)output.routeContributions[slot*hidden+column]=dot;
    }
    __syncthreads();
    if(wave==0) {
        float routed=0.0f;
#pragma unroll
        for(int route=0;route<topK;++route)routed=rocm_native_quantised_accumulate_rn(routed,complete[route][lane]);
        namespace fp=llaminar2::device_fp32_contract;
        const float shared=row<live?fp::multiply(scratch.gate_values[row],complete[topK][lane]):0.0f;
        const auto i=row*hidden+column;
        output.routedOutput[i]=routed;
        output.sharedOutput[i]=shared;
        output.combined[i]=fp::add(routed,shared);
    }
}
} // namespace

/** @brief Resolve one immutable source's original serial-M1 partition count while preparing capture. */
static int preparedPartitions(unsigned source,int columns,int width)
{
    int parts=0;
    if(!rocmGemv_native_quantised_query_serial_m1_config(canonicalDeviceVnniCodebookId(source),columns,width,&parts,nullptr) ||
       parts<1 || parts>width/32 || parts>NativeQuantisedGroupedDecodePolicy::maximum_k_partitions)
        throw std::invalid_argument("Grouped FFN original source lacks a valid serial-M1 partition schedule");
    return parts;
}

SourceFormatProgram SourceFormatProgram::prepare(const SourceFormatDeclaration& declaration)
{
    std::array<int,3> routed{},shared{};
    for(unsigned role=0;role<routed.size();++role) {
        const bool down=role==2;
        routed[role]=preparedPartitions(declaration.routedSources()[role].codebook_id,
            down?hidden:declaration.routedColumns(),down?declaration.routedColumns():hidden);
        // Each immutable shared descriptor supplied a single original arithmetic policy.
        const auto policy=declaration.formats().shared[role].policy_codebook_mask;
        shared[role]=preparedPartitions(std::countr_zero(policy),
            down?hidden:declaration.sharedColumns(),down?declaration.sharedColumns():hidden);
    }
    return {declaration,routed,shared};
}

bool launchFormatProgram(const RoutePlan& plan,const std::array<DeviceNativeQuantisedMatrixDesc,3>& sharedWeights,
    const Output& output,int device,void* stream,const DirectBankInput& direct,const SourceFormatProgram& program)
{
    const int routedColumns=program.routedColumns(), sharedColumns=program.sharedColumns();
    const auto& formats=program.formats();
    if(device<0 || reinterpret_cast<uintptr_t>(stream)<=2 || !direct.runtime || !direct.indices ||
       !direct.weights || !direct.bankIdentity ||
       (direct.demand!=MoEGroupedPlanDemand::None && direct.demand!=MoEGroupedPlanDemand::DeferredAcceptedRows) ||
       !plan.live || !plan.hiddenCodes || !plan.hiddenScales || !plan.hiddenFP32 ||
       !output.routedCodes || !output.routedScales || !output.sharedCodes || !output.sharedScales ||
       !output.sharedInputGate || !output.gateValues || !output.sharedGateScratch || !output.sharedUpScratch ||
       !output.routedOutput || !output.sharedOutput || !output.combined || !output.routeContributions ||
       routedColumns<=0 || routedColumns%32 || routedColumns>INT_MAX/(capacity*topK) ||
       sharedColumns<=0 || sharedColumns%32 || sharedColumns>INT_MAX/capacity ||
       (direct.affineSums && (routedColumns!=160 || sharedColumns!=160)) ||
       !generic::validFormats(formats.routedGateUp) || !generic::validFormats(formats.routedDown))
        return false;
    const Scratch scratch{output.routedCodes,output.routedScales,output.sharedCodes,output.sharedScales,output.gateValues};
    const auto& routedParts=program.routedPartitions();
    std::array<SealedSharedRole,3> roles{};
    for(unsigned role=0;role<roles.size();++role) {
        const bool down=role==2;
        const auto& desc=sharedWeights[role];
        const int n=down?hidden:sharedColumns;
        const int k=down?sharedColumns:hidden;
        NativeQuantisedExecutionFormatEnvelope actual{};
        if(desc.n!=n || desc.k!=k || !NativeQuantisedDescriptorValidation::publishable(desc) ||
           !desc.source_identity_present || !addNativeQuantisedExecutionFormat(actual,desc.codebook_id,
                {desc.source_codebook_id,bool(desc.source_is_superblock),true},false) ||
           actual.execution_codebook_mask!=formats.shared[role].execution_codebook_mask ||
           actual.policy_codebook_mask!=formats.shared[role].policy_codebook_mask)return false;
        roles[role]={desc,actual,program.sharedPartitions()[role]};
    }
    const auto native=static_cast<hipStream_t>(stream);
    constexpr int owners=threads/tileThreads;
    const int maxTiles=capacity*(topK*(routedColumns/32)+sharedColumns/32);
    const int gateGrid=std::min(gateBlocks,std::max(capacity,(maxTiles+owners-1)/owners));
    const auto gateFamilies=ResidentMoEFFNDecoderProgram::fromExecutionMask(
        formats.routedGateUp.execution_codebook_mask | formats.shared[0].execution_codebook_mask |
        formats.shared[1].execution_codebook_mask);
    const auto launchGate=[&]<unsigned Groups,int RoutedCodebook=-1>() {
        hipLaunchKernelGGL((groupedFFNFormatProgramGate<Groups,RoutedCodebook>),dim3(gateGrid),dim3(threads),0,native,
            plan.live,plan.hiddenCodes,plan.hiddenScales,plan.hiddenFP32,scratch,output,direct,
            routedColumns,sharedColumns,formats.routedGateUp,routedParts[0],routedParts[1],roles[0],roles[1]);
    };
    if(program.pairsGateAndUp())
        launchGate.template operator()<1,NATIVE_QUANTISED_PF_Q4_1>();
    else if(gateFamilies.groups()==1) launchGate.template operator()<1>();
    else launchGate.template operator()<15>();
    if(hipGetLastError()!=hipSuccess)return false;
    const auto launchDown=[&]<unsigned Groups,bool OneBlockPerPartition,int RoutedPhysical=-1>() {
        hipLaunchKernelGGL((groupedFFNFormatProgramDown<Groups,OneBlockPerPartition,RoutedPhysical>),dim3(hidden/64,capacity),dim3(threads),0,native,
            plan.live,scratch,output,direct,routedColumns,sharedColumns,formats.routedDown,
            routedParts[2],roles[2]);
    };
    if(formats.shared[2].execution_codebook_mask==nativeQuantisedCodebookMaskBit(NATIVE_QUANTISED_PF_Q8_0)) {
        switch(formats.routedDown.execution_codebook_mask) {
        case nativeQuantisedCodebookMaskBit(NATIVE_QUANTISED_PF_Q4_0): launchDown.template operator()<15,false,NATIVE_QUANTISED_PF_Q4_0>();return hipGetLastError()==hipSuccess;
        case nativeQuantisedCodebookMaskBit(NATIVE_QUANTISED_PF_IQ4_NL): launchDown.template operator()<15,false,NATIVE_QUANTISED_PF_IQ4_NL>();return hipGetLastError()==hipSuccess;
        case nativeQuantisedCodebookMaskBit(NATIVE_QUANTISED_PF_IQ4_XS_SOURCE): launchDown.template operator()<15,false,NATIVE_QUANTISED_PF_IQ4_XS_SOURCE>();return hipGetLastError()==hipSuccess;
        case nativeQuantisedCodebookMaskBit(NATIVE_QUANTISED_PF_Q4_1): launchDown.template operator()<15,false,NATIVE_QUANTISED_PF_Q4_1>();return hipGetLastError()==hipSuccess;
        case nativeQuantisedCodebookMaskBit(NATIVE_QUANTISED_PF_Q5_0): launchDown.template operator()<15,false,NATIVE_QUANTISED_PF_Q5_0>();return hipGetLastError()==hipSuccess;
        case nativeQuantisedCodebookMaskBit(NATIVE_QUANTISED_PF_Q5_1): launchDown.template operator()<15,false,NATIVE_QUANTISED_PF_Q5_1>();return hipGetLastError()==hipSuccess;
        case nativeQuantisedCodebookMaskBit(NATIVE_QUANTISED_PF_Q6_K): launchDown.template operator()<15,false,NATIVE_QUANTISED_PF_Q6_K>();return hipGetLastError()==hipSuccess;
        case nativeQuantisedCodebookMaskBit(NATIVE_QUANTISED_PF_Q3_K): launchDown.template operator()<15,false,NATIVE_QUANTISED_PF_Q3_K>();return hipGetLastError()==hipSuccess;
        case nativeQuantisedCodebookMaskBit(NATIVE_QUANTISED_PF_Q2_K): launchDown.template operator()<15,false,NATIVE_QUANTISED_PF_Q2_K>();return hipGetLastError()==hipSuccess;
        case nativeQuantisedCodebookMaskBit(NATIVE_QUANTISED_PF_IQ3_S): launchDown.template operator()<15,false,NATIVE_QUANTISED_PF_IQ3_S>();return hipGetLastError()==hipSuccess;
        case nativeQuantisedCodebookMaskBit(NATIVE_QUANTISED_PF_IQ3_XXS): launchDown.template operator()<15,false,NATIVE_QUANTISED_PF_IQ3_XXS>();return hipGetLastError()==hipSuccess;
        case nativeQuantisedCodebookMaskBit(NATIVE_QUANTISED_PF_IQ2_S): launchDown.template operator()<15,false,NATIVE_QUANTISED_PF_IQ2_S>();return hipGetLastError()==hipSuccess;
        case nativeQuantisedCodebookMaskBit(NATIVE_QUANTISED_PF_IQ2_XS): launchDown.template operator()<15,false,NATIVE_QUANTISED_PF_IQ2_XS>();return hipGetLastError()==hipSuccess;
        case nativeQuantisedCodebookMaskBit(NATIVE_QUANTISED_PF_IQ2_XXS): launchDown.template operator()<15,false,NATIVE_QUANTISED_PF_IQ2_XXS>();return hipGetLastError()==hipSuccess;
        case nativeQuantisedCodebookMaskBit(NATIVE_QUANTISED_PF_IQ1_S): launchDown.template operator()<15,false,NATIVE_QUANTISED_PF_IQ1_S>();return hipGetLastError()==hipSuccess;
        case nativeQuantisedCodebookMaskBit(NATIVE_QUANTISED_PF_IQ1_M): launchDown.template operator()<15,false,NATIVE_QUANTISED_PF_IQ1_M>();return hipGetLastError()==hipSuccess;
        case nativeQuantisedCodebookMaskBit(NATIVE_QUANTISED_PF_Q8_0): launchDown.template operator()<15,false,NATIVE_QUANTISED_PF_Q8_0>();return hipGetLastError()==hipSuccess;
        case nativeQuantisedCodebookMaskBit(NATIVE_QUANTISED_PF_EXPANDED_INT8_MIN): launchDown.template operator()<15,false,NATIVE_QUANTISED_PF_EXPANDED_INT8_MIN>();return hipGetLastError()==hipSuccess;
        default: break;
        }
    }
    const bool shortDown=roles[2].partitions==sharedColumns/32 && routedParts[2]==routedColumns/32;
    if(shortDown) {
        const auto families=ResidentMoEFFNDecoderProgram::fromExecutionMask(
            formats.routedDown.execution_codebook_mask | formats.shared[2].execution_codebook_mask);
        if(families.groups()==1)launchDown.template operator()<1,true>();
        else launchDown.template operator()<15,true>();
    } else launchDown.template operator()<15,false>();
    return hipGetLastError()==hipSuccess;
}
