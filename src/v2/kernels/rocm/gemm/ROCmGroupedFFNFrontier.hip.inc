/**
 * @file ROCmGroupedFFNFrontier.hip.inc
 * @brief Counted, source-ordered MI50 M4 grouped gate/up publication and local FFN output.
 *
 * Included by the original grouped projection owner to share exact K32 leaf
 * arithmetic. The original upstream plan publishes mutable descriptor slots
 * and canonical route-to-group mappings. Two native nodes retain an explicit
 * stream edge, allowing each phase to keep its own register and grid geometry.
 */
#include "ROCmGroupedFFNFrontier.h"
#include "ROCmResidentMoEFFN.h"
#include "kernels/rocm/moe/ROCmSharedExpertGateAddBody.cuh"
#include "kernels/rocm/moe/ROCmOriginalRouterPhase.hip.inc"
#include "kernels/rocm/moe/ROCmNativeRuntimeBankView.h"

namespace llaminar2::rocm::grouped_ffn {
namespace {
constexpr int threads=256,tileThreads=128,gateBlocks=120;
constexpr int sharedBlocks=sharedWidth/32;

struct Scratch {
    std::int8_t* routed_codes;
    float* routed_scales;
    std::int8_t* shared_codes;
    float* shared_scales;
    float* gate_values;
};

struct Args {
    RoutePlan input;
    std::array<DeviceNativeQuantisedMatrixDesc,3> shared;
    Scratch scratch;
    Output output;
    DirectBankInput direct;
};

struct Projections {float gate,up;};

using AcquiredBank=router_native::AcquiredRuntimeDecodeBank<native_runtime_abi::DeviceMoELayerRuntimeView>;

struct Route {int expert; int participant; float weight;};

/** @brief Resolve one original route coordinate without moving activation or descriptor bytes. */
__device__ __forceinline__ Route selectRoute(const DirectBankInput& input,const AcquiredBank& bank,int slot,int live)
{
    if(slot>=live*topK)
        return {-1,-1,0.0f};
    const int expert=static_cast<int>(input.indices[slot]);
    if(expert<0 || expert>=experts)
        return {-1,-1,0.0f};
    const auto membership=bank.intermediateShardMembership(expert);
    const float weight=membership.disposition==MoEIntermediateShardRowDisposition::Local ? input.weights[slot] : 0.0f;
    return {expert,membership.participant,weight};
}

/** @brief Keep original route metadata and the optional deferred accepted-prefix input visible. */
__device__ __forceinline__ void publishRoutes(const DirectBankInput& input,const AcquiredBank& bank,int live)
{
    auto* runtime=bank.runtime();
    if(runtime->prefill_route_capacity<capacity*topK || !runtime->route_expert_ids ||
       !runtime->route_participant_ids || !runtime->route_weights ||
       (input.demand==MoEGroupedPlanDemand::DeferredAcceptedRows &&
        (runtime->deferred_verifier_route_capacity<capacity*topK ||
         !runtime->deferred_verifier_route_expert_ids || !runtime->deferred_verifier_route_participant_ids)))
        __builtin_trap();
    const int slot=threadIdx.x;
    if(slot<capacity*topK) {
        const auto route=selectRoute(input,bank,slot,live);
        runtime->route_expert_ids[slot]=route.expert;
        runtime->route_participant_ids[slot]=route.participant;
        runtime->route_weights[slot]=route.weight;
        if(input.demand==MoEGroupedPlanDemand::DeferredAcceptedRows) {
            runtime->deferred_verifier_route_expert_ids[slot]=route.expert;
            runtime->deferred_verifier_route_participant_ids[slot]=route.participant;
        }
    }
    if(slot==0) {
        const auto* ticket=runtime->overlay_epoch_ticket;
        input.bankIdentity[0]=reinterpret_cast<std::uintptr_t>(&bank.bank());
        input.bankIdentity[1]=bank.bank().epoch;
        input.bankIdentity[2]=ticket?ticket->selector:0;
    }
}

/** @brief Acquire the original ready bank on each node and verify generation across their stream edge. */
template<bool Verify>
__device__ __forceinline__ AcquiredBank acquireBank(const DirectBankInput& input)
{
    const auto bank=AcquiredBank::acquire(reinterpret_cast<AcquiredBank::Runtime*>(input.runtime));
    if(bank.bank().expert_count!=experts || !bank.runtime()->intermediate_shard.enabled())
        __builtin_trap();
    if constexpr(Verify) {
        const auto* ticket=bank.runtime()->overlay_epoch_ticket;
        if(input.bankIdentity[0]!=reinterpret_cast<std::uintptr_t>(&bank.bank()) ||
           input.bankIdentity[1]!=bank.bank().epoch || input.bankIdentity[2]!=(ticket?ticket->selector:0))
            __builtin_trap();
    }
    return bank;
}

/** @brief Joint original gate/up range loads with independent ordered accumulators. */
template<NativeQuantisedFormatPrefill Format,int Columns,bool ReuseSum=false>
__device__ __forceinline__ Projections pairedProjections(const DeviceNativeQuantisedMatrixDesc& gate,const DeviceNativeQuantisedMatrixDesc& up,
    const std::int8_t* codes,const float* scales,int column,int parts,int lane,const std::int16_t* sums=nullptr)
{
    using Traits=PrefillNATIVE_QUANTISEDTraits<Format>;
    const int width=Columns>0?Columns:gate.n;
    const int worker=lane/16;
    const int perPart=(hidden/32+parts-1)/parts;
    float gateTotal=0.0f,upTotal=0.0f;
#pragma unroll 1
    for(int group=0;group<parts;group+=4) {
        const int originalPart=group+worker;
        const int begin=originalPart*perPart;
        const int end=min(begin+perPart,hidden/32);
        float g=0.0f,u=0.0f;
        if(originalPart<parts)
            for(int block=begin;block<end;++block) {
                const auto index=static_cast<std::size_t>(block)*width+column;
                const auto* a4=reinterpret_cast<const int32_t*>(codes+block*32);
                const auto scale=scales[block];
                const auto* gatePayload=gate.payload+index*Traits::payload_bytes;
                const auto* upPayload=up.payload+index*Traits::payload_bytes;
                int sum=0;
                if constexpr(ReuseSum)sum=sums[block];
                int32_t decoded[8];
                moe_prefill_decode_groups<Format,false>(gatePayload,decoded,nullptr);
                g=moe_prefill_accumulate_decoded_block<Format,ReuseSum>(gate,index,gatePayload,decoded,a4,scale,g,sum);
                moe_prefill_decode_groups<Format,false>(upPayload,decoded,nullptr);
                u=moe_prefill_accumulate_decoded_block<Format,ReuseSum>(up,index,upPayload,decoded,a4,scale,u,sum);
            }
        if(parts==1)return {g,u};
#pragma unroll
        for(int part=0;part<4;++part) {
            const int partner=lane%16+part*16;
            const float gPart=__shfl(g,partner,64),uPart=__shfl(u,partner,64);
            if(group+part<parts) {
                gateTotal=rocm_native_quantised_accumulate_rn(gateTotal,gPart);
                upTotal=rocm_native_quantised_accumulate_rn(upTotal,uPart);
            }
        }
    }
    return {gateTotal,upTotal};
}

/** @brief Publish both original activation banks and the counted shared sigmoid plane. */
template<bool Direct,int RoutedColumns,int GateBlocks,NativeQuantisedFormatPrefill Gate=NATIVE_QUANTISED_PF_Q4_1,
    bool ReuseSum=false,bool PublishRoutedSum=false>
__global__ __launch_bounds__(threads)
void groupedFFNPairedGatePublish(Args p)
{
    constexpr int routedBlocks=RoutedColumns/32;
    const auto input=p.input;
    const auto planes=p.scratch;
    const int live=__builtin_amdgcn_readfirstlane(*input.live);
    if(live<0 || live>capacity)__builtin_trap();
    const auto bank=[&]{
        if constexpr(Direct)return acquireBank<false>(p.direct);
        else return 0;
    }();
    if constexpr(Direct)
        if(blockIdx.x==0)publishRoutes(p.direct,bank,live);
    __shared__ float gate_sums[16];
    __shared__ float publication[threads/tileThreads][32];

    const int tid=threadIdx.x;
    if(blockIdx.x<live) {
        const float gate=llaminar2::rocm::moe_shared_gate::gateValue256(
            input.hiddenFP32+blockIdx.x*hidden,p.output.sharedInputGate,hidden,gate_sums);
        if(tid==0)planes.gate_values[blockIdx.x]=gate;
    }
    const int owner=tid/tileThreads;
    const int lane=tid%64,tileLane=tid%tileThreads;
    const int routedTiles=live*topK*routedBlocks;
    const int tiles=routedTiles+live*sharedBlocks;
    for(int start=blockIdx.x*(threads/tileThreads);start<tiles;start+=GateBlocks*(threads/tileThreads)) {
        const int tile=start+owner;
        bool active=tile<tiles;
        const bool routed=tile<routedTiles;
        const int local=routed?tile:tile-routedTiles;
        const int row=routed?local/(routedBlocks*topK):local/sharedBlocks;
        const int original=routed?local/routedBlocks:row;
        int compact=row;
        int expert=0;
        if(active && routed) {
            if constexpr(Direct) {
                const auto selection=selectRoute(p.direct,bank,original,live);
                expert=selection.expert;
                compact=selection.participant==static_cast<int>(bank.runtime()->participant_id)?original:-1;
            } else {
                compact=input.originalToGrouped[original];
                expert=input.originalExpertIds[original];
            }
        }
        if(active && routed && compact<0)active=false;
        if(active && (expert<0 || expert>=experts || compact>=capacity*(routed?topK:1)))
            __builtin_trap();
        const int span=routed?RoutedColumns:sharedWidth;
        const int blocks=span/32;
        const int block=local%blocks;
        const int column=block*32+lane%16+(tileLane/64)*16;
        if(active) {
            const auto* codes=input.hiddenCodes+row*hidden;
            const auto* scales=input.hiddenScales+row*(hidden/32);
            Projections value;
            const int selected=__builtin_amdgcn_readfirstlane(expert);
            if(routed) {
                const auto* gateDesc=input.routedGate;
                const auto* upDesc=input.routedUp;
                if constexpr(Direct) {
                    const auto& row=bank.bank().experts[selected];
                    if(row.weight_format!=DeviceMoEWeightFormat::NativeQuantised ||
                       row.projection_set!=DeviceMoEProjectionSet::IntermediateShard)
                        __builtin_trap();
                    gateDesc=&row.gate;
                    upDesc=&row.up;
                } else {
                    gateDesc+=selected;
                    upDesc+=selected;
                }
                const auto& gate=*gateDesc;
                const auto& up=*upDesc;
                if(!prefill_native_quantised_desc_shape_ok<Gate>(gate,RoutedColumns,hidden) ||
                   !prefill_native_quantised_desc_shape_ok<Gate>(up,RoutedColumns,hidden))
                    __builtin_trap();
                if constexpr(Gate==NATIVE_QUANTISED_PF_Q5_1)
                    if(!gate.source_identity_present || !up.source_identity_present ||
                       gate.source_codebook_id!=7 || up.source_codebook_id!=7 ||
                       !gate.source_is_superblock || !up.source_is_superblock)
                        __builtin_trap();
                const std::int16_t* sums=nullptr;
                if constexpr(ReuseSum)sums=p.direct.affineSums->hiddenRows+row*(hidden/32);
                value=pairedProjections<Gate,RoutedColumns,ReuseSum>(gate,up,codes,scales,column,16,lane,sums);
            } else {
                value=pairedProjections<NATIVE_QUANTISED_PF_Q8_0,sharedWidth>(p.shared[0],p.shared[1],
                    codes,scales,column,16,lane);
            }
            if(lane<16) {
                if(!routed) {
                    p.output.sharedGateScratch[row*sharedWidth+column]=value.gate;
                    p.output.sharedUpScratch[row*sharedWidth+column]=value.up;
                }
                publication[owner][column%32]=decode_equivalent::swigluValue(value.gate,value.up);
            }
        }
        __syncthreads();
        if(active && tileLane<32) {
            using namespace llaminar2::rocm::decode_equivalent;
            const float value=publication[owner][tileLane];
            float magnitude=fabsf(value);
#pragma unroll
            for(int mask=16;mask>0;mask>>=1)magnitude=fmaxf(magnitude,__shfl_xor(magnitude,mask,32));
            const float scale=q8Scale(magnitude);
            auto* q=routed?planes.routed_codes:planes.shared_codes;
            auto* s=routed?planes.routed_scales:planes.shared_scales;
            // The grouped route slot is the original authority for scratch identity.
            if(tileLane==0)s[compact*blocks+block]=scale;
            const auto code=static_cast<std::int8_t>(quantizeQ8FromScale(value,scale));
            q[compact*span+block*32+tileLane]=code;
            if constexpr(PublishRoutedSum) {
                if(routed) {
                    int sum=code;
#pragma unroll
                    for(int mask=16;mask>0;mask>>=1)sum+=__shfl_xor(sum,mask,32);
                    if(tileLane==0)p.direct.affineSums->routedRows[compact*blocks+block]=static_cast<std::int16_t>(sum);
                }
            }
        }
        __syncthreads();
    }

}

/** @brief Original five one-block serial partitions with the producer's exact signed integer sum. */
template<NativeQuantisedFormatPrefill Format>
__device__ __forceinline__ float exactK32AffineDown(const DeviceNativeQuantisedMatrixDesc& down,
    int column,const std::int8_t* codes,const float* scales,const std::int16_t* sums)
{
    using Traits=PrefillNATIVE_QUANTISEDTraits<Format>;
    float total=0.0f;
#pragma unroll 1
    for(int block=0;block<5;++block) {
        const std::size_t index=std::size_t(block)*hidden+column;
        const auto* data=down.payload+index*Traits::payload_bytes;
        int32_t decoded[8];
        moe_prefill_decode_groups<Format,false>(data,decoded,nullptr);
        const auto* activation=reinterpret_cast<const int32_t*>(codes+block*32);
        const float partial=moe_prefill_accumulate_decoded_block<Format,true>(
            down,index,data,decoded,activation,scales[block],0.0f,sums[block]);
        total=rocm_native_quantised_accumulate_rn(total,partial);
    }
    return total;
}

/** @brief Column-major token-local down, with full wave-owned useful output lanes. */
template<bool Direct,int RoutedColumns,NativeQuantisedFormatPrefill Down=NATIVE_QUANTISED_PF_Q5_1,bool ReuseSum=false>
__global__ __launch_bounds__(256,2)
void groupedFFNTokenDown(Args p)
{
    constexpr int routedBlocks=RoutedColumns/32;
    constexpr int Columns=64;
    const auto input=p.input;
    const auto scratch=p.scratch;
    const auto output=p.output;
    const int live=__builtin_amdgcn_readfirstlane(*input.live);
    const int row=blockIdx.y;
    const int x=threadIdx.x%Columns;
    const int co=blockIdx.x*Columns+x;
    const int wave=threadIdx.x/Columns;
    if(live<0 || live>capacity)__builtin_trap();
    if(row>=live) {
        if(wave==0) {
            const auto i=row*hidden+co;
            output.routedOutput[i]=0.0f;
            output.sharedOutput[i]=0.0f;
            output.combined[i]=0.0f;
            if(output.routeContributions)
                for(int route=0;route<topK;++route)
                    output.routeContributions[(row*topK+route)*hidden+co]=0.0f;
        }
        return;
    }
    const auto bank=[&]{
        if constexpr(Direct)return acquireBank<true>(p.direct);
        else return 0;
    }();
    __shared__ float complete[topK+1][Columns];
    for(int route=wave;route<topK+1;route+=256/Columns) {
        float value=0.0f;
        if(route<topK) {
            const int original=row*topK+route;
            int compact,expert;
            if constexpr(Direct) {
                const auto* runtime=bank.runtime();
                expert=__builtin_amdgcn_readfirstlane(runtime->route_expert_ids[original]);
                const int participant=__builtin_amdgcn_readfirstlane(runtime->route_participant_ids[original]);
                compact=participant==static_cast<int>(runtime->participant_id)?original:-1;
            } else {
                compact=__builtin_amdgcn_readfirstlane(input.originalToGrouped[original]);
                expert=__builtin_amdgcn_readfirstlane(input.originalExpertIds[original]);
            }
            if(compact>=0) {
                if(compact>=capacity*topK || expert<0 || expert>=experts)__builtin_trap();
                const DeviceNativeQuantisedMatrixDesc* down;
                float weight;
                if constexpr(Direct) {
                    down=&bank.bank().experts[expert].down;
                    weight=bank.runtime()->route_weights[original];
                    if(!prefill_native_quantised_desc_shape_ok<Down>(*down,hidden,RoutedColumns))
                        __builtin_trap();
                    if constexpr(Down==NATIVE_QUANTISED_PF_Q8_0)
                        if(!down->source_identity_present || down->source_codebook_id!=19)
                            __builtin_trap();
                } else {
                    down=&input.routedDown[expert];
                    weight=input.groupWeights[compact];
                }
                const auto* codes=scratch.routed_codes+compact*RoutedColumns;
                const auto* scales=scratch.routed_scales+compact*routedBlocks;
                const float dot=[&]{
                    if constexpr(ReuseSum) {
                        static_assert(RoutedColumns==160 && Down==NATIVE_QUANTISED_PF_Q5_1 && Direct);
                        return exactK32AffineDown<Down>(*down,co,codes,scales,
                            p.direct.affineSums->routedRows+compact*routedBlocks);
                    } else return moe_prefill_dot_desc_serial_m1<Down>(*down,co,codes,scales,hidden,routedBlocks);
                }();
                value=rocm_native_quantised_weight_route_rn(weight,dot);
            }
            if(output.routeContributions)output.routeContributions[original*hidden+co]=value;
        } else {
            const float dot=moe_prefill_dot_desc_serial_m1<NATIVE_QUANTISED_PF_Q8_0>(
                p.shared[2],co,scratch.shared_codes+row*sharedWidth,
                scratch.shared_scales+row*sharedBlocks,hidden,5);
            value=rocm_native_quantised_accumulate_rn(0.0f,rocm_native_quantised_weight_route_rn(1.0f,dot));
        }
        complete[route][x]=value;
    }
    __syncthreads();
    if(wave==0) {
        float routed=0.0f;
#pragma unroll
        for(int route=0;route<topK;++route)routed=rocm_native_quantised_accumulate_rn(routed,complete[route][x]);
        namespace fp=llaminar2::device_fp32_contract;
        const float shared=fp::multiply(scratch.gate_values[row],complete[topK][x]);
        const auto i=row*hidden+co;
        output.routedOutput[i]=routed;
        output.sharedOutput[i]=shared;
        output.combined[i]=fp::add(routed,shared);
    }
}

} // namespace

bool launch(const RoutePlan& plan,const std::array<DeviceNativeQuantisedMatrixDesc,3>& sharedWeights,
            const Output& output,int device,void* stream,const DirectBankInput& direct,
            RoutedGeometry geometry,RoutedDecoder decoder)
{
    const int routed=routedColumns(geometry);
    if(geometry!=RoutedGeometry::F128 && (geometry!=RoutedGeometry::F160 || !direct.runtime))
        return false;
    if(decoder!=RoutedDecoder::Q4K_Q51 && geometry!=RoutedGeometry::F160)
        return false;
    if(device<0 || reinterpret_cast<std::uintptr_t>(stream)<=2 || !plan.live ||
       !output.sharedCodes || !output.sharedScales || !output.gateValues || !output.sharedGateScratch || !output.sharedUpScratch ||
       !output.routedOutput || !output.sharedOutput || !output.combined || !output.routeContributions)
        return false;
    int routedGateParts=0,sharedGateParts=0,routedDownParts=0,sharedDownParts=0;
    if(!rocmGemv_native_quantised_query_serial_m1_config(routedGateCodebook(decoder),routed,hidden,&routedGateParts,nullptr) ||
       !rocmGemv_native_quantised_query_serial_m1_config(19,sharedWidth,hidden,&sharedGateParts,nullptr) ||
       !rocmGemv_native_quantised_query_serial_m1_config(routedDownCodebook(decoder),hidden,routed,&routedDownParts,nullptr) ||
       !rocmGemv_native_quantised_query_serial_m1_config(19,hidden,sharedWidth,&sharedDownParts,nullptr) ||
       routedGateParts!=16 || sharedGateParts!=16 || routedDownParts!=routed/32 || sharedDownParts!=5)
        return false;
    const Args args{plan,sharedWeights,{output.routedCodes,output.routedScales,output.sharedCodes,
        output.sharedScales,output.gateValues},output,direct};
    const auto native=static_cast<hipStream_t>(stream);
    if(direct.runtime) {
        if(!direct.indices || !direct.weights || !direct.bankIdentity)return false;
        if(geometry==RoutedGeometry::F160) {
            constexpr int wide=routedColumns(RoutedGeometry::F160), wideGateBlocks=120;
            if(direct.affineSums) {
                switch(decoder) {
                case RoutedDecoder::Q4K_Q51:
                    hipLaunchKernelGGL((groupedFFNPairedGatePublish<true,wide,wideGateBlocks,NATIVE_QUANTISED_PF_Q4_1,true,true>),
                        dim3(wideGateBlocks),dim3(threads),0,native,args);
                    break;
                case RoutedDecoder::Q4K_Q80:
                    return false;
                case RoutedDecoder::Q5K_Q80:
                    hipLaunchKernelGGL((groupedFFNPairedGatePublish<true,wide,wideGateBlocks,NATIVE_QUANTISED_PF_Q5_1,true,false>),
                        dim3(wideGateBlocks),dim3(threads),0,native,args);
                    break;
                }
                if(hipGetLastError()!=hipSuccess)return false;
                if(decoder==RoutedDecoder::Q4K_Q51)
                    hipLaunchKernelGGL((groupedFFNTokenDown<true,wide,NATIVE_QUANTISED_PF_Q5_1,true>),
                        dim3(hidden/64,capacity),dim3(threads),0,native,args);
                else
                    hipLaunchKernelGGL((groupedFFNTokenDown<true,wide,NATIVE_QUANTISED_PF_Q8_0>),
                        dim3(hidden/64,capacity),dim3(threads),0,native,args);
                return hipGetLastError()==hipSuccess;
            }
            if(decoder==RoutedDecoder::Q5K_Q80)
                hipLaunchKernelGGL((groupedFFNPairedGatePublish<true,wide,wideGateBlocks,NATIVE_QUANTISED_PF_Q5_1>),
                    dim3(wideGateBlocks),dim3(threads),0,native,args);
            else
                hipLaunchKernelGGL((groupedFFNPairedGatePublish<true,wide,wideGateBlocks>),
                    dim3(wideGateBlocks),dim3(threads),0,native,args);
            if(hipGetLastError()!=hipSuccess)return false;
            if(decoder==RoutedDecoder::Q4K_Q51)
                hipLaunchKernelGGL((groupedFFNTokenDown<true,wide>),dim3(hidden/64,capacity),dim3(threads),0,native,args);
            else
                hipLaunchKernelGGL((groupedFFNTokenDown<true,wide,NATIVE_QUANTISED_PF_Q8_0>),
                    dim3(hidden/64,capacity),dim3(threads),0,native,args);
        } else {
            hipLaunchKernelGGL((groupedFFNPairedGatePublish<true,routedWidth,gateBlocks>),dim3(gateBlocks),dim3(threads),0,native,args);
            if(hipGetLastError()!=hipSuccess)return false;
            hipLaunchKernelGGL((groupedFFNTokenDown<true,routedWidth>),dim3(hidden/64,capacity),dim3(threads),0,native,args);
        }
    } else {
        hipLaunchKernelGGL((groupedFFNPairedGatePublish<false,routedWidth,gateBlocks>),dim3(gateBlocks),dim3(threads),0,native,args);
        if(hipGetLastError()!=hipSuccess)return false;
        hipLaunchKernelGGL((groupedFFNTokenDown<false,routedWidth>),dim3(hidden/64,capacity),dim3(threads),0,native,args);
    }
    return hipGetLastError()==hipSuccess;
}

#include "ROCmGroupedFFNGenericProgram.hip.inc"

} // namespace llaminar2::rocm::grouped_ffn
