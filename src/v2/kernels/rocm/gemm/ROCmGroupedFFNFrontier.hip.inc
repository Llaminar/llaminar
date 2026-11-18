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

} // namespace

/**
 * @brief Borrow original grouped route slots after their stream-ordered publication.
 * @details This view owns no route bytes. Only the same acquired bank and
 * existing epoch/ticket receipt issued by publishRoutes authorize the borrow.
 */
class AcquiredGroupedRoutePublication final {
    AcquiredBank bank_;
    __device__ explicit AcquiredGroupedRoutePublication(const DirectBankInput& input)
        : bank_(acquireBank<true>(input)) {}
public:
    /** @return A retained view of the exact bank that emitted the requested slots. */
    __device__ static AcquiredGroupedRoutePublication acquire(const DirectBankInput& input)
    { return AcquiredGroupedRoutePublication(input); }
    /** @return The ready local Down role from this original live publication. */
    __device__ PublishedNativeProjection localDownIf(bool active,int slot,int live) const
    {
        if(active)return bank_.localNativeDownFromGroupedSlot(slot,live*topK);
        return bank_.localNativeProjectionOrInactive(-1,PublishedNativeProjectionRole::Down);
    }
    /** @return Original FP32 route weight for this same verified local slot. */
    __device__ float weight(int slot) const { return bank_.runtime()->route_weights[slot]; }
};

namespace {

/** @brief Physical affine-Q4 fields already owned by the acquired bank. */
struct RoutedQ4Planes {
    const uint8_t* payload{};
    const void* scales{};
    const void* mins{};
};

#if defined(__gfx906__)
/** @brief Borrow a ready routed role once for the wave-owned output tile.
 * @param role The original ready descriptor in this launch's acquired active bank.
 * @return The original payload, scale and minimum addresses for this role.
 * @pre Every active lane of the wave borrowed the same role for one original
 * expert slot. A lane-varying projection must use its own descriptor loads.
 * @pre The original acquired bank generation and all its role planes stay live
 * and immutable until this launch retires. The original execution owner holds
 * that lifetime across publication and replay.
 */
__device__ __forceinline__ RoutedQ4Planes borrowRoutedQ4Planes(
    const DeviceNativeQuantisedMatrixDesc& role)
{
    static_assert(offsetof(DeviceNativeQuantisedMatrixDesc,payload)==0);
    static_assert(offsetof(DeviceNativeQuantisedMatrixDesc,scales)==8);
    static_assert(offsetof(DeviceNativeQuantisedMatrixDesc,mins)==16);
    const auto uniform=waveUniformAddress(reinterpret_cast<uintptr_t>(&role));
    using Words4=uint32_t __attribute__((ext_vector_type(4)));
    Words4 primary;
    uint64_t minimum;
    // The acquire pins the role. Coherent loads also see a refreshed address
    // when a later launch reuses a retired bank; the wait completes both reads.
    asm volatile("s_load_dwordx4 %0, %2, 0x0 glc\n\t"
                 "s_load_dwordx2 %1, %2, 0x10 glc\n\t"
                 "s_waitcnt lgkmcnt(0)"
                 : "=&s"(primary), "=&s"(minimum) : "s"(uniform) : "memory");
    return {reinterpret_cast<const uint8_t*>(composeWaveAddress(primary[0],primary[1])),
            reinterpret_cast<const void*>(composeWaveAddress(primary[2],primary[3])),
            reinterpret_cast<const void*>(minimum)};
}
#endif

/** @brief Read the packed role's original global bytes without changing its layout. */
template<class T>
__device__ __forceinline__ T readPackedGlobal(const void* pointer)
{
    const auto* source=reinterpret_cast<const unsigned char __attribute__((address_space(1)))*>(
        reinterpret_cast<uintptr_t>(pointer));
    T value;
    __builtin_memcpy(&value,source,sizeof(T));
    return value;
}

/** @brief Append one affine K32 pair with the retained M1 integer and FP32 operations. */
template<bool ReuseSum>
__device__ __forceinline__ void accumulateRoutedQ4Pair(RoutedQ4Planes gate,RoutedQ4Planes up,
    size_t index,const int* activation,float scale,int sum,float& g,float& u)
{
    const auto gateWords=readPackedGlobal<int4>(gate.payload+index*16);
    const auto upWords=readPackedGlobal<int4>(up.payload+index*16);
    const auto dots=q4PairedDots<!ReuseSum>(gateWords,upWords,activation,sum);
    const auto coefficient=[&](const void* plane) {
        return __half2float(__ushort_as_half(readPackedGlobal<uint16_t>(
            static_cast<const uint16_t*>(plane)+index)));
    };
    g=rocm_native_quantised_commit_corrected_single_scale_block(
        g,dots.gate,coefficient(gate.scales),dots.sum,coefficient(gate.mins),scale);
    u=rocm_native_quantised_commit_corrected_single_scale_block(
        u,dots.up,coefficient(up.scales),dots.sum,coefficient(up.mins),scale);
}

/** @brief Original packed operands for one routed K32 block. */
struct RoutedQ4Block {
    int4 gateWords,upWords;
    uint16_t gateScale,gateMin,upScale,upMin;
};

/** @brief Load one original K32 block of each ready role. */
__device__ __forceinline__ RoutedQ4Block loadRoutedQ4Block(
    RoutedQ4Planes gate,RoutedQ4Planes up,size_t index)
{
    return {readPackedGlobal<int4>(gate.payload+index*16),
            readPackedGlobal<int4>(up.payload+index*16),
            readPackedGlobal<uint16_t>(static_cast<const uint16_t*>(gate.scales)+index),
            readPackedGlobal<uint16_t>(static_cast<const uint16_t*>(gate.mins)+index),
            readPackedGlobal<uint16_t>(static_cast<const uint16_t*>(up.scales)+index),
            readPackedGlobal<uint16_t>(static_cast<const uint16_t*>(up.mins)+index)};
}

/** @brief Append one affine K32 pair with the retained M1 integer and FP32 operations. */
template<bool ReuseSum>
__device__ __forceinline__ void commitRoutedQ4Block(const RoutedQ4Block& weight,
    int block,const int8_t* codes,const float* scales,const int16_t* sums,float& g,float& u)
{
    const auto* activation=reinterpret_cast<const int32_t*>(codes+block*32);
    const float scale=scales[block];
    const auto dots=q4PairedDots<!ReuseSum>(
        weight.gateWords,weight.upWords,activation,ReuseSum?sums[block]:0);
    g=rocm_native_quantised_commit_corrected_single_scale_block(
        g,dots.gate,__half2float(__ushort_as_half(weight.gateScale)),dots.sum,
        __half2float(__ushort_as_half(weight.gateMin)),scale);
    u=rocm_native_quantised_commit_corrected_single_scale_block(
        u,dots.up,__half2float(__ushort_as_half(weight.upScale)),dots.sum,
        __half2float(__ushort_as_half(weight.upMin)),scale);
}

/** @brief Joint original gate/up range loads with independent ordered accumulators. */
template<NativeQuantisedFormatPrefill Format,int Columns,bool ReuseSum=false,bool RoutedWaveQ4=false,int K32Packets=1>
__device__ __forceinline__ Projections pairedProjections(const DeviceNativeQuantisedMatrixDesc& gate,const DeviceNativeQuantisedMatrixDesc& up,
    const std::int8_t* codes,const float* scales,int column,int parts,int lane,const std::int16_t* sums=nullptr)
{
    using Traits=PrefillNATIVE_QUANTISEDTraits<Format>;
    const int width=Columns>0?Columns:gate.n;
#if defined(__gfx906__)
    constexpr bool scalarQ4=RoutedWaveQ4 && Format==NATIVE_QUANTISED_PF_Q4_1;
    RoutedQ4Planes gatePlanes{},upPlanes{};
    if constexpr(scalarQ4) {
        // The caller selects one original routed slot for this wave. Lanes
        // differ only in output column and K partition, never in role address.
        gatePlanes=borrowRoutedQ4Planes(gate);
        upPlanes=borrowRoutedQ4Planes(up);
    }
#endif
    const int worker=lane/16;
    const int perPart=(hidden/32+parts-1)/parts;
    float gateTotal=0.0f,upTotal=0.0f;
#pragma unroll 1
    for(int group=0;group<parts;group+=4) {
        const int originalPart=group+worker;
        const int begin=originalPart*perPart;
        const int end=min(begin+perPart,hidden/32);
        float g=0.0f,u=0.0f;
        static_assert(K32Packets==1 || K32Packets==2);
#if defined(__gfx906__)
        if constexpr(scalarQ4 && K32Packets==2) {
            if(originalPart<parts) {
                int block=begin;
                // Every weight load belongs to this original serial partition.
                // Expose the two independent memory groups together while
                // preserving one ascending FP32 commit chain for each role.
#pragma unroll 1
                for(;block+1<end;block+=2) {
                    const auto index=static_cast<size_t>(block)*width+column;
                    const auto first=loadRoutedQ4Block(gatePlanes,upPlanes,index);
                    const auto second=loadRoutedQ4Block(gatePlanes,upPlanes,index+width);
                    commitRoutedQ4Block<ReuseSum>(first,block,codes,scales,sums,g,u);
                    commitRoutedQ4Block<ReuseSum>(second,block+1,codes,scales,sums,g,u);
                }
                if(block<end) {
                    const auto index=static_cast<size_t>(block)*width+column;
                    const auto tail=loadRoutedQ4Block(gatePlanes,upPlanes,index);
                    commitRoutedQ4Block<ReuseSum>(tail,block,codes,scales,sums,g,u);
                }
            }
        } else
#endif
        if(originalPart<parts)
            for(int block=begin;block<end;++block) {
                const auto index=static_cast<std::size_t>(block)*width+column;
                const auto* a4=reinterpret_cast<const int32_t*>(codes+block*32);
                const auto scale=scales[block];
#if defined(__gfx906__)
                if constexpr(scalarQ4) {
                    accumulateRoutedQ4Pair<ReuseSum>(gatePlanes,upPlanes,index,a4,scale,
                        ReuseSum?sums[block]:0,g,u);
                } else
#endif
                {
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
        const auto localRoles=[&] {
            if constexpr(Direct) {
                const int id=active && routed?__builtin_amdgcn_readfirstlane(static_cast<int>(p.direct.indices[original])):-1;
                return bank.localNativeGateUpOrInactive(id);
            } else return 0;
        }();
        if(active && routed) {
            if constexpr(Direct) {
                compact=localRoles.local()?original:-1;
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
                    gateDesc=&localRoles.gate.descriptor();
                    upDesc=&localRoles.up.descriptor();
                } else {
                    gateDesc+=selected;
                    upDesc+=selected;
                }
                const auto& gate=*gateDesc;
                const auto& up=*upDesc;
                if constexpr(!Direct) {
                    if(!prefill_native_quantised_desc_shape_ok<Gate>(gate,RoutedColumns,hidden) ||
                       !prefill_native_quantised_desc_shape_ok<Gate>(up,RoutedColumns,hidden))
                        __builtin_trap();
                    if constexpr(Gate==NATIVE_QUANTISED_PF_Q5_1)
                        if(!gate.source_identity_present || !up.source_identity_present ||
                           gate.source_codebook_id!=7 || up.source_codebook_id!=7 ||
                           !gate.source_is_superblock || !up.source_is_superblock)
                            __builtin_trap();
                }
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

/** @brief Fold original one-block serial partitions of a cold-sealed fixed role.
 * @details The fixed launch proves the K32 shape and one block per partition.
 * Each partial starts at +0 and the existing ordered fold retains its RN operations.
 */
template<NativeQuantisedFormatPrefill Format,int Blocks,bool ReuseSum=false>
__device__ __forceinline__ float fixedPublishedDown(const DeviceNativeQuantisedMatrixDesc& down,
    int column,const std::int8_t* codes,const float* scales,const std::int16_t* sums=nullptr)
{
    using Traits=PrefillNATIVE_QUANTISEDTraits<Format>;
    float total=0.0f;
#pragma unroll 1
    for(int block=0;block<Blocks;++block) {
        const std::size_t index=std::size_t(block)*hidden+column;
        const auto* data=down.payload+index*Traits::payload_bytes;
        int32_t decoded[8];
        moe_prefill_decode_groups<Format,false>(data,decoded,nullptr);
        const auto* activation=reinterpret_cast<const int32_t*>(codes+block*32);
        int sum=0;
        if constexpr(ReuseSum)sum=sums[block];
        const float partial=moe_prefill_accumulate_decoded_block<Format,ReuseSum>(
            down,index,data,decoded,activation,scales[block],0.0f,sum);
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
    const auto published=[&]{
        if constexpr(Direct)return AcquiredGroupedRoutePublication::acquire(p.direct);
        else return 0;
    }();
    __shared__ float complete[topK+1][Columns];
    for(int route=wave;route<topK+1;route+=256/Columns) {
        float value=0.0f;
        if(route<topK) {
            const int original=row*topK+route;
            const auto localDown=[&] {
                if constexpr(Direct)return published.localDownIf(true,original,live);
                else return 0;
            }();
            const int compact=[&] {
                if constexpr(Direct)return static_cast<bool>(localDown)?original:-1;
                else return __builtin_amdgcn_readfirstlane(input.originalToGrouped[original]);
            }();
            if(compact>=0) {
                const DeviceNativeQuantisedMatrixDesc* down;
                float weight;
                if constexpr(Direct) {
                    down=&localDown.descriptor();
                    weight=published.weight(original);
                } else {
                    const int expert=__builtin_amdgcn_readfirstlane(input.originalExpertIds[original]);
                    if(compact>=capacity*topK || expert<0 || expert>=experts)__builtin_trap();
                    down=&input.routedDown[expert];
                    weight=input.groupWeights[compact];
                }
                const auto* codes=scratch.routed_codes+compact*RoutedColumns;
                const auto* scales=scratch.routed_scales+compact*routedBlocks;
                const float dot=[&]{
                    if constexpr(ReuseSum) {
                        static_assert(RoutedColumns==160 && Down==NATIVE_QUANTISED_PF_Q5_1 && Direct);
                        return fixedPublishedDown<Down,routedBlocks,true>(*down,co,codes,scales,
                            p.direct.affineSums->routedRows+compact*routedBlocks);
                    } else if constexpr(Direct) {
                        return fixedPublishedDown<Down,routedBlocks>(*down,co,codes,scales);
                    } else return moe_prefill_dot_desc_serial_m1<Down>(*down,co,codes,scales,hidden,routedBlocks);
                }();
                value=rocm_native_quantised_weight_route_rn(weight,dot);
            }
            if(output.routeContributions)output.routeContributions[original*hidden+co]=value;
        } else {
            const auto* codes=scratch.shared_codes+row*sharedWidth;
            const auto* scales=scratch.shared_scales+row*sharedBlocks;
            const float dot=[&] {
                if constexpr(Direct)
                    return fixedPublishedDown<NATIVE_QUANTISED_PF_Q8_0,sharedBlocks>(p.shared[2],co,codes,scales);
                else
                    return moe_prefill_dot_desc_serial_m1<NATIVE_QUANTISED_PF_Q8_0>(
                        p.shared[2],co,codes,scales,hidden,sharedBlocks);
            }();
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
            RoutedGeometry geometry,RoutedDecoder decoder,const BoundSourceFormatProgram* bound)
{
    if(direct.runtime) {
        const auto* measured=bound?std::get_if<MeasuredProgram>(&bound->program()):nullptr;
        if(!measured || !bound->matches(direct.runtime,device) || measured->geometry!=geometry ||
           measured->decoder!=decoder || bool(direct.affineSums)!=(measured->sums==AffineSumSource::RouterPublication) ||
           (direct.demand!=MoEGroupedPlanDemand::None && direct.demand!=MoEGroupedPlanDemand::DeferredAcceptedRows) ||
           !plan.hiddenCodes || !plan.hiddenScales || !plan.hiddenFP32 ||
           !output.routedCodes || !output.routedScales || !output.sharedInputGate)
            return false;
        for(unsigned index=0;index<sharedWeights.size();++index) {
            const auto& role=sharedWeights[index];
            const bool down=index==2;
            if(role.n!=(down?hidden:sharedWidth) || role.k!=(down?sharedWidth:hidden) ||
               !NativeQuantisedDescriptorValidation::publishable(role) || role.codebook_id!=19 ||
               !role.source_identity_present || role.source_codebook_id!=19 || role.source_is_superblock)
                return false;
        }
    } else if(bound) return false;
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
