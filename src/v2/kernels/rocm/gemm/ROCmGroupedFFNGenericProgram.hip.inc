/**
 * @file ROCmGroupedFFNGenericProgram.hip.inc
 * @brief Two stream-ordered native grouped projections over original acquired bank roles.
 *
 * Each output tile consumes its original row and route. The gate and up
 * readers can share an activation load while retaining independent FP32 sums.
 * The launch owner supplies exact sealed source and physical format envelopes.
 */
#include "ROCmGroupedFFNGenericNumerics.hip.inc"

/**
 * @brief Shared descriptor copy certified once by the original capture preparation.
 * @details Only launchFormatProgram can seed this immutable value. Kernel
 * readers get the same published-native view as the acquired original bank.
 */
class PreparedGroupedSharedRole final {
    friend bool launchFormatProgram(const RoutePlan&,const DeviceRowRange&,const std::array<DeviceNativeQuantisedMatrixDesc,3>&,
        const Output&,int,void*,const DirectBankInput&,const BoundSourceFormatProgram&);
    DeviceNativeQuantisedMatrixDesc descriptor_{};
    NativeQuantisedExecutionFormatEnvelope formats_{};
    int partitions_{};
    PreparedGroupedSharedRole() = default;
    PreparedGroupedSharedRole(const DeviceNativeQuantisedMatrixDesc& descriptor,
        NativeQuantisedExecutionFormatEnvelope formats,int partitions)
        : descriptor_(descriptor),formats_(formats),partitions_(partitions) {}
public:
    /** @return Immutable device descriptor certified at preparation. */
    __device__ __forceinline__ PublishedNativeProjection projection() const
    { return PublishedNativeProjection(&descriptor_); }
    /** @return One register value from these capture-validated shared fields. */
    __device__ __forceinline__ generic::ReadyProjectionFields fields() const
    { return generic::ReadyProjectionFields(descriptor_); }
    /** @return The original role's retained numerical partition count. */
    __host__ __device__ int partitions() const noexcept { return partitions_; }
    /** @return Its retained source/physical envelope. */
    __host__ __device__ NativeQuantisedExecutionFormatEnvelope formats() const noexcept { return formats_; }
};

namespace {
using SealedSharedRole=PreparedGroupedSharedRole;

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
template<NativeQuantisedFormatPrefill Format,bool Routed,int K32Packets=1>
__device__ __forceinline__ Projections sealedPair(PublishedNativeProjection gateRole,
    PublishedNativeProjection upRole,const generic::ProjectionRow& row,int parts,int column,int lane)
{
    const auto& gate=gateRole.descriptor();
    const auto& up=upRole.descriptor();
    // A reachable mutable expert may use a different native physical reader.
    // Shape, source and plane validity belong to the published-role authority.
    if constexpr(Routed)
        if(gate.codebook_id!=Format || up.codebook_id!=Format)__builtin_trap();
    if constexpr(PrefillNATIVE_QUANTISEDTraits<Format>::is_asymmetric &&
                 !PrefillNATIVE_QUANTISEDTraits<Format>::is_dual_scale) {
        if(row.sum32)
            return pairedProjections<Format,0,true,Routed,K32Packets>(
                gate,up,row.codes,row.scales,column,parts,lane,row.sum32);
    }
    return pairedProjections<Format,0,false,Routed,K32Packets>(gate,up,row.codes,row.scales,column,parts,lane);
}

template<unsigned Groups,int RoutedCodebook,int FixedQ5Pitch,int K32Packets,bool DedicatedTiles>
__global__ __launch_bounds__(threads)
void groupedFFNFormatProgramGate(const int32_t* liveOwner,const int8_t* hiddenCodes,
    const float* hiddenScales,const float* hiddenFP32,Scratch scratch,Output output,DirectBankInput direct,
    int physicalRows,int routedColumns,int sharedColumns,NativeQuantisedExecutionFormatEnvelope routedFormats,
    int gateParts,int upParts,SealedSharedRole sharedGate,SealedSharedRole sharedUp)
{
    const int tid=threadIdx.x;
    const int live=__builtin_amdgcn_readfirstlane(*liveOwner);
    if(live<0 || live>physicalRows)__builtin_trap();
    const auto bank=acquireBank<false>(direct);
    if(blockIdx.x==0)publishRoutes(direct,bank,live,physicalRows);
    if(live==0)return;

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
        // The original router packet already owns this slot's expert choice.
        // Resolve locality/readiness once and retain both roles from that row.
        const int id=active && routed?__builtin_amdgcn_readfirstlane(static_cast<int>(direct.indices[slot])):-1;
        const auto picked=bank.localNativeGateUpOrInactive(id);
        if(routed)active=picked.local();
        if(active) {
            const generic::ProjectionUsers<1> users{{{hiddenCodes+row*hidden,hiddenScales+row*(hidden/32),
                direct.affineSums?direct.affineSums->hiddenRows+row*(hidden/32):nullptr}},1};
            constexpr bool batchIQ3XXS=RoutedCodebook==NATIVE_QUANTISED_PF_IQ3_XXS;
            if constexpr(RoutedCodebook>=0 && !batchIQ3XXS) {
                const auto values=routed
                    ?sealedPair<static_cast<NativeQuantisedFormatPrefill>(RoutedCodebook),true,K32Packets>(
                        picked.gate,picked.up,users.rows[0],gateParts,column,lane)
                    :sealedPair<NATIVE_QUANTISED_PF_Q8_0,false>(
                        sharedGate.projection(),sharedUp.projection(),users.rows[0],sharedGate.partitions(),column,lane);
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
                    const auto projection=routed
                        ? generic::ReadyProjectionFields::from(isGate?picked.gate:picked.up)
                        : (isGate?sharedGate.fields():sharedUp.fields());
                    const auto formats=routed?routedFormats:(isGate?sharedGate.formats():sharedUp.formats());
                    const int parts=routed?(isGate?gateParts:upParts):
                        (isGate?sharedGate.partitions():sharedUp.partitions());
                    const auto values=[&] {
                        if constexpr(batchIQ3XXS) {
                            // The bound original gate/up envelopes prove these physical
                            // readers for every bank reachable by this capture.
                            return routed
                                ?generic::orderedProjectionFormat<NATIVE_QUANTISED_PF_IQ3_XXS,1,4,false,0,true>(
                                    projection,column,hidden,users,parts,lane)
                                :generic::orderedProjectionFormat<NATIVE_QUANTISED_PF_IQ2_S,1,4,false>(
                                    projection,column,hidden,users,parts,lane);
                        }
                        else if constexpr(FixedQ5Pitch>0)
                            return routed
                                ?generic::orderedProjectionFormat<NATIVE_QUANTISED_PF_Q5_1,1,4,false,FixedQ5Pitch>(
                                    projection,column,hidden,users,parts,lane)
                                :generic::orderedProjectionFormat<NATIVE_QUANTISED_PF_Q8_0,1,4,false,160>(
                                    projection,column,hidden,users,parts,lane);
                        else
                            return generic::orderedProjection<1,4,Groups>(projection,column,hidden,users,parts,lane,formats);
                    }();
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
        // A dedicated tile publishes once. Only a workgroup that reuses this
        // LDS slot for another tile must wait for the previous readers.
        if constexpr(DedicatedTiles) break;
        else __syncthreads();
    }
}

/**
 * @brief Issue independent Q5_1 packet reads ahead of the ordered serial fold.
 * @tparam Blocks Number of original single-packet K partitions in this routed shard.
 * @tparam PacketCount Outstanding packet group chosen for this routed width.
 * @param role Current bank's prepared down projection for this live local expert.
 * @param column Original hidden output column owned by this lane.
 * @param codes Original routed SwiGLU row in Q8, indexed in K32 order.
 * @param scales One original activation scale per K32 packet.
 * @return Both dot and affine correction rounded in their serial M1 order.
 */
template<int Blocks,int PacketCount>
__device__ __forceinline__ float orderedQ51DownPackets(const DeviceNativeQuantisedMatrixDesc& role,int column,
    const std::int8_t* codes,const float* scales)
{
    static_assert(Blocks==4 || Blocks==8);
    static_assert(PacketCount==2 || PacketCount==4);
    using Traits=PrefillNATIVE_QUANTISEDTraits<NATIVE_QUANTISED_PF_Q5_1>;
    using Payload=std::array<std::uint32_t,Traits::payload_bytes/sizeof(std::uint32_t)>;
    static_assert(sizeof(Payload)==Traits::payload_bytes);
    const auto partial=[&](const Payload& raw,int block,std::size_t index) {
        (void)index; // The payload was loaded by the caller; metadata addresses by block/column.
        const auto* bytes=reinterpret_cast<const std::uint8_t*>(raw.data());
        int32_t decoded[8];
        moe_prefill_decode_groups<NATIVE_QUANTISED_PF_Q5_1,false>(bytes,decoded,nullptr);
        return moe_prefill_accumulate_decoded_block_at<NATIVE_QUANTISED_PF_Q5_1>(
            role,block,column,hidden,bytes,decoded,reinterpret_cast<const int32_t*>(codes+block*32),scales[block],0.0f);
    };
    float total=0.0f;
#pragma unroll 1
    for(int block=0;block<Blocks;block+=PacketCount) {
        const auto firstIndex=std::size_t(block)*hidden+column;
        const auto secondIndex=firstIndex+hidden;
        const auto first=readPackedGlobal<Payload>(role.payload+firstIndex*Traits::payload_bytes);
        const auto second=readPackedGlobal<Payload>(role.payload+secondIndex*Traits::payload_bytes);
        if constexpr(PacketCount==4) {
            const auto thirdIndex=secondIndex+hidden;
            const auto fourthIndex=thirdIndex+hidden;
            const auto third=readPackedGlobal<Payload>(role.payload+thirdIndex*Traits::payload_bytes);
            const auto fourth=readPackedGlobal<Payload>(role.payload+fourthIndex*Traits::payload_bytes);
            const float a=partial(first,block,firstIndex);
            const float b=partial(second,block+1,secondIndex);
            const float c=partial(third,block+2,thirdIndex);
            const float d=partial(fourth,block+3,fourthIndex);
            total=rocm_native_quantised_accumulate_rn(total,a);
            total=rocm_native_quantised_accumulate_rn(total,b);
            total=rocm_native_quantised_accumulate_rn(total,c);
            total=rocm_native_quantised_accumulate_rn(total,d);
        } else {
            const float a=partial(first,block,firstIndex);
            const float b=partial(second,block+1,secondIndex);
            total=rocm_native_quantised_accumulate_rn(total,a);
            total=rocm_native_quantised_accumulate_rn(total,b);
        }
    }
    return total;
}

/** @brief Compute local contributions once per original expert and reduce in original slot order. */
template<unsigned Groups,bool OneBlockPerPartition,int RoutedPhysical=-1,int FixedRoutedWidth=0>
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
    if(live<0 || live>static_cast<int>(gridDim.y))__builtin_trap();
    const auto published=AcquiredGroupedRoutePublication::acquire(direct);
    const int row=blockIdx.y;
    __shared__ float complete[topK+1][columns];
    for(int route=wave;route<=topK;route+=threads/columns) {
        float dot=0.0f;
        const bool routed=route<topK;
        const int slot=routed?row*topK+route:row;
        const auto local=published.localDownIf(row<live && routed,slot,live);
        if(row<live && (!routed || local)) {
            float value;
            if constexpr(FixedRoutedWidth!=0) {
                static_assert(RoutedPhysical==NATIVE_QUANTISED_PF_Q5_1 &&
                              (FixedRoutedWidth==128 || FixedRoutedWidth==256));
                if(routed) {
                    value=orderedQ51DownPackets<FixedRoutedWidth/32,FixedRoutedWidth==128?4:2>(
                        local.descriptor(),column,scratch.routed_codes+slot*FixedRoutedWidth,
                        scratch.routed_scales+slot*(FixedRoutedWidth/32));
                } else {
                    value=fixedPublishedDown<NATIVE_QUANTISED_PF_Q8_0,sharedWidth/32>(
                        sharedDown.projection().descriptor(),column,scratch.shared_codes+row*sharedWidth,
                        scratch.shared_scales+row*(sharedWidth/32));
                }
            } else {
                const auto projection=routed?generic::ReadyProjectionFields::from(local):sharedDown.fields();
                const int span=routed?routedColumns:sharedColumns;
                const int blocks=span/32;
                const auto formats=routed?routedFormats:sharedDown.formats();
                const auto parts=routed?routedParts:sharedDown.partitions();
                const auto* codes=routed?scratch.routed_codes:scratch.shared_codes;
                const auto* scales=routed?scratch.routed_scales:scratch.shared_scales;
                const generic::ProjectionUsers<1> user{{{codes+slot*span,scales+slot*blocks,
                    routed && direct.affineSums?direct.affineSums->routedRows+slot*blocks:nullptr}},1};
                if constexpr(RoutedPhysical>=0) {
                    constexpr auto physical=static_cast<NativeQuantisedFormatPrefill>(RoutedPhysical);
                    if(routed && projection.descriptor().codebook_id!=RoutedPhysical)__builtin_trap();
                    value=routed?generic::orderedProjectionFormat<physical,1,1,OneBlockPerPartition>(
                        projection,column,span,user,parts,lane).value[0]:
                        generic::orderedProjectionFormat<NATIVE_QUANTISED_PF_Q8_0,1,1,OneBlockPerPartition>(
                        projection,column,span,user,parts,lane).value[0];
                } else {
                    value=generic::orderedProjection<1,1,Groups,OneBlockPerPartition>(projection,column,span,user,parts,lane,formats).value[0];
                }
            }
            dot=routed?rocm_native_quantised_weight_route_rn(published.weight(slot),value):
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

bool launchFormatProgram(const RoutePlan& plan,const DeviceRowRange& rows,
    const std::array<DeviceNativeQuantisedMatrixDesc,3>& sharedWeights,
    const Output& output,int device,void* stream,const DirectBankInput& direct,const BoundSourceFormatProgram& bound)
{
    const int physicalRows=rows.physicalRows();
    const auto geometry=rows.captureIdentity();
    if(!bound.matches(direct.runtime,device) || !supportsFormatRows(physicalRows) ||
       rows.countOwner()!=plan.live || geometry.first_physical_row!=0 ||
       geometry.count_capacity!=physicalRows || geometry.repeats_per_row!=1)
        return false;
    const auto* formatsProgram=std::get_if<SourceFormatProgram>(&bound.program());
    if(!formatsProgram)return false;
    const auto& program=*formatsProgram;
    const int routedColumns=program.routedColumns(), sharedColumns=program.sharedColumns();
    const auto& formats=program.formats();
    if(device<0 || reinterpret_cast<uintptr_t>(stream)<=2 || !direct.runtime || !direct.indices ||
       !direct.weights || !direct.bankIdentity ||
       (direct.demand!=MoEGroupedPlanDemand::None && direct.demand!=MoEGroupedPlanDemand::DeferredAcceptedRows) ||
       !plan.live || !plan.hiddenCodes || !plan.hiddenScales || !plan.hiddenFP32 ||
       !output.routedCodes || !output.routedScales || !output.sharedCodes || !output.sharedScales ||
       !output.sharedInputGate || !output.gateValues || !output.sharedGateScratch || !output.sharedUpScratch ||
       !output.routedOutput || !output.sharedOutput || !output.combined || !output.routeContributions ||
       routedColumns<=0 || routedColumns%32 || routedColumns>INT_MAX/(physicalRows*topK) ||
       sharedColumns<=0 || sharedColumns%32 || sharedColumns>INT_MAX/physicalRows ||
       (direct.affineSums && (physicalRows!=capacity || routedColumns!=160 || sharedColumns!=160)) ||
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
    const int maxTiles=physicalRows*(topK*(routedColumns/32)+sharedColumns/32);
    const int tileGrid=std::max(physicalRows,(maxTiles+owners-1)/owners);
    const auto gateFamilies=ResidentMoEFFNDecoderProgram::fromExecutionMask(
        formats.routedGateUp.execution_codebook_mask | formats.shared[0].execution_codebook_mask |
        formats.shared[1].execution_codebook_mask);
    const auto table=bound.dispatchTable();
    const auto reuseLimit=bound.reusableGateBlocks();
    const bool measured=table==native_quantised::VerifierDispatchTable::MI50;
    const auto recordGate=[&]<unsigned Groups,int RoutedCodebook,int FixedQ5Pitch,int K32Packets,bool DedicatedTiles>() {
        const int grid=DedicatedTiles?tileGrid:static_cast<int>(std::min<std::int64_t>(reuseLimit,tileGrid));
        hipLaunchKernelGGL((groupedFFNFormatProgramGate<Groups,RoutedCodebook,FixedQ5Pitch,K32Packets,DedicatedTiles>),
            dim3(grid),dim3(threads),0,native,plan.live,plan.hiddenCodes,plan.hiddenScales,plan.hiddenFP32,
            scratch,output,direct,physicalRows,routedColumns,sharedColumns,formats.routedGateUp,
            routedParts[0],routedParts[1],roles[0],roles[1]);
    };
    // The indexed readers below hide lookup latency by reusing two workgroups
    // per CU on the six-row MI50 grid. Other measured gate readers benefit
    // from dedicating a wave pair to each tile. Selection occurs at capture.
    using namespace native_quantised_formats;
    const bool reuseIndexedTiles=measured && physicalRows==6 && routedColumns==256 && sharedColumns==256 &&
        (program.hasOriginalGateReaders(IQ2_S,Q8_0,sharedWeights) ||
         program.hasOriginalGateReaders(IQ2_S,Q4_0,sharedWeights) ||
         program.hasOriginalGateReaders(IQ3_S,IQ4_NL,sharedWeights));
    const auto launchGate=[&]<unsigned Groups,int FixedQ5Pitch=0>() {
        // Reused workgroups hide packet latency on small separate-reader grids.
        // When the resident grid would repeat tiles, distribute the original
        // Q8 tiles to independent wave pairs and retire them after publication.
        if(tileGrid>reuseLimit && !reuseIndexedTiles)
            recordGate.template operator()<Groups,-1,FixedQ5Pitch,1,true>();
        else
            recordGate.template operator()<Groups,-1,FixedQ5Pitch,1,false>();
    };
    if(!measured) {
        // The portable reader supports every published format and preserves the
        // original K partitions. HIP reports the scheduling unit count in the native inventory.
        if(gateFamilies.groups()==1)recordGate.template operator()<1,-1,0,1,false>();
        else recordGate.template operator()<15,-1,0,1,false>();
    }
    else if(tileGrid>reuseLimit && routedColumns==256 && sharedColumns==256 &&
            program.hasOriginalGateReaders(IQ3_XXS,IQ2_S,sharedWeights)) {
        recordGate.template operator()<15,NATIVE_QUANTISED_PF_IQ3_XXS,0,1,true>();
    }
    else if(program.pairsGateAndUp()) {
        // With fewer workgroups than CUs, two outstanding packets hide the
        // original load dependency. A small tail also benefits on long grids
        // after two full resident rounds. In original intermediate grids a
        // second packet raised register lifetime and slowed the complete parent.
        const auto tail=tileGrid%reuseLimit;
        if(tileGrid<reuseLimit/2 || (tileGrid>=2*reuseLimit && tail>0 && tail<=reuseLimit/4))
            recordGate.template operator()<1,NATIVE_QUANTISED_PF_Q4_1,0,2,true>();
        else
            recordGate.template operator()<1,NATIVE_QUANTISED_PF_Q4_1,0,1,true>();
    }
    else if(const auto pitch=program.fixedQ5GatePitch(sharedWeights)) {
        switch(*pitch) {
            case FixedQ5GatePitch::F128: launchGate.template operator()<1,128>(); break;
            case FixedQ5GatePitch::F160: launchGate.template operator()<1,160>(); break;
            case FixedQ5GatePitch::F256: launchGate.template operator()<1,256>(); break;
        }
    }
    else if(gateFamilies.groups()==1) launchGate.template operator()<1>();
    else launchGate.template operator()<15>();
    if(hipGetLastError()!=hipSuccess)return false;
    const auto launchDown=[&]<unsigned Groups,bool OneBlockPerPartition,int RoutedPhysical=-1,int FixedRoutedWidth=0>() {
        hipLaunchKernelGGL((groupedFFNFormatProgramDown<Groups,OneBlockPerPartition,RoutedPhysical,FixedRoutedWidth>),dim3(hidden/64,physicalRows),dim3(threads),0,native,
            plan.live,scratch,output,direct,routedColumns,sharedColumns,formats.routedDown,
            routedParts[2],roles[2]);
    };
    if(const auto pitch=measured?program.fixedQ51DownPitch(sharedWeights[2]):std::nullopt) {
        // Affine-sum publication has its own F160 layout, disjoint from these widths.
        switch(*pitch) {
        case FixedQ51DownPitch::F128: launchDown.template operator()<15,true,NATIVE_QUANTISED_PF_Q5_1,128>(); break;
        case FixedQ51DownPitch::F256: launchDown.template operator()<15,true,NATIVE_QUANTISED_PF_Q5_1,256>(); break;
        }
        return hipGetLastError()==hipSuccess;
    }
    if(measured && formats.shared[2].execution_codebook_mask==nativeQuantisedCodebookMaskBit(NATIVE_QUANTISED_PF_Q8_0)) {
        switch(formats.routedDown.execution_codebook_mask) {
        case nativeQuantisedCodebookMaskBit(NATIVE_QUANTISED_PF_Q4_0): launchDown.template operator()<15,false,NATIVE_QUANTISED_PF_Q4_0>();return hipGetLastError()==hipSuccess;
        case nativeQuantisedCodebookMaskBit(NATIVE_QUANTISED_PF_IQ4_NL): launchDown.template operator()<15,false,NATIVE_QUANTISED_PF_IQ4_NL>();return hipGetLastError()==hipSuccess;
        case nativeQuantisedCodebookMaskBit(NATIVE_QUANTISED_PF_IQ4_XS_SOURCE): launchDown.template operator()<15,false,NATIVE_QUANTISED_PF_IQ4_XS_SOURCE>();return hipGetLastError()==hipSuccess;
        case nativeQuantisedCodebookMaskBit(NATIVE_QUANTISED_PF_MXFP4): launchDown.template operator()<15,false,NATIVE_QUANTISED_PF_MXFP4>();return hipGetLastError()==hipSuccess;
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
    const bool shortDown=roles[2].partitions()==sharedColumns/32 && routedParts[2]==routedColumns/32;
    if(shortDown) {
        const auto families=ResidentMoEFFNDecoderProgram::fromExecutionMask(
            formats.routedDown.execution_codebook_mask | formats.shared[2].execution_codebook_mask);
        if(families.groups()==1)launchDown.template operator()<1,true>();
        else launchDown.template operator()<15,true>();
    } else launchDown.template operator()<15,false>();
    return hipGetLastError()==hipSuccess;
}
