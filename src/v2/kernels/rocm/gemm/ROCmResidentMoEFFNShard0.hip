/**
 * @file ROCmResidentMoEFFNShard0.hip
 * @brief Resident MoE FFN shard 0 (decoder groups 1..8 and lifecycle launchers).
 */
#include "ROCmNativeQuantisedMoEDeviceHelpers.hip.inc"
#include "ROCmResidentMoEFFNBody.hip.inc"

LLAMINAR_DEFINE_ROCM_IQ_GRID_INIT(resident_ffn_0)

namespace llaminar2::rocm
{
namespace
{
thread_local ResidentMoEFFNProgram g_resident_moe_ffn_program=ResidentMoEFFNProgram::Original;
/** @brief Name both exact scheduler entries for one explicit decoder-structure union. */
template<unsigned Groups>
const void* residentMoEFFNFunctionForWidth(ResidentMoEFFNSchedulerWidth width)
{
    constexpr auto mask=ResidentMoEFFNDecoderProgram::familyMask<Groups>();
    switch(width)
    {
    case ResidentMoEFFNSchedulerWidth::UInt32:
        return reinterpret_cast<const void*>(&rocmResidentMoEFFNKernel<std::uint32_t,mask>);
    case ResidentMoEFFNSchedulerWidth::UInt64:
        return reinterpret_cast<const void*>(&rocmResidentMoEFFNKernel<std::uint64_t,mask>);
    }
    throw std::invalid_argument("Resident FFN decoder program has an invalid scheduler width");
}
}
/** @copydoc currentResidentMoEFFNProgram */
ResidentMoEFFNProgram currentResidentMoEFFNProgram() noexcept {return g_resident_moe_ffn_program;}
/** @copydoc ScopedResidentMoEFFNProgram::ScopedResidentMoEFFNProgram */
ScopedResidentMoEFFNProgram::ScopedResidentMoEFFNProgram(ResidentMoEFFNProgram program)
    :previous_(g_resident_moe_ffn_program)
{
    if(::llaminar2::isGraphCaptureActive())throw std::logic_error("Resident FFN program must be selected before capture");
    if(program!=ResidentMoEFFNProgram::Original&&program!=ResidentMoEFFNProgram::PartitionLocal&&
       program!=ResidentMoEFFNProgram::OrdinaryPhases&&program!=ResidentMoEFFNProgram::NativeLayoutRouter)
        throw std::invalid_argument("Resident FFN program selection is unknown");
    g_resident_moe_ffn_program=program;
}
ScopedResidentMoEFFNProgram::~ScopedResidentMoEFFNProgram(){g_resident_moe_ffn_program=previous_;}

/** @copydoc residentMoEFFNKernelFunction */
const void* residentMoEFFNKernelFunctionShard1(ResidentMoEFFNDecoderProgram program, ResidentMoEFFNSchedulerWidth width);
const void* residentMoEFFNKernelFunction(ResidentMoEFFNDecoderProgram program, ResidentMoEFFNSchedulerWidth width)
{
    if(currentResidentMoEFFNProgram()==ResidentMoEFFNProgram::PartitionLocal)
    {
        if(program.groups()!=1)
            throw std::invalid_argument("Partition-local FFN diagnostic requires the complete affine/byte execution envelope {5,7,19,23}");
        if(width!=ResidentMoEFFNSchedulerWidth::UInt32&&width!=ResidentMoEFFNSchedulerWidth::UInt64)
            throw std::invalid_argument("Partition-local FFN diagnostic scheduler width is unknown");
        return width==ResidentMoEFFNSchedulerWidth::UInt32
            ?reinterpret_cast<const void*>(&rocmPersistentMoEFFNKernel<std::uint32_t>)
            :reinterpret_cast<const void*>(&rocmPersistentMoEFFNKernel<std::uint64_t>);
    }
    if(currentResidentMoEFFNProgram()==ResidentMoEFFNProgram::OrdinaryPhases)
        throw std::invalid_argument("Ordinary FFN has two native entries; use its exact ordinary program identity");
    // All fifteen source-envelope unions are real compiled entries. No admission
    // failure, current route or current physical residency changes this choice.
    switch(program.groups())
    {
    case 1:return residentMoEFFNFunctionForWidth<1>(width);
    case 2:return residentMoEFFNFunctionForWidth<2>(width);
    case 3:return residentMoEFFNFunctionForWidth<3>(width);
    case 4:return residentMoEFFNFunctionForWidth<4>(width);
    case 5:return residentMoEFFNFunctionForWidth<5>(width);
    case 6:return residentMoEFFNFunctionForWidth<6>(width);
    case 7:return residentMoEFFNFunctionForWidth<7>(width);
    case 8:return residentMoEFFNFunctionForWidth<8>(width);
    case 9:
    case 10:
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:return residentMoEFFNKernelFunctionShard1(program, width);
    }
    throw std::invalid_argument("Resident FFN decoder program is not in its complete registry");
}
/** @brief Resolve exactly the existing serial-M1 source-policy authority for either submission mechanism. */
void resolveResidentMoEFFNPolicies(ResidentMoEFFNArgs& args,ResidentMoEFFNPolicyMasks masks)
{
    const auto policy=[](int N,int K,std::uint32_t mask) {
        ROCmMoESerialM1PolicyTable original{};
        if(!llaminar2::rocm::native_quantised_internal::buildROCmMoESerialM1PolicyTable(N,K,mask,original))
            throw std::invalid_argument("Resident FFN has no complete original serial-M1 partition policy");
        return ResidentMoEPartitionPolicy{original.k_partitions_0_7,original.k_partitions_8_15,
            original.k_partitions_16_23,original.k_partitions_24_31,original.maximum_k_partitions};
    };
    if(!args.runtime||!args.hidden||!args.shared_gate||!args.shared_up||!args.shared_down||
        !args.original.enabled()||args.rows.physicalRows()!=1||
        args.hidden_columns<=0||args.routed_columns<=0||args.shared_columns<=0||args.experts<=0||args.top_k<=0||
        args.top_k>kDeviceMoEMaxTopK||args.hidden_columns%32||args.routed_columns%32||args.shared_columns%32)
        throw std::invalid_argument("Resident FFN requires an original native M1 fragment and complete aligned geometry");
    if(!masks.execution.routed_gate||!masks.execution.routed_down||
        !masks.execution.shared_gate||!masks.execution.shared_down)
        throw std::invalid_argument("Resident FFN requires every original physical role envelope");
    args.execution=masks.execution;
    args.routed_gate_policy=policy(args.routed_columns,args.hidden_columns,masks.routed_gate);
    args.routed_down_policy=policy(args.hidden_columns,args.routed_columns,masks.routed_down);
    args.shared_gate_policy=policy(args.shared_columns,args.hidden_columns,masks.shared_gate);
    args.shared_down_policy=policy(args.hidden_columns,args.shared_columns,masks.shared_down);
}
namespace
{
/** @brief Cold exact ordinary function/geometry observation; residency is evidence, never a grid clamp. */
ResidentMoEFFNEntryEvidence prepareOrdinaryMoEFFNEntry(DeviceId device,const void* function,
    std::uint64_t grid,std::size_t dynamic_shared)
{
    const auto& hardware=rocmHardwareMetadata().at(device.ordinal);
    constexpr unsigned threads=256;
    // HSA dispatch packet work-item dimensions are uint32. The useful grid is
    // never padded or clamped to occupancy; an unrepresentable launch rejects.
    if(!function||!grid||grid>std::numeric_limits<unsigned>::max()/threads||
        hardware.maximum_threads_per_block<int(threads)||dynamic_shared>hardware.lds_per_block)
        throw std::invalid_argument("Ordinary FFN exceeds its exact native grid/CTA/LDS extent");
    int current=-1;
    if(hipGetDevice(&current)!=hipSuccess||current!=device.ordinal)
        throw std::logic_error("Ordinary FFN preparation has a foreign current worker");
    hipFuncAttributes attributes{};int resident_blocks=0;
    if(hipFuncGetAttributes(&attributes,function)!=hipSuccess||
        hipOccupancyMaxActiveBlocksPerMultiprocessor(&resident_blocks,function,threads,dynamic_shared)!=hipSuccess||
        attributes.numRegs<=0||attributes.maxThreadsPerBlock<int(threads)||resident_blocks<=0||
        attributes.localSizeBytes!=0||attributes.sharedSizeBytes>hardware.lds_per_block-dynamic_shared)
        throw std::invalid_argument("Ordinary FFN function has invalid compiler resources or exact dynamic LDS residency");
    return {.function=function,.grid=unsigned(grid),.threads=threads,.dynamic_shared_bytes=dynamic_shared,
        .resources={unsigned(attributes.numRegs),unsigned(attributes.maxThreadsPerBlock),unsigned(resident_blocks),
            attributes.localSizeBytes,attributes.sharedSizeBytes}};
}
/** @brief Bind two useful ordinary work grids and exact local activation/fold storage from original policies. */
ResidentMoEFFNProgramEvidence prepareOrdinaryMoEFFNEvidence(DeviceId device,const ResidentMoEFFNArgs& args,
    ResidentMoEFFNDecoderProgram program,std::uint64_t generation)
{
    const auto tiles=[](int columns,unsigned parts) {
        if(!parts||parts>unsigned(NativeQuantisedGroupedDecodePolicy::maximum_k_partitions))
            throw std::invalid_argument("Ordinary FFN source policy exceeds its original supported partition inventory");
        const auto width=256/parts;
        return (std::uint64_t(columns)+width-1)/width;
    };
    const auto grid_a=tiles(args.routed_columns,args.routed_gate_policy.maximum)*unsigned(args.top_k)+
        tiles(args.shared_columns,args.shared_gate_policy.maximum);
    const auto grid_b=(std::uint64_t(args.hidden_columns)+31)/32;
    const auto hidden=std::uint64_t(args.hidden_columns),values=std::uint64_t(args.top_k)*args.routed_columns+args.shared_columns;
    const auto dynamic_a=hidden+8*(hidden/32);
    const unsigned workers=std::min(8u,args.routed_down_policy.maximum);
    if(!workers||!args.shared_down_policy.maximum||
        args.routed_down_policy.maximum>unsigned(NativeQuantisedGroupedDecodePolicy::maximum_k_partitions)||
        args.shared_down_policy.maximum>unsigned(NativeQuantisedGroupedDecodePolicy::maximum_k_partitions))
        throw std::invalid_argument("Ordinary FFN down source policy is incomplete");
    const auto partial_rows=std::max((8/workers)*args.routed_down_policy.maximum,args.shared_down_policy.maximum);
    const auto dynamic_b=values+8*(values/32)+std::uint64_t(32)*partial_rows*sizeof(float)+
        std::uint64_t(args.top_k)*32*sizeof(float);
    if(dynamic_a>std::numeric_limits<std::size_t>::max()||dynamic_b>std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("Ordinary FFN exact dynamic LDS exceeds the native address type");
    const auto functions=ordinaryMoEFFNKernelFunctions(program);
    ResidentMoEFFNProgramEvidence evidence{.kind=ResidentMoEFFNLaunchKind::OrdinaryPhases,.device=device,.generation=generation};
    evidence.entries[0]=prepareOrdinaryMoEFFNEntry(device,functions[0],grid_a,std::size_t(dynamic_a));
    evidence.entries[1]=prepareOrdinaryMoEFFNEntry(device,functions[1],grid_b,std::size_t(dynamic_b));
    return evidence;
}
}

namespace
{
/** @brief Reuse the existing finite source-envelope registry for both ordinary phases. */
template<unsigned Groups>
std::array<const void*,2> ordinaryMoEFFNFunctionsForFamily()
{
    constexpr auto mask=ResidentMoEFFNDecoderProgram::familyMask<Groups>();
    return {reinterpret_cast<const void*>(&rocmOrdinaryMoEFFNGateUpKernel<mask>),
        reinterpret_cast<const void*>(&rocmOrdinaryMoEFFNDownKernel<mask>)};
}
}
/** @copydoc ordinaryMoEFFNKernelFunctions */
std::array<const void*,2> ordinaryMoEFFNKernelFunctions(ResidentMoEFFNDecoderProgram program)
{
    if(!program.requiredMask()||(program.requiredMask()&~ResidentMoEFFNDecoderProgram::all_decoders))
        throw std::invalid_argument("Ordinary FFN requires a complete supported immutable execution envelope");
    switch(program.groups())
    {
    case 1:return ordinaryMoEFFNFunctionsForFamily<1>();
    case 2:return ordinaryMoEFFNFunctionsForFamily<2>();
    case 3:return ordinaryMoEFFNFunctionsForFamily<3>();
    case 4:return ordinaryMoEFFNFunctionsForFamily<4>();
    case 5:return ordinaryMoEFFNFunctionsForFamily<5>();
    case 6:return ordinaryMoEFFNFunctionsForFamily<6>();
    case 7:return ordinaryMoEFFNFunctionsForFamily<7>();
    case 8:return ordinaryMoEFFNFunctionsForFamily<8>();
    case 9:return ordinaryMoEFFNFunctionsForFamily<9>();
    case 10:return ordinaryMoEFFNFunctionsForFamily<10>();
    case 11:return ordinaryMoEFFNFunctionsForFamily<11>();
    case 12:return ordinaryMoEFFNFunctionsForFamily<12>();
    case 13:return ordinaryMoEFFNFunctionsForFamily<13>();
    case 14:return ordinaryMoEFFNFunctionsForFamily<14>();
    case 15:return ordinaryMoEFFNFunctionsForFamily<15>();
    }
    throw std::invalid_argument("Ordinary FFN source-envelope family is invalid");
}

/** @copydoc PreparedMoEFFNProgram::PreparedMoEFFNProgram */
PreparedMoEFFNProgram::PreparedMoEFFNProgram(ResidentMoEFFNProgramEvidence evidence,
    std::optional<NativeCooperativeKernelAdmission> cooperative)
    :evidence_(evidence),cooperative_(std::move(cooperative))
{
    if((evidence_.kind!=ResidentMoEFFNLaunchKind::Cooperative&&evidence_.kind!=ResidentMoEFFNLaunchKind::OrdinaryPhases&&evidence_.kind!=ResidentMoEFFNLaunchKind::OrdinaryResident)||
        !evidence_.device.is_rocm()||evidence_.device.ordinal<0||!evidence_.generation)
        throw std::invalid_argument("Prepared FFN has an incomplete typed native program identity");
    for(const auto& entry:evidence_.nativeEntries())
        if(!entry.function||!entry.grid||(entry.threads!=256&&entry.threads!=1024)||!entry.resources.registers_per_thread||
            entry.resources.maximum_threads_per_block<entry.threads||!entry.resources.blocks_per_processor||entry.resources.private_bytes)
            throw std::invalid_argument("Prepared FFN has incomplete native entry/geometry/resource evidence");
    if(evidence_.kind!=ResidentMoEFFNLaunchKind::OrdinaryPhases)
    {
        const auto& entry=evidence_.entries[0];
        if(!cooperative_||cooperative_->device()!=evidence_.device||cooperative_->generation()!=evidence_.generation||
            cooperative_->function()!=entry.function||cooperative_->threads()!=entry.threads||
            cooperative_->dynamicSharedBytes()!=entry.dynamic_shared_bytes||entry.grid>cooperative_->maximumGrid()||
            evidence_.entries[1].function)
            throw std::invalid_argument("Prepared FFN cooperative observations do not match its actual admission");
    }
    else if(cooperative_||evidence_.entries[0].function==evidence_.entries[1].function)
        throw std::invalid_argument("Prepared ordinary FFN must retain two real entries and no cooperative admission");
}

/** @copydoc prepareResidentMoEFFN */
PreparedMoEFFNProgram prepareResidentMoEFFN(DeviceId device,ResidentMoEFFNArgs& args,
    ResidentMoEFFNPolicyMasks masks,std::uint64_t generation)
{
    if(!device.is_rocm()||!generation||isGraphCaptureActive())
        throw std::logic_error("FFN preparation requires its native owner generation outside capture");
    resolveResidentMoEFFNPolicies(args,masks);
    const auto program=ResidentMoEFFNDecoderProgram::fromExecutionMask(masks.execution.combined());
    if(currentResidentMoEFFNProgram()==ResidentMoEFFNProgram::OrdinaryPhases)
    {
        args.resident_grid=0; // Ordinary useful grids belong to the typed entry evidence.
        return PreparedMoEFFNProgram(prepareOrdinaryMoEFFNEvidence(device,args,program,generation),std::nullopt);
    }
    // Producer maxima bound folds/SwiGLU and output folds as every original
    // policy has at least one partition. Hidden quantization needs eight Q8
    // groups/CTA, no more than ceil(H/64)/4 CTAs. No live count is read here.
    const std::array phases{
        ResidentMoEFFNTaskGeometry::projection(args.routed_columns,args.top_k,args.routed_gate_policy.maximum),
        ResidentMoEFFNTaskGeometry::projection(args.hidden_columns,args.top_k,args.routed_down_policy.maximum),
        ResidentMoEFFNTaskGeometry::projection(args.shared_columns,1,args.shared_gate_policy.maximum),
        ResidentMoEFFNTaskGeometry::projection(args.hidden_columns,1,args.shared_down_policy.maximum)};
    const auto& geometry=*std::max_element(phases.begin(),phases.end(),
        [](const auto& left,const auto& right){return left.tasks()<right.tasks();});
    std::uint64_t structural_tasks=0;
    if(currentResidentMoEFFNProgram()==ResidentMoEFFNProgram::PartitionLocal)
    {
        const auto tiles=[](int columns,unsigned partitions)->std::uint64_t {
            if(!partitions||partitions>NativeQuantisedGroupedDecodePolicy::maximum_k_partitions||partitions>256)
                throw std::invalid_argument("Partition-local FFN has an invalid original partition policy");
            const auto width=256/partitions;
            return (std::uint64_t(columns)+width-1)/width;
        };
        const auto gate=tiles(args.routed_columns,args.routed_gate_policy.maximum)*args.top_k+
            tiles(args.shared_columns,args.shared_gate_policy.maximum);
        const auto down=tiles(args.hidden_columns,args.routed_down_policy.maximum)*args.top_k+
            tiles(args.hidden_columns,args.shared_down_policy.maximum);
        const auto values=std::uint64_t(args.top_k)*args.routed_columns+args.shared_columns;
        const auto quant=(values+255)/256;
        const auto hidden=(std::uint64_t(args.hidden_columns)+255)/256;
        structural_tasks=std::max({gate,down,quant,hidden});
    }
    // Select the exact useful-CTA scheduler width BEFORE admitting its native
    // function. H/F/topK are positive signed32 source axes: their products and
    // summed routed/shared task counts fit uint64; no tensor address narrows.
    const auto width=structural_tasks?residentMoEFFNSchedulerWidthForExtent(structural_tasks):
        geometry.uses32BitScheduler()?ResidentMoEFFNSchedulerWidth::UInt32:ResidentMoEFFNSchedulerWidth::UInt64;
    const void* function=residentMoEFFNKernelFunction(program,width);
    const auto admission=NativeCooperativeKernelBoundary<NativeCooperativeBackend::ROCm>::prepare(device,
        function,256,0,generation);
    args.resident_grid=structural_tasks
        ?static_cast<unsigned>(std::min(std::uint64_t(admission.maximumGrid()),structural_tasks))
        :geometry.grid(admission.maximumGrid());
    ResidentMoEFFNProgramEvidence evidence{.kind=ResidentMoEFFNLaunchKind::Cooperative,.device=device,.generation=generation};
    evidence.entries[0]={admission.function(),args.resident_grid,admission.threads(),
        admission.dynamicSharedBytes(),admission.resources()};
    return PreparedMoEFFNProgram(evidence,admission);
}
/** @copydoc launchResidentMoEFFN */
void launchResidentMoEFFN(const PreparedMoEFFNProgram& program,const ResidentMoEFFNArgs& args,
    void* stream,std::uint64_t generation)
{
    const auto native=static_cast<hipStream_t>(stream);
    if(!native||native==hipStreamLegacy||native==hipStreamPerThread)
        throw std::invalid_argument("FFN program requires its exact non-default native stream");
    if(generation!=program.evidence_.generation)
        throw std::logic_error("FFN program owner generation changed before native submission");
    ResidentMoEFFNArgs copied=args;void* parameters[]{&copied};
    if(program.evidence_.kind==ResidentMoEFFNLaunchKind::Cooperative)
    {
        if(args.resident_grid!=program.evidence_.entries[0].grid)
            throw std::logic_error("Cooperative FFN changed its sealed native grid");
        NativeCooperativeKernelBoundary<NativeCooperativeBackend::ROCm>::launch(*program.cooperative_,parameters,
            args.resident_grid,stream,generation);
        return;
    }
    for(const auto& entry:program.evidence_.nativeEntries())
    {
        const auto status=hipLaunchKernel(entry.function,dim3(entry.grid),dim3(entry.threads),parameters,
            entry.dynamic_shared_bytes,native);
        if(status!=hipSuccess)throw std::runtime_error(std::string("ordinary FFN native phase launch failed: ")+hipGetErrorString(status));
    }
}
}

// ============================================================================
// Reduce kernel: sum K-partitions from scatter buffer to final output.
// Simple summation — no per-column scales or bias (handled externally).
//
// Grid: dim3((N + TILE_N - 1) / TILE_N), Block: dim3(TILE_N / CPT)
// ============================================================================
/**
 * @brief Fold one original tile in the unchanged ascending-KB explicit-RN order.
 * @tparam TILE_N Original reducer tile width.
 * @tparam CPT Original independent columns per lane.
 * @param d_partial_fp32 Original producer's distinct [KB][N] partial bank.
 * @param d_C_fp32 Original output owner.
 * @param N Original physical width.
 * @param kblocks Original resolved arithmetic partition count.
 * @param tile Projection-local output tile.
 * @param lane Original workgroup lane.
 */
/** @brief One original increasing-partition RN reducer result. @param partials Original plane.
 * @param n Original output column. @param N Original column stride. @param kblocks Exact original partitions. @return Original bits. */
