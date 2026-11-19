/**
 * @file ROCmFullNativeHCFFNLaunch.hip.inc
 * @brief Cold full-function admission and exact-stream submission of canonical HC/FFN.
 *
 * Original phase resolvers own arithmetic; the tuned bodies declare their exact
 * phase-exclusive LDS geometry. Only the actual composed function supplies
 * residency evidence. One retained barrier bank has
 * disjoint prefix/suffix slots; no standalone admission, hidden allocation or
 * host row-count observation is used to prepare or submit the full entry.
 */
#pragma once
#include "ROCmFullNativeHCFFN.h"

namespace llaminar2::rocm {
/** @copydoc fullNativeHCFFNKernelFunction */
const void* fullNativeHCFFNKernelFunction(ResidentMoEFFNDecoderProgram program)
{
#define LLAMINAR_FULL_HC_FAMILY(G) case G:return reinterpret_cast<const void*>(&fullNativeHCFFNKernel<ResidentMoEFFNDecoderProgram::familyMask<G>()>);
    switch(program.groups())
    {
        LLAMINAR_FULL_HC_FAMILY(1) LLAMINAR_FULL_HC_FAMILY(2) LLAMINAR_FULL_HC_FAMILY(3)
        LLAMINAR_FULL_HC_FAMILY(4) LLAMINAR_FULL_HC_FAMILY(5) LLAMINAR_FULL_HC_FAMILY(6)
        LLAMINAR_FULL_HC_FAMILY(7) LLAMINAR_FULL_HC_FAMILY(8) LLAMINAR_FULL_HC_FAMILY(9)
        LLAMINAR_FULL_HC_FAMILY(10) LLAMINAR_FULL_HC_FAMILY(11) LLAMINAR_FULL_HC_FAMILY(12)
        LLAMINAR_FULL_HC_FAMILY(13) LLAMINAR_FULL_HC_FAMILY(14) LLAMINAR_FULL_HC_FAMILY(15)
    }
#undef LLAMINAR_FULL_HC_FAMILY
    throw std::invalid_argument("Full native HC/FFN source envelope is invalid");
}

/**
 * @brief Validate the physical boundary connecting original previous write, HC and FFN.
 * @param args Complete borrowed operation, with backend-owned barrier offsets.
 * @details Individual phase validators retain their original alias rules. This
 * check adds the new write boundary; exact in-place residual write is permitted,
 * but input block/logits/gain cannot alias its output. Independent original row
 * authorities remain device-owned and are retained by the graph owner.
 */
static void validateFullNativeHCFFNBoundary(const FullNativeHCFFNArgs& args)
{
    const auto& p=args.prefix;const auto& hc=p.hc;const auto& write=p.write;
    const auto& suffix=args.suffix;
    const auto base=reinterpret_cast<std::uintptr_t>(p.barrier_state);
    if(!base||base%alignof(std::uint32_t)||
        reinterpret_cast<std::uintptr_t>(suffix.barrier_state)!=base+FullNativeHCFFNBarrierLayout::suffix_byte_offset||
        !write.residual||!write.block||!write.gateLogits||!write.output||write.output!=hc.norm.input||
        hc.norm.hidden<=0||hc.norm.hidden!=suffix.ffn.hidden_columns||hc.mixed!=suffix.ffn.hidden||
        hc.norm.extent.rows!=1||suffix.ffn.rows.physicalRows()!=1||hc.up.rows.physicalRows()!=1)
        throw std::invalid_argument("Full native HC/FFN requires the original complete write/HC/hidden/barrier boundary");
    const auto H=static_cast<std::size_t>(hc.norm.hidden),K=4*H;
    const std::array writes{Q8ResidualStorageSpan{write.output,K*sizeof(float),alignof(float)}};
    const std::array reads{
        Q8ResidualStorageSpan{write.residual,write.residual==write.output?0:K*sizeof(float),alignof(float)},
        Q8ResidualStorageSpan{write.block,H*sizeof(float),alignof(float)},
        Q8ResidualStorageSpan{write.gateLogits,4*sizeof(float),alignof(float)},
        Q8ResidualStorageSpan{hc.norm.gain,K*sizeof(float),alignof(float)}};
    validateQ8ResidualStorage(writes,reads);
}

/**
 * @brief Seal the full-only published role shortcut against every resolved source.
 * @param args SAME original cold-resolved arguments. @param masks Original source envelopes.
 * @details Standalone checked launchers retain their original policies without
 * this full-only shortcut. The actual resolver remains the source of all P
 * values; no generated policy, arithmetic or admission is substituted.
 */
static void validateFullPublishedProjectionPolicies(const ResidentMoEFFNArgs& args,
    ResidentMoEFFNPolicyMasks masks)
{
    const auto validate=[](ResidentMoEPartitionPolicy policy,std::uint32_t mask,int reduction) {
        const unsigned expected=MoEProjectionNumericalContract::orderedKPartitionsForWidth(reduction);
        const std::uint64_t words[]{policy.p0,policy.p1,policy.p2,policy.p3};
        for(unsigned source=0;source<32;++source)if(mask&(std::uint32_t{1}<<source)) {
            const unsigned count=unsigned((words[source/8]>>((source%8)*8))&0xffu);
            if(!expected||count!=expected||policy.maximum!=expected)
                throw std::invalid_argument("Full published FFN role disagrees with canonical ordered source partitions");
        }
    };
    validate(args.routed_gate_policy,masks.routed_gate,args.hidden_columns);
    validate(args.routed_down_policy,masks.routed_down,args.routed_columns);
    validate(args.shared_gate_policy,masks.shared_gate,args.hidden_columns);
    validate(args.shared_down_policy,masks.shared_down,args.shared_columns);
}
/** @copydoc prepareFullNativeHCFFN */
PreparedMoEFFNProgram prepareFullNativeHCFFN(DeviceId device,FullNativeHCFFNArgs& args,
    ResidentMoEFFNPolicyMasks masks,std::uint64_t generation)
{
    if(!device.is_rocm()||!generation||isGraphCaptureActive())
        throw std::invalid_argument("Full native HC/FFN requires its original cold ROCm owner");
    validateFullNativeHCFFNBoundary(args);
    resolvePersistentHCReadPolicy(args.prefix.hc,args.prefix.output_demand);
    const auto original_suffix=resolveNativeLayoutRouterFFNGeometry(device,args.suffix,masks,generation);
    validateFullPublishedProjectionPolicies(args.suffix.ffn,masks);
    const auto& hardware=rocmHardwareMetadata().at(device.ordinal);
    if(hardware.lds_per_block<=sizeof(FullNativeHCFFNShared))
        throw std::invalid_argument("Full native HC/FFN static phase storage exceeds device LDS capacity");
    const auto suffix=fullNativeHCFFNSuffixGeometry(args.suffix,original_suffix.decoder.executionMask(),
        hardware.lds_per_block-sizeof(FullNativeHCFFNShared));
    args.suffix.full_entry_panel={suffix.panel_rows,suffix.panel_blocks};
    // A single down/up owner bounds every positive resident-grid ownership
    // stripe. This removes a storage/occupancy cycle without a second admission
    // pass. The kernel uses the same typed envelope, executing only live work.
    const auto prefix_bytes=fullNativeHCFFNPrefixDynamicBytes(args.prefix,1);
    if(!prefix_bytes)
        throw std::invalid_argument("Full native HC/FFN has no valid prefix storage geometry");
    const auto dynamic_bytes=std::max(prefix_bytes,suffix.dynamic_shared_bytes);
    const void* function=fullNativeHCFFNKernelFunction(original_suffix.decoder);
    const auto admission=NativeCooperativeKernelBoundary<NativeCooperativeBackend::ROCm>::prepare(
        device,function,1024,dynamic_bytes,generation,NativeResidentLaunchPolicy::Ordinary);
    const auto& hc=args.prefix.hc;
    using Geometry=PersistentHCReadGeometry<1024>;
    const std::uint64_t down=(std::uint64_t(hc.down_weights.n)+Geometry::down_columns-1)/Geometry::down_columns*
        static_cast<unsigned>(hc.partitions)+(hc.injection.weights?4u:0u);
    const std::uint64_t quant=(std::uint64_t(hc.down_weights.k)/32+Geometry::quant_cohorts-1)/Geometry::quant_cohorts;
    const std::uint64_t publish=(std::uint64_t(hc.down_weights.n)+Geometry::threads-1)/Geometry::threads;
    const std::uint64_t up=(std::uint64_t(hc.norm.hidden)+Geometry::up_channels-1)/Geometry::up_channels;
    const std::uint64_t useful=std::max({std::uint64_t{4},down,quant,publish,up,suffix.useful_ctas});
    unsigned bound=admission.maximumGrid();
    if(hc.grid_policy==PersistentHCGridPolicy::OnePerProcessor)
        bound=std::min(bound,static_cast<unsigned>(rocmHardwareMetadata().at(device.ordinal).compute_units));
    else if(hc.grid_policy!=PersistentHCGridPolicy::MaximumResident)
        throw std::invalid_argument("Full native HC/FFN requires a declared HC grid policy");
    const unsigned grid=static_cast<unsigned>(std::min(useful,std::uint64_t(bound)));
    if(!grid)throw std::invalid_argument("Full native HC/FFN has no admitted useful resident grid");
    args.prefix.hc.resident_grid=grid;args.suffix.ffn.resident_grid=grid;
    ResidentMoEFFNProgramEvidence evidence{.kind=ResidentMoEFFNLaunchKind::OrdinaryResident,.device=device,.generation=generation};
    evidence.entries[0]={admission.function(),grid,admission.threads(),admission.dynamicSharedBytes(),admission.resources()};
    return PreparedMoEFFNProgram(evidence,admission);
}

/** @copydoc launchFullNativeHCFFN */
void launchFullNativeHCFFN(const PreparedMoEFFNProgram& program,const FullNativeHCFFNArgs& args,
    void* stream,std::uint64_t generation)
{
    const auto grid=args.suffix.ffn.resident_grid;
    const auto decoder=ResidentMoEFFNDecoderProgram::fromExecutionMask(args.suffix.ffn.execution.combined());
    const auto& hardware=rocmHardwareMetadata().at(program.evidence_.device.ordinal);
    if(hardware.lds_per_block<=sizeof(FullNativeHCFFNShared))
        throw std::logic_error("Full native HC/FFN retained device LDS capacity changed");
    const auto suffix_geometry=fullNativeHCFFNSuffixGeometry(args.suffix,decoder.executionMask(),
        hardware.lds_per_block-sizeof(FullNativeHCFFNShared));
    const FullNativeHCFFNSuffixPanel panel{suffix_geometry.panel_rows,suffix_geometry.panel_blocks};
    const auto prefix_bytes=fullNativeHCFFNPrefixDynamicBytes(args.prefix,1);
    const auto dynamic_bytes=std::max(prefix_bytes,suffix_geometry.dynamic_shared_bytes);
    if(!prefix_bytes||prefix_bytes>32768||(prefix_bytes&15u)||!program.cooperative_||program.cooperative_->policy()!=NativeResidentLaunchPolicy::Ordinary||
        program.evidence_.kind!=ResidentMoEFFNLaunchKind::OrdinaryResident||
        program.evidence_.entries[0].function!=fullNativeHCFFNKernelFunction(decoder)||
        program.evidence_.entries[0].grid!=grid||!grid||args.prefix.hc.resident_grid!=grid||
        args.suffix.full_entry_panel!=panel||program.evidence_.entries[0].dynamic_shared_bytes!=dynamic_bytes||
        grid>program.cooperative_->maximumGrid()||generation!=program.evidence_.generation||
        !args.prefix.barrier_state||args.suffix.barrier_state!=args.prefix.barrier_state+FullNativeHCFFNBarrierLayout::suffix_word_offset||
        !stream||stream==hipStreamLegacy||stream==hipStreamPerThread)
        throw std::logic_error("Full native HC/FFN exact prepared program/stream/binding changed");
    // Preserve the suffix's real first-kernarg ABI. Passing the host envelope as
    // one argument would silently invalidate its canonical scalar field loads.
    NativeLayoutRouterFFNArgs suffix=args.suffix;
    FullNativeHCFFNPrefixArgs prefix=args.prefix;
    // This immutable extent is already validated against the complete retained
    // launch identity above. Capture it after the two original argument records;
    // suffix field offsets and every tensor/state owner remain unchanged.
    unsigned captured_prefix_bytes=static_cast<unsigned>(prefix_bytes);
    void* parameters[]{&suffix,&prefix,&captured_prefix_bytes};
    const auto status=hipLaunchKernel(program.evidence_.entries[0].function,dim3(grid),dim3(1024),parameters,
        program.evidence_.entries[0].dynamic_shared_bytes,static_cast<hipStream_t>(stream));
    if(status!=hipSuccess)
        throw std::runtime_error(std::string("Full native HC/FFN resident launch: ")+hipGetErrorString(status));
}
}
