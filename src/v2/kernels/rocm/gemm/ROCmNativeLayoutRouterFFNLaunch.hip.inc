/**
 * @file ROCmNativeLayoutRouterFFNLaunch.hip.inc
 * @brief Exact residency admission and ordinary submission for the column-mapped FFN program.
 */
#include "ROCmNativeLayoutDownGeometry.h"
#include "ROCmNativeLayoutGateUpGeometry.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
namespace llaminar2::rocm {
const void* nativeLayoutRouterFFNKernelFunction(ResidentMoEFFNDecoderProgram program)
{
#define LLAMINAR_LAYOUT_FAMILY(G) case G:return reinterpret_cast<const void*>(&nativeLayoutRouterFFNKernel<ResidentMoEFFNDecoderProgram::familyMask<G>()>);
    switch(program.groups())
    {LLAMINAR_LAYOUT_FAMILY(1) LLAMINAR_LAYOUT_FAMILY(2) LLAMINAR_LAYOUT_FAMILY(3)
     LLAMINAR_LAYOUT_FAMILY(4) LLAMINAR_LAYOUT_FAMILY(5) LLAMINAR_LAYOUT_FAMILY(6)
     LLAMINAR_LAYOUT_FAMILY(7) LLAMINAR_LAYOUT_FAMILY(8) LLAMINAR_LAYOUT_FAMILY(9)
     LLAMINAR_LAYOUT_FAMILY(10) LLAMINAR_LAYOUT_FAMILY(11) LLAMINAR_LAYOUT_FAMILY(12)
     LLAMINAR_LAYOUT_FAMILY(13) LLAMINAR_LAYOUT_FAMILY(14) LLAMINAR_LAYOUT_FAMILY(15)}
#undef LLAMINAR_LAYOUT_FAMILY
    throw std::invalid_argument("Native-layout FFN source envelope is invalid");
}
/** @brief Original source-resolved suffix work/storage, independent of the composing native entry. */
struct NativeLayoutRouterFFNLaunchGeometry {
    ResidentMoEFFNDecoderProgram decoder; ///< Complete original physical source family.
    std::size_t dynamic_shared_bytes; ///< Exact phase-exclusive suffix LDS extent.
    std::uint64_t useful_ctas; ///< Maximum live phase work; never an occupancy estimate.
};
/**
 * @brief Resolve canonical suffix policy and live work without admitting another function.
 * @param device Original worker. @param args Existing canonical source/publication bindings.
 * @param masks Complete original policy/decoder envelopes. @param generation Nonzero retained owner generation.
 * @return Exact suffix work and storage reused by standalone and full-entry admission.
 */
static NativeLayoutRouterFFNLaunchGeometry resolveNativeLayoutRouterFFNGeometry(
    DeviceId device,NativeLayoutRouterFFNArgs& args,ResidentMoEFFNPolicyMasks masks,std::uint64_t generation)
{
    if(!device.is_rocm()||!generation||isGraphCaptureActive()||!args.router_weights||!args.router_logits||
        !args.router_publication.valid()||!args.barrier_state)throw std::invalid_argument("Native-layout FFN requires original cold router publication");
    auto& a=args.ffn;resolveResidentMoEFFNPolicies(a,masks);
    if(a.experts>kDeviceMoEMaxExperts||a.top_k>kDeviceMoEMaxTopK||a.rows.physicalRows()!=1)
        throw std::invalid_argument("Native-layout FFN requires the original admitted M1 router geometry");
    // The canonical source tree, not a native launch-tuning choice, owns P.
    // Every admitted source/alias resolves here before capture. A future wider
    // tree requires a new native implementation, never a silent banked mode.
    constexpr unsigned canonical_parts=MoEProjectionNumericalContract::ordered_k_partitions;
    static_assert(canonical_parts<=16,"Native-layout wave down requires the canonical source tree to fit its16 workers");
    if(a.routed_down_policy.maximum>canonical_parts||a.shared_down_policy.maximum>canonical_parts)
        throw std::invalid_argument("Native-layout wave down requires the canonical original source partition bound");
    const auto program=ResidentMoEFFNDecoderProgram::fromExecutionMask(masks.execution.combined());
    if(a.routed_gate_policy.maximum>canonical_parts||a.shared_gate_policy.maximum>canonical_parts)
        throw std::invalid_argument("Native-layout wave gate/up requires the canonical original source partition bound");
    const unsigned max_gate=std::max(a.routed_gate_policy.maximum,a.shared_gate_policy.maximum);
    const auto gate_wave=NativeLayoutDownWaveGeometry::resolve(max_gate);
    if(!gate_wave.workers)
        throw std::invalid_argument("Native-layout gate/up original partition geometry is unsupported");
    const auto routed_wave=NativeLayoutDownWaveGeometry::resolve(a.routed_down_policy.maximum);
    const auto shared_wave=NativeLayoutDownWaveGeometry::resolve(a.shared_down_policy.maximum);
    if(!routed_wave.workers||!shared_wave.workers)
        throw std::invalid_argument("Native-layout down original partition geometry is unsupported");
    const std::size_t hidden_bytes=std::size_t(a.hidden_columns)+std::size_t(a.hidden_columns/32)*8;
    const std::size_t values=std::size_t(a.top_k)*a.routed_columns+a.shared_columns;
    // Original gate/up partial planes are retired. Their exact dynamic region
    // now holds each CTA's immutable route pairs while the original static
    // selector stays live. H=32*m gives hidden_bytes=40*m, hence align8.
    static_assert(alignof(native_layout_ffn::PhaseProjection)==8);
    if(hidden_bytes%alignof(native_layout_ffn::PhaseProjection))
        throw std::invalid_argument("Native-layout gate/up role-record prefix is unaligned");
    const std::size_t first=hidden_bytes+std::size_t(2)*a.top_k*sizeof(native_layout_ffn::PhaseProjection);
    const std::size_t second=values+values/32*8+(std::size_t(1)+a.top_k)*NativeLayoutDownGeometry::columns*sizeof(float);
    const std::size_t dynamic=std::max(first,second);
    const auto gate=NativeLayoutGateUpGeometry::tasks(a.routed_columns,a.top_k,gate_wave.columns);
    const auto shared=NativeLayoutGateUpGeometry::tasks(a.shared_columns,1,gate_wave.columns);
    const std::uint64_t output=(std::uint64_t(a.hidden_columns)+NativeLayoutDownGeometry::columns-1)/NativeLayoutDownGeometry::columns;
    const std::uint64_t router=(std::uint64_t(a.experts)+7)/8;
    const std::uint64_t useful=std::max({NativeLayoutGateUpGeometry::usefulCTAs(gate),
        NativeLayoutGateUpGeometry::usefulCTAs(shared),output,router});
    return {program,dynamic,useful};
}
/** @copydoc prepareNativeLayoutRouterFFN */
PreparedMoEFFNProgram prepareNativeLayoutRouterFFN(DeviceId device,NativeLayoutRouterFFNArgs& args,
    ResidentMoEFFNPolicyMasks masks,std::uint64_t generation)
{
    const auto geometry=resolveNativeLayoutRouterFFNGeometry(device,args,masks,generation);
    const void* function=nativeLayoutRouterFFNKernelFunction(geometry.decoder);
    const auto admission=NativeCooperativeKernelBoundary<NativeCooperativeBackend::ROCm>::prepare(device,function,
        1024,geometry.dynamic_shared_bytes,generation,NativeResidentLaunchPolicy::Ordinary);
    auto& a=args.ffn;
    a.resident_grid=unsigned(std::min(geometry.useful_ctas,std::uint64_t(admission.maximumGrid())));
    ResidentMoEFFNProgramEvidence evidence{.kind=ResidentMoEFFNLaunchKind::OrdinaryResident,.device=device,.generation=generation};
    evidence.entries[0]={admission.function(),a.resident_grid,admission.threads(),admission.dynamicSharedBytes(),admission.resources()};
    return PreparedMoEFFNProgram(evidence,admission);
}
void launchNativeLayoutRouterFFN(const PreparedMoEFFNProgram& program,const NativeLayoutRouterFFNArgs& args,
    void* stream,std::uint64_t generation)
{
    if(!program.cooperative_||program.cooperative_->policy()!=NativeResidentLaunchPolicy::Ordinary||
       program.evidence_.kind!=ResidentMoEFFNLaunchKind::OrdinaryResident||
       program.evidence_.entries[0].grid!=args.ffn.resident_grid)
        throw std::logic_error("Native-layout FFN native program binding changed");
    NativeLayoutRouterFFNArgs copy=args;void* parameters[]{&copy};
    if(!stream||stream==hipStreamLegacy||stream==hipStreamPerThread||generation!=program.evidence_.generation||
        args.ffn.resident_grid>program.cooperative_->maximumGrid()||!args.barrier_state)
        throw std::logic_error("Native-layout ordinary resident launch binding changed");
    const auto status=hipLaunchKernel(program.evidence_.entries[0].function,dim3(args.ffn.resident_grid),dim3(1024),
        parameters,program.evidence_.entries[0].dynamic_shared_bytes,static_cast<hipStream_t>(stream));
    if(status!=hipSuccess)throw std::runtime_error(std::string("Native-layout ordinary resident launch: ")+hipGetErrorString(status));
}
}
