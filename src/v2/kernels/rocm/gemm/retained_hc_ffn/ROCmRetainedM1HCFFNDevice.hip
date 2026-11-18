/**
 * @file ROCmRetainedM1HCFFNDevice.hip
 * @brief Native retained fixed-shape HC/FFN from the authenticated fast-body replay.
 *
 * The original graph owns the tensors, row range, descriptor bank and persistent
 * scratch. This translation unit owns only the exact resident device entry
 * and its byte layout. Castagna ancestry and MIT terms accompany the body.
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "CastagnaNativeWeightExport.h"
#include "ROCmRetainedM1HCFFNDevice.h"
#include "kernels/common/NativeCooperativeKernel.cuh"
#include "kernels/rocm/moe/ROCmNativeRuntimeBankView.h"
#include "RuntimeNativeExpertTableView.hip.inc"
#include "CanonicalFFNQ8.h"
#include "kernels/common/DeviceNativeQuantisedContributionContract.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
#include "kernels/rocm/common/ROCmSignedDot4.h"
#include "kernels/qwen4exp/GPUBranchRMSNormBody.cuh"
#include "kernels/common/OriginalSingleQ8Block32.cuh"
#include "kernels/common/GPUHCDownPublication.cuh"
#include "kernels/common/QuantizedActivationOperandPrecision.h"
#include "kernels/rocm/gemm/ROCmNativeQuantisedDecodeEquivalentMath.h"
#include "kernels/rocm/moe/ROCmOriginalRouterPhase.hip.inc"
#include "kernels/common/FullNativeHCFFNBarrierLayout.h"
#include <bit>
#include <type_traits>

#include "FastBankDeviceSection.inc"

namespace llaminar2::rocm {
using namespace qfn_retained_m1_hc_ffn_device;

/** @return The compiler's retained instance byte extent for the original PMA quote. */
std::size_t retainedM1HCFFNScratchBytes() noexcept
{
    constexpr std::size_t prefix=(sizeof(InstanceOriginalScratch)+15)&~std::size_t(15);
    return prefix+sizeof(CurrentHCPrefixState)+FullNativeHCFFNBarrierLayout::bytes;
}

/** @return The exact fixed-shape native kernel for cold residency admission. */
const void* retainedM1HCFFNKernelFunction() noexcept
{
    return reinterpret_cast<const void*>(&hc_persist_kernel<
        1,true,false,false,false,qfn_castagna_reference::GateSource::Q4_K,true,true,true,true,true>);
}

/** @copydoc retainedM1HCFFNKernelFunction(FullNativeHCFFNOutputDemand) */
const void* retainedM1HCFFNKernelFunction(FullNativeHCFFNOutputDemand demand,
    MoESelectedDescriptorDemand descriptors) noexcept
{
    if(descriptors==MoESelectedDescriptorDemand::AcquiredBankOnly) {
        if(demand.bits())return reinterpret_cast<const void*>(&hc_persist_kernel<
            1,true,false,false,false,qfn_castagna_reference::GateSource::Q4_K,true,true,true,true,true,false>);
        return reinterpret_cast<const void*>(&hc_persist_kernel<
            1,true,false,false,false,qfn_castagna_reference::GateSource::Q4_K,false,true,true,true,true,false>);
    }
    if(descriptors!=MoESelectedDescriptorDemand::ExternalTables)return nullptr;
    if(demand.bits())return retainedM1HCFFNKernelFunction();
    return reinterpret_cast<const void*>(&hc_persist_kernel<
        1,true,false,false,false,qfn_castagna_reference::GateSource::Q4_K,false,true,true,true,true>);
}

/** @copydoc prepareRetainedM1HCFFN */
PreparedMoEFFNProgram prepareRetainedM1HCFFN(DeviceId device,FullNativeHCFFNArgs& full,
    ResidentMoEFFNPolicyMasks masks,std::uint64_t generation)
{
    if(!device.is_rocm()||full.prefix.implementation!=FullNativeHCFFNImplementation::RetainedGFX906M1)
        throw std::invalid_argument("Retained M1 HC/FFN requires an original ROCm experimental declaration");
    const auto& hardware=rocmHardwareMetadata().at(device.ordinal);
    if((hardware.isa_target!="gfx906"&&!hardware.isa_target.starts_with("gfx906:"))||
       hardware.compute_units!=NB||reinterpret_cast<std::uintptr_t>(full.prefix.hc.mixed)%16)
        throw std::invalid_argument("Retained M1 HC/FFN requires exact gfx906/60CU and aligned mixed storage");
    resolvePersistentHCReadPolicy(full.prefix.hc,full.prefix.output_demand);
    const auto* function=retainedM1HCFFNKernelFunction(full.prefix.output_demand,full.prefix.selected_descriptor_demand);
    if(!function)throw std::invalid_argument("Retained M1 HC/FFN selected descriptor demand is invalid");
    const auto native=NativeCooperativeKernelBoundary<NativeCooperativeBackend::ROCm>::prepare(
        device,function,NT,0,generation,NativeResidentLaunchPolicy::Ordinary);
    if(native.maximumGrid()<NB)
        throw std::invalid_argument("Retained M1 HC/FFN is not fully resident on the exact worker");
    full.prefix.hc.resident_grid=NB;
    full.suffix.ffn.resident_grid=NB;
    full.suffix.ffn.execution=masks.execution;
    ResidentMoEFFNProgramEvidence evidence{.kind=ResidentMoEFFNLaunchKind::OrdinaryResident,.device=device,.generation=generation};
    evidence.entries[0]={native.function(),NB,native.threads(),native.dynamicSharedBytes(),native.resources()};
    return PreparedMoEFFNProgram(evidence,native);
}

/** @copydoc launchRetainedM1HCFFNDevice */
void launchRetainedM1HCFFNDevice(const FullNativeHCFFNArgs& full,
    const DeviceNativeQuantisedMatrixDesc& shared_gate,const DeviceNativeQuantisedMatrixDesc& shared_up,
    const DeviceNativeQuantisedMatrixDesc& shared_down,void* scratch,std::uint64_t generation,void* stream)
{
    if(reinterpret_cast<std::uintptr_t>(stream)<=2 || !scratch || !generation ||
       reinterpret_cast<std::uintptr_t>(scratch)%alignof(InstanceOriginalScratch))
        throw std::logic_error("Retained M1 HC/FFN lost its original stream/slot binding");
    const auto& p=full.prefix;
    const auto& r=full.suffix;
    const auto& hc=p.hc;
    auto& instance=*static_cast<InstanceOriginalScratch*>(scratch);
    const hc_src source{.x=p.write.residual,.out=p.write.block,.raw=p.write.gateLogits,.ps1=.25f,.ps2=2.f};
    hc_ffn ffn{};
    ffn.canonical_router_fp16=static_cast<const std::uint16_t*>(r.router_weights);
    ffn.canonical_normalize_weights=r.normalize_router_weights;
    ffn.w_sg=r.ffn.shared_input_gate;
    // The raw logits remain private until every CTA completes the exact
    // original selection. An observed probability write cannot alias them.
    ffn.logits=instance.g_hcp_shexp[0];
    ffn.on=1;ffn.ff=r.ffn.shared_columns;
    ffn.native_sgate=shared_gate;ffn.native_sup=shared_up;ffn.native_sdown=shared_down;
    auto& moe=ffn.moe;
    moe.runtime=reinterpret_cast<native_runtime_abi::DeviceMoELayerRuntimeView*>(r.ffn.runtime);
    moe.active_rows=r.ffn.rows.countOwner();
    moe.public_ids=r.router_indices;moe.public_weights=r.router_weights_output;
    if(p.selected_descriptor_demand==MoESelectedDescriptorDemand::ExternalTables)
        moe.publication=r.router_publication;
    moe.ffr=r.ffn.routed_columns;moe.nused=r.ffn.top_k;moe.clamp=1e-20f;
    moe.out=r.ffn.combined_output;moe.on=1;moe.dedup=1;

    qfn_castagna_native::CanonicalPortObservation outputs{};
    outputs.hc_down=hc.raw_down;
    if(p.output_demand.contains(FullNativeHCFFNOutputRole::HCActivated))outputs.hc_activated=hc.publication.activated;
    if(p.output_demand.contains(FullNativeHCFFNOutputRole::HCUp))outputs.hc_up=hc.up.output;
    if(p.output_demand.contains(FullNativeHCFFNOutputRole::RouterProbabilities))outputs.router_probabilities=r.router_logits;
    outputs.route_contributions=r.ffn.route_contributions;
    outputs.routed_output=r.ffn.routed_output;outputs.shared_output=r.ffn.shared_output;
    outputs.shared_gate=r.ffn.shared_gate_output;outputs.shared_up=r.ffn.shared_up_output;
    const qfn_castagna_native::InstanceScratchBinding binding{scratch,retainedM1HCFFNScratchBytes(),generation};
    const hc_ar ar{};
    const auto launch=[&]<bool Observe,bool PublishRoleTables>() {
        hc_persist_kernel<1,true,false,false,false,qfn_castagna_reference::GateSource::Q4_K,Observe,true,true,true,true,PublishRoleTables>
           <<<NB,NT,0,static_cast<hipStream_t>(stream)>>>(source,hc.norm.gain,hc.down_weights,hc.up.weights,
            hc.norm.epsilon,.25f,.25f,p.write.output,hc.norm.output,hc.mixed,ffn,ar,outputs,binding);
    };
    const auto issue=[&]<bool PublishRoleTables>() {
        if(p.output_demand.bits())launch.template operator()<true,PublishRoleTables>();
        else launch.template operator()<false,PublishRoleTables>();
    };
    switch(p.selected_descriptor_demand) {
    case MoESelectedDescriptorDemand::ExternalTables: issue.template operator()<true>(); break;
    case MoESelectedDescriptorDemand::AcquiredBankOnly: issue.template operator()<false>(); break;
    default: throw std::logic_error("Retained M1 HC/FFN lost its sealed descriptor demand");
    }
    const auto status=hipGetLastError();
    if(status!=hipSuccess)
        throw std::runtime_error(std::string("Retained M1 HC/FFN native launch failed: ")+hipGetErrorString(status));
}
}
