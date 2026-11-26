/**
 * @file ROCmPersistentHCRead.hip
 * @brief Standalone PersistentHCRead and CTA-owned HC read kernels and admission.
 */
#define LLAMINAR_DEFINE_PERSISTENT_HC_READ_KERNEL 1
#include "ROCmNativeQuantisedMoEDeviceHelpers.hip.inc"
#include "ROCmPersistentHCReadBody.hip.inc"
#include "ROCmNativeLayoutRouterFFNBody.hip.inc"
#include "ROCmFullNativeHCFFNBody.hip.inc"

LLAMINAR_DEFINE_ROCM_IQ_GRID_INIT(persistent_hc)


namespace llaminar2::rocm {
/** @brief Validate the borrowed original physical phase bindings without reading live GPU state. */
void validatePersistentHCRead(const PersistentHCReadArgs& a,bool ctaOwned,
 FullNativeHCFFNOutputDemand output_demand) {
 const int H=a.norm.hidden,R=a.down_weights.n;
 if(a.norm.extent.rows!=1||H<=0||(ctaOwned?H%8:H%32)||static_cast<std::int64_t>(H)*4!=a.down_weights.k||
    R<64||R>512||R%32||!std::isfinite(a.norm.epsilon)||a.norm.epsilon<=0.0f||
    !a.norm.input||!a.norm.gain||!a.norm.output||!a.input_codes||!a.input_scales||
    !a.down_partials||!a.raw_down||!a.mixed||
    (output_demand.contains(FullNativeHCFFNOutputRole::HCActivated)&&!a.publication.activated)||
    a.down_weights.codebook_id!=19||a.up.weights.codebook_id!=19||
    !a.down_weights.source_identity_present||!a.up.weights.source_identity_present||
    a.publication.columns!=R||a.up.weights.k!=R||a.up.weights.n!=4*H||
    a.up.rows.physicalRows()!=1||a.publication.rows.physicalRows()!=1||
    a.up.rows.captureIdentity()!=a.publication.rows.captureIdentity()||
    a.up.primary!=a.publication.primary||a.up.residual!=a.publication.residual||
    a.up.primaryScales!=a.publication.primaryScales||a.up.residualScales!=a.publication.residualScales||
    a.up.quantizer!=Q8ResidualQuantizerPolicy::ROCmFiniteHalfNativeReciprocal||
    (output_demand.contains(FullNativeHCFFNOutputRole::HCUp)&&!a.up.output)||
    a.publication.primary!=a.input_codes||a.publication.primaryScales!=a.input_scales||
    (a.injection.weights==nullptr)!=(a.injection.output==nullptr)||
    (a.injection.weights&&(a.injection.rows.physicalRows()!=1||
      (a.injection.load_pairs!=1&&a.injection.load_pairs!=4&&a.injection.load_pairs!=8))))
  throw std::invalid_argument("Persistent HC requires exact original M1 branch/native19/publication bindings");
 (void)requireQ8ResidualNativeMatrix(a.down_weights,1,R,4*H);
 (void)requireQ8ResidualNativeMatrix(a.up.weights,1,4*H,R);
 const std::size_t K=static_cast<std::size_t>(4)*H;
 // Each phase is checked separately: the original full input code/scale bank
 // deliberately becomes the shorter two-term primary prefix after down retires.
 const std::array normWrites{Q8ResidualStorageSpan{a.norm.output,K*sizeof(float),alignof(float)}};
 const std::array normReads{Q8ResidualStorageSpan{a.norm.gain,K*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.norm.input,a.norm.input==a.norm.output?0:K*sizeof(float),alignof(float)}};
 validateQ8ResidualStorage(normWrites,normReads);
 const std::array downWrites{Q8ResidualStorageSpan{a.input_codes,K,16},
  Q8ResidualStorageSpan{a.input_scales,K/32*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.down_partials,static_cast<std::size_t>(a.partitions)*R*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.injection.output,a.injection.output?4*sizeof(float):0,alignof(float)}};
 const std::array downReads{Q8ResidualStorageSpan{a.norm.output,K*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.norm.input,a.norm.input==a.norm.output?0:K*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.norm.gain,K*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.down_weights.payload,K/32*R*32,16},
  Q8ResidualStorageSpan{a.down_weights.scales,K/32*R*sizeof(std::uint16_t),alignof(std::uint16_t)},
  Q8ResidualStorageSpan{a.injection.weights,a.injection.weights?4*K*sizeof(float):0,alignof(float)}};
 validateQ8ResidualStorage(downWrites,downReads);
 const std::array endWrites{Q8ResidualStorageSpan{a.raw_down,static_cast<std::size_t>(R)*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.publication.activated,
   output_demand.contains(FullNativeHCFFNOutputRole::HCActivated)&&a.publication.activated!=a.raw_down?
    static_cast<std::size_t>(R)*sizeof(float):0,alignof(float)},
  Q8ResidualStorageSpan{a.publication.primary,static_cast<std::size_t>(R),16},
  Q8ResidualStorageSpan{a.publication.residual,static_cast<std::size_t>(R),16},
  Q8ResidualStorageSpan{a.publication.primaryScales,static_cast<std::size_t>(R)/32*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.publication.residualScales,static_cast<std::size_t>(R)/32*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.up.output,output_demand.contains(FullNativeHCFFNOutputRole::HCUp)?K*sizeof(float):0,alignof(float)},
  Q8ResidualStorageSpan{a.mixed,static_cast<std::size_t>(H)*sizeof(float),alignof(float)}};
 const std::array endReads{Q8ResidualStorageSpan{a.norm.output,K*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.norm.input,a.norm.input==a.norm.output?0:K*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.norm.gain,K*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.down_partials,static_cast<std::size_t>(a.partitions)*R*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.down_weights.payload,K/32*R*32,16},
  Q8ResidualStorageSpan{a.down_weights.scales,K/32*R*sizeof(std::uint16_t),alignof(std::uint16_t)},
  Q8ResidualStorageSpan{a.up.weights.payload,K/32*R*32,16},
  Q8ResidualStorageSpan{a.up.weights.scales,K/32*R*sizeof(std::uint16_t),alignof(std::uint16_t)},
  Q8ResidualStorageSpan{a.injection.output,a.injection.output?4*sizeof(float):0,alignof(float)}};
 validateQ8ResidualStorage(endWrites,endReads);
}
/** @copydoc resolvePersistentHCReadPolicy */
void resolvePersistentHCReadPolicy(PersistentHCReadArgs& args,FullNativeHCFFNOutputDemand output_demand) {
 if(isGraphCaptureActive())
  throw std::invalid_argument("Persistent HC source-policy resolution requires cold preparation");
 const auto& env=debugEnv().rocm;
 if(llaminar2::rocm::native_quantised_internal::nativeQ8SerialScopeActive()||llaminar2::rocm::native_quantised_internal::kbOverride()>0||
    llaminar2::rocm::native_quantised_internal::targetWavesOverride()>0||
    env.native_quantised_gemv_kb>0||env.native_quantised_gemv_target_waves>0)
  throw std::invalid_argument("Persistent HC cannot replace an explicitly overridden original arithmetic policy");
 if(args.down_weights.codebook_id!=19||!args.down_weights.source_identity_present)
  throw std::invalid_argument("Persistent HC lost original signed-Q8 source policy");
 args.partitions=llaminar2::rocm::native_quantised_internal::resolveSerialM1KPartitionsForPolicy(
  canonicalDeviceVnniCodebookId(args.down_weights.source_codebook_id),args.down_weights.n,args.down_weights.k);
 if(args.partitions<=0)throw std::invalid_argument("Persistent HC original split-K source policy is absent");
 validatePersistentHCRead(args,false,output_demand);
}
} // namespace llaminar2::rocm

#include "ROCmCTAOwnedHCReadBody.hip.inc"
namespace llaminar2::rocm {
/** @copydoc prepareCTAOwnedHCRead */
CTAOwnedHCReadProgram prepareCTAOwnedHCRead(DeviceId device,PersistentHCReadArgs& args,std::uint64_t generation) {
 if(!device.is_rocm()||!generation||isGraphCaptureActive())
  throw std::invalid_argument("CTA-owned HC requires its exact original native owner cold");
 int currentDevice=-1;
 if(hipGetDevice(&currentDevice)!=hipSuccess||currentDevice!=device.ordinal)
  throw std::invalid_argument("CTA-owned HC cold resource admission has a foreign current native device");
 const auto& env=debugEnv().rocm;
 if(llaminar2::rocm::native_quantised_internal::nativeQ8SerialScopeActive()||llaminar2::rocm::native_quantised_internal::kbOverride()>0||
    llaminar2::rocm::native_quantised_internal::targetWavesOverride()>0||
    env.native_quantised_gemv_kb>0||env.native_quantised_gemv_target_waves>0)
  throw std::invalid_argument("CTA-owned HC cannot replace an overridden original arithmetic policy");
 args.partitions=llaminar2::rocm::native_quantised_internal::resolveSerialM1KPartitionsForPolicy(
  canonicalDeviceVnniCodebookId(args.down_weights.source_codebook_id),args.down_weights.n,args.down_weights.k);
 requireCTAOwnedHCTailGeometry(args.norm.hidden,args.partitions,args.raw_down==args.publication.activated);
 // The complete storage validator is shared with the original implementation;
 // its H%32 restriction belongs to that cooperative program, not this tail.
 validatePersistentHCRead(args,true);
 CTAOwnedHCReadProgram result;result.device_=device;result.generation_=generation;
 result.hidden_=args.norm.hidden;result.rank_=args.down_weights.n;result.partitions_=args.partitions;
 const auto select=[&]<unsigned G>() {return llaminar2::rocm::native_quantised_internal::inspectNativeQ8SerialSymbol(
  reinterpret_cast<const void*>(ctaOwnedHCTailKernel<G>),"ctaOwnedHCTailKernel",1024);};
 switch(args.down_weights.n/32) {
 case 2:result.resources_=select.template operator()<2>();break;
 case 3:result.resources_=select.template operator()<3>();break;
 case 4:result.resources_=select.template operator()<4>();break;
 case 5:result.resources_=select.template operator()<5>();break;
 case 6:result.resources_=select.template operator()<6>();break;
 case 7:result.resources_=select.template operator()<7>();break;
 case 8:result.resources_=select.template operator()<8>();break;
 case 9:result.resources_=select.template operator()<9>();break;
 case 10:result.resources_=select.template operator()<10>();break;
 case 11:result.resources_=select.template operator()<11>();break;
 case 12:result.resources_=select.template operator()<12>();break;
 case 13:result.resources_=select.template operator()<13>();break;
 case 14:result.resources_=select.template operator()<14>();break;
 case 15:result.resources_=select.template operator()<15>();break;
 case 16:result.resources_=select.template operator()<16>();break;

 default:throw std::invalid_argument("CTA-owned HC has no complete original short-K group specialization");
 }
 return result;
}
/** @copydoc launchCTAOwnedHCRead */
void launchCTAOwnedHCRead(const CTAOwnedHCReadProgram& program,const PersistentHCReadArgs& args,
 void* stream,std::uint64_t generation) {
 if(reinterpret_cast<std::uintptr_t>(stream)<=2||program.generation_!=generation||!generation||
    !program.device_.is_rocm()||program.hidden_!=args.norm.hidden||program.rank_!=args.down_weights.n||
    program.partitions_!=args.partitions||!program.resources_.function)
  throw std::invalid_argument("CTA-owned HC exact prepared program/workspace geometry changed");
 if(!rocmQuantGemm_quantizeActivationsBlockwise(args.norm.output,args.input_codes,args.input_scales,
    1,args.down_weights.k,program.device_.ordinal,stream,32))
  throw std::runtime_error("CTA-owned HC original SingleQ8 producer failed");
 NativeM1ReductionTicket ticket;
 if(!enqueueNativeM1Producer(ticket,args.input_codes,args.down_weights.payload,args.down_weights.scales,
    args.down_weights.mins,args.down_weights.emins,args.raw_down,nullptr,args.down_partials,
    args.down_weights.n,args.down_weights.k,args.down_weights.codebook_id,
    canonicalDeviceVnniCodebookId(args.down_weights.source_codebook_id),program.device_.ordinal,stream,args.input_scales))
  throw std::runtime_error("CTA-owned HC original down producer failed");
 CTAOwnedHCTicketConsumer::submit(ticket,program,args,program.device_.ordinal,stream);
}
}


namespace llaminar2::rocm {
/** @copydoc preparePersistentHCRead */
NativeCooperativeKernelAdmission preparePersistentHCRead(DeviceId device,PersistentHCReadArgs& args,
 std::uint64_t generation) {
 if(!device.is_rocm()||!generation||isGraphCaptureActive())
  throw std::invalid_argument("Persistent HC admission requires its original native owner cold");
 resolvePersistentHCReadPolicy(args);
 FullNativeHCFFNPrefixArgs prefix{args,{},nullptr,FullNativeHCFFNOutputDemand::all()};
 const auto dynamic_bytes=fullNativeHCFFNPrefixDynamicBytes(prefix,1);
 if(!dynamic_bytes||dynamic_bytes>32768||(dynamic_bytes&15u))
  throw std::invalid_argument("Persistent HC has no valid 1024-thread prefix dynamic LDS geometry");
 const auto admission=NativeCooperativeKernelBoundary<NativeCooperativeBackend::ROCm>::prepare(device,
  reinterpret_cast<const void*>(persistentHCReadKernel1024),1024,dynamic_bytes,generation,
  NativeResidentLaunchPolicy::Ordinary);
 using Geometry=PersistentHCReadGeometry<1024>;
 const std::uint64_t down=(std::uint64_t(args.down_weights.n)+Geometry::down_columns-1)/Geometry::down_columns*
  static_cast<unsigned>(args.partitions)+(args.injection.weights?4u:0u);
 const std::uint64_t quant=(std::uint64_t(args.down_weights.k)/32+Geometry::quant_cohorts-1)/Geometry::quant_cohorts;
 const std::uint64_t publish=(std::uint64_t(args.down_weights.n)+Geometry::threads-1)/Geometry::threads;
 const std::uint64_t up=(std::uint64_t(args.norm.hidden)+Geometry::up_channels-1)/Geometry::up_channels;
 const std::uint64_t useful=std::max({std::uint64_t{4},down,quant,publish,up});
 unsigned bound=admission.maximumGrid();
 const unsigned cus=static_cast<unsigned>(rocmHardwareMetadata().at(device.ordinal).compute_units);
 const unsigned safe_cus=cus>4u?(cus-4u):cus;
 if(args.grid_policy==PersistentHCGridPolicy::OnePerProcessor)
  bound=std::min(bound,safe_cus);
 else if(args.grid_policy!=PersistentHCGridPolicy::MaximumResident)
  throw std::invalid_argument("Persistent HC grid policy is not declared");
 args.resident_grid=static_cast<unsigned>(std::min(useful,std::uint64_t(bound)));
 return admission;
}
/** @copydoc launchPersistentHCRead */
void launchPersistentHCRead(const NativeCooperativeKernelAdmission& admission,const PersistentHCReadArgs& args,
 std::uint32_t* barrier_state,void* stream,std::uint64_t generation,
 PersistentHCWriteInput previous_write) {
 if(admission.function()!=reinterpret_cast<const void*>(persistentHCReadKernel1024)||admission.threads()!=1024||
    admission.policy()!=NativeResidentLaunchPolicy::Ordinary||admission.generation()!=generation||
    !barrier_state||!args.resident_grid||args.resident_grid>admission.maximumGrid()||
    reinterpret_cast<std::uintptr_t>(stream)<=2)
  throw std::invalid_argument("Persistent HC exact prepared program/grid changed");
 FullNativeHCFFNPrefixArgs prefix{args,previous_write,barrier_state,FullNativeHCFFNOutputDemand::all()};
 unsigned prefix_bytes=static_cast<unsigned>(fullNativeHCFFNPrefixDynamicBytes(prefix,args.resident_grid));
 if(!prefix_bytes||prefix_bytes>admission.dynamicSharedBytes()||(prefix_bytes&15u))
  throw std::invalid_argument("Persistent HC dynamic LDS exceeds admitted reservation");
 void* parameters[]{&prefix,&prefix_bytes};
 const hipError_t status=hipLaunchKernel(admission.function(),dim3(args.resident_grid),dim3(1024),
  parameters,prefix_bytes,static_cast<hipStream_t>(stream));
 if(status!=hipSuccess)
  throw std::runtime_error(std::string("Persistent HC 1024-thread launch failed: ")+hipGetErrorString(status));
}
} // namespace llaminar2::rocm

#if !defined(__HIP_DEVICE_COMPILE__)
#include "ROCmRoutedGridDonation.h"

namespace llaminar2::rocm {
/** @copydoc inspectRoutedGridDonation */
RoutedGridDonation inspectRoutedGridDonation(std::uint32_t routed_formats, unsigned prefix_bytes,
    std::size_t hidden_bytes, std::size_t panel_and_records_bytes) noexcept {
    const auto after_hidden = prefix_bytes > hidden_bytes ? unsigned(prefix_bytes - hidden_bytes) : 0u;
    const auto available = after_hidden > panel_and_records_bytes ? after_hidden - panel_and_records_bytes : 0;
    const auto layout = routedGridLayout(routed_formats, available);
    return {layout.iq2_entries, layout.iq3_small_entries, layout.iq3_extra_small_entries};
}
}
#endif
