/**
 * @file CPUFullExpertReferenceContract.inc
 * @brief Independent whole-expert versus original native fragment arithmetic observations.
 *
 * Two logical experts retain original IDs without allocating512 experts. All
 * engines execute on the actual local CPU with one PMA; cohort addresses are
 * synthetic arithmetic metadata, not a multi-socket deployment claim. Scalar
 * FP64 source-dequant math is an observation oracle, never an execution path.
 */
#include "utils/CPUProjectionTestWorkspace.h"
#include "kernels/cpu/gemm/CPUWeightPreparationMemory.h"
#include "kernels/cpu/gemm/CPUNativeVNNIGemv.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
#include <iostream>
#include <iomanip>
#include <sstream>
namespace llaminar2::test {
namespace {
/** @brief Separate absolute error and reference-normalized L2 without a fitted tolerance. */
struct FullExpertError { double maximum=0,relative_l2=0,reference_l2=0; };
/**
 * @brief Measure an observed component without choosing an acceptance threshold.
 * @tparam T Independent reference scalar precision.
 * @param actual Actual production output prefix.
 * @param expected Matched reference prefix with identical coordinates.
 * @return Absolute maximum, normalized L2 and reference L2; a zero reference uses absolute L2.
 * @throws std::invalid_argument For unequal extents.
 * @throws std::runtime_error For nonfinite observations.
 */
template<class T> FullExpertError fullExpertError(std::span<const float> actual,std::span<const T> expected) {
    if(actual.size()!=expected.size())throw std::invalid_argument("Full expert observation extent mismatch");
    double error=0,reference=0,maximum=0;
    for(size_t i=0;i<actual.size();++i) {
        if(!std::isfinite(actual[i]) || !std::isfinite(expected[i]))throw std::runtime_error("Nonfinite full expert observation");
        const double delta=double(actual[i])-double(expected[i]);error+=delta*delta;reference+=double(expected[i])*double(expected[i]);
        maximum=std::max(maximum,std::abs(delta));
    }
    return {maximum,reference?std::sqrt(error/reference):std::sqrt(error),std::sqrt(reference)};
}
/**
 * @brief Publish actual component observations, not a model-noise acceptance bound.
 * @param label Exact native source-format case.
 * @param rows Actual tested row count.
 * @param path Component comparison, retaining full/fragment/oracle identity.
 * @param error Measured finite output distance.
 */
void recordFullExpertError(const std::string& label,int rows,const std::string& path,const FullExpertError& error) {
    std::cout<<std::setprecision(17)<<"CPU_FULL_EXPERT_REFERENCE {\"case\":\""<<label<<"\",\"rows\":"<<rows
        <<",\"path\":\""<<path<<"\",\"max_abs\":"<<error.maximum<<",\"relative_l2\":"<<error.relative_l2
        <<",\"reference_l2\":"<<error.reference_l2<<"}\n";
}
/**
 * @brief Evaluate independent dequantized mathematical FFN without prepared-engine reuse.
 * @param weights Original decoded gate/up/down matrices, never shard partials.
 * @param input Matched original hidden rows.
 * @param rows Actual row count.
 * @param H Source hidden width.
 * @param F Complete original intermediate width.
 * @return FP64 output with source-dequantized weights and an independent SiLU expression.
 */
std::vector<double> fullExpertScalarOracle(const std::array<std::vector<float>,3>& weights,
    std::span<const float> input,int rows,int H,int F) {
    std::vector<double> result(size_t(rows)*H),activated(F);
    for(int row=0;row<rows;++row) {
        for(int n=0;n<F;++n) {
            double gate=0,up=0;
            for(int k=0;k<H;++k) {
                gate=std::fma(double(weights[0][size_t(n)*H+k]),double(input[size_t(row)*H+k]),gate);
                up=std::fma(double(weights[1][size_t(n)*H+k]),double(input[size_t(row)*H+k]),up);
            }
            activated[n]=(gate/(1.0+std::exp(-gate)))*up;
        }
        for(int n=0;n<H;++n) {
            double sum=0;for(int k=0;k<F;++k)sum=std::fma(double(weights[2][size_t(n)*F+k]),activated[k],sum);
            result[size_t(row)*H+n]=sum;
        }
    }
    return result;
}
/**
 * @brief Compare independent whole and fragment engines under original format policies.
 * @param label Exact source-format case recorded with each observation.
 * @param creators Existing canonical source creators for gate/up/down.
 * @param formats Source identities used by the original cohort and exact prepared-weight BOM.
 * @note One actual local CPU/PMA owns all physical engine copies; the two-member
 * cohort supplies arithmetic geometry only. Full and fragment workspaces execute
 * serially. Diagnostic TP/FP64 distances do not define model accuracy bounds.
 */
void observeFullExpertReference(const std::string& label,
    const std::array<QuantizedVerifierWeightCreator,3>& creators,const std::array<ExpertWeightFormat,3>& formats) {
    constexpr int H=2560,F=640,P=2;
    MoEOverlayLayerWeightManifest manifest{3,{{{ExpertTierWeightProjection::Gate,F,H,formats[0]},
        {ExpertTierWeightProjection::Up,F,H,formats[1]},{ExpertTierWeightProjection::Down,H,F,formats[2]}}}};
    const auto cohort=MoEExpertShardCohort::checked(manifest,E,
        {{4,0,GlobalDeviceAddress::cpu(0)},{9,1,GlobalDeviceAddress::cpu(1)}},MoEExpertShardCohort::balancedIntervals(manifest,P));
    size_t admitted=0,staging=0;
    // Actual immediate preparations are synchronous and serialized. Contribute
    // the maximum canonical append workspace, separately from retained engines.
    const auto observe_staging=[&](size_t rows,size_t columns,const ExpertWeightFormat& format) {
        if(!format.isNativeVnni())return;
        const auto* native=native_vnni_formats::forSourceIdentity(format.native_vnni.codebook_id,format.native_vnni.is_superblock);
        if(!native)throw std::invalid_argument("Full expert fixture has no canonical native source format");
        for(const auto& entry:native_vnni_formats::kAllSourceFormats)if(entry.metadata==native) {
            staging=std::max(staging,CPUWeightPreparationMemory::sourceNative(entry.quant_type,rows,columns).temporary_bytes);return;
        }
        throw std::logic_error("Full expert fixture canonical native format is unnamed");
    };
    for(int role=0;role<3;++role) {
        admitted+=2*resolveExpertPreparedProjectionMemoryGeometry(role==2?H:F,role==2?F:H,formats[role]).cpu_bytes;
        observe_staging(role==2?H:F,role==2?F:H,formats[role]);
        const auto key=role==0?WeightRole::MoEExpertGate:role==1?WeightRole::MoEExpertUp:WeightRole::MoEExpertDown;
        for(int p=0;p<P;++p) {const auto part=cohort.ownership(p).projection(key);
            admitted+=2*resolveExpertPreparedProjectionMemoryGeometry(part.rows,part.columns,formats[role]).cpu_bytes;
            observe_staging(part.rows,part.columns,formats[role]);}
    }
    PhysicalMemoryPlanBuilder bill;
    const PhysicalMemoryResource resource{0,DeviceId::cpu(),admitted+staging,admitted+staging};
    bill.add(resource,PhysicalMemoryOwner::RoutedExpertWeights,admitted);
    bill.add(resource,PhysicalMemoryOwner::WeightLoadStaging,staging);
    auto memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(bill.build()),0);
    const auto placement=CPUWeightStoragePlacement::accounted(memory,DeviceId::cpu(),PhysicalMemoryOwner::RoutedExpertWeights,CPUWeightStoragePlacement::local());
    {
    Engines full;std::array<Engines,P> fragments;
    std::array<std::array<std::shared_ptr<TensorBase>,3>,2> sources;
    std::array<std::array<std::vector<float>,3>,2> dequant;
    const auto assign=[](PreparedFragmentEngines& engines,int role,std::shared_ptr<ITensorGemm> engine) {
        if(!engine)throw std::runtime_error("Full expert reference preparation failed");
        if(role==0)engines.gate=std::move(engine);else if(role==1)engines.up=std::move(engine);else engines.down=std::move(engine);
    };
    for(int expert=0;expert<2;++expert)for(int role=0;role<3;++role) {
        sources[expert][role]=creators[role](role==2?std::vector<size_t>{H,F}:std::vector<size_t>{F,H},uint32_t(71+expert*11+role));
        const auto& source=sources[expert][role];dequant[expert][role].resize(size_t(H)*F);source->to_fp32(dequant[expert][role].data());
        assign(full[expert],role,KernelFactory::prepareExpertGemmLocal(source,DeviceId::cpu(),KernelFactory::GemmPreparationKind::AUTO,placement));
        const auto key=role==0?WeightRole::MoEExpertGate:role==1?WeightRole::MoEExpertUp:WeightRole::MoEExpertDown;
        for(int p=0;p<P;++p) {
            const auto rectangle=cohort.ownership(p).projection(key);auto compact=nativeFragment(*source,rectangle);
            std::vector<float> decoded(size_t(rectangle.rows)*rectangle.columns);compact->to_fp32(decoded.data());
            for(int n=0;n<rectangle.rows;++n)
                EXPECT_EQ(std::memcmp(decoded.data()+size_t(n)*rectangle.columns,
                    dequant[expert][role].data()+size_t(n+rectangle.first_row)*rectangle.source_columns+rectangle.first_column,
                    size_t(rectangle.columns)*sizeof(float)),0)<<label<<" role="<<role<<" member="<<p;
            assign(fragments[p][expert],role,KernelFactory::prepareExpertGemmLocal(compact,DeviceId::cpu(),KernelFactory::GemmPreparationKind::AUTO,placement));
        }
    }
    EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RoutedExpertWeights,PhysicalMemoryMaterializationKind::NewAllocation),admitted);
    EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,PhysicalMemoryMaterializationKind::NewAllocation),0u);
    std::vector<float> hidden(3*H);for(size_t i=0;i<hidden.size();++i)hidden[i]=std::sin(float(i)*.013f)*.1f;
    std::vector<float> serialFull;std::array<std::vector<float>,P> serialFragments;
    for(int rows:{1,3}) {
        const auto input=std::span<const float>(hidden).first(size_t(rows)*H);
        const auto fullRows=preparedRouteRows(full,F,input,rows,H);if(rows==1)serialFull=fullRows;
        std::array<std::vector<float>,P> partial;
        for(int p=0;p<P;++p) {partial[p]=preparedRouteRows(fragments[p],F/P,input,rows,H);if(rows==1)serialFragments[p]=partial[p];}
        std::array<CanonicalRouteDownPartialRows,P> banks{{{partial[0]},{partial[1]}}};
        std::vector<float> combined(fullRows.size(),-991),weighted(size_t(rows)*H),fullWeighted(weighted.size()),weights(size_t(rows)*K);
        for(size_t slot=0;slot<weights.size();++slot)weights[slot]=slot%2?.75f:.25f;
        ASSERT_TRUE(cpu::moe::reduceCanonicalRouteDownPartialsOrderedFMA({banks,combined,weights.data(),weighted.data(),rows,K,H}));
        ASSERT_TRUE(cpu::moe::reduceCanonicalRouteRowsOrderedFMA({fullRows.data(),weights.data(),nullptr,fullWeighted.data(),rows,K,H}));
        for(int row=0;row<rows;++row)for(int h=0;h<H;++h) {
            const float once=std::fma(.75f,fullRows[(size_t(row)*K+1)*H+h],std::fma(.25f,fullRows[size_t(row)*K*H+h],0.0f));
            EXPECT_EQ(std::memcmp(&once,&fullWeighted[size_t(row)*H+h],sizeof(float)),0);
        }
        recordFullExpertError(label,rows,"full_vs_fragments_routes",fullExpertError<float>(combined,fullRows));
        recordFullExpertError(label,rows,"full_vs_fragments_one_weight",fullExpertError<float>(weighted,fullWeighted));
        std::array<std::vector<double>,2> oracle{{fullExpertScalarOracle(dequant[0],input,rows,H,F),fullExpertScalarOracle(dequant[1],input,rows,H,F)}};
        std::vector<double> oracleWeighted(weighted.size());
        for(int row=0;row<rows;++row)for(int h=0;h<H;++h)
            oracleWeighted[size_t(row)*H+h]=std::fma(.75,oracle[1][size_t(row)*H+h],.25*oracle[0][size_t(row)*H+h]);
        recordFullExpertError(label,rows,"source_fp64_vs_full",fullExpertError<double>(fullWeighted,oracleWeighted));
        recordFullExpertError(label,rows,"source_fp64_vs_fragments",fullExpertError<double>(weighted,oracleWeighted));
    }
    // Same-input projection diagnostics distinguish earlier gate/up changes
    // from down-GEMM partition reassociation. No alternate stage backend exists.
    for(int expert=0;expert<2;++expert) {
        const auto& w=full[expert];const auto& a=fragments[0][expert];const auto& b=fragments[1][expert];
        CPUProjectionTestWorkspace workspace(1,H,cpuProjectionTestRequirements(1,
            {w.gate.get(),w.up.get(),w.down.get(),a.gate.get(),a.up.get(),a.down.get(),b.gate.get(),b.up.get(),b.down.get()}));
        FP32Tensor input({1,H}),gate({1,F}),up({1,F}),down({1,H});std::copy_n(hidden.data(),H,input.mutable_data());
        ASSERT_TRUE(w.gate->multiply_tensor(&input,&gate,1,F,H,true,1,0,nullptr,nullptr,-1,workspace.get()));
        ASSERT_TRUE(w.up->multiply_tensor(&input,&up,1,F,H,true,1,0,nullptr,nullptr,-1,workspace.get()));
        ASSERT_TRUE(w.down->multiply_tensor_with_fused_swiglu(&gate,&up,&down,1,H,F,1,0,workspace.get()));
        std::vector<float> fragmentGate(F),fragmentUp(F);std::array<std::vector<float>,P> matchedDown;
        AlignedVector<Q8_1Block> wholeQ8(F/32);cpu::native_vnni::swiglu_quantize_activations_to_q8_1(
            gate.data(),up.data(),wholeQ8.data(),1,F,F/32,CPUProjectionNumericalPolicy::GPUAlignedExpert);
        for(int p=0;p<P;++p) {
            const auto& engines=fragments[p][expert];FP32Tensor g({1,F/P}),u({1,F/P}),matchedG({1,F/P}),matchedU({1,F/P}),d({1,H});
            ASSERT_TRUE(engines.gate->multiply_tensor(&input,&g,1,F/P,H,true,1,0,nullptr,nullptr,-1,workspace.get()));
            ASSERT_TRUE(engines.up->multiply_tensor(&input,&u,1,F/P,H,true,1,0,nullptr,nullptr,-1,workspace.get()));
            std::copy_n(g.data(),F/P,fragmentGate.data()+p*(F/P));std::copy_n(u.data(),F/P,fragmentUp.data()+p*(F/P));
            std::copy_n(gate.data()+p*(F/P),F/P,matchedG.mutable_data());std::copy_n(up.data()+p*(F/P),F/P,matchedU.mutable_data());
            AlignedVector<Q8_1Block> localQ8(F/P/32);cpu::native_vnni::swiglu_quantize_activations_to_q8_1(
                matchedG.data(),matchedU.data(),localQ8.data(),1,F/P,F/P/32,CPUProjectionNumericalPolicy::GPUAlignedExpert);
            EXPECT_EQ(std::memcmp(localQ8.data(),wholeQ8.data()+p*(F/P/32),localQ8.size()*sizeof(Q8_1Block)),0)<<label;
            ASSERT_TRUE(engines.down->multiply_tensor_with_fused_swiglu(&matchedG,&matchedU,&d,1,H,F/P,1,0,workspace.get()));
            matchedDown[p].assign(d.data(),d.data()+H);
            EXPECT_EQ(std::memcmp(d.data(),serialFragments[p].data()+size_t(expert)*H,H*sizeof(float)),0)
                <<label<<" fragment direct projection must match its actual stage";
        }
        EXPECT_EQ(std::memcmp(fragmentGate.data(),gate.data(),F*sizeof(float)),0)<<label;
        EXPECT_EQ(std::memcmp(fragmentUp.data(),up.data(),F*sizeof(float)),0)<<label;
        EXPECT_EQ(std::memcmp(down.data(),serialFull.data()+size_t(expert)*H,H*sizeof(float)),0)<<label;
        const auto prefix="expert"+std::to_string(expert)+"_";
        recordFullExpertError(label,1,prefix+"gate_full_vs_fragments",fullExpertError<float>(fragmentGate,{gate.data(),size_t(F)}));
        recordFullExpertError(label,1,prefix+"up_full_vs_fragments",fullExpertError<float>(fragmentUp,{up.data(),size_t(F)}));
        std::array<CanonicalRouteDownPartialRows,P> banks{{{matchedDown[0]},{matchedDown[1]}}};
        std::vector<float> combined(H);const float required=1;
        ASSERT_TRUE(cpu::moe::combineCanonicalRouteDownPartialsOrdered({banks,combined,&required,1,1,H}));
        recordFullExpertError(label,1,prefix+"matched_down_full_vs_fragments",fullExpertError<float>(combined,{down.data(),size_t(H)}));
        recordFullExpertError(label,1,prefix+"direct_full_vs_stage",fullExpertError<float>({down.data(),size_t(H)},std::span<const float>(serialFull).subspan(size_t(expert)*H,H)));
    }
    }
    EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RoutedExpertWeights,PhysicalMemoryMaterializationKind::NewAllocation),0u);
}
}
/** @test Matched whole/native320 sources expose error components without fitting a model tolerance. */
TEST(CPUExpertShardExecution, IndependentFullExpertVersusNative320FragmentContract) {
    recordReplayScope();::testing::Test::RecordProperty("arithmetic_population_synthetic","true");
    if(!hasCPUBackend())initCPUBackend(-1);TeamScope team(1);
    const QuantizedVerifierWeightCreator floating=[](const auto& shape,uint32_t seed)->std::unique_ptr<TensorBase> {
        return TestTensorFactory::createFP32Random(shape,-.02f,.02f,seed);
    };
    const auto fp=ExpertWeightFormat::floating(TensorType::FP32);observeFullExpertReference("FP32_control",{floating,floating,floating},{fp,fp,fp});
    const auto& registry=quantizedMoEVerifierFormats();
    const auto find=[&](TensorType type)->const QuantizedVerifierFormatCase& {
        const auto found=std::find_if(registry.begin(),registry.end(),[&](const auto& value){return value.tensor_type==type;});
        if(found==registry.end())throw std::logic_error("Native full expert observation source absent");return *found;
    };
    for(const auto types:{std::pair{TensorType::Q4_K,TensorType::Q5_1},std::pair{TensorType::Q4_K,TensorType::Q8_0},std::pair{TensorType::Q5_K,TensorType::Q8_0}}) {
        const auto& gate=find(types.first);const auto& down=find(types.second);
        const auto g=ExpertWeightFormat::nativeVnni({gate.source_codebook_id,gate.source_is_superblock,true});
        const auto d=ExpertWeightFormat::nativeVnni({down.source_codebook_id,down.source_is_superblock,true});
        observeFullExpertReference(std::string(tensorTypeName(types.first))+"_"+tensorTypeName(types.second),{gate.create,gate.create,down.create},{g,g,d});
    }
}
} // namespace llaminar2::test
