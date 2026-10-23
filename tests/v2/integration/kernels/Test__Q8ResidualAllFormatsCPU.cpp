/**
 * @file Test__Q8ResidualAllFormatsCPU.cpp
 * @brief Explicit CPU integration sweep of original-format TwoTerm rows,
 * threads and tails.
 *
 * Source weights and captured-stage policy remain production-owned. The M1 row
 * loop exists only as an independent diagnostic oracle. ISA aliases start new
 * processes because the canonical runtime ISA policy is immutable after first
 * use. Disabled performance/profile selectors are opt-in diagnostics; hardware
 * counters enclose one selected warmed chain and never enter correctness gates.
 */
#include "../../utils/Q8ResidualAllFormatCPUHarness.h"
#include "utils/MPIBootstrap.h"
#include "../../utils/HC103CPUControlChain.h"
#include "kernels/cpu/gemm/CPUNativeVNNIGemv.h"

/** @test All positive worker teams execute every format on a tiny complete
 * source. */
TEST(Q8ResidualAllFormatsCPU, NativeRowsThreadsAndLargeSmallReuse) {
  const int budget = std::max(
      1, llaminar2::MPIBootstrap::detectCPUTopology().cores_per_socket);
  std::vector<int> workers(budget);
  std::iota(workers.begin(), workers.end(), 1);
  std::vector<int> representative{1, std::min(budget, 3), budget};
  std::sort(representative.begin(), representative.end());
  representative.erase(
      std::unique(representative.begin(), representative.end()),
      representative.end());
  for (const auto &format : llaminar2::test::quantizedVerifierFormats()) {
    const std::array tinyLives{3, 0, 1, 3};
    llaminar2::test::q8_all_formats::proveCPUThreads(format, 3, 5, 256, workers,
                                                     false, tinyLives);
    ASSERT_FALSE(::testing::Test::HasFailure());
    for (int rows : {1, 9, 15, 33, 64, 65}) {
      const std::array single{1};
      const std::span<const int> teams =
          rows == 65 ? std::span<const int>(representative)
                     : std::span<const int>(single);
      llaminar2::test::q8_all_formats::proveCPUThreads(format, rows, 65, 512,
                                                       teams);
      ASSERT_FALSE(::testing::Test::HasFailure());
    }
  }
}
/** @test Large physical work uses bounded distinct-row oracles and retains
 * poisoned suffixes. */
TEST(Q8ResidualAllFormatsCPU, NativeLargeTiledRows) {
  const std::array team{1};
  for (const auto &format : llaminar2::test::quantizedVerifierFormats()) {
    for (int rows : {512, 1024, 2048}) {
      const std::array lives{rows, 1, 0, std::min(rows, 15), rows};
      llaminar2::test::q8_all_formats::proveCPUThreads(format, rows, 65, 256,
                                                       team, true, lives);
      ASSERT_FALSE(::testing::Test::HasFailure());
    }
    // A separate untiled (seven distinct authored rows) large bank prevents the
    // one-row tiling specialization from being mistaken for general M support.
    const std::array lives{2048, 512, 1024, 1, 0, 2048};
    llaminar2::test::q8_all_formats::proveCPUThreads(format, 2048, 5, 256, team,
                                                     false, lives);
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
}

/** @test Calibrate frozen native CPU HC arithmetic against actual prepared control stages. */
TEST(Q8ResidualAllFormatsCPU, HC103OriginalPreparedControlChain) {
 const int budget=std::max(1,llaminar2::MPIBootstrap::detectCPUTopology().cores_per_socket);
 std::vector<int> teams{1,std::min(3,budget),budget};
 std::sort(teams.begin(),teams.end());teams.erase(std::unique(teams.begin(),teams.end()),teams.end());
 const std::array one{1};
 for(const auto &format:llaminar2::test::quantizedVerifierFormats()) {
  for(bool alias:{false,true}) {
   llaminar2::test::hc103_cpu::prove(format,3,64,256,teams,alias);
   ASSERT_FALSE(::testing::Test::HasFailure());
   for(int rows:{1,9,33}) {
    llaminar2::test::hc103_cpu::prove(format,rows,64,256,one,alias);
    ASSERT_FALSE(::testing::Test::HasFailure());
   }
   // Original256-value source blocks cannot represent K320. The canonical
   // non-superblock families cover original HC rank320 and partial activation CTA.
   if(!format.source_is_superblock) {
    llaminar2::test::hc103_cpu::prove(format,9,72,320,one,alias);
    ASSERT_FALSE(::testing::Test::HasFailure());
   }
  }
 }
}
namespace {
/** @brief Calibrate only real CPU primitive activation/read, without NativeVNNI admission. */
void hc103OriginalPrimitiveTails() {
 using namespace llaminar2;
 using namespace llaminar2::test;
 q8_all_formats::Threads workers(1);
 auto primitive=qwen4exp::createCPUQwen4ExpKernels();
 for(int width:{1,7,8,15,16,17,31,32,33,255,256,257,320}) {
  std::vector<float> raw(std::size_t(3)*width),actual(raw.size()),expected(raw.size());
  for(std::size_t i=0;i<raw.size();++i)raw[i]=std::array{-0.f,0.f,1.e-38f,-1.e-38f,.00000001f,-.00000001f,3.9999998f,-4.0000005f}[i%8];
  for(int live:{3,1,0}) {
   const qwen4exp::RowExtent rows{3,&live};
   std::fill(actual.begin(),actual.end(),-12345.f);expected=actual;
   primitive->residualDown({rows,width,raw.data(),actual.data()},nullptr);
   hc103_original::residualDown({rows,width,raw.data(),expected.data()},nullptr);
   ASSERT_EQ(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)),0);
   actual=raw;expected=raw;
   primitive->residualDown({rows,width,actual.data(),actual.data()},nullptr);
   hc103_original::residualDown({rows,width,expected.data(),expected.data()},nullptr);
   ASSERT_EQ(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)),0);
  }
  std::vector<float> normalized(std::size_t(3)*4*width),logits(normalized.size()),read(std::size_t(3)*width),gold(read.size());
  for(std::size_t i=0;i<logits.size();++i){logits[i]=raw[i%raw.size()];normalized[i]=float(int(i%29)-14)*.03125f;}
  for(int live:{3,1,0}) {
   const qwen4exp::RowExtent rows{3,&live};std::fill(read.begin(),read.end(),-12345.f);gold=read;
   primitive->residualRead({rows,width,normalized.data(),logits.data(),read.data()},nullptr);
   hc103_original::residualRead({rows,width,normalized.data(),logits.data(),gold.data()},nullptr);
   ASSERT_EQ(std::memcmp(read.data(),gold.data(),read.size()*sizeof(float)),0);
  }
 }
}
} // namespace
/** @test Freeze ISA-specific channel tails and actual TwoTerm finite/half-overflow admission. */
TEST(Q8ResidualAllFormatsCPU, HC103OriginalPrimitiveTailsAndPublicationAdmission) {
 using namespace llaminar2;
 using namespace llaminar2::test;
 hc103OriginalPrimitiveTails();ASSERT_FALSE(::testing::Test::HasFailure());
 q8_all_formats::Threads workers(1);
 const auto &formats=quantizedVerifierFormats();
 const auto it=std::find_if(formats.begin(),formats.end(),[](const auto &f){return f.tensor_type==TensorType::Q8_0;});
 ASSERT_NE(it,formats.end());q8_all_formats::CPUFixture up(*it,3,256,256,false);
 CPUDeviceContext context(DeviceId::cpu(),1);std::vector<Q8_1Block> p(24),r(24);
 for(float invalid:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),1.e20f}) {
  std::fill_n(up.input.mutable_data(),up.input.numel(),0.f);up.input.mutable_data()[0]=invalid;
  std::fill_n(up.output.mutable_data(),up.output.numel(),-12345.f);
  EXPECT_THROW(up.stage->execute(&context),std::invalid_argument);
  EXPECT_THROW(hc103_original::publish(up.input.data(),3,256,p.data(),r.data()),std::invalid_argument);
  for(std::size_t i=0;i<up.output.numel();++i)ASSERT_EQ(up.output.data()[i],-12345.f);
 }
}

/** @test Explicit scalar primitive policy retains every original tail/live/alias byte. */
TEST(Q8ResidualAllFormatsCPU, DISABLED_HC103OriginalScalarPrimitives) {
 ASSERT_EQ(llaminar2::activeISALevel(),llaminar2::ISALevel::Scalar);
 hc103OriginalPrimitiveTails();
}
/** @test Scalar is deliberately outside original NativeVNNI prepared-workspace geometry. */
TEST(Q8ResidualAllFormatsCPU, DISABLED_HC103ScalarRejectsNativePreparedGeometry) {
 using namespace llaminar2;
 using namespace llaminar2::test;
 ASSERT_EQ(activeISALevel(),ISALevel::Scalar);
 EXPECT_EQ(CPUExecutionGeometry::local().maximum_native_row_tile,0u);
 EXPECT_FALSE(CPUExecutionGeometry::local().isValid());
 const auto &formats=quantizedVerifierFormats();
 const auto it=std::find_if(formats.begin(),formats.end(),[](const auto &f){return f.tensor_type==TensorType::Q8_0;});
 ASSERT_NE(it,formats.end());
 try {
  q8_all_formats::CPUFixture unsupported(*it,3,256,256,false);
  FAIL()<<"Scalar must not fabricate NativeVNNI workspace execution geometry";
 } catch(const std::invalid_argument &error) {
  EXPECT_EQ(std::string(error.what()),"CPU projection workspace requires participant-owned worker/cache/ISA geometry");
 }
}

/** @test Actual fused stage/direct typed arithmetic retains original CPU ISA publication. */
TEST(Q8ResidualAllFormatsCPU, HC103FusedStageAndTypedOverload) {
 using namespace llaminar2;
 using namespace llaminar2::test;
 const int budget=std::max(1,MPIBootstrap::detectCPUTopology().cores_per_socket);
 std::vector<int> teams{1,std::min(3,budget),budget};
 std::sort(teams.begin(),teams.end());teams.erase(std::unique(teams.begin(),teams.end()),teams.end());
 const std::array one{1};
 for(const auto &format:quantizedVerifierFormats())for(bool alias:{false,true})
  for(auto operation:{hc103_cpu::ChainOperation::FusedStage,hc103_cpu::ChainOperation::TypedOverload}) {
   hc103_cpu::prove(format,3,64,256,teams,alias,operation);
   ASSERT_FALSE(::testing::Test::HasFailure());
   // Two complete publication tiles must retain disjoint ownership even when
   // the activation aliases raw down output and live rows change between calls.
   // K512 also exercises every superblock source on a real multiworker team.
   hc103_cpu::prove(format,3,64,512,teams,alias,operation);
   ASSERT_FALSE(::testing::Test::HasFailure());
   for(int rows:{1,9,33}) {
    hc103_cpu::prove(format,rows,64,256,one,alias,operation);
    ASSERT_FALSE(::testing::Test::HasFailure());
   }
   if(!format.source_is_superblock) {
    hc103_cpu::prove(format,9,72,320,teams,alias,operation);
    ASSERT_FALSE(::testing::Test::HasFailure());
   }
  }
}
/** @test Malformed source/row/workspace/alias declarations reject before authoring activation or output. */
TEST(Q8ResidualAllFormatsCPU, HC103TypedAdmissionAndStageIdentity) {
 using namespace llaminar2;
 using namespace llaminar2::test;
 const auto &formats=quantizedVerifierFormats();const auto it=std::find_if(formats.begin(),formats.end(),[](const auto &f){return f.tensor_type==TensorType::Q8_0;});
 ASSERT_NE(it,formats.end());q8_all_formats::Threads workers(1);
 q8_all_formats::CPUFixture up(*it,3,256,256,false);
 CPUDeviceContext context(DeviceId::cpu(),1);
 FP32Tensor activated({3,256},DeviceId::cpu()),otherActivation({3,256},DeviceId::cpu());
 const auto raw=std::vector<float>(up.input.data(),up.input.data()+up.input.numel());
 const auto rows=DeviceRowRange::deviceCounted(3,&up.count);
 auto reset=[&] {
  std::copy(raw.begin(),raw.end(),up.input.mutable_data());
  std::fill_n(activated.mutable_data(),activated.numel(),-12345.f);
  std::fill_n(up.output.mutable_data(),up.output.numel(),-12345.f);
 };
 auto unchanged=[&] {
  EXPECT_EQ(std::memcmp(raw.data(),up.input.data(),raw.size()*sizeof(float)),0);
  for(std::size_t i=0;i<activated.numel();++i)EXPECT_EQ(activated.data()[i],-12345.f);
  for(std::size_t i=0;i<up.output.numel();++i)EXPECT_EQ(up.output.data()[i],-12345.f);
 };
 reset();
 EXPECT_THROW(up.kernel->multiply_q8_residual_corrected(HCResidualDownInput{up.input,activated},&up.output,rows,256,256,nullptr),std::runtime_error);
 unchanged();
 CPUProjectionTestWorkspace primaryOnly(cpuProjectionQ8WorkspaceRequirements(3,256));
 EXPECT_THROW(up.kernel->multiply_q8_residual_corrected(HCResidualDownInput{up.input,activated},&up.output,rows,256,256,primaryOnly.get()),std::runtime_error);
 unchanged();
 CPUProjectionTestWorkspace undersized(cpuQ8ResidualCorrectionWorkspaceRequirements(1,256,256));
 EXPECT_THROW(up.kernel->multiply_q8_residual_corrected(HCResidualDownInput{up.input,activated},&up.output,rows,256,256,undersized.get()),std::runtime_error);
 unchanged();
 EXPECT_THROW(up.kernel->multiply_q8_residual_corrected(HCResidualDownInput{up.input,activated},&up.output,rows,255,256,up.workspace->get()),std::invalid_argument);
 unchanged();
 EXPECT_THROW(up.kernel->multiply_q8_residual_corrected(HCResidualDownInput{up.input,activated},&up.output,rows,256,224,up.workspace->get()),std::invalid_argument);
 unchanged();
 for(const auto malformed:{DeviceRowRange::fullyActive(4),DeviceRowRange::deviceCounted(4,&up.count).slice(0,3),DeviceRowRange::deviceCounted(4,&up.count).slice(1,3),DeviceRowRange::deviceCounted(3,&up.count).repeatEachRow(2)}) {
  EXPECT_THROW(up.kernel->multiply_q8_residual_corrected(HCResidualDownInput{up.input,activated},&up.output,malformed,256,256,up.workspace->get()),std::invalid_argument);
  unchanged();
 }
 up.count=4;
 EXPECT_THROW(up.kernel->multiply_q8_residual_corrected(HCResidualDownInput{up.input,activated},&up.output,rows,256,256,up.workspace->get()),std::invalid_argument);
 unchanged();up.count=3;
 EXPECT_THROW(up.kernel->multiply_q8_residual_corrected(HCResidualDownInput{up.input,up.output},&up.output,rows,256,256,up.workspace->get()),std::invalid_argument);
 unchanged();
 EXPECT_THROW(up.kernel->multiply_q8_residual_corrected(HCResidualDownInput{up.input,activated},&up.input,rows,256,256,up.workspace->get()),std::invalid_argument);
 unchanged();
 // Canonical host view API supplies a genuine interior overlap; storage owns one parent.
 auto parent=std::make_shared<FP32Tensor>(std::vector<std::size_t>{3*256+1},DeviceId::cpu());
 auto rawView=parent->create_view({3,256},0),interior=parent->create_view({3,256},1);
 ASSERT_NE(rawView,nullptr);ASSERT_NE(interior,nullptr);
 EXPECT_THROW(up.kernel->multiply_q8_residual_corrected(HCResidualDownInput{*rawView,*interior},&up.output,rows,256,256,up.workspace->get()),std::invalid_argument);
 unchanged();
 // Runtime nonfinite/half-overflow retains the original activation side effect,
 // while the publisher's team barrier prohibits any projection output store.
 FP32Tensor invalidExpected({3,256},DeviceId::cpu());
 for(float invalid:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),1.e20f}) {
  reset();up.input.mutable_data()[0]=invalid;
  const qwen4exp::RowExtent extent{3,&up.count};
  hc103_original::residualDown({extent,256,up.input.data(),invalidExpected.mutable_data()},nullptr);
  EXPECT_THROW(up.kernel->multiply_q8_residual_corrected(HCResidualDownInput{up.input,activated},&up.output,rows,256,256,up.workspace->get()),std::invalid_argument);
  ASSERT_EQ(std::memcmp(activated.data(),invalidExpected.data(),activated.numel()*sizeof(float)),0);
  for(std::size_t i=0;i<up.output.numel();++i)ASSERT_EQ(up.output.data()[i],-12345.f);
 }
 reset();
 const auto params=hc103_cpu::fusedParams(up,up.input,activated,up.output);
 hc103_cpu::StageBorrow good(hc103_cpu::fusedStage(up,up.input,activated,up.output));
 const auto signature=good.stage->graphCaptureVariantSignature();
 auto changedActivation=hc103_cpu::fusedParams(up,up.input,otherActivation,up.output);
 GEMMStage other(changedActivation);EXPECT_NE(signature,other.graphCaptureVariantSignature());
 int otherCount=3;auto changedRows=params;changedRows.verifier_row_range=DeviceRowRange::deviceCounted(3,&otherCount);
 GEMMStage otherRowStage(changedRows);EXPECT_NE(signature,otherRowStage.graphCaptureVariantSignature());
 up.count=0;EXPECT_EQ(signature,good.stage->graphCaptureVariantSignature());up.count=3;
 good.stage->resetSessionState();EXPECT_EQ(signature,good.stage->graphCaptureVariantSignature());
 auto outputSlice=params;outputSlice.output_range={0,1};
 EXPECT_THROW((void)GEMMStage(outputSlice),std::invalid_argument);
 auto malformedRows=params;malformedRows.verifier_row_range=DeviceRowRange::fullyActive(1);
 EXPECT_THROW((void)GEMMStage(malformedRows),std::invalid_argument);
 auto malformedCapacity=params;malformedCapacity.verifier_row_range=DeviceRowRange::deviceCounted(4,&up.count).slice(0,3);
 EXPECT_THROW((void)GEMMStage(malformedCapacity),std::invalid_argument);
 auto wrongBankId=params;wrongBankId.hc_residual_down->buffer=BufferId::GATE_PROJ;
 EXPECT_THROW((void)GEMMStage(wrongBankId),std::invalid_argument);
 auto wrongAlias=params;wrongAlias.hc_residual_down->activated=&up.input;
 EXPECT_THROW((void)GEMMStage(wrongAlias),std::invalid_argument);
 auto wrongArithmetic=params;wrongArithmetic.quantized_operand_precision=QuantizedActivationOperandPrecision::SingleQ8;
 EXPECT_THROW((void)GEMMStage(wrongArithmetic),std::invalid_argument);
 auto wrongTransform=params;wrongTransform.do_swiglu=true;
 EXPECT_THROW((void)GEMMStage(wrongTransform),std::invalid_argument);
 auto wrongSource=params;auto foreignSource=q8_all_formats::source(*it,256,256);
 wrongSource.B=foreignSource.get();
 EXPECT_THROW((void)GEMMStage(wrongSource),std::invalid_argument);
 auto stale=params;++stale.prepared_ref->binding_id;
 EXPECT_THROW((void)GEMMStage(stale),std::invalid_argument);unchanged();
}

/** @test All original CPU formats execute the declarative fused HC builder at decode/prefill extents. */
TEST(Q8ResidualAllFormatsCPU, HC103TypedBuilderAllFormats) {
 using namespace llaminar2::test;
 const std::array one{1};
 for(const auto& format:quantizedVerifierFormats())for(bool alias:{false,true})for(int rows:{1,512}) {
  hc103_cpu::prove(format,rows,64,256,one,alias,hc103_cpu::ChainOperation::TypedBuilder);
  ASSERT_FALSE(::testing::Test::HasFailure());
 }
}

/** @test Counted CPU native verifier plain/bias/mixed/SwiGLU preserves every inactive byte. */
TEST(Q8ResidualAllFormatsCPU, CountedNativeVerifierExtent) {
 using namespace llaminar2;using namespace llaminar2::test;using namespace q8_all_formats;
 Threads workers(1);
 const auto& formats=quantizedVerifierFormats();
 for(std::size_t index=0;index<formats.size();++index)for(int capacity:{1,3,9}) {
  const auto& format=formats[index];
  SCOPED_TRACE(::testing::Message()<<format.label<<" capacity="<<capacity);
  CPUFixture first(format,capacity,256,256,false);
  CPUFixture second(formats[(index+1)%formats.size()],capacity,256,256,false);
  auto requirements=dynamic_cast<IWorkspaceConsumer*>(first.kernel)->getWorkspaceRequirements(capacity,256,256);
  requirements.merge(dynamic_cast<IWorkspaceConsumer*>(second.kernel)->getWorkspaceRequirements(capacity,256,256));
  CPUProjectionTestWorkspace workspace(capacity,256,requirements);
  FP32Tensor gate({std::size_t(capacity),256},DeviceId::cpu()),bias({256},DeviceId::cpu());
  for(std::size_t i=0;i<gate.numel();++i)gate.mutable_data()[i]=.1f+float(i%23)*.003f;
  for(std::size_t i=0;i<bias.numel();++i)bias.mutable_data()[i]=float(int(i%7)-3)*.007f;
  auto* q8=static_cast<Q8_1Block*>(workspace.get()->getBuffer(kCPUProjectionQ8));
  auto* transformed=static_cast<float*>(workspace.get()->getBuffer(kCPUSwiGLUInput));
  ASSERT_NE(q8,nullptr);ASSERT_NE(transformed,nullptr);
  const std::size_t qbytes=std::size_t(capacity)*8*sizeof(Q8_1Block);
  std::vector<float> goldFirst(first.output.numel()),goldSecond(second.output.numel());
  std::vector<std::uint8_t> goldQ(qbytes);
  std::vector<float> goldTransform(gate.numel());
  for(int operation=0;operation<3;++operation) {
   // Ordinary full authority freezes the production witness before dirty replays.
   const auto submit=[&] {
    if(operation==2)return first.kernel->multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
     &gate,&first.input,&first.output,capacity,256,256,1.f,0.f,workspace.get());
    std::vector<ITensorGemm::TensorProjectionDesc> projections{
     {first.kernel,&first.output,256,operation==1?&bias:nullptr,"first"}};
    if(operation==1)projections.push_back({second.kernel,&second.output,256,&bias,"mixed"});
    return first.kernel->multiply_fused_verifier_rows_decode_equivalent(
     &first.input,projections,capacity,256,nullptr,workspace.get());
   };
   std::copy(first.values.begin(),first.values.end(),first.input.mutable_data());
   ASSERT_TRUE(submit());
   std::copy_n(first.output.data(),goldFirst.size(),goldFirst.data());
   if(operation==1)std::copy_n(second.output.data(),goldSecond.size(),goldSecond.data());
   std::memcpy(goldQ.data(),q8,qbytes);
   if(operation==2)std::copy_n(transformed,goldTransform.size(),goldTransform.data());
   int count=capacity;
   for(int replay=0;replay<20;++replay) {
    count=std::array{capacity,std::min(capacity,1),0,std::max(0,capacity-1),capacity}[replay%5];
    SCOPED_TRACE(::testing::Message()<<"operation="<<operation<<" replay="<<replay<<" live="<<count);
    std::copy(first.values.begin(),first.values.end(),first.input.mutable_data());
    std::fill(first.input.mutable_data()+std::size_t(count)*256,first.input.mutable_data()+first.input.numel(),std::numeric_limits<float>::quiet_NaN());
    std::fill_n(first.output.mutable_data(),first.output.numel(),-12345.f);
    std::fill_n(second.output.mutable_data(),second.output.numel(),-12345.f);
    std::memset(q8,0x5a,qbytes);std::fill_n(transformed,gate.numel(),-12345.f);
    auto scope=first.kernel->beginVerifierDecodeEquivalentScope(DeviceRowRange::deviceCounted(capacity,&count));
    ASSERT_TRUE(submit());
    EXPECT_EQ(std::memcmp(first.output.data(),goldFirst.data(),std::size_t(count)*256*sizeof(float)),0);
    if(operation==1)EXPECT_EQ(std::memcmp(second.output.data(),goldSecond.data(),std::size_t(count)*256*sizeof(float)),0);
    EXPECT_EQ(std::memcmp(q8,goldQ.data(),std::size_t(count)*8*sizeof(Q8_1Block)),0);
    for(std::size_t i=std::size_t(count)*256;i<first.output.numel();++i) {
     ASSERT_EQ(first.output.data()[i],-12345.f);ASSERT_EQ(second.output.data()[i],-12345.f);
     if(operation==2)ASSERT_EQ(transformed[i],-12345.f);
    }
    if(operation==2)EXPECT_EQ(std::memcmp(transformed,goldTransform.data(),std::size_t(count)*256*sizeof(float)),0);
    for(std::size_t i=std::size_t(count)*8*sizeof(Q8_1Block);i<qbytes;++i)
     ASSERT_EQ(reinterpret_cast<const std::uint8_t*>(q8)[i],0x5a);
   }
   // Invalid publications fail before any output, Q8, or transform mutation.
   for(int invalid:{-1,capacity+1}) {
    count=invalid;std::memset(q8,0x5a,qbytes);
    std::fill_n(first.output.mutable_data(),first.output.numel(),-12345.f);
    std::fill_n(second.output.mutable_data(),second.output.numel(),-12345.f);
    std::fill_n(transformed,gate.numel(),-12345.f);
    auto scope=first.kernel->beginVerifierDecodeEquivalentScope(DeviceRowRange::deviceCounted(capacity,&count));
    EXPECT_THROW(submit(),std::invalid_argument);
    for(std::size_t i=0;i<qbytes;++i)ASSERT_EQ(reinterpret_cast<const std::uint8_t*>(q8)[i],0x5a);
    for(std::size_t i=0;i<first.output.numel();++i){ASSERT_EQ(first.output.data()[i],-12345.f);ASSERT_EQ(second.output.data()[i],-12345.f);ASSERT_EQ(transformed[i],-12345.f);}
   }
  }
 }
}

/** @test Canonical fixed and counted expanded/sliced rows retain their logical authority. */
TEST(Q8ResidualAllFormatsCPU, CountedNativeVerifierRowGeometry) {
 using namespace llaminar2;using namespace llaminar2::test;using namespace q8_all_formats;
 Threads workers(1);
 const auto& formats=quantizedVerifierFormats();
 const auto found=std::find_if(formats.begin(),formats.end(),[](const auto& f){return f.tensor_type==TensorType::Q8_0;});
 ASSERT_NE(found,formats.end());CPUFixture fixture(*found,4,256,256,false);
 CPUProjectionTestWorkspace workspace(4,256,dynamic_cast<IWorkspaceConsumer*>(fixture.kernel)->getWorkspaceRequirements(4,256,256));
 const std::vector<ITensorGemm::TensorProjectionDesc> projections{{fixture.kernel,&fixture.output,256,nullptr,"rows"}};
 const auto submit=[&]{return fixture.kernel->multiply_fused_verifier_rows_decode_equivalent(&fixture.input,projections,4,256,nullptr,workspace.get());};
 ASSERT_TRUE(submit());const std::vector<float> gold(fixture.output.data(),fixture.output.data()+fixture.output.numel());
 auto* bank=static_cast<Q8_1Block*>(workspace.get()->getBuffer(kCPUProjectionQ8));ASSERT_NE(bank,nullptr);
 const std::vector<Q8_1Block> goldBank(bank,bank+32);
 const auto verify=[&](DeviceRowRange rows,int live) {
  std::fill_n(fixture.output.mutable_data(),fixture.output.numel(),-12345.f);std::memset(bank,0x5a,32*sizeof(Q8_1Block));
  auto scope=fixture.kernel->beginVerifierDecodeEquivalentScope(rows);ASSERT_TRUE(submit());
  EXPECT_EQ(std::memcmp(fixture.output.data(),gold.data(),std::size_t(live)*256*sizeof(float)),0);
  EXPECT_EQ(std::memcmp(bank,goldBank.data(),std::size_t(live)*8*sizeof(Q8_1Block)),0);
  for(std::size_t i=std::size_t(live)*256;i<fixture.output.numel();++i)ASSERT_EQ(fixture.output.data()[i],-12345.f);
  for(std::size_t i=std::size_t(live)*8*sizeof(Q8_1Block);i<32*sizeof(Q8_1Block);++i)ASSERT_EQ(reinterpret_cast<std::uint8_t*>(bank)[i],0x5a);
 };
 verify(DeviceRowRange::fullyActive(2).repeatEachRow(2),4);
 verify(DeviceRowRange::fullyActive(2).repeatEachRow(3).slice(1,4),4);
 int count=0;
 for(int replay=0;replay<20;++replay) {
  count=replay%3;verify(DeviceRowRange::deviceCounted(2,&count).repeatEachRow(3).slice(1,4),count==0?0:count==1?2:4);
 }
 {
 count=0;auto scope=fixture.kernel->beginVerifierDecodeEquivalentScope(DeviceRowRange::deviceCounted(4,&count));
 std::fill_n(fixture.output.mutable_data(),fixture.output.numel(),-12345.f);std::memset(bank,0x5a,32*sizeof(Q8_1Block));
 EXPECT_THROW(fixture.kernel->multiply_fused_verifier_rows_decode_equivalent(&fixture.input,projections,4,256,nullptr,nullptr),std::runtime_error);
 // Empty calls still reject incomplete input before their no-work return.
 /** @brief Valid-capacity diagnostic tensor withholding its authoritative host pointer. */
 struct UnpublishedInput final:FP32Tensor {
  using FP32Tensor::FP32Tensor;
  const float* data() const override{return nullptr;}
 };
 UnpublishedInput unpublished({4,256},DeviceId::cpu());
 EXPECT_FALSE(fixture.kernel->multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(&unpublished,&fixture.input,&fixture.output,4,256,256,1,0,workspace.get()));
 FP32Tensor shortInput({1},DeviceId::cpu());
 EXPECT_FALSE(fixture.kernel->multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(&shortInput,&fixture.input,&fixture.output,4,256,256,1,0,workspace.get()));
 for(std::size_t i=0;i<fixture.output.numel();++i)ASSERT_EQ(fixture.output.data()[i],-12345.f);
 for(std::size_t i=0;i<32*sizeof(Q8_1Block);++i)ASSERT_EQ(reinterpret_cast<std::uint8_t*>(bank)[i],0x5a);
 }
 // Ordinary TwoTerm accepts canonical expanded/sliced rows independently of HC.
 // Freeze full ordinary output and both exact Q8_1 block banks before histories.
 auto* primary=static_cast<Q8_1Block*>(fixture.workspace->get()->getBuffer(kCPUProjectionQ8));
 auto* residual=static_cast<Q8_1Block*>(fixture.workspace->get()->getBuffer(kCPUProjectionResidualQ8));
 ASSERT_NE(primary,nullptr);ASSERT_NE(residual,nullptr);
 fixture.kernel->multiply_q8_residual_corrected(&fixture.input,&fixture.output,
  DeviceRowRange::fullyActive(4),256,256,fixture.workspace->get());
 const std::vector<float> twoTermGold(fixture.output.data(),fixture.output.data()+fixture.output.numel());
 const std::vector<Q8_1Block> primaryGold(primary,primary+32),residualGold(residual,residual+32);
 const auto verifyTwoTerm=[&](DeviceRowRange rows,int live) {
  std::fill_n(fixture.output.mutable_data(),fixture.output.numel(),-12345.f);
  std::memset(primary,0x5a,32*sizeof(Q8_1Block));std::memset(residual,0x5a,32*sizeof(Q8_1Block));
  fixture.kernel->multiply_q8_residual_corrected(&fixture.input,&fixture.output,rows,256,256,fixture.workspace->get());
  EXPECT_EQ(std::memcmp(fixture.output.data(),twoTermGold.data(),std::size_t(live)*256*sizeof(float)),0);
  EXPECT_EQ(std::memcmp(primary,primaryGold.data(),std::size_t(live)*8*sizeof(Q8_1Block)),0);
  EXPECT_EQ(std::memcmp(residual,residualGold.data(),std::size_t(live)*8*sizeof(Q8_1Block)),0);
  for(std::size_t i=std::size_t(live)*256;i<fixture.output.numel();++i)ASSERT_EQ(fixture.output.data()[i],-12345.f);
  for(std::size_t i=std::size_t(live)*8*sizeof(Q8_1Block);i<32*sizeof(Q8_1Block);++i) {
   ASSERT_EQ(reinterpret_cast<const std::uint8_t*>(primary)[i],0x5a);
   ASSERT_EQ(reinterpret_cast<const std::uint8_t*>(residual)[i],0x5a);
  }
 };
 verifyTwoTerm(DeviceRowRange::fullyActive(2).repeatEachRow(2),4);
 verifyTwoTerm(DeviceRowRange::fullyActive(2).repeatEachRow(3).slice(1,4),4);
 for(int replay=0;replay<20;++replay) {
  count=replay%3;
  verifyTwoTerm(DeviceRowRange::deviceCounted(2,&count).repeatEachRow(3).slice(1,4),count==0?0:count==1?2:4);
 }

}

/** @test Missing aggregate fused ordered-partial storage rejects before quantizer publication. */
TEST(Q8ResidualAllFormatsCPU, CountedNativeVerifierFusedBankAdmission) {
 using namespace llaminar2;using namespace llaminar2::test;using namespace q8_all_formats;
 using namespace llaminar2::cpu::native_vnni;
 Threads workers(3);
 ASSERT_EQ(nativeVNNIInvocationWorkerCount(),3);
 // N64/K131072 exceeds the original 16M FLOP K-parallel threshold.
 // Each of two M1 members owns one N tile: two tasks < three workers,
 // so neither ISA can select the private output-tile-local bank route.
 const auto& formats=quantizedVerifierFormats();
 const auto found=std::find_if(formats.begin(),formats.end(),[](const auto& f){return f.tensor_type==TensorType::Q8_0;});
 ASSERT_NE(found,formats.end());CPUFixture fixture(*found,1,64,131072,false);
 FP32Tensor sibling({1,64},DeviceId::cpu());
 const auto* packed=fixture.kernel->exportCPUNativeVNNIPackedWeights();ASSERT_NE(packed,nullptr);
 const std::array descriptors{FusedVerifierRowsDesc{packed,fixture.output.mutable_data(),nullptr,64,64},
  FusedVerifierRowsDesc{packed,sibling.mutable_data(),nullptr,64,64}};
 const auto plan=planNativeVNNIFusedRows(descriptors,1,3);
 ASSERT_TRUE(plan.all_k_partitioned);
 ASSERT_EQ(plan.partial_route,NativeVNNIFusedPartialRoute::SharedBank);
 ASSERT_GT(plan.sharedPartialFloats(),0u)<<"Chosen real K-partitioned bundle must require its shared ordered bank";
 // Deliberately admit one fewer float through the normal test arena authority.
 auto requirements=cpuProjectionQ8WorkspaceRequirements(1,131072);
 requirements.merge(cpuProjectionPartialWorkspaceRequirements(plan.sharedPartialFloats()-1));
 CPUProjectionTestWorkspace workspace(requirements);
 auto* bank=static_cast<Q8_1Block*>(workspace.get()->getBuffer(kCPUProjectionQ8));ASSERT_NE(bank,nullptr);
 const std::size_t bytes=4096*sizeof(Q8_1Block);std::memset(bank,0x5a,bytes);
 std::fill_n(fixture.output.mutable_data(),64,-12345.f);std::fill_n(sibling.mutable_data(),64,-12345.f);
 int live=1;auto scope=fixture.kernel->beginVerifierDecodeEquivalentScope(DeviceRowRange::deviceCounted(1,&live));
 const std::vector<ITensorGemm::TensorProjectionDesc> projections{{fixture.kernel,&fixture.output,64,nullptr,"a"},{fixture.kernel,&sibling,64,nullptr,"b"}};
 EXPECT_THROW(fixture.kernel->multiply_fused_verifier_rows_decode_equivalent(&fixture.input,projections,1,131072,nullptr,workspace.get()),std::runtime_error);
 for(std::size_t i=0;i<bytes;++i)ASSERT_EQ(reinterpret_cast<std::uint8_t*>(bank)[i],0x5a);
 for(int i=0;i<64;++i){ASSERT_EQ(fixture.output.data()[i],-12345.f);ASSERT_EQ(sibling.data()[i],-12345.f);}
}

/** @brief Select physical workers without admitting logical siblings as the budget. */
enum class HC103TimingWorkers { One, PhysicalSocket };
/** @brief Reuse the byte-calibrated full prepared chain for opt-in CPU timing.
 * @param rows Exact decode or prefill row capacity.
 * @param selection One worker or detected physical cores per socket.
 * @param observation Paired timing default, or one explicit profiler side.
 */
static void hc103CandidateTiming(int rows,HC103TimingWorkers selection,
 llaminar2::test::hc103_cpu::ChainObservation observation=
  llaminar2::test::hc103_cpu::ChainObservation::PairedEvents) {
 using namespace llaminar2;using namespace llaminar2::test;
 const int budget=std::max(1,MPIBootstrap::detectCPUTopology().cores_per_socket);
 const int workers=selection==HC103TimingWorkers::One?1:budget;
 const auto& formats=quantizedVerifierFormats();
 const auto format=std::find_if(formats.begin(),formats.end(),[](const auto& f){return f.tensor_type==TensorType::Q8_0;});
 ASSERT_NE(format,formats.end());
 const std::array teams{workers};
 hc103_cpu::prove(*format,rows,2560,320,teams,false,hc103_cpu::ChainOperation::FusedStage,
  hc103_cpu::ChainTiming{.workers=workers,.observation=observation});
}
/** @test Opt-in exact M1 full-chain paired CPU timing with one physical worker selection. */
TEST(Q8ResidualAllFormatsCPU, DISABLED_HC103CandidateM1OnePairedChainEvents) {
 hc103CandidateTiming(1,HC103TimingWorkers::One);
}
/** @test Opt-in exact M1 full-chain paired CPU timing with socket physical worker selection. */
TEST(Q8ResidualAllFormatsCPU, DISABLED_HC103CandidateM1SocketPairedChainEvents) {
 hc103CandidateTiming(1,HC103TimingWorkers::PhysicalSocket);
}
/** @test Opt-in exact M512 full-chain paired CPU timing with one physical worker selection. */
TEST(Q8ResidualAllFormatsCPU, DISABLED_HC103CandidateM512OnePairedChainEvents) {
 hc103CandidateTiming(512,HC103TimingWorkers::One);
}
/** @test Opt-in exact M512 full-chain paired CPU timing with socket physical worker selection. */
TEST(Q8ResidualAllFormatsCPU, DISABLED_HC103CandidateM512SocketPairedChainEvents) {
 hc103CandidateTiming(512,HC103TimingWorkers::PhysicalSocket);
}

/** @test Opt-in single candidate chain profile: M1, one physical workers. */
TEST(Q8ResidualAllFormatsCPU, DISABLED_HC103ProfileCandidateM1One) {
 hc103CandidateTiming(1,HC103TimingWorkers::One,
  llaminar2::test::hc103_cpu::ChainObservation::ProfileCandidate);
}

/** @test Opt-in single candidate chain profile: M1, socket physical workers. */
TEST(Q8ResidualAllFormatsCPU, DISABLED_HC103ProfileCandidateM1Socket) {
 hc103CandidateTiming(1,HC103TimingWorkers::PhysicalSocket,
  llaminar2::test::hc103_cpu::ChainObservation::ProfileCandidate);
}

/** @test Opt-in single candidate chain profile: M512, one physical workers. */
TEST(Q8ResidualAllFormatsCPU, DISABLED_HC103ProfileCandidateM512One) {
 hc103CandidateTiming(512,HC103TimingWorkers::One,
  llaminar2::test::hc103_cpu::ChainObservation::ProfileCandidate);
}

/** @test Opt-in single candidate chain profile: M512, socket physical workers. */
TEST(Q8ResidualAllFormatsCPU, DISABLED_HC103ProfileCandidateM512Socket) {
 hc103CandidateTiming(512,HC103TimingWorkers::PhysicalSocket,
  llaminar2::test::hc103_cpu::ChainObservation::ProfileCandidate);
}

/** @test Opt-in single original chain profile: M1, one physical workers. */
TEST(Q8ResidualAllFormatsCPU, DISABLED_HC103ProfileOriginalM1One) {
 hc103CandidateTiming(1,HC103TimingWorkers::One,
  llaminar2::test::hc103_cpu::ChainObservation::ProfileOriginal);
}

/** @test Opt-in single original chain profile: M1, socket physical workers. */
TEST(Q8ResidualAllFormatsCPU, DISABLED_HC103ProfileOriginalM1Socket) {
 hc103CandidateTiming(1,HC103TimingWorkers::PhysicalSocket,
  llaminar2::test::hc103_cpu::ChainObservation::ProfileOriginal);
}

/** @test Opt-in single original chain profile: M512, one physical workers. */
TEST(Q8ResidualAllFormatsCPU, DISABLED_HC103ProfileOriginalM512One) {
 hc103CandidateTiming(512,HC103TimingWorkers::One,
  llaminar2::test::hc103_cpu::ChainObservation::ProfileOriginal);
}

/** @test Opt-in single original chain profile: M512, socket physical workers. */
TEST(Q8ResidualAllFormatsCPU, DISABLED_HC103ProfileOriginalM512Socket) {
 hc103CandidateTiming(512,HC103TimingWorkers::PhysicalSocket,
  llaminar2::test::hc103_cpu::ChainObservation::ProfileOriginal);
}
