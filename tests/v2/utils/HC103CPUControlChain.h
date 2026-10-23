/**
 * @file HC103CPUControlChain.h
 * @brief Real prepared down/TwoTerm-up CPU chain calibrated against frozen arithmetic.
 *
 * The independent oracle owns nonlinear and publication math. Both chains retain
 * the same original prepared projection source, numerical policy and workspace.
 * No replacement projection or inference path is supplied. All storage belongs
 * to this test invocation; stages retire their workspace borrow before arenas.
 * Opt-in hardware observations use canonical process-owned counter control
 * around one warmed selected chain, separately from paired timing.
 * Timed controls call production unfused primitives with their public validation;
 * independent frozen arithmetic remains exclusively in the external byte proof.
 */
#pragma once
#include "Q8ResidualAllFormatCPUHarness.h"
#include "HC103OriginalCPU.h"
#include <bit>
#include <chrono>
#include <optional>
#include "../performance/kernels/native_vnni_dispatch/NativeVNNIProfilerControl.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpHyperconnectionBuilder.h"
#include "execution/local_execution/graph/ComputeGraph.h"
namespace llaminar2::test::hc103_cpu {
using namespace q8_all_formats;
/** @brief Explicit diagnostic call site; original control remains the default. */
enum class ChainOperation { OriginalControl, FusedStage, TypedOverload, TypedBuilder };
/** @brief Opt-in observation; profiling never enters the paired timing loop. */
enum class ChainObservation { PairedEvents, ProfileCandidate, ProfileOriginal };
/** @brief Explicit physical-worker measurement request; absent means correctness only. */
struct ChainTiming {
 int workers; ///< Positive physical worker team selected by the exact test.
 ChainObservation observation=ChainObservation::PairedEvents; ///< Timing default preserves original selectors.
};
/** @brief Construct the actual immutable prepared up stage over the chosen activation owner. */
inline std::unique_ptr<GEMMStage> upStage(CPUFixture &up,FP32Tensor &activated,FP32Tensor &destination) {
 auto stage=std::make_unique<GEMMStage>(GEMMStage::Params{
  .device_id=DeviceId::cpu(),.A=&activated,.B=up.weight.get(),.C=&destination,
  .m=up.m,.n=up.n,.k=up.k,.a_buffer_id=BufferId::NORMALIZED,.c_buffer_id=BufferId::GATE_PROJ,
  .force_decode_equivalent_verifier_prefill=true,
  .verifier_row_range=DeviceRowRange::deviceCounted(up.m,&up.count),
  .prepared_ref=up.binding.ref,.prepared_store=up.binding.store.get(),
  .quantized_operand_precision=QuantizedActivationOperandPrecision::TwoTermQ8});
 stage->bindWorkspace(up.workspace->get());return stage;
}
/** @brief Bind the typed production HC input transform on the real prepared up stage.
 * @param up Original prepared source, geometry, count owner and workspace.
 * @param raw Original down output.
 * @param activated Retained activation, optionally exactly raw.
 * @param destination Distinct original up output.
 * @return Complete typed stage declaration with original prepared authority.
 */
inline GEMMStage::Params fusedParams(CPUFixture &up,TensorBase &raw,
 TensorBase &activated,TensorBase &destination) {
 return GEMMStage::Params{
  .device_id=DeviceId::cpu(),.A=&raw,.B=up.weight.get(),.C=&destination,
  .hc_residual_down=GEMMStage::HCResidualDownPublication{&activated,
    &raw==&activated?BufferId::NORMALIZED:BufferId::UP_PROJ},
  .m=up.m,.n=up.n,.k=up.k,.a_buffer_id=BufferId::NORMALIZED,.c_buffer_id=BufferId::GATE_PROJ,
  .force_decode_equivalent_verifier_prefill=true,
  .verifier_row_range=DeviceRowRange::deviceCounted(up.m,&up.count),
  .prepared_ref=up.binding.ref,.prepared_store=up.binding.store.get(),
  .quantized_operand_precision=QuantizedActivationOperandPrecision::TwoTermQ8};
}
/** @brief Materialize only the existing stage/workspace borrow, without executing arithmetic.
 * @param up Actual prepared up fixture and named workspace.
 * @param raw Original down output.
 * @param activated Existing activation output or exact raw alias.
 * @param destination Distinct original up output.
 * @return Stage borrowing the fixture's existing prepared source/workspace.
 */
inline std::unique_ptr<GEMMStage> fusedStage(CPUFixture &up,FP32Tensor &raw,
 FP32Tensor &activated,FP32Tensor &destination) {
 auto stage=std::make_unique<GEMMStage>(fusedParams(up,raw,activated,destination));
 stage->bindWorkspace(up.workspace->get());return stage;
}
/** @brief End each stage's workspace borrow before normal or assertion-path destruction. */
struct StageBorrow {
 std::unique_ptr<GEMMStage> stage;
 explicit StageBorrow(std::unique_ptr<GEMMStage> value):stage(std::move(value)){}
 ~StageBorrow(){stage->unbindWorkspace();}
};
/**
 * @brief Calibrate full prepared HC read chains, all live bytes and poisoned reuse.
 * @param format Canonical original source creator/codebook, shared by both engines.
 * @param rows Physical row capacity, retained through twenty prefix histories.
 * @param hidden Original per-branch width; down K and up N are exactly4H.
 * @param rank Complete32-channel low-rank activation width, source-format admitted.
 * @param teams Positive independent OpenMP teams.
 * @param alias Exact rawDown/activated alias, never an interior overlap.
 * @param operation Original control, real fused stage, typed overload or complete declarative builder.
 * @param timing Optional opt-in paired timing or one selected profile after byte calibration.
 */
inline void prove(const QuantizedVerifierFormatCase &format,int rows,int hidden,int rank,
 std::span<const int> teams,bool alias,ChainOperation operation=ChainOperation::OriginalControl,
 std::optional<ChainTiming> timing=std::nullopt) {
 // Only test tooling reads the canonical immutable profiler request here.
 // Reject misuse before fixture preparation, never profile a paired sweep.
 std::string profileRequest,profileOutput;
 if(timing) {
  using namespace native_vnni_dispatch;
  profileRequest=profilerRequestId();
  profileOutput=profilerEnvironment(kPerfStatsPathEnvironment);
  const auto batch=profilerEnvironment(kProfilerBatchPathEnvironment);
  switch(timing->observation) {
   case ChainObservation::PairedEvents:
    if(!profileRequest.empty()||!profileOutput.empty()||!batch.empty())
     throw std::invalid_argument("HC103 paired events reject profiler request/output metadata");
    break;
   case ChainObservation::ProfileCandidate:
   case ChainObservation::ProfileOriginal:
    if(profileRequest.empty()||profileOutput.empty()||!batch.empty())
     throw std::invalid_argument("HC103 exact profile requires one canonical request/output and no batch");
    break;
   default:throw std::invalid_argument("HC103 invalid chain observation");
  }
 }
 SCOPED_TRACE(::testing::Message()<<format.label<<" rows="<<rows<<" H="<<hidden<<" rank="<<rank<<" alias="<<alias);
 ASSERT_FALSE(teams.empty());Threads capacity(*std::max_element(teams.begin(),teams.end()));
 CPUFixture down(format,rows,rank,4*hidden,false),up(format,rows,4*hidden,rank,false);
 auto primitive=qwen4exp::createCPUQwen4ExpKernels();ASSERT_NE(primitive,nullptr);
 FP32Tensor referenceActivation({std::size_t(rows),std::size_t(rank)},DeviceId::cpu());
 FP32Tensor referenceUp({std::size_t(rows),std::size_t(4*hidden)},DeviceId::cpu());
 FP32Tensor &activated=alias?down.output:up.input;
 const bool builderPath=operation==ChainOperation::TypedBuilder;
 FP32Tensor branchInput({std::size_t(rows),std::size_t(4*hidden)},DeviceId::cpu());
 FP32Tensor normGain({4,std::size_t(hidden)},DeviceId::cpu());
 FP32Tensor builderRead({std::size_t(rows),std::size_t(hidden)},DeviceId::cpu());
 std::fill_n(normGain.mutable_data(),normGain.numel(),1.f);
 std::unique_ptr<ComputeGraph> builder;
 /** @brief End builder workspace borrows while both prepared fixtures still exist. */
 struct BuilderBorrow {std::unique_ptr<ComputeGraph>& graph;
  ~BuilderBorrow(){if(graph)for(const auto& name:graph->getExecutionOrder())
   if(auto* consumer=dynamic_cast<IWorkspaceConsumer*>(graph->getNode(name)->stage.get()))consumer->unbindWorkspace();}
 } builderBorrow{builder};
 StageBorrow control(operation==ChainOperation::FusedStage?fusedStage(up,down.output,activated,up.output):upStage(up,activated,up.output));
 const auto signature=control.stage->graphCaptureVariantSignature();
 if(operation==ChainOperation::FusedStage) {
  StageBorrow original(upStage(up,activated,up.output));
  ASSERT_NE(signature,original.stage->graphCaptureVariantSignature());
  const auto contract=control.stage->bufferContract();
  if(alias) {
   ASSERT_EQ(contract.inouts.size(),1u);EXPECT_EQ(contract.inouts[0].id,BufferId::NORMALIZED);
   EXPECT_FALSE(contract.inouts[0].prepare_write_storage);
  } else {
   ASSERT_TRUE(std::any_of(contract.outputs.begin(),contract.outputs.end(),[](const auto &b){return b.id==BufferId::UP_PROJ;}));
  }
 }
 StageBorrow reference(upStage(up,referenceActivation,referenceUp));
 auto *primary=static_cast<Q8_1Block*>(up.workspace->get()->getBuffer(kCPUProjectionQ8));
 auto *residual=static_cast<Q8_1Block*>(up.workspace->get()->getBuffer(kCPUProjectionResidualQ8));
 ASSERT_NE(primary,nullptr);ASSERT_NE(residual,nullptr);
 if(builderPath) {
  using F=Qwen4ExpFloatBinding;builder=std::make_unique<ComputeGraph>();
  auto downBinding=down.binding.binding,upBinding=up.binding.binding;
  downBinding.prepared=down.binding.ref;upBinding.prepared=up.binding.ref;
  const auto upRows=DeviceRowRange::deviceCounted(rows,&up.count);
  const Qwen4ExpHyperconnectionContext declaration{.device=DeviceId::cpu(),.prefix="hc103_",
   .rows={rows,Qwen4ExpIntInput{Qwen4ExpNativeIntInput{DeviceId::cpu(),&up.count,1}}},
   .hidden=hidden,.lowRank=rank,.epsilon=1e-6f,
   .arithmetic=Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier,
   .projectionRows=upRows,
   .upOperandPrecision=QuantizedActivationOperandPrecision::TwoTermQ8,.upProjectionRows=upRows};
  Qwen4ExpHyperconnectionBuilder::appendFinalRead(*builder,declaration,
   {Qwen4ExpImmutableWeightBinding(normGain),Qwen4ExpPreparedProjection(downBinding,*down.binding.store,DeviceId::cpu()),
    Qwen4ExpPreparedProjection(upBinding,*up.binding.store,DeviceId::cpu())},
   {F(branchInput,BufferId::RESIDUAL),F(down.input,BufferId::NORMALIZED),F(down.output,BufferId::GATE_PROJ),
    F(activated,alias?BufferId::GATE_PROJ:BufferId::UP_PROJ),F(up.output,BufferId::ATTN_OUTPUT),F(builderRead,BufferId::HIDDEN_STATE)},{});
  ASSERT_EQ(builder->getExecutionOrder().size(),4u);
  ASSERT_EQ(builder->getNode("hc103_read_activation"),nullptr);
  auto* downStage=dynamic_cast<GEMMStage*>(builder->getNode("hc103_read_down")->stage.get());
  auto* upStage=dynamic_cast<GEMMStage*>(builder->getNode("hc103_read_up")->stage.get());
  ASSERT_NE(downStage,nullptr);ASSERT_NE(upStage,nullptr);
  ASSERT_TRUE(upStage->getParams().hc_residual_down);
  EXPECT_EQ(upStage->getParams().A,&down.output);
  EXPECT_EQ(upStage->getParams().hc_residual_down->activated,&activated);
  EXPECT_EQ(builder->getNode("hc103_read_up")->dependencies,std::vector<std::string>{"hc103_read_down"});
  downStage->bindWorkspace(down.workspace->get());upStage->bindWorkspace(up.workspace->get());
 }
 const auto builderUp=builderPath?dynamic_cast<GEMMStage*>(builder->getNode("hc103_read_up")->stage.get()):nullptr;
 const auto builderSignature=builderUp?builderUp->graphCaptureVariantSignature():0;
 const std::size_t blocks=std::size_t(rows)*rank/32;
 std::vector<Q8_1Block> expectedPrimary(blocks),expectedResidual(blocks),savedPrimary(blocks),savedResidual(blocks);
 std::vector<float> rawDown(std::size_t(rows)*rank),read(std::size_t(rows)*hidden),expectedRead(read.size());
 const auto downRaw=down.raw,upRaw=up.raw;
 const auto *packedDown=down.kernel->exportCPUNativeVNNIPackedWeights();
 const auto *packedUp=up.kernel->exportCPUNativeVNNIPackedWeights();
 ASSERT_NE(packedDown,nullptr);ASSERT_NE(packedUp,nullptr);
 const auto *downAddress=packedDown->interleavedBase(),*upAddress=packedUp->interleavedBase();
 const std::vector<std::uint8_t> downPrepared(downAddress,downAddress+cpu::native_vnni::interleavedWorkspaceSize(*packedDown));
 const std::vector<std::uint8_t> upPrepared(upAddress,upAddress+cpu::native_vnni::interleavedWorkspaceSize(*packedUp));
 for(int team:teams) {
  Threads workers(team);CPUDeviceContext context(DeviceId::cpu(),team);
  for(int replay=0;replay<20;++replay) {
   const int live=std::array{rows,std::min(rows,1),0,std::min(rows,9),rows}[replay%5];
   SCOPED_TRACE(::testing::Message()<<"team="<<team<<" replay="<<replay<<" live="<<live);
   up.count=live;
   if(builderPath) {
    std::copy(down.values.begin(),down.values.end(),branchInput.mutable_data());
    std::fill(branchInput.mutable_data()+std::size_t(live)*4*hidden,
      branchInput.mutable_data()+branchInput.numel(),std::numeric_limits<float>::quiet_NaN());
    std::fill_n(down.input.mutable_data(),down.input.numel(),std::numeric_limits<float>::quiet_NaN());
    std::fill_n(down.output.mutable_data(),down.output.numel(),-12345.f);
    if(!alias)std::fill_n(activated.mutable_data(),activated.numel(),-12345.f);
    std::fill_n(up.output.mutable_data(),up.output.numel(),-12345.f);
    std::fill_n(builderRead.mutable_data(),builderRead.numel(),-12345.f);
    std::memset(primary,0x5a,blocks*sizeof(*primary));std::memset(residual,0x5a,blocks*sizeof(*residual));
    for(const auto& name:builder->getExecutionOrder()) {
     ASSERT_TRUE(builder->getNode(name)->stage->execute(&context));
     if(name=="hc103_read_down")std::copy_n(down.output.data(),rawDown.size(),rawDown.data());
    }
   } else {
    std::copy(down.values.begin(),down.values.end(),down.input.mutable_data());
    std::fill(down.input.mutable_data()+std::size_t(live)*4*hidden,down.input.mutable_data()+down.input.numel(),std::numeric_limits<float>::quiet_NaN());
    std::fill_n(down.output.mutable_data(),down.output.numel(),-12345.f);
    if(live)ASSERT_TRUE(down.kernel->multiply_tensor(&down.input,&down.output,live,rank,4*hidden,true,1,0,nullptr,nullptr,-1,down.workspace->get()));
    std::copy_n(down.output.data(),rawDown.size(),rawDown.data());
    if(!alias)std::fill_n(activated.mutable_data(),activated.numel(),-12345.f);
   }
   std::fill_n(referenceActivation.mutable_data(),referenceActivation.numel(),-12345.f);
   const qwen4exp::RowExtent extent{rows,&up.count};
   if(operation==ChainOperation::OriginalControl)
    primitive->residualDown({extent,rank,down.output.data(),activated.mutable_data()},nullptr);
   hc103_original::residualDown({extent,rank,rawDown.data(),referenceActivation.mutable_data()},nullptr);

   if(!builderPath){std::memset(primary,0x5a,blocks*sizeof(*primary));std::memset(residual,0x5a,blocks*sizeof(*residual));}
   std::memset(expectedPrimary.data(),0x5a,blocks*sizeof(Q8_1Block));std::memset(expectedResidual.data(),0x5a,blocks*sizeof(Q8_1Block));
   if(!builderPath)std::fill_n(up.output.mutable_data(),up.output.numel(),-12345.f);
   if(operation==ChainOperation::TypedOverload)
    up.kernel->multiply_q8_residual_corrected(HCResidualDownInput{down.output,activated},&up.output,
      DeviceRowRange::deviceCounted(rows,&up.count),4*hidden,rank,up.workspace->get());
   else if(!builderPath)ASSERT_TRUE(control.stage->execute(&context));
   ASSERT_EQ(std::memcmp(activated.data(),referenceActivation.data(),rawDown.size()*sizeof(float)),0)<<"activated";
   EXPECT_EQ(control.stage->graphCaptureVariantSignature(),signature);
   if(builderUp) {
    EXPECT_EQ(builderUp->graphCaptureVariantSignature(),builderSignature);
    EXPECT_TRUE(builderUp->prepareGraphLaunch(&context,nullptr));
    EXPECT_EQ(builderUp->graphCaptureVariantSignature(),builderSignature);
    builderUp->resetSessionState();
    EXPECT_EQ(builderUp->graphCaptureVariantSignature(),builderSignature);
   }
   std::copy_n(primary,blocks,savedPrimary.data());std::copy_n(residual,blocks,savedResidual.data());
   hc103_original::publish(referenceActivation.data(),live,rank,expectedPrimary.data(),expectedResidual.data());
   ASSERT_EQ(std::memcmp(primary,expectedPrimary.data(),blocks*sizeof(Q8_1Block)),0)<<"primary half-scale/qs/sum";
   ASSERT_EQ(std::memcmp(residual,expectedResidual.data(),blocks*sizeof(Q8_1Block)),0)<<"residual half-scale/qs/sum";
   std::fill(read.begin(),read.end(),-12345.f);std::fill(expectedRead.begin(),expectedRead.end(),-12345.f);
   if(builderPath)std::copy_n(builderRead.data(),read.size(),read.data());
   else primitive->residualRead({extent,hidden,down.input.data(),up.output.data(),read.data()},nullptr);
   std::memset(primary,0xa5,blocks*sizeof(*primary));std::memset(residual,0xa5,blocks*sizeof(*residual));
   std::fill_n(referenceUp.mutable_data(),referenceUp.numel(),-12345.f);
   ASSERT_TRUE(reference.stage->execute(&context));
   ASSERT_EQ(std::memcmp(up.output.data(),referenceUp.data(),up.output.numel()*sizeof(float)),0)<<"up";
   ASSERT_EQ(std::memcmp(primary,savedPrimary.data(),std::size_t(live)*rank/32*sizeof(Q8_1Block)),0);
   ASSERT_EQ(std::memcmp(residual,savedResidual.data(),std::size_t(live)*rank/32*sizeof(Q8_1Block)),0);
   const auto *primaryBytes=reinterpret_cast<const std::uint8_t*>(primary);
   const auto *residualBytes=reinterpret_cast<const std::uint8_t*>(residual);
   for(std::size_t i=std::size_t(live)*rank/32*sizeof(Q8_1Block);i<blocks*sizeof(Q8_1Block);++i) {
    ASSERT_EQ(primaryBytes[i],0xa5);ASSERT_EQ(residualBytes[i],0xa5);
   }
   hc103_original::residualRead({extent,hidden,down.input.data(),referenceUp.data(),expectedRead.data()},nullptr);
   ASSERT_EQ(std::memcmp(read.data(),expectedRead.data(),read.size()*sizeof(float)),0)<<"read";
   for(std::size_t i=std::size_t(live)*rank;i<rawDown.size();++i)ASSERT_EQ(activated.data()[i],-12345.f);
   for(std::size_t i=std::size_t(live)*4*hidden;i<up.output.numel();++i)ASSERT_EQ(up.output.data()[i],-12345.f);
   for(std::size_t i=std::size_t(live)*hidden;i<read.size();++i)ASSERT_EQ(read[i],-12345.f);
   if(!alias)ASSERT_EQ(std::memcmp(down.output.data(),rawDown.data(),rawDown.size()*sizeof(float)),0);
  }
 }
 if(timing) {
  ASSERT_EQ(operation,ChainOperation::FusedStage);
  ASSERT_FALSE(alias);
  ASSERT_GT(timing->workers,0);
  Threads workers(timing->workers);CPUDeviceContext context(DeviceId::cpu(),timing->workers);
  int actualTeam=0;
#pragma omp parallel
  {
#pragma omp single
   actualTeam=omp_get_num_threads();
  }
  ASSERT_EQ(actualTeam,timing->workers);
  down.count=up.count=rows;
  std::copy(down.values.begin(),down.values.end(),down.input.mutable_data());
  FP32Tensor referenceRaw({std::size_t(rows),std::size_t(rank)},DeviceId::cpu());
  const qwen4exp::RowExtent extent{rows,&up.count};
  // Both samples include the same prepared down/up and final read boundaries.
  // All tensor/vector/workspace allocation and team admission precede samples.
  const auto candidate=[&] {
   if(!down.kernel->multiply_tensor(&down.input,&down.output,rows,rank,4*hidden,true,1,0,nullptr,nullptr,-1,down.workspace->get()))
    throw std::runtime_error("HC103 timed candidate down failed");
   if(!control.stage->execute(&context))throw std::runtime_error("HC103 timed fused up failed");
   primitive->residualRead({extent,hidden,down.input.data(),up.output.data(),read.data()},nullptr);
  };
  const auto original=[&] {
   if(!down.kernel->multiply_tensor(&down.input,&referenceRaw,rows,rank,4*hidden,true,1,0,nullptr,nullptr,-1,down.workspace->get()))
    throw std::runtime_error("HC103 timed production unfused down failed");
   primitive->residualDown({extent,rank,referenceRaw.data(),referenceActivation.mutable_data()},nullptr);
   if(!reference.stage->execute(&context))throw std::runtime_error("HC103 timed production unfused up failed");
   primitive->residualRead({extent,hidden,down.input.data(),referenceUp.data(),expectedRead.data()},nullptr);
  };
  // The saved control body remains a numerical oracle only. Timed samples and
  // exact counter intervals must include production validation and dispatch on
  // both sides instead of comparing a compiled primitive with an inline oracle.
  const auto frozenOriginal=[&] {
   if(!down.kernel->multiply_tensor(&down.input,&referenceRaw,rows,rank,4*hidden,true,1,0,nullptr,nullptr,-1,down.workspace->get()))
    throw std::runtime_error("HC103 timed original down failed");
   hc103_original::residualDown({extent,rank,referenceRaw.data(),referenceActivation.mutable_data()},nullptr);
   if(!reference.stage->execute(&context))throw std::runtime_error("HC103 timed original up failed");
   hc103_original::residualRead({extent,hidden,down.input.data(),referenceUp.data(),expectedRead.data()},nullptr);
  };
  std::vector<Q8_1Block> candidatePrimary(blocks),candidateResidual(blocks);
  const auto check=[&] {
   candidate();
   std::copy_n(primary,blocks,candidatePrimary.data());
   std::copy_n(residual,blocks,candidateResidual.data());
   const auto checkReference=[&] {
   EXPECT_EQ(std::memcmp(activated.data(),referenceActivation.data(),activated.numel()*sizeof(float)),0);
   EXPECT_EQ(std::memcmp(up.output.data(),referenceUp.data(),up.output.numel()*sizeof(float)),0);
   EXPECT_EQ(std::memcmp(read.data(),expectedRead.data(),read.size()*sizeof(float)),0);
   hc103_original::publish(referenceActivation.data(),rows,rank,expectedPrimary.data(),expectedResidual.data());
   EXPECT_EQ(std::memcmp(candidatePrimary.data(),expectedPrimary.data(),blocks*sizeof(Q8_1Block)),0)<<"timed candidate primary";
   EXPECT_EQ(std::memcmp(candidateResidual.data(),expectedResidual.data(),blocks*sizeof(Q8_1Block)),0)<<"timed candidate residual";
   EXPECT_EQ(std::memcmp(primary,expectedPrimary.data(),blocks*sizeof(Q8_1Block)),0)<<"timed reference primary";
   EXPECT_EQ(std::memcmp(residual,expectedResidual.data(),blocks*sizeof(Q8_1Block)),0);
   };
   // Candidate FP32 outputs and copied banks survive both independent control
   // invocations. Prove the timed production chain first, then the unchanged
   // frozen chain, using the identical complete byte-comparison body for each.
   original();checkReference();
   frozenOriginal();checkReference();
  };
  check();ASSERT_FALSE(::testing::Test::HasFailure());
  if(timing->observation!=ChainObservation::PairedEvents) {
   using namespace native_vnni_dispatch;
   const bool candidateSide=timing->observation==ChainObservation::ProfileCandidate;
   LinuxPerfControl profiler(profileRequest,profileOutput);
   const auto threadIds=capturePersistentOpenMPThreadIds();
   ASSERT_EQ(threadIds.size(),std::size_t(actualTeam));
   profiler.prepare(threadIds);
   // Open events before warming the selected side; no preparation, snapshots,
   // other variant,32-copy loop or paired samples enter the counter interval.
   for(int warm=0;warm<5;++warm) {
    if(candidateSide)candidate();else original();
   }
   ASSERT_EQ(capturePersistentOpenMPThreadIds(),threadIds);
   std::cout<<"HC103_CPU_PROFILE_CASE,request="<<profileRequest
    <<",side="<<(candidateSide?"candidate":"original")<<",source="<<format.label
    <<",isa="<<int(activeISALevel())<<",M="<<rows<<",H="<<hidden<<",rank="<<rank
    <<",workers="<<actualTeam<<",invocations=1"
    <<",boundary=prepared_down_activation_publication_up_read,normalization=outside_region,control=production_unfused\n";
   if(candidateSide)profileExactOpenMPRegion(profiler,profileRequest,profileOutput,candidate);
   else profileExactOpenMPRegion(profiler,profileRequest,profileOutput,original);
   ASSERT_EQ(capturePersistentOpenMPThreadIds(),threadIds);
  } else {
   for(int warm=0;warm<5;++warm){candidate();original();}
   const int copies=rows==1?32:1;
   const auto sample=[&](const auto& chain) {
    const auto start=std::chrono::steady_clock::now();
    for(int copy=0;copy<copies;++copy)chain();
    return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/copies;
   };
   std::cout<<"HC103_CPU_CANDIDATE_CASE,source="<<format.label<<",isa="<<int(activeISALevel())
    <<",M="<<rows<<",H="<<hidden<<",rank="<<rank<<",workers="<<actualTeam<<",copies="<<copies
    <<",boundary=prepared_down_activation_publication_up_read,normalization=outside_both_samples,control=production_unfused\n";
   for(int pair=0;pair<31;++pair) {
    double candidateUs,originalUs;
    if(pair%2==0){originalUs=sample(original);candidateUs=sample(candidate);}
    else {candidateUs=sample(candidate);originalUs=sample(original);}
    std::cout<<"HC103_CPU_CANDIDATE_PAIR,"<<pair<<","<<(pair%2==0?"AB":"BA")
     <<","<<originalUs<<","<<candidateUs<<"\n";
   }
  }
  check();ASSERT_FALSE(::testing::Test::HasFailure());
 }
 control.stage->resetSessionState();
 EXPECT_EQ(control.stage->graphCaptureVariantSignature(),signature);
 EXPECT_EQ(originalBytes(*down.weight),downRaw);EXPECT_EQ(originalBytes(*up.weight),upRaw);
 EXPECT_EQ(packedDown->interleavedBase(),downAddress);EXPECT_EQ(packedUp->interleavedBase(),upAddress);
 EXPECT_TRUE(std::equal(downPrepared.begin(),downPrepared.end(),downAddress));EXPECT_TRUE(std::equal(upPrepared.begin(),upPrepared.end(),upAddress));
 if(timing&&timing->observation!=ChainObservation::PairedEvents) {
  ASSERT_FALSE(::testing::Test::HasFailure());
  std::cout<<"HC103_CPU_PROFILE_COMPLETE,request="<<profileRequest
   <<",invocations=1,activation_banks_up_read=byte_exact,source_conservation=byte_exact\n";
 }
 std::cout<<"HC103_CPU_CONTROL source="<<format.label<<" isa="<<int(activeISALevel())<<" rows="<<rows<<" H="<<hidden<<" rank="<<rank<<" alias="<<alias<<" operation="<<int(operation)<<" activation_banks_up_read=byte_exact\n";
}
} // namespace llaminar2::test::hc103_cpu
