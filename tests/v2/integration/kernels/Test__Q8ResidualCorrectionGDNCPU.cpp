/**
 * @file Test__Q8ResidualCorrectionGDNCPU.cpp
 * @brief Explicit Integration-only original-source GDN output correction proof.
 *
 * Model-sized N2560/K6144 work is deliberately absent from Unit targets. Actual
 * prepared native Q8_0 weights and GEMMStage use one PMA-admitted invocation
 * workspace. One reusable M1 pair supplies the independent serial-byte oracle;
 * all retained-prefix replays poison inactive inputs, outputs and both Q8 banks.
 * Functional evidence here is independent of later clean-window timing.
 */
#include <gtest/gtest.h>
#include "../../utils/Qwen4ExpFAOutputStageFixture.h"
#include "backends/BackendManager.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include "loaders/PreparedWeightStore.h"
#include "tensors/Tensors.h"
#include "../../utils/PreparedWeightTestHarness.h"
#include "../../utils/CPUProjectionTestWorkspace.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>
using namespace llaminar2;
namespace
{
/** @brief Original native Q8 blocks; preparation may pack but never requantize these source bits. */
std::vector<uint8_t> sourceBytes(int n,int k)
{
 std::vector<uint8_t>raw(std::size_t(n)*k/32*sizeof(Q8_0Block));
 for(std::size_t b=0;b<raw.size()/sizeof(Q8_0Block);++b)
 {
  Q8_0Block block{};block.d=fp32_to_fp16(.002f*float(1+b%11));
  for(int c=0;c<32;++c)block.qs[c]=static_cast<int8_t>((b*13+c*17)%255-127);
  std::memcpy(raw.data()+b*sizeof(block),&block,sizeof(block));
 }
 return raw;
}
/** @brief Revoke the stage workspace borrow before its admitted physical owner retires. */
struct StageWorkspaceBorrow final
{
 GEMMStage &stage;
 /** @brief Borrow only the exact declared stage workspace for this invocation lifetime. */
 StageWorkspaceBorrow(GEMMStage &s,DeviceWorkspaceManager *workspace):stage(s)
 {
  try {stage.bindWorkspace(workspace);}
  catch(...) {stage.unbindWorkspace();throw;}
 }
 /** @brief Revoke the borrow before workspace destruction. */
 ~StageWorkspaceBorrow(){stage.unbindWorkspace();}
 StageWorkspaceBorrow(const StageWorkspaceBorrow &)=delete;
 StageWorkspaceBorrow &operator=(const StageWorkspaceBorrow &)=delete;
};
}

/**
 * @brief Execute standalone GDN or the actual FA composer result with original-source serial-byte oracles.
 * @param fa_output Select actual FA Stage5 construction; false preserves standalone GDN coverage.
 *
 * The gate is declared as the producer; execution isolates projection on already
 * gated source values. CPU construction proves a compact query, then retained
 * count changes exercise the stage API rather than advertising padded CPU graphs.
 */
void provePreparedOutput(bool fa_output)
{
 if(!hasCPUBackend())initCPUBackend(-1);
 // Direct NativeVNNI workshares use the canonical CTest OpenMP physical-core
 // budget. DeviceContext does not set that team; no fixture-local cap is applied.
 // Timing/scaling remains a separate clean-window experiment.
 auto context=IDeviceContext::create(DeviceId::cpu());
 constexpr int n=2560,k=6144;
 for(const int m:std::array{1,15,512,1024,2048})
 {
  SCOPED_TRACE(::testing::Message()<<"CPU GDN M="<<m<<" N="<<n<<" K="<<k);
  auto raw=sourceBytes(n,k);Q8_0Tensor source({std::size_t(n),std::size_t(k)},raw);
  PreparedWeightStore store{ModelContextId{801946}};
  auto binding=test::makePreparedWeightTestBinding(&source,DeviceId::cpu(),fa_output ? "blk.3.attn_output.weight" : "blk.0.ssm_out.weight",store.modelId());
  if(fa_output)binding.slice={.source_rows=n,.source_cols=k,.row_count=n,.col_count=k};
  binding.prepared=store.prepareGemm(binding);auto *kernel=store.gemmKernel(*binding.prepared);ASSERT_NE(kernel,nullptr);
  const auto *packed=kernel->exportCPUNativeVNNIPackedWeights();ASSERT_NE(packed,nullptr);
  const auto *sourcePointer=packed->native_interleaved.data();
  const std::vector<uint8_t>packedBefore(packed->native_interleaved.begin(),packed->native_interleaved.end());
  FP32Tensor input({std::size_t(m),std::size_t(k)},DeviceId::cpu()),output({std::size_t(m),std::size_t(n)},DeviceId::cpu());
  FP32Tensor ordinaryOutput({std::size_t(m),std::size_t(n)},DeviceId::cpu());
  FP32Tensor serialInput({1,std::size_t(k)},DeviceId::cpu()),serialOutput({1,std::size_t(n)},DeviceId::cpu());
  std::vector<float>original(std::size_t(m)*k),expected(std::size_t(m)*n);
  for(int row=0;row<m;++row)for(int c=0;c<k;++c)
  {
   float value=c%32==0?float(row%7+1)*17.25f:std::sin(float(c*7+row*11))*.31f;
   if(row%17==0&&c<32)value=0;
   if(row%17==1&&c<32)value=std::numeric_limits<float>::denorm_min()*float(c+1);
   if(row%17==2&&c<32)value=c==0?127.f:float(c%11-5)+.5f;
   original[std::size_t(row)*k+c]=value;
  }
  std::copy(original.begin(),original.end(),input.mutable_data());
  int32_t live=m;
  std::vector<int> originalLengths{m};
  ComputeGraph faGraph;
  std::unique_ptr<test::Qwen4ExpFAOutputStageBuilder>faBuilder;
  std::unique_ptr<FP32Tensor>faGate;
  std::unique_ptr<GEMMStage>legacyStage;
  GEMMStage *actualStage=nullptr;
  if(fa_output)
  {
   GraphConfig config;Qwen4ExpGraphConfigBuilder::populateFromDirectory(test::qwen4MetadataFixture(false),config);
   config.default_device=DeviceId::cpu();config.qfn_hc_up_activation=QuantizedActivationOperandPrecision::SingleQ8;
   config.qfn_gdn_output_activation=QuantizedActivationOperandPrecision::SingleQ8;
   config.qfn_fa_output_activation=QuantizedActivationOperandPrecision::TwoTermQ8;
   faBuilder=std::make_unique<test::Qwen4ExpFAOutputStageBuilder>(config);faBuilder->setPreparedWeightStore(&store);
   faGate=std::make_unique<FP32Tensor>(std::vector<std::size_t>{std::size_t(m),std::size_t(k)});
   faGraph.addNode("layer3_attn_output_gate",std::make_unique<AttentionOutputGateStage>(AttentionOutputGateStage::Params{
    .device_id=DeviceId::cpu(),.input=&input,.gate=faGate.get(),.output=&input,.seq_len=m,.feature_dim=k,
    .input_buffer_id=BufferId::ATTN_OUTPUT,.gate_buffer_id=BufferId::FA_GATE,.output_buffer_id=BufferId::ATTN_OUTPUT}),DeviceId::cpu());
   ForwardInput invocation;invocation.device=DeviceId::cpu();invocation.batch_size=1;invocation.seq_len=m;
   invocation.real_seq_len=m;invocation.execution_phase=ForwardExecutionPhase::Prefill;invocation.sequence_lengths=&originalLengths;
   ActivationBuffers buffers;buffers.attn_output=&input;buffers.attn_proj=&output;
   ASSERT_EQ(faBuilder->append(faGraph,invocation,buffers,binding),"layer3_wo_proj");
   actualStage=dynamic_cast<GEMMStage*>(faGraph.getNode("layer3_wo_proj")->stage.get());ASSERT_NE(actualStage,nullptr);
   ASSERT_TRUE(actualStage->getParams().verifier_row_range);
   ASSERT_EQ(actualStage->getParams().verifier_row_range->countOwner(),originalLengths.data());
   EXPECT_EQ(actualStage->getParams().B,&source);EXPECT_EQ(actualStage->getParams().prepared_store,&store);
   ASSERT_TRUE(actualStage->getParams().prepared_ref);
   EXPECT_EQ(actualStage->getParams().prepared_ref->binding_id,binding.prepared->binding_id);
   EXPECT_EQ(actualStage->getParams().force_decode_equivalent_verifier_prefill,m>1);
  }
  else
  {
   legacyStage=std::make_unique<GEMMStage>(GEMMStage::Params{.device_id=DeviceId::cpu(),.A=&input,.B=&source,.C=&output,.m=m,.n=n,.k=k,
    .alpha=1,.beta=0,.transpose_B=false,.a_buffer_id=BufferId::NORMALIZED,.c_buffer_id=BufferId::ATTN_PROJ,
    .force_decode_equivalent_verifier_prefill=m==15,.verifier_row_range=DeviceRowRange::deviceCounted(m,&live),
    .prepared_ref=binding.prepared,.prepared_store=&store,.quantized_operand_precision=QuantizedActivationOperandPrecision::TwoTermQ8});
   actualStage=legacyStage.get();
  }
  auto &stage=*actualStage;
  auto requirements=stage.getWorkspaceRequirements(m,n,k);ASSERT_EQ(requirements.buffers.size(),2u);
  auto *consumer=dynamic_cast<IWorkspaceConsumer*>(kernel);ASSERT_NE(consumer,nullptr);
  requirements.merge(consumer->getWorkspaceRequirements(m,n,k));
  test::CPUProjectionTestWorkspace workspace(requirements);StageWorkspaceBorrow borrow(stage,workspace.get());
  auto *primary=static_cast<Q8_1Block*>(workspace.get()->getBuffer(kCPUProjectionQ8));
  auto *residual=static_cast<Q8_1Block*>(workspace.get()->getBuffer(kCPUProjectionResidualQ8));
  ASSERT_NE(primary,nullptr);ASSERT_NE(residual,nullptr);
  const std::size_t blocks=std::size_t(m)*k/32;
  ASSERT_TRUE(kernel->multiply_tensor(&input,&ordinaryOutput,m,n,k,true,1,0,nullptr,nullptr,-1,workspace.get()));
  const std::vector<Q8_1Block>ordinary(primary,primary+blocks);
  const std::vector<float>ordinaryBefore(ordinaryOutput.data(),ordinaryOutput.data()+ordinaryOutput.numel());
  // The scalar oracle reuses one host pair and admitted bank, never a separate
  // reference implementation or a production row-replay path.
  for(int row=0;row<m;++row)
  {
   std::copy_n(original.data()+std::size_t(row)*k,k,serialInput.mutable_data());
   ASSERT_TRUE(kernel->multiply_tensor(&serialInput,&serialOutput,1,n,k,true,1,0,nullptr,nullptr,-1,workspace.get()));
   ASSERT_EQ(std::memcmp(primary,ordinary.data()+std::size_t(row)*k/32,std::size_t(k)/32*sizeof(Q8_1Block)),0);
   kernel->multiply_q8_residual_corrected(&serialInput,&serialOutput,DeviceRowRange::fullyActive(1),n,k,workspace.get());
   std::copy_n(serialOutput.data(),n,expected.data()+std::size_t(row)*n);
  }
  const auto signature=stage.graphCaptureVariantSignature(),used=workspace.get()->used();ASSERT_NE(signature,0u);
  const std::vector<std::byte>dirty(blocks*sizeof(Q8_1Block),std::byte{0x5a});
  for(const int next:std::array{m,std::min(9,m),1,0,std::min(15,m),m,0})
  {
   SCOPED_TRACE(::testing::Message()<<"live="<<next);
   live=next;originalLengths[0]=next;std::copy(original.begin(),original.end(),input.mutable_data());
   std::fill_n(input.mutable_data()+std::size_t(live)*k,std::size_t(m-live)*k,std::numeric_limits<float>::quiet_NaN());
   std::fill_n(output.mutable_data(),output.numel(),-12345.f);
   std::memcpy(primary,dirty.data(),dirty.size());std::memcpy(residual,dirty.data(),dirty.size());
   ASSERT_TRUE(stage.execute(context.get()));
   if(live)
   {
    EXPECT_EQ(std::memcmp(output.data(),expected.data(),std::size_t(live)*n*sizeof(float)),0);
    EXPECT_EQ(std::memcmp(primary,ordinary.data(),std::size_t(live)*k/32*sizeof(Q8_1Block)),0);
   }
   EXPECT_TRUE(std::all_of(output.data()+std::size_t(live)*n,output.data()+output.numel(),[](float x){return x==-12345.f;}));
   const auto activeBytes=std::size_t(live)*k/32*sizeof(Q8_1Block),tailBytes=dirty.size()-activeBytes;
   if(tailBytes)
   {
    EXPECT_EQ(std::memcmp(reinterpret_cast<const std::byte*>(primary)+activeBytes,dirty.data()+activeBytes,tailBytes),0);
    EXPECT_EQ(std::memcmp(reinterpret_cast<const std::byte*>(residual)+activeBytes,dirty.data()+activeBytes,tailBytes),0);
   }
   EXPECT_EQ(stage.graphCaptureVariantSignature(),signature);EXPECT_EQ(workspace.get()->used(),used);
   EXPECT_EQ(workspace.get()->getBuffer(kCPUProjectionQ8),primary);EXPECT_EQ(workspace.get()->getBuffer(kCPUProjectionResidualQ8),residual);
  }
  // Revisit ordinary execution with the original live source; correction may
  // neither alter the immutable source nor contaminate later primary dispatch.
  std::copy(original.begin(),original.end(),input.mutable_data());
  ASSERT_TRUE(kernel->multiply_tensor(&input,&ordinaryOutput,m,n,k,true,1,0,nullptr,nullptr,-1,workspace.get()));
  EXPECT_EQ(std::memcmp(primary,ordinary.data(),blocks*sizeof(Q8_1Block)),0);
  EXPECT_EQ(std::memcmp(ordinaryBefore.data(),ordinaryOutput.data(),ordinaryBefore.size()*sizeof(float)),0);
  EXPECT_EQ(packed->native_interleaved.data(),sourcePointer);
  EXPECT_TRUE(std::equal(packedBefore.begin(),packedBefore.end(),packed->native_interleaved.begin()));
  EXPECT_EQ(std::memcmp(raw.data(),source.raw_data(),raw.size()),0);EXPECT_EQ(store.gemmKernel(*binding.prepared),kernel);
 }
}

/** @test Existing standalone original GDN arithmetic coverage remains unchanged. */
TEST(Q8ResidualCorrectionGDNCPU, ActualPreparedStageRetainsOriginalSourceAndLiveRows)
{
 provePreparedOutput(false);
}
/** @test Actual FA composer result executes exact serial bytes with the original compact CPU authority. */
TEST(Q8ResidualCorrectionGDNCPU, FAActualComposerRetainsOriginalSourceAndLiveRows)
{
 provePreparedOutput(true);
}
