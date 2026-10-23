/**
 * @file HC103CPUControlChain.h
 * @brief Real prepared down/TwoTerm-up CPU chain calibrated against frozen arithmetic.
 *
 * The independent oracle owns nonlinear and publication math. Both chains retain
 * the same original prepared projection source, numerical policy and workspace.
 * No replacement projection or inference path is supplied. All storage belongs
 * to this test invocation; stages retire their workspace borrow before arenas.
 */
#pragma once
#include "Q8ResidualAllFormatCPUHarness.h"
#include "HC103OriginalCPU.h"
#include <bit>
namespace llaminar2::test::hc103_cpu {
using namespace q8_all_formats;
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
 */
inline void prove(const QuantizedVerifierFormatCase &format,int rows,int hidden,int rank,
 std::span<const int> teams,bool alias) {
 SCOPED_TRACE(::testing::Message()<<format.label<<" rows="<<rows<<" H="<<hidden<<" rank="<<rank<<" alias="<<alias);
 ASSERT_FALSE(teams.empty());Threads capacity(*std::max_element(teams.begin(),teams.end()));
 CPUFixture down(format,rows,rank,4*hidden,false),up(format,rows,4*hidden,rank,false);
 auto primitive=qwen4exp::createCPUQwen4ExpKernels();ASSERT_NE(primitive,nullptr);
 FP32Tensor referenceActivation({std::size_t(rows),std::size_t(rank)},DeviceId::cpu());
 FP32Tensor referenceUp({std::size_t(rows),std::size_t(4*hidden)},DeviceId::cpu());
 FP32Tensor &activated=alias?down.output:up.input;
 StageBorrow control(upStage(up,activated,up.output));
 StageBorrow reference(upStage(up,referenceActivation,referenceUp));
 auto *primary=static_cast<Q8_1Block*>(up.workspace->get()->getBuffer(kCPUProjectionQ8));
 auto *residual=static_cast<Q8_1Block*>(up.workspace->get()->getBuffer(kCPUProjectionResidualQ8));
 ASSERT_NE(primary,nullptr);ASSERT_NE(residual,nullptr);
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
   std::copy(down.values.begin(),down.values.end(),down.input.mutable_data());
   std::fill(down.input.mutable_data()+std::size_t(live)*4*hidden,down.input.mutable_data()+down.input.numel(),std::numeric_limits<float>::quiet_NaN());
   std::fill_n(down.output.mutable_data(),down.output.numel(),-12345.f);
   if(live)ASSERT_TRUE(down.kernel->multiply_tensor(&down.input,&down.output,live,rank,4*hidden,true,1,0,nullptr,nullptr,-1,down.workspace->get()));
   std::copy_n(down.output.data(),rawDown.size(),rawDown.data());
   std::fill_n(referenceActivation.mutable_data(),referenceActivation.numel(),-12345.f);
   if(!alias)std::fill_n(activated.mutable_data(),activated.numel(),-12345.f);
   const qwen4exp::RowExtent extent{rows,&up.count};
   primitive->residualDown({extent,rank,down.output.data(),activated.mutable_data()},nullptr);
   hc103_original::residualDown({extent,rank,rawDown.data(),referenceActivation.mutable_data()},nullptr);
   ASSERT_EQ(std::memcmp(activated.data(),referenceActivation.data(),rawDown.size()*sizeof(float)),0)<<"activated";
   std::memset(primary,0x5a,blocks*sizeof(*primary));std::memset(residual,0x5a,blocks*sizeof(*residual));
   std::memset(expectedPrimary.data(),0x5a,blocks*sizeof(Q8_1Block));std::memset(expectedResidual.data(),0x5a,blocks*sizeof(Q8_1Block));
   std::fill_n(up.output.mutable_data(),up.output.numel(),-12345.f);
   ASSERT_TRUE(control.stage->execute(&context));
   std::copy_n(primary,blocks,savedPrimary.data());std::copy_n(residual,blocks,savedResidual.data());
   hc103_original::publish(referenceActivation.data(),live,rank,expectedPrimary.data(),expectedResidual.data());
   ASSERT_EQ(std::memcmp(primary,expectedPrimary.data(),blocks*sizeof(Q8_1Block)),0)<<"primary half-scale/qs/sum";
   ASSERT_EQ(std::memcmp(residual,expectedResidual.data(),blocks*sizeof(Q8_1Block)),0)<<"residual half-scale/qs/sum";
   std::fill(read.begin(),read.end(),-12345.f);std::fill(expectedRead.begin(),expectedRead.end(),-12345.f);
   primitive->residualRead({extent,hidden,down.input.data(),up.output.data(),read.data()},nullptr);
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
 EXPECT_EQ(originalBytes(*down.weight),downRaw);EXPECT_EQ(originalBytes(*up.weight),upRaw);
 EXPECT_EQ(packedDown->interleavedBase(),downAddress);EXPECT_EQ(packedUp->interleavedBase(),upAddress);
 EXPECT_TRUE(std::equal(downPrepared.begin(),downPrepared.end(),downAddress));EXPECT_TRUE(std::equal(upPrepared.begin(),upPrepared.end(),upAddress));
 std::cout<<"HC103_CPU_CONTROL source="<<format.label<<" isa="<<int(activeISALevel())<<" rows="<<rows<<" H="<<hidden<<" rank="<<rank<<" alias="<<alias<<" activation_banks_up_read=byte_exact\n";
}
} // namespace llaminar2::test::hc103_cpu
