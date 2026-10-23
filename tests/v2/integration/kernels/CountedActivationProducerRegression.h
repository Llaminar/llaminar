/**
 * @file CountedActivationProducerRegression.h
 * @brief Direct original/full-active versus counted producer admission and replay.
 *
 * Included by the existing all-format device fixture: its Storage and graph
 * retirement owners remain the only allocation/stream/test lifecycle. This
 * isolates the producer boundary from prepared projection/epilogue adapters.
 * Original and counted launches use identical source values and original math;
 * inactive capacity and guard bytes must retain their authored poison.
 */
#pragma once
#include "utils/DebugEnv.h"
#include <array>
#include <cstdint>
#include <limits>
#include <utility>
namespace {
/** @brief Explicit existing producer operations, never an alternate precision mode. */
enum class DirectProducer { Plain, Sums, SwiGLU, SwiGLUSums, PlainViaNullSums, SwiGLUViaNullSums };
/** @brief Call canonical backend declarations; ordinary ABI is the full-active oracle. */
bool submitDirectProducer(DeviceId device,DirectProducer kind,bool counted,
 const float* gate,const float* up,int8_t* bytes,float* scales,int32_t* sums,
 int m,int k,int block,void* stream,const DeviceRowRange* rows) {
 if(kind==DirectProducer::PlainViaNullSums||kind==DirectProducer::SwiGLUViaNullSums)sums=nullptr;
#ifdef HAVE_CUDA
 if(device.is_cuda()) {
  switch(kind) {
  case DirectProducer::Plain:
   return counted?cudaQuantGemm_quantizeActivationsBlockwise_rows(gate,bytes,scales,m,k,device.ordinal,stream,rows):cudaQuantGemm_quantizeActivationsBlockwise(gate,bytes,scales,m,k,device.ordinal,stream);
  case DirectProducer::PlainViaNullSums:
  case DirectProducer::Sums:
   return counted?cudaQuantGemm_quantizeActivationsBlockwiseWithSums_rows(gate,bytes,scales,sums,m,k,device.ordinal,stream,rows):cudaQuantGemm_quantizeActivationsBlockwiseWithSums(gate,bytes,scales,sums,m,k,device.ordinal,stream);
  case DirectProducer::SwiGLU:
   return counted?cudaOps_fused_swiglu_quantize_blockwise_rows(gate,up,bytes,scales,m,k,device.ordinal,stream,rows):cudaOps_fused_swiglu_quantize_blockwise(gate,up,bytes,scales,m,k,device.ordinal,stream);
  case DirectProducer::SwiGLUViaNullSums:
  case DirectProducer::SwiGLUSums:
   return counted?cudaOps_fused_swiglu_quantize_blockwise_with_sums_rows(gate,up,bytes,scales,sums,m,k,device.ordinal,stream,rows):cudaOps_fused_swiglu_quantize_blockwise_with_sums(gate,up,bytes,scales,sums,m,k,device.ordinal,stream);
  }
 }
#endif
#ifdef HAVE_ROCM
 if(device.is_rocm()) {
  switch(kind) {
  case DirectProducer::Plain:
   return counted?rocmQuantGemm_quantizeActivationsBlockwise_rows(gate,bytes,scales,m,k,device.ordinal,stream,block,rows):rocmQuantGemm_quantizeActivationsBlockwise(gate,bytes,scales,m,k,device.ordinal,stream,block);
  case DirectProducer::PlainViaNullSums:
  case DirectProducer::Sums:
   return counted?rocmQuantGemm_quantizeActivationsBlockwiseWithSums_rows(gate,bytes,scales,sums,m,k,device.ordinal,stream,block,rows):rocmQuantGemm_quantizeActivationsBlockwiseWithSums(gate,bytes,scales,sums,m,k,device.ordinal,stream,block);
  case DirectProducer::SwiGLU:
   return counted?hipOps_fused_swiglu_quantize_blockwise_rows(gate,up,bytes,scales,m,k,device.ordinal,stream,rows):hipOps_fused_swiglu_quantize_blockwise(gate,up,bytes,scales,m,k,device.ordinal,stream);
  case DirectProducer::SwiGLUViaNullSums:
  case DirectProducer::SwiGLUSums:
   return counted?hipOps_fused_swiglu_quantize_blockwise_with_sums_rows(gate,up,bytes,scales,sums,m,k,device.ordinal,stream,rows):hipOps_fused_swiglu_quantize_blockwise_with_sums(gate,up,bytes,scales,sums,m,k,device.ordinal,stream);
  }
 }
#endif
 throw std::invalid_argument("Unknown direct producer backend/operation");
}
/** @brief Invalid ordinal proves malformed admission rejects before device selection. */
void proveDirectProducerAdmission(DeviceId device) {
 int hostCount=3;const auto rows=DeviceRowRange::deviceCounted(3,&hostCount);
 for(auto kind:{DirectProducer::Plain,DirectProducer::Sums,DirectProducer::SwiGLU,DirectProducer::SwiGLUSums,DirectProducer::PlainViaNullSums,DirectProducer::SwiGLUViaNullSums}) {
  SCOPED_TRACE(int(kind));
  // Non-null sums reaches its own submitting bridge rather than optional delegation.
  auto* sum=reinterpret_cast<int32_t*>(std::uintptr_t(16));
  for(std::uintptr_t stream:{0u,1u,2u}) {
   EXPECT_THROW(submitDirectProducer(device,kind,true,nullptr,nullptr,nullptr,nullptr,sum,3,256,32,reinterpret_cast<void*>(stream),&rows),std::invalid_argument);
  }
  for(const auto shape:{std::pair{0,256},std::pair{-1,256},std::pair{3,0},std::pair{3,-1}}) {
   EXPECT_THROW(submitDirectProducer(device,kind,true,nullptr,nullptr,nullptr,nullptr,sum,shape.first,shape.second,32,reinterpret_cast<void*>(std::uintptr_t(3)),&rows),std::invalid_argument);
  }
  EXPECT_THROW(submitDirectProducer(device,kind,true,nullptr,nullptr,nullptr,nullptr,sum,1,256,32,reinterpret_cast<void*>(std::uintptr_t(3)),&rows),std::invalid_argument);
 }
}
/** @brief Restore the existing HIP diagnostic dispatch selector after a test case. */
struct DirectProducerVariant {
 int saved=mutableDebugEnv().rocm.blockwise_quant_variant;
 bool owns;
 DirectProducerVariant(DeviceId device,int variant):owns(device.is_rocm()){if(owns)mutableDebugEnv().rocm.blockwise_quant_variant=variant;}
 ~DirectProducerVariant(){if(owns)mutableDebugEnv().rocm.blockwise_quant_variant=saved;}
};
/**
 * @brief Compare every retained bank byte with the same original full-active producer.
 * @param device Actual backend participant, with an existing exact worker stream.
 * @param kind Plain/sums/SwiGLU existing publication program.
 * @param m Retained physical rows, including meaningful counted M1 empty replay.
 * @param k Original admitted source width, including HIP scalar/register tails.
 * @param block Original plain HIP block width32/64/128.
 * @param variant Existing HIP diagnostic selection; zero retains automatic policy.
 * @param unaligned Exercise V6's original scalar source/store path through legal views.
 */
void proveDirectProducerRows(DeviceId device,DirectProducer kind,int m,int k,int block,int variant,bool unaligned) {
 auto* backend=getBackendFor(device);require(backend&&backend->deviceCount()>device.ordinal,"Direct producer backend absent");
 auto& worker=GPUDeviceContextPool::instance().getContext(device);
 worker.submitAsync([&] {
  SCOPED_TRACE(::testing::Message()<<device.toString()<<" kind="<<int(kind)<<" M="<<m<<" K="<<k<<" block="<<block<<" variant="<<variant<<" unaligned="<<unaligned);
  DirectProducerVariant selected(device,variant);
  const int bpr=(k+block-1)/block,offset=unaligned?1:0;
  const std::size_t data=std::size_t(m)*k,byteCapacity=std::size_t(m+1)*k+offset;
  const std::size_t bankData=std::size_t(m)*bpr,bankCapacity=std::size_t(m+1)*bpr;
  // Borrowed DMA vectors precede the native owner on every unwind path.
  std::vector<float> gate(std::size_t(m+1)*k+offset),up(gate.size());
  for(std::size_t i=0;i<gate.size();++i){gate[i]=std::sin(float(i*7+11))*.31f;up[i]=.125f+std::cos(float(i*13+3))*.27f;}
  std::vector<int8_t> bytePoison(byteCapacity,int8_t(-87));
  std::vector<float> scalePoison(bankCapacity,-987.25f);
  std::vector<int32_t> sumPoison(bankCapacity,0x6b6b6b6b),liveHost(1,m);
  Storage s{device,backend,ExplicitGPUStream(worker.defaultStream()).get()};
  const auto* dg=s.input(gate)+offset;const auto* du=s.input(up)+offset;
  auto* count=s.input(liveHost);const auto rows=DeviceRowRange::deviceCounted(m,count);
  std::array<int8_t*,2> bytes{s.allocate<int8_t>(byteCapacity),s.allocate<int8_t>(byteCapacity)};
  std::array<float*,2> scales{s.allocate<float>(bankCapacity),s.allocate<float>(bankCapacity)};
  std::array<int32_t*,2> sums{s.allocate<int32_t>(bankCapacity),s.allocate<int32_t>(bankCapacity)};
  std::unique_ptr<IGPUGraphCapture> original;
  CapturedWorkRetirement retirement(s);
  const bool withSums=kind==DirectProducer::Sums||kind==DirectProducer::SwiGLUSums;
  auto capture=[&](std::unique_ptr<IGPUGraphCapture>& graph,int side) {
   graph=worker.createGraphCapture(s.stream);require(bool(graph),"Direct producer native graph missing");
   ScopedBackendGraphCapture recording(worker,*graph,"Counted direct quantizer");
   require(recording.begin(),"Direct producer capture begin failed");
   require(submitDirectProducer(device,kind,side==0,dg,du,bytes[side]+offset,scales[side],withSums?sums[side]:nullptr,m,k,block,s.stream,side==0?&rows:nullptr),"Direct producer enqueue failed");
   recording.finish();require(graph->instantiate(),"Direct producer instantiate failed");
   std::vector<GPUGraphKernelNodeInfo> nodes;std::string why;
   require(graph->inspectKernelNodes(nodes,&why),"Direct producer kernel inventory failed");
   ASSERT_EQ(nodes.size(),1u);ASSERT_EQ(graph->nodeCount(),nodes.size());
   ASSERT_TRUE(nodes[0].name_resolved);
   EXPECT_NE(nodes[0].name.find("quantize"),std::string::npos);
   EXPECT_EQ(nodes[0].local_memory_bytes_per_thread,0u);
   const bool swiglu=kind==DirectProducer::SwiGLU||kind==DirectProducer::SwiGLUSums||kind==DirectProducer::SwiGLUViaNullSums;
   std::string family;
   if(swiglu)family="fused_swiglu_quantize_blockwise_kernel";
   else if(device.is_cuda())family="quantize_activations_blockwise_kernel";
   else if(block!=32)family="quantizeActivationsQ8_blockwise_generic_kernel";
   else if(withSums)family=m>16?"quantizeActivationsQ8_register_blocks_kernel":"quantizeActivationsQ8_blockwise_v4_with_sums_kernel";
   else {
    const int actual=variant?variant:(m<=1?4:m<=16?2:6);
    family=actual==6?"quantizeActivationsQ8_register_blocks_kernel":actual==1?"quantizeActivationsQ8_blockwise_kernel":"quantizeActivationsQ8_blockwise_v"+std::to_string(actual)+"_kernel";
   }
   EXPECT_NE(nodes[0].name.find(family),std::string::npos)<<nodes[0].name;
  };
  capture(s.graph,0);capture(original,1);ASSERT_FALSE(::testing::Test::HasFailure());
  for(int side=0;side<2;++side){s.upload(bytes[side],bytePoison);s.upload(scales[side],scalePoison);s.upload(sums[side],sumPoison);}
  require(original->launch(),"Original full-active producer replay failed");
  const auto goldBytes=s.download(bytes[1],byteCapacity);const auto goldScales=s.download(scales[1],bankCapacity);const auto goldSums=s.download(sums[1],bankCapacity);
  EXPECT_NE(goldBytes,bytePoison);EXPECT_NE(goldScales,scalePoison);
  for(int replay=0;replay<20;++replay) {
   const int live=std::array{m,1,0,std::min(m,3),0,std::max(0,m-1),m}[replay%7];
   liveHost[0]=live;s.upload(count,liveHost);s.upload(bytes[0],bytePoison);s.upload(scales[0],scalePoison);s.upload(sums[0],sumPoison);
   require(s.graph->launch(),"Counted direct producer replay failed");
   const auto gotBytes=s.download(bytes[0],byteCapacity);const auto gotScales=s.download(scales[0],bankCapacity);const auto gotSums=s.download(sums[0],bankCapacity);
   for(std::size_t i=0;i<byteCapacity;++i) {
    const bool active=i>=std::size_t(offset)&&i<std::size_t(offset)+data&&(i-offset)/k<std::size_t(live);
    ASSERT_EQ(gotBytes[i],active?goldBytes[i]:bytePoison[i])<<"byte="<<i<<" live="<<live;
   }
   for(std::size_t i=0;i<bankCapacity;++i) {
    const bool scaleActive=i<bankData&&i/bpr<std::size_t(live);
    ASSERT_EQ(std::memcmp(&gotScales[i],scaleActive?&goldScales[i]:&scalePoison[i],sizeof(float)),0)<<"scale="<<i;
    const std::size_t sumRow=device.is_cuda()?i%std::size_t(m):i/bpr;
    const bool sumActive=withSums&&i<bankData&&sumRow<std::size_t(live);
    ASSERT_EQ(gotSums[i],sumActive?goldSums[i]:sumPoison[i])<<"sum="<<i;
   }
  }
 }).get();
}
/** @brief Cover original producer leaves without changing their numerical programs. */
void proveDirectProducerCatalog(DeviceId device) {
 for(auto kind:{DirectProducer::Plain,DirectProducer::Sums,DirectProducer::SwiGLU,DirectProducer::SwiGLUSums,DirectProducer::PlainViaNullSums,DirectProducer::SwiGLUViaNullSums})
  for(int m:{1,3,17}) {
   proveDirectProducerRows(device,kind,m,256,32,0,false);
   ASSERT_FALSE(::testing::Test::HasFailure());
  }
 if(device.is_rocm()) {
  for(int variant:{1,2,3,4,5}) {
   proveDirectProducerRows(device,DirectProducer::Plain,3,256,32,variant,false);
   ASSERT_FALSE(::testing::Test::HasFailure());
   if(variant!=5) {
    // Nine blocks per row makes V2's row boundary cut a32-lane subgroup
    // within an otherwise live wave/workgroup, without crossing a CTA barrier.
    proveDirectProducerRows(device,DirectProducer::Plain,3,257,32,variant,false);
    ASSERT_FALSE(::testing::Test::HasFailure());
   }
  }
  // Automatic M17 selects native register V6; K257+unaligned pointers exercise
  // its scalar source/store tail rather than invoking unsupported V5 views.
  for(auto kind:{DirectProducer::Plain,DirectProducer::Sums}) {
   proveDirectProducerRows(device,kind,17,257,32,0,true);
   ASSERT_FALSE(::testing::Test::HasFailure());
   proveDirectProducerRows(device,kind,17,260,32,0,false);
   ASSERT_FALSE(::testing::Test::HasFailure());
  }
  for(int block:{64,128}) {
   proveDirectProducerRows(device,DirectProducer::Plain,3,257,block,0,false);
   ASSERT_FALSE(::testing::Test::HasFailure());
  }
 }
}
} // namespace
#ifdef HAVE_CUDA
/** @test CUDA producer host admission rejects before device/API submit. */
TEST(Q8ResidualAllFormatsDevice,CUDADirectProducerAdmission){proveDirectProducerAdmission(DeviceId::cuda(std::numeric_limits<int>::max()));}
/** @test CUDA direct producer full/partial/empty histories retain every bank/guard byte. */
TEST(Q8ResidualAllFormatsDevice,CUDADirectProducerRows){proveDirectProducerCatalog(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
/** @test HIP producer host admission rejects before device/API submit. */
TEST(Q8ResidualAllFormatsDevice,ROCmDirectProducerAdmission){proveDirectProducerAdmission(DeviceId::rocm(std::numeric_limits<int>::max()));}
/** @test HIP variants and legal tails retain every inactive producer bank/guard byte. */
TEST(Q8ResidualAllFormatsDevice,ROCmDirectProducerRows){proveDirectProducerCatalog(DeviceId::rocm(0));}
#endif
