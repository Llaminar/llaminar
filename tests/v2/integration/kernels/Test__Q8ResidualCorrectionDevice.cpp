/**
 * @file Test__Q8ResidualCorrectionDevice.cpp
 * @brief Captured prepared-engine/stage proof of native two-term Q8, including actual FA Stage5 composition.
 *
 * Production packers preserve the native Q8_0 payload/scale bits. Their exact
 * device pointers enter a normal PreparedWeightStore handle and GEMMStage.
 * One retained graph replays large/small/empty live prefixes without recapture.
 * Independent prepared serial calls are the byte oracle; actual SingleQ8
 * dispatches authenticate primary bytes/scales, including adversarial blocks.
 * Independent pre103 down/read programs and a frozen TwoTerm publisher calibrate
 * complete prepared HC chains before fusion, with exact bank/guard histories.
 * A separate authored HC-up/read composition preserves a nonzero signal lost
 * by SingleQ8, with independent FP64 source arithmetic and live-prefix replay.
 * Explicit HC-up K-pair diagnostics keep the production32x64 tile intact
 * while comparing actual production with an explicit original single-K32 reference.
 * M512 HC-down tile comparisons retain a forced baseline, all native Q8 source
 * aliases and dirty M512/M64 histories independently of installed Auto policy.
 */
#include <gtest/gtest.h>
#include "qwen4exp/HC103OriginalGPU.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpHyperconnectionBuilder.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include <map>
#include <tuple>
#include <sstream>
#include "utils/GpuPreparedGemmHarness.h"
#include "utils/Q8ResidualAllFormatOracle.h"
#include "../../utils/Qwen4ExpFAOutputStageFixture.h"
#include "kernels/common/Q8ResidualGroupedGeometry.h"
#include "kernels/common/Q8ResidualGroupedDiagnostic.h"
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "kernels/KernelFactory.h"
#include "kernels/cuda/gemm/CUDAQuantisedGemmKernel.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "kernels/rocm/gemm/ROCmWeightPacker.h"
#ifdef HAVE_ROCM
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "kernels/rocm/gemm/ROCmVNNIPrefillLaunch.h"
#include "kernels/rocm/gemm/ROCmDenseProductionPrefillOverlay.h"
#include "kernels/rocm/gemm/ROCmNativeQuantisedArchitecture.h"
#include "utils/DebugEnv.h"
#endif
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"
#include "../../utils/PreparedWeightTestHarness.h"
#include "../../utils/HCUpCancellationFixture.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPrimitiveStage.h"
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
using namespace llaminar2;
namespace {
using Factory=llaminar::v2::kernels::KernelFactory;
/** @brief Fail infrastructure before a retained graph can reference invalid storage. */
void require(bool value,const char*why){if(!value)throw std::runtime_error(why);}
/** @brief Retain native allocations and retire graph before its captured addresses. */
struct Storage {
 DeviceId device;IBackend*backend;void*stream;
 std::shared_ptr<std::vector<std::shared_ptr<DeviceTransferBuffer>>> owners=std::make_shared<std::vector<std::shared_ptr<DeviceTransferBuffer>>>();
 std::unique_ptr<IGPUGraphCapture> graph;
 /** @brief Retire submitted diagnostic work, then its graph, before captured native allocations. */
 ~Storage(){retire();}
 /** @brief Complete native work and retire its graph; teardown failure is fatal.
  * This is a diagnostic terminal boundary, never a per-stage inference sync.
  */
 void retire()noexcept{if(!backend->synchronizeStream(stream,device.ordinal))std::terminate();if(graph){graph->reset();graph.reset();}}
 /** @brief Retain one exact transfer-authority device allocation until graph retirement. */
 template<class T>T*allocate(std::size_t n){auto owner=TransferEngine::instance().allocateDeviceTransferBuffer(n*sizeof(T),device);require(owner&&owner->isBound(),"Native correction fixture allocation failed");auto*p=static_cast<T*>(owner->mutableDeviceData());owners->push_back(std::move(owner));return p;}
 /** @brief Publish initial/replay diagnostic data on the exact stream, outside measured work. */
 template<class T>void upload(T*p,const std::vector<T>&x){const bool copied=backend->hostToDeviceOnStream(p,x.data(),x.size()*sizeof(T),device.ordinal,stream);if(!backend->synchronizeStream(stream,device.ordinal))std::terminate();require(copied,"Correction fixture publication failed after its exact stream retired");}
 /** @brief Materialize and initialize an immutable fixture plane before capture. */
 template<class T>T*input(const std::vector<T>&x){auto*p=allocate<T>(x.size());upload(p,x);return p;}
 /** @brief Terminally observe only the explicitly requested live prefix; empty observations transfer nothing. */
 template<class T>std::vector<T>download(const T*p,std::size_t n){if(!n)return {};std::vector<T>x(n);const bool copied=backend->deviceToHostOnStream(x.data(),p,n*sizeof(T),device.ordinal,stream);if(!backend->synchronizeStream(stream,device.ordinal))std::terminate();require(copied,"Correction fixture terminal copy failed after its exact stream retired");return x;}
};
/** @brief Retire captured work before every captured tensor/workspace owner.
 * Declare this scope after all such owners and before the first native submit;
 * exception and ASSERT unwind then follows the same completion-before-storage
 * order as successful replay. Storage remains alive outside the scope.
 */
struct CapturedWorkRetirement {
 Storage&storage;
 explicit CapturedWorkRetirement(Storage&s):storage(s){}
 CapturedWorkRetirement(const CapturedWorkRetirement&)=delete;
 CapturedWorkRetirement&operator=(const CapturedWorkRetirement&)=delete;
 ~CapturedWorkRetirement(){storage.retire();}
};
/** @brief Admit only the supplied actual stage/engine named buffers. */
struct Workspace {
 std::shared_ptr<PhysicalMemoryAuthority>memory;std::unique_ptr<DeviceWorkspaceManager>value;
 Workspace(DeviceId device,const WorkspaceRequirements&r){auto bytes=r.total_bytes_with_alignment();PhysicalMemoryPlanBuilder plan;plan.add({.world_rank=0,.device=device,.total_bytes=bytes,.admission_available_bytes=bytes},PhysicalMemoryOwner::ExecutionWorkspace,bytes);memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);value=std::make_unique<DeviceWorkspaceManager>(device,bytes,memory);require(value->allocate(r),"Correction fixture workspace admission failed");}
};
/** @brief Native source bits have distinct signs/scales and are never requantized. */
std::vector<std::uint8_t> nativeWeights(int n,int k){std::vector<std::uint8_t>raw(std::size_t(n)*k/32*34);for(std::size_t b=0;b<raw.size()/34;++b){Q8_0Block block{};block.d=fp32_to_fp16(.002f*float(1+b%11));for(int i=0;i<32;++i)block.qs[i]=static_cast<std::int8_t>((b*13+i*17)%255-127);std::memcpy(raw.data()+b*34,&block,34);}return raw;}
/** @brief Authenticate exact native source bits before device publication. */
void authenticate(const std::vector<std::uint8_t>&raw,const std::vector<std::uint8_t>&payload,const std::vector<std::uint16_t>&scales,int n,int k){for(int row=0;row<n;++row)for(int b=0;b<k/32;++b){const auto*source=raw.data()+(std::size_t(row)*(k/32)+b)*34;std::uint16_t d;std::memcpy(&d,source,2);ASSERT_EQ(d,scales[std::size_t(b)*n+row]);ASSERT_EQ(std::memcmp(source+2,payload.data()+(std::size_t(b)*n+row)*32,32),0);}}
/** @brief Use actual production native packing and preuploaded prepared-engine construction. */
std::shared_ptr<ITensorGemm> prepare(Q8_0Tensor&source,const std::vector<std::uint8_t>&raw,Storage&s,int n,int k){
#ifdef HAVE_CUDA
 if(s.device.is_cuda()){cuda::CUDAPackedWeights packed;require(cuda::packWeightsToCUDA(&source,packed),"Production CUDA native pack failed");authenticate(raw,packed.native_quantised,packed.native_scales,n,k);return std::make_shared<cuda::CUDAQuantisedGemmKernel>(n,k,s.device.ordinal,s.input(packed.native_quantised),s.input(packed.native_scales),nullptr,nullptr,packed.native_codebook_id,packed.native_blocks_per_row,s.owners,packed.native_source_identity);}
#endif
#ifdef HAVE_ROCM
 if(s.device.is_rocm()){rocm::ROCmPackedWeights packed;require(rocm::packNativeQuantised(&source,packed),"Production HIP native pack failed");authenticate(raw,packed.native_quantised_payload,packed.native_quantised_scales,n,k);return std::make_shared<rocm::ROCmQuantisedGemmKernel>(n,k,s.device.ordinal,s.input(packed.native_quantised_payload),s.input(packed.native_quantised_scales),nullptr,nullptr,packed.native_quantised_codebook_id,packed.native_quantised_blocks_per_row,s.owners,packed.native_source_identity);}
#endif
 throw std::runtime_error("Correction fixture selected unavailable native backend");}
/**
 * @brief Register exact original source/engine identities through the production prepared-store boundary.
 * @param store Sole prepared handle owner used by the actual GEMMStage.
 * @param source Original native Q8_0 source tensor, retained through captured work.
 * @param device Exact original packing and execution participant.
 * @param engine Actual production pack/engine sharing its native allocation owners.
 * @param sourceName Original HC or GDN source role; default preserves existing HC cases.
 * @return Immutable binding to the original tensor and exact prepared engine.
 */
WeightBinding registerEngine(PreparedWeightStore&store,Q8_0Tensor&source,DeviceId device,std::shared_ptr<ITensorGemm>engine,const char *sourceName="hc.attn_up.weight"){auto binding=test::makePreparedWeightTestBinding(&source,device,sourceName,store.modelId());auto handle=std::make_shared<Factory::PreparedGemmHandle>();handle->tensor=&source;handle->device_id=device;handle->kind=device.is_cuda()?Factory::GemmPreparationKind::CUDA_INT8_PACKED:Factory::GemmPreparationKind::ROCM_INT8_PACKED;handle->variant=static_cast<int>(TensorType::Q8_0);handle->prepared_weights=std::make_shared<Factory::PreparedGemmWeights>();handle->prepared_weights->kind=handle->kind;handle->prepared_weights->owned_kernel=std::move(engine);handle->prepared_weights->kernel=handle->prepared_weights->owned_kernel.get();binding.prepared=store.registerPreparedGemmHandle(binding,test::preparedKindForDevice(device),device,std::move(handle));return binding;}
/** @brief Prove retained native execution and exact actual production primary quantization. */
void prove(DeviceId device){auto*backend=getBackendFor(device);require(backend&&backend->deviceCount()>device.ordinal,"Requested native correction backend unavailable");auto&worker=GPUDeviceContextPool::instance().getContext(device);auto context=IDeviceContext::create(device,1);worker.submitAsync([&]{
 for(auto geometry:std::array{std::array{1,65},std::array{15,127},std::array{512,64},std::array{1024,64},std::array{1,10240},std::array{15,10240},std::array{512,10240},std::array{1024,10240}}){const int m=geometry[0],n=geometry[1],k=320;SCOPED_TRACE(::testing::Message()<<device.toString()<<" M="<<m<<" N="<<n);Storage s{device,backend,ExplicitGPUStream(worker.defaultStream()).get()};auto raw=nativeWeights(n,k);Q8_0Tensor source({std::size_t(n),std::size_t(k)},raw);PreparedWeightStore store{ModelContextId{801944}};auto binding=registerEngine(store,source,device,prepare(source,raw,s,n,k));auto*kernel=store.gemmKernel(*binding.prepared);kernel->setGPUStream(s.stream);DeviceNativeQuantisedMatrixDesc before;require(kernel->exportNativeQuantisedMatrixDesc(before),"Prepared source export failed");
  auto input=std::make_shared<FP32Tensor>(std::vector<std::size_t>{std::size_t(m),std::size_t(k)},DeviceId::cpu());auto output=std::make_shared<FP32Tensor>(std::vector<std::size_t>{std::size_t(m),std::size_t(n)},DeviceId::cpu());FP32Tensor baseline({std::size_t(m),std::size_t(n)},DeviceId::cpu());
  for(int row=0;row<m;++row)for(int c=0;c<k;++c){float value=c%32==0?float((row%7)+1)*17.25f:std::sin(float(c*7+row*11))*.31f;if(row%17==0&&c<32)value=0;if(row%17==1&&c<32)value=std::numeric_limits<float>::denorm_min()*float(c+1);if(row%17==2&&c<32)value=c==0?127.f:float(c%11-5)+.5f;if(row%17==3&&c<32)value=c==0?1e9f:float(c-16)*1e6f;input->mutable_data()[std::size_t(row)*k+c]=value;}
  const std::vector<float>fixtureInput(input->data(),input->data()+input->numel());
  for(auto*t:std::array<TensorBase*,3>{input.get(),output.get(),&baseline}){TransferEngine::prepareDeviceInput(t,device,s.stream);TransferEngine::requireDeviceInput(t,device,s.stream);}
  // FP32 create_view only borrows host data; it has no native GPU alias
  // contract. Independent precreated M1 tensors are the serial diagnostic
  // oracle. They contain the exact same immutable fixture row values and are
  // never substituted into the captured grouped production operation.
  std::vector<std::shared_ptr<FP32Tensor>>inputRows,outputRows;
  for(int row=0;row<m;++row){auto serialInput=std::make_shared<FP32Tensor>(std::vector<std::size_t>{1,std::size_t(k)},DeviceId::cpu());auto serialOutput=std::make_shared<FP32Tensor>(std::vector<std::size_t>{1,std::size_t(n)},DeviceId::cpu());
   std::copy_n(fixtureInput.data()+std::size_t(row)*k,k,serialInput->mutable_data());
   for(auto*t:std::array{serialInput.get(),serialOutput.get()}){TransferEngine::prepareDeviceInput(t,device,s.stream);TransferEngine::requireDeviceInput(t,device,s.stream);}
   require(serialInput->gpu_data_ptr()&&serialOutput->gpu_data_ptr(),"Independent native serial tensor preparation failed");
   inputRows.push_back(std::move(serialInput));outputRows.push_back(std::move(serialOutput));}

  auto*count=s.input(std::vector<int>{m});GEMMStage stage({.device_id=device,.A=input.get(),.B=&source,.C=output.get(),.m=m,.n=n,.k=k,.alpha=1,.beta=0,.transpose_B=false,.a_buffer_id=BufferId::NORMALIZED,.c_buffer_id=BufferId::GATE_PROJ,.force_decode_equivalent_verifier_prefill=true,.verifier_row_range=DeviceRowRange::deviceCounted(m,count),.prepared_ref=binding.prepared,.prepared_store=&store,.quantized_operand_precision=QuantizedActivationOperandPrecision::TwoTermQ8});stage.setGPUStream(s.stream);const auto requirements=stage.getWorkspaceRequirements(m,n,k);ASSERT_EQ(requirements.buffers.size(),4u);Workspace correction(device,requirements);stage.bindWorkspace(correction.value.get());auto*ordinaryWorkspaceConsumer=dynamic_cast<IWorkspaceConsumer*>(kernel);require(ordinaryWorkspaceConsumer,"Prepared native engine lacks workspace contract");auto ordinaryRequirements=ordinaryWorkspaceConsumer->getWorkspaceRequirements(m,n,k);
  const std::array projectionColumns{n};ordinaryWorkspaceConsumer->appendFusedProjectionWorkspaceRequirements(ordinaryRequirements,m,projectionColumns,k);Workspace ordinary(device,ordinaryRequirements);
  const std::vector<ITensorGemm::TensorProjectionDesc>ordinaryProjection{{kernel,&baseline,n,nullptr,"hc.up"}};
  CapturedWorkRetirement retirement(s);
  // Authenticate the exact production entrypoint used by each HC-up phase.
  // Grouped M15 must exercise the verifier API, not generic prefill by accident.
  if(m==15){auto scope=kernel->beginVerifierDecodeEquivalentScope(DeviceRowRange::fullyActive(m));ASSERT_TRUE(kernel->multiply_fused_verifier_rows_decode_equivalent(input.get(),ordinaryProjection,m,k,nullptr,ordinary.value.get()));}
  else ASSERT_TRUE(kernel->multiply_tensor(input.get(),&baseline,m,n,k,false,1,0,nullptr,nullptr,device.ordinal,ordinary.value.get()));auto ordinaryBytes=s.download(static_cast<const signed char*>(ordinary.value->getBuffer(kGPUProjectionPrimaryQ8)),std::size_t(m)*k);auto ordinaryScales=s.download(static_cast<const float*>(ordinary.value->getBuffer(kGPUProjectionPrimaryQ8Scales)),std::size_t(m)*k/32);
  // Retain each actual serial M1 primary before the next serial call reuses
  // ordinary workspace. All diagnostic copies use the exact producer stream;
  // one terminal observation compares every primary byte/scale in every row.
  auto*serialResult=s.allocate<float>(std::size_t(m)*n);
  auto*serialPrimary=s.allocate<signed char>(std::size_t(m)*k);auto*serialScales=s.allocate<float>(std::size_t(m)*k/32);
  for(int row=0;row<m;++row){ASSERT_TRUE(kernel->multiply_tensor(inputRows[row].get(),outputRows[row].get(),1,n,k,false,1,0,nullptr,nullptr,device.ordinal,ordinary.value.get()));
   require(backend->deviceToDevice(serialPrimary+std::size_t(row)*k,ordinary.value->getBuffer(kGPUProjectionPrimaryQ8),k,device.ordinal,s.stream),"Serial primary native copy failed");
   require(backend->deviceToDevice(serialScales+std::size_t(row)*k/32,ordinary.value->getBuffer(kGPUProjectionPrimaryQ8Scales),std::size_t(k)/32*4,device.ordinal,s.stream),"Serial scale native copy failed");}
  const auto serialPrimaryBytes=s.download(serialPrimary,std::size_t(m)*k);const auto serialPrimaryScales=s.download(serialScales,std::size_t(m)*k/32);
  ASSERT_EQ(std::memcmp(ordinaryBytes.data(),serialPrimaryBytes.data(),ordinaryBytes.size()),0);
  ASSERT_EQ(std::memcmp(ordinaryScales.data(),serialPrimaryScales.data(),ordinaryScales.size()*4),0);
  s.graph=worker.createGraphCapture(s.stream);{ScopedBackendGraphCapture capture(worker,*s.graph,"Prepared two-term Q8 HC-up");require(capture.begin(),"Correction stage capture failed");require(stage.execute(context.get()),"Correction stage submission failed");capture.finish();}require(s.graph->instantiate(),"Correction stage graph instantiation failed");std::vector<GPUGraphKernelNodeInfo>nodes;std::string error;require(s.graph->inspectKernelNodes(nodes,&error),"Correction graph inspection failed");ASSERT_EQ(nodes.size(),2u);for(const auto&node:nodes)EXPECT_EQ(node.local_memory_bytes_per_thread,0u);
  for(int live:std::array{m,std::min(2,m),0,m}){s.upload(count,std::vector<int>{live});s.upload(static_cast<float*>(output->gpu_data_ptr()),std::vector<float>(std::size_t(m)*n,-12345.f));require(s.graph->launch(),"Retained correction graph replay failed");auto actual=s.download(static_cast<const float*>(output->gpu_data_ptr()),std::size_t(m)*n);auto primary=s.download(static_cast<const signed char*>(correction.value->getBuffer(kGPUProjectionPrimaryQ8)),std::size_t(live)*k);auto scales=s.download(static_cast<const float*>(correction.value->getBuffer(kGPUProjectionPrimaryQ8Scales)),std::size_t(live)*k/32);if(live){EXPECT_EQ(std::memcmp(primary.data(),ordinaryBytes.data(),primary.size()),0);EXPECT_EQ(std::memcmp(scales.data(),ordinaryScales.data(),scales.size()*4),0);}
   for(int row=0;row<live;++row){kernel->multiply_q8_residual_corrected(inputRows[row].get(),outputRows[row].get(),DeviceRowRange::fullyActive(1),n,k,correction.value.get());
    require(backend->deviceToDevice(serialResult+std::size_t(row)*n,outputRows[row]->gpu_data_ptr(),std::size_t(n)*4,device.ordinal,s.stream),"Independent serial output gather failed");}
   auto expected=s.download(serialResult,std::size_t(live)*n);if(live){EXPECT_EQ(std::memcmp(actual.data(),expected.data(),expected.size()*4),0);}for(std::size_t i=std::size_t(live)*n;i<actual.size();++i)ASSERT_EQ(actual[i],-12345.f);
  }
  DeviceNativeQuantisedMatrixDesc after;require(kernel->exportNativeQuantisedMatrixDesc(after),"Prepared source re-export failed");EXPECT_EQ(before.payload,after.payload);EXPECT_EQ(before.scales,after.scales);EXPECT_EQ(before.source_codebook_id,after.source_codebook_id);s.retire();stage.unbindWorkspace();
 }
 }).get();}
/**
 * @brief Execute the original GDN output geometry through the actual retained corrected stage.
 * @param device Exact native participant owning source, tensors, workspace and diagnostic stream.
 *
 * Only one precreated M1 tensor pair is used by the independent serial oracle.
 * Ordered D2D copies select original input rows and retain results/primary bytes;
 * the diagnostic joins once per completed plane, never once per serial row.
 * Every upload/download below is test setup or terminal observation outside the
 * captured stage. No host live-count readback determines a production launch.
 */
// FA producer topology is declared to exercise the exact Stage5 composer;
// arithmetic capture isolates its resulting projection after an already-gated
// original input. This is not full attention/gate graph qualification.
void proveGDN(DeviceId device, bool fa_output = false)
{
 auto *backend=getBackendFor(device);
 require(backend&&backend->deviceCount()>device.ordinal,"Requested GDN correction backend unavailable");
 auto context=IDeviceContext::create(device,1);
 auto &worker=GPUDeviceContextPool::instance().getContext(device);
 worker.submitAsync([&] {
  constexpr int n=2560,k=6144;
  for(const int m:std::array{1,15,512,1024,2048})
  {
   SCOPED_TRACE(::testing::Message()<<device.toString()<<" GDN M="<<m<<" N="<<n<<" K="<<k);
   Storage s{device,backend,ExplicitGPUStream(worker.defaultStream()).get()};
   auto raw=nativeWeights(n,k);Q8_0Tensor source({std::size_t(n),std::size_t(k)},raw);
   PreparedWeightStore store{ModelContextId{801945}};
   auto binding=registerEngine(store,source,device,prepare(source,raw,s,n,k),
       fa_output ? "blk.3.attn_output.weight" : "blk.0.ssm_out.weight");
   if(fa_output)
   {
    binding.slice={.source_rows=n,.source_cols=k,.row_count=n,.col_count=k};
    require(store.adoptPreparedGemmForBinding(binding,device),"FA original full-source adoption failed");
   }
   auto *kernel=store.gemmKernel(*binding.prepared);require(kernel,"GDN prepared source missing");
   kernel->setGPUStream(s.stream);
   DeviceNativeQuantisedMatrixDesc before;require(kernel->exportNativeQuantisedMatrixDesc(before),"GDN source export failed");
   FP32Tensor input({std::size_t(m),std::size_t(k)},DeviceId::cpu());
   FP32Tensor output({std::size_t(m),std::size_t(n)},DeviceId::cpu());
   FP32Tensor baseline({std::size_t(m),std::size_t(n)},DeviceId::cpu());
   FP32Tensor serialInput({1,std::size_t(k)},DeviceId::cpu()),serialOutput({1,std::size_t(n)},DeviceId::cpu());
   std::vector<float>original(std::size_t(m)*k);
   for(int row=0;row<m;++row)for(int c=0;c<k;++c)
   {
    float value=c%32==0?float(row%7+1)*17.25f:std::sin(float(c*7+row*11))*.31f;
    if(row%17==0&&c<32)value=0;
    if(row%17==1&&c<32)value=std::numeric_limits<float>::denorm_min()*float(c+1);
    if(row%17==2&&c<32)value=c==0?127.f:float(c%11-5)+.5f;
    original[std::size_t(row)*k+c]=value;
   }
   std::copy(original.begin(),original.end(),input.mutable_data());
   // The FA arm executes the ACTUAL production composer result, not a copied
   // Params approximation. Existing GDN standalone coverage is unchanged.
   auto *queryWords=s.allocate<int32_t>(fa_output ? std::size_t(4*m+8) : 1u);
   auto *count=fa_output ? queryWords+4*m+2 : queryWords;
   ComputeGraph faGraph;
   std::unique_ptr<FP32Tensor> faGate;
   std::unique_ptr<test::Qwen4ExpFAOutputStageBuilder> faBuilder;
   std::unique_ptr<GEMMStage> legacyStage;
   GEMMStage *actualStage=nullptr;
   if(fa_output)
   {
    GraphConfig config;
    Qwen4ExpGraphConfigBuilder::populateFromDirectory(test::qwen4MetadataFixture(false),config);
    config.default_device=device;config.qfn_hc_up_activation=QuantizedActivationOperandPrecision::SingleQ8;
    config.qfn_gdn_output_activation=QuantizedActivationOperandPrecision::SingleQ8;
    config.qfn_fa_output_activation=QuantizedActivationOperandPrecision::TwoTermQ8;
    faBuilder=std::make_unique<test::Qwen4ExpFAOutputStageBuilder>(config);
    faBuilder->setPreparedWeightStore(&store);
    faGate=std::make_unique<FP32Tensor>(std::vector<std::size_t>{std::size_t(m),std::size_t(k)});
    faGraph.addNode("layer3_attn_output_gate",std::make_unique<AttentionOutputGateStage>(AttentionOutputGateStage::Params{
     .device_id=device,.input=&input,.gate=faGate.get(),.output=&input,.seq_len=m,.feature_dim=k,
     .input_buffer_id=BufferId::ATTN_OUTPUT,.gate_buffer_id=BufferId::FA_GATE,.output_buffer_id=BufferId::ATTN_OUTPUT}),device);
    DevicePrefillChunkGraphBinding chunk{.backend=backend,.request_token_ids_device=queryWords,
     .request_position_ids_device=queryWords+m,.request_total_rows_device=queryWords+4*m,
     .cached_tokens_device=queryWords+4*m+1,.chunk_token_ids_device=queryWords+2*m,
     .chunk_position_ids_device=queryWords+3*m,.chunk_real_rows_device=count,
     .chunk_row_stride_device=queryWords+4*m+3,.request_row_capacity=m,.bucket_seq_len=m,
     .pad_token_id=0};
    std::uint64_t queryIdentity=0x51464e46414f0001ULL;
    const auto mixQuery=[&](std::uint64_t value){queryIdentity^=value+0x9e3779b97f4a7c15ULL+(queryIdentity<<6)+(queryIdentity>>2);};
    mixQuery(reinterpret_cast<std::uintptr_t>(backend));
    for(const auto *pointer:{chunk.request_token_ids_device,chunk.request_position_ids_device,
        chunk.request_total_rows_device,chunk.cached_tokens_device,
        static_cast<const int32_t*>(chunk.chunk_token_ids_device),static_cast<const int32_t*>(chunk.chunk_position_ids_device),
        static_cast<const int32_t*>(chunk.chunk_real_rows_device),static_cast<const int32_t*>(chunk.chunk_row_stride_device)})
        mixQuery(reinterpret_cast<std::uintptr_t>(pointer));
    mixQuery(static_cast<std::uint64_t>(device.type));mixQuery(device.ordinal);
    mixQuery(m);mixQuery(chunk.pad_token_id);chunk.capture_identity=queryIdentity ? queryIdentity : 1;
    ForwardInput invocation;invocation.device=device;invocation.batch_size=1;invocation.seq_len=m;
    invocation.bucket_seq_len=m;invocation.execution_role=ForwardExecutionRole::MainInference;
    invocation.execution_phase=ForwardExecutionPhase::Prefill;invocation.token_ids_device=chunk.chunk_token_ids_device;
    invocation.position_ids_device=chunk.chunk_position_ids_device;invocation.sequence_lengths_device=count;
    invocation.device_prefill_chunk=chunk;
    ActivationBuffers buffers;buffers.attn_output=&input;buffers.attn_proj=&output;
    require(faBuilder->append(faGraph,invocation,buffers,binding)=="layer3_wo_proj","FA production composer omitted Stage5");
    actualStage=dynamic_cast<GEMMStage*>(faGraph.getNode("layer3_wo_proj")->stage.get());
    require(actualStage,"FA composer did not produce actual GEMMStage");
    const auto &params=actualStage->getParams();
    require(params.B==&source && params.prepared_store==&store && params.prepared_ref &&
     params.prepared_ref->binding_id==binding.prepared->binding_id && params.verifier_row_range &&
     params.verifier_row_range->countOwner()==count,"FA composer changed original source/query authority");
    require(params.force_decode_equivalent_verifier_prefill==(m>1),"FA composer lost actual grouped arithmetic policy");
   }
   else
   {
    legacyStage=std::make_unique<GEMMStage>(GEMMStage::Params{.device_id=device,.A=&input,.B=&source,.C=&output,.m=m,.n=n,.k=k,
     .alpha=1,.beta=0,.transpose_B=false,.a_buffer_id=BufferId::NORMALIZED,.c_buffer_id=BufferId::ATTN_PROJ,
     .force_decode_equivalent_verifier_prefill=m==15,.verifier_row_range=DeviceRowRange::deviceCounted(m,count),
     .prepared_ref=binding.prepared,.prepared_store=&store,.quantized_operand_precision=QuantizedActivationOperandPrecision::TwoTermQ8});
    actualStage=legacyStage.get();
   }
   auto &stage=*actualStage;
   stage.setGPUStream(s.stream);
   const auto requirements=stage.getWorkspaceRequirements(m,n,k);ASSERT_EQ(requirements.buffers.size(),4u);
   Workspace correction(device,requirements);stage.bindWorkspace(correction.value.get());
   auto *consumer=dynamic_cast<IWorkspaceConsumer*>(kernel);require(consumer,"GDN engine lacks workspace contract");
   auto ordinaryRequirements=consumer->getWorkspaceRequirements(m,n,k);
   const std::array columns{n};consumer->appendFusedProjectionWorkspaceRequirements(ordinaryRequirements,m,columns,k);
   Workspace ordinary(device,ordinaryRequirements);
   auto *oracle=s.allocate<float>(std::size_t(m)*n);
   auto *serialPrimary=s.allocate<signed char>(std::size_t(m)*k);
   auto *serialScales=s.allocate<float>(std::size_t(m)*k/32);
   CapturedWorkRetirement retirement(s);
   for(auto *tensor:std::array<TensorBase*,5>{&input,&output,&baseline,&serialInput,&serialOutput})
   {
    TransferEngine::prepareDeviceInput(tensor,device,s.stream);
    TransferEngine::requireDeviceInput(tensor,device,s.stream);
   }
   s.upload(count,std::vector<int32_t>{m});
   const auto ordinaryProjection=std::vector<ITensorGemm::TensorProjectionDesc>{{kernel,&baseline,n,nullptr,"gdn.out"}};
   if(m==15 || (fa_output && m>1))
   {
    auto scope=kernel->beginVerifierDecodeEquivalentScope(DeviceRowRange::fullyActive(m));
    ASSERT_TRUE(kernel->multiply_fused_verifier_rows_decode_equivalent(&input,ordinaryProjection,m,k,nullptr,ordinary.value.get()));
   }
   else
   {
    ASSERT_TRUE(kernel->multiply_tensor(&input,&baseline,m,n,k,false,1,0,nullptr,nullptr,device.ordinal,ordinary.value.get()));
   }
   const auto ordinaryBytes=s.download(static_cast<const signed char*>(ordinary.value->getBuffer(kGPUProjectionPrimaryQ8)),std::size_t(m)*k);
   const auto ordinaryScales=s.download(static_cast<const float*>(ordinary.value->getBuffer(kGPUProjectionPrimaryQ8Scales)),std::size_t(m)*k/32);
   // These exact-stream diagnostic commands retain all M1 results before the
   // next row reuses the same pair/workspace. There is no per-row sync/readback.
   for(int row=0;row<m;++row)
   {
    require(backend->deviceToDevice(serialInput.gpu_data_ptr(),static_cast<const float*>(input.gpu_data_ptr())+std::size_t(row)*k,
     std::size_t(k)*sizeof(float),device.ordinal,s.stream),"GDN serial input selection failed");
    ASSERT_TRUE(kernel->multiply_tensor(&serialInput,&serialOutput,1,n,k,false,1,0,nullptr,nullptr,device.ordinal,ordinary.value.get()));
    require(backend->deviceToDevice(serialPrimary+std::size_t(row)*k,ordinary.value->getBuffer(kGPUProjectionPrimaryQ8),
     k,device.ordinal,s.stream),"GDN serial primary retention failed");
    require(backend->deviceToDevice(serialScales+std::size_t(row)*k/32,ordinary.value->getBuffer(kGPUProjectionPrimaryQ8Scales),
     std::size_t(k)/32*sizeof(float),device.ordinal,s.stream),"GDN serial scale retention failed");
    kernel->multiply_q8_residual_corrected(&serialInput,&serialOutput,DeviceRowRange::fullyActive(1),n,k,correction.value.get());
    require(backend->deviceToDevice(oracle+std::size_t(row)*n,serialOutput.gpu_data_ptr(),std::size_t(n)*sizeof(float),
     device.ordinal,s.stream),"GDN serial output retention failed");
   }
   const auto serialBytes=s.download(serialPrimary,std::size_t(m)*k);
   const auto serialScaleValues=s.download(serialScales,std::size_t(m)*k/32);
   const auto expected=s.download(oracle,std::size_t(m)*n);
   ASSERT_EQ(std::memcmp(ordinaryBytes.data(),serialBytes.data(),ordinaryBytes.size()),0);
   ASSERT_EQ(std::memcmp(ordinaryScales.data(),serialScaleValues.data(),ordinaryScales.size()*sizeof(float)),0);
   const auto signature=stage.graphCaptureVariantSignature();ASSERT_NE(signature,0u);
   s.graph=worker.createGraphCapture(s.stream);
   {
    ScopedBackendGraphCapture capture(worker,*s.graph,"Prepared two-term GDN output");
    require(capture.begin(),"GDN stage capture failed");require(stage.execute(context.get()),"GDN stage submission failed");capture.finish();
   }
   require(s.graph->instantiate(),"GDN stage graph instantiation failed");
   std::vector<GPUGraphKernelNodeInfo>nodes;std::string error;
   require(s.graph->inspectKernelNodes(nodes,&error),"GDN graph inspection failed");ASSERT_EQ(nodes.size(),2u);
   const auto geometry=q8ResidualGroupedGeometry(device.is_cuda()?Q8ResidualGroupedBackend::CUDA:Q8ResidualGroupedBackend::ROCm,m,n);
   unsigned publishers=0,projections=0;
   for(const auto &node:nodes){
    EXPECT_EQ(node.local_memory_bytes_per_thread,0u);EXPECT_TRUE(node.name_resolved);
    EXPECT_EQ(node.grid_y,1u);EXPECT_EQ(node.grid_z,1u);EXPECT_EQ(node.block_y,1u);EXPECT_EQ(node.block_z,1u);
    if(node.name.find("publish")!=std::string::npos){++publishers;EXPECT_EQ(node.block_x,256u);EXPECT_EQ(node.grid_x,(std::uint64_t(m)*(k/32)+7)/8);}
    if(node.name.find("project")!=std::string::npos){++projections;EXPECT_EQ(node.block_x,static_cast<unsigned>(geometry.threads));EXPECT_EQ(node.grid_x,geometry.blocks);
     EXPECT_EQ(node.name.find("projectGrouped")!=std::string::npos,geometry.strategy==Q8ResidualGroupedStrategy::CooperativeMatrix);}
   }
   EXPECT_EQ(publishers,1u);EXPECT_EQ(projections,1u);
   const auto ownerCount=s.owners->size(),used=correction.value->used();
   const auto primary=static_cast<signed char*>(correction.value->getBuffer(kGPUProjectionPrimaryQ8));
   const auto residual=static_cast<signed char*>(correction.value->getBuffer(kGPUProjectionResidualQ8));
   const auto scales=static_cast<float*>(correction.value->getBuffer(kGPUProjectionPrimaryQ8Scales));
   const auto residualScales=static_cast<float*>(correction.value->getBuffer(kGPUProjectionResidualQ8Scales));
   const std::vector<signed char>dirtyBytes(std::size_t(m)*k,90);
   const std::vector<float>dirtyScales(std::size_t(m)*k/32,-8765.f),dirtyOutput(std::size_t(m)*n,-12345.f);
   for(const int live:std::array{m,std::min(9,m),1,0,std::min(15,m),m,0})
   {
    SCOPED_TRACE(::testing::Message()<<"live="<<live);
    auto poisoned=original;std::fill(poisoned.begin()+std::size_t(live)*k,poisoned.end(),std::numeric_limits<float>::quiet_NaN());
    s.upload(static_cast<float*>(input.gpu_data_ptr()),poisoned);
    s.upload(static_cast<float*>(output.gpu_data_ptr()),dirtyOutput);
    s.upload(primary,dirtyBytes);s.upload(residual,dirtyBytes);s.upload(scales,dirtyScales);s.upload(residualScales,dirtyScales);
    s.upload(count,std::vector<int32_t>{live});
    require(s.graph->launch(),"GDN retained replay failed");
    const auto actual=s.download(static_cast<const float*>(output.gpu_data_ptr()),std::size_t(m)*n);
    const auto actualPrimary=s.download(primary,std::size_t(m)*k),actualResidual=s.download(residual,std::size_t(m)*k);
    const auto actualScales=s.download(scales,std::size_t(m)*k/32),actualResidualScales=s.download(residualScales,std::size_t(m)*k/32);
    if(live)
    {
     EXPECT_EQ(std::memcmp(actual.data(),expected.data(),std::size_t(live)*n*sizeof(float)),0);
     EXPECT_EQ(std::memcmp(actualPrimary.data(),ordinaryBytes.data(),std::size_t(live)*k),0);
     EXPECT_EQ(std::memcmp(actualScales.data(),ordinaryScales.data(),std::size_t(live)*k/32*sizeof(float)),0);
    }
    EXPECT_TRUE(std::all_of(actual.begin()+std::size_t(live)*n,actual.end(),[](float x){return x==-12345.f;}));
    EXPECT_TRUE(std::all_of(actualPrimary.begin()+std::size_t(live)*k,actualPrimary.end(),[](signed char x){return x==90;}));
    EXPECT_TRUE(std::all_of(actualResidual.begin()+std::size_t(live)*k,actualResidual.end(),[](signed char x){return x==90;}));
    EXPECT_TRUE(std::all_of(actualScales.begin()+std::size_t(live)*k/32,actualScales.end(),[](float x){return x==-8765.f;}));
    EXPECT_TRUE(std::all_of(actualResidualScales.begin()+std::size_t(live)*k/32,actualResidualScales.end(),[](float x){return x==-8765.f;}));
    EXPECT_EQ(stage.graphCaptureVariantSignature(),signature);EXPECT_EQ(s.owners->size(),ownerCount);
    EXPECT_EQ(correction.value->used(),used);EXPECT_EQ(correction.value->getBuffer(kGPUProjectionPrimaryQ8),primary);
   }
   DeviceNativeQuantisedMatrixDesc after;require(kernel->exportNativeQuantisedMatrixDesc(after),"GDN final source export failed");
   EXPECT_EQ(before.payload,after.payload);EXPECT_EQ(before.scales,after.scales);EXPECT_EQ(before.source_codebook_id,after.source_codebook_id);
   EXPECT_EQ(std::memcmp(raw.data(),source.raw_data(),raw.size()),0);
   s.retire();stage.unbindWorkspace();
  }
 }).get();
}
/**
 * @brief Capture actual corrected HC-up followed by a nonzero-gate ResidualRead.
 * @param device Original native participant retained by every prepared owner.
 *
 * Authored source/input bytes admit an independent FP64 signed-dot/read oracle.
 * One M1 and one M15 graph retain the original count address through partial,
 * empty and large-again replay. No reference implementation executes natively.
 */
void proveCancellationRead(DeviceId device)
{
    namespace hc = test::hc_up_cancellation;
    using Primitive = Qwen4ExpPrimitiveStage;
    using Float = Qwen4ExpFloatBinding;
    auto *backend = getBackendFor(device);
    require(backend && backend->deviceCount() > device.ordinal,"HC cancellation backend unavailable");
    auto& worker = GPUDeviceContextPool::instance().getContext(device);
    auto context = IDeviceContext::create(device,1);
    worker.submitAsync([&]
    {
        constexpr int n = 10240, k = hc::kColumns, hidden = n / 4;
        for (int m : {1,15})
        {
            SCOPED_TRACE(::testing::Message() << device.toString() << " HC cancellation M=" << m);
            Storage storage{device,backend,ExplicitGPUStream(worker.defaultStream()).get()};
            const auto raw = hc::weights(n);
            const auto inputValues = hc::activations(m);
            const auto branches = hc::normalized(m,n);
            const auto oracle = hc::reference(raw,inputValues,branches,m,n);
            Q8_0Tensor source({std::size_t(n),std::size_t(k)},raw);
            PreparedWeightStore store{ModelContextId{801955}};
            const auto binding = registerEngine(store,source,device,prepare(source,raw,storage,n,k));
            auto *kernel = store.gemmKernel(*binding.prepared);
            require(kernel,"HC cancellation prepared engine missing");
            kernel->setGPUStream(storage.stream);
            DeviceNativeQuantisedMatrixDesc original;
            require(kernel->exportNativeQuantisedMatrixDesc(original),"HC cancellation original native descriptor missing");
            FP32Tensor input({std::size_t(m),std::size_t(k)},DeviceId::cpu());
            FP32Tensor normalized({std::size_t(m),std::size_t(n)},DeviceId::cpu());
            FP32Tensor singleLogits({std::size_t(m),std::size_t(n)},DeviceId::cpu());
            FP32Tensor correctedLogits({std::size_t(m),std::size_t(n)},DeviceId::cpu());
            FP32Tensor singleRead({std::size_t(m),std::size_t(hidden)},DeviceId::cpu());
            FP32Tensor correctedRead({std::size_t(m),std::size_t(hidden)},DeviceId::cpu());
            auto *count = storage.input(std::vector<int>{m});
            std::copy(inputValues.begin(),inputValues.end(),input.mutable_data());
            std::copy(branches.begin(),branches.end(),normalized.mutable_data());
            const auto rows = DeviceRowRange::deviceCounted(m,count);
            GEMMStage::Params params{.device_id=device,.A=&input,.B=&source,.C=&singleLogits,
                .m=m,.n=n,.k=k,.a_buffer_id=BufferId::NORMALIZED,.c_buffer_id=BufferId::GATE_PROJ,
                .force_decode_equivalent_verifier_prefill=true,.verifier_row_range=rows,
                .prepared_ref=binding.prepared,.prepared_store=&store};
            GEMMStage single(params);
            params.C = &correctedLogits;
            params.quantized_operand_precision = QuantizedActivationOperandPrecision::TwoTermQ8;
            GEMMStage corrected(params);
            single.setGPUStream(storage.stream);
            corrected.setGPUStream(storage.stream);
            auto ordinaryRequirements = single.getWorkspaceRequirements(m,n,k);
            auto *consumer = dynamic_cast<IWorkspaceConsumer*>(kernel);
            require(consumer,"HC cancellation ordinary workspace consumer missing");
            const std::array columns{n};
            consumer->appendFusedProjectionWorkspaceRequirements(ordinaryRequirements,m,columns,k);
            Workspace ordinary(device,ordinaryRequirements);
            Workspace residual(device,corrected.getWorkspaceRequirements(m,n,k));
            single.bindWorkspace(ordinary.value.get());
            corrected.bindWorkspace(residual.value.get());
            const Qwen4ExpStageRows primitiveRows{m,Qwen4ExpIntInput(Qwen4ExpNativeIntInput{device,count,1})};
            Primitive readSingle({.device_id=device,.rows=primitiveRows,.operation=Primitive::ResidualRead{hidden,
                Float(normalized,BufferId::NORMALIZED),Float(singleLogits,BufferId::GATE_PROJ),Float(singleRead,BufferId::ATTN_OUTPUT)}});
            Primitive readCorrected({.device_id=device,.rows=primitiveRows,.operation=Primitive::ResidualRead{hidden,
                Float(normalized,BufferId::NORMALIZED),Float(correctedLogits,BufferId::GATE_PROJ),Float(correctedRead,BufferId::ATTN_OUTPUT)}});
            readSingle.setGPUStream(storage.stream);
            readCorrected.setGPUStream(storage.stream);
            // This final local is destroyed first on every ASSERT/exception exit:
            // submitted work and graph retire before tensor/workspace backing.
            CapturedWorkRetirement retirement(storage);
            for (auto *tensor : std::vector<TensorBase*>{&input,&normalized,&singleLogits,&correctedLogits,&singleRead,&correctedRead})
            {
                TransferEngine::prepareDeviceInput(tensor,device,storage.stream);
                TransferEngine::requireDeviceInput(tensor,device,storage.stream);
            }
            ASSERT_TRUE(single.execute(context.get()));
            ASSERT_TRUE(readSingle.execute(context.get()));
            const auto singleResult = storage.download(static_cast<const float*>(singleRead.gpu_data_ptr()),std::size_t(m)*hidden);
            const auto ordinaryPrimary = storage.download(static_cast<const signed char*>(ordinary.value->getBuffer(kGPUProjectionPrimaryQ8)),std::size_t(m)*k);
            const auto ordinaryScales = storage.download(static_cast<const float*>(ordinary.value->getBuffer(kGPUProjectionPrimaryQ8Scales)),std::size_t(m)*k/32);
            storage.graph = worker.createGraphCapture(storage.stream);
            {
                ScopedBackendGraphCapture capture(worker,*storage.graph,"HC cancellation corrected up and read");
                require(capture.begin(),"HC cancellation capture begin failed");
                require(corrected.execute(context.get()),"HC cancellation corrected up submission failed");
                require(readCorrected.execute(context.get()),"HC cancellation read submission failed");
                capture.finish();
            }
            require(storage.graph->instantiate(),"HC cancellation graph instantiate failed");
            std::vector<GPUGraphKernelNodeInfo> nodes;
            std::string error;
            require(storage.graph->inspectKernelNodes(nodes,&error),"HC cancellation graph inspection failed");
            ASSERT_EQ(nodes.size(),3u); // Native quantizer, corrected projection, actual read.
            for (const auto& node : nodes)
            {
                EXPECT_EQ(node.local_memory_bytes_per_thread,0u);
            }
            const auto signature = corrected.graphCaptureVariantSignature();
            const auto readSignature = readCorrected.graphCaptureVariantSignature();
            const auto used = residual.value->used();
            const auto ownerCount = storage.owners->size();
            const auto primary = residual.value->getBuffer(kGPUProjectionPrimaryQ8);
            const auto residualBank = residual.value->getBuffer(kGPUProjectionResidualQ8);
            for (int live : std::array{m,std::min(9,m),1,0,m})
            {
                SCOPED_TRACE(::testing::Message() << "live=" << live);
                auto poisonedInput = inputValues;
                auto poisonedBranches = branches;
                std::fill(poisonedInput.begin()+std::size_t(live)*k,poisonedInput.end(),std::numeric_limits<float>::quiet_NaN());
                std::fill(poisonedBranches.begin()+std::size_t(live)*n,poisonedBranches.end(),std::numeric_limits<float>::quiet_NaN());
                storage.upload(count,std::vector<int>{live});
                storage.upload(static_cast<float*>(input.gpu_data_ptr()),poisonedInput);
                storage.upload(static_cast<float*>(normalized.gpu_data_ptr()),poisonedBranches);
                storage.upload(static_cast<float*>(correctedLogits.gpu_data_ptr()),std::vector<float>(std::size_t(m)*n,hc::kSentinel));
                storage.upload(static_cast<float*>(correctedRead.gpu_data_ptr()),std::vector<float>(std::size_t(m)*hidden,hc::kSentinel));
                require(storage.graph->launch(),"HC cancellation retained graph replay failed");
                const auto actualLogits = storage.download(static_cast<const float*>(correctedLogits.gpu_data_ptr()),std::size_t(m)*n);
                const auto actualRead = storage.download(static_cast<const float*>(correctedRead.gpu_data_ptr()),std::size_t(m)*hidden);
                const auto primaryBytes = storage.download(static_cast<const signed char*>(primary),std::size_t(live)*k);
                const auto primaryScales = storage.download(static_cast<const float*>(residual.value->getBuffer(kGPUProjectionPrimaryQ8Scales)),std::size_t(live)*k/32);
                if (live)
                {
                    EXPECT_EQ(std::memcmp(primaryBytes.data(),ordinaryPrimary.data(),primaryBytes.size()),0);
                    EXPECT_EQ(std::memcmp(primaryScales.data(),ordinaryScales.data(),primaryScales.size()*sizeof(float)),0);
                }
                for (std::size_t i=0; i<std::size_t(live)*n; ++i)
                {
                    ASSERT_TRUE(std::isfinite(actualLogits[i]));
                    EXPECT_NEAR(actualLogits[i],oracle.logits[i],hc::kLogitAbsoluteError);
                }
                for (std::size_t i=0; i<std::size_t(live)*hidden; ++i)
                {
                    EXPECT_GT(std::abs(oracle.read[i]),.2);
                    ASSERT_TRUE(std::isfinite(singleResult[i]));
                    EXPECT_GT(std::abs(double(singleResult[i])-oracle.read[i]),hc::kSingleReadErrorFloor);
                    ASSERT_TRUE(std::isfinite(actualRead[i]));
                    EXPECT_NEAR(actualRead[i],oracle.read[i],hc::kReadAbsoluteError);
                }
                for (std::size_t i=std::size_t(live)*n; i<actualLogits.size(); ++i)
                {
                    ASSERT_EQ(actualLogits[i],hc::kSentinel);
                }
                for (std::size_t i=std::size_t(live)*hidden; i<actualRead.size(); ++i)
                {
                    ASSERT_EQ(actualRead[i],hc::kSentinel);
                }
                EXPECT_EQ(corrected.graphCaptureVariantSignature(),signature);
                EXPECT_EQ(readCorrected.graphCaptureVariantSignature(),readSignature);
                EXPECT_EQ(residual.value->used(),used);
                EXPECT_EQ(storage.owners->size(),ownerCount);
                EXPECT_EQ(residual.value->getBuffer(kGPUProjectionPrimaryQ8),primary);
                EXPECT_EQ(residual.value->getBuffer(kGPUProjectionResidualQ8),residualBank);
            }
            DeviceNativeQuantisedMatrixDesc after;
            require(kernel->exportNativeQuantisedMatrixDesc(after),"HC cancellation final source descriptor missing");
            EXPECT_EQ(original.payload,after.payload);
            EXPECT_EQ(original.scales,after.scales);
            EXPECT_EQ(original.source_codebook_id,after.source_codebook_id);
            EXPECT_EQ(std::memcmp(raw.data(),source.raw_data(),raw.size()),0);
            storage.retire();
            single.unbindWorkspace();
            corrected.unbindWorkspace();
        }
    }).get();
}
/** @brief Freeze a complete original HC read chain before installing103. */
/** @brief Original/typed fusion comparison or explicit fixed-tile K-pair staging. */
enum class HC103Path {Control,Typed,GroupedKPair};
/** @brief Native replay, paired events, or one event-free retained profile variant. */
enum class HC103Observation {Correctness,Paired,ProfileProduction,ProfileReference};
/** @brief Test-only native-address substitution without transferring allocation ownership. */
class HC103AddressTensor final:public FP32Tensor {
public:
 using FP32Tensor::FP32Tensor;
 /** @brief Replace only the diagnostic virtual binding; the real allocation remains owned.
  * @param address Another admitted native address, or null to restore the original.
  */
 void substitute(void* address){substitute_=address;}
 /** @return Native address visible to consumers; no device query occurs. */
 void* gpu_data_ptr() override{return substitute_?substitute_:FP32Tensor::gpu_data_ptr();}
 /** @return Same diagnostic address for const and mutable admission. */
 const void* gpu_data_ptr()const override{return substitute_?substitute_:FP32Tensor::gpu_data_ptr();}
private:void* substitute_=nullptr;
};
/** @brief Compare original calibration or actual typed HC builder using the same frozen oracle.
 * @param device Exact native participant and worker.
 * @param m Original physical row capacity; live replay count stays device-owned.
 * @param hidden Per-branch width; the real trunk/up width is four times this.
 * @param lowRank Complete source-admitted HC bottleneck width.
 * @param alias Whether raw down and activated use the exact same tensor.
 * @param path Original chain, typed fusion comparison, or actual production/original single-group reference.
 * @param observation Correctness histories, opt-in paired events or event-free selected profiling.
 * @param format Original registry source; null retains the historical Q8_0 case.
 */
void proveHC103Chain(DeviceId device,int m,int hidden,int lowRank,bool alias,HC103Path path,
 HC103Observation observation=HC103Observation::Correctness,const test::QuantizedVerifierFormatCase *format=nullptr) {
 const bool timing=observation==HC103Observation::Paired;
 const bool profiling=observation==HC103Observation::ProfileProduction||observation==HC103Observation::ProfileReference;
 const bool kPair=path==HC103Path::GroupedKPair;
 const bool ordinaryDown=observation!=HC103Observation::Correctness;
 const bool typed=path!=HC103Path::Control;
 require(!kPair||(device.is_rocm()&&m>=32&&hidden>=16),"HC K-pair diagnostic requires its native ROCm grouped geometry");
 require(!profiling||kPair,"Event-free HC profiling requires an explicit K-group variant");
 auto *backend=getBackendFor(device);require(backend&&backend->deviceCount()>device.ordinal,"HC103 backend unavailable");
 auto &worker=GPUDeviceContextPool::instance().getContext(device);auto context=IDeviceContext::create(device,1);
 worker.submitAsync([&]{
  SCOPED_TRACE(::testing::Message()<<device.toString()<<" M/H/R="<<m<<'/'<<hidden<<'/'<<lowRank<<" alias="<<alias);
  int productionGroups=1;
#ifdef HAVE_ROCM
  if(kPair){
   using Arch=native_quantised::NativeQuantisedArchitecture;
   productionGroups=Arch::fromROCmISATarget(rocmHardwareMetadata().at(device.ordinal).isa_target)==
    Arch::rocm(9,0,6)?2:1;
  }
#endif
  const int trunk=4*hidden;Storage s{device,backend,ExplicitGPUStream(worker.defaultStream()).get()};
  auto downBytes=nativeWeights(lowRank,trunk),upBytes=nativeWeights(trunk,lowRank);
  std::unique_ptr<TensorBase> downSource,upSource;
  if(format) {
   if(kPair){
    test::requireQuantizedVerifierSourceShape(*format,lowRank,trunk);
    test::requireQuantizedVerifierSourceShape(*format,trunk,lowRank);
   }
   downSource=test::q8_all_formats::source(*format,lowRank,trunk);
   upSource=test::q8_all_formats::source(*format,trunk,lowRank);
  } else {
   downSource=std::make_unique<Q8_0Tensor>(std::vector<std::size_t>{std::size_t(lowRank),std::size_t(trunk)},downBytes);
   upSource=std::make_unique<Q8_0Tensor>(std::vector<std::size_t>{std::size_t(trunk),std::size_t(lowRank)},upBytes);
  }
  const auto savedDownBytes=kPair?test::q8_all_formats::originalBytes(*downSource):std::vector<std::uint8_t>{};
  const auto savedUpBytes=kPair?test::q8_all_formats::originalBytes(*upSource):std::vector<std::uint8_t>{};
  auto downPrepared=test::makeGpuPreparedGemm(downSource.get(),device,"hc103.down.weight",ModelContextId{803103});
  auto upPrepared=test::makeGpuPreparedGemm(upSource.get(),device,"hc103.up.weight",ModelContextId{803104});
  auto *downEngine=downPrepared.kernel,*upEngine=upPrepared.kernel;
  DeviceNativeQuantisedMatrixDesc originalUp;require(upEngine->exportNativeQuantisedMatrixDesc(originalUp),"HC103 original prepared source missing");
  require(!kPair||(originalUp.valid()&&originalUp.codebook_id==19&&originalUp.source_identity_present),
   "HC K-pair diagnostic requires an authenticated original native Q8 source");
  if(kPair){
   const auto sourceCodebook=format?format->source_codebook_id:std::uint8_t{19};
   const bool sourceSuperblock=format&&format->source_is_superblock;
   DeviceNativeQuantisedMatrixDesc originalDown;
   require(downEngine->exportNativeQuantisedMatrixDesc(originalDown)&&originalDown.valid(),
    "HC K-pair original prepared down descriptor missing");
   for(const auto* descriptor:{&originalDown,&originalUp})
    require(descriptor->codebook_id==19&&descriptor->source_identity_present&&
     descriptor->source_codebook_id==sourceCodebook&&descriptor->source_is_superblock==sourceSuperblock,
     "HC K-pair prepared source identity differs from its requested original alias");
   require(originalDown.n==lowRank&&originalDown.k==trunk&&originalDown.blocks_per_row==unsigned(trunk/32)&&
    originalUp.n==trunk&&originalUp.k==lowRank&&originalUp.blocks_per_row==unsigned(lowRank/32),
    "HC K-pair prepared source geometry differs from its admitted original operands");
  }
  FP32Tensor normalized({std::size_t(m+1),std::size_t(trunk)},DeviceId::cpu());
  std::array<std::unique_ptr<FP32Tensor>,2> rawDown,activated,up,read;
  FP32Tensor branchInput({std::size_t(m+1),std::size_t(trunk)},DeviceId::cpu());
  FP32Tensor normGain({std::size_t(trunk)},DeviceId::cpu());
  std::fill_n(normGain.mutable_data(),trunk,1.f);
  for(int side=0;side<2;++side) {
   rawDown[side]=std::make_unique<HC103AddressTensor>(std::vector<std::size_t>{std::size_t(m+1),std::size_t(lowRank)},DeviceId::cpu());
   activated[side]=std::make_unique<HC103AddressTensor>(std::vector<std::size_t>{std::size_t(m+1),std::size_t(lowRank)},DeviceId::cpu());
   up[side]=std::make_unique<HC103AddressTensor>(std::vector<std::size_t>{std::size_t(m+1),std::size_t(trunk)},DeviceId::cpu());
   read[side]=std::make_unique<FP32Tensor>(std::vector<std::size_t>{std::size_t(m+1),std::size_t(hidden)},DeviceId::cpu());
  }
  auto *count=s.allocate<int32_t>(1);const auto rows=DeviceRowRange::deviceCounted(m,count);
  std::array<std::unique_ptr<GEMMStage>,2> downStages,upStages;
  std::array<std::unique_ptr<Workspace>,2> downWorkspace,upWorkspace;
  /** @brief Clear all stage borrows after graph retirement but before workspace release. */
  struct StageBorrows {
   std::array<std::unique_ptr<GEMMStage>,2>&down;std::array<std::unique_ptr<GEMMStage>,2>&up;
   ~StageBorrows(){for(int side=0;side<2;++side){if(down[side])down[side]->unbindWorkspace();if(up[side])up[side]->unbindWorkspace();}}
  } borrows{downStages,upStages};
  for(int side=0;side<2;++side) {
   auto *raw=alias?activated[side].get():rawDown[side].get();
   GEMMStage::Params downParams{.device_id=device,.A=&normalized,.B=downSource.get(),.C=raw,.m=m,.n=lowRank,.k=trunk,.transpose_B=false,
    .a_buffer_id=BufferId::NORMALIZED,.c_buffer_id=BufferId::GATE_PROJ,.force_decode_equivalent_verifier_prefill=!ordinaryDown,
    .verifier_row_range=ordinaryDown?std::nullopt:std::optional{rows},.prepared_ref=downPrepared.ref,.prepared_store=downPrepared.store.get()};
   downStages[side]=std::make_unique<GEMMStage>(downParams);downStages[side]->setGPUStream(s.stream);
   downWorkspace[side]=std::make_unique<Workspace>(device,downStages[side]->getWorkspaceRequirements(m,lowRank,trunk));
   downStages[side]->bindWorkspace(downWorkspace[side]->value.get());
   GEMMStage::Params upParams{.device_id=device,.A=activated[side].get(),.B=upSource.get(),.C=up[side].get(),.m=m,.n=trunk,.k=lowRank,.transpose_B=false,
    .a_buffer_id=BufferId::GATE_PROJ,.c_buffer_id=BufferId::ATTN_OUTPUT,.force_decode_equivalent_verifier_prefill=true,
    .verifier_row_range=rows,.prepared_ref=upPrepared.ref,.prepared_store=upPrepared.store.get(),
    .quantized_operand_precision=QuantizedActivationOperandPrecision::TwoTermQ8};
   if(kPair){
    upParams.A=raw;
    upParams.hc_residual_down=GEMMStage::HCResidualDownPublication{
     activated[side].get(),alias?BufferId::GATE_PROJ:BufferId::UP_PROJ};
   }
   upStages[side]=std::make_unique<GEMMStage>(upParams);upStages[side]->setGPUStream(s.stream);
   upWorkspace[side]=std::make_unique<Workspace>(device,upStages[side]->getWorkspaceRequirements(m,trunk,lowRank));
   upStages[side]->bindWorkspace(upWorkspace[side]->value.get());
  }
  std::unique_ptr<qwen4exp::IQwen4ExpKernels> primitives;
#ifdef HAVE_CUDA
  if(device.is_cuda())primitives=qwen4exp::createCUDAQwen4ExpKernels();
#endif
#ifdef HAVE_ROCM
  if(device.is_rocm())primitives=qwen4exp::createROCmQwen4ExpKernels();
#endif
  require(bool(primitives),"HC103 primitive implementation absent");
  const std::size_t banks=std::size_t(m)*lowRank,scales=banks/32;
  auto *frozenPrimary=s.allocate<signed char>(banks),*frozenResidual=s.allocate<signed char>(banks);
  auto *frozenPrimaryScales=s.allocate<float>(scales),*frozenResidualScales=s.allocate<float>(scales);
  std::unique_ptr<IGPUGraphCapture> referenceGraph,witnessGraph;
  std::unique_ptr<ComputeGraph> builder;
  std::array<GEMMStage*,2> actualDown{downStages[0].get(),downStages[1].get()},actualUp{upStages[0].get(),upStages[1].get()};
  /** @brief Clear builder stage borrows after terminal graph retirement, before workspaces. */
  struct BuilderBorrows {
   std::unique_ptr<ComputeGraph>&graph;
   ~BuilderBorrows(){if(graph)for(const auto&name:graph->getExecutionOrder())
    if(auto*stage=dynamic_cast<IWorkspaceConsumer*>(graph->getNode(name)->stage.get()))stage->unbindWorkspace();}
  } builderBorrows{builder};
  // This final scope drains every graph before all stage/tensor/source owners.
  CapturedWorkRetirement retirement(s);
  const auto prepareOperands=[&] {
   for(auto *tensor:std::vector<TensorBase*>{&normalized,rawDown[0].get(),rawDown[1].get(),activated[0].get(),activated[1].get(),up[0].get(),up[1].get(),read[0].get(),read[1].get()}) {
    std::fill_n(tensor->mutable_data(),tensor->numel(),0.125f);
    TransferEngine::prepareDeviceInput(tensor,device,s.stream);TransferEngine::requireDeviceInput(tensor,device,s.stream);
   }
  };
  if(!typed)prepareOperands();
  if(typed) {
   std::fill_n(branchInput.mutable_data(),branchInput.numel(),0.125f);
   for(auto*tensor:std::array<TensorBase*,2>{&branchInput,&normGain}) {
    TransferEngine::prepareDeviceInput(tensor,device,s.stream);TransferEngine::requireDeviceInput(tensor,device,s.stream);
   }
   using F=Qwen4ExpFloatBinding;builder=std::make_unique<ComputeGraph>();
   const Qwen4ExpHyperconnectionContext hc{.device=device,.prefix="hc103_",
    .rows={m,Qwen4ExpIntInput{Qwen4ExpNativeIntInput{device,count,1}}},.hidden=hidden,.lowRank=lowRank,.epsilon=1e-6f,
    .arithmetic=ordinaryDown?Qwen4ExpProjectionArithmetic::Ordinary:Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier,
    .projectionRows=ordinaryDown?std::nullopt:std::optional{rows},
    .upOperandPrecision=QuantizedActivationOperandPrecision::TwoTermQ8,.upProjectionRows=rows};
   auto*raw=alias?activated[0].get():rawDown[0].get();
   const Qwen4ExpHyperconnectionReadBuffers buffers{F(branchInput,BufferId::RESIDUAL),F(normalized,BufferId::NORMALIZED),
    F(*raw,BufferId::GATE_PROJ),F(*activated[0],alias?BufferId::GATE_PROJ:BufferId::UP_PROJ),
    F(*up[0],BufferId::ATTN_OUTPUT),F(*read[0],BufferId::HIDDEN_STATE)};
   // Registration accepts a const caller binding and publishes its prepared
   // reference in the store-owned copy. The builder consumes that exact frozen
   // binding, rather than the caller's still-unprepared registration metadata.
   const auto downBinding=downPrepared.store->binding(downPrepared.ref);
   const auto upBinding=upPrepared.store->binding(upPrepared.ref);
   require(downBinding&&upBinding,"HC103 actual builder lost its stored prepared bindings");
   Qwen4ExpHyperconnectionBuilder::appendFinalRead(*builder,hc,
    {Qwen4ExpImmutableWeightBinding(normGain),Qwen4ExpPreparedProjection(*downBinding,*downPrepared.store,device),
     Qwen4ExpPreparedProjection(*upBinding,*upPrepared.store,device)},buffers,{});
   ASSERT_EQ(builder->getNode("hc103_read_activation"),nullptr);
   ASSERT_EQ(builder->getExecutionOrder().size(),4u);
   actualDown[0]=dynamic_cast<GEMMStage*>(builder->getNode("hc103_read_down")->stage.get());
   actualUp[0]=dynamic_cast<GEMMStage*>(builder->getNode("hc103_read_up")->stage.get());
   require(actualDown[0]&&actualUp[0],"HC103 actual builder lost prepared GEMM stages");
   ASSERT_TRUE(actualUp[0]->getParams().hc_residual_down.has_value());
   EXPECT_EQ(actualUp[0]->getParams().A,raw);EXPECT_EQ(actualUp[0]->getParams().hc_residual_down->activated,activated[0].get());
   actualDown[0]->bindWorkspace(downWorkspace[0]->value.get());actualUp[0]->bindWorkspace(upWorkspace[0]->value.get());
   for(const auto&name:builder->getExecutionOrder())builder->getNode(name)->stage->setGPUStream(s.stream);
   const auto before=actualUp[0]->graphCaptureVariantSignature();
   // Native addresses are materialized after the declarative signature exists.
   // First preparation seals those allocations without changing the cache key.
   prepareOperands();
   EXPECT_EQ(actualUp[0]->graphCaptureVariantSignature(),before);
   ASSERT_TRUE(actualUp[0]->prepareGraphLaunch(context.get(),s.stream));
   EXPECT_EQ(actualUp[0]->graphCaptureVariantSignature(),before);
   ASSERT_TRUE(actualUp[0]->prepareGraphLaunch(context.get(),s.stream));
   EXPECT_EQ(actualUp[0]->graphCaptureVariantSignature(),before);
   if(!format&&m==3&&hidden==8&&!alias&&observation==HC103Observation::Correctness) {
    const auto valid=actualUp[0]->getParams();
    auto malformed=valid;malformed.output_range={0,1};
    EXPECT_THROW((void)GEMMStage(malformed),std::invalid_argument);
    malformed=valid;malformed.verifier_row_range=DeviceRowRange::fullyActive(1);
    EXPECT_THROW((void)GEMMStage(malformed),std::invalid_argument);
    malformed=valid;malformed.verifier_row_range=DeviceRowRange::deviceCounted(m+1,count).slice(1,m);
    EXPECT_THROW((void)GEMMStage(malformed),std::invalid_argument);
    malformed=valid;malformed.prepared_ref=downPrepared.ref;malformed.prepared_store=downPrepared.store.get();
    EXPECT_THROW((void)GEMMStage(malformed),std::invalid_argument);
    malformed=valid;malformed.hc_residual_down=GEMMStage::HCResidualDownPublication{up[0].get(),BufferId::ATTN_OUTPUT};
    EXPECT_THROW((void)GEMMStage(malformed),std::invalid_argument);
    EXPECT_THROW(actualUp[0]->prepareGraphLaunch(context.get(),nullptr),std::invalid_argument);
    for(auto operands:std::array{std::pair{rawDown[0].get(),rawDown[1].get()},
       std::pair{activated[0].get(),activated[1].get()},std::pair{up[0].get(),up[1].get()}}) {
     auto* address=dynamic_cast<HC103AddressTensor*>(operands.first);
     require(address,"HC103 diagnostic operand missing");
     address->substitute(operands.second->gpu_data_ptr());
     EXPECT_THROW(actualUp[0]->graphCaptureVariantSignature(),std::logic_error);
     EXPECT_THROW(actualUp[0]->prepareGraphLaunch(context.get(),s.stream),std::logic_error);
     address->substitute(nullptr);
     EXPECT_EQ(actualUp[0]->graphCaptureVariantSignature(),before);
    }
    // A different admitted tensor address is legal for a fresh stage only.
    auto* address=dynamic_cast<HC103AddressTensor*>(activated[0].get());
    address->substitute(activated[1]->gpu_data_ptr());
    {
     GEMMStage fresh(valid);fresh.setGPUStream(s.stream);fresh.bindWorkspace(upWorkspace[0]->value.get());
     EXPECT_NO_THROW((void)fresh.prepareGraphLaunch(context.get(),s.stream));fresh.unbindWorkspace();
    }
    address->substitute(nullptr);
    EXPECT_EQ(actualUp[0]->graphCaptureVariantSignature(),before);
    Workspace replacement(device,actualUp[0]->getWorkspaceRequirements(m,trunk,lowRank));
    actualUp[0]->bindWorkspace(replacement.value.get());
    EXPECT_THROW(actualUp[0]->graphCaptureVariantSignature(),std::logic_error);
    EXPECT_THROW(actualUp[0]->prepareGraphLaunch(context.get(),s.stream),std::logic_error);
    {
     GEMMStage fresh(valid);fresh.setGPUStream(s.stream);fresh.bindWorkspace(replacement.value.get());
     EXPECT_NO_THROW((void)fresh.prepareGraphLaunch(context.get(),s.stream));fresh.unbindWorkspace();
    }
    actualUp[0]->bindWorkspace(upWorkspace[0]->value.get());
    EXPECT_EQ(actualUp[0]->graphCaptureVariantSignature(),before);
    // Restore the exact tuple before any real graph is recorded or replayed.
    // The public stage API rejects null before changing the retained stream.
    EXPECT_THROW(actualUp[0]->setGPUStream(nullptr),std::invalid_argument);
    EXPECT_EQ(actualUp[0]->graphCaptureVariantSignature(),before);
    EXPECT_NO_THROW((void)actualUp[0]->prepareGraphLaunch(context.get(),s.stream));
    void* changedStream=worker.getOrCreateAuxiliaryStream("hc103_identity_changed_stream");
    require(changedStream&&changedStream!=s.stream,"HC103 distinct identity stream unavailable");
    actualUp[0]->setGPUStream(changedStream);
    EXPECT_THROW(actualUp[0]->prepareGraphLaunch(context.get(),changedStream),std::logic_error);
    actualUp[0]->setGPUStream(s.stream);
    EXPECT_NO_THROW((void)actualUp[0]->prepareGraphLaunch(context.get(),s.stream));
    for(auto precision:{QuantizedActivationOperandPrecision::SingleQ8,QuantizedActivationOperandPrecision::TwoTermQ8}) {
     auto ordinary=valid;ordinary.hc_residual_down.reset();ordinary.A=activated[0].get();
     ordinary.a_buffer_id=BufferId::UP_PROJ;
     ordinary.quantized_operand_precision=precision;
     GEMMStage unchanged(ordinary);const auto identity=unchanged.graphCaptureVariantSignature();
     unchanged.setGPUStream(s.stream);unchanged.bindWorkspace(upWorkspace[0]->value.get());
     EXPECT_TRUE(unchanged.prepareGraphLaunch(context.get(),s.stream));
     EXPECT_EQ(unchanged.graphCaptureVariantSignature(),identity);
     unchanged.resetSessionState();EXPECT_EQ(unchanged.graphCaptureVariantSignature(),identity);
     unchanged.unbindWorkspace();
    }
   }
   const auto contract=actualUp[0]->bufferContract();
   EXPECT_EQ(contract.prepared_weights.size(),1u);
   EXPECT_EQ(contract.outputs.size(),alias?1u:2u);EXPECT_EQ(contract.inouts.size(),alias?1u:0u);
  }
  auto originalDown=[&](qwen4exp::ResidualDownArgs a) {
#ifdef HAVE_CUDA
   if(device.is_cuda()){test103::cudaOriginalDown(a,s.stream);return;}
#endif
#ifdef HAVE_ROCM
   if(device.is_rocm()){test103::rocmOriginalDown(a,s.stream);return;}
#endif
   throw std::runtime_error("HC103 frozen down unavailable");
  };
  auto originalRead=[&](qwen4exp::ResidualReadArgs a) {
#ifdef HAVE_CUDA
   if(device.is_cuda()){test103::cudaOriginalRead(a,s.stream);return;}
#endif
#ifdef HAVE_ROCM
   if(device.is_rocm()){test103::rocmOriginalRead(a,s.stream);return;}
#endif
   throw std::runtime_error("HC103 frozen read unavailable");
  };
  // Borrow the same admitted source, count, outputs and workspace banks for
  // reference capture and prepublication rejection; no alternate storage owner.
  auto referenceProjection=[&](int side){
   return GPUQ8ResidualCorrectionArgs{originalUp,rows,Q8ResidualQuantizerPolicy::ROCmFiniteHalfNativeReciprocal,
    static_cast<const float*>((alias?activated[side]:rawDown[side])->gpu_data_ptr()),
    static_cast<float*>(up[side]->gpu_data_ptr()),
    static_cast<signed char*>(upWorkspace[side]->value->getBuffer(kGPUProjectionPrimaryQ8)),
    static_cast<signed char*>(upWorkspace[side]->value->getBuffer(kGPUProjectionResidualQ8)),
    static_cast<float*>(upWorkspace[side]->value->getBuffer(kGPUProjectionPrimaryQ8Scales)),
    static_cast<float*>(upWorkspace[side]->value->getBuffer(kGPUProjectionResidualQ8Scales))};
  };
  auto chain=[&](int side) {
   if(typed) {
    if(side==0) {
     for(const auto&name:builder->getExecutionOrder())require(builder->getNode(name)->stage->execute(context.get()),"HC103 actual builder recording rejected");
     return;
    }
    // Norm is unchanged by103; both paths retain its exact production operation.
    require(builder->getNode("hc103_branch_norm")->stage->execute(context.get()),"HC103 reference norm rejected");
   }
   require(downStages[side]->execute(context.get()),"HC103 prepared down rejected");
   const qwen4exp::RowExtent extent{m,count};
   if(kPair){
#ifdef HAVE_ROCM
    rocmHCResidualDownGroupedSingleGroupReference({referenceProjection(side),
     static_cast<float*>(activated[side]->gpu_data_ptr())},s.stream);
    primitives->residualRead({extent,hidden,static_cast<const float*>(normalized.gpu_data_ptr()),
     static_cast<const float*>(up[side]->gpu_data_ptr()),static_cast<float*>(read[side]->gpu_data_ptr())},s.stream);
    return;
#endif
   }
   qwen4exp::ResidualDownArgs activation{extent,lowRank,static_cast<const float*>((alias?activated[side]:rawDown[side])->gpu_data_ptr()),static_cast<float*>(activated[side]->gpu_data_ptr())};
   if(side==0)primitives->residualDown(activation,s.stream);else originalDown(activation);
   require(upStages[side]->execute(context.get()),"HC103 original prepared TwoTerm up rejected");
   qwen4exp::ResidualReadArgs mix{extent,hidden,static_cast<const float*>(normalized.gpu_data_ptr()),static_cast<const float*>(up[side]->gpu_data_ptr()),static_cast<float*>(read[side]->gpu_data_ptr())};
   if(side==0)primitives->residualRead(mix,s.stream);else originalRead(mix);
  };
  const int copies=timing&&m==1?32:1;
  std::array<std::vector<GPUGraphKernelNodeInfo>,2> inventory;
  auto capture=[&](std::unique_ptr<IGPUGraphCapture>&graph,int side) {
   graph=worker.createGraphCapture(s.stream);ScopedBackendGraphCapture recording(worker,*graph,"HC103 original complete control chain");
   require(recording.begin(),"HC103 capture begin failed");for(int copy=0;copy<copies;++copy)chain(side);recording.finish();
   require(graph->instantiate(),"HC103 graph instantiate failed");
   std::vector<GPUGraphKernelNodeInfo> nodes;std::string error;require(graph->inspectKernelNodes(nodes,&error),"HC103 graph inventory failed");
   inventory[side]=nodes;EXPECT_EQ(graph->nodeCount(),nodes.size());for(const auto&node:nodes) {
    EXPECT_EQ(node.local_memory_bytes_per_thread,0u);require(node.name_resolved,"HC103 unresolved native kernel");
    std::cout<<(kPair?"HC103_KPAIR_KERNEL,":"HC103_CONTROL_KERNEL,")<<side<<','<<node.name<<','<<node.grid_x<<','<<node.grid_y<<','<<node.grid_z
     <<','<<node.block_x<<','<<node.block_y<<','<<node.block_z<<','<<node.registers_per_thread<<','<<node.static_shared_memory_bytes<<','<<node.local_memory_bytes_per_thread;
    if(kPair)std::cout<<",function="<<node.function_identity<<",dynamic_shared="<<node.dynamic_shared_memory_bytes;
    std::cout<<'\n';
   }
   std::cout<<(kPair?"HC103_KPAIR_GRAPH,":"HC103_CONTROL_GRAPH,")<<device.toString()<<','<<m<<','<<hidden<<','<<lowRank<<','<<side<<','<<copies<<','<<nodes.size()<<','<<(ordinaryDown?"ordinary_full_down":"counted_verifier_down");
   if(kPair)std::cout<<",role="<<(side==0?"production":"single_group_reference")
    <<",staged_groups="<<(side==0?productionGroups:1);
   std::cout<<'\n';
  };
  s.upload(count,std::vector<int32_t>{m});
#ifdef HAVE_ROCM
  if(kPair&&observation==HC103Observation::Correctness&&m==32&&!alias){
   const GPUHCResidualDownPublication valid{referenceProjection(1),static_cast<float*>(activated[1]->gpu_data_ptr())};
   const float poison=-2718.25f;
   for(auto* tensor:{activated[1].get(),up[1].get()})
    s.upload(static_cast<float*>(tensor->gpu_data_ptr()),std::vector<float>(tensor->numel(),poison));
   for(auto* bank:{valid.projection.primary,valid.projection.residual})
    s.upload(bank,std::vector<signed char>(banks,-37));
   for(auto* bank:{valid.projection.primaryScales,valid.projection.residualScales})
    s.upload(bank,std::vector<float>(scales,poison));
   auto rejected=worker.createGraphCapture(s.stream);
   {
    ScopedBackendGraphCapture recording(worker,*rejected,"HC K-pair invalid-before-publication");
    require(recording.begin(),"HC K-pair rejection capture failed");
    auto rejectBoth=[&](GPUHCResidualDownPublication invalid,void* stream){
     EXPECT_THROW(rocmHCResidualDownGroupedSingleGroupReference(invalid,stream),std::invalid_argument);
     EXPECT_THROW(rocmQ8ResidualGroupedSingleGroupReference(invalid.projection,stream),std::invalid_argument);
    };
    rejectBoth(valid,nullptr);
    auto invalid=valid;invalid.projection.rows=DeviceRowRange::deviceCounted(31,count);rejectBoth(invalid,s.stream);
    invalid=valid;invalid.projection.weights.n=63;rejectBoth(invalid,s.stream);
    invalid=valid;invalid.projection.weights.codebook_id=0;rejectBoth(invalid,s.stream);
    invalid=valid;invalid.projection.primary=invalid.projection.residual;rejectBoth(invalid,s.stream);
    invalid=valid;invalid.projection.rows=DeviceRowRange::deviceCounted(m,reinterpret_cast<const int32_t*>(valid.projection.output));rejectBoth(invalid,s.stream);
    invalid=valid;invalid.activated=nullptr;
    EXPECT_THROW(rocmHCResidualDownGroupedSingleGroupReference(invalid,s.stream),std::invalid_argument);
    invalid=valid;invalid.activated=valid.projection.output;
    EXPECT_THROW(rocmHCResidualDownGroupedSingleGroupReference(invalid,s.stream),std::invalid_argument);
    recording.finish();
   }
   // Zero captured operations proves rejection precedes publication, while
   // native byte observations retain every admitted bank and output guard.
   EXPECT_EQ(rejected->nodeCount(),0u);
   std::vector<GPUGraphKernelNodeInfo> rejectedNodes;std::string error;
   require(rejected->inspectKernelNodes(rejectedNodes,&error),"HC K-pair rejection inventory failed");
   EXPECT_TRUE(rejectedNodes.empty());rejected->reset();
   for(auto* tensor:{activated[1].get(),up[1].get()})
    EXPECT_EQ(s.download(static_cast<const float*>(tensor->gpu_data_ptr()),tensor->numel()),
     std::vector<float>(tensor->numel(),poison));
   for(const auto* bank:{valid.projection.primary,valid.projection.residual})
    EXPECT_EQ(s.download(bank,banks),std::vector<signed char>(banks,-37));
   for(const auto* bank:{valid.projection.primaryScales,valid.projection.residualScales})
    EXPECT_EQ(s.download(bank,scales),std::vector<float>(scales,poison));
   ASSERT_FALSE(::testing::Test::HasFailure());
  }
#endif
  if(profiling){
   // Match the paired run's full-live authored input, including its guard row;
   // setup uploads stay outside the selected retained execution interval.
   std::vector<float> input(std::size_t(m+1)*trunk);
   for(int row=0;row<=m;++row)for(int column=0;column<trunk;++column)
    input[std::size_t(row)*trunk+column]=row<m?
     (column%32==0?float(1+row%7)*3.125f:std::sin(float(row*11+column*7))*.37f):
     std::numeric_limits<float>::quiet_NaN();
   s.upload(static_cast<float*>(branchInput.gpu_data_ptr()),input);
  }
  if(kPair)require(actualUp[1]->prepareGraphLaunch(context.get(),s.stream),"HC K-pair original workspace admission failed");
  if(!profiling||observation==HC103Observation::ProfileProduction)capture(s.graph,0);
  if(!profiling||observation==HC103Observation::ProfileReference)capture(referenceGraph,1);
  if(kPair){
   using Key=std::tuple<std::uintptr_t,std::string,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned>;
   std::array<std::map<Key,int>,2> common;
   std::array<std::uintptr_t,2> producers{};
   for(int side=0;side<2;++side){
    if(inventory[side].empty())continue;
    const int kernelTemplate=side==0?0:1;
    const int stagedGroups=side==0?productionGroups:1;
    int projections=0,publishers=0;
    for(const auto&node:inventory[side]){
     if(node.name.find("projectGrouped")!=std::string::npos){
      ++projections;producers[side]=node.function_identity;
      EXPECT_TRUE(node.name.find("projectGrouped<"+std::to_string(kernelTemplate)+">")!=std::string::npos||
       node.name.find("projectGroupedILi"+std::to_string(kernelTemplate)+"EE")!=std::string::npos)<<node.name;
      EXPECT_EQ(node.grid_x,unsigned(((m+31)/32)*((trunk+63)/64)));
      EXPECT_EQ(node.grid_y,1u);EXPECT_EQ(node.grid_z,1u);
      EXPECT_EQ(node.block_x,256u);EXPECT_EQ(node.block_y,1u);EXPECT_EQ(node.block_z,1u);
      EXPECT_EQ(node.static_shared_memory_bytes,4608u*stagedGroups);
      EXPECT_EQ(node.dynamic_shared_memory_bytes,0u);
      EXPECT_EQ(node.local_memory_bytes_per_thread,0u);
     }else{
      if(node.name.find("publishHCResidualDown")!=std::string::npos)++publishers;
      ++common[side][Key{node.function_identity,node.name,node.grid_x,node.grid_y,node.grid_z,
       node.block_x,node.block_y,node.block_z}];
     }
    }
    EXPECT_EQ(projections,copies);EXPECT_EQ(publishers,copies);
   }
   if(!profiling){
    if(productionGroups!=1)EXPECT_NE(producers[0],producers[1]);
    EXPECT_EQ(common[0],common[1]);
   }
   ASSERT_FALSE(::testing::Test::HasFailure());
  }
  if(profiling){
   // Intercepted SDMA timing markers have a known SDK timestamp-owner defect.
   // This mode creates no events and executes only the selected retained chain.
   // Drain all setup copies at this explicit diagnostic boundary before replay.
   require(backend->synchronizeStream(s.stream,device.ordinal),"HC profile setup retirement failed");
   auto& selected=observation==HC103Observation::ProfileProduction?s.graph:referenceGraph;
   for(int warm=0;warm<8;++warm)require(selected->launch(),"HC profile warmup failed");
   for(int replay=0;replay<31;++replay)require(selected->launch(),"HC profile observation failed");
   require(backend->synchronizeStream(s.stream,device.ordinal),"HC profile terminal retirement failed");
   std::cout<<"HC103_KPAIR_PROFILE,"<<device.toString()<<','<<m<<','<<hidden<<','<<lowRank<<','
    <<(observation==HC103Observation::ProfileProduction?productionGroups:1)
    <<",role="<<(observation==HC103Observation::ProfileProduction?"production":"single_group_reference")
    <<",row_tile=32,column_tile=64,threads=256,warmup=8,observations=31,event_free=1\n";
   s.retire();referenceGraph.reset();return;
  }
  if(typed&&!kPair) {
   ASSERT_EQ(inventory[0].size()+copies,inventory[1].size());
   using Key=std::tuple<std::string,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned>;
   std::array<std::map<Key,int>,2> remaining;
   int fused=0,originalDownCount=0,originalPublish=0;
   std::array<int,2> readCounts{};
   for(int side=0;side<2;++side)for(const auto&node:inventory[side]) {
    const bool fusedProducer=node.name.find("publishHCResidualDown")!=std::string::npos;
    const bool frozenDown=node.name.find("downKernel")!=std::string::npos&&node.name.find("original")!=std::string::npos;
    const bool plainPublish=node.name.find("publish")!=std::string::npos&&node.name.find("q8_residual")!=std::string::npos&&!fusedProducer;
    const bool readNode=node.name.find("readKernel")!=std::string::npos;
    if(fusedProducer) {
     ASSERT_EQ(side,0);++fused;EXPECT_EQ(node.grid_x,unsigned((std::size_t(m)*lowRank+255)/256));
     EXPECT_EQ(node.grid_y,1u);EXPECT_EQ(node.grid_z,1u);EXPECT_EQ(node.block_x,256u);EXPECT_EQ(node.block_y,1u);EXPECT_EQ(node.block_z,1u);
    } else if(frozenDown){ASSERT_EQ(side,1);++originalDownCount;}
    else if(plainPublish){ASSERT_EQ(side,1);++originalPublish;}
    else if(readNode)++readCounts[side];
    else ++remaining[side][Key{node.name,node.grid_x,node.grid_y,node.grid_z,node.block_x,node.block_y,node.block_z}];
   }
   EXPECT_EQ(fused,copies);EXPECT_EQ(originalDownCount,copies);EXPECT_EQ(originalPublish,copies);
   EXPECT_EQ(readCounts[0],copies);EXPECT_EQ(readCounts[1],copies);
   EXPECT_EQ(remaining[0],remaining[1]);
  }
  GPUQ8ResidualCorrectionArgs frozen{originalUp,rows,device.is_cuda()?Q8ResidualQuantizerPolicy::CUDAFloatScaleNativeReciprocal:Q8ResidualQuantizerPolicy::ROCmFiniteHalfNativeReciprocal,
   static_cast<const float*>(activated[1]->gpu_data_ptr()),static_cast<float*>(up[1]->gpu_data_ptr()),frozenPrimary,frozenResidual,frozenPrimaryScales,frozenResidualScales};
  witnessGraph=worker.createGraphCapture(s.stream);{
   ScopedBackendGraphCapture recording(worker,*witnessGraph,"HC103 frozen independent publication witness");require(recording.begin(),"HC103 witness capture failed");
#ifdef HAVE_CUDA
   if(device.is_cuda())test103::cudaOriginalPublish(frozen,s.stream);
#endif
#ifdef HAVE_ROCM
   if(device.is_rocm())test103::rocmOriginalPublish(frozen,s.stream);
#endif
   recording.finish();
  }require(witnessGraph->instantiate(),"HC103 witness instantiate failed");
  std::vector<GPUGraphKernelNodeInfo>witnessNodes;std::string witnessError;require(witnessGraph->inspectKernelNodes(witnessNodes,&witnessError),"HC103 witness inventory failed");
  ASSERT_EQ(witnessGraph->nodeCount(),1u);ASSERT_EQ(witnessNodes.size(),1u);EXPECT_EQ(witnessNodes[0].local_memory_bytes_per_thread,0u);
  const std::array downSignatures{actualDown[0]->graphCaptureVariantSignature(),actualDown[1]->graphCaptureVariantSignature()};
  const std::array upSignatures{actualUp[0]->graphCaptureVariantSignature(),actualUp[1]->graphCaptureVariantSignature()};
  const auto retainedOwners=s.owners->size();
  const int replays=observation==HC103Observation::Correctness?20:1;
  for(int replay=0;replay<replays;++replay) {
   const int live=timing||replay%4==0?m:replay%4==1?std::min(m,1):replay%4==2?0:std::max(0,m-1);
   SCOPED_TRACE(::testing::Message()<<"replay="<<replay<<" live="<<live);
   std::vector<float> input(std::size_t(m+1)*trunk);
   for(int row=0;row<=m;++row)for(int c=0;c<trunk;++c)input[std::size_t(row)*trunk+c]=row<live?
    (c%32==0?float(1+row%7)*3.125f:std::sin(float(row*11+c*7))*.37f):std::numeric_limits<float>::quiet_NaN();
   s.upload(static_cast<float*>(typed?branchInput.gpu_data_ptr():normalized.gpu_data_ptr()),input);s.upload(count,std::vector<int32_t>{live});
   const float poison=-84931.25f-replay;
   for(int side=0;side<2;++side) {
    for(auto *tensor:{rawDown[side].get(),activated[side].get(),up[side].get(),read[side].get()})
     s.upload(static_cast<float*>(tensor->gpu_data_ptr()),std::vector<float>(tensor->numel(),poison));
    for(auto name:{kGPUProjectionPrimaryQ8,kGPUProjectionResidualQ8})s.upload(static_cast<signed char*>(upWorkspace[side]->value->getBuffer(name)),std::vector<signed char>(banks,91));
    for(auto name:{kGPUProjectionPrimaryQ8Scales,kGPUProjectionResidualQ8Scales})s.upload(static_cast<float*>(upWorkspace[side]->value->getBuffer(name)),std::vector<float>(scales,poison));
   }
   s.upload(frozenPrimary,std::vector<signed char>(banks,91));s.upload(frozenResidual,std::vector<signed char>(banks,91));
   s.upload(frozenPrimaryScales,std::vector<float>(scales,poison));s.upload(frozenResidualScales,std::vector<float>(scales,poison));
   require(s.graph->launch(),"HC103 public control replay failed");require(referenceGraph->launch(),"HC103 frozen primitive chain replay failed");
   require(witnessGraph->launch(),"HC103 frozen publisher replay failed");
   auto compare=[&](const char*name,const float *a,const float *b,std::size_t elements,int width) {
    SCOPED_TRACE(name);
    auto actual=s.download(a,elements),expected=s.download(b,elements);
    if(std::memcmp(actual.data(),expected.data(),elements*sizeof(float))!=0)for(std::size_t i=0;i<elements;++i) {
     uint32_t got,want;std::memcpy(&got,&actual[i],4);std::memcpy(&want,&expected[i],4);
     if(got!=want){ADD_FAILURE()<<name<<" first_word="<<i<<" row="<<i/width<<" observed=0x"<<std::hex<<got<<" expected=0x"<<want<<std::dec;break;}
    }
    ASSERT_EQ(std::memcmp(actual.data(),expected.data(),elements*sizeof(float)),0);
    for(std::size_t i=std::size_t(live)*width;i<elements;++i)ASSERT_EQ(std::memcmp(&actual[i],&poison,sizeof(float)),0)<<"inactive/guard word="<<i;
   };
   if(!alias)compare("rawDown",static_cast<float*>(rawDown[0]->gpu_data_ptr()),static_cast<float*>(rawDown[1]->gpu_data_ptr()),rawDown[0]->numel(),lowRank);
   compare("activated",static_cast<float*>(activated[0]->gpu_data_ptr()),static_cast<float*>(activated[1]->gpu_data_ptr()),activated[0]->numel(),lowRank);
   compare("up",static_cast<float*>(up[0]->gpu_data_ptr()),static_cast<float*>(up[1]->gpu_data_ptr()),up[0]->numel(),trunk);
   compare("read",static_cast<float*>(read[0]->gpu_data_ptr()),static_cast<float*>(read[1]->gpu_data_ptr()),read[0]->numel(),hidden);
   for(int side=0;side<2;++side) {
    for(auto bank:std::array{std::pair{kGPUProjectionPrimaryQ8,frozenPrimary},std::pair{kGPUProjectionResidualQ8,frozenResidual}}) {
     auto actual=s.download(static_cast<const signed char*>(upWorkspace[side]->value->getBuffer(bank.first)),banks),expected=s.download(bank.second,banks);
     ASSERT_EQ(std::memcmp(actual.data(),expected.data(),banks),0);for(std::size_t i=std::size_t(live)*lowRank;i<banks;++i)ASSERT_EQ(actual[i],91);
    }
    for(auto bank:std::array{std::pair{kGPUProjectionPrimaryQ8Scales,frozenPrimaryScales},std::pair{kGPUProjectionResidualQ8Scales,frozenResidualScales}}) {
     auto actual=s.download(static_cast<const float*>(upWorkspace[side]->value->getBuffer(bank.first)),scales),expected=s.download(bank.second,scales);
     ASSERT_EQ(std::memcmp(actual.data(),expected.data(),scales*sizeof(float)),0);for(std::size_t i=std::size_t(live)*lowRank/32;i<scales;++i)ASSERT_EQ(actual[i],poison);
    }
   }
   for(int side=0;side<2;++side) {
    EXPECT_EQ(actualDown[side]->graphCaptureVariantSignature(),downSignatures[side]);
    EXPECT_EQ(actualUp[side]->graphCaptureVariantSignature(),upSignatures[side]);
   }
   EXPECT_EQ(s.owners->size(),retainedOwners);
   DeviceNativeQuantisedMatrixDesc after;require(upEngine->exportNativeQuantisedMatrixDesc(after),"HC103 original source vanished");
   EXPECT_EQ(after.payload,originalUp.payload);EXPECT_EQ(after.scales,originalUp.scales);
   EXPECT_EQ(after.source_codebook_id,originalUp.source_codebook_id);
   if(kPair){
    EXPECT_EQ(after.source_identity_present,originalUp.source_identity_present);
    EXPECT_EQ(after.source_is_superblock,originalUp.source_is_superblock);
    EXPECT_EQ(after.codebook_id,originalUp.codebook_id);
    EXPECT_EQ(after.n,originalUp.n);EXPECT_EQ(after.k,originalUp.k);
    EXPECT_EQ(after.blocks_per_row,originalUp.blocks_per_row);
   }
   ASSERT_FALSE(::testing::Test::HasFailure());
  }
  if(kPair){
   EXPECT_EQ(savedDownBytes,test::q8_all_formats::originalBytes(*downSource));
   EXPECT_EQ(savedUpBytes,test::q8_all_formats::originalBytes(*upSource));
  }
  if(observation!=HC103Observation::Correctness) {
   struct Events {IBackend*b;int device;void*stream;void*first;void*last;/** @brief Drain pending event use before release on every exception path. */ ~Events(){if(!b->synchronizeStream(stream,device))std::terminate();if(last)b->destroyEvent(last,device);if(first)b->destroyEvent(first,device);}} events{backend,device.ordinal,s.stream,backend->createTimingEvent(device.ordinal),backend->createTimingEvent(device.ordinal)};
   require(events.first&&events.last,"HC103 timing event creation failed");
   auto sample=[&](IGPUGraphCapture &graph){require(backend->recordEvent(events.first,device.ordinal,s.stream),"HC103 start event failed");require(graph.launch(),"HC103 timed replay failed");require(backend->recordEvent(events.last,device.ordinal,s.stream),"HC103 stop event failed");require(backend->waitForEvent(events.last,device.ordinal),"HC103 terminal timing observation failed");float ms=0;require(backend->eventElapsedTimeMs(events.first,events.last,device.ordinal,&ms),"HC103 elapsed event failed");require(std::isfinite(ms)&&ms>0,"HC103 invalid event duration");return ms*1000.f/copies;};
   for(int warm=0;warm<5;++warm){(void)sample(*s.graph);(void)sample(*referenceGraph);}
   if(kPair){
    const char* sourceName=format?format->label:"Q8_0";
    std::cout<<"HC103_KPAIR_CASE,"<<device.toString()<<','<<m<<','<<hidden<<','<<lowRank<<','<<alias<<','<<copies
     <<",production_vs_single_group_reference,production_staged_groups="<<productionGroups
     <<",reference_staged_groups=1,same_fused_publication_and_read,ordinary_full_down,counted_two_term_up,source="<<sourceName
     <<",sample_columns=production0/production1/reference0/reference1/production_mean/reference_mean\n";
    for(int pair=0;pair<31;++pair){
     std::array<float,2> production{},reference{};
     const std::array order=pair&1?std::array{1,0,0,1}:std::array{0,1,1,0};
     std::array<int,2> next{};
     for(const int side:order){
      const float us=sample(side==0?*s.graph:*referenceGraph);
      (side==0?production:reference)[next[side]++]=us;
     }
     std::cout<<"HC103_KPAIR_PAIR,"<<device.toString()<<','<<m<<','<<hidden<<','<<lowRank<<','<<sourceName<<','<<pair
      <<','<<(pair&1?"BAAB":"ABBA")<<','<<production[0]<<','<<production[1]<<','<<reference[0]<<','<<reference[1]
      <<','<<(production[0]+production[1])*.5f<<','<<(reference[0]+reference[1])*.5f<<",complete_chain_us\n";
    }
   }else{
   if(typed)std::cout<<"HC103_CANDIDATE_CASE,"<<device.toString()<<','<<m<<','<<hidden<<','<<lowRank<<','<<alias<<','<<copies
    <<",actual_typed_builder_vs_frozen_original,ordinary_full_down,counted_two_term_up\n";
   for(int pair=0;pair<31;++pair){float a,b;if(pair&1){b=sample(*referenceGraph);a=sample(*s.graph);}else{a=sample(*s.graph);b=sample(*referenceGraph);}std::cout<<(typed?"HC103_CANDIDATE_PAIR,":"HC103_CONTROL_PAIR,")<<device.toString()<<','<<m<<','<<hidden<<','<<lowRank<<','<<pair<<','<<a<<','<<b<<'\n';}
   }
  }
  s.retire();referenceGraph.reset();witnessGraph.reset();for(int side=0;side<2;++side){downStages[side]->unbindWorkspace();upStages[side]->unbindWorkspace();}
 }).get();
}
/** @brief Preserve every original control selector while sharing candidate fixture plumbing.
 * @param device Exact participant.
 * @param m Physical rows.
 * @param hidden Per-branch width.
 * @param lowRank HC rank.
 * @param alias Exact raw/activated alias selection.
 * @param timing Existing disabled event observation selection.
 * @param format Optional unchanged original format override.
 */
void proveHC103Control(DeviceId device,int m,int hidden,int lowRank,bool alias,bool timing=false,
 const test::QuantizedVerifierFormatCase* format=nullptr) {
 proveHC103Chain(device,m,hidden,lowRank,alias,HC103Path::Control,
  timing?HC103Observation::Paired:HC103Observation::Correctness,format);
}
/**
 * @brief Reuse the complete HC103 chain for one original signed-Q8 source alias.
 * @param sourceName Existing Q8_0, Q8_1 or Q8_K registry name.
 * @param observation Byte lifecycle, paired native events or one event-free profile.
 * @param measuredRows Exact opt-in complete-chain row extent; existing cases retain512.
 * @details Q8_0/Q8_1 include actual M512/H2560/R320 geometry plus
 * H64/R96 boundary cases with an odd three-K32-group tail. Original Q8_K
 * requires complete K256 source blocks and is explicitly H64/R256 geometry,
 * never a relabeled R320 tensor. Only these new cases consume the existing
 * LLAMINAR_TEST_ROCM_DEVICE selector; historical fixture ordinals stay intact.
 */
void proveHC103KPair(const char* sourceName,HC103Observation observation,int measuredRows=512) {
#ifdef HAVE_ROCM
 int ordinal=0;
 if(const char* selected=std::getenv("LLAMINAR_TEST_ROCM_DEVICE")){
  const char* end=selected+std::strlen(selected);
  const auto parsed=std::from_chars(selected,end,ordinal);
  require(parsed.ec==std::errc{}&&parsed.ptr==end&&ordinal>=0,
   "LLAMINAR_TEST_ROCM_DEVICE requires a nonnegative native ordinal");
 }
 const auto device=DeviceId::rocm(ordinal);
 auto* backend=getBackendFor(device);
 require(backend&&backend->deviceCount()>ordinal,"HC K-pair native participant unavailable without visibility changes");
 const auto& hardware=rocmHardwareMetadata().at(ordinal);
 const auto& format=test::quantizedMoEVerifierFormat(sourceName);
 require(format.device_execution_codebook_id==19,"HC K-pair requires an original signed-Q8 source alias");
 const bool superblock=format.tensor_type==TensorType::Q8_K;
 std::ostringstream uuid;
 uuid<<std::hex<<std::setfill('0');
 for(const auto byte:hardware.uuid)uuid<<std::setw(2)<<unsigned(byte);
 std::cout<<"HC103_KPAIR_HARDWARE,"<<device.toString()<<','<<hardware.pci_address<<','<<hardware.isa_target
  <<",uuid="<<uuid.str()<<",source="<<sourceName<<",source_codebook="<<unsigned(format.source_codebook_id)
  <<",source_superblock="<<format.source_is_superblock<<'\n';
 if(observation==HC103Observation::Correctness){
  for(int rows:{32,33,64,65,511,512,513})for(bool alias:{false,true})
   proveHC103Chain(device,rows,64,superblock?256:96,alias,HC103Path::GroupedKPair,observation,&format);
  if(!superblock)proveHC103Chain(device,32,64,32,false,HC103Path::GroupedKPair,observation,&format);
 }
 proveHC103Chain(device,measuredRows,superblock?64:2560,superblock?256:320,false,
  HC103Path::GroupedKPair,observation,&format);
#else
 (void)sourceName;(void)observation;(void)measuredRows;
 throw std::runtime_error("HC K-pair diagnostic requires the compiled ROCm backend");
#endif
}
/** @brief Exercise actual HC preparation and native/map retirement boundaries.
 * @param device Exact backend participant; no projection arithmetic is substituted.
 * Repeated preparation/reset retains original native records. Invalid release
 * must fail before native free; valid retirement occurs only after the stage.
 */
void proveHC103CapturedStorageOwnership(DeviceId device) {
 auto* backend=getBackendFor(device);require(bool(backend),"HC ownership backend missing");
 auto& worker=GPUDeviceContextPool::instance().getContext(device);
 auto context=IDeviceContext::create(device,1);
 worker.submitAsync([&] {
  const auto stream=ExplicitGPUStream(worker.defaultStream()).get();
  // Exercise actual backend record lookup, including interior spans. The
  // owner precedes every borrow so exceptional unwinding retires borrowers first.
  auto release=[&](void* pointer){if(pointer)backend->free(pointer,device.ordinal);};
  std::unique_ptr<void,decltype(release)> allocation(backend->allocate(256,device.ordinal),release);
  require(bool(allocation),"HC native span witness missing");
  auto* base=static_cast<std::byte*>(allocation.get());
  {
   auto interior=backend->retainNativeAllocation(base+32,64,device.ordinal);
   auto identical=backend->retainNativeAllocation(base+32,64,device.ordinal);
   EXPECT_TRUE(interior.sameStorage(identical));
   EXPECT_THROW(backend->retainNativeAllocation(base+224,64,device.ordinal),std::invalid_argument);
   EXPECT_THROW(backend->retainNativeAllocation(base,0,device.ordinal),std::invalid_argument);
   EXPECT_THROW(backend->retainNativeAllocation(base,64,backend->deviceCount()),std::invalid_argument);
   EXPECT_THROW(backend->free(base+32,device.ordinal),std::invalid_argument);
   EXPECT_THROW(backend->free(base,device.ordinal),std::logic_error);
  }
  allocation.reset();
  auto sourceBytes=nativeWeights(32,32);Q8_0Tensor source({32,32},sourceBytes);
  auto prepared=test::makeGpuPreparedGemm(&source,device,"hc103.ownership",ModelContextId{803107});
  // Mapped activated/C are outside the admitted API; mapped raw stays supported.
  for(const auto [mapped,alias]:std::array{std::pair{false,false},std::pair{false,true},std::pair{true,false}}) {
   FP32Tensor nativeRaw({1,32},DeviceId::cpu()),activated({1,32},DeviceId::cpu()),output({1,32},DeviceId::cpu());
   auto mappedRaw=mapped?FP32Tensor::createMapped({1,32},device):nullptr;
   require(!mapped||bool(mappedRaw),"HC mapped raw owner allocation failed");
   TensorBase* raw=mapped?mappedRaw.get():alias?&activated:&nativeRaw;
   // One physical host upload source is initialized/published only once, even
   // when the typed raw input is exactly the activated output object.
   std::vector<TensorBase*> distinct{raw};
   if(raw!=&activated)distinct.push_back(&activated);
   distinct.push_back(&output);
   for(auto* tensor:distinct)
    if(!tensor->isMapped())TransferEngine::allocateDeviceStorage(tensor,device);
   GEMMStage::Params params{.device_id=device,.A=raw,.B=&source,.C=&output,
    .hc_residual_down=GEMMStage::HCResidualDownPublication{&activated,BufferId::GATE_PROJ},
    .m=1,.n=32,.k=32,.transpose_B=false,
    .a_buffer_id=alias?BufferId::GATE_PROJ:BufferId::NORMALIZED,.c_buffer_id=BufferId::ATTN_OUTPUT,
    .prepared_ref=prepared.ref,.prepared_store=prepared.store.get(),
    .quantized_operand_precision=QuantizedActivationOperandPrecision::TwoTermQ8};
   auto stage=std::make_unique<GEMMStage>(params);stage->setGPUStream(stream);
   Workspace workspace(device,stage->getWorkspaceRequirements(1,32,32));
   /** @brief Close the engine borrow before workspace retirement on every exit. */
   struct Unbind { std::unique_ptr<GEMMStage>& stage;
    ~Unbind(){if(stage)stage->unbindWorkspace();}
   } unbind{stage};
   stage->bindWorkspace(workspace.value.get());
   const auto signature=stage->graphCaptureVariantSignature();
   require(stage->prepareGraphLaunch(context.get(),stream),"HC ownership first prepare failed");
   Storage captured{device,backend,stream};
   CapturedWorkRetirement retirement(captured);
   const auto initializeFinite=[&] {
    // Called only before capture or after an explicit terminal observation.
    // Each distinct host source remains immutable until its upload completes.
    for(auto* tensor:distinct) {
     std::fill_n(tensor->mutable_data(),32,0.125f);
     TransferEngine::prepareDeviceInput(tensor,device,stream);
     TransferEngine::requireDeviceInput(tensor,device,stream);
    }
   };
   initializeFinite();
   captured.graph=worker.createGraphCapture(stream);
   require(bool(captured.graph),"HC ownership graph missing");
   {
    ScopedBackendGraphCapture recording(worker,*captured.graph,"HC retained storage ownership");
    require(recording.begin(),"HC ownership capture begin failed");
    require(stage->execute(context.get()),"HC ownership actual typed stage failed");
    recording.finish();
   }
   require(captured.graph->instantiate(),"HC ownership graph instantiate failed");
   std::vector<GPUGraphKernelNodeInfo> nodes;std::string inventoryError;
   require(captured.graph->inspectKernelNodes(nodes,&inventoryError),"HC ownership graph inventory failed");
   EXPECT_FALSE(nodes.empty());EXPECT_EQ(captured.graph->nodeCount(),nodes.size());
   const auto observeBytes=[&] {
    return std::array{
     captured.download(static_cast<const std::uint8_t*>(activated.gpu_data_ptr()),32*sizeof(float)),
     captured.download(static_cast<const std::uint8_t*>(output.gpu_data_ptr()),32*sizeof(float))};
   };
   // Reset clears the host stream, not the retained graph/storage identity.
   // Unbound submission must still reject before a kernel can be launched.
   const auto resetAndRebind=[&](int history) {
    switch(history%3) {
     case 0:stage->resetSessionState();break;
     case 1:stage->resetSessionStatePreservingCapturedReplay();break;
     case 2:stage->resetSessionStatePreservingLazyInitialization();break;
    }
    EXPECT_FALSE(stage->hasGPUStream());
    EXPECT_EQ(stage->graphCaptureVariantSignature(),signature);
    EXPECT_THROW(stage->prepareGraphLaunch(context.get(),stream),std::logic_error);
    EXPECT_THROW(stage->execute(context.get()),std::logic_error);
    stage->setGPUStream(stream);
    require(stage->prepareGraphLaunch(context.get(),stream),"HC reset exact rebind failed");
    EXPECT_EQ(stage->graphCaptureVariantSignature(),signature);
   };
   // Same retained graph, same finite starting bytes, same number of replays:
   // first observe the valid history without rejected retirement attempts.
   for(int history=0;history<20;++history) {
    require(captured.graph->launch(),"HC no-rejection retained replay failed");
    EXPECT_EQ(stage->graphCaptureVariantSignature(),signature);
    require(stage->prepareGraphLaunch(context.get(),stream),"HC no-rejection reprepare failed");
    resetAndRebind(history);
   }
   const auto noRejection=observeBytes();
   initializeFinite();
   const auto originalAccounting=backend->deviceAllocationAccounting(device.ordinal);
   require(originalAccounting.supported,"HC canonical allocation accounting unavailable");
   const auto requireSameAccounting=[&] {
    const auto current=backend->deviceAllocationAccounting(device.ordinal);
    EXPECT_TRUE(current.supported);
    EXPECT_EQ(current.active_allocations,originalAccounting.active_allocations);
    EXPECT_EQ(current.active_bytes,originalAccounting.active_bytes);
   };
   for(int history=0;history<20;++history) {
    require(captured.graph->launch(),"HC ownership retained replay failed");
    EXPECT_EQ(stage->graphCaptureVariantSignature(),signature);
    require(stage->prepareGraphLaunch(context.get(),stream),"HC ownership reprepare failed");
    for(auto* tensor:distinct) {
     const void* original=tensor->gpu_data_ptr();
     if(tensor->isMapped()) {
      EXPECT_THROW(backend->free(tensor->gpu_data_ptr(),device.ordinal),std::invalid_argument);
     } else {
      EXPECT_THROW(backend->free(tensor->gpu_data_ptr(),device.ordinal),std::logic_error);
     }
     EXPECT_EQ(tensor->gpu_data_ptr(),original);
     requireSameAccounting();
    }
    // Native borrowers reject their actual allocation base; mapped aliases are
    // excluded by the same native-map membership boundary before runtime free.
    resetAndRebind(history);
   }
   // Terminal byte observation proves rejected native retirement did not
   // change valid numerical state; this is a lifecycle metamorphic witness,
   // not an independent arithmetic oracle or a timing measurement.
   EXPECT_EQ(observeBytes(),noRejection);
   requireSameAccounting();
   captured.retire();
   // Public Tensor release can observe/download values or retire an event before
   // its final native-free rejection. Exercise that invalid operation only
   // after captured work retires: physical ownership and pointers must survive,
   // but no zero-side-effect concurrent coherence guarantee is claimed.
   for(auto* tensor:distinct) {
    const void* original=tensor->gpu_data_ptr();
    if(tensor->isMapped()) {
     EXPECT_THROW(tensor->releaseDeviceMemory(),std::invalid_argument);
    } else {
     EXPECT_THROW(tensor->releaseDeviceMemory(),std::logic_error);
    }
    EXPECT_EQ(tensor->gpu_data_ptr(),original);
    requireSameAccounting();
   }
   stage->unbindWorkspace();stage.reset();
   for(auto* tensor:{static_cast<TensorBase*>(&nativeRaw),static_cast<TensorBase*>(&activated),static_cast<TensorBase*>(&output)})
    if(tensor->gpu_data_ptr())EXPECT_TRUE(tensor->releaseDeviceMemory());
  }
  // Retaining a mapped input preserves its original owner even if the tensor
  // object retires. No submitted GPU work remains in this lifecycle witness.
  auto mapped=FP32Tensor::createMapped({1,32},device);
  require(bool(mapped),"HC mapped retention witness missing");
  std::weak_ptr<MappedHostTransferRegion> originalRegion=mapped->mappedHostTransferRegion();
  auto retained=TransferEngine{}.retainCapturedDeviceStorage(*mapped,device,mapped->size_bytes());
  mapped.reset();EXPECT_FALSE(originalRegion.expired());
  retained.reset();EXPECT_TRUE(originalRegion.expired());
 }).get();
}

#include "NativeQ8SmallNTileProbe.inc"

/** @brief Calibrate grouped rows, real HC shape and exact alias before extraction. */
void proveHC103Backend(DeviceId device) {
 for(int m:{1,3,9,17,33,513})for(bool alias:{false,true})proveHC103Control(device,m,8,32,alias);
 for(int m:{1,9,512})proveHC103Control(device,m,2560,320,false);
}
}
#ifdef HAVE_CUDA
TEST(Q8ResidualCorrectionDevice, CUDA){prove(DeviceId::cuda(0));}
/** @test Actual CUDA HC-up correction preserves a nonzero cancellation-sensitive read. */
TEST(Q8ResidualCorrectionDevice, CUDAHCUpCancellationRead){proveCancellationRead(DeviceId::cuda(0));}
TEST(Q8ResidualCorrectionDevice, CUDAGDNOriginalSource){proveGDN(DeviceId::cuda(0));}
TEST(Q8ResidualCorrectionDevice, CUDAFAActualComposer){proveGDN(DeviceId::cuda(0),true);}
#endif
#ifdef HAVE_ROCM
TEST(Q8ResidualCorrectionDevice, ROCm){prove(DeviceId::rocm(0));}
/** @test Actual ROCm HC-up correction preserves a nonzero cancellation-sensitive read. */
TEST(Q8ResidualCorrectionDevice, ROCmHCUpCancellationRead){proveCancellationRead(DeviceId::rocm(0));}
TEST(Q8ResidualCorrectionDevice, ROCmGDNOriginalSource){proveGDN(DeviceId::rocm(0));}
TEST(Q8ResidualCorrectionDevice, ROCmFAActualComposer){proveGDN(DeviceId::rocm(0),true);}
#endif

#ifdef HAVE_CUDA
/** @test Original CUDA prepared HC activation/publication/up/read byte calibration. */
TEST(Q8ResidualCorrectionDevice,CUDAHC103OriginalControlChain){proveHC103Backend(DeviceId::cuda(0));}
/** @test Every original source codebook retains the independent CUDA publication bytes. */
TEST(Q8ResidualCorrectionDevice,CUDAHC103OriginalAllFormats){for(const auto &f:test::quantizedVerifierFormats())for(bool alias:{false,true})proveHC103Control(DeviceId::cuda(0),3,64,256,alias,false,&f);}
/** @test Explicit Release complete-chain A/A calibration; no candidate result is invented. */
TEST(Q8ResidualCorrectionDevice,DISABLED_CUDAHC103OriginalPairedChainEvents){for(int m:{1,9,512})proveHC103Control(DeviceId::cuda(0),m,2560,320,false,true);}
#endif
#ifdef HAVE_ROCM
/** @test Original HIP prepared HC activation/publication/up/read byte calibration. */
TEST(Q8ResidualCorrectionDevice,ROCmHC103OriginalControlChain){proveHC103Backend(DeviceId::rocm(0));}
/** @test Every original source codebook retains the independent HIP publication bytes. */
TEST(Q8ResidualCorrectionDevice,ROCmHC103OriginalAllFormats){for(const auto &f:test::quantizedVerifierFormats())for(bool alias:{false,true})proveHC103Control(DeviceId::rocm(0),3,64,256,alias,false,&f);}
/** @test Explicit Release complete-chain A/A calibration, all raw balanced samples. */
TEST(Q8ResidualCorrectionDevice,DISABLED_ROCmHC103OriginalPairedChainEvents){for(int m:{1,9,512})proveHC103Control(DeviceId::rocm(0),m,2560,320,false,true);}
#endif

#ifdef HAVE_CUDA
/** @test Actual CUDA HC stage pins native records and existing mapped raw owners. */
TEST(Q8ResidualCorrectionDevice,CUDAHC103CapturedStorageOwnership){proveHC103CapturedStorageOwnership(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
/** @test Actual HIP HC stage pins native records and existing mapped raw owners. */
TEST(Q8ResidualCorrectionDevice,ROCmHC103CapturedStorageOwnership){proveHC103CapturedStorageOwnership(DeviceId::rocm(0));}
#endif

#ifdef HAVE_CUDA
/** @test Actual declarative HC TwoTerm path preserves original odd/live/reset bytes. */
TEST(Q8ResidualCorrectionDevice,CUDAHC103TypedBuilderChain) {
 for(int m:{1,3,9,17,33,513})for(bool alias:{false,true})
  proveHC103Chain(DeviceId::cuda(0),m,8,32,alias,HC103Path::Typed);
 for(int m:{1,9,512})proveHC103Chain(DeviceId::cuda(0),m,2560,320,false,HC103Path::Typed);
}
/** @test Every original registry source executes M1/M512 and exact alias/disjoint typed publication. */
TEST(Q8ResidualCorrectionDevice,CUDAHC103TypedAllFormats) {
 for(const auto& format:test::quantizedVerifierFormats())for(bool alias:{false,true})for(int m:{1,3,512})
  proveHC103Chain(DeviceId::cuda(0),m,64,256,alias,HC103Path::Typed,HC103Observation::Correctness,&format);
}
#endif

#ifdef HAVE_ROCM
/** @test Actual declarative HC TwoTerm path preserves original odd/live/reset bytes. */
TEST(Q8ResidualCorrectionDevice,ROCmHC103TypedBuilderChain) {
 for(int m:{1,3,9,17,33,513})for(bool alias:{false,true})
  proveHC103Chain(DeviceId::rocm(0),m,8,32,alias,HC103Path::Typed);
 for(int m:{1,9,512})proveHC103Chain(DeviceId::rocm(0),m,2560,320,false,HC103Path::Typed);
}
/** @test Every original registry source executes M1/M512 and exact alias/disjoint typed publication. */
TEST(Q8ResidualCorrectionDevice,ROCmHC103TypedAllFormats) {
 for(const auto& format:test::quantizedVerifierFormats())for(bool alias:{false,true})for(int m:{1,3,512})
  proveHC103Chain(DeviceId::rocm(0),m,64,256,alias,HC103Path::Typed,HC103Observation::Correctness,&format);
}
#endif

#ifdef HAVE_CUDA
/** @test Opt-in one-shape actual builder/frozen-original complete-chain native events. */
TEST(Q8ResidualCorrectionDevice,DISABLED_CUDAHC103CandidateM1PairedChainEvents) {
 proveHC103Chain(DeviceId::cuda(0),1,2560,320,false,HC103Path::Typed,HC103Observation::Paired);
}
#endif

#ifdef HAVE_CUDA
/** @test Opt-in one-shape actual builder/frozen-original complete-chain native events. */
TEST(Q8ResidualCorrectionDevice,DISABLED_CUDAHC103CandidateM512PairedChainEvents) {
 proveHC103Chain(DeviceId::cuda(0),512,2560,320,false,HC103Path::Typed,HC103Observation::Paired);
}
#endif

#ifdef HAVE_ROCM
/** @test Opt-in one-shape actual builder/frozen-original complete-chain native events. */
TEST(Q8ResidualCorrectionDevice,DISABLED_ROCmHC103CandidateM1PairedChainEvents) {
 proveHC103Chain(DeviceId::rocm(0),1,2560,320,false,HC103Path::Typed,HC103Observation::Paired);
}
#endif

#ifdef HAVE_ROCM
/** @test Opt-in one-shape actual builder/frozen-original complete-chain native events. */
TEST(Q8ResidualCorrectionDevice,DISABLED_ROCmHC103CandidateM512PairedChainEvents) {
 proveHC103Chain(DeviceId::rocm(0),512,2560,320,false,HC103Path::Typed,HC103Observation::Paired);
}

/** @test Actual HC-up shape plus row boundaries retain all Q8_0 bytes and graph identities. */
TEST(Q8ResidualCorrectionDevice,ROCmHCUpKPairCapturedReplayQ8_0) {
 proveHC103KPair("Q8_0",HC103Observation::Correctness);
}
/** @test The same K-group staging comparison preserves the independent Q8_1 source. */
TEST(Q8ResidualCorrectionDevice,ROCmHCUpKPairCapturedReplayQ8_1) {
 proveHC103KPair("Q8_1",HC103Observation::Correctness);
}
/** @test Q8_K uses its real K256 source geometry through partial, empty and reset replay. */
TEST(Q8ResidualCorrectionDevice,ROCmHCUpKPairCapturedReplayQ8_K) {
 proveHC103KPair("Q8_K",HC103Observation::Correctness);
}
/** @test Opt-in complete M512/H2560/R320 chain events, actual production versus original single-group reference. */
TEST(Q8ResidualCorrectionDevice,DISABLED_ROCmHCUpKPairM512PairedChainQ8_0) {
 proveHC103KPair("Q8_0",HC103Observation::Paired);
}
/** @test Opt-in identical physical comparison over the independent Q8_1 source. */
TEST(Q8ResidualCorrectionDevice,DISABLED_ROCmHCUpKPairM512PairedChainQ8_1) {
 proveHC103KPair("Q8_1",HC103Observation::Paired);
}
/** @test Opt-in Q8_K M512/H64/R256 chain events, explicitly separate source geometry. */
TEST(Q8ResidualCorrectionDevice,DISABLED_ROCmHCUpKPairM512PairedChainQ8_K) {
 proveHC103KPair("Q8_K",HC103Observation::Paired);
}
/** @test Opt-in matched Q8_0 row spectrum at actual N10240/K320, one complete chain per sample. */
TEST(Q8ResidualCorrectionDevice,DISABLED_ROCmHCUpKPairRowSpectrumQ8_0) {
 for(int rows:{32,128,512,1024,2048})
  proveHC103KPair("Q8_0",HC103Observation::Paired,rows);
}
/** @test Event-free profile of compiled production staging only, Q8_0 actual M512 HC-up shape. */
TEST(Q8ResidualCorrectionDevice,DISABLED_ROCmHCUpKPairM512ProfileProductionQ8_0) {
 proveHC103KPair("Q8_0",HC103Observation::ProfileProduction);
}
/** @test Event-free profile of original single-group reference only, Q8_0 actual M512 HC-up shape. */
TEST(Q8ResidualCorrectionDevice,DISABLED_ROCmHCUpKPairM512ProfileReferenceQ8_0) {
 proveHC103KPair("Q8_0",HC103Observation::ProfileReference);
}
/** @test Event-free single-group production profile retains independent Q8_1 source identity. */
TEST(Q8ResidualCorrectionDevice,DISABLED_ROCmHCUpKPairM512ProfileProductionQ8_1) {
 proveHC103KPair("Q8_1",HC103Observation::ProfileProduction);
}
/** @test Event-free original single-group reference profile retains independent Q8_1 source identity. */
TEST(Q8ResidualCorrectionDevice,DISABLED_ROCmHCUpKPairM512ProfileReferenceQ8_1) {
 proveHC103KPair("Q8_1",HC103Observation::ProfileReference);
}
/** @test Event-free single-group production profile of explicitly separate Q8_K H64/R256 geometry. */
TEST(Q8ResidualCorrectionDevice,DISABLED_ROCmHCUpKPairM512ProfileProductionQ8_K) {
 proveHC103KPair("Q8_K",HC103Observation::ProfileProduction);
}
/** @test Event-free original single-group reference profile of explicitly separate Q8_K H64/R256 geometry. */
TEST(Q8ResidualCorrectionDevice,DISABLED_ROCmHCUpKPairM512ProfileReferenceQ8_K) {
 proveHC103KPair("Q8_K",HC103Observation::ProfileReference);
}
#endif

#ifdef HAVE_ROCM
/** @test Original prepared shared-gate geometry retains bytes across both tiles and448/64 reset histories. */
TEST(Q8ResidualCorrectionDevice,ROCmNativeQ8N320TileByteResetK2560) {
 proveNativeQ8SmallNTiles(DeviceId::rocm(0),2560);
}
/** @test Original prepared HC-down geometry retains bytes across both tiles and448/64 reset histories. */
TEST(Q8ResidualCorrectionDevice,ROCmNativeQ8N320TileByteResetK10240) {
 proveNativeQ8SmallNTiles(DeviceId::rocm(0),10240);
}
/** @test Opt-in paired events for one exact prepared source/M; no production policy selection. */
TEST(Q8ResidualCorrectionDevice,DISABLED_ROCmNativeQ8N320M64K2560TileEvents) {
 proveNativeQ8SmallNTiles(DeviceId::rocm(0),2560,64);
}
/** @test Opt-in paired events for one exact prepared source/M; no production policy selection. */
TEST(Q8ResidualCorrectionDevice,DISABLED_ROCmNativeQ8N320M64K10240TileEvents) {
 proveNativeQ8SmallNTiles(DeviceId::rocm(0),10240,64);
}
/** @test Opt-in paired events for one exact prepared source/M; no production policy selection. */
TEST(Q8ResidualCorrectionDevice,DISABLED_ROCmNativeQ8N320M448K2560TileEvents) {
 proveNativeQ8SmallNTiles(DeviceId::rocm(0),2560,448);
}
/** @test Opt-in paired events for one exact prepared source/M; no production policy selection. */
TEST(Q8ResidualCorrectionDevice,DISABLED_ROCmNativeQ8N320M448K10240TileEvents) {
 proveNativeQ8SmallNTiles(DeviceId::rocm(0),10240,448);
}
#endif

#ifdef HAVE_ROCM
/** @test Installed Auto and MT16 match explicit MT32 for each original Q8 source through dirty448/64 reuse. */
TEST(Q8ResidualCorrectionDevice,ROCmNativeQ8N320RowTileByteResetK2560) {
 proveNativeQ8SmallNTiles(DeviceId::rocm(0),2560,0,NativeQ8TileComparison::Rows);
 for(const char* name:{"Q8_1","Q8_K"}){
  const auto& format=test::quantizedMoEVerifierFormat(name);
  proveNativeQ8SmallNTiles(DeviceId::rocm(0),2560,0,NativeQ8TileComparison::Rows,&format);
 }
}
/** @test HC-down retains every original Q8 source tuple and MT32 control through the same row histories. */
TEST(Q8ResidualCorrectionDevice,ROCmNativeQ8N320RowTileByteResetK10240) {
 proveNativeQ8SmallNTiles(DeviceId::rocm(0),10240,0,NativeQ8TileComparison::Rows);
 for(const char* name:{"Q8_1","Q8_K"}){
  const auto& format=test::quantizedMoEVerifierFormat(name);
  proveNativeQ8SmallNTiles(DeviceId::rocm(0),10240,0,NativeQ8TileComparison::Rows,&format);
 }
}
/** @test One original source/M with balanced isolated MT32/MT16 graph event pairs. */
TEST(Q8ResidualCorrectionDevice,DISABLED_ROCmNativeQ8N320M64K2560RowTileEvents) {
 proveNativeQ8SmallNTiles(DeviceId::rocm(0),2560,64,NativeQ8TileComparison::Rows);
}
/** @test One original source/M with balanced isolated MT32/MT16 graph event pairs. */
TEST(Q8ResidualCorrectionDevice,DISABLED_ROCmNativeQ8N320M64K10240RowTileEvents) {
 proveNativeQ8SmallNTiles(DeviceId::rocm(0),10240,64,NativeQ8TileComparison::Rows);
}
/** @test One original source/M with balanced isolated MT32/MT16 graph event pairs. */
TEST(Q8ResidualCorrectionDevice,DISABLED_ROCmNativeQ8N320M448K2560RowTileEvents) {
 proveNativeQ8SmallNTiles(DeviceId::rocm(0),2560,448,NativeQ8TileComparison::Rows);
}
/** @test One original source/M with balanced isolated MT32/MT16 graph event pairs. */
TEST(Q8ResidualCorrectionDevice,DISABLED_ROCmNativeQ8N320M448K10240RowTileEvents) {
 proveNativeQ8SmallNTiles(DeviceId::rocm(0),10240,448,NativeQ8TileComparison::Rows);
}

/** @test Actual M512 HC-down bytes, dirty partial/reset histories and guards for Q8_0. */
TEST(Q8ResidualCorrectionDevice,ROCmNativeQ8N320M512K10240TileByteResetQ8_0) {
 proveNativeQ8PrefillTiles("Q8_0",0);
}
/** @test Actual M512 HC-down bytes retain the independent Q8_1 source descriptor. */
TEST(Q8ResidualCorrectionDevice,ROCmNativeQ8N320M512K10240TileByteResetQ8_1) {
 proveNativeQ8PrefillTiles("Q8_1",0);
}
/** @test Actual M512 HC-down bytes retain the original Q8_K superblock source. */
TEST(Q8ResidualCorrectionDevice,ROCmNativeQ8N320M512K10240TileByteResetQ8_K) {
 proveNativeQ8PrefillTiles("Q8_K",0);
}
/** @test Opt-in forced NT128/MT32 versus NT64/MT16 and NT64/MT32 event pairs, Q8_0. */
TEST(Q8ResidualCorrectionDevice,DISABLED_ROCmNativeQ8N320M512K10240TileEventsQ8_0) {
 proveNativeQ8PrefillTiles("Q8_0",512);
}
/** @test Opt-in identical physical comparisons through the original Q8_1 source. */
TEST(Q8ResidualCorrectionDevice,DISABLED_ROCmNativeQ8N320M512K10240TileEventsQ8_1) {
 proveNativeQ8PrefillTiles("Q8_1",512);
}
/** @test Opt-in identical physical comparisons through the original Q8_K source. */
TEST(Q8ResidualCorrectionDevice,DISABLED_ROCmNativeQ8N320M512K10240TileEventsQ8_K) {
 proveNativeQ8PrefillTiles("Q8_K",512);
}
#endif
