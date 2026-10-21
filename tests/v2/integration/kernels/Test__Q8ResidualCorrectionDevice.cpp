/**
 * @file Test__Q8ResidualCorrectionDevice.cpp
 * @brief Captured prepared-engine/stage proof of native two-term Q8, including actual FA Stage5 composition.
 *
 * Production packers preserve the native Q8_0 payload/scale bits. Their exact
 * device pointers enter a normal PreparedWeightStore handle and GEMMStage.
 * One retained graph replays large/small/empty live prefixes without recapture.
 * Independent prepared serial calls are the byte oracle; actual SingleQ8
 * dispatches authenticate primary bytes/scales, including adversarial blocks.
 * A separate authored HC-up/read composition preserves a nonzero signal lost
 * by SingleQ8, with independent FP64 source arithmetic and live-prefix replay.
 */
#include <gtest/gtest.h>
#include "../../utils/Qwen4ExpFAOutputStageFixture.h"
#include "kernels/common/Q8ResidualGroupedGeometry.h"
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
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"
#include "../../utils/PreparedWeightTestHarness.h"
#include "../../utils/HCUpCancellationFixture.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPrimitiveStage.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
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
 if(s.device.is_cuda()){cuda::CUDAPackedWeights packed;require(cuda::packWeightsToCUDA(&source,packed),"Production CUDA native pack failed");authenticate(raw,packed.native_vnni,packed.native_scales,n,k);return std::make_shared<cuda::CUDAQuantisedGemmKernel>(n,k,s.device.ordinal,s.input(packed.native_vnni),s.input(packed.native_scales),nullptr,nullptr,packed.native_codebook_id,packed.native_blocks_per_row,s.owners,packed.native_source_identity);}
#endif
#ifdef HAVE_ROCM
 if(s.device.is_rocm()){rocm::ROCmPackedWeights packed;require(rocm::packNativeVNNI(&source,packed),"Production HIP native pack failed");authenticate(raw,packed.native_vnni_payload,packed.native_vnni_scales,n,k);return std::make_shared<rocm::ROCmQuantisedGemmKernel>(n,k,s.device.ordinal,s.input(packed.native_vnni_payload),s.input(packed.native_vnni_scales),nullptr,nullptr,packed.native_vnni_codebook_id,packed.native_vnni_blocks_per_row,s.owners,packed.native_source_identity);}
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
 for(auto geometry:std::array{std::array{1,65},std::array{15,127},std::array{512,64},std::array{1024,64},std::array{1,10240},std::array{15,10240},std::array{512,10240},std::array{1024,10240}}){const int m=geometry[0],n=geometry[1],k=320;SCOPED_TRACE(::testing::Message()<<device.toString()<<" M="<<m<<" N="<<n);Storage s{device,backend,ExplicitGPUStream(worker.defaultStream()).get()};auto raw=nativeWeights(n,k);Q8_0Tensor source({std::size_t(n),std::size_t(k)},raw);PreparedWeightStore store{ModelContextId{801944}};auto binding=registerEngine(store,source,device,prepare(source,raw,s,n,k));auto*kernel=store.gemmKernel(*binding.prepared);kernel->setGPUStream(s.stream);DeviceNativeVNNIMatrixDesc before;require(kernel->exportNativeVNNIMatrixDesc(before),"Prepared source export failed");
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
  DeviceNativeVNNIMatrixDesc after;require(kernel->exportNativeVNNIMatrixDesc(after),"Prepared source re-export failed");EXPECT_EQ(before.payload,after.payload);EXPECT_EQ(before.scales,after.scales);EXPECT_EQ(before.source_codebook_id,after.source_codebook_id);s.retire();stage.unbindWorkspace();
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
   DeviceNativeVNNIMatrixDesc before;require(kernel->exportNativeVNNIMatrixDesc(before),"GDN source export failed");
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
   DeviceNativeVNNIMatrixDesc after;require(kernel->exportNativeVNNIMatrixDesc(after),"GDN final source export failed");
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
            DeviceNativeVNNIMatrixDesc original;
            require(kernel->exportNativeVNNIMatrixDesc(original),"HC cancellation original native descriptor missing");
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
            DeviceNativeVNNIMatrixDesc after;
            require(kernel->exportNativeVNNIMatrixDesc(after),"HC cancellation final source descriptor missing");
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
