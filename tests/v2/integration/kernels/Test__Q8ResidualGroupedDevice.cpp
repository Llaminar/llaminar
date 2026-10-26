/**
 * @file Test__Q8ResidualGroupedDevice.cpp
 * @brief Captured prepared-engine/stage proof of backend-native two-term Q8 arithmetic.
 *
 * Production packers preserve the native Q8_0 payload/scale bits. Their exact
 * device pointers enter a normal PreparedWeightStore handle and GEMMStage.
 * One retained graph replays large/small/empty live prefixes without recapture.
 * Independent prepared serial calls are the byte oracle; actual SingleQ8
 * dispatches authenticate primary bytes/scales, including adversarial blocks.
 * Capture geometry follows the original source codebook and ordered short-K
 * precedence before the grouped row/matrix policy on both native backends.
 */
#include <gtest/gtest.h>
#include "kernels/common/DeviceFP32NumericalContract.h"
#include "kernels/common/Q8ResidualGroupedGeometry.h"
#include "kernels/common/Q8ResidualShortKGeometry.h"
#include "kernels/common/Q8ResidualCorrection.h"
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
#include <array>
#include <bit>
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
/** @brief Register exact source/engine identities through the production prepared-store boundary. */
WeightBinding registerEngine(PreparedWeightStore&store,Q8_0Tensor&source,DeviceId device,std::shared_ptr<ITensorGemm>engine){auto binding=test::makePreparedWeightTestBinding(&source,device,"hc.attn_up.weight",store.modelId());auto handle=std::make_shared<Factory::PreparedGemmHandle>();handle->tensor=&source;handle->device_id=device;handle->kind=device.is_cuda()?Factory::GemmPreparationKind::CUDA_INT8_PACKED:Factory::GemmPreparationKind::ROCM_INT8_PACKED;handle->variant=static_cast<int>(TensorType::Q8_0);handle->prepared_weights=std::make_shared<Factory::PreparedGemmWeights>();handle->prepared_weights->kind=handle->kind;handle->prepared_weights->owned_kernel=std::move(engine);handle->prepared_weights->kernel=handle->prepared_weights->owned_kernel.get();binding.prepared=store.registerPreparedGemmHandle(binding,test::preparedKindForDevice(device),device,std::move(handle));return binding;}
/** @brief Prove retained native execution and exact actual production primary quantization. */
void prove(DeviceId device){auto*backend=getBackendFor(device);require(backend&&backend->deviceCount()>device.ordinal,"Requested native correction backend unavailable");auto&worker=GPUDeviceContextPool::instance().getContext(device);auto context=IDeviceContext::create(device,1);worker.submitAsync([&]{
 for(auto geometry:std::array{std::array{1,65},std::array{15,127},std::array{31,65},std::array{32,65},std::array{33,65},std::array{63,65},std::array{64,65},std::array{65,65},std::array{127,65},std::array{128,65},std::array{129,65},std::array{511,65},std::array{512,65},std::array{513,65},std::array{1023,65},std::array{1024,65},std::array{1,10240},std::array{15,10240},std::array{512,10240},std::array{1024,10240}}){const int m=geometry[0],n=geometry[1],k=n==65&&m==32?32:(n==65&&m==64?64:(n==65&&m==65?352:320));SCOPED_TRACE(::testing::Message()<<device.toString()<<" M="<<m<<" N="<<n);Storage s{device,backend,ExplicitGPUStream(worker.defaultStream()).get()};auto raw=nativeWeights(n,k);Q8_0Tensor source({std::size_t(n),std::size_t(k)},raw);PreparedWeightStore store{ModelContextId{801944}};auto binding=registerEngine(store,source,device,prepare(source,raw,s,n,k));auto*kernel=store.gemmKernel(*binding.prepared);kernel->setGPUStream(s.stream);DeviceNativeVNNIMatrixDesc before;require(kernel->exportNativeVNNIMatrixDesc(before),"Prepared source export failed");
  auto input=std::make_shared<FP32Tensor>(std::vector<std::size_t>{std::size_t(m),std::size_t(k)},DeviceId::cpu());auto output=std::make_shared<FP32Tensor>(std::vector<std::size_t>{std::size_t(m),std::size_t(n)},DeviceId::cpu());FP32Tensor baseline({std::size_t(m),std::size_t(n)},DeviceId::cpu());
  for(int row=0;row<m;++row)for(int c=0;c<k;++c){float value=c%32==0?float((row%7)+1)*17.25f:std::sin(float(c*7+row*11))*.31f;if(row%17==0&&c<32)value=0;if(row%17==1&&c<32)value=std::numeric_limits<float>::denorm_min()*float(c+1);if(row%17==2&&c<32)value=c==0?127.f:float(c%11-5)+.5f;if(row%17==3&&c<32)value=c==0?1e9f:float(c-16)*1e6f;input->mutable_data()[std::size_t(row)*k+c]=value;}
  const std::vector<float>fixtureInput(input->data(),input->data()+input->numel());
  // One setup-owned M1 pair supplies the serial oracle. Exact-stream
  // diagnostic D2D commands select rows and retain all results before one
  // terminal plane observation, avoiding M tensor owners or per-row waits.
  FP32Tensor serialInput({1,std::size_t(k)},DeviceId::cpu()),serialOutput({1,std::size_t(n)},DeviceId::cpu());
  auto*count=s.allocate<int>(1);GEMMStage stage({.device_id=device,.A=input.get(),.B=&source,.C=output.get(),.m=m,.n=n,.k=k,.alpha=1,.beta=0,.transpose_B=false,.a_buffer_id=BufferId::NORMALIZED,.c_buffer_id=BufferId::GATE_PROJ,.force_decode_equivalent_verifier_prefill=true,.verifier_row_range=DeviceRowRange::deviceCounted(m,count),.prepared_ref=binding.prepared,.prepared_store=&store,.quantized_operand_precision=QuantizedActivationOperandPrecision::TwoTermQ8});stage.setGPUStream(s.stream);const auto requirements=stage.getWorkspaceRequirements(m,n,k);ASSERT_EQ(requirements.buffers.size(),4u);Workspace correction(device,requirements);stage.bindWorkspace(correction.value.get());auto*ordinaryWorkspaceConsumer=dynamic_cast<IWorkspaceConsumer*>(kernel);require(ordinaryWorkspaceConsumer,"Prepared native engine lacks workspace contract");auto ordinaryRequirements=ordinaryWorkspaceConsumer->getWorkspaceRequirements(m,n,k);
  const std::array projectionColumns{n};ordinaryWorkspaceConsumer->appendFusedProjectionWorkspaceRequirements(ordinaryRequirements,m,projectionColumns,k);Workspace ordinary(device,ordinaryRequirements);
  const std::vector<ITensorGemm::TensorProjectionDesc>ordinaryProjection{{kernel,&baseline,n,nullptr,"hc.up"}};
  CapturedWorkRetirement retirement(s);
  for(auto*t:std::array<TensorBase*,5>{input.get(),output.get(),&baseline,&serialInput,&serialOutput}) {
   TransferEngine::prepareDeviceInput(t,device,s.stream);TransferEngine::requireDeviceInput(t,device,s.stream);
  }
  s.upload(count,std::vector<int>{m});
  // One actual grouped crossover proves rejection before publisher/projection
  // kernels or destination writes. Prepared-engine validation may first join
  // the original input producer event.
  // All invalid calls share the same real source/workspace; a final observation
  // proves neither output nor any of the four native banks was modified.
  if(m==64&&n==65) {
   const std::vector<signed char>guardBytes(std::size_t(m)*k,90);
   const std::vector<float>guardScales(std::size_t(m)*k/32,-8765.f),guardOutput(std::size_t(m)*n,-12345.f);
   auto*p0=static_cast<signed char*>(correction.value->getBuffer(kGPUProjectionPrimaryQ8));
   auto*p1=static_cast<signed char*>(correction.value->getBuffer(kGPUProjectionResidualQ8));
   auto*d0=static_cast<float*>(correction.value->getBuffer(kGPUProjectionPrimaryQ8Scales));
   auto*d1=static_cast<float*>(correction.value->getBuffer(kGPUProjectionResidualQ8Scales));
   s.upload(p0,guardBytes);s.upload(p1,guardBytes);s.upload(d0,guardScales);s.upload(d1,guardScales);
   s.upload(static_cast<float*>(output->gpu_data_ptr()),guardOutput);
   const GPUQ8ResidualCorrectionArgs args{before,DeviceRowRange::deviceCounted(m,count),
    device.is_cuda()?Q8ResidualQuantizerPolicy::CUDAFloatScaleNativeReciprocal:Q8ResidualQuantizerPolicy::ROCmFiniteHalfNativeReciprocal,
    static_cast<const float*>(input->gpu_data_ptr()),static_cast<float*>(output->gpu_data_ptr()),p0,p1,d0,d1};
   const auto submit=[&](const GPUQ8ResidualCorrectionArgs&a,void*stream) {
#ifdef HAVE_CUDA
    if(device.is_cuda()){cudaQ8ResidualCorrectedProjection(a,stream);return;}
#endif
#ifdef HAVE_ROCM
    if(device.is_rocm()){rocmQ8ResidualCorrectedProjection(a,stream);return;}
#endif
    throw std::logic_error("Unavailable correction fixture backend");
   };
   for(const auto stream:std::array<void*,3>{nullptr,reinterpret_cast<void*>(std::uintptr_t{1}),reinterpret_cast<void*>(std::uintptr_t{2})})
    EXPECT_THROW(submit(args,stream),std::invalid_argument);
   auto bad=args;bad.weights.k=k-1;EXPECT_THROW(submit(bad,s.stream),std::invalid_argument);
   bad=args;bad.weights.n=0;EXPECT_THROW(submit(bad,s.stream),std::invalid_argument);
   bad=args;bad.residual=bad.primary;EXPECT_THROW(submit(bad,s.stream),std::invalid_argument);
   bad=args;bad.weights.source_identity_present=false;EXPECT_THROW(submit(bad,s.stream),std::invalid_argument);
   bad=args;bad.quantizer=Q8ResidualQuantizerPolicy::CPUHalfAfterQuantization;EXPECT_THROW(submit(bad,s.stream),std::invalid_argument);
   EXPECT_THROW(kernel->multiply_q8_residual_corrected(input.get(),output.get(),DeviceRowRange::fullyActive(m+1),n,k,correction.value.get()),std::invalid_argument);
   auto shortRequirements=requirements;--shortRequirements.buffers.front().size_bytes;
   Workspace shortWorkspace(device,shortRequirements);
   EXPECT_THROW(kernel->multiply_q8_residual_corrected(input.get(),output.get(),DeviceRowRange::fullyActive(m),n,k,shortWorkspace.value.get()),std::logic_error);
   EXPECT_EQ(s.download(static_cast<const float*>(output->gpu_data_ptr()),guardOutput.size()),guardOutput);
   EXPECT_EQ(s.download(p0,guardBytes.size()),guardBytes);EXPECT_EQ(s.download(p1,guardBytes.size()),guardBytes);
   EXPECT_EQ(s.download(d0,guardScales.size()),guardScales);EXPECT_EQ(s.download(d1,guardScales.size()),guardScales);
  }
  // Authenticate the exact production entrypoint used by each HC-up phase.
  // Grouped M15 must exercise the verifier API, not generic prefill by accident.
  if(m==15){auto scope=kernel->beginVerifierDecodeEquivalentScope(DeviceRowRange::fullyActive(m));ASSERT_TRUE(kernel->multiply_fused_verifier_rows_decode_equivalent(input.get(),ordinaryProjection,m,k,nullptr,ordinary.value.get()));}
  else ASSERT_TRUE(kernel->multiply_tensor(input.get(),&baseline,m,n,k,false,1,0,nullptr,nullptr,device.ordinal,ordinary.value.get()));auto ordinaryBytes=s.download(static_cast<const signed char*>(ordinary.value->getBuffer(kGPUProjectionPrimaryQ8)),std::size_t(m)*k);auto ordinaryScales=s.download(static_cast<const float*>(ordinary.value->getBuffer(kGPUProjectionPrimaryQ8Scales)),std::size_t(m)*k/32);
  // Retain each actual serial M1 primary before the next serial call reuses
  // ordinary workspace. All diagnostic copies use the exact producer stream;
  // one terminal observation compares every primary byte/scale in every row.
  auto*serialResult=s.allocate<float>(std::size_t(m)*n);
  auto*serialPrimary=s.allocate<signed char>(std::size_t(m)*k);auto*serialScales=s.allocate<float>(std::size_t(m)*k/32);
  for(int row=0;row<m;++row){
   require(backend->deviceToDevice(serialInput.gpu_data_ptr(),static_cast<const float*>(input->gpu_data_ptr())+std::size_t(row)*k,std::size_t(k)*sizeof(float),device.ordinal,s.stream),"Serial input row selection failed");
   ASSERT_TRUE(kernel->multiply_tensor(&serialInput,&serialOutput,1,n,k,false,1,0,nullptr,nullptr,device.ordinal,ordinary.value.get()));
   require(backend->deviceToDevice(serialPrimary+std::size_t(row)*k,ordinary.value->getBuffer(kGPUProjectionPrimaryQ8),k,device.ordinal,s.stream),"Serial primary native copy failed");
   require(backend->deviceToDevice(serialScales+std::size_t(row)*k/32,ordinary.value->getBuffer(kGPUProjectionPrimaryQ8Scales),std::size_t(k)/32*4,device.ordinal,s.stream),"Serial scale native copy failed");}
  const auto serialPrimaryBytes=s.download(serialPrimary,std::size_t(m)*k);const auto serialPrimaryScales=s.download(serialScales,std::size_t(m)*k/32);
  ASSERT_EQ(std::memcmp(ordinaryBytes.data(),serialPrimaryBytes.data(),ordinaryBytes.size()),0);
  ASSERT_EQ(std::memcmp(ordinaryScales.data(),serialPrimaryScales.data(),ordinaryScales.size()*4),0);
  // Build the immutable serial oracle once; replay only observes its valid prefix.
  auto*serialResidual=s.allocate<signed char>(std::size_t(m)*k);auto*serialResidualScales=s.allocate<float>(std::size_t(m)*k/32);
  for(int row=0;row<m;++row){
   require(backend->deviceToDevice(serialInput.gpu_data_ptr(),static_cast<const float*>(input->gpu_data_ptr())+std::size_t(row)*k,std::size_t(k)*sizeof(float),device.ordinal,s.stream),"Corrected serial input row selection failed");
   kernel->multiply_q8_residual_corrected(&serialInput,&serialOutput,DeviceRowRange::fullyActive(1),n,k,correction.value.get());
   require(backend->deviceToDevice(serialResult+std::size_t(row)*n,serialOutput.gpu_data_ptr(),std::size_t(n)*4,device.ordinal,s.stream),"Independent serial output gather failed");
   require(backend->deviceToDevice(serialResidual+std::size_t(row)*k,correction.value->getBuffer(kGPUProjectionResidualQ8),k,device.ordinal,s.stream),"Serial residual byte retention failed");
   require(backend->deviceToDevice(serialResidualScales+std::size_t(row)*k/32,correction.value->getBuffer(kGPUProjectionResidualQ8Scales),std::size_t(k)/32*4,device.ordinal,s.stream),"Serial residual scale retention failed");
  }
  const auto serialResidualBytes=s.download(serialResidual,std::size_t(m)*k);
  const auto serialResidualScaleBytes=s.download(serialResidualScales,std::size_t(m)*k/32);
  bool checkedPrecision=false;const auto ordinaryOutput=s.download(static_cast<const float*>(baseline.gpu_data_ptr()),std::size_t(m)*n);
  s.graph=worker.createGraphCapture(s.stream);{ScopedBackendGraphCapture capture(worker,*s.graph,"Prepared two-term Q8 HC-up");require(capture.begin(),"Correction stage capture failed");require(stage.execute(context.get()),"Correction stage submission failed");capture.finish();}require(s.graph->instantiate(),"Correction stage graph instantiation failed");std::vector<GPUGraphKernelNodeInfo>nodes;std::string error;require(s.graph->inspectKernelNodes(nodes,&error),"Correction graph inspection failed");ASSERT_EQ(nodes.size(),2u);
  const auto capturedBackend=device.is_cuda()?Q8ResidualGroupedBackend::CUDA:Q8ResidualGroupedBackend::ROCm;
  const auto capturedGeometry=q8ResidualGroupedGeometry(capturedBackend,m,n,before.codebook_id);
  const auto shortGeometry=q8ResidualShortKGeometry(capturedBackend,m,n,k,before.codebook_id);
  unsigned capturedProjectionCount=0;
  for(const auto&node:nodes) {
   EXPECT_EQ(node.local_memory_bytes_per_thread,0u);
   if(node.name.find("project")==std::string::npos)continue;
   ++capturedProjectionCount;EXPECT_TRUE(node.name_resolved);
   EXPECT_EQ(node.block_x,static_cast<unsigned>(shortGeometry?shortGeometry->threads:capturedGeometry.threads));
   EXPECT_EQ(node.grid_x,shortGeometry?shortGeometry->blocks:capturedGeometry.blocks);EXPECT_EQ(node.grid_y,1u);EXPECT_EQ(node.grid_z,1u);
   EXPECT_EQ(node.name.find("projectShortK")!=std::string::npos,shortGeometry.has_value());
   EXPECT_EQ(node.name.find("projectGrouped")!=std::string::npos,!shortGeometry&&capturedGeometry.strategy==Q8ResidualGroupedStrategy::CooperativeMatrix);
   if(shortGeometry){EXPECT_EQ(node.static_shared_memory_bytes,shortGeometry->shared_bytes);EXPECT_EQ(node.dynamic_shared_memory_bytes,0u);}
  }
  ASSERT_EQ(capturedProjectionCount,1u);
  const auto signature=stage.graphCaptureVariantSignature(),ownerCount=s.owners->size(),workspaceUsed=correction.value->used();
  const auto primaryBank=static_cast<signed char*>(correction.value->getBuffer(kGPUProjectionPrimaryQ8));
  const auto residualBank=static_cast<signed char*>(correction.value->getBuffer(kGPUProjectionResidualQ8));
  const auto primaryScaleBank=static_cast<float*>(correction.value->getBuffer(kGPUProjectionPrimaryQ8Scales));
  const auto residualScaleBank=static_cast<float*>(correction.value->getBuffer(kGPUProjectionResidualQ8Scales));
  const std::vector<signed char>dirtyBytes(std::size_t(m)*k,90);
  const std::vector<float>dirtyScales(std::size_t(m)*k/32,-8765.f);
  std::vector<int> replayRows{m,std::min(2,m),0,m};if(m==1024&&n==65)for(int live=1;live<=m;++live)replayRows.push_back(live);
  for(int live:replayRows){
   auto poisoned=fixtureInput;std::fill(poisoned.begin()+std::size_t(live)*k,poisoned.end(),std::numeric_limits<float>::quiet_NaN());
   s.upload(static_cast<float*>(input->gpu_data_ptr()),poisoned);
   s.upload(primaryBank,dirtyBytes);s.upload(residualBank,dirtyBytes);s.upload(primaryScaleBank,dirtyScales);s.upload(residualScaleBank,dirtyScales);
   s.upload(count,std::vector<int>{live});s.upload(static_cast<float*>(output->gpu_data_ptr()),std::vector<float>(std::size_t(m)*n,-12345.f));
   require(s.graph->launch(),"Retained correction graph replay failed");
   auto actual=s.download(static_cast<const float*>(output->gpu_data_ptr()),std::size_t(m)*n);
   const auto primary=s.download(primaryBank,std::size_t(m)*k),q1=s.download(residualBank,std::size_t(m)*k);
   const auto scales=s.download(primaryScaleBank,std::size_t(m)*k/32),d1=s.download(residualScaleBank,std::size_t(m)*k/32);
   if(live){
    EXPECT_EQ(std::memcmp(primary.data(),ordinaryBytes.data(),std::size_t(live)*k),0);
    EXPECT_EQ(std::memcmp(scales.data(),ordinaryScales.data(),std::size_t(live)*k/32*sizeof(float)),0);
    EXPECT_EQ(std::memcmp(q1.data(),serialResidualBytes.data(),std::size_t(live)*k),0);
    EXPECT_EQ(std::memcmp(d1.data(),serialResidualScaleBytes.data(),std::size_t(live)*k/32*sizeof(float)),0);
   }
   EXPECT_TRUE(std::all_of(primary.begin()+std::size_t(live)*k,primary.end(),[](signed char x){return x==90;}));
   EXPECT_TRUE(std::all_of(q1.begin()+std::size_t(live)*k,q1.end(),[](signed char x){return x==90;}));
   EXPECT_TRUE(std::all_of(scales.begin()+std::size_t(live)*k/32,scales.end(),[](float x){return x==-8765.f;}));
   EXPECT_TRUE(std::all_of(d1.begin()+std::size_t(live)*k/32,d1.end(),[](float x){return x==-8765.f;}));
   EXPECT_EQ(stage.graphCaptureVariantSignature(),signature);EXPECT_EQ(s.owners->size(),ownerCount);EXPECT_EQ(correction.value->used(),workspaceUsed);
   EXPECT_EQ(correction.value->getBuffer(kGPUProjectionPrimaryQ8),primaryBank);EXPECT_EQ(correction.value->getBuffer(kGPUProjectionResidualQ8),residualBank);
   EXPECT_EQ(correction.value->getBuffer(kGPUProjectionPrimaryQ8Scales),primaryScaleBank);EXPECT_EQ(correction.value->getBuffer(kGPUProjectionResidualQ8Scales),residualScaleBank);
   // An independent scalar INT8 dot authenticates the device fragment mapping,
   // two ordered FP32 folds, and improvement over the actual SingleQ8 output.
   // Sampling columns bounds diagnostic CPU cost; serial comparison covers all.
   if(live==m&&!checkedPrecision) {
    double singleError=0,correctedError=0;
    for(int row=0;row<m;++row)for(int column:std::array{0,n/2,n-1}) {
     float sum0=0,sum1=0;double ideal=0;
     for(int group=0;group<k/32;++group) {
      const auto activation=std::size_t(row)*(k/32)+group;
      const auto native=(std::size_t(column)*(k/32)+group)*34;
      std::uint16_t weightBits;std::memcpy(&weightBits,raw.data()+native,2);
      const float weightScale=fp16_to_fp32(weightBits);int dot0=0,dot1=0;
      for(int element=0;element<32;++element) {
       const int weight=static_cast<signed char>(raw[native+2+element]);
       dot0+=int(primary[activation*32+element])*weight;
       dot1+=int(q1[activation*32+element])*weight;
       ideal+=double(fixtureInput[std::size_t(row)*k+group*32+element])*double(weightScale)*weight;
      }
      const float c0=device_fp32_contract::multiply(device_fp32_contract::multiply(scales[activation],weightScale),float(dot0));
      const float c1=device_fp32_contract::multiply(device_fp32_contract::multiply(d1[activation],weightScale),float(dot1));
      sum0=device_fp32_contract::add(sum0,c0);sum1=device_fp32_contract::add(sum1,c1);
     }
     const float oracle=device_fp32_contract::add(sum0,sum1);
     EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[std::size_t(row)*n+column]),std::bit_cast<std::uint32_t>(oracle));
     const double e0=double(ordinaryOutput[std::size_t(row)*n+column])-ideal;
     const double e1=double(actual[std::size_t(row)*n+column])-ideal;
     singleError+=e0*e0;correctedError+=e1*e1;
    }
    EXPECT_LT(correctedError,singleError);checkedPrecision=true;
   }
   auto expected=s.download(serialResult,std::size_t(live)*n);if(live){EXPECT_EQ(std::memcmp(actual.data(),expected.data(),expected.size()*4),0);}for(std::size_t i=std::size_t(live)*n;i<actual.size();++i)ASSERT_EQ(actual[i],-12345.f);
  }
  DeviceNativeVNNIMatrixDesc after;require(kernel->exportNativeVNNIMatrixDesc(after),"Prepared source re-export failed");EXPECT_EQ(before.payload,after.payload);EXPECT_EQ(before.scales,after.scales);EXPECT_EQ(before.source_codebook_id,after.source_codebook_id);s.retire();stage.unbindWorkspace();
 }
 }).get();}
}
#ifdef HAVE_CUDA
TEST(Q8ResidualGroupedDevice, CUDA){prove(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
TEST(Q8ResidualGroupedDevice, ROCm){prove(DeviceId::rocm(0));}
#endif
