/**
 * @file Test__ROCmWoVerifierTransaction.cpp
 * @brief Actual attention-output GEMM replay and complete-operation candidate timing.
 *
 * Original Wo dimensions come from each model weight, including TP-sized local K.
 * TP-sized controls use local synthetic matrices: they do not certify TP source
 * slice loading, global source offsets or the later collective.
 * Three controls and an independent M1 oracle own separate prepared bindings,
 * workspaces and outputs. Quantization and ordered reduction stay inside capture.
 * The preceding attention output gate and subsequent TP collective are outside
 * this explicit projection boundary. Only ordinary SingleQ8 is qualified here.
 */
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#include "backends/GPUDeviceContextPool.h"
#include "backends/rocm/HIPGraphCapture.h"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "execution/local_execution/coherence/GpuCoherence.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/rocm/gemm/ROCmNativeQuantisedVerifierLaunch.h"
#include "kernels/rocm/gemm/ROCmNativeQuantisedArchitecture.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "transfer/TransferEngine.h"
#include "../../../utils/GpuPreparedGemmHarness.h"
#include "../../../utils/QuantizedVerifierFormats.h"
#include "../../../utils/ROCmWoVerifierExpectations.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::rocm;
using namespace llaminar2::test;
namespace {
/** @brief Collect one retained native transaction outside setup and observations. */
class ProfileWindow final {
    bool active_=false;
public:
    /** @brief Resume only after the caller has drained setup and poisoned outputs. */
    ProfileWindow() {
        if(roctxRangePushA("ROCmWoVerifierTransaction")<0)throw std::runtime_error("Wo profiler range failed");
        if(roctxProfilerResume(0)!=0) {(void)roctxRangePop();throw std::runtime_error("Wo profiler resume failed");}
        active_=true;
    }
    /** @brief Pause before output observation; cleanup remains unconditional on exceptions. */
    void finish() {
        if(!active_)return;
        const int paused=roctxProfilerPause(0),popped=roctxRangePop();active_=false;
        if(paused!=0 || popped<0)throw std::runtime_error("Wo profiler pause failed");
    }
    /** @brief Never let a failed replay leave profiling active for another kernel. */
    ~ProfileWindow() {
        if(active_) {
            const int paused=roctxProfilerPause(0),popped=roctxRangePop();
            if(paused!=0 || popped<0)std::terminate();
        }
    }
    ProfileWindow(const ProfileWindow&)=delete;
    ProfileWindow &operator=(const ProfileWindow&)=delete;
};
/** @brief Replay, paired events or one separately profiled actual-Automatic transaction. */
enum class Observation {Replay,Paired,Profile};
/** @brief Original hosted extent or an explicitly unqualified retained capacity. */
enum class Extent {ExactHosted,Guarded};
/** @brief Fail the component before issuing more work with an invalid owner. */
void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
/** @brief Join exact-stream work before any fixture-owned storage retires. */
struct Drain {
    hipStream_t stream;
    /** @brief Assertion unwinding must retain asynchronous input/output owners. */
    ~Drain(){if(hipStreamSynchronize(stream)!=hipSuccess)std::terminate();}
};
/** @brief Suppress a captured producer family while preserving the original graph. */
class Omission {
    hipGraphExec_t graph_;hipStream_t stream_;const std::vector<hipGraphNode_t> &nodes_;bool active_=false;
public:
    /** @param graph Retained executable. @param stream Exact stream. @param nodes Original producers. */
    Omission(hipGraphExec_t graph,hipStream_t stream,const std::vector<hipGraphNode_t> &nodes):graph_(graph),stream_(stream),nodes_(nodes){}
    /** @brief Disable every repeated instance of this producer before dirty replay. */
    void omit(){require(!active_ && !nodes_.empty(),"Invalid Wo omission");active_=true;for(auto node:nodes_)require(hipGraphNodeSetEnabled(graph_,node,0)==hipSuccess,"Wo omission failed");}
    /** @brief Restore every original producer after its work retires. */
    void restore() noexcept {if(active_){if(hipStreamSynchronize(stream_)!=hipSuccess)std::terminate();for(auto node:nodes_)if(hipGraphNodeSetEnabled(graph_,node,1)!=hipSuccess)std::terminate();active_=false;}}
    /** @brief Failed assertions cannot leak an omitted producer. */
    ~Omission(){restore();}
    Omission(const Omission &)=delete;Omission &operator=(const Omission &)=delete;
};
/** @brief Original full/TP-local Wo projection; K is attention width, not gated Q width. */
struct Geometry {const char *name;int n,k;};
constexpr std::array<Geometry,6> geometries{{
    {"Dense",5120,6144},{"DenseTP2",5120,3072},{"DenseTP4",5120,1536},
    {"MoE",2048,4096},{"MoETP2",2048,2048},{"MoETP4",2048,1024}}};
/** @brief One source-format and geometry combination, independent of the runtime selector. */
struct Case {QuantizedVerifierFormatCase source;Geometry geometry;};
/** @brief Own a real prepared projection and its persistent arena until all graphs retire. */
struct StageOwner {
    std::unique_ptr<DeviceWorkspaceManager> workspace;
    std::unique_ptr<GEMMStage> stage;
    WorkspaceRequirements requirements;
    hipStream_t stream;
    /** @param p Original stage configuration. @param context Native device authority.
     * @param exact Preparation, capture and retirement stream. */
    StageOwner(GEMMStage::Params p,IDeviceContext &context,hipStream_t exact):stream(exact){
        const auto device=p.device_id;const int m=p.m,n=p.n,k=p.k;
        stage=std::make_unique<GEMMStage>(std::move(p));requirements=stage->getWorkspaceRequirements(m,n,k);
        try {
            workspace=std::make_unique<DeviceWorkspaceManager>(device,requirements.total_bytes_with_alignment());
            require(workspace->allocate(requirements),"Wo workspace allocation failed");
            stage->bindWorkspace(workspace.get());stage->setGPUStream(stream);
            require(stage->prepareGraphLaunch(&context,stream),"Wo preparation failed");
        }catch(...){if(hipStreamSynchronize(stream)!=hipSuccess)std::terminate();stage->unbindWorkspace();throw;}
    }
    /** @brief Destroy graph owners first, then release borrowed stage workspace. */
    ~StageOwner(){if(hipStreamSynchronize(stream)!=hipSuccess)std::terminate();stage->unbindWorkspace();}
};
/** @brief Replay each physical and live extent through the real projection.
 * @param spec Original source and model geometry. @param observation Replay,31 triples or a profiled M16.
 * @param extent Whether recording retains the original exact-hosted request extent. */
void prove(const Case &spec,Observation observation,Extent extent=Extent::ExactHosted){
    int count=0;if(hipGetDeviceCount(&count)!=hipSuccess || !count)GTEST_SKIP()<<"No ROCm device";
    initializeROCmHardwareMetadata();int ordinal=-1;
    for(int device=0;device<count;++device)if(nativeQuantisedHardwareForDevice(device).architecture==native_quantised::NativeQuantisedArchitecture::rocm(11,0,0)){ordinal=device;break;}
    if(ordinal<0)GTEST_SKIP()<<"No native gfx1100 participant";
    const auto device=DeviceId::rocm(ordinal);auto worker=GPUDeviceContextPool::instance().retainAMDContext(ordinal);
    worker->submitAndWait([&]{
        const auto stream=static_cast<hipStream_t>(worker->defaultStream());auto context=IDeviceContext::create(device);ASSERT_TRUE(context);
        const bool timed=observation==Observation::Paired,profiled=observation==Observation::Profile,hosted=extent==Extent::ExactHosted;
        ASSERT_TRUE(observation==Observation::Replay || timed || profiled);
        if(profiled)ASSERT_TRUE(hosted);
        const auto &[label,n,k]=spec.geometry;const auto &format=spec.source;const int repeats=timed?8:1;
        constexpr std::array candidates{NativeQuantisedVerifierCandidate::Dot4,NativeQuantisedVerifierCandidate::Automatic,NativeQuantisedVerifierCandidate::WmmaM16N64};
        constexpr std::array names{"dot4","automatic","wmma"};
        std::printf("ROCM_WO_VERIFIER_MODE,%s,%s,%d,%d,%d\n",format.label,label,n,k,int(hosted));
        std::shared_ptr<TensorBase> weight;std::array<GpuPreparedGemm,4> prepared;
        std::array<DeviceNativeQuantisedMatrixDesc,4> descriptors;
        std::unique_ptr<FP32Tensor> input,serial_input,serial_output;std::vector<uint32_t> oracle(size_t(16)*n);
        const Drain setup_drain{stream};weight=format.create({size_t(n),size_t(k)},89501);
        for(int owner=0;owner<4;++owner){
            prepared[owner]=makeGpuPreparedGemm(weight.get(),device,"test.wo_verifier",ModelContextId{uint64_t(89501+owner)});
            auto *kernel=dynamic_cast<ROCmQuantisedGemmKernel *>(prepared[owner].kernel);ASSERT_TRUE(kernel);
            ASSERT_EQ(prepared[owner].store->binding(prepared[owner].ref)->tensor,weight.get());
            ASSERT_TRUE(kernel->exportNativeQuantisedMatrixDesc(descriptors[owner]));const auto &desc=descriptors[owner];
            ASSERT_TRUE(desc.source_identity_present);ASSERT_EQ(desc.source_codebook_id,format.source_codebook_id);
            ASSERT_EQ(bool(desc.source_is_superblock),format.source_is_superblock);ASSERT_EQ(desc.codebook_id,format.device_execution_codebook_id);
            ASSERT_EQ(desc.n,n);ASSERT_EQ(desc.k,k);ASSERT_EQ(desc.blocks_per_row,k/32u);
            for(int prior=0;prior<owner;++prior){ASSERT_NE(desc.payload,descriptors[prior].payload);ASSERT_NE(prepared[owner].kernel,prepared[prior].kernel);}
            std::printf("ROCM_WO_VERIFIER_SOURCE,%s,%s,%d,%u,%u,%u,%d,%d\n",format.label,label,owner,unsigned(desc.source_codebook_id),unsigned(desc.source_is_superblock),unsigned(desc.codebook_id),n,k);
        }
        input=TestTensorFactory::createFP32({16,size_t(k)});
        for(size_t i=0;i<input->numel();++i)input->mutable_data()[i]=float(int((i*29+i/k*11)%251)-125)/128.f;
        ASSERT_TRUE(input->ensureOnDevice(device,stream));serial_input=TestTensorFactory::createFP32({1,size_t(k)});ASSERT_TRUE(serial_input->allocateOnDevice(device,stream));
        serial_output=TestTensorFactory::createFP32({1,size_t(n)});ASSERT_TRUE(serial_output->allocateOnDevice(device,stream));
        const auto params=[&](int owner,int m,FP32Tensor *a,FP32Tensor *out,std::optional<DeviceRowRange> rows){
            GEMMStage::Params p;p.device_id=device;p.A=a;p.B=weight.get();p.C=out;p.m=m;p.n=n;p.k=k;
            p.alpha=1;p.beta=0;p.transpose_B=false;p.gemm_context=GemmContext::ATTN;
            p.prepared_ref=prepared[owner].ref;p.prepared_store=prepared[owner].store.get();
            p.force_decode_equivalent_verifier_prefill=m>1;p.verifier_row_range=rows;
            return p;
        };
        const auto execute=[&](StageOwner &owner,FP32Tensor *a,FP32Tensor *out){return with_gpu_coherence(device,{a},{out},stream,[&]{return owner.stage->execute(context.get());});};
        {
            StageOwner serial(params(3,1,serial_input.get(),serial_output.get(),std::nullopt),*context,stream);const Drain drain{stream};
            for(int row=0;row<16;++row){
                ASSERT_EQ(hipMemcpyAsync(serial_input->gpu_data_ptr(),static_cast<const float *>(input->gpu_data_ptr())+row*k,k*sizeof(float),hipMemcpyDeviceToDevice,stream),hipSuccess);
                TransferEngine::publishCurrentDeviceWrite(serial_input.get(),stream);ASSERT_TRUE(execute(serial,serial_input.get(),serial_output.get()));
                ASSERT_EQ(hipMemcpyAsync(oracle.data()+size_t(row)*n,serial_output->gpu_data_ptr(),n*sizeof(float),hipMemcpyDeviceToHost,stream),hipSuccess);
            }
            ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);
        }
        ASSERT_TRUE(std::all_of(oracle.begin(),oracle.end(),[](auto word){return std::isfinite(std::bit_cast<float>(word));}));
        ASSERT_TRUE(std::any_of(oracle.begin(),oracle.end(),[](auto word){return word!=0;}));
        for(int m=profiled?16:2;m<=16;++m){
            SCOPED_TRACE(::testing::Message()<<format.label<<" "<<label<<" M="<<m);
            auto live=TestTensorFactory::createINT32({1});live->mutable_int32_data()[0]=m;
            std::array<std::unique_ptr<FP32Tensor>,3> outputs;std::vector<uint32_t> actual(size_t(m)*n);
            std::array<std::unique_ptr<StageOwner>,3> stages;
            std::array<std::unique_ptr<IGPUGraphCapture>,3> fragments,graphs;
            std::array<std::array<std::vector<hipGraphNode_t>,2>,3> families;
            int32_t published=m;const Drain drain{stream};ASSERT_TRUE(live->ensureOnDevice(device,stream));
            auto *count_owner=static_cast<int32_t *>(live->gpu_data_ptr());const auto rows=DeviceRowRange::deviceCounted(m,count_owner);
            for(int path=0;path<3;++path){
                outputs[path]=TestTensorFactory::createFP32({size_t(m),size_t(n)});ASSERT_TRUE(outputs[path]->allocateOnDevice(device,stream));
                stages[path]=std::make_unique<StageOwner>(params(path,m,input.get(),outputs[path].get(),rows),*context,stream);auto &owner=*stages[path];
                ScopedCapturedVerifierExtent extent(hosted?CapturedVerifierExtent::exactHostedRequest(device,rows):CapturedVerifierExtent{});
                ScopedNativeQuantisedVerifierCandidate select(candidates[path],ordinal);
                ASSERT_TRUE(execute(owner,input.get(),outputs[path].get()));ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);
                TransferEngine::requireDeviceInput(input.get(),device,stream);TransferEngine::requireDeviceInput(live.get(),device,stream);
                std::vector<GraphCaptureDependencyLedger::StagePlan> plans;
                for(int repeat=0;repeat<repeats;++repeat)plans.push_back({.stage_identity=owner.stage.get(),.stage_name="attention_output_projection",.external_inputs={input.get(),live.get()},.outputs={outputs[path].get()}});
                GraphCaptureDependencyLedger ledger(device,stream,std::move(plans),"Attention-output verifier projection");
                graphs[path]=worker->createGraphCapture(stream);fragments[path]=graphs[path]->createOrderedTimelineFragment();ASSERT_TRUE(fragments[path]);
                ScopedBackendGraphCapture capture(*worker,*fragments[path],"original Wo projection",&ledger);ASSERT_TRUE(capture.begin());
                for(int repeat=0;repeat<repeats;++repeat){ScopedGraphCaptureStage stage(owner.stage.get());ASSERT_TRUE(owner.stage->execute(context.get()));stage.complete();}
                capture.finish();const std::array<GPUOrderedTimelineStep,1> steps{{{.name=names[path],.kind=GPUOrderedTimelineStepKind::CapturedFragment,.capture=fragments[path].get()}}};
                ASSERT_TRUE(graphs[path]->buildOrderedTimelineTransaction(steps,profiled?GPUOrderedTimelineInstrumentation::Disabled:GPUOrderedTimelineInstrumentation::PerStepEvents));ASSERT_TRUE(graphs[path]->instantiate());
                auto *parent=dynamic_cast<HIPGraphCapture *>(graphs[path].get());ASSERT_TRUE(parent);
                size_t size=0;ASSERT_EQ(hipGraphGetNodes(parent->graph(),nullptr,&size),hipSuccess);std::vector<hipGraphNode_t> nodes(size);ASSERT_EQ(hipGraphGetNodes(parent->graph(),nodes.data(),&size),hipSuccess);
                const auto quant=owner.workspace->getBuffer(GemmWorkspaceBuffers::QUANT_A),scales=owner.workspace->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE),sums=owner.workspace->getBuffer(GemmWorkspaceBuffers::SUMS_A_BLOCKWISE);
                using Partial=std::pair<uintptr_t,int>;std::vector<Partial> produced,reduced;
                for(auto node:nodes){
                    hipGraphNodeType type;ASSERT_EQ(hipGraphNodeGetType(node,&type),hipSuccess);if(type!=hipGraphNodeTypeKernel)continue;
                    hipKernelNodeParams launch{};ASSERT_EQ(hipGraphKernelNodeGetParams(node,&launch),hipSuccess);
                    const char *symbol=hipKernelNameRefByPtr(launch.func,stream);ASSERT_TRUE(symbol);const std::string_view name(symbol);
                    if(name.find("quantizeActivationsQ8_blockwise_v4_with_sums_kernel")!=std::string_view::npos){
                        families[path][0].push_back(node);ASSERT_NE(name.find("ILb1E"),std::string_view::npos);
                        ASSERT_EQ(*static_cast<const float **>(launch.kernelParams[0]),input->gpu_data_ptr());ASSERT_EQ(*static_cast<int8_t **>(launch.kernelParams[1]),quant);
                        ASSERT_EQ(*static_cast<float **>(launch.kernelParams[2]),scales);ASSERT_EQ(*static_cast<int32_t **>(launch.kernelParams[3]),sums);
                        ASSERT_EQ(*static_cast<int *>(launch.kernelParams[4]),m);ASSERT_EQ(*static_cast<int *>(launch.kernelParams[5]),k);ASSERT_EQ(*static_cast<int *>(launch.kernelParams[6]),k/32);
                        ASSERT_EQ(static_cast<const DeviceRowRange *>(launch.kernelParams[7])->captureIdentity(),rows.captureIdentity());
                        ASSERT_EQ(launch.gridDim.x,unsigned(m*(k/32)));ASSERT_EQ(launch.gridDim.y,1u);ASSERT_EQ(launch.gridDim.z,1u);ASSERT_EQ(launch.blockDim.x,32u);
                    }
                    if(name.find("gemv_native_quantised_reduce_small_m_kernel_t")!=std::string_view::npos){
                        ASSERT_EQ(*static_cast<float **>(launch.kernelParams[1]),outputs[path]->gpu_data_ptr());ASSERT_EQ(*static_cast<int *>(launch.kernelParams[2]),m);ASSERT_EQ(*static_cast<int *>(launch.kernelParams[3]),n);
                        const int partitions=*static_cast<int *>(launch.kernelParams[4]);ASSERT_GT(partitions,1);
                        ASSERT_EQ(static_cast<const DeviceRowRange *>(launch.kernelParams[5])->captureIdentity(),rows.captureIdentity());
                        ASSERT_EQ(static_cast<const ProjectionRowWork *>(launch.kernelParams[6])->captureIdentity(),ProjectionRowWork::fromPrefix(rows).captureIdentity());
                        reduced.emplace_back(reinterpret_cast<uintptr_t>(*static_cast<const float **>(launch.kernelParams[0])),partitions);
                    }
                    if(name.find("native_quantised_verifier_wmma_m16n64")==std::string_view::npos)continue;
                    families[path][1].push_back(node);const auto &packet=*static_cast<const NativeQuantisedSmallMBatchedArgs *>(launch.kernelParams[3]);const auto &desc=descriptors[path];
                    ASSERT_EQ(packet.num_projections,1);ASSERT_EQ(packet.N[0],n);ASSERT_EQ(packet.payload[0],desc.payload);ASSERT_EQ(packet.scales[0],desc.scales);ASSERT_EQ(packet.mins[0],desc.mins);ASSERT_EQ(packet.emins[0],desc.emins);ASSERT_EQ(packet.codebook[0],desc.codebook_id);
                    ASSERT_EQ(packet.output[0],outputs[path]->gpu_data_ptr());ASSERT_EQ(packet.bias[0],nullptr);ASSERT_EQ(packet.rows.captureIdentity(),rows.captureIdentity());ASSERT_EQ(packet.work.captureIdentity(),ProjectionRowWork::fromPrefix(rows).captureIdentity());
                    ASSERT_EQ(*static_cast<const int8_t **>(launch.kernelParams[0]),quant);ASSERT_EQ(*static_cast<const float **>(launch.kernelParams[1]),scales);ASSERT_EQ(*static_cast<const int32_t **>(launch.kernelParams[2]),sums);
                    ASSERT_EQ(*static_cast<int *>(launch.kernelParams[4]),k);ASSERT_EQ(launch.gridDim.x,unsigned(packet.total_grid_n));ASSERT_EQ(launch.gridDim.x,unsigned((n+63)/64));ASSERT_GT(launch.gridDim.y,0u);ASSERT_EQ(launch.gridDim.z,1u);ASSERT_EQ(launch.blockDim.x,128u);
                    const int span=*static_cast<int *>(launch.kernelParams[5]);ASSERT_EQ(span,(k/32+int(launch.gridDim.y)-1)/int(launch.gridDim.y));
                    const bool partial=name.find("VerifierWmmaPublicationE2E")!=std::string_view::npos;
                    if(partial){
                        const auto begin=reinterpret_cast<uintptr_t>(packet.partial[0]),end=begin+size_t(launch.gridDim.y)*m*n*sizeof(float);ASSERT_GT(end,begin);
                        produced.emplace_back(begin,int(launch.gridDim.y));int admitted=0;
                        for(const char *bank:{GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL,GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL_BATCHED}){
                            const auto base=reinterpret_cast<uintptr_t>(owner.workspace->getBuffer(bank));const auto bytes=owner.workspace->getBufferSize(bank);admitted+=base && begin>=base && end<=base+bytes;}
                        ASSERT_EQ(admitted,1);
                    }else ASSERT_EQ(launch.gridDim.y,1u);
                    std::printf("ROCM_WO_VERIFIER_PACKET,%s,%s,%d,%s,%u,%d,%d\n",format.label,label,m,names[path],launch.gridDim.y,span,int(partial));
                }
                ASSERT_EQ(families[path][0].size(),size_t(repeats));
                const bool expected_native=path==2 || (path==1 && hosted && expectedROCmWoVerifierNative(format.tensor_type,n,k,m));
                ASSERT_EQ(families[path][1].size(),expected_native?size_t(repeats):0u);
                if(!produced.empty()){std::sort(produced.begin(),produced.end());std::sort(reduced.begin(),reduced.end());ASSERT_EQ(produced,reduced);}
                std::vector<GPUGraphKernelNodeInfo> inventory;ASSERT_TRUE(graphs[path]->inspectKernelNodes(inventory));std::set<std::string> printed;
                for(const auto &node:inventory){ASSERT_TRUE(node.valid());ASSERT_EQ(node.local_memory_bytes_per_thread,0u);if(printed.insert(node.name).second)
                    std::printf("ROCM_WO_VERIFIER_RESOURCE,%s,%s,%d,%s,%u,%zu,%s\n",format.label,label,m,names[path],node.registers_per_thread,node.static_shared_memory_bytes,node.name.c_str());}
                std::printf("ROCM_WO_VERIFIER_ENGINE,%s,%s,%d,%s,%zu\n",format.label,label,m,names[path],families[path][1].size());
            }
            const auto poison=[&](int path){
                require(hipMemsetAsync(outputs[path]->gpu_data_ptr(),0xa5,actual.size()*sizeof(uint32_t),stream)==hipSuccess,"Wo output poison failed");
                for(const auto &buffer:stages[path]->requirements.buffers)if(buffer.content_lifetime==WorkspaceContentLifetime::ParticipantExecution){const auto bytes=stages[path]->workspace->getBufferSize(buffer.name);if(bytes)require(hipMemsetAsync(stages[path]->workspace->getBuffer(buffer.name),0xff,bytes,stream)==hipSuccess,"Wo scratch poison failed");}
            };
            const auto observe=[&](int path){require(hipMemcpyAsync(actual.data(),outputs[path]->gpu_data_ptr(),actual.size()*sizeof(uint32_t),hipMemcpyDeviceToHost,stream)==hipSuccess,"Wo output observation failed");require(hipStreamSynchronize(stream)==hipSuccess,"Wo observation join failed");};
            const auto replay=[&](int path){poison(path);require(graphs[path]->launch(),"Wo retained replay failed");observe(path);};
            const auto matches=[&]{for(size_t i=0;i<actual.size();++i){const auto expected=i<size_t(published)*n?oracle[i]:0xa5a5a5a5u;if(actual[i]!=expected)return ::testing::AssertionFailure()<<"element="<<i<<" live="<<published<<" expected_bits="<<expected<<" actual_bits="<<actual[i];}return ::testing::AssertionSuccess();};
            if(profiled){
                ASSERT_EQ(families[1][1].size(),1u);
                for(int warm=0;warm<10;++warm)ASSERT_TRUE(graphs[1]->launch());
                ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);poison(1);ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);
                {ProfileWindow window;const bool launched=graphs[1]->launch();const bool joined=hipStreamSynchronize(stream)==hipSuccess;
                    window.finish();ASSERT_TRUE(launched && joined);}
                observe(1);ASSERT_TRUE(matches());std::printf("ROCM_WO_VERIFIER_PROFILE_OBSERVED,%s,%s,%d\n",format.label,label,m);continue;
            }
            std::vector<int> sequence{m,1,0,m,m};for(int prefix=2;prefix<m;++prefix)sequence.push_back(prefix);
            for(int prefix:sequence){published=prefix;ASSERT_EQ(hipMemcpyAsync(count_owner,&published,sizeof(published),hipMemcpyHostToDevice,stream),hipSuccess);
                for(int path=0;path<3;++path){replay(path);ASSERT_TRUE(matches())<<names[path];
                    if(published==0 || published==m){auto *parent=dynamic_cast<HIPGraphCapture *>(graphs[path].get());for(int family=0;family<2;++family)if(!families[path][family].empty()){
                        Omission omit(parent->executable(),stream,families[path][family]);omit.omit();replay(path);ASSERT_EQ(bool(matches()),published==0)<<"omitted="<<family;omit.restore();replay(path);ASSERT_TRUE(matches());}}
                    if(!published){stages[path]->stage->resetSessionStatePreservingCapturedReplay();ASSERT_TRUE(stages[path]->stage->prepareGraphLaunch(context.get(),stream));}
                }
            }
            if(timed){published=m;ASSERT_EQ(hipMemcpyAsync(count_owner,&published,sizeof(published),hipMemcpyHostToDevice,stream),hipSuccess);
                for(int warm=0;warm<8;++warm)for(auto &graph:graphs)ASSERT_TRUE(graph->launch());
                for(int sample=0;sample<31;++sample){std::array<double,3> us{};for(int order=0;order<3;++order){const int rotation=(sample/2)%3,path=sample%2?(rotation+2-order)%3:(rotation+order)%3;
                    replay(path);ASSERT_TRUE(matches());const auto event=graphs[path]->consumeOrderedTimelineTiming();ASSERT_EQ(event.state,GPUOrderedTimelineTimingState::Complete);ASSERT_EQ(event.samples.size(),1u);
                    us[path]=event.samples[0].elapsed_ms*1000./repeats;ASSERT_TRUE(std::isfinite(us[path]) && us[path]>0);}
                    std::printf("ROCM_WO_VERIFIER_SAMPLE,%s,%s,%d,%d,%.9f,%.9f,%.9f\n",format.label,label,m,sample,us[0],us[1],us[2]);}
            }
        }
    });
}
/** @brief All original source formats at each measured model-owned projection geometry. */
std::vector<Case> cases(){std::vector<Case> out;for(auto geometry:geometries)for(const auto &source:quantizedMoEVerifierFormats())out.push_back({source,geometry});return out;}
}
/** @brief One independent projection owner per geometry/source registration. */
class ROCmWoVerifierTransaction:public ::testing::TestWithParam<Case>{};
TEST_P(ROCmWoVerifierTransaction,AutomaticReplay){prove(GetParam(),Observation::Replay);}
TEST_P(ROCmWoVerifierTransaction,GuardedReplay){prove(GetParam(),Observation::Replay,Extent::Guarded);}
TEST_P(ROCmWoVerifierTransaction,DISABLED_AutomaticPaired){prove(GetParam(),Observation::Paired);}
/** @brief Profile one of the two qualified operation/geometry anchors. */
void profileWo(const char *name){
    ASSERT_EQ(::testing::UnitTest::GetInstance()->test_to_run_count(),1)
        << "A profiler launch must select exactly one Wo geometry before device setup";
    const auto geometry=std::find_if(geometries.begin(),geometries.end(),[&](const auto &g){return std::string_view(g.name)==name;});
    const auto &formats=quantizedMoEVerifierFormats();const auto format=std::find_if(formats.begin(),formats.end(),[](const auto &f){return f.tensor_type==TensorType::IQ4_XS;});
    ASSERT_NE(geometry,geometries.end());ASSERT_NE(format,formats.end());prove({*format,*geometry},Observation::Profile);
}
TEST(ROCmWoVerifierProfile,DISABLED_DenseIQ4XS16){profileWo("Dense");}
TEST(ROCmWoVerifierProfile,DISABLED_MoETP4IQ4XS16){profileWo("MoETP4");}
INSTANTIATE_TEST_SUITE_P(AllSources,ROCmWoVerifierTransaction,::testing::ValuesIn(cases()),
    [](const ::testing::TestParamInfo<Case> &info){return std::string(info.param.geometry.name)+"_"+info.param.source.label;});
