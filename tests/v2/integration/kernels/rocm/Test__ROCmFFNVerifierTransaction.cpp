/**
 * @file Test__ROCmFFNVerifierTransaction.cpp
 * @brief Connected dense gate/up and SwiGLU/down replay against independent M1.
 *
 * Every comparison owns original prepared gate/up/down handles, a merged persistent
 * workspace and outputs. Actual production stages retain the internal publication
 * edge. Current Automatic is measured separately from forced gate/up with Automatic
 * down, so an existing down win cannot be misattributed to the proposed GU change.
 */
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#include "kernels/common/VerifierProjectionBundle.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/rocm/HIPGraphCapture.h"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "execution/compute_stages/stages/FusedGateUpGEMMStage.h"
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
constexpr int hidden=5120, intermediate=17408;
constexpr std::array<int,3> columns{intermediate,intermediate,hidden};
constexpr std::array<int,3> reductions{hidden,hidden,intermediate};
/** @brief Whole-chain controls distinguish existing down admission from GU improvement. */
enum class Policy { Dot4, PriorAutomatic, Automatic, WmmaGateUpAutomaticDown, Wmma };
/** @brief Seed candidate, installed selection, or guarded-capacity exclusion. */
enum class Comparison { ForcedGU, Automatic, GuardedAutomatic };
/** @brief Collect one retained native transaction outside setup and observations. */
class ProfileWindow final {
    bool active_=false;
public:
    /** @brief Resume only after the caller has drained setup and poisoned outputs. */
    ProfileWindow() {
        if(roctxRangePushA("ROCmFFNVerifierTransaction")<0)throw std::runtime_error("FFN profiler range failed");
        if(roctxProfilerResume(0)!=0) {(void)roctxRangePop();throw std::runtime_error("FFN profiler resume failed");}
        active_=true;
    }
    /** @brief Pause before output observation; cleanup remains unconditional on exceptions. */
    void finish() {
        if(!active_)return;
        const int paused=roctxProfilerPause(0),popped=roctxRangePop();active_=false;
        if(paused!=0 || popped<0)throw std::runtime_error("FFN profiler pause failed");
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
/** @brief Fail the component before issuing more work with an invalid owner. */
void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
/** @brief Join exact-stream work before any fixture-owned storage retires. */
struct Drain {
    hipStream_t stream;
    /** @brief Assertion unwinding must retain asynchronous input/output owners. */
    ~Drain(){if(hipStreamSynchronize(stream)!=hipSuccess)std::terminate();}
};
/** @brief Retain both production stages and the common maximum workspace. */
struct Chain {
    std::unique_ptr<DeviceWorkspaceManager> workspace;
    std::unique_ptr<FusedGateUpGEMMStage> gu;
    std::unique_ptr<GEMMStage> down;
    WorkspaceRequirements requirements;
    hipStream_t stream;
    /** @param g Original GU bindings. @param d Connected down bindings.
     * @param context Exact participant. @param exact Exact preparation/capture stream. */
    Chain(FusedGateUpGEMMStage::Params g,GEMMStage::Params d,IDeviceContext &context,hipStream_t exact):stream(exact){
        const int m=g.m;const auto device=g.device_id;
        gu=std::make_unique<FusedGateUpGEMMStage>(std::move(g));down=std::make_unique<GEMMStage>(std::move(d));
        requirements=gu->getWorkspaceRequirements(m,intermediate,hidden);
        requirements.merge(down->getWorkspaceRequirements(m,hidden,intermediate));
        try {
            workspace=std::make_unique<DeviceWorkspaceManager>(device,requirements.total_bytes_with_alignment());
            require(workspace->allocate(requirements),"FFN workspace allocation failed");
            gu->bindWorkspace(workspace.get());down->bindWorkspace(workspace.get());
            gu->setGPUStream(stream);down->setGPUStream(stream);
            require(gu->prepareGraphLaunch(&context,stream) && down->prepareGraphLaunch(&context,stream),"FFN preparation failed");
        }catch(...){if(hipStreamSynchronize(stream)!=hipSuccess)std::terminate();gu->unbindWorkspace();down->unbindWorkspace();throw;}
    }
    /** @brief Graphs retire first; release stage borrows before their workspace. */
    ~Chain(){if(hipStreamSynchronize(stream)!=hipSuccess)std::terminate();gu->unbindWorkspace();down->unbindWorkspace();}
};
/** @brief Suppress a captured producer family while preserving the original graph. */
class Omission {
    hipGraphExec_t graph_;hipStream_t stream_;const std::vector<hipGraphNode_t> &nodes_;bool active_=false;
public:
    /** @param graph Retained executable. @param stream Exact stream. @param nodes Original producers. */
    Omission(hipGraphExec_t graph,hipStream_t stream,const std::vector<hipGraphNode_t> &nodes):graph_(graph),stream_(stream),nodes_(nodes){}
    /** @brief Disable every repeated instance of this producer before dirty replay. */
    void omit(){require(!active_ && !nodes_.empty(),"Invalid FFN omission");active_=true;for(auto node:nodes_)require(hipGraphNodeSetEnabled(graph_,node,0)==hipSuccess,"FFN omission failed");}
    /** @brief Restore every original producer after its work retires. */
    void restore() noexcept {if(active_){if(hipStreamSynchronize(stream_)!=hipSuccess)std::terminate();for(auto node:nodes_)if(hipGraphNodeSetEnabled(graph_,node,1)!=hipSuccess)std::terminate();active_=false;}}
    /** @brief Failed assertions cannot leak an omitted producer. */
    ~Omission(){restore();}
    Omission(const Omission &)=delete;Omission &operator=(const Omission &)=delete;
};
/** @brief Preserve stage-specific candidate policy while recording a real FFN chain.
 * @param policy Explicit comparison arm. @param down Whether this is the down stage.
 * @return Immutable candidate for subsequent graph construction only. */
NativeQuantisedVerifierCandidate candidate(Policy policy,bool down){
    if(policy==Policy::Dot4 || (policy==Policy::PriorAutomatic && !down))return NativeQuantisedVerifierCandidate::Dot4;
    if(policy==Policy::Automatic || policy==Policy::PriorAutomatic || (down && policy==Policy::WmmaGateUpAutomaticDown))return NativeQuantisedVerifierCandidate::Automatic;
    return NativeQuantisedVerifierCandidate::WmmaM16N64;
}
/** @brief Prove actual-chain bytes and measure incremental GU admission.
 * @param format Original gate/up/down source format. @param timed Collect31 rotated complete-chain samples.
 * @param mixed Keep gate original while up/down use independent Q8_0 sources.
 * @param comparison Explicit candidate or measured production selection.
 * @param profile_rows One separately profiled exact width; setup and observations stay outside the marker. */
void prove(const QuantizedVerifierFormatCase &format,bool timed,bool mixed=false,
    Comparison comparison=Comparison::ForcedGU,std::optional<int> profile_rows=std::nullopt){
    int count=0;if(hipGetDeviceCount(&count)!=hipSuccess || !count)GTEST_SKIP()<<"No ROCm device";
    initializeROCmHardwareMetadata();int ordinal=-1;
    for(int device=0;device<count;++device)if(nativeQuantisedHardwareForDevice(device).architecture==native_quantised::NativeQuantisedArchitecture::rocm(11,0,0)){ordinal=device;break;}
    if(ordinal<0)GTEST_SKIP()<<"No native gfx1100 participant";
    const auto device=DeviceId::rocm(ordinal);auto worker=GPUDeviceContextPool::instance().retainAMDContext(ordinal);
    worker->submitAndWait([&]{
        const auto stream=static_cast<hipStream_t>(worker->defaultStream());auto context=IDeviceContext::create(device);ASSERT_TRUE(context);
        const bool installed=comparison!=Comparison::ForcedGU,hosted=comparison!=Comparison::GuardedAutomatic;
        const std::array<Policy,4> policies{Policy::Dot4,Policy::PriorAutomatic,installed?Policy::Automatic:Policy::WmmaGateUpAutomaticDown,Policy::Wmma};
        const std::array<const char *,4> names{"dot4","prior_automatic",installed?"automatic":"wmma_gu_auto_down","wmma"};
        if(profile_rows){ASSERT_EQ(comparison,Comparison::Automatic);ASSERT_FALSE(timed);ASSERT_FALSE(mixed);ASSERT_EQ(*profile_rows,16);}
        std::printf("ROCM_FFN_VERIFIER_MODE,%s,%d,%d\n",format.label,int(mixed),int(comparison));
        const int repeats=timed?8:1;
        std::array<std::shared_ptr<TensorBase>,3> weights;
        std::array<GpuPreparedFFNFixture,5> prepared;
        std::array<std::array<DeviceNativeQuantisedMatrixDesc,3>,5> descriptors;
        std::unique_ptr<FP32Tensor> input,serial_input;
        std::array<std::unique_ptr<FP32Tensor>,3> serial_outputs;
        std::array<std::vector<uint32_t>,3> oracle;
        const Drain setup_drain{stream};
        for(int role=0;role<3;++role)weights[role]=mixed && role>0
            ? TestTensorFactory::createQ8_0Random({size_t(columns[role]),size_t(reductions[role])},89401+role)
            : format.create({size_t(columns[role]),size_t(reductions[role])},89401+role);
        for(int owner=0;owner<5;++owner){
            prepared[owner]=makeGpuPreparedFFNFixture(weights[0].get(),weights[1].get(),weights[2].get(),device,"test.connected_ffn",ModelContextId{uint64_t(89401+owner)});
            const std::array<ITensorGemm *,3> kernels{prepared[owner].gate_kernel,prepared[owner].up_kernel,prepared[owner].down_kernel};
            const std::array<PreparedWeightRef,3> refs{prepared[owner].gate_ref,prepared[owner].up_ref,prepared[owner].down_ref};
            for(int role=0;role<3;++role){
                auto *kernel=dynamic_cast<ROCmQuantisedGemmKernel *>(kernels[role]);ASSERT_TRUE(kernel);
                ASSERT_EQ(prepared[owner].store->binding(refs[role])->tensor,weights[role].get());
                ASSERT_TRUE(kernel->exportNativeQuantisedMatrixDesc(descriptors[owner][role]));
                const auto &desc=descriptors[owner][role];
                const auto &source=mixed && role>0?*std::find_if(quantizedMoEVerifierFormats().begin(),quantizedMoEVerifierFormats().end(),[](const auto &f){return f.tensor_type==TensorType::Q8_0;}):format;
                ASSERT_TRUE(desc.source_identity_present);ASSERT_EQ(desc.source_codebook_id,source.source_codebook_id);
                ASSERT_EQ(bool(desc.source_is_superblock),source.source_is_superblock);ASSERT_EQ(desc.codebook_id,source.device_execution_codebook_id);
                ASSERT_EQ(desc.n,columns[role]);ASSERT_EQ(desc.k,reductions[role]);ASSERT_EQ(desc.blocks_per_row,reductions[role]/32u);
                for(int previous=0;previous<role;++previous){ASSERT_NE(kernels[role],kernels[previous]);ASSERT_NE(refs[role].binding_id,refs[previous].binding_id);}
                for(int previous=0;previous<owner;++previous)ASSERT_NE(desc.payload,descriptors[previous][role].payload);
                std::printf("ROCM_FFN_VERIFIER_SOURCE,%s,%d,%d,%d,%u,%u,%u,%d,%d\n",format.label,int(mixed),owner,role,
                    unsigned(desc.source_codebook_id),unsigned(desc.source_is_superblock),unsigned(desc.codebook_id),desc.n,desc.k);
            }
        }
        input=TestTensorFactory::createFP32({16,hidden});
        for(size_t i=0;i<input->numel();++i)input->mutable_data()[i]=float(int((i*29+i/hidden*11)%251)-125)/128.f;
        ASSERT_TRUE(input->ensureOnDevice(device,stream));serial_input=TestTensorFactory::createFP32({1,hidden});ASSERT_TRUE(serial_input->allocateOnDevice(device,stream));
        for(int role=0;role<3;++role){serial_outputs[role]=TestTensorFactory::createFP32({1,size_t(columns[role])});ASSERT_TRUE(serial_outputs[role]->allocateOnDevice(device,stream));oracle[role].resize(size_t(16)*columns[role]);}
        const auto make_chain=[&](int owner,int m,FP32Tensor *a,const std::array<std::unique_ptr<FP32Tensor>,3> &outputs,std::optional<DeviceRowRange> rows){
            FusedGateUpGEMMStage::Params g;g.device_id=device;g.m=m;g.k=hidden;g.input=a;
            g.w_gate=weights[0].get();g.output_gate=outputs[0].get();g.n_gate=intermediate;g.prepared_ref_gate=prepared[owner].gate_ref;
            g.w_up=weights[1].get();g.output_up=outputs[1].get();g.n_up=intermediate;g.prepared_ref_up=prepared[owner].up_ref;
            g.prepared_store=prepared[owner].store.get();g.force_decode_equivalent_verifier_prefill=m>1;g.verifier_row_range=rows;
            GEMMStage::Params d;d.device_id=device;d.m=m;d.n=hidden;d.k=intermediate;d.A=outputs[1].get();d.gate_input=outputs[0].get();
            d.B=weights[2].get();d.C=outputs[2].get();d.do_swiglu=true;d.alpha=1;d.beta=0;d.transpose_B=false;
            d.prepared_ref=prepared[owner].down_ref;d.prepared_store=prepared[owner].store.get();d.force_decode_equivalent_verifier_prefill=m>1;d.verifier_row_range=rows;
            return std::make_unique<Chain>(std::move(g),std::move(d),*context,stream);
        };
        const auto execute=[&](Chain &chain,FP32Tensor *a,const std::array<std::unique_ptr<FP32Tensor>,3> &outputs,Policy policy){
            {ScopedNativeQuantisedVerifierCandidate select(candidate(policy,false),ordinal);
                if(!with_gpu_coherence(device,{a},{outputs[0].get(),outputs[1].get()},stream,[&]{return chain.gu->execute(context.get());}))return false;}
            {ScopedNativeQuantisedVerifierCandidate select(candidate(policy,true),ordinal);
                return with_gpu_coherence(device,{outputs[0].get(),outputs[1].get()},{outputs[2].get()},stream,[&]{return chain.down->execute(context.get());});}
        };
        {
            auto serial=make_chain(4,1,serial_input.get(),serial_outputs,std::nullopt);const Drain drain{stream};
            for(int row=0;row<16;++row){
                ASSERT_EQ(hipMemcpyAsync(serial_input->gpu_data_ptr(),static_cast<const float *>(input->gpu_data_ptr())+row*hidden,hidden*sizeof(float),hipMemcpyDeviceToDevice,stream),hipSuccess);
                TransferEngine::publishCurrentDeviceWrite(serial_input.get(),stream);ASSERT_TRUE(execute(*serial,serial_input.get(),serial_outputs,Policy::Automatic));
                for(int role=0;role<3;++role)ASSERT_EQ(hipMemcpyAsync(oracle[role].data()+size_t(row)*columns[role],serial_outputs[role]->gpu_data_ptr(),columns[role]*sizeof(float),hipMemcpyDeviceToHost,stream),hipSuccess);
            }
            ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);
        }
        for(const auto &out:oracle){ASSERT_TRUE(std::all_of(out.begin(),out.end(),[](auto word){return std::isfinite(std::bit_cast<float>(word));}));ASSERT_TRUE(std::any_of(out.begin(),out.end(),[](auto word){return word!=0;}));}
        for(int m=profile_rows.value_or(2);m<=profile_rows.value_or(16);++m){
            SCOPED_TRACE(::testing::Message()<<format.label<<" mixed="<<mixed<<" M="<<m);
            auto live=TestTensorFactory::createINT32({1});live->mutable_int32_data()[0]=m;
            std::array<std::array<std::unique_ptr<FP32Tensor>,3>,4> outputs;
            std::array<std::vector<uint32_t>,3> actual;
            std::array<std::unique_ptr<Chain>,4> chains;
            std::array<std::unique_ptr<IGPUGraphCapture>,4> fragments,graphs;
            // Two Q8 publishers and the GU/down native families are independently omitted.
            std::array<std::array<std::vector<hipGraphNode_t>,4>,4> families;
            int32_t published=m; // Host staging outlives the exact-stream drain.
            const Drain drain{stream};ASSERT_TRUE(live->ensureOnDevice(device,stream));
            auto *count_owner=static_cast<int32_t *>(live->gpu_data_ptr());const auto rows=DeviceRowRange::deviceCounted(m,count_owner);
            for(int path=0;path<4;++path){
                for(int role=0;role<3;++role){outputs[path][role]=TestTensorFactory::createFP32({size_t(m),size_t(columns[role])});ASSERT_TRUE(outputs[path][role]->allocateOnDevice(device,stream));actual[role].resize(size_t(m)*columns[role]);}
                chains[path]=make_chain(path,m,input.get(),outputs[path],rows);auto &chain=*chains[path];
                ScopedCapturedVerifierExtent extent(hosted?CapturedVerifierExtent::exactHostedRequest(device,rows):CapturedVerifierExtent{});
                ASSERT_TRUE(execute(chain,input.get(),outputs[path],policies[path]));ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);
                TransferEngine::requireDeviceInput(input.get(),device,stream);TransferEngine::requireDeviceInput(live.get(),device,stream);
                std::vector<GraphCaptureDependencyLedger::StagePlan> plans;
                for(int repeat=0;repeat<repeats;++repeat){
                    const auto producer=plans.size();
                    plans.push_back({.stage_identity=chain.gu.get(),.stage_name="dense_gate_up",.external_inputs={input.get(),live.get()},.outputs={outputs[path][0].get(),outputs[path][1].get()}});
                    plans.push_back({.stage_identity=chain.down.get(),.stage_name="dense_swiglu_down",.external_inputs={live.get()},
                        .internal_inputs={{outputs[path][0].get(),producer},{outputs[path][1].get(),producer}},.outputs={outputs[path][2].get()}});
                }
                GraphCaptureDependencyLedger ledger(device,stream,std::move(plans),"Connected dense FFN verifier");
                graphs[path]=worker->createGraphCapture(stream);fragments[path]=graphs[path]->createOrderedTimelineFragment();ASSERT_TRUE(fragments[path]);
                ScopedBackendGraphCapture capture(*worker,*fragments[path],"original GU through SwiGLU/down",&ledger);ASSERT_TRUE(capture.begin());
                for(int repeat=0;repeat<repeats;++repeat){
                    {ScopedGraphCaptureStage stage(chain.gu.get());ScopedNativeQuantisedVerifierCandidate select(candidate(policies[path],false),ordinal);ASSERT_TRUE(chain.gu->execute(context.get()));stage.complete();}
                    ASSERT_EQ(ScopedVerifierProjectionBundle::current(),nullptr);
                    {ScopedGraphCaptureStage stage(chain.down.get());ScopedNativeQuantisedVerifierCandidate select(candidate(policies[path],true),ordinal);ASSERT_TRUE(chain.down->execute(context.get()));stage.complete();}
                }
                capture.finish();const std::array<GPUOrderedTimelineStep,1> steps{{{.name=names[path],.kind=GPUOrderedTimelineStepKind::CapturedFragment,.capture=fragments[path].get()}}};
                ASSERT_TRUE(graphs[path]->buildOrderedTimelineTransaction(steps,profile_rows?GPUOrderedTimelineInstrumentation::Disabled:GPUOrderedTimelineInstrumentation::PerStepEvents));ASSERT_TRUE(graphs[path]->instantiate());
                auto *parent=dynamic_cast<HIPGraphCapture *>(graphs[path].get());ASSERT_TRUE(parent);
                size_t size=0;ASSERT_EQ(hipGraphGetNodes(parent->graph(),nullptr,&size),hipSuccess);std::vector<hipGraphNode_t> nodes(size);ASSERT_EQ(hipGraphGetNodes(parent->graph(),nodes.data(),&size),hipSuccess);
                const auto quant=chain.workspace->getBuffer(GemmWorkspaceBuffers::QUANT_A);
                const auto scales=chain.workspace->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE);
                const auto sums=chain.workspace->getBuffer(GemmWorkspaceBuffers::SUMS_A_BLOCKWISE);
                std::array<int,3> contributions{};
                // Native producers and reducers are enumerated independently;
                // graph node enumeration does not promise topological order.
                using Partial = std::tuple<int,uintptr_t,int>;
                std::vector<Partial> produced,reduced;
                for(auto node:nodes){
                    hipGraphNodeType type;ASSERT_EQ(hipGraphNodeGetType(node,&type),hipSuccess);if(type!=hipGraphNodeTypeKernel)continue;
                    hipKernelNodeParams launch{};ASSERT_EQ(hipGraphKernelNodeGetParams(node,&launch),hipSuccess);
                    const char *symbol=hipKernelNameRefByPtr(launch.func,stream);ASSERT_TRUE(symbol);const std::string_view name(symbol);
                    if(name.find("quantizeActivationsQ8_blockwise_v4_with_sums_kernel")!=std::string_view::npos){
                        families[path][0].push_back(node);ASSERT_NE(name.find("ILb1E"),std::string_view::npos);ASSERT_EQ(*static_cast<const float **>(launch.kernelParams[0]),input->gpu_data_ptr());
                        ASSERT_EQ(*static_cast<int8_t **>(launch.kernelParams[1]),quant);ASSERT_EQ(*static_cast<float **>(launch.kernelParams[2]),scales);ASSERT_EQ(*static_cast<int32_t **>(launch.kernelParams[3]),sums);
                        ASSERT_EQ(*static_cast<int *>(launch.kernelParams[4]),m);ASSERT_EQ(*static_cast<int *>(launch.kernelParams[5]),hidden);ASSERT_EQ(*static_cast<int *>(launch.kernelParams[6]),hidden/32);
                        ASSERT_EQ(static_cast<const DeviceRowRange *>(launch.kernelParams[7])->captureIdentity(),rows.captureIdentity());
                        ASSERT_EQ(launch.gridDim.x,unsigned(m*(hidden/32)));ASSERT_EQ(launch.gridDim.y,1u);ASSERT_EQ(launch.gridDim.z,1u);ASSERT_EQ(launch.blockDim.x,32u);
                    }
                    if(name.find("rocm_fused_swiglu_quantize_blockwise_kernel")!=std::string_view::npos){
                        families[path][1].push_back(node);ASSERT_NE(name.find("ILb1E"),std::string_view::npos);ASSERT_EQ(*static_cast<const float **>(launch.kernelParams[0]),outputs[path][0]->gpu_data_ptr());ASSERT_EQ(*static_cast<const float **>(launch.kernelParams[1]),outputs[path][1]->gpu_data_ptr());
                        ASSERT_EQ(*static_cast<int8_t **>(launch.kernelParams[2]),quant);ASSERT_EQ(*static_cast<float **>(launch.kernelParams[3]),scales);ASSERT_EQ(*static_cast<int32_t **>(launch.kernelParams[4]),sums);
                        ASSERT_EQ(*static_cast<int *>(launch.kernelParams[5]),m);ASSERT_EQ(*static_cast<int *>(launch.kernelParams[6]),intermediate);
                        ASSERT_EQ(static_cast<const DeviceRowRange *>(launch.kernelParams[7])->captureIdentity(),rows.captureIdentity());
                        ASSERT_EQ(launch.gridDim.x,unsigned(m));ASSERT_EQ(launch.gridDim.y,1u);ASSERT_EQ(launch.gridDim.z,1u);ASSERT_EQ(launch.blockDim.x,256u);
                    }
                    if(name.find("gemv_native_quantised_reduce_small_m_batched_kernel_t")!=std::string_view::npos){
                        const auto &packet=*static_cast<const NativeQuantisedSmallMBatchedArgs *>(launch.kernelParams[0]);
                        const int partitions=*static_cast<int *>(launch.kernelParams[1]);ASSERT_GT(partitions,1);
                        ASSERT_EQ(packet.rows.captureIdentity(),rows.captureIdentity());ASSERT_EQ(packet.work.captureIdentity(),ProjectionRowWork::fromPrefix(rows).captureIdentity());
                        ASSERT_GE(packet.num_projections,1);ASSERT_LE(packet.num_projections,2);ASSERT_EQ(launch.gridDim.x,unsigned(packet.total_grid_n));
                        for(int member=0;member<packet.num_projections;++member){
                            int role=-1;for(int p=0;p<2;++p)if(packet.output[member]==outputs[path][p]->gpu_data_ptr())role=p;
                            ASSERT_GE(role,0);const auto &desc=descriptors[path][role];
                            ASSERT_EQ(packet.payload[member],desc.payload);ASSERT_EQ(packet.scales[member],desc.scales);ASSERT_EQ(packet.mins[member],desc.mins);ASSERT_EQ(packet.emins[member],desc.emins);
                            ASSERT_EQ(packet.N[member],columns[role]);ASSERT_EQ(packet.codebook[member],desc.codebook_id);ASSERT_EQ(packet.bias[member],nullptr);
                            reduced.emplace_back(role,reinterpret_cast<uintptr_t>(packet.partial[member]),partitions);
                        }
                    }
                    if(name.find("gemv_native_quantised_reduce_small_m_kernel_t")!=std::string_view::npos){
                        ASSERT_EQ(*static_cast<float **>(launch.kernelParams[1]),outputs[path][2]->gpu_data_ptr());
                        ASSERT_EQ(*static_cast<int *>(launch.kernelParams[2]),m);ASSERT_EQ(*static_cast<int *>(launch.kernelParams[3]),hidden);
                        const int partitions=*static_cast<int *>(launch.kernelParams[4]);ASSERT_GT(partitions,1);
                        ASSERT_EQ(static_cast<const DeviceRowRange *>(launch.kernelParams[5])->captureIdentity(),rows.captureIdentity());
                        ASSERT_EQ(static_cast<const ProjectionRowWork *>(launch.kernelParams[6])->captureIdentity(),ProjectionRowWork::fromPrefix(rows).captureIdentity());
                        reduced.emplace_back(2,reinterpret_cast<uintptr_t>(*static_cast<const float **>(launch.kernelParams[0])),partitions);
                    }
                    if(name.find("native_quantised_verifier_wmma_m16n64")==std::string_view::npos)continue;
                    const auto &packet=*static_cast<const NativeQuantisedSmallMBatchedArgs *>(launch.kernelParams[3]);
                    const int k=*static_cast<int *>(launch.kernelParams[4]);const bool down=k==intermediate;ASSERT_TRUE(down || k==hidden);
                    if(profile_rows && path==2)std::printf("ROCM_FFN_VERIFIER_PROFILE_PACKET,%s,%d,%d,%d,%u,%u,%u,%u\n",
                        format.label,m,down?2:0,packet.num_projections,launch.gridDim.x,launch.gridDim.y,launch.gridDim.z,launch.blockDim.x);
                    families[path][down?3:2].push_back(node);ASSERT_EQ(packet.rows.captureIdentity(),rows.captureIdentity());ASSERT_EQ(packet.work.captureIdentity(),ProjectionRowWork::fromPrefix(rows).captureIdentity());
                    ASSERT_EQ(*static_cast<const int8_t **>(launch.kernelParams[0]),quant);ASSERT_EQ(*static_cast<const float **>(launch.kernelParams[1]),scales);ASSERT_EQ(*static_cast<const int32_t **>(launch.kernelParams[2]),sums);
                    ASSERT_EQ(launch.gridDim.x,unsigned(packet.total_grid_n));ASSERT_GT(launch.gridDim.y,0u);ASSERT_EQ(launch.gridDim.z,1u);ASSERT_EQ(launch.blockDim.x,128u);
                    ASSERT_EQ(*static_cast<int *>(launch.kernelParams[5]),(k/32+int(launch.gridDim.y)-1)/int(launch.gridDim.y));
                    const bool partial=name.find("VerifierWmmaPublicationE2E")!=std::string_view::npos;
                    if(!partial)ASSERT_EQ(launch.gridDim.y,1u);
                    ASSERT_GE(packet.num_projections,1);ASSERT_LE(packet.num_projections,down?1:2);
                    std::vector<std::pair<uintptr_t,uintptr_t>> planes;
                    for(int member=0;member<packet.num_projections;++member){
                        int role=-1;for(int p=0;p<3;++p)if(packet.payload[member]==descriptors[path][p].payload)role=p;
                        ASSERT_GE(role,0);ASSERT_EQ(role==2,down);const auto &desc=descriptors[path][role];
                        ASSERT_EQ(packet.N[member],columns[role]);ASSERT_EQ(packet.codebook[member],desc.codebook_id);ASSERT_EQ(packet.scales[member],desc.scales);ASSERT_EQ(packet.mins[member],desc.mins);ASSERT_EQ(packet.emins[member],desc.emins);
                        ASSERT_EQ(packet.output[member],outputs[path][role]->gpu_data_ptr());ASSERT_EQ(packet.bias[member],nullptr);++contributions[role];
                        if(partial){
                            const auto begin=reinterpret_cast<uintptr_t>(packet.partial[member]),end=begin+size_t(launch.gridDim.y)*m*columns[role]*sizeof(float);ASSERT_GT(end,begin);
                            produced.emplace_back(role,begin,int(launch.gridDim.y));
                            int admitted=0;for(const char *bank:{GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL,GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL_BATCHED}){
                                const auto base=reinterpret_cast<uintptr_t>(chain.workspace->getBuffer(bank));const auto bytes=chain.workspace->getBufferSize(bank);admitted+=base && begin>=base && end<=base+bytes;}
                            ASSERT_EQ(admitted,1);for(const auto &[other_begin,other_end]:planes)ASSERT_TRUE(end<=other_begin || begin>=other_end);planes.emplace_back(begin,end);
                        }
                    }
                }
                ASSERT_EQ(families[path][0].size(),size_t(repeats));ASSERT_EQ(families[path][1].size(),size_t(repeats));
                if(path==0){ASSERT_TRUE(families[path][2].empty());ASSERT_TRUE(families[path][3].empty());}
                if(path==1)ASSERT_TRUE(families[path][2].empty());
                if(path==1 || path==2){
                    if(m==2 || !hosted)ASSERT_TRUE(families[path][3].empty());
                    if(m==16 && hosted)ASSERT_EQ(contributions[2],repeats);
                    if(hosted && !mixed && format.tensor_type==TensorType::IQ4_XS)ASSERT_EQ(contributions[2],m>=5?repeats:0);
                }
                if(path==3 || (path==2 && !installed)){ASSERT_EQ(contributions[0],repeats);ASSERT_EQ(contributions[1],repeats);}
                if(path==2 && installed){
                    if(m==2 || mixed || !hosted)ASSERT_TRUE(families[path][2].empty());
                    if(m==16 && !mixed && hosted){ASSERT_EQ(contributions[0],repeats);ASSERT_EQ(contributions[1],repeats);}
                }
                if(path==3)ASSERT_EQ(contributions[2],repeats);
                for(int role=0;role<3;++role)ASSERT_TRUE(contributions[role]==0 || contributions[role]==repeats);
                // Dot4 may have its own reducer. Only a native partial creates
                // this producer/reducer equality obligation for its exact role.
                std::erase_if(reduced,[&](const Partial &entry){return std::none_of(produced.begin(),produced.end(),[&](const Partial &p){return std::get<0>(p)==std::get<0>(entry);});});
                std::sort(produced.begin(),produced.end());std::sort(reduced.begin(),reduced.end());ASSERT_EQ(produced,reduced);
                std::vector<GPUGraphKernelNodeInfo> inventory;ASSERT_TRUE(graphs[path]->inspectKernelNodes(inventory));std::set<std::string> printed;
                for(const auto &node:inventory){ASSERT_TRUE(node.valid());ASSERT_EQ(node.local_memory_bytes_per_thread,0u);if(printed.insert(node.name).second)
                    std::printf("ROCM_FFN_VERIFIER_RESOURCE,%s,%d,%d,%s,%u,%zu,%s\n",format.label,int(mixed),m,names[path],node.registers_per_thread,node.static_shared_memory_bytes,node.name.c_str());}
                std::printf("ROCM_FFN_VERIFIER_ENGINE,%s,%d,%d,%s,%d,%d,%d\n",format.label,int(mixed),m,names[path],contributions[0],contributions[1],contributions[2]);
            }
            // The incremental control changes GU only; down must keep the same actual engine.
            ASSERT_EQ(families[1][3].size(),families[2][3].size());
            const auto poison=[&](int path){
                for(int role=0;role<3;++role)require(hipMemsetAsync(outputs[path][role]->gpu_data_ptr(),0xa5,actual[role].size()*sizeof(uint32_t),stream)==hipSuccess,"FFN output poison failed");
                for(const auto &buffer:chains[path]->requirements.buffers)if(buffer.content_lifetime==WorkspaceContentLifetime::ParticipantExecution){
                    const auto bytes=chains[path]->workspace->getBufferSize(buffer.name);if(bytes)require(hipMemsetAsync(chains[path]->workspace->getBuffer(buffer.name),0xff,bytes,stream)==hipSuccess,"FFN scratch poison failed");}
            };
            const auto observe=[&](int path){
                for(int role=0;role<3;++role)require(hipMemcpyAsync(actual[role].data(),outputs[path][role]->gpu_data_ptr(),actual[role].size()*sizeof(uint32_t),hipMemcpyDeviceToHost,stream)==hipSuccess,"FFN output observation failed");
                require(hipStreamSynchronize(stream)==hipSuccess,"FFN observation join failed");
            };
            const auto replay=[&](int path){poison(path);require(graphs[path]->launch(),"FFN retained replay failed");observe(path);};
            const auto matches_role=[&](int role){
                for(size_t i=0;i<actual[role].size();++i){const auto expected=i<size_t(published)*columns[role]?oracle[role][i]:0xa5a5a5a5u;
                    if(actual[role][i]!=expected)return ::testing::AssertionFailure()<<"role="<<role<<" element="<<i<<" live="<<published<<" expected_bits="<<expected<<" actual_bits="<<actual[role][i];}
                return ::testing::AssertionSuccess();
            };
            const auto matches=[&]{for(int role=0;role<3;++role){const auto result=matches_role(role);if(!result)return result;}return ::testing::AssertionSuccess();};
            if(profile_rows){
                ASSERT_EQ(families[2][2].size(),1u);ASSERT_EQ(families[2][3].size(),1u);
                for(int warm=0;warm<10;++warm)ASSERT_TRUE(graphs[2]->launch());
                ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);poison(2);ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);
                {ProfileWindow window;const bool launched=graphs[2]->launch();const bool joined=hipStreamSynchronize(stream)==hipSuccess;
                    window.finish();ASSERT_TRUE(launched && joined);}
                observe(2);ASSERT_TRUE(matches());std::printf("ROCM_FFN_VERIFIER_PROFILE_OBSERVED,%s,%d\n",format.label,m);continue;
            }
            std::vector<int> sequence{m,1,0,m,m};for(int prefix=2;prefix<m;++prefix)sequence.push_back(prefix);
            for(int prefix:sequence){
                published=prefix;ASSERT_EQ(hipMemcpyAsync(count_owner,&published,sizeof(published),hipMemcpyHostToDevice,stream),hipSuccess);
                for(int path=0;path<4;++path){
                    replay(path);ASSERT_TRUE(matches())<<names[path];
                    if(published==0 || published==m){
                        auto *parent=dynamic_cast<HIPGraphCapture *>(graphs[path].get());
                        for(int family=0;family<4;++family)if(!families[path][family].empty()){
                            Omission omit(parent->executable(),stream,families[path][family]);omit.omit();replay(path);ASSERT_EQ(bool(matches()),published==0)<<"omitted="<<family;
                            if(family==1 || family==3){ASSERT_TRUE(matches_role(0));ASSERT_TRUE(matches_role(1));}
                            omit.restore();replay(path);ASSERT_TRUE(matches());
                        }
                    }
                    if(!published){chains[path]->gu->resetSessionStatePreservingCapturedReplay();chains[path]->down->resetSessionStatePreservingCapturedReplay();
                        ASSERT_TRUE(chains[path]->gu->prepareGraphLaunch(context.get(),stream));ASSERT_TRUE(chains[path]->down->prepareGraphLaunch(context.get(),stream));}
                }
            }
            if(timed){
                published=m;ASSERT_EQ(hipMemcpyAsync(count_owner,&published,sizeof(published),hipMemcpyHostToDevice,stream),hipSuccess);
                for(int warm=0;warm<8;++warm)for(auto &graph:graphs)ASSERT_TRUE(graph->launch());
                for(int sample=0;sample<31;++sample){std::array<double,4> us{};
                    for(int order=0;order<4;++order){const int rotation=(sample/2)%4;const int path=sample%2?(rotation+3-order)%4:(rotation+order)%4;replay(path);ASSERT_TRUE(matches());
                        const auto event=graphs[path]->consumeOrderedTimelineTiming();ASSERT_EQ(event.state,GPUOrderedTimelineTimingState::Complete);ASSERT_EQ(event.samples.size(),1u);
                        us[path]=event.samples[0].elapsed_ms*1000./repeats;ASSERT_TRUE(std::isfinite(us[path]) && us[path]>0);}
                    std::printf("ROCM_FFN_VERIFIER_SAMPLE,%s,%d,%d,%d,%.9f,%.9f,%.9f,%.9f\n",format.label,int(mixed),m,sample,us[0],us[1],us[2],us[3]);}
            }
        }
    });
}
}
/** @brief Each original source format owns an independent connected-stage replay. */
class ROCmFFNVerifierTransaction:public ::testing::TestWithParam<QuantizedVerifierFormatCase>{};
TEST_P(ROCmFFNVerifierTransaction,AutomaticReplay){prove(GetParam(),false,false,Comparison::Automatic);}
TEST_P(ROCmFFNVerifierTransaction,DISABLED_Paired){prove(GetParam(),true);}
TEST_P(ROCmFFNVerifierTransaction,DISABLED_AutomaticPaired){prove(GetParam(),true,false,Comparison::Automatic);}
TEST(ROCmFFNVerifierMixed,IndependentSourcesReplay){prove(quantizedMoEVerifierFormats().front(),false,true,Comparison::Automatic);}
TEST(ROCmFFNVerifierMixed,DISABLED_IndependentSourcesPaired){prove(quantizedMoEVerifierFormats().front(),true,true);}
/** @test Guarded capacity cannot acquire the hosted GU/down timing authority. */
TEST(ROCmFFNVerifierAutomatic,GuardedCapacityReplay){prove(quantizedMoEVerifierFormats().front(),false,false,Comparison::GuardedAutomatic);}
/** @test One actual IQ4_XS complete chain supplies separately isolated GU and down profiles. */
TEST(ROCmFFNVerifierProfile,DISABLED_IQ4XSM16){
    const auto &formats=quantizedMoEVerifierFormats();const auto format=std::find_if(formats.begin(),formats.end(),[](const auto &f){return f.tensor_type==TensorType::IQ4_XS;});
    ASSERT_NE(format,formats.end());prove(*format,false,false,Comparison::Automatic,16);
}
INSTANTIATE_TEST_SUITE_P(AllSources,ROCmFFNVerifierTransaction,::testing::ValuesIn(quantizedMoEVerifierFormats()),
    [](const ::testing::TestParamInfo<QuantizedVerifierFormatCase> &info){return std::string(info.param.label);});
