/**
 * @file Test__ROCmGDNVerifierTransaction.cpp
 * @brief Complete four-source GDN verifier transactions against independent M1.
 *
 * Each candidate owns a separate prepared store, kernels, workspace and outputs.
 * Original quantized QKV/Z weights and F32 alpha/beta (prepared as FP16RNE)
 * retain their source identities. TwoTermQ8 corrects QKV through its own producer;
 * forced WMMA then applies only to Z. The complete stage is the timing boundary.
 */
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include "backends/GPUDeviceContextPool.h"
#include "backends/rocm/HIPGraphCapture.h"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "execution/compute_stages/stages/GDNProjectionStage.h"
#include <rocprofiler-sdk-roctx/roctx.h>
#include "execution/local_execution/coherence/GpuCoherence.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/rocm/gemm/ROCmNativeQuantisedVerifierLaunch.h"
#include "kernels/rocm/gemm/ROCmNativeQuantisedArchitecture.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "loaders/PreparedHalfWeights.h"
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
#include <vector>
#include <tuple>

using namespace llaminar2;
using namespace llaminar2::rocm;
using namespace llaminar2::test;
namespace {
/** @brief Candidate comparison retains the original hosted-width scheduling fact. */
enum class PairPolicy {ForcedWmma,GuardedAutomatic,ExactHostedAutomatic};
constexpr int input_width=2560;
constexpr std::array<int,4> widths{10240,6144,48,48};
/** @brief Collect one retained native transaction outside setup and observations. */
class ProfileWindow final {
    bool active_=false;
public:
    /** @brief Resume only after the caller has drained setup and poisoned outputs. */
    ProfileWindow() {
        if(roctxRangePushA("ROCmGDNVerifierTransaction")<0)throw std::runtime_error("GDN profiler range failed");
        if(roctxProfilerResume(0)!=0) {(void)roctxRangePop();throw std::runtime_error("GDN profiler resume failed");}
        active_=true;
    }
    /** @brief Pause before output observation; cleanup remains unconditional on exceptions. */
    void finish() {
        if(!active_)return;
        const int paused=roctxProfilerPause(0),popped=roctxRangePop();active_=false;
        if(paused!=0 || popped<0)throw std::runtime_error("GDN profiler pause failed");
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
/** @brief Reject an incomplete fixture before any further native work. */
void require(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
/** @brief Retire exact-stream work before borrowed owners and host observations. */
struct Drain {
    hipStream_t stream;
    /** @brief Join on assertion unwind as well as normal completion. */
    ~Drain() {if(hipStreamSynchronize(stream)!=hipSuccess)std::terminate();}
};
/** @brief Four immutable original sources resolved by one independent store. */
struct PreparedOwner {
    std::vector<std::shared_ptr<LoadOrchestrator>> loaders;
    std::unique_ptr<PreparedWeightStore> store;
    std::array<PreparedWeightRef,4> refs;
    std::array<ITensorGemm *,4> kernels{};
    /** @param weights Original native QKV/Z and materialized original half controls.
     * @param device Native participant. @param identity Distinct owner/model identity. */
    PreparedOwner(const std::array<std::shared_ptr<TensorBase>,4> &weights,DeviceId device,uint64_t identity)
        :store(std::make_unique<PreparedWeightStore>(ModelContextId{identity})) {
        for(size_t p=0;p<4;++p) {
            const auto name="test.gdn_verifier."+std::to_string(p);
            refs[p]=p<2?registerGpuPreparedGemmInStore(weights[p].get(),device,name,store->modelId(),store.get(),&loaders)
                :registerGpuPreparedFloatingPointGemmInStore(weights[p].get(),device,name,*store,loaders,FloatingActivationOperandPrecision::FP16RNE);
            kernels[p]=store->gemmKernel(refs[p]);
            require(kernels[p] && store->binding(refs[p])->tensor==weights[p].get(),"GDN prepared source binding changed");
            for(size_t previous=0;previous<p;++previous)
                require(refs[p].binding_id!=refs[previous].binding_id && kernels[p]!=kernels[previous],"GDN source handles alias");
        }
    }
};
/** @brief Own the complete stage's persistent workspace until its graphs retire. */
struct StageOwner {
    std::unique_ptr<DeviceWorkspaceManager> workspace;
    std::unique_ptr<GDNProjectionStage> stage;
    WorkspaceRequirements requirements;
    hipStream_t stream;
    /** @param params Complete four-source operation. @param context Native participant.
     * @param exact Exact stream used for preparation, recording and retirement. */
    StageOwner(GDNProjectionStage::Params params,IDeviceContext &context,hipStream_t exact):stream(exact) {
        const int m=params.m,k=params.k;const auto device=params.device_id;
        stage=std::make_unique<GDNProjectionStage>(std::move(params));requirements=stage->getWorkspaceRequirements(m,0,k);
        try {
            workspace=std::make_unique<DeviceWorkspaceManager>(device,requirements.total_bytes_with_alignment());
            require(workspace->allocate(requirements),"GDN workspace allocation failed");stage->bindWorkspace(workspace.get());
            require(stage->prepareGraphLaunch(&context,stream),"GDN stage preparation failed");
        } catch(...) {if(hipStreamSynchronize(stream)!=hipSuccess)std::terminate();stage->unbindWorkspace();throw;}
    }
    /** @brief Retire every borrowed workspace reference before releasing storage. */
    ~StageOwner() {if(hipStreamSynchronize(stream)!=hipSuccess)std::terminate();stage->unbindWorkspace();}
};
/** @brief Disable a real captured family and restore it even on failed assertions. */
class NodeOmission {
    hipGraphExec_t graph_;hipStream_t stream_;const std::vector<hipGraphNode_t> &nodes_;bool active_=false;
public:
    /** @param graph Retained parent. @param stream Exact stream. @param nodes Borrowed native family. */
    NodeOmission(hipGraphExec_t graph,hipStream_t stream,const std::vector<hipGraphNode_t> &nodes):graph_(graph),stream_(stream),nodes_(nodes) {}
    /** @brief Install the omission before dirty replay. */
    void omit() {require(!active_ && !nodes_.empty(),"Invalid GDN omission");active_=true;for(auto node:nodes_)require(hipGraphNodeSetEnabled(graph_,node,0)==hipSuccess,"GDN omission failed");}
    /** @brief Restore after draining every replay using the omitted family. */
    void restore() noexcept {if(active_){if(hipStreamSynchronize(stream_)!=hipSuccess)std::terminate();for(auto node:nodes_)if(hipGraphNodeSetEnabled(graph_,node,1)!=hipSuccess)std::terminate();active_=false;}}
    /** @brief Assertion unwinding cannot leak the injected defect. */
    ~NodeOmission() {restore();}
    NodeOmission(const NodeOmission &)=delete;NodeOmission &operator=(const NodeOmission &)=delete;
};
/** @brief Compare complete stage candidates through dirty count/reset replay and paired events.
 * @param format Original QKV/Z source format, independent of prepared execution encoding.
 * @param precision SingleQ8 or corrected TwoTermQ8 for QKV only.
 * @param timed Collect31 alternating pairs at every full physical M2..16.
 * @param mixed Use independent Q8_0 Z while preserving the selected QKV source.
 * @param pairing Explicit candidate or production Automatic with its extent contract.
 * @param profile_rows Optional single width for separately scoped counter collection. */
void prove(const QuantizedVerifierFormatCase &format,QuantizedActivationOperandPrecision precision,bool timed,bool mixed=false,PairPolicy pairing=PairPolicy::ForcedWmma,std::optional<int> profile_rows=std::nullopt) {
    int count=0;if(hipGetDeviceCount(&count)!=hipSuccess || !count)GTEST_SKIP()<<"No ROCm device";
    initializeROCmHardwareMetadata();
    int ordinal=-1;
    for(int candidate=0;candidate<count;++candidate)
        if(nativeQuantisedHardwareForDevice(candidate).architecture==native_quantised::NativeQuantisedArchitecture::rocm(11,0,0)){ordinal=candidate;break;}
    if(ordinal<0)GTEST_SKIP()<<"No native gfx1100 participant";
    const auto device=DeviceId::rocm(ordinal);auto worker=GPUDeviceContextPool::instance().retainAMDContext(ordinal);
    worker->submitAndWait([&] {
        const auto stream=static_cast<hipStream_t>(worker->defaultStream());const bool corrected=precision==QuantizedActivationOperandPrecision::TwoTermQ8;
        const bool automatic=pairing!=PairPolicy::ForcedWmma;
        if(profile_rows){ASSERT_EQ(pairing,PairPolicy::ExactHostedAutomatic);ASSERT_FALSE(timed);ASSERT_FALSE(mixed);ASSERT_GE(*profile_rows,2);ASSERT_LE(*profile_rows,16);}
        const int repeats=timed?8:1;const char *mode=corrected?"TwoTermQ8":"SingleQ8";
        auto context=IDeviceContext::create(device);ASSERT_TRUE(context);
        std::array<std::shared_ptr<TensorBase>,4> originals,weights;
        std::array<std::unique_ptr<PreparedOwner>,3> prepared;
        std::array<std::array<DeviceNativeQuantisedMatrixDesc,2>,3> descriptors;
        std::unique_ptr<FP32Tensor> input,serial_input;
        std::array<std::unique_ptr<FP32Tensor>,4> serial_outputs;
        std::array<std::vector<uint32_t>,4> oracle;
        const Drain setup_drain{stream};
        originals[0]=format.create({size_t(widths[0]),input_width},89301);
        originals[1]=mixed?TestTensorFactory::createQ8_0Random({size_t(widths[1]),input_width},89302)
            :format.create({size_t(widths[1]),input_width},89302);
        for(size_t p=2;p<4;++p)originals[p]=TestTensorFactory::createFP32Random({size_t(widths[p]),input_width},-.125f,.125f,89301+p);
        weights=originals;for(size_t p=2;p<4;++p)weights[p]=materializePreparedHalfWeights(*originals[p]);
        for(int owner=0;owner<3;++owner)prepared[owner]=std::make_unique<PreparedOwner>(weights,device,89301+owner);
        for(int owner=0;owner<3;++owner)for(int p=0;p<4;++p) {
            for(int prior=0;prior<owner;++prior)ASSERT_NE(prepared[owner]->kernels[p],prepared[prior]->kernels[p]);
            if(p>=2)ASSERT_EQ(prepared[owner]->kernels[p]->floatingActivationOperandPrecision(),FloatingActivationOperandPrecision::FP16RNE);
            if(p<2) {
                auto *kernel=dynamic_cast<ROCmQuantisedGemmKernel *>(prepared[owner]->kernels[p]);ASSERT_TRUE(kernel);
                ASSERT_TRUE(kernel->exportNativeQuantisedMatrixDesc(descriptors[owner][p]));
                const auto &desc=descriptors[owner][p];
                const auto &source=mixed && p==1?*std::find_if(quantizedMoEVerifierFormats().begin(),quantizedMoEVerifierFormats().end(),
                    [](const auto &f){return f.tensor_type==TensorType::Q8_0;}):format;
                ASSERT_TRUE(desc.source_identity_present);ASSERT_EQ(desc.source_codebook_id,source.source_codebook_id);
                ASSERT_EQ(bool(desc.source_is_superblock),source.source_is_superblock);ASSERT_EQ(desc.codebook_id,source.device_execution_codebook_id);
                ASSERT_EQ(desc.n,widths[p]);ASSERT_EQ(desc.k,input_width);ASSERT_EQ(desc.blocks_per_row,input_width/32u);
                ASSERT_EQ(originals[p]->native_type(),source.tensor_type);
            }
        }
        std::printf("ROCM_GDN_VERIFIER_MODE,%s,%s,%d,%d\n",format.label,mode,int(mixed),int(pairing));
        std::printf("ROCM_GDN_VERIFIER_CASE,%s,%s,%d,%d,%d,%d,%d,%d\n",format.label,mode,int(mixed),input_width,widths[0],widths[1],widths[2],widths[3]);
        input=TestTensorFactory::createFP32({16,input_width});
        for(size_t i=0;i<input->numel();++i)input->mutable_data()[i]=float(int((i*29+i/input_width*11)%251)-125)/128.f;
        ASSERT_TRUE(input->ensureOnDevice(device,stream));serial_input=TestTensorFactory::createFP32({1,input_width});ASSERT_TRUE(serial_input->allocateOnDevice(device,stream));
        for(size_t p=0;p<4;++p){serial_outputs[p]=TestTensorFactory::createFP32({1,size_t(widths[p])});ASSERT_TRUE(serial_outputs[p]->allocateOnDevice(device,stream));oracle[p].resize(size_t(16)*widths[p]);}
        const auto params=[&](int owner,int m,FP32Tensor *a,const std::array<std::unique_ptr<FP32Tensor>,4> &out,std::optional<DeviceRowRange> rows) {
            GDNProjectionStage::Params p;p.device_id=device;p.m=m;p.k=input_width;p.input=a;
            p.w_qkv=weights[0].get();p.output_qkv=out[0].get();p.n_qkv=widths[0];p.prepared_ref_qkv=prepared[owner]->refs[0];
            p.w_z=weights[1].get();p.output_z=out[1].get();p.n_z=widths[1];p.prepared_ref_z=prepared[owner]->refs[1];
            p.w_a=weights[2].get();p.output_a=out[2].get();p.n_a=widths[2];p.prepared_ref_a=prepared[owner]->refs[2];
            p.w_b=weights[3].get();p.output_b=out[3].get();p.n_b=widths[3];p.prepared_ref_b=prepared[owner]->refs[3];
            p.prepared_store=prepared[owner]->store.get();p.force_decode_equivalent_verifier_prefill=m>1;p.verifier_row_range=rows;
            p.qkv_operand_precision=precision;
            // SingleQ8 historically accepts unused correction-row metadata.
            // Exercise that compatibility while TwoTerm retains exact rows.
            p.qkv_row_range=corrected?rows.value_or(DeviceRowRange::fullyActive(m)):DeviceRowRange::fullyActive(m);
            return p;
        };
        const auto execute=[&](StageOwner &owner,FP32Tensor *a,const std::array<std::unique_ptr<FP32Tensor>,4> &out) {
            return with_gpu_coherence(device,{a},{out[0].get(),out[1].get(),out[2].get(),out[3].get()},stream,[&]{return owner.stage->execute(context.get());});
        };
        {
            StageOwner serial(params(2,1,serial_input.get(),serial_outputs,std::nullopt),*context,stream);const Drain drain{stream};
            for(int row=0;row<16;++row) {
                ASSERT_EQ(hipMemcpyAsync(serial_input->gpu_data_ptr(),static_cast<const float *>(input->gpu_data_ptr())+row*input_width,input_width*sizeof(float),hipMemcpyDeviceToDevice,stream),hipSuccess);
                TransferEngine::publishCurrentDeviceWrite(serial_input.get(),stream);ASSERT_TRUE(execute(serial,serial_input.get(),serial_outputs));
                for(size_t p=0;p<4;++p)ASSERT_EQ(hipMemcpyAsync(oracle[p].data()+size_t(row)*widths[p],serial_outputs[p]->gpu_data_ptr(),widths[p]*sizeof(float),hipMemcpyDeviceToHost,stream),hipSuccess);
            }
            ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);
        }
        for(const auto &out:oracle){ASSERT_TRUE(std::all_of(out.begin(),out.end(),[](auto word){return std::isfinite(std::bit_cast<float>(word));}));ASSERT_TRUE(std::any_of(out.begin(),out.end(),[](auto word){return word!=0;}));}
        for(int m=profile_rows.value_or(2);m<=profile_rows.value_or(16);++m) {
            SCOPED_TRACE(::testing::Message()<<format.label<<" "<<mode<<" mixed="<<mixed<<" M="<<m);
            auto live=TestTensorFactory::createINT32({1});live->mutable_int32_data()[0]=m;
            std::array<std::array<std::unique_ptr<FP32Tensor>,4>,2> outputs;
            std::array<std::vector<uint32_t>,4> actual;
            std::array<std::unique_ptr<StageOwner>,2> owners;
            std::array<std::unique_ptr<IGPUGraphCapture>,2> fragments,graphs;
            std::array<std::vector<hipGraphNode_t>,2> native_nodes,correction_nodes,correction_publish_nodes,quant_nodes;
            const Drain drain{stream};ASSERT_TRUE(live->ensureOnDevice(device,stream));
            auto *count_owner=static_cast<int32_t *>(live->gpu_data_ptr());const auto rows=DeviceRowRange::deviceCounted(m,count_owner);
            for(int path=0;path<2;++path) {
                for(size_t p=0;p<4;++p){outputs[path][p]=TestTensorFactory::createFP32({size_t(m),size_t(widths[p])});ASSERT_TRUE(outputs[path][p]->allocateOnDevice(device,stream));actual[p].resize(size_t(m)*widths[p]);}
                owners[path]=std::make_unique<StageOwner>(params(path,m,input.get(),outputs[path],rows),*context,stream);
                ScopedCapturedVerifierExtent extent(pairing==PairPolicy::ExactHostedAutomatic
                    ? CapturedVerifierExtent::exactHostedRequest(device,rows):CapturedVerifierExtent{});
                ScopedNativeQuantisedVerifierCandidate candidate(path
                    ? (automatic?NativeQuantisedVerifierCandidate::Automatic:NativeQuantisedVerifierCandidate::WmmaM16N64)
                    : NativeQuantisedVerifierCandidate::Dot4,ordinal);
                ASSERT_TRUE(execute(*owners[path],input.get(),outputs[path]));ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);
                TransferEngine::requireDeviceInput(input.get(),device,stream);TransferEngine::requireDeviceInput(live.get(),device,stream);
                GraphCaptureDependencyLedger::StagePlan plan{.stage_identity=owners[path]->stage.get(),.stage_name="four_source_gdn_verifier",.external_inputs={input.get(),live.get()},
                    .outputs={outputs[path][0].get(),outputs[path][1].get(),outputs[path][2].get(),outputs[path][3].get()}};
                GraphCaptureDependencyLedger ledger(device,stream,std::vector<GraphCaptureDependencyLedger::StagePlan>(repeats,plan),"Complete GDN verifier transaction");
                graphs[path]=worker->createGraphCapture(stream);fragments[path]=graphs[path]->createOrderedTimelineFragment();ASSERT_TRUE(fragments[path]);
                ScopedBackendGraphCapture capture(*worker,*fragments[path],"four original GDN sources",&ledger);ASSERT_TRUE(capture.begin());
                for(int repeat=0;repeat<repeats;++repeat){ScopedGraphCaptureStage stage_scope(owners[path]->stage.get());ASSERT_TRUE(owners[path]->stage->execute(context.get()));stage_scope.complete();}
                capture.finish();
                const std::array<GPUOrderedTimelineStep,1> steps{{{.name=path?"wmma":"dot4",.kind=GPUOrderedTimelineStepKind::CapturedFragment,.capture=fragments[path].get()}}};
                ASSERT_TRUE(graphs[path]->buildOrderedTimelineTransaction(steps,profile_rows?GPUOrderedTimelineInstrumentation::Disabled:GPUOrderedTimelineInstrumentation::PerStepEvents));ASSERT_TRUE(graphs[path]->instantiate());
                auto *parent=dynamic_cast<HIPGraphCapture *>(graphs[path].get());ASSERT_TRUE(parent);
                size_t size=0;ASSERT_EQ(hipGraphGetNodes(parent->graph(),nullptr,&size),hipSuccess);std::vector<hipGraphNode_t> nodes(size);ASSERT_EQ(hipGraphGetNodes(parent->graph(),nodes.data(),&size),hipSuccess);
                std::array<int,2> contributions{};
                std::map<int,std::pair<uintptr_t,uintptr_t>> role_planes;
                const auto *workspace=owners[path]->workspace.get();
                const auto quant=workspace->getBuffer(GemmWorkspaceBuffers::QUANT_A);
                const auto scales=workspace->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE);
                const auto sums=workspace->getBuffer(GemmWorkspaceBuffers::SUMS_A_BLOCKWISE);
                for(auto node:nodes) {
                    hipGraphNodeType type;ASSERT_EQ(hipGraphNodeGetType(node,&type),hipSuccess);if(type!=hipGraphNodeTypeKernel)continue;
                    hipKernelNodeParams launch{};ASSERT_EQ(hipGraphKernelNodeGetParams(node,&launch),hipSuccess);
                    const char *symbol=hipKernelNameRefByPtr(launch.func,stream);ASSERT_TRUE(symbol);const std::string_view name(symbol);
                    if(name.find("quantizeActivationsQ8_blockwise_v4_with_sums_kernel")!=std::string_view::npos) {
                        quant_nodes[path].push_back(node);
                        ASSERT_EQ(*static_cast<const float **>(launch.kernelParams[0]),input->gpu_data_ptr());
                        ASSERT_EQ(*static_cast<int8_t **>(launch.kernelParams[1]),quant);
                        ASSERT_EQ(*static_cast<float **>(launch.kernelParams[2]),scales);
                        ASSERT_EQ(*static_cast<int32_t **>(launch.kernelParams[3]),sums);
                        ASSERT_EQ(*static_cast<int *>(launch.kernelParams[4]),m);
                        ASSERT_EQ(*static_cast<int *>(launch.kernelParams[5]),input_width);
                        ASSERT_EQ(*static_cast<int *>(launch.kernelParams[6]),input_width/32);
                        ASSERT_NE(name.find("ILb1E"),std::string_view::npos);
                        ASSERT_EQ(static_cast<const DeviceRowRange *>(launch.kernelParams[7])->captureIdentity(),rows.captureIdentity());
                        ASSERT_EQ(launch.gridDim.x,unsigned(m*(input_width/32)));
                        ASSERT_EQ(launch.gridDim.y,1u);ASSERT_EQ(launch.gridDim.z,1u);
                        ASSERT_EQ(launch.blockDim.x,32u);ASSERT_EQ(launch.blockDim.y,1u);ASSERT_EQ(launch.blockDim.z,1u);
                    }
                    if(name.find("q8_residual")!=std::string_view::npos) {
                        ASSERT_TRUE(corrected);
                        const bool publication=name.find("publish")!=std::string_view::npos;
                        ASSERT_TRUE(publication || name.find("project")!=std::string_view::npos);
                        (publication?correction_publish_nodes[path]:correction_nodes[path]).push_back(node);
                        const auto &args=*static_cast<const GPUQ8ResidualCorrectionArgs *>(launch.kernelParams[0]);
                        ASSERT_EQ(args.rows.captureIdentity(),rows.captureIdentity());
                        ASSERT_EQ(args.quantizer,Q8ResidualQuantizerPolicy::ROCmFiniteHalfNativeReciprocal);
                        ASSERT_EQ(args.input,input->gpu_data_ptr());ASSERT_EQ(args.output,outputs[path][0]->gpu_data_ptr());
                        const auto tuple=[](const DeviceNativeQuantisedMatrixDesc &d){return std::tie(d.payload,d.scales,d.mins,d.emins,d.n,d.k,d.blocks_per_row,d.codebook_id,
                            d.allocation_payload_bytes_per_block,d.allocation_has_mins,d.allocation_has_emins,d.source_codebook_id,d.source_is_superblock,d.source_identity_present,d.reserved,d.reserved_tail);};
                        ASSERT_EQ(tuple(args.weights),tuple(descriptors[path][0]));
                        const std::array<const void *,4> banks{args.primary,args.residual,args.primaryScales,args.residualScales};
                        const std::array<std::string,4> names{kGPUProjectionPrimaryQ8,kGPUProjectionResidualQ8,kGPUProjectionPrimaryQ8Scales,kGPUProjectionResidualQ8Scales};
                        const std::array<size_t,4> bytes{size_t(m)*input_width,size_t(m)*input_width,size_t(m)*input_width/32*sizeof(float),size_t(m)*input_width/32*sizeof(float)};
                        for(size_t bank=0;bank<4;++bank) {
                            ASSERT_EQ(banks[bank],workspace->getBuffer(names[bank]));ASSERT_GE(workspace->getBufferSize(names[bank]),bytes[bank]);
                            const auto begin=reinterpret_cast<uintptr_t>(banks[bank]),end=begin+bytes[bank];ASSERT_NE(begin,0u);
                            for(size_t previous=0;previous<bank;++previous){const auto other=reinterpret_cast<uintptr_t>(banks[previous]);ASSERT_TRUE(end<=other || begin>=other+bytes[previous]);}
                        }
                    }
                    if(name.find("native_quantised_verifier_wmma_m16n64")==std::string_view::npos)continue;
                    ASSERT_EQ(path,1);native_nodes[path].push_back(node);ASSERT_EQ(launch.blockDim.x,128u);
                    const auto &packet=*static_cast<const NativeQuantisedSmallMBatchedArgs *>(launch.kernelParams[3]);
                    ASSERT_EQ(packet.rows.captureIdentity(),rows.captureIdentity());ASSERT_EQ(*static_cast<int *>(launch.kernelParams[4]),input_width);
                    ASSERT_EQ(launch.gridDim.x,unsigned(packet.total_grid_n));ASSERT_GT(launch.gridDim.y,0u);
                    ASSERT_EQ(*static_cast<const int8_t **>(launch.kernelParams[0]),quant);
                    ASSERT_EQ(*static_cast<const float **>(launch.kernelParams[1]),scales);
                    ASSERT_EQ(*static_cast<const int32_t **>(launch.kernelParams[2]),sums);
                    ASSERT_EQ(*static_cast<int *>(launch.kernelParams[5]),(input_width/32+int(launch.gridDim.y)-1)/int(launch.gridDim.y));
                    const bool partial=name.find("VerifierWmmaPublicationE2E")!=std::string_view::npos;
                    const bool standalone=name.find("VerifierWmmaPublicationE0E")!=std::string_view::npos;
                    ASSERT_TRUE(partial || standalone || name.find("VerifierWmmaPublicationE1E")!=std::string_view::npos);
                    if(!partial)ASSERT_EQ(launch.gridDim.y,1u);
                    if(standalone){ASSERT_TRUE(corrected);ASSERT_EQ(packet.num_projections,1);}
                    if(profile_rows)std::printf("ROCM_GDN_VERIFIER_PROFILE_PACKET,%s,%s,%d,%d,%u,%u,%u,%u\n",
                        format.label,mode,m,packet.num_projections,launch.gridDim.x,launch.gridDim.y,launch.gridDim.z,launch.blockDim.x);
                    std::vector<std::pair<uintptr_t,uintptr_t>> packet_planes;
                    for(int index=0;index<packet.num_projections;++index) {
                        int role=-1;for(int p=0;p<2;++p)if(packet.payload[index]==descriptors[path][p].payload)role=p;
                        ASSERT_GE(role,corrected?1:0);ASSERT_EQ(packet.N[index],widths[role]);ASSERT_EQ(packet.codebook[index],descriptors[path][role].codebook_id);
                        ASSERT_EQ(packet.scales[index],descriptors[path][role].scales);ASSERT_EQ(packet.mins[index],descriptors[path][role].mins);ASSERT_EQ(packet.emins[index],descriptors[path][role].emins);
                        ASSERT_EQ(packet.output[index],outputs[path][role]->gpu_data_ptr());ASSERT_EQ(packet.bias[index],nullptr);++contributions[role];
                        if(partial) {
                            const auto begin=reinterpret_cast<uintptr_t>(packet.partial[index]);
                            const auto end=begin+size_t(launch.gridDim.y)*m*widths[role]*sizeof(float);ASSERT_GT(end,begin);
                            int admitted_banks=0;
                            for(const char *bank:{GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL,GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL_BATCHED}) {
                                const auto base=reinterpret_cast<uintptr_t>(workspace->getBuffer(bank));
                                const auto bytes=workspace->getBufferSize(bank);
                                admitted_banks+=base && begin>=base && end<=base+bytes;
                            }
                            ASSERT_EQ(admitted_banks,1)<<"Native partial escaped its original workspace bank";
                            const auto plane=std::pair{begin,end};const auto [it,inserted]=role_planes.emplace(role,plane);ASSERT_EQ(it->second,plane);
                            // Roles in one packet run concurrently. Sequential
                            // subgroup packets may reuse an admitted bank.
                            for(const auto &[other_begin,other_end]:packet_planes)ASSERT_TRUE(end<=other_begin || begin>=other_end);
                            packet_planes.push_back(plane);
                        }
                    }
                }
                ASSERT_EQ(quant_nodes[path].size(),size_t(repeats));
                ASSERT_EQ(correction_nodes[path].size(),corrected?size_t(repeats):0u);
                ASSERT_EQ(correction_publish_nodes[path].size(),corrected?size_t(repeats):0u);
                if(!path || (automatic && (mixed || pairing==PairPolicy::GuardedAutomatic || m==2)))
                    ASSERT_TRUE(native_nodes[path].empty());
                // Independent strong anchors do not call the selector to
                // manufacture their own expected engine. The timing collector
                // checks every emitted source/mode/M against frozen masks.
                if(path && (!automatic || (!corrected && !mixed && m==16 && pairing==PairPolicy::ExactHostedAutomatic)))
                    ASSERT_FALSE(native_nodes[path].empty());
                if(path && automatic && corrected && !mixed && m==16 && pairing==PairPolicy::ExactHostedAutomatic){
                    if(format.tensor_type==TensorType::Q8_0)ASSERT_FALSE(native_nodes[path].empty());
                    if(format.tensor_type==TensorType::Q4_0)ASSERT_TRUE(native_nodes[path].empty());
                }
                if(!native_nodes[path].empty()){
                    ASSERT_EQ(contributions[0],corrected?0:repeats);ASSERT_EQ(contributions[1],repeats);
                }
                std::vector<GPUGraphKernelNodeInfo> inventory;ASSERT_TRUE(graphs[path]->inspectKernelNodes(inventory));std::set<std::string> printed;
                for(const auto &node:inventory){ASSERT_TRUE(node.valid());ASSERT_EQ(node.local_memory_bytes_per_thread,0u);if(printed.insert(node.name).second)
                    std::printf("ROCM_GDN_VERIFIER_RESOURCE,%s,%s,%d,%d,%d,%u,%zu,%s\n",format.label,mode,int(mixed),m,path,node.registers_per_thread,node.static_shared_memory_bytes,node.name.c_str());}
                std::printf("ROCM_GDN_VERIFIER_ENGINE,%s,%s,%d,%d,%d,%zu,%zu\n",format.label,mode,int(mixed),m,path,native_nodes[path].size(),correction_nodes[path].size());
            }
            // TwoTerm publishes the same primary Q8 representation before Z.
            // Omit every activation publisher together so an earlier valid
            // primary bank cannot conceal the diagnostic missing producer.
            auto activation_nodes=quant_nodes;
            for(int path=0;path<2;++path)activation_nodes[path].insert(activation_nodes[path].end(),correction_publish_nodes[path].begin(),correction_publish_nodes[path].end());
            int32_t published=m;
            const auto poison=[&](int path) {
                for(size_t p=0;p<4;++p)require(hipMemsetAsync(outputs[path][p]->gpu_data_ptr(),0xa5,actual[p].size()*sizeof(uint32_t),stream)==hipSuccess,"GDN output poison failed");
                for(const auto &buffer:owners[path]->requirements.buffers) {
                    if(buffer.content_lifetime!=WorkspaceContentLifetime::ParticipantExecution)continue;
                    const auto bytes=owners[path]->workspace->getBufferSize(buffer.name);
                    if(bytes)require(hipMemsetAsync(owners[path]->workspace->getBuffer(buffer.name),0xff,bytes,stream)==hipSuccess,"GDN scratch poison failed");
                }
            };
            const auto replay=[&](int path) {
                poison(path);require(graphs[path]->launch(),"GDN retained replay failed");
                for(size_t p=0;p<4;++p)require(hipMemcpyAsync(actual[p].data(),outputs[path][p]->gpu_data_ptr(),actual[p].size()*sizeof(uint32_t),hipMemcpyDeviceToHost,stream)==hipSuccess,"GDN observation failed");
                require(hipStreamSynchronize(stream)==hipSuccess,"GDN observation join failed");
            };
            const auto matches_role=[&](size_t p) {
                for(size_t i=0;i<actual[p].size();++i) {
                    const auto expected=i<size_t(published)*widths[p]?oracle[p][i]:0xa5a5a5a5u;
                    if(actual[p][i]!=expected)return ::testing::AssertionFailure()<<"projection="<<p<<" element="<<i<<" live="<<published
                        <<" expected_bits="<<expected<<" actual_bits="<<actual[p][i];
                }
                return ::testing::AssertionSuccess();
            };
            const auto matches=[&] {
                for(size_t p=0;p<4;++p){const auto result=matches_role(p);if(!result)return result;}return ::testing::AssertionSuccess();
            };
            if(profile_rows){
                ASSERT_EQ(native_nodes[1].size(),1u);
                for(int warm=0;warm<10;++warm)ASSERT_TRUE(graphs[1]->launch());
                ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);poison(1);ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);
                {ProfileWindow window;const bool launched=graphs[1]->launch();const bool joined=hipStreamSynchronize(stream)==hipSuccess;
                    window.finish();ASSERT_TRUE(launched && joined);}
                for(size_t p=0;p<4;++p)ASSERT_EQ(hipMemcpyAsync(actual[p].data(),outputs[1][p]->gpu_data_ptr(),actual[p].size()*sizeof(uint32_t),hipMemcpyDeviceToHost,stream),hipSuccess);
                ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);ASSERT_TRUE(matches());
                std::printf("ROCM_GDN_VERIFIER_PROFILE_OBSERVED,%s,%s,%d\n",format.label,mode,m);continue;
            }
            std::vector<int> sequence{m,1,0,m,m};for(int prefix=2;prefix<m;++prefix)sequence.push_back(prefix);
            for(int prefix:sequence) {
                published=prefix;ASSERT_EQ(hipMemcpyAsync(count_owner,&published,sizeof(published),hipMemcpyHostToDevice,stream),hipSuccess);
                for(int path=0;path<2;++path) {
                    replay(path);ASSERT_TRUE(matches())<<"path="<<path<<" live="<<published;
                    if(published==0 || published==m) {
                        auto *parent=dynamic_cast<HIPGraphCapture *>(graphs[path].get());
                        for(const auto *family:{&activation_nodes[path],&native_nodes[path],&correction_nodes[path]})if(!family->empty()){
                            NodeOmission omission(parent->executable(),stream,*family);omission.omit();replay(path);ASSERT_EQ(bool(matches()),published==0);
                            // Disabling one producer must leave independent
                            // outputs byte-correct, including the half controls.
                            ASSERT_TRUE(matches_role(2));ASSERT_TRUE(matches_role(3));
                            if(family==&correction_nodes[path])ASSERT_TRUE(matches_role(1));
                            if(corrected && family==&native_nodes[path])ASSERT_TRUE(matches_role(0));
                            omission.restore();replay(path);ASSERT_TRUE(matches());}
                    }
                    if(!published){owners[path]->stage->resetSessionStatePreservingCapturedReplay();ASSERT_TRUE(owners[path]->stage->prepareGraphLaunch(context.get(),stream));}
                }
            }
            if(timed) {
                published=m;ASSERT_EQ(hipMemcpyAsync(count_owner,&published,sizeof(published),hipMemcpyHostToDevice,stream),hipSuccess);
                for(int warm=0;warm<8;++warm)for(auto &graph:graphs)ASSERT_TRUE(graph->launch());
                for(int sample=0;sample<31;++sample){std::array<double,2> us{};
                    for(int order=0;order<2;++order){const int path=(sample+order)%2;replay(path);ASSERT_TRUE(matches());
                        const auto event=graphs[path]->consumeOrderedTimelineTiming();ASSERT_EQ(event.state,GPUOrderedTimelineTimingState::Complete);ASSERT_EQ(event.samples.size(),1u);
                        us[path]=event.samples[0].elapsed_ms*1000./repeats;ASSERT_TRUE(std::isfinite(us[path]) && us[path]>0);}
                    std::printf("ROCM_GDN_VERIFIER_PAIR,%s,%s,%d,%d,%d,%.9f,%.9f\n",format.label,mode,int(mixed),m,sample,us[0],us[1]);}
            }
        }
    });
}
}
/** @brief Every original source alias has its own complete-operation replay. */
class ROCmGDNVerifierTransaction:public ::testing::TestWithParam<QuantizedVerifierFormatCase> {};
TEST_P(ROCmGDNVerifierTransaction,SingleQ8Replay){prove(GetParam(),QuantizedActivationOperandPrecision::SingleQ8,false);}
TEST_P(ROCmGDNVerifierTransaction,TwoTermQ8Replay){prove(GetParam(),QuantizedActivationOperandPrecision::TwoTermQ8,false);}
TEST_P(ROCmGDNVerifierTransaction,DISABLED_SingleQ8Paired){prove(GetParam(),QuantizedActivationOperandPrecision::SingleQ8,true);}
TEST_P(ROCmGDNVerifierTransaction,DISABLED_TwoTermQ8Paired){prove(GetParam(),QuantizedActivationOperandPrecision::TwoTermQ8,true);}
TEST(ROCmGDNVerifierMixed,IndependentSourcesReplay){for(auto precision:{QuantizedActivationOperandPrecision::SingleQ8,QuantizedActivationOperandPrecision::TwoTermQ8})prove(quantizedMoEVerifierFormats().front(),precision,false,true);}
TEST_P(ROCmGDNVerifierTransaction,SingleQ8AutomaticReplay){prove(GetParam(),QuantizedActivationOperandPrecision::SingleQ8,false,false,PairPolicy::ExactHostedAutomatic);}
TEST_P(ROCmGDNVerifierTransaction,TwoTermQ8AutomaticReplay){prove(GetParam(),QuantizedActivationOperandPrecision::TwoTermQ8,false,false,PairPolicy::ExactHostedAutomatic);}
TEST_P(ROCmGDNVerifierTransaction,DISABLED_SingleQ8AutomaticPaired){prove(GetParam(),QuantizedActivationOperandPrecision::SingleQ8,true,false,PairPolicy::ExactHostedAutomatic);}
TEST_P(ROCmGDNVerifierTransaction,DISABLED_TwoTermQ8AutomaticPaired){prove(GetParam(),QuantizedActivationOperandPrecision::TwoTermQ8,true,false,PairPolicy::ExactHostedAutomatic);}
/** @test Mixed sources and guarded capacity cannot borrow complete hosted timing. */
TEST(ROCmGDNVerifierAutomatic,GuardedAndMixedRemainOrdered){
    for(auto precision:{QuantizedActivationOperandPrecision::SingleQ8,QuantizedActivationOperandPrecision::TwoTermQ8}){
        prove(quantizedMoEVerifierFormats().front(),precision,false,false,PairPolicy::GuardedAutomatic);
        prove(quantizedMoEVerifierFormats().front(),precision,false,true,PairPolicy::ExactHostedAutomatic);
    }
}
/** @test Strong Q8_0 anchors expose the two distinct native packet populations. */
TEST(ROCmGDNVerifierProfile,DISABLED_SingleQ8M16){
    const auto &formats=quantizedMoEVerifierFormats();const auto format=std::find_if(formats.begin(),formats.end(),[](const auto &f){return f.tensor_type==TensorType::Q8_0;});
    ASSERT_NE(format,formats.end());prove(*format,QuantizedActivationOperandPrecision::SingleQ8,false,false,PairPolicy::ExactHostedAutomatic,16);
}
TEST(ROCmGDNVerifierProfile,DISABLED_TwoTermQ8M16){
    const auto &formats=quantizedMoEVerifierFormats();const auto format=std::find_if(formats.begin(),formats.end(),[](const auto &f){return f.tensor_type==TensorType::Q8_0;});
    ASSERT_NE(format,formats.end());prove(*format,QuantizedActivationOperandPrecision::TwoTermQ8,false,false,PairPolicy::ExactHostedAutomatic,16);
}
INSTANTIATE_TEST_SUITE_P(AllSources,ROCmGDNVerifierTransaction,::testing::ValuesIn(quantizedMoEVerifierFormats()),
    [](const ::testing::TestParamInfo<QuantizedVerifierFormatCase> &info){return std::string(info.param.label);});
