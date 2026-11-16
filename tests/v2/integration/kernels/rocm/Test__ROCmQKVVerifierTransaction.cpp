/**
 * @file Test__ROCmQKVVerifierTransaction.cpp
 * @brief Retained actual QKV/KV verifier stages against independent serial stages.
 *
 * Three distinct prepared sources share one store per execution owner. The
 * serial owner has independent kernels, workspace and activation quantization.
 * Counted grouped stages retain their complete topology through empty, partial,
 * repeated and reset replay. Local TP dimensions describe component geometry;
 * they do not admit the model graph's excluded MTP QueryInterval slice mode.
 */
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#include "backends/GPUDeviceContextPool.h"
#include "backends/rocm/HIPGraphCapture.h"
#include "execution/compute_stages/stages/FusedQKVGEMMStage.h"
#include "execution/local_execution/coherence/GpuCoherence.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/rocm/gemm/ROCmNativeQuantisedVerifierLaunch.h"
#include "kernels/rocm/gemm/ROCmNativeQuantisedArchitecture.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "transfer/TransferEngine.h"
#include "utils/PerfStatsCollector.h"
#include "../../../utils/GpuPreparedGemmHarness.h"
#include "../../../utils/QuantizedVerifierFormats.h"
#include "../../../utils/TestTensorFactory.h"
#include <array>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <stdexcept>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::rocm;
using namespace llaminar2::test;
namespace {
/** @brief Source-owned Q/gate and KV widths from the Qwen release geometry catalog. */
struct Geometry { const char *name; int k; std::array<int,3> n; };
constexpr Geometry dense_full{"Qwen36_27B_Full",5120,{12288,1024,1024}};
constexpr Geometry dense_tp2{"Qwen36_27B_LocalTP2",5120,{6144,512,512}};
constexpr Geometry moe_full{"Qwen36_35BA3B_Full",2048,{8192,512,512}};
constexpr Geometry moe_tp4{"Qwen36_35BA3B_LocalTP4_ReplicatedKV",2048,{2048,512,512}};
/** @brief Economy coverage is independent of the always-complete prefix replay. */
enum class TimingRows { None, EveryPrefix, FullExtent };
/** @brief Candidate controls distinguish exact hosted scheduling from ordinary capacity. */
enum class PairPolicy { ForcedWmma, GuardedAutomatic, ExactHostedAutomatic };
/** @brief Independent source/bias controls do not inherit homogeneous admission. */
enum class WeightSources { Homogeneous, Mixed, HomogeneousWithBias, MixedWithBias };
/** @brief Unknown complete bundles cannot borrow a measured scalar member. */
enum class AutomaticDomain { MeasuredBundle, UnmeasuredBundle };
/** @brief One explicit physical width for an isolated native-counter observation. */
struct ProfileSelection { int rows; };

/** @brief Keep oracle, setup and observations outside the exact captured profiler region. */
class ProfileWindow final {
    bool active_=false;
public:
    /** @brief Resume only after the caller has drained setup and poisoned outputs. */
    ProfileWindow() {
        if(roctxRangePushA("ROCmQKVVerifierTransaction")<0)throw std::runtime_error("QKV profiler range failed");
        if(roctxProfilerResume(0)!=0) {(void)roctxRangePop();throw std::runtime_error("QKV profiler resume failed");}
        active_=true;
    }
    /** @brief Pause before output observation; cleanup remains unconditional on exceptions. */
    void finish() {
        if(!active_)return;
        const int paused=roctxProfilerPause(0),popped=roctxRangePop();active_=false;
        if(paused!=0 || popped<0)throw std::runtime_error("QKV profiler pause failed");
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

/** @brief Fail fixture setup immediately while retaining normal C++ cleanup. */
void require(bool ok, const char *what) { if (!ok) throw std::runtime_error(what); }
/** @brief Join the exact stream before graphs, tensors or prepared owners retire. */
struct Drain {
    hipStream_t stream;
    /** @brief Retire all submitted work on success and assertion unwinding. */
    ~Drain() { if (hipStreamSynchronize(stream)!=hipSuccess) std::terminate(); }
};
/** @brief Select only submission evidence; restore the caller's collector policy. */
class SubmissionScope {
    std::array<std::optional<std::string>,2> old_;
    static constexpr std::array<const char *,2> keys{"LLAMINAR_PERF_STATS_JSON","LLAMINAR_PERF_STATS_FILTER"};
    /** @brief Restore the exact prior presence/value of both process-owned options. */
    void restore() noexcept {
        for (size_t i=0;i<keys.size();++i)
            if (old_[i] ? setenv(keys[i],old_[i]->c_str(),1) : unsetenv(keys[i])) std::terminate();
    }
public:
    /** @brief Admit only the two counters used to authenticate captured submissions. */
    SubmissionScope() {
        for (size_t i=0;i<keys.size();++i) if (const char *v=std::getenv(keys[i])) old_[i]=v;
        if (setenv(keys[0],"1",1) || setenv(keys[1],
                "kernel.rocm_fused_small_m_shared_quant_calls,kernel.rocm_native_quantised_verifier_wmma_submission",1)) {
            restore();throw std::runtime_error("Cannot enable QKV submission evidence");
        }
        PerfStatsCollector::reloadConfigurationFromEnvironment(); PerfStatsCollector::reset();
    }
    /** @brief Retire observations before restoring the caller's collection policy. */
    ~SubmissionScope() {
        PerfStatsCollector::reset();
        restore();
        PerfStatsCollector::reloadConfigurationFromEnvironment();
    }
    SubmissionScope(const SubmissionScope &)=delete;
    SubmissionScope &operator=(const SubmissionScope &)=delete;
};
/** @return Host submission totals; neither counter may advance during device replay. */
std::array<double,2> submissions(int projections,int k) {
    std::array<double,2> result{};
    for (const auto &r:PerfStatsCollector::snapshot({"kernel.rocm_fused_small_m_shared_quant_calls",
            "kernel.rocm_native_quantised_verifier_wmma_submission"})) {
        if(r.name=="rocm_fused_small_m_shared_quant_calls")
            require(r.tags.at("projections")==std::to_string(projections) && r.tags.at("k")==std::to_string(k),
                "QKV counter names a different transaction");
        result[r.name=="rocm_fused_small_m_shared_quant_calls"?0:1]+=r.value;
    }
    return result;
}
/** @brief Own stage bindings independently from each candidate's prepared kernels. */
struct StageOwner {
    std::unique_ptr<DeviceWorkspaceManager> workspace;
    std::unique_ptr<FusedQKVGEMMStage> stage;
    hipStream_t stream;
    /** @param params Complete stage operation. @param context Exact backend participant.
     * @param exact Nondefault stream retained through every use of the binding. */
    StageOwner(FusedQKVGEMMStage::Params params, IDeviceContext &context, hipStream_t exact):stream(exact) {
        const int m=params.m,k=params.k; const auto device=params.device_id;
        stage=std::make_unique<FusedQKVGEMMStage>(std::move(params));
        const auto requirements=stage->getWorkspaceRequirements(m,0,k);
        try {
            workspace=std::make_unique<DeviceWorkspaceManager>(device,requirements.total_bytes_with_alignment());
            require(workspace->allocate(requirements),"QKV workspace allocation failed");
            stage->bindWorkspace(workspace.get());
            require(stage->prepareGraphLaunch(&context,stream),"QKV stage preparation failed");
        } catch (...) {
            if(hipStreamSynchronize(stream)!=hipSuccess)std::terminate();
            stage->unbindWorkspace();throw;
        }
    }
    /** @brief Unbind prepared kernels before their persistent workspace retires. */
    ~StageOwner() {
        if (hipStreamSynchronize(stream)!=hipSuccess) std::terminate();
        stage->unbindWorkspace();
    }
};
/** @brief Temporarily remove real imported graph nodes, restoring them on every exit. */
class NodeOmission {
    hipGraphExec_t executable_; hipStream_t stream_; const std::vector<hipGraphNode_t> &nodes_; bool applied_=false;
public:
    /** @param executable Instantiated retained parent. @param stream Exact recording stream.
     * @param nodes Actual parent-node family borrowed until restoration. */
    NodeOmission(hipGraphExec_t executable,hipStream_t stream,const std::vector<hipGraphNode_t> &nodes)
        :executable_(executable),stream_(stream),nodes_(nodes) {}
    /** @brief Disable the selected family without changing packets or topology. */
    void omit() {
        require(!applied_ && !nodes_.empty(),"Invalid QKV node omission"); applied_=true;
        for (auto node:nodes_) require(hipGraphNodeSetEnabled(executable_,node,0)==hipSuccess,"QKV omission failed");
    }
    /** @brief Join pending observations and restore every node, including partial omission. */
    void restore() noexcept {
        if (!applied_) return;
        if (hipStreamSynchronize(stream_)!=hipSuccess) std::terminate();
        for (auto node:nodes_) if (hipGraphNodeSetEnabled(executable_,node,1)!=hipSuccess) std::terminate();
        applied_=false;
    }
    /** @brief Never leave a fault installed after an assertion unwinds the fixture. */
    ~NodeOmission() { restore(); }
    NodeOmission(const NodeOmission &)=delete;
    NodeOmission &operator=(const NodeOmission &)=delete;
};

/** @brief Execute both arithmetic candidates through the actual fused stage.
 * @param format Complete source-format identity. @param geometry Model-owned local dimensions.
 * @param set Exact QKV or KV-only transaction. @param timing Which live extents receive31 paired samples.
 * @param sources Distinct Q/K/V codebooks with learned biases, or homogeneous originals.
 * @param pairing Explicit native control or actual Automatic with its original extent contract.
 * @param domain Measured complete operation or an unknown bundle containing a measured scalar member.
 * @param profile Isolate one native width for counters, with timing and oracle outside the region. */
void prove(const QuantizedVerifierFormatCase &format,Geometry geometry,AttentionProjectionSet set,TimingRows timing,
    WeightSources sources=WeightSources::Homogeneous,PairPolicy pairing=PairPolicy::ForcedWmma,
    AutomaticDomain domain=AutomaticDomain::MeasuredBundle,std::optional<ProfileSelection> profile=std::nullopt) {
    int count=0;
    if (hipGetDeviceCount(&count)!=hipSuccess || !count) GTEST_SKIP()<<"No ROCm device";
    const auto device=DeviceId::rocm(0);
    auto worker=GPUDeviceContextPool::instance().retainAMDContext(0);
    worker->submitAndWait([&] {
        if (nativeQuantisedHardwareForDevice(0).architecture!=native_quantised::NativeQuantisedArchitecture::rocm(11,0,0))
            GTEST_SKIP()<<"Native gfx1100 required";
        const auto stream=static_cast<hipStream_t>(worker->defaultStream());
        const bool qkv=set==AttentionProjectionSet::QueryKeyValue, timed=timing!=TimingRows::None;
        const bool mixed=sources==WeightSources::Mixed || sources==WeightSources::MixedWithBias;
        const bool biased=sources==WeightSources::HomogeneousWithBias || sources==WeightSources::MixedWithBias;
        const bool automatic=pairing!=PairPolicy::ForcedWmma;
        if(profile) {
            ASSERT_GE(profile->rows,2);ASSERT_LE(profile->rows,16);
            ASSERT_EQ(pairing,PairPolicy::ExactHostedAutomatic);ASSERT_FALSE(timed);
            ASSERT_EQ(sources,WeightSources::Homogeneous);ASSERT_EQ(domain,AutomaticDomain::MeasuredBundle);
        }
        const int first=qkv?0:1, projections=qkv?3:2, repeats=timed?8:1;
        const SubmissionScope observation;
        auto context=IDeviceContext::create(device); ASSERT_TRUE(context);
        std::array<std::shared_ptr<TensorBase>,3> weights;
        std::array<std::unique_ptr<FP32Tensor>,3> bias;
        std::array<uint8_t,3> source_codes{};
        std::array<GpuPreparedQKVFixture,3> prepared;
        std::array<std::array<DeviceNativeQuantisedMatrixDesc,3>,2> descriptors;
        std::unique_ptr<FP32Tensor> input,serial_input;
        std::array<std::unique_ptr<FP32Tensor>,3> serial_outputs;
        std::array<std::vector<uint32_t>,3> oracle;
        // All outer owners predate this guard: failed setup joins their exact
        // stream before any tensor, prepared handle or host observation retires.
        const Drain setup_drain{stream};
        const char *source_label=mixed?(biased?"IQ2S_IQ4XS_Q80_Bias":"IQ2S_IQ4XS_Q80")
            :(biased?"Q4_0_Bias":format.label);
        for (int p=0;p<3;++p) {
            const auto *source=&format;
            if (mixed) {
                source=nullptr;
                const auto type=std::array{TensorType::IQ2_S,TensorType::IQ4_XS,TensorType::Q8_0}[p];
                for (const auto &f:quantizedMoEVerifierFormats()) if (f.tensor_type==type) source=&f;
            }
            ASSERT_NE(source,nullptr);source_codes[p]=source->source_codebook_id;
            weights[p]=source->create({size_t(geometry.n[p]),size_t(geometry.k)},88101+p);
            bias[p]=TestTensorFactory::createFP32({size_t(geometry.n[p])});
            for (int col=0;col<geometry.n[p];++col) bias[p]->mutable_data()[col]=float((col*7+p)%17-8)/128.0f;
            ASSERT_TRUE(bias[p]->ensureOnDevice(device,stream));
        }
        std::printf("ROCM_QKV_VERIFIER_CASE,%s,%s,%d,%d,%d,%d,%u,%u,%u,%d,%d\n",source_label,geometry.name,
            geometry.k,geometry.n[0],geometry.n[1],geometry.n[2],unsigned(source_codes[0]),unsigned(source_codes[1]),unsigned(source_codes[2]),projections,int(biased));
        std::printf("ROCM_QKV_VERIFIER_MODE,%s,%s,%d,%d\n",source_label,geometry.name,projections,int(pairing));
        // Each candidate and the serial oracle own separate mutable kernel state.
        // Only immutable original tensors are shared across their preparations.
        for (int owner=0;owner<3;++owner)
            prepared[owner]=makeGpuPreparedQKVFixture(weights[0].get(),weights[1].get(),weights[2].get(),device,
                "test.qkv_verifier."+std::to_string(owner),ModelContextId{uint64_t(88101+owner)});
        for (int owner=0;owner<2;++owner) {
            const std::array kernels{prepared[owner].q_kernel,prepared[owner].k_kernel,prepared[owner].v_kernel};
            for (int p=0;p<3;++p) {
                auto *kernel=dynamic_cast<ROCmQuantisedGemmKernel *>(kernels[p]); ASSERT_TRUE(kernel);
                ASSERT_TRUE(kernel->exportNativeQuantisedMatrixDesc(descriptors[owner][p]));
            }
        }
        input=TestTensorFactory::createFP32({16,size_t(geometry.k)});
        for (size_t i=0;i<input->numel();++i) input->mutable_data()[i]=float(int((i*29+i/geometry.k*11)%251)-125)/128.0f;
        ASSERT_TRUE(input->ensureOnDevice(device,stream));
        serial_input=TestTensorFactory::createFP32({1,size_t(geometry.k)});
        ASSERT_TRUE(serial_input->allocateOnDevice(device,stream));
        for (int p=0;p<3;++p) {
            serial_outputs[p]=TestTensorFactory::createFP32({1,size_t(geometry.n[p])});
            ASSERT_TRUE(serial_outputs[p]->allocateOnDevice(device,stream));
            oracle[p].resize(size_t(16)*geometry.n[p]);
        }
        const auto params=[&](int owner,int m,FP32Tensor *a,const std::array<std::unique_ptr<FP32Tensor>,3> &out,
                std::optional<DeviceRowRange> rows) {
            FusedQKVGEMMStage::Params p;
            p.device_id=device;p.projection_set=set;p.input=a;p.m=m;p.k=geometry.k;
            if (qkv) {p.wq=weights[0].get();p.output_q=out[0].get();p.n_q=geometry.n[0];p.prepared_ref_q=prepared[owner].q_ref;
                if (biased) p.bias_q=bias[0].get();}
            p.wk=weights[1].get();p.output_k=out[1].get();p.n_k=geometry.n[1];p.prepared_ref_k=prepared[owner].k_ref;
            p.wv=weights[2].get();p.output_v=out[2].get();p.n_v=geometry.n[2];p.prepared_ref_v=prepared[owner].v_ref;
            if (biased) {p.bias_k=bias[1].get();p.bias_v=bias[2].get();}
            p.force_decode_equivalent_verifier_prefill=true;p.verifier_row_range=rows;p.prepared_store=prepared[owner].store.get();return p;
        };
        const auto execute=[&](StageOwner &owner,FP32Tensor *a,const std::array<std::unique_ptr<FP32Tensor>,3> &out) {
            if (qkv) return with_gpu_coherence(device,{a},{out[0].get(),out[1].get(),out[2].get()},stream,[&]{return owner.stage->execute(context.get());});
            return with_gpu_coherence(device,{a},{out[1].get(),out[2].get()},stream,[&]{return owner.stage->execute(context.get());});
        };
        {
            StageOwner serial(params(2,1,serial_input.get(),serial_outputs,std::nullopt),*context,stream);
            const Drain drain{stream};
            for (int row=0;row<16;++row) {
                ASSERT_EQ(hipMemcpyAsync(serial_input->gpu_data_ptr(),static_cast<const float *>(input->gpu_data_ptr())+row*geometry.k,
                    size_t(geometry.k)*sizeof(float),hipMemcpyDeviceToDevice,stream),hipSuccess);
                TransferEngine::publishCurrentDeviceWrite(serial_input.get(),stream);
                ASSERT_TRUE(execute(serial,serial_input.get(),serial_outputs));
                for (int p=first;p<3;++p) ASSERT_EQ(hipMemcpyAsync(oracle[p].data()+size_t(row)*geometry.n[p],serial_outputs[p]->gpu_data_ptr(),
                    size_t(geometry.n[p])*sizeof(float),hipMemcpyDeviceToHost,stream),hipSuccess);
            }
            ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);
        }
        for (int p=first;p<3;++p) {
            ASSERT_TRUE(std::all_of(oracle[p].begin(),oracle[p].end(),[](auto w){return std::isfinite(std::bit_cast<float>(w));}));
            ASSERT_TRUE(std::any_of(oracle[p].begin(),oracle[p].end(),[](auto w){return w!=0;}));
        }
        for (int m=profile?profile->rows:2;m<=(profile?profile->rows:16);++m) {
            SCOPED_TRACE(::testing::Message()<<source_label<<" "<<geometry.name<<" projections="<<projections<<" M="<<m);
            auto live=TestTensorFactory::createINT32({1});live->mutable_int32_data()[0]=m;
            std::array<std::array<std::unique_ptr<FP32Tensor>,3>,2> outputs;
            std::array<std::vector<uint32_t>,3> actual;
            std::array<std::unique_ptr<StageOwner>,2> owners;
            std::array<std::unique_ptr<IGPUGraphCapture>,2> fragments,graphs;
            std::array<std::vector<hipGraphNode_t>,2> quant_nodes,native_nodes;
            const Drain drain{stream};
            ASSERT_TRUE(live->ensureOnDevice(device,stream));
            auto *count_owner=static_cast<int32_t *>(live->gpu_data_ptr());
            const auto rows=DeviceRowRange::deviceCounted(m,count_owner);
            for (int path=0;path<2;++path) {
                for (int p=0;p<3;++p) {outputs[path][p]=TestTensorFactory::createFP32({size_t(m),size_t(geometry.n[p])});
                    ASSERT_TRUE(outputs[path][p]->allocateOnDevice(device,stream));actual[p].resize(size_t(m)*geometry.n[p]);}
                owners[path]=std::make_unique<StageOwner>(params(path,m,input.get(),outputs[path],rows),*context,stream);
                ScopedCapturedVerifierExtent extent(pairing==PairPolicy::ExactHostedAutomatic
                    ? CapturedVerifierExtent::exactHostedRequest(device,rows) : CapturedVerifierExtent{});
                ScopedNativeQuantisedVerifierCandidate candidate(path
                    ? (automatic?NativeQuantisedVerifierCandidate::Automatic:NativeQuantisedVerifierCandidate::WmmaM16N64)
                    : NativeQuantisedVerifierCandidate::Dot4,0);
                ASSERT_TRUE(execute(*owners[path],input.get(),outputs[path]));ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);
                TransferEngine::requireDeviceInput(input.get(),device,stream);
                TransferEngine::requireDeviceInput(live.get(),device,stream);
                GraphCaptureDependencyLedger::StagePlan plan{
                    .stage_identity=owners[path]->stage.get(),.stage_name="counted_qkv_verifier",
                    .external_inputs={input.get(),live.get()}};
                for(int p=first;p<3;++p) {
                    plan.outputs.push_back(outputs[path][p].get());
                    if(biased) {
                        TransferEngine::requireDeviceInput(bias[p].get(),device,stream);
                        plan.external_inputs.push_back(bias[p].get());
                    }
                }
                // Each repeated operation is an explicit stage in the same
                // capture plan. Preparation stays outside the retained graph.
                GraphCaptureDependencyLedger ledger(device,stream,
                    std::vector<GraphCaptureDependencyLedger::StagePlan>(repeats,plan),"QKV complete-operation replay");
                const auto before=submissions(projections,geometry.k);
                graphs[path]=worker->createGraphCapture(stream);fragments[path]=graphs[path]->createOrderedTimelineFragment();ASSERT_TRUE(fragments[path]);
                ScopedBackendGraphCapture capture(*worker,*fragments[path],"actual counted QKV verifier stage",&ledger);ASSERT_TRUE(capture.begin());
                for (int repeat=0;repeat<repeats;++repeat) {
                    ScopedGraphCaptureStage stage_scope(owners[path]->stage.get());
                    ASSERT_TRUE(owners[path]->stage->execute(context.get()));stage_scope.complete();
                }
                capture.finish();
                const std::array<GPUOrderedTimelineStep,1> steps{{{.name=path?(automatic?"automatic":"wmma"):"dot4",.kind=GPUOrderedTimelineStepKind::CapturedFragment,.capture=fragments[path].get()}}};
                ASSERT_TRUE(graphs[path]->buildOrderedTimelineTransaction(steps,profile
                    ?GPUOrderedTimelineInstrumentation::Disabled:GPUOrderedTimelineInstrumentation::PerStepEvents));ASSERT_TRUE(graphs[path]->instantiate());
                auto *parent=dynamic_cast<HIPGraphCapture *>(graphs[path].get());ASSERT_TRUE(parent);
                size_t size=0;ASSERT_EQ(hipGraphGetNodes(parent->graph(),nullptr,&size),hipSuccess);
                std::vector<hipGraphNode_t> nodes(size);ASSERT_EQ(hipGraphGetNodes(parent->graph(),nodes.data(),&size),hipSuccess);
                std::array<int,3> contributions{};
                std::map<int,std::pair<uintptr_t,uintptr_t>> partial_planes;
                const auto *workspace=owners[path]->workspace.get();
                const auto quant=workspace->getBuffer(GemmWorkspaceBuffers::QUANT_A);
                const auto scales=workspace->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE);
                const auto sums=workspace->getBuffer(GemmWorkspaceBuffers::SUMS_A_BLOCKWISE);
                for (auto node:nodes) {
                    hipGraphNodeType type;ASSERT_EQ(hipGraphNodeGetType(node,&type),hipSuccess);if (type!=hipGraphNodeTypeKernel)continue;
                    hipKernelNodeParams launch{};ASSERT_EQ(hipGraphKernelNodeGetParams(node,&launch),hipSuccess);
                    const char *symbol=hipKernelNameRefByPtr(launch.func,stream);ASSERT_TRUE(symbol);
                    const std::string_view name(symbol);
                    if (name.find("quantizeActivationsQ8_blockwise_v4_with_sums_kernel")!=std::string_view::npos) {
                        quant_nodes[path].push_back(node);
                        ASSERT_EQ(*static_cast<const float **>(launch.kernelParams[0]),input->gpu_data_ptr());
                        ASSERT_EQ(*static_cast<int8_t **>(launch.kernelParams[1]),quant);
                        ASSERT_EQ(*static_cast<float **>(launch.kernelParams[2]),scales);
                        ASSERT_EQ(*static_cast<int32_t **>(launch.kernelParams[3]),sums);
                        ASSERT_EQ(*static_cast<int *>(launch.kernelParams[4]),m);
                        ASSERT_EQ(*static_cast<int *>(launch.kernelParams[5]),geometry.k);
                        ASSERT_EQ(*static_cast<int *>(launch.kernelParams[6]),geometry.k/32);
                        ASSERT_NE(name.find("ILb1E"),std::string_view::npos);
                        ASSERT_EQ(static_cast<const DeviceRowRange *>(launch.kernelParams[7])->captureIdentity(),rows.captureIdentity());
                        ASSERT_EQ(launch.gridDim.x,unsigned(m*(geometry.k/32)));
                        ASSERT_EQ(launch.gridDim.y,1u);ASSERT_EQ(launch.gridDim.z,1u);
                        ASSERT_EQ(launch.blockDim.x,32u);
                        ASSERT_EQ(launch.blockDim.y,1u);ASSERT_EQ(launch.blockDim.z,1u);
                    }
                    if (name.find("native_quantised_verifier_wmma_m16n64")==std::string_view::npos)continue;
                    ASSERT_EQ(path,1);native_nodes[path].push_back(node);ASSERT_EQ(launch.blockDim.x,128u);
                    const auto &packet=*static_cast<const NativeQuantisedSmallMBatchedArgs *>(launch.kernelParams[3]);
                    ASSERT_EQ(packet.rows.captureIdentity(),rows.captureIdentity());ASSERT_EQ(*static_cast<int *>(launch.kernelParams[4]),geometry.k);
                    ASSERT_EQ(*static_cast<const int8_t **>(launch.kernelParams[0]),quant);
                    ASSERT_EQ(*static_cast<const float **>(launch.kernelParams[1]),scales);
                    ASSERT_EQ(*static_cast<const int32_t **>(launch.kernelParams[2]),sums);
                    ASSERT_GT(launch.gridDim.y,0u);ASSERT_EQ(launch.gridDim.x,unsigned(packet.total_grid_n));
                    ASSERT_EQ(*static_cast<int *>(launch.kernelParams[5]),(geometry.k/32+int(launch.gridDim.y)-1)/int(launch.gridDim.y));
                    const bool partial=name.find("VerifierWmmaPublicationE2E")!=std::string_view::npos;
                    ASSERT_TRUE(partial || name.find("VerifierWmmaPublicationE1E")!=std::string_view::npos);
                    if(!partial)ASSERT_EQ(launch.gridDim.y,1u);
                    unsigned role_mask=0;
                    for (int index=0;index<packet.num_projections;++index) {
                        int role=-1;for (int p=first;p<3;++p)if(packet.payload[index]==descriptors[path][p].payload)role=p;
                        ASSERT_GE(role,first);ASSERT_EQ(packet.N[index],geometry.n[role]);ASSERT_EQ(packet.codebook[index],descriptors[path][role].codebook_id);
                        role_mask|=1u<<role;
                        ASSERT_EQ(packet.scales[index],descriptors[path][role].scales);
                        ASSERT_EQ(packet.mins[index],descriptors[path][role].mins);
                        ASSERT_EQ(packet.emins[index],descriptors[path][role].emins);
                        ASSERT_EQ(packet.output[index],outputs[path][role]->gpu_data_ptr());
                        ASSERT_EQ(packet.bias[index],biased?bias[role]->gpu_data_ptr():nullptr);++contributions[role];
                        if(partial) {
                            const auto base=reinterpret_cast<uintptr_t>(workspace->getBuffer(GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL_BATCHED));
                            const auto bytes=workspace->getBufferSize(GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL_BATCHED);
                            const auto begin=reinterpret_cast<uintptr_t>(packet.partial[index]);
                            const auto end=begin+size_t(launch.gridDim.y)*m*geometry.n[role]*sizeof(float);
                            ASSERT_NE(base,0u);ASSERT_GE(begin,base);ASSERT_LE(end,base+bytes);ASSERT_GT(end,begin);
                            const auto plane=std::pair{begin,end};
                            const auto [it,inserted]=partial_planes.emplace(role,plane);ASSERT_EQ(it->second,plane);
                            if(inserted)for(const auto &[other,span]:partial_planes)if(other!=role)
                                ASSERT_TRUE(end<=span.first || begin>=span.second)<<"Projection scratch planes overlap";
                        }
                    }
                    if(profile)std::printf("ROCM_QKV_VERIFIER_PROFILE_PACKET,%s,%s,%d,%d,%u,%u,%u,%u,%u\n",
                        source_label,geometry.name,projections,m,role_mask,launch.gridDim.x,launch.gridDim.y,launch.gridDim.z,launch.blockDim.x);
                }
                ASSERT_EQ(quant_nodes[path].size(),size_t(repeats));
                const bool actual_native=!native_nodes[path].empty();
                if (path && (!automatic || actual_native))
                    for(int p=0;p<3;++p)ASSERT_EQ(contributions[p],p>=first?repeats:0);
                if (automatic) {
                    const bool excluded=mixed || biased || pairing==PairPolicy::GuardedAutomatic ||
                        domain==AutomaticDomain::UnmeasuredBundle;
                    if(excluded)ASSERT_FALSE(actual_native);
                    else if(path && format.tensor_type==TensorType::Q4_0) {
                        // Independent anchors from the frozen complete-operation
                        // panel: every Q4_0 geometry wins at16 and loses at2.
                        // An accidentally all-Dot4 implementation must fail the
                        // registered replay even without the external collector.
                        if(m==16)ASSERT_TRUE(actual_native);
                        if(m==2)ASSERT_FALSE(actual_native);
                    }
                    // Actual Automatic measurements reject these local QKV
                    // crossovers even though the first forced panel won.
                    if(qkv && geometry.k==moe_tp4.k && geometry.n==moe_tp4.n &&
                        ((m==7 && (format.tensor_type==TensorType::Q4_1 || format.tensor_type==TensorType::Q4_K || format.tensor_type==TensorType::IQ2_XS)) ||
                         (m==8 && format.tensor_type==TensorType::Q3_K) ||
                         (m==9 && (format.tensor_type==TensorType::Q2_K || format.tensor_type==TensorType::Q4_K || format.tensor_type==TensorType::IQ1_S))))ASSERT_FALSE(actual_native);
                }
                // The external collector joins this actual packet inventory to
                // frozen measured keys; it does not call the production selector
                // to manufacture its expectation of Automatic's chosen engine.
                std::printf("ROCM_QKV_VERIFIER_ENGINE,%s,%s,%d,%d,%d,%d,%zu\n",
                    source_label,geometry.name,projections,m,path,int(actual_native),native_nodes[path].size());
                const auto after=submissions(projections,geometry.k);ASSERT_EQ(after[0]-before[0],repeats);ASSERT_EQ(after[1]-before[1],native_nodes[path].size());
                std::vector<GPUGraphKernelNodeInfo> inventory;ASSERT_TRUE(graphs[path]->inspectKernelNodes(inventory));
                std::set<std::string> printed;
                for(const auto &node:inventory) {ASSERT_TRUE(node.valid());ASSERT_EQ(node.local_memory_bytes_per_thread,0u);
                    if(printed.insert(node.name).second) std::printf("ROCM_QKV_VERIFIER_RESOURCE,%s,%s,%d,%d,%d,%u,%zu,%s\n",source_label,geometry.name,projections,m,path,node.registers_per_thread,node.static_shared_memory_bytes,node.name.c_str());}
            }
            const auto captured=submissions(projections,geometry.k);int32_t published=m;
            const auto poison=[&](int path) {
                for(int p=0;p<3;++p) require(hipMemsetAsync(outputs[path][p]->gpu_data_ptr(),0xa5,actual[p].size()*sizeof(uint32_t),stream)==hipSuccess,"QKV output poison failed");
                for(const char *name:{GemmWorkspaceBuffers::QUANT_A,GemmWorkspaceBuffers::SCALES_A_BLOCKWISE,GemmWorkspaceBuffers::SUMS_A_BLOCKWISE,
                        GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL,GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL_BATCHED}) {
                    const auto bytes=owners[path]->workspace->getBufferSize(name);if(bytes)require(hipMemsetAsync(owners[path]->workspace->getBuffer(name),0xff,bytes,stream)==hipSuccess,"QKV scratch poison failed");}
            };
            const auto observe=[&](int path) {
                for(int p=0;p<3;++p)require(hipMemcpyAsync(actual[p].data(),outputs[path][p]->gpu_data_ptr(),actual[p].size()*sizeof(uint32_t),hipMemcpyDeviceToHost,stream)==hipSuccess,"QKV observation failed");
                require(hipStreamSynchronize(stream)==hipSuccess,"QKV observation completion failed");
            };
            const auto replay=[&](int path) {
                poison(path);require(graphs[path]->launch(),"QKV retained launch failed");observe(path);
            };
            const auto matches=[&] {
                for(int p=0;p<3;++p)for(size_t i=0;i<actual[p].size();++i)
                    if(actual[p][i]!=(p>=first && i<size_t(published)*geometry.n[p]?oracle[p][i]:0xa5a5a5a5u))return false;
                return true;
            };
            if(profile) {
                ASSERT_FALSE(native_nodes[1].empty());
                for(int warm=0;warm<10;++warm)ASSERT_TRUE(graphs[1]->launch());
                ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);
                poison(1);ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);
                {
                    ProfileWindow window;
                    const bool launched=graphs[1]->launch();
                    const bool joined=hipStreamSynchronize(stream)==hipSuccess;
                    window.finish();ASSERT_TRUE(launched && joined);
                }
                observe(1);ASSERT_TRUE(matches());
                ASSERT_EQ(submissions(projections,geometry.k),captured);
                std::printf("ROCM_QKV_VERIFIER_PROFILE_OBSERVED,%s,%s,%d,%d\n",source_label,geometry.name,projections,m);
                continue;
            }
            std::vector<int> sequence{m,1,0,m,m};for(int live_rows=2;live_rows<m;++live_rows)sequence.push_back(live_rows);
            for(int live_rows:sequence) {
                published=live_rows;ASSERT_EQ(hipMemcpyAsync(count_owner,&published,sizeof(published),hipMemcpyHostToDevice,stream),hipSuccess);
                for(int path=0;path<2;++path) {
                    replay(path);ASSERT_TRUE(matches())<<"live="<<published;
                    auto *parent=dynamic_cast<HIPGraphCapture *>(graphs[path].get());
                    for(const auto *family:{&quant_nodes[path],&native_nodes[path]})if(!family->empty()) {
                        NodeOmission omission(parent->executable(),stream,*family);omission.omit();replay(path);ASSERT_EQ(matches(),published==0);
                        omission.restore();replay(path);ASSERT_TRUE(matches());}
                    if(!published) {owners[path]->stage->resetSessionStatePreservingCapturedReplay();ASSERT_TRUE(owners[path]->stage->prepareGraphLaunch(context.get(),stream));}
                }
            }
            // FullExtent supplies all M2..16 candidate keys for the existing
            // exact-hosted contract. Prefix safety was still proved above;
            // EveryPrefix separately measures an arbitrary retained capacity.
            if(timed)for(int live_rows=timing==TimingRows::FullExtent?m:0;live_rows<=m;++live_rows) {
                published=live_rows;ASSERT_EQ(hipMemcpyAsync(count_owner,&published,sizeof(published),hipMemcpyHostToDevice,stream),hipSuccess);
                for(int warm=0;warm<8;++warm)for(auto &graph:graphs)ASSERT_TRUE(graph->launch());
                for(int sample=0;sample<31;++sample) {
                    std::array<double,2> us{};
                    for(int order=0;order<2;++order) {const int path=(sample+order)%2;replay(path);ASSERT_TRUE(matches());
                        const auto event=graphs[path]->consumeOrderedTimelineTiming();ASSERT_EQ(event.state,GPUOrderedTimelineTimingState::Complete);ASSERT_EQ(event.samples.size(),1u);
                        us[path]=event.samples[0].elapsed_ms*1000.0/repeats;ASSERT_TRUE(std::isfinite(us[path]) && us[path]>0);}
                    std::printf("ROCM_QKV_VERIFIER_PAIR,%s,%s,%d,%d,%d,%d,%.9f,%.9f\n",source_label,geometry.name,projections,m,live_rows,sample,us[0],us[1]);
                }
            }
            ASSERT_EQ(submissions(projections,geometry.k),captured);
        }
    });
}
}

/** @brief All source aliases keep their independent prepared arithmetic identity. */
class ROCmQKVVerifierTransaction:public ::testing::TestWithParam<QuantizedVerifierFormatCase> {};
TEST_P(ROCmQKVVerifierTransaction,DenseFullReplay) {for(auto set:{AttentionProjectionSet::QueryKeyValue,AttentionProjectionSet::KeyValueOnly})prove(GetParam(),dense_full,set,TimingRows::None);}
TEST_P(ROCmQKVVerifierTransaction,DenseLocalTP2Replay) {for(auto set:{AttentionProjectionSet::QueryKeyValue,AttentionProjectionSet::KeyValueOnly})prove(GetParam(),dense_tp2,set,TimingRows::None);}
TEST_P(ROCmQKVVerifierTransaction,MoEFullReplay) {for(auto set:{AttentionProjectionSet::QueryKeyValue,AttentionProjectionSet::KeyValueOnly})prove(GetParam(),moe_full,set,TimingRows::None);}
TEST_P(ROCmQKVVerifierTransaction,MoELocalTP4Replay) {for(auto set:{AttentionProjectionSet::QueryKeyValue,AttentionProjectionSet::KeyValueOnly})prove(GetParam(),moe_tp4,set,TimingRows::None);}
TEST_P(ROCmQKVVerifierTransaction,DISABLED_DenseFullPaired) {for(auto set:{AttentionProjectionSet::QueryKeyValue,AttentionProjectionSet::KeyValueOnly})prove(GetParam(),dense_full,set,TimingRows::EveryPrefix);}
TEST_P(ROCmQKVVerifierTransaction,DISABLED_MoELocalTP4Paired) {for(auto set:{AttentionProjectionSet::QueryKeyValue,AttentionProjectionSet::KeyValueOnly})prove(GetParam(),moe_tp4,set,TimingRows::EveryPrefix);}
TEST_P(ROCmQKVVerifierTransaction,DISABLED_DenseFullExtentPaired) {for(auto set:{AttentionProjectionSet::QueryKeyValue,AttentionProjectionSet::KeyValueOnly})prove(GetParam(),dense_full,set,TimingRows::FullExtent);}
TEST_P(ROCmQKVVerifierTransaction,DISABLED_DenseLocalTP2FullExtentPaired) {for(auto set:{AttentionProjectionSet::QueryKeyValue,AttentionProjectionSet::KeyValueOnly})prove(GetParam(),dense_tp2,set,TimingRows::FullExtent);}
TEST_P(ROCmQKVVerifierTransaction,DISABLED_MoEFullExtentPaired) {for(auto set:{AttentionProjectionSet::QueryKeyValue,AttentionProjectionSet::KeyValueOnly})prove(GetParam(),moe_full,set,TimingRows::FullExtent);}
TEST_P(ROCmQKVVerifierTransaction,DISABLED_MoELocalTP4FullExtentPaired) {for(auto set:{AttentionProjectionSet::QueryKeyValue,AttentionProjectionSet::KeyValueOnly})prove(GetParam(),moe_tp4,set,TimingRows::FullExtent);}
TEST(ROCmQKVVerifierTransactionMixed,ActualStageBiasReplay) {for(auto set:{AttentionProjectionSet::QueryKeyValue,AttentionProjectionSet::KeyValueOnly})prove(quantizedMoEVerifierFormats().front(),moe_tp4,set,TimingRows::None,WeightSources::MixedWithBias);}
TEST_P(ROCmQKVVerifierTransaction,DenseFullAutomaticReplay) {for(auto set:{AttentionProjectionSet::QueryKeyValue,AttentionProjectionSet::KeyValueOnly})prove(GetParam(),dense_full,set,TimingRows::None,WeightSources::Homogeneous,PairPolicy::ExactHostedAutomatic);}
TEST_P(ROCmQKVVerifierTransaction,DenseLocalTP2AutomaticReplay) {for(auto set:{AttentionProjectionSet::QueryKeyValue,AttentionProjectionSet::KeyValueOnly})prove(GetParam(),dense_tp2,set,TimingRows::None,WeightSources::Homogeneous,PairPolicy::ExactHostedAutomatic);}
TEST_P(ROCmQKVVerifierTransaction,MoEFullAutomaticReplay) {for(auto set:{AttentionProjectionSet::QueryKeyValue,AttentionProjectionSet::KeyValueOnly})prove(GetParam(),moe_full,set,TimingRows::None,WeightSources::Homogeneous,PairPolicy::ExactHostedAutomatic);}
TEST_P(ROCmQKVVerifierTransaction,MoELocalTP4AutomaticReplay) {for(auto set:{AttentionProjectionSet::QueryKeyValue,AttentionProjectionSet::KeyValueOnly})prove(GetParam(),moe_tp4,set,TimingRows::None,WeightSources::Homogeneous,PairPolicy::ExactHostedAutomatic);}
TEST_P(ROCmQKVVerifierTransaction,DISABLED_DenseFullAutomaticPaired) {for(auto set:{AttentionProjectionSet::QueryKeyValue,AttentionProjectionSet::KeyValueOnly})prove(GetParam(),dense_full,set,TimingRows::FullExtent,WeightSources::Homogeneous,PairPolicy::ExactHostedAutomatic);}
TEST_P(ROCmQKVVerifierTransaction,DISABLED_DenseLocalTP2AutomaticPaired) {for(auto set:{AttentionProjectionSet::QueryKeyValue,AttentionProjectionSet::KeyValueOnly})prove(GetParam(),dense_tp2,set,TimingRows::FullExtent,WeightSources::Homogeneous,PairPolicy::ExactHostedAutomatic);}
TEST_P(ROCmQKVVerifierTransaction,DISABLED_MoEFullAutomaticPaired) {for(auto set:{AttentionProjectionSet::QueryKeyValue,AttentionProjectionSet::KeyValueOnly})prove(GetParam(),moe_full,set,TimingRows::FullExtent,WeightSources::Homogeneous,PairPolicy::ExactHostedAutomatic);}
TEST_P(ROCmQKVVerifierTransaction,DISABLED_MoELocalTP4AutomaticPaired) {for(auto set:{AttentionProjectionSet::QueryKeyValue,AttentionProjectionSet::KeyValueOnly})prove(GetParam(),moe_tp4,set,TimingRows::FullExtent,WeightSources::Homogeneous,PairPolicy::ExactHostedAutomatic);}
TEST(ROCmQKVVerifierTransactionAutomatic,GuardedCapacityAndMixedBiasRemainOrdered) {
    for(auto geometry:{dense_full,dense_tp2,moe_full,moe_tp4})
        for(auto set:{AttentionProjectionSet::QueryKeyValue,AttentionProjectionSet::KeyValueOnly})
            prove(quantizedMoEVerifierFormats().front(),geometry,set,TimingRows::None,WeightSources::Homogeneous,PairPolicy::GuardedAutomatic);
    for(auto sources:{WeightSources::Mixed,WeightSources::HomogeneousWithBias,WeightSources::MixedWithBias})
        for(auto set:{AttentionProjectionSet::QueryKeyValue,AttentionProjectionSet::KeyValueOnly})
            prove(quantizedMoEVerifierFormats().front(),moe_tp4,set,TimingRows::None,sources,PairPolicy::ExactHostedAutomatic);
}
/** @test The original unmeasured bundle blocks a member's qualified singleton geometry. */
TEST(ROCmQKVVerifierTransactionAutomatic,UnmeasuredBundleCannotBorrowScalarPolicy) {
    constexpr Geometry unmeasured{"Unmeasured_QKV_ScalarWinner",5120,{17408,1024,1024}};
    prove(quantizedMoEVerifierFormats().front(),unmeasured,AttentionProjectionSet::QueryKeyValue,
        TimingRows::None,WeightSources::Homogeneous,PairPolicy::ExactHostedAutomatic,AutomaticDomain::UnmeasuredBundle);
}

/** @brief One complete operation per selected profiler region, retaining its independent M1 oracle. */
class ROCmQKVVerifierNativeProfile:public ::testing::TestWithParam<std::pair<Geometry,AttentionProjectionSet>> {};
/** @test Profile actual Q4_0 Automatic at the independently established full-width native anchors. */
TEST_P(ROCmQKVVerifierNativeProfile,DISABLED_ExactNativeM16) {
    const auto &[geometry,set]=GetParam();
    prove(quantizedMoEVerifierFormats().front(),geometry,set,TimingRows::None,
        WeightSources::Homogeneous,PairPolicy::ExactHostedAutomatic,AutomaticDomain::MeasuredBundle,ProfileSelection{16});
}
INSTANTIATE_TEST_SUITE_P(Q4Anchors,ROCmQKVVerifierNativeProfile,::testing::Values(
    std::pair{dense_full,AttentionProjectionSet::QueryKeyValue},std::pair{dense_full,AttentionProjectionSet::KeyValueOnly},
    std::pair{dense_tp2,AttentionProjectionSet::QueryKeyValue},std::pair{dense_tp2,AttentionProjectionSet::KeyValueOnly},
    std::pair{moe_full,AttentionProjectionSet::QueryKeyValue},std::pair{moe_full,AttentionProjectionSet::KeyValueOnly},
    std::pair{moe_tp4,AttentionProjectionSet::QueryKeyValue},std::pair{moe_tp4,AttentionProjectionSet::KeyValueOnly}),
    [](const ::testing::TestParamInfo<std::pair<Geometry,AttentionProjectionSet>> &p) {
        return std::string(p.param.first.name)+(p.param.second==AttentionProjectionSet::QueryKeyValue?"_QKV":"_KV");
    });
INSTANTIATE_TEST_SUITE_P(AllSources,ROCmQKVVerifierTransaction,::testing::ValuesIn(quantizedMoEVerifierFormats()),
    [](const ::testing::TestParamInfo<QuantizedVerifierFormatCase> &info){return std::string(info.param.label);});
