/**
 * @file ROCmGatedNormSingleQ8ProductionTest.h
 * @brief Real prepared-owner norm/SingleQ8 capture, bytes, dirty reset and complete events.
 *
 * Reuses this translation unit's CapturedDecodeLaunch, original native projection,
 * canonical format registry and production GPU preparation helper. No cloned
 * kernel graph, dispatch override, diagnostic arithmetic or synthetic gain is
 * substituted for the selected GEMMStage/ITensorGemm composite. Test-only
 * publications/readbacks and terminal joins stay outside every event interval.
 */
#pragma once
#include "execution/compute_stages/stages/GEMMStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "kernels/common/GPUQ8ResidualNative.h"
#include "kernels/rocm/gdn/ROCmGatedNormSingleQ8.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "tensors/TensorSlice.h"
#include <array>
#include <bit>
#include <cstddef>
#include <cstring>
#include <type_traits>
extern "C" bool rocmGDN_gated_rmsnorm(const float*,const float*,const float*,float*,
    int,int,int,int,float,bool,llaminar2::GatedNormActivation,int,void*);
namespace gated_norm_single_q8_production
{
using namespace llaminar2;
/** @brief Fatal diagnostic admission; never select another arithmetic program. */
inline void require(bool value,const char* why) {if(!value)throw std::runtime_error(why);}
/**
 * @brief Recover one source-owned captured argument without constructing a placeholder.
 * @tparam T Exact trivially-copyable native argument type, including checked descriptors.
 * @param node Retained kernel node whose immutable argument storage is still alive.
 * @param index Source-authenticated zero-based argument slot in that native ABI.
 * @return Value reconstructed from all sizeof(T) captured bytes, without a default constructor.
 * @throws std::runtime_error If the retained argument storage is absent.
 * @details Copying through bytes avoids pointer-object strict aliasing. bit_cast
 * also admits ProjectionRowWork, whose checked factories intentionally provide
 * no default constructor; no descriptor fields or replacement policy are guessed.
 */
template<class T>T argument(const hipKernelNodeParams& node,int index)
{
    require(node.kernelParams&&node.kernelParams[index],"Missing captured argument");
    static_assert(std::is_trivially_copyable_v<T>);
    std::array<std::byte,sizeof(T)> bytes{};
    std::memcpy(bytes.data(),node.kernelParams[index],bytes.size());
    return std::bit_cast<T>(bytes);
}
/** @brief Preserve exact native producer/reducer argument and geometry identity. */
inline void sameProjection(const hipKernelNodeParams& a,const hipKernelNodeParams& b,bool producer)
{
    require(a.func==b.func&&a.gridDim.x==b.gridDim.x&&a.gridDim.y==b.gridDim.y&&a.gridDim.z==b.gridDim.z&&
        a.blockDim.x==b.blockDim.x&&a.blockDim.y==b.blockDim.y&&a.blockDim.z==b.blockDim.z&&a.sharedMemBytes==b.sharedMemBytes,
        "Composite changed original projection function/geometry");
    const auto pointer=[&](const hipKernelNodeParams& p,int i){return argument<const void*>(p,i);};
    for(int i=0;i<(producer?9:2);++i)require(pointer(a,i)==pointer(b,i),"Composite changed original projection pointer");
    for(int i=producer?9:2;i<(producer?12:4);++i)
        require(argument<int>(a,i)==argument<int>(b,i),"Composite changed original projection scalar");
    require(argument<ProjectionRowWork>(a,producer?12:4).captureIdentity()==
        argument<ProjectionRowWork>(b,producer?12:4).captureIdentity(),"Composite changed original projection row authority");
}
/** @brief One exact native stream retired after all graph/borrowed-storage owners. */
struct Stream
{
    hipStream_t value=nullptr;
    Stream(){require(hipStreamCreateWithFlags(&value,hipStreamNonBlocking)==hipSuccess,"Cannot create norm stream");}
    ~Stream(){if(value&&(hipStreamSynchronize(value)!=hipSuccess||hipStreamDestroy(value)!=hipSuccess))std::terminate();}
};
/** @brief Existing PMA/BOM-backed workspace; no anonymous reserve. */
struct Workspace
{
    std::shared_ptr<PhysicalMemoryAuthority> authority;
    std::unique_ptr<DeviceWorkspaceManager> value;
    Workspace(DeviceId device,const WorkspaceRequirements& requirements)
    {
        const auto bytes=requirements.total_bytes_with_alignment();PhysicalMemoryPlanBuilder plan;
        plan.add({.world_rank=0,.device=device,.total_bytes=bytes,.admission_available_bytes=bytes},
            PhysicalMemoryOwner::ExecutionWorkspace,bytes);
        authority=std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
        value=std::make_unique<DeviceWorkspaceManager>(device,bytes,authority);
        require(value->allocate(requirements),"Norm workspace admission failed");
    }
};
/** @brief One original owner pair, retained native captures and complete byte bank observations. */
class Cell
{
public:
    DeviceId device=DeviceId::rocm(0);Stream stream;
    int n,k,parts=1;GatedRMSNormProjectionParameters policy;
    std::unique_ptr<TensorBase> weights;
    test::GpuPreparedGemm prepared;
    std::shared_ptr<FP32Tensor> values_owner;
    FP32Tensor& values; ///< Same actual tensor; shared ownership admits its real native alias view.
    FP32Tensor gate,gamma,output;
    std::unique_ptr<Workspace> workspace;
    std::unique_ptr<IDeviceContext> context;
    std::unique_ptr<GEMMStage> stage;
    CapturedDecodeLaunch control,candidate;
    std::vector<std::pair<void*,size_t>> banks,immutable;
    std::vector<std::vector<std::uint8_t>> original_immutable;
    std::uint64_t signature=0,recorded_submissions=0;
    /** @return Setup/submission evidence only; retained native replay must not increment this host counter. */
    std::uint64_t submissionCount()const {
        std::uint64_t count=0;for(const auto& record:PerfStatsCollector::snapshot({"gdn_output.normalization_submission"}))
            if(record.domain=="gdn_output"&&record.name=="normalization_submission"&&record.tags.at("preparation")=="fused_single_q8")count+=record.count;
        return count;
    }
    /** @brief Materialize actual original sources/scratch, bind once, then capture both programs. */
    Cell(const test::QuantizedVerifierFormatCase& format,int columns,int width,
        GatedNormActivation activation,bool subtract,float epsilon=1e-6f)
        :n(columns),k(width),policy{128,128,epsilon,subtract,activation},
         weights(format.create({static_cast<size_t>(columns),static_cast<size_t>(width)},91)),
         prepared(test::makeGpuPreparedGemm(weights.get(),device,"norm_projection.original")),
         values_owner(std::make_shared<FP32Tensor>(std::vector<size_t>{1u,static_cast<size_t>(width+16)})),
         values(*values_owner),gate({1u,static_cast<size_t>(width+16)}),
         gamma({144u}),output({1u,static_cast<size_t>(columns+16)})
    {
        try {
            for(auto* tensor:{&values,&gate,&gamma,&output})
                require(tensor->allocateOnDevice(device),"Norm tensor allocation failed");
            auto* consumer=dynamic_cast<IWorkspaceConsumer*>(prepared.kernel);
            require(consumer,"Original prepared source has no workspace contract");
            workspace=std::make_unique<Workspace>(device,consumer->getWorkspaceRequirements(3,n,k));
            context=IDeviceContext::create(device,1);require(bool(context),"Missing exact native context");
            GEMMStage::Params params;
            params.device_id=device;params.A=&values;params.B=weights.get();params.C=&output;
            params.gated_norm=GEMMStage::GatedNormPublication{&gate,&gamma,BufferId::GDN_Z,policy};
            params.m=1;params.n=n;params.k=k;params.a_buffer_id=BufferId::ATTN_OUTPUT;params.c_buffer_id=BufferId::ATTN_PROJ;
            params.prepared_ref=prepared.ref;params.prepared_store=prepared.store.get();
            stage=std::make_unique<GEMMStage>(params);stage->setGPUStream(stream.value);
            stage->bindWorkspace(workspace->value.get());publish(0);join();
            require(stage->prepareGraphLaunch(context.get(),stream.value),"Composite cold preparation failed");
            signature=stage->graphCaptureVariantSignature();require(signature!=0,"Missing composite capture identity");
            for(const auto* name:{GemmWorkspaceBuffers::QUANT_A,GemmWorkspaceBuffers::SCALES_A_BLOCKWISE,
                GemmWorkspaceBuffers::SUMS_A_BLOCKWISE,GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL,GemmWorkspaceBuffers::SCALES_A})
                banks.emplace_back(workspace->value->getBuffer(name),workspace->value->getBufferSize(name));
            DeviceNativeVNNIMatrixDesc descriptor;
            require(prepared.kernel->inspectPreparedNativeVNNIMatrixDesc(descriptor),"Missing original native planes");
            const auto physical=requireQ8ResidualNativeMatrix(descriptor,1,n,k);
            const auto blocks=static_cast<size_t>(n)*descriptor.blocks_per_row;
            immutable={{const_cast<uint8_t*>(descriptor.payload),blocks*(descriptor.allocation_payload_bytes_per_block?
                descriptor.allocation_payload_bytes_per_block:physical.payload_bytes_per_block)},
                {const_cast<void*>(descriptor.scales),blocks*2}};
            if(descriptor.mins)immutable.emplace_back(const_cast<void*>(descriptor.mins),blocks*2);
            if(descriptor.emins)immutable.emplace_back(const_cast<void*>(descriptor.emins),blocks*4);
            for(const auto& bank:immutable)original_immutable.push_back(download(bank.first,bank.second));
            const std::vector<const TensorBase*> inputs{&values,&gate,&gamma};
            const std::vector<TensorBase*> outputs{&values,&output};
            // Recording declares dependencies but cannot import an external
            // producer event. Join the setup publications on this exact stream
            // before either capture, as the production executor does.
            for(const auto* input:inputs)
                TransferEngine::requireDeviceInput(const_cast<TensorBase*>(input),device,stream.value);
            const auto before_submissions=submissionCount();
            // Capture the actual selected program first: the control cannot
            // initialize a missing source/table/workspace on its behalf.
            require(candidate.capture(stream.value,device,inputs,outputs,[&]{return stage->execute(context.get());}),
                "Actual composite stage capture failed");
            require(control.capture(stream.value,device,inputs,outputs,[&]{
                if(!rocmGDN_gated_rmsnorm(static_cast<const float*>(values.gpu_data_ptr()),
                    static_cast<const float*>(gate.gpu_data_ptr()),static_cast<const float*>(gamma.gpu_data_ptr()),
                    static_cast<float*>(values.gpu_data_ptr()),1,k,128,128,policy.eps,policy.subtract_one,policy.activation,
                    0,stream.value))return false;
                TransferEngine::publishDeviceWrite(&values,device,stream.value);
                return prepared.kernel->multiply_tensor(&values,&output,1,n,k,false,1.f,0.f,nullptr,nullptr,0,workspace->value.get());
            }),"Original complete capture failed");
            recorded_submissions=submissionCount();
            require(recorded_submissions==before_submissions+1,"Actual composite capture submission counter missing");
            inspect();
        }catch(...){join();throw;}
    }
    /** @brief Retire native work/graphs before any borrowed allocation can unwind. */
    ~Cell(){join();control.reset();candidate.reset();if(stage)stage->unbindWorkspace();}
    /** @brief Test-only terminal observation boundary, never production ordering. */
    void join()const {if(hipStreamSynchronize(stream.value)!=hipSuccess)std::terminate();}
    /** @brief Observe only an explicitly owned native bank, outside captured/timed work. */
    std::vector<uint8_t> download(const void* pointer,size_t bytes)const
    {
        std::vector<uint8_t> result(bytes);
        require(hipMemcpyAsync(result.data(),pointer,bytes,hipMemcpyDeviceToHost,stream.value)==hipSuccess,"Norm byte observation failed");
        join();return result;
    }
    /** @brief Author directed real values and immutable gate/gamma guards for one dirty ordinal. */
    std::vector<float> authored(int dirty,bool z=false)const
    {
        std::vector<float> result(k+16,std::bit_cast<float>(std::uint32_t{0x4e123456}));
        for(int i=0;i<k;++i)result[i]=z?float((i+dirty)%19-9)*0.17f:float((i*7+dirty*11)%47-23)*0.03125f;
        result[0]=0.f;result[1]=-0.f;result[2]=std::numeric_limits<float>::denorm_min();
        result[32]=dirty==1?127.f:2.125f;result[64]=-2.f;
        if(!z&&dirty%4==2)for(int i=0;i<k;++i)result[i]=i%2?-0.f:0.f;
        if(z&&dirty%4==3)result[17]=std::numeric_limits<float>::max();
        return result;
    }
    /** @brief Restore input bytes and poison exact persistent capacities before each replay. */
    void publish(int dirty,unsigned char poison=0xa5)
    {
        const auto raw=authored(dirty),z=authored(dirty,true);
        std::vector<float> gain(144,std::bit_cast<float>(std::uint32_t{0x4e123456}));
        for(int i=0;i<128;++i)gain[i]=float((i%11)-5)*0.0625f+1.f;
        for(const auto& item:std::array<std::pair<FP32Tensor*,const std::vector<float>*>,3>{{{&values,&raw},{&gate,&z},{&gamma,&gain}}}) {
            require(hipMemcpyAsync(item.first->gpu_data_ptr(),item.second->data(),item.second->size()*sizeof(float),
                hipMemcpyHostToDevice,stream.value)==hipSuccess,"Norm input restoration failed");
            TransferEngine::publishDeviceWrite(item.first,device,stream.value);
        }
        require(hipMemsetAsync(output.gpu_data_ptr(),poison,output.size_bytes(),stream.value)==hipSuccess,"Norm output poison failed");
        for(const auto& bank:banks)require(hipMemsetAsync(bank.first,poison,bank.second,stream.value)==hipSuccess,"Norm bank poison failed");
        join(); // Own the local host-vector lifetimes; outside graph/events.
    }
    /** @brief Authenticate exact producer ABI and every unchanged native projection argument. */
    void inspect()
    {
        const auto before=control.kernelNodes(),after=candidate.kernelNodes();
        require(before.size()==after.size()+1,"Composite did not remove exactly one preparation kernel");
        std::vector<hipKernelNodeParams> original;
        for(const auto& node:before) {
            const auto* raw=hipKernelNameRefByPtr(node.func,stream.value);require(raw,"Missing original raw function symbol");
            const std::string name(raw);
            if(name.find("gemv_native_vnni_kernel_t")!=std::string::npos||name.find("gemv_native_vnni_reduce_kernel_t")!=std::string::npos) {
                original.push_back(node);
                if(name.find("reduce_kernel_t")==std::string::npos)parts=argument<int>(node,11);
            }
            else require(name.find("rocm_gated_rmsnorm_kernel")!=std::string::npos||name.find("quantizeActivationsQ8")!=std::string::npos,
                "Foreign original complete-boundary kernel");
        }
        size_t fused=0,matched=0;
        for(const auto& node:after) {
            const auto* raw=hipKernelNameRefByPtr(node.func,stream.value);require(raw,"Missing candidate raw symbol");
            const std::string name(raw);
            if(name.find("rocm_gdn_gated_norm_single_q8_kernel")!=std::string::npos) {
                ++fused;const auto native=argument<rocm::GatedNormSingleQ8Input>(node,0);
                require(native.input==values.gpu_data_ptr()&&native.output==values.gpu_data_ptr()&&native.gate==gate.gpu_data_ptr()&&
                    native.gamma==gamma.gpu_data_ptr()&&native.columns==k&&native.parameters==policy&&
                    native.quant==banks[0].first&&native.scales==banks[1].first,"Captured composite producer ABI differs");
                require(node.gridDim.x==static_cast<unsigned>(k/128)&&node.gridDim.y==1&&node.gridDim.z==1&&
                    node.blockDim.x==128&&node.blockDim.y==1&&node.blockDim.z==1&&node.sharedMemBytes==0,
                    "Composite producer native geometry differs");
                hipFuncAttributes resource{};require(hipFuncGetAttributes(&resource,node.func)==hipSuccess&&resource.localSizeBytes==0,
                    "Composite producer has private storage");
            }else {
                const auto old=std::find_if(original.begin(),original.end(),[&](const auto& n){return n.func==node.func;});
                require(old!=original.end(),"Composite inserted a different projection");
                sameProjection(node,*old,name.find("reduce_kernel_t")==std::string::npos);++matched;
            }
        }
        require(fused==1&&matched==original.size()&&!original.empty(),"Incomplete original projection/producer census");
        for(const auto& side:std::array<std::pair<const char*,const std::vector<hipKernelNodeParams>*>,2>{{{"control",&before},{"candidate",&after}}})
            for(const auto& node:*side.second) {
                const auto* raw=hipKernelNameRefByPtr(node.func,stream.value);require(raw,"Missing retained native symbol");
                hipFuncAttributes attributes{};require(hipFuncGetAttributes(&attributes,node.func)==hipSuccess,"Missing actual native resources");
                std::cout<<"GDN_NORM_SINGLE_Q8_PRODUCTION_CAPTURE,side="<<side.first<<",M=1,N="<<n<<",K="<<k
                    <<",grid="<<node.gridDim.x<<"x"<<node.gridDim.y<<"x"<<node.gridDim.z
                    <<",block="<<node.blockDim.x<<",dynamic_lds="<<node.sharedMemBytes<<",static_lds="<<attributes.sharedSizeBytes
                    <<",registers="<<attributes.numRegs<<",private="<<attributes.localSizeBytes<<",raw_symbol="<<raw<<"\n";
            }
    }
    /** @brief Compare every produced/unused byte, guards and immutable native plane after a dirty replay. */
    std::vector<std::vector<uint8_t>> observe(bool fused,int dirty,unsigned char poison)
    {
        publish(dirty,poison);require((fused?candidate:control).launch(stream.value),"Norm retained replay failed");join();
        return currentObservation(dirty,poison);
    }
    /** @brief Observe the ACTUAL completed result without restoring, submitting or replaying anything. */
    std::vector<std::vector<uint8_t>> currentObservation(int dirty,unsigned char poison)
    {
        require(submissionCount()==recorded_submissions,"Native replay performed host stage submission work");
        std::vector<std::vector<uint8_t>> result{download(values.gpu_data_ptr(),values.size_bytes()),download(output.gpu_data_ptr(),output.size_bytes())};
        for(const auto& bank:banks)result.push_back(download(bank.first,bank.second));
        for(size_t i=0;i<immutable.size();++i)require(download(immutable[i].first,immutable[i].second)==original_immutable[i],"Original prepared weight plane changed");
        const auto z=authored(dirty,true);const auto gate_bytes=download(gate.gpu_data_ptr(),gate.size_bytes());
        require(std::memcmp(gate_bytes.data(),z.data(),gate_bytes.size())==0,"Immutable gate/guard changed");
        const auto guard=authored(dirty);require(std::memcmp(result[0].data()+k*sizeof(float),guard.data()+k,16*sizeof(float))==0,"Normalized FP32 guard changed");
        require(std::all_of(result[1].begin()+n*sizeof(float),result[1].end(),[&](uint8_t b){return b==poison;}),"Projection guard changed");
        require(std::all_of(result[2].begin()+k,result[2].end(),[&](uint8_t b){return b==poison;}),"Quantized inactive capacity changed");
        require(std::all_of(result[3].begin()+k/32*sizeof(float),result[3].end(),[&](uint8_t b){return b==poison;}),"Scale inactive capacity changed");
        require(std::all_of(result[5].begin()+(parts>1?parts*n*sizeof(float):0),result[5].end(),[&](uint8_t b){return b==poison;}),"Partial inactive capacity changed");
        std::vector<float> gain(144,std::bit_cast<float>(std::uint32_t{0x4e123456}));
        for(int i=0;i<128;++i)gain[i]=float((i%11)-5)*0.0625f+1.f;
        const auto gain_bytes=download(gamma.gpu_data_ptr(),gamma.size_bytes());
        require(std::memcmp(gain_bytes.data(),gain.data(),gain_bytes.size())==0,"Immutable gain/guard changed");
        const auto sum=result[4];require(std::all_of(sum.begin(),sum.end(),[&](uint8_t b){return b==poison;}),"Unused scalar sums bank changed");
        const auto row_scale=result[6];require(std::all_of(row_scale.begin(),row_scale.end(),[&](uint8_t b){return b==poison;}),"Unused row-scale bank changed");
        return result;
    }
    /** @brief A failed native device selection must return before ANY producer can modify a bank. */
    void failedDeviceSelection()
    {
        rocm::GatedNormSingleQ8Input native{};bool found=false;
        for(const auto& node:candidate.kernelNodes()) {
            const auto* raw=hipKernelNameRefByPtr(node.func,stream.value);
            if(raw&&std::string(raw).find("rocm_gdn_gated_norm_single_q8_kernel")!=std::string::npos) {
                native=argument<rocm::GatedNormSingleQ8Input>(node,0);found=true;
            }
        }
        require(found,"Missing exact captured producer tuple for selection rejection");
        publish(9,0xa5);const auto before=currentObservation(9,0xa5);
        (void)hipGetLastError(); // Clear only prior diagnostic runtime status.
        require(!rocm::launchGatedNormSingleQ8(native,std::numeric_limits<int>::max(),stream.value),
            "Failed device selection was admitted");
        (void)hipGetLastError(); // Expected invalid-device API status is test-local.
        join();require(currentObservation(9,0xa5)==before,"Failed selection enqueued a producer");
        EXPECT_THROW(rocm::launchGatedNormSingleQ8(native,0,nullptr),std::invalid_argument);
    }
    /** @brief Assert the actual stage binding rejects aliases and unsupported transforms before capture. */
    void invalid()
    {
        auto params=stage->getParams();params.gated_norm->gate=&values;
        EXPECT_THROW(static_cast<void>(GEMMStage{params}),std::invalid_argument);
        params=stage->getParams();params.gated_norm->gamma=&values;EXPECT_THROW(static_cast<void>(GEMMStage{params}),std::invalid_argument);
        params=stage->getParams();params.verifier_row_range=DeviceRowRange::fullyActive(1);EXPECT_THROW(static_cast<void>(GEMMStage{params}),std::invalid_argument);
        params=stage->getParams();params.quantized_operand_precision=QuantizedActivationOperandPrecision::TwoTermQ8;EXPECT_THROW(static_cast<void>(GEMMStage{params}),std::invalid_argument);
        params=stage->getParams();params.prepared_ref.reset();EXPECT_THROW(static_cast<void>(GEMMStage{params}),std::invalid_argument);
        auto alias=std::make_shared<TensorSlice>(std::static_pointer_cast<TensorBase>(values_owner),
            SliceMetadata{.mode=SliceMode::FULL,.original_rows=1,.original_cols=static_cast<size_t>(k+16),
                .slice_start=0,.slice_end=1,.inner_is_presliced=true});
        require(alias->gpu_data_ptr()==values.gpu_data_ptr()&&alias->current_device()==std::optional{device},
            "Alias negative must expose the actual same native storage owner");
        EXPECT_THROW(prepared.kernel->validate_gated_rmsnorm_single_q8({values,*alias,gamma,policy},&output,n,k,workspace->value.get()),std::invalid_argument);
        EXPECT_THROW(prepared.kernel->validate_gated_rmsnorm_single_q8({values,gate,*alias,policy},&output,n,k,workspace->value.get()),std::invalid_argument);
        params=stage->getParams();params.gated_norm->gate=alias.get();
        GEMMStage aliased_stage(params);aliased_stage.setGPUStream(stream.value);aliased_stage.bindWorkspace(workspace->value.get());
        EXPECT_THROW(aliased_stage.prepareGraphLaunch(context.get(),stream.value),std::invalid_argument);
        auto* consumer=dynamic_cast<IWorkspaceConsumer*>(prepared.kernel);
        Workspace foreign(device,consumer->getWorkspaceRequirements(3,n,k));
        EXPECT_THROW(stage->bindWorkspace(foreign.value.get()),std::logic_error);
        require(stage->graphCaptureVariantSignature()==signature,"Rejected foreign binding changed original owner");
    }
};
/** @brief Current original source catalog; no implementation-only synthetic decoder oracle. */
inline void matrix()
{
    require(PerfStatsCollector::isDomainEnabled("gdn_output"),"Enable exact gdn_output.normalization_submission evidence for the native owner gate");
    size_t cases=0,replays=0;
    for(const auto& format:test::quantizedVerifierFormats())
    for(const int heads:{1,3,12}) {
        // Whole256 source blocks are retained for every superblock format.
        const int k=format.source_is_superblock?((heads+1)/2)*256:heads*128;const int n=heads==12?65:17;
        for(const auto activation:{GatedNormActivation::Identity,GatedNormActivation::SiLU,GatedNormActivation::Sigmoid}) {
            Cell cell(format,n,k,activation,heads==3);++cases;
            for(int dirty=0;dirty<4;++dirty) {
                const auto poison=dirty%2?0x5a:0xa5;
                const auto original=cell.observe(false,dirty,poison),fused=cell.observe(true,dirty,poison);
                require(original==fused,"Actual norm/quant/scale/sum/partial/output bytes differ");replays+=2;
                cell.stage->resetSessionStatePreservingCapturedReplay();
                require(cell.stage->graphCaptureVariantSignature()==cell.signature,"Data reset changed composite capture identity");
                // Existing captured replay needs no rebind/recapture/host stage action.
            }
            cell.stage->setGPUStream(cell.stream.value);
            require(cell.stage->prepareGraphLaunch(cell.context.get(),cell.stream.value),"Identical cold reentry failed");
            require(cell.stage->graphCaptureVariantSignature()==cell.signature,"Cold reentry changed retained signature");
            cell.invalid();
            if(format.tensor_type==TensorType::Q8_0&&heads==1&&activation==GatedNormActivation::Identity) {
                cell.failedDeviceSelection();
                require(cell.observe(false,13,0x5a)==cell.observe(true,13,0x5a),"Device-selection rejection damaged retained replay");
                replays+=2;
            }
        }
    }
    const auto& catalog=test::quantizedVerifierFormats();
    const auto q8=std::find_if(catalog.begin(),catalog.end(),[](const auto& f){return f.tensor_type==TensorType::Q8_0;});
    require(q8!=catalog.end(),"Missing epsilon witness source");
    // Original norm accepts these epsilon words. All-zero dirty2 produces
    // all-NaN heads for eps0/negative/NaN; prove the actual old half-wave V4
    // max/scale/RNE against the fused width32 subgroups instead of inferring it.
    for(const auto epsilon:{0.f,-0.f,std::bit_cast<float>(std::uint32_t{0x7fc12345}),-1e-6f}) {
        Cell cell(*q8,17,256,GatedNormActivation::Identity,false,epsilon);++cases;
        for(int dirty=0;dirty<4;++dirty) {
            const auto poison=dirty%2?0x5a:0xa5;
            require(cell.observe(false,dirty,poison)==cell.observe(true,dirty,poison),
                "Original epsilon/nonfinite scalar bytes differ");replays+=2;
        }
    }
    std::cout<<"GDN_NORM_SINGLE_Q8_PRODUCTION_BYTES,cases="<<cases<<",dirty_replays="<<replays
        <<",native_node_delta=-1,all_format_consumers=1,passed=1\n";
}
/** @brief Retire diagnostic timing handles on all exit paths after their exact stream joins. */
struct Events
{
    hipStream_t stream;hipEvent_t begin=nullptr,end=nullptr;
    explicit Events(hipStream_t value):stream(value) {
        require(hipEventCreate(&begin)==hipSuccess,"Cannot create event begin");
        if(hipEventCreate(&end)!=hipSuccess){if(hipEventDestroy(begin)!=hipSuccess)std::terminate();begin=nullptr;throw std::runtime_error("Cannot create event end");}
    }
    ~Events(){if(hipStreamSynchronize(stream)!=hipSuccess||(end&&hipEventDestroy(end)!=hipSuccess)||(begin&&hipEventDestroy(begin)!=hipSuccess))std::terminate();}
};
/** @brief Independently restored complete one-operation paired events; no repeated in-place normalization. */
inline void events()
{
    require(PerfStatsCollector::isDomainEnabled("gdn_output"),"Enable exact composite submission evidence");
    const auto& formats=test::quantizedVerifierFormats();const auto format=std::find_if(formats.begin(),formats.end(),[](const auto& f){return f.tensor_type==TensorType::Q8_0;});
    require(format!=formats.end(),"Missing actual native19 source");Cell cell(*format,2560,1536,GatedNormActivation::Sigmoid,false);
    require(cell.observe(false,7,0xa5)==cell.observe(true,7,0xa5),"Pre-event complete bytes failed");
    Events events(cell.stream.value);const auto begin=events.begin,end=events.end;
    std::array<std::vector<std::vector<uint8_t>>,2> actual_final;
    bool retain_final=false;
    const auto run=[&](bool fused,int dirty) {
        cell.publish(dirty);require(hipEventRecord(begin,cell.stream.value)==hipSuccess,"Event begin failed");
        require((fused?cell.candidate:cell.control).launch(cell.stream.value),"Complete timed graph replay failed");
        require(hipEventRecord(end,cell.stream.value)==hipSuccess&&hipEventSynchronize(end)==hipSuccess,"Event retirement failed");
        float ms=0;require(hipEventElapsedTime(&ms,begin,end)==hipSuccess&&ms>0,"Invalid native event");
        // Retain each actual final timed result before another arm can restore
        // the shared banks. All observation is after the interval has ended.
        if(retain_final)actual_final[fused?1:0]=cell.currentObservation(dirty,0xa5);
        return ms*1000.;
    };
    for(int i=0;i<5;++i){run(false,i);run(true,i);}
    for(int i=0;i<12;++i) {
        retain_final=i==11;
        double raw,fused;if(i%2){fused=run(true,i);raw=run(false,i);}else{raw=run(false,i);fused=run(true,i);}
        std::cout<<"GDN_NORM_SINGLE_Q8_PRODUCTION_PAIR,pair="<<i<<",raw_us="<<raw<<",fused_us="<<fused<<",operations=1\n";
    }
    require(!actual_final[0].empty()&&actual_final[0]==actual_final[1],"ACTUAL final timed banks/guards/source checks differ");
    std::cout<<"GDN_NORM_SINGLE_Q8_PRODUCTION_TIMED_BYTES,pair=11,actual_arms=2,restore_before_observation=0,passed=1\n";
    require(cell.observe(false,31,0x5a)==cell.observe(true,31,0x5a),"Additional restored dirty bytes failed");
}
} // namespace gated_norm_single_q8_production
TEST_F(NativeVNNIPerfTest,GatedNormSingleQ8ProductionOwnerAndAllFormatBytes)
{
    if(!has_device_)GTEST_SKIP()<<"ROCm device required";
    gated_norm_single_q8_production::matrix();
}
TEST_F(NativeVNNIPerfTest,DISABLED_GatedNormSingleQ8ProductionCompleteEvents)
{
    if(!has_device_)GTEST_SKIP()<<"ROCm device required";
    gated_norm_single_q8_production::events();
}
