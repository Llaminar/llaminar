/**
 * @file Test__GEMMStagePrefixDevice.cpp
 * @brief Actual prepared GEMMStage capture keeps a native M1 count through0/1/0 replay.
 *
 * The same captured stage and stable pointer consume changing execution-owned
 * words. Poison inactive source rows, retain dirty output, and compare the one
 * active row to an independent exactly-representable one-hot projection.
 * The direct fixture mirrors graph setup: the stage declares its complete named
 * workspace, one PMA admits it, and capture/replay borrow that persistent arena.
 * Captured work retires before workspace or prepared tensor owners on every exit.
 */
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "kernels/qwen4exp/QwenSparseMath.h"
#include "tensors/Tensors.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "transfer/TransferEngine.h"
#include "../../../utils/PreparedWeightTestHarness.h"
#include "../../../utils/GpuPreparedGemmHarness.h"
#include "../../../utils/QuantizedVerifierFormats.h"
#include "kernels/qwen4exp/QwenSparseQueryRowWork.h"
#include <cmath>
#include <cstring>
#include <cstdint>
#include <optional>
#include <functional>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace llaminar2;
namespace {
/** @brief Fail the exact native infrastructure operation without changing execution policy. */
void checked(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
/**
 * @brief Direct-fixture equivalent of graph-owned workspace admission and binding.
 *
 * The prepared stage is the only sizing authority. No fixture reserve or raw
 * allocation masks missing production requirements. Declare after the stage;
 * the graph retirement scope must retire before this borrow is revoked.
 */
class StageWorkspace final
{
public:
    /**
     * @brief Admit exactly the stage's immutable shape and bind its prepared engine.
     * @param stage Prepared consumer whose lifetime encloses this binding.
     * @param device Exact participant admitted by the one memory authority.
     * @param rows Maximum physical rows embedded in the retained graph.
     * @param columns Prepared native output columns.
     * @param reduction Prepared native input/reduction columns.
     * @throws std::runtime_error For missing requirements, allocation or binding.
     */
    StageWorkspace(IWorkspaceConsumer& stage, DeviceId device, int rows, int columns, int reduction)
        : stage_(stage)
    {
        const auto requirements = stage_.getWorkspaceRequirements(rows, columns, reduction);
        const auto bytes = requirements.total_bytes_with_alignment();
        checked(bytes != 0, "Counted floating stage declared no workspace");
        PhysicalMemoryPlanBuilder plan;
        plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
            PhysicalMemoryOwner::ExecutionWorkspace, bytes);
        memory_ = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
        workspace_ = std::make_unique<DeviceWorkspaceManager>(device, bytes, memory_);
        checked(workspace_->allocate(requirements), "M1 stage workspace admission failed");
        try
        {
            stage_.bindWorkspace(workspace_.get());
            checked(stage_.hasWorkspace(), "M1 stage did not bind its declared workspace");
        }
        catch (...)
        {
            stage_.unbindWorkspace();
            throw;
        }
    }
    /** @brief Revoke prepared-engine and stage borrows before arena/PMA retirement. */
    ~StageWorkspace() { stage_.unbindWorkspace(); }
    /** @return Admitted numerical banks for out-of-capture poison/terminal checks. */
    DeviceWorkspaceManager& value() const {return *workspace_;}
    StageWorkspace(const StageWorkspace&) = delete;
    StageWorkspace& operator=(const StageWorkspace&) = delete;
private:
    IWorkspaceConsumer& stage_; ///< Stage outlives this non-owning binding.
    std::shared_ptr<PhysicalMemoryAuthority> memory_; ///< Outlives all admitted bytes.
    std::unique_ptr<DeviceWorkspaceManager> workspace_; ///< Exact prepared-engine named banks.
};

/** @brief Diagnostic terminal join and graph retirement before captured owners unwind. */
class CapturedWorkRetirement final
{
public:
    /**
     * @brief Borrow graph and exact stream after all native/host transfer owners exist.
     * @param backend Infrastructure owner of the submitted graph and copies.
     * @param device Exact participant used for capture and terminal observation.
     * @param stream Non-default producer stream, never a guessed substitute.
     * @param graph Retained graph, retired before any captured storage release.
     */
    CapturedWorkRetirement(IBackend& backend, DeviceId device, void* stream,
        std::unique_ptr<IGPUGraphCapture>& graph)
        : backend_(backend), device_(device), stream_(stream), graph_(graph) {}
    /** @brief Fail fatally if terminal retirement cannot make owner destruction safe. */
    ~CapturedWorkRetirement()
    {
        if(!backend_.synchronizeStream(stream_,device_.ordinal))std::terminate();
        if(graph_){graph_->reset();graph_.reset();}
    }
    CapturedWorkRetirement(const CapturedWorkRetirement&) = delete;
    CapturedWorkRetirement& operator=(const CapturedWorkRetirement&) = delete;
private:
    IBackend& backend_; ///< Infrastructure backend remains alive through the worker task.
    DeviceId device_; ///< Same physical participant used by capture and publication.
    void* stream_; ///< Exact non-default producer stream, never a guessed substitute.
    std::unique_ptr<IGPUGraphCapture>& graph_; ///< Retired before workspace/tensor release.
};

/**
 * @brief Exercise the actual prepared floating stage through empty/active replay.
 * @param device Exact native participant; the fixture owns all preparation.
 * @throws std::runtime_error For incomplete preparation, capture or publication.
 */
void proveM1(DeviceId device)
{
    auto*backend=getBackendFor(device);
    if(!backend || backend->deviceCount()<=device.ordinal)GTEST_SKIP()<<"GPU unavailable";
    auto context=IDeviceContext::create(device,1);
    auto&worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&] {
        constexpr int n=33,k=257,activeColumn=37;
        const auto stream=ExplicitGPUStream(worker.defaultStream()).get();
        for(TensorType type:{TensorType::FP32,TensorType::FP16,TensorType::BF16}) {
            FP32Tensor input({1,k}),output({1,n}),floatWeight({n,k});
            FP16Tensor halfWeight({n,k});BF16Tensor bfloatWeight({n,k});INT32Tensor count({1});
            TensorBase*weight=type==TensorType::FP32?static_cast<TensorBase*>(&floatWeight):
                type==TensorType::FP16?static_cast<TensorBase*>(&halfWeight):static_cast<TensorBase*>(&bfloatWeight);
            for(int row=0;row<n;++row)for(int col=0;col<k;++col) {
                const float value=float((row+col)%13-6)/16.0f;const int index=row*k+col;
                if(type==TensorType::FP32)static_cast<float*>(weight->raw_mutable_data())[index]=value;
                else static_cast<std::uint16_t*>(weight->raw_mutable_data())[index]=type==TensorType::FP16 ?
                    qwen4exp::detail::encodeHalf(value):std::uint16_t(qwen4exp::detail::floatBits(value)>>16);
            }
            std::fill_n(input.mutable_data(),k,0.0f);std::fill_n(output.mutable_data(),n,-71.0f);
            *static_cast<int*>(count.raw_mutable_data())=0;
            for(auto*tensor:{static_cast<TensorBase*>(&input),static_cast<TensorBase*>(&output),weight,static_cast<TensorBase*>(&count)}) {
                TransferEngine::prepareDeviceInput(tensor,device,stream);
                TransferEngine::requireDeviceInput(tensor,device,stream);
            }
            auto prepared=test::makePreparedGemmFixture(weight,device,"qsa.indexer.m1.counted",ModelContextId{99281+static_cast<unsigned>(type)});
            const auto*owner=static_cast<const std::int32_t*>(count.gpu_data_ptr());
            const auto rows=DeviceRowRange::deviceCounted(1,owner);
            checked(rows.countOwner()==owner,"M1 typed prefix discarded its exact native count owner");
            GEMMStage stage({.device_id=device,.A=&input,.B=weight,.C=&output,.m=1,.n=n,.k=k,
                .force_decode_equivalent_verifier_prefill=true,.verifier_row_range=rows,
                .prepared_ref=prepared.ref,.prepared_store=prepared.store.get()});
            stage.setGPUStream(stream);std::string error;checked(stage.validatePreparedWeights(&error),error.c_str());
            StageWorkspace workspace(stage,device,1,n,k);
            const auto identity=stage.graphCaptureVariantSignature();checked(identity!=0,"M1 native count has no capture identity");
            std::vector<float> source(k),actual(n),expected(n,-71.0f);
            checked(backend->synchronizeStream(stream,device.ordinal),"M1 setup fence failed");
            auto graph=worker.createGraphCapture(stream);
            CapturedWorkRetirement retirement(*backend,device,stream,graph);
            {
                ScopedBackendGraphCapture capture(worker,*graph,"QSA real prepared M1 counted projection");
                checked(capture.begin(),"M1 capture begin failed");checked(stage.execute(context.get()),"M1 GEMMStage capture failed");capture.finish();
            }
            checked(graph->nodeCount()>0,"M1 prepared graph captured no work");checked(graph->instantiate(),"M1 instantiate failed");
            for(int live:{0,1,0}) {
                std::fill(source.begin(),source.end(),live?0.0f:std::numeric_limits<float>::quiet_NaN());
                if(live){source[activeColumn]=1.0f;for(int row=0;row<n;++row)expected[row]=float((row+activeColumn)%13-6)/16.0f;}
                checked(backend->hostToDeviceOnStream(input.gpu_data_ptr(),source.data(),input.size_bytes(),device.ordinal,stream),"M1 source publication failed");
                checked(backend->hostToDeviceOnStream(count.gpu_data_ptr(),&live,sizeof(live),device.ordinal,stream),"M1 count publication failed");
                EXPECT_EQ(stage.graphCaptureVariantSignature(),identity);
                checked(graph->launch(),"M1 graph replay failed");
                checked(backend->deviceToHostOnStream(actual.data(),output.gpu_data_ptr(),output.size_bytes(),device.ordinal,stream),"M1 output observation failed");
                checked(backend->synchronizeStream(stream,device.ordinal),"M1 terminal observation fence failed");
                EXPECT_EQ(actual,expected)<<device.toString()<<" type="<<static_cast<int>(type)<<" live="<<live;
            }
            // Retirement joins the exact stream and destroys the graph before workspace.
        }
    }).get();
}
/** @brief One real source representation; creators retain canonical quantized coverage. */
struct QueryWorkFormat {std::string label;bool quantized;test::QuantizedVerifierWeightCreator create;};
/** @return All floating representations and the complete original-format quantized registry. */
std::vector<QueryWorkFormat> queryWorkFormats()
{
    std::vector<QueryWorkFormat> result;
    for(auto type:{TensorType::FP32,TensorType::FP16,TensorType::BF16}){
        const char* label=type==TensorType::FP32?"FP32":type==TensorType::FP16?"FP16":"BF16";
        result.push_back({label,false,[type](const std::vector<size_t>& shape,std::uint32_t){
            std::unique_ptr<TensorBase> value;
            if(type==TensorType::FP32)value=std::make_unique<FP32Tensor>(shape);
            else if(type==TensorType::FP16)value=std::make_unique<FP16Tensor>(shape);
            else value=std::make_unique<BF16Tensor>(shape);
            for(std::size_t i=0;i<shape[0]*shape[1];++i){const float x=float(int(i%29)-14)/32.0f;
                if(type==TensorType::FP32)static_cast<float*>(value->raw_mutable_data())[i]=x;
                else static_cast<std::uint16_t*>(value->raw_mutable_data())[i]=type==TensorType::FP16?
                    qwen4exp::detail::encodeHalf(x):std::uint16_t(qwen4exp::detail::floatBits(x)>>16);}
            return value;}});
    }
    for(const auto& format:test::quantizedMoEVerifierFormats())result.push_back({format.label,true,format.create});
    return result;
}
/**
 * @brief Poison one named numerical bank without touching captured pointer/publication tables.
 * @details Live-row checks use original quantizer layouts; partial storage has
 * backend-owned split-K tiling, so its complete unchanged-byte proof is empty-only.
 */
struct QueryWorkBank {
    std::string name;std::size_t width,element;bool blockMajor,emptyOnly;
    std::vector<std::uint8_t> poison,observed;
};
/**
 * @brief Exercise actual prepared stage authorship against ordinary captured M1.
 * @param device Exact native participant owning every graph and storage lease.
 * @param grouped Whether to sweep1..16 rows per three independent request owners.
 * @test No-work rows preserve poisoned source/Q8/scales/sums/output, empty work
 * preserves every partial byte, and the first2052-visible row matches ordinary
 * production M1 without a verifier/work scope. One capture survives dirty replay.
 */
void proveQueryWork(DeviceId device,bool grouped)
{
    auto* backend=getBackendFor(device);if(!backend||backend->deviceCount()<=device.ordinal)GTEST_SKIP()<<"GPU unavailable";
    auto context=IDeviceContext::create(device,1);auto& worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&]{
        constexpr int n=33,k=512,maxRows=48;const auto stream=ExplicitGPUStream(worker.defaultStream()).get();
        for(const auto& format:queryWorkFormats()){
            SCOPED_TRACE(format.label);auto weight=format.create({n,k},7391);
            if(!format.quantized){TransferEngine::prepareDeviceInput(weight.get(),device,stream);TransferEngine::requireDeviceInput(weight.get(),device,stream);}
            std::optional<test::GpuPreparedGemm> gpu;std::optional<test::PreparedGemmFixture> floating;
            PreparedWeightStore* store=nullptr;PreparedWeightRef ref;
            if(format.quantized){gpu.emplace(test::makeGpuPreparedGemm(weight.get(),device,"qsa.query.work."+format.label,ModelContextId{99731}));store=gpu->store.get();ref=gpu->ref;}
            else{floating.emplace(test::makePreparedGemmFixture(weight.get(),device,"qsa.query.work."+format.label,ModelContextId{99731}));store=floating->store.get();ref=floating->ref;}
            auto* kernel=store->gemmKernel(ref);checked(kernel!=nullptr,"Prepared query kernel missing");kernel->setGPUStream(stream);
            FP32Tensor serialInput({1,k}),serialOutput({1,n});
            for(auto* t:{&serialInput,&serialOutput}){std::fill_n(t->mutable_data(),t->numel(),0.0f);TransferEngine::prepareDeviceInput(t,device,stream);TransferEngine::requireDeviceInput(t,device,stream);}
            GEMMStage serial({.device_id=device,.A=&serialInput,.B=weight.get(),.C=&serialOutput,.m=1,.n=n,.k=k,
                .prepared_ref=ref,.prepared_store=store});serial.setGPUStream(stream);
            std::string error;checked(serial.validatePreparedWeights(&error),error.c_str());StageWorkspace serialWorkspace(serial,device,1,n,k);
            checked(kernel->prepareFusedProjectionGraphCapture(1),"Ordinary query resources not capture-ready");
            std::array<std::vector<float>,2> source,oracle;
            checked(backend->synchronizeStream(stream,device.ordinal),"Query oracle setup fence failed");
            auto oracleGraph=worker.createGraphCapture(stream);CapturedWorkRetirement oracleRetirement(*backend,device,stream,oracleGraph);
            EXPECT_FALSE(ITensorGemm::VerifierKernelModeScope::active());EXPECT_EQ(currentProjectionRowWork(1),nullptr);
            {ScopedBackendGraphCapture capture(worker,*oracleGraph,"ordinary M1 query oracle");checked(capture.begin(),"Query oracle begin failed");checked(serial.execute(context.get()),"Query oracle capture failed");capture.finish();}
            checked(oracleGraph->instantiate(),"Query oracle instantiate failed");
            for(int pattern=0;pattern<2;++pattern){source[pattern].assign(maxRows*k,0.0f);oracle[pattern].resize(maxRows*n);
                for(int row=0;row<maxRows;++row){source[pattern][row*k+(row*17+3)%k]=pattern?-0.5f:0.5f;
                    checked(backend->hostToDeviceOnStream(serialInput.gpu_data_ptr(),source[pattern].data()+row*k,k*sizeof(float),device.ordinal,stream),"Serial query source failed");
                    checked(oracleGraph->launch(),"Serial query oracle replay failed");
                    checked(backend->deviceToHostOnStream(oracle[pattern].data()+row*n,serialOutput.gpu_data_ptr(),n*sizeof(float),device.ordinal,stream),"Serial query oracle download failed");
                    checked(backend->synchronizeStream(stream,device.ordinal),"Serial oracle terminal join failed");}
                for(float x:oracle[pattern])checked(std::isfinite(x),"Nonfinite ordinary query oracle");}
            checked(std::any_of(oracle[0].begin(),oracle[0].end(),[](float x){return x!=0.0f;}),"Vacuous all-zero query source witness");
            const int requests=grouped?3:1;
            for(int rows=1;rows<=(grouped?16:1);++rows){
                SCOPED_TRACE(rows);const int physical=requests*rows;
                FP32Tensor input({std::size_t(physical),k}),output({std::size_t(physical+1),n});
                std::fill_n(input.mutable_data(),input.numel(),0.0f);std::fill_n(output.mutable_data(),output.numel(),-71.0f);
                std::vector<std::unique_ptr<INT32Tensor>> owners;ProjectionRowWorkDeclaration declaration{.rowsPerRequest=rows,.sourceCapacity=4096,.firstRequiredLogicalRow=qwen4exp::QwenSparseFirstRankingLogicalRow};
                for(int request=0;request<requests;++request){auto owner=std::make_unique<INT32Tensor>(std::vector<std::size_t>{2});std::fill_n(static_cast<std::int32_t*>(owner->raw_mutable_data()),2,0);
                    TransferEngine::prepareDeviceInput(owner.get(),device,stream);TransferEngine::requireDeviceInput(owner.get(),device,stream);
                    declaration.sources.push_back({.tensor=owner.get(),.buffer=BufferId::QFN_QSA_REPLAY,.frontierByteOffset=0,.liveByteOffset=sizeof(std::int32_t)});owners.push_back(std::move(owner));}
                for(auto* t:{&input,&output}){TransferEngine::prepareDeviceInput(t,device,stream);TransferEngine::requireDeviceInput(t,device,stream);}
                GEMMStage stage({.device_id=device,.A=&input,.B=weight.get(),.C=&output,.m=physical,.n=n,.k=k,
                    .force_decode_equivalent_verifier_prefill=grouped,.projection_row_work=declaration,.prepared_ref=ref,.prepared_store=store});
                stage.setGPUStream(stream);checked(stage.validatePreparedWeights(&error),error.c_str());StageWorkspace workspace(stage,device,physical,n,k);
                checked(kernel->prepareFusedProjectionGraphCapture(1),"Query work resources not capture-ready");checked(stage.prepareGraphLaunch(context.get(),stream),"Query work cold prepare failed");
                if(requests>1){const auto req=stage.getWorkspaceRequirements(physical,n,k);std::size_t tables=0;
                    for(const auto& bank:req.buffers)if(bank.name.find("projection_original_rows_")==0){++tables;EXPECT_EQ(bank.size_bytes,std::size_t(requests)*sizeof(ProjectionRequestPublication));}
                    EXPECT_EQ(tables,1u)<<"Arbitrary source owners must use one exact address table";}
                std::vector<QueryWorkBank> banks;
                auto add=[&](const char* name,std::size_t width,std::size_t element,bool blockMajor=false,bool emptyOnly=false){const auto bytes=workspace.value().getBufferSize(name);if(!bytes)return;
                    QueryWorkBank bank{name,width,element,blockMajor,emptyOnly};bank.poison.resize(bytes);bank.observed.resize(bytes);
                    if(element==1)std::fill(bank.poison.begin(),bank.poison.end(),91);
                    else if(element==2){const std::uint16_t poison=0x7fc1;for(std::size_t i=0;i<bytes;i+=2)std::memcpy(bank.poison.data()+i,&poison,2);}
                    else{const std::uint32_t poison=0x7fc12345;for(std::size_t i=0;i<bytes;i+=4)std::memcpy(bank.poison.data()+i,&poison,4);}
                    if(!emptyOnly)checked(bytes>=std::size_t(physical)*width*element,"Query numerical bank lost physical rows");banks.push_back(std::move(bank));};
                add(GemmWorkspaceBuffers::QUANT_A,k,1);add(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE,k/32,4);
                add(GemmWorkspaceBuffers::SUMS_A_BLOCKWISE,k/32,4,device.is_cuda());add(GemmWorkspaceBuffers::FULL_A_FP16,k,2);
                for(const auto* name:{GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL,GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL_BATCHED,GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS,
                    GemmWorkspaceBuffers::CUDA_CONCURRENT_DECODE_GEMV_KPAR_PARTIALS,GemmWorkspaceBuffers::CUDA_NATIVE_VNNI_PREFILL_CANONICAL_KPART_PARTIALS})add(name,1,4,false,true);
                std::vector<float> frameSource(physical*k),expected(output.numel(),-71.0f),actual(output.numel());
                std::vector<std::array<std::int32_t,2>> words(requests);
                checked(backend->synchronizeStream(stream,device.ordinal),"Query work setup fence failed");
                const auto identity=stage.graphCaptureVariantSignature();auto graph=worker.createGraphCapture(stream);CapturedWorkRetirement retirement(*backend,device,stream,graph);
                {ScopedBackendGraphCapture capture(worker,*graph,"actual prepared causal query work");checked(capture.begin(),"Query work capture begin failed");checked(stage.execute(context.get()),"Query work capture failed");capture.finish();}
                checked(graph->instantiate(),"Query work instantiate failed");EXPECT_FALSE(ITensorGemm::VerifierKernelModeScope::active());
                for(int frame=0;frame<6;++frame){SCOPED_TRACE(frame);const int pattern=frame%2;std::vector<bool> needed(physical,false);
                    for(int request=0;request<requests;++request){int frontier=0,live=rows;
                        if(frame==1){frontier=2051-request;live=std::min(rows,request+1);}
                        else if(frame==2||frame==5)frontier=2051+request;
                        else if(frame==3){frontier=2050-2*request;live=request==1?std::max(0,rows-2):rows;}
                        else if(frame==4){frontier=2051;live=0;}
                        words[request]={frontier,live};for(int row=0;row<rows;++row)needed[request*rows+row]=row<live&&frontier+row>=2051;
                        checked(backend->hostToDeviceOnStream(owners[request]->gpu_data_ptr(),words[request].data(),sizeof(words[request]),device.ordinal,stream),"Original request publication failed");}
                    std::fill(frameSource.begin(),frameSource.end(),std::numeric_limits<float>::quiet_NaN());
                    for(int row=0;row<physical;++row)if(needed[row]){std::copy_n(source[pattern].data()+row*k,k,frameSource.data()+row*k);std::copy_n(oracle[pattern].data()+row*n,n,expected.data()+row*n);}
                    checked(backend->hostToDeviceOnStream(input.gpu_data_ptr(),frameSource.data(),input.size_bytes(),device.ordinal,stream),"Query source publication failed");
                    for(auto& bank:banks)checked(backend->hostToDeviceOnStream(workspace.value().getBuffer(bank.name),bank.poison.data(),bank.poison.size(),device.ordinal,stream),"Numerical scratch poison failed");
                    EXPECT_EQ(stage.graphCaptureVariantSignature(),identity);checked(graph->launch(),"Query retained replay failed");
                    checked(backend->deviceToHostOnStream(actual.data(),output.gpu_data_ptr(),output.size_bytes(),device.ordinal,stream),"Query terminal output failed");
                    for(auto& bank:banks)checked(backend->deviceToHostOnStream(bank.observed.data(),workspace.value().getBuffer(bank.name),bank.observed.size(),device.ordinal,stream),"Query scratch observation failed");
                    checked(backend->synchronizeStream(stream,device.ordinal),"Query terminal join failed");
                    EXPECT_EQ(std::memcmp(actual.data(),expected.data(),output.size_bytes()),0)<<format.label;
                    const bool empty=std::none_of(needed.begin(),needed.end(),[](bool x){return x;});
                    for(const auto& bank:banks){if(empty){EXPECT_EQ(bank.observed,bank.poison)<<bank.name;continue;}if(bank.emptyOnly)continue;
                        for(int row=0;row<physical;++row)if(!needed[row])for(std::size_t d=0;d<bank.width;++d){const auto i=(bank.blockMajor?d*physical+row:std::size_t(row)*bank.width+d)*bank.element;
                            EXPECT_EQ(std::memcmp(bank.observed.data()+i,bank.poison.data()+i,bank.element),0)<<bank.name<<" row="<<row;}}
                }
            }
        }
    }).get();
}

}
#ifdef HAVE_CUDA
TEST(GEMMStagePrefixDevice,CUDA){proveM1(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
TEST(GEMMStagePrefixDevice,ROCm){proveM1(DeviceId::rocm(0));}
#endif

#ifdef HAVE_CUDA
TEST(GEMMStageQueryWorkDevice,CUDAOrdinaryM1AllFormats){proveQueryWork(DeviceId::cuda(0),false);}
TEST(GEMMStageQueryWorkDevice,CUDAGroupedOwnersAllFormats){proveQueryWork(DeviceId::cuda(0),true);}
#endif
#ifdef HAVE_ROCM
TEST(GEMMStageQueryWorkDevice,ROCmOrdinaryM1AllFormats){proveQueryWork(DeviceId::rocm(0),false);}
TEST(GEMMStageQueryWorkDevice,ROCmGroupedOwnersAllFormats){proveQueryWork(DeviceId::rocm(0),true);}
#endif
