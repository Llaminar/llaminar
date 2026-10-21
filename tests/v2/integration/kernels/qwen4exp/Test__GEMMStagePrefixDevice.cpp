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
}
#ifdef HAVE_CUDA
TEST(GEMMStagePrefixDevice,CUDA){proveM1(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
TEST(GEMMStagePrefixDevice,ROCm){proveM1(DeviceId::rocm(0));}
#endif
