/**
 * @file Test__QwenSparseColdSnapshot.cpp
 * @brief Actual cold QSA arena allocation, immutable snapshot source and counted retained replay.
 *
 * No source is manually materialized for diagnostics. The real stage contract
 * and executor allocate the admitted arena first, then freeze snapshot sources.
 * One captured indexer replays full/partial/empty work without pointer changes,
 * storage growth, capacity downloads or removal of any requested output.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/compute_stages/stages/qwen4exp/QwenSparseAttentionStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "snapshots/SnapshotSemanticObservation.h"
#include "tensors/TensorFactory.h"
#include "transfer/TransferEngine.h"
#include "../../../mocks/MockMPIContext.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
using namespace llaminar2;
namespace {
/** @brief Build the actual host-only role owners and exercise native executor setup on one backend. */
void proveColdSnapshot(DeviceId device){
    auto* backend=getBackendFor(device);if(!backend||backend->deviceCount()<=device.ordinal)GTEST_SKIP();
    if(!hasCPUBackend())initCPUBackend(-1);
    auto& worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&]{
        constexpr int rows=9,context=32,blocks=8,positions=32;
        using Stage=QwenSparseAttentionStage;using Float=Qwen4ExpFloatBinding;
        using Int=Qwen4ExpIntBinding;using Weight=Qwen4ExpImmutableWeightBinding;
        auto execution=IDeviceContext::create(device,1);ASSERT_TRUE(execution);
        void* stream=worker.defaultStream();ASSERT_NE(stream,nullptr);
        llaminar2::test::MockMPIContext mpi;TensorFactory factory(mpi);BufferArena arena(ArenaConfig{.factory=&factory});
        const auto declare=[&](BufferId role,int n,int width,const char*type){ASSERT_TRUE(arena.registerBuffer(role,n,width,type,device));};
        declare(BufferId::QFN_QSA_QUERY,rows,512,"FP32");declare(BufferId::QFN_QSA_RAW_KEY,rows,128,"FP32");
        declare(BufferId::QFN_QSA_PARTIAL_KEYS,4,128,"FP32");declare(BufferId::QFN_QSA_POOLED_KEYS,blocks,128,"FP32");
        declare(BufferId::QFN_QSA_PREPARED_QUERY,rows+1,512,"FP32");declare(BufferId::QFN_QSA_SCORES,rows,blocks,"FP32");
        declare(BufferId::QFN_QSA_SELECTED,rows,2051,"INT32");declare(BufferId::QFN_QSA_SELECTED_COUNT,rows,1,"INT32");
        declare(BufferId::QFN_QSA_REPLAY,1,3,"INT32");ASSERT_TRUE(arena.allocate());
        const auto f=[&](BufferId role)->TensorBase&{auto*value=dynamic_cast<TensorBase*>(arena.getTensor(role));if(!value)throw std::logic_error("Missing cold QSA native owner");return *value;};
        auto& query=f(BufferId::QFN_QSA_QUERY);auto& raw=f(BufferId::QFN_QSA_RAW_KEY);auto& partial=f(BufferId::QFN_QSA_PARTIAL_KEYS);
        auto& pools=f(BufferId::QFN_QSA_POOLED_KEYS);auto& prepared=f(BufferId::QFN_QSA_PREPARED_QUERY);
        auto& score=f(BufferId::QFN_QSA_SCORES);auto& selected=f(BufferId::QFN_QSA_SELECTED);
        auto& counts=f(BufferId::QFN_QSA_SELECTED_COUNT);auto& replay=f(BufferId::QFN_QSA_REPLAY);
        for(auto* tensor:{&query,&raw,&partial,&pools,&prepared,&score,&selected,&counts,&replay})EXPECT_EQ(tensor->gpu_data_ptr(),nullptr);
        std::fill_n(static_cast<float*>(query.raw_mutable_data()),query.numel(),1.f);
        for(int row=0;row<rows;++row)std::fill_n(static_cast<float*>(raw.raw_mutable_data())+row*128,128,float(row+1));
        std::fill_n(static_cast<float*>(partial.raw_mutable_data()),partial.numel(),0.f);
        std::fill_n(static_cast<float*>(pools.raw_mutable_data()),pools.numel(),-71.f);
        qwen4exp::QwenSparseReplay initial{0,rows,0};std::memcpy(replay.raw_mutable_data(),&initial,sizeof(initial));
        FP32Tensor gain({128}),cosine({positions,32}),sine({positions,32});
        std::fill_n(gain.mutable_data(),gain.numel(),1.f);std::fill_n(cosine.mutable_data(),cosine.numel(),1.f);
        std::fill_n(sine.mutable_data(),sine.numel(),0.f);
        constexpr std::size_t offset=512*sizeof(float);
        ComputeGraph graph;graph.addNode("layer3_qsa_indexer",std::make_unique<Stage>(Stage::Params{
            .device_id=device,.rowCapacity=rows,.tokenCapacity=context,.replay=Int(replay,BufferId::QFN_QSA_REPLAY),
            .operation=Stage::Indexer{Float(query,BufferId::QFN_QSA_QUERY),Float(raw,BufferId::QFN_QSA_RAW_KEY),
                Float(partial,BufferId::QFN_QSA_PARTIAL_KEYS),Float(pools,BufferId::QFN_QSA_POOLED_KEYS),Weight(gain),Weight(gain),
                Weight(cosine),Weight(sine),Float(prepared,BufferId::QFN_QSA_PREPARED_QUERY,offset),Float(score,BufferId::QFN_QSA_SCORES),
                Int(selected,BufferId::QFN_QSA_SELECTED),Int(counts,BufferId::QFN_QSA_SELECTED_COUNT),blocks,positions}}),device);
        int live=rows;std::map<std::string,std::size_t>observedRows;
        DeviceGraphExecutor executor;executor.setArena(&arena);
        executor.setSnapshotCallback([&](const std::string&,const StageDumpInfo&dump){
            for(const auto& output:dump.outputs){SnapshotSemanticObservation normalized(output);const auto&value=normalized.view();
                observedRows[value.name]=value.rows;
                if(std::string(value.name)=="prepared_query"){
                    EXPECT_EQ(value.rows,std::size_t(live));for(std::size_t i=0;i<value.rows*value.cols;++i)
                        EXPECT_NEAR(static_cast<const float*>(value.data)[i],1.f,2e-5f);
                }else if(std::string(value.name)=="selected"){
                    EXPECT_EQ(value.rows,std::size_t(live));const auto*ids=static_cast<const int*>(value.data);
                    for(int row=0;row<live;++row)for(int column=0;column<2051;++column)
                        EXPECT_EQ(ids[row*2051+column],column<=row ? column:-1);
                }else if(std::string(value.name)=="selected_count"){
                    EXPECT_EQ(value.rows,std::size_t(live));for(int row=0;row<live;++row)EXPECT_EQ(static_cast<const int*>(value.data)[row],row+1);
                }else if(std::string(value.name)=="pooled_keys")EXPECT_EQ(value.rows,std::size_t(live/4));
            }
        });
        // The sole cold materialization comes from declared production roles.
        ASSERT_TRUE(executor.allocateGraphStorageForCapture(graph,execution.get(),"cold_qsa_source"));
        ASSERT_NE(prepared.gpu_data_ptr(),nullptr);ASSERT_EQ(prepared.current_device(),std::optional<DeviceId>(device));
        const auto* sourceBase=prepared.gpu_data_ptr();
        for(auto* input:{&query,&raw,&partial,&pools,&replay})TransferEngine::prepareDeviceInput(input,device,stream);
        for(auto* weight:{&gain,&cosine,&sine})TransferEngine::prepareDeviceInput(weight,device,stream);
        ASSERT_TRUE(executor.prepareGraphStorageForCapture(graph,execution.get(),stream,"cold_qsa_frontier"));
        DeviceGraphExecutor::GraphSnapshotManifest manifest;
        ASSERT_TRUE(executor.prepareSnapshotsForGraphCapture(graph,execution.get(),stream,"cold_qsa_manifest",&manifest));
        ASSERT_EQ(manifest.bound_slot_count,4u);ASSERT_EQ(manifest.storage_allocation_count,1u);
        auto* snapshotBase=manifest.storage_arena->gpu_data_ptr();const auto snapshotCapacity=manifest.storage_capacity_bytes;
        const auto& descriptors=manifest.stage_copies.at("layer3_qsa_indexer").outputs;
        for(const auto&copy:descriptors)if(copy.name=="prepared_query")EXPECT_EQ(copy.source_ptr,static_cast<const std::byte*>(sourceBase)+offset);
        auto capture=worker.createGraphCapture(stream);ASSERT_TRUE(capture);
        {ScopedBackendGraphCapture recording(worker,*capture,"Cold QSA source and counted snapshots");ASSERT_TRUE(recording.begin());
            ASSERT_TRUE(executor.recordSetupGraphMaterialization(graph,execution.get(),nullptr,&manifest));recording.finish();}
        ASSERT_TRUE(capture->instantiate());
        for(int active:{rows,2,0,rows}){
            live=active;initial.live_rows=live;ASSERT_TRUE(backend->hostToDeviceOnStream(replay.gpu_data_ptr(),&initial,sizeof(initial),device.ordinal,stream));
            TransferEngine::publishDeviceWrite(&replay,device,stream);observedRows.clear();
            ASSERT_TRUE(capture->launch());const auto before=manifest.observed_download_bytes;
            ASSERT_TRUE(executor.publishSnapshotsAfterGraphExecution(graph,stream,"cold_qsa_retained_replay",&manifest));
            const std::size_t expected=64+(std::size_t(live)+1)*8+std::size_t(live)*512*4+std::size_t(live)*4+
                std::size_t(live)*(live+1)/2*4+std::size_t(live/4)*128*4;
            EXPECT_EQ(manifest.observed_download_bytes-before,expected);
            EXPECT_EQ(observedRows.size(),4u);EXPECT_EQ(prepared.gpu_data_ptr(),sourceBase);
            EXPECT_EQ(manifest.storage_arena->gpu_data_ptr(),snapshotBase);EXPECT_EQ(manifest.storage_capacity_bytes,snapshotCapacity);
            EXPECT_EQ(manifest.storage_allocation_count,1u);
        }
        capture.reset();ASSERT_TRUE(backend->synchronizeStream(stream,device.ordinal));
    });
}
}
/** @brief Explicit CUDA cold setup/native retained replay; excluded from Unit. */
TEST(QwenSparseColdSnapshot, CUDA){proveColdSnapshot(DeviceId::cuda(0));}
/** @brief Explicit ROCm cold setup/native retained replay; excluded from Unit. */
TEST(QwenSparseColdSnapshot, ROCm){proveColdSnapshot(DeviceId::rocm(0));}
