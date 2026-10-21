/**
 * @file Test__QwenSparseDeviceStages.cpp
 * @brief Retained CUDA/HIP sparse stages consuming real native cache storage.
 *
 * Fixture setup uses TransferEngine for all arena storage and the real cache
 * factory for native banks. Every representation records indexer, selected
 * attention and accepted-tail publication once, then replays changing canonical
 * row parameters and live/accepted counts without replacing any pointer.
 */
#include <gtest/gtest.h>
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/compute_stages/stages/qwen4exp/QwenSparseAttentionStage.h"
#include "execution/local_execution/graph/StageVerifier.h"
#include "tensors/TensorVerification.h"
#include "utils/DebugEnv.h"
#include "kernels/KernelFactory.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "kernels/qwen4exp/QwenSparseMath.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"
#include "QwenSparseNativeCodecFixture.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <limits>

namespace
{
using namespace llaminar2;
using namespace llaminar2::qwen4exp;
using Stage=QwenSparseAttentionStage;
using Float=Qwen4ExpFloatBinding;
using Int=Qwen4ExpIntBinding;
using Weight=Qwen4ExpImmutableWeightBinding;
/** @brief Fail fixture infrastructure immediately without choosing another execution path. */
void require(bool valid,const char *message) { if(!valid) throw std::runtime_error(message); }
/** @brief Setup/terminal boundaries for already-admitted native storage. */
struct Storage
{
    DeviceId device;
    IBackend *backend;
    void *stream;
    std::unique_ptr<IGPUGraphCapture> graph;
    /** @brief Retire captured pointers only after the terminal fixture observation. */
    ~Storage() { (void)backend->synchronizeStream(stream,device.ordinal);graph.reset(); }
    /** @brief Upload test setup/control on the exact ordered stream outside capture. */
    void upload(void *target,const void *source,std::size_t bytes)
    {
        if (!backend->hostToDeviceOnStream(target,source,bytes,device.ordinal,stream))
        {
            // An API failure may follow partial enqueue. Keep the caller's
            // source span alive through exact-stream retirement before throw.
            if (!backend->synchronizeStream(stream,device.ordinal)) std::terminate();
            throw std::runtime_error("QSA stage fixture upload failed");
        }
        if (!backend->synchronizeStream(stream,device.ordinal)) std::terminate();
    }
    /** @brief Publish terminal observation; never called by the captured stage. */
    template<class T> std::vector<T> download(const void *source,std::size_t count)
    {
        std::vector<T> result(count);
        require(backend->deviceToHostOnStream(result.data(),source,count*sizeof(T),device.ordinal,stream),"QSA stage terminal transfer failed");
        require(backend->synchronizeStream(stream,device.ordinal),"QSA stage terminal fence failed");
        return result;
    }
};
/** @brief Independent deterministic source values, bounded for meaningful codec comparisons. */
void fill(FP32Tensor &tensor,int phase)
{
    for(std::size_t index=0;index<tensor.numel();++index)
        tensor.mutable_data()[index]=float((index*13+phase)%67)/41.0f-0.8f;
}
/** @brief Authenticate captured native cache consumption and exact acceptance for one backend. */
void prove(DeviceId device)
{
    auto *backend=getBackendFor(device);
    if(!backend || backend->deviceCount()<=device.ordinal) GTEST_SKIP()<<"Requested backend unavailable";
    auto context=IDeviceContext::create(device,1);
    require(bool(context),"QSA stage context construction failed");
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&] {
        constexpr int rows=3,tokens=16,positions=32,blocks=4;
        TurboQuantContext tq(256,31);
        for(auto precision : {ActivationPrecision::FP32,ActivationPrecision::FP16,ActivationPrecision::BF16,
                              ActivationPrecision::Q8_1,ActivationPrecision::TQ4,ActivationPrecision::TQ8}) {
            SCOPED_TRACE(static_cast<int>(precision));
            Storage storage{device,backend,ExplicitGPUStream(worker.defaultStream()).get(),{}};
            llaminar::v2::kernels::KVCacheConfig config;
            config.precision=precision;config.device=device;config.num_layers=1;config.batch_size=1;
            config.max_seq_len=tokens;config.n_kv_heads=2;config.head_dim=256;config.turboquant_ctx=&tq;
            auto cache=llaminar::v2::kernels::KernelFactory::createKVCache(config);
            require(bool(cache),"QSA real native cache construction failed");
            FP32Tensor query({rows,512}),raw({rows,128}),prior({4,128}),pools({blocks,128}),
                prepared({rows,512}),scores({rows,blocks}),gain({128}),cosine({positions,32}),sine({positions,32}),
                coreQuery({rows,6144}),output({rows,6144}),committed({4,128});
            INT32Tensor control({4+rows*sizeof(attention::AttentionDeviceParams)/4}),selected({rows,2051}),count({rows}),
                partials({rows*24*8*sizeof(QwenSparseAttentionPartial)/4}),observations({rows*24*sizeof(QwenSparseAttentionObservation)/4});
            fill(query,3);fill(raw,5);fill(prior,7);fill(pools,11);fill(coreQuery,13);
            std::fill_n(gain.mutable_data(),gain.numel(),1.0f);
            for(int position=0;position<positions;++position)for(int d=0;d<32;++d) {
                cosine.mutable_data()[position*32+d]=std::cos(float(position+d)*0.013f);
                sine.mutable_data()[position*32+d]=std::sin(float(position+d)*0.013f);
            }
            for(TensorBase *tensor : std::vector<TensorBase *>{&query,&raw,&prior,&pools,&prepared,&scores,&gain,&cosine,&sine,
                    &coreQuery,&output,&committed,&control,&selected,&count,&partials,&observations}) {
                TransferEngine::prepareDeviceInput(tensor,device,storage.stream);
                TransferEngine::requireDeviceInput(tensor,device,storage.stream);
            }
            const auto *rowPointer=reinterpret_cast<const attention::AttentionDeviceParams *>(
                static_cast<const int *>(control.gpu_data_ptr())+4);
            const bool preRotary=precision==ActivationPrecision::Q8_1 || precision==ActivationPrecision::TQ4 || precision==ActivationPrecision::TQ8;
            const attention::AttentionKeyCachePolicy policy{.encoding=preRotary ? attention::AttentionKeyCacheEncoding::PreRotaryDeviceTransform :
                attention::AttentionKeyCacheEncoding::PostRotary,.rope_theta=10000,.partial_rotary_factor=0.25f};
            auto native=cache->acquireSparseAttentionNativeView(0,0,rowPointer,policy,storage.stream);
            qwen4exp::test::NativeCodecFixture key(native.keys.codec,tokens),value(native.values.codec,tokens);
            storage.upload(const_cast<void *>(native.keys.data),key.storage.data(),key.storage.size());
            storage.upload(const_cast<void *>(native.values.data),value.storage.data(),value.storage.size());
            if(native.keys.codec==TensorType::AQ8)storage.upload(const_cast<float *>(native.keys.anchor),key.anchor.data(),key.anchor.size()*4);
            if(native.values.codec==TensorType::TQ4 || native.values.codec==TensorType::TQ8) {
                // Reconstruct with the cache's canonical model/layer/global-head basis.
                for(int token=0;token<tokens;++token)for(int head=0;head<2;++head)for(int d=0;d<256;++d) {
                    double coordinate=0;
                    const auto &rotation=tq.for_layer(0).for_layer(head).rotation();
                    for(int i=0;i<256;++i) {
                        const unsigned code=unsigned(token*19+head*23+i*11)%(native.values.codec==TensorType::TQ4 ? 16u : 256u);
                        coordinate+=double(rotation.matrix[i*256+d])*value.centroids[code]*0.5;
                    }
                    value.decoded[(token*2+head)*256+d]=static_cast<float>(coordinate);
                }
            }
            const Int replay(control,BufferId::QFN_QSA_REPLAY);
            Stage indexer({.device_id=device,.rowCapacity=rows,.tokenCapacity=tokens,.replay=replay,
                .operation=Stage::Indexer{Float(query,BufferId::QFN_QSA_QUERY),Float(raw,BufferId::QFN_QSA_RAW_KEY),
                    Float(prior,BufferId::QFN_QSA_PARTIAL_KEYS),Float(pools,BufferId::QFN_QSA_POOLED_KEYS),Weight(gain),Weight(gain),
                    Weight(cosine),Weight(sine),Float(prepared,BufferId::QFN_QSA_PREPARED_QUERY),Float(scores,BufferId::QFN_QSA_SCORES),
                    Int(selected,BufferId::QFN_QSA_SELECTED),Int(count,BufferId::QFN_QSA_SELECTED_COUNT),blocks,positions}});
            Stage::SelectedAttention attentionOp{{*cache,0,0,policy},Float(coreQuery,BufferId::QFN_QSA_PREPARED_QUERY),
                Int(selected,BufferId::QFN_QSA_SELECTED),Int(count,BufferId::QFN_QSA_SELECTED_COUNT),Float(output,BufferId::ATTN_OUTPUT),
                {},Int(control,BufferId::QFN_QSA_REPLAY,16),Int(partials,BufferId::QFN_QSA_PARTIALS),Int(observations,BufferId::QFN_QSA_OBSERVATIONS)};
            if(preRotary) { attentionOp.keyCosine=Weight(cosine);attentionOp.keySine=Weight(sine);attentionOp.keyPositionCapacity=positions; }
            Stage selectedStage({.device_id=device,.rowCapacity=rows,.tokenCapacity=tokens,.replay=replay,.operation=attentionOp});
            Stage commit({.device_id=device,.rowCapacity=rows,.tokenCapacity=tokens,.replay=replay,
                .operation=Stage::CommitPartial{Float(prior,BufferId::QFN_QSA_PARTIAL_KEYS),Float(raw,BufferId::QFN_QSA_RAW_KEY),
                    Int(control,BufferId::QFN_QSA_REPLAY,12),Float(committed,BufferId::QFN_PLE_CONV_HISTORY)}});
            for(auto *stage : std::array<IComputeStage *,3>{&indexer,&selectedStage,&commit})stage->setGPUStream(storage.stream);
            storage.graph=worker.createGraphCapture(storage.stream);
            {
                ScopedBackendGraphCapture capture(worker,*storage.graph,"QSA native selected stages");
                require(capture.begin(),"QSA stage capture begin failed");
                require(indexer.execute(context.get()),"QSA indexer stage capture failed");
                require(selectedStage.execute(context.get()),"QSA selected stage capture failed");
                require(commit.execute(context.get()),"QSA commit stage capture failed");
                capture.finish();
            }
            EXPECT_EQ(storage.graph->nodeCount(),7u); // Four indexer, two core, one acceptance kernel; zero gathers/copies.
            require(storage.graph->instantiate(),"QSA stage retained executable creation failed");
            // Ordinary CommitPartial has no immutable transaction prior:
            // acceptance zero preserves its destination rather than restoring
            // another bank. Also exercise real nonempty tail writes; prefix3
            // with accepted1 only completes a block and writes no tail at all.
            struct TailCase { int prefix,live,accepted; };
            constexpr std::array<TailCase,10> cases{{
                {3,3,1},{3,1,0},{3,0,0},{3,2,1},{3,3,2},
                {3,3,3},{2,3,1},{5,3,1},{7,3,2},{3,3,0}}};
            for(const auto [prefix,live,accepted] : cases)for(int origin : {0,14}) {
                SCOPED_TRACE(prefix);
                SCOPED_TRACE(live);
                SCOPED_TRACE(accepted);
                SCOPED_TRACE(origin);
                const int positionOrigin=5;
                std::vector<int> words(control.numel(),0);
                words[0]=prefix;words[1]=live;words[2]=positionOrigin;words[3]=accepted;
                auto *rowParams=reinterpret_cast<attention::AttentionDeviceParams *>(words.data()+4);
                for(int row=0;row<rows;++row)rowParams[row]={prefix+row+1,tokens,positionOrigin+prefix+row,0,origin,tokens};
                storage.upload(control.gpu_data_ptr(),words.data(),words.size()*4);
                std::vector<float> poison(rows*6144,-12345.0f);
                storage.upload(output.gpu_data_ptr(),poison.data(),poison.size()*4);
                std::vector<float> expectedTail(512,-24680.0f);
                storage.upload(committed.gpu_data_ptr(),expectedTail.data(),committed.size_bytes());
                require(storage.graph->launch(),"QSA stage retained replay failed");
                const auto actual=storage.download<float>(output.gpu_data_ptr(),rows*6144);
                const auto indices=storage.download<int>(selected.gpu_data_ptr(),rows*2051);
                const auto counts=storage.download<int>(count.gpu_data_ptr(),rows);
                const auto traffic=storage.download<QwenSparseAttentionObservation>(observations.gpu_data_ptr(),rows*24);
                const auto tail=storage.download<float>(committed.gpu_data_ptr(),512);
                std::vector<float> transformed(key.decoded),expected(rows*6144,-12345.0f);
                if(preRotary)for(int logical=0;logical<tokens;++logical)for(int head=0;head<2;++head)for(int d=0;d<64;++d) {
                    const int physical=(origin+logical)%tokens,partner=d<32 ? d+32 : d-32;
                    const auto offset=(physical*2+head)*256;
                    volatile float first=key.decoded[offset+d]*cosine.data()[(positionOrigin+logical)*32+d%32];
                    volatile float second=key.decoded[offset+partner]*(d<32 ? -1.0f : 1.0f)*sine.data()[(positionOrigin+logical)*32+d%32];
                    transformed[offset+d]=first+second;
                }
                QwenSparseReplay hostReplay{prefix,live,positionOrigin};
                QwenSparseAttention reference{.replay=&hostReplay,.query=coreQuery.data(),
                    .kv={.keys={.data=transformed.data(),.token_stride_bytes=2048,.head_stride_bytes=1024},
                         .values=value.reference(),.addressing=QwenSparseKVAddressing::FixedLogicalView,.logical_view={origin,tokens}},
                    .selected=indices.data(),.selected_count=counts.data(),.output=expected.data(),.row_capacity=rows,.token_capacity=tokens};
                cpuQwenSparseAttention(reference);
                for(int row=0;row<live;++row) {
                    EXPECT_EQ(counts[row],prefix+row+1);
                    for(int d=0;d<6144;++d) EXPECT_NEAR(actual[row*6144+d],expected[row*6144+d],4e-6);
                    for(int head=0;head<24;++head) {
                        EXPECT_EQ(traffic[row*24+head].key_elements,std::uint64_t(counts[row])*(preRotary ? 320 : 256));
                        EXPECT_EQ(traffic[row*24+head].value_elements,std::uint64_t(counts[row])*256);
                    }
                }
                EXPECT_TRUE(std::all_of(actual.begin()+live*6144,actual.end(),[](float x){return x==-12345.0f;}));
                if(live>0 && accepted>0) {
                    // Independent oracle: concatenate only the original packed
                    // tail and accepted candidate rows, then drop complete4-row
                    // blocks. Prior rows are packed, never absolute-token indexed.
                    std::vector<float> acceptedKeys(prior.data(),prior.data()+(prefix%4)*128);
                    acceptedKeys.insert(acceptedKeys.end(),raw.data(),raw.data()+accepted*128);
                    const auto completeFloats=(acceptedKeys.size()/(4*128))*(4*128);
                    std::copy(acceptedKeys.begin()+completeFloats,acceptedKeys.end(),expectedTail.begin());
                }
                EXPECT_EQ(std::memcmp(tail.data(),expectedTail.data(),committed.size_bytes()),0);
                // This one comparison proves both authored tail bytes and the
                // untouched suffix, including the entire bank for ordinary0.

            }
            storage.graph.reset(); // Cache and arena lifetimes end only after executable retirement.
        }
    }).get();
}
}
#ifdef HAVE_CUDA
/** @test CUDA retained backend-neutral sparse stages use every real native cache codec. */
TEST(QwenSparseDeviceStages, CUDA) { prove(llaminar2::DeviceId::cuda(0)); }
#endif
#ifdef HAVE_ROCM
/** @test HIP retained backend-neutral sparse stages use every real native cache codec. */
TEST(QwenSparseDeviceStages, ROCm) { prove(llaminar2::DeviceId::rocm(0)); }
#endif

#if LLAMINAR_ASSERTIONS_ACTIVE
namespace
{
/** @brief Restore parsed strict diagnostics after a backend-local fixture run. */
class ScopedObservedValidation final
{
public:
    /** @brief Keep nonempty numerical validation strict; disable failure payload dumps. */
    ScopedObservedValidation() : saved_(debugEnv().validation)
    {
        auto &config=mutableDebugEnv().validation;
        config.validate_buffers=true;
        config.fail_on_zero=true;
        config.fail_on_nan=true;
        config.dump_on_failure=false;
        config.sample_rows=4;
    }
    /** @brief Restore the original parsed policy after native work has retired. */
    ~ScopedObservedValidation() { mutableDebugEnv().validation=saved_; }
private:
    ValidationConfig saved_;
};

/**
 * @brief Capture the actual Indexer once and verify its authored rows on replay.
 * @param device Exact CUDA/HIP participant; no cross-backend substitution.
 *
 * Both outputs and the replay frontier are original tensor subviews. Leading
 * guards and inactive capacity contain NaNs; first1/2/3 tokens have zero pools.
 * Validation must acquire the current native frontier, including after larger
 * replays, without downloading that count. Terminal test-only uploads/fences
 * author control/fault cases outside graph execution.
 */
void proveObservedIndexer(DeviceId device)
{
    auto *backend=getBackendFor(device);
    if (!backend || backend->deviceCount()<=device.ordinal)
        GTEST_SKIP()<<"Requested backend unavailable";
    auto context=IDeviceContext::create(device,1);
    require(bool(context),"Observed Indexer device context unavailable");
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&] {
        ScopedObservedValidation validation;
        constexpr int rows=8,tokens=16,blocks=4,positions=32;
        FP32Tensor query({rows,512}),raw({rows,128}),prior({4,128}),
            pools({blocks+1,128}),prepared({rows+1,512}),scores({rows,blocks}),
            gain({128}),cosine({positions,32}),sine({positions,32});
        INT32Tensor control({4}),selected({rows,2051}),counts({rows});
        // Declared after every tensor: all exception paths retire the exact
        // stream/executable before those original native allocations unwind.
        Storage storage{device,backend,ExplicitGPUStream(worker.defaultStream()).get(),{}};
        for (auto *tensor : {&query,&raw,&prior,&gain,&cosine})
            std::fill_n(tensor->mutable_data(),tensor->numel(),1.f);
        std::fill_n(sine.mutable_data(),sine.numel(),0.f);
        for (TensorBase *tensor : std::vector<TensorBase *>{&query,&raw,&prior,&pools,&prepared,
                &scores,&gain,&cosine,&sine,&control,&selected,&counts})
        {
            TransferEngine::prepareDeviceInput(tensor,device,storage.stream);
            TransferEngine::requireDeviceInput(tensor,device,storage.stream);
        }
        auto stage=std::make_unique<Stage>(Stage::Params{
            .device_id=device,.rowCapacity=rows,.tokenCapacity=tokens,
            .replay=Int(control,BufferId::QFN_QSA_REPLAY,4),
            .operation=Stage::Indexer{
                Float(query,BufferId::QFN_QSA_QUERY),Float(raw,BufferId::QFN_QSA_RAW_KEY),
                Float(prior,BufferId::QFN_QSA_PARTIAL_KEYS),Float(pools,BufferId::QFN_QSA_POOLED_KEYS,128*4),
                Weight(gain),Weight(gain),Weight(cosine),Weight(sine),
                Float(prepared,BufferId::QFN_QSA_PREPARED_QUERY,512*4),Float(scores,BufferId::QFN_QSA_SCORES),
                Int(selected,BufferId::QFN_QSA_SELECTED),Int(counts,BufferId::QFN_QSA_SELECTED_COUNT),blocks,positions}});
        stage->setGPUStream(storage.stream);
        ComputeNode node("native_observed_indexer",std::move(stage),device);
        ASSERT_FALSE(node.stage->allowsZeroOutput());
        storage.graph=worker.createGraphCapture(storage.stream);
        {
            ScopedBackendGraphCapture capture(worker,*storage.graph,"Observed Indexer validation");
            require(capture.begin(),"Observed Indexer capture begin failed");
            require(node.stage->execute(context.get()),"Observed Indexer capture failed");
            capture.finish();
        }
        ASSERT_EQ(storage.graph->nodeCount(),4u);
        require(storage.graph->instantiate(),"Observed Indexer instantiate failed");
        std::vector<float> poolValues(pools.numel()),preparedValues(prepared.numel());
        // Twenty retained replays repeatedly cross full->short->empty without
        // rebind/recapture. The exact native frontier changes on every replay.
        constexpr std::array<std::array<int,2>,8> cases{{
            {0,1},{0,8},{0,2},{0,3},{3,1},{5,2},{7,1},{0,0}}};
        for (int iteration=0;iteration<20;++iteration)
        {
            const auto [prefix,live]=cases[iteration%cases.size()];
            SCOPED_TRACE(iteration);
            SCOPED_TRACE(prefix);
            SCOPED_TRACE(live);
            const std::array<int,4> words{-777,prefix,live,0};
            std::fill(poolValues.begin(),poolValues.end(),std::numeric_limits<float>::quiet_NaN());
            std::fill(preparedValues.begin(),preparedValues.end(),std::numeric_limits<float>::quiet_NaN());
            std::fill_n(poolValues.data()+128,(prefix/4)*128,1.f);
            if (!prefix && live<4)
                std::fill_n(poolValues.data()+128,blocks*128,0.f);
            storage.upload(control.gpu_data_ptr(),words.data(),sizeof(words));
            storage.upload(pools.gpu_data_ptr(),poolValues.data(),pools.size_bytes());
            storage.upload(prepared.gpu_data_ptr(),preparedValues.data(),prepared.size_bytes());
            require(storage.graph->launch(),"Observed Indexer retained launch failed");
            EXPECT_NO_THROW(verifyStageExit(node,-1));
            ASSERT_FALSE(::testing::Test::HasFailure());
        }
        // Valid one-pool output, followed by independently authored bad observed
        // values. No malformed frontier is submitted to the inference kernel.
        std::array<int,4> words{-777,3,1,0};
        storage.upload(control.gpu_data_ptr(),words.data(),sizeof(words));
        require(storage.graph->launch(),"Observed Indexer crossing replay failed");
        EXPECT_NO_THROW(verifyStageExit(node,-1));
        for (float poison : {0.f,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()})
        {
            std::fill_n(poolValues.data()+128,128,poison);
            storage.upload(pools.gpu_data_ptr(),poolValues.data(),pools.size_bytes());
            EXPECT_THROW(verifyStageExit(node,-1),verification::VerificationFailure);
        }
        std::fill_n(poolValues.data()+128,128,1.f);
        storage.upload(pools.gpu_data_ptr(),poolValues.data(),pools.size_bytes());
        for (float poison : {0.f,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()})
        {
            std::fill_n(preparedValues.data()+512,512,poison);
            storage.upload(prepared.gpu_data_ptr(),preparedValues.data(),prepared.size_bytes());
            EXPECT_THROW(verifyStageExit(node,-1),verification::VerificationFailure);
        }
        require(storage.graph->launch(),"Observed Indexer final valid replay failed");
        EXPECT_NO_THROW(verifyStageExit(node,-1));
        for (auto [priorRows,liveRows] : std::array<std::array<int,2>,4>{{{-1,1},{0,-1},{tokens,1},{0,rows+1}}})
        {
            words={-777,priorRows,liveRows,0};
            storage.upload(control.gpu_data_ptr(),words.data(),sizeof(words));
            EXPECT_THROW(verifyStageExit(node,-1),verification::VerificationFailure);
        }
        // Storage's exact-stream retirement precedes every captured tensor.
    }).get();
}
} // namespace
#ifdef HAVE_CUDA
/** @test CUDA native counts distinguish legitimate empty pools from invalid/nonfinite outputs. */
TEST(QwenSparseDeviceStages, CUDAObservedIndexerValidation)
{
    proveObservedIndexer(DeviceId::cuda(0));
}
#endif
#ifdef HAVE_ROCM
/** @test HIP native counts distinguish legitimate empty pools from invalid/nonfinite outputs. */
TEST(QwenSparseDeviceStages, ROCmObservedIndexerValidation)
{
    proveObservedIndexer(DeviceId::rocm(0));
}
#endif
#endif
