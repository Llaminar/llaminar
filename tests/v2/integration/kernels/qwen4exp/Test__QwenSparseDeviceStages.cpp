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
#include "QwenSparseSelectionCases.h"
#include "memory/BufferArena.h"
#include "tensors/TensorFactory.h"
#include "../../../mocks/MockMPIContext.h"
#include <cxxabi.h>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <set>
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

namespace
{
/**
 * @brief Inspect exact native functions emitted by the real Indexer stage.
 * @param graph Retained participant-local graph, inspected without executing it.
 * @param device Exact backend that owns the graph.
 * @param rows Immutable stage row capacity.
 * @param blocks Immutable admitted score/pool capacity.
 * @param policy Explicit implementation admitted for this captured row capacity.
 * @return Function identities in semantic role order, never native enumeration order.
 */
std::array<std::uintptr_t,4> inspectSelectorForwarding(const IGPUGraphCapture& graph,DeviceId device,
    int rows,int blocks,QwenSparsePoolSelection policy)
{
    std::string selector;
    switch(policy){
    case QwenSparsePoolSelection::Installed:selector=rows<=attention::kMaxGroupedVerifierAttentionRows?"selectPoolsWide":"selectPools";break;
    case QwenSparsePoolSelection::Bounded256_512:selector="selectPoolsBounded<256, 512>";break;
    case QwenSparsePoolSelection::Bounded512_512:selector="selectPoolsBounded<512, 512>";break;
    case QwenSparsePoolSelection::Bounded1024_1024:selector="selectPoolsBounded<1024, 1024>";break;
    default:throw std::invalid_argument("Forwarding regression received an unknown explicit selector policy");}
    const bool templated=policy==QwenSparsePoolSelection::Bounded256_512 || policy==QwenSparsePoolSelection::Bounded512_512 ||
        policy==QwenSparsePoolSelection::Bounded1024_1024;
    const bool rowCTA=policy!=QwenSparsePoolSelection::Installed || rows<=attention::kMaxGroupedVerifierAttentionRows;
    const std::string space=device.is_cuda()?"llaminar2::qwen4exp::cuda_sparse_detail::":"llaminar2::qwen4exp::rocm_sparse_detail::";
    std::array<std::string,4> names{space+"preparePools",space+"prepareQueries",space+"scorePools",(templated?"void ":"")+space+selector};
    for(auto& name:names)name+="(llaminar2::qwen4exp::QwenSparseIndexer)";
    const std::array<unsigned,4> gridX{unsigned((rows+6)/4),unsigned(rows*4),unsigned((blocks+7)/8),
        unsigned(rowCTA?rows:(rows+7)/8)};
    std::vector<GPUGraphKernelNodeInfo> nodes;std::string detail;
    require(graph.nodeCount()==4 && graph.inspectKernelNodes(nodes,&detail) && nodes.size()==4,"Actual Indexer stage must capture exactly four inspectable native kernels");
    std::array<std::uintptr_t,4> functions{};
    for(const auto& node:nodes){
        require(node.valid() && node.name_resolved && node.nesting_depth==0,"Indexer native node lacks exact flat function identity");
        int status=0;std::unique_ptr<char,decltype(&std::free)> name(abi::__cxa_demangle(node.name.c_str(),nullptr,nullptr,&status),&std::free);
        require(status==0 && name,"Cannot demangle actual Indexer stage native function");
        const auto found=std::find(names.begin(),names.end(),std::string(name.get()));
        require(found!=names.end(),"Stage forwarded a foreign selector/function specialization");const auto role=std::size_t(found-names.begin());
        require(!functions[role],"Stage captured a duplicate native role");functions[role]=node.function_identity;
        // The two existing rotary producers can contain libdevice's deliberate
        // range-reduction stack storage. The canonical compiler spill gate owns
        // that distinction; this test also fixes their function identities across
        // policies. Scoring and the selector must have no private storage.
        require(node.grid_x==gridX[role] && node.grid_y==unsigned(role==2?rows:1) && node.grid_z==1 &&
            node.block_x==unsigned(role<2?128:256) && node.block_y==1 && node.block_z==1 &&
            node.dynamic_shared_memory_bytes==0 && (role<2 || node.local_memory_bytes_per_thread==0),
            ("Stage selector forwarding changed actual native launch/resource geometry: role="+std::to_string(role)+
            " function="+name.get()+" grid="+std::to_string(node.grid_x)+":"+std::to_string(node.grid_y)+":"+std::to_string(node.grid_z)+
            " block="+std::to_string(node.block_x)+":"+std::to_string(node.block_y)+":"+std::to_string(node.block_z)+
            " dynamic_shared="+std::to_string(node.dynamic_shared_memory_bytes)+
            " local="+std::to_string(node.local_memory_bytes_per_thread)).c_str());
    }
    require(std::set<std::uintptr_t>(functions.begin(),functions.end()).size()==4,"Indexer native roles alias one function");return functions;
}

/**
 * @brief Prove common Stage::Indexer forwards every explicit policy into retained native graphs.
 * @param device Exact CUDA/HIP participant; unavailable devices skip explicitly.
 * @details BufferArena owns every mutable native byte. TransferEngine prepares
 * bindings before capture. Terminal test-only transfers inspect complete dirty
 * output capacity and live produced scores; no mirror drives captured execution.
 * Contextual M1/M16 cross8195/8196, mixed rows and long/empty/short/reset
 * under the same retained native function; M33 admits only original policies.
 */
void proveStageSelectorForwarding(DeviceId device)
{
    auto* backend=getBackendFor(device);if(!backend || backend->deviceCount()<=device.ordinal)GTEST_SKIP()<<"Requested backend unavailable";
    if(!hasCPUBackend())initCPUBackend(-1);
    auto execution=IDeviceContext::create(device,1);require(bool(execution),"Selector forwarding device context unavailable");
    auto& worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&]{
        constexpr int tokens=8320,blocks=tokens/4,positions=tokens+17;
        constexpr std::array policies{QwenSparsePoolSelection::Installed,QwenSparsePoolSelection::Bounded256_512,
            QwenSparsePoolSelection::Bounded512_512,QwenSparsePoolSelection::Bounded1024_1024};
        for(int rows:{1,16,33}){
            SCOPED_TRACE(rows);
            llaminar2::test::MockMPIContext mpi;TensorFactory factory(mpi);BufferArena arena(ArenaConfig{.factory=&factory});
            const auto declare=[&](BufferId id,int n,int width,const char* type){require(arena.registerBuffer(id,n,width,type,device),"Selector arena role admission failed");};
            declare(BufferId::QFN_QSA_QUERY,rows,512,"FP32");declare(BufferId::QFN_QSA_RAW_KEY,rows,128,"FP32");
            declare(BufferId::QFN_QSA_PARTIAL_KEYS,4,128,"FP32");declare(BufferId::QFN_QSA_POOLED_KEYS,blocks,128,"FP32");
            declare(BufferId::QFN_QSA_PREPARED_QUERY,rows,512,"FP32");declare(BufferId::QFN_QSA_SCORES,rows,blocks,"FP32");
            declare(BufferId::QFN_QSA_SELECTED,rows,2051,"INT32");declare(BufferId::QFN_QSA_SELECTED_COUNT,rows,1,"INT32");
            declare(BufferId::QFN_QSA_REPLAY,1,3,"INT32");require(arena.allocate(),"Selector arena allocation failed");
            const auto tensor=[&](BufferId id)->TensorBase&{auto* value=dynamic_cast<TensorBase*>(arena.getTensor(id));require(value,"Selector arena role is absent");return *value;};
            auto& query=tensor(BufferId::QFN_QSA_QUERY);auto& raw=tensor(BufferId::QFN_QSA_RAW_KEY);auto& prior=tensor(BufferId::QFN_QSA_PARTIAL_KEYS);
            auto& pools=tensor(BufferId::QFN_QSA_POOLED_KEYS);auto& prepared=tensor(BufferId::QFN_QSA_PREPARED_QUERY);auto& scores=tensor(BufferId::QFN_QSA_SCORES);
            auto& selected=tensor(BufferId::QFN_QSA_SELECTED);auto& counts=tensor(BufferId::QFN_QSA_SELECTED_COUNT);auto& replay=tensor(BufferId::QFN_QSA_REPLAY);
            FP32Tensor gain({128}),cosine({positions,32}),sine({positions,32});
            std::fill_n(gain.mutable_data(),gain.numel(),1.f);std::fill_n(cosine.mutable_data(),cosine.numel(),1.f);std::fill_n(sine.mutable_data(),sine.numel(),0.f);
            for(auto* value:{&query,&raw,&prior,&pools,&prepared,&scores})for(std::size_t i=0;i<value->numel();++i)
                static_cast<float*>(value->raw_mutable_data())[i]=float((i*13+7)%67)/41.f-0.8f;
            for(auto* value:{&selected,&counts,&replay})std::fill_n(static_cast<int*>(value->raw_mutable_data()),value->numel(),-77);
            const std::vector<float> poolSeed(static_cast<const float*>(pools.raw_data()),static_cast<const float*>(pools.raw_data())+pools.numel());
            const std::vector<float> preparedPoison(prepared.numel(),-271.75f),scorePoison(scores.numel(),-8191.25f);
            const std::vector<int> selectedPoison(selected.numel(),-77),countPoison(counts.numel(),-99);
            // Last native owner: its graph/stream retires before arena/weight/source backing.
            Storage storage{device,backend,ExplicitGPUStream(worker.defaultStream()).get(),{}};
            for(TensorBase* value:std::vector<TensorBase*>{&query,&raw,&prior,&pools,&prepared,&scores,&selected,&counts,&replay,&gain,&cosine,&sine}){
                TransferEngine::prepareDeviceInput(value,device,storage.stream);TransferEngine::requireDeviceInput(value,device,storage.stream);}
            struct Result{std::vector<float> pools,prepared,scores;std::vector<int> selected,counts;};
            std::vector<Result> reference;std::set<std::uint64_t> signatures;std::set<std::uintptr_t> selectors;std::array<std::uintptr_t,3> producers{};
            // Replay every retained selector across pool boundaries, mixed live
            // rows and empty requests without rebuilding the graph.
            const std::array cases{std::pair{4096,rows},std::pair{4096,std::min(rows,2)},std::pair{2047,rows},
                std::pair{8194,1},std::pair{8195,1},std::pair{8187,rows},std::pair{8195,std::min(rows,2)},
                std::pair{0,0},std::pair{0,1},std::pair{8195,rows}};
            const auto expectedPolicies=policies.size();
            for(const auto policy:policies){
                SCOPED_TRACE(static_cast<int>(policy));
                Stage::Indexer input{Float(query,BufferId::QFN_QSA_QUERY),Float(raw,BufferId::QFN_QSA_RAW_KEY),Float(prior,BufferId::QFN_QSA_PARTIAL_KEYS),
                    Float(pools,BufferId::QFN_QSA_POOLED_KEYS),Weight(gain),Weight(gain),Weight(cosine),Weight(sine),
                    Float(prepared,BufferId::QFN_QSA_PREPARED_QUERY),Float(scores,BufferId::QFN_QSA_SCORES),
                    Int(selected,BufferId::QFN_QSA_SELECTED),Int(counts,BufferId::QFN_QSA_SELECTED_COUNT),blocks,positions};input.poolSelection=policy;
                Stage stage({.device_id=device,.rowCapacity=rows,.tokenCapacity=tokens,.replay=Int(replay,BufferId::QFN_QSA_REPLAY),.operation=input});
                stage.setGPUStream(storage.stream);const auto signature=stage.graphCaptureVariantSignature();ASSERT_NE(signature,0u);EXPECT_TRUE(signatures.insert(signature).second);
                storage.graph=worker.createGraphCapture(storage.stream);
                {ScopedBackendGraphCapture capture(worker,*storage.graph,"QSA explicit selector stage forwarding");require(capture.begin(),"Selector Stage capture begin failed");
                    require(stage.execute(execution.get()),"Actual Indexer stage capture failed");capture.finish();}
                const auto native=inspectSelectorForwarding(*storage.graph,device,rows,blocks,policy);EXPECT_TRUE(selectors.insert(native[3]).second);
                if(policy==QwenSparsePoolSelection::Installed)std::copy_n(native.begin(),3,producers.begin());else EXPECT_TRUE(std::equal(producers.begin(),producers.end(),native.begin()));
                require(storage.graph->instantiate(),"Selector stage instantiate failed");
                for(std::size_t iteration=0;iteration<cases.size();++iteration){const auto [prefix,live]=cases[iteration];SCOPED_TRACE(iteration);
                    const std::array<int,3> words{prefix,live,0};storage.upload(replay.gpu_data_ptr(),words.data(),sizeof(words));
                    storage.upload(pools.gpu_data_ptr(),poolSeed.data(),pools.size_bytes());storage.upload(prepared.gpu_data_ptr(),preparedPoison.data(),prepared.size_bytes());
                    storage.upload(scores.gpu_data_ptr(),scorePoison.data(),scores.size_bytes());storage.upload(selected.gpu_data_ptr(),selectedPoison.data(),selected.size_bytes());
                    storage.upload(counts.gpu_data_ptr(),countPoison.data(),counts.size_bytes());require(storage.graph->launch(),"Selector Stage retained replay failed");
                    Result result{storage.download<float>(pools.gpu_data_ptr(),pools.numel()),storage.download<float>(prepared.gpu_data_ptr(),prepared.numel()),
                        storage.download<float>(scores.gpu_data_ptr(),scores.numel()),storage.download<int>(selected.gpu_data_ptr(),selected.numel()),storage.download<int>(counts.gpu_data_ptr(),counts.numel())};
                    const auto unchanged=storage.download<int>(replay.gpu_data_ptr(),3);EXPECT_TRUE(std::equal(words.begin(),words.end(),unchanged.begin()));
                    for(int row=0;row<rows;++row){const int visible=prefix+row+1,complete=visible/4;std::vector<std::int32_t> expected;
                        if(row<live){if(complete>512)for(int b=0;b<complete;++b)require(std::isfinite(result.scores[row*blocks+b])&&result.scores[row*blocks+b]>=0.f,"Nonfinite/negative produced score");
                            expected=selection_test::oracle(complete>512?result.scores.data()+row*blocks:nullptr,visible);}
                        ASSERT_EQ(result.counts[row],int(expected.size()));
                        for(int i=0;i<2051;++i)EXPECT_EQ(result.selected[row*2051+i],i<int(expected.size())?expected[std::size_t(i)]:-77);
                        for(int b=0;b<blocks;++b)if(row>=live||complete<=512||b>=complete)EXPECT_EQ(result.scores[row*blocks+b],scorePoison[row*blocks+b]);
                        if(row>=live)for(int d=0;d<512;++d)EXPECT_EQ(result.prepared[row*512+d],preparedPoison[row*512+d]);}
                    if(policy==QwenSparsePoolSelection::Installed)reference.push_back(result);else{const auto& expected=reference[iteration];
                        EXPECT_EQ(result.selected,expected.selected);EXPECT_EQ(result.counts,expected.counts);
                        EXPECT_EQ(std::memcmp(result.pools.data(),expected.pools.data(),pools.size_bytes()),0);
                        EXPECT_EQ(std::memcmp(result.prepared.data(),expected.prepared.data(),prepared.size_bytes()),0);
                        EXPECT_EQ(std::memcmp(result.scores.data(),expected.scores.data(),scores.size_bytes()),0);}
                    EXPECT_EQ(stage.graphCaptureVariantSignature(),signature);ASSERT_FALSE(::testing::Test::HasFailure());}
                EXPECT_EQ(inspectSelectorForwarding(*storage.graph,device,rows,blocks,policy),native);storage.graph.reset();
            }
            EXPECT_EQ(signatures.size(),expectedPolicies);EXPECT_EQ(selectors.size(),expectedPolicies);
        }
    }).get();
}
} // namespace
#ifdef HAVE_CUDA
/** @test Real CUDA Indexer stage policy forwarding is witnessed by native function/grid identity. */
TEST(QwenSparseDeviceStages, CUDASelectorPolicyForwarding){proveStageSelectorForwarding(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
/** @test Real ROCm Indexer stage policy forwarding is witnessed by native function/grid identity. */
TEST(QwenSparseDeviceStages, ROCmSelectorPolicyForwarding){proveStageSelectorForwarding(DeviceId::rocm(0));}
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
