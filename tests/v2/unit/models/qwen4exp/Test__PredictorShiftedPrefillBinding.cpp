/**
 * @file Test__PredictorShiftedPrefillBinding.cpp
 * @brief Actual bucket/draft arena and frozen-source read/write capability regression.
 *
 * Published native Q8_0 matrix shapes and real host arena roots authenticate
 * setup ownership. The original shape cases keep explicit test descriptors;
 * the HC isolation case prepares its real native HC engines without executing
 * them. Neither is advertised as native capture evidence. A coherent rebuilt core
 * owns the real source/layout and public body-entry validators under test.
 * CPU arena roots use an explicit model-free fixture admission authority.
 */
#include "../../../utils/CPUPreparedStoreTestAdmission.h"
#include "../../../utils/CPUActivationArenaFixture.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPredictorResidualBinding.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>
#include <deque>
#include "Qwen4ExpSidecarFixture.h"
#include "../../../utils/Qwen4ExpPlacementFixture.h"
#include "../../../utils/PreparedWeightTestHarness.h"
#include "models/qwen4exp/Qwen4ExpGraphConfigBuilder.h"
#include "models/qwen4exp/Qwen4ExpBufferLayout.h"
#include "models/qwen4exp/Qwen4ExpSidecarManifest.h"
#include "models/qwen35moe/Qwen35MoESchema.h"
#include "models/qwen35moe/Qwen35MoEGraph.h"
#include "execution/local_execution/graph/GraphResolver.h"
#include "memory/BufferArena.h"
#include "loaders/PreparedWeightStore.h"
#include "tensors/Tensors.h"
#include "../../../utils/Qwen4ExpHCQueryFixture.h"
using namespace llaminar2;
namespace {
/**
 * @brief Real native-format source shapes and setup-only prepared identities.
 *
 * Original Q8_0 host blocks retain published projection/HC dimensions. No
 * weights are prepared and no kernel is executed: the explicit test registry
 * supplies only the frozen metadata consumed by fromFrozen. One fixture owns
 * these roughly70MiB native tensors for all mutations, never giant expert banks.
 */
struct FrozenResidualFixture
{
    test::Qwen4ExpSidecarFixture source;
    GraphConfig config;
    PreparedWeightStore store{ModelContextId{88119}};
    std::deque<WeightBinding> bindings;
    MTPDepthWeightBindings weights;
    LayerGraphSourceIdentity identity;

    /** @brief Admit the actual source48 policy and unchanged native matrix extents. */
    FrozenResidualFixture()
        : identity(LayerGraphSourceIdentity::qwen4ExpSharedPredictor(48,
            Qwen4ExpSidecarManifest(source.main,source.predictor).predictorMetadata(),48,0,0))
    {
        config.default_device=DeviceId::cpu();
        config.max_activation_rows=512;config.max_request_count=1;config.max_seq_len=4096;
        Qwen4ExpGraphConfigBuilder::populateFromDirectory(source.main,config);
        config.mtp.enabled=true;config.mtp.draft_tokens=1;config.mtp.graph_capacity_draft_tokens=1;
        config.mtp.max_request_batch=1;
        config.local_n_heads=config.n_heads;config.local_n_kv_heads=config.n_kv_heads;
        config.head_start=0;
        test::admitQwen4ExpTestTable(config);
        weights.depth_index=0;weights.source_layer_index=48;
        auto &f=weights.fa_block;
        f.wq=matrix("attn_q",12288,2560);f.wk=matrix("attn_k",512,2560);f.wv=matrix("attn_v",512,2560);
        f.wo=matrix("attn_output",2560,6144);
        f.shared_expert_gate=matrix("ffn_gate_shexp",640,2560);
        f.shared_expert_up=matrix("ffn_up_shexp",640,2560);
        f.shared_expert_down=matrix("ffn_down_shexp",2560,640);
        for(const auto role:{std::pair{"hc_attn",&f.flash_next.attention},std::pair{"hc_ffn",&f.flash_next.ffn}})
        {
            role.second->norm=matrix(std::string(role.first)+"_norm",4,2560,false);
            role.second->down=matrix(std::string(role.first)+"_down",320,10240);
            role.second->up=matrix(std::string(role.first)+"_up",10240,320);
            role.second->inject=matrix(std::string(role.first)+"_inject",4,10240);
        }
    }
    /**
     * @brief Retain one real native source tensor and a frozen test-only descriptor.
     * @param role Original source48 tensor suffix.
     * @param n Native output rows.
     * @param k Native input columns.
     * @param quantized Q8_0 matrix, or original FP32 normalization gain.
     * @return Stable owned binding; no executable preparation is advertised.
     */
    const WeightBinding *matrix(const std::string &role,size_t n,size_t k,bool quantized=true)
    {
        std::shared_ptr<TensorBase> tensor;
        if(quantized)tensor=std::make_shared<Q8_0Tensor>(std::vector<size_t>{n,k},std::vector<uint8_t>(n*k/32*34));
        else tensor=std::make_shared<FP32Tensor>(std::vector<size_t>{n,k});
        auto binding=test::makePreparedWeightTestBinding(tensor.get(),DeviceId::cpu(),"blk.48."+role+".weight",store.modelId());
        binding.tensor_owner=std::move(tensor);
        binding.prepared=store.registerPreparedForTest(binding,PreparedWeightKind::CpuPackedGemm,DeviceId::cpu());
        bindings.push_back(std::move(binding));return &bindings.back();
    }
    /** @brief Prepare only original native HC projections for the actual isolation composer, never fake executable refs. */
    void prepareHCMixers()
    {
        std::vector<const TensorBase *> matrices;
        for (const auto &binding : bindings)
            if (binding.identity.canonical_name.find(".hc_") != std::string::npos && binding.tensor->native_type() == TensorType::Q8_0)
                matrices.push_back(binding.tensor);
        test::admitCPUPreparedStoreForTest(store, matrices);
        for(auto &binding:bindings)
            if(binding.identity.canonical_name.find(".hc_")!=std::string::npos &&
               binding.tensor->native_type()==TensorType::Q8_0)
                binding.prepared=store.prepareGemm(binding);
    }
    /** @return Exact frozen full-participant layout; this reads metadata only. */
    Qwen4ExpPredictorProjectionLayout layout() const
    {
        const auto policy=resolveMTPShiftedKVHeadLayout(config.mtp,config.dense_tp_enabled,1);
        return Qwen4ExpPredictorProjectionLayout::fromFrozen(policy,config,weights,DeviceId::cpu());
    }
    /** @return Actual widened schema allocation for one requested existing role. */
    std::vector<size_t> shape(const std::string &name) const
    {
        auto schema=Qwen35MoESchemaFactory{}.createSchema();
        applyQwen4ExpBufferLayout(schema,config);
        GraphResolverConfig resolver;
        resolver.seq_len=512;resolver.d_model=2560;resolver.n_heads=resolver.local_n_heads=24;
        resolver.n_kv_heads=resolver.local_n_kv_heads=2;resolver.head_dim=256;resolver.default_device=DeviceId::cpu();
        resolver.custom_formulas["mtp_kv_prefill_rows"]=512;
        resolver.custom_formulas["mtp_target_query_rows"]=2;
        resolver.custom_formulas["terminal_logit_row_capacity"] = 2;
        resolver.custom_formulas["mtp_kv_dim"]=512;
        resolver.custom_formulas["mtp_q_dim"]=6144;
        applyQwen4ExpBufferGeometry(resolver,config);
        for(const auto &spec:schema.layer_buffers)if(spec.name==name)return BufferAllocator::resolve(spec,resolver).shape;
        throw std::logic_error("Missing actual predictor schema role");
    }
};
/** @brief Expose only production shared-body entrypoints for setup-role rejection proofs. */
class ResidualBodyProbe final : public Qwen35MoEGraph
{
public:
    using Qwen35MoEGraph::Qwen35MoEGraph;
    using Qwen35Graph::buildMTPFAKVCacheAppendGraphForLayer;
    using Qwen35Graph::buildFAAttentionGraphForLayer;
    using Qwen35MoEGraph::buildFFNGraphForLayer;
    /**
     * @brief Compose the actual predictor HC while an intentionally incompatible main input remains scoped.
     * @param graph Participant-local HC destination.
     * @param input Enclosing main input; its query owner must not replace predictor rows.
     * @param buffers Existing predictor activation roles.
     * @param predictor Authenticated source48/read-only rows and projection layout.
     */
    void appendPredictorHC(ComputeGraph &graph,const ForwardInput &input,ActivationBuffers &buffers,
        const Qwen4ExpPredictorResidualBinding &predictor)
    {
        ForwardInputConstructionScope invocation(*this,input);
        PrefillCollectiveRowsScope prefill(*this,input);
        (void)addBranchBlockRead(graph,"mtp0_",buffers,48,predictor.rows().tokens().physicalRows(),
            DeviceId::cpu(),ResidualBlock::Attention,&predictor);
    }
};
/** @brief Require the actual invocation guard, excluding unrelated missing-cache rejection. */
template<class Work> void rejectsInvocation(Work work)
{
    try { work();FAIL()<<"Expected predictor invocation rejection"; }
    catch(const std::invalid_argument &e){EXPECT_NE(std::string(e.what()).find("another predictor invocation"),std::string::npos)<<e.what();}
}
}

/** @test Actual M512 shifted roots coexist with unchanged compact Draft capacity and no dummy write capability. */
TEST(PredictorShiftedPrefillBinding, BucketArenaHasReadOwnershipWithoutDraftWrites)
{
    FrozenResidualFixture f;
    EXPECT_EQ(f.shape("mtp_embedding"),(std::vector<size_t>{512,2560}));
    EXPECT_EQ(f.shape("mtp_norm_hidden"),(std::vector<size_t>{512,10240}));
    EXPECT_EQ(f.shape("mtp_norm_embedding"),(std::vector<size_t>{512,2560}));
    EXPECT_EQ(f.shape("mtp_concat"),(std::vector<size_t>{512,20480}));
    EXPECT_EQ(f.shape("mtp_k"),(std::vector<size_t>{512,512}));
    EXPECT_EQ(f.shape("mtp_v"),(std::vector<size_t>{512,512}));
    const auto projected=f.shape("mtp_projected"),terminal=f.shape("mtp_hidden");
    ASSERT_EQ(projected,(std::vector<size_t>{512,10240}));
    ASSERT_EQ(terminal,(std::vector<size_t>{2,10240}));
    BufferArena arena{::llaminar2::test::cpuArenaFixtureConfig()};
    ASSERT_TRUE(arena.registerBuffer(BufferId::MTP_PROJECTED,projected[0],projected[1],"FP32",DeviceId::cpu()));
    ASSERT_TRUE(arena.registerBuffer(BufferId::MTP_HIDDEN,terminal[0],terminal[1],"FP32",DeviceId::cpu()));
    ASSERT_TRUE(arena.allocate());
    const auto before=arena.stats();
    const auto *intermediate=arena.getTensor(BufferId::MTP_PROJECTED),*recursive=arena.getTensor(BufferId::MTP_HIDDEN);
    std::vector<int> live{512};
    const auto rows=Qwen4ExpPredictorRows::admit(1,512,DeviceId::cpu(),&live,nullptr);
    // Old construction demanded a nonexistent bucket-wide recursive output.
    EXPECT_THROW((void)Qwen4ExpPredictorTrunkTarget::fromArena(arena,Qwen4ExpPredictorTrunk::Terminal,
        512,10240,DeviceId::cpu()),std::invalid_argument);
    const auto read=Qwen4ExpPredictorResidualBinding::fromArena(Qwen4ExpPredictorInvocation::ShiftedPrefill,
        f.identity,f.weights,rows,f.layout(),arena,10240);
    EXPECT_THROW((void)Qwen4ExpPredictorResidualBinding::fromArena(Qwen4ExpPredictorInvocation::Draft,
        f.identity,f.weights,rows,f.layout(),arena,10240),std::invalid_argument);
    EXPECT_THROW((void)Qwen4ExpPredictorResidualBinding::fromArena(static_cast<Qwen4ExpPredictorInvocation>(99),
        f.identity,f.weights,rows,f.layout(),arena,10240),std::invalid_argument);
    EXPECT_EQ(read.invocation(),Qwen4ExpPredictorInvocation::ShiftedPrefill);
    EXPECT_EQ(&read.weights(),&f.weights);EXPECT_EQ(read.rows().tokens().physicalRows(),512);
    EXPECT_EQ(read.rows().tokens().countOwner(),live.data());
    EXPECT_EQ(read.rows().branches().countOwner(),live.data());
    EXPECT_EQ(std::get<Qwen4ExpNativeIntInput>(read.rows().primitives().liveRows->value).data,live.data());
    read.requireLayerSource(toLegacyLayerWeights(f.weights.fa_block));
    live[0]=511;
    EXPECT_THROW((void)Qwen4ExpPredictorResidualBinding::forShiftedPrefill(f.identity,f.weights,rows,f.layout()),std::invalid_argument);
    live[0]=512;
    EXPECT_NO_THROW(read.requireInvocation(Qwen4ExpPredictorInvocation::ShiftedPrefill));
    EXPECT_THROW((void)read.attentionOutput(),std::logic_error);
    EXPECT_THROW((void)read.ffnOutput(),std::logic_error);
    const auto draft=Qwen4ExpPredictorResidualBinding::fromArena(Qwen4ExpPredictorInvocation::Draft,
        f.identity,f.weights,Qwen4ExpPredictorRows::admit(1,1,DeviceId::cpu(),nullptr,nullptr),
        f.layout(),arena,10240);
    EXPECT_EQ(draft.invocation(),Qwen4ExpPredictorInvocation::Draft);
    EXPECT_EQ(draft.attentionOutput().binding().tensor,intermediate);
    EXPECT_EQ(draft.ffnOutput().binding().tensor,recursive);
    EXPECT_EQ(arena.getTensor(BufferId::MTP_PROJECTED),intermediate);
    EXPECT_EQ(arena.getTensor(BufferId::MTP_HIDDEN),recursive);
    EXPECT_EQ(arena.stats().total_buffers,before.total_buffers);
    EXPECT_EQ(arena.stats().total_bytes,before.total_bytes);
    EXPECT_EQ(recursive->shape(),terminal);
    ResidualBodyProbe probe(f.config);
    ActivationBuffers buffers;
    const auto layer=toLegacyLayerWeights(f.weights.fa_block);
    const auto identity=FullAttentionLayerIdentity::qwen4ExpSharedPredictor(f.identity);
    rejectsInvocation([&]{(void)probe.buildFAAttentionGraphForLayer(layer,buffers,identity,512,1,nullptr,nullptr,
        nullptr,DeviceId::cpu(),nullptr,nullptr,"mtp0_",&read);});
    rejectsInvocation([&]{(void)probe.buildFFNGraphForLayer(layer,buffers,f.identity,512,1,DeviceId::cpu(),
        nullptr,nullptr,nullptr,&read);});
    rejectsInvocation([&]{(void)probe.buildMTPFAKVCacheAppendGraphForLayer(layer,buffers,identity,1,1,nullptr,
        nullptr,nullptr,nullptr,DeviceId::cpu(),"mtp0_",0,&draft);});
}

/** @test Explicit predictor rows remain independent from an enclosing padded/multi-request main query under either HC policy. */
TEST(PredictorShiftedPrefillBinding, HCCountFixCannotCaptureEnclosingMainAuthority)
{
    FrozenResidualFixture f;
    f.prepareHCMixers();
    std::vector<int> predictor_live{1};
    const auto predictor=Qwen4ExpPredictorResidualBinding::forShiftedPrefill(f.identity,f.weights,
        Qwen4ExpPredictorRows::admit(1,1,DeviceId::cpu(),&predictor_live,nullptr),f.layout());
    test::HCQueryActivations activation(1);
    const std::vector<int> main_live{0,0};
    ForwardInput main;main.device=DeviceId::cpu();main.batch_size=2;main.seq_len=512;
    main.real_seq_len=9;main.sequence_lengths=&main_live;
    for(const auto policy:{QuantizedActivationOperandPrecision::SingleQ8,QuantizedActivationOperandPrecision::TwoTermQ8})
    {
        auto config=f.config;config.qfn_hc_up_activation=policy;
        ResidualBodyProbe probe(config);probe.setPreparedWeightStore(&f.store);
        ComputeGraph graph;
        ASSERT_NO_THROW(probe.appendPredictorHC(graph,main,activation.buffers,predictor));
        for(const auto suffix:{"read_down","read_up"})
        {
            const auto &stage=test::hcQueryStage(graph,std::string("mtp0_hc_attn_")+suffix);
            const auto &params=stage.getParams();
            ASSERT_TRUE(params.verifier_row_range);
            EXPECT_EQ(params.verifier_row_range->countOwner(),predictor_live.data());
            EXPECT_NE(params.verifier_row_range->countOwner(),main_live.data());
            EXPECT_EQ(params.verifier_row_range->physicalRows(),1);
            EXPECT_TRUE(params.force_decode_equivalent_verifier_prefill);
            EXPECT_EQ(params.quantized_operand_precision,std::string_view(suffix)=="read_up"?policy:
                QuantizedActivationOperandPrecision::SingleQ8);
        }
    }
}
