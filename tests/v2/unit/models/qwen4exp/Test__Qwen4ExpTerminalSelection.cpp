/**
 * @file Test__Qwen4ExpTerminalSelection.cpp
 * @brief Construct actual auxiliary terminal graphs from distinct original and replica owners.
 *
 * Published HC widths and real CPU Q8_0 preparations exercise the original
 * graph builder without inference, model loading or accelerators. Only the
 * synthetic vocabulary is reduced to eight entries; this is binding proof,
 * not checkpoint numerical or native capture qualification.
 */
#include "Qwen4ExpMetadataFixture.h"
#include "models/qwen4exp/Qwen4ExpGraph.h"
#include "models/qwen4exp/Qwen4ExpGraphConfigBuilder.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "execution/compute_stages/stages/LMHeadStage.h"
#include "execution/compute_stages/stages/RMSNormStage.h"
#include "../../../utils/PreparedWeightTestHarness.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <utility>
#include <vector>
#include <deque>

using namespace llaminar2;
namespace
{
/** @brief Expose the actual schema context under the same production auxiliary scope. */
class ContextProbe final : public Qwen35MoEGraph
{
public:
    using Qwen35MoEGraph::Qwen35MoEGraph;
    /** @return Current head source after the graph-construction scope has unwound. */
    const WeightBinding *unscopedHead() const { return modelLMHeadBinding(); }
    /** @return Unmodified production tensor selection for one mirrored terminal row. */
    TensorContext terminalContext()
    {
        DecodeReplicatedDenseScope scope(*this, 1);
        return buildTensorContext();
    }
};
/** @brief One owner retains distinct primary/replica sources and all prepared references. */
struct TerminalFixture
{
    static constexpr int hidden=2560, channels=4*hidden, rank=320, vocabulary=8;
    PreparedWeightStore store{ModelContextId{52917}};
    std::deque<WeightBinding> owned;
    ModelWeightBindings primary, replica;
    FP32Tensor trunk{{1,channels}}, normalized{{1,hidden}}, branchNormalized{{1,channels}},
        downRows{{1,rank}}, readLogits{{1,channels}}, logits{{1,vocabulary}};

    /** @brief Prepare only small native source banks; no graph or kernel is executed. */
    TerminalFixture()
    {
        for (auto *weights : {&primary, &replica})
        {
            auto &hc=weights->hyper_connection_head;
            hc.norm=make("output_hc_norm.weight",4,hidden,false);
            hc.down=make("output_hc_down.weight",rank,channels,true);
            hc.up=make("output_hc_up.weight",channels,rank,true);
            weights->lm_head=make("output.weight",vocabulary,hidden,false,true);
            weights->get_layer_weights=[](int) { return LayerWeightBindings{}; };
        }
        // Replicas preserve original logical source identity but have distinct
        // frozen instances, backing tensors and actual prepared store entries.
        for (const auto pair : {std::pair{primary.hyper_connection_head.norm, replica.hyper_connection_head.norm},
                               std::pair{primary.hyper_connection_head.down, replica.hyper_connection_head.down},
                               std::pair{primary.hyper_connection_head.up, replica.hyper_connection_head.up},
                               std::pair{primary.lm_head, replica.lm_head}})
        {
            EXPECT_NE(pair.first->tensor,pair.second->tensor);
            EXPECT_NE(pair.first->binding_id,pair.second->binding_id);
            EXPECT_EQ(pair.first->identity.canonical_name,pair.second->identity.canonical_name);
            EXPECT_EQ(pair.first->identity.logical_id,pair.second->identity.logical_id);
            EXPECT_NE(pair.first->identity.instance_id,pair.second->identity.instance_id);
        }
    }
    /** @brief Retain an immutable original tensor and, for projections, its real prepared engine. */
    const WeightBinding *make(const char *name,int n,int k,bool quantized,bool prepare=false)
    {
        std::shared_ptr<TensorBase> tensor;
        if(quantized)
            tensor=std::make_shared<Q8_0Tensor>(std::vector<size_t>{size_t(n),size_t(k)},
                std::vector<uint8_t>(size_t(n)*size_t(k)/32*34));
        else
            tensor=std::make_shared<FP32Tensor>(std::vector<size_t>{size_t(n),size_t(k)});
        if (!quantized)
            std::fill_n(tensor->mutable_data(),tensor->numel(),1.0f);
        auto binding=test::makePreparedWeightTestBinding(tensor.get(),DeviceId::cpu(),name,store.modelId());
        binding.tensor_owner=std::move(tensor);
        if(quantized||prepare) binding.prepared=store.prepareGemm(binding);
        owned.push_back(std::move(binding));
        return &owned.back();
    }
    /** @return Authenticated HC geometry plus a tiny synthetic terminal vocabulary. */
    GraphConfig config(bool fullDense=false) const
    {
        GraphConfig config;
        Qwen4ExpGraphConfigBuilder::populateFromDirectory(test::qwen4MetadataFixture(false),config);
        config.default_device=DeviceId::cpu();
        config.vocab_size=config.vocab_local=vocabulary;
        config.dense_tp_enabled=true;
        config.dense_tp_decode_replicated=fullDense;
        config.lm_head_column_parallel=!fullDense;
        config.mtp.terminal_head_policy=MTPTerminalHeadPolicy::MirroredFullVocabulary;
        config.qfn_hc_up_activation=QuantizedActivationOperandPrecision::SingleQ8;
        return config;
    }
    /** @brief Inject retained original sources and explicit activation roles into the production builder. */
    void bind(QwenGraphBase &graph,const ModelWeightBindings &selected)
    {
        graph.setWeightBindings(primary);
        graph.setDecodeReplicatedDenseWeightBindings(selected);
        graph.setPreparedWeightStore(&store);
        ModelBuffers buffers;
        buffers.current_hidden=&trunk; buffers.logits=&logits;
        buffers.layer_buffers.normalized=&normalized;
        buffers.layer_buffers.extensions[BufferId::QFN_HC_NORMALIZED]=&branchNormalized;
        buffers.layer_buffers.extensions[BufferId::QFN_HC_DOWN]=&downRows;
        buffers.layer_buffers.extensions[BufferId::QFN_HC_READ_LOGITS]=&readLogits;
        graph.setBuffers(buffers);
    }
    /** @return Actual common graph entrypoint, including policy resolution and prepared stage construction. */
    ComputeGraph build(QwenGraphBase &graph)
    {
        return graph.buildLMHeadGraph(&trunk,&logits,1,DeviceId::cpu());
    }
};
/** @brief Compare every frozen prepared-reference identity field without requiring a production equality API. */
void expectPreparedOwner(const std::optional<PreparedWeightRef> &actual,
                         const std::optional<PreparedWeightRef> &expected)
{
    ASSERT_TRUE(actual);
    ASSERT_TRUE(expected);
    EXPECT_EQ(actual->model_id, expected->model_id);
    EXPECT_EQ(actual->binding_id, expected->binding_id);
    EXPECT_EQ(actual->kind, expected->kind);
    EXPECT_EQ(actual->device, expected->device);
}
/** @brief Assert source pointer and prepared-store ownership on the real generated nodes. */
void expectTerminalOwner(const ComputeGraph &graph,const ModelWeightBindings &selected,
                         PreparedWeightStore &store)
{
    const auto *normNode=graph.getNode("hc_head_branch_norm");
    ASSERT_NE(normNode,nullptr);
    const auto *norm=dynamic_cast<const Qwen4ExpPrimitiveStage *>(normNode->stage.get());
    ASSERT_NE(norm,nullptr);
    const auto *operation=std::get_if<Qwen4ExpPrimitiveStage::BranchRMSNorm>(&norm->getParams().operation);
    ASSERT_NE(operation,nullptr);
    EXPECT_EQ(operation->gain.tensor,selected.hyper_connection_head.norm->tensor);
    for(const auto pair : {std::pair{"hc_head_read_down",selected.hyper_connection_head.down},
                          std::pair{"hc_head_read_up",selected.hyper_connection_head.up}})
    {
        const auto *node=graph.getNode(pair.first);
        ASSERT_NE(node,nullptr);
        const auto *projection=dynamic_cast<const GEMMStage *>(node->stage.get());
        ASSERT_NE(projection,nullptr);
        EXPECT_EQ(projection->getParams().B,pair.second->tensor);
        expectPreparedOwner(projection->getParams().prepared_ref,pair.second->prepared);
        EXPECT_EQ(projection->getParams().prepared_store,&store);
    }
    const auto *headNode=graph.getNode("lm_head");
    ASSERT_NE(headNode,nullptr);
    const auto *head=dynamic_cast<const LMHeadStage *>(headNode->stage.get());
    ASSERT_NE(head,nullptr);
    EXPECT_EQ(head->getParams().lm_head_weight,selected.lm_head->tensor);
    expectPreparedOwner(head->getParams().prepared_ref,selected.lm_head->prepared);
    EXPECT_EQ(head->getParams().prepared_store,&store);
}
} // namespace

/** @test Mirrored and full-dense terminal graphs bind all four actual replica source owners. */
TEST(Qwen4ExpTerminalSelection, ActualHeadGraphSelectsCompleteReplicaAndPrimarySources)
{
    TerminalFixture f;
    for(bool fullDense : {false,true})
    {
        Qwen4ExpGraph graph(f.config(fullDense),nullptr);
        f.bind(graph,f.replica);
        const auto built=f.build(graph);
        expectTerminalOwner(built,f.replica,f.store);
    }
    auto primaryConfig=f.config();
    primaryConfig.lm_head_column_parallel=false;
    Qwen4ExpGraph primary(primaryConfig,nullptr);
    f.bind(primary,f.replica);
    const auto built=f.build(primary);
    expectTerminalOwner(built,f.primary,f.store);
}

/** @test A complete primary mixer and a foreign ordinary norm cannot replace a missing replica HC operand. */
TEST(Qwen4ExpTerminalSelection, MissingReplicaOperandOrTensorRejectsActualHeadGraph)
{
    TerminalFixture f;
    const auto ordinaryNorm=f.make("output_norm.weight",1,TerminalFixture::hidden,false);
    for(bool fullDense : {false,true})
    for(const auto member : {&HyperConnectionWeights<const WeightBinding *>::norm,
                            &HyperConnectionWeights<const WeightBinding *>::down,
                            &HyperConnectionWeights<const WeightBinding *>::up})
    for(bool nullTensor : {false,true})
    {
        auto selected=f.replica;
        auto malformed=*(selected.hyper_connection_head.*member);
        malformed.tensor=nullptr;
        selected.hyper_connection_head.*member=nullTensor ? &malformed : nullptr;
        selected.final_norm=ordinaryNorm;
        Qwen4ExpGraph graph(f.config(fullDense),nullptr);
        f.bind(graph,selected);
        EXPECT_THROW((void)f.build(graph),std::runtime_error);
    }
}

/** @test Schema-context selection does not evaluate an ordinary final norm for a mirrored QFN terminal. */
TEST(Qwen4ExpTerminalSelection, SchemaContextRetainsModelTerminalContract)
{
    TerminalFixture f;
    ContextProbe graph(f.config(),nullptr);
    f.bind(graph,f.replica);
    const auto context=graph.terminalContext();
    EXPECT_EQ(context.model_weights.at("final_norm"),nullptr);
    EXPECT_EQ(context.model_weight_bindings.at("final_norm"),nullptr);
    EXPECT_EQ(context.model_weights.at("lm_head"),f.replica.lm_head->tensor);
    EXPECT_EQ(context.model_weight_bindings.at("lm_head"),f.replica.lm_head);
    EXPECT_EQ(graph.unscopedHead(),f.primary.lm_head);
}

/** @test Ordinary mirrored models still require their selected ordinary normalization tensor. */
TEST(Qwen4ExpTerminalSelection, OrdinaryReplicatedNormRemainsRequired)
{
    TerminalFixture f;
    auto config=f.config(); config.flash_next.reset();
    Qwen35MoEGraph graph(config,nullptr);
    f.bind(graph,f.replica);
    EXPECT_THROW((void)f.build(graph),std::runtime_error);
    auto selected=f.replica;
    selected.final_norm=f.make("output_norm.weight",1,TerminalFixture::hidden,false);
    f.bind(graph,selected);
    const auto built=f.build(graph);
    const auto *node=built.getNode("final_norm"); ASSERT_NE(node,nullptr);
    const auto *norm=dynamic_cast<const RMSNormStage *>(node->stage.get()); ASSERT_NE(norm,nullptr);
    EXPECT_EQ(norm->getParams().gamma,selected.final_norm->tensor);
    const auto *head=dynamic_cast<const LMHeadStage *>(built.getNode("lm_head")->stage.get());
    ASSERT_NE(head,nullptr);
    EXPECT_EQ(head->getParams().lm_head_weight,selected.lm_head->tensor);
}

/** @test Unselected complete replicas cannot hide a missing original terminal mixer operand. */
TEST(Qwen4ExpTerminalSelection, MissingPrimaryMixerRejectsActualHeadGraph)
{
    TerminalFixture f;
    auto config=f.config(); config.lm_head_column_parallel=false;
    for(const auto member : {&HyperConnectionWeights<const WeightBinding *>::norm,
                            &HyperConnectionWeights<const WeightBinding *>::down,
                            &HyperConnectionWeights<const WeightBinding *>::up})
    {
        auto primary=f.primary;
        primary.hyper_connection_head.*member=nullptr;
        Qwen4ExpGraph graph(config,nullptr);
        f.bind(graph,f.replica);
        graph.setWeightBindings(primary);
        EXPECT_THROW((void)f.build(graph),std::runtime_error);
    }
}

/** @test A complete primary vocabulary head cannot replace a missing selected replica head. */
TEST(Qwen4ExpTerminalSelection, MissingReplicaHeadOrTensorRejectsActualHeadGraph)
{
    TerminalFixture f;
    for(bool fullDense : {false,true})
    for(bool nullTensor : {false,true})
    {
        auto selected=f.replica;
        auto malformed=*selected.lm_head;
        malformed.tensor=nullptr;
        selected.lm_head=nullTensor ? &malformed : nullptr;
        Qwen4ExpGraph graph(f.config(fullDense),nullptr);
        f.bind(graph,selected);
        EXPECT_THROW((void)f.build(graph),std::runtime_error);
    }
}
