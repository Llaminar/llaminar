/**
 * @file Test__Qwen4ExpHyperconnectionBuilder.cpp
 * @brief Real CPU prepared-projection proof of reusable GR graph composition.
 *
 * Tiny production FP32 GEMMs and native branch stages execute the complete
 * read/block/write graph, compared to independently evaluated checkpoint
 * equations. Structural regressions reject invalid prepared identities,
 * lifetime aliases, graph-name conflicts and repeated paired-write use before
 * graph mutation. No checkpoint or accelerator is required.
 */
#include <gtest/gtest.h>
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpHyperconnectionBuilder.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "loaders/PreparedWeightStore.h"
#include "tensors/Tensors.h"
#include "utils/OpenMPUtils.h"
#include "../../../utils/PreparedWeightTestHarness.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

using namespace llaminar2;
namespace
{
using Builder=Qwen4ExpHyperconnectionBuilder;
using Stage=Qwen4ExpPrimitiveStage;
using Float=Qwen4ExpFloatBinding;
using Weight=Qwen4ExpImmutableWeightBinding;
/** @brief Small real model-weight and activation fixture with complete prepared identities. */
struct Fixture
{
    static constexpr int rows=3,hidden=8,rank=4,channels=4*hidden;
    PreparedWeightStore store{ModelContextId{99173}};
    FP32Tensor gain{{4,hidden},DeviceId::cpu()},down{{rank,channels},DeviceId::cpu()},
        up{{channels,rank},DeviceId::cpu()},inject{{4,channels},DeviceId::cpu()};
    FP32Tensor embedding{{rows,hidden},DeviceId::cpu()},trunk{{rows,channels},DeviceId::cpu()},
        normalized{{rows,channels},DeviceId::cpu()},bottleneck{{rows,rank},DeviceId::cpu()},
        activated{{rows,rank},DeviceId::cpu()},logits{{rows,channels},DeviceId::cpu()},mixed{{rows,hidden},DeviceId::cpu()},
        block{{rows,hidden},DeviceId::cpu()},injectionLogits{{rows,4},DeviceId::cpu()},output{{rows,channels},DeviceId::cpu()};
    WeightBinding downBinding,upBinding,injectBinding;
    /** @brief Initialize finite fixtures and use production CPU weight preparation. */
    Fixture()
    {
        for(int i=0;i<channels;++i)gain.mutable_data()[i]=0.5f+static_cast<float>(i/hidden)*0.25f;
        for(TensorBase *tensor:std::vector<TensorBase *>{&down,&up,&inject})
            for(std::size_t index=0;index<tensor->numel();++index)
                tensor->mutable_data()[index]=static_cast<float>(static_cast<int>(index%11)-5)/37.0f;
        for(std::size_t index=0;index<embedding.numel();++index)
            embedding.mutable_data()[index]=static_cast<float>(static_cast<int>(index%13)-6)/11.0f;
        downBinding=prepare(down,"mixer.down.weight");
        upBinding=prepare(up,"mixer.up.weight");
        injectBinding=prepare(inject,"mixer.inject.weight");
    }
    /** @brief Prepare an exact immutable matrix via the existing test identity helper. */
    WeightBinding prepare(TensorBase &tensor,const char *name)
    {
        auto binding=test::makePreparedWeightTestBinding(&tensor,DeviceId::cpu(),name,store.modelId());
        binding.prepared=store.prepareGemm(binding);
        return binding;
    }
    /** @brief Return mandatory live geometry and the explicit projection arithmetic policy. */
    Qwen4ExpHyperconnectionContext context(const std::string &prefix="hc.")
    {
        return {DeviceId::cpu(),nullptr,prefix,{rows,{}},hidden,rank,1e-6f,
                Qwen4ExpProjectionArithmetic::Ordinary,{}};
    }
    /** @brief Resolve typed read roles from the exact production prepared store. */
    Qwen4ExpHyperconnectionReadWeights readWeights()
    {
        return {Weight(gain),Qwen4ExpPreparedProjection(downBinding,store,DeviceId::cpu()),
                Qwen4ExpPreparedProjection(upBinding,store,DeviceId::cpu())};
    }
    /** @brief Bind paired read/write identities from this same mixer. */
    Qwen4ExpHyperconnectionBlockWeights blockWeights()
    {
        return {readWeights(),Qwen4ExpPreparedProjection(injectBinding,store,DeviceId::cpu())};
    }
    /** @brief Return independent caller-admitted arena identities. */
    Qwen4ExpHyperconnectionReadBuffers buffers()
    {
        return {Float(trunk,BufferId::HIDDEN_STATE),Float(normalized,BufferId::NORMALIZED),
            Float(bottleneck,BufferId::Q_PROJ),Float(activated,BufferId::K_PROJ),
            Float(logits,BufferId::GATE_PROJ),Float(mixed,BufferId::ATTN_OUTPUT)};
    }
    /** @brief Return block-result/injection/trunk-output identities. */
    Qwen4ExpHyperconnectionWriteBuffers writeBuffers()
    {
        return {Float(block,BufferId::FFN_OUTPUT),Float(injectionLogits,BufferId::V_PROJ),Float(output,BufferId::RESIDUAL)};
    }
    /** @brief Seed the trunk with the real production expansion stage. */
    void seed(ComputeGraph &graph)
    {
        graph.addNode("seed",std::make_unique<Stage>(Stage::Params{.rows={rows,{}},
            .operation=Stage::ResidualExpand{hidden,Float(embedding,BufferId::FFN_NORMALIZED),
                                                   Float(trunk,BufferId::HIDDEN_STATE)}}),DeviceId::cpu());
    }
};
/** @brief Independent double sigmoid for the checkpoint-equation oracle. */
double sigmoid(double x){return 1/(1+std::exp(-x));}
} // namespace

/** @test Complete read/block/write composition reuses real projections and checkpoint scaling. */
TEST(Qwen4ExpHyperconnectionBuilder, RealCPUPreparedGraphMatchesIndependentEquations)
{
    llaminar::v2::ThreadCountGuard threads(1);
    Fixture f;
    ComputeGraph graph;f.seed(graph);
    auto read=Builder::appendRead(graph,f.context(),f.blockWeights(),f.buffers(),{"seed"});
    graph.addNode("block",std::make_unique<Stage>(Stage::Params{.rows={Fixture::rows,{}},
        .operation=Stage::ResidualDown{Fixture::hidden,Float(f.mixed,BufferId::ATTN_OUTPUT),
                                                     Float(f.block,BufferId::FFN_OUTPUT)}}),DeviceId::cpu());
    graph.addDependency("block",read.terminal());
    const auto terminal=Builder::appendWrite(graph,std::move(read),f.writeBuffers(),{"block"});
    EXPECT_EQ(terminal,"hc.write");EXPECT_EQ(graph.size(),9u);
    const auto *down=dynamic_cast<GEMMStage *>(graph.getNode("hc.read_down")->stage.get());
    const auto *up=dynamic_cast<GEMMStage *>(graph.getNode("hc.read_up")->stage.get());
    const auto *inject=dynamic_cast<GEMMStage *>(graph.getNode("hc.write_projection")->stage.get());
    ASSERT_NE(down,nullptr);ASSERT_NE(up,nullptr);ASSERT_NE(inject,nullptr);
    EXPECT_EQ(down->getParams().k,Fixture::channels);EXPECT_EQ(down->getParams().n,Fixture::rank);
    EXPECT_EQ(up->getParams().k,Fixture::rank);EXPECT_EQ(up->getParams().n,Fixture::channels);
    EXPECT_EQ(inject->getParams().n,4);EXPECT_EQ(inject->getParams().alpha,1.0f);
    EXPECT_EQ(inject->getParams().prepared_ref->binding_id,f.injectBinding.prepared->binding_id);
    CPUDeviceContext context(DeviceId::cpu(),1);
    for(auto *stage:graph.getExecutionStages())ASSERT_TRUE(stage->execute(&context));
    for(int row=0;row<Fixture::rows;++row)
    {
        std::vector<double> norm(Fixture::channels),activation(Fixture::rank),readLogits(Fixture::channels),mixed(Fixture::hidden);
        double squareSum=0;
        for(int channel=0;channel<Fixture::hidden;++channel)
        {
            const double value=f.embedding.data()[row*Fixture::hidden+channel];squareSum+=value*value;
        }
        const double inverse=1/std::sqrt(squareSum/Fixture::hidden+1e-6);
        for(int branch=0;branch<4;++branch)
            for(int channel=0;channel<Fixture::hidden;++channel)
                norm[branch*Fixture::hidden+channel]=f.embedding.data()[row*Fixture::hidden+channel]*inverse*
                    f.gain.data()[branch*Fixture::hidden+channel];
        for(int output=0;output<Fixture::rank;++output)
        {
            double sum=0;for(int k=0;k<Fixture::channels;++k)sum+=f.down.data()[output*Fixture::channels+k]*norm[k];
            activation[output]=(sum/4)*sigmoid(sum/4);
        }
        for(int output=0;output<Fixture::channels;++output)
            for(int k=0;k<Fixture::rank;++k)readLogits[output]+=f.up.data()[output*Fixture::rank+k]*activation[k];
        for(int channel=0;channel<Fixture::hidden;++channel)
            for(int branch=0;branch<4;++branch)
                mixed[channel]+=norm[branch*Fixture::hidden+channel]*sigmoid(readLogits[branch*Fixture::hidden+channel])/4;
        for(int branch=0;branch<4;++branch)
        {
            double sum=0;for(int k=0;k<Fixture::channels;++k)sum+=f.inject.data()[branch*Fixture::channels+k]*norm[k];
            const double injection=2*sigmoid(sum/4);
            for(int channel=0;channel<Fixture::hidden;++channel)
            {
                const double scaled=mixed[channel]/4;
                const double expected=f.embedding.data()[row*Fixture::hidden+channel]+injection*scaled*sigmoid(scaled);
                EXPECT_NEAR(f.output.data()[(row*4+branch)*Fixture::hidden+channel],expected,2e-5);
            }
        }
    }
    EXPECT_THROW(Builder::appendWrite(graph,std::move(read),f.writeBuffers(),{"block"}),std::invalid_argument);
}

/** @test Final mixers create read-only graphs and explicit verifier policy reaches all prepared projections. */
TEST(Qwen4ExpHyperconnectionBuilder, FinalReadAndVerifierPolicyRemainDeclarative)
{
    llaminar::v2::ThreadCountGuard threads(1);
    Fixture f;ComputeGraph graph;f.seed(graph);
    auto context=f.context("final.");context.arithmetic=Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier;
    context.projectionRows=DeviceRowRange::fullyActive(Fixture::rows);
    auto buffers=f.buffers();buffers.activatedDown=buffers.down;
    EXPECT_EQ(Builder::appendFinalRead(graph,context,f.readWeights(),buffers,{"seed"}),"final.read");
    EXPECT_EQ(graph.size(),6u);EXPECT_EQ(graph.getNode("final.write"),nullptr);
    for(const auto *name:{"final.read_down","final.read_up"})
    {
        auto *gemm=dynamic_cast<GEMMStage *>(graph.getNode(name)->stage.get());ASSERT_NE(gemm,nullptr);
        EXPECT_TRUE(gemm->getParams().force_decode_equivalent_verifier_prefill);
        ASSERT_TRUE(gemm->getParams().verifier_row_range);
        EXPECT_EQ(gemm->getParams().verifier_row_range->physicalRows(),Fixture::rows);
    }
}

/** @test Invalid identities, lifetimes, names and edges are rejected before graph mutation. */
TEST(Qwen4ExpHyperconnectionBuilder, RejectsInvalidPreparedIdentityAndPartialGraphPublication)
{
    llaminar::v2::ThreadCountGuard threads(1);
    Fixture f;auto binding=f.downBinding;
    binding.immutable=false;
    EXPECT_THROW(Qwen4ExpPreparedProjection(binding,f.store,DeviceId::cpu()),std::invalid_argument);
    EXPECT_THROW(Qwen4ExpPreparedProjection(f.downBinding,f.store,DeviceId::cuda(0)),std::invalid_argument);
    binding=f.downBinding;binding.tensor=&f.up;
    EXPECT_THROW(Qwen4ExpPreparedProjection(binding,f.store,DeviceId::cpu()),std::invalid_argument);
    auto projection=Qwen4ExpPreparedProjection(f.downBinding,f.store,DeviceId::cpu());
    EXPECT_THROW(projection.requireGeometry(Fixture::channels,Fixture::rank),std::invalid_argument);
    ComputeGraph graph;f.seed(graph);const auto before=graph.size();
    EXPECT_THROW(Builder::appendFinalRead(graph,f.context(),f.readWeights(),f.buffers(),{"missing"}),std::invalid_argument);
    EXPECT_EQ(graph.size(),before);
    auto buffers=f.buffers();buffers.normalized=buffers.trunk;
    EXPECT_THROW(Builder::appendFinalRead(graph,f.context(),f.readWeights(),buffers,{"seed"}),std::invalid_argument);
    EXPECT_EQ(graph.size(),before);
    auto context=f.context();context.projectionRows=DeviceRowRange::fullyActive(Fixture::rows);
    EXPECT_THROW(Builder::appendFinalRead(graph,context,f.readWeights(),f.buffers(),{"seed"}),std::invalid_argument);
    EXPECT_EQ(graph.size(),before);
    (void)Builder::appendFinalRead(graph,f.context(),f.readWeights(),f.buffers(),{"seed"});
    const auto installed=graph.size();
    EXPECT_THROW(Builder::appendFinalRead(graph,f.context(),f.readWeights(),f.buffers(),{"seed"}),std::invalid_argument);
    EXPECT_EQ(graph.size(),installed);
}

/** @test Up-only query rows cannot invent a second primitive/common verifier authority. */
TEST(Qwen4ExpHyperconnectionBuilder, UpRowsAuthenticateTheOriginalPrimitiveAndVerifier)
{
    Fixture f;
    const int32_t original=Fixture::rows,foreign=Fixture::rows;
    for(int mutation=0;mutation!=8;++mutation)
    {
        auto context=f.context();
        context.upOperandPrecision=QuantizedActivationOperandPrecision::TwoTermQ8;
        context.rows.liveRows=Qwen4ExpNativeIntInput{DeviceId::cpu(),&original,1};
        context.upProjectionRows=DeviceRowRange::deviceCounted(Fixture::rows,&original);
        const char *diagnostic="";
        if(mutation==0){context.upProjectionRows=DeviceRowRange::deviceCounted(Fixture::rows,&foreign);diagnostic="complete original primitive row identity";}
        if(mutation==1){context.upProjectionRows=DeviceRowRange::fullyActive(Fixture::rows);diagnostic="complete original primitive row identity";}
        if(mutation==2){context.arithmetic=Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier;
            context.projectionRows=DeviceRowRange::deviceCounted(Fixture::rows,&foreign);diagnostic="different authorities";}
        if(mutation==3){context.upOperandPrecision=QuantizedActivationOperandPrecision::SingleQ8;diagnostic="require explicit TwoTermQ8";}
        if(mutation==4){context.upProjectionRows=DeviceRowRange::deviceCounted(Fixture::rows-1,&original);diagnostic="original primitive capacity";}
        if(mutation==5){context.rows.capacity=512;context.upProjectionRows=DeviceRowRange::deviceCounted(256,&original).repeatEachRow(2);
            diagnostic="complete original primitive row identity";}
        if(mutation==6){context.rows.capacity=512;context.arithmetic=Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier;
            context.upProjectionRows.reset();context.projectionRows=DeviceRowRange::deviceCounted(256,&original).repeatEachRow(2);
            diagnostic="complete original primitive row identity";}
        if(mutation==7){context.upProjectionRows.reset();diagnostic="complete original primitive row identity";}
        ComputeGraph graph;
        try
        {
            (void)Builder::appendFinalRead(graph,context,f.readWeights(),f.buffers(),{});
            FAIL()<<"Expected row-authority rejection before constructing the unsupported FP32 correction source";
        }
        catch(const std::invalid_argument &e){EXPECT_NE(std::string(e.what()).find(diagnostic),std::string::npos)<<e.what();}
        EXPECT_EQ(graph.size(),0u);
    }
}

/** @test Equal physical width/pointer cannot substitute expanded or shifted logical count geometry. */
TEST(Qwen4ExpHyperconnectionBuilder, UpRowsRejectExpandedCountAuthorityAtEqualPhysicalWidth)
{
    Fixture f;
    int32_t live=9;
    for(int mutation=0;mutation!=4;++mutation)
    {
        const bool expanded=mutation==0 || mutation==1;
        auto context=f.context();context.rows.capacity=512;
        context.rows.liveRows=Qwen4ExpNativeIntInput{DeviceId::cpu(),&live,1};
        context.upOperandPrecision=QuantizedActivationOperandPrecision::TwoTermQ8;
        context.upProjectionRows=expanded ? DeviceRowRange::deviceCounted(256,&live).repeatEachRow(2)
            : DeviceRowRange::deviceCounted(1024,&live).slice(256,512);
        // The expanded negative passed all old physical-width/capacity/pointer
        // predicates, yet resolved live9 as18 and rejected canonical count512.
        if(expanded)
        {
            EXPECT_EQ(context.upProjectionRows->physicalRows(),512);
            EXPECT_EQ(context.upProjectionRows->capacity(),512);
            EXPECT_EQ(context.upProjectionRows->countOwner(),&live);
            EXPECT_EQ(context.upProjectionRows->activeRowsFor(9),18);
            EXPECT_EQ(context.upProjectionRows->activeRowsFor(512),-1);
        }
        if(mutation==1)
        {
            context.arithmetic=Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier;
            context.projectionRows=context.upProjectionRows;
            context.upProjectionRows.reset();
        }
        if(mutation==2)context.upProjectionRows.reset();
        ComputeGraph graph;
        try
        {
            (void)Builder::appendFinalRead(graph,context,f.readWeights(),f.buffers(),{});
            FAIL()<<"Expected original count-geometry rejection before native source construction";
        }
        catch(const std::invalid_argument &e)
        {
            const auto expected=mutation!=3?"complete original primitive row identity":"original primitive capacity";
            EXPECT_NE(std::string(e.what()).find(expected),std::string::npos)<<e.what();
        }
        EXPECT_EQ(graph.size(),0u);
    }
}
