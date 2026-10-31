/**
 * @file Test__Qwen4ExpPLEBuilder.cpp
 * @brief Execute the complete lexical graph against independent checkpoint equations.
 *
 * Original FP32 table rows, real prepared projections, branch normalization,
 * gating and dilated convolution exercise the composition on CPU. RAM and
 * mmap must produce the same values and preserve accepted histories. This
 * fixture intentionally includes an EOS and nonzero prior convolution rows.
 */
#include "../../../utils/CPUPreparedStoreTestAdmission.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPLEBuilder.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "loaders/PreparedWeightStore.h"
#include "tensors/Tensors.h"
#include "utils/OpenMPUtils.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "loaders/HostImmutableRowProvider.h"
#include <limits>
#include <type_traits>
#include "../../../utils/PreparedWeightTestHarness.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <unistd.h>

using namespace llaminar2;
namespace
{
using Float=Qwen4ExpFloatBinding;
using Int=Qwen4ExpIntBinding;
using Weight=Qwen4ExpImmutableWeightBinding;
/** @brief Every admitted mode enters one unambiguous append API, without per-call compatibility casts. */
using PLEAppend=std::string (*)(ComputeGraph &,const Qwen4ExpPLEContext &,
    const Qwen4ExpPLEWeights &,const Qwen4ExpPLEBuffers &,
    Qwen4ExpPLEBuilder::LookupStorage,const std::vector<std::string> &);
static_assert(std::is_same_v<decltype(&Qwen4ExpPLEBuilder::append),PLEAppend>);
static_assert(std::is_convertible_v<ImmutableLookupStage::CPUStorage,Qwen4ExpPLEBuilder::LookupStorage>);
static_assert(std::is_convertible_v<ImmutableLookupStage::GPUStorage,Qwen4ExpPLEBuilder::LookupStorage>);
static_assert(std::is_convertible_v<ImmutableLookupStage::ResidentGPUStorage,Qwen4ExpPLEBuilder::LookupStorage>);
static_assert(std::is_convertible_v<ImmutableLookupStage::Storage,Qwen4ExpPLEBuilder::LookupStorage>);
static_assert(std::is_convertible_v<Qwen4ExpFixedHeadLookupStorage,Qwen4ExpPLEBuilder::LookupStorage>);
static_assert(std::is_convertible_v<Qwen4ExpPooledLookupStorage,Qwen4ExpPLEBuilder::LookupStorage>);
/** @brief Real native source with its descriptor and allocation certificate. */
struct Source
{
    static constexpr int tableRows=16*7;
    std::array<float,tableRows> values{};
    int fd=-1;
    std::string path;
    /** @brief Publish a tiny original table without touching project model files. */
    Source()
    {
        char pattern[]="/tmp/qfn-ple-composition-XXXXXX";
        fd=::mkstemp(pattern);
        if(fd<0) throw std::runtime_error("Cannot create PLE fixture source");
        path=pattern;
        for(int i=0;i<tableRows;++i) values[i]=static_cast<float>((i*7)%23-11)/13;
        if(::write(fd,values.data(),sizeof(values))!=sizeof(values))
        {
            ::close(fd);::unlink(path.c_str());fd=-1;
            throw std::runtime_error("Cannot publish PLE fixture source");
        }
    }
    ~Source(){if(fd>=0)::close(fd);::unlink(path.c_str());}
    /** @brief Admit only the selected native backing under one physical owner. */
    std::shared_ptr<HostImmutableRowProvider> open(ImmutableRowStorage storage) const
    {
        ImmutableRowTableSource source{.path=path,.tensor_name="per_layer_token_embd.weight",
            .rows=tableRows,.row_elements=1,.format=GGUFTensorType::F32};
        auto policy=ImmutableRowTableStoragePolicy::forStorage(storage,4096,2);
        PhysicalMemoryPlanBuilder plan;
        plan.add({.world_rank=0,.device=DeviceId::cpu(),.total_bytes=1u<<20,.admission_available_bytes=1u<<20},
            PhysicalMemoryOwner::ImmutableTableStorage,ImmutableRowTable::physicalBytes(source,policy));
        auto authority=std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
        return HostImmutableRowProvider::fromWholeTable(std::make_unique<ImmutableRowTable>(
            source,policy,DeviceId::cpu(),authority,ImmutableRowTableHostPlacement::taskPolicy()));
    }
};
/** @brief Tiny real prepared matrices with disjoint typed arena roles. */
struct Fixture
{
    static constexpr int rows=4,hidden=16,channels=4*hidden;
    PreparedWeightStore store{ModelContextId{99174}};
    FP32Tensor keyW{{channels,hidden}},valueW{{hidden,hidden}},keyGain{{channels}},queryGain{{channels}},
        convGain{{channels}},convW{{channels,4}};
    FP32Tensor embedding{{rows,hidden}},key{{rows,channels}},value{{rows,hidden}},query{{rows,channels}},
        gated{{rows,channels}},normalized{{rows,channels}},history{{9,channels}},trunk{{rows,channels}};
    INT32Tensor tokens{{rows}},prior{{2}},ids{{rows,32}},native{{rows,16}},live{{1}};
    WeightBinding keyBinding,valueBinding;
    qwen4exp::NgramHashPolicy hash;
    std::vector<float> original,priorConvolution;
    /** @brief Deterministic signed finite inputs and original immutable prepared weights. */
    Fixture()
    {
        for(auto *tensor:{&keyW,&valueW,&convW})
            for(std::size_t i=0;i<tensor->numel();++i)
                tensor->mutable_data()[i]=static_cast<float>(static_cast<int>((i*3)%17)-8)/51;
        for(auto *tensor:{&keyGain,&queryGain,&convGain})
            for(std::size_t i=0;i<tensor->numel();++i)tensor->mutable_data()[i]=0.5f+static_cast<float>(i%11)/13;
        for(auto *tensor:{&history,&trunk})
            for(std::size_t i=0;i<tensor->numel();++i)
                tensor->mutable_data()[i]=static_cast<float>(static_cast<int>((i*7)%29)-14)/19;
        original.assign(trunk.data(),trunk.data()+trunk.numel());
        priorConvolution.assign(history.data(),history.data()+history.numel());
        hash.multipliers={9007199254740993ULL,9007199254740995ULL,9007199254740997ULL};
        for(int head=0;head<16;++head){hash.moduli[head]=7;hash.offsets[head]=head*7;}
        const std::array<std::int32_t,rows> input{17,static_cast<int>(hash.eosToken),19,23};
        std::memcpy(tokens.raw_mutable_data(),input.data(),sizeof(input));
        auto *p=static_cast<std::int32_t *>(prior.raw_mutable_data());p[0]=11;p[1]=13;
        *static_cast<int *>(live.raw_mutable_data())=rows;
        test::admitCPUPreparedStoreForTest(store, {&keyW, &valueW});
        keyBinding=prepare(keyW,"ple_key.weight");valueBinding=prepare(valueW,"ple_value.weight");
    }
    /** @brief Preserve native prepared-store identity for both projection shapes. */
    WeightBinding prepare(TensorBase &tensor,const char *name)
    {
        auto binding=test::makePreparedWeightTestBinding(&tensor,DeviceId::cpu(),name,store.modelId());
        binding.prepared=store.prepareGemm(binding);return binding;
    }
    /** @brief Exact immutable lexical execution policy, with a replayed live count. */
    Qwen4ExpPLEContext context()
    {
        return {.device=DeviceId::cpu(),.prefix="L1_ple_",.rows={rows,Int(live,BufferId::PREFILL_CHUNK_GEOMETRY)},
            .hidden=hidden,.epsilon=1e-6f,.hash=hash,.nativeFormat=TensorType::FP32,.nativeRowElements=1,.nativeRowBytes=sizeof(float)};
    }
    /** @brief Mandatory model operands retained separately from transient activation storage. */
    Qwen4ExpPLEWeights weights()
    {
        return {Qwen4ExpPreparedProjection(keyBinding,store,DeviceId::cpu()),
            Qwen4ExpPreparedProjection(valueBinding,store,DeviceId::cpu()),
            Weight(keyGain),Weight(queryGain),Weight(convGain),Weight(convW)};
    }
    /** @brief Semantic role mapping without imperative device-specific graph logic. */
    Qwen4ExpPLEBuffers buffers()
    {
        return {Int(tokens,BufferId::REQUEST_TOKEN_IDS),Int(prior,BufferId::QFN_PLE_TOKEN_HISTORY),
            Int(ids,BufferId::QFN_NGRAM_ROW_IDS),
            Float(embedding,BufferId::QFN_PLE_EMBEDDING),Float(key,BufferId::QFN_PLE_KEY),
            Float(value,BufferId::QFN_PLE_VALUE),Float(query,BufferId::QFN_PLE_QUERY),
            Float(gated,BufferId::QFN_PLE_GATED),Float(normalized,BufferId::QFN_PLE_NORMALIZED),
            Float(history,BufferId::QFN_PLE_CONV_HISTORY),Float(trunk,BufferId::HIDDEN_STATE)};
    }
};
/** @brief Independent double-precision actual-gain normalization for four branches. */
std::vector<double> normalize(std::vector<double> values,const float *gain)
{
    for(int branch=0;branch<4;++branch)
    {
        double sum=0;for(int c=0;c<Fixture::hidden;++c)sum+=std::pow(values[branch*Fixture::hidden+c],2);
        const double inv=1/std::sqrt(sum/Fixture::hidden+1e-6);
        for(int c=0;c<Fixture::hidden;++c)values[branch*Fixture::hidden+c]*=inv*gain[branch*Fixture::hidden+c];
    }
    return values;
}
}

/** @test Both native storage policies execute the entire PLE graph and preserve prior accepted state. */
TEST(Qwen4ExpPLEBuilder, RAMAndMmapPreparedGraphsMatchIndependentEquations)
{
    llaminar::v2::ThreadCountGuard threads(1);CPUDeviceContext cpu(DeviceId::cpu(),1);Source source;
    for(auto storage:{ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped})
    {
        Fixture f;auto table=source.open(storage);ComputeGraph graph;
        auto terminal=Qwen4ExpPLEBuilder::append(graph,f.context(),f.weights(),f.buffers(),
            ImmutableLookupStage::CPUStorage{table,Qwen4ExpIntBinding(f.native,BufferId::QFN_NGRAM_NATIVE_ROWS)},{});
        EXPECT_EQ(terminal,"L1_ple_convolution");EXPECT_EQ(graph.size(),9u);
        for(auto *stage:graph.getExecutionStages())ASSERT_TRUE(stage->execute(&cpu));
        std::vector<std::vector<double>> normRows;
        const auto *tokens=static_cast<const std::int32_t *>(f.tokens.raw_data());
        const auto *prior=static_cast<const std::int32_t *>(f.prior.raw_data());
        for(int row=0;row<Fixture::rows;++row)
        {
            std::vector<double> embed(Fixture::hidden),key(Fixture::channels),value(Fixture::hidden),query(Fixture::channels);
            for(int head=0;head<16;++head)
            {
                std::uint64_t hash=0;bool crossed=false;
                for(int tap=0;tap<(head<8?2:3);++tap)
                {
                    int index=row-tap;
                    auto token=crossed?f.hash.eosToken:(index<0?prior[index+2]:tokens[index]);
                    hash^=static_cast<std::uint64_t>(token)*f.hash.multipliers[tap];
                    if(tap>0 && token==f.hash.eosToken)crossed=true;
                }
                embed[head]=source.values[hash%7+head*7];
                EXPECT_EQ(f.embedding.data()[row*Fixture::hidden+head],embed[head]);
            }
            for(int out=0;out<Fixture::channels;++out)
                for(int k=0;k<Fixture::hidden;++k)key[out]+=embed[k]*f.keyW.data()[out*Fixture::hidden+k];
            for(int out=0;out<Fixture::hidden;++out)
                for(int k=0;k<Fixture::hidden;++k)value[out]+=embed[k]*f.valueW.data()[out*Fixture::hidden+k];
            std::copy_n(f.original.begin()+row*Fixture::channels,Fixture::channels,query.begin());
            key=normalize(std::move(key),f.keyGain.data());query=normalize(std::move(query),f.queryGain.data());
            std::vector<double> gated(Fixture::channels);
            for(int branch=0;branch<4;++branch)
            {
                double dot=0;for(int c=0;c<Fixture::hidden;++c)dot+=key[branch*Fixture::hidden+c]*query[branch*Fixture::hidden+c];
                dot/=std::sqrt(static_cast<double>(Fixture::hidden));
                const double signedRoot=(dot>0?1:(dot<0?-1:0))*std::sqrt(std::max(std::abs(dot),1e-6));
                for(int c=0;c<Fixture::hidden;++c)gated[branch*Fixture::hidden+c]=value[c]/(1+std::exp(-signedRoot));
            }
            normRows.push_back(normalize(gated,f.convGain.data()));
            for(int c=0;c<Fixture::channels;++c)
            {
                double conv=0;
                for(int tap=0;tap<4;++tap)
                {
                    const int index=row-9+3*tap;
                    const double x=index<0?f.priorConvolution[(index+9)*Fixture::channels+c]:normRows[index][c];
                    conv+=x*f.convW.data()[c*4+tap];
                }
                const double expected=f.original[row*Fixture::channels+c]+gated[c]+conv/(1+std::exp(-conv));
                EXPECT_NEAR(f.trunk.data()[row*Fixture::channels+c],expected,3e-5);
            }
        }
        EXPECT_EQ(std::memcmp(f.history.data(),f.priorConvolution.data(),f.history.size_bytes()),0);
        EXPECT_EQ(prior[0],11);EXPECT_EQ(prior[1],13);
        EXPECT_EQ(table->sourceStats().rows,Fixture::rows*16u);
        for(int live:{1,0,3})
        {
            *static_cast<int *>(f.live.raw_mutable_data())=live;
            std::copy(f.original.begin(),f.original.end(),f.trunk.mutable_data());
            for(auto *stage:graph.getExecutionStages())ASSERT_TRUE(stage->execute(&cpu));
            EXPECT_EQ(std::memcmp(f.trunk.data()+live*Fixture::channels,f.original.data()+live*Fixture::channels,
                (Fixture::rows-live)*Fixture::channels*sizeof(float)),0);
        }
    }
}

/** @test Invalid operands cannot leave a partially wired lexical block in the graph. */
TEST(Qwen4ExpPLEBuilder, RejectsAliasesAndMissingDependenciesBeforeGraphMutation)
{
    llaminar::v2::ThreadCountGuard threads(1);Source source;Fixture f;
    auto storage=ImmutableLookupStage::CPUStorage{source.open(ImmutableRowStorage::RAM),Qwen4ExpIntBinding(f.native,BufferId::QFN_NGRAM_NATIVE_ROWS)};
    ComputeGraph graph;auto buffers=f.buffers();buffers.normalized=buffers.trunk;
    EXPECT_THROW(Qwen4ExpPLEBuilder::append(graph,f.context(),f.weights(),buffers,storage,{}),std::invalid_argument);
    EXPECT_EQ(graph.size(),0u);
    EXPECT_THROW(Qwen4ExpPLEBuilder::append(graph,f.context(),f.weights(),f.buffers(),storage,{"absent"}),std::invalid_argument);
    EXPECT_EQ(graph.size(),0u);
}

/** @test Actual prepared key/value nodes retain verifier math, original count and serial-row bytes. */
TEST(Qwen4ExpPLEBuilder, GroupedPreparedProjectionsUseOriginalQueryAndPreserveInactiveBytes)
{
    llaminar::v2::ThreadCountGuard threads(1);CPUDeviceContext cpu(DeviceId::cpu(),1);Source source;
    for(const int capacity:{1,Fixture::rows})
    {
        SCOPED_TRACE(capacity);Fixture f;ComputeGraph graph;
        auto context=f.context();context.rows={capacity,Qwen4ExpNativeIntInput{DeviceId::cpu(),
            static_cast<const std::int32_t *>(f.live.raw_data()),1}};
        context.arithmetic=Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier;
        context.projectionRows=DeviceRowRange::deviceCounted(capacity,static_cast<const std::int32_t *>(f.live.raw_data()));
        Qwen4ExpPLEBuilder::append(graph,context,f.weights(),f.buffers(),
            ImmutableLookupStage::CPUStorage{source.open(ImmutableRowStorage::RAM),Int(f.native,BufferId::QFN_NGRAM_NATIVE_ROWS)},{});
        std::array<GEMMStage *,2> projections{};
        for(std::size_t i=0;i<projections.size();++i)
        {
            const auto *node=graph.getNode(i==0?"L1_ple_key":"L1_ple_value");ASSERT_NE(node,nullptr);
            projections[i]=dynamic_cast<GEMMStage *>(node->stage.get());ASSERT_NE(projections[i],nullptr);
            const auto &params=projections[i]->getParams();
            ASSERT_TRUE(params.force_decode_equivalent_verifier_prefill);ASSERT_TRUE(params.verifier_row_range);
            EXPECT_EQ(params.verifier_row_range->captureIdentity(),context.projectionRows->captureIdentity());
            EXPECT_EQ(params.prepared_store,&f.store);
            const auto &original=i==0?f.keyBinding.prepared:f.valueBinding.prepared;
            ASSERT_TRUE(params.prepared_ref);ASSERT_TRUE(original);
            EXPECT_EQ(params.prepared_ref->model_id,original->model_id);
            EXPECT_EQ(params.prepared_ref->binding_id,original->binding_id);
            EXPECT_EQ(params.prepared_ref->kind,original->kind);EXPECT_EQ(params.prepared_ref->device,original->device);
        }
        const std::array signatures{projections[0]->graphCaptureVariantSignature(),projections[1]->graphCaptureVariantSignature()};
        for(const int live:{capacity,capacity-1,0,1,0,capacity})
        {
            SCOPED_TRACE(live);*static_cast<std::int32_t *>(f.live.raw_mutable_data())=live;
            std::fill_n(f.embedding.mutable_data(),f.embedding.numel(),std::numeric_limits<float>::quiet_NaN());
            for(int row=0;row<live;++row)for(int k=0;k<Fixture::hidden;++k)
                f.embedding.mutable_data()[row*Fixture::hidden+k]=float((row*7+k)%17-8)/16.0f;
            for(std::size_t i=0;i<projections.size();++i)
            {
                auto *stage=projections[i];const auto &params=stage->getParams();const int width=params.n;
                auto &output=i==0?f.key:f.value;
                std::fill_n(output.mutable_data(),output.numel(),-1903.25f);
                std::vector<float> expected(output.numel(),-1903.25f);
                FP32Tensor serialInput({1,Fixture::hidden}),serialOutput({1,static_cast<std::size_t>(width)});
                auto serialParams=params;serialParams.A=&serialInput;serialParams.C=&serialOutput;serialParams.m=1;
                serialParams.force_decode_equivalent_verifier_prefill=false;serialParams.verifier_row_range.reset();
                GEMMStage serial(serialParams);
                for(int row=0;row<live;++row)
                {
                    std::copy_n(f.embedding.data()+row*Fixture::hidden,Fixture::hidden,serialInput.mutable_data());
                    ASSERT_TRUE(serial.execute(&cpu));
                    std::copy_n(serialOutput.data(),width,expected.data()+row*width);
                }
                EXPECT_EQ(stage->graphCaptureVariantSignature(),signatures[i]);ASSERT_NE(signatures[i],0u);
                ASSERT_TRUE(stage->execute(&cpu));
                EXPECT_EQ(std::memcmp(output.raw_data(),expected.data(),output.size_bytes()),0);
            }
        }
    }
}

/** @test Complete row identity is required before adding any node, not merely equal physical width. */
TEST(Qwen4ExpPLEBuilder, RejectsForeignSlicedRepeatedAndOmittedVerifierRowsBeforeMutation)
{
    llaminar::v2::ThreadCountGuard threads(1);Source source;Fixture f;
    auto context=f.context();const auto *count=static_cast<const std::int32_t *>(f.live.raw_data());
    context.rows={Fixture::rows,Qwen4ExpNativeIntInput{DeviceId::cpu(),count,1}};
    context.arithmetic=Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier;
    std::int32_t foreign=Fixture::rows;
    const std::array invalid{
        DeviceRowRange::deviceCounted(Fixture::rows,&foreign),
        DeviceRowRange::deviceCounted(Fixture::rows+1,count).slice(1,Fixture::rows),
        DeviceRowRange::deviceCounted(Fixture::rows/2,count).repeatEachRow(2),
        DeviceRowRange::fullyActive(Fixture::rows)};
    const auto storage=ImmutableLookupStage::CPUStorage{source.open(ImmutableRowStorage::RAM),Int(f.native,BufferId::QFN_NGRAM_NATIVE_ROWS)};
    for(const auto rows:invalid)
    {
        ComputeGraph graph;context.projectionRows=rows;
        try
        {
            (void)Qwen4ExpPLEBuilder::append(graph,context,f.weights(),f.buffers(),storage,{});
            FAIL()<<"Expected complete original primitive row identity rejection";
        }
        catch(const std::invalid_argument &error)
        { EXPECT_NE(std::string(error.what()).find("complete original primitive row identity"),std::string::npos)<<error.what(); }
        EXPECT_EQ(graph.size(),0u);
    }
    context.projectionRows.reset();ComputeGraph missing;
    EXPECT_THROW(Qwen4ExpPLEBuilder::append(missing,context,f.weights(),f.buffers(),storage,{}),std::invalid_argument);
    EXPECT_EQ(missing.size(),0u);
    context.projectionRows=DeviceRowRange::deviceCounted(Fixture::rows,count);
    for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})
    {
        ComputeGraph foreignOwner;context.rows.liveRows=Qwen4ExpNativeIntInput{device,count,1};
        EXPECT_THROW(Qwen4ExpPLEBuilder::append(foreignOwner,context,f.weights(),f.buffers(),storage,{}),std::invalid_argument);
        EXPECT_EQ(foreignOwner.size(),0u);
    }
    context.rows.liveRows=Qwen4ExpNativeIntInput{DeviceId::cpu(),count,2};ComputeGraph countCapacity;
    EXPECT_THROW(Qwen4ExpPLEBuilder::append(countCapacity,context,f.weights(),f.buffers(),storage,{}),std::invalid_argument);
    EXPECT_EQ(countCapacity.size(),0u);
}

/** @test Count values are replay data, while the original count owner belongs to both GEMM identities. */
TEST(Qwen4ExpPLEBuilder, BothProjectionCaptureIdentitiesRetainCountOwnerWithoutLiveValue)
{
    llaminar::v2::ThreadCountGuard threads(1);Source source;Fixture f;std::int32_t first=0,second=0;
    const auto signature=[&](const std::int32_t *count) {
        auto context=f.context();context.rows={Fixture::rows,Qwen4ExpNativeIntInput{DeviceId::cpu(),count,1}};
        context.arithmetic=Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier;
        context.projectionRows=DeviceRowRange::deviceCounted(Fixture::rows,count);ComputeGraph graph;
        Qwen4ExpPLEBuilder::append(graph,context,f.weights(),f.buffers(),
            ImmutableLookupStage::CPUStorage{source.open(ImmutableRowStorage::RAM),Int(f.native,BufferId::QFN_NGRAM_NATIVE_ROWS)},{});
        auto *key=dynamic_cast<GEMMStage *>(graph.getNode("L1_ple_key")->stage.get());
        auto *value=dynamic_cast<GEMMStage *>(graph.getNode("L1_ple_value")->stage.get());
        if(!key||!value)throw std::logic_error("Missing actual prepared PLE projection");
        return std::array{key->graphCaptureVariantSignature(),value->graphCaptureVariantSignature()};
    };
    const auto original=signature(&first);first=Fixture::rows;
    EXPECT_EQ(signature(&first),original);const auto changed=signature(&second);
    EXPECT_NE(changed[0],original[0]);EXPECT_NE(changed[1],original[1]);
}

/** @test Ordinary graph construction does not select serial-row arithmetic to hide inactive padding. */
TEST(Qwen4ExpPLEBuilder, OrdinaryProjectionsKeepTheirOriginalPolicy)
{
    llaminar::v2::ThreadCountGuard threads(1);Source source;Fixture f;ComputeGraph graph;
    auto context=f.context();const auto storage=ImmutableLookupStage::CPUStorage{
        source.open(ImmutableRowStorage::RAM),Int(f.native,BufferId::QFN_NGRAM_NATIVE_ROWS)};
    Qwen4ExpPLEBuilder::append(graph,context,f.weights(),f.buffers(),storage,{});
    for(const char *name:{"L1_ple_key","L1_ple_value"})
    {
        const auto *stage=dynamic_cast<const GEMMStage *>(graph.getNode(name)->stage.get());ASSERT_NE(stage,nullptr);
        EXPECT_FALSE(stage->getParams().force_decode_equivalent_verifier_prefill);
        EXPECT_FALSE(stage->getParams().verifier_row_range);
    }
    context.projectionRows=DeviceRowRange::deviceCounted(Fixture::rows,static_cast<const std::int32_t *>(f.live.raw_data()));
    ComputeGraph rejected;
    EXPECT_THROW(Qwen4ExpPLEBuilder::append(rejected,context,f.weights(),f.buffers(),storage,{}),std::invalid_argument);
    EXPECT_EQ(rejected.size(),0u);
}
