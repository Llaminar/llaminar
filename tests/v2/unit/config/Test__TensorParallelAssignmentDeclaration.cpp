/**
 * @file Test__TensorParallelAssignmentDeclaration.cpp
 * @brief Device-free original-source/consumer declaration and retained assignment guards.
 *
 * Metadata and endpoint observations are authored fixtures, never model/runtime
 * evidence. The tests invoke the real canonical resolver and compare directory
 * versus published profile source adapters; no backend or payload is opened.
 */
#include "planning/TensorParallelSourceDeclaration.h"
#include "loaders/ModelContext.h"
#include "../models/qwen4exp/Qwen4ExpMetadataFixture.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>

namespace llaminar2::test
{
namespace
{
/** @return Complete authored original K/V source descriptors without model payload. */
GGUFModel declarationDirectory(bool predictor=true)
{
    auto original=qwen4MetadataFixture(predictor);
    for(int layer=3;layer<48;layer+=4)for(const char *role:{"attn_k.weight","attn_v.weight"})
        original.tensors.push_back({.name="blk."+std::to_string(layer)+"."+role,.dimensions={512,2560},.type=GGUFTensorType::Q8_0});
    return original;
}
/** @return Nonordinal actual-domain identities authored without hardware discovery. */
std::vector<DeviceId> members(int degree,DeviceType type=DeviceType::ROCm)
{
    std::vector<DeviceId> result;
    for(int i=0;i<degree;++i)result.emplace_back(type,19-2*i);
    return result;
}
/** @brief Compare every canonical semantic interval and original physical identity. */
void equalAssignments(const TensorParallelConfig &a,const TensorParallelConfig &b)
{
    ASSERT_EQ(a.worldSize(),b.worldSize());
    for(int rank=0;rank<a.worldSize();++rank)
    {
        const auto &x=a.forRank(rank),&y=b.forRank(rank);
        EXPECT_EQ(x.device,y.device);EXPECT_EQ(x.local_rank,y.local_rank);
        EXPECT_EQ(x.head_start,y.head_start);EXPECT_EQ(x.head_count,y.head_count);
        EXPECT_EQ(x.kv_head_start,y.kv_head_start);EXPECT_EQ(x.kv_head_count,y.kv_head_count);
        EXPECT_EQ(x.d_ff_start,y.d_ff_start);EXPECT_EQ(x.d_ff_count,y.d_ff_count);
        EXPECT_EQ(x.vocab_start,y.vocab_start);EXPECT_EQ(x.vocab_count,y.vocab_count);
        EXPECT_EQ(x.work_fraction,y.work_fraction);
    }
}
}
/** @test Complete original directory/profile consumers agree at every supported head count. */
TEST(TensorParallelAssignmentDeclaration, OriginalDirectoryProfileAndCanonicalFactoryAgreeEveryDegree)
{
    const auto directory=declarationDirectory();const Qwen4ExpConfig original(directory);
    const auto profile=ModelMemoryProfile::fromGGUF(directory);
    MTPRuntimeConfig mtp;mtp.enabled=false;mtp.graph_capacity_draft_tokens=0;
    const auto gdns=originalQfnTPConsumers(original);
    ASSERT_EQ(gdns.size(),36u);
    for(const auto &gdn:gdns){EXPECT_EQ(gdn.key_heads,16);EXPECT_EQ(gdn.value_heads,48);}
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})for(int degree=1;degree<=8;++degree)
    {
        SCOPED_TRACE(::testing::Message()<<int(backend)<<"/"<<degree);
        const auto devices=members(degree,backend);
        const auto from_directory=resolveQfnTPDeclaration(directory,TensorParallelMainPlacement::FullMainRankLocal,
            mtp,DenseParallelPolicy::TensorParallel,devices);
        const auto from_profile=resolveProfileTPDeclaration(profile,TensorParallelMainPlacement::FullMainRankLocal,
            mtp,DenseParallelPolicy::TensorParallel,devices);
        ASSERT_EQ(from_directory.coverage(),KVHeadCoveragePolicy::QueryInterval);
        ASSERT_TRUE(from_directory.queries());ASSERT_TRUE(from_profile.queries());
        EXPECT_EQ(from_directory.queries()->granularity(),3);
        EXPECT_EQ(from_directory.queries()->consumers(),from_profile.queries()->consumers());
        const auto a=from_directory.partition(devices,{},profile.d_ff,profile.vocab_size);
        const auto b=from_profile.partition(devices,{},profile.d_ff,profile.vocab_size);
        from_directory.validate(a,devices);from_profile.validate(a,devices);equalAssignments(a,b);
        const auto direct=TensorParallelConfig::proportionalSplit(devices,std::vector<float>(degree,1.f),
            profile.n_heads,profile.n_kv_heads,profile.d_ff,profile.vocab_size,KVHeadCoveragePolicy::QueryInterval,
            QueryHeadPartitionDeclaration::fromOriginalConsumers(profile.n_heads,gdns));
        equalAssignments(a,direct);
        std::vector<int> q_owners(profile.n_heads),kv_owners(profile.n_kv_heads);
        for(const auto &local:a.assignments())
        {
            EXPECT_TRUE(GDNHeadAssignment::hasIntegralKeyHeadBoundaries(16,48,local.head_start,local.head_count,24));
            for(int q=local.head_start;q<local.headEnd();++q)
            {++q_owners[q];EXPECT_GE(q/12,local.kv_head_start);EXPECT_LT(q/12,local.kvHeadEnd());}
            for(int kv=0;kv<2;++kv)
            {
                const bool needed=kv*12<local.headEnd() && local.head_start<(kv+1)*12;
                EXPECT_EQ(needed,kv>=local.kv_head_start && kv<local.kvHeadEnd());if(needed)++kv_owners[kv];
            }
        }
        EXPECT_TRUE(std::all_of(q_owners.begin(),q_owners.end(),[](int n){return n==1;}));
        EXPECT_TRUE(std::all_of(kv_owners.begin(),kv_owners.end(),[](int n){return n>0;}));
    }
}
/** @test Existing retained full/phase/predictor families remain explicitly distinct from ordinary query coverage. */
TEST(TensorParallelAssignmentDeclaration, RetainedFamiliesKeepTheirDeclaredPolicy)
{
    const auto directory=declarationDirectory();const Qwen4ExpConfig original(directory);const auto devices=members(4);
    MTPRuntimeConfig mtp;mtp.enabled=false;mtp.graph_capacity_draft_tokens=0;
    for(const auto dense:{DenseParallelPolicy::Replicated,DenseParallelPolicy::PrefillTensorParallelDecodeReplicated,
            DenseParallelPolicy::TensorParallelDecodeMirroredEmbedding})
        EXPECT_EQ(resolveQfnTPDeclaration(directory,TensorParallelMainPlacement::FullMainRankLocal,mtp,dense,devices).coverage(),KVHeadCoveragePolicy::Legacy);
    EXPECT_EQ(resolveQfnTPDeclaration(directory,TensorParallelMainPlacement::OtherDeclared,mtp,DenseParallelPolicy::TensorParallel,devices).coverage(),KVHeadCoveragePolicy::Legacy);
    mtp.graph_capacity_draft_tokens=1;
    EXPECT_EQ(resolveQfnTPDeclaration(directory,TensorParallelMainPlacement::FullMainRankLocal,mtp,DenseParallelPolicy::TensorParallel,devices).coverage(),KVHeadCoveragePolicy::Legacy);
    mtp.graph_capacity_draft_tokens=0;mtp.enabled=true;
    EXPECT_EQ(resolveQfnTPDeclaration(directory,TensorParallelMainPlacement::FullMainRankLocal,mtp,DenseParallelPolicy::TensorParallel,devices).coverage(),KVHeadCoveragePolicy::Legacy);
}
/** @test Missing/foreign source and fake endpoint identities cannot silently produce a valid query assignment. */
TEST(TensorParallelAssignmentDeclaration, RejectsMissingForeignSourceAndPhysicalIdentity)
{
    const auto directory=declarationDirectory(false);const auto devices=members(4);
    MTPRuntimeConfig mtp;mtp.enabled=false;mtp.graph_capacity_draft_tokens=0;
    const auto source=resolveQfnTPDeclaration(directory,TensorParallelMainPlacement::FullMainRankLocal,mtp,DenseParallelPolicy::TensorParallel,devices);
    auto assigned=source.partition(devices,{},640,248320);EXPECT_NO_THROW(source.validate(assigned,devices));
    auto wrong=devices;wrong[1]=wrong[0];EXPECT_THROW(source.partition(wrong,{},640,248320),std::invalid_argument);
    wrong=devices;wrong[1]=DeviceId::cuda(1);EXPECT_THROW(source.partition(wrong,{},640,248320),std::invalid_argument);
    EXPECT_THROW(source.partition({DeviceId::cpu()}, {},640,248320),std::invalid_argument);
    wrong=devices;std::reverse(wrong.begin(),wrong.end());EXPECT_THROW(source.validate(assigned,wrong),std::invalid_argument);
    const auto legacy=TensorParallelConfig::proportionalSplit(devices,std::vector<float>(4,1.f),24,2,640,248320);
    EXPECT_THROW(source.validate(legacy,devices),std::invalid_argument);
    EXPECT_THROW(TensorParallelAssignmentDeclaration::fromOriginal({24,2},KVHeadCoveragePolicy::QueryInterval),std::invalid_argument);
    EXPECT_THROW(TensorParallelAssignmentDeclaration::fromOriginal({24,2},KVHeadCoveragePolicy::QueryInterval,
        QueryHeadPartitionDeclaration::fromOriginalConsumers(28,{})),std::invalid_argument);
    auto missing=ModelMemoryProfile::fromGGUF(declarationDirectory(false));missing.flash_next.reset();
    EXPECT_THROW(resolveProfileTPDeclaration(missing,TensorParallelMainPlacement::FullMainRankLocal,mtp,DenseParallelPolicy::TensorParallel,devices),std::invalid_argument);
    EXPECT_THROW(source.partition(members(9),{},640,248320),std::invalid_argument);
    EXPECT_THROW(source.partition(devices,{1.f,0.f,1.f,1.f},640,248320),std::invalid_argument);
    EXPECT_THROW(source.partition(devices,{1.f,std::numeric_limits<float>::quiet_NaN(),1.f,1.f},640,248320),std::invalid_argument);
}
/** @test Floating source coverage is declared before splitting; missing/unknown metadata cannot select QueryInterval. */
TEST(TensorParallelAssignmentDeclaration, OriginalProjectionFormatsSelectDeclaredFamilyBeforeAssignment)
{
    auto directory=declarationDirectory(false);MTPRuntimeConfig mtp;mtp.enabled=false;mtp.graph_capacity_draft_tokens=0;
    const auto devices=members(4);
    for(auto &tensor:directory.tensors)if(tensor.name=="blk.3.attn_v.weight")tensor.type=GGUFTensorType::F16;
    const auto profile=ModelMemoryProfile::fromGGUF(directory);
    EXPECT_EQ(originalQfnKVProjectionSource(profile),TensorParallelKVProjectionSource::FloatingFullBank);
    const auto source=resolveQfnTPDeclaration(directory,TensorParallelMainPlacement::FullMainRankLocal,mtp,DenseParallelPolicy::TensorParallel,devices);
    EXPECT_EQ(source.coverage(),KVHeadCoveragePolicy::Legacy);
    const auto from_profile=resolveProfileTPDeclaration(profile,TensorParallelMainPlacement::FullMainRankLocal,mtp,DenseParallelPolicy::TensorParallel,devices);
    equalAssignments(source.partition(devices,{},640,248320),from_profile.partition(devices,{},640,248320));
    auto malformed=profile;malformed.tensors.erase(std::remove_if(malformed.tensors.begin(),malformed.tensors.end(),
        [](const auto &tensor){return tensor.name=="blk.3.attn_k.weight";}),malformed.tensors.end());
    EXPECT_THROW(originalQfnKVProjectionSource(malformed),std::invalid_argument);
    malformed=profile;for(auto &tensor:malformed.tensors)if(tensor.name=="blk.3.attn_k.weight")tensor.quant_type="unknown-original-format";
    EXPECT_THROW(originalQfnKVProjectionSource(malformed),std::invalid_argument);
}
/** @test The concrete facade exposes exactly its existing loader-owned metadata without creating a weight owner. */
TEST(TensorParallelAssignmentDeclaration, PassiveContextDirectoryIsTheOriginalBorrowedOwner)
{
    const auto context=ModelContext::createForTesting("authored-directory-only.gguf");ASSERT_TRUE(context);
    const IModelContext &view=*context;
    const auto *before=view.ggufDirectory();ASSERT_EQ(before,&context->model());
    const auto tensors=before->tensors.size(),metadata=before->metadata.size();
    EXPECT_EQ(view.ggufDirectory(),before);EXPECT_EQ(before->tensors.size(),tensors);
    EXPECT_EQ(before->metadata.size(),metadata);EXPECT_EQ(context->concreteWeightManager(),nullptr);
}
/** @test CPU NUMA participants retain rank-indexed shards within one host address space. */
TEST(TensorParallelAssignmentDeclaration, CPURanksShareAddressSpaceWithoutLosingShardIdentity)
{
    const auto directory=declarationDirectory(false);
    const auto profile=ModelMemoryProfile::fromGGUF(directory);
    MTPRuntimeConfig mtp;mtp.enabled=false;mtp.graph_capacity_draft_tokens=0;
    // These are storage IDs after actual CPU-domain validation, not invented
    // NUMA ordinals. Both participants intentionally name the same host memory.
    const std::vector<DeviceId> devices{DeviceId::cpu(),DeviceId::cpu()};
    const auto source=resolveProfileTPDeclaration(profile,TensorParallelMainPlacement::FullMainRankLocal,
        mtp,DenseParallelPolicy::TensorParallel,devices);
    ASSERT_EQ(source.coverage(),KVHeadCoveragePolicy::Legacy);
    const auto assigned=source.partition(devices,{},profile.d_ff,profile.vocab_size);
    ASSERT_TRUE(assigned.validate())<<assigned.validationError();
    EXPECT_NO_THROW(source.validate(assigned,devices));
    const auto &first=assigned.forRank(0),&second=assigned.forRank(1);
    EXPECT_EQ(first.device,DeviceId::cpu());EXPECT_EQ(second.device,DeviceId::cpu());
    EXPECT_EQ(first.local_rank,0);EXPECT_EQ(second.local_rank,1);
    EXPECT_EQ(first.head_start,0);EXPECT_EQ(first.headEnd(),second.head_start);
    EXPECT_EQ(second.headEnd(),profile.n_heads);
    EXPECT_EQ(first.dFFEnd(),second.d_ff_start);EXPECT_EQ(second.dFFEnd(),profile.d_ff);
    EXPECT_EQ(first.vocabEnd(),second.vocab_start);EXPECT_EQ(second.vocabEnd(),profile.vocab_size);
    EXPECT_THROW(assigned.forDevice(DeviceId::cpu()),std::invalid_argument);
    EXPECT_THROW(assigned.forRank(2),std::out_of_range);
    auto stale=assigned.assignments();stale[1].local_rank=0;
    EXPECT_FALSE(TensorParallelConfig(stale).validate());
    EXPECT_THROW(source.validate(TensorParallelConfig(stale),devices),std::invalid_argument);
    // Device uniqueness remains mandatory for both accelerator backends.
    for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})
    {
        auto duplicate=assigned.assignments();for(auto &local:duplicate)local.device=device;
        const TensorParallelConfig invalid(duplicate);
        EXPECT_FALSE(invalid.validate());
        EXPECT_NE(invalid.validationError().find("Duplicate device"),std::string::npos);
        EXPECT_THROW(invalid.forDevice(device),std::invalid_argument);
    }
    const auto single=TensorParallelConfig::singleDevice(DeviceId::cpu(),24,2,640,248320);
    EXPECT_EQ(&single.forDevice(DeviceId::cpu()),&single.forRank(0));
}
} // namespace llaminar2::test
