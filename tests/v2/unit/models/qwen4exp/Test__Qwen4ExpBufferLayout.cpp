/**
 * @file Test__Qwen4ExpBufferLayout.cpp
 * @brief Exact QFN schema/admission geometry across CPU/GPU and partial placements.
 *
 * Device-free tests resolve the production declaration inventory at different
 * row/request capacities. Unowned lexical/indexer state and GPU-only partials
 * must be absent, while residual width never changes compute/head dimensions.
 */
#include "Qwen4ExpMetadataFixture.h"
#include "../../../utils/Qwen4ExpPlacementFixture.h"
#include "models/qwen4exp/Qwen4ExpGraph.h"
#include "models/qwen4exp/Qwen4ExpGraphConfigBuilder.h"
#include "models/qwen4exp/Qwen4ExpBufferLayout.h"
#include "models/GraphTypes.h"
#include "execution/local_execution/graph/GraphResolver.h"
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include "planning/ImmutableRowTablePlacementPlanner.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "execution/local_execution/orchestrators/RankOrchestrator.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <set>

namespace llaminar2::test
{
/** @test Only an admitted resident PLE consumer preserves prefill failures across response admission. */
TEST(Qwen4ExpBufferLayout, FailureLifetimeFollowsExactAdmittedConsumer)
{
    using Policy=sampling_math::DeviceGenerationForwardFailurePolicy;
    for(const auto device:{DeviceId::cpu(),DeviceId::cuda(0),DeviceId::rocm(0)})
        for(const auto tier:{ImmutableRowStoragePreference::RAM,ImmutableRowStoragePreference::MemoryMapped,
            ImmutableRowStoragePreference::VRAM})
        {
            if(device.is_cpu() && tier==ImmutableRowStoragePreference::VRAM)continue;
            GraphConfig config;config.default_device=device;
            Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(false),config);
            admitQwen4ExpTestTable(config,tier);
            Qwen4ExpGraph graph(config);
            EXPECT_EQ(graph.deviceGenerationForwardFailurePolicy(),tier==ImmutableRowStoragePreference::VRAM
                ?Policy::PreserveUntilRequestReset:Policy::ResetAtResponseAdmission);
            if(device.is_gpu())
            {
                auto missing=config;missing.ngram_table_placement.reset();Qwen4ExpGraph missingGraph(missing);
                EXPECT_THROW(missingGraph.deviceGenerationForwardFailurePolicy(),std::logic_error);
                auto foreign=config;foreign.default_device=device.is_cuda()?DeviceId::rocm(0):DeviceId::cuda(0);
                Qwen4ExpGraph foreignGraph(foreign);
                EXPECT_THROW(foreignGraph.deviceGenerationForwardFailurePolicy(),std::invalid_argument);
                missing.pp_layer_offset=4;missing.n_layers=4;Qwen4ExpGraph nonowner(missing);
                EXPECT_EQ(nonowner.deviceGenerationForwardFailurePolicy(),Policy::ResetAtResponseAdmission);
            }
        }
}
namespace
{
/** @return One pure metadata envelope for comparing the same declaration inventory. */
Qwen4ExpBufferDimensions dimensions(const GraphConfig &model, std::optional<ImmutableRowStorage> tier)
{
    return {.main_layers=static_cast<int>(model.flash_next->mainLayers()),
        .first_layer=model.pp_layer_offset, .layer_count=model.n_layers,
        .compute_width=model.d_model, .context_tokens=model.max_seq_len,
        .request_capacity=model.max_request_count, .flattened_rows=model.max_activation_rows,
        .local_query_heads=model.local_n_heads,
        .retained_mtp_candidate_rows=retainsMTPGraphCapacity(model.mtp)
            ? resolveMTPRetainedTargetQueryRows(model.mtp) : 0,
        .device=model.default_device, .ngram_lookup=tier.transform(immutableRowLookupExecution)};
}
}

/** @test Every arena name has one identity and backend/PP placement chooses only actual consumers. */
TEST(Qwen4ExpBufferLayout, InventoryIsUniqueAndPlacementDoesNotAllocateUnusedState)
{
    std::set<BufferId> ids;
    std::set<std::string> names;
    for(const auto &item:qwen4ExpBufferDeclarations())
    {
        EXPECT_TRUE(ids.insert(item.id).second);
        EXPECT_TRUE(names.insert(item.spec.name).second);
    }
    for(auto device:{DeviceId::cpu(),DeviceId::cuda(0),DeviceId::rocm(0)})
        for(int stages:{1,4,8,48})
            for(int rank=0;rank<stages;++rank)
            {
                GraphConfig model;
                model.n_layers=48/stages;model.pp_layer_offset=rank*model.n_layers;
                model.default_device=device;model.max_activation_rows=512;
                Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(),model);
                admitQwen4ExpTestTable(model);
                GraphSchema schema;
                schema.layer_buffers.push_back({"hidden",{"seq_len","d_model"},"fp32",BufferSemantic::InOut});
                applyQwen4ExpBufferLayout(schema,model);
                const auto contains=[&](const char *name) {
                    return std::any_of(schema.layer_buffers.begin(),schema.layer_buffers.end(),
                        [&](const BufferSpec &spec){return spec.name==name;});
                };
                const bool ple=model.pp_layer_offset<=1 && model.pp_layer_offset+model.n_layers>1;
                bool sparse=false;
                for(int layer=model.pp_layer_offset;layer<model.pp_layer_offset+model.n_layers;++layer)
                    sparse=sparse || (layer+1)%4==0;
                EXPECT_EQ(contains("qfn_ple_conv_history"),ple);
                EXPECT_EQ(contains("qfn_ngram_lookup_state"),ple && device.is_gpu());
                EXPECT_EQ(contains("qfn_qsa_pooled_keys"),sparse);
                EXPECT_EQ(contains("qfn_qsa_partials"),sparse && device.is_gpu());
                EXPECT_EQ(schema.layer_buffers.front().shape.back(),"qfn_trunk");
                EXPECT_THROW(applyQwen4ExpBufferLayout(schema,model),std::invalid_argument);
            }
}
/** @test M512/M1024/M2048 and request counts have exact independent storage formulas. */
TEST(Qwen4ExpBufferLayout, FlatRowsAndRequestCursorsAreIndependentAndUseOneSharedPartialBank)
{
    for(auto device:{DeviceId::cpu(),DeviceId::cuda(0),DeviceId::rocm(0)})
        for(int rows:{1,15,512,1024,2048})
            for(int requests:{1,3,8})
            {
                GraphConfig model;
                model.default_device=device;model.max_activation_rows=rows;
                model.max_request_count=requests;model.max_seq_len=65536;
                Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(),model);
                admitQwen4ExpTestTable(model);
                GraphResolverConfig resolver;
                resolver.seq_len=rows;resolver.d_model=2560;resolver.local_n_heads=12;
                applyQwen4ExpBufferGeometry(resolver,model);
                EXPECT_EQ(resolver.custom_formulas.at("qfn_trunk"),10240u);
                EXPECT_EQ(resolver.custom_formulas.at("qfn_native_words"),16u*90/4);
                EXPECT_EQ(resolver.custom_formulas.at("qfn_qsa_requests"),12u*requests);
                EXPECT_EQ(resolver.custom_formulas.at("qfn_pools"),16384u);
                EXPECT_EQ(resolver.custom_formulas.at("qfn_replay_words"),
                    requests*sizeof(qwen4exp::QwenSparseReplay)/4+
                    (device.is_gpu()?rows*sizeof(attention::AttentionDeviceParams)/4:0));
                EXPECT_EQ(resolver.custom_formulas.at("qfn_partial_words"),device.is_gpu()
                    ?12u*qwen4exp::QwenSparsePartitions*sizeof(qwen4exp::QwenSparseAttentionPartial)/4:0u);
            }
}
} // namespace llaminar2::test

namespace llaminar2::test
{
/** @test Candidate key retention multiplies sparse layers and flattened verifier rows exactly once. */
TEST(Qwen4ExpBufferLayout, RetainedCandidateKeysFollowExactMTPEnvelope)
{
    for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
        for (const int requests : {1, 3, 8})
            for (const int depth : {0, 1, 7, 15})
            {
                GraphConfig model;
                model.default_device = device;
                model.max_activation_rows = 512;
                model.max_seq_len = 65536;
                model.max_request_count = requests;
                model.mtp.enabled = false;
                model.mtp.graph_capacity_draft_tokens = depth;
                model.mtp.max_request_batch = requests;
                Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(), model);
                admitQwen4ExpTestTable(model);
                GraphSchema schema;
                applyQwen4ExpBufferLayout(schema, model);
                const auto bank = std::find_if(schema.layer_buffers.begin(), schema.layer_buffers.end(),
                    [](const BufferSpec &spec) { return spec.name == "qfn_qsa_candidate_raw_keys"; });
                EXPECT_EQ(bank != schema.layer_buffers.end(), depth > 0);
                const auto counter = std::find_if(schema.layer_buffers.begin(), schema.layer_buffers.end(),
                    [](const BufferSpec &spec) { return spec.name == "qfn_qsa_candidate_copy_bytes"; });
                EXPECT_EQ(counter != schema.layer_buffers.end(), depth > 0);
                for(const auto* name:{"qfn_qsa_transaction_prior","qfn_qsa_transaction_prior_traffic"}) {
                    const auto prior=std::find_if(schema.layer_buffers.begin(),schema.layer_buffers.end(),
                        [&](const BufferSpec& spec){return spec.name==name;});
                    EXPECT_EQ(prior!=schema.layer_buffers.end(),depth>0);
                    if(prior!=schema.layer_buffers.end()) {
                        EXPECT_EQ(prior->dtype,"int32");
                        EXPECT_EQ(prior->shape,(std::vector<std::string>{"qfn_qsa_layers","qfn_requests",
                            std::string(name)=="qfn_qsa_transaction_prior" ? "qfn_qsa_prior_words":"2"}));
                    }
                }
                GraphResolverConfig resolver;
                resolver.seq_len = 512;
                resolver.d_model = 2560;
                resolver.local_n_heads = 24;
                applyQwen4ExpBufferGeometry(resolver, model);
                EXPECT_EQ(resolver.custom_formulas.at("qfn_qsa_prior_words"),388u);
                EXPECT_EQ(resolver.custom_formulas.at("qfn_qsa_layers"), 12u);
                EXPECT_EQ(resolver.custom_formulas.at("qfn_mtp_candidate_rows"),
                    depth > 0 ? requests * (depth + 1) : 0);
                if (depth > 0)
                {
                    ASSERT_NE(counter, schema.layer_buffers.end());
                    EXPECT_EQ(counter->dtype, "int32");
                    const auto counterAllocation = BufferAllocator::resolve(*counter, resolver);
                    EXPECT_EQ(counterAllocation.shape, (std::vector<std::size_t>{12u,
                        static_cast<std::size_t>(requests), 2u}));
                    EXPECT_EQ(counterAllocation.shape.back() * sizeof(std::int32_t), sizeof(std::uint64_t));
                    const auto allocation = BufferAllocator::resolve(*bank, resolver);
                    EXPECT_EQ(allocation.shape, (std::vector<std::size_t>{12u,
                        static_cast<std::size_t>(requests * (depth + 1)), 128u}));
                }
            }
}
} // namespace llaminar2::test

namespace llaminar2::test
{
/** @test A real graph cannot allocate lookup state before concrete placement or for another consumer. */
TEST(Qwen4ExpBufferLayout, GraphRequiresExactResolvedPLEConsumer)
{
    GraphConfig model;
    model.default_device=DeviceId::cuda(0);
    Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(), model);
    GraphSchema absent;
    EXPECT_THROW(applyQwen4ExpBufferLayout(absent,model),std::invalid_argument);
    admitQwen4ExpTestTable(model,ImmutableRowStoragePreference::VRAM);
    model.default_device=DeviceId::cuda(1);
    GraphSchema foreign;
    EXPECT_THROW(applyQwen4ExpBufferLayout(foreign,model),std::invalid_argument);
    model.ngram_table_placement.reset();
    model.pp_layer_offset=2; model.n_layers=2;
    GraphSchema nonconsumer;
    EXPECT_NO_THROW(applyQwen4ExpBufferLayout(nonconsumer,model));
}

/**
 * @test Resident mode omits all hosted scratch and every tier delta equals the same arena declarations.
 *
 * Higher-priority base geometry remains unchanged across M1 through M2048 and
 * retained verifier capacities. There is no anonymous staging allowance or
 * request-count-derived approximation in the final lookup BOM.
 */
TEST(Qwen4ExpBufferLayout, ConcreteModeDeltaMatchesDeclarationsAndNeverChangesOtherOwners)
{
    for(const auto device:{DeviceId::cpu(),DeviceId::cuda(0),DeviceId::rocm(1)})
        for(const int rows:{1,15,512,1024,2048})
            for(const int depth:{0,1,15})
            {
                GraphConfig model;
                model.default_device=device; model.max_activation_rows=rows;
                model.max_seq_len=65536; model.max_request_count=3;
                model.mtp.enabled=false; model.mtp.graph_capacity_draft_tokens=depth;
                model.mtp.max_request_batch=3;
                Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(),model);
                // This fixture declares an unsharded participant. Metadata
                // supplies global heads; placement owns the local head count.
                model.local_n_heads=model.n_heads;
                const auto geometry=Qwen4ExpMemoryGeometry::fromConfig(*model.flash_next);
                GraphResolverConfig resolver;
                resolver.seq_len=rows; resolver.d_model=model.d_model;
                resolver.local_n_heads=model.local_n_heads;
                const GraphSchema common{};
                const auto unresolved=dimensions(model,std::nullopt);
                const auto base=qwen4ExpArenaExtensionBytes(common,resolver,geometry,unresolved);
                EXPECT_THROW(qwen4ExpLookupArenaBytes(resolver,geometry,unresolved),std::invalid_argument);
                for(const auto tier:{ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped,ImmutableRowStorage::VRAM})
                {
                    const auto concrete=dimensions(model,tier);
                    if(!device.is_gpu() && tier==ImmutableRowStorage::VRAM)
                    {
                        EXPECT_THROW(qwen4ExpLookupArenaBytes(resolver,geometry,concrete),std::invalid_argument);
                        continue;
                    }
                    const auto total=qwen4ExpArenaExtensionBytes(common,resolver,geometry,concrete);
                    const auto delta=qwen4ExpLookupArenaBytes(resolver,geometry,concrete);
                    EXPECT_EQ(total,base+delta);
                    GraphSchema schema;
                    applyQwen4ExpBufferLayout(schema,geometry,concrete);
                    const auto has=[&](const char *name) {
                        return std::any_of(schema.layer_buffers.begin(),schema.layer_buffers.end(),
                            [&](const BufferSpec &spec){return spec.name==name;});
                    };
                    EXPECT_EQ(has("qfn_ngram_native_rows"),tier!=ImmutableRowStorage::VRAM);
                    EXPECT_EQ(has("qfn_ngram_lookup_state"),device.is_gpu() && tier!=ImmutableRowStorage::VRAM);
                    if(tier==ImmutableRowStorage::VRAM) EXPECT_EQ(delta,0u);
                    else EXPECT_GT(delta,0u);
                }
            }
}
} // namespace llaminar2::test

namespace llaminar2::test
{
/** @test Factory and rank projections retain requested policy separately from the exact sealed result. */
TEST(Qwen4ExpBufferLayout, RuntimeProjectionsPreserveRequestedAndResolvedNgramPolicies)
{
    for(const auto preference:{ImmutableRowStoragePreference::Automatic,ImmutableRowStoragePreference::VRAM,
                              ImmutableRowStoragePreference::RAM,ImmutableRowStoragePreference::MemoryMapped})
    {
        GraphConfig model; model.default_device=DeviceId::cuda(0);
        Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(),model);
        admitQwen4ExpTestTable(model,preference);
        RuntimeConfig runtime;
        runtime.ngram_table_policy={preference,65536,17};
        runtime.ngram_table_placement=model.ngram_table_placement;
        const auto device=InferenceRunnerConfig::fromRuntime(runtime);
        const auto rank=RankOrchestrator::Config::fromRuntime(runtime);
        EXPECT_EQ(device.ngram_table_policy,runtime.ngram_table_policy);
        EXPECT_EQ(rank.ngram_table_policy,runtime.ngram_table_policy);
        EXPECT_EQ(device.ngram_table_placement,runtime.ngram_table_placement);
        EXPECT_EQ(rank.ngram_table_placement,runtime.ngram_table_placement);
    }
}
} // namespace llaminar2::test
