/**
 * @file Test__MoEProjectionArenaGeometry.cpp
 * @brief Device-free proof that projection admission and graph storage are identical.
 *
 * All source codebooks and floating formats use the real schema/resolver. The
 * independent arithmetic checks every concurrent buffer, including the MTP
 * layer envelope, and ensures the old whole-expert bank is not also retained.
 * QFN checks use canonical GGUF main/predictor projections and the actual graph
 * registry injection, keeping compute-H expert output separate from HC residuals.
 */
#include "execution/moe/MoEProjectionArenaGeometry.h"
#include "execution/local_execution/graph/GraphResolver.h"
#include "models/qwen35moe/Qwen35MoESchema.h"
#include "execution/local_execution/graph/GraphBuilderRegistry.h"
#include "models/qwen35moe/Qwen35MoEGraph.h"
#include "planning/ActivationMemoryEstimator.h"
#include "collective/DeviceCountedAllGather.h"
#include "tensors/NativeQuantisedFormatInfo.h"
#include "../../utils/Qwen4ExpPlanningGGUFFixture.h"
#include "../models/qwen4exp/Qwen4ExpSidecarFixture.h"
#include "../../mocks/MockLocalTPContext.h"
#include "loaders/ModelContext.h"
#include "models/qwen4exp/Qwen4ExpGraphConfigBuilder.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <future>
#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>

using namespace llaminar2;

namespace
{
/** @brief Author a complete original matrix with symmetric native cliques.
 * @param backend One homogeneous GPU family. @param degree Original visible endpoints.
 * @param clique Size of each connected clique;1 means all edges are absent.
 * @return Immutable-test inventory with explicit sparse physical ordinals. */
RankInventory countedInventory(DeviceType backend, std::size_t degree, std::size_t clique)
{
    RankInventory rank;
    rank.rank = 0;
    for (std::size_t i = 0; i < degree; ++i)
        rank.gpus.push_back({.type = backend, .local_device_id = int(i * 3u + 2u)});
    auto &matrix = backend == DeviceType::CUDA ? rank.p2p_cuda : rank.p2p_rocm;
    auto &count = backend == DeviceType::CUDA ? rank.p2p_cuda_count : rank.p2p_rocm_count;
    count = int(degree);
    matrix.resize(degree * degree);
    for (std::size_t i = 0; i < degree; ++i)
        for (std::size_t j = 0; j < degree; ++j)
            matrix[i * degree + j] = i != j && i / clique == j / clique;
    return rank;
}
/** @return Reversed declared membership, independent of inventory vector order. */
std::vector<DeviceId> countedMembers(const RankInventory &rank)
{
    std::vector<DeviceId> members;
    for (auto it = rank.gpus.rbegin(); it != rank.gpus.rend(); ++it)
        members.emplace_back(it->type, it->local_device_id);
    return members;
}
}

/** @test Every directed original edge contributes exactly its selected canonical physical owners. */
TEST(MoEProjectionArenaGeometry, CountedExchangeBOMIsExactAndOverflowChecked)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    for (const std::size_t degree : {2u, 3u, 4u, 8u, 17u})
    for (const std::size_t clique : {std::size_t(1), std::size_t(2), degree})
    for (const std::size_t capacity : {1u, 257u, 1024u * 1024u})
    {
        const auto rank = countedInventory(backend, degree, clique);
        const auto members = countedMembers(rank);
        const auto topology = DeviceCountedAllGatherTopology::resolve(rank, members);
        EXPECT_THROW(topology.memoryFor(0, 0), std::invalid_argument);
        EXPECT_EQ(topology.coverage(), clique == 1 ? PeerAccessCoverage::None
            : clique >= degree ? PeerAccessCoverage::Complete : PeerAccessCoverage::Partial);
        EXPECT_TRUE(topology.sameRoutes(DeviceCountedAllGatherTopology::resolve(rank, members)));
        for (std::size_t p = 0; p < degree; ++p)
        {
            for (std::size_t q = 0; q < degree; ++q)
            {
                if (p == q) continue;
                EXPECT_EQ(topology.payloadStorage(p,q),DeviceCountedAllGatherTransport::NativeCollective);
            }
            const auto quote = topology.memoryFor(p, capacity);
            EXPECT_EQ(quote.host_bytes,0u);
            EXPECT_EQ(quote.device_bytes,NativeCountedAllGatherLayout(degree).bytes());
            EXPECT_THROW(topology.payloadStorage(p, p), std::out_of_range);
            EXPECT_THROW(topology.memoryFor(p, std::numeric_limits<std::size_t>::max()), std::invalid_argument);
        }
        EXPECT_THROW(topology.memoryFor(degree, capacity), std::out_of_range);
    }
}

/** @test Original observations must be total and symmetric before any physical claim is possible. */
TEST(MoEProjectionArenaGeometry, CountedTopologyRejectsIncompleteAndOneWayObservations)
{
    auto rank = countedInventory(DeviceType::ROCm, 4, 2);
    const auto members = countedMembers(rank);
    const auto resolve = [&](const RankInventory &observed, std::span<const DeviceId> selected) {
        return DeviceCountedAllGatherTopology::resolve(observed, selected);
    };
    EXPECT_THROW(resolve(rank, std::span<const DeviceId>{}), std::invalid_argument);
    EXPECT_THROW(resolve(rank, std::span(members).first(1)), std::invalid_argument);
    auto duplicate = members;duplicate[1] = duplicate[0];
    EXPECT_THROW(resolve(rank, duplicate), std::invalid_argument);
    auto foreign = members;foreign[0] = DeviceId::rocm(999);
    EXPECT_THROW(resolve(rank, foreign), std::invalid_argument);
    auto mixed = members;mixed[0] = DeviceId::cuda(mixed[0].ordinal);
    EXPECT_THROW(resolve(rank, mixed), std::invalid_argument);
    auto malformed = rank;malformed.p2p_rocm.pop_back();
    EXPECT_THROW(resolve(malformed, members), std::invalid_argument);
    malformed = rank;malformed.p2p_rocm_count = 0;
    EXPECT_THROW(resolve(malformed, members), std::invalid_argument);
    malformed = rank;malformed.rank = -1;
    EXPECT_THROW(resolve(malformed, members), std::invalid_argument);
    for (const bool direction : {false, true})
    {
        malformed = rank;
        malformed.p2p_rocm[0 * 4 + 1] = direction;
        malformed.p2p_rocm[1 * 4 + 0] = !direction;
        EXPECT_THROW(resolve(malformed, members), std::invalid_argument);
    }
    const auto frozen = resolve(rank, members);
    rank.p2p_rocm.assign(16, false);
    EXPECT_FALSE(frozen.sameRoutes(resolve(rank, members)));
    EXPECT_EQ(frozen.payloadStorage(2,3),DeviceCountedAllGatherTransport::NativeCollective);
}

namespace
{
    /** @brief Metadata-only two-main-layer plus one-sidecar fixture, with unequal widths. */
    ModelMemoryProfile profileFor(const std::string &format)
    {
        ModelMemoryProfile p;
        p.architecture = "qwen35moe";
        p.n_layers = 3; p.mtp_layer_count = 1;
        p.d_model = 2048; p.d_ff = 1024;
        p.n_heads = 16; p.n_kv_heads = 2; p.head_dim = 128; p.vocab_size = 128;
        p.expert_count = 16; p.expert_used_count = 8; p.expert_feed_forward_length = 512;
        // The production estimator selects the hybrid schema from source
        // tensor geometry, not the architecture spelling. Include that union
        // before the expert tensors so this really exercises its MoE branch.
        const auto projection = [&](const char *name, std::size_t n, std::size_t k) {
            p.tensors.push_back({.name = name, .quant_type = "F32",
                .elements = n * k, .K = k, .layer_index = 0});
        };
        projection("blk.0.attn_q.weight", 4096, 2048);
        projection("blk.0.attn_gate.weight", 2048, 2048);
        projection("blk.0.attn_qkv.weight", 4096, 2048);
        projection("blk.0.ssm_out.weight", 2048, 2048);
        projection("blk.0.ssm_alpha.weight", 16, 2048);
        projection("blk.0.ssm_beta.weight", 16, 2048);
        for (int layer = 0; layer < p.n_layers; ++layer)
        {
            const std::size_t width = layer == 2 ? 1024u : 512u;
            p.tensors.push_back({.name = "blk." + std::to_string(layer) + ".ffn_down_exps.weight",
                .quant_type = format, .elements = width * p.d_model * p.expert_count,
                .K = width, .layer_index = layer});
        }
        return p;
    }
}

/** @test Search and admission share the exact source-column partition invariant. */
TEST(MoEProjectionArenaGeometry, IntegralOutputPartitionIsSharedBySearchAndAdmission)
{
    for (const int columns : {256, 384, 512, 1536, 2048, 6144})
    for (int degree = 2; degree <= 8; ++degree)
    {
        SCOPED_TRACE(::testing::Message() << "columns=" << columns << " degree=" << degree);
        auto profile = profileFor("Q6_K");
        profile.d_model = columns;
        for (auto &tensor : profile.tensors)
            if (tensor.name.find("ffn_down_exps") != std::string::npos)
                tensor.elements = tensor.K * columns * profile.expert_count;
        const bool integral = columns % degree == 0;
        EXPECT_EQ(MoEProjectionArenaGeometry::hasIntegralOutputPartition(columns, degree), integral);
        if (integral)
        {
            const auto layout = MoEProjectionArenaGeometry::resolve(profile, degree);
            EXPECT_EQ(layout.banks()[3].words_per_row, columns / degree);
            EXPECT_EQ(layout.banks()[4].words_per_row, columns);
        }
        else EXPECT_THROW(MoEProjectionArenaGeometry::resolve(profile, degree), std::invalid_argument);
    }
    EXPECT_THROW(MoEProjectionArenaGeometry::hasIntegralOutputPartition(0, 2), std::invalid_argument);
    EXPECT_THROW(MoEProjectionArenaGeometry::hasIntegralOutputPartition(512, 1), std::invalid_argument);
}

/** @test All formats and native degrees price exactly the buffers the real resolver declares. */
TEST(MoEProjectionArenaGeometry, EveryFormatReplacesWholeExpertStorageExactly)
{
    std::vector<std::string> formats{"F16", "BF16", "F32"};
    for (const auto &format : native_quantised_formats::kAllSourceFormats)
        formats.emplace_back(format.quant_type);
    for (const auto &format : formats)
    for (const int degree : {2, 4, 8})
    {
        SCOPED_TRACE(format + "/" + std::to_string(degree));
        const auto profile = profileFor(format);
        const auto layout = MoEProjectionArenaGeometry::resolve(profile, degree);
        const bool floating = format == "F16" || format == "BF16" || format == "F32";
        const std::size_t packet = 8u * (1u + (floating ? 1024u : 1024u / 4u + 1024u / 32u));
        const std::array<std::size_t, 5> expected{packet, packet * degree,
            8u * 2048u / degree, 2048u / degree, 2048u};
        auto schema = Qwen35MoESchemaFactory{}.createSchema();
        const auto original = schema.layer_buffers.size();
        layout.replaceWholeExpertSchema(schema);
        EXPECT_EQ(schema.layer_buffers.size(), original + 4);
        EXPECT_EQ(std::count_if(schema.layer_buffers.begin(), schema.layer_buffers.end(),
            [](const auto &spec) { return spec.name == "moe_canonical_route_contributions"; }), 0);
        EXPECT_THROW(layout.replaceWholeExpertSchema(schema), std::invalid_argument);
        GraphResolverConfig resolver;
        layout.bindResolver(resolver);
        for (const int rows : {1, 16, 32, 512, 1536})
        {
            resolver.custom_formulas["moe_activation_rows"] = rows;
            std::size_t bytes = 0;
            for (std::size_t index = 0; index < layout.banks().size(); ++index)
            {
                const auto &bank = layout.banks()[index];
                EXPECT_EQ(bank.words_per_row, expected[index]);
                EXPECT_EQ(resolver.buffer_name_to_id.at(std::string(bank.name)), bank.id);
                const auto it = std::find_if(schema.layer_buffers.begin(), schema.layer_buffers.end(),
                    [&](const auto &spec) { return spec.name == bank.name; });
                ASSERT_NE(it, schema.layer_buffers.end());
                const auto resolved = BufferAllocator::resolve(*it, resolver);
                ASSERT_EQ(resolved.shape.size(), 2);
                EXPECT_EQ(resolved.shape[0], rows);
                EXPECT_EQ(resolved.shape[1], expected[index]);
                EXPECT_EQ(resolved.dtype, "fp32");
                bytes += resolved.shape[0] * resolved.shape[1] * sizeof(float);
            }
            EXPECT_EQ(layout.bytes(rows), bytes);
            EXPECT_EQ(layout.packetCapacityBytes(rows), rows * packet * sizeof(std::uint32_t));
            ActivationGraphMemoryGeometry geometry{
                .resident_graph_rows = rows, .local_d_ff = 512, .local_n_heads = 8,
                .local_n_kv_heads = 1, .total_shards = degree, .mtp_target_query_rows = 16};
            for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
            {
                geometry.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
                const auto whole = ActivationMemoryEstimator::estimate(profile, geometry, device);
                geometry.routed_compute_policy = RoutedExpertComputePolicy::GateUpOwnedDownColumns;
                const auto split = ActivationMemoryEstimator::estimate(profile, geometry, device);
                const auto admitted_rows = static_cast<std::size_t>(std::max(rows, 16));
                EXPECT_EQ(split, whole - admitted_rows * (8u + degree) * 2048u * sizeof(float) +
                    layout.bytes(admitted_rows));
                if (!floating && degree == 2) EXPECT_LT(split, whole);
            }
        }
    }
}

/** @test Mixed-format sidecars determine the true maximum, not a model-level quantization label. */
TEST(MoEProjectionArenaGeometry, FloatingSidecarUsesItsActualPacketWidth)
{
    auto profile = profileFor("Q6_K");
    profile.tensors.back().quant_type = "BF16";
    const auto layout = MoEProjectionArenaGeometry::resolve(profile, 2);
    EXPECT_EQ(layout.banks()[0].words_per_row, 8u * 1025u);
}

/** @test Incomplete metadata and unrepresentable geometries cannot enter physical admission. */
TEST(MoEProjectionArenaGeometry, RejectsInvalidInputsWithoutInventingCapacity)
{
    const auto valid = profileFor("IQ3_S");
    for (const int degree : {0, 1, 3, -2})
        EXPECT_THROW(MoEProjectionArenaGeometry::resolve(valid, degree), std::invalid_argument);
    auto profile = valid;
    profile.tensors.pop_back();
    EXPECT_THROW(MoEProjectionArenaGeometry::resolve(profile, 2), std::invalid_argument);
    profile = valid; profile.tensors.push_back(profile.tensors.back());
    EXPECT_THROW(MoEProjectionArenaGeometry::resolve(profile, 2), std::invalid_argument);
    profile = valid; profile.tensors.back().quant_type = "Unknown";
    EXPECT_THROW(MoEProjectionArenaGeometry::resolve(profile, 2), std::invalid_argument);
    profile = valid; ++profile.tensors.back().elements;
    EXPECT_THROW(MoEProjectionArenaGeometry::resolve(profile, 2), std::invalid_argument);
    const auto layout = MoEProjectionArenaGeometry::resolve(valid, 2);
    EXPECT_THROW(layout.bytes(0), std::invalid_argument);
    EXPECT_THROW(layout.bytes(std::numeric_limits<std::size_t>::max()), std::overflow_error);
}

/** @test The real registry constructs a declaration before injecting model metadata. */
TEST(MoEProjectionArenaGeometry, RegisteredBuilderWaitsForSourceInjectionBeforeResolvingSchema)
{
    GraphConfig config;
    config.moe.routed_compute_policy = RoutedExpertComputePolicy::GateUpOwnedDownColumns;
    std::shared_ptr<IGraphBuilder> builder;
    ASSERT_NO_THROW(builder = GraphBuilderRegistry::create("qwen35moe", config, nullptr));
    ASSERT_NE(builder, nullptr);
    EXPECT_THROW(builder->getSchema(), std::logic_error);
    EXPECT_THROW(builder->setModelContext(nullptr), std::invalid_argument);
    // Failed injection still cannot fall back to the whole-expert schema.
    EXPECT_THROW(builder->getSchema(), std::logic_error);
    config.moe.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
    ASSERT_NO_THROW(builder = GraphBuilderRegistry::create("qwen35moe", config, nullptr));
    EXPECT_NO_THROW(builder->getSchema());
}

namespace
{
    /**
     * @brief Add complete original down directories without allocating tensor payloads.
     * @param directory Original synthetic GGUF directory, before profile extraction.
     * @param layers Number of original main/predictor source coordinates to populate.
     */
    void addQFNDownDirectory(GGUFModel &directory, int layers)
    {
        for (int layer = 0; layer < layers; ++layer)
        {
            GGUFTensorInfo tensor{
                .name = "blk." + std::to_string(layer) + ".ffn_down_exps.weight",
                .dimensions = {640, 2560, 512}, .type = GGUFTensorType::Q8_0};
            test::Qwen4ExpSidecarFixture::sealBytes(tensor);
            directory.tensors.push_back(std::move(tensor));
        }
    }

    /**
     * @brief Check all five original-source banks against independent QFN byte arithmetic.
     * @param profile Actual canonical GGUF profile, not a hand-authored layer-count projection.
     */
    void expectQFNProjectionBanks(const ModelMemoryProfile &profile)
    {
        ASSERT_EQ(profile.d_model, 2560);
        ASSERT_TRUE(profile.flash_next);
        ASSERT_EQ(profile.flash_next->residual_width, 10240u);
        for (const int degree : {2, 4, 8})
        {
            SCOPED_TRACE(::testing::Message() << "TP=" << degree);
            const auto layout = MoEProjectionArenaGeometry::resolve(profile, degree);
            // Per route: original route-ID word + 640 Q8 bytes + 20 FP32 scales.
            constexpr std::size_t packet_words = 10u * (1u + 640u / 4u + 640u / 32u);
            const std::array<std::size_t, 5> expected{
                packet_words, packet_words * degree, 10u * 2560u / degree,
                2560u / degree, 2560u};
            std::size_t words = 0;
            for (std::size_t index = 0; index < expected.size(); ++index)
            {
                EXPECT_EQ(layout.banks()[index].words_per_row, expected[index]);
                words += expected[index];
            }
            for (const std::size_t rows : {1u, 9u, 15u, 512u})
                EXPECT_EQ(layout.bytes(rows), rows * words * sizeof(std::uint32_t));
        }
    }
}

/** @test The canonical QFN main48-plus-inline-predictor projection retains original layer48. */
TEST(MoEProjectionArenaGeometry, FlashNextInlinePredictorUsesCanonicalTotalLayerCount)
{
    auto directory = test::qwen4MetadataFixture(true);
    addQFNDownDirectory(directory, 49);
    const auto profile = ModelMemoryProfile::fromGGUF(directory);
    ASSERT_EQ(profile.n_layers, 48);
    ASSERT_EQ(profile.mainLayerCount(), 48);
    ASSERT_EQ(profile.mtp_layer_count, 1);
    ASSERT_EQ(profile.totalLayerCount(), 49);
    ASSERT_EQ(profile.tensors.back().layer_index, 48);
    ASSERT_EQ(profile.tensors.back().K, 640u);
    expectQFNProjectionBanks(profile);

    auto missing = profile;
    missing.tensors.pop_back();
    EXPECT_THROW(MoEProjectionArenaGeometry::resolve(missing, 4), std::invalid_argument);
    auto duplicate = profile;
    duplicate.tensors.push_back(profile.tensors.back());
    EXPECT_THROW(MoEProjectionArenaGeometry::resolve(duplicate, 4), std::invalid_argument);
    auto foreign = profile;
    foreign.tensors.back().layer_index = profile.totalLayerCount();
    EXPECT_THROW(MoEProjectionArenaGeometry::resolve(foreign, 4), std::invalid_argument);
    auto malformed = profile;
    ++malformed.tensors.back().elements;
    EXPECT_THROW(MoEProjectionArenaGeometry::resolve(malformed, 4), std::invalid_argument);
}

/** @test Active sidecar planning and its main-only runtime graph resolve their own original source inventories. */
TEST(MoEProjectionArenaGeometry, FlashNextSharedPredictorPlanningAndMainGraphInventoriesStayExact)
{
    test::Qwen4ExpSidecarFixture source;
    addQFNDownDirectory(source.main, 48);
    const auto planning = ModelMemoryProfile::fromGGUF(
        source.main, source.predictor, SharedPredictorMaterialization::Active);
    ASSERT_EQ(planning.mainLayerCount(), 48);
    ASSERT_EQ(planning.totalLayerCount(), 49);
    ASSERT_EQ(std::count_if(planning.tensors.begin(), planning.tensors.end(), [](const auto &tensor) {
        return tensor.name == "blk.48.ffn_down_exps.weight";
    }), 1);
    expectQFNProjectionBanks(planning);

    // bindProjectionArena uses the original MAIN directory, not fromLoader's
    // authenticated but inactive sidecar metadata. Never require phantom down
    // tensors or waive missing tensors in a complete active inventory.
    const auto main = ModelMemoryProfile::fromGGUF(source.main);
    ASSERT_EQ(main.totalLayerCount(), 48);
    expectQFNProjectionBanks(main);
    const auto inactive = ModelMemoryProfile::fromGGUF(
        source.main, source.predictor, SharedPredictorMaterialization::Inactive);
    ASSERT_EQ(inactive.totalLayerCount(), 49);
    EXPECT_THROW(MoEProjectionArenaGeometry::resolve(inactive, 4), std::invalid_argument);
}

/** @test Actual registry/model-context injection resolves the main-only QFN schema without an inactive sidecar bank. */
TEST(MoEProjectionArenaGeometry, FlashNextMainGraphBindsOriginalDirectoryBeforeSchemaResolution)
{
    test::Qwen4ExpPlanningGGUFFixture source;
    auto context = ModelContext::createForTesting(source.path());
    ASSERT_NE(context, nullptr);
    context->concreteLoader().setUseMmap(false);
    ASSERT_TRUE(context->concreteLoader().loadModel(source.path()));
    // The tail owns original main layers24..47 and no PLE layer1. Its
    // projection binding still validates the complete original main48 source;
    // schema ownership must not invent an unrelated table placement in this
    // metadata-only regression.
    const auto original = ModelMemoryProfile::fromGGUF(context->model());
    ASSERT_EQ(original.mainLayerCount(), 48);
    ASSERT_EQ(original.totalLayerCount(), 48);
    GraphConfig config;
    config.pp_layer_offset = 24;
    config.n_layers = 24;
    config.default_device = DeviceId::rocm(0);
    config.local_n_heads = 6;
    config.local_n_kv_heads = 2;
    Qwen4ExpGraphConfigBuilder::populateFromDirectory(context->model(), config);
    ASSERT_EQ(config.n_layers, 24);
    ASSERT_EQ(config.pp_layer_offset, 24);
    config.moe.routed_compute_policy = RoutedExpertComputePolicy::GateUpOwnedDownColumns;
    test::MockLocalTPContext cohort;
    cohort.setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1),
        GlobalDeviceAddress::rocm(2), GlobalDeviceAddress::rocm(3)});
    cohort.setBackend(CollectiveBackendType::RCCL);
    config.tp_ctx = &cohort;
    auto builder = GraphBuilderRegistry::create("qwen4exp", config, nullptr);
    ASSERT_NE(builder, nullptr);
    EXPECT_THROW(builder->getSchema(), std::logic_error);
    ASSERT_NO_THROW(builder->setModelContext(context));
    const auto schema = builder->getSchema();
    EXPECT_EQ(schema.layer_template_names.size(), 24u);
    EXPECT_EQ(std::count_if(schema.layer_buffers.begin(), schema.layer_buffers.end(), [](const auto &bank) {
        return bank.name == "moe_projection_route_columns";
    }), 1);
    EXPECT_EQ(std::count_if(schema.layer_buffers.begin(), schema.layer_buffers.end(), [](const auto &bank) {
        return bank.name == "moe_canonical_route_contributions";
    }), 0);
    const auto resolver = builder->getResolverConfig(9);
    EXPECT_EQ(resolver.custom_formulas.at("moe_projection_local_columns_words"), 640u);
    EXPECT_EQ(resolver.custom_formulas.at("moe_projection_gathered_columns_words"), 2560u);
}

/** @test Every logical degree through eight and larger metadata tables have exact O(N) storage, without a policy cap. */
TEST(MoEProjectionArenaGeometry, NativeCountedCompletionLayoutHasNoParticipantCap)
{
    for(std::size_t degree=1;degree<=32;++degree) {
        const NativeCountedAllGatherLayout layout(degree);
        EXPECT_EQ(layout.extents(),32u);
        EXPECT_EQ(layout.operands(),32u+8u*degree);
        EXPECT_EQ(layout.receipts(),32u+72u*degree);
        EXPECT_EQ(layout.counters(),32u+136u*degree);
        EXPECT_EQ(layout.bytes(),32u+144u*degree);
    }
    EXPECT_THROW(NativeCountedAllGatherLayout(0),std::invalid_argument);
    EXPECT_THROW(NativeCountedAllGatherLayout(std::numeric_limits<std::size_t>::max()),std::invalid_argument);
}

/** @test Replay the production frame body across dirty reuse, data reset, stale receipts and absorbing abort. */
TEST(MoEProjectionArenaGeometry, NativeCountedCompletionRecordedFrameReplay)
{
    for(std::size_t degree=1;degree<=8;++degree) {
        const NativeCountedAllGatherLayout layout(degree);
        std::vector<std::uint64_t> memory(layout.bytes()/sizeof(std::uint64_t),0);
        auto* frame=reinterpret_cast<NativeCountedAllGatherFrame*>(memory.data());
        auto* bytes=reinterpret_cast<unsigned char*>(memory.data());
        auto* extents=reinterpret_cast<std::uint64_t*>(bytes+layout.extents());
        auto* operands=reinterpret_cast<ncclLlaminarCompletionOperand*>(bytes+layout.operands());
        auto* receipts=reinterpret_cast<ncclLlaminarByteCompletion*>(bytes+layout.receipts());
        std::uint32_t abort=0;int original=0;
        const ncclLlaminarCompletionAuthority authority{&abort,&original};
        for(std::uint64_t replay=1;replay<=20;++replay) {
            ASSERT_TRUE(prepareNativeCountedAllGatherInvocation(memory.data(),degree,19,4099));
            EXPECT_EQ(frame->sequence,replay);
            for(std::size_t root=0;root<degree;++root) {
                extents[root]=replay%4==0?0:(replay+root)%2?257:4099;
                EXPECT_EQ(operands[root].sequence,&frame->sequence);
                receipts[root]={19,4099,replay,extents[root],authority.abortFlag,authority.deviceCommunicator,
                    operands[root].message,ncclLlaminarCompletionPhase::Completed,0};
            }
            EXPECT_TRUE(completeNativeCountedAllGatherInvocation(memory.data(),degree,authority,0));
            EXPECT_EQ(frame->first_error,0);
        }
        // Reset only packet/count data. A new invocation cannot accept a prior
        // receipt even when every payload byte count happens to be identical.
        ASSERT_TRUE(prepareNativeCountedAllGatherInvocation(memory.data(),degree,19,4099));
        EXPECT_FALSE(completeNativeCountedAllGatherInvocation(memory.data(),degree,authority,0));
        EXPECT_NE(frame->first_error,0);
        EXPECT_FALSE(prepareNativeCountedAllGatherInvocation(memory.data(),degree,19,4099));
        // Independent admitted frame: abort and wrap are absorbing as well.
        std::fill(memory.begin(),memory.end(),0);
        ASSERT_TRUE(prepareNativeCountedAllGatherInvocation(memory.data(),degree,19,4099));
        EXPECT_FALSE(completeNativeCountedAllGatherInvocation(memory.data(),degree,authority,1));
        std::fill(memory.begin(),memory.end(),0);frame->sequence=~std::uint64_t(0);
        EXPECT_FALSE(prepareNativeCountedAllGatherInvocation(memory.data(),degree,19,4099));
        EXPECT_EQ(frame->sequence,~std::uint64_t(0));
    }
}

/** @test The actual interface owner revokes direct-issued native borrows and distinguishes equal-membership contexts. */
TEST(MoEProjectionArenaGeometry, NativeCountedCompletionContextBorrowRetirement)
{
    NativeLocalTPContextBorrow retained;
    {
        auto original=std::make_unique<test::MockLocalTPContext>();
        auto other=std::make_unique<test::MockLocalTPContext>();
        retained=original->borrowOriginalNativeContext();
        EXPECT_TRUE(retained.sameOriginal(original->borrowOriginalNativeContext()));
        EXPECT_FALSE(retained.sameOriginal(other->borrowOriginalNativeContext()));
        {const auto use=retained.acquire();EXPECT_EQ(&use.context(),original.get());}
    }
    EXPECT_THROW(retained.acquire(),std::logic_error);
}

/** @test A later participant's pre-collective failure aborts before collection blocks on the first future. */
TEST(MoEProjectionArenaGeometry, NativeCountedCompletionPreparationFailureDrainsOriginalParticipants)
{
    struct Context final:test::MockLocalTPContext {
        std::mutex mutex;
        std::condition_variable wake;
        bool aborted=false;
        bool signalNativeCompletionAbort()override {
            {std::lock_guard lock(mutex);aborted=true;}
            wake.notify_all();return true;
        }
    } context;
    std::promise<void> entered;
    auto first=std::async(std::launch::async,[&]{context.prepareOriginalNativeParticipant([&]{
        std::unique_lock lock(context.mutex);entered.set_value();
        context.wake.wait(lock,[&]{return context.aborted;});
    });});
    entered.get_future().get();
    auto later=std::async(std::launch::async,[&]{context.prepareOriginalNativeParticipant([]{
        throw std::runtime_error("recorded pre-collective participant failure");
    });});
    // Exactly the former adversarial collection order. The original shared
    // failure boundary must wake first before later.get() can report anything.
    EXPECT_NO_THROW(first.get());
    EXPECT_THROW(later.get(),std::runtime_error);
    EXPECT_TRUE(context.aborted);
}

/** @test Neither rejected nor throwing abort publication may return while another original participant can remain blocked. */
TEST(MoEProjectionArenaGeometry, NativeCountedCompletionAbortPublicationFailureIsFatal)
{
    struct Rejecting final:test::MockLocalTPContext {
        bool signalNativeCompletionAbort()override{return false;}
    } rejecting;
    struct Throwing final:test::MockLocalTPContext {
        bool signalNativeCompletionAbort()override{throw std::runtime_error("recorded abort publication failure");}
    } throwing;
    EXPECT_DEATH(rejecting.prepareOriginalNativeParticipant([]{throw std::runtime_error("original failure");}),"");
    EXPECT_DEATH(throwing.prepareOriginalNativeParticipant([]{throw std::runtime_error("original failure");}),"");
}
