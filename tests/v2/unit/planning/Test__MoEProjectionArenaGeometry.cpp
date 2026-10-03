/**
 * @file Test__MoEProjectionArenaGeometry.cpp
 * @brief Device-free proof that projection admission and graph storage are identical.
 *
 * All source codebooks and floating formats use the real schema/resolver. The
 * independent arithmetic checks every concurrent buffer, including the MTP
 * layer envelope, and ensures the old whole-expert bank is not also retained.
 */
#include "execution/moe/MoEProjectionArenaGeometry.h"
#include "execution/local_execution/graph/GraphResolver.h"
#include "models/qwen35moe/Qwen35MoESchema.h"
#include "execution/local_execution/graph/GraphBuilderRegistry.h"
#include "models/qwen35moe/Qwen35MoEGraph.h"
#include "planning/ActivationMemoryEstimator.h"
#include "collective/DeviceCountedAllGather.h"
#include "tensors/NativeVnniFormatInfo.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <string>
#include <vector>

using namespace llaminar2;

/** @test Domain physical staging is one directed edge per peer, not one set per model layer. */
TEST(MoEProjectionArenaGeometry, CountedExchangeBOMIsExactAndOverflowChecked)
{
    for (const std::size_t degree : {2u, 3u, 4u, 8u})
    for (const std::size_t capacity : {1u, 257u, 1024u * 1024u})
    {
        const auto channel = CapturedTransferChannel::memoryFor(capacity);
        const auto collective = DeviceCountedAllGather::memoryFor(degree, capacity);
        EXPECT_EQ(collective.host_bytes, (degree - 1) * channel.mapped_host_bytes);
        EXPECT_EQ(collective.device_bytes, 2 * (degree - 1) * channel.cursor_bytes_per_device + degree * sizeof(std::uint64_t));
    }
    EXPECT_THROW(DeviceCountedAllGather::memoryFor(1, 1024), std::invalid_argument);
    EXPECT_THROW(DeviceCountedAllGather::memoryFor(2, 0), std::invalid_argument);
    EXPECT_THROW(DeviceCountedAllGather::memoryFor(std::numeric_limits<std::size_t>::max(), 1024), std::overflow_error);
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
    for (const auto &format : native_vnni_formats::kAllSourceFormats)
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
