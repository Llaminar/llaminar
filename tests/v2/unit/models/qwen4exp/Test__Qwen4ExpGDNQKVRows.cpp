/**
 * @file Test__Qwen4ExpGDNQKVRows.cpp
 * @brief Real original prepared-QKV and complete ForwardInput row binding contracts.
 *
 * The fixture calls the protected production graph resolver under its original
 * construction scopes. It prepares synthetic native source bytes on CPU but
 * executes no projection, model, graph or accelerator. Full bundle arithmetic
 * and selected Stage2 execution belong to the separate integration fixture.
 */
#include "Qwen4ExpMetadataFixture.h"
#include "../../../utils/PreparedWeightTestHarness.h"
#include "models/qwen35/Qwen35Graph.h"
#include "models/qwen4exp/Qwen4ExpGraphConfigBuilder.h"
#include "tensors/Tensors.h"
#include <gtest/gtest.h>

namespace
{
using namespace llaminar2;
constexpr auto single = QuantizedActivationOperandPrecision::SingleQ8;
constexpr auto two = QuantizedActivationOperandPrecision::TwoTermQ8;
/** @brief Expose the existing resolver without constructing replacement query authority. */
class QueryGraph final : public Qwen35Graph
{
public:
    /** @brief Retain the same immutable setup policy as the production graph. */
    explicit QueryGraph(const GraphConfig &config) : Qwen35Graph(config, nullptr) {}
    /**
     * @brief Resolve one original source and invocation using the production scopes.
     * @param input Original complete forward invocation whose count is borrowed.
     * @param tensor Actual original native source owner.
     * @param binding Exact immutable store binding, or a deliberately malformed copy.
     * @param layer Original global source layer, never a PP-local ordinal.
     * @return The actual production descriptor or the unchanged SingleQ8 null arm.
     */
    std::optional<DeviceRowRange> rows(const ForwardInput &input, TensorBase *tensor,
        const WeightBinding *binding, int layer = 0)
    {
        ForwardInputConstructionScope invocation(*this, input);
        PrefillCollectiveRowsScope prefill(*this, input);
        return gdnQKVProjectionRows(tensor, binding, layer, input.seq_len,
            input.batch_size, input.device, input.sequence_lengths_device);
    }
};
/** @brief One real original full native matrix, prepared once per test before row resolution. */
struct Source final
{
    /** @brief Build the genuine source/store binding without source conversion or backend startup. */
    Source() : tensor(std::vector<size_t>{10240, 2560},
        std::vector<uint8_t>(size_t{10240} * 2560 / 32 * sizeof(Q8_0Block))),
        prepared(test::makePreparedGemmFixture(&tensor, DeviceId::cpu(), "blk.0.attn_qkv.weight"))
    {
        Qwen4ExpGraphConfigBuilder::populateFromDirectory(test::qwen4MetadataFixture(false), config);
        config.default_device = DeviceId::cpu();
        config.qfn_gdn_qkv_activation = two;
        prepared.binding.slice = {.source_rows=10240, .source_cols=2560, .row_count=10240, .col_count=2560};
    }
    Q8_0Tensor tensor; ///< Original source outlives its immutable prepared store.
    test::PreparedGemmFixture prepared;
    GraphConfig config;
};
/** @return An actual compact CPU main query with no invented device owner. */
ForwardInput query(int rows)
{
    ForwardInput input;
    input.device = DeviceId::cpu(); input.seq_len = rows; input.real_seq_len = rows;
    return input;
}
}

/** @test Original count pointer and complete physical shape survive CPU prefill and scalar decode. */
TEST(Qwen4ExpGDNQKVRows, PreparedOriginalSourceRetainsCanonicalCompactQuery)
{
    Source source;
    QueryGraph graph(source.config);
    graph.setPreparedWeightStore(source.prepared.store.get());
    for (const int count : {1, 9, 15, 512})
    for (const bool counted : {false, true})
    {
        std::vector<int> lengths{count};
        auto input = query(count); input.sequence_lengths = counted ? &lengths : nullptr;
        const auto rows = graph.rows(input, &source.tensor, &source.prepared.binding);
        ASSERT_TRUE(rows);
        EXPECT_EQ(rows->physicalRows(), count);
        EXPECT_EQ(rows->countOwner(), counted ? lengths.data() : nullptr);
        const auto identity = rows->captureIdentity();
        if (counted)
        {
            // The original immutable count address, not its current contents,
            // belongs to recording identity. Construction still required compactness.
            lengths[0] = 0;
            EXPECT_EQ(rows->activeRowsFor(0), 0);
            EXPECT_EQ(rows->captureIdentity(), identity);
        }
    }
    auto decode = query(1); decode.execution_phase = ForwardExecutionPhase::Decode;
    EXPECT_TRUE(graph.rows(decode, &source.tensor, &source.prepared.binding));
    auto ordinary = source.config; ordinary.qfn_gdn_qkv_activation = single;
    QueryGraph unchanged(ordinary);
    EXPECT_FALSE(unchanged.rows(query(9), nullptr, nullptr));
}

/** @test Malformed source/store identities cannot acquire a descriptor even with correct dimensions. */
TEST(Qwen4ExpGDNQKVRows, RejectsForeignPartialAndUnpreparedOriginalSources)
{
    Source source;
    QueryGraph graph(source.config);
    graph.setPreparedWeightStore(source.prepared.store.get());
    for (int mutation = 0; mutation != 10; ++mutation)
    {
        auto binding = source.prepared.binding;
        if (mutation == 0) binding.immutable = false;
        if (mutation == 1) binding.identity.role = WeightRole::Other;
        if (mutation == 2) binding.identity.canonical_name = "blk.3.attn_qkv.weight";
        if (mutation == 3) binding.identity.layer = 1;
        if (mutation == 4) binding.slice.col_count /= 2;
        if (mutation == 5) ++binding.binding_id;
        if (mutation == 6) binding.tensor = nullptr;
        if (mutation == 7) binding.floating_operand_precision = FloatingActivationOperandPrecision::FP16RNE;
        if (mutation == 8) graph.setPreparedWeightStore(nullptr);
        auto input = query(9);
        if (mutation == 9) input.device = DeviceId::cuda(0); // No GPU is initialized: absent prepared source rejects first.
        EXPECT_THROW(graph.rows(input, &source.tensor, &binding), std::exception);
        graph.setPreparedWeightStore(source.prepared.store.get());
    }
    EXPECT_THROW(graph.rows(query(9), &source.tensor, &source.prepared.binding, 1), std::invalid_argument);
}

/** @test Original CPU request authority must prove compactness, role, phase and exact owner before graph insertion. */
TEST(Qwen4ExpGDNQKVRows, RejectsPaddedForeignAndUnknownQueries)
{
    Source source;
    QueryGraph graph(source.config);
    graph.setPreparedWeightStore(source.prepared.store.get());
    for (const auto lengths : {std::vector<int>{}, std::vector<int>{0}, std::vector<int>{8},
        std::vector<int>{10}, std::vector<int>{-1}, std::vector<int>{9,9}})
    {
        auto input = query(9); input.sequence_lengths = &lengths;
        EXPECT_THROW(graph.rows(input, &source.tensor, &source.prepared.binding), std::invalid_argument);
    }
    for (int mutation = 0; mutation != 8; ++mutation)
    {
        auto input = query(9); const int32_t foreign = 9;
        if (mutation == 0) input.real_seq_len = 8;
        if (mutation == 1) input.bucket_seq_len = 16;
        if (mutation == 2) input.batch_size = 2;
        if (mutation == 3) input.sequence_lengths_device = &foreign;
        if (mutation == 4) input.execution_phase = static_cast<ForwardExecutionPhase>(255);
        if (mutation == 5) input.execution_role = static_cast<ForwardExecutionRole>(255);
        if (mutation == 6) input.graph_submission_intent = static_cast<ForwardGraphSubmissionIntent>(255);
        if (mutation == 7) input.seq_len = 0;
        EXPECT_THROW(graph.rows(input, &source.tensor, &source.prepared.binding), std::exception);
    }
    const std::vector<int> not_admitted{0};
    auto declaration = query(9); declaration.sequence_lengths = &not_admitted;
    declaration.graph_submission_intent = ForwardGraphSubmissionIntent::WorkspaceDeclaration;
    const auto rows = graph.rows(declaration, &source.tensor, &source.prepared.binding);
    ASSERT_TRUE(rows);
    EXPECT_EQ(rows->countOwner(), not_admitted.data());
    // This resolver declares geometry only. Existing construction intent owns
    // the separate graph/executor rejection; no declaration is executed here.
    declaration.graph_submission_intent = ForwardGraphSubmissionIntent::Execute;
    EXPECT_THROW(graph.rows(declaration, &source.tensor, &source.prepared.binding), std::invalid_argument);
}
