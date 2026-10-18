/**
 * @file Test__MoERoutingSnapshotSelection.cpp
 * @brief Prove named router descriptors select only their actual publications.
 *
 * Captured prefill alternatives share one stable snapshot arena. M512 computes
 * probabilities on compact TP-owned rows, while smaller buckets replicate that
 * computation. An IDs/weights-only filter must not retain the unrelated score
 * buffer and turn a smaller bucket into a larger physical snapshot allocation.
 * These device-free checks exercise the production discovery/publication API;
 * backend capture tests separately prove actual retained arena and payloads.
 */
#include "snapshots/SnapshotCapture.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

using namespace llaminar2;
namespace
{
    /**
     * @brief Build one concrete named FP32 descriptor for discovery/publication.
     * @param name Actual producer output name; null represents an invalid unnamed output.
     * @param values Immutable host-observed payload owned by the calling fixture.
     * @param rows Actual physical row count, independent of any logical TP assembly.
     * @param columns Actual physical feature count.
     * @return A singleton dump whose metadata and payload remain caller-owned.
     */
    StageDumpInfo output(const char *name, const std::vector<float> &values,
                         size_t rows = 1, size_t columns = 2)
    {
        StageDumpInfo dump;
        dump.addOutput(name, values.data(), rows, columns);
        return dump;
    }

    /**
     * @brief Apply the exact semantic predicate used by graph descriptor filtering.
     * @param stage Concrete producer, including its transaction context when present.
     * @param dump Actual singleton output view.
     * @param requested Exact requested semantic key.
     * @return Whether this descriptor really owns the requested publication.
     */
    bool selects(const std::string &stage, const StageDumpInfo &dump,
                 const std::string &requested)
    {
        const auto keys = SnapshotCapture::possibleKeysForStage(stage, dump);
        return std::find(keys.begin(), keys.end(), requested) != keys.end();
    }
}

/** @test Every active named alias advertises and captures only its own value. */
TEST(MoERoutingSnapshotSelection, SingletonAliasesPreserveSemanticsAndContexts)
{
    const std::vector<float> values{-0.0f, 2.0f};
    const std::array<std::pair<const char *, const char *>, 8> aliases{{
        {"router_logits", "MOE_ROUTER_OUTPUT"}, {"logits", "MOE_ROUTER_OUTPUT"},
        {"routing_indices", "MOE_ROUTING_INDICES"}, {"indices", "MOE_ROUTING_INDICES"},
        {"output_indices_tensor", "MOE_ROUTING_INDICES"},
        {"routing_weights", "MOE_ROUTING_WEIGHTS"}, {"weights", "MOE_ROUTING_WEIGHTS"},
        {"output_weights_tensor", "MOE_ROUTING_WEIGHTS"}}};
    for (const auto &[stage, prefix] : std::array<std::pair<std::string, std::string>, 4>{{
             {"layer7_moe_routing", "layer7_"},
             {"layer7_moe_routing_publish", "layer7_"},
             {"MTP0_moe_routing", "MTP0_"},
             {"prefill_chunk_2::layer7_moe_routing", "PREFILL_CHUNK_2_layer7_"}}})
    for (const auto &[name, suffix] : aliases)
    {
        SCOPED_TRACE(stage + ":" + name);
        const auto dump = output(name, values);
        const std::string key = prefix + suffix;
        EXPECT_EQ(SnapshotCapture::possibleKeysForStage(stage, dump),
                  (std::vector<std::string>{key}));
        SnapshotCapture capture;
        capture.captureStage(stage, dump);
        const auto actual = capture.getShared(key);
        ASSERT_TRUE(actual);
        EXPECT_EQ(capture.keys().size(), 1u);
        EXPECT_EQ(std::memcmp(actual->data.data(), values.data(), values.size() * sizeof(float)), 0);
    }
}

/** @test Reordering cannot create positional semantics; unknown outputs remain absent. */
TEST(MoERoutingSnapshotSelection, ReorderedNamedOutputsAndUnknownScratchStayIndependent)
{
    const std::vector<float> scores{0.25f, 0.75f}, ids{1.0f, 0.0f}, weights{0.75f, 0.25f};
    auto dump = output("output_weights_tensor", weights);
    dump.outputs.push_back(output("output_indices_tensor", ids).outputs.front());
    dump.outputs.push_back(output("router_logits", scores).outputs.front());
    const std::string stage = "layer3_moe_routing";
    EXPECT_EQ(SnapshotCapture::possibleKeysForStage(stage, dump),
              (std::vector<std::string>{"layer3_MOE_ROUTING_WEIGHTS",
                  "layer3_MOE_ROUTING_INDICES", "layer3_MOE_ROUTER_OUTPUT"}));
    SnapshotCapture capture;
    capture.captureStage(stage, dump);
    ASSERT_TRUE(capture.getShared("layer3_MOE_ROUTER_OUTPUT"));
    EXPECT_EQ(capture.getShared("layer3_MOE_ROUTER_OUTPUT")->data, scores);
    EXPECT_EQ(capture.getShared("layer3_MOE_ROUTING_INDICES")->data, ids);
    EXPECT_EQ(capture.getShared("layer3_MOE_ROUTING_WEIGHTS")->data, weights);

    StageDumpInfo unknown;
    for (const char *name : {"scratch", "temporary", "unrelated", "", static_cast<const char *>(nullptr)})
    {
        const auto singleton = output(name, scores);
        EXPECT_TRUE(SnapshotCapture::possibleKeysForStage(stage, singleton).empty());
        unknown.outputs.push_back(singleton.outputs.front());
    }
    EXPECT_TRUE(SnapshotCapture::possibleKeysForStage(stage, unknown).empty());
    SnapshotCapture absent;
    absent.captureStage(stage, unknown);
    EXPECT_TRUE(absent.keys().empty()) << "Unknown triples must not revive positional router inference";
}

/** @test Exact named publication preserves already-observed explicit row ownership. */
TEST(MoERoutingSnapshotSelection, SelectedPublicationPreservesTypedRowsAndWords)
{
    const std::vector<float> values{-0.0f, 2.0f, 3.0f, 4.0f};
    auto dump = output("output_weights_tensor", values, 2, 2);
    const SnapshotOwnedRows owned{{{1, 1}}};
    dump.outputs.front().row_layout = owned;
    SnapshotCapture capture;
    capture.captureStage("mtp_condition::layer7_moe_routing_publish", dump);
    const auto actual = capture.getShared("MTP_CONDITION_layer7_MOE_ROUTING_WEIGHTS");
    ASSERT_TRUE(actual);
    EXPECT_EQ(actual->publication, SnapshotPublication::RowPartition);
    EXPECT_EQ(actual->row_ownership, owned);
    EXPECT_EQ(actual->rows, 2u);
    EXPECT_EQ(actual->cols, 2u);
    EXPECT_EQ(std::memcmp(actual->data.data(), values.data(), values.size() * sizeof(float)), 0);
    EXPECT_FALSE(capture.getShared("MTP_CONDITION_layer7_MOE_ROUTER_OUTPUT"));
}

/** @test All M1..512 route-only descriptors exclude nonmonotonic probability storage. */
TEST(MoERoutingSnapshotSelection, EveryPrefillRowCountRetainsOnlyRequestedRouteBytes)
{
    const std::vector<float> score_storage(512 * 256), route_storage(512 * 8);
    for (size_t rows = 1; rows <= 512; ++rows)
    {
        const auto scores = output("router_logits", score_storage, rows == 512 ? 256 : rows, 256);
        const auto ids = output("output_indices_tensor", route_storage, rows, 8);
        const auto weights = output("output_weights_tensor", route_storage, rows, 8);
        for (int layer = 0; layer < 40; ++layer)
        {
            const std::string prefix = "layer" + std::to_string(layer);
            const auto stage = prefix + "_moe_routing";
            size_t selected_bytes = 0;
            for (const auto &dump : {scores, ids, weights})
                if (selects(stage, dump, prefix + "_MOE_ROUTING_INDICES") ||
                    selects(stage, dump, prefix + "_MOE_ROUTING_WEIGHTS"))
                    selected_bytes += dump.outputs.front().rows * dump.outputs.front().cols * sizeof(float);
            ASSERT_EQ(selected_bytes, rows * 16 * sizeof(float)) << stage << " M=" << rows;
            EXPECT_TRUE(selects(stage, scores, prefix + "_MOE_ROUTER_OUTPUT"));
            EXPECT_FALSE(selects(stage, ids, prefix + "_MOE_ROUTER_OUTPUT"));
            EXPECT_FALSE(selects(stage, weights, prefix + "_MOE_ROUTER_OUTPUT"));
        }
    }
}
