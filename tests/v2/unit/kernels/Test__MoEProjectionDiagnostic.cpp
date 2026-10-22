/**
 * @file Test__MoEProjectionDiagnostic.cpp
 * @brief Device-free admission tests for the canonical trainer's M1 branch.
 *
 * Tiny authored descriptor metadata proves original versus physical geometry,
 * strict integer parsing and source-family preservation. No model path is
 * opened; no native runtime or prepared tensor is constructed.
 */
#include <gtest/gtest.h>
#include "../../performance/kernels/native_vnni_dispatch/NativeVNNIMoEProjectionDiagnostic.h"

namespace
{
using Json = nlohmann::json;
using namespace llaminar2::test::native_vnni_dispatch;
/** @return The explicit diagnostic request shape used by the existing trainer. */
Json request()
{
    return {{"schema", "native-vnni-moe-projection-diagnostic-v1"}, {"operation", "owner-gate-up-pack-m1"},
        {"owner_map", "ordinal"}, {"provenance", "retained-descriptors-synthetic-seeded"}, {"m", 1},
        {"layer", 0}, {"participant", 2}, {"top_k", 10}, {"device_ordinals", {0, 1, 2, 3}},
        {"route_profile", "uniform"}, {"warmups", 31}, {"samples", 101}, {"operations_per_sample", 1},
        {"descriptor_document", "/retained/metadata-only.json"}};
}
/** @return Metadata only; its source path deliberately cannot be opened. */
Json descriptors()
{
    return {{"files", {{{"path", "/nonexistent/model.gguf"}, {"stat", {1, 2, 3, 4, 5}},
        {"header_data_offset", 64}, {"tensors", {
            {{"name", "blk.0.ffn_gate_exps.weight"}, {"dimensions", {2560, 640, 512}}, {"format_label", "Q4_K"}},
            {{"name", "blk.0.ffn_up_exps.weight"}, {"dimensions", {2560, 640, 512}}, {"format_label", "Q4_K"}},
            {{"name", "blk.0.ffn_down_exps.weight"}, {"dimensions", {640, 2560, 512}}, {"format_label", "Q5_1"}}
        }}}}}};
}
}

/** The declaration carries original-H policy and exact physical interval separately. */
TEST(MoEProjectionDiagnostic, PreservesOriginalAndPhysicalOperationIdentity)
{
    const auto spec = parseMoEProjectionDiagnostic(request(), descriptors());
    const auto down = spec.ownership().projection(llaminar2::WeightRole::MoEExpertDown);
    EXPECT_EQ(spec.source.hidden_size, 2560);
    EXPECT_EQ(spec.source.expert_count, 512);
    EXPECT_EQ(spec.source.routed.gate, "Q4_K");
    EXPECT_EQ(spec.source.routed.down, "Q5_1");
    EXPECT_EQ(down.source_rows, 2560);
    EXPECT_EQ(down.first_row, 1280);
    EXPECT_EQ(down.rows, 640);
    EXPECT_TRUE(spec.source.mixture_evidence_ids.empty());
    EXPECT_NE(spec.evidenceId().find("descriptor_layer0"), std::string::npos);
}

/** Fractional, boolean and out-of-range values never become another admitted integer. */
TEST(MoEProjectionDiagnostic, RejectsLossyIntegerDeclarations)
{
    for (const auto *field : {"m", "layer", "participant", "top_k", "warmups", "samples", "operations_per_sample"})
        for (const Json invalid : {Json(1.9), Json(true), Json(std::uint64_t{1} << 63)})
        {
            auto bad = request(); bad[field] = invalid;
            EXPECT_THROW((void)parseMoEProjectionDiagnostic(bad, descriptors()), std::invalid_argument) << field;
        }
    auto bad = request(); bad["device_ordinals"][0] = 0.5;
    EXPECT_THROW((void)parseMoEProjectionDiagnostic(bad, descriptors()), std::invalid_argument);
    auto metadata = descriptors(); metadata["files"][0]["tensors"][0]["dimensions"][0] = 2560.5;
    EXPECT_THROW((void)parseMoEProjectionDiagnostic(request(), metadata), std::invalid_argument);
}

/** Unsupported operations and inherited legacy pair controls fail before GPU setup. */
TEST(MoEProjectionDiagnostic, RejectsUnknownOperationAndLegacyControls)
{
    for (const auto *field : {"schema", "operation", "owner_map", "provenance"})
    {
        auto bad = request(); bad[field] = "unknown";
        EXPECT_THROW((void)parseMoEProjectionDiagnostic(bad, descriptors()), std::invalid_argument);
    }
    auto bad = request(); bad["minimum_finalists"] = 12;
    EXPECT_THROW((void)parseMoEProjectionDiagnostic(bad, descriptors()), std::invalid_argument);
}

/** Neither topology errors nor partial descriptor geometry can be repaired by aliasing. */
TEST(MoEProjectionDiagnostic, RejectsInconsistentOwnershipOrSourceGeometry)
{
    auto bad = request(); bad["participant"] = 4;
    EXPECT_THROW((void)parseMoEProjectionDiagnostic(bad, descriptors()), std::invalid_argument);
    bad = request(); bad["device_ordinals"] = {0, 0, 2, 3};
    EXPECT_THROW((void)parseMoEProjectionDiagnostic(bad, descriptors()), std::invalid_argument);
    auto metadata = descriptors(); metadata["files"][0]["tensors"][2]["dimensions"][1] = 640;
    EXPECT_THROW((void)parseMoEProjectionDiagnostic(request(), metadata), std::invalid_argument);
    metadata = descriptors(); metadata["files"][0]["tensors"].erase(1);
    EXPECT_THROW((void)parseMoEProjectionDiagnostic(request(), metadata), std::invalid_argument);
}
