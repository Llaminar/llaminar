/**
 * @file Test__MoEProjectionDiagnostic.cpp
 * @brief Device-free admission tests for the canonical trainer's M1 branch.
 *
 * Tiny authored descriptor metadata proves original versus physical geometry,
 * strict integer parsing and source-family preservation. No model path is
 * opened; no native runtime or prepared tensor is constructed. Partition execution
 * contests preserve ordinary requests and the shared chronological order.
 * Explicit synthetic source fixtures preserve every canonical original alias
 * without borrowing model, descriptor-file or prepared-payload identity.
 * Promoted representation admission preserves the original source family and
 * accepts only its canonical expanded-asymmetric CPU migration contract.
 */
#include <gtest/gtest.h>
#include "../../performance/kernels/native_vnni_dispatch/NativeVNNIMoEProjectionDiagnostic.h"
#include "../../performance/kernels/native_vnni_dispatch/NativeVNNIGeometryTimingOrder.h"

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

/** Partition execution is an authored diagnostic choice; omitted requests preserve the original contest. */
TEST(MoEProjectionDiagnostic, PartitionContestPreservesExplicitIdentityAndDefault)
{
    const auto original = parseMoEProjectionDiagnostic(request(), descriptors());
    EXPECT_EQ(original.contest, MoEProjectionContest::GateUpGeometry);
    auto explicit_geometry = request(); explicit_geometry["contest"] = "gate-up-geometry";
    EXPECT_EQ(parseMoEProjectionDiagnostic(explicit_geometry, descriptors()).evidenceId(), original.evidenceId());
    auto worker_request = request(); worker_request["contest"] = "inline64-vs-parallel-partitions64";
    const auto pair = parseMoEProjectionDiagnostic(worker_request, descriptors());
    EXPECT_EQ(pair.contest, MoEProjectionContest::Inline64VsParallelPartitions64);
    EXPECT_STREQ(moeProjectionContestName(pair.contest), "inline64-vs-parallel-partitions64");
    EXPECT_EQ(pair.source.hidden_size, original.source.hidden_size);
    EXPECT_EQ(pair.source.hidden_size, 2560);
    EXPECT_EQ(pair.source.routed_expert_width, 640);
    EXPECT_EQ(pair.source.routed.gate, original.source.routed.gate);
    EXPECT_EQ(pair.ownership().projection(llaminar2::WeightRole::MoEExpertDown).first_row,
        original.ownership().projection(llaminar2::WeightRole::MoEExpertDown).first_row);
    EXPECT_NE(pair.evidenceId(), original.evidenceId());
    EXPECT_NE(pair.evidenceId().find("inline64-vs-parallel-partitions64"), std::string::npos);
}

/** Unknown/implicit values cannot select a different graph or reuse old evidence identity. */
TEST(MoEProjectionDiagnostic, RejectsUnknownPartitionContest)
{
    for (const Json invalid : {Json("inline64-vs-staged64"), Json("four-worker-planes"), Json(""),
             Json(true), Json(1), Json(nullptr), Json::object()})
    {
        auto bad = request(); bad["contest"] = invalid;
        EXPECT_THROW((void)parseMoEProjectionDiagnostic(bad, descriptors()), std::invalid_argument);
    }
    EXPECT_THROW((void)moeProjectionContestName(static_cast<MoEProjectionContest>(255)), std::invalid_argument);
}

/** Projecting the canonical three-way inventory preserves a balanced real two-way contest. */
TEST(MoEProjectionDiagnostic, PartitionContestUsesBalancedTwoWayProjection)
{
    std::array<std::array<int, 2>, 2> positions{}, before{};
    for (std::size_t round = 0; round < kMoERouteGeometryTimingOrders.size(); ++round)
    {
        int position = 0; std::array<std::size_t, 2> actual{};
        for (auto candidate : moeRouteGeometryTimingOrder(round))
        {
            if (candidate >= 2) continue;
            ASSERT_LT(position, 2);
            ++positions[candidate][position]; actual[position++] = candidate;
        }
        ASSERT_EQ(position, 2);
        ASSERT_NE(actual[0], actual[1]); ++before[actual[0]][actual[1]];
    }
    for (std::size_t candidate = 0; candidate < 2; ++candidate)
        for (std::size_t position = 0; position < 2; ++position)
            EXPECT_EQ(positions[candidate][position], 3);
    EXPECT_EQ(before[0][1], 3); EXPECT_EQ(before[1][0], 3);
}


/** True automatic production capture has its own explicit identity, never a Serial/K1 alias. */
TEST(MoEProjectionDiagnostic, AutomaticContestPreservesOldContestsAndSourceIdentity)
{
    auto authored = request(); authored["contest"] = "automatic-vs-parallel-partitions64";
    const auto automatic = parseMoEProjectionDiagnostic(authored, descriptors());
    EXPECT_EQ(automatic.contest, MoEProjectionContest::AutomaticVsParallelPartitions64);
    EXPECT_STREQ(moeProjectionContestName(automatic.contest), "automatic-vs-parallel-partitions64");
    auto inline_request = request(); inline_request["contest"] = "inline64-vs-parallel-partitions64";
    const auto inline_pair = parseMoEProjectionDiagnostic(inline_request, descriptors());
    const auto original = parseMoEProjectionDiagnostic(request(), descriptors());
    EXPECT_NE(automatic.evidenceId(), inline_pair.evidenceId());
    EXPECT_NE(automatic.evidenceId(), original.evidenceId());
    EXPECT_EQ(automatic.source.evidenceId(), original.source.evidenceId());
    EXPECT_EQ(original.contest, MoEProjectionContest::GateUpGeometry);
    EXPECT_EQ(inline_pair.contest, MoEProjectionContest::Inline64VsParallelPartitions64);
}

namespace
{
/** @return Honest small synthetic inventory; no descriptor path, model layer or fake file metadata. */
Json syntheticRequest(const std::string &format)
{
    auto result = request(); result.erase("layer"); result.erase("descriptor_document");
    result["participant"] = 0; result["contest"] = "automatic-vs-parallel-partitions64";
    result["provenance"] = "synthetic-source-fixture"; result["operations_per_sample"] = 32;
    result["source_fixture"] = {{"hidden_size", 2560}, {"expert_width", 640}, {"expert_count", 32},
        {"formats", {{"gate", format}, {"up", format}, {"down", "Q8_0"}}}};
    return result;
}
}

/** Every original source alias retains canonical provenance without creating model descriptor evidence. */
TEST(MoEProjectionDiagnostic, SyntheticFixturesPreserveEveryCanonicalSourceAlias)
{
    for (const auto &format : llaminar2::native_vnni_formats::kAllSourceFormats)
    {
        const auto spec = parseMoEProjectionDiagnostic(syntheticRequest(std::string(format.quant_type)), Json::object());
        EXPECT_EQ(spec.source_kind, MoEProjectionSourceKind::SyntheticSourceFixture);
        EXPECT_EQ(spec.source.routed.gate, format.quant_type);
        EXPECT_EQ(spec.source.expert_count, 32); EXPECT_EQ(spec.top_k, 10);
        EXPECT_EQ(spec.layer, -1); EXPECT_TRUE(spec.descriptor_document.empty());
        EXPECT_TRUE(spec.descriptor_evidence.at("model_identity").is_null());
        const auto &gate = spec.descriptor_evidence.at("original_source_roles").at("gate");
        EXPECT_EQ(gate.at("source_codebook").get<int>(), format.metadata->codebook_id);
        EXPECT_EQ(gate.at("source_is_superblock").get<bool>(), format.metadata->is_superblock);
        EXPECT_NE(spec.evidenceId().find("synthetic_source_fixture"), std::string::npos);
        EXPECT_EQ(spec.evidenceId().find("descriptor_layer"), std::string::npos);
        EXPECT_TRUE(spec.source.mixture_evidence_ids.empty());
    }
}

/** Synthetic provenance rejects fabricated descriptor authority and malformed/partial source declarations. */
TEST(MoEProjectionDiagnostic, SyntheticFixturesRejectMixedAuthorityAndLossyBounds)
{
    for (const auto *field : {"layer", "descriptor_document"})
    {
        auto bad = syntheticRequest("Q6_K"); bad[field] = request().at(field);
        EXPECT_THROW((void)parseMoEProjectionDiagnostic(bad, descriptors()), std::invalid_argument);
    }
    auto mixed = request(); mixed["source_fixture"] = syntheticRequest("Q6_K").at("source_fixture");
    EXPECT_THROW((void)parseMoEProjectionDiagnostic(mixed, descriptors()), std::invalid_argument);
    for (const auto *field : {"hidden_size", "expert_width", "expert_count"})
        for (const Json invalid : {Json(2560.5), Json(true), Json(std::uint64_t{1} << 63), Json(-1)})
        {
            auto bad = syntheticRequest("Q6_K"); bad["source_fixture"][field] = invalid;
            EXPECT_THROW((void)parseMoEProjectionDiagnostic(bad, Json::object()), std::invalid_argument);
        }
    auto bad = syntheticRequest("Q6_K"); bad["source_fixture"]["hidden_size"] = 2592;
    EXPECT_THROW((void)parseMoEProjectionDiagnostic(bad, Json::object()), std::invalid_argument);
    bad = syntheticRequest("Q6_K"); bad["source_fixture"]["original_gguf"] = "invented.gguf";
    EXPECT_THROW((void)parseMoEProjectionDiagnostic(bad, Json::object()), std::invalid_argument);
    bad = syntheticRequest("Q6_K"); bad["source_fixture"]["formats"]["up"] = "Q8_0";
    EXPECT_THROW((void)parseMoEProjectionDiagnostic(bad, Json::object()), std::invalid_argument);
    bad = syntheticRequest("not-a-native-format");
    EXPECT_THROW((void)parseMoEProjectionDiagnostic(bad, Json::object()), std::invalid_argument);
}

/** Automatic timing remains physicalM1 and the existing public serial oracle's topK<=16 contract. */
TEST(MoEProjectionDiagnostic, AutomaticContestRejectsMultirowAndOversizedRouteCapacity)
{
    auto valid = syntheticRequest("IQ2_S"); valid["top_k"] = 16;
    EXPECT_EQ(parseMoEProjectionDiagnostic(valid, Json::object()).top_k, 16);
    for (const int rows : {0, 2, 18})
    {
        auto bad = valid; bad["m"] = rows;
        EXPECT_THROW((void)parseMoEProjectionDiagnostic(bad, Json::object()), std::invalid_argument);
    }
    auto bad = valid; bad["top_k"] = 17;
    EXPECT_THROW((void)parseMoEProjectionDiagnostic(bad, Json::object()), std::invalid_argument);
}


/** Original requests retain their exact identity; genuine promoted preparation is explicitly distinct. */
TEST(MoEProjectionDiagnostic, Promoted23RepresentationPreservesDefaultAndOriginalSource)
{
    auto authored = syntheticRequest("Q5_K");
    const auto original = parseMoEProjectionDiagnostic(authored, Json::object());
    EXPECT_EQ(original.prepared_representation, MoEProjectionPreparedRepresentation::OriginalGpu);
    authored["prepared_representation"] = "original-gpu-prepared";
    EXPECT_EQ(parseMoEProjectionDiagnostic(authored, Json::object()).evidenceId(), original.evidenceId());
    authored["prepared_representation"] = "cpu-prepared-promoted23";
    const auto promoted = parseMoEProjectionDiagnostic(authored, Json::object());
    EXPECT_EQ(promoted.prepared_representation, MoEProjectionPreparedRepresentation::CpuPreparedPromoted23);
    EXPECT_EQ(promoted.source.evidenceId(), original.source.evidenceId());
    EXPECT_EQ(promoted.source.routed.gate, "Q5_K");
    EXPECT_NE(promoted.evidenceId(), original.evidenceId());
    EXPECT_NE(promoted.evidenceId().find("cpu-prepared-promoted23"), std::string::npos);
    const auto &gate = promoted.descriptor_evidence.at("original_source_roles").at("gate");
    EXPECT_EQ(gate.at("source_codebook").get<int>(), 7);
    EXPECT_TRUE(gate.at("source_is_superblock").get<bool>());
    EXPECT_EQ(gate.at("canonical_execution_codebook").get<int>(), 7);
    EXPECT_EQ(gate.at("migration_execution_codebook").get<int>(), 23);
}

/** Supported original asymmetric sources derive from the canonical CPU migration contract, not a fake23 source. */
TEST(MoEProjectionDiagnostic, Promoted23SourcesUseCanonicalMigrationContract)
{
    std::set<std::string> accepted;
    for (const auto &source : llaminar2::native_vnni_formats::kAllSourceFormats)
    {
        const bool expected = source.metadata->is_asymmetric &&
            llaminar2::migrationStableDeviceVnniFormat(*source.metadata).codebook_id ==
                llaminar2::kNativeVnniExpandedInt8MinCodebook;
        EXPECT_EQ(moeProjectionSupportsPromoted23(source.quant_type), expected);
        auto authored = syntheticRequest(std::string(source.quant_type));
        authored["prepared_representation"] = "cpu-prepared-promoted23";
        if (expected)
        {
            accepted.emplace(source.quant_type);
            const auto spec = parseMoEProjectionDiagnostic(authored, Json::object());
            EXPECT_EQ(spec.source.routed.gate, source.quant_type);
            EXPECT_EQ(spec.descriptor_evidence.at("original_source_roles").at("gate").at("source_codebook").get<int>(),
                source.metadata->codebook_id);
        }
        else EXPECT_THROW((void)parseMoEProjectionDiagnostic(authored, Json::object()), std::invalid_argument);
    }
    EXPECT_EQ(accepted, (std::set<std::string>{"Q5_1", "Q5_K", "IQ1_S"}));
    EXPECT_FALSE(moeProjectionSupportsPromoted23("23"));
    EXPECT_FALSE(moeProjectionSupportsPromoted23("Q4_K"));
}

/** Unknown or mixed preparation authority fails before synthetic weights, allocation or transfer setup. */
TEST(MoEProjectionDiagnostic, Promoted23RejectsUnknownAndMixedPreparationAuthority)
{
    for (const Json invalid : {Json("23"), Json("cpu"), Json(""), Json(true), Json(23), Json(nullptr)})
    {
        auto bad = syntheticRequest("Q5_K"); bad["prepared_representation"] = invalid;
        EXPECT_THROW((void)parseMoEProjectionDiagnostic(bad, Json::object()), std::invalid_argument);
    }
    auto bad = request(); bad["prepared_representation"] = "cpu-prepared-promoted23";
    EXPECT_THROW((void)parseMoEProjectionDiagnostic(bad, descriptors()), std::invalid_argument);
    for (const auto *contest : {"gate-up-geometry", "inline64-vs-parallel-partitions64"})
    {
        bad = syntheticRequest("Q5_K"); bad["contest"] = contest;
        bad["prepared_representation"] = "cpu-prepared-promoted23";
        EXPECT_THROW((void)parseMoEProjectionDiagnostic(bad, Json::object()), std::invalid_argument);
    }
    bad = syntheticRequest("23"); bad["prepared_representation"] = "cpu-prepared-promoted23";
    EXPECT_THROW((void)parseMoEProjectionDiagnostic(bad, Json::object()), std::invalid_argument);
    EXPECT_THROW((void)moeProjectionPreparedRepresentationName(static_cast<MoEProjectionPreparedRepresentation>(255)),
        std::invalid_argument);
}

/** Promotion does not widen the original M1/topK oracle or add authored staging/environment controls. */
TEST(MoEProjectionDiagnostic, Promoted23KeepsBoundedSyntheticOperation)
{
    auto authored = syntheticRequest("IQ1_S"); authored["prepared_representation"] = "cpu-prepared-promoted23";
    for (const auto &[field, invalid] : std::vector<std::pair<std::string, Json>>{{"m", 2}, {"top_k", 17},
             {"promotion_chunk_units", 64}, {"promotion_epoch", 2}})
    {
        auto bad = authored; bad[field] = invalid;
        EXPECT_THROW((void)parseMoEProjectionDiagnostic(bad, Json::object()), std::invalid_argument);
    }
    // CPU wire units span64 output columns. The new fixture does not send
    // padded columns merely because an allocation could retain them.
    auto padded = syntheticRequest("Q5_1"); padded["source_fixture"]["expert_width"] = 32;
    EXPECT_NO_THROW((void)parseMoEProjectionDiagnostic(padded, Json::object()));
    padded["prepared_representation"] = "cpu-prepared-promoted23";
    EXPECT_THROW((void)parseMoEProjectionDiagnostic(padded, Json::object()), std::invalid_argument);
    authored["source_fixture"]["expert_count"] = 4; authored["top_k"] = 4;
    const auto smoke = parseMoEProjectionDiagnostic(authored, Json::object());
    EXPECT_EQ(smoke.source.expert_count, 4); EXPECT_EQ(smoke.top_k, 4);
    EXPECT_EQ(smoke.rows, 1); EXPECT_EQ(smoke.device_ordinals.size(), 4u);
}
